"""Read-only recording of Predict's exact Chainlink report feed.

Requires separately configured Chainlink credentials/entitlement. Captures raw
reports and v3 integer fields; does not verify OCR signatures or infer prices
from another feed. Existing Predict/PM/Binance services are not modified.
"""
from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
import hashlib
import hmac
import json
import os
from pathlib import Path
import queue
import re
import signal
import threading
import time
import uuid

import aiohttp

from data_collectors import predict_fun_recorder as storage

REST = 'https://api.dataengine.chain.link'
WS = 'wss://ws.dataengine.chain.link'
SCHEMA = 'predict_chainlink_raw_v1'


class AccessRefused(RuntimeError):
    pass


def reference_feed(path):
    ref = json.loads(path.read_text())
    category = ref['category']
    storage.category_markets(category, category['slug'])
    variants = [m['variantData'] for m in category['markets']]
    feeds = {v['priceFeedId'].lower() for v in variants}
    if len(feeds) != 1 or any(v['priceFeedProvider'] != 'CHAINLINK' or v['priceFeedSymbol'] != 'BTCUSDT' for v in variants):
        raise ValueError('REFERENCE_FEED_IDENTITY')
    feed = feeds.pop()
    if not re.fullmatch(r'0x0003[0-9a-f]{60}', feed):
        raise ValueError('UNSUPPORTED_FEED_SCHEMA')
    return feed, ref


def auth_headers(path, key, secret, timestamp_ms):
    """Official HMAC request format, including the complete query string."""
    empty_hash = hashlib.sha256(b'').hexdigest()
    message = f'GET {path} {empty_hash} {key} {timestamp_ms}'
    return {'Authorization': key, 'X-Authorization-Timestamp': str(timestamp_ms),
            'X-Authorization-Signature-SHA256': hmac.new(secret.encode(), message.encode(), hashlib.sha256).hexdigest()}


def decode_report(envelope, expected_feed):
    report = envelope['report']
    if report['feedID'].lower() != expected_feed:
        raise ValueError('ENVELOPE_FEED_MISMATCH')
    encoded = report['fullReport']
    if not isinstance(encoded, str) or len(encoded) > 2_000_002:
        raise ValueError('REPORT_HEX_SIZE')
    body = bytes.fromhex(encoded.removeprefix('0x'))
    if len(body) < 224 or len(body) % 32:
        raise ValueError('FULL_REPORT_ABI')
    word = lambda at: int.from_bytes(body[at:at + 32], 'big')
    regions = []
    for slot, is_bytes in ((96, True), (128, False), (160, False)):
        offset = word(slot)
        if offset < 224 or offset % 32 or offset + 32 > len(body):
            raise ValueError('DYNAMIC_OFFSET')
        count = word(offset)
        size = ((count + 31) // 32) * 32 if is_bytes else count * 32
        if offset + 32 + size > len(body):
            raise ValueError('TRUNCATED_DYNAMIC_DATA')
        regions.append((offset, offset + 32 + size, count))
    ordered = sorted(regions)
    if any(a[1] > b[0] for a, b in zip(ordered, ordered[1:])):
        raise ValueError('OVERLAPPING_DYNAMIC_DATA')
    if regions[1][2] != regions[2][2]:
        raise ValueError('SIGNATURE_ARRAY_LENGTHS')
    offset, _, length = regions[0]
    blob = body[offset + 32:offset + 32 + length]
    if len(blob) != 288 or blob[:32].hex() != expected_feed[2:]:
        raise ValueError('REPORT_FEED_OR_SCHEMA_MISMATCH')
    words = [blob[i:i + 32] for i in range(0, 288, 32)]
    values = {}
    for index, name, bits in ((1, 'validFromTimestamp', 32), (2, 'observationsTimestamp', 32),
                              (3, 'nativeFee', 192), (4, 'linkFee', 192), (5, 'expiresAt', 32)):
        value = int.from_bytes(words[index], 'big')
        if value >= 1 << bits:
            raise ValueError('NONCANONICAL_UNSIGNED_WORD')
        values[name] = value
    if not values['validFromTimestamp'] <= values['observationsTimestamp'] <= values['expiresAt']:
        raise ValueError('REPORT_TIME_ORDER')
    for index, name in ((6, 'benchmark_price_integer'), (7, 'bid_integer'), (8, 'ask_integer')):
        value = int.from_bytes(words[index], 'big', signed=True)
        if not -(1 << 191) <= value < 1 << 191:
            raise ValueError('NONCANONICAL_SIGNED_WORD')
        values[name] = str(value)  # Preserve large fixed-point integers exactly.
    return {'feed_id': expected_feed, **values, 'signature_verification': 'NOT_PERFORMED',
            'price_decimals': None, 'units': 'UNSCALED_REQUIRES_EXACT_FEED_METADATA',
            'full_report_sha256': hashlib.sha256(body).hexdigest()}


class Writer(storage.RawWriter):
    """Reuse the reviewed bounded, threaded storage engine with own identity."""
    def __init__(self, root, feed, reference, reference_path):
        self.run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ_') + uuid.uuid4().hex[:10]
        self.path = root / 'runs' / self.run_id
        self.path.mkdir(parents=True, exist_ok=False)
        own_source = Path(__file__).read_bytes()
        storage_source = Path(storage.__file__).read_bytes()
        (self.path / 'collector_source.py').write_bytes(own_source)
        (self.path / 'storage_source.py').write_bytes(storage_source)
        (self.path / 'reference.json').write_text(json.dumps(reference, indent=2) + '\n')
        self.metadata = {'schema': SCHEMA, 'run_id': self.run_id, 'feed_id': feed,
                         'started': storage.clock(), 'network': 'CHAINLINK_MAINNET',
                         'source_sha256': hashlib.sha256(own_source).hexdigest(),
                         'storage_source_sha256': hashlib.sha256(storage_source).hexdigest(),
                         'reference_source': str(reference_path),
                         'reference_sha256': hashlib.sha256((self.path / 'reference.json').read_bytes()).hexdigest(),
                         'stamp_point': 'AFTER_AIOHTTP_RECEIVE_BEFORE_JSON_PARSE',
                         'research_status': 'UNVERIFIED_SIGNATURES_UNCALIBRATED',
                         'durability': 'gzip flush at most 1s; no fsync; abrupt loss may truncate tail'}
        (self.path / 'manifest.json').write_text(json.dumps(self.metadata, indent=2) + '\n')
        self.q = queue.Queue(maxsize=20000)
        self.seq = self.persisted = 0
        self.error = None
        self.thread = threading.Thread(target=self._write, daemon=True)
        self.thread.start()


class Recorder:
    def __init__(self, writer, feed, key, secret):
        self.writer, self.feed, self.key, self.secret = writer, feed, key, secret
        self.stop = asyncio.Event()
        self.last_live_mono_ns = None
        self.last_observation = None
        self.n_live = self.n_refused = 0
        self.connected = False

    def event(self, kind, **fields):
        self.writer.emit('events', kind, **fields)

    def consume(self, raw, stamp, origin):
        try:
            decoded = decode_report(json.loads(raw), self.feed)
        except (ValueError, TypeError, KeyError, AttributeError):
            self.n_refused += 1
            self.writer.emit('events', 'REPORT_REFUSED', stamp=stamp, origin=origin)
            return
        if self.last_observation is not None and decoded['observationsTimestamp'] < self.last_observation:
            self.writer.emit('events', 'SOURCE_TIME_REGRESSION', stamp=stamp)
        self.last_observation = decoded['observationsTimestamp']
        self.writer.emit('decoded', 'REPORT', stamp=stamp, origin=origin, **decoded)
        if origin == 'WS':
            self.last_live_mono_ns = stamp['recv_mono_ns']
            self.n_live += 1

    async def latest(self, session):
        path = '/api/v1/reports/latest?feedID=' + self.feed
        sent = storage.clock()
        headers = auth_headers(path, self.key, self.secret, sent['recv_ns'] // 1_000_000)
        async with session.get(REST + path, headers=headers, allow_redirects=False) as response:
            raw = await response.text()
            stamp = storage.clock()
            # Provider error text may echo account identity; retain only its hash.
            self.writer.emit('rest', 'RESPONSE', stamp=stamp, sent=sent, path=path,
                             status=response.status, raw=raw if response.status == 200 else None,
                             body_sha256=hashlib.sha256(raw.encode()).hexdigest())
            if response.status in (400, 401, 403):
                raise AccessRefused('CHAINLINK_ACCESS_OR_FEED_ENTITLEMENT_REFUSED')
            if response.status != 200:
                raise RuntimeError('CHAINLINK_REST_UNAVAILABLE')
            self.consume(raw, stamp, 'REST_LATEST')
            if self.n_refused:
                raise RuntimeError('LATEST_REPORT_SCHEMA_REFUSED')

    async def health(self):
        while not self.stop.is_set():
            self.event('HEARTBEAT', connected=self.connected, n_live_reports=self.n_live,
                       n_refused=self.n_refused, queue_depth=self.writer.q.qsize(),
                       last_live_age_s=None if self.last_live_mono_ns is None else
                       (time.monotonic_ns() - self.last_live_mono_ns) / 1e9,
                       content_liveness='NOT_CERTIFIED')
            if self.writer.error:
                raise RuntimeError('WRITER_FAILED')
            try:
                await asyncio.wait_for(self.stop.wait(), 15)
            except asyncio.TimeoutError:
                pass

    async def stream(self, session):
        delay = 1
        while not self.stop.is_set():
            connection = uuid.uuid4().hex
            path = '/api/v1/ws?feedIDs=' + self.feed
            headers = auth_headers(path, self.key, self.secret, time.time_ns() // 1_000_000)
            try:
                async with session.ws_connect(WS + path, headers=headers, heartbeat=20,
                                              max_msg_size=2_000_000) as ws:
                    self.connected = True
                    delay = 1
                    self.event('WS_CONNECTED', connection_id=connection)
                    while not self.stop.is_set():
                        try:
                            msg = await asyncio.wait_for(ws.receive(), 1)
                        except asyncio.TimeoutError:
                            continue
                        stamp = storage.clock()
                        if msg.type == aiohttp.WSMsgType.TEXT:
                            self.writer.emit('chainlink_ws', 'FRAME', stamp=stamp,
                                             connection_id=connection, raw=msg.data)
                            self.consume(msg.data, stamp, 'WS')
                        elif msg.type == aiohttp.WSMsgType.BINARY:
                            self.writer.emit('chainlink_ws', 'BINARY_FRAME', stamp=stamp,
                                             connection_id=connection, raw_hex=msg.data.hex())
                            self.event('UNSUPPORTED_BINARY_FRAME', connection_id=connection)
                        elif msg.type in (aiohttp.WSMsgType.CLOSE, aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                            break
            except aiohttp.WSServerHandshakeError as exc:
                if exc.status in (400, 401, 403):
                    raise AccessRefused('CHAINLINK_WS_ACCESS_REFUSED') from None
                self.event('WS_FAILURE', connection_id=connection, error_type=type(exc).__name__)
            except (aiohttp.ClientError, OSError, asyncio.TimeoutError) as exc:
                self.event('WS_FAILURE', connection_id=connection, error_type=type(exc).__name__)
            finally:
                self.connected = False
                self.event('WS_BOUNDARY', connection_id=connection)
            if not self.stop.is_set():
                try:
                    await asyncio.wait_for(self.stop.wait(), delay)
                except asyncio.TimeoutError:
                    pass
                delay = min(delay * 2, 30)

    async def run(self):
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30)) as session:
            await self.latest(session)
            tasks = [asyncio.create_task(self.health()), asyncio.create_task(self.stream(session))]
            try:
                await asyncio.gather(*tasks)
            finally:
                for task in tasks:
                    task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
        if not self.n_live:
            raise RuntimeError('NO_LIVE_REPORTS_OBSERVED')


async def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-category', type=Path, required=True)
    parser.add_argument('--root', type=Path, default=Path('data/predict_chainlink'))
    parser.add_argument('--key-env', default='CHAINLINK_DATA_STREAMS_API_KEY')
    parser.add_argument('--secret-env', default='CHAINLINK_DATA_STREAMS_API_SECRET')
    parser.add_argument('--duration-s', type=float, default=0)
    args = parser.parse_args(argv)
    feed, reference = reference_feed(args.reference_category)
    key, secret = os.environ.get(args.key_env), os.environ.get(args.secret_env)
    if not key or not secret:
        parser.error(f'configure {args.key_env} and {args.secret_env}; Predict API credentials are not Chainlink credentials')
    if not 0 <= args.duration_s < float('inf'):
        parser.error('duration must be finite and nonnegative')
    writer = Writer(args.root, feed, reference, args.reference_category)
    recorder = Recorder(writer, feed, key, secret)
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, recorder.stop.set)
    timer = loop.call_later(args.duration_s, recorder.stop.set) if args.duration_s else None
    status = 'STOPPED_UNCERTIFIED'
    try:
        print(f'Recording exact-feed reports to {writer.path}', flush=True)
        await recorder.run()
    except BaseException as exc:
        status = 'FAILED_' + type(exc).__name__
        raise
    finally:
        if timer:
            timer.cancel()
        try:
            if not writer.error:
                recorder.event('PROCESS_END', n_live_reports=recorder.n_live, n_refused=recorder.n_refused,
                               last_observations_timestamp=recorder.last_observation)
        except BaseException:
            status = 'FAILED_FINAL_STATUS_WRITE'
            raise
        finally:
            await writer.close(status)


if __name__ == '__main__':
    try:
        asyncio.run(main())
    except AccessRefused:
        print('Chainlink access refused; configure credentials and exact-feed entitlement.')
        raise SystemExit(78)
