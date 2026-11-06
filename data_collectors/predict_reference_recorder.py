"""Record Predict's unsigned Chainlink reference-price relay, not signed reports.

The topic is used by Predict's public website but absent from its API docs.
Existing Predict credentials suffice. Preserve published seconds precision,
raw messages, duplicate observations and receipt clocks without interpolation.
"""
from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
from decimal import Decimal
import hashlib
import json
import os
from pathlib import Path
import queue
import signal
import threading
import time
import uuid

import aiohttp

from data_collectors import predict_fun_recorder as storage

SCHEMA = 'predict_reference_relay_v1'


def reference_identity(path):
    ref = json.loads(path.read_text())
    category = ref['category']
    storage.category_markets(category, category['slug'])
    mapping = ref['website_price_feed']
    markets = [m for m in category['markets'] if str(m['id']) == mapping['marketId']]
    if len(markets) != 1:
        raise ValueError('REFERENCE_MARKET_MISMATCH')
    variant = markets[0]['variantData']
    if (mapping['priceFeedProvider'] != 'CHAINLINK' or mapping['priceFeedSymbol'] != 'BTCUSDT'
            or variant['priceFeedProvider'] != 'CHAINLINK' or variant['priceFeedSymbol'] != 'BTCUSDT'
            or variant['priceFeedId'] != ref['provider_feed_id']):
        raise ValueError('REFERENCE_FEED_MISMATCH')
    relay_id = mapping['priceFeedId']
    if not isinstance(relay_id, str) or not relay_id.isdigit() or int(relay_id) <= 0:
        raise ValueError('REFERENCE_RELAY_ID')
    return relay_id, ref


def price_frame(raw, relay_id):
    frame = json.loads(raw, parse_float=Decimal)
    if not isinstance(frame, dict) or frame.get('type') != 'M' or frame.get('topic') != 'chainlinkAssetPriceUpdate/' + relay_id:
        raise ValueError('REFERENCE_TOPIC_MISMATCH')
    data = frame['data']
    if type(data.get('priceFeedId')) is not int or str(data['priceFeedId']) != relay_id:
        raise ValueError('REFERENCE_PAYLOAD_ID_MISMATCH')
    for key in ('publishTime', 'timestamp'):
        if type(data.get(key)) is not int or data[key] < 0:
            raise ValueError('REFERENCE_SOURCE_TIME_SCHEMA')
    if data['timestamp'] < data['publishTime']:
        raise ValueError('REFERENCE_RELAY_TIME_ORDER')
    if not isinstance(data.get('price'), (int, Decimal)) or isinstance(data['price'], bool):
        raise ValueError('REFERENCE_PRICE_SCHEMA')
    value = Decimal(data['price'])
    if not value.is_finite() or value <= 0:
        raise ValueError('REFERENCE_PRICE_INVALID')
    return {'price': str(value), 'publish_time_s': data['publishTime'],
            'relay_timestamp_s': data['timestamp'], 'relay_feed_id': relay_id,
            'reference_source': 'UNSIGNED_PREDICT_CHAINLINK_RELAY', 'source_time_precision': 'SECONDS'}


class Writer(storage.RawWriter):
    def __init__(self, root, relay_id, ref):
        self.run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ_') + uuid.uuid4().hex[:10]
        self.path = root / 'runs' / self.run_id
        self.path.mkdir(parents=True, exist_ok=False)
        own = Path(__file__).read_bytes()
        dependency = Path(storage.__file__).read_bytes()
        (self.path / 'collector_source.py').write_bytes(own)
        (self.path / 'storage_source.py').write_bytes(dependency)
        reference = json.dumps(ref, indent=2) + '\n'
        (self.path / 'reference.json').write_text(reference)
        self.metadata = {'schema': SCHEMA, 'run_id': self.run_id, 'started': storage.clock(),
                         'source_sha256': hashlib.sha256(own).hexdigest(),
                         'storage_source_sha256': hashlib.sha256(dependency).hexdigest(),
                         'reference_sha256': hashlib.sha256(reference.encode()).hexdigest(),
                         'topic': 'chainlinkAssetPriceUpdate/' + relay_id,
                         'provider_feed_id': ref['provider_feed_id'], 'symbol': 'BTCUSDT',
                         'mapping_as_of_ns': ref['as_of_ns'], 'endpoint': storage.WS,
                         'stamp_point': 'AFTER_AIOHTTP_RECEIVE_BEFORE_JSON_PARSE',
                         'source': 'UNSIGNED_PREDICT_CHAINLINK_RELAY',
                         'signature_verification': 'NO_SIGNED_REPORTS_DELIVERED',
                         'content_liveness': 'NOT_CERTIFIED',
                         'durability': 'gzip flush at most 1s; no fsync; abrupt loss may truncate tail'}
        (self.path / 'manifest.json').write_text(json.dumps(self.metadata, indent=2) + '\n')
        self.q = queue.Queue(maxsize=20000)
        self.seq = self.persisted = 0
        self.error = None
        self.thread = threading.Thread(target=self._write, daemon=True)
        self.thread.start()


class Recorder:
    def __init__(self, writer, relay_id, key):
        self.writer, self.relay_id, self.key = writer, relay_id, key
        self.topic = 'chainlinkAssetPriceUpdate/' + relay_id
        self.stop = asyncio.Event()
        self.n_prices = self.n_refused = self.n_duplicates = 0
        self.last_price_stamp = self.last_tuple = None
        self.last_publish_s = None
        self.connected = self.acked = False

    def event(self, kind, **fields):
        self.writer.emit('events', kind, **fields)

    def consume(self, raw, stamp):
        try:
            row = price_frame(raw, self.relay_id)
        except (ValueError, KeyError, TypeError, AttributeError):
            self.n_refused += 1
            self.writer.emit('events', 'PRICE_FRAME_REFUSED', stamp=stamp)
            return
        identity = (row['publish_time_s'], row['relay_timestamp_s'], row['price'])
        duplicate = identity == self.last_tuple
        regressed = self.last_publish_s is not None and row['publish_time_s'] < self.last_publish_s
        self.writer.emit('reference_prices', 'PRICE', stamp=stamp,
                         provider_feed_id=self.writer.metadata['provider_feed_id'],
                         duplicate_previous_payload=duplicate, publish_time_regressed=regressed, **row)
        self.n_prices += 1
        self.n_duplicates += duplicate
        self.last_tuple, self.last_publish_s, self.last_price_stamp = identity, row['publish_time_s'], stamp

    async def health(self):
        while not self.stop.is_set():
            sampled = storage.clock()
            health = {**sampled, 'run_id': self.writer.run_id, 'pid': os.getpid(),
                      'connected': self.connected, 'subscription_acknowledged': self.acked,
                      'n_prices': self.n_prices, 'n_refused': self.n_refused,
                      'n_duplicates': self.n_duplicates, 'queue_depth': self.writer.q.qsize(),
                      'last_price_receipt_age_s': None if self.last_price_stamp is None else
                      (sampled['recv_mono_ns'] - self.last_price_stamp['recv_mono_ns']) / 1e9,
                      'last_publish_time_s': self.last_publish_s,
                      'source_epoch_difference_s': None if self.last_publish_s is None else
                      sampled['recv_ns'] / 1e9 - self.last_publish_s,
                      'content_liveness': 'NOT_CERTIFIED'}
            self.event('HEARTBEAT', **health)
            path = self.writer.path / 'health.json'
            def save():
                temporary = path.with_suffix('.tmp')
                temporary.write_text(json.dumps(health, indent=2) + '\n')
                temporary.replace(path)
            await asyncio.to_thread(save)
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
            self.acked = False
            try:
                async with session.ws_connect(storage.WS, headers={'x-api-key': self.key},
                                              heartbeat=20, max_msg_size=1024 * 1024) as ws:
                    self.connected = True
                    self.event('WS_CONNECTED', connection_id=connection)
                    await ws.send_json({'method': 'subscribe', 'requestId': 1, 'params': [self.topic]})
                    self.event('SUBSCRIBE_REQUESTED', connection_id=connection, topic=self.topic, request_id=1)
                    started = last_rx = last_price_rx = time.monotonic()
                    while not self.stop.is_set():
                        if not self.acked and time.monotonic() - started > 15:
                            raise ValueError('SUBSCRIPTION_ACK_TIMEOUT')
                        if self.acked and time.monotonic() - last_price_rx > 45:
                            self.event('REFERENCE_PRICE_SILENT', connection_id=connection)
                            raise ValueError('REFERENCE_PRICE_SILENT')
                        try:
                            msg = await asyncio.wait_for(ws.receive(), 1)
                        except asyncio.TimeoutError:
                            if time.monotonic() - last_rx > 45:
                                raise ValueError('WS_SILENT')
                            continue
                        stamp = storage.clock()
                        if msg.type != aiohttp.WSMsgType.TEXT:
                            raise ValueError('WS_NON_TEXT_OR_CLOSED')
                        self.writer.emit('raw_ws', 'FRAME', stamp=stamp, connection_id=connection, raw=msg.data)
                        last_rx = time.monotonic()
                        frame = storage.parse_predict_frame(msg.data)
                        if frame['type'] == 'R':
                            if frame.get('requestId') != 1 or frame.get('success') is not True:
                                raise storage.FatalAccess('RELAY_SUBSCRIPTION_REFUSED')
                            self.acked = True
                            delay = 1
                            self.event('SUBSCRIPTION_ACKNOWLEDGED', connection_id=connection, topic=self.topic)
                        elif frame.get('topic') == 'heartbeat':
                            await ws.send_json({'method': 'heartbeat', 'data': frame['data']})
                        elif frame.get('topic') == self.topic:
                            before = self.n_prices
                            self.consume(msg.data, stamp)
                            if self.n_prices > before:
                                last_price_rx = time.monotonic()
                        else:
                            self.event('UNEXPECTED_TOPIC', connection_id=connection)
            except aiohttp.WSServerHandshakeError as exc:
                if exc.status in (400, 401, 403):
                    raise storage.FatalAccess('PREDICT_RELAY_ACCESS_REFUSED') from None
                self.event('WS_FAILURE', connection_id=connection, error_type=type(exc).__name__)
            except (aiohttp.ClientError, OSError, asyncio.TimeoutError, ValueError, KeyError) as exc:
                self.event('WS_FAILURE', connection_id=connection, error_type=type(exc).__name__)
            finally:
                self.connected = self.acked = False
                self.event('WS_BOUNDARY', connection_id=connection)
            if not self.stop.is_set():
                try:
                    await asyncio.wait_for(self.stop.wait(), delay)
                except asyncio.TimeoutError:
                    pass
                delay = min(delay * 2, 30)

    async def run(self):
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=None, sock_connect=15)) as session:
            tasks = [asyncio.create_task(self.health()), asyncio.create_task(self.stream(session))]
            try:
                await asyncio.gather(*tasks)
            finally:
                for task in tasks:
                    task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
        if not self.n_prices:
            raise RuntimeError('NO_REFERENCE_PRICES_OBSERVED')


async def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--root', type=Path, default=Path('data/predict_reference'))
    parser.add_argument('--api-key-env', default='PREDICT_FUN_API_KEY')
    parser.add_argument('--duration-s', type=float, default=0)
    args = parser.parse_args(argv)
    relay_id, ref = reference_identity(args.reference)
    key = os.environ.get(args.api_key_env)
    if not key:
        parser.error(f'configure {args.api_key_env}')
    if not 0 <= args.duration_s < float('inf'):
        parser.error('duration must be finite and nonnegative')
    writer = Writer(args.root, relay_id, ref)
    recorder = Recorder(writer, relay_id, key)
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, recorder.stop.set)
    timer = loop.call_later(args.duration_s, recorder.stop.set) if args.duration_s else None
    status = 'STOPPED_UNCERTIFIED'
    try:
        print(f'Recording unsigned Predict reference relay to {writer.path}', flush=True)
        await recorder.run()
    except BaseException as exc:
        status = 'FAILED_' + type(exc).__name__
        raise
    finally:
        if timer:
            timer.cancel()
        try:
            if not writer.error:
                recorder.event('PROCESS_END', n_prices=recorder.n_prices, n_refused=recorder.n_refused)
        except BaseException:
            status = 'FAILED_FINAL_STATUS_WRITE'
            raise
        finally:
            await writer.close(status)


if __name__ == '__main__':
    try:
        asyncio.run(main())
    except storage.FatalAccess:
        print('Predict reference relay access or subscription refused.')
        raise SystemExit(78)
