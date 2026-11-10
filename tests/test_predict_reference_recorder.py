import asyncio
import copy
import gzip
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import aiohttp
import pytest

from data_collectors import predict_reference_recorder as relay

FIXTURE = Path(__file__).parent / 'fixtures/predict_reference_relay.json'


def fixture():
    return json.loads(FIXTURE.read_text())


def test_live_captured_price_and_published_seconds_precision():
    row = relay.price_frame(fixture()['raw'], '1')
    assert row['price'] == '81740.005'
    assert row['publish_time_s'] == 1792675275
    assert row['relay_timestamp_s'] == 1792675276
    assert row['source_time_precision'] == 'SECONDS'
    assert row['reference_source'] == 'UNSIGNED_PREDICT_CHAINLINK_RELAY'


@pytest.mark.parametrize('field,value,reason', [
    ('priceFeedId', 2, 'REFERENCE_PAYLOAD_ID_MISMATCH'),
    ('priceFeedId', True, 'REFERENCE_PAYLOAD_ID_MISMATCH'),
    ('publishTime', 1792675275.5, 'REFERENCE_SOURCE_TIME_SCHEMA'),
    ('publishTime', True, 'REFERENCE_SOURCE_TIME_SCHEMA'),
    ('timestamp', 0, 'REFERENCE_RELAY_TIME_ORDER'),
    ('price', -1, 'REFERENCE_PRICE_INVALID'),
    ('price', True, 'REFERENCE_PRICE_SCHEMA'),
    ('price', 'NaN', 'REFERENCE_PRICE_SCHEMA')])
def test_corrupt_price_identity_and_source_timestamps_refused(field, value, reason):
    frame = json.loads(fixture()['raw'])
    frame['data'][field] = value
    with pytest.raises(ValueError, match=reason):
        relay.price_frame(json.dumps(frame), '1')


def test_wrong_topic_refused_and_decimal_wire_value_preserved():
    raw = fixture()['raw']
    with pytest.raises(ValueError, match='REFERENCE_TOPIC_MISMATCH'):
        relay.price_frame(raw, '2')
    raw = raw.replace('81740.005', '81740.0050000000000000001')
    assert relay.price_frame(raw, '1')['price'] == '81740.0050000000000000001'


def test_reference_market_mapping_and_provider_falsifiers(tmp_path):
    ref = fixture()['reference']
    path = tmp_path / 'ref.json'
    path.write_text(json.dumps(ref))
    assert relay.reference_identity(path)[0] == '1'
    for field, value, error in [('marketId', '1', 'REFERENCE_MARKET_MISMATCH'),
                                ('priceFeedProvider', 'BINANCE', 'REFERENCE_FEED_MISMATCH'),
                                ('priceFeedId', 'BTCUSDT', 'REFERENCE_RELAY_ID')]:
        changed = copy.deepcopy(ref)
        changed['website_price_feed'][field] = value
        path.write_text(json.dumps(changed))
        with pytest.raises(ValueError, match=error):
            relay.reference_identity(path)


def test_local_ws_ack_raw_capture_duplicates_and_refusal(tmp_path):
    obj = fixture()
    writer = relay.Writer(tmp_path, '1', obj['reference'])
    recorder = relay.Recorder(writer, '1', 'private-test-token')
    ack = json.dumps({'type': 'R', 'requestId': 1, 'success': True})
    heartbeat = json.dumps({'type': 'M', 'topic': 'heartbeat', 'data': 1792675276000})
    bad = json.loads(obj['raw'])
    bad['data']['priceFeedId'] = 2
    frames = [ack, obj['raw'], obj['raw'], heartbeat, json.dumps(bad)]
    sent = []

    class Socket:
        def __init__(self):
            self.messages = iter(frames)

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def send_json(self, data):
            sent.append(data)

        async def receive(self):
            message = next(self.messages, None)
            if message is None:
                recorder.stop.set()
                return SimpleNamespace(type=aiohttp.WSMsgType.CLOSED)
            return SimpleNamespace(type=aiohttp.WSMsgType.TEXT, data=message)

    class Session:
        def ws_connect(self, url, **kwargs):
            assert url == relay.storage.WS
            assert kwargs['headers'] == {'x-api-key': 'private-test-token'}
            return Socket()

    async def run():
        try:
            await asyncio.wait_for(recorder.stream(Session()), 2)
        finally:
            await writer.close('LOCAL_CONTROL')

    asyncio.run(run())
    assert recorder.n_prices == 2 and recorder.n_duplicates == 1 and recorder.n_refused == 1
    assert sent == [{'method': 'subscribe', 'requestId': 1, 'params': ['chainlinkAssetPriceUpdate/1']},
                    {'method': 'heartbeat', 'data': 1792675276000}]
    rows = []
    for path in writer.path.glob('*/*.gz'):
        with gzip.open(path, 'rt') as handle:
            rows.extend(json.loads(line) for line in handle)
    raw_rows = sorted((r for r in rows if r['source'] == 'raw_ws'), key=lambda x: x['seq'])
    assert [r['raw'] for r in raw_rows] == frames
    prices = sorted((r for r in rows if r['source'] == 'reference_prices'), key=lambda x: x['seq'])
    assert [r['duplicate_previous_payload'] for r in prices] == [False, True]
    assert prices[0]['recv_ns'] == raw_rows[1]['recv_ns']
    assert prices[0]['recv_mono_ns'] == raw_rows[1]['recv_mono_ns']
    assert sorted(r['seq'] for r in rows) == list(range(len(rows)))
    manifest = json.loads((writer.path / 'manifest.json').read_text())
    assert manifest['schema'] == relay.SCHEMA
    assert manifest['source_sha256'] == hashlib.sha256(Path(relay.__file__).read_bytes()).hexdigest()
    assert 'private-test-token' not in json.dumps(rows) + json.dumps(manifest)


def test_subscription_denial_is_fatal(tmp_path):
    writer = relay.Writer(tmp_path, '1', fixture()['reference'])
    recorder = relay.Recorder(writer, '1', 'private-token')

    class Socket:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def send_json(self, data):
            pass

        async def receive(self):
            return SimpleNamespace(type=aiohttp.WSMsgType.TEXT,
                                   data='{"type":"R","requestId":1,"success":false}')

    class Session:
        def ws_connect(self, *args, **kwargs):
            return Socket()

    async def run():
        try:
            with pytest.raises(relay.storage.FatalAccess):
                await recorder.stream(Session())
        finally:
            await writer.close('LOCAL_DENIAL_CONTROL')

    asyncio.run(run())
    assert recorder.n_prices == 0 and not recorder.acked
