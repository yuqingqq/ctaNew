"""Report/auth controls and local transport test; no Chainlink credentials."""
import asyncio
from collections import Counter
import copy
import gzip
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import aiohttp
import pytest

from data_collectors import predict_chainlink_recorder as oracle

FIXTURE = Path(__file__).parent / 'fixtures/chainlink_v3_sdk_example.json'


def example():
    return json.loads(FIXTURE.read_text())['envelope']


def change_word(envelope, position, value):
    result = copy.deepcopy(envelope)
    data = bytearray.fromhex(result['report']['fullReport'][2:])
    data[position:position + 32] = value.to_bytes(32, 'big')
    result['report']['fullReport'] = '0x' + data.hex()
    return result


def test_official_sdk_example_known_values_and_unscaled_precision():
    envelope = example()
    decoded = oracle.decode_report(envelope, envelope['report']['feedID'])
    # Fixed values cross-checked using eth-abi 5.2.0, independent of this decoder.
    assert decoded['validFromTimestamp'] == 1722348998
    assert decoded['observationsTimestamp'] == 1722348998
    assert decoded['expiresAt'] == 1722435398
    assert decoded['benchmark_price_integer'] == '695410242043786500'
    assert decoded['bid_integer'] == '695331621895065600'
    assert decoded['ask_integer'] == '695662122213138400'
    assert decoded['price_decimals'] is None
    assert decoded['signature_verification'] == 'NOT_PERFORMED'


def test_wrong_envelope_and_inner_feed_are_refused():
    envelope = example()
    feed = envelope['report']['feedID']
    with pytest.raises(ValueError, match='ENVELOPE_FEED_MISMATCH'):
        oracle.decode_report(envelope, '0x0003' + '0' * 60)
    offset = int.from_bytes(bytes.fromhex(envelope['report']['fullReport'][2:])[96:128], 'big')
    mutated = change_word(envelope, offset + 32, 0)
    with pytest.raises(ValueError, match='REPORT_FEED_OR_SCHEMA_MISMATCH'):
        oracle.decode_report(mutated, feed)


@pytest.mark.parametrize('position,value,message', [
    (96, 0, 'DYNAMIC_OFFSET'), (96, 225, 'DYNAMIC_OFFSET'),
    (224, 10**9, 'TRUNCATED_DYNAMIC_DATA'), (128, 768, 'OVERLAPPING_DYNAMIC_DATA')])
def test_malformed_abi_offsets_and_overlap_refused(position, value, message):
    envelope = example()
    with pytest.raises(ValueError, match=message):
        oracle.decode_report(change_word(envelope, position, value), envelope['report']['feedID'])


def test_truncated_full_report_refused():
    envelope = example()
    envelope['report']['fullReport'] = envelope['report']['fullReport'][:-64]
    with pytest.raises(ValueError, match='TRUNCATED_DYNAMIC_DATA'):
        oracle.decode_report(envelope, envelope['report']['feedID'])


def test_noncanonical_integer_and_time_order_refused():
    envelope = example()
    blob = int.from_bytes(bytes.fromhex(envelope['report']['fullReport'][2:])[96:128], 'big') + 32
    for pos, value, message in [(blob + 32, 1 << 32, 'NONCANONICAL_UNSIGNED_WORD'),
                                (blob + 6 * 32, 1 << 200, 'NONCANONICAL_SIGNED_WORD'),
                                (blob + 5 * 32, 1, 'REPORT_TIME_ORDER')]:
        with pytest.raises(ValueError, match=message):
            oracle.decode_report(change_word(envelope, pos, value), envelope['report']['feedID'])


def test_hmac_headers_exact_query_signature():
    # Fixed expected HMAC independently computed with OpenSSL dgst -sha256.
    headers = oracle.auth_headers('/api/v1/ws?feedIDs=0x1234', 'test-key', 'test-secret', 1700000000123)
    assert headers['Authorization'] == 'test-key'
    assert headers['X-Authorization-Timestamp'] == '1700000000123'
    assert headers['X-Authorization-Signature-SHA256'] == '5407217f0e3f09b149b59f95bf6263f6502a15a15b1376752e5911029cc933a5'
    changed = oracle.auth_headers('/api/v1/ws?feedIDs=0x4321', 'test-key', 'test-secret', 1700000000123)
    assert changed['X-Authorization-Signature-SHA256'] != headers['X-Authorization-Signature-SHA256']


def reference(tmp_path):
    feed = example()['report']['feedID']
    market = {'id': 42, 'variantData': {'priceFeedProvider': 'CHAINLINK',
              'priceFeedSymbol': 'BTCUSDT', 'priceFeedId': feed},
              'outcomes': [{'name': 'Up', 'indexSet': 1, 'onChainId': '101'},
                           {'name': 'Down', 'indexSet': 2, 'onChainId': '102'}]}
    obj = {'category': {'slug': 'btc-updown-5m-0', 'marketVariant': 'CRYPTO_UP_DOWN',
           'startsAt': '1970-01-01T00:00:00Z', 'endsAt': '1970-01-01T00:05:00Z', 'markets': [market]}}
    path = tmp_path / 'reference.json'
    path.write_text(json.dumps(obj))
    return path, obj


def test_reference_pins_feed_and_refuses_wrong_provider(tmp_path):
    path, obj = reference(tmp_path)
    feed, _ = oracle.reference_feed(path)
    assert feed == example()['report']['feedID']
    obj['category']['markets'][0]['variantData']['priceFeedProvider'] = 'PYTH'
    path.write_text(json.dumps(obj))
    with pytest.raises(ValueError, match='REFERENCE_FEED_IDENTITY'):
        oracle.reference_feed(path)


def test_missing_credentials_create_no_run(tmp_path, monkeypatch):
    path, _ = reference(tmp_path)
    monkeypatch.delenv('CHAINLINK_DATA_STREAMS_API_KEY', raising=False)
    monkeypatch.delenv('CHAINLINK_DATA_STREAMS_API_SECRET', raising=False)
    out = tmp_path / 'absent-output'
    with pytest.raises(SystemExit) as error:
        asyncio.run(oracle.main(['--reference-category', str(path), '--root', str(out)]))
    assert error.value.code == 2
    assert not out.exists()


def test_local_ws_preserves_raw_bad_frame_and_own_storage_identity(tmp_path):
    path, obj = reference(tmp_path)
    feed = example()['report']['feedID']
    writer = oracle.Writer(tmp_path / 'output', feed, obj, path)
    recorder = oracle.Recorder(writer, feed, 'secret-test-key', 'secret-test-secret')
    good = json.dumps(example())
    bad = json.dumps({'report': {'feedID': '0x' + '0' * 64, 'fullReport': '0x00'}})

    class Socket:
        def __init__(self):
            self.messages = iter([good, bad])

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def receive(self):
            message = next(self.messages, None)
            if message is None:
                recorder.stop.set()
                return SimpleNamespace(type=aiohttp.WSMsgType.CLOSED)
            return SimpleNamespace(type=aiohttp.WSMsgType.TEXT, data=message)

    class Session:
        def ws_connect(self, url, **kwargs):
            assert url == oracle.WS + '/api/v1/ws?feedIDs=' + feed
            assert kwargs['headers']['Authorization'] == 'secret-test-key'
            return Socket()

    async def run():
        await recorder.stream(Session())
        await writer.close('LOCAL_CONTROL')

    asyncio.run(run())
    assert recorder.n_live == 1 and recorder.n_refused == 1
    rows = []
    for file in writer.path.glob('*/*.gz'):
        with gzip.open(file, 'rt') as handle:
            rows.extend(json.loads(line) for line in handle)
    assert sorted(r['seq'] for r in rows) == list(range(len(rows)))
    frames = [r for r in rows if r['source'] == 'chainlink_ws']
    assert [r['raw'] for r in sorted(frames, key=lambda x: x['seq'])] == [good, bad]
    decoded = next(r for r in rows if r['source'] == 'decoded')
    assert decoded['recv_ns'] == frames[0]['recv_ns']
    assert decoded['recv_mono_ns'] == frames[0]['recv_mono_ns']
    manifest = json.loads((writer.path / 'manifest.json').read_text())
    assert manifest['schema'] == oracle.SCHEMA
    assert manifest['source_sha256'] == hashlib.sha256(Path(oracle.__file__).read_bytes()).hexdigest()
    artifacts = ''.join(p.read_text() for p in writer.path.glob('*.json')) + json.dumps(rows)
    assert 'secret-test-key' not in artifacts and 'secret-test-secret' not in artifacts


def test_access_denial_is_fatal_and_error_body_cannot_leak_account_identity(tmp_path):
    path, obj = reference(tmp_path)
    feed = example()['report']['feedID']
    writer = oracle.Writer(tmp_path / 'output', feed, obj, path)
    recorder = oracle.Recorder(writer, feed, 'private-account-key', 'private-account-secret')

    class Response:
        status = 401

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def text(self):
            return 'access refused for private-account-key'

    class Session:
        def get(self, url, **kwargs):
            assert url == oracle.REST + '/api/v1/reports/latest?feedID=' + feed
            assert kwargs['allow_redirects'] is False
            return Response()

    async def run():
        try:
            with pytest.raises(oracle.AccessRefused):
                await recorder.latest(Session())
        finally:
            await writer.close('LOCAL_DENIED_CONTROL')

    asyncio.run(run())
    assert recorder.n_live == 0
    for file in writer.path.glob('*/*.gz'):
        with gzip.open(file, 'rt') as handle:
            text = handle.read()
            assert 'private-account-key' not in text
            assert 'private-account-secret' not in text
            row = json.loads(text)
            assert row['status'] == 401 and row['raw'] is None
