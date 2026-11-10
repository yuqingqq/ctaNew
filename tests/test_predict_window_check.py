"""Window monitor controls: real parser paths, absent feeds and refused corruption."""
import gzip
import hashlib
import json
from datetime import datetime, timezone

import pytest

from data_collectors import predict_window_check as qc

START = 1792675800  # 2026-10-22 13:30 UTC


def gz_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(gzip.compress(''.join(json.dumps(r) + '\n' for r in rows).encode()))


def setup_tape(tmp_path, monkeypatch):
    monkeypatch.setattr(qc.time, 'time_ns', lambda: (START + 720) * qc.NS)
    hour = datetime.fromtimestamp(START, timezone.utc).strftime('%Y%m%d_%H')
    for root, schema, source in [('predict', qc.predict.SCHEMA, 'predict_ws'),
                                 ('reference', qc.reference.SCHEMA, 'reference_prices')]:
        run = tmp_path / root / 'runs/r1'
        run.mkdir(parents=True)
        (run / 'collector_source.py').write_text('fixture')
        (run / 'manifest.json').write_text(json.dumps({
            'schema': schema, 'run_id': 'r1', 'source_sha256': hashlib.sha256(b'fixture').hexdigest(),
            'provider_feed_id': 'provider'}))
        rows, events = [], []
        for i in range(2):
            start = START + i * 300
            row = {'run_id': 'r1', 'seq': 10 + i * 5, 'source': source, 'recv_ns': (start + 1) * qc.NS}
            if root == 'predict':
                row['raw'] = json.dumps({'type': 'M', 'topic': f'predictOrderbook/{i+1}', 'data': {
                    'marketId': i+1, 'updateTimestampMs': start*1000,
                    'bids': [[.4, 2]], 'asks': [[.6, 2]]}})
                events.append({'kind': 'MARKET_DISCOVERED', 'market_id': i+1,
                               'slug': f'btc-updown-5m-{start}', 'end': start+300})
            else:
                row.update(reference_source='UNSIGNED_PREDICT_CHAINLINK_RELAY',
                           provider_feed_id='provider', publish_time_regressed=False)
            rows.append(row)
        gz_rows(run / source / (hour + '.jsonl.gz'), rows)
        if events:
            gz_rows(run / 'events' / (hour + '.jsonl.gz'), events)
    metas = []
    for i in range(2):
        start = START + i * 300
        slug = f'btc-updown-5m-{start}'
        metas.append({'slug': slug, 'recv_ns': start*qc.NS, 'window_start': start,
                      'window_end': start+300, 'outcomes': ['Up', 'Down'], 'clobTokenIds': ['UP', 'DOWN']})
        book = [{'asset_id': 'UP', 'timestamp': str(start*1000),
                 'bids': [{'price': '.4', 'size': '2'}], 'asks': [{'price': '.6', 'size': '2'}]}]
        path = tmp_path / 'pm/raw/20261022' / (slug + '.jsonl.gz')
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(gzip.compress(f'{(start+1)*qc.NS}\t{json.dumps(book)}\n'.encode()))
    (tmp_path / 'pm/markets.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in metas))
    for folder, kind in [('pm/prices/crypto_prices_twap_sixty', 'twap'),
                         ('pm/prices/crypto_prices', 'spot'), ('futures', 'futures')]:
        path = tmp_path / folder / (hour + '.csv')
        path.parent.mkdir(parents=True, exist_ok=True)
        lines = []
        for i in range(2):
            stamp = (START+i*300+1)*qc.NS
            if kind == 'twap':
                lines.append(f'{stamp}\t'+json.dumps({'topic': 'crypto_prices_twap_sixty',
                    'payload': {'symbol': 'btc/usd', 'window_s': 60}})+'\n')
            elif kind == 'spot':
                lines.append(f'{stamp},1,btcusdt,81000\n')
            else:
                lines.append(f'{stamp},1,1,1,81000,2,81001,2\n')
        path.write_text(''.join(lines))
    return ['--scope-start', '2026-10-22T13:29:01Z', '--predict-root', str(tmp_path/'predict'),
            '--reference-root', str(tmp_path/'reference'), '--pm-root', str(tmp_path/'pm'),
            '--futures-root', str(tmp_path/'futures'), '--out', str(tmp_path/'out')]


def receipt(tmp_path):
    return json.loads((tmp_path/'out/latest.json').read_text())


def test_nominal_windows_partial_start_and_grace():
    assert qc.expected_windows((START-59)*qc.NS, (START+419)*qc.NS, 120) == []
    assert qc.expected_windows((START-59)*qc.NS, (START+420)*qc.NS, 120) == [START]
    assert qc.expected_windows(START*qc.NS, (START+720)*qc.NS, 120) == [START, START+300]


def test_complete_tape_and_cached_receipt_provenance(tmp_path, monkeypatch):
    args = setup_tape(tmp_path, monkeypatch)
    qc.main(args)
    first = receipt(tmp_path)
    assert first['n_expected_windows'] == 2
    assert first['n_windows_without_some_feed_observations'] == 0
    assert not first['data_quality_certified']
    qc.main(args)
    second = receipt(tmp_path)
    assert second['n_windows_rechecked'] == 0
    assert second['windows'][0]['origin_receipt'] == first['windows'][0]['origin_receipt']
    assert (tmp_path/'out/receipts'/second['windows'][0]['origin_receipt']).exists()


def test_entire_feed_absent_is_missing_in_every_expected_window(tmp_path, monkeypatch):
    args = setup_tape(tmp_path, monkeypatch)
    next((tmp_path/'reference/runs/r1/reference_prices').glob('*.gz')).unlink()
    qc.main(args)
    result = receipt(tmp_path)
    assert result['n_expected_windows'] == 2
    assert result['n_windows_without_some_feed_observations'] == 2
    assert all(r['missing_observation_feeds'] == ['predict_reference'] for r in result['windows'])


def test_missing_window_and_missing_metadata_stay_in_expected_population(tmp_path, monkeypatch):
    args = setup_tape(tmp_path, monkeypatch)
    (tmp_path/'pm/raw/20261022'/f'btc-updown-5m-{START+300}.jsonl.gz').unlink()
    qc.main(args)
    assert receipt(tmp_path)['windows'][1]['pm_file_status'] == 'MISSING_FILE'
    assert receipt(tmp_path)['windows'][1]['missing_observation_feeds'] == ['pm_book']
    (tmp_path/'pm/markets.jsonl').unlink()
    qc.main(args + ['--recheck-all'])
    assert all(r['pm_metadata_status'] == 'MISSING_METADATA' for r in receipt(tmp_path)['windows'])


def test_wrong_run_identity_refused_at_cli_and_no_clean_receipt(tmp_path, monkeypatch):
    args = setup_tape(tmp_path, monkeypatch)
    path = next((tmp_path/'predict/runs/r1/predict_ws').glob('*.gz'))
    rows = [json.loads(line) for line in gzip.decompress(path.read_bytes()).splitlines()]
    rows[0]['run_id'] = 'wrong'
    gz_rows(path, rows)
    with pytest.raises(ValueError, match='RUN_IDENTITY'):
        qc.main(args)
    assert not (tmp_path/'out/latest.json').exists()
    assert json.loads((tmp_path/'out/last_failure.json').read_text())['status'] == 'CHECK_REFUSED'


def test_sealed_truncation_refused_active_prefix_flagged(tmp_path):
    path = tmp_path/'t.jsonl.gz'
    path.write_bytes(gzip.compress(b'{"valid":true}\n')[:-6])
    with pytest.raises(ValueError, match='TRUNCATED_SEALED_GZIP'):
        list(qc.input_lines(path, False, {}))
    evidence = {}
    assert list(qc.input_lines(path, True, evidence)) == ['{"valid":true}\n']
    assert evidence[str(path)]['tail'] == 'ACTIVE_GZIP_PREFIX'


def test_incomplete_feed_set_refused():
    with pytest.raises(ValueError, match='INCOMPLETE_EXPECTED_FEED_SET'):
        qc.make_window(START, {}, 'KNOWN_BEFORE_OPEN', 'SEALED_FILE')
