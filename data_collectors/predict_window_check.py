"""Read-only BTC 5m window observation checks, with no API calls or P&L.

Expected windows come from the clock and configured collection start, not
from discovered files. Nonempty windows are not certified complete. Reports
distinguish absent observations from long update gaps on a quiet feed.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import fcntl
from datetime import datetime, timezone
import gzip
import hashlib
import io
import json
from pathlib import Path
import time
import uuid

from data_collectors import predict_fun_recorder as predict
from data_collectors import predict_pm_preflight as pm
from data_collectors import predict_reference_recorder as reference

NS = 1_000_000_000
FEEDS = ('predict_book', 'predict_reference', 'pm_book', 'pm_twap60',
         'binance_futures', 'spot_mirror')


def expected_windows(start_ns, now_ns, grace_s):
    if start_ns < 0 or now_ns < start_ns or grace_s < 0:
        raise ValueError('INVALID_SCOPE_CLOCK')
    first = ((start_ns + 300 * NS - 1) // (300 * NS)) * 300
    last_end = ((now_ns - grace_s * NS) // (300 * NS)) * 300
    return list(range(first, max(first, last_end), 300))


def utc_ns(text):
    return int(predict.utc_timestamp(text) * NS)


def hours_for(windows):
    return {datetime.fromtimestamp(t, timezone.utc).strftime('%Y%m%d_%H') for t in windows}


def input_lines(path, active, inputs):
    """Read one captured file prefix; sealed corruption is always refused."""
    raw = path.read_bytes()
    entry = {'sha256': hashlib.sha256(raw).hexdigest(), 'bytes': len(raw), 'tail': 'COMPLETE'}
    inputs[str(path)] = entry
    handle = gzip.GzipFile(fileobj=io.BytesIO(raw)) if path.suffix == '.gz' else io.BytesIO(raw)
    try:
        for line in handle:
            if not line.endswith(b'\n'):
                if not active:
                    raise ValueError('SEALED_PARTIAL_LINE')
                entry['tail'] = 'ACTIVE_PARTIAL_LINE'
                break
            yield line.decode('utf-8')
    except EOFError:
        if not active:
            raise ValueError('TRUNCATED_SEALED_GZIP') from None
        entry['tail'] = 'ACTIVE_GZIP_PREFIX'
    finally:
        handle.close()


def add_point(points, window_set, feed, recv_ns, flag=None):
    if type(recv_ns) is not int or recv_ns < 0:
        raise ValueError('INVALID_RECEIPT_TIMESTAMP')
    slot = recv_ns // (300 * NS) * 300
    if slot in window_set:
        points[slot][feed].append(recv_ns)
        if flag:
            points[slot][feed + '_flags'][flag] += 1


def metric(times, start, flags):
    ordered = sorted(times)
    if not ordered:
        return {'n_observations': 0, 'status': 'NO_OBSERVATIONS', 'flags': dict(flags)}
    bounds = [start * NS, *ordered, (start + 300) * NS]
    gap = max(b - a for a, b in zip(bounds, bounds[1:])) / NS
    return {'n_observations': len(ordered), 'status': 'OBSERVATIONS_PRESENT',
            'first_recv_ns': ordered[0], 'last_recv_ns': ordered[-1],
            'max_receipt_gap_including_edges_s': gap,
            'update_gap_over_5s': gap > 5, 'flags': dict(flags)}


def make_window(start, metrics, metadata_status, pm_file_status):
    if set(metrics) != set(FEEDS):
        raise ValueError('INCOMPLETE_EXPECTED_FEED_SET')
    missing = [name for name in FEEDS if metrics[name]['n_observations'] == 0]
    return {'start_s': start, 'end_s': start + 300,
            'slug': f'btc-updown-5m-{start}', 'missing_observation_feeds': missing,
            'pm_metadata_status': metadata_status, 'pm_file_status': pm_file_status,
            'long_update_gap_feeds': [name for name in FEEDS if metrics[name].get('update_gap_over_5s')],
            'feeds': metrics}


def scan_predict(root, source, windows, points, inputs, diagnostics, now_ns):
    hours = hours_for(windows)
    current_hour = datetime.fromtimestamp(now_ns / NS, timezone.utc).strftime('%Y%m%d_%H')
    for run in sorted(root.glob('runs/*')):
        paths = sorted((run / source).glob('*.jsonl.gz'))
        paths = [p for p in paths if p.name.split('.')[0] in hours]
        if not paths:
            continue
        manifest = json.loads((run / 'manifest.json').read_text())
        expected_schema = predict.SCHEMA if source == 'predict_ws' else reference.SCHEMA
        if manifest['schema'] != expected_schema:
            raise ValueError('UNSUPPORTED_COLLECTOR_SCHEMA')
        if pm.digest(run / 'collector_source.py') != manifest['source_sha256']:
            raise ValueError('COLLECTOR_SOURCE_IDENTITY')
        for path in paths:
            active = path.name.startswith(current_hour) and not (run / 'summary.json').exists()
            previous_seq = -1
            for line in input_lines(path, active, inputs):
                row = json.loads(line)
                if row['run_id'] != manifest['run_id'] or type(row['seq']) is not int or row['seq'] <= previous_seq:
                    raise ValueError('RUN_IDENTITY_OR_SEQUENCE_REGRESSION')
                previous_seq = row['seq']
                if row['source'] != source:
                    raise ValueError('SOURCE_DIRECTORY_IDENTITY')
                if source == 'reference_prices':
                    if row['reference_source'] != 'UNSIGNED_PREDICT_CHAINLINK_RELAY' or row['provider_feed_id'] != manifest['provider_feed_id']:
                        raise ValueError('REFERENCE_SOURCE_IDENTITY')
                    add_point(points, windows, 'predict_reference', row['recv_ns'],
                              'PUBLISH_TIME_REGRESSION' if row['publish_time_regressed'] else None)
                    continue
                frame = predict.parse_predict_frame(row['raw'])
                if frame['type'] != 'M' or not frame['topic'].startswith('predictOrderbook/'):
                    continue
                mid = int(frame['topic'].split('/')[1])
                # Metadata is producer-recorded; never derive a market's window from receive time.
                mapping = diagnostics['predict_market_windows'].get(str(mid))
                if mapping is None:
                    diagnostics['unmapped_book_frames'] += 1
                    continue
                start = mapping
                if start not in windows or not start * NS <= row['recv_ns'] < (start + 300) * NS:
                    continue
                check = predict.book_diagnostics(frame['data'], mid)
                flag = 'LOCKED_OR_CROSSED' if check['locked_or_crossed'] else 'ONE_SIDED_OR_EMPTY' if not check['two_sided'] else None
                add_point(points, windows, 'predict_book', row['recv_ns'], flag)


def market_mappings(root, windows, inputs, now_ns):
    hours = hours_for(windows | {min(windows) - 300})
    current_hour = datetime.fromtimestamp(now_ns / NS, timezone.utc).strftime('%Y%m%d_%H')
    mapping = {}
    for run in sorted(root.glob('runs/*')):
        for path in sorted((run / 'events').glob('*.jsonl.gz')):
            if path.name.split('.')[0] not in hours:
                continue
            active = path.name.startswith(current_hour) and not (run / 'summary.json').exists()
            for line in input_lines(path, active, inputs):
                row = json.loads(line)
                if row['kind'] == 'MARKET_DISCOVERED':
                    start = row['end'] - 300
                    if row['slug'] != f'btc-updown-5m-{start}':
                        raise ValueError('PREDICT_MARKET_WINDOW_IDENTITY')
                    mid = str(row['market_id'])
                    if mid in mapping and mapping[mid] != start:
                        raise ValueError('PREDICT_CONFLICTING_MARKET_IDENTITY')
                    mapping[mid] = start
    return mapping


def scan_pm(root, windows, points, inputs, now_ns, diagnostics):
    slugs = {f'btc-updown-5m-{t}' for t in windows}
    issues = Counter()
    if (root / 'markets.jsonl').exists():
        metas = pm.metadata(root / 'markets.jsonl', slugs, now_ns, issues)
    else:
        metas = {}
        issues['MISSING_METADATA_LEDGER'] += 1
    diagnostics['unscoped_metadata_issues'] = dict(issues)
    statuses = {}
    for start in windows:
        slug = f'btc-updown-5m-{start}'
        meta = metas.get(slug)
        status = 'MISSING_METADATA'
        if meta:
            if meta['window_start'] != start or meta['window_end'] != start + 300:
                raise ValueError('PM_WINDOW_IDENTITY')
            status = 'KNOWN_BEFORE_OPEN' if meta['recv_ns'] <= start * NS else 'LEARNED_AFTER_OPEN'
            diagnostics['selected_pm_metadata_sha256'][slug] = hashlib.sha256(json.dumps(meta, sort_keys=True).encode()).hexdigest()
        day = datetime.fromtimestamp(start, timezone.utc).strftime('%Y%m%d')
        stem = root / 'raw' / day / (slug + '.jsonl')
        path = Path(str(stem) + '.gz') if Path(str(stem) + '.gz').exists() else stem
        file_status = 'SEALED_FILE' if path.suffix == '.gz' else 'UNSEALED_FILE'
        if not path.exists():
            file_status = 'MISSING_FILE'
        elif meta:
            for line in input_lines(path, path.suffix != '.gz', inputs):
                if not line.strip():
                    continue
                stamp, raw = line.split('\t', 1)
                recv = int(stamp)
                if not start * NS <= recv < (start + 300) * NS:
                    continue
                for _, bid, ask in pm.pm_updates(json.loads(raw), meta['_up_token']):
                    touch = pm.touch_status(bid, ask)
                    add_point(points, windows, 'pm_book', recv, touch if touch != 'TWO_SIDED' else None)
        statuses[start] = status, file_status
    return statuses


def scan_external(folder, feed, windows, points, inputs, now_ns):
    hours = hours_for(windows)
    current_hour = datetime.fromtimestamp(now_ns / NS, timezone.utc).strftime('%Y%m%d_%H')
    for hour in sorted(hours):
        stem = folder / (hour + '.csv')
        path = Path(str(stem) + '.gz') if Path(str(stem) + '.gz').exists() else stem
        if not path.exists():
            continue
        active = path.suffix != '.gz' and hour == current_hour
        for line in input_lines(path, active, inputs):
            if not line.strip():
                continue
            if feed == 'pm_twap60':
                stamp, raw = line.split('\t', 1)
                frame = json.loads(raw)
                data = frame['payload']
                if data['symbol'].lower() != 'btc/usd':
                    continue
                if frame['topic'] != 'crypto_prices_twap_sixty' or data['window_s'] != 60:
                    raise ValueError('PM_TWAP_IDENTITY')
                add_point(points, windows, feed, int(stamp))
            else:
                fields = line.strip().split(',')
                if fields[0] == 'recv_ns':
                    continue
                if feed == 'spot_mirror' and fields[2].lower() != 'btcusdt':
                    continue
                if feed == 'binance_futures' and len(fields) != 8:
                    raise ValueError('FUTURES_ROW_SCHEMA')
                add_point(points, windows, feed, int(fields[0]))


def run(args):
    now_ns = time.time_ns()
    start_ns = utc_ns(args.scope_start)
    expected = expected_windows(start_ns, now_ns, args.grace_s)
    name = datetime.fromtimestamp(now_ns / NS, timezone.utc).strftime('%Y%m%dT%H%M%SZ_') + uuid.uuid4().hex[:8] + '.json'
    args.out.mkdir(parents=True, exist_ok=True)
    code_hash = pm.digest(Path(__file__))
    dependencies = {str(Path(m.__file__)): pm.digest(Path(m.__file__)) for m in (predict, pm, reference)}
    config = {'scope_start_ns': start_ns, 'grace_s': args.grace_s,
              'predict_root': str(args.predict_root), 'reference_root': str(args.reference_root),
              'pm_root': str(args.pm_root), 'futures_root': str(args.futures_root),
              'code_sha256': code_hash, 'dependencies': dependencies}
    previous = json.loads((args.out / 'latest.json').read_text()) if (args.out / 'latest.json').exists() else {}
    old_rows = {r['start_s']: r for r in previous.get('windows', [])} if previous.get('config') == config else {}
    selected = set(expected) if args.recheck_all else {s for s in expected if s not in old_rows or old_rows[s]['missing_observation_feeds']}
    points = {s: {**{name: [] for name in FEEDS}, **{name + '_flags': Counter() for name in FEEDS}} for s in selected}
    diagnostics = {'predict_market_windows': {}, 'unmapped_book_frames': 0,
                   'selected_pm_metadata_sha256': {}, 'unscoped_metadata_issues': {}}
    inputs = {}
    if selected:
        print(f'Checking {len(selected)} mature BTC windows', flush=True)
        diagnostics['predict_market_windows'] = market_mappings(args.predict_root, selected, inputs, now_ns)
        scan_predict(args.predict_root, 'predict_ws', selected, points, inputs, diagnostics, now_ns)
        scan_predict(args.reference_root, 'reference_prices', selected, points, inputs, diagnostics, now_ns)
        pm_statuses = scan_pm(args.pm_root, selected, points, inputs, now_ns, diagnostics)
        for feed, folder in [('pm_twap60', args.pm_root / 'prices/crypto_prices_twap_sixty'),
                             ('spot_mirror', args.pm_root / 'prices/crypto_prices'),
                             ('binance_futures', args.futures_root)]:
            scan_external(folder, feed, selected, points, inputs, now_ns)
        for start in selected:
            metrics = {name: metric(points[start][name], start, points[start][name + '_flags']) for name in FEEDS}
            old_rows[start] = make_window(start, metrics, *pm_statuses[start])
            old_rows[start]['checked_at_ns'] = now_ns
            old_rows[start]['origin_receipt'] = name
    windows = [old_rows[s] for s in expected]
    missing = [r for r in windows if r['missing_observation_feeds']]
    receipt = {'purpose': 'WINDOW_OBSERVATION_MONITOR_NOT_DAY_CERTIFICATION',
               'as_of_utc': datetime.fromtimestamp(now_ns / NS, timezone.utc).isoformat(),
               'as_of_ns': now_ns, 'config': config, 'n_expected_windows': len(expected),
               'n_windows_rechecked': len(selected), 'n_windows_without_some_feed_observations': len(missing),
               'status': 'WAITING_FOR_FIRST_MATURE_WINDOW' if not expected else 'MISSING_OBSERVATIONS' if missing else 'OBSERVATIONS_PRESENT_ALL_EXPECTED_WINDOWS',
               'data_quality_certified': False, 'windows': windows, 'input_prefixes': inputs,
               'diagnostics': diagnostics,
               'limits': ['Presence is not completeness, queue initialization, fill identification, or day admission.',
                          'An update gap can be a quiet market or upstream publication delay; it is not proven packet loss.',
                          'Source sequence jumps within one directory are expected because local sequencing interleaves sources.',
                          'Previously observed windows are reused; --recheck-all rereads them. Missing observations are retried.',
                          'No outcomes, P&L, Dublin execution measurements, or pre-collection history inferred.']}
    destination = args.out / 'receipts' / name
    destination.parent.mkdir(exist_ok=True)
    with destination.open('x') as handle:
        json.dump(receipt, handle, indent=2)
        handle.write('\n')
    temporary = args.out / 'latest.tmp'
    temporary.write_text(json.dumps(receipt, indent=2) + '\n')
    temporary.replace(args.out / 'latest.json')
    print(json.dumps({k: receipt[k] for k in ('as_of_utc', 'n_expected_windows', 'n_windows_rechecked',
                                            'n_windows_without_some_feed_observations', 'status')}), flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scope-start', required=True)
    parser.add_argument('--grace-s', type=int, default=120)
    parser.add_argument('--predict-root', type=Path, default=Path('data/predict_fun'))
    parser.add_argument('--reference-root', type=Path, default=Path('data/predict_reference'))
    parser.add_argument('--pm-root', type=Path, default=Path('data/pm_5min'))
    parser.add_argument('--futures-root', type=Path, default=Path('data/mm_hf/raw/bookTicker/BTCUSDT'))
    parser.add_argument('--out', type=Path, default=Path('data/predict_fun/window_checks'))
    parser.add_argument('--recheck-all', action='store_true')
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    lock = (args.out / '.lock').open('a')
    try:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        lock.close()
        raise SystemExit('CHECK_ALREADY_RUNNING') from None
    try:
        run(args)
    except Exception as exc:
        args.out.mkdir(parents=True, exist_ok=True)
        failure = {'status': 'CHECK_REFUSED', 'as_of_ns': time.time_ns(), 'error_type': type(exc).__name__,
                   'error': str(exc), 'data_quality_certified': False}
        (args.out / 'last_failure.json').write_text(json.dumps(failure, indent=2) + '\n')
        raise
    finally:
        lock.close()


if __name__ == '__main__':
    main()
