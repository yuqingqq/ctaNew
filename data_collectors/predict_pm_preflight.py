"""Read-only, outcome-blind cross-venue data feasibility check.

This counts book availability on a diagnostic time grid. Rows are NOT orders,
fills, independent samples, or a market-making performance test. No API calls.
Only sealed gzip files are read; existing outputs are never overwritten.
"""
from __future__ import annotations

import argparse
from bisect import bisect_right
from collections import Counter
from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
from pathlib import Path

NS = 1_000_000_000


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def touch_status(bid, ask):
    if bid is None or ask is None:
        return 'ONE_SIDED_OR_EMPTY'
    if not all(math.isfinite(v) and 0 <= v <= 1 for v in (bid, ask)):
        raise ValueError('INVALID_PRICE')
    return 'CROSSED' if bid > ask else 'LOCKED' if bid == ask else 'TWO_SIDED'


def select_windows(audit, stride):
    if stride < 1:
        raise ValueError('INVALID_STRIDE')
    cohort = audit['complete_window_coverage']
    if not cohort or len({w['start_s'] for w in cohort}) != len(cohort):
        raise ValueError('EMPTY_OR_DUPLICATED_COHORT')
    if [w['start_s'] for w in cohort] != sorted(w['start_s'] for w in cohort):
        raise ValueError('UNSORTED_COHORT')
    stress_ids = {g['market_id'] for g in audit['book_gap_alerts']}
    regular_ids = {w['market_id'] for w in cohort[::stride]}
    return [{**w, 'selection': 'REGULAR_AND_STRESS' if w['market_id'] in regular_ids & stress_ids
             else 'REGULAR' if w['market_id'] in regular_ids else 'KNOWN_GAP_STRESS'}
            for w in cohort if w['market_id'] in regular_ids | stress_ids]


def metadata(path, slugs, cutoff, issues=None):
    first = {}
    with path.open() as handle:
        for line in handle:
            row = json.loads(line)
            if not isinstance(row, dict) or 'slug' not in row:
                if issues is not None:
                    issues['UNIDENTIFIABLE_METADATA_RECORD'] += 1
                continue
            if row['slug'] in slugs and row['recv_ns'] <= cutoff:
                previous = first.get(row['slug'])
                if previous is None or row['recv_ns'] < previous['recv_ns']:
                    first[row['slug']] = row
    for row in first.values():
        names = json.loads(row['outcomes']) if isinstance(row['outcomes'], str) else row['outcomes']
        tokens = row['clobTokenIds']
        if isinstance(tokens, str):
            tokens = json.loads(tokens)
        if names != ['Up', 'Down'] or len(tokens) != 2 or tokens[0] == tokens[1]:
            raise ValueError('PM_OUTCOME_IDENTITY')
        row['_up_token'] = str(tokens[0])
    return first


def pm_updates(obj, token):
    """Use explicitly published UP BBO; do not infer fills or combine tokens."""
    result = []
    for event in obj if isinstance(obj, list) else [obj]:
        if not isinstance(event, dict):
            raise ValueError('PM_EVENT_SCHEMA')
        source = event.get('timestamp')
        if 'bids' in event and 'asks' in event and str(event.get('asset_id')) == token:
            bids = [float(x['price']) for x in event['bids'] if float(x['size']) > 0]
            asks = [float(x['price']) for x in event['asks'] if float(x['size']) > 0]
            result.append((source, max(bids, default=None), min(asks, default=None)))
        if event.get('event_type') == 'price_change':
            for change in event['price_changes']:
                if str(change.get('asset_id')) == token:
                    bid, ask = float(change['best_bid']), float(change['best_ask'])
                    # PM price-change BBO uses boundary sentinels for an absent side.
                    # Neither 0 nor 1 is a tradable PM limit price.
                    result.append((source, None if bid == 0 else bid, None if ask == 1 else ask))
    return result


def append_state(states, recv, source, bid, ask):
    if states and recv < states[-1][0]:
        raise ValueError('RECEIPT_REGRESSION')
    kind = touch_status(bid, ask)
    source_ns = None if source is None else int(source) * 1_000_000
    states.append((recv, source_ns, kind))


def available(states, times, when, max_age_ns):
    index = bisect_right(times, when) - 1
    if index < 0:
        return 'NO_BOOK_YET'
    recv, source, kind = states[index]
    if kind != 'TWO_SIDED':
        return kind
    if when - recv > max_age_ns:
        return 'RECEIPT_OLDER_THAN_LIMIT'
    if source is None:
        return 'NO_SOURCE_TIMESTAMP'
    if source > recv:
        return 'SOURCE_CLOCK_AHEAD'
    if when - source > 5 * NS:
        return 'SOURCE_OLDER_THAN_5S'
    return 'AVAILABLE'


def write_json(path, value):
    with path.open('x') as handle:
        json.dump(value, handle, indent=2)
        handle.write('\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit', type=Path, required=True)
    parser.add_argument('--pm-root', type=Path, default=Path('data/pm_5min'))
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--stride', type=int, default=10)
    parser.add_argument('--supersedes', type=Path)
    args = parser.parse_args()
    snapshot_path = args.audit / 'snapshot.json'
    audit_path = args.audit / 'full_prefix_audit.json'
    snapshot = json.loads(snapshot_path.read_text())
    audit = json.loads(audit_path.read_text())
    if not audit['sequence_reconciled'] or audit['cutoff_recv_ns'] != snapshot['cutoff_recv_ns']:
        raise ValueError('AUDIT_IDENTITY_OR_SEQUENCE')
    run = Path(snapshot['run_path'])
    if audit['run_id'] != snapshot['manifest']['run_id'] or digest(run / 'collector_source.py') != audit['collector_source_sha256']:
        raise ValueError('COLLECTOR_IDENTITY')
    windows = select_windows(audit, args.stride)
    cutoff = snapshot['cutoff_recv_ns']
    limit = snapshot['target_exclusive_seq']
    args.out.mkdir(parents=True, exist_ok=False)
    write_json(args.out / 'design.json', {
        'purpose': 'OUTCOME_BLIND_FEASIBILITY_NOT_SIGNAL_OR_PNL',
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'input_cutoff_ns': cutoff, 'target_exclusive_seq': limit,
        'source_sha256': digest(Path(__file__)),
        'supersedes': str(args.supersedes) if args.supersedes else None,
        'population': windows, 'parent_cohort_n': len(audit['complete_window_coverage']),
        'grid': '300 points per window, at start + 0.5s + integer seconds',
        'receipt_age_limit_s': 1, 'source_age_limit_s': 5,
        'interpretation': 'Engineering availability only. Grid points are not actions. '
                          'No outcomes, lead-lag, null, confidence interval, or trading value evaluated.'})
    slugs = {f"btc-updown-5m-{w['start_s']}" for w in windows}
    meta_path = args.pm_root / 'markets.jsonl'
    metadata_issues = Counter()
    meta = metadata(meta_path, slugs, cutoff, metadata_issues)
    by_id = {w['market_id']: w for w in windows}
    pred = {mid: [] for mid in by_id}
    inputs = {str(snapshot_path): digest(snapshot_path), str(audit_path): digest(audit_path)}
    # A canonical subset identifies metadata even while the source ledger grows.
    write_json(args.out / 'pm_metadata.json', meta)
    inputs[str(args.out / 'pm_metadata.json')] = digest(args.out / 'pm_metadata.json')
    hours = set()
    for w in windows:
        for t in (w['start_s'] - 300, w['start_s'] + 300):
            hours.add(datetime.fromtimestamp(t, timezone.utc).strftime('%Y%m%d_%H'))
        hours.add(datetime.fromtimestamp(w['start_s'], timezone.utc).strftime('%Y%m%d_%H'))
    for path in sorted((run / 'predict_ws').glob('*.jsonl.gz')):
        if path.name.split('.')[0] not in hours:
            continue
        print('Predict', path.name, flush=True)
        before = digest(path)
        with gzip.open(path, 'rt') as handle:
            for line in handle:
                row = json.loads(line)
                if row['seq'] >= limit:
                    break
                if row['run_id'] != audit['run_id']:
                    raise ValueError('WS_RUN_IDENTITY')
                frame = json.loads(row['raw'])
                topic = frame.get('topic', '')
                if not topic.startswith('predictOrderbook/'):
                    continue
                mid = int(topic.split('/')[1])
                if mid not in by_id:
                    continue
                data = frame['data']
                if data['marketId'] != mid:
                    raise ValueError('WS_MARKET_IDENTITY')
                if row['recv_ns'] >= (by_id[mid]['start_s'] + 300) * NS:
                    continue
                bids, asks = data['bids'], data['asks']
                append_state(pred[mid], row['recv_ns'], data['updateTimestampMs'],
                             float(bids[0][0]) if bids else None, float(asks[0][0]) if asks else None)
        if digest(path) != before:
            raise ValueError('INPUT_CHANGED_DURING_READ')
        inputs[str(path)] = before
    rows = []
    for w in windows:
        slug = f"btc-updown-5m-{w['start_s']}"
        date = datetime.fromtimestamp(w['start_s'], timezone.utc).strftime('%Y%m%d')
        path = args.pm_root / 'raw' / date / (slug + '.jsonl.gz')
        pm = []
        metadata_status = 'KNOWN_BEFORE_OPEN'
        if slug not in meta:
            metadata_status = 'MISSING_PM_METADATA'
        elif meta[slug]['recv_ns'] > w['start_s'] * NS:
            metadata_status = 'METADATA_LEARNED_AFTER_OPEN'
        elif meta[slug]['window_start'] != w['start_s'] or meta[slug]['window_end'] != w['start_s'] + 300:
            raise ValueError('PM_WINDOW_IDENTITY')
        if not path.exists():
            metadata_status = 'MISSING_PM_FILE'
        elif metadata_status == 'KNOWN_BEFORE_OPEN':
            print('PM', slug, flush=True)
            before = digest(path)
            with gzip.open(path, 'rt') as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    stamp, raw = line.split('\t', 1)
                    recv = int(stamp)
                    if recv >= (w['start_s'] + 300) * NS:
                        continue
                    for source, bid, ask in pm_updates(json.loads(raw), meta[slug]['_up_token']):
                        append_state(pm, recv, source, bid, ask)
            if digest(path) != before:
                raise ValueError('INPUT_CHANGED_DURING_READ')
            inputs[str(path)] = before
        states = {'pm': pm, 'predict': pred[w['market_id']]}
        times = {key: [x[0] for x in value] for key, value in states.items()}
        statuses = {key: Counter() for key in states}
        joint = Counter()
        phases = {'FIRST_240_SECONDS': Counter(), 'FINAL_60_SECONDS': Counter()}
        for i in range(300):
            when = w['start_s'] * NS + i * NS + NS // 2
            values = {key: available(value, times[key], when, NS) for key, value in states.items()}
            if metadata_status != 'KNOWN_BEFORE_OPEN':
                values['pm'] = metadata_status
            for key, value in values.items():
                statuses[key][value] += 1
            status = ('BOTH_AVAILABLE' if all(v == 'AVAILABLE' for v in values.values())
                      else 'PM=' + values['pm'] + '|PREDICT=' + values['predict'])
            joint[status] += 1
            phases['FIRST_240_SECONDS' if i < 240 else 'FINAL_60_SECONDS'][status] += 1
        rows.append({**w, 'slug': slug, 'pm_metadata_status': metadata_status,
                     'n_pm_up_bbo_observations': len(pm), 'n_predict_book_observations': len(states['predict']),
                     'n_grid_points': 300, 'statuses': {k: dict(v) for k, v in statuses.items()},
                     'joint_statuses': dict(joint), 'phase_joint_statuses': {k: dict(v) for k, v in phases.items()}})
    total = Counter()
    for row in rows:
        total.update(row['joint_statuses'])
    phase_total = {key: Counter() for key in phases}
    for row in rows:
        for key, counts in row['phase_joint_statuses'].items():
            phase_total[key].update(counts)
    if sum(total.values()) != 300 * len(rows):
        raise ValueError('POPULATION_RECONCILIATION')
    write_json(args.out / 'receipt.json', {
        'purpose': 'ENGINEERING_AVAILABILITY_ONLY', 'signal_value_evaluated': False,
        'data_quality_certified': False, 'as_of_utc': datetime.now(timezone.utc).isoformat(),
        'input_cutoff_ns': cutoff, 'parent_cohort_n': len(audit['complete_window_coverage']),
        'n_windows': len(rows), 'n_grid_points': sum(total.values()),
        'source_sha256': digest(Path(__file__)), 'design_sha256': digest(args.out / 'design.json'),
        'supersedes': {'receipt': str(args.supersedes),
                       'reason': 'Handle PM empty-side 0/1 BBO sentinels explicitly; add first-240s/final-60s availability counts.'} if args.supersedes else None,
        'metadata_ledger_unscoped_records_at_read': dict(metadata_issues),
        'joint_statuses': dict(total), 'phase_joint_statuses': {k: dict(v) for k, v in phase_total.items()},
        'windows': rows, 'input_sha256': inputs,
        'limits': ['BBO presence does not certify depth, queue, fills, or full-day continuity.',
                   'Known-gap stress windows are selected for engineering review, not unbiased inference.',
                   'No settlement labels or trading outcomes read. No lead-lag or alpha result.',
                   'Prior audit owns local sequence coverage; this check does not recertify the full tape.',
                   'The one-second diagnostic grid does not measure sub-second opportunity or lead time.']})
    print(json.dumps({'n_windows': len(rows), 'n_grid_points': sum(total.values()), 'joint_statuses': dict(total)}))


if __name__ == '__main__':
    main()
