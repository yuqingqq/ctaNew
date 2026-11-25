"""TAKE-STREAM v1: merge the day files, checks (K1 parity, K4 take timing, K5 venue identities re-derived), readings as predicates and
report.md (declaration.md). Every number comes from the day files, the ledgers and the trigger files.
Usage: PF_SLOT=claude predictfun/research/run_light.sh <tag> -- python3 predictfun/reviews/take_stream_20261125/stream_report.py merge|check|report
"""
import collections
import gzip
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

H = Path(__file__).resolve().parent
sys.path.insert(0, str(H))
import stream_replay as S  # noqa: E402
import vm_report as VMR  # noqa: E402  (sent_below_min: the minimum-order checker, falsifier-proven in the venue comparison)

VR, V, OD, TR, C = S.VR, S.V, S.OD, S.TR, S.C
OUT, CONFIGS, EPS = S.OUT, S.CONFIGS, S.EPS


def merge():
    fn = OUT / 'summary.json'
    C.require(not fn.exists(), 'summary.json exists (write-once)')
    days = [json.loads((OUT / f'day_{d}.json').read_text()) for d in OD.DAYS3]
    trig = {d: dict(collections.Counter(pd.read_parquet(OUT / f'triggers_{d}.parquet').kind)) for d in OD.DAYS3}
    TR.dump(fn, dict(asof='2026-11-25', rows=[r for d in days for r in d['rows']], parity=[p for d in days for p in d['parity']], triggers_per_day=trig,
                     cpu_s=sum(d['cpu_s'] for d in days), cpu_s_by_day={d['day']: d['cpu_s'] for d in days}, peak_rss_mib=max(d['peak_rss_mib'] for d in days)))
    print('merged', len(days))


def check():
    sm = json.loads((OUT / 'summary.json').read_text())
    res = {}

    def ok(name, cond, detail=''):
        res[name] = dict(ok=bool(cond), detail=detail)
        print(('PASS ' if cond else 'FAIL ') + name, detail)

    rows = sm['rows']
    ok('population', len(rows) == 408, f'{len(rows)} complete markets')
    ok('K1_grid_parity', len(sm['parity']) >= 10 and all(all(x['same'].values()) for x in sm['parity']), f"{len(sm['parity'])} markets")
    b = collections.Counter()
    for r in rows:
        for name, a in r['arms'].items():
            raw = (OUT / f"{r['mid']}_{name}.json.gz").read_bytes()
            C.require(hashlib.sha256(raw).hexdigest() == a['sha256'], 'a ledger changed')
            rec = json.loads(gzip.decompress(raw))
            a2 = VR.native_account(rec, r['payout'])
            b['account_mismatch'] += int(any(abs(a2[k] - a[k]) > 1e-6 for k in a2))
            net = CONFIGS[name]['net']
            for v in rec['iocs'].values():
                b['timing'] += int(abs(v['act'] - v['submit'] - net - S.HOLD) > 1e-9)
            b['below_min'] += VMR.sent_below_min(rec)
            b['bound'] += int(not (a['unresolved_lo_c'] - 1e-6 <= a['total_c'] <= a['unresolved_hi_c'] + 1e-6))
            b['ledgers'] += 1
    ok('K4_take_timing', b['timing'] == 0, f"{b['timing']} takes off decision + network + 150 ms")
    ok('K5_accounting_rederived', b['ledgers'] == len(rows) * len(CONFIGS) and b['account_mismatch'] == 0, f"{b['ledgers']} ledgers, {b['account_mismatch']} mismatches")
    ok('K5_bounds', b['bound'] == 0, f"{b['bound']} rows outside their bounds")
    ok('K5_min_buy', b['below_min'] == 0, f"{b['below_min']} sent takes with a buy part below 1 USDT")
    try:
        S.selftest(); VR.selftest(); st = True
    except AssertionError:
        st = False
    ok('K2_K3_selftests', st, 'stream / depletion triggers, the venue harness')
    res['all_pass'] = all(v['ok'] for v in res.values() if isinstance(v, dict))
    fn = H / 'checks.json'
    C.require(not fn.exists(), 'checks.json exists (write-once)')
    TR.dump(fn, res)
    print('ALL PASS' if res['all_pass'] else 'SOME CHECK FAILED')
    sys.exit(0 if res['all_pass'] else 1)


def block(rows, name):
    A = [r['arms'][name] for r in rows]
    s = lambda k: float(sum(a[k] for a in A)) / 100
    day = collections.defaultdict(float)
    for r in rows:
        day[r['day']] += r['arms'][name]['total_c'] / 100
    ioc, sent = collections.Counter(), collections.Counter()
    tv = collections.defaultdict(lambda: [0., 0.])
    for a in A:
        ioc.update(a['ioc_status']); sent.update(a['sig_kind_sent'])
        for h, (v, q) in a['take_value'].items():
            tv[h][0] += v; tv[h][1] += q
    n_ioc = sum(ioc.values())
    return dict(total_usd=s('total_c'), lower_usd=s('unresolved_lo_c'), upper_usd=s('unresolved_hi_c'), by_day_usd=dict(sorted(day.items())),
                days_positive=int(sum(v > 0 for v in day.values())), round_trips_usd=s('rt_net_c'), exit_usd=s('rt_exit_net_c'), reverse_take_usd=s('rt_take_net_c'),
                held_usd=s('held_settle_net_c'), taker_shares=float(sum(a['taker_shares'] for a in A)), takes_sent_by_kind=dict(sent), ioc_status=dict(ioc),
                limit_moved_share=ioc.get('limit_moved', 0) / n_ioc if n_ioc else None,
                take_value_after_fee_c_per_share={h.replace('_c', ''): (v / q if q > 0 else None) for h, (v, q) in tv.items()})


def readings():
    sm = json.loads((OUT / 'summary.json').read_text())
    ck = json.loads((H / 'checks.json').read_text())
    C.require(ck['all_pass'], 'checks did not pass')
    rows = sm['rows']
    out = dict(asof='2026-11-25', markets=len(rows), triggers_per_day=sm['triggers_per_day'], configs={n: block(rows, n) for n in CONFIGS})

    def diff_days(a, b):
        d = collections.defaultdict(float)
        for r in rows:
            d[r['day']] += (r['arms'][a]['total_c'] - r['arms'][b]['total_c']) / 100
        return dict(total_usd=sum(d.values()), by_day_usd=dict(sorted(d.items())), days_positive=int(sum(v > 0 for v in d.values())))
    out['differences'] = {'STREAM-172 - GRID-172': diff_days('STREAM-172', 'GRID-172'), 'STREAM+DEPL-172 - STREAM-172': diff_days('STREAM+DEPL-172', 'STREAM-172'),
                          'STREAM+DEPLS-172 - STREAM-172': diff_days('STREAM+DEPLS-172', 'STREAM-172')}
    good = lambda n: bool(out['configs'][n]['lower_usd'] > 0 and out['configs'][n]['days_positive'] >= 4)
    best = max(('STREAM-172', 'STREAM+DEPL-172', 'STREAM+DEPLS-172'), key=lambda n: out['configs'][n]['total_usd'])
    at192 = {'STREAM-172': 'STREAM-192', 'STREAM+DEPLS-172': 'STREAM+DEPLS-192'}.get(best)
    out['readings'] = dict(stream_beats_grid=out['differences']['STREAM-172 - GRID-172']['days_positive'] >= 4,
                           depletion_adds=out['differences']['STREAM+DEPL-172 - STREAM-172']['days_positive'] >= 4,
                           strict_depletion_adds=out['differences']['STREAM+DEPLS-172 - STREAM-172']['days_positive'] >= 4,
                           best_trigger_set_172=best, profitable_172=good(best), multiplicity=3,
                           holds_at_192=(good(at192) if at192 else 'not_run'))
    fn = H / 'readings.json'
    C.require(not fn.exists(), 'readings.json exists (write-once)')
    md = render(out, ck)
    TR.dump(fn, out)
    (H / 'report.md').write_text(md)
    print(json.dumps(out['readings']))


def usd(x):
    return '—' if x is None else f'{x:+,.2f}'


def render(out, ck):
    L = ['# TAKE-STREAM v1 — stream-decided takes and a book-shrinking trigger under venue rules (2026-11-25)', '',
         f"[declaration.md](declaration.md) (frozen before any run). {out['markets']} complete markets on the five consumed v3 days; development evidence. "
         'Every config uses the venue-matched harness (150 ms hold, 1 USDT minimum, fees in shares, queue L, exits 58 / 37 / 95 ms, settlement).', '',
         '| Config | Total at settlement | lower / upper | days + | maker exits | reverse takes | held | taker shares | limit moved | take value 1 s / 10 s (¢/sh) |',
         '|---|---:|---:|---:|---:|---:|---:|---:|---:|---|']
    for n, b in out['configs'].items():
        tv = b['take_value_after_fee_c_per_share']
        L.append(f"| {n} | {usd(b['total_usd'])} | {usd(b['lower_usd'])} / {usd(b['upper_usd'])} | {b['days_positive']} / 5 | {usd(b['exit_usd'])} | {usd(b['reverse_take_usd'])} | "
                 f"{usd(b['held_usd'])} | {b['taker_shares']:,.0f} | {b['limit_moved_share']:.0%} | {tv.get('m1s', 0) or 0:+.2f} / {tv.get('m10s', 0) or 0:+.2f} |")
    L += ['', '## Differences (by day)', '']
    for k, d in out['differences'].items():
        L.append(f"- {k}: {usd(d['total_usd'])}, positive on {d['days_positive']} / 5 days ({', '.join(f'{x} {usd(v)}' for x, v in d['by_day_usd'].items())})")
    L += ['', '## Triggers per day', ''] + [f'- {d}: {c}' for d, c in out['triggers_per_day'].items()]
    L += ['', '## Readings (computed)', ''] + [f'- `{k}`: {v}' for k, v in out['readings'].items()]
    L += ['', '## Checks', ''] + [f"- {'PASS' if v['ok'] else 'FAIL'} `{k}`: {v['detail']}" for k, v in ck.items() if isinstance(v, dict)]
    L += ['', '## Limits', '', '- Consumed days; complete markets only (stage A). Our order is not in the recorded book: competitors and makers react to it in reality.',
          '- Decisions are causal (state at the tick before the decision, book known at the decision); the stream rule ignores window rolls between messages.',
          '- The depletion rules were not tuned on outcomes; the strict floor comes from a day outside the test (sizes only).']
    return '\n'.join(L) + '\n'


if __name__ == '__main__':
    dict(merge=merge, check=check, report=readings)[sys.argv[1]]()
