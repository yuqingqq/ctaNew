"""TAKE-REPRICE v1: merge, checks, readings as predicates and report.md (declaration.md). Comparator STREAM-172 from the
take_stream_20261125 day files (same markets, same branch and harness).
Usage: PF_SLOT=claude predictfun/research/run_light.sh <tag> -- python3 predictfun/reviews/take_reprice_20261125/reprice_report.py check|report
"""
import collections
import gzip
import hashlib
import json
import sys
from pathlib import Path

H = Path(__file__).resolve().parent
sys.path.insert(0, str(H))
sys.path.insert(0, str(H.parent / 'take_exit_venue_20261125'))
import reprice_replay as RP  # noqa: E402
import vm_report as VMR  # noqa: E402

S, VR, OD, TR, C = RP.S, RP.VR, RP.OD, RP.TR, RP.C
OUT, CONFIGS = RP.OUT, RP.CONFIGS


def rows_all():
    rp = [r for d in OD.DAYS3 for r in json.loads((OUT / f'day_{d}.json').read_text())['rows']]
    st = {(r['day'], int(r['mid'])): r['arms']['STREAM-172'] for d in OD.DAYS3 for r in json.loads((S.OUT / f'day_{d}.json').read_text())['rows']}
    for r in rp:
        r['arms']['STREAM-172'] = st[(r['day'], int(r['mid']))]
    return rp


def check():
    rows = rows_all(); res = {}

    def ok(name, cond, detail=''):
        res[name] = dict(ok=bool(cond), detail=detail); print(('PASS ' if cond else 'FAIL ') + name, detail)
    ok('population', len(rows) == 408, f'{len(rows)} complete markets')
    b = collections.Counter()
    for r in rows:
        for name, (D, net) in CONFIGS.items():
            a = r['arms'][name]
            raw = (OUT / f"{r['mid']}_{name}.json.gz").read_bytes()
            C.require(hashlib.sha256(raw).hexdigest() == a['sha256'], 'a ledger changed')
            rec = json.loads(gzip.decompress(raw))
            a2 = VR.native_account(rec, r['payout'])
            b['mismatch'] += int(any(abs(a2[k] - a[k]) > 1e-6 for k in a2))
            b['timing'] += sum(abs(v['act'] - v['submit'] - net - S.HOLD) > 1e-9 for v in rec['iocs'].values())
            b['below_min'] += VMR.sent_below_min(rec)
            b['bound'] += int(not (a['unresolved_lo_c'] - 1e-6 <= a['total_c'] <= a['unresolved_hi_c'] + 1e-6))
            b['ledgers'] += 1
    ok('take_timing', b['timing'] == 0, f"{b['timing']} takes off decision + network + 150 ms")
    ok('accounting_rederived', b['ledgers'] == 408 * len(CONFIGS) and b['mismatch'] == 0, f"{b['ledgers']} ledgers, {b['mismatch']} mismatches")
    ok('min_buy', b['below_min'] == 0, f"{b['below_min']} sent takes below 1 USDT")
    ok('bounds', b['bound'] == 0, f"{b['bound']} rows outside their bounds")
    try:
        RP.selftest(); st = True
    except AssertionError:
        st = False
    ok('selftest', st, 'dFV trigger: near-strike fires, far does not, quiet suppresses')
    res['all_pass'] = all(v['ok'] for v in res.values() if isinstance(v, dict))
    fn = H / 'checks.json'; C.require(not fn.exists(), 'checks.json exists (write-once)'); TR.dump(fn, res)
    print('ALL PASS' if res['all_pass'] else 'SOME CHECK FAILED'); sys.exit(0 if res['all_pass'] else 1)


def block(rows, name):
    A = [r['arms'][name] for r in rows]; s = lambda k: float(sum(a[k] for a in A)) / 100
    day = collections.defaultdict(float)
    for r in rows:
        day[r['day']] += r['arms'][name]['total_c'] / 100
    ioc, tv = collections.Counter(), collections.defaultdict(lambda: [0., 0.])
    for a in A:
        ioc.update(a['ioc_status'])
        for h, (v, q) in a['take_value'].items():
            tv[h][0] += v; tv[h][1] += q
    n = sum(ioc.values())
    return dict(total_usd=s('total_c'), lower_usd=s('unresolved_lo_c'), upper_usd=s('unresolved_hi_c'), by_day_usd=dict(sorted(day.items())),
                days_positive=int(sum(v > 0 for v in day.values())), exit_usd=s('rt_exit_net_c'), reverse_usd=s('rt_take_net_c'), held_usd=s('held_settle_net_c'),
                taker_shares=float(sum(a['taker_shares'] for a in A)), takes_sent=n, filled=ioc.get('filled', 0) + ioc.get('partial', 0),
                limit_moved_share=ioc.get('limit_moved', 0) / n if n else None,
                take_value_c=dict((h.replace('_c', ''), v / q if q > 0 else None) for h, (v, q) in tv.items()))


def report():
    ck = json.loads((H / 'checks.json').read_text()); C.require(ck['all_pass'], 'checks did not pass')
    rows = rows_all()
    names = list(CONFIGS) + ['STREAM-172']
    out = dict(asof='2026-11-25', markets=len(rows), triggers=dict((D, sum(r['n_triggers'][D] for r in rows)) for D in rows[0]['n_triggers']), configs={n: block(rows, n) for n in names})

    def diff(a, b):
        d = collections.defaultdict(float)
        for r in rows:
            d[r['day']] += (r['arms'][a]['total_c'] - r['arms'][b]['total_c']) / 100
        return dict(total_usd=sum(d.values()), by_day_usd=dict(sorted(d.items())), days_positive=int(sum(v > 0 for v in d.values())))
    out['vs_stream'] = {n: diff(n, 'STREAM-172') for n in CONFIGS}
    good = lambda n: bool(out['configs'][n]['lower_usd'] > 0 and out['configs'][n]['days_positive'] >= 4)
    best = max(('REPRICE-4-172', 'REPRICE-8-172', 'REPRICE-16-172'), key=lambda n: out['configs'][n]['total_usd'])
    out['readings'] = dict(**{f'beats_stream_{n.split("-")[1]}': out['vs_stream'][n]['days_positive'] >= 4 for n in ('REPRICE-4-172', 'REPRICE-8-172', 'REPRICE-16-172')},
                           **{f'profitable_{n.split("-")[1]}': good(n) for n in ('REPRICE-4-172', 'REPRICE-8-172', 'REPRICE-16-172')},
                           best=best, multiplicity=3, holds_at_192=good('REPRICE-8-192'))
    L = ['# TAKE-REPRICE v1 — takes only on large move-implied repricings (2026-11-25)', '',
         f"[declaration.md](declaration.md) (frozen before any run). {out['markets']} complete markets, five consumed days; thresholds informed by the winners on these days "
         '(optimistic by construction). Venue-matched harness; comparator STREAM-172.', '',
         '| Config | Total | lower / upper | days + | takes sent | filled | limit moved | taker sh | exits | reverse | held | take value 1 s / 10 s |', '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|']
    u = lambda x: '—' if x is None else f'{x:+,.2f}'
    for n in names:
        b = out['configs'][n]; tv = b['take_value_c']
        L.append(f"| {n} | {u(b['total_usd'])} | {u(b['lower_usd'])} / {u(b['upper_usd'])} | {b['days_positive']} / 5 | {b['takes_sent']:,} | {b['filled']:,} | "
                 f"{b['limit_moved_share']:.0%} | {b['taker_shares']:,.0f} | {u(b['exit_usd'])} | {u(b['reverse_usd'])} | {u(b['held_usd'])} | "
                 f"{(tv.get('m1s') or 0):+.2f} / {(tv.get('m10s') or 0):+.2f} |")
    L += ['', '## vs STREAM-172 (by day)', ''] + [f"- {n}: {u(d['total_usd'])}, positive on {d['days_positive']} / 5 days" for n, d in out['vs_stream'].items()]
    L += ['', f"Triggers (all markets): {out['triggers']}", '', '## Readings (computed)', ''] + [f'- `{k}`: {v}' for k, v in out['readings'].items()]
    L += ['', '## Checks', ''] + [f"- {'PASS' if v['ok'] else 'FAIL'} `{k}`: {v['detail']}" for k, v in ck.items() if isinstance(v, dict)]
    fn = H / 'readings.json'; C.require(not fn.exists(), 'readings.json exists (write-once)')
    TR.dump(fn, out); (H / 'report.md').write_text('\n'.join(L) + '\n')
    print(json.dumps(out['readings']))


if __name__ == '__main__':
    dict(check=check, report=report)[sys.argv[1]]()
