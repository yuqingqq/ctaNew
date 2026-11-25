"""TAKE-REPRICE v1 stage L (declaration.md): the frozen repricing rule at a shorter network path (2 / 12 ms), same markets and harness.
Usage: PF_SLOT=<slot> predictfun/research/run_light.sh <tag> -- python3 predictfun/reviews/take_reprice_20261125/reprice_latency.py day <day> | report
"""
import collections
import json
import resource
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

H = Path(__file__).resolve().parent
R = H.parents[2]
sys.path.insert(0, str(H))
import reprice_replay as RP  # noqa: E402  (frozen at 2a5aa46fc)

S, VR, V, F, OD, MM, TR, C = RP.S, RP.VR, RP.V, RP.F, RP.OD, RP.MM, RP.TR, RP.C
OUT = S.X.DATA / 'take_reprice_latency_20261125'
CONFIGS = {'REPRICE-8-152': (8., 2.), 'REPRICE-16-152': (16., 2.), 'REPRICE-8-162': (8., 12.), 'REPRICE-16-162': (16., 12.)}
DAY_CPU_CAP, TOTAL_CPU_CAP, FIRST_DAY = 700., 2000., '20261023'
FILES = ['declaration.md', 'reprice_replay.py', 'reprice_latency.py']


def bound():
    for f in FILES:
        C.require(subprocess.check_output(['git', 'show', 'HEAD:' + str((H / f).relative_to(R))], cwd=R) == (H / f).read_bytes(), f'{f} is not committed')
    S.bound()


def day_run(day):
    resource.setrlimit(resource.RLIMIT_CPU, (int(DAY_CPU_CAP) + 60, int(DAY_CPU_CAP) + 61))
    bound()
    fn = OUT / f'day_{day}.json'
    C.require(not fn.exists(), f'{fn.name} exists (write-once)')
    if day != FIRST_DAY:
        pj = json.loads((H / 'latency_projection.json').read_text())
        C.require(pj['projected_cpu_s'] <= TOTAL_CPU_CAP, 'the first-day projection exceeds the cap')
    OUT.mkdir(parents=True, exist_ok=True)
    vs = json.loads((VR.OUT / 'summary.json').read_text())
    comp = {int(r['mid']): r['payout'] for r in vs['rows'] if r['day'] == day and r['status'] == 'complete'}
    bk, bm = OD.seg_arrays(OD.SEG_OF[day]); bk = bk.astype(float); bm = bm.astype(float)
    cs = np.cumsum(np.r_[0., np.diff(np.log(bm))] ** 2)
    edges, src = TR.T.P.frozen_edges(), MM.KP.Tape()
    reg = MM.registry(day).set_index('mid'); off = src.offset(day); t0c = time.process_time(); rows = []
    for h, g in reg.groupby('hour'):
        mids = [int(m) for m in g.index if int(m) in comp]
        if not mids:
            continue
        for m, W, lst in VR.worlds_all(day, mids, src, off, reg):
            ws = float(reg.loc[m, 'ws']) * 1000.
            p, st, info = VR.vm_path(W, ws, 50.)
            C.require(st == 'complete', f'{m} {st}')
            t0, end = float(info['t_start']), float(info['end_rel'])
            cache = OD.make_cache3(t0, end, OD.SEG_OF[day]); old = dict(end=end, ident=MM.ident_of(day, m, ws, info, p)); y = comp[m]
            row = dict(day=day, mid=m, payout=y, arms={}); trig = {}
            for D in sorted({c[0] for c in CONFIGS.values()}):
                tr = [(x + S.FEED + S.DECIDE - t0, dd) for x, dd in RP.reprice_triggers(bk, bm, cs, ws, t0 - S.FEED - S.DECIDE, t0 + end, D)]
                tr = [(x, dd) for x, dd in tr if 0. <= x < end]
                trig[D] = (np.array([x for x, _ in tr], float), np.array([dd for _, dd in tr], int), np.array([f'reprice{int(D)}'] * len(tr)))
            for name, (D, net) in CONFIGS.items():
                rec = S.one_stream(p, cache, old, edges, 200., V.BASE_SLACK, trig[D], net)
                rec.update(day=day, mid=m, config=name)
                sha = F.write_once_gz(OUT / f'{m}_{name}.json.gz', rec)
                a = VR.native_account(rec, y)
                a.update(sha256=sha, ioc_status=rec['metrics']['ioc_status'], take_value=S.take_value(rec))
                row['arms'][name] = a
            rows.append(row)
    spent = time.process_time() - t0c
    if day == FIRST_DAY:
        pfn = H / 'latency_projection.json'
        C.require(not pfn.exists(), 'latency_projection.json exists (write-once)')
        TR.dump(pfn, dict(asof='2026-11-25', spent_cpu_s=spent, markets=len(rows), projected_cpu_s=spent * 408 / max(len(rows), 1) * 1.1, cap_cpu_s=TOTAL_CPU_CAP))
    TR.dump(fn, dict(asof='2026-11-25', day=day, rows=rows, cpu_s=spent))
    print(f'{day}: {len(rows)} markets, cpu {spent:.0f} s', flush=True)


def report():
    rows = [r for d in OD.DAYS3 for r in json.loads((OUT / f'day_{d}.json').read_text())['rows']]
    base = {(r['day'], int(r['mid'])): r['arms'] for d in OD.DAYS3 for r in json.loads((RP.OUT / f'day_{d}.json').read_text())['rows']}
    C.require(len(rows) == 408, 'population')
    out = dict(asof='2026-11-25', markets=len(rows), configs={})
    for name in list(CONFIGS) + ['REPRICE-8-172', 'REPRICE-16-172']:
        A = [r['arms'][name] if name in r['arms'] else base[(r['day'], int(r['mid']))][name] for r in rows]
        day = collections.defaultdict(float)
        for r, a in zip(rows, A):
            day[r['day']] += a['total_c'] / 100
        ioc, tv = collections.Counter(), collections.defaultdict(lambda: [0., 0.])
        for a in A:
            ioc.update(a['ioc_status'])
            for h, (v, q) in a['take_value'].items():
                tv[h][0] += v; tv[h][1] += q
        n = sum(ioc.values())
        out['configs'][name] = dict(total_usd=sum(a['total_c'] for a in A) / 100, lower_usd=sum(a['unresolved_lo_c'] for a in A) / 100,
                                    days_positive=int(sum(v > 0 for v in day.values())), by_day_usd=dict(sorted(day.items())),
                                    takes_sent=n, filled=ioc.get('filled', 0) + ioc.get('partial', 0), limit_moved_share=ioc.get('limit_moved', 0) / n if n else None,
                                    taker_shares=sum(a['taker_shares'] for a in A), exit_usd=sum(a['rt_exit_net_c'] for a in A) / 100, held_usd=sum(a['held_settle_net_c'] for a in A) / 100,
                                    take_value_c={h.replace('_c', ''): v / q if q > 0 else None for h, (v, q) in tv.items()})
    good = lambda n: bool(out['configs'][n]['lower_usd'] > 0 and out['configs'][n]['days_positive'] >= 4)
    out['readings'] = dict(pays_at_152=good('REPRICE-8-152') or good('REPRICE-16-152'), pays_at_162=good('REPRICE-8-162') or good('REPRICE-16-162'), multiplicity=4)
    fn = H / 'latency_readings.json'
    C.require(not fn.exists(), 'latency_readings.json exists (write-once)')
    TR.dump(fn, out)
    for n, b in out['configs'].items():
        print(f"{n:16s} total {b['total_usd']:+9.2f} (lower {b['lower_usd']:+9.2f}) days+ {b['days_positive']}  sent {b['takes_sent']:5d} filled {b['filled']:4d} "
              f"limit moved {b['limit_moved_share']:.0%}  value 1s/10s {b['take_value_c'].get('m1s') or 0:+.2f}/{b['take_value_c'].get('m10s') or 0:+.2f}  exits {b['exit_usd']:+8.2f} held {b['held_usd']:+8.2f}")
    print(json.dumps(out['readings']))


if __name__ == '__main__':
    day_run(sys.argv[2]) if sys.argv[1] == 'day' else report()
