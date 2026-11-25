"""VENUE-MATCHED COMPARISON stage 1 (declaration.md): the FLOOR v1 harness with the take reaching the venue 208 ms after the decision
(58 ms order-live + 150 ms speed bump), BASE and FLOOR on the 408 complete v3 markets, settled with canonical ties, against the same
markets at 58 ms (the saved ledgers re-valued with ties).
Usage: PF_SLOT=claude predictfun/research/run_light.sh <tag> -- python3 predictfun/reviews/take_exit_venue_20261125/stage1.py selftest|run|report
"""
import collections
import gzip
import json
import resource
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

H = Path(__file__).resolve().parent
R = H.parents[2]
sys.path.insert(0, str(H))
import venue_common as V  # noqa: E402

F, OD, MM, TR, C = V.F, V.OD, V.MM, V.TR, V.C
OUT = F.X.DATA / 'take_exit_venue_s1_20261125'
S1 = MM.S1
DELAY = 208.
CPU_CAP = 1400.
FILES = ['declaration.md', 'venue_common.py', 'stage1.py']
REVIEW_EVIDENCE = H.parent / 'automatic/20261125_take_floor/evidence.log'   # review 4e67e0856: its independent recount (settlement_usd, ties corrected)


def review_recount():
    t = REVIEW_EVIDENCE.read_text()
    j = json.loads(t[t.index('{'):t.rindex('}') + 1])
    return {arm: float(j['arms'][arm.lower()]['settlement_usd']) for arm in V.ARMS}


def bound():
    F.bound()
    for f in FILES:
        C.require(subprocess.check_output(['git', 'show', 'HEAD:' + str((H / f).relative_to(R))], cwd=R) == (H / f).read_bytes(), f'{f} is not committed')


def run():
    resource.setrlimit(resource.RLIMIT_CPU, (int(CPU_CAP) + 60, int(CPU_CAP) + 61))
    bound()
    C.require(not (OUT / 'summary.json').exists(), 'summary.json exists (write-once)')
    OUT.mkdir(parents=True, exist_ok=True)
    v1 = json.loads((F.OUT / 'summary.json').read_text())
    complete = {(r['day'], int(r['mid'])): r for r in v1['rows']}
    v3 = {(r['day'], int(r['mid'])): r for r in json.loads((OD.OUT3 / 'summary.json').read_text())['rows'] if r['setting'] == 'STRICT58'}
    edges, src = TR.T.P.frozen_edges(), MM.KP.Tape()
    rows, t0 = [], time.process_time()
    for di, day in enumerate(OD.DAYS3):
        reg = MM.registry(day).set_index('mid')
        rg = pd.read_parquet(S1 / day / 'registry.parquet').set_index('mid')
        off = src.offset(day)
        for h, g in reg.groupby('hour'):
            mids = [int(m) for m in g.index if (day, int(m)) in complete]
            if not mids:
                continue
            for m, st, p, info, yup in F.worlds_yup(day, mids, src, off, reg):
                C.require(st == 'complete', f'{m} {st}')
                y = V.payout_of(rg.loc[m])
                C.require(yup == float(rg.loc[m, 'outcome'] == 'Up'), 'loader outcome differs from the registry')
                ws = float(reg.loc[m, 'ws']) * 1000.
                cache = OD.make_cache3(float(info['t_start']), float(info['end_rel']), OD.SEG_OF[day])
                old = dict(end=float(info['end_rel']), ident=MM.ident_of(day, m, ws, info, p))
                row = dict(day=day, mid=m, hour=int(h), y=y, tie=y == .5)
                refs = dict(BASE=(OD.OUT3 / f'{m}_STRICT58.json.gz', v3[(day, m)]['sha256']), FLOOR=(F.OUT / f'{m}_FLOOR.json.gz', complete[(day, m)]['sha256']))
                for arm, slack in V.ARMS.items():
                    rec = V.one_delay(p, cache, old, edges, 200., slack, DELAY)
                    rec.update(day=day, mid=m, hour=int(h), arm=arm, payout=y)
                    rec = json.loads(json.dumps(TR.clean(rec), allow_nan=False))
                    row[arm + '_sha256'] = F.write_once_gz(OUT / f'{m}_{arm}.json.gz', rec)
                    row[arm] = V.settle_account(rec, y)
                    row[arm + '_ioc'] = dict(rec['metrics']['ioc_status'], sent=rec['metrics']['takes_sent'], filled=rec['metrics']['takes_filled'])
                    fn, sha = refs[arm]
                    C.require(F.sha_file(fn) == sha, f'{fn.name} changed')
                    row[arm + '_58'] = V.settle_account(json.loads(gzip.decompress(fn.read_bytes())), y)
                rows.append(row)
                C.require(F.rss_mib() < F.RSS_GUARD_MIB, 'RSS over the self-guard')
        spent = time.process_time() - t0
        print(f'{day}: {sum(r["day"] == day for r in rows)} markets, cpu {spent:.0f} s', flush=True)
        if di == 0:
            proj = spent * len(OD.DAYS3) * 1.1
            fn = H / 'stage1_projection.json'
            C.require(not fn.exists(), 'stage1_projection.json exists (write-once)')
            TR.dump(fn, dict(asof='2026-11-25', spent_cpu_s=spent, projected_cpu_s=proj, cap_cpu_s=CPU_CAP, go=proj <= CPU_CAP))
            if proj > CPU_CAP:
                sys.exit(f'REFUSED: projected {proj:.0f} CPU s > cap {CPU_CAP:.0f}')
    C.require(len(rows) == len(complete), f'{len(rows)} markets run, {len(complete)} expected')
    TR.dump(OUT / 'summary.json', dict(asof='2026-11-25', delay_ms=DELAY, rows=rows, cpu_s=time.process_time() - t0, peak_rss_mib=F.rss_mib()))
    print('stage 1 done', len(rows))


def report():
    sm = json.loads((OUT / 'summary.json').read_text())
    rows = sm['rows']
    tot = lambda arm, k: float(sum(r[arm][k] for r in rows)) / 100
    out = dict(asof='2026-11-25', markets=len(rows), delay_ms=sm['delay_ms'], ties=sum(r['tie'] for r in rows), arms={})
    keys = ('settle_net_c', 'rt_net_c', 'rt_exit_net_c', 'rt_take_net_c', 'held_settle_net_c', 'fees_c', 'mark_net_c')
    for arm in V.ARMS:
        day = collections.defaultdict(lambda: [0., 0.])
        for r in rows:
            day[r['day']][0] += r[arm]['settle_net_c'] / 100; day[r['day']][1] += r[arm + '_58']['settle_net_c'] / 100
        ioc = collections.Counter()
        for r in rows:
            ioc.update(r[arm + '_ioc'])
        out['arms'][arm] = dict(
            at_208={k.replace('_c', '_usd'): tot(arm, k) for k in keys}, at_58={k.replace('_c', '_usd'): tot(arm + '_58', k) for k in keys},
            shares_208=dict(taken=float(sum(r[arm]['taker_shares'] for r in rows)), held=float(sum(r[arm]['held_shares'] for r in rows))),
            shares_58=dict(taken=float(sum(r[arm + '_58']['taker_shares'] for r in rows)), held=float(sum(r[arm + '_58']['held_shares'] for r in rows))),
            by_day_usd={d: dict(at_208=v[0], at_58=v[1], speed_bump_cost=v[0] - v[1]) for d, v in sorted(day.items())},
            days_positive_208=int(sum(v[0] > 0 for v in day.values())), ioc_208=dict(ioc))
    pd_ = collections.defaultdict(float)
    for r in rows:
        pd_[r['day']] += (r['FLOOR']['settle_net_c'] - r['BASE']['settle_net_c']) / 100
    out['floor_minus_base_208'] = dict(total_usd=sum(pd_.values()), by_day_usd=dict(sorted(pd_.items())), days_positive=int(sum(v > 0 for v in pd_.values())))
    out['checks'] = dict(
        ties_8=out['ties'] == 8,
        reviewer_recount_reproduced={arm: abs(out['arms'][arm]['at_58']['settle_net_usd'] - rc[arm]) < 1e-6 for arm, rc in [(a, review_recount()) for a in V.ARMS]})
    out['readings'] = {arm: dict(positive_at_208=out['arms'][arm]['at_208']['settle_net_usd'] > 0, days_positive_208=out['arms'][arm]['days_positive_208'],
                                 speed_bump_cost_usd=out['arms'][arm]['at_208']['settle_net_usd'] - out['arms'][arm]['at_58']['settle_net_usd'])
                       for arm in V.ARMS}
    fn = H / 'stage1_readings.json'
    C.require(not fn.exists(), 'stage1_readings.json exists (write-once)')
    TR.dump(fn, out)
    print(json.dumps(dict(checks=out['checks'], readings=out['readings'], floor_minus_base_208=out['floor_minus_base_208']), indent=1))


def selftest():
    up = dict(up_index=1, outcome='Up', start_price=1., end_price=2.)
    assert V.payout_of(up) == 1. and V.payout_of(dict(up, outcome='Down')) == 0. and V.payout_of(dict(up, end_price=1.)) == .5
    rec = dict(fills=[[0., 0, 0, 50., 52., -1], [500., 1, 1, 46., 46., 0], [900., 2, 1, 10., 60., 0]], ioc_results=[dict(gid=0, filled=50., fee_c=48.)],
               mark=98., fees_c=48.)
    cash, pos = -50 * 52. + 46 * 46. + 10 * 60., -6.
    rec.update(cash=cash, position=pos, net_value_c=cash + pos * 98. - 48.)
    tie = V.settle_account(rec, .5)
    assert abs(tie['held_settle_net_c'] - (-6 * (50 - 60.))) < 1e-9 and tie['held_won_shares'] == 0.
    assert abs(tie['settle_net_c'] - (cash + pos * 50. - 48.)) < 1e-9
    for bad in (.3, 2.):
        try:
            V.settle_account(rec, bad)
        except ValueError:
            continue
        raise AssertionError(f'payout {bad} accepted')
    try:
        V.payout_of(dict(up, outcome='Unknown'))
    except ValueError:
        pass
    else:
        raise AssertionError('unknown outcome accepted')
    print('stage1 selftest PASS')


if __name__ == '__main__':
    dict(selftest=selftest, run=run, report=report)[sys.argv[1]]()
