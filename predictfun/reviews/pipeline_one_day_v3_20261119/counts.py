"""Every number the v3 (step 0 / G0) report quotes, computed from the saved outputs (declared before they were read; no number typed by hand):
the benchmark v3 gate lines, the keep / pull rule (policy v3.1) per day and whether its values equal v2's in every cell, and the corrected attribution
of MODEL-J - PULL-ALL at B by the rule's action (review 1adf6a33 P2): pulled at the first live step / pulled later / never pulled with RETAIN
predictions / no RETAIN prediction, per side, from the keep / pull run's own M50 fills, the rule's pull steps and the structural ends - its totals
must equal the committed policy v3.1 values (a reconciliation, printed as a predicate). No sum over scenarios is printed as an expected return.
Usage (run_light): counts.py [research data dir]"""
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'research'))
import pf_keep_pull as KP                                 # noqa: E402
import pf_keep_pull_scenarios as SC                       # noqa: E402
R = sys.argv[1] if len(sys.argv) > 1 else 'data/predict_fun/research'; B = 'M50|A58|L37|N100'
bj = json.loads(open(f'{R}/pipeline_bench_v3_20261119/bench.json').read())
print(f"BENCH v3: economic gate {bj['gate']}, rank / quantity diagnostic {bj['diagnostic']}")
for kind in ('retain', 'post'):
    for s in ('0', '1'):
        for d in ('20261026', '20261027'):
            e = bj['economic'][kind][s][d]
            print(f"  {kind} side {s} {d}: {e['E1 VALUE LEVEL']} / {e['E2 BOUNDARY SIGN']} / {e['E3 LOSS AMOUNT']}; acting {e['acting']:,}, resting {e['resting']:,}, incoherent "
                  f"{e['incoherent']:,} (zero quantity {e['zero_quantity_refused']:,}); value level {e['value_level_ratio']}")
for day in ('20261026', '20261027'):
    j3 = json.loads(open(f'{R}/pipeline_policy_v3_20261119/policy_{day}.json').read()); j2 = json.loads(open(f'{R}/pipeline_policy_v2_20261119/policy_{day}.json').read())
    cells = [(r, nm, k) for r in j2['values'] for nm in j2['values'][r] for k in ('KEEP', 'score', 'PULLALL', 'E', 'pulls')]
    same = all(abs(j3['values'][r][nm][k] - j2['values'][r][nm][k]) <= 1e-9 * max(1., abs(j2['values'][r][nm][k])) for r, nm, k in cells)
    rd = {k: (j2['readings'].get(k), j3['readings'].get(k)) for k in set(j2['readings']) | set(j3['readings']) if j2['readings'].get(k) != j3['readings'].get(k)}
    m = j3['values']['MODEL-J'][B]
    print(f"POLICY v3.1 {day}: at B MODEL-J {m['KEEP'] + m['score']:+,.2f}, PULL-ALL {m['KEEP'] + m['PULLALL']:+,.2f}, difference {m['score'] - m['PULLALL']:+,.2f}; pulls "
          f"{j3['pulled_orders']:,}; parity {j3['parity']}; VALUES EQUAL v2 IN ALL {len(cells)} CELLS: {same}; readings that differ: {rd}")
    # the attribution at B from the keep / pull run's own fills
    o = pd.read_parquet(f'{R}/keep_pull_20261117/{day}/orders.parquet', columns=['oid', 'side', 'p', 'mark', 'status', 't_full', 'g_struct']).set_index('oid')
    f = pd.read_parquet(f'{R}/keep_pull_20261117/{day}/fills.parquet', columns=['oid', 'clock', 't', 'q']); f = f[f.clock == 50.]
    el = o[o.status == 'eligible']; endj = pd.Series(SC.endj_n(el.t_full.values, el.g_struct.values, KP.NOTICE), index=el.index)
    pr = pd.read_parquet(f'{R}/pipeline_bench_v3_20261119/pred_{day}.parquet', columns=['kind', 'oid', 'j', 'J_V', 'J_ok']); rr = pr[pr.kind == 'retain']
    T = rr[rr.J_ok.astype(bool) & (rr.J_V < 0)].groupby('oid').j.min(); has = set(rr.oid.astype(int))
    f = f[f.oid.isin(el.index)]; sg = np.where(el.side.reindex(f.oid).values == 0, 1., -1.); v = f.q.values * sg * (el.mark.reindex(f.oid).values - el.p.reindex(f.oid).values)
    Tf = T.reindex(f.oid).values; ej = endj.reindex(f.oid).values; t = f.t.values
    mj_gone = np.isfinite(Tf) & (Tf < ej) & (t >= KP.landing(np.nan_to_num(Tf, nan=KP.J0).astype(int))); pa_gone = (KP.J0 < ej) & (t >= KP.landing(KP.J0))
    per = pd.DataFrame(dict(oid=f.oid.values, mj=v * ~mj_gone, pa=v * ~pa_gone)).groupby('oid').sum().reindex(el.index).fillna(0.)
    act = T.reindex(el.index).notna().values & (T.reindex(el.index).fillna(10 ** 6).values < endj.values)
    grp = np.where(act & (T.reindex(el.index).values == KP.J0), 'pulled at the first live step', np.where(act, 'pulled later', np.where(el.index.isin(list(has)), 'never pulled, with RETAIN predictions', 'no RETAIN prediction')))
    ok = abs(per.mj.sum() - (m['KEEP'] + m['score'])) <= 1e-6 * max(1., abs(per.mj.sum())) and abs(per.pa.sum() - (m['KEEP'] + m['PULLALL'])) <= 1e-6 * max(1., abs(per.pa.sum()))
    print(f'  ATTRIBUTION {day} (MODEL-J - PULL-ALL at B; totals reconcile with the committed policy v3.1 values: {ok})')
    for side in ('both', 0, 1):
        ms = np.ones(len(el), bool) if side == 'both' else (el.side.values == side)
        for g in ('pulled at the first live step', 'pulled later', 'never pulled, with RETAIN predictions', 'no RETAIN prediction'):
            mg = ms & (grp == g)
            print(f"    {('bid' if side == 0 else 'ask') if side != 'both' else 'both':<4} {g:<38} orders {int(mg.sum()):>6,}  MODEL-J {per.mj.values[mg].sum():>+11,.2f}  PULL-ALL {per.pa.values[mg].sum():>+11,.2f}  "
                  f'difference {(per.mj.values - per.pa.values)[mg].sum():>+10,.2f}')
