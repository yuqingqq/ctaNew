"""Fresh-refresh predictions (2026-11-05; user: "we can use the latest feed to refresh the prediction"): merge the 50 ms grid guard
predictions (preds_t60k) with the same models' predictions at every Predict frame's known time in the first 65 s (preds_te60k, from
rows_rl_d2_pk_ev) into preds_t60kE. Grid rows win exact (mid, tau) ties. The 1 s / 250 ms mid predictions are not recomputed on
frames: every-frame rows carry the latest grid value (as-of), used only by A1 / M / Q arms.
Checks (refuse on failure): the two runs' in-sample pull thresholds are identical per day and side (same models); every-frame taus
lie in [ws, ws + 65 s); merged taus strictly increase within each market.
2026-11-06 (whole-window refresh, declared in STATUS): PF_EV_PSTEM (default te) / PF_EV_SPAN_S (65) / PF_EV_OUT (preds_t60kE) select the
every-frame run, its span and the output; PF_EV_PARITY=<merged stem> refuses unless this output restricted to [ws, ws + PF_EV_PARITY_S
(65) s) equals that file restricted the same way, row for row. Defaults reproduce the original run.
Usage: python3 -m predictfun.research.pf_preds_merge_ev <day> ...
"""
import sys, json, os
import numpy as np, pandas as pd
from . import pf_io as io

DAYS = sys.argv[1:]
assert DAYS and all(d not in ('20261028', '20261029', '20261030') and d < '20261104' for d in DAYS), DAYS
PST = os.environ.get('PF_EV_PSTEM', 'te'); SPAN = float(os.environ.get('PF_EV_SPAN_S', '65')) * 1000.0; OUTS = os.environ.get('PF_EV_OUT', 'preds_t60kE')
PAR = os.environ.get('PF_EV_PARITY', ''); PAR_S = float(os.environ.get('PF_EV_PARITY_S', '65')) * 1000.0
RT = json.load(open(f'{io.OUT}/step5/guard_preds_report_t_60.json')); RE = json.load(open(f'{io.OUT}/step5/guard_preds_report_{PST}_60.json'))
rep = {}
for d in DAYS:
    assert RT[d]['thresholds'] == RE[d]['thresholds'], f'{d}: thresholds differ - not the same models'
    G = pd.read_parquet(f'{io.OUT}/step5/{d}/preds_t60k.parquet'); E = pd.read_parquet(f'{io.OUT}/step5/{d}/preds_{PST}60k.parquet')
    ws = pd.read_parquet(f'{io.OUT}/step1/{d}/registry.parquet', columns=['mid', 'ws']).set_index('mid').ws * 1000.0
    rel = E.tau.values - E.mid.map(ws).values
    assert np.all((rel >= 0) & (rel < SPAN)), f'{d}: every-frame rows outside [ws, ws + {SPAN / 1000:g} s)'
    E = E.drop(columns=['pred_1000', 'pred_250'])
    E = pd.merge_asof(E.sort_values('tau'), G[['mid', 'tau', 'pred_1000', 'pred_250']].sort_values('tau'), on='tau', by='mid', direction='backward')
    E = E[G.columns]
    M = pd.concat([G.assign(_o=0), E.assign(_o=1)], ignore_index=True).sort_values(['mid', 'tau', '_o'], kind='stable')
    dup = M.duplicated(['mid', 'tau'], keep='first'); M = M[~dup].drop(columns='_o').reset_index(drop=True)
    assert (M.groupby('mid').tau.diff().dropna() > 0).all(), f'{d}: taus not increasing'
    par = None
    if PAR:                         # overlap parity with an earlier merged file
        P = pd.read_parquet(f'{io.OUT}/step5/{d}/{PAR}.parquet')
        cut = lambda x: x[(x.tau.values - x.mid.map(ws).values) < PAR_S].reset_index(drop=True)
        a_, b_ = cut(M), cut(P)
        if not (len(a_) == len(b_) and a_.equals(b_[a_.columns])):
            sys.exit(f'REFUSED: {d}: {OUTS} restricted to {PAR_S / 1000:g} s differs from {PAR} ({len(a_)} vs {len(b_)} rows)')
        par = dict(rows_compared=len(a_), equal=True); del P, a_, b_
    M.to_parquet(f'{io.OUT}/step5/{d}/{OUTS}.parquet', index=False)
    rep[d] = dict(grid_rows=len(G), frame_rows=len(E), ties_dropped=int(dup.sum()), merged=len(M),
                  v_bid_nan_frame_rows=float(E.v_bid.isna().mean()), parity=par)
    print(d, rep[d], flush=True)
io.write_json(f"{io.OUT}/step5/preds_merge_ev_report{'' if OUTS == 'preds_t60kE' else '_' + OUTS}.json", rep)
