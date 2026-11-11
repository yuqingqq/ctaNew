"""B13d (2026-11-11, diagnosis, descriptive). Diagnosis check (descriptive): at our live at-best cancels on 11-08 (first minute), was our side's L1 size below the other
side's in the research frame strictly before the cancel (G4 under MICRO withholds the lighter side)? And how often does
the L1 size balance flip per second at random times in the first minute?"""
import json, numpy as np, pandas as pd, sys
sys.path.insert(0, '/home/yuqing/ctaNew')
from predictfun.research import pf_b12b_reason as BB, pf_b12d_bursts as DB
RP = BB.RP; sc = json.load(open(f'{RP}/scope.json')); W0 = {int(w['market']): w['ws'] * 1000.0 for w in sc['windows']}
FR = DB.frames(W0)
core = pd.concat([pd.read_parquet(f'{BB.LV}/core_orders_{r}.parquet') for r in ('v6', 'basis')], ignore_index=True)
core = core[core.t_Accepted.notna() & core.refused.isna() & core.market.isin(list(W0)) & core.t_cancel.notna()].copy()
core['side'] = (core.lane % 2).astype(int); core['ws'] = core.market.map(W0).astype(float)
core = core[(core.t_cancel >= core.ws) & (core.t_cancel < core.ws + 60000.0) & (core.lane < 2)]
light = []; at_best = []
for o in core.itertuples():
    tf, S = FR[o.market]; j = np.searchsorted(tf, o.t_cancel, side='left') - 1
    if j < 0: continue
    P0, S0 = S[0]; P1, S1 = S[1]
    best = (P0 if o.side == 0 else P1)[j, 0]
    if best != o.tick: continue
    bs, as_ = S0[j, 0], S1[j, 0]
    own_side, other = (bs, as_) if o.side == 0 else (as_, bs)
    at_best.append(1); light.append(own_side < other)
light = np.array(light)
print('touch-lane cancels at the best (first minute):', len(light), '| our side lighter than the other at the cancel: %.1f%%' % (100 * light.mean()))
flips = []
for m, (tf, S) in FR.items():
    ws = W0[m]; k = (tf >= ws) & (tf < ws + 60000.0)
    bs = S[0][1][k, 0]; as_ = S[1][1][k, 0]; sp = S[1][0][k, 0] - S[0][0][k, 0]
    sgn = np.sign(bs - as_); ok = (sp == 1)
    s = sgn[ok]; s = s[s != 0]
    flips.append(np.sum(s[1:] != s[:-1]) / 60.0)
print('L1 size-balance flips per second (1-tick spread frames), per window p25/p50/p75:', np.round(np.percentile(flips, [25, 50, 75]), 2))
