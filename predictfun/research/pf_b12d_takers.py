"""B12d POST HOC (2026-11-11, written after reading b12d.txt; descriptive, no reading). B12d's three mechanisms
(dodging follow-ups, reading thinning, holding the level) each account for about a tenth of the top makers' edge, and
the top makers STAY after being hit. One candidate is left that the tape can show: who the takers are. If the top
makers also take, their own (informed) taker orders hit other makers and never their own quotes.

Runs the committed pf_b12b_reason build unchanged (c882da77) and joins the cohort's chain match records
(live_vs_replay/matches_20261108.parquet: mid, exec_s, tx, taker, maker, ptick, shares) to the research legs on (mid,
exec_s, ptick, shares). Only keys with exactly one match row are used. Signer id -> maker address = the modal address of
the signer's unambiguous legs (checked: our id holds our address from who.json). Taker order -> taker address = the
modal taker of its unambiguous legs; none: status.
Outputs: (1) for each top-5 maker, its taker volume in the first minute (shares, share of all taker volume); (2) the
value to the makers hit (c/sh at 1 s, at-best legs) of taker orders sent by top-5 addresses against other takers; (3)
the B12b part A gap (top 5 against O, within guard) on legs whose taker is not a top-5 address, against all; (4) each
maker group's share of at-best fill shares that came from top-5 taker orders.
Usage: predictfun/research/run_light.sh b12d_takers -- python3 -m predictfun.research.pf_b12d_takers <out_dir>
"""
import json
import os
import sys

import numpy as np
import pandas as pd

from . import pf_b12_makers as B
from . import pf_b12b_reason as BB

MATCH = f'{BB.LV}/matches_20261108.parquet'
WHO = f'{BB.LV}/who.json'


def main():
    out = sys.argv[1]; os.makedirs(out, exist_ok=True)
    ok, ctl = BB.controls()
    if not ok:
        sys.exit(f'CONTROL FAILED -> no table: {ctl}')
    X = BB.build(); R = X['R']; W0 = X['W0']; OWN = X['OWN']
    k0 = BB.reconcile(R, X['days'])
    if not k0['ok']:
        sys.exit(f'CONTROL FAILED (K0) -> no table: {k0}')
    M = pd.read_parquet(MATCH, columns=['mid', 'exec_s', 'taker', 'maker', 'ptick', 'shares'])
    M = M[M.mid.isin(W0)].copy(); M['exec_s'] = M.exec_s.astype('int64'); M['sh6'] = M.shares.round(6)
    key = ['mid', 'exec_s', 'ptick', 'sh6']
    cnt = M.groupby(key).size().rename('n').reset_index(); Mu = M.merge(cnt[cnt.n == 1][key], on=key)
    L = R[['lid', 'mid', 'exec_s', 'ptick', 'shares', 'signer', 'grp']].copy(); L['exec_s'] = L.exec_s.astype('int64'); L['sh6'] = L.shares.round(6)
    lc = L.groupby(key).size().rename('nl').reset_index(); Lu = L.merge(lc[lc.nl == 1][key], on=key)
    J = Lu.merge(Mu[key + ['taker', 'maker']], on=key, how='inner')
    stat = dict(legs=int(len(R)), legs_unambiguous=int(len(J)), share_joined=float(len(J) / len(R)))
    addr = J.groupby('signer').maker.agg(lambda s: s.value_counts().index[0])
    purity = J.groupby('signer').maker.agg(lambda s: s.value_counts().iloc[0] / len(s))
    who = json.load(open(WHO)); own_addr = str(who['our_address']).lower()
    stat['own_address_ok'] = bool(OWN in addr.index and str(addr[OWN]).lower() == own_addr)
    stat['signer_purity_p10'] = float(purity.quantile(0.10))
    if not stat['own_address_ok']:
        sys.exit(f'CONTROL FAILED (own address) -> no table: {stat}')
    tk = J.groupby('grp').taker.agg(lambda s: s.value_counts().index[0])
    R['taker'] = tk.reindex(R.grp.values).values
    top5 = sorted(int(s) for s in R.signer[R.g == 'top 5'].unique())
    t5a = {s: str(addr.get(s, '')).lower() for s in top5}; t5set = {a for a in t5a.values() if a}
    R['taker_l'] = R.taker.astype(str).str.lower()
    R['tk_top5'] = R.taker_l.isin(t5set); R['tk_known'] = R.taker.notna()
    # (1) taker volume of each top-5 maker address (first minute, all legs: shares of maker legs they hit)
    kn = R[R.tk_known]; tot = float(kn.shares.sum())
    take = {str(s): dict(taker_shares=float(kn.shares[kn.taker_l == a].sum()), taker_share_pct=float(kn.shares[kn.taker_l == a].sum() / tot * 100.0) if tot else np.nan)
            for s, a in t5a.items()}
    # (2) value to the makers hit, by taker group (at-best legs, any guard state)
    V = R[R.val_ok & (R.status == 'ok') & (R.depth == 0) & R.tk_known]
    val_by_taker = {}
    for nm, mk in (('top-5 taker', V.tk_top5.values), ('other taker', ~V.tk_top5.values)):
        x = V[mk]; val_by_taker[nm] = dict(legs=int(len(x)), shares=float(x.shares.sum()), net1=B.wmean(x.shares.values, x.v1.values),
                                           move1=B.wmean(x.shares.values, x.v1.values - x.spr.values))
    # (3) part A gap without top-5 taker orders; (4) share of fills from top-5 takers by maker group
    A = R[R.val_ok & (R.status == 'ok') & (R.depth == 0) & R.g50.isin([B.G_ON, B.G_OFF]) & (R.sweep == 0)]
    D = A[A.g.isin(['top 5', '6-20', 'rest'])]
    gap = {}
    for nm, mk in (('all part A legs', np.ones(len(D), bool)), ('taker known', D.tk_known.values),
                   ('taker known, not a top-5 address', D.tk_known.values & ~D.tk_top5.values), ('taker a top-5 address', D.tk_top5.values)):
        x = D[mk]
        if len(x) and (x.g == 'top 5').any() and (x.g != 'top 5').any():
            d = BB.decompose_n((x.g50.values == B.G_OFF).astype(np.int64), x.g.values == 'top 5', x.shares.values, x.v1.values, 2)
            gap[nm] = dict(legs=int(len(x)), WITHIN_G=d['WITHIN'])
    share = {}
    for g in BB.GR:
        x = BB.sel(A[A.tk_known], g); s_ = x.shares.values
        share[g] = dict(legs=int(len(x)), from_top5_takers_pct=float(s_[x.tk_top5.values].sum() / s_.sum() * 100.0) if s_.sum() else np.nan,
                        net1_from_top5=B.wmean(s_[x.tk_top5.values], x.v1.values[x.tk_top5.values]),
                        net1_from_others=B.wmean(s_[~x.tk_top5.values], x.v1.values[~x.tk_top5.values]))
    res = dict(note='POST HOC, descriptive; same build and statuses as b12b.json', controls_ok=ok, K0=k0['ok'], stat=stat, top5_signers=top5,
               top5_as_takers=take, value_by_taker=val_by_taker, partA_gap=gap, fills_from_top5_takers=share)
    json.dump(res, open(f'{out}/b12d_takers.json', 'w'), indent=1, default=float)
    f = lambda v: 'n/a' if not BB.fin(v) else f'{v:+.2f}'
    Lt = ['B12d POST HOC: who the takers are (descriptive).', f"Join: {stat}", '',
          'Top-5 makers as takers (first minute; shares of maker legs their taker orders hit, % of taker volume with a known taker):']
    Lt += [f"  maker {s}: {v['taker_shares']:,.1f} sh ({v['taker_share_pct']:.2f}%)" for s, v in take.items()]
    Lt.append('Value to the makers hit, at-best legs, by taker: ' + '; '.join(f"{k}: legs {v['legs']}, net {f(v['net1'])}, move {f(v['move1'])}" for k, v in val_by_taker.items()))
    Lt.append('Part A gap, top 5 against O within guard (c/sh at 1 s): ' + '; '.join(f"{k}: {f(v['WITHIN_G'])} ({v['legs']} legs)" for k, v in gap.items()))
    Lt.append('Share of each group\'s part A fill shares from top-5 taker orders (net from them / from others): ' + '; '.join(
        f"{g}: {v['from_top5_takers_pct']:.1f}% ({f(v['net1_from_top5'])} / {f(v['net1_from_others'])})" for g, v in share.items()))
    open(f'{out}/b12d_takers.txt', 'w').write('\n'.join(Lt) + '\n')
    print('\n'.join(Lt))


if __name__ == '__main__':
    main()
