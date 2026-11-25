"""MAKER3-BOOK-CONDITIONS v1 addendum (USER 2026-11-25: "most spread is captured before 270 s when market is calm, mainly the turnovers"?).
Rebuild Maker3's completed round trips on all 934 markets with the econ FIFO rule (pf_mt_econ: lots per address x market, order
(settle_ms, sid, px_rank, mk_idx, maker before taker), unit j bought pairs with unit j sold, self-trades out of the lots), so completed
turnover can be split by WHEN the round trip closes and the book class of each leg (P clock). Residual units are valued to resolution.
Reconciled per market to econ/maker_market rt_pnl and resid_res (an independent builder). Inputs: legs_m3.parquet (build.py).
Usage: run_light.sh mbc_fifo -- python3 .../fifo.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

H = Path(__file__).resolve().parent
R = H.parents[2]
V = R / 'data/predict_fun/research/maker_trace_v2_20261123'
OUT = R / 'data/predict_fun/research/maker3_book_conditions_20261125'
M3 = '0x57FbC891E89184DA071AE9D97E745531cd7F718a'
DAYS = ['20261024', '20261025', '20261026', '20261027']
TB = ['pre', '000-060', '060-120', '120-180', '180-240', '240-270', '270-285', '285-295', '295-300', 'post_stamp']
LATE = {'270-285', '285-295', '295-300', 'post_stamp'}


def pairs_of(g):
    """Two pointers over buys and sells in econ order -> (buy row, sell row, matched q) and residual (row, remaining q)."""
    b = g[g.s > 0]
    s = g[g.s < 0]
    bi, si = b.index.to_numpy(), s.index.to_numpy()
    bq, sq = b.q.to_numpy(float).copy(), s.q.to_numpy(float).copy()
    i = j = 0
    out = []
    while i < len(bi) and j < len(si):
        m = min(bq[i], sq[j])
        if m > 0:
            out.append((bi[i], si[j], m))
        bq[i] -= m
        sq[j] -= m
        if bq[i] <= 1e-12:
            i += 1
        if sq[j] <= 1e-12:
            j += 1
    res = [(bi[k], bq[k]) for k in range(i, len(bi)) if bq[k] > 1e-12] + [(si[k], sq[k]) for k in range(j, len(si)) if sq[k] > 1e-12]
    return out, res


def main():
    out = OUT / 'fifo_pairs.parquet'
    assert not out.exists(), 'write-once'
    L = pd.read_parquet(OUT / 'legs_m3.parquet')
    L = L[~L.self_trade.astype(bool)].copy()
    L['_ro'] = (L.role != 'maker').astype(int)
    L = L.sort_values(['day', 'mid', 'settle_ms', 'sid', 'px_rank', 'mk_idx', '_ro'], kind='mergesort').reset_index(drop=True)
    L['ord'] = np.arange(len(L))
    P, Rz = [], []
    for (day, mid), g in L.groupby(['day', 'mid'], sort=False):
        pr, rs = pairs_of(g)
        P += pr
        Rz += rs
    P = pd.DataFrame(P, columns=['ib', 'is_', 'm'])
    Rz = pd.DataFrame(Rz, columns=['ir', 'rq'])
    cols = ['day', 'mid', 'ord', 't_rel_s', 'tb', 'cls_P', 'cls_U', 'role', 'p', 'V']
    b = L.loc[P.ib, cols].reset_index(drop=True).add_prefix('b_')
    s = L.loc[P.is_, cols].reset_index(drop=True).add_prefix('s_')
    pairs = pd.concat([P[['m']], b, s], axis=1)
    assert (pairs.b_day == pairs.s_day).all() and (pairs.b_mid == pairs.s_mid).all()
    pairs['gross'] = pairs.m * (pairs.s_p - pairs.b_p)
    first_is_buy = pairs.b_ord < pairs.s_ord
    for k in ('t_rel_s', 'tb', 'cls_P', 'cls_U', 'role'):
        pairs[f'entry_{k}'] = np.where(first_is_buy, pairs[f'b_{k}'], pairs[f's_{k}'])
        pairs[f'exit_{k}'] = np.where(first_is_buy, pairs[f's_{k}'], pairs[f'b_{k}'])
    pairs['day'], pairs['mid'] = pairs.b_day, pairs.b_mid
    r = L.loc[Rz.ir, ['day', 'mid', 's', 'p', 'V', 't_rel_s', 'tb', 'cls_P', 'role']].reset_index(drop=True)
    resid = pd.concat([Rz[['rq']], r], axis=1)
    resid['value_res'] = resid.s * resid.rq * (resid.V - resid.p)
    # reconciliation against the independent econ builder
    mm = pq.read_table(V / 'econ/maker_market.parquet', filters=[('addr', '=', M3), ('day', 'in', DAYS)],
                       columns=['day', 'mid', 'rt_pnl', 'resid_res', 'rt_sh', 'fee_res', 'net_res', 'status']).to_pandas()
    mm = mm[mm.status == 'ok'].set_index(['day', 'mid'])
    j = pd.concat([pairs.groupby(['day', 'mid']).gross.sum().rename('rt'), pairs.groupby(['day', 'mid']).m.sum().rename('rt_sh'),
                   resid.groupby(['day', 'mid']).value_res.sum().rename('resid')], axis=1).reindex(mm.index).fillna(0.0)
    rec = dict(markets=len(mm), pairs=len(pairs), residual_rows=len(resid),
               max_abs_rt_diff=float((j.rt - mm.rt_pnl).abs().max()), max_abs_rt_sh_diff=float((j.rt_sh - mm.rt_sh).abs().max()),
               max_abs_resid_diff=float((j.resid - mm.resid_res).abs().max()))
    rec['reconciles'] = bool(rec['max_abs_rt_diff'] < 1e-6 and rec['max_abs_resid_diff'] < 1e-6 and rec['max_abs_rt_sh_diff'] < 1e-6)
    assert rec['reconciles'], rec
    pairs.to_parquet(out, index=False)
    resid.to_parquet(OUT / 'fifo_residual.parquet', index=False)
    T = H / 'tables'

    def tab(d, keys):
        x = d.groupby(keys, observed=True).agg(matched_sh=('m', 'sum'), gross=('gross', 'sum'), markets=('mid', 'nunique')).reset_index()
        x['c_per_matched_sh'] = 100 * x.gross / x.matched_sh
        return x
    pairs['exit_late'] = np.where(pairs.exit_tb.isin(LATE), '>=270', '<270')
    pairs['entry_late'] = np.where(pairs.entry_tb.isin(LATE), '>=270', '<270')
    pairs['both_calm'] = (pairs.entry_cls_P == 'CALM') & (pairs.exit_cls_P == 'CALM')
    pairs['roles'] = np.where((pairs.entry_role == 'maker') & (pairs.exit_role == 'maker'), 'maker->maker', 'with_taker')
    t_exit = tab(pairs, ['exit_tb'])
    t_exit['exit_tb'] = pd.Categorical(t_exit.exit_tb, TB, ordered=True)
    t_exit = t_exit.sort_values('exit_tb')
    t_exit.to_csv(T / 'f1_rt_by_exit_tb.csv', index=False)
    t_el = tab(pairs, ['entry_late', 'exit_late'])
    t_el.to_csv(T / 'f2_rt_entry_exit_late.csv', index=False)
    t_cls = tab(pairs, ['exit_late', 'entry_cls_P', 'exit_cls_P'])
    t_cls.to_csv(T / 'f3_rt_late_cls.csv', index=False)
    t_day = tab(pairs, ['day', 'exit_late'])
    t_day.to_csv(T / 'f4_rt_day_exit_late.csv', index=False)
    t_role = tab(pairs, ['exit_late', 'roles'])
    t_role.to_csv(T / 'f5_rt_roles.csv', index=False)
    t_calm = tab(pairs, ['exit_late', 'both_calm'])
    t_calm.to_csv(T / 'f6_rt_both_calm.csv', index=False)
    t_calm_u = tab(pairs.assign(both_calm_U=(pairs.entry_cls_U == 'CALM') & (pairs.exit_cls_U == 'CALM')), ['exit_late', 'both_calm_U'])
    t_calm_u.to_csv(T / 'f6b_rt_both_calm_U.csv', index=False)
    rz = resid.groupby(['tb', 'cls_P'], observed=True).agg(rem_sh=('rq', 'sum'), value_res=('value_res', 'sum')).reset_index()
    rz.to_csv(T / 'f7_residual_by_acq_tb_cls.csv', index=False)
    tot = pairs.gross.sum()
    early = pairs[pairs.exit_late == '<270'].gross.sum()
    pos = pairs.gross.clip(lower=0).sum()
    dsg = {d: float(np.sign(x[x.exit_late == '<270'].gross.sum())) for d, x in pairs.groupby('day')}
    dsl = {d: float(np.sign(x[x.exit_late == '>=270'].gross.sum())) for d, x in pairs.groupby('day')}
    rec.update(rt_total_usd=float(tot), rt_exit_before_270_usd=float(early), rt_exit_ge_270_usd=float(tot - early),
               rt_exit_before_240_usd=float(pairs[~pairs.exit_tb.isin(LATE | {'240-270'})].gross.sum()),
               matched_sh_total=float(pairs.m.sum()), matched_sh_exit_before_270=float(pairs[pairs.exit_late == '<270'].m.sum()),
               rt_both_calm_exit_before_270_usd=float(pairs[(pairs.exit_late == '<270') & pairs.both_calm].gross.sum()),
               rt_both_calm_usd=float(pairs[pairs.both_calm].gross.sum()),
               rt_positive_gross_usd=float(pos), day_signs_exit_before_270=dsg, day_signs_exit_ge_270=dsl,
               resid_total_usd=float(resid.value_res.sum()),
               resid_acquired_ge_270_usd=float(resid[resid.tb.isin(LATE)].value_res.sum()))
    rec['share_rt_from_exits_before_270'] = rec['rt_exit_before_270_usd'] / tot
    (H / 'fifo.json').write_text(json.dumps(rec, indent=1) + '\n')
    pd.set_option('display.width', 250)
    print(json.dumps(rec, indent=1))
    for name, x in (('by exit time', t_exit), ('entry x exit late', t_el), ('roles', t_role), ('both calm (P)', t_calm), ('both calm (U)', t_calm_u),
                    ('days', t_day)):
        print('==', name)
        print(x.round(2).to_string(index=False))
    print('== late x classes (top)')
    print(t_cls.sort_values('gross').round(2).to_string(index=False))
    print('== residual by acquisition')
    print(rz.round(1).to_string(index=False))


if __name__ == '__main__':
    main()
