"""MAKER3-BOOK-CONDITIONS v1, stage 3: the declared tables and computed readings (contract.md) from the build outputs.
Writes small CSV tables + readings.json into this packet. Usage: run_light.sh mbc_sum -- python3 .../summarize.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

H = Path(__file__).resolve().parent
R = H.parents[2]
OUT = R / 'data/predict_fun/research/maker3_book_conditions_20261125'
T = H / 'tables'
DAYS = ('20261024', '20261025', '20261026', '20261027')
CLS = ['CALM', 'THIN', 'WIDE', 'ONE_SIDED', 'UNKNOWN']
TB = ['pre', '000-060', '060-120', '120-180', '180-240', '240-270', '270-285', '285-295', '295-300', 'post_stamp']
MIN_LEGS, MIN_MKTS = 30, 3


def per_share(d):
    """Per-share cents for summed cells; a value's denominator is the shares where it is defined."""
    d = d.copy()
    for v, qv in (('net', 'q'), ('gross', 'q'), ('edge', 'q_edge'), ('out', 'q_edge'), ('v_1s', 'q_v_1s'), ('v_10s', 'q_v_10s'),
                  ('v_end', 'q_v_end')):
        d[f'{v}_c'] = np.where(d[qv] > 0, 100 * d[v] / d[qv].where(d[qv] > 0), np.nan)
    return d


SUMC = ['legs', 'q', 'gross', 'fee', 'net', 'edge', 'out', 'q_edge', 'v_1s', 'v_10s', 'v_60s', 'v_end', 'q_v_1s', 'q_v_10s', 'q_v_60s', 'q_v_end']


def agg(d, keys):
    return d.groupby(keys, observed=True, dropna=False)[SUMC].sum().reset_index()


def conc(m, keys):
    """Top-3-market fraction of the absolute per-market net contributions, and market count, per cell (Maker3 per-market cells)."""
    pm = m.groupby(keys + ['day', 'mid'], observed=True)['net'].sum().abs().reset_index()

    def f(x):
        a = np.sort(x.net.to_numpy())[::-1]
        return pd.Series(dict(markets=len(a), top3_abs_frac=float(a[:3].sum() / a.sum()) if a.sum() > 0 else np.nan))
    return pm.groupby(keys, observed=True).apply(f, include_groups=False).reset_index()


def main():
    T.mkdir(exist_ok=True)
    A = pd.read_parquet(OUT / 'cells_all.parquet')
    M = pd.read_parquet(OUT / 'cells_m3.parquet')
    K = pd.read_parquet(OUT / 'markets.parquet')
    A['vol'] = np.where(A.cls.isin(['THIN', 'WIDE']), 'THIN_or_WIDE', A.cls)
    M['vol'] = np.where(M.cls.isin(['THIN', 'WIDE']), 'THIN_or_WIDE', M.cls)
    M['late'] = np.where(M.tb.isin(['270-285', '285-295', '295-300', 'post_stamp']), '>=270', '<270')
    A['late'] = np.where(A.tb.isin(['270-285', '285-295', '295-300', 'post_stamp']), '>=270', '<270')
    mk = M[M.role == 'maker']
    out = {}
    # T1: activity by time bucket x class (P clock), Maker3 share of all maker shares
    a1 = agg(A[A.clock == 'P'], ['tb', 'cls', 'grp'])
    tot = a1.groupby(['tb', 'cls'])['q'].sum().rename('q_all')
    m1 = a1[a1.grp == 'MAKER3'].set_index(['tb', 'cls'])[['legs', 'q']].join(tot, how='right').fillna(0).reset_index()
    m1['m3_share_of_maker_volume'] = m1.q / m1.q_all
    m1['m3_frac_of_own_volume'] = m1.q / m1.q.sum()
    m1['tb'] = pd.Categorical(m1.tb, TB, ordered=True)
    m1 = m1.sort_values(['tb', 'cls'])
    m1.to_csv(T / 't1_activity_tb_cls_P.csv', index=False)
    # T2: economics by class x group (P and U)
    t2 = per_share(agg(A, ['clock', 'grp', 'cls']))
    t2.to_csv(T / 't2_econ_cls_grp.csv', index=False)
    t2v = per_share(agg(A, ['clock', 'grp', 'vol', 'late']))
    t2v.to_csv(T / 't2b_econ_vol_late_grp.csv', index=False)
    # T3: time bucket x class, Maker3 vs FIELD (P)
    t3 = per_share(agg(A[A.clock == 'P'], ['grp', 'tb', 'cls']))
    t3.to_csv(T / 't3_econ_tb_cls_grp_P.csv', index=False)
    # T4: moneyness x class x late, Maker3 maker legs (P)
    t4 = per_share(agg(mk[mk.clock == 'P'], ['late', 'money', 'cls']))
    t4.to_csv(T / 't4_m3_money_cls_late_P.csv', index=False)
    # T5: spread band x late, Maker3 maker (P)
    t5 = per_share(agg(mk[mk.clock == 'P'], ['late', 'band']))
    t5.to_csv(T / 't5_m3_band_late_P.csv', index=False)
    # T6: day split
    t6 = per_share(agg(mk, ['clock', 'day', 'vol', 'late']))
    t6.to_csv(T / 't6_m3_day_vol_late.csv', index=False)
    t6f = per_share(agg(A, ['clock', 'grp', 'day', 'vol', 'late']))
    t6f.to_csv(T / 't6b_day_vol_late_grp.csv', index=False)
    # T7: inventory class x class x late, Maker3 maker legs (P)
    t7 = per_share(agg(mk[mk.clock == 'P'], ['late', 'vol', 'inv']))
    t7.to_csv(T / 't7_m3_inv_vol_late_P.csv', index=False)
    # T8: Maker3 taker legs
    tk = M[M.role == 'taker']
    t8 = per_share(agg(tk[tk.clock == 'P'], ['late', 'cls', 'inv']))
    t8.to_csv(T / 't8_m3_taker_cls_inv_late_P.csv', index=False)
    # T9: market level
    act = K[K._merge == 'both'].copy()
    t9 = act.groupby('late_class').agg(markets=('mid', 'size'), net=('net_res', 'sum'), rt=('rt_pnl', 'sum'), resid=('resid_res', 'sum'),
                                       sh_mk=('sh_mk', 'sum'), sh_tk=('sh_tk', 'sum'), decided270=('decided270', 'mean'),
                                       net_pos_abs=('net_pos', lambda x: x.abs().mean())).reset_index()
    t9['net_per_market'] = t9.net / t9.markets
    t9.to_csv(T / 't9_market_late_class.csv', index=False)
    t9d = act.groupby(['late_class', 'day']).agg(markets=('mid', 'size'), net=('net_res', 'sum'), rt=('rt_pnl', 'sum'),
                                                  resid=('resid_res', 'sum')).reset_index()
    t9d.to_csv(T / 't9b_market_late_class_day.csv', index=False)
    t9e = act.groupby(['decided270']).agg(markets=('mid', 'size'), net=('net_res', 'sum'), rt=('rt_pnl', 'sum'), resid=('resid_res', 'sum'),
                                          resid_resol=('resid_resol', 'sum')).reset_index()
    t9e.to_csv(T / 't9c_market_decided270.csv', index=False)
    # concentration for Maker3 cells
    cc = conc(mk, ['clock', 'vol', 'late'])
    cc.to_csv(T / 'conc_m3_vol_late.csv', index=False)
    cc2 = conc(mk, ['clock', 'cls'])
    cc2.to_csv(T / 'conc_m3_cls.csv', index=False)
    # ---------------- readings (contract.md (a)-(e))
    P = mk[mk.clock == 'P']
    q_all = P.q.sum()
    q_tw = P[P.vol == 'THIN_or_WIDE'].q.sum()
    q_late = P[P.late == '>=270'].q.sum()
    q_tw_late = P[(P.vol == 'THIN_or_WIDE') & (P.late == '>=270')].q.sum()
    out['a_m3_volume_frac_thin_or_wide_P'] = float(q_tw / q_all)
    out['a_m3_volume_frac_late270'] = float(q_late / q_all)
    out['a_m3_late270_volume_frac_thin_or_wide_P'] = float(q_tw_late / q_late)
    out['a_m3_late270_volume_frac_one_sided_P'] = float(P[(P.cls == 'ONE_SIDED') & (P.late == '>=270')].q.sum() / q_late)
    rb = {}
    for clk in ('P', 'U'):
        for late in ('<270', '>=270', 'all'):
            for vol in ('THIN_or_WIDE', 'CALM', 'ONE_SIDED'):
                d = mk[(mk.clock == clk) & (mk.vol == vol) & ((mk.late == late) if late != 'all' else True)]
                s = per_share(agg(d.assign(k=1), ['k']))
                dd = per_share(agg(d, ['day']))
                signs = {r.day: (float(np.sign(r.net)) if r.legs >= MIN_LEGS else None) for r in dd.itertuples()}
                sv = [v for v in signs.values() if v is not None]
                nm = d.groupby(['day', 'mid']).ngroups
                c = conc(d.assign(k=1), ['k'])
                rb[f'{clk}|{late}|{vol}'] = dict(legs=int(s.legs.sum()), shares=float(s.q.sum()), markets=int(nm), net_usd=float(s.net.sum()),
                                                 net_c=float(s.net_c.iloc[0]) if len(s) else None, edge_c=float(s.edge_c.iloc[0]) if len(s) else None,
                                                 out_c=float(s.out_c.iloc[0]) if len(s) else None,
                                                 day_net_signs=signs, day_consistent=bool(len(sv) >= 2 and len(set(sv)) == 1),
                                                 top3_abs_frac=float(c.top3_abs_frac.iloc[0]) if len(c) else None,
                                                 concentrated=bool(len(c) and c.top3_abs_frac.iloc[0] > 0.5),
                                                 support='ok' if (s.legs.sum() >= MIN_LEGS and nm >= MIN_MKTS) else 'THIN-SUPPORT')
    out['b_c_m3_cells'] = rb
    rd = {}
    for clk in ('P', 'U'):
        g = per_share(agg(A[A.clock == clk], ['grp', 'vol', 'late']))
        for (vol, late), x in g.groupby(['vol', 'late']):
            m3 = x[x.grp == 'MAKER3']
            fd = x[x.grp == 'FIELD']
            ot = x[x.grp == 'OTHER_TOP10']
            if len(m3) and len(fd):
                rd[f'{clk}|{late}|{vol}'] = dict(m3_net_c=float(m3.net_c.iloc[0]), field_net_c=float(fd.net_c.iloc[0]),
                                                 other_top10_net_c=float(ot.net_c.iloc[0]) if len(ot) else None,
                                                 m3_minus_field_c=float(m3.net_c.iloc[0] - fd.net_c.iloc[0]),
                                                 m3_share_of_maker_volume=float(m3.q.iloc[0] / x.q.sum()))
    out['d_m3_minus_field'] = rd
    out['e_sign_survives_U'] = {k: (None if rb[k]['net_usd'] == 0 else bool(np.sign(rb[k]['net_usd']) == np.sign(rb['U' + k[1:]]['net_usd'])))
                                for k in rb if k.startswith('P|')}
    tw_all, calm_all = rb['P|all|THIN_or_WIDE'], rb['P|all|CALM']
    out['b_thin_or_wide_beats_calm_per_share_P'] = bool(tw_all['net_c'] > calm_all['net_c'])
    out['b_thin_or_wide_positive_every_day_P'] = bool(all(v == 1.0 for v in tw_all['day_net_signs'].values() if v is not None))
    out['market_level_total_net_usd'] = float(act.net_res.sum())
    out['market_level_total_rt_usd'] = float(act.rt_pnl.sum())
    out['market_level_total_resid_usd'] = float(act.resid_res.sum())
    # reconciliation: Maker3 cells (P, both roles) sum to its maker_market net per market
    pm = M[M.clock == 'P'].groupby(['day', 'mid'])['net'].sum()
    mm = act.set_index(['day', 'mid']).net_res
    j = pd.concat([pm.rename('cells'), mm.rename('mm')], axis=1)
    out['recon_markets'] = int(len(j))
    out['recon_max_abs_diff_usd'] = float((j.cells - j.mm).abs().max())
    out['recon_markets_missing'] = int(j.isna().any(axis=1).sum())
    (H / 'readings.json').write_text(json.dumps(out, indent=1, default=float) + '\n')
    pd.set_option('display.width', 250)
    pd.set_option('display.max_rows', 200)
    print(json.dumps({k: v for k, v in out.items() if not isinstance(v, dict)}, indent=1))


if __name__ == '__main__':
    main()
