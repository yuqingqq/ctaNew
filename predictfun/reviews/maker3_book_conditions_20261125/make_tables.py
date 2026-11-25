"""MAKER3-BOOK-CONDITIONS v1: render the report's tables from the saved outputs (no number is typed by hand). Writes report_tables.md."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

H = Path(__file__).resolve().parent
R = H.parents[2]
OUT = R / 'data/predict_fun/research/maker3_book_conditions_20261125'
TB = ['000-060', '060-120', '120-180', '180-240', '240-270', '270-285', '285-295', '295-300']
LATE = ['270-285', '285-295', '295-300', 'post_stamp']


def md(df):
    cols = list(df.columns)
    lines = ['| ' + ' | '.join(map(str, cols)) + ' |', '|' + '|'.join('---' if i == 0 else '---:' for i in range(len(cols))) + '|']
    for r in df.itertuples(index=False):
        lines.append('| ' + ' | '.join('' if (isinstance(v, float) and np.isnan(v)) else str(v) for v in r) + ' |')
    return '\n'.join(lines)


def pc(x, d=1):
    return f'{100 * x:.{d}f}%'


def c(x, d=2):
    return '' if x is None or (isinstance(x, float) and not np.isfinite(x)) else f'{x:+.{d}f}'


def usd(x):
    return f'{x:+,.0f}'


def main():
    out = []
    g = pd.read_parquet(OUT / 'grid.parquet')
    g = g[g.st != 'boundary_censored'].copy()
    g['tb'] = pd.cut(g.k, [0, 60, 120, 180, 240, 270, 285, 295, 300], right=False, labels=TB).astype(str)
    g['spr'] = g.ask_c - g.bid_c
    g['thin'] = np.minimum(g.bid_sh, g.ask_sh)
    g['midc'] = (g.ask_c + g.bid_c) / 2
    rows = []
    for tb in TB:
        x = g[g.tb == tb]
        o = x[x.st == 'ok']
        rows.append(dict(window_s=tb, market_seconds=len(x), one_sided=pc((x.st == 'one_sided').mean()), spread_gt_1c=pc((o.spr > 1 + 1e-7).mean()),
                         spread_gt_4c=pc((o.spr > 4 + 1e-7).mean()), thinner_touch_median_sh=f'{o.thin.median():.0f}',
                         two_sided_at_extreme_price=pc(((o.midc <= 10) | (o.midc >= 90)).mean())))
    out.append('### R1 Market-wide book on the 1 s grid (1,151 markets, outcome-blind)\n\n' + md(pd.DataFrame(rows)))
    A = pd.read_parquet(OUT / 'cells_all.parquet')
    M = pd.read_parquet(OUT / 'cells_m3.parquet')
    a = A[(A.clock == 'P')]
    m3 = a[a.grp == 'MAKER3']
    tot = m3.q.sum()
    rows = []
    for tb in TB:
        x = m3[m3.tb == tb]
        allq = a[a.tb == tb].q.sum()
        rows.append(dict(window_s=tb, share_of_m3_volume=pc(x.q.sum() / tot), in_thin_or_wide=pc(x[x.cls.isin(['THIN', 'WIDE'])].q.sum() / x.q.sum()),
                         in_one_sided=pc(x[x.cls == 'ONE_SIDED'].q.sum() / x.q.sum()), m3_share_of_all_maker_volume=pc(x.q.sum() / allq)))
    out.append('### R2 Maker3 passive volume by stamp time (book at P = stamp - 2.5 s)\n\n' + md(pd.DataFrame(rows)))

    def econ(d):
        q = d.q.sum()
        qe = d.q_edge.sum()
        return dict(shares_k=f'{q / 1e3:,.1f}', net_usd=usd(d.net.sum()), net_c=c(100 * d.net.sum() / q), edge_c=c(100 * d.edge.sum() / qe) if qe else '',
                    adverse_c=c(100 * d.out.sum() / qe) if qe else '', m1s_c=c(100 * d.v_1s.sum() / d.q_v_1s.sum()),
                    m10s_c=c(100 * d.v_10s.sum() / d.q_v_10s.sum()), end_c=c(100 * d.v_end.sum() / d.q_v_end.sum()) if d.q_v_end.sum() else '')
    rows = []
    for cls in ['CALM', 'THIN', 'WIDE', 'ONE_SIDED']:
        for grp in ['MAKER3', 'OTHER_TOP10', 'FIELD']:
            d = a[(a.grp == grp) & (a.cls == cls)]
            u = A[(A.clock == 'U') & (A.grp == grp) & (A.cls == cls)]
            rows.append(dict(book_at_P=cls, group=grp, **econ(d), net_c_if_book_at_U=c(100 * u.net.sum() / u.q.sum())))
    out.append('### R3 Passive-fill economics by book class (cents per share; net and adverse to resolution)\n\n' + md(pd.DataFrame(rows)))
    rd = json.loads((H / 'readings.json').read_text())
    rows = []
    for late in ['<270', '>=270']:
        for vol in ['CALM', 'THIN_or_WIDE', 'ONE_SIDED']:
            k = f'P|{late}|{vol}'
            b, dd = rd['b_c_m3_cells'][k], rd['d_m3_minus_field'].get(k, {})
            rows.append(dict(stamp=late, book_at_P=vol, m3_shares_k=f"{b['shares'] / 1e3:,.1f}", m3_markets=b['markets'], m3_net_usd=usd(b['net_usd']),
                             m3_net_c=c(b['net_c']), other_top10_net_c=c(dd.get('other_top10_net_c')), field_net_c=c(dd.get('field_net_c')),
                             m3_day_signs=''.join({1.0: '+', -1.0: '-', None: '.'}[v] for v in b['day_net_signs'].values()),
                             top3_market_share=pc(b['top3_abs_frac'], 0) if b['top3_abs_frac'] == b['top3_abs_frac'] else '',
                             m3_net_c_at_U=c(rd['b_c_m3_cells'][f'U|{late}|{vol}']['net_c'])))
    out.append('### R4 Early vs late, Maker3 against peers (P clock; day signs Oct 24/25/26/27)\n\n' + md(pd.DataFrame(rows)))
    mk = M[(M.role == 'maker') & (M.clock == 'P')]
    rows = []
    for money in ['NEAR', 'EXTREME']:
        for cls in ['CALM', 'THIN', 'WIDE', 'ONE_SIDED']:
            d = mk[mk.tb.isin(LATE) & (mk.money == money) & (mk.cls == cls)]
            if len(d) and d.q.sum() > 0:
                rows.append(dict(price=money, book_at_P=cls, legs=int(d.legs.sum()), markets=d.groupby(['day', 'mid']).ngroups, **econ(d)))
    out.append('### R5 Maker3 late (stamp >= 270 s) passive fills by moneyness (NEAR = 10-90 c)\n\n' + md(pd.DataFrame(rows)))
    L = pd.read_parquet(OUT / 'legs_m3.parquet')
    K = pd.read_parquet(OUT / 'markets.parquet')
    L = L.merge(K[['day', 'mid', 'late_class']], on=['day', 'mid'])
    rows = []
    for tb in TB:
        x = L[L.tb == tb]
        rows.append(dict(window_s=tb, all_markets_net_usd=usd(x.net_res.sum()), volatile_late_markets=usd(x[x.late_class == 'VOLATILE_LATE'].net_res.sum()),
                         calm_late_markets=usd(x[x.late_class == 'CALM_LATE'].net_res.sum()), pinned_late_markets=usd(x[x.late_class == 'PINNED_LATE'].net_res.sum())))
    x = L[~L.tb.isin(TB)]
    rows.append(dict(window_s='pre/post_stamp', all_markets_net_usd=usd(x.net_res.sum()), volatile_late_markets=usd(x[x.late_class == 'VOLATILE_LATE'].net_res.sum()),
                     calm_late_markets=usd(x[x.late_class == 'CALM_LATE'].net_res.sum()), pinned_late_markets=usd(x[x.late_class == 'PINNED_LATE'].net_res.sum())))
    out.append('### R6 Where Maker3 net (both roles, value to resolution) falls in the window\n\n' + md(pd.DataFrame(rows)))
    act = K[K._merge == 'both']
    rows = []
    for cl in ['VOLATILE_LATE', 'CALM_LATE', 'PINNED_LATE', 'MIXED']:
        x = act[act.late_class == cl]
        dsg = ''.join('+' if v > 0 else '-' for v in x.groupby('day').net_res.sum().reindex(['20261024', '20261025', '20261026', '20261027']).fillna(0))
        rows.append(dict(market_class=cl, markets=len(x), decided_by_270s=pc(x.decided270.mean(), 0), net_usd=usd(x.net_res.sum()),
                         round_trips_usd=usd(x.rt_pnl.sum()), residual_usd=usd(x.resid_res.sum()), net_per_market=c(x.net_res.mean()),
                         oct26_net_usd=usd(x[x.day == '20261026'].net_res.sum()), day_signs=dsg))
    out.append('### R7 Markets by their last-minute book (share of 240-299 s seconds), Maker3 whole-market economics\n\n' + md(pd.DataFrame(rows)))
    tk = L[(L.role == 'taker') & (L.t_rel_s < 270)]
    rows = []
    for inv in ['ADD', 'REDUCE', 'FLIP', 'OPEN']:
        x = tk[tk.inv == inv]
        e = np.isfinite(x.edge_P)
        rows.append(dict(take_effect=inv, legs=len(x), shares_k=f'{x.q.sum() / 1e3:,.1f}', net_usd=usd(x.net_res.sum()), net_c=c(100 * x.net_res.sum() / x.q.sum()),
                         fee_c=c(100 * x.fee_res.sum() / x.q.sum()), edge_c=c(100 * x.edge_P[e].sum() / x.q[e].sum()),
                         m1s_c=c(100 * x.v_1s.sum() / x.q[np.isfinite(x.v_1s)].sum()), end_c=c(100 * x.v_end.sum() / x.q[np.isfinite(x.v_end)].sum())))
    out.append(f"### R8 Maker3 taker legs before 270 s (all {len(L[L.role == 'taker'])} taker legs; {len(L[(L.role == 'taker') & (L.t_rel_s >= 270)])} at >= 270 s)\n\n"
               + md(pd.DataFrame(rows)))
    m = L[(L.role == 'maker') & (L.inv != 'SELF')].copy()
    m['ab'] = pd.cut(m.pos_before.abs(), [-1, 1e-9, 50, 200, 500, 1000, 1e9], labels=['0', '0-50', '50-200', '200-500', '500-1000', '>1000']).astype(str)
    m['red'] = m.inv.isin(['REDUCE', 'FLIP'])
    rows = []
    for late, lm in (('<270', m.t_rel_s < 270), ('>=270', m.t_rel_s >= 270)):
        for ab in ['0-50', '50-200', '200-500', '500-1000', '>1000']:
            x = m[lm & (m.ab == ab)]
            e = np.isfinite(x.edge_P)
            rows.append(dict(stamp=late, abs_position_before=ab, legs=len(x), reducing_share_of_shares=pc((x.q * x.red).sum() / x.q.sum()),
                             edge_c_reducing=c(100 * x.edge_P[e & x.red].sum() / x.q[e & x.red].sum()),
                             edge_c_adding=c(100 * x.edge_P[e & ~x.red].sum() / x.q[e & ~x.red].sum())))
    out.append('### R9 Skew proxy (uncontrolled): Maker3 passive fills by position size\n\n' + md(pd.DataFrame(rows)))
    (H / 'report_tables.md').write_text('\n\n'.join(out) + '\n')
    print('\n\n'.join(out))


if __name__ == '__main__':
    main()
