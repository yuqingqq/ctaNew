"""COMPUTED CLAIMS of the MAKER-TRACE-RECONSTRUCTION-v1 results receipt (session aea5bfb9, 2026-11-23). Every statement report.md makes is
evaluated here from the frozen report's outputs (outputs/*.csv, readings.json) and the checker receipt (checks_*.json); report.md quotes
claims.json, never a typed number. Units: USD; c_ = cents per share; pooled = Oct 24-27 (4 days: point estimates, no intervals, rule 8).
Write-once. Usage: claims.py"""
import glob
import json
import sys
from pathlib import Path

import pandas as pd

D = Path('/home/yuqing/ctaNew/predictfun/reviews/maker_trace_20261122'); O = D / 'outputs'; OUTP = D / 'claims.json'
DAYS = ['20261024', '20261025', '20261026', '20261027']


def need(ok, msg):
    if not ok:
        raise SystemExit(f'claims: {msg}')


def main():
    need(not OUTP.exists(), f'{OUTP} exists (write-once)')
    R = json.loads((O / 'readings.json').read_text())
    chk = sorted(glob.glob(str(D / 'checks_*.json'))); need(len(chk) >= 1, 'no checker receipt'); CK = json.loads(Path(chk[-1]).read_text())
    E = pd.read_csv(O / 'econ_identity_day.csv', dtype={'day': str}); S = pd.read_csv(O / 'econ_segments.csv', dtype={'day': str})
    A = pd.read_csv(O / 'econ_address_day.csv', dtype={'day': str}); N = pd.read_csv(O / 'resolution_null.csv', dtype={'day': str})
    W = pd.read_csv(O / 'rewards_market.csv', dtype={'day': str}); P = pd.read_csv(O / 'rank_persistence.csv', dtype={'day': str})
    C = {}
    # instruments
    C['checker'] = dict(file=Path(chk[-1]).name, passed=sum(bool(v) for v in CK['checks'].values()), n=len(CK['checks']),
                        failed=sorted(k for k, v in CK['checks'].items() if not v),
                        reparse_failed=sorted(f'{m}:{k}' for m, r in CK.get('reparse', {}).items() for k, v in r['checks'].items() if not v))
    t1, t2 = R['R1']['T1'], R['R1']['T2']; ck2 = CK['details']['priority_tests_T1_T2']
    C['R1'] = dict(T1=dict(violations=t1['violations'], den=t1['den'], rate=t1['rate'], certain=t1['violations_certain'], reading=t1['reading']),
                   T2=dict(violations=t2['violations'], den=t2['den'], rate=t2['rate'], certain=t2['violations_certain'], reading=t2['reading'],
                           checker_recomputed_records=ck2.get('T2_records'), checker_rate=ck2.get('T2_records', 0) / t2['den']),
                   both_below_1pct_even_with_checker_counts=bool(t1['rate'] <= .01 and ck2.get('T2_records', 0) / t2['den'] <= .01))
    g = R['R2']; C['R2'] = dict(defects_total=g['defects_total'], scored_own_orders=g['n']['scored_own_orders'],
                                coverage_orders=g['coverage_of_filled_own_orders']['S_FIFO']['orders'])
    # economics, pooled
    def row(s, grp, day='pooled'):
        x = E[(E.id_set == s) & (E.grp == grp) & (E.day == day)]; need(len(x) == 1, f'econ row {s}/{grp}/{day}'); return x.iloc[0]
    for s in ('exante_sh', 'exante_pnl'):
        for grp in ('in', 'field', 'nonmaker'):
            r = row(s, grp); sh = r.sh_mk + r.sh_tk
            C[f'{s}.{grp}'] = dict(n_addr=int(r.n_addr), sh_mk=round(r.sh_mk), sh_tk=round(r.sh_tk), maker_frac=round(r.maker_frac, 3) if sh else None,
                                  net=round(r.net_res, 2), gross=round(r.gross_res, 2), fees=round(r.fee_res, 2), rt=round(r.rt_pnl_res, 2), resid=round(r.resid_res, 2),
                                  net_maker_legs=round(r.net_res_mk, 2), net_taker_legs=round(r.net_res - r.net_res_mk, 2), reb25=round(r.reb25_res, 2),
                                  c_net=round(100 * r.net_res / sh, 4) if sh else None,
                                  net_by_day={d: round(row(s, grp, d).net_res, 2) for d in DAYS}, sign_consistent=bool(r.sign_consistent),
                                  identity_gross_eq_rt_plus_resid=bool(abs(r.gross_res - (r.rt_pnl_res + r.resid_res)) < 1e-6 * max(1, abs(r.gross_res))),
                                  identity_net_eq_gross_minus_fees=bool(abs(r.net_res - (r.gross_res - r.fee_res)) < 1e-6 * max(1, abs(r.gross_res))))
    tot = sum(row('exante_sh', g).net_res for g in ('in', 'field', 'nonmaker')); fees = sum(row('exante_sh', g).fee_res for g in ('in', 'field', 'nonmaker'))
    C['zero_sum'] = dict(sum_net_all_addresses=round(tot, 2), sum_fees=round(fees, 2), net_plus_fees=round(tot + fees, 2))
    # segments (maker legs unless stated): per-share ordering checks, pooled and per day
    def seg(s, grp, role, dim, day):
        x = S[(S.id_set == s) & (S.grp == grp) & (S.role == role) & (S.dim == dim) & (S.day == day)].set_index('bucket'); return x
    def cnet(x, b):
        return float(100 * x.loc[b, 'net_res'] / x.loc[b, 'sh']) if b in x.index and x.loc[b, 'sh'] > 0 else float('nan')
    def c1s(x, b):
        return float(100 * x.loc[b, 'v_1s'] / x.loc[b, 'sh_1s']) if b in x.index and x.loc[b, 'sh_1s'] > 0 else float('nan')
    for grp in ('in', 'field'):
        out = {}
        for day in ['pooled'] + DAYS:
            dp = seg('exante_sh', grp, 'maker', 'depth', day); sw = seg('exante_sh', grp, 'maker', 'sweep', day); ts = seg('exante_sh', grp, 'maker', 'tseg', day)
            tk = seg('exante_sh', grp, 'taker', 'all', day)
            dv = [cnet(dp, b) for b in ('d0', 'd1', 'd2', 'd3p')]
            out[day] = dict(depth_c_net=[round(v, 4) for v in dv], depth_monotone_increasing=bool(all(a < b for a, b in zip(dv, dv[1:]))),
                            depth_d0_below_deeper=bool(dv[0] < min(dv[1:])),
                            sweep_c_net=round(cnet(sw, 'sweep'), 4), single_c_net=round(cnet(sw, 'single_price'), 4), sweep_gt_single=bool(cnet(sw, 'sweep') > cnet(sw, 'single_price')),
                            early_c_net=round(cnet(ts, 'w000_060'), 4), late_c_net=round(cnet(ts, 'w240_300'), 4), early_gt_late=bool(cnet(ts, 'w000_060') > cnet(ts, 'w240_300')),
                            taker_c_1s=round(c1s(tk, tk.index[0]) if len(tk) else float('nan'), 4), taker_c_net=round(cnet(tk, tk.index[0]) if len(tk) else float('nan'), 4),
                            taker_1s_positive=bool(len(tk) and c1s(tk, tk.index[0]) > 0))
        out['days_count'] = {k: sum(bool(out[d][k]) for d in DAYS) for k in ('depth_d0_below_deeper', 'depth_monotone_increasing', 'sweep_gt_single', 'early_gt_late', 'taker_1s_positive')}
        C[f'segments.exante_sh.{grp}'] = out
    # rebates and rewards
    r = row('exante_sh', 'in'); f = row('exante_sh', 'field')
    C['rebate_scenario'] = dict(note='advertised 25% taker-fee rebate, unverified as paid; shown beside, never inside, net P&L',
                                in_reb25=round(r.reb25_res, 2), in_net=round(r.net_res, 2), in_net_maker_legs=round(r.net_res_mk, 2), in_ratio_reb_to_net=round(r.reb25_res / r.net_res, 3),
                                field_reb25=round(f.reb25_res, 2), field_net=round(f.net_res, 2), field_ratio_reb_to_net=round(f.reb25_res / f.net_res, 3))
    pool = W[W.day.isin(DAYS)].groupby('day').pool_units.sum()
    mk_net = {d: round(row('exante_sh', 'in', d).net_res + row('exante_sh', 'field', d).net_res, 2) for d in DAYS}
    C['reward_pool'] = dict(units='unverified (per-window schedule hourlyRate); per-maker rewards unidentifiable', pool_units_by_day={d: float(pool.get(d, 0)) for d in DAYS},
                            all_makers_net_usd_by_day=mk_net)
    # resolution luck (exante_sh in)
    n = N[(N.id_set == 'exante_sh') & (N.grp == 'in') & (N.day.isin(DAYS))].set_index('day')
    C['resolution_null.exante_sh.in'] = dict(percentile={d: float(n.loc[d, 'percentile']) for d in DAYS if d in n.index}, n_draws=int(n.n_draws.iloc[0]),
                                             within_central_95=bool(all(.025 <= float(n.loc[d, 'percentile']) <= .975 for d in DAYS if d in n.index)))
    # addresses (exante_sh members, pooled)
    a = A[(A.day == 'pooled') & (A.in_exante_sh.astype(str) == 'True')].copy(); need(len(a) >= 1, 'no exante_sh member rows')
    byday = A[(A.day.isin(DAYS)) & (A.in_exante_sh.astype(str) == 'True')].pivot_table(index='addr', columns='day', values='net_res', aggfunc='sum')
    pos_all = [x for x in byday.index if (byday.loc[x].reindex(DAYS).fillna(0) > 0).all()]
    a['pure_maker'] = a.maker_frac >= .999
    C['addresses.exante_sh'] = dict(n=len(a), n_net_positive_pooled=int((a.net_res > 0).sum()), n_positive_every_day=len(pos_all),
                                    positive_every_day=[dict(addr=x, maker_frac=round(float(a.set_index('addr').loc[x, 'maker_frac']), 3),
                                                             net=round(float(a.set_index('addr').loc[x, 'net_res']), 2), rt=round(float(a.set_index('addr').loc[x, 'rt_pnl_res']), 2),
                                                             resid=round(float(a.set_index('addr').loc[x, 'resid_res']), 2)) for x in pos_all],
                                    n_pure_makers=int(a.pure_maker.sum()), net_range=[round(float(a.net_res.min()), 2), round(float(a.net_res.max()), 2)])
    ps = P[(P.kind == 'member_summary') & (P.metric == 'sh') & (P.in_exante_sh.astype(str) == 'True')]
    C['rank_persistence.exante_sh'] = dict(mean_days_in_daily_top10=round(float(ps.n_days_top10.mean()), 2), n=len(ps))
    OUTP.write_text(json.dumps(C, indent=1, default=str) + '\n'); print(json.dumps({k: C[k] for k in ('checker', 'R1', 'R2')}, indent=1, default=str)[:1500])


if __name__ == '__main__':
    main()
