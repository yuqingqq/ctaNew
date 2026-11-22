"""READINGS of ARRIVAL-QUEUE-DECOMPOSITION-v1 (declaration.md section READINGS; frozen with the builder before the run; session aea5bfb9, 2026-11-22).
Reads the frozen run's outputs (hash-checked against its audit.json) and writes the receipt copies and the declared reading tables / predicates;
it selects nothing and fits nothing. Sparse (declared): a cell with fewer than SPARSE_MIN_FILLED states with a fill. Internal control: R1's
per-interval and union counts / values are computed twice (from intervals.csv directly and from cells.csv's queue view summed over its bands)
and must agree, or the readings refuse. No interval estimates (4 UTC days < 5): point estimates only.
Usage (run_light): read_arrival_queue.py (the rehearsal sets AQ_RUN_DIR / AQ_READ_DIR elsewhere and AQ_QUIET=1: no value is printed)"""
import gzip
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path('/home/yuqing/ctaNew'); R = REPO / 'predictfun/reviews/ladder_arrival_queue_20261122'
D = Path(os.environ.get('AQ_RUN_DIR', REPO / 'data/predict_fun/research/ladder_arrival_queue_v1_20261122'))     # the rehearsal points both elsewhere
O = Path(os.environ.get('AQ_READ_DIR', R / 'outputs')); QUIET = os.environ.get('AQ_QUIET') == '1'
H = (0, 50, 100, 1000, 5000, 30000); IVS = ('I0', 'I1', 'I2', 'I3', 'union'); SPARSE_MIN_FILLED = 20; KEY_H = (0, 1000, 5000, 30000)


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def need(cond, msg):
    if not cond:
        raise SystemExit(f'READINGS REFUSED: {msg}')


def load():
    audit = json.loads((D / 'audit.json').read_text())
    need(all(audit['outputs'][n] == sha(D / n) for n in ('states.csv', 'intervals.csv', 'cells.csv')), 'an output differs from its audit.json hash')
    S = pd.read_csv(D / 'states.csv', dtype={'day': str}, float_precision='round_trip'); I_ = pd.read_csv(D / 'intervals.csv', dtype={'day': str}, float_precision='round_trip')
    T = pd.read_csv(D / 'cells.csv', dtype={'day': str, 'side': str, 'band': str}, float_precision='round_trip'); return audit, S, I_, T


def union_direct(I_):
    g = I_.groupby(['day', 'm', 'gid'])
    U = g.agg(fills=('fills', 'sum'), shares=('shares', 'sum'), vEND=('vEND', 'sum'), common=('common', 'all'), risk=('follow_up', lambda x: bool((x != 'not_at_risk').any())))
    for h in H:
        U[f'v{h}'] = g[f'v{h}'].apply(lambda x: x.sum() if x.notna().all() else np.nan); U[f'ok{h}'] = g[f'ok{h}'].all()
    return U.reset_index()


def r1(S, I_, T):
    """The decomposition over all 4,840 states, per interval and the union, computed twice (the control)."""
    U = union_direct(I_); rows = []
    for iv in IVS:
        X_ = U if iv == 'union' else I_[I_.interval == iv].assign(risk=lambda d: d.follow_up != 'not_at_risk'); com = X_[X_.common.astype(bool)]
        r = dict(interval=iv, opportunities=int(len(X_)), at_risk=int(X_.risk.sum()), any_fill=int((X_.fills > 0).sum()), fills=int(X_.fills.sum()), shares=float(X_.shares.sum()),
                 end_net=float(X_.vEND.sum()), common_states=int(len(com)), common_shares=float(com.shares.sum()))
        r['any_fill_rate'] = r['any_fill'] / r['at_risk'] if r['at_risk'] else np.nan
        for h in H:
            r[f'common_net_{h}'] = float(com[f'v{h}'].sum()); r[f'common_per_share_{h}'] = r[f'common_net_{h}'] / r['common_shares'] if r['common_shares'] else np.nan
            nat = X_[X_[f'ok{h}'].astype(bool)]; r[f'native_net_{h}'] = float(nat[f'v{h}'].sum()); r[f'native_shares_{h}'] = float(nat.shares.sum())
        rows.append(r)
    A = pd.DataFrame(rows)
    q = T[(T.view == 'queue') & (T.day == 'all') & (T.side == 'all')]; bad = []
    for iv in IVS:
        for coh in ('common', 'native'):
            g = q[(q.interval == iv) & (q.cohort == coh)]; a = A[A.interval == iv].iloc[0]
            same = int(g.opportunities.sum()) == a.opportunities and int(g.at_risk.sum()) == a.at_risk and int(g.any_fill.sum()) == a.any_fill and int(g.fills.sum()) == a.fills \
                and np.isclose(g.shares.sum(), a.shares) and np.isclose(g.end_net.sum(), a.end_net)
            for h in H:
                same = same and np.isclose(g[f'h{h}_net'].sum(), a[f'{coh}_net_{h}'], rtol=1e-9, atol=1e-6)
            bad += [] if same else [(iv, coh)]
    need(not bad, f'R1 differs between intervals.csv and cells.csv: {bad}')
    UC = U[U.common.astype(bool)].set_index(['day', 'm', 'gid']); IC = I_.set_index(['day', 'm', 'gid']).loc[UC.index]; frac = {}
    for h in KEY_H:
        tot = float(UC[f'v{h}'].sum()); frac[str(h)] = {iv: float(IC[IC.interval == iv][f'v{h}'].sum()) / tot if tot else np.nan for iv in IVS[:4]}
    frac['shares'] = {iv: float(IC[IC.interval == iv].shares.sum()) / float(UC.shares.sum()) for iv in IVS[:4]}
    pred = dict(union_common_states=int(len(UC)), loss_at_1s_every_interval={iv: bool(A[A.interval == iv].iloc[0]['common_net_1000'] < 0) for iv in IVS},
                persists_30s_below_1s={iv: bool(A[A.interval == iv].iloc[0]['common_net_30000'] < A[A.interval == iv].iloc[0]['common_net_1000']) for iv in IVS},
                union_fraction_by_interval=frac)
    return A, pred


def cells(T, view, ivs=IVS, coh='common'):
    """One view's all-day / all-side cells (common cohort) with the declared columns and the sparse flag."""
    g = T[(T.view == view) & (T.day == 'all') & (T.side == 'all') & (T.cohort == coh) & T.interval.isin(ivs)].copy()
    out = g[['band', 'interval', 'opportunities', 'at_risk', 'any_fill', 'any_fill_rate', 'fills', 'shares', 'partial', 'filled_out', 'end_net']].copy()
    for h in KEY_H:
        out[f'per_share_{h}'] = g[f'h{h}_per_share']; out[f'per_opportunity_{h}'] = g[f'h{h}_per_opportunity']
    vq = g['h1000_valid_shares'].where(g['h1000_valid_shares'] > 0)
    out['fill_positive_per_share_1000'] = g['h1000_fill_positive'] / vq; out['fill_negative_per_share_1000'] = g['h1000_fill_negative'] / vq
    out['common_states'] = g['h1000_valid']; out['sparse'] = out.any_fill < SPARSE_MIN_FILLED; return out.reset_index(drop=True)


def contrasts(T):
    """Within each queue band and interval: q5 / positive_below_q5 minus non_adverse per share at 1 s and 30 s (common), only between non-sparse
    cells; per day the same contrast where both day cells are non-sparse, and the count of days sharing the pooled sign."""
    g = T[(T.view == 'queue_pressure') & (T.cohort == 'common')].copy(); g['qb'] = g.band.str.split('|').str[0]; g['bb'] = g.band.str.split('|').str[1]; out = []
    for (qb, iv), x in g[g.side == 'all'].groupby(['qb', 'interval']):
        for alt in ('q5', 'positive_below_q5'):
            for h in (1000, 30000):
                pooled = x[x.day == 'all'].set_index('bb'); r = dict(queue_band=qb, interval=iv, band=alt, horizon_ms=h)
                ok_ = all(b in pooled.index and pooled.loc[b, 'any_fill'] >= SPARSE_MIN_FILLED for b in (alt, 'non_adverse'))
                r['status'] = 'evaluable' if ok_ else 'insufficient_support'
                r['alt_filled_states'] = int(pooled.loc[alt, 'any_fill']) if alt in pooled.index else 0; r['ref_filled_states'] = int(pooled.loc['non_adverse', 'any_fill']) if 'non_adverse' in pooled.index else 0
                r['diff_per_share'] = float(pooled.loc[alt, f'h{h}_per_share'] - pooled.loc['non_adverse', f'h{h}_per_share']) if ok_ else np.nan
                days = []
                for day, y in x[x.day != 'all'].groupby('day'):
                    y = y.set_index('bb')
                    if ok_ and all(b in y.index and y.loc[b, 'any_fill'] >= SPARSE_MIN_FILLED for b in (alt, 'non_adverse')):
                        days.append(float(y.loc[alt, f'h{h}_per_share'] - y.loc['non_adverse', f'h{h}_per_share']))
                r['days_evaluable'] = len(days); r['days_same_sign'] = int(sum(np.sign(d) == np.sign(r['diff_per_share']) for d in days)) if ok_ else 0; out.append(r)
    return pd.DataFrame(out)


def order_of(df, iv, h, bands):
    x = df[(df.interval == iv) & df.band.isin(bands) & ~df['sparse']].sort_values(f'per_share_{h}')
    return '<'.join(x.band) if len(x) == len(bands) else 'insufficient_support'


def venue(S):
    va = S[S.venue_status == 'available']; d = va.client_minus_venue.dropna()
    return dict(status=S.venue_status.value_counts().to_dict(), n=int(len(d)), equal=int((d.abs() <= 1e-9).sum()), above=int((d > 1e-9).sum()), below=int((d < -1e-9).sum()),
                p10=float(d.quantile(.1)) if len(d) else None, p50=float(d.quantile(.5)) if len(d) else None, p90=float(d.quantile(.9)) if len(d) else None,
                median_abs=float(d.abs().median()) if len(d) else None, notice_gap_agrees=int(va.notice_gap_agrees.astype(str).eq('True').sum()), known_remaining_agree=int(S.R_agrees.sum()),
                by_side={str(s): dict(n=int(len(x)), p50=float(x.client_minus_venue.median())) for s, x in va.groupby('side')})


def main():
    audit, S, I_, T = load(); O.mkdir(parents=True, exist_ok=True); need(not (O / 'readings.json').exists(), 'the readings exist (write-once)')
    for n in ('states.csv', 'intervals.csv', 'cells.csv'):
        with open(O / f'{n}.gz', 'wb') as fh, gzip.GzipFile(fileobj=fh, mode='wb', mtime=0) as g:
            g.write((D / n).read_bytes())
        need(hashlib.sha256(gzip.decompress((O / f'{n}.gz').read_bytes())).hexdigest() == audit['outputs'][n], f'{n}: the receipt copy differs')
    (O / 'audit.json').write_bytes((D / 'audit.json').read_bytes())
    A, p1 = r1(S, I_, T); A.to_csv(O / 'r1_decomposition.csv', index=False)
    tabs = dict(r2_queue=cells(T, 'queue'), r3_queue_pressure=cells(T, 'queue_pressure'), r3b_queue_pressure_flow=cells(T, 'queue_pressure_flow'), r4_progress=cells(T, 'progress'),
                r4b_pressure_flow=cells(T, 'pressure_flow'))
    for n, x in tabs.items():
        x.to_csv(O / f'{n}.csv', index=False)
    C_ = contrasts(T); C_.to_csv(O / 'r3_contrasts.csv', index=False); q = tabs['r2_queue']; pg = tabs['r4_progress']
    qb3 = ['front', 'within_R', 'beyond_R']
    rd = dict(asof_outputs=audit['asof'], states=int(len(S)), sparse_min_filled=SPARSE_MIN_FILLED, r1=p1,
              r2_queue_order={iv: {str(h): order_of(q, iv, h, qb3) for h in (1000, 30000)} for iv in IVS},
              r2_queue_any_fill_rate={iv: {b: float(q[(q.interval == iv) & (q.band == b)].any_fill_rate.iloc[0]) for b in qb3 if len(q[(q.interval == iv) & (q.band == b)])} for iv in IVS},
              r3_evaluable=int((C_.status == 'evaluable').sum()), r3_contrasts=int(len(C_)),
              r3_evaluable_negative_all_days=int(((C_.status == 'evaluable') & (C_.diff_per_share < 0) & (C_.days_evaluable > 0) & (C_.days_same_sign == C_.days_evaluable)).sum()),
              r4_progress_order={iv: {str(h): order_of(pg, iv, h, ['advanced', 'unchanged']) for h in (1000, 30000)} for iv in IVS},
              r5_venue=venue(S), multiplicity=audit['summary']['multiplicity'], intervals_quoted='none (4 UTC days < 5): point estimates only')
    (O / 'readings.json').write_text(json.dumps(rd, indent=1, default=str) + '\n')
    print(f'READINGS WRITTEN: {len(tabs) + 2} tables, R1 control passed' if QUIET else json.dumps(rd, indent=1, default=str)[:6000])


if __name__ == '__main__':
    main()
