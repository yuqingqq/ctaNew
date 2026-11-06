"""Stage 1 of the research flow (docs §50; redeclared in STATUS.yml 2026-11-04 before any run, user: Predict's mid reaches us
~50 ms late, so it cannot be the anchor): is there a fresh, Binance-based fair price good enough to anchor on? 8 read days, live
0-240 s, rows every 500 ms (rows_rl_d2, Tokyo clock with 2 ms decision).

Candidates (fresh: known ~3.5 ms after Binance), in cents: A2 p_bn (Binance mid x 60 s median Chainlink-relay / Binance ratio vs
the start price, 30 min realised 1 s volatility, time left); A4 Binance-only logistic recalibration fitted leaving one day out on
[z = ln(S basis / K) / (sigma sqrt(tau_rem)), sigma sqrt(tau_rem), basis_bps, tau_rem]. Benchmark only: A0 Predict mid as we know
it at tau (~50 ms late).
(c) Nowcast: the venue mid at tau (already set, not yet received) minus A0, regressed on A - A0.
(a) Settlement: Brier score (A / 100 - outcome)^2 per day and by time left; days better than A0; day-clustered 95% CI (G = 8).
(b) Convergence: Predict's mid change at +1 s / +10 s / +60 s (venue mid at updateTimestampMs <= tau + h minus the known mid)
regressed on the gap A - A0 per day: slope (share of the gap that closes) and R^2.
Gate (computed): good anchor if the +10 s slope >= 0.3 with CI excluding 0 AND Brier <= A0's + 0.005 on >= 6/8 days. Statuses: rows without p_bn (no volatility), without a mid at a horizon (past the window), dropped and counted.
Usage: PF_RL_DDEC=2 python3 -m predictfun.research.pf_anchor_eval
"""
import os
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from . import pf_io as io

DAYS = ['20261023', '20261024', '20261025', '20261026', '20261027', '20261031', '20261101', '20261102']
OUTD = f'{io.OUT}/step4/anchor_eval'; os.makedirs(OUTD, exist_ok=True)
H = (0.0, 1000.0, 10000.0, 60000.0); TL = [(60, 120), (120, 180), (180, 240), (240, 301)]
st = dict(rows=0, no_pbn=0, no_outcome=0)


def logit(p):
    p = np.clip(p, 0.005, 0.995); return np.log(p / (1 - p))


parts = []
for d in DAYS:
    S1 = f'{io.OUT}/step1/{d}'
    r = pd.read_parquet(f'{io.OUT}/step5/{d}/rows_rl_d2.parquet', columns=['mid', 'tau', 't_rel', 'tau_rem', 'pf_mid', 'pf_micro_dev', 'p_bn', 'sigma', 'basis_bps', 'y1000'])
    keep = ((r.t_rel < 240) & r.pf_mid.notna()).values & ((np.floor(r.tau.values / 50.0).astype(np.int64) % 10) == 0)
    r = r[keep].reset_index(drop=True); st['rows'] += len(r)
    reg = pd.read_parquet(f'{S1}/registry.parquet', columns=['mid', 'outcome']).set_index('mid')
    r['y'] = r.mid.map(lambda m: 1.0 if reg.outcome.get(m) == 'Up' else (0.0 if reg.outcome.get(m) == 'Down' else np.nan))
    bk = pd.concat([pd.read_parquet(f'{S1}/books/{f}', columns=['mid', 't_ms', 'b1_p', 'a1_p']) for f in sorted(os.listdir(f'{S1}/books'))])
    bk = bk[(bk.b1_p >= 0) & (bk.a1_p > bk.b1_p)].sort_values(['mid', 't_ms'], kind='stable')
    BY = {m: (g.t_ms.values.astype(float), (g.b1_p.values + g.a1_p.values) / 2.0) for m, g in bk.groupby('mid', sort=False)}; del bk
    for h in (0.0,) + H[2:]:
        out = np.full(len(r), np.nan)
        for mid, idx in r.groupby('mid', sort=False).indices.items():
            v = BY.get(mid)
            if v is None:
                continue
            tv, mv = v; j = np.searchsorted(tv, r.tau.values[idx] + h, side='right') - 1
            out[idx] = np.where(j >= 0, mv[np.clip(j, 0, None)], np.nan) - r.pf_mid.values[idx]
        r[f'y{int(h)}'] = out
    r['day'] = d; parts.append(r); del BY
    io.mem_check(f'day {d}')
R = pd.concat(parts, ignore_index=True)
st['no_pbn'] = int(R.p_bn.isna().sum()); st['no_outcome'] = int(R.y.isna().sum())
R = R[R.p_bn.notna() & R.y.notna()].reset_index(drop=True)
R['A0'] = R.pf_mid; R['A2'] = R.p_bn
sst = R.sigma * np.sqrt(R.tau_rem); z = logit(R.A2 / 100)                      # monotone in ln(S basis / K) / (sigma sqrt(tau))
X = np.column_stack([z, sst, R.basis_bps.fillna(0), R.tau_rem])
R['A4'] = np.nan
for d in DAYS:
    tr = (R.day != d).values
    m = LogisticRegression(C=1.0, max_iter=1000).fit(X[tr], R.y.values[tr]); R.loc[~tr, 'A4'] = 100 * m.predict_proba(X[~tr])[:, 1]
R.drop(columns=['pf_micro_dev']).to_parquet(f'{OUTD}/rows.parquet', index=False)
CAND = ['A0', 'A2', 'A4']; LAB = dict(A0='Predict mid (late)', A2='p_bn (Binance)', A4='Binance recal.')
print(f'statuses {st}; rows used {len(R)}')
print('\n(a) settlement Brier score (lower is better), mean over days; vs A0 with day-clustered 95% CI and days better')
br = {c: np.array([np.mean((R[R.day == d][c] / 100 - R[R.day == d].y) ** 2) for d in DAYS]) for c in CAND}
gate = {}
for c in CAND:
    line = f'  {c} {LAB[c]:16s} {br[c].mean():.4f}'
    if c != 'A0':
        dd = br[c] - br['A0']; se = dd.std(ddof=1) / np.sqrt(len(dd)); lo, hi = dd.mean() - 2.365 * se, dd.mean() + 2.365 * se
        ok = int((dd <= 0.005).sum()); gate[f'{c}_brier_ok'] = bool(ok >= 6)
        line += f'  vs A0 {dd.mean():+.4f} ({lo:+.4f}..{hi:+.4f}), better on {int((dd < 0).sum())}/8 days, within +0.005 on {ok}/8'
    print(line)
print('  by time left (s): ' + '  '.join(f'[{a},{b})' for a, b in TL))
for c in CAND:
    print(f'  {c} ' + '  '.join(f"{np.mean(((R[c] / 100 - R.y) ** 2)[(R.tau_rem >= a) & (R.tau_rem < b)]):.4f}" for a, b in TL))
print('\n(b) convergence: Predict mid change at +h regressed on the gap A - A0 (slope = share of the gap that closes), mean over days')
conv = {}
for c in ('A2', 'A4'):
    g = (R[c] - R.A0).values
    for h in H:
        col = 'y1000' if h == 1000.0 else f'y{int(h)}'
        sl, r2 = [], []
        for d in DAYS:
            m = (R.day == d).values & np.isfinite(R[col].values) & np.isfinite(g)
            x, y = g[m], R[col].values[m]; b = np.sum(x * y) / np.sum(x * x); sl.append(b); r2.append(1 - np.sum((y - b * x) ** 2) / np.sum((y - y.mean()) ** 2))
        sl = np.array(sl); se = sl.std(ddof=1) / np.sqrt(len(sl)); conv[f'{c}|{int(h)}'] = dict(slope=float(sl.mean()), lo=float(sl.mean() - 2.365 * se), hi=float(sl.mean() + 2.365 * se), r2=float(np.mean(r2)))
    print(f"  {c} {LAB[c]:16s} " + '  '.join(f"{'now (unseen)' if h == 0 else '+' + str(int(h / 1000)) + ' s'} slope {conv[f'{c}|{int(h)}']['slope']:+.3f} ({conv[f'{c}|{int(h)}']['lo']:+.3f}..{conv[f'{c}|{int(h)}']['hi']:+.3f}) R2 {conv[f'{c}|{int(h)}']['r2']:.3f}" for h in H))
    gate[f'{c}_leads'] = bool(conv[f'{c}|10000']['slope'] >= 0.3 and conv[f'{c}|10000']['lo'] > 0)
good = {c: bool(gate[f'{c}_leads'] and gate[f'{c}_brier_ok']) for c in ('A2', 'A4')}
print('\ngate (computed):', gate, '-> good anchor:', good)
io.write_json(f'{OUTD}/report.json', dict(statuses=st, brier={c: br[c].tolist() for c in CAND}, convergence=conv, gate=gate, good_anchor=good))
