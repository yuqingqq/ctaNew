"""SEVERE EVENTS AMONG 'LOW-RISK' DECISIONS: IS THERE INFORMATION? (2026-11-17, session aea5bfb9; USER "Yes go" on: a bounded test of
whether any available causal input separates the decisions that turn severe from the safe ones among those the model calls low-risk).
Declared before any run; descriptive with declared predicates; no model or controller change.

POPULATION: forward days Oct24-27, each decision's past-only P(severe) from the RUNNING-ERRORS expanding folds (head full_l2_p; day D from
the model fitted on the days before D); LOW-RISK band = p < 0.05 (SEVERE-SUPPORT: 30,080 decisions, 675 severe events - reproduced).
Severe = loss beyond 30 c within the 1 s action value (the joint-moment label).
INFORMATION SETS:
 P    the global model's own p (the baseline ranking).
 M87  an in-band LightGBM (pinned parameters, 200 rounds) on the 87 inputs + p, trained past-only on the band decisions of the forward days
      before D (Oct25 <- Oct24; Oct26 <- Oct24-25; Oct27 <- Oct24-26).
 M87+ M87's inputs + the NEW causal candidates: (a) UTC hour; (b) this market's earlier decisions closed >= 2 s before t (decided at or before
      t - 3 s; optimistic about data delays): count, severe count, mean realized loss; (c) the cross-market running severe O/E of RUNNING-ERRORS
      (same-day decisions decided in [t - 300 s - 60 min, t - 300 s], (observed + 1) / (sum p + 1)).
 Univariate: each new candidate's within-band AUC (oriented, pooled Oct24-27) and its sign by day.
READINGS (test days Oct25-27; AUC within the band):
 I1 EXISTING-INPUTS-UNDERUSED: M87's pooled AUC >= P's + 0.02 AND M87 above P on >= 2 of 3 days -> YES, else NOT-SHOWN.
 I2 NEW-INFORMATION: M87+'s pooled AUC >= M87's + 0.02 AND M87+ above M87 on >= 2 of 3 days -> YES, else NOT-SHOWN.
 Printed: log loss of each set, the in-band models' top inputs, the univariate AUCs.
CONTROLS (any failure -> no output): band counts reproduce SEVERE-SUPPORT (30,080 / 675); in-band training days precede the test day; the
 same-market history uses only decisions decided at or before t - 3 s (planted error: a history reaching past t must be refused); the
 cross-market ratio window ends at or before t - 300 s.
LIMITS: consumed days; three test days (no interval, rule 8); the earlier public-tape / other-venue test (<= +0.002 AUC, FINDINGS) is not
 repeated; simulated labels; the neutral decision population.
Usage (run_light, PF_SLOT=claude): pf_severe_info.py selftest | run
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pf_gain_loss as GL                                 # noqa: E402
import pf_gain_loss_running as RN                         # noqa: E402

J, JF, C, pd = GL.J, GL.JF, GL.C, GL.pd
OUT = GL.OUT / 'severe_info'
BAND = 0.05
HIST_LAG = 3000.
NEW = ['x_hour', 'x_mkt_n', 'x_mkt_sev', 'x_mkt_loss', 'x_day_oe']
MUT_HIST_LEAK = 0                                         # planted error (selftest only)


def market_history(z):
    """Per decision: earlier same-market decisions decided at or before t - HIST_LAG - their count, severe count and mean loss."""
    n = np.zeros(len(z)); sv = np.zeros(len(z)); ls = np.full(len(z), np.nan); last_t = np.full(len(z), -np.inf)
    lag = -1000. if MUT_HIST_LEAK else HIST_LAG
    for _, g in z.groupby('m'):
        ix = g.index.to_numpy(); t = g.decision_ms.to_numpy(float); o = np.argsort(t, kind='stable'); ix, t = ix[o], t[o]
        sev = z.loc[ix, 'sev'].to_numpy(); loss = z.loc[ix, 'loss'].to_numpy()
        cs_s = np.r_[0., np.cumsum(sev)]; cs_l = np.r_[0., np.cumsum(loss)]
        hi = np.searchsorted(t, t - lag, side='right')
        n[ix] = hi; sv[ix] = cs_s[hi]; ls[ix] = np.where(hi > 0, cs_l[hi] / np.maximum(hi, 1), np.nan)
        last_t[ix] = np.where(hi > 0, t[np.maximum(hi - 1, 0)], -np.inf)
    ok = bool(np.all(last_t <= z.decision_ms.to_numpy(float) - HIST_LAG + 1e-9))
    return n, sv, ls, ok


def selftest():
    z = pd.DataFrame(dict(m=[1, 1, 1, 2], decision_ms=[0., 1000., 5000., 0.], sev=[1., 0., 0., 1.], loss=[40., 0., 5., 50.]))
    n, sv, ls, ok = market_history(z)
    bad = [] if (list(n) == [0, 0, 2, 0] and list(sv) == [0, 0, 1, 0] and ok and abs(ls[2] - 20.) < 1e-12) else [f'history {n} {sv} {ls} {ok}']
    globals()['MUT_HIST_LEAK'] = 1
    try:
        caught = not market_history(z)[3]
    finally:
        globals()['MUT_HIST_LEAK'] = 0
    if bad or not caught:
        sys.exit(f'SEVERE-INFO SELFTEST FAILED -> no output: {bad}, planted leak caught {caught}')
    print('severe-info selftests passed; planted history leak caught', flush=True)


def auc(y, s):
    y = np.asarray(y, float); s = np.asarray(s, float); ok = np.isfinite(s); y, s = y[ok], s[ok]
    n1 = int(y.sum()); n0 = len(y) - n1
    if not n1 or not n0:
        return float('nan')
    r = pd.Series(s).rank(method='average').to_numpy()
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def logloss(y, p):
    p = np.clip(np.asarray(p, float), 1e-9, 1 - 1e-9); return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def run():
    selftest(); start = time.time(); OUT.mkdir(parents=True, exist_ok=False)
    import lightgbm as lgb
    fr = RN.load(); feats = GL.feature_list('full'); frames = []
    for D in RN.PRED_DAYS:
        z = fr[D].copy().reset_index(drop=True)
        z['p'] = lgb.Booster(model_file=str(RN.OUT / f'fold_{D}' / 'full_l2_p.txt')).predict(z[feats].to_numpy(np.float32), num_threads=1)
        day0 = float(pd.Timestamp(f'{D[:4]}-{D[4:6]}-{D[6:]}', tz='UTC').value // 10 ** 6)
        z['x_hour'] = ((z.decision_ms - day0) // 3600000).astype(float)
        n, sv, ls, ok = market_history(z); C.require(ok, 'same-market history reaches past t - 3 s')
        z['x_mkt_n'] = n; z['x_mkt_sev'] = sv; z['x_mkt_loss'] = ls
        zs = z.sort_values('decision_ms'); t = zs.decision_ms.to_numpy(float)
        C.require(RN.window_ok(t, RN.LAG, RN.WINDOW), 'cross-market window reaches past t - 300 s')
        nn = RN.trailing(t, zs.sev.to_numpy(), RN.LAG, RN.WINDOW); sp = RN.trailing(t, zs.p.to_numpy(), RN.LAG, RN.WINDOW)
        z.loc[zs.index, 'x_day_oe'] = (nn + 1.) / (sp + 1.); frames.append(z)
    Z = pd.concat(frames, ignore_index=True); B = Z[Z.p < BAND].copy()
    C.require(len(B) == 30080 and int(B.sev.sum()) == 675, f'band population differs from SEVERE-SUPPORT ({len(B)}, {int(B.sev.sum())})')
    asof = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    L = [f'SEVERE EVENTS AMONG LOW-RISK DECISIONS (past-only p < {BAND}; no model change); as of {asof}',
         f'controls: band {len(B):,} decisions / {int(B.sev.sum())} severe = SEVERE-SUPPORT; histories end at or before t - 3 s; cross-market window at or before t - 300 s; '
         'in-band training precedes each test day; planted leak refused', '',
         'PART 1: univariate within-band AUC of each new causal candidate (oriented: > 0.5 = higher value, more severe), pooled Oct24-27 | by day']
    for c in ['p'] + NEW:
        a = auc(B.sev, B[c]); L.append(f'  {c:<11} {a:.3f} | ' + ' '.join(f'{d} {auc(g.sev, g[c]):.3f}' for d, g in B.groupby('day')))
    # in-band models, past-only on forward days
    days = list(RN.PRED_DAYS); res = {}; imps = {}
    for D in days[1:]:
        tr = B[B.day < D]; te = B[B.day == D]; C.require(tr.day.max() < D, 'in-band training must precede the test day')
        for name, cols in (('M87', feats + ['p']), ('M87+', feats + ['p'] + NEW)):
            m = lgb.train({**C.V.PARAMS, 'objective': 'binary'}, lgb.Dataset(tr[cols].to_numpy(np.float32), label=tr.sev.to_numpy(), feature_name=cols), num_boost_round=200)
            s = m.predict(te[cols].to_numpy(np.float32), num_threads=1); res[(D, name)] = s
            g = m.feature_importance(importance_type='gain'); imps[(D, name)] = sorted(zip(cols, g / g.sum()), key=lambda x: -x[1])[:8]
        res[(D, 'P')] = te.p.to_numpy()
    L += ['', 'PART 2: within-band AUC and log loss on the test days (in-band models trained past-only on earlier forward days): P | M87 | M87+']
    tests = days[1:]; pooled = {}
    for name in ('P', 'M87', 'M87+'):
        y = np.concatenate([B[B.day == D].sev.to_numpy() for D in tests]); s = np.concatenate([res[(D, name)] for D in tests]); pooled[name] = auc(y, s)
    for D in tests:
        y = B[B.day == D].sev.to_numpy()
        L.append(f'  {D} (train {", ".join(x for x in days if x < D)}; {len(y):,} decisions, {int(y.sum())} severe): AUC ' +
                 ' | '.join(f'{nm} {auc(y, res[(D, nm)]):.3f}' for nm in ('P', 'M87', 'M87+')) + ' || log loss ' +
                 ' | '.join(f'{nm} {logloss(y, res[(D, nm)]):.4f}' for nm in ('P', 'M87', 'M87+')))
    L.append(f'  pooled AUC: P {pooled["P"]:.3f} | M87 {pooled["M87"]:.3f} | M87+ {pooled["M87+"]:.3f}')
    L.append('  M87+ top inputs (last fold): ' + ', '.join(f'{c} {v:.0%}' for c, v in imps[(tests[-1], 'M87+')]))
    d1 = sum(auc(B[B.day == D].sev, res[(D, 'M87')]) > auc(B[B.day == D].sev, res[(D, 'P')]) for D in tests)
    d2 = sum(auc(B[B.day == D].sev, res[(D, 'M87+')]) > auc(B[B.day == D].sev, res[(D, 'M87')]) for D in tests)
    i1 = pooled['M87'] >= pooled['P'] + 0.02 and d1 >= 2
    i2 = pooled['M87+'] >= pooled['M87'] + 0.02 and d2 >= 2
    L += ['', f'I1 EXISTING-INPUTS-UNDERUSED: {"YES" if i1 else "NOT-SHOWN"} (M87 {pooled["M87"]:.3f} vs P {pooled["P"]:.3f}; above on {d1} / 3 days)',
          f'I2 NEW-INFORMATION: {"YES" if i2 else "NOT-SHOWN"} (M87+ {pooled["M87+"]:.3f} vs M87 {pooled["M87"]:.3f}; above on {d2} / 3 days)', '',
          'LIMITS: consumed days; three test days, no interval; the earlier tape / other-venue test (<= +0.002 AUC) not repeated; simulated labels; neutral population']
    (OUT / 'severe_info.txt').write_text('\n'.join(L) + '\n')
    C.dump(OUT / 'severe_info.json', dict(asof=asof, predicates=dict(I1=i1, I2=i2), pooled_auc=pooled, seconds=time.time() - start))
    print('\n'.join(L), flush=True)


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'selftest':
        selftest()
    else:
        C.require(len(sys.argv) > 1 and sys.argv[1] == 'run', 'usage: pf_severe_info.py selftest | run')
        run()
