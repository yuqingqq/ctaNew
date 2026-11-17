"""RUNNING ERRORS OF THE GAIN / LOSS MODEL (2026-11-17, session aea5bfb9; USER after auditing the diagnostic: "Next: test whether recent,
fully observed probability and severity errors predict subsequent errors, using past-only folds. That directly tests the proposed
running-calibration idea while keeping coverage and model error open as explanations"). Declared before any run; descriptive with declared
predicates; no controller change.

CORRECTIONS CARRIED (USER audit of pf_gain_loss_diag): its leave-one-day-out folds are cross-day diagnostics - only the Oct25 fold used
earlier days only; coverage is not ruled out (familiarity was ranked within the test set); GAIN-LOSS's PASS excludes its failed loss
calibration and matched participation only by day x kind count (re-checked here within market x side, part 4).
PAST-ONLY FOLDS (expanding): each day D in Oct24..Oct27 is predicted by the GAIN-LOSS recipe (pf_gain_loss: the six 'full' hurdle heads,
pinned LightGBM, 200 rounds, the 87 inputs) fitted on every eligible decision of the days before D (Oct23..D-1; Oct26 = set L, the same
selection rule). The Oct26 fold must reproduce the existing GAIN-LOSS predictions exactly.
PER DECISION: severe event s = 1{loss > 30 c}; p = P(severe) (head l2_p); severity = excess beyond 30 c; s_hat = E[excess | severe]
(head l2_amt); loss L = max(-W, 0) and L_hat = l1 + l2 predictions.
RUNNING ERRORS (causal): for a decision at t, every same-day decision (all markets, both sides and kinds) decided in [t - lag - window,
t - lag] - its 1 s outcome long closed. PRIMARY lag 300 s (trade records reach us late: REST p50 6.9 s, p95 380 s), window 60 min; printed
also lag 2 s, window 30 min and day-so-far. Count ratio R_cnt = (observed severe + 1) / (sum p + 1); severity ratio R_sev = (sum excess
+ 3 x 30) / (sum s_hat over the observed severe events + 3 x 30); loss ratio R_loss = (sum L + 30) / (sum L_hat + 30). A day starts at 1.
TESTS: (1) the counts behind the USER's clue on set L (3-day heads): predicted / observed severe count and loss per event by day x kind;
(2) persistence: 60-min UTC blocks - lag-1 correlation of block log O/E (count; and severity where a block has >= 3 events), pooled over
the four predicted days and by day; decision level - forward O/E in terciles of the trailing ratio; (3) running recalibration: p' = min(1,
p x R_cnt), s' = s_hat x R_sev, L' = l1 + p' x s' - severe log loss, count O/E and loss RMSE against the unadjusted model, by day;
(4) GAIN-LOSS's matched participation re-run with selection inside every day x kind x market x side stratum (GL vs DIRECT, top 10 / 25 /
50% of each stratum; 200-draw random selection inside the same strata).
PREDICATES (declared; primary lag / window): Q1 PROBABILITY-ERRORS-PERSIST: block lag-1 correlation > 0.2 AND top-tercile forward O/E
exceeds the bottom tercile's by > 0.15 pooled AND that direction on >= 3 of 4 days. Q2 SEVERITY-ERRORS-PERSIST: the same for severity.
Q3 RUNNING-CALIBRATION-HELPS: p' lowers severe log loss on >= 3 of 4 days AND moves the pooled count O/E toward 1. G2S (part 4):
GL >= DIRECT in >= 3 of 4 day x kind cells at >= 2 of 3 rates -> GL-SELECTS-BETTER-STRATIFIED, else NOT-SHOWN.
CONTROLS (any failure -> no output): every fold trains only on earlier days; the Oct26 fold reproduces the GAIN-LOSS predictions
(|diff| <= 1e-9); the training population of each fold is every eligible decision of its days; running windows hold only decisions at
or before t - lag; synthetic positive control (persistent errors -> positive correlation) and negative control (independent errors -> ~0)
in the selftest, and a planted future leak in the window is refused.
LIMITS: consumed days; four predicted days (no interval, CLAUDE.md rule 8); simulated labels; the neutral decision population.
Usage (run_light, PF_SLOT=claude): pf_gain_loss_running.py selftest | run
"""
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pf_gain_loss as GL                                 # noqa: E402

J, JF, F, C, pd = GL.J, GL.JF, GL.F, GL.C, GL.pd
OUT = GL.OUT / 'running'
DAYS = ('20261023', '20261024', '20261025', '20261026', '20261027')
PRED_DAYS = DAYS[1:]
LAG, WINDOW = 300000., 3600000.
SENS = ((2000., 3600000.), (300000., 1800000.), (300000., 86400000.))
SEV_PRIOR, SEV_SCALE, LOSS_PRIOR = 3., 30., 30.
MUT_FUTURE = 0                                            # planted error (selftest only)


def trailing(t, x, lag, window):
    """Sum of x over decisions with t_i in [t_j - lag - window, t_j - lag], t sorted ascending."""
    cs = np.r_[0., np.cumsum(x)]
    hi = np.searchsorted(t, t - lag + (60000. if MUT_FUTURE else 0.), side='right'); lo = np.searchsorted(t, t - lag - window, side='left')
    return cs[hi] - cs[np.minimum(lo, hi)]


def window_ok(t, lag, window):
    """Every decision's window ends at or before its own t - lag."""
    hi = np.searchsorted(t, t - lag + (60000. if MUT_FUTURE else 0.), side='right')
    last = np.where(hi > 0, t[np.maximum(hi - 1, 0)], -np.inf)
    return bool(np.all(last <= t - lag + 1e-9))


def selftest():
    bad = []; rng = np.random.default_rng(7)
    t = np.arange(0., 4000. * 1000., 1000.)
    if not window_ok(t, 2000., 60000.):
        bad.append('window refused on valid input')
    s = trailing(t, np.ones(len(t)), 2000., 10000.)
    if s[20] != 11. or s[0] != 0.:
        bad.append(f'trailing sums {s[20]} {s[0]}')
    level = np.cumsum(rng.normal(0, 1, 40))                                    # persistent block-level error (random walk)
    pos = np.corrcoef(level[:-1], level[1:])[0, 1]
    neg = np.corrcoef(rng.normal(0, 1, 40)[:-1], rng.normal(0, 1, 40)[1:])[0, 1]
    if not (pos > 0.5 and abs(neg) < 0.5):
        bad.append(f'persistence controls {pos:.2f} {neg:.2f}')
    globals()['MUT_FUTURE'] = 1
    try:
        caught = not window_ok(t, 2000., 60000.)
    finally:
        globals()['MUT_FUTURE'] = 0
    if bad or not caught:
        sys.exit(f'RUNNING-ERRORS SELFTEST FAILED -> no output: {bad}, future leak caught {caught}')
    print('running-errors selftests passed; planted future leak caught', flush=True)


def load():
    fr = {}
    for d in DAYS[:3]:
        fr[d] = JF.read(d)[0]
    for d in DAYS[3:]:
        lr = json.loads((GL.OUT / f'label_{d}' / 'receipt.json').read_text())
        C.require(C.sha(GL.OUT / f'label_{d}' / 'actions.parquet') == lr['outputs']['actions.parquet'], 'set L changed')
        z = pd.read_parquet(GL.OUT / f'label_{d}' / 'actions.parquet'); fr[d] = z[z.status == 'eligible'].copy()
    for d, z in fr.items():
        z['day'] = d; z['w'] = z.y_w.astype(float); g, l1, l2 = GL.components(z.w.values)
        z['loss'] = l1 + l2; z['exc'] = l2; z['sev'] = (l2 > 1e-9).astype(float)
    return fr


def fit_fold(train, out):
    feats = GL.feature_list('full'); JF.feature_contract(feats); X = train[feats].to_numpy(np.float32)
    g, l1, l2 = GL.components(train.w.values); comp = dict(g=g, l1=l1, l2=l2); arts = {}
    for c in GL.COMP:
        y = comp[c]; pos = y > 1e-9
        for head, Xh, yh, obj in ((f'full_{c}_p', X, pos.astype(float), 'binary'), (f'full_{c}_amt', X[pos], y[pos] / GL.SCALE, 'regression')):
            path = out / f'{head}.txt'; J.FM.fit(Xh, yh, feats, path, obj); arts[head] = dict(path=str(path))
    return arts


def run():
    selftest(); start = time.time(); OUT.mkdir(parents=True, exist_ok=False); fr = load(); P = {}
    E0 = pd.read_parquet(GL.OUT / 'eval' / 'set_L.parquet')
    for D in PRED_DAYS:
        before = [d for d in DAYS if d < D]; C.require(max(before) < D, 'fold is not past-only')
        train = pd.concat([fr[d] for d in before], ignore_index=True); d_ = OUT / f'fold_{D}'; d_.mkdir()
        arts = fit_fold(train, d_); z = fr[D].copy(); pr = GL.predict(z, 'full', arts)
        z['p'] = pr['l2_p']; z['l2h'] = pr['l2']; z['shat'] = np.where(pr['l2_p'] > 0, pr['l2'] / np.maximum(pr['l2_p'], 1e-300), 0.)
        z['l1h'] = pr['l1']; z['lh'] = pr['l1'] + pr['l2']; z['train_days'] = ','.join(before)
        if D == '20261026':
            ref = E0[E0.day.astype(str) == D].set_index('uid').loc[z.uid]
            C.require(np.allclose(z.lh.values, (ref.GL_l1 + ref.GL_l2).values, rtol=0, atol=1e-9), 'the Oct26 fold does not reproduce GAIN-LOSS')
        P[D] = z.sort_values('decision_ms').reset_index(drop=True)
        print(json.dumps(dict(stage='fold', day=D, train_days=before, n_train=len(train), seconds=round(time.time() - start))), flush=True)
    L = [f'RUNNING ERRORS OF THE GAIN / LOSS MODEL (past-only expanding folds; no controller change); as of {time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}',
         'controls: folds train on earlier days only; the Oct26 fold reproduces GAIN-LOSS exactly; windows end at or before t - lag; selftest controls pass', '']
    # (1) the clue, on set L with the 3-day heads
    L.append('PART 1: set L, 3-day heads - predicted / observed severe events (count ratio) | predicted / observed excess per event | n')
    E = E0.copy(); E['day'] = E.day.astype(str); g_, l1_, l2_ = GL.components(E.w.values); E['exc'] = l2_
    for (d, k), g in E.groupby(['day', 'kind']):
        n = int((g.exc > 1e-9).sum()); pc = float(g.GL_l2_p.sum()); per_pred = float(g.GL_l2.sum()) / max(pc, 1e-9); per_obs = float(g.exc.sum()) / max(n, 1)
        L.append(f'  {d} {k:<6} {pc:7.1f} / {n:4d} (x{pc / max(n, 1):.2f}) | {per_pred:6.2f} / {per_obs:6.2f} c (x{per_pred / max(per_obs, 1e-9):.2f}) | {len(g):5,}')
    # running ratios
    def ratios(z, lag, window):
        t = z.decision_ms.to_numpy(float); C.require(window_ok(t, lag, window), 'running window reaches past t - lag')
        n = trailing(t, z.sev.to_numpy(), lag, window); sp = trailing(t, z.p.to_numpy(), lag, window)
        ex = trailing(t, z.exc.to_numpy(), lag, window); sh = trailing(t, (z.shat * z.sev).to_numpy(), lag, window)
        lo = trailing(t, z.loss.to_numpy(), lag, window); lh = trailing(t, z.lh.to_numpy(), lag, window)
        return (n + 1.) / (sp + 1.), (ex + SEV_PRIOR * SEV_SCALE) / (sh + SEV_PRIOR * SEV_SCALE), (lo + LOSS_PRIOR) / (lh + LOSS_PRIOR)
    # (2) persistence - blocks
    L += ['', 'PART 2: persistence (past-only predictions). Blocks = 60-min UTC hours; lag-1 correlation of block log O/E (count | severity, blocks with >= 3 events)']
    bl = []
    for D, z in P.items():
        day0 = float(pd.Timestamp(f'{D[:4]}-{D[4:6]}-{D[6:]}', tz='UTC').value // 10 ** 6); b = ((z.decision_ms - day0) // 3600000).astype(int)
        for h, g in z.groupby(b):
            n = int(g.sev.sum()); bl.append(dict(day=D, h=int(h), oe=(n + .5) / (g.p.sum() + .5), n=n,
                                                 sev=(g.exc.sum() / max((g.shat * g.sev).sum(), 1e-9)) if n >= 3 else np.nan))
    B = pd.DataFrame(bl).sort_values(['day', 'h']); pairs = []
    for D, g in B.groupby('day'):
        g = g.set_index('h')
        for h in g.index:
            if h + 1 in g.index:
                pairs.append(dict(day=D, a=np.log(g.loc[h, 'oe']), b=np.log(g.loc[h + 1, 'oe']), sa=np.log(g.loc[h, 'sev']) if np.isfinite(g.loc[h, 'sev']) else np.nan,
                                  sb=np.log(g.loc[h + 1, 'sev']) if np.isfinite(g.loc[h + 1, 'sev']) else np.nan))
    Pp = pd.DataFrame(pairs); cc = float(np.corrcoef(Pp.a, Pp.b)[0, 1]); sv = Pp.dropna(subset=['sa', 'sb'])
    cs_ = float(np.corrcoef(sv.sa, sv.sb)[0, 1]) if len(sv) > 3 else float('nan')
    L.append(f'  pooled: count r = {cc:+.3f} ({len(Pp)} pairs) | severity r = {cs_:+.3f} ({len(sv)} pairs)')
    L.append('  by day: ' + ' | '.join(f'{D} count {np.corrcoef(g.a, g.b)[0, 1]:+.2f}' + (f', severity {np.corrcoef(g.dropna().sa, g.dropna().sb)[0, 1]:+.2f}' if g.dropna().shape[0] > 3 else '')
                                        for D, g in Pp.groupby('day')))
    # (2b) decision-level terciles of the trailing ratio, every lag / window
    L.append('  decision level: forward O/E (observed / predicted severe count) in terciles of the trailing count ratio | forward severity ratio in terciles of the trailing severity ratio')
    res = {}
    for lag, window in ((LAG, WINDOW),) + SENS:
        rows = []
        for D, z in P.items():
            rc, rs, rl = ratios(z, lag, window); rows.append(z.assign(rc=rc, rs=rs, rl=rl))
        Z = pd.concat(rows, ignore_index=True)
        cut = np.quantile(Z.rc, [1 / 3, 2 / 3]); Z['tc'] = np.digitize(Z.rc, cut); cut2 = np.quantile(Z.rs, [1 / 3, 2 / 3]); Z['ts'] = np.digitize(Z.rs, cut2)
        fo = {q: float(Z[Z.tc == q].sev.sum() / max(Z[Z.tc == q].p.sum(), 1e-9)) for q in range(3)}
        fs = {q: float(Z[(Z.ts == q) & (Z.sev > 0)].exc.sum() / max(Z[(Z.ts == q) & (Z.sev > 0)].shat.sum(), 1e-9)) for q in range(3)}
        days_dir = sum(int(g[g.tc == 2].sev.sum() / max(g[g.tc == 2].p.sum(), 1e-9) > g[g.tc == 0].sev.sum() / max(g[g.tc == 0].p.sum(), 1e-9)) for D, g in Z.groupby('day'))
        days_dir_s = sum(int(g[(g.ts == 2) & (g.sev > 0)].exc.sum() / max(g[(g.ts == 2) & (g.sev > 0)].shat.sum(), 1e-9) >
                             g[(g.ts == 0) & (g.sev > 0)].exc.sum() / max(g[(g.ts == 0) & (g.sev > 0)].shat.sum(), 1e-9)) for D, g in Z.groupby('day'))
        res[(lag, window)] = dict(Z=Z, fo=fo, fs=fs, dd=days_dir, dds=days_dir_s)
        L.append(f'    lag {lag / 1000:5.0f} s, window {window / 60000:6.0f} min: count O/E by tercile x{fo[0]:.2f} / x{fo[1]:.2f} / x{fo[2]:.2f} (top > bottom on {days_dir} / 4 days) | '
                 f'severity x{fs[0]:.2f} / x{fs[1]:.2f} / x{fs[2]:.2f} (top > bottom on {days_dir_s} / 4 days)')
    pr_ = res[(LAG, WINDOW)]
    q1 = cc > 0.2 and (pr_['fo'][2] - pr_['fo'][0]) > 0.15 and pr_['dd'] >= 3
    q2 = (cs_ > 0.2) and (pr_['fs'][2] - pr_['fs'][0]) > 0.15 and pr_['dds'] >= 3
    # (3) running recalibration, primary lag / window
    Z = pr_['Z']; Z['p2'] = np.minimum(1., Z.p * Z.rc); Z['s2'] = Z.shat * Z.rs; Z['l2'] = Z.l1h + Z.p2 * Z.s2
    L += ['', "PART 3: running recalibration (lag 300 s, window 60 min) by day: severe log loss p -> p' | count O/E p -> p' | loss total ratio -> | loss RMSE -> (c)"]
    wins = 0; oe0 = oe1 = 0.
    for D, g in Z.groupby('day'):
        def ll(p):
            p = np.clip(p, 1e-9, 1 - 1e-9); return float(-np.mean(g.sev * np.log(p) + (1 - g.sev) * np.log(1 - p)))
        a, b = ll(g.p), ll(g.p2); wins += int(b < a)
        L.append(f'  {D} (train {g.train_days.iloc[0]}): {a:.4f} -> {b:.4f} | x{g.p.sum() / g.sev.sum():.2f} -> x{g.p2.sum() / g.sev.sum():.2f} | x{g.lh.sum() / g.loss.sum():.2f} -> '
                 f'x{g.l2.sum() / g.loss.sum():.2f} | {np.sqrt(np.mean((g.lh - g.loss) ** 2)):.2f} -> {np.sqrt(np.mean((g.l2 - g.loss) ** 2)):.2f}')
    oe0 = abs(np.log(Z.p.sum() / Z.sev.sum())); oe1 = abs(np.log(Z.p2.sum() / Z.sev.sum()))
    q3 = wins >= 3 and oe1 < oe0
    # (4) stratified matched participation
    L += ['', 'PART 4: GAIN-LOSS matched participation with selection inside every day x kind x market x side stratum - realized mean W of the selected: GL / DIRECT | random in strata (mean)']
    rng = np.random.default_rng(20261117); w2 = {}
    for rate in GL.RATES:
        win_ = 0
        for (d, k), g in E.groupby(['day', 'kind']):
            sel = {}
            for mdl in ('GL', 'DIRECT'):
                parts = []
                for _, s in g.groupby(['m', 'side']):
                    n = int(math.ceil(rate * len(s))); parts.append(s.sort_values([mdl, 'hash'], ascending=[False, True]).w.iloc[:n])
                sel[mdl] = float(pd.concat(parts).mean())
            null = []
            for _ in range(200):
                parts = [s.w.values[rng.permutation(len(s))[:int(math.ceil(rate * len(s)))]] for _, s in g.groupby(['m', 'side'])]
                null.append(float(np.concatenate(parts).mean()))
            win_ += int(sel['GL'] >= sel['DIRECT'])
            L.append(f'  top {rate:.0%} {d} {k:<6}: {sel["GL"]:+6.2f} / {sel["DIRECT"]:+6.2f} | random {np.mean(null):+6.2f}')
        w2[rate] = win_
    g2s = sum(v >= 3 for v in w2.values()) >= 2
    L += ['', f'Q1 PROBABILITY-ERRORS-PERSIST: {"PERSIST" if q1 else "NOT-SHOWN"} (block r {cc:+.3f}; top - bottom tercile O/E {pr_["fo"][2] - pr_["fo"][0]:+.2f}; direction on {pr_["dd"]} / 4 days)',
          f'Q2 SEVERITY-ERRORS-PERSIST: {"PERSIST" if q2 else "NOT-SHOWN"} (block r {cs_:+.3f}; top - bottom {pr_["fs"][2] - pr_["fs"][0]:+.2f}; direction on {pr_["dds"]} / 4 days)',
          f'Q3 RUNNING-CALIBRATION-HELPS: {"HELPS" if q3 else "NOT-SHOWN"} (log loss lower on {wins} / 4 days; pooled |log O/E| {oe0:.3f} -> {oe1:.3f})',
          f'G2S STRATIFIED MATCHED PARTICIPATION: {"GL-SELECTS-BETTER-STRATIFIED" if g2s else "NOT-SHOWN"} (' + ', '.join(f'{w2[r]} / 4 at {r:.0%}' for r in GL.RATES) + ')',
          '', 'LIMITS: consumed days; four predicted days, no interval (rule 8); simulated labels; the neutral decision population; coverage and model error remain open']
    (OUT / 'running_errors.txt').write_text('\n'.join(L) + '\n')
    C.dump(OUT / 'running_errors.json', dict(asof=C.stamp(), predicates=dict(Q1=q1, Q2=q2, Q3=q3, G2S=g2s), block_r=dict(count=cc, severity=cs_),
                                             g2s=w2, seconds=time.time() - start))
    print('\n'.join(L), flush=True)


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'selftest':
        selftest()
    else:
        C.require(len(sys.argv) > 1 and sys.argv[1] == 'run', 'usage: pf_gain_loss_running.py selftest | run')
        run()
