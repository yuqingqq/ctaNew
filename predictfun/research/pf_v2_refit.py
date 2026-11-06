"""Candidate pm_ladder_v2 = pm_ladder_v1 with the guard models, thresholds and per-minute lines refit every day on a trailing window
(USER 2026-11-06: "it may need to update given a window" ... "Yes we can test this"). Everything else is v1's: engine rules, arm,
features SETS['ALL'], LightGBM PARAMS / ROUNDS, label (v per side clipped +-10), threshold quantile 60, line percentiles.
For target day T (refit at T 00Z from complete days only):
  window   = the last W = 7 admissible days before T in DAYS (10-28..30 are sealed and never used; the tape starts 10-23). A shorter
             window (10-31: 5 days, 11-01: 6) is marked short: in-sample rehearsal only.
  models   = one LightGBM per side on the window's grid rows (step4/badfill_info_tx_tk/grid.parquet for the 8 read days, days/<d>.parquet
             for later days); c60[side] = 60% quantile of the model's in-sample predictions (v1's rule).
  lines    = v1's line rule inside the window: leave one window day out (fit on the other days, own c60), predict that day's 50 ms rows
             (step5/<d>/rows_rl_d2_pk, live rows = t_rel < 240 and a known Predict mid), v = prediction - c60 (float32), start-known
             withhold (-1); per minute 0-240 s, bid + ask pooled, finite v: np.percentile 50 / 60 / 70 / 80 / 90 / 95 rounded to 4 dp,
             plus the constant keys zero / c006 / inf.
  T values = the window models on T's 50 ms rows, v = prediction - c60, live rows, float32, start-known withhold.
Outputs: step5/<T>/preds_v2w7k.parquet (preds_t60k with v_bid / v_ask replaced), step5/<T>/lines_v2w7.json (the engine's minute-lines
format), step5/<T>/v2w7/guard_{bid,ask}.txt and refit.json (window, thresholds, lines, hashes).
Start-known rows come from day d's preds_t60 / preds_t60k pair (rows where pf_preds_startknown set both v to -1); the v1 pipeline writes
those first for every day.
Checks (each refuses): (3) always: the start-known mask is exact (no row of preds_t60 has both v = -1; preds_t60k equals preds_t60 off
the mask). PF_V2_CHECK=1 runs instead: (1) fold parity: window = the 8 read days, the fold without 10-23 reproduces
step5/20261023/preds_t60.parquet v exactly (falsifier: c + 1e-6 must fail); (2) line parity: the line rule on the 8 read days' preds_t60k
reproduces every key of step4/minute_lines_t60k.json (falsifier: the rule on 7 of the days must differ).
Usage: python3 -m predictfun.research.pf_v2_refit <T> [T ...]   |   PF_V2_CHECK=1 python3 -m predictfun.research.pf_v2_refit
"""
import hashlib, json, os, sys, time
import numpy as np, pandas as pd, lightgbm as lgb
from . import pf_io as io
from .pf_model_spec import SETS, PARAMS, ROUNDS

W = 7; K = 60; PCTS = [50, 60, 70, 80, 90, 95]; TAG = 'v2w7'
READ = ['20261023', '20261024', '20261025', '20261026', '20261027', '20261031', '20261101', '20261102']
SEALED = ('20261028', '20261029', '20261030')
FE = SETS['ALL']
GDIR = f'{io.OUT}/step4/badfill_info_tx_tk'; GRID = f'{GDIR}/grid.parquet'; LINES_V1 = f'{io.OUT}/step4/minute_lines_t60k.json'


def days_before(T):
    """admissible complete UTC days before T, from 10-23 on, sealed days skipped."""
    out, d = [], pd.Timestamp('2026-10-23')
    while d < pd.Timestamp(T):
        s = d.strftime('%Y%m%d')
        if s not in SEALED:
            out.append(s)
        d += pd.Timedelta(days=1)
    return out


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


G_CACHE = {}


def grid(days):
    """grid rows of these days, in DAYS order (the read days from grid.parquet in its own row order)."""
    if 'g' not in G_CACHE:
        G_CACHE['g'] = pd.read_parquet(GRID, columns=['day', 'v_bid', 'v_ask'] + FE)
    G = G_CACHE['g']; parts = []
    for d in days:
        if d in READ:
            parts.append(G[(G.day == d).values])
        else:
            p = f'{GDIR}/days/{d}.parquet'
            if not os.path.exists(p):
                sys.exit(f'REFUSED: no grid rows for {d} ({p})')
            parts.append(pd.read_parquet(p, columns=['day', 'v_bid', 'v_ask'] + FE))
    return parts


def fit(parts):
    """v1's fit (pf_step5_guard_preds.fit / pf_freeze_pm_ladder_v1): models per side and c60 from the in-sample predictions."""
    X = np.concatenate([p[FE].values.astype(np.float32) for p in parts]) if len(parts) > 1 else parts[0][FE].values.astype(np.float32)
    ms, cs = {}, {}
    for side in ('bid', 'ask'):
        y = np.concatenate([p[f'v_{side}'].values for p in parts])
        ds = lgb.Dataset(X, pd.Series(y).clip(-10, 10).values, params=PARAMS, free_raw_data=True).construct()
        m = lgb.train(PARAMS, ds, ROUNDS); del ds
        cs[side] = float(np.quantile(m.predict(X, num_threads=1), K / 100.0)); ms[side] = m
    n = len(X); del X
    return ms, cs, n


def sk_mask(d):
    """rows of day d withheld by the start-known rule, from the v1 pair preds_t60 / preds_t60k (check 3)."""
    a = pd.read_parquet(f'{io.OUT}/step5/{d}/preds_t60.parquet', columns=['mid', 'tau', 'v_bid', 'v_ask'])
    b = pd.read_parquet(f'{io.OUT}/step5/{d}/preds_t60k.parquet', columns=['mid', 'tau', 'v_bid', 'v_ask'])
    assert len(a) == len(b) and (a.mid.values == b.mid.values).all() and (a.tau.values == b.tau.values).all(), f'{d}: t60 / t60k rows'
    if ((a.v_bid.values == -1.0) & (a.v_ask.values == -1.0)).any():
        sys.exit(f'REFUSED: {d} preds_t60 has rows with both v = -1; the mask is not derivable')
    m = (b.v_bid.values == -1.0) & (b.v_ask.values == -1.0)
    for c in ('v_bid', 'v_ask'):
        x, y = a[c].values[~m], b[c].values[~m]
        if not (np.array_equal(np.isnan(x), np.isnan(y)) and np.array_equal(x[~np.isnan(x)], y[~np.isnan(y)])):
            sys.exit(f'REFUSED: {d} preds_t60k differs from preds_t60 off the start-known rows ({c})')
    return m


def day_rows(d):
    r = pd.read_parquet(f'{io.OUT}/step5/{d}/rows_rl_d2_pk.parquet', columns=list(dict.fromkeys(['mid', 'tau', 't_rel', 'pf_mid'] + FE)))
    live = ((r.t_rel < 240) & r.pf_mid.notna()).values
    return r, live


def values(ms, cs, r, live, mask):
    X = r[FE].values.astype(np.float32); out = {}
    for s in ('bid', 'ask'):
        v = np.full(len(r), np.nan, np.float64)
        v[live] = ms[s].predict(X[live], num_threads=1)
        out[s] = np.where(mask, np.float32(-1.0), (v - cs[s]).astype(np.float32)).astype(np.float32)
    return out


def rel_of(d, mid, tau):
    ws = pd.read_parquet(f'{io.OUT}/step1/{d}/registry.parquet', columns=['mid', 'ws']).set_index('mid').ws
    return (tau - pd.Series(mid).map(ws).values * 1000.0) / 1000.0


def lines_from(vals):
    """vals: list of (rel, v_bid, v_ask) per day -> the minute-lines dict (pf_minute_pnl's cut rule)."""
    rel = np.concatenate([x[0] for x in vals]); vb = np.concatenate([x[1] for x in vals]); va = np.concatenate([x[2] for x in vals])
    ok = (rel >= 0) & (rel < 240); mi = np.minimum((rel // 60).astype(int), 3)
    cut = {}
    for i in range(4):
        sel = ok & (mi == i); v = np.r_[vb[sel], va[sel]]; v = v[np.isfinite(v)]
        cut[i] = [round(float(x), 4) for x in np.percentile(v, PCTS)]
    L = {str(p): [cut[i][j] for i in range(4)] for j, p in enumerate(PCTS)}
    L.update(zero=[0.0] * 4, c006=[0.06] * 4, inf=[1e9] * 4)
    return L


def check():
    rep = {}
    # (2) line parity on v1's own out-of-fold values
    v1 = json.load(open(LINES_V1))['lines']; vals = {}
    for d in READ:
        p = pd.read_parquet(f'{io.OUT}/step5/{d}/preds_t60k.parquet', columns=['mid', 'tau', 'v_bid', 'v_ask'])
        vals[d] = (rel_of(d, p.mid.values, p.tau.values), p.v_bid.values, p.v_ask.values)
    L = lines_from([vals[d] for d in READ]); diff = {k: (L[k], v1[k]) for k in v1 if L.get(k) != v1[k]}
    if diff:
        sys.exit(f'REFUSED: the line rule does not reproduce minute_lines_t60k.json: {diff}')
    Lf = lines_from([vals[d] for d in READ[1:]])
    if all(Lf.get(k) == v1[k] for k in v1):
        sys.exit('FALSIFIER DID NOT FIRE: 7-day lines equal the 8-day lines')
    rep['line_parity'] = dict(equal=True, keys=sorted(v1), falsifier_fired=True); del vals
    print('line parity PASS', flush=True)
    # (1) fold parity: the fold without 10-23
    d0 = READ[0]; ms, cs, n = fit(grid(READ[1:])); r, live = day_rows(d0)
    P = pd.read_parquet(f'{io.OUT}/step5/{d0}/preds_t60.parquet', columns=['mid', 'tau', 'v_bid', 'v_ask'])
    assert len(P) == len(r) and (P.mid.values == r.mid.values).all() and np.allclose(P.tau.values, r.tau.values), 'row alignment'
    nomask = np.zeros(len(r), bool)

    def bad(c):
        v = values(ms, c, r, live, nomask); out = {}
        for s in ('bid', 'ask'):
            a, b = v[s], P[f'v_{s}'].values
            if not (np.array_equal(np.isnan(a), np.isnan(b)) and np.array_equal(a[~np.isnan(a)], b[~np.isnan(b)])):
                ok = ~np.isnan(a) & ~np.isnan(b); out[s] = float(np.max(np.abs(a[ok] - b[ok]))) if ok.any() else None
        return out
    b0 = bad(cs)
    if b0:
        sys.exit(f'REFUSED: fold fit does not reproduce {d0} preds_t60: max |diff| {b0}')
    fz = bad({'bid': cs['bid'] + 1e-6, 'ask': cs['ask']})
    if not fz:
        sys.exit('FALSIFIER DID NOT FIRE: fold parity passed a moved threshold')
    rep['fold_parity'] = dict(day=d0, train_rows=n, thresholds=cs, rows=int(len(r)), live=int(live.sum()), equal=True, falsifier=fz)
    print('fold parity PASS', rep['fold_parity'], flush=True)
    io.write_json(f'{io.OUT}/step5/v2_refit_check.json', rep)


def refit(T):
    t0 = time.time(); prior = days_before(T); win = prior[-W:]; short = len(win) < W
    if T in SEALED or not win:
        sys.exit(f'REFUSED: {T}')
    parts = grid(win)
    ms, cs, n = fit(parts)
    vals, folds = [], {}
    for i, d in enumerate(win):
        fm, fc, fn = fit([p for j, p in enumerate(parts) if j != i])
        r, live = day_rows(d); mask = sk_mask(d); v = values(fm, fc, r, live, mask)
        vals.append((rel_of(d, r.mid.values, r.tau.values), v['bid'], v['ask']))
        folds[d] = dict(train_rows=fn, thresholds=fc, live=int(live.sum()), startknown_rows=int(mask.sum()))
        print(T, 'fold', d, folds[d], f'{time.time() - t0:.0f}s', flush=True); del fm, r, live, mask, v
        io.mem_check(f'{T} fold {d}')
    L = lines_from(vals); del vals
    r, live = day_rows(T); mask = sk_mask(T); v = values(ms, cs, r, live, mask)
    pk = pd.read_parquet(f'{io.OUT}/step5/{T}/preds_t60k.parquet')
    assert len(pk) == len(r) and (pk.mid.values == r.mid.values).all() and (pk.tau.values == r.tau.values).all(), f'{T}: row alignment'
    pk['v_bid'] = v['bid']; pk['v_ask'] = v['ask']
    od = f'{io.OUT}/step5/{T}/{TAG}'; os.makedirs(od, exist_ok=True)
    pk.to_parquet(f'{io.OUT}/step5/{T}/preds_{TAG}k.parquet', index=False)
    for s in ms:
        ms[s].save_model(f'{od}/guard_{s}.txt')
    lf = f'{io.OUT}/step5/{T}/lines_{TAG}.json'
    io.write_json(lf, dict(source=f'pf_v2_refit {TAG}: leave-one-day-out v inside the window {win}, bid + ask pooled, per minute (v1 rule)',
                           written_utc=pd.Timestamp.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ'), minutes=['0-60', '60-120', '120-180', '180-240'], lines=L))
    rel = rel_of(T, r.mid.values, r.tau.values); m0 = (rel >= 0) & (rel < 60)
    share = {s: dict(above_90=float(np.mean(v[s][m0][np.isfinite(v[s][m0])] > L['90'][0])),
                     above_95=float(np.mean(v[s][m0][np.isfinite(v[s][m0])] >= L['95'][0]))) for s in ('bid', 'ask')}
    rep = dict(T=T, window=win, short_window=short, W=W, train_rows=n, thresholds=cs, lines=L, folds=folds, minute0_share=share,
               model_sha256={s: sha(f'{od}/guard_{s}.txt') for s in ms}, lines_sha256=sha(lf), preds_rows=int(len(pk)), live=int(live.sum()),
               startknown_rows=int(mask.sum()), grid_sha256=sha(GRID), wall_s=round(time.time() - t0, 1))
    io.write_json(f'{od}/refit.json', rep)
    print(T, 'DONE', {k: rep[k] for k in ('window', 'short_window', 'thresholds', 'minute0_share', 'wall_s')}, 'lines90', L['90'], 'lines95', L['95'], flush=True)


def main():
    if os.environ.get('PF_V2_CHECK') == '1':
        check(); return 0
    for T in sys.argv[1:]:
        refit(T)
    return 0


if __name__ == '__main__':
    sys.exit(main())
