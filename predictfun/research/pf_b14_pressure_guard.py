"""B14 (2026-11-11, USER "run the prediction guard"): the live guard rebuilt with B13a's 13 engineered pressure features, then
run as the guard of the frozen v3 strategy. Does it pick better moments to quote, and does v3 do better with it?

The live guard (pm_ladder_v1 / v3, pf_step5_guard_preds.py with PF_GRID=step4/badfill_info_tx_tk/grid.parquet PF_ROWS=rows_rl_d2_pk
PF_PSTEM=t PF_GUARD_KS=60): per side a LightGBM model (pinned PARAMS / ROUNDS) of the value of keeping an order at the known best
through the next exposure (grid label v_bid / v_ask, clipped +-10), on the features ALL (32), trained on the guard grid (8 read days,
~134k rows a day); threshold c = the 60% quantile of its in-sample predictions; guard value v = prediction - c on live rows
(t_rel < 240 s and a known Predict mid) of step5/<day>/rows_rl_d2_pk; read days leave one day out (LODO), 11-03 from the fit on all
8 days. preds_t60 -> pf_preds_startknown -> preds_t60k (v = -1 before the start price is known). The pull and rung line is the
per-minute 90th percentile and the size line the 95th of preds_t60k's v (bid and ask pooled, rel in [0, 240) s, 8 read days;
step4/minute_lines_t60k.json, in-sample on read days).

EMPH guard: the same recipe, code path and settings with the features ALL + B13a's 13 engineered pressure features (pf_b13a_pressure_
model.engineer, trailing only, per window), computed on rows_rl_d2_pk, the live clock the grid's features come from (they match it
exactly; B13a itself ran on step5 rows.parquet, another clock). Lines by the same rule: the per-minute 90th / 95th percentiles of
EMPH's start-known v on the 8 read days. The frozen engine reads the frozen line file, so EMPH's v is mapped per minute by the
increasing line through (its p90 -> line 90, its p95 -> line 95): the engine then pulls, quotes the rung and sizes up exactly where
EMPH's own lines would (rows at -1, the start withhold, and NaN rows unchanged).

Days: 10-23..10-27, 10-31, 11-01, 11-02 (read, LODO) and 11-03 (fit on all 8; out of sample for the guard and the lines). All 9 were
read by earlier research; nothing later is read (no protected day: 11-04..11-06, 11-09..11-13; not 11-07 / 11-08).

Strategy replay: the frozen v3 engine pf_step4_ladder_v3.py, arm pf_ladder_v3_build.ARM, run_forward_v3.sh's env (PF_RL_DDEC=2
PF_RL_LPLACE=40 PF_RL_LCANCEL=22 PF_RL_LPENDCANCEL=95 PF_RL_LFILL=100 PF_TIMING=e2 PF_TXTIME=ws PF_TXKEY=tk), timing J (PF_TXJIT=frame)
and X (PF_TXSHIFT=-50 PF_QACT=max), PF_PREDS = preds_t60k (REF, tags b14rJ / b14rX) or preds_e60km (EMPH, b14eJ / b14eX). Values per
day from summary.json: settle $ = gross_usd (held to settlement, no rebate), +1 s $ = mo1_usd.

Readings DECLARED BEFORE THE FIRST RUN (evaluated in code; t intervals 95%, day-clustered):
 R120 skill    on the guard grid's rows of each read day (out of fold for both guards): the mean label (c/sh, bid + ask pooled) on
               the rows each guard lets quote (its out-of-fold v above its own per-minute 90th percentile on the grid rows, pooled over
               the 8 days), EMPH minus BASE (BASE = preds_t60's v, the live guard). IMPROVES if the interval (t(7)) is above 0 and >= 6 of
               8 days are positive; WORSE if it is below 0; else NO-CHANGE. VOID if the falsifier PERM (EMPH refit with the 13 features'
               rows permuted within each day) also reads IMPROVES.
 R121 settle   v3 replay, EMPH minus REF per day, settle $: BETTER if the 9-day sum is > 0 at J and at X and >= 6 of 9 days are positive at
               each; WORSE if the sum is < 0 at both and <= 3 of 9 days are positive at each; else MIXED.
 R122 +1 s     the same rule on +1 s $.
Beside (descriptive): per-day values; 11-03 alone; fills, shares, windows positive; the share of grid rows each guard lets quote; the
bad share (label <= -2 c) on them; per-side rank correlation of v with the label.
Controls (any failure -> no table):
 C1 parity     the BASE recipe through this file's own fit / apply code reproduces preds_t60's v exactly (float32, NaN pattern) on 11-03 (fit
               on all 8 days) and on 10-23 (LODO); falsifier: the threshold moved by 1e-6 must differ.
 C2 lines      the rule (per-minute p90 / p95 of preds_t60k, read days) reproduces step4/minute_lines_t60k.json's lines 90 and 95 (4 dp).
 C3 mapping    on every day, per minute, v' > line 90 exactly where v > EMPH's p90 and v' >= line 95 exactly where v >= its p95; -1 and
               NaN rows unchanged; falsifier: a map built on p90 + 0.001 must break it.
 C4 features   B13a's planted-frame check of engineer() (no look-ahead, no crossing of windows); every grid row finds its features.
 C5 alignment  EMPH files have preds_t60's rows (mid, tau) and pred_1000 / pred_250 columns unchanged.
 C6 replay     the REF runs reproduce the stored v3 replays i2J / i2X (base engine, same env) per day: gross_usd, mo1_usd, fills, shares.
 C7 rules      the reading predicates on synthetic inputs (each label reachable).
Limits fixed in advance: the 13 features were designed after looking at 11-08 and tested by B13a on these 9 days (another target and
clock), so this is in-sample for the feature choice: exploratory, not a validation (a validation needs >= 5 untouched days). Lines are
in-sample on read days for both guards (the same rule). The replay's fill model is not validated (B1a: 1.5-2x live's shares), so only
the difference under one model is read. At a minute boundary the last value of the old minute (<= 100 ms) meets the new minute's line
under the old minute's map. The soft pull and incremental size are not in the research engine: this tests the guard only.

Usage (each step through run_light.sh; predictfun/research/run_b14.sh runs them all in order):
  python3 -m predictfun.research.pf_b14_pressure_guard selftest | features | build | map | report
"""
import json
import os
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd

from . import pf_io as io
from .pf_model_spec import SETS, PARAMS, ROUNDS
from .pf_b13a_pressure_model import engineer, EXTRA, controls as b13a_controls

READ = ['20261023', '20261024', '20261025', '20261026', '20261027', '20261031', '20261101', '20261102']
FRESH = '20261103'
DAYS = READ + [FRESH]
S4 = f'{io.OUT}/step4'
S5 = f'{io.OUT}/step5'
GRID = f'{S4}/badfill_info_tx_tk/grid.parquet'
ROWS = 'rows_rl_d2_pk'
OUTD = f'{io.OUT}/b14_20261111'
LINES_FILE = f'{S4}/minute_lines_t60k.json'
FE = SETS['ALL']
EMPH = FE + EXTRA
K = 60
NEED = ['pf_imb1', 'bn_imb', 'pf_lbs', 'pf_las', 'bn_ntr1000', 'pf_dm250', 'bn_flow250']
T7, T8 = 2.365, 2.306
SEED_PERM = 20261111
NAN = float('nan')


def minute_of(rel):
    return np.minimum(np.maximum(rel // 60, 0), 3).astype(int)


def rel_of(d, mid, tau):
    ws = pd.read_parquet(f'{io.OUT}/step1/{d}/registry.parquet', columns=['mid', 'ws']).set_index('mid').ws
    return (np.asarray(tau, float) - pd.Series(mid).map(ws).values * 1000.0) / 1000.0


# ---------------------------------------------------------------- features

def ex_path(d):
    return f'{OUTD}/ex_{d}.parquet'


def features_day(d):
    """EXTRA for every row of step5/<d>/rows_rl_d2_pk, in the file's row order (computed on a (mid, tau) sorted copy)."""
    r = pd.read_parquet(f'{S5}/{d}/{ROWS}.parquet', columns=['mid', 'tau'] + NEED)
    order = np.lexsort((r.tau.values, r.mid.values))
    rs = r.iloc[order].reset_index(drop=True)
    e = engineer(rs)
    out = pd.DataFrame(index=np.arange(len(r)))
    inv = np.empty_like(order); inv[order] = np.arange(len(order))
    for c in EXTRA:
        out[c] = e[c].values[inv]
    out.insert(0, 'tau', r.tau.values); out.insert(0, 'mid', r.mid.values)
    return out


# ---------------------------------------------------------------- fit / apply (pf_step5_guard_preds.fit / write, generalised to a column list)

def fit(X, Y):
    ms, cs = {}, {}
    for side in ('bid', 'ask'):
        ds = lgb.Dataset(X, Y[side], params=PARAMS, free_raw_data=True).construct()
        m = lgb.train(PARAMS, ds, ROUNDS); del ds
        p = m.predict(X, num_threads=1); ms[side] = m; cs[side] = float(np.quantile(p, K / 100.0)); del p
    return ms, cs


def day_matrix(d, cols, ex=None):
    r = pd.read_parquet(f'{S5}/{d}/{ROWS}.parquet', columns=list(dict.fromkeys(['mid', 'tau', 't_rel', 'pf_mid'] + FE)))
    live = ((r.t_rel < 240) & r.pf_mid.notna()).values
    X = r[FE].values.astype(np.float32)
    r = r[['mid', 'tau']]
    if cols is not FE:
        if ex is None:
            ex = pd.read_parquet(ex_path(d))
        assert len(ex) == len(r) and (ex.mid.values == r.mid.values).all() and (ex.tau.values == r.tau.values).all(), f'{d}: ex alignment'
        X = np.concatenate([X, ex[EXTRA].values.astype(np.float32)], axis=1)
    return r, live, X


def apply(ms, cs, live, X):
    o = {}
    for s in ('bid', 'ask'):
        vh = np.where(live, ms[s].predict(X, num_threads=1), np.nan)
        o[s] = (vh - cs[s]).astype(np.float32)
    return o


def same_f32(a, b):
    a = np.asarray(a, np.float32); b = np.asarray(b, np.float32)
    return a.shape == b.shape and np.array_equal(np.isnan(a), np.isnan(b)) and np.array_equal(a[~np.isnan(a)], b[~np.isnan(b)])


# ---------------------------------------------------------------- readings

def interval(x, t):
    x = np.asarray(x, float)
    if len(x) < 2 or not np.all(np.isfinite(x)):
        return NAN, NAN, NAN
    m = float(x.mean()); h = t * float(x.std(ddof=1)) / np.sqrt(len(x))
    return m, m - h, m + h


def label_r120(d, t=T7, need=6):
    d = np.asarray(d, float)
    if len(d) != 8 or not np.all(np.isfinite(d)):
        return 'n/a'
    _, lo, hi = interval(d, t)
    if lo > 0 and int((d > 0).sum()) >= need:
        return 'IMPROVES'
    if hi < 0:
        return 'WORSE'
    return 'NO-CHANGE'


def label_r12x(dj, dx, up='BETTER', down='WORSE'):
    dj = np.asarray(dj, float); dx = np.asarray(dx, float)
    if len(dj) != 9 or len(dx) != 9 or not (np.all(np.isfinite(dj)) and np.all(np.isfinite(dx))):
        return 'n/a'
    if dj.sum() > 0 and dx.sum() > 0 and (dj > 0).sum() >= 6 and (dx > 0).sum() >= 6:
        return up
    if dj.sum() < 0 and dx.sum() < 0 and (dj > 0).sum() <= 3 and (dx > 0).sum() <= 3:
        return down
    return 'MIXED'


def gate_on(v, minute, q90):
    """Rows a guard lets quote: v above its own per-minute 90th percentile."""
    return np.isfinite(v) & (v > q90[minute])


def grid_q90(vb, va, minute):
    """Per-minute p90 of a guard's values on the grid rows, bid and ask pooled."""
    q = np.full(4, NAN)
    for m in range(4):
        x = np.r_[vb[minute == m], va[minute == m]]; x = x[np.isfinite(x)]
        q[m] = np.percentile(x, 90) if len(x) else NAN
    return q


def skill_by_day(J, guard):
    """Per read day: mean label on the rows the guard lets quote (bid + ask pooled), share quoted, bad share, rank corr per side."""
    mn = minute_of(J.t_rel.values)
    q = grid_q90(J[f'{guard}_bid'].values, J[f'{guard}_ask'].values, mn)
    out = {}
    for d in READ:
        s = (J.day == d).values
        ys, ons = [], []
        rc = {}
        for side in ('bid', 'ask'):
            v = J[f'{guard}_{side}'].values[s]; y = J[f'label_{side}'].values[s]; m = mn[s]
            on = gate_on(v, m, q); ys.append(y[on]); ons.append(on)
            ok = np.isfinite(v) & np.isfinite(y)
            rc[side] = float(pd.Series(v[ok]).rank().corr(pd.Series(y[ok]).rank())) if ok.sum() > 30 else NAN
        y = np.concatenate(ys); on = np.concatenate(ons)
        out[d] = dict(mean_on=float(y.mean()) if len(y) else NAN, share_on=float(on.mean()), bad_on=float(np.mean(y <= -2)) if len(y) else NAN,
                      n_on=int(len(y)), rank_corr=rc)
    return out, [float(x) for x in q]


# ---------------------------------------------------------------- controls

def selftest():
    rep = {}; ok = True

    def chk(name, cond):
        nonlocal ok
        rep[name] = bool(cond); ok = ok and bool(cond)
    _, b = b13a_controls()
    chk('C4_b13a_features', b.get('F1_features') is True)
    # C7 rules
    chk('C7_r120', label_r120([0.1] * 8) == 'IMPROVES' and label_r120([-0.1] * 8) == 'WORSE' and label_r120([0.1, -0.1] * 4) == 'NO-CHANGE'
        and label_r120([0.1] * 7) == 'n/a' and label_r120([0.2, 0.2, 0.2, 0.2, 0.2, -0.01, -0.01, -0.01]) == 'NO-CHANGE')
    chk('C7_r12x', label_r12x([1.0] * 9, [1.0] * 9) == 'BETTER' and label_r12x([-1.0] * 9, [-1.0] * 9) == 'WORSE'
        and label_r12x([1.0] * 9, [-1.0] * 9) == 'MIXED' and label_r12x([5.0] + [-0.1] * 8, [5.0] + [-0.1] * 8) == 'MIXED'
        and label_r12x([1.0] * 8, [1.0] * 9) == 'n/a')
    # C3 mapping on synthetic values (and its falsifier)
    rng = np.random.default_rng(1)
    v = rng.normal(0, 0.05, 4000).astype(np.float32); m = rng.integers(0, 4, 4000); v[:7] = -1.0; v[7:11] = np.nan
    L90 = np.array([0.0492, 0.0623, 0.0789, 0.0917]); L95 = np.array([0.0698, 0.0841, 0.1009, 0.1094])
    q90 = np.array([np.percentile(v[(m == i) & np.isfinite(v) & (v != -1)], 90) for i in range(4)])
    q95 = np.array([np.percentile(v[(m == i) & np.isfinite(v) & (v != -1)], 95) for i in range(4)])
    keep = np.zeros(4000, bool); keep[:7] = True
    w = map_values(v, m, keep, q90, q95, L90, L95)
    chk('C3_map_synthetic', mapping_ok(v, w, m, keep, q90, q95, L90, L95))
    chk('C3_map_falsifier', not mapping_ok(v, map_values(v, m, keep, q90 + 0.001, q95, L90, L95), m, keep, q90, q95, L90, L95))
    # gate_on / q90 on a planted grid: exactly the top 10% per minute quote
    vb = np.tile(np.arange(100, dtype=float), 4); va = vb.copy(); mm = np.repeat(np.arange(4), 100)
    q = grid_q90(vb, va, mm)
    chk('C7_gate', np.allclose(q, 89.1) and int(gate_on(vb, mm, q).sum()) == 40)
    rep['ok'] = ok
    print(json.dumps(rep))
    return ok


def _f32(x, rel):
    """The float32 nearest x on the side rel says ('gt' / 'le' / 'ge' / 'lt' x, compared as float64)."""
    y = np.float32(x)
    if rel == 'gt':
        return y if float(y) > x else np.nextafter(y, np.float32(np.inf))
    if rel == 'le':
        return y if float(y) <= x else np.nextafter(y, np.float32(-np.inf))
    if rel == 'ge':
        return y if float(y) >= x else np.nextafter(y, np.float32(np.inf))
    return y if float(y) < x else np.nextafter(y, np.float32(-np.inf))


def map_values(v, minute, keep, q90, q95, L90, L95):
    """Per minute, the increasing line through (q90 -> L90) and (q95 -> L95), stored as float32 so that the engine's comparisons
    (float32 promoted to float64 against the line) fall exactly where v's do: w > L90 iff v > q90, w >= L95 iff v >= q95."""
    v = np.asarray(v, np.float32); vd = v.astype(np.float64)
    slope = (L95[minute] - L90[minute]) / (q95[minute] - q90[minute])
    w = (L90[minute] + (vd - q90[minute]) * slope).astype(np.float32)
    wd = w.astype(np.float64)
    for m in range(4):
        s = (minute == m) & np.isfinite(vd)
        a90, b90 = _f32(L90[m], 'gt'), _f32(L90[m], 'le'); a95, b95 = _f32(L95[m], 'ge'), _f32(L95[m], 'lt')
        up = s & (vd > q90[m]) & (wd <= L90[m]); w[up] = a90
        dn = s & (vd <= q90[m]) & (wd > L90[m]); w[dn] = b90
        wd = w.astype(np.float64)
        hi = s & (vd >= q95[m]) & (wd < L95[m]); w[hi] = a95
        mid = s & (vd > q90[m]) & (vd < q95[m]) & (wd >= L95[m]); w[mid] = b95
        wd = w.astype(np.float64)
    w = np.where(np.isnan(v) | keep, v, w)
    return w.astype(np.float32)


def mapping_ok(v, w, minute, keep, q90, q95, L90, L95):
    """The engine's decisions on w equal the guard's own lines on v (float32 promoted to float64, as the engine compares)."""
    v = np.asarray(v, np.float32).astype(np.float64); w = np.asarray(w, np.float32).astype(np.float64)
    act = np.isfinite(v) & ~keep
    a = (w[act] > L90[minute[act]]) == (v[act] > q90[minute[act]])
    b = (w[act] >= L95[minute[act]]) == (v[act] >= q95[minute[act]])
    c = np.array_equal(np.isnan(v), np.isnan(w)) and np.array_equal(w[keep], v[keep])
    return bool(a.all() and b.all() and c)


# ---------------------------------------------------------------- steps

def step_features():
    os.makedirs(OUTD, exist_ok=True)
    rep = {}
    for d in DAYS:
        p = ex_path(d)
        if os.path.exists(p):
            sys.exit(f'REFUSED: {p} exists')
        e = features_day(d); e.to_parquet(p, index=False)
        rep[d] = dict(rows=int(len(e)), nan_share={c: float(e[c].isna().mean()) for c in EXTRA})
        print(d, len(e), flush=True); del e; io.mem_check(f'features {d}')
    io.write_json(f'{OUTD}/features_report.json', rep)


def load_grid():
    """The guard grid (file order) with EXTRA attached per day from that day's rows (only the grid's rows are kept)."""
    G = pd.read_parquet(GRID, columns=['day', 'mid', 'tau', 't_rel', 'v_bid', 'v_ask'] + FE)
    if sorted(G.day.unique()) != READ:
        sys.exit(f'REFUSED: the grid holds {sorted(G.day.unique())}')
    parts = []
    for d in READ:
        k = G.loc[G.day == d, ['mid', 'tau']]
        e = k.merge(pd.read_parquet(ex_path(d)), on=['mid', 'tau'], how='left', validate='one_to_one', indicator=True)
        if (e['_merge'] != 'both').any() or len(e) != len(k):
            sys.exit(f'REFUSED: C4 {d}: {int((e["_merge"] != "both").sum())} grid rows without a row in {ROWS}')
        e.index = k.index; parts.append(e[EXTRA])
    E = pd.concat(parts).reindex(G.index); del parts
    for c in EXTRA:
        G[c] = E[c].values
    miss = int(E.isna().all(axis=1).sum()); del E
    return G, miss


def step_build():
    for d in DAYS:
        for stem in ('preds_e60', 'preds_e60k', 'preds_e60km'):
            if os.path.exists(f'{S5}/{d}/{stem}.parquet'):
                sys.exit(f'REFUSED: {S5}/{d}/{stem}.parquet exists')
    G, miss = load_grid(); io.mem_check('grid')
    Y = {s: G[f'v_{s}'].clip(-10, 10).values for s in ('bid', 'ask')}
    XE = np.concatenate([G[FE].values.astype(np.float32), G[EXTRA].values.astype(np.float32)], axis=1)
    G.drop(columns=FE + EXTRA, inplace=True)
    XB = XE[:, :len(FE)]
    day = G.day.values
    rep = dict(grid_rows=int(len(G)), grid_rows_without_any_engineered_value=miss, thresholds={}, controls={})
    # C1 parity: this file's fit / apply = the live guard's preds_t60, on 11-03 (all 8 days) and 10-23 (LODO); falsifier: c + 1e-6
    par = {}
    for d, train in ((FRESH, np.ones(len(G), bool)), ('20261023', day != '20261023')):
        ms, cs = fit(XB[train], {s: Y[s][train] for s in Y})
        rows, live, X = day_matrix(d, FE)
        o = apply(ms, cs, live, X)
        ref = pd.read_parquet(f'{S5}/{d}/preds_t60.parquet', columns=['mid', 'tau', 'v_bid', 'v_ask'])
        al = (ref.mid.values == rows.mid.values).all() and (ref.tau.values == rows.tau.values).all()
        eq = al and same_f32(o['bid'], ref.v_bid.values) and same_f32(o['ask'], ref.v_ask.values)
        cs2 = {s: cs[s] + 1e-6 for s in cs}; o2 = apply(ms, cs2, live, X)
        fal = not (same_f32(o2['bid'], ref.v_bid.values) and same_f32(o2['ask'], ref.v_ask.values))
        par[d] = dict(equal=bool(eq), falsifier_differs=bool(fal), thresholds=cs)
        del ms, X, o, o2, ref; io.mem_check(f'parity {d}')
    rep['controls']['C1_parity'] = par
    if not all(p['equal'] and p['falsifier_differs'] for p in par.values()):
        io.write_json(f'{OUTD}/build_report.json', rep); sys.exit('REFUSED: C1 parity failed (see build_report.json)')
    # PERM: the 13 features' rows permuted within each day (grid only, for R120's falsifier)
    rng = np.random.default_rng(SEED_PERM)
    XP = XE.copy()
    for d in READ:
        idx = np.nonzero(day == d)[0]
        XP[idx, len(FE):] = XE[rng.permutation(idx), len(FE):]
    oof = pd.DataFrame(dict(day=G.day.values, mid=G.mid.values, tau=G.tau.values, t_rel=G.t_rel.values,
                            label_bid=G.v_bid.values, label_ask=G.v_ask.values))
    for c in ('emph_bid', 'emph_ask', 'perm_bid', 'perm_ask'):
        oof[c] = np.float32(np.nan)
    del G
    # EMPH LODO (and PERM LODO on the grid); EMPH on all 8 for 11-03
    for d in READ:
        tr = day != d; te = day == d
        ms, cs = fit(XE[tr], {s: Y[s][tr] for s in Y}); rep['thresholds'][d] = cs
        for s in ('bid', 'ask'):
            oof.loc[te, f'emph_{s}'] = (ms[s].predict(XE[te], num_threads=1) - cs[s]).astype(np.float32)
        rows, live, X = day_matrix(d, EMPH)
        write_preds(d, apply(ms, cs, live, X), rows); del ms, X
        mp, cp = fit(XP[tr], {s: Y[s][tr] for s in Y})
        for s in ('bid', 'ask'):
            oof.loc[te, f'perm_{s}'] = (mp[s].predict(XP[te], num_threads=1) - cp[s]).astype(np.float32)
        del mp; io.mem_check(f'fold {d}'); print('fold', d, cs, flush=True)
    ms, cs = fit(XE, Y); rep['thresholds'][FRESH] = cs
    rows, live, X = day_matrix(FRESH, EMPH)
    write_preds(FRESH, apply(ms, cs, live, X), rows)
    for s in ('bid', 'ask'):
        ms[s].save_model(f'{OUTD}/guard_emph_{s}.txt')
    del X
    # BASE on the grid rows = preds_t60's v (the live guard, out of fold)
    parts = []
    for d in READ:
        p = pd.read_parquet(f'{S5}/{d}/preds_t60.parquet', columns=['mid', 'tau', 'v_bid', 'v_ask'])
        p['day'] = d; parts.append(p)
    B = pd.concat(parts, ignore_index=True); del parts
    oof = oof.merge(B.rename(columns={'v_bid': 'base_bid', 'v_ask': 'base_ask'}), on=['day', 'mid', 'tau'], how='left', validate='one_to_one')
    oof.to_parquet(f'{OUTD}/oof_grid.parquet', index=False)
    rep['oof_rows'] = int(len(oof)); rep['oof_base_missing'] = int(oof.base_bid.isna().sum())
    io.write_json(f'{OUTD}/build_report.json', rep)
    print(json.dumps({k: rep[k] for k in ('grid_rows', 'oof_rows', 'oof_base_missing')}), flush=True)


def write_preds(d, o, rows):
    src = pd.read_parquet(f'{S5}/{d}/preds_t60.parquet', columns=['mid', 'tau', 'pred_1000', 'pred_250'])
    if not ((src.mid.values == rows.mid.values).all() and (src.tau.values == rows.tau.values).all()):
        sys.exit(f'REFUSED: C5 {d} rows differ from preds_t60')
    src['v_bid'] = o['bid']; src['v_ask'] = o['ask']
    src.to_parquet(f'{S5}/{d}/preds_e60.parquet', index=False)


def per_minute_q(stem, days):
    """Per-minute p90 / p95 of a start-known guard file's v (bid + ask pooled, rel in [0, 240)) over days (pf_minute_pct's rule)."""
    vs = {m: [] for m in range(4)}
    for d in days:
        p = pd.read_parquet(f'{S5}/{d}/{stem}.parquet', columns=['mid', 'tau', 'v_bid', 'v_ask'])
        rel = rel_of(d, p.mid.values, p.tau.values); ok = (rel >= 0) & (rel < 240); mn = minute_of(rel)
        for m in range(4):
            s = ok & (mn == m)
            x = np.r_[p.v_bid.values[s], p.v_ask.values[s]]; vs[m].append(x[np.isfinite(x)])
    q90 = np.array([np.percentile(np.concatenate(vs[m]), 90) for m in range(4)])
    q95 = np.array([np.percentile(np.concatenate(vs[m]), 95) for m in range(4)])
    return q90, q95


def step_map():
    L = json.load(open(LINES_FILE))['lines']; L90 = np.array(L['90'], float); L95 = np.array(L['95'], float)
    rep = dict(controls={})
    # C2: the rule reproduces the frozen lines
    t90, t95 = per_minute_q('preds_t60k', READ)
    rep['controls']['C2_lines'] = dict(rule_90=[round(float(x), 4) for x in t90], rule_95=[round(float(x), 4) for x in t95],
                                       frozen_90=list(L90), frozen_95=list(L95),
                                       equal=bool(np.allclose(np.round(t90, 4), L90, atol=1e-12) and np.allclose(np.round(t95, 4), L95, atol=1e-12)))
    if not rep['controls']['C2_lines']['equal']:
        io.write_json(f'{OUTD}/map_report.json', rep); sys.exit('REFUSED: C2 lines differ')
    q90, q95 = per_minute_q('preds_e60k', READ)
    rep['emph_q90'] = [float(x) for x in q90]; rep['emph_q95'] = [float(x) for x in q95]
    if not np.all(q95 > q90):
        sys.exit('REFUSED: EMPH p95 not above p90 in some minute')
    c3 = {}
    for d in DAYS:
        out = f'{S5}/{d}/preds_e60km.parquet'
        if os.path.exists(out):
            sys.exit(f'REFUSED: {out} exists')
        pk = pd.read_parquet(f'{S5}/{d}/preds_e60k.parquet'); p0 = pd.read_parquet(f'{S5}/{d}/preds_e60.parquet', columns=['v_bid', 'v_ask'])
        rel = rel_of(d, pk.mid.values, pk.tau.values); mn = minute_of(rel)
        ok_all = True; fal_all = True
        for s in ('bid', 'ask'):
            v = pk[f'v_{s}'].values.astype(np.float32)
            keep = (v == np.float32(-1.0)) & (p0[f'v_{s}'].values != np.float32(-1.0))
            w = map_values(v, mn, keep, q90, q95, L90, L95)
            ok_all = ok_all and mapping_ok(v, w, mn, keep, q90, q95, L90, L95)
            fal_all = fal_all and not mapping_ok(v, map_values(v, mn, keep, q90 + 0.001, q95, L90, L95), mn, keep, q90, q95, L90, L95)
            pk[f'v_{s}'] = w
        c3[d] = dict(ok=bool(ok_all), falsifier_breaks=bool(fal_all))
        if not (ok_all and fal_all):
            rep['controls']['C3_mapping'] = c3; io.write_json(f'{OUTD}/map_report.json', rep); sys.exit(f'REFUSED: C3 mapping {d}')
        pk.to_parquet(out, index=False)
    rep['controls']['C3_mapping'] = c3
    io.write_json(f'{OUTD}/map_report.json', rep)
    print(json.dumps(rep), flush=True)


def summary(tag, d):
    from .pf_inv_report import name
    from .pf_ladder_v3_build import ARM
    stem = 'preds_t60k' if tag.startswith(('b14r', 'i2')) else 'preds_e60km'
    s = json.load(open(f'{S4}/ladder_tokyoreal_custom_e2_{stem}_{tag}/{d}/summary.json'))
    return s, s['arms'][name(ARM)]


def step_report():
    from .pf_ladder_v3_build import ARM
    rep = dict(arm=ARM, controls={}, readings={})
    # C6 REF reproduces i2J / i2X
    c6 = {}
    for tm in ('J', 'X'):
        for d in DAYS:
            _, a = summary(f'b14r{tm}', d); _, b = summary(f'i2{tm}', d)
            c6[f'{tm}_{d}'] = all(abs(float(a[k]) - float(b[k])) < 1e-9 for k in ('gross_usd', 'mo1_usd', 'fills', 'shares'))
    rep['controls']['C6_replay'] = c6
    if not all(c6.values()):
        io.write_json(f'{OUTD}/report.json', rep); sys.exit('REFUSED: C6 the REF replays do not reproduce i2J / i2X')
    # envs as declared
    for tm in ('J', 'X'):
        for d in DAYS:
            for tag, stem in ((f'b14r{tm}', 'preds_t60k'), (f'b14e{tm}', 'preds_e60km')):
                s, _ = summary(tag, d); e = s['env']
                want = dict(PF_PREDS=stem, PF_ARMS=ARM, PF_TIMING='e2', PF_TXTIME='ws', PF_TXKEY='tk', PF_RL_LPLACE='40', PF_RL_LCANCEL='22',
                            PF_RL_LPENDCANCEL='95', PF_RL_LFILL='100', PF_RL_DDEC='2')
                want.update(dict(PF_TXJIT='frame') if tm == 'J' else dict(PF_TXSHIFT='-50', PF_QACT='max'))
                bad = {k: (e.get(k), v) for k, v in want.items() if e.get(k) != v}
                if bad or e.get('PF_R60_FILE'):
                    sys.exit(f'REFUSED: {tag} {d} env differs: {bad}')
    # R120
    J = pd.read_parquet(f'{OUTD}/oof_grid.parquet')
    sk = {g: skill_by_day(J, g) for g in ('base', 'emph', 'perm')}
    dE = [sk['emph'][0][d]['mean_on'] - sk['base'][0][d]['mean_on'] for d in READ]
    dP = [sk['perm'][0][d]['mean_on'] - sk['base'][0][d]['mean_on'] for d in READ]
    r120 = label_r120(dE); fal = label_r120(dP)
    rep['readings']['R120'] = 'VOID' if fal == 'IMPROVES' else r120
    rep['r120'] = dict(delta_emph=dE, delta_perm=dP, interval_emph=interval(dE, T7), interval_perm=interval(dP, T7), falsifier=fal,
                       per_day={g: sk[g][0] for g in sk}, grid_q90={g: sk[g][1] for g in sk})
    # R121 / R122
    rows = []
    for tm in ('J', 'X'):
        for d in DAYS:
            _, a = summary(f'b14r{tm}', d); _, b = summary(f'b14e{tm}', d)
            rows.append(dict(timing=tm, day=d, **{f'ref_{k}': float(a[k]) for k in ('gross_usd', 'mo1_usd', 'mo60_usd', 'fills', 'shares', 'windows', 'windows_positive')},
                             **{f'emph_{k}': float(b[k]) for k in ('gross_usd', 'mo1_usd', 'mo60_usd', 'fills', 'shares', 'windows', 'windows_positive')}))
    T = pd.DataFrame(rows); T.to_csv(f'{OUTD}/replay_days.csv', index=False)
    dd = {}
    for k in ('gross_usd', 'mo1_usd'):
        for tm in ('J', 'X'):
            t = T[T.timing == tm]; dd[(k, tm)] = (t[f'emph_{k}'] - t[f'ref_{k}']).values
    rep['readings']['R121'] = label_r12x(dd[('gross_usd', 'J')], dd[('gross_usd', 'X')])
    rep['readings']['R122'] = label_r12x(dd[('mo1_usd', 'J')], dd[('mo1_usd', 'X')])
    rep['replay'] = {f'{k}_{tm}': dict(sum_ref=float(T[T.timing == tm][f'ref_{k}'].sum()), sum_emph=float(T[T.timing == tm][f'emph_{k}'].sum()),
                                       delta_sum=float(v.sum()), days_positive=int((v > 0).sum()), interval=interval(v, T8),
                                       fresh_1103=float(v[DAYS.index(FRESH)]))
                     for (k, tm), v in dd.items()}
    rep['replay_fills'] = {tm: dict(ref=float(T[T.timing == tm].ref_fills.sum()), emph=float(T[T.timing == tm].emph_fills.sum()),
                                    ref_shares=float(T[T.timing == tm].ref_shares.sum()), emph_shares=float(T[T.timing == tm].emph_shares.sum()))
                           for tm in ('J', 'X')}
    io.write_json(f'{OUTD}/report.json', rep)
    lines = ['B14 pressure guard (as of the run; see report.json)', f"R120 {rep['readings']['R120']}  R121 {rep['readings']['R121']}  R122 {rep['readings']['R122']}"]
    lines.append('R120 per day (EMPH - BASE, mean label on quoted grid rows, c/sh): ' + ', '.join(f'{d[4:]} {x:+.4f}' for d, x in zip(READ, dE)))
    lines.append('R120 falsifier PERM - BASE: ' + ', '.join(f'{d[4:]} {x:+.4f}' for d, x in zip(READ, dP)) + f' -> {fal}')
    for (k, tm), v in dd.items():
        lines.append(f'{k} {tm}: ' + ', '.join(f'{d[4:]} {x:+.2f}' for d, x in zip(DAYS, v)) + f' | sum {v.sum():+.2f}, days > 0 {(v > 0).sum()}/9')
    open(f'{OUTD}/report.txt', 'w').write('\n'.join(lines) + '\n'); print('\n'.join(lines))


if __name__ == '__main__':
    mode = sys.argv[1] if len(sys.argv) > 1 else ''
    if mode == 'selftest':
        sys.exit(0 if selftest() else 1)
    if not selftest():
        sys.exit('REFUSED: selftest failed')
    {'features': step_features, 'build': step_build, 'map': step_map, 'report': step_report}[mode]()
