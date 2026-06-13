"""harness_v3 — clean walk-forward harness for the linear β-residual line.

Replaces the s94b.grouped_oof / feature_reengineering.ceiling path. Built
to FEATURE_REENGINEERING_PLAN.md v3 §2 after the Step 94–102 retraction
(flawed harness). Production LGBM unaffected.

WHY WALK-FORWARD (not the s94b shuffled grouped-OOF): §4 (the only real
EV) compares a *tradeable forward* linear book to production V3.1's
honest-forward per-cycle PnL. "Strictly-past" σ_idio and preprocessing
are only well-defined on a chronological calendar. The shuffled
grouped-OOF was the (now-retracted) D1 stationary-ceiling tool, not a
profitable-system test.

§2a preprocess  : static frozen TRANSFORM_MAP (NO live kurtosis routing,
                  fix #10). symz = per-symbol STRICT-PAST rolling z
                  (shift(1)); then per-cycle rank->inv-normal. xsr =
                  per-cycle rank->inv-normal directly (already-normalized
                  / cross-sectionally-meaningful feats). NaN -> per-symbol
                  strict-past rolling median; no past => row dropped (NO
                  cross-symbol fill).
§2b target fix  : σ_idio recomputed per-symbol STRICT-PAST causal
                  (alpha_beta.shift(1).rolling(W).std()), NO cross-symbol
                  fallback. Symbol w/o W-history is INELIGIBLE that fold.
                  tz rebuilt from it; the frozen panel sigma_idio is NEVER
                  consumed.
§2c envelope    : Ridge α∈{1,3,10,30} (α picked on the fold's cal window)
                  + LGBM early-stopped on cal. Ceiling = best honest
                  member.
§2d inference   : block-bootstrap CI (block≥6, n_boot≥1000); P2
                  within-selected-universe placebo.
§2e universe    : native per-fold coverage on ONE common 4h dec grid.
§2f cost        : gate at maker (~1 bps/unit); taker disclosed bound.
§2g self-checks : BLOCKING (abort) — (1) preprocessing is prefix-causal
                  to 1e-12; (2) post-preprocess PIT |corr(feat, next-cycle
                  alpha_beta)| < 0.10; (3) σ_idio uses no future / no
                  cross-symbol info. Run `python3 -m
                  linear_model.composite_study.harness_v3` for the
                  1-fold self-test (the reviewable unit before §4).
"""
from __future__ import annotations
import importlib.util
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb

warnings.filterwarnings("ignore")
REPO = Path("/home/yuqing/ctaNew")
sys.path.insert(0, str(REPO))


def _imp(n, r):
    s = importlib.util.spec_from_file_location(n, REPO / r)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


s94 = _imp("s94", "linear_model/scripts/94_info_ceiling_d1.py")
s92 = s94._imp("s92", "linear_model/scripts/92_tsmom_base.py")
from ml.research.alpha_v4_xs_1d import _multi_oos_splits, _slice  # noqa: E402

load_close, trail = s92.load_close, s92.trailing_ret_pit
L, BLOCK, COST, ANN, OOS = s92.L, s92.BLOCK, s92.COST, s92.ANN, s92.OOS
MAKER = COST * (1.0 / 4.5)              # ≈ HL-maker (~1 bps/unit) — gate cost
TAKER = COST * (3.5 / 4.5)             # disclosed pessimistic bound
GATE = 1.5

# strict-past σ_idio rolling window (4h cycles): 120≈20d, min 30≈5d. FROZEN.
SIG_W, SIG_MP = 120, 30
# strict-past per-symbol z window for symz feats. FROZEN.
SYMZ_W, SYMZ_MP = 504, 120
PIT_THRESH = 0.10                      # project canonical look-ahead sniff

LEAK = {"return_pct", "btc_ret_fwd", "alpha_beta", "exit_time", "symbol",
        "open_time", "fold", "s_t", "tz", "sigma_idio"}

# ─── §2a STATIC FROZEN transform map (domain-reasoned; no live routing) ───
# symz : symbol-scale-specific level/return/vol/slope/change → per-symbol
#        STRICT-PAST z, then per-cycle rank→inv-normal.
# xsr  : already-normalized (z / pctile / xs_rank) or natively cross-section
#        comparable (funding / beta / corr / dominance) → per-cycle
#        rank→inv-normal directly.
TRANSFORM_MAP = {
    "autocorr_pctile_7d": "xsr",
    "return_1d": "symz", "return_8h": "symz",
    "atr_pct": "symz", "vwap_slope_96": "symz",
    "funding_rate": "xsr", "funding_rate_z_7d": "xsr",
    "funding_rate_1d_change": "symz",
    "bars_since_high": "symz", "bars_since_high_xs_rank": "xsr",
    "beta_btc_pit": "xsr",
    "dom_btc_change_288b": "symz", "dom_btc_z_1d": "xsr",
    "beta_to_btc_change_5d": "symz", "corr_to_btc_1d": "xsr",
    "corr_to_btc_change_3d": "symz",
    "idio_vol_to_btc_1h": "symz", "idio_vol_to_btc_1d": "symz",
    "obv_z_1d": "xsr", "vol_zscore_4h_over_7d": "xsr",
    "s_t": "symz",
}


def _xsrank_invnorm(df, cols):
    """Per-cycle (per open_time) rank → inv-normal. PIT: contemporaneous
    cross-section only, no future, no per-symbol leakage."""
    out = df[cols].copy()
    g = df.groupby("open_time")
    for c in cols:
        r = g[c].rank(method="average")
        n = g[c].transform("count")
        out[c] = norm.ppf(((r - 0.5) / n).clip(1e-6, 1 - 1e-6)).astype(
            "float64")
    return out


def _symz_strictpast(df, cols):
    """Per-symbol STRICT-PAST rolling z = (x − past_mean)/past_std with
    shift(1) (current row excluded). NaN → per-symbol strict-past rolling
    median; still-NaN rows flagged for drop (NO cross-symbol fill)."""
    d = df.sort_values(["symbol", "open_time"])
    out = pd.DataFrame(index=d.index)
    drop = pd.Series(False, index=d.index)
    gb = d.groupby("symbol", sort=False)
    for c in cols:
        x = gb[c]
        sh = x.shift(1)
        mu = sh.groupby(d["symbol"], sort=False).transform(
            lambda s: s.rolling(SYMZ_W, min_periods=SYMZ_MP).mean())
        sd = sh.groupby(d["symbol"], sort=False).transform(
            lambda s: s.rolling(SYMZ_W, min_periods=SYMZ_MP).std())
        z = (d[c] - mu) / sd.where(sd > 1e-12)
        med = sh.groupby(d["symbol"], sort=False).transform(
            lambda s: s.rolling(SYMZ_W, min_periods=SYMZ_MP).median())
        z = z.fillna((d[c] - med) / sd.where(sd > 1e-12))
        out[c] = z
        drop |= z.isna()
    return out.reindex(df.index), drop.reindex(df.index).fillna(True)


def preprocess(df):
    """Static-map preprocess on the WHOLE chronological panel. Every
    transform is prefix-causal (symz = per-symbol shift(1).rolling; xsr =
    within-open_time cross-section), so whole-panel ≡ strictly-past
    train-only refit. selfcheck_prefix_causal *empirically verifies* this
    equivalence at interior cut points (not by construction alone).
    Returns (feat_df, drop_mask)."""
    feats = [c for c in df.columns if c in TRANSFORM_MAP]
    sym_cols = [c for c in feats if TRANSFORM_MAP[c] == "symz"]
    xsr_cols = [c for c in feats if TRANSFORM_MAP[c] == "xsr"]
    parts = []
    drop = pd.Series(False, index=df.index)
    if sym_cols:
        zz, dz = _symz_strictpast(df, sym_cols)
        rk = _xsrank_invnorm(df.assign(**{c: zz[c] for c in sym_cols}),
                             sym_cols)
        parts.append(rk)
        drop |= dz
    if xsr_cols:
        parts.append(_xsrank_invnorm(df, xsr_cols))
    F = pd.concat(parts, axis=1)[feats]
    drop |= F.isna().any(axis=1)
    return F, drop


# ─── §2b STRICT-PAST per-symbol σ_idio (NO cross-symbol fallback) ───
def strict_sigma_idio(df):
    """σ_idio[sym,t] = std of the symbol's OWN past non-overlapping idio
    returns: alpha_beta.shift(1).rolling(SIG_W, min SIG_MP).std() per
    symbol. Causal (shift1 on the 4h non-overlap grid ⇒ the shifted
    cycle's exit_time == current open_time, i.e. just-closed & observable).
    No history ⇒ NaN ⇒ symbol INELIGIBLE that cycle (no cross-symbol
    median, fix #7). The frozen panel sigma_idio is never touched."""
    d = df.sort_values(["symbol", "open_time"])
    sig = d.groupby("symbol", sort=False)["alpha_beta"].transform(
        lambda s: s.shift(1).rolling(SIG_W, min_periods=SIG_MP).std())
    return sig.reindex(df.index)


def load_panel():
    """4h dec grid, OOS folds 1–9 (chronological _multi_oos_splits/_slice),
    native universe, raw features + alpha_beta + s_t. NO frozen
    sigma_idio/tz consumed (tz is rebuilt per fold from strict_sigma_idio)."""
    pan = pd.read_parquet(s92.PANEL)
    pan["open_time"] = pd.to_datetime(pan["open_time"], utc=True)
    pan["exit_time"] = pd.to_datetime(pan["exit_time"], utc=True)
    hl = pd.read_csv(s92.HL_MAP)
    keep = set(hl[(hl.on_hl) & (hl.hl_day_vol_usd >= 2e6)]["symbol"])
    syms = sorted(s for s in pan["symbol"].unique()
                  if s in keep and s not in {"BIOUSDT", "VVVUSDT",
                                             "BTCUSDT"})
    btc = load_close("BTCUSDT").set_index("open_time")["close"]
    btc_rL = trail(btc).rename("ret_btc_L")
    parts = []
    for s in syms:
        c = load_close(s)
        if c is None or len(c) < L + 1000:
            continue
        c = c.set_index("open_time")
        df = pd.concat([trail(c["close"]).rename("ret_asset_L"), btc_rL],
                       axis=1).reset_index()
        df["symbol"] = s
        parts.append(df)
    sig = pd.concat(parts, ignore_index=True)
    sig["open_time"] = pd.to_datetime(sig["open_time"], utc=True)
    d = pan.merge(sig, on=["symbol", "open_time"], how="inner")
    d["s_t"] = (d["ret_asset_L"]
                - d["beta_btc_pit"] * d["ret_btc_L"]).astype("float64")
    d = d.dropna(subset=["s_t", "alpha_beta"]).sort_values(
        ["symbol", "open_time"]).reset_index(drop=True)
    folds = _multi_oos_splits(d)
    d["fold"] = -1
    for fid in range(len(folds)):
        d.loc[_slice(d, folds[fid])[2].index, "fold"] = fid
    oos = d[d["fold"].isin(OOS)].copy()
    grid = sorted(oos["open_time"].unique())[::BLOCK]
    dec = oos[oos["open_time"].isin(set(grid))].sort_values(
        ["symbol", "open_time"]).reset_index(drop=True)
    return dec, folds, btc, pan


# ─── §2c model envelope ───
def model_envelope(Xtr, ytr, Xca, yca, Xte):
    """Ridge α∈{1,3,10,30} (α by best cal-window IC) + LGBM early-stopped
    on the cal window. Returns {member: te_pred}. Ceiling = best honest
    member (decided downstream on the gate metric, not here)."""
    out = {}
    sc = StandardScaler().fit(Xtr)
    Xtr_s, Xca_s, Xte_s = sc.transform(Xtr), sc.transform(Xca), sc.transform(Xte)
    best_a, best_ic = None, -9.0
    for a in (1.0, 3.0, 10.0, 30.0):
        r = Ridge(alpha=a).fit(Xtr_s, ytr)
        ic = pd.Series(r.predict(Xca_s)).corr(
            pd.Series(yca), method="spearman")
        out[f"ridge_a{int(a)}"] = r.predict(Xte_s)
        if ic is not None and ic > best_ic:
            best_ic, best_a = ic, a
    out["ridge_best"] = out[f"ridge_a{int(best_a)}"]
    m = lgb.LGBMRegressor(num_leaves=63, n_estimators=2000,
                          learning_rate=0.02, subsample=0.8,
                          colsample_bytree=0.8, random_state=0,
                          n_jobs=-1, verbose=-1)
    m.fit(Xtr, ytr, eval_set=[(Xca, yca)],
          callbacks=[lgb.early_stopping(80, verbose=False)])
    out["lgbm_es"] = m.predict(Xte, num_iteration=m.best_iteration_)
    return out


def walk_forward(dec, folds, members=("ridge_best", "lgbm_es"), verbose=True):
    """Per-fold: strict-past σ_idio → tz; preprocess; envelope. Returns dec
    rows that were predicted, with per-member OOF preds + tz_strict +
    sig_strict, all on the common 4h dec grid."""
    F_all, drop_all = preprocess(dec)
    sig = strict_sigma_idio(dec)
    feats = list(F_all.columns)
    base = dec.copy()
    base["sig_strict"] = sig
    base["tz_strict"] = (base["alpha_beta"] / base["sig_strict"]).clip(-5, 5)
    elig = (~drop_all) & base["sig_strict"].notna() & (
        base["sig_strict"] > 1e-12) & base["tz_strict"].notna()
    recs = []
    for fid in OOS:
        fo = folds[fid]
        tr_m = ((dec["open_time"] < fo["cal_start"]) & elig)
        ca_m = ((dec["open_time"] >= fo["cal_start"]) &
                (dec["open_time"] < fo["cal_end"]) &
                (dec["exit_time"] < fo["test_start"]) & elig)
        te_m = ((dec["fold"] == fid) & elig)
        if tr_m.sum() < 800 or ca_m.sum() < 50 or te_m.sum() < 20:
            if verbose:
                print(f"  fold {fid}: skipped (tr={tr_m.sum()} "
                      f"ca={ca_m.sum()} te={te_m.sum()})", flush=True)
            continue
        Xtr, Xca, Xte = (F_all.loc[tr_m, feats].to_numpy(float),
                         F_all.loc[ca_m, feats].to_numpy(float),
                         F_all.loc[te_m, feats].to_numpy(float))
        ytr = base.loc[tr_m, "tz_strict"].to_numpy(float)
        yca = base.loc[ca_m, "tz_strict"].to_numpy(float)
        pr = model_envelope(Xtr, ytr, Xca, yca, Xte)
        r = base.loc[te_m, ["symbol", "open_time", "fold", "alpha_beta",
                            "sig_strict", "tz_strict"]].copy()
        for mb in members:
            r[mb] = pr[mb]
        recs.append(r)
        if verbose:
            print(f"  fold {fid}: tr={tr_m.sum()} ca={ca_m.sum()} "
                  f"te={te_m.sum()} syms={r.symbol.nunique()}", flush=True)
    return pd.concat(recs, ignore_index=True) if recs else pd.DataFrame()


# ─── §2d inference helpers ───
def block_bootstrap_ci(x, stat, block=7, n_boot=1000, seed=0):
    x = np.asarray(x, float)
    n = len(x)
    if n < block + 1:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    nb = int(np.ceil(n / block))
    out = np.empty(n_boot)
    for b in range(n_boot):
        st = rng.integers(0, n - block + 1, nb)
        out[b] = stat(np.concatenate([x[s:s + block] for s in st])[:n])
    return (float(np.percentile(out, 2.5)),
            float(np.percentile(out, 97.5)))


def sharpe(x):
    x = np.asarray(x, float)
    return float(x.mean() / x.std(ddof=1) * ANN) if x.std(ddof=1) > 1e-12 \
        else np.nan


def linear_book(pred_frame, predcol, cost=MAKER):
    """Naive sign(pred) equal-weight per-cycle β-residual book on the
    common 4h dec grid. Returns per-cycle net bps DataFrame [time, net,
    gross, fold]."""
    f = pred_frame.sort_values(["symbol", "open_time"]).copy()
    f["pos"] = np.sign(f[predcol])
    f.loc[f["pos"] == 0, "pos"] = 1.0
    n = f.groupby("open_time")["symbol"].transform("count")
    f["w"] = f["pos"] / n
    f["dw"] = f.groupby("symbol")["w"].diff().abs().fillna(f["w"].abs())
    p = f.groupby(["open_time", "fold"]).apply(
        lambda g: pd.Series({
            "gross": (g["w"] * g["alpha_beta"]).sum() * 1e4,
            "cost": g["dw"].sum() * cost})).reset_index()
    p["net"] = p["gross"] - p["cost"]
    return p.sort_values("open_time").reset_index(drop=True)


def p2_placebo(pred_frame, predcol, n_perm=1000, cost=MAKER, seed=0):
    """P2: permute the prediction WITHIN the selected universe per cycle
    (destroys the cross-sectional ranking, keeps marginals/breadth).
    Returns the placebo net-Sharpe distribution."""
    rng = np.random.default_rng(seed)
    base = pred_frame[["symbol", "open_time", "fold", "alpha_beta",
                       predcol]].copy()
    out = np.empty(n_perm)
    for k in range(n_perm):
        b = base.copy()
        b[predcol] = b.groupby("open_time")[predcol].transform(
            lambda s: rng.permutation(s.values))
        out[k] = sharpe(linear_book(b, predcol, cost)["net"].to_numpy())
    return out


# ─── §2g BLOCKING self-checks ───
def selfcheck_prefix_causal(dec, sample_syms=8, n_cuts=5, seed=0):
    """(1) GENUINE prefix-causal test: for each sampled symbol, at several
    INTERIOR cut points t*, recompute the FULL preprocess() (symz∘xsrank
    AND the direct-xsr feats) on the strictly-past prefix open_time ≤ t*
    only, and assert the prefix value at row t* equals the WHOLE-PANEL
    preprocess() value at that same (symbol,time) row to 1e-12. This
    actually proves whole-panel preprocess ≡ strictly-past refit (catches
    any future/cross-fold smuggling), not the near-vacuous last-row check."""
    full, dfull = preprocess(dec)
    feats = list(full.columns)
    rng = np.random.default_rng(seed)
    syms = rng.choice(dec["symbol"].unique(),
                      min(sample_syms, dec["symbol"].nunique()),
                      replace=False)
    worst = 0.0
    tested = 0
    for s in syms:
        ds = dec[dec["symbol"] == s].sort_values("open_time")
        if len(ds) < SYMZ_MP + 60:
            continue
        for frac in np.linspace(0.45, 0.97, n_cuts):
            i = int(len(ds) * frac)
            t_star = ds["open_time"].iloc[i]
            row_idx = ds.index[i]
            # strictly-past prefix across ALL symbols (xsr needs the
            # contemporaneous cross-section, which is observable at t*):
            pref = dec[dec["open_time"] <= t_star]
            fp, _ = preprocess(pref)
            if row_idx not in fp.index:
                continue
            a = full.loc[row_idx, feats].to_numpy(float)
            b = fp.loc[row_idx, feats].to_numpy(float)
            m = np.isfinite(a) & np.isfinite(b)
            if m.any():
                worst = max(worst, float(np.max(np.abs(a[m] - b[m]))))
            tested += 1
    return (worst < 1e-12 and tested > 0), worst


def selfcheck_pit(dec):
    """(2) post-preprocess PIT: |corr(transformed feat, the SAME symbol's
    NEXT-cycle alpha_beta)| < 0.10 for every feature (project canonical
    look-ahead sniff). Next-cycle = the realized future ⇒ a transformed
    PIT feature must not correlate with it beyond noise."""
    F, drop = preprocess(dec)
    d = dec.sort_values(["symbol", "open_time"]).copy()
    d["fwd1"] = d.groupby("symbol")["alpha_beta"].shift(-1)
    F = F.reindex(d.index)
    keep = (~drop.reindex(d.index).fillna(True)) & d["fwd1"].notna()
    fwd = pd.Series(d.loc[keep, "fwd1"].values)
    afwd = fwd.abs()
    worst_c, worst_f = 0.0, None
    for c in F.columns:
        v = pd.Series(F.loc[keep, c].values)
        # linear sniff AND magnitude/tail sniff (a leak that scales with
        # σ_idio can hide in |feat| vs |future residual|):
        for lab, x, y in ((f"{c}", v, fwd), (f"|{c}|", v.abs(), afwd)):
            cc = x.corr(y, method="spearman")
            if cc is not None and abs(cc) > worst_c:
                worst_c, worst_f = abs(cc), lab
    return worst_c < PIT_THRESH, worst_c, worst_f


def selfcheck_target_strictpast(dec, sample_syms=5, seed=0):
    """(3) σ_idio uses NO future and NO cross-symbol info: rebuild it from
    a per-symbol prefix-masked array; must equal strict_sigma_idio to
    1e-12, AND assert the shifted cycle's exit_time ≤ current open_time
    (just-closed, observable), AND assert zero cross-symbol fill (NaN ⇒
    dropped, never imputed)."""
    sig = strict_sigma_idio(dec)
    d = dec.sort_values(["symbol", "open_time"]).copy()
    rng = np.random.default_rng(seed)
    syms = rng.choice(d["symbol"].unique(),
                      min(sample_syms, d["symbol"].nunique()),
                      replace=False)
    worst, exit_ok, xsym = 0.0, True, 0.0
    for s in syms:
        ds = d[d["symbol"] == s].sort_values("open_time")
        man = ds["alpha_beta"].shift(1).rolling(
            SIG_W, min_periods=SIG_MP).std()
        a = sig.reindex(ds.index).to_numpy(float)
        b = man.to_numpy(float)
        msk = np.isfinite(a) & np.isfinite(b)
        if msk.any():
            worst = max(worst, float(np.max(np.abs(a[msk] - b[msk]))))
        # EXPLICIT no-cross-symbol assert (not a comment): recompute σ on
        # a frame containing ONLY this symbol — if any cross-symbol info
        # were used, the isolated value would differ from the panel value.
        iso = strict_sigma_idio(d[d["symbol"] == s]).reindex(ds.index
                                                             ).to_numpy(float)
        mi = np.isfinite(a) & np.isfinite(iso)
        if mi.any():
            xsym = max(xsym, float(np.max(np.abs(a[mi] - iso[mi]))))
        et = ds["exit_time"].shift(1)
        ot = ds["open_time"]
        v = et.notna()
        exit_ok &= bool((et[v] <= ot[v]).all())
    return (worst < 1e-12 and exit_ok and xsym < 1e-12), worst, exit_ok, xsym


def run_selfchecks(dec):
    print("\n── §2g BLOCKING self-checks ──", flush=True)
    c1, w1 = selfcheck_prefix_causal(dec)
    print(f"  (1) prefix-causal preprocess : "
          f"{'PASS' if c1 else 'FAIL'}  (max|Δ|={w1:.2e}, need <1e-12)",
          flush=True)
    c2, w2, wf = selfcheck_pit(dec)
    print(f"  (2) post-preprocess PIT      : "
          f"{'PASS' if c2 else 'FAIL'}  (worst |corr|={w2:.4f} on "
          f"'{wf}', need <{PIT_THRESH})", flush=True)
    c3, w3, eok, xs = selfcheck_target_strictpast(dec)
    print(f"  (3) σ_idio strict-past       : "
          f"{'PASS' if c3 else 'FAIL'}  (prefix max|Δ|={w3:.2e}; "
          f"exit_time≤open_time={eok}; no-xsym max|Δ|={xs:.2e})",
          flush=True)
    ok = c1 and c2 and c3
    print(f"  → {'ALL PASS — harness trustworthy' if ok else 'BLOCKED — '
          'do NOT score; fix before any §4/§3.5 run'}", flush=True)
    return ok


def main():
    print("=" * 92, flush=True)
    print("  harness_v3 — 1-fold self-test (reviewable unit before §4)",
          flush=True)
    print("=" * 92, flush=True)
    t0 = time.time()
    dec, folds, btc, pan = load_panel()
    print(f"  panel: rows={len(dec)} syms={dec.symbol.nunique()} "
          f"cycles={dec.open_time.nunique()} "
          f"span {dec.open_time.min().date()}→{dec.open_time.max().date()} "
          f"feats={sum(c in TRANSFORM_MAP for c in dec.columns)}",
          flush=True)
    ok = run_selfchecks(dec)
    if not ok:
        print(f"\n  ABORT: self-checks failed. Total {time.time()-t0:.0f}s",
              flush=True)
        return
    # smoke: a single fold end-to-end (cheap), confirm preds are sane
    f1 = [f for f in OOS][len(OOS) // 2]
    pf = walk_forward(dec, folds, verbose=False)
    if len(pf):
        sub = pf[pf["fold"] == f1]
        ic = pd.Series(sub["ridge_best"]).corr(
            sub["tz_strict"], method="spearman")
        print(f"\n  smoke fold {f1}: rows={len(sub)} "
              f"ridge_best IC vs tz_strict={ic:+.4f} "
              f"(sane, NOT a verdict)", flush=True)
    print(f"\n  harness_v3 READY. Total {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
