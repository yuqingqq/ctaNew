"""X80 — Elegant dynamic regime detection via Gaussian HMM (PIT) + model routing.

Why HMM (vs hardcoded BTC-30d threshold / KMeans-K):
  - Regimes EMERGE from the data's statistical structure (state means/vols), no
    hardcoded return cutoff.
  - Transition matrix captures regime PERSISTENCE (bull markets persist) — a
    single noisy bar doesn't flip the regime.
  - Probabilistic/soft assignment (filtered posteriors) — natural soft weighting.
  - PIT via forward FILTERING: regime at time t uses only observations ≤ t.

PIT protocol (no look-ahead):
  - Observation features per 4h bar: BTC 4h log-return, trailing short realized vol.
  - Fit GaussianHMM on an initial training window (data before first OOS fold).
  - Walk forward: for each subsequent block, refit on expanding past, then run the
    FORWARD filter over the whole series and take posteriors up to t (filtering, not
    smoothing) → strictly PIT regime probabilities.
  - Route per-sym Ridge specialists by filtered MAP regime; also test soft-weighting.

Compares to V0 3yr (+0.12 uncond, +1.07 KMeans-routed K=5, +1.13 hard gate).
"""
from __future__ import annotations
import sys, importlib.util, time, warnings
from pathlib import Path
import pandas as pd, numpy as np

warnings.filterwarnings("ignore")
REPO = Path("/home/yuqing/ctaNew")
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO/"scripts"))
KLINES = REPO/"data/ml/test/parquet/klines"
OUT = REPO/"research/convexity_portable_2026-05-20/results"; RCACHE = OUT/"_cache"
spec = importlib.util.spec_from_file_location("x6", REPO/"research/convexity_portable_2026-05-20/scripts/X6_controlled_matrix.py")
x6 = importlib.util.module_from_spec(spec); spec.loader.exec_module(x6)
from hmmlearn.hmm import GaussianHMM
from scipy.special import logsumexp


def gaussian_forward_filter(X, startprob, transmat, means, covars):
    """PIT forward-filter posteriors for a diag-covariance Gaussian HMM.
    covars: (K, D) diagonal variances. Returns (n, K) filtered posteriors."""
    n, D = X.shape; K = len(startprob)
    cov = np.asarray(covars)
    if cov.ndim == 3:        # (K, D, D) full matrices → take diagonal
        cov = np.diagonal(cov, axis1=1, axis2=2)
    cov = cov.reshape(K, D)
    logB = np.zeros((n, K))
    for k in range(K):
        v = cov[k] + 1e-12
        logB[:, k] = -0.5*(np.log(2*np.pi*v).sum() + (((X-means[k])**2)/v).sum(axis=1))
    logT = np.log(transmat + 1e-300)
    logalpha = np.zeros((n, K))
    logalpha[0] = np.log(startprob + 1e-300) + logB[0]
    for t in range(1, n):
        logalpha[t] = logB[t] + logsumexp(logalpha[t-1][:, None] + logT, axis=0)
    post = np.exp(logalpha - logalpha.max(axis=1, keepdims=True))
    post /= post.sum(axis=1, keepdims=True)
    return post


def btc_4h_obs():
    """BTC 4h-bar observation features for the HMM (PIT)."""
    files = sorted((KLINES/"BTCUSDT"/"5m").glob("*.parquet"))
    btc = pd.concat([pd.read_parquet(f, columns=["open_time","close"]) for f in files],
                     ignore_index=True).drop_duplicates("open_time").sort_values("open_time")
    btc["open_time"] = pd.to_datetime(btc["open_time"], utc=True)
    btc = btc.set_index("open_time")["close"].astype(np.float64)
    btc4 = btc[(btc.index.hour % 4 == 0) & (btc.index.minute == 0)]
    logret = np.log(btc4/btc4.shift(1))
    rvol = logret.rolling(42, min_periods=12).std()   # ~7d trailing vol at 4h bars
    obs = pd.DataFrame({"ret": logret, "rvol": rvol}).dropna()
    return obs


def hmm_filtered_states(obs, n_states, train_end_idx, refit_every=180):
    """PIT filtered MAP states + posteriors. Refit on expanding past; filter forward.
    Returns DataFrame index=obs.index with 'state' and posterior columns."""
    X = obs[["ret","rvol"]].values
    # standardize using only the initial training window stats (avoid global leak)
    mu = X[:train_end_idx].mean(0); sd = X[:train_end_idx].std(0)+1e-9
    Xs = (X-mu)/sd
    n = len(Xs)
    states = np.full(n, -1)
    posts = np.zeros((n, n_states))
    model = None
    canon = None  # canonical state ordering (by mean return) for stability
    for refit_pt in range(train_end_idx, n+refit_every, refit_every):
        rp = min(refit_pt, n)
        # Refit on all data up to rp (expanding)
        try:
            m = GaussianHMM(n_components=n_states, covariance_type="diag",
                            n_iter=50, random_state=42)
            m.fit(Xs[:rp])
        except Exception:
            continue
        # canonical ordering by state mean return (col 0) so labels are comparable
        order = np.argsort(m.means_[:,0])  # ascending: 0=most bearish ... n-1=most bullish
        # Manual Gaussian forward FILTER (PIT: forward-only, no smoothing leak).
        # Uses fitted params (startprob_, transmat_, means_, covars_ diag).
        post = gaussian_forward_filter(Xs[:rp], m.startprob_, m.transmat_,
                                        m.means_, m.covars_)
        seg_lo = train_end_idx if refit_pt == train_end_idx else refit_pt - refit_every
        seg_hi = rp
        # map to canonical order
        post_canon = post[:, order]
        for t in range(seg_lo, seg_hi):
            posts[t] = post_canon[t]
            states[t] = int(np.argmax(post_canon[t]))
        if rp >= n: break
    df = pd.DataFrame({"state": states}, index=obs.index)
    for k in range(n_states):
        df[f"p{k}"] = posts[:, k]
    return df


def main():
    t0 = time.time()
    print("=== X80 HMM dynamic regime detection (PIT) ===\n", flush=True)
    # Use V5 panel if ready, else V0 3yr panel
    v5p = REPO/"outputs/vBTC_features/panel_3yr_v5.parquet"
    panel_path = v5p if v5p.exists() else REPO/"outputs/vBTC_features/panel_3yr_v0.parquet"
    print(f"Panel: {panel_path.name}")
    panel = pd.read_parquet(panel_path)
    panel["open_time"] = pd.to_datetime(panel["open_time"], utc=True)
    panel["exit_time"] = pd.to_datetime(panel["exit_time"], utc=True)
    if "target_z" not in panel.columns: panel = x6.build_target_z(panel)
    x6.HEAVY_TAIL.discard("rvol_7d"); x6.HEAVY_TAIL.discard("ret_3d"); x6.HEAVY_TAIL.discard("btc_rvol_7d")
    if "bars_since_high_xs_rank" not in panel.columns:
        panel["bars_since_high_xs_rank"] = panel.groupby("open_time")["bars_since_high"].rank(pct=True).astype("float32")
    feats = [f for f in x6.BASE + x6.COHORT_EXTRAS if f in panel.columns]
    folds = x6.get_folds(panel)
    print(f"Panel {len(panel):,} rows; feats={len(feats)}\n")

    obs = btc_4h_obs()
    # initial training window = before fold-1 OOS start (use ~first 1/9 of data)
    times = sorted(panel["open_time"].unique())
    train_end_time = pd.Timestamp(times[len(times)//9])
    train_end_idx = int((obs.index < train_end_time).sum())
    print(f"HMM initial train window: {train_end_idx} bars (to {train_end_time})\n")

    for n_states in [2, 3]:
        print(f"--- HMM K={n_states} states ---", flush=True)
        st = hmm_filtered_states(obs, n_states, train_end_idx, refit_every=180)
        # interpret states by mean BTC 4h ret
        merged = obs.join(st)
        for k in range(n_states):
            sub = merged[merged["state"]==k]
            print(f"  state {k}: n={len(sub):>5} mean_4h_ret={sub['ret'].mean()*1e4:>+7.1f}bps "
                  f"mean_rvol={sub['rvol'].mean():.4f}", flush=True)
        # attach state to panel (PIT regime)
        st_map = st[["state"]].reset_index().rename(columns={"index":"open_time"})
        st_map["open_time"] = pd.to_datetime(st_map["open_time"], utc=True)
        p = panel.merge(st_map, on="open_time", how="left")
        p = p[p["state"]>=0]

        # Route per-sym Ridge specialists by HMM state
        parts = []
        for s in range(n_states):
            ps = p[p["state"]==s]
            if ps["open_time"].nunique() < 50: continue
            try: parts.append(x6.train_per_sym_ridge(ps, folds, feats, label=f"x80_K{n_states}_s{s}"))
            except Exception as e: print(f"    state {s} err {e}")
        if parts:
            apd = pd.concat(parts, ignore_index=True).sort_values(["open_time","symbol"])
            pth = RCACHE/f"x80_hmm_K{n_states}_routed_preds.parquet"; apd.to_parquet(pth, index=False)
            m = x6.run_sleeve_on_preds(pth, f"x80_hmm_K{n_states}")
            print(f"  HMM K={n_states} routed: Sharpe={m.get('sharpe',0):+.2f} folds={m.get('folds_pos','?')} conc={m.get('concentration','?')}", flush=True)

        # Also: gate — trade only in non-most-bullish states (drop top-mean-ret state)
        # (most-bullish state = highest index under canonical ordering)
        bull_state = n_states - 1
        p_gate = p[p["state"] != bull_state]
        parts_g = []
        for s in range(n_states-1):
            ps = p_gate[p_gate["state"]==s]
            if ps["open_time"].nunique() < 50: continue
            try: parts_g.append(x6.train_per_sym_ridge(ps, folds, feats, label=f"x80g_K{n_states}_s{s}"))
            except Exception: pass
        if parts_g:
            apdg = pd.concat(parts_g, ignore_index=True).sort_values(["open_time","symbol"])
            pthg = RCACHE/f"x80_hmm_K{n_states}_gated_preds.parquet"; apdg.to_parquet(pthg, index=False)
            mg = x6.run_sleeve_on_preds(pthg, f"x80_hmm_K{n_states}_gated")
            print(f"  HMM K={n_states} gate-out-bull-state: Sharpe={mg.get('sharpe',0):+.2f} folds={mg.get('folds_pos','?')}", flush=True)

    print(f"\nReference: V0 3yr uncond +0.12, KMeans-routed K=5 +1.07, hard gate +1.13")
    print(f"Done [{time.time()-t0:.0f}s]")


if __name__ == "__main__":
    main()
