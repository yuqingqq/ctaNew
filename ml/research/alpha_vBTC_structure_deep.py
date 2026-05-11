"""Deep-dive analysis of alpha-vs-BTC structure (existing 39-symbol pool).

Three analyses:
  1. PCA factor interpretation — which symbols load strongest on each PC?
     Tells us what the dominant factors *are* semantically (e.g., is PC1
     "memes-vs-majors" or "meme cluster" or "L1-rotation"?)
  2. Data-driven clustering — hierarchical clustering on alpha correlation.
     Identifies groups of names that move together vs BTC, independent of
     human-defined sectors.
  3. Alpha decay — how does cumulative alpha capture change with holding
     horizon? Helps choose optimal h.

Output informs Phase 3 feature engineering directly.
"""
from __future__ import annotations
import sys, warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from features_ml.cross_sectional import list_universe, build_kline_features

OUT_DIR = REPO / "outputs/vBTC_structure_deep"
OUT_DIR.mkdir(parents=True, exist_ok=True)
BTC_SYMBOL = "BTCUSDT"
BETA_WINDOW = 288


def main():
    print("Loading kline data...")
    universe = sorted(list_universe(min_days=200))
    feats = {s: build_kline_features(s) for s in universe}
    feats = {s: f for s, f in feats.items() if not f.empty}
    btc_close = feats[BTC_SYMBOL]["close"].copy()
    btc_close.index = pd.to_datetime(btc_close.index, utc=True)
    btc_ret = btc_close.pct_change()

    # Compute β-adjusted alpha-vs-BTC at h=48 for each symbol, sample every h
    print("\nComputing alpha-vs-BTC at h=48 ...")
    alpha_matrix = {}    # {symbol: pd.Series of alpha at non-overlapping cycles}
    beta_matrix = {}     # {symbol: pd.Series of beta at cycle starts}
    for s, f in feats.items():
        if s == BTC_SYMBOL: continue
        my_close = f["close"].copy()
        my_close.index = pd.to_datetime(my_close.index, utc=True)
        my_ret = my_close.pct_change()
        joined = pd.DataFrame({"my_ret": my_ret, "btc_ret": btc_ret.reindex(my_close.index, method="ffill"),
                                 "my_close": my_close,
                                 "btc_close": btc_close.reindex(my_close.index, method="ffill")}).dropna()
        if len(joined) < BETA_WINDOW + 100: continue
        cov = (joined["my_ret"] * joined["btc_ret"]).rolling(BETA_WINDOW).mean() - \
              joined["my_ret"].rolling(BETA_WINDOW).mean() * joined["btc_ret"].rolling(BETA_WINDOW).mean()
        var = joined["btc_ret"].rolling(BETA_WINDOW).var().replace(0, np.nan)
        beta = (cov / var).clip(-5, 5).shift(1)
        my_fwd = joined["my_close"].pct_change(48).shift(-48)
        btc_fwd = joined["btc_close"].pct_change(48).shift(-48)
        alpha = my_fwd - beta * btc_fwd
        # Sample non-overlapping cycles
        alpha_sub = alpha.iloc[::48].dropna()
        beta_sub = beta.iloc[::48].dropna()
        alpha_matrix[s] = alpha_sub
        beta_matrix[s] = beta_sub

    # Align into a wide DataFrame
    alpha_df = pd.DataFrame(alpha_matrix)
    alpha_df = alpha_df.dropna(thresh=int(0.7 * len(alpha_df.columns)))
    print(f"  alpha matrix: {alpha_df.shape}  (cycles × symbols)")

    # ===== 1. PCA factor interpretation =====
    print("\n" + "=" * 90)
    print("1. PCA FACTOR INTERPRETATION — top loadings per PC")
    print("=" * 90)
    aligned = alpha_df.fillna(0)
    centered = aligned - aligned.mean()
    std = centered.std()
    std[std == 0] = 1
    normed = centered / std
    cov_matrix = normed.cov().fillna(0)
    eigvals, eigvecs = np.linalg.eigh(cov_matrix.values)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    sym_order = list(cov_matrix.columns)
    total_var = eigvals.sum()

    # Show top 5 PCs with their loadings
    for k in range(5):
        if eigvals[k] <= 0: break
        loadings = pd.Series(eigvecs[:, k], index=sym_order).sort_values()
        var_pct = 100 * eigvals[k] / total_var
        print(f"\n  PC{k+1} ({var_pct:.1f}% variance)")
        # Top 5 negative
        print(f"    most negative loadings:")
        for sym, ld in loadings.head(5).items():
            print(f"      {sym:<14}  {ld:>+.3f}")
        # Top 5 positive
        print(f"    most positive loadings:")
        for sym, ld in loadings.tail(5).iloc[::-1].items():
            print(f"      {sym:<14}  {ld:>+.3f}")

    # ===== 2. Hierarchical clustering on correlation =====
    print("\n" + "=" * 90)
    print("2. DATA-DRIVEN CLUSTERS (hierarchical clustering on alpha correlation)")
    print("=" * 90)
    try:
        from scipy.cluster.hierarchy import linkage, fcluster
        from scipy.spatial.distance import squareform
        corr = aligned.corr().fillna(0)
        # Distance = 1 - corr (clamped to non-negative)
        dist_arr = (1 - corr).clip(lower=0).to_numpy().copy()
        np.fill_diagonal(dist_arr, 0)
        condensed = squareform(dist_arr, checks=False)
        Z = linkage(condensed, method="average")
        # Try a few cluster counts
        for k_clusters in [4, 6, 8]:
            print(f"\n  K={k_clusters} clusters:")
            labels = fcluster(Z, t=k_clusters, criterion="maxclust")
            for c in range(1, k_clusters + 1):
                members = [sym_order[i] for i, l in enumerate(labels) if l == c]
                if members:
                    print(f"    Cluster {c} ({len(members):>2}): {sorted(members)}")
    except ImportError:
        print("  scipy not available — skipping clustering")

    # ===== 3. Alpha decay across holding horizons =====
    print("\n" + "=" * 90)
    print("3. ALPHA DECAY — cumulative |α| across holding horizons")
    print("=" * 90)
    print("  How fast does the alpha decay as we hold longer? Picks optimal h.")
    print()
    # For each holding horizon h, compute mean realized |α| per cycle
    horizons = [12, 24, 48, 96, 144, 288, 576, 864]
    print(f"  {'h (5-min bars)':<16} {'h (hours)':<11} {'mean |α|/cycle (bps)':>22} "
          f"{'cycles':>8} {'ann_factor':>12}")
    decay_rows = []
    for h in horizons:
        all_alphas = []
        for s, f in feats.items():
            if s == BTC_SYMBOL: continue
            my_close = f["close"].copy()
            my_close.index = pd.to_datetime(my_close.index, utc=True)
            my_ret = my_close.pct_change()
            j = pd.DataFrame({"my_ret": my_ret, "btc_ret": btc_ret.reindex(my_close.index, method="ffill"),
                                 "my_close": my_close,
                                 "btc_close": btc_close.reindex(my_close.index, method="ffill")}).dropna()
            if len(j) < BETA_WINDOW + h + 50: continue
            cov_h = (j["my_ret"] * j["btc_ret"]).rolling(BETA_WINDOW).mean() - \
                    j["my_ret"].rolling(BETA_WINDOW).mean() * j["btc_ret"].rolling(BETA_WINDOW).mean()
            var_h = j["btc_ret"].rolling(BETA_WINDOW).var().replace(0, np.nan)
            beta_h = (cov_h / var_h).clip(-5, 5).shift(1)
            my_fwd_h = j["my_close"].pct_change(h).shift(-h)
            btc_fwd_h = j["btc_close"].pct_change(h).shift(-h)
            alpha_h = (my_fwd_h - beta_h * btc_fwd_h).iloc[::h].dropna()
            all_alphas.append(alpha_h.abs().values)
        if not all_alphas: continue
        flat = np.concatenate(all_alphas)
        cycles_per_year = (288 * 365) / h
        ann_factor = np.sqrt(cycles_per_year)
        mean_abs_bps = flat.mean() * 1e4
        decay_rows.append({"h_bars": h, "h_hours": h * 5 / 60,
                            "mean_abs_alpha_bps": mean_abs_bps,
                            "cycles": len(flat), "ann_factor": ann_factor})
        print(f"  {h:<16} {h*5/60:<11.1f} {mean_abs_bps:>+22.2f} "
              f"{len(flat):>8} {ann_factor:>12.1f}")

    # Per-cycle alpha vs annualization rate
    print("\n  Interpretation:")
    print("    If mean |α|/h is roughly proportional to √h, alpha is white-noise (no persistence).")
    print("    If mean |α|/h scales sub-√h, alpha decays (mean-reversion within h).")
    print("    If mean |α|/h scales super-√h, alpha persists (momentum).")

    # Save outputs
    pd.DataFrame(decay_rows).to_csv(OUT_DIR / "alpha_decay.csv", index=False)
    print(f"\n  saved → {OUT_DIR}")


if __name__ == "__main__":
    main()
