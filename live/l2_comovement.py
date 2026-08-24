"""Does alt order-book move TOGETHER with BTC's book? (user's mechanism for the null). If imbalance is a COMMON factor
co-moving with BTC, it predicts market/raw direction but washes out in the beta-neutral cross-section = why OB adds no
alpha. Measures, per era, for imbalance (imb1), sustained-imbalance (imb_ewma), and liquidity (z-scored liq1):
  - corr(alt, BTC) distribution
  - mean pairwise alt-alt corr
  - PCA: %variance in PC1 (the common factor) + corr(PC1, BTC)
  - regression R²(alt ~ BTC) = fraction of an alt's book that is common
High co-movement => imbalance is mostly beta (market-wide), the idiosyncratic part is small & noisy => null cross-sectionally.
"""
import glob
from pathlib import Path
import numpy as np, pandas as pd
import warnings; warnings.filterwarnings("ignore")
from live.bookdepth_persist import persist_feats

def build(col, lo, hi, zscore=False):
    cols = {}
    for f in glob.glob("/home/yuqing/ctaNew/data/ml/cache/l2_*.parquet"):
        sym = Path(f).stem[3:]; d = pd.read_parquet(f); d.index = pd.to_datetime(d.index, utc=True)
        if col == "imb_ewma":
            s = persist_feats(d["l2_imb1"].sort_index())["imb_ewma"]
        elif col in d.columns:
            s = d[col].sort_index()
        else:
            continue
        s = s[(s.index >= lo) & (s.index < hi)]
        if len(s) > 150: cols[sym] = s
    P = pd.DataFrame(cols).sort_index()
    if zscore: P = (P - P.mean()) / P.std()
    return P

def analyze(P, label):
    if "BTCUSDT" not in P.columns or P.shape[1] < 10:
        print(f"  {label}: insufficient (syms={P.shape[1]})"); return
    btc = P["BTCUSDT"]; alts = [c for c in P.columns if c != "BTCUSDT"]
    cs = pd.Series({c: P[c].corr(btc) for c in alts}).dropna()
    r2 = pd.Series({c: P[c].corr(btc) ** 2 for c in alts}).dropna()   # OLS R² = corr² for single regressor
    A = P[alts].loc[:, P[alts].notna().mean() > 0.6]
    cm = A.corr(); pair = cm.values[np.triu_indices_from(cm.values, 1)]
    Az = ((A - A.mean()) / A.std()).dropna(how="all")
    Az = Az.loc[:, Az.notna().mean() > 0.8].dropna()
    pc1 = np.nan; pc1_btc = np.nan
    if Az.shape[0] > 50 and Az.shape[1] > 5:
        U, S, Vt = np.linalg.svd(Az.values - Az.values.mean(0), full_matrices=False)
        pc1 = (S[0] ** 2) / (S ** 2).sum()
        scores = U[:, 0] * S[0]
        b = btc.reindex(Az.index)
        pc1_btc = abs(pd.Series(scores, index=Az.index).corr(b))
    print(f"  {label:16s}| corr(alt,BTC): median {cs.median():+.2f} [p25 {cs.quantile(.25):+.2f}, p75 {cs.quantile(.75):+.2f}] "
          f"| %>0.3: {(cs > 0.3).mean()*100:.0f}% | mean pairwise {np.nanmean(pair):+.2f} "
          f"| PC1 var {pc1*100:.0f}% (corr w/ BTC {pc1_btc:.2f}) | median R²(alt~BTC) {r2.median()*100:.0f}%")

def main():
    for era, lo, hi in [("RECENT", "2025-10-01", "2026-07-15"), ("OOS", "2023-01-01", "2025-09-01")]:  # full both-era (post re-fetch)
        print(f"### {era} ###")
        analyze(build("l2_imb1", lo, hi), "imbalance (imb1)")
        analyze(build("imb_ewma", lo, hi), "sustained (ewma)")
        analyze(build("l2_liq1", lo, hi, zscore=True), "liquidity-z (liq1)")
        print()
    print("read: HIGH corr/PC1/R² => alt books co-move w/ BTC = common factor => predicts raw direction but washes out")
    print("      in the beta-neutral book => explains the OB-alpha null. COMOVEDONE")

if __name__ == "__main__":
    main()
