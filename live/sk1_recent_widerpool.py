"""Committed recent-holdout + wider-pool evaluation (review F3 fix — these decisive negatives
were previously only in inline scripts / prose). Reproduces:
  (A) SK1 recent forward holdout: train crowding+pred squeeze classifier on ALL OOS, apply to
      the recent window; discrete skip vs matched-count placebo. -> recent Sharpe ~ negative,
      skip rate elevated (non-stationary crowding->squeeze map).
  (B) wider-pool select: pick 1L/2S-side shorts from ranks 1-N by crowding P(squeeze); OOS
      (nested) vs recent (forward holdout) vs naive top-K vs random-from-N.
Usage: python3 live/sk1_recent_widerpool.py
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
import warnings; warnings.filterwarnings("ignore")
REPO = Path("/home/yuqing/ctaNew"); D = REPO / "live/state/convexity"; rng = np.random.default_rng(15)
sys.path.insert(0, str(REPO)); import live.sq1_crowding_predictive as sq
FEATS = sq.FEATS; COLS = ["pred"] + FEATS + ["fx"]

def sharpe(x): return x.mean() / x.std(ddof=1) * np.sqrt(365) if x.std(ddof=1) > 0 else np.nan
def maxdd(x): eq = np.cumsum(x); return float((eq - np.maximum.accumulate(eq)).min())

def rows_for(book):
    bb = pd.read_parquet(D / f"{book}/v0full_hl60.parquet", columns=["symbol", "open_time", "pred", "fold"])
    bb["open_time"] = pd.to_datetime(bb["open_time"], utc=True)
    pan = pd.read_parquet(REPO / "outputs/vBTC_features/panel_expanded_v0.parquet",
                          columns=["symbol", "open_time", "alpha_vs_btc_realized"])
    pan["open_time"] = pd.to_datetime(pan["open_time"], utc=True)
    pan = pan[(pan.open_time.dt.hour % 4 == 0) & (pan.open_time.dt.minute == 0)].sort_values(["symbol", "open_time"])
    pan["fwd24"] = pan.groupby("symbol")["alpha_vs_btc_realized"].transform(
        lambda s: s.rolling(6).sum().shift(-5)) * 1e4
    bb = bb.merge(pan[["symbol", "open_time", "fwd24"]], on=["symbol", "open_time"], how="left")
    bb["r"] = bb.groupby("open_time")["pred"].rank(method="first")
    cr = pd.read_parquet(D / "crowding_panel.parquet"); cr["open_time"] = pd.to_datetime(cr["open_time"], utc=True)
    bb = bb.merge(cr, on=["symbol", "open_time"], how="left"); bb["fx"] = bb["funding_rate_z_7d"] * bb["funding_dispersion"]
    return bb

def train_on_oos():
    oos = sq.build_rows().reset_index(drop=True); oos["fx"] = oos["funding_rate_z_7d"] * oos["funding_dispersion"]
    Xthr = oos[oos.r <= 2]["fwd24"].quantile(0.90)
    tr = oos[oos.r <= 3].dropna(subset=["fwd24"])
    mu = np.nanmean(tr[COLS].to_numpy(), 0); sd = np.nanstd(tr[COLS].to_numpy(), 0); sd[sd == 0] = 1
    y = (tr["fwd24"] > Xthr).astype(int).to_numpy()
    m = LogisticRegression(C=1.0, max_iter=200).fit(np.nan_to_num((tr[COLS].to_numpy() - mu) / sd), y)
    thr90 = np.quantile(m.predict_proba(np.nan_to_num((tr[COLS].to_numpy() - mu) / sd))[:, 1], 0.90)
    return m, mu, sd, Xthr, thr90

def P(m, mu, sd, df): return m.predict_proba(np.nan_to_num((df[COLS].to_numpy() - mu) / sd))[:, 1]

def main():
    m, mu, sd, Xthr, thr90 = train_on_oos()
    # ---- (A) SK1 recent forward holdout ----
    rec = rows_for("hl_tgt_res_base_clean")
    sh = rec[rec.r <= 2].dropna(subset=["fwd24"]).copy()
    sh["P"] = P(m, mu, sd, sh); sh["sqz"] = (sh["fwd24"] > Xthr).astype(int); sh["skip"] = sh["P"] > thr90
    days = []; tev = 0; tp = []; E = []
    for t, g in sh.groupby("open_time"):
        if len(g) < 2: continue
        g = g.head(2); taken = ~g["skip"].to_numpy()
        tev += int(g["sqz"].to_numpy()[taken].sum())
        tp.append(((-g["fwd24"]).to_numpy()[taken] * 0.5).sum() - 9.0 * 0.5 * taken.sum())
        days.append(t.date()); E.append((int(g["skip"].sum()), (-g["fwd24"]).to_numpy(), g["sqz"].to_numpy()))
    tp = pd.Series(tp, index=days).groupby(level=0).sum()
    pe = []; ps = []
    for _ in range(300):
        ev = 0; pr = []
        for cnt, pnl, sqz in E:
            sk = np.array([False, False]) if cnt == 0 else (np.array([True, True]) if cnt == 2
                 else (lambda a: (a.__setitem__(rng.integers(0, 2), True), a)[1])(np.array([False, False])))
            taken = ~sk; ev += int(sqz[taken].sum()); pr.append((pnl[taken] * 0.5).sum() - 9.0 * 0.5 * taken.sum())
        pe.append(ev); ps.append(sharpe(pd.Series(pr, index=days).groupby(level=0).sum()))
    pe = np.array(pe); ps = np.array(ps)
    print(f"(A) SK1 RECENT forward holdout: skip rate {sh['skip'].mean():.2f} (OOS-calibrated ~0.17)")
    print(f"    net Sharpe treatment {sharpe(tp):+.2f} vs placebo mean {ps.mean():+.2f} "
          f"[p95 {np.percentile(ps,95):+.2f}] -> {'BEATS' if sharpe(tp)>np.percentile(ps,95) else 'WITHIN band'}")
    print(f"    squeeze events taken {tev} vs placebo mean {pe.mean():.0f} [p5 {np.percentile(pe,5):.0f}]")
    # ---- (B) wider-pool select, recent forward holdout ----
    sl = rec[rec.r <= 8].dropna(subset=["fwd24"]).copy(); sl["P"] = P(m, mu, sd, sl)
    sl["sqz"] = (sl["fwd24"] > Xthr).astype(int); sl["spnl"] = -sl["fwd24"]
    print(f"\n(B) wider-pool select, RECENT forward holdout — net short-side Sharpe (2 picks, 9bps/leg):")
    for name in ("naive(top-2)", "crowd-from-3", "crowd-from-5", "crowd-from-8", "random-from-8"):
        dd = []; pnl = []; ga = []
        for t, g in sl.groupby("open_time"):
            if name.startswith("naive"): pick = g[g.r <= 2]
            elif name.startswith("crowd"): N = int(name.split("-")[-1]); pick = g[g.r <= N].nsmallest(2, "P")
            else: gg = g[g.r <= 8]; pick = gg.sample(min(2, len(gg)), random_state=rng.integers(1e9))
            if len(pick) < 2: continue
            ga.append(pick["spnl"].mean()); pnl.append(pick["spnl"].mean() - 9.0 * 2); dd.append(t.date())
        s = pd.Series(pnl, index=dd).groupby(level=0).sum()
        print(f"    {name:16s} grossA/name {np.mean(ga):+6.1f}  net Sharpe {sharpe(s):+.2f}")
    print("SKRWDONE")

if __name__ == "__main__":
    main()
