"""Probe #2 — Is the convex outcome SYMBOL-SYSTEMATIC or TIME-RANDOM per name?

Probe #1 showed trade-level features can't separate winners from losers above
multi-test noise. Natural next question: maybe the separation is at the
SYMBOL level — some names are *reliably* big winners across many cycles,
others are reliably big losers — and the strategy's PnL comes from a stable
name-class structure rather than per-cycle skill or noise.

Decomposition:
  A. For each symbol with ≥10 entered longs, compute:
     - n_WIN  (extreme-winner cycles, top decile of contrib_bps)
     - n_LOSE (extreme-loser cycles, bot decile)
     - n_MID
     - win_share = n_WIN / total_entered
     - loss_share = n_LOSE / total_entered
     - sum_contrib = total bps contributed
  B. Plot the distribution: are symbols clustered into "systematic winners"
     (high win_share, low loss_share) and "systematic losers", OR is every
     symbol roughly symmetric (memo-like — large win cycles AND large loss
     cycles)?
  C. Characterize SYSTEMATIC-WINNER vs SYSTEMATIC-LOSER symbols on
     symbol-level properties (mean atr_pct, listing age proxy, mean
     idio_skew_1d, mean funding_rate, mean log_quote_volume_90d etc.)
  D. Ex-VVV view: of the non-VVV symbols, who are the top contributors and
     are they a "class" (memes / newer / low-float / etc.)?

Pure descriptive characterization. Output: ranked symbol list + group-level
feature signatures, which become the next probe hypothesis (if a symbol-class
signature exists PIT, test whether it predicts ex-ante which names will be
the systematic winners).
"""
from __future__ import annotations
import json, sys, time, warnings
from pathlib import Path
import numpy as np, pandas as pd

warnings.filterwarnings("ignore")
REPO = Path("/home/yuqing/ctaNew")
OUT = REPO / "research/mechanism_probes_2026-05-20"; OUT.mkdir(parents=True, exist_ok=True)
LEGS = REPO / "research/convexity_forensic_2026-05-19/entered_legs.parquet"
PANEL = REPO / "outputs/vBTC_features/panel_variants_with_funding.parquet"
KLINES = REPO / "data/ml/test/parquet/klines"


def listing_first_ts(sym):
    d = KLINES / sym / "5m"
    if not d.exists(): return None
    fs = sorted(d.glob("*.parquet"))
    if not fs: return None
    try: return pd.Timestamp(fs[0].stem, tz="UTC")
    except Exception: return None


def main():
    t0 = time.time()
    legs = pd.read_parquet(LEGS)
    legs["time"] = pd.to_datetime(legs["time"], utc=True)
    L = legs[legs["side"] == "long"].copy()
    p10, p90 = L["contrib_bps"].quantile([0.10, 0.90])
    L["grp"] = np.where(L["contrib_bps"] >= p90, "WIN",
                np.where(L["contrib_bps"] <= p10, "LOSE", "MID"))

    # A. per-symbol decomposition
    agg = L.groupby("symbol").agg(
        n_total=("contrib_bps", "size"),
        n_WIN=("grp", lambda x: (x == "WIN").sum()),
        n_LOSE=("grp", lambda x: (x == "LOSE").sum()),
        n_MID=("grp", lambda x: (x == "MID").sum()),
        sum_contrib_bps=("contrib_bps", "sum"),
        mean_contrib_bps=("contrib_bps", "mean"),
        n_first_entry=("time", "min"),
    ).reset_index()
    agg = agg[agg["n_total"] >= 10].copy()
    agg["win_share"] = agg["n_WIN"] / agg["n_total"]
    agg["loss_share"] = agg["n_LOSE"] / agg["n_total"]
    agg["win_minus_loss"] = agg["win_share"] - agg["loss_share"]
    agg = agg.sort_values("sum_contrib_bps", ascending=False)

    # B. classification: systematic winners (top-quartile win_minus_loss),
    # systematic losers (bottom-quartile)
    q75, q25 = agg["win_minus_loss"].quantile([0.75, 0.25])
    agg["class"] = np.where(agg["win_minus_loss"] >= q75, "SYS_WINNER",
                    np.where(agg["win_minus_loss"] <= q25, "SYS_LOSER", "MID"))
    nW = int((agg["class"] == "SYS_WINNER").sum())
    nL = int((agg["class"] == "SYS_LOSER").sum())

    # C. symbol-level feature signature: mean of selected PIT features over
    #    each symbol's entered cycles
    SYM_FEATS = ["atr_pct", "idio_vol_1d_vs_bk", "idio_vol_to_btc_1d",
                 "idio_skew_1d", "idio_kurt_1d", "idio_max_abs_12b",
                 "name_idio_share_1d", "name_factor_loading_1d",
                 "funding_rate", "funding_rate_z_7d",
                 "return_1d", "dom_level_vs_bk", "aggr_ratio_4h"]
    panel_cols = set(pd.read_parquet(PANEL, columns=None).head(1).columns)
    SYM_FEATS = [c for c in SYM_FEATS if c in panel_cols]
    pan = pd.read_parquet(PANEL, columns=["symbol", "open_time"] + SYM_FEATS)
    pan["open_time"] = pd.to_datetime(pan["open_time"], utc=True)
    D = L.rename(columns={"time": "open_time"}).merge(
        pan, on=["symbol", "open_time"], how="left")
    sym_sig = D.groupby("symbol")[SYM_FEATS].mean().reset_index()
    agg = agg.merge(sym_sig, on="symbol", how="left")

    # Listing age proxy
    listings = {s: listing_first_ts(s) for s in agg["symbol"]}
    panel_end = pd.Timestamp("2026-05-06", tz="UTC")
    agg["listing_days_at_panel_end"] = agg["symbol"].map(
        lambda s: (panel_end - listings[s]).days if listings.get(s) is not None else None)

    # D. group-level signatures (mean of features in SYS_WINNER vs SYS_LOSER)
    sig_w = agg[agg["class"] == "SYS_WINNER"][SYM_FEATS + ["listing_days_at_panel_end"]].mean()
    sig_l = agg[agg["class"] == "SYS_LOSER"][SYM_FEATS + ["listing_days_at_panel_end"]].mean()
    sig_diff = (sig_w - sig_l).abs().sort_values(ascending=False)

    # ex-VVV top contributors
    exv = agg[agg["symbol"] != "VVVUSDT"].sort_values("sum_contrib_bps", ascending=False).head(10)

    # save
    out = {
        "n_symbols_with_>=10_legs": int(len(agg)),
        "n_sys_winner": nW, "n_sys_loser": nL,
        "win_minus_loss_q25_q75": [round(float(q25), 3), round(float(q75), 3)],
        "top_8_contributors": agg[["symbol", "n_total", "n_WIN", "n_LOSE",
                                    "sum_contrib_bps", "win_share", "loss_share",
                                    "win_minus_loss", "class"]].head(8).round(3).to_dict("records"),
        "bottom_8_contributors": agg.sort_values("sum_contrib_bps").head(8)[
            ["symbol", "n_total", "n_WIN", "n_LOSE", "sum_contrib_bps",
             "win_share", "loss_share", "win_minus_loss", "class"]
        ].round(3).to_dict("records"),
        "exVVV_top10": exv[["symbol", "sum_contrib_bps", "win_share",
                             "loss_share", "class"]].round(3).to_dict("records"),
        "sys_winner_vs_loser_mean_feature_signature": {
            f: {"sys_winner_mean": round(float(sig_w[f]), 5),
                "sys_loser_mean": round(float(sig_l[f]), 5),
                "abs_diff": round(float(sig_diff[f]), 5)}
            for f in sig_diff.index},
        "elapsed_s": round(time.time() - t0, 1),
    }
    (OUT / "probe2_results.json").write_text(json.dumps(out, indent=2, default=str))

    # printout
    print(f"symbols with ≥10 long entries: {len(agg)}")
    print(f"\nTop 8 contributors (sum_contrib bps):")
    print(f"  {'symbol':<14} {'n':>4} {'#W':>3} {'#L':>3} {'sum':>10} "
          f"{'win%':>5} {'loss%':>5} {'W-L':>6}  class")
    for r in out["top_8_contributors"]:
        print(f"  {r['symbol']:<14} {int(r['n_total']):>4} {int(r['n_WIN']):>3} "
              f"{int(r['n_LOSE']):>3} {r['sum_contrib_bps']:>+10.0f} "
              f"{r['win_share']*100:>5.1f} {r['loss_share']*100:>5.1f} "
              f"{r['win_minus_loss']*100:>+6.1f}  {r['class']}")
    print(f"\nBottom 8 (largest negative contributors):")
    for r in out["bottom_8_contributors"]:
        print(f"  {r['symbol']:<14} {int(r['n_total']):>4} {int(r['n_WIN']):>3} "
              f"{int(r['n_LOSE']):>3} {r['sum_contrib_bps']:>+10.0f} "
              f"{r['win_share']*100:>5.1f} {r['loss_share']*100:>5.1f} "
              f"{r['win_minus_loss']*100:>+6.1f}  {r['class']}")
    print(f"\nex-VVV top-10 contributors:")
    for r in out["exVVV_top10"]:
        print(f"  {r['symbol']:<14} {r['sum_contrib_bps']:>+10.0f}  W%={r['win_share']*100:>5.1f}"
              f"  L%={r['loss_share']*100:>5.1f}  {r['class']}")
    print(f"\nSYS_WINNER vs SYS_LOSER mean feature signature (top diffs):")
    print(f"  {'feature':<28} {'winners':>12} {'losers':>12} {'diff':>10}")
    for f in sig_diff.index[:10]:
        print(f"  {f:<28} {sig_w[f]:>+12.5g} {sig_l[f]:>+12.5g} "
              f"{sig_w[f]-sig_l[f]:>+10.5g}")
    print("\nPROBE2_DONE", flush=True)


if __name__ == "__main__":
    main()
