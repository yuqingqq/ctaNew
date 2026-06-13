"""WHY does V3.1 capture the meme long-tails? (owner-requested 2026-05-19)

Owner: profit concentrates in long-tail memes even on a large universe.
Decompose the MECHANISM on the production book (production_sleeves.parquet
× all_predictions return_pct, + funding_rate_z_7d from the linear panel):

  A SELECTOR  : does rolling-IC top-15 preferentially load high-vol names?
  B SKILL vs VOL : per-leg hit-rate (directional skill) vs avg |move|
                   (vol amplification), split by realized-vol quartile.
  C SIDE+FUNDING : long vs short edge; does the short edge align with
                   extreme positive funding (crowded-long squeeze)?
  D VVV ANATOMY : per-fold VVV PnL, side, hit-rate, single-cycle lottery
                   check (is the 62% a few huge cycles?).

"meme/long-tail" = top realized-return-vol quartile (operational, not
hand-labelled). Diagnostic only; nothing adopted. Production unaffected.
"""
from __future__ import annotations
import importlib.util
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
REPO = Path("/home/yuqing/ctaNew")
OUTD = REPO / "linear_model/composite_study/results"
APD = REPO / "outputs/vBTC_audit_panel/all_predictions.parquet"
PS = REPO / "outputs/vBTC_sleeve_horizon/production_sleeves.parquet"
LINPAN = REPO / "outputs/vBTC_features_btc_only_111_full_pit/panel_btc_only_111.parquet"


def main():
    print("=" * 92, flush=True)
    print("  WHY V3.1 CAPTURES THE MEME LONG-TAILS — mechanism decomposition",
          flush=True)
    print("=" * 92, flush=True)
    ap = pd.read_parquet(APD)[["symbol", "open_time", "return_pct",
                               "alpha_A", "pred", "fold"]]
    ap["open_time"] = pd.to_datetime(ap["open_time"], utc=True)
    ps = pd.read_parquet(PS)
    ps["time"] = pd.to_datetime(ps["time"], utc=True)
    fund = pd.read_parquet(LINPAN, columns=["symbol", "open_time",
                                            "funding_rate_z_7d"])
    fund["open_time"] = pd.to_datetime(fund["open_time"], utc=True)
    retm = ap.set_index(["symbol", "open_time"])["return_pct"].to_dict()
    fzm = fund.set_index(["symbol", "open_time"])["funding_rate_z_7d"].to_dict()

    # realized-vol quartile per symbol (operational "long-tail" def)
    vol = ap.groupby("symbol")["return_pct"].std()
    q = pd.qcut(vol, 4, labels=["Q1_low", "Q2", "Q3", "Q4_high"])
    volq = q.to_dict()
    print(f"\n  realized-vol quartiles (std of 4h return): "
          f"Q1≤{vol.quantile(.25)*1e4:.0f}bps  "
          f"Q4≥{vol.quantile(.75)*1e4:.0f}bps  "
          f"max={vol.idxmax()} {vol.max()*1e4:.0f}bps", flush=True)

    # ---- per-leg table ----
    legs = []
    for _, r in ps[ps.traded].iterrows():
        t = r["time"]
        f = r["fold"]
        for side, names in (("L", r["long_basket"]), ("S", r["short_basket"])):
            k = max(len(names), 1)
            for s in names:
                rv = retm.get((s, t))
                if rv is None:
                    continue
                sgn = 1.0 if side == "L" else -1.0
                legs.append(dict(sym=s, t=t, fold=f, side=side,
                                 ret=rv, pnl=sgn * rv / k,
                                 win=int(sgn * rv > 0),
                                 absmove=abs(rv),
                                 fz=fzm.get((s, t), np.nan),
                                 vq=str(volq.get(s, "NA"))))
    d = pd.DataFrame(legs)
    print(f"  {len(d)} traded legs, {d.sym.nunique()} symbols", flush=True)

    # ---- A: selector — is the traded set higher-vol than the pool? ----
    pool_vol = vol.median() * 1e4
    traded_vol = vol.loc[d.sym.unique()].median() * 1e4
    print(f"\n── A SELECTOR ──", flush=True)
    print(f"  median vol: candidate pool={pool_vol:.0f}bps | "
          f"ever-traded symbols={traded_vol:.0f}bps", flush=True)
    vq_share = d.groupby("vq").size() / len(d) * 100
    print(f"  traded-leg share by vol quartile: "
          + " ".join(f"{k}={v:.0f}%" for k, v in vq_share.items())
          + "  (uniform=25% each ⇒ skew = selector loads vol)", flush=True)

    # ---- B: skill (hit-rate) vs vol amplification (|move|) ----
    print(f"\n── B SKILL vs VOL (by realized-vol quartile) ──", flush=True)
    print(f"  {'vq':<8} {'legs':>6} {'hit%':>6} {'avg|mv|bps':>11} "
          f"{'sumPnL':>9}", flush=True)
    for vq, g in d.groupby("vq"):
        print(f"  {vq:<8} {len(g):>6} {g.win.mean()*100:>5.1f} "
              f"{g.absmove.mean()*1e4:>11.0f} {g.pnl.sum()*1e4:>+9.0f}",
              flush=True)
    print(f"  → if Q4_high has ~same hit% as others but dominates sumPnL,"
          f" the edge is VOL-AMPLIFIED (not extra skill on memes)",
          flush=True)

    # ---- C: long vs short, funding alignment ----
    print(f"\n── C SIDE + FUNDING ──", flush=True)
    for side, g in d.groupby("side"):
        sl = side == "S"
        print(f"  {('SHORT' if sl else 'LONG '):5}: legs={len(g)} "
              f"hit%={g.win.mean()*100:.1f} sumPnL={g.pnl.sum()*1e4:+.0f} "
              f"mean funding_z={g.fz.mean():+.2f} "
              f"(win funding_z={g[g.win==1].fz.mean():+.2f} / "
              f"loss={g[g.win==0].fz.mean():+.2f})", flush=True)
    print(f"  → DDI prior: short side carries alpha; crowded-long "
          f"(high +funding_z) → squeeze-down ⇒ short wins", flush=True)

    # ---- D: VVV anatomy ----
    print(f"\n── D VVVUSDT ANATOMY (the 62%-of-net name) ──", flush=True)
    v = d[d.sym == "VVVUSDT"]
    if len(v):
        print(f"  {len(v)} legs | hit%={v.win.mean()*100:.1f} | "
              f"sumPnL={v.pnl.sum()*1e4:+.0f}bps | side mix "
              f"L={ (v.side=='L').sum() }/S={ (v.side=='S').sum() } | "
              f"mean funding_z={v.fz.mean():+.2f}", flush=True)
        pf = v.groupby("fold")["pnl"].sum() * 1e4
        print(f"  per-fold PnL: "
              + " ".join(f"f{int(k)}={x:+.0f}" for k, x in pf.items()),
              flush=True)
        top = v.reindex(v.pnl.abs().sort_values(ascending=False).index).head(5)
        sh5 = top.pnl.sum() / v.pnl.sum() * 100 if v.pnl.sum() else np.nan
        print(f"  top-5 single legs = {sh5:.0f}% of VVV PnL "
              f"(lottery check; legs: "
              + " ".join(f"{r.side}{r.ret*1e4:+.0f}" for _, r in top.iterrows())
              + ")", flush=True)

    # ---- synthesis ----
    q4 = d[d.vq == "Q4_high"]
    q4_share = q4.pnl.sum() / d.pnl.sum() * 100 if d.pnl.sum() else np.nan
    hit_gap = q4.win.mean() - d[d.vq != "Q4_high"].win.mean()
    mv_ratio = q4.absmove.mean() / max(d[d.vq != "Q4_high"].absmove.mean(),
                                       1e-9)
    print(f"\n  SYNTHESIS: Q4_high (long-tail) = {q4_share:.0f}% of net "
          f"PnL; its hit-rate edge vs rest = {hit_gap*100:+.1f}pp; its "
          f"|move| is {mv_ratio:.1f}× larger. ⇒ "
          + ("VOL-AMPLIFIED THIN-SKILL bet (hit-rate ≈ rest, PnL via "
             "huge moves) — concentration is mechanical (selector loads "
             "vol; K=3 directional amplifies it), not broad alpha."
             if abs(hit_gap) < 0.03 else
             "GENUINE extra directional skill on long-tail names."),
          flush=True)
    d.to_csv(OUTD / "meme_mechanism_legs.csv", index=False)
    print(f"\nSaved {OUTD}/meme_mechanism_legs.csv", flush=True)


if __name__ == "__main__":
    main()
