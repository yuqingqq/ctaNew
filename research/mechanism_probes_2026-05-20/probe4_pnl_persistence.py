"""Probe #4 — Symbol-level PnL persistence: does the strategy's OWN past
output predict its near-future output per name?

Probes 1-3 closed direction-prediction from features at every level. The
ex-VVV-still-positive rotation observation (VVV→AXS→PENDLE) could be either:
  (a) FORECASTABLE in real time — "the name that has recently produced winning
      cycles will continue to," a tradeable rotation signal that doesn't
      need feature-based direction; or
  (b) ONLY RETROSPECTIVE — each "next winner" only identifiable after it
      already pumped (the realized rotation looks like a pattern only with
      hindsight).

This probe tests (a) directly. PIT.
  - For each (symbol, time t) where the strategy made at least N=3 past
    entries: trailing_7d_signed_contribution(symbol,t) = mean contrib_bps
    of any entered legs (long or short) in the window (t-7d, t-1cycle),
    strictly prior to t. PIT.
  - Label: realized contribution of the symbol's NEXT entered leg after t
    (sign and magnitude).
  - OOS-symbol (5 groups, seed 20260519): does the sign / magnitude of past
    7d contribution predict the next entered leg's sign / magnitude?
  - Placebo: shuffle the label.

If trailing-PnL OOS-symbol directional acc > 0.54 (clearly above the 0.515
lifecycle ceiling) → genuine new rotation signal — tradeable as "enter the
name with the highest recent realized contribution."
If ≈ 0.50 → rotation is purely retrospective; ex-VVV-positive is
sample-path luck, not a forecastable mechanism.

Magnitude/sizing angle also reported: does past contrib predict |next contrib|
(useful for sizing even if direction is unforecastable).
"""
from __future__ import annotations
import json, sys, time, warnings
from pathlib import Path
import numpy as np, pandas as pd

warnings.filterwarnings("ignore")
REPO = Path("/home/yuqing/ctaNew")
OUT = REPO / "research/mechanism_probes_2026-05-20"; OUT.mkdir(parents=True, exist_ok=True)
LEGS = REPO / "research/convexity_forensic_2026-05-19/entered_legs.parquet"
SEED = 20260519
WIN = pd.Timedelta(days=7)


def main():
    t0 = time.time()
    L = pd.read_parquet(LEGS)
    L["time"] = pd.to_datetime(L["time"], utc=True)
    L = L.sort_values(["symbol", "time"]).reset_index(drop=True)

    # PIT trailing-7d mean signed contribution per symbol, strictly < t
    rows = []
    for sym, g in L.groupby("symbol", sort=False):
        c = g["contrib_bps"].to_numpy()
        t = g["time"].to_numpy()
        # for each leg i, mean of legs j with t[j] in (t[i]-7d, t[i])
        for i in range(len(g)):
            mask = (t < t[i]) & (t >= (t[i] - WIN.to_numpy()))
            past = c[mask]
            if len(past) >= 3:
                rows.append({"symbol": sym, "time": pd.Timestamp(t[i]),
                             "trail_signed_mean": float(past.mean()),
                             "trail_abs_mean": float(np.abs(past).mean()),
                             "n_past": int(len(past)),
                             "next_contrib": float(c[i])})
    D = pd.DataFrame(rows)
    print(f"n_legs_with_>=3_past_7d: {len(D):,} | symbols={D['symbol'].nunique()}",
          flush=True)
    if len(D) < 200:
        print("INSUFFICIENT", flush=True)
        (OUT / "probe4_results.json").write_text(json.dumps({"status": "insufficient"}, indent=2))
        return

    # OOS-symbol classifier: train sign of (trail_signed_mean → next_contrib sign)
    syms = sorted(D["symbol"].unique())
    rng = np.random.RandomState(SEED); shf = syms.copy(); rng.shuffle(shf)
    gmap = {s: i % 5 for i, s in enumerate(shf)}
    D["g"] = D["symbol"].map(gmap)
    D["next_sign"] = np.sign(D["next_contrib"])

    def dir_acc(D_, feat, lab):
        accs = []
        for g in range(5):
            tr = D_[(D_["g"] != g)]
            te = D_[(D_["g"] == g)]
            if len(tr) < 100 or len(te) < 30: continue
            rel = np.sign(np.corrcoef(tr[feat].rank(), tr[lab])[0, 1])
            if rel == 0: rel = 1.0
            pred = np.sign(rel * (te[feat] - te[feat].median()))
            actual = np.sign(te[lab].to_numpy())
            m = (pred != 0) & (actual != 0)
            if m.sum() < 30: continue
            accs.append(float((pred[m] == actual[m]).mean()))
        return float(np.mean(accs)) if accs else np.nan, accs

    a_full, gs_full = dir_acc(D, "trail_signed_mean", "next_sign")
    D2 = D.copy()
    D2["next_sign_sh"] = D["next_sign"].sample(frac=1, random_state=42).to_numpy()
    a_plac, _ = dir_acc(D2, "trail_signed_mean", "next_sign_sh")

    # magnitude: does past |contrib| predict next |contrib|?
    D["next_abs"] = D["next_contrib"].abs()
    mag_corr = float(D[["trail_abs_mean", "next_abs"]].corr().iloc[0, 1])
    # Q5-Q1 spread by past trail_signed_mean: do the top-trailing-PnL names
    # produce systematically higher next-contrib?
    D["sk_q"] = pd.qcut(D["trail_signed_mean"], 5, labels=False, duplicates="drop")
    by_q = D.groupby("sk_q")["next_contrib"].mean()
    by_q_n = D.groupby("sk_q")["next_contrib"].size()

    out = {
        "n_legs_with_history": len(D),
        "symbols": int(D["symbol"].nunique()),
        "OOS_symbol_dir_acc_pnl_persistence": round(a_full, 4),
        "placebo_acc_shuffled":               round(a_plac, 4),
        "per_group_acc":                      [round(x, 3) for x in gs_full],
        "trail_abs_vs_next_abs_corr":         round(mag_corr, 4),
        "next_contrib_mean_by_trailing_quintile_bps": {
            int(k): round(float(v), 2) for k, v in by_q.items()},
        "next_contrib_n_per_quintile": {int(k): int(v) for k, v in by_q_n.items()},
        "lifecycle_probe_r24_ceiling_for_reference": 0.515,
        "interpretation": {
            "if_dir_acc >> 0.515 vs placebo ~0.50":
                "rotation signal real and tradeable; ex-VVV-positive is forecastable",
            "if_dir_acc ~ 0.50":
                "rotation is purely retrospective; ex-VVV-positive is sample-path luck",
            "mag_corr": "if >>0; past size predicts future size (useful for sizing)"},
        "elapsed_s": round(time.time() - t0, 1),
    }
    (OUT / "probe4_results.json").write_text(json.dumps(out, indent=2, default=str))
    print(json.dumps(out, indent=2, default=str), flush=True)
    print("PROBE4_DONE", flush=True)


if __name__ == "__main__":
    main()
