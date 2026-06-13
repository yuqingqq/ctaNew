"""Step 104 — DATA WALKTHROUGH (transparency, not a test). Show, with real
numbers for SOLUSDT: (A) raw 5m kline, (B) raw OI panel row, (C) raw
aggTrades, (D) a full panel row, (E) how s_t is built from raw klines,
(F) what the target alpha_beta is, (G) how a bet is placed & scored.
"""
from __future__ import annotations
import importlib.util, sys, glob, warnings
from pathlib import Path
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 40)
REPO = Path("/home/yuqing/ctaNew")
sys.path.insert(0, str(REPO))


def _imp(n, r):
    s = importlib.util.spec_from_file_location(n, REPO / r)
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m


s94 = _imp("s94", "linear_model/scripts/94_info_ceiling_d1.py")
SYM = "SOLUSDT"
L, BLK = 288, 48                                   # 24h trailing, 4h grid


def main():
    print("=" * 100, flush=True)
    print(f"  STEP 104 — DATA WALKTHROUGH ({SYM}); real numbers, raw → "
          "signal → bet → outcome", flush=True)
    print("=" * 100, flush=True)

    # (A) raw 5m kline
    kf = sorted(glob.glob(str(REPO / f"data/ml/test/parquet/klines/{SYM}/5m/*.parquet")))
    k = pd.read_parquet(kf[len(kf)//2])
    print("\n(A) RAW 5m KLINE (one file, first 4 rows) — the base data:", flush=True)
    print(k.head(4).to_string(index=False), flush=True)

    # (B) raw OI panel
    oi = pd.read_parquet(REPO / "outputs/vBTC_features_oi/oi_panel.parquet")
    oi = oi[oi.symbol == SYM].sort_values("open_time")
    print("\n(B) RAW OI panel rows (PIT positioning feats, 2 rows):", flush=True)
    print(oi.iloc[5000:5002].to_string(index=False), flush=True)

    # (C) raw aggTrades
    af = sorted(glob.glob(str(REPO / f"data/ml/test/parquet/aggTrades/{SYM}/*.parquet")))
    a = pd.read_parquet(af[len(af)//2]).head(4)
    print(f"\n(C) RAW aggTrades ({Path(af[len(af)//2]).name}, 4 rows) "
          "(is_buyer_maker=True ⇒ taker SELL):", flush=True)
    print(a.to_string(index=False), flush=True)

    # build the model frame
    dec, syms, btc, pan = s94.build(universe_oi=False)
    d = dec[dec.symbol == SYM].sort_values("open_time").reset_index(drop=True)

    # (D) one full panel row
    raw = pan[pan.symbol == SYM].sort_values("open_time").reset_index(drop=True)
    row = raw.iloc[len(raw)//2]
    print(f"\n(D) ONE FULL PANEL ROW for {SYM} @ {row['open_time']} "
          "(27 cols incl. PIT features + target):", flush=True)
    print(row.to_string(), flush=True)

    # (E)+(F)+(G): reconstruct s_t from raw, show target, show the bet
    c = s94.load_close(SYM).set_index("open_time")["close"]
    bc = btc                                          # BTC close series
    retA = c.pct_change(L).shift(1)                    # trailing 24h, PIT
    retB = bc.pct_change(L).shift(1)
    fwdA = c.shift(-BLK) / c - 1.0                     # forward 4h (the horizon)
    fwdB = bc.shift(-BLK) / bc - 1.0
    e = d.set_index("open_time")
    idx = e.index[np.linspace(50, len(e)-50, 10).astype(int)]
    print("\n(E)+(F)+(G) — for 10 example decision points:\n"
          "  s_t = trailing_24h_ret(SOL) − beta_pit·trailing_24h_ret(BTC)   "
          "[from PAST only]\n"
          "  alpha_beta = fwd_4h_ret(SOL) − beta_pit·fwd_4h_ret(BTC)        "
          "[the FUTURE target]\n"
          "  bet pos = −sign(s_t)  (fade/convergence)   payoff = pos·alpha_beta",
          flush=True)
    hdr = (f"\n  {'time':16s} {'retA_24h':>9s} {'retB_24h':>9s} {'beta':>6s} "
           f"{'s_t':>9s} {'pos':>4s} | {'fwdA_4h':>8s} {'fwdB_4h':>8s} "
           f"{'alpha_beta':>10s} | {'payoff_bps':>10s} {'res':>4s}")
    print(hdr, flush=True)
    nW = 0
    for t in idx:
        r = e.loc[t]
        st = float(r["s_t"]); be = float(r["beta_btc_pit"])
        ab = float(r["alpha_beta"])
        rA = float(retA.get(t, np.nan)); rB = float(retB.get(t, np.nan))
        fA = float(fwdA.get(t, np.nan)); fB = float(fwdB.get(t, np.nan))
        pos = -np.sign(st) if st != 0 else 1.0
        pay = pos * ab * 1e4
        win = pay > 0; nW += win
        print(f"  {str(t)[:16]} {rA:+9.4f} {rB:+9.4f} {be:6.2f} {st:+9.4f} "
              f"{pos:+4.0f} | {fA:+8.4f} {fB:+8.4f} {ab:+10.5f} | "
              f"{pay:+10.1f} {'WIN' if win else 'LOSS':>4s}", flush=True)

    print(f"\n  → s_t reconstructed from PAST klines ≈ stored s_t? "
          f"(check first row): retA−beta·retB = "
          f"{float(retA.get(idx[0],np.nan)) - float(e.loc[idx[0],'beta_btc_pit'])*float(retB.get(idx[0],np.nan)):+.4f}"
          f"  vs stored s_t = {float(e.loc[idx[0],'s_t']):+.4f}", flush=True)
    print(f"  → these 10: {nW}/10 wins (full-sample hit ≈50%, the "
          "coin-flip we measured)", flush=True)

    print("""
  READING IT:
   • retA_24h / retB_24h = how SOL / BTC moved over the PAST 24h (inputs).
   • beta = SOL's PIT sensitivity to BTC. s_t strips BTC out → SOL's own
     (idiosyncratic) recent move. POSITIVE s_t = SOL outran its BTC-hedge
     recently; the convergence bet says it reverts → pos = −sign(s_t) = −1
     (SHORT the residual). Negative s_t → +1 (LONG).
   • alpha_beta = the SAME residual but measured over the NEXT 4h (the
     scoreboard). We never see it at decision time.
   • payoff = pos · alpha_beta. WIN if our faded direction matched the
     realized next-4h residual. Across all rows that is ~50/50 — the data
     does not reliably predict its own next-4h residual sign.
""", flush=True)


if __name__ == "__main__":
    main()
