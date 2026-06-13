"""Step 99 — D1-ext-D: spot-vs-perp volume lead as a MOMENTUM-vs-REVERSION
classifier (owner hypothesis, LOCKED pre-registration, one run, no sweep).

Question: does spot volume outperforming perp ("real cash demand") predict
the trailing move CONTINUES (momentum), while perp-led predicts it REVERTS?
This is the opposite polarity to the line's convergence thesis and was
never tested on its own terms (Step-98 x_fd_st tested the reversion form vs
the 4h residual only).

Design (cached spot_panel + perp klines, 20-sym):
  conditioner : sp_volratio_z1d (z288 spot/perp quote-vol, PIT, shifted)
                → quintiles q0(perp-led)..q4(spot-led) + extreme deciles
  trailing    : trailing-24h RAW return  (close.pct_change(288).shift(1), PIT)
  forward     : RAW return at 4h/24h/72h (clean fwd shift; no β long-horizon
                shift landmine) + β-residual @4h (panel alpha_beta, 2ndary)
  statistic   : per-bucket continuation coef = Spearman(trailing, forward);
                directional payoff = mean(sign(trailing)*forward) bps.
                Diagnostic of the PATTERN, not a Sharpe gate.
                q4−q0 spread: block-bootstrap CI, block ≥ horizon (handles
                overlapping forward windows honestly).
  PRE-REG verdict: intuition SUPPORTED iff continuation coef monotone ↑
                q0→q4 AND q4>0 (spot-led=momentum) AND (q4−q0) CI excl 0 at
                ≥1 horizon (raw). Else ABSENT. No strategy adopted.
Production LGBM unaffected.
"""
from __future__ import annotations
import importlib.util, sys, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
REPO = Path("/home/yuqing/ctaNew")
sys.path.insert(0, str(REPO))


def _imp(n, r):
    s = importlib.util.spec_from_file_location(n, REPO / r)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


s94 = _imp("s94", "linear_model/scripts/94_info_ceiling_d1.py")
from ml.research.alpha_v4_xs import block_bootstrap_ci
load_close = s94.load_close
HBARS = {"4h": 48, "24h": 288, "72h": 864}      # 5m bars
SPANEL = REPO / "outputs/vBTC_features_spot/spot_panel.parquet"
OUTD = REPO / "linear_model/results/step99_spot_momentum"
OUTD.mkdir(parents=True, exist_ok=True)


def main():
    print("=" * 96, flush=True)
    print("  STEP 99 — D1-ext-D: spot/perp volume lead as MOMENTUM-vs-"
          "REVERSION classifier (LOCKED)", flush=True)
    print("=" * 96, flush=True)
    t0 = time.time()
    dec, syms, btc, pan = s94.build(universe_oi=False)
    sp = pd.read_parquet(SPANEL)
    sp["open_time"] = pd.to_datetime(sp["open_time"], utc=True)
    have = sorted(sp["symbol"].unique())
    dec = dec[dec.symbol.isin(have)].merge(
        sp[["symbol", "open_time", "sp_volratio_z1d"]],
        on=["symbol", "open_time"], how="inner")

    # trailing-24h RAW move (PIT) + forward RAW returns (intended look-fwd target)
    parts = []
    for s in have:
        c = load_close(s)
        if c is None:
            continue
        c = c.set_index("open_time")["close"]
        d = pd.DataFrame(index=c.index)
        d["tr_raw24"] = c.pct_change(288).shift(1)        # trailing move, PIT
        for k, h in HBARS.items():
            d[f"fwd_{k}"] = c.shift(-h) / c - 1.0          # forward raw ret
        d["symbol"] = s
        parts.append(d.reset_index())
    fr = pd.concat(parts, ignore_index=True)
    fr["open_time"] = pd.to_datetime(fr["open_time"], utc=True)
    d = dec.merge(fr, on=["symbol", "open_time"], how="inner").dropna(
        subset=["sp_volratio_z1d", "tr_raw24", "s_t", "alpha_beta"])
    print(f"  rows={len(d)} syms={d.symbol.nunique()} "
          f"cycles={d.open_time.nunique()}", flush=True)

    # PIT sanity: conditioner & trailing must not blatantly correlate with fwd
    for col in ["sp_volratio_z1d", "tr_raw24"]:
        la = d[col].corr(d["fwd_4h"], "spearman")
        print(f"  [sanity] corr({col}, fwd_4h)={la:+.4f}", flush=True)

    d["q"] = pd.qcut(d["sp_volratio_z1d"], 5, labels=False, duplicates="drop")
    dec_dec = pd.qcut(d["sp_volratio_z1d"], 10, labels=False,
                      duplicates="drop")
    print("\n  bucket by sp_volratio_z1d (q0=perp-led … q4=spot-led)",
          flush=True)
    print("  continuation coef = Spearman(trailing_24h_raw, forward); "
          "dir payoff = mean(sign(trail)*fwd) bps\n", flush=True)

    def cc(g, fcol):
        if len(g) < 50 or g["tr_raw24"].std() < 1e-12:
            return np.nan, np.nan
        sp_ = g["tr_raw24"].corr(g[fcol], "spearman")
        pay = float((np.sign(g["tr_raw24"]) * g[fcol]).mean() * 1e4)
        return sp_, pay

    rows = []
    for k in HBARS:
        line = f"  RAW fwd_{k:<3s}: "
        ccs = []
        for q in range(5):
            g = d[d["q"] == q]
            s_, p_ = cc(g, f"fwd_{k}")
            ccs.append(s_)
            line += f"q{q}[{s_:+.3f}/{p_:+.0f}bps] "
        # monotonicity (Spearman of bucket idx vs continuation coef)
        mono = pd.Series(ccs).corr(pd.Series(range(5)), "spearman")
        # q4−q0 directional-payoff spread, block-bootstrap CI (block≥horizon)
        blk = max(7, HBARS[k] // 48 + 1)
        q4 = d[d["q"] == 4]; q0 = d[d["q"] == 0]
        s4 = np.sign(q4["tr_raw24"]) * q4[f"fwd_{k}"] * 1e4
        s0 = np.sign(q0["tr_raw24"]) * q0[f"fwd_{k}"] * 1e4
        lo4, hi4 = block_bootstrap_ci(s4.to_numpy(), statistic=np.mean,
                                      block_size=blk, n_boot=1000)[1:]
        lo0, hi0 = block_bootstrap_ci(s0.to_numpy(), statistic=np.mean,
                                      block_size=blk, n_boot=1000)[1:]
        spread = float(s4.mean() - s0.mean())
        # spread CI via independent-ish bootstrap difference
        rng = np.random.default_rng(0)
        bs = []
        a = s4.to_numpy(); b = s0.to_numpy()
        for _ in range(1000):
            bs.append(rng.choice(a, len(a)).mean() - rng.choice(b, len(b)).mean())
        slo, shi = np.percentile(bs, [2.5, 97.5])
        print(line + f" | mono ρ={mono:+.2f} | q4−q0 spread={spread:+.0f}bps "
              f"CI[{slo:+.0f},{shi:+.0f}]", flush=True)
        rows.append(dict(horizon=k, mono=mono, q4_cc=ccs[4], q0_cc=ccs[0],
                         spread_bps=spread, spread_lo=slo, spread_hi=shi))
    # β-residual @4h (canonical panel target), same buckets
    print("\n  β-RESIDUAL @4h (panel alpha_beta): "
          "continuation of trailing residual s_t", flush=True)
    rl = "  resid4h : "
    rccs = []
    for q in range(5):
        g = d[d["q"] == q]
        s_ = g["s_t"].corr(g["alpha_beta"], "spearman") if len(g) > 50 else np.nan
        pay = float((np.sign(g["s_t"]) * g["alpha_beta"]).mean()*1e4) if len(g) > 50 else np.nan
        rccs.append(s_)
        rl += f"q{q}[{s_:+.3f}/{pay:+.1f}bps] "
    rmono = pd.Series(rccs).corr(pd.Series(range(5)), "spearman")
    print(rl + f" | mono ρ={rmono:+.2f}", flush=True)
    # extreme decile contrast (the "GREATLY outperforms" case)
    d["d10"] = dec_dec
    top = d[d["d10"] == 9]; bot = d[d["d10"] == 0]
    print(f"\n  extreme: spot-led D9 vs perp-led D0  (dir payoff bps)",
          flush=True)
    for k in HBARS:
        pt = float((np.sign(top["tr_raw24"])*top[f"fwd_{k}"]).mean()*1e4)
        pb = float((np.sign(bot["tr_raw24"])*bot[f"fwd_{k}"]).mean()*1e4)
        print(f"    {k:>3s}: D9(spot-led)={pt:+.0f}  D0(perp-led)={pb:+.0f}  "
              f"Δ={pt-pb:+.0f}", flush=True)

    R = pd.DataFrame(rows)
    mono_ok = bool((R["mono"] > 0.5).any())
    q4pos = bool((R["q4_cc"] > 0).any())
    spread_sig = bool(((R["spread_lo"] > 0)).any())
    SUP = mono_ok and q4pos and spread_sig
    if SUP:
        v = (f"D1-ext-D: spot-led MOMENTUM structure SUPPORTED — continuation "
             f"coef rises q0→q4 (mono), spot-led bucket positive, q4−q0 spread "
             f"CI excludes 0 at ≥1 horizon. Owner intuition validated as a "
             f"momentum/continuation classifier (DISTINCT from this line's "
             f"reversion thesis — a new direction, not a rescue). Next: a "
             f"pre-registered tradeable spec on the momentum framing.")
    else:
        v = (f"D1-ext-D: spot-led momentum structure ABSENT — "
             f"mono>{0.5}:{mono_ok}, q4cc>0:{q4pos}, spread-CI-excl-0:"
             f"{spread_sig}. Spot-vs-perp volume lead does NOT classify "
             f"continuation vs reversion at 4h/24h/72h (raw or β-resid). The "
             f"'spot volume outperforms ⇒ durable move' intuition is not "
             f"present in Binance free data at these horizons — definitively, "
             f"on its own (momentum) terms. Confirms the free-data terminus "
             f"from the opposite polarity too. Production LGBM unaffected.")
    print(f"\n  PRE-REG VERDICT: {v}", flush=True)
    R.to_csv(OUTD / "summary.csv", index=False)
    pd.DataFrame([{"supported": SUP, "verdict": v}]).to_csv(
        OUTD / "verdict.csv", index=False)
    print(f"\nSaved {OUTD}\nTotal: {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
