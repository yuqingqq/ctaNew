"""Long-tail sweep runner. Each candidate is an env-override REPLAY through the FROZEN v3 driver
(run_convexity_v3_regime_gate.sh) — no bot edit, no pred regen. Replays ~22s each; runs POOL at a time.

Waves:
  w1_squeeze : attack the BULL May short-squeeze (owns the -4937 maxDD). Levers that veto/taper shorting
               recent rockets, de-gross bull, or vol-target the book.
  w2_grind   : attack the BEAR shallow-grind bleed (-1391 bps, Sharpe -1.54) WITHOUT hurting BEAR_DEEP.
               Levers: grind corr-short pool, deeper depth-ramp onset, mid-bear de-gross band.

Usage: python3 -m live.phase_longtail_sweep <wave> [pool]
Writes live/state/longtail/ledger.csv and one state dir per tag.
"""
import sys, os, time, subprocess, json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np, pandas as pd
import warnings; warnings.filterwarnings("ignore")

REPO = Path("/home/yuqing/ctaNew")
DRIVER = REPO / "live/run_convexity_v3_regime_gate.sh"
ROOT = REPO / "live/state/longtail"
LEDGER = ROOT / "ledger.csv"
ANN = np.sqrt(365)
GRIND_THR = -0.25

# ---------------- wave plans (env overrides applied on top of the frozen driver) ----------------
def plan(wave):
    P = []
    if wave == "w1_squeeze":
        # (a) drop recent-ROCKET shorts (ret_3d > thr) — direct anti-squeeze veto, symmetric to SHORT_MIN=-0.20
        for thr in ["0.15", "0.20", "0.30", "0.40"]:
            P.append((f"w1_smax{thr}", {"SHORT_MAX_RET3D": thr}))
        # (b) TAPER rocket shorts instead of dropping (keep some, half/zero weight)
        for thr, mult in [("0.15", "0.5"), ("0.20", "0.5"), ("0.20", "0.0"), ("0.30", "0.5")]:
            P.append((f"w1_taper{thr}x{mult}", {"SHORT_RET3D_TAPER_THR": thr, "SHORT_RET3D_TAPER_MULT": mult}))
        # (c) de-gross the bull sleeve (blunt)
        for m in ["0.75", "0.50", "0.25"]:
            P.append((f"w1_bullmult{m}", {"BULL_GROSS_MULT": m}))
        # (d) vol-target the whole book — de-gross when trailing realized vol high (squeeze = vol spike)
        for vt in ["40", "60", "80"]:
            P.append((f"w1_voltgt{vt}", {"VOL_TARGET": vt, "VOL_TARGET_WIN": "30",
                                          "VOL_TARGET_FLOOR": "0.30", "VOL_TARGET_CAP": "1.00"}))
    elif wave == "w1b_ranker":
        # last honest bull-scoped lever: change WHICH names the bull sleeve shorts (not sizing).
        # current = return_1d (short recent gainers = squeeze-prone by design). Test alternatives.
        for r in ["pred", "rvol_7d", "atr_pct", "comp_rvol", "comp_ret1d"]:
            P.append((f"w1b_rank_{r}", {"BULL_SHORT_RANK": r}))
    elif wave == "w2_grind":
        # (a) grind-conditional corr-short pool (validated +23bp/mo per code comment)
        for pool in ["4", "6", "8"]:
            P.append((f"w2_corrpool{pool}", {"SHORT_CORR_GRIND_POOL": pool, "SHORT_CORR_GRIND_THR": "-0.25"}))
        # (b) deeper depth-ramp onset — push shallow grind (-10..-25%) gross toward 0
        for d0 in ["0.15", "0.18", "0.20", "0.25"]:
            P.append((f"w2_d0_{d0}", {"BEAR_DEPTH_D0": d0, "BEAR_DEPTH_D1": "0.30"}))
        # (c) mid-bear de-gross band (grind zone only, keep deep + mild full)
        for lo, hi in [("-0.22", "-0.13"), ("-0.25", "-0.12"), ("-0.25", "-0.15")]:
            P.append((f"w2_mid{lo}_{hi}", {"BEAR_MID_LO": lo, "BEAR_MID_HI": hi}))
    else:
        raise SystemExit(f"unknown wave {wave}")
    return P

# ---------------- eval helpers (same defs as phase_longtail_eval) ----------------
def dsh(s):
    d = (s.fillna(0) / 1e4).resample("1D").sum()
    return float(d.mean() / d.std() * ANN) if d.std() > 0 else np.nan
def maxdd(s):
    eq = s.fillna(0).cumsum(); return float((eq - eq.cummax()).min())
def load_cyc(state_dir):
    f = Path(state_dir) / "state" / "cycles.csv"
    if not f.exists(): return None
    c = pd.read_csv(f); c["open_time"] = pd.to_datetime(c["open_time"], utc=True)
    return c.sort_values("open_time").set_index("open_time")
def masks(c):
    reg = c["regime"].astype(str); b = c["btc_ret_30d"]
    return {"ALL": pd.Series(True, index=c.index), "BULL": reg == "bull", "SIDE": reg == "side",
            "BEAR_DEEP": (reg == "bear") & (b < GRIND_THR), "BEAR_GRIND": (reg == "bear") & (b >= GRIND_THR)}
def metrics(c):
    m = masks(c); out = {}
    for k, mk in m.items():
        s = c.loc[mk, "pnl_bps"]
        out[f"{k}_tot"] = round(float(s.fillna(0).sum()), 0)
        out[f"{k}_sh"] = round(dsh(s), 3)
        out[f"{k}_dd"] = round(maxdd(s), 0)
    return out

def run_one(tag, overrides):
    sd = ROOT / tag
    sd.mkdir(parents=True, exist_ok=True)
    args = [f"{k}={v}" for k, v in overrides.items()]
    t0 = time.time()
    subprocess.run(["bash", str(DRIVER), str(sd), *args], cwd=str(REPO),
                   stdout=open(sd / "sweep.log", "w"), stderr=subprocess.STDOUT)
    c = load_cyc(sd)
    if c is None:
        return dict(tag=tag, ok=0, **{k: v for k, v in overrides.items()})
    rec = dict(tag=tag, ok=1, elapsed=round(time.time() - t0), **metrics(c))
    rec["_ov"] = json.dumps(overrides)
    return rec

def main():
    wave = sys.argv[1]
    pool = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    P = plan(wave)
    print(f"[{wave}] {len(P)} configs, pool={pool}", flush=True)
    recs = []
    with ThreadPoolExecutor(max_workers=pool) as ex:
        futs = {ex.submit(run_one, tag, ov): tag for tag, ov in P}
        for f in futs:
            pass
        for f in futs:
            r = f.result()
            recs.append(r)
            print(f"  {r['tag']:22s} ALL_sh {r.get('ALL_sh','?'):>7} ALL_dd {r.get('ALL_dd','?'):>8} "
                  f"BULL_dd {r.get('BULL_dd','?'):>8} GRIND_tot {r.get('BEAR_GRIND_tot','?'):>7} "
                  f"GRIND_sh {r.get('BEAR_GRIND_sh','?'):>7} DEEP_tot {r.get('BEAR_DEEP_tot','?'):>7}", flush=True)
    df = pd.DataFrame(recs); df["wave"] = wave
    hdr = not LEDGER.exists()
    df.to_csv(LEDGER, mode="a", header=hdr, index=False)
    print(f"[{wave}] done -> {LEDGER}", flush=True)

if __name__ == "__main__":
    main()
