"""Step 72: validation audit for the latest linear-model result.

This is intentionally a checker, not a new experiment.  It rebuilds the current
Step-71 headline variant and compares:

1. The existing logged engine used by Step 71.
2. A causal-order engine where exits are decided from state known at decision
   time t, then weights earn alpha[t], then trade state is updated.
3. A fold-1 nested-selection sensitivity, because the shared nested helper has
   no prior folds for k=1 and currently picks using the full run.

The goal is to quantify whether the reported result survives calculation-order
fixes before trusting it.
"""
from __future__ import annotations

import importlib.util
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path("/home/yuqing/ctaNew")
sys.path.insert(0, str(REPO))


def _imp(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, REPO / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


s64 = _imp("s64", "linear_model/scripts/64_meanrev_v2_backtest.py")
s65 = _imp("s65", "linear_model/scripts/65_tail_attrib_deconc.py")
s71 = _imp("s71", "linear_model/scripts/71_battery_alleligible.py")
s59 = s64.s59
from ml.research.alpha_v4_xs import block_bootstrap_ci


BASE_COST = s64.COST
GRID_STEP71 = [g for g in s64.GRID if g["hurdle"] == 0 and g["sub"] == "ic_pos"]


def _metrics(df: pd.DataFrame) -> dict:
    n = df["net"].to_numpy(float)
    sh = s59._sharpe(n)
    lo, hi = block_bootstrap_ci(n, statistic=s59._sharpe, block_size=7, n_boot=800)[1:]
    fp = sum(1 for _, g in df.groupby("fold") if s59._sharpe(g["net"]) > 0)
    return {
        "sharpe": float(sh),
        "lo": float(lo),
        "hi": float(hi),
        "fp": int(fp),
        "mean": float(n.mean()),
        "total": float(n.sum()),
    }


def _print(label: str, df: pd.DataFrame) -> dict:
    m = _metrics(df)
    fold_bits = []
    for f, g in df.groupby("fold"):
        fold_bits.append(f"f{int(f)}:{s59._sharpe(g['net']):+.1f}/{g['net'].sum():+.0f}")
    print(
        f"  {label:28s} Sh {m['sharpe']:+.2f}[{m['lo']:+.2f},{m['hi']:+.2f}] "
        f"fp={m['fp']}/9 mean={m['mean']:+.2f} total={m['total']:,.0f}",
        flush=True,
    )
    print("    " + "  ".join(fold_bits), flush=True)
    return m


def run_causal_order(apd, alpha_w, fund_w, pz_w, tic_w, sig, beta_w, p, cost=BASE_COST):
    """Mean-reversion engine with causal decision ordering.

    At decision time t:
    - Build subset/z from current predictions.
    - Exit using cumulative PnL and age known before alpha[t].
    - Refill and set weights.
    - Earn alpha[t]/funding[t] with those weights.
    - Update cumulative PnL/age for positions carried through this cycle.
    """
    times = sorted(apd[apd["fold"].isin(s64.OOS)]["open_time"].unique())[::s64.BLOCK]
    fold_of = apd.drop_duplicates("open_time").set_index("open_time")["fold"].to_dict()
    n_slots, sub, zin, hurdle = p["N"], p["sub"], p["zin"], p["hurdle"]
    opn: dict[str, dict] = {}
    prev_w: dict[str, float] = {}
    rows = []
    trades = []
    posrows = []

    for t in times:
        if t not in pz_w.index or t not in tic_w.index:
            continue
        pz = pz_w.loc[t]
        tic = tic_w.loc[t]
        ar = alpha_w.loc[t] if t in alpha_w.index else None
        fr = fund_w.loc[t] if t in fund_w.index else None
        bt = beta_w.loc[t] if beta_w is not None and t in beta_w.index else None

        subset = set(tic.dropna().sort_values(ascending=False).head(15).index) if sub == "top15" else set(tic.index[tic > 0])
        ss = [s for s in subset if pd.notna(pz.get(s))]
        if len(ss) >= 4:
            v = pz[ss]
            sd = v.std()
            z = (pz - v.mean()) / sd if sd > 1e-12 else pz * 0.0
        else:
            z = pd.Series(dtype=float)

        # Exit before seeing alpha[t].
        for s in list(opn):
            st = opn[s]
            zv = z.get(s, np.nan)
            should_exit = (
                (s not in subset)
                or (pd.notna(zv) and abs(zv) < s64.Z_OUT)
                or st["age"] >= s64.MAX_HOLD
                or st["cum"] <= s64.STOP_BPS * -1
            )
            if should_exit:
                trades.append(
                    {
                        "symbol": s,
                        "side": st["side"],
                        "entry": st["e_t"],
                        "exit": t,
                        "age": st["age"],
                        "cum_bps": st["cum"],
                    }
                )
                del opn[s]

        held = set(opn)
        cur_l = {s for s, st in opn.items() if st["side"] > 0}
        cur_s = {s for s, st in opn.items() if st["side"] < 0}
        need_l, need_s = n_slots - len(cur_l), n_slots - len(cur_s)
        if len(ss) >= 4:
            zr = z[ss].sort_values(ascending=False)

            def ok(sym: str) -> bool:
                return abs(pz.get(sym, 0.0)) * float(sig.get(sym, s64.sig_med)) * 1e4 >= hurdle

            for s in zr.index:
                if need_l <= 0 or zr[s] < zin:
                    break
                if s in held or not ok(s):
                    continue
                opn[s] = {"side": 1, "cum": 0.0, "age": 0, "e_t": t}
                held.add(s)
                need_l -= 1
            for s in reversed(list(zr.index)):
                if need_s <= 0 or zr[s] > -zin:
                    break
                if s in held or not ok(s):
                    continue
                opn[s] = {"side": -1, "cum": 0.0, "age": 0, "e_t": t}
                held.add(s)
                need_s -= 1

        L = sum(1 for st in opn.values() if st["side"] > 0)
        S = sum(1 for st in opn.values() if st["side"] < 0)
        w = {}
        for s, st in opn.items():
            w[s] = (0.5 / L) if st["side"] > 0 and L else (-0.5 / S if st["side"] < 0 and S else 0.0)

        net_beta = 0.0
        if bt is not None:
            for s, wi in w.items():
                bv = bt.get(s)
                if bv is not None and not pd.isna(bv):
                    net_beta += wi * bv
        w_btc = -net_beta

        gross = 0.0
        funding = 0.0
        period_trade_pnl: dict[str, float] = {}
        for s, wi in w.items():
            a = ar.get(s) if ar is not None else np.nan
            if a is not None and not pd.isna(a):
                contrib = wi * a * 1e4
                gross += contrib
                posrows.append(
                    {
                        "time": t,
                        "fold": fold_of.get(t, 0),
                        "symbol": s,
                        "side": opn[s]["side"],
                        "weight": wi,
                        "contrib_bps": contrib,
                    }
                )
                period_trade_pnl[s] = (a if opn[s]["side"] > 0 else -a) * 1e4
            fv = fr.get(s) if fr is not None else np.nan
            if fv is not None and not pd.isna(fv):
                funding += -wi * fv * 1e4

        all_keys = set(w) | set(prev_w)
        tc = sum(abs(w.get(k, 0.0) - prev_w.get(k, 0.0)) for k in all_keys) * cost
        tc += abs(w_btc - prev_w.get("__BTC__", 0.0)) * cost
        rows.append(
            {
                "time": t,
                "fold": fold_of.get(t, 0),
                "gross": gross,
                "funding": funding,
                "cost": tc,
                "net": gross + funding - tc,
                "n_open": len(opn),
            }
        )

        # Only now is alpha[t] known for the next decision.
        for s, pnl in period_trade_pnl.items():
            if s in opn:
                opn[s]["cum"] += pnl
                opn[s]["age"] += 1

        prev_w = dict(w)
        prev_w["__BTC__"] = w_btc

    return pd.DataFrame(rows), pd.DataFrame(trades), pd.DataFrame(posrows)


def nested_with(run_fn, apd, aw, fw, pzw, tw, sig, bw, fold1_policy="current"):
    runs = {i: run_fn(apd, aw, fw, pzw, tw, sig, bw, p) for i, p in enumerate(GRID_STEP71)}
    nd = []
    for k in range(1, 10):
        if k == 1:
            if fold1_policy == "current":
                pick = sorted(runs, key=lambda j: s59._sharpe(runs[j][0]["net"]))[len(runs) // 2]
            elif fold1_policy == "first":
                pick = 0
            elif fold1_policy == "second":
                pick = 1 if len(runs) > 1 else 0
            else:
                raise ValueError(fold1_policy)
        else:
            pick = max(
                runs,
                key=lambda j: s59._sharpe(runs[j][0][runs[j][0]["fold"] < k]["net"].to_numpy()),
            )
        df = runs[pick][0]
        nd.append(df[df["fold"] == k])
    return pd.concat(nd).sort_values("time")


def main():
    print("=" * 92, flush=True)
    print("  STEP 72: validation audit of Step-71 calculations", flush=True)
    print("=" * 92, flush=True)
    t0 = time.time()

    print("\nRebuilding Step-71 drop-2 all-eligible PIT inputs...", flush=True)
    o = s71.build(["BIOUSDT", "VVVUSDT"])
    apd, aw, pzw, tw, fw, bw, sig = o[0], o[1], o[2], o[3], o[4], o[5], o[6]

    gb = s64.GRID
    s64.GRID = GRID_STEP71
    s65.COST = BASE_COST
    try:
        original, _, _ = s65.nested(apd, aw, fw, pzw, tw, sig, bw, "design")
    finally:
        s64.GRID = gb
        s65.COST = BASE_COST

    print("\nEngine / selection comparison:", flush=True)
    rows = []
    rows.append({"case": "original_current_fold1", **_print("original/current fold1", original)})
    rows.append(
        {
            "case": "original_fold1_first",
            **_print(
                "original/fold1 first",
                nested_with(lambda *args: s65.runL(*args, constraint="design"), apd, aw, fw, pzw, tw, sig, bw, "first"),
            ),
        }
    )
    rows.append(
        {
            "case": "causal_current_fold1",
            **_print("causal/current fold1", nested_with(run_causal_order, apd, aw, fw, pzw, tw, sig, bw, "current")),
        }
    )
    rows.append(
        {
            "case": "causal_fold1_first",
            **_print("causal/fold1 first", nested_with(run_causal_order, apd, aw, fw, pzw, tw, sig, bw, "first")),
        }
    )

    out = REPO / "linear_model/results/step72_validation_audit"
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "engine_timing_summary.csv", index=False)
    print(f"\nSaved {out / 'engine_timing_summary.csv'}", flush=True)
    print(f"Total: {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
