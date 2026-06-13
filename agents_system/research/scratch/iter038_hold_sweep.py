"""iter-038 — HOLD-sweep: fixed 4h hold (HOLD=1, full rebalance) vs 24h/6-sleeve (HOLD=6 current).

Reuses the iter-032 FAST engine VERBATIM (precompute_cycles / fast_cyc_weights) and the iter-031
held-book + iter-012 vol-norm stop mechanics. The ONLY thing varied is i31.HOLD ∈ {1,2,3,6}.

HOLD semantics (from i31.heldbook / volnorm_stop):
  active = cyc_w[t-HOLD+1 : t+1]; net = sum(active weights)/HOLD.
  HOLD=1 -> net = the current cycle's fresh book (full rebalance every 4h cycle, FULL turnover).
  HOLD=6 -> net = mean of last 6 cycles' books (24h hold, 6 overlapping sleeves, amortized turnover).

To expose the cost/freshness tradeoff we re-implement heldbook/volnorm_stop here with per-cycle
GROSS (pre-cost return), NET (post-cost), and TURNOVER instrumentation — identical math to i31,
verified equal to i31 to numerical tolerance.

Universes (x132 V0 preds, 2021-26, 8 folds, @4.5bps):
  - established-70 = HL68∩x132 (the clean within-model read per iter-034)
  - full-mature   = full-156 with the iter-035 maturity≥180d (180 4h-bar) per-cycle history gate
"""
from __future__ import annotations
import time, pickle, importlib.util as _ilu
from pathlib import Path
import numpy as np
import pandas as pd

REPO = Path("/home/yuqing/ctaNew")
SCR = REPO/"agents_system/research/scratch"
RC = REPO/"research/convexity_portable_2026-05-20/results/_cache"
OUT = REPO/"outputs/iter038"; OUT.mkdir(parents=True, exist_ok=True)

X132_PREDS = RC/"x132_expanded_v0_preds.parquet"
HL70_PREDS = RC/"x64_HL70_V5mv3_7cx_filter_t0.3_preds.parquet"
ANN = 6*365

_s = _ilu.spec_from_file_location("i32", SCR/"iter032_expanded_universe.py")
i32 = _ilu.module_from_spec(_s); _s.loader.exec_module(i32)
i31 = i32.i31
precompute_cycles = i32.precompute_cycles
fast_cyc_weights = i32.fast_cyc_weights
PRIMARY_COST = i31.PRIMARY_COST

# iter-012 stop constants (copied from i31)
GFLOOR = i31.GFLOOR; HEAL = i31.HEAL; RDAYS = i31.RDAYS
VOL_WIN = i31.VOL_WIN; WARMUP = i31.WARMUP; REC_K = i31.REC_K
SQRT_WIN = np.sqrt(VOL_WIN)


def ann(x):
    x = pd.Series(x).dropna()
    return x.mean()/x.std()*np.sqrt(ANN) if len(x) > 2 and x.std() > 0 else np.nan


def heldbook_instr(cyc_w, rs, cost, HOLD):
    """i31.heldbook math, instrumented. Returns per-cycle net(bps), gross(bps), turnover."""
    prev = {}; net_pnl = []; gross = []; turns = []
    for t in range(len(cyc_w)):
        active = cyc_w[max(0, t-HOLD+1):t+1]; net = {}
        for w in active:
            for s, wt in w.items(): net[s] = net.get(s, 0)+wt/HOLD
        alls = set(net) | set(prev)
        turn = sum(abs(net.get(s, 0)-prev.get(s, 0)) for s in alls)
        rl = rs[t]
        c = sum(net.get(s, 0)*rl.get(s, 0.0) for s in net if np.isfinite(rl.get(s, 0.0)))
        if not np.isfinite(c): c = 0.0
        ccost = turn*0.5*cost
        gross.append(c*1e4); net_pnl.append((c-ccost)*1e4); turns.append(turn)
        prev = net
    return (np.asarray(net_pnl), np.asarray(gross), np.asarray(turns))


def volnorm_stop_instr(cyc_w, rs, cost, HOLD, k=REC_K):
    """i31.volnorm_stop math, instrumented. Returns per-cycle net(bps), gross(bps), turnover."""
    n = len(cyc_w)
    net_pnl = np.empty(n); gross = np.empty(n); turns = np.empty(n)
    incr = np.empty(n); prev = {}
    eq = 0.0; peak = 0.0; stopped = False
    stop_peak = 0.0; trough = 0.0; stop_t = 0
    for t in range(n):
        dd = eq - peak
        if t >= 2:
            lo = max(0, t-VOL_WIN); seg = incr[lo:t]; seg = seg[np.isfinite(seg)]
            sigma = float(seg.std()) if len(seg) >= 2 else 0.0
        else:
            sigma = 0.0
        trig = k*sigma*SQRT_WIN
        can_fire = (t >= WARMUP) and (sigma > 0)
        if not stopped:
            if can_fire and (-dd >= trig):
                stopped = True; stop_peak = peak; trough = eq; stop_t = t
        else:
            trough = min(trough, eq); gap = stop_peak - trough
            healed = (gap > 0) and ((eq - trough) >= HEAL*gap)
            timed = (t - stop_t) >= RDAYS
            if (healed and eq > trough) or timed:
                stopped = False
        g = GFLOOR if stopped else 1.0
        active = cyc_w[max(0, t-HOLD+1):t+1]; net = {}
        for w in active:
            for s, wt in w.items(): net[s] = net.get(s, 0)+wt/HOLD
        scaled = {s: g*v for s, v in net.items()}
        alls = set(scaled) | set(prev)
        turn = sum(abs(scaled.get(s, 0.0)-prev.get(s, 0.0)) for s in alls)
        rl = rs[t]
        c = sum(scaled.get(s, 0.0)*rl.get(s, 0.0) for s in scaled if np.isfinite(rl.get(s, 0.0)))
        if not np.isfinite(c): c = 0.0
        ccost = turn*0.5*cost
        gross[t] = c*1e4; net_pnl[t] = (c-ccost)*1e4; turns[t] = turn
        step = net_pnl[t]; incr[t] = step if np.isfinite(step) else 0.0
        eq += incr[t]
        if eq > peak: peak = eq
        prev = scaled
    return net_pnl, gross, turns


def metrics_full(net_bps, gross_bps, turns):
    m = i31.metrics(net_bps)
    p = pd.Series(net_bps).dropna()
    gp = pd.Series(gross_bps).dropna()
    m["gross_tot"] = float(gp.sum())
    m["gross_Sharpe"] = float(ann(gp/1e4))
    m["avg_turn"] = float(np.nanmean(turns))   # avg per-cycle turnover (sum |Δw|)
    m["cost_tot"] = float(gp.sum() - p.sum())   # total cost paid (bps)
    return m


def main():
    t0 = time.time()
    print("="*100)
    print("iter-038 — HOLD-sweep (4h fixed vs 24h/6-sleeve) on champion held-book, BASE & +iter-012 stop")
    print("="*100, flush=True)

    PX = pickle.loads((REPO/"outputs/iter032/panel_x132.pkl").read_bytes())
    print(f"  loaded x132 panel: {len(PX['syms'])} syms, {len(PX['times'])} 4h-cycles", flush=True)
    PCX = precompute_cycles(PX)

    full156 = [s for s in PX["syms"] if s != "BTCUSDT"]
    hl70 = sorted(pd.read_parquet(HL70_PREDS, columns=["symbol"])["symbol"].unique())
    hl_in_x132 = [s for s in hl70 if s in set(PX["syms"]) and s != "BTCUSDT"]
    print(f"  established-70 (HL∩x132): {len(hl_in_x132)} legs; full-156: {len(full156)}", flush=True)

    def ids(names):
        return [PCX["sidx"][s] for s in names if s in PCX["sidx"]]

    # maturity≥180d per-cycle history gate (iter-032/035)
    draw = pd.read_parquet(X132_PREDS, columns=["symbol", "open_time", "pred"])
    draw["open_time"] = pd.to_datetime(draw["open_time"], utc=True)
    draw = draw[(draw["open_time"].dt.hour % 4 == 0) & (draw["open_time"].dt.minute == 0)].copy()
    draw = draw.sort_values(["symbol", "open_time"])
    draw["hist_bars"] = draw.groupby("symbol").cumcount()
    hb = draw.set_index(["symbol", "open_time"])["hist_bars"].to_dict()
    times = PCX["times"]; syms_by_id = PCX["syms"]
    hist_arr = []
    for ti, ot in enumerate(times):
        idz = PCX["cyc"][ti][0]
        hist_arr.append(np.array([hb.get((syms_by_id[j], ot), 0) for j in idz]))
    def hist_ok_180(ti, idz):
        return hist_arr[ti] >= 180

    fbt = PCX["fold_by_time"]
    folds_arr = np.array([fbt.get(t, -1) for t in times])

    # VERIFY instrumented engine == i31 at HOLD=6 (default) on established-70
    i31.HOLD = 6
    cyc_w70, rs70 = fast_cyc_weights(PCX, ids(hl_in_x132), hist_ok=None)
    ref_base = i31.heldbook(cyc_w70, rs70, PRIMARY_COST)*1e4
    ref_stop = i31.volnorm_stop(cyc_w70, rs70, PRIMARY_COST)
    ib_net, _, _ = heldbook_instr(cyc_w70, rs70, PRIMARY_COST, 6)
    is_net, _, _ = volnorm_stop_instr(cyc_w70, rs70, PRIMARY_COST, 6)
    print(f"  VERIFY instr==i31 @HOLD6: base maxabs {np.nanmax(np.abs(ref_base-ib_net)):.2e}  "
          f"stop maxabs {np.nanmax(np.abs(ref_stop-is_net)):.2e}", flush=True)

    universes = [
        ("established-70", ids(hl_in_x132), None),
        ("full-mature(>=180d)", ids(full156), hist_ok_180),
    ]
    HOLDS = [1, 2, 3, 6]

    rows = []
    for uname, sids, hg in universes:
        print("\n" + "="*100)
        print(f"UNIVERSE: {uname}")
        print("="*100, flush=True)
        print(f"  {'HOLD':>4} {'hrs':>4} {'var':<5}{'Sharpe':>8}{'maxDD':>9}{'Calmar':>8}"
              f"{'netPnL':>9}{'grossPnL':>9}{'cost':>8}{'avgTurn':>9}{'%pos':>6}", flush=True)
        for HOLD in HOLDS:
            cyc_w, rs = fast_cyc_weights(PCX, sids, hist_ok=hg)
            for kind, instr in (("base", heldbook_instr), ("stop", volnorm_stop_instr)):
                net, gross, turns = instr(cyc_w, rs, PRIMARY_COST, HOLD)
                m = metrics_full(net, gross, turns)
                rows.append(dict(universe=uname, HOLD=HOLD, hold_hrs=HOLD*4, variant=kind,
                                 Sharpe=m["Sharpe"], maxDD=m["maxDD"], Calmar=m["Calmar"],
                                 netPnL=m["tot"], grossPnL=m["gross_tot"], cost=m["cost_tot"],
                                 gross_Sharpe=m["gross_Sharpe"], avg_turn=m["avg_turn"],
                                 pct_pos=m["pct_pos"]))
                print(f"  {HOLD:>4} {HOLD*4:>4} {kind:<5}{m['Sharpe']:>+8.2f}{m['maxDD']:>+9.0f}"
                      f"{m['Calmar']:>+8.2f}{m['tot']:>+9.0f}{m['gross_tot']:>+9.0f}"
                      f"{m['cost_tot']:>8.0f}{m['avg_turn']:>9.3f}{m['pct_pos']:>6.1f}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(OUT/"iter038_hold_sweep.csv", index=False)
    print(f"\nDone [{time.time()-t0:.0f}s]  -> {OUT/'iter038_hold_sweep.csv'}", flush=True)
    return df


if __name__ == "__main__":
    main()
