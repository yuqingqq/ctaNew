"""iter3: HARDEN the iter2 Target-A finding (sync order-parameter -> next-day realized vol).

(a) VOLUME control  — partial(state, vol_next | vol, vol_5d, log dvol). Rules out volume-proxying.
(b) ECONOMIC test   — era-locked: does adding the sync state to a vol-history forecast improve
                      out-of-sample next-day-vol prediction (rank-IC, RMSE), and at what extra
                      turnover of the implied inverse-vol sizing? Fit on TRAIN era, eval on TEST
                      era, BOTH directions.
(c) TAIL target     — partial(state, crash_next | vol, vol_5d) and (state, downside_next | ...).
(d) DRIFT<->SYNC    — light: is synchronization higher in RECENT (where iter1's manifold drift is)?

All targets are day d+1; state and controls use only day <= d. Confirmatory = (a); (c) exploratory.
Run:  python3 -m live.emergent_iter3_volstate
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from live.flow_harness import CUT, load_panel
from live.emergent_harness import block_spear_ci
from live.emergent_iter2_state import per_bar_state

MIN_BARS_DAY = 100   # a symbol needs >=100 of ~288 5-min bars for a reliable daily stat


def build_daily() -> pd.DataFrame:
    cols = ["symbol", "bar_time", "return_5min", "signed_pressure_5min", "imb1",
            "buy_quote_5min", "sell_quote_5min"]
    D = load_panel(cols)
    D["bar_time"] = pd.to_datetime(D["bar_time"], utc=True)
    print(f"panel {len(D):,} rows | {D.symbol.nunique()} syms", flush=True)

    S = per_bar_state(D)                       # per-bar excess-alignment state (n>=30 syms)
    Sd = (S.assign(day=S.index.floor("1D")).groupby("day")
          [["exc_signed_pressure_5min", "exc_imb1"]].mean())

    D["day"] = D["bar_time"].dt.floor("1D")
    D["quote"] = D["buy_quote_5min"].fillna(0) + D["sell_quote_5min"].fillna(0)
    D["logret"] = np.log1p(D["return_5min"].clip(lower=-0.99))
    D["negret"] = D["return_5min"].where(D["return_5min"] < 0)
    sym = D.groupby(["symbol", "day"]).agg(
        vol=("return_5min", "std"), n=("return_5min", "size"),
        dvol=("quote", "sum"), logret=("logret", "sum"),
        dn=("negret", "std"), mn=("return_5min", "min")).reset_index()
    sym = sym[sym["n"] >= MIN_BARS_DAY]
    sym["ret"] = np.expm1(sym["logret"])
    mkt = sym.groupby("day").agg(vol=("vol", "median"), dvol=("dvol", "median"),
                                 dn=("dn", "median"), mn=("mn", "median"),
                                 nsym=("symbol", "size"))
    mkt["crash07"] = (sym.assign(c=sym["ret"] < -0.07).groupby("day")["c"].mean())
    mkt["crash10"] = (sym.assign(c=sym["ret"] < -0.10).groupby("day")["c"].mean())

    df = mkt.join(Sd, how="inner").sort_index()
    df["vol_5d"] = df["vol"].rolling(5).mean()
    df["ldvol"] = np.log(df["dvol"].clip(lower=1))
    for t in ("vol", "crash07", "crash10", "dn"):
        df[f"{t}_next"] = df[t].shift(-1)
    return df.dropna()


def _resid(y, X):
    b = np.linalg.lstsq(X, y, rcond=None)[0]
    return y - X @ b


def part_a(df, states):
    print("\n=== (a) VOLUME control: is the sync->vol_next signal a volume proxy? ===", flush=True)
    print("    (compare control [vol,vol5d] vs [vol,vol5d,log dvol] in the SAME panel)", flush=True)
    m = df.index < CUT
    for st in states:
        for era, mask in (("OOS", m), ("REC", ~m)):
            d = df[mask]
            base_ctrl = np.column_stack([np.ones(len(d)), d["vol"], d["vol_5d"]])
            vol_ctrl = np.column_stack([base_ctrl, d["ldvol"]])
            row = []
            for ctrl in (base_ctrl, vol_ctrl):
                rs = _resid(d[st].to_numpy(), ctrl); rv = _resid(d["vol_next"].to_numpy(), ctrl)
                r, lo, up = block_spear_ci(rs, rv, block=10)
                row.append(f"{r:+.3f}[{lo:+.3f},{up:+.3f}]{'*' if (lo>0 or up<0) else ' '}")
            print(f"  {st:<26} {era}: no-vol {row[0]:<26} +vol {row[1]:<26} (n={len(d)})",
                  flush=True)


def part_b(df, state):
    print(f"\n=== (b) ECONOMIC: era-locked next-day-vol forecast, base[vol,vol5d] vs "
          f"+{state} ===", flush=True)
    m = df.index < CUT
    def fit_eval(train, test):
        Xtr_b = np.column_stack([np.ones(train.shape[0]), train["vol"], train["vol_5d"]])
        Xte_b = np.column_stack([np.ones(test.shape[0]), test["vol"], test["vol_5d"]])
        Xtr_a = np.column_stack([Xtr_b, train[state]]); Xte_a = np.column_stack([Xte_b, test[state]])
        y = train["vol_next"].to_numpy(); yt = test["vol_next"].to_numpy()
        out = {}
        for tag, Xtr, Xte in (("base", Xtr_b, Xte_b), ("aug", Xtr_a, Xte_a)):
            beta = np.linalg.lstsq(Xtr, y, rcond=None)[0]
            pred = Xte @ beta
            ic = pd.Series(pred).corr(pd.Series(yt), method="spearman")
            rmse = float(np.sqrt(np.mean((pred - yt) ** 2)))
            w = 1.0 / np.clip(pred, np.nanpercentile(pred, 5), None)
            turn = float(np.mean(np.abs(np.diff(w))) / np.mean(w))
            out[tag] = (ic, rmse, turn)
        return out
    for tr_era, te_era, tr_m, te_m in (("OOS", "REC", m, ~m), ("REC", "OOS", ~m, m)):
        o = fit_eval(df[tr_m], df[te_m])
        (ib, rb, tb), (ia, ra, ta) = o["base"], o["aug"]
        print(f"  fit {tr_era}->eval {te_era}: rankIC base {ib:+.3f} -> aug {ia:+.3f} "
              f"(Δ {ia-ib:+.3f}) | RMSE {rb:.4f}->{ra:.4f} ({100*(ra-rb)/rb:+.1f}%) | "
              f"turnover {tb:.3f}->{ta:.3f}", flush=True)


def part_c(df, states):
    print("\n=== (c) TAIL risk: partial(state, target_next | vol, vol_5d) ===", flush=True)
    m = df.index < CUT
    for tgt in ("crash07_next", "dn_next"):
        print(f"  target={tgt}", flush=True)
        for st in states:
            for era, mask in (("OOS", m), ("REC", ~m)):
                d = df[mask]
                X = np.column_stack([np.ones(len(d)), d["vol"], d["vol_5d"]])
                rs = _resid(d[st].to_numpy(), X); ry = _resid(d[tgt].to_numpy(), X)
                base, lo, up = block_spear_ci(rs, ry, block=10)
                sig = "off-0" if (lo > 0 or up < 0) else "spans 0"
                print(f"    {st:<26} {era}: partial {base:+.3f} [{lo:+.3f},{up:+.3f}] {sig}",
                      flush=True)


def part_d(df, states):
    print("\n=== (d) DRIFT<->SYNC (light): is synchronization higher in RECENT? ===", flush=True)
    m = df.index < CUT
    for st in states:
        print(f"  {st:<26} mean OOS {df.loc[m, st].mean():+.3f} | REC {df.loc[~m, st].mean():+.3f} "
              f"| contemp spearman(state,vol) {df[st].corr(df['vol'], method='spearman'):+.3f}",
              flush=True)


def main():
    df = build_daily()
    print(f"daily rows {len(df)} | {df.index.min().date()}..{df.index.max().date()} | "
          f"OOS {int((df.index<CUT).sum())} / REC {int((df.index>=CUT).sum())}", flush=True)
    states = ["exc_signed_pressure_5min", "exc_imb1"]
    part_a(df, states)
    part_b(df, "exc_signed_pressure_5min")
    part_c(df, states)
    part_d(df, states)


if __name__ == "__main__":
    main()
