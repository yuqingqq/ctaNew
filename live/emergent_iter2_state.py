"""iter2 (H-STATE): a market-wide SYNCHRONIZATION order-parameter from many atoms.

Each symbol's flow/imbalance sign is a "spin"; cross-symbol alignment (magnetization) is a
macro state a single-feature IC cannot see. Breadth grows 44->177, and |mean sign| ~ 1/sqrt(n),
so we use EXCESS alignment vs the independent-signs null (breadth-invariant):
    excess = (|mag| - sqrt(2/(pi n))) / sqrt((1-2/pi)/n)     # SDs above the no-herding null

Two targets (both sidestep the pointwise-return wall):
  A) forward market realized vol  — does synchronization predict the next day's vol, beyond
     vol's own persistence? (2nd moment)
  B) FORECASTABILITY of the reversal signal — is the short-term reversal (−return_5min) XS-IC
     stronger in synchronized regimes? If yes, size the reversal sleeve by the state — an
     emergent, usable payoff without OB predicting return.

Era-locked (OOS-defined quantile thresholds applied unchanged to REC) + block CI.
Run:  python3 -m live.emergent_iter2_state
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from live.flow_harness import CUT, load_panel, xsic
from live.emergent_harness import block_ci

MIN_N = 30          # symbols per bar for a reliable state
E0 = np.sqrt(2 / np.pi)
S0 = np.sqrt(1 - 2 / np.pi)


def per_bar_state(D: pd.DataFrame) -> pd.DataFrame:
    bt = D["bar_time"].to_numpy("datetime64[ns]")
    codes, uniq = pd.factorize(bt, sort=True)
    k = len(uniq)
    n = np.bincount(codes, minlength=k).astype(float)
    out = {"n": n}
    for feat in ("signed_pressure_5min", "imb1"):
        sgn = np.sign(D[feat].to_numpy())
        sgn[~np.isfinite(sgn)] = 0.0
        mag = np.bincount(codes, weights=sgn, minlength=k) / np.maximum(n, 1)
        excess = (np.abs(mag) - E0 / np.sqrt(n)) / (S0 / np.sqrt(n))
        out[f"mag_{feat}"] = mag
        out[f"exc_{feat}"] = excess
    S = pd.DataFrame(out, index=pd.DatetimeIndex(uniq).tz_localize("UTC"))
    return S[S["n"] >= MIN_N]


def target_B(D, S):
    print("=== TARGET B: forecastability — reversal (−return_5min) XS-IC by alignment quintile ===",
          flush=True)
    ic = xsic(D, "return_5min", "fwd_5m")           # per-bar, negative (reversal)
    ic.index = pd.to_datetime(ic.index, utc=True)
    for state in ("exc_signed_pressure_5min", "exc_imb1"):
        s = S[state].copy()
        s.index = pd.to_datetime(s.index, utc=True)
        df = pd.DataFrame({"ic": ic}).join(s.rename("st"), how="inner").dropna()
        m_oos = df.index < CUT
        # era-locked quintile thresholds from OOS
        qs = np.nanquantile(df.loc[m_oos, "st"], [0.2, 0.4, 0.6, 0.8])
        lab = np.digitize(df["st"].to_numpy(), qs)   # 0..4
        df["q"] = lab
        print(f"\n  state={state}  (Q0=least synchronized … Q4=most; thresholds from OOS)", flush=True)
        print(f"  {'Q':<3}{'OOS meanIC [95% CI]':<34}{'REC meanIC [95% CI]':<34}", flush=True)
        for q in range(5):
            oo = df[(df.q == q) & m_oos]["ic"]
            rr = df[(df.q == q) & ~m_oos]["ic"]
            a1, l1, u1 = block_ci(oo, block_days=1)
            a2, l2, u2 = block_ci(rr, block_days=1)
            print(f"  {q:<3}{f'{a1:+.4f} [{l1:+.4f},{u1:+.4f}]':<34}"
                  f"{f'{a2:+.4f} [{l2:+.4f},{u2:+.4f}]':<34}", flush=True)
        # monotonic strengthening of reversal (more negative) with alignment?
        mo = df[m_oos].groupby("q")["ic"].mean()
        mr = df[~m_oos].groupby("q")["ic"].mean()
        print(f"  Q4−Q0 (OOS): {mo.get(4,np.nan)-mo.get(0,np.nan):+.4f} | "
              f"(REC): {mr.get(4,np.nan)-mr.get(0,np.nan):+.4f}  "
              "(negative = reversal STRONGER when synchronized)", flush=True)


def target_A(D, S):
    print("\n=== TARGET A: does synchronization predict next-day market realized vol? ===", flush=True)
    D = D.copy()
    D["day"] = D["bar_time"].dt.floor("1D")
    # per-symbol daily realized vol from 5-min returns -> market median per day
    rv = (D.groupby(["symbol", "day"])["return_5min"].std().reset_index()
          .groupby("day")["return_5min"].median().rename("vol"))
    Sd = S.copy()
    Sd["day"] = Sd.index.floor("1D")
    state_d = Sd.groupby("day")[["exc_signed_pressure_5min", "exc_imb1"]].mean()
    df = state_d.join(rv, how="inner").sort_index()
    df["vol_next"] = df["vol"].shift(-1)
    df = df.dropna()
    # richer vol control: today's vol + 5-day trailing vol (rule out simple vol-regime proxying)
    df["vol_5d"] = df["vol"].rolling(5).mean()
    df = df.dropna()
    m_oos = df.index < CUT

    def block_spear_ci(a, b, block=10, nboot=2000, seed=7):
        n = len(a)
        if n < block * 3:
            return (np.nan, np.nan, np.nan)
        rng = np.random.default_rng(seed)
        nb = int(np.ceil(n / block)); hi = n - block
        base = pd.Series(a).corr(pd.Series(b), method="spearman")
        out = np.empty(nboot)
        for i in range(nboot):
            st = rng.integers(0, hi + 1, nb)
            idx = (st[:, None] + np.arange(block)[None, :]).ravel()[:n]
            out[i] = pd.Series(a[idx]).corr(pd.Series(b[idx]), method="spearman")
        lo, up = np.nanpercentile(out, [2.5, 97.5])
        return (float(base), float(lo), float(up))

    for state in ("exc_signed_pressure_5min", "exc_imb1"):
        for era, mask in (("OOS", m_oos), ("REC", ~m_oos)):
            d = df[mask]
            ctrl = np.column_stack([np.ones(len(d)), d["vol"].to_numpy(), d["vol_5d"].to_numpy()])

            def resid(y):
                b = np.linalg.lstsq(ctrl, y, rcond=None)[0]
                return y - ctrl @ b
            rs = resid(d[state].to_numpy()); rv = resid(d["vol_next"].to_numpy())
            base, lo, up = block_spear_ci(rs, rv, block=10)
            raw = d[state].corr(d["vol_next"], method="spearman")
            sig = "off-0" if (lo > 0 or up < 0) else "spans 0"
            print(f"  {state:<26} {era}: raw {raw:+.3f} | partial(vol+vol5d) {base:+.3f} "
                  f"[{lo:+.3f},{up:+.3f}] {sig}  (n={len(d)})", flush=True)


def main():
    cols = ["symbol", "bar_time", "return_5min", "signed_pressure_5min", "imb1", "fwd_5m"]
    D = load_panel(cols)
    D["bar_time"] = pd.to_datetime(D["bar_time"], utc=True)
    print(f"panel {len(D):,} rows | {D.symbol.nunique()} syms", flush=True)
    S = per_bar_state(D)
    print(f"state bars (n>={MIN_N}): {len(S):,} | "
          f"OOS {int((S.index<CUT).sum()):,} / REC {int((S.index>=CUT).sum()):,}\n", flush=True)
    target_B(D, S)
    target_A(D, S)


if __name__ == "__main__":
    main()
