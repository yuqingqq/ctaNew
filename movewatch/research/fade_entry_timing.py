"""Run: python3 research/fade_entry_timing.py [--stress 3]

Descriptive (all windows already read for pump questions; decides nothing): do we enter forward
test F1's short too early? Population as F1: non-meme heavy pumps (vol_persist > A2 heavy cut) in
the research rows (research/heavy_fade.py: rows_screen + rows_conf, 2020-01..2026-09).

Entries, declared before the run (t = alert minute; every trade is held 20 h = 1,200 min):
- delay: E4 (F1 itself, t+241), E6 (t+361), E8 (t+481), E12 (t+721), E24 (t+1441);
- confirmation: search the minute closes from t+241 to t+1441 for the first one that meets the
  rule and enter there; no trigger means no trade:
  WEAK3      close <= 0.97 x the +4 h price (the fade has started),
  SPIKE10    close >= 1.10 x the +4 h price (sell the next leg up),
  OFFPEAK10  close <= 0.90 x the highest close since the alert (10% off the post-alert high).

Net = -(exit/entry - 1) - 2 x (fee + stress x slippage(liquidity at the alert)) + funding over the
actual hold (Binance Vision, as the research). A confirmation entry trades a subset, so each one
is also shown against E4 on the same rows: better timing or just a different subset.
Reconciliation: E4 must equal the research S24 net of the same rows (else the run stops)."""
import json
import sys
import numpy as np, pandas as pd
sys.path.insert(0, "/home/yuqing/ctaNew/movewatch")
import research.btc_hedge as bh
import research.heavy_fade as hf
import research.premium_angles as pa
import research.replay as rp
import research.signal_search as ss
from research.common import Stale, panel_meta

STRESS = 3 if "--stress" in sys.argv and sys.argv[sys.argv.index("--stress") + 1] == "3" else 1
HOLD = 1200
DELAYS = {"E4": 241, "E6": 361, "E8": 481, "E12": 721, "E24": 1441}
CONF = ("WEAK3", "SPIKE10", "OFFPEAK10")
SPAN = 1441 + HOLD + 1                         # closes t .. t+2641


def population() -> pd.DataFrame:
    parts = []
    for w in ("screen", "conf"):
        d = pd.read_parquet(hf.OUT / f"rows_{w}.parquet")
        parts.append(d[(d["dir"] == 1) & (d["vol_persist"] > hf.HEAVY) & d["r_S24"].notna()])
    x = pd.concat(parts, ignore_index=True)
    x = x[~x["asset"].map(lambda a: pa.base(a) in pa.MEME)].reset_index(drop=True)
    x["win"] = np.where(x["window"].str.startswith("hfv_"), "C", np.where(x["window"].isin(["val_a", "val_b"]), "V", "S"))
    return x


def segments() -> dict:
    out = {k: ss.WINDOWS[k] for k in hf.SCREEN}
    out.update({s["alerts"]: s for s in hf.conf_segments()})
    return out


def entries(path: np.ndarray) -> dict:
    """Entry minute offset per rule (None = no trade). path[m] = close at alert minute + m."""
    out = dict(DELAYS)
    p4 = path[241]
    win = path[241:1442]
    peak = np.fmax.accumulate(np.where(np.isfinite(path[:1442]), path[:1442], -np.inf))[241:1442]
    for name, hit in (("WEAK3", win <= 0.97 * p4), ("SPIKE10", win >= 1.10 * p4), ("OFFPEAK10", win <= 0.90 * peak)):
        k = np.flatnonzero(hit & np.isfinite(win))
        out[name] = 241 + int(k[0]) if len(k) else None
    return out


def build(x: pd.DataFrame) -> pd.DataFrame:
    segs, recs = segments(), []
    for name, g in x.groupby("window"):
        w = segs[name]
        rp.PANEL_DIR = w["panel_dir"]
        syms, idx, C, V = rp.load_panel(*w["panel"])
        pdir = rp.OUT_DIR / f"panel_{w['panel'][0]}_{w['panel'][1]}"
        stale = Stale(panel_meta(w["panel_dir"], pdir, syms, idx), len(syms))
        T, t_ms0 = C.shape[1], int(idx[0].value // 1_000_000)
        for r in g.itertuples():
            i, t = int(r.i), int(r.t)
            path = np.full(SPAN, np.nan)
            hi = min(T, t + SPAN)
            path[: hi - t] = np.asarray(C[i, t:hi], np.float64)
            for rule, m in entries(path).items():
                rec = {"row": r.Index, "rule": rule, "entry_m": m}
                if m is not None and m + HOLD < SPAN and np.isfinite(path[m]) and np.isfinite(path[m + HOLD]) \
                        and not stale.coin(i, t + m, t + m + HOLD):
                    rec.update(r=path[m + HOLD] / path[m] - 1.0,
                               f0=t_ms0 + (t + m) * 60_000 + 59_999, f1=t_ms0 + (t + m + HOLD) * 60_000 + 59_999)
                recs.append(rec)
        rp.mem(f"paths {name}")
        del C, V
    return pd.DataFrame(recs)


def line(lab: str, v: np.ndarray, t: pd.Series, n_pop: int, base: np.ndarray | None = None) -> str:
    s = hf.stats(v, t)
    ci = f"[{100 * s['ci'][0]:+.2f}, {100 * s['ci'][1]:+.2f}]" if s["status"] == "OK" else "(few)"
    w, l = v[v > 0], v[v <= 0]
    out = (f"{lab:10} trades {len(v):5d} ({100 * len(v) / n_pop:3.0f}%) mean {100 * v.mean():+6.2f}% {ci:18} med {100 * np.median(v):+5.1f}% "
           f"win {100 * (v > 0).mean():3.0f}% p5 {100 * np.percentile(v, 5):+6.1f}% worst {100 * v.min():+5.0f}% L/W {-l.sum() / w.sum():.2f}")
    if base is not None:
        out += f" | E4 same rows {100 * np.nanmean(base):+.2f}%"
    return out


if __name__ == "__main__":
    x = population()
    print(f"F1 population: {len(x)} non-meme heavy pumps; stress {STRESS} + funding; hold 20 h")
    e = build(x)
    e = e.join(x[["time", "asset", "liq", "win"]], on="row")
    ok = e["r"].notna()
    pairs = {(a, m) for a, f0, f1 in zip(e.loc[ok, "asset"], e.loc[ok, "f0"], e.loc[ok, "f1"]) for m in bh.Funding().months(f0, f1)}
    print("funding:", json.dumps(bh.fetch_funding(pairs)))
    fund = bh.Funding()
    e["fund"] = [fund.total(a, f0, f1) if o else np.nan for a, f0, f1, o in zip(e["asset"], e["f0"], e["f1"], ok)]
    e["net"] = -e["r"] - np.array([2 * (bh.FEE + STRESS * bh.slip(v)) for v in e["liq"]]) + e["fund"]
    e4 = e[e["rule"] == "E4"].set_index("row")
    ref = pd.Series(hf.net(x, "S24", STRESS), index=x.index)
    both = e4["net"].notna() & ref.notna()
    gap = float((e4.loc[both, "net"] - ref[both]).abs().max())
    print(f"reconciliation: E4 n {int(e4['net'].notna().sum())} vs research S24 n {int(ref.notna().sum())}; max |diff| {gap:.2e}")
    if gap > 1e-9 or int(e4["net"].notna().sum()) != int(ref.notna().sum()):
        raise SystemExit("REFUSED: E4 does not reproduce the research S24 trade")
    for lab, sel in (("all", None), ("C 2020-25", "C"), ("V+S 25-26", ("V", "S"))):
        print(f"\n== {lab}")
        xs = x if sel is None else x[x["win"].isin([sel] if isinstance(sel, str) else list(sel))]
        for rule in list(DELAYS) + list(CONF):
            g = e[(e["rule"] == rule) & e["row"].isin(xs.index) & e["net"].notna()]
            base = e4.loc[g["row"], "net"].to_numpy() if rule in CONF else None
            print(line(rule, g["net"].to_numpy(), g["time"], len(xs), base))
    print("\nmean per trade by year:")
    yr = e["time"].dt.year
    rules = list(DELAYS) + list(CONF)
    print(f"{'year':6} " + " ".join(f"{r:>10}" for r in rules))
    for y in sorted(yr.unique()):
        print(f"{y:<6} " + " ".join(f"{100 * e.loc[(yr == y) & (e['rule'] == r), 'net'].mean():+9.2f}%" for r in rules))
    print("\nwhen the confirmation entries trigger (hours after the alert, median):",
          {r: round(float(e.loc[(e["rule"] == r) & e["entry_m"].notna(), "entry_m"].median()) / 60, 1) for r in CONF})
