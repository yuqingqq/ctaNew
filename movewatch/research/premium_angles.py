"""Run: python3 research/premium_angles.py

Descriptive (windows already read; decides nothing): signal 3 (`premium_h4`) traded from more
angles than research/premium_posthoc.py. Rows are the heavy-fade screen rows (S = discovery +
holdout, V = val_a + val_b), with premium_h4 joined from the signal-search rows on (asset, time).
Trades enter at t + 241 and exit at t + 1441 (24 h) or t + 4321 (72 h), stress 1, with funding.

Rule, stated before the run: an angle HOLDS only if its mean has the same sign in S and V and
each window's 95% CI (cluster-week bootstrap) excludes 0. With this many angles a pass is still a
forward-only candidate, not evidence.

Third look (`--conf`, rule stated before it ran): every PUMP angle whose mean is > 0 in both S
and V is re-measured on the 2020-01..2025-08 replay pumps (heavy-fade conf rows; consumed for
pump questions, but never split this way), with the same cuts. Descriptive. Dumps: x4_72 from
the signal-search rows (the heavy-fade rows carry no dump trades); 2020-25 dumps are NOT read."""
import json, sys
import numpy as np, pandas as pd
sys.path.insert(0, "/home/yuqing/ctaNew/movewatch")
import research.heavy_fade as hf
import research.signal_search as ss
import research.btc_hedge as bh
import research.runners as rn

A3 = next(c for c in json.load(open(ss.ROOT / "docs" / "SIGNAL_SEARCH_FREEZE.json"))["cells"] if c["id"] == "A3")
D3, P3 = A3["cuts"]
MEME = set(json.load(open(ss.ROOT / "docs" / "coingecko_meme_rank1500_2026-10-09.json"))["symbols"])


def base(sym: str) -> str:
    s = sym.removesuffix("USDT")
    for pre in ("1000000", "1000", "1M"):
        if s.startswith(pre) and len(s) > len(pre) + 1:
            return s[len(pre):]
    return s


def load() -> pd.DataFrame:
    df = pd.read_parquet(hf.OUT / "rows_screen.parquet")
    ref = pd.concat([ss.load_rows(ss.SCREEN), ss.load_rows(ss.VAL)], ignore_index=True)
    ref = ref[["asset", "time", "dir", "premium_h4"]].drop_duplicates(["asset", "time", "dir"])
    df = df.merge(ref, on=["asset", "time", "dir"], how="left")
    df["win"] = np.where(df["window"].isin(ss.SCREEN), "S", "V")
    df["meme"] = df["asset"].map(lambda a: base(a) in MEME)
    return df


def pnl(d: pd.DataFrame, cell: str, side: str) -> np.ndarray:
    if side == "short":
        return hf.net(d, cell)
    cost = np.array([2 * (bh.FEE + bh.slip(v)) for v in d["liq"].to_numpy(float)])
    return d[f"r_{cell}"].to_numpy(float) - cost - d[f"fund_{cell}"].to_numpy(float)


def holds(s: dict, v: dict) -> bool:
    if s["status"] != "OK" or v["status"] != "OK" or np.sign(s["mean"]) != np.sign(v["mean"]):
        return False
    return all((c["ci"][0] > 0) if c["mean"] > 0 else (c["ci"][1] < 0) for c in (s, v))


def fmt(c: dict) -> str:
    if c["status"] != "OK":
        return f"n{c['n']:>4} {'(too few)':>22}"
    return f"n{c['n']:>4} {100 * c['mean']:+6.2f}% [{100 * c['ci'][0]:+6.2f},{100 * c['ci'][1]:+6.2f}] med {100 * c['median']:+5.1f}%"


def masks(p: pd.DataFrame, lo: float, hi: float, q10: float, q90: float, pumps: bool) -> dict:
    pr, vp = p["premium_h4"], p["vol_persist"]
    g = {"all": pr.notna(), "discount tercile": pr <= lo, "middle tercile": (pr > lo) & (pr <= hi), "premium tercile": pr > hi,
         f"deepest 10% (<= {1e4 * q10:+.0f} bp)": pr <= q10, f"highest 10% (> {1e4 * q90:+.0f} bp)": pr > q90}
    if pumps:
        g.update({"quiet & premium": (vp <= hf.QUIET) & (pr > hi), "quiet & discount": (vp <= hf.QUIET) & (pr <= lo),
                  "heavy & premium": (vp > hf.HEAVY) & (pr > hi), "heavy & discount": (vp > hf.HEAVY) & (pr <= lo),
                  "non-meme discount": (~p["meme"]) & (pr <= lo), "non-meme heavy & discount": (~p["meme"]) & (vp > hf.HEAVY) & (pr <= lo),
                  "meme premium": p["meme"] & (pr > hi)})
    return g


def pump_cuts(p: pd.DataFrame) -> tuple:
    q10, q90 = np.quantile(p.loc[p["win"] == "S", "premium_h4"].dropna(), [0.1, 0.9])  # outcome-blind, S feature distribution
    return D3, P3, q10, q90


def run_pumps(df: pd.DataFrame) -> list:
    p = df[(df["dir"] == 1) & df["premium_h4"].notna()]
    out = []
    for g, m in masks(p, *pump_cuts(p), True).items():
        x = p[m.to_numpy()]
        for cell in ("S24", "S72", "S24_stop25"):
            for side in (("short",) if cell.endswith("stop25") else ("short", "long")):
                res = {w: hf.stats(pnl(x[x["win"] == w], cell, side), x.loc[x["win"] == w, "time"]) for w in ("S", "V")}
                out.append((g, cell, side, res["S"], res["V"], holds(res["S"], res["V"])))
    return out


def run_dumps() -> list:
    d = pd.concat([ss.load_rows(ss.SCREEN).assign(win="S"), ss.load_rows(ss.VAL).assign(win="V")], ignore_index=True)
    d = d[(d["dir"] == -1) & d["premium_h4"].notna() & d["x4_72"].notna()]
    sp = d.loc[d["win"] == "S", "premium_h4"]
    lo, hi = np.quantile(sp, [1 / 3, 2 / 3])
    q10, q90 = np.quantile(sp, [0.1, 0.9])
    out = []
    for g, m in masks(d, lo, hi, q10, q90, False).items():
        x = d[m.to_numpy()]
        for side in ("short", "long"):  # long = buy the dump = fade
            res = {w: hf.stats(ss.net(x[x["win"] == w], "x4_72", side == "short"), x.loc[x["win"] == w, "time"]) for w in ("S", "V")}
            out.append((g, "4->72h", side, res["S"], res["V"], holds(res["S"], res["V"])))
    return out


def conf_pumps(df_screen: pd.DataFrame) -> pd.DataFrame:
    cache = hf.OUT / "premium_conf_pumps.parquet"
    c = pd.read_parquet(hf.OUT / "rows_conf.parquet")
    c = c[c["dir"] == 1].copy()  # pumps only: the 2020-25 dumps stay unread
    if cache.exists():
        got = pd.read_parquet(cache)
    else:
        got = rn.first4h(c[["asset", "time"]].drop_duplicates().reset_index(drop=True))[["asset", "time", "premium_h4"]]
        got.to_parquet(cache)
    c = c.merge(got, on=["asset", "time"], how="left")
    c["win"], c["meme"] = "C", c["asset"].map(lambda a: base(a) in MEME)
    return c


if __name__ == "__main__":
    df = load()
    print(f"rows {len(df)}; premium_h4 joined: pumps {((df['dir'] == 1) & df['premium_h4'].notna()).sum()}")
    pumps, dumps = run_pumps(df), run_dumps()
    for lab, rows in (("PUMPS", pumps), ("DUMPS, signal-search rows, 4->72h (long = buy the dump)", dumps)):
        print(f"\n== {lab}: group | trade | S | V | holds")
        for g, cell, side, s, v, h in rows:
            print(f"{g:28} {side:5} {cell:10} | {fmt(s)} | {fmt(v)} | {'HOLDS' if h else ''}")
    allrows = pumps + dumps
    print(f"\nangles {len(allrows)}, holding {sum(r[-1] for r in allrows)}: "
          f"{[(g, side, cell, 'positive' if s['mean'] > 0 else 'negative') for g, cell, side, s, v, h in allrows if h]}")
    if "--conf" in sys.argv:
        carry = [(g, cell, side) for g, cell, side, s, v, h in pumps
                 if s["status"] == "OK" and v["status"] == "OK" and s["mean"] > 0 and v["mean"] > 0]
        c = conf_pumps(df)
        p = df[(df["dir"] == 1) & df["premium_h4"].notna()]
        mk = masks(c, *pump_cuts(p), True)
        print(f"\n== THIRD LOOK, 2020-01..2025-08 pumps ({len(c)}; premium missing {c['premium_h4'].isna().sum()}): "
              f"{len(carry)} angles positive in both S and V")
        kept = 0
        for g, cell, side in carry:
            x = c[mk[g].to_numpy()]
            r = hf.stats(pnl(x, cell, side), x["time"])
            ok = r["status"] == "OK" and r["mean"] > 0
            kept += ok
            print(f"{g:28} {side:5} {cell:10} | C {fmt(r)} | {'still positive' if ok else ''}"
                  f"{' CI>0' if ok and r['ci'][0] > 0 else ''}")
        print(f"positive on C: {kept} of {len(carry)}")
