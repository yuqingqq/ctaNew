"""Run: python3 research/premium_posthoc.py

Descriptive (windows already read; decides nothing): signal 3, the perp premium at +4 h (A3,
`premium_h4`), as a trade. Pumps by the frozen A3 terciles; dumps by outcome-blind terciles of the
S dump premium. Rows and nets are the signal search's (x4_72, stress 1, funding). The 2020-25
dump alerts are NOT read here."""
import json, sys
import numpy as np, pandas as pd
sys.path.insert(0, "/home/yuqing/ctaNew/movewatch")
import research.signal_search as ss

A3 = next(c for c in json.load(open(ss.ROOT / "docs" / "SIGNAL_SEARCH_FREEZE.json"))["cells"] if c["id"] == "A3")
WINDOWS = (("S", ss.SCREEN), ("V", ss.VAL))


def rows(w, d_: int) -> pd.DataFrame:
    d = ss.load_rows(w)
    d = d[(d["dir"] == d_) & d["premium_h4"].notna() & d["x4_72"].notna()].copy()
    d["long"], d["short"] = ss.net(d, "x4_72", d_ == 1), ss.net(d, "x4_72", d_ == -1)
    return d


def table(d: pd.DataFrame, cuts) -> None:
    d["grp"] = np.where(d["premium_h4"] <= cuts[0], "discount", np.where(d["premium_h4"] <= cuts[1], "middle", "premium"))
    raw = np.expm1(d["dir"] * d["raw4_72"])
    print(f"  {'group':9} {'n':>5} {'prem bp':>8} {'coin med':>8} {'+50%/-30%':>9} {'long mean':>9} {'long med':>8} {'short mean':>10} {'short med':>9}")
    for g, x in d.groupby("grp"):
        r = raw[x.index]
        tail = (r >= 0.5).mean() if x["dir"].iat[0] == 1 else (r >= 0.3).mean()
        print(f"  {g:9} {len(x):5d} {1e4 * x['premium_h4'].median():+8.1f} {100 * r.median():+7.1f}% {100 * tail:8.1f}% "
              f"{100 * x['long'].mean():+8.2f}% {100 * x['long'].median():+7.2f}% {100 * x['short'].mean():+9.2f}% {100 * x['short'].median():+8.2f}%")


print("PUMPS, A3 cuts (frozen):", np.round(1e4 * np.array(A3["cuts"]), 2), "bp; coin med = raw coin return +4h->+72h; tail = share >= +50%")
for lab, w in WINDOWS:
    d = rows(w, 1)
    ok = d["vol_persist"].notna()
    print(f"{lab}: {len(d)} pumps | IC(premium_h4, x4_72) {ss.spearman(d['premium_h4'].to_numpy(), d['x4_72'].to_numpy()):+.3f}"
          f" | rank corr with vol_persist {ss.spearman(d.loc[ok, 'premium_h4'].to_numpy(), d.loc[ok, 'vol_persist'].to_numpy()):+.3f}"
          f" | partial IC given vol_persist {ss.partial_ic(d, 'premium_h4', 'x4_72', ('vol_persist',)):+.3f}"
          f" | vol_persist partial IC given premium {ss.partial_ic(d, 'vol_persist', 'x4_72', ('premium_h4',)):+.3f}")
    table(d, A3["cuts"])

s = rows(ss.SCREEN, -1)
dcuts = np.quantile(s["premium_h4"], [1 / 3, 2 / 3])
print("\nDUMPS, outcome-blind S terciles:", np.round(1e4 * dcuts, 2), "bp; long = buy the dump; tail = rebound >= +30% (coin up)")
for lab, w in WINDOWS:
    d = rows(w, -1)
    print(f"{lab}: {len(d)} dumps | IC(premium_h4, x4_72 continuation) {ss.spearman(d['premium_h4'].to_numpy(), d['x4_72'].to_numpy()):+.3f}")
    table(d, dcuts)
    top = d["week"].value_counts().index[0]
    print(f"  without the busiest week ({top}, {(d['week'] == top).mean():.0%} of dumps), buy mean:",
          {g: round(100 * x["long"].mean(), 2) for g, x in d[d["week"] != top].groupby("grp")})

print("\nEXAMPLES (V pumps, coin return +4h->+72h)")
v = rows(ss.VAL, 1)
v["coin"] = np.expm1(v["raw4_72"])
for lab, x in (("premium, ran", v[v["premium_h4"] > A3["cuts"][1]].nlargest(3, "coin")),
               ("premium, faded", v[v["premium_h4"] > A3["cuts"][1]].nsmallest(2, "coin")),
               ("discount, faded", v[v["premium_h4"] <= A3["cuts"][0]].nsmallest(2, "coin")),
               ("discount, ran", v[v["premium_h4"] <= A3["cuts"][0]].nlargest(2, "coin"))):
    for r in x.itertuples():
        print(f"  {lab:16} {r.time:%Y-%m-%d %H:%M} {r.asset:14} premium {1e4 * r.premium_h4:+7.1f} bp  vol_persist {r.vol_persist:+.2f}  coin {100 * r.coin:+7.1f}%")
