"""Run: python3 research/fade_runners.py

Descriptive (every window here is already read for pump questions; decides nothing; F1 unchanged):
what do forward test F1's big losers, the pumps that keep running, have in common at entry?
Population as F1 (= research/fade_entry_timing.py population()): non-meme heavy pumps, short
+4 h -> +24 h, stress 1 + funding (hf.net "S24"), 2020-01..2026-09.

Outcomes, declared before the run:
  BIG LOSS  net <= -20% (the runners)
  BIG WIN   net >= +20% (a feature that only picks volatile coins raises both)
  SPIKE     price touched +25% above entry during the hold (rows' stopped_S24_stop25)

Features, each known at entry (t = alert minute; entry = close of minute t+241):
  pre_1h      log C[t] / C[t-60]                the alerting move
  pre_24h     log C[t] / C[t-1440]              run-up over the day up to the alert
  post_4h     log C[t+241] / C[t]               alert -> entry: still rising or already fading
  hour4       log C[t+241] / C[t+181]           the last hour before entry
  off_peak    log C[t+241] / max C[t..t+241]    distance below the post-alert high (<= 0)
  rv_4h       std of 1-minute log returns over t..t+241, x sqrt(60) (hourly)
  vol_persist hf: log10 volume hours 2-4 / the 3 h before the alert (the heavy rule's variable)
  vol_h4      log10 V[t+181..t+240] / (V[t+1..t+180] / 3): hour-4 volume vs hours 1-3
  liq         liquidity at the alert (hf rows)
  mkt_4h      median coin log return t..t+241 (liq >= $5M, the coin excluded)
  co_pumps    other bases with a NEW pump alert in (t-1 h, t+4 h], per 100 live coins at t
  fund_pre    funding settled in the 8 h before entry (sum of rates; > 0 = longs pay shorts)
  premium_h4  perp premium over hour 4 (signal 3; missing where Vision had no premium file)
  rank        30-day volume rank (1 = most traded)
  age_d       days since listing
Missing values are a status per feature and era, never a silent drop; rows whose hold has no
funding archive are status NO_FUNDING (as in the research, which counts 2,271 trades).
Cuts: quintiles of each feature on C 2020-25 only, applied unchanged to S+V 2025-26.

Per feature and tail (bottom fifth Q1 <= the 20% cut, top fifth Q5 > the 80% cut), per era
(C, S+V): n, big-loss share in the tail vs the rest of the era, big-win share in the tail vs the
rest, tail mean net vs the rest. Pooled tail mean with its cluster-week 95% CI (hf.stats).
Computed predicates (a tail needs n >= 30 in each era, else it cannot pass):
  SEPARATES  big-loss share in the tail >= 1.5 x the rest's, in C and in S+V
  SKIP PAYS  tail mean net < 0 in C and in S+V, and the pooled CI upper bound < 0
  ALSO WINS  big-win share in the tail >= 1.5 x the rest's in C and in S+V (volatility, not runners)
Multiplicity: 15 features x 2 tails = 30 looks; with no real effect one or two pass by chance.
Controls: ORACLE (the trade's own +4h->+24h move: look-ahead) must pass SEPARATES and SKIP PAYS on
Q5, else the run stops; NOISE (seeded uniform) is reported beside the features.
Reconciliation (the run stops on failure): every population row joins the premium/rank/age frame
once; r_S24 and vol_persist recomputed from the panels equal the rows'.
Also printed: medians of each feature for big losers vs the rest, and the 20 worst trades.

--posthoc (added 2026-10-10 AFTER reading the run above, so it is selected on seen data and decides
nothing): 5 of the 20 worst trades had funding <= -50 bp in the 8 h before entry. Shows how much of
the big losses is funding paid by the short during the hold, whether pre-entry funding persists
into the hold, and the trades with fund_pre <= -50 / -100 bp per era (n, big loss / big win share,
mean net, pooled CI)."""
import json
import math
import sys
import numpy as np, pandas as pd
sys.path.insert(0, "/home/yuqing/ctaNew/movewatch")
import research.btc_hedge as bh
import research.fade_entry_timing as fet
import research.heavy_fade as hf
import research.premium_angles as pa
import research.premium_classes as pc
import research.replay as rp
from research.common import Stale, mkt_median, panel_meta

BIG = 0.20
RATIO = 1.5
MIN_TAIL = 30
SEED = 20261010
FEATS = ["pre_1h", "pre_24h", "post_4h", "hour4", "off_peak", "rv_4h", "vol_persist", "vol_h4", "liq",
         "mkt_4h", "co_pumps", "fund_pre", "premium_h4", "rank", "age_d"]
CONTROLS = ["ORACLE", "NOISE"]
SHOW = {"pre_1h": (100, "%"), "pre_24h": (100, "%"), "post_4h": (100, "%"), "hour4": (100, "%"), "off_peak": (100, "%"),
        "rv_4h": (100, "%/h"), "vol_persist": (1, ""), "vol_h4": (1, ""), "liq": (1e-6, "$M"), "mkt_4h": (100, "%"),
        "co_pumps": (1, "/100"), "fund_pre": (1e4, "bp"), "premium_h4": (1e4, "bp"), "rank": (1, ""), "age_d": (1, "d"),
        "ORACLE": (100, "%"), "NOISE": (1, "")}
H8_MS = 8 * 3600 * 1000


def fmt(f: str, v: float) -> str:
    k, u = SHOW[f]
    return f"{k * v:+.3g}{u}" if np.isfinite(v) else "nan"


def build(x: pd.DataFrame) -> pd.DataFrame:
    segs, recs = fet.segments(), {}
    for name, g in x.groupby("window"):
        w = segs[name]
        rp.PANEL_DIR = w["panel_dir"]
        syms, idx, C, V = rp.load_panel(*w["panel"])
        pdir = rp.OUT_DIR / f"panel_{w['panel'][0]}_{w['panel'][1]}"
        stale = Stale(panel_meta(w["panel_dir"], pdir, syms, idx), len(syms))
        LIQ = np.load(rp.OUT_DIR / f"blocks_{w['panel'][0]}_{w['panel'][1]}_r15.npz")["LIQ"]
        T, t_ms0 = C.shape[1], int(idx[0].value // 1_000_000)
        al = hf._alerts(w)
        al = al[al["direction"] == 1].assign(b=lambda d: d["asset"].map(pa.base))

        for r in g.itertuples():
            i, t = int(r.i), int(r.t)

            def px(j):
                return float(C[i, j]) if 0 <= j < T and np.isfinite(C[i, j]) else np.nan

            def lr(a, b):
                pa_, pb = px(a), px(b)
                return math.log(pb / pa_) if np.isfinite(pa_) and np.isfinite(pb) and not stale.coin(i, a, b) else np.nan

            rec = {"pre_1h": lr(t - 60, t), "pre_24h": lr(t - 1440, t), "post_4h": lr(t, t + 241), "hour4": lr(t + 181, t + 241)}
            seg = np.asarray(C[i, t:t + 242], np.float64) if t + 242 <= T else np.full(242, np.nan)
            if np.isfinite(seg).all() and not stale.coin(i, t, t + 241):
                rec["off_peak"] = math.log(seg[-1] / seg.max())
                rec["rv_4h"] = float(np.std(np.diff(np.log(seg)))) * math.sqrt(60)
            else:
                rec["off_peak"] = rec["rv_4h"] = np.nan
            v13 = float(np.asarray(V[i, t + 1:t + 181], np.float64).sum())
            v4 = float(np.asarray(V[i, t + 181:t + 241], np.float64).sum())
            rec["vol_h4"] = math.log10(v4 / (v13 / 3)) if v4 > 0 and v13 > 0 and not stale.coin(i, t, t + 240) else np.nan
            rec["mkt_4h"] = mkt_median(C, LIQ, t, t + 241, stale, exclude=i)
            live = int((np.isfinite(np.asarray(C[:, t], np.float64)) & ~stale.mask(t, t)).sum())
            near = al[(al["time"] > r.time - pd.Timedelta(hours=1)) & (al["time"] <= r.time + pd.Timedelta(hours=4))]
            rec["co_pumps"] = 100 * near.loc[near["b"] != pa.base(r.asset), "b"].nunique() / live if live else np.nan
            rec["f_entry"] = t_ms0 + (t + 241) * 60_000 + 59_999
            # reconciliation inputs
            p4, p24 = px(t + 241), px(t + 1441)
            rec["r_chk"] = p24 / p4 - 1.0 if np.isfinite(p4) and np.isfinite(p24) else np.nan
            after = float(np.asarray(V[i, t + 61:t + 241], np.float64).sum())
            before = float(np.asarray(V[i, t - 179:t + 1], np.float64).sum()) if t >= 179 else 0.0
            rec["vp_chk"] = math.log10(after / before) if after > 0 and before > 0 else np.nan
            recs[r.Index] = rec
        rp.mem(f"features {name}")
        del C, V, LIQ
    return pd.DataFrame.from_dict(recs, orient="index")


def tail_line(x: pd.DataFrame, f: str, net: np.ndarray, cuts: tuple[float, float]) -> list[dict]:
    fin = x[f].notna().to_numpy()
    out = []
    for tail, m in (("Q1", (x[f] <= cuts[0]).to_numpy()), ("Q5", (x[f] > cuts[1]).to_numpy())):
        res = {"feature": f, "tail": tail}
        for era in ("C", "SV"):
            e = (x["era"] == era).to_numpy()
            a, b = m & e, fin & ~m & e
            res[era] = {"n": int(a.sum()), "loss": float(x["big_loss"][a].mean()) if a.any() else np.nan,
                        "loss_rest": float(x["big_loss"][b].mean()) if b.any() else np.nan,
                        "win": float(x["big_win"][a].mean()) if a.any() else np.nan,
                        "win_rest": float(x["big_win"][b].mean()) if b.any() else np.nan,
                        "mean": float(net[a].mean()) if a.any() else np.nan, "rest": float(net[b].mean()) if b.any() else np.nan}
        s = hf.stats(net[m], x["time"][m])
        res["pooled"] = {"n": s["n"], "mean": s.get("mean", np.nan), "ci": s.get("ci"), "status": s["status"]}
        res["rest_pooled"] = float(net[fin & ~m].mean())
        enough = all(res[e]["n"] >= MIN_TAIL for e in ("C", "SV"))

        def ratio_ok(k):
            return all(res[e][k] >= RATIO * res[e][f"{k}_rest"] and res[e][k] > 0 for e in ("C", "SV"))
        res["SEPARATES"] = bool(enough and ratio_ok("loss"))
        res["SKIP_PAYS"] = bool(enough and all(res[e]["mean"] < 0 for e in ("C", "SV")) and s["status"] == "OK" and s["ci"][1] < 0)
        res["ALSO_WINS"] = bool(enough and ratio_ok("win"))
        out.append(res)
    return out


def show_tail(res: dict, cut: float) -> str:
    def r(e, k):
        a, b = res[e][k], res[e][f"{k}_rest"]
        return f"{100 * a:4.1f}/{100 * b:4.1f}%"
    p = res["pooled"]
    ci = f"[{100 * p['ci'][0]:+5.1f},{100 * p['ci'][1]:+5.1f}]" if p["status"] == "OK" else "(few)"
    flags = " ".join(k for k in ("SEPARATES", "SKIP_PAYS", "ALSO_WINS") if res[k])
    return (f"{res['feature']:11} {res['tail']} {'<=' if res['tail'] == 'Q1' else '> '}{fmt(res['feature'], cut):>9} "
            f"n {res['C']['n']:4d}/{res['SV']['n']:3d} | big loss tail/rest C {r('C', 'loss')} SV {r('SV', 'loss')} "
            f"| big win C {r('C', 'win')} SV {r('SV', 'win')} | mean C {100 * res['C']['mean']:+6.2f}% SV {100 * res['SV']['mean']:+6.2f}% "
            f"pooled {100 * p['mean']:+6.2f}% {ci:15} rest {100 * res['rest_pooled']:+5.2f}% {flags}")


def posthoc() -> None:
    f = pd.read_parquet(hf.OUT / "runner_features.parquet")
    pop = fet.population()[["asset", "time", "r_S24", "fund_S24", "liq"]]
    x = f.merge(pop, on=["asset", "time"], how="left", validate="one_to_one")
    if x["r_S24"].isna().any() or len(x) != len(f):
        raise SystemExit("REFUSED: post-hoc join")
    cost = np.array([2 * (bh.FEE + bh.slip(v)) for v in x["liq"]])
    if np.abs(-x["r_S24"] - cost + x["fund_S24"] - x["net"]).max() > 1e-12:
        raise SystemExit("REFUSED: net does not split into price + cost + funding")
    bl = x[x["big_loss"]]
    print(f"big losers {len(bl)}: summed loss split: price {100 * (-bl['r_S24']).sum() / len(bl):+.1f}%, "
          f"funding {100 * bl['fund_S24'].sum() / len(bl):+.1f}%, cost {-100 * cost[x['big_loss']].mean():+.2f}% per trade; "
          f"funding alone <= -5%: {int((bl['fund_S24'] <= -0.05).sum())}")
    ok = x["fund_pre"].notna()
    print(f"pre-entry funding vs funding over the hold: Spearman {x.loc[ok, 'fund_pre'].corr(x.loc[ok, 'fund_S24'], method='spearman'):+.2f}")
    for cut in (-0.005, -0.01):
        m = (x["fund_pre"] <= cut).to_numpy()
        parts = []
        for era in ("C", "SV"):
            e = (x["era"] == era).to_numpy()
            a, b = m & e, ~m & e
            parts.append(f"{era} n {int(a.sum())} big loss {100 * x['big_loss'][a].mean():.0f}% (rest {100 * x['big_loss'][b].mean():.0f}%) "
                         f"big win {100 * x['big_win'][a].mean():.0f}% (rest {100 * x['big_win'][b].mean():.0f}%) "
                         f"mean {100 * x['net'][a].mean():+.2f}% (price {100 * -x['r_S24'][a].mean():+.2f}%, funding {100 * x['fund_S24'][a].mean():+.2f}%)")
        s_ = hf.stats(x["net"].to_numpy()[m], x["time"][m])
        ci = f"[{100 * s_['ci'][0]:+.2f}, {100 * s_['ci'][1]:+.2f}]" if s_["status"] == "OK" else f"({s_['status']})"
        print(f"fund_pre <= {1e4 * cut:.0f} bp: " + " | ".join(parts) + f" | pooled mean {100 * s_['mean']:+.2f}% {ci}; "
              f"all others {100 * x['net'][~m].mean():+.2f}%")


if __name__ == "__main__" and "--posthoc" in sys.argv:
    posthoc()
elif __name__ == "__main__":
    x = fet.population()
    fr = pc.frame(require="vol_persist")[["asset", "time", "premium_h4", "rank", "age_d"]].drop_duplicates(["asset", "time"])
    x = x.merge(fr, on=["asset", "time"], how="left", indicator=True)
    if len(x) != len(fet.population()) or (x["_merge"] != "both").any():
        raise SystemExit(f"REFUSED: frame join: rows {len(x)}, unmatched {int((x['_merge'] != 'both').sum())}")
    x = x.drop(columns="_merge")
    x["era"] = np.where(x["win"] == "C", "C", "SV")
    e = build(x)
    x = x.join(e)
    dr = (x["r_chk"] - x["r_S24"]).abs()
    dv = (x["vp_chk"] - x["vol_persist"]).abs()
    print(f"reconciliation: r_S24 max |diff| {dr.max():.1e} (nan {int(dr.isna().sum())}); "
          f"vol_persist max |diff| {dv.max():.1e} (nan {int(dv.isna().sum())})")
    if dr.isna().any() or dr.max() > 1e-12 or dv.isna().any() or dv.max() > 1e-12:
        raise SystemExit("REFUSED: panel recompute does not reproduce the rows")
    pairs = {(a, m) for a, f in zip(x["asset"], x["f_entry"]) for m in bh.Funding().months(f - H8_MS, f)}
    print("funding:", json.dumps(bh.fetch_funding(pairs)))
    fund = bh.Funding()
    x["fund_pre"] = [fund.total(a, f - H8_MS, f) for a, f in zip(x["asset"], x["f_entry"])]
    x["ORACLE"] = x["r_S24"]
    x["NOISE"] = np.random.default_rng(SEED).uniform(size=len(x))
    net = hf.net(x, "S24", 1)
    no_fund = ~np.isfinite(net)                     # funding archive missing: a status, as the research (n 2,271)
    print(f"status NO_FUNDING (no Vision funding archive for the hold): {int(no_fund.sum())} rows: "
          f"{list(zip(x.loc[no_fund, 'asset'], x.loc[no_fund, 'time'].astype(str)))}")
    x, net = x[~no_fund].reset_index(drop=True), net[~no_fund]
    if len(x) != int(np.isfinite(hf.net(fet.population(), "S24", 1)).sum()):
        raise SystemExit("REFUSED: population differs from the research S24 rows")
    x["net"] = net
    x["big_loss"], x["big_win"] = net <= -BIG, net >= BIG
    x["spike"] = x["stopped_S24_stop25"].astype(bool)
    x[["time", "asset", "era", "net", "big_loss", "big_win", "spike"] + FEATS].to_parquet(hf.OUT / "runner_features.parquet")

    print(f"\nF1 population {len(x)} (C {int((x.era == 'C').sum())}, S+V {int((x.era == 'SV').sum())}); "
          f"big losses (net <= -20%) C {int(x.big_loss[x.era == 'C'].sum())} / SV {int(x.big_loss[x.era == 'SV'].sum())}; "
          f"big wins (>= +20%) C {int(x.big_win[x.era == 'C'].sum())} / SV {int(x.big_win[x.era == 'SV'].sum())}; "
          f"spiked +25% C {int(x.spike[x.era == 'C'].sum())} / SV {int(x.spike[x.era == 'SV'].sum())}")
    big_sum = -net[x.big_loss].sum()
    print(f"big losses cost {100 * big_sum / len(x):.2f}% per trade on average (all trades' mean {100 * net.mean():+.2f}%)")
    print("missing (status, not dropped): " + ", ".join(
        f"{f} C {int(x.loc[x.era == 'C', f].isna().sum())}/SV {int(x.loc[x.era == 'SV', f].isna().sum())}"
        for f in FEATS if x[f].isna().any()))

    print("\n== medians: big losers vs the rest (C | S+V)")
    for f in FEATS:
        cells = []
        for era in ("C", "SV"):
            g = x[x.era == era]
            cells.append(f"{fmt(f, g.loc[g.big_loss, f].median()):>9} vs {fmt(f, g.loc[~g.big_loss, f].median()):>9}")
        print(f"{f:11} C {cells[0]} | SV {cells[1]}")

    print("\n== tails (cuts = C quintiles); big loss / big win shown as tail/rest; mean net per trade")
    allres = []
    for f in FEATS + CONTROLS:
        c = x.loc[x.era == "C", f].dropna()
        cuts = (float(c.quantile(0.2)), float(c.quantile(0.8)))
        for res, cut in zip(tail_line(x, f, net, cuts), cuts):
            res["cut"] = cut
            allres.append(res)
            print(show_tail(res, cut))
    orc = next(r for r in allres if r["feature"] == "ORACLE" and r["tail"] == "Q5")
    if not (orc["SEPARATES"] and orc["SKIP_PAYS"]):
        raise SystemExit("REFUSED: the ORACLE positive control did not fire; the predicates are broken")
    real = [r for r in allres if r["feature"] in FEATS]
    for k in ("SEPARATES", "SKIP_PAYS", "ALSO_WINS"):
        hits = [f"{r['feature']} {r['tail']}" for r in real if r[k]]
        print(f"{k}: {len(hits)} of {len(real)} looks: {hits}")
    both = [f"{r['feature']} {r['tail']}" for r in real if r["SEPARATES"] and r["SKIP_PAYS"] and not r["ALSO_WINS"]]
    print(f"SEPARATES and SKIP PAYS without ALSO WINS: {both}")
    noise = [r for r in allres if r["feature"] == "NOISE"]
    print(f"NOISE control: SEPARATES {[r['tail'] for r in noise if r['SEPARATES']]}, SKIP PAYS {[r['tail'] for r in noise if r['SKIP_PAYS']]}")
    (hf.OUT / "runner_tails.json").write_text(json.dumps(allres, indent=1, default=float))

    print("\n== the 20 worst trades")
    cols = ["pre_1h", "pre_24h", "post_4h", "off_peak", "vol_persist", "liq", "mkt_4h", "co_pumps", "fund_pre", "premium_h4", "rank", "age_d"]
    print(f"{'time':16} {'asset':14} era {'net':>7} spike " + " ".join(f"{c:>10}" for c in cols))
    for r in x.nsmallest(20, "net").itertuples():
        print(f"{r.time:%Y-%m-%d %H:%M} {r.asset:14} {r.era:3} {100 * r.net:+6.0f}% {'yes' if r.spike else 'no ':5} "
              + " ".join(f"{fmt(c, getattr(r, c)):>10}" for c in cols))
