"""Q3 primary statistic (RESEARCH_LOOP_20260707): does return_1d RANKING add value beyond generic
deep-bull long exposure? Stateless book-level paired test on the overlay's touched cycles.

Binding design-review fixes implemented:
- F2 persistence-matched placebo: per EPISODE a seeded random bijection remaps each pool name to a
  proxy whose return_1d it inherits for ranking; later entrants get appended deterministically.
  Ranking on remapped values preserves within-episode pick persistence. Per-arm turnover reported.
- F7 primary statistic: book-level totals, signal vs 1000-seed placebo distribution; exact
  exchangeability p-value reported. (Full-bot deltas are a separate consistency check, not this.)
- F4 control arms measured identically: BTC-long at same gross; top-2 by |corr_to_btc_1d|
  (beta PROXY — the panel's PIT coupling feature, not true beta; labeled as such).
- F3 episode accounting: episode count, top-episode share, drop-one-episode jackknife band.
- Costs: gross-of-cost primary; net secondary via per-arm mean turnover x 14.5 bps/leg x 2 legs.
- OOS = the test window (67 episodes). Recent = DESCRIPTIVE ONLY (6 episodes, 47 cycles < 100).
Book measurement (identical all arms): per touched cycle, equal-weight K=2 picks, pnl = sum of
next 6 bars' return_pct (24h hold) x gross 0.5, bps.
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd

REPO = Path("/home/yuqing/ctaNew")
STATE = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
    "/tmp/claude-1001/-home-yuqing-ctaNew/ecbd8f4c-236c-426c-85e5-e1f6b6edd11d/scratchpad/sideflat_canon")
GROSS, K, HOLD_BARS, COST_BPS_LEG, NSEED = 0.5, 2, 6, 14.5, 1000

def btc_fwd24_from_vision(t0, t1):
    """BTC 24h forward return (bps) per 4h bar from Binance Vision monthly klines."""
    import io, zipfile, requests
    from concurrent.futures import ThreadPoolExecutor
    def fm(per):
        try:
            r = requests.get("https://data.binance.vision/data/futures/um/monthly/klines/BTCUSDT/4h/"
                             f"BTCUSDT-4h-{per.strftime('%Y-%m')}.zip", timeout=20)
            if r.status_code != 200: return None
            z = zipfile.ZipFile(io.BytesIO(r.content)); raw = z.read(z.namelist()[0]).decode()
            hdr = 0 if raw.split(",", 1)[0] == "open_time" else None
            x = pd.read_csv(io.StringIO(raw), header=hdr)
            x = x.iloc[:, [0, 4]]; x.columns = ["open_time", "close"]
            v = pd.to_numeric(x["open_time"], errors="coerce")
            u = "us" if v.dropna().median() > 1e15 else "ms"
            x["open_time"] = pd.to_datetime(v, unit=u, utc=True)
            x["close"] = pd.to_numeric(x["close"], errors="coerce")
            return x
        except Exception:
            return None
    pers = pd.period_range(pd.Timestamp(t0).to_period("M") - 1, pd.Timestamp(t1).to_period("M") + 1, freq="M")
    with ThreadPoolExecutor(max_workers=12) as ex:
        parts = [q for q in ex.map(fm, pers) if q is not None]
    if not parts:
        raise RuntimeError("BTC kline fetch returned nothing")
    b = (pd.concat(parts).dropna().drop_duplicates("open_time")
         .set_index("open_time").sort_index()["close"])
    return ((b.shift(-HOLD_BARS) / b - 1) * 1e4)

def prep(win, panel):
    cy = pd.read_csv(STATE / f"k4_deepmom_{win}/cycles.csv", parse_dates=["open_time"]).sort_values("open_time")
    deep = cy["regime"].eq("bull") & (cy["btc_ret_30d"] >= 0.15)
    cy["ep"] = (deep & ~deep.shift(1, fill_value=False)).cumsum().where(deep)
    touched = cy[deep][["open_time", "ep"]]
    pr = pd.read_parquet(STATE / f"k4_deepmom_{win}/predictions.parquet",
                         columns=["symbol", "open_time", "eligible"])
    pr["open_time"] = pd.to_datetime(pr["open_time"], utc=True)
    pr = pr[pr["eligible"] == True]
    pool = pr.merge(panel, on=["symbol", "open_time"]).dropna(subset=["return_1d"])
    cycles = []   # (open_time, ep, names[], r1d[], fwd24[], corr[])
    for (t, ep), g in pool.merge(touched, on="open_time").groupby(["open_time", "ep"]):
        if len(g) >= K:
            cycles.append((t, ep, g["symbol"].to_numpy(), g["return_1d"].to_numpy(),
                           g["fwd24"].to_numpy(), g["corr_to_btc_1d"].to_numpy()))
    return cycles

def arm_totals(cycles, pick_fn):
    tot, turns, per_ep, prev = 0.0, [], {}, None
    n = 0
    for t, ep, names, r1d, fwd, corr in cycles:
        idx = pick_fn(t, ep, names, r1d, corr)
        if idx is None: continue
        pnl = np.nanmean(fwd[idx]) * GROSS
        if np.isnan(pnl): continue
        picks = set(names[idx])
        turns.append(1.0 if prev is None else len(picks - prev) / K)
        prev = picks
        tot += pnl; per_ep[ep] = per_ep.get(ep, 0.0) + pnl; n += 1
    return tot, (np.mean(turns) if turns else np.nan), per_ep, n

def main():
    panel = pd.read_parquet(REPO / "outputs/vBTC_features/panel_expanded_v0.parquet",
                            columns=["symbol", "open_time", "return_pct", "return_1d", "corr_to_btc_1d"])
    panel["open_time"] = pd.to_datetime(panel["open_time"], utc=True)
    panel = panel.sort_values(["symbol", "open_time"])
    panel["fwd24"] = panel.groupby("symbol")["return_pct"].transform(
        lambda s: s.rolling(HOLD_BARS).sum().shift(-(HOLD_BARS - 1))) * 1e4
    r1_lookup = {}  # (t) -> dict(name -> r1d), built lazily from cycles' own pools + full panel per t
    for win in ("oos", "ins"):
        cycles = prep(win, panel)
        eps = sorted({ep for _, ep, *_ in cycles})
        # full-universe r1d per touched t (proxies may sit outside the eligible pool)
        times = sorted({t for t, *_ in cycles})
        pt = panel[panel["open_time"].isin(times)]
        r1_by_t = {t: dict(zip(g["symbol"], g["return_1d"])) for t, g in pt.groupby("open_time")}

        sig_tot, sig_turn, sig_ep, nsig = arm_totals(cycles,
            lambda t, ep, names, r1d, corr: np.argpartition(-r1d, K - 1)[:K])
        beta_tot, beta_turn, _, _ = arm_totals(cycles,
            lambda t, ep, names, r1d, corr: (np.argpartition(-np.abs(np.nan_to_num(corr, nan=-9)), K - 1)[:K]
                                             if np.isfinite(corr).sum() >= K else None))
        # BTC-long control: the panel is built EX-BTC (175 alts, zero BTCUSDT rows), so BTC forward
        # returns MUST come from Vision klines. Results-review F1: the original panel-based lookup
        # silently produced +0 on the empty series — hard-fail instead of silently zeroing a control.
        btcp = btc_fwd24_from_vision(min(t for t, *_ in cycles), max(t for t, *_ in cycles))
        vals = np.array([btcp.get(t, np.nan) for t, *_ in cycles], dtype=float)
        if not np.isfinite(vals).any():
            raise RuntimeError("BTC-long control has no data — refusing to report a silent +0")
        btc_tot = float(np.nansum(vals)) * GROSS

        # placebo: episode-frozen bijections
        ep_cycle_names = {}
        for t, ep, names, *_ in cycles: ep_cycle_names.setdefault(ep, []).append(list(names))
        seeds_tot, seeds_turn = [], []
        for seed in range(1, NSEED + 1):
            maps = {}
            for ep, name_lists in ep_cycle_names.items():
                rng = np.random.default_rng(1000003 * seed + int(ep))
                mapping, used = {}, set()
                for names in name_lists:
                    new = [x for x in names if x not in mapping]
                    if not new: continue
                    pool_all = list(dict.fromkeys([x for lst in name_lists for x in lst]))
                    avail = [x for x in pool_all if x not in used]
                    rng.shuffle(avail)
                    for nname, tgt in zip(new, avail):
                        mapping[nname] = tgt; used.add(tgt)
                maps[ep] = mapping
            def pfn(t, ep, names, r1d, corr, maps=maps):
                m = maps[ep]; rt = r1_by_t[t]
                vals = np.array([rt.get(m.get(x, x), np.nan) for x in names])
                ok = np.isfinite(vals)
                if ok.sum() < K: return None
                vals[~ok] = -np.inf
                return np.argpartition(-vals, K - 1)[:K]
            ptot, pturn, _, _ = arm_totals(cycles, pfn)
            seeds_tot.append(ptot); seeds_turn.append(pturn)
        seeds_tot = np.array(seeds_tot)
        pval = ((seeds_tot >= sig_tot).sum() + 1) / (NSEED + 1)
        jk = {ep: sig_tot - v for ep, v in sig_ep.items()}
        top_ep = max(sig_ep.values()) if sig_ep else float("nan")
        c_sig = COST_BPS_LEG * 2 * GROSS * sig_turn * nsig
        c_pla = COST_BPS_LEG * 2 * GROSS * np.nanmean(seeds_turn) * nsig
        tag = "TEST" if win == "oos" else "DESCRIPTIVE (episode-limited: <3 material episodes, 47<100 cycles)"
        print(f"\n===== {win.upper()} — {tag} =====")
        print(f"episodes={len(eps)} scored_cycles={nsig}")
        print(f"SIGNAL   gross {sig_tot:+.0f} bps  turn/cyc {sig_turn:.2f}  net {sig_tot - c_sig:+.0f}")
        print(f"PLACEBO  median {np.median(seeds_tot):+.0f}  p90 {np.percentile(seeds_tot, 90):+.0f}  "
              f"p95 {np.percentile(seeds_tot, 95):+.0f}  turn {np.nanmean(seeds_turn):.2f}  "
              f"net(median) {np.median(seeds_tot) - c_pla:+.0f}")
        print(f"RANKING claim: signal rank p{100 * (seeds_tot < sig_tot).mean():.0f} of {NSEED} seeds; "
              f"exact p-value {pval:.4f}")
        print(f"CONTROLS gross: beta-proxy-ranked {beta_tot:+.0f} (turn {beta_turn:.2f})  BTC-long {btc_tot:+.0f}")
        print(f"episode profile: top-episode share {top_ep / max(abs(sig_tot), 1e-9):+.1%}; "
              f"jackknife (drop-one-episode) signal-total range "
              f"[{min(jk.values()):+.0f}, {max(jk.values()):+.0f}]")
    print("\nQ3DONE")

if __name__ == "__main__":
    main()
