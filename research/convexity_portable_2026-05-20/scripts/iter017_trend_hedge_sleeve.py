"""iter-017 — SCOPING DIAGNOSTIC: a TREND-FOLLOWING / CRISIS-ALPHA HEDGE SLEEVE run ALONGSIDE the
mean-rev book.

NOT a predictor (proven impossible: iter-006/008/010). NOT net-short-the-book (iter-008 squeezed).
A SEPARATE directional time-series-momentum engine sized as a HEDGE. The honest question:

  Are the trend sleeve's returns NEGATIVELY correlated with the mean-rev book DURING the drawdown
  episodes (crisis alpha), so combining book + w*trend shrinks portfolio maxDD / raises Calmar —
  NET of the whipsaw cost the trend sleeve pays in calm/choppy periods?

A trend-follower is REACTIVE (long sustained uptrends, short sustained downtrends). It tends to
profit in PROLONGED moves and lose in chop. The test is correlation-in-crisis + combined-portfolio
DD + episode-LOFO, NOT forecast skill.

Reuses X123 machinery verbatim:
  - the alt-index reconstruction (.shift(1) eq-weight cum log-ret of the panel's own alts ex BTC/ETH),
  - the X123 EXT/HL70/S44 per-cycle parquet for pnl_base (mean-rev book PnL), regime, fold, alt30.

The trend sleeve (simple, parameter-light, PIT):
  - time-series momentum: sign(trailing-Nd return) of the alt-index AND of BTC.
  - lookbacks: 30d and 90d (two standard TSMOM windows; we report each + their average; NO tuning
    to episodes).
  - sleeve return per cycle = position(sign, .shift(1)) * forward-HOLD-bar log return of the traded
    asset, minus cost on position flips.
  - traded asset = the equal-weight alt index (the thing the book is exposed to) AND BTC (reference).

STEP 3 decisive tests:
  1. crisis-alpha: corr(trend, book) OVERALL and WITHIN each DD episode; per-episode trend PnL.
  2. whipsaw cost: trend PnL in calm/non-episode periods (the insurance premium).
  3. combine book + w*trend at several hedge weights -> combined maxDD / Calmar; episode-LOFO.
  4. honest framing: does it actually profit in these alt-bears (esp. 2025_q4 which bounced) or
     whipsaw like the net-short?

Outputs: results/iter017_trend_sleeve_{EXT,HL70,S44}.parquet (per-cycle book + trend variants).
Console: all tables. Does NOT modify any cached preds or prior parquets.
"""
from __future__ import annotations
import time
from pathlib import Path
import pandas as pd, numpy as np

REPO = Path("/home/yuqing/ctaNew")
OUT = REPO / "research/convexity_portable_2026-05-20/results"
KLINES = REPO / "data/ml/test/parquet/klines"

HOLD = 6            # trade horizon (24h on 4h bars), same as the book
WIN30 = 180         # ~30d in 4h bars
WIN90 = 540         # ~90d in 4h bars
COST_BPS = 4.5      # production cost
COSTS = [1.0, 3.0, 4.5]
HEDGE_W = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5]  # trend-sleeve hedge weights (sleeve gross is ~1x leg)
SEED = 12345
N_PLACEBO = 200

EXT_EPISODES = [
    ("2022_luna",   "2022-05-01", "2022-07-31"),
    ("2022_ftx",    "2022-11-01", "2023-01-31"),
    ("2024_summer", "2024-06-01", "2024-09-30"),
    ("2025_q4",     "2025-09-01", "2025-12-31"),
]

X123 = {
    "EXT":  OUT / "X123_altbear_short_EXT.parquet",
    "HL70": OUT / "X123_altbear_short_HL70.parquet",
    "S44":  OUT / "X123_altbear_short_S44.parquet",
}
PREDS = {  # only used to recover the per-universe symbol set for the alt-index
    "EXT":  REPO / "research/convexity_portable_2026-05-20/results/_cache/x113_ext_v0_preds.parquet",
    "HL70": REPO / "research/convexity_portable_2026-05-20/results/_cache/x64_HL70_V5mv3_7cx_filter_t0.3_preds.parquet",
    "S44":  REPO / "research/convexity_portable_2026-05-20/results/_cache/x70_v0_3yr_preds.parquet",
}


def load_close(sym):
    sd = KLINES / sym / "5m"
    if not sd.exists():
        return None
    dfs = [pd.read_parquet(f, columns=["open_time", "close"]) for f in sorted(sd.glob("*.parquet"))]
    df = pd.concat(dfs, ignore_index=True).drop_duplicates("open_time").sort_values("open_time")
    df["open_time"] = pd.to_datetime(df["open_time"], utc=True)
    return df.set_index("open_time")["close"].astype(np.float64)


def ann(x):
    x = pd.Series(x).dropna()
    return x.mean() / x.std() * np.sqrt(6 * 365) if len(x) > 2 and x.std() > 0 else np.nan


def stats(pnl):
    p = pd.Series(pnl).dropna(); pb = p * 1e4
    eq = pb.cumsum(); mdd = (eq - eq.cummax()).min()
    annr = pb.mean() * 6 * 365
    cal = (annr / abs(mdd)) if (mdd < 0 and np.isfinite(mdd)) else np.nan
    return {"sharpe": ann(p), "maxDD": mdd, "calmar": cal,
            "totPnL": eq.iloc[-1] if len(eq) else np.nan, "pct_pos": (pb > 0).mean() * 100}


def calmar_of(pnl):
    pb = pd.Series(pnl).dropna() * 1e4
    if len(pb) < 3: return np.nan
    eq = pb.cumsum(); mdd = (eq - eq.cummax()).min()
    return (pb.mean() * 6 * 365 / abs(mdd)) if (mdd < 0 and np.isfinite(mdd)) else np.nan


def maxdd_of(pnl):
    pb = pd.Series(pnl).dropna() * 1e4
    if len(pb) < 1: return np.nan
    eq = pb.cumsum(); return (eq - eq.cummax()).min()


# --------------------------------------------------------------------------- build trend sleeve
def build_trend(label):
    """Build the trend-following sleeve per-cycle returns aligned to the X123 EXT/HL70/S44 panel.

    Returns a DataFrame indexed by open_time with: pnl_base, regime, fold, is_side, flag, alt30,
    and the trend variants pnl_trend_* (per-cycle, cost-netted)."""
    base = pd.read_parquet(X123[label])
    base["open_time"] = pd.to_datetime(base["open_time"], utc=True)
    base = base.sort_values("open_time").reset_index(drop=True)
    times = pd.DatetimeIndex(base["open_time"])

    # universe symbols for the alt-index (panel's own alts ex BTC/ETH) — same as X123
    syms = sorted(pd.read_parquet(PREDS[label], columns=["symbol"])["symbol"].unique())

    btc = load_close("BTCUSDT")
    b4 = btc[(btc.index.hour % 4 == 0) & (btc.index.minute == 0)]
    btc_r = np.log(b4 / b4.shift(1))                       # per-bar BTC log-ret

    # equal-weight alt index (cum log-ret of panel alts ex BTC/ETH)
    ret_map = {}
    for s in syms:
        if s in ("BTCUSDT", "ETHUSDT"):
            continue
        c = load_close(s)
        if c is None:
            continue
        c4 = c[(c.index.hour % 4 == 0) & (c.index.minute == 0)]
        ret_map[s] = np.log(c4 / c4.shift(1))
    ret4 = pd.concat([v.rename(k) for k, v in ret_map.items()], axis=1).sort_index()
    alt_r = ret4.mean(axis=1)                              # per-bar eq-weight alt-index log-ret
    alt_cum = alt_r.cumsum()
    btc_cum = btc_r.cumsum()

    # --- PIT trailing-window signs (.shift(1)): position taken at decision time ---
    def tsmom_sign(cum, win):
        sig = (cum - cum.shift(win)).shift(1)              # trailing-win cum log-ret, lagged
        return np.sign(sig)

    # --- forward HOLD-bar realized return of each asset (what the position earns this cycle) ---
    alt_fwd = (alt_cum.shift(-HOLD) - alt_cum)
    btc_fwd = (btc_cum.shift(-HOLD) - btc_cum)

    # build per-cycle sleeve returns on the panel timeline
    df = pd.DataFrame(index=times)
    df["alt_fwd"] = alt_fwd.reindex(times).values
    df["btc_fwd"] = btc_fwd.reindex(times).values
    for win, tag in [(WIN30, "30d"), (WIN90, "90d")]:
        df[f"alt_pos_{tag}"] = tsmom_sign(alt_cum, win).reindex(times).values
        df[f"btc_pos_{tag}"] = tsmom_sign(btc_cum, win).reindex(times).values

    # gross-return variants (before cost). position*forward_return.
    df["g_alt_30d"] = df["alt_pos_30d"] * df["alt_fwd"]
    df["g_alt_90d"] = df["alt_pos_90d"] * df["alt_fwd"]
    df["g_btc_30d"] = df["btc_pos_30d"] * df["btc_fwd"]
    df["g_btc_90d"] = df["btc_pos_90d"] * df["btc_fwd"]
    # combined dual-lookback (average of 30d+90d signs on the alt index): the standard TSMOM blend
    df["alt_pos_avg"] = (df["alt_pos_30d"] + df["alt_pos_90d"]) / 2.0
    df["btc_pos_avg"] = (df["btc_pos_30d"] + df["btc_pos_90d"]) / 2.0
    df["g_alt_avg"] = df["alt_pos_avg"] * df["alt_fwd"]
    df["g_btc_avg"] = df["btc_pos_avg"] * df["btc_fwd"]
    # combined alt+btc TSMOM blend (equal weight of the two avg sleeves)
    df["g_altbtc_avg"] = 0.5 * df["g_alt_avg"] + 0.5 * df["g_btc_avg"]

    # cost: charge on position turnover (held HOLD bars => effective per-cycle turnover ~ |Δpos|/HOLD,
    # consistent with the book's held-book costing). Approximate sleeve turnover = |pos_t - pos_{t-1}|.
    def cost_net(posname, gname, cost):
        pos = df[posname].fillna(0.0).values
        turn = np.abs(np.diff(pos, prepend=pos[0]))        # position flip magnitude
        g = df[gname].fillna(0.0).values
        return g - (turn / HOLD) * 0.5 * cost              # symmetric round-trip approx, held HOLD

    for tag, posn, gn in [("alt_30d", "alt_pos_30d", "g_alt_30d"),
                          ("alt_90d", "alt_pos_90d", "g_alt_90d"),
                          ("alt_avg", "alt_pos_avg", "g_alt_avg"),
                          ("btc_30d", "btc_pos_30d", "g_btc_30d"),
                          ("btc_90d", "btc_pos_90d", "g_btc_90d"),
                          ("btc_avg", "btc_pos_avg", "g_btc_avg")]:
        df[f"pnl_trend_{tag}"] = cost_net(posn, gn, cost=COST_BPS * 1e-4)
    # altbtc blend cost: sum of the two sleeve costs
    df["pnl_trend_altbtc_avg"] = (df["pnl_trend_alt_avg"] + df["pnl_trend_btc_avg"]) / 2.0

    # attach book + regime/fold/episode tags
    df = df.reset_index().rename(columns={"index": "open_time"})
    for c in ["fold", "regime", "is_side", "flag", "alt30", "btc30", "pnl_base"]:
        df[c] = base[c].values
    return df


# --------------------------------------------------------------------------- episode helpers
def episode_mask(times, a, b):
    ti = pd.DatetimeIndex(times)
    return (ti >= pd.Timestamp(a, tz="UTC")) & (ti <= pd.Timestamp(b, tz="UTC"))


def in_any_episode(times, episodes):
    ti = pd.DatetimeIndex(times)
    m = np.zeros(len(ti), dtype=bool)
    for _, a, b in episodes:
        m |= ((ti >= pd.Timestamp(a, tz="UTC")) & (ti <= pd.Timestamp(b, tz="UTC")))
    return m


TREND_VARIANTS = ["alt_30d", "alt_90d", "alt_avg", "btc_30d", "btc_90d", "btc_avg", "altbtc_avg"]


# --------------------------------------------------------------------------- STEP 3 tests
def run_universe(label, rng, is_ext=False):
    print(f"\n{'='*78}\n=== {label} ===\n{'='*78}", flush=True)
    df = build_trend(label)
    times = df["open_time"]
    book = df["pnl_base"].values

    # ---- standalone trend-sleeve stats (the raw return stream) ----
    print(f"\n[trend-sleeve STANDALONE stats @ {COST_BPS}bps] (book ref: Sharpe {ann(book):+.2f} "
          f"maxDD {maxdd_of(book):+.0f} Calmar {calmar_of(book):+.2f} tot {(pd.Series(book)*1e4).sum():+.0f})",
          flush=True)
    print(f"  {'variant':>12}{'Sharpe':>8}{'maxDD':>9}{'Calmar':>8}{'totPnL':>9}{'%pos':>7}{'corr_book':>11}",
          flush=True)
    for v in TREND_VARIANTS:
        t = df[f"pnl_trend_{v}"].values
        s = stats(t)
        cc = np.corrcoef(pd.Series(book).fillna(0), pd.Series(t).fillna(0))[0, 1]
        print(f"  {v:>12}{s['sharpe']:>+8.2f}{s['maxDD']:>+9.0f}{s['calmar']:>+8.2f}"
              f"{s['totPnL']:>+9.0f}{s['pct_pos']:>7.1f}{cc:>+11.3f}", flush=True)

    # ---- TEST 1: crisis-alpha / negative correlation IN drawdowns ----
    print(f"\n[TEST 1: corr(trend, book) OVERALL vs WITHIN each DD episode] (negative-in-crisis = win)",
          flush=True)
    if is_ext:
        hdr = f"  {'variant':>12}{'overall':>9}"
        for ename, _, _ in EXT_EPISODES:
            hdr += f"{ename[:10]:>11}"
        print(hdr, flush=True)
        for v in TREND_VARIANTS:
            t = df[f"pnl_trend_{v}"].fillna(0).values
            bb = pd.Series(book).fillna(0).values
            row = f"  {v:>12}{np.corrcoef(bb, t)[0,1]:>+9.3f}"
            for ename, a, b in EXT_EPISODES:
                m = episode_mask(times, a, b)
                if m.sum() > 3 and pd.Series(t[m]).std() > 0 and pd.Series(bb[m]).std() > 0:
                    row += f"{np.corrcoef(bb[m], t[m])[0,1]:>+11.3f}"
                else:
                    row += f"{'na':>11}"
            print(row, flush=True)

        # per-episode trend PnL (does it MAKE money where the book bleeds?)
        print(f"\n[TEST 1b: per-episode PnL (bps) — book vs trend variants] @ {COST_BPS}bps", flush=True)
        hdr = f"  {'episode':>14}{'n':>5}{'book':>9}"
        for v in TREND_VARIANTS:
            hdr += f"{v[:9]:>10}"
        print(hdr, flush=True)
        for ename, a, b in EXT_EPISODES:
            m = episode_mask(times, a, b)
            if m.sum() < 5:
                print(f"  {ename:>14}{int(m.sum()):>5}  (too few)", flush=True); continue
            row = f"  {ename:>14}{int(m.sum()):>5}{(pd.Series(book)[m]*1e4).sum():>+9.0f}"
            for v in TREND_VARIANTS:
                row += f"{(df[f'pnl_trend_{v}'][m]*1e4).sum():>+10.0f}"
            print(row, flush=True)

    # ---- TEST 2: whipsaw cost in CALM (non-episode) periods ----
    if is_ext:
        print(f"\n[TEST 2: whipsaw cost — trend PnL in CALM (non-episode) periods] (insurance premium)",
              flush=True)
        calm = ~in_any_episode(times, EXT_EPISODES)
        print(f"  calm cycles: {calm.sum()} of {len(calm)} ({calm.mean()*100:.0f}%); "
              f"book calm totPnL {(pd.Series(book)[calm]*1e4).sum():+.0f}", flush=True)
        print(f"  {'variant':>12}{'calm_tot':>10}{'calm_Sharpe':>12}{'calm_%pos':>11}", flush=True)
        for v in TREND_VARIANTS:
            t = df[f"pnl_trend_{v}"][calm]
            print(f"  {v:>12}{(t*1e4).sum():>+10.0f}{ann(t):>+12.2f}{(t>0).mean()*100:>11.1f}", flush=True)

    # ---- TEST 3: combined portfolio book + w*trend at hedge weights ----
    print(f"\n[TEST 3: combined portfolio book + w*trend — maxDD/Calmar at hedge weights] @ {COST_BPS}bps",
          flush=True)
    # pick the two most defensible sleeves: alt_avg (the book's own complex) and altbtc_avg (diversified)
    for v in ["alt_avg", "altbtc_avg", "btc_avg"]:
        t = df[f"pnl_trend_{v}"].fillna(0).values
        print(f"  --- sleeve = {v} ---", flush=True)
        print(f"    {'w':>5}{'Sharpe':>8}{'maxDD':>9}{'Calmar':>8}{'totPnL':>9}{'ddVSbook':>10}", flush=True)
        base_dd = maxdd_of(book)
        for w in HEDGE_W:
            comb = book + w * t
            s = stats(comb)
            ddc = (s['maxDD'] - base_dd) / abs(base_dd) * 100 if base_dd < 0 else np.nan
            print(f"    {w:>5.2f}{s['sharpe']:>+8.2f}{s['maxDD']:>+9.0f}{s['calmar']:>+8.2f}"
                  f"{s['totPnL']:>+9.0f}{ddc:>+9.1f}%", flush=True)

    # ---- TEST 3b: episode-LOFO of the combined DD benefit ----
    if is_ext:
        print(f"\n[TEST 3b: episode-LOFO — does combined DD/Calmar benefit survive dropping any 1 episode?]",
              flush=True)
        for v in ["alt_avg", "altbtc_avg"]:
            t = df[f"pnl_trend_{v}"].fillna(0).values
            # use the best-Calmar hedge weight from a coarse grid (report it, but also show full curve robustness)
            w_best = max(HEDGE_W[1:], key=lambda w: (calmar_of(book + w * t) if np.isfinite(calmar_of(book + w * t)) else -9))
            comb = book + w_best * t
            full_b = calmar_of(book); full_c = calmar_of(comb)
            print(f"  sleeve {v} @ best w={w_best:.2f}: FULL book Calmar {full_b:+.2f} -> combined {full_c:+.2f} "
                  f"(lift {full_c-full_b:+.2f})", flush=True)
            allpos = True
            for ename, a, b in EXT_EPISODES:
                keep = ~episode_mask(times, a, b)
                cb = calmar_of(book[keep]); cc = calmar_of(comb[keep])
                lift = cc - cb if (np.isfinite(cc) and np.isfinite(cb)) else np.nan
                pos = np.isfinite(lift) and lift > 0
                allpos &= pos
                ddb = maxdd_of(book[keep]); ddc = maxdd_of(comb[keep])
                print(f"    drop {ename:<14} book Cal {cb:+.2f} comb {cc:+.2f} lift {lift:+.2f} "
                      f"| bookDD {ddb:+.0f} combDD {ddc:+.0f} {'>0' if pos else 'NEG'}", flush=True)
            print(f"    episode-LOFO combined-Calmar lift stays >0 dropping each: "
                  f"{'PASS' if allpos else 'FAIL'}", flush=True)

    # ---- TEST 4: honest framing — does trend profit in EACH alt-bear, or whipsaw? ----
    if is_ext:
        print(f"\n[TEST 4: honest framing — directional profit per episode (esp. 2025_q4 which bounced)]",
              flush=True)
        for v in ["alt_avg", "altbtc_avg"]:
            print(f"  sleeve {v}:", flush=True)
            for ename, a, b in EXT_EPISODES:
                m = episode_mask(times, a, b)
                if m.sum() < 5: continue
                t = df[f"pnl_trend_{v}"][m] * 1e4
                eq = t.cumsum(); mdd = (eq - eq.cummax()).min()
                # split: was the sleeve net long or net short during the episode?
                posn = "alt_pos_avg" if v.startswith("alt") else "btc_pos_avg"
                mean_pos = df[posn][m].mean()
                print(f"    {ename:<14} tot {t.sum():>+7.0f}  intra-maxDD {mdd:>+7.0f}  "
                      f"meanPos {mean_pos:>+.2f} ({'net-short' if mean_pos<0 else 'net-long'}) "
                      f"%pos {(t>0).mean()*100:>4.0f}", flush=True)

    # ---- G4-style: combined-portfolio DD vs random-sign trend placebo (is the SIGN load-bearing?) ----
    if is_ext:
        print(f"\n[G4-style placebo: is the trend SIGN load-bearing, or does any random-sign sleeve of "
              f"equal magnitude hedge as well?] ({N_PLACEBO} seeds)", flush=True)
        for v in ["alt_avg", "altbtc_avg"]:
            t = df[f"pnl_trend_{v}"].fillna(0).values
            w = 1.0
            comb_real = book + w * t
            real_dd = maxdd_of(comb_real); real_cal = calmar_of(comb_real)
            # placebo: keep the |sleeve return| magnitude, randomize the sign per cycle
            mag = np.abs(t)
            dds = np.empty(N_PLACEBO); cals = np.empty(N_PLACEBO)
            for i in range(N_PLACEBO):
                sgn = rng.choice([-1.0, 1.0], size=len(t))
                pc = book + w * (sgn * mag)
                dds[i] = maxdd_of(pc); cals[i] = calmar_of(pc)
            # better DD = less negative; rank = fraction of placebos WORSE (more negative) than real
            dd_rank = (dds < real_dd).mean() * 100      # % of placebos with worse (deeper) DD than real
            cal_rank = (cals < real_cal).mean() * 100
            print(f"  sleeve {v} @ w=1: real combined maxDD {real_dd:+.0f} Calmar {real_cal:+.2f} | "
                  f"random-sign placebo maxDD p50 {np.nanpercentile(dds,50):+.0f} "
                  f"p95(best) {np.nanpercentile(dds,95):+.0f}", flush=True)
            print(f"    real DD better than {dd_rank:.0f}% of random-sign placebos "
                  f"({'PASS>=p95' if dd_rank>=95 else 'weak/FAIL'}); "
                  f"Calmar better than {cal_rank:.0f}% "
                  f"({'PASS>=p95' if cal_rank>=95 else 'weak/FAIL'})", flush=True)

    # ---- G8 cost sensitivity of the combined portfolio (alt_avg @ w=1) ----
    print(f"\n[G8 cost: combined (alt_avg @ w=1) vs book across cost levels]", flush=True)
    df2 = build_trend(label)  # rebuild not needed; reuse gross to recost
    for cb in COSTS:
        # recost the sleeve at cost cb; book pnl_base already includes its own 4.5bps (constant ref)
        pos = df["alt_pos_avg"].fillna(0.0).values
        turn = np.abs(np.diff(pos, prepend=pos[0]))
        g = df["g_alt_avg"].fillna(0.0).values
        t_c = g - (turn / HOLD) * 0.5 * cb * 1e-4
        comb = book + 1.0 * t_c
        sb = stats(book); sc = stats(comb)
        print(f"    @{cb:>4.1f}bps book Cal {sb['calmar']:+.2f} DD {sb['maxDD']:+.0f} | "
              f"combined Cal {sc['calmar']:+.2f} DD {sc['maxDD']:+.0f}", flush=True)

    # per-cycle parquet
    keep_cols = (["open_time", "fold", "regime", "is_side", "flag", "alt30", "btc30", "pnl_base"]
                 + [f"pnl_trend_{v}" for v in TREND_VARIANTS]
                 + ["alt_pos_avg", "btc_pos_avg"])
    df[keep_cols].to_parquet(OUT / f"iter017_trend_sleeve_{label}.parquet", index=False)
    print(f"\n  per-cycle -> iter017_trend_sleeve_{label}.parquet", flush=True)


def main():
    t0 = time.time()
    rng = np.random.default_rng(SEED)
    print("=== iter-017: TREND-FOLLOWING / CRISIS-ALPHA HEDGE SLEEVE (scoping diagnostic) ===", flush=True)
    print(f"trend = TSMOM sign(trailing 30d & 90d) on eq-weight ALT-index AND BTC; held HOLD={HOLD}; "
          f"cost {COST_BPS}bps; combine book + w*trend.", flush=True)
    run_universe("EXT", rng, is_ext=True)
    run_universe("HL70", rng)
    run_universe("S44", rng)
    print(f"\nDone [{time.time()-t0:.0f}s]", flush=True)


if __name__ == "__main__":
    main()
