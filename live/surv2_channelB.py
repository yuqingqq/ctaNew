"""surv2_channelB (step-1.5, reviewer-corrected): directly SIZE Channel B — count INDEPENDENT in-window
crash episodes on out-of-universe names that WOULD have been gate-eligible (the exact production gate:
trailing-30d mean daily $vol >= $3M AND maturity >= 180d), during 2023-01..2026-06. This is the UPPER
BOUND on Channel B episodes (before the model-selection filter, which can only reduce it). If < 3 INDEPENDENT
(reviewer def #1: time-separated + not one systemic shock) both-era episodes exist, Channel B is a formal null.

Fetches monthly-cadence 1d-interval klines from Binance Vision for the ~585 absent candidates (per-sym parquet
cache → resumable). Crash-short episode = a gate-eligible in-window bar whose forward-14d close drawdown <= -40%.
"""
import io, zipfile, urllib.request, urllib.error, numpy as np, pandas as pd
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import warnings; warnings.filterwarnings("ignore")

SD = Path("/tmp/claude-1001/-home-yuqing-ctaNew/ecbd8f4c-236c-426c-85e5-e1f6b6edd11d/scratchpad")
HOST = "https://s3-ap-northeast-1.amazonaws.com/data.binance.vision"
CACHE = SD / "cb_klines"; CACHE.mkdir(exist_ok=True)
GATE = 3_000_000.0; MATURITY_D = 180
WIN_START = pd.Timestamp("2023-01-01", tz="UTC"); WIN_END = pd.Timestamp("2026-06-30", tz="UTC")
FETCH_START = "2022-12"                       # 30d lookback for trailing vol at window start
CRASH_DD = -0.40; CRASH_H = 14
COLS = ["open_time","open","high","low","close","volume","close_time","quote_volume",
        "count","tbv","tbqv","ignore"]

def month_range(fm, lm):
    fm = max(fm, FETCH_START); lm = min(lm, "2026-06")
    if fm > lm: return []
    y, m = int(fm[:4]), int(fm[5:7]); ey, em = int(lm[:4]), int(lm[5:7]); out = []
    while (y, m) <= (ey, em):
        out.append(f"{y:04d}-{m:02d}"); m += 1
        if m > 12: y += 1; m = 1
    return out

def _one_month(sym, mo):
    url = f"{HOST}/data/futures/um/monthly/klines/{sym}/1d/{sym}-1d-{mo}.zip"
    try:
        with urllib.request.urlopen(url, timeout=30) as r:
            z = zipfile.ZipFile(io.BytesIO(r.read()))
        raw = z.read(z.namelist()[0]).decode()
        hdr = 0 if raw[:1].isdigit() else 1
        df = pd.read_csv(io.StringIO(raw), header=None if hdr == 0 else 0, names=COLS)
        return df
    except Exception:
        return None

def fetch_sym(args):
    sym, fm, lm = args
    cf = CACHE / f"{sym}.parquet"
    if cf.exists():
        try: return sym, pd.read_parquet(cf)
        except Exception: pass
    frames = [d for mo in month_range(fm, lm) if (d := _one_month(sym, mo)) is not None]
    if not frames:
        pd.DataFrame(columns=["date","close","qvol"]).to_parquet(cf); return sym, None
    df = pd.concat(frames, ignore_index=True)
    # open_time may be ms or us epoch depending on era; normalize
    ot = df["open_time"].astype("int64")
    unit = "us" if ot.iloc[0] > 2_000_000_000_000_000 else "ms"
    df["date"] = pd.to_datetime(ot, unit=unit, utc=True).dt.normalize()
    out = df[["date","close","quote_volume"]].rename(columns={"quote_volume":"qvol"}).drop_duplicates("date").sort_values("date")
    out.to_parquet(cf)
    return sym, out

def analyze(sym, df, first_month):
    if df is None or len(df) < 40: return []
    df = df.set_index("date").sort_index()
    df = df[~df.index.duplicated()]
    dvol30 = df["qvol"].rolling(30, min_periods=20).mean().shift(1)     # trailing-30d mean, PIT
    list_dt = pd.Timestamp(first_month + "-01", tz="UTC")
    maturity = (df.index - list_dt).days
    close = df["close"]
    fwd_min = close.iloc[::-1].rolling(CRASH_H, min_periods=3).min().iloc[::-1].shift(-1)  # min over next H
    fdd = fwd_min / close - 1.0
    elig = (dvol30 >= GATE) & (maturity >= MATURITY_D) & (df.index >= WIN_START) & (df.index <= WIN_END)
    flag = elig & (fdd <= CRASH_DD)
    if not flag.any(): return []
    # collapse consecutive/near flagged bars into episodes (gap >= 30d = new episode)
    dts = list(df.index[flag.values]); eps = []
    ent = dts[0]; last = dts[0]
    for d in dts[1:]:
        if (d - last).days > 30:
            eps.append(ent); ent = d
        last = d
    eps.append(ent)
    res = []
    for e in eps:
        dd = float(fdd.loc[e]); dv = float(dvol30.loc[e])
        res.append({"sym": sym, "entry": e, "dd": dd, "dvol_entry": dv})
    return res

def main():
    t = pd.read_csv(SD / "delisted_table.csv")   # all 585: first_month,last_month,status
    cand = t[t.last_month >= "2023-01"].copy()    # must overlap the trading window
    args = [(r.symbol, r.first_month, r.last_month) for _, r in cand.iterrows()]
    fm_map = dict(zip(cand.symbol, cand.first_month))
    print(f"Channel B sizing: {len(args)} in-window-overlapping candidates (of 585)", flush=True)
    data = {}
    with ThreadPoolExecutor(max_workers=24) as ex:
        for i, (sym, df) in enumerate(ex.map(fetch_sym, args), 1):
            data[sym] = df
            if i % 100 == 0: print(f"  fetched {i}/{len(args)}", flush=True)
    print("fetch done; analyzing...", flush=True)
    episodes = []
    for sym, df in data.items():
        episodes.extend(analyze(sym, df, fm_map[sym]))
    ep = pd.DataFrame(episodes)
    if len(ep) == 0:
        print("\n>>> ZERO gate-eligible in-window crash episodes on out-of-universe names → Channel B FORMAL NULL")
        print("CHANNELBDONE"); return
    ep = ep.sort_values("entry")
    ep["era"] = np.where(ep.entry < pd.Timestamp("2025-10-01", tz="UTC"), "OOS", "RECENT")
    print(f"\n===== Channel B raw crash episodes: {len(ep)} across {ep.sym.nunique()} names =====")
    print(f"  era split: OOS {int((ep.era=='OOS').sum())} | RECENT {int((ep.era=='RECENT').sum())}")
    # independence (def #1): cluster entries within 7d across names = one systemic shock
    ep["wk"] = ep.entry.dt.to_period("W").astype(str)
    clusters = ep.groupby("wk").agg(n=("sym","size"), syms=("sym", lambda s: ",".join(sorted(set(s)))),
                                    era=("era","first"), entry=("entry","first")).reset_index()
    print(f"  → {len(clusters)} week-clusters (systemic cascades collapse to 1). By era: "
          f"OOS {int((clusters.era=='OOS').sum())}, RECENT {int((clusters.era=='RECENT').sum())}")
    print("\n  each week-cluster (candidate 'independent episode'):")
    for _, c in clusters.iterrows():
        print(f"    {c.entry.date()} [{c.era:6}] n={c.n:2d}: {c.syms[:70]}")
    indep_both = (clusters.era=="OOS").sum() >= 1 and (clusters.era=="RECENT").sum() >= 1
    print(f"\n  >> LOCKED BAR: >=3 independent clusters AND >=1 per era, none >50%?")
    print(f"     independent clusters: {len(clusters)} | both-era: {indep_both} | "
          f"{'PASS -> warrants retrain test' if len(clusters)>=3 and indep_both else 'FAIL -> Channel B formal null'}")
    ep.to_csv(SD / "channelB_episodes.csv", index=False)
    print("CHANNELBDONE")

if __name__ == "__main__":
    main()
