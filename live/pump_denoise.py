"""De-noise the weak positioning-tilted froth-dump short and RE-JUDGE on the spec curve (median net + %specs clearing
CI>0 + avg n/fold), NOT one number — because filtering is where the +9% overfit last time. Covered-train expanding
walk-forward, week-clustered CI. De-noisers, targeting what the noise IS (the +188% squeeze tail + self-defeating
funding):
  LEAN   : drop funding features (funding is a -2.9% drag / self-defeating), keep price-action + positioning. no n cost
  CLS    : select by classification P(dump=fwd_ret<=-20%) not regression on the heavy-tailed return.          no n cost
  SGUARD : veto short names whose crowd L/S is bottom-quartile among candidates (crowded shorts = squeeze fuel). n cost
  CONF   : abstain on names where the 3-seed ensemble disagrees (pred std above basket median).                n cost
A de-noiser EARNS its place only if it lifts %CI>0 (currently 0/curve) or the median WITHOUT collapsing n.
"""
from pathlib import Path
import numpy as np, pandas as pd
import warnings; warnings.filterwarnings("ignore")
from lightgbm import LGBMRegressor, LGBMClassifier
KD = Path("/home/yuqing/ctaNew/data/ml/cache")
SD = Path("/tmp/claude-1001/-home-yuqing-ctaNew/ecbd8f4c-236c-426c-85e5-e1f6b6edd11d/scratchpad")
N_FUND = 21; COST = 0.0040; SEEDS = 3; DUMP = -0.20
PRICE = ["climax", "climax_build", "runup_3d", "runup_1d", "parab", "rvol_7d", "dist_ath", "taker", "age_d"]
FUND = ["funding", "funding_chg", "funding_z"]; POS = ["oi_chg", "tt_ls", "ls", "taker_ls"]
MCOLS = {"sum_open_interest": "oi", "sum_toptrader_long_short_ratio": "tt_ls",
         "count_long_short_ratio": "ls", "sum_taker_long_short_vol_ratio": "taker_ls"}
_kw = dict(n_estimators=250, num_leaves=7, learning_rate=0.03, min_child_samples=30, subsample=0.8,
           colsample_bytree=0.7, reg_lambda=5.0, verbose=-1)
def mkr(s): return LGBMRegressor(random_state=s, **_kw)
def mkc(s): return LGBMClassifier(random_state=s, **_kw)

def load_metrics(sym):
    p = KD / f"metrics_{sym}.parquet"
    if not p.exists(): return None
    d = pd.read_parquet(p)
    if not isinstance(d.index, pd.DatetimeIndex):
        tc = "create_time" if "create_time" in d.columns else "calc_time"
        d = d.set_index(pd.to_datetime(d[tc], utc=True))
    return d[~d.index.duplicated()].sort_index()

def asof(s, times, side):
    idx = s.index.values.astype("datetime64[ns]"); tt = pd.to_datetime(times, utc=True).values.astype("datetime64[ns]")
    pos = np.searchsorted(idx, tt, side=side) - 1
    return np.where(pos >= 0, s.values[np.clip(pos, 0, len(s) - 1)], np.nan)

def rederive(e, side):
    e = e.copy()
    for c in ["oi_chg", "tt_ls", "ls", "taker_ls"]: e[c] = np.nan
    for sym, g in e.groupby("sym"):
        d = load_metrics(sym)
        if d is None: continue
        gi = g.index
        for col, oc in MCOLS.items():
            if col not in d.columns: continue
            s = d[col].dropna()
            if not len(s): continue
            if col == "sum_open_interest":
                e.loc[gi, "oi_chg"] = asof(s, g["t"], side) / asof(s, g["t"] - pd.Timedelta(days=3), side) - 1
            else:
                e.loc[gi, oc] = asof(s, g["t"], side)
    return e

def wk_boot(t, x, rng):
    x = np.asarray(x, float); t = pd.to_datetime(np.asarray(t), utc=True)
    wk = pd.Series(t).dt.to_period("W").astype(str).values
    grps = [x[wk == w] for w in pd.unique(wk)]
    if len(grps) < 4: return (np.nan, np.nan)
    out = [np.concatenate([grps[i] for i in rng.integers(0, len(grps), len(grps))]).mean() for _ in range(2500)]
    return tuple(np.percentile(out, [2.5, 97.5]))

def walk(pan, cov, cfg, base_q, seed, start, step_days):
    lean, cls, sguard, conf = cfg["lean"], cfg["cls"], cfg["sguard"], cfg["conf"]
    feats = (PRICE + POS) if lean else (PRICE + FUND + POS)
    p = pan.sort_values("t").reset_index(drop=True)
    cur = pd.Timestamp(start, tz="UTC"); st = pd.Timedelta(days=step_days); rows = []
    while cur <= p["t"].max():
        tr = p[(p.t < cur) & (p["rid"].isin(cov))]
        te = p[(p.t >= cur) & (p.t < cur + st) & (p["rid"].isin(cov))]
        if len(tr) >= 250 and len(te) >= 10:
            med = tr[feats].median()
            Xtr, Xte = tr[feats].fillna(med), te[feats].fillna(med)
            if cls:
                y = (tr["fwd_ret"].values <= DUMP).astype(int)
                Ps = np.array([mkc(seed + k).fit(Xtr, y).predict_proba(Xte)[:, 1] for k in range(SEEDS)])
            else:
                y = tr["fwd_ret"].clip(-0.9, 2.0).values
                Ps = np.array([-mkr(seed + k).fit(Xtr, y).predict(Xte) for k in range(SEEDS)])  # dump_score = -pred_ret
            te = te.copy(); te["score"] = Ps.mean(0); te["pstd"] = Ps.std(0)
            te["ct"] = pd.qcut(te["score"].rank(method="first"), base_q, labels=False, duplicates="drop")
            b = te[te.ct == te.ct.max()].copy()                       # highest dump_score bucket
            if sguard and b["ls"].notna().any():
                thr = te["ls"].quantile(0.25); b = b[(b["ls"] >= thr) | b["ls"].isna()]  # drop crowded-short
            if conf and len(b) > 2:
                b = b[b["pstd"] <= b["pstd"].median()]                # keep agreeing names
            if len(b):
                b["net"] = -b["fwd_ret"] + b["funding"] * N_FUND - COST; rows.append(b)
        cur = cur + st
    if not rows: return None
    S = pd.concat(rows); rng = np.random.default_rng(seed)
    lo, up = wk_boot(S["t"], S["net"].values, rng)
    return len(S), len(rows), S["net"].mean(), np.median(S["net"]), lo, up

SPECS = [(sd, start, step) for sd in ["right", "left"] for start in ["2024-11-20", "2024-12-20"] for step in [35, 42, 49]]

def curve(pan, cov, cfg, base_q):
    ms, cis, ns = [], 0, []
    for sd, start, step in SPECS:
        r = walk(pan[sd], cov[sd], cfg, base_q, 0, start, step)
        if r is None: continue
        n, nf, m, md, lo, up = r; ms.append(m); ns.append(n / max(nf, 1)); cis += int(lo > 0)
    ms = np.array(ms)
    return ms.mean(), np.median(ms), (ms > 0).mean(), cis / len(ms), np.mean(ns), len(ms)

def main():
    e0 = pd.read_csv(SD / "pump_enriched.csv"); e0["t"] = pd.to_datetime(e0["t"], utc=True)
    e0 = e0.dropna(subset=["fwd_ret", "funding"]).reset_index(drop=True); e0["rid"] = e0.index
    pan = {sd: rederive(e0, sd) for sd in ["right", "left"]}
    for sd in pan: pan[sd]["rid"] = pan[sd].index
    cov = {sd: set(pan[sd].loc[pan[sd]["tt_ls"].notna(), "rid"]) for sd in pan}
    print(f"covered rows: {len(cov['right'])} | spec curve = {len(SPECS)} configs each\n")
    F = lambda lean=0, cls=0, sguard=0, conf=0: dict(lean=lean, cls=cls, sguard=sguard, conf=conf)
    combos = [
        ("BASELINE tercile      ", F(), 3), ("BASELINE decile       ", F(), 10),
        ("LEAN (no funding)     ", F(lean=1), 3), ("CLS (P(dump))         ", F(cls=1), 3),
        ("LEAN+CLS              ", F(lean=1, cls=1), 3), ("LEAN+CLS +SGUARD      ", F(lean=1, cls=1, sguard=1), 3),
        ("LEAN+CLS +CONF        ", F(lean=1, cls=1, conf=1), 3), ("LEAN+CLS +SGUARD+CONF ", F(lean=1, cls=1, sguard=1, conf=1), 3),
    ]
    print(f"{'de-noiser':24s} base | net-mean net-med  %pos  %CI>0  avg-n/fold")
    for name, cfg, q in combos:
        mean, med, pos, ci, navg, nspec = curve(pan, cov, cfg, q)
        print(f"{name:24s} q{q:<3d}| {mean*100:+5.1f}%  {med*100:+5.1f}%  {pos*100:3.0f}%  {ci*100:4.0f}%   {navg:.1f}")
    print("\nread: a de-noiser earns it only if %CI>0 lifts off 0 (or median rises a lot) without avg-n collapsing. DENOISEDONE")

if __name__ == "__main__":
    main()
