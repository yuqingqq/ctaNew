"""Phase A window×horizon IC-surface SCREEN (RESEARCH_LOOP_20260707 addenda 8 + 8b, PRE-REGISTERED).

SCREEN ONLY — non-verdict-bearing. For 21 feature windows × {5 alpha horizons, 2 marginal
(24h→h excess) labels, 5 raw-return robustness labels} × 2 eras, computes per-cycle XS:
  ic_raw   : feature ranks vs label ranks
  ic_pred  : feature ranks ⊥ base-book pred rank (1-D absorption diagnostic)
  ic_v0    : feature ranks ⊥ the FULL 16-rank V0-span (V0_LEAN ∪ resid_rev_2/3) — FLAG COLUMN
t/CI use horizon-length day blocks (ceil(h/24h) days: 1/1/1/2/3) per 8b-1.
Flag rule (8b): ic_v0 |t|≥3 BOTH eras same sign; h>24h additionally requires same-sign |t|≥2
both eras on the marginal label. Adjacency reported cosmetically only. Calibrated false-flag
rate ≈ 4e-4 over the 105 alpha cells. Flags feed ≤3 Phase B book cells (8b-3 sleeve baseline).

Conventions: features shift(1) at 5m, 4h cadence (X6b); resid_ret = incumbent β_288 idio sums
(C3: min_periods w/2); corr min_periods max(36, w/4) (C5); dd full-window (C4); ret full-window
(pct_change). Universe = clean-book rows (pred present ⇒ EXCL + liveness). Labels from panel
alpha_vs_btc_realized (alpha[t] = fwd close(t)→t+4h): rolling(k).sum().shift(-(k-1)), NaN'd
where open_time[t+k-1] − open_time[t] ≠ (k−1)·4h (8b-7). Family-common population masks (8b-5).

Output: live/IC_SURFACE_WINDOW_HORIZON.csv + printed flag list.
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import rankdata
import warnings; warnings.filterwarnings("ignore")
REPO = Path("/home/yuqing/ctaNew"); sys.path.insert(0, str(REPO))
import live.train_twobook_models as tt

KLINES = REPO / "data/ml/test/parquet/klines"
D = REPO / "live/state/convexity"
REC0 = pd.Timestamp("2025-10-04", tz="UTC")
OOS0, OOS1 = pd.Timestamp("2023-01-01", tz="UTC"), pd.Timestamp("2025-10-01", tz="UTC")
CYC = pd.Timedelta(hours=4)

# label spec: name -> (source_col, k_cycles, k_skip, block_days)
#   label = rolling(k - skip).sum() over cycles skip+1..k, shifted to row t; grid guard on full span
ALPHA = [("h4", 1, 0, 1), ("h12", 3, 0, 1), ("h24", 6, 0, 1), ("h48", 12, 0, 2), ("h72", 18, 0, 3)]
MARG = [("m48", 12, 6, 2), ("m72", 18, 6, 3)]
RAW = [("r4", 1, 0, 1), ("r12", 3, 0, 1), ("r24", 6, 0, 1), ("r48", 12, 0, 2), ("r72", 18, 0, 3)]
BLOCK = {n: b for n, _, _, b in ALPHA + MARG + RAW}
MARG_OF = {"h48": "m48", "h72": "m72"}

FAM = {"ret": ["ret_12h", "ret_24h", "ret_36h", "ret_3d_c", "ret_5d", "ret_7d"],
       "resid_ret": ["resid_ret_12h", "resid_ret_24h", "resid_ret_36h", "resid_ret_3d", "resid_ret_5d"],
       "resid_rev": ["resid_rev_2c", "resid_rev_3c", "resid_rev_6c", "resid_rev_12c"],
       "corr": ["corr_12h", "corr_1d", "corr_3d"],
       "dd": ["dd_1d", "dd_3d", "dd_7d"]}

def load_closes(sym):
    sd = KLINES / sym / "5m"
    if not sd.exists(): return None
    dfs = [pd.read_parquet(f, columns=["open_time", "close"]) for f in sorted(sd.glob("*.parquet"))]
    if not dfs: return None
    df = pd.concat(dfs, ignore_index=True).drop_duplicates("open_time").sort_values("open_time")
    df["open_time"] = pd.to_datetime(df["open_time"], utc=True)
    return df.set_index("open_time")["close"].astype(np.float64)

def build_features(sym, c, br):
    out = {}
    mr = np.log(c / c.shift(1))
    ci = mr.index.intersection(br.index)
    mr_a, br_a = mr.reindex(ci), br.reindex(ci)
    for name, w in [("ret_12h", 144), ("ret_24h", 288), ("ret_36h", 432),
                    ("ret_3d_c", 864), ("ret_5d", 1440), ("ret_7d", 2016)]:
        out[name] = c.pct_change(w).shift(1)                     # full window (pinned)
    cov = mr_a.rolling(288, min_periods=72).cov(br_a)
    var = br_a.rolling(288, min_periods=72).var()
    beta = (cov / var.replace(0, np.nan)).shift(1)
    idio = mr_a - beta * br_a
    for name, w in [("resid_ret_12h", 144), ("resid_ret_24h", 288), ("resid_ret_36h", 432),
                    ("resid_ret_3d", 864), ("resid_ret_5d", 1440)]:
        out[name] = idio.rolling(w, min_periods=w // 2).sum().shift(1)   # C3: w/2 (pinned)
    for name, w in [("corr_12h", 144), ("corr_1d", 288), ("corr_3d", 864)]:
        out[name] = mr_a.rolling(w, min_periods=max(36, w // 4)).corr(br_a).shift(1)  # C5 (pinned)
    for name, w in [("dd_1d", 288), ("dd_3d", 864), ("dd_7d", 2016)]:
        out[name] = c / c.rolling(w).max() - 1                   # C4 basis: full window (pinned)
    f = pd.DataFrame(out).astype(np.float32)
    f = f[(f.index.hour % 4 == 0) & (f.index.minute == 0)]
    f["symbol"] = sym
    return f.reset_index().rename(columns={"index": "open_time"})

def block_t(x, days, block_days):
    """block t: per-block means (block = block_days calendar days), t on the block series (8b-1)."""
    b = (days.astype("int64") // block_days)
    bm = pd.Series(x.values, index=b.values).groupby(level=0).mean()
    if len(bm) < 20 or bm.std(ddof=1) == 0: return np.nan, np.nan
    return float(bm.mean()), float(bm.mean() / (bm.std(ddof=1) / np.sqrt(len(bm))))

def main():
    print("panel + preds...", flush=True)
    V0 = list(tt.V0_LEAN)
    PAN = pd.read_parquet(tt.PANEL, columns=["symbol", "open_time", "return_pct",
                                             "alpha_vs_btc_realized"] + V0)
    PAN["open_time"] = pd.to_datetime(PAN["open_time"], utc=True)
    PAN = PAN[(PAN.open_time.dt.hour % 4 == 0) & (PAN.open_time.dt.minute == 0)]
    PAN = PAN.sort_values(["symbol", "open_time"]).reset_index(drop=True)
    g = PAN.groupby("symbol")
    ga = g["alpha_vs_btc_realized"]
    for k in (2, 3, 6, 12):  # resid_rev candidates; _2c/_3c == deployed long-book features
        PAN[f"resid_rev_{k}c"] = -ga.transform(lambda s, k=k: s.shift(1).rolling(k).sum())
    for src, spec in (("alpha_vs_btc_realized", ALPHA + MARG), ("return_pct", RAW)):
        gs = g[src]
        for name, k, skip, _ in spec:
            lab = gs.transform(lambda s, k=k, sk=skip: s.rolling(k - sk).sum().shift(-(k - 1)))
            if k > 1:  # grid guard over the FULL span (8b-7)
                ok = (g["open_time"].shift(-(k - 1)) - PAN["open_time"]) == (k - 1) * CYC
                lab = lab.where(ok)
            PAN[f"lab_{name}"] = lab.astype(np.float32)
    preds = []
    for b in ("hl_tgt_res_base_clean", "hl_v4base_oos_clean"):
        p = pd.read_parquet(D / b / "v0full_hl60.parquet", columns=["symbol", "open_time", "pred"])
        p["open_time"] = pd.to_datetime(p["open_time"], utc=True)
        preds.append(p)
    PRED = pd.concat(preds, ignore_index=True).drop_duplicates(["symbol", "open_time"])
    PAN = PAN.merge(PRED, on=["symbol", "open_time"], how="left")
    PAN = PAN[PAN["pred"].notna()]                      # clean-book universe (8b-6)

    print("features (per symbol)...", flush=True)
    btc = load_closes("BTCUSDT"); br = np.log(btc / btc.shift(1))
    parts = []
    syms = sorted(PAN["symbol"].unique())
    for i, sym in enumerate(syms):
        c = load_closes(sym)
        if c is None: continue
        parts.append(build_features(sym, c, br))
        if (i + 1) % 40 == 0: print(f"  {i+1}/{len(syms)}", flush=True)
    F = pd.concat(parts, ignore_index=True)
    F["open_time"] = pd.to_datetime(F["open_time"], utc=True)
    PAN = PAN.merge(F, on=["symbol", "open_time"], how="left")

    V0SPAN = V0 + ["resid_rev_2c", "resid_rev_3c"]       # 16-rank span (8b-4)
    lab_cols = [f"lab_{n}" for n in BLOCK]
    feat_all = [f for cols in FAM.values() for f in cols]

    print("per-cycle ICs...", flush=True)
    acc = {}   # (feature, label) -> list[(t, n, ic_raw, ic_pred, ic_v0)]
    eras = {"rec": PAN.open_time >= REC0,
            "oos": (PAN.open_time >= OOS0) & (PAN.open_time < OOS1)}
    res = {}
    for era, mask in eras.items():
        acc.clear()
        E = PAN[mask]
        nc = 0
        for t, gg in E.groupby("open_time"):
            base_ok = gg[V0SPAN].notna().all(axis=1).to_numpy()
            if base_ok.sum() < 10: continue
            sub = gg[base_ok]
            nc += 1
            for fam, cols in FAM.items():
                fok = sub[cols].notna().all(axis=1).to_numpy()   # family-common mask (8b-5)
                if fok.sum() < 10: continue
                S = sub[fok]
                n = len(S)
                X = rankdata(S[V0SPAN].to_numpy(), axis=0).astype(np.float64)
                Y = rankdata(S[cols].to_numpy(), axis=0).astype(np.float64)
                p = rankdata(S["pred"].to_numpy()).astype(np.float64)
                X -= X.mean(0); Y -= Y.mean(0); p -= p.mean()
                # residual-norm guard (17-fix-2): skip degenerate cycles (near-zero span/pred norm)
                if np.linalg.norm(X) < 1e-9 or (p @ p) < 1e-9: continue
                B, *_ = np.linalg.lstsq(X, Y, rcond=None)
                Rv = Y - X @ B
                Rp = Y - np.outer(p, (p @ Y) / (p @ p))
                for lc in lab_cols:
                    lv = S[lc].to_numpy(dtype=np.float64)
                    lok = ~np.isnan(lv)
                    if lok.sum() < 10: continue
                    l = rankdata(lv[lok]); l -= l.mean()
                    ln = np.sqrt((l * l).sum())
                    def ics(M):
                        Ms = M[lok] - M[lok].mean(0)
                        nn = np.sqrt((Ms * Ms).sum(0)) * ln
                        return (Ms.T @ l) / np.where(nn > 0, nn, np.nan)
                    ir, ip, iv = ics(Y), ics(Rp), ics(Rv)
                    for j, fc in enumerate(cols):
                        acc.setdefault((fc, lc), []).append((t, int(lok.sum()), ir[j], ip[j], iv[j]))
        rows = []
        for (fc, lc), v in acc.items():
            df = pd.DataFrame(v, columns=["t", "n", "raw", "pred", "v0"]).dropna(subset=["raw"])
            if df.empty: continue
            bd = BLOCK[lc.replace("lab_", "")]
            days = (df["t"].dt.tz_convert("UTC").astype("int64") // 86_400_000_000_000)
            m_r, t_r = block_t(df["raw"], days, bd)
            m_p, t_p = block_t(df["pred"], days, bd)
            m_v, t_v = block_t(df["v0"], days, bd)
            rows.append((era, fc, lc.replace("lab_", ""), len(df), int(df["n"].median()), bd,
                         m_r, t_r, m_p, t_p, m_v, t_v))
        res[era] = rows
        print(f"  {era}: {nc} cycles, {len(rows)} cells", flush=True)

    R = pd.DataFrame([r for era in res for r in res[era]],
                     columns=["era", "feature", "label", "n_cyc", "med_n_sym", "block_days",
                              "ic_raw", "t_raw", "ic_pred_orth", "t_pred_orth",
                              "ic_v0_orth", "t_v0_orth"])
    R.to_csv(REPO / "live/IC_SURFACE_WINDOW_HORIZON.csv", index=False)
    print(f"\nwrote live/IC_SURFACE_WINDOW_HORIZON.csv ({len(R)} rows)")

    # flag rule (8b): v0-orth |t|>=3 both eras same sign; h>24h also marginal |t|>=2 both eras same sign
    def cell(fc, lc, col):
        r = R[(R.feature == fc) & (R.label == lc)]
        d = {e: (r[r.era == e][col].iloc[0] if len(r[r.era == e]) else np.nan) for e in ("rec", "oos")}
        return d["rec"], d["oos"]
    alpha_labels = [n for n, *_ in ALPHA]
    flags = []
    for fc in feat_all:
        for lc in alpha_labels:
            tr, to = cell(fc, lc, "t_v0_orth")
            if np.isnan(tr) or np.isnan(to): continue
            if not (abs(tr) >= 3 and abs(to) >= 3 and np.sign(tr) == np.sign(to)): continue
            note = ""
            if lc in MARG_OF:
                mr_, mo_ = cell(fc, MARG_OF[lc], "t_v0_orth")
                if not (abs(mr_) >= 2 and abs(mo_) >= 2 and np.sign(mr_) == np.sign(mo_)
                        and np.sign(mr_) == np.sign(tr)):
                    note = f"  [FAILS marginal test: m t_rec {mr_:+.1f} t_oos {mo_:+.1f}]"
                else:
                    note = f"  [marginal OK: {mr_:+.1f}/{mo_:+.1f}]"
            flags.append((fc, lc, tr, to, note))
    print("\nFLAGS (v0-span-orth |t|>=3 both eras same sign; calibrated null rate ~4e-4/105 cells;"
          "\n       h>24h requires the 24h->h marginal-label test; adjacency cosmetic only):")
    real = [f for f in flags if "FAILS" not in f[4]]
    for fc, lc, tr, to, note in sorted(flags, key=lambda x: -min(abs(x[2]), abs(x[3]))):
        print(f"  {fc:16s} x {lc:4s}  t_rec {tr:+.1f}  t_oos {to:+.1f}{note}")
    if not flags: print("  none")
    print(f"\n{len(real)} flag(s) pass all tests -> Phase B budget is <=3 book cells (8b-3 baseline)."
          f" Screen output only — never evidence.")
    print("SURFACEDONE")

if __name__ == "__main__":
    main()
