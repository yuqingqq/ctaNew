"""convexity_portable paper-bot — Stage A (replay) + skeleton for live.

Strategy (frozen baseline + iter-012 stop):
  - 4h decision grid; predictions from per-sym Ridge (artifact: live/models/convexity_portable.pkl).
  - BTC 30d return classifies regime: bull (>+0.10), bear (<-0.10), else side.
  - BULL  -> trend-follow on mom_30d (cross-section: long top-K=5, short bot-K=5)
  - SIDE  -> mean-rev on pred with beta-neutral leg sizing
  - BEAR  -> flat (no positions)
  - 6 overlapping sleeves, 24h hold, equal-weight 1/K per leg, sum/HOLD across active sleeves.
  - Cost 4.5bps/leg; turn*0.5*COST per cycle (matches X116 engine).
  - iter-012 vol-norm reactive equity-DD stop overlay: if drawdown >= k=2.0 * sigma(trailing-180-bar
    equity increments) * sqrt(180), de-gross to g_floor=0.40 until 50%-heal or 90-bar timeout
    (eq>trough guard). Warmup 60 bars. PIT (uses equity through t-1).

Modes:
  python -m live.convexity_paper_bot --replay 60     # replay last 60 days from x132 preds
  python -m live.convexity_paper_bot --replay-all    # replay full preds history (for validation)
  python -m live.convexity_paper_bot --cycle         # one live cycle (Stage B, not wired yet)
  python -m live.convexity_paper_bot --check-state   # print active sleeves + equity

State (live/state/convexity/): cycles.csv, sleeves.csv, trades.csv, predictions.parquet,
equity.csv, regime.csv, universe.csv, positions.json. See SCHEMA.md.
"""
from __future__ import annotations
import argparse, gc, hashlib, json, logging, os, pickle, sys, time, warnings
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")

REPO = Path(__file__).resolve().parent.parent
# STATE is overridable so the two-book forward test can run each book into its own dir
# (BookA flow + BookB price) without clobbering. Default = the original single-book dir.
STATE = Path(os.environ.get("CONVEXITY_STATE", str(REPO/"live/state/convexity"))); STATE.mkdir(parents=True, exist_ok=True)
ARTIFACT = REPO/"live/models/convexity_portable.pkl"
PANEL    = REPO/"outputs/vBTC_features/panel_expanded_v0.parquet"
DEFAULT_PREDS = REPO/"research/convexity_portable_2026-05-20/results/_cache/x132_expanded_v0_preds.parquet"
PREDS    = Path(os.environ.get("CONVEXITY_PREDS_PATH", str(DEFAULT_PREDS)))
# Universe-meta source is ALWAYS the full preds file (for maturity calc) — the override
# preds file may be a narrow window (e.g. H2 only) which would falsely fail maturity.
UNIVERSE_META_PREDS = DEFAULT_PREDS
KLINES   = REPO/"data/ml/test/parquet/klines"
RESEARCH_LOG = STATE/"replay_vs_research.csv"

# Strategy constants (FROZEN per shared/baseline.md)
K = int(os.environ.get("STRAT_K", "5"))
# asymmetric K (default mode): separate long/short basket sizes (default = K). xs_z shorts informative.
K_LONG = int(os.environ.get("STRAT_K_LONG", str(K)))
K_SHORT = int(os.environ.get("STRAT_K_SHORT", str(K)))
HOLD = int(os.environ.get("STRAT_HOLD", "6"))
# COST per leg in fraction (4.5e-4 = 4.5 bps). Env override COST_BPS_LEG (in bps).
COST = float(os.environ["COST_BPS_LEG"])*1e-4 if "COST_BPS_LEG" in os.environ else 4.5e-4
REGIME_BULL_THR = float(os.environ.get("REGIME_BULL_THR", "0.10"))
REGIME_BEAR_THR = float(os.environ.get("REGIME_BEAR_THR", "-0.10"))
# Hysteresis: require N consecutive raw-bull cycles before SWITCHING IN to bull (and same
# for bear). Reduces boundary-flip whipsaw. Empirically (OOS 2025-10→2026-05): 9 of 14
# bull episodes lasted <1d at Sharpe −18.86 due to boundary chatter.
REGIME_HYSTERESIS_N = int(os.environ.get("REGIME_HYSTERESIS_N", "3"))
# BULL_MODE: which construction the strategy uses in BULL regime.
#   "mom"          : mom_30d trend-follow, equal-weight legs (KEEPS +0.286 net beta) — BASELINE
#   "betaneut_mom" : mom_30d trend-follow, beta-neutral leg sizing (kills the long-beta leak)
#   "sidealpha"    : V0 pred mean-rev with beta-neutral (treat bull EXACTLY like side)
BULL_MODE = os.environ.get("BULL_MODE", "mom")
# SIDE_MODE: which construction in SIDE regime.
#   "default"          : current top-K=5 long / bot-K=5 short, beta-neutral
#   "short_btc_hedge"  : drop the (broken) long leg; trade only bot-K=3 alt shorts + BTC long
#                        sized to neutralize basket beta (BTC β=1; w_btc = avg(β_short))
SIDE_MODE = os.environ.get("SIDE_MODE", "default")
SIDE_SHORT_K = int(os.environ.get("SIDE_SHORT_K", "3"))   # used by short_btc_hedge
# long_defensive_basket_hedge: stage-1 top-N pred candidates, then pick K most-defensive
# (high corr_to_btc, low rvol, low atr) — parameter-free rank composite (iter-038/039).
SIDE_DEF_N = int(os.environ.get("SIDE_DEF_N", "12"))
DEF_FEATS = ["corr_to_btc_1d", "rvol_7d", "atr_pct"]   # defensive-tilt features (PIT, in panel)
# regime_switch: pred_disp (model conviction) threshold; >=THR -> model-long, else defensive-long.
SIDE_SWITCH_THR = float(os.environ.get("SIDE_SWITCH_THR", "1.0"))
BTC_HEDGE_KEY = "_BTC_HEDGE_"   # sentinel key in weight dict for the synthetic BTC long
# V7: per-name confidence threshold gate (only trade alt when |pred| > threshold;
# else fallback to BTC). Applies in SIDE_MODE=confidence_btc_hedge.
PRED_THRESHOLD = float(os.environ.get("PRED_THRESHOLD", "0.5"))   # abs value in pred z-units
# pred-disp adaptive gate: if per-cycle pred_disp is below the trailing-N percentile,
# the model is producing flat predictions = no tail-edge = go FLAT in side regime.
# (bull keeps trading regardless — mom_30d signal isn't affected by pred dispersion.)
DISP_GATE = os.environ.get("DISP_GATE", "0") == "1"
DISP_GATE_PCTILE = float(os.environ.get("DISP_GATE_PCTILE", "0.30"))
DISP_GATE_LOOKBACK = int(os.environ.get("DISP_GATE_LOOKBACK", "252"))
DISP_GATE_MIN_HISTORY = int(os.environ.get("DISP_GATE_MIN_HISTORY", "60"))
# iter-012 vol-norm stop overlay
STOP_K_SIGMA = float(os.environ.get("STOP_K_SIGMA", "2.0"))
STOP_G_FLOOR = float(os.environ.get("STOP_G_FLOOR", "0.40"))
STOP_SIGMA_WINDOW = int(os.environ.get("STOP_SIGMA_WINDOW", "180"))
STOP_WARMUP = 60
STOP_HEAL_FRAC = 0.5
STOP_TIMEOUT_BARS = 90
# Universe filter (deploy-spec: maturity≥180d + hygiene + liquidity_floor + dedup; iter-036)
MIN_HISTORY_DAYS = 180
HYGIENE_EXCLUDE = {"USDCUSDT","BUSDUSDT","TUSDUSDT","DAIUSDT","FDUSDUSDT",   # stables
                   "PAXGUSDT","XAUTUSDT",                                     # non-crypto-beta
                   "WBTCUSDT","WBETHUSDT","WSTETHUSDT","WETHUSDT","STETHUSDT"} # wrapped
# Optional ALLOW-list restriction: comma-sep symbols. Useful for per-sym signal-decay tests.
SYM_ALLOWLIST = set(os.environ["SYM_ALLOWLIST"].split(",")) if "SYM_ALLOWLIST" in os.environ else None
# Dynamic per-cycle allowlist (e.g. rolling per-sym IC filter). Parquet with cols
# (open_time, symbol) — presence = allowed at that cycle.
DYN_ALLOW_PATH = os.environ.get("CONVEXITY_DYNAMIC_ALLOWLIST_PATH")
_DYN_ALLOW: dict = {}
if DYN_ALLOW_PATH and Path(DYN_ALLOW_PATH).exists():
    _dyn = pd.read_parquet(DYN_ALLOW_PATH)
    _dyn["open_time"] = pd.to_datetime(_dyn["open_time"], utc=True)
    for _ot, _g in _dyn.groupby("open_time"):
        _DYN_ALLOW[_ot] = set(_g["symbol"])
LIQ_FLOOR_DOLLAR_VOL_30D = 3_000_000.0     # $3M/day exec floor
DEDUP_CORR_THRESHOLD = 0.90                # drop high-corr names; keep longer-history one

INITIAL_EQUITY = 10_000.0

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("convexity_paper_bot")


# =====================================================================
# Data loaders
# =====================================================================

def load_artifact() -> dict:
    with open(ARTIFACT, "rb") as f: return pickle.load(f)


def load_preds(start: pd.Timestamp | None = None) -> pd.DataFrame:
    """Walk-forward preds from research (used by replay mode)."""
    cols = ["symbol","open_time","alpha_A","return_pct","exit_time","pred","fold"]
    d = pd.read_parquet(PREDS, columns=cols)
    d["open_time"] = pd.to_datetime(d["open_time"], utc=True)
    d["exit_time"] = pd.to_datetime(d["exit_time"], utc=True)
    d = d[(d["open_time"].dt.hour%4==0) & (d["open_time"].dt.minute==0)].copy()
    if start is not None: d = d[d["open_time"] >= start]
    return d.sort_values(["open_time","symbol"]).reset_index(drop=True)


def load_close_4h(sym: str, last_n_files: int | None = None) -> pd.Series:
    sd = KLINES/sym/"5m"
    if not sd.exists(): return pd.Series(dtype="float64")
    files = sorted(sd.glob("*.parquet"))
    if last_n_files is not None: files = files[-last_n_files:]   # windowed load (live --cycle only needs trailing)
    dfs = [pd.read_parquet(f, columns=["open_time","close"]) for f in files]
    if not dfs: return pd.Series(dtype="float64")
    df = pd.concat(dfs, ignore_index=True).drop_duplicates("open_time").sort_values("open_time")
    df["open_time"] = pd.to_datetime(df["open_time"], utc=True)
    c = df.set_index("open_time")["close"].astype(np.float64)
    return c[(c.index.hour%4==0) & (c.index.minute==0)]


def compute_mom30_and_beta(syms: list[str], lookback_days: int | None = None) -> tuple[pd.DataFrame, dict[str, pd.Series]]:
    """mom_30d and beta-to-BTC per sym, 4h grid, PIT (.shift(1)).
    lookback_days: if set (live --cycle), load only the last ~lookback_days of klines per sym — mom30/beta
    need only ~30d (180 4h-bars) trailing, so full-history loads are wasted. None = full (replay/bootstrap)."""
    nfiles = (lookback_days + 5) if lookback_days else None   # daily 5m files ≈ days
    btc = load_close_4h("BTCUSDT", nfiles); br = np.log(btc/btc.shift(1)); bvar = br.rolling(180, min_periods=42).var()
    mom_rows, betas = [], {}
    for sym in syms:
        c = load_close_4h(sym, nfiles)
        if c.empty: continue
        mom_rows.append(pd.DataFrame({"symbol": sym, "open_time": c.index,
                                       "mom30": (c/c.shift(180)-1).shift(1).values}))
        r = np.log(c/c.shift(1)); ri, bi = r.align(br, join="inner")
        cov = ri.rolling(180, min_periods=42).cov(bi); var = bvar.reindex(ri.index).replace(0, np.nan)
        betas[sym] = (cov/var).shift(1)
    mom = pd.concat(mom_rows, ignore_index=True) if mom_rows else pd.DataFrame()
    mom["open_time"] = pd.to_datetime(mom["open_time"], utc=True) if not mom.empty else mom
    return mom, betas


def compute_btc_30d() -> pd.Series:
    btc = load_close_4h("BTCUSDT")
    return (btc/btc.shift(180)-1).rename("btc_ret_30d")


# =====================================================================
# Universe filter (per iter-036 deploy spec) — precomputed from panel
# =====================================================================

def precompute_universe_meta(panel_df: pd.DataFrame | None = None) -> dict[str, dict]:
    """Per-sym {earliest_4h_bar, latest_4h_bar, n_4h_bars}. The panel itself encodes
    effective listing+history via row coverage — BUT a sliced replay panel doesn't,
    so we always read the FULL preds file for maturity, regardless of replay window."""
    full = pd.read_parquet(UNIVERSE_META_PREDS, columns=["symbol","open_time"])
    full["open_time"] = pd.to_datetime(full["open_time"], utc=True)
    full = full[(full["open_time"].dt.hour%4==0) & (full["open_time"].dt.minute==0)]
    meta = {}
    for sym, g in full.groupby("symbol"):
        ot = g["open_time"]
        meta[sym] = dict(earliest=ot.min(), latest=ot.max(), n=len(g))
    return meta


def eligible_universe_at(univ_meta: dict, asof: pd.Timestamp, dvol_cache: dict) -> dict[str, dict]:
    """Cheap per-cycle filter using precomputed meta."""
    out = {}
    for sym, m in univ_meta.items():
        rec = {"in_universe": False, "reason": "", "trailing_days": 0,
               "dvol30": dvol_cache.get(sym, np.nan)}
        if SYM_ALLOWLIST is not None and sym not in SYM_ALLOWLIST:
            rec["reason"] = "allowlist"; out[sym] = rec; continue
        # dynamic per-cycle allowlist takes precedence
        if _DYN_ALLOW:
            allowed_t = _DYN_ALLOW.get(asof)
            if allowed_t is not None and sym not in allowed_t:
                rec["reason"] = "dyn_allowlist"; out[sym] = rec; continue
        if sym in HYGIENE_EXCLUDE:
            rec["reason"] = "hygiene"; out[sym] = rec; continue
        td = (asof - m["earliest"]).days; rec["trailing_days"] = td
        if td < MIN_HISTORY_DAYS:
            rec["reason"] = f"maturity_{td}d_<180"; out[sym] = rec; continue
        dv = dvol_cache.get(sym, np.nan)
        if np.isfinite(dv) and dv < LIQ_FLOOR_DOLLAR_VOL_30D:
            rec["reason"] = f"liquidity_{dv:.0f}_<{LIQ_FLOOR_DOLLAR_VOL_30D:.0f}"; out[sym] = rec; continue
        rec["in_universe"] = True; out[sym] = rec
    return out


def precompute_dvol_cache(syms: list[str]) -> dict[str, float]:
    """One-shot 30d dollar-volume per sym, computed from the most-recent daily kline files."""
    cache = {}
    for sym in syms:
        sd = KLINES/sym/"5m"
        if not sd.exists(): continue
        files = sorted(sd.glob("*.parquet"))
        if len(files) < 5: continue
        try:
            recent = [pd.read_parquet(f, columns=["open_time","close","volume"])
                      for f in files[-30:]]
            r = pd.concat(recent, ignore_index=True)
            cache[sym] = float((r["close"]*r["volume"]).sum() / max(1, len(files[-30:])))
        except Exception:
            pass
    return cache


# =====================================================================
# Per-cycle strategy logic (matches X116_hl70_lagging_flip.hb(False))
# =====================================================================

def regime_for_cycle(btc30: float) -> str:
    """Raw (no-hysteresis) classifier."""
    if not np.isfinite(btc30): return "unknown"
    if btc30 > REGIME_BULL_THR: return "bull"
    if btc30 < REGIME_BEAR_THR: return "bear"
    return "side"


def apply_hysteresis(raw_regimes: list[str], n: int = REGIME_HYSTERESIS_N) -> list[str]:
    """Walk through raw regime labels in time order; require N consecutive raw cycles in a
    new regime before switching IN. Output: the EFFECTIVE regime per cycle that the
    strategy actually uses. Switching OUT to side is instantaneous (no hysteresis on side
    — side is the default, so we only stick on bull/bear if recently confirmed). Hysteresis
    on entry only; instant exit on disconfirming label."""
    out = []
    pending = None; pending_run = 0
    current = "side"   # default at start
    for raw in raw_regimes:
        if raw == current:
            pending = None; pending_run = 0
            out.append(current); continue
        # disagreement with current
        if pending == raw:
            pending_run += 1
        else:
            pending = raw; pending_run = 1
        # if raw says "side" (the default) — switch immediately (no hysteresis on exit)
        if raw == "side":
            current = "side"; pending = None; pending_run = 0
        elif pending_run >= n:
            current = raw; pending = None; pending_run = 0
        out.append(current)
    return out


def _defensive_long_syms(gg):
    """Stage-2 defensive pick among gg (expects DEF_FEATS cols). Falls back to top-pred."""
    cand = gg.nlargest(SIDE_DEF_N, "pred")
    if len(cand) < K: return None
    if set(DEF_FEATS).issubset(cand.columns) and cand[DEF_FEATS].notna().all(axis=1).sum() >= K:
        dscore = (cand["corr_to_btc_1d"].rank(pct=True) - cand["rvol_7d"].rank(pct=True)
                  - cand["atr_pct"].rank(pct=True))
        return cand.assign(_d=dscore).nlargest(K, "_d")["symbol"].tolist()
    return cand.nlargest(K, "pred")["symbol"].tolist()


def _long_plus_basket_hedge(gg, L, betas_at_t):
    """Build weights: equal-weight longs L + held beta-neutral eligible-basket short."""
    bL = np.nanmean([betas_at_t.get(s, np.nan) for s in L])
    if not np.isfinite(bL) or bL <= 0: bL = 0.8
    w = {s: 1.0/K for s in L}
    hedge_syms = [s for s in gg["symbol"].tolist() if s not in set(L)]
    if hedge_syms:
        hw = float(bL) / len(hedge_syms)
        for s in hedge_syms: w[s] = w.get(s, 0.0) - hw
    return w


def select_legs(grp: pd.DataFrame, regime: str, betas_at_t: dict[str, float],
                pred_disp_full: float | None = None) -> dict:
    """Returns sleeve weight dict for this cycle (empty in bear).

    pred_disp_full: full-universe pred dispersion for the cycle (used by regime_switch).

    BULL construction depends on BULL_MODE env var:
      mom (default)  : sort by mom_30d, equal-weight legs (keep long-beta tailwind)
      betaneut_mom   : sort by mom_30d, beta-neutral legs
      sidealpha      : sort by pred (V0 mean-rev), beta-neutral — treat bull EXACTLY like side

    SIDE construction depends on SIDE_MODE env var:
      default          : 5L/5S beta-neutral (current behavior)
      short_btc_hedge  : K=3 alt shorts + BTC long sized to neutralize basket beta
    """
    if regime == "bear": return {}

    # SIDE regime with short_btc_hedge mode — no longs, K=3 shorts + BTC hedge
    if regime == "side" and SIDE_MODE == "short_btc_hedge":
        gg = grp.dropna(subset=["pred"])
        if len(gg) < 2*SIDE_SHORT_K: return {}
        gg = gg.sort_values("pred")
        S = gg.head(SIDE_SHORT_K)["symbol"].tolist()
        # avg beta of shorts; default 0.8 if missing
        bS = np.nanmean([betas_at_t.get(s, np.nan) for s in S])
        if not np.isfinite(bS) or bS <= 0: bS = 0.8
        # K shorts at weight -1/K each → basket beta = -bS; BTC long weight = bS to neutralize
        w = {BTC_HEDGE_KEY: float(bS)}
        for s in S: w[s] = w.get(s, 0) - 1.0/SIDE_SHORT_K
        return w

    # V_LONG_HEDGE: K=3 alt longs + BTC short sized to neutralize basket beta.
    # Mirror of short_btc_hedge for current-regime where long is the working leg.
    if regime == "side" and SIDE_MODE == "long_btc_hedge":
        gg = grp.dropna(subset=["pred"])
        if len(gg) < 2*SIDE_SHORT_K: return {}
        gg = gg.sort_values("pred")
        L = gg.tail(SIDE_SHORT_K)["symbol"].tolist()
        bL = np.nanmean([betas_at_t.get(s, np.nan) for s in L])
        if not np.isfinite(bL) or bL <= 0: bL = 0.8
        # K longs at +1/K each → basket beta = +bL; BTC short weight = -bL to neutralize
        w = {BTC_HEDGE_KEY: -float(bL)}
        for s in L: w[s] = w.get(s, 0) + 1.0/SIDE_SHORT_K
        return w

    # LONG_BASKET_HEDGE (design A, iter-035): keep the working LONG leg (model top-K by pred),
    # replace the value-negative model short leg with a HELD equal-weight eligible-basket short,
    # sized (total notional = avg long beta) to neutralize the long basket's beta. The 6-sleeve
    # aggregation makes the broad basket short low-turnover automatically (cost amortization).
    # Rationale: short SELECTION alpha is anti-informative OOS (iter-033); only its beta hedging
    # was ever useful, and an alt basket hedges an alt-long book far better than BTC (iter-035).
    if regime == "side" and SIDE_MODE == "long_basket_hedge":
        gg = grp.dropna(subset=["pred"])
        if len(gg) < 2*K: return {}
        gg = gg.sort_values("pred")
        L = gg.tail(K)["symbol"].tolist()
        bL = np.nanmean([betas_at_t.get(s, np.nan) for s in L])
        if not np.isfinite(bL) or bL <= 0: bL = 0.8
        w = {s: 1.0/K for s in L}                                   # long leg: +1/K each
        hedge_syms = [s for s in gg["symbol"].tolist() if s not in set(L)]
        if hedge_syms:                                             # short eligible basket, total = bL
            hw = float(bL) / len(hedge_syms)
            for s in hedge_syms: w[s] = w.get(s, 0.0) - hw
        return w

    # LONG_DEFENSIVE_BASKET_HEDGE (iter-039): Design A + defensive two-stage long.
    # Stage 1: top-N candidates by pred. Stage 2: pick K most-defensive among them
    # (high corr_to_btc, low rvol, low atr) via parameter-free rank composite — the
    # STABLE cross-regime signature (iter-038: oracle winners are low-vol/high-corr).
    # Short the held eligible basket (beta-neutralized) as in long_basket_hedge.
    if regime == "side" and SIDE_MODE == "long_defensive_basket_hedge":
        gg = grp.dropna(subset=["pred"])
        if len(gg) < 2*K: return {}
        cand = gg.nlargest(SIDE_DEF_N, "pred")
        if len(cand) < K: return {}
        if set(DEF_FEATS).issubset(cand.columns) and cand[DEF_FEATS].notna().all(axis=1).sum() >= K:
            dscore = (cand["corr_to_btc_1d"].rank(pct=True)
                      - cand["rvol_7d"].rank(pct=True)
                      - cand["atr_pct"].rank(pct=True))
            L = cand.assign(_d=dscore).nlargest(K, "_d")["symbol"].tolist()
        else:
            L = cand.nlargest(K, "pred")["symbol"].tolist()   # fallback if feats missing
        bL = np.nanmean([betas_at_t.get(s, np.nan) for s in L])
        if not np.isfinite(bL) or bL <= 0: bL = 0.8
        w = {s: 1.0/K for s in L}
        hedge_syms = [s for s in gg["symbol"].tolist() if s not in set(L)]
        if hedge_syms:
            hw = float(bL) / len(hedge_syms)
            for s in hedge_syms: w[s] = w.get(s, 0.0) - hw
        return w

    # REGIME_SWITCH (iter-044/045): route by model conviction (pred_disp). High pred_disp ->
    # model has conviction -> model-long; low pred_disp (flat preds) -> defensive-long. Both
    # use the held basket hedge. pred_disp is the best PIT switch signal (zero lag, known at t).
    if regime == "side" and SIDE_MODE == "regime_switch":
        gg = grp.dropna(subset=["pred"])
        if len(gg) < 2*K: return {}
        pdisp = pred_disp_full if (pred_disp_full is not None and np.isfinite(pred_disp_full)) else float(gg["pred"].std())
        if np.isfinite(pdisp) and pdisp >= SIDE_SWITCH_THR:
            L = gg.nlargest(K, "pred")["symbol"].tolist()        # model mode
        else:
            L = _defensive_long_syms(gg)                          # defensive mode
            if L is None: return {}
        return _long_plus_basket_hedge(gg, L, betas_at_t)

    # V7 SIDE regime confidence-gated: bidirectional alt picks only when |pred| > threshold;
    # BTC long sized to neutralize residual basket beta. Fall back to flat (no alt + no BTC)
    # when no symbol exceeds threshold (low-conviction cycles).
    if regime == "side" and SIDE_MODE == "confidence_btc_hedge":
        gg = grp.dropna(subset=["pred"])
        if len(gg) < 2*SIDE_SHORT_K: return {}
        gg = gg.sort_values("pred")
        K_max = SIDE_SHORT_K
        # short candidates: K_max lowest preds with pred < -τ
        S_cand = gg.head(K_max)
        S = S_cand[S_cand["pred"] < -PRED_THRESHOLD]["symbol"].tolist()
        # long candidates: K_max highest preds with pred > +τ
        L_cand = gg.tail(K_max)
        L = L_cand[L_cand["pred"] > +PRED_THRESHOLD]["symbol"].tolist()
        if not S and not L: return {}   # no conviction → flat
        # basket weights: alts at ±1/K_max each
        w = {}
        for s in S: w[s] = -1.0/K_max
        for s in L: w[s] = +1.0/K_max
        # net basket beta from alts; BTC hedge sized to neutralize
        bS_avg = np.nanmean([betas_at_t.get(s, np.nan) for s in S]) if S else np.nan
        bL_avg = np.nanmean([betas_at_t.get(s, np.nan) for s in L]) if L else np.nan
        if not np.isfinite(bS_avg) or bS_avg <= 0: bS_avg = 0.8
        if not np.isfinite(bL_avg) or bL_avg <= 0: bL_avg = 0.8
        net_basket_beta = (len(L)/K_max)*bL_avg - (len(S)/K_max)*bS_avg
        w[BTC_HEDGE_KEY] = -float(net_basket_beta)   # BTC long when basket net-short, vice versa
        return w

    # Default path (and bull regime)
    if regime == "bull":
        key = "pred" if BULL_MODE == "sidealpha" else "mom30"
        do_bn = BULL_MODE in ("sidealpha", "betaneut_mom")
    else:  # side default
        key = "pred"; do_bn = True
    gg = grp.dropna(subset=[key])
    if len(gg) < 2*K: return {}
    if len(gg) < (K_LONG + K_SHORT): return {}
    gg = gg.sort_values(key)
    L = gg.tail(K_LONG)["symbol"].tolist()
    S = gg.head(K_SHORT)["symbol"].tolist()
    a = b = 1.0
    if do_bn:
        bL = np.nanmean([betas_at_t.get(s, np.nan) for s in L])
        bS = np.nanmean([betas_at_t.get(s, np.nan) for s in S])
        if np.isfinite(bL) and np.isfinite(bS) and bL > 0 and bS > 0:
            a = 2*bS/(bL+bS); b = 2*bL/(bL+bS)
    w = {}
    for s in L: w[s] = w.get(s, 0) + a/K_LONG   # long basket gross = a
    for s in S: w[s] = w.get(s, 0) - b/K_SHORT  # short basket gross = b
    return w


def aggregate_active_sleeves(sleeves: deque) -> dict:
    """net position = sum/HOLD over active sleeves."""
    net = {}
    for w in sleeves:
        for s, wt in w.items(): net[s] = net.get(s, 0.0) + wt/HOLD
    return net


# =====================================================================
# iter-012 vol-norm reactive equity-DD stop
# =====================================================================

class VolNormStop:
    def __init__(self, k=STOP_K_SIGMA, g_floor=STOP_G_FLOOR, sigma_win=STOP_SIGMA_WINDOW,
                 warmup=STOP_WARMUP, heal=STOP_HEAL_FRAC, timeout=STOP_TIMEOUT_BARS):
        self.k = k; self.g_floor = g_floor; self.sigma_win = sigma_win
        self.warmup = warmup; self.heal = heal; self.timeout = timeout
        self.eq_hist = deque(maxlen=sigma_win+1)   # equity increments
        self.peak = INITIAL_EQUITY
        self.engaged = False
        self.engage_dd = 0.0
        self.engage_age = 0
        self.trough = INITIAL_EQUITY
    def update(self, equity_pre_t: float, equity_post_t: float, bar_idx: int) -> tuple[float, dict]:
        """Returns (gross_mult, diag_dict). PIT — uses equity through t-1 for sigma threshold."""
        # threshold uses equity through t-1 only
        if len(self.eq_hist) >= self.warmup:
            inc = np.diff(np.array(self.eq_hist))
            sigma = float(np.std(inc)) if len(inc) >= 2 else 0.0
        else:
            sigma = 0.0
        thr = self.k * sigma * np.sqrt(self.sigma_win)
        self.peak = max(self.peak, equity_pre_t)
        dd = self.peak - equity_pre_t
        if not self.engaged:
            if bar_idx >= self.warmup and dd >= thr and thr > 0:
                self.engaged = True; self.engage_dd = dd; self.engage_age = 0
                self.trough = equity_pre_t
            gross_mult = 1.0
        else:
            self.engage_age += 1
            self.trough = min(self.trough, equity_pre_t)
            healed = (self.peak - equity_pre_t) <= self.engage_dd * self.heal
            timed = self.engage_age >= self.timeout
            above_trough = equity_pre_t > self.trough
            if (healed or timed) and above_trough:
                self.engaged = False; gross_mult = 1.0
            else:
                gross_mult = self.g_floor
        # record equity AFTER this bar's MtM for next-bar sigma calc
        self.eq_hist.append(equity_post_t)
        return gross_mult, dict(sigma=sigma, threshold=thr, dd=dd, peak=self.peak,
                                engaged=self.engaged, engage_age=self.engage_age)


# =====================================================================
# Replay engine (Stage A)
# =====================================================================

def run_replay(start: pd.Timestamp | None, end: pd.Timestamp | None) -> dict:
    log.info("loading preds + features")
    d = load_preds(start=start)
    if end is not None: d = d[d["open_time"] <= end]
    syms = sorted(d["symbol"].unique())
    log.info(f"replay: {len(d):,} rows × {len(syms)} syms × {d['open_time'].min().date()}→{d['open_time'].max().date()}")
    mom, betas = compute_mom30_and_beta(syms)
    btc30 = compute_btc_30d()
    d = d.merge(mom, on=["symbol","open_time"], how="left")
    d = d.merge(btc30.reset_index(), on="open_time", how="left").dropna(subset=["btc_ret_30d"])
    # defensive-tilt features (only needed by SIDE_MODE=long_defensive_basket_hedge; harmless otherwise)
    if SIDE_MODE in ("long_defensive_basket_hedge", "regime_switch"):
        _pf = pd.read_parquet(PANEL, columns=["symbol","open_time"]+DEF_FEATS)
        _pf["open_time"] = pd.to_datetime(_pf["open_time"], utc=True)
        d = d.merge(_pf, on=["symbol","open_time"], how="left")
    d["regime_raw"] = d["btc_ret_30d"].apply(regime_for_cycle)
    # apply N=3 hysteresis on raw regime labels in time order — kills the boundary-flip whipsaw.
    raw_by_t = d.sort_values("open_time").groupby("open_time")["regime_raw"].first()
    eff = apply_hysteresis(raw_by_t.tolist(), n=REGIME_HYSTERESIS_N)
    regime_eff = dict(zip(raw_by_t.index, eff))
    d["regime"] = d["open_time"].map(regime_eff)
    log.info(f"hysteresis (N={REGIME_HYSTERESIS_N}): raw {d['regime_raw'].value_counts().to_dict()} -> eff {d['regime'].value_counts().to_dict()}")

    # BTC forward 4h return for the BTC_HEDGE position (used only by SIDE_MODE=short_btc_hedge)
    btc_close = load_close_4h("BTCUSDT")
    btc_fwd_4h = btc_close.pct_change().shift(-1).rename("btc_fwd_4h")
    btc_fwd_map = btc_fwd_4h.dropna().to_dict()

    by_t = {ot: g for ot, g in d.groupby("open_time")}
    times = sorted(by_t.keys())

    # Pre-compute per-cycle pred_disp + trailing-N percentile threshold (PIT — uses
    # values through t-1 for threshold; current cycle's pred_disp is compared against it).
    disp_per_t = {ot: float(by_t[ot]["pred"].std()) for ot in times}
    disp_hist = deque(maxlen=DISP_GATE_LOOKBACK)
    disp_skip = {}   # ot -> bool (whether to skip in side regime this cycle)
    for ot in times:
        cur = disp_per_t[ot]
        if DISP_GATE and len(disp_hist) >= DISP_GATE_MIN_HISTORY:
            thr = float(np.quantile(np.array(disp_hist), DISP_GATE_PCTILE))
            disp_skip[ot] = cur < thr
        else:
            disp_skip[ot] = False
        disp_hist.append(cur)
    if DISP_GATE:
        n_skip = sum(disp_skip.values()); log.info(f"disp-gate (pctile {DISP_GATE_PCTILE} lookback {DISP_GATE_LOOKBACK}): {n_skip}/{len(times)} cycles flagged for skip")

    # state
    active_sleeves: deque = deque(maxlen=HOLD)
    prev_agg: dict = {}
    equity = INITIAL_EQUITY
    stop = VolNormStop()
    cycles_rows, regime_rows, equity_rows, sleeves_rows, trades_rows, pred_rows, univ_rows = [], [], [], [], [], [], []

    # universe meta + dvol cache precomputed ONCE (meta uses FULL preds file, not slice)
    log.info("precomputing universe meta + 30d dvol cache (one-shot)")
    univ_meta = precompute_universe_meta()
    dvol_cache = precompute_dvol_cache(syms)
    log.info(f"  univ_meta: {len(univ_meta)} syms; dvol_cache: {len(dvol_cache)} syms")
    last_univ = {}
    last_regime = None
    sleeve_serial = 0
    sleeve_ids_active: deque = deque(maxlen=HOLD)

    t0 = time.time()
    for bar_idx, ot in enumerate(times):
        g = by_t[ot]
        # universe per-cycle (cheap — dict lookups only, no I/O)
        univ = eligible_universe_at(univ_meta, ot, dvol_cache)
        last_univ = univ   # keep latest for end-of-replay universe.csv snapshot
        eligible_syms = {s for s, r in univ.items() if r["in_universe"]}
        g_elig = g[g["symbol"].isin(eligible_syms)].copy()

        regime = g["regime"].iloc[0] if len(g) else "unknown"
        # betas at this t (latest available shifted-by-1 already)
        betas_at_t = {}
        for s, ser in betas.items():
            if ot in ser.index:
                v = ser.loc[ot]; betas_at_t[s] = float(v) if np.isfinite(v) else np.nan

        # build new sleeve weights for THIS cycle
        new_w = select_legs(g_elig, regime, betas_at_t, pred_disp_full=disp_per_t.get(ot))
        # disp-gate override: if model dispersion is in the bottom pctile, side regime goes flat
        if DISP_GATE and regime == "side" and disp_skip.get(ot, False):
            new_w = {}
        active_sleeves.append(new_w)
        sleeve_serial += 1
        # exit event for sleeve that fell out (if any)
        if len(sleeve_ids_active) == HOLD:
            old_id = sleeve_ids_active[0]
            # exit logged below after MtM
        sleeve_ids_active.append(sleeve_serial)

        # aggregate target across active sleeves
        gross_target = sum(abs(w) for w in aggregate_active_sleeves(active_sleeves).values())
        net_target_raw = aggregate_active_sleeves(active_sleeves)
        # stop overlay
        gross_mult, stop_diag = stop.update(equity, equity, bar_idx)   # pre-MtM call; equity_post fills later
        net_after = {s: w*gross_mult for s, w in net_target_raw.items()}

        # cost from turnover (prev_agg -> net_after)
        all_keys = set(net_after) | set(prev_agg)
        turn = sum(abs(net_after.get(s, 0) - prev_agg.get(s, 0)) for s in all_keys)
        cost_bps_cycle = turn * 0.5 * COST * 1e4

        # mark to market using realized return_pct (4h-forward, already in panel as alpha_A's return)
        rmap = dict(zip(g["symbol"], g["return_pct"]))
        # Inject BTC forward return for the synthetic BTC hedge (if present in net_after)
        if BTC_HEDGE_KEY in net_after:
            rmap[BTC_HEDGE_KEY] = float(btc_fwd_map.get(ot, 0.0))
        gross_pnl = sum(net_after.get(s, 0) * rmap.get(s, 0.0)
                        for s in net_after if np.isfinite(rmap.get(s, 0.0)))
        cost_unit = turn * 0.5 * COST
        pnl_unit = gross_pnl - cost_unit
        equity_pre = equity
        equity = equity_pre * (1.0 + pnl_unit)
        # let stop see post-equity for the next cycle's sigma calc
        stop.eq_hist[-1] = equity if stop.eq_hist else stop.eq_hist
        # log rows
        univ_hash = hashlib.sha1(",".join(sorted(eligible_syms)).encode()).hexdigest()[:8]
        cycles_rows.append(dict(
            cycle_id=bar_idx, open_time=ot,
            n_universe=len(eligible_syms), n_predicted=len(g),
            btc_ret_30d=float(g["btc_ret_30d"].iloc[0]) if len(g) else np.nan,
            regime=regime,
            pred_disp=float(g_elig["pred"].std()) if len(g_elig)>1 else np.nan,
            top_k_long=",".join(sorted(s for s, w in new_w.items() if w > 0)),
            bot_k_short=",".join(sorted(s for s, w in new_w.items() if w < 0)),
            gross_target=gross_target, gross_after_stop=sum(abs(v) for v in net_after.values()),
            net_target=sum(net_target_raw.values()),
            sleeve_count=len(active_sleeves),
            stop_engaged=bool(stop_diag["engaged"]),
            stop_k_sigma=float(stop_diag["threshold"]),
            equity_pre=equity_pre, equity_post=equity,
            pnl_bps=(equity-equity_pre)/equity_pre*1e4 if equity_pre>0 else np.nan,
            gross_pnl_bps=gross_pnl*1e4, cost_bps=cost_bps_cycle, turnover=turn,
            n_trades=int(sum(1 for s in all_keys if abs(net_after.get(s,0)-prev_agg.get(s,0))>1e-6)),
            univ_hash=univ_hash,
            notes=""))
        regime_rows.append(dict(cycle_id=bar_idx, open_time=ot,
            btc_ret_30d=cycles_rows[-1]["btc_ret_30d"], regime=regime,
            transition_from=last_regime or ""))
        equity_rows.append(dict(cycle_id=bar_idx, open_time=ot, equity=equity,
            peak=stop_diag["peak"], drawdown_bps=stop_diag["dd"]/max(stop_diag["peak"],1)*1e4,
            sigma_180_bps=stop_diag["sigma"], stop_threshold_bps=stop_diag["threshold"],
            stop_engaged=bool(stop_diag["engaged"])))
        # predictions snapshot
        ps = g[["symbol","pred"]].copy(); ps["cycle_id"]=bar_idx; ps["open_time"]=ot
        ps["pred_xs_rank"] = ps["pred"].rank() / max(1,len(ps))
        ps["eligible"] = ps["symbol"].isin(eligible_syms)
        ps["selected_long"] = ps["symbol"].isin([s for s,w in new_w.items() if w>0])
        ps["selected_short"] = ps["symbol"].isin([s for s,w in new_w.items() if w<0])
        pred_rows.append(ps)
        # sleeve entry/exit
        sleeves_rows.append(dict(sleeve_id=sleeve_serial, event="enter", cycle_id=bar_idx,
            open_time=ot, weights_json=json.dumps({s:round(w,6) for s,w in new_w.items()}),
            gross=sum(abs(w) for w in new_w.values()), net=sum(new_w.values()),
            regime_at_entry=regime, ret_bps=np.nan, cost_bps=np.nan, pnl_bps=np.nan))

        prev_agg = net_after; last_regime = regime

    cycles = pd.DataFrame(cycles_rows)
    cycles.to_csv(STATE/"cycles.csv", index=False)
    pd.DataFrame(regime_rows).to_csv(STATE/"regime.csv", index=False)
    pd.DataFrame(equity_rows).to_csv(STATE/"equity.csv", index=False)
    pd.DataFrame(sleeves_rows).to_csv(STATE/"sleeves.csv", index=False)
    if pred_rows: pd.concat(pred_rows, ignore_index=True).to_parquet(STATE/"predictions.parquet")
    # universe.csv (last-cycle snapshot only for size)
    univ_df = pd.DataFrame([dict(symbol=s, **r) for s,r in last_univ.items()])
    univ_df.to_csv(STATE/"universe.csv", index=False)

    # quick summary
    p = cycles["pnl_bps"].values / 1e4
    sh = float(p.mean()/p.std()*np.sqrt(6*365)) if p.std()>0 else np.nan
    cum = pd.Series(cycles["pnl_bps"].values).cumsum()
    ddv = float((cum - cum.cummax()).min()) if len(cum) else np.nan
    summary = dict(
        cycles=len(cycles), start=str(cycles["open_time"].min()), end=str(cycles["open_time"].max()),
        Sharpe_ann=float(sh) if np.isfinite(sh) else None,
        totPnL_bps=float(cycles["pnl_bps"].sum()), maxDD_bps=float(ddv) if np.isfinite(ddv) else None,
        equity_final=float(cycles["equity_post"].iloc[-1]) if len(cycles) else np.nan,
        stop_engaged_pct=float(cycles["stop_engaged"].mean()*100) if len(cycles) else 0.0,
        regime_counts=cycles["regime"].value_counts().to_dict(),
        elapsed_s=round(time.time()-t0,1),
    )
    (STATE/"replay_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    log.info(f"replay done: {summary}")
    return summary


POSITIONS = STATE/"positions.json"


def _save_state(state: dict):
    POSITIONS.write_text(json.dumps(state, default=str, indent=2))


def _load_state() -> dict | None:
    if not POSITIONS.exists(): return None
    return json.loads(POSITIONS.read_text())


def _state_from_replay_final(equity, peak, eq_hist, active_sleeves, prev_agg, sleeve_serial,
                             last_cycle_id, last_open_time, stop_state):
    return dict(
        equity=float(equity), peak=float(peak), eq_hist=[float(x) for x in eq_hist],
        active_sleeves=[{k: float(v) for k,v in w.items()} for w in active_sleeves],
        prev_agg={k: float(v) for k,v in prev_agg.items()},
        sleeve_serial=int(sleeve_serial),
        last_cycle_id=int(last_cycle_id),
        last_open_time=str(last_open_time) if last_open_time is not None else None,
        stop=stop_state,
    )


def run_cycle() -> dict:
    """Single live cycle (or multi-cycle catch-up): load state, find new cycles in
    PREDS since last_open_time, process them, APPEND to state files, save positions.json."""
    state = _load_state()
    if state is None:
        log.error("no positions.json — run --replay-from <date> or --bootstrap-state first."); sys.exit(2)

    # Find new cycles
    d = load_preds(start=pd.Timestamp(state["last_open_time"]) if state["last_open_time"] else None)
    d = d[d["open_time"] > pd.Timestamp(state["last_open_time"])] if state["last_open_time"] else d
    if len(d) == 0:
        log.info(f"no new cycles past {state['last_open_time']} — nothing to do."); return {"new_cycles": 0}
    syms = sorted(d["symbol"].unique())
    log.info(f"cycle: catching up {d['open_time'].nunique()} new cycle(s) "
             f"{d['open_time'].min()}→{d['open_time'].max()}")
    mom, betas = compute_mom30_and_beta(syms, lookback_days=45)   # live: only new cycles need trailing-30d betas
    btc30 = compute_btc_30d()
    d = d.merge(mom, on=["symbol","open_time"], how="left")
    d = d.merge(btc30.reset_index(), on="open_time", how="left").dropna(subset=["btc_ret_30d"])
    d["regime_raw"] = d["btc_ret_30d"].apply(regime_for_cycle)
    # hysteresis: seed from the last N+5 raw regimes already logged (need to recompute raw
    # from cycles.csv's btc_ret_30d since we only persist the effective regime).
    cyc_path = STATE/"cycles.csv"; seed_raw = []
    if cyc_path.exists():
        old = pd.read_csv(cyc_path).sort_values("open_time").tail(REGIME_HYSTERESIS_N+5)
        seed_raw = [regime_for_cycle(b) for b in old["btc_ret_30d"]]
    raw_by_t = d.sort_values("open_time").groupby("open_time")["regime_raw"].first()
    all_raw = seed_raw + raw_by_t.tolist()
    all_eff = apply_hysteresis(all_raw, n=REGIME_HYSTERESIS_N)
    eff_new = all_eff[len(seed_raw):]
    regime_eff = dict(zip(raw_by_t.index, eff_new))
    d["regime"] = d["open_time"].map(regime_eff)
    log.info(f"hysteresis seeded with {len(seed_raw)} prior raw cycles; new cycles regime: {pd.Series(eff_new).value_counts().to_dict()}")
    by_t = {ot: g for ot, g in d.groupby("open_time")}
    times = sorted(by_t.keys())

    # restore state
    active_sleeves: deque = deque([{k: float(v) for k,v in w.items()} for w in state["active_sleeves"]], maxlen=HOLD)
    prev_agg = {k: float(v) for k,v in state["prev_agg"].items()}
    equity = float(state["equity"])
    sleeve_serial = int(state["sleeve_serial"])
    stop = VolNormStop()
    stop.peak = float(state["stop"]["peak"]); stop.engaged = bool(state["stop"]["engaged"])
    stop.engage_dd = float(state["stop"]["engage_dd"]); stop.engage_age = int(state["stop"]["engage_age"])
    stop.trough = float(state["stop"]["trough"])
    stop.eq_hist = deque([float(x) for x in state["stop"]["eq_hist"]], maxlen=STOP_SIGMA_WINDOW+1)
    last_cycle_id = int(state["last_cycle_id"])
    last_regime = None

    univ_meta = precompute_universe_meta()
    dvol_cache = precompute_dvol_cache(syms)

    cycles_rows, regime_rows, equity_rows, sleeves_rows, pred_rows = [], [], [], [], []
    bar_idx_base = last_cycle_id + 1
    for i, ot in enumerate(times):
        cid = bar_idx_base + i
        g = by_t[ot]
        univ = eligible_universe_at(univ_meta, ot, dvol_cache)
        eligible_syms = {s for s, r in univ.items() if r["in_universe"]}
        g_elig = g[g["symbol"].isin(eligible_syms)].copy()
        regime = g["regime"].iloc[0] if len(g) else "unknown"
        betas_at_t = {}
        for s, ser in betas.items():
            if ot in ser.index:
                v = ser.loc[ot]; betas_at_t[s] = float(v) if np.isfinite(v) else np.nan
        new_w = select_legs(g_elig, regime, betas_at_t)
        active_sleeves.append(new_w); sleeve_serial += 1
        gross_target = sum(abs(w) for w in aggregate_active_sleeves(active_sleeves).values())
        net_target_raw = aggregate_active_sleeves(active_sleeves)
        gross_mult, stop_diag = stop.update(equity, equity, len(stop.eq_hist))
        net_after = {s: w*gross_mult for s, w in net_target_raw.items()}
        all_keys = set(net_after) | set(prev_agg)
        turn = sum(abs(net_after.get(s,0) - prev_agg.get(s,0)) for s in all_keys)
        rmap = dict(zip(g["symbol"], g["return_pct"]))
        gross_pnl = sum(net_after.get(s,0) * rmap.get(s,0.0)
                        for s in net_after if np.isfinite(rmap.get(s,0.0)))
        cost_unit = turn * 0.5 * COST
        equity_pre = equity; equity = equity_pre * (1.0 + gross_pnl - cost_unit)
        if stop.eq_hist: stop.eq_hist[-1] = equity
        univ_hash = hashlib.sha1(",".join(sorted(eligible_syms)).encode()).hexdigest()[:8]
        cycles_rows.append(dict(
            cycle_id=cid, open_time=ot,
            n_universe=len(eligible_syms), n_predicted=len(g),
            btc_ret_30d=float(g["btc_ret_30d"].iloc[0]),
            regime=regime,
            pred_disp=float(g_elig["pred"].std()) if len(g_elig)>1 else np.nan,
            top_k_long=",".join(sorted(s for s,w in new_w.items() if w>0)),
            bot_k_short=",".join(sorted(s for s,w in new_w.items() if w<0)),
            gross_target=gross_target, gross_after_stop=sum(abs(v) for v in net_after.values()),
            net_target=sum(net_target_raw.values()),
            sleeve_count=len(active_sleeves),
            stop_engaged=bool(stop_diag["engaged"]),
            stop_k_sigma=float(stop_diag["threshold"]),
            equity_pre=equity_pre, equity_post=equity,
            pnl_bps=(equity-equity_pre)/equity_pre*1e4,
            gross_pnl_bps=gross_pnl*1e4, cost_bps=cost_unit*1e4, turnover=turn,
            n_trades=int(sum(1 for s in all_keys if abs(net_after.get(s,0)-prev_agg.get(s,0))>1e-6)),
            univ_hash=univ_hash, notes=""))
        regime_rows.append(dict(cycle_id=cid, open_time=ot,
            btc_ret_30d=cycles_rows[-1]["btc_ret_30d"], regime=regime,
            transition_from=last_regime or ""))
        equity_rows.append(dict(cycle_id=cid, open_time=ot, equity=equity,
            peak=stop_diag["peak"], drawdown_bps=stop_diag["dd"]/max(stop_diag["peak"],1)*1e4,
            sigma_180_bps=stop_diag["sigma"], stop_threshold_bps=stop_diag["threshold"],
            stop_engaged=bool(stop_diag["engaged"])))
        ps = g[["symbol","pred"]].copy(); ps["cycle_id"]=cid; ps["open_time"]=ot
        ps["pred_xs_rank"] = ps["pred"].rank() / max(1,len(ps))
        ps["eligible"] = ps["symbol"].isin(eligible_syms)
        ps["selected_long"] = ps["symbol"].isin([s for s,w in new_w.items() if w>0])
        ps["selected_short"] = ps["symbol"].isin([s for s,w in new_w.items() if w<0])
        pred_rows.append(ps)
        sleeves_rows.append(dict(sleeve_id=sleeve_serial, event="enter", cycle_id=cid,
            open_time=ot, weights_json=json.dumps({s:round(w,6) for s,w in new_w.items()}),
            gross=sum(abs(w) for w in new_w.values()), net=sum(new_w.values()),
            regime_at_entry=regime, ret_bps=np.nan, cost_bps=np.nan, pnl_bps=np.nan))
        prev_agg = net_after; last_regime = regime

    # APPEND to existing logs
    def _append(rows, fname):
        if not rows: return
        df = pd.DataFrame(rows); path = STATE/fname
        df.to_csv(path, mode="a", header=not path.exists(), index=False)
    _append(cycles_rows, "cycles.csv")
    _append(regime_rows, "regime.csv")
    _append(equity_rows, "equity.csv")
    _append(sleeves_rows, "sleeves.csv")
    if pred_rows:
        new = pd.concat(pred_rows, ignore_index=True)
        path = STATE/"predictions.parquet"
        if path.exists():
            old = pd.read_parquet(path); pd.concat([old, new], ignore_index=True).to_parquet(path)
        else: new.to_parquet(path)

    # save state
    final_state = _state_from_replay_final(
        equity, stop.peak, list(stop.eq_hist), list(active_sleeves), prev_agg, sleeve_serial,
        cycles_rows[-1]["cycle_id"], cycles_rows[-1]["open_time"],
        dict(peak=stop.peak, engaged=stop.engaged, engage_dd=stop.engage_dd,
             engage_age=stop.engage_age, trough=stop.trough,
             eq_hist=[float(x) for x in stop.eq_hist]))
    _save_state(final_state)
    log.info(f"appended {len(cycles_rows)} cycle(s); equity {state['equity']:.2f}→{equity:.2f} "
             f"({(equity-state['equity'])/max(state['equity'],1)*1e4:+.0f} bps)")
    return dict(new_cycles=len(cycles_rows), equity=float(equity),
                last_open_time=str(cycles_rows[-1]["open_time"]))


def bootstrap_state_from_replay():
    """After --replay-all (or --replay N) runs, snapshot final state into positions.json
    so subsequent --cycle calls can resume."""
    cyc_path = STATE/"cycles.csv"; eq_path = STATE/"equity.csv"
    if not cyc_path.exists() or not eq_path.exists():
        log.error("no cycles.csv/equity.csv — run --replay first."); sys.exit(2)
    c = pd.read_csv(cyc_path); e = pd.read_csv(eq_path)
    if len(c)==0: log.error("empty cycles.csv"); sys.exit(2)
    last = c.iloc[-1]
    # reconstruct sleeves from sleeves.csv last HOLD enter events
    sl = pd.read_csv(STATE/"sleeves.csv")
    sl_enter = sl[sl["event"]=="enter"].tail(HOLD)
    active = [json.loads(r["weights_json"]) for _, r in sl_enter.iterrows()]
    # eq_hist from equity.csv last STOP_SIGMA_WINDOW+1
    eq_hist = e["equity"].tail(STOP_SIGMA_WINDOW+1).tolist()
    prev_agg = aggregate_active_sleeves(deque(active, maxlen=HOLD))
    state = _state_from_replay_final(
        last["equity_post"], e["peak"].max(), eq_hist, active, prev_agg,
        sl["sleeve_id"].max() if len(sl) else 0,
        last["cycle_id"], last["open_time"],
        dict(peak=float(e["peak"].iloc[-1]), engaged=bool(last["stop_engaged"]),
             engage_dd=0.0, engage_age=0, trough=float(last["equity_post"]),
             eq_hist=[float(x) for x in eq_hist]))
    _save_state(state)
    log.info(f"bootstrapped positions.json: equity={state['equity']:.2f}, "
             f"last_open_time={state['last_open_time']}, sleeves={len(state['active_sleeves'])}")


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--replay", type=int, help="replay last N days (fresh state, overwrite logs)")
    g.add_argument("--replay-all", action="store_true")
    g.add_argument("--replay-from", type=str, help="ISO date, e.g. 2025-10-01")
    ap.add_argument("--replay-end", type=str, help="ISO date (optional end for replay-from)")
    g.add_argument("--bootstrap-state", action="store_true",
                   help="after a replay, snapshot final state into positions.json")
    g.add_argument("--cycle", action="store_true",
                   help="process new cycles since positions.json last_open_time (append)")
    g.add_argument("--check-state", action="store_true")
    args = ap.parse_args()
    if args.check_state:
        for f in sorted(STATE.glob("*")):
            sz = f.stat().st_size if f.is_file() else 0
            print(f"  {f.name:<30} {sz:>10}B")
        if POSITIONS.exists():
            s = _load_state()
            print(f"  positions: equity={s['equity']:.2f}, last={s['last_open_time']}, "
                  f"sleeves={len(s['active_sleeves'])}, stop_engaged={s['stop']['engaged']}")
        return
    if args.bootstrap_state:
        bootstrap_state_from_replay(); return
    if args.cycle:
        run_cycle(); return
    start = None; end = None
    if args.replay_all:
        start = None
    elif args.replay is not None:
        end = pd.Timestamp.utcnow().tz_convert("UTC") if pd.Timestamp.utcnow().tz else pd.Timestamp.utcnow().tz_localize("UTC")
        # use panel's actual end if it's older
        try:
            panel_end = pd.read_parquet(PREDS, columns=["open_time"])["open_time"].max()
            if isinstance(panel_end, pd.Timestamp) and panel_end.tz is None: panel_end = panel_end.tz_localize("UTC")
            end = panel_end
        except Exception: pass
        start = end - pd.Timedelta(days=args.replay)
    elif args.replay_from:
        start = pd.Timestamp(args.replay_from, tz="UTC")
    if args.replay_end:
        end = pd.Timestamp(args.replay_end, tz="UTC")
    run_replay(start, end)


if __name__ == "__main__":
    main()
