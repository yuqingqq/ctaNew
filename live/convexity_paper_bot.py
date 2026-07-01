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
# iter13 (meta-labeling #172): optional SECOND preds file whose pred ranks the LONG leg only (e.g. resid_rev-blended
# model). Base PREDS ranks shorts. Separates alpha-direction (short) from long-tradeability without one global model.
PREDS_LONG = os.environ.get("CONVEXITY_PREDS_LONG", "")
# Universe-meta source = the LIVE PANEL (full-history, grows as new symbols list). MUST NOT be a frozen
# research preds cache (the old x132 file froze at 160 syms on 2026-05-20 → silently capped eligibility,
# excluding ~15 newer listings that ARE in the panel: ASTER/CC/etc.). Env-overridable for tests.
UNIVERSE_META_PREDS = Path(os.environ.get("CONVEXITY_UNIVERSE_META", str(PANEL)))
KLINES   = REPO/"data/ml/test/parquet/klines"
RESEARCH_LOG = STATE/"replay_vs_research.csv"

# Strategy constants (FROZEN per shared/baseline.md)
K = int(os.environ.get("STRAT_K", "5"))
# asymmetric K (default mode): separate long/short basket sizes (default = K). xs_z shorts informative.
K_LONG = int(os.environ.get("STRAT_K_LONG", str(K)))
K_SHORT = int(os.environ.get("STRAT_K_SHORT", str(K)))
HOLD = int(os.environ.get("STRAT_HOLD", "6"))
# Regime-specific hold: bull mean-rev edge REVERSES within 24h (squeeze-back) — validated +40bps@4h -> -21bps@24h —
# so in bull use a SHORTER hold (aggregate only the freshest BULL_HOLD sleeves). side/bear edge compounds -> keep HOLD.
BULL_HOLD = int(os.environ.get("STRAT_HOLD_BULL", str(HOLD)))
# COST per leg in fraction (4.5e-4 = 4.5 bps). Env override COST_BPS_LEG (in bps).
COST = float(os.environ["COST_BPS_LEG"])*1e-4 if "COST_BPS_LEG" in os.environ else 4.5e-4
REGIME_BULL_THR = float(os.environ.get("REGIME_BULL_THR", "0.10"))
REGIME_BEAR_THR = float(os.environ.get("REGIME_BEAR_THR", "-0.10"))
SIDE_BETA_NEUT = os.environ.get("SIDE_BETA_NEUT","1")=="1"   # beta-neut reweight in side (else equal-weight)
MOM_WINDOW = int(os.environ.get("MOM_WINDOW","180"))   # bull momentum lookback in 4h bars (180=30d)
BEAR_MODE = os.environ.get("BEAR_MODE", "flat")   # flat (production: sit out bear) | side (trade bear via mean-rev L/S)
# Hysteresis: require N consecutive raw-bull cycles before SWITCHING IN to bull (and same
# for bear). Reduces boundary-flip whipsaw. Empirically (OOS 2025-10→2026-05): 9 of 14
# bull episodes lasted <1d at Sharpe −18.86 due to boundary chatter.
REGIME_HYSTERESIS_N = int(os.environ.get("REGIME_HYSTERESIS_N", "3"))
# BULL_MODE: which construction the strategy uses in BULL regime.
#   "mom"          : mom_30d trend-follow, equal-weight legs (KEEPS +0.286 net beta) — BASELINE
#   "betaneut_mom" : mom_30d trend-follow, beta-neutral leg sizing (kills the long-beta leak)
#   "sidealpha"    : V0 pred mean-rev with beta-neutral (treat bull EXACTLY like side)
BULL_MODE = os.environ.get("BULL_MODE", "mom")
BULL_K = int(os.environ.get("BULL_K","0"))   # bull-specific K (0=use global K_LONG/K_SHORT)
BEAR_K = int(os.environ.get("BEAR_K","0"))   # bear-specific K (0=use global)
# 2026-06-08 bear de-gross: scale bear-equal basket gross (1.0=full v2; <1 trades bear edge at lower size to cut the
# bear tail — iter7 found maxDD/51%-loss-concentration all bear). Off by default (1.0). 0.0 == BEAR_MODE=flat.
BEAR_GROSS_MULT = float(os.environ.get("BEAR_GROSS_MULT","1.0"))
BULL_GROSS_MULT = float(os.environ.get("BULL_GROSS_MULT","1.0"))   # scale bull-entry sleeve gross (0=flat in bull)
# Bull sub-regime gate: the short edge works in MILD/early bull (btc_ret_30d ~0.10-0.15: pump topping -> reverts,
# +45bps/cyc) but FAILS in DEEP sustained melt-up (squeeze zone, -6bps). Flat the deep-bull cycles. 99=off.
BULL_DEEP_THR = float(os.environ.get("BULL_DEEP_THR","99"))
# 5m entry-confirmation (bull short): don't short a name still making a 30m new high (still pumping -> squeeze).
# Lifts bull short t-stat 0.91->1.25 (validated on 5m). ENTRY_FLAG_PARQUET = {symbol,open_time,made_nh30,off_high2}.
ENTRY_FLAG_PARQUET = os.environ.get("ENTRY_FLAG_PARQUET","")
BULL_ENTRY_NH30 = os.environ.get("BULL_ENTRY_NH30","0")=="1"
# "strong" entry gate: short only names that have ROLLED OVER (no new high AND >x% below 4h high). t->1.98 on 5m.
BULL_ENTRY_MODE = os.environ.get("BULL_ENTRY_MODE","nh30")   # nh30 (skip new-highs) | strong (require rolled-over)
BULL_ENTRY_NH_COL  = os.environ.get("BULL_ENTRY_NH_COL","made_nh30")   # new-high column (nh15/nh30/nh60)
BULL_ENTRY_OFF_COL = os.environ.get("BULL_ENTRY_OFF_COL","off_high2")  # off-high column (off1/off2/off3)
BULL_ENTRY_PLACEBO_SEED = int(os.environ.get("BULL_ENTRY_PLACEBO_SEED","-1"))  # >=0: keep RANDOM same-size subset
# scale the bear LONG leg only (1.0=full/byte-identical; 0.0=short-only bear). The bear long leg is negative-alpha
# (-5bps, <50% hit), the bear short leg positive (+11bps) — cutting bear longs may be a real Sharpe gain, not just
# insurance. NB: <1.0 makes the bear book net-SHORT (adds short beta — profitable in a falling bear, risk on reversal).
BEAR_LONG_MULT = float(os.environ.get("BEAR_LONG_MULT","1.0"))
# symmetric test: scale the BULL long leg (1.0=full/byte-identical). Bull long-alpha is weak/mixed; small sample.
BULL_LONG_MULT = float(os.environ.get("BULL_LONG_MULT","1.0"))
BULL_LONG_INSTRUMENT = os.environ.get("BULL_LONG_INSTRUMENT","alt")  # alt (model longs) | btc (same $ into BTC long)
# bear: cut alt longs (negative-alpha) and hedge the resulting net-short with a BTC long (beta-neutral) — removes
# the long drag WITHOUT the directional net-short bet of BEAR_LONG_MULT=0. 0=off (default).
BEAR_HEDGE_BTC = os.environ.get("BEAR_HEDGE_BTC","0")=="1"
# HONEST ACCOUNTING: charge realized funding carry on the held book (1=on; 0=off/byte-identical). Validated ~-0.7
# Sharpe on the deploy base. Uses contemporaneous funding_rate (the carry actually paid — not a predictive feature,
# so no leak), scaled FUND_CYCLE_FRAC of the 8h rate per 4h cycle.
CHARGE_FUNDING = os.environ.get("CHARGE_FUNDING","0")=="1"
FUND_CYCLE_FRAC = float(os.environ.get("FUND_CYCLE_FRAC","0.5"))   # 4h bar ≈ half an 8h funding interval
# PER-SYMBOL depth-aware cost: charge each leg its real per-fill cost (fee+impact) instead of flat turn*0.5*COST.
# DEPTH_COST_CSV = path to {symbol, cost_10k/50k/100k} (per-fill bps); DEPTH_COST_TIER picks the AUM column.
# Per-fill (no 0.5): cost = Σ|Δnet[s]|·c_fill(s). Missing symbols -> median. OFF (flat) unless DEPTH_COST_CSV set.
DEPTH_COST_CSV  = os.environ.get("DEPTH_COST_CSV","")
DEPTH_COST_TIER = os.environ.get("DEPTH_COST_TIER","cost_10k")
# TAIL GATE: drop names whose walk-forward tail-risk score (p_tail) is in the top (1-pctile) per cycle.
TAIL_SCORE_PARQUET = os.environ.get("TAIL_SCORE_PARQUET","")
TAIL_SKIP_PCTILE   = float(os.environ.get("TAIL_SKIP_PCTILE","1.0"))   # 1.0=off; e.g. 0.90 drops top-10% tail-risk
BTC_HEDGE_COST_BPS = os.environ.get("BTC_HEDGE_COST_BPS","")   # per-fill bps for the BTC hedge; "" = median fallback
_PERSYM_COST = None; _PERSYM_COST_MED = None
def _persym_cost_map():
    global _PERSYM_COST, _PERSYM_COST_MED
    if _PERSYM_COST is None and DEPTH_COST_CSV:
        _df = pd.read_csv(DEPTH_COST_CSV)
        _PERSYM_COST = {r["symbol"]: float(r[DEPTH_COST_TIER])*1e-4 for _, r in _df.iterrows()}
        _PERSYM_COST_MED = float(_df[DEPTH_COST_TIER].median())*1e-4
        if BTC_HEDGE_COST_BPS:   # explicit BTC hedge cost (BTC not in HL capacity file; far more liquid than median)
            _PERSYM_COST["_BTC_HEDGE_"] = float(BTC_HEDGE_COST_BPS)*1e-4
    return _PERSYM_COST, _PERSYM_COST_MED

_BTC_FUND = None
def _btc_funding_at(ot):
    """8h BTC perp funding rate active at `ot` (PIT .asof) from the cached series. A BTC long pays positive funding.
    Panel excludes BTC, so this is the only source for charging the BTC-hedge carry. 0.0 if unavailable."""
    global _BTC_FUND
    if _BTC_FUND is None:
        try:
            _f = pd.read_parquet("data/ml/cache/funding_BTCUSDT.parquet")
            _f["calc_time"] = pd.to_datetime(_f["calc_time"], utc=True)
            _BTC_FUND = _f.set_index("calc_time")["funding_rate"].sort_index()
        except Exception:
            _BTC_FUND = pd.Series(dtype=float)
    if len(_BTC_FUND) == 0: return 0.0
    v = _BTC_FUND.asof(ot)
    return float(v) if pd.notna(v) else 0.0
def cost_of(net_after: dict, prev_agg: dict) -> float:
    """Per-cycle cost (fraction of equity). Per-symbol real per-fill cost if DEPTH_COST_CSV set; else flat turn*0.5*COST."""
    keys = set(net_after) | set(prev_agg)
    if DEPTH_COST_CSV:
        cm, med = _persym_cost_map()
        return sum(abs(net_after.get(s,0)-prev_agg.get(s,0)) * cm.get(s, med) for s in keys)
    turn = sum(abs(net_after.get(s,0)-prev_agg.get(s,0)) for s in keys)
    return turn * 0.5 * COST
# depth-conditional bear de-gross band (toxic mid-bear). Default [-99,99] => BEAR_GROSS_MULT applies to ALL bear
# (backward-compatible). Set e.g. LO=-0.22 HI=-0.13 to de-gross ONLY the grinding mid-bear zone, keep deep/mild full.
BEAR_MID_LO = float(os.environ.get("BEAR_MID_LO","-99"))
BEAR_MID_HI = float(os.environ.get("BEAR_MID_HI","99"))
# BEAR DEPTH RAMP (2026-07-01): systematic (no hard threshold) bear gross control. The bear short is anti-alpha in
# the shallow GRIND (shorts stop falling faster than the sinking market / squeeze) and strongly +alpha in DEEP
# capitulation (short leg t+3.2). Instead of an on/off btc_30d cutoff, scale bear gross CONTINUOUSLY with drawdown
# depth: bg = FLOOR at depth<=D0 (mild grind), ramps linearly to 1.0 at depth>=D1 (deep). Monotonic, mechanism-
# grounded, robust to endpoint choice. depth = max(0,-btc_ret_30d). Off by default. Faithful (shared select_legs).
BEAR_DEPTH_RAMP = os.environ.get("BEAR_DEPTH_RAMP","0") == "1"
BEAR_DEPTH_D0 = float(os.environ.get("BEAR_DEPTH_D0","0.10"))   # drawdown at which bear gross = FLOOR (bear onset)
BEAR_DEPTH_D1 = float(os.environ.get("BEAR_DEPTH_D1","0.30"))   # drawdown at which bear gross = full (deep capitulation)
BEAR_DEPTH_FLOOR = float(os.environ.get("BEAR_DEPTH_FLOOR","0.0"))  # gross at/below D0
# AUTO-ADAPTIVE regime sizer (PIT report-card): bucket cycles by btc_ret_30d (width AUTO_BINW), throttle a bucket's
# new-sleeve gross to AUTO_THROTTLE once that bucket's TRAILING realized per-cycle PnL has gone net-negative (needs
# >= AUTO_MINLOOK settled cycles). Strictly point-in-time (uses only prior cycles), no hand-set band, adapts to drift.
# Default OFF => production byte-unchanged. iter21 = the deployable version of the iter19 screen.
AUTO_SIZER   = os.environ.get("AUTO_SIZER","0") == "1"
AUTO_BINW    = float(os.environ.get("AUTO_BINW","0.04"))
AUTO_MINLOOK = int(os.environ.get("AUTO_MINLOOK","20"))
AUTO_THROTTLE= float(os.environ.get("AUTO_THROTTLE","0.5"))
AUTO_RAND_SEED = int(os.environ.get("AUTO_RAND_SEED","0"))   # >0 => placebo: throttle RANDOM buckets (matched count)
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
# longdef_shortmr (iter2 of 2026-06-03 non-stop loop): high-vol-long ROOT CAUSE = idiosyncratic cascades.
# iter1 diag (within high-vol long candidates): bounce-vs-cascade discriminated by corr_to_btc_1d (IC +0.055,
# cascades LOW-corr), idio_vol_to_btc_1h (cascades HIGH, z+0.86), atr_pct (cascades HIGH, z+0.88). So among the
# top-N pred fallers pick the K most MARKET-LINKED / lowest-idio-vol (bounce-prone), skip idiosyncratic knives.
LONGDEF_FEATS = ["corr_to_btc_1d", "idio_vol_to_btc_1h", "atr_pct"]
SIDE_LONGDEF_N = int(os.environ.get("SIDE_LONGDEF_N", "12"))
# iter7 (2026-06-03): HARD-SKIP the extreme-idio-vol long candidates (cascade-prone per iter1) BEFORE
# the top-K pred pick — keeps the model's pred ordering among survivors (unlike iter2 which re-ranked).
# A nonlinear tail-skip the linear Ridge can't express. 1.0 = off; e.g. 0.80 drops top-20% idio-vol longs.
LONG_IDIO_SKIP_PCT = float(os.environ.get("LONG_IDIO_SKIP_PCT", "1.0"))
# iter12 (2026-06-03): LEG-SPECIFIC resid-rev tradeability gate for the LONG leg only (meta-labeling #172).
# iter11 proved resid_rev as a GLOBAL feature fixes A-long (-0.19→+0.41) but corrupts the 3 working legs.
# So apply it ONLY to the long pool: keep only "washed-out" names (resid_rev>=thr = recent BTC-residual LOSS),
# fallback to top-pred if <K pass. Base pred ranker preserved for shorts + other book. 0=off.
LONG_RESIDREV_GATE = os.environ.get("LONG_RESIDREV_GATE", "0") == "1"
LONG_RESIDREV_N = int(os.environ.get("LONG_RESIDREV_N", "3"))      # trailing bars (3=12h)
LONG_RESIDREV_THR = float(os.environ.get("LONG_RESIDREV_THR", "0.0"))
# Long-winner suppression: don't long recent RALLY names (ret_3d>thr) — they revert DOWN (momentum-long is a
# model error, not the edge). Cohort diag 2026-06-08: long picks ret_3d>+8% earn fwd -34bp Sharpe -0.90 vs
# losers +54/+1.33; filtering the ~14% winner-longs lifts long-leg Sharpe +0.19 (8/9 folds). 999=off.
LONG_MAX_RET3D = float(os.environ.get("LONG_MAX_RET3D", "999"))
# Regime-conditional LONG_MAX (2026-06-30): rocket-longs (ret_3d>thr) REVERT in side (veto helps +0.30 side Sh) but
# are the CAPITULATION BOUNCE in bear (veto cuts the +52bp deep-bear long cohort). Restrict the long-winner veto to
# these regimes. Empty = all regimes (backward compatible). Applies to LONG_MAX/LONG_MIN only.
LONG_MAX_RET3D_REGIMES = set(x for x in os.environ.get("LONG_MAX_RET3D_REGIMES", "").split(",") if x)
LONG_MIN_RET3D = float(os.environ.get("LONG_MIN_RET3D", "-999"))   # placebo/inverse: drop recent-LOSER longs (ret_3d<thr)
# Short-loser suppression (MIRROR of long-winner gate, short side): don't SHORT recent CRASH names (ret_3d<thr) —
# they violently BOUNCE, squeezing the short. Cohort diag 2026-06-13: short picks ret_3d<-0.20 earn fwd -31bp/pick
# (Sharpe -0.01) vs rockets ret_3d>+0.20 +76bp/pick (+0.13). Symmetric non-linear tail veto the linear ranker can't
# encode (same form as LONG_MAX_RET3D=+0.20). -999=off; principled symmetric value = -0.20.
SHORT_MIN_RET3D = float(os.environ.get("SHORT_MIN_RET3D", "-999"))
SHORT_MAX_RET3D = float(os.environ.get("SHORT_MAX_RET3D", "999"))   # placebo/inverse: drop recent-WINNER (rocket) shorts (ret_3d>thr)
# SHORT rocket-tail SIZE-TAPER (2026-06-30): rocket-shorts (ret_3d>THR) are +EV on AVERAGE (+23..+41bp side/bull) but
# carry a catastrophic squeeze left-tail (p5=-1193bp; the Nov-25 ZK blowup). A hard SHORT_MAX veto removes the tail
# but also the +EV (bleeds Feb/Apr/May -1.5..-1.9k each). Instead SHRINK the rocket-short weight (keep the +EV mean,
# cut the tail's portfolio impact). THR=999=off; principled THR=0.20, MULT~0.4-0.6. Faithful (in shared select_legs).
SHORT_RET3D_TAPER_THR = float(os.environ.get("SHORT_RET3D_TAPER_THR", "999"))
SHORT_RET3D_TAPER_MULT = float(os.environ.get("SHORT_RET3D_TAPER_MULT", "1.0"))
# GRIND-conditional idiosyncratic short (2026-07-01): in the shallow/grinding bear the pred short is anti-alpha
# (no market-wide reversion), BUT low-corr (idiosyncratic) names still revert on their own (reversion-IC by corr:
# Q1 lo-corr stays +; within-pool corr-selection lifts grind short +23bp, 3/4 months). Mechanism: when b30 in the
# grind band, re-pick the K shorts as the LOWEST-corr names within the bottom-POOL-by-pred. Bear-path only. 0=off.
SHORT_CORR_GRIND_POOL = int(os.environ.get("SHORT_CORR_GRIND_POOL", "0"))    # pred short-pool size to corr-select from (0=off)
SHORT_CORR_GRIND_THR = float(os.environ.get("SHORT_CORR_GRIND_THR", "-0.25"))  # apply when btc_ret_30d >= this (grind, not deep)
RAND_LONG_DROP_PCT = float(os.environ.get("RAND_LONG_DROP_PCT", "0"))   # placebo: drop TOP long in PCT% of cycles (random)
RAND_LONG_DROP_SEED = int(os.environ.get("RAND_LONG_DROP_SEED", "0"))
def _rand_drop_fires(ot):   # deterministic per (cycle, seed)
    if RAND_LONG_DROP_PCT <= 0: return False
    h = (int(ot.value // 10**9) * 2654435761 + RAND_LONG_DROP_SEED * 40503) % 100000
    return (h / 100000.0) < RAND_LONG_DROP_PCT
# 2026-06-14 CARRY-AWARE funding filters (use PIT-LAGGED funding — the panel's contemporaneous funding_rate LEAKS the
# forward move: long cohort spread +40->+24->+12 as you lag 0/4/8h, so SELECTION must use settled funding shifted
# FUND_LAG_BARS). The +4.22 backtest is PRICE-ONLY (credits ZERO funding); strategy book PAYS ~-3.8 bps/8h carry
# (-0.44 Sharpe). ASYMMETRY (PIT-clean cohorts): SHORT negative-funding names have BETTER alpha that GROWS with lag
# (Q0-rest +33.8 at lag2) => carry is BOUND to the short edge, filtering FAILS (monotone-neg at every floor, confirmed).
# LONG negative-funding (heavily-shorted/oversold) names: the price edge was look-ahead (collapses with lag) BUT the
# CARRY INCOME is real (funding persists) — the model ranks price-alpha and ignores carry, leaving long funding income
# on the table. LONG_FUND_CEIL drops longs whose PIT funding > ceil (crowded positive-funding longs) to TILT the leg
# toward carry-harvesting names. SHORT_FUND_FLOOR (BPS/8h, -999=off) kept as tested-negative. LONG_FUND_CEIL (+999=off).
SHORT_FUND_FLOOR = float(os.environ.get("SHORT_FUND_FLOOR", "-999"))   # BPS/8h; -999=off. drop shorts with PIT funding < floor
LONG_FUND_CEIL = float(os.environ.get("LONG_FUND_CEIL", "999"))        # BPS/8h; +999=off. drop longs with PIT funding > ceil
FUND_LAG_BARS = int(os.environ.get("FUND_LAG_BARS", "2"))             # bars to lag funding for PIT selection (2*4h=8h settled)
RAND_SHORT_DROP_PCT = float(os.environ.get("RAND_SHORT_DROP_PCT", "0"))   # matched placebo: drop TOP short in PCT% cycles
RAND_SHORT_DROP_SEED = int(os.environ.get("RAND_SHORT_DROP_SEED", "0"))
def _rand_short_drop_fires(ot):
    if RAND_SHORT_DROP_PCT <= 0: return False
    h = (int(ot.value // 10**9) * 2654435761 + RAND_SHORT_DROP_SEED * 92821 + 17) % 100000
    return (h / 100000.0) < RAND_SHORT_DROP_PCT
# vol-aware leg sizing (2026-06-02 root-cause: tail losses concentrate in high-idio-vol QUALITY names
# that pass all gates; volatility is ungated. Scale each leg's weight by inverse-vol, normalized to
# keep the SAME basket gross -> de-concentrates the tail without excluding liquid names).
# SIZING_MODE: equal (DEFAULT, = current 1/K behavior) | inv_vol | inv_sqrt_vol | inv_atr | volcap
SIZING_MODE = os.environ.get("SIZING_MODE", "equal")
SIZING_FEAT = os.environ.get("SIZING_FEAT", "idio_vol_to_btc_1h")  # PIT feature in panel
VOLCAP_PCTILE = float(os.environ.get("VOLCAP_PCTILE", "0.80"))     # for volcap mode: halve weight above this cross-sec pctile
# 2026-06-22 per-name concentration cap (live-DD root-cause: a single recurring short — ME in 12/18 cycles —
# stacks across ~4 of 6 active sleeves so one squeeze bleeds the whole book; validated: most-concentrated short
# quartile CVaR5 -358 vs -220 with NO mean-return gain). CONC_CAP = max |net weight| per name as a fraction of
# that SIDE's gross; excess water-fills onto the other same-side names -> per-side gross (hence dollar-neutrality)
# unchanged. 0 = OFF (production byte-identical). Applied to the aggregated net book each cycle.
CONC_CAP = float(os.environ.get("CONC_CAP", "0"))
# regime-scope the concentration cap (squeeze tail is bear-specific). Empty = all regimes; e.g. "bear" = bear-only.
CONC_CAP_REGIMES = set(x for x in os.environ.get("CONC_CAP_REGIMES", "").split(",") if x)
RANDCAP_SEED = int(os.environ.get("RANDCAP_SEED", "0"))   # for SIZING_MODE=randcap (volcap placebo)
# 2026-06-08 short-conviction tilt: within the K short legs, tilt weight toward higher conviction (more-negative
# short pred). Stacks MULTIPLICATIVELY on inv_vol then renormalizes to the same short gross. factor = conv_rank^TILT
# (most-conviction short -> rank K). 0 = OFF (production default; weights = pure inv_vol). iter3 found per-pick
# |pred| predicts SHORT success (+109bp Q4-Q0); iter4 gross replay convrank +1.50 Sharpe / 9-9 folds / placebo p100.
SHORT_CONV_TILT = float(os.environ.get("SHORT_CONV_TILT", "0"))
_SIZING_FEATS = ([] if SIZING_MODE == "equal" else [SIZING_FEAT])
# regime_switch: pred_disp (model conviction) threshold; >=THR -> model-long, else defensive-long.
SIDE_SWITCH_THR = float(os.environ.get("SIDE_SWITCH_THR", "1.0"))
RANDSHORT_SEED = int(os.environ.get("RANDSHORT_SEED", "1"))   # for randshort isolation test (short=random-K)
BTC_HEDGE_KEY = "_BTC_HEDGE_"   # sentinel key in weight dict for the synthetic BTC long
# V7: per-name confidence threshold gate (only trade alt when |pred| > threshold;
# else fallback to BTC). Applies in SIDE_MODE=confidence_btc_hedge.
PRED_THRESHOLD = float(os.environ.get("PRED_THRESHOLD", "0.5"))   # abs value in pred z-units
# EV/conviction floor for SIDE_MODE=default: after top-K selection, drop legs whose |pred| is below
# this (low-EV in thin regimes). 0.0 = OFF (production no-op). env-gated study knob.
PRED_FLOOR = float(os.environ.get("PRED_FLOOR", "0.0"))
# pred-disp adaptive gate: if per-cycle pred_disp is below the trailing-N percentile,
# the model is producing flat predictions = no tail-edge = go FLAT in side regime.
# (bull keeps trading regardless — mom_30d signal isn't affected by pred dispersion.)
DISP_GATE = os.environ.get("DISP_GATE", "0") == "1"
DISP_GATE_PCTILE = float(os.environ.get("DISP_GATE_PCTILE", "0.30"))
DISP_GATE_LOOKBACK = int(os.environ.get("DISP_GATE_LOOKBACK", "252"))
DISP_GATE_MIN_HISTORY = int(os.environ.get("DISP_GATE_MIN_HISTORY", "60"))
# REGIME GATE (2026-06-30): de-gross the WHOLE book when the strategy's trailing realized cross-sectional edge
# (top-K vs bottom-K by pred fwd return) has gone negative — detects momentum (edge-off) vs mean-rev (edge-on)
# regime. Edge persists at ~10-day scale (rho +0.45), so trailing edge predicts next-block edge. PIT: only uses
# cycles whose HOLD-bar fwd window has fully closed before the decision bar. threshold=0 (not fitted). Off-by-default.
REGIME_GATE = os.environ.get("REGIME_GATE", "0") == "1"
REGIME_GATE_W = int(os.environ.get("REGIME_GATE_W", "180"))        # trailing window (cycles); 180=30d, 120=20d
REGIME_GATE_FLOOR = float(os.environ.get("REGIME_GATE_FLOOR", "0.0"))  # gross mult when edge<0 (0=flat=validated; 0.3=toehold hedge)
REGIME_GATE_K = int(os.environ.get("REGIME_GATE_K", "2"))          # K per side for the edge thermometer
REGIME_GATE_MINHIST = int(os.environ.get("REGIME_GATE_MINHIST", "60"))  # min closed cycles before gating
REGIME_GATE_MODE = os.environ.get("REGIME_GATE_MODE", "binary")    # binary (full/floor at edge>0) | continuous (pctile-scaled)
REGIME_GATE_UNIV = os.environ.get("REGIME_GATE_UNIV", "full")      # full (all preds) | eligible (post maturity/liquidity filter)
SHORT_HYST_N = int(os.environ.get("SHORT_HYST_N", "0"))            # bull short pick-hysteresis band (0=off; retain incumbent if in top kS+N)
# VOLATILITY TARGETING (2026-07-01): adaptive gross overlay — size INVERSELY to trailing realized vol so RISK (not
# notional) is held constant. Motivation: trailing vol predicts forward drawdown (rho -0.66) even though per-cycle
# direction is noise (vol clusters/persists). gross *= clip(VOL_TARGET / trailing_vol_bps, FLOOR, CAP). trailing_vol
# = std of last WIN cycle equity %-changes (PIT, from stop.eq_hist — faithful across replay/cycle/decide). 0=off.
VOL_TARGET = float(os.environ.get("VOL_TARGET", "0"))              # target per-cycle vol (bps); 0=off. ~median trailing vol => avg mult~1
VOL_TARGET_WIN = int(os.environ.get("VOL_TARGET_WIN", "30"))       # trailing window (cycles) for realized-vol estimate
VOL_TARGET_FLOOR = float(os.environ.get("VOL_TARGET_FLOOR", "0.30"))  # min gross mult (cap leverage-down in calm)
VOL_TARGET_CAP = float(os.environ.get("VOL_TARGET_CAP", "1.00"))   # max gross mult (1.0 = de-gross-only; >1 allows scaling up)
def vol_target_mult(eq_hist):
    """PIT adaptive gross multiplier from trailing realized vol. eq_hist = deque of equity levels (through t-1)."""
    if VOL_TARGET <= 0 or len(eq_hist) < VOL_TARGET_WIN + 1:
        return 1.0
    e = np.asarray(list(eq_hist)[-(VOL_TARGET_WIN + 1):], dtype=float)
    rets = np.diff(e) / e[:-1]
    tv = float(np.std(rets)) * 1e4
    if tv <= 0:
        return 1.0
    return float(min(VOL_TARGET_CAP, max(VOL_TARGET_FLOOR, VOL_TARGET / tv)))
# DATA-DRIVEN bull short ranker: pred (current) underperforms vol features in bull (atr/idio/rvol IC ~2x pred &
# regime-stable; pred collapses in momentum regime). Rank bull shorts by this feature instead (highest=short).
BULL_SHORT_RANK = os.environ.get("BULL_SHORT_RANK", "pred")        # pred | rvol_7d | return_1d | atr_pct | comp_rvol | comp_ret1d
_BULL_RANK_MAP = {"rvol_7d":"rvol_7d","return_1d":"return_1d","atr_pct":"atr_pct","idio_vol_to_btc_1h":"idio_vol_to_btc_1h",
                  "comp_rvol":"rvol_7d","comp_ret1d":"return_1d","comp_atr":"atr_pct"}
_BULL_RANK_FEAT = _BULL_RANK_MAP.get(BULL_SHORT_RANK, "") if BULL_SHORT_RANK != "pred" else ""
REGIME_GATE_PLACEBO_SEED = int(os.environ.get("REGIME_GATE_PLACEBO_SEED", "-1"))  # >=0: de-gross SAME frac at random
# REGIME_GATE_SKIP_REGIMES (2026-06-30): regimes where the trailing-edge gate is NOT applied (mult forced to 1.0).
# Rationale: the gate assumes edge-PERSISTENCE (calibrated on side, rho+0.45). In BEAR the edge MEAN-REVERTS up —
# deep-capitulation alpha arrives exactly when the trailing edge is most negative, so the gate de-grosses into it
# and cuts bear from +1.69 to +0.79 Sharpe (and worsens maxDD). Exempting bear restores the (no-gate) bear book
# while keeping the validated side/bull gate. Faithful: applied identically in replay/cycle/decide.
REGIME_GATE_SKIP_REGIMES = set(x for x in os.environ.get("REGIME_GATE_SKIP_REGIMES", "").split(",") if x)
# Phase 2 CAUSE-based leading confirm: de-gross when BTC trend is unusually SMOOTH (momentum regime) — leads the
# lagging perf signal. Threshold = trailing-EXPANDING percentile of btc_smooth (PIT, non-fitted, adaptive).
REGIME_CAUSE = os.environ.get("REGIME_CAUSE", "0") == "1"
REGIME_CAUSE_PCTILE = float(os.environ.get("REGIME_CAUSE_PCTILE", "0.66"))  # de-gross when smoothness above this pctile
REGIME_CAUSE_MODE = os.environ.get("REGIME_CAUSE_MODE", "or")  # or (perf OR cause) | standalone (cause only)
# iter-012 vol-norm stop overlay
STOP_SKIP_REGIMES = set(x for x in os.environ.get("STOP_SKIP_REGIMES", "").split(",") if x)  # regimes where DD-stop is OFF (e.g. "bear")
STOP_K_SIGMA = float(os.environ.get("STOP_K_SIGMA", "2.0"))
STOP_G_FLOOR = float(os.environ.get("STOP_G_FLOOR", "0.40"))
STOP_SIGMA_WINDOW = int(os.environ.get("STOP_SIGMA_WINDOW", "180"))
STOP_WARMUP = 60
STOP_HEAL_FRAC = 0.5
STOP_TIMEOUT_BARS = 90
# Universe filter (deploy-spec: maturity≥180d + hygiene + liquidity_floor + dedup; iter-036)
MIN_HISTORY_DAYS = int(os.environ.get("CONVEXITY_MIN_HISTORY_DAYS", "180"))  # env-sweepable for the gating study
# PIT_DVOL=1 (DEFAULT) → per-cycle trailing-30d liquidity gate (honest, validated 2026-06-01: removes the
# end-of-sample dvol look-ahead; combined two-book +3.38 honest vs +3.71 look-ahead). Set 0 for legacy.
PIT_DVOL = os.environ.get("CONVEXITY_PIT_DVOL", "1") == "1"
HYGIENE_EXCLUDE = {"USDCUSDT","BUSDUSDT","TUSDUSDT","DAIUSDT","FDUSDUSDT",   # stables
                   "STABLEUSDT","STBLUSDT",                                   # stablecoin tokens (~0 crypto beta)
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
LIQ_FLOOR_DOLLAR_VOL_30D = float(os.environ.get("LIQ_FLOOR_DOLLAR_VOL_30D", "3000000"))  # $3M/day exec floor (env-gated)
DEDUP_CORR_THRESHOLD = 0.90                # drop high-corr names; keep longer-history one
# Liveness gate: drop DELISTED/HALTED names from the universe. A halted symbol's klines forward-fill a flat
# price -> zero returns -> rvol_7d≈0, the *calmest* possible, so the high-vol exclude never catches it and it
# lands in the low-vol book. The 30d-dvol floor eventually catches it (volume decays) but lags ~30d; this gate
# catches it in ~LIVENESS_WIN_DAYS. PIT (trailing window, .asof). (caught: VINEUSDT, delisted ~2026-04-28.)
# Entry-hour gate (cohort attribution: 12:00/16:00 UTC entries are weakest, 7/9 folds). Skip or down-weight.
SKIP_ENTRY_HOURS = set(int(x) for x in os.environ.get("SKIP_ENTRY_HOURS", "").split(",") if x.strip())
WEAK_ENTRY_HOURS = set(int(x) for x in os.environ.get("WEAK_ENTRY_HOURS", "8,12,16").split(",") if x.strip())
ENTRY_HOUR_SCALE = float(os.environ.get("ENTRY_HOUR_SCALE", "1.0"))   # down-weight WEAK_ENTRY_HOURS entries
ENTRY_HOUR_REGIMES = set(x for x in os.environ.get("ENTRY_HOUR_REGIMES", "").split(",") if x.strip())  # limit gate to these regimes
LIVENESS_GATE = os.environ.get("CONVEXITY_LIVENESS_GATE", "1") == "1"
LIVENESS_WIN_DAYS = int(os.environ.get("CONVEXITY_LIVENESS_WIN_DAYS", "7"))
LIVENESS_MAX_ZERO_FRAC = float(os.environ.get("CONVEXITY_LIVENESS_MAX_ZERO_FRAC", "0.85"))  # >85% flat days = dead
_LIVENESS_CACHE: dict = {}                 # sym -> date-indexed trailing zero-return fraction (set in precompute)

INITIAL_EQUITY = float(os.environ.get("CONVEXITY_EQUITY", "10000"))   # env-overridable; keeps ALL state
                                                                      # (equity/peak/eq_hist/stop) on one scale

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
    _avail = set(pd.read_parquet(PREDS, columns=None).columns) if False else None
    try:
        import pyarrow.parquet as _pq; _have = set(_pq.ParquetFile(PREDS).schema.names)
    except Exception:
        _have = set()
    if "mom" in _have: cols = cols + ["mom"]      # optional momentum col for longmom_shortmr mode
    for _c in ("pred_long", "pred_short"):        # iter14 meta-labeling: embedded per-leg rankers
        if _c in _have: cols = cols + [_c]
    if "ret_3d" in _have: cols = cols + ["ret_3d"]   # LIVE long-winner gate input (decide-preds carry it;
                                                     # the labeled PANEL lags the decide bar so its ret_3d is NaN)
    d = pd.read_parquet(PREDS, columns=cols)
    d["open_time"] = pd.to_datetime(d["open_time"], utc=True)
    d["exit_time"] = pd.to_datetime(d["exit_time"], utc=True)
    d = d[(d["open_time"].dt.hour%4==0) & (d["open_time"].dt.minute==0)].copy()
    if start is not None: d = d[d["open_time"] >= start]
    if PREDS_LONG:   # dual-pred: merge long-leg ranker's pred as 'pred_long'
        dl = pd.read_parquet(PREDS_LONG, columns=["symbol","open_time","pred"]).rename(columns={"pred":"pred_long"})
        dl["open_time"] = pd.to_datetime(dl["open_time"], utc=True)
        d = d.merge(dl, on=["symbol","open_time"], how="left")
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
                                       "mom30": (c/c.shift(MOM_WINDOW)-1).shift(1).values}))
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
        if PIT_DVOL:
            s = dvol_cache.get(sym); dv0 = float(s.asof(asof)) if (s is not None and len(s)) else np.nan
        else:
            dv0 = dvol_cache.get(sym, np.nan)
        rec = {"in_universe": False, "reason": "", "trailing_days": 0, "dvol30": dv0}
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
        dv = dv0
        if np.isfinite(dv) and dv < LIQ_FLOOR_DOLLAR_VOL_30D:
            rec["reason"] = f"liquidity_{dv:.0f}_<{LIQ_FLOOR_DOLLAR_VOL_30D:.0f}"; out[sym] = rec; continue
        if LIVENESS_GATE:                                  # drop delisted/halted (flat price over trailing window)
            zs = _LIVENESS_CACHE.get(sym)
            if zs is not None and len(zs):
                zf = float(zs.asof(asof))
                if np.isfinite(zf) and zf > LIVENESS_MAX_ZERO_FRAC:
                    rec["reason"] = f"dead_{zf:.2f}flat_>{LIVENESS_MAX_ZERO_FRAC}"; out[sym] = rec; continue
        rec["in_universe"] = True; out[sym] = rec
    return out


def precompute_dvol_cache(syms: list[str]) -> dict[str, float]:
    """One-shot 30d dollar-volume per sym, computed from the most-recent daily kline files.
    NOTE: this is END-OF-SAMPLE (a single value used for every cycle) → look-ahead in the liquidity
    gate (~+0.17 Sharpe, see docs/convexity_system_review_loop.md item 3). Set CONVEXITY_PIT_DVOL=1 to
    use the PIT trailing-30d version below instead (honest; recommended for any reported backtest)."""
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


def precompute_dvol_cache_pit(syms: list[str], last_n_files: int | None = None) -> dict[str, pd.Series]:
    """PIT liquidity gate: per-sym date-indexed series of trailing-30d MEAN DAILY dollar-volume.
    eligible_universe_at(asof) reads .asof(asof) → each cycle sees only data available at that time.
    last_n_files: live --cycle only needs the trailing window (~60d) — full-history reads are wasted there;
    replay/bootstrap pass None for the whole series."""
    # Optional pickle cache (loop speed-up; full-history replay only). No-op unless CONVEXITY_DVOL_CACHE_PKL set.
    _pkl = os.environ.get("CONVEXITY_DVOL_CACHE_PKL", "")
    if _pkl and last_n_files is None and Path(_pkl).exists():
        with open(_pkl, "rb") as fh: _d = pickle.load(fh)
        _LIVENESS_CACHE.update(_d.get("liveness", {}))
        return _d.get("dvol", {})
    cache = {}
    for sym in syms:
        sd = KLINES/sym/"5m"
        if not sd.exists(): continue
        files = sorted(sd.glob("*.parquet"))
        if last_n_files is not None: files = files[-last_n_files:]
        if len(files) < 5: continue
        try:
            df = pd.concat([pd.read_parquet(f, columns=["open_time","close","volume"]) for f in files], ignore_index=True)
            df["open_time"] = pd.to_datetime(df["open_time"], utc=True)
            di = df.set_index("open_time")
            daily = (di["close"]*di["volume"]).resample("1D").sum()
            trail = daily.rolling(30, min_periods=10).mean().dropna()   # mean daily $vol over trailing 30d
            if len(trail): cache[sym] = trail
            # liveness (same pass): trailing fraction of FLAT days (daily close unchanged = halted/delisted)
            if LIVENESS_GATE:
                dclose = di["close"].resample("1D").last().dropna()
                flat = (dclose.diff().abs() < 1e-12).astype(float)      # 1 = no price change that day
                zf = flat.rolling(LIVENESS_WIN_DAYS, min_periods=max(3, LIVENESS_WIN_DAYS//2)).mean().dropna()
                if len(zf): _LIVENESS_CACHE[sym] = zf
        except Exception:
            pass
    if _pkl and last_n_files is None:
        with open(_pkl, "wb") as fh: pickle.dump({"dvol": cache, "liveness": dict(_LIVENESS_CACHE)}, fh)
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


def effective_regime_series(upto=None) -> pd.Series:
    """Gap-free post-hysteresis regime per 4h bar, derived PURELY from compute_btc_30d so decide / settle /
    replay are bit-identical regardless of cycles.csv coverage or catch-up length after an outage (the old
    cycles.csv-seeded hysteresis left a gap on catch-up -> decide/settle could disagree at a transition and
    pick different legs/stop-skip). dropna() drops warmup (regime_for_cycle(NaN)='unknown'); index<=upto so a
    stray later bar cannot leak in."""
    b30 = compute_btc_30d().dropna()
    if upto is not None:
        b30 = b30[b30.index <= upto]
    if b30.empty:
        return pd.Series(dtype=object)
    raw = [regime_for_cycle(float(x)) for x in b30.values]
    eff = apply_hysteresis(raw, n=REGIME_HYSTERESIS_N)
    return pd.Series(eff, index=b30.index)


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


def _vol_scaled_weights(syms, gg, gross):
    """{sym: weight} summing to `gross`, scaled per SIZING_MODE. equal/missing-feat => gross/N each.
    Higher SIZING_FEAT (vol) -> smaller weight; normalized so basket gross is unchanged (beta-neut a,b hold approx)."""
    n = len(syms)
    if n == 0: return {}
    if SIZING_MODE == "equal" or SIZING_FEAT not in gg.columns:
        return {s: gross/n for s in syms}
    fmap = gg.set_index("symbol")[SIZING_FEAT].to_dict()
    vals = {s: fmap.get(s, np.nan) for s in syms}
    present = [v for v in vals.values() if np.isfinite(v) and v > 0]
    med = float(np.median(present)) if present else 1.0
    vol = {s: (v if (np.isfinite(v) and v > 0) else med) for s, v in vals.items()}
    if SIZING_MODE in ("inv_vol", "inv_atr"):
        raw = {s: 1.0/vol[s] for s in syms}
    elif SIZING_MODE == "inv_sqrt_vol":
        raw = {s: 1.0/np.sqrt(vol[s]) for s in syms}
    elif SIZING_MODE == "volcap":
        thr = float(gg[SIZING_FEAT].quantile(VOLCAP_PCTILE))
        raw = {s: (0.5 if vol[s] > thr else 1.0) for s in syms}
    elif SIZING_MODE == "randcap":
        # PLACEBO for volcap: down-weight (×0.5) a RANDOM matched-count subset instead of the high-vol one.
        # If volcap's edge is real vol-targeting, volcap should beat the randcap seed distribution.
        import hashlib as _hl
        k = int(round((1.0 - VOLCAP_PCTILE) * n))
        if k <= 0: raw = {s: 1.0 for s in syms}
        else:
            seed = int(_hl.sha1((",".join(sorted(syms)) + f"|{RANDCAP_SEED}").encode()).hexdigest()[:8], 16)
            capped = set(np.random.default_rng(seed).choice(list(syms), size=min(k, n), replace=False).tolist())
            raw = {s: (0.5 if s in capped else 1.0) for s in syms}
    else:
        raw = {s: 1.0 for s in syms}
    tot = sum(raw.values()) or 1.0
    return {s: gross*raw[s]/tot for s in syms}


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
    if regime == "bear":
        if BEAR_MODE == "flat": return {}              # production default — sit out bear
        if BEAR_MODE == "equal":                       # bear: EQUAL-weight L/S (dollar-neutral, NO beta reweighting)
            kbL, kbS = (BEAR_K, BEAR_K) if BEAR_K > 0 else (K_LONG, K_SHORT)
            gg = grp.dropna(subset=["pred"])
            if len(gg) < (kbL + kbS): return {}
            lpool = gg                                 # long-winner suppression (same as default path)
            if LONG_MAX_RET3D < 999 and "ret_3d" in gg.columns and (not LONG_MAX_RET3D_REGIMES or regime in LONG_MAX_RET3D_REGIMES):
                _k = gg[(gg["ret_3d"] <= LONG_MAX_RET3D) | gg["ret_3d"].isna()]
                if len(_k) >= kbL: lpool = _k
            if RAND_LONG_DROP_PCT > 0 and len(lpool) > kbL and _rand_drop_fires(grp["open_time"].iloc[0]):
                _rc = "pred_long" if "pred_long" in lpool.columns and lpool["pred_long"].notna().any() else "pred"
                lpool = lpool.drop(lpool[_rc].idxmax())
            if "pred_long" in lpool.columns and lpool["pred_long"].notna().sum() >= kbL:
                L = lpool.dropna(subset=["pred_long"]).nlargest(kbL, "pred_long")["symbol"].tolist()
            else:
                L = lpool.nlargest(kbL, "pred")["symbol"].tolist()
            spool = gg                                 # short-loser suppression (mirror of long-winner, bear path)
            if SHORT_MIN_RET3D > -999 and "ret_3d" in gg.columns:
                _k = gg[(gg["ret_3d"] >= SHORT_MIN_RET3D) | gg["ret_3d"].isna()]
                if len(_k) >= kbS: spool = _k
            if SHORT_MAX_RET3D < 999 and "ret_3d" in spool.columns:
                _k = spool[(spool["ret_3d"] <= SHORT_MAX_RET3D) | spool["ret_3d"].isna()]
                if len(_k) >= kbS: spool = _k
            _skey = "pred_short" if ("pred_short" in spool.columns and spool["pred_short"].notna().sum() >= kbS) else "pred"
            S = spool.dropna(subset=[_skey]).nsmallest(kbS, _skey)["symbol"].tolist() if _skey=="pred_short" else spool.nsmallest(kbS, "pred")["symbol"].tolist()
            # GRIND-conditional idiosyncratic short: in shallow bear (b30>=THR) pred short fails, but low-corr names
            # still revert -> re-pick the K shorts as lowest corr_to_btc within the bottom-POOL-by-pred.
            if SHORT_CORR_GRIND_POOL > 0 and "corr_to_btc_1d" in spool.columns and "btc_ret_30d" in grp.columns:
                if float(grp["btc_ret_30d"].iloc[0]) >= SHORT_CORR_GRIND_THR:
                    _sc = spool.dropna(subset=["corr_to_btc_1d", _skey])
                    _pool = _sc.nsmallest(min(SHORT_CORR_GRIND_POOL, len(_sc)), _skey)
                    if len(_pool) >= kbS:
                        S = _pool.nsmallest(kbS, "corr_to_btc_1d")["symbol"].tolist()
            # depth-conditional de-gross: by default BEAR_GROSS_MULT applies to ALL bear; if BEAR_MID_LO/HI are set,
            # it applies ONLY when btc_ret_30d is in the toxic mid-bear band [LO,HI), full size (1.0) elsewhere.
            # iter18 finding: bear is U-shaped — deep capitulation (Sh +10.6) & near-side bear (+10.5) thrive, the
            # GRINDING middle (~btc_ret_30d -0.22..-0.13, Sh -5) is the toxic zone that owns the maxDD.
            bg = BEAR_GROSS_MULT
            if (BEAR_MID_LO > -90 or BEAR_MID_HI < 90) and "btc_ret_30d" in grp.columns:
                b30 = float(grp["btc_ret_30d"].iloc[0])
                bg = BEAR_GROSS_MULT if (BEAR_MID_LO <= b30 < BEAR_MID_HI) else 1.0
            if BEAR_DEPTH_RAMP and "btc_ret_30d" in grp.columns:   # systematic continuous depth-ramp (no hard cutoff)
                depth = max(0.0, -float(grp["btc_ret_30d"].iloc[0]))
                r = (depth - BEAR_DEPTH_D0) / (BEAR_DEPTH_D1 - BEAR_DEPTH_D0) if BEAR_DEPTH_D1 > BEAR_DEPTH_D0 else 1.0
                bg = BEAR_DEPTH_FLOOR + (1.0 - BEAR_DEPTH_FLOOR) * min(1.0, max(0.0, r))
            if BEAR_HEDGE_BTC:                          # cut alt longs (negative-alpha), hedge net-short with BTC long
                bS = np.nanmean([betas_at_t.get(s, np.nan) for s in S])
                if not np.isfinite(bS) or bS <= 0: bS = 0.8
                w = {BTC_HEDGE_KEY: bg*bS}              # BTC long sized to neutralize the short basket's beta
                for s in S: w[s] = w.get(s, 0) - bg/kbS
                return w
            w = {}
            for s in L: w[s] = w.get(s, 0) + bg*BEAR_LONG_MULT/kbL  # bear long leg scaled by BEAR_LONG_MULT (negative-alpha in bear)
            for s in S: w[s] = w.get(s, 0) - bg/kbS     # -$ equal per name (short leg = the bear edge, kept full)
            return w
        if BEAR_MODE in ("side", "shortbias"): regime = "side"   # trade bear via the side mean-rev (beta-neut) path

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

    # SURGICAL hybrid (2026-06-03): long by MOMENTUM (fix the knife), short by MEAN-REV pred (keep +3.07 short). Beta-neutral.
    if regime == "side" and SIDE_MODE == "longmom_shortmr":
        gg = grp.dropna(subset=["pred"])
        if "mom30" not in gg.columns: return {}
        gg2 = gg.dropna(subset=["mom30"])
        if len(gg2) < K_LONG or len(gg) < K_SHORT: return {}
        L = gg2.nlargest(K_LONG, "mom30")["symbol"].tolist()      # long top-momentum (uptrends, not knives)
        S = gg.nsmallest(K_SHORT, "pred")["symbol"].tolist()    # short bottom-pred (mean-rev short-the-pumps, the +3.07 gem)
        bL = np.nanmean([betas_at_t.get(s, np.nan) for s in L]); bS = np.nanmean([betas_at_t.get(s, np.nan) for s in S])
        a = b = 1.0
        if np.isfinite(bL) and np.isfinite(bS) and bL > 0 and bS > 0: a = 2*bS/(bL+bS); b = 2*bL/(bL+bS)
        w = {}
        for s in L: w[s] = w.get(s, 0) + a/K_LONG
        for s in S: w[s] = w.get(s, 0) - b/K_SHORT
        return w

    # LONGDEF_SHORTMR (iter2 2026-06-03): long = DEFENSIVE pick among top-N pred fallers (most market-linked /
    # lowest idio-vol — bounce-prone, NOT idiosyncratic knives), short = bottom-K pred (the +3.07 mean-rev gem).
    if regime == "side" and SIDE_MODE == "longdef_shortmr":
        gg = grp.dropna(subset=["pred"])
        if len(gg) < (K_LONG + K_SHORT): return {}
        cand = gg.nlargest(SIDE_LONGDEF_N, "pred")
        if set(LONGDEF_FEATS).issubset(cand.columns) and cand[LONGDEF_FEATS].notna().all(axis=1).sum() >= K_LONG:
            dscore = (cand["corr_to_btc_1d"].rank(pct=True) - cand["idio_vol_to_btc_1h"].rank(pct=True)
                      - cand["atr_pct"].rank(pct=True))                    # high corr, low idio-vol, low atr
            L = cand.assign(_d=dscore).nlargest(K_LONG, "_d")["symbol"].tolist()
        else:
            L = cand.nlargest(K_LONG, "pred")["symbol"].tolist()           # fallback: plain top-pred
        S = gg.nsmallest(K_SHORT, "pred")["symbol"].tolist()
        bL = np.nanmean([betas_at_t.get(s, np.nan) for s in L]); bS = np.nanmean([betas_at_t.get(s, np.nan) for s in S])
        a = b = 1.0
        if np.isfinite(bL) and np.isfinite(bS) and bL > 0 and bS > 0: a = 2*bS/(bL+bS); b = 2*bL/(bL+bS)
        w = {}
        for s in L: w[s] = w.get(s, 0) + a/K_LONG
        for s in S: w[s] = w.get(s, 0) - b/K_SHORT
        return w

    # RANDSHORT isolation (2026-06-03): identical to default, but SHORT = RANDOM-K (not bottom-K by pred).
    # Isolates whether the bottom-K short SELECTION helps the book vs an uninformed random short (same long, weights, gross).
    if regime == "side" and SIDE_MODE == "randshort":
        gg = grp.dropna(subset=["pred"])
        if len(gg) < (K_LONG + K_SHORT): return {}
        gg = gg.sort_values("pred")
        L = gg.tail(K_LONG)["symbol"].tolist()
        pool = gg.iloc[:-K_LONG]
        seed = (RANDSHORT_SEED * 1000003 + int(pd.Timestamp(gg["open_time"].iloc[0]).value % 1000000)) % (2**31)
        S = pool.sample(min(K_SHORT, len(pool)), random_state=seed)["symbol"].tolist()
        bL = np.nanmean([betas_at_t.get(s, np.nan) for s in L]); bS = np.nanmean([betas_at_t.get(s, np.nan) for s in S])
        a = b = 1.0
        if np.isfinite(bL) and np.isfinite(bS) and bL > 0 and bS > 0: a = 2*bS/(bL+bS); b = 2*bL/(bL+bS)
        w = {}
        for s in L: w[s] = w.get(s, 0) + a/K_LONG
        for s in S: w[s] = w.get(s, 0) - b/K_SHORT
        return w

    # BEAR-MOMENTUM short (mirror of surgical): long by mean-rev pred, SHORT by lowest momentum (downtrends).
    if regime == "side" and SIDE_MODE == "longmr_shortmom":
        gg = grp.dropna(subset=["pred"])
        if "mom30" not in gg.columns: return {}
        gg2 = gg.dropna(subset=["mom30"])
        if len(gg) < K_LONG or len(gg2) < K_SHORT: return {}
        L = gg.nlargest(K_LONG, "pred")["symbol"].tolist()      # long mean-rev (oversold dips)
        S = gg2.nsmallest(K_SHORT, "mom30")["symbol"].tolist()    # short lowest momentum (downtrends = bear momentum)
        bL = np.nanmean([betas_at_t.get(s, np.nan) for s in L]); bS = np.nanmean([betas_at_t.get(s, np.nan) for s in S])
        a = b = 1.0
        if np.isfinite(bL) and np.isfinite(bS) and bL > 0 and bS > 0: a = 2*bS/(bL+bS); b = 2*bL/(bL+bS)
        w = {}
        for s in L: w[s] = w.get(s, 0) + a/K_LONG
        for s in S: w[s] = w.get(s, 0) - b/K_SHORT
        return w

    # BULL short-the-pumps + BTC-long hedge: bull short edge is +40@4h (over-extended alts revert) but the alt-long
    # leg is a BAD beta hedge in a melt-up (laggards under-rise) -> short beta bleeds -> squeeze. Hedge with a CLEAN
    # BTC long instead, to strip the market-driven part of the squeeze and keep the idiosyncratic reversion alpha.
    if regime == "bull" and BULL_DEEP_THR < 90 and "btc_ret_30d" in grp.columns \
       and float(grp["btc_ret_30d"].iloc[0]) >= BULL_DEEP_THR:
        return {}                                       # deep-bull squeeze zone -> sit flat (edge only in mild bull)
    if regime == "bull" and BULL_MODE == "short_btc_hedge":
        kbS = BULL_K if BULL_K > 0 else K_SHORT
        gg = grp.dropna(subset=["pred"])
        if len(gg) < kbS: return {}
        spool = gg
        if SHORT_MAX_RET3D < 999 and "ret_3d" in spool.columns:
            _k = spool[(spool["ret_3d"] <= SHORT_MAX_RET3D) | spool["ret_3d"].isna()]
            if len(_k) >= kbS: spool = _k
        # 5m entry-confirmation gate (mirror of default path): short only ROLLED-OVER names
        if BULL_ENTRY_NH30 and BULL_ENTRY_NH_COL in spool.columns:
            _nh = spool[BULL_ENTRY_NH_COL]
            if BULL_ENTRY_MODE == "strong" and BULL_ENTRY_OFF_COL in spool.columns:
                _k = spool[((_nh != 1) & (spool[BULL_ENTRY_OFF_COL] == 1)) | _nh.isna()]
            else:
                _k = spool[(_nh != 1) | _nh.isna()]
            if len(_k) >= kbS: spool = _k
        S = spool.nsmallest(kbS, "pred")["symbol"].tolist()
        bg = BULL_GROSS_MULT if BULL_GROSS_MULT > 0 else 1.0
        bS = np.nanmean([betas_at_t.get(s, np.nan) for s in S])
        if not np.isfinite(bS) or bS <= 0: bS = 0.8
        w = {BTC_HEDGE_KEY: bg*bS}                      # BTC long sized to neutralize the short basket's beta
        for s in S: w[s] = w.get(s, 0) - bg/kbS
        return w

    # Default path (and bull regime)
    if regime == "bull":
        key = "pred" if BULL_MODE == "sidealpha" else "mom30"
        do_bn = BULL_MODE in ("sidealpha", "betaneut_mom")
    else:  # side default
        key = "pred"; do_bn = SIDE_BETA_NEUT
    if regime == "bull" and BULL_K > 0: kL, kS = BULL_K, BULL_K
    elif regime == "bear" and BEAR_K > 0: kL, kS = BEAR_K, BEAR_K
    else: kL, kS = K_LONG, K_SHORT
    gg = grp.dropna(subset=[key])
    if len(gg) < 2*K: return {}
    if len(gg) < (kL + kS): return {}
    gg = gg.sort_values(key)
    # iter7: hard-skip cascade-prone (extreme idio-vol) names from the LONG pool only, keep pred ranking
    long_pool = gg
    if LONG_IDIO_SKIP_PCT < 1.0 and "idio_vol_to_btc_1h" in gg.columns:
        iv = gg["idio_vol_to_btc_1h"]
        thr = iv.quantile(LONG_IDIO_SKIP_PCT)
        keep = gg[(iv <= thr) | iv.isna()]
        if len(keep) >= K_LONG: long_pool = keep
    # iter12 leg-specific resid-rev gate: long only washed-out names (recent BTC-residual loss), base pred preserved
    if LONG_RESIDREV_GATE and "resid_rev" in long_pool.columns:
        keep = long_pool[long_pool["resid_rev"] >= LONG_RESIDREV_THR]
        if len(keep) >= K_LONG: long_pool = keep
    # 2026-06-08 long-winner suppression: drop recent rally names from the LONG pool (they revert down)
    if LONG_MAX_RET3D < 999 and "ret_3d" in long_pool.columns and (not LONG_MAX_RET3D_REGIMES or regime in LONG_MAX_RET3D_REGIMES):
        keep = long_pool[(long_pool["ret_3d"] <= LONG_MAX_RET3D) | long_pool["ret_3d"].isna()]
        if len(keep) >= kL: long_pool = keep
    # 2026-06-14 long carry-harvest: drop crowded positive-funding longs (PIT funding > ceil); keeps negative-funding
    # (heavily-shorted/oversold) names that PAY longs -> harvests carry income the price-ranking model ignores.
    if LONG_FUND_CEIL < 990 and "fund_pit" in long_pool.columns:
        keep = long_pool[(long_pool["fund_pit"]*1e4 <= LONG_FUND_CEIL) | long_pool["fund_pit"].isna()]
        if len(keep) >= kL: long_pool = keep
    if LONG_MIN_RET3D > -999 and "ret_3d" in long_pool.columns:      # inverse placebo: drop recent-LOSER longs
        keep = long_pool[(long_pool["ret_3d"] >= LONG_MIN_RET3D) | long_pool["ret_3d"].isna()]
        if len(keep) >= kL: long_pool = keep
    if RAND_LONG_DROP_PCT > 0 and len(long_pool) > kL and _rand_drop_fires(grp["open_time"].iloc[0]):
        rc = "pred_long" if "pred_long" in long_pool.columns and long_pool["pred_long"].notna().any() else key
        long_pool = long_pool.drop(long_pool[rc].idxmax())           # placebo: drop the TOP long candidate
    # short-loser suppression: drop recent CRASH names from the SHORT pool (they bounce/squeeze). Mirror of long-winner.
    short_pool = gg
    if SHORT_MIN_RET3D > -999 and "ret_3d" in gg.columns:
        keep = gg[(gg["ret_3d"] >= SHORT_MIN_RET3D) | gg["ret_3d"].isna()]
        if len(keep) >= kS: short_pool = keep
    if SHORT_MAX_RET3D < 999 and "ret_3d" in short_pool.columns:      # inverse placebo: drop recent-WINNER (rocket) shorts
        keep = short_pool[(short_pool["ret_3d"] <= SHORT_MAX_RET3D) | short_pool["ret_3d"].isna()]
        if len(keep) >= kS: short_pool = keep
    # carry-aware short veto (PIT-lagged funding): TESTED-NEGATIVE — short alpha is BOUND to carry (monotone-fails at
    # every floor; the negative-funding squeeze names carry the most alpha, robust with lag). Kept env-gated/off.
    if SHORT_FUND_FLOOR > -990 and "fund_pit" in short_pool.columns:
        keep = short_pool[(short_pool["fund_pit"]*1e4 >= SHORT_FUND_FLOOR) | short_pool["fund_pit"].isna()]
        if len(keep) >= kS: short_pool = keep
    if RAND_SHORT_DROP_PCT > 0 and len(short_pool) > kS and _rand_short_drop_fires(grp["open_time"].iloc[0]):
        short_pool = short_pool.drop(short_pool[key].idxmin())        # placebo: drop the TOP (most-negative-pred) short
    # 5m ENTRY-CONFIRMATION (bull only): skip shorts still making a 30m new high (still-pumping -> squeeze-prone)
    if regime == "bull" and BULL_ENTRY_NH30 and BULL_ENTRY_NH_COL in short_pool.columns:
        _nh = short_pool[BULL_ENTRY_NH_COL]
        if BULL_ENTRY_MODE == "strong" and BULL_ENTRY_OFF_COL in short_pool.columns:
            keep = short_pool[((_nh != 1) & (short_pool[BULL_ENTRY_OFF_COL] == 1)) | _nh.isna()]
        else:
            keep = short_pool[(_nh != 1) | _nh.isna()]
        # matched-fraction placebo: keep a RANDOM subset of the SAME size as the gate (tests if signal>random reduction)
        if BULL_ENTRY_PLACEBO_SEED >= 0 and kS <= len(keep) < len(short_pool):
            _ts = int(pd.Timestamp(grp["open_time"].iloc[0]).value)
            rng = np.random.default_rng(BULL_ENTRY_PLACEBO_SEED * 1_000_003 + (_ts % 1_000_003))
            keep = short_pool.iloc[np.sort(rng.choice(len(short_pool), size=len(keep), replace=False))]
        if len(keep) >= kS: short_pool = keep
    # iter13/14 dual-pred + meta-labels: long ranked by pred_long, short by pred_short (embedded or via PREDS_LONG).
    if "pred_long" in long_pool.columns and long_pool["pred_long"].notna().sum() >= kL:
        L = long_pool.dropna(subset=["pred_long"]).nlargest(kL, "pred_long")["symbol"].tolist()
    else:
        L = long_pool.tail(kL)["symbol"].tolist()
    if "pred_short" in short_pool.columns and short_pool["pred_short"].notna().sum() >= kS:
        S = short_pool.dropna(subset=["pred_short"]).nsmallest(kS, "pred_short")["symbol"].tolist()
    else:
        S = short_pool.head(kS)["symbol"].tolist()
    # DATA-DRIVEN bull short ranker override: rank by vol/return feature instead of pred (stronger & regime-stable)
    if regime == "bull" and _BULL_RANK_FEAT:
        _f = _BULL_RANK_FEAT
        if _f in short_pool.columns and short_pool[_f].notna().sum() >= kS:
            sp = short_pool.dropna(subset=[_f])
            if BULL_SHORT_RANK.startswith("comp_"):                       # composite: short low-pred AND high-feature
                sc = sp["pred"].rank(pct=True) - sp[_f].rank(pct=True)
                S = sp.assign(_sc=sc).nsmallest(kS, "_sc")["symbol"].tolist()
            else:                                                          # pure feature: short HIGHEST (high vol/return reverts)
                S = sp.nlargest(kS, _f)["symbol"].tolist()
    # SHORT pick-hysteresis (bull churn-reduction): retain last cycle's shorts if still in top-(kS+N) by pred,
    # fill remaining slots with fresh nsmallest. Cuts turnover without abandoning conviction. Bull-only, env-gated.
    if SHORT_HYST_N > 0 and regime == "bull":
        _band = short_pool.head(kS + SHORT_HYST_N)["symbol"].tolist()
        _inc = [s for s in globals().get("_PREV_SHORTS", []) if s in _band][:kS]
        S = (_inc + [s for s in S if s not in _inc])[:kS]
        globals()["_PREV_SHORTS"] = S
    # EV/conviction floor (env-gated; 0=off): drop selected legs whose base pred is below floor (long) /
    # above -floor (short) — sit out low-conviction picks in thin regimes. Missing pred => kept.
    if PRED_FLOOR > 0 and "pred" in gg.columns:
        pv = gg.set_index("symbol")["pred"].to_dict()
        L = [s for s in L if pv.get(s, 1e9) >= PRED_FLOOR]
        S = [s for s in S if pv.get(s, -1e9) <= -PRED_FLOOR]
    a = b = 1.0
    if do_bn:
        bL = np.nanmean([betas_at_t.get(s, np.nan) for s in L])
        bS = np.nanmean([betas_at_t.get(s, np.nan) for s in S])
        if np.isfinite(bL) and np.isfinite(bS) and bL > 0 and bS > 0:
            a = 2*bS/(bL+bS); b = 2*bL/(bL+bS)
    w = {}
    lw = _vol_scaled_weights(L, gg, a)          # long basket gross = a (vol-scaled within basket)
    sw = _vol_scaled_weights(S, gg, b)          # short basket gross = b
    # short-conviction tilt (env-gated; 0=off): multiply inv_vol weights by conv_rank^TILT, renorm to gross b.
    if SHORT_CONV_TILT > 0 and len(S) > 1:
        skey = "pred_short" if ("pred_short" in gg.columns and gg["pred_short"].notna().any()) else key
        pv = gg.set_index("symbol")[skey].to_dict()
        order = sorted(S, key=lambda s: pv.get(s, 0.0))           # ascending pred: most-negative (conviction) first
        fac = {s: float(len(S) - i) ** SHORT_CONV_TILT for i, s in enumerate(order)}  # rank K..1, ^TILT
        sw = {s: sw[s] * fac[s] for s in S}
        tot = sum(sw.values()) or 1.0
        sw = {s: b * sw[s] / tot for s in S}                      # renorm to same short gross
    _lmult = BULL_LONG_MULT if regime == "bull" else 1.0   # symmetric bull-long cut (small/noisy sample)
    if regime == "bull" and BULL_LONG_INSTRUMENT == "btc":
        # FAIR-COMPARISON mode: same long DOLLAR (BULL_LONG_MULT × alt-long gross) but allocated to BTC, not alts.
        w[BTC_HEDGE_KEY] = w.get(BTC_HEDGE_KEY, 0.0) + sum(lw[s]*_lmult for s in L)
    else:
        for s in L: w[s] = w.get(s, 0) + lw[s]*_lmult
    _r3 = gg.set_index("symbol")["ret_3d"].to_dict() if ("ret_3d" in gg.columns and SHORT_RET3D_TAPER_THR < 999) else {}
    for s in S:
        _sw = sw[s]
        if _r3:                                            # rocket-short size-taper: shrink (not veto) high-ret_3d shorts
            _v = _r3.get(s, np.nan)
            if np.isfinite(_v) and _v > SHORT_RET3D_TAPER_THR: _sw *= SHORT_RET3D_TAPER_MULT
        w[s] = w.get(s, 0) - _sw
    if regime == "bull" and BULL_GROSS_MULT != 1.0:        # de-gross/flat the bull sleeve (bull is -alpha at honest cost)
        w = {s: v*BULL_GROSS_MULT for s, v in w.items()}
    return w


SLEEVE_DECAY_TAU = float(os.environ.get("SLEEVE_DECAY_TAU", "0"))   # P6: 0=equal (current); >0 = exp(-age/tau) age-decay

def merge_panel_features(d):
    """Merge the PIT panel features selection/sizing need (defensive-tilt, vol-sizing SIZING_FEAT, bull short ranker,
    ret_3d gates). MUST be applied identically in run_replay/run_cycle/run_decide — otherwise the live paths silently
    fall back to defaults (equal-weight sizing, pred ranker). Live-faithfulness bug fix (2026-06-30)."""
    _need = list(dict.fromkeys(
        (DEF_FEATS if SIDE_MODE in ("long_defensive_basket_hedge", "regime_switch") else [])
        + (LONGDEF_FEATS if SIDE_MODE == "longdef_shortmr" else [])
        + (["idio_vol_to_btc_1h"] if LONG_IDIO_SKIP_PCT < 1.0 else [])
        + (["ret_3d"] if ((LONG_MAX_RET3D < 999 or LONG_MIN_RET3D > -999 or SHORT_MIN_RET3D > -999
                           or SHORT_MAX_RET3D < 999) and "ret_3d" not in d.columns) else [])
        + (["corr_to_btc_1d"] if (SHORT_CORR_GRIND_POOL > 0 and "corr_to_btc_1d" not in d.columns) else [])
        + ([_BULL_RANK_FEAT] if _BULL_RANK_FEAT else []) + _SIZING_FEATS))
    if _need:
        _pf = pd.read_parquet(PANEL, columns=["symbol", "open_time"] + _need)
        _pf["open_time"] = pd.to_datetime(_pf["open_time"], utc=True)
        d = d.merge(_pf, on=["symbol", "open_time"], how="left")
    return d


def cycle_raw_edge(g) -> float | None:
    """Regime thermometer for one cycle: top-K vs bottom-K (by pred) forward L/S edge. Used to build the trailing
    edge history that the REGIME_GATE de-grosses on. Shared by run_replay / run_cycle / run_decide so the live
    gate is byte-identical to the validated replay gate."""
    gg = g.dropna(subset=["pred", "return_pct"])
    if len(gg) < 2 * REGIME_GATE_K or gg["pred"].std() == 0:
        return None
    gg = gg.sort_values("pred")
    return float(gg.tail(REGIME_GATE_K)["return_pct"].mean() - gg.head(REGIME_GATE_K)["return_pct"].mean())


def regime_gross_mult(edge_hist, ot) -> float:
    """PIT whole-book gross multiplier from the trailing realized L/S edge. edge_hist: list of (ts, edge). Only
    cycles whose HOLD-bar fwd window has fully closed before the decision bar `ot` are used (exit-lagged). Binary:
    full size if trailing-W edge>0 else REGIME_GATE_FLOOR. Identical formula to run_replay's regime_mult."""
    if not REGIME_GATE or not edge_hist:
        return 1.0
    oti = pd.Timestamp(ot).value
    hold_ns = int(pd.Timedelta(hours=4 * HOLD).value)
    avail = [float(e) for (t, e) in edge_hist if pd.Timestamp(t).value + hold_ns <= oti]
    if len(avail) < REGIME_GATE_MINHIST:
        return 1.0
    tr = float(np.mean(avail[-REGIME_GATE_W:]))   # live path supports binary (validated/recommended) mode
    return 1.0 if tr > 0 else REGIME_GATE_FLOOR


def aggregate_active_sleeves(sleeves: deque, regime: str = None) -> dict:
    """net position = weighted sum over active sleeves. Default equal (1/HOLD). SLEEVE_DECAY_TAU>0 weights
    fresher sleeves more via exp(-age/tau), normalized to the same total gross (age 0 = newest=last appended).
    In bull (if BULL_HOLD<HOLD): aggregate only the freshest BULL_HOLD sleeves at full gross — bull edge reverses
    within the hold, so a shorter effective hold captures it before the squeeze-back."""
    net = {}; n = len(sleeves)
    if regime == "bull" and BULL_HOLD < HOLD and n > 0:
        for w in list(sleeves)[-BULL_HOLD:]:
            for s, wt in w.items(): net[s] = net.get(s, 0.0) + wt/BULL_HOLD
        return net
    if SLEEVE_DECAY_TAU > 0 and n:
        wts = np.array([np.exp(-((n-1-i))/SLEEVE_DECAY_TAU) for i in range(n)]); wts = wts/wts.sum()
        for sl, ww in zip(sleeves, wts):
            for s, wt in sl.items(): net[s] = net.get(s, 0.0) + wt*ww
    else:
        for w in sleeves:
            for s, wt in w.items(): net[s] = net.get(s, 0.0) + wt/HOLD
    return net


def apply_conc_cap(net: dict, cap_frac: float) -> dict:
    """Cap each name's |weight| to cap_frac * (that side's gross), water-filling the excess onto the OTHER
    same-side names in proportion to their weight. Each side's gross is preserved exactly, so the book stays
    dollar-neutral. cap_frac in (0,1]; values below 1/n_side just equalize that side."""
    if not net or cap_frac <= 0: return net
    # BTC_HEDGE is the synthetic market hedge, not a concentrated alpha name — exempt it. (A single-name side is
    # degenerate for water-fill: it would shrink the lone name to cap_frac×itself with nowhere to redistribute,
    # breaking dollar-neutrality — that's why BTC-long deployed at ~10% instead of 25%.)
    hedge = {k: net[k] for k in (BTC_HEDGE_KEY,) if k in net}
    work = {k: v for k, v in net.items() if k != BTC_HEDGE_KEY}
    out = dict(hedge)
    for positive in (True, False):
        side = {s: abs(w) for s, w in work.items() if (w > 0) == positive and abs(w) > 1e-12}
        if not side: continue
        cap = cap_frac * sum(side.values())
        for _ in range(50):
            over = {s: m for s, m in side.items() if m > cap + 1e-12}
            if not over: break
            excess = sum(m - cap for m in over.values())
            for s in over: side[s] = cap
            recv = {s: m for s, m in side.items() if m < cap - 1e-12}
            base = sum(recv.values())
            if base <= 1e-12: break                       # all at cap (cap_frac<=1/n) -> already equalized
            for s in recv: side[s] += excess * side[s] / base
        sgn = 1.0 if positive else -1.0
        for s, m in side.items(): out[s] = sgn * m
    return out


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
    def update(self, equity_pre_t: float, equity_post_t: float, bar_idx: int, regime: str | None = None) -> tuple[float, dict]:
        """Returns (gross_mult, diag_dict). PIT — uses equity through t-1 for sigma threshold.
        If `regime` is in STOP_SKIP_REGIMES, the de-gross is disabled (pro-cyclical vs mean-rev in volatile regimes)."""
        if regime is not None and regime in STOP_SKIP_REGIMES:
            self.eq_hist.append(equity_post_t)   # keep sigma history continuous
            return 1.0, dict(sigma=0.0, threshold=0.0, dd=0.0, peak=self.peak, engaged=False, engage_age=0, skipped=True)
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


class AutoRegimeSizer:
    """PIT regime report-card. Buckets cycles by btc_ret_30d (width binw); throttles a bucket's new-sleeve gross
    to `throttle` once that bucket's trailing realized per-cycle PnL is net-negative (>= minlook settled cycles).
    record() is called AFTER each cycle's PnL is known; mult() is called BEFORE -> strictly point-in-time.
    rand_seed>0 => placebo: pre-assign a RANDOM throttled/full label per bucket (matched throttle frequency)."""
    def __init__(self, binw, minlook, throttle, rand_seed=0):
        self.binw, self.minlook, self.throttle = binw, minlook, throttle
        self._sum, self._cnt = {}, {}
        self._rand = np.random.RandomState(rand_seed) if rand_seed > 0 else None
        self._rand_label = {}
    def _bin(self, b30):
        return int(np.floor(b30 / self.binw)) if np.isfinite(b30) else None
    def mult(self, b30):
        bi = self._bin(b30)
        if bi is None: return 1.0
        if self._rand is not None:                       # placebo: random (but persistent) bucket labels
            if bi not in self._rand_label: self._rand_label[bi] = self._rand.rand()
            return self.throttle if self._rand_label[bi] < 0.33 else 1.0
        n = self._cnt.get(bi, 0)
        if n >= self.minlook and (self._sum[bi] / n) < 0.0: return self.throttle
        return 1.0
    def record(self, b30, pnl_bps):
        bi = self._bin(b30)
        if bi is None: return
        self._sum[bi] = self._sum.get(bi, 0.0) + pnl_bps; self._cnt[bi] = self._cnt.get(bi, 0) + 1


# =====================================================================
# Replay engine (Stage A)
# =====================================================================

def run_replay(start: pd.Timestamp | None, end: pd.Timestamp | None) -> dict:
    log.info("loading preds + features")
    d = load_preds(start=start)
    if end is not None: d = d[d["open_time"] <= end]
    syms = sorted(d["symbol"].unique())
    log.info(f"replay: {len(d):,} rows × {len(syms)} syms × {d['open_time'].min().date()}→{d['open_time'].max().date()}")
    # iter12: PIT resid-rev = -(trailing sum of PAST per-bar realized residual alpha); higher = more washed-out
    if LONG_RESIDREV_GATE and "alpha_A" in d.columns:
        d = d.sort_values(["symbol","open_time"])
        d["resid_rev"] = -d.groupby("symbol")["alpha_A"].transform(
            lambda s: s.shift(1).rolling(LONG_RESIDREV_N).sum())
        log.info(f"resid-rev gate ON (N={LONG_RESIDREV_N} bars, thr={LONG_RESIDREV_THR}): "
                 f"{d['resid_rev'].notna().mean()*100:.0f}% rows have resid_rev")
    mom, betas = compute_mom30_and_beta(syms)
    btc30 = compute_btc_30d()
    d = d.merge(mom, on=["symbol","open_time"], how="left")
    d = d.merge(btc30.reset_index(), on="open_time", how="left").dropna(subset=["btc_ret_30d"])
    # defensive-tilt + vol-sizing + ranker features (PIT, from panel) — shared helper (live paths use it too)
    d = merge_panel_features(d)
    # PIT-lagged funding for carry-aware selection (contemporaneous funding_rate LEAKS the fwd move — see env block)
    if SHORT_FUND_FLOOR > -990 or LONG_FUND_CEIL < 990:
        _fp = pd.read_parquet(PANEL, columns=["symbol","open_time","funding_rate"]).sort_values(["symbol","open_time"])
        _fp["open_time"] = pd.to_datetime(_fp["open_time"], utc=True)
        _fp["fund_pit"] = _fp.groupby("symbol")["funding_rate"].shift(FUND_LAG_BARS)
        d = d.merge(_fp[["symbol","open_time","fund_pit"]], on=["symbol","open_time"], how="left")
    if CHARGE_FUNDING:
        # contemporaneous funding_rate = carry actually paid on the held book (cost accounting, not a feature)
        _fc = pd.read_parquet(PANEL, columns=["symbol","open_time","funding_rate"]).sort_values(["symbol","open_time"])
        _fc["open_time"] = pd.to_datetime(_fc["open_time"], utc=True)
        d = d.merge(_fc.rename(columns={"funding_rate":"fund_chg"}), on=["symbol","open_time"], how="left")
    # TAIL GATE: drop names with high walk-forward tail-risk score (per-cycle pctile) — squeeze/crash filter.
    if TAIL_SCORE_PARQUET and TAIL_SKIP_PCTILE < 1.0:
        _ts = pd.read_parquet(TAIL_SCORE_PARQUET, columns=["symbol","open_time","p_tail"])
        _ts["open_time"] = pd.to_datetime(_ts["open_time"], utc=True)
        d = d.merge(_ts, on=["symbol","open_time"], how="left")
        _cut = d.groupby("open_time")["p_tail"].transform(lambda s: s.quantile(TAIL_SKIP_PCTILE))
        _before = len(d)
        d = d[(d["p_tail"] <= _cut) | d["p_tail"].isna()].copy()
        log.info(f"TAIL_GATE pctile={TAIL_SKIP_PCTILE}: dropped {_before-len(d)}/{_before} high-tail-risk rows")
    if ENTRY_FLAG_PARQUET and BULL_ENTRY_NH30:
        _efc = ["symbol","open_time", BULL_ENTRY_NH_COL] + ([BULL_ENTRY_OFF_COL] if BULL_ENTRY_MODE=="strong" else [])
        _ef = pd.read_parquet(ENTRY_FLAG_PARQUET, columns=list(dict.fromkeys(_efc))).drop_duplicates(["symbol","open_time"])
        _ef["open_time"] = pd.to_datetime(_ef["open_time"], utc=True)
        d = d.merge(_ef, on=["symbol","open_time"], how="left")
    d["regime_raw"] = d["btc_ret_30d"].apply(regime_for_cycle)
    # gap-free hysteresis from the full btc_ret_30d series (decide/settle/replay identical) — see effective_regime_series
    reg_ser = effective_regime_series(upto=d["open_time"].max())
    d["regime"] = d["open_time"].map(reg_ser)
    log.info(f"hysteresis (N={REGIME_HYSTERESIS_N}, gap-free): raw {d['regime_raw'].value_counts().to_dict()} -> eff {d['regime'].value_counts().to_dict()}")

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

    # universe meta + dvol cache precomputed ONCE (also used by the eligible-universe regime thermometer below)
    log.info("precomputing universe meta + 30d dvol cache (one-shot)")
    univ_meta = precompute_universe_meta()
    dvol_cache = precompute_dvol_cache_pit(syms) if PIT_DVOL else precompute_dvol_cache(syms)

    # REGIME GATE: trailing realized cross-sectional edge -> whole-book gross multiplier (PIT, exit-time gated)
    regime_mult = {ot: 1.0 for ot in times}
    if REGIME_GATE:
        K = REGIME_GATE_K; hold_td = pd.Timedelta(hours=4*HOLD)
        re_ot, re_edge = [], []
        for ot in times:
            gg = by_t[ot].dropna(subset=["pred","return_pct"])
            if REGIME_GATE_UNIV == "eligible":                          # thermometer on the TRADED (eligible) universe
                _u = eligible_universe_at(univ_meta, ot, dvol_cache)
                gg = gg[gg["symbol"].isin({s for s, r in _u.items() if r["in_universe"]})]
            if len(gg) < 2*K or gg["pred"].std() == 0: continue
            gg = gg.sort_values("pred")
            re_ot.append(ot); re_edge.append(float(gg.tail(K)["return_pct"].mean() - gg.head(K)["return_pct"].mean()))
        re_edge = np.array(re_edge)
        re_exit_i = (pd.DatetimeIndex(re_ot) + hold_td).asi8             # exit time in int64 ns (UTC)
        trail_hist = []                                                # series of trailing-edge means (for continuous pctile)
        for ot in times:
            avail = re_edge[re_exit_i <= pd.Timestamp(ot).value]        # only cycles fully closed before decision bar
            if len(avail) >= REGIME_GATE_MINHIST:
                tr = avail[-REGIME_GATE_W:].mean()
                if REGIME_GATE_MODE == "continuous":
                    # size = trailing edge's percentile rank within its OWN PIT history -> [floor, 1] (non-fitted)
                    rank = (np.array(trail_hist) < tr).mean() if trail_hist else 1.0
                    regime_mult[ot] = REGIME_GATE_FLOOR + (1.0 - REGIME_GATE_FLOOR) * rank
                else:
                    regime_mult[ot] = 1.0 if tr > 0 else REGIME_GATE_FLOOR
                trail_hist.append(tr)
        if REGIME_GATE_PLACEBO_SEED >= 0:                              # matched-fraction placebo: de-gross same # at random
            deg = [ot for ot in times if regime_mult[ot] < 1.0]
            rng = np.random.default_rng(REGIME_GATE_PLACEBO_SEED)
            elig = [ot for ot in times if regime_mult.get(ot, 1.0) == 1.0 or True]
            pick = set(rng.choice(len(times), size=len(deg), replace=False).tolist())
            regime_mult = {ot: (REGIME_GATE_FLOOR if i in pick else 1.0) for i, ot in enumerate(times)}
        n_deg = sum(1 for ot in times if regime_mult[ot] < 1.0)
        log.info(f"REGIME_GATE W={REGIME_GATE_W} floor={REGIME_GATE_FLOOR} K={K}: de-grossed {n_deg}/{len(times)} cycles"
                 + (f" [PLACEBO seed={REGIME_GATE_PLACEBO_SEED}]" if REGIME_GATE_PLACEBO_SEED>=0 else ""))
        # persist the raw edge history so the LIVE path (run_cycle/run_decide via bootstrap) gates identically
        (STATE/"edge_hist.json").write_text(json.dumps([[str(t), float(e)] for t, e in zip(re_ot, re_edge)]))
    # Phase 2 CAUSE-based LEADING confirm: de-gross when BTC trend smoothness is above its trailing-expanding
    # percentile (momentum regime). PIT: smoothness uses only closes <= decision bar; threshold from past history.
    if REGIME_CAUSE:
        bc = load_close_4h("BTCUSDT")
        r30 = bc/bc.shift(180)-1; vol = bc.pct_change().rolling(180).std()
        smooth = (r30.abs()/(vol*np.sqrt(180))).dropna()
        s_ns = smooth.index.values.astype("datetime64[ns]").astype("int64")  # force ns (load_close_4h idx may be us/ms)
        s_v = smooth.values
        n_cause = 0
        for ot in times:
            oti = np.datetime64(pd.Timestamp(ot)).astype("datetime64[ns]").astype("int64")
            past = s_v[s_ns <= oti]                                   # PIT: smoothness known through decision bar
            if len(past) >= REGIME_GATE_MINHIST:
                thr = np.quantile(past[:-1], REGIME_CAUSE_PCTILE) if len(past) > 1 else np.inf
                cause_deg = past[-1] > thr                            # current smoothness above trailing pctile
                if REGIME_CAUSE_MODE == "standalone":
                    regime_mult[ot] = REGIME_GATE_FLOOR if cause_deg else 1.0
                elif cause_deg:                                      # "or": de-gross if EITHER perf or cause fires
                    regime_mult[ot] = REGIME_GATE_FLOOR
                if regime_mult[ot] < 1.0: n_cause += 1
        log.info(f"REGIME_CAUSE pctile={REGIME_CAUSE_PCTILE} mode={REGIME_CAUSE_MODE}: total de-grossed {sum(1 for ot in times if regime_mult[ot]<1.0)}/{len(times)}")

    # state
    active_sleeves: deque = deque(maxlen=HOLD)
    prev_agg: dict = {}
    equity = INITIAL_EQUITY
    stop = VolNormStop()
    cycles_rows, regime_rows, equity_rows, sleeves_rows, trades_rows, pred_rows, univ_rows = [], [], [], [], [], [], []
    log.info(f"  univ_meta: {len(univ_meta)} syms; dvol_cache: {len(dvol_cache)} syms")
    last_univ = {}
    last_regime = None
    sleeve_serial = 0
    sleeve_ids_active: deque = deque(maxlen=HOLD)
    auto = AutoRegimeSizer(AUTO_BINW, AUTO_MINLOOK, AUTO_THROTTLE, AUTO_RAND_SEED) if AUTO_SIZER else None

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
        # entry-hour gate: scale down weak UTC hours (08/12/16 < 00/04/20; long-leg bleed in active US/EU hours).
        # ENTRY_HOUR_REGIMES limits it to regimes where robust (side/bear; reverses in bull) — empty = all regimes.
        _hr_ok = (not ENTRY_HOUR_REGIMES) or (regime in ENTRY_HOUR_REGIMES)
        if _hr_ok and SKIP_ENTRY_HOURS and ot.hour in SKIP_ENTRY_HOURS:
            new_w = {}
        elif _hr_ok and ENTRY_HOUR_SCALE != 1.0 and ot.hour in WEAK_ENTRY_HOURS:
            new_w = {s: w*ENTRY_HOUR_SCALE for s, w in new_w.items()}
        # disp-gate override: if model dispersion is in the bottom pctile, side regime goes flat
        if DISP_GATE and regime == "side" and disp_skip.get(ot, False):
            new_w = {}
        # AUTO regime sizer (PIT): throttle this cycle's new-sleeve gross if its btc_ret_30d bucket has been losing
        _b30 = float(g["btc_ret_30d"].iloc[0]) if len(g) else float("nan")
        if auto is not None:
            _am = auto.mult(_b30)
            if _am != 1.0: new_w = {s: w*_am for s, w in new_w.items()}
        active_sleeves.append(new_w)
        sleeve_serial += 1
        # exit event for sleeve that fell out (if any)
        if len(sleeve_ids_active) == HOLD:
            old_id = sleeve_ids_active[0]
            # exit logged below after MtM
        sleeve_ids_active.append(sleeve_serial)

        # aggregate target across active sleeves
        net_target_raw = aggregate_active_sleeves(active_sleeves, regime)
        if CONC_CAP > 0 and (not CONC_CAP_REGIMES or regime in CONC_CAP_REGIMES):
            net_target_raw = apply_conc_cap(net_target_raw, CONC_CAP)
        gross_target = sum(abs(w) for w in net_target_raw.values())
        # stop overlay
        gross_mult, stop_diag = stop.update(equity, equity, bar_idx, regime)   # pre-MtM call; equity_post fills later
        if regime not in REGIME_GATE_SKIP_REGIMES:
            gross_mult *= regime_mult.get(ot, 1.0)                              # REGIME GATE de-gross overlay (exempt skip-regimes)
        gross_mult *= vol_target_mult(stop.eq_hist)                             # VOL-TARGET adaptive overlay (PIT, from trailing eq)
        net_after = {s: w*gross_mult for s, w in net_target_raw.items()}

        # cost from turnover (prev_agg -> net_after)
        all_keys = set(net_after) | set(prev_agg)
        turn = sum(abs(net_after.get(s, 0) - prev_agg.get(s, 0)) for s in all_keys)
        cost_bps_cycle = cost_of(net_after, prev_agg) * 1e4

        # mark to market using realized return_pct (4h-forward, already in panel as alpha_A's return)
        rmap = dict(zip(g["symbol"], g["return_pct"]))
        # Inject BTC forward return for the synthetic BTC hedge (if present in net_after)
        if BTC_HEDGE_KEY in net_after:
            rmap[BTC_HEDGE_KEY] = float(btc_fwd_map.get(ot, 0.0))
        gross_pnl = sum(net_after.get(s, 0) * rmap.get(s, 0.0)
                        for s in net_after if np.isfinite(rmap.get(s, 0.0)))
        # per-leg attribution (authoritative — actual sleeve-aggregated, regime-gated, beta-neutral positions).
        # raw = tradeable return_pct; alpha = BTC-beta-residualized alpha_A (matches "alpha-resid" leg numbers).
        amap = dict(zip(g["symbol"], g["alpha_A"]))
        _lk = [s for s in net_after if net_after[s] > 0]; _sk = [s for s in net_after if net_after[s] < 0]
        long_ret_bps   = sum(net_after[s]*rmap.get(s,0.0) for s in _lk if np.isfinite(rmap.get(s,np.nan)))*1e4
        short_ret_bps  = sum(net_after[s]*rmap.get(s,0.0) for s in _sk if np.isfinite(rmap.get(s,np.nan)))*1e4
        long_alpha_bps = sum(net_after[s]*amap.get(s,0.0) for s in _lk if np.isfinite(amap.get(s,np.nan)))*1e4
        short_alpha_bps= sum(net_after[s]*amap.get(s,0.0) for s in _sk if np.isfinite(amap.get(s,np.nan)))*1e4
        cost_unit = cost_of(net_after, prev_agg)
        # funding carry: a long pays +rate, a short receives -rate → pnl contribution = -net*rate (per interval)
        fund_unit = 0.0
        if CHARGE_FUNDING:
            fmap = dict(zip(g["symbol"], g["fund_chg"]))
            if BTC_HEDGE_KEY in net_after: fmap[BTC_HEDGE_KEY] = _btc_funding_at(ot)   # BTC perp carry (panel has no BTC)
            fund_unit = FUND_CYCLE_FRAC * sum(net_after[s] * fmap.get(s, 0.0)
                                              for s in net_after if np.isfinite(fmap.get(s, np.nan)))
        pnl_unit = gross_pnl - cost_unit - fund_unit
        equity_pre = equity
        equity = equity_pre * (1.0 + pnl_unit)
        if auto is not None: auto.record(_b30, pnl_unit*1e4)   # feed realized per-cycle PnL into the report-card (PIT)
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
            gross_pnl_bps=gross_pnl*1e4, cost_bps=cost_bps_cycle, fund_bps=fund_unit*1e4, turnover=turn,
            long_ret_bps=long_ret_bps, short_ret_bps=short_ret_bps,
            long_alpha_bps=long_alpha_bps, short_alpha_bps=short_alpha_bps,
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
    # persist the EXACT final state so live --cycle/--decide resume BYTE-IDENTICALLY (avoids the lossy bootstrap
    # reconstruction of prev_agg/stop from CSVs, which diverges at the catch-up boundary).
    if len(cycles_rows):
        _ehp = STATE/"edge_hist.json"
        _eh = json.loads(_ehp.read_text()) if _ehp.exists() else []
        _save_state(_state_from_replay_final(
            equity, stop.peak, list(stop.eq_hist), list(active_sleeves), prev_agg, sleeve_serial,
            cycles_rows[-1]["cycle_id"], cycles_rows[-1]["open_time"],
            dict(peak=stop.peak, engaged=stop.engaged, engage_dd=stop.engage_dd, engage_age=stop.engage_age,
                 trough=stop.trough, eq_hist=[float(x) for x in stop.eq_hist]),
            edge_hist=_eh))
    log.info(f"replay done: {summary}")
    return summary


POSITIONS = STATE/"positions.json"


def _save_state(state: dict):
    POSITIONS.write_text(json.dumps(state, default=str, indent=2))


def _load_state() -> dict | None:
    if not POSITIONS.exists(): return None
    return json.loads(POSITIONS.read_text())


def _assert_live_supported():
    """run_cycle/run_decide faithfully reproduce only the SUBSET of replay features wired into the live path.
    Hard-stop if a replay-only toggle is enabled — silent live/replay divergence is worse than refusing to run.
    (As live coverage grows, remove items here once they're implemented in cycle/decide.)"""
    bad = []
    if REGIME_GATE and REGIME_GATE_MODE != "binary": bad.append("REGIME_GATE_MODE=continuous (live gate is binary-only)")
    if REGIME_GATE and REGIME_GATE_UNIV != "full":   bad.append("REGIME_GATE_UNIV=eligible (live thermometer is full-universe)")
    if REGIME_CAUSE:                                 bad.append("REGIME_CAUSE (cause gate not in live)")
    if TAIL_SCORE_PARQUET and TAIL_SKIP_PCTILE < 1.0: bad.append("TAIL_SCORE gate (tailscore not merged in live)")
    if SHORT_FUND_FLOOR > -990:                      bad.append("SHORT_FUND_FLOOR (fund_pit not merged in live)")
    if BULL_ENTRY_NH30:                              bad.append("BULL_ENTRY_NH30 (5m entry flags not merged in live)")
    if DISP_GATE:                                    bad.append("DISP_GATE (disp-skip precompute is replay-only)")
    if LONG_FUND_CEIL < 999:                         bad.append("LONG_FUND_CEIL (fund_pit not merged in live)")
    if AUTO_SIZER:                                   bad.append("AUTO_SIZER (auto-regime sizer is replay-only)")
    if REGIME_GATE_PLACEBO_SEED >= 0:                bad.append("REGIME_GATE_PLACEBO_SEED (research placebo, replay-only)")
    if ENTRY_HOUR_SCALE != 1.0 or SKIP_ENTRY_HOURS:  bad.append("entry-hour controls (SKIP/WEAK_ENTRY_HOURS not in live)")
    if bad:
        raise SystemExit("LIVE-FAITHFULNESS GUARD: the live path does not implement these replay-only toggles, "
                         "so --cycle/--decide would DIVERGE from replay: " + "; ".join(bad) +
                         ". Unset them or wire them into run_cycle/run_decide first.")


def _state_from_replay_final(equity, peak, eq_hist, active_sleeves, prev_agg, sleeve_serial,
                             last_cycle_id, last_open_time, stop_state, edge_hist=None):
    return dict(
        equity=float(equity), peak=float(peak), eq_hist=[float(x) for x in eq_hist],
        active_sleeves=[{k: float(v) for k,v in w.items()} for w in active_sleeves],
        prev_agg={k: float(v) for k,v in prev_agg.items()},
        sleeve_serial=int(sleeve_serial),
        last_cycle_id=int(last_cycle_id),
        last_open_time=str(last_open_time) if last_open_time is not None else None,
        stop=stop_state,
        edge_hist=edge_hist if edge_hist is not None else [],   # REGIME_GATE trailing-edge history (PIT)
    )


def run_cycle() -> dict:
    """Single live cycle (or multi-cycle catch-up): load state, find new cycles in
    PREDS since last_open_time, process them, APPEND to state files, save positions.json."""
    _assert_live_supported()
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
    if (LONG_MAX_RET3D < 999 or LONG_MIN_RET3D > -999 or SHORT_MIN_RET3D > -999 or SHORT_MAX_RET3D < 999) \
            and "ret_3d" not in d.columns:   # long-winner / short-loser gates need ret_3d on the settle track too
        _r3 = pd.read_parquet(PANEL, columns=["symbol","open_time","ret_3d"])
        _r3["open_time"] = pd.to_datetime(_r3["open_time"], utc=True)
        d = d.merge(_r3, on=["symbol","open_time"], how="left")
    # live: betas/dvol for the new cycles need only ~30d trailing + the catch-up span; the fixed 45/70-day
    # windows re-read klines the new bars never use. Adaptive window auto-expands on a multi-day outage.
    # Validated bit-identical on the latest bar (beta 7e-16, dvol 1e-6) vs the 45/70-day read.
    win_days = 32 + (d["open_time"].max() - d["open_time"].min()).days + 2
    mom, betas = compute_mom30_and_beta(syms, lookback_days=win_days)
    btc30 = compute_btc_30d()
    btc_fwd_map = load_close_4h("BTCUSDT").pct_change().shift(-1).dropna().to_dict()  # BTC hedge fwd return (same as replay)
    d = d.merge(mom, on=["symbol","open_time"], how="left")
    d = d.merge(btc30.reset_index(), on="open_time", how="left").dropna(subset=["btc_ret_30d"])
    d = merge_panel_features(d)   # vol-sizing / ranker features — same as replay (else live falls back to equal-weight)
    if CHARGE_FUNDING:            # contemporaneous funding carry charged on the held book — same as run_replay
        _fc = pd.read_parquet(PANEL, columns=["symbol","open_time","funding_rate"])
        _fc["open_time"] = pd.to_datetime(_fc["open_time"], utc=True)
        d = d.merge(_fc.rename(columns={"funding_rate":"fund_chg"}), on=["symbol","open_time"], how="left")
    d["regime_raw"] = d["btc_ret_30d"].apply(regime_for_cycle)   # logged to cycles.csv; not used for the regime now
    # gap-free hysteresis from the full btc_ret_30d series — identical to what decide computed for the same bar
    reg_ser = effective_regime_series(upto=d["open_time"].max())
    d["regime"] = d["open_time"].map(reg_ser)
    log.info(f"hysteresis (gap-free btc-series): cycles regime {d['regime'].value_counts().to_dict()}")
    by_t = {ot: g for ot, g in d.groupby("open_time")}
    times = sorted(by_t.keys())

    # restore state
    active_sleeves: deque = deque([{k: float(v) for k,v in w.items()} for w in state["active_sleeves"]], maxlen=HOLD)
    prev_agg = {k: float(v) for k,v in state["prev_agg"].items()}
    edge_hist = list(state.get("edge_hist", []))   # REGIME_GATE trailing-edge history (PIT, exit-lagged)
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
    # live: PIT dvol for the new cycles needs only the same ~30d+catch-up window (full-history is wasted)
    # FAITHFULNESS: use the SAME full-history dvol/liveness cache as run_replay (load the shared pickle), not a
    # windowed recompute — a windowed/stale cache changes the eligible universe vs replay (n_universe drift).
    # The pickle must be refreshed (rebuilt) to include the latest klines before the live --cycle is run.
    dvol_cache = precompute_dvol_cache_pit(syms) if PIT_DVOL else precompute_dvol_cache(syms)

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
        net_target_raw = aggregate_active_sleeves(active_sleeves, regime)
        if CONC_CAP > 0 and (not CONC_CAP_REGIMES or regime in CONC_CAP_REGIMES):   # concentration cap (same as replay)
            net_target_raw = apply_conc_cap(net_target_raw, CONC_CAP)
        gross_target = sum(abs(w) for w in net_target_raw.values())
        gross_mult, stop_diag = stop.update(equity, equity, len(stop.eq_hist), regime)
        if regime not in REGIME_GATE_SKIP_REGIMES:
            gross_mult *= regime_gross_mult(edge_hist, ot)                       # REGIME GATE (same as run_replay)
        gross_mult *= vol_target_mult(stop.eq_hist)                              # VOL-TARGET (same as run_replay)
        net_after = {s: w*gross_mult for s, w in net_target_raw.items()}
        _re = cycle_raw_edge(g)                                                  # extend trailing-edge history (PIT)
        if _re is not None: edge_hist.append([str(ot), _re])
        all_keys = set(net_after) | set(prev_agg)
        turn = sum(abs(net_after.get(s,0) - prev_agg.get(s,0)) for s in all_keys)
        rmap = dict(zip(g["symbol"], g["return_pct"]))
        if BTC_HEDGE_KEY in net_after:                                  # BTC hedge fwd return — same as run_replay
            rmap[BTC_HEDGE_KEY] = float(btc_fwd_map.get(ot, 0.0))
        _nan_legs = [s for s in net_after if abs(net_after.get(s, 0)) > 1e-9 and not np.isfinite(rmap.get(s, np.nan))]
        if _nan_legs:   # held legs with no settled return are booked as 0% — surface incomplete labels, don't hide them
            log.warning(f"settle {ot}: {len(_nan_legs)} held legs have NaN return_pct, booked as 0%: {_nan_legs[:6]}")
        gross_pnl = sum(net_after.get(s,0) * rmap.get(s,0.0)
                        for s in net_after if np.isfinite(rmap.get(s,0.0)))
        # per-leg attribution — same as the replay path (line ~979). Was MISSING here, so catch-up/settle rows
        # logged NaN long/short_(ret|alpha)_bps from ~06-05 onward (pnl_bps was unaffected). alpha_A = realized
        # BTC-beta-residualized alpha; ret = tradeable return_pct. Empty leg -> sum() is 0.0, never NaN.
        amap = dict(zip(g["symbol"], g["alpha_A"]))
        _lk = [s for s in net_after if net_after[s] > 0]; _sk = [s for s in net_after if net_after[s] < 0]
        long_ret_bps    = sum(net_after[s]*rmap.get(s,0.0) for s in _lk if np.isfinite(rmap.get(s,np.nan)))*1e4
        short_ret_bps   = sum(net_after[s]*rmap.get(s,0.0) for s in _sk if np.isfinite(rmap.get(s,np.nan)))*1e4
        long_alpha_bps  = sum(net_after[s]*amap.get(s,0.0) for s in _lk if np.isfinite(amap.get(s,np.nan)))*1e4
        short_alpha_bps = sum(net_after[s]*amap.get(s,0.0) for s in _sk if np.isfinite(amap.get(s,np.nan)))*1e4
        cost_unit = cost_of(net_after, prev_agg)
        fund_unit = 0.0   # funding carry — same accounting as run_replay (was MISSING in live: overstated equity)
        if CHARGE_FUNDING and "fund_chg" in g.columns:
            fmap = dict(zip(g["symbol"], g["fund_chg"]))
            if BTC_HEDGE_KEY in net_after: fmap[BTC_HEDGE_KEY] = _btc_funding_at(ot)   # BTC perp carry (panel has no BTC)
            fund_unit = FUND_CYCLE_FRAC * sum(net_after[s]*fmap.get(s,0.0)
                                              for s in net_after if np.isfinite(fmap.get(s, np.nan)))
        equity_pre = equity; equity = equity_pre * (1.0 + gross_pnl - cost_unit - fund_unit)
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
            gross_pnl_bps=gross_pnl*1e4, cost_bps=cost_unit*1e4, fund_bps=fund_unit*1e4, turnover=turn,
            long_ret_bps=long_ret_bps, short_ret_bps=short_ret_bps,
            long_alpha_bps=long_alpha_bps, short_alpha_bps=short_alpha_bps,
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

    # APPEND to existing logs, dedup-safe so a crash between this append and _save_state below can't
    # produce duplicate rows when --cycle re-runs the same bar on restart. cycles/regime/equity are
    # 1-row-per-open_time → dedup on open_time; sleeves is many-rows-per-cycle → exact-row dedup.
    def _append(rows, fname, key="open_time"):
        if not rows: return
        df = pd.DataFrame(rows); path = STATE/fname
        if path.exists():
            comb = pd.concat([pd.read_csv(path), df], ignore_index=True)
            comb = comb.drop_duplicates(key, keep="last") if (key and key in comb.columns) \
                   else comb.drop_duplicates(keep="last")
            comb.to_csv(path, index=False)          # full rewrite (these logs are small)
        else:
            df.to_csv(path, index=False)
    _append(cycles_rows, "cycles.csv", "open_time")
    _append(regime_rows, "regime.csv", "open_time")
    _append(equity_rows, "equity.csv", "open_time")
    _append(sleeves_rows, "sleeves.csv", None)      # None → exact-row dedup (many rows per cycle)
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
             eq_hist=[float(x) for x in stop.eq_hist]),
        edge_hist=edge_hist[-(REGIME_GATE_W + HOLD + 50):])   # keep a bounded tail; gate only needs trailing W
    _save_state(final_state)
    log.info(f"appended {len(cycles_rows)} cycle(s); equity {state['equity']:.2f}→{equity:.2f} "
             f"({(equity-state['equity'])/max(state['equity'],1)*1e4:+.0f} bps)")
    return dict(new_cycles=len(cycles_rows), equity=float(equity),
                last_open_time=str(cycles_rows[-1]["open_time"]))


def run_decide() -> dict:
    """Predict-at-close DECIDE: select legs for the LATEST (unlabeled) bar without marking PnL or
    advancing/saving state. Writes decision.json (legs + per-symbol turnover vs the live book) so the
    HL probe can measure the REAL execution price at the bar — the settle --cycle books PnL 4h later.
    Faithful by construction: decide-preds == settle-preds (verified 3e-8), and select_legs is
    deterministic, so these legs equal what --cycle will trade when the bar settles."""
    _assert_live_supported()
    state = _load_state()
    if state is None:
        log.error("no positions.json — bootstrap first."); sys.exit(2)
    d = load_preds(start=pd.Timestamp(state["last_open_time"]) if state["last_open_time"] else None)
    if state["last_open_time"]:
        d = d[d["open_time"] > pd.Timestamp(state["last_open_time"])]
    if len(d) == 0:
        log.info(f"no unlabeled bar past {state['last_open_time']} to decide."); return {"decided": 0}
    ot = d["open_time"].max()                                  # the just-opened bar
    d = d[d["open_time"] == ot]
    syms = sorted(d["symbol"].unique())
    mom, betas = compute_mom30_and_beta(syms, lookback_days=34)
    btc30 = compute_btc_30d()
    d = d.merge(mom, on=["symbol", "open_time"], how="left")
    d = d.merge(btc30.reset_index(), on="open_time", how="left")
    d = merge_panel_features(d)   # vol-sizing / ranker features — same as replay (else decide falls back to equal-weight)
    # regime via the shared gap-free hysteresis series (compute_btc_30d-derived) — see effective_regime_series.
    # NOT a cycles.csv seed: that lags the settle path and leaves a gap on catch-up, so decide and settle could
    # disagree on the regime at a transition and pick different legs / stop-skip.
    reg_ser = effective_regime_series(upto=ot)
    regime = reg_ser.loc[ot] if ot in reg_ser.index else "side"
    # restore sleeve/stop state (read-only — we don't save)
    active_sleeves = deque([{k: float(v) for k, v in w.items()} for w in state["active_sleeves"]], maxlen=HOLD)
    prev_agg = {k: float(v) for k, v in state["prev_agg"].items()}
    equity = float(state["equity"])
    stop = VolNormStop()
    stop.peak = float(state["stop"]["peak"]); stop.engaged = bool(state["stop"]["engaged"])
    stop.engage_dd = float(state["stop"]["engage_dd"]); stop.engage_age = int(state["stop"]["engage_age"])
    stop.trough = float(state["stop"]["trough"])
    stop.eq_hist = deque([float(x) for x in state["stop"]["eq_hist"]], maxlen=STOP_SIGMA_WINDOW+1)
    univ_meta = precompute_universe_meta()
    dvol_cache = precompute_dvol_cache_pit(syms) if PIT_DVOL else precompute_dvol_cache(syms)  # full cache = replay-faithful
    univ = eligible_universe_at(univ_meta, ot, dvol_cache)
    eligible_syms = {s for s, r in univ.items() if r["in_universe"]}
    g_elig = d[d["symbol"].isin(eligible_syms)].copy()
    betas_at_t = {s: float(ser.loc[ot]) for s, ser in betas.items()
                  if ot in ser.index and np.isfinite(ser.loc[ot])}
    new_w = select_legs(g_elig, regime, betas_at_t)
    active_sleeves.append(new_w)
    net_target_raw = aggregate_active_sleeves(active_sleeves, regime)         # pass regime so STRAT_HOLD_BULL is honored
    if CONC_CAP > 0 and (not CONC_CAP_REGIMES or regime in CONC_CAP_REGIMES):  # concentration cap (same as replay/settle)
        net_target_raw = apply_conc_cap(net_target_raw, CONC_CAP)
    gross_mult, _ = stop.update(equity, equity, len(stop.eq_hist), regime)   # pass regime so STOP_SKIP_REGIMES
    if regime not in REGIME_GATE_SKIP_REGIMES:
        gross_mult *= regime_gross_mult(list(state.get("edge_hist", [])), ot)  # REGIME GATE (same as settle/replay)
    gross_mult *= vol_target_mult(stop.eq_hist)                                # VOL-TARGET (same as settle/replay)
    net_after = {s: w*gross_mult for s, w in net_target_raw.items()}          # is honored on decide as on settle
    all_keys = set(net_after) | set(prev_agg)
    turnover = {s: round(net_after.get(s, 0) - prev_agg.get(s, 0), 6) for s in all_keys
                if abs(net_after.get(s, 0) - prev_agg.get(s, 0)) > 1e-6}
    decision = dict(
        open_time=str(ot), regime=regime, equity=round(equity, 2), gross_mult=round(gross_mult, 4),
        n_eligible=len(eligible_syms),
        longs=sorted(s for s, w in new_w.items() if w > 0),
        shorts=sorted(s for s, w in new_w.items() if w < 0),
        net_after={s: round(w, 6) for s, w in net_after.items()},
        turnover=turnover)
    (STATE/"decision.json").write_text(json.dumps(decision, indent=2))
    log.info(f"DECIDE {ot} [{regime}]: L={[s.replace('USDT','') for s in decision['longs']]} "
             f"S={[s.replace('USDT','') for s in decision['shorts']]} | {len(turnover)} legs to execute "
             f"(equity {equity:.0f}, gross_mult {gross_mult:.2f}) → decision.json")
    return decision


def bootstrap_state_from_replay():
    """Snapshot final replay state into positions.json so --cycle can resume.
    run_replay now writes the EXACT state (incl. stop internals engage_dd/engage_age/trough + edge_hist) directly,
    so if positions.json already exists we KEEP it — reconstructing from CSVs is lossy (stop internals are not
    logged per-cycle). The CSV reconstruction below is a legacy fallback for when no exact state is present."""
    if POSITIONS.exists():
        log.info("positions.json already present (exact state from run_replay) — keeping it; --bootstrap-state is a "
                 "no-op. Delete positions.json first to force a (lossy) CSV reconstruction.")
        return
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
    # prev_agg must be the DEPLOYED book (what the next cycle trades against), not the raw aggregate: replay applies
    # conc_cap then a stop×regime-gate gross multiplier. Reconstruct that — apply conc_cap, then scale to the realized
    # gross_after_stop from cycles.csv (which already folds in stop + regime gate). Else turnover/cost are wrong on
    # the resume cycle (gross_after_stop != gross_target on ~400/1433 cycles). NOTE: run_replay also writes the EXACT
    # positions.json directly; this lossy path is only for legacy CSV-only bootstraps.
    _pa = aggregate_active_sleeves(deque(active, maxlen=HOLD), last.get("regime"))
    if CONC_CAP > 0 and (not CONC_CAP_REGIMES or last.get("regime") in CONC_CAP_REGIMES):
        _pa = apply_conc_cap(_pa, CONC_CAP)
    _g = sum(abs(w) for w in _pa.values()); _ga = float(last.get("gross_after_stop", _g))
    prev_agg = {s: w*(_ga/_g) for s, w in _pa.items()} if _g > 1e-12 else _pa
    _eh_path = STATE/"edge_hist.json"
    edge_hist = json.loads(_eh_path.read_text()) if _eh_path.exists() else []   # REGIME_GATE trailing-edge history
    state = _state_from_replay_final(
        last["equity_post"], e["peak"].max(), eq_hist, active, prev_agg,
        sl["sleeve_id"].max() if len(sl) else 0,
        last["cycle_id"], last["open_time"],
        dict(peak=float(e["peak"].iloc[-1]), engaged=bool(last["stop_engaged"]),
             engage_dd=0.0, engage_age=0, trough=float(last["equity_post"]),  # NOT logged per-cycle → placeholders
             eq_hist=[float(x) for x in eq_hist]),
        edge_hist=edge_hist)
    if bool(last["stop_engaged"]):
        log.warning("CSV reconstruction at a STOP-ENGAGED cutpoint: stop internals (engage_dd/engage_age/trough) "
                    "are not logged per-cycle, reconstructed as placeholders → stop-release timing may differ from "
                    "the exact replay state. Prefer run_replay's exact positions.json (do not delete it).")
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
    g.add_argument("--decide", action="store_true",
                   help="predict-at-close: select legs for the latest unlabeled bar → decision.json (no mark/save)")
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
    if args.decide:
        run_decide(); return
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
