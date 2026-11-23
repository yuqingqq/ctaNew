"""posting_opportunity.py - POSTING-OPPORTUNITY v1 (P-2026-005 MAKER-TRACE; session aea5bfb9, 2026-11-23): WHICH OBSERVABLE MARKET CONDITIONS
PRECEDE VISIBLE FRESH PRICE-LEVEL CREATIONS (improve / re_establish / deep / at_empty_touch) on the 19 BOOK markets (Oct 24-27), plus a SEPARATE
graded maker-identity layer - built EXACTLY to sections 1-3 of predictfun/reviews/maker_trace_20261122/posting_chain_contract.md (frozen before
computation, commit 8e09f3391), the ONE next priority of review 3f640ad9b (reviews/automatic/20261123_take_queue_review/report.md). USER goal: why
the top makers post (create / improve / re-establish price levels) given observable market conditions. DESCRIPTIVE on four consumed days (G = 4 <
5: point estimates, no intervals); p is a circular-shift tail fraction, not independent-day confidence. WORDING: an OBSERVED MARKET OPPORTUNITY,
not an identified maker decision time; adding size at an occupied price is not a fresh creation and its absence is never 'no posting'; no fit, no
replay, no model skill, no profitability claim.
S1 POPULATION every market x side (bid / ask, the book's YES-space convention) x second t = ws + 1000 k, k = 0 .. K - 1, K = (we - ws) // 1000
  (300: 19 x 300 x 2 rows), seconds with no fill and no creation included. Each market is loaded ONCE (chain_context.load_market). OUTCOMES =
  visible fresh-cell creations on that side whose creation interval (upd(f-1), upd(f)] lies INSIDE (t, t + 1000] (integer ms: t <= upd(f-1)
  and upd(f) <= t + 1000). Fresh cells = queue_front v1's E1 candidates exactly (queue_front.market_core on queue_front.market_book /
  market_fills: shown 0 in usable frame f - 1, > 0 in f, no gap; t_open < we); types = queue_front's NON-exclusive flags is_improve (strictly
  better than the side's best in f - 1) / is_re_establish (shown > 0 within the prior 5 s and emptied by matched fills) / is_deep (worse than
  the best) / is_at_empty_touch (own side empty in f - 1). Per row: E_any and E_<type> (event-second indicators, types overlap, never extra
  seconds), N_<type> (creation counts) and keys_<type> (exact cell-event keys mid|side|tick|fid) beside each indicator.
STATUSES (all counted): ROW status at the variant's read instant x: book_missing (no usable frame <= x) / book_stale (age > 1,000 ms) -> the
  row leaves every ratio of that variant. CREATION statuses (never events): straddle (an E1 interval crossing an anchor boundary: it cannot be
  assigned to one second), gap (queue_front status_only 'gap': a creation across a frame gap, > 1 s, a straddle by construction), unknown_cell
  (status_only 'unknown': the cell unknown in f - 1), outside_population (touches no row). READING (contract silent on rows TOUCHED by a status
  creation): the outcome is defined as a visible creation inside (t, t + 1000], so a touched row's outcome is never unknown; touched rows stay
  in the population with their counts (n_straddle / n_gap / n_unknown_cell) saved - excluding them would select rows on creations. Events of
  an E1 creation whose own-side touch was unknown in f - 1 (touch_unknown: improve / deep / at_empty undeterminable) count in 'any' only and
  are counted (type_unknown_events).
S2 STATE frozen comparison set at x = t (D0, primary) and x = t - 1000 (D1000: the WHOLE state lagged one second), from usable frames with upd
  <= x (chain_context.book / level_size), the BTC grid <= x and fills with hi <= x: spread (ask - bid ticks; 1 / 2 / 3-4 / >= 5; one-sided,
  crossed or unknown -> missing), depth_touch_own / depth_touch_opp (shown shares at the side's / the opposite side's best; 0 = empty side; 0 /
  (0, 50] / (50, 200] / > 200), depth_2_own (shares at own best - 1 tick (bid) / + 1 (ask), away from the spread; missing without an own
  touch), btc_5s_aligned (chain_context.align_cat(+1 bid / -1 ask, r_5s, MOVE_THR 1.0 bp): with / against / flat / missing), fair_gap (the
  EXISTING p_fair exactly as chain_take uses it: chain_take.Ctx.btc -> chain_context.btc_ret 'p_fair' = pf_mt_decision.btc_feats 'pf', no
  refit; bid 100 p_fair - best bid, ask best ask - 100 p_fair, cents; <= 0 / (0, 1] / (1, 3] / > 3; missing p_fair or own touch -> the feature
  status 'missing', excluded from the fair_gap contrast only), flow_5s (matched taker shares hitting this side's best: maker legs of the
  lifecycle fills table (every maker) at their RECORD's best tick on that side (bid max / ask min: the touch the taker hit), record hi in (x -
  5 s, x]; every leg of a record must carry the record's hi, else refused; 0 / > 0), time_left (take_logic.tl_codes bins). No feature's price
  level is chosen from the outcome.
  COMPARISONS Mantel-Haenszel person-time rate ratios (take_logic.mh_rr; event-seconds / seconds) of every outcome type for spread >= 2 vs 1,
  fair_gap > 1 vs <= 0, btc with vs flat, against vs flat, own touch depth 0-50 vs > 200, flow > 0 vs 0, time left <= 30 s vs (60, 240] s.
  STRATA market x side x time-left bin (the variant's); READING: the time-left contrast uses market x side (time left is its exposure). IMPROVE:
  only rows whose ANCHOR spread (D0, at t) is >= 2 enter improve ratios (opportunity availability; the same population at D1000, whose lag
  moves the compared state only), and the improve spread contrast is 3+ vs 2.
  NULL 500 circular shifts of each market-side's creation-event schedule (all types shifted together: their event-second vectors, a bijection
  on the 300 seconds, so no collision), offsets Generator(PCG64(SeedSequence([20261123, day, mid, side]))).integers(0, K) per market-side (a
  pilot market carries the full run's draws); the states stay fixed; p_hi / p_lo = (1 + #{null >= / <= obs}) / 501 (take_logic.tail_counts:
  ties within 1e-12 relative, NaN replicates on both sides).
  MARKET CONSISTENCY supported market = >= 30 exposed and >= 30 reference seconds and >= 1 event-second; its RR = the WITHIN-MARKET MH over its
  own strata (READING: the declared matching; the crude within-market RR is saved beside); RR > 1 / < 1 means beyond the 1e-12 relative tie.
  Day readings (day-pooled MH, supported / > 1 / < 1 counts) and every support count are saved.
  VERDICTS take_logic's rule: insufficient_support (< 5 supported markets or < 30 event-seconds in informative strata) | supported (RR > 1,
  p_hi <= 0.05, RR > 1 in >= 2/3 of the supported markets) | supported_lower (the mirror: RR < 1, p_lo <= 0.05, RR < 1 in >= 2/3) |
  not_supported (READING: take_logic's 'inconsistent' is folded in; its predicate is saved); timing_sensitive when the D1000 verdict differs.
  Multiplicity: 7 contrasts x 5 outcome types = 35, all listed, none selected.
S3 IDENTITY (separate; anonymous creations stay in every denominator). Candidate creator of an E1 creation = the address of its candidate first
  fill: queue_front_v2's earliest one-address set when that output exists (run only; ASSUMED interface: levels.csv bound by its receipt.json
  sha256, columns QF2_REQ, candidates = status in QF2_CAND with a holder; any other schema is refused), otherwise (and always in the pilot) the
  frozen v1 functions (queue_front._records / _new_life on market_core's own inputs, replicated and checked to reproduce market_core's status
  and holder on every E1 level) under the v2 STRADDLE RULE (contract 5 item 2: a settlement-only record joins the creation interval only if
  its WHOLE hard window lies inside (upd(f-1), upd(f)]; an E1 interval has no gap (<= 1 s) while every bounded window is >= 2 s - both checked
  - so no record can, which is exactly _new_life with upd(f-1) replaced by upd(f) in its settlement test: such records become untimed). The
  candidate exists only for v2 front_identified levels (unexplained_decrease / ambiguous_order / unknown / gap / no_fill -> none). LINK only
  where the lifecycle posting window of the candidate's filled hash (the lead leg's mk_hash; lifecycle orders: [min scenario post_lo (-1 =
  unbounded if any), S_NOPRIO post_hi]; equal to correction_v1/post_regrade_orders.csv for top-3 hashes, else refused) CONTAINS the creation
  interval (integer ms: lo <= upd(f-1) + 1 or unbounded, upd(f) <= hi). Grade = that order's post_regrade grade: G2 bound_constrained ->
  candidate_constrained, any other grade (or no regrade row: a non-top-3 hash) -> candidate_scenario; otherwise owner_unknown (reason kept).
  The eventual first filler is not automatically the creator: a candidate link with its grade. Top-3 context: shares of improve /
  re_establish / deep among candidate-linked top-3 creations vs all creations, counts per grade; INVENTORY only for candidate_constrained
  links: the recorded venue position at upd(f-1) (chain_context.position, legs with hi <= upd(f-1)); relation bid while short / ask while long
  = reducing, bid while long / ask while short = adding, flat = from_flat; timing_sensitive when an end of chain_context.pending's feasible
  range has another relation; a conditional estimate with counts, never pooled with unknown owners. Shares and counts only; no verdict on
  maker intent.
OUTPUTS write-once (take_logic_v2.write_out: a partial directory renamed at the end): rows.parquet (the population with features / outcomes /
  statuses / keys), events.csv (every fresh-cell candidate: key, status, types, identity), ratios.csv, consistency.csv, identity.csv,
  claims.json (computed predicates / verdicts only), receipt.json (sha256 of sources, inputs and outputs; CPU by stage; VmHWM; the identity
  dependency). run -> v2/chains/posting_opportunity; run --pilot <day> --out <dir> -> the pilot (prints the full-run CPU projection).
GUARDS MT_RUN=v2; measurement days only; MT_PO_MUT (the selftest's planted errors) and queue_front / take_logic mutants refused in production;
  the full run refuses unless the tool, its selftest, the contract and the reused tools are tracked and unmodified (the freeze) and the
  recorded pilot projection is <= 120 CPU s; fills_valued sha256 = the accepted binding; post_regrade_orders.csv = its receipt's sha256 and row
  count; VmHWM self-guard 1,450 MiB (pf_mt_common.GUARD_MIB, never raised); CPU self-guard.
Usage (run_light, MT_RUN=v2): posting_opportunity.py run | posting_opportunity.py run --pilot 20261024 --out <dir>
selftest: posting_opportunity_selftest.py"""
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RESEARCH = HERE.parents[2] / 'research'
for _p in (str(RESEARCH), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import pf_mt_common as C  # noqa: E402
import chain_context as CX  # noqa: E402
import chain_take as CT  # noqa: E402
import queue_front as QF  # noqa: E402
import take_logic as TL  # noqa: E402
import take_logic_v2 as TL2  # noqa: E402

need, Forbidden = C.need, C.Forbidden
NAN, INF = float('nan'), float('inf')
VERSION = 'posting_opportunity_v1'
CONTRACT_FILE = C.RECEIPT_V1 / 'posting_chain_contract.md'
CONTRACT_COMMIT = '8e09f3391'
REVIEW_FILE = C.REPO / 'predictfun' / 'reviews' / 'automatic' / '20261123_take_queue_review' / 'report.md'
REVIEW_COMMIT = '3f640ad9b'
TOOL_FILES = (HERE / 'posting_opportunity.py', HERE / 'posting_opportunity_selftest.py', CONTRACT_FILE, HERE / 'queue_front.py',
              HERE / 'chain_context.py', HERE / 'chain_take.py', HERE / 'take_logic.py', HERE / 'take_logic_v2.py')
DEP_FILES = (RESEARCH / 'pf_mt_common.py', RESEARCH / 'pf_mt_econ.py', RESEARCH / 'pf_mt_lifecycle.py', RESEARCH / 'pf_mt_decision.py',
             HERE / 'closes.py', QF.CONTRACT_FILE, TL.CONTRACT_FILE, REVIEW_FILE)
TOP3 = tuple(QF.TOP3)
TOP3_L = tuple(QF.TOP3_L)
N_BOOK = 19
PILOT_DAY = '20261024'
PILOT_RUN = 'r1'
FV_SHA256 = QF.FV_SHA256

# ------------------------------------------------------------------ declared constants (the contract's; none fitted) ------------------------------------------------------------------
STEP_MS = 1_000
K_EXPECT = 300
D_SENS = 1_000
VARIANTS = ('D0', 'D1000')
VSUF = dict(D0='0', D1000='1')
TYPES = ('any', 'improve', 're_establish', 'deep', 'at_empty_touch')
FLAG_OF = dict(improve='is_improve', re_establish='is_re_establish', deep='is_deep', at_empty_touch='is_at_empty_touch')
MIX_TYPES = ('improve', 're_establish', 'deep', 'at_empty_touch')
SIDES = ('bid', 'ask')
NT = int(QF.NT)
SPREAD_LAB = ('1', '2', '3-4', '>=5', 'missing')
DEPTH_LAB = ('0', '(0,50]', '(50,200]', '>200', 'missing')
BTC_LAB = ('with', 'against', 'flat', 'missing')
FG_LAB = ('<=0', '(0,1]', '(1,3]', '>3', 'missing')
FLOW_LAB = ('0', '>0')
TL_LAB = tuple(TL.TL_LAB)
ROW_ST = ('ok', 'book_missing', 'book_stale')
CRE_ST = ('event', 'straddle', 'gap', 'unknown_cell', 'outside_population')
FLOW_MS = 5_000
THR_5S = float(CX.MOVE_THR['r_5s'])
CONTRASTS = (('spread', 'spread >= 2 ticks', 'spread 1 tick'), ('fair_gap', 'fair_gap > 1 c', 'fair_gap <= 0 c'),
             ('btc_with', 'btc_5s with', 'btc_5s flat'), ('btc_against', 'btc_5s against', 'btc_5s flat'),
             ('depth_own', 'own touch depth 0-50 sh', 'own touch depth > 200 sh'), ('flow', 'flow_5s > 0', 'flow_5s = 0'),
             ('time_left', 'time left <= 30 s', 'time left in (60,240] s'))
CN_NAMES = tuple(c[0] for c in CONTRASTS)
IMPROVE_SPREAD = ('spread 3+ ticks', 'spread 2 ticks')
SEED, N_SHIFT = int(TL.SEED), int(TL.N_SHIFT)                 # 20261123, 500
MIN_EVENTS, MIN_MARKETS, MIN_MKT_S = TL.MIN_EVENTS, TL.MIN_MARKETS, TL.MIN_MKT_S   # 30, 5, 30
P_MAX, C_MAX = TL.P_MAX, TL.C_HI_MAX                           # 0.05; 1 + #{null beyond} <= 25
SHARE_NUM, SHARE_DEN = 2, 3
TIE = float(TL.TIE_REL)                                        # 1e-12: RR > 1 means RR > 1 + 1e-12
VERDICTS = ('supported', 'supported_lower', 'not_supported', 'insufficient_support', 'timing_sensitive')
IDENT = ('candidate_constrained', 'candidate_scenario', 'owner_unknown')
G2 = 'G2_bound_constrained'
GRADES = ('G0_unbounded_or_invalid', 'G1_conflict', 'G2_bound_constrained', 'G3_constrained_status_flagged', 'G4_scenario_only')
NO_GRADE = 'no_regrade_row'
SCEN = ('S_FIFO', 'S_FIFO_LO', 'S_NOPRIO')
POS_EPS = float(TL.POS_EPS)
GROUPS = tuple(QF.GROUPS)
RELATIONS = ('reducing', 'adding', 'from_flat')
MUTANTS = ('keep_spread1', 'assign_straddle', 'leak_1s', 'link_any_window', 'v1_straddle')
FOREIGN_MUT = ('MT_QF_MUT', 'MT_TL_MUT', 'MT_TL2_MUT')
CPU_CAP_RUN_S, CPU_CAP_PILOT_S = 150., 20.
PROJ_MAX_S = 120.
FRAMES = ('rows.parquet',)
CSV_FILES = ('events.csv', 'ratios.csv', 'consistency.csv', 'identity.csv')
OUT_FILES = FRAMES + CSV_FILES + ('claims.json', 'receipt.json')
RG_DT = dict(day=str, mid=np.int64, rank=np.int64, addr=str, mk_hash=str, y_side=str, tick=np.int64, post_lo_union_ms=np.int64,
             post_hi_ms=np.int64, grade=str)
QF2_REQ = ('day', 'mid', 'side', 'tick', 'frame', 't_open_ms', 't_create_ms', 'unit', 'status', 'candidate_addr', 'candidate_hash')
QF2_RENAME = {'candidate_addr': 'holder', 'candidate_hash': 'front_hash'}   # integrator alignment to queue_front_v2's actual levels.csv (pilot r1 header)
QF2_CAND = ('front_candidate',)
FEAT = ('st', 'age', 'spread', 'sp_c', 'd_own', 'd_own_c', 'd_opp', 'd_opp_c', 'd2', 'd2_c', 'r5', 'btc_c', 'pf', 'fg', 'fg_c', 'flow', 'flow_c',
        'tl', 'tl_c')
EV_COLS = ('key', 'day', 'mid', 'side', 'tick', 'frame', 'fid', 't_open_ms', 't_create_ms', 'k_lo', 'k_hi', 'row_k', 'status', 'is_improve',
           'is_re_establish', 'is_deep', 'is_at_empty_touch', 'touch_unknown', 'mechanism_v1', 'best_prev_tick', 'size_created', 'qf_status_v1',
           'qf_timing_v1', 'qf_status_v2', 'qf_timing_v2', 'qf2_status', 'cand_source', 'cand_addr', 'cand_group', 'cand_hash', 'cand_sid',
           'cand_mk_idx', 'cand_hi_ms', 'win_lo_ms', 'win_hi_ms', 'win_unbounded', 'win_contains', 'grade', 'identity', 'identity_reason', 'inv_q',
           'inv_pos_min', 'inv_pos_max', 'inv_relation', 'inv_timing_sensitive', 'inv_ledger_conditional')


# ------------------------------------------------------------------ small helpers ------------------------------------------------------------------
def _now():
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())


def _rel(p):
    p = Path(p)
    try:
        return str(p.relative_to(C.REPO))
    except ValueError:
        return str(p)


_sha, _json, _clean, hwm, guard, _git = QF._sha, QF._json, QF._clean, QF.hwm, QF.guard, QF._git


def paths():
    """Run-time paths on the MT_RUN roots (v2: C.RECEIPT = the receipt's v2/ directory)."""
    ch = C.RECEIPT / 'chains'
    return dict(out=ch / 'posting_opportunity', pilot_receipt=ch / f'posting_opportunity_pilot_{PILOT_DAY}_{PILOT_RUN}' / 'receipt.json',
                regrade=ch / 'correction_v1', qf2=ch / 'queue_front_v2')


def pilot_out(day):
    return C.RECEIPT / 'chains' / f'posting_opportunity_pilot_{day}_{PILOT_RUN}'


def mutant():
    """The selftest's planted-error switch MT_PO_MUT, read at call time; production refuses any value."""
    m = os.environ.get('MT_PO_MUT', '')
    need(m == '' or m in MUTANTS, f'MT_PO_MUT={m!r} is not a declared mutant {MUTANTS} (refused)')
    return m


def _gt1(x):
    return np.asarray(x, float) > 1. + TIE


def _lt1(x):
    return np.asarray(x, float) < 1. - TIE


def _mh(a, b, T1, T0):
    """Mantel-Haenszel person-time RR from per-stratum exposed / reference events and person-time (informative strata only)."""
    rr = TL.mh_rr(np.atleast_2d(np.asarray(a, float)), np.atleast_2d(np.asarray(b, float)), np.asarray(T1, float), np.asarray(T0, float))[0]
    return float(rr[0])


# ------------------------------------------------------------------ bins (contract section 2) ------------------------------------------------------------------
def spread_codes(sp):
    sp = np.asarray(sp, float)
    with np.errstate(invalid='ignore'):
        bad = ~np.isfinite(sp) | (sp < 0.5)
        return np.select([bad, sp < 1.5, sp < 2.5, sp < 4.5], [4, 0, 1, 2], 3).astype(np.int8)


def depth_codes(d):
    d = np.asarray(d, float)
    with np.errstate(invalid='ignore'):
        return np.select([np.isnan(d) | (d < 0), d <= 0, d <= 50, d <= 200], [4, 0, 1, 2], 3).astype(np.int8)


def fg_codes(g):
    g = np.asarray(g, float)
    with np.errstate(invalid='ignore'):
        return np.select([~np.isfinite(g), g <= 0, g <= 1, g <= 3], [4, 0, 1, 2], 3).astype(np.int8)


def flow_codes(x):
    return (np.asarray(x, float) > 0).astype(np.int8)


def relation(side, q):
    """bid (side 0) while short / ask while long = reducing; bid while long / ask while short = adding; |q| <= 1e-6 = from_flat."""
    q = np.asarray(q, float)
    long_, short = q > POS_EPS, q < -POS_EPS
    sd = np.asarray(side)
    red = ((sd == 0) & short) | ((sd == 1) & long_)
    add = ((sd == 0) & long_) | ((sd == 1) & short)
    return np.where(red, 'reducing', np.where(add, 'adding', np.where(np.isfinite(q), 'from_flat', ''))).astype(object)


# ------------------------------------------------------------------ inputs ------------------------------------------------------------------
def read_regrade(rdir, markets, hashes):
    """correction_v1/post_regrade_orders.csv (every top-3 filled order with its post grade) bound to receipt_post.json (sha256, row count);
    (day, mid, mk_hash) unique; every grade declared; every rank's address the ex-ante top 3 -> {(day, mid): {hash: row}}."""
    rdir = Path(rdir)
    po, pr = rdir / 'post_regrade_orders.csv', rdir / 'receipt_post.json'
    need(po.is_file() and pr.is_file(), f'input missing: {po} / {pr} (refused)')
    R = json.loads(pr.read_text())
    so = _sha(po)
    hashes[_rel(po)] = so
    hashes[_rel(pr)] = _sha(pr)
    need(((R.get('outputs') or {}).get('post_regrade_orders.csv') or {}).get('sha256') == so,
         "post_regrade_orders.csv differs from its receipt's sha256 (refused: mismatched artifacts)")
    hdr = pd.read_csv(po, nrows=0).columns
    miss = [c for c in RG_DT if c not in hdr]
    need(not miss, f'post_regrade_orders.csv lacks {miss} (refused)')
    D = pd.read_csv(po, usecols=list(RG_DT), dtype=RG_DT, keep_default_na=False)
    need(len(D) == int((R.get('rows') or {}).get('post_regrade_orders', -1)), "post_regrade_orders.csv rows differ from its receipt (refused)")
    need(bool(D['grade'].isin(GRADES).all()), f"post_regrade_orders.csv: a grade outside {GRADES} (refused)")
    key = pd.MultiIndex.from_arrays([D['day'].to_numpy(dtype=object), D['mid'].to_numpy(np.int64), D['mk_hash'].to_numpy(dtype=object)])
    need(key.is_unique, 'post_regrade_orders.csv: an order (day, mid, mk_hash) twice (refused)')
    rk = D['rank'].to_numpy(np.int64)
    ad = np.array([str(a).lower() for a in D['addr'].to_numpy(dtype=object)], dtype=object)
    need(bool(np.isin(rk, (1, 2, 3)).all()) and all(bool((ad[rk == r] == TOP3_L[r - 1]).all()) for r in (1, 2, 3)),
         "post_regrade_orders.csv: a rank outside 1-3 or an address unlike its rank's top-3 address (refused)")
    keep = {(str(d), int(m)) for d, m in markets}
    out = {k: {} for k in keep}
    for r in D.to_dict('records'):
        k = (str(r['day']), int(r['mid']))
        if k in out:
            out[k][str(r['mk_hash'])] = dict(grade=str(r['grade']), lo=int(r['post_lo_union_ms']), hi=int(r['post_hi_ms']), addr=str(r['addr']).lower(),
                                             side=str(r['y_side']), tick=int(r['tick']), rank=int(r['rank']))
    return out, dict(rows=int(len(D)), rows_run=int(sum(len(v) for v in out.values())), grades=dict(Counter(D['grade'].tolist())))


def read_qf2(qdir, markets, hashes):
    """queue_front_v2's levels.csv under this tool's ASSUMED interface (QF2_REQ; candidates = status in QF2_CAND with a holder), bound to its
    receipt.json sha256; any other schema or vocabulary is refused (align before the full run) -> {(day, mid): DataFrame}, info."""
    qdir = Path(qdir)
    lp, rp = qdir / 'levels.csv', qdir / 'receipt.json'
    need(lp.is_file() and rp.is_file(), f'queue_front_v2 output incomplete: {lp} / {rp} (refused)')
    R = json.loads(rp.read_text())
    sl = _sha(lp)
    hashes[_rel(lp)] = sl
    hashes[_rel(rp)] = _sha(rp)
    need(((R.get('outputs') or {}).get('levels.csv') or {}).get('sha256') == sl,
         "queue_front_v2 levels.csv differs from its receipt's sha256 (refused: mismatched artifacts)")
    hdr = pd.read_csv(lp, nrows=0).columns
    miss = [c for c in QF2_REQ if c not in hdr]
    need(not miss, f"queue_front_v2 levels.csv lacks {miss}: this tool's assumed interface is {QF2_REQ} (refused; align the adapter before the run)")
    D = pd.read_csv(lp, usecols=list(QF2_REQ), dtype=dict(day=str, side=str, unit=str, status=str, candidate_addr=str, candidate_hash=str),
                    keep_default_na=False).rename(columns=QF2_RENAME)
    keep = {(str(d), int(m)) for d, m in markets}
    sel = np.array([(str(d), int(m)) in keep for d, m in zip(D['day'].to_numpy(dtype=object), D['mid'].to_numpy(np.int64))], bool)
    D = D[sel].reset_index(drop=True)
    need(bool(D['status'].isin(QF2_CAND).any()), f"queue_front_v2 levels.csv carries no {QF2_CAND} status in the run's markets (refused: the "
                                                 "vocabulary differs from this adapter's assumption)")
    out = {k: D[(D['day'].to_numpy(dtype=object) == k[0]) & (D['mid'].to_numpy(np.int64) == k[1])].reset_index(drop=True) for k in keep}
    return out, dict(dir=_rel(qdir), levels_sha256=sl, rows_run=int(len(D)), candidates=int(D['status'].isin(QF2_CAND).sum()))


# ------------------------------------------------------------------ one market: flows, features ------------------------------------------------------------------
def flow_index(fl, cnt):
    """Per side: the record hi (sorted) and cumulative shares of the maker legs at their record's best tick on that side (bid max / ask min:
    the touch the taker hit); on-grid legs only (off-grid legs counted). Every leg of a record must carry the record's hi (refused otherwise)."""
    out = {}
    g = np.asarray(fl['grid'], bool)
    cnt['flow_legs_offgrid'] += int((~g).sum())
    for s in (0, 1):
        sel = np.flatnonzero(g & (np.asarray(fl['side']) == s))
        if not len(sel):
            out[s] = (np.zeros(0, np.int64), np.zeros(1))
            continue
        rid = pd.factorize(pd.Series(np.asarray(fl['sid'], dtype=object)[sel], dtype=object))[0].astype(np.int64)
        nr = int(rid.max()) + 1
        tk, hi, q = np.asarray(fl['tick'], np.int64)[sel], np.asarray(fl['hi'], np.int64)[sel], np.asarray(fl['q'], float)[sel]
        hmax, hmin = np.full(nr, np.iinfo(np.int64).min, np.int64), np.full(nr, np.iinfo(np.int64).max, np.int64)
        np.maximum.at(hmax, rid, hi)
        np.minimum.at(hmin, rid, hi)
        need(bool((hmax == hmin).all()), f'flow: {int((hmax != hmin).sum())} records whose legs carry different hi (refused: a record is one match)')
        best = np.full(nr, -1 if s == 0 else NT + 1, np.int64)
        (np.maximum if s == 0 else np.minimum).at(best, rid, tk)
        at = tk == best[rid]
        h, qq = hi[at], q[at]
        o = np.argsort(h, kind='stable')
        out[s] = (h[o], np.r_[0., np.cumsum(qq[o])])
        cnt[f'flow_records_{SIDES[s]}'] += nr
    return out


def flow_at(FI, s, x):
    """Shares of side s's touch legs with record hi in (x - 5 s, x]."""
    h, cq = FI[s]
    x = np.asarray(x, np.int64)
    return cq[np.searchsorted(h, x, 'right')] - cq[np.searchsorted(h, x - FLOW_MS, 'right')]


def features(M, ctx, FI, ws, we, K):
    """The frozen state set at x = T (D0) and x = T - 1000 (D1000) for both sides; every read at or before x (MT_PO_MUT=leak_1s: x + 1 s)."""
    mut = mutant()
    T = int(ws) + STEP_MS * np.arange(K, dtype=np.int64)
    lk = STEP_MS if mut == 'leak_1s' else 0          # planted error: every read one second LATER
    X = np.concatenate([T, T - D_SENS]).astype(np.int64) + lk
    n = len(X)
    Bk = CX.book(M, X)
    bst = np.asarray(Bk['status'], dtype=object)
    st = np.where(bst == 'no_frame', 1, np.where(bst == 'stale', 2, 0)).astype(np.int8)
    spread = np.asarray(Bk['spread'], float)
    B = ctx.btc(M, X)
    okb = np.asarray(B['code']) == 0
    pf = np.where(okb, np.asarray(B['pf'], float), NAN)
    r5 = np.where(okb, np.asarray(B['r_5s'], float), NAN)
    tl = np.asarray(CX.time_left(M, X), np.int64) / 1000.
    tlc = TL.tl_codes(tl)
    out = {}
    for s in (0, 1):
        own = np.asarray(Bk['bid_tick' if s == 0 else 'ask_tick'], float)
        osz = np.asarray(Bk['bid_sz' if s == 0 else 'ask_sz'], float)
        psz = np.asarray(Bk['ask_sz' if s == 0 else 'bid_sz'], float)
        fin = np.isfinite(own)
        t2 = np.where(fin, np.nan_to_num(own, nan=0., posinf=0., neginf=0.) + (-1. if s == 0 else 1.), -1.).astype(np.int64)
        okt = fin & (t2 >= 0) & (t2 < NT)
        L2 = CX.level_size(M, X, np.full(n, s, np.int64), np.where(okt, t2, 0))
        d2 = np.where(okt, np.asarray(L2['size'], float), NAN)
        bc = TL.align_codes(np.full(n, 1. if s == 0 else -1.), r5, THR_5S)
        with np.errstate(invalid='ignore'):
            fg = np.where(fin & np.isfinite(pf), (100. * pf - own) if s == 0 else (own - 100. * pf), NAN)
        fw = flow_at(FI, s, X)
        out[s] = dict(st=st, age=np.asarray(Bk['age_ms'], np.int64), spread=spread, sp_c=spread_codes(spread), d_own=osz, d_own_c=depth_codes(osz),
                      d_opp=psz, d_opp_c=depth_codes(psz), d2=d2, d2_c=depth_codes(d2), r5=r5, btc_c=bc, pf=pf, fg=fg, fg_c=fg_codes(fg), flow=fw,
                      flow_c=flow_codes(fw), tl=tl, tl_c=tlc)
    return T, out


# ------------------------------------------------------------------ one market: creations ------------------------------------------------------------------
def creations(day, mid, ws, K, LV):
    """Every fresh-cell candidate of queue_front.market_core -> its creation status, row second and cell-event key."""
    mut = mutant()
    n = len(LV)
    a = LV['t_open_ms'].to_numpy(np.int64) if n else np.zeros(0, np.int64)
    b = LV['t_create_ms'].to_numpy(np.int64) if n else np.zeros(0, np.int64)
    k_lo, k_hi = np.floor_divide(a - ws, STEP_MS), np.floor_divide(b - 1 - ws, STEP_MS)
    unit = LV['unit'].to_numpy(dtype=object) if n else np.zeros(0, object)
    qst = LV['status'].to_numpy(dtype=object) if n else np.zeros(0, object)
    e1 = unit == 'E1'
    touch = (k_hi >= 0) & (k_lo <= K - 1)
    inside = e1 & touch & (k_lo == k_hi)
    row_k = np.where(inside, k_lo, -1)
    if mut == 'assign_straddle':                       # planted error: a straddling interval assigned to the second holding upd(f)
        inside = e1 & touch & (k_hi >= 0) & (k_hi <= K - 1)
        row_k = np.where(inside, k_hi, -1)
    status = QF._first([~touch, ~e1 & (qst == 'gap'), ~e1, ~inside], ['outside_population', 'gap', 'unknown_cell', 'straddle'], 'event', n)
    side = LV['side'].to_numpy(dtype=object) if n else np.zeros(0, object)
    sd = np.where(side == 'bid', 0, 1).astype(np.int64)
    tick = LV['tick'].to_numpy(np.int64) if n else np.zeros(0, np.int64)
    fid = LV['fid'].to_numpy(np.int64) if n else np.zeros(0, np.int64)
    key = np.array([f'{mid}|{s_}|{t_}|{f_}' for s_, t_, f_ in zip(side.tolist(), tick.tolist(), fid.tolist())], dtype=object)
    flags = {ty: (LV[FLAG_OF[ty]].to_numpy(dtype=bool) if n else np.zeros(0, bool)) for ty in MIX_TYPES}
    tunk = LV['touch_unknown'].to_numpy(dtype=bool) if n else np.zeros(0, bool)
    return dict(n=n, a=a, b=b, k_lo=k_lo, k_hi=k_hi, row_k=row_k, status=status, side=sd, tick=tick, fid=fid, key=key, flags=flags, tunk=tunk, e1=e1)


def row_outcomes(CR, K):
    """Per row (side-major: bid k = 0..K-1, then ask): E / N / keys per type and the counts of status creations touching it."""
    out = {}
    ev = np.flatnonzero(CR['status'] == 'event')
    ridx = CR['side'][ev] * K + CR['row_k'][ev]
    for ty in TYPES:
        m = np.ones(len(ev), bool) if ty == 'any' else CR['flags'][ty][ev]
        N = np.bincount(ridx[m], minlength=2 * K).astype(np.int32) if len(ev) else np.zeros(2 * K, np.int32)
        keys = np.full(2 * K, '', dtype=object)
        if m.any():                                  # the keys of one row joined with ';' (a key itself carries '|')
            sub = pd.Series(CR['key'][ev][m], dtype=object).groupby(ridx[m]).agg(lambda s: ';'.join(sorted(s)))
            keys[sub.index.to_numpy(np.int64)] = sub.to_numpy(dtype=object)
        out['N_' + ty] = N
        out['E_' + ty] = (N > 0).astype(np.int8)
        out['keys_' + ty] = keys
    for stn in ('straddle', 'gap', 'unknown_cell'):
        ix = np.flatnonzero(CR['status'] == stn)
        c_ = np.zeros(2 * K, np.int64)
        for s in (0, 1):                             # one difference array per side: the rows k_lo .. k_hi (clipped) each status touches
            ss = ix[CR['side'][ix] == s]
            if len(ss):
                dd = np.zeros(K + 1, np.int64)
                np.add.at(dd, np.maximum(CR['k_lo'][ss], 0), 1)
                np.add.at(dd, np.minimum(CR['k_hi'][ss], K - 1) + 1, -1)
                c_[s * K:(s + 1) * K] = np.cumsum(dd[:-1])
        out['n_' + stn] = c_.astype(np.int32)
    tu = ev[CR['tunk'][ev]]
    out['n_type_unknown'] = np.bincount(CR['side'][tu] * K + CR['row_k'][tu], minlength=2 * K).astype(np.int32) if len(tu) else np.zeros(2 * K, np.int32)
    return out


# ------------------------------------------------------------------ one market: the identity layer (contract section 3) ------------------------------------------------------------------
def fallback_fronts(day, mid, ws, we, bk, fl, LV):
    """queue_front v1's frozen functions on market_core's own inputs (replicated here, checked to reproduce market_core's E1 status / holder /
    front timing exactly), then the v2 STRADDLE RULE: _new_life with upd(f-1) replaced by upd(f) in its settlement-only test (the premises that
    make this identical to 'the whole hard window inside (upd(f-1), upd(f)]' - E1 intervals <= 1 s, bounded windows >= 2 s - are checked)
    -> per E1 level (LV's E1 order): st_v1, timing_v1, st_v2, lead_v2 (leg index, -1 none), timing_v2. MT_PO_MUT=v1_straddle keeps v1."""
    mut = mutant()
    E1 = LV[LV['unit'] == 'E1'] if len(LV) else LV
    nE = int(len(E1))
    z = dict(nE=nE, st_v1=np.zeros(0, object), timing_v1=np.zeros(0, object), st_v2=np.zeros(0, object), lead_v2=np.zeros(0, np.int64),
             timing_v2=np.zeros(0, object), n_v1_v2_changed=0)
    if nE == 0:
        return z
    t, S, gap, F = bk['t'], bk['S'], bk['gap'], int(bk['F'])
    n = int(fl['n'])
    hi, lo, g = fl['hi'], fl['lo'], fl['grid']
    cellA = np.where(g, fl['side'] * NT + fl['tick'], -1).astype(np.int64)
    sidc = pd.factorize(pd.Series(fl['sid'], dtype=object), sort=True)[0].astype(np.int64) if n else np.zeros(0, np.int64)
    base = np.int64(min([ws - 10 ** 7] + ([int(t[0])] if F else []) + ([int(hi.min())] if n else [])) - 1)

    def off(x):
        return np.clip(np.asarray(x, np.int64) - base, 0, QF.KEY - 1)
    pos = S > 0
    nanS = np.isnan(S)
    ff, cc = np.nonzero(pos[1:] & ~pos[:-1])
    ff, cc = ff.astype(np.int64) + 1, cc.astype(np.int64)
    inpop = t[ff - 1] < we
    ff, cc = ff[inpop], cc[inpop]
    isg = gap[ff]
    pnan = nanS[ff - 1, cc] & ~isg
    fresh = ~isg & ~pnan
    f, c = ff[fresh], cc[fresh]
    sd_lv = np.where(E1['side'].to_numpy(dtype=object) == 'bid', 0, 1)
    need(len(f) == nE and np.array_equal(f, E1['frame'].to_numpy(np.int64)) and np.array_equal(c, sd_lv * NT + E1['tick'].to_numpy(np.int64)),
         f'market {day} {mid}: the replicated E1 candidates differ from queue_front.market_core (refused)')
    acode = pd.factorize(pd.Series(fl['addr'], dtype=object))[0].astype(np.int64) if n else np.zeros(0, np.int64)
    R = QF._records(t, S, gap, F, fl, g, cellA, sidc, acode, base)
    nbrk = QF._acc_next(~pos | gap[:, None], F)
    To = t[f - 1]
    b = np.where(f + 1 < F, nbrk[np.minimum(f + 1, F - 1), c], F).astype(np.int64)
    inb = b < F
    bc = np.minimum(b, F - 1)
    btype = np.where(~inb, QF.END_B, np.where(gap[bc], QF.GAP_B, np.where(nanS[bc, c], QF.UNK_B, QF.ZERO_B)))
    zero = btype == QF.ZERO_B
    tb = np.where(inb, t[bc], QF.BIG)
    zbw = zero & (tb <= we)
    T_last = np.where(zero, np.minimum(tb, we), we)
    cK = c * QF.KEY
    k_we = int(np.searchsorted(t, we, 'left'))
    kend = np.minimum(np.where(zero, b, b - 1), k_we)
    J0 = np.searchsorted(R['rbk'], cK + off(To), 'right').astype(np.int64)
    J1 = np.maximum(np.searchsorted(R['rbk'], cK + off(T_last + QF.MAX_LAG_MS), 'right').astype(np.int64), J0)
    V0 = np.searchsorted(R['ruk'], cK + off(To), 'right').astype(np.int64)
    V1 = np.maximum(np.searchsorted(R['ruk'], cK + (QF.KEY - 1), 'right').astype(np.int64), V0)
    A1 = QF._new_life(R, nE, f, c, To, kend, b, btype, zero, zbw, tb, t, we, J0, J1, V0, V1, acode, fl['addr'], '')
    lead1 = A1['lead']
    hold1 = np.where(lead1 >= 0, np.asarray(fl['addr'], dtype=object)[np.maximum(lead1, 0)] if n else '', '').astype(object)
    isf1 = A1['st'] == 'front_identified'
    need(np.array_equal(A1['st'].astype(str), E1['status'].to_numpy(dtype=object).astype(str))
         and np.array_equal(hold1.astype(str), E1['holder'].astype(str).to_numpy(dtype=object).astype(str))
         and np.array_equal(np.where(isf1, A1['timing'], '').astype(str), E1['front_timing'].astype(str).to_numpy(dtype=object).astype(str)),
         f'market {day} {mid}: the replicated v1 front logic differs from queue_front.market_core (refused)')
    need(bool(((t[f] - To) <= QF.GAP_MS).all()), 'internal: an E1 creation interval longer than the gap limit')
    rb = np.asarray(R['rbnd'], bool)
    need(bool((R['rhi'][rb] - R['rlo'][rb] >= QF.MAX_LAG_MS).all()), 'internal: a bounded record window shorter than 2 s')
    To2 = To if mut == 'v1_straddle' else t[f]         # v2 straddle rule; the planted error keeps the v1 lower-endpoint rule
    A2 = QF._new_life(R, nE, f, c, To2, kend, b, btype, zero, zbw, tb, t, we, J0, J1, V0, V1, acode, fl['addr'], '')
    return dict(nE=nE, st_v1=A1['st'], timing_v1=np.where(isf1, A1['timing'], '').astype(object), st_v2=A2['st'], lead_v2=A2['lead'],
                timing_v2=np.where(A2['st'] == 'front_identified', A2['timing'], '').astype(object),
                n_v1_v2_changed=int((A1['st'].astype(str) != A2['st'].astype(str)).sum()))


def order_windows(O, hashes):
    """The lifecycle posting window of each hash: [min over S_FIFO / S_FIFO_LO / S_NOPRIO post_lo (-1 when any is unbounded), S_NOPRIO post_hi]
    (chain_post_fill's union; equal to post_regrade_orders for top-3 hashes) -> {hash: (lo, hi, signer)}."""
    hs = sorted({str(h) for h in hashes if str(h)})
    if not hs:
        return {}
    need(O is not None, 'no lifecycle orders table (refused)')
    for c_ in ('scenario', 'mk_hash', 'mk_signer', 'post_lo_ms', 'post_hi_ms'):
        need(c_ in O.columns, f'lifecycle orders: column {c_} missing (refused)')
    sel = np.isin(O['mk_hash'].to_numpy(dtype=object), np.array(hs, dtype=object))
    X = O[sel]
    out = {}
    if not len(X):
        return out
    df = pd.DataFrame(dict(h=X['mk_hash'].to_numpy(dtype=object), sc=X['scenario'].to_numpy(dtype=object), plo=X['post_lo_ms'].to_numpy(np.int64),
                           phi=X['post_hi_ms'].to_numpy(np.int64), sig=[str(s).lower() for s in X['mk_signer'].to_numpy(dtype=object)]))
    df = df[df['sc'].isin(SCEN)]
    nop = df[df['sc'] == 'S_NOPRIO']
    need(nop['h'].is_unique, 'lifecycle orders: a hash twice among the S_NOPRIO rows (refused)')
    plo = df.groupby('h')['plo'].min()
    nsc = df.groupby('h')['sc'].nunique()
    for h, ph, sg in zip(nop['h'].tolist(), nop['phi'].tolist(), nop['sig'].tolist()):
        p = int(plo[h])
        out[h] = (-1 if p < 0 else p, int(ph), sg, int(nsc[h]))
    return out


def identity(day, mid, M, fl, LV, CR, FB, RGm, Q2m, cnt):
    """Contract section 3 per candidate row of LV (creations touching the population): candidate first-fill address (source: queue_front_v2
    when given, else the v2-straddle fallback), its filled hash's lifecycle posting window, containment, grade, identity, inventory."""
    mut = mutant()
    n = len(LV)
    o = {k: np.full(n, '', dtype=object) for k in ('qf_status_v1', 'qf_timing_v1', 'qf_status_v2', 'qf_timing_v2', 'qf2_status', 'cand_source',
                                                   'cand_addr', 'cand_group', 'cand_hash', 'cand_sid', 'grade', 'identity', 'identity_reason',
                                                   'inv_relation')}
    for k in ('cand_mk_idx', 'cand_hi_ms', 'win_lo_ms', 'win_hi_ms'):
        o[k] = np.full(n, -1, np.int64)
    for k in ('win_unbounded', 'win_contains', 'inv_timing_sensitive', 'inv_ledger_conditional'):
        o[k] = np.zeros(n, bool)
    for k in ('inv_q', 'inv_pos_min', 'inv_pos_max'):
        o[k] = np.full(n, NAN)
    if n == 0:
        return o
    ei = np.flatnonzero(CR['e1'])
    o['qf_status_v1'][ei] = FB['st_v1']
    o['qf_timing_v1'][ei] = FB['timing_v1']
    o['qf_status_v2'][ei] = FB['st_v2']
    o['qf_timing_v2'][ei] = FB['timing_v2']
    addr = np.asarray(fl['addr'], dtype=object)
    cand = np.zeros(n, bool)
    fb_lead = np.full(n, -1, np.int64)
    fb_lead[ei] = np.where(FB['st_v2'] == 'front_identified', FB['lead_v2'], -1)
    fb_ok = fb_lead >= 0
    fb_addr = np.where(fb_ok, addr[np.maximum(fb_lead, 0)] if len(addr) else '', '').astype(object)
    if Q2m is None:
        cand = fb_ok.copy()
        o['cand_source'][cand] = 'v1_functions_v2_straddle'
        o['cand_addr'][cand] = fb_addr[cand]
        o['cand_hash'][cand] = np.asarray(fl['mk_hash'], dtype=object)[fb_lead[cand]]
        o['cand_sid'][cand] = np.asarray(fl['sid'], dtype=object)[fb_lead[cand]]
        o['cand_mk_idx'][cand] = np.asarray(fl['mk_idx'], np.int64)[fb_lead[cand]]
        o['cand_hi_ms'][cand] = np.asarray(fl['hi'], np.int64)[fb_lead[cand]]
    else:
        side = np.where(CR['side'] == 0, 'bid', 'ask')
        fr = LV['frame'].to_numpy(np.int64)
        qk = {(str(r['side']), int(r['tick']), int(r['frame'])): r for r in Q2m.to_dict('records')}
        n_unmatched = n_time = 0
        for i in ei.tolist():
            r = qk.get((side[i], int(CR['tick'][i]), int(fr[i])))
            if r is None:
                n_unmatched += 1
                continue
            if int(r['t_open_ms']) != int(CR['a'][i]) or int(r['t_create_ms']) != int(CR['b'][i]):
                n_time += 1
                continue
            o['qf2_status'][i] = str(r['status'])
            h = str(r['holder']).lower()
            if str(r['status']) in QF2_CAND and h:
                cand[i] = True
                o['cand_source'][i] = 'queue_front_v2'
                o['cand_addr'][i] = h
                o['cand_hash'][i] = str(r['front_hash'])
        need(n_time == 0, f'market {day} {mid}: queue_front_v2 levels with another creation interval for the same cell frame ({n_time}; refused)')
        cnt['qf2_e1_unmatched'] += n_unmatched
        cnt['qf2_vs_fallback_both_same_address'] += int((cand & fb_ok & (o['cand_addr'] == fb_addr)).sum())
        cnt['qf2_vs_fallback_both_other_address'] += int((cand & fb_ok & (o['cand_addr'] != fb_addr)).sum())
        cnt['qf2_only'] += int((cand & ~fb_ok).sum())
        cnt['fallback_only'] += int((~cand & fb_ok).sum())
    grp = np.full(n, '', dtype=object)
    for k_, a_ in enumerate(TOP3_L):
        grp[cand & (o['cand_addr'] == a_)] = GROUPS[k_]
    grp[cand & (grp == '')] = 'other'
    o['cand_group'] = grp
    W = order_windows(M.orders, o['cand_hash'][cand].tolist())
    touch = CR['status'] != 'outside_population'
    a, b = CR['a'], CR['b']
    for i in range(n):
        if not touch[i]:
            o['identity_reason'][i] = 'outside_population'
            continue
        if not CR['e1'][i]:
            o['identity'][i], o['identity_reason'][i] = 'owner_unknown', f'not_fresh_e1:{CR["status"][i]}'
            continue
        if not cand[i]:
            st_ = o['qf2_status'][i] if Q2m is not None else o['qf_status_v2'][i]
            o['identity'][i], o['identity_reason'][i] = 'owner_unknown', f'no_front_candidate:{st_ or "unmatched"}'
            continue
        h = o['cand_hash'][i]
        if not o['cand_addr'][i]:
            o['identity'][i], o['identity_reason'][i] = 'owner_unknown', 'candidate_without_address'
            continue
        w = W.get(h)
        if w is None:
            o['identity'][i], o['identity_reason'][i] = 'owner_unknown', 'no_order_window'
            continue
        lo_, hi_, sg, nsc = w
        need(sg == o['cand_addr'][i], f'market {day} {mid}: order {h} signer {sg} differs from its candidate fill address (refused)')
        o['win_lo_ms'][i], o['win_hi_ms'][i], o['win_unbounded'][i] = lo_, hi_, lo_ < 0
        contains = (lo_ < 0 or lo_ <= int(a[i]) + 1) and int(b[i]) <= hi_ and (lo_ < 0 or hi_ >= lo_)
        o['win_contains'][i] = bool(contains)
        rg = (RGm or {}).get(h)
        if rg is not None:
            need(rg['lo'] == lo_ and rg['hi'] == hi_ and rg['addr'] == o['cand_addr'][i] and rg['side'] == ('bid' if CR['side'][i] == 0 else 'ask')
                 and rg['tick'] == int(CR['tick'][i]),
                 f'market {day} {mid}: post_regrade_orders row of {h} differs from the lifecycle window / candidate cell (refused: mismatched artifacts)')
            grade = rg['grade']
        else:
            grade = NO_GRADE
            cnt['candidate_top3_without_regrade_row'] += int(grp[i] in GROUPS[:3])
        o['grade'][i] = grade
        if not (contains or mut == 'link_any_window'):     # planted error: a link whatever the window
            o['identity'][i], o['identity_reason'][i] = 'owner_unknown', 'window_not_containing'
            continue
        o['identity'][i] = 'candidate_constrained' if grade == G2 else 'candidate_scenario'
        o['identity_reason'][i] = 'ok'
    con = np.flatnonzero(o['identity'] == 'candidate_constrained')
    for a_ in sorted(set(o['cand_addr'][con].tolist())):
        ix = con[o['cand_addr'][con] == a_]
        need(a_ in M.addrs, f'market {day} {mid}: a constrained candidate {a_} outside the loaded addresses (refused)')
        tq = (a[ix] + 1).astype(np.int64)
        p = CX.position(M, a_, tq, clock='hi')
        pe = CX.pending(M, a_, tq)
        q = np.asarray(p['pos'], float)
        o['inv_q'][ix] = q
        o['inv_pos_min'][ix] = np.asarray(pe['pos_min'], float)
        o['inv_pos_max'][ix] = np.asarray(pe['pos_max'], float)
        rel = relation(CR['side'][ix], q)
        o['inv_relation'][ix] = rel
        o['inv_timing_sensitive'][ix] = (relation(CR['side'][ix], pe['pos_min']) != rel) | (relation(CR['side'][ix], pe['pos_max']) != rel)
        o['inv_ledger_conditional'][ix] = np.asarray(p['status'], dtype=object) == 'excluded_leg_possible'
    # the candidate's lead leg must be one of its fills_valued legs (address agreement with the econ table) for top-3 fallback candidates
    if Q2m is None:
        for i in np.flatnonzero(cand & np.isin(grp, GROUPS[:3])).tolist():
            a_ = o['cand_addr'][i]
            G = pd.concat([x for x in (M.legs.get(a_), M.excluded.get(a_)) if x is not None and len(x)], ignore_index=True) \
                if any(x is not None and len(x) for x in (M.legs.get(a_), M.excluded.get(a_))) else None
            ok = G is not None and bool(((G['sid'].astype(str).to_numpy(dtype=object) == o['cand_sid'][i])
                                         & (G['mk_idx'].to_numpy(np.int64) == o['cand_mk_idx'][i])).any())
            cnt['candidate_lead_not_in_fills_valued'] += int(not ok)
    return o


# ------------------------------------------------------------------ one market ------------------------------------------------------------------
def offsets(day, mid, K):
    """The declared shift offsets of the market's two sides: Generator(PCG64(SeedSequence([20261123, day, mid, side]))).integers(0, K)."""
    return np.vstack([np.random.Generator(np.random.PCG64(np.random.SeedSequence([SEED, int(day), int(mid), s]))).integers(0, K, size=N_SHIFT)
                      for s in (0, 1)]).astype(np.int64)


def market_pass(day, mid, M, ctx, RGm=None, Q2m=None):
    """One market: queue_front's book / fills / fresh cells, the frozen state set, the creations per row, the identity layer -> a part."""
    day, mid = str(day), int(mid)
    ws, we = int(M.ws), int(M.we)
    K = (we - ws) // STEP_MS
    need(K >= 1, f'market {day} {mid}: window shorter than one second (refused)')
    cnt = Counter()
    bk = QF.market_book(M)
    fl, fcnt = QF.market_fills(M, None)
    cnt.update({f'legs:{k}': int(v) for k, v in fcnt.items()})
    core = QF.market_core(day, mid, ws, we, bk, fl)
    LV = core['levels']
    cnt.update({f'qf:{k}': int(v) for k, v in core['counts'].items()})
    FB = fallback_fronts(day, mid, ws, we, bk, fl, LV)
    cnt['v1_v2_status_changed'] += int(FB['n_v1_v2_changed'])
    FI = flow_index(fl, cnt)
    T, Fx = features(M, ctx, FI, ws, we, K)
    CR = creations(day, mid, ws, K, LV)
    rows = dict(side=np.repeat(np.array([0, 1], np.int8), K), k=np.tile(np.arange(K, dtype=np.int64), 2), t=np.tile(T, 2))
    for d, sl in (('0', slice(0, K)), ('1', slice(K, 2 * K))):
        for fld in FEAT:
            rows[f'{fld}_{d}'] = np.concatenate([np.asarray(Fx[0][fld])[sl], np.asarray(Fx[1][fld])[sl]])
    rows.update(row_outcomes(CR, K))
    rows['imp_pop'] = (rows['st_0'] == 0) & np.isin(rows['sp_c_0'], (1, 2, 3))
    ID = identity(day, mid, M, fl, LV, CR, FB, RGm, Q2m, cnt)
    n = CR['n']
    ev = dict(key=CR['key'], day=np.full(n, day, dtype=object), mid=np.full(n, mid, np.int64), side=np.where(CR['side'] == 0, 'bid', 'ask').astype(object),
              tick=CR['tick'], frame=LV['frame'].to_numpy(np.int64) if n else np.zeros(0, np.int64), fid=CR['fid'], t_open_ms=CR['a'], t_create_ms=CR['b'],
              k_lo=CR['k_lo'], k_hi=CR['k_hi'], row_k=CR['row_k'], status=CR['status'], touch_unknown=CR['tunk'],
              mechanism_v1=LV['mechanism'].to_numpy(dtype=object) if n else np.zeros(0, object),
              best_prev_tick=LV['best_prev_tick'].to_numpy(float) if n else np.zeros(0),
              size_created=LV['size_created'].to_numpy(float) if n else np.zeros(0))
    for ty in MIX_TYPES:
        ev[FLAG_OF[ty]] = CR['flags'][ty]
    ev.update(ID)
    EV = pd.DataFrame({c_: ev[c_] for c_ in EV_COLS})
    for s_ in CRE_ST:
        cnt[f'creations_{s_}'] += int((CR['status'] == s_).sum())
    return dict(day=day, mid=mid, ws=ws, we=we, K=int(K), rows=rows, events=EV, U=offsets(day, mid, K), counts=cnt)


# ------------------------------------------------------------------ the panel ------------------------------------------------------------------
def assemble(parts):
    """Market parts -> the panel P (rows side-major within each market; market-side ms = 2 mi + side) with the shift offsets."""
    need(len(parts) > 0, 'no market part (refused)')
    keys = list(parts[0]['rows'].keys())
    P = {k: np.concatenate([np.asarray(p['rows'][k]) for p in parts]) for k in keys}
    P['mi'] = np.concatenate([np.full(2 * p['K'], i, np.int64) for i, p in enumerate(parts)])
    P['ms'] = P['mi'] * 2 + P['side'].astype(np.int64)
    P['ms_K'] = np.repeat(np.array([p['K'] for p in parts], np.int64), 2)
    P['ms_base'] = np.r_[0, np.cumsum(P['ms_K'])[:-1]].astype(np.int64)
    P['U'] = np.vstack([p['U'] for p in parts]).astype(np.int64)
    P['markets'] = [dict(day=p['day'], mid=p['mid'], ws=p['ws'], we=p['we'], K=p['K']) for p in parts]
    P['days'] = sorted({p['day'] for p in parts})
    P['mday'] = np.array([P['days'].index(p['day']) for p in parts], np.int64)
    P['N'] = int(len(P['mi']))
    return P


def population_checks(P):
    """Computed predicates: every market-side holds every second ws .. ws + 1000 (K - 1) on the 1 s grid; rows = sum 2 K; K = 300 each."""
    ok_grid = True
    for i, m in enumerate(P['markets']):
        for s in (0, 1):
            b0 = int(P['ms_base'][2 * i + s])
            sl = slice(b0, b0 + m['K'])
            ok_grid &= bool(np.array_equal(P['t'][sl], m['ws'] + STEP_MS * np.arange(m['K'], dtype=np.int64))
                            and bool((P['side'][sl] == s).all()) and bool((P['mi'][sl] == i).all()))
    K_tot = int(sum(2 * m['K'] for m in P['markets']))
    return dict(markets=len(P['markets']), rows=int(P['N']), expected_rows=K_tot, rows_19x300x2=N_BOOK * K_EXPECT * 2,
                every_market_side_every_second=bool(ok_grid), all_markets_300_s=bool(all(m['K'] == K_EXPECT for m in P['markets'])),
                complete=bool(ok_grid and int(P['N']) == K_tot))


def shift_dest(P, idx):
    """The destination row of each event row idx under every circular shift of its market-side's schedule -> (N_SHIFT, n_e)."""
    idx = np.asarray(idx, np.int64)
    if len(idx) == 0:
        return np.zeros((N_SHIFT, 0), np.int64)
    ms = P['ms'][idx]
    k = P['k'][idx]
    Km, b0 = P['ms_K'][ms], P['ms_base'][ms]
    return np.ascontiguousarray((b0[:, None] + (k[:, None] + P['U'][ms]) % Km[:, None]).T)


def contrast_code(P, v, ty, cn, n_mk):
    """Exposure group (1 exposed / 0 reference / -1 out) and stratum of contrast cn for outcome ty at variant v -> (code, grp, S, per)."""
    mut = mutant()
    d = VSUF[v]
    ok = P['st_' + d] == 0
    restrict = ty == 'improve' and mut != 'keep_spread1'      # planted error keep_spread1: spread-1 rows stay in improve ratios
    if restrict:
        ok = ok & P['imp_pop']
    sp, fg, bc, dc, fw, tl = (P[f'{x}_{d}'].astype(np.int64) for x in ('sp_c', 'fg_c', 'btc_c', 'd_own_c', 'flow_c', 'tl_c'))
    if cn == 'spread':
        grp = np.select([(sp == 2) | (sp == 3), sp == 1], [1, 0], -1) if restrict else np.select([(sp >= 1) & (sp <= 3), sp == 0], [1, 0], -1)
    elif cn == 'fair_gap':
        grp = np.select([(fg == 2) | (fg == 3), fg == 0], [1, 0], -1)
    elif cn == 'btc_with':
        grp = np.select([bc == 0, bc == 2], [1, 0], -1)
    elif cn == 'btc_against':
        grp = np.select([bc == 1, bc == 2], [1, 0], -1)
    elif cn == 'depth_own':
        grp = np.select([(dc == 0) | (dc == 1), dc == 3], [1, 0], -1)
    elif cn == 'flow':
        grp = np.select([fw == 1, fw == 0], [1, 0], -1)
    elif cn == 'time_left':
        grp = np.select([tl >= 5, (tl == 2) | (tl == 3)], [1, 0], -1)
    else:
        raise ValueError(f'unknown contrast {cn!r}')
    grp = np.where(ok, grp, -1).astype(np.int64)
    ms = P['ms'].astype(np.int64)
    if cn == 'time_left':
        strat, S, per = ms, 2 * n_mk, 2
    else:
        strat, S, per = ms * 7 + tl, 14 * n_mk, 14
    code = np.where(grp >= 0, strat * 2 + grp, -1).astype(np.int64)
    need(bool((code < 2 * S).all()), f'{cn}: a stratum code outside its range (refused)')
    return code, grp, S, per, ok


def verdict(rr, c_hi, c_lo, n_ev, n_sup, n_gt1, n_lt1):
    """take_logic's rule with its mirror (computed, never printed beside a number)."""
    if n_sup < MIN_MARKETS or n_ev < MIN_EVENTS:
        return 'insufficient_support'
    up = bool(_gt1(rr)) and c_hi is not None and int(c_hi) <= C_MAX and SHARE_DEN * int(n_gt1) >= SHARE_NUM * int(n_sup)
    lo = bool(_lt1(rr)) and c_lo is not None and int(c_lo) <= C_MAX and SHARE_DEN * int(n_lt1) >= SHARE_NUM * int(n_sup)
    if up:
        return 'supported'
    if lo:
        return 'supported_lower'
    return 'not_supported'


def ratio_block(P, v, ty, cn, ev_idx, SH):
    """One (variant, outcome, contrast): pooled MH RR, its shift null, within-market MH / crude RR per market, day readings, the verdict."""
    n_mk = len(P['markets'])
    code, grp, S, per, ok = contrast_code(P, v, ty, cn, n_mk)
    pt, obs = TL.stratum_counts(code, S, 2, ev_idx)
    nul = TL.null_counts(code, S, 2, SH)
    T1, T0 = pt[:, 1], pt[:, 0]
    rr = TL.mh_rr(np.vstack([obs[:, 1][None, :], nul[:, :, 1]]), np.vstack([obs[:, 0][None, :], nul[:, :, 0]]), T1, T0)[0]
    o, nl = float(rr[0]), rr[1:]
    c_hi, c_lo = TL.tail_counts(o, nl)
    inf_s = (T1 > 0) & (T0 > 0)
    n_ev = int(round(float((obs[:, 1] + obs[:, 0])[inf_s].sum())))
    Ts = np.where(inf_s, T1 + T0, 1.)
    num_s, den_s = np.where(inf_s, obs[:, 1] * T0 / Ts, 0.), np.where(inf_s, obs[:, 0] * T1 / Ts, 0.)
    mof = np.arange(S) // per
    num_m, den_m = np.bincount(mof, weights=num_s, minlength=n_mk), np.bincount(mof, weights=den_s, minlength=n_mk)
    a_m, b_m = np.bincount(mof, weights=obs[:, 1], minlength=n_mk), np.bincount(mof, weights=obs[:, 0], minlength=n_mk)
    T1_m, T0_m = np.bincount(mof, weights=T1, minlength=n_mk), np.bincount(mof, weights=T0, minlength=n_mk)
    with np.errstate(divide='ignore', invalid='ignore'):
        rr_m = np.where(den_m > 0, num_m / np.where(den_m > 0, den_m, 1.), np.where(num_m > 0, INF, NAN))
        cr_m = np.where((b_m > 0) & (T1_m > 0) & (T0_m > 0), (a_m / np.where(T1_m > 0, T1_m, 1.)) / (b_m / np.where(T0_m > 0, T0_m, 1.)),
                        np.where((a_m > 0) & (T1_m > 0), INF, NAN))
    sup = (a_m + b_m >= 1) & (T1_m >= MIN_MKT_S) & (T0_m >= MIN_MKT_S)
    gt1, lt1, cgt1 = sup & _gt1(rr_m), sup & _lt1(rr_m), sup & _gt1(cr_m)
    n_sup, n_gt1, n_lt1 = int(sup.sum()), int(gt1.sum()), int(lt1.sum())
    a_t, b_t, T1t, T0t = float(obs[:, 1].sum()), float(obs[:, 0].sum()), float(T1.sum()), float(T0.sum())
    with np.errstate(divide='ignore', invalid='ignore'):
        crude = (a_t / T1t) / (b_t / T0t) if (T1t > 0 and T0t > 0 and b_t > 0) else (INF if (a_t > 0 and T1t > 0) else NAN)
    fin = nl[np.isfinite(nl)]
    qn = np.quantile(fin, [.05, .5, .95]) if len(fin) else [NAN] * 3
    vr = verdict(o, c_hi, c_lo, n_ev, n_sup, n_gt1, n_lt1)
    sig_hi = c_hi is not None and int(c_hi) <= C_MAX
    sig_lo = c_lo is not None and int(c_lo) <= C_MAX
    imp = ty == 'improve' and mutant() != 'keep_spread1'
    exl, rfl = (IMPROVE_SPREAD if (imp and cn == 'spread') else next((e, r) for c_, e, r in CONTRASTS if c_ == cn))
    row = dict(variant=v, type=ty, contrast=cn, exposed=exl, reference=rfl,
               strata='market x side' if cn == 'time_left' else 'market x side x time-left bin',
               population=('rows with book ok at the variant instant and anchor spread >= 2 (improve availability)' if imp
                           else 'rows with book ok at the variant instant'),
               RR_MH=o, RR_crude=float(crude), a_exposed=a_t, T_exposed=T1t, b_reference=b_t, T_reference=T0t, n_strata=int(S),
               n_strata_informative=int(inf_s.sum()), n_events_informative=n_ev,
               rows_status_excluded=int((P['st_' + VSUF[v]] != 0).sum()), rows_availability_excluded=int(((P['st_' + VSUF[v]] == 0) & ~ok).sum()),
               rows_other_bins=int((ok & (grp < 0)).sum()), null_n=int(len(nl)), null_n_nan=int(np.isnan(nl).sum()), null_n_inf=int(np.isinf(nl).sum()),
               null_mean=float(fin.mean()) if len(fin) else NAN, null_p05=float(qn[0]), null_p50=float(qn[1]), null_p95=float(qn[2]),
               c_hi=-1 if c_hi is None else int(c_hi), c_lo=-1 if c_lo is None else int(c_lo), p_hi=NAN if c_hi is None else c_hi / (N_SHIFT + 1),
               p_lo=NAN if c_lo is None else c_lo / (N_SHIFT + 1), n_markets_supported=n_sup, n_markets_rr_gt1=n_gt1, n_markets_rr_lt1=n_lt1,
               share_rr_gt1=(n_gt1 / n_sup) if n_sup else NAN, share_rr_lt1=(n_lt1 / n_sup) if n_sup else NAN, n_markets_crude_gt1=int(cgt1.sum()),
               inconsistent_hi=bool(sig_hi and _gt1(o) and not (SHARE_DEN * n_gt1 >= SHARE_NUM * n_sup)),
               inconsistent_lo=bool(sig_lo and _lt1(o) and not (SHARE_DEN * n_lt1 >= SHARE_NUM * n_sup)), verdict_variant=vr, verdict='')
    cons = []
    for mi, mm in enumerate(P['markets']):
        cons.append(dict(row='market', variant=v, type=ty, contrast=cn, day=mm['day'], mid=mm['mid'], a=float(a_m[mi]), T1=float(T1_m[mi]),
                         b=float(b_m[mi]), T0=float(T0_m[mi]), rr_mh=float(rr_m[mi]), rr_crude=float(cr_m[mi]), supported=bool(sup[mi]),
                         rr_gt1=bool(gt1[mi]), rr_lt1=bool(lt1[mi]), crude_gt1=bool(cgt1[mi])))
    for di, dd in enumerate(P['days']):
        md = P['mday'] == di
        sm = md[mof]
        nm_, dn_ = float(num_s[sm].sum()), float(den_s[sm].sum())
        rr_d = nm_ / dn_ if dn_ > 0 else (INF if nm_ > 0 else NAN)
        ns, ng, nl_ = int(sup[md].sum()), int(gt1[md].sum()), int(lt1[md].sum())
        cons.append(dict(row='day', variant=v, type=ty, contrast=cn, day=dd, mid=-1, a=float(a_m[md].sum()), T1=float(T1_m[md].sum()),
                         b=float(b_m[md].sum()), T0=float(T0_m[md].sum()), rr_mh=rr_d, n_supported=ns, n_rr_gt1=ng, n_rr_lt1=nl_,
                         share_gt1=(ng / ns) if ns else NAN))
    cons.append(dict(row='day', variant=v, type=ty, contrast=cn, day='all', mid=-1, a=a_t, T1=T1t, b=b_t, T0=T0t, rr_mh=o, n_supported=n_sup,
                     n_rr_gt1=n_gt1, n_rr_lt1=n_lt1, share_gt1=(n_gt1 / n_sup) if n_sup else NAN))
    return row, cons


def ratio_tables(P):
    """Every variant x outcome x contrast -> (rows keyed (variant, type, contrast), consistency rows, final verdicts)."""
    ev_idx = {ty: np.flatnonzero(P['E_' + ty] > 0) for ty in TYPES}
    RK, cons = {}, []
    for ty in TYPES:
        SH = shift_dest(P, ev_idx[ty])
        for v in VARIANTS:
            for cn in CN_NAMES:
                r, c_ = ratio_block(P, v, ty, cn, ev_idx[ty], SH)
                RK[(v, ty, cn)] = r
                cons.extend(c_)
        del SH
    FIN = {}
    for ty in TYPES:
        for cn in CN_NAMES:
            v0, v1 = RK[('D0', ty, cn)]['verdict_variant'], RK[('D1000', ty, cn)]['verdict_variant']
            fv = v0 if v0 == v1 else 'timing_sensitive'
            FIN[(ty, cn)] = fv
            RK[('D0', ty, cn)]['verdict'] = RK[('D1000', ty, cn)]['verdict'] = fv
    return RK, cons, FIN


# ------------------------------------------------------------------ the identity tables (contract section 3) ------------------------------------------------------------------
def identity_tables(EVT, days):
    """identity.csv: owner status x candidate group, owner_unknown reasons, grade counts, the type mix of candidate-linked top-3 creations vs
    all creations (counts per grade), and the candidate_constrained inventory relations; per population (event / straddle) and scope."""
    rows = []
    T3 = GROUPS[:3]
    for sc, d in [('pooled', 'all')] + [('day', x) for x in days]:
        X = EVT if sc == 'pooled' else EVT[EVT['day'].astype(str) == d]
        for pop in ('event', 'straddle'):
            Y = X[X['status'] == pop]
            n_all = int(len(Y))
            base = dict(scope=sc, day=d, population=pop)
            for idn in IDENT:
                Z = Y[Y['identity'] == idn]
                rows.append(dict(base, table='owner', group='all', grade='', category=idn, n=int(len(Z)), denom=n_all, share=len(Z) / n_all if n_all else NAN))
                if idn != 'owner_unknown':
                    for gname in GROUPS + ('top3',):
                        Zg = Z[Z['cand_group'].isin(T3 if gname == 'top3' else (gname,))]
                        rows.append(dict(base, table='owner', group=gname, grade='', category=idn, n=int(len(Zg)), denom=n_all,
                                         share=len(Zg) / n_all if n_all else NAN))
            U = Y[Y['identity'] == 'owner_unknown']
            for rs, k in sorted(Counter(U['identity_reason'].astype(str).tolist()).items()):
                rows.append(dict(base, table='owner_unknown_reason', group='all', grade='', category=rs, n=int(k), denom=int(len(U)),
                                 share=k / len(U) if len(U) else NAN))
            Cd = Y[Y['identity'].isin(IDENT[:2])]
            for gname in GROUPS + ('top3',):
                Zg = Cd[Cd['cand_group'].isin(T3 if gname == 'top3' else (gname,))]
                for gr in GRADES + (NO_GRADE,):
                    k = int((Zg['grade'] == gr).sum())
                    rows.append(dict(base, table='grade', group=gname, grade=gr, category='candidate_linked', n=k, denom=int(len(Zg)),
                                     share=k / len(Zg) if len(Zg) else NAN))
            for ty in MIX_TYPES:
                k = int(Y[FLAG_OF[ty]].astype(bool).sum())
                rows.append(dict(base, table='type_mix', group='all_creations', grade='any', category=ty, n=k, denom=n_all, share=k / n_all if n_all else NAN))
            for gname in T3 + ('top3',):
                for gr in IDENT[:2] + ('any_candidate',):
                    Zg = Cd[Cd['cand_group'].isin(T3 if gname == 'top3' else (gname,))]
                    if gr != 'any_candidate':
                        Zg = Zg[Zg['identity'] == gr]
                    nz = int(len(Zg))
                    for ty in MIX_TYPES:
                        k = int(Zg[FLAG_OF[ty]].astype(bool).sum())
                        rows.append(dict(base, table='type_mix', group=gname, grade=gr, category=ty, n=k, denom=nz, share=k / nz if nz else NAN))
            Cc = Y[Y['identity'] == 'candidate_constrained']
            for gname in T3 + ('top3',):
                Zg = Cc[Cc['cand_group'].isin(T3 if gname == 'top3' else (gname,))]
                nz = int(len(Zg))
                for rl in RELATIONS:
                    k = int((Zg['inv_relation'] == rl).sum())
                    rows.append(dict(base, table='inventory', group=gname, grade='candidate_constrained', category=rl, n=k, denom=nz,
                                     share=k / nz if nz else NAN))
                for fl_ in ('inv_timing_sensitive', 'inv_ledger_conditional'):
                    k = int(Zg[fl_].astype(bool).sum())
                    rows.append(dict(base, table='inventory', group=gname, grade='candidate_constrained', category=fl_, n=k, denom=nz,
                                     share=k / nz if nz else NAN))
                for ty in MIX_TYPES:
                    Zt = Zg[Zg[FLAG_OF[ty]].astype(bool)]
                    for rl in RELATIONS:
                        k = int((Zt['inv_relation'] == rl).sum())
                        rows.append(dict(base, table='inventory_by_type', group=gname, grade='candidate_constrained', category=f'{ty}:{rl}', n=k,
                                         denom=int(len(Zt)), share=k / len(Zt) if len(Zt) else NAN))
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ tables, claims ------------------------------------------------------------------
def rows_frame(P):
    """rows.parquet: one row per market x side x second (labels for the bins and statuses)."""
    mk = P['markets']
    mi = P['mi']
    df = dict(day=np.array([mk[i]['day'] for i in mi], dtype=object), mid=np.array([mk[i]['mid'] for i in mi], np.int64),
              side=np.where(P['side'] == 0, 'bid', 'ask').astype(object), k=P['k'], t_ms=P['t'])
    labs = dict(st=ROW_ST, sp_c=SPREAD_LAB, d_own_c=DEPTH_LAB, d_opp_c=DEPTH_LAB, d2_c=DEPTH_LAB, btc_c=BTC_LAB, fg_c=FG_LAB, flow_c=FLOW_LAB, tl_c=TL_LAB)
    names = dict(st='status', age='book_age_ms', spread='spread_ticks', sp_c='spread_bin', d_own='depth_touch_own', d_own_c='depth_touch_own_bin',
                 d_opp='depth_touch_opp', d_opp_c='depth_touch_opp_bin', d2='depth_2_own', d2_c='depth_2_own_bin', r5='btc_r_5s_bp', btc_c='btc_5s_aligned',
                 pf='p_fair', fg='fair_gap_c', fg_c='fair_gap_bin', flow='flow_5s_sh', flow_c='flow_5s_bin', tl='time_left_s', tl_c='time_left_bin')
    for v in VARIANTS:
        d = VSUF[v]
        for fld in FEAT:
            x = P[f'{fld}_{d}']
            df[f'{names[fld]}_{v}'] = np.array(labs[fld], dtype=object)[x.astype(np.int64)] if fld in labs else x
    df['improve_population'] = P['imp_pop'].astype(bool)
    for ty in TYPES:
        df[f'E_{ty}'] = P['E_' + ty].astype(np.int8)
        df[f'N_{ty}'] = P['N_' + ty].astype(np.int32)
        df[f'keys_{ty}'] = P['keys_' + ty]
    for stn in ('straddle', 'gap', 'unknown_cell', 'type_unknown'):
        df[f'n_{stn}'] = P['n_' + stn].astype(np.int32)
    return pd.DataFrame(df)


def reconcile(P, EVT, RK, IDT):
    """Computed predicates: every aggregate against the saved keyed rows."""
    ev = EVT[EVT['status'] == 'event']
    out = dict(events_equal_row_counts=int(len(ev)) == int(P['N_any'].sum()),
               keys_equal=sorted(';'.join(x for x in P['keys_any'] if x).split(';')) == sorted(ev['key'].tolist()) if len(ev) else int(P['N_any'].sum()) == 0)
    for ty in MIX_TYPES:
        out[f'{ty}_events_equal_row_counts'] = int(ev[FLAG_OF[ty]].astype(bool).sum()) == int(P['N_' + ty].sum())
    out['indicator_le_count'] = all(bool((P['E_' + ty] <= np.minimum(P['N_' + ty], 1)).all() and (P['E_' + ty] == (P['N_' + ty] > 0)).all()) for ty in TYPES)
    ok_t = True
    for (v, ty, cn), r in RK.items():
        code = contrast_code(P, v, ty, cn, len(P['markets']))[0]
        ex_, rf_ = (code >= 0) & (code % 2 == 1), (code >= 0) & (code % 2 == 0)
        ev_i = P['E_' + ty] > 0
        ok_t &= (int(r['T_exposed']) == int(ex_.sum()) and int(r['T_reference']) == int(rf_.sum())
                 and int(r['a_exposed']) == int((ex_ & ev_i).sum()) and int(r['b_reference']) == int((rf_ & ev_i).sum()))
    out['ratio_counts_equal_rows'] = bool(ok_t)
    pe = IDT[(IDT['table'] == 'owner') & (IDT['scope'] == 'pooled') & (IDT['population'] == 'event') & (IDT['group'] == 'all')]
    out['identity_counts_equal_events'] = ({r['category']: int(r['n']) for r in pe.to_dict('records')}
                                           == {idn: int((ev['identity'] == idn).sum()) for idn in IDENT})
    out['identity_complete'] = bool(ev['identity'].isin(IDENT).all()) if len(ev) else True
    out['all'] = all(bool(v) for v in out.values())
    return out


def analyse(kind, days, parts, cnt, dependency=None):
    """Parts -> (frames, tables, claims, digest)."""
    P = assemble(parts)
    popc = population_checks(P)
    need(popc['complete'], f'population check failed (refused): {popc}')
    RK, cons, FIN = ratio_tables(P)
    EVT = pd.concat([p['events'] for p in parts], ignore_index=True)
    IDT = identity_tables(EVT, P['days'])
    rec = reconcile(P, EVT, RK, IDT)
    need(rec['all'], f'reconciliation failed (refused): {rec}')
    claims = make_claims(kind, days, P, EVT, RK, FIN, IDT, popc, rec, cnt, dependency)
    frames = {'rows.parquet': rows_frame(P)}
    tables = {'events.csv': EVT, 'ratios.csv': pd.DataFrame([RK[(v, ty, cn)] for ty in TYPES for cn in CN_NAMES for v in VARIANTS]),
              'consistency.csv': pd.DataFrame(cons), 'identity.csv': IDT}
    return frames, tables, claims, digest(claims)


def make_claims(kind, days, P, EVT, RK, FIN, IDT, popc, rec, cnt, dependency):
    def st_counts(v):
        x = P['st_' + VSUF[v]].astype(np.int64)
        return {ROW_ST[i]: int((x == i).sum()) for i in range(len(ROW_ST))}
    cre = {s_: int((EVT['status'] == s_).sum()) for s_ in CRE_ST}
    ev = EVT[EVT['status'] == 'event']
    by_type = {ty: int(ev[FLAG_OF[ty]].astype(bool).sum()) for ty in MIX_TYPES}
    rows_with = {ty: int(P['E_' + ty].sum()) for ty in TYPES}
    ratios = []
    for ty in TYPES:
        for cn in CN_NAMES:
            ent = dict(type=ty, contrast=cn, verdict=FIN[(ty, cn)])
            for v in VARIANTS:
                r = RK[(v, ty, cn)]
                ent[v] = dict(RR_MH=r['RR_MH'], RR_crude=r['RR_crude'], p_hi=r['p_hi'], p_lo=r['p_lo'], n_events_informative=r['n_events_informative'],
                              n_markets_supported=r['n_markets_supported'], n_markets_rr_gt1=r['n_markets_rr_gt1'], n_markets_rr_lt1=r['n_markets_rr_lt1'],
                              verdict=r['verdict_variant'], exposed=r['exposed'], reference=r['reference'],
                              predicates=dict(rr_gt_1=bool(_gt1(r['RR_MH'])), rr_lt_1=bool(_lt1(r['RR_MH'])),
                                              p_hi_le_0_05=bool(0 <= r['c_hi'] <= C_MAX), p_lo_le_0_05=bool(0 <= r['c_lo'] <= C_MAX),
                                              supported_markets_ge_5=bool(r['n_markets_supported'] >= MIN_MARKETS),
                                              events_ge_30=bool(r['n_events_informative'] >= MIN_EVENTS),
                                              share_gt1_ge_2_3=bool(r['n_markets_supported'] and SHARE_DEN * r['n_markets_rr_gt1'] >= SHARE_NUM * r['n_markets_supported']),
                                              share_lt1_ge_2_3=bool(r['n_markets_supported'] and SHARE_DEN * r['n_markets_rr_lt1'] >= SHARE_NUM * r['n_markets_supported']),
                                              inconsistent_hi=r['inconsistent_hi'], inconsistent_lo=r['inconsistent_lo']))
            ent['flips_at_D1000'] = ent['D0']['verdict'] != ent['D1000']['verdict']
            ratios.append(ent)
    lists = {vd: [f'{e["type"]}:{e["contrast"]}' for e in ratios if e['verdict'] == vd] for vd in VERDICTS}
    pe = IDT[(IDT['table'] == 'owner') & (IDT['scope'] == 'pooled') & (IDT['population'] == 'event')]
    owner = {f"{r['group']}:{r['category']}": int(r['n']) for r in pe.to_dict('records')}
    mix = IDT[(IDT['table'] == 'type_mix') & (IDT['scope'] == 'pooled') & (IDT['population'] == 'event')]
    mixd = {f"{r['group']}:{r['grade']}:{r['category']}": dict(n=int(r['n']), denom=int(r['denom']), share=r['share']) for r in mix.to_dict('records')}
    gr = IDT[(IDT['table'] == 'grade') & (IDT['scope'] == 'pooled') & (IDT['population'] == 'event') & (IDT['group'] == 'top3')]
    inv = IDT[(IDT['table'] == 'inventory') & (IDT['scope'] == 'pooled') & (IDT['population'] == 'event')]
    return dict(version=VERSION, kind=kind, contract=dict(file=_rel(CONTRACT_FILE), commit=CONTRACT_COMMIT, sections='1-3'),
                review=dict(file=_rel(REVIEW_FILE), commit=REVIEW_COMMIT), days=list(days), markets=[f"{m['day']}_{m['mid']}" for m in P['markets']],
                population=dict(popc, rows_by_status_D0=st_counts('D0'), rows_by_status_D1000=st_counts('D1000'),
                                improve_population_rows=int(P['imp_pop'].sum()), creations_by_status=cre, events_by_type=by_type,
                                rows_with_event=rows_with, type_unknown_events=int(ev['touch_unknown'].astype(bool).sum()),
                                rows_touched_by_straddle=int((P['n_straddle'] > 0).sum()), rows_touched_by_gap=int((P['n_gap'] > 0).sum()),
                                rows_touched_by_unknown_cell=int((P['n_unknown_cell'] > 0).sum())),
                ratios=ratios, verdict_lists=dict(supported_associations=lists['supported'], supported_lower_contradicting_the_exposure=lists['supported_lower'],
                                                  not_supported=lists['not_supported'], insufficient_support=lists['insufficient_support'],
                                                  timing_sensitive=lists['timing_sensitive']),
                identity=dict(owner_events=owner, unidentified_links=owner.get('all:owner_unknown', 0), type_mix=mixd,
                              grades_top3={r['grade']: int(r['n']) for r in gr.to_dict('records')},
                              inventory_constrained={f"{r['group']}:{r['category']}": int(r['n']) for r in inv.to_dict('records')},
                              reading='shares and counts only; a candidate link with its grade, never the identified creator; no verdict on maker intent',
                              dependency=dependency),
                reconciliation=rec,
                null=dict(design="500 circular shifts of each market-side's creation-event schedule (all types together); states fixed", n_shifts=N_SHIFT,
                          seed=SEED, offsets='Generator(PCG64(SeedSequence([20261123, day, mid, side]))).integers(0, K) per market-side',
                          p='(1 + #{null >= / <= obs}) / 501; ties within 1e-12 relative; NaN replicates on both sides'),
                rules=dict(verdict='take_logic: insufficient_support (< 5 supported markets or < 30 informative event-seconds) | supported (RR > 1, '
                                   'p_hi <= 0.05, >= 2/3 of supported markets RR > 1) | supported_lower (mirror: RR < 1, p_lo, RR < 1) | not_supported '
                                   "(take_logic's inconsistent folded in, predicate saved); timing_sensitive when the D1000 verdict differs",
                           market_rr='within-market Mantel-Haenszel over the market\'s strata (crude beside)', tie=TIE, min_events=MIN_EVENTS,
                           min_markets=MIN_MARKETS, min_market_seconds=MIN_MKT_S, p_max=P_MAX, c_max=C_MAX, share=f'{SHARE_NUM}/{SHARE_DEN}',
                           improve='rows with anchor (D0) spread >= 2 only; spread contrast 3+ vs 2', strata='market x side x time-left bin; '
                           'time-left contrast: market x side', d1000='every state read at t - 1000 ms (outcome window unchanged)'),
                multiplicity=dict(contrasts=len(CN_NAMES), outcome_types=len(TYPES), tests=len(CN_NAMES) * len(TYPES), all_listed=True, selected=None),
                wording=dict(opportunity='an observed market opportunity, not an identified maker decision time',
                             occupied='adding size at an occupied price is not a fresh creation; its absence is never "no posting"',
                             p='a circular-shift tail fraction (timing alignment under fixed state paths), not independent-day confidence',
                             days='four consumed days: point estimates, no intervals (G = 4 < 5)'),
                counts={k: int(v) for k, v in sorted(cnt.items()) if not k.startswith('legs:') and not k.startswith('qf:')},
                descriptive_only=True, intervals=None)


def digest(claims):
    """The pilot / run print: the key numbers (no verdict that the tables do not carry)."""
    sel = {}
    for e in claims['ratios']:
        if e['type'] in ('any', 'improve'):
            sel[f"{e['type']}:{e['contrast']}"] = dict(RR_D0=e['D0']['RR_MH'], p_hi=e['D0']['p_hi'], p_lo=e['D0']['p_lo'],
                                                        markets=f"{e['D0']['n_markets_rr_gt1']}>1 {e['D0']['n_markets_rr_lt1']}<1 /{e['D0']['n_markets_supported']}",
                                                        n_ev=e['D0']['n_events_informative'], RR_D1000=e['D1000']['RR_MH'], verdict=e['verdict'])
    idn = claims['identity']
    top = {k: v for k, v in idn['type_mix'].items() if k.split(':')[0] in ('all_creations', 'top3') and k.split(':')[2] in ('improve', 're_establish', 'deep')}
    return dict(population=claims['population'], ratios_any_improve=sel, verdict_lists=claims['verdict_lists'], owner_events=idn['owner_events'],
                type_mix_top3_vs_all=top, grades_top3=idn['grades_top3'], inventory_constrained=idn['inventory_constrained'],
                reconciliation=claims['reconciliation']['all'])


# ------------------------------------------------------------------ receipt, write-once ------------------------------------------------------------------
def receipt(kind, days, hashes, cnt, stages, t0, asof0, outs, git, extra=None):
    srcs = {_rel(f): _sha(f) for f in TOOL_FILES + DEP_FILES if Path(f).exists()}
    rec = dict(version=VERSION, kind=kind, contract=dict(file=_rel(CONTRACT_FILE), commit=CONTRACT_COMMIT, sha256=srcs.get(_rel(CONTRACT_FILE))),
               review=dict(file=_rel(REVIEW_FILE), commit=REVIEW_COMMIT, sha256=srcs.get(_rel(REVIEW_FILE))), module=str(Path(__file__).resolve()),
               argv=list(sys.argv), run=C.RUN, mutant=os.environ.get('MT_PO_MUT', ''), asof_start=asof0, asof_end=_now(), days=list(days),
               addresses=list(TOP3), inputs=dict(sorted(hashes.items())), sources=srcs, outputs=outs, git=git,
               counts=dict(sorted((str(k), int(v)) for k, v in cnt.items())),
               constants=dict(STEP_MS=STEP_MS, K_EXPECT=K_EXPECT, D_SENS=D_SENS, VARIANTS=VARIANTS, TYPES=TYPES, CONTRASTS=CONTRASTS, SEED=SEED,
                              N_SHIFT=N_SHIFT, MIN_EVENTS=MIN_EVENTS, MIN_MARKETS=MIN_MARKETS, MIN_MKT_S=MIN_MKT_S, P_MAX=P_MAX, C_MAX=C_MAX,
                              SHARE=f'{SHARE_NUM}/{SHARE_DEN}', TIE=TIE, THR_5S=THR_5S, FLOW_MS=FLOW_MS, SPREAD_LAB=SPREAD_LAB, DEPTH_LAB=DEPTH_LAB,
                              FG_LAB=FG_LAB, BTC_LAB=BTC_LAB, TL_LAB=TL_LAB, GRADES=GRADES, G2=G2, QF2_REQ=QF2_REQ, QF2_CAND=QF2_CAND,
                              MUTANTS=MUTANTS, PROJ_MAX_S=PROJ_MAX_S, FV_SHA256=FV_SHA256, PILOT_RUN=PILOT_RUN),
               cpu_s=dict(by_stage=dict(stages), process_total=round(time.process_time(), 3)), wall_s=round(time.time() - t0, 2),
               vmhwm_mib=round(hwm(), 1), peak_rss='VmHWM of this process', guard_mib=C.GUARD_MIB)
    rec.update(extra or {})
    return rec


def run_core(kind, days, parts, out, hashes, cnt, stages, tick, git, asof0, t0, dependency=None, extra_fn=None):
    """Shared tail of the run / pilot (and the selftest's end-to-end): panel, ratios, nulls, identity tables, claims, write-once."""
    for p in parts:
        cnt.update(p['counts'])
    frames, tables, claims, dg = analyse(kind, days, parts, cnt, dependency)
    tick('tables')
    guard('tables')

    def rec_fn(outs):
        tick('write')
        ex = dict(dependency=dependency)
        ex.update(extra_fn(stages) if extra_fn else {})
        return receipt(kind, days, hashes, cnt, stages, t0, asof0, outs, git, ex)
    rec = TL2.write_out(out, frames, tables, claims, rec_fn)
    return rec, claims, dg


def preflight_projection(path, hashes):
    """The full run launches only below the recorded pilot projection limit (<= 120 CPU s)."""
    path = Path(path)
    need(path.is_file(), f'no recorded pilot projection: {path} is missing (run the pilot first; refused before launch)')
    J = json.loads(path.read_text())
    pr = (J.get('projection') or {}).get('full_run_cpu_s')
    need(J.get('kind') == 'pilot' and J.get('version') == VERSION and isinstance(pr, (int, float)) and np.isfinite(pr),
         f'{path.name}: not a {VERSION} pilot receipt with a finite projection.full_run_cpu_s (refused before launch)')
    need(float(pr) <= PROJ_MAX_S, f'the recorded pilot projects {float(pr):.1f} CPU s for the full run > {PROJ_MAX_S:.0f} s (refused before launch)')
    hashes[_rel(path)] = _sha(path)
    return dict(pilot_receipt=_rel(path), full_run_cpu_s=float(pr), limit_s=PROJ_MAX_S)


def _load_default(day, mid, root, hc):
    return CX.load_market(day, int(mid), TOP3, root=root, hash_cache=hc)


def execute(kind, days, out, inp=None, pilot_receipt=None, require_freeze=True, log=print, loader=None, book=None, regrade_dir=None, qf2_dir=None,
            fv_sha=FV_SHA256):
    """kind 'run' (every measurement day -> v2/chains/posting_opportunity; freeze + recorded pilot projection required; queue_front_v2's
    output when it exists) or 'pilot' (its day, out given; always the v1-functions + v2-straddle identity fallback). loader / book /
    regrade_dir / qf2_dir / fv_sha are injected only by the selftest."""
    c_imp = time.process_time()
    t0, asof0 = time.time(), _now()
    need(mutant() == '', f"MT_PO_MUT={os.environ.get('MT_PO_MUT')!r} is set: production never runs a planted error (refused before any input is read)")
    for ev_ in FOREIGN_MUT:
        need(os.environ.get(ev_, '') == '', f"{ev_}={os.environ.get(ev_)!r} is set: a reused tool's planted error would enter this run (refused before "
                                            "any input is read)")
    need(C.RUN == 'v2', f'MT_RUN must be v2 (is {C.RUN!r}; refused)')
    need(kind in ('run', 'pilot'), f'kind {kind!r} (refused)')
    days = tuple(str(d) for d in days)
    need(len(days) >= 1 and len(set(days)) == len(days) and all(d in C.MEASURE_DAYS for d in days),
         'days must be distinct measurement days (refused before any path is built)')
    out = Path(out)
    need(not out.exists(), f'write-once: {out} exists (refused before any input is read)')
    need(out.parent.is_dir(), f'the output parent directory is missing: {out.parent} (refused)')
    PT = paths()
    hashes, cnt = {}, Counter()
    stages = dict(imports=round(c_imp, 3))
    last = [time.process_time()]
    cap = CPU_CAP_RUN_S if kind == 'run' else CPU_CAP_PILOT_S

    def tick(nm):
        now = time.process_time()
        stages[nm] = round(stages.get(nm, 0.) + now - last[0], 3)
        last[0] = now
        need(now < cap, f'CPU self-guard: {now:.1f} s >= {cap:.0f} s at {nm} (refused)')
    git = _git([f for f in TOOL_FILES if Path(f).exists()])
    pre = {}
    if kind == 'run':
        need(days == tuple(C.MEASURE_DAYS), 'the full run covers every measurement day (refused)')
        if require_freeze:
            need(all(Path(f).exists() for f in TOOL_FILES) and bool(git.get('unmodified')),
                 'freeze: posting_opportunity.py, its selftest, the contract and the reused tools must be tracked and unmodified (refused)')
        pre = preflight_projection(Path(pilot_receipt) if pilot_receipt else PT['pilot_receipt'], hashes)
    inp = Path(inp) if inp else C.OUT
    t3 = CX.top3(inp)
    need(tuple(a.lower() for a in t3) == TOP3_L, 'chain_context.top3 differs from the module top 3 (refused)')
    hashes[_rel(inp / 'econ' / 'identities.json')] = _sha(inp / 'econ' / 'identities.json')
    bm = CX.book_markets() if book is None else book
    need(all(d in bm for d in days), f'days {days} are not all BOOK manifest days (refused)')
    markets = [(d, int(m)) for d in days for m in sorted(int(x) for x in bm[d])]
    tick('preflight')
    RG, rginfo = read_regrade(PT['regrade'] if regrade_dir is None else regrade_dir, markets, hashes)
    qdir = Path(PT['qf2'] if qf2_dir is None else qf2_dir)
    Q2, q2info = None, None
    if kind == 'run' and qdir.exists():
        Q2, q2info = read_qf2(qdir, markets, hashes)
    dependency = dict(identity_source='queue_front_v2' if Q2 is not None else 'v1_functions_v2_straddle',
                      queue_front_v2=q2info if Q2 is not None else dict(dir=_rel(qdir), present=bool(qdir.exists()),
                                                                          used=False, why='pilot: always the fallback' if kind == 'pilot' else 'output absent'),
                      fallback='queue_front._records / _new_life on market_core inputs (replicated, parity-checked), settlement-only test with upd(f) '
                               'for upd(f-1): the v2 straddle rule (contract section 5 item 2); candidates = v2 front_identified only',
                      regrade=rginfo)
    tick('inputs')
    loader = _load_default if loader is None else loader
    ctx = CT.make_ctx(CX)
    hc, parts = {}, []
    for d, m in markets:
        M = loader(d, m, inp, hc)
        need(str(M.day) == d and int(M.mid) == m and bool(M.book), f'loaded market {M.day} {M.mid} is not the BOOK market {d} {m} (refused)')
        for k_, v_ in (M.inputs or {}).items():
            if isinstance(v_, dict) and v_.get('sha256'):
                hashes.setdefault(k_, v_['sha256'])
        tick('context_load')
        parts.append(market_pass(d, m, M, ctx, RG.get((d, m), {}), None if Q2 is None else Q2.get((d, m))))
        del M
        cnt['markets'] += 1
        tick('market')
        guard(f'market {d} {m}')
    fvp = inp / 'econ' / 'fills_valued.parquet'
    fsha = hc.get(str(fvp.resolve())) or _sha(fvp)
    hashes['econ/fills_valued.parquet'] = fsha
    need(fsha == fv_sha, f'fills_valued sha256 {fsha[:12]} differs from the accepted binding {str(fv_sha)[:12]} (refused)')
    tick('bindings')
    n_mk = len(markets)

    def extra_fn(stg):
        ex = dict(preflight=pre)
        if kind == 'pilot':
            r_mk = N_BOOK / max(1, n_mk)
            fixed = sum(stg.get(k, 0.) for k in ('imports', 'preflight', 'inputs', 'bindings'))
            scal = sum(stg.get(k, 0.) for k in ('context_load', 'market', 'tables', 'write'))
            proj = fixed + scal * r_mk
            ex['projection'] = dict(full_run_cpu_s=round(proj, 2), pilot_cpu_s=round(time.process_time(), 3), limit_s=PROJ_MAX_S,
                                    launch_allowed=bool(proj <= PROJ_MAX_S),
                                    rule=f'fixed (imports + preflight + inputs + bindings) {fixed:.2f} + (context_load + market + tables + write) '
                                         f'{scal:.2f} x {N_BOOK}/{n_mk}; the first load carries the one-off input sha256s (scaled: conservative)')
        return ex
    rec, claims, dg = run_core(kind, days, parts, out, hashes, cnt, stages, tick, git, asof0, t0, dependency, extra_fn)
    log(json.dumps(_clean(dict(out=str(out), digest=dg, projection=rec.get('projection'), cpu_s=rec['cpu_s'], vmhwm_mib=rec['vmhwm_mib'])),
                   indent=1, sort_keys=True, default=str))
    return rec


def main(argv):
    a = list(argv[1:])
    usage = 'usage: MT_RUN=v2 posting_opportunity.py run | MT_RUN=v2 posting_opportunity.py run --pilot <day> [--out <dir>]'
    if a == ['run']:
        execute('run', C.MEASURE_DAYS, paths()['out'])
        return 0
    if len(a) in (3, 5) and a[0] == 'run' and '--pilot' in a:
        opts = {}
        rest = a[1:]
        for i in range(0, len(rest), 2):
            need(rest[i] in ('--pilot', '--out') and rest[i] not in opts, usage)
            opts[rest[i]] = rest[i + 1]
        day = opts['--pilot']
        need(day in C.MEASURE_DAYS, f'pilot day {day} is not a measurement day (refused before any path is built)')
        execute('pilot', (day,), Path(opts['--out']) if '--out' in opts else pilot_out(day))
        return 0
    raise SystemExit(usage)


if __name__ == '__main__':
    sys.exit(main(sys.argv))
