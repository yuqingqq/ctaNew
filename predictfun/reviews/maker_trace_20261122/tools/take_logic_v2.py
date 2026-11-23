"""TAKE-LOGIC v2 (AMBIGUITY-AWARE) (P-2026-005 MAKER-TRACE; session aea5bfb9, 2026-11-23): built EXACTLY to
predictfun/reviews/maker_trace_20261122/take_logic_v2_contract.md (frozen before computation, commit b298933fb) - the ONE next priority of review
af916aba6 (reviews/automatic/20261123_take_logic_review/report.md, findings 1-5). It completes the USER's inventory / take-logic question; not a
new strategy or model. DESCRIPTIVE on four consumed days (Oct 24-27; G = 4 < 5: point estimates, no intervals); p is a circular-shift tail
fraction, not independent-day confidence; no fit, no replay, no model skill, no profitability claim.
UNCHANGED FROM v1 (imported from take_logic.py, never re-implemented): the population (maker-seconds: every ex-ante top-3 address x each of the
  19 BOOK markets x every second ws .. we - 1000 of the market's whole span: take_logic.market_part / assemble / population_checks), the anchor
  t < x_lo_hard_ms <= t + 1000 (take_logic.events), the state builders (q from status-ok legs with hi <= t, BTC aligned to q, time left, own
  fills 5 s, unwind edge: take_logic.variant_codes), every bin and the MH strata with rank in every pooled stratum (take_logic.def_codes /
  mh_rr / tail_counts), 500 circular shifts with seed 20261123 (take_logic.assemble's per-pair offsets), the market-support and verdict rules
  (take_logic.market_rr / verdict).
NAMING (finding 2): the anchor is a FIXED MARKET-SECOND, PRE-EARLIEST-MATCH PROXY, not an identified decision time; q(t) is RECONSTRUCTED VENUE
  inventory, not client-known inventory.
RAW TAKES every record of v2/chains/take/takes.csv (4,147; ambiguous records included) at its anchor second (4,124 on the panel; the 23 before
  the window keep their status before_population; every other status counted). Side s (+1 buy / -1 sell, YES space), size q_all / dq. A record
  flagged side_mixed has an unknown side (status side_unknown: in O3, never in O1 / O2).
OUTCOMES per maker-second, de-duplicated within the second (one indicator per outcome; outcome types are never summed):
  O1 opposing take (PRIMARY): >= 1 raw take with side = -sign(q(t)); defined only for q(t) != 0 (q = 0 and unknown q are statuses); a
     direction relative to the reconstructed anchor, not proof of a private inventory reduction. Per clock (D0 / D1000 / pend), the direction
     is derived against that clock's q.
  O2 same-direction take: >= 1 raw take with side = +sign(q(t)).    O3 any take: >= 1 raw take, at any q.
  O4 |q|-reduction across the saved feasible variants: chain_take's recorded / certain(D = 0) / simultaneous classes (takes.csv cls_rec /
     cls_cert_0 / cls_sim; take_class must equal chain_take.combine of the three on every row, else refused), fine subtypes GROUPED BEFORE
     agreement (chain_take.REDUCING = reduce_partial / full_close / reversal_reducing -> reduces; add / from_flat / overshoot / no_change ->
     not); per take reduces_all | reduces_some | reduces_none | unavailable (no available variant, or an unclassified one: chain_take
     mech_inventory's convention); per second O4 = >= 1 reduces_all take, and separately >= 1 reduces_some take (variant disagreement).
     Coverage and disagreement published by |q| bin (coverage.csv), beside v1's classified-reduce and fine-class (timing_sensitive) counts.
RATIOS the v1 set RR_against_5s / RR_with_5s / RR_size / RR_end / RR_burst / RR_edge (take_logic.RATIOS), q != 0 only, for O1, O2, O3 and O4
  (reduces_all), rank 1 and pooled, on two panels: full = the retained-ledger panel; clean = maker-seconds with no earlier excluded leg
  (take_logic's inv_ledger_conditional False at t; that D0 flag is used for every clock: the conservative, fixed subset). Clocks D0 (primary),
  D1000 and pend (sensitivities). Market consistency: the declared CRUDE within-market comparison (take_logic.market_rr).
NULLS O1 / O2: each maker-market's RAW SIDE-SPECIFIC EVENT SCHEDULE (raw takes with their sides) circularly shifted by the v1 offsets (uniform
  mod the pair's 300 instants; 500 shifts; seed 20261123); the direction re-derived against the DESTINATION second's q of the clock in use,
  de-duplicated per destination second; the MH ratio recomputed. Raw counts per pair preserved and collisions (raw takes sharing a destination
  second) counted (both checked; a refusal if the shift breaks them). O3 / O4: the v1-style shift of the outcome vector (take_logic.shifted /
  null_counts), the BENCHMARK only. p_hi = (1 + #{null >= obs}) / 501 (take_logic.tail_counts).
DECISION (declared before results) PRIMARY O1 RR_size (|q| > 200 vs (0, 50]), pooled, D0, full panel: verdict (v1 rule) not supported ->
  'inventory size as a trimming trigger: WITHDRAWN as a maker rule' (no further bins or variants); supported in all six views {full, clean} x
  {D0, D1000, pend} -> 'qualified candidate association' (not causal, not a profitable rule); otherwise 'unidentified'. Six views = the
  conservative reading of 'full panel AND clean-ledger subset AND D1000 AND pending legs' (the four named views are reported beside). Rank 1
  the same, secondary. Every other ratio is listed with its v1 verdict. Multiplicity 6 ratios x 4 outcomes x 2 scopes x 2 panels = 96.
CORRECTIONS BUNDLED census_fix.csv recomputed from v2/chains/correction_v1/census_graded.csv: the base populations counted separately (maker-fill
  rows, take records: different units, never summed) and the overlapping take attributes kept as attributes (mint, multi-record, merge,
  self-trade, sweep); the old observed_execution sum = base + attributes (a predicate per row). Stated as data: v1 reduce-classified takes whose
  side equals the D0 anchor inventory sign (takes, event-seconds) and the overshoot records. Review recount (838 / 689 / 48) as predicates.
OUTPUTS write-once into v2/chains/take_logic_v2 (run) or the pilot directory: panel.parquet (per maker-second state + outcomes), events.parquet
  (every take record: keys, side, size, anchor second, variant classes, O4 status, direction per clock), ratios.csv, coverage.csv,
  consistency.csv, census_fix.csv, claims.json (computed predicates incl. the decision readings), receipt.json (sha256 of sources, inputs and
  outputs; CPU by stage; VmHWM = peak RSS). run -> v2/chains/take_logic_v2; run --pilot <day> --out <dir> -> the pilot (prints the projection).
GUARDS MT_RUN=v2; measurement days only; MT_TL2_MUT (the selftest's planted errors shift_labels / drop_ambiguous / sum_types) and v1's MT_TL_MUT
  are refused in production; the full run refuses unless the recorded v2 pilot projection is <= 40 CPU s and the tool, its selftest, the
  contract, take_logic.py, chain_context.py and chain_take.py are tracked and unmodified (the freeze); fills_valued sha256 must equal the
  accepted binding and chain_take's receipt, every input both receipts name must agree (else refused: mismatched artifacts); VmHWM self-guard
  1,450 MiB (pf_mt_common.GUARD_MIB, never raised); CPU self-guard.
Usage (run_light, MT_RUN=v2): take_logic_v2.py run | take_logic_v2.py run --pilot 20261024 --out <dir>      selftest: take_logic_v2_selftest.py"""
import json
import os
import shutil
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
import take_logic as TL  # noqa: E402

X = CT.X
need = C.need
NAN, INF = float('nan'), float('inf')
VERSION = 'take_logic_v2'
CONTRACT_FILE = C.RECEIPT_V1 / 'take_logic_v2_contract.md'
CONTRACT_COMMIT = 'b298933fb'
TOOL_FILES = (HERE / 'take_logic_v2.py', HERE / 'take_logic_v2_selftest.py', CONTRACT_FILE, HERE / 'take_logic.py', HERE / 'chain_context.py',
              HERE / 'chain_take.py')
DEP_FILES = TL.DEP_FILES + (TL.CONTRACT_FILE,)
TOP3 = TL.TOP3
N_BOOK = TL.N_BOOK
PILOT_DAY = TL.PILOT_DAY

# ------------------------------------------------------------------ declared constants (the contract's; none fitted) ------------------------------------------------------------------
VARIANTS = TL.VARIANTS                            # D0 (primary), D1000, pend (sensitivities)
PANELS = ('full', 'clean')
OUTCOMES = ('O1', 'O2', 'O3', 'O4')
DIRECTIONAL = ('O1', 'O2')
OUTCOME_LAB = dict(O1='opposing take: >= 1 raw take with side = -sign(q(t)) (q != 0; relative to the reconstructed anchor)',
                   O2='same-direction take: >= 1 raw take with side = +sign(q(t)) (q != 0)', O3='any take: >= 1 raw take, at any q',
                   O4='|q|-reduction: >= 1 take reducing |q| in every available saved variant (reduces_all)')
NULL_DESIGN = dict(O1='raw_side_schedule_shift', O2='raw_side_schedule_shift', O3='outcome_vector_shift_benchmark', O4='outcome_vector_shift_benchmark')
RATIOS = tuple(r for r in TL.RATIOS if r[0] in TL.PRIMARY)       # RR_against_5s, RR_with_5s, RR_size, RR_end, RR_burst, RR_edge
DEFS = tuple(dict.fromkeys(r[1] for r in RATIOS))                # btc_5s, size, end, burst, edge
SCOPES2 = TL.M2_SCOPES                                           # rank1, pooled
SCOPES = TL.SCOPES
PRIMARY_KEY = ('O1', 'RR_size')
O4_ST = ('reduces_all', 'reduces_some', 'reduces_none', 'unavailable')
GROUP_RED = tuple(CT.REDUCING)                                   # reduce_partial, full_close, reversal_reducing
GROUP_NOT = ('add', 'from_flat', 'overshoot', 'no_change')
VAR_COLS = ('cls_rec', 'cls_cert_0', 'cls_sim')                  # chain_take: recorded / certain(D = 0) / simultaneous (take_class = take_class_0)
DIR_LAB = ('opposing', 'same', 'q0', 'q_unknown', 'side_unknown', 'not_in_panel')
MUTANTS = ('shift_labels', 'drop_ambiguous', 'sum_types')
NAMED_VIEWS = (('full', 'D0'), ('clean', 'D0'), ('full', 'D1000'), ('full', 'pend'))
ALL_VIEWS = tuple((pn, v) for pn in PANELS for v in VARIANTS)
READINGS = dict(
    withdrawn='inventory size as a trimming trigger: WITHDRAWN as a maker rule (no further bins or variants are tried)',
    qualified_candidate_association='qualified candidate association (inventory size <-> opposing takes); not causal, not a profitable rule',
    unidentified='unidentified (supported in some views only)')
DECISION_RULE = ('PRIMARY O1 RR_size (|q| > 200 vs (0, 50]), pooled, D0, full panel. Its v1 verdict not supported -> withdrawn; supported in all six '
                 'views {full, clean} x {D0, D1000, pend} -> qualified_candidate_association (the conservative reading of: full panel AND clean-ledger '
                 'subset AND D1000 AND pending legs; the four named views are reported beside); otherwise unidentified. Rank 1 the same (secondary).')
NAMING = dict(anchor='fixed market-second, pre-earliest-match proxy (t < x_lo_hard_ms <= t + 1000); not an identified decision time',
              inventory='q(t) = reconstructed venue inventory (status-ok legs with hi <= t); not client-known inventory',
              O1='a direction relative to the reconstructed anchor inventory, not proof that the take reduced the maker\'s private inventory')
CENSUS_EXPECT = dict(maker_fill_rows=9026, take_records=4147, mint=3709, multi_record=2722, merge=16, self_trade=38, old_observed_execution_sum=19658)
CENSUS_ATTR = (('mint', 'take_kind_mint'), ('multi_record', 'take_multi_record_order'), ('merge', 'take_kind_merge'), ('self_trade', 'take_self_trade'),
               ('sweep', 'take_sweep'))
CENSUS_TAKE_TYPES = tuple(f'take_{c}' for c in CT.CLS)
CENSUS_NEED = ('action_type', 'scope', 'day', 'subset', 'n', 'evidence_grade')
STATED_EXPECT = dict(reduce_side_equals_sign_takes=46, reduce_side_equals_sign_event_seconds=13, overshoot_records=3)
REVIEW_RECOUNT = dict(ambiguous=838, ambiguous_ge1_reducing=689, ambiguous_all_reducing=48)
CPU_CAP_RUN_S, CPU_CAP_PILOT_S = 60., 10.
PROJ_MAX_S, PROJ_TARGET_S = 40., 25.
CSV_FILES = ('ratios.csv', 'coverage.csv', 'consistency.csv', 'census_fix.csv')
PQ_FILES = ('panel.parquet', 'events.parquet')
OUT_FILES = PQ_FILES + CSV_FILES + ('claims.json', 'receipt.json')
EXTRA_READ = ('tk_hash', 'q_all', 'cls_rec', 'cls_sim', 'cls_cert_0')
TAKE_READ = TL.TAKE_READ + EXTRA_READ
STR_COLS = ('day', 'addr', 'sid', 'status', TL.CLASS_COL, 'tk_hash', 'cls_rec', 'cls_sim', 'cls_cert_0')
COV_DIGEST = ('scope', 'qbin', 'maker_seconds', 'raw_takes', 'event_seconds_O3', 'event_seconds_O1', 'event_seconds_O2', 'takes_O4_reduces_all',
              'takes_O4_reduces_some', 'takes_O4_reduces_none', 'takes_O4_unavailable', 'event_seconds_O4_reduces_all', 'event_seconds_O4_disagreement',
              'o4_coverage_takes', 'o4_disagreement_share_takes', 'event_seconds_v1_reduce', 'event_seconds_v1_timing_sensitive',
              'v1_fine_disagreement_share_takes')

_now, _rel = TL._now, TL._rel


def paths():
    """Run-time paths on the MT_RUN roots (v2: C.RECEIPT = the receipt's v2/ directory)."""
    ch = C.RECEIPT / 'chains'
    tk = ch / 'take'
    return dict(takes=tk / 'takes.csv', summary=tk / 'summary.csv', take_receipt=tk / 'receipt.json', census=ch / 'correction_v1' / 'census_graded.csv',
                out=ch / 'take_logic_v2', pilot_receipt=ch / f'take_logic_v2_pilot_{PILOT_DAY}_r1' / 'receipt.json')


def pilot_out(day):
    return C.RECEIPT / 'chains' / f'take_logic_v2_pilot_{day}_r1'


def mutant():
    """The selftest's planted-error switch MT_TL2_MUT, read at call time; production refuses any value."""
    m = os.environ.get('MT_TL2_MUT', '')
    need(m == '' or m in MUTANTS, f'MT_TL2_MUT={m!r} is not a declared mutant {MUTANTS} (refused)')
    return m


def _str_arr(col):
    """A string column as an object array of str, missing values as '' (pandas 3 str columns keep NaN)."""
    return np.array(['' if pd.isna(x) else str(x) for x in np.asarray(col, dtype=object)], dtype=object)


def clean_mask(P):
    """The clean-ledger subset: maker-seconds with no earlier excluded leg (take_logic inv_ledger_conditional False at t, every clock)."""
    return ~np.asarray(P['lg'], bool)


# ------------------------------------------------------------------ saved variant classes, O4 ------------------------------------------------------------------
def class_codes(names):
    """chain_take class names -> chain_take.CI codes; '' (absent variant) -> -1; any other name refused."""
    names = np.asarray(names, dtype=object)
    bad = sorted({str(x) for x in names if x != '' and x not in CT.CI})
    need(not bad, f'takes: variant class(es) outside chain_take.CLS {bad} (refused)')
    return np.array([CT.CI[x] if x != '' else -1 for x in names], np.int16)


def check_variant_reproduction(T):
    """take_class must equal chain_take.combine(recorded, certain(0), simultaneous) on every row (the saved variants generate it), else refused."""
    for c in VAR_COLS + (TL.CLASS_COL,):
        need(c in T.columns, f'takes: required column {c} missing (refused)')
    cr, cc, cs = (class_codes(_str_arr(T[c])) for c in VAR_COLS)
    need(bool((cr >= 0).all()), 'takes: a record without a recorded-variant class (refused)')
    got = np.array(CT.CLS, dtype=object)[CT.combine(cr, cc, cs)]
    tc = _str_arr(T[TL.CLASS_COL])
    bad = np.flatnonzero(got != tc)
    need(len(bad) == 0, f'the saved variant classes do not reproduce {TL.CLASS_COL} through chain_take.combine on {len(bad)} row(s) '
                        f'(first {bad[:3].tolist()}; refused: mismatched artifacts)')
    return dict(rows=int(len(T)), reproduces=True, n_certain_present=int((cc >= 0).sum()), n_simultaneous_present=int((cs >= 0).sum()),
                rule='take_class == chain_take.combine(cls_rec, cls_cert_0, cls_sim) on every row')


def o4_status(T):
    """Per take: codes 0 reduces_all / 1 reduces_some / 2 reduces_none / 3 unavailable from the saved recorded / certain(0) / simultaneous
    classes, fine subtypes grouped BEFORE agreement; absent variants ignored; no available variant or any other class (unclassified) ->
    unavailable. -> (status codes, number of available variants)."""
    M_ = np.vstack([class_codes(_str_arr(T[c])) for c in VAR_COLS])
    pres = M_ >= 0
    red = np.isin(M_, [CT.CI[c] for c in GROUP_RED])
    notr = np.isin(M_, [CT.CI[c] for c in GROUP_NOT])
    other = pres & ~red & ~notr
    n_av, n_red = pres.sum(0), red.sum(0)
    st = np.where(n_red == n_av, 0, np.where(n_red > 0, 1, 2))
    st = np.where((n_av == 0) | other.any(0), 3, st)
    return st.astype(np.int64), n_av.astype(np.int64)


def review_recount(T, o4):
    """The review's recount of the ambiguous records (computed; compared only on the complete 4,147-row table)."""
    amb = _str_arr(T[TL.CLASS_COL]) == 'timing_sensitive'
    d = dict(ambiguous=int(amb.sum()), ambiguous_ge1_reducing=int((amb & (o4 <= 1)).sum()), ambiguous_all_reducing=int((amb & (o4 == 0)).sum()),
             o4_status_all_records={O4_ST[i]: int((o4 == i).sum()) for i in range(len(O4_ST))})
    d['matches_review'] = bool(all(d[k] == v for k, v in REVIEW_RECOUNT.items())) if len(T) == TL.N_TAKES_ALL else None
    return d


# ------------------------------------------------------------------ raw takes, outcomes, the raw-schedule null ------------------------------------------------------------------
def raw_takes(P, T2, o4, nav):
    """The raw take schedule: every take record assigned to an anchor second (every class, ambiguous records included)."""
    mut = mutant()
    asg = np.asarray(T2['pop_status'], dtype=object) == 'assigned'
    keep = asg.copy()
    if mut == 'drop_ambiguous':                                   # planted error: ambiguous records silently dropped
        keep &= _str_arr(T2[TL.CLASS_COL]) != 'timing_sensitive'
    ix = np.flatnonzero(keep)
    flags = _str_arr(T2['status'])[ix]
    side_ok = np.array(['side_mixed' not in f.split('+') for f in flags], bool)
    return dict(ix=ix, gi=T2['gi'].to_numpy(np.int64)[ix], s=T2['s'].to_numpy(np.int64)[ix], side_ok=side_ok, o4=np.asarray(o4, np.int64)[ix],
                nav=np.asarray(nav, np.int64)[ix], ts=_str_arr(T2[TL.CLASS_COL])[ix] == 'timing_sensitive', n_assigned=int(asg.sum()), n_raw=int(len(ix)))


def outcomes(P, RT, codes):
    """Per maker-second outcome indicators (int8 0 / 1, one per second): O1 / O2 per clock, O3, O4 (reduces_all), O4_some (disagreement),
    O4_unavailable; and each raw take's direction code per clock (DIR_LAB)."""
    N = int(P['N'])
    gi, s, ok = RT['gi'], RT['s'], RT['side_ok']
    E, D = {}, {}
    for v in VARIANTS:
        q = np.asarray(codes[v]['q'], float)
        sg = np.asarray(codes[v]['sg'], np.int64)
        d = s * sg[gi]
        fin = np.isfinite(q[gi])
        D[v] = np.select([~ok, ~fin, d < 0, d > 0], [4, 3, 0, 1], 2).astype(np.int64)
        for o, c in (('O1', 0), ('O2', 1)):
            e = np.zeros(N, np.int8)
            e[gi[D[v] == c]] = 1
            E[(o, v)] = e
    e3 = np.zeros(N, np.int8)
    e3[gi] = 1
    if mutant() == 'sum_types':                                   # planted error: overlapping outcome types summed per second
        q0 = np.zeros(N, np.int8)
        q0[gi[D['D0'] >= 2]] = 1
        e3 = (E[('O1', 'D0')] + E[('O2', 'D0')] + q0).astype(np.int8)
    E['O3'] = e3
    for nm, c in (('O4', 0), ('O4_some', 1), ('O4_unavailable', 3)):
        e = np.zeros(N, np.int8)
        e[gi[RT['o4'] == c]] = 1
        E[nm] = e
    return E, D


def raw_null(P, RT, sg, DEST):
    """O1 / O2 under the raw side-specific schedule shift: every raw take moves WITH ITS SIDE by its pair's offset (DEST = take_logic.shifted of
    the raw takes' seconds), its direction is re-derived against the DESTINATION second's sign(q) (sg of the clock in use) and the events are
    de-duplicated per destination second -> {'O1': (replicate, second), 'O2': (replicate, second)} event pairs."""
    N = int(P['N'])
    sg = np.asarray(sg, np.int64)
    shp = DEST.shape
    if mutant() == 'shift_labels':                                # planted error: the labelled vector shifted (direction kept from the source)
        d = np.broadcast_to((RT['s'] * sg[RT['gi']])[None, :], shp)
    else:
        d = RT['s'][None, :] * sg[DEST]
    ok = np.broadcast_to(np.asarray(RT['side_ok'], bool)[None, :], shp)
    rows = np.broadcast_to(np.arange(TL.N_SHIFT, dtype=np.int64)[:, None], shp)
    out = {}
    for o, m in (('O1', ok & (d < 0)), ('O2', ok & (d > 0))):
        Z = np.zeros((TL.N_SHIFT, N), bool)
        Z[rows[m], DEST[m]] = True
        ri, ci = np.nonzero(Z)
        out[o] = (ri.astype(np.int64), ci.astype(np.int64))
    return out


def collision_stats(P, RT, DEST):
    """Raw takes sharing a second (observed and in every shift replicate: a bijection keeps them), mixed-side seconds, raw counts per pair."""
    gi, s = RT['gi'], RT['s']
    n, npair = int(len(gi)), len(P['pairs'])
    unit = 'raw takes sharing a (destination) second beyond the first; the shift is a bijection within its pair'
    if n == 0:
        return dict(raw_takes=0, distinct_seconds=0, collisions_extra_takes=0, seconds_with_ge2_takes=0, max_takes_per_second=0, mixed_side_seconds=0,
                    null_collisions_extra_takes_min=0, null_collisions_extra_takes_max=0, collisions_invariant_under_shift=True,
                    raw_counts_per_pair_preserved=True, shifted_takes_stay_in_pair=True, unit=unit)
    u, inv, cnt = np.unique(gi, return_inverse=True, return_counts=True)
    inv = np.asarray(inv).reshape(-1)
    hb, hs = np.zeros(len(u), bool), np.zeros(len(u), bool)
    hb[inv[s > 0]] = True
    hs[inv[s < 0]] = True
    srt = np.sort(DEST, axis=1)
    extra_null = n - (1 + (srt[:, 1:] != srt[:, :-1]).sum(1))
    src = P['pair'][gi]
    dst = P['pair'][DEST]
    base = np.bincount(src, minlength=npair)
    kept = all(np.array_equal(np.bincount(dst[r], minlength=npair), base) for r in range(TL.N_SHIFT))
    return dict(raw_takes=n, distinct_seconds=int(len(u)), collisions_extra_takes=int(n - len(u)), seconds_with_ge2_takes=int((cnt >= 2).sum()),
                max_takes_per_second=int(cnt.max()), mixed_side_seconds=int((hb & hs).sum()),
                null_collisions_extra_takes_min=int(extra_null.min()), null_collisions_extra_takes_max=int(extra_null.max()),
                collisions_invariant_under_shift=bool(np.all(extra_null == n - len(u))), raw_counts_per_pair_preserved=bool(kept),
                shifted_takes_stay_in_pair=bool(np.all(dst == src[None, :])), unit=unit)


# ------------------------------------------------------------------ ratios (v1 machinery, v2 outcomes / panels / nulls) ------------------------------------------------------------------
def panel_def(rank, Cv, dn, pn, lg):
    """take_logic.def_codes; the clean panel drops the ledger-conditional maker-seconds (code and group -1)."""
    code, grp, S, G = TL.def_codes(rank, Cv, dn)
    if pn == 'clean':
        lg = np.asarray(lg, bool)
        code, grp = np.where(lg, -1, code), np.where(lg, -1, grp)
    return code, grp, S, G


def obs_counts(code, S, G, Ev):
    """Person-time (S, G) and event-seconds (S, G) of the coded maker-seconds (Ev: the outcome indicator, used as weights)."""
    code = np.asarray(code, np.int64)
    ok = code >= 0
    pt = np.bincount(code[ok], minlength=S * G).reshape(S, G).astype(float)
    ob = np.bincount(code[ok], weights=np.asarray(Ev, float)[ok], minlength=S * G).reshape(S, G)
    return pt, ob


def pair_counts(ri, ci, code, S, G):
    """Null event-seconds per (replicate, stratum, group) from (replicate, second) event pairs."""
    c = np.asarray(code, np.int64)[ci]
    ok = c >= 0
    return np.bincount(ri[ok] * (S * G) + c[ok], minlength=TL.N_SHIFT * S * G).reshape(TL.N_SHIFT, S, G).astype(float)


def ratio_row(P, Cv, lg, pn, v, sc, o, rn, dn, g, exl, rfl, code, grp, S, pt, obs, nul, Ev, rank, mkt, n_mk):
    """One ratio: MH RR, its null tail counts, support, the crude market consistency and the v1 verdict -> (row, consistency rows)."""
    sm = (np.arange(S) < S // 3) if sc == 'rank1' else np.ones(S, bool)
    T1, T0 = pt[sm, g], pt[sm, 0]
    Aa = np.vstack([obs[sm, g][None, :], nul[:, sm, g]])
    Bb = np.vstack([obs[sm, 0][None, :], nul[:, sm, 0]])
    rr = TL.mh_rr(Aa, Bb, T1, T0)[0]
    rr_obs, nl = float(rr[0]), rr[1:]
    c_hi, c_lo = TL.tail_counts(rr_obs, nl)
    inf_s = (T1 > 0) & (T0 > 0)
    a_t, b_t, T1t, T0t = float(obs[sm, g].sum()), float(obs[sm, 0].sum()), float(T1.sum()), float(T0.sum())
    n_ev = int(round(float((obs[sm, g] + obs[sm, 0])[inf_s].sum())))
    rmask = TL._scope_mask(rank, sc)
    pmask = rmask & ~lg if pn == 'clean' else rmask
    mc = TL.market_rr(grp, g, rmask, mkt, n_mk, Ev)
    n_sup, n_gt1 = int(mc['sup'].sum()), int(mc['gt1'].sum())
    if T1t > 0 and T0t > 0 and b_t > 0:
        crude = (a_t / T1t) / (b_t / T0t)
    else:
        crude = INF if (a_t > 0 and T1t > 0) else NAN
    fin = nl[np.isfinite(nl)]
    qn = np.quantile(fin, [.05, .5, .95]) if len(fin) else [NAN] * 3
    qb = Cv['qb']
    row = dict(panel=pn, variant=v, scope=sc, outcome=o, ratio=rn, definition=dn,
               primary_decision=bool((o, rn) == PRIMARY_KEY and pn == 'full' and v == 'D0' and sc == 'pooled'),
               outcome_definition=OUTCOME_LAB[o], null_design=NULL_DESIGN[o], exposed=exl, reference=rfl, strata=TL.STRATA_LAB[dn],
               RR_MH=rr_obs, RR_crude=float(crude), a_exposed=a_t, T_exposed=T1t, b_reference=b_t, T_reference=T0t, n_strata=int(sm.sum()),
               n_strata_informative=int(inf_s.sum()), n_events_informative=n_ev,
               maker_seconds_q0_excluded=int((pmask & (qb == 0)).sum()), maker_seconds_other_excluded=int((pmask & (qb > 0) & (grp != 0) & (grp != g)).sum()),
               maker_seconds_ledger_excluded=int((rmask & lg).sum()) if pn == 'clean' else 0,
               null_n=int(len(nl)), null_n_nan=int(np.isnan(nl).sum()), null_n_inf=int(np.isinf(nl).sum()),
               null_mean=float(fin.mean()) if len(fin) else NAN, null_p05=float(qn[0]), null_p50=float(qn[1]), null_p95=float(qn[2]),
               c_hi=-1 if c_hi is None else int(c_hi), c_lo=-1 if c_lo is None else int(c_lo),
               p_hi=NAN if c_hi is None else c_hi / (TL.N_SHIFT + 1), p_lo=NAN if c_lo is None else c_lo / (TL.N_SHIFT + 1),
               n_markets_supported=n_sup, n_markets_rr_gt1=n_gt1, market_share_rr_gt1=(n_gt1 / n_sup) if n_sup else NAN,
               market_consistency='crude within-market comparison (declared)', verdict_rule=TL.verdict(rr_obs, c_hi, n_ev, n_sup, n_gt1))
    cr = []
    for mi in range(n_mk):
        mm = P['markets'][mi]
        cr.append(dict(row='market', panel=pn, variant=v, scope=sc, outcome=o, ratio=rn, day=mm['day'], mid=mm['mid'], a=float(mc['a'][mi]),
                       T1=float(mc['T1'][mi]), b=float(mc['b'][mi]), T0=float(mc['T0'][mi]), rr_crude=float(mc['rr'][mi]),
                       supported=bool(mc['sup'][mi]), rr_gt1=bool(mc['gt1'][mi])))
    for di, d in enumerate(P['days']):
        dm = P['mday'] == di
        ns, ng = int(mc['sup'][dm].sum()), int(mc['gt1'][dm].sum())
        cr.append(dict(row='summary', panel=pn, variant=v, scope=sc, outcome=o, ratio=rn, day=d, n_supported=ns, n_rr_gt1=ng, share=(ng / ns) if ns else NAN))
    cr.append(dict(row='summary', panel=pn, variant=v, scope=sc, outcome=o, ratio=rn, day='all', n_supported=n_sup, n_rr_gt1=n_gt1,
                   share=(n_gt1 / n_sup) if n_sup else NAN))
    return row, cr


def ratio_tables(P, codes, E, RT, DEST, views=ALL_VIEWS, outs=OUTCOMES):
    """Every ratio x outcome x scope for the (panel, clock) views: MH RR, null (O1 / O2 raw-schedule shift; O3 / O4 the outcome-vector
    benchmark), market consistency -> (rows keyed (panel, clock, scope, outcome, ratio), consistency rows)."""
    rank, mkt = P['rank'].astype(np.int64), P['mkt'].astype(np.int64)
    n_mk = len(P['markets'])
    lg = np.asarray(P['lg'], bool)
    SH = {o: TL.shifted(P, np.flatnonzero(E[o])) for o in outs if o not in DIRECTIONAL}
    RK, cons = {}, []
    for v in VARIANTS:
        pns = [pn for pn in PANELS if (pn, v) in views]
        if not pns:
            continue
        Cv = codes[v]
        RN = raw_null(P, RT, Cv['sg'], DEST) if any(o in DIRECTIONAL for o in outs) else {}
        for pn in pns:
            for dn in DEFS:
                code, grp, S, G = panel_def(rank, Cv, dn, pn, lg)
                for o in outs:
                    Ev = np.asarray(E[(o, v)] if o in DIRECTIONAL else E[o], float)
                    pt, obs = obs_counts(code, S, G, Ev)
                    nul = pair_counts(RN[o][0], RN[o][1], code, S, G) if o in DIRECTIONAL else TL.null_counts(code, S, G, SH[o])
                    for rn, dn_, g, exl, rfl in RATIOS:
                        if dn_ != dn:
                            continue
                        for sc in SCOPES2:
                            row, cr = ratio_row(P, Cv, lg, pn, v, sc, o, rn, dn, g, exl, rfl, code, grp, S, pt, obs, nul, Ev, rank, mkt, n_mk)
                            RK[(pn, v, sc, o, rn)] = row
                            cons.extend(cr)
    return RK, cons


def decision(RK, scope):
    """The declared reading of O1 RR_size for a scope (pooled primary, rank 1 secondary) from the v1 verdicts of the six views."""
    vv = {(pn, v): RK[(pn, v, scope, 'O1', 'RR_size')]['verdict_rule'] for pn, v in ALL_VIEWS}
    prim = vv[('full', 'D0')]
    four = all(vv[k] == 'supported' for k in NAMED_VIEWS)
    six = all(vv[k] == 'supported' for k in ALL_VIEWS)
    key = 'withdrawn' if prim != 'supported' else ('qualified_candidate_association' if six else 'unidentified')
    r = RK[('full', 'D0', scope, 'O1', 'RR_size')]
    return dict(scope=scope, role='primary' if scope == 'pooled' else 'secondary', outcome='O1', ratio='RR_size', reading=key, reading_text=READINGS[key],
                views={f'{pn}:{v}': vv[(pn, v)] for pn, v in ALL_VIEWS},
                view_RR={f'{pn}:{v}': RK[(pn, v, scope, 'O1', 'RR_size')]['RR_MH'] for pn, v in ALL_VIEWS},
                view_p_hi={f'{pn}:{v}': RK[(pn, v, scope, 'O1', 'RR_size')]['p_hi'] for pn, v in ALL_VIEWS},
                primary=dict(RR_MH=r['RR_MH'], RR_crude=r['RR_crude'], p_hi=r['p_hi'], n_events_informative=r['n_events_informative'],
                             n_markets_supported=r['n_markets_supported'], n_markets_rr_gt1=r['n_markets_rr_gt1'], verdict=prim),
                predicates=dict(primary_supported=prim == 'supported', four_named_views_supported=four, all_six_views_supported=six,
                                clean_D1000_and_clean_pend_supported=all(vv[k] == 'supported' for k in (('clean', 'D1000'), ('clean', 'pend')))),
                rule=DECISION_RULE)


# ------------------------------------------------------------------ coverage, census, statements ------------------------------------------------------------------
def coverage_table(P, codes, E, RT, D, EV1):
    """Raw takes, O1 / O2 / O3 / O4 counts, O4 coverage and grouped disagreement by |q| bin (panel x clock x scope), beside v1's classified
    reduce and fine-class (timing_sensitive) event-seconds."""
    lg = np.asarray(P['lg'], bool)
    gi = RT['gi']
    labs = TL.Q_LAB + ('unknown',)
    rows = []
    for pn in PANELS:
        pm = ~lg if pn == 'clean' else np.ones(int(P['N']), bool)
        for v in VARIANTS:
            q = np.asarray(codes[v]['q'], float)
            lev = np.where(np.isfinite(q), codes[v]['qb'].astype(np.int64), len(TL.Q_LAB))
            d = D[v]
            for sc in SCOPES:
                sm = TL._scope_mask(P['rank'], sc) & pm
                for L in range(len(labs)):
                    ms = sm & (lev == L)
                    if L == len(TL.Q_LAB) and not ms.any():
                        continue
                    tk = ms[gi]
                    n_tk = int(tk.sum())
                    r = dict(panel=pn, variant=v, scope=sc, qbin=labs[L], maker_seconds=int(ms.sum()), raw_takes=n_tk,
                             event_seconds_O3=int(E['O3'][ms].sum()), takes_O1=int((tk & (d == 0)).sum()), event_seconds_O1=int(E[('O1', v)][ms].sum()),
                             takes_O2=int((tk & (d == 1)).sum()), event_seconds_O2=int(E[('O2', v)][ms].sum()), takes_q0=int((tk & (d == 2)).sum()),
                             takes_side_unknown=int((tk & (d == 4)).sum()))
                    for i, st in enumerate(O4_ST):
                        r[f'takes_O4_{st}'] = int((tk & (RT['o4'] == i)).sum())
                    r.update(event_seconds_O4_reduces_all=int(E['O4'][ms].sum()), event_seconds_O4_disagreement=int(E['O4_some'][ms].sum()),
                             event_seconds_O4_unavailable=int(E['O4_unavailable'][ms].sum()), takes_v1_timing_sensitive=int((tk & RT['ts']).sum()),
                             event_seconds_v1_reduce=int(EV1['E_reduce'][ms].sum()), event_seconds_v1_timing_sensitive=int(EV1['E_timing_sensitive'][ms].sum()))
                    r['o4_coverage_takes'] = (r['takes_O4_reduces_all'] + r['takes_O4_reduces_none']) / n_tk if n_tk else NAN
                    r['o4_disagreement_share_takes'] = r['takes_O4_reduces_some'] / n_tk if n_tk else NAN
                    r['v1_fine_disagreement_share_takes'] = r['takes_v1_timing_sensitive'] / n_tk if n_tk else NAN
                    rows.append(r)
    return pd.DataFrame(rows)


def read_census(PT, hashes):
    p = PT['census']
    need(p.is_file(), f'input missing: {p} (refused)')
    hashes[_rel(p)] = X._sha(p)
    hdr = pd.read_csv(p, nrows=0).columns
    for c in CENSUS_NEED:
        need(c in hdr, f'census_graded.csv: required column {c} missing (refused)')
    return pd.read_csv(p, usecols=list(CENSUS_NEED), dtype=dict(action_type=str, scope=str, day=str, subset=str, evidence_grade=str))


def census_fix(CEN):
    """census_graded rows -> per scope x day: the base populations (maker-fill rows, take records; different units, never summed), the
    overlapping take attributes, the old observed_execution sum and its decomposition (computed)."""
    for c in CENSUS_NEED:
        need(c in CEN.columns, f'census: required column {c} missing (refused)')
    at, sc, dy, eg = (_str_arr(CEN[c]) for c in ('action_type', 'scope', 'day', 'evidence_grade'))
    n = pd.to_numeric(CEN['n'], errors='coerce').to_numpy(float)
    need(bool(np.isfinite(n).all()), 'census: a non-numeric n (refused)')
    rows = []
    for s_, d_ in sorted(set(zip(sc.tolist(), dy.tolist()))):
        m = (sc == s_) & (dy == d_)
        mf = float(n[m & (at == 'maker_fill')].sum())
        tr = float(n[m & np.isin(at, CENSUS_TAKE_TYPES)].sum())
        attrs = {k: float(n[m & (at == a)].sum()) for k, a in CENSUS_ATTR}
        old = float(n[m & (eg == 'observed_execution')].sum())
        a_tot = float(sum(attrs.values()))
        rest = old - mf - tr - a_tot
        r = dict(scope=s_, day=d_, maker_fill_rows=int(round(mf)), take_records=int(round(tr)))
        r.update({f'attr_{k}': int(round(v)) for k, v in attrs.items()})
        r.update(attribute_incidences=int(round(a_tot)), old_observed_execution_sum=int(round(old)), old_minus_base_minus_attributes=int(round(rest)),
                 old_sum_is_base_plus_attributes=bool(abs(rest) < 1e-9),
                 units='maker-fill rows = maker legs; take records = aggregated take records; attributes overlap take records (not executions)')
        rows.append(r)
    return pd.DataFrame(rows)


def census_claims(CF, n_take_rows):
    r = CF[(CF['scope'] == 'pooled') & (CF['day'] == 'all')] if len(CF) else CF
    if len(r) != 1:
        return dict(pooled_all=None, predicates=dict(pooled_all_row_present=False))
    row = r.iloc[0]
    got = dict(maker_fill_rows=int(row['maker_fill_rows']), take_records=int(row['take_records']), mint=int(row['attr_mint']),
               multi_record=int(row['attr_multi_record']), merge=int(row['attr_merge']), self_trade=int(row['attr_self_trade']), sweep=int(row['attr_sweep']),
               attribute_incidences=int(row['attribute_incidences']), old_observed_execution_sum=int(row['old_observed_execution_sum']))
    pred = {f'{k}_is_{v}': bool(got[k] == v) for k, v in CENSUS_EXPECT.items()}
    pred.update(pooled_all_row_present=True, take_records_equal_takes_csv_rows=bool(got['take_records'] == int(n_take_rows)),
                every_row_old_sum_is_base_plus_attributes=bool(np.asarray(CF['old_sum_is_base_plus_attributes'], bool).all()))
    return dict(pooled_all=got, contract_numbers=dict(CENSUS_EXPECT), predicates=pred,
                statement='base populations counted separately: maker-fill rows (maker legs) and take records (aggregated take records) are different '
                          'units, never summed; mint / multi-record / merge / self-trade / sweep are overlapping attributes of take records, not executions')


def stated_as_data(P, C0, T2, kind):
    """v1 reduce-classified takes (overshoot included, v1's mapping) whose side equals the D0 anchor inventory sign; the overshoot records."""
    asg = np.asarray(T2['pop_status'], dtype=object) == 'assigned'
    gi = T2['gi'].to_numpy(np.int64)
    s = T2['s'].to_numpy(np.int64)
    red = np.asarray(T2['type'], dtype=object) == 'reduce'
    sg = np.zeros(len(T2), np.int64)
    sg[asg] = np.asarray(C0['sg'], np.int64)[gi[asg]]
    m = asg & red & (sg != 0) & (s == sg)
    d = dict(reduce_side_equals_sign_takes=int(m.sum()), reduce_side_equals_sign_event_seconds=int(len(np.unique(gi[m]))),
             overshoot_records=int((_str_arr(T2[TL.CLASS_COL]) == 'overshoot').sum()), clock='D0',
             definition='v1 type reduce (reduce_partial / full_close / reversal_reducing / overshoot) with side == sign(q(t)) at the anchor; '
                        'stated as data, not removed')
    d['expected_full_run'] = dict(STATED_EXPECT)
    d['matches_expected'] = bool(all(d[k] == v for k, v in STATED_EXPECT.items())) if kind == 'run' else None
    return d


# ------------------------------------------------------------------ frames ------------------------------------------------------------------
def panel_frame(P, codes, E, EV1):
    pr = P['pair']
    cols = dict(day=np.array([pm['day'] for pm in P['pairs']], dtype=object)[pr], mid=np.array([pm['mid'] for pm in P['pairs']], np.int64)[pr],
                rank=P['rank'], pair=pr, mkt=P['mkt'], k=P['k'], t_ms=P['t'])
    for f in TL.MS_FIELDS + TL.PS_FIELDS:
        cols[f] = P[f]
    cols['clean'] = clean_mask(P)
    for v in VARIANTS:
        cols[f'qbin_{v}'] = codes[v]['qb']
        cols[f'O1_{v}'] = E[('O1', v)]
        cols[f'O2_{v}'] = E[('O2', v)]
    cols.update(O3=E['O3'], O4_reduces_all=E['O4'], O4_disagreement=E['O4_some'], O4_unavailable=E['O4_unavailable'])
    for t_ in TL.TYPES_ANY:
        cols[f'v1_E_{t_}'] = EV1['E_' + t_].astype(np.int8)
        cols[f'v1_N_{t_}'] = EV1['N_' + t_].astype(np.int32)
    return pd.DataFrame(cols)


def events_frame(P, T2, RT, D, codes, o4, nav):
    n = len(T2)
    cols = {c: T2[c].to_numpy() for c in TAKE_READ if c in T2.columns}
    for c in STR_COLS:
        if c in T2.columns:
            cols[c] = _str_arr(T2[c])
    st = np.asarray(T2['pop_status'], dtype=object)
    gi = T2['gi'].to_numpy(np.int64)
    asg = st == 'assigned'
    cols.update(v1_type=np.asarray(T2['type'], dtype=object), pop_status=st, gi=gi)
    k, ta = np.full(n, -1, np.int64), np.full(n, -1, np.int64)
    k[asg], ta[asg] = P['k'][gi[asg]], P['t'][gi[asg]]
    inr = np.zeros(n, bool)
    inr[RT['ix']] = True
    lgx = np.zeros(n, bool)
    lgx[asg] = np.asarray(P['lg'], bool)[gi[asg]]
    cols.update(k_anchor=k, t_anchor_ms=ta, in_raw_schedule=inr, o4_status=np.array(O4_ST, dtype=object)[np.asarray(o4, np.int64)],
                n_variants_available=np.asarray(nav, np.int64), ledger_conditional_at_anchor=lgx)
    for v in VARIANTS:
        qv = np.full(n, NAN)
        qv[asg] = np.asarray(codes[v]['q'], float)[gi[asg]]
        dv = np.full(n, len(DIR_LAB) - 1, np.int64)
        dv[RT['ix']] = D[v]
        cols[f'q_{v}'] = qv
        cols[f'dir_{v}'] = np.array(DIR_LAB, dtype=object)[dv]
    return pd.DataFrame(cols)


# ------------------------------------------------------------------ claims, receipt, write-once ------------------------------------------------------------------
def make_claims(kind, days, P, T2, RT, E, D, codes, RK, dec, popc, repro, recount, colls, cen, stated, o4):
    lg = np.asarray(P['lg'], bool)
    N = int(P['N'])
    pmask = {pn: (~lg if pn == 'clean' else np.ones(N, bool)) for pn in PANELS}
    ms = {pn: {sc: int((TL._scope_mask(P['rank'], sc) & pmask[pn]).sum()) for sc in SCOPES} for pn in PANELS}
    ev = {}
    for pn in PANELS:
        pm = pmask[pn]
        tk_in = pm[RT['gi']]
        e_ = {}
        for v in VARIANTS:
            for o, c in (('O1', 0), ('O2', 1)):
                e_[f'{o}:{v}'] = dict(event_seconds=int(E[(o, v)][pm].sum()), takes=int(((D[v] == c) & tk_in).sum()))
        e_['O3'] = dict(event_seconds=int(E['O3'][pm].sum()), takes=int(tk_in.sum()))
        for nm, c in (('O4_reduces_all', 0), ('O4_disagreement', 1), ('O4_unavailable', 3)):
            key = {'O4_reduces_all': 'O4', 'O4_disagreement': 'O4_some', 'O4_unavailable': 'O4_unavailable'}[nm]
            e_[nm] = dict(event_seconds=int(E[key][pm].sum()), takes=int(((RT['o4'] == c) & tk_in).sum()))
        ev[pn] = e_
    statuses = dict(
        directions_raw_takes={v: {DIR_LAB[i]: int((D[v] == i).sum()) for i in range(len(DIR_LAB) - 1)} for v in VARIANTS},
        q0_maker_seconds={v: {sc: int((TL._scope_mask(P['rank'], sc) & (codes[v]['qb'] == 0) & np.isfinite(codes[v]['q'])).sum()) for sc in SCOPES}
                          for v in VARIANTS},
        q_unknown_maker_seconds={v: int((~np.isfinite(np.asarray(codes[v]['q'], float))).sum()) for v in VARIANTS},
        ledger_conditional=dict(maker_seconds=int(lg.sum()), raw_takes=int(lg[RT['gi']].sum()), event_seconds_O3=int(E['O3'][lg].sum()),
                                event_seconds_O4=int(E['O4'][lg].sum()), event_seconds_O1_D0=int(E[('O1', 'D0')][lg].sum()),
                                event_seconds_O2_D0=int(E[('O2', 'D0')][lg].sum())))
    ratios = [dict(key=f"{r['panel']}|{r['variant']}|{r['scope']}|{r['outcome']}|{r['ratio']}", RR_MH=r['RR_MH'], p_hi=r['p_hi'],
                   n_events_informative=r['n_events_informative'], markets_rr_gt1=f"{r['n_markets_rr_gt1']}/{r['n_markets_supported']}",
                   verdict=r['verdict_rule']) for r in RK.values()]
    n_tests = len(RATIOS) * len(OUTCOMES) * len(SCOPES2) * len(PANELS)
    return dict(
        version=VERSION, kind=kind, contract=dict(file=_rel(CONTRACT_FILE), commit=CONTRACT_COMMIT), days=list(days), naming=NAMING, reproduction=repro,
        review_recount=recount, population=popc, maker_seconds=ms,
        takes=dict(records=int(len(T2)), pop_status=dict(Counter(str(x) for x in np.asarray(T2['pop_status'], dtype=object))),
                   raw_schedule=dict(n_assigned=RT['n_assigned'], n_raw=RT['n_raw'], dropped=RT['n_assigned'] - RT['n_raw'],
                                     all_assigned_kept=bool(RT['n_raw'] == RT['n_assigned'])),
                   o4_status_records={O4_ST[i]: int((np.asarray(o4) == i).sum()) for i in range(len(O4_ST))},
                   o4_status_raw={O4_ST[i]: int((RT['o4'] == i).sum()) for i in range(len(O4_ST))}),
        events=ev, statuses=statuses, collisions=colls, decision=dec, ratios=ratios,
        multiplicity=dict(ratios=len(RATIOS), outcomes=len(OUTCOMES), scopes=len(SCOPES2), panels=len(PANELS), tests=n_tests,
                          clocks='D0 primary; D1000 and pend sensitivities (not counted)', all_listed=True, selected=None),
        null=dict(O1_O2='raw side-specific schedule: each pair\'s raw takes shifted with their sides by its offset, the direction re-derived at the '
                        'destination second\'s q of the clock in use, de-duplicated per destination second',
                  O3_O4='v1-style shift of the outcome (event-second) vector, the benchmark only', n_shifts=TL.N_SHIFT, seed=TL.SEED,
                  offsets='take_logic.assemble: Generator(PCG64(SeedSequence([seed, day, mid, rank]))).integers(0, n) per pair (n = 300 instants)',
                  p_hi='(1 + #{null >= obs}) / (n_shifts + 1); NaN replicates counted on both sides; ties within 1e-12 relative'),
        rules=dict(verdict='take_logic.verdict (v1): insufficient_support (< 5 supported markets or < 30 informative event-seconds) | supported (RR > 1, '
                           'p_hi <= 0.05, RR > 1 in >= 2/3 of the supported markets) | inconsistent | not_supported',
                   min_events=TL.MIN_EVENTS, min_supported_markets=TL.MIN_MARKETS, min_market_maker_seconds=TL.MIN_MKT_S, p_max=TL.P_MAX,
                   c_hi_max=TL.C_HI_MAX, market_share=f'{TL.SHARE_NUM}/{TL.SHARE_DEN}', decision=DECISION_RULE,
                   clean_panel='maker-seconds with no earlier excluded leg (inv_ledger_conditional False at t; the D0 flag for every clock)'),
        census_fix=cen, stated_as_data=stated, descriptive_only=True, intervals=None)


def receipt(kind, days, hashes, cnt, stages, t0, asof0, outs, git, extra=None):
    srcs = {_rel(f): X._sha(f) for f in TOOL_FILES + DEP_FILES if Path(f).exists()}
    rec = dict(version=VERSION, kind=kind, contract=dict(file=_rel(CONTRACT_FILE), commit=CONTRACT_COMMIT, sha256=srcs.get(_rel(CONTRACT_FILE))),
               module=str(Path(__file__).resolve()), run=C.RUN, mutant=os.environ.get('MT_TL2_MUT', ''), asof_start=asof0, asof_end=_now(),
               days=list(days), addresses=list(TOP3), inputs=dict(sorted(hashes.items())), sources=srcs, outputs=outs, git=git,
               counts=dict(sorted((str(k), v) for k, v in cnt.items())),
               constants=dict(VARIANTS=VARIANTS, PANELS=PANELS, OUTCOMES=OUTCOMES, RATIOS=[r[0] for r in RATIOS], DEFS=DEFS, SCOPES=SCOPES2,
                              N_SHIFT=TL.N_SHIFT, SEED=TL.SEED, POS_EPS=TL.POS_EPS, Q_LAB=TL.Q_LAB, MIN_EVENTS=TL.MIN_EVENTS,
                              MIN_MARKETS=TL.MIN_MARKETS, MIN_MKT_S=TL.MIN_MKT_S, P_MAX=TL.P_MAX, C_HI_MAX=TL.C_HI_MAX, VAR_COLS=VAR_COLS,
                              GROUP_RED=GROUP_RED, GROUP_NOT=GROUP_NOT, O4_ST=O4_ST, DIR_LAB=DIR_LAB, NULL_DESIGN=NULL_DESIGN,
                              PROJ_MAX_S=PROJ_MAX_S, PROJ_TARGET_S=PROJ_TARGET_S, CENSUS_EXPECT=CENSUS_EXPECT, STATED_EXPECT=STATED_EXPECT,
                              REVIEW_RECOUNT=REVIEW_RECOUNT),
               cpu_s=dict(by_stage=dict(stages), process_total=round(time.process_time(), 3)), wall_s=round(time.time() - t0, 2),
               vmhwm_mib=round(CT.hwm(), 1), peak_rss='VmHWM of this process', guard_mib=C.GUARD_MIB)
    rec.update(extra or {})
    return rec


def write_out(out, frames, tables, claims, rec_fn):
    """Write-once: every file in a partial directory, the receipt last (it hashes the others), then one rename."""
    out = Path(out)
    need(not out.exists(), f'write-once: {out} exists (refused)')
    tmp = out.with_name(f'.{out.name}.partial{os.getpid()}')
    need(not tmp.exists(), f'partial directory {tmp} exists (refused)')
    tmp.mkdir(parents=True)
    try:
        outs = {}
        for nm, df in frames.items():
            df.to_parquet(tmp / nm, index=False)
            outs[nm] = dict(rows=int(len(df)))
        for nm, df in tables.items():
            df.to_csv(tmp / nm, index=False)
            outs[nm] = dict(rows=int(len(df)))
        X._json(tmp / 'claims.json', claims)
        outs['claims.json'] = {}
        for nm in outs:
            outs[nm].update(sha256=X._sha(tmp / nm), bytes=int((tmp / nm).stat().st_size))
        rec = rec_fn(outs)
        X._json(tmp / 'receipt.json', rec)
        need(not out.exists(), f'write-once: {out} appeared during the run (refused)')
        os.rename(tmp, out)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    return rec


def run_core(kind, days, parts, T, out, hashes, cnt, stages, tick, git, asof0, t0, expect_pairs, CEN, repro=None, recount=None, n_take_rows=None,
             extra_fn=None):
    """Shared tail of the run / pilot (and the selftest's end-to-end): panel, population check, raw takes, outcomes, ratios, nulls, tables,
    claims, write-once."""
    P = TL.assemble(parts)
    tick('assemble')
    popc = TL.population_checks(P, expect_pairs)
    need(popc['complete'], f'population check failed (refused): {popc}')
    T2, EV1 = TL.events(P, T)
    for k_, v_ in Counter(str(x) for x in T2['pop_status']).items():
        cnt[f'takes_{k_}'] += v_
    cnt['maker_seconds'] = int(P['N'])
    o4, nav = o4_status(T2)
    RT = raw_takes(P, T2, o4, nav)
    need(RT['n_raw'] == RT['n_assigned'], f"raw schedule {RT['n_raw']} != assigned takes {RT['n_assigned']} (refused: a take record was dropped)")
    codes = {v: TL.variant_codes(P, v) for v in VARIANTS}
    E, D = outcomes(P, RT, codes)
    need(all(int(np.max(e, initial=0)) <= 1 for e in E.values()), 'an outcome indicator above 1 (refused: outcome types summed)')
    DEST = TL.shifted(P, RT['gi'])
    colls = collision_stats(P, RT, DEST)
    need(colls['raw_counts_per_pair_preserved'] and colls['shifted_takes_stay_in_pair'] and colls['collisions_invariant_under_shift'],
         f'the raw-schedule shift broke an invariant (refused): {colls}')
    tick('events')
    RK, cons = ratio_tables(P, codes, E, RT, DEST)
    del DEST
    tick('nulls')
    CT.guard('nulls')
    cov = coverage_table(P, codes, E, RT, D, EV1)
    CF = census_fix(CEN)
    cen = census_claims(CF, len(T) if n_take_rows is None else n_take_rows)
    stated = stated_as_data(P, codes['D0'], T2, kind)
    dec = {sc: decision(RK, sc) for sc in ('pooled', 'rank1')}
    claims = make_claims(kind, days, P, T2, RT, E, D, codes, RK, dec, popc, repro, recount, colls, cen, stated, o4)
    frames = {'panel.parquet': panel_frame(P, codes, E, EV1), 'events.parquet': events_frame(P, T2, RT, D, codes, o4, nav)}
    tables = {'ratios.csv': pd.DataFrame(list(RK.values())), 'coverage.csv': cov, 'consistency.csv': pd.DataFrame(cons), 'census_fix.csv': CF}
    tick('tables')
    CT.guard('tables')

    def rec_fn(outs):
        tick('write')
        return receipt(kind, days, hashes, cnt, stages, t0, asof0, outs, git, extra_fn(stages) if extra_fn else None)
    rec = write_out(out, frames, tables, claims, rec_fn)
    return rec, claims, dict(RK=RK, coverage=cov, collisions=colls, decision=dec, census=CF)


# ------------------------------------------------------------------ run / pilot ------------------------------------------------------------------
def preflight_projection(path, hashes):
    """The full run starts only below the recorded v2 pilot projection limit (<= 40 CPU s)."""
    path = Path(path)
    need(path.is_file(), f'no recorded pilot projection: {path} is missing (run the pilot first; refused before launch)')
    J = json.loads(path.read_text())
    pr = (J.get('projection') or {}).get('full_run_cpu_s')
    need(J.get('kind') == 'pilot' and J.get('version') == VERSION and isinstance(pr, (int, float)) and np.isfinite(pr),
         f'{path.name}: not a {VERSION} pilot receipt with a finite projection.full_run_cpu_s (refused before launch)')
    need(float(pr) <= PROJ_MAX_S, f'the recorded pilot projects {float(pr):.1f} CPU s for the full run > {PROJ_MAX_S:.0f} s (refused before launch)')
    hashes[_rel(path)] = X._sha(path)
    return dict(pilot_receipt=_rel(path), full_run_cpu_s=float(pr), limit_s=PROJ_MAX_S, pilot_sources=J.get('sources'))


def read_takes(PT, hashes):
    for k in ('takes', 'summary', 'take_receipt'):
        need(PT[k].is_file(), f'input missing: {PT[k]} (refused)')
        hashes[_rel(PT[k])] = X._sha(PT[k])
    hdr = pd.read_csv(PT['takes'], nrows=0).columns
    for c in TAKE_READ:
        need(c in hdr, f'takes.csv: required column {c} missing (refused)')
    T = pd.read_csv(PT['takes'], usecols=list(TAKE_READ), float_precision='round_trip',
                    dtype=dict(day=str, addr=str, sid=str, status=str, take_class=str, tk_hash=str, cls_rec=str, cls_sim=str, cls_cert_0=str,
                               mid=np.int64, rank=np.int64, s=np.int64, x_lo_hard_ms=np.int64, hi_ms=np.int64, settle_ms=np.int64))
    S = pd.read_csv(PT['summary'], dtype={'scope': str, 'day': str})
    tkr = json.loads(PT['take_receipt'].read_text())
    need(int((tkr.get('outputs') or {}).get('takes', -1)) == len(T), "takes.csv rows differ from chain_take's receipt outputs (refused)")
    repro = TL.check_reproduction(T, S)
    repro['variants'] = check_variant_reproduction(T)
    return T, repro, tkr


def execute(kind, days, out, inp=None, pilot_receipt=None, require_freeze=True, log=print):
    """kind 'run' (every measurement day -> v2/chains/take_logic_v2; freeze + recorded pilot projection required) or 'pilot' (its day, out given)."""
    c_imp = time.process_time()
    t0, asof0 = time.time(), _now()
    need(mutant() == '', f"MT_TL2_MUT={os.environ.get('MT_TL2_MUT')!r} is set: production never runs a planted error (refused before any input is read)")
    need(os.environ.get('MT_TL_MUT', '') == '', f"MT_TL_MUT={os.environ.get('MT_TL_MUT')!r} is set: v1's planted errors would enter the shared state "
                                                 "builders (refused before any input is read)")
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
    git = X._git([f for f in TOOL_FILES if Path(f).exists()])
    pre = {}
    if kind == 'run':
        need(days == tuple(C.MEASURE_DAYS), 'the full run covers every measurement day (refused)')
        if require_freeze:
            need(all(Path(f).exists() for f in TOOL_FILES) and bool(git.get('unmodified')),
                 'freeze: take_logic_v2.py, its selftest, the contract, take_logic.py, chain_context.py and chain_take.py must be tracked and '
                 'unmodified (refused)')
        pre = preflight_projection(Path(pilot_receipt) if pilot_receipt else PT['pilot_receipt'], hashes)
    tick('preflight')
    inp = Path(inp) if inp else C.OUT
    T_all, repro, tkr = read_takes(PT, hashes)
    CEN = read_census(PT, hashes)
    tick('inputs')
    addrs, book = CT.population(inp / 'econ' / 'identities.json', CT.E.BOOK_MANIFEST, hashes)
    need(tuple(a.lower() for a in CX.top3(inp)) == tuple(a.lower() for a in TOP3), 'chain_context.top3 differs from chain_take.TOP3 (refused)')
    bm = CX.book_markets()
    need({d: sorted(int(m) for m in v) for d, v in bm.items()} == {d: sorted(int(m) for m in v) for d, v in book.items()},
         'chain_context.book_markets differs from chain_take.population (refused)')
    need(all(d in book for d in days), f'days {days} are not all BOOK manifest days (refused)')
    T = TL.prepare_takes(T_all[T_all['day'].astype(str).isin(days).to_numpy()])
    take_days = Counter(str(x) for x in T_all['day'])
    o4_all, _ = o4_status(T_all)
    recount = review_recount(T_all, o4_all)
    fv_exp = json.loads(CT.FV_RECEIPT.read_text())['inputs']['econ/fills_valued.parquet']['sha256']
    hashes[_rel(CT.FV_RECEIPT)] = X._sha(CT.FV_RECEIPT)
    tick('population')
    hc, parts = {}, []
    for d in days:
        for mid in book[d]:
            M = CX.load_market(d, int(mid), TOP3, root=inp, hash_cache=hc)
            for k_, v_ in M.inputs.items():
                if isinstance(v_, dict) and v_.get('sha256'):
                    hashes.setdefault(k_, v_['sha256'])
            tick('context_load')
            parts.append(TL.market_part(d, int(mid), M))
            del M
            cnt['markets'] += 1
            tick('states')
            CT.guard(f'market {d} {mid}')
    fvk = str((inp / 'econ' / 'fills_valued.parquet').resolve())
    need(hc.get(fvk) == fv_exp, f'fills_valued sha256 {str(hc.get(fvk))[:12]} differs from the accepted binding {fv_exp[:12]} (refused)')
    need(tkr['inputs'].get('econ/fills_valued.parquet') == hc.get(fvk),
         "fills_valued sha256 differs from chain_take's receipt: the saved takes and this context are different data (refused)")
    both = [k for k in hashes if isinstance(tkr['inputs'].get(k), str)]
    mism = [k for k in both if tkr['inputs'][k] != hashes[k]]
    need(not mism, f"inputs differ from chain_take's receipt {mism[:5]} (refused: mismatched artifacts)")
    cnt['inputs_matching_take_receipt'] = len(both)
    n_mk = int(cnt['markets'])
    n_pilot_takes = sum(take_days.get(d, 0) for d in days)

    def extra_fn(stg):
        ex = dict(preflight=pre, take_rows_by_day=dict(take_days), class_reproduction=repro)
        if kind == 'pilot':
            r_mk = N_BOOK / max(1, n_mk)
            r_ev = max(r_mk, sum(take_days.get(d, 0) for d in C.MEASURE_DAYS) / max(1, n_pilot_takes))
            fixed = sum(stg.get(k, 0.) for k in ('imports', 'preflight', 'inputs', 'population'))
            scal_mk = stg.get('context_load', 0.) + stg.get('states', 0.)
            scal_ev = sum(stg.get(k, 0.) for k in ('assemble', 'events', 'nulls', 'tables', 'write'))
            proj = fixed + scal_mk * r_mk + scal_ev * r_ev
            ex['projection'] = dict(full_run_cpu_s=round(proj, 2), pilot_cpu_s=round(time.process_time(), 3), limit_s=PROJ_MAX_S, target_s=PROJ_TARGET_S,
                                    launch_allowed=bool(proj <= PROJ_MAX_S), within_target=bool(proj <= PROJ_TARGET_S),
                                    rule=f'fixed (imports + preflight + inputs + population) {fixed:.2f} + (context_load + states) {scal_mk:.2f} x '
                                         f'{N_BOOK}/{n_mk} + (assemble + events + nulls + tables + write) {scal_ev:.2f} x max(market ratio, take ratio) '
                                         f'{r_ev:.2f}; the first load carries the one-off fills_valued sha256 (scaled: conservative)')
        return ex
    rec, claims, R = run_core(kind, days, parts, T, out, hashes, cnt, stages, tick, git, asof0, t0, 3 * n_mk, CEN, repro=repro, recount=recount,
                              n_take_rows=len(T_all), extra_fn=extra_fn)
    log(json.dumps(X._clean(digest(out, claims, R, rec)), indent=1, sort_keys=True, default=str))
    return rec


def digest(out, claims, R, rec):
    RK = R['RK']
    key = {}
    for sc in SCOPES2:
        for o in OUTCOMES:
            r = RK[('full', 'D0', sc, o, 'RR_size')]
            key[f'{sc}:{o}'] = dict(RR_MH=r['RR_MH'], RR_crude=r['RR_crude'], p_hi=r['p_hi'], p_lo=r['p_lo'], n_events=r['n_events_informative'],
                                    a=r['a_exposed'], T1=r['T_exposed'], b=r['b_reference'], T0=r['T_reference'],
                                    markets_rr_gt1=f"{r['n_markets_rr_gt1']}/{r['n_markets_supported']}", null_p50=r['null_p50'], verdict=r['verdict_rule'])
    views = {f'{sc}:{o}': {f'{pn}:{v}': [RK[(pn, v, sc, o, 'RR_size')]['RR_MH'], RK[(pn, v, sc, o, 'RR_size')]['p_hi'],
                                         RK[(pn, v, sc, o, 'RR_size')]['verdict_rule']] for pn, v in ALL_VIEWS}
             for sc in SCOPES2 for o in DIRECTIONAL}
    cov = R['coverage']
    sel = cov[(cov['panel'] == 'full') & (cov['variant'] == 'D0') & cov['scope'].isin(['pooled', 'rank1'])]
    covd = [{k: rr[k] for k in COV_DIGEST} for rr in sel.to_dict('records')]
    return dict(out=str(out), kind=claims['kind'], days=claims['days'], maker_seconds=claims['maker_seconds'], takes=claims['takes'], events=claims['events'],
                decision={sc: dict(reading=d['reading'], views=d['views'], primary=d['primary'], predicates=d['predicates'])
                          for sc, d in claims['decision'].items()},
                RR_size_full_D0=key, RR_size_views=views, coverage_full_D0=covd, collisions=claims['collisions'], statuses=claims['statuses'],
                census=claims['census_fix'], stated_as_data=claims['stated_as_data'], review_recount=claims['review_recount'],
                projection=rec.get('projection'), cpu_s=rec['cpu_s'], vmhwm_mib=rec['vmhwm_mib'])


def main(argv):
    a = list(argv[1:])
    usage = 'usage: MT_RUN=v2 take_logic_v2.py run | MT_RUN=v2 take_logic_v2.py run --pilot <day> [--out <dir>]'
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
