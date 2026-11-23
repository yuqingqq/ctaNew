"""LOGIC-CHAIN CORRECTION v1 - POST REGRADE, CENSUS EVIDENCE GRADES, EPISODE PATHS (P-2026-005 MAKER-TRACE logic-chain packet; session
aea5bfb9 builder A, 2026-11-23). Frozen rules: predictfun/reviews/maker_trace_20261122/chain_correction_contract.md sections 1 (POST regrade,
review finding 1), 4 (census grades + episode paths, finding 5) and the section-5 falsifiers 'grade' and 'observed vs inferred' (proved in
chain_correct_post_selftest.py with env-gated mutants). Review: predictfun/reviews/automatic/20261123_logic_chains_review/report.md. The
original receipt v2/report_logic_chains.md and every saved output stay unedited.
SAVED EVIDENCE ONLY: reads the inputs below and nothing else (no lifecycle / context / chain tool re-run, no new data, no fit). DESCRIPTIVE
ONLY on four consumed days (Oct 24-27, point estimates, no intervals): no model skill, validation, intent or profitability claim.
INPUTS (C.RECEIPT = v2/): chains/post_fill/orders.csv (the 4,149 filled top-3 orders = the unit); order_estimator/{census_orders,
       census_links, episodes, episode_summary, cases, order_ledger}.csv; chains/take/actions.csv; chains/pull/pulls.csv; and, per (day,
       market) present in orders.csv only (measurement days, refused otherwise before a path is built), the lifecycle
       C.OUT/life/<day>_<mid>_orders.parquet columns scenario / mk_hash / mk_signer / post_class / post_lo_ms / post_hi_ms.
FACTS  verified on the saved rows (hard checks; a failure refuses the run): post_fill post_lo_union = the earliest of the three scenario
       post_lo (-1 when any is unbounded), its width = post_hi - post_lo_union and its post_class the width class of that UNION
       (order_estimator.order_core / chain_post_fill.post_class); the estimator's census width equals it; a lifecycle 'constrained' post
       class has three equal scenario bounds (pf_mt_lifecycle._classes: every scenario equals the scenario-free H1-H3 bound). The lifecycle
       post_class is one value per order: checked across the three scenario rows; a disagreement is counted and graded as not constrained.
GRADE  (contract 1, first match; G1_FLAGS / INVALID below) G0 unbounded_or_invalid: union window unbounded or the order invalid (packet
       reason invalid_order, or a lifecycle INVALID flag) / G1 conflict: a capacity / clock / post-after-first-fill flag in the packet status
       or the estimator's order_flags, or the estimator's clock_bad / a first-fill clock reason / G2 bound_constrained: lifecycle class
       observed or constrained and packet status ok (the window holds under the scenario-free bound; still conditional on H1 / H2 /
       capacity and the 2 s maximum match lag) / G3 constrained_status_flagged: observed or constrained, status not ok / G4 scenario_only:
       every other row (assumption_sensitive, unresolved, scenario rows disagree): the width is a scenario estimate only. Width bins of the
       union (scenario) width: le_1s (<= LATE_MS) / le_10s (<= EARLY_MS) / gt_10s / unbounded. Allowed claim: "n orders have a
       bound-constrained posting window <= 10 s" (G2). Never a share of ALL orders that rested <= 10 s; never "not long-resting queue
       builders"; a G4 row says nothing about its true rest.
LOGIC  the saved POST logic rows (order_estimator episodes.csv): evidence grade B_constrained only for G2, C_scenario for G4, E_unresolved
       for G0 / G1 (contract) and for G3 (contract silent: the conservative reading); queue_priority_before_demand may be 'contradicted'
       only for G2 with width <= EARLY_MS (G2 wider: timing_sensitive), G4 'timing_sensitive', G0 / G1 / G3 'unidentified' (contract silent:
       conservative). The other POST mechanisms keep their saved statuses beside the corrected grade. Originals kept beside.
CENSUS every action count with its evidence grade (contract 4): observed_execution (maker fills, take records; a take class is computed
       on the retained status-ok ledger) / continued_execution (keep_after_fill: the same hash filled again, not a private KEEP) /
       scenario_dependent_link (reprice / refill, split tied / sequential_under_scenarios / overlap_certain / clock_refused; a later
       first-filled same-side hash, never an identified repost: link_hypotheses keeps 'unrelated' feasible, checked on the saved feasible
       sets) / cancel_compatible_ending (pull_identified False, counted from pulls.csv) / feasible_timing_sensitive_overlap (two_sided,
       ladder_overlap; the overlap_certain subset shown apart) and, for types the contract does not list: scenario_dependent_ending
       (exhausted_proven / capacity_tight / capacity_tight_view) / unresolved_alternatives (end_censored_possible / unidentified, no filled
       successor) / post_window:<G> (post_new_level / post_same_level split by the section-1 grade, chain_take's level rule re-derived and
       checked equal). Also the post_fill successor of every order (next_kind x link status). The writer refuses an ungraded row and a
       scenario-dependent link counted as an identified repost.
PATHS  per rank the first two episodes in cases.csv row order (first appearance; predetermined, never outcome-selected): post (side, tick,
       size alternatives, window grade) -> fills -> terminal (EX / PL / EC per scenario, cancel-compatible views) -> successor (link grade)
       -> takes (class) -> the saved episode economics, each with the alternative that would change its logic.
OUTPUTS C.RECEIPT/chains/correction_v1/ (the directory is shared with chain_correct_pull_take.py): post_regrade.csv, post_regrade_orders.csv,
       episodes_post_regraded.csv, census_graded.csv, paths.md, claims_post.json (computed predicates only, the reviewer's recount
       reproduced), receipt_post.json (sha256 of the sources, every input read and every output written; CPU, VmHWM). Write-once FILE BY
       FILE: refused before any input is read if any of these names exists; every byte is built before the first write; exclusive create.
GUARDS MT_RUN=v2; no MT_CC_MUT mutant in a run; measurement days only; VmHWM self-guard pf_mt_common.GUARD_MIB (never raised); CPU self-guard
       CPU_CAP_S; run (not --dry) refuses unless this tool, its selftest and the contract are tracked and unmodified (the freeze).
Usage (run_light, MT_RUN=v2, PF_SLOT=claude2): chain_correct_post.py run [--dry]     (--dry: reads and checks everything, writes nothing,
       prints the schema checks and the reproduction predicates)      selftest: chain_correct_post_selftest.py"""
import hashlib
import json
import os
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

sys.dont_write_bytecode = True                                   # no __pycache__ side effect in the receipt tree

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

HERE = Path(__file__).resolve().parent
RESEARCH = HERE.parents[2] / 'research'
for _p in (str(RESEARCH), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import pf_mt_common as C  # noqa: E402
import order_estimator as OE  # noqa: E402   constants only: LATE_MS / EARLY_MS / CLOCK_FLAGS and the lifecycle INVALID set (OE.LC)

need, OK = C.need, C.STATUS_OK
VERSION = 'chain_correct_post_v1'
NAN = float('nan')
CONTRACT_FILE = C.RECEIPT_V1 / 'chain_correction_contract.md'
SELFTEST_FILE = HERE / 'chain_correct_post_selftest.py'
TOOL_FILES = (Path(__file__).resolve(), SELFTEST_FILE, CONTRACT_FILE)
DEP_FILES = (HERE / 'order_estimator.py', RESEARCH / 'pf_mt_common.py', RESEARCH / 'pf_mt_lifecycle.py')

# ------------------------------------------------------------------ declared rules (contract 1 / 4; none fitted) ------------------------------------------------------------------
SCEN = tuple(C.SCENARIOS)                                        # S_FIFO, S_FIFO_LO, S_NOPRIO
LATE_MS, EARLY_MS = OE.LATE_MS, OE.EARLY_MS                      # 1,000 / 10,000 ms (the estimator's constants)
INVALID = frozenset(OE.LC.INVALID)
G1_FLAGS = frozenset(('capacity_contradiction', 'capacity_bounds_void', 'clock_capacity_conflict', 'post_after_first_fill')) | frozenset(OE.CLOCK_FLAGS)
CERT = (C.OBSERVED, C.CONSTRAINED)
G0, G1, G2, G3, G4 = ('G0_unbounded_or_invalid', 'G1_conflict', 'G2_bound_constrained', 'G3_constrained_status_flagged', 'G4_scenario_only')
GRADES = (G0, G1, G2, G3, G4)
GRADE_RULE = {G0: 'union window unbounded or order invalid', G1: 'capacity / clock / post-after-first-fill conflict flag',
              G2: 'lifecycle class observed or constrained, and status ok', G3: 'lifecycle class observed or constrained, status not ok',
              G4: 'every other row (assumption_sensitive / unresolved / scenario rows disagree)'}
WINDOW_BASIS = {G0: 'none: window unbounded or order invalid', G1: 'void: conflict flags',
                G2: 'lifecycle scenario-free bound (conditional on H1 / H2 / capacity and the maximum match lag)',
                G3: 'lifecycle scenario-free bound with a flagged status (not certified)', G4: 'scenario union estimate only'}
LOGIC_GRADE = {G0: 'E_unresolved', G1: 'E_unresolved', G2: 'B_constrained', G3: 'E_unresolved', G4: 'C_scenario'}
QUEUE_MECH = 'queue_priority_before_demand'
BINS = ('le_1s', 'le_10s', 'gt_10s', 'unbounded')
SHORT = ('late_post', 'short_rest')
LIFE_DISAGREE = 'scenario_rows_disagree'
SCOPES = ('pooled', 'rank1', 'rank2', 'rank3')
# the reviewer's recount (report finding 1 / 5): reproduced as recorded predicates, never a refusal
REVIEWER = dict(n_orders=4149, late_post=576, short_rest=2341, scenario_short=2917, short_life_constrained=957,
                short_life_constrained_status_ok=940, short_assumption_sensitive=1960, short_conflict_flagged=413, successor_reprice=2962,
                successor_refill=577, successor_tied=2502, successor_reprice_tied=2223, successor_refill_tied=279, successor_sequential=1037)
REVIEWER_CONFLICT_FLAGS = frozenset(('capacity_bounds_void', 'clock_capacity_conflict', 'post_after_first_fill'))   # the reviewer's regex
# census grades (contract 4) and the families for the types it does not list
CENSUS_GRADES = ('observed_execution', 'continued_execution', 'scenario_dependent_link', 'cancel_compatible_ending',
                 'feasible_timing_sensitive_overlap', 'scenario_dependent_ending', 'unresolved_alternatives') + tuple(f'post_window:{g}' for g in GRADES)
TAKE_CLASS_TYPES = tuple(f'take_{c}' for c in ('from_flat', 'add', 'reduce_partial', 'full_close', 'reversal_reducing', 'overshoot', 'no_change',
                                                'unclassified', 'timing_sensitive'))
TAKE_ATTR_TYPES = ('take_self_trade', 'take_sweep', 'take_multi_record_order', 'take_kind_mint', 'take_kind_merge')
PULL_GRADE = dict(cancel_possible='cancel_compatible_ending', exhausted_proven='scenario_dependent_ending', capacity_tight='scenario_dependent_ending',
                  capacity_tight_view='scenario_dependent_ending', end_censored_possible='unresolved_alternatives', unidentified='unresolved_alternatives')
LINK_SUBSET = dict(tied=('tied', False), sequential_identified=('sequential_under_scenarios', False), overlap_identified=('overlap_certain', False),
                   clock_refused=('clock_refused', False))
BASIS = dict(
    maker_fill='public match record of a resting own order (outcome-selected: unfilled orders are invisible)',
    take_class='public take record (observed); its inventory class is computed on the retained status-ok ledger (recorded / certain / '
               'simultaneous variants agree)',
    take_timing_sensitive='public take record (observed); the recorded / certain / simultaneous inventory variants disagree',
    take_attr='attribute of a public take record',
    continued_execution='the same hash filled again later (observed persistence), not a private KEEP; a cancel in flight is compatible',
    link='a later first-filled same-side hash (link_hypotheses under the declared scenarios), not a newly submitted replacement',
    cancel_compatible_ending='EX / PL tie: exhaustion retained (pull_identified False)',
    scenario_dependent_ending='end class under a scenario view or the three declared scenarios (not scenario-free)',
    unresolved_alternatives='several end histories feasible (end-censored / unidentified)',
    overlap='feasible overlapping lives (timing-sensitive); overlap_certain = co-presence proven by fills (H1, maximum match lag)',
    post_window='count of first-filled orders by level novelty (unfilled posts invisible); posting window graded by contract 1: ',
    successor_none='no later first-filled same-side order: an unfilled successor or none (invisible in fills)')
CENSUS_COLS = ('source', 'action_type', 'scope', 'day', 'subset', 'n', 'n_action_total', 'evidence_grade', 'identified_repost', 'n_identified',
               'identified_basis', 'population', 'grade_basis')
OUT_FILES = ('post_regrade.csv', 'post_regrade_orders.csv', 'episodes_post_regraded.csv', 'census_graded.csv', 'paths.md', 'claims_post.json',
             'receipt_post.json')
INPUTS = (('post_orders', 'chains/post_fill/orders.csv'), ('census_orders', 'order_estimator/census_orders.csv'),
          ('census_links', 'order_estimator/census_links.csv'), ('episodes', 'order_estimator/episodes.csv'),
          ('episode_summary', 'order_estimator/episode_summary.csv'), ('cases', 'order_estimator/cases.csv'),
          ('order_ledger', 'order_estimator/order_ledger.csv'), ('actions', 'chains/take/actions.csv'), ('pulls', 'chains/pull/pulls.csv'))
NEED = dict(
    post_orders=('day', 'mid', 'rank', 'addr', 'mk_hash', 'y_side', 'tick', 'n_fills', 'cum_shares', 'first_hi_ms', 'last_hi_ms', 'ws_ms', 'we_ms',
                 'post_lo_union_ms', 'post_lo_fifo_ms', 'post_hi_ms', 'window_width_ms', 'rest_bin', 'post_class', 'post_class_reason',
                 'queue_class', 'n_later_fills', 'size_status', 'union_histories', 'next_kind', 'next_link_status', 'n_children', 'next_hash',
                 'next_gap_lo_ms', 'next_gap_hi_ms', 'status'),
    census_orders=('day', 'mid', 'rank', 'hash', 'conflict', 'clock_bad', 'post_w', 'post_class', 'order_flags'),
    census_links=('day', 'mid', 'rank', 'kind', 'status'),
    episodes=('episode_id', 'case_key', 'link', 'link_seq', 'mechanism', 'decision_window_lo_ms', 'decision_window_hi_ms', 'identification_status',
              'conditional_status', 'evidence_grade', 'timing_sensitive', 'status'),
    episode_summary=('episode_id', 'day', 'mid', 'addr', 'rank', 'n_cases', 'case_keys', 'ep_n_legs', 'ep_n_maker', 'ep_n_taker', 'ep_sh', 'ep_fee',
                     'ep_net_end', 'ep_net_res', 'ep_mq', 'ep_rq', 'ep_pos_in', 'ep_pos_out', 'ep_outcome', 'mg_status', 'mg_close_net', 'mg_resid',
                     'mg_total', 'mg_close_net_c_sh'),
    cases=('kind', 'day', 'mid', 'addr', 'key', 'rank', 'case_key', 'episode_id', 'y_side', 'tick', 'n_fills', 'cum', 'size_status', 'union', 'ub',
           'post_lo_union', 'post_hi', 'rest_ub_ms', 'P_cmd', 'P_drop', 'P_lot', 'P_clip', 'prior_sensitive', 'status', 's', 'q', 'p_avg',
           'pos_before', 'pos_after', 'cls', 'cls_simultaneous', 'cls_certain', 'timing_sensitive', 'burst_cert', 'burst_pos'),
    order_ledger=('case_key', 'row_kind', 'hypothesis', 'role', 'other_hash', 'kind', 'status', 'gap_lo_ms', 'gap_hi_ms', 'feasible_union',
                  'feasible_S_FIFO', 'feasible_S_FIFO_LO', 'feasible_S_NOPRIO', 'size_lo', 'size_hi'),
    actions=('action_type', 'scope', 'day', 'n', 'timing_sensitive_n', 'n_compatible', 'n_contradicted', 'n_timing_sensitive', 'n_unidentified',
             'population', 'extra'),
    pulls=('day', 'mid', 'scope', 'addr', 'mk_hash', 'view', 'removal_class', 'end_histories', 'identification', 'pull_identified'))
USECOLS = dict(pulls=NEED['pulls'])
LIFE_COLS = ('scenario', 'mk_hash', 'mk_signer', 'post_class', 'post_lo_ms', 'post_hi_ms')
CPU_CAP_S = 30.
MUTANTS = ('grade_by_width', 'g2_before_g1', 'tied_as_identified', 'ungraded_ok')     # planted errors: selftest only, refused by run


# ------------------------------------------------------------------ small helpers ------------------------------------------------------------------
def _mut():
    return os.environ.get('MT_CC_MUT', '')


def _now():
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())


def hwm():
    with open('/proc/self/status') as f:
        for line in f:
            if line.startswith('VmHWM:'):
                return int(line.split()[1]) / 1024.
    return 0.


def guard(where, lim=C.GUARD_MIB):
    h = hwm()
    need(h < lim, f'memory self-guard: VmHWM {h:.0f} MiB >= {lim:.0f} MiB at {where}')
    return h


def cpu_guard(where):
    c = time.process_time()
    need(c < CPU_CAP_S, f'CPU self-guard: {c:.1f} s >= {CPU_CAP_S} s at {where}')
    return c


def _sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 22), b''):
            h.update(b)
    return h.hexdigest()


def _rel(p):
    p = Path(p).resolve()
    try:
        return str(p.relative_to(C.REPO))
    except ValueError:
        return str(p)


def _clean(o):
    """JSON-safe copy: NaN / inf -> None, numpy scalars -> Python, keys -> str, sets -> sorted lists."""
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (set, frozenset)):
        return sorted(_clean(v) for v in o)
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (bool, np.bool_)):
        return bool(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, (float, np.floating)):
        return float(o) if np.isfinite(o) else None
    return o


def _json_bytes(obj):
    return (json.dumps(_clean(obj), indent=1, sort_keys=True, allow_nan=False) + '\n').encode()


def _git(files):
    """HEAD and whether the files are tracked and unmodified (recorded; the run's freeze test reads 'unmodified')."""
    try:
        g = ['git', '-C', str(C.REPO)]
        head = subprocess.run(g + ['rev-parse', 'HEAD'], capture_output=True, text=True, timeout=30).stdout.strip() or None
        rel = [_rel(f) for f in files]
        tracked = subprocess.run(g + ['ls-files', '--error-unmatch', '--'] + rel, capture_output=True, text=True, timeout=30).returncode == 0
        st = subprocess.run(g + ['status', '--porcelain', '--'] + rel, capture_output=True, text=True, timeout=30)
        return dict(head=head, files=rel, tracked=tracked, unmodified=bool(tracked and st.returncode == 0 and not st.stdout.strip()))
    except (OSError, ValueError, subprocess.SubprocessError):
        return dict(head=None, files=None, tracked=None, unmodified=None)


def _strs(col):
    """a column -> object array of Python str (missing -> '')."""
    return np.array([v if isinstance(v, str) else ('' if (v is None or pd.isna(v)) else str(v)) for v in col], object)


def _flags(x):
    """a '+'-joined status -> its flag set ('ok' and empty -> none)."""
    if not isinstance(x, str):
        if x is None or pd.isna(x):
            return frozenset()
        x = str(x)
    return frozenset(f for f in x.split('+') if f and f != OK)


def _b(x):
    """a saved boolean cell -> bool / None (missing); anything else refuses."""
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if isinstance(x, str):
        need(x in ('True', 'False'), f'not a boolean cell: {x!r}')
        return x == 'True'
    if x is None or pd.isna(x):
        return None
    if isinstance(x, (int, float, np.integer, np.floating)) and float(x) in (0., 1.):
        return bool(x)
    raise ValueError(f'not a boolean cell: {x!r}')


def _fin(x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return NAN
    return v if np.isfinite(v) else NAN


def _fmt(x, nd=2):
    v = _fin(x)
    if np.isnan(v):
        return 'na'
    return f'{v:.{nd}f}' if nd else f'{int(round(v))}'


def _pred(mask):
    m = np.asarray(mask, bool)
    return dict(n=int(len(m)), n_true=int(m.sum()), holds=bool(m.all()))


def _scope_mask(rank, day, scope, d):
    """rank / day arrays -> rows of (scope, day); scope pooled / rank1..3, day a day or 'all'."""
    m = np.ones(len(rank), bool)
    if scope != 'pooled':
        need(scope in SCOPES, f'unknown scope {scope!r} (refused)')
        m &= np.asarray(rank) == int(scope[4:])
    if d != 'all':
        m &= np.asarray(day, object) == d
    return m


# ------------------------------------------------------------------ contract 1: the grade ------------------------------------------------------------------
def width_bin(width_ms, post_lo_union_ms):
    if post_lo_union_ms < 0 or width_ms < 0:
        return 'unbounded'
    return 'le_1s' if width_ms <= LATE_MS else ('le_10s' if width_ms <= EARLY_MS else 'gt_10s')


def width_class(width_ms, post_lo_union_ms):
    """chain_post_fill.post_class's width rule (the union window)."""
    if post_lo_union_ms < 0 or width_ms < 0:
        return 'unidentified'
    return 'late_post' if width_ms <= LATE_MS else ('short_rest' if width_ms <= EARLY_MS else 'early_or_late_unidentified')


def grade(o):
    """contract 1, first match. o: width_ms, post_lo_union_ms, reason (packet post_class_reason), flags (packet status | estimator
    order_flags), clock_bad, life_class (the lifecycle post_class common to the three scenario rows, else LIFE_DISAGREE), status (packet).
    -> (grade, rule)."""
    m = _mut()
    w, lo = int(o['width_ms']), int(o['post_lo_union_ms'])
    if m == 'grade_by_width' and lo >= 0 and 0 <= w <= EARLY_MS and o['status'] == OK:
        return G2, 'MUTANT grade_by_width'                       # planted: the receipt's error (a finite short union graded as proven)
    fl = frozenset(o['flags'])
    rules = [(G0, lo < 0 or w < 0 or o['reason'] in ('invalid_order', 'window_unbounded') or bool(fl & INVALID)),
             (G1, bool(fl & G1_FLAGS) or bool(o['clock_bad']) or o['reason'] == 'first_fill_clock'),
             (G2, o['life_class'] in CERT and o['status'] == OK and not fl),     # 'status ok': no flag in the packet or the estimator
             (G3, o['life_class'] in CERT),
             (G4, True)]
    if m == 'g2_before_g1':                                       # planted: precedence broken
        rules[1], rules[2] = rules[2], rules[1]
    for g, ok in rules:
        if ok:
            return g, GRADE_RULE[g]
    raise AssertionError('unreachable')


def logic_correction(g, width_ms, mechanism, ident, tsens):
    """contract 1 'Logic-table POST rows' -> (evidence_grade, identification_status, timing_sensitive)."""
    eg = LOGIC_GRADE[g]
    if mechanism != QUEUE_MECH:
        return eg, ident, bool(tsens)
    if g == G2:
        st = 'contradicted' if 0 <= int(width_ms) <= EARLY_MS else 'timing_sensitive'
    elif g == G4:
        st = 'timing_sensitive'
    else:
        st = 'unidentified'                                       # G0 / G1 / G3
    return eg, st, st == 'timing_sensitive'


# ------------------------------------------------------------------ inputs ------------------------------------------------------------------
def read_inputs(rroot, life_dir):
    """the saved CSVs (schema-checked) and the lifecycle rows of the top-3 hashes of each (day, market) in orders.csv."""
    D, info = {}, {}
    for k, rel in INPUTS:
        p = Path(rroot) / rel
        need(p.is_file(), f'input {k} is missing: {p} (refused)')
        cols = list(pd.read_csv(p, nrows=0).columns)
        miss = [c for c in NEED[k] if c not in cols]
        need(not miss, f'input {k}: required columns missing {miss} (refused)')
        df = pd.read_csv(p, usecols=list(USECOLS[k]) if k in USECOLS else None, dtype={'day': str} if 'day' in cols else None,
                         float_precision='round_trip', low_memory=False)
        D[k] = df
        info[k] = dict(path=_rel(p), sha256=_sha(p), bytes=p.stat().st_size, rows=int(len(df)), columns=len(cols), required_columns=len(NEED[k]),
                       missing_columns=0)
    P = D['post_orders']
    need(len(P) > 0 and not P.duplicated(['day', 'mid', 'mk_hash']).any(), 'post_fill orders: empty or a (day, market, hash) repeats (refused)')
    days = sorted(set(P['day'].astype(str)))
    need(all(d in C.MEASURE_DAYS for d in days), f'post_fill days {days} are not all measurement days (refused before any lifecycle path is built)')
    parts = []
    for d, mid in sorted(set(zip(P['day'].astype(str), P['mid'].astype(np.int64)))):
        p = Path(life_dir) / f'{d}_{int(mid)}_orders.parquet'
        need(p.is_file(), f'lifecycle orders missing: {p} (refused)')
        hs = set(P.loc[(P['day'] == d) & (P['mid'] == mid), 'mk_hash'].astype(str))
        t = pq.read_table(p, columns=list(LIFE_COLS)).to_pandas()
        n_read = len(t)
        for c in ('scenario', 'mk_hash', 'mk_signer', 'post_class'):
            t[c] = t[c].astype(object)
        t = t[t['mk_hash'].isin(hs)].copy()
        for c in ('scenario', 'mk_hash', 'mk_signer', 'post_class'):
            need(all(isinstance(v, str) for v in t[c]), f'lifecycle {p.name}: {c} is not a string column (refused)')
        t['day'], t['mid'] = d, np.int64(mid)
        parts.append(t)
        info[f'life/{p.name}'] = dict(path=_rel(p), sha256=_sha(p), bytes=p.stat().st_size, rows=int(n_read), rows_used=int(len(t)))
    L = pd.concat(parts, ignore_index=True)
    return D, L, info


def life_pivot(L):
    """(day, mid, mk_hash) x scenario -> cls_<s>, plo_<s>, phi_<s>, sig_<s>; every order needs its three scenario rows."""
    key = ['day', 'mid', 'mk_hash']
    need(not L.duplicated(key + ['scenario']).any(), 'lifecycle: a (hash, scenario) row repeats (refused)')
    bad = set(L['scenario']) - set(SCEN)
    need(not bad, f'lifecycle: unknown scenarios {sorted(bad)} (refused)')
    base = L[key].drop_duplicates().reset_index(drop=True)
    for s in SCEN:
        x = L[L['scenario'] == s][key + ['post_class', 'post_lo_ms', 'post_hi_ms', 'mk_signer']]
        x = x.rename(columns={'post_class': f'cls_{s}', 'post_lo_ms': f'plo_{s}', 'post_hi_ms': f'phi_{s}', 'mk_signer': f'sig_{s}'})
        base = base.merge(x, on=key, how='left', validate='one_to_one')
    need(base[[f'cls_{s}' for s in SCEN]].notna().all().all(), 'lifecycle: an order lacks one of the three scenario rows (refused)')
    return base


def _keys(df, h='mk_hash'):
    df = df.copy()
    df['day'] = _strs(df['day'])
    df['mid'] = df['mid'].astype(np.int64)
    df[h] = _strs(df[h])
    return df


def assemble(P, CO, LP):
    P, CO, LP = _keys(P), _keys(CO, 'hash'), _keys(LP)
    co = CO.rename(columns={'hash': 'mk_hash', 'rank': 'co_rank', 'post_w': 'co_post_w', 'post_class': 'co_post_class', 'conflict': 'co_conflict',
                            'clock_bad': 'co_clock_bad', 'order_flags': 'co_order_flags'})
    co = co[['day', 'mid', 'mk_hash', 'co_rank', 'co_post_w', 'co_post_class', 'co_conflict', 'co_clock_bad', 'co_order_flags']]
    M = P.merge(co, on=['day', 'mid', 'mk_hash'], how='left', validate='one_to_one', indicator='_co')
    need(len(CO) == len(P) and bool((M['_co'] == 'both').all()), 'census_orders does not match the post_fill orders one to one (refused)')
    M = M.merge(LP, on=['day', 'mid', 'mk_hash'], how='left', validate='one_to_one', indicator='_lf')
    need(bool((M['_lf'] == 'both').all()), 'a post_fill order has no lifecycle rows (refused)')
    return M.drop(columns=['_co', '_lf'])


# ------------------------------------------------------------------ contract 1: per order ------------------------------------------------------------------
def order_table(M):
    """-> (T: one row per order with its grade and every input used, hard checks, soft checks)."""
    n = len(M)
    lo = M['post_lo_union_ms'].to_numpy(np.int64)
    hi = M['post_hi_ms'].to_numpy(np.int64)
    w = M['window_width_ms'].to_numpy(np.int64)
    plo = np.stack([M[f'plo_{s}'].to_numpy(np.int64) for s in SCEN], 1)
    phi = np.stack([M[f'phi_{s}'].to_numpy(np.int64) for s in SCEN], 1)
    reason, pcls, status = _strs(M['post_class_reason']), _strs(M['post_class']), _strs(M['status'])
    exp_lo = np.where((plo < 0).any(1), -1, plo.min(1))
    exp_w = np.where(lo >= 0, hi - lo, -1)
    wcls = np.array([width_class(int(a), int(b)) for a, b in zip(w, lo)], object)
    exp_cls = np.where(np.isin(reason, ['invalid_order', 'first_fill_clock']), 'unidentified', wcls)
    bins = np.array([width_bin(int(a), int(b)) for a, b in zip(w, lo)], object)
    cls3 = np.stack([_strs(M[f'cls_{s}']) for s in SCEN], 1)
    same = (cls3[:, 0] == cls3[:, 1]) & (cls3[:, 1] == cls3[:, 2])
    life = np.where(same, cls3[:, 0], LIFE_DISAGREE).astype(object)
    addr = np.array([str(a).lower() for a in M['addr']], object)
    sig_ok = np.ones(n, bool)
    for s in SCEN:
        sig_ok &= np.array([str(a).lower() for a in M[f'sig_{s}']], object) == addr
    cons = life == C.CONSTRAINED
    eq3 = (plo[:, 0] == plo[:, 1]) & (plo[:, 1] == plo[:, 2]) & (plo[:, 0] >= 0)
    est_map = dict(late_post='late_post', short_rest='intermediate', early_or_late_unidentified='early_possible', unidentified='window_unbounded')
    co_cls = _strs(M['co_post_class'])
    wr = ~np.isin(reason, ['invalid_order', 'first_fill_clock'])
    hard = dict(
        union_lower_is_min_of_three_scenarios=_pred(lo == exp_lo),
        fifo_lower_is_the_S_FIFO_row=_pred(M['post_lo_fifo_ms'].to_numpy(np.int64) == plo[:, 0]),
        post_hi_is_every_scenario_row=_pred((phi == hi[:, None]).all(1)),
        width_is_post_hi_minus_union_lower=_pred(w == exp_w),
        post_class_is_the_union_width_class=_pred(pcls == exp_cls),
        estimator_union_width_equals_packet_width=_pred(M['co_post_w'].to_numpy(np.int64) == w),
        constrained_implies_three_equal_scenario_bounds=_pred(eq3[cons]),
        lifecycle_signer_is_the_order_address=_pred(sig_ok),
        estimator_rank_equals_packet_rank=_pred(M['co_rank'].to_numpy(np.int64) == M['rank'].to_numpy(np.int64)))
    soft = dict(
        life_post_class_identical_across_scenarios=_pred(same),
        rest_bin_equals_union_width_bin=_pred(_strs(M['rest_bin']) == bins),
        estimator_class_equals_packet_class=_pred(np.array([est_map.get(a) == b for a, b in zip(pcls[wr], co_cls[wr])], bool)),
        unresolved_life_rows_are_unbounded=_pred(lo[life == C.UNRESOLVED] < 0),
        life_observed_post_class_rows=dict(n=int((life == C.OBSERVED).sum())))
    fp = [_flags(x) for x in status]
    fc = [_flags(x) for x in M['co_order_flags']]
    clock = np.array([bool(_b(x)) for x in M['co_clock_bad']], bool)
    conflict = np.array([bool(_b(x)) for x in M['co_conflict']], bool)
    g_, rule_, g1m, eg_, qs_ = [], [], [], [], []
    for i in range(n):
        fl = fp[i] | fc[i]
        g, rule = grade(dict(width_ms=int(w[i]), post_lo_union_ms=int(lo[i]), reason=str(reason[i]), flags=fl, clock_bad=bool(clock[i]),
                             life_class=str(life[i]), status=str(status[i])))
        g_.append(g)
        rule_.append(rule)
        g1m.append('+'.join(sorted(fl & G1_FLAGS)))
        eg, qs, _ = logic_correction(g, int(w[i]), QUEUE_MECH, '', False)
        eg_.append(eg)
        qs_.append(qs)
    g_ = np.array(g_, object)
    T = pd.DataFrame(dict(
        day=_strs(M['day']), mid=M['mid'].to_numpy(np.int64), rank=M['rank'].to_numpy(np.int64), addr=_strs(M['addr']),
        mk_hash=_strs(M['mk_hash']), y_side=_strs(M['y_side']), tick=M['tick'].to_numpy(np.int64),
        n_fills=M['n_fills'].to_numpy(np.int64), cum_shares=M['cum_shares'].to_numpy(float), first_hi_ms=M['first_hi_ms'].to_numpy(np.int64),
        post_lo_union_ms=lo, **{f'post_lo_{s}_ms': plo[:, j] for j, s in enumerate(SCEN)}, post_hi_ms=hi, window_width_ms=w, width_bin=bins,
        rest_bin_saved=_strs(M['rest_bin']), post_class_packet=pcls, post_class_reason=reason,
        scenario_short=np.isin(pcls, list(SHORT)), **{f'life_post_class_{s}': cls3[:, j] for j, s in enumerate(SCEN)}, life_post_class=life,
        life_class_identical=same, status_packet=status, census_order_flags=_strs(M['co_order_flags']), census_conflict=conflict,
        census_clock_bad=clock, g1_flags_matched=np.array(g1m, object), grade=g_, grade_rule=np.array(rule_, object),
        window_basis=np.array([WINDOW_BASIS[g] for g in g_], object), logic_evidence_grade=np.array(eg_, object),
        queue_status_corrected=np.array(qs_, object),
        bound_constrained_le_10s=(g_ == G2) & np.isin(bins, ['le_1s', 'le_10s'])))
    return T, hard, soft


def regrade_grid(T, days):
    """contract 1 reporting: grade x width bin per scope (pooled, rank1..3) x day (each, all), counts and filled shares with denominators."""
    rows = []
    rk, dy = T['rank'].to_numpy(np.int64), T['day'].to_numpy(object)
    gv, bv, sh = T['grade'].to_numpy(object), T['width_bin'].to_numpy(object), T['cum_shares'].to_numpy(float)
    for sc in SCOPES:
        for d in list(days) + ['all']:
            m = _scope_mask(rk, dy, sc, d)
            n0, sh0 = int(m.sum()), float(sh[m].sum())
            cg, cb = Counter(gv[m]), Counter(bv[m])
            cgb = Counter(zip(gv[m], bv[m]))
            shg = Counter()
            for g, b, x in zip(gv[m], bv[m], sh[m]):
                shg[(g, b)] += float(x)
            for g in GRADES:
                for b in BINS:
                    k = int(cgb.get((g, b), 0))
                    rows.append(dict(scope=sc, day=d, grade=g, width_bin=b, n_orders=k, n_scope_day=n0, frac_of_scope_day=k / n0 if n0 else NAN,
                                     n_grade=int(cg.get(g, 0)), frac_of_grade=k / cg[g] if cg.get(g) else NAN, n_bin=int(cb.get(b, 0)),
                                     frac_of_bin=k / cb[b] if cb.get(b) else NAN, filled_sh=round(shg.get((g, b), 0.), 6),
                                     filled_sh_scope_day=round(sh0, 6), frac_sh_of_scope_day=shg.get((g, b), 0.) / sh0 if sh0 > 0 else NAN,
                                     window_basis=WINDOW_BASIS[g], population='outcome_selected_filled_orders'))
    return pd.DataFrame(rows)


def regrade_episodes(EP, T):
    """the saved POST logic rows with the corrected evidence grade / queue status beside the originals."""
    E = EP[EP['link'].astype(str) == 'POST'].copy().reset_index(drop=True)
    parts = [str(k).split('|') for k in E['case_key']]
    need(all(len(p) == 5 and p[0] == 'order' for p in parts), 'episodes: a POST row whose case_key is not an order key (refused)')
    E['case_day'] = np.array([p[1] for p in parts], object)
    E['case_mid'] = np.array([int(p[2]) for p in parts], np.int64)
    E['case_addr'] = np.array([p[3] for p in parts], object)
    E['case_hash'] = np.array([p[4] for p in parts], object)
    E['case_day'], E['case_hash'] = _strs(E['case_day']), _strs(E['case_hash'])
    t = T[['day', 'mid', 'mk_hash', 'addr', 'grade', 'window_width_ms', 'width_bin', 'life_post_class', 'status_packet', 'post_lo_union_ms', 'post_hi_ms']]
    t = t.rename(columns={'day': 'case_day', 'mid': 'case_mid', 'mk_hash': 'case_hash', 'addr': 'order_addr', 'grade': 'post_grade',
                          'status_packet': 'packet_status', 'post_lo_union_ms': 'order_post_lo_union_ms', 'post_hi_ms': 'order_post_hi_ms'}).copy()
    t['case_day'], t['case_hash'] = _strs(t['case_day']), _strs(t['case_hash'])
    X = E.merge(t, on=['case_day', 'case_mid', 'case_hash'], how='left', validate='many_to_one', indicator='_t')
    need(bool((X['_t'] == 'both').all()), 'episodes: a POST row whose order is not in post_fill orders.csv (refused)')
    X = X.drop(columns=['_t'])
    need(all(str(a).lower() == str(b) for a, b in zip(X['order_addr'], X['case_addr'])), 'episodes: a POST case address differs from its order (refused)')
    cor = [logic_correction(g, w_, m_, i_, bool(_b(ts))) for g, w_, m_, i_, ts in
           zip(X['post_grade'], X['window_width_ms'], X['mechanism'], X['identification_status'], X['timing_sensitive'])]
    X['evidence_grade_corrected'] = [c[0] for c in cor]
    X['identification_status_corrected'] = [c[1] for c in cor]
    X['timing_sensitive_corrected'] = [c[2] for c in cor]
    X['evidence_grade_changed'] = X['evidence_grade'].astype(object).to_numpy(object) != X['evidence_grade_corrected'].to_numpy(object)
    X['identification_status_changed'] = X['identification_status'].astype(object).to_numpy(object) != X['identification_status_corrected'].to_numpy(object)
    bounded = X['order_post_lo_union_ms'].to_numpy(np.int64) >= 0
    q = X['mechanism'].astype(str).to_numpy(object) == QUEUE_MECH
    w = X['window_width_ms'].to_numpy(np.int64)
    eg0 = X['evidence_grade'].astype(object).to_numpy(object)
    id0 = X['identification_status'].astype(object).to_numpy(object)
    hard = dict(post_rows_decision_window_is_the_union_window=_pred(
        (X['decision_window_lo_ms'].to_numpy(np.int64)[bounded] == X['order_post_lo_union_ms'].to_numpy(np.int64)[bounded])
        & (X['decision_window_hi_ms'].to_numpy(np.int64)[bounded] == X['order_post_hi_ms'].to_numpy(np.int64)[bounded])))
    soft = dict(
        original_grade_is_B_for_every_bounded_union=_pred(np.where(bounded, eg0 == 'B_constrained', eg0 == 'E_unresolved')),
        original_queue_contradicted_iff_union_le_10s=_pred((id0[q] == 'contradicted') == (bounded[q] & (w[q] <= EARLY_MS))))
    X = X.drop(columns=['order_addr', 'order_post_lo_union_ms', 'order_post_hi_ms'])
    return X, hard, soft


# ------------------------------------------------------------------ contract 4: census ------------------------------------------------------------------
def action_family(t):
    """the contract-4 evidence-grade family of a chain_take action type; None = no declared grade (the census writer refuses the row)."""
    if t == 'maker_fill' or t in TAKE_CLASS_TYPES or t in TAKE_ATTR_TYPES:
        return 'observed_execution'
    if t == 'keep_after_fill':
        return 'continued_execution'
    if t in ('reprice', 'refill'):
        return 'scenario_dependent_link'
    if t in ('two_sided', 'ladder_overlap'):
        return 'feasible_timing_sensitive_overlap'
    if t in ('post_new_level', 'post_same_level'):
        return 'post_window'
    p = str(t).split(':')
    if len(p) == 3 and p[0] == 'pull_possible' and p[1] in ('S_FIFO', 'S_NOPRIO'):
        return PULL_GRADE.get(p[2])
    return None


def link_subset(status):
    """a link_hypotheses pair status -> (subset, identified_repost). No status identifies a repost: 'unrelated' stays feasible for every pair
    (checked on the saved feasible sets) and no private command is observed."""
    if _mut() == 'tied_as_identified' and status == 'tied':
        return 'sequential_under_scenarios', True                # planted: the receipt's reading of tied successors as reposts
    need(status in LINK_SUBSET, f'unknown link status {status!r} (refused)')
    return LINK_SUBSET[status]


def _basis(t, fam):
    if t == 'maker_fill':
        return BASIS['maker_fill']
    if t == 'take_timing_sensitive':
        return BASIS['take_timing_sensitive']
    if t in TAKE_CLASS_TYPES:
        return BASIS['take_class']
    if t in TAKE_ATTR_TYPES:
        return BASIS['take_attr']
    if fam in ('scenario_dependent_link',):
        return BASIS['link']
    if fam == 'feasible_timing_sensitive_overlap':
        return BASIS['overlap']
    return BASIS.get(fam, '')


def post_levels(P):
    """chain_take's post_new_level / post_same_level rule re-derived: per (day, market, address) the orders in (first_hi_ms, mk_hash) order;
    the first at its (side, tick) is post_new_level. -> {(day, mid, hash): type}."""
    out = {}
    for _, g in P.groupby(['day', 'mid', 'addr'], sort=False):
        seen = set()
        for h, fh, sd, tk in sorted(zip(g['mk_hash'], g['first_hi_ms'], g['y_side'], g['tick']), key=lambda x: (int(x[1]), str(x[0]))):
            k = (str(sd), int(tk))
            out[(str(g['day'].iloc[0]), int(g['mid'].iloc[0]), str(h))] = 'post_new_level' if k not in seen else 'post_same_level'
            seen.add(k)
    return out


def census(A, T, P, PU, CL, LED):
    """-> (census DataFrame, hard checks, soft checks). Every action count with its evidence grade; links split tied / sequential."""
    rows, hard, soft = [], {}, {}
    lv = post_levels(P)
    lvt = np.array([lv[(d, int(m), h)] for d, m, h in zip(T['day'], T['mid'], T['mk_hash'])], object)
    t_rank, t_day, t_g = T['rank'].to_numpy(np.int64), T['day'].to_numpy(object), T['grade'].to_numpy(object)
    pu_rank = np.array([int(str(s)[4:]) if str(s).startswith('rank') else -1 for s in PU['scope']], np.int64)
    need(bool((pu_rank > 0).all()), 'pulls.csv: a scope other than rank1..3 (refused)')
    pu_day, pu_view = PU['day'].astype(str).to_numpy(object), PU['view'].astype(str).to_numpy(object)
    pu_ident = np.array([bool(_b(x)) for x in PU['pull_identified']], bool)
    cl_rank, cl_day = CL['rank'].to_numpy(np.int64), CL['day'].astype(str).to_numpy(object)
    cl_kind, cl_st = CL['kind'].astype(str).to_numpy(object), CL['status'].astype(str).to_numpy(object)
    chk = dict(link_split_sums=[], link_split_matches_mechanism_counts=[], link_split_equals_census_links=[], unrelated_in_every_saved_feasible_set=[],
               two_sided_split_sums=[], ladder_all_overlap_certain=[], post_levels_rederived_equal=[], pull_cancel_counts_equal_pulls_csv=[])
    for r in A.to_dict('records'):
        t, sc, d, n = str(r['action_type']), str(r['scope']), str(r['day']), int(r['n'])
        fam = action_family(t)
        base = dict(source='chains/take/actions.csv', action_type=t, scope=sc, day=d, n_action_total=n, population=r['population'],
                    identified_repost=False, n_identified=NAN, identified_basis='')
        if fam == 'scenario_dependent_link':
            ex = json.loads(r['extra']) if isinstance(r['extra'], str) and r['extra'].strip() else {}
            ls, fs = ex.get('link_status', {}), ex.get('feasible', {})
            chk['link_split_sums'].append(sum(int(v) for v in ls.values()) == n)
            chk['link_split_matches_mechanism_counts'].append(
                int(ls.get('tied', 0)) == int(r['n_timing_sensitive']) and int(ls.get('sequential_identified', 0)) == int(r['n_compatible'])
                and int(ls.get('overlap_identified', 0)) == int(r['n_contradicted']))
            chk['unrelated_in_every_saved_feasible_set'].append(all('unrelated' in str(k).split('+') for k in fs))
            m = _scope_mask(cl_rank, cl_day, sc, d) & (cl_kind == t)
            chk['link_split_equals_census_links'].append(dict(Counter(cl_st[m])) == {str(k): int(v) for k, v in ls.items() if int(v)})
            for st in sorted(ls):
                sub, ident = link_subset(str(st))
                k = int(ls[st])
                rows.append(dict(base, subset=sub, n=k, evidence_grade=fam, identified_repost=ident, n_identified=k if ident else 0,
                                 identified_basis='0 by link_hypotheses: unrelated stays feasible (facts_soft.unrelated_in_every_saved_feasible_set)',
                                 grade_basis=_basis(t, fam) + f' (link status {st})'))
        elif t == 'two_sided':
            ex = json.loads(r['extra']) if isinstance(r['extra'], str) and r['extra'].strip() else {}
            ce = ex.get('certain', {})
            kc, kf = int(ce.get('True', 0)), int(ce.get('False', 0))
            chk['two_sided_split_sums'].append(kc + kf == n and kf == int(r['timing_sensitive_n']))
            for sub, k in (('overlap_certain', kc), ('feasible_only', kf)):
                rows.append(dict(base, subset=sub, n=k, evidence_grade=fam, grade_basis=_basis(t, fam)))
        elif t == 'ladder_overlap':
            chk['ladder_all_overlap_certain'].append(int(r['n_compatible']) == n and int(r['timing_sensitive_n']) == 0)
            rows.append(dict(base, subset='overlap_certain', n=n, evidence_grade=fam,
                             grade_basis=_basis(t, fam) + '; chain_take counts only overlap_certain pairs'))
        elif fam == 'post_window':
            m = _scope_mask(t_rank, t_day, sc, d) & (lvt == t)
            chk['post_levels_rederived_equal'].append(int(m.sum()) == n)
            cg = Counter(t_g[m])
            for g in GRADES:
                rows.append(dict(base, subset=g, n=int(cg.get(g, 0)), evidence_grade=f'post_window:{g}', grade_basis=BASIS['post_window'] + WINDOW_BASIS[g]))
        elif fam == 'cancel_compatible_ending':
            view = t.split(':')[1]
            m = _scope_mask(pu_rank, pu_day, sc, d) & (pu_view == view)
            chk['pull_cancel_counts_equal_pulls_csv'].append(int(m.sum()) == n)
            rows.append(dict(base, subset='all', n=n, evidence_grade=fam, n_identified=int(pu_ident[m].sum()),
                             identified_basis='pulls.csv pull_identified == True', grade_basis=_basis(t, fam)))
        else:
            rows.append(dict(base, subset='all', n=n, evidence_grade=fam or '', grade_basis=_basis(t, fam) if fam else 'no declared grade'))
    # the post_fill successor of every filled order (the receipt's 2,962 reprice / 577 refill)
    p_rank, p_day = P['rank'].to_numpy(np.int64), P['day'].astype(str).to_numpy(object)
    nk, ns = P['next_kind'].astype(str).to_numpy(object), P['next_link_status'].astype(str).to_numpy(object)
    for sc in SCOPES:
        for d in sorted(set(p_day)) + ['all']:
            m = _scope_mask(p_rank, p_day, sc, d)
            for kind in ('reprice', 'refill', 'none'):
                mk = m & (nk == kind)
                tot = int(mk.sum())
                for st, k in sorted(Counter(ns[mk]).items()):
                    base = dict(source='chains/post_fill/orders.csv', action_type=f'successor_{kind}', scope=sc, day=d, n=int(k), n_action_total=tot,
                                population='outcome_selected_filled_orders', identified_repost=False, n_identified=NAN,
                                identified_basis='relations are not saved in orders.csv; link_hypotheses keeps unrelated feasible')
                    if kind == 'none':
                        need(st == 'none', f'post_fill: next_kind none with link status {st!r} (refused)')
                        rows.append(dict(base, subset='none', evidence_grade='unresolved_alternatives', grade_basis=BASIS['successor_none']))
                    else:
                        sub, ident = link_subset(str(st))
                        rows.append(dict(base, subset=sub, evidence_grade='scenario_dependent_link', identified_repost=ident,
                                         n_identified=int(k) if ident else NAN, grade_basis=BASIS['link'] + f' (link status {st})'))
    CEN = pd.DataFrame(rows, columns=list(CENSUS_COLS))
    for k, v in chk.items():
        (hard if k in ('link_split_sums', 'two_sided_split_sums', 'post_levels_rederived_equal') else soft)[k] = _pred(v)
    soft['pull_identified_true_in_pulls_csv'] = dict(n=int(len(pu_ident)), n_true=int(pu_ident.sum()), holds=bool(not pu_ident.any()))
    lk = LED[LED['row_kind'].astype(str) == 'link']
    soft['unrelated_in_every_ledger_link_hypothesis'] = _pred([('unrelated' in str(h).split('|')) for h in lk['hypothesis']])
    hard['every_census_row_graded'] = _pred(CEN['evidence_grade'].isin(CENSUS_GRADES).to_numpy(bool))
    return CEN, hard, soft


def census_bytes(CEN):
    """the census writer (contract 5 'observed vs inferred'): refuses an action row without an evidence grade and a scenario-dependent link
    counted as an identified repost."""
    if _mut() != 'ungraded_ok':                                   # planted: the ungraded-row refusal switched off
        bad = CEN[~CEN['evidence_grade'].isin(CENSUS_GRADES)]
        if len(bad):
            b0 = bad.iloc[0]
            raise ValueError(f"census refuses an ungraded action row: {b0['action_type']} / {b0['subset']} (evidence grade {b0['evidence_grade']!r})")
    ident = np.array([bool(x) for x in CEN['identified_repost']], bool)
    nid = CEN['n_identified'].to_numpy(float)
    lk = (CEN['evidence_grade'].to_numpy(object) == 'scenario_dependent_link') & (ident | (np.nan_to_num(nid, nan=0.) > 0))
    if lk.any():
        b0 = CEN[lk].iloc[0]
        raise ValueError(f"census refuses a scenario-dependent link counted as an identified repost: {b0['action_type']} / {b0['subset']}")
    return CEN.to_csv(index=False).encode()


# ------------------------------------------------------------------ contract 4: paths ------------------------------------------------------------------
def select_paths(CS):
    """per rank, the first two episodes in cases.csv row order (first appearance): predetermined, never outcome-selected."""
    sel = []
    for r in (1, 2, 3):
        seen = []
        for e, rk in zip(CS['episode_id'], CS['rank']):
            if int(rk) == r and str(e) not in seen:
                seen.append(str(e))
                if len(seen) == 2:
                    break
        sel += [(r, e) for e in seen]
    return sel


def _rel_s(x, ws):
    v = _fin(x)
    return 'na' if np.isnan(v) or v < 0 else f'ws{(v - ws) / 1000.:+.3f}s'


def _scen_sets(led):
    """size_end ledger rows -> ({hypothesis: (size_lo, size_hi, feasible_union)}, {scenario: feasible set})."""
    hyp, per = {}, {s: [] for s in SCEN}
    for r in led:
        if str(r['row_kind']) != 'size_end':
            continue
        h = str(r['hypothesis'])
        hyp[h] = (_fin(r['size_lo']), _fin(r['size_hi']), bool(_b(r['feasible_union'])))
        for s in SCEN:
            if _b(r[f'feasible_{s}']):
                per[s].append(h)
    return hyp, {s: sorted(v) for s, v in per.items()}


def post_alternative(g, b, life):
    if g == G0:
        return 'window unbounded or order invalid: any posting time is compatible.'
    if g == G1:
        return 'conflict flags void the capacity claims: the posting time is unresolved.'
    if g == G3:
        return 'constrained window with a flagged status: not certified; an early post is not excluded by this row.'
    if g == G4:
        return (f'scenario window only (lifecycle {life}): the scenario-free bound is not saved; an earlier post whose first partial fill '
                'was masked by replenishment would make this a long-resting order.')
    if b in ('le_1s', 'le_10s'):
        return ('bound-constrained <= 10 s: an early queue-building post is excluded only under H1 / H2 / capacity and the 2 s maximum match '
                'lag; a later match or a broken premise reopens it.')
    return 'bound-constrained but wider than 10 s: an early queue-building post and a late post are both compatible.'


def paths_md(sel, CS, ES, EP, EPR, LED, P, T, PU):
    """contract 4 paths: per selected episode post -> fills -> terminal -> successor -> takes -> saved economics, with alternatives."""
    cs = CS.to_dict('records')
    eids = {e for _, e in sel}
    ck = {str(c['case_key']) for c in cs if str(c['episode_id']) in eids}
    keys = {(str(c['day']), int(c['mid']), str(c['key'])) for c in cs if str(c['episode_id']) in eids and str(c['kind']) == 'order'}

    def pick(df, h='mk_hash'):                                    # only the rows of the selected orders (to_dict on the whole table is costly)
        m = np.array([(d, int(x), y) in keys for d, x, y in zip(_strs(df['day']), df['mid'], _strs(df[h]))], bool)
        return df[m].to_dict('records')
    led, ep, epr = {}, {}, {}
    for src, dst in ((LED, led), (EP, ep), (EPR, epr)):
        for r in src[src['case_key'].astype(str).isin(ck)].to_dict('records'):
            dst.setdefault(str(r['case_key']), []).append(r)
    pmap = {(str(r['day']), int(r['mid']), str(r['mk_hash'])): r for r in pick(P)}
    tmap = {(str(r['day']), int(r['mid']), str(r['mk_hash'])): r for r in pick(T)}
    pu = {}
    for r in pick(PU):
        pu.setdefault((str(r['day']), int(r['mid']), str(r['mk_hash'])), {})[str(r['view'])] = r
    esm = {str(r['episode_id']): r for r in ES.to_dict('records')}
    L = ['# Logic-chain correction v1 - episode paths (contract section 4)', '',
         'Selection (predetermined, not outcome-selected): per rank, the first two episodes in `order_estimator/cases.csv` row order of first '
         'appearance. Saved evidence only; descriptive, four consumed days; no identified intent. Times are venue ms relative to the market '
         'window start ws. Evidence grades: observed_execution, continued_execution, scenario_dependent_link (tied / sequential_under_scenarios), '
         'cancel_compatible_ending, post_window:G0..G4 (section 1); size priors are prior-driven estimates.', '']
    for k, (r, eid) in enumerate(sel, 1):
        need(eid in esm, f'episode {eid} is not in episode_summary.csv (refused)')
        e = esm[eid]
        rows = [(i, c) for i, c in enumerate(cs) if str(c['episode_id']) == eid]
        L += [f'## Path {k} - rank {r}: episode {eid}', '',
              f"day {e['day']}, market {int(e['mid'])}, address {e['addr']}; cases {e['case_keys']} (first appearance: cases.csv data row "
              f'{rows[0][0] + 1})', '']
        for i, c in rows:
            if str(c['kind']) != 'order':
                continue
            key = (str(c['day']), int(c['mid']), str(c['key']))
            need(key in pmap and key in tmap, f'case {c["case_key"]}: order not in post_fill orders.csv (refused)')
            p, t = pmap[key], tmap[key]
            ws = float(p['ws_ms'])
            hyp, per = _scen_sets(led.get(str(c['case_key']), []))
            sz = []
            for h in ('EX', 'PL', 'EC'):
                if h in hyp and hyp[h][2]:
                    lo_, hi_ = hyp[h][0], hyp[h][1]
                    sz.append(f'{h} {_fmt(lo_)} sh' if h == 'EX' else (f'{h} [{_fmt(lo_)}, {_fmt(hi_)}] sh' if not np.isnan(hi_) else f'{h} >= {_fmt(lo_)} sh'))
            q = [x for x in epr.get(str(c['case_key']), []) if str(x['mechanism']) == QUEUE_MECH]
            qtxt = (f"queue_priority_before_demand {q[0]['identification_status']} -> {q[0]['identification_status_corrected']}; evidence grade "
                    f"{q[0]['evidence_grade']} -> {q[0]['evidence_grade_corrected']}") if q else 'no saved POST row'
            L += [f"- **POST** order {str(c['key'])[:12]}: {p['y_side']} at {int(p['tick'])}c; size alternatives {', '.join(sz) or 'none saved'} "
                  f"(size status {c['size_status']}; prior-driven: P_cmd {c['P_cmd']}, P_drop {c['P_drop']}, P_lot {c['P_lot']}, P_clip {c['P_clip']}). "
                  f"Union window [{_rel_s(t['post_lo_union_ms'], ws)}, {_rel_s(t['post_hi_ms'], ws)}], width {_fmt(t['window_width_ms'], 0)} ms "
                  f"({t['width_bin']}); lifecycle post_class {t['life_post_class']}; packet status {t['status_packet']} -> **{t['grade']}** "
                  f"({t['window_basis']}). Logic table: {qtxt}.",
                  f"  - alternative that changes the logic: {post_alternative(t['grade'], t['width_bin'], t['life_post_class'])}",
                  f"- **FILLS** {int(p['n_fills'])} fill(s), {_fmt(p['cum_shares'])} sh (observed executions); first <= {_rel_s(p['first_hi_ms'], ws)}, "
                  f"last <= {_rel_s(p['last_hi_ms'], ws)}; {int(p['n_later_fills'])} later fill(s) of the same hash (continued execution, not a private "
                  f"KEEP); queue at the first fill {p['queue_class']}."]
            v = pu.get(key, {})
            ends = [x for x in ep.get(str(c['case_key']), []) if str(x['link']) == 'END']
            etxt = '; '.join(f"{x['mechanism']} {x['identification_status']}" + (f" (window condition {x['conditional_status']})"
                                                                              if isinstance(x['conditional_status'], str) and x['conditional_status'] else '')
                             for x in ends) or 'none'
            u = set(str(p['union_histories']).split('|'))
            alt = ('EX / PL tie: if the order was exhausted there was no pull decision; if pulled, the END context is a candidate trigger only.'
                   if {'EX', 'PL'} <= u else f"no pull history in the declared scenarios; the size stays {c['size_status']}.")
            L += [f"- **TERMINAL** feasible end histories {p['union_histories']}; per scenario "
                  + (', '.join(f"{s} {{{','.join(per[s])}}}" for s in SCEN) if hyp else '(not in the ledger)')
                  + f"; cancel-compatible (EX/PL tie) under S_FIFO: {'yes' if 'S_FIFO' in v else 'no'}, S_NOPRIO: {'yes' if 'S_NOPRIO' in v else 'no'}"
                  + (f" (pull_identified {', '.join(str(_b(v[x]['pull_identified'])) for x in sorted(v))}; removal class "
                     f"{', '.join(str(v[x]['removal_class']) for x in sorted(v))})" if v else '')
                  + f"; saved END rows: {etxt}.",
                  f'  - alternative that changes the logic: {alt}']
            nk, nst = str(p['next_kind']), str(p['next_link_status'])
            if nk == 'none':
                L += [f"- **SUCCESSOR** none: {BASIS['successor_none']}.",
                      '  - alternative that changes the logic: a pull without a repost and an unfilled repost are both compatible.']
            else:
                sub, ident = link_subset(nst)
                rel = [x for x in led.get(str(c['case_key']), []) if str(x['row_kind']) == 'link' and str(x['role']) == 'child'
                       and str(x['other_hash']) == str(p['next_hash'])]
                alt = ('the successor may already have been resting (a ladder / overlapping order) or be unrelated: no repost decision is identified.'
                       if sub == 'tied' else 'an unrelated later order: a sequential timing does not identify a repost or its intent.')
                L += [f"- **SUCCESSOR** {nk} to {str(p['next_hash'])[:12]} (first of {int(p['n_children'])} filled successor(s)); link status {nst}, "
                      f"gap [{_fmt(p['next_gap_lo_ms'], 0)}, {_fmt(p['next_gap_hi_ms'], 0)}] ms; relations "
                      f"{rel[0]['hypothesis'] if rel else 'not in the ledger'} -> {sub}, evidence grade scenario_dependent_link; identified repost: "
                      f"{'yes' if ident else 'no'}.",
                      f'  - alternative that changes the logic: {alt}']
        tk = [(i, c) for i, c in rows if str(c['kind']) == 'take']
        if not tk:
            L += [f"- **TAKES** no selected take case in this episode (its window holds {_fmt(e['ep_n_taker'], 0)} taker leg(s), not classified here)."]
        for i, c in tk:
            logic = [x for x in ep.get(str(c['case_key']), []) if str(x['link']) == 'TAKE']
            ltxt = '; '.join(f"{x['mechanism']} {x['identification_status']}" for x in logic) or 'none'
            ts = bool(_b(c['timing_sensitive']))
            side = 'buy' if _fin(c['s']) > 0 else 'sell'
            L += [f"- **TAKE** record {str(c['key'])[:13]}: YES {side} {_fmt(c['q'])} sh @ {_fmt(c['p_avg'], 3)}; position {_fmt(c['pos_before'])} -> "
                  f"{_fmt(c['pos_after'])}; class recorded {c['cls']} / certain {c['cls_certain']} / simultaneous {c['cls_simultaneous']} "
                  f"(timing-sensitive {ts}); own fills in the 5 s before: certain {_fmt(c['burst_cert'], 0)}, possible {_fmt(c['burst_pos'], 0)}; "
                  f"status {c['status']}; evidence grade observed_execution (class on the retained status-ok ledger). Saved TAKE rows: {ltxt}.",
                  '  - alternative that changes the logic: '
                  + ('another fill ordering inside the timing windows changes the inventory class.' if ts else
                     'an excluded earlier leg or an unobserved inventory path could change reduce vs add (the class is conditional on the retained ledger).')]
        L += [f"- **ECONOMICS** (saved episode_summary.csv; descriptive: a later outcome is not the value of an action) legs {_fmt(e['ep_n_legs'], 0)} "
              f"(maker {_fmt(e['ep_n_maker'], 0)}, taker {_fmt(e['ep_n_taker'], 0)}), {_fmt(e['ep_sh'])} sh; ep_net_end {_fmt(e['ep_net_end'])}, "
              f"ep_net_res {_fmt(e['ep_net_res'])}, fee {_fmt(e['ep_fee'])}; mq {_fmt(e['ep_mq'])}, rq {_fmt(e['ep_rq'])}; position "
              f"{_fmt(e['ep_pos_in'])} -> {_fmt(e['ep_pos_out'])}; outcome {e['ep_outcome']}; matched gap {e['mg_status']}: close_net "
              f"{_fmt(e['mg_close_net'])}, resid {_fmt(e['mg_resid'])}, total {_fmt(e['mg_total'])} ({_fmt(e['mg_close_net_c_sh'])} c/sh).", '']
    return '\n'.join(L) + '\n'


# ------------------------------------------------------------------ build / claims ------------------------------------------------------------------
def reproduction(T, P):
    """the reviewer's recount (findings 1 / 5) as recorded predicates: reviewer value, computed value, equality."""
    pc, life = T['post_class_packet'].to_numpy(object), T['life_post_class_S_NOPRIO'].to_numpy(object)
    short = np.isin(pc, list(SHORT))
    ok = T['status_packet'].to_numpy(object) == OK
    rf = np.array([bool(_flags(x) & REVIEWER_CONFLICT_FLAGS) for x in T['status_packet']], bool)
    nk, ns = P['next_kind'].astype(str).to_numpy(object), P['next_link_status'].astype(str).to_numpy(object)
    got = dict(n_orders=len(T), late_post=int((pc == 'late_post').sum()), short_rest=int((pc == 'short_rest').sum()), scenario_short=int(short.sum()),
               short_life_constrained=int((short & (life == C.CONSTRAINED)).sum()),
               short_life_constrained_status_ok=int((short & (life == C.CONSTRAINED) & ok).sum()),
               short_assumption_sensitive=int((short & (life == C.SENSITIVE)).sum()), short_conflict_flagged=int((short & rf).sum()),
               successor_reprice=int((nk == 'reprice').sum()), successor_refill=int((nk == 'refill').sum()),
               successor_tied=int(((nk != 'none') & (ns == 'tied')).sum()), successor_reprice_tied=int(((nk == 'reprice') & (ns == 'tied')).sum()),
               successor_refill_tied=int(((nk == 'refill') & (ns == 'tied')).sum()),
               successor_sequential=int(((nk != 'none') & (ns == 'sequential_identified')).sum()))
    return {k: dict(reviewer=REVIEWER[k], computed=got[k], equal=got[k] == REVIEWER[k], difference=got[k] - REVIEWER[k]) for k in REVIEWER}


def build(D, L):
    P, CO, CL, EP, ES, CS, LED, A, PU = (D[k] for k, _ in INPUTS)
    days = sorted(set(P['day'].astype(str)))
    LP = life_pivot(L)
    M = assemble(P, CO, LP)
    T, h1, s1 = order_table(M)
    grid = regrade_grid(T, days)
    EPR, h2, s2 = regrade_episodes(EP, T)
    CEN, h3, s3 = census(A, T, P, PU, CL, LED)
    sel = select_paths(CS)
    md = paths_md(sel, CS, ES, EP, EPR, LED, P, T, PU)
    hard, soft = {**h1, **h2, **h3}, {**s1, **s2, **s3}
    rep = reproduction(T, P)
    g, b = T['grade'].to_numpy(object), T['width_bin'].to_numpy(object)
    short = T['scenario_short'].to_numpy(bool)
    eg0, eg1 = EPR['evidence_grade'].astype(object).to_numpy(object), EPR['evidence_grade_corrected'].to_numpy(object)
    q = EPR['mechanism'].astype(str).to_numpy(object) == QUEUE_MECH
    id0, id1 = EPR['identification_status'].astype(object).to_numpy(object), EPR['identification_status_corrected'].to_numpy(object)
    sd = CEN['subset'].to_numpy(object)
    pa_ = CEN[(CEN['scope'] == 'pooled') & (CEN['day'] == 'all')]             # one count per action (scopes and days overlap)
    act_pa = pa_[pa_['source'] == 'chains/take/actions.csv']
    suc_pa = pa_[pa_['source'] == 'chains/post_fill/orders.csv']
    lk_pa = act_pa[act_pa['evidence_grade'] == 'scenario_dependent_link']
    claims = dict(
        version=VERSION, contract='chain_correction_contract.md sections 1, 4, 5 (grade / observed vs inferred)',
        population=dict(unit='filled top-3 orders (outcome-selected: unfilled orders are invisible)', n_orders=int(len(T)), days=days,
                        n_markets=int(len(set(zip(T['day'], T['mid'])))), by_rank={f'rank{r}': int((T['rank'] == r).sum()) for r in (1, 2, 3)}),
        facts_hard=hard, facts_soft=soft, reviewer_reproduction=rep,
        reviewer_reproduction_all_equal=bool(all(v['equal'] for v in rep.values())),
        grades=dict(by_grade={x: int((g == x).sum()) for x in GRADES},
                    by_grade_bin={f'{x}|{y}': int(((g == x) & (b == y)).sum()) for x in GRADES for y in BINS},
                    scenario_short_by_grade={x: int(((g == x) & short).sum()) for x in GRADES},
                    g2_scenario_short_equals_reviewer_constrained_ok=bool(int(((g == G2) & short).sum()) == REVIEWER['short_life_constrained_status_ok']),
                    g1_scenario_short=int(((g == G1) & short).sum())),
        allowed_claims=dict(n_bound_constrained_window_le_10s=int(((g == G2) & np.isin(b, ['le_1s', 'le_10s'])).sum()),
                            n_bound_constrained_window_le_1s=int(((g == G2) & (b == 'le_1s')).sum()), n_orders_denominator=int(len(T)),
                            statement_form='n orders have a bound-constrained posting window <= 10 s (conditional on H1 / H2 / capacity and the maximum match lag)'),
        scenario_estimates=dict(union_width_class_counts={x: int((T['post_class_packet'] == x).sum()) for x in
                                                          ('late_post', 'short_rest', 'early_or_late_unidentified', 'unidentified')},
                                note_form='scenario union widths: estimates, not proofs of resting time'),
        logic_rows=dict(n_post_rows=int(len(EPR)), n_post_orders=int(EPR['case_key'].nunique()),
                        evidence_grade_original=dict(Counter(eg0)), evidence_grade_corrected=dict(Counter(eg1)),
                        evidence_grade_changed=int((eg0 != eg1).sum()), queue_rows=int(q.sum()),
                        queue_status_original=dict(Counter(id0[q])), queue_status_corrected=dict(Counter(id1[q])),
                        queue_contradicted_withdrawn=int(((id0 == 'contradicted') & (id1 != 'contradicted') & q).sum()),
                        post_grade_of_rows=dict(Counter(EPR['post_grade'].to_numpy(object)[q]))),
        census=dict(n_rows=int(len(CEN)), all_rows_graded=bool(CEN['evidence_grade'].isin(CENSUS_GRADES).all()),
                    identified_repost_rows=int(np.array([bool(x) for x in CEN['identified_repost']], bool).sum()),
                    tied_rows_identified=int(((sd == 'tied') & np.array([bool(x) for x in CEN['identified_repost']], bool)).sum()),
                    actions_by_grade_pooled_all={str(k): int(v) for k, v in act_pa.groupby('evidence_grade')['n'].sum().items()},
                    links_pooled_all={f'{a}|{s}': int(n) for a, s, n in zip(lk_pa['action_type'], lk_pa['subset'], lk_pa['n'])},
                    successors_pooled_all={f'{a}|{s}': int(n) for a, s, n in zip(suc_pa['action_type'], suc_pa['subset'], suc_pa['n'])},
                    n_pull_identified_pooled_all={str(a): (int(_fin(x)) if np.isfinite(_fin(x)) else None)
                                                  for a, x in zip(act_pa['action_type'], act_pa['n_identified']) if str(a).endswith(':cancel_possible')}),
        paths=dict(selection='per rank the first two episodes in cases.csv row order of first appearance', episodes=[dict(rank=r, episode_id=e) for r, e in sel]))
    return dict(T=T, grid=grid, EPR=EPR, CEN=CEN, md=md, claims=claims, hard=hard, soft=soft, sel=sel)


def serialize(res):
    return [('post_regrade.csv', res['grid'].to_csv(index=False).encode()),
            ('post_regrade_orders.csv', res['T'].to_csv(index=False).encode()),
            ('episodes_post_regraded.csv', res['EPR'].to_csv(index=False).encode()),
            ('census_graded.csv', census_bytes(res['CEN'])),
            ('paths.md', res['md'].encode()),
            ('claims_post.json', _json_bytes(res['claims']))]


def write_files(out, files):
    """write-once, file by file: every name refused if it exists; exclusive create; the files of this call removed again on a failure."""
    for nm, _ in files:
        need(not (out / nm).exists(), f'write-once: {out / nm} exists (refused)')
    done = []
    try:
        for nm, b in files:
            with open(out / nm, 'xb') as f:
                done.append(out / nm)
                f.write(b)
    except BaseException:
        for p in done:
            p.unlink(missing_ok=True)
        raise


def run(dry=False, rroot=None, life_dir=None, out=None, require_freeze=True, log=print):
    t0, asof0 = time.time(), _now()
    need(C.RUN == 'v2', f'MT_RUN must be v2 (is {C.RUN!r})')
    need(not _mut(), f'MT_CC_MUT={_mut()!r} is set: a planted mutant never runs here (refused)')
    rroot = Path(rroot) if rroot else C.RECEIPT
    life_dir = Path(life_dir) if life_dir else C.OUT / 'life'
    out = Path(out) if out else C.RECEIPT / 'chains' / 'correction_v1'
    ex = [str(out / nm) for nm in OUT_FILES if (out / nm).exists()]
    need(not ex, f'write-once: {", ".join(ex)} exists (refused before any input is read)')
    git = _git(TOOL_FILES)
    if not dry and require_freeze:
        need(bool(git.get('unmodified')), 'freeze: the tool, its selftest and the contract must be tracked and unmodified before the run (refused)')
    D, L, info = read_inputs(rroot, life_dir)
    guard('inputs')
    cpu_guard('inputs')
    res = build(D, L)
    guard('tables')
    cpu_guard('tables')
    failed = sorted(k for k, v in res['hard'].items() if not v['holds'])
    if dry:
        digest = dict(mode='dry (nothing written)', schema={k: dict(rows=v['rows'], columns=v.get('columns'), required_columns=v.get('required_columns'),
                                                                     missing_columns=v.get('missing_columns'), sha256=v['sha256'][:16])
                                                             for k, v in info.items() if not k.startswith('life/')},
                      lifecycle_files=dict(n=sum(k.startswith('life/') for k in info), rows_used=sum(v.get('rows_used', 0) for k, v in info.items()
                                                                                                    if k.startswith('life/'))),
                      hard_checks=res['hard'], hard_failed=failed, soft_checks=res['soft'], reviewer_reproduction=res['claims']['reviewer_reproduction'],
                      grades=res['claims']['grades'], allowed_claims=res['claims']['allowed_claims'], logic_rows=res['claims']['logic_rows'],
                      census=res['claims']['census'], paths=res['claims']['paths'], would_write=[str(out / nm) for nm in OUT_FILES],
                      git=git, cpu_s=round(time.process_time(), 3), vmhwm_mib=round(hwm(), 1))
        log(json.dumps(_clean(digest), indent=1, sort_keys=True))
        need(not failed, f'hard checks failed: {failed} (a run would refuse)')
        return digest
    need(not failed, f'hard checks failed: {failed} (refused before any output is written)')
    files = serialize(res)
    srcs = {_rel(f): _sha(f) for f in TOOL_FILES + DEP_FILES if Path(f).is_file()}
    rec = dict(version=VERSION, kind='run', contract=_rel(CONTRACT_FILE), module=_rel(Path(__file__)), run=C.RUN, asof_start=asof0, asof_end=_now(),
               git=git, sources=srcs, inputs=info,
               outputs={nm: dict(sha256=hashlib.sha256(b).hexdigest(), bytes=len(b)) for nm, b in files},
               rows=dict(post_regrade=len(res['grid']), post_regrade_orders=len(res['T']), episodes_post_regraded=len(res['EPR']),
                         census_graded=len(res['CEN'])),
               constants=dict(LATE_MS=LATE_MS, EARLY_MS=EARLY_MS, G1_FLAGS=sorted(G1_FLAGS), INVALID=sorted(INVALID), CERT=list(CERT), GRADES=list(GRADES),
                              LOGIC_GRADE=LOGIC_GRADE, BINS=list(BINS), SCOPES=list(SCOPES), REVIEWER=REVIEWER,
                              REVIEWER_CONFLICT_FLAGS=sorted(REVIEWER_CONFLICT_FLAGS), CENSUS_GRADES=list(CENSUS_GRADES)),
               hard_checks=res['hard'], hard_failed=failed,
               cpu_s=dict(total=round(time.process_time(), 3), cap=CPU_CAP_S), wall_s=round(time.time() - t0, 2), vmhwm_mib=round(hwm(), 1),
               guard_mib=C.GUARD_MIB)
    out.mkdir(parents=True, exist_ok=True)
    write_files(out, files + [('receipt_post.json', _json_bytes(rec))])
    log(json.dumps(_clean(dict(out=str(out), outputs=rec['outputs'], rows=rec['rows'], cpu_s=rec['cpu_s'], vmhwm_mib=rec['vmhwm_mib'],
                               reviewer_reproduction_all_equal=res['claims']['reviewer_reproduction_all_equal'])), indent=1, sort_keys=True))
    return rec


if __name__ == '__main__':
    a = sys.argv[1:]
    if a == ['run']:
        run()
    elif a == ['run', '--dry']:
        run(dry=True)
    else:
        raise SystemExit('usage: MT_RUN=v2 chain_correct_post.py run [--dry]')
