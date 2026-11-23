"""chains_v4.py - SIX COMPLETE CHAINS v4 (P-2026-005 MAKER-TRACE; TOUCH-TRANSITION-AND-RESPONSE v1, section 5; session aea5bfb9 builder,
2026-11-23). Built EXACTLY to predictfun/reviews/maker_trace_20261122/touch_response_contract.md section 5 (frozen 668e560db) = the chain
corrections of review 01a72b4e2 (reviews/automatic/20261123_posting_chain_review/report.md, findings 3-4). USER goal: the top makers' ACTION
CHAINS and their logic. It IMPORTS the frozen chains_v3.py (never edited) and overrides only what section 5 corrects: the selection (the same
6 predetermined episodes), every input, reader, binding, reconciliation and the POST / ENDING / REPOST / TAKES builders are v3's, called
unchanged. SAVED OUTPUTS ONLY (no market, book, lifecycle, queue or FIFO is recomputed); DESCRIPTIVE on consumed days (Oct 24-27; the six
chains lie on Oct 24): no intent, no identified private action, no model skill, no profitability claim; filled orders are outcome-selected.
CORRECTIONS
  finding 3  PRE-STATE carries the order's post grade (correction_v1/post_regrade_orders.csv, the POST link's row). v3's state vectors are kept
             as a CONDITIONAL WINDOW DESCRIPTION; G0 / G1 / G3 windows and G4 windows (an earlier post masked by replenishment possible) get
             the alternative 'unresolved_outside_window', so they are never ONE-HISTORY; only a G2 (bound-constrained) window may agree.
             Evidence grade observed_state:post_window:<G>. The summary table (the post grade beside each reading) and the agreement lines
             are recomputed from these graded records and verified against them.
  finding 4  QUEUE masked netting has three states per compatible queue_front_v2 level row: possible (saved True) / excluded (saved False on a
             row whose calculation ran: queue_front_v2._new_life_v2 computes alt_netting_possible = alt_sites_life + alt_sites_unexplained > 0
             ONLY for a level with an earliest record - timing_basis book_candidate / settlement, n_earliest_set >= 1 - and both saved sites
             must be 0) / not_evaluated (every other False: the boolean is initialised False and the no-earliest-record branch returns before
             the calculation; a missing / inconsistent cell; v1 labels; no compatible level). Chain state: possible if any level is possible,
             else not_evaluated if any (or no level), else excluded; the masked-netting history is dropped only when excluded. READING: on
             queue_front_v2's code an evaluated row always has alt_sites_life >= 1, so 'excluded' cannot occur on its real output; the state
             is implemented and proved on a synthetic row. Gross markouts read 'gross c/sh' (the common cohort too), the resolution view
             'net c/sh' (v3's VALUE builder run with this label function; its numbers unchanged).
NEW LINK 1b TRANSITION (contract 5: the post cell matched to a section-1 frame pair / section-2 episode where its creation interval allows,
  with the candidate grade): the compatible fresh-level lives of the QUEUE link (queue_front_v2 levels.csv; v1 levels without it), each with
  its creation interval (t_open_ms, t_create_ms] = (upd(f-1), upd(f)]. Per creation: its position against the post window [post_lo_union,
  post_hi] (inside = posting_opportunity section 3's containment: lo < 0 or lo <= upd(f-1) + 1, and upd(f) <= hi; straddles / before /
  after); the touch_transition census frame pair with the same (upd(f-1), upd(f)] (class + two-side sub-label; none -> no_census_pair,
  several -> ambiguous_census_pair: statuses, never dropped); the section-2 episode on the order's side whose end is 'response' at upd(f)
  (and at upd(f-1) when its lower bound is saved): retreat time, step, exposure, response kind and the saved state; the CANDIDATE GRADE =
  posting_opportunity events.csv identity of that creation (candidate_constrained / candidate_scenario / owner_unknown with its reason, and
  whether its candidate hash is this order). Retained histories: one 'at_creation:level_<id>:<census context>[:response_to_<episode>(kind)]'
  per creation inside / straddling the window; 'joined_shown_cell' (added to the already-shown cell: no fresh-creation transition, its frame
  pair unidentified) when a compatible level is shown at an instant inside the window (t_create < post_hi and its last instant > post_lo);
  'masked_recreation_in_window' beside it unless the QUEUE masked-netting state is excluded; 'creation_not_saved' without a compatible
  level; 'unresolved_outside_window' unless G2. A candidate grade never removes a history (the eventual filler is never stated as the
  creator). Creations before / after the window are context lines only. touch_transition absent -> PENDING (draft) / refusal (run).
ASSUMED INTERFACE (touch_transition is built in parallel; a run refuses any other schema, a draft marks the link PENDING with the reason):
  v2/chains/touch_transition/receipt.json listing with sha256 census.csv (day, mid, upd_prev_ms, upd_ms, cls [+ sub]) and episodes.csv
  (day, mid, side, episode_id, start_ms, delta_ticks, end_type, end_ms, exposure_ms, response_kind [+ end_lo_ms]); accepted aliases TT_COLS
  / TT_OPT; display columns TT_SHOW when present. posting_opportunity events.csv (bound by its receipt) is read only when touch_transition is.
LINKS per chain in order: 1 pre_state, 1b transition, 2 post, 3 queue, 4a ending, 4b repost, 5 takes, 6 value (readings ONE-HISTORY / TIED /
  NO-EVENT / PENDING, v3's rule).
CHECKS recorded predicates (claims.json; a run refuses on any failure, a draft records it): v3's (selection, every link read, no identified
  private action, TIED never identified, submitted size never set, same-hash fills one order, bindings, reconciliations) plus: every link
  graded in its link's vocabulary (the pre-state carrying its post grade), no ONE-HISTORY pre-state unless G2, the unresolved alternative
  unless G2, no excluded alt-netting on a not-evaluated row, exclusion only with an evaluated False on every level, the summary table and
  agreement lines equal to the graded records, gross markouts labelled. The VERIFIER refuses in both modes before any write: v3's rendering
  rules and these properties, stated on the records / the raw saved cells independently of the builders.
OUTPUTS write-once (a partial directory renamed at the end): run -> v2/chains/chains_v4/ (requires every sibling receipt with its files bound:
  posting_opportunity, queue_front_v2, touch_transition), run --draft -> v2/chains/chains_v4_draft/ (TRANSITION links PENDING while
  touch_transition is absent): chains.md, chain_links.csv (one row per link), claims.json (computed predicates only), receipt.json (sha256 of
  sources, inputs, bindings and outputs; CPU by stage; VmHWM).
GUARDS MT_RUN=v2; MT_CH4_MUT (the selftest's planted errors) refused outside the selftest, MT_CH3_MUT always; the output refused if it exists
  (before any input); run refuses unless this tool, its selftest, the contract and the imported chains_v3.py / chains_v3_selftest.py are
  tracked and unmodified (the freeze); VmHWM self-guard pf_mt_common.GUARD_MIB (never raised); CPU self-guard CPU_CAP_S.
Usage (run_light, MT_RUN=v2): chains_v4.py run --draft | chains_v4.py run        selftest: chains_v4_selftest.py"""
import hashlib
import json
import os
import re
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
import chains_v3 as CH3  # noqa: E402   (the frozen v3 tool: imported, never edited)

C = CH3.C
need, OK = C.need, C.STATUS_OK
NAN, INF = float('nan'), float('inf')
_s, _fin, _int, _fmt, _sg, _rel_s, _h12, _sarr, _pred, _rel = (CH3._s, CH3._fin, CH3._int, CH3._fmt, CH3._sg, CH3._rel_s, CH3._h12, CH3._sarr,
                                                               CH3._pred, CH3._rel)
G0, G1, G2, G3, G4 = CH3.G0, CH3.G1, CH3.G2, CH3.G3, CH3.G4
VERSION = 'chains_v4'
CONTRACT_FILE = C.RECEIPT_V1 / 'touch_response_contract.md'
CONTRACT_COMMIT = '668e560db'
REVIEW = dict(commit='01a72b4e2', file='predictfun/reviews/automatic/20261123_posting_chain_review/report.md', findings='3-4')
SELFTEST_FILE = HERE / 'chains_v4_selftest.py'
TOOL_FILES = (Path(__file__).resolve(), SELFTEST_FILE, CONTRACT_FILE)
FREEZE_FILES = TOOL_FILES + (HERE / 'chains_v3.py', HERE / 'chains_v3_selftest.py')
DEP_FILES = (HERE / 'chains_v3.py', HERE / 'chains_v3_selftest.py', CH3.CONTRACT_FILE, RESEARCH / 'pf_mt_common.py')

# ------------------------------------------------------------------ declared constants (the contract's; none fitted) ------------------------------------------------------------------
LINKS = (('1', 'pre_state'), ('1b', 'transition'), ('2', 'post'), ('3', 'queue'), ('4a', 'ending'), ('4b', 'repost'), ('5', 'takes'), ('6', 'value'))
TITLE = dict(CH3.TITLE, transition='TRANSITION')
READINGS = CH3.READINGS
UNRESOLVED = 'unresolved_outside_window'
OUTSIDE = {G0: 'G0: the window is unbounded or the order invalid - the post may lie anywhere before the first fill',
           G1: 'G1: conflict flags void the window - the post may lie outside it',
           G3: 'G3: the bound carries a flagged status (not certified) - the post may lie outside it',
           G4: 'G4: a scenario-only window - an earlier post masked by replenishment outside the enumerated seconds is possible'}
PRE_GRADES = tuple(f'observed_state:post_window:{g}' for g in (G0, G1, G2, G3, G4))
POST_GRADES = tuple(f'post_window:{g}' for g in (G0, G1, G2, G3, G4))
LINK_GRADES = dict(pre_state=PRE_GRADES + ('pending',), transition=('transition_context', 'pending'), post=POST_GRADES,
                   queue=('scenario_queue_history',), ending=('cancel_compatible_ending', 'scenario_dependent_ending', 'unresolved_alternatives'),
                   repost=('scenario_dependent_link', 'unresolved_alternatives'), takes=('observed_execution',), value=('recorded_fill_accounting',))
GRADES = tuple(dict.fromkeys(g for v in LINK_GRADES.values() for g in v))
ALT_STATES = ('possible', 'excluded', 'not_evaluated')
ALT_TAG = dict(possible='masked_netting_possible', not_evaluated='masked_netting_not_evaluated')
EVAL_BASES = ('book_candidate', 'settlement')                    # queue_front_v2 timing_basis of a level with an earliest record
ALT_EVIDENCE = ('timing_basis', 'n_earliest_set', 'alt_sites_life', 'alt_sites_unexplained')
JOINED, MASKED, NOT_SAVED = 'joined_shown_cell', 'masked_recreation_in_window', 'creation_not_saved'
POSITIONS = ('inside', 'straddles', 'before', 'after')
MUTANTS = ('v3_pre_state_rule', 'alt_false', 'summary_ungraded', 'v3_value_label')   # planted errors: the selftest only
CPU_CAP_S = 8.
BARE_RE = re.compile(r'\$, (?:[+-][0-9]+\.[0-9]+|na) c/sh')    # v3's unlabelled per-share markout right after net dollars
TABLE_RE = re.compile(r'^\| \d+ \| \d+ \| ')
AGREE_RE = re.compile(r'^- \*\*AGREEMENT\*\* histories agree on: (?P<agree>.*?); TIED: (?P<tied>.*?); pending: (?P<pend>.*?); '
                      r'first branching link: (?P<fb>.*)$')

# ---- the sibling outputs: files, required / shown columns (posting_opportunity and queue_front_v2 as v3 plus the columns section 5 needs) ----
QF2_NEED = CH3.SIBLINGS['queue_front_v2']['need']
QF2_SHOW = dict(levels=CH3.QF2_SHOW['levels'] + ('n_earliest_set', 'alt_sites_life', 'alt_sites_unexplained', 'frame', 'fid'),
                candidates=CH3.QF2_SHOW['candidates'] + ('n_earliest_set', 'alt_sites_life', 'alt_sites_unexplained'),
                fills=CH3.QF2_SHOW['fills'], occupancy=CH3.QF2_SHOW['occupancy'])
PO_EVENTS_NEED = ('day', 'mid', 'side', 'tick', 't_open_ms', 't_create_ms', 'status', 'identity', 'identity_reason', 'cand_hash', 'win_contains', 'grade')
PO_EVENTS_SHOW = ('key', 'fid', 'cand_addr')
TT_FILES = dict(census=('reconcile.csv', 'census.csv', 'census.parquet'), episodes=('episodes.csv', 'episodes.parquet'))   # integrator: touch_transition's per-creation frame-pair classes live in reconcile.csv (its census.csv is aggregated)
TT_COLS = dict(census=dict(day=('day',), mid=('mid',), lo=('t_open_ms', 'upd_prev_ms', 't_lo_ms', 'lo_ms'), hi=('t_create_ms', 'upd_ms', 't_hi_ms', 'hi_ms'),
                           cls=('cls', 'class', 'transition')),
               episodes=dict(day=('day',), mid=('mid',), side=('side',), eid=('episode_id',), start=('start_ms', 't_start_ms'),
                             delta=('delta_ticks', 'delta'), end=('end_type', 'end'), end_hi=('end_ms', 'end_hi_ms', 't_end_ms'),
                             exposure=('exposure_ms',), kind=('resp_type', 'response_kind', 'kind')))
TT_OPT = dict(census=dict(sub=('sublabel', 'sub', 'sub_label', 'two_side_sub')), episodes=dict(end_lo=('resp_lo_ms', 'end_lo_ms', 'response_lo_ms')))
TT_SHOW = dict(census=(), episodes=('spread_bin_D0', 'depth_own_bin_D0', 'btc_5s_aligned_D0', 'fair_gap_bin_D0', 'time_left_s_D0',
                                    'spread_bin', 'spread_ticks', 'delta_bin', 'depth_own_bin', 'depth_touch_own_bin', 'btc_5s_aligned', 'fair_gap_bin',
                                    'time_left_bin', 'timing_sensitive', 'owner_status', 'owner_grade', 'owner_hash', 'owner_addr'))
SIBLINGS = dict(
    posting_opportunity=dict(dir='chains/posting_opportunity', files=dict(rows=('rows.parquet',)), optional=dict(events=('events.csv',))),
    queue_front_v2=dict(dir='chains/queue_front_v2', files=dict(levels=('levels.csv',), candidates=('candidates.csv',)),
                        optional=dict(fills=('fills.csv',), occupancy=('occupancy.csv',))),
    touch_transition=dict(dir='chains/touch_transition', files=dict(census=TT_FILES['census'], episodes=TT_FILES['episodes']), optional={}))
LINK_COLS = CH3.LINK_COLS + ('post_grade', 'v3_reading', 'alt_netting_state', 'alt_level_states', 'candidate_grades')
OUT_FILES = CH3.OUT_FILES
_V3_VLINE = CH3._vline                                            # v3's label function (the mutant's; restored after every value link)


# ------------------------------------------------------------------ small helpers ------------------------------------------------------------------
def _mut():
    return os.environ.get('MT_CH4_MUT', '')


def cpu_guard(where, c0=0.):
    c = time.process_time() - c0
    need(c < CPU_CAP_S, f'CPU self-guard: {c:.2f} s >= {CPU_CAP_S} s at {where}')
    return c


def _b3(x):
    """a saved boolean cell -> True / False / None (missing or not a boolean: never read as False)."""
    try:
        return CH3._b(x)
    except ValueError:
        return None


def _header(p):
    p = Path(p)
    return pq.ParquetFile(p).schema_arrow.names if p.suffix == '.parquet' else list(pd.read_csv(p, nrows=0).columns)


def _resolve(have, spec):
    """logical column -> the first accepted name present (None when absent)."""
    return {k: next((c for c in names if c in have), None) for k, names in spec.items()}


def _mid_filter(df, mids):
    m = pd.to_numeric(df['mid'], errors='coerce').to_numpy(float)
    return df[np.isin(m, np.array(sorted(mids), float))].reset_index(drop=True)


# ------------------------------------------------------------------ siblings (v3's rule, the v4 specs) ------------------------------------------------------------------
def sibling_state(rroot, name):
    """a sibling output directory -> status ok (its receipt lists every file present with the same sha256) / absent / incomplete / unbound /
    mismatch / unreadable, with the reason (chains_v3.sibling_state's rule on the v4 SIBLINGS)."""
    sp = SIBLINGS[name]
    d = Path(rroot) / sp['dir']
    st = dict(name=name, dir=_rel(d), status='absent', reason='', receipt=None, files={}, absent_optional=[], notes=[])
    rp = d / 'receipt.json'
    if not rp.is_file():
        st['reason'] = f'{name} receipt absent ({_rel(rp)})'
        return st
    try:
        rec = json.loads(rp.read_text())
    except (OSError, ValueError) as e:
        st.update(status='unreadable', reason=f'{name} receipt unreadable: {type(e).__name__}')
        return st
    st['receipt'] = dict(path=_rel(rp), sha256=CH3._sha(rp))
    opt = sp['optional']
    for k, names in list(sp['files'].items()) + list(opt.items()):
        p = next((d / n for n in names if (d / n).is_file()), None)
        if p is None and k in opt:
            st['absent_optional'].append(k)
            continue
        if p is None:
            st.update(status='incomplete', reason=f'{name}: none of {list(names)} present')
            return st
        want, got = CH3.bound_sha(rec, 'outputs', p.name), CH3._sha(p)
        if want is None:
            st.update(status='unbound', reason=f'{name} receipt does not list {p.name} with a sha256')
            return st
        if want != got:
            st.update(status='mismatch', reason=f'{name}/{p.name} sha256 {got[:12]} differs from its receipt {want[:12]}')
            return st
        st['files'][k] = dict(path=str(p), sha256=got, bytes=p.stat().st_size)
    st['status'] = 'ok'
    return st


def load_sibling(st, mids, read=None):
    """read an ok sibling's files (chain markets only) and check their columns: a required file's mismatch becomes the sibling's status
    (schema_mismatch, the reason naming the columns); an optional file with missing columns is not used (noted). read = the file keys read
    (default: every bound file; a bound file not read is noted)."""
    if st['status'] != 'ok':
        return None
    nm = st['name']
    out = {}
    for k, f in list(st['files'].items()):
        p = Path(f['path'])
        if read is not None and k not in read:
            st['notes'].append(f'{p.name}: bound, not read in this mode')
            continue
        have = _header(p)
        if nm == 'posting_opportunity' and k == 'rows':          # v3's posting_opportunity reader, unchanged
            d0, d1, stc, tk = CH3.po_columns(have)
            req = CH3.SIBLINGS['posting_opportunity']['need']['rows']
            miss = ([c for c in req if c not in have] + ([] if tk else [f'one of {list(CH3.PO_TIME_KEYS)}'])
                    + [f_ for f_ in CH3.FEATURES if d0[f_] is None])
            use = list(dict.fromkeys(list(req) + [c for c in [tk, stc] + list(d0.values()) + list(d1.values()) if c]
                                     + ['day'] * ('day' in have) + [c for c in CH3.PO_OPTIONAL if c in have]))
            out['cols'] = dict(d0=d0, d1=d1, status=stc, time=tk)
        elif nm == 'posting_opportunity':                          # events.csv: the identity layer (the transition link's candidate grade)
            miss = [c for c in PO_EVENTS_NEED if c not in have]
            use = list(dict.fromkeys(list(PO_EVENTS_NEED) + [c for c in PO_EVENTS_SHOW if c in have]))
        elif nm == 'queue_front_v2':
            miss = [c for c in QF2_NEED[k] if c not in have]
            use = list(dict.fromkeys(list(QF2_NEED[k]) + [c for c in QF2_SHOW[k] if c in have]))
        else:                                                      # touch_transition: logical columns through TT_COLS / TT_OPT
            res, opt = _resolve(have, TT_COLS[k]), _resolve(have, TT_OPT.get(k, {}))
            miss = [f'{lk} (one of {list(TT_COLS[k][lk])})' for lk, c in res.items() if c is None]
            out.setdefault('tt_cols', {})[k] = dict(res, **opt)
            use = list(dict.fromkeys([c for c in list(res.values()) + list(opt.values()) if c] + [c for c in TT_SHOW.get(k, ()) if c in have]))
        if miss and k in SIBLINGS[nm]['optional']:
            st['notes'].append(f'{p.name}: required columns missing {miss} (not used)')
            del st['files'][k]
            continue
        if miss:
            st.update(status='schema_mismatch', reason=f'{nm}/{p.name}: required columns missing {miss}')
            return None
        df = _mid_filter(CH3._read_any(p, use), mids)
        if nm == 'touch_transition' and k == 'census':                # integrator: reconcile.csv repeats a frame pair once per creation;
            cc = out['tt_cols']['census']                              # one class per (day, mid, lo, hi) pair -> de-duplicate on the pair + class
            df = df.drop_duplicates(subset=[c for c in (cc['day'], cc['mid'], cc['lo'], cc['hi'], cc['cls'], cc.get('sub')) if c]).reset_index(drop=True)
        out[k] = df
        f['rows_used'] = int(len(df))
    return out


# ------------------------------------------------------------------ finding 3: the graded pre-state ------------------------------------------------------------------
def grade_pre_state(L, g):
    """v3's pre-state record (its window vectors = a conditional window description) graded by the post grade g: unless g is G2 the
    alternative unresolved_outside_window is retained (never ONE-HISTORY), with the hidden action named."""
    need(g in (G0, G1, G2, G3, G4), f"chain {L['chain']}: unknown post grade {g!r} (refused)")
    L.update(post_grade=g, window_basis=CH3.WINDOW_BASIS[g], v3_reading='PENDING')
    if L['status'] == 'pending':
        return L
    win = list(L['alternatives'])
    L['v3_reading'] = 'TIED' if len(win) >= 2 else 'ONE-HISTORY'   # v3's rule on the same vectors (recorded, never rendered)
    outside = g != G2 and _mut() != 'v3_pre_state_rule'             # planted error (selftest only): v3's rule, the grade ignored
    alts = win + ([UNRESOLVED] if outside else [])
    hid = [x for x in (L['hidden_action'],) if x]
    if outside:
        hid.append(f"whether the actual post lies inside its window at all ({OUTSIDE[g]}): the {len(win)} vector(s) describe the state INSIDE the "
                   "window only")
    note = (f'{UNRESOLVED} retained ({OUTSIDE[g]})' if outside else
            ('G2: the window bounds the post (conditional on H1 / H2 / capacity and the maximum match lag)' if g == G2 else
             f'{UNRESOLVED} NOT retained (planted v3 rule)'))
    L.update(alternatives=alts, n_histories=len(alts), hidden_action='; '.join(hid), evidence_grade=f'observed_state:post_window:{g}',
             window_alternatives=win, summary=f"{g}: {L['summary']}" + (f'; + {UNRESOLVED}' if outside else ''),
             lines=[f"  - post grade {g} ({CH3.WINDOW_BASIS[g]}): the vectors below are a CONDITIONAL WINDOW DESCRIPTION (the saved state inside "
                    f"this window), never by themselves an identified posting state; {note}"] + list(L['lines']))
    return L


def link_pre_state(ch, p, t, PO, sib):
    return grade_pre_state(CH3.link_pre_state(ch, p, t, PO, sib), _s(t['grade']))


# ------------------------------------------------------------------ finding 4: the alt-netting state ------------------------------------------------------------------
def alt_raw(r):
    """the saved masked-netting cells of one queue_front_v2 level / candidate row (raw, uninterpreted)."""
    return dict(level_id=_s(r.get('level_id', '')), status=_s(r.get('status', '')),
                alt_netting_possible=(_s(r['alt_netting_possible']) if 'alt_netting_possible' in r else 'absent'),
                timing_basis=_s(r.get('timing_basis', '')), n_earliest_set=_int(r.get('n_earliest_set', NAN), -1),
                alt_sites_life=_int(r.get('alt_sites_life', NAN), -1), alt_sites_unexplained=_int(r.get('alt_sites_unexplained', NAN), -1),
                has_evidence_cols=bool(all(c in r for c in ALT_EVIDENCE)))


def alt_state(raw):
    """possible (saved True) / excluded (saved False whose calculation ran - an earliest record - and whose saved sites are both 0, as
    alt = sites > 0 requires) / not_evaluated (every other value: the initial False of a level without an earliest record, missing,
    inconsistent, v1 labels): an unknown state is never False."""
    b = _b3(raw['alt_netting_possible']) if raw['alt_netting_possible'] != 'absent' else None
    if b is True:
        return 'possible'
    if b is False and _mut() == 'alt_false':                      # planted error (selftest only): v3's reading, a saved False = excluded
        return 'excluded'
    ran = bool(raw['has_evidence_cols'] and raw['timing_basis'] in EVAL_BASES and raw['n_earliest_set'] >= 1)
    if b is False and ran and raw['alt_sites_life'] == 0 and raw['alt_sites_unexplained'] == 0:
        return 'excluded'
    return 'not_evaluated'


def chain_alt_state(states):
    """the chain's masked-netting state over its compatible levels (no level: not_evaluated)."""
    if not states:
        return 'excluded' if _mut() == 'alt_false' else 'not_evaluated'
    if 'possible' in states:
        return 'possible'
    return 'not_evaluated' if 'not_evaluated' in states else 'excluded'


def _raw_true(raw):
    return raw['alt_netting_possible'].strip().lower() in ('true', '1', '1.0')


def _raw_evaluated_false(raw):
    """the verifier's statement on the RAW cells (written apart from alt_state): a False with an earliest record and zero saved sites."""
    return bool(raw['alt_netting_possible'].strip().lower() in ('false', '0', '0.0') and raw['has_evidence_cols']
                and raw['timing_basis'] in ('book_candidate', 'settlement') and raw['n_earliest_set'] > 0
                and raw['alt_sites_life'] == 0 and raw['alt_sites_unexplained'] == 0)


def _raw_unevaluated(raw):
    return bool(not _raw_true(raw) and not _raw_evaluated_false(raw))


def link_queue(ch, p, fills, QF1, QF2, sib, lv):
    """v3's queue link (chains_v3.link_queue) with the three-state masked netting (finding 4); lv = the compatible fresh-level lives."""
    ws = _int(p['ws_ms'])
    v2 = QF2 is not None
    olines, n_orders = CH3.render_orders(fills, ws)
    qc = _s(p['queue_class'])
    pos = CH3.QPOS.get(qc, ('front_at_first_fill', 'behind_at_first_fill'))
    lvt = [f"level {_s(r['level_id'])} ({_s(r['status'])})" for r in lv] or ['no_saved_fresh_level_life']
    marker = '' if v2 else ' [v1 labels, see section 5]'
    raws = [alt_raw(r) for r in lv]
    for x in raws:
        x['state'] = alt_state(x)
    st = chain_alt_state([x['state'] for x in raws])
    if st in ALT_TAG:
        alts = [f'{a} @ {b_} + {ALT_TAG[st]}' for a in pos for b_ in lvt] + [f'{a} @ {b_} + as_saved' for a in pos for b_ in lvt]
    else:
        alts = [f'{a} @ {b_}' for a in pos for b_ in lvt]
    h = ch['order_hash']
    occ = []
    if v2:
        CA = QF2['candidates']
        cm = (_sarr(CA['day']) == ch['day']) & (np.array([_int(x) for x in CA['mid']]) == ch['mid'])
        crow = []
        for r in CA[cm].to_dict('records'):
            es_ = {x.strip().lower() for x in re.split(r'[|;,]', _s(r['earliest_set_hashes'])) if x.strip()}
            ch_ = _s(r['candidate_hash']).lower()
            if h == ch_ or h in es_:
                crow.append(f"front-candidate set at level {_s(r['level_id'])} ({'the candidate hash' if h == ch_ else 'an earliest-set member'}"
                            + ''.join(f"; {c} {_s(r[c])}" for c in CH3.QF2_SHOW['candidates'] if c in r and c != 'alt_netting_possible')
                            + f"; alt_netting {alt_state(alt_raw(r))} (saved {_s(r.get('alt_netting_possible', ''))}))")
        FLv = QF2.get('fills')
        if FLv is not None:
            fmv = (_sarr(FLv['mk_hash']) == h) & (_sarr(FLv['day']) == ch['day']) & (np.array([_int(x) for x in FLv['mid']]) == ch['mid'])
            crow += [f"{_s(r['role'])} at level {_s(r['level_id'])}" + (f" k={_int(r['k'])}" if 'k' in r else '') for r in FLv[fmv].to_dict('records')]
        OC = QF2.get('occupancy')
        if OC is not None:
            om = (_sarr(OC['mk_hash']) == h) & (_sarr(OC['day']) == ch['day']) & (np.array([_int(x) for x in OC['mid']]) == ch['mid'])
            occ = [f"endpoint occupancy {_s(r['occupancy'])}" + (f" at {_rel_s(r['window_lo_ms'], ws)}" if 'window_lo_ms' in r else '')
                   + ''.join(f"; {c} {_s(r[c])}" for c in ('size_at_window_lo', 'post_grade', 'lookup_status') if c in r)
                   + ' (an endpoint observation, never an actual-post queue label)' for r in OC[om].to_dict('records')]
    else:
        FR = QF1['fronts']
        fm = (_sarr(FR['mk_hash']) == h) & (_sarr(FR['day']) == ch['day']) & (np.array([_int(x) for x in FR['mid']]) == ch['mid'])
        crow = []
        for r in FR[fm].to_dict('records'):
            lvl = _s(r['level_id'])
            fh = set(_sarr(FR.loc[(_sarr(FR['level_id']) == lvl) & (np.array([_int(x) for x in FR['mid']]) == ch['mid'])
                                  & (_sarr(FR['role']) == 'front'), 'mk_hash']))
            role = _s(r['role'])
            lab = 'later_same_hash' if (role == 'behind' and h in fh) else ('front_candidate' if role == 'front' else 'later_other')
            crow.append(f"v1 {role} k={_int(r['k'])} at level {lvl} -> {lab}")
    hid = []
    if len(pos) > 1:
        hid.append(f"whether orders were ahead at the first fill (S_FIFO ahead [{_fmt(p['ahead_lo_fifo'])}, {_fmt(p['ahead_hi_fifo'])}] sh at the "
                   f"first fill; [{_fmt(p['ahead_acc_min'])}, {_fmt(p['ahead_acc_max'])}] sh at acceptance)")
    if len(lvt) > 1:
        hid.append('which fresh-level life the first fill matched in (' + ', '.join(lvt) + ')')
    if st == 'possible':
        hid.append('a cancellation / replenishment masked inside a frame interval (net book change; section 5 finding 1)')
    elif st == 'not_evaluated':
        hid.append('a cancellation / replenishment masked inside a frame interval, NOT EVALUATED upstream (queue_front_v2 computes '
                   'alt_netting_possible only for a level with an earliest record; its initial False is not an exclusion): kept')
    lvl_txt = []
    for r, x in zip(lv, raws):
        lvl_txt.append(f"level {_s(r['level_id'])} created {_rel_s(r.get('t_create_ms', r['t_open_ms']), ws)}, status {_s(r['status'])}"
                       + (f"/{_s(r['reason'])}" if 'reason' in r else '')
                       + (f", creation context {_s(r['creation_context'])}" if 'creation_context' in r else
                          (f", mechanism {_s(r['mechanism'])}" if 'mechanism' in r else ''))
                       + f", alt_netting {x['state']} (saved {x['alt_netting_possible']}"
                       + (f"; timing_basis {x['timing_basis'] or 'none'}, earliest set {x['n_earliest_set']}, sites {x['alt_sites_life']}+"
                          f"{x['alt_sites_unexplained']}" if x['has_evidence_cols'] else '; no evaluation columns') + ')'
                       + (f", alt_first_other_address {_s(r['alt_first_other_address'])}" if 'alt_first_other_address' in r else '')
                       + (f", members {_int(r['n_members'])}" if 'n_members' in r else ''))
    lines = olines + [
        f"  - S_FIFO queue class {qc} -> positions {'|'.join(pos)}; ahead at acceptance [{_fmt(p['ahead_acc_min'])}, {_fmt(p['ahead_acc_max'])}] sh, "
        f"at the first fill [{_fmt(p['ahead_lo_fifo'])}, {_fmt(p['ahead_hi_fifo'])}] sh; tied first fills {_int(p['n_tied_first'])}, fronts after a "
        f"partial {_int(p['n_front_after_partial'])}; lo-clock differs {_s(p['lo_clock_queue_differs'])}; S_NOPRIO has no queue (price-time "
        f"premise stated, not tested)",
        f"  - fresh-level lives compatible with the first fill{marker}: " + ('; '.join(lvl_txt) if lvl_txt else 'none'),
        f"  - masked-netting state over these lives: {st} (review 01a72b4e2 finding 4: an unknown / not-evaluated state stays not_evaluated, never False)",
        f"  - {'candidate / fill rows (queue_front_v2)' if v2 else 'v1 fronts.csv rows'} of this hash{marker}: {'; '.join(crow) if crow else 'none'}"]
    if v2:
        lines.append(f"  - endpoint occupancy (queue_front_v2 occupancy.csv): {'; '.join(occ) if occ else ('none saved' if QF2.get('occupancy') is not None else 'file absent')}")
    return CH3.new_link(ch, '3', 'queue', status='filled' if v2 else 'filled_v1_labels',
                        source=('queue_front_v2/levels+candidates' if v2 else 'queue_front/levels.csv+fronts.csv (v1 labels, see section 5)')
                        + '; post_fill/orders.csv; life fills', evidence_grade='scenario_queue_history', alternatives=alts, n_histories=len(alts),
                        hidden_action='; '.join(hid), n_fills=len(fills), n_orders_rendered=n_orders,
                        pending_reason='' if v2 else (sib['reason'] or 'queue_front_v2 not ok'), alt_netting_state=st, alt_levels=raws,
                        summary=f"class {qc}; {len(lv)} compatible level(s){marker}; masked netting {st}", lines=lines)


# ------------------------------------------------------------------ contract 5: the transition / response context ------------------------------------------------------------------
def window_position(lo, hi, a, b):
    """a creation interval (a, b] (integer ms: [a + 1, b]) against the post window [lo, hi] (lo < 0 = unbounded): inside (posting_opportunity
    section 3's containment) / after (a + 1 > hi) / before (b < lo) / straddles (overlapping, not contained)."""
    if (lo < 0 or lo <= a + 1) and b <= hi:
        return 'inside'
    if a + 1 > hi:
        return 'after'
    if lo >= 0 and b < lo:
        return 'before'
    return 'straddles'


def life_span(r, we):
    """a compatible level's shown life [t_create, t_last] (t_last: chains_v3.compatible_levels' rule)."""
    end_ms = _int(r['end_ms'])
    et = _s(r.get('end_type', 'zero' if end_ms >= 0 else ''))
    t_last = min(end_ms, we) if (end_ms >= 0 and et in ('zero', '')) else we
    return _int(r.get('t_create_ms', r['t_open_ms'])), t_last


def _ep_text(e, ws, side):
    st = '; '.join(f'{c} {v}' for c, v in e['show'].items() if v != '')
    return (f"response episode {e['eid']}: {side} retreat at {_rel_s(e['start'], ws)} by {_fmt(e['delta'], 0)} tick(s), response "
            f"({e['kind'] or 'kind not saved'}) after {_fmt(e['exposure'], 0)} ms" + (f"; saved state / owner: {st}" if st else ''))


def link_transition(ch, p, t, lv, TT, EVI, sib_tt, alt_st):
    """contract 5: the post cell's compatible creations matched to the census frame pair / the response episode ending there, with the
    candidate grade (posting_opportunity identity); the retained transition contexts of the order's post."""
    src = ('touch_transition census + episodes; queue_front_v2 levels (creation intervals); posting_opportunity events.csv (identity); '
           'correction_v1/post_regrade_orders.csv (window, grade)')
    g = _s(t['grade'])
    if TT is None:
        L = CH3.pending_link(ch, '1b', 'transition', src, sib_tt['reason'] or f"touch_transition status {sib_tt['status']}")
        L['post_grade'] = g
        return L
    ws, we = _int(p['ws_ms']), _int(p['we_ms'])
    side, tick = _s(p['y_side']), _int(p['tick'])
    lo, hi = _int(t['post_lo_union_ms']), _int(t['post_hi_ms'])
    CE, EP, ccen, cep = TT['census'], TT['episodes'], TT['tt_cols']['census'], TT['tt_cols']['episodes']
    CEm = CE[(_sarr(CE[ccen['day']]) == ch['day']) & (np.array([_int(x) for x in CE[ccen['mid']]], np.int64) == ch['mid'])].reset_index(drop=True)
    c_lo = np.array([_int(x) for x in CEm[ccen['lo']]], np.int64)
    c_hi = np.array([_int(x) for x in CEm[ccen['hi']]], np.int64)
    EPm = EP[(_sarr(EP[cep['day']]) == ch['day']) & (np.array([_int(x) for x in EP[cep['mid']]], np.int64) == ch['mid'])
             & (_sarr(EP[cep['side']]) == side) & (_sarr(EP[cep['end']]) == 'response')].reset_index(drop=True)
    e_hi = np.array([_int(x) for x in EPm[cep['end_hi']]], np.int64)
    e_lo = np.array([_int(x) for x in EPm[cep['end_lo']]], np.int64) if cep.get('end_lo') else None
    if EVI is not None:
        EVm = EVI[(_sarr(EVI['day']) == ch['day']) & (np.array([_int(x) for x in EVI['mid']], np.int64) == ch['mid'])
                  & (_sarr(EVI['side']) == side) & (np.array([_int(x) for x in EVI['tick']], np.int64) == tick)].reset_index(drop=True)
        i_a = np.array([_int(x) for x in EVm['t_open_ms']], np.int64)
        i_b = np.array([_int(x) for x in EVm['t_create_ms']], np.int64)
    cre = []
    for r in lv:
        a, b = _int(r['t_open_ms']), _int(r.get('t_create_ms', r['t_open_ms']))
        c = dict(level_id=_s(r['level_id']), status=_s(r['status']), a=a, b=b, pos=window_position(lo, hi, a, b), life=life_span(r, we),
                 context=_s(r.get('creation_context', r.get('mechanism', ''))))
        k = np.flatnonzero((c_lo == a) & (c_hi == b))
        if len(k) == 1:
            cr = CEm.iloc[int(k[0])]
            sub = _s(cr[ccen['sub']]) if ccen.get('sub') else ''
            c.update(cstat='matched', cls=_s(cr[ccen['cls']]) + (f'/{sub}' if sub else ''))
        else:
            c.update(cstat='no_census_pair' if len(k) == 0 else 'ambiguous_census_pair', cls='')
        ke = np.flatnonzero(e_hi == b)
        if e_lo is not None:
            ke = ke[e_lo[ke] == a]
        c['estat'] = 'response_episode' if len(ke) == 1 else ('no_response_episode' if len(ke) == 0 else 'ambiguous_response_episode')
        c['ep'] = None
        if len(ke) == 1:
            er = EPm.iloc[int(ke[0])]
            c['ep'] = dict(eid=_s(er[cep['eid']]), kind=_s(er[cep['kind']]), start=_int(er[cep['start']]), delta=_fin(er[cep['delta']]),
                           exposure=_fin(er[cep['exposure']]), show={x: _s(er[x]) for x in TT_SHOW['episodes'] if x in EPm.columns})
        if EVI is None:
            c.update(ident='not_saved', ident_reason='posting_opportunity events.csv not bound', cand='', rel='unknown')
        else:
            ki = np.flatnonzero((i_a == a) & (i_b == b))
            if len(ki) == 1:
                er = EVm.iloc[int(ki[0])]
                cand = _s(er['cand_hash']).lower()
                c.update(ident=_s(er['identity']), ident_reason=_s(er['identity_reason']), cand=cand,
                         rel='this_order' if (cand and cand == ch['order_hash']) else ('other_order' if cand else 'no_candidate'))
            else:
                c.update(ident='not_in_events' if len(ki) == 0 else 'ambiguous_identity_rows', cand='', rel='unknown',
                         ident_reason='no posting_opportunity creation row at this cell interval' if len(ki) == 0 else f'{len(ki)} rows')
        cre.append(c)

    def tok(c):
        s = f"at_creation:level_{c['level_id']}:{c['cls'] if c['cstat'] == 'matched' else c['cstat']}"
        if c['ep'] is not None:
            s += f":response_to_{c['ep']['eid']}({c['ep']['kind']})"
        return s + (':straddles_window' if c['pos'] == 'straddles' else '')
    lo_eff = lo if lo >= 0 else -INF
    joined = (not cre) or any(l_lo < hi and l_hi > lo_eff and l_hi > l_lo for (l_lo, l_hi) in (c['life'] for c in cre))
    cand_cre = [c for c in cre if c['pos'] in ('inside', 'straddles')]
    alts = [tok(c) for c in cand_cre]
    if not cre:
        alts.append(NOT_SAVED)
    if joined:
        alts.append(JOINED)
        if alt_st != 'excluded':
            alts.append(MASKED)
    if g != G2:
        alts.append(UNRESOLVED)
    if not alts:
        alts = ['no_compatible_context', NOT_SAVED]
    parts = []
    if len(alts) >= 2:
        if cand_cre:
            parts.append(f"whether the order was posted at a visible creation of its cell ({', '.join('level ' + c['level_id'] for c in cand_cre)}; "
                         'at most a candidate creator - the eventual filler is never stated as the creator)')
        if JOINED in alts:
            parts.append('or added to the already-shown cell (no fresh-creation transition; its frame pair unidentified)')
        if MASKED in alts:
            parts.append(f'or re-created it inside a frame interval (masked netting {alt_st})')
        if NOT_SAVED in alts:
            parts.append("the cell's creation is not among the saved fresh-level lives")
        if UNRESOLVED in alts:
            parts.append(f'or posted outside its window ({OUTSIDE[g]})')
        if not parts:
            parts.append('which of: ' + ', '.join(alts))
    lines = [f"  - post cell {side} {tick}c; post window [{_rel_s(lo, ws) if lo >= 0 else 'unbounded'}, {_rel_s(hi, ws)}] ({g}); {len(cre)} compatible "
             f"fresh-level creation(s) (the QUEUE link's lives), each matched by its creation interval (upd(f-1), upd(f)] to the census frame pair "
             f"and to a response episode ending there on side {side}"]
    for c in cre:
        lines.append(f"  - level {c['level_id']} created in ({_rel_s(c['a'], ws)}, {_rel_s(c['b'], ws)}] (status {c['status']}"
                     + (f", context {c['context']}" if c['context'] else '') + f") - {c['pos']} the post window; census pair: "
                     + (c['cls'] if c['cstat'] == 'matched' else c['cstat']) + '; '
                     + (_ep_text(c['ep'], ws, side) if c['ep'] is not None else c['estat'])
                     + f"; candidate grade {c['ident']} ({c['ident_reason'] or 'no reason saved'}; candidate {_h12(c['cand']) or 'none'}: {c['rel']})"
                     + ('' if c['pos'] in ('inside', 'straddles') else ' [context only: outside the post window]'))
    npos, ncs = Counter(c['pos'] for c in cre), Counter(c['cstat'] for c in cre)
    census = ((f"{len(cre)} compatible creation(s): " + ', '.join(f'{k_} {npos[k_]}' for k_ in POSITIONS if npos[k_])
               + f"; census {dict(sorted(ncs.items()))}; response episodes {sum(c['ep'] is not None for c in cre)}") if cre else 'no compatible creation saved')
    chk = dict(transition_census_pair_unique=bool(all(c['cstat'] != 'ambiguous_census_pair' for c in cre)),
               transition_response_episode_unique=bool(all(c['estat'] != 'ambiguous_response_episode' for c in cre)),
               transition_identity_row_unique=bool(all(c['ident'] != 'ambiguous_identity_rows' for c in cre)),
               transition_candidate_grades_available=bool(EVI is not None))
    cg = '|'.join(sorted({f"{c['ident']}:{c['rel']}" for c in cand_cre}))
    return CH3.new_link(ch, '1b', 'transition', source=src, evidence_grade='transition_context', census=census, alternatives=alts,
                        n_histories=len(alts), hidden_action='; '.join(parts), observed_public=True, n_records=len(cre), post_grade=g,
                        candidate_grades=cg, creations=cre, summary=f"{len(alts)} retained context(s): {', '.join(alts)}", lines=lines, checks=chk)


# ------------------------------------------------------------------ finding 4: gross labels (v3's VALUE builder, this label function) ------------------------------------------------------------------
def _vline_v4(lab, V):
    if _mut() == 'v3_value_label':                                # planted error (selftest only): v3's unlabelled c/sh
        return _V3_VLINE(lab, V)
    H = CH3.HORIZONS
    return (f"  - {lab}: " + '; '.join(f"{h.upper() if h == 'end' else h} valid {V[h]['n_valid']}/{V[h]['n_all']} rows, {_fmt(V[h]['sh_valid'])}/"
                                       f"{_fmt(V[h]['sh_all'])} sh, gross {_sg(V[h]['gross'])} $, net {_sg(V[h]['net'])} $, {_sg(V[h]['c_sh'])} gross c/sh"
                                       + (f" (invalid {V[h]['invalid']})" if V[h]['invalid'] else '') for h in H)
            + f"; common cohort {V['common']['n_valid']} rows: " + ', '.join(f"{h} {_sg(V['common'][f'c_sh_{h}'])}" for h in H)
            + f" gross c/sh; resolution (net_res) valid {V['res']['n_valid']}/{V['res']['n_all']}: {_sg(V['res']['net'])} $ ({_sg(V['res']['c_sh'])} net c/sh)")


def link_value(ch, p, FV, CL, CM, MG):
    """chains_v3.link_value unchanged (every number and check), its view lines rendered by _vline_v4 (the label only)."""
    CH3._vline = _vline_v4
    try:
        return CH3.link_value(ch, p, FV, CL, CM, MG)
    finally:
        CH3._vline = _V3_VLINE


# ------------------------------------------------------------------ rendering / verification ------------------------------------------------------------------
def render_link(L):
    tok = L['reading']
    n = int(L['n_histories'])
    head = (f"- **{L['item']} {TITLE[L['link']]}** — **{tok}** ({n} retained histor{'y' if n == 1 else 'ies'}) · grade `{L['evidence_grade']}` · "
            f"{L['status']}" + (f" · census: {L['census']}" if L['census'] else ''))
    out = [head]
    if tok == 'PENDING':
        out.append(f"  - pending: {L['pending_reason']}")
        return out
    if tok == 'TIED':
        out.append(f"  - retained histories: {' | '.join(L['alternatives'])}")
        out.append(f"  - hidden action that changes the explanation: {L['hidden_action']}")
    elif tok == 'NO-EVENT':
        out.append('  - no take record of this address in the episode window: every retained history agrees')
    else:
        out.append(f"  - every retained history agrees: {L['alternatives'][0]}")
    return out + L['lines']


def _agree_line(a):
    return (f"- **AGREEMENT** histories agree on: {', '.join(a['agree']) or 'none'}; TIED: {', '.join(a['tied']) or 'none'}; pending: "
            f"{', '.join(a['pending']) or 'none'}; first branching link: {a['first_branch'] or 'none'}")


def chains_md(mode, chains, links, sib, asof):
    by = {}
    for L in links:
        by.setdefault(L['chain'], []).append(L)
    out = [f"# Six complete chains v4 (touch_response_contract.md section 5) - {mode.upper()}", '',
           f"As of {asof}. Mode **{mode}**. Selection (predetermined, not outcome-selected): per rank, the first two episodes in "
           "`order_estimator/cases.csv` row order of first appearance = `correction_v1/paths.md` (the same six as v3). Saved outputs only; "
           "descriptive; one consumed day (Oct 24) of four; filled orders are outcome-selected; no identified intent or private action. Times "
           "are venue ms relative to the market window start ws.", '',
           'Corrections (review 01a72b4e2): finding 3 - the PRE-STATE link carries the post grade; its state vectors are a conditional window '
           f'description; G0 / G1 / G3 / G4 windows keep `{UNRESOLVED}` and are never ONE-HISTORY unless the grade is G2; this table is recomputed '
           'from the graded records. Finding 4 - an unknown / not-evaluated queue alt-netting state stays `not_evaluated`, never False; gross '
           'markouts read "gross c/sh". New (contract 5): the 1b TRANSITION link - the post cell matched to a census frame pair / response '
           'episode where its creation interval allows, with the candidate grade (a candidate, never the creator).', '',
           'Readings: **ONE-HISTORY** = every retained history agrees; **TIED** = at least two retained histories change the explanation (the '
           'hidden action is named); **NO-EVENT** = no take record; **PENDING** = a sibling output is absent (draft). Grades: '
           + ', '.join(f'`{g}`' for g in GRADES) + '.', '',
           'Siblings: ' + '; '.join(f"{nm} **{s['status']}**" + (f" ({s['reason']})" if s['reason'] else '') for nm, s in sorted(sib.items()))
           + '. Without touch_transition the TRANSITION links are PENDING; without queue_front_v2 the queue links carry v1 labels; without '
           'posting_opportunity the pre-state links are PENDING.', '',
           '| chain | rank | episode | post grade | ' + ' | '.join(TITLE[lk] for _, lk in LINKS) + ' | first branch |',
           '|---|---|---|---|' + '---|' * len(LINKS) + '---|']
    for ch in chains:
        ls = by[ch['chain']]
        rd = {x['link']: x['reading'] for x in ls}
        pre = next(x for x in ls if x['link'] == 'pre_state')
        if _mut() == 'summary_ungraded':                          # planted error (selftest only): the table from the ungraded v3 pre-state rule
            rd['pre_state'] = pre.get('v3_reading', rd['pre_state'])
        out.append(f"| {ch['chain']} | {ch['rank']} | {ch['episode_id']} | {pre.get('post_grade', '')} | "
                   + ' | '.join(rd.get(lk, '') for _, lk in LINKS) + f" | {CH3.agreement(ls)['first_branch'] or 'none'} |")
    out.append('')
    for ch in chains:
        ls = by[ch['chain']]
        out += [f"## Chain {ch['chain']} - rank {ch['rank']}: episode {ch['episode_id']}", '',
                f"day {ch['day']}, market {ch['mid']}, address {ch['addr']}; case order {_h12(ch['order_hash'])}; cases.csv data row {ch['case_row']}; "
                f"episode window [{_rel_s(ch['t0'], ch['ws'])}, {_rel_s(ch['t1'], ch['ws'])}]", '']
        for L in ls:
            out += render_link(L)
        a = CH3.agreement(ls)
        out.append(_agree_line(a))
        out += [f"  - {k}: {v}" for k, v in a['hidden'].items()]
        out.append('')
    return '\n'.join(out) + '\n'


def parse_table(md):
    """the summary table -> {chain: dict(rank, episode, cells)}; cells = [post grade] + one reading per link + [first branch]."""
    out = {}
    for line in md.splitlines():
        if line.startswith('## '):
            break
        if TABLE_RE.match(line):
            parts = [c.strip() for c in line.strip().strip('|').split('|')]
            out[int(parts[0])] = dict(rank=parts[1], episode=parts[2], cells=parts[3:])
    return out


def parse_agreement(md):
    out, cur = {}, None
    for line in md.splitlines():
        m = CH3.CHAIN_RE.match(line)
        if m:
            cur = int(m['k'])
            continue
        m = AGREE_RE.match(line)
        if m and cur is not None:
            out[cur] = tuple([] if m[g] == 'none' else m[g].split(', ') for g in ('agree', 'tied', 'pend')) + (m['fb'],)
    return out


def summary_mismatches(md, links, facts):
    """the summary table rows and the AGREEMENT lines against the graded per-link records (and the post grade against the regrade row)."""
    by = {}
    for L in links:
        by.setdefault(int(L['chain']), []).append(L)
    tab, agr, v = parse_table(md), parse_agreement(md), []
    if sorted(tab) != sorted(by):
        v.append(f'the summary table lists chains {sorted(tab)}, the graded link records {sorted(by)}')
    for k, ls in sorted(by.items()):
        rd = {x['link']: x['reading'] for x in ls}
        a = CH3.agreement(ls)
        want = [facts[k]['post_grade']] + [rd.get(lk, '') for _, lk in LINKS] + [a['first_branch'] or 'none']
        got = tab.get(k, {}).get('cells')
        if got != want:
            v.append(f'chain {k}: the summary table row {got} differs from the graded link records {want}')
        if agr.get(k) != (a['agree'], a['tied'], a['pending'], a['first_branch'] or 'none'):
            v.append(f'chain {k}: the AGREEMENT line {agr.get(k)} differs from the graded link records')
    return v


def _value_ok(lines):
    txt = '\n'.join(lines)
    return bool('gross c/sh' in txt and not BARE_RE.search(txt))


def verify(links, md, facts):
    """the rendering verifier (contract properties, never the builders' code): v3's rules (a TIED link renders TIED and nothing IDENTIFIED;
    same-hash fills = ONE order; the submitted size never set; every link rendered once) plus section 5 / findings 3-4 on the records and
    the raw saved cells -> violations."""
    v = []
    B = CH3.parse_md(md)
    if '**IDENTIFIED**' in md or re.search(r'identified repost: yes', md):
        v.append('an IDENTIFIED token / identified repost in the rendering')
    for L in links:
        key = (int(L['chain']), TITLE[L['link']])
        b = B.get(key)
        if b is None:
            v.append(f'chain {key[0]} {key[1]}: not rendered')
            continue
        want = CH3.expected_token(L)
        if b['token'] != want or L['reading'] != want or b['n'] != int(L['n_histories']):
            v.append(f"chain {key[0]} {key[1]}: rendered {b['token']} / recorded {L['reading']} with {L['n_histories']} retained histories "
                     f"(declared {want}: a TIED link is never rendered as identified)")
        f = facts[int(L['chain'])]
        g = f['post_grade']
        if L['link'] == 'queue' and L['status'] != 'pending':
            od = [m for m in (CH3.ORDER_RE.match(x) for x in b['lines']) if m]
            if len(od) != f['n_distinct_hashes'] or sum(int(m['k']) for m in od) != f['n_fills'] or int(L['n_orders_rendered']) != f['n_distinct_hashes']:
                v.append(f"chain {key[0]} QUEUE: {len(od)} order line(s) for {f['n_fills']} fill(s) of {f['n_distinct_hashes']} hash(es) "
                         '(repeated same-hash fills must render as one order)')
            raws = L.get('alt_levels', [])
            for x in raws:
                if _raw_unevaluated(x) and x.get('state') != 'not_evaluated':
                    v.append(f"chain {key[0]} QUEUE: level {x['level_id']} (saved alt_netting_possible {x['alt_netting_possible']}, timing_basis "
                             f"{x['timing_basis'] or 'none'}) recorded {x.get('state')} (a not-evaluated alt-netting state read as False)")
            if L.get('alt_netting_state') == 'excluded' and not (raws and all(_raw_evaluated_false(x) for x in raws)):
                v.append(f"chain {key[0]} QUEUE: masked netting excluded without an evaluated False on every compatible level (a not-evaluated "
                         'alt-netting state read as False)')
            if L.get('alt_netting_state') != 'excluded' and not any('masked_netting_' in a for a in L['alternatives']):
                v.append(f"chain {key[0]} QUEUE: the masked-netting history is missing with state {L.get('alt_netting_state')}")
        if L['link'] == 'post' and L['status'] != 'pending':
            sl = [x for x in b['lines'] if x.startswith('  - size:')]
            if (len(sl) != 1 or 'submitted size unknown' not in sl[0] or re.search(r'submitted size [0-9]', sl[0])
                    or np.isfinite(_fin(L['submitted_size']))):
                v.append(f"chain {key[0]} POST: a submitted size is set ({_fmt(L['submitted_size'])} sh; filled lower bound {_fmt(f['filled'])} sh)")
        if L['link'] == 'pre_state' and L['status'] != 'pending':
            if L.get('post_grade') != g or L['evidence_grade'] != f'observed_state:post_window:{g}':
                v.append(f"chain {key[0]} PRE-STATE: the post grade is not carried ({L.get('post_grade')} / {L['evidence_grade']}; the regrade row says {g})")
            if g != G2 and (L['reading'] == 'ONE-HISTORY' or b['token'] == 'ONE-HISTORY' or UNRESOLVED not in L['alternatives']
                            or UNRESOLVED not in '\n'.join(b['lines'])):
                v.append(f"chain {key[0]} PRE-STATE: {b['token']} without {UNRESOLVED} on a {g} window (never ONE-HISTORY unless the grade is G2)")
        if L['link'] == 'transition':
            if f['tt_ok'] == (L['status'] == 'pending'):
                v.append(f"chain {key[0]} TRANSITION: status {L['status']} while touch_transition is {'ok' if f['tt_ok'] else 'not ok'}")
            if L['status'] != 'pending' and g != G2 and UNRESOLVED not in L['alternatives']:
                v.append(f"chain {key[0]} TRANSITION: no {UNRESOLVED} on a {g} window")
        if L['link'] == 'value' and not _value_ok(b['lines']):
            v.append(f"chain {key[0]} VALUE: a gross markout without the 'gross c/sh' label (a bare c/sh after net dollars)")
    return v + summary_mismatches(md, links, facts)


# ------------------------------------------------------------------ claims ------------------------------------------------------------------
def make_claims(mode, chains, links, sel, sel_md, checks, bind, sib, violations, md, facts):
    L = links
    n_t = Counter(x['reading'] for x in L)
    pend = [f"chain {x['chain']} {x['link']}: {x['pending_reason']}" for x in L if x['status'] == 'pending']
    by_link = {lk: dict(Counter(x['reading'] for x in L if x['link'] == lk)) for _, lk in LINKS}
    pre = [x for x in L if x['link'] == 'pre_state' and x['status'] != 'pending']
    qu = [x for x in L if x['link'] == 'queue' and x['status'] != 'pending']
    tr = [x for x in L if x['link'] == 'transition']
    B = CH3.parse_md(md)
    sm = summary_mismatches(md, links, facts)
    fg = {int(x['chain']): facts[int(x['chain'])]['post_grade'] for x in L}
    preds = dict(
        six_chains=bool(len(chains) == 6),
        selection_equals_correction_v1_paths_md=bool(sel == sel_md),
        every_chain_has_every_link=bool(all(sorted(x['link'] for x in L if x['chain'] == c['chain']) == sorted(lk for _, lk in LINKS) for c in chains)),
        every_link_graded=bool(all(x['evidence_grade'] in LINK_GRADES[x['link']] for x in L)),
        every_link_has_a_declared_reading=bool(all(x['reading'] in READINGS for x in L)),
        no_link_asserts_an_identified_private_action=bool(not any(x['identified_private_action'] for x in L)),
        tied_links_never_rendered_identified=bool(all(x['reading'] == 'TIED' for x in L if x['status'] != 'pending' and int(x['n_histories']) >= 2)),
        every_tied_link_names_its_hidden_action=bool(all(x['hidden_action'] for x in L if x['reading'] == 'TIED')),
        submitted_size_never_set=bool(all(not np.isfinite(_fin(x['submitted_size'])) for x in L if x['link'] == 'post')),
        same_hash_fills_render_one_order=bool(all(int(x['n_orders_rendered']) == 1 for x in L if x['link'] == 'queue' and x['status'] != 'pending')),
        pre_state_carries_post_grade=bool(all(x.get('post_grade') == fg[int(x['chain'])] and x['evidence_grade'] == f"observed_state:post_window:{fg[int(x['chain'])]}"
                                              for x in pre)),
        no_one_history_pre_state_unless_g2=bool(all(x['reading'] != 'ONE-HISTORY' or fg[int(x['chain'])] == G2 for x in pre)),
        unresolved_alternative_unless_g2=bool(all(UNRESOLVED in x['alternatives'] for x in pre + [y for y in tr if y['status'] != 'pending']
                                                  if fg[int(x['chain'])] != G2)),
        no_false_alt_netting_on_not_evaluated_row=bool(all(y.get('state') == 'not_evaluated' for x in qu for y in x.get('alt_levels', []) if _raw_unevaluated(y))),
        masked_netting_kept_unless_evaluated_false=bool(all(x.get('alt_netting_state') != 'excluded'
                                                            or (x.get('alt_levels') and all(_raw_evaluated_false(y) for y in x['alt_levels'])) for x in qu)),
        summary_table_equals_graded_records=bool(not [m for m in sm if 'summary table' in m] and len(parse_table(md)) == len(chains)),
        agreement_lines_equal_graded_records=bool(not [m for m in sm if 'AGREEMENT line' in m]),
        gross_markouts_labelled=bool(all(_value_ok(B.get((int(x['chain']), 'VALUE'), {}).get('lines', [])) for x in L if x['link'] == 'value')),
        transition_pending_iff_sibling_absent=bool(all((x['status'] == 'pending') != facts[int(x['chain'])]['tt_ok'] for x in tr)),
        rendering_verifier_clean=bool(not violations),
        no_pending_link=bool(not pend))
    claims = dict(
        version=VERSION, mode=mode, contract=f'touch_response_contract.md section 5 ({CONTRACT_COMMIT})', review=REVIEW,
        population=dict(unit='six predetermined episodes (correction_v1 paths.md selection), one row per link', n_chains=len(chains), n_links=len(L),
                        days=sorted({c['day'] for c in chains}),
                        episodes=[dict(chain=c['chain'], rank=c['rank'], episode_id=c['episode_id']) for c in chains]),
        predicates=preds,
        readings=dict(n_by_reading=dict(n_t), by_link=by_link, by_grade=dict(Counter(x['evidence_grade'] for x in L)),
                      pre_state_by_post_grade={g: dict(Counter(x['reading'] for x in pre if fg[int(x['chain'])] == g)) for g in (G0, G1, G2, G3, G4)}),
        summary_table={str(k): v for k, v in parse_table(md).items()},
        agreement={str(c['chain']): CH3.agreement([x for x in L if x['chain'] == c['chain']]) for c in chains},
        pre_state={str(x['chain']): dict(post_grade=x.get('post_grade'), reading=x['reading'], v3_reading=x.get('v3_reading'), n_histories=int(x['n_histories']),
                                         unresolved_outside_window=bool(UNRESOLVED in x['alternatives']), status=x['status']) for x in L if x['link'] == 'pre_state'},
        queue_alt_netting={str(x['chain']): dict(state=x.get('alt_netting_state'), reading=x['reading'], n_histories=int(x['n_histories']),
                                                 levels=[dict(level_id=y['level_id'], status=y['status'], saved=y['alt_netting_possible'],
                                                              timing_basis=y['timing_basis'], n_earliest_set=y['n_earliest_set'],
                                                              sites=[y['alt_sites_life'], y['alt_sites_unexplained']], state=y['state'])
                                                         for y in x.get('alt_levels', [])]) for x in qu},
        transition={str(x['chain']): dict(status=x['status'], reading=x['reading'], n_histories=int(x['n_histories']), census=x['census'],
                                          alternatives=list(x['alternatives']), candidate_grades=x.get('candidate_grades', ''),
                                          pending_reason=x['pending_reason']) for x in tr},
        pending=pend, siblings={k: dict(status=v['status'], reason=v['reason']) for k, v in sib.items()},
        queue_labels={str(x['chain']): x['status'] for x in L if x['link'] == 'queue'},
        reconciliation={k: v for k, v in checks.items()},
        reconciliation_all_hold=bool(all(v['holds'] for v in checks.values())),
        bindings={k: v['binding']['status'] for k, v in bind.items()},
        bindings_all_ok=bool(all(v['binding']['status'] == 'ok' for v in bind.values())),
        not_measured='no model skill, policy, intent, validation or profitability: an accounting of saved public records and labels')
    return claims


# ------------------------------------------------------------------ build / run ------------------------------------------------------------------
def _link_row(L):
    d = {}
    for c in LINK_COLS:
        if c == 'alternatives':
            d[c] = '|'.join(L['alternatives'])
        elif c == 'alt_level_states':
            d[c] = ';'.join(f"{x['level_id']}:{x['state']}" for x in L.get('alt_levels', []))
        else:
            d[c] = L.get(c, '')
    return d


def run(mode='draft', rroot=None, droot=None, out=None, require_freeze=True, selftest=False, log=print):
    t0, asof0 = time.time(), CH3._now()
    c_start = time.process_time()
    stages, last = {}, [c_start]

    def stage(nm):
        c = time.process_time()
        stages[nm] = round(c - last[0], 4)
        last[0] = c
        CH3.guard(nm)
        cpu_guard(nm, c_start)                                    # this run's CPU (the selftest runs many in one process)
    need(mode in ('draft', 'run'), f'unknown mode {mode!r}')
    need(C.RUN == 'v2', f'MT_RUN must be v2 (is {C.RUN!r})')
    need(not CH3._mut(), f'MT_CH3_MUT={CH3._mut()!r} is set: chains_v4 never runs a v3 planted mutant (refused)')
    need(selftest or not _mut(), f'MT_CH4_MUT={_mut()!r} is set: a planted mutant never runs outside the selftest (refused)')
    need(_mut() in ('',) + MUTANTS, f'unknown MT_CH4_MUT {_mut()!r} (refused)')
    rroot = Path(rroot) if rroot else C.RECEIPT
    droot = Path(droot) if droot else C.OUT
    out = Path(out) if out else rroot / 'chains' / ('chains_v4' if mode == 'run' else 'chains_v4_draft')
    need(not out.exists(), f'write-once: {out} exists (refused before any input is read)')
    sib = {nm: sibling_state(rroot, nm) for nm in SIBLINGS}
    if mode == 'run':
        bad = {nm: s['reason'] for nm, s in sib.items() if s['status'] != 'ok'}
        need(not bad, f'run requires the sibling outputs bound by their receipts: {bad} (refused; use run --draft)')
    git = CH3._git(FREEZE_FILES)
    if mode == 'run' and require_freeze:
        need(bool(git.get('unmodified')), 'freeze: the tool, its selftest, the contract and chains_v3.py / chains_v3_selftest.py must be tracked and '
                                          'unmodified before a run (refused)')
    stage('preflight')
    IN, cnt = CH3.Inputs(rroot), Counter()

    def path(k):
        root, rel, _ = CH3.INPUTS[k]
        return (rroot if root == 'R' else droot) / rel
    D = {}
    for k in ('cases', 'episode_summary'):
        D[k] = CH3.read_csv(path(k), k, CH3.NEED[k])
        IN.add(k, path(k), len(D[k]), CH3.INPUTS[k][2])
    pm = path('paths_md')
    need(pm.is_file(), f'input paths_md is missing: {_rel(pm)} (refused)')
    pmd = pm.read_text()
    IN.add('paths_md', pm, pmd.count('\n'), CH3.INPUTS['paths_md'][2])
    sel, sel_md = CH3.select_chains(D['cases']), CH3.paths_md_selection(pmd)
    need(len(sel) == 6, f'the predetermined selection has {len(sel)} episodes, not 6 (refused)')
    chains = CH3.chain_bases(sel, D['cases'], D['episode_summary'])
    days = sorted({c['day'] for c in chains})
    need(all(d in C.MEASURE_DAYS for d in days), f'chain days {days} are not all measurement days (refused before any other path is built)')
    mids = sorted({c['mid'] for c in chains})
    addrs = sorted({c['addr'] for c in chains})
    hashes = {(c['day'], c['mid'], c['order_hash']) for c in chains}
    for k in ('order_ledger', 'episodes', 'post_regrade_orders', 'episodes_post_regraded', 'census_graded', 'post_orders', 'pulls', 'qf1_levels',
              'qf1_fronts', 'matched_gap'):
        D[k] = CH3.read_csv(path(k), k, CH3.NEED[k])
        IN.add(k, path(k), len(D[k]), CH3.INPUTS[k][2])
    for k, mc in (('qf1_levels', 'mid'), ('qf1_fronts', 'mid'), ('matched_gap', 'm')):    # the chain markets only (rows counted in the receipt)
        D[k] = D[k][np.isin(np.array([_int(x) for x in D[k][mc]], np.int64), np.array(mids, np.int64))].reset_index(drop=True)
    for k in ('post_regrade_orders', 'post_orders'):
        D[k]['_rank'] = [_int(x) for x in D[k]['rank']]
    stage('csv_inputs')
    D['events'] = CH3.read_pq(path('events'), 'events', CH3.EV_READ)
    IN.add('events', path('events'), len(D['events']), CH3.INPUTS['events'][2])
    D['events'] = D['events'][np.isin(np.array([_int(x) for x in D['events']['mid']], np.int64), np.array(mids, np.int64))].reset_index(drop=True)
    D['events']['addr_l'] = [str(a).lower() for a in D['events']['addr']]
    D['closes_markets'] = CH3.read_pq(path('closes_markets'), 'closes_markets', CH3.CM_READ)
    D['closes_markets']['addr_l'] = [str(a).lower() for a in D['closes_markets']['addr']]
    IN.add('closes_markets', path('closes_markets'), len(D['closes_markets']), CH3.INPUTS['closes_markets'][2])
    need(len(days) == 1, f'chain days {days}: the closes reader is declared for one day (refused)')
    D['closes'] = CH3.read_groups(path('closes'), 'closes', CH3.CL_READ, mids, addrs, cnt, day=days[0])
    IN.add('closes', path('closes'), len(D['closes']), CH3.INPUTS['closes'][2])
    stage('closes')
    D['fv'] = CH3.read_groups(path('fills_valued'), 'fills_valued', CH3.FV_READ, mids, addrs, cnt)
    IN.add('fills_valued', path('fills_valued'), len(D['fv']), CH3.INPUTS['fills_valued'][2])
    stage('fills_valued')
    life = {}
    for d, m in sorted({(c['day'], c['mid']) for c in chains}):
        lp = droot / 'life' / f'{d}_{m}_fills.parquet'
        t = CH3.read_pq(lp, f'life/{d}_{m}', CH3.LIFE_READ)
        hs = {h for (dd, mm, h) in hashes if dd == d and mm == m}
        life[(d, m)] = t[np.isin(np.array([_s(x).lower() for x in t['mk_hash']], object), sorted(hs))].copy()
        IN.add(f'life/{d}_{m}_fills.parquet', lp, len(life[(d, m)]), CH3.LIFE_BIND + (f'life/{d}_{m}_fills.parquet',))
    TT = load_sibling(sib['touch_transition'], mids)
    PO = load_sibling(sib['posting_opportunity'], mids, read=('rows', 'events') if TT is not None else ('rows',))
    QF2 = load_sibling(sib['queue_front_v2'], mids)
    EVI = PO.get('events') if PO is not None else None
    if mode == 'run':
        bad = {nm: s['reason'] for nm, s in sib.items() if s['status'] != 'ok'}
        need(not bad, f'run requires the sibling outputs with their declared columns: {bad} (refused)')
    stage('small_inputs')
    # ---- indices (v3's) ----
    P, T = D['post_orders'], D['post_regrade_orders']
    for X in (P, T):
        X['_key'] = list(zip(_sarr(X['day']), [_int(x) for x in X['mid']], [_s(h).lower() for h in X['mk_hash']]))
    pmap = {k: r for k, r in zip(P['_key'], P.to_dict('records')) if k in hashes}
    tmap = {k: r for k, r in zip(T['_key'], T.to_dict('records')) if k in hashes}
    led, ep, epr = {}, {}, {}
    cks = {c['order_case']['case_key'] for c in chains} | {_s(t_['case_key']) for c in chains for t_ in c['take_cases']}
    for src, dst in ((D['order_ledger'], led), (D['episodes'], ep), (D['episodes_post_regraded'], epr)):
        for r in src[src['case_key'].astype(str).isin(cks)].to_dict('records'):
            dst.setdefault(_s(r['case_key']), []).append(r)
    PU = D['pulls']
    pu = {}
    for r in PU.to_dict('records'):
        k = (_s(r['day']), _int(r['mid']), _s(r['mk_hash']).lower())
        if k in hashes:
            need(_s(r['view']) not in pu.get(k, {}), f'pulls.csv: a (order, view) row repeats for {k} (refused)')
            pu.setdefault(k, {})[_s(r['view'])] = r
    QF1 = dict(levels=D['qf1_levels'], fronts=D['qf1_fronts'])
    # ---- links ----
    links, facts, checks = [], {}, {}
    for ch in chains:
        k = (ch['day'], ch['mid'], ch['order_hash'])
        need(k in pmap and k in tmap, f"chain {ch['chain']}: order {_h12(k[2])} is not in post_fill / post_regrade_orders (refused)")
        p, t = pmap[k], tmap[k]
        ch['ws'] = _int(p['ws_ms'])
        oc = ch['order_case']
        lf = life[(ch['day'], ch['mid'])]
        lf = lf[np.array([_s(x).lower() for x in lf['mk_hash']], object) == ch['order_hash']]
        fills = sorted((dict(hash=_s(r['mk_hash']).lower(), sid=_s(r['sid']), mk_idx=_int(r['mk_idx']), shares=_fin(r['shares']), hi=_int(r['hi_ms']),
                             xlo=_int(r['x_lo_hard_ms']), status=_s(r['status'])) for r in lf.to_dict('records')),
                       key=lambda f: (f['hi'], f['xlo'], f['sid'], f['mk_idx']))
        facts[ch['chain']] = dict(n_fills=len(fills), n_distinct_hashes=len({f['hash'] for f in fills}), filled=_fin(p['cum_shares']),
                                  post_grade=_s(t['grade']), tt_ok=bool(TT is not None))
        fv = D['fv']
        cross = fv[(np.array([_int(x) for x in fv['mid']]) == ch['mid']) & (fv['addr_l'].to_numpy(object) == ch['addr'])
                   & (_sarr(fv['role']) == 'taker') & (_sarr(fv['hash']) == ch['order_hash'])]
        led_rows, ck_ = led.get(oc['case_key'], []), oc['case_key']
        ep_rows = ep.get(ck_, []) + [x for t_ in ch['take_cases'] for x in ep.get(_s(t_['case_key']), [])]
        LV = QF2['levels'] if QF2 is not None else QF1['levels']
        f1 = fills[0] if fills else None
        lv = CH3.compatible_levels(LV, ch, _s(p['y_side']), _int(p['tick']), f1, _int(p['we_ms'])) if f1 else []
        Lq = link_queue(ch, p, fills, QF1, QF2, sib['queue_front_v2'], lv)
        ls = [link_pre_state(ch, p, t, PO, sib['posting_opportunity']),
              link_transition(ch, p, t, lv, TT, EVI, sib['touch_transition'], Lq['alt_netting_state']),
              CH3.link_post(ch, p, t, led_rows, epr.get(ck_, []), D['census_graded'], T, (int(len(cross)), float(cross['q'].to_numpy(float).sum()))),
              Lq,
              CH3.link_ending(ch, p, led_rows, pu.get(k, {}), ep_rows, D['census_graded'], PU),
              CH3.link_repost(ch, p, led_rows, D['census_graded'], P),
              CH3.link_takes(ch, D['events'], ep_rows, ch['ws']),
              link_value(ch, p, D['fv'], D['closes'], D['closes_markets'], D['matched_gap'])]
        fc = dict(life_fills_count_equals_post_fill=bool(len(fills) == _int(p['n_fills'])),
                  life_fills_shares_equal_post_fill=bool(CH3._close(sum(f['shares'] for f in fills), p['cum_shares'])),
                  life_fills_at_the_order_cell=bool((_sarr(lf['y_side']) == _s(p['y_side'])).all()
                                                    and (np.array([_int(x) for x in lf['y_ptick']]) == _int(p['tick'])).all()))
        for L in ls:
            L['reading'] = CH3.expected_token(L)
            L['histories_agree'] = bool(L['status'] != 'pending' and int(L['n_histories']) == 1)
            for kk, vv in L['checks'].items():
                checks.setdefault(kk, []).append(vv)
        for kk, vv in fc.items():
            checks.setdefault(kk, []).append(vv)
        links += ls
    stage('links')
    checks = {k: _pred(v) for k, v in sorted(checks.items())}
    asof = CH3._now()
    md = chains_md(mode, chains, links, sib, asof)
    violations = verify(links, md, facts)
    need(not violations, f'rendering check failed (refused before any write): {violations[:4]}')
    claims = make_claims(mode, chains, links, sel, sel_md, checks, IN.info, sib, violations, md, facts)
    if mode == 'run':
        failed = [k for k, v in claims['predicates'].items() if not v] + ([] if claims['reconciliation_all_hold'] else ['reconciliation'])
        failed += [] if claims['bindings_all_ok'] else ['bindings']
        need(not failed, f'a run refuses on failed predicates {failed} (refused before any write)')
    LK = pd.DataFrame([_link_row(L) for L in links], columns=list(LINK_COLS))
    files = [('chains.md', md.encode()), ('chain_links.csv', LK.to_csv(index=False).encode()), ('claims.json', CH3._json_bytes(claims))]
    stage('render')
    srcs = {_rel(f): (CH3._sha(f) if Path(f).is_file() else None) for f in TOOL_FILES + DEP_FILES}
    sib_rec = {nm: dict(status=s['status'], reason=s['reason'], receipt=s['receipt'], absent_optional=s.get('absent_optional', []),
                        notes=s.get('notes', []),
                        files={k: dict(path=_rel(f['path']), sha256=f['sha256'], bytes=f['bytes'], rows_used=f.get('rows_used'))
                               for k, f in s['files'].items()}) for nm, s in sib.items()}
    rec = dict(version=VERSION, mode=mode, contract=dict(file=_rel(CONTRACT_FILE), commit=CONTRACT_COMMIT,
                                                         sha256=CH3._sha(CONTRACT_FILE) if CONTRACT_FILE.is_file() else None),
               review=REVIEW, module=_rel(Path(__file__)), base_module=_rel(Path(CH3.__file__)), run=C.RUN, asof_start=asof0, asof_end=CH3._now(),
               argv=list(sys.argv), git=git, sources=srcs, inputs=IN.info, siblings=sib_rec, counts=dict(cnt), out=_rel(out),
               outputs={nm: dict(sha256=hashlib.sha256(b).hexdigest(), bytes=len(b)) for nm, b in files},
               rows=dict(chain_links=len(LK), chains=len(chains)),
               interfaces=dict(touch_transition=dict(files={k: list(v) for k, v in TT_FILES.items()}, columns=TT_COLS, optional=TT_OPT, show=TT_SHOW,
                                                     resolved=(TT or {}).get('tt_cols')),
                               posting_opportunity_events=dict(need=list(PO_EVENTS_NEED), show=list(PO_EVENTS_SHOW), read=bool(EVI is not None)),
                               queue_front_v2_alt_evidence=list(ALT_EVIDENCE)),
               constants=dict(LINKS=[list(x) for x in LINKS], GRADES=list(GRADES), LINK_GRADES={k: list(v) for k, v in LINK_GRADES.items()},
                              ALT_STATES=list(ALT_STATES), EVAL_BASES=list(EVAL_BASES), UNRESOLVED=UNRESOLVED, OUTSIDE=OUTSIDE,
                              HISTORIES=[JOINED, MASKED, NOT_SAVED], POSITIONS=list(POSITIONS), READINGS=list(READINGS), CPU_CAP_S=CPU_CAP_S),
               cpu_s=dict(by_stage=stages, process_total=round(time.process_time(), 3), cap=CPU_CAP_S), wall_s=round(time.time() - t0, 2),
               vmhwm_mib=round(CH3.hwm(), 1), guard_mib=C.GUARD_MIB, mutant=_mut() or None)
    CH3.write_out(out, files + [('receipt.json', CH3._json_bytes(rec))])
    log(json.dumps(CH3._clean(dict(out=str(out), mode=mode, readings=claims['readings']['n_by_reading'], by_link=claims['readings']['by_link'],
                                   predicates=claims['predicates'], reconciliation_all_hold=claims['reconciliation_all_hold'],
                                   failed_reconciliation=[k for k, v in claims['reconciliation'].items() if not v['holds']],
                                   bindings=claims['bindings'], siblings=claims['siblings'], pending=len(claims['pending']),
                                   pre_state=claims['pre_state'], queue_alt_netting={k: v['state'] for k, v in claims['queue_alt_netting'].items()},
                                   cpu_s=rec['cpu_s'], vmhwm_mib=rec['vmhwm_mib'], counts=rec['counts'])), indent=1, sort_keys=True))
    return dict(out=out, claims=claims, receipt=rec, links=links, md=md)


if __name__ == '__main__':
    a = sys.argv[1:]
    if a == ['run', '--draft']:
        run(mode='draft')
    elif a == ['run']:
        run(mode='run')
    else:
        raise SystemExit('usage: MT_RUN=v2 chains_v4.py run --draft | run')
