"""competing_transition.py - COMPETING-TRANSITION-AND-CONTINUATION v1, sections 1-4 (P-2026-005 MAKER-TRACE; session aea5bfb9 builder,
2026-11-23). Built EXACTLY to predictfun/reviews/maker_trace_20261122/competing_transition_contract.md sections 1-4 and their acceptance items
(frozen before computation, commit 9909beacc) = the ONE next priority of review a91f6bb52 (reviews/automatic/20261123_touch_response_review/
report.md, findings 1-4; next-priority items 1-4 and 6). Section 5 (candidate continuation economics) is tools/continuation_econ.py, not here.
SAVED EVIDENCE ONLY: v2/chains/touch_transition/episodes.parquet (the 3,796 retreat episodes of the 19 BOOK markets, Oct 24-27) and, for the
two-side sub-labels only, v2/chains/touch_transition/reconcile.csv; each bound to the sha256 and row count that directory's tracked receipt.json
records. No tape, no market reconstruction, no other data path. DESCRIPTIVE on four related consumed days: point estimates only - no p-value,
no permutation, no significance verdict, no interval, no fitted or validated rule, no model skill, no profitability.
S1 OUTCOME of every episode (all rows kept): the next detected transition, mutually exclusive, from the saved end_type / end_reason / end_class
  / resp_type: same_restoring / same_partial / same_overshoot (end_type response) | other_improve / other_retreat (end_reason other_side; the
  end class improve_ / retreat_ of the OTHER side) | two_side_both_up / _both_down / _narrow / _widen from the reconcile.csv rows of the SAME
  end frame pair (day, mid, end_frame; each such row in the census population, class two_side, one sub-label, times = the episode's end pair
  (t_start + exposure_lo, t_end]; anything else refused), two_side_sub_unknown when no saved row exists | same_retreat_further
  (same_side_retreat) | horizon_no_change (no_response_1s) | censored_<reason> for end_type censored_gap (gap / unknown_touch / empty_side /
  crossed / window_end / book_end). A competing move is an outcome, never a discarded non-response. RECONCILIATION (refused on failure):
  episode totals = the bound population (3,796 in the run), mutual exclusion (exactly one declared outcome per row), end-type identities, and
  the class-direction identities of every row (retreat direction, delta ticks and bin; a response is improve_<side> with its resp_type and its
  price vs best_prev / best_retreat, its times; other-side classes of the other side; a further retreat of the same side; horizon / censor
  classes; a non-response carries no response field; a known two-side sub-label only with its saved row).
S2 STATE the five frozen contrasts from the saved D0 / LAG label columns, exactly touch_transition.contrast_code's bins: btc_with ('with' vs
  'flat'), btc_against ('against' vs 'flat'), fair_gap ('(1,3]' + '>3' vs '<=0'), depth_own ('0' + '(0,50]' vs '>200'), time_left
  ('(10,30]' + '[0,10]' vs '(120,240]' + '(60,120]'); state_status ok required at that clock (LAG lag_missing is out). All five reported,
  none selected. Per contrast x clock: the outcome fractions (every class and group) of each group over the whole group and over its
  common-support strata; the same-side-response risk difference within strata market (day, mid) x side x delta bin x time-left bin of the
  same clock (the time_left contrast omits its own bin), pooled over the common-support strata (both groups present) with fixed harmonic
  weights w = n1 n0 / (n1 + n0), risk = same-side responses / episodes; the raw all-population counts and risks beside it; the per-day and
  per-side pooled RDs and signs; market contributions c_m = sum_{k in m} w_k RD_k / W (sum c_m = RD) and the largest one's share. Censor
  bounds: lower = responses / all (censored = non-response), upper = (responses + censored) / all; the RD bounds pair them (sum w (p1_lo -
  p0_hi) / W .. sum w (p1_hi - p0_lo) / W).
S3 SUPPORT (the frozen count floor; a reporting-adequacy floor, not power) per clock within the mixed strata: each group >= 30 episodes,
  >= 10 same-side responses and >= 10 competing non-response transitions (other side / two side / further retreat), and >= 5 markets in each
  of which both groups hold >= 5 episodes. Every count shown (state missing, other bins, single-group strata). READING per contrast: thin
  unless adequate at BOTH D0 and LAG; adequate_direction_consistent if the D0 and LAG pooled RDs share a nonzero sign and the day RD carries
  that sign on >= 3 of the 4 days at both clocks; else adequate_inconsistent. timing_sensitive = the D0 / LAG signs differ (descriptive).
S4 TIMING (interval bounds, never a latency estimate) per response: retreat interval (t_prev_ms, t_start_ms], response interval (resp_lo_ms,
  resp_hi_ms]: lower = max(0, resp_lo - t_start), upper = resp_hi - t_prev; the next-frame flag (saved no_frame_between) and the
  observer-clock endpoint difference resp_hi - t_start separately. Quantiles summarise bounds; they are not an identified latency.
READINGS where the contract is silent (conservative; READINGS_DECLARED, recorded in the receipt).
OUTPUTS write-once v2/chains/competing_transition/ (a partial directory renamed at the end; refused if it exists): outcomes.csv, fractions.csv,
  rd.csv, support.csv, timing.csv, claims.json (computed predicates / readings only), receipt.json (sha256 of sources, inputs and outputs;
  CPU by stage; VmHWM). run --dry writes nothing and prints the reconciliation and the readings.
GUARDS MT_RUN=v2; MT_CT_MUT (the selftest's planted errors) refused in production before any input is read; the producing receipt a tracked,
  unmodified touch_transition_v1 v2 full run whose frozen definitions equal this tool's; every input bound to its sha256 and row count; the
  full run refuses unless this tool, its selftest and the contract are tracked and unmodified (the freeze); the measurement days only;
  VmHWM self-guard 1,450 MiB (pf_mt_common.GUARD_MIB, never raised); CPU self-guard.
Usage (run_light, MT_RUN=v2): competing_transition.py run [--dry]      selftest: competing_transition_selftest.py"""
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

sys.dont_write_bytecode = True

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
RESEARCH = HERE.parents[2] / 'research'
if str(RESEARCH) not in sys.path:
    sys.path.insert(0, str(RESEARCH))
import pf_mt_common as C  # noqa: E402

need = C.need
NAN = float('nan')
VERSION = 'competing_transition_v1'
CONTRACT_FILE = C.RECEIPT_V1 / 'competing_transition_contract.md'
CONTRACT_COMMIT = '9909beacc'
REVIEW_FILE = C.REPO / 'predictfun' / 'reviews' / 'automatic' / '20261123_touch_response_review' / 'report.md'
REVIEW_COMMIT = 'a91f6bb52'
TOOL_FILES = (HERE / 'competing_transition.py', HERE / 'competing_transition_selftest.py', CONTRACT_FILE)
DEP_FILES = (RESEARCH / 'pf_mt_common.py', REVIEW_FILE)
SRC_VERSION = 'touch_transition_v1'
SRC_FILES = ('episodes.parquet', 'reconcile.csv')
N_EP_CONTRACT, N_MARKETS = 3796, 19
CPU_CAP_S = 10.

# ------------------------------------------------------------------ declared constants (the contract's; none fitted) ------------------------------------------------------------------
SIDES = ('bid', 'ask')
DELTA_LAB = ('1', '2', '>=3')
VARIANTS = ('D0', 'LAG')
CLASSES = ('unchanged', 'retreat_bid', 'retreat_ask', 'improve_bid', 'improve_ask', 'two_side', 'gap', 'unknown')
END_TYPES = ('response', 'censored_other', 'censored_gap', 'no_response_1s')
END_REASONS = ('response', 'other_side', 'two_side', 'same_side_retreat', 'gap', 'unknown_touch', 'empty_side', 'crossed', 'book_end', 'window_end',
               'horizon')
RT_LAB = ('restoring', 'partial', 'overshoot')
TWO_SUB = ('both_up', 'both_down', 'narrow', 'widen')
UNK_SUB = ('unknown_touch', 'empty_side', 'crossed')
CENSOR_REASONS = ('gap',) + UNK_SUB + ('window_end', 'book_end')
CENSOR_CLASS = dict(gap=('gap', ''), unknown_touch=('unknown',), empty_side=('unknown',), crossed=('unknown',), window_end=('',), book_end=('',))
O_SAME = tuple('same_' + r for r in RT_LAB)
O_OTHER = ('other_improve', 'other_retreat')
O_TWO = tuple('two_side_' + s for s in TWO_SUB) + ('two_side_sub_unknown',)
O_FURTHER = ('same_retreat_further',)
O_HORIZON = ('horizon_no_change',)
O_CENS = tuple('censored_' + r for r in CENSOR_REASONS)
OUTCOME_GROUPS = (('same_response', O_SAME), ('other_side', O_OTHER), ('two_side', O_TWO), ('further_retreat', O_FURTHER),
                  ('horizon', O_HORIZON), ('censored', O_CENS))
OUTCOMES = tuple(o for _, os_ in OUTCOME_GROUPS for o in os_)
OGROUPS = tuple(g for g, _ in OUTCOME_GROUPS)
GROUP_OF = {o: g for g, os_ in OUTCOME_GROUPS for o in os_}
COMPETING = ('other_side', 'two_side', 'further_retreat')          # the competing non-response transitions of the support floor
GROUP_END_TYPE = dict(same_response='response', other_side='censored_other', two_side='censored_other', further_retreat='censored_other',
                      horizon='no_response_1s', censored='censored_gap')
GROUP_END_REASON = dict(same_response=('response',), other_side=('other_side',), two_side=('two_side',), further_retreat=('same_side_retreat',),
                        horizon=('horizon',), censored=CENSOR_REASONS)
BTC_LAB = ('with', 'against', 'flat', 'missing')                  # posting_opportunity.BTC_LAB
FG_LAB = ('<=0', '(0,1]', '(1,3]', '>3', 'missing')                # posting_opportunity.FG_LAB
DEPTH_LAB = ('0', '(0,50]', '(50,200]', '>200', 'missing')          # posting_opportunity.DEPTH_LAB
TL_LAB = ('>300', '(240,300]', '(120,240]', '(60,120]', '(30,60]', '(10,30]', '[0,10]')   # take_logic.TL_LAB
TL_IDX = {lab: i for i, lab in enumerate(TL_LAB)}
STATUS_LAB = ('ok', 'lag_missing')
STATE_COL = dict(btc='btc_5s_aligned', fg='fair_gap_bin', depth='depth_own_bin', tl='time_left_bin')
STATE_LABS = dict(btc=BTC_LAB, fg=FG_LAB, depth=DEPTH_LAB, tl=TL_LAB + ('',))
# (name, exposed description, reference description, state variable, exposed labels, reference labels) = touch_transition.contrast_code
CONTRASTS = (('btc_with', 'btc_5s with', 'btc_5s flat', 'btc', ('with',), ('flat',)),
             ('btc_against', 'btc_5s against', 'btc_5s flat', 'btc', ('against',), ('flat',)),
             ('fair_gap', 'fair_gap > 1 c', 'fair_gap <= 0 c', 'fg', ('(1,3]', '>3'), ('<=0',)),
             ('depth_own', 'own touch depth <= 50 sh', 'own touch depth > 200 sh', 'depth', ('0', '(0,50]'), ('>200',)),
             ('time_left', 'time left <= 30 s', 'time left in (60,240] s', 'tl', ('(10,30]', '[0,10]'), ('(120,240]', '(60,120]')))
CN_NAMES = tuple(c[0] for c in CONTRASTS)
CSPEC = {c[0]: c for c in CONTRASTS}
MIN_GROUP_EP, MIN_RESP, MIN_COMP, MIN_MKT, MIN_MKT_EP = 30, 10, 10, 5, 5     # the frozen section-3 floor
DAYS_MIN, DAYS_N = 3, 4                                                       # the RD sign on >= 3 of the 4 days
READINGS = ('adequate_direction_consistent', 'adequate_inconsistent', 'thin')
QS = (0.10, 0.25, 0.50, 0.75, 0.90)
MUT_ENV = 'MT_CT_MUT'
MUTANTS = ('swap_labels', 'drop_competitors', 'blank_competitors', 'guess_sub', 'equal_weights', 'censor_upper_drop', 'timing_prev',
           'floor_all_population', 'tl_own_strata', 'days_any')
READINGS_DECLARED = dict(
    censored='censored = end_type censored_gap (gap / unknown_touch / empty_side / crossed / window_end / book_end); horizon_no_change is its own '
             'outcome (the contract table lists it apart): a non-response that is neither censored nor a competing transition',
    competing='competing non-response transitions of the support floor = other_improve / other_retreat / two_side_* (sub_unknown included) / '
              'same_retreat_further',
    risk='the RD risk = same-side responses / all episodes of the stratum-group (censored counted as non-responses: the lower convention); the '
         'censor bounds pair lower with upper risks',
    support_clocks='a contrast is adequate only if the floor holds at BOTH D0 and LAG (the reading uses both clocks)',
    markets='>= 5 markets in each of which BOTH groups hold >= 5 episodes within the mixed strata',
    days='the pooled sign (D0 = LAG, nonzero) must be carried by the day RD on >= 3 of the 4 measurement days at BOTH clocks; a day without '
         'mixed strata or with RD 0 does not carry it',
    two_side_sub='from the reconcile.csv rows of the same (day, mid, end_frame): in the census population, class two_side, one sub-label, times '
                 'equal to the end pair, else refused; no saved row -> two_side_sub_unknown',
    strata='market (day, mid) x side x delta bin x time-left bin of the SAME clock (touch_transition.contrast_code); the time_left contrast omits '
           'its own bin',
    concentration='c_m = sum over the mixed strata of market m of w RD / W (sum c_m = RD); top market = the largest contribution in the pooled '
                  'direction, share = c_m / RD (above 1 under offsets); the largest absolute share and the top weight share beside',
    timing='lower = max(0, resp_lo - t_start), upper = resp_hi - t_prev, endpoint = resp_hi - t_start, next frame = saved no_frame_between')
# ---- saved inputs: columns read (each file bound to its producing receipt) ----
EP_STR = ('day', 'side', 'delta_bin', 'end_type', 'end_reason', 'end_class', 'resp_type')
EP_INT = ('mid', 'episode_id', 'frame', 't_prev_ms', 't_start_ms', 'end_frame', 't_end_ms', 'exposure_ms', 'exposure_lo_ms', 'resp_lo_ms',
          'resp_hi_ms')
EP_FLOAT = ('best_prev', 'best_retreat', 'delta_ticks', 'resp_best')
EP_BOOL = ('no_frame_between',)
ST_STR = tuple(f'{c}_{v}' for v in VARIANTS for c in ('state_status',) + tuple(STATE_COL.values()))
EP_READ = EP_STR + EP_INT + EP_FLOAT + EP_BOOL + ST_STR
RC_READ = ('day', 'mid', 'frame', 't_open_ms', 't_create_ms', 'pair_status', 'cls', 'sublabel')
RC_INT = ('mid', 'frame', 't_open_ms', 't_create_ms')
OUT_CSV = ('outcomes.csv', 'fractions.csv', 'rd.csv', 'support.csv', 'timing.csv')
OUT_FILES = OUT_CSV + ('claims.json', 'receipt.json')
SRC_CONST = dict(CONTRASTS=[[c[0], c[1], c[2]] for c in CONTRASTS], DELTA_LAB=list(DELTA_LAB), END_TYPES=list(END_TYPES),
                 END_REASONS=list(END_REASONS), RT_LAB=list(RT_LAB), VARIANTS=list(VARIANTS), CLASSES=list(CLASSES))
SRC_CONST_SETS = dict(TWO_SUB=TWO_SUB, UNK_SUB=UNK_SUB)


# ------------------------------------------------------------------ small helpers ------------------------------------------------------------------
def _now():
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())


def hwm():
    try:
        with open('/proc/self/status') as f:
            for line in f:
                if line.startswith('VmHWM:'):
                    return int(line.split()[1]) / 1024.
    except OSError:
        pass
    return 0.


def guard(where):
    h = hwm()
    need(h < C.GUARD_MIB, f'memory self-guard: VmHWM {h:.0f} MiB >= {C.GUARD_MIB:.0f} MiB at {where}')
    return h


def cpu_guard(where):
    c = time.process_time()
    need(c < CPU_CAP_S, f'CPU self-guard: {c:.2f} s >= {CPU_CAP_S} s at {where}')
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
    """JSON-safe copy: NaN / inf -> None, numpy scalars / arrays -> Python, keys -> str, tuples -> lists."""
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, np.ndarray):
        return [_clean(v) for v in o.tolist()]
    if isinstance(o, (bool, np.bool_)):
        return bool(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, (float, np.floating)):
        return float(o) if np.isfinite(o) else None
    if isinstance(o, Path):
        return str(o)
    return o


def _json(path, obj):
    Path(path).write_bytes((json.dumps(_clean(obj), indent=1, sort_keys=True, allow_nan=False) + '\n').encode())


def _jl(obj):
    return json.dumps(_clean(obj), sort_keys=True, allow_nan=False)


def _git(files):
    """HEAD and whether the files are tracked and unmodified (the run's freeze test reads 'unmodified')."""
    try:
        g = ['git', '-C', str(C.REPO)]
        head = subprocess.run(g + ['rev-parse', 'HEAD'], capture_output=True, text=True, timeout=30).stdout.strip() or None
        rel = [_rel(f) for f in files]
        tracked = subprocess.run(g + ['ls-files', '--error-unmatch', '--'] + rel, capture_output=True, text=True, timeout=30).returncode == 0
        st = subprocess.run(g + ['status', '--porcelain', '--'] + rel, capture_output=True, text=True, timeout=30)
        return dict(head=head, files=rel, tracked=tracked, unmodified=bool(tracked and st.returncode == 0 and not st.stdout.strip()))
    except (OSError, ValueError, subprocess.SubprocessError):
        return dict(head=None, files=None, tracked=None, unmodified=False)


def mutant():
    """The selftest's planted-error switch MT_CT_MUT, read at call time; production refuses any value."""
    m = os.environ.get(MUT_ENV, '')
    need(m == '' or m in MUTANTS, f'{MUT_ENV}={m!r} is not a declared mutant {MUTANTS} (refused)')
    return m


def _in(x, labs):
    s = set(labs)
    return np.fromiter((v in s for v in x), dtype=bool, count=len(x))


def _sign(x):
    x = float(x)
    return 0 if (not np.isfinite(x) or x == 0.) else (1 if x > 0 else -1)


def _qd(x):
    x = np.asarray(x, np.float64)
    out = dict(n=int(len(x)), mean=float(x.mean()) if len(x) else NAN, min=float(x.min()) if len(x) else NAN,
               max=float(x.max()) if len(x) else NAN)
    for q in QS:
        out[f'p{int(round(q * 100)):02d}'] = float(np.quantile(x, q)) if len(x) else NAN
    return out


# ------------------------------------------------------------------ saved inputs ------------------------------------------------------------------
def bound_sha(rec, key):
    """The sha256 the producing receipt records for one of its outputs (rec['outputs'][key]['sha256'])."""
    o = rec.get('outputs') if isinstance(rec, dict) else None
    v = o.get(key) if isinstance(o, dict) else None
    if isinstance(v, dict):
        v = v.get('sha256')
    return v.lower() if isinstance(v, str) and re.fullmatch(r'[0-9a-fA-F]{64}', v) else None


def bind(path, rec, key, hashes):
    """An input file bound to the sha256 its producing receipt records (refused when missing, unbound or different)."""
    path = Path(path)
    need(path.is_file(), f'input {_rel(path)} is missing (refused)')
    want = bound_sha(rec, key)
    need(want is not None, f'the producing receipt records no sha256 for {key} (refused: unbound input)')
    got = _sha(path)
    need(got == want, f'{key}: sha256 {got[:16]} differs from the producing receipt {want[:16]} (refused)')
    hashes[_rel(path)] = dict(sha256=got, bound_sha256=want, bytes=int(path.stat().st_size))
    return got


def check_source_receipt(rec, rehearsal):
    """The producing receipt: a touch_transition_v1 v2 full run without a planted error on the measurement days, whose frozen definitions equal
    this tool's -> the episode count it records."""
    need(isinstance(rec, dict) and rec.get('version') == SRC_VERSION and rec.get('kind') == 'run' and rec.get('mutant', '') == ''
         and rec.get('run') == 'v2', f'the producing receipt is not a {SRC_VERSION} v2 full run without a planted error (refused)')
    need(rec.get('days') == list(C.MEASURE_DAYS), f'the producing receipt days {rec.get("days")} are not the measurement days (refused)')
    K = rec.get('constants') if isinstance(rec.get('constants'), dict) else {}
    for k, want in SRC_CONST.items():
        need(K.get(k) == want, f'the producing receipt constant {k} = {K.get(k)!r} differs from this tool {want!r} (refused: frozen definitions)')
    for k, want in SRC_CONST_SETS.items():
        got = K.get(k)
        need(isinstance(got, list) and sorted(got) == sorted(want),
             f'the producing receipt constant {k} = {got!r} differs from this tool {list(want)!r} (refused: frozen definitions)')
    n = (rec.get('counts') or {}).get('episodes')
    outs = rec.get('outputs') if isinstance(rec.get('outputs'), dict) else {}
    need(isinstance(n, int) and (outs.get('episodes.parquet') or {}).get('rows') == n,
         f'the producing receipt episode counts disagree ({n} vs {(outs.get("episodes.parquet") or {}).get("rows")}) (refused)')
    need(rehearsal or n == N_EP_CONTRACT, f'the producing receipt holds {n} episodes, the contract population {N_EP_CONTRACT} (refused)')
    need(isinstance((outs.get('reconcile.csv') or {}).get('rows'), int), 'the producing receipt records no reconcile.csv row count (refused)')
    return int(n)


def ep_arrays(D, what='episodes'):
    """A frame holding EP_READ -> dict of fresh numpy arrays (strings as Python str; a null in a string column or a non-integer / non-boolean
    column refused)."""
    miss = [c for c in EP_READ if c not in D.columns]
    need(not miss, f'{what} lacks {miss} (refused: interface)')
    A = {}
    for c in EP_STR + ST_STR:
        s = D[c]
        need(int(s.isna().sum()) == 0, f'{what}: column {c} holds nulls (refused)')
        A[c] = np.array([str(v) for v in s.tolist()], dtype=object)
    for c in EP_INT:
        need(pd.api.types.is_integer_dtype(D[c].dtype), f'{what}: column {c} is not integer ({D[c].dtype}) (refused)')
        A[c] = np.array(D[c].to_numpy(), dtype=np.int64)
    for c in EP_FLOAT:
        need(pd.api.types.is_numeric_dtype(D[c].dtype), f'{what}: column {c} is not numeric ({D[c].dtype}) (refused)')
        A[c] = np.array(D[c].to_numpy(dtype=np.float64), dtype=np.float64)
    for c in EP_BOOL:
        need(D[c].dtype == bool, f'{what}: column {c} is not boolean ({D[c].dtype}) (refused)')
        A[c] = np.array(D[c].to_numpy(), dtype=bool)
    return A


def rc_arrays(D, what='reconcile.csv'):
    """A frame holding RC_READ -> dict of fresh numpy arrays (integers checked whole, strings as Python str)."""
    miss = [c for c in RC_READ if c not in D.columns]
    need(not miss, f'{what} lacks {miss} (refused: interface)')
    R = {}
    for c in RC_READ:
        if c in RC_INT:
            ss = [str(x) for x in D[c].tolist()]
            need(all(re.fullmatch(r'-?[0-9]+', x) for x in ss), f'{what}: column {c} holds non-integer values (refused)')
            R[c] = np.array([int(x) for x in ss], dtype=np.int64)
        else:
            s = D[c]
            need(int(s.isna().sum()) == 0, f'{what}: column {c} holds nulls (refused)')
            R[c] = np.array([str(x) for x in s.tolist()], dtype=object)
    return R


def read_episodes(path):
    import pyarrow.parquet as pq
    names = pq.read_schema(path).names
    miss = [c for c in EP_READ if c not in names]
    need(not miss, f'episodes.parquet lacks {miss} (refused: interface)')
    D = pq.read_table(path, columns=list(EP_READ)).to_pandas()
    return ep_arrays(D, 'episodes.parquet'), int(len(D))


def read_reconcile(path):
    hdr = list(pd.read_csv(path, nrows=0).columns)
    miss = [c for c in RC_READ if c not in hdr]
    need(not miss, f'reconcile.csv lacks {miss} (refused: interface)')
    D = pd.read_csv(path, usecols=list(RC_READ), dtype=str, keep_default_na=False)
    return rc_arrays(D, 'reconcile.csv'), int(len(D))


def validate(A):
    """Every read label inside its frozen set; time_left_bin empty exactly when that clock's state is missing; unique episode keys."""
    n = len(A['day'])
    need(all(len(v) == n for v in A.values()), 'internal: ragged episode arrays')
    for c, labs in (('side', SIDES), ('delta_bin', DELTA_LAB), ('end_type', END_TYPES), ('end_reason', END_REASONS),
                    ('end_class', CLASSES + ('',)), ('resp_type', RT_LAB + ('',))):
        bad = ~_in(A[c], labs)
        need(not bad.any(), f'{c}: {int(bad.sum())} values outside {labs}, e.g. {A[c][bad][:3].tolist()} (refused)')
    for v in VARIANTS:
        st = A[f'state_status_{v}']
        bad = ~_in(st, STATUS_LAB)
        need(not bad.any(), f'state_status_{v}: {int(bad.sum())} values outside {STATUS_LAB} (refused)')
        for var, col in STATE_COL.items():
            x = A[f'{col}_{v}']
            bad = ~_in(x, STATE_LABS[var])
            need(not bad.any(), f'{col}_{v}: {int(bad.sum())} labels outside the frozen bins, e.g. {x[bad][:3].tolist()} (refused)')
        need(bool(((A[f'time_left_bin_{v}'] == '') == (st != 'ok')).all()), f'time_left_bin_{v} is not empty exactly when the state is missing (refused)')
    keys = set(zip(A['day'].tolist(), A['mid'].tolist(), A['episode_id'].tolist()))
    need(len(keys) == n, 'episode keys (day, mid, episode_id) are not unique (refused)')


# ------------------------------------------------------------------ section 1: outcomes and reconciliation ------------------------------------------------------------------
def two_side_subs(A, R):
    """Per episode the two-side sub-label of its ending pair from the saved reconcile rows of the same (day, mid, end_frame) -> (sub, n_rows);
    every such row must lie in the census population with class two_side, one sub-label and the episode's end-pair times (refused otherwise);
    no row -> '' (two_side_sub_unknown). MT_CT_MUT=guess_sub is a planted error."""
    n = len(A['day'])
    sub = np.full(n, '', dtype=object)
    nrow = np.zeros(n, np.int64)
    two = (A['end_type'] == 'censored_other') & (A['end_reason'] == 'two_side')
    idx = {}
    for j, key in enumerate(zip(R['day'].tolist(), R['mid'].tolist(), R['frame'].tolist())):
        idx.setdefault(key, []).append(j)
    for i in np.flatnonzero(two).tolist():
        key = (str(A['day'][i]), int(A['mid'][i]), int(A['end_frame'][i]))
        js = idx.get(key, [])
        if not js:
            continue
        t_open = int(A['t_start_ms'][i]) + int(A['exposure_lo_ms'][i])
        t_create = int(A['t_end_ms'][i])
        st = {(str(R['pair_status'][j]), str(R['cls'][j]), str(R['sublabel'][j]), int(R['t_open_ms'][j]), int(R['t_create_ms'][j])) for j in js}
        need(len(st) == 1, f'two-side sub-label: the saved rows of {key} disagree {sorted(st)} (refused)')
        ps, cl, sl, to, tc = st.pop()
        need(ps == 'in_census' and cl == 'two_side' and sl in TWO_SUB,
             f'two-side sub-label: the saved rows of {key} are {ps} / {cl} / {sl!r}, not an in-census two_side pair (refused)')
        need(to == t_open and tc == t_create,
             f'two-side sub-label: the saved rows of {key} span ({to}, {tc}], the episode end pair ({t_open}, {t_create}] (refused)')
        sub[i] = sl
        nrow[i] = len(js)
    if mutant() == 'guess_sub':                   # planted error: an unmatched two-side end given a guessed sub-label
        sub[two & (nrow == 0)] = 'both_down'
    return sub, nrow


def _label1(s, et, er, ec, rt, sb):
    o = 'ask' if s == 'bid' else 'bid'
    if et == 'response':
        return ('same_' + rt) if rt in RT_LAB else ''
    if et == 'censored_other':
        if er == 'other_side':
            return {f'improve_{o}': 'other_improve', f'retreat_{o}': 'other_retreat'}.get(ec, '')
        if er == 'two_side':
            return ('two_side_' + sb) if sb else 'two_side_sub_unknown'
        if er == 'same_side_retreat':
            return 'same_retreat_further'
        return ''
    if et == 'no_response_1s':
        return 'horizon_no_change' if er == 'horizon' else ''
    if et == 'censored_gap':
        return ('censored_' + er) if er in CENSOR_REASONS else ''
    return ''


def outcome_labels(A, sub):
    """The next detected transition of every episode ('' when the saved columns fit no declared outcome: refused by mutual exclusion).
    MT_CT_MUT=swap_labels / blank_competitors are planted errors."""
    lab = np.array([_label1(*z) for z in zip(A['side'].tolist(), A['end_type'].tolist(), A['end_reason'].tolist(), A['end_class'].tolist(),
                                              A['resp_type'].tolist(), sub.tolist())], dtype=object)
    m = mutant()
    if m == 'swap_labels':                        # planted error: outcome labels swapped within their groups
        sw = {'other_improve': 'other_retreat', 'other_retreat': 'other_improve', 'same_restoring': 'same_overshoot', 'same_overshoot': 'same_restoring'}
        lab = np.array([sw.get(x, x) for x in lab.tolist()], dtype=object)
    elif m == 'blank_competitors':                # planted error: competing moves left without an outcome
        lab = np.array(['' if GROUP_OF.get(x) in COMPETING else x for x in lab.tolist()], dtype=object)
    return lab


def group_of(lab):
    return np.array([GROUP_OF.get(x, '') for x in np.asarray(lab, dtype=object).tolist()], dtype=object)


def _ident1(s, o, et, er, ec, rt, bp, br, dt, db, rb, rlo, rhi, t0, tend, expo, elo, nfb, nrow):
    """'' when outcome o agrees with the saved columns of its episode, else the first failing class-direction identity."""
    sg = 1. if s == 'bid' else -1.
    so = 'ask' if s == 'bid' else 'bid'
    if not (np.isfinite(bp) and np.isfinite(br) and sg * (bp - br) > 0):
        return 'retreat direction'
    if not abs(abs(bp - br) - dt) <= 1e-9:
        return 'retreat delta_ticks'
    if db != DELTA_LAB[0 if dt < 1.5 else (1 if dt < 2.5 else 2)]:
        return 'retreat delta bin'
    g = GROUP_OF.get(o)
    if g is None:
        return 'undeclared outcome'
    if et != GROUP_END_TYPE[g] or er not in GROUP_END_REASON[g]:
        return 'end type / reason'
    if g == 'same_response':
        if ec != f'improve_{s}' or rt != o[len('same_'):]:
            return 'response class'
        if not np.isfinite(rb):
            return 'response price missing'
        if o == 'same_restoring' and not rb == bp:
            return 'restoring price'
        if o == 'same_overshoot' and not sg * (rb - bp) > 0:
            return 'overshoot price'
        if o == 'same_partial' and not (sg * (rb - br) > 0 and sg * (bp - rb) > 0):
            return 'partial price'
        if not (t0 <= rlo <= rhi and rhi == tend and expo == rhi - t0 and elo == rlo - t0):
            return 'response times'
        if nfb and rlo != t0:
            return 'next-frame flag'
        return ''
    if rt != '' or np.isfinite(rb) or rlo != -1 or rhi != -1 or nfb:
        return 'non-response carries response fields'
    if g == 'other_side':
        if ec != ('improve_' if o == 'other_improve' else 'retreat_') + so:
            return 'other-side class'
    elif g == 'two_side':
        if ec != 'two_side':
            return 'two-side class'
        if (o == 'two_side_sub_unknown') != (nrow == 0):
            return 'two-side sub-label provenance'
    elif g == 'further_retreat':
        if ec != f'retreat_{s}':
            return 'further-retreat class'
    elif g == 'horizon':
        if ec != '':
            return 'horizon class'
    else:
        r = o[len('censored_'):]
        if er != r or ec not in CENSOR_CLASS[r]:
            return 'censor reason / class'
    return ''


def identity_violations(A, lab, nrow):
    it = zip(A['side'].tolist(), lab.tolist(), A['end_type'].tolist(), A['end_reason'].tolist(), A['end_class'].tolist(), A['resp_type'].tolist(),
             A['best_prev'].tolist(), A['best_retreat'].tolist(), A['delta_ticks'].tolist(), A['delta_bin'].tolist(), A['resp_best'].tolist(),
             A['resp_lo_ms'].tolist(), A['resp_hi_ms'].tolist(), A['t_start_ms'].tolist(), A['t_end_ms'].tolist(), A['exposure_ms'].tolist(),
             A['exposure_lo_ms'].tolist(), A['no_frame_between'].tolist(), nrow.tolist())
    out = []
    for i, z in enumerate(it):
        r = _ident1(*z)
        if r:
            out.append((i, r))
    return out


def outcome_table(A, R, n_expected):
    """Section 1: the mutually exclusive next-transition outcome of every episode and its reconciliation (episode totals, mutual exclusion,
    end-type identities, class-direction identities); refused on any failure. MT_CT_MUT=drop_competitors is a planted error (with the
    labelling ones)."""
    sub, nrow = two_side_subs(A, R)
    lab = outcome_labels(A, sub)
    if mutant() == 'drop_competitors':            # planted error: competing moves discarded as non-responses
        keep = ~_in(group_of(lab), COMPETING)
        A = {k: v[keep] for k, v in A.items()}
        lab, sub, nrow = lab[keep], sub[keep], nrow[keep]
    n = int(len(lab))
    need(n == int(n_expected), f'episode totals: {n} outcome rows, the bound population holds {n_expected} (refused)')
    oh = np.stack([lab == o for o in OUTCOMES], axis=1) if n else np.zeros((0, len(OUTCOMES)), bool)
    rs = oh.sum(1)
    need(bool((rs == 1).all()), f'mutual exclusion: {int((rs != 1).sum())} episodes without exactly one declared outcome, e.g. rows '
                                f'{np.flatnonzero(rs != 1)[:3].tolist()} labelled {lab[rs != 1][:3].tolist()} (refused)')
    cnt = {o: int(k) for o, k in zip(OUTCOMES, oh.sum(0).tolist())}
    need(sum(cnt.values()) == n, 'episode totals: the outcome counts do not add to the population (refused)')
    grp = group_of(lab)
    gcnt = {g: int((grp == g).sum()) for g in OGROUPS}
    etc = {e: int((A['end_type'] == e).sum()) for e in END_TYPES}
    for e in END_TYPES:
        k = sum(gcnt[g] for g in OGROUPS if GROUP_END_TYPE[g] == e)
        need(k == etc[e], f'end-type identity: {e} holds {etc[e]} episodes, its outcome groups {k} (refused)')
    bad = identity_violations(A, lab, nrow)
    if bad:
        i0, r0 = bad[0]
        raise ValueError(f'class-direction identity: {len(bad)} violations {dict(Counter(r for _, r in bad))}; first row {i0} '
                         f'({A["day"][i0]}, {int(A["mid"][i0])}, {int(A["episode_id"][i0])}) labelled {lab[i0]!r}: {r0} (refused)')
    two = grp == 'two_side'
    recon = dict(n_episodes=n, totals_equal_population=bool(n == int(n_expected) and sum(cnt.values()) == n),
                 mutual_exclusion=bool((rs == 1).all()), end_type_identity=bool(all(sum(gcnt[g] for g in OGROUPS if GROUP_END_TYPE[g] == e) == etc[e]
                                                                                for e in END_TYPES)),
                 class_direction_violations=int(len(bad)), outcome_counts=cnt, outcome_group_counts=gcnt, end_type_counts=etc,
                 two_side_sub=dict(n_two_side=int(two.sum()), n_sub_known=int((two & (sub != '')).sum()), n_sub_unknown=int((two & (sub == '')).sum()),
                                   saved_rows_matched=int(nrow[two].sum())))
    return A, lab, grp, sub, nrow, recon


# ------------------------------------------------------------------ section 2: states, contrasts, risk differences ------------------------------------------------------------------
def contrast_groups(A, v, cn):
    """1 exposed / 0 reference / -1 out (the clock's state missing, or another bin), from the saved label column of clock v."""
    _, _, _, var, ex, rf = CSPEC[cn]
    x = A[f'{STATE_COL[var]}_{v}']
    ok = A[f'state_status_{v}'] == 'ok'
    return np.where(ok & _in(x, ex), 1, np.where(ok & _in(x, rf), 0, -1)).astype(np.int64)


def strata(A, v, cn, mk):
    """Stratum per episode: market x side x delta bin x time-left bin of clock v; the time_left contrast omits its own bin (MT_CT_MUT=
    tl_own_strata keeps it: a planted error)."""
    sd = (A['side'] == 'ask').astype(np.int64)
    db = np.array([DELTA_LAB.index(x) for x in A['delta_bin'].tolist()], np.int64)
    base = (np.asarray(mk, np.int64) * 2 + sd) * 3 + db
    if cn == 'time_left' and mutant() != 'tl_own_strata':
        return base
    tl = np.array([TL_IDX.get(x, len(TL_LAB)) for x in A[f'time_left_bin_{v}'].tolist()], np.int64)
    return base * (len(TL_LAB) + 1) + tl


def risk_lo(r, n):
    """The lower response risk r / n (censored counted as non-responses); NaN without episodes."""
    r, n = np.asarray(r, np.float64), np.asarray(n, np.float64)
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(n > 0, r / np.where(n > 0, n, 1.), NAN)


def risk_hi(r, c, n):
    """The upper response risk (r + c) / n (censored counted as responses); MT_CT_MUT=censor_upper_drop drops them (a planted error)."""
    r, c, n = (np.asarray(x, np.float64) for x in (r, c, n))
    num = r if mutant() == 'censor_upper_drop' else r + c
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(n > 0, num / np.where(n > 0, n, 1.), NAN)


def rd_pool(n1, r1, c1, n0, r0, c0, sel=None):
    """The harmonic-weight pooled same-side-response risk difference over the selected common-support strata (both groups present):
    w = n1 n0 / (n1 + n0) (MT_CT_MUT=equal_weights: w = 1, a planted error), RD_k = r1 / n1 - r0 / n0, censor bounds sum w (p1_lo - p0_hi) / W
    .. sum w (p1_hi - p0_lo) / W -> dict(rd, lo, hi, W, n_strata, w, d)."""
    n1, r1, c1, n0, r0, c0 = (np.asarray(x, np.float64) for x in (n1, r1, c1, n0, r0, c0))
    m = (n1 > 0) & (n0 > 0)
    if sel is not None:
        m = m & np.asarray(sel, bool)
    w = np.where(m, n1 * n0 / np.where(m, n1 + n0, 1.), 0.)
    if mutant() == 'equal_weights':
        w = m.astype(np.float64)
    p1, p0 = np.where(m, risk_lo(r1, n1), 0.), np.where(m, risk_lo(r0, n0), 0.)
    p1h, p0h = np.where(m, risk_hi(r1, c1, n1), 0.), np.where(m, risk_hi(r0, c0, n0), 0.)
    d = p1 - p0
    W = float(w.sum())
    if not W > 0:
        return dict(rd=NAN, lo=NAN, hi=NAN, W=0., n_strata=0, w=w, d=d)
    return dict(rd=float((w * d).sum() / W), lo=float((w * (p1 - p0h)).sum() / W), hi=float((w * (p1h - p0)).sum() / W), W=W,
                n_strata=int(m.sum()), w=w, d=d)


def fraction_rows(base, lab, grp):
    """Outcome fractions of one population: every declared outcome and every outcome group (zeros kept)."""
    n = int(len(lab))
    c, cg = Counter(np.asarray(lab, dtype=object).tolist()), Counter(np.asarray(grp, dtype=object).tolist())
    rows = [dict(base, level='outcome', outcome=o, n=int(c.get(o, 0)), n_group=n, fraction=(c.get(o, 0) / n) if n else NAN) for o in OUTCOMES]
    rows += [dict(base, level='outcome_group', outcome=g, n=int(cg.get(g, 0)), n_group=n, fraction=(cg.get(g, 0) / n) if n else NAN) for g in OGROUPS]
    return rows


def contrast_block(A, lab, grp, v, cn, mk, mkeys, mday, days, fractions=True):
    """One (clock, contrast): groups from the saved labels, strata, the section-3 support counts, the pooled harmonic RD with its censor bounds,
    the day / side / market scopes, contributions, the all-population counts and the outcome fractions; identities refused on failure.
    MT_CT_MUT=floor_all_population is a planted error."""
    n_mk = len(mkeys)
    g = contrast_groups(A, v, cn)
    st = strata(A, v, cn, mk)
    n = int(len(g))
    S = int(st.max()) + 1 if n else 1
    resp = grp == 'same_response'
    comp = _in(grp, COMPETING)
    cens = grp == 'censored'
    hor = grp == 'horizon'
    g1, g0 = g == 1, g == 0

    def bc(m):
        return np.bincount(st[m], minlength=S).astype(np.int64)
    n1, n0, r1, r0 = bc(g1), bc(g0), bc(g1 & resp), bc(g0 & resp)
    c1, c0, k1, k0, h1, h0 = bc(g1 & cens), bc(g0 & cens), bc(g1 & comp), bc(g0 & comp), bc(g1 & hor), bc(g0 & hor)
    mixed = (n1 > 0) & (n0 > 0)
    s_mk = np.full(S, -1, np.int64)
    s_sd = np.full(S, -1, np.int64)
    s_mk[st] = mk
    s_sd[st] = (A['side'] == 'ask').astype(np.int64)
    s_day = np.where(s_mk >= 0, np.asarray(mday, np.int64)[np.maximum(s_mk, 0)], -1)

    def scope(sel):
        Q = rd_pool(n1, r1, c1, n0, r0, c0, sel)
        mm = mixed & np.asarray(sel, bool)
        return dict(rd=Q['rd'], lo=Q['lo'], hi=Q['hi'], W=Q['W'], n_strata=Q['n_strata'], sign=_sign(Q['rd']), n1=int(n1[mm].sum()),
                    n0=int(n0[mm].sum()), r1=int(r1[mm].sum()), r0=int(r0[mm].sum()), c1=int(c1[mm].sum()), c0=int(c0[mm].sum()))
    P = rd_pool(n1, r1, c1, n0, r0, c0)
    pooled = scope(np.ones(S, bool))
    day_sc = [scope(s_day == di) for di in range(len(days))]
    side_sc = [scope(s_sd == si) for si in range(len(SIDES))]
    mkt_sc = [scope(s_mk == j) for j in range(n_mk)]
    W = pooled['W']
    if W > 0:
        c_m = np.bincount(s_mk[mixed], weights=(P['w'] * P['d'])[mixed], minlength=n_mk) / W
        w_m = np.bincount(s_mk[mixed], weights=P['w'][mixed], minlength=n_mk) / W
        tol = 1e-9 * max(1., W)
        need(abs(sum(x['W'] for x in day_sc) - W) <= tol and abs(sum(x['W'] for x in side_sc) - W) <= tol
             and abs(sum(x['W'] for x in mkt_sc) - W) <= tol, f'{v} {cn}: the day / side / market weights do not add to the pooled weight (refused)')
        need(abs(float(c_m.sum()) - pooled['rd']) <= 1e-9, f'{v} {cn}: the market contributions do not add to the pooled RD (refused)')
        need(pooled['lo'] - 1e-12 <= pooled['rd'] <= pooled['hi'] + 1e-12, f'{v} {cn}: the censor bounds do not bracket the RD (refused)')
    else:
        c_m, w_m = np.full(n_mk, NAN), np.full(n_mk, NAN)
        need(pooled['n_strata'] == 0 and pooled['n1'] == 0 and pooled['n0'] == 0, f'{v} {cn}: common support without weight (refused)')
    # ---- section-3 support: every count, the floor within the mixed strata ----
    ok_st = A[f'state_status_{v}'] == 'ok'
    n_missing, n_other = int((~ok_st).sum()), int((ok_st & (g < 0)).sum())
    n1a, n0a = int(g1.sum()), int(g0.sum())
    need(n1a + n0a + n_other + n_missing == n, f'{v} {cn}: groups + other bins + state missing differ from the episodes (refused)')
    single1, single0 = (n1 > 0) & (n0 == 0), (n0 > 0) & (n1 == 0)
    n1u, n0u = int(n1[single1].sum()), int(n0[single0].sum())
    need(pooled['n1'] + n1u == n1a and pooled['n0'] + n0u == n0a, f'{v} {cn}: the common-support denominators do not reconcile (refused)')
    m1 = np.bincount(s_mk[mixed], weights=n1[mixed], minlength=n_mk)
    m0 = np.bincount(s_mk[mixed], weights=n0[mixed], minlength=n_mk)
    q = dict(n1=pooled['n1'], n0=pooled['n0'], r1=pooled['r1'], r0=pooled['r0'], k1=int(k1[mixed].sum()), k0=int(k0[mixed].sum()), m1=m1, m0=m0)
    if mutant() == 'floor_all_population':        # planted error: the floor counted over the whole groups, not the common-support strata
        occ = (n1 + n0) > 0
        q = dict(n1=n1a, n0=n0a, r1=int(r1.sum()), r0=int(r0.sum()), k1=int(k1.sum()), k0=int(k0.sum()),
                 m1=np.bincount(s_mk[occ], weights=n1[occ], minlength=n_mk), m0=np.bincount(s_mk[occ], weights=n0[occ], minlength=n_mk))
    mk_both = int(((q['m1'] >= MIN_MKT_EP) & (q['m0'] >= MIN_MKT_EP)).sum())
    floor = dict(episodes=bool(q['n1'] >= MIN_GROUP_EP and q['n0'] >= MIN_GROUP_EP), responses=bool(q['r1'] >= MIN_RESP and q['r0'] >= MIN_RESP),
                 competing=bool(q['k1'] >= MIN_COMP and q['k0'] >= MIN_COMP), markets=bool(mk_both >= MIN_MKT))
    adequate = bool(all(floor.values()))
    support = dict(n_episodes=n, n_state_missing=n_missing, n_other_bins=n_other, n_exposed_all=n1a, n_reference_all=n0a,
                   n_strata_occupied=int(((n1 + n0) > 0).sum()), n_strata_mixed=int(mixed.sum()), n_strata_exposed_only=int(single1.sum()),
                   n_strata_reference_only=int(single0.sum()), n_exposed_mixed=pooled['n1'], n_reference_mixed=pooled['n0'],
                   n_exposed_single_group=n1u, n_reference_single_group=n0u, responses_exposed_mixed=pooled['r1'],
                   responses_reference_mixed=pooled['r0'], competing_exposed_mixed=int(k1[mixed].sum()), competing_reference_mixed=int(k0[mixed].sum()),
                   censored_exposed_mixed=pooled['c1'], censored_reference_mixed=pooled['c0'], horizon_exposed_mixed=int(h1[mixed].sum()),
                   horizon_reference_mixed=int(h0[mixed].sum()), n_markets_mixed=int(((m1 + m0) > 0).sum()),
                   n_markets_both_ge5=int(((m1 >= MIN_MKT_EP) & (m0 >= MIN_MKT_EP)).sum()), n_markets_exposed_ge5=int((m1 >= MIN_MKT_EP).sum()),
                   n_markets_reference_ge5=int((m0 >= MIN_MKT_EP).sum()), floor_episodes=floor['episodes'], floor_responses=floor['responses'],
                   floor_competing=floor['competing'], floor_markets=floor['markets'], adequate_clock=adequate)
    # ---- the raw all-population counts beside the matched RD ----
    allp = dict(n1=n1a, r1=int((g1 & resp).sum()), c1=int((g1 & cens).sum()), k1=int((g1 & comp).sum()), h1=int((g1 & hor).sum()),
                n0=n0a, r0=int((g0 & resp).sum()), c0=int((g0 & cens).sum()), k0=int((g0 & comp).sum()), h0=int((g0 & hor).sum()))
    allp.update(p1_lo=float(risk_lo([allp['r1']], [n1a])[0]), p1_hi=float(risk_hi([allp['r1']], [allp['c1']], [n1a])[0]),
                p0_lo=float(risk_lo([allp['r0']], [n0a])[0]), p0_hi=float(risk_hi([allp['r0']], [allp['c0']], [n0a])[0]))
    allp['rd_crude'] = allp['p1_lo'] - allp['p0_lo']
    mixed_risk = dict(p1_lo=float(risk_lo([pooled['r1']], [pooled['n1']])[0]), p1_hi=float(risk_hi([pooled['r1']], [pooled['c1']], [pooled['n1']])[0]),
                      p0_lo=float(risk_lo([pooled['r0']], [pooled['n0']])[0]), p0_hi=float(risk_hi([pooled['r0']], [pooled['c0']], [pooled['n0']])[0]))
    # ---- contribution concentration ----
    s = pooled['sign']
    if W > 0 and s != 0:
        j = int(np.argmax(s * c_m))
        a = np.abs(c_m)
        top = dict(market=f'{mkeys[j][0]}:{mkeys[j][1]}', contribution=float(c_m[j]), share_of_rd=float(c_m[j] / pooled['rd']),
                   weight_share=float(w_m[j]), max_abs_share=float(a.max() / a.sum()) if a.sum() > 0 else NAN)
    else:
        top = dict(market='', contribution=NAN, share_of_rd=NAN, weight_share=NAN, max_abs_share=NAN)
    frac = []
    if fractions:
        inm = mixed[st]
        for gname, gm in (('exposed', g1), ('reference', g0)):
            for sc, sm in (('all', gm), ('mixed', gm & inm)):
                frac += fraction_rows(dict(variant=v, contrast=cn, group=gname, scope=sc), lab[sm], grp[sm])
    return dict(g=g, pooled=pooled, days=day_sc, sides=side_sc, markets=mkt_sc, c_m=c_m, w_m=w_m, top=top, support=support, floor=floor,
                adequate=adequate, sign=pooled['sign'], day_signs=[x['sign'] for x in day_sc], allp=allp, mixed_risk=mixed_risk, frac=frac)


def reading(B0, B1):
    """Section 3 reading of one contrast from its D0 and LAG blocks (READINGS_DECLARED support_clocks / days). MT_CT_MUT=days_any is a planted
    error."""
    if not (B0['adequate'] and B1['adequate']):
        return 'thin'
    s = B0['sign']
    k = 1 if mutant() == 'days_any' else DAYS_MIN
    ok = (s != 0 and B1['sign'] == s and sum(1 for x in B0['day_signs'] if x == s) >= k and sum(1 for x in B1['day_signs'] if x == s) >= k)
    return 'adequate_direction_consistent' if ok else 'adequate_inconsistent'


def population_bounds(grp):
    """Same-side response probability bounds of a population: lower = responses / all, upper = (responses + censored) / all."""
    grp = np.asarray(grp, dtype=object)
    n = int(len(grp))
    r, c = int((grp == 'same_response').sum()), int((grp == 'censored').sum())
    return dict(n=n, responses=r, censored=c, lower=float(risk_lo([r], [n])[0]), upper=float(risk_hi([r], [c], [n])[0]))


# ------------------------------------------------------------------ section 4: timing bounds ------------------------------------------------------------------
def timing_table(A, grp):
    """Per response: lower = max(0, resp_lo - t_start), upper = resp_hi - t_prev, endpoint = resp_hi - t_start, next frame (saved flag); the
    identity lower <= endpoint <= upper refused on failure. MT_CT_MUT=timing_prev reads t_prev for the lower bound (a planted error)."""
    i = np.flatnonzero(np.asarray(grp, dtype=object) == 'same_response')
    tp, ts, rl, rh = (np.asarray(A[c], np.int64)[i] for c in ('t_prev_ms', 't_start_ms', 'resp_lo_ms', 'resp_hi_ms'))
    lo = np.maximum(0, rl - (tp if mutant() == 'timing_prev' else ts)).astype(np.int64)
    hi = (rh - tp).astype(np.int64)
    ep = (rh - ts).astype(np.int64)
    nf = np.asarray(A['no_frame_between'], bool)[i]
    need(bool(((lo <= ep) & (ep <= hi)).all()), f'timing identity: lower <= endpoint <= upper fails on {int((~((lo <= ep) & (ep <= hi))).sum())} responses (refused)')
    T = pd.DataFrame(dict(day=np.asarray(A['day'], dtype=object)[i], mid=np.asarray(A['mid'], np.int64)[i],
                          episode_id=np.asarray(A['episode_id'], np.int64)[i], side=np.asarray(A['side'], dtype=object)[i],
                          resp_type=np.asarray(A['resp_type'], dtype=object)[i], t_prev_ms=tp, t_start_ms=ts, resp_lo_ms=rl, resp_hi_ms=rh, lower_ms=lo,
                          upper_ms=hi, endpoint_ms=ep, next_frame=nf))
    summ = dict(n_responses=int(len(i)), next_frame_n=int(nf.sum()), next_frame_fraction=float(nf.mean()) if len(i) else NAN,
                lower_zero_n=int((lo == 0).sum()), lower_ms=_qd(lo), upper_ms=_qd(hi), endpoint_ms=_qd(ep),
                convention='retreat (t_prev, t_start], response (resp_lo, resp_hi]; bounds of an interval, not an identified latency')
    return T, summ


# ------------------------------------------------------------------ tables and claims ------------------------------------------------------------------
def outcomes_frame(A, lab, grp, sub, nrow, BLK):
    d = {c: A[c] for c in ('day', 'mid', 'episode_id', 'side', 'frame', 't_prev_ms', 't_start_ms', 'delta_bin', 'end_type', 'end_reason', 'end_frame',
                           'end_class', 't_end_ms', 'resp_type')}
    d.update(outcome=lab, outcome_group=grp, response=grp == 'same_response', competing=_in(grp, COMPETING), censored=grp == 'censored',
             two_side_sub=sub, two_side_saved_rows=nrow)
    for v in VARIANTS:
        for c in ('state_status',) + tuple(STATE_COL.values()):
            d[f'{c}_{v}'] = A[f'{c}_{v}']
        for cn in CN_NAMES:
            g = BLK[(v, cn)]['g']
            d[f'grp_{cn}_{v}'] = np.where(g == 1, 'exposed', np.where(g == 0, 'reference', 'out'))
    return pd.DataFrame(d)


def _rd_row(v, cn, sc, key, Q, extra=None):
    r = dict(variant=v, contrast=cn, scope=sc, key=key, rd=Q['rd'], rd_censor_lo=Q['lo'], rd_censor_hi=Q['hi'], weight_sum=Q['W'],
             n_strata_mixed=Q['n_strata'], sign=Q['sign'], n1=Q['n1'], n0=Q['n0'], r1=Q['r1'], r0=Q['r0'], c1=Q['c1'], c0=Q['c0'])
    r.update(extra or {})
    return r


def rd_frame(BLK, days, mkeys):
    rows = []
    for (v, cn), B in BLK.items():
        a, t, P = B['allp'], B['top'], B['pooled']
        rows.append(_rd_row(v, cn, 'pooled', 'all', P, dict(
            contribution=P['rd'], weight_share=1. if P['W'] > 0 else NAN, n1_all=a['n1'], r1_all=a['r1'], c1_all=a['c1'], n0_all=a['n0'],
            r0_all=a['r0'], c0_all=a['c0'], p1_lo_all=a['p1_lo'], p1_hi_all=a['p1_hi'], p0_lo_all=a['p0_lo'], p0_hi_all=a['p0_hi'],
            rd_crude_all=a['rd_crude'], top_market=t['market'], top_contribution=t['contribution'], top_share_of_rd=t['share_of_rd'],
            top_weight_share=t['weight_share'], max_abs_share=t['max_abs_share'])))
        for di, dd in enumerate(days):
            rows.append(_rd_row(v, cn, 'day', dd, B['days'][di]))
        for si, sd in enumerate(SIDES):
            rows.append(_rd_row(v, cn, 'side', sd, B['sides'][si]))
        for j, (dd, mm) in enumerate(mkeys):
            rows.append(_rd_row(v, cn, 'market', f'{dd}:{mm}', B['markets'][j], dict(contribution=float(B['c_m'][j]), weight_share=float(B['w_m'][j]))))
    return pd.DataFrame(rows)


def support_frame(BLK, READ):
    return pd.DataFrame([dict(variant=v, contrast=cn, **B['support'], reading=READ[cn]) for (v, cn), B in BLK.items()])


def fractions_frame(lab, grp, BLK):
    rows = fraction_rows(dict(variant='all', contrast='population', group='all', scope='all'), lab, grp)
    for B in BLK.values():
        rows += B['frac']
    return pd.DataFrame(rows)


def make_claims(recon, BLK, READ, TS, PB, days, mkeys):
    cs = {}
    for cn in CN_NAMES:
        B = {v: BLK[(v, cn)] for v in VARIANTS}
        s0, s1 = B['D0']['sign'], B['LAG']['sign']
        cs[cn] = dict(
            reading=READ[cn], exposed=CSPEC[cn][1], reference=CSPEC[cn][2],
            adequate={v: B[v]['adequate'] for v in VARIANTS},
            floor_failed={v: [k for k, ok in B[v]['floor'].items() if not ok] for v in VARIANTS},
            rd={v: B[v]['pooled']['rd'] for v in VARIANTS},
            rd_censor_bounds={v: [B[v]['pooled']['lo'], B[v]['pooled']['hi']] for v in VARIANTS},
            sign={v: B[v]['sign'] for v in VARIANTS},
            d0_lag_sign_agree=bool(s0 != 0 and s0 == s1),
            timing_sensitive=bool(s0 != 0 and s1 != 0 and s0 != s1),
            days_carrying_pooled_sign={v: int(sum(1 for x in B[v]['day_signs'] if s0 != 0 and x == s0)) for v in VARIANTS},
            day_signs={v: dict(zip(days, B[v]['day_signs'])) for v in VARIANTS},
            side_signs={v: dict(zip(SIDES, [x['sign'] for x in B[v]['sides']])) for v in VARIANTS},
            top_market={v: B[v]['top'] for v in VARIANTS},
            groups={v: dict(exposed_all=B[v]['allp']['n1'], reference_all=B[v]['allp']['n0'], exposed_mixed=B[v]['pooled']['n1'],
                            reference_mixed=B[v]['pooled']['n0'], response_exposed_all=[B[v]['allp']['p1_lo'], B[v]['allp']['p1_hi']],
                            response_reference_all=[B[v]['allp']['p0_lo'], B[v]['allp']['p0_hi']],
                            response_exposed_mixed=[B[v]['mixed_risk']['p1_lo'], B[v]['mixed_risk']['p1_hi']],
                            response_reference_mixed=[B[v]['mixed_risk']['p0_lo'], B[v]['mixed_risk']['p0_hi']]) for v in VARIANTS})
    rc = Counter(READ.values())
    return dict(version=VERSION, contract_commit=CONTRACT_COMMIT, review_commit=REVIEW_COMMIT,
                population=dict(n_episodes=recon['n_episodes'], n_markets=len(mkeys), days=list(days), reconciliation=recon, response_bounds=PB),
                contrasts=cs, readings=dict(counts={r: int(rc.get(r, 0)) for r in READINGS},
                                            all_thin_or_inconsistent=bool(rc.get('adequate_direction_consistent', 0) == 0)),
                timing=TS)


def analyse(A, R, n_expected, n_markets=None, tables=True):
    """Sections 1-4 on the episode arrays A and the reconcile arrays R -> tables, claims and the per-(clock, contrast) blocks."""
    validate(A)
    A, lab, grp, sub, nrow, recon = outcome_table(A, R, n_expected)
    days = sorted(set(A['day'].tolist()))
    need(len(days) == DAYS_N and all(d in C.MEASURE_DAYS for d in days),
         f'days {days}: the day rule reads the {DAYS_N} measurement days {C.MEASURE_DAYS} (refused)')
    mkeys = sorted(set(zip(A['day'].tolist(), A['mid'].tolist())))
    need(n_markets is None or len(mkeys) == n_markets, f'{len(mkeys)} markets, the contract population holds {n_markets} (refused)')
    mi = {k: j for j, k in enumerate(mkeys)}
    mk = np.array([mi[k] for k in zip(A['day'].tolist(), A['mid'].tolist())], np.int64)
    mday = np.array([days.index(d) for d, _ in mkeys], np.int64)
    BLK = {(v, cn): contrast_block(A, lab, grp, v, cn, mk, mkeys, mday, days, fractions=tables) for v in VARIANTS for cn in CN_NAMES}
    READ = {cn: reading(BLK[('D0', cn)], BLK[('LAG', cn)]) for cn in CN_NAMES}
    T, TS = timing_table(A, grp)
    PB = population_bounds(grp)
    claims = make_claims(recon, BLK, READ, TS, PB, days, mkeys)
    out = dict(claims=claims, BLK=BLK, READ=READ, recon=recon, timing=TS, PB=PB, days=days, mkeys=mkeys, lab=lab, grp=grp, tables=None)
    if tables:
        out['tables'] = {'outcomes.csv': outcomes_frame(A, lab, grp, sub, nrow, BLK), 'fractions.csv': fractions_frame(lab, grp, BLK),
                         'rd.csv': rd_frame(BLK, days, mkeys), 'support.csv': support_frame(BLK, READ), 'timing.csv': T}
    return out


def summary_lines(RES):
    """The dry run's printout: reconciliation, the per-(contrast, clock) numbers, the readings and the timing bounds."""
    cl = RES['claims']
    pop = cl['population']
    L = ['RECON ' + _jl(pop['reconciliation']), 'RESPONSE_BOUNDS ' + _jl(pop['response_bounds'])]
    for cn in CN_NAMES:
        for v in VARIANTS:
            B = RES['BLK'][(v, cn)]
            a, p, mr, s = B['allp'], B['pooled'], B['mixed_risk'], B['support']
            L.append(f'CONTRAST {cn} {v} ' + _jl(dict(
                n_all=[a['n1'], a['n0']], n_mixed=[p['n1'], p['n0']], resp_all_lo_hi=[[a['p1_lo'], a['p1_hi']], [a['p0_lo'], a['p0_hi']]],
                resp_mixed_lo_hi=[[mr['p1_lo'], mr['p1_hi']], [mr['p0_lo'], mr['p0_hi']]], rd=p['rd'], rd_censor=[p['lo'], p['hi']], W=p['W'],
                rd_crude_all=a['rd_crude'], strata_mixed=p['n_strata'], responses_mixed=[p['r1'], p['r0']],
                competing_mixed=[s['competing_exposed_mixed'], s['competing_reference_mixed']], censored_mixed=[p['c1'], p['c0']],
                state_missing=s['n_state_missing'], other_bins=s['n_other_bins'], single_group=[s['n_exposed_single_group'], s['n_reference_single_group']],
                markets_both_ge5=s['n_markets_both_ge5'], floor=B['floor'], adequate=B['adequate'], day_rd=[x['rd'] for x in B['days']],
                day_signs=B['day_signs'], side_rd=[x['rd'] for x in B['sides']], side_signs=[x['sign'] for x in B['sides']], top_market=B['top'])))
        c = cl['contrasts'][cn]
        L.append(f'READING {cn} ' + _jl({k: c[k] for k in ('reading', 'adequate', 'floor_failed', 'rd', 'sign', 'd0_lag_sign_agree', 'timing_sensitive',
                                                          'days_carrying_pooled_sign')}))
    L.append('READINGS ' + _jl(cl['readings']))
    L.append('TIMING ' + _jl(cl['timing']))
    return L


# ------------------------------------------------------------------ write-once output ------------------------------------------------------------------
def write_out(out, tables, claims, rec_fn):
    """Write-once: every file in a partial directory, the receipt last (it hashes the others), then one rename."""
    out = Path(out)
    need(not out.exists(), f'write-once: {out} exists (refused)')
    tmp = out.with_name(f'.{out.name}.partial{os.getpid()}')
    need(not tmp.exists(), f'partial directory {tmp} exists (refused)')
    tmp.mkdir(parents=True)
    try:
        outs = {}
        for nm, df in tables.items():
            df.to_csv(tmp / nm, index=False)
            outs[nm] = dict(rows=int(len(df)))
        _json(tmp / 'claims.json', claims)
        outs['claims.json'] = {}
        for nm in outs:
            outs[nm].update(sha256=_sha(tmp / nm), bytes=int((tmp / nm).stat().st_size))
        rec = rec_fn(outs)
        _json(tmp / 'receipt.json', rec)
        need(not out.exists(), f'write-once: {out} appeared during the run (refused)')
        os.rename(tmp, out)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    return rec


def paths():
    """Run-time paths on the MT_RUN roots (v2: C.RECEIPT = the receipt's v2/ directory)."""
    ch = C.RECEIPT / 'chains'
    return dict(src=ch / 'touch_transition', out=ch / 'competing_transition')


def execute(dry, src=None, out=None, rehearsal=False, log=print):
    """run (write-once) or run --dry (no write) on the bound saved evidence. rehearsal=True (the selftest only: synthetic source trees) skips
    the git freeze / tracked-receipt requirements and the contract population size; it is recorded in the receipt."""
    t0, asof0 = time.time(), _now()
    stages, last = {}, [time.process_time()]

    def tick(name):
        c = time.process_time()
        stages[name] = round(stages.get(name, 0.) + c - last[0], 3)
        last[0] = c
    need(mutant() == '', f'{MUT_ENV}={os.environ.get(MUT_ENV)!r} is set: production never runs a planted error (refused before any input is read)')
    need(C.RUN == 'v2', f'MT_RUN={C.RUN!r}: this tool reads and writes the v2 receipt tree only (refused)')
    P = paths()
    src = Path(src) if src is not None else P['src']
    out = Path(out) if out is not None else P['out']
    git = _git(TOOL_FILES)
    if not dry:
        need(not out.exists(), f'write-once: {out} exists (refused)')
        need(rehearsal or bool(git.get('unmodified')),
             f'freeze: {[_rel(f) for f in TOOL_FILES]} must be tracked and unmodified before the run (refused; commit them first)')
    tick('preflight')
    rec_p = src / 'receipt.json'
    need(rec_p.is_file(), f'the producing receipt {_rel(rec_p)} is missing (refused)')
    rec = json.loads(rec_p.read_text())
    srcg = _git([rec_p])
    need(rehearsal or bool(srcg.get('unmodified')), f'the producing receipt {_rel(rec_p)} is not tracked and unmodified (refused)')
    n_rec = check_source_receipt(rec, rehearsal)
    hashes = {_rel(rec_p): dict(sha256=_sha(rec_p), bytes=int(rec_p.stat().st_size), tracked=srcg.get('tracked'), unmodified=srcg.get('unmodified'),
                                role='producing receipt (binds the two inputs)')}
    for nm in SRC_FILES:
        bind(src / nm, rec, nm, hashes)
    tick('bindings')
    A, n_ep = read_episodes(src / 'episodes.parquet')
    need(n_ep == n_rec, f'episodes.parquet holds {n_ep} rows, its receipt {n_rec} (refused)')
    R, n_rc = read_reconcile(src / 'reconcile.csv')
    n_rc_rec = int(rec['outputs']['reconcile.csv']['rows'])
    need(n_rc == n_rc_rec, f'reconcile.csv holds {n_rc} rows, its receipt {n_rc_rec} (refused)')
    hashes[_rel(src / 'episodes.parquet')]['rows'] = n_ep
    hashes[_rel(src / 'reconcile.csv')]['rows'] = n_rc
    tick('read')
    guard('read')
    RES = analyse(A, R, n_rec, None if rehearsal else N_MARKETS)
    tick('analyse')
    guard('analyse')
    cpu_guard('analyse')
    if dry:
        for line in summary_lines(RES):
            log(line)
        log('DRY ' + _jl(dict(wrote=None, rehearsal=bool(rehearsal), out_would_be=_rel(out), out_exists=out.exists(), freeze_ok=bool(git.get('unmodified')),
                              git=git, source_receipt_git=srcg, inputs=hashes, cpu_s=dict(by_stage=stages, process_total=round(time.process_time(), 3)),
                              vmhwm_mib=round(hwm(), 1))))
        return RES

    def rec_fn(outs):
        return dict(version=VERSION, kind='run', rehearsal=bool(rehearsal), argv=list(sys.argv), run=C.RUN, mutant=os.environ.get(MUT_ENV, ''),
                    asof_start=asof0, asof_end=_now(),
                    contract=dict(file=_rel(CONTRACT_FILE), commit=CONTRACT_COMMIT, sha256=_sha(CONTRACT_FILE), sections='1-4 and their acceptance items'),
                    review=dict(file=_rel(REVIEW_FILE), commit=REVIEW_COMMIT, sha256=_sha(REVIEW_FILE)),
                    sources={_rel(f): _sha(f) for f in TOOL_FILES + DEP_FILES}, inputs=hashes, outputs=outs,
                    population=dict(n_episodes=RES['recon']['n_episodes'], n_markets=len(RES['mkeys']), days=RES['days']),
                    readings={cn: RES['READ'][cn] for cn in CN_NAMES},
                    constants=dict(OUTCOMES=OUTCOMES, OUTCOME_GROUPS=OUTCOME_GROUPS, COMPETING=COMPETING, CONTRASTS=CONTRASTS, VARIANTS=VARIANTS,
                                   FLOOR=dict(MIN_GROUP_EP=MIN_GROUP_EP, MIN_RESP=MIN_RESP, MIN_COMP=MIN_COMP, MIN_MKT=MIN_MKT, MIN_MKT_EP=MIN_MKT_EP),
                                   DAYS_RULE=[DAYS_MIN, DAYS_N], READINGS=READINGS, MUTANTS=MUTANTS, N_EP_CONTRACT=N_EP_CONTRACT, N_MARKETS=N_MARKETS,
                                   CPU_CAP_S=CPU_CAP_S),
                    design=dict(p_values=False, permutations=False, intervals=False, significance_verdicts=False,
                                note='descriptive point estimates on four related consumed days; a thin or inconsistent answer ends this line'),
                    readings_declared=READINGS_DECLARED,
                    git=dict(head=git.get('head'), tool_files=git.get('files'), tracked=git.get('tracked'), unmodified=git.get('unmodified')),
                    cpu_s=dict(by_stage=dict(stages), process_total=round(time.process_time(), 3)), wall_s=round(time.time() - t0, 2),
                    vmhwm_mib=round(hwm(), 1), peak_rss='VmHWM of this process', guard_mib=C.GUARD_MIB)
    write_out(out, RES['tables'], RES['claims'], rec_fn)
    tick('write')
    log('RUN ' + _jl(dict(out=_rel(out), files=list(OUT_FILES), readings={cn: RES['READ'][cn] for cn in CN_NAMES}, rehearsal=bool(rehearsal),
                          cpu_s=dict(by_stage=stages, process_total=round(time.process_time(), 3)), vmhwm_mib=round(hwm(), 1))))
    return RES


def main(argv):
    need(len(argv) in (1, 2) and argv[0] == 'run' and set(argv[1:]) <= {'--dry'}, 'usage: competing_transition.py run [--dry]')
    execute(dry='--dry' in argv)
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
