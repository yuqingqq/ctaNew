"""LOGIC-CHAIN CORRECTION v1 - TERMINAL-WINDOW PUBLICATION AND TAKE RELABEL (P-2026-005 MAKER-TRACE logic-chain packet; session aea5bfb9
builder, 2026-11-23). Contract: chain_correction_contract.md sections 2 (PULL publication; review findings 3, 4), 3 (TAKE relabel; findings
2, 4, 6) and the section 5 falsifiers 'sign' and 'omitted control'; review 6bb458b30 = automatic/20261123_logic_chains_review/report.md.
SAVED EVIDENCE ONLY: reads v2/chains/pull/{summary,pulls,controls,census}.csv + receipt.json and v2/chains/take/{summary,takes,
take_controls,actions}.csv + receipt.json (pf_mt_common roots, MT_RUN=v2) and opens no other data path; reruns no lifecycle / context /
chain tool and builds no new null. DESCRIPTIVE on consumed days (Oct 24-27: four point estimates, G = 4 < 5: no interval); not a model, a
validation or a profitability claim; it measures no model skill. Every rule below is declared before any output of this tool exists.
BINDING  pull inputs: sha256 = the pull receipt's output hashes and rows = its row counts; take inputs: rows = the take receipt's counts and
       sha256 = the reviewed snapshot REVIEW_TAKE_SHA (the review's supplement.json; the take receipt binds no output hash, finding 7).
       A mismatch refuses.
TERMINAL WINDOWS (section 2). Published groups: cancel_possible -> 'cancel_compatible_terminal_window' (an EX | PL tie, never an
       identified cancellation), its successor <= 5 s split (a later first-filled same-side hash, never an identified replacement) and
       exhausted_proven (an S_FIFO-only class); S_FIFO primary, S_NOPRIO beside. Every other saved group stays in the original summary,
       unpublished here (listed in the receipt). pull_families.csv: per view x group x scope x day x D x feature BOTH declared reference
       families side by side - presence (the primary: a union of same-side certain-presence spans pooled with active-span fallbacks) and
       fill_anchored (the declared supplementary countercheck) - with their support (n_events, n_with_controls, n_obs, n_ctl_valid, n_reps),
       obs, reference mean / quantiles, diff and p_upper / p_lower_refdraw (= conditional reference-draw tail fractions over >= 200
       replicates, never independent-day confidence) and a family status (ok / reps_below_200 / no_observation / no_reference).
       OMITTED CONTROL: a published key lacking either family refuses the publication. pull_membership.csv: per view x group x scope x day
       the ok-window events, their presence membership (matched / active-span fallback / unavailable = controls.csv draw_status ok /
       fallback / no_eligible), fill_anchored membership (anchored / unavailable), the cross tab, and the reconciliation with the summary
       support and the census (ok-window orders); the successor split joins pulls.csv (successor != 'none'), whose draw statuses must equal
       controls.csv's.
PREDICATES (computed; section 2) at the receipt's headline (S_FIFO, D = 250 ms, pooled): (1) BTC against the order within 1 s
       (btc_against_1s_some) ENRICHED under both families, pooled and on every day; enriched = diff > 0 and p_upper <= ALPHA = 0.05 with
       >= 200 replicates (the conservative reading of an undefined word; diff > 0 alone is reported beside); (2) the inventory-built
       difference (inv_built_5s_some) > 0 on every day under fill_anchored (presence beside); (3) the cancel-compatible diff > the
       exhausted_proven diff on every day, per family and feature (BTC, inventory) and jointly. Sensitivity: the same at S_FIFO D = 0 /
       1000 / union and S_NOPRIO D = 250 (no exhausted_proven class there: (3) not evaluable, value False). A missing row makes a predicate
       False with the row named, never True.
TAKES (section 3). take_relabelled.csv: the saved take_vs_control rows; features renamed to the TAKE side (r_<w>_against ->
       btc_against_take_side_<w>, ...: chain_take.window_feats(..., s) receives s = the take side), inventory_reading 'withdrawn', the
       group's reference-match levels beside (full stratum / sign dropped = n_fallback1 / inventory dropped = n_fallback2 / no control / no
       claim). The reducing_vs_adding rows carry the same take-side alignment and are not republished (counted in the receipt).
       take_case_alignment.csv: CASES ONLY, NO MATCHED REFERENCE: per population (claim_all; claim_without_excluded_before) x scope x day x
       D x window (1 / 5 / 30 s) x class (take_class_<D>; plus reducing_all and all_classes) the window-category counts in take-side
       alignment and in row-level inventory alignment. Inventory side = the sign of the position held before the take: -take side for the
       reducing classes and overshoot (an opposing take), +take side for add; undetermined for timing_sensitive / from_flat / no_change /
       unclassified. take_excluded_before.csv: class counts (take_class = D 0) of the claim-eligible takes with and without an earlier
       excluded leg (their inventory classes are conditional on the retained ledger).
FALSIFIERS (section 5). sign: SIGN_CASES (a reducing sell of a long with BTC up is against the take side and with the inventory; an adding
       buy with BTC down is against both; ...) through the production helpers, each against the category of the position actually held,
       run before every publication; mutant MT_CC_MUT=flip_inventory_sign. Omitted control: above; mutant MT_CC_MUT=drop_fill_anchored.
       Label: the word 'pull' never labels a cancel-compatible row, an output column or a claims key / string (file names and the
       receipt's paths are exempt).
RECONCILIATIONS (each must hold or the run refuses; dry prints them): receipt rows; every saved cancel_possible row not identified, tied,
       EX|PL; membership = summary support = census; the saved obs of every BTC take_vs_control row recomputed exactly from takes.csv;
       class counts = take summary = actions; match levels constant per group, additive, equal to the take receipt's pools; excluded_before
       flags = n_excl_before > 0 = the receipt's count.
CLAIMS claims_pull_take.json: definitions, the computed predicates and every reviewer-number reproduction (recorded predicates, not
       refusals; tolerance = half the reviewer's last digit). RECEIPT receipt_pull_take.json: sha256 of the tool, its selftest, the
       contract, pf_mt_common, every input and every other output; checks, counts, git state, CPU by stage, VmHWM.
OUTPUTS C.RECEIPT / chains / correction_v1 (shared with chain_correct_post.py): per-file write-once - refused before any input is read when
       one of this tool's seven names exists; each file is hard-linked from a partial file, so an existing name is never replaced.
GUARDS MT_RUN=v2; a set MT_CC_MUT refuses (a planted-error switch never writes or prints a result); run refuses unless this tool, its
       selftest and the contract are tracked and unmodified (the freeze; dry records the state only); VmHWM self-guard
       pf_mt_common.GUARD_MIB; CPU self-guard CPU_CAP_S.
Usage (run_light, MT_RUN=v2): chain_correct_pull_take.py run | chain_correct_pull_take.py run --dry    selftest: chain_correct_pull_take_selftest.py"""
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RESEARCH = HERE.parents[2] / 'research'
for _p in (str(RESEARCH), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import pf_mt_common as C  # noqa: E402

need = C.need
NAN = float('nan')
VERSION = 'chain_correct_pull_take_v1'
CONTRACT = 'predictfun/reviews/maker_trace_20261122/chain_correction_contract.md (sections 2, 3; section 5 sign and omitted control)'
REVIEW = 'predictfun/reviews/automatic/20261123_logic_chains_review/report.md (6bb458b30), findings 2, 3, 4, 6'
CONTRACT_FILE = C.RECEIPT_V1 / 'chain_correction_contract.md'
TOOL_FILES = (HERE / 'chain_correct_pull_take.py', HERE / 'chain_correct_pull_take_selftest.py', CONTRACT_FILE)
DEP_FILES = (RESEARCH / 'pf_mt_common.py',)
PULL_DIR = C.RECEIPT / 'chains' / 'pull'
TAKE_DIR = C.RECEIPT / 'chains' / 'take'
OUT_DIR = C.RECEIPT / 'chains' / 'correction_v1'
PULL_FILES = ('summary.csv', 'pulls.csv', 'controls.csv', 'census.csv')
TAKE_FILES = ('summary.csv', 'takes.csv', 'take_controls.csv', 'actions.csv')
TABLES = ('pull_families', 'pull_membership', 'take_relabelled', 'take_case_alignment', 'take_excluded_before')
CLAIMS_FILE, RECEIPT_FILE = 'claims_pull_take.json', 'receipt_pull_take.json'
OUT_FILES = tuple(f'{t}.csv' for t in TABLES) + (CLAIMS_FILE, RECEIPT_FILE)
REVIEW_TAKE_SHA = {   # automatic/20261123_logic_chains_review/supplement.json take_output_hashes_review_snapshot (the reviewed bytes)
    'summary.csv': 'ac5b38aec75963a017b7d2a726041da02c99b2ae823901a3d79184020f0713b5',
    'actions.csv': '4dcbbdde133527cdc43001513eddf651d6343b06a328cbe138073c45d83f1814',
    'take_controls.csv': 'a38b93e7a0f24c6f7326f65dcd147eba508f1b32e466561f4e2f59bdfedca437',
    'takes.csv': '33442969f9da3fcd1133b210462258ad12c40b11e07e47383c65c0ab324a20ec'}
CPU_CAP_S = 30.

# ------------------------------------------------------------------ declared constants: terminal windows (section 2) ------------------------------------------------------------------
VIEWS = ('S_FIFO', 'S_NOPRIO')
VIEW_ROLE = {'S_FIFO': 'primary', 'S_NOPRIO': 'beside'}
FAMILIES = ('presence', 'fill_anchored')
PFX = {'presence': 'pres', 'fill_anchored': 'fa'}
CC_GROUPS = ('cancel_possible', 'cp_successor_5s', 'cp_no_successor_5s')
PUB_GROUPS = CC_GROUPS + ('exhausted_proven',)
CEN_GROUPS = ('cancel_possible', 'exhausted_proven')            # census classes (the successor split is not a census class)
EXPECTED_GROUPS = {'S_FIFO': PUB_GROUPS, 'S_NOPRIO': CC_GROUPS}  # exhausted_proven is an S_FIFO-only class (chain_pull CLASS rule)
GROUP_LABEL = {'cancel_possible': 'cancel_compatible_terminal_window', 'cp_successor_5s': 'cancel_compatible_successor_le5s',
               'cp_no_successor_5s': 'cancel_compatible_no_successor_le5s', 'exhausted_proven': 'exhausted_proven_reference'}
DAYS = tuple(C.MEASURE_DAYS)
DAYS_ALL = DAYS + ('all',)
SCOPES = ('rank1', 'rank2', 'rank3', 'pooled')
D_PULL = ('0', '250', '1000', 'union')
D_HEAD = '250'                                                  # the receipt's and the review's headline profile
BTC_F, INV_F, TAU_F = 'btc_against_1s_some', 'inv_built_5s_some', 'tau_start'
ALPHA, MIN_REPS = 0.05, 200
SENS = (('S_FIFO', '0'), ('S_FIFO', '1000'), ('S_FIFO', 'union'), ('S_NOPRIO', '250'))
SUM_KEY = ['view', 'group', 'scope', 'day', 'D', 'feature', 'kind']
STAT_COLS = ['n_with_controls', 'n_obs', 'n_ctl_valid', 'n_reps', 'obs', 'null_mean', 'null_p05', 'null_p50', 'null_p95', 'diff', 'p_upper', 'p_lower']
STAT_OUT = {'null_mean': 'ref_mean', 'null_p05': 'ref_p05', 'null_p50': 'ref_p50', 'null_p95': 'ref_p95', 'p_upper': 'p_upper_refdraw',
            'p_lower': 'p_lower_refdraw'}
SUM_COLS = SUM_KEY + ['control', 'n_events'] + STAT_COLS
CT_COLS = ['day', 'mid', 'scope', 'mk_hash', 'view', 'cls', 'control', 'draw_status', 'n_draws']
PL_COLS = ['day', 'mid', 'scope', 'mk_hash', 'view', 'window_status', 'successor', 'draws_presence', 'draws_fill_anchored', 'pull_identified',
           'identification', 'end_histories']
CEN_COLS = ['view', 'day', 'scope', 'cls', 'window_status', 'orders']
PRES_ST = (('ok', 'matched'), ('fallback', 'fallback'), ('no_eligible', 'unavailable'))
FA_ST = (('anchors', 'anchored'), ('none', 'unavailable'))
SUCCESSORS = ('none', 'refill', 'reprice', 'refill+reprice')

# ------------------------------------------------------------------ declared constants: takes (section 3) ------------------------------------------------------------------
D_TAKE = (0, 50, 250, 1000)
D_TAKE_HEAD = 0
WINDOWS = ('1s', '5s', '30s')
CLS = ('from_flat', 'add', 'reduce_partial', 'full_close', 'reversal_reducing', 'overshoot', 'no_change', 'unclassified', 'timing_sensitive')
REDUCING = ('reduce_partial', 'full_close', 'reversal_reducing')
INV_SIDE = {'reduce_partial': -1, 'full_close': -1, 'reversal_reducing': -1, 'overshoot': -1, 'add': 1}   # x take side; absent = undetermined
DIRCAT = ('with', 'against', 'flat', 'mixed', 'gap', 'missing')   # chain_take.DIRCAT (selftest-pinned)
UNDET = len(DIRCAT)                                               # the inventory code of an undetermined inventory side
MOVE_THR = {'r_1s': 0.5, 'r_5s': 1.0, 'r_30s': 2.0}               # chain_take.MOVES / contract 7.0 (selftest-pinned)
RENAME = {f'r_{w}_{a}': f'btc_{a}_take_side_{w}' for w in WINDOWS for a in ('with', 'against')}
RENAME.update({'r_5s_flat': 'btc_flat_5s', 'r_5s_mixed': 'btc_mixed_5s'})
BTC_CAT = {f'r_{w}_{a}': (w, a) for w in WINDOWS for a in ('with', 'against')}
BTC_CAT.update({'r_5s_flat': ('5s', 'flat'), 'r_5s_mixed': ('5s', 'mixed')})
TAKE_OTHER = ('burst3_certain', 'burst_possible_any', 'n_rec_1s', 'n_rec_5s', 'offset_sh_rec_5s', 'spread_start', 'depth_taken_start', 'tau_s',
              'pending_n', 'reduction_if_taken')
LEVEL_COLS = ['n_fallback1', 'n_fallback2', 'n_no_control', 'n_no_claim']
TK_COLS = (['day', 'mid', 'rank', 'sid', 's', 'claim', 'status', 'n_excl_before', 'take_class'] + [f'take_class_{D}' for D in D_TAKE]
           + [f'cat_r_{w}_{D}' for D in D_TAKE for w in WINDOWS])
TC_COLS = ['comparison', 'scope', 'day', 'D', 'feature', 'kind', 'n_takes'] + LEVEL_COLS + ['obs', 'null_mean', 'p05', 'p50', 'p95', 'p_hi', 'p_lo',
                                                                                          'n_draws', 'odds_ratio']
TS_COLS = (['scope', 'day', 'n_takes', 'n_claim'] + [f'cls_{c}' for c in CLS]
           + ['tvc_r_5s_against_obs', 'tvc_r_5s_against_null_mean', 'tvc_r_5s_against_n_takes'])
AC_COLS = ['action_type', 'scope', 'day', 'n']
CLASS_ROWS = tuple((c, (i,)) for i, c in enumerate(CLS)) + (('reducing_all', tuple(CLS.index(c) for c in REDUCING)),
                                                            ('all_classes', tuple(range(len(CLS)))))
POPS = ('claim_all', 'claim_without_excluded_before')
CASE_LABEL = 'cases_only_no_reference'
STR_DT = dict.fromkeys(['day', 'D', 'view', 'group', 'control', 'scope', 'feature', 'kind', 'mk_hash', 'cls', 'draw_status', 'window_status',
                        'successor', 'draws_presence', 'draws_fill_anchored', 'identification', 'end_histories', 'take_class', 'status',
                        'comparison', 'action_type', 'sid'] + TK_COLS[9:], str)
STR_DT.pop('D')                                                  # take_controls D is an integer; the pull summary reads its D as str below

# sign falsifier (section 5): (case, take side s, position before, take dq, class, BTC r_5s bp, expected take-side category, expected inventory)
SIGN_CASES = (
    ('a reducing sell of a long, BTC up', -1, 100., -40., 'reduce_partial', 2., 'against', 'with'),
    ('an adding buy to a long, BTC down', 1, 100., 40., 'add', -2., 'against', 'against'),
    ('a reducing buy of a short, BTC up', 1, -100., 40., 'reduce_partial', 2., 'with', 'against'),
    ('an adding sell to a short, BTC up', -1, -100., -40., 'add', 2., 'against', 'against'),
    ('an overshooting sell of a long, BTC up', -1, 100., -300., 'overshoot', 2., 'against', 'with'),
    ('a full close of a short by a buy, BTC down', 1, -100., 100., 'full_close', -2., 'against', 'with'))
LABEL_WORD = 'pull'

P1 = 'btc_against_1s_enriched_both_families_pooled_and_every_day'
P1S = 'btc_against_1s_diff_positive_both_families_pooled_and_every_day'
P2 = 'inventory_built_diff_positive_every_day_fill_anchored'
P2B = 'inventory_built_diff_positive_every_day_presence'
P3 = 'cancel_compatible_enrichment_exceeds_exhausted_every_day'
DEFINITIONS = dict(
    population='ex-ante top 3 (rank1-3) x the 19 BOOK markets, Oct 24-27 2026: 4 consumed days, point estimates, no interval (G = 4 < 5); '
               'filled orders are outcome-selected, take records are complete',
    cancel_compatible_terminal_window='the window [last fill hard low - D, removal hi] of a filled top-3 order whose end is an EX|PL tie '
                                      '(exhaustion and cancellation both feasible); every saved row is identification tied: never an '
                                      'identified cancellation or decision time',
    successor_le5s='a later first-filled same-side hash of the address within 5 s (x_links refill / reprice): not an identified '
                   'replacement after a cancellation',
    exhausted_proven_reference='S_FIFO removal set = {exhausted}: a priority-conditional reference whose window is fixed by the last fill '
                               'hard low to hi; order age, remaining size and remaining market time are not matched to the '
                               'cancel-compatible events; published separately, never pooled',
    families='presence = the declared primary reference: a union of same-side certain-presence spans (a relay of orders passes and the '
             'control uses the event order price) with active-span fallbacks pooled in its means; fill_anchored = the declared '
             'supplementary countercheck anchored at non-last fills of valid same-side orders; the two families support different event '
             'subsets (the membership table), not a same-row decomposition',
    p_refdraw='p_upper / p_lower_refdraw = (1 + #replicates >= / <= obs) / (1 + n) over >= 200 reference-draw replicates of the group '
              'statistic: a conditional reference-draw tail fraction, not independent-day confidence',
    enriched=f'diff > 0 and p_upper_refdraw <= {ALPHA} with n_reps >= {MIN_REPS}: this packet\'s conservative reading of the contract word '
             '(the diff-only reading is reported beside)',
    take_btc='btc_with / against_take_side_<w> = BTC with / against the TAKE side (chain_take.window_feats receives s = the take side in '
             'YES space); for a reducing take, BTC against the take side is BTC with the inventory held before it; the inventory-aligned '
             'reading of the saved statistic is withdrawn and no new null is built',
    take_reference='matched no-take instants of the same address and market: stratum = phase (pre / live, not remaining time), recorded '
                   'position sign, |position| bin; fallback1 drops the sign, fallback2 the |position| bin as well; every level enters '
                   'the saved estimates',
    take_case_alignment='cases only, no matched reference: per class, the BTC window category in take-side alignment and in row-level '
                        'inventory alignment; inventory side = the sign of the position held before the take: -take side for the '
                        'reducing classes and overshoot, +take side for add, undetermined for timing_sensitive / from_flat / no_change / '
                        'unclassified',
    take_excluded_before='inventory classes come from status-ok legs; a take with an earlier excluded leg is conditional on the retained '
                         'ledger (counted per class; every class table is also shown without these takes)',
    renames=RENAME)


def _mut(name):
    """The env-gated planted-error switch (selftest only), read at call time."""
    return os.environ.get('MT_CC_MUT', '') == name


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


def guard(where, lim=C.GUARD_MIB):
    h = hwm()
    need(h < lim, f'memory self-guard: VmHWM {h:.0f} MiB >= {lim:.0f} MiB at {where}')
    return h


def sha256(path):
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
    """A JSON-safe copy (NaN / inf -> None, numpy scalars -> Python, keys -> str)."""
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (bool, np.bool_)):
        return bool(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, (float, np.floating)):
        return float(o) if np.isfinite(o) else None
    if isinstance(o, Path):
        return str(o)
    return o


def _json_text(obj):
    return json.dumps(_clean(obj), indent=1, sort_keys=True, allow_nan=False) + '\n'


def _git(files):
    """HEAD and whether the tool files are tracked and unmodified (the freeze)."""
    try:
        g = ['git', '-C', str(C.REPO)]
        head = subprocess.run(g + ['rev-parse', 'HEAD'], capture_output=True, text=True, timeout=30).stdout.strip() or None
        rel = [_rel(f) for f in files]
        tracked = subprocess.run(g + ['ls-files', '--error-unmatch', '--'] + rel, capture_output=True, text=True, timeout=30).returncode == 0
        st = subprocess.run(g + ['status', '--porcelain', '--'] + rel, capture_output=True, text=True, timeout=30)
        return dict(head=head, tool_files=rel, tracked=tracked, unmodified=bool(tracked and st.returncode == 0 and not st.stdout.strip()))
    except (OSError, ValueError, subprocess.SubprocessError):
        return dict(head=None, tool_files=None, tracked=None, unmodified=None)


def _stager():
    c0 = time.process_time()
    st, last = {}, [c0]

    def tick(name):
        now = time.process_time()
        st[name] = round(st.get(name, 0.) + now - last[0], 3)
        last[0] = now
        need(now - c0 < CPU_CAP_S, f'CPU self-guard: {now - c0:.1f} s >= {CPU_CAP_S} s')
    return st, tick, c0


def chk(checks, name, ok, **info):
    checks[name] = dict(ok=bool(ok), **info)
    return bool(ok)


def _fin(x):
    try:
        return x is not None and bool(np.isfinite(x))
    except TypeError:
        return False


def _unknown(values, allowed):
    return sorted(str(x) for x in set(values) - set(allowed))


# ------------------------------------------------------------------ inputs ------------------------------------------------------------------
def read_saved(path, cols, dtype=None):
    """The named columns of a saved CSV: exact float parse ('round_trip'); only '' is missing, so labels such as 'none' / 'n/a' stay
    labels; refuses a missing column."""
    path = Path(path)
    need(path.is_file(), f'input missing: {path} (refused)')
    have = list(pd.read_csv(path, nrows=0).columns)
    miss = [c for c in cols if c not in have]
    need(not miss, f'{path.parent.name}/{path.name}: missing columns {miss} (refused)')
    use = [c for c in have if c in set(cols)]
    dt = {k: v for k, v in (dtype or {}).items() if k in use}
    return pd.read_csv(path, usecols=use, dtype=dt, float_precision='round_trip', keep_default_na=False, na_values=[''], low_memory=False)


def bind_and_read(pull_dir, take_dir, review_take_sha):
    """sha256 / bytes of every input, the receipt bindings (refusals), then the frames (named columns only)."""
    pull_dir, take_dir = Path(pull_dir), Path(take_dir)
    inputs = {}
    for d, names in ((pull_dir, PULL_FILES + ('receipt.json',)), (take_dir, TAKE_FILES + ('receipt.json',))):
        for f in names:
            p = d / f
            need(p.is_file(), f'input missing: {p} (refused)')
            inputs[_rel(p)] = dict(sha256=sha256(p), bytes=int(p.stat().st_size))
    pr = json.loads((pull_dir / 'receipt.json').read_text())
    tr = json.loads((take_dir / 'receipt.json').read_text())
    for f in PULL_FILES:
        exp = ((pr.get('outputs') or {}).get(f) or {}).get('sha256')
        got = inputs[_rel(pull_dir / f)]['sha256']
        need(exp is not None and got == exp, f'binding: pull {f} sha256 {got[:12]} differs from its receipt ({str(exp)[:12]}) (refused)')
    if review_take_sha is not None:
        for f in TAKE_FILES:
            got = inputs[_rel(take_dir / f)]['sha256']
            need(got == review_take_sha.get(f), f'binding: take {f} sha256 {got[:12]} differs from the reviewed snapshot '
                                                f'({str(review_take_sha.get(f))[:12]}) (refused)')
    pdt = dict(STR_DT, D=str)
    F = dict(S=read_saved(pull_dir / 'summary.csv', SUM_COLS, pdt), CT=read_saved(pull_dir / 'controls.csv', CT_COLS, pdt),
             PL=read_saved(pull_dir / 'pulls.csv', PL_COLS, pdt), CEN=read_saved(pull_dir / 'census.csv', CEN_COLS, pdt),
             TK=read_saved(take_dir / 'takes.csv', TK_COLS, STR_DT), TC=read_saved(take_dir / 'take_controls.csv', TC_COLS, STR_DT),
             TS=read_saved(take_dir / 'summary.csv', TS_COLS, STR_DT), AC=read_saved(take_dir / 'actions.csv', AC_COLS, STR_DT))
    for f, k in (('summary.csv', 'S'), ('controls.csv', 'CT'), ('pulls.csv', 'PL'), ('census.csv', 'CEN')):
        exp = (pr.get('rows') or {}).get(f[:-4])
        need(exp == len(F[k]), f'binding: pull {f} has {len(F[k])} rows, its receipt {exp} (refused)')
        inputs[_rel(pull_dir / f)]['rows'] = len(F[k])
    for f, k in (('summary.csv', 'TS'), ('takes.csv', 'TK'), ('take_controls.csv', 'TC'), ('actions.csv', 'AC')):
        exp = (tr.get('outputs') or {}).get(f[:-4])
        need(exp == len(F[k]), f'binding: take {f} has {len(F[k])} rows, its receipt {exp} (refused)')
        inputs[_rel(take_dir / f)]['rows'] = len(F[k])
    bindings = dict(pull_sha256_equal_pull_receipt=True, pull_rows_equal_pull_receipt=True, take_rows_equal_take_receipt=True,
                    take_sha256_equal_review_snapshot=(review_take_sha is not None))
    return F, inputs, tr, bindings


# ------------------------------------------------------------------ terminal windows (section 2) ------------------------------------------------------------------
def summary_vocab(S):
    for col, allowed in (('view', VIEWS), ('control', FAMILIES), ('scope', SCOPES), ('day', DAYS_ALL), ('D', D_PULL), ('kind', ('mean', 'rate'))):
        bad = _unknown(S[col].unique(), allowed)
        need(not bad, f'pull summary: unknown {col} values {bad[:5]} (refused)')


def fam_status(nwc, nobs, nreps):
    nwc, nobs, nreps = (np.asarray(x, float) for x in (nwc, nobs, nreps))
    return np.where(nwc <= 0, 'no_reference', np.where(nobs <= 0, 'no_observation', np.where(nreps < MIN_REPS, 'reps_below_200', 'ok')))


def pull_families(S):
    """Section 2: both declared reference families side by side for every published key (view x group x scope x day x D x feature x
    kind); refuses an omitted family, a duplicate row or an absent published group."""
    need(not _unknown(S['control'].unique(), FAMILIES), f"pull summary: unknown control families {_unknown(S['control'].unique(), FAMILIES)} (refused)")
    P = S[S['group'].isin(PUB_GROUPS)]
    if _mut('drop_fill_anchored'):                  # planted error (selftest only): the primary-only presentation of the old receipt
        P = P[P['control'] != 'fill_anchored']
    for v, gs in EXPECTED_GROUPS.items():
        for g in gs:
            need(bool(((P['view'] == v) & (P['group'] == g)).any()), f'pull summary: published group {g} absent under {v} (refused)')
    need(not bool(P.duplicated(SUM_KEY + ['control']).any()), 'pull summary: duplicate rows per key and family (refused)')
    fam = {f: P[P['control'] == f].set_index(SUM_KEY)[['n_events'] + STAT_COLS] for f in FAMILIES}
    a, b = fam['presence'].index, fam['fill_anchored'].index
    lack_f, lack_p = a.difference(b), b.difference(a)
    need(len(lack_f) == 0 and len(lack_p) == 0,
         f'omitted control: {len(lack_f)} published rows lack the fill_anchored family and {len(lack_p)} lack the presence family '
         f'(first: {list(lack_f[:1]) or list(lack_p[:1])}) (refused)')
    J = fam['presence'].add_prefix('pres_').join(fam['fill_anchored'].add_prefix('fa_'), how='inner')
    need(bool((J['pres_n_events'] == J['fa_n_events']).all()), 'pull summary: n_events differs between the two families of a key (refused)')
    J = J.reset_index()
    out = pd.DataFrame({'view': J['view'], 'view_role': J['view'].map(VIEW_ROLE), 'group': J['group'].map(GROUP_LABEL), 'group_saved': J['group'],
                        'scope': J['scope'], 'day': J['day'], 'D': J['D'], 'feature': J['feature'], 'kind': J['kind'], 'n_events': J['pres_n_events']})
    for f in FAMILIES:
        p = PFX[f]
        out[f'{p}_status'] = fam_status(J[f'{p}_n_with_controls'], J[f'{p}_n_obs'], J[f'{p}_n_reps'])
        for c in STAT_COLS:
            out[f'{p}_{STAT_OUT.get(c, c)}'] = J[f'{p}_{c}'].to_numpy()
    order = dict(_v=('view', VIEWS), _g=('group_saved', PUB_GROUPS), _s=('scope', SCOPES), _d=('day', DAYS_ALL), _D=('D', D_PULL))
    for k, (col, vals) in order.items():
        out[k] = out[col].map({x: i for i, x in enumerate(vals)}).to_numpy()
    return out.sort_values(['_v', '_g', '_s', '_d', '_D', 'feature'], kind='mergesort').drop(columns=list(order)).reset_index(drop=True)


def pull_membership(CT, PL, S, CEN):
    """Section 2: per view x published group x scope x day the ok-window events and their reference membership in both families (from
    controls.csv; the successor split through pulls.csv), reconciled with the summary support and the census."""
    need(not _unknown(CT['control'].unique(), FAMILIES), 'controls: unknown control family (refused)')
    for fam, sts in (('presence', PRES_ST), ('fill_anchored', FA_ST)):
        bad = _unknown(CT.loc[CT['control'] == fam, 'draw_status'].unique(), [c for c, _ in sts])
        need(not bad, f'controls: unknown {fam} draw status {bad} (refused)')
    ek = ['day', 'mid', 'mk_hash', 'view']
    need(not bool(CT.duplicated(ek + ['control']).any()), 'controls: duplicate event x family rows (refused)')
    parts = []
    for fam, p in (('presence', 'p'), ('fill_anchored', 'f')):
        x = CT.loc[CT['control'] == fam, ek + ['scope', 'cls', 'draw_status', 'n_draws']]
        parts.append(x.rename(columns={'draw_status': f'st_{p}', 'n_draws': f'nd_{p}'}))
    E = parts[0].merge(parts[1], on=ek + ['scope', 'cls'], how='outer', validate='one_to_one', indicator=True)
    need(bool((E['_merge'] == 'both').all()), 'controls: an event lacks one reference family row, or its scope / class differs between them (refused)')
    E = E.drop(columns=['_merge'])
    need(not bool(PL.duplicated(ek).any()), 'pulls: duplicate order x view rows (refused)')
    bad = _unknown(PL['successor'].unique(), SUCCESSORS)
    need(not bad, f'pulls: unknown successor values {bad} (refused)')
    cc = E[E['cls'] == 'cancel_possible']
    PLo = PL.loc[PL['window_status'] == 'ok', ek + ['successor', 'draws_presence', 'draws_fill_anchored']]
    M = cc.merge(PLo, on=ek, how='outer', validate='one_to_one', indicator=True)
    need(bool((M['_merge'] == 'both').all()), 'controls / pulls: the ok-window cancel-compatible events differ between the two files (refused)')
    need(bool((M['draws_presence'] == M['st_p']).all()) and bool((M['draws_fill_anchored'] == M['st_f']).all()),
         'controls / pulls: the draw statuses of an event disagree (refused)')
    succ = (M['successor'] != 'none').to_numpy(bool)
    frames = {'cancel_possible': M, 'cp_successor_5s': M[succ], 'cp_no_successor_5s': M[~succ], 'exhausted_proven': E[E['cls'] == 'exhausted_proven']}
    P = S[S['group'].isin(PUB_GROUPS)]
    agg = P.groupby(['view', 'group', 'control', 'scope', 'day'])[['n_events', 'n_with_controls']].agg(['min', 'max'])
    need(bool((agg[('n_events', 'min')] == agg[('n_events', 'max')]).all() and (agg[('n_with_controls', 'min')] == agg[('n_with_controls', 'max')]).all()),
         'pull summary: the support differs across the D / feature rows of one group (refused)')
    sup = {k: (int(r[('n_events', 'min')]), int(r[('n_with_controls', 'min')])) for k, r in agg.iterrows()}
    cen = {k: int(v) for k, v in CEN[CEN['window_status'] == 'ok'].groupby(['view', 'day', 'scope', 'cls'])['orders'].sum().items()}
    rows = []
    for v in VIEWS:
        for g in EXPECTED_GROUPS[v]:
            X = frames[g]
            X = X[X['view'] == v]
            sc_, d_ = X['scope'].to_numpy(object), X['day'].to_numpy(object)
            sp, sf = X['st_p'].to_numpy(object), X['st_f'].to_numpy(object)
            ndp, ndf = X['nd_p'].to_numpy(np.int64), X['nd_f'].to_numpy(np.int64)
            for sc in SCOPES:
                ms = np.ones(len(X), bool) if sc == 'pooled' else (sc_ == sc)
                for d in DAYS_ALL:
                    m = (ms & (d_ == d)) if d != 'all' else ms
                    r = dict(view=v, view_role=VIEW_ROLE[v], group=GROUP_LABEL[g], group_saved=g, scope=sc, day=d, n_events=int(m.sum()))
                    for code, lab in PRES_ST:
                        r[f'pres_{lab}'] = int((m & (sp == code)).sum())
                    for code, lab in FA_ST:
                        r[f'fa_{lab}'] = int((m & (sf == code)).sum())
                    r['pres_with_draws'], r['fa_with_draws'] = int((m & (ndp > 0)).sum()), int((m & (ndf > 0)).sum())
                    for cp, lp in PRES_ST:
                        for cf, lf in FA_ST:
                            r[f'x_pres_{lp}_fa_{lf}'] = int((m & (sp == cp) & (sf == cf)).sum())
                    se, sfa = sup.get((v, g, 'presence', sc, d)), sup.get((v, g, 'fill_anchored', sc, d))
                    r['summary_n_events'] = se[0] if se else 0
                    r['summary_pres_n_with_controls'] = se[1] if se else 0
                    r['summary_fa_n_with_controls'] = sfa[1] if sfa else 0
                    r['census_ok_orders'] = float(cen.get((v, d, sc, g), 0)) if g in CEN_GROUPS else NAN
                    r['reconciled'] = bool(r['n_events'] == r['summary_n_events'] and (se is None) == (sfa is None)
                                           and r['pres_with_draws'] == r['summary_pres_n_with_controls'] and r['fa_with_draws'] == r['summary_fa_n_with_controls']
                                           and (g not in CEN_GROUPS or r['census_ok_orders'] == r['n_events']))
                    rows.append(r)
    return pd.DataFrame(rows)


def pooled_index(S):
    """(view, group, family, day, D, feature) -> the pooled summary row (published groups, the predicate / reproduction features)."""
    q = S[(S['scope'] == 'pooled') & S['group'].isin(PUB_GROUPS) & S['feature'].isin([BTC_F, INV_F, TAU_F])]
    out = {}
    for r in q.to_dict('records'):
        k = (r['view'], r['group'], r['control'], r['day'], r['D'], r['feature'])
        need(k not in out, f'pull summary: duplicate row {k} (refused)')
        out[k] = r
    return out


def _brief(r):
    if r is None:
        return None
    return dict(n_events=int(r['n_events']), n_with_controls=int(r['n_with_controls']), n_obs=int(r['n_obs']), n_reps=int(r['n_reps']), obs=r['obs'],
                ref_mean=r['null_mean'], diff=r['diff'], p_upper_refdraw=r['p_upper'])


def enriched(r):
    return bool(r is not None and r['n_obs'] > 0 and r['n_reps'] >= MIN_REPS and _fin(r['diff']) and r['diff'] > 0 and _fin(r['p_upper'])
                and r['p_upper'] <= ALPHA)


def positive(r):
    return bool(r is not None and r['n_obs'] > 0 and _fin(r['diff']) and r['diff'] > 0)


def predicates(I, view, D):
    """The section 2 predicates for one view and D profile (pooled scope); computed from the saved rows, a missing row -> False."""
    def get(g, f, d, ft):
        return I.get((view, g, f, d, D, ft))
    out = {}
    btc = {f: {d: get('cancel_possible', f, d, BTC_F) for d in DAYS_ALL} for f in FAMILIES}
    miss = [f'{f}/{d}' for f in FAMILIES for d in DAYS_ALL if btc[f][d] is None]
    ev = {f: {d: dict(_brief(btc[f][d]) or {}, enriched=enriched(btc[f][d])) for d in DAYS_ALL} for f in FAMILIES}
    out[P1] = dict(value=bool(not miss and all(enriched(btc[f][d]) for f in FAMILIES for d in DAYS_ALL)), missing_rows=miss,
                   rule=f'cancel-compatible {BTC_F}: diff > 0 and p_upper_refdraw <= {ALPHA} with n_reps >= {MIN_REPS} under presence AND '
                        f'fill_anchored, day all and each of {len(DAYS)} days', by_family_day=ev)
    out[P1S] = dict(value=bool(not miss and all(positive(btc[f][d]) for f in FAMILIES for d in DAYS_ALL)), missing_rows=miss,
                    rule='as above with diff > 0 only (beside)')
    for f, name, role in (('fill_anchored', P2, 'declared'), ('presence', P2B, 'beside')):
        inv = {d: get('cancel_possible', f, d, INV_F) for d in DAYS}
        missf = [d for d in DAYS if inv[d] is None]
        out[name] = dict(value=bool(not missf and all(positive(inv[d]) for d in DAYS)), missing_rows=missf, role=role,
                         rule=f'cancel-compatible {INV_F} under {f}: diff > 0 on each of {len(DAYS)} days', by_day={d: _brief(inv[d]) for d in DAYS})
    ex_defined = any(k[0] == view and k[1] == 'exhausted_proven' for k in I)
    per, allv = {}, True
    for f in FAMILIES:
        per[f] = {}
        for ft in (BTC_F, INV_F):
            days = {}
            for d in DAYS:
                a, b = get('cancel_possible', f, d, ft), get('exhausted_proven', f, d, ft)
                ok = a is not None and b is not None and _fin(a['diff']) and _fin(b['diff']) and a['diff'] > b['diff']
                days[d] = dict(cancel_compatible_diff=None if a is None else a['diff'], exhausted_diff=None if b is None else b['diff'], value=bool(ok))
            per[f][ft] = dict(value=bool(all(x['value'] for x in days.values())), by_day=days)
            allv = allv and per[f][ft]['value']
    out[P3] = dict(value=bool(ex_defined and allv), evaluable=bool(ex_defined),
                   status='ok' if ex_defined else 'not evaluable: exhausted_proven is an S_FIFO-only class',
                   rule=f'cancel-compatible diff > exhausted_proven diff on each of {len(DAYS)} days, per family and feature ({BTC_F}, {INV_F})',
                   by_family_feature=per)
    return out


def _rep(id_, reviewer, computed, tol, unit, source='review'):
    c = None if computed is None else float(computed)
    ok = c is not None and np.isfinite(c) and abs(c - reviewer) <= tol + 1e-9
    return dict(id=id_, source=source, unit=unit, reviewer=reviewer, computed=c, tolerance=tol, match=bool(ok))


def terminal_reproductions(I, mem):
    items = []

    def pc(r, k):
        return None if r is None else 100. * r[k]
    for fam, ft, tag, rv in (('presence', BTC_F, 'btc_against_1s', (34.62, 17.70, 16.93, 2715)), ('presence', INV_F, 'inventory_built_5s', (87.56, 71.02, 16.55, 2734)),
                             ('fill_anchored', BTC_F, 'btc_against_1s', (28.34, 19.27, 9.06, 1401)),
                             ('fill_anchored', INV_F, 'inventory_built_5s', (86.15, 78.80, 7.35, 1408))):
        r = I.get(('S_FIFO', 'cancel_possible', fam, 'all', D_HEAD, ft))
        b = f'{fam}_{tag}_pooled_all'
        items += [_rep(f'{b}_obs_pct', rv[0], pc(r, 'obs'), .005, 'percent'), _rep(f'{b}_reference_pct', rv[1], pc(r, 'null_mean'), .005, 'percent'),
                  _rep(f'{b}_diff_pp', rv[2], pc(r, 'diff'), .005, 'pp'), _rep(f'{b}_n_obs', rv[3], None if r is None else r['n_obs'], 0, 'events')]
    for ft, tag, diffs, ns in ((BTC_F, 'btc_against_1s', (9.70, 9.38, 10.43, 4.17), (501, 312, 407, 181)),
                               (INV_F, 'inventory_built_5s', (1.05, -7.77, 20.34, 22.24), (501, 319, 407, 181))):
        for d, dv, nv in zip(DAYS, diffs, ns):
            r = I.get(('S_FIFO', 'cancel_possible', 'fill_anchored', d, D_HEAD, ft))
            items += [_rep(f'fill_anchored_{tag}_{d}_diff_pp', dv, pc(r, 'diff'), .005, 'pp'),
                      _rep(f'fill_anchored_{tag}_{d}_n_obs', nv, None if r is None else r['n_obs'], 0, 'events')]
    for grp, tag, rv in (('exhausted_proven', 'exhausted', 24.84), ('cancel_possible', 'cancel_compatible', 4.17)):
        r = I.get(('S_FIFO', grp, 'fill_anchored', DAYS[-1], D_HEAD, BTC_F))
        items.append(_rep(f'fill_anchored_btc_against_1s_{DAYS[-1]}_{tag}_diff_pp', rv, pc(r, 'diff'), .005, 'pp'))
    m = mem[(mem['view'] == 'S_FIFO') & (mem['group_saved'] == 'cancel_possible') & (mem['scope'] == 'pooled') & (mem['day'] == 'all')]
    m = m.iloc[0] if len(m) == 1 else None
    for col, rv in (('n_events', 2741), ('pres_matched', 908), ('pres_fallback', 1826), ('pres_unavailable', 7)):
        items.append(_rep(f'presence_membership_cancel_compatible_{col}', rv, None if m is None else m[col], 0, 'events'))
    r = I.get(('S_FIFO', 'cancel_possible', 'presence', 'all', D_HEAD, TAU_F))
    items += [_rep('presence_tau_start_obs_s', 154, None if r is None else r['obs'], .5, 's'),
              _rep('presence_tau_start_reference_s', 173, None if r is None else r['null_mean'], .5, 's')]
    s5 = mem[(mem['view'] == 'S_FIFO') & (mem['group_saved'] == 'cp_successor_5s') & (mem['scope'] == 'pooled') & (mem['day'] == 'all')]
    items.append(_rep('successor_le5s_events', 1573, s5['n_events'].iloc[0] if len(s5) == 1 else None, 0, 'events', source='receipt'))
    return items


# ------------------------------------------------------------------ takes (section 3) ------------------------------------------------------------------
def take_vocab(TK, TC):
    for D in D_TAKE:
        bad = _unknown(TK[f'take_class_{D}'].unique(), CLS)
        need(not bad, f'takes: unknown take_class_{D} values {bad} (refused)')
        for w in WINDOWS:
            bad = _unknown(TK[f'cat_r_{w}_{D}'].unique(), DIRCAT)
            need(not bad, f'takes: unknown cat_r_{w}_{D} values {bad} (refused)')
    need(not _unknown(TK['take_class'].unique(), CLS), 'takes: unknown take_class values (refused)')
    need(not _unknown(TK['rank'].unique(), (1, 2, 3)), 'takes: rank outside 1-3 (refused)')
    need(not _unknown(TK['day'].unique(), DAYS), 'takes: a day outside the measurement days (refused)')
    need(not _unknown(TK['s'].unique(), (-1, 1)), 'takes: a take side outside +-1 (refused)')
    need(not _unknown(TK['claim'].astype(str).unique(), ('True', 'False')), 'takes: claim is not True / False (refused)')
    need(not _unknown(TC['comparison'].unique(), ('take_vs_control', 'reducing_vs_adding')), 'take_controls: unknown comparison (refused)')
    bad = _unknown(TC['feature'].unique(), tuple(RENAME) + TAKE_OTHER)
    need(not bad, f'take_controls: unknown features {bad} (refused)')


def take_groups(TK):
    """(scope, day) -> boolean mask over takes: rank1-3 and pooled x each day and all."""
    rank, day = TK['rank'].to_numpy(np.int64), TK['day'].to_numpy(object)
    G = {}
    for sc in SCOPES:
        ms = np.ones(len(TK), bool) if sc == 'pooled' else (rank == int(sc[4:]))
        for d in DAYS_ALL:
            G[(sc, d)] = (ms & (day == d)) if d != 'all' else ms
    return G


def take_relabel(TC, G, checks):
    """Section 3: the saved take_vs_control rows, renamed to the take side, with the group's reference-match levels beside."""
    T = TC[TC['comparison'] == 'take_vs_control'].reset_index(drop=True)
    n_rva = int((TC['comparison'] == 'reducing_vs_adding').sum())
    keys = list(zip(T['scope'].to_numpy(object), T['day'].to_numpy(object)))
    bad = sorted({f'{a}|{b}' for a, b in keys if (a, b) not in G})
    need(not bad, f'take_controls: a scope x day outside rank1-3 / pooled x days / all: {bad[:3]} (refused)')
    gsize = {k: int(v.sum()) for k, v in G.items()}
    grp = np.array([gsize[k] for k in keys], np.int64)
    lv = T[LEVEL_COLS].to_numpy(float)
    need(bool(np.isfinite(lv).all()), 'take_controls: a take_vs_control row without its match-level counts (refused)')
    const = T.groupby(['scope', 'day'])[LEVEL_COLS].nunique()
    chk(checks, 'take_match_levels_constant_per_group', bool((const.to_numpy() <= 1).all()))
    full = grp - lv.sum(1)
    chk(checks, 'take_match_levels_nonnegative', bool((full >= 0).all()))
    binr = (T['kind'] == 'binary').to_numpy(bool)
    nt = T['n_takes'].to_numpy(np.int64)
    chk(checks, 'take_binary_n_takes_reconcile', bool((nt[binr] == (grp - lv[:, 2] - lv[:, 3])[binr]).all()), n_rows=int(binr.sum()))
    fs = T['feature'].to_numpy(object)
    isb = np.array([f in RENAME for f in fs], bool)
    out = pd.DataFrame({'comparison': T['comparison'].to_numpy(object), 'reference': 'matched_no_take_instants', 'scope': T['scope'].to_numpy(object),
                        'day': T['day'].to_numpy(object), 'D': T['D'].to_numpy(np.int64), 'feature': [RENAME.get(f, f) for f in fs], 'feature_saved': fs,
                        'alignment': np.where(isb, 'take_side', 'not_btc'), 'inventory_reading': np.where(isb, 'withdrawn', 'not_applicable'),
                        'kind': T['kind'].to_numpy(object), 'n_takes': nt, 'grp_n_takes': grp, 'n_full_stratum': full.astype(np.int64),
                        'n_sign_dropped': lv[:, 0].astype(np.int64), 'n_inventory_dropped': lv[:, 1].astype(np.int64),
                        'n_no_control': lv[:, 2].astype(np.int64), 'n_no_claim': lv[:, 3].astype(np.int64)})
    for c in ('obs', 'null_mean', 'p05', 'p50', 'p95', 'p_hi', 'p_lo', 'n_draws', 'odds_ratio'):
        out[c] = T[c].to_numpy()
    return out, n_rva


def take_obs_reconcile(TC, TK, G, claim):
    """Every BTC take_vs_control row whose group has no no-control / no-claim take: obs = the take-side category share recomputed from
    takes.csv (exact); the saved statistic is the take-side one the rename names."""
    cats = {(w, D): TK[f'cat_r_{w}_{D}'].to_numpy(object) for D in D_TAKE for w in WINDOWS}
    T = TC[(TC['comparison'] == 'take_vs_control') & TC['feature'].isin(list(BTC_CAT))]
    n_chk, n_skip, bad = 0, 0, []
    for r in T.to_dict('records'):
        if r['n_no_control'] > 0 or r['n_no_claim'] > 0:
            n_skip += 1
            continue
        w, val = BTC_CAT[r['feature']]
        x = cats[(w, int(r['D']))][G[(r['scope'], r['day'])] & claim] == val
        n_chk += 1
        if len(x) != int(r['n_takes']):
            bad.append(dict(scope=r['scope'], day=r['day'], D=int(r['D']), feature=r['feature'], n=len(x), n_saved=int(r['n_takes'])))
            continue
        obs = float(x.mean()) if len(x) else NAN
        same = (not len(x) and not _fin(r['obs'])) or (len(x) and _fin(r['obs']) and abs(obs - r['obs']) <= 1e-12)
        if not same:
            bad.append(dict(scope=r['scope'], day=r['day'], D=int(r['D']), feature=r['feature'], obs=obs, obs_saved=r['obs']))
    return dict(n_rows_checked=n_chk, n_rows_skipped_unreconstructible=n_skip, n_mismatch=len(bad), first_mismatches=bad[:3])


def take_class_reconcile(TK, TS, AC, G):
    """take_class counts per scope x day = the take summary's cls_* and n_takes = the actions' take_<class> n."""
    cls = TK['take_class'].to_numpy(object)
    bad, n_act = [], 0
    for r in TS.to_dict('records'):
        k = (r['scope'], r['day'])
        if k not in G:
            bad.append(dict(summary_group=k))
            continue
        m = G[k]
        if int(m.sum()) != int(r['n_takes']):
            bad.append(dict(summary_group=k, n_takes=int(m.sum()), saved=int(r['n_takes'])))
        for c in CLS:
            if int((m & (cls == c)).sum()) != int(r[f'cls_{c}']):
                bad.append(dict(summary_group=k, cls=c, n=int((m & (cls == c)).sum()), saved=int(r[f'cls_{c}'])))
    for r in AC.to_dict('records'):
        t = str(r['action_type'])
        if not t.startswith('take_') or t[5:] not in CLS:
            continue
        k = (r['scope'], r['day'])
        if k not in G:
            bad.append(dict(action_group=k))
            continue
        n_act += 1
        if int((G[k] & (cls == t[5:])).sum()) != int(r['n']):
            bad.append(dict(action_group=k, action=t, n=int((G[k] & (cls == t[5:])).sum()), saved=int(r['n'])))
    return dict(n_summary_rows=len(TS), n_action_rows=n_act, n_mismatch=len(bad), first_mismatches=bad[:3])


def take_summary_vs_controls(TS, TC):
    """The take summary's tvc_r_5s_against (D 0) = its take_controls row (the statistic the receipt quoted)."""
    T = TC[(TC['comparison'] == 'take_vs_control') & (TC['feature'] == 'r_5s_against') & (TC['D'] == D_TAKE_HEAD)]
    idx = {(r['scope'], r['day']): r for r in T.to_dict('records')}

    def same(a, b):
        return (not _fin(a) and not _fin(b)) or (_fin(a) and _fin(b) and abs(a - b) <= 1e-12)
    bad, n = [], 0
    for r in TS.to_dict('records'):
        c = idx.get((r['scope'], r['day']))
        if c is None:
            if _fin(r['tvc_r_5s_against_obs']):
                bad.append(dict(group=(r['scope'], r['day']), reason='summary value without a take_controls row'))
            continue
        n += 1
        ok = (same(r['tvc_r_5s_against_obs'], c['obs']) and same(r['tvc_r_5s_against_null_mean'], c['null_mean'])
              and _fin(r['tvc_r_5s_against_n_takes']) and int(r['tvc_r_5s_against_n_takes']) == int(c['n_takes']))
        if not ok:
            bad.append(dict(group=(r['scope'], r['day'])))
    return dict(n_rows_compared=n, n_mismatch=len(bad), first_mismatches=bad[:3])


def take_match_levels(rel, tr):
    """Per scope x day the match levels (from the relabelled rows); additivity days -> all -> pooled; the pooled row = the receipt pools."""
    cols = ('grp_n_takes', 'n_full_stratum', 'n_sign_dropped', 'n_inventory_dropped', 'n_no_control', 'n_no_claim')
    L = {(r['scope'], r['day']): {c: int(r[c]) for c in cols} for r in rel.drop_duplicates(['scope', 'day']).to_dict('records')}
    add_ok = True
    for rk in ('rank1', 'rank2', 'rank3'):
        parts, a = [L.get((rk, d)) for d in DAYS], L.get((rk, 'all'))
        add_ok = add_ok and a is not None and all(x is not None for x in parts) and all(a[c] == sum(x[c] for x in parts) for c in cols)
    p = L.get(('pooled', 'all'))
    ranks = [L.get((rk, 'all')) for rk in ('rank1', 'rank2', 'rank3')]
    add_ok = add_ok and p is not None and all(x is not None for x in ranks) and all(p[c] == sum(x[c] for x in ranks) for c in cols)
    pools = (tr.get('pools') or {}).get('pool_level') or {}
    rec_ok = bool(p is not None and p['n_full_stratum'] == pools.get('stratum') and p['n_sign_dropped'] == pools.get('fallback1')
                  and p['n_inventory_dropped'] == pools.get('fallback2') and p['n_no_control'] == pools.get('no_control')
                  and p['n_no_claim'] == pools.get('no_claim'))
    return {f'{k[0]}|{k[1]}': v for k, v in L.items()}, bool(add_ok), rec_ok


def btc_side_code(side, r, thr):
    """chain_context 7.0 / chain_take convention: with iff side x r >= +thr, against iff side x r <= -thr, else flat (side +1 = buy YES)."""
    x = np.asarray(side, float) * np.asarray(r, float)
    return np.where(x >= thr, 0, np.where(x <= -thr, 1, 2)).astype(np.int64)


def inventory_multiplier(cls):
    """Per take: +1 = the position held before the take is on the take side (add), -1 = opposite (the reducing classes and overshoot),
    0 = undetermined (timing_sensitive, from_flat, no_change, unclassified)."""
    m = np.array([INV_SIDE.get(c, 0) for c in cls], np.int64)
    return -m if _mut('flip_inventory_sign') else m      # planted error (selftest only): the global sign flip the review rejected


def inventory_code(tcode, mult):
    """Take-side category code -> inventory-side code: unchanged when the inventory side is the take side, with <-> against swapped when
    it is opposite, UNDET when undetermined (flat / mixed / gap / missing have no side)."""
    t, m = np.asarray(tcode, np.int64), np.asarray(mult, np.int64)
    sw = np.where(t == 0, 1, np.where(t == 1, 0, t))
    return np.where(m == 0, UNDET, np.where(m > 0, t, sw)).astype(np.int64)


def sign_falsifier():
    """Section 5 'sign' on SIGN_CASES through the production helpers: the inventory category from the class rule must equal the category of
    the position actually held (the sign of the position before the take) and the declared expectation -> failures (empty = pass)."""
    thr = MOVE_THR['r_5s']
    fails = []
    for name, s, before, dq, cls, r, e_take, e_inv in SIGN_CASES:
        t = int(btc_side_code(s, r, thr))
        i = int(inventory_code(np.array([t]), inventory_multiplier([cls]))[0])
        held = int(btc_side_code(np.sign(before), r, thr))
        ok = int(np.sign(dq)) == s and DIRCAT[t] == e_take and i < UNDET and DIRCAT[i] == e_inv and i == held
        if not ok:
            fails.append(dict(case=name, take_side=DIRCAT[t], inventory=DIRCAT[i] if i < UNDET else 'undetermined', held=DIRCAT[held]))
    return fails


def _side_label(name):
    if name == 'reducing_all':
        return '-take_side'
    if name == 'all_classes':
        return 'row_level'
    v = INV_SIDE.get(name)
    return '+take_side' if v == 1 else '-take_side' if v == -1 else 'undetermined'


def take_case_alignment(TK, G, claim):
    """Section 3 cases-only rates: per population x scope x day x D x window x class the BTC window categories in take-side and in row-level
    inventory alignment (class basis take_class_<D>)."""
    excl = TK['n_excl_before'].to_numpy(np.int64) > 0
    pops = ((POPS[0], claim), (POPS[1], claim & ~excl))
    ci, di = {c: i for i, c in enumerate(CLS)}, {c: i for i, c in enumerate(DIRCAT)}
    nC, nK = len(CLS), len(DIRCAT)
    lab = [(nm, list(sel), _side_label(nm)) for nm, sel in CLASS_ROWS]
    rows = []
    for D in D_TAKE:
        cls = TK[f'take_class_{D}'].to_numpy(object)
        cc = np.array([ci[c] for c in cls], np.int64)
        mult = inventory_multiplier(cls)
        for w in WINDOWS:
            tc = np.array([di[c] for c in TK[f'cat_r_{w}_{D}'].to_numpy(object)], np.int64)
            ic = inventory_code(tc, mult)
            for pop, pm in pops:
                for (sc, d), gm in G.items():
                    m = pm & gm
                    bt = np.bincount(cc[m] * nK + tc[m], minlength=nC * nK).reshape(nC, nK)
                    bi = np.bincount(cc[m] * (nK + 1) + ic[m], minlength=nC * (nK + 1)).reshape(nC, nK + 1)
                    for nm, sel, side in lab:
                        t, i = bt[sel].sum(0), bi[sel].sum(0)
                        n, ndet = int(t.sum()), int(i[:nK].sum())
                        rows.append(dict(population=pop, scope=sc, day=d, D=D, window=w, take_class=nm, class_basis=f'take_class_{D}', label=CASE_LABEL,
                                         n_takes=n, n_btc_with_take_side=int(t[0]), n_btc_against_take_side=int(t[1]), n_flat=int(t[2]),
                                         n_mixed=int(t[3]), n_gap=int(t[4]), n_missing=int(t[5]),
                                         rate_btc_with_take_side=t[0] / n if n else NAN, rate_btc_against_take_side=t[1] / n if n else NAN,
                                         inventory_side=side, n_inventory_determined=ndet, n_btc_with_inventory=int(i[0]),
                                         n_btc_against_inventory=int(i[1]), rate_btc_with_inventory=i[0] / ndet if ndet else NAN,
                                         rate_btc_against_inventory=i[1] / ndet if ndet else NAN))
    return pd.DataFrame(rows)


def take_excluded_before(TK, G, claim):
    """Section 3: claim-eligible take counts per scope x day x class (take_class, D 0) with and without an earlier excluded leg."""
    cls = TK['take_class'].to_numpy(object)
    excl = TK['n_excl_before'].to_numpy(np.int64) > 0
    rows = []
    for (sc, d), gm in G.items():
        for c in CLS + ('all_classes',):
            m = gm & claim & ((cls == c) if c != 'all_classes' else True)
            n, ne = int(m.sum()), int((m & excl).sum())
            rows.append(dict(scope=sc, day=d, take_class=c, class_basis='take_class_0', n_claim_eligible=n, n_with_excluded_before=ne,
                             n_without_excluded_before=n - ne,
                             inventory_class_statement='conditional_on_retained_ledger' if ne else 'no_earlier_excluded_leg'))
    return pd.DataFrame(rows)


def take_reproductions(rel, xb):
    items = []
    r = rel[(rel['scope'] == 'rank1') & (rel['day'] == 'all') & (rel['D'] == D_TAKE_HEAD) & (rel['feature_saved'] == 'r_5s_against')]
    r = r.iloc[0] if len(r) == 1 else None
    items += [_rep('take_rank1_btc_against_take_side_5s_obs_pct', 5.055, None if r is None else 100. * r['obs'], .0005, 'percent'),
              _rep('take_rank1_btc_against_take_side_5s_reference_pct', 2.686, None if r is None else 100. * r['null_mean'], .0005, 'percent')]
    p = rel[(rel['scope'] == 'pooled') & (rel['day'] == 'all')]
    p = p.iloc[0] if len(p) else None
    for col, rv in (('n_full_stratum', 3977), ('n_sign_dropped', 125), ('n_inventory_dropped', 45), ('grp_n_takes', 4147)):
        items.append(_rep(f'take_match_{col}', rv, None if p is None else p[col], 0, 'takes'))
    x = xb[(xb['scope'] == 'pooled') & (xb['day'] == 'all')]
    xm = dict(zip(x['take_class'], x['n_with_excluded_before']))
    for c, rv in (('all_classes', 248), ('reduce_partial', 107), ('add', 93), ('timing_sensitive', 46), ('no_change', 2)):
        items.append(_rep(f'take_excluded_before_{c}', rv, xm.get(c), 0, 'takes'))
    return items


NOT_REPRODUCIBLE = (dict(id='excluded_top3_book_legs', reviewer='16 excluded top-three BOOK legs: 14 source conflict, 2 off_grid',
                         status='not_reproducible_from_declared_inputs', reason='counted from econ/fills_valued.parquet, which this tool does not open'),)


# ------------------------------------------------------------------ label guard ------------------------------------------------------------------
def label_hits(tables, claims):
    """Section 2 wording: the word never labels an output column, any cell of a cancel-compatible row (group_saved in CC_GROUPS) or any
    claims key / string -> the hits."""
    hits = []
    for nm, df in tables.items():
        hits += [f'{nm}: column {c}' for c in df.columns if LABEL_WORD in str(c).lower()]
        if 'group_saved' not in df.columns:
            continue
        sub = df[df['group_saved'].isin(CC_GROUPS)]
        for c in sub.columns:
            if pd.api.types.is_numeric_dtype(sub[c]):
                continue
            hit = sub[c].astype(str).str.lower().str.contains(LABEL_WORD, regex=False, na=False)
            if bool(hit.any()):
                hits.append(f'{nm}: a cancel-compatible row labels {c} = {sub[c][hit].iloc[0]!r}')

    def walk(o, path):
        if isinstance(o, dict):
            for k, v in o.items():
                if LABEL_WORD in str(k).lower():
                    hits.append(f'claims: key {path}/{k}')
                walk(v, f'{path}/{k}')
        elif isinstance(o, (list, tuple)):
            for j, v in enumerate(o):
                walk(v, f'{path}[{j}]')
        elif isinstance(o, str) and LABEL_WORD in o.lower():
            hits.append(f'claims: {path} = {o!r}')
    walk(claims, '')
    return hits


def label_guard(tables, claims):
    h = label_hits(tables, claims)
    need(not h, f"label: the word '{LABEL_WORD}' labels cancel-compatible output ({len(h)}: {h[:3]}) (refused)")


# ------------------------------------------------------------------ build / write / run ------------------------------------------------------------------
def _values(pred):
    return {k: dict(value=v['value'], **{x: v[x] for x in ('missing_rows', 'evaluable') if x in v}) for k, v in pred.items()}


def build(pull_dir, take_dir, review_take_sha, tick):
    checks = {}
    F, inputs, tr, bindings = bind_and_read(pull_dir, take_dir, review_take_sha)
    tick('bind_read')
    guard('inputs')
    S, CT, PL, CEN = F['S'], F['CT'], F['PL'], F['CEN']
    summary_vocab(S)
    fam = pull_families(S)
    mem = pull_membership(CT, PL, S, CEN)
    chk(checks, 'membership_reconciles_summary_and_census', bool(mem['reconciled'].all()), n_rows=len(mem), n_unreconciled=int((~mem['reconciled']).sum()))
    chk(checks, 'cancel_possible_rows_tied_never_identified', bool(PL['pull_identified'].astype(str).eq('False').all()
                                                                    and PL['identification'].eq('tied').all() and PL['end_histories'].eq('EX|PL').all()), n_rows=len(PL))
    I = pooled_index(S)
    pred = predicates(I, 'S_FIFO', D_HEAD)
    sens = {f'{v}_D{D}': predicates(I, v, D) for v, D in SENS}
    reps = terminal_reproductions(I, mem)
    tick('terminal_window')
    TK, TC, TS, AC = F['TK'], F['TC'], F['TS'], F['AC']
    take_vocab(TK, TC)
    claim = TK['claim'].astype(str).eq('True').to_numpy()
    G = take_groups(TK)
    rel, n_rva = take_relabel(TC, G, checks)
    x = take_obs_reconcile(TC, TK, G, claim)
    chk(checks, 'take_btc_obs_recomputed_from_takes', x['n_mismatch'] == 0 and x['n_rows_checked'] > 0, **x)
    x = take_class_reconcile(TK, TS, AC, G)
    chk(checks, 'take_class_counts_equal_summary_and_actions', x['n_mismatch'] == 0, **x)
    x = take_summary_vs_controls(TS, TC)
    chk(checks, 'take_summary_tvc_equals_take_controls', x['n_mismatch'] == 0 and x['n_rows_compared'] > 0, **x)
    levels, add_ok, rec_ok = take_match_levels(rel, tr)
    chk(checks, 'take_match_levels_additive', add_ok)
    chk(checks, 'take_match_levels_equal_receipt_pools', rec_ok, receipt=(tr.get('pools') or {}).get('pool_level'), pooled=levels.get('pooled|all'))
    flag = np.array(['excluded_before' in str(s_).split('+') for s_ in TK['status'].to_numpy(object)], bool)
    nx = TK['n_excl_before'].to_numpy(np.int64) > 0
    rec_n = (tr.get('statuses') or {}).get('excluded_before')
    chk(checks, 'take_excluded_before_flags_agree', bool((flag == nx).all()) and rec_n == int(nx.sum()), n=int(nx.sum()), receipt=rec_n)
    chk(checks, 'take_class_equals_take_class_0', bool((TK['take_class'] == TK[f'take_class_{D_TAKE_HEAD}']).all()))
    ali = take_case_alignment(TK, G, claim)
    xb = take_excluded_before(TK, G, claim)
    reps += take_reproductions(rel, xb)
    tick('take')
    sf = sign_falsifier()
    chk(checks, 'sign_falsifier', not sf, failures=sf)
    tables = dict(pull_families=fam, pull_membership=mem, take_relabelled=rel, take_case_alignment=ali, take_excluded_before=xb)
    claims = dict(definitions=dict(DEFINITIONS, headline=dict(view='S_FIFO', view_beside='S_NOPRIO', D_ms=D_HEAD, scope='pooled', btc_feature=BTC_F,
                                                                inventory_feature=INV_F, alpha_refdraw=ALPHA, min_reps=MIN_REPS)),
                  terminal_window_predicates=dict(headline=pred, sensitivity=sens),
                  reproductions=dict(items=reps, n_items=len(reps), n_match=int(sum(r['match'] for r in reps)), all_match=bool(all(r['match'] for r in reps)),
                                     not_reproducible=list(NOT_REPRODUCIBLE)))
    hits = label_hits(tables, claims)
    chk(checks, 'cancel_compatible_label_clean', not hits, hits=hits[:5])
    tick('claims')
    guard('tables')
    counts = dict(groups_published=list(PUB_GROUPS), groups_not_published=_unknown(S['group'].unique(), PUB_GROUPS),
                  reducing_vs_adding_rows_not_republished=n_rva, take_match_levels=levels, tables={k: int(len(v)) for k, v in tables.items()})
    return dict(tables=tables, claims=claims, checks=checks, inputs=inputs, bindings=bindings, counts=counts)


def write_outputs(out, tables, claims, make_receipt):
    """Per-file write-once: every file is written as a partial in the shared directory, then hard-linked to its name (never replaces an
    existing name); on any failure the names this call created are removed. Returns the output hashes (without the receipt)."""
    out = Path(out)
    for f in OUT_FILES:
        need(not (out / f).exists(), f'write-once: {out / f} exists (refused)')
    out.mkdir(parents=True, exist_ok=True)
    tmp, made, pid = {}, [], os.getpid()
    try:
        for t in TABLES:
            p = out / f'.{t}.csv.partial{pid}'
            tables[t].to_csv(p, index=False)
            tmp[f'{t}.csv'] = p
        p = out / f'.{CLAIMS_FILE}.partial{pid}'
        p.write_text(_json_text(claims))
        tmp[CLAIMS_FILE] = p
        hashes = {f: dict(sha256=sha256(q), bytes=int(q.stat().st_size), rows=int(len(tables[f[:-4]])) if f.endswith('.csv') else None)
                  for f, q in tmp.items()}
        p = out / f'.{RECEIPT_FILE}.partial{pid}'
        p.write_text(_json_text(make_receipt(hashes)))
        tmp[RECEIPT_FILE] = p
        for f in OUT_FILES:
            try:
                os.link(tmp[f], out / f)
            except FileExistsError:
                raise ValueError(f'write-once: {out / f} appeared during the run (refused)') from None
            made.append(out / f)
    except BaseException:
        for q in made:
            q.unlink(missing_ok=True)
        raise
    finally:
        for q in tmp.values():
            q.unlink(missing_ok=True)
    return hashes


def run(dry=False, pull_dir=None, take_dir=None, out=None, git_fn=None, require_freeze=True, review_take_sha=REVIEW_TAKE_SHA, log=print):
    need(C.RUN == 'v2', f'MT_RUN must be v2 (is {C.RUN!r})')
    mut = os.environ.get('MT_CC_MUT', '')
    need(not mut, f'mutant MT_CC_MUT={mut!r} is set: a planted-error switch never writes or prints a result (refused)')
    pull_dir = Path(pull_dir) if pull_dir else PULL_DIR
    take_dir = Path(take_dir) if take_dir else TAKE_DIR
    out = Path(out) if out else OUT_DIR
    t0, asof0 = time.time(), _now()
    stages, tick, c0 = _stager()
    if not dry:
        for f in OUT_FILES:
            need(not (out / f).exists(), f'write-once: {out / f} exists (refused before any input is read)')
    git = (git_fn or _git)(TOOL_FILES)
    if not dry and require_freeze:
        need(bool(git.get('unmodified')), 'freeze: the tool, its selftest and the correction contract must be tracked and unmodified before the run (refused)')
    B = build(pull_dir, take_dir, review_take_sha, tick)
    failed = sorted(k for k, v in B['checks'].items() if not v['ok'])
    pred = B['claims']['terminal_window_predicates']
    if dry:
        reps = B['claims']['reproductions']
        res = dict(mode='dry', writes_nothing=True, out=str(out), schema_checks=B['checks'], failed_checks=failed, bindings=B['bindings'],
                   predicates=dict(headline=pred['headline'], sensitivity={k: _values(v) for k, v in pred['sensitivity'].items()}),
                   reproductions=dict(n_items=reps['n_items'], n_match=reps['n_match'], all_match=reps['all_match'],
                                      items=[{k: r[k] for k in ('id', 'reviewer', 'computed', 'match')} for r in reps['items']],
                                      not_reproducible=reps['not_reproducible']),
                   would_write={f'{t}.csv': int(len(B['tables'][t])) for t in TABLES}, groups_not_published=B['counts']['groups_not_published'],
                   git=git, cpu_s=round(time.process_time() - c0, 3), stages=stages, vmhwm_mib=round(hwm(), 1))
        log(_json_text(res))
        need(not failed, f'dry: checks failed {failed} (a run would refuse)')
        return res
    need(not failed, f'checks failed {failed} (refused before writing)')
    holder = {}

    def make_receipt(hashes):
        tick('write')
        holder['r'] = dict(version=VERSION, kind='run', contract=CONTRACT, review=REVIEW, module=_rel(__file__), run=C.RUN, asof_start=asof0, asof_end=_now(),
                           git=git, sources={_rel(f): sha256(f) for f in TOOL_FILES + DEP_FILES if Path(f).is_file()}, inputs=B['inputs'],
                           bindings=B['bindings'], outputs=hashes, checks=B['checks'], counts=B['counts'],
                           predicates={'headline': _values(pred['headline']), **{k: _values(v) for k, v in pred['sensitivity'].items()}},
                           reproductions_all_match=B['claims']['reproductions']['all_match'],
                           constants=dict(ALPHA=ALPHA, MIN_REPS=MIN_REPS, D_HEAD=D_HEAD, SENS=SENS, D_TAKE=D_TAKE, INV_SIDE=INV_SIDE, RENAME=RENAME,
                                          PUB_GROUPS=PUB_GROUPS, GROUP_LABEL=GROUP_LABEL, REVIEW_TAKE_SHA=review_take_sha),
                           cpu_s=dict(by_stage=stages, total=round(time.process_time() - c0, 3), cap=CPU_CAP_S), wall_s=round(time.time() - t0, 2),
                           vmhwm_mib=round(hwm(), 1), guard_mib=C.GUARD_MIB)
        return holder['r']
    write_outputs(out, B['tables'], B['claims'], make_receipt)
    r = holder['r']
    log(json.dumps(_clean(dict(out=str(out), outputs={k: v['rows'] for k, v in r['outputs'].items()}, cpu_s=r['cpu_s'], vmhwm_mib=r['vmhwm_mib'],
                               reproductions_all_match=r['reproductions_all_match'], predicates=r['predicates']['headline'])), indent=1))
    return r


if __name__ == '__main__':
    a = sys.argv[1:]
    if a == ['run']:
        run()
    elif a == ['run', '--dry']:
        run(dry=True)
    else:
        raise SystemExit('usage: MT_RUN=v2 chain_correct_pull_take.py run | chain_correct_pull_take.py run --dry')
