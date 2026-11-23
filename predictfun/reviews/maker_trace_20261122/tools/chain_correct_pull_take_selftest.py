"""chain_correct_pull_take_selftest.py - synthetic proof of chain_correct_pull_take.py (P-2026-005 MAKER-TRACE logic-chain correction v1;
chain_correction_contract.md sections 2, 3 and the section 5 falsifiers 'sign' and 'omitted control'; 2026-11-23). In process; synthetic
inputs only, written to a temporary directory (no saved data path is opened). Sections: engine pins (chain_take's take-side BTC convention
through its real window_feats / classify on a fake context; its DIRCAT / CLS / REDUCING / MOVES; the production paths); the sign falsifier
and its flip_inventory_sign mutant; hand-computed relabel / reconciliation / case-alignment / excluded-before counts; the omitted-control
falsifier (known-bad inputs and the drop_fill_anchored mutant); membership and its reconciliations; the computed predicates on two synthetic
worlds, a missing day, a p threshold and a replicate floor; the label guard; every writer end to end (run + dry) with the write-once, freeze,
binding, omitted-control and mutant refusals (each reason asserted) and an independent text scan that the word 'pull' labels no
cancel-compatible output. Usage (run_light, MT_RUN=v2): chain_correct_pull_take_selftest.py"""
import json
import os
import shutil
import sys
import tempfile
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RESEARCH = HERE.parents[2] / 'research'
for _p in (str(RESEARCH), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import chain_correct_pull_take as K  # noqa: E402
import chain_take as T  # noqa: E402   (engine pins only: pure functions on a fake context; no data path)

RES = []
NAN = float('nan')
DAYS = K.DAYS
D0, D1, D2, D3 = DAYS
VARIANT_A = dict(ex_spike=True, neg_inv_day=D1)      # BTC enriched every day; fill_anchored inventory negative on D1; exhausted BTC spike on D3
VARIANT_B = dict(ex_spike=False, neg_inv_day=None)   # every declared predicate holds


def GIT_OK(files):
    return dict(head='0' * 40, tool_files=[str(f) for f in files], tracked=True, unmodified=True)


def GIT_BAD(files):
    return dict(head='0' * 40, tool_files=[str(f) for f in files], tracked=False, unmodified=False)


def check(name, ok, info=''):
    RES.append((name, bool(ok), str(info)[:400]))


def expect(name, fn, needle):
    try:
        fn()
    except Exception as e:  # noqa: BLE001
        check(name, needle in str(e), f'{type(e).__name__}: {e}')
        return
    check(name, False, 'did not raise')


def section(name, fn, *a):
    try:
        fn(*a)
    except Exception as e:  # noqa: BLE001
        check(f'{name}: section crashed', False, f'{type(e).__name__}: {e} | {traceback.format_exc()[-900:]}')


class Mutant:
    """MT_CC_MUT=<name> inside the block (the tool reads it at call time)."""

    def __init__(self, name):
        self.name = name

    def __enter__(self):
        os.environ['MT_CC_MUT'] = self.name
        return self

    def __exit__(self, *exc):
        os.environ.pop('MT_CC_MUT', None)
        return False


# ------------------------------------------------------------------ engine pins ------------------------------------------------------------------
class FakeCtx:
    """chain_take's context bridge reduced to what window_feats reads: constant BTC moves, a one-tick spread."""

    def __init__(self, r1, r5, r30):
        self.r = (float(r1), float(r5), float(r30))

    def btc(self, M, x):
        n = len(np.asarray(x))
        return dict(code=np.zeros(n, np.int64), r_1s=np.full(n, self.r[0]), r_5s=np.full(n, self.r[1]), r_30s=np.full(n, self.r[2]), pf=np.full(n, .5))

    def book(self, M, x):
        return dict(spread=np.full(len(np.asarray(x)), 1.))


def fake_grid():
    """A grid with no instant strictly inside [0, 100]: the window is its two ends."""
    G = dict(G0=0, cum_ok=np.zeros(3), pf=np.full(2, NAN), spread=np.full(2, NAN))
    for nm, _ in T.MOVES:
        G[f'cum_up_{nm}'], G[f'cum_dn_{nm}'] = np.zeros(3), np.zeros(3)
    return G


def s_engine():
    check('engine: DIRCAT, CLS, REDUCING and the move thresholds equal chain_take', K.DIRCAT == T.DIRCAT and K.CLS == T.CLS and K.REDUCING == T.REDUCING
          and K.MOVE_THR == dict(T.MOVES), (T.DIRCAT, T.MOVES))
    lo, hi, sides = np.zeros(2, np.int64), np.full(2, 100, np.int64), np.array([-1, 1])
    for nm in ('r_1s', 'r_5s', 'r_30s'):
        up = T.window_feats(FakeCtx(1., 2., 3.), None, fake_grid(), lo, hi, sides)[f'cat_{nm}']
        dn = T.window_feats(FakeCtx(-1., -2., -3.), None, fake_grid(), lo, hi, sides)[f'cat_{nm}']
        got = [T.DIRCAT[int(x)] for x in up] + [T.DIRCAT[int(x)] for x in dn]
        check(f'engine: chain_take.window_feats {nm} is aligned with the TAKE side (sell, buy x BTC up, down)', got == ['against', 'with', 'with', 'against'], got)
    pairs = []
    for s_ in (-1, 1):
        for r in (2., -2., .5):
            mine = int(K.btc_side_code(s_, r, K.MOVE_THR['r_5s']))
            eng = int(T.window_feats(FakeCtx(0., r, 0.), None, fake_grid(), lo[:1], hi[:1], np.array([s_]))['cat_r_5s'][0])
            pairs.append((s_, r, mine, eng))
    check('engine: btc_side_code of the tool equals chain_take.window_feats on every side x move', all(a == b for _, _, a, b in pairs), pairs)
    got = [(c[0], T.CLS[int(T.classify(np.array([c[2]]), np.array([c[3]]))[0])], c[4]) for c in K.SIGN_CASES]
    check('engine: the class of every sign case is chain_take.classify(before, dq)', all(a == b for _, a, b in got), got)
    paths = (K.PULL_DIR, K.TAKE_DIR, K.OUT_DIR)
    check('engine: production paths are the v2 receipt chains pull / take / correction_v1 (MT_RUN=v2)',
          [str(p).split('maker_trace_20261122/')[-1] for p in paths] == ['v2/chains/pull', 'v2/chains/take', 'v2/chains/correction_v1'], paths)
    check('engine: the seven declared output names', K.OUT_FILES == ('pull_families.csv', 'pull_membership.csv', 'take_relabelled.csv', 'take_case_alignment.csv',
                                                                    'take_excluded_before.csv', 'claims_pull_take.json', 'receipt_pull_take.json'), K.OUT_FILES)


# ------------------------------------------------------------------ the sign falsifier ------------------------------------------------------------------
def s_sign():
    f0 = K.sign_falsifier()
    check('sign falsifier: every declared case passes unmutated (a reducing sell of a long with BTC up is against the take side and with the '
          'inventory; an adding buy with BTC down is against both)', f0 == [] and len(K.SIGN_CASES) == 6, f0)
    with Mutant('flip_inventory_sign'):
        f1 = K.sign_falsifier()
    check('sign falsifier: the flip_inventory_sign mutant is caught on every case', len(f1) == len(K.SIGN_CASES), f1)
    t = K.btc_side_code(-1, 2., 1.)
    i = K.inventory_code(np.array([t]), K.inventory_multiplier(['reduce_partial']))[0]
    check('sign: a reducing sell of a long with BTC up is against the take side and with the inventory', (K.DIRCAT[int(t)], K.DIRCAT[int(i)]) == ('against', 'with'))
    t = K.btc_side_code(1, -2., 1.)
    i = K.inventory_code(np.array([t]), K.inventory_multiplier(['add']))[0]
    check('sign: an adding buy with BTC down is against the take side and against the inventory', (K.DIRCAT[int(t)], K.DIRCAT[int(i)]) == ('against', 'against'))
    i = K.inventory_code(np.array([1, 2]), K.inventory_multiplier(['timing_sensitive', 'reduce_partial']))
    check('sign: timing_sensitive has no inventory side; a side-free category (flat) stays flat', int(i[0]) == K.UNDET and K.DIRCAT[int(i[1])] == 'flat', i)


# ------------------------------------------------------------------ synthetic take world ------------------------------------------------------------------
TC_FEATS = (('r_1s_with', 'binary'), ('r_1s_against', 'binary'), ('r_5s_with', 'binary'), ('r_5s_against', 'binary'), ('r_30s_with', 'binary'),
            ('r_30s_against', 'binary'), ('r_5s_flat', 'binary'), ('r_5s_mixed', 'binary'), ('burst3_certain', 'binary'),
            ('offset_sh_rec_5s', 'continuous'), ('reduction_if_taken', 'binary'))
TC_GROUPS = [(f'rank{r}', d) for r in (1, 2, 3) for d in DAYS] + [(f'rank{r}', 'all') for r in (1, 2, 3)] + [('pooled', 'all')]
TC_ORDER = ['comparison', 'scope', 'day', 'D', 'feature', 'kind', 'n_takes', 'n_fallback1', 'n_fallback2', 'n_no_control', 'n_no_claim', 'obs', 'null_mean',
            'p05', 'p50', 'p95', 'p_hi', 'p_lo', 'n_draws', 'odds_ratio', 'n_reducing', 'n_adding', 'n_excluded_class', 'mean_reducing', 'mean_adding']


def group_mask(TK, sc, d):
    m = np.ones(len(TK), bool) if sc == 'pooled' else (TK['rank'].to_numpy() == int(sc[4:]))
    return (m & (TK['day'].to_numpy(object) == d)) if d != 'all' else m


def take_world():
    """32 takes: 8 hand cases + a reduce_partial 'flat' and an add 'missing' take per rank x day. level = the reference-match level."""
    rows = []

    def add(rank, day, s, cls, cat, excl=0, cls1000=None, level=0):
        r = dict(day=day, mid=200 + DAYS.index(day), rank=rank, sid=f'sid{len(rows):04d}', s=s, claim=True, status='excluded_before' if excl else 'ok',
                 n_excl_before=excl, take_class=cls, level=level)
        for D in K.D_TAKE:
            r[f'take_class_{D}'] = cls1000 if (D == 1000 and cls1000) else cls
            for w in K.WINDOWS:
                r[f'cat_r_{w}_{D}'] = cat
        rows.append(r)
    add(1, D0, -1, 'reduce_partial', 'against', cls1000='timing_sensitive')   # a reducing sell of a long with BTC up
    add(1, D0, 1, 'add', 'against', level=1)                                    # an adding buy with BTC down (sign dropped)
    add(1, D1, 1, 'timing_sensitive', 'with')
    add(2, D1, -1, 'overshoot', 'with', level=2)                                 # a seller with BTC down; the long held before (inventory dropped)
    add(3, D2, 1, 'reduce_partial', 'with')                                      # a reducing buy of a short with BTC up
    add(1, D3, -1, 'from_flat', 'flat')
    add(1, D2, 1, 'add', 'mixed', excl=1)
    add(2, D3, -1, 'reduce_partial', 'gap', excl=2)
    for rk in (1, 2, 3):
        for d in DAYS:
            add(rk, d, 1, 'reduce_partial', 'flat')
            add(rk, d, -1, 'add', 'missing')
    return pd.DataFrame(rows)


def take_controls(TK):
    lvl = TK['level'].to_numpy()
    rows = []
    for D in K.D_TAKE:
        for f, kind in TC_FEATS:
            for sc, d in TC_GROUPS:
                m = group_mask(TK, sc, d)
                n = int(m.sum())
                if f in K.BTC_CAT:
                    w, val = K.BTC_CAT[f]
                    obs = float((TK[f'cat_r_{w}_{D}'].to_numpy(object)[m] == val).mean()) if n else NAN
                else:
                    obs = .25 if n else NAN
                base = dict(scope=sc, day=d, D=D, feature=f, kind=kind, n_takes=n, odds_ratio=NAN)
                rows.append(dict(base, comparison='take_vs_control', n_fallback1=float((m & (lvl == 1)).sum()), n_fallback2=float((m & (lvl == 2)).sum()),
                                 n_no_control=0., n_no_claim=0., obs=obs, null_mean=.1, p05=.05, p50=.1, p95=.15, p_hi=.002, p_lo=1., n_draws=500,
                                 n_reducing=NAN, n_adding=NAN, n_excluded_class=NAN, mean_reducing=NAN, mean_adding=NAN))
                if f != 'reduction_if_taken':
                    rows.append(dict(base, comparison='reducing_vs_adding', n_fallback1=NAN, n_fallback2=NAN, n_no_control=NAN, n_no_claim=NAN, obs=.1,
                                     null_mean=0., p05=-.1, p50=0., p95=.1, p_hi=.2, p_lo=.8, n_draws=500, n_reducing=n // 2, n_adding=n - n // 2,
                                     n_excluded_class=0, mean_reducing=.2, mean_adding=.1))
    return pd.DataFrame(rows)[TC_ORDER]


def take_summary(TK, TC):
    cls = TK['take_class'].to_numpy(object)
    tv = TC[(TC['comparison'] == 'take_vs_control') & (TC['feature'] == 'r_5s_against') & (TC['D'] == 0)]
    idx = {(r['scope'], r['day']): r for r in tv.to_dict('records')}
    rows = []
    for sc in K.SCOPES:
        for d in K.DAYS_ALL:
            m = group_mask(TK, sc, d)
            r = dict(scope=sc, day=d, n_takes=int(m.sum()), n_claim=int(m.sum()))
            for c in K.CLS:
                r[f'cls_{c}'] = int((m & (cls == c)).sum())
            c_ = idx.get((sc, d))
            r['tvc_r_5s_against_obs'] = c_['obs'] if c_ else NAN
            r['tvc_r_5s_against_null_mean'] = c_['null_mean'] if c_ else NAN
            r['tvc_r_5s_against_n_takes'] = float(c_['n_takes']) if c_ else NAN
            rows.append(r)
    return pd.DataFrame(rows)


def take_actions(TK):
    cls = TK['take_class'].to_numpy(object)
    rows = []
    for sc in K.SCOPES:
        for d in K.DAYS_ALL:
            m = group_mask(TK, sc, d)
            rows.append(dict(action_type='maker_fill', scope=sc, day=d, n=7, status='unidentified'))
            for c in K.CLS:
                n = int((m & (cls == c)).sum())
                if n:
                    rows.append(dict(action_type=f'take_{c}', scope=sc, day=d, n=n, status='unidentified'))
            rows.append(dict(action_type='take_kind_mint', scope=sc, day=d, n=1, status='unidentified'))
    return pd.DataFrame(rows)


def write_take_world(d):
    d.mkdir(parents=True, exist_ok=True)
    TK = take_world()
    TC = take_controls(TK)
    TS, AC = take_summary(TK, TC), take_actions(TK)
    p = TC[(TC['comparison'] == 'take_vs_control') & (TC['scope'] == 'pooled') & (TC['day'] == 'all')].iloc[0]
    for f, df in (('takes.csv', TK.drop(columns=['level'])), ('take_controls.csv', TC), ('summary.csv', TS), ('actions.csv', AC)):
        df.to_csv(d / f, index=False)
    nx = int((TK['n_excl_before'] > 0).sum())
    rec = dict(outputs=dict(takes=len(TK), take_controls=len(TC), actions=len(AC), summary=len(TS)),
               pools=dict(pool_level=dict(stratum=int(len(TK) - p['n_fallback1'] - p['n_fallback2']), fallback1=int(p['n_fallback1']),
                                          fallback2=int(p['n_fallback2']), no_control=0, no_claim=0)),
               statuses=dict(excluded_before=nx, ok=len(TK) - nx))
    (d / 'receipt.json').write_text(json.dumps(rec))
    return TK, TC, TS, AC


def s_take():
    TK = take_world()
    TC = take_controls(TK)
    claim = np.ones(len(TK), bool)
    G = K.take_groups(TK)
    ck = {}
    R, n_rva = K.take_relabel(TC, G, ck)
    check('relabel: only the take_vs_control rows are republished; the reducing_vs_adding rows are counted',
          (R['comparison'] == 'take_vs_control').all() and n_rva == int((TC['comparison'] == 'reducing_vs_adding').sum())
          and len(R) == int((TC['comparison'] == 'take_vs_control').sum()), (len(R), n_rva))
    r = R[R['feature_saved'] == 'r_5s_against'].iloc[0]
    check('relabel: r_5s_against -> btc_against_take_side_5s, take-side alignment, inventory reading withdrawn',
          (r['feature'], r['alignment'], r['inventory_reading']) == ('btc_against_take_side_5s', 'take_side', 'withdrawn'), tuple(r)[:9])
    b = R[R['feature_saved'] == 'burst3_certain']
    check('relabel: a non-BTC feature keeps its name', set(b['feature']) == {'burst3_certain'} and set(b['alignment']) == {'not_btc'})
    p = R[(R['scope'] == 'pooled') & (R['day'] == 'all')].iloc[0]
    got = tuple(int(p[c]) for c in ('grp_n_takes', 'n_full_stratum', 'n_sign_dropped', 'n_inventory_dropped'))
    check('relabel: match levels beside (pooled: 32 takes = 30 full stratum + 1 sign dropped + 1 inventory dropped)', got == (32, 30, 1, 1), got)
    check('relabel: its reconciliation checks pass (constant per group, non-negative, binary n_takes)', len(ck) == 3 and all(v['ok'] for v in ck.values()), ck)
    lv, add_ok, rec_ok = K.take_match_levels(R, dict(pools=dict(pool_level=dict(stratum=30, fallback1=1, fallback2=1, no_control=0, no_claim=0))))
    check('reconcile: match levels additive and equal to the receipt pools', add_ok and rec_ok and lv['pooled|all']['n_full_stratum'] == 30, lv.get('pooled|all'))
    bad = K.take_match_levels(R, dict(pools=dict(pool_level=dict(stratum=31, fallback1=1, fallback2=1, no_control=0, no_claim=0))))
    check('reconcile: a receipt pool count that differs is flagged (the check can fire)', bad[2] is False)
    TCx = TC.copy()
    TCx.loc[TCx.index[0], 'feature'] = 'r_2s_against'
    expect('refuse: an unknown take feature', lambda: K.take_vocab(TK, TCx), 'unknown features')
    x = K.take_obs_reconcile(TC, TK, G, claim)
    check('reconcile: every BTC take_vs_control obs is recomputed exactly from the takes', x['n_mismatch'] == 0 and x['n_rows_checked'] == 8 * 4 * 16, x)
    TCb = TC.copy()
    j = TCb.index[(TCb['comparison'] == 'take_vs_control') & (TCb['feature'] == 'r_5s_against') & (TCb['scope'] == 'rank1') & (TCb['day'] == D0)
                  & (TCb['D'] == 0)][0]
    TCb.loc[j, 'obs'] = TCb.loc[j, 'obs'] + 1e-6
    x = K.take_obs_reconcile(TCb, TK, G, claim)
    check('reconcile: a perturbed saved obs is flagged (the check can fire)', x['n_mismatch'] == 1, x)
    TS, AC = take_summary(TK, TC), take_actions(TK)
    x = K.take_class_reconcile(TK, TS, AC, G)
    check('reconcile: class counts equal the take summary and the actions', x['n_mismatch'] == 0 and x['n_action_rows'] > 0, x)
    TS2 = TS.copy()
    TS2.loc[TS2.index[0], 'cls_add'] = TS2.loc[TS2.index[0], 'cls_add'] + 1
    check('reconcile: a summary class count off by one is flagged', K.take_class_reconcile(TK, TS2, AC, G)['n_mismatch'] == 1)
    x = K.take_summary_vs_controls(TS, TC)
    check('reconcile: the take summary tvc_r_5s_against equals its take_controls rows', x['n_mismatch'] == 0 and x['n_rows_compared'] == 16, x)
    A = K.take_case_alignment(TK, G, claim)

    def cell(Af, pop, sc, d, D, w, c):
        return Af[(Af['population'] == pop) & (Af['scope'] == sc) & (Af['day'] == d) & (Af['D'] == D) & (Af['window'] == w) & (Af['take_class'] == c)].iloc[0]
    x = cell(A, 'claim_all', 'rank1', D0, 0, '5s', 'reduce_partial')
    got = tuple(int(x[c]) for c in ('n_takes', 'n_btc_against_take_side', 'n_btc_with_inventory', 'n_btc_against_inventory', 'n_inventory_determined'))
    check('cases: rank1 day 1 reduce_partial = 2 takes, 1 against the take side, that one WITH the inventory (the reducing sell of a long, BTC up)',
          got == (2, 1, 1, 0, 2), got)
    x = cell(A, 'claim_all', 'rank1', D0, 0, '5s', 'add')
    got = tuple(int(x[c]) for c in ('n_takes', 'n_btc_against_take_side', 'n_btc_against_inventory', 'n_btc_with_inventory'))
    check('cases: rank1 day 1 add = 2 takes, 1 against the take side and against the inventory (the adding buy, BTC down)', got == (2, 1, 1, 0), got)
    x = cell(A, 'claim_all', 'rank1', D1, 0, '5s', 'timing_sensitive')
    check('cases: a timing_sensitive take has no inventory side (undetermined, empty rate)', int(x['n_takes']) == 1 and int(x['n_inventory_determined']) == 0
          and x['inventory_side'] == 'undetermined' and not np.isfinite(x['rate_btc_with_inventory']), tuple(x))
    x = cell(A, 'claim_all', 'rank2', D1, 0, '5s', 'overshoot')
    got = tuple(int(x[c]) for c in ('n_takes', 'n_btc_with_take_side', 'n_btc_against_inventory'))
    check('cases: an overshooting sell with BTC down: with the take side, against the long held before', got == (1, 1, 1) and x['inventory_side'] == '-take_side', got)
    x = cell(A, 'claim_all', 'rank3', D2, 0, '30s', 'reduce_partial')
    got = tuple(int(x[c]) for c in ('n_takes', 'n_btc_with_take_side', 'n_btc_against_inventory'))
    check('cases: a reducing buy of a short with BTC up: with the take side, against the short held', got == (2, 1, 1), got)
    x = cell(A, 'claim_all', 'rank1', D0, 1000, '5s', 'timing_sensitive')
    check('cases: the class basis follows D (take_class_1000 of the reducing sell is timing_sensitive: undetermined there)',
          int(x['n_takes']) == 1 and int(x['n_inventory_determined']) == 0 and x['class_basis'] == 'take_class_1000')
    x = cell(A, 'claim_all', 'pooled', 'all', 0, '5s', 'all_classes')
    y = cell(A, 'claim_without_excluded_before', 'pooled', 'all', 0, '5s', 'all_classes')
    check('cases: pooled all_classes = 32 takes, 30 with a determined inventory side; without earlier excluded legs 30 takes',
          (int(x['n_takes']), int(x['n_inventory_determined']), int(y['n_takes'])) == (32, 30, 30), (int(x['n_takes']), int(x['n_inventory_determined']), int(y['n_takes'])))
    check('cases: every row is labelled cases_only_no_reference; 2 populations x 20 groups x 4 D x 3 windows x 11 classes',
          set(A['label']) == {K.CASE_LABEL} and len(A) == 2 * 20 * 4 * 3 * 11, len(A))
    with Mutant('flip_inventory_sign'):
        Am = K.take_case_alignment(TK, G, claim)
    xm = cell(Am, 'claim_all', 'rank1', D0, 0, '5s', 'reduce_partial')
    got = (int(xm['n_btc_with_inventory']), int(xm['n_btc_against_inventory']))
    check('sign falsifier: under flip_inventory_sign the reducing sell reads against the inventory - the hand count (1, 0) catches it', got != (1, 0), got)
    X = K.take_excluded_before(TK, G, claim)

    def xb(c):
        return X[(X['scope'] == 'pooled') & (X['day'] == 'all') & (X['take_class'] == c)].iloc[0]
    got = (int(xb('all_classes')['n_with_excluded_before']), int(xb('add')['n_with_excluded_before']), int(xb('reduce_partial')['n_with_excluded_before']),
           int(xb('all_classes')['n_without_excluded_before']), xb('add')['inventory_class_statement'])
    check('excluded before: pooled 2 takes (1 add, 1 reduce_partial), 30 without; the class statement is conditional on the retained ledger',
          got == (2, 1, 1, 30, 'conditional_on_retained_ledger'), got)


# ------------------------------------------------------------------ synthetic terminal-window world ------------------------------------------------------------------
def pull_world():
    """Per day x view x class x rank four ok-window events: presence ok / fallback / fallback / no_eligible, fill_anchored anchors / anchors /
    none / none, successor none / reprice / none / refill+reprice; one reaches_we cancel-compatible order per day x view (no controls)."""
    ctrl, pulls, k = [], [], 0
    for di, d in enumerate(DAYS):
        for v in K.VIEWS:
            for cls in (('cancel_possible', 'exhausted_proven') if v == 'S_FIFO' else ('cancel_possible',)):
                for rk in (1, 2, 3):
                    for j in range(4):
                        k += 1
                        h = f'0x{k:064x}'
                        sp, sf = ('ok', 'fallback', 'fallback', 'no_eligible')[j], ('anchors', 'anchors', 'none', 'none')[j]
                        su = ('none', 'reprice', 'none', 'refill+reprice')[j] if cls == 'cancel_possible' else 'none'
                        for fam, st in (('presence', sp), ('fill_anchored', sf)):
                            ctrl.append(dict(day=d, mid=100 + di, scope=f'rank{rk}', mk_hash=h, view=v, cls=cls, control=fam, draw_status=st,
                                             n_draws=0 if st in ('no_eligible', 'none') else 200, draw_span_ms=1000, succ=su != 'none'))
                        if cls == 'cancel_possible':
                            pulls.append(dict(day=d, mid=100 + di, scope=f'rank{rk}', addr='0xa', mk_hash=h, view=v, window_status='ok', successor=su,
                                              draws_presence=sp, draws_fill_anchored=sf, pull_identified=False, identification='tied', end_histories='EX|PL'))
            k += 1
            pulls.append(dict(day=d, mid=100 + di, scope='rank1', addr='0xa', mk_hash=f'0x{k:064x}', view=v, window_status='reaches_we', successor='none',
                              draws_presence='n/a', draws_fill_anchored='n/a', pull_identified=False, identification='tied', end_histories='EX|PL'))
    return pd.DataFrame(ctrl), pd.DataFrame(pulls)


def census(ctrl, pulls):
    ev = ctrl[ctrl['control'] == 'presence'][['view', 'day', 'scope', 'cls']].assign(window_status='ok')
    nk = pulls[pulls['window_status'] != 'ok'][['view', 'day', 'scope', 'window_status']].assign(cls='cancel_possible')
    E = pd.concat([ev, nk], ignore_index=True)
    rows = []
    for (v, c, w), g in E.groupby(['view', 'cls', 'window_status']):
        for sc in K.SCOPES:
            gs = g if sc == 'pooled' else g[g['scope'] == sc]
            for d in K.DAYS_ALL:
                n = len(gs) if d == 'all' else int((gs['day'] == d).sum())
                if n:
                    rows.append(dict(view=v, day=d, scope=sc, cls=c, reason='-', window_status=w, end_histories='EX|PL' if c == 'cancel_possible' else 'EX',
                                     orders=n, fills=2 * n, shares=10. * n, share_of_orders=.5))
    return pd.DataFrame(rows)


SUM_FEATS = (('btc_against_1s_some', 'rate'), ('inv_built_5s_some', 'rate'), ('tau_start', 'mean'), ('a_1s_start', 'mean'))


def stat(g, fam, ft, d, var):
    if ft == 'btc_against_1s_some':
        ref, diff = .2, ((.30 if (var['ex_spike'] and d == DAYS[-1]) else .04) if g == 'exhausted_proven' else .15)
    elif ft == 'inv_built_5s_some':
        ref, diff = .6, (.01 if g == 'exhausted_proven' else (-.05 if (fam == 'fill_anchored' and d == var['neg_inv_day']) else .10))
    elif ft == 'tau_start':
        ref, diff = 170., -20.
    else:
        ref, diff = 0., -.1
    return ref + diff, ref, diff


def summary(ctrl, pulls, var):
    rows = []
    for v in K.VIEWS:
        cv = ctrl[ctrl['view'] == v]
        for g in K.EXPECTED_GROUPS[v]:
            if g == 'exhausted_proven':
                X = cv[cv['cls'] == 'exhausted_proven']
            else:
                X = cv[cv['cls'] == 'cancel_possible']
                if g != 'cancel_possible':
                    X = X[X['succ'] == (g == 'cp_successor_5s')]
            for fam in K.FAMILIES:
                Xf = X[X['control'] == fam]
                for sc in K.SCOPES:
                    Xs = Xf if sc == 'pooled' else Xf[Xf['scope'] == sc]
                    for d in K.DAYS_ALL:
                        Xd = Xs if d == 'all' else Xs[Xs['day'] == d]
                        ne = len(Xd)
                        if not ne:
                            continue
                        nwc = int((Xd['n_draws'] > 0).sum())
                        for ft, kind in SUM_FEATS:
                            for D in (K.D_PULL if kind == 'rate' else K.D_PULL[:3]):
                                obs, ref, diff = stat(g, fam, ft, d, var)
                                rows.append(dict(view=v, group=g, control=fam, scope=sc, day=d, D=D, feature=ft, kind=kind, n_events=ne,
                                                 n_with_controls=nwc, n_obs=nwc, n_ctl_valid=200 * nwc, n_reps=200, obs=obs, null_mean=ref,
                                                 null_p05=ref - .01, null_p50=ref, null_p95=ref + .01, diff=diff,
                                                 p_upper=.004975124378109453 if diff > 0 else 1., p_lower=1. if diff > 0 else .004975124378109453))
    for fam in K.FAMILIES:     # an unpublished group: kept in the summary, never published
        rows.append(dict(view='S_FIFO', group='capacity_tight', control=fam, scope='pooled', day='all', D='250', feature='tau_start', kind='mean', n_events=3,
                         n_with_controls=3, n_obs=3, n_ctl_valid=600, n_reps=200, obs=1., null_mean=1., null_p05=1., null_p50=1., null_p95=1., diff=0.,
                         p_upper=.5, p_lower=.5))
    return pd.DataFrame(rows)


def write_pull_world(d, base, drop=None, corrupt_hash=False):
    d.mkdir(parents=True, exist_ok=True)
    ctrl, pulls, cen, S = base
    if drop is not None:
        S = S[~drop(S)]
    out = {'summary.csv': S, 'pulls.csv': pulls, 'controls.csv': ctrl.drop(columns=['succ']), 'census.csv': cen}
    for f, df in out.items():
        df.to_csv(d / f, index=False)
    rec = dict(outputs={f: dict(sha256=K.sha256(d / f), bytes=(d / f).stat().st_size) for f in out}, rows={f[:-4]: len(df) for f, df in out.items()})
    if corrupt_hash:
        rec['outputs']['controls.csv']['sha256'] = '0' * 64
    (d / 'receipt.json').write_text(json.dumps(rec))


def s_omitted(SA):
    F = K.pull_families(SA)
    pub = SA[SA['group'].isin(K.PUB_GROUPS)]
    check('omitted control: a complete summary publishes one row per key with both families side by side',
          len(F) == len(pub) // 2 and F['pres_n_obs'].notna().all() and F['fa_n_obs'].notna().all() and (F['pres_status'] == 'ok').all()
          and (F['fa_status'] == 'ok').all(), (len(F), len(pub)))
    check('omitted control: S_FIFO primary (first), S_NOPRIO beside; an unpublished group stays out',
          set(F['view_role']) == {'primary', 'beside'} and not F['group_saved'].isin(['capacity_tight']).any() and F['view'].iloc[0] == 'S_FIFO')
    r = F[(F['view'] == 'S_FIFO') & (F['group_saved'] == 'cancel_possible') & (F['scope'] == 'pooled') & (F['day'] == 'all') & (F['D'] == '250')
          & (F['feature'] == K.BTC_F)].iloc[0]
    check('omitted control: the headline row carries both families with their support (48 events; 36 / 24 with references)',
          (r['group'], int(r['n_events']), int(r['pres_n_with_controls']), int(r['fa_n_with_controls'])) == ('cancel_compatible_terminal_window', 48, 36, 24)
          and abs(r['pres_diff'] - .15) < 1e-12 and abs(r['fa_diff'] - .15) < 1e-12 and 'fa_p_upper_refdraw' in F.columns, tuple(r)[:12])
    one = SA.index[(SA['control'] == 'fill_anchored') & (SA['group'] == 'cp_successor_5s') & (SA['view'] == 'S_NOPRIO')][:1]
    expect('omitted control: a published key without its fill_anchored row is refused', lambda: K.pull_families(SA.drop(index=one)), 'omitted control')
    one2 = SA.index[(SA['control'] == 'presence') & (SA['group'] == 'exhausted_proven')][:1]
    expect('omitted control: a published key without its presence row is refused', lambda: K.pull_families(SA.drop(index=one2)), 'omitted control')
    with Mutant('drop_fill_anchored'):
        expect('omitted control: the drop_fill_anchored mutant is refused (caught)', lambda: K.pull_families(SA), 'omitted control')
    expect('refuse: a published group absent under a view', lambda: K.pull_families(SA[~((SA['view'] == 'S_NOPRIO') & (SA['group'] == 'cp_successor_5s'))]),
           'absent under S_NOPRIO')
    expect('refuse: a duplicate row per key and family', lambda: K.pull_families(pd.concat([SA, SA.iloc[:1]], ignore_index=True)), 'duplicate')


def s_membership(ctrl, pulls, cen, SA):
    M = K.pull_membership(ctrl, pulls, SA, cen)

    def row(Mf, v, g, sc, d):
        return Mf[(Mf['view'] == v) & (Mf['group_saved'] == g) & (Mf['scope'] == sc) & (Mf['day'] == d)].iloc[0]
    r = row(M, 'S_FIFO', 'cancel_possible', 'pooled', 'all')
    got = tuple(int(r[c]) for c in ('n_events', 'pres_matched', 'pres_fallback', 'pres_unavailable', 'fa_anchored', 'fa_unavailable', 'pres_with_draws', 'fa_with_draws'))
    check('membership: S_FIFO cancel-compatible pooled = 48 events: 12 matched / 24 fallback / 12 unavailable; 24 anchored / 24 not; draws 36 / 24',
          got == (48, 12, 24, 12, 24, 24, 36, 24), got)
    got = tuple(int(r[f'x_pres_{a}_fa_{b}']) for a in ('matched', 'fallback', 'unavailable') for b in ('anchored', 'unavailable'))
    check('membership: the cross tab (matched / fallback / unavailable x anchored / unavailable) = 12 0 12 12 0 12', got == (12, 0, 12, 12, 0, 12), got)
    a, b = row(M, 'S_FIFO', 'cp_successor_5s', 'pooled', 'all'), row(M, 'S_FIFO', 'cp_no_successor_5s', 'pooled', 'all')
    got = tuple(int(x[c]) for x in (a, b) for c in ('n_events', 'pres_matched', 'pres_fallback', 'pres_unavailable'))
    check('membership: the successor split = 24 (0 / 12 / 12) and 24 (12 / 12 / 0)', got == (24, 0, 12, 12, 24, 12, 12, 0), got)
    r = row(M, 'S_NOPRIO', 'cancel_possible', 'rank2', D1)
    got = tuple(int(r[c]) for c in ('n_events', 'pres_matched', 'pres_fallback', 'pres_unavailable'))
    check('membership: one rank x day cell (S_NOPRIO rank2 day 2) = 4 events 1 / 2 / 1', got == (4, 1, 2, 1), got)
    check('membership: every row reconciles with the summary support and the census', bool(M['reconciled'].all()), M[~M['reconciled']].head(3).to_dict('records'))
    check('membership: one row per view x published group x scope x day; S_NOPRIO has no exhausted_proven row',
          len(M) == 7 * 4 * 5 and not ((M['view'] == 'S_NOPRIO') & (M['group_saved'] == 'exhausted_proven')).any(), len(M))
    S2 = SA.copy()
    m2 = (S2['view'] == 'S_FIFO') & (S2['group'] == 'exhausted_proven') & (S2['scope'] == 'rank2') & (S2['day'] == D1)
    S2.loc[m2, 'n_events'] = S2.loc[m2, 'n_events'] + 1
    bad = K.pull_membership(ctrl, pulls, S2, cen)
    bad = bad[~bad['reconciled']]
    check('membership: a summary support off by one event fails reconciliation on exactly that row (the check can fire)',
          len(bad) == 1 and bad.iloc[0]['group_saved'] == 'exhausted_proven' and bad.iloc[0]['scope'] == 'rank2', bad.to_dict('records'))
    P2 = pulls.copy()
    i = P2.index[P2['window_status'] == 'ok'][0]
    P2.loc[i, 'draws_presence'] = 'fallback' if P2.loc[i, 'draws_presence'] != 'fallback' else 'ok'
    expect('membership: a pulls / controls draw-status disagreement is refused', lambda: K.pull_membership(ctrl, P2, SA, cen), 'disagree')
    C2 = ctrl.copy()
    C2.loc[C2.index[0], 'draw_status'] = 'weird'
    expect('membership: an unknown draw status is refused', lambda: K.pull_membership(C2, pulls, SA, cen), 'unknown presence draw status')
    expect('membership: an event without one family row is refused', lambda: K.pull_membership(ctrl.drop(index=ctrl.index[1]), pulls, SA, cen),
           'lacks one reference family')


def s_predicates(SA, SB):
    for nm, S, exp in (('A', SA, (True, False, False)), ('B', SB, (True, True, True))):
        p = K.predicates(K.pooled_index(S), 'S_FIFO', '250')
        got = (p[K.P1]['value'], p[K.P2]['value'], p[K.P3]['value'])
        check(f'predicates: world {nm} gives BTC enriched / inventory every day / cancel-compatible > exhausted = {exp}', got == exp, got)
    p = K.predicates(K.pooled_index(SA), 'S_FIFO', '250')
    d = p[K.P3]['by_family_feature']
    got = (d['presence'][K.BTC_F]['value'], d['fill_anchored'][K.BTC_F]['value'], d['presence'][K.INV_F]['value'], d['fill_anchored'][K.INV_F]['value'])
    check('predicates: world A detail - the BTC comparison fails on the last day in both families; inventory fails under fill_anchored only',
          got == (False, False, True, False) and d['presence'][K.BTC_F]['by_day'][D3]['value'] is False and d['presence'][K.BTC_F]['by_day'][D0]['value'] is True, got)
    check('predicates: the presence inventory predicate (beside) holds in world A while the declared fill_anchored one fails',
          p[K.P2B]['value'] is True and p[K.P2]['value'] is False and p[K.P2]['by_day'][D1]['diff'] < 0)
    drop = ((SB['view'] == 'S_FIFO') & (SB['group'] == 'cancel_possible') & (SB['control'] == 'fill_anchored') & (SB['scope'] == 'pooled') & (SB['day'] == D3)
            & (SB['D'] == '250') & (SB['feature'] == K.BTC_F))
    p = K.predicates(K.pooled_index(SB[~drop]), 'S_FIFO', '250')
    check('predicates: a missing day row makes the predicate False with the row named (never True)',
          p[K.P1]['value'] is False and p[K.P1]['missing_rows'] == [f'fill_anchored/{D3}'], p[K.P1]['missing_rows'])
    S = SB.copy()
    m = (S['group'] == 'cancel_possible') & (S['feature'] == K.BTC_F) & (S['day'] == D2) & (S['control'] == 'presence')
    S.loc[m, 'p_upper'] = .3
    p = K.predicates(K.pooled_index(S), 'S_FIFO', '250')
    check('predicates: diff > 0 with p_upper 0.3 is not enriched; the diff-only reading beside stays True', p[K.P1]['value'] is False and p[K.P1S]['value'] is True)
    S = SB.copy()
    S.loc[(S['group'] == 'cancel_possible') & (S['feature'] == K.BTC_F), 'n_reps'] = 150
    p = K.predicates(K.pooled_index(S), 'S_FIFO', '250')
    check('predicates: fewer than 200 replicates are never enriched', p[K.P1]['value'] is False and p[K.P1S]['value'] is True)
    p = K.predicates(K.pooled_index(SB), 'S_NOPRIO', '250')
    check('predicates: S_NOPRIO has no exhausted class - comparison not evaluable, value False; its BTC predicate is computed',
          p[K.P3]['evaluable'] is False and p[K.P3]['value'] is False and p[K.P1]['value'] is True)
    p = K.predicates(K.pooled_index(SB), 'S_FIFO', 'union')
    check('predicates: the D = union profile carries the rate features', p[K.P1]['value'] is True and p[K.P2]['value'] is True)


def s_label():
    df = pd.DataFrame(dict(group_saved=['cancel_possible', 'capacity_tight'], group=['pull_window', 'pull_like'], n=[1, 2]))
    expect('label: a cancel-compatible row labelled with the word is refused', lambda: K.label_guard({'t': df}, {}), 'label')
    df2 = pd.DataFrame(dict(group_saved=['cancel_possible', 'capacity_tight'], group=['cancel_compatible_terminal_window', 'pull_like'], n=[1, 2]))
    check('label: the cell scope is the cancel-compatible rows (an unpublished-group cell is not one)', K.label_hits({'t': df2}, {}) == [])
    expect('label: an output column carrying the word is refused', lambda: K.label_guard({'t': df2.rename(columns={'n': 'n_pulls'})}, {}), 'label')
    expect('label: a claims key carrying the word is refused', lambda: K.label_guard({}, {'a': {'pull_rate': 1}}), 'label')
    expect('label: a claims string carrying the word (any case) is refused', lambda: K.label_guard({}, {'a': ['ok', 'a Pull rule']}), 'label')


def s_label_scan(out):
    """Independent text scan of the written outputs (not the tool's guard): every CSV header, every cancel-compatible row, the claims."""
    bad, n_cc = [], 0
    for f in K.OUT_FILES:
        if not f.endswith('.csv'):
            continue
        lines = (out / f).read_text().splitlines()
        if K.LABEL_WORD in lines[0].lower():
            bad.append((f, 'header'))
        hdr = lines[0].split(',')
        if 'group_saved' not in hdr:
            continue
        gi = hdr.index('group_saved')
        for ln in lines[1:]:
            if ln.split(',')[gi] in K.CC_GROUPS:
                n_cc += 1
                if K.LABEL_WORD in ln.lower():
                    bad.append((f, ln[:100]))
    text = (out / K.CLAIMS_FILE).read_text().lower()
    if K.LABEL_WORD in text:
        j = text.index(K.LABEL_WORD)
        bad.append((K.CLAIMS_FILE, text[max(0, j - 60):j + 20]))
    check('label: the word labels no cancel-compatible row, no output column and nothing in the claims (text scan of every output)', not bad, bad[:3])
    check('label: the text scan saw cancel-compatible rows (it can fire)', n_cc > 0, n_cc)


def s_end_to_end(tmp, base):
    pdir, tdir = tmp / 'in' / 'pull', tmp / 'in' / 'take'
    write_pull_world(pdir, base)
    write_take_world(tdir)
    tsha = {f: K.sha256(tdir / f) for f in K.TAKE_FILES}
    kw = dict(pull_dir=pdir, take_dir=tdir, git_fn=GIT_OK, review_take_sha=tsha, log=lambda *_: None)
    out = tmp / 'out'
    out.mkdir()
    (out / 'post_regrade.csv').write_text('sibling\n')              # the sibling tool shares the directory
    K.run(out=out, **kw)
    names = sorted(p.name for p in out.iterdir())
    check('e2e: the run writes exactly the seven declared files beside the sibling file (no partial left, the sibling kept)',
          names == sorted(K.OUT_FILES + ('post_regrade.csv',)) and (out / 'post_regrade.csv').read_text() == 'sibling\n', names)
    R = json.loads((out / K.RECEIPT_FILE).read_text())
    others = [f for f in K.OUT_FILES if f != K.RECEIPT_FILE]
    check('e2e: the receipt binds the sha256 of every other output', set(R['outputs']) == set(others)
          and all(K.sha256(out / f) == R['outputs'][f]['sha256'] for f in others), sorted(R['outputs']))
    check('e2e: the receipt binds every input (8 CSVs + 2 receipts), the tool source and the contract',
          len(R['inputs']) == 10 and all(len(v['sha256']) == 64 for v in R['inputs'].values()) and any(k.endswith('chain_correct_pull_take.py') for k in R['sources'])
          and any(k.endswith('chain_correction_contract.md') for k in R['sources']), (len(R['inputs']), sorted(R['sources'])))
    check('e2e: the receipt records every check passing, CPU and VmHWM', all(v['ok'] for v in R['checks'].values()) and R['cpu_s']['total'] > 0
          and R['vmhwm_mib'] > 0, [k for k, v in R['checks'].items() if not v['ok']])
    CL = json.loads((out / K.CLAIMS_FILE).read_text())
    p = CL['terminal_window_predicates']['headline']
    check('e2e: the claims carry the computed predicates (world A: True / False / False)', (p[K.P1]['value'], p[K.P2]['value'], p[K.P3]['value']) == (True, False, False))
    rp = CL['reproductions']
    check('e2e: the claims carry every reviewer-number reproduction as a recorded predicate (synthetic: no match required)',
          rp['n_items'] == len(rp['items']) >= 40 and all(isinstance(x['match'], bool) for x in rp['items']) and len(rp['not_reproducible']) == 1, rp['n_items'])
    fam = pd.read_csv(out / 'pull_families.csv', dtype=str, keep_default_na=False)
    check('e2e: pull_families.csv has both families on every row, S_FIFO first', (fam['pres_n_obs'] != '').all() and (fam['fa_n_obs'] != '').all()
          and fam['view'].iloc[0] == 'S_FIFO', len(fam))
    rel = pd.read_csv(out / 'take_relabelled.csv', dtype=str, keep_default_na=False)
    check('e2e: take_relabelled.csv carries take-side names only (no saved r_<w> name left as a feature)',
          not rel['feature'].str.match(r'^r_\d').any() and (rel['comparison'] == 'take_vs_control').all(), len(rel))
    s_label_scan(out)
    h0 = K.sha256(out / K.RECEIPT_FILE)
    expect('refuse: write-once when the outputs exist (asserted reason)', lambda: K.run(out=out, **kw), 'write-once')
    check('refuse: the existing outputs are untouched', K.sha256(out / K.RECEIPT_FILE) == h0 and sorted(p.name for p in out.iterdir()) == names)
    out2 = tmp / 'out2'
    out2.mkdir()
    (out2 / 'take_relabelled.csv').write_text('sentinel\n')
    (out2 / 'post_regrade.csv').write_text('sibling\n')
    expect('refuse: write-once when one declared name exists, before any input is read', lambda: K.run(out=out2, **kw), 'refused before any input is read')
    check('refuse: the sentinel and the sibling file are untouched and nothing else is written',
          sorted(p.name for p in out2.iterdir()) == ['post_regrade.csv', 'take_relabelled.csv'] and (out2 / 'take_relabelled.csv').read_text() == 'sentinel\n')
    out4 = tmp / 'out4'
    res = K.run(dry=True, out=out4, **kw)
    check('e2e: dry writes nothing and returns the schema checks and the predicates', not out4.exists() and res['failed_checks'] == []
          and K.P1 in res['predicates']['headline'] and res['would_write']['pull_families.csv'] == len(fam), res.get('failed_checks'))
    expect('refuse: the freeze (the tool files not tracked / unmodified)', lambda: K.run(out=tmp / 'out5', **dict(kw, git_fn=GIT_BAD)), 'freeze')
    expect('refuse: a take input other than the reviewed snapshot', lambda: K.run(out=tmp / 'out6', **dict(kw, review_take_sha=dict(tsha, **{'takes.csv': '0' * 64}))),
           'binding: take')
    pbad = tmp / 'in_bad' / 'pull'
    write_pull_world(pbad, base, corrupt_hash=True)
    expect('refuse: a pull input whose sha256 differs from its receipt', lambda: K.run(out=tmp / 'out7', **dict(kw, pull_dir=pbad)), 'binding: pull')
    pom = tmp / 'in_om' / 'pull'
    write_pull_world(pom, base, drop=lambda S: ((S['view'] == 'S_FIFO') & (S['group'] == 'cancel_possible') & (S['control'] == 'fill_anchored')
                                                & (S['scope'] == 'pooled') & (S['day'] == 'all') & (S['D'] == '250') & (S['feature'] == 'tau_start')))
    expect('refuse: omitted control on the production path (a published key without fill_anchored; its receipt consistent)',
           lambda: K.run(out=tmp / 'out8', **dict(kw, pull_dir=pom)), 'omitted control')
    with Mutant('drop_fill_anchored'):
        expect('refuse: a set mutant never writes (run)', lambda: K.run(out=tmp / 'out9', **kw), 'mutant')
        expect('refuse: a set mutant never prints (dry)', lambda: K.run(dry=True, out=tmp / 'out9', **kw), 'mutant')
    check('refuse: nothing written by any refused run', not any((tmp / f'out{i}').exists() for i in (5, 6, 7, 8, 9)))


def main():
    tmp = Path(tempfile.mkdtemp(prefix='chain_correct_pt_selftest_'))
    os.environ.pop('MT_CC_MUT', None)
    try:
        ctrl, pulls = pull_world()
        cen = census(ctrl, pulls)
        SA, SB = summary(ctrl, pulls, VARIANT_A), summary(ctrl, pulls, VARIANT_B)
        section('engine pins', s_engine)
        section('sign', s_sign)
        section('take', s_take)
        section('omitted control', s_omitted, SA)
        section('membership', s_membership, ctrl, pulls, cen, SA)
        section('predicates', s_predicates, SA, SB)
        section('label', s_label)
        section('end to end', s_end_to_end, tmp, (ctrl, pulls, cen, SA))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
        os.environ.pop('MT_CC_MUT', None)
    n_bad = sum(not ok for _, ok, _ in RES)
    for name, ok, info in RES:
        print(('PASS ' if ok else 'FAIL ') + name + ('' if ok else f'  :: {info}'))
    print(json.dumps(dict(checks=len(RES), failed=n_bad, vmhwm_mib=round(K.hwm(), 1))))
    sys.exit(1 if n_bad else 0)


if __name__ == '__main__':
    main()
