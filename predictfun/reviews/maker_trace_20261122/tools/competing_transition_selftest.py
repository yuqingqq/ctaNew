"""competing_transition_selftest.py - synthetic, in-process proof of competing_transition.py (P-2026-005 MAKER-TRACE,
COMPETING-TRANSITION-AND-CONTINUATION v1; competing_transition_contract.md sections 1-4 and their acceptance items, commit 9909beacc). Every
fixture is planted here: episodes whose saved columns agree with their outcome (YES ticks around 50, frames 40 ms apart), reconcile rows for
the known two-side sub-labels; no data file of the packet is read.
CHECKS (each named; every planted error MT_CT_MUT fails its named check while the base passes it):
 all-classes fixture  one episode per declared outcome (delta 2; 8 saved rows for the four known two-side sub-labels, none for the unknown)
                      -> 18 outcomes mapped once, mutual exclusion, totals, end-type identities; known-bad inputs refused (an overshooting
                      'restoring' price, a same-side 'other_side' class, a larger bound population, two-side rows off the end pair / of
                      another class / disagreeing). swap_labels -> class-direction identity; drop_competitors -> episode totals;
                      blank_competitors -> mutual exclusion; guess_sub -> two-side sub-label provenance.
 harmonic toy         strata (4: 3 r + 1 c | 4: 1 r), (2: 0 r | 6: 3 r + 1 c), (3: 3 r | none): RD = (2 x 0.5 - 1.5 x 0.5) / 3.5, W = 3.5,
                      the single-group stratum excluded, censor bounds 0 .. 0.75 / 3.5; the same toy through the pipeline (toy world).
                      equal_weights / censor_upper_drop fail it. Population bounds 3/18 .. 9/18 on the all-classes fixture.
 timing toy           (lower, upper, endpoint, next frame) = (0, 110, 60, True) and (60, 130, 90, False); medians 30 / 120 / 75.
                      timing_prev fails it.
 planted world        8 markets on 4 days; per market x side 'with' 3/4 vs 'flat' 1/4 respond -> btc_with RD 0.5 at D0 and LAG, every day
                      and side +, support adequate -> adequate_direction_consistent; the four other contrasts thin; LAG-missing episodes
                      counted, not dropped; tables (fractions sum to 1, outcomes.csv, rd.csv raw counts beside, support.csv).
 swapped-LAG world    D0 +0.5, LAG -0.5 -> adequate_inconsistent, timing_sensitive.
 day-split world      days 1-2 RD 1.0, days 3-4 RD -0.25 (pooled 0.375, 2 of 4 days) -> adequate_inconsistent; days_any fails it.
 time-left world      (10,30] vs (60,120] in the same market x side x delta -> RD 0.5 over 16 strata, consistent; tl_own_strata fails it.
 too-small world      12 / 12 common-support episodes but 60 / 60 over the whole groups -> thin; floor_all_population fails it.
 inputs / e2e         bindings (sha256 / unbound / missing), the producing receipt's frozen definitions, reader interfaces and nulls, label
                      sets; the production path end to end through execute() as a rehearsal on a synthetic source tree (parquet + csv +
                      receipt bound by sha256): dry (prints, writes nothing) and run (the seven outputs, receipt hashes, write-once on a
                      second run), a tampered input and an untracked receipt refused.
 production           MT_CT_MUT set or undeclared refused before any input is read; the full run refused before any input (the freeze);
                      an unknown command line refused.
Usage (run_light, MT_RUN=v2): competing_transition_selftest.py"""
import json
import os
import shutil
import sys
import tempfile
import time
import traceback
from pathlib import Path

sys.dont_write_bytecode = True

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import competing_transition as CTR  # noqa: E402

C = CTR.C
NAN = float('nan')
RES = []
DAYS = tuple(C.MEASURE_DAYS)
OK_ST = dict(status='ok', btc='flat', fg='(0,1]', depth='(50,200]', tl='(60,120]')
MISS_ST = dict(status='lag_missing', btc='missing', fg='missing', depth='missing', tl='')
COMP_CYCLE = ('other_improve', 'other_retreat', 'two_side_both_up', 'two_side_sub_unknown', 'same_retreat_further')
TOY = dict(n1=[4, 2, 3], r1=[3, 0, 3], c1=[1, 0, 0], n0=[4, 6, 0], r0=[1, 3, 0], c0=[0, 1, 0])
TOY_RD, TOY_W, TOY_HI = 0.25 / 3.5, 3.5, 0.75 / 3.5


def check(name, ok, info=''):
    RES.append((name, bool(ok), str(info)[:900]))


def expect(name, fn, needle):
    try:
        fn()
    except Exception as e:  # noqa: BLE001
        check(name, needle in str(e), f'{type(e).__name__}: {e}')
        return
    check(name, False, 'did not raise')


def section(name, fn, *a):
    try:
        return fn(*a)
    except Exception as e:  # noqa: BLE001
        check(f'{name}: section crashed', False, f'{type(e).__name__}: {e} | {traceback.format_exc()[-1800:]}')
        return None


def run_chk(fn):
    """A named check: True passes; any other return value or an exception fails (the text is kept)."""
    try:
        r = fn()
    except Exception as e:  # noqa: BLE001
        return False, f'{type(e).__name__}: {e}'
    return (r is True), ('' if r is True else str(r))


def _eq(a, b, tol=1e-12):
    try:
        a, b = float(a), float(b)
    except (TypeError, ValueError):
        return False
    return (np.isnan(a) and np.isnan(b)) or abs(a - b) <= tol * max(1., abs(a), abs(b))


class Mut:
    """MT_CT_MUT=m inside the block (removed on exit)."""

    def __init__(self, m):
        self.m = m

    def __enter__(self):
        os.environ[CTR.MUT_ENV] = self.m
        return self

    def __exit__(self, *a):
        os.environ.pop(CTR.MUT_ENV, None)
        return False


# ------------------------------------------------------------------ synthetic episodes ------------------------------------------------------------------
def st(**kw):
    d = dict(OK_ST)
    d.update(kw)
    return d


def ep_row(day, mid, eid, side, outcome, s0, s1, delta):
    """One synthetic episode whose saved columns agree with its outcome: best_prev 50, the retreat of delta ticks, frames 40 ms apart; a response
    is next-frame (60 ms); a competing transition ends at f + 2."""
    sg = 1. if side == 'bid' else -1.
    so = 'ask' if side == 'bid' else 'bid'
    bp = 50.
    tp = 1_000_000 + 10_000 * eid
    t0 = tp + 40
    f = 10 + 10 * eid
    r = dict(day=day, mid=mid, episode_id=eid, side=side, frame=f, t_prev_ms=tp, t_start_ms=t0, best_prev=bp, best_retreat=bp - sg * delta,
             other_best=bp + sg * 3, delta_ticks=float(delta), delta_bin='1' if delta < 1.5 else ('2' if delta < 2.5 else '>=3'), end_type='',
             end_reason='', end_frame=-1, end_class='', t_end_ms=-1, exposure_ms=0, exposure_lo_ms=0, no_frame_between=False, resp_type='',
             resp_best=NAN, resp_lo_ms=-1, resp_hi_ms=-1)
    g = CTR.GROUP_OF[outcome]
    if g == 'same_response':
        rt = outcome[len('same_'):]
        rb = dict(restoring=bp, overshoot=bp + sg, partial=bp - sg)[rt]
        r.update(end_type='response', end_reason='response', end_frame=f + 1, end_class=f'improve_{side}', t_end_ms=t0 + 60, exposure_ms=60,
                 exposure_lo_ms=0, no_frame_between=True, resp_type=rt, resp_best=rb, resp_lo_ms=t0, resp_hi_ms=t0 + 60)
    elif g in ('other_side', 'two_side', 'further_retreat'):
        if g == 'other_side':
            er, ec = 'other_side', ('improve_' if outcome == 'other_improve' else 'retreat_') + so
        elif g == 'two_side':
            er, ec = 'two_side', 'two_side'
        else:
            er, ec = 'same_side_retreat', f'retreat_{side}'
        r.update(end_type='censored_other', end_reason=er, end_frame=f + 2, end_class=ec, t_end_ms=t0 + 80, exposure_ms=80, exposure_lo_ms=40)
    elif g == 'horizon':
        r.update(end_type='no_response_1s', end_reason='horizon', t_end_ms=t0 + 1000, exposure_ms=1000, exposure_lo_ms=1000)
    else:
        rs = outcome[len('censored_'):]
        ec = CTR.CENSOR_CLASS[rs][0]
        r.update(end_type='censored_gap', end_reason=rs, end_frame=(f + 2) if ec else -1, end_class=ec, t_end_ms=t0 + 40, exposure_ms=40,
                 exposure_lo_ms=40)
    for v, s in (('D0', s0), ('LAG', s1)):
        r.update({f'state_status_{v}': s['status'], f'btc_5s_aligned_{v}': s['btc'], f'fair_gap_bin_{v}': s['fg'], f'depth_own_bin_{v}': s['depth'],
                  f'time_left_bin_{v}': s['tl']})
    return r


def rc_pair(r, sub):
    """Two saved reconcile rows of a two-side end pair (the same frame pair, class, sub-label and times)."""
    base = dict(day=r['day'], mid=r['mid'], frame=r['end_frame'], t_open_ms=r['t_start_ms'] + r['exposure_lo_ms'], t_create_ms=r['t_end_ms'],
                pair_status='in_census', cls='two_side', sublabel=sub)
    return [dict(base), dict(base)]


def world(spec):
    """spec rows (day, mid, side, outcome, D0 state, LAG state, delta) -> (episodes frame, reconcile frame); ids per market; every episode also
    gets an unrelated saved row at its own retreat pair (the join must ignore it)."""
    E, RC, nxt = [], [], {}
    for day, mid, side, outcome, s0, s1, delta in spec:
        eid = nxt.get((day, mid), 0)
        nxt[(day, mid)] = eid + 1
        r = ep_row(day, mid, eid, side, outcome, s0, s1, delta)
        E.append(r)
        if outcome.startswith('two_side_') and outcome != 'two_side_sub_unknown':
            RC.extend(rc_pair(r, outcome[len('two_side_'):]))
        RC.append(dict(day=day, mid=mid, frame=r['frame'], t_open_ms=r['t_prev_ms'], t_create_ms=r['t_start_ms'], pair_status='in_census',
                       cls=f'retreat_{side}', sublabel=''))
    return pd.DataFrame(E), pd.DataFrame(RC)


def arrays(E, RC):
    return CTR.ep_arrays(E, 'fixture'), CTR.rc_arrays(RC, 'fixture reconcile')


def analyse_world(W, tables=True):
    E, RC = W
    A, R = arrays(E, RC)
    return CTR.analyse(A, R, len(E), tables=tables)


def all_classes():
    return world([(DAYS[0], 7, 'bid' if i % 2 == 0 else 'ask', o, dict(OK_ST), dict(OK_ST), 2) for i, o in enumerate(CTR.OUTCOMES)])


def planted(effect=None, lag_swap=False, extra_missing=4):
    """8 markets (2 per day), both sides, delta 1, time left (60,120]; per market x side 4 'with' + 4 'flat' episodes; effect(day index) ->
    (responses of the 4 'with', responses of the 4 'flat'); the non-responses cycle through competing outcomes; extra_missing episodes per day
    in another D0 bin and LAG-missing."""
    eff = effect or (lambda di: (3, 1))
    spec, k = [], 0
    for di, day in enumerate(DAYS):
        nw, nf = eff(di)
        for mm in (1, 2):
            mid = 100 * (di + 1) + mm
            for side in CTR.SIDES:
                for lab, nr in (('with', nw), ('flat', nf)):
                    for j in range(4):
                        if j < nr:
                            o = 'same_restoring'
                        else:
                            o = COMP_CYCLE[k % len(COMP_CYCLE)]
                            k += 1
                        lag = ('flat' if lab == 'with' else 'with') if lag_swap else lab
                        spec.append((day, mid, side, o, st(btc=lab), st(btc=lag), 1))
            if mm == 1:
                for j in range(extra_missing):
                    spec.append((day, mid, 'bid', 'other_improve', st(btc='missing', tl='>300'), dict(MISS_ST), 1))
    return world(spec)


def tl_world():
    """8 markets; per market x side 4 episodes with time left (10,30] (3 respond) and 4 with (60,120] (1 responds); btc flat everywhere."""
    spec = []
    for di, day in enumerate(DAYS):
        for mm in (1, 2):
            mid = 100 * (di + 1) + mm
            for side in CTR.SIDES:
                for tl, nr in (('(10,30]', 3), ('(60,120]', 1)):
                    for j in range(4):
                        spec.append((day, mid, side, 'same_restoring' if j < nr else 'other_improve', st(tl=tl), st(tl=tl), 1))
    return world(spec)


def small_world():
    """6 markets (2, 2, 1, 1 per day): a mixed stratum (bid, delta 1) of 2 'with' + 2 'flat' (1 response each) and two single-group strata
    (ask delta 1: 8 'with'; ask delta 2: 8 'flat'; 4 responses each) per market."""
    spec = []
    for di, day in enumerate(DAYS):
        for mm in range(2 if di < 2 else 1):
            mid = 100 * (di + 1) + mm + 1
            for lab in ('with', 'flat'):
                for j in range(2):
                    spec.append((day, mid, 'bid', 'same_restoring' if j == 0 else 'other_improve', st(btc=lab), st(btc=lab), 1))
            for lab, delta in (('with', 1), ('flat', 2)):
                for j in range(8):
                    spec.append((day, mid, 'ask', 'same_restoring' if j < 4 else 'other_retreat', st(btc=lab), st(btc=lab), delta))
    return world(spec)


def toy_world():
    """The harmonic toy through the pipeline (btc_with, D0 = LAG) in market 1 of day 1: bid delta 1 = with 4 (3 responses + 1 censored) vs flat
    4 (1 response); ask delta 1 = with 2 (none) vs flat 6 (3 responses + 1 censored); ask delta 2 = with 3 responses only (single group); one
    other-bin episode on each other day."""
    spec = []

    def add(side, lab, outs, delta=1):
        for o in outs:
            spec.append((DAYS[0], 1, side, o, st(btc=lab), st(btc=lab), delta))
    add('bid', 'with', ['same_restoring'] * 3 + ['censored_empty_side'])
    add('bid', 'flat', ['same_restoring'] + ['other_improve'] * 3)
    add('ask', 'with', ['other_retreat'] * 2)
    add('ask', 'flat', ['same_restoring'] * 3 + ['censored_empty_side'] + ['other_improve'] * 2)
    add('ask', 'with', ['same_restoring'] * 3, delta=2)
    for d in DAYS[1:]:
        spec.append((d, 2, 'bid', 'other_improve', st(btc='missing'), st(btc='missing'), 1))
    return world(spec)


def fixtures():
    return dict(all=all_classes(), planted=planted(), lagswap=planted(lag_swap=True),
                daysplit=planted(effect=lambda di: (4, 0) if di < 2 else (1, 2), extra_missing=0), tl=tl_world(), small=small_world(), toy=toy_world())


def src_receipt(n_ep, n_rc, sha_ep='0' * 64, sha_rc='0' * 64):
    """A synthetic producing receipt with this tool's frozen definitions (the real one is checked in the dry run)."""
    K = dict(CTR.SRC_CONST)
    K.update(TWO_SUB=list(CTR.TWO_SUB), UNK_SUB=list(CTR.UNK_SUB))
    return json.loads(json.dumps(dict(version=CTR.SRC_VERSION, kind='run', mutant='', run='v2', days=list(DAYS), constants=K, counts=dict(episodes=n_ep),
                                      outputs={'episodes.parquet': dict(rows=n_ep, sha256=sha_ep), 'reconcile.csv': dict(rows=n_rc, sha256=sha_rc)})))


def frac_sums_ok(F):
    G = F[F['n_group'] > 0].groupby(['variant', 'contrast', 'group', 'scope', 'level'])['fraction'].sum()
    return bool(len(G) and np.allclose(G.to_numpy(dtype=np.float64), 1.0))


# ------------------------------------------------------------------ named checks (shared by the base and the planted errors) ------------------------------------------------------------------
def chk_all_classes(FX):
    E, RC = FX['all']
    A, R = arrays(E, RC)
    _, lab, grp, sub, nrow, recon = CTR.outcome_table(A, R, len(E))
    if lab.tolist() != list(CTR.OUTCOMES):
        return f'labels {lab.tolist()}'
    if recon['outcome_counts'] != {o: 1 for o in CTR.OUTCOMES}:
        return f'counts {recon["outcome_counts"]}'
    if recon['two_side_sub'] != dict(n_two_side=5, n_sub_known=4, n_sub_unknown=1, saved_rows_matched=8):
        return f'two-side {recon["two_side_sub"]}'
    if recon['end_type_counts'] != dict(response=3, censored_other=8, censored_gap=6, no_response_1s=1):
        return f'end types {recon["end_type_counts"]}'
    if not (recon['totals_equal_population'] and recon['mutual_exclusion'] and recon['end_type_identity'] and recon['class_direction_violations'] == 0):
        return f'predicates {recon}'
    return True


def chk_harmonic(FX):
    P = CTR.rd_pool(TOY['n1'], TOY['r1'], TOY['c1'], TOY['n0'], TOY['r0'], TOY['c0'])
    if not (_eq(P['rd'], TOY_RD) and _eq(P['W'], TOY_W) and P['n_strata'] == 2 and np.allclose(P['w'], [2., 1.5, 0.])):
        return f'toy rd {P["rd"]} W {P["W"]} strata {P["n_strata"]} w {P["w"].tolist()}'
    B = analyse_world(FX['toy'], tables=False)['BLK'][('D0', 'btc_with')]
    p, s = B['pooled'], B['support']
    if not (_eq(p['rd'], TOY_RD) and _eq(p['W'], TOY_W) and p['n_strata'] == 2 and s['n_exposed_single_group'] == 3 and (p['n1'], p['n0']) == (6, 10)):
        return f'toy world rd {p["rd"]} W {p["W"]} strata {p["n_strata"]} single {s["n_exposed_single_group"]} n {(p["n1"], p["n0"])}'
    return True


def chk_censor(FX):
    P = CTR.rd_pool(TOY['n1'], TOY['r1'], TOY['c1'], TOY['n0'], TOY['r0'], TOY['c0'])
    if not (abs(P['lo']) <= 1e-12 and _eq(P['hi'], TOY_HI)):
        return f'toy bounds {P["lo"]} .. {P["hi"]}'
    lo, hi = float(CTR.risk_lo([4], [10])[0]), float(CTR.risk_hi([4], [2], [10])[0])
    if not (_eq(lo, 0.4) and _eq(hi, 0.6)):
        return f'risk bounds {lo} .. {hi}'
    p = analyse_world(FX['toy'], tables=False)['BLK'][('D0', 'btc_with')]['pooled']
    if not (abs(p['lo']) <= 1e-12 and _eq(p['hi'], TOY_HI) and (p['c1'], p['c0']) == (1, 1)):
        return f'toy world bounds {p["lo"]} .. {p["hi"]} censored {(p["c1"], p["c0"])}'
    E, RC = FX['all']
    A, R = arrays(E, RC)
    pb = CTR.population_bounds(CTR.outcome_table(A, R, len(E))[2])
    if not (_eq(pb['lower'], 3 / 18) and _eq(pb['upper'], 9 / 18) and (pb['responses'], pb['censored'], pb['n']) == (3, 6, 18)):
        return f'population bounds {pb}'
    return True


def chk_timing():
    A = dict(day=np.array([DAYS[0]] * 3, dtype=object), mid=np.array([1, 1, 1], np.int64), episode_id=np.array([0, 1, 2], np.int64),
             side=np.array(['bid', 'ask', 'bid'], dtype=object), resp_type=np.array(['restoring', 'partial', ''], dtype=object),
             t_prev_ms=np.array([1000, 2000, 3000], np.int64), t_start_ms=np.array([1050, 2040, 3040], np.int64),
             resp_lo_ms=np.array([1050, 2100, -1], np.int64), resp_hi_ms=np.array([1110, 2130, -1], np.int64),
             no_frame_between=np.array([True, False, False]))
    grp = np.array(['same_response', 'same_response', 'other_side'], dtype=object)
    T, S = CTR.timing_table(A, grp)
    got = [tuple(int(x) for x in r) + (bool(nf),) for r, nf in zip(T[['lower_ms', 'upper_ms', 'endpoint_ms']].to_numpy().tolist(), T['next_frame'].tolist())]
    if got != [(0, 110, 60, True), (60, 130, 90, False)]:
        return f'timing rows {got}'
    if not (S['n_responses'] == 2 and S['next_frame_n'] == 1 and _eq(S['next_frame_fraction'], 0.5) and _eq(S['lower_ms']['p50'], 30.)
            and _eq(S['upper_ms']['p50'], 120.) and _eq(S['endpoint_ms']['p50'], 75.) and S['lower_zero_n'] == 1):
        return f'timing summary {S}'
    return True


def chk_small(FX):
    Rz = analyse_world(FX['small'], tables=False)
    s = Rz['BLK'][('D0', 'btc_with')]['support']
    got = (Rz['READ']['btc_with'], s['n_exposed_mixed'], s['n_reference_mixed'], s['n_exposed_all'], s['n_reference_all'], s['floor_episodes'],
           s['n_markets_both_ge5'])
    return True if got == ('thin', 12, 12, 60, 60, False, 0) else f'too-small world {got}'


def chk_tl(FX):
    Rz = analyse_world(FX['tl'], tables=False)
    p = Rz['BLK'][('D0', 'time_left')]['pooled']
    got = (Rz['READ']['time_left'], round(p['rd'], 12) if np.isfinite(p['rd']) else None, p['n_strata'], Rz['READ']['btc_with'])
    return True if got == ('adequate_direction_consistent', 0.5, 16, 'thin') else f'time-left world {got}'


def chk_daysplit(FX):
    Rz = analyse_world(FX['daysplit'], tables=False)
    B = Rz['BLK'][('D0', 'btc_with')]
    got = (Rz['READ']['btc_with'], round(B['pooled']['rd'], 12), [round(x['rd'], 12) for x in B['days']], B['day_signs'], B['adequate'],
           Rz['BLK'][('LAG', 'btc_with')]['adequate'])
    want = ('adequate_inconsistent', 0.375, [1.0, 1.0, -0.25, -0.25], [1, 1, -1, -1], True, True)
    return True if got == want else f'day-split world {got}'


# ------------------------------------------------------------------ tests ------------------------------------------------------------------
def test_all_classes(FX):
    ok, info = run_chk(lambda: chk_all_classes(FX))
    check('S1 all-classes fixture: 18 declared outcomes mapped once each, mutual exclusion, totals, end-type identities; 4 known two-side '
          'sub-labels from 8 saved rows, 1 unknown without a row', ok, info)
    E, RC = FX['all']
    E4 = E.copy()
    E4.loc[0, 'resp_best'] = 51.0                              # row 0: same_restoring (bid, best_prev 50) whose price now overshoots
    expect('S1 refused: a restoring response whose price overshoots (class-direction identity)', lambda: CTR.outcome_table(*arrays(E4, RC), len(E4)),
           'class-direction identity')
    i = int(np.flatnonzero(E['end_reason'].to_numpy(dtype=object) == 'other_side')[0])
    E5 = E.copy()
    E5.loc[i, 'end_class'] = 'improve_' + str(E.loc[i, 'side'])
    expect('S1 refused: an other-side end carrying a same-side class (no declared outcome: mutual exclusion)',
           lambda: CTR.outcome_table(*arrays(E5, RC), len(E5)), 'mutual exclusion')
    expect('S1 refused: a bound population larger than the outcome rows (episode totals)', lambda: CTR.outcome_table(*arrays(E, RC), len(E) + 1),
           'episode totals')
    i5 = CTR.OUTCOMES.index('two_side_both_up')
    key = (str(E.loc[i5, 'day']), int(E.loc[i5, 'mid']), int(E.loc[i5, 'end_frame']))
    js = RC.index[(RC['day'] == key[0]) & (RC['mid'] == key[1]) & (RC['frame'] == key[2])].tolist()
    check('fixture: two saved rows of the both_up end pair', len(js) == 2, js)
    for nm, col, val, both, needle in (('a time off the episode end pair', 't_create_ms', 1, True, 'span'),
                                      ('rows of another class', 'cls', 'retreat_bid', True, 'not an in-census two_side pair'),
                                      ('rows that disagree on the sub-label', 'sublabel', 'narrow', False, 'disagree')):
        RCx = RC.copy()
        for j in (js if both else js[:1]):
            RCx.loc[j, col] = (int(RCx.loc[j, col]) + val) if col == 't_create_ms' else val
        expect(f'S1 two-side join refused: {nm}', lambda RCx=RCx: CTR.outcome_table(*arrays(E, RCx), len(E)), needle)


def test_toys(FX):
    P = CTR.rd_pool(TOY['n1'], TOY['r1'], TOY['c1'], TOY['n0'], TOY['r0'], TOY['c0'])
    d = P['d'].tolist()
    check('S2 harmonic toy by hand: w = (2, 1.5, 0), RD_k = (0.5, -0.5, -), RD = 0.25 / 3.5 (equal weights would give 0, n-weights 0)',
          np.allclose(P['w'], [2., 1.5, 0.]) and _eq(d[0], 0.5) and _eq(d[1], -0.5) and _eq(P['rd'], TOY_RD), (P['w'].tolist(), d, P['rd']))
    E = CTR.rd_pool([3], [3], [0], [0], [0], [0])
    check('S2 a single-group stratum carries no weight (RD NaN, W 0)', np.isnan(E['rd']) and E['W'] == 0. and E['n_strata'] == 0, E['rd'])


def test_planted(FX):
    Rz = analyse_world(FX['planted'])
    for v in CTR.VARIANTS:
        B = Rz['BLK'][(v, 'btc_with')]
        p, s, t = B['pooled'], B['support'], B['top']
        check(f'planted {v} btc_with: harmonic RD 0.5 over 16 mixed strata (W 32), every day and side +',
              _eq(p['rd'], 0.5) and _eq(p['W'], 32.) and p['n_strata'] == 16 and B['day_signs'] == [1, 1, 1, 1]
              and [x['sign'] for x in B['sides']] == [1, 1], (p, B['day_signs']))
        check(f'planted {v} btc_with: support 64 / 64 episodes, 48 / 16 responses, 16 / 48 competing, 8 markets -> adequate',
              (s['n_exposed_mixed'], s['n_reference_mixed'], s['responses_exposed_mixed'], s['responses_reference_mixed'], s['competing_exposed_mixed'],
               s['competing_reference_mixed'], s['n_markets_both_ge5']) == (64, 64, 48, 16, 16, 48, 8) and B['adequate'], s)
        check(f'planted {v} btc_with: concentration - each of the 8 markets carries 1/8 of the RD and of the weight',
              _eq(t['share_of_rd'], 0.125) and _eq(t['weight_share'], 0.125) and _eq(t['max_abs_share'], 0.125), t)
    check('planted: btc_with adequate_direction_consistent; the four other contrasts thin',
          Rz['READ'] == dict(btc_with='adequate_direction_consistent', btc_against='thin', fair_gap='thin', depth_own='thin', time_left='thin'), Rz['READ'])
    s0, s1 = Rz['BLK'][('D0', 'btc_with')]['support'], Rz['BLK'][('LAG', 'btc_with')]['support']
    check('planted: 16 episodes in another D0 bin are LAG state-missing (counted, never dropped)',
          (s0['n_other_bins'], s0['n_state_missing'], s1['n_other_bins'], s1['n_state_missing']) == (16, 0, 0, 16), (s0, s1))
    c = Rz['claims']
    cb = c['contrasts']['btc_with']
    check('planted claims: readings counted, D0 / LAG signs agree, 4 days carry the sign at both clocks, not timing-sensitive',
          c['readings']['counts'] == dict(adequate_direction_consistent=1, adequate_inconsistent=0, thin=4) and c['readings']['all_thin_or_inconsistent'] is False
          and cb['d0_lag_sign_agree'] and cb['days_carrying_pooled_sign'] == dict(D0=4, LAG=4) and not cb['timing_sensitive'], (c['readings'], cb))
    T = Rz['tables']
    F = T['fractions.csv']
    sel = (F['variant'] == 'D0') & (F['contrast'] == 'btc_with') & (F['group'] == 'exposed') & (F['scope'] == 'all') & (F['level'] == 'outcome')
    fr = dict(zip(F.loc[sel, 'outcome'].tolist(), F.loc[sel, 'fraction'].tolist()))
    check('fractions.csv: exposed same_restoring 0.75 (48 / 64); every populated (clock, contrast, group, scope, level) sums to 1',
          _eq(fr.get('same_restoring'), 0.75) and frac_sums_ok(F) and len(F) == 24 + 10 * 4 * 24, (fr.get('same_restoring'), len(F)))
    O = T['outcomes.csv']
    check('outcomes.csv: one row per episode with its outcome and the exposed / reference / out group per clock and contrast',
          len(O) == len(Rz['lab']) and O['outcome'].tolist() == Rz['lab'].tolist() and set(O['grp_btc_with_D0']) == {'exposed', 'reference', 'out'}
          and int((O['grp_btc_with_LAG'] == 'out').sum()) == 16, len(O))
    Rd = T['rd.csv']
    pr = Rd[(Rd['variant'] == 'D0') & (Rd['contrast'] == 'btc_with') & (Rd['scope'] == 'pooled')].iloc[0]
    check('rd.csv pooled row: RD 0.5 with the raw all-population counts beside it (64 / 64, risks 0.75 / 0.25, crude 0.5)',
          _eq(pr['rd'], 0.5) and int(pr['n1_all']) == 64 and int(pr['n0_all']) == 64 and _eq(pr['p1_lo_all'], 0.75) and _eq(pr['p0_lo_all'], 0.25)
          and _eq(pr['rd_crude_all'], 0.5) and len(Rd) == 10 * 15, (pr.to_dict(), len(Rd)))
    Sp = T['support.csv']
    check('support.csv: one row per clock x contrast with the counts and the contrast reading', len(Sp) == 10
          and set(Sp['reading']) == {'adequate_direction_consistent', 'thin'} and int(Sp['adequate_clock'].sum()) == 2, len(Sp))


def test_worlds(FX):
    Rz = analyse_world(FX['lagswap'], tables=False)
    c = Rz['claims']['contrasts']['btc_with']
    check('swapped-LAG world: D0 +0.5, LAG -0.5 (both adequate) -> adequate_inconsistent, timing_sensitive',
          Rz['READ']['btc_with'] == 'adequate_inconsistent' and _eq(c['rd']['D0'], 0.5) and _eq(c['rd']['LAG'], -0.5) and c['timing_sensitive']
          and not c['d0_lag_sign_agree'] and c['adequate'] == dict(D0=True, LAG=True), c)


def test_mutants(FX):
    cases = (('swap_labels', 'S1 class-direction identity (all-classes fixture)', lambda: chk_all_classes(FX), 'class-direction identity'),
             ('drop_competitors', 'S1 episode totals (all-classes fixture)', lambda: chk_all_classes(FX), 'episode totals'),
             ('blank_competitors', 'S1 mutual exclusion (all-classes fixture)', lambda: chk_all_classes(FX), 'mutual exclusion'),
             ('guess_sub', 'S1 two-side sub-label provenance (all-classes fixture)', lambda: chk_all_classes(FX), 'sub-label provenance'),
             ('equal_weights', 'S2 harmonic weights n1 n0 / (n1 + n0) (2-stratum toy and toy world)', lambda: chk_harmonic(FX), None),
             ('censor_upper_drop', 'S2 censor bounds (toy, toy world, population)', lambda: chk_censor(FX), None),
             ('timing_prev', 'S4 timing bounds (hand values)', chk_timing, None),
             ('floor_all_population', 'S3 floor within the mixed strata (too-small world -> thin)', lambda: chk_small(FX), None),
             ('tl_own_strata', 'S2 the time_left contrast omits its own bin (time-left world)', lambda: chk_tl(FX), None),
             ('days_any', 'S3 day rule >= 3 of 4 days (day-split world -> adequate_inconsistent)', lambda: chk_daysplit(FX), None))
    check('every declared MT_CT_MUT has a case', sorted(c[0] for c in cases) == sorted(CTR.MUTANTS), [c[0] for c in cases])
    for m, name, fn, needle in cases:
        ok, info = run_chk(fn)
        check(f'base passes: {name}', ok, info)
        with Mut(m):
            okm, infom = run_chk(fn)
        check(f'planted error {m} caught by: {name}', (not okm) and (needle is None or needle in infom), infom or 'passed under the planted error')


def test_input_refusals(tmp, FX):
    p = tmp / 'bind.txt'
    p.write_text('x')
    good = CTR._sha(p)
    hs = {}
    check('binding: a file whose sha256 the producing receipt records is bound', CTR.bind(p, {'outputs': {'bind.txt': {'sha256': good}}}, 'bind.txt', hs) == good
          and len(hs) == 1, hs)
    expect('binding refused: a different sha256', lambda: CTR.bind(p, {'outputs': {'bind.txt': {'sha256': '0' * 64}}}, 'bind.txt', {}),
           'differs from the producing receipt')
    expect('binding refused: no recorded sha256', lambda: CTR.bind(p, {'outputs': {}}, 'bind.txt', {}), 'unbound input')
    expect('binding refused: a missing file', lambda: CTR.bind(tmp / 'nope.txt', {'outputs': {'nope.txt': {'sha256': good}}}, 'nope.txt', {}), 'is missing')
    rec = src_receipt(10, 5)
    check('producing receipt: the frozen definitions accepted (rehearsal)', CTR.check_source_receipt(rec, True) == 10)
    for nm, mod, needle in (('a changed contrast description', lambda r: r['constants']['CONTRASTS'][0].__setitem__(2, 'btc_5s flat '), 'frozen definitions'),
                            ('a pilot receipt', lambda r: r.__setitem__('kind', 'pilot'), 'full run'),
                            ('a planted error in the producer', lambda r: r.__setitem__('mutant', 'state_leak'), 'full run'),
                            ('another day set', lambda r: r.__setitem__('days', list(DAYS[:3])), 'measurement days'),
                            ('disagreeing episode counts', lambda r: r['outputs']['episodes.parquet'].__setitem__('rows', 11), 'episode counts disagree'),
                            ('a missing TWO_SUB', lambda r: r['constants'].pop('TWO_SUB'), 'TWO_SUB')):
        r2 = json.loads(json.dumps(rec))
        mod(r2)
        expect(f'producing receipt refused: {nm}', lambda r2=r2: CTR.check_source_receipt(r2, True), needle)
    expect('producing receipt refused: not the contract population (3,796) outside a rehearsal', lambda: CTR.check_source_receipt(rec, False),
           'contract population')
    E, RC = FX['all']
    E.drop(columns=['resp_best']).to_parquet(tmp / 'lack.parquet', index=False)
    expect('reader refused: episodes.parquet lacking a read column', lambda: CTR.read_episodes(tmp / 'lack.parquet'), 'lacks')
    RC.drop(columns=['sublabel']).to_csv(tmp / 'lack.csv', index=False)
    expect('reader refused: reconcile.csv lacking a read column', lambda: CTR.read_reconcile(tmp / 'lack.csv'), 'lacks')
    E2 = E.copy()
    E2['end_class'] = E2['end_class'].astype(object)
    E2.loc[0, 'end_class'] = None
    expect('reader refused: a null in a string column', lambda: CTR.ep_arrays(E2, 'fixture'), 'holds nulls')
    E3 = E.copy()
    E3['no_frame_between'] = E3['no_frame_between'].astype(np.int64)
    expect('reader refused: a non-boolean next-frame flag', lambda: CTR.ep_arrays(E3, 'fixture'), 'not boolean')
    A, R = arrays(E, RC)
    A2 = dict(A)
    A2['btc_5s_aligned_D0'] = A['btc_5s_aligned_D0'].copy()
    A2['btc_5s_aligned_D0'][0] = 'up'
    expect('validate refused: a state label outside the frozen bins', lambda: CTR.validate(A2), 'outside the frozen bins')
    A3 = dict(A)
    A3['end_type'] = A['end_type'].copy()
    A3['end_type'][0] = 'response_late'
    expect('validate refused: an undeclared end_type', lambda: CTR.validate(A3), 'end_type')
    A4 = dict(A)
    A4['time_left_bin_LAG'] = A['time_left_bin_LAG'].copy()
    A4['time_left_bin_LAG'][0] = ''
    expect('validate refused: a time-left bin missing while the state is ok', lambda: CTR.validate(A4), 'time_left_bin_LAG')
    check('validate: the all-classes fixture passes', run_chk(lambda: CTR.validate(A) is None)[0])


def test_e2e(tmp, FX):
    Ep, RCp = FX['planted']
    Ea, RCa = FX['all']
    E = pd.concat([Ep, Ea], ignore_index=True)
    RC = pd.concat([RCp, RCa], ignore_index=True)
    RC['key'] = [f'k{i}' for i in range(len(RC))]
    RC['episode_role'] = 'not_a_response'
    src = tmp / 'src'
    src.mkdir()
    E.to_parquet(src / 'episodes.parquet', index=False)
    RC.to_csv(src / 'reconcile.csv', index=False)
    rec = src_receipt(len(E), len(RC), CTR._sha(src / 'episodes.parquet'), CTR._sha(src / 'reconcile.csv'))
    (src / 'receipt.json').write_text(json.dumps(rec))
    out = tmp / 'e2e_out'
    lines = []
    R1 = CTR.execute(True, src=src, out=out, rehearsal=True, log=lines.append)
    heads = [ln.split(' ', 1)[0] for ln in lines]
    check('e2e dry: bindings, parquet / csv readers and the analysis print the reconciliation and readings and write nothing',
          not out.exists() and heads.count('RECON') == 1 and heads.count('CONTRAST') == 10 and heads.count('READING') == 5 and 'TIMING' in heads
          and heads[-1] == 'DRY' and R1['READ']['btc_with'] == 'adequate_direction_consistent', heads)
    lines2 = []
    R2 = CTR.execute(False, src=src, out=out, rehearsal=True, log=lines2.append)
    files = sorted(q.name for q in out.iterdir()) if out.exists() else []
    check('e2e run: exactly the seven outputs', files == sorted(CTR.OUT_FILES), files)
    rc = json.loads((out / 'receipt.json').read_text())
    check('e2e receipt: output sha256 / bytes / rows match the written files; rehearsal recorded; both inputs bound',
          all(rc['outputs'][nm]['sha256'] == CTR._sha(out / nm) and rc['outputs'][nm]['bytes'] == (out / nm).stat().st_size
              for nm in CTR.OUT_FILES if nm != 'receipt.json') and rc['rehearsal'] is True and rc['kind'] == 'run'
          and sum(1 for k, v in rc['inputs'].items() if k.endswith(('episodes.parquet', 'reconcile.csv')) and v.get('sha256') == v.get('bound_sha256')) == 2
          and [rc['outputs'][nm]['rows'] for nm in ('rd.csv', 'support.csv', 'fractions.csv', 'outcomes.csv', 'timing.csv')] == [160, 10, 984, len(E), 67],
          {k: rc['outputs'][k].get('rows') for k in rc['outputs']})
    O = pd.read_csv(out / 'outcomes.csv', dtype={'day': str}, keep_default_na=False)
    check('e2e outcomes.csv read back: one row per episode, the in-process outcomes, every declared outcome present',
          len(O) == len(E) and O['outcome'].tolist() == R2['lab'].tolist() and set(CTR.OUTCOMES) <= set(O['outcome']), len(O))
    cl = json.loads((out / 'claims.json').read_text())
    check('e2e claims.json: computed readings, reconciliation and timing summary', cl['contrasts']['btc_with']['reading'] == 'adequate_direction_consistent'
          and cl['population']['reconciliation']['class_direction_violations'] == 0 and cl['population']['n_episodes'] == len(E)
          and cl['timing']['n_responses'] == 67 and cl['population']['n_markets'] == 9, cl['population'].get('n_markets'))
    expect('e2e: a second run refuses write-once', lambda: CTR.execute(False, src=src, out=out, rehearsal=True, log=lines2.append), 'write-once')
    expect('e2e: outside a rehearsal the untracked synthetic producing receipt is refused',
           lambda: CTR.execute(True, src=src, out=tmp / 'never', log=lines.append), 'not tracked and unmodified')
    with open(src / 'reconcile.csv', 'a') as f:
        f.write('\n')
    expect('e2e: a tampered reconcile.csv is refused by its binding', lambda: CTR.execute(True, src=src, out=tmp / 'never', rehearsal=True, log=lines.append),
           'differs from the producing receipt')
    check('e2e: nothing written by the refused calls', not (tmp / 'never').exists())


def test_production_refusals(tmp):
    for m, needle in (('swap_labels', 'MT_CT_MUT'), ('bogus', 'not a declared mutant')):
        with Mut(m):
            expect(f'refuse: production with MT_CT_MUT={m} before any input is read', lambda: CTR.execute(True, src=tmp / 'absent', out=tmp / 'never2'),
                   needle)
    g = CTR._git(CTR.TOOL_FILES)
    needle = 'freeze' if not g.get('unmodified') else 'is missing'
    expect(f'refuse: the full run before any input ({needle})', lambda: CTR.execute(False, src=tmp / 'absent', out=tmp / 'run_out'), needle)
    expect('refuse: an unknown command line', lambda: CTR.main(['run', '--force']), 'usage')
    check('refuse: nothing written by a refused production call', not (tmp / 'run_out').exists() and not (tmp / 'never2').exists())
    P = CTR.paths()
    check('production paths: reads v2/chains/touch_transition, writes v2/chains/competing_transition',
          P['src'] == C.RECEIPT_V1 / 'v2' / 'chains' / 'touch_transition' and P['out'] == C.RECEIPT_V1 / 'v2' / 'chains' / 'competing_transition', P)


def main():
    t0 = time.process_time()
    check('selftest environment: MT_RUN=v2, no planted error set', C.RUN == 'v2' and CTR.MUT_ENV not in os.environ, C.RUN)
    tmp = Path(tempfile.mkdtemp(prefix='competing_transition_selftest_'))
    try:
        FX = section('fixtures', fixtures)
        if FX:
            section('all_classes', test_all_classes, FX)
            section('toys', test_toys, FX)
            section('planted', test_planted, FX)
            section('worlds', test_worlds, FX)
            section('mutants', test_mutants, FX)
            section('input_refusals', test_input_refusals, tmp, FX)
            section('e2e', test_e2e, tmp, FX)
        section('production_refusals', test_production_refusals, tmp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
        os.environ.pop(CTR.MUT_ENV, None)
    fails = [r for r in RES if not r[1]]
    for name, ok, info in RES:
        print(('PASS ' if ok else 'FAIL ') + name + ('' if ok else f'  | {info}'))
    print(json.dumps(dict(checks=len(RES), passed=len(RES) - len(fails), failed=len(fails), cpu_s=round(time.process_time() - t0, 3),
                          process_cpu_s=round(time.process_time(), 3), vmhwm_mib=round(CTR.hwm(), 1))))
    return 1 if fails else 0


if __name__ == '__main__':
    sys.exit(main())
