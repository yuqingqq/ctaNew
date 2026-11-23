"""queue_front_v2_selftest.py - synthetic proof of queue_front_v2.py (P-2026-005 MAKER-TRACE, QUEUE-FRONT v2 = posting_chain_contract.md
section 5; review 3f640ad9b findings 1-6). The REAL chain_context pipeline runs on synthetic BOOK markets built frame by frame (200 ms
frames) with v1's fixture builder queue_front_selftest.MK (imported; v1's suite is not run). The five contract controls, each with its
planted mutant (MT_QF2_MUT, caught by its named check evaluating False without raising):
(C1) the reviewer's masked-cancellation / replenishment counterexample (check_review.py's frames and records verbatim; both hidden
     histories reproduce the public frames and put each fill inside its hard window) -> front_candidate, book_candidate,
     alt_netting_possible and alt_first_addresses = both makers (history A's first filler and history B's); no certified status anywhere
     [certify_net_decrease];
(C2) a creation-window straddle (a window opening inside (upd(f-1), upd(f)] and ending after it) -> untimed, the level ambiguous_order
     with both makers; settle_inside on hand values [lower_endpoint_settle = v1's rule];
(C3) same-hash partial fills -> later_same_hash, never later_other; economics earliest_set / later_other / later_same_hash with all /
     valid rows and shares hand-computed [same_hash_behind = v1's labelling];
(C4) no-fill levels kept in every denominator (levels, candidates, shares, the fresh-context mix) [drop_no_fill];
(C5) a later-frame perturbation (frames after the candidate's hi frame and a later fill) leaves the earlier labels unchanged while a
     cancel before the fill changes the status (positive control) [whole_life_unexplained].
Also: v1 parity on every fixture (no violation outside the straddle levels; the straddle transitions tabulated), the front-win verdict
rule on v1's planted world (unidentified with alternative histories; wins_front_more_than_fills under the certify mutant: the rule can
fire), the regrade reader's refusals, the production / overwrite / preflight refusals (each asserting its message) and the pilot path end
to end through execute() with an injected loader (every writer; aggregates reconciled to the keyed rows; occupancy with the post grade,
including an order at a cell never fresh). Usage (run_light, MT_RUN=v2): queue_front_v2_selftest.py"""
import json
import os
import shutil
import sys
import tempfile
import time
import traceback
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import queue_front_v2 as Q2  # noqa: E402
import queue_front_selftest as ST  # noqa: E402  (v1's fixture builder and file writers only)

QF, C, CX = Q2.QF, Q2.C, Q2.CX
MUT = Q2.MUT_ENV
RES = []
A1, A2, A3 = Q2.TOP3
A1L, A2L, A3L = Q2.TOP3_L
OTHER, OTHER2 = ST.OTHER, ST.OTHER2
DAY, DAY2 = ST.DAY, ST.DAY2
EARLY = ('unit', 'status', 'reason', 'candidate_addr', 'candidate_group', 'timing_basis', 'candidate_book_ms', 'candidate_lo_ms', 'candidate_hi_ms',
         'earliest_set_addresses', 'n_earliest_set', 'n_unexplained', 'n_old_life', 'alt_netting_possible', 'alt_sites_life', 'alt_sites_unexplained',
         'alt_first_addresses', 'alt_first_other_address', 'creation_context')


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


def build(m):
    """v1's MK.build; one lifecycle order row per (scenario, hash) when several legs share a hash (the same-hash fixture)."""
    M, fvm = m.build()
    if M.orders is not None and len(M.orders):
        M.orders = M.orders.drop_duplicates(['scenario', 'mk_hash']).reset_index(drop=True)
    return M, fvm


def part(m, pfm=None):
    M, fvm = build(m)
    return Q2.market_pass(m.day, m.mid, M, fvm, pfm)


def lrow(P, side, tick, rt, m):
    L = P['levels']
    if not len(L):
        return None
    msk = (L['side'] == side).to_numpy() & (L['tick'].to_numpy(np.int64) == tick) & (L['t_create_ms'].to_numpy(np.int64) == m.ws + rt)
    return L[msk].iloc[0].to_dict() if int(msk.sum()) == 1 else None


def brief(r):
    if r is None:
        return None
    keys = EARLY + ('n_members', 'n_later_same_hash', 'n_later_other', 'n_unattributed', 'straddle_record', 'v1_status', 'v1_timing', 'end_type')
    return {k: r.get(k) for k in keys}


# ------------------------------------------------------------------ fixtures ------------------------------------------------------------------
def fixtures():
    F = {}
    m = ST.MK(DAY, 919001, 1)                       # C1: the reviewer's counterexample (check_review.py), verbatim
    m.setc(2000, 3400, 'bid', 45, 100).setc(3400, 4200, 'bid', 45, 70).setc(4200, 30001, 'bid', 45, 50)
    m.fill(A1, 'bid', 45, 30, 5000)
    m.fill(OTHER, 'bid', 45, 20, 5100)
    F['C1'] = m
    m = ST.MK(DAY, 919003, 3)                       # C2: OTHER's window [1.9 s, 3.9 s] opens inside (1.8 s, 2.0 s] and ends after it
    m.setc(2000, 3400, 'bid', 45, 100).setc(3400, 30001, 'bid', 45, 70)
    m.fill(A1, 'bid', 45, 30, 5000)
    m.fill(OTHER, 'bid', 45, 20, 3900)
    F['C2'] = m
    m = ST.MK(DAY, 919002, 2)                       # C3: one 100-share order fills 30 then 20 (one hash); two later fills of others
    m.setc(2000, 4600, 'bid', 45, 100).setc(4600, 8800, 'bid', 45, 70).setc(8800, 12800, 'bid', 45, 50).setc(12800, 16800, 'bid', 45, 40)
    m.setc(16800, 30001, 'bid', 45, 35)
    x = m.fill(A1, 'bid', 45, 30, 5000, v=(0.3, 0.6, 0.9))
    y = m.fill(A1, 'bid', 45, 20, 9000, v=(0.1, 0.2, 0.3))
    y['mk_hash'] = x['mk_hash']
    m.fill(OTHER, 'bid', 45, 10, 13000, v=(-0.1, -0.3, -0.5))
    m.fill(OTHER2, 'bid', 45, 5, 17000, v=(1., 1., 1.), st=('ok', 'beyond_end', 'ok'))
    F['C3'] = m
    m = ST.MK(DAY, 919004, 4)                       # C4: a cancelled level, a level never filled; an order at bid 40 (shown from the start)
    m.setc(2000, 6000, 'bid', 47, 50).setc(25000, 30001, 'bid', 48, 20).setc(14000, 30001, 'bid', 40, 90)
    m.fill(A2, 'bid', 40, 10, 15000)
    F['C4'] = m
    b, p, c = ST.MK(DAY, 919051, 13), ST.MK(DAY, 919052, 13), ST.MK(DAY, 919053, 13)   # C5: base / perturbed after the hi frame / control
    b.setc(2000, 4600, 'bid', 45, 100).setc(4600, 8800, 'bid', 45, 70).setc(8800, 20000, 'bid', 45, 50)
    b.fill(A1, 'bid', 45, 30, 5000)
    b.fill(OTHER, 'bid', 45, 20, 9000)
    p.setc(2000, 4600, 'bid', 45, 100).setc(4600, 5200, 'bid', 45, 70).setc(5200, 6000, 'bid', 45, 40).setc(6000, 7000, 'bid', 45, 0)
    p.setc(7000, 20000, 'bid', 45, 33).drop(10000, 12000)
    p.fill(A1, 'bid', 45, 30, 5000)
    p.fill(OTHER, 'bid', 45, 20, 9000)
    p.fill(OTHER2, 'bid', 45, 7, 15000)
    c.setc(2000, 2600, 'bid', 45, 100).setc(2600, 4600, 'bid', 45, 60).setc(4600, 8800, 'bid', 45, 30).setc(8800, 20000, 'bid', 45, 10)
    c.fill(A1, 'bid', 45, 30, 5000)
    c.fill(OTHER, 'bid', 45, 20, 9000)
    F['C5b'], F['C5p'], F['C5c'] = b, p, c
    return F


# ------------------------------------------------------------------ the five controls (re-run under their mutants) ------------------------------------------------------------------
def chk_C1(FX):
    m = FX['C1']
    hA = [(1900, 100), (3300, -30), (4100, -20)]                          # history A: A fills at 3.3 s, B at 4.1 s
    hB = [(1900, 100), (3300, -30), (4100, -20), (4500, -30), (4501, 30)]  # history B: a cancel at 3.3 s, B at 4.1 s, A at 4.5 s + a refill
    seen = [sum(q for ts, q in hA if ts <= t - m.ws) for t in sorted(m.frames)]
    seen_b = [sum(q for ts, q in hB if ts <= t - m.ws) for t in sorted(m.frames)]
    book = [bk.get(('bid', 45), 0.) for _, bk in sorted(m.frames.items())]
    a, b = m.legs[0], m.legs[1]
    inside = (a['lo'] <= m.ws + 3300 <= a['hi'] and a['lo'] <= m.ws + 4500 <= a['hi'] and b['lo'] <= m.ws + 4100 <= b['hi'])
    P = part(m)
    r = lrow(P, 'bid', 45, 2000, m)
    st_all = set(P['levels']['status'].tolist())
    alt = set(str(r['alt_first_addresses']).split('|')) if r is not None else set()
    ok = (seen == seen_b == book and inside and r is not None and r['status'] == 'front_candidate' and r['timing_basis'] == 'book_candidate'
          and r['candidate_addr'] == A1L and bool(r['alt_netting_possible']) and int(r['alt_sites_life']) == 8 and alt == {A1L, OTHER}
          and bool(r['alt_first_other_address']) and 'front_identified' not in st_all and r['v1_status'] == 'front_identified')
    return ok, dict(row=brief(r), public_equal=seen == seen_b == book, inside=inside, statuses=sorted(st_all))


def chk_C2(FX):
    m = FX['C2']
    P = part(m)
    r = lrow(P, 'bid', 45, 2000, m)
    cn = P['counts']
    si = Q2.settle_inside
    unit = (not si(1900, 3900, 1800, 2000) and si(1850, 1990, 1800, 2000) and not si(1800, 1990, 1800, 2000) and si(1801, 2000, 1800, 2000)
            and not si(1850, 2001, 1800, 2000) and si(1900, 3900, 1800, 2000, lower_endpoint=True))
    ok = (unit and r is not None and r['status'] == 'ambiguous_order' and r['reason'] == 'untimed_record_could_be_first'
          and r['earliest_set_addresses'] == '|'.join(sorted([A1L, OTHER])) and bool(r['straddle_record']) and r['v1_status'] == 'front_identified'
          and r['v1_timing'] == 'settlement' and cn['rec_settlement_only'] == 0 and cn['rec_book_unmatched'] == 1 and cn['rec_v1_settlement_only'] == 1
          and cn['rec_straddle_untimed'] == 1)
    return ok, dict(row=brief(r), unit=unit, rec={k: cn[k] for k in ('rec_settlement_only', 'rec_book_unmatched', 'rec_v1_settlement_only',
                                                                    'rec_straddle_untimed')})


def chk_C3(FX):
    m = FX['C3']
    P = part(m)
    r = lrow(P, 'bid', 45, 2000, m)
    Fr = P['fills']
    got = [(a_, r_, float(q_)) for a_, r_, q_ in zip(Fr['addr'], Fr['role'], Fr['q'])]
    want = [(A1L, 'earliest_set', 30.), (A1L, 'later_same_hash', 20.), (OTHER, 'later_other', 10.), (OTHER2, 'later_other', 5.)]
    EC = Q2.econ_table(Fr, [('pooled', 'all')])
    g = EC[((EC['by'] == 'candidate') & (EC['group'] == 'all') & (EC['creation_context'] == 'all')).to_numpy()].set_index('role')
    es, sh, lo_ = g.loc['earliest_set'], g.loc['later_same_hash'], g.loc['later_other']
    econ = (int(es['all_rows']) == 1 and int(es['valid_rows']) == 1 and abs(es['valid_shares'] - 30.) < 1e-9 and abs(es['v_10s_c_sh'] - 2.0) < 1e-9
            and int(sh['all_rows']) == 1 and abs(sh['v_10s_c_sh'] - 1.0) < 1e-9 and int(lo_['all_rows']) == 2 and int(lo_['valid_rows']) == 1
            and abs(lo_['all_shares'] - 15.) < 1e-9 and abs(lo_['valid_shares'] - 10.) < 1e-9 and abs(lo_['valid_row_share'] - 0.5) < 1e-12
            and abs(lo_['valid_share_share'] - 10. / 15.) < 1e-12 and abs(lo_['v_10s_c_sh'] + 3.0) < 1e-9 and lo_['excluded'] == 'st_10s:1')
    ok = (r is not None and r['status'] == 'front_candidate' and r['candidate_addr'] == A1L and got == want and int(r['n_later_same_hash']) == 1
          and int(r['n_later_other']) == 2 and econ)
    return ok, dict(row=brief(r), fills=got, econ=g[['all_rows', 'valid_rows', 'all_shares', 'valid_shares', 'v_10s_c_sh', 'excluded']].to_dict('index'))


def chk_C4(FX):
    m = FX['C4']
    P = part(m)
    a, b = lrow(P, 'bid', 47, 2000, m), lrow(P, 'bid', 48, 25000, m)
    tables, claims, _ = Q2.analyse('selftest', (DAY,), [(m.day, m.mid)], [P], Counter())
    CD, SH, MC = tables['candidates.csv'], tables['shares.csv'], tables['mechanism.csv']
    pool = SH[((SH['scope'] == 'pooled') & (SH['group'] == 'top3')).to_numpy()].iloc[0]
    nf = MC[((MC['table'] == 'fresh_creation_context') & (MC['scope'] == 'pooled') & MC['status'].isin(['no_fill', 'all_fresh'])).to_numpy()]
    ok = (a is not None and b is not None and a['status'] == 'no_fill' and a['reason'] == 'returned_to_zero' and b['status'] == 'no_fill'
          and b['reason'] == 'window_end_without_fill' and len(CD) == 2 and set(CD['status']) == {'no_fill'} and set(CD['candidate_addr']) == {''}
          and int(pool['n_fresh']) == 2 and int(pool['n_no_fill']) == 2 and int(pool['bound_denominator']) == 0
          and set(nf['denom'].tolist()) == {2} and int(nf['n'].sum()) == 4)
    return ok, dict(a=brief(a), b=brief(b), cand=len(CD), shares={k: pool[k] for k in ('n_fresh', 'n_no_fill', 'bound_denominator')},
                    mix=nf[['status', 'category', 'n', 'denom']].values.tolist())


def chk_C5(FX):
    b, p, c = FX['C5b'], FX['C5p'], FX['C5c']
    rb, rp, rc = lrow(part(b), 'bid', 45, 2000, b), lrow(part(p), 'bid', 45, 2000, p), lrow(part(c), 'bid', 45, 2000, c)
    same = rb is not None and rp is not None and all(rb[k] == rp[k] for k in EARLY) and rb['status'] == 'front_candidate'
    real = rb is not None and rp is not None and (rb['n_members'] != rp['n_members'] and rb['end_ms'] != rp['end_ms'])
    moved = rb is not None and rc is not None and rc['status'] == 'unexplained_decrease' != rb['status']
    return bool(same and real and moved), dict(base=brief(rb), perturbed=brief(rp), control=brief(rc), differ=[k for k in EARLY if rb and rp and rb[k] != rp[k]])


SUITE = (('C1', chk_C1, "(C1) the reviewer's masked-cancellation / replenishment counterexample: identical public frames, both histories' first "
                        'fillers remain (alt_first_addresses), alt_netting_possible, front_candidate by book_candidate; never certified'),
         ('C2', chk_C2, '(C2) a creation-window straddle -> untimed (not settlement_only), the level ambiguous_order with both makers; '
                        'settle_inside on hand values'),
         ('C3', chk_C3, '(C3) same-hash partial fills -> later_same_hash (not later_other); economics earliest_set / later_other / later_same_hash '
                        'with all / valid rows and shares as hand-computed'),
         ('C4', chk_C4, '(C4) no-fill levels kept in levels, candidates, shares (n_fresh, n_no_fill) and the fresh-context mix'),
         ('C5', chk_C5, '(C5) a later-frame perturbation leaves the earlier labels unchanged; a cancel before the fill changes the status'))
MUT_OF = dict(certify_net_decrease='C1', lower_endpoint_settle='C2', same_hash_behind='C3', drop_no_fill='C4', whole_life_unexplained='C5')


def test_mutants(FX):
    fns = {nm: fn for nm, fn, _ in SUITE}
    check('mutants: one control per declared mutant', sorted(MUT_OF) == sorted(Q2.MUTANTS), sorted(MUT_OF))
    for mu, nm in MUT_OF.items():
        os.environ[MUT] = mu
        try:
            try:
                ok, info, raised = (*fns[nm](FX), False)
            except Exception as e:  # noqa: BLE001
                ok, info, raised = False, f'{type(e).__name__}: {e}', True
        finally:
            os.environ.pop(MUT, None)
        check(f'mutant {mu}: caught by {nm} evaluating False without raising', not ok and not raised, info)
    check('mutants: MT_QF2_MUT unset after the suite', MUT not in os.environ)


def test_parity(FX):
    rows = {k: part(FX[k])['parity'] for k in FX}
    nv = {k: r['n_violations'] for k, r in rows.items()}
    check('parity: every fixture level without a straddle record reproduces v1 exactly (status, reason, front record, set, counts, fills)',
          sum(nv.values()) == 0 and sum(r['n_levels'] for r in rows.values()) >= 9, dict(violations=nv, by_key={k: r['by_key'] for k, r in rows.items()
                                                                                                                 if r['n_violations']}))
    check('parity: the straddle levels are tabulated (C2 v1 front_identified -> v2 ambiguous_order; C5p v1 front_identified -> v2 unknown)',
          rows['C2']['affected_transitions'] == {'front_identified->ambiguous_order': 1}
          and rows['C5p']['affected_transitions'] == {'front_identified->unknown': 1}
          and all(r['n_affected'] == 0 for k, r in rows.items() if k not in ('C2', 'C5p')), {k: r['affected_transitions'] for k, r in rows.items()})


def test_readings():
    mks = ST.world('planted')
    markets = [(m.day, m.mid) for m in mks]
    parts = [part(m) for m in mks]
    _, c1, _ = Q2.analyse('selftest', (DAY, DAY2), markets, parts, Counter())
    os.environ[MUT] = 'certify_net_decrease'
    try:
        parts2 = [part(m) for m in mks]
    finally:
        os.environ.pop(MUT, None)
    _, c2, _ = Q2.analyse('selftest', (DAY, DAY2), markets, parts2, Counter())
    V = {(r['scope'], r['reading']): r for r in c1['readings']}
    V2 = {(r['scope'], r['reading']): r for r in c2['readings']}
    w1, w2 = V[('rank1', 'front_win_scenario_bound')], V2[('rank1', 'front_win_scenario_bound')]
    others = ('scenario_lower_gt_fill_share', 'p_hi_le_0_05', 'supported_markets_ge_5', 'market_share_ge_2_3')
    check('readings (planted world): every other predicate holds, no_alt_netting is False -> unidentified (alternative histories block it)',
          w1['verdict'] == 'unidentified' and w1['predicates']['no_alt_netting'] is False and all(w1['predicates'][k] for k in others)
          and abs(w1['values']['p_hi'] - 1 / 501) < 1e-12, dict(verdict=w1['verdict'], predicates=w1['predicates']))
    check('readings (positive control): the same world under certify_net_decrease -> wins_front_more_than_fills (the rule can fire)',
          w2['verdict'] == 'wins_front_more_than_fills' and w2['predicates']['no_alt_netting'] is True, dict(verdict=w2['verdict'], predicates=w2['predicates']))
    lc = V[('rank1', 'largest_creation_context')]
    check('readings: largest creation context of rank 1 front candidates = improve (54 of 60), computed with counts and denominators',
          lc['verdict'] == 'improve' and lc['values']['counts']['improve'] == 54 and lc['values']['n_front_candidates'] == 60
          and lc['beside']['largest_is_majority'] is True, lc)
    pop = c1['population']
    check('readings: 66 front candidates (all book_candidate, every one alt_netting_possible), claims carry the withdrawn inferences',
          pop['fresh_by_status']['front_candidate'] == 66 and pop['front_candidate_by_basis'] == {'book_candidate': 66, 'settlement': 0}
          and pop['alt']['front_candidate_alt_netting_possible'] == 66 and set(c1['withdrawn']) == {'front_pays', 'faster_than_others', 'certified_front'},
          dict(st=pop['fresh_by_status'], basis=pop['front_candidate_by_basis'], alt=pop['alt']))


# ------------------------------------------------------------------ inputs, refusals, the pilot path end to end ------------------------------------------------------------------
def write_rg(rg, D, po_sha, grades):
    rg.mkdir(parents=True, exist_ok=True)
    G = D[['day', 'mid', 'rank', 'addr', 'mk_hash', 'y_side', 'tick', 'post_lo_union_ms', 'post_hi_ms']].copy()
    G['grade'] = list(grades)
    G.to_csv(rg / 'post_regrade_orders.csv', index=False)
    rec = dict(outputs={'post_regrade_orders.csv': dict(sha256=QF._sha(rg / 'post_regrade_orders.csv'))}, inputs=dict(post_orders=dict(sha256=po_sha)))
    (rg / 'receipt_post.json').write_text(json.dumps(rec))


def test_regrade(tmp):
    D = pd.DataFrame([dict(day=DAY, mid=1, rank=1, addr=A1, mk_hash='0xh1', y_side='bid', tick=45, n_fills=1, cum_shares=10., post_lo_union_ms=100,
                           post_hi_ms=200, window_status='bounded', queue_class='unidentified'),
                      dict(day=DAY, mid=1, rank=2, addr=A2, mk_hash='0xh2', y_side='ask', tick=55, n_fills=2, cum_shares=5., post_lo_union_ms=-1,
                           post_hi_ms=300, window_status='window_unbounded', queue_class='unidentified')], columns=list(ST.PFC))
    pf = tmp / 'rg_pf'
    ST.write_pf(pf, D, 'f' * 64)
    po_sha = QF._sha(pf / 'orders.csv')
    mk = [(DAY, 1)]
    PF, _ = QF.read_post_fill(pf, mk, {}, 'f' * 64)
    rg = tmp / 'rg_ok'
    write_rg(rg, D, po_sha, ['G2_bound_constrained', 'G0_unbounded_or_invalid'])
    RG, info = Q2.read_regrade(rg, mk, {}, po_sha)
    J = Q2.join_grades(PF, RG)
    check('regrade: bound to its receipt and to the post_fill orders.csv read; every order carries its grade',
          list(J['post_grade']) == ['G2_bound_constrained', 'G0_unbounded_or_invalid'] and info['rows_run'] == 2, list(J['post_grade']))
    bad = tmp / 'rg_tamper'
    write_rg(bad, D, po_sha, ['G2_bound_constrained', 'G0_unbounded_or_invalid'])
    (bad / 'post_regrade_orders.csv').write_text((bad / 'post_regrade_orders.csv').read_text().replace('G2_bound_constrained', 'G4_scenario_only'))
    expect('regrade: a csv unlike its receipt -> refused', lambda: Q2.read_regrade(bad, mk, {}, po_sha), "differs from its receipt's sha256")
    oth = tmp / 'rg_other_orders'
    write_rg(oth, D, 'e' * 64, ['G2_bound_constrained', 'G0_unbounded_or_invalid'])
    expect('regrade: a receipt naming another post_fill orders.csv -> refused', lambda: Q2.read_regrade(oth, mk, {}, po_sha), 'mismatched artifacts')
    one = tmp / 'rg_missing'
    write_rg(one, D.iloc[:1], po_sha, ['G2_bound_constrained'])
    RG1, _ = Q2.read_regrade(one, mk, {}, po_sha)
    expect('regrade: a post_fill order without a regrade row -> refused', lambda: Q2.join_grades(PF, RG1), 'without a regrade row')
    shf = tmp / 'rg_shifted'
    write_rg(shf, D.assign(post_lo_union_ms=[150, -1]), po_sha, ['G2_bound_constrained', 'G0_unbounded_or_invalid'])
    RG2, _ = Q2.read_regrade(shf, mk, {}, po_sha)
    expect('regrade: a regrade window unlike post_fill -> refused', lambda: Q2.join_grades(PF, RG2), 'post_lo_union_ms differs')


def test_refusals(tmp):
    d = tmp / 'exists'
    d.mkdir()
    expect('refuse: write-once on an existing output directory (overwrite, v1 write_out)', lambda: QF.write_out(d, {}, {}, lambda o: {}), 'write-once')
    expect('refuse: execute into an existing output directory before any input is read', lambda: Q2.execute('pilot', (Q2.PILOT_DAY,), d), 'write-once')
    for mu, needle in (('certify_net_decrease', 'MT_QF2_MUT'), ('bogus', 'not a declared mutant')):
        os.environ[MUT] = mu
        try:
            expect(f'refuse: production with MT_QF2_MUT={mu} before any input is read', lambda: Q2.execute('pilot', (Q2.PILOT_DAY,), tmp / 'never'), needle)
        finally:
            os.environ.pop(MUT, None)
    os.environ['MT_QF_MUT'] = 'drop_no_fill'
    try:
        expect("refuse: v1's MT_QF_MUT set", lambda: Q2.execute('pilot', (Q2.PILOT_DAY,), tmp / 'never'), 'MT_QF_MUT')
    finally:
        os.environ.pop('MT_QF_MUT', None)
    expect('refuse: a pilot day outside the measurement days', lambda: Q2.execute('pilot', (C.ID_DAY,), tmp / 'never'), 'measurement days')
    check('refuse: nothing written by a refused call', not (tmp / 'never').exists())
    g = QF._git([f for f in Q2.TOOL_FILES if Path(f).exists()])
    needle = 'freeze' if not g.get('unmodified') else 'no recorded pilot projection'
    expect(f'refuse: the full run before any input ({needle})',
           lambda: Q2.execute('run', C.MEASURE_DAYS, tmp / 'run_out', pilot_receipt=tmp / 'absent.json'), needle)
    expect('refuse: the full run without a recorded pilot projection', lambda: Q2.preflight_projection(tmp / 'absent.json', {}), 'no recorded pilot projection')
    for nm, J, needle in (('a projection > 120 CPU s', dict(kind='pilot', version=Q2.VERSION, projection=dict(full_run_cpu_s=121.), parity=dict(n_violations=0)),
                           '> 120 s (refused before launch)'),
                          ('a pilot with parity violations', dict(kind='pilot', version=Q2.VERSION, projection=dict(full_run_cpu_s=20.),
                                                                  parity=dict(n_violations=3)), 'v1-parity violations'),
                          ("v1's pilot receipt", dict(kind='pilot', version=QF.VERSION, projection=dict(full_run_cpu_s=20.)), 'not a queue_front_v2 pilot')):
        pth = tmp / f'pre_{abs(hash(nm)) % 10 ** 8}.json'
        pth.write_text(json.dumps(J))
        expect(f'refuse: preflight on {nm}', lambda p=pth: Q2.preflight_projection(p, {}), needle)
    ok_p = tmp / 'pre_ok.json'
    ok_p.write_text(json.dumps(dict(kind='pilot', version=Q2.VERSION, projection=dict(full_run_cpu_s=60.), parity=dict(n_violations=0))))
    h = {}
    r = Q2.preflight_projection(ok_p, h)
    check('preflight: a 60 s projection with 0 parity violations passes; the pilot receipt is hashed as an input', r['full_run_cpu_s'] == 60. and len(h) == 1)


def test_e2e(tmp, FX):
    mks = [FX[k] for k in ('C1', 'C2', 'C3', 'C4', 'C5b', 'C5p', 'C5c')]
    root = tmp / 'root'
    (root / 'econ').mkdir(parents=True)
    (root / 'econ' / 'identities.json').write_text(json.dumps(dict(ex_ante=dict(day=C.ID_DAY, top10_shares=[dict(addr=a, rank=i + 1)
                                                                                                            for i, a in enumerate(Q2.TOP3)]))))
    rg0 = [dict(day='20261023', mid=800001, sid='X1', mk_idx=0, role='maker', addr=OTHER, hash='hx', y_side='bid', s=1, q=1., p=.5, self_trade=False,
                settle_ms=1, status='ok', st_res='ok', st_1s='ok', v_1s=0., st_10s='ok', v_10s=0., st_60s='ok', v_60s=0.)]
    fvp = root / 'econ' / 'fills_valued.parquet'
    ST.write_fv(fvp, [rg0, [r for m in mks for r in m.fv_rows(taker=True)]])
    fsha = QF._sha(fvp)
    unb = FX['C2'].legs[0]['mk_hash']
    D = ST.pf_orders(mks, {unb: dict(post_lo_union_ms=-1, window_status='window_unbounded')})
    D = D.drop_duplicates(['day', 'mid', 'mk_hash']).reset_index(drop=True)
    pf = tmp / 'pf'
    ST.write_pf(pf, D, fsha)
    grades = ['G0_unbounded_or_invalid' if h == unb else ('G2_bound_constrained' if i % 2 == 0 else 'G4_scenario_only') for i, h in enumerate(D['mk_hash'])]
    rg = tmp / 'rg'
    write_rg(rg, D, QF._sha(pf / 'orders.csv'), grades)
    built = {(m.day, m.mid): build(m)[0] for m in mks}
    book = {DAY: sorted(m.mid for m in mks)}
    out = tmp / 'e2e_pilot'
    run = lambda o: Q2.execute('pilot', (DAY,), o, inp=root, loader=lambda d, mid, r, hc: built[(d, mid)], book=book, pf_dir=pf, rg_dir=rg,  # noqa: E731
                               fv_sha=fsha, log=lambda s: None)
    rec = run(out)
    names = sorted(p.name for p in out.iterdir())
    check('e2e: execute (pilot path, injected loader) writes exactly the declared outputs', names == sorted(Q2.OUT_FILES), names)
    Rj = json.loads((out / 'receipt.json').read_text())
    pj = Rj.get('projection') or {}
    check('e2e: receipt.json hashes every other output, the sources and inputs (regrade + its receipt, post_fill, fills_valued); a finite '
          'projection; parity recorded; no mutant',
          all(Rj['outputs'][f]['sha256'] == QF._sha(out / f) for f in Q2.OUT_FILES if f != 'receipt.json') and Rj['mutant'] == ''
          and Rj['inputs'].get('econ/fills_valued.parquet') == fsha and QF._rel(rg / 'post_regrade_orders.csv') in Rj['inputs']
          and QF._rel(rg / 'receipt_post.json') in Rj['inputs'] and QF._rel(pf / 'orders.csv') in Rj['inputs'] and len(Rj['sources']) >= 10
          and isinstance(pj.get('full_run_cpu_s'), float) and Rj['parity']['n_violations'] == 0 and rec['kind'] == 'pilot',
          dict(proj=pj, n_src=len(Rj['sources']), inputs=sorted(Rj['inputs'])[:8], parity=Rj.get('parity')))
    exp = [part(m) for m in mks]
    LV = pd.read_csv(out / 'levels.csv', dtype={'day': str})
    CD = pd.read_csv(out / 'candidates.csv', dtype={'day': str})
    FLc = pd.read_csv(out / 'fills.csv', dtype={'day': str})
    n_e1 = int((LV['unit'] == 'E1').sum())
    fc = (CD['status'] == 'front_candidate').to_numpy()
    ca = CD['candidate_addr'].fillna('').astype(str).to_numpy()
    check('e2e: levels.csv = every market part; candidates.csv = one keyed row per E1 level (day, mid, level_id unique); a candidate address '
          'exactly on front_candidate rows, each with alt_netting_possible and a timing basis',
          len(LV) == sum(len(p['levels']) for p in exp) and len(CD) == n_e1 and not CD.duplicated(['day', 'mid', 'level_id']).any()
          and bool((ca[fc] != '').all()) and bool((ca[~fc] == '').all()) and bool(CD.loc[fc, 'alt_netting_possible'].astype(bool).all())
          and set(CD.loc[fc, 'timing_basis']) <= set(Q2.BASES), dict(levels=len(LV), cand=len(CD), e1=n_e1, fc=int(fc.sum())))
    EC = pd.read_csv(out / 'economics.csv', dtype={'day': str})
    g = EC[((EC['scope'] == 'pooled') & (EC['by'] == 'candidate') & (EC['group'] == 'all') & (EC['creation_context'] == 'all')).to_numpy()].set_index('role')
    rec_ok = all(int(g.loc[r_, 'all_rows']) == int((FLc['role'] == r_).sum())
                 and abs(float(g.loc[r_, 'valid_shares']) - float(FLc.loc[(FLc['role'] == r_).to_numpy() & FLc['valid'].astype(bool).to_numpy(), 'q_valid'].sum())) < 1e-9
                 for r_ in Q2.ROLES)
    check('e2e: economics reconcile to fills.csv by role (all rows, valid shares); later_same_hash present (C3)',
          rec_ok and int((FLc['role'] == 'later_same_hash').sum()) >= 1, g[['all_rows', 'valid_rows', 'valid_shares']].to_dict('index'))
    OC = pd.read_csv(out / 'occupancy.csv', dtype={'day': str})
    occ = Counter(OC['occupancy'].tolist())
    gm = dict(zip(D['mk_hash'], grades))
    c4 = OC[(OC['mid'] == FX['C4'].mid).to_numpy()]
    c1 = OC[(OC['mid'] == FX['C1'].mid).to_numpy()]
    Cl = json.loads((out / 'claims.json').read_text())
    osum = {r['occupancy']: (r['n'], r['denom']) for r in Cl['occupancy'] if r['scope'] == 'pooled' and r['group'] == 'top3'}
    check('e2e: occupancy - one row per top-3 order with its post grade; the order at a cell never fresh is cell_occupied_at_window_lo (frame '
          'time and age carried), empty endpoints and the unbounded window counted; the claims summary = the keyed rows',
          len(OC) == len(D) == 7 and occ == Counter(cell_empty_at_window_lo=5, cell_occupied_at_window_lo=1, window_unbounded=1)
          and all(gm[h] == gr for h, gr in zip(OC['mk_hash'], OC['post_grade'])) and len(c4) == 1
          and c4['occupancy'].iloc[0] == 'cell_occupied_at_window_lo' and int(c4['frame_ms_at_window_lo'].iloc[0]) == int(c4['window_lo_ms'].iloc[0])
          and int(c4['frame_age_ms'].iloc[0]) == 0 and float(c4['size_at_window_lo'].iloc[0]) == 100. and bool(c1['is_front_candidate_hash'].iloc[0])
          and not bool(c4['is_front_candidate_hash'].iloc[0]) and osum.get('cell_occupied_at_window_lo') == (1, 7) and osum.get('cell_empty_at_window_lo') == (5, 7)
          and sum(n for n, _ in osum.values()) == 7, dict(occ=dict(occ), summary=osum))
    pop = Cl['population']
    check('e2e: claims - reconciliation all true, parity 0 violations with the two straddle levels, straddle counts, the no-creation legs '
          'outside every fresh level, computed verdicts, the withdrawn inferences',
          all(Cl['reconciliation'].values()) and Cl['parity']['n_violations'] == 0 and Cl['parity']['n_affected'] == 2
          and pop['straddle']['levels_with_straddle_record'] == 2 and pop['straddle']['v1_settlement_records_untimed_in_v2'] == 2
          and pop['record_timing']['settlement_only'] == 0 and pop['legs']['grid'] == 16 and pop['legs']['member_of_a_fresh_level'] == 15
          and all(r['verdict'] in ('insufficient_support', 'unidentified', 'wins_front_more_than_fills') or r['verdict'] in Q2.CONTEXTS
                  for r in Cl['readings']) and set(Cl['withdrawn']) == {'front_pays', 'faster_than_others', 'certified_front'},
          dict(rec=Cl['reconciliation'], parity=Cl['parity'], straddle=pop['straddle'], legs=pop['legs'], verdicts=[r['verdict'] for r in Cl['readings']]))
    SP = pd.read_csv(out / 'speed.csv', dtype={'day': str})
    check('e2e: speed.csv carries the endpoint proxy with its lower / upper interval columns', all(f'{c_}_p50' in SP.columns for c_ in
                                                                                               ('speed_ms', 'speed_lo_ms', 'speed_hi_ms')), list(SP.columns)[:12])
    expect('e2e: a second write into the written directory is refused (before any input is read)', lambda: run(out), 'write-once')


def main():
    t0 = time.process_time()
    check('selftest environment: MT_RUN=v2, no MT_QF2_MUT / MT_QF_MUT', C.RUN == 'v2' and MUT not in os.environ and 'MT_QF_MUT' not in os.environ, C.RUN)
    tmp = Path(tempfile.mkdtemp(prefix='queue_front_v2_selftest_'))
    try:
        FX = section('fixtures', fixtures)
        if FX:
            for nm, fn, title in SUITE:
                r = section(nm, fn, FX)
                if r:
                    check(title, r[0], r[1])
            section('parity', test_parity, FX)
            section('mutants', test_mutants, FX)
        section('readings', test_readings)
        section('regrade', test_regrade, tmp)
        section('refusals', test_refusals, tmp)
        if FX:
            section('e2e', test_e2e, tmp, FX)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
        os.environ.pop(MUT, None)
        os.environ.pop('MT_QF_MUT', None)
    fails = [r for r in RES if not r[1]]
    for name, ok, info in RES:
        print(('PASS ' if ok else 'FAIL ') + name + ('' if ok else f'  | {info}'))
    print(json.dumps(dict(checks=len(RES), failed=len(fails), cpu_s=round(time.process_time() - t0, 3), process_cpu_s=round(time.process_time(), 3),
                          vmhwm_mib=round(QF.hwm(), 1))))
    return 1 if fails else 0


if __name__ == '__main__':
    sys.exit(main())
