"""chain_correct_post_selftest.py - synthetic proof of chain_correct_post.py (P-2026-005 MAKER-TRACE logic-chain correction v1; contract
chain_correction_contract.md sections 1, 4 and the section-5 falsifiers 'grade' and 'observed vs inferred'; 2026-11-23). In-process,
synthetic inputs only (a temporary directory): no real data path is opened. Sections:
  precedence   the contract-1 grade table, first match, on rows matching several rules (+ mutant g2_before_g1 must be caught);
  grade        falsifier: a union-short but assumption_sensitive row never grades G2 (unit + world; + mutant grade_by_width caught);
  logic        the corrected POST logic-row grade / queue status for every grade;
  census       falsifier 'observed vs inferred': a tied link is never counted as an identified repost (+ mutant tied_as_identified caught);
               the census writer refuses an ungraded action row and an identified link (+ mutant ungraded_ok caught);
  paths        selection = the first two episodes per rank in cases.csv order of FIRST APPEARANCE (not sorted ids);
  end to end   every writer on a synthetic world (seven files, receipt hashes = bytes, claims parse, a sibling file untouched); --dry writes
               nothing; refusals with their reasons asserted (overwrite, mutant in a run, non-measurement day, missing scenario row, broken
               union fact, ungraded action type).
Usage (run_light, MT_RUN=v2): chain_correct_post_selftest.py"""
import hashlib
import json
import os
import shutil
import sys
import tempfile
import traceback
from collections import Counter
from pathlib import Path

sys.dont_write_bytecode = True

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pyarrow as pa  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

HERE = Path(__file__).resolve().parent
RESEARCH = HERE.parents[2] / 'research'
for _p in (str(RESEARCH), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import chain_correct_post as CP  # noqa: E402

RES = []
NAN = float('nan')
G0, G1, G2, G3, G4 = CP.GRADES


def check(name, ok, info=''):
    RES.append((name, bool(ok), str(info)[:400]))


def refuses(fn, needle):
    try:
        fn()
    except Exception as e:  # noqa: BLE001
        return needle in str(e), f'{type(e).__name__}: {e}'
    return False, 'did not raise'


def expect(name, fn, needle):
    ok, info = refuses(fn, needle)
    check(name, ok, info)


def section(name, fn, *a):
    try:
        fn(*a)
    except Exception as e:  # noqa: BLE001
        check(f'{name}: section crashed', False, f'{type(e).__name__}: {e} | {traceback.format_exc()[-900:]}')


class Mut:
    """MT_CC_MUT set inside the block only (the planted errors of chain_correct_post)."""
    def __init__(self, m):
        self.m = m

    def __enter__(self):
        self.old = os.environ.get('MT_CC_MUT')
        os.environ['MT_CC_MUT'] = self.m

    def __exit__(self, *a):
        if self.old is None:
            os.environ.pop('MT_CC_MUT', None)
        else:
            os.environ['MT_CC_MUT'] = self.old


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


# ------------------------------------------------------------------ the synthetic world ------------------------------------------------------------------
A1, A2, A3 = CP.OE.TOP3
ADDR = {1: A1, 2: A2, 3: A3}
DAY_A, DAY_B = '20261024', '20261025'
MA, MB = 1000001, 1000002
WS = {MA: 1_800_000_000_000, MB: 1_800_100_000_000}
OTHER = '0x' + 'ab' * 20


def H(i):
    return '0x' + f'{i:064x}'


E2, E5, E9 = (f'{DAY_A}_{MA}_r1_e002', f'{DAY_A}_{MA}_r1_e005', f'{DAY_A}_{MA}_r1_e009')
E11, E12, E20 = (f'{DAY_A}_{MA}_r2_e011', f'{DAY_A}_{MA}_r2_e012', f'{DAY_B}_{MB}_r3_e020')
T1, T3 = '01a00000-0000-7000-8000-000000000001', '01a00000-0000-7000-8000-000000000003'
CONS, SENS, UNRES = 'constrained', 'assumption_sensitive', 'unresolved'
# d: ms before post_hi per scenario (S_FIFO, S_FIFO_LO, S_NOPRIO), -1 = unbounded; g: the expected contract-1 grade
SPEC = [
    dict(h=H(1), rank=1, day=DAY_A, mid=MA, side='bid', tick=40, t=10_000, d=(500, 500, 500), life=(CONS,) * 3, st='ok', nk=('refill', 'tied', H(12)), g=G2),
    dict(h=H(2), rank=1, day=DAY_A, mid=MA, side='bid', tick=41, t=12_000, d=(5000, 4000, 3000), life=(SENS,) * 3, st='ok',
         nk=('reprice', 'sequential_identified', H(3)), g=G4),
    dict(h=H(3), rank=1, day=DAY_A, mid=MA, side='ask', tick=60, t=40_000, d=(20_000,) * 3, life=(CONS,) * 3, st='ok', g=G2),
    dict(h=H(4), rank=2, day=DAY_A, mid=MA, side='bid', tick=42, t=15_000, d=(-1, 3000, -1), life=(SENS,) * 3,
         st='gap_in_window+incomplete_gap+window_unbounded', cf='capacity_bounds_void+clock_capacity_conflict', nk=('reprice', 'tied', H(5)), g=G0),
    dict(h=H(5), rank=2, day=DAY_A, mid=MA, side='bid', tick=43, t=20_000, d=(800,) * 3, life=(CONS,) * 3, st='post_after_first_fill',
         cf='post_after_first_fill', nk=('refill', 'sequential_identified', H(6)), g=G1),
    dict(h=H(6), rank=2, day=DAY_A, mid=MA, side='ask', tick=61, t=25_000, d=(3000,) * 3, life=(CONS,) * 3, st='ahead_hi_negative', cf='ahead_hi_negative', g=G3),
    dict(h=H(7), rank=3, day=DAY_B, mid=MB, side='bid', tick=44, t=5_000, d=(1500,) * 3, life=(UNRES,) * 3, st='ok', g=G4),
    dict(h=H(8), rank=3, day=DAY_B, mid=MB, side='bid', tick=45, t=6_000, d=(900,) * 3, life=(CONS, CONS, SENS), st='ok', g=G4),
    dict(h=H(9), rank=3, day=DAY_B, mid=MB, side='ask', tick=62, t=7_000, d=(700,) * 3, life=(CONS,) * 3, st='unplaceable_leg', cf='unplaceable_leg', inv=True, g=G0),
    dict(h=H(10), rank=1, day=DAY_B, mid=MB, side='bid', tick=46, t=8_000, d=(600,) * 3, life=(CONS,) * 3, st='ok', clock=True, nk=('reprice', 'tied', H(1)), g=G1),
    dict(h=H(11), rank=2, day=DAY_B, mid=MB, side='bid', tick=47, t=9_000, d=(2000,) * 3, life=(CONS,) * 3, st='ok', cf='capacity_bounds_void', g=G1),
    dict(h=H(12), rank=1, day=DAY_A, mid=MA, side='bid', tick=40, t=30_000, d=(9000,) * 3, life=(CONS,) * 3, st='ok', g=G2)]
EXPECT_G = {s['h']: s['g'] for s in SPEC}
CASE_ORDERS = (H(1), H(2), H(3), H(4), H(5), H(6))                      # the orders with saved POST logic rows


def derive(s):
    """the packet fields of a SPEC row by chain_post_fill's rules (independent of the tool)."""
    ph = WS[s['mid']] + s['t']
    plo = [ph - d if d >= 0 else -1 for d in s['d']]
    lo = -1 if min(plo) < 0 else min(plo)
    w = ph - lo if lo >= 0 else -1
    if s.get('inv'):
        pc, why = 'unidentified', 'invalid_order'
    elif lo < 0:
        pc, why = 'unidentified', 'window_unbounded'
    elif s.get('clock'):
        pc, why = 'unidentified', 'first_fill_clock'
    elif w <= 1000:
        pc, why = 'late_post', 'width_le_late_ms'
    elif w <= 10_000:
        pc, why = 'short_rest', 'width_le_early_ms'
    else:
        pc, why = 'early_or_late_unidentified', 'width_gt_early_ms'
    rb = 'unbounded' if w < 0 else ('le_1s' if w <= 1000 else ('le_10s' if w <= 10_000 else 'gt_10s'))
    est = 'window_unbounded' if w < 0 else ('late_post' if w <= 1000 else ('intermediate' if w <= 10_000 else 'early_possible'))
    return dict(post_hi=ph, plo=plo, lo=lo, w=w, pc=pc, why=why, rb=rb, est=est)


def levels_independent(spec):
    """chain_take's level rule, independently: per (day, market, address) by (first fill, hash); the first at (side, tick) is new."""
    out, groups = {}, {}
    for s in spec:
        groups.setdefault((s['day'], s['mid'], s['rank']), []).append(s)
    for g in groups.values():
        seen = set()
        for s in sorted(g, key=lambda x: (WS[x['mid']] + x['t'], x['h'])):
            out[s['h']] = 'post_new_level' if (s['side'], s['tick']) not in seen else 'post_same_level'
            seen.add((s['side'], s['tick']))
    return out


def build_world(root, p_mod=None, life_mod=None, a_mod=None):
    """write every input of chain_correct_post into root/v2 and root/life; *_mod: callbacks on the frames before writing."""
    rr, life = Path(root) / 'v2', Path(root) / 'life'
    for d in ('chains/post_fill', 'chains/take', 'chains/pull', 'order_estimator'):
        (rr / d).mkdir(parents=True, exist_ok=True)
    life.mkdir(parents=True, exist_ok=True)
    P, CO, LF = [], [], []
    for s in SPEC:
        x = derive(s)
        nk = s.get('nk', ('none', 'none', ''))
        P.append(dict(day=s['day'], mid=s['mid'], rank=s['rank'], addr=ADDR[s['rank']], mk_hash=s['h'], y_side=s['side'], tick=s['tick'], n_fills=2,
                      cum_shares=10., first_hi_ms=x['post_hi'], last_hi_ms=x['post_hi'] + 1000, ws_ms=WS[s['mid']], we_ms=WS[s['mid']] + 300_000,
                      post_lo_union_ms=x['lo'], post_lo_fifo_ms=x['plo'][0], post_hi_ms=x['post_hi'], window_width_ms=x['w'], rest_bin=x['rb'],
                      post_class=x['pc'], post_class_reason=x['why'], queue_class='unidentified', n_later_fills=1, size_status='tied',
                      union_histories='EX|PL', next_kind=nk[0], next_link_status=nk[1], n_children=0 if nk[0] == 'none' else 1,
                      next_hash=nk[2], next_gap_lo_ms=NAN if nk[0] == 'none' else -500., next_gap_hi_ms=NAN if nk[0] == 'none' else 2500.,
                      status=s['st']))
        cf = s.get('cf', 'ok')
        CO.append(dict(day=s['day'], mid=s['mid'], rank=s['rank'], hash=s['h'], n_fills=2, cum=10., size_status='tied', union='EX|PL',
                       conflict=bool(CP._flags(cf) & CP.OE.X_FLAGS), clock_bad=bool(s.get('clock', False)), post_w=x['w'], post_class=x['est'],
                       order_flags=cf))
        for j, sc in enumerate(CP.SCEN):
            LF.append(dict(day=s['day'], mid=s['mid'], scenario=sc, mk_hash=s['h'], mk_signer=ADDR[s['rank']].lower(), post_class=s['life'][j],
                           post_lo_ms=x['plo'][j], post_hi_ms=x['post_hi']))
    for i, (d, m) in enumerate(((DAY_A, MA), (DAY_B, MB))):                       # another maker's orders: never read into the grade
        for sc in CP.SCEN:
            LF.append(dict(day=d, mid=m, scenario=sc, mk_hash=H(900 + i), mk_signer=OTHER, post_class=CONS, post_lo_ms=1, post_hi_ms=2))
    P, CO, LF = pd.DataFrame(P), pd.DataFrame(CO), pd.DataFrame(LF)
    if p_mod:
        P = p_mod(P)
    if life_mod:
        LF = life_mod(LF)
    P.to_csv(rr / 'chains/post_fill/orders.csv', index=False)
    CO.to_csv(rr / 'order_estimator/census_orders.csv', index=False)
    for (d, m), g in LF.groupby(['day', 'mid']):
        pq.write_table(pa.Table.from_pandas(g.drop(columns=['day', 'mid']).reset_index(drop=True), preserve_index=False), life / f'{d}_{m}_orders.parquet')
    sp = {s['h']: s for s in SPEC}
    # the saved POST logic rows (+ rows of other links, which the regrade must leave out)
    EP = []
    for h in CASE_ORDERS:
        s, x = sp[h], derive(sp[h])
        ck = f"order|{s['day']}|{s['mid']}|{ADDR[s['rank']].lower()}|{h}"
        b = x['lo'] >= 0
        q0 = 'unidentified' if not b else ('contradicted' if x['w'] <= 10_000 else 'timing_sensitive')
        for mech, ident, ts in (('queue_priority_before_demand', q0, q0 == 'timing_sensitive'), ('touch_join_or_improve', 'compatible', False),
                                ('reaction_to_price_change', 'timing_sensitive', True), ('inventory_offset', 'contradicted', False)):
            EP.append(dict(episode_id='ep', case_key=ck, link='POST', link_seq=len(EP), mechanism=mech,
                           decision_window_lo_ms=x['lo'] if b else WS[s['mid']] - 300_000, decision_window_hi_ms=x['post_hi'], identification_status=ident,
                           conditional_status='', evidence_grade='B_constrained' if b else 'E_unresolved', timing_sensitive=ts, status=s['st'],
                           supporting_evidence=f"rest_ub={x['w']}ms"))
        EP.append(dict(episode_id='ep', case_key=ck, link='END', link_seq=len(EP), mechanism='pull_after_adverse_move', decision_window_lo_ms=1,
                       decision_window_hi_ms=2, identification_status='unidentified', conditional_status='timing_sensitive', evidence_grade='E_unresolved',
                       timing_sensitive=True, status=s['st'], supporting_evidence=''))
        EP.append(dict(episode_id='ep', case_key=ck, link='KEEP', link_seq=len(EP), mechanism='retain_after_fill', decision_window_lo_ms=1,
                       decision_window_hi_ms=2, identification_status='compatible', conditional_status='', evidence_grade='A_observed',
                       timing_sensitive=False, status=s['st'], supporting_evidence=''))
    for sid, (d, m, r) in ((T1, (DAY_A, MA, 1)), (T3, (DAY_B, MB, 3))):
        EP.append(dict(episode_id='ep', case_key=f'take|{d}|{m}|{ADDR[r].lower()}|{sid}', link='TAKE', link_seq=len(EP), mechanism='inventory_unwind',
                       decision_window_lo_ms=1, decision_window_hi_ms=2, identification_status='compatible', conditional_status='',
                       evidence_grade='A_observed', timing_sensitive=False, status='ok', supporting_evidence=''))
    EP = pd.DataFrame(EP)
    EP['episode_id'] =[E2 if H(1) in k else (E5 if H(2) in k else (E9 if H(3) in k else (E11 if (H(4) in k or H(5) in k) else
                        (E12 if H(6) in k else (E2 if T1 in k else E20))))) for k in EP['case_key']]
    EP.to_csv(rr / 'order_estimator/episodes.csv', index=False)
    # cases (row order = the deterministic selection order; first appearances E5, E2 | E11, E12 | E20)
    CS = []
    for kind, h, eid in (('order', H(2), E5), ('order', H(1), E2), ('order', H(3), E9), ('take', T1, E2), ('order', H(4), E11), ('order', H(5), E11),
                         ('order', H(6), E12), ('take', T3, E20)):
        if kind == 'order':
            s, x = sp[h], derive(sp[h])
            CS.append(dict(kind='order', day=s['day'], mid=s['mid'], addr=ADDR[s['rank']].lower(), key=h, rank=s['rank'],
                           case_key=f"order|{s['day']}|{s['mid']}|{ADDR[s['rank']].lower()}|{h}", episode_id=eid, y_side=s['side'], tick=s['tick'],
                           n_fills=2, cum=10., size_status='tied', union='EX|PL', ub=20., post_lo_union=x['lo'], post_hi=x['post_hi'], rest_ub_ms=x['w'],
                           P_cmd="('EX', 10.0)", P_drop="('PL', 12.0)", P_lot="('EX', 10.0)", P_clip="('EX', 10.0)", prior_sensitive=True,
                           status=s['st'], s=NAN, q=NAN, p_avg=NAN, pos_before=NAN, pos_after=NAN, cls='', cls_simultaneous='', cls_certain='',
                           timing_sensitive=NAN, burst_cert=NAN, burst_pos=NAN))
        else:
            d, m, r = (DAY_A, MA, 1) if h == T1 else (DAY_B, MB, 3)
            CS.append(dict(kind='take', day=d, mid=m, addr=ADDR[r].lower(), key=h, rank=r, case_key=f'take|{d}|{m}|{ADDR[r].lower()}|{h}',
                           episode_id=eid, y_side='', tick=NAN, n_fills=NAN, cum=NAN, size_status='', union='', ub=NAN, post_lo_union=NAN, post_hi=NAN,
                           rest_ub_ms=NAN, P_cmd='', P_drop='', P_lot='', P_clip='', prior_sensitive=NAN, status='ok', s=1., q=5., p_avg=0.4,
                           pos_before=-20., pos_after=-15., cls='reduce_partial', cls_simultaneous='reduce_partial', cls_certain='reduce_partial',
                           timing_sensitive=False, burst_cert=3., burst_pos=4.))
    pd.DataFrame(CS).to_csv(rr / 'order_estimator/cases.csv', index=False)
    ES = []
    for eid, d, m, r in ((E2, DAY_A, MA, 1), (E5, DAY_A, MA, 1), (E9, DAY_A, MA, 1), (E11, DAY_A, MA, 2), (E12, DAY_A, MA, 2), (E20, DAY_B, MB, 3)):
        ES.append(dict(episode_id=eid, day=d, mid=m, addr=ADDR[r].lower(), rank=r, n_cases=1, case_keys='x', ep_n_legs=5, ep_n_maker=4, ep_n_taker=1,
                       ep_sh=50., ep_fee=0.1, ep_net_end=1.5, ep_net_res=2., ep_mq=40., ep_rq=10., ep_pos_in=0., ep_pos_out=-10.,
                       ep_outcome='profitable_continuation', mg_status='ok', mg_close_net=1., mg_resid=0.5, mg_total=1.5, mg_close_net_c_sh=2.))
    pd.DataFrame(ES).to_csv(rr / 'order_estimator/episode_summary.csv', index=False)
    ck1 = f"order|{DAY_A}|{MA}|{A1.lower()}|{H(1)}"
    LED = [dict(case_key=ck1, row_kind='size_end', hypothesis=h_, role='', other_hash='', kind='', status='tied', gap_lo_ms=NAN, gap_hi_ms=NAN,
                feasible_union=fu, feasible_S_FIFO=f1, feasible_S_FIFO_LO=f2, feasible_S_NOPRIO=f3, size_lo=lo_, size_hi=hi_)
           for h_, fu, f1, f2, f3, lo_, hi_ in (('EX', True, True, True, True, 10., 10.), ('PL', True, False, False, True, 10.01, 20.),
                                                ('EC', False, False, False, False, 10.01, NAN))]
    LED.append(dict(case_key=ck1, row_kind='link', hypothesis='overlap|replace|replenish|unrelated', role='child', other_hash=H(12), kind='refill',
                    status='tied', gap_lo_ms=-500., gap_hi_ms=2500., feasible_union=NAN, feasible_S_FIFO=NAN, feasible_S_FIFO_LO=NAN,
                    feasible_S_NOPRIO=NAN, size_lo=NAN, size_hi=NAN))
    pd.DataFrame(LED).to_csv(rr / 'order_estimator/order_ledger.csv', index=False)
    CL = ([dict(day=DAY_A, mid=MA, rank=1, kind='reprice', status='tied')] * 3 + [dict(day=DAY_A, mid=MA, rank=1, kind='reprice', status='sequential_identified')] * 2
          + [dict(day=DAY_A, mid=MA, rank=2, kind='refill', status='sequential_identified')] * 2)
    pd.DataFrame(CL).to_csv(rr / 'order_estimator/census_links.csv', index=False)
    lv = levels_independent(SPEC)
    n_new, n_same = sum(v == 'post_new_level' for v in lv.values()), sum(v == 'post_same_level' for v in lv.values())
    r1 = [s['h'] for s in SPEC if s['rank'] == 1]
    A = []

    def act(t, n, scope='pooled', **kw):
        A.append(dict(dict(action_type=t, scope=scope, day='all', n=n, timing_sensitive_n=0, n_compatible=0, n_contradicted=0, n_timing_sensitive=0,
                           n_unidentified=0, population='take_records_complete' if t.startswith('take_') else 'outcome_selected_filled_orders',
                           extra='{}'), **kw))
    act('maker_fill', 14)
    act('keep_after_fill', 2, n_compatible=2)
    act('reprice', 5, n_compatible=2, n_timing_sensitive=3, timing_sensitive_n=3,
        extra=json.dumps({'link_status': {'sequential_identified': 2, 'tied': 3},
                          'feasible': {'overlap+replace+replenish+unrelated': 3, 'replace+replenish+unrelated': 2}}))
    act('refill', 2, n_compatible=2, extra=json.dumps({'link_status': {'sequential_identified': 2}, 'feasible': {'replace+replenish+unrelated': 2}}))
    act('two_sided', 4, n_compatible=1, n_timing_sensitive=3, timing_sensitive_n=3, extra=json.dumps({'certain': {'True': 1, 'False': 3}}))
    act('ladder_overlap', 2, n_compatible=2)
    act('post_new_level', n_new)
    act('post_same_level', n_same)
    act('post_new_level', sum(lv[h] == 'post_new_level' for h in r1), scope='rank1')
    act('post_same_level', sum(lv[h] == 'post_same_level' for h in r1), scope='rank1')
    act('pull_possible:S_FIFO:cancel_possible', 2)
    act('pull_possible:S_NOPRIO:cancel_possible', 3)
    act('pull_possible:S_FIFO:exhausted_proven', 1)
    act('pull_possible:S_NOPRIO:unidentified', 1)
    for t, n in (('take_reduce_partial', 3), ('take_add', 1), ('take_timing_sensitive', 1), ('take_self_trade', 1), ('take_kind_mint', 2)):
        act(t, n)
    A = pd.DataFrame(A)
    if a_mod:
        A = a_mod(A)
    A.to_csv(rr / 'chains/take/actions.csv', index=False)
    PU = []
    for view, hs in (('S_FIFO', (H(1), H(2))), ('S_NOPRIO', (H(1), H(2), H(5)))):
        for h in hs:
            s = sp[h]
            PU.append(dict(day=s['day'], mid=s['mid'], scope=f"rank{s['rank']}", addr=ADDR[s['rank']], mk_hash=h, view=view, removal_class=CONS,
                           end_histories='EX|PL', identification='tied', pull_identified=False))
    pd.DataFrame(PU).to_csv(rr / 'chains/pull/pulls.csv', index=False)
    return rr, life


def run_quiet(rr, life, out, **kw):
    return CP.run(rroot=rr, life_dir=life, out=out, require_freeze=False, log=lambda *_: None, **kw)


# ------------------------------------------------------------------ sections ------------------------------------------------------------------
BASE = dict(width_ms=500, post_lo_union_ms=1, reason='width_le_late_ms', flags=frozenset(), clock_bad=False, life_class=CONS, status='ok')


def s_precedence():
    check('precedence: the table order is G0, G1, G2, G3, G4', CP.GRADES == ('G0_unbounded_or_invalid', 'G1_conflict', 'G2_bound_constrained',
                                                                           'G3_constrained_status_flagged', 'G4_scenario_only'))
    cases = [('G0 before G1 / G2: unbounded window carrying conflict flags, constrained, ok',
              dict(BASE, width_ms=-1, post_lo_union_ms=-1, reason='window_unbounded', flags=frozenset({'capacity_bounds_void'})), G0),
             ('G0: a lifecycle INVALID flag on a bounded window beats a conflict flag',
              dict(BASE, flags=frozenset({'unplaceable_leg', 'capacity_bounds_void'}), status='unplaceable_leg'), G0),
             ('G0: packet reason invalid_order', dict(BASE, reason='invalid_order'), G0),
             ('G1 before G2: an estimator-only capacity flag on a constrained ok row', dict(BASE, flags=frozenset({'capacity_bounds_void'})), G1),
             ('G1 before G3: post_after_first_fill on a constrained row', dict(BASE, flags=frozenset({'post_after_first_fill'}), status='post_after_first_fill'), G1),
             ('G1: clock_bad', dict(BASE, clock_bad=True), G1),
             ('G1: first_fill_clock reason', dict(BASE, reason='first_fill_clock'), G1)]
    cases += [(f'G1: flag {f}', dict(BASE, flags=frozenset({f}), status=f), G1) for f in sorted(CP.G1_FLAGS)]
    cases += [('G2: constrained, ok', BASE, G2), ('G2: observed, ok', dict(BASE, life_class='observed'), G2),
              ('G2 needs no estimator flag: constrained + estimator ahead_hi_negative -> G3', dict(BASE, flags=frozenset({'ahead_hi_negative'})), G3),
              ('G3: constrained, status ahead_hi_negative', dict(BASE, flags=frozenset({'ahead_hi_negative'}), status='ahead_hi_negative'), G3),
              ('G3: observed, status flagged', dict(BASE, life_class='observed', flags=frozenset({'leg_conflict'}), status='leg_conflict'), G3),
              ('G4: assumption_sensitive, ok', dict(BASE, life_class=SENS), G4), ('G4: unresolved, bounded', dict(BASE, life_class=UNRES), G4),
              ('G4: scenario rows disagree', dict(BASE, life_class=CP.LIFE_DISAGREE), G4),
              ('G4 when wide too', dict(BASE, width_ms=50_000, reason='width_gt_early_ms', life_class=SENS), G4)]
    for name, o, exp in cases:
        got = CP.grade(o)[0]
        check(f'precedence: {name}', got == exp, f'got {got}, expected {exp}')
    o = dict(BASE, clock_bad=True)                                 # G1 and G2 both hold
    with Mut('g2_before_g1'):
        got = CP.grade(o)[0]
    check('precedence mutant g2_before_g1 is caught (G1 row graded G2 under it)', got == G2 and CP.grade(o)[0] == G1, got)


def s_grade_unit():
    o = dict(BASE, width_ms=5000, reason='width_le_early_ms', life_class=SENS)
    check('grade falsifier: a union-short assumption_sensitive ok row is G4, never G2', CP.grade(o)[0] == G4, CP.grade(o))
    with Mut('grade_by_width'):
        got = CP.grade(o)[0]
    check('grade falsifier: mutant grade_by_width is caught (it grades the row G2)', got == G2, got)
    o2 = dict(BASE, width_ms=900, reason='width_le_late_ms', life_class=CP.LIFE_DISAGREE)
    check('grade falsifier: scenario rows that disagree never grade G2', CP.grade(o2)[0] == G4)


def s_logic():
    lc = CP.logic_correction
    q = CP.QUEUE_MECH
    rows = [((G2, 500, q, 'contradicted', False), ('B_constrained', 'contradicted', False)),
            ((G2, 20_000, q, 'timing_sensitive', True), ('B_constrained', 'timing_sensitive', True)),
            ((G4, 500, q, 'contradicted', False), ('C_scenario', 'timing_sensitive', True)),
            ((G4, 20_000, q, 'timing_sensitive', True), ('C_scenario', 'timing_sensitive', True)),
            ((G0, -1, q, 'unidentified', False), ('E_unresolved', 'unidentified', False)),
            ((G1, 800, q, 'contradicted', False), ('E_unresolved', 'unidentified', False)),
            ((G3, 3000, q, 'contradicted', False), ('E_unresolved', 'unidentified', False)),
            ((G4, 500, 'touch_join_or_improve', 'compatible', False), ('C_scenario', 'compatible', False)),
            ((G2, 500, 'inventory_offset', 'contradicted', False), ('B_constrained', 'contradicted', False))]
    for a, exp in rows:
        got = lc(*a)
        check(f'logic: {a[0]} w={a[1]} {a[2]}', got == exp, f'got {got}, expected {exp}')


def s_paths_unit():
    cs = pd.DataFrame(dict(episode_id=['e5', 'e2', 'e5', 'e9', 'e2', 'e11', 'e12', 'e13', 'e20'], rank=[1, 1, 1, 1, 1, 2, 2, 2, 3]))
    got = CP.select_paths(cs)
    check('paths: the first two episodes per rank by first appearance (not sorted ids), up to 6',
          got == [(1, 'e5'), (1, 'e2'), (2, 'e11'), (2, 'e12'), (3, 'e20')], got)


def links_honest(CEN):
    """the observed-vs-inferred falsifier on a census: tied links stay 'tied', never identified, and no link counts an identified repost."""
    lk = CEN[CEN['evidence_grade'] == 'scenario_dependent_link']
    tied = lk[lk['subset'] == 'tied']
    ident = np.array([bool(x) for x in lk['identified_repost']], bool)
    n_id = float(np.nan_to_num(lk['n_identified'].to_numpy(float), nan=0.).sum())
    rep = tied[(tied['source'] == 'chains/take/actions.csv') & (tied['action_type'] == 'reprice') & (tied['scope'] == 'pooled')]
    suc = tied[(tied['action_type'] == 'successor_reprice') & (tied['scope'] == 'pooled') & (tied['day'] == 'all')]
    return (not ident.any()) and n_id == 0 and len(rep) == 1 and int(rep['n'].iloc[0]) == 3 and len(suc) == 1 and int(suc['n'].iloc[0]) == 2


def s_world(tmp):
    rr, life = build_world(tmp / 'w1')
    D, L, info = CP.read_inputs(rr, life)
    res = CP.build(D, L)
    T = res['T'].set_index('mk_hash')
    check('world: every order graded as designed', all(T.at[h, 'grade'] == g for h, g in EXPECT_G.items()),
          {h[-2:]: (T.at[h, 'grade'], g) for h, g in EXPECT_G.items() if T.at[h, 'grade'] != g})
    check('world: every hard fact holds on a consistent world', all(v['holds'] for v in res['hard'].values()),
          [k for k, v in res['hard'].items() if not v['holds']])
    check('world: another maker\'s lifecycle rows are never joined', len(res['T']) == len(SPEC) and H(900) not in set(res['T']['mk_hash']))
    check('world: life classes that disagree are counted and graded G4', T.at[H(8), 'life_post_class'] == CP.LIFE_DISAGREE
          and res['soft']['life_post_class_identical_across_scenarios']['n_true'] == len(SPEC) - 1)
    check('world: G1 flags recorded per order', T.at[H(11), 'g1_flags_matched'] == 'capacity_bounds_void' and T.at[H(10), 'g1_flags_matched'] == '')

    def honest_grades(T_):
        return not ((T_['grade'] == G2) & ~T_['life_post_class'].isin(CP.CERT)).any()
    check('grade falsifier (world): no non-certified row graded G2', honest_grades(res['T']))
    with Mut('grade_by_width'):
        rm = CP.build(D, L)
    check('grade falsifier (world): mutant grade_by_width is caught', not honest_grades(rm['T']),
          rm['T'][['mk_hash', 'grade', 'life_post_class']].to_dict('records'))
    with Mut('g2_before_g1'):
        rp = CP.build(D, L)
    check('precedence (world): mutant g2_before_g1 is caught (the clock-flagged row becomes G2)',
          rp['T'].set_index('mk_hash').at[H(10), 'grade'] == G2 and T.at[H(10), 'grade'] == G1)
    # logic rows
    E = res['EPR']
    exp_q = {H(1): ('B_constrained', 'contradicted'), H(2): ('C_scenario', 'timing_sensitive'), H(3): ('B_constrained', 'timing_sensitive'),
             H(4): ('E_unresolved', 'unidentified'), H(5): ('E_unresolved', 'unidentified'), H(6): ('E_unresolved', 'unidentified')}
    q = E[E['mechanism'] == CP.QUEUE_MECH].set_index('case_hash')
    check('logic (world): 24 POST rows only (END / KEEP / TAKE rows left out)', len(E) == 24 and set(E['link']) == {'POST'}, len(E))
    check('logic (world): corrected queue grade / status per order', all((q.at[h, 'evidence_grade_corrected'], q.at[h, 'identification_status_corrected']) == v
                                                                       for h, v in exp_q.items()),
          {h[-2:]: (q.at[h, 'evidence_grade_corrected'], q.at[h, 'identification_status_corrected']) for h in exp_q})
    check('logic (world): originals kept beside', q.at[H(2), 'evidence_grade'] == 'B_constrained' and q.at[H(2), 'identification_status'] == 'contradicted'
          and bool(q.at[H(2), 'identification_status_changed']) and not bool(q.at[H(1), 'identification_status_changed']))
    o = E[(E['mechanism'] == 'touch_join_or_improve')].set_index('case_hash')
    check('logic (world): other mechanisms keep their status with the corrected grade',
          o.at[H(2), 'identification_status_corrected'] == 'compatible' and o.at[H(2), 'evidence_grade_corrected'] == 'C_scenario')
    check('logic (world): claims count the withdrawn contradictions', res['claims']['logic_rows']['queue_contradicted_withdrawn'] == 3,
          res['claims']['logic_rows'])
    # census
    C_ = res['CEN']
    check('census falsifier: a tied link is never an identified repost (tied 3 / successor tied 2)', links_honest(C_),
          C_[C_['evidence_grade'] == 'scenario_dependent_link'].to_dict('records')[:4])
    with Mut('tied_as_identified'):
        cm = CP.build(D, L)
    check('census falsifier: mutant tied_as_identified is caught', not links_honest(cm['CEN']))
    check('census: every row graded', bool(C_['evidence_grade'].isin(CP.CENSUS_GRADES).all()) and res['hard']['every_census_row_graded']['holds'])
    seq = C_[(C_['action_type'] == 'reprice') & (C_['subset'] == 'sequential_under_scenarios')]
    check('census: sequential links counted apart and never as identified', len(seq) == 1 and int(seq['n'].iloc[0]) == 2 and not bool(seq['identified_repost'].iloc[0]))
    lv = levels_independent(SPEC)
    ps = C_[(C_['action_type'] == 'post_same_level') & (C_['scope'] == 'pooled')].set_index('subset')
    pn = C_[(C_['action_type'] == 'post_new_level') & (C_['scope'] == 'pooled')].set_index('subset')
    exp_new = Counter(EXPECT_G[h] for h, v in lv.items() if v == 'post_new_level')
    check('census: post levels re-derived (11 new / 1 same) and split by the section-1 grade',
          int(ps.at[G2, 'n']) == 1 and int(ps['n'].sum()) == 1 and all(int(pn.at[g, 'n']) == exp_new.get(g, 0) for g in CP.GRADES)
          and res['hard']['post_levels_rederived_equal']['holds'] and sum(v == 'post_new_level' for v in lv.values()) == 11)
    ts = C_[C_['action_type'] == 'two_sided'].set_index('subset')
    check('census: two-sided split overlap_certain 1 / feasible_only 3, overlap family grade',
          int(ts.at['overlap_certain', 'n']) == 1 and int(ts.at['feasible_only', 'n']) == 3 and set(ts['evidence_grade']) == {'feasible_timing_sensitive_overlap'})
    cc = C_[C_['action_type'] == 'pull_possible:S_NOPRIO:cancel_possible']
    check('census: cancel-compatible endings graded with pull_identified counted from pulls.csv',
          len(cc) == 1 and cc['evidence_grade'].iloc[0] == 'cancel_compatible_ending' and float(cc['n_identified'].iloc[0]) == 0.
          and res['soft']['pull_cancel_counts_equal_pulls_csv']['holds'])
    g = dict(zip(C_['action_type'], C_['evidence_grade']))
    check('census: families per type', g['maker_fill'] == 'observed_execution' and g['keep_after_fill'] == 'continued_execution'
          and g['take_timing_sensitive'] == 'observed_execution' and g['pull_possible:S_FIFO:exhausted_proven'] == 'scenario_dependent_ending'
          and g['pull_possible:S_NOPRIO:unidentified'] == 'unresolved_alternatives' and g['successor_none'] == 'unresolved_alternatives', g)
    check('census: link split equals census_links', res['soft']['link_split_equals_census_links']['holds'])
    # the census writer
    bad = C_.copy()
    bad.loc[0, 'evidence_grade'] = ''
    ok, info = refuses(lambda: CP.census_bytes(bad), 'census refuses an ungraded action row')
    check('census writer refuses an ungraded action row (reason asserted)', ok, info)
    with Mut('ungraded_ok'):
        ok_m, info_m = refuses(lambda: CP.census_bytes(bad), 'census refuses an ungraded action row')
    check('census writer mutant ungraded_ok is caught (the refusal no longer fires)', not ok_m, info_m)
    bad2 = C_.copy()
    i = int(np.flatnonzero((bad2['subset'] == 'tied').to_numpy())[0])
    bad2.loc[i, 'identified_repost'] = True
    expect('census writer refuses a tied link counted as an identified repost', lambda: CP.census_bytes(bad2), 'counted as an identified repost')
    check('census writer passes the honest census', len(CP.census_bytes(C_)) > 0)
    # paths
    md = res['md']
    heads = [f'## Path {k} - rank {r}: episode {e}' for k, (r, e) in enumerate([(1, E5), (1, E2), (2, E11), (2, E12), (3, E20)], 1)]
    pos = [md.find(x) for x in heads]
    check('paths: five predetermined episodes in first-appearance order', all(p >= 0 for p in pos) and pos == sorted(pos) and '## Path 6' not in md, pos)
    blk = md[pos[1]:pos[2]]
    check('paths: the E2 path shows post -> fills -> terminal -> successor -> take -> economics in order',
          all(x in blk for x in ('**POST**', '**FILLS**', '**TERMINAL**', '**SUCCESSOR**', '**TAKE**', '**ECONOMICS**'))
          and blk.index('**POST**') < blk.index('**FILLS**') < blk.index('**TERMINAL**') < blk.index('**SUCCESSOR**') < blk.index('**TAKE**')
          < blk.index('**ECONOMICS**'), blk[:300])
    check('paths: post / terminal / successor / take each carry their alternative', blk.count('alternative that changes the logic') == 4,
          blk.count('alternative that changes the logic'))
    check('paths: the tied successor is shown tied, never as an identified repost', 'link status tied' in blk and 'identified repost: no' in blk
          and 'identified repost: yes' not in md)
    check('paths: the per-scenario terminal sets come from the ledger', 'S_FIFO {EX}' in blk and 'S_NOPRIO {EX,PL}' in blk, blk[:900])
    check('paths: an order-less (take-only) episode is still a path', '**TAKE**' in md[pos[4]:] and '**POST**' not in md[pos[4]:])


def s_end_to_end(tmp):
    rr, life = build_world(tmp / 'w2')
    out = rr / 'chains' / 'correction_v1'
    out.mkdir(parents=True)
    sib = out / 'pull_publication.csv'
    sib.write_text('sibling,file\n1,2\n')
    s0 = sha(sib)
    rec = run_quiet(rr, life, out)
    files = {nm: out / nm for nm in CP.OUT_FILES}
    check('e2e: all seven files written', all(p.is_file() for p in files.values()), [nm for nm, p in files.items() if not p.is_file()])
    check('e2e: the sibling file is untouched', sha(sib) == s0)
    rj = json.loads(files['receipt_post.json'].read_text())
    check('e2e: receipt binds the sha256 of every other output', all(rj['outputs'][nm]['sha256'] == sha(files[nm]) for nm in CP.OUT_FILES[:-1]))
    want = {k for k, _ in CP.INPUTS} | {f'life/{DAY_A}_{MA}_orders.parquet', f'life/{DAY_B}_{MB}_orders.parquet'}
    check('e2e: receipt binds every input read (9 CSVs + 2 lifecycle files)', set(rj['inputs']) == want
          and all(rj['inputs'][k]['sha256'] == sha(Path(rj['inputs'][k]['path']) if Path(rj['inputs'][k]['path']).is_absolute()
                                                   else CP.C.REPO / rj['inputs'][k]['path']) for k in want), sorted(rj['inputs']))
    check('e2e: receipt carries sources, CPU and VmHWM', CP._rel(CP.__file__) in rj['sources'] and rj['cpu_s']['total'] > 0 and rj['vmhwm_mib'] > 0)
    cl = json.loads(files['claims_post.json'].read_text())
    check('e2e: claims grade counts', cl['grades']['by_grade'] == {G0: 2, G1: 3, G2: 3, G3: 1, G4: 3}, cl['grades']['by_grade'])
    check('e2e: allowed claim = G2 with a window <= 10 s (2; <= 1 s: 1)', cl['allowed_claims']['n_bound_constrained_window_le_10s'] == 2
          and cl['allowed_claims']['n_bound_constrained_window_le_1s'] == 1)
    rep = cl['reviewer_reproduction']
    check('e2e: reviewer predicates recorded (computed, not asserted equal on a synthetic world)',
          rep['scenario_short']['computed'] == 8 and rep['scenario_short']['reviewer'] == 2917 and rep['scenario_short']['equal'] is False
          and rep['short_life_constrained_status_ok']['computed'] == 3 and rep['successor_tied']['computed'] == 3,
          {k: rep[k]['computed'] for k in ('scenario_short', 'short_life_constrained_status_ok', 'successor_tied')})
    check('e2e: claims never state a resting share of all orders', 'rested' not in files['claims_post.json'].read_text())
    cc = cl['census']
    check('e2e: census claims count each action once (pooled / all rows only)',
          cc['actions_by_grade_pooled_all'].get('scenario_dependent_link') == 7 and cc['actions_by_grade_pooled_all'].get('observed_execution') == 22
          and cc['actions_by_grade_pooled_all'].get(f'post_window:{G2}') == 3, cc['actions_by_grade_pooled_all'])
    check('e2e: census claims split links and successors tied / sequential',
          cc['links_pooled_all'] == {'reprice|sequential_under_scenarios': 2, 'reprice|tied': 3, 'refill|sequential_under_scenarios': 2}
          and cc['successors_pooled_all'] == {'successor_reprice|sequential_under_scenarios': 1, 'successor_reprice|tied': 2,
                                              'successor_refill|sequential_under_scenarios': 1, 'successor_refill|tied': 1, 'successor_none|none': 7}
          and cc['n_pull_identified_pooled_all'] == {'pull_possible:S_FIFO:cancel_possible': 0, 'pull_possible:S_NOPRIO:cancel_possible': 0},
          {k: cc[k] for k in ('links_pooled_all', 'successors_pooled_all', 'n_pull_identified_pooled_all')})
    g = pd.read_csv(files['post_regrade.csv'], dtype={'day': str})
    x = g[(g['scope'] == 'pooled') & (g['day'] == 'all') & (g['grade'] == G2) & (g['width_bin'] == 'le_1s')]
    check('e2e: grid 4 scopes x 3 days x 5 grades x 4 bins with denominators', len(g) == 240 and int(x['n_orders'].iloc[0]) == 1
          and int(x['n_scope_day'].iloc[0]) == 12 and abs(float(x['frac_of_scope_day'].iloc[0]) - 1 / 12) < 1e-12, len(g))
    po = pd.read_csv(files['post_regrade_orders.csv'], dtype={'day': str}).set_index('mk_hash')
    check('e2e: per-order file carries the grade and its inputs', all(po.at[h, 'grade'] == gg for h, gg in EXPECT_G.items())
          and {'life_post_class', 'status_packet', 'census_order_flags', 'g1_flags_matched', 'window_width_ms'} <= set(po.columns))
    ep = pd.read_csv(files['episodes_post_regraded.csv'])
    check('e2e: episodes file = 24 POST rows with corrected columns beside', len(ep) == 24 and {'evidence_grade', 'evidence_grade_corrected',
                                                                                              'identification_status_corrected'} <= set(ep.columns))
    ce = pd.read_csv(files['census_graded.csv'], dtype={'day': str})
    check('e2e: census file parses back with every row graded', len(ce) > 0 and bool(ce['evidence_grade'].isin(CP.CENSUS_GRADES).all()))
    check('e2e: paths file written', files['paths.md'].read_text().startswith('# Logic-chain correction v1'))
    h0 = {nm: sha(p) for nm, p in files.items()}
    # write-once: a second run refuses before any input is read and changes nothing
    ok, info = refuses(lambda: run_quiet(rr, life, out), 'write-once')
    check('refusal: an existing output refuses the run (write-once, before any input is read)', ok and 'refused before any input is read' in info, info)
    check('refusal: nothing was overwritten', all(sha(p) == h0[nm] for nm, p in files.items()))
    out2 = rr / 'chains' / 'correction_v1b'
    out2.mkdir()
    (out2 / 'census_graded.csv').write_text('x\n')
    ok, info = refuses(lambda: run_quiet(rr, life, out2), 'write-once')
    check('refusal: one existing name refuses and no other output is created', ok and sorted(p.name for p in out2.iterdir()) == ['census_graded.csv'], info)
    # dry: writes nothing
    out3 = rr / 'chains' / 'correction_dry'
    dg = run_quiet(rr, life, out3, dry=True)
    check('dry: nothing written (no directory created), hard checks pass', not out3.exists() and dg['hard_failed'] == [], dg.get('hard_failed'))
    with Mut('grade_by_width'):
        ok, info = refuses(lambda: run_quiet(rr, life, rr / 'chains' / 'c_mut'), 'planted mutant')
    check('refusal: a mutant never runs in run mode', ok, info)


def s_refusals(tmp):
    def bad_day(P):
        P.loc[0, 'day'] = '20261108'
        return P
    rr, life = build_world(tmp / 'w3', p_mod=bad_day)
    expect('refusal: a non-measurement day in the orders', lambda: run_quiet(rr, life, tmp / 'w3' / 'out'), 'not all measurement days')

    def drop_row(LF):
        return LF[~((LF['mk_hash'] == H(1)) & (LF['scenario'] == 'S_FIFO_LO'))]
    rr, life = build_world(tmp / 'w4', life_mod=drop_row)
    expect('refusal: an order without one of its three scenario rows', lambda: run_quiet(rr, life, tmp / 'w4' / 'out'), 'lacks one of the three scenario rows')

    def break_union(P):
        i = int(np.flatnonzero((P['mk_hash'] == H(2)).to_numpy())[0])
        P.loc[i, 'post_lo_union_ms'] = int(P.loc[i, 'post_lo_union_ms']) + 1000
        P.loc[i, 'window_width_ms'] = int(P.loc[i, 'post_hi_ms']) - int(P.loc[i, 'post_lo_union_ms'])
        return P
    rr, life = build_world(tmp / 'w5', p_mod=break_union)
    ok, info = refuses(lambda: run_quiet(rr, life, tmp / 'w5' / 'out'), 'hard checks failed')
    check('refusal: a broken union fact (lower end not the scenario minimum) refuses before any output', ok and 'union_lower_is_min_of_three_scenarios' in info
          and not (tmp / 'w5' / 'out').exists(), info)

    def extra_type(A):
        return pd.concat([A, A.iloc[[0]].assign(action_type='take_inferred_pull')], ignore_index=True)
    rr, life = build_world(tmp / 'w6', a_mod=extra_type)
    ok, info = refuses(lambda: run_quiet(rr, life, tmp / 'w6' / 'out'), 'every_census_row_graded')
    check('refusal: an action type without a declared grade refuses the run', ok and not (tmp / 'w6' / 'out').exists(), info)


def main():
    tmp = Path(tempfile.mkdtemp(prefix='chain_correct_post_selftest_'))
    os.environ.pop('MT_CC_MUT', None)
    try:
        section('precedence', s_precedence)
        section('grade', s_grade_unit)
        section('logic', s_logic)
        section('paths', s_paths_unit)
        section('world', s_world, tmp)
        section('end to end', s_end_to_end, tmp)
        section('refusals', s_refusals, tmp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    n_bad = sum(not ok for _, ok, _ in RES)
    for name, ok, info in RES:
        print(('PASS ' if ok else 'FAIL ') + name + ('' if ok else f'  :: {info}'))
    print(json.dumps(dict(checks=len(RES), failed=n_bad, vmhwm_mib=round(CP.hwm(), 1))))
    sys.exit(1 if n_bad else 0)


if __name__ == '__main__':
    main()
