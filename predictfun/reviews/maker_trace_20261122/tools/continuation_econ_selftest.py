"""continuation_econ_selftest.py - synthetic, in-process proof of continuation_econ.py (P-2026-005 MAKER-TRACE; COMPETING-TRANSITION-AND-CONTINUATION
v1 section 5; session aea5bfb9 builder, 2026-11-23). No real data path is opened: a synthetic saved-evidence tree (every file bound by sha256 in
its producer's receipt, the real column names and dtypes, dictionary-encoded fills_valued columns) is written to a temporary directory and the
production run() reads it end to end.
TOY (hand-computed FIFO, address A1, market M1, END = we = 1,300,000 ms): order H1 = a 12 sh YES bid at 40c filled at 1,010,000 (fee 0.012);
  its ok non-self lots in FIFO order: H1 buy 12 @ .40, taker sell 6 @ .45 (1,015,000, fee .06), maker ask H4 sell 3 @ .50 (1,080,000, fee .009),
  taker sell 1 @ .42 (1,305,000, after END, fee .01) -> closes c1 6 sh (hold 5 s, pnl .30, fees .006 + .06, net .234), c2 3 sh (hold 70 s, pnl .30,
  fees .003 + .009, net .288), c3 1 sh (after END, pnl .02, fees .001 + .01, net .009); residual 2 sh: to END 2 (.43 - .40) = .06, to resolution
  (V = 1) 2 (1 - .40) = 1.2, fee .012 * 2 / 12 = .002. H1's self-trade (5 sh) and record_mismatch (2 sh) legs are excluded lots. M2: H7 sell 2
  @ .62 closed by a taker buy 2 @ .60 after 4 s (c4, net .04); H6 buy 1 @ .55 unclosed (V = 0). A2 (rank 5): H2 buy 4 @ .39 closed by a maker
  sell 4 @ .41 after 4 s (c5, net .08). A3 (outside the top 10): H3 sell 5 @ .60 unclosed. Every other take record of A1 has excluded legs.
CASES
 T1 the toy: per-order windows le10s / le60s / leEND / afterEND, closing-earlier (H4) kept apart, residual, fees, unavailable (H3), pooled per
    scope; every acceptance predicate and verifier check holds; every writer runs (five files; receipt sha256 = the written bytes).
 T2 a take crossing END (t10 straddles, t9 after): excluded from every window, counted as statuses, the saved unbounded counts reproduced;
    mutant no_end_bound caught (takes_within_end_bound, closes_within_end_bound); a run with it refuses naming the check.
 T3 a duplicated close across two links (H1 is response link e1 AND chain 1; c2 is H1's entry close and H4's closing-earlier close): pooled
    once; mutant no_close_dedup caught (pooled_closes_distinct); a duplicated closes.parquet row refused.
 T4 a self-trade / non-ok leg entering value: counted, never valued; mutant no_cohort caught (valued_legs_clean).
 T5 a same-frame deep creation inheriting restoring context (chain 2's ask 62c cell in the frame of a response at 61c; link e5's order at 55c
    vs resp_tick 57): contextual_only; mutant no_tick_guard caught (restoring_context_tick_equal).
 T6 write-once refusal asserting its message; run --dry writes nothing; a mutant / require_freeze=False / an unknown mutant outside the
    selftest refused with their messages.
 T7 instrument falsifiers: the chains.md parser flags a malformed creation line; an input changed after its receipt fails inputs_bound and a
    run refuses naming it.
Usage (run_light, MT_RUN=v2): continuation_econ_selftest.py"""
import hashlib
import json
import os
import sys
import tempfile
import traceback
from pathlib import Path

sys.dont_write_bytecode = True

import pandas as pd  # noqa: E402
import pyarrow as pa  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import continuation_econ as CE  # noqa: E402

NAN = float('nan')
DAY = '20261024'
M1, M2 = 3000001, 3000002
WS = {M1: 1_000_000, M2: 2_000_000}
WE = {M1: 1_300_000, M2: 2_300_000}
A1 = '0xAAAA' + '0' * 35 + '1'                                     # rank 1 (top 3, top 10)
A2 = '0xBBBB' + '0' * 35 + '2'                                     # rank 5 (top 10, not top 3)
A3 = '0xCCCC' + '0' * 35 + '3'                                     # outside the closes population
AX = '0xDDDD' + '0' * 35 + '4'                                     # rank 2, no candidate order (filtered out)
H1, H2, H3, H4, H5, H6, H7, H8, H9 = ('0x' + str(i) * 64 for i in range(1, 10))
TKH = '0x' + 'e' * 64
EXP = dict(n_links=5, n_chains=2)
CENSUS_TR = "1 compatible creation(s): inside 1; census {'matched': 1}; response episodes 1"
RESULTS = []


def quiet(*a, **k):
    return None


def check(name, ok, detail=''):
    RESULTS.append((name, bool(ok)))
    print(f"{'PASS' if ok else 'FAIL'} {name}" + ('' if ok or not detail else f' - {str(detail)[:400]}'), flush=True)


def expect_refusal(name, fn, *needles):
    try:
        fn()
    except ValueError as e:
        msg = str(e)
        check(name, all(n in msg for n in needles), f'message: {msg}')
        return
    check(name, False, 'not refused')


def near(a, b, tol=1e-9):
    return abs(float(a) - float(b)) <= tol


def cs_eq(s, n, sh, pnl, fee, net, nd=0):
    return s['n'] == n and s['n_dust'] == nd and near(s['sh'], sh) and near(s['pnl'], pnl) and near(s['fee'], fee) and near(s['net'], net)


def with_mut(m, fn):
    os.environ[CE.MUT_ENV] = m
    try:
        return fn()
    finally:
        os.environ.pop(CE.MUT_ENV, None)


# ------------------------------------------------------------------ the synthetic saved-evidence tree ------------------------------------------------------------------
FV_T = dict(day=pa.string(), mid=pa.int64(), sid=pa.string(), mk_idx=pa.int64(), role=pa.string(), addr=pa.string(), hash=pa.string(), y_side=pa.string(),
            s=pa.int64(), q=pa.float64(), p=pa.float64(), self_trade=pa.bool_(), settle_ms=pa.int64(), V=pa.float64(), st_res=pa.string(),
            st_10s=pa.string(), v_10s=pa.float64(), st_60s=pa.string(), v_60s=pa.float64(), m_end=pa.float64(), st_end=pa.string(), v_end=pa.float64(),
            fee_fill=pa.float64(), net_res=pa.float64(), mq=pa.float64(), rq=pa.float64(), status=pa.string())
XC_T = dict(day=pa.string(), mid=pa.int64(), addr=pa.string(), rank=pa.int64(), entry_sid=pa.string(), entry_mk_idx=pa.int64(), entry_hash=pa.string(),
            entry_role=pa.string(), entry_side=pa.string(), entry_settle_ms=pa.int64(), entry_p=pa.float64(), close_sid=pa.string(), close_mk_idx=pa.int64(),
            close_hash=pa.string(), close_role=pa.string(), close_settle_ms=pa.int64(), close_p=pa.float64(), role_pair=pa.string(), sh=pa.float64(),
            pnl=pa.float64(), fee_fill_entry=pa.float64(), fee_fill_close=pa.float64(), fee_res_entry=pa.float64(), fee_res_close=pa.float64(),
            net=pa.float64(), net_feeres=pa.float64(), hold_ms=pa.int64(), dust=pa.bool_(), res_ok=pa.bool_())
EP_T = dict(day=pa.string(), mid=pa.int64(), episode_id=pa.int64(), side=pa.string(), delta_ticks=pa.float64(), end_type=pa.string(), end_reason=pa.string(),
            exposure_ms=pa.int64(), no_frame_between=pa.bool_(), resp_type=pa.string(), resp_lo_ms=pa.int64(), resp_hi_ms=pa.int64(), state_status_D0=pa.string(),
            state_ms_D0=pa.int64(), time_left_s_D0=pa.float64(), cand_addr=pa.string(), cand_group=pa.string(), cand_hash=pa.string(), grade=pa.string(),
            identity=pa.string(), identity_reason=pa.string(), alt_netting=pa.string(), alt_first_other=pa.string(), alt_first_addresses=pa.string(),
            pre_state_link=pa.string(), po_parity=pa.string(), win_unbounded=pa.bool_(), win_contains=pa.bool_(), resp_tick=pa.int64(), resp_size_shown=pa.float64())
EV_T = dict(day=pa.string(), mid=pa.int64(), addr=pa.string(), sid=pa.string(), s=pa.int64(), dq=pa.float64(), x_lo_hard_ms=pa.int64(), hi_ms=pa.int64(),
            status=pa.string(), o4_status=pa.string(), dir_D0=pa.string(), dir_D1000=pa.string(), dir_pend=pa.string())


def fv(mid, sid, role, addr, hsh, side, s, q, p, self_, settle, V, m_end, v10, v60, vend, fee, net_res, mq, rq, status='ok', mk=0):
    return dict(day=DAY, mid=mid, sid=sid, mk_idx=mk, role=role, addr=addr, hash=hsh, y_side=side, s=s, q=q, p=p, self_trade=self_, settle_ms=settle, V=V,
                st_res='ok', st_10s='ok', v_10s=v10, st_60s='ok', v_60s=v60, m_end=m_end, st_end='ok', v_end=vend, fee_fill=fee, net_res=net_res, mq=mq,
                rq=rq, status=status)


FV_ROWS = [
    fv(M1, 'sA1', 'maker', A1, H1, 'bid', 1, 12., .40, False, 1_010_000, 1., .43, .24, .36, .36, .012, 7.188, 10., 2.),
    fv(M1, 'sA2', 'maker', A1, H1, 'bid', 1, 5., .40, True, 1_020_000, 1., .43, .10, .15, .15, .005, 2.995, 0., 0.),
    fv(M1, 'sA3', 'maker', A1, H1, 'bid', 1, 2., .40, False, 1_030_000, 1., .43, .04, .06, .06, .002, 1.198, NAN, NAN, status='record_mismatch', mk=1),
    fv(M1, 'tA1', 'taker', A1, TKH, 'ask', -1, 6., .45, False, 1_015_000, 1., .43, 0., 0., .12, .06, -3.36, 6., 0.),
    fv(M1, 'sA4', 'maker', A1, H4, 'ask', -1, 3., .50, False, 1_080_000, 1., .43, .03, .06, .21, .009, -1.509, 3., 0.),
    fv(M1, 'tA9', 'taker', A1, TKH, 'ask', -1, 1., .42, False, 1_305_000, 1., .43, 0., 0., 0., .01, -.59, 1., 0.),
    fv(M1, 'tA0', 'taker', A1, TKH, 'ask', -1, 1., .41, False, 1_001_000, 1., .43, 0., 0., 0., 0., 0., NAN, NAN, status='src_status'),
    fv(M1, 'tA12', 'taker', A1, TKH, 'bid', 1, 2., .44, False, 1_041_000, 1., .43, 0., 0., 0., 0., 0., NAN, NAN, status='src_status'),
    fv(M1, 'tA11', 'taker', A1, TKH, 'ask', -1, 1., .42, False, 1_050_000, 1., .43, 0., 0., 0., 0., 0., NAN, NAN, status='src_status'),
    fv(M1, 'tA10', 'taker', A1, TKH, 'ask', -1, 1., .42, False, 1_300_500, 1., .43, 0., 0., 0., 0., 0., NAN, NAN, status='src_status'),
    fv(M2, 'sB1', 'maker', A1, H7, 'ask', -1, 2., .62, False, 2_106_000, 0., .58, .02, .04, .08, 0., 1.24, 2., 0.),
    fv(M2, 'tB3', 'taker', A1, TKH, 'bid', 1, 2., .60, False, 2_110_000, 0., .58, 0., 0., -.04, 0., -1.2, 2., 0.),
    fv(M2, 'sB2', 'maker', A1, H6, 'bid', 1, 1., .55, False, 2_201_000, 0., .58, .01, .02, .03, 0., -.55, 0., 1.),
    fv(M1, 'sC1', 'maker', A2, H2, 'bid', 1, 4., .39, False, 1_100_000, 1., .43, .04, .08, .16, 0., 2.44, 4., 0.),
    fv(M1, 'sC2', 'maker', A2, H5, 'ask', -1, 4., .41, False, 1_104_000, 1., .43, 0., 0., .08, 0., -2.36, 4., 0.),
    fv(M2, 'sD1', 'maker', A3, H3, 'ask', -1, 5., .60, False, 2_050_000, 0., .58, .05, .05, .10, 0., 3.0, 0., 5.),
    fv(M1, 'sX1', 'maker', AX, H9, 'bid', 1, 7., .30, False, 1_020_000, 1., .43, 0., 0., 0., 0., 4.9, 7., 0.),
    fv(M1, 'sX2', 'maker', AX, H8, 'ask', -1, 7., .31, False, 1_022_000, 1., .43, 0., 0., 0., 0., -4.83, 7., 0.)]


def cl(mid, addr, rank, es, eh, er, eside, eset, ep, cs, ch, cr, cset, cp, sh, ffe, ffc):
    pnl = sh * ((cp - ep) if eside == 'buy' else (ep - cp))
    return dict(day=DAY, mid=mid, addr=addr, rank=rank, entry_sid=es, entry_mk_idx=0, entry_hash=eh, entry_role=er, entry_side=eside, entry_settle_ms=eset,
                entry_p=ep, close_sid=cs, close_mk_idx=0, close_hash=ch, close_role=cr, close_settle_ms=cset, close_p=cp, role_pair=f'{er}->{cr}', sh=sh,
                pnl=pnl, fee_fill_entry=ffe, fee_fill_close=ffc, fee_res_entry=ffe, fee_res_close=ffc, net=pnl - ffe - ffc, net_feeres=pnl - ffe - ffc,
                hold_ms=cset - eset, dust=False, res_ok=True)


XC_ROWS = [
    cl(M1, A1, 1, 'sA1', H1, 'maker', 'buy', 1_010_000, .40, 'tA1', TKH, 'taker', 1_015_000, .45, 6., .006, .06),     # c1 hold 5 s
    cl(M1, A1, 1, 'sA1', H1, 'maker', 'buy', 1_010_000, .40, 'sA4', H4, 'maker', 1_080_000, .50, 3., .003, .009),     # c2 hold 70 s
    cl(M1, A1, 1, 'sA1', H1, 'maker', 'buy', 1_010_000, .40, 'tA9', TKH, 'taker', 1_305_000, .42, 1., .001, .01),     # c3 after END
    cl(M2, A1, 1, 'sB1', H7, 'maker', 'sell', 2_106_000, .62, 'tB3', TKH, 'taker', 2_110_000, .60, 2., 0., 0.),       # c4 hold 4 s
    cl(M1, A2, 5, 'sC1', H2, 'maker', 'buy', 1_100_000, .39, 'sC2', H5, 'maker', 1_104_000, .41, 4., 0., 0.),         # c5 hold 4 s
    cl(M1, AX, 2, 'sX1', H9, 'maker', 'buy', 1_020_000, .30, 'sX2', H8, 'maker', 1_022_000, .31, 7., 0., 0.)]          # another address


def ep(mid, eid, side, end_type, rtype='', rlo=-1, rhi=-1, rtick=-1, ident='', chash='', caddr='', cgroup='', grade='', pre='', altn='', alto='', state_ms=0):
    resp = end_type == 'response'
    return dict(day=DAY, mid=mid, episode_id=eid, side=side, delta_ticks=1.0, end_type=end_type, end_reason='response' if resp else 'other_side',
                exposure_ms=50, no_frame_between=True, resp_type=rtype, resp_lo_ms=rlo, resp_hi_ms=rhi, state_status_D0='ok', state_ms_D0=state_ms,
                time_left_s_D0=(WE[mid] - state_ms) / 1000., cand_addr=caddr, cand_group=cgroup, cand_hash=chash, grade=grade, identity=ident,
                identity_reason='fixture' if ident else '', alt_netting=altn, alt_first_other=alto, alt_first_addresses=caddr, pre_state_link=pre,
                po_parity='equal' if ident else '', win_unbounded=False, win_contains=bool(chash), resp_tick=rtick, resp_size_shown=10.0 if resp else NAN)


A1l, A2l, A3l = A1.lower(), A2.lower(), A3.lower()
EP_ROWS = [
    ep(M1, 0, 'bid', 'censored_other', state_ms=1_001_000),
    ep(M1, 1, 'bid', 'response', 'restoring', 1_007_950, 1_008_000, 40, 'candidate_scenario', H1, A1l, 'rank1', 'G4_scenario_only',
       'unresolved_outside_window', 'possible', 'not_possible', 1_007_900),
    ep(M1, 2, 'ask', 'response', 'restoring', 1_070_000, 1_070_040, 50, 'candidate_constrained', H4, A1l, 'rank1', 'G2_bound_constrained',
       'episode_state_G2_conditional', 'not_possible', 'not_possible', 1_069_000),
    ep(M1, 3, 'bid', 'response', 'partial', 1_090_000, 1_090_050, 39, 'candidate_constrained', H2, A2l, 'other', 'G2_bound_constrained',
       'episode_state_G2_conditional', 'not_possible', 'not_possible', 1_089_000),
    ep(M1, 7, 'bid', 'response', 'restoring', 1_150_000, 1_150_040, 41, 'owner_unknown', '', '', '', '', 'owner_unknown', '', '', 1_149_000),
    ep(M2, 4, 'ask', 'response', 'restoring', 2_040_000, 2_040_050, 60, 'candidate_scenario', H3, A3l, 'other', 'no_regrade_row',
       'unresolved_outside_window', 'not_evaluated', 'possible', 2_039_000),
    ep(M2, 5, 'bid', 'response', 'overshoot', 2_199_950, 2_200_000, 57, 'candidate_scenario', H6, A1l, 'rank1', 'G4_scenario_only',
       'unresolved_outside_window', 'possible', 'not_possible', 2_199_000),
    ep(M2, 6, 'ask', 'response', 'restoring', 2_102_000, 2_102_050, 61, 'owner_unknown', '', '', '', '', 'owner_unknown', '', '', 2_101_900)]


def ev(mid, addr, sid, s, dq, xl, hi, o4, d0, d1=None, dp=None):
    return dict(day=DAY, mid=mid, addr=addr, sid=sid, s=s, dq=dq, x_lo_hard_ms=xl, hi_ms=hi, status='ok', o4_status=o4, dir_D0=d0, dir_D1000=d1 or d0,
                dir_pend=dp or d0)


EV_ROWS = [
    ev(M1, A1, 'tA0', -1, -1., 1_000_500, 1_001_000, 'reduces_none', 'q0'),
    ev(M1, A1, 'tA1', -1, -6., 1_014_000, 1_015_000, 'reduces_all', 'opposing'),
    ev(M1, A1, 'tA12', 1, 2., 1_040_000, 1_041_000, 'reduces_none', 'same', 'same', 'opposing'),
    ev(M1, A1, 'tA11', -1, -1., -1, 1_050_000, 'unavailable', 'q_unknown'),
    ev(M1, A1, 'tA10', -1, -1., 1_299_500, 1_300_500, 'reduces_all', 'opposing'),       # straddles END
    ev(M1, A1, 'tA9', -1, -1., 1_304_000, 1_305_000, 'reduces_all', 'opposing'),        # after END
    ev(M2, A1, 'tB3', 1, 2., 2_109_000, 2_110_000, 'reduces_all', 'opposing'),
    ev(M1, AX, 'tX1', 1, 3., 1_020_000, 1_021_000, 'reduces_none', 'same')]


def co(mid, h, addr, group, eid, n, n_ok, mq, rq, v10, v60, vend, res, tstat, tunb=-1, tend=-1, t10=-1, t60=-1):
    return dict(table='order', day=DAY, mid=f'{mid}.0', cand_hash=h, unit_sid='', cand_addr=addr.lower(), cand_group=group, n_responses_linked='1.0',
                response_keys=f'{DAY}_{mid}_{eid}', creation_hi_ms='0.0', fv_rows=f'{n}.0', fv_rows_status_ok=f'{n_ok}.0', turnover_mq_sh=repr(float(mq)),
                residual_rq_sh=repr(float(rq)), v10s_net=repr(v10), v60s_net=repr(v60), vend_net=repr(vend), res_net=repr(res), takes_status=tstat,
                takes_unbounded_after=f'{tunb}.0', takes_end=f'{tend}.0', takes_10s=f'{t10}.0', takes_60s=f'{t60}.0', scope='', metric='', value='')


CO_ROWS = [
    co(M1, H1, A1, 'rank1', 1, 3, 2, 10., 2., .9, .9, .9, 9.9, 'evaluated', 1, 4, 1, 2),
    co(M1, H4, A1, 'rank1', 2, 1, 1, 3., 0., .03 - .009, .06 - .009, .21 - .009, -1.509, 'evaluated', 0, 2, 0, 0),
    co(M1, H2, A2, 'other', 3, 1, 1, 4., 0., .04, .08, .16, 2.44, 'not_evaluated'),
    co(M2, H3, A3, 'other', 4, 1, 1, 0., 5., .05, .05, .10, 3.0, 'not_evaluated'),
    co(M2, H6, A1, 'rank1', 5, 1, 1, 0., 1., .01, .02, .03, -.55, 'evaluated', 0, 0, 0, 0),
    dict({c: '' for c in CE.CO_READ}, table='summary', scope='all', metric='n_orders', value='5')]


def chain_rows(k, epi, mid, h):
    out = []
    for it, ln in CE.CHAIN_ITEMS:
        one = ln in ('takes', 'value')
        out.append(dict(chain=k, rank=1, episode_id=epi, day=DAY, mid=mid, addr=A1l, order_hash=h, item=it, link=ln, status='filled',
                        evidence_grade='fixture_grade', census=CENSUS_TR if ln == 'transition' else '', reading='ONE-HISTORY' if one else 'TIED',
                        n_histories=1 if one else 2, alternatives='one' if one else 'alt_a|alt_b', hidden_action='', summary=f'{ln} summary',
                        post_grade='G2_bound_constrained' if ln in ('pre_state', 'transition') else '', alt_netting_state='', candidate_grades=''))
    return out


CL4_ROWS = chain_rows(1, f'{DAY}_{M1}_r1_e001', M1, H1) + chain_rows(2, f'{DAY}_{M2}_r1_e002', M2, H7)
PR_ROWS = [dict(day=DAY, mid=M1, rank=1, addr=A1, mk_hash=H1, y_side='bid', tick=40, post_lo_union_ms=1_005_000, post_hi_ms=1_009_000, grade='G2_bound_constrained'),
           dict(day=DAY, mid=M2, rank=1, addr=A1, mk_hash=H7, y_side='ask', tick=62, post_lo_union_ms=2_100_000, post_hi_ms=2_105_000, grade='G2_bound_constrained'),
           dict(day=DAY, mid=M1, rank=2, addr=AX, mk_hash=H9, y_side='bid', tick=30, post_lo_union_ms=-1, post_hi_ms=1_020_000, grade='G0_unbounded_or_invalid')]
PF_ROWS = [dict(day=DAY, mid=M1, rank=1, addr=A1, mk_hash=H1, ws_ms=WS[M1], we_ms=WE[M1], n_own_rows=3, mq_sh=10.0, rq_sh=2.0),
           dict(day=DAY, mid=M2, rank=1, addr=A1, mk_hash=H7, ws_ms=WS[M2], we_ms=WE[M2], n_own_rows=1, mq_sh=2.0, rq_sh=0.0),
           dict(day=DAY, mid=M1, rank=1, addr=A1, mk_hash=H4, ws_ms=WS[M1], we_ms=WE[M1], n_own_rows=1, mq_sh=3.0, rq_sh=0.0),
           dict(day=DAY, mid=M2, rank=1, addr=A1, mk_hash=H6, ws_ms=WS[M2], we_ms=WE[M2], n_own_rows=1, mq_sh=0.0, rq_sh=1.0)]


def chains_md_text():
    def block(k, mid, h, epi, side, tick, lv, a, b, eid, cls, closes):
        return [f'## Chain {k} - rank 1: episode {epi}', '', f'day {DAY}, market {mid}, address {A1l}; case order {h[:12]}', '',
                '- **1 PRE-STATE** — **TIED** (2 retained histories) · grade `observed_state:post_window:G2_bound_constrained` · filled',
                '  - retained histories: spread=1 | unresolved_outside_window',
                f'- **1b TRANSITION** — **TIED** (3 retained histories) · grade `transition_context` · filled · census: {CENSUS_TR}',
                '  - retained histories: at_creation | joined_shown_cell | masked_recreation_in_window',
                f'  - post cell {side} {tick}c; post window [ws+5.000s, ws+9.000s] (G2_bound_constrained); 1 compatible fresh-level creation(s)',
                f'  - level {lv} created in (ws+{a}s, ws+{b}s] (status event, context improve) - inside the post window; census pair: {cls}; response '
                f'episode {eid}: {side} retreat at ws+1.000s by 1 tick(s), response (restoring) after 100 ms; candidate grade owner_unknown (fixture; '
                'candidate none: no_candidate)',
                '- **2 POST** — **TIED** (2 retained histories) · grade `post_window:G2_bound_constrained` · filled',
                '- **6 VALUE** — **ONE-HISTORY** (1 retained history) · grade `recorded_fill_accounting` · filled',
                f'  - case order {h[:12]} own maker rows (fixture): 10s valid; {closes}, hold p50 4.0 s',
                '- **AGREEMENT** histories agree on: takes, value; TIED: pre_state; pending: none; first branching link: pre_state', '']
    return '\n'.join(['# Six complete chains v4 (fixture) - RUN', '', '| chain | rank |', '']
                     + block(1, M1, H1, f'{DAY}_{M1}_r1_e001', 'bid', 40, 5, '7.950', '8.000', 1, 'improve_bid',
                             'its closes 3 close(s) (+0 dust), 10.00 sh, pnl +0.62 $, net +0.53 $')
                     + block(2, M2, H7, f'{DAY}_{M2}_r1_e002', 'ask', 62, 9, '102.000', '102.050', 6, 'two_side/both_down',
                             'its closes 1 close(s) (+0 dust), 2.00 sh, pnl +0.04 $, net +0.04 $')) + '\n'


def _table(rows, types, dict_cols=()):
    cols = {}
    for c, t in types.items():
        a = pa.array([r[c] for r in rows], type=t)
        cols[c] = a.dictionary_encode() if c in dict_cols else a
    return pa.table(cols)


def _sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write_tree(root, dup_close=False):
    """the saved-evidence tree under root/R (receipt root) and root/D (data root), every file bound by its producer's receipt."""
    R, D = Path(root) / 'R', Path(root) / 'D'
    P = {}

    def put(base, rel):
        p = base / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        return p
    P['fv'] = put(D, 'econ/fills_valued.parquet')
    fvr = sorted(FV_ROWS, key=lambda r: (r['mid'], r['settle_ms']))
    pq.write_table(_table(fvr, FV_T, dict_cols=('day', 'role', 'addr', 'status', 'st_res', 'st_10s', 'st_60s', 'st_end')), P['fv'], row_group_size=6)
    P['xc'] = put(R, 'closes/closes.parquet')
    pq.write_table(_table(XC_ROWS + ([XC_ROWS[0]] if dup_close else []), XC_T), P['xc'])
    P['ep'] = put(R, 'chains/touch_transition/episodes.parquet')
    pq.write_table(_table(EP_ROWS, EP_T), P['ep'])
    P['ev'] = put(R, 'chains/take_logic_v2/events.parquet')
    pq.write_table(_table(EV_ROWS, EV_T), P['ev'])
    P['co'] = put(R, 'chains/touch_transition/continuations.csv')
    pd.DataFrame(CO_ROWS, columns=list(CE.CO_READ) + ['scope', 'metric', 'value']).to_csv(P['co'], index=False)
    P['cl4'] = put(R, 'chains/chains_v4/chain_links.csv')
    pd.DataFrame(CL4_ROWS, columns=list(CE.CL4_READ)).to_csv(P['cl4'], index=False)
    P['md'] = put(R, 'chains/chains_v4/chains.md')
    P['md'].write_text(chains_md_text())
    P['pr'] = put(R, 'chains/correction_v1/post_regrade_orders.csv')
    pd.DataFrame(PR_ROWS).to_csv(P['pr'], index=False)
    P['pf'] = put(R, 'chains/post_fill/orders.csv')
    pd.DataFrame(PF_ROWS).to_csv(P['pf'], index=False)
    S = {k: _sha(p) for k, p in P.items()}

    def rec(rel, obj):
        put(R, rel).write_text(json.dumps(obj, indent=1))
    rec('chains/touch_transition/receipt.json', dict(outputs={'continuations.csv': dict(sha256=S['co']), 'episodes.parquet': dict(sha256=S['ep'])},
                                                   inputs={'econ/fills_valued.parquet': S['fv'],
                                                           'predictfun/fixture/v2/chains/take_logic_v2/events.parquet': S['ev']}))
    rec('chains/chains_v4/receipt.json', dict(outputs={'chain_links.csv': dict(sha256=S['cl4']), 'chains.md': dict(sha256=S['md'])},
                                              inputs=dict(fills_valued=dict(sha256=S['fv']), closes=dict(sha256=S['xc']))))
    rec('chains/correction_v1/receipt_post.json', dict(outputs={'post_regrade_orders.csv': dict(sha256=S['pr'])}))
    rec('chains/post_fill/receipt.json', dict(outputs={'orders.csv': dict(sha256=S['pf'])}))
    rec('chains/take_logic_v2/receipt.json', dict(outputs={'events.parquet': dict(sha256=S['ev'])}, inputs={'data/fixture/econ/fills_valued.parquet': S['fv']},
                                                  addresses=[A1, AX, '0xEEEE' + '0' * 35 + '5'],
                                                  constants=dict(DIR_LAB=list(CE.DIR_LAB), O4_ST=list(CE.O4_ST))))
    rec('closes/closes_receipt.json', dict(outputs={'closes.parquet': dict(sha256=S['xc'])}, inputs=dict(fills_valued=dict(sha256=S['fv'])),
                                          addresses=[dict(addr=A1, rank=1), dict(addr=AX, rank=2), dict(addr=A2, rank=5)]))
    return R, D


# ------------------------------------------------------------------ the cases ------------------------------------------------------------------
def cases(root):
    R, D = write_tree(root)

    def run_(out, mode='dry', rr=R, dd=D):
        return CE.run(mode=mode, rroot=rr, droot=dd, out=out, require_freeze=False, selftest=True, expect=EXP, git_probe=False, log=quiet)
    OUT = root / 'out_run'
    res = run_(OUT, mode='run')
    O, P, L = res['orders'], res['pooled'], {x['link_id']: x for x in res['links']}
    acc = res['claims']['acceptance']
    # ---- T1 the FIFO toy ----
    o1 = O[(DAY, M1, H1)]
    check('T1 population: 5 response links, 2 chains, 6 distinct orders', len([x for x in res['links'] if x['kind'] == 'response']) == 5
          and len([x for x in res['links'] if x['kind'] == 'chain']) == 2 and len(O) == 6)
    check('T1 H1 le10s = c1', cs_eq(o1['csum']['le10s'], 1, 6., .30, .066, .234), o1['csum']['le10s'])
    check('T1 H1 le60s = c1 (c2 holds 70 s)', cs_eq(o1['csum']['le60s'], 1, 6., .30, .066, .234), o1['csum']['le60s'])
    check('T1 H1 leEND = c1 + c2', cs_eq(o1['csum']['leEND'], 2, 9., .60, .078, .522), o1['csum']['leEND'])
    check('T1 H1 afterEND (the later close) = c3', cs_eq(o1['csum']['afterEND'], 1, 1., .02, .011, .009), o1['csum']['afterEND'])
    check('T1 H1 leEND by closing leg', near(o1['by_role']['taker']['sh'], 6.) and near(o1['by_role']['taker']['net'], .234)
          and near(o1['by_role']['maker']['sh'], 3.) and near(o1['by_role']['maker']['net'], .288), o1['by_role'])
    check('T1 H1 residual kept apart', near(o1['resid']['sh'], 2.) and near(o1['resid']['val_end'], .06) and near(o1['resid']['val_res'], 1.2)
          and near(o1['resid']['fee'], .002) and near(o1['resid']['net_res'], 1.198), o1['resid'])
    check('T1 H1 exact link available, attribution = mq, fees decompose', o1['exact'] == 'available' and o1['attribution_ok'] and o1['fees_ok']
          and near(o1['attributed_sh'], 10.))
    o4 = O[(DAY, M1, H4)]
    check('T1 H4 closing-earlier kept apart (never its completed close)', cs_eq(o4['esum'], 1, 3., .30, .012, .288)
          and all(not o4['close_keys'][b] for b in CE.BUCKETS) and o4['exact'] == 'available', o4['esum'])
    o2, o3, o6, o7 = O[(DAY, M1, H2)], O[(DAY, M2, H3)], O[(DAY, M2, H6)], O[(DAY, M2, H7)]
    check('T1 H2 (rank 5) le10s = c5', cs_eq(o2['csum']['le10s'], 1, 4., .08, 0., .08) and o2['exact'] == 'available', o2['csum'])
    check('T1 H3 unavailable (outside the closes population), residual kept', o3['exact'] == 'unavailable'
          and o3['exact_reason'] == 'address_outside_closes_population' and near(o3['resid']['sh'], 5.) and near(o3['resid']['val_end'], .10)
          and near(o3['resid']['val_res'], 3.0), (o3['exact'], o3['exact_reason'], o3['resid']))
    check('T1 H6 no close, residual 1 sh', o6['exact'] == 'available' and near(o6['resid']['sh'], 1.) and near(o6['resid']['val_end'], .03)
          and near(o6['resid']['val_res'], -.55), o6['resid'])
    check('T1 H7 le10s = c4', cs_eq(o7['csum']['le10s'], 1, 2., .04, 0., .04), o7['csum'])
    pr = P['response_all']
    check('T1 pooled response_all', cs_eq(pr['closes']['le10s'], 2, 10., .38, .066, .314) and cs_eq(pr['closes']['le60s'], 2, 10., .38, .066, .314)
          and cs_eq(pr['closes']['leEND'], 3, 13., .68, .078, .602) and cs_eq(pr['closes']['afterEND'], 1, 1., .02, .011, .009)
          and cs_eq(pr['earlier'], 1, 3., .30, .012, .288) and near(pr['residual_available']['sh'], 3.) and near(pr['residual_available']['val_end'], .09)
          and near(pr['residual_available']['val_res'], .65) and near(pr['residual_unavailable']['sh'], 5.) and pr['n_available'] == 4
          and pr['n_unavailable'] == 1, {k: pr[k] for k in ('closes', 'earlier', 'residual_available', 'residual_unavailable')})
    pt, po = P['response_top3'], P['response_other']
    check('T1 pooled top3 / other split', pt['n_orders'] == 3 and po['n_orders'] == 2 and cs_eq(pt['closes']['leEND'], 2, 9., .60, .078, .522)
          and cs_eq(po['closes']['leEND'], 1, 4., .08, 0., .08))
    pc = P['chains']
    check('T1 pooled chains', pc['n_orders'] == 2 and cs_eq(pc['closes']['le10s'], 2, 8., .34, .066, .274) and cs_eq(pc['closes']['leEND'], 3, 11., .64, .078, .562)
          and cs_eq(pc['closes']['afterEND'], 1, 1., .02, .011, .009) and pc['earlier']['n'] == 0, pc['closes'])
    failed = [k for k, v in acc.items() if not v['holds']]
    check('T1 every acceptance predicate and verifier check holds', res['claims']['acceptance_all_hold'] and not failed and not res['violations'],
          (failed, res['violations']))
    check('T1 diagnostics: taker closes are recorded takes', res['claims']['diagnostics']['taker_closes_are_recorded_takes']['holds']
          and res['claims']['diagnostics']['taker_closes_are_recorded_takes']['n'] == 3)
    names = sorted(p.name for p in OUT.iterdir())
    check('T1 the five files written', names == sorted(CE.OUT_FILES), names)
    rc = json.loads((OUT / 'receipt.json').read_text())
    check('T1 receipt sha256 = the written bytes', all(_sha(OUT / n) == rc['outputs'][n]['sha256'] for n in CE.OUT_FILES if n != 'receipt.json'))
    lk, od = pd.read_csv(OUT / 'links_econ.csv'), pd.read_csv(OUT / 'orders_econ.csv')
    check('T1 links_econ / orders_econ shapes', list(lk.columns) == list(CE.LINK_COLS) and len(lk) == 7 and list(od.columns) == list(CE.ORDER_COLS)
          and len(od) == 6, (lk.shape, od.shape))
    cj = json.loads((OUT / 'claims.json').read_text())
    md = (OUT / 'chains_econ.md').read_text()
    check('T1 claims.json / chains_econ.md readable', cj['acceptance_all_hold'] is True and '## Chain 1 ' in md and '## Chain 2 ' in md)
    check('T1 chains_v4 its-closes reconcile and chain parse counts', acc['chain_its_closes_reconcile']['holds'] and acc['chain_its_closes_reconcile']['n'] == 2
          and acc['chain_responses_equal_census']['holds'] and acc['chain_context_intervals_match_episodes']['n'] == 2)
    # ---- T2 a take crossing END ----
    e1 = L[f'resp:{DAY}_{M1}_1']
    T = e1['T']
    check('T2 take statuses (t10 straddles END, t9 after END)', T['status'] == {'after_END': 1, 'before_anchor': 1, 'inside_END': 2, 'straddles_END': 1,
                                                                                 'unbounded_lower': 1}, T['status'])
    check('T2 windows exclude the END-crossing takes', T['10s']['n'] == 1 and T['60s']['n'] == 2 and T['END']['n'] == 2
          and {k[3] for k in T['keys']['END']} == {'tA1', 'tA12'})
    check('T2 the saved unbounded counts reproduced (4 subsequent, 2 inside END)', e1['saved']['saved_takes_reproduced'] and e1['saved']['saved_takes_end'] == 4
          and acc['saved_take_counts_reproduced']['holds'] and acc['saved_take_counts_reproduced']['n'] == 3)
    check('T2 a subsequent take is not automatically an exit', T['END']['fifo_closing'] == 1 and near(T['END']['fifo_closing_sh'], 6.)
          and T['END']['dir_D0'] == {'opposing': 1, 'same': 1} and T['END']['clock_disagree'] == 1, T['END'])
    check('T2 verifier holds on the production build', res['checks']['takes_within_end_bound'] and res['checks']['closes_within_end_bound'])
    m1 = with_mut('no_end_bound', lambda: run_(root / 'o_m1'))
    check('T2 mutant no_end_bound caught', m1['checks']['takes_within_end_bound'] is False and m1['checks']['closes_within_end_bound'] is False, m1['checks'])
    expect_refusal('T2 a run with the mutant refuses naming the check', lambda: with_mut('no_end_bound', lambda: run_(root / 'o_m1r', mode='run')),
                   'takes_within_end_bound', 'refused before any write')
    # ---- T3 a duplicated close across two links ----
    check('T3 H1 carries two links, one order row', len(o1['links']) == 2 and res['claims']['guards']['dedup']['orders_with_repeated_links'] == 1)
    check('T3 union counts each close once', cs_eq(P['union']['closes']['leEND'], 4, 15., .72, .078, .642) and P['union']['shared_close_keys'] == 1
          and P['union']['touching_distinct'] == 5 and res['checks']['pooled_closes_distinct'], P['union']['closes']['leEND'])
    check('T3 pooled takes distinct (one take after two linked creations counts once)', P['union']['takes']['windows']['END']['n_distinct'] == 3
          and P['union']['takes']['windows']['END']['overlap'] == 2 and res['checks']['pooled_takes_distinct'])
    dg = res['digest']['scopes']
    check('T3 the printed reconciliation carries the distinct-key take windows and the unavailable reasons',
          dg['union']['takes']['windows']['END'] == dict(n=3, order_sum=5, abs_dq=10.0, dir_D0={'opposing': 2, 'same': 1}, fifo_close_keys=2, fifo_sh=8.0)
          and dg['response_all']['unavailable_reasons'] == {'address_outside_closes_population': 1} and dg['response_all']['closes']['afterEND']['n'] == 1
          and dg['union']['takes']['end_status'] == {'after_END': 1, 'inside_END': 3, 'straddles_END': 1}, (dg['union']['takes'], dg['response_all']))
    m2 = with_mut('no_close_dedup', lambda: run_(root / 'o_m2'))
    check('T3 mutant no_close_dedup caught', m2['checks']['pooled_closes_distinct'] is False, m2['checks'])
    R2, D2 = write_tree(root / 'dup', dup_close=True)
    expect_refusal('T3 a duplicated closes.parquet row refused', lambda: run_(root / 'o_dup', rr=R2, dd=D2), 'a close key twice', 'duplicated close')
    # ---- T4 a self-trade / non-ok leg entering value ----
    check('T4 the excluded legs are counted, never valued', o1['valued'] == [(DAY, M1, 'sA1', 0, 'maker')] and o1['n_excl_self'] == 1
          and o1['n_excl_status'] == 1 and o1['excl_statuses'] == {'record_mismatch': 1} and near(o1['clean_sh'], 12.), o1['valued'])
    check('T4 verifier holds on the production build', res['checks']['valued_legs_clean'])
    m3 = with_mut('no_cohort', lambda: run_(root / 'o_m3'))
    check('T4 mutant no_cohort caught', m3['checks']['valued_legs_clean'] is False, m3['checks'])
    # ---- T5 a same-frame deep creation inheriting restoring context ----
    c1l, c2l, e5 = L['chain:1'], L['chain:2'], L[f'resp:{DAY}_{M2}_5']
    check('T5 deep creation -> contextual_only (tick_mismatch)', c2l['ctx_status'] == 'contextual_only' and c2l['contexts'][0]['reasons'] == ['tick_mismatch']
          and e5['ctx_status'] == 'contextual_only' and e5['ctx_reasons'] == ['tick_mismatch'], (c2l['contexts'], e5['ctx_reasons']))
    check('T5 a tick-equal restoring cell is kept', c1l['ctx_status'] == 'restoring_cell' and e1['ctx_status'] == 'restoring_cell'
          and L[f'resp:{DAY}_{M2}_4']['ctx_status'] == 'restoring_cell')
    check('T5 verifier holds on the production build', res['checks']['restoring_context_tick_equal'])
    m4 = with_mut('no_tick_guard', lambda: run_(root / 'o_m4'))
    check('T5 mutant no_tick_guard caught', m4['checks']['restoring_context_tick_equal'] is False, m4['checks'])
    # ---- T6 write-once and production refusals ----
    expect_refusal('T6 overwrite refused with its message', lambda: run_(OUT, mode='run'), 'write-once', 'exists (refused before any input is read)')
    od_ = root / 'o_dry'
    run_(od_)
    check('T6 run --dry writes nothing', not od_.exists() and not [p for p in root.iterdir() if p.name.startswith('.o_dry')])
    expect_refusal('T6 a mutant outside the selftest refused', lambda: with_mut('no_cohort', lambda: CE.run(mode='dry', rroot=R, droot=D, out=root / 'o_x', log=quiet)),
                   'a planted mutant never runs outside the selftest')
    expect_refusal('T6 require_freeze=False outside the selftest refused',
                   lambda: CE.run(mode='dry', rroot=R, droot=D, out=root / 'o_y', require_freeze=False, log=quiet), 'are the selftest only')
    expect_refusal('T6 an unknown mutant refused', lambda: with_mut('bogus', lambda: run_(root / 'o_z')), 'is not a declared mutant')
    # ---- T7 instrument falsifiers ----
    pm = CE.parse_chains_md('## Chain 1 - rank 1: episode e\n- **1b TRANSITION** — x\n  - level 5 created in (ws+1.000s, ws+1.050s] (status event) - inside '
                            'the post window; census pair: improve_bid; response episode 4: bid retreat at ws+0.900s\n  - level 6 created in (garbled\n'
                            '- **2 POST** — y\n  - level 7 created in (ws+2.000s, ws+2.050s] (status event) - inside the post window; census pair: x; '
                            'no_response_episode\n')
    cr_ = pm.get(1, {}).get('creations', [])
    check('T7 parser: a good line parsed, a malformed line flagged, lines outside the block ignored', len(cr_) == 2 and cr_[0]['parsed'] and cr_[0]['eid'] == '4'
          and not cr_[1]['parsed'], cr_)
    mdp = R / 'chains/chains_v4/chains.md'
    mdp.write_text(mdp.read_text() + '\n')
    rb = run_(root / 'o_b')
    check('T7 an input changed after its receipt fails inputs_bound', rb['claims']['bindings']['chains_md'] == 'mismatch'
          and not rb['claims']['acceptance']['inputs_bound']['holds'], rb['claims']['bindings'])
    expect_refusal('T7 a run refuses the unbound input', lambda: run_(root / 'o_b2', mode='run'), 'inputs_bound', 'refused before any write')


def main():
    if CE.C.RUN != 'v2':
        print(f'FAIL MT_RUN must be v2 (is {CE.C.RUN!r})')
        return 1
    os.environ.pop(CE.MUT_ENV, None)
    with tempfile.TemporaryDirectory(prefix='ce_selftest_') as td:
        try:
            cases(Path(td))
        except Exception:                                         # a crash is a failure, never a pass
            traceback.print_exc()
            check('no crash', False, 'exception (traceback above)')
    n_ok = sum(1 for _, ok in RESULTS if ok)
    print(f'continuation_econ_selftest: {n_ok}/{len(RESULTS)} passed', flush=True)
    return 0 if RESULTS and n_ok == len(RESULTS) else 1


if __name__ == '__main__':
    sys.exit(main())
