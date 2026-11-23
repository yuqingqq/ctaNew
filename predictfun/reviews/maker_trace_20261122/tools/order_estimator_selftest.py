"""SELFTEST of tools/order_estimator.py (contract order_estimator_contract.md section 9; P-2026-005 MAKER-TRACE, 2026-11-23). SYNTHETIC only: no data
file is read; synthetic markets go through the frozen pf_mt_lifecycle.run_market and its book (_book) exactly as the run does, BTC through
pf_mt_decision.btc_feats on a synthetic grid. A failed check prints FAIL with its values; the process exits 1. Cases (contract 9 numbering):
 1  identical public tape: A = a 5-share order exhausted + another maker's 45 behind it; B = a 50-share order filled 5, the other 45 cancelled
    unseen while the other maker posts 45 in the same frame interval. Identical tapes -> identical lifecycle and estimator outputs; EX and PL
    feasible (tied); P_cmd = EX 5 sh (prior-driven); the PL alternative's size range holds B's true 50
 2  positive controls: a sweep through the price (S_FIFO rule 1: EX only under priority; S_NOPRIO tied -> distinguished_under_priority, C);
    a capacity-tight two-fill order (UB = cum in every scenario -> EX distinguished, B); its later fill proves a remainder after fill 1 (A)
 3  repost: P pulled then Q new at the same price vs Q resting since before and first filled later: identical tapes -> link tied; positive:
    a shared record (overlap_identified) and a posting bound after P's removal bound (sequential_identified); one hash = one order
 4  overlapping orders cannot reuse capacity: the P_nomask joint rule (forced claims; the greedy representative never assigns one addition
    in full to two orders)
 5  gaps: a posting across a frame gap gives no P_nomask candidate (incomplete_gap) and no certified front; a removal across a gap keeps
    missing_evidence and the pull unidentified
 6  same-ms ties: a take sharing its settle ms with an own fill -> simultaneous_action, both evaluations, timing_sensitive
 7  receipt earlier than the UUID: flagged, refused from every precision bin (refused_clock), counted, row kept; a take loses its certain variant
 8  inventory classes by actual |position| (overshoot, reversal, full close, partial, add, from flat)
 9  perturbation: BTC / book / legs changed after a decision window leave its decision features (state_change) unchanged; changed inside
    the window they change them (falsifier)
 10 the sample: deterministic, row-order independent, market-first round-robin, <= 4, statuses, key columns only
 11 refusals with reasons: a forbidden day, write-once, a wrong population, a fills_valued byte / hash mismatch, two spellings, a day mismatch
 +  units: priors, clip, st_all, width bins, link_hypotheses under both caller spellings, episodes, the own-order scorer (positive + falsifier),
    a take end to end (overshoot -> inventory_unwind contradicted), the logic statuses.
Usage (run_light, PF_SLOT=claude4): MT_RUN=v2 order_estimator_selftest.py"""
import hashlib
import json
import os
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RS = Path('/home/yuqing/ctaNew/predictfun/research')
for _p in (str(RS), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import pf_mt_common as C  # noqa: E402
import pf_mt_lifecycle as LC  # noqa: E402
import pf_mt_lifecycle_selftest as LS  # noqa: E402
import order_estimator as OE  # noqa: E402

T0, DAY = LS.T0, LS.DAY
A1, A2, A3 = OE.TOP3
L1, L2, L3 = OE.LOW3
OTHER = '0x00000000000000000000000000000000000000Bb'
TK = '0x00000000000000000000000000000000000000Cc'
NAN = float('nan')
RES = []


def H(tag):
    return '0x' + hashlib.sha256(tag.encode()).hexdigest()


def ok(name, cond, info=''):
    RES.append((name, bool(cond)))
    if not cond:
        print(f'FAIL {name}: {info}')


def refuses(name, word, fn, *a, **kw):
    try:
        fn(*a, **kw)
    except Exception as e:                 # the reason must be the declared one (an earlier guard must not pass for it)
        ok(name, word in str(e), f'raised {type(e).__name__}: {e}')
        return
    ok(name, False, 'did not refuse')


# ------------------------------------------------------------------ synthetic worlds ------------------------------------------------------------------
def world(mid, orders, recs, times=LS.GRID, recv_shift=None):
    """orders: hidden histories {(side, tick): [[(t_rel, size), ...], ...]}; recs: [(settle_rel, [(hash, signer, side, tick, q)], taker)]."""
    m = LS.Mk(mid)
    for settle, makers, tk in recs:
        m.rec(settle, makers, tk=tk)
    m.frames(times, LS.sim(orders))
    legs, records, frames, levels = m.dfs()
    if recv_shift:                                   # {record index: receipt ms relative to T0} (the receipt-earlier-than-UUID case)
        for i, rel in recv_shift.items():
            sid = records.sid.iloc[i]
            ns = (T0 + rel) * 10 ** 6
            records.loc[records.sid == sid, 'first_recv_ns'] = ns
            legs.loc[legs.sid == sid, 'first_recv_ns'] = ns
    fills, ords, links, summ = LC.run_market(legs, records, frames, levels, m.market())
    return dict(m=m, legs=legs, records=records, frames=frames, levels=levels, fills=fills, orders=ords, links=links, summ=summ)


def fv_from(legs):
    rows = []
    for r in legs.itertuples(index=False):
        s = 1 if r.y_side == 'bid' else -1
        for role, addr, h, sg in (('maker', r.mk_signer, r.mk_hash, s), ('taker', r.tk_signer, r.tk_hash, -s)):
            if str(addr).lower() not in OE.LOW3:
                continue
            rows.append(dict(day=r.day, mid=r.mid, sid=r.sid, mk_idx=r.mk_idx, px_rank=0, role=role, addr=addr, hash=h, y_side='bid' if sg > 0 else 'ask',
                             s=sg, q=r.shares, p=r.y_price, self_trade=str(r.mk_signer).lower() == str(r.tk_signer).lower(), settle_ms=r.settle_ms,
                             m_end=0.5, v_end=sg * r.shares * (0.5 - r.y_price), fee_fill=0., net_res=0., mq=0., rq=r.shares, status='ok'))
    df = pd.DataFrame(rows, columns=list(OE.FV_COLS))
    df['addr_l'] = df.addr.str.lower()
    return df


def bundle(W):
    m = W['m']; mk = m.market()
    ws, we = int(mk['ws']) * 1000, int(mk['we']) * 1000
    B = LC._book(W['frames'], W['levels'], m.mid, ws, we)
    fv = fv_from(W['legs'])
    Wn = OE.windows_of(W['fills'])
    mkr = fv.role == 'maker'
    settle = dict(zip(zip(fv.sid[mkr], fv.hash[mkr]), fv.settle_ms[mkr].astype(int)))
    Mk = dict(day=DAY, mid=m.mid, ws_ms=ws, we_ms=we, orders=W['orders'], fills=W['fills'], links=W['links'], B=B, touch=OE.touch_arrays(B),
              FF=OE.fill_frames(W['fills'], B), settle=settle, L={}, inv={})
    for a in sorted(set(fv.addr_l)):
        Lg = OE.legs_frame(fv[fv.addr_l == a], Wn)
        Mk['L'][a], Mk['inv'][a] = Lg, OE.Inventory(Lg)
    return Mk, fv


def cases(mid, rows):
    return pd.DataFrame([dict(kind=k, day=DAY, mid=mid, addr=a, key=key, rank=OE.LOW3.index(a) + 1, pick=i, round=0, episode_id=e)
                         for i, (k, key, a, e) in enumerate(rows)])


def bt_from_mid(g0_ms, mid):
    """a BTC grid in pf_mt_decision.load_btc's array form."""
    n = len(mid); G0 = g0_ms // 100
    okk = np.isfinite(mid) & (mid > 0)
    lm = np.full(n, np.nan); lm[okk] = np.log(mid[okk])
    ret = np.zeros(n); ret[1:] = 1e4 * (lm[1:] - lm[:-1]); ret[~np.isfinite(ret)] = 0.
    csq = np.zeros(n + 1); csq[1:] = np.cumsum(ret * ret)
    okc = np.zeros(n + 1, np.int64); okc[1:] = np.cumsum(okk)
    return dict(G0=G0, n=n, ok=okk, lm=lm, csq=csq, okc=okc)


G0_MS = T0 - 400_000
BTC_MID = 100_000. * np.exp(np.cumsum(np.random.default_rng(7).normal(0., 2e-5, 4_400)))


def ctx_of(mid_series=BTC_MID):
    BT = bt_from_mid(G0_MS, mid_series)
    return OE.LocalContext(btc_override=lambda d: BT)


def frame_eq(a, b):
    a, b = pd.DataFrame(a), pd.DataFrame(b)
    return a.shape == b.shape and a.astype(str).equals(b.astype(str))


def core_of(W, h, rank_addr=None):
    R3 = OE.rows3_of(W['orders'])
    fl = W['fills'][W['fills'].mk_hash == h]
    fv = fv_from(W['legs'])
    settle = {s: int(t) for s, hh, t, r in zip(fv.sid, fv.hash, fv.settle_ms, fv.role) if r == 'maker' and hh == h}
    return OE.order_core(R3[h], fl, settle=settle)


def main():  # noqa: C901
    t0 = time.process_time()
    ctx = ctx_of()
    # ================= 1: the identical public tape =================
    hM, hN = H('hM'), H('hN')
    WA = world(101, {('bid', 50): [[(1050, 5.), (4060, 0.)], [(1070, 45.), (7060, 35.)]]},
               [(4060, [(hM, A1, 'bid', 50, 5.)], TK), (7060, [(hN, OTHER, 'bid', 50, 10.)], TK)])
    WB = world(101, {('bid', 50): [[(1050, 50.), (4060, 45.), (4070, 0.)], [(4080, 45.), (7060, 35.)]]},
               [(4060, [(hM, A1, 'bid', 50, 5.)], TK), (7060, [(hN, OTHER, 'bid', 50, 10.)], TK)])
    ok('1_tapes_identical', all(frame_eq(WA[k], WB[k]) for k in ('legs', 'records', 'frames', 'levels')))
    ok('1_lifecycle_identical', all(frame_eq(WA[k], WB[k]) for k in ('fills', 'orders', 'links')))
    outs = {}
    for nm, W in (('A', WA), ('B', WB)):
        Mk, _ = bundle(W)
        outs[nm] = OE.market_cases(Mk, cases(101, [('order', hM, L1, 'e1')]), ctx, Counter())
    ok('1_estimator_identical', all(frame_eq(outs['A'][i], outs['B'][i]) for i in range(6)))
    crow, ledger = pd.DataFrame(outs['A'][0]), pd.DataFrame(outs['A'][1])
    r = crow.iloc[0]
    ok('1_tied', r.size_status == 'tied' and {'EX', 'PL'} <= set(r.union.split('|')), f'{r.size_status} {r.union}')
    ok('1_P_cmd_is_5sh_EX', r.P_cmd == ('EX', 5.0), f'{r.P_cmd}')
    pl = ledger[(ledger.row_kind == 'size_end') & (ledger.hypothesis == 'PL')].iloc[0]
    ok('1_alternative_50_feasible', bool(pl.feasible_union) and pl.size_lo <= 50 and (np.isnan(pl.size_hi) or pl.size_hi >= 50), f'{pl.to_dict()}')
    ok('1_prior_labelled', r.grade == 'E_unresolved' and not bool(r.prior_sensitive), f'{r.grade} {r.prior_sensitive} {r.P_drop}')
    lgA = pd.DataFrame(outs['A'][2])
    end = lgA[lgA.link == 'END']
    ok('1_pull_unidentified', len(end) == 2 and (end.identification_status == 'unidentified').all(), f'{end[["mechanism", "identification_status"]].values}')
    ok('1_logic_statuses_valid', lgA.identification_status.isin(OE.STATUSES).all() and lgA.conditional_status.isin(OE.STATUSES + ('',)).all())
    # ================= 2: positive controls =================
    hS, hX = H('hS'), H('hX')
    WS = world(102, {('bid', 50): [[(1050, 5.), (4060, 0.)]], ('bid', 49): [[(1060, 20.), (4060, 10.)]]},
               [(4060, [(hX, OTHER, 'bid', 49, 10.), (hS, A1, 'bid', 50, 5.)], TK)])
    cS = core_of(WS, hS)
    ok('2a_sweep_distinguished_under_priority', cS['size_status'] == 'distinguished_under_priority' and cS['feas']['S_FIFO'] == {'EX'}
       and cS['feas']['S_FIFO_LO'] == {'EX'} and 'PL' in cS['feas']['S_NOPRIO'] and cS['grade'] == 'C_priority', f"{cS['size_status']} {cS['feas']}")
    hP = H('hP')
    WC = world(103, {('bid', 50): [[(1050, 10.), (3060, 6.), (6060, 0.)]]},
               [(3060, [(hP, A1, 'bid', 50, 4.)], TK), (6060, [(hP, A1, 'bid', 50, 6.)], TK)])
    cP = core_of(WC, hP)
    ok('2b_capacity_tight_distinguished', cP['size_status'] == 'distinguished' and cP['union'] == {'EX'} and cP['grade'] == 'B_constrained'
       and all(abs(cP['ub'][s] - 10.) < 1e-9 for s in OE.SCEN), f"{cP['size_status']} {cP['union']} {cP['ub']}")
    ok('2c_later_fill_proves_remainder', abs(cP['rem_lo'][0] - 6.) < 1e-9 and cP['n'] == 2, f"{cP['rem_lo']}")
    Mk, _ = bundle(WC)
    o2 = OE.market_cases(Mk, cases(103, [('order', hP, L1, 'e1')]), ctx, Counter())
    lg2 = pd.DataFrame(o2[2])
    kp = lg2[lg2.link == 'KEEP']
    ok('2c_keep_observed', len(kp) == 1 and kp.identification_status.iloc[0] == 'compatible' and kp.evidence_grade.iloc[0] == 'A_observed',
       f'{kp[["identification_status", "evidence_grade"]].values}')
    ok('2b_end_exhaustion_row', (lg2[lg2.link == 'END'].mechanism == 'exhaustion').all() and len(lg2[lg2.link == 'END']) == 1)
    ok('2b_one_hash_one_order', len(OE.rows3_of(WC['orders'])) == 1 and cP['n'] == 2)
    # ================= 3: repost =================
    hP2, hQ2 = H('hP2'), H('hQ2')
    recs3 = [(3060, [(hP2, A1, 'bid', 50, 4.)], TK), (6060, [(hQ2, A1, 'bid', 50, 6.)], TK)]
    WR1 = world(104, {('bid', 50): [[(1050, 10.), (3060, 6.), (3510, 0.)], [(3520, 6.), (6060, 0.)]]}, recs3)
    WR2 = world(104, {('bid', 50): [[(1050, 4.), (3060, 0.)], [(1060, 6.), (6060, 0.)]]}, recs3)
    ok('3_tapes_identical', all(frame_eq(WR1[k], WR2[k]) for k in ('legs', 'records', 'frames', 'levels', 'fills', 'orders', 'links')))
    lk = WR1['links']
    ok('3_refill_link_present', ((lk.prev_hash == hP2) & (lk.next_hash == hQ2) & (lk.kind == 'refill')).sum() == 1, f'{lk.to_dict("records")}')
    res = [OE.link_hypotheses(core_of(W, hP2), core_of(W, hQ2)) for W in (WR1, WR2)]
    nz = lambda d: {k: (None if isinstance(v, float) and np.isnan(v) else v) for k, v in d.items()}  # noqa: E731
    ok('3_tied_both_worlds', nz(res[0]) == nz(res[1]), f'{res}')
    ok('3_tied', res[0]['status'] == 'tied' and set(res[0]['feasible']) == {'overlap', 'replace', 'replenish', 'unrelated'}, f'{res[0]}')
    WR4 = world(105, {('bid', 50): [[(1050, 4.), (3060, 0.)], [(5050, 6.), (6060, 0.)]]}, [(3060, [(hP2, A1, 'bid', 50, 4.)], TK),
                                                                                         (6060, [(hQ2, A1, 'bid', 50, 6.)], TK)])
    r4 = OE.link_hypotheses(core_of(WR4, hP2), core_of(WR4, hQ2))
    ok('3_sequential_identified', r4['status'] == 'sequential_identified' and 'overlap' not in r4['feasible'], f'{r4}')
    hP5, hQ5 = H('hP5'), H('hQ5')
    WR5 = world(106, {('bid', 50): [[(1050, 4.), (3060, 0.)], [(1060, 5.), (3060, 3.), (6060, 0.)]]},
                [(3060, [(hP5, A1, 'bid', 50, 4.), (hQ5, A1, 'bid', 50, 2.)], TK), (6060, [(hQ5, A1, 'bid', 50, 3.)], TK)])
    sid_p = set(WR5['fills'][WR5['fills'].mk_hash == hP5].sid); sid_q = set(WR5['fills'][WR5['fills'].mk_hash == hQ5].sid)
    r5 = OE.link_hypotheses(core_of(WR5, hP5), core_of(WR5, hQ5), shared_record=bool(sid_p & sid_q))
    ok('3_overlap_identified_shared_record', r5['status'] == 'overlap_identified' and bool(sid_p & sid_q), f'{r5}')
    cQ5 = core_of(WR5, hQ5)
    Mk5, _ = bundle(WR5)
    o5 = pd.DataFrame(OE.market_cases(Mk5, cases(106, [('order', hQ5, L1, 'e1')]), ctx, Counter())[0])
    ok('3_survivor_behind_at_first_fill', o5.front_class.iloc[0] == 'behind_at_first_fill' and abs(cQ5['ahead_lo_fifo'] - 4.) < 1e-9,
       f'{o5.front_class.iloc[0]} {cQ5["ahead_lo_fifo"]}')
    # chain_take spelling of the same pair: the one definition answers identically
    def ct(c):
        return dict(post_lo_union=c['post_lo_u'], post_hi=c['post_hi'], first_hi=c['first_hi'], last_lh=c['last_lo_hard'], removal_lo=c['removal_lo'],
                    end_hi=c['end_hi'], tick=c['tick'], ex_any='EX' in c['union'], pl_any='PL' in c['union'], ec_any='EC' in c['union'])
    r4b = OE.link_hypotheses(ct(core_of(WR4, hP2)), ct(core_of(WR4, hQ2)))
    ok('3_link_spellings_agree', r4b['status'] == r4['status'] and r4b['feasible'] == r4['feasible'], f'{r4b} vs {r4}')
    # ================= 4: overlapping orders cannot reuse capacity =================
    Bs = dict(F=6, t=np.arange(0, 600, 100, dtype=np.int64), gap=np.zeros(6, bool))
    CA = dict(add=np.array([0., 12., 10., 0., 0.]), bad=np.zeros(5, bool), tol=np.full(5, 0.01 + 1e-9))
    mk_c = lambda h, lo, fh: dict(hash=h, cum=10., post_lo_u=lo, post_hi=500, first_hi=fh, conflict=False)  # noqa: E731
    NMj = OE.cell_nomask([mk_c('x', 0, 500), mk_c('y', 0, 501)], Bs, CA, 0)
    ok('4_greedy_distinct_intervals', NMj['x']['rep'] == 2 and NMj['y']['rep'] == 1, f"{ {h: (v['rep'], list(v['cand'])) for h, v in NMj.items()} }")
    NMz = OE.cell_nomask([mk_c('x', 0, 500), mk_c('y', 0, 501), mk_c('z', 250, 502)], Bs, CA, 0)
    ok('4_forced_claim_removes_candidate', list(NMz['x']['cand']) == [1] and list(NMz['y']['cand']) == [1] and list(NMz['z']['cand']) == [2],
       f"{ {h: list(v['cand']) for h, v in NMz.items()} }")
    ok('4_never_overassigned', NMz['x']['rep'] == 1 and NMz['y']['status'] == 'joint_unassignable' and NMz['z']['rep'] == 2,
       f"{ {h: (v['rep'], v['status']) for h, v in NMz.items()} }")
    # ================= 5: gaps =================
    hG = H('hG')
    gap_times = [x for x in LS.GRID if not (2100 <= x <= 3300)]
    WG = world(107, {('bid', 50): [[(2500, 8.), (5060, 0.)]]}, [(5060, [(hG, A1, 'bid', 50, 8.)], TK)], times=gap_times)
    MkG, _ = bundle(WG)
    oG = pd.DataFrame(OE.market_cases(MkG, cases(107, [('order', hG, L1, 'e1')]), ctx, Counter())[0]).iloc[0]
    ok('5_posting_across_gap_no_candidate', oG.nomask_status == 'incomplete_gap' and 'gap_in_window' in oG.status and 'incomplete_gap' in oG.status,
       f'{oG.nomask_status} {oG.status}')
    ok('5_no_certified_front', oG.front_class in ('front_at_acceptance_or_reached', 'unidentified'), f'{oG.front_class}')
    hR = H('hR')
    gap2 = [x for x in LS.GRID if not (2500 <= x <= 4000)]
    WG2 = world(108, {('bid', 50): [[(1050, 10.), (2060, 6.), (3000, 0.)]]}, [(2060, [(hR, A1, 'bid', 50, 4.)], TK)], times=gap2)
    cR = core_of(WG2, hR)
    ok('5_removal_across_gap_missing_evidence', 'missing_evidence' in cR['removal_set']['S_NOPRIO'] and {'EX', 'PL'} <= cR['union'],
       f"{cR['removal_set']} {cR['union']}")
    ok('5_true_cancel_inside_PL_window', cR['removal_lo'] <= T0 + 3000 and (cR['pl_hi'] < 0 or cR['pl_hi'] >= T0 + 3000), f"{cR['removal_lo'] - T0} {cR['pl_hi'] - T0}")
    MkG2, _ = bundle(WG2)
    lgG2 = pd.DataFrame(OE.market_cases(MkG2, cases(108, [('order', hR, L1, 'e1')]), ctx, Counter())[2])
    ok('5_pull_unidentified', (lgG2[lgG2.link == 'END'].identification_status == 'unidentified').all() and len(lgG2[lgG2.link == 'END']) == 2)
    # ================= 6: same-ms tie =================
    sids = [f's{i}' for i in range(4)]
    fvt = pd.DataFrame([dict(day=DAY, mid=1, sid=sids[0], mk_idx=0, px_rank=0, role='maker', addr=A1, hash='h1', y_side='bid', s=1, q=5., p=.5,
                             self_trade=False, settle_ms=19_000, m_end=.5, v_end=0., fee_fill=0., net_res=0., mq=0., rq=0., status='ok'),
                        dict(day=DAY, mid=1, sid=sids[1], mk_idx=0, px_rank=0, role='maker', addr=A1, hash='h2', y_side='bid', s=1, q=5., p=.5,
                             self_trade=False, settle_ms=20_000, m_end=.5, v_end=0., fee_fill=0., net_res=0., mq=0., rq=0., status='ok'),
                        dict(day=DAY, mid=1, sid=sids[2], mk_idx=0, px_rank=0, role='taker', addr=A1, hash='t1', y_side='ask', s=-1, q=8., p=.5,
                             self_trade=False, settle_ms=20_000, m_end=.5, v_end=0., fee_fill=0., net_res=0., mq=0., rq=0., status='ok')],
                       columns=list(OE.FV_COLS))
    # s0 settles before the take but its window overlaps it (not certainly before); s1 shares the take's settle ms
    win = {sids[0]: (17_000, 17_000, 19_000, 'book_drop'), sids[1]: (18_000, 18_000, 20_000, 'book_drop'), sids[2]: (18_000, 18_000, 20_000, 'book_drop')}
    Lt = OE.legs_frame(fvt, win)
    tk = OE.take_core(sids[2], Lt)
    ok('6_simultaneous_flag', 'simultaneous_action' in tk['flags'])
    ok('6_two_evaluations', tk['classes']['recorded'] == 'reversal_reducing' and tk['classes']['simultaneous'] == 'reduce_partial'
       and tk['classes']['certain'] == 'from_flat' and tk['timing_sensitive'], f"{tk['classes']}")
    # fill burst (contract 5 / 6 / 7.3; chain_take's expression): f1 inside [lo - 5 s, lo) -> certain; f2 hi inside but window starting
    # before lo - 5 s -> possible only (the old hi-only rule counted it certain: falsifier); f3 overlapping lo -> possible; f4 far -> neither
    rowb = lambda sid, role, st_, s_, q_: dict(day=DAY, mid=1, sid=sid, mk_idx=0, px_rank=0, role=role, addr=A1, hash='hb' + sid, y_side='bid' if s_ > 0 else 'ask',  # noqa: E731
                                               s=s_, q=q_, p=.5, self_trade=False, settle_ms=st_, m_end=.5, v_end=0., fee_fill=0., net_res=0., mq=0., rq=0., status='ok')
    fvb = pd.DataFrame([rowb('T', 'taker', 20_000, -1, 5.), rowb('f1', 'maker', 15_000, 1, 1.), rowb('f2', 'maker', 14_000, 1, 1.),
                        rowb('f3', 'maker', 19_000, 1, 1.), rowb('f4', 'maker', 5_000, 1, 1.)], columns=list(OE.FV_COLS))
    winb = {'T': (18_000, 18_000, 20_000, 'book_drop'), 'f1': (13_000, 13_000, 15_000, 'book_drop'), 'f2': (12_000, 12_000, 14_000, 'book_drop'),
            'f3': (17_000, 17_000, 19_000, 'book_drop'), 'f4': (3_000, 3_000, 5_000, 'book_drop')}
    tb = OE.take_core('T', OE.legs_frame(fvb, winb))
    ok('u_burst_contract_rule', tb['burst_cert'] == 1 and tb['burst_pos'] == 3, f"{tb['burst_cert']} {tb['burst_pos']}")
    # ================= 7: receipt earlier than the UUID =================
    hK = H('hK')
    WK = world(109, {('bid', 50): [[(1050, 5.), (4060, 0.)]]}, [(4060, [(hK, A1, 'bid', 50, 5.)], A2)], recv_shift={0: 3560})
    fvK = fv_from(WK['legs'])
    settleK = dict(zip(zip(fvK.sid[fvK.role == 'maker'], fvK.hash[fvK.role == 'maker']), fvK.settle_ms[fvK.role == 'maker'].astype(int)))
    rowsK, _, _ = OE.census_market(DAY, 109, WK['orders'], WK['fills'], WK['links'], settleK)
    cK = pd.DataFrame(rowsK)
    ok('7_flagged', bool(cK['clock_bad'].iloc[0]) and 'receipt_before_uuid' in cK['order_flags'].iloc[0], f"{cK['order_flags'].iloc[0]}")
    ok('7_no_size_claim', cK['size_status'].iloc[0] == 'clock_refused', f"{cK['size_status'].iloc[0]}")
    covK = OE.coverage_rows(cK, pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame())
    pk = covK[(covK.event == 'posting') & (covK.clock == 'acceptance_union')]
    ok('7_refused_from_bins', list(pk.category) == ['refused_clock'] and int(pk.n_units.sum()) == 1, f'{pk.to_dict("records")}')
    fm = covK[(covK.event == 'fill_match')]
    ok('7_fill_bins_refused', set(fm.category) == {'refused_clock'}, f'{fm.to_dict("records")}')
    LK = OE.legs_frame(fvK[fvK.addr_l == L2], OE.windows_of(WK['fills']))
    tkK = OE.take_core(WK['records'].sid.iloc[0], LK)
    ok('7_take_flagged_no_certain', 'receipt_before_uuid' in tkK['flags'] and 'certain' not in tkK['variants'], f"{tkK['flags']} {tkK['variants']}")
    MkK, _ = bundle(WK)
    lgK = pd.DataFrame(OE.market_cases(MkK, cases(109, [('take', WK['records'].sid.iloc[0], L2, 'e1')]), ctx, Counter())[2])
    ok('7_take_logic_unidentified', (lgK.identification_status == 'unidentified').sum() >= 2 and (lgK.evidence_grade == 'X_conflict').all(),
       f'{lgK[["mechanism", "identification_status", "evidence_grade"]].values}')
    # ================= 8: inventory classes =================
    exp = {(5, -3): 'reversal_reducing', (5, -7): 'overshoot', (5, 0): 'full_close', (5, 2): 'reduce_partial', (5, 8): 'add', (0, 3): 'from_flat',
           (-5, 3): 'reversal_reducing', (-5, -8): 'add', (-5, 5): 'overshoot'}
    got = {k: OE.inv_class(*k) for k in exp}
    ok('8_classes', got == exp, f'{got}')
    # ================= 9: perturbation =================
    MkA, fvA = bundle(WA)
    cs1 = cases(101, [('order', hM, L1, 'e1')])
    base = pd.DataFrame(OE.market_cases(MkA, cs1, ctx, Counter())[2])
    post_hi = int(WA['orders'][WA['orders'].mk_hash == hM].post_hi_ms.iloc[0])
    pst = lambda df: list(df[df.link == 'POST'].state_change)  # noqa: E731
    g = G0_MS + 100 * np.arange(len(BTC_MID))
    after = BTC_MID.copy(); after[g >= post_hi] *= 1.01
    ok('9_btc_after_unchanged', pst(pd.DataFrame(OE.market_cases(MkA, cs1, ctx_of(after), Counter())[2])) == pst(base))
    inside = BTC_MID.copy(); inside[(g >= post_hi - 3000) & (g < post_hi - 100)] *= 1.002
    ok('9_btc_inside_changes', pst(pd.DataFrame(OE.market_cases(MkA, cs1, ctx_of(inside), Counter())[2])) != pst(base))
    lv = WA['levels'].copy(); fr = WA['frames']
    late = set(fr.fid[fr.upd_ms >= post_hi + 200]); early = set(fr.fid[(fr.upd_ms < post_hi) & (fr.upd_ms >= post_hi - 1000)])
    lv2 = lv.copy(); lv2.loc[lv2.fid.isin(late) & (lv2.side == 'ask') & (lv2.ptick == 99), 'ptick'] = 60      # the same edit as lv3, after the window
    Mk2 = dict(MkA, B=LC._book(fr, lv2, 101, MkA['ws_ms'], MkA['we_ms']))
    Mk2['touch'] = OE.touch_arrays(Mk2['B'])
    ok('9_book_after_unchanged', pst(pd.DataFrame(OE.market_cases(Mk2, cs1, ctx, Counter())[2])) == pst(base))
    lv3 = lv.copy(); lv3.loc[lv3.fid.isin(early) & (lv3.side == 'ask') & (lv3.ptick == 99), 'ptick'] = 60
    Mk3 = dict(MkA, B=LC._book(fr, lv3, 101, MkA['ws_ms'], MkA['we_ms']))
    Mk3['touch'] = OE.touch_arrays(Mk3['B'])
    ok('9_book_inside_changes', pst(pd.DataFrame(OE.market_cases(Mk3, cs1, ctx, Counter())[2])) != pst(base))
    fv_late = pd.concat([fvA, fvA.iloc[[0]].assign(sid='late', settle_ms=post_hi + 500, q=33.)], ignore_index=True)
    Wn = OE.windows_of(WA['fills']); Wn['late'] = (post_hi - 1500, post_hi - 1500, post_hi + 500, 'book_drop')
    Mk4 = dict(MkA, L=dict(MkA['L']), inv=dict(MkA['inv']))
    Mk4['L'][L1] = OE.legs_frame(fv_late[fv_late.addr_l == L1], Wn); Mk4['inv'][L1] = OE.Inventory(Mk4['L'][L1])
    ok('9_leg_after_unchanged', pst(pd.DataFrame(OE.market_cases(Mk4, cs1, ctx, Counter())[2])) == pst(base))
    fv_early = pd.concat([fvA, fvA.iloc[[0]].assign(sid='early', settle_ms=post_hi - 1500, q=33.)], ignore_index=True)
    Wn['early'] = (post_hi - 3500, post_hi - 3500, post_hi - 1500, 'book_drop')
    Mk5 = dict(MkA, L=dict(MkA['L']), inv=dict(MkA['inv']))
    Mk5['L'][L1] = OE.legs_frame(fv_early[fv_early.addr_l == L1], Wn); Mk5['inv'][L1] = OE.Inventory(Mk5['L'][L1])
    ok('9_leg_inside_changes', pst(pd.DataFrame(OE.market_cases(Mk5, cs1, ctx, Counter())[2])) != pst(base))
    # ================= 10: the sample =================
    rows = ([('order', '20261024', 11, L1, f'o11_{i}') for i in range(3)] + [('order', '20261024', 12, L1, 'o12_0')]
            + [('order', '20261025', 21, L1, f'o21_{i}') for i in range(6)] + [('take', '20261024', 11, L1, f't11_{i}') for i in range(2)]
            + [('order', '20261024', 13, L3, f'p13_{i}') for i in range(5)] + [('order', '20261024', 14, L3, f'p14_{i}') for i in range(5)])
    K = pd.DataFrame(rows, columns=['kind', 'day', 'mid', 'addr', 'key'])
    days = ['20261024', '20261025']
    sel, man = OE.select_cases(K, days)
    sel2, man2 = OE.select_cases(K.sample(frac=1., random_state=3).reset_index(drop=True), days)
    ok('10_row_order_independent', frame_eq(sel, sel2) and frame_eq(man, man2))
    hx = lambda *p: hashlib.sha256('|'.join(str(x) for x in p).encode()).hexdigest()  # noqa: E731
    want21 = sorted([f'o21_{i}' for i in range(6)], key=lambda k: hx(20261123, '20261025', 21, L1, k))[:4]
    got21 = list(sel[(sel.kind == 'order') & (sel.day == '20261025') & (sel.addr == L1)].key)
    ok('10_hash_rank_within_market', got21 == want21, f'{got21} vs {want21}')
    mo = sorted([13, 14], key=lambda m: hx(20261123, '20261024', L3, m, 'market'))
    rk = {m: sorted([f'p{m}_{i}' for i in range(5)], key=lambda k, m=m: hx(20261123, '20261024', m, L3, k)) for m in (13, 14)}
    want3 = [rk[mo[0]][0], rk[mo[1]][0], rk[mo[0]][1], rk[mo[1]][1]]
    got3 = list(sel[(sel.kind == 'order') & (sel.day == '20261024') & (sel.addr == L3)].key)
    ok('10_market_first_round_robin', got3 == want3, f'{got3} vs {want3}')
    st = {(r.kind, r.addr, r.day): (r.status, r.n_selected) for r in man.itertuples()}
    ok('10_statuses', st[('order', L1, '20261024')] == ('ok', 4) and st[('take', L1, '20261024')] == ('fewer_candidates', 2)
       and st[('order', L2, '20261024')] == ('no_candidates', 0) and st[('order', L1, '20261025')] == ('ok', 4), f'{st}')
    ok('10_le4', (man.n_selected <= 4).all() and len(man) == 2 * 3 * 2)
    refuses('10_key_columns_only', 'five key columns', OE.select_cases, K.assign(cum=1.), days)
    # ================= 11: refusals =================
    refuses('11_forbidden_day', 'not a consumed day', OE.life_paths, OE.C.OUT, '20261028', 1)
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        (td / 'out' / 'order_estimator_pilot_20261024_r1').mkdir(parents=True)
        refuses('11_write_once', 'write-once', OE.run, pilot=('20261024', 1), R=td, out_root=td / 'out', log=lambda *_: None)
        (td / 'econ').mkdir()
        ids = dict(ex_ante=dict(top10_shares=[dict(addr=A2, rank=1), dict(addr=A1, rank=2), dict(addr=A3, rank=3)]))
        (td / 'econ' / 'identities.json').write_text(json.dumps(ids))
        refuses('11_wrong_population', 'differ from the decision contract TOP3', OE.load_population, td)
        p = td / 'fv.bin'; p.write_bytes(b'x' * 10)
        refuses('11_fv_bytes', 'not the accepted', OE.check_fv, p)
        refuses('11_fv_hash', 'differs from the accepted receipt', OE.check_fv, p, want_bytes=10)
        two = fvt.copy(); two.loc[0, 'addr'] = A1.lower(); two['mid'] = 11; two['day'] = '20261024'
        two.drop(columns=[]).to_parquet(td / 'fv2.parquet', index=False)
        refuses('11_two_spellings', 'two spellings', OE.read_fv, td / 'fv2.parquet', {'20261024': [11]}, Counter())
        dm = fvt.copy(); dm['mid'] = 11; dm['day'] = '20261025'
        dm.to_parquet(td / 'fv3.parquet', index=False)
        refuses('11_day_mismatch', 'under another day', OE.read_fv, td / 'fv3.parquet', {'20261024': [11]}, Counter())
    # ================= units =================
    stub = dict(union={'EX', 'PL'}, cum=7.3, hist_ub={'PL': NAN, 'EC': NAN})
    p1 = OE.priors(stub, clip=10.)
    ok('u_priors_lot_clip', p1['P_cmd'] == ('EX', 7.3) and p1['P_lot'] == ('PL', 10.0) and p1['P_clip'] == ('PL', 10.0) and p1['prior_sensitive'], f'{p1}')
    p2 = OE.priors(dict(stub, hist_ub={'PL': 8., 'EC': NAN}), clip=5.)
    ok('u_priors_no_fit', p2['P_lot'] == ('EX', 7.3) and p2['P_clip'] == ('EX', 7.3) and not p2['prior_sensitive'], f'{p2}')
    p3 = OE.priors(dict(union={'EX', 'PL'}, cum=5., hist_ub={'PL': NAN}), drop_r=45., with_drop=True)
    ok('u_prior_drop', p3['P_drop'] == ('PL', 50.0) and p3['P_cmd'] == ('EX', 5.0) and p3['prior_sensitive'] and not p3['prior_sensitive3'], f'{p3}')
    ok('u_clip', OE.clip_for(Counter({10.: 3, 5.: 2}), 10.) == 5.0 and OE.clip_for(Counter({10.: 3, 5.: 2}), NAN) == 10.0
       and np.isnan(OE.clip_for(Counter({7.: 1}), NAN)))
    T, F_ = np.ones(4, bool), np.zeros(4, bool)
    ok('u_st_all', OE.st_all(T, T) == 'compatible' and OE.st_all(F_, T) == 'contradicted' and OE.st_all(np.array([1, 0, 1, 0], bool), T) == 'timing_sensitive'
       and OE.st_all(T, F_) == 'unidentified')
    ok('u_width_bins', [OE.width_bin(w) for w in (50, 51, 250, 1000, 1001, -1, NAN)] == ['<=50', '50-250', '50-250', '250-1000', '>1000', 'unbounded', 'unbounded'])
    ce = pd.DataFrame(dict(day=[DAY] * 4, mid=[1, 1, 1, 2], addr=[L1, L1, L1, L1], t0=[0, 50, 500, 0], t1=[100, 200, 600, 100]))
    ok('u_episodes', list(OE.episode_ids(ce)) == [1, 1, 2, 3], f'{list(OE.episode_ids(ce))}')
    own = pd.DataFrame([dict(run='v6', mid=103, mk_hash=hP, link_status='unique', size_sh=10., filled_sh=10., final_state='filled', act_lo_venue_ms=T0 + 1000.,
                             act_hi_venue_ms=T0 + 1100., status='ok'),
                        dict(run='v6', mid=103, mk_hash=hP, link_status='unique', size_sh=12., filled_sh=10., final_state='cancelled', act_lo_venue_ms=T0 + 1000.,
                             act_hi_venue_ms=T0 + 1100., status='ok'),
                        dict(run='v6', mid=101, mk_hash=hM, link_status='unique', size_sh=50., filled_sh=5., final_state='cancelled', act_lo_venue_ms=T0 + 1000.,
                             act_hi_venue_ms=T0 + 1100., status='ok'),
                        dict(run='v6', mid=102, mk_hash=hS, link_status='unique', size_sh=50., filled_sh=5., final_state='cancelled', act_lo_venue_ms=T0 + 1000.,
                             act_hi_venue_ms=T0 + 1100., status='ok')])
    gt = OE.gt_rows(own, {101: WA['orders'], 102: WS['orders'], 103: WC['orders']})
    ok('u_gt_verdicts', sorted(zip(gt.mid, gt.final_state, gt.verdict)) == sorted([(103, 'filled', 'covered'), (103, 'cancelled', 'defect'),
                                                                                  (101, 'cancelled', 'covered'), (102, 'cancelled', 'priority_miss')]),
       f'{gt[["mid", "final_state", "verdict"]].values}')
    ok('u_gt_size_and_posting', bool(gt[(gt.mid == 101)].size_in_range.iloc[0]) and bool(gt[(gt.mid == 103) & (gt.final_state == 'filled')].size_in_range.iloc[0])
       and gt.posting_overlaps_truth.notna().all(), f'{gt[["mid", "size_in_range", "posting_overlaps_truth"]].values}')
    # a take end to end: A1 long 5 after its maker fill, then sells 10 -> overshoot in every variant
    hN2 = H('hN2')
    WT = world(110, {('bid', 50): [[(1050, 5.), (4060, 0.)], [(1070, 45.), (7060, 35.), (8060, 25.)]]},
               [(4060, [(hM, A1, 'bid', 50, 5.)], TK), (7060, [(hN2, OTHER, 'bid', 50, 10.)], TK), (8060, [(hN2, OTHER, 'bid', 50, 10.)], A1)])
    MkT, _ = bundle(WT)
    sidT = WT['records'].sid.iloc[2]
    oT = OE.market_cases(MkT, cases(110, [('take', sidT, L1, 'e1'), ('order', hM, L1, 'e1')]), ctx, Counter())
    crT, lgT = pd.DataFrame(oT[0]), pd.DataFrame(oT[2])
    tr = crT[crT.kind == 'take'].iloc[0]
    ok('u_take_overshoot', tr.cls == 'overshoot' and tr.cls_certain == 'overshoot' and abs(tr.pos_before - 5.) < 1e-9 and abs(tr.pos_after + 5.) < 1e-9,
       f'{tr.cls} {tr.cls_certain} {tr.pos_before} {tr.pos_after}')
    uw = lgT[(lgT.link == 'TAKE') & (lgT.mechanism == 'inventory_unwind')]
    ok('u_take_unwind_contradicted', len(uw) == 1 and uw.identification_status.iloc[0] == 'contradicted', f'{uw.identification_status.values}')
    ok('u_take_rows_valid', lgT.identification_status.isin(OE.STATUSES).all() and set(lgT.link) >= {'POST', 'END', 'TAKE'})
    # the chain_context bridge (contract 7.0) on a stub with the documented names: mapping + the POST rules end to end
    class StubCC:
        VERSION = 'stub'

        @staticmethod
        def load_market(day, mid, addrs, root=None):
            return None

        @staticmethod
        def btc_ret(M, t):
            n = len(t)
            return dict(code=np.zeros(n, np.int8), r_1s=np.full(n, 1.5), r_5s=np.full(n, -2.), r_30s=np.full(n, 3.), p_fair=np.full(n, .6))

        @staticmethod
        def book(M, t):
            n = len(t)
            return dict(spread=np.full(n, 2.), mid_c=np.full(n, 51.), bid_tick=np.full(n, 50.), ask_tick=np.full(n, 52.),
                        age_ms=np.zeros(n, np.int64), stale=np.zeros(n, bool))

        @staticmethod
        def touch_distance(M, t, side, tick):
            return dict(ticks=np.zeros(len(t)), cat=np.full(len(t), 'at', dtype=object))

        @staticmethod
        def level_size(M, t, side, tick):
            return dict(size=np.full(len(t), 7.))

        @staticmethod
        def position(M, addr, t):
            return dict(recorded=np.full(len(t), 4.), certain=np.full(len(t), 3.))
    with tempfile.TemporaryDirectory() as td:
        cctx, prov = OE.make_context(td, require_chain=True, mod=StubCC)
        x = np.array([T0 + 1000, T0 + 2000], np.int64)
        b, bk_, iv = cctx.btc(dict(M=None), x), cctx.book_at(dict(M=None), x, 0, 50), cctx.inventory(dict(M=None), L1, x)
        ok('cc_adapter_mapping', prov['provider'] == 'chain_context' and list(b['r_5s']) == [-2., -2.] and list(b['pf']) == [.6, .6]
           and list(bk_['cat']) == ['at', 'at'] and list(bk_['depth']) == [7., 7.] and list(bk_['spread']) == [2., 2.]
           and list(iv['recorded']) == [4., 4.] and list(iv['certain']) == [3., 3.], f'{prov} {b} {bk_} {iv}')
        bad = type('Bad', (), {k: staticmethod(lambda *a, **k: None) for k in ('load_market', 'btc_ret', 'book', 'touch_distance', 'level_size')})
        refuses('cc_adapter_missing_function', 'required function position missing', OE.make_context, td, True, bad)
    MkS, _ = bundle(WA)
    MkS['M'] = None
    lgS = pd.DataFrame(OE.market_cases(MkS, cs1, cctx, Counter())[2]).set_index('mechanism')
    ok('cc_rules_end_to_end', lgS.loc['touch_join_or_improve', 'identification_status'] == 'compatible'
       and lgS.loc['inventory_offset', 'identification_status'] == 'contradicted'
       and lgS.loc['reaction_to_price_change', 'identification_status'] == 'compatible'
       and lgS.loc['pull_after_adverse_move', 'conditional_status'] == 'compatible'
       and lgS.loc['pull_after_adverse_move', 'identification_status'] == 'unidentified',
       f"{lgS[['identification_status', 'conditional_status']].to_dict()}")
    # ================= summary =================
    n_ok = sum(1 for _, v in RES if v)
    print(json.dumps(dict(checks=len(RES), passed=n_ok, failed=[k for k, v in RES if not v], cpu_s=round(time.process_time() - t0, 2),
                          vmhwm_mib=round(LC.vmhwm_mib(), 1)), indent=1))
    if n_ok != len(RES):
        print('ORDER ESTIMATOR SELFTEST FAILED')
        sys.exit(1)
    print('ORDER ESTIMATOR SELFTEST PASSED')


if __name__ == '__main__':
    main()
