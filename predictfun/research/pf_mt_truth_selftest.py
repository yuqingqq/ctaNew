"""Selftest of pf_mt_truth (synthetic only; reads no data). Usage (run_light, PF_SLOT=claude): pf_mt_truth_selftest.py
P1 build_truth on a planted export: one band PER RUN (v6 [-100, +50], basis [-150, +1000] ms) applied to each run's orders; link unique /
   unique_by_time / ambiguous; every status, final state, venue-band time, per-notice bound list and receipt count = its hand value.
P2 also: x_size_ub (B1), the queue tie-break by record rank (N1, HX4), unique (order, field) defects (N2), caveats (N3), the multi-fill
   lo-order test (N4, H4), boundary_censored exclusion and the GT completeness gate (B2), and the dry gate (passes / refuses missing /
   refuses differing; B3).
P2 writer round trip (write-once parquet + registry, read_truth) and score() on planted lifecycle tables: every coverage, identity,
   posting, size, removal, persistence, queue and fill-window count and every defect row equals its hand-computed value.
K  known-bad inputs refused with the reason asserted (each differs from the good input in the planted defect only): truth clock registry
   on the bot clock; truth venue times not converted; ns times in the lifecycle; a duplicate hash in truth; a lifecycle row for a hash not
   in the public data; an unknown class; an unknown removal member; a market outside GT; an overwrite; kernel/core disagreement; an empty
   clock band; a malformed own hash; a duplicate own leg."""
import copy
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pf_mt_common as C  # noqa: E402
import pf_mt_truth as T  # noqa: E402

B = 1_794_110_000_000.0
OUR, XB, XC = '0x' + 'a' * 40, '0x' + 'b' * 40, '0x' + 'c' * 40
OUR_MIX = '0x' + 'aA' * 20                  # the same address in EIP-55-style mixed case (real REST signers)
NAN = np.nan


def hh(n):
    return '0x' + f'{n:064x}'


H1, H2, H3, H4, H10, H11, H13, H20, H99 = (hh(n) for n in (1, 2, 3, 4, 10, 11, 13, 20, 99))
HX1, HX2, HX3, HX4 = hh(0xA1), hh(0xA2), hh(0xA3), hh(0xA4)
FAIL, N = [], dict(checks=0, refusals=0)


def check(name, got, exp):
    N['checks'] += 1
    if got != exp:
        FAIL.append(f'{name}: got {got!r} expected {exp!r}')


def refused(name, fn, reason):
    N['refusals'] += 1
    try:
        fn()
    except ValueError as e:
        if reason not in str(e):
            FAIL.append(f'{name}: refused for another reason: {e}')
        return
    FAIL.append(f'{name}: NOT refused (expected {reason!r})')


# ------------------------------------------------------------------ planted export ------------------------------------------------------------------
# (cid, market, up, tick, size, lane, req, place, signed, k_acc, c_acc, cancel_req, ack, echo, done, rejected, refused, filled)
O6 = [(1, 101, True, 40, 5., 0, 999, 1000, 1000.25, 1030, 1030, 2500, 2520, 2600, 2520, NAN, None, 5.),
      (2, 101, False, 60, 5., 1, 4949, 4950, 4950.25, 4980, 4980, 5100, 5120, 5200, 5120, NAN, None, 5.),
      (3, 101, False, 60, 5., 3, 6099, 6100, 6100.25, 6130, 6130, 6300, 6320, 6400, 6320, NAN, None, 5.),
      (4, 101, True, 41, 25., 2, 7999, 8000, 8000.25, 8040, 8040, 11000, 11030, 11100, 11030, NAN, None, 10.),
      (5, 101, True, 40, 5., 2, 499, 500, 500.25, 530, 530, 3500, 3520, 3600, 3520, NAN, None, 0.),
      (6, 102, True, 30, 5., 0, 12000, 12000.5, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, 'risk NetCap', 0.),
      (7, 102, True, 31, 5., 0, 12999, 13000, 13000.25, NAN, NAN, NAN, NAN, NAN, NAN, 13030, None, 0.),
      (8, 102, False, 70, 5., 1, 13999, 14000, 14000.25, NAN, NAN, 14100, 14150, NAN, 14150, NAN, None, 0.),
      (9, 102, True, 32, 5., 0, 14999, 15000, 15000.25, 15040, 15040, NAN, NAN, NAN, NAN, NAN, None, 0.),
      (10, 102, True, 33, 5., 0, 22799, 22800, 22800.25, 22830, 22830, 23600, 23620, NAN, 23620, NAN, None, 5.),
      (11, 102, True, 33, 5., 2, 22899, 22900, 22900.25, 22930, 22930, 23700, 23720, NAN, 23720, NAN, None, 5.),
      (12, 102, True, 34, 25., 2, 29999, 30000, 30000.25, NAN, 30040, 30500, 30520, NAN, 30520, NAN, None, 0.),
      (13, 102, False, 80, 5., 1, 49899, 49900, 49900.25, 49930, 49930, NAN, NAN, NAN, NAN, NAN, None, 5.)]
OB = [(20, 201, False, 55, 5., 1, 39999, 40000, 40000.25, 40030, 40030, NAN, NAN, NAN, NAN, NAN, None, 5.),
      (21, 201, True, 45, 5., 0, 41999, 42000, 42000.25, 42030, 42030, 42300, 42320, NAN, 42320, NAN, None, 0.)]
# kernel fill notices (cid, at, match_ms, size, price, status)
F6 = [(1, 2400, 2000, 5., .40, 'Matched'), (1, 3100, 3000, 5., .40, 'Confirmed'),
      (2, 5010, 4000, 5., .40, 'Matched'), (2, 5600, 5000, 5., .40, 'Confirmed'),
      (3, 6200, 6000, 5., .40, 'Matched'), (3, 7300, 7000, 5., .40, 'Confirmed'),
      (4, 9100, 9000, 4., .41, 'Matched'), (4, 9600, 9000, 6., .41, 'Matched'), (4, 10100, 10000, 4., .41, 'Confirmed'), (4, 10200, 10000, 6., .41, 'Confirmed'),
      (10, 23500, 23000, 5., .33, 'Matched'), (10, 24100, 24000, 5., .33, 'Confirmed'),
      (11, 23600, 23000, 5., .33, 'Matched'), (11, 24200, 24000, 5., .33, 'Confirmed'),
      (13, 50500, 50000, 5., .80, 'Matched'), (13, 51200, 51000, 5., .80, 'Confirmed')]       # c13: price .80 != token price .20
FB = [(20, 40800, 40000, 5., .45, 'Matched'), (20, 41150, 41000, 5., .45, 'Confirmed'), (21, 42200, 42000, 2., .45, 'Failed')]  # basis band [-150, 1000]
# public maker legs (mid, exec_ms, tx, maker, hash, yes_side, ptick, shares)
LEGS = [(101, 3000, 't1', OUR_MIX, H1, 'bid', 40, 5.), (101, 5000, 't2', OUR, H2, 'ask', 60, 5.), (101, 7000, 't3', OUR, H3, 'ask', 60, 5.),
        (101, 10000, 't4a', OUR, H4, 'bid', 41, 4.), (101, 10000, 't4b', OUR, H4, 'bid', 41, 6.), (102, 24000, 't10', OUR, H10, 'bid', 33, 5.),
        (102, 24000, 't11', OUR, H11, 'bid', 33, 5.), (102, 51000, 't13', OUR, H13, 'ask', 80, 5.), (201, 41000, 't20', OUR, H20, 'ask', 55, 5.),
        (201, 43000, 't99', OUR, H99, 'bid', 50, 3.),
        (101, 2000, 'tx1', XB, HX1, 'bid', 40, 3.), (102, 31000, 'tx2', XC, HX2, 'bid', 34, 2.), (101, 5000, 'tx3', XC, HX3, 'ask', 60, 1.),
        (102, 30000, 'tx4', XB, HX4, 'bid', 34, 2.)]


def export():
    def orders(rows, run):
        K = pd.DataFrame([dict(cid=r[0], market=r[1], up=r[2], price=float(r[3]), size=r[4], t_place=B + r[7], refused=r[16], t_signed=B + r[8],
                               t_accepted=B + r[9], t_CancelPendingAudit=B + r[12], t_UserCancel=B + r[13]) for r in rows])
        P = pd.DataFrame([dict(market=r[1], id=i, lane=r[5], tick=r[3], size=r[4], t_req=B + r[6], t_Accepted=B + r[10], t_CancelDone=B + r[14],
                               t_Rejected=B + r[15], t_cancel=B + r[11], filled=r[17], cid=r[0], refused=r[16], up=r[2]) for i, r in enumerate(rows)])
        return K, P

    def notices(rows):
        return pd.DataFrame([dict(cid=r[0], at=B + r[1], match_ms=B + r[2], size=r[3], price=r[4], status=r[5]) for r in rows])
    K6, P6 = orders(O6, 'v6'); KB, PB = orders(OB, 'basis')
    M = pd.DataFrame([dict(mid=r[0], exec_s=int((B + r[1]) // 1000), tx=r[2], maker=r[3], maker_hash=r[4], yes_side=r[5], ptick=r[6], shares=r[7]) for r in LEGS])
    return dict(windows=pd.DataFrame(dict(market=[101, 102, 201], ws=[0, 300, 600], run=['v6', 'v6', 'basis'])), our=OUR, sha={},
                kernel_orders_v6=K6, core_orders_v6=P6, kernel_fills_v6=notices(F6), kernel_orders_basis=KB, core_orders_basis=PB,
                kernel_fills_basis=notices(FB), matches=M)


def p1_build():
    Tr, rec = T.build_truth(export())
    check('band per run', {r: (v['d_lo_ms'], v['d_hi_ms']) for r, v in rec['clock_band']['per_run'].items()}, {'v6': (-100.0, 50.0), 'basis': (-150.0, 1000.0)})
    o = Tr[Tr.cid >= 0].set_index('cid')
    check('row bands = run bands', {c: (o.d_lo_ms[c], o.d_hi_ms[c]) for c in (1, 13, 20, 21)},
          {1: (-100.0, 50.0), 13: (-100.0, 50.0), 20: (-150.0, 1000.0), 21: (-150.0, 1000.0)})
    check('notice bound lists', {c: [float(x) - B for x in o.match_hi_list_venue_ms[c].split('|') if x] for c in (4, 20, 5)},
          {4: [9150.0, 9650.0], 20: [41000.0], 5: []})
    check('status', o.status.to_dict(), {1: 'ok', 2: 'ok', 3: 'ok', 4: 'ok', 5: 'unfilled', 6: 'refused', 7: 'rejected', 8: 'unconfirmed_live',
                                         9: 'no_removal_record', 10: 'link_ambiguous', 11: 'link_ambiguous', 12: 'unfilled', 13: 'fill_price_mismatch',
                                         20: 'ok', 21: 'unfilled'})
    check('link', {c: (o.mk_hash[c], o.link_status[c]) for c in (1, 2, 3, 4, 10, 11, 13, 20, 5)},
          {1: (H1, 'unique'), 2: (H2, 'unique_by_time'), 3: (H3, 'unique_by_time'), 4: (H4, 'unique'), 10: ('', 'ambiguous'), 11: ('', 'ambiguous'),
           13: (H13, 'unique'), 20: (H20, 'unique'), 5: ('', 'none')})
    check('final', o.final_state.to_dict(), {1: 'filled', 2: 'filled', 3: 'filled', 4: 'cancelled', 5: 'cancelled', 6: 'other', 7: 'other', 8: 'cancelled',
                                             9: 'expired', 10: 'filled', 11: 'filled', 12: 'cancelled', 13: 'filled', 20: 'filled', 21: 'cancelled'})
    check('yes space', {c: (o.y_side[c], o.y_ptick[c], o.quote[c], o.index_set[c], o.token_price[c]) for c in (1, 2, 20)},
          {1: ('bid', 40, 'Bid', 1, .4), 2: ('ask', 60, 'Bid', 2, .4), 20: ('ask', 55, 'Bid', 2, .45)})
    v = lambda c, k: None if np.isnan(o[k][c]) else round(o[k][c] - B, 3)  # noqa: E731
    exp = {1: (900.25, 1080, 2450, 2450, 900.25, 2450, None, None), 2: (4850.25, 5000, 5000, 5000, 4850.25, 5000, None, None),
           3: (6000.25, 6180, 6250, 6250, 6000.25, 6250, None, None), 4: (7900.25, 8090, 9150, 9650, 10900, 11080, 8090, 10900),
           5: (400.25, 580, None, None, 3400, 3570, 580, 3400), 8: (13900.25, None, None, None, 14000, 14200, None, None),
           9: (14900.25, 15090, None, None, 14900.25, None, None, None), 12: (29900.25, 30090, None, None, 30400, 30570, 30090, 30400),
           20: (39850.25, 41000, 41000, 41000, 39850.25, 41000, None, None), 21: (41850.25, 43030, None, None, 42150, 43320, None, None)}  # basis band
    ks = ['act_lo_venue_ms', 'act_hi_venue_ms', 'match_hi_first_venue_ms', 'match_hi_last_venue_ms', 'rm_lo_venue_ms', 'rm_hi_venue_ms',
          'present_lo_venue_ms', 'present_hi_venue_ms']
    check('venue times', {c: tuple(v(c, k) for k in ks) for c in exp}, exp)
    check('true removal', {c: o.true_removal[c] for c in (1, 4, 6, 8, 9)}, {1: 'exhausted', 4: 'cancelled_or_expired', 6: '', 8: 'cancelled_or_expired', 9: 'cancelled_or_expired'})
    check('legs', {c: (o.n_legs[c], o.legs_sh[c], o.leg_shares[c]) for c in (4, 5)}, {4: (2, 10., '4.0|6.0'), 5: (0, 0., '')})
    check('accepted src', (o.accepted_src[12], o.accepted_src[1], o.accepted_src[8]), ('core.t_Accepted', 'kernel.t_accepted', ''))
    check('remaining', (o.remaining_lb_sh[4], o.remaining_lb_sh[1]), (15., 0.))
    u = Tr[Tr.cid < 0].set_index('mk_hash')
    check('unlinked public hashes', {h: (u.status[h], u.x_link_candidates[h], u.mid[h]) for h in u.index},
          {H10: ('public_hash_unlinked', '10|11', 102), H11: ('public_hash_unlinked', '10|11', 102), H99: ('public_hash_unlinked', '', 201)})
    check('rows', (rec['rows'], len(Tr)), (18, 18))
    check('own public', rec['own_public'], dict(legs=10, hashes=9, shares=48.0, hash_identity_conflicts=0))
    check('checks', rec['checks'], {'v6': dict(orders=13, core_filled_eq_matched=13, matched_notices=8, matched_price_eq_token_price=7, accepted_from_core_only=1),
                                    'basis': dict(orders=2, core_filled_eq_matched=2, matched_notices=1, matched_price_eq_token_price=1, accepted_from_core_only=0)})
    check('scored', rec['scored'], dict(orders=5, legs=6, shares=30.0, exhausted=4, cancelled=1))
    check('gt', {k: rec['gt_markets'][k] for k in ('scope', 'with_own_orders', 'with_accepted', 'with_own_fills', 'own_fill_mids')},
          dict(scope=3, with_own_orders=3, with_accepted=3, with_own_fills=3, own_fill_mids=[101, 102, 201]))
    check('presence rows', rec['presence_rows'], 3)          # c21's 270 ms life is shorter than the basis band width (1150 ms)
    check('link evidence', sorted((e['cid'], e['mk_hash'], e['link'], e['linked'], e['time_consistent']) for e in rec['link_evidence']),
          sorted([(2, H2, 'unique_by_time', True, True), (2, H3, 'unique_by_time', False, False), (3, H2, 'unique_by_time', False, False),
                  (3, H3, 'unique_by_time', True, True)] + [(c, h, 'ambiguous', False, True) for c in (10, 11) for h in (H10, H11)]))
    check('signers', rec['signer']['signers_of_own_legs'], [OUR])
    check('columns', list(Tr.columns), T.COLNAMES)
    check('code hashes', sorted(rec['code']), ['pf_mt_common.py', 'pf_mt_truth.py'])
    return Tr, rec


# ------------------------------------------------------------------ planted lifecycle ------------------------------------------------------------------
def orow(sc, mid, h, signer, side, pt, nf, cum, size_lb, size_exact, size_cls, post, post_cls, persist, persist_cls, ahead_hi, q_cls, rset, rwin, r_cls):
    b = lambda x: int(B + x) if x >= 0 else -1  # noqa: E731
    return dict(scenario=sc, mid=mid, mk_hash=h, mk_signer=signer, y_side=side, y_ptick=pt, n_fills=nf, cum_shares=cum, first_hi_ms=int(B + 1),
                last_hi_ms=int(B + 1), size_lb=size_lb, size_exact=size_exact, size_class=size_cls, post_lo_ms=b(post[0]), post_hi_ms=b(post[1]),
                post_class=post_cls, post_method='capacity', persist_lo_ms=b(persist[0]), persist_hi_ms=b(persist[1]), persist_class=persist_cls,
                ahead_lo=0. if ahead_hi == ahead_hi else NAN, ahead_hi=ahead_hi, queue_class=q_cls, removal_set=rset, removal_lo_ms=b(rwin[0]),
                removal_hi_ms=b(rwin[1]), removal_class=r_cls, removal_method='m', status='ok')


def lifecycle():
    CO, OB_, SE, UN = C.CONSTRAINED, C.OBSERVED, C.SENSITIVE, C.UNRESOLVED
    N = (-1, -1)
    fifo = [  # S_FIFO rows; S_FIFO_LO copies them with two planted identity defects; S_NOPRIO below
        (101, H1, OUR, 'bid', 40, 1, 5., 5., 5., CO, (800, 2450), CO, (2450, 2450), CO, 0., SE, 'exhausted', (2400, 2450), SE),
        (101, H2, OUR, 'ask', 60, 1, 5., 5., NAN, OB_, (5100, 5200), CO, (4900, 5100), CO, NAN, UN, 'cancelled_or_expired', (4900, 5000), CO),
        (101, H3, OUR, 'ask', 60, 1, 5., 6., NAN, OB_, (-1, 6250), OB_, N, UN, NAN, UN, 'end_censored', N, UN),
        (101, H4, OUR, 'bid', 41, 2, 10., 10., 10., SE, (7000, 9000), CO, (9150, 9650), CO, 0., SE, 'exhausted', (9600, 9700), SE),  # front vs ITSELF
        (201, H20, OUR, 'ask', 55, 1, 5., 5., 5., SE, (39000, 41000), CO, (40850, 40850), CO, NAN, UN, 'exhausted', (40800, 40850), SE),
        (102, H13, OUR, 'ask', 80, 1, 5., 5., NAN, OB_, N, UN, N, UN, NAN, UN, 'missing_evidence', N, UN),
        (101, HX1, XB, 'bid', 40, 1, 3., 3., NAN, OB_, N, UN, N, UN, 0., CO, 'end_censored', N, UN),
        (102, HX2, XC, 'bid', 34, 1, 2., 2., NAN, OB_, N, UN, N, UN, 0., SE, 'end_censored', N, UN),
        (101, HX3, XC, 'ask', 60, 1, 1., 1., NAN, OB_, N, UN, N, UN, 0., CO, 'end_censored', N, UN),
        (102, HX4, XB, 'bid', 34, 2, 2., 2., NAN, OB_, N, UN, N, UN, 0., SE, 'end_censored', N, UN)]    # tie at hi: record rank decides
    rows = [orow('S_FIFO', *r) for r in fifo]
    for r in fifo:
        r = list(r)
        if r[1] == H1:
            r[5] = 2                                   # planted: n_fills 2 vs one true leg
        if r[1] == HX3:
            r[2] = OUR_MIX                             # planted: our signer (mixed case) on a hash unknown to truth
        rows.append(orow('S_FIFO_LO', *r))
    nop = [(101, H1, OUR, 'bid', 40, 1, 5., 5., NAN, OB_, (800, 2450), CO, (2450, 2450), CO, NAN, UN, 'exhausted|cancelled_or_expired', (900, 2500), CO),
           (101, H2, OUR, 'ask', 60, 1, 5., 5., NAN, OB_, (4800, 5000), CO, N, UN, NAN, UN, 'cancelled_or_expired|exhausted', (4800, 5100), CO),
           (101, H3, OUR, 'ask', 60, 1, 5., 6., NAN, OB_, (-1, 6250), OB_, N, UN, NAN, UN, 'end_censored', N, UN),
           (101, H4, OUR, 'bid', 41, 2, 10., 10., NAN, OB_, (7000, 8000), CO, (9150, 9650), CO, NAN, UN, 'cancelled_or_expired|exhausted', (10800, 11100), CO),
           (101, HX1, XB, 'bid', 40, 1, 3., 3., NAN, OB_, N, UN, N, UN, NAN, UN, 'end_censored', N, UN),
           (102, HX2, XC, 'bid', 34, 1, 2., 2., NAN, OB_, N, UN, N, UN, NAN, UN, 'end_censored', N, UN),
           (101, HX3, XC, 'ask', 60, 1, 1., 1., NAN, OB_, N, UN, N, UN, NAN, UN, 'end_censored', N, UN),
           (102, HX4, XB, 'bid', 34, 2, 2., 2., NAN, OB_, N, UN, N, UN, NAN, UN, 'end_censored', N, UN)]
    rows += [orow('S_NOPRIO', *r) for r in nop]
    rows = [dict(r, mk_signer=OUR_MIX) if r['scenario'] == 'S_FIFO' and r['mk_signer'] == OUR else r for r in rows]   # mixed-case own signer
    O = pd.DataFrame(rows)
    O['x_size_ub'] = NAN                                   # planted size upper bounds (S_FIFO and S_FIFO_LO): H1 holds, H2 / H4 violated
    for sc in ('S_FIFO', 'S_FIFO_LO'):
        for h, v in ((H1, 5.), (H2, 4.5), (H4, 20.)):
            O.loc[(O.scenario == sc) & (O.mk_hash == h), 'x_size_ub'] = v
    f = lambda mid, sid, h, i, side, pt, sh, hi, lo, rk, rh=False: dict(  # noqa: E731   x_lo_hard_ms = hi - MAX_LAG_MS (-1 receipt-hi)
        mid=mid, sid=sid, mk_hash=h, mk_idx=i, y_side=side, y_ptick=pt, shares=sh, hi_ms=int(B + hi), lo_ms=int(B + lo) if lo >= 0 else -1,
        t_method='book', f_before=1, f_after=2, status='ok', x_rec_rank=rk, x_lo_hard_ms=-1 if rh else int(B + hi - C.MAX_LAG_MS))
    F = pd.DataFrame([f(101, 's1', H1, 0, 'bid', 40, 5., 2450, 1500, 0), f(101, 's1', HX1, 1, 'bid', 40, 3., 1600, 1500, 0),
                      f(101, 's2', H2, 0, 'ask', 61, 5., 5100, 5050, 1),          # planted: tick 61 vs 60; refined lo after the latest match (hard holds)
                      f(101, 's3', H3, 0, 'ask', 60, 5., 8400, 8300, 2),          # planted: hard lo 6400 after the latest match 6250 (defect)
                      f(101, 's4a', H4, 0, 'bid', 41, 4., 9300, 9200, 3), f(101, 's4b', H4, 0, 'bid', 41, 6., 9650, 9300, 4),  # refined lo #1 > bound #1 only
                      f(201, 's20', H20, 0, 'ask', 55, 2.5, 40100, 40000, 7, rh=True),  # planted: wrong size (phantom + uncovered leg); receipt hi
                      f(102, 'sx2', HX2, 0, 'bid', 34, 2., 30300, 30200, 5), f(101, 'sx3', HX3, 0, 'ask', 60, 1., 5000, -1, 6),
                      f(102, 'sx4a', HX4, 0, 'bid', 34, 1., 30300, 30200, 9), f(102, 'sx4b', HX4, 1, 'bid', 34, 1., 30300, 29000, 8)])
    return O, F


def p2_score(Tr, rec):
    O, F = lifecycle()
    pub = set(pd.DataFrame(LEGS, columns=list('abcdefgh')).e)
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        refused('K dry gate: target missing', lambda: T.dry_gate(Tr, rec, d), 'target truth table')
        T.write_truth(Tr, rec, d)
        TT = T.read_truth(d / 'own_orders.parquet')
        check('registry round trip', {r: b['d_hi_ms'] for r, b in TT.attrs['mt_truth']['clock_band']['per_run'].items()}, {'v6': 50.0, 'basis': 1000.0})
        check('dry gate passes on the table it wrote', T.dry_gate(Tr, rec, d), True)
        res = T.score(O, F, TT, pub, [101, 102, 201])
        res_c = T.score(O, F, TT, pub, [101, 102, 201], censored_mids=[201])
        check('censored window excluded and counted', (res_c['population']['boundary_censored'], res_c['population']['scored_own_orders']),
              (dict(windows=[201], lifecycle_orders_rows=2, lifecycle_fills_rows=1, own_orders=1, own_legs=1, own_shares=5.0), 4))
        Fh = F.copy(); h4 = Fh.mk_hash == H4                  # hard lo #1 (9200) > notice bound #1 (9150): an observed lo-order defect
        Fh.loc[h4, 'hi_ms'] = [int(B + 11200), int(B + 11300)]; Fh.loc[h4, 'x_lo_hard_ms'] = [int(B + 9200), int(B + 9300)]
        res_h = T.score(O, Fh, TT, pub, [101, 102, 201])
        check('hard lo-order defect', (res_h['fills']['lo_order'], res_h['fills']['defects']),
              (dict(multi_fill_orders=1, compared=1, count_mismatch=0, violations=1, refined_violations=1), 4))
        T._write_once(d / 'scores.json', lambda p: p.write_text('{}'))
        refused('K write-once scores', lambda: T._write_once(d / 'scores.json', lambda p: p.write_text('{}')), 'exists (write-once)')
        refused('K write-once truth', lambda: T.write_truth(Tr, rec, d), 'exists (write-once)')
    with tempfile.TemporaryDirectory() as d:                  # the dry gate refuses a table this code does not reproduce
        T.write_truth(Tr.assign(size_sh=Tr.size_sh + (Tr.cid == 1) * 1.0), rec, Path(d))
        refused('K dry gate: differing table', lambda: T.dry_gate(Tr, rec, Path(d)), 'does not reproduce the written truth table')
    with tempfile.TemporaryDirectory() as d:                  # B2 completeness over the selected GT windows (301 boundary_censored)
        d = Path(d); (d / 'life').mkdir()
        (d / 'extract_20261108.json').write_text(json.dumps(dict(ws_population=dict(selected=[101, 102, 201, 301]))))
        pd.DataFrame(dict(mid=[101, 102, 201, 301], status=['ok', 'ok', 'ok', 'boundary_censored+hour_missing'])).to_parquet(d / 'markets_20261108.parquet')
        for m in (101, 102, 201):
            for k in (f'20261108_{m}_orders.parquet', f'20261108_{m}_fills.parquet', f'20261108_{m}.json'):
                (d / 'life' / k).write_text('x')
        G = T.gt_lifecycle_inputs(d, '20261108')
        check('GT completeness', (G['selected'], G['censored'], G['required']), ([101, 102, 201, 301], [301], [101, 102, 201]))
        (d / 'life' / '20261108_102.json').unlink()
        refused('K incomplete lifecycle', lambda: T.gt_lifecycle_inputs(d, '20261108'), 'incomplete lifecycle for the selected GT windows')
    return res, TT, O, F, pub


def expect_score(res):
    p = res['population']
    check('population', {k: p[k] for k in ('truth_rows', 'scored_own_orders', 'scored_legs', 'scored_shares', 'gt_markets', 'lifecycle_markets',
                                           'own_fill_markets_without_lifecycle', 'presence_orders', 'boundary_censored')},
          dict(truth_rows=18, scored_own_orders=5, scored_legs=6, scored_shares=30.0, gt_markets=3, lifecycle_markets=3,
               own_fill_markets_without_lifecycle=[], presence_orders=3,
               boundary_censored=dict(windows=[], lifecycle_orders_rows=0, lifecycle_fills_rows=0, own_orders=0, own_legs=0, own_shares=0.0)))
    check('caveats', (res['caveats']['outcome_selection'].startswith('only filled orders print a public hash'), res['caveats']['clock_bands_ms'],
                      len(res['caveats']['clock_premise']) > 0), (True, {'v6': [-100.0, 50.0], 'basis': [-150.0, 1000.0]}, True))
    fb = res['fills']
    check('fills coverage', (fb['executions'], fb['shares'], fb['phantom_fills'], fb['identity_mismatches'], fb['defects']),
          (dict(n=6, covered=5, rate=0.833333), dict(n=30.0, covered=25.0, rate=0.833333), 1, 1, 3))
    check('fills window (observed: x_lo_hard_ms)', fb['window']['observed'],
          dict(n=6, disjoint=1, inside=3, overlap=2, failed=1, defects=1, sensitive_misses=0, gap_ms_failed=dict(min=150.0, max=150.0)))
    check('fills window (refined: lo_ms)', fb['window_refined']['assumption_sensitive'],
          dict(n=6, disjoint=2, overlap=4, failed=2, defects=0, sensitive_misses=2, gap_ms_failed=dict(min=50.0, max=2050.0)))
    check('fills lo order', fb['lo_order'], dict(multi_fill_orders=1, compared=1, count_mismatch=0, violations=0, refined_violations=1))
    S, L, N = (res['scenarios'][k] for k in ('S_FIFO', 'S_FIFO_LO', 'S_NOPRIO'))
    z = lambda **kw: dict(dict(failed=0, defects=0, sensitive_misses=0), **kw)  # noqa: E731
    check('S_FIFO coverage', S['coverage'], dict(orders=dict(n=5, covered=5, rate=1.0), executions=dict(n=6, covered=6, rate=1.0), shares=dict(n=30.0, covered=30.0, rate=1.0)))
    check('S_NOPRIO coverage', N['coverage'], dict(orders=dict(n=5, covered=4, rate=0.8), executions=dict(n=6, covered=5, rate=0.833333), shares=dict(n=30.0, covered=25.0, rate=0.833333)))
    check('S_FIFO identity', S['identity'], dict(n=5, signer=0, mid=0, y_side=0, y_ptick=0, n_fills=0, cum_shares=0, own_signer_hash_not_in_truth=0,
                                                 own_hash_excluded_from_truth_scoring=1, defects=0))
    check('S_FIFO_LO identity', L['identity'], dict(n=5, signer=0, mid=0, y_side=0, y_ptick=0, n_fills=1, cum_shares=0, own_signer_hash_not_in_truth=1,
                                                    own_hash_excluded_from_truth_scoring=1, defects=2))
    check('S_FIFO posting constrained', S['posting'][C.CONSTRAINED], z(n=4, disjoint=1, inside=3, failed=1, defects=1, gap_ms_failed=dict(min=100.0, max=100.0)))
    check('S_FIFO posting observed', S['posting'][C.OBSERVED], z(n=1, disjoint=0, inside=1))
    check('S_NOPRIO posting', (N['posting'][C.CONSTRAINED], N['posting'][C.OBSERVED]), (z(n=3, inside=2, overlap=1), z(n=1, inside=1, overlap=0)))
    check('S_FIFO size lb', S['size']['lb'][C.OBSERVED], z(n=5, holds=4, violated=1, failed=1, defects=1))
    check('S_FIFO size exact', {c: S['size']['exact'][c] for c in (C.CONSTRAINED, C.OBSERVED, C.SENSITIVE)},
          {C.CONSTRAINED: z(n=1, equal=1, not_stated=0, unequal=0), C.OBSERVED: z(n=2, equal=0, not_stated=2, unequal=0),
           C.SENSITIVE: z(n=2, equal=1, not_stated=0, unequal=1, failed=1, sensitive_misses=1)})
    check('S_FIFO size ub', {c: S['size']['ub'][c] for c in (C.CONSTRAINED, C.OBSERVED, C.SENSITIVE)},
          {C.CONSTRAINED: z(n=1, holds=1, not_stated=0, violated=0), C.OBSERVED: z(n=2, holds=0, not_stated=1, violated=1, failed=1, defects=1),
           C.SENSITIVE: z(n=2, holds=0, not_stated=1, violated=1, failed=1, sensitive_misses=1)})
    check('S_FIFO exhausted-only', {c: S['removal']['exhausted_only'][c] for c in (C.SENSITIVE, C.CONSTRAINED, C.UNRESOLVED)},
          {C.SENSITIVE: z(n=3, not_exhausted_only=0, not_filled=1, truly_filled=2, failed=1, sensitive_misses=1),
           C.CONSTRAINED: z(n=1, not_exhausted_only=1, not_filled=0, truly_filled=0), C.UNRESOLVED: z(n=1, not_exhausted_only=1, not_filled=0, truly_filled=0)})
    check('S_FIFO membership', {c: S['removal']['membership'][c] for c in (C.SENSITIVE, C.CONSTRAINED, C.UNRESOLVED)},
          {C.SENSITIVE: z(n=3, contradicted=1, member=2, uninformative=0, failed=1, sensitive_misses=1),
           C.CONSTRAINED: z(n=1, contradicted=1, member=0, uninformative=0, failed=1, defects=1), C.UNRESOLVED: z(n=1, contradicted=0, member=0, uninformative=1)})
    check('S_FIFO removal window', {c: S['removal']['window'][c] for c in (C.SENSITIVE, C.CONSTRAINED, C.UNRESOLVED)},
          {C.SENSITIVE: z(n=3, disjoint=1, no_claim=0, overlap=2, failed=1, sensitive_misses=1, gap_ms_failed=dict(min=1200.0, max=1200.0)),
           C.CONSTRAINED: z(n=1, disjoint=0, no_claim=0, overlap=1), C.UNRESOLVED: z(n=1, disjoint=0, no_claim=1, overlap=0)})
    check('S_FIFO set vs cause', S['removal']['set_vs_true_cause'], {'cancelled_or_expired': {'exhausted': 1}, 'end_censored': {'exhausted': 1},
                                                                     'exhausted': {'cancelled_or_expired': 1, 'exhausted': 2}})
    check('S_NOPRIO removal', (N['removal']['membership'][C.CONSTRAINED], N['removal']['window'][C.CONSTRAINED], N['removal']['set_vs_true_cause']),
          (z(n=3, member=3, uninformative=0), z(n=3, inside=3, no_claim=0),
           {'cancelled_or_expired|exhausted': {'cancelled_or_expired': 1, 'exhausted': 2}, 'end_censored': {'exhausted': 1}}))
    check('S_FIFO persistence', (S['persistence'][C.CONSTRAINED], S['persistence'][C.UNRESOLVED]),
          (z(n=4, no_claim=0, outside_possible_life=1, within=3, failed=1, defects=1, gap_ms_failed=dict(min=100.0, max=100.0)),
           z(n=1, no_claim=1, outside_possible_life=0, within=0)))
    check('S_FIFO queue own', S['queue']['own_front_claims'][C.SENSITIVE], z(n=2, evidence_against=1, no_evidence=1, failed=1, sensitive_misses=1))
    check('S_FIFO queue other', (S['queue']['other_makers_front_claims'][C.CONSTRAINED], S['queue']['other_makers_front_claims'][C.SENSITIVE]),
          (z(n=2, evidence_against=1, no_evidence=0, no_ref_time=1, failed=1, defects=1),
           z(n=2, evidence_against=1, no_evidence=1, no_ref_time=0, failed=1, sensitive_misses=1)))     # HX4: rank-first fill precedes the presence
    check('S_NOPRIO queue', sum(v['n'] for v in N['queue']['own_front_claims'].values()) + sum(v['n'] for v in N['queue']['other_makers_front_claims'].values()), 0)
    check('totals', (S['defects_total'], L['defects_total'], N['defects_total'], S['sensitive_misses_total'], L['sensitive_misses_total'], N['sensitive_misses_total']),
          (6, 8, 1, 7, 7, 0))
    check('defects_total', res['defects_total'], dict(fills=3, S_FIFO=6, S_FIFO_LO=8, S_NOPRIO=1))
    got = sorted((d['scenario'], d['category'], d['mk_hash']) for d in res['defect_rows'])
    exp = sorted([('fills', 'fills', H2), ('fills', 'fills', H3), ('fills', 'fills', H20)] +
                 [(s, c, h) for s in ('S_FIFO', 'S_FIFO_LO') for c, h in (('posting', H2), ('size_lb', H3), ('size_ub', H2), ('removal_membership', H2),
                                                                          ('persistence', H2), ('queue_other', HX1))] +
                 [('S_FIFO_LO', 'identity', H1), ('S_FIFO_LO', 'identity', HX3), ('S_NOPRIO', 'size_lb', H3)])
    check('defect rows', got, exp)
    check('unique defects', res['unique_defects']['by_field'],
          {f: 1 for f in ('fills.fill_identity', 'fills.fill_phantom', 'fills.fill_window', 'identity.n_fills',
                          'identity.own_signer_hash_unknown', 'persistence', 'posting', 'queue_other', 'removal_membership', 'size_lb', 'size_ub')})
    check('unique defects n', res['unique_defects']['n'], 11)


def k_refusals(TT, O, F, pub):
    gt = [101, 102, 201]

    def with_attrs(df, fn=None):
        df = df.copy(); df.attrs = copy.deepcopy(TT.attrs)
        if fn:
            fn(df)
        return df
    bad = with_attrs(TT); bad.attrs['mt_truth']['columns']['act_lo_venue_ms']['clock'] = T.BOT
    refused('K clock registry', lambda: T.score(O, F, bad, pub, gt), 'clock-domain mismatch: truth column act_lo_venue_ms')
    bad2 = with_attrs(TT); bad2.attrs['mt_truth']['clock_band']['per_run']['basis']['d_lo_ms'] = -100.0
    refused('K run band vs registry', lambda: T.score(O, F, bad2, pub, gt), "a row band differs from its run's registered band")
    refused('K no size upper bound column', lambda: T.score(O.drop(columns=['x_size_ub']), F, TT, pub, gt), "lifecycle orders: missing columns ['x_size_ub']")
    refused('K no record rank column', lambda: T.score(O, F.drop(columns=['x_rec_rank']), TT, pub, gt), "lifecycle fills: missing columns ['x_rec_rank']")
    refused('K no hard lo column', lambda: T.score(O, F.drop(columns=['x_lo_hard_ms']), TT, pub, gt), "lifecycle fills: missing columns ['x_lo_hard_ms']")

    def unconv(df):
        df['act_lo_venue_ms'] = df.signed_bot_ms
    refused('K unconverted venue times', lambda: T.score(O, F, with_attrs(TT, unconv), pub, gt), 'clock-domain mismatch: act_lo_venue_ms != signed_bot_ms')
    O2 = O.copy(); O2.loc[0, 'post_lo_ms'] = O2.loc[0, 'post_lo_ms'] * 1_000_000
    refused('K ns times', lambda: T.score(O2, F, TT, pub, gt), 'clock/unit mismatch: orders.post_lo_ms')
    dup = with_attrs(pd.concat([TT, TT[TT.cid == 1].assign(cid=99)], ignore_index=True))
    refused('K duplicate truth hash', lambda: T.score(O, F, dup, pub, gt), 'duplicate order hash in truth')
    O3 = O.copy(); O3.loc[0, 'mk_hash'] = '0x' + 'f' * 64
    refused('K hash not public', lambda: T.score(O3, F, TT, pub, gt), 'row for a hash not in the public data')
    O7 = O.copy(); O7.loc[0, 'mk_hash'] = H1.replace('0x', '0X')
    refused('K lifecycle hash format', lambda: T.score(O7, F, TT, pub | {H1.replace('0x', '0X')}, gt), 'lifecycle orders hash format')
    F3 = F.copy(); F3.loc[0, 'mk_hash'] = '0x' + 'e' * 64
    refused('K fill hash not public', lambda: T.score(O, F3, TT, pub, gt), 'fills row for a hash not in the public data')
    O4 = O.copy(); O4.loc[0, 'post_class'] = 'certain'
    refused('K unknown class', lambda: T.score(O4, F, TT, pub, gt), 'unknown class in post_class')
    O5 = O.copy(); O5.loc[0, 'removal_set'] = 'exhausted|cancelled'
    refused('K unknown removal member', lambda: T.score(O5, F, TT, pub, gt), 'unknown removal member')
    O6_ = O.copy(); O6_.loc[0, 'mid'] = 999
    refused('K market outside GT', lambda: T.score(O6_, F, TT, pub, gt), 'outside the GT markets')
    refused('K no registry', lambda: T.score(O, F, TT.copy().pipe(lambda d: (setattr(d, 'attrs', {}), d)[1]), pub, gt), 'no clock registry')
    # build refusals
    e = export(); e['core_orders_v6'].loc[0, 'size'] = 6.
    refused('K kernel/core disagree', lambda: T.build_truth(e), 'kernel and core disagree')
    e = export(); e['kernel_fills_v6'] = pd.concat([e['kernel_fills_v6'], pd.DataFrame([dict(cid=1, at=B + 2900, match_ms=B + 3000, size=5., price=.4, status='Confirmed')])],
                                                   ignore_index=True)
    refused('K empty band', lambda: T.build_truth(e), 'clock band EMPTY')
    e = export(); e['matches'].loc[0, 'maker_hash'] = '0x' + 'A' * 64
    refused('K malformed own hash', lambda: T.build_truth(e), 'not 0x + 64 lowercase hex')
    e = export(); e['matches'] = pd.concat([e['matches'], e['matches'].iloc[[0]]], ignore_index=True)
    refused('K duplicate own leg', lambda: T.build_truth(e), 'duplicate own leg')


def p3_link_multiset():
    """c14 shares c4's (market, side, tick, filled 10, Confirmed second); only the leg split (5 + 5 vs 4 + 6) separates them."""
    e = export(); H14 = hh(14)
    e['kernel_orders_v6'] = pd.concat([e['kernel_orders_v6'], pd.DataFrame([dict(cid=14, market=101, up=True, price=41., size=10., t_place=B + 8500, refused=None,
                                       t_signed=B + 8500.25, t_accepted=B + 8530, t_CancelPendingAudit=NAN, t_UserCancel=NAN)])], ignore_index=True)
    e['core_orders_v6'] = pd.concat([e['core_orders_v6'], pd.DataFrame([dict(market=101, id=99, lane=0, tick=41, size=10., t_req=B + 8499, t_Accepted=B + 8530,
                                     t_CancelDone=NAN, t_Rejected=NAN, t_cancel=NAN, filled=10., cid=14, refused=None, up=True)])], ignore_index=True)
    e['kernel_fills_v6'] = pd.concat([e['kernel_fills_v6'], pd.DataFrame([dict(cid=14, at=B + a, match_ms=B + m, size=5., price=.41, status=s) for a, m, s in
                                      ((9150, 9000, 'Matched'), (9200, 9000, 'Matched'), (10150, 10000, 'Confirmed'), (10250, 10000, 'Confirmed'))])], ignore_index=True)
    e['matches'] = pd.concat([e['matches'], pd.DataFrame([dict(mid=101, exec_s=int((B + 10000) // 1000), tx=t, maker=OUR, maker_hash=H14, yes_side='bid', ptick=41,
                                                               shares=5.) for t in ('t14a', 't14b')])], ignore_index=True)
    Tr, _ = T.build_truth(e); o = Tr[Tr.cid >= 0].set_index('cid')
    check('P3 multiset link', {c: (o.mk_hash[c], o.link_status[c], o.status[c]) for c in (4, 14)},
          {4: (H4, 'unique_by_time', 'ok'), 14: (H14, 'unique_by_time', 'ok')})


def run_all():
    Tr, rec = p1_build()
    p3_link_multiset()
    res, TT, O, F, pub = p2_score(Tr, rec)
    expect_score(res)
    k_refusals(TT, O, F, pub)


# planted source mutations of pf_mt_truth.py: each must make run_all() fail (a check, a refusal or an exception)
MUTANTS = [
    ('band dropped on act_lo', "D['act_lo_venue_ms'] = D.signed_bot_ms + D0", "D['act_lo_venue_ms'] = D.signed_bot_ms"),
    ('act_hi ignores the first match', "np.fmin((D.accepted_bot_ms + D1).to_numpy(), D.mhi0.to_numpy())", "(D.accepted_bot_ms + D1).to_numpy()"),
    ('link ignores Confirmed seconds', 'r.secs <= r.conf and', 'True and'),
    ('link ignores the leg split', 'len(a) == len(b) and all(abs(x - y) <= TOL_SH for x, y in zip(a, b))', 'True'),
    ('per-run band ignored (pooled)', "b = band['per_run'][r]; d0, d1 = b['d_lo_ms'], b['d_hi_ms']",
     "d0, d1 = max(v['d_lo_ms'] for v in band['per_run'].values()), min(v['d_hi_ms'] for v in band['per_run'].values())"),
    ('run band vs registry unchecked', 'refuse(((T.d_lo_ms - rlo).abs() <= 1e-9).all() and', 'refuse(True or'),
    ('size ub unchecked', 'ubv = ub_st & (t.size_sh.to_numpy() > ub + TOL_SH)', 'ubv = np.zeros(len(own), dtype=bool)'),
    ('queue tie-break by mk_idx', "fills.sort_values(['mk_hash', 'hi_ms', 'x_rec_rank']", "fills.sort_values(['mk_hash', 'hi_ms', 'mk_idx']"),
    ('lifecycle completeness unchecked', "refuse(not miss, f'incomplete lifecycle", "refuse(True, f'incomplete lifecycle"),
    ('censored windows scored', "T = T0[~T0.mid.isin(cen)].set_index('mk_hash')", "T = T0.set_index('mk_hash')"),
    ('multi-fill lo order unchecked', 'if a >= 0 and a > b]', 'if False]'),
    ('observed fill window on the refined lo', 'flo, fhi = _rint(Fo.x_lo_hard_ms, Fo.hi_ms)', 'flo, fhi = _rint(Fo.lo_ms, Fo.hi_ms)'),
    ('refined fill window = observed', 'fo_r, fg_r = _rel(tlo, thi, rlo_r, rhi_r)', 'fo_r, fg_r = _rel(tlo, thi, flo, fhi)'),
    ('N4 hard test on the refined lo', 'los = sorted(float(x) for x in f.x_lo_hard_ms)', 'los = sorted(float(x) for x in f.lo_ms)'),
    ('signer compared case-sensitively', 'own.mk_signer.astype(str).str.lower().to_numpy() != OUR', 'own.mk_signer.to_numpy() != OUR'),
    ('own-signer filter case-sensitive', '(O.mk_signer.astype(str).str.lower() == OUR)', '(O.mk_signer == OUR)'),
    ('own public legs case-sensitive', 'L = M[M.maker.str.lower() == our].copy()', 'L = M[M.maker == our].copy()'),
    ('dry gate accepts a missing target', "refuse(p.exists(), f'dry gate", "refuse(True, f'dry gate"),
    ('dry gate accepts a differing table', "refuse(same and reg_same, 'dry gate", "refuse(True, 'dry gate"),
    ('partial fill called filled', 'np.select([full & filled, ack,', 'np.select([filled, ack,'),
    ('cancel ack without d_hi', '[D.mhi1, D.cancel_ack_bot_ms + D1], np.nan)', '[D.mhi1, D.cancel_ack_bot_ms + D0], np.nan)'),
    ('presence end on d_hi', 'phi = (D.cancel_req_bot_ms + D0).where(plo.notna())', 'phi = (D.cancel_req_bot_ms + D1).where(plo.notna())'),
    ('fill price check off', 'filled & D.pbad.fillna(False).astype(bool)', 'filled & False'),
    ('sensitive counted as defect', 'defects=nf if c in (C.OBSERVED, C.CONSTRAINED) else 0', 'defects=nf if c in (C.OBSERVED, C.CONSTRAINED, C.SENSITIVE) else 0'),
    ('disjoint misses truth-after-claim', 'dis = (thi < rlo) | (tlo > rhi)', 'dis = (thi < rlo)'),
    ('inside ignores the upper end', 'ins = (rlo <= tlo) & (thi <= rhi)', 'ins = (rlo <= tlo)'),
    ('size_lb test reversed', 'lbv = t.size_sh.to_numpy() < own.size_lb.to_numpy() - TOL_SH', 'lbv = t.size_sh.to_numpy() > own.size_lb.to_numpy() + TOL_SH'),
    ('exhausted-only = contains exhausted', "only_ex = np.array([s == frozenset({'exhausted'}) for s in S], dtype=bool)", "only_ex = np.array(['exhausted' in s for s in S], dtype=bool)"),
    ('uninformative members ignored', 'uninf = np.array([bool(s & UNINFORMATIVE) for s in S], dtype=bool)', 'uninf = np.zeros(len(S), dtype=bool)'),
    ('persistence upper end unchecked', 'out = claim & ((plo < llo) | (phi > lhi))', 'out = claim & (plo < llo)'),
    ('queue counts the order itself', 'hit = (lo <= tr) & (tr <= hi) & (cid != sc)', 'hit = (lo <= tr) & (tr <= hi)'),
    ('queue frame = fill upper end', 'tref = O.mk_hash.map(ff.lo_ms)', 'tref = O.mk_hash.map(ff.hi_ms)'),
    ('execution match ignores shares', 'if abs(a[i] - b[j]) <= TOL_SH:', 'if True:'),
    ('fill tick unchecked', 'bi = (f.mid != t.mid) | (f.y_side != t.y_side) | (f.y_ptick != t.y_ptick)', 'bi = (f.mid != t.mid) | (f.y_side != t.y_side)'),
    ('coverage counts every order', 'cov = T.index.isin(own.mk_hash)', 'cov = np.ones(len(T), dtype=bool)'),
    ('registry clock unchecked', 'refuse(clk in (VST, VBAND),', 'refuse(True,'),
    ('band identity unchecked', 'refuse(not (off > 1e-6).any(), f', 'refuse(True, f'),
    ('unit check off', 'refuse(not bad.any(), f\'clock/unit', 'refuse(True, f\'clock/unit'),
    ('public-hash check off', "refuse(not miss.any(), f'lifecycle", "refuse(True, f'lifecycle"),
    ('hash format check off', "refuse(fmt.all(), f'lifecycle", "refuse(True, f'lifecycle"),
    ('duplicate truth hash allowed', "refuse(hx.is_unique, f'duplicate order hash in truth", "refuse(True, f'duplicate order hash in truth"),
    ('GT market check off', "refuse(orders.mid.isin(gt).all() and fills.mid.isin(gt).all(),", "refuse(True,"),
    ('unknown removal member allowed', "refuse(not bad, f'unknown removal member", "refuse(True, f'unknown removal member"),
]


def run_mutants():
    import types
    global T
    src, orig, rows = Path(T.__file__).read_text(), T, []
    try:
        for name, old, new in MUTANTS:
            if src.count(old) != 1:
                rows.append((name, f'INVALID: pattern found {src.count(old)} times')); continue
            m = types.ModuleType('pf_mt_truth_mutant'); m.__file__ = orig.__file__
            exec(compile(src.replace(old, new), f'<mutant: {name}>', 'exec'), m.__dict__)
            T = m; FAIL.clear()
            try:
                run_all()
            except Exception as e:  # an exception kills the mutant too
                FAIL.append(f'exception {type(e).__name__}: {e}')
            rows.append((name, 'killed: ' + FAIL[0][:110] if FAIL else 'SURVIVED'))
    finally:
        T = orig; FAIL.clear()
    for name, r in rows:
        print(f'  mutant {name:38s} {r}')
    bad = [n for n, r in rows if not r.startswith('killed')]
    return len(rows) - len(bad), len(rows), bad


def main():
    run_all()
    if FAIL:
        print(f'SELFTEST FAIL ({len(FAIL)} of {N["checks"]} checks + {N["refusals"]} refusals)'); print('\n'.join(FAIL)); return 1
    nc, nr = N['checks'], N['refusals']
    k, n, bad = run_mutants()
    print(f'SELFTEST {"PASS" if not bad else "FAIL"}: {nc} exact checks (P1 build: band, link, statuses, venue times, receipt; P3 split-leg link; '
          f'P2 score: coverage, identity, posting, size, removal, persistence, queue, fill windows, defect rows), {nr} known-bad inputs refused '
          f'with the reason asserted; mutants killed {k} of {n}{"" if not bad else " - SURVIVED / INVALID: " + ", ".join(bad)}')
    return 0 if not bad else 1


if __name__ == '__main__':
    sys.exit(main())
