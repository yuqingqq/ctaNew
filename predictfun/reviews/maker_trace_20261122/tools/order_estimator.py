"""MAKER-ORDER-ESTIMATOR (P-2026-005 MAKER-TRACE). Frozen rules: predictfun/reviews/maker_trace_20261122/order_estimator_contract.md (this
module implements its sections 3-6, 8-10; the census chains of section 7 are separate tools). Plans: 20261123_maker_decision_timing,
20261123_maker_decision_logic, 20261123_maker_order_estimator; review 20261123_matched_gap_review ('ONE next priority'); USER directions
2026-11-23. DESCRIPTIVE ONLY on consumed days (Oct 24-27: four point estimates, no intervals). Not a fitted decision model, an imitation
strategy, a replay, a validation, a calibrated probability or a profitability claim; it measures no model skill.
SAMPLE   contract 3: maker-order units (day, market, address, order_hash) = every hash of the address in the S_NOPRIO lifecycle rows; take units
         (day, market, address, settlement_id) = every record whose taker is the address (fills_valued role taker); rank in a market = sha256
         '20261123|day|market|address|key' ascending; market order sha256 '20261123|day|address|market|market'; market-first round-robin, 4 per
         (address, day, kind); key columns only; every selected case kept with its status.
ORDER    contract 4: hypotheses EX / PL / EC per scenario row (removal_set, x_size_ub) and their union; size status distinguished (B) /
         distinguished_under_priority (C) / tied / conflict (X); priors P_cmd (primary representative) / P_drop / P_lot / P_clip, prior-labelled,
         prior_sensitive; acceptance window = union over scenarios [min post_lo, first hi] (S_FIFO beside), resting age [0, post_hi - post_lo],
         late (<= 1 s) / early possible (> 10 s); P_nomask candidate additions with the joint rule; ahead at acceptance over the pre-post frames
         and the front classes; front_after_partial; PL effect window and remaining; link_hypotheses (overlap / replace / replenish /
         unrelated): the single definition (chain_take imports it). Lifecycle rows are reused as accepted (never re-run); the book always
         through pf_mt_lifecycle._book.
TAKE     contract 5: record + taker-order identity; recorded (strict-earlier settle ms, econ FIFO order, ok rows, self-trades out), simultaneous
         and certain (hi < the take's hard lower end) inventory variants; actual-|position| class; preceding own maker fill; fill burst.
EPISODE  contract 6: windows per case, union-find per (day, market, address); economics from fills_valued (v_end, fee_fill, net_res, mq, rq)
         with matched_gap_r1's market row beside; the logic table, one row per (link, mechanism), with the declared status rules; contrasts.
CONTEXT  market-context features come from chain_context.py (contract 7.0) when it provides them; otherwise the contract-7.0 definitions in
         LocalContext (BTC: pf_mt_decision.load_btc / btc_feats unchanged; book: pf_mt_lifecycle._book). The receipt names the provider. A
         feature at instant x uses data <= x only (BTC grid row floor(x / 100) x 100; the last usable frame with upd_ms <= x).
CLOCKS   a record with receipt < UUID time (receipt_before_uuid) or without a UUID (hi_from_receipt) gives no bound (status, counted, kept).
OUTPUTS  contract 10, write-once (partial directory renamed at the end): RECEIPT/order_estimator/ (full run; requires the freeze) or
         RECEIPT/order_estimator_pilot_<day>_r<rank>/ (--pilot DAY RANK: the whole selection and census, the cases of one address-day, the GT
         check, and the measured projection).
GUARDS   MT_RUN=v2; consumed days only (pf_mt_common); VmHWM self-guard pf_mt_common.GUARD_MIB; CPU self-guard CPU_CAP_S; fills_valued sha256
         must equal the accepted matched_gap_r1 receipt's.
Usage (run_light, PF_SLOT=claude4): MT_RUN=v2 order_estimator.py run [--pilot DAY RANK]      selftest: order_estimator_selftest.py"""
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

RS = Path('/home/yuqing/ctaNew/predictfun/research')
TOOLS = Path(__file__).resolve().parent
for _p in (str(RS), str(TOOLS)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import pf_mt_common as C  # noqa: E402
import pf_mt_lifecycle as LC  # noqa: E402

need = C.need
VERSION = 'order_estimator_v1'
CONTRACT = C.RECEIPT_V1 / 'order_estimator_contract.md'
SELFTEST = TOOLS / 'order_estimator_selftest.py'
CHAIN_CONTEXT = TOOLS / 'chain_context.py'
BOOK_MANIFEST = C.REPO / 'predictfun/reviews/ladder_async_fast_20261122/manifest.json'
TOP3 = ('0x6Da6Cb464F92AE7aD4Ec3d239c81719Cb1D0Ae03', '0x7d32c568a9bEA1714036972601c51F4902C67091',
        '0x57FbC891E89184DA071AE9D97E745531cd7F718a')          # decision_contract.md population table, ranks 1-3
LOW3 = tuple(a.lower() for a in TOP3)
SCEN = tuple(C.SCENARIOS)                                         # S_FIFO, S_FIFO_LO, S_NOPRIO
FIFO_S = ('S_FIFO', 'S_FIFO_LO')
SEED, N_PER, N_BOOK = 20261123, 4, 19
TOL = LC.TOL_SH                                                   # 0.01 sh display tolerance (lifecycle)
POS_EPS = 1e-6
LATE_MS, EARLY_MS = 1_000, 10_000
LOT = 5.0
D_PROFILES = (0, 250, 1000)
BIN_EDGES = (50, 250, 1000)
EP_PAD_MS, PRE_MS, STEP = 30_000, 300_000, 100
MOVE = (('r_1s', 0.5), ('r_5s', 1.0), ('r_30s', 2.0))             # bps; contract 7.0 declared bins
MOVE_D = dict(MOVE)
LB_MS, BURST_N, BURST_MS, KEEP_REACT_MS = 5_000, 3, 5_000, 1_000
MAX_INSTANTS = 6_001
FV_SHA256, FV_BYTES = '2a874707cc7801d6760e2de36075870700cc0be8e72cd9a6df0213424221cde6', 1_078_229_581
FV_COLS = ('day', 'mid', 'sid', 'mk_idx', 'px_rank', 'role', 'addr', 'hash', 'y_side', 's', 'q', 'p', 'self_trade', 'settle_ms', 'm_end',
           'v_end', 'fee_fill', 'net_res', 'mq', 'rq', 'status')
CPU_CAP_S = 90.
X_FLAGS = frozenset(('capacity_contradiction', 'clock_capacity_conflict', 'capacity_bounds_void', 'queue_contradiction', 'ahead_hi_negative',
                     'fifo_front_conflict', 'fifo_reappearance', 'fifo_time_violation', 'fifo_tie_conflict', 'post_after_first_fill',
                     'identity_conflict', 'dup_hash_in_record', 'malformed_identity', 'unplaceable_leg', 'subcent_price', 'no_feasible_history'))
Q_CONFLICT = frozenset(('queue_contradiction', 'ahead_hi_negative', 'fifo_front_conflict', 'capacity_contradiction', 'clock_capacity_conflict',
                        'capacity_bounds_void'))
CLOCK_FLAGS = frozenset(('receipt_before_uuid', 'hi_from_receipt'))
NAN, INF = float('nan'), float('inf')
STATUSES = ('compatible', 'contradicted', 'timing_sensitive', 'insufficient_comparison', 'unidentified')
LOGIC_COLS = ('episode_id', 'case_key', 'link', 'link_seq', 'next_link', 'mechanism', 'decision_window_lo_ms', 'decision_window_hi_ms',
              'decision_clock', 'state_change', 'alternatives', 'proposed_tradeoff', 'observed_action', 'resulting_state', 'supporting_evidence',
              'competing_explanation', 'what_would_falsify', 'identification_status', 'conditional_status', 'evidence_grade', 'timing_sensitive',
              'status')
# declared texts per (link, mechanism): alternatives, proposed trade-off, competing explanation, what would falsify (contract 6)
TEXT = {
    ('POST', 'queue_priority_before_demand'): (
        'post at this tick now | post at another tick | wait | take liquidity',
        'an early resting order secures queue priority so later ordinary flow fills it, at the cost of exposure while resting',
        'late quote at the touch just before the fill; reaction to a price change; inventory-offsetting order',
        'resting age <= 10 s at every feasible instant (posting bound within EARLY_MS of the first fill)'),
    ('POST', 'touch_join_or_improve'): (
        'post at this tick now | post at another tick | wait | take liquidity',
        'joining / improving the touch buys fill probability at the price of queue position and selection',
        'an order resting behind the touch that the touch later reached',
        'the order behind its own-side touch at every pre-post instant'),
    ('POST', 'reaction_to_price_change'): (
        'post at this tick now | post at another tick | wait | take liquidity',
        'requoting after a BTC move keeps the quote aligned with the reference',
        'a continuous / scheduled requote unrelated to the move',
        'no BTC move (|r_5s| >= 1 bp) in the 5 s before any feasible instant'),
    ('POST', 'inventory_offset'): (
        'post at this tick now | post at another tick | wait | take liquidity',
        'quoting the side that reduces inventory lowers exposure at the cost of the other side\'s turnover',
        'symmetric two-sided quoting regardless of inventory',
        'the order adds to the recorded position at every feasible instant'),
    ('KEEP', 'retain_after_fill'): (
        'keep the remainder | cancel | reprice',
        'keeping the remainder preserves its priority for further fills against adverse selection after a fill',
        'no review in time (latency) / inattention',
        'a cancel before the next fill (not observable: the next fill proves persistence)'),
    ('END', 'pull_after_adverse_move'): (
        'keep | cancel the remainder | reprice | post again after exhaustion',
        'withdrawing after an adverse reference move avoids a stale fill at the cost of queue priority and later favourable fills',
        'exhaustion (no remainder); expiry at market end; an unrelated cancel',
        'EX proven (price priority / capacity) or no adverse BTC move in the decision window'),
    ('END', 'inventory_driven_pull'): (
        'keep | cancel the remainder | reprice | post again after exhaustion',
        'pulling the side that adds to accumulated inventory limits exposure',
        'exhaustion; a price-driven pull; expiry',
        'no inventory built on the order side in the 5 s before any instant of the decision window'),
    ('END', 'exhaustion'): (
        'keep | cancel the remainder | reprice | post again after exhaustion',
        'the order was consumed by flow; nothing to decide',
        'a remainder pulled unseen',
        'a later fill of the same hash'),
    ('LINK', 'replace_or_replenish'): (
        'post a new order after the end | keep a second resting order (ladder) | none',
        'resuming at the price after a pull / exhaustion restores participation; a pre-existing second order keeps older priority',
        'an unrelated new order; an overlapping ladder order resting since before',
        'the child shares a record with the parent or fills while the parent certainly rests (overlap certain)'),
    ('TAKE', 'inventory_unwind'): (
        'wait | post passively on the opposite side | take now',
        'paying spread, fee and impact to reduce unwanted exposure',
        'a directional opportunity; noise',
        'the take adds to |position| in every inventory variant'),
    ('TAKE', 'directional'): (
        'wait | post passively on the opposite side | take now',
        'paying to capture a directional move signalled by BTC',
        'an inventory unwind',
        'BTC moved against the take direction at every instant of the window'),
    ('TAKE', 'after_fill_burst'): (
        'wait | post passively on the opposite side | take now',
        'unwinding after a burst of own passive fills',
        'a market move; a scheduled unwind',
        'no own maker fill possibly in the 5 s before the window'),
}


class Refused(ValueError):
    pass


def refuse(ok, msg):
    if not ok:
        raise Refused(msg)


# ------------------------------------------------------------------ small helpers ------------------------------------------------------------------
def _split(st):
    return {x for x in str(st).split('+') if x and x != C.STATUS_OK}


def _flags(s):
    return '+'.join(sorted(s)) if s else C.STATUS_OK


def _h(*parts):
    return hashlib.sha256('|'.join(str(p) for p in parts).encode()).hexdigest()


def _sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for ch in iter(lambda: f.read(1 << 22), b''):
            h.update(ch)
    return h.hexdigest()


def _fin(x):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return NAN
    return x if math.isfinite(x) else NAN


def _r(x, nd=3):
    x = _fin(x)
    return None if np.isnan(x) else round(x, nd)


def width_bin(w):
    """precision bin of a window width (ms); NaN / negative = unbounded."""
    w = _fin(w)
    if np.isnan(w) or w < 0:
        return 'unbounded'
    for e, lab in zip(BIN_EDGES, ('<=50', '50-250', '250-1000')):
        if w <= e:
            return lab
    return '>1000'


def side_sign(y_side):
    return 1 if y_side == 'bid' else -1


def vmhwm_mib():
    return LC.vmhwm_mib()


# ------------------------------------------------------------------ the sample (contract 3) ------------------------------------------------------------------
def select_cases(keys, days, addrs=LOW3):
    """keys: DataFrame [kind ('order' | 'take'), day, mid, addr (lowercase), key] - key columns only. -> (selected, manifest): the
    deterministic market-first round-robin of contract 3; the manifest has one row per (kind, address, day)."""
    need(list(keys.columns)[:5] == ['kind', 'day', 'mid', 'addr', 'key'] and len(keys.columns) == 5,
         'select_cases takes the five key columns only (selection precedes any other column)')
    need(set(keys.kind) <= {'order', 'take'} and set(keys.addr) <= set(addrs), 'select_cases: unknown kind / address')
    rows, man = [], []
    for kind in ('order', 'take'):
        for a in addrs:
            for d in days:
                g = keys[(keys.kind == kind) & (keys.addr == a) & (keys.day == d)]
                per = {}
                for m, gm in g.groupby('mid', sort=True):
                    ks = sorted(set(gm.key.astype(str)))
                    per[int(m)] = sorted(ks, key=lambda k, m=int(m): _h(SEED, d, m, a, k))
                morder = sorted(per, key=lambda m: _h(SEED, d, a, m, 'market'))
                pick, r = [], 0
                while len(pick) < N_PER and any(len(per[m]) > r for m in morder):
                    for m in morder:
                        if len(pick) < N_PER and len(per[m]) > r:
                            pick.append((m, per[m][r], r))
                    r += 1
                for i, (m, k, rr) in enumerate(pick):
                    rows.append(dict(kind=kind, day=d, mid=m, addr=a, key=k, rank=LOW3.index(a) + 1, pick=i, round=rr,
                                     market_pos=morder.index(m), key_sha=_h(SEED, d, m, a, k), market_sha=_h(SEED, d, a, m, 'market')))
                n_c = sum(len(v) for v in per.values())
                man.append(dict(kind=kind, addr=a, rank=LOW3.index(a) + 1, day=d, n_candidates=n_c, n_markets=len(per), n_selected=len(pick),
                                status='ok' if len(pick) == N_PER else ('no_candidates' if n_c == 0 else 'fewer_candidates')))
    cols = ['kind', 'day', 'mid', 'addr', 'key', 'rank', 'pick', 'round', 'market_pos', 'key_sha', 'market_sha']
    return pd.DataFrame(rows, columns=cols), pd.DataFrame(man)


# ------------------------------------------------------------------ order hypotheses (contract 4.1 - 4.3, 4.5) ------------------------------------------------------------------
def rows3_of(orders):
    """lifecycle orders (all scenario rows) -> {hash: {scenario: row dict}}; refuses a hash without its three rows."""
    out = defaultdict(dict)
    for r in orders.to_dict('records'):
        need(r['scenario'] in SCEN, f'unknown scenario {r["scenario"]!r}')
        need(r['scenario'] not in out[r['mk_hash']], f'duplicate scenario row for {r["mk_hash"]}')
        out[r['mk_hash']][r['scenario']] = r
    for h, d in out.items():
        need(set(d) == set(SCEN), f'hash {h}: scenario rows {sorted(d)}')
    return dict(out)


def clip_counts(orders):
    """S_FIFO size_exact counts per lowercase signer (rounded 0.01) for P_clip."""
    f = orders[(orders.scenario == 'S_FIFO') & np.isfinite(orders.size_exact.to_numpy(float))]
    out = defaultdict(Counter)
    for g, v in zip(f.mk_signer.str.lower(), f.size_exact.to_numpy(float)):
        out[g][round(v, 2)] += 1
    return out


def clip_for(counter, own):
    """leave-one-out modal repeated clip: the most frequent other size_exact occurring >= 2 times (ties -> the smaller), else NaN."""
    if not counter:
        return NAN
    c = Counter(counter)
    if np.isfinite(own):
        c[round(own, 2)] -= 1
    best = max(c.values()) if c else 0
    if best < 2:
        return NAN
    return float(min(v for v, n in c.items() if n == best))


def order_core(r3, fl, settle=None, clip=NAN, presorted=False):
    """One order from its three lifecycle rows (r3) and its fills (fl: life fills rows of the hash; presorted = already in (hi_ms, x_rec_rank,
    sid) order). settle: {sid: settle_ms} of its legs (the receipt-earlier-than-UUID test; None = not available, flag clock_unchecked).
    -> dict of the contract-4 fields (no book)."""
    need(set(r3) == set(SCEN), 'order_core needs the three scenario rows')
    b = r3['S_NOPRIO']
    for s in SCEN:
        need(r3[s]['mk_hash'] == b['mk_hash'] and int(r3[s]['first_hi_ms']) == int(b['first_hi_ms'])
             and abs(float(r3[s]['cum_shares']) - float(b['cum_shares'])) < 1e-9 and int(r3[s]['removal_lo_ms']) == int(b['removal_lo_ms']),
             f'scenario rows of {b["mk_hash"]} disagree on identity / fills')
    cum, n = float(b['cum_shares']), int(b['n_fills'])
    first_hi, last_hi = int(b['first_hi_ms']), int(b['last_hi_ms'])
    flags = set()
    for s in SCEN:
        flags |= _split(r3[s]['status'])
    if fl is None:                                                   # the own-order check reads no fills (contract 9)
        flags.add('fills_not_read')
        fl = pd.DataFrame({c: pd.Series(dtype=t) for c, t in (('hi_ms', 'int64'), ('x_rec_rank', 'int64'), ('sid', 'str'), ('status', 'str'),
                                                               ('x_lo_hard_ms', 'int64'), ('shares', 'float64'), ('lo_ms', 'int64'))})
    else:
        fl = fl.sort_values(['hi_ms', 'x_rec_rank', 'sid'], kind='stable') if (len(fl) > 1 and not presorted) else fl
        if len(fl) != n:
            flags.add('fills_mismatch')
    fst = [_split(x) for x in fl.status] if len(fl) else []
    if any('hi_from_receipt' in f for f in fst):
        flags.add('hi_from_receipt')
    if settle is None:
        flags.add('clock_unchecked')
    else:
        for sid, hi in zip(fl.sid, fl.hi_ms):
            st = settle.get(sid)
            if st is not None and int(st) > 0 and int(hi) < int(st):
                flags.add('receipt_before_uuid')
    clock_bad = bool(flags & CLOCK_FLAGS)
    feas, ub = {}, {}
    for s in SCEN:
        r = r3[s]; rs = set(str(r['removal_set']).split('|'))
        u = _fin(r['x_size_ub']); ub[s] = u
        room = np.isnan(u) or (u - cum > TOL)
        f = set()
        if 'exhausted' in rs:
            f.add('EX')
        if 'cancelled_or_expired' in rs and room:
            f.add('PL')
        if 'end_censored' in rs and room:
            f.add('EC')
        feas[s] = f
    union = set().union(*feas.values())
    if not union:
        flags.add('no_feasible_history')
    conflict = bool(flags & X_FLAGS)
    if conflict:
        st = 'conflict'
    elif clock_bad:
        st = 'clock_refused'                                         # contract 4.8: feasibility shown, no claim
    elif len(union) == 1 and all(feas[s] == union for s in SCEN):
        st = 'distinguished'
    elif len(feas['S_FIFO']) == 1 and feas['S_FIFO'] == feas['S_FIFO_LO'] and len(feas['S_NOPRIO']) >= 2:
        st = 'distinguished_under_priority'
    else:
        st = 'tied'
    grade = dict(distinguished='B_constrained', distinguished_under_priority='C_priority', tied='E_unresolved', conflict='X_conflict',
                 clock_refused='X_conflict')[st]
    hist_ub = {}
    for h in ('PL', 'EC'):
        us = [ub[s] for s in SCEN if h in feas[s]]
        hist_ub[h] = (NAN if any(np.isnan(x) for x in us) else max(us)) if us else NAN
    ub_all = [ub[s] for s in SCEN]
    ub_u = NAN if any(np.isnan(x) for x in ub_all) else max(ub_all)
    plo = [int(r3[s]['post_lo_ms']) for s in SCEN]
    post_hi = int(b['post_hi_ms'])
    post_lo_u = -1 if min(plo) < 0 else min(plo)
    rest_ub = float(post_hi - post_lo_u) if post_lo_u >= 0 else NAN
    if post_lo_u < 0:
        pcls = 'window_unbounded'
    elif rest_ub <= LATE_MS:
        pcls = 'late_post'
    elif rest_ub > EARLY_MS:
        pcls = 'early_possible'
    else:
        pcls = 'intermediate'
    rlo = int(b['removal_lo_ms'])
    rhi = {s: int(r3[s]['removal_hi_ms']) for s in SCEN}
    end_hi = -1 if any(v < 0 for v in rhi.values()) else max(rhi.values())
    pl_s = [rhi[s] for s in SCEN if 'PL' in feas[s]]
    pl_hi = (-1 if any(v < 0 for v in pl_s) else max(pl_s)) if pl_s else -2          # -2: PL infeasible
    xlh = fl.x_lo_hard_ms.to_numpy(np.int64) if len(fl) else np.zeros(0, np.int64)
    hi = fl.hi_ms.to_numpy(np.int64) if len(fl) else np.zeros(0, np.int64)
    q = fl.shares.to_numpy(float) if len(fl) else np.zeros(0)
    done = np.cumsum(q)
    rem_lo = cum - done                                        # its later fills (observed)
    rem_hi = (ub_u - done) if np.isfinite(ub_u) else np.full(len(q), NAN)
    sides = {r3[s]['y_side'] for s in SCEN}
    need(len(sides) == 1, 'scenario rows disagree on side')
    ys = sides.pop()
    core = dict(hash=b['mk_hash'], signer=str(b['mk_signer']), y_side=ys, s=side_sign(ys), side_i=LC.SIDE.get(ys, -1), tick=int(b['y_ptick']),
                n=n, cum=cum, first_hi=first_hi, last_hi=last_hi, flags=flags, conflict=conflict, clock_bad=clock_bad, feas=feas, union=union,
                size_status=st, grade=grade, ub=ub, ub_u=ub_u, hist_ub=hist_ub, size_exact_fifo=_fin(r3['S_FIFO']['size_exact']),
                post_lo=dict(zip(SCEN, plo)), post_lo_u=post_lo_u, post_lo_fifo=int(r3['S_FIFO']['post_lo_ms']), post_hi=post_hi, rest_ub=rest_ub,
                post_class=pcls, post_class_s=str(b['post_class']), removal_class_s=str(b['removal_class']), removal_lo=rlo, removal_hi=rhi,
                end_hi=end_hi, pl_hi=pl_hi,
                removal_set=dict((s, str(r3[s]['removal_set'])) for s in SCEN), removal_method=dict((s, str(r3[s]['removal_method'])) for s in SCEN),
                ahead_lo_fifo=_fin(r3['S_FIFO']['ahead_lo']), ahead_hi_fifo=_fin(r3['S_FIFO']['ahead_hi']),
                first_lo_hard=int(xlh[0]) if len(xlh) else -1, last_lo_hard=rlo, fill_hi=hi, fill_lo_hard=xlh, fill_q=q,
                fill_sid=fl.sid.to_numpy(object) if len(fl) else np.zeros(0, object),
                fill_ahi_fifo=fl.x_ahead_hi_S_FIFO.to_numpy(float) if len(fl) and 'x_ahead_hi_S_FIFO' in fl.columns else np.full(len(q), NAN),
                fill_lo=fl.lo_ms.to_numpy(np.int64) if len(fl) else np.zeros(0, np.int64), rem_lo=rem_lo, rem_hi=rem_hi,
                n_front_after_partial=int(np.sum(fl.x_ahead_hi_S_FIFO.to_numpy(float)[1:] == 0.)) if len(fl) > 1 and 'x_ahead_hi_S_FIFO' in fl.columns else 0)
    core['fill_w_hard'] = np.where(xlh >= 0, hi - xlh, -1).astype(np.int64)
    core['fill_w_ev'] = np.where(core['fill_lo'] >= 0, hi - core['fill_lo'], -1).astype(np.int64)
    core['queue_fifo'] = ('conflicting' if flags & Q_CONFLICT else 'behind_at_first_fill' if core['ahead_lo_fifo'] > TOL else
                          'front_at_first_fill_fifo' if core['ahead_hi_fifo'] <= TOL else 'unidentified')
    core['priors'] = priors(core, clip=clip)
    return core


def summary_row(d, mid, rank, core):
    """the coverage fields of one order (census and sample share them)."""
    pr = core['priors']
    return dict(day=d, mid=mid, rank=rank, hash=core['hash'], n_fills=core['n'], cum=core['cum'], size_status=core['size_status'],
                union='|'.join(sorted(core['union'])), conflict=core['conflict'], clock_bad=core['clock_bad'],
                post_w=(core['post_hi'] - core['post_lo_u']) if core['post_lo_u'] >= 0 else -1, post_class=core['post_class'],
                post_w_fifo=(core['post_hi'] - core['post_lo_fifo']) if core['post_lo_fifo'] >= 0 else -1, pl='PL' in core['union'],
                pl_w=(core['pl_hi'] - core['removal_lo']) if (core['pl_hi'] >= 0 and core['removal_lo'] >= 0) else -1,
                queue_fifo=core['queue_fifo'], n_front_after_partial=core['n_front_after_partial'], P_cmd=pr['P_cmd'][0], P_lot=pr['P_lot'][0],
                P_clip=pr['P_clip'][0], prior_sensitive3=pr['prior_sensitive3'],
                priors_feasible=all(pr[k][0] in core['union'] for k in ('P_cmd', 'P_lot', 'P_clip')) or not core['union'],
                fill_w_hard=list(core['fill_w_hard']), fill_w_ev=list(core['fill_w_ev']), order_flags=_flags(core['flags']))


def priors(core, clip=NAN, drop_r=NAN, with_drop=False):
    """contract 4.2: (history, size) per prior; P_drop only when the book was read (with_drop)."""
    U, cum, hub = core['union'], core['cum'], core['hist_ub']

    def fits(sz, h):
        u = hub.get(h, NAN)
        return h in U and (np.isnan(u) or sz <= u + 1e-9)

    def larger(sz):
        for h in ('PL', 'EC'):
            if fits(sz, h):
                return (h, round(sz, 6))
        return None
    if 'EX' in U:
        cmd = ('EX', cum)
    elif 'EC' in U:
        cmd = ('EC', NAN)
    elif 'PL' in U:
        cmd = ('PL', NAN)
    else:
        cmd = ('none', NAN)
    m = math.ceil((cum - TOL) / LOT) * LOT
    lot = cmd if abs(m - cum) <= TOL else (larger(m) or cmd)
    if np.isnan(clip):
        clp = cmd
    elif abs(clip - cum) <= TOL:
        clp = ('EX', cum) if 'EX' in U else cmd
    elif clip > cum + TOL:
        clp = larger(clip) or cmd
    else:
        clp = cmd
    out = dict(P_cmd=cmd, P_lot=lot, P_clip=clp)
    if with_drop:
        if np.isfinite(drop_r) and drop_r > TOL and 'PL' in U:
            u = hub.get('PL', NAN)
            sz = cum + (min(drop_r, u - cum) if np.isfinite(u) else drop_r)
            out['P_drop'] = ('PL', round(sz, 6)) if sz > cum + TOL else cmd
        else:
            out['P_drop'] = cmd
    def key(v):
        h, z = v
        return h, (None if (z is None or not np.isfinite(float(z))) else round(float(z), 2))
    ks = [k for k in ('P_cmd', 'P_drop', 'P_lot', 'P_clip') if k in out]
    out['prior_sensitive'] = len({key(out[k]) for k in ks}) > 1
    out['prior_sensitive3'] = len({key(out[k]) for k in ('P_cmd', 'P_lot', 'P_clip')}) > 1
    out['n_priors'] = len(ks)
    return out


# ------------------------------------------------------------------ links (contract 4.6: the single definition) ------------------------------------------------------------------
def _lh_fields(o):
    """the fields link_hypotheses reads, from an order_core dict or a caller's row dict (post_lo_union / last_lh / ex_any / pl_any / ec_any /
    records spellings accepted, so every tool calls the one definition)."""
    post_lo = o['post_lo_u'] if 'post_lo_u' in o else o['post_lo_union']
    last_lh = o['last_lo_hard'] if 'last_lo_hard' in o else o['last_lh']
    U = set(o['union']) if 'union' in o else {h for h, k in (('EX', 'ex_any'), ('PL', 'pl_any'), ('EC', 'ec_any')) if bool(o.get(k))}
    return dict(post_lo=int(post_lo), post_hi=int(o['post_hi']), first_hi=int(o['first_hi']), last_lh=int(last_lh),
                removal_lo=int(o['removal_lo']), end_hi=int(o['end_hi']), union=U, tick=int(o['tick']), records=o.get('records'))


def link_hypotheses(P, Q, shared_record=None):
    """P earlier, Q later (one signer and side). -> dict(feasible / relations, overlap_certain, status, gap range). shared_record None: from
    both dicts' 'records' (sid sets) when present, else False."""
    p, q = _lh_fields(P), _lh_fields(Q)
    if shared_record is None:
        shared_record = bool(p['records'] and q['records'] and (set(p['records']) & set(q['records'])))
    end_hi = p['end_hi'] if p['end_hi'] >= 0 else INF
    p_last = p['last_lh']                                            # -1: unbounded (receipt-hi)
    ov_cert = bool(shared_record) or (p_last >= 0 and q['first_hi'] < p_last)
    ov_feas = ov_cert or q['post_lo'] < 0 or q['post_lo'] < end_hi
    rep = (not ov_cert) and ('PL' in p['union']) and (p['removal_lo'] < 0 or q['post_hi'] > p['removal_lo'])
    repl = (not ov_cert) and ('EX' in p['union']) and (p_last < 0 or q['post_hi'] > p_last)
    rel = [r for r, ok in (('overlap', ov_feas), ('replace', rep), ('replenish', repl), ('unrelated', True)) if ok]
    st = 'overlap_identified' if ov_cert else ('sequential_identified' if not ov_feas else 'tied')
    if bool(P.get('clock_bad')) or bool(Q.get('clock_bad')):
        st = 'clock_refused'                                            # contract 4.8: a refused clock bounds nothing
    g_lo = (q['post_lo'] - end_hi) if (q['post_lo'] >= 0 and np.isfinite(end_hi)) else NAN
    g_hi = (q['post_hi'] - p_last) if p_last >= 0 else NAN
    cls = (Q.get('post_class_s'), P.get('removal_class_s'))
    scen_dep = None if None in cls else not (cls[0] == C.CONSTRAINED and cls[1] == C.CONSTRAINED)
    return dict(feasible=tuple(rel), relations='|'.join(rel), overlap_certain=ov_cert, overlap_feasible=bool(ov_feas), replace_feasible=bool(rep),
                replenish_feasible=bool(repl), status=st, gap_lo_ms=g_lo, gap_hi_ms=g_hi, scenario_dependent=scen_dep,
                kind='refill' if p['tick'] == q['tick'] else 'reprice', d_ticks=int(q['tick'] - p['tick']))


# ------------------------------------------------------------------ book side of an order (contract 4.2 P_drop, 4.3 P_nomask, 4.4) ------------------------------------------------------------------
def touch_arrays(B):
    """own-side touch per usable frame: best bid tick (-1 none) / best ask tick (NT none) and unknown flags (an unknown cell better than the best
    known level)."""
    F, NT = B['F'], LC.NT
    if F == 0:
        return dict(bid=np.zeros(0, int), ask=np.zeros(0, int), unk_b=np.zeros(0, bool), unk_a=np.zeros(0, bool))
    bid, ask = B['shown'][:, 0, :], B['shown'][:, 1, :]
    pb, pa_ = bid > TOL, ask > TOL
    bb = np.where(pb.any(1), NT - 1 - np.argmax(pb[:, ::-1], 1), -1)
    ba = np.where(pa_.any(1), np.argmax(pa_, 1), NT)
    ix = np.arange(NT)[None, :]
    unk_b = (np.isnan(bid) & (ix > bb[:, None])).any(1)
    unk_a = (np.isnan(ask) & (ix < ba[:, None])).any(1)
    return dict(bid=bb, ask=ba, unk_b=unk_b, unk_a=unk_a)


def fill_frames(fills, B):
    """per life fill (all makers): fb_h (last frame before x_lo_hard; -1), fa (first frame >= hi; F = none), E (evidence frame index; -1)."""
    t, F = B['t'], B['F']
    xl = fills.x_lo_hard_ms.to_numpy(np.int64); hi = fills.hi_ms.to_numpy(np.int64)
    fb_h = np.where(xl >= 0, np.searchsorted(t, xl, 'left') - 1, -1)
    fa = np.searchsorted(t, hi, 'left')
    E = pd.Index(B['fid']).get_indexer(fills.x_evidence_fid.to_numpy(np.int64)) if F else np.full(len(fills), -1)
    return dict(fb_h=fb_h.astype(np.int64), fa=fa.astype(np.int64), E=np.asarray(E, np.int64))


def cell_prints(fills, FF, B, side_i, tick):
    """prints at one cell: conservative (every frame f whose interval (f-1, f] may hold the fill: [max(fb_h + 1, 1), min(fa, F - 1)], the
    lifecycle's _cells rule) and at the evidence frame (E, else fa) with their counts."""
    F = B['F']
    m = (fills.y_side.map(LC.SIDE).fillna(-1).to_numpy(int) == side_i) & (fills.y_ptick.to_numpy(int) == tick)
    q = fills.shares.to_numpy(float)[m]
    cons = np.zeros(F + 1); ev = np.zeros(F + 1); nev = np.zeros(F + 1, np.int64)
    if F == 0 or not m.any():
        return cons[:F], ev[:F], nev[:F]
    st = np.maximum(FF['fb_h'][m] + 1, 1); en = np.minimum(FF['fa'][m], F - 1)
    ok = st <= en
    np.add.at(cons, st[ok], q[ok]); np.add.at(cons, en[ok] + 1, -q[ok])
    e = np.where(FF['E'][m] >= 1, FF['E'][m], FF['fa'][m])
    ok = (e >= 1) & (e < F)
    np.add.at(ev, e[ok], q[ok]); np.add.at(nev, e[ok], 1)
    return np.cumsum(cons)[:F], ev[:F], nev[:F]


def window_frames(B, lo_ms, hi_ms, ws_ms):
    """pre-post frames: from the frame at / before lo (unbounded lo -> the first frame >= max(ws - 300 s, first frame)) to the last frame
    before hi -> (fa, fz, flags)."""
    t, F = B['t'], B['F']
    fl = set()
    if F == 0:
        return 0, -1, {'no_frames'}
    if lo_ms < 0:
        fl.add('window_unbounded')
        fa = int(np.searchsorted(t, max(ws_ms - PRE_MS, int(t[0])), 'left'))
    else:
        fa = int(np.searchsorted(t, lo_ms, 'right')) - 1
        if fa < 0:
            fl.add('no_frame_before_window'); fa = 0
    fz = int(np.searchsorted(t, hi_ms, 'left')) - 1
    if fz < fa:
        fl.add('no_frames')
    return fa, fz, fl


def cell_arrays(B, fills, FF, side_i, tick):
    """one cell's book arrays: shown, prints (conservative and at the evidence frame), the visible net addition of every frame interval
    (f, f+1] (add[f] = shown[f+1] - shown[f] + evidence prints in it; NaN / across a gap = unknown) and its tolerance."""
    F = B['F']
    shown = B['shown'][:, side_i, tick]
    cons, ev, nev = cell_prints(fills, FF, B, side_i, tick)
    add = np.full(max(F - 1, 0), NAN); bad = np.ones(max(F - 1, 0), bool); tol = np.full(max(F - 1, 0), TOL)
    if F >= 2:
        add = shown[1:] - shown[:-1] + ev[1:]
        bad = np.isnan(add) | B['gap'][1:]
        tol = TOL * (1 + nev[1:]) + 1e-9
    return dict(shown=shown, cons=cons, ev=ev, nev=nev, add=add, bad=bad, tol=tol)


def nomask_candidates(core, CA, fa, fz):
    """single-order P_nomask candidates: frame f in [fa, fz] (posting in (t_f, t_f+1]) whose visible net addition >= cum - tol; unknown
    intervals (gap / unknown cell) are never candidates -> (candidate frames, incomplete flag)."""
    n = len(CA['add'])
    if fz < fa or n == 0:
        return np.zeros(0, np.int64), False
    f = np.arange(fa, min(fz, n - 1) + 1)
    if len(f) == 0:
        return np.zeros(0, np.int64), False
    ok = ~CA['bad'][f] & (CA['add'][f] >= core['cum'] - CA['tol'][f])
    return f[ok], bool(CA['bad'][f].any())


def cell_nomask(cell_cores, B, CA, ws_ms):
    """P_nomask over every order at one cell (contract 4.3): single-order candidates; forced claims (an order with exactly one candidate
    claims its cum there; a candidate stays only if the addition covers the other orders' forced claims plus its cum); the joint
    representative: in ascending (first fill hi, hash) each order takes the latest candidate whose remaining visible addition covers its cum
    (one addition is never assigned in full to two orders). -> {hash: dict(cand, inc, rep, status)}"""
    res = {}
    for oc in cell_cores:
        fa, fz, _ = window_frames(B, oc['post_lo_u'], oc['post_hi'], ws_ms)
        cand, inc = nomask_candidates(oc, CA, fa, fz)
        res[oc['hash']] = dict(cand=cand, inc=inc, rep=-1, core=oc)
    live = {h: x for h, x in res.items() if not x['core']['conflict'] and x['core']['post_lo_u'] >= 0}
    forced = np.zeros(len(CA['add']) + 1)
    for x in live.values():
        if len(x['cand']) == 1:
            forced[x['cand'][0]] += x['core']['cum']
    for x in live.values():
        c = x['cand']
        if len(c):
            own = x['core']['cum'] if len(c) == 1 else 0.             # its own forced claim is not a competitor
            keep = CA['add'][c] >= forced[c] - own + x['core']['cum'] - CA['tol'][c]
            x['joint_removed'] = bool((~keep).any())
            x['cand'] = c[keep]
    cap = np.where(CA['bad'], 0., np.nan_to_num(CA['add']))
    for h in sorted(live, key=lambda h: (live[h]['core']['first_hi'], h)):
        x = live[h]
        for f in x['cand'][::-1]:
            if cap[f] >= x['core']['cum'] - CA['tol'][f]:
                x['rep'] = int(f); cap[f] -= x['core']['cum']
                break
    for h, x in res.items():
        oc = x['core']
        if oc['post_lo_u'] < 0:
            x['status'] = 'window_unbounded'
        elif oc['conflict']:
            x['status'] = 'conflict'
        elif len(x['cand']) == 0:
            x['status'] = 'incomplete_gap' if x['inc'] else 'no_visible_addition'
        elif x['rep'] < 0:
            x['status'] = 'joint_unassignable'
        else:
            x['status'] = 'single_candidate' if len(x['cand']) == 1 else 'multiple'
    return res


def order_book(core, B, touch, fills, FF, CA, NM, ws_ms):
    """contract 4.2 (P_drop), 4.3 (P_nomask: NM = cell_nomask of its cell), 4.4 (ahead at acceptance, front classes, time to touch) for one
    order; CA = cell_arrays of its cell."""
    out = dict(book_flags=set())
    side_i, tick = core['side_i'], core['tick']
    if side_i < 0 or B['F'] == 0 or CA is None:
        out['book_flags'].add('no_frames')
        out.update(ahead_lo=NAN, ahead_hi=NAN, front_class='unidentified', nomask_status='no_frames', nomask_n=0, nomask_rep_lo=-1,
                   nomask_rep_hi=-1, nomask_span_lo=-1, nomask_span_hi=-1, nomask_rest_lo=NAN, nomask_rest_hi=NAN, nomask_class='unidentified',
                   drop_r=NAN, drop_frame_ms=-1, touch_first_ms=-1, touch_at_start=None)
        core['priors'] = priors(core, clip=core.get('clip', NAN), with_drop=True)
        return out
    t, F = B['t'], B['F']
    shown, cons = CA['shown'], CA['cons']
    fa, fz, wfl = window_frames(B, core['post_lo_u'], core['post_hi'], ws_ms)
    out['book_flags'] |= wfl
    # ---- ahead at acceptance (FIFO; history 'posted just after frame f') ----
    if fz >= fa:
        v = shown[fa:fz + 1]
        kn = v[~np.isnan(v)]
        if len(kn) < len(v):
            out['book_flags'].add('unknown_cell_in_window')
        a_lo, a_hi = (float(kn.min()), float(kn.max())) if len(kn) else (NAN, NAN)
        if B['gap'][fa + 1:fz + 1].any():
            out['book_flags'].add('gap_in_window')
    else:
        a_lo = a_hi = NAN
    out['ahead_lo'], out['ahead_hi'] = a_lo, a_hi
    alo, ahi = core['ahead_lo_fifo'], core['ahead_hi_fifo']
    if core['flags'] & Q_CONFLICT or (np.isfinite(a_hi) and a_hi <= TOL and np.isfinite(alo) and alo > TOL):
        fc = 'conflicting'
    elif not np.isfinite(a_hi):
        fc = 'unidentified'
    elif a_hi <= TOL:
        fc = 'front_at_acceptance'
    elif np.isfinite(alo) and alo > TOL:
        fc = 'behind_at_first_fill'
    elif np.isfinite(ahi) and ahi <= TOL:
        fc = 'front_reached_after_clearing' if a_lo > TOL else 'front_at_acceptance_or_reached'
    else:
        fc = 'unidentified'
    if core['post_lo_u'] < 0 and fc in ('front_at_acceptance', 'front_reached_after_clearing'):
        # an unbounded window cannot certify the state at acceptance (it may precede the first frame read)
        fc = 'front_at_acceptance_or_reached' if (np.isfinite(ahi) and ahi <= TOL) else 'unidentified'
    out['front_class'] = fc
    # ---- time to the touch ----
    own = touch['bid'] if side_i == 0 else touch['ask']
    if fz >= fa:
        j = np.flatnonzero(own[fa:fz + 1] == tick)
        out['touch_first_ms'] = int(t[fa + j[0]]) if len(j) else -1
        out['touch_at_start'] = bool(own[fa] == tick)
    else:
        out['touch_first_ms'], out['touch_at_start'] = -1, None
    # ---- P_nomask (joint, cell level) ----
    x = NM[core['hash']]
    cand, inc, rep = x['cand'], x['inc'], x['rep']
    if inc:
        out['book_flags'].add('incomplete_gap')
    if x.get('joint_removed'):
        out['book_flags'].add('joint_claim_removed_candidate')
    out['nomask_status'], out['nomask_n'] = x['status'], int(len(cand))
    if len(cand) and rep >= 0:
        f0, f1 = int(cand.min()), int(cand.max())
        out.update(nomask_rep_lo=int(t[rep]), nomask_rep_hi=int(t[rep + 1]), nomask_span_lo=int(t[f0]), nomask_span_hi=int(t[f1 + 1]))
        rlo, rhi = max(0., core['post_hi'] - t[f1 + 1]), max(0., core['post_hi'] - t[f0])
        out['nomask_rest_lo'], out['nomask_rest_hi'] = float(rlo), float(rhi)
        out['nomask_class'] = 'late_post' if rhi <= LATE_MS else ('early' if rlo > EARLY_MS else 'mixed')
    else:
        out.update(nomask_rep_lo=-1, nomask_rep_hi=-1, nomask_span_lo=-1, nomask_span_hi=-1, nomask_rest_lo=NAN, nomask_rest_hi=NAN,
                   nomask_class='unidentified')
    # ---- P_drop: the first unexplained drop after the last fill, up to the PL window end ----
    drop_r, drop_t = NAN, -1
    if 'PL' in core['union']:
        mo = np.isin(fills.sid.to_numpy(object), core['fill_sid']) & (fills.mk_hash.to_numpy(object) == core['hash'])
        g0 = int(FF['fa'][mo].max()) if mo.any() else F
        end_f = (int(np.searchsorted(t, core['pl_hi'], 'right')) - 1) if core['pl_hi'] >= 0 else F - 1
        f = np.arange(max(g0 + 1, 1), min(end_f, F - 1) + 1)
        if len(f):
            dr = shown[f - 1] - shown[f]
            unk = np.isnan(dr) | B['gap'][f]
            ex = np.maximum(np.nan_to_num(dr), 0.) - cons[f]
            j = np.flatnonzero(~unk & (ex > TOL))
            if unk[:j[0] if len(j) else len(f)].any():
                out['book_flags'].add('drop_scan_unknown')        # an unknown interval before the first visible excess
            if len(j):
                drop_r, drop_t = float(ex[j[0]]), int(t[f[j[0]]])
    out['drop_r'], out['drop_frame_ms'] = drop_r, drop_t
    core['priors'] = priors(core, clip=core.get('clip', NAN), drop_r=drop_r, with_drop=True)
    return out


# ------------------------------------------------------------------ inventory (contract 5 / 7.0) ------------------------------------------------------------------
def inv_class(b, a):
    """actual |position| class (contract 5)."""
    if abs(b) <= POS_EPS:
        return 'from_flat'
    if abs(a) <= POS_EPS:
        return 'full_close'
    if np.sign(a) == np.sign(b):
        if abs(a) < abs(b) - POS_EPS:
            return 'reduce_partial'
        return 'add' if abs(a) > abs(b) + POS_EPS else 'no_change'
    return 'reversal_reducing' if abs(a) < abs(b) - POS_EPS else 'overshoot'


REDUCING = frozenset(('reduce_partial', 'full_close', 'reversal_reducing'))


def legs_frame(fv_m, win):
    """the address's fills_valued rows in one market + their record windows (win: sid -> (x_lo_hard, lo, hi, t_status)) in econ's FIFO order;
    lot = ok, not a self-trade."""
    L = fv_m.copy()
    L['role_k'] = (L.role == 'taker').astype(int)
    L = L.sort_values(['settle_ms', 'sid', 'px_rank', 'mk_idx', 'role_k'], kind='stable').reset_index(drop=True)
    w = [win.get(x, (-1, -1, -1, 'no_window')) for x in L.sid]
    L['xlo'] = np.array([x[0] for x in w], np.int64); L['elo'] = np.array([x[1] for x in w], np.int64)
    L['whi'] = np.array([x[2] for x in w], np.int64); L['wst'] = [x[3] for x in w]
    L['lot'] = (L.status == C.STATUS_OK).to_numpy(bool) & ~L.self_trade.to_numpy(bool)
    L['sq'] = np.where(L.lot, L.s.to_numpy(float) * L.q.to_numpy(float), 0.)
    st = L.settle_ms.to_numpy(np.int64); whi = L.whi.to_numpy(np.int64)
    L['clock_bad'] = (L.wst == 'no_window').to_numpy(bool) | (L.xlo.to_numpy(np.int64) < 0) | ((st > 0) & (whi >= 0) & (whi < st)) | (st <= 0)
    return L


class Inventory:
    """recorded (econ order: settle_ms < x) and certain (window hi < x; clock-bad legs never certain, counted) positions of one address in
    one market (contract 7.0)."""

    def __init__(self, L):
        lot = L[L.lot]
        self.st = lot.settle_ms.to_numpy(np.int64); self.c_st = np.r_[0., np.cumsum(lot.sq.to_numpy(float))]
        good = ~lot.clock_bad.to_numpy(bool)
        hi = lot.whi.to_numpy(np.int64)[good]; sq = lot.sq.to_numpy(float)[good]
        o = np.argsort(hi, kind='stable')
        self.hi = hi[o]; self.c_hi = np.r_[0., np.cumsum(sq[o])]
        self.n_clock_bad = int((~good).sum())

    def recorded(self, x):
        return self.c_st[np.searchsorted(self.st, np.asarray(x, np.int64), 'left')]

    def certain(self, x):
        return self.c_hi[np.searchsorted(self.hi, np.asarray(x, np.int64), 'left')]


# ------------------------------------------------------------------ context (contract 7.0) ------------------------------------------------------------------
def instants(lo, hi):
    """candidate instants of a window: its ends and every 100 ms grid point inside (the latest MAX_INSTANTS kept, flagged by the caller)."""
    lo, hi = int(lo), int(hi)
    if hi < lo:
        return np.array([hi], np.int64)
    a = -(-lo // STEP) * STEP
    x = np.unique(np.r_[np.int64(lo), np.arange(a, hi + 1, STEP, dtype=np.int64), np.int64(hi)])
    return x[-MAX_INSTANTS:]


class LocalContext:
    """contract 7.0 definitions on the market bundle (used by the selftest and when chain_context.py is absent; the receipt names the
    provider). Interface (shared with ChainContext): btc(Mk, x), book_at(Mk, x, side_i, tick), inventory(Mk, addr, x)."""
    provider = 'local_contract_7_0'

    def __init__(self, btc_dir=None, btc_override=None):
        self.btc_dir, self.over, self._bt = btc_dir, btc_override, {}

    def _btc(self, day):
        if self.over is not None:
            return self.over(day)
        if day not in self._bt:
            import pf_mt_decision as DM
            self._bt[day] = DM.load_btc(self.btc_dir, day, Counter())
        return self._bt[day]

    def btc(self, Mk, x):
        import pf_mt_decision as DM
        f = DM.btc_feats(self._btc(Mk['day']), np.asarray(x, np.int64), int(Mk['ws_ms']), int(Mk['we_ms']))
        return dict(code=np.asarray(f['code']), r_1s=f['r_1s'], r_5s=f['r_5s'], r_30s=f['r_30s'], pf=f['pf'])

    @staticmethod
    def inventory(Mk, addr, x):
        inv = Mk['inv'][addr]
        return dict(recorded=inv.recorded(x), certain=inv.certain(x))

    @staticmethod
    def book_at(Mk, x, side_i, tick):
        B, T = Mk['B'], Mk['touch']
        x = np.asarray(x, np.int64); n = len(x)
        out = dict(dist=np.full(n, NAN), cat=np.full(n, 'missing', object), spread=np.full(n, NAN), mid=np.full(n, NAN),
                   depth=np.full(n, NAN), age=np.full(n, -1, np.int64), stale=np.ones(n, bool), bid=np.full(n, NAN), ask=np.full(n, NAN))
        if B is None or B['F'] == 0:
            return out
        j = np.searchsorted(B['t'], x, 'right') - 1
        ok = j >= 0
        jj = np.maximum(j, 0)
        bb, ba = T['bid'][jj].astype(float), T['ask'][jj].astype(float)
        hb, ha = bb >= 0, ba < LC.NT
        out['bid'] = np.where(ok & hb, bb, NAN); out['ask'] = np.where(ok & ha, ba, NAN)
        out['spread'] = np.where(ok & hb & ha, ba - bb, NAN)
        out['mid'] = np.where(ok & hb & ha, (ba + bb) / 2., NAN)
        out['age'] = np.where(ok, x - B['t'][jj], -1)
        out['stale'] = ~ok | (out['age'] > 1000)
        if side_i in (0, 1) and 0 <= tick < LC.NT:
            out['depth'] = np.where(ok, B['shown'][jj, side_i, tick], NAN)
            own, unk, has = (bb, T['unk_b'][jj], hb) if side_i == 0 else (ba, T['unk_a'][jj], ha)
            d = (own - tick) if side_i == 0 else (tick - own)
            d = np.where(has, d, -1.)                                     # an empty own side: the order would set the touch (inside)
            out['dist'] = np.where(ok & ~unk, d, NAN)
            cat = np.where(d > 0, 'behind', np.where(d == 0, 'at', 'inside')).astype(object)
            stale = out['age'] > LC.GAP_MS                                 # no usable frame within 1 s: an unknown span (lifecycle B)
            out['cat'] = np.where(~ok, 'missing', np.where(unk, 'unknown', np.where(stale, 'stale', cat))).astype(object)
        return out


CC_NAMES = ('load_market', 'btc_ret', 'book', 'touch_distance', 'level_size', 'position')


class ChainContext:
    """the bridge to chain_context.py (contract 7.0: the one definition of market context), on its documented stable names; Mk['M'] is the
    market chain_context.load_market returned. Same interface as LocalContext."""
    provider = 'chain_context'

    def __init__(self, mod):
        for nm in CC_NAMES:
            refuse(callable(getattr(mod, nm, None)), f'chain_context: required function {nm} missing (refused)')
        self.CC = mod

    def load(self, day, mid, root):
        return self.CC.load_market(day, int(mid), list(LOW3), root=root)

    def btc(self, Mk, x):
        f = self.CC.btc_ret(Mk['M'], np.asarray(x, np.int64))
        return dict(code=np.asarray(f['code']), r_1s=np.asarray(f['r_1s'], float), r_5s=np.asarray(f['r_5s'], float),
                    r_30s=np.asarray(f['r_30s'], float), pf=np.asarray(f['p_fair'], float))

    def inventory(self, Mk, addr, x):
        p = self.CC.position(Mk['M'], addr, np.asarray(x, np.int64))
        return dict(recorded=np.asarray(p['recorded'], float), certain=np.asarray(p['certain'], float))

    def book_at(self, Mk, x, side_i, tick):
        M, x = Mk['M'], np.asarray(x, np.int64)
        n = len(x)
        b = self.CC.book(M, x)
        out = dict(spread=np.asarray(b['spread'], float), mid=np.asarray(b['mid_c'], float), bid=np.asarray(b['bid_tick'], float),
                   ask=np.asarray(b['ask_tick'], float), age=np.asarray(b['age_ms']), stale=np.asarray(b['stale'], bool),
                   dist=np.full(n, NAN), cat=np.full(n, 'missing', object), depth=np.full(n, NAN))
        if side_i in (0, 1) and 0 <= tick < LC.NT:
            td = self.CC.touch_distance(M, x, side_i, tick)
            out['dist'] = np.asarray(td['ticks'], float)
            out['cat'] = np.asarray(td['cat'], object)
            out['depth'] = np.asarray(self.CC.level_size(M, x, side_i, tick)['size'], float)
        return out


def make_context(R, require_chain=False, mod=None):
    """the context provider: chain_context.py (ChainContext) when it is present with CC_NAMES, else LocalContext (refused when require_chain:
    the full run reads market context from chain_context only)."""
    prov = dict(provider='local_contract_7_0')
    if mod is None and CHAIN_CONTEXT.exists():
        try:
            import chain_context as mod                                   # noqa: F811
        except Exception as e:                                            # reported, never silent
            prov['chain_context_import_error'] = repr(e)[:300]
            mod = None
    if mod is not None:
        ctx = ChainContext(mod)
        prov.update(provider='chain_context', chain_context_version=getattr(mod, 'VERSION', None),
                    chain_context_sha256=_sha(CHAIN_CONTEXT) if CHAIN_CONTEXT.exists() else None)
    else:
        ctx = LocalContext(btc_dir=Path(R) / 'btc')
    refuse(not require_chain or prov['provider'] == 'chain_context',
           'the full run reads market context from chain_context.py only (contract 7.0); it is missing or failed to import')
    return ctx, prov


def btc_window(ctx, Mk, x, s):
    """aligned BTC categories at instants x for side s: per horizon a = s x r; with / against / flat / missing."""
    f = ctx.btc(Mk, x)
    ok = np.asarray(f['code']) == 0
    res = dict(ok=ok)
    for k, thr in MOVE:
        a = s * np.asarray(f[k], float)
        res[k] = a
        res[k + '_cat'] = np.where(~ok | np.isnan(a), 'missing', np.where(a >= thr, 'with', np.where(a <= -thr, 'against', 'flat'))).astype(object)
        res[k + '_raw'] = np.asarray(f[k], float)
    res['pf'] = np.asarray(f.get('pf', np.full(len(x), NAN)), float)
    return res


def st_all(cond, valid):
    """compatible if cond holds at every valid instant, contradicted if at none, timing_sensitive otherwise; unidentified without one."""
    v = int(np.sum(valid))
    if v == 0:
        return 'unidentified'
    c = int(np.sum(cond & valid))
    return 'compatible' if c == v else ('contradicted' if c == 0 else 'timing_sensitive')


def cats(a):
    return '|'.join(sorted({str(x) for x in a})) if len(a) else ''


def rng(a, nd=2):
    a = np.asarray(a, float); a = a[np.isfinite(a)]
    return f'[{a.min():.{nd}f},{a.max():.{nd}f}]' if len(a) else '[na]'


# ------------------------------------------------------------------ takes (contract 5) ------------------------------------------------------------------
def burst_counts(lot, lgood, lo, take_settle):
    """contract 5 / 6 / 7.3 fill burst at a window start lo (the same expression as chain_take.burst_counts): certain = own maker legs whose
    window lies inside [lo - 5 s, lo) (x_lo_hard >= lo - 5 s and hi < lo; clock-good legs only); possible = own maker legs whose window meets
    [lo - 5 s, lo) (bounded: hi >= lo - 5 s and x_lo_hard < lo; unbounded x_lo_hard: hi >= lo - 5 s and settled at or before the take)."""
    g = lgood[lgood.role == 'maker']
    cert = int(((g.xlo >= lo - BURST_MS) & (g.whi < lo)).sum())
    m = lot[lot.role == 'maker']
    bnd = m.xlo >= 0
    poss = int(((bnd & (m.whi >= lo - BURST_MS) & (m.xlo < lo)) | (~bnd & (m.whi >= lo - BURST_MS) & (m.settle_ms <= take_settle))).sum())
    return cert, poss


def take_core(sid, L):
    """one take record of the address (sid) against its market legs L (legs_frame) -> identity, window, inventory variants, classes."""
    me = L[(L.sid == sid) & (L.role == 'taker')]
    need(len(me) >= 1, f'take {sid}: no taker leg of the address')
    fl = set()
    if (me.status != C.STATUS_OK).any():
        fl.add('excluded_row')
    if me.self_trade.any():
        fl.add('self_trade')
    st_ms = int(me.settle_ms.iloc[0]); xlo, elo, whi, wst = int(me.xlo.iloc[0]), int(me.elo.iloc[0]), int(me.whi.iloc[0]), me.wst.iloc[0]
    if wst == 'no_window':
        fl.add('no_window')
    if st_ms > 0 and whi >= 0 and whi < st_ms:
        fl.add('receipt_before_uuid')
    if st_ms <= 0 or xlo < 0:
        fl.add('hi_from_receipt')
    s = int(np.sign(me.s.to_numpy(float).sum())) if len(set(me.s)) == 1 else 0
    if s == 0:
        fl.add('mixed_take_side')
        s = int(me.s.iloc[0])
    dq = float((me.s.to_numpy(float) * me.q.to_numpy(float))[(me.status == C.STATUS_OK).to_numpy(bool) & ~me.self_trade.to_numpy(bool)].sum())
    lot = L[L.lot & (L.sid != sid)]
    before = float(lot.sq[lot.settle_ms < st_ms].sum())
    simul = lot[lot.settle_ms == st_ms]
    if len(simul):
        fl.add('simultaneous_action')
    b_incl = before + float(simul.sq.sum())
    cert_ok = xlo >= 0 and not (fl & CLOCK_FLAGS)
    lgood = lot[~lot.clock_bad]
    if len(lgood) < len(lot):
        fl.add('legs_clock_bad_in_market')
    b_cert = float(lgood.sq[lgood.whi < xlo].sum()) if cert_ok else NAN
    pend = lot[(lot.whi > xlo) & (lot.xlo < whi) | lot.clock_bad] if cert_ok else lot.iloc[0:0]
    var = dict(recorded=(before, before + dq), simultaneous=(b_incl, b_incl + dq))
    if cert_ok:
        var['certain'] = (b_cert, b_cert + dq)
    cls = {k: inv_class(b, a) for k, (b, a) in var.items()}
    prev = lot[(lot.role == 'maker') & (lot.settle_ms < st_ms)]
    if len(prev):
        p = prev.iloc[-1]
        d_lo = max(0, xlo - int(p.whi)) if (xlo >= 0 and p.whi >= 0) else NAN
        d_hi = (whi - int(p.xlo)) if (p.xlo >= 0 and whi >= 0) else NAN
    else:
        d_lo = d_hi = NAN
    burst_cert, burst_pos = burst_counts(lot, lgood, xlo, st_ms) if cert_ok else (-1, burst_counts(lot, lgood, xlo, st_ms)[1] if xlo >= 0 else -1)
    tk_hash = str(me.hash.iloc[0])
    grp = L[(L.role == 'taker') & (L.hash == tk_hash)]
    return dict(sid=sid, tk_hash=tk_hash, n_records_taker_order=int(grp.sid.nunique()), taker_order_q=float(grp.q.sum()), s=s, q=float(me.q.sum()),
                dq=dq, p_avg=float((me.p * me.q).sum() / me.q.sum()), p_min=float(me.p.min()), p_max=float(me.p.max()), settle_ms=st_ms, xlo=xlo,
                elo=elo, hi=whi, flags=fl, variants=var, classes=cls, cls=cls['recorded'],
                timing_sensitive=len(set(cls.values())) > 1, pend_n=int(len(pend)), pend_q=float(pend.sq.sum()),
                prev_fill_lo_ms=d_lo, prev_fill_hi_ms=d_hi, burst_cert=burst_cert, burst_pos=burst_pos,
                fee=float(me.fee_fill.sum()), v_end=float(me.v_end.sum()))


# ------------------------------------------------------------------ the logic table (contract 6) ------------------------------------------------------------------
def logic_row(ep, ck, link, mech, lo, hi, clock, state, action, result, evidence, ident, cond, grade, tsens, status):
    alt, tr, comp, fals = TEXT[(link, mech)]
    need(ident in STATUSES and (cond in STATUSES or cond == ''), f'bad status {ident!r} / {cond!r}')
    return dict(episode_id=ep, case_key=ck, link=link, link_seq=0, next_link='', mechanism=mech, decision_window_lo_ms=int(lo),
                decision_window_hi_ms=int(hi), decision_clock=clock, state_change=state, alternatives=alt, proposed_tradeoff=tr,
                observed_action=action, resulting_state=result, supporting_evidence=evidence, competing_explanation=comp,
                what_would_falsify=fals, identification_status=ident, conditional_status=cond, evidence_grade=grade,
                timing_sensitive=bool(tsens), status=status)


def order_logic(ep, ck, core, bk, Mk, ctx, addr, links_out):
    """the logic rows of a maker-order case: POST (4 mechanisms), KEEP per kept fill, END, LINK per child."""
    rows = []
    d, ws, we, s = Mk['day'], Mk['ws_ms'], Mk['we_ms'], core['s']
    st = _flags(core['flags'] | bk.get('book_flags', set()))
    lo = core['post_lo_u'] if core['post_lo_u'] >= 0 else max(ws - PRE_MS, int(Mk['B']['t'][0]) if Mk['B'] and Mk['B']['F'] else ws - PRE_MS)
    x = instants(lo, core['post_hi'] - 1)
    bw = btc_window(ctx, Mk, x, s)
    bkx = ctx.book_at(Mk, x, core['side_i'], core['tick'])
    iv = ctx.inventory(Mk, addr, x); pr, pc_ = s * iv['recorded'], s * iv['certain']
    tau = (we - x) / 1000.
    state = (f"btc_r5s{rng(bw['r_5s_raw'])}bps cat={cats(bw['r_5s_cat'])}; r30s cat={cats(bw['r_30s_cat'])}; touch={cats(bkx['cat'])} "
             f"dist{rng(bkx['dist'], 0)}; spread{rng(bkx['spread'], 0)}; depth_at_price{rng(bkx['depth'], 1)}; s*pos_rec{rng(pr, 1)} "
             f"s*pos_cert{rng(pc_, 1)}; tau{rng(tau, 1)}s; phase={'pre' if core['post_hi'] < ws else ('live' if lo >= ws else 'straddle')}")
    result = (f"ahead_at_acceptance[{_r(bk.get('ahead_lo'))},{_r(bk.get('ahead_hi'))}]sh front={bk.get('front_class')}; first fill "
              f"[{core['first_lo_hard']},{core['first_hi']}]")
    action = (f"posted {core['y_side']} {core['tick']}c size in {{{','.join(sorted(core['union']))}}} (cum {core['cum']:.2f}, ub {_r(core['ub_u'])}) "
              f"accepted in [{core['post_lo_u']},{core['post_hi']}] (estimated; submission unidentified)")
    grade = 'E_unresolved' if core['post_lo_u'] < 0 else 'B_constrained'      # the union window holds under every declared scenario
    # queue priority before demand
    if core['post_lo_u'] < 0:
        q_st = 'unidentified'
    elif core['rest_ub'] <= EARLY_MS:
        q_st = 'contradicted'
    else:
        q_st = 'timing_sensitive'
    ev = f"rest_ub={_r(core['rest_ub'], 0)}ms class={core['post_class']}; P_nomask={bk.get('nomask_status')} rest[{_r(bk.get('nomask_rest_lo'), 0)},{_r(bk.get('nomask_rest_hi'), 0)}] (prior)"
    rows.append(logic_row(ep, ck, 'POST', 'queue_priority_before_demand', lo, core['post_hi'], 'acceptance (venue)', state, action, result, ev,
                          q_st, '', grade, q_st == 'timing_sensitive', st))
    valid = np.isin(bkx['cat'], ['at', 'inside', 'behind'])
    t_st = st_all(np.isin(bkx['cat'], ['at', 'inside']), valid)
    rows.append(logic_row(ep, ck, 'POST', 'touch_join_or_improve', lo, core['post_hi'], 'acceptance (venue)', state, action, result,
                          f"touch cats over {int(valid.sum())}/{len(x)} instants: {cats(bkx['cat'])}", t_st, '', grade, t_st == 'timing_sensitive', st))
    r_st = st_all(np.abs(bw['r_5s_raw']) >= MOVE_D['r_5s'], bw['ok'] & np.isfinite(bw['r_5s_raw']))
    rows.append(logic_row(ep, ck, 'POST', 'reaction_to_price_change', lo, core['post_hi'], 'acceptance (venue); BTC E_ms', state, action, result,
                          f"|r_5s|>=1bp at {int(np.sum((np.abs(bw['r_5s_raw']) >= 1.) & bw['ok']))}/{int(bw['ok'].sum())} instants", r_st, '', grade,
                          r_st == 'timing_sensitive', st))
    cat_i = np.r_[np.where(pr < -POS_EPS, 'offset', np.where(pr > POS_EPS, 'add', 'flat')), np.where(pc_ < -POS_EPS, 'offset', np.where(pc_ > POS_EPS, 'add', 'flat'))]
    if set(cat_i) == {'flat'}:
        i_st = 'unidentified'
    elif set(cat_i) == {'offset'}:
        i_st = 'compatible'
    elif set(cat_i) == {'add'}:
        i_st = 'contradicted'
    else:
        i_st = 'timing_sensitive'
    rows.append(logic_row(ep, ck, 'POST', 'inventory_offset', lo, core['post_hi'], 'acceptance (venue); recorded + certain inventory', state, action,
                          result, f'inventory relation over instants x variants: {cats(cat_i)}', i_st, '', grade, i_st == 'timing_sensitive', st))
    # KEEP per kept fill
    hi_f, xl_f = core['fill_hi'], core['fill_lo_hard']
    for j in range(len(hi_f) - 1):
        gap_max = float(hi_f[j + 1] - xl_f[j]) if xl_f[j] >= 0 else NAN
        k_st = 'timing_sensitive' if (np.isfinite(gap_max) and gap_max <= KEEP_REACT_MS) else ('compatible' if np.isfinite(gap_max) else 'unidentified')
        xx = np.array([hi_f[j], hi_f[j + 1]], np.int64)
        bm = ctx.book_at(Mk, xx, core['side_i'], core['tick'])
        bt = btc_window(ctx, Mk, xx[1:], s)
        adv_mid = s * (bm['mid'][1] - bm['mid'][0])
        rows.append(logic_row(ep, ck, 'KEEP', 'retain_after_fill', hi_f[j], hi_f[j + 1], 'match (UUID upper bounds)',
                              f"fill {j + 1}/{len(hi_f)}; s*dmid between fills {_r(adv_mid, 1)}c; s*r_5s at next {_r(bt['r_5s'][0])}bps",
                              'kept the remainder (observed: the same hash filled again)',
                              f"remaining after fill {j + 1} in [{_r(core['rem_lo'][j])},{_r(core['rem_hi'][j])}] sh",
                              f"next fill of the same hash; longest feasible gap {_r(gap_max, 0)} ms", k_st, '', 'A_observed', k_st == 'timing_sensitive',
                              _flags(core['flags'])))
    # END
    U = core['union']
    if 'PL' in U:
        e_lo = core['removal_lo']
        e_hi = core['pl_hi'] if core['pl_hi'] >= 0 else (int(Mk['B']['t'][-1]) if Mk['B'] and Mk['B']['F'] else we)
        e_lo2 = e_lo if e_lo >= 0 else core['last_hi'] - C.MAX_LAG_MS
        xe = instants(e_lo2, max(e_hi, e_lo2))
        be = btc_window(ctx, Mk, xe, s)
        dpos = s * (ctx.inventory(Mk, addr, xe)['recorded'] - ctx.inventory(Mk, addr, xe - LB_MS)['recorded'])
        ident = 'unidentified' if ({'EX', 'EC'} & U) else 'compatible'
        c1 = st_all(be['r_5s'] <= -MOVE_D['r_5s'], be['ok'] & np.isfinite(be['r_5s']))
        c2 = st_all(dpos > POS_EPS, np.ones(len(xe), bool))
        pri = core['priors']
        res = (f"hypotheses {{{','.join(sorted(U))}}}; withdrawn r in (0.01,{_r(core['hist_ub'].get('PL', NAN) - core['cum'])}] sh; "
               f"P_cmd={pri['P_cmd'][0]} P_drop={pri.get('P_drop', ('na',))[0]} P_lot={pri['P_lot'][0]} P_clip={pri['P_clip'][0]}")
        stt = f"btc against(r_5s) {int(np.sum(be['r_5s'] <= -1.))}/{int(be['ok'].sum())}; s*dpos(5s){rng(dpos, 1)}; tau{rng((we - xe) / 1000., 1)}s"
        rows.append(logic_row(ep, ck, 'END', 'pull_after_adverse_move', e_lo2, e_hi, 'cancel effect (venue); D=0', stt,
                              'end of life: pull not established' if ident == 'unidentified' else 'pulled (exhaustion excluded)', res,
                              f"size status {core['size_status']} ({core['grade']}); removal methods {core['removal_method']['S_NOPRIO']}", ident, c1,
                              core['grade'], c1 == 'timing_sensitive', _flags(core['flags'])))
        rows.append(logic_row(ep, ck, 'END', 'inventory_driven_pull', e_lo2, e_hi, 'cancel effect (venue); D=0; recorded inventory', stt,
                              'end of life: pull not established' if ident == 'unidentified' else 'pulled (exhaustion excluded)', res,
                              f'inventory built on the order side over the 5 s before {int(np.sum(dpos > POS_EPS))}/{len(xe)} instants', ident, c2,
                              core['grade'], c2 == 'timing_sensitive', _flags(core['flags'])))
    else:
        rows.append(logic_row(ep, ck, 'END', 'exhaustion', core['last_hi'], core['last_hi'], 'match (UUID upper bound)',
                              f"size status {core['size_status']}", 'exhausted at its last fill', f"original size {core['cum']:.2f} sh",
                              f"{core['grade']}: {core['removal_method']}", 'compatible', '', core['grade'], False, _flags(core['flags'])))
    # LINK per child
    for lk in links_out:
        rs = lk['status']
        l_st = 'compatible' if rs == 'sequential_identified' else ('contradicted' if rs == 'overlap_identified' else 'timing_sensitive')
        rows.append(logic_row(ep, ck, 'LINK', 'replace_or_replenish', core['last_hi'], lk['child_first_hi'], 'acceptance (venue) of the child',
                              f"{lk['kind']} d_ticks={lk['d_ticks']} gap[{_r(lk['gap_lo_ms'], 0)},{_r(lk['gap_hi_ms'], 0)}]ms",
                              f"next order {lk['child'][:10]} posted (estimated)", f"relations {lk['relations']}", f"pair status {rs}",
                              'unidentified' if rs == 'clock_refused' else l_st, '',
                              'A_observed' if lk['overlap_certain'] else 'B_constrained', l_st == 'timing_sensitive', 'ok'))
    if core['clock_bad']:                                           # contract 4.8: no claim from a refused clock; the evaluation stays visible
        for r in rows:
            if r['link'] != 'KEEP':
                r['conditional_status'] = r['conditional_status'] or r['identification_status']
                r['identification_status'], r['evidence_grade'], r['timing_sensitive'] = 'unidentified', 'X_conflict', False
    return rows


def take_logic(ep, ck, tk, Mk, ctx):
    rows = []
    d, ws, we, s = Mk['day'], Mk['ws_ms'], Mk['we_ms'], tk['s']
    bad = bool(tk['flags'] & CLOCK_FLAGS) or tk['xlo'] < 0
    lo = tk['xlo'] if tk['xlo'] >= 0 else tk['hi'] - C.MAX_LAG_MS
    x = instants(lo, tk['hi'])
    bw = btc_window(ctx, Mk, x, s)
    side_taken = 1 if s > 0 else 0                                     # a YES buy lifts the ask
    bk = ctx.book_at(Mk, x[:1], side_taken, int(round(tk['p_max' if s > 0 else 'p_min'] * 100)))
    touch = bk['ask'][0] if s > 0 else bk['bid'][0]
    through = (tk['p_max'] * 100 - touch) if s > 0 else (touch - tk['p_min'] * 100)
    edge = s * (bw['pf'] - tk['p_avg']) * 100.
    st = _flags(tk['flags'])
    state = (f"pos rec {tk['variants']['recorded'][0]:+.2f} (cert {_r(tk['variants'].get('certain', (NAN,))[0])}); pend {tk['pend_n']} legs "
             f"{tk['pend_q']:+.2f}; btc r_5s{rng(bw['r_5s_raw'])} cat={cats(bw['r_5s_cat'])}; r_30s cat={cats(bw['r_30s_cat'])}; touch {touch} "
             f"ticks_through {_r(through, 0)}; edge_vs_pfair{rng(edge, 2)}c; tau{rng((we - x) / 1000., 1)}s; prev own fill "
             f"[{_r(tk['prev_fill_lo_ms'], 0)},{_r(tk['prev_fill_hi_ms'], 0)}]ms; burst cert {tk['burst_cert']} poss {tk['burst_pos']}")
    action = f"took {'YES buy' if s > 0 else 'YES sell'} {tk['q']:.2f} sh @avg {tk['p_avg']:.2f} (record {tk['sid'][:13]}, taker order {tk['n_records_taker_order']} records)"
    result = f"pos {tk['variants']['recorded'][0]:+.2f} -> {tk['variants']['recorded'][1]:+.2f} ({tk['cls']})"
    cl = set(tk['classes'].values())
    if cl == {'from_flat'}:
        u_st = 'unidentified'
    elif cl <= REDUCING:
        u_st = 'compatible'
    elif cl <= {'add', 'overshoot'}:
        u_st = 'contradicted'
    else:
        u_st = 'timing_sensitive'
    grade = 'X_conflict' if bad else 'A_observed'
    rows.append(logic_row(ep, ck, 'TAKE', 'inventory_unwind', lo, tk['hi'], 'match (hard window)', state, action, result,
                          f"classes {tk['classes']}", u_st if not bad else 'unidentified', '', grade, u_st == 'timing_sensitive', st))
    v = bw['ok'] & np.isfinite(bw['r_5s'])
    cs = set(bw['r_5s_cat'][v])
    if not v.any():
        d_st = 'unidentified'
    elif cs == {'with'}:
        d_st = 'compatible'
    elif cs == {'against'}:
        d_st = 'contradicted'
    elif cs == {'flat'}:
        d_st = 'insufficient_comparison'
    else:
        d_st = 'timing_sensitive'
    rows.append(logic_row(ep, ck, 'TAKE', 'directional', lo, tk['hi'], 'match (hard window); BTC E_ms', state, action, result,
                          f"r_5s cats {cats(bw['r_5s_cat'][v])} over {int(v.sum())} instants", d_st if not bad else 'unidentified', '', grade,
                          d_st == 'timing_sensitive', st))
    if tk['burst_cert'] < 0 or tk['burst_pos'] < 0:
        b_st = 'unidentified'
    elif tk['burst_cert'] >= BURST_N:
        b_st = 'compatible'
    elif tk['burst_pos'] == 0:
        b_st = 'contradicted'
    else:
        b_st = 'timing_sensitive'
    rows.append(logic_row(ep, ck, 'TAKE', 'after_fill_burst', lo, tk['hi'], 'match (hard window)', state, action, result,
                          f"own maker fills in the 5 s before: certain {tk['burst_cert']}, possible {tk['burst_pos']}", b_st, '', grade,
                          b_st == 'timing_sensitive', st))
    return rows


# ------------------------------------------------------------------ episodes (contract 6) ------------------------------------------------------------------
def episode_ids(cases):
    """cases: rows with day, mid, addr, t0, t1 -> connected-episode index per row (union-find over overlapping windows per day x market x
    address; ids ordered by window start)."""
    eid = np.full(len(cases), -1, np.int64)
    k = 0
    for _, g in cases.groupby(['day', 'mid', 'addr'], sort=True):
        g = g.sort_values(['t0', 't1'], kind='stable')
        cur_end = None
        for i, (a, b) in zip(g.index, zip(g.t0, g.t1)):
            if cur_end is None or a > cur_end:
                k += 1; cur_end = b
            else:
                cur_end = max(cur_end, b)
            eid[cases.index.get_loc(i)] = k
    return eid


def episode_econ(L, inv, t0, t1):
    m = L.lot & (L.settle_ms >= t0) & (L.settle_ms <= t1)
    E = L[m]
    ve = E.v_end.to_numpy(float)
    net_end = float(np.nansum(ve) - E.fee_fill.to_numpy(float).sum())
    return dict(ep_n_legs=int(len(E)), ep_n_maker=int((E.role == 'maker').sum()), ep_n_taker=int((E.role == 'taker').sum()),
                ep_sh=float(E.q.sum()), ep_v_end=float(np.nansum(ve)), ep_n_v_end_na=int(np.isnan(ve).sum()), ep_fee=float(E.fee_fill.sum()),
                ep_net_end=net_end, ep_net_res=float(np.nansum(E.net_res.to_numpy(float))), ep_mq=float(E.mq.sum()), ep_rq=float(E.rq.sum()),
                ep_pos_in=float(inv.recorded(np.array([t0]))[0]), ep_pos_out=float(inv.recorded(np.array([t1 + 1]))[0]),
                ep_outcome='profitable_continuation' if net_end > 0 else ('losing_continuation' if net_end < 0 else 'flat'),
                ep_n_excluded=int((~L.lot & (L.settle_ms >= t0) & (L.settle_ms <= t1) & (L.status != C.STATUS_OK)).sum()))


# ------------------------------------------------------------------ census over every top-3 lifecycle row (contract 8) ------------------------------------------------------------------
def census_market(d, mid, orders, fills, links, settle):
    """per top-3 order (no book): the estimator fields; per top-3 x_links pair: the link status. settle: {(sid, hash): settle_ms}."""
    sig = orders.mk_signer.str.lower()
    O = orders[sig.isin(LOW3)]
    R3 = rows3_of(O)
    fs = fills[fills.mk_hash.isin(list(R3))].sort_values(['mk_hash', 'hi_ms', 'x_rec_rank', 'sid'], kind='stable')
    fb = {h: g for h, g in fs.groupby('mk_hash', sort=False)}
    cc = clip_counts(orders)
    cores, rows = {}, []
    for h, r3 in R3.items():
        fl = fb.get(h, fills.iloc[0:0])
        g = str(r3['S_NOPRIO']['mk_signer']).lower()
        st = {sid: settle.get((sid, h)) for sid in fl.sid}
        core = order_core(r3, fl, settle=st, clip=clip_for(cc.get(g, Counter()), _fin(r3['S_FIFO']['size_exact'])), presorted=True)
        cores[h] = core
        rows.append(summary_row(d, mid, LOW3.index(g) + 1, core))
    sids = {h: set(g.sid) for h, g in fb.items()}
    lrows = []
    lk = links[links.mk_signer.str.lower().isin(LOW3)]
    for r in lk.itertuples(index=False):
        P, Q = cores.get(r.prev_hash), cores.get(r.next_hash)
        if P is None or Q is None:
            lrows.append(dict(day=d, mid=mid, rank=LOW3.index(str(r.mk_signer).lower()) + 1, kind=r.kind, status='core_missing', x_latest=bool(r.x_latest)))
            continue
        res = link_hypotheses(P, Q, shared_record=bool(sids.get(r.prev_hash, set()) & sids.get(r.next_hash, set())))
        lrows.append(dict(day=d, mid=mid, rank=LOW3.index(str(r.mk_signer).lower()) + 1, kind=r.kind, status=res['status'], x_latest=bool(r.x_latest),
                          replace_feasible=res['replace_feasible'], replenish_feasible=res['replenish_feasible'], overlap_feasible=res['overlap_feasible']))
    return rows, lrows, cores


def coverage_rows(census, lcensus, takes, sample, sample_takes):
    """coverage.csv: scope x rank x day x event x clock x category with orders / executions / shares (takes: records / shares)."""
    out = []

    def add(scope, df, event, clock, cat_col, w_col=None, unit='orders', refusable=False):
        if not len(df):
            return
        X = df.copy()
        cb_ = X['clock_bad'].to_numpy(bool) if 'clock_bad' in X.columns else np.zeros(len(X), bool)     # link pairs carry no clock column
        cf_ = X['conflict'].to_numpy(bool) if 'conflict' in X.columns else np.zeros(len(X), bool)
        if w_col is not None:
            X['cat'] = [('refused_clock' if cb else ('conflicting' if cf else width_bin(w))) for w, cb, cf in zip(X[w_col], cb_, cf_)]
        else:
            X['cat'] = np.where(cb_ & refusable, 'refused_clock', X[cat_col].astype(str).to_numpy(object))
        for (rk, d, c), g in X.groupby(['rank', 'day', 'cat'], sort=True):
            out.append(dict(scope=scope, rank=int(rk), day=d, event=event, clock=clock, category=c, n_units=int(len(g)),
                            n_exec=int(g['n_fills'].sum()) if 'n_fills' in g else int(len(g)),
                            shares=round(float(g['cum'].sum() if 'cum' in g else g['q'].sum()), 4), unit=unit))
    for scope, df in (('census', census), ('sample', sample)):
        if not len(df):
            continue
        add(scope, df, 'posting', 'acceptance_union', None, 'post_w')
        add(scope, df, 'posting', 'acceptance_S_FIFO', None, 'post_w_fifo')
        for D in D_PROFILES[1:]:
            X = df.copy(); X['post_wD'] = np.where(X['post_w'] >= 0, X['post_w'] + D, -1)
            add(scope, X, 'posting', f'submission_D{D}_declared', None, 'post_wD')
        add(scope, df, 'posting_class', 'acceptance_union', 'post_class', refusable=True)
        X = df[df['pl'].to_numpy(bool)]
        add(scope, X, 'removal_PL', 'cancel_effect', None, 'pl_w')
        for D in D_PROFILES[1:]:
            Y = X.copy(); Y['pl_wD'] = np.where(Y['pl_w'] >= 0, Y['pl_w'] + D, -1)
            add(scope, Y, 'removal_PL', f'cancel_submission_D{D}_declared', None, 'pl_wD')
        add(scope, df, 'size_end', 'evidence', 'size_status')
        add(scope, df, 'hypotheses', 'evidence', 'union')
        add(scope, df, 'prior_P_cmd', 'prior', 'P_cmd')
        add(scope, df, 'prior_sensitive', 'prior', 'prior_sensitive3')
        add(scope, df, 'queue', 'S_FIFO', 'queue_fifo', refusable=True)
        if 'front_class' in df:
            add(scope, df, 'front_class', 'S_FIFO + book', 'front_class', refusable=True)
            add(scope, df, 'posting_prior', 'P_nomask (prior)', 'nomask_status', refusable=True)
            add(scope, df, 'prior_sensitive4', 'prior (P_cmd, P_drop, P_lot, P_clip)', 'prior_sensitive')
        F = df[['rank', 'day', 'clock_bad', 'conflict', 'fill_w_hard', 'fill_w_ev']].explode(['fill_w_hard', 'fill_w_ev'])
        F = F[F.fill_w_hard.notna()]
        if len(F):
            F = F.assign(n_fills=1, cum=0.)
            add(scope, F.assign(fwh=F.fill_w_hard.astype(float)), 'fill_match', 'hard [hi-2000, hi]', None, 'fwh', unit='fills')
            add(scope, F.assign(fwe=F.fill_w_ev.astype(float)), 'fill_match', 'evidence [lo, hi] (scenario refinement)', None, 'fwe', unit='fills')
    for scope, df in (('census', lcensus),):
        if len(df):
            add(scope, df.assign(n_fills=1, cum=0.), 'link', 'link_hypotheses', 'status', unit='pairs')
    for scope, df in (('census', takes), ('sample', sample_takes)):
        if len(df):
            T = df.assign(n_fills=1, cum=df['q'])
            add(scope, T, 'take_match', 'hard [hi-2000, hi]', None, 'w_hard', unit='records')
            add(scope, T, 'take_match', 'evidence [lo, hi] (scenario refinement)', None, 'w_ev', unit='records')
            if 'cls' in T:
                add(scope, T, 'take_class', 'recorded inventory', 'cls', unit='records')
                add(scope, T, 'take_timing_sensitive', 'inventory variants', 'timing_sensitive', unit='records')
    return pd.DataFrame(out)


# ------------------------------------------------------------------ own-order instrument check (contract 9) ------------------------------------------------------------------
GT_COLS = ('run', 'mid', 'mk_hash', 'link_status', 'size_sh', 'filled_sh', 'final_state', 'act_lo_venue_ms', 'act_hi_venue_ms', 'status')


def gt_rows(own, life_orders_by_mid):
    """own: truth/v2/own_orders rows (linked); life_orders_by_mid: {mid: lifecycle orders (all scenarios)} -> per own order the hypotheses vs
    the truth (no depth: P_drop / P_nomask not evaluated)."""
    out = []
    for mid, g in own.groupby('mid', sort=True):
        O = life_orders_by_mid.get(int(mid))
        if O is None:
            for r in g.itertuples(index=False):
                out.append(dict(mid=int(mid), hash=r.mk_hash, status='no_lifecycle')); continue
            continue
        R3 = rows3_of(O[O.mk_hash.isin(set(g.mk_hash))])
        for r in g.itertuples(index=False):
            r3 = R3.get(r.mk_hash)
            if r3 is None:
                out.append(dict(mid=int(mid), hash=r.mk_hash, status='hash_not_in_lifecycle')); continue
            core = order_core(r3, None, settle=None)
            truth = {'filled': {'EX'}, 'cancelled': {'PL'}, 'expired': {'PL', 'EC'}}.get(str(r.final_state), set())
            in_u = bool(truth & core['union']) if truth else None
            in_fifo = bool(truth & core['feas']['S_FIFO']) if truth else None
            if not truth:
                verdict = 'truth_unknown'
            elif not in_u:
                verdict = 'miss_conflict' if core['conflict'] else 'defect'     # every declared scenario excluded the true history
            elif not in_fifo:
                verdict = 'priority_miss'
            else:
                verdict = 'covered'
            size = _fin(r.size_sh)
            if truth == {'EX'}:
                s_ok = abs(size - core['cum']) <= max(TOL, 1e-6 * size)
            else:
                us = [core['hist_ub'].get(h, NAN) for h in truth if h in core['union']]
                u = NAN if (not us or any(np.isnan(x) for x in us)) else max(us)
                s_ok = (size > core['cum'] + TOL - 1e-9) and (np.isnan(u) or size <= u + TOL)
            a_lo, a_hi = _fin(r.act_lo_venue_ms), _fin(r.act_hi_venue_ms)
            p_lo = core['post_lo_u'] if core['post_lo_u'] >= 0 else -INF
            post_ok = None if (np.isnan(a_lo) or np.isnan(a_hi)) else bool(a_lo <= core['post_hi'] and a_hi >= p_lo)
            pr = core['priors']
            out.append(dict(mid=int(mid), hash=r.mk_hash, run=r.run, final_state=r.final_state, size_sh=size, filled_sh=_fin(r.filled_sh), cum=core['cum'],
                            union='|'.join(sorted(core['union'])), size_status=core['size_status'], truth='|'.join(sorted(truth)), verdict=verdict,
                            size_in_range=bool(s_ok) if truth else None, posting_overlaps_truth=post_ok,
                            P_cmd_ok=(pr['P_cmd'][0] in truth) if truth else None, P_lot_ok=(pr['P_lot'][0] in truth) if truth else None,
                            P_clip_ok=(pr['P_clip'][0] in truth) if truth else None, status='ok', flags=_flags(core['flags'])))
    return pd.DataFrame(out)


# ------------------------------------------------------------------ I/O ------------------------------------------------------------------
def load_population(R):
    ids = json.loads((Path(R) / 'econ/identities.json').read_text())
    top = [x['addr'] for x in ids['ex_ante']['top10_shares'][:3]]
    refuse([x['rank'] for x in ids['ex_ante']['top10_shares'][:3]] == [1, 2, 3], 'identities.json: ranks 1-3 not first')
    refuse(tuple(top) == TOP3, f'identities.json ranks 1-3 {top} differ from the decision contract TOP3 (refused)')
    man = json.loads(BOOK_MANIFEST.read_text())
    book = {d: [int(m) for m in v] for d, v in man['complete'].items()}
    refuse(set(book) <= set(C.MEASURE_DAYS), f'BOOK manifest days {sorted(book)} outside the measurement days (refused)')
    refuse(sum(len(v) for v in book.values()) == N_BOOK, f'BOOK manifest has {sum(len(v) for v in book.values())} markets, not {N_BOOK}')
    return book


def read_fv(path, book, cnt):
    """the top-3 rows of the BOOK markets, through the row-group mid statistics; every row's day must be its market's manifest day."""
    mids = np.array(sorted(m for v in book.values() for m in v), np.int64)
    day_of = {m: d for d, v in book.items() for m in v}
    pf = pq.ParquetFile(path); md = pf.metadata
    names = [md.row_group(0).column(k).path_in_schema for k in range(md.num_columns)] if md.num_row_groups else []
    refuse(all(c in names for c in FV_COLS), f'fills_valued lacks {[c for c in FV_COLS if c not in names]}')
    jm = names.index('mid')
    vm, va = pa.array(mids), pa.array(list(LOW3))
    parts = []
    for i in range(md.num_row_groups):
        st = md.row_group(i).column(jm).statistics
        if st is not None and st.has_min_max:
            k = int(np.searchsorted(mids, int(st.min), 'left'))
            if k >= len(mids) or mids[k] > int(st.max):
                cnt['fv_groups_pruned'] += 1
                continue
        t = pf.read_row_group(i, columns=list(FV_COLS))
        cnt['fv_groups_read'] += 1; cnt['fv_rows_scanned'] += t.num_rows
        t = pa.table({c: (t.column(c).cast(t.column(c).type.value_type) if pa.types.is_dictionary(t.column(c).type) else t.column(c)) for c in FV_COLS})
        m = pc.fill_null(pc.and_(pc.is_in(t.column('mid'), value_set=vm), pc.is_in(pc.utf8_lower(t.column('addr')), value_set=va)), False)
        if pc.any(m).as_py():
            parts.append(t.filter(m))
        del t
    refuse(parts, 'fills_valued: no row of the top 3 in the BOOK markets')
    df = pa.concat_tables(parts).to_pandas()
    df['addr_l'] = df.addr.str.lower()
    sp = df.groupby('addr_l').addr.nunique()
    refuse((sp <= 1).all(), f'fills_valued: an address under two spellings {sp[sp > 1].to_dict()} (refused)')
    bad = df.day.to_numpy(object) != np.array([day_of[int(m)] for m in df.mid], object)
    refuse(not bad.any(), f'fills_valued: {int(bad.sum())} rows of a BOOK market under another day (refused)')
    cnt['fv_rows_top3_book'] = int(len(df))
    return df


def check_fv(p, want=FV_SHA256, want_bytes=FV_BYTES):
    """fills_valued must be the file the accepted matched_gap_r1 receipt hashed (bytes and sha256), else refuse."""
    p = Path(p)
    refuse(p.stat().st_size == want_bytes, f'fills_valued is {p.stat().st_size} bytes, not the accepted {want_bytes} (refused)')
    h = _sha(p)
    refuse(h == want, f'fills_valued sha256 {h} differs from the accepted receipt {want} (refused)')
    return h


def life_paths(R, d, mid):
    need(d in C.ALLOWED_DAYS or d == C.NOV8, f'day {d} is not a consumed day (never opened)')
    return {k: Path(R) / 'life' / f'{d}_{int(mid)}_{k}.parquet' for k in ('orders', 'fills', 'x_links')}


def depth_paths(R, d, mid):
    need(d in C.ALLOWED_DAYS or d == C.NOV8, f'day {d} is not a consumed day (never opened)')
    return {k: Path(R) / 'depth' / f'{d}_{int(mid)}_{k}.parquet' for k in ('frames', 'levels')}


def market_times(R, d, mids):
    need(d in C.MEASURE_DAYS, f'day {d} is not a measurement day')
    T = pd.read_parquet(Path(R) / f'markets_{d}.parquet', columns=['mid', 'ws', 'we', 'status'])
    T = T[T.mid.isin(mids)]
    refuse(len(T) == len(set(mids)) and T.mid.is_unique, f'markets_{d}: BOOK markets missing or doubled')
    return {int(r.mid): (int(round(float(r.ws) * 1000)), int(round(float(r.we) * 1000)), str(r.status)) for r in T.itertuples(index=False)}


def windows_of(fills):
    """sid -> (x_lo_hard, lo, hi, t_method) from the life fills (every leg of a record shares one window)."""
    g = fills.drop_duplicates('sid')
    return {s: (int(a), int(b), int(c), str(m)) for s, a, b, c, m in zip(g.sid, g.x_lo_hard_ms, g.lo_ms, g.hi_ms, g.t_method)}


def freeze_state(paths):
    st = {}
    for p in paths:
        rel = str(Path(p).relative_to(C.REPO))
        tr = subprocess.run(['git', '-C', str(C.REPO), 'ls-files', '--error-unmatch', rel], capture_output=True).returncode == 0
        clean = tr and subprocess.run(['git', '-C', str(C.REPO), 'diff', '--quiet', 'HEAD', '--', rel]).returncode == 0
        st[rel] = dict(tracked=tr, clean=bool(clean), sha256=_sha(p) if Path(p).exists() else None)
    head = subprocess.run(['git', '-C', str(C.REPO), 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip()
    return head, st


def _write_csv(df, p):
    df.to_csv(p, index=False)


def _jsonable(x):
    if isinstance(x, (set, frozenset)):
        return sorted(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return None if not np.isfinite(x) else float(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, Path):
        return str(x)
    return str(x)


# ------------------------------------------------------------------ the market pass (cases) ------------------------------------------------------------------
def market_cases(Mk, cases_m, ctx, cnt):
    """the detailed cases of one market (Mk: day, mid, ws_ms, we_ms, orders, fills, links, B, touch, FF, L (address legs by addr), inv,
    settle) -> (case rows, ledger rows, logic rows, timeline events, sample census rows, sample take rows)."""
    R3 = rows3_of(Mk['orders'])
    fb = {h: g for h, g in Mk['fills'].groupby('mk_hash', sort=False)}
    cc = clip_counts(Mk['orders'])
    cache = {}

    def core_of(h):
        if h not in cache:
            r3 = R3[h]; g = str(r3['S_NOPRIO']['mk_signer']).lower()
            fl = fb.get(h, Mk['fills'].iloc[0:0])
            st = {sid: Mk['settle'].get((sid, h)) for sid in fl.sid} if g in LOW3 else None
            c = order_core(r3, fl, settle=st, clip=clip_for(cc.get(g, Counter()), _fin(r3['S_FIFO']['size_exact'])))
            c['clip'] = clip_for(cc.get(g, Counter()), _fin(r3['S_FIFO']['size_exact']))
            cache[h] = c
        return cache[h]
    sids = {h: set(g.sid) for h, g in fb.items()}
    cells = {}

    def cell_of(core):
        k = (core['side_i'], core['tick'])
        if k not in cells:
            if core['side_i'] < 0 or Mk['B'] is None or Mk['B']['F'] == 0:
                cells[k] = (None, None)
            else:
                cc_ = [core_of(h) for h, r3 in R3.items() if r3['S_NOPRIO']['y_side'] == core['y_side'] and int(r3['S_NOPRIO']['y_ptick']) == core['tick']]
                CA = cell_arrays(Mk['B'], Mk['fills'], Mk['FF'], core['side_i'], core['tick'])
                cells[k] = (CA, cell_nomask(cc_, Mk['B'], CA, Mk['ws_ms']))
                cnt['cells_evaluated'] += 1
        return cells[k]
    crow, ledger, logic, tl, srow, strow = [], [], [], [], [], []
    for c in cases_m.itertuples(index=False):
        ck = f'{c.kind}|{c.day}|{c.mid}|{c.addr}|{c.key}'
        L, inv = Mk['L'][c.addr], Mk['inv'][c.addr]
        ep = c.episode_id
        if c.kind == 'order':
            core = core_of(c.key)
            CA, NM = cell_of(core)
            bk = order_book(core, Mk['B'], Mk['touch'], Mk['fills'], Mk['FF'], CA, NM, Mk['ws_ms'])
            pr = core['priors']
            kids = Mk['links'][(Mk['links'].prev_hash == c.key)]
            par = Mk['links'][(Mk['links'].next_hash == c.key)]
            links_out = []
            for role, rows_ in (('child', kids), ('parent', par)):
                for r in rows_.itertuples(index=False):
                    other = r.next_hash if role == 'child' else r.prev_hash
                    if other not in R3:
                        continue
                    P, Q = (core, core_of(other)) if role == 'child' else (core_of(other), core)
                    res = link_hypotheses(P, Q, shared_record=bool(sids.get(P['hash'], set()) & sids.get(Q['hash'], set())))
                    res.update(role=role, other=other, child=Q['hash'], child_first_hi=Q['first_hi'], x_latest=bool(r.x_latest))
                    ledger.append(dict(case_key=ck, row_kind='link', hypothesis=res['relations'], role=role, other_hash=other, kind=res['kind'],
                                       d_ticks=res['d_ticks'], status=res['status'], overlap_certain=res['overlap_certain'], gap_lo_ms=res['gap_lo_ms'],
                                       gap_hi_ms=res['gap_hi_ms'], other_first_hi_ms=core_of(other)['first_hi'], other_cum=core_of(other)['cum'],
                                       other_union='|'.join(sorted(core_of(other)['union'])), x_latest=bool(r.x_latest)))
                    if role == 'child':
                        links_out.append(res)
            for hname in ('EX', 'PL', 'EC'):
                fe = {s: hname in core['feas'][s] for s in SCEN}
                if hname == 'EX':
                    s_lo, s_hi, w_lo, w_hi = core['cum'], core['cum'], core['last_hi'], core['last_hi']
                else:
                    s_lo, s_hi = core['cum'] + TOL, core['hist_ub'].get(hname, NAN)
                    w_lo, w_hi = core['removal_lo'], (core['pl_hi'] if hname == 'PL' else -1)
                ledger.append(dict(case_key=ck, row_kind='size_end', hypothesis=hname, feasible_union=hname in core['union'],
                                   **{f'feasible_{s}': fe[s] for s in SCEN}, size_lo=s_lo, size_hi=s_hi, window_lo_ms=w_lo, window_hi_ms=w_hi,
                                   remaining_lo=(0. if hname == 'EX' else TOL), remaining_hi=(0. if hname == 'EX' else s_hi - core['cum']),
                                   P_cmd=pr['P_cmd'][0] == hname, P_drop=pr.get('P_drop', ('',))[0] == hname, P_lot=pr['P_lot'][0] == hname,
                                   P_clip=pr['P_clip'][0] == hname, status=core['size_status'], grade=core['grade']))
            row = dict(case_key=ck, kind='order', day=c.day, mid=c.mid, addr=c.addr, rank=c.rank, key=c.key, pick=c.pick, round=c.round,
                       episode_id=ep, y_side=core['y_side'], tick=core['tick'], n_fills=core['n'], cum=core['cum'], size_status=core['size_status'],
                       grade=core['grade'], union='|'.join(sorted(core['union'])), ub=core['ub_u'], post_lo_union=core['post_lo_u'],
                       post_lo_fifo=core['post_lo_fifo'], post_hi=core['post_hi'], rest_ub_ms=core['rest_ub'], post_class=core['post_class'],
                       removal_lo=core['removal_lo'], pl_hi=core['pl_hi'], end_hi=core['end_hi'], first_lo_hard=core['first_lo_hard'],
                       last_hi=core['last_hi'], ahead_lo_fifo=core['ahead_lo_fifo'], ahead_hi_fifo=core['ahead_hi_fifo'],
                       ahead_at_acceptance_lo=bk['ahead_lo'], ahead_at_acceptance_hi=bk['ahead_hi'], front_class=bk['front_class'],
                       n_front_after_partial=core['n_front_after_partial'], touch_first_ms=bk['touch_first_ms'], touch_at_start=bk['touch_at_start'],
                       nomask_status=bk['nomask_status'], nomask_n=bk['nomask_n'], nomask_rep_lo=bk['nomask_rep_lo'], nomask_rep_hi=bk['nomask_rep_hi'],
                       nomask_rest_lo=bk['nomask_rest_lo'], nomask_rest_hi=bk['nomask_rest_hi'], nomask_class=bk['nomask_class'],
                       drop_r=bk['drop_r'], drop_frame_ms=bk['drop_frame_ms'], P_cmd=pr['P_cmd'], P_drop=pr.get('P_drop'), P_lot=pr['P_lot'],
                       P_clip=pr['P_clip'], prior_sensitive=pr['prior_sensitive'], clip=core.get('clip', NAN), n_links=len(links_out),
                       status=_flags(core['flags'] | bk['book_flags']))
            crow.append(row)
            srow.append(dict(summary_row(c.day, c.mid, c.rank, core), front_class=bk['front_class'], nomask_status=bk['nomask_status'],
                             prior_sensitive=pr['prior_sensitive']))
            logic += order_logic(ep, ck, core, bk, Mk, ctx, c.addr, links_out)
            for j, (h_, x_, q_, sid_) in enumerate(zip(core['fill_hi'], core['fill_lo_hard'], core['fill_q'], core['fill_sid'])):
                tl.append(dict(episode_id=ep, t_ms=int(h_), event='case_fill', case_key=ck, detail=f'{core["y_side"]} {core["tick"]} q {q_:.2f} lo_hard {x_} sid {sid_[:13]}'))
            tl.append(dict(episode_id=ep, t_ms=int(core['post_lo_u']), event='case_post_window_lo', case_key=ck, detail=f'union; fifo {core["post_lo_fifo"]}'))
            tl.append(dict(episode_id=ep, t_ms=int(core['post_hi']), event='case_post_window_hi', case_key=ck, detail='first fill hi'))
            if 'PL' in core['union']:
                tl.append(dict(episode_id=ep, t_ms=int(core['removal_lo']), event='case_PL_window_lo', case_key=ck, detail=f'withdrawn r <= {_r(core["hist_ub"].get("PL", NAN) - core["cum"])}'))
                tl.append(dict(episode_id=ep, t_ms=int(core['pl_hi']), event='case_PL_window_hi', case_key=ck, detail=core['removal_method']['S_NOPRIO']))
        else:
            tk = take_core(c.key, L)
            row = dict(case_key=ck, kind='take', day=c.day, mid=c.mid, addr=c.addr, rank=c.rank, key=c.key, pick=c.pick, round=c.round, episode_id=ep,
                       tk_hash=tk['tk_hash'], n_records_taker_order=tk['n_records_taker_order'], s=tk['s'], q=tk['q'], p_avg=tk['p_avg'],
                       settle_ms=tk['settle_ms'], x_lo_hard=tk['xlo'], lo=tk['elo'], hi=tk['hi'], pos_before=tk['variants']['recorded'][0],
                       pos_after=tk['variants']['recorded'][1], cls=tk['cls'], cls_simultaneous=tk['classes']['simultaneous'],
                       cls_certain=tk['classes'].get('certain', 'unavailable'), timing_sensitive=tk['timing_sensitive'], pend_n=tk['pend_n'],
                       pend_q=tk['pend_q'], prev_fill_lo_ms=tk['prev_fill_lo_ms'], prev_fill_hi_ms=tk['prev_fill_hi_ms'], burst_cert=tk['burst_cert'],
                       burst_pos=tk['burst_pos'], fee=tk['fee'], v_end=tk['v_end'], status=_flags(tk['flags']))
            crow.append(row)
            cb = bool(tk['flags'] & CLOCK_FLAGS)
            strow.append(dict(rank=c.rank, day=c.day, q=tk['q'], clock_bad=cb, conflict=False, w_hard=(tk['hi'] - tk['xlo']) if tk['xlo'] >= 0 else -1,
                              w_ev=(tk['hi'] - tk['elo']) if tk['elo'] >= 0 else -1, cls=tk['cls'], timing_sensitive=tk['timing_sensitive']))
            logic += take_logic(ep, ck, tk, Mk, ctx)
            tl.append(dict(episode_id=ep, t_ms=int(tk['hi']), event='case_take', case_key=ck, detail=f"{tk['cls']} {tk['dq']:+.2f} sh @ {tk['p_avg']:.2f}"))
    return crow, ledger, logic, tl, srow, strow


# ------------------------------------------------------------------ run ------------------------------------------------------------------
def run(pilot=None, R=None, out_root=None, log=print):  # noqa: C901
    t_all = time.process_time()
    stage_cpu = {}
    t_last = [time.process_time()]

    def stage(name):
        now = time.process_time()
        stage_cpu[name] = round(stage_cpu.get(name, 0.) + now - t_last[0], 3)
        t_last[0] = now
        refuse(now - t_all <= CPU_CAP_S, f'CPU self-guard: {now - t_all:.1f} s > {CPU_CAP_S} s at {name}')
        LC.guard(name)
    refuse(C.RUN == 'v2', f'MT_RUN must be v2 (is {C.RUN!r})')
    R = Path(R or C.OUT)
    out_root = Path(out_root or C.RECEIPT)
    if pilot:
        pd_, prank = pilot
        refuse(pd_ in C.MEASURE_DAYS and prank in (1, 2, 3), f'--pilot needs a measurement day and rank 1-3, not {pilot}')
        name = f'order_estimator_pilot_{pd_}_r{prank}'
    else:
        name = 'order_estimator'
    out = out_root / name
    refuse(not out.exists(), f'write-once: {out} exists (refusing before any input is read)')
    srcs = [Path(__file__).resolve(), SELFTEST, CONTRACT] + ([CHAIN_CONTEXT] if CHAIN_CONTEXT.exists() else [])
    head, frz = freeze_state(srcs)
    frozen = all(v['tracked'] and v['clean'] for v in frz.values())
    refuse(pilot or frozen, f'the full run requires the freeze (tracked, unmodified): {frz}')
    cnt = Counter()
    book = load_population(R)
    days = sorted(book)
    ctx, prov = make_context(R, require_chain=not pilot)
    stage('setup')
    # ---- selection (key columns only) ----
    okeys = []
    for d in days:
        for m in book[d]:
            o = pd.read_parquet(life_paths(R, d, m)['orders'], columns=['scenario', 'mk_hash', 'mk_signer'])
            o = o[o.scenario == 'S_NOPRIO']
            sg = o.mk_signer.str.lower()
            sp = o[sg.isin(LOW3)].groupby(sg[sg.isin(LOW3)]).mk_signer.nunique()
            refuse((sp <= 1).all(), f'life orders {d}_{m}: an address under two spellings (refused)')
            okeys += [('order', d, int(m), a, h) for a, h in zip(sg[sg.isin(LOW3)], o.mk_hash[sg.isin(LOW3)])]
    fv = read_fv(R / 'econ/fills_valued.parquet', book, cnt)
    tk = fv[fv.role == 'taker'][['day', 'mid', 'addr_l', 'sid']].drop_duplicates()
    keys = pd.DataFrame(okeys + [('take', d, int(m), a, s) for d, m, a, s in tk.itertuples(index=False)], columns=['kind', 'day', 'mid', 'addr', 'key'])
    sel, man = select_cases(keys, days)
    man_csv = sel.to_csv(index=False)
    man_sha = hashlib.sha256(man_csv.encode()).hexdigest()
    stage('select')
    # ---- census (every top-3 lifecycle row; no book) ----
    mk_rows = fv.role == 'maker'
    settle = dict(zip(zip(fv.sid[mk_rows], fv.hash[mk_rows]), fv.settle_ms[mk_rows].astype(int)))
    tag = fv[fv.role == 'taker'].groupby(['mid', 'addr_l', 'sid'], sort=True).agg(q=('q', 'sum'), settle_ms=('settle_ms', 'first')).reset_index()
    cen, lcen, tcen, life = [], [], [], {}
    for d in days:
        for m in book[d]:
            p = life_paths(R, d, m)
            O, Fl, Lk = (pd.read_parquet(p[k]) for k in ('orders', 'fills', 'x_links'))
            life[(d, m)] = (O, Fl, Lk)
            r1, r2, _ = census_market(d, m, O, Fl, Lk, settle)
            cen += r1; lcen += r2
            W = windows_of(Fl)
            for r_ in tag[tag.mid == m].itertuples(index=False):
                w = W.get(r_.sid, (-1, -1, -1, 'no_window'))
                stv = int(r_.settle_ms)
                cb = w[3] == 'no_window' or w[0] < 0 or stv <= 0 or (w[2] < stv)
                tcen.append(dict(rank=LOW3.index(r_.addr_l) + 1, day=d, mid=m, sid=r_.sid, q=float(r_.q), clock_bad=cb, conflict=False,
                                 w_hard=(w[2] - w[0]) if w[0] >= 0 else -1, w_ev=(w[2] - w[1]) if w[1] >= 0 else -1))
    census, lcensus, tcensus = pd.DataFrame(cen), pd.DataFrame(lcen), pd.DataFrame(tcen)
    stage('census')
    # ---- the detailed cases ----
    cs = sel if not pilot else sel[(sel.day == pilot[0]) & (sel['rank'] == pilot[1])]
    cs = cs.reset_index(drop=True)
    win_all = {}
    t0s, t1s = [], []
    mt = {d: market_times(R, d, book[d]) for d in days}
    for c in cs.itertuples(index=False):
        O, Fl, Lk = life[(c.day, c.mid)]
        ws, we, _ = mt[c.day][c.mid]
        if c.kind == 'order':
            oo = O[O.mk_hash == c.key]
            plo = int(oo.post_lo_ms.min())
            ends = [int(oo.last_hi_ms.iloc[0])] + [int(x) for x in oo.removal_hi_ms if int(x) >= 0]
            t0s.append(max(plo if plo >= 0 else ws - PRE_MS, ws - PRE_MS)); t1s.append(min(max(ends) + EP_PAD_MS, we))
        else:
            W = win_all.setdefault((c.day, c.mid), windows_of(Fl))
            w = W.get(c.key, (-1, -1, -1, 'no_window'))
            lo = w[0] if w[0] >= 0 else w[2] - C.MAX_LAG_MS
            t0s.append(max(lo - EP_PAD_MS, ws - PRE_MS)); t1s.append(min(w[2] + EP_PAD_MS, we))
    cs['t0'], cs['t1'] = t0s, t1s
    cs['episode_id'] = [f'{d}_{m}_r{rk}_e{e:03d}' for d, m, rk, e in zip(cs.day, cs.mid, cs['rank'], episode_ids(cs))] if len(cs) else []
    crow, ledger, logic, tl, srow, strow, eprow = [], [], [], [], [], [], []
    inputs = {}
    n_mk = 0
    stage('case_windows')
    btc_days = sorted(set(cs.day))
    for d in btc_days:                                        # the BTC grid once per day with cases (its own stage: the projection unit)
        if hasattr(ctx, '_btc'):
            ctx._btc(d)
    stage('btc_load')
    mg = pd.read_csv(C.RECEIPT / 'matched_gap_r1' / 'per_market.csv')
    mg = mg[(mg['view'] == 'carry')]
    cc_inputs = {}
    for (d, m), cm in cs.groupby(['day', 'mid'], sort=True):
        O, Fl, Lk = life[(d, m)]
        dp = depth_paths(R, d, m)
        ws, we, mst = mt[d][m]
        M = None
        if prov['provider'] == 'chain_context':                 # one book build: chain_context's (pf_mt_lifecycle._book, unchanged)
            M = ctx.load(d, m, R)
            refuse(int(M.ws) == ws and int(M.we) == we, f'chain_context market times differ from markets_{d} for {m} (refused)')
            refuse(M.depth is not None, f'chain_context loaded no depth for BOOK market {m} (refused)')
            B = M.depth['B']
            cc_inputs[f'{d}_{m}'] = dict(inputs=M.inputs, checks=M.checks, cost=M.cost)
        else:
            frames, levels = pd.read_parquet(dp['frames']), pd.read_parquet(dp['levels'])
            B = LC._book(frames, levels, int(m), ws, we)
            del frames, levels
        W = windows_of(Fl)
        Mk = dict(day=d, mid=int(m), ws_ms=ws, we_ms=we, orders=O, fills=Fl, links=Lk, B=B, touch=touch_arrays(B), FF=fill_frames(Fl, B),
                  settle=settle, L={}, inv={}, M=M)
        for a in set(cm.addr):
            Lg = legs_frame(fv[(fv.mid == m) & (fv.addr_l == a)], W)
            Mk['L'][a], Mk['inv'][a] = Lg, Inventory(Lg)
            cnt['legs_without_window'] += int((Lg.wst == 'no_window').sum())
        r1, r2, r3_, r4, r5, r6 = market_cases(Mk, cm, ctx, cnt)
        crow += r1; ledger += r2; logic += r3_; tl += r4; srow += r5; strow += r6
        for e, ce in cm.groupby('episode_id', sort=True):
            a = ce.addr.iloc[0]; t0, t1 = int(ce.t0.min()), int(ce.t1.max())
            econ = episode_econ(Mk['L'][a], Mk['inv'][a], t0, t1)
            g = mg[(mg['m'] == int(m)) & (mg['party'] == f'rank{LOW3.index(a) + 1}')]
            ctxm = (dict(mg_status=str(g['status'].iloc[0]), mg_close_net=_fin(g['close_net'].iloc[0]), mg_resid=_fin(g['resid'].iloc[0]),
                         mg_total=_fin(g['total'].iloc[0]), mg_close_net_c_sh=_fin(g['close_net_c_sh'].iloc[0])) if len(g) else dict(mg_status='no_row'))
            eprow.append(dict(episode_id=e, day=d, mid=int(m), addr=a, rank=LOW3.index(a) + 1, t0=t0, t1=t1, n_cases=len(ce),
                              case_keys='|'.join(f'{k}:{x[:12]}' for k, x in zip(ce.kind, ce.key)), **econ, **ctxm))
            Lg = Mk['L'][a]
            ins = Lg[(Lg.settle_ms >= t0) & (Lg.settle_ms <= t1)]
            xs = ins.settle_ms.to_numpy(np.int64)
            pos = Mk['inv'][a].recorded(xs + 1) if len(xs) else np.zeros(0)
            for r_, p_ in zip(ins.itertuples(index=False), pos):
                tl.append(dict(episode_id=e, t_ms=int(r_.whi), event=f'leg_{r_.role}', case_key='',
                               detail=f"{'buy' if r_.s > 0 else 'sell'} {r_.q:.2f} @ {r_.p:.2f} hash {str(r_.hash)[:10]} sid {r_.sid[:13]} lo_hard {r_.xlo} "
                                      f"status {r_.status}{' self' if r_.self_trade else ''}; pos after {p_:+.2f}"))
        inputs.update({f'depth/{d}_{m}_{k}.parquet': _sha(p) for k, p in dp.items()})
        n_mk += 1
        del B, Mk, M
        stage(f'cases_market_{n_mk:02d}')
    lg = pd.DataFrame(logic, columns=list(LOGIC_COLS))
    if len(lg):
        lg = lg.sort_values(['episode_id', 'decision_window_lo_ms', 'case_key', 'link', 'mechanism'], kind='stable').reset_index(drop=True)
        lg['link_seq'] = lg.groupby('episode_id').cumcount()
        nxt = []
        for e, g in lg.groupby('episode_id', sort=False):
            lk = list(g.case_key + '#' + g.link)
            nxt += [next((y for y in lk[i + 1:] if y != lk[i]), '') for i in range(len(lk))]
        lg['next_link'] = nxt
        E = pd.DataFrame(eprow).set_index('episode_id')
        lg = lg.join(E[[c for c in E.columns if c.startswith('ep_') or c.startswith('mg_')]], on='episode_id')
    stage('logic')
    # ---- contrasts (contract 6) ----
    CR = pd.DataFrame(crow)
    con = []
    if len(CR) and len(lg):
        for (rk, kind), g in CR.groupby(['rank', 'kind']):
            if kind == 'order':
                tch = lg[(lg.link == 'POST') & (lg.mechanism == 'touch_join_or_improve')].set_index('case_key').identification_status
                ivs = lg[(lg.link == 'POST') & (lg.mechanism == 'inventory_offset')].set_index('case_key').identification_status
                X = pd.DataFrame(dict(d1=g.case_key.map(tch), d2=g.case_key.map(ivs)))
                dims = ('touch_join_or_improve', 'inventory_offset')
            else:
                dr = lg[(lg.link == 'TAKE') & (lg.mechanism == 'directional')].set_index('case_key').identification_status
                X = pd.DataFrame(dict(d1=g.cls, d2=g.case_key.map(dr)))
                dims = ('take_class', 'directional')
            ct = X.groupby(['d1', 'd2'], dropna=False).size()
            ok = X.d1.nunique() >= 2 and X.d2.nunique() >= 2
            for (a1, a2), n in ct.items():
                con.append(dict(rank=int(rk), kind=kind, dim1=dims[0], v1=a1, dim2=dims[1], v2=a2, n=int(n), overlap_ok=bool(ok)))
    # ---- own-order instrument check (contract 9) ----
    gt = pd.DataFrame()
    own_p = R / 'truth/v2/own_orders.parquet'
    if own_p.exists():
        own = pd.read_parquet(own_p, columns=list(GT_COLS))
        own = own[(own.mk_hash.astype(str) != '') & own.link_status.isin(['unique', 'unique_by_time'])]
        lob = {}
        for mm in sorted(set(own.mid.astype(int))):
            p = life_paths(R, C.NOV8, mm)['orders']
            if p.exists():
                lob[mm] = pd.read_parquet(p)
                inputs[f'life/{C.NOV8}_{mm}_orders.parquet'] = _sha(p)
        gt = gt_rows(own, lob)
        inputs['truth/v2/own_orders.parquet'] = _sha(own_p)
    stage('gt')
    # ---- inputs, checks, outputs ----
    inputs['econ/fills_valued.parquet'] = check_fv(R / 'econ/fills_valued.parquet')
    for d in days:
        inputs[f'markets_{d}.parquet'] = _sha(R / f'markets_{d}.parquet')
        for m in book[d]:
            for k, p in life_paths(R, d, m).items():
                inputs[f'life/{d}_{m}_{k}.parquet'] = _sha(p)
    for d in sorted(set(cs.day)):
        inputs[f'btc/btc_{d}.parquet'] = _sha(R / 'btc' / f'btc_{d}.parquet')
    inputs['econ/identities.json'] = _sha(R / 'econ/identities.json')
    inputs['manifest.json'] = _sha(BOOK_MANIFEST)
    inputs['matched_gap_r1/per_market.csv'] = _sha(C.RECEIPT / 'matched_gap_r1' / 'per_market.csv')
    stage('hash_inputs')
    cov = coverage_rows(census, lcensus, tcensus, pd.DataFrame(srow), pd.DataFrame(strow))
    LD = pd.DataFrame(ledger)
    ord_keys = set(cs[cs.kind == 'order'].kind + '|' + cs[cs.kind == 'order'].day + '|' + cs[cs.kind == 'order'].mid.astype(str) + '|'
                   + cs[cs.kind == 'order'].addr + '|' + cs[cs.kind == 'order'].key)
    checks = dict(
        n_selected=int(len(sel)), selected_le_96=bool(len(sel) <= 96), per_addr_day_kind_le_4=bool((man.n_selected <= N_PER).all()),
        selection_keys_unique=bool(not sel.duplicated(['kind', 'day', 'mid', 'addr', 'key']).any()),
        every_case_reconstructed=bool(len(CR) == len(cs)), census_orders=int(len(census)), census_links=int(len(lcensus)), census_takes=int(len(tcensus)),
        census_orders_equal_probe_4149=bool(len(census) == 4149),           # logs/mt_oe_inspect.log: every top-3 hash on the 19 BOOK markets
        logic_statuses_valid=bool(lg.identification_status.isin(STATUSES).all()) if len(lg) else None,
        logic_rows_have_episode=bool(lg.episode_id.isin(set(cs.episode_id)).all()) if len(lg) else None,
        ledger_three_hypotheses_per_order=(bool((LD[LD.row_kind == 'size_end'].groupby('case_key').size() == 3).all()
                                                and set(LD[LD.row_kind == 'size_end'].case_key) == ord_keys) if len(LD) else None),
        census_priors_within_feasible=bool(census.priors_feasible.all()) if len(census) else None,
        census_clock_refused_not_binned=(bool(int(cov[(cov['scope'] == 'census') & (cov['event'] == 'posting') & (cov['clock'] == 'acceptance_union')
                                                       & (cov['category'] == 'refused_clock')]['n_units'].sum()) == int(census['clock_bad'].sum()))
                                         if len(census) and len(cov) else None))
    T = Counter()
    if len(census):
        T.update({f'census_size_status:{k}': int(v) for k, v in census.size_status.value_counts().items()})
        T.update({f'census_post_class:{k}': int(v) for k, v in census.post_class.value_counts().items()})
        T.update({f'census_prior_sensitive3:{k}': int(v) for k, v in census.prior_sensitive3.value_counts().items()})
        T['census_clock_refused'] = int(census.clock_bad.sum())
    if len(lcensus):
        T.update({f'census_link_status:{k}': int(v) for k, v in lcensus.status.value_counts().items()})
    if len(CR):
        for col in ('size_status', 'front_class', 'nomask_status', 'prior_sensitive', 'cls', 'timing_sensitive'):
            if col in CR:
                T.update({f'sample_{col}:{k}': int(v) for k, v in CR[col].dropna().value_counts().items()})
    if len(lg):
        T.update({f'logic:{a}:{b}': int(v) for (a, b), v in lg.groupby(['mechanism', 'identification_status']).size().items()})
    if len(gt) and 'verdict' in gt:
        T.update({f'gt_verdict:{k}': int(v) for k, v in gt.verdict.value_counts().items()})
    tmp = out_root / f'.{name}.partial{os.getpid()}'
    refuse(not tmp.exists(), f'{tmp} exists')
    (tmp / 'timelines').mkdir(parents=True)
    man.assign(manifest_sha256=man_sha).to_csv(tmp / 'manifest.csv', index=False)
    sel_k = sel.assign(case_key=sel.kind + '|' + sel.day + '|' + sel.mid.astype(str) + '|' + sel.addr + '|' + sel.key)
    if len(CR):
        cases_df = sel_k.merge(CR.drop(columns=[c for c in ('kind', 'day', 'mid', 'addr', 'key', 'rank', 'pick', 'round') if c in CR.columns]),
                               on='case_key', how='left')
        cases_df['reconstructed'] = cases_df.case_key.isin(set(CR.case_key))
    else:
        cases_df = sel_k.assign(reconstructed=False)
    cases_df.loc[~cases_df.reconstructed, 'status'] = 'not_reconstructed_in_this_run' if pilot else 'missing'
    _write_csv(cases_df, tmp / 'cases.csv')
    _write_csv(pd.DataFrame(ledger), tmp / 'order_ledger.csv')
    _write_csv(lg, tmp / 'episodes.csv')
    _write_csv(pd.DataFrame(eprow), tmp / 'episode_summary.csv')
    _write_csv(cov, tmp / 'coverage.csv')
    _write_csv(pd.DataFrame(con), tmp / 'contrasts.csv')
    _write_csv(gt, tmp / 'gt_check.csv')
    if len(census):
        census.drop(columns=['fill_w_hard', 'fill_w_ev']).to_csv(tmp / 'census_orders.csv', index=False)
    _write_csv(lcensus, tmp / 'census_links.csv')
    TL = pd.DataFrame(tl)
    if len(TL):
        for e, g in TL.groupby('episode_id', sort=True):
            g.sort_values(['t_ms', 'event'], kind='stable').to_csv(tmp / 'timelines' / f'{e}.csv', index=False)
    stage('write')
    cpu = time.process_time() - t_all
    rec = dict(version=VERSION, contract=str(CONTRACT.relative_to(C.REPO)), mode='pilot' if pilot else 'full', pilot=pilot, run=C.RUN, git_head=head,
               freeze=frz, frozen=frozen, context_provider=prov, chain_context_markets=cc_inputs, manifest_sha256=man_sha, n_cases_selected=int(len(sel)),
               n_cases_reconstructed=int(len(CR)), n_markets_with_cases=n_mk, manifest=man.to_dict('records'), inputs=inputs,
               counts=dict(sorted({**cnt, **T}.items())), checks=checks, stage_cpu_s=stage_cpu, cpu_s=round(cpu, 3), vmhwm_mib=round(vmhwm_mib(), 1),
               guard_mib=C.GUARD_MIB, slot=os.environ.get('PF_SLOT'), argv=sys.argv,
               declared=dict(SEED=SEED, N_PER=N_PER, TOL=TOL, LATE_MS=LATE_MS, EARLY_MS=EARLY_MS, LOT=LOT, D_PROFILES=D_PROFILES, BIN_EDGES=BIN_EDGES,
                             EP_PAD_MS=EP_PAD_MS, MOVE=dict(MOVE), LB_MS=LB_MS, BURST=(BURST_N, BURST_MS), KEEP_REACT_MS=KEEP_REACT_MS),
               premises=['one hash = one signed order (H1)', 'a removed order never re-enters; a resting order never grows (H2)',
                         'orders are cancelled whole (the x_size_ub premise)', 'book / UUID / collector clocks aligned (FINDINGS C23)',
                         'venue-state features (book at upd_ms, BTC at E_ms) are not what the maker received'])
    if pilot:
        n_cases_p = max(len(CR), 1)
        case_cpu = sum(v for k, v in stage_cpu.items() if k.startswith('cases_market_')) + stage_cpu.get('logic', 0.)
        btc_day = stage_cpu.get('btc_load', 0.) / max(len(btc_days), 1)
        fixed = sum(v for k, v in stage_cpu.items() if not k.startswith('cases_market_') and k not in ('logic', 'btc_load'))
        per_mk = case_cpu / max(n_mk, 1)
        by_mk = fixed + btc_day * len(days) + per_mk * N_BOOK
        by_case = fixed + btc_day * len(days) + case_cpu / n_cases_p * 96
        rec['projection'] = dict(
            basis='pilot process_time per stage: the fixed stages (setup, selection, census of every top-3 row, GT check, input hashing, write) '
                  'already cover the full run; BTC per day x 4 days; the case stage (book + cases + context + episodes + logic) scaled by markets '
                  'with cases (<= 19) and, separately, by cases (<= 96); the larger is the projection',
            fixed_s=round(fixed, 3), btc_per_day_s=round(btc_day, 3), case_stage_s=round(case_cpu, 3), markets_in_pilot=n_mk,
            cases_in_pilot=int(len(CR)), per_market_s=round(per_mk, 3), per_case_s=round(case_cpu / n_cases_p, 3),
            full_run_s_by_markets=round(by_mk, 2), full_run_s_by_cases=round(by_case, 2), full_run_s=round(max(by_mk, by_case), 2),
            full_run_core_h=round(max(by_mk, by_case) / 3600., 5),
            note='process_time excludes the interpreter start and imports before main (about 1 s, in run_light cpu_user_s)')
    (tmp / 'receipt.json').write_text(json.dumps(rec, indent=1, default=_jsonable))
    os.rename(tmp, out)
    log(json.dumps(dict(out=str(out), cpu_s=rec['cpu_s'], stage_cpu_s=stage_cpu, n_cases=len(CR), projection=rec.get('projection'),
                        vmhwm_mib=rec['vmhwm_mib']), indent=1, default=_jsonable))
    return rec


if __name__ == '__main__':
    a = sys.argv[1:]
    need(len(a) >= 1 and a[0] == 'run', 'usage: MT_RUN=v2 order_estimator.py run [--pilot DAY RANK]')
    if '--pilot' in a:
        i = a.index('--pilot')
        need(len(a) >= i + 3, 'usage: --pilot DAY RANK')
        run(pilot=(a[i + 1], int(a[i + 2])))
    else:
        need(len(a) == 1, 'usage: MT_RUN=v2 order_estimator.py run [--pilot DAY RANK]')
        run()
