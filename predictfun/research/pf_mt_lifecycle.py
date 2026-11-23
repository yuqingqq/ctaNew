"""MAKER-TRACE-RECONSTRUCTION-v1 lifecycle (contract.md section 5; P-2026-005, session aea5bfb9, 2026-11-22). Per market: every maker
order's fills, presence, size, posting, queue and removal implied by the public match records and the full aggregated YES-space book
frames; each inference is classed observed / constrained / assumption_sensitive / unresolved (pf_mt_common.CLASSES). Pure core:
run_market(legs, records, frames, levels, market) -> (fills, orders, links, summary); the CLI only reads, guards and writes once.
Declared premises and rules (none fitted; constants below):
 T  fill window per maker leg (all legs of one record share it: one settlement = one match instant). hi = min(settle_ms,
    first_recv_ns // 1e6) (settle_ms <= 0 -> receipt, status hi_from_receipt). Evidence frame E = the FIRST usable frame in
    [hi - MAX_LAG_MS, F_hi] (F_hi = first usable frame with upd_ms >= hi) that could reflect the record: (a) a settlementsPending
    increment at one of its YES ticks, either side, of at least its smallest leg value (shares x YES price) less max(1.1 %, $0.01)
    (pf_step1_timing's tolerance; merged bookings included), or (b) a displayed-size drop at the maker's YES side / tick of at least the
    record's shares there less max(10 %, 0.5 sh) (pf_step1_booktime), or an unknown cell; t_method names the evidence at E (pend_inc
    preferred). The first, not the closest, candidate: an earlier E only loosens lo. The reviewed exact-value-first rule is kept as
    x_lo_exact_ms only. lo = upd_ms of the usable frame before E (f_before); f_after = F_hi. Premise of lo: a frame reflects every
    match at or before its upd_ms and the true reflecting frame shows a booking or a drop (not both masked). Masking (a same-frame refill
    plus a netted booking) can put this lo AFTER the match, so lo_ms is a scenario refinement only (S_FIFO_LO placement, the queue frame):
    every observed / constrained claim uses the evidence-free lower end x_lo_hard_ms = hi - MAX_LAG_MS (declared lag premise; -1 for a
    receipt-hi fill, which also gets no evidence lo: the receipt may follow the match by seconds). No evidence -> lo = -1, status
    fill_lo_unbounded. Book, UUID and receipt clocks aligned (FINDINGS C23). Event SEQUENCE (which fill or record is later) is the settle
    order (hi, then record rank) in every scenario, receipt-hi fills tied (no place in it); a scenario's fill clock only places fills.
 B  usable frames: status flags within USABLE_FRAME (pf_mt_extract WS_FLAGS that keep a full snapshot: dup, same_ms_changed, new_conn,
    gap, and off_grid / dup_level whose affected cells are located); any other flag (bad_frame, mid_mismatch, out_of_order, unknown), a
    pending entry out of range, an invalid level row, an unlocated off_grid / dup_level or no level at all inside [ws, we] (a wipe)
    excludes the frame (unknown, never an empty book; counted). A level whose status (or x_status) is not ok, or a (frame, side, tick) cell with several rows, makes that cell UNKNOWN (NaN:
    no bound, no evidence against; a possible drop candidate). A tick absent from a usable frame shows 0; usable frames more than GAP_MS
    apart bound an unknown span (nothing inferred inside it; a removal span crossing one, or an unknown cell, adds missing_evidence).
 P  placeable legs: a YES side, a YES price in (0, 1), finite positive shares, no bad_leg flag; on-grid = a whole-cent price. Others are
    kept as rows (status unplaceable / subcent_price) but enter no book cell; a record with an unplaceable leg, mixed maker sides or a
    maker on the taker's side gets no price-time inference.
 H1 a hash has one (signer, YES side, tick, price, quote, indexSet), also against its own taker legs, else identity_conflict; a hash twice
    in one record (dup_hash_in_record), with an unplaceable / off-grid / malformed leg (unplaceable_leg, subcent_price, malformed_identity)
    is likewise kept, never used for book presence / capacity / priority (post / queue / removal unresolved); its persistence span needs
    only its fills and is kept (observed) unless the hash may not be one order (NO_PRESENCE: identity_conflict, malformed_identity).
 H2 proven presence: frames [min f_after, max fb_h] (hard; fb_h = the last frame before hi - MAX_LAG_MS) / [min pos + 1, max pos] under
    a scenario's fill clock (pos = the last frame not reflecting the fill); persistence (persist_lo / hi) = the certain span [min hi, max
    x_lo_hard_ms], the same in every scenario row; observed only when it is non-empty (else -1 / -1, unresolved).
 H3 at every usable frame, shown(side, tick) >= the sum over the orders proven present there of their fills after it (less TOL_SH per
    order accounted, _tol: the venue shows some orders 0.01 sh short and the shortfalls add up; a posting test adds the order itself), else a status
    (capacity_contradiction = hard window logic, clock_capacity_conflict = only under a scenario's clock), never fitted away: the order's
    capacity claims are then void in that context (capacity_bounds_void), the violated cells unknown for every other order, and none of
    its size / posting / removal classes is constrained. Under each scenario: posting lower bound = the latest frame before presence whose room (shown less the others' needs) < the order's fills
    (post_method capacity, joint form accepted as contract H3); size upper bound x_size_ub = cum + min slack over presence; zero slack =
    capacity_tight: GONE by the first frame after its last fill, exhausted OR cancelled (no size_exact from the book);
    removal upper bound = the first frame after the last fill with no room for the order (book_gone).
 S  scenarios (pf_mt_common.SCENARIOS). Fill clock hi: reflected at f_after; lo: just after f_before (unbounded: hi - MAX_LAG_MS).
    makers[] order (mk_idx) is data, never evidence (amendment 1: real records list makers worst-first 66 %, best-first 8 %, neither 26 %).
    price_time adds, from prices and later fills only: (1) in a record spanning >= 2 YES prices every maker strictly better for the taker
    than the record's worst maker price was consumed: exhausted there (size_exact = cum); (2) of the k >= 2 makers at the record's worst
    price at most one keeps a balance: if exactly one fills again later (a strictly later settle ms; fills of one settle ms are unordered
    and never 'later', so a tie never voids a group: FIFO itself orders a tied fill first) the other k - 1 are exhausted there and the
    survivor had their shares ahead (ahead_lo), if none does only the count constraint is reported; an order exhausted at its last settle ms
    is claimed whichever of its fills there carries the rule, but exhausted in two records of that ms it is FIFO-inconsistent
    (fifo_tie_conflict, no claim); an order with a receipt-hi fill has no reliable sequence (no rule, out of T1 / T2); (3) after a
    fill an order with a balance is at the front of its price, so the first later record (taker of another signer) reaching its price on
    its side bounds its removal (fifo_front) and a later fill's queue frame has nothing ahead; a record trading strictly through its price
    before its first fill bounds its posting (fifo_through). Queue at the queue frame (f_before, else the clock's frame before the fill)
    of the first fill (min hi, then record order): [ahead_lo, ahead_hi] = [rule-2 shares else 0, shown there - the order's fills after it];
    a raw ahead_hi in [-TOL_SH, 0) is clipped to 0 (ahead_hi_clipped), below -TOL_SH the order was not resting there (H3): NaN
    (ahead_hi_negative). Priority tests: T1 price = rule-1 makers whose order fills again later; T2 time = rule-2 groups with >= 2 members
    filling again later; each on the settle order (strictly later settle ms) and window-certain (x_lo_hard_j >= hi_i), with
    denominators (tied legs / groups counted and included; receipt-hi ones excluded). Contradictions: queue_contradiction (ahead_lo > ahead_hi), post_after_first_fill. Violators get statuses
    fifo_reappearance / fifo_time_violation and no exhaustion claim.
 R  removal_set = every REMOVAL member not excluded: exhausted always (original size unknown); cancelled_or_expired unless exhaustion is
    proven by an S_FIFO rule (fifo_price / fifo_time; the book - book_gone, capacity_tight - proves only GONE); end_censored when no removal bound up to the last usable frame (or the only bound is a
    fifo_front record past it); missing_evidence when the span from max f_after crosses an unknown span or cell, or the book ends > GAP_MS
    before we. removal window = [max x_lo_hard_ms, the bound]; removal_method names the
    evidence (fifo_price, fifo_time, capacity_tight, book_gone, fifo_front, book_empty_after = contract's shown(p) = 0 after the fill,
    unprinted_drop = contract's drop without a print at p, end_open). Never a single invented time.
 C  classes compare in code the three scenario results with the scenario-free H1-H3 result (_hard: window logic, valid for any fill time
    inside its window and any priority rule; each scenario reports the tighter of it and its own bound): persistence (>= 2 fills) and the
    size lower bound are observed; every scenario equal to a non-trivial scenario-free result = constrained; any scenario different =
    assumption_sensitive (two clocks agreeing by coincidence are not H1-H3: found by the selftest's FIFO simulator); no non-trivial bound
    anywhere = unresolved (posting / queue / removal); queue has no scenario-free result: S_NOPRIO reports the trivial [0, NaN], so the
    queue is assumption_sensitive exactly when a FIFO scenario reports anything else, else unresolved.
 L  x_links per signer and YES side: each hash's latest predecessor (last fill strictly before its first fill) at the same tick (refill)
    and at another tick (reprice), with the gap; descriptive only (unfilled orders are invisible).
Statuses: '+'-joined sorted flags ('ok' = none), extract flags carried as leg_<flag>; removal_set '|'-joined sorted REMOVAL members.
Usage (run_light, PF_SLOT=claude3): pf_mt_lifecycle.py market <day> <mid>      selftest: pf_mt_lifecycle_selftest.py"""
import hashlib
import json
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pf_mt_common as C  # noqa: E402

need = C.need
GAP_MS = 1000                       # usable frames further apart bound an unknown span (contract section 5)
TOL_SH = 0.01                       # displayed sizes are rounded to 0.01 sh (EXPLORATION_2026-11-01 sec. 11: 395.97 - 114.334 -> 281.64),
#                                     PER ORDER: real BOOK cells show some orders 0.01 short and the shortfalls add up (2026-10-24 3697902
#                                     ask 62: 10 -> 9.99, then 279.98 for 10 + 145 + 125), so H3 allows TOL_SH x the orders accounted (_tol)
PEND_REL, PEND_ABS = 0.011, 0.01    # pending value match (pf_step1_timing)
NO_PRESENCE = frozenset(('identity_conflict', 'malformed_identity'))   # the hash may not be one order: no persistence span from its fills
DROP_REL, DROP_ABS = 0.10, 0.5      # drop size tolerance (pf_step1_booktime E4)
USABLE_FRAME = frozenset(('dup', 'same_ms_changed', 'new_conn', 'gap', 'off_grid', 'dup_level'))
NT = 101
SIDE = {'bid': 0, 'ask': 1}
SCEN = tuple(C.SCENARIOS)
TRIVIAL_RM = frozenset(('cancelled_or_expired', 'end_censored', 'exhausted'))
INVALID = frozenset(('identity_conflict', 'dup_hash_in_record', 'unplaceable_leg', 'subcent_price', 'malformed_identity', 'amount_mismatch',
                     'leg_recon_mk_sum', 'leg_recon_tk_amount'))
REC_FLAGS = ('incomplete_record', 'amount_mismatch', 'maker_side_mixed', 'taker_side_mismatch', 'hi_from_receipt', 'record_unplaceable_leg')
NAN = float('nan')
for _s, _d in C.SCENARIOS.items():
    need(_d['fill_clock'] in ('hi', 'lo') and _d['priority'] in ('price_time', 'none'), f'scenario {_s} not understood')


def vmhwm_mib():
    with open('/proc/self/status') as f:
        for line in f:
            if line.startswith('VmHWM:'):
                return int(line.split()[1]) / 1024.
    return 0.


def guard(where, limit=C.GUARD_MIB):
    m = vmhwm_mib()
    if m > limit:
        raise MemoryError(f'self-guard: VmHWM {m:.0f} MiB > {limit:.0f} MiB at {where}')
    return m


def _cols(df, name):
    miss = [c for c in C.SCHEMAS[name] if c not in df.columns]
    need(not miss, f'{name} missing columns {miss}')


def _split(st):
    return {x for x in str(st).split('+') if x and x != C.STATUS_OK}


def _flags(s):
    return '+'.join(sorted(s)) if s else C.STATUS_OK


def _conform(df, name):
    """Schema columns first (exact dtypes, no missing str / int values), then the x_ extras."""
    sch = C.SCHEMAS[name]
    extra = [c for c in df.columns if c not in sch]
    need(all(c.startswith('x_') for c in extra), f'{name}: extra columns without the x_ prefix {[c for c in extra if not c.startswith("x_")]}')
    _cols(df, name)
    out = df[list(sch) + extra].copy()
    for c, d in sch.items():
        if d != 'float64':
            need(not out[c].isna().any(), f'{name}.{c} has missing values')
        out[c] = out[c].astype(d)
    return out


def _pend(x):
    """frames.pend_bids / pend_asks text (pf_mt_extract: json.dumps of the raw settlementsPending list) -> [(price, usd)], or None when an
    entry is not a number pair with price in (0, 1) and usd >= 0. A text that is not a JSON list of pairs is refused (an extract defect)."""
    if x is None or (not isinstance(x, str) and pd.isna(x)) or x in ('', '[]'):
        return []
    try:
        v = json.loads(x)
    except ValueError:
        v = None
    need(isinstance(v, list) and all(isinstance(e, list) and len(e) == 2 for e in v), f'pend text is not a JSON [[price, usd], ...] list: {str(x)[:60]!r}')
    num = lambda z: isinstance(z, (int, float)) and not isinstance(z, bool)  # noqa: E731
    if not all(num(p) and num(u) and 0 < p < 1 and np.isfinite(u) and u >= 0 for p, u in v):
        return None
    return [(float(p), float(u)) for p, u in v]


# ------------------------------------------------------------------ inputs ------------------------------------------------------------------
def _prep(legs, records, mid):
    """Refuse extract-invariant breaches -> (L sorted by settle_ms, sid, mk_idx; placeable; off-grid; extract flags per leg)."""
    _cols(legs, 'legs'); _cols(records, 'records')
    need((legs.mid == mid).all(), f'legs hold a market other than {mid}')
    need((records.mid == mid).all(), f'records hold a market other than {mid}')
    need(records.sid.is_unique, 'duplicate record sid')
    need(not legs.duplicated(['sid', 'mk_idx']).any(), 'duplicate leg key (sid, mk_idx)')
    need(((legs.mk_idx >= 0) & (legs.mk_idx < legs.n_makers)).all(), 'leg mk_idx outside [0, n_makers)')
    need((legs.first_recv_ns > 0).all(), 'leg without a first receipt time')
    need(legs.sid.isin(records.sid).all(), 'legs reference a sid absent from records')
    r = records.set_index('sid').loc[legs.sid.to_numpy()]
    for c in ('settle_ms', 'first_recv_ns', 'n_makers', 'tk_hash', 'tk_signer', 'tk_y_side'):
        need((r[c].to_numpy() == legs[c].to_numpy()).all(), f'legs and records disagree on {c}')
    L = legs.sort_values(['settle_ms', 'sid', 'mk_idx'], kind='stable').reset_index(drop=True)
    q = L.shares.to_numpy(float); yp = L.y_price.to_numpy(float); k = L.y_ptick.to_numpy()
    xf = [_split(x) for x in L.status]
    place = (L.y_side.isin(list(SIDE)).to_numpy(bool) & np.isfinite(q) & (q > 0) & np.isfinite(yp) & (yp > 0) & (yp < 1)
             & np.array(['bad_leg' not in f for f in xf], bool))
    for i in np.flatnonzero(place):
        qt, ix, p = L.mk_quote.iat[i], int(L.mk_index_set.iat[i]), float(L.mk_price.iat[i])
        s2, p2 = C.yes_space(qt, ix, p)
        need(s2 == L.y_side.iat[i] and abs(p2 - yp[i]) < 1e-9, f'leg YES side / price disagree with yes_space({qt}, {ix}, {p})')
    need((~place | ((np.abs(yp * 100 - k) <= 0.5 + 1e-9) & (k >= 0) & (k < NT))).all(), 'leg y_ptick inconsistent with y_price')
    sub = place & ((np.abs(yp * 100 - k) > 1e-6) | np.array(['off_grid' in f for f in xf], bool))
    return L, place, sub, xf


def _book(frames, levels, mid, ws_ms, we_ms):
    """usable frames sorted by (upd_ms, fid): t, fid, shown / pending [F, side, tick] (NaN = unknown cell), gap flags (gap[f]: frames
    f-1, f more than GAP_MS apart); exclusions counted by reason."""
    _cols(frames, 'frames'); _cols(levels, 'levels')
    frames = frames.reset_index(drop=True); levels = levels.reset_index(drop=True)
    need((frames.mid == mid).all() and (levels.mid == mid).all(), f'frames / levels hold a market other than {mid}')
    need(frames.fid.is_unique, 'duplicate frame fid')
    need(levels.fid.isin(frames.fid).all(), 'levels reference an unknown frame fid')
    st = frames.status.astype(str).to_numpy(object)
    fl = [_split(x) for x in st]
    use = np.array([f <= USABLE_FRAME for f in fl], bool)
    why = Counter(st[~use].tolist())
    pend = {}
    for i in np.flatnonzero(use):
        pb, pa = _pend(frames.pend_bids.iat[i]), _pend(frames.pend_asks.iat[i])
        if pb is None or pa is None:
            use[i] = False; why['pend_out_of_range'] += 1
        else:
            pend[i] = (pb, pa)
    lv = levels[levels.fid.isin(frames.fid[use])].copy()
    lv['fpos'] = pd.Index(frames.fid.to_numpy()).get_indexer(lv.fid.to_numpy())
    sh = lv.shares.to_numpy(float)
    okv = (lv.side.isin(list(SIDE)) & lv.ptick.between(0, NT - 1)).to_numpy(bool) & np.isfinite(sh) & (sh >= 0)
    for i in sorted(set(lv.fpos[~okv].tolist())):
        use[i] = False; why['bad_level'] += 1
    lv = lv[okv & use[lv.fpos.to_numpy()]]
    rok = (lv.status.astype(str) == C.STATUS_OK).to_numpy(bool)
    if 'x_status' in lv.columns:
        rok &= (lv.x_status.astype(str) == C.STATUS_OK).to_numpy(bool)
    dup = lv.duplicated(['fid', 'side', 'ptick'], keep=False).to_numpy(bool)
    nok_f, dup_f = set(lv.fpos[~rok].tolist()), set(lv.fpos[dup].tolist())
    for i in np.flatnonzero(use):                       # an off_grid / dup_level frame must show where: else the whole frame is unknown
        for flag, located in (('off_grid', nok_f), ('dup_level', dup_f)):
            if use[i] and flag in fl[i] and i not in located:
                use[i] = False; why[f'{flag}_unlocated'] += 1
    has_lv = np.zeros(len(frames), bool); has_lv[lv.fpos.to_numpy()] = True
    upd = frames.upd_ms.to_numpy(np.int64)
    for i in np.flatnonzero(use & ~has_lv & (upd >= ws_ms) & (upd <= we_ms)):    # a level-less book inside the window is a wipe, not depth 0
        use[i] = False; why['empty_book_in_window'] += 1
    lv = lv[use[lv.fpos.to_numpy()]]
    rok = (lv.status.astype(str) == C.STATUS_OK).to_numpy(bool)
    if 'x_status' in lv.columns:
        rok &= (lv.x_status.astype(str) == C.STATUS_OK).to_numpy(bool)
    unk = ~rok | lv.duplicated(['fid', 'side', 'ptick'], keep=False).to_numpy(bool)
    fo = frames[use].sort_values(['upd_ms', 'fid'], kind='stable')
    t = fo.upd_ms.to_numpy(np.int64); fid = fo.fid.to_numpy(np.int64); F = len(fo)
    need(F == 0 or t[0] > 0, 'frame upd_ms not positive')
    shown = np.zeros((F, 2, NT)); pnd = np.zeros((F, 2, NT))
    if len(lv):
        ix = (pd.Index(fid).get_indexer(lv.fid.to_numpy()), lv.side.map(SIDE).to_numpy(int), lv.ptick.to_numpy(int))
        shown[tuple(x[~unk] for x in ix)] = lv.shares.to_numpy(float)[~unk]
        shown[tuple(x[unk] for x in ix)] = np.nan
    for j, i in enumerate(fo.index):
        for si, entries in enumerate(pend[i]):
            for p, u in entries:
                pnd[j, si, int(round(p * 100))] += u
    gap = np.zeros(F, bool)
    gap[1:] = np.diff(t) > GAP_MS
    conn = fo.conn.astype(str).to_numpy()
    return dict(F=F, t=t, fid=fid, shown=shown, pend=pnd, gap=gap, n_rows=int(len(frames)), excluded=dict(sorted(why.items())),
                n_gaps=int(gap.sum()), unknown_cells=int(np.isnan(shown).sum()), n_conn=int(len(set(conn))),
                n_conn_switch=int((conn[1:] != conn[:-1]).sum()) if F > 1 else 0, span_ms=[int(t[0]), int(t[-1])] if F else None)


# ------------------------------------------------------------------ fill windows + record-level S_FIFO facts ------------------------------------------------------------------
def _timing(L, records, B, place, sub):
    """Per leg (L order): hi, lo, fb, fa (usable-frame indices; fb -1 = none, fa F = none), t_method, evidence frames, record rank and
    flags, and the price facts of amendment 1 (exh_p: strictly better than its record's worst maker price; wk: the record's number of
    worst-price makers on a worst-price leg); per record the facts the price-time rules need; descriptive makers[] listing counts."""
    F, t, shown, pend, gap = B['F'], B['t'], B['shown'], B['pend'], B['gap']
    N = len(L)
    q = L.shares.to_numpy(float); yp = L.y_price.to_numpy(float)
    s = np.where(place, L.y_side.map(SIDE).fillna(-1).to_numpy(), -1).astype(int); k = np.where(place, L.y_ptick.to_numpy(), -1).astype(int)
    grid = place & ~sub
    nm = L.n_makers.to_numpy(int); tks = L.tk_y_side.map(SIDE).fillna(-1).to_numpy().astype(int)
    st = L.settle_ms.to_numpy(np.int64); rv = L.first_recv_ns.to_numpy(np.int64) // 10 ** 6
    hi = np.where(st > 0, np.minimum(st, rv), rv).astype(np.int64)
    lo_h = np.where(st > 0, hi - C.MAX_LAG_MS, -1).astype(np.int64)        # evidence-free lower end (declared lag); receipt-hi: none
    fb_h = np.where(lo_h >= 0, np.searchsorted(t, lo_h, 'left') - 1, -1).astype(np.int64)
    X = dict(hi=hi, lo_h=lo_h, fb_h=fb_h, rcpt=st <= 0, lo=np.full(N, -1, np.int64), fb=np.full(N, -1, np.int64), fa=np.full(N, F, np.int64), meth=np.full(N, 'none', object),
             e_fid=np.full(N, -1, np.int64), d_fid=np.full(N, -1, np.int64), lo_x=np.full(N, -1, np.int64), exh_p=np.zeros(N, bool),
             wk=np.zeros(N, np.int64),
             rank=np.zeros(N, np.int64), flags=[set() for _ in range(N)])
    groups = {}
    R = records.set_index('sid')
    grp = L.groupby('sid', sort=False).indices
    order = sorted(grp, key=lambda x: (int(hi[grp[x][0]]), x))
    tests = Counter()
    rec = []
    for rank, sid in enumerate(order):
        ii = np.asarray(grp[sid])                                   # mk_idx order: data only (descriptive listing counts), never evidence
        h = int(hi[ii[0]]); fl = set(); pl = place[ii]
        if st[ii[0]] <= 0:
            fl.add('hi_from_receipt')
        rr = R.loc[sid]
        if len(ii) < nm[ii[0]]:
            fl.add('incomplete_record')
        elif abs(q[ii].sum() - float(rr.amount_filled)) > 1e-6 * max(1., float(rr.amount_filled)):
            fl.add('amount_mismatch')
        if not pl.all():
            fl.add('record_unplaceable_leg')
        mixed = len(set(s[ii][pl].tolist())) > 1
        if mixed:
            fl.add('maker_side_mixed')
        if (pl & (s[ii] == tks[ii])).any():
            fl.add('taker_side_mismatch')
        fifo_ok = pl.all() and not mixed and 'taker_side_mismatch' not in fl
        if fifo_ok:
            bad = np.where(s[ii] == 1, yp[ii], -yp[ii])             # larger = worse for the taker
            worst = np.abs(bad - bad.max()) < 1e-9
            if len(np.unique(np.round(yp[ii], 6))) >= 2:            # rule 1: strictly better than the record's worst price
                X['exh_p'][ii] = ~worst
                d = np.diff(bad)
                tests['multi_price_records'] += 1
                tests['listing_best_first' if (d >= -1e-12).all() else 'listing_worst_first' if (d <= 1e-12).all() else 'listing_neither'] += 1
            if worst.sum() >= 2:                                    # rule 2: k worst-price makers, at most one keeps a balance
                X['wk'][ii[worst]] = int(worst.sum()); groups[rank] = ii[worst]
        elif len(ii) >= 2:
            tests['multi_maker_records_without_price_time'] += 1
        E_p = E_d = E_x = -1; drop_unknown = False
        fa = int(np.searchsorted(t, h, 'left'))
        gi = ii[grid[ii]]
        if F >= 2 and len(gi) and st[ii[0]] > 0:                        # receipt-hi: the match may precede the window
            a0 = max(int(np.searchsorted(t, h - C.MAX_LAG_MS, 'left')), 1); b0 = min(fa, F - 1)
            if a0 <= b0:
                for sc, kc in sorted(set(zip(s[gi].tolist(), k[gi].tolist()))):
                    m = (s[gi] == sc) & (k[gi] == kc); qq = q[gi][m]; vv = qq * yp[gi][m]
                    vals = sorted(set([round(float(vv.sum()), 9)] + [round(float(x), 9) for x in vv]))
                    inc = pend[a0:b0 + 1, :, kc] - pend[a0 - 1:b0, :, kc]
                    hit = np.zeros(b0 - a0 + 1, bool)                       # exact value (pf_step1_timing): x_lo_exact only
                    for v in vals:
                        hit |= ((inc > 1e-9) & (np.abs(inc - v) <= max(PEND_REL * v, PEND_ABS))).any(1)
                    j = np.flatnonzero(hit)
                    if len(j) and (E_x < 0 or a0 + j[0] < E_x):
                        E_x = a0 + int(j[0])
                    v0 = vals[0]                                             # could include the record (merged with others' bookings)
                    j = np.flatnonzero(((inc > 1e-9) & (inc >= v0 - max(PEND_REL * v0, PEND_ABS))).any(1))
                    if len(j) and (E_p < 0 or a0 + j[0] < E_p):
                        E_p = a0 + int(j[0])
                    qc = float(qq.sum())
                    dd = shown[a0 - 1:b0, sc, kc] - shown[a0:b0 + 1, sc, kc]          # NaN = an unknown cell: a possible candidate
                    j = np.flatnonzero((dd >= max(TOL_SH, qc - max(DROP_REL * qc, DROP_ABS))) | np.isnan(dd))
                    if len(j) and (E_d < 0 or a0 + j[0] < E_d):
                        E_d = a0 + int(j[0]); drop_unknown = bool(np.isnan(dd[j[0]]))
        cand = [x for x in (E_p, E_d) if x >= 1]                          # the first frame that could reflect the record (either evidence)
        E = min(cand) if cand else -1
        meth = 'none' if E < 1 else ('pend_inc' if E == E_p else 'book_drop')
        if meth == 'book_drop' and drop_unknown and E == E_d:
            fl.add('drop_at_unknown_cell')
        Ex = E_x if E_x >= 1 else E_d                                       # the reviewed exact-match-first rule, for comparison
        X['lo_x'][ii] = int(t[Ex - 1]) if Ex >= 1 else -1
        fb = E - 1 if E >= 1 else -1
        need(fb < 0 or int(t[fb]) < h, f'fill window inverted for record {sid}')
        if fb < 0:
            fl.add('fill_lo_unbounded')
        elif gap[fb + 1:min(fa, F - 1) + 1].any():
            fl.add('window_gap')
        X['lo'][ii] = int(t[fb]) if fb >= 0 else -1; X['fb'][ii] = fb; X['fa'][ii] = fa; X['meth'][ii] = meth; X['rank'][ii] = rank
        X['e_fid'][ii] = int(B['fid'][E]) if E >= 1 else -1; X['d_fid'][ii] = int(B['fid'][E_d]) if E_d >= 1 else -1
        for i in ii:
            X['flags'][i] |= fl
        side = int(s[ii[0]]) if fifo_ok and st[ii[0]] > 0 else -1           # receipt-hi records have no reliable place in the sequence
        rec.append(dict(rank=rank, hi=h, lo=int(t[fb]) if fb >= 0 else -1, lo_h=int(lo_h[ii[0]]), side=side,
                        worst=(float(yp[ii].max()) if side == 1 else float(yp[ii].min())) if side >= 0 else NAN, tk=str(L.tk_signer.iat[ii[0]])))
    X['rec'] = pd.DataFrame(rec, columns=['rank', 'hi', 'lo', 'lo_h', 'side', 'worst', 'tk'])
    X['tests'] = tests
    X['groups'] = groups
    X['n_records_with_legs'] = len(order)
    X['s'], X['k'] = s, k
    return X


# ------------------------------------------------------------------ order hashes (H1) ------------------------------------------------------------------
def _orders(L, records, X, place, sub, xf):
    hi, lo = X['hi'], X['lo']
    idn = list(zip(L.mk_signer, L.y_side, L.y_ptick, L.y_price.round(6).fillna(-1.), L.mk_quote, L.mk_index_set))
    tk, ntk = {}, Counter(records.tk_hash)
    for h, g, ys in zip(records.tk_hash, records.tk_signer, records.tk_y_side):
        tk.setdefault(h, set()).add((g, ys))
    grp = L.groupby('mk_hash', sort=False).indices
    hs = sorted(grp, key=lambda h: (int(hi[grp[h]].min()), h))
    n = len(hs)
    O = dict(hash=hs, H=n, fills=[np.asarray(grp[h]) for h in hs], valid=np.zeros(n, bool), sig=np.empty(n, object), ys=np.empty(n, object),
             s=np.full(n, -1, int), k=np.full(n, -1, int), n=np.zeros(n, int), cum=np.zeros(n), first_hi=np.zeros(n, np.int64),
             last_hi=np.zeros(n, np.int64), rlo=np.full(n, -1, np.int64), p_lo=np.full(n, -1, np.int64), p_hi=np.full(n, -1, np.int64),
             as_taker=np.zeros(n, int), n_unb=np.zeros(n, int), flags=[set() for _ in range(n)], ref=np.zeros(n, int))
    oid = np.zeros(len(L), int)
    for o, h in enumerate(hs):
        ii = O['fills'][o]; oid[ii] = o; fl = O['flags'][o]
        i0 = ii[np.lexsort((X['rank'][ii], hi[ii]))[0]]                      # first fill: min hi, then record order (never mk_idx)
        g, ys = idn[i0][0], idn[i0][1]
        if len({idn[i] for i in ii}) > 1 or any(x != (g, ys) for x in tk.get(h, ())):
            fl.add('identity_conflict')
        if L.sid.iloc[ii].duplicated().any():
            fl.add('dup_hash_in_record')
        if not place[ii].all():
            fl.add('unplaceable_leg')
        if sub[ii].any():
            fl.add('subcent_price')
        if any('malformed' in xf[i] for i in ii):
            fl.add('malformed_identity')
        for i in ii:
            fl |= {f for f in X['flags'][i] if f in REC_FLAGS} | {f'leg_{f}' for f in xf[i]}
        O['valid'][o] = not (fl & INVALID)
        O['sig'][o] = g; O['ys'][o] = str(ys); O['ref'][o] = i0; O['as_taker'][o] = ntk.get(h, 0)
        O['s'][o] = SIDE.get(ys, -1); O['k'][o] = int(idn[i0][2])
        O['n'][o] = len(ii); O['cum'][o] = float(L.shares.to_numpy()[ii].sum())
        O['first_hi'][o] = int(hi[ii].min()); O['last_hi'][o] = int(hi[ii].max())
        lb = X['lo_h'][ii][X['lo_h'][ii] >= 0]                       # removal_lo / persistence use the evidence-free lower ends (B2)
        O['rlo'][o] = int(lb.max()) if len(lb) else -1
        O['n_unb'][o] = int((lo[ii] < 0).sum())
        if not (fl & NO_PRESENCE) and len(ii) >= 2 and len(lb) and int(lb.max()) >= O['first_hi'][o]:     # H2 needs only the fills
            O['p_lo'][o], O['p_hi'][o] = O['first_hi'][o], int(lb.max())
    O['oid'] = oid
    return O


# ------------------------------------------------------------------ book-derived per-cell arrays ------------------------------------------------------------------
def _cells(L, X, O, B, place, sub):
    """book cells (side, tick) of the placeable on-grid legs: shown [F, nc], unexplained drops U (prefix sums), cell per leg / order."""
    F = B['F']; grid = place & ~sub
    keys = sorted(set(zip(X['s'][grid].tolist(), X['k'][grid].tolist())))
    cidx = {kk: i for i, kk in enumerate(keys)}
    c = np.array([cidx[(a, b)] if g else -1 for a, b, g in zip(X['s'], X['k'], grid)], int)
    nc = len(keys)
    SH = B['shown'][:, [a for a, _ in keys], [b for _, b in keys]] if nc else np.zeros((F, 0))
    q = L.shares.to_numpy(float); fb, fa = X['fb_h'], X['fa']
    # prints possibly inside the frame interval (f-1, f]: f in [max(fb_h + 1, 1), min(fa, F - 1)] (every on-grid fill, valid order or not)
    d = np.zeros((F + 1, nc))
    st_, en_ = np.maximum(fb + 1, 1), np.minimum(fa, F - 1); m = (st_ <= en_) & (c >= 0)
    np.add.at(d, (st_[m], c[m]), q[m]); np.add.at(d, (en_[m] + 1, c[m]), -q[m])
    prints = np.cumsum(d[:F], 0)
    drop = np.zeros((F, nc))
    if F > 1:
        drop[1:] = np.maximum(SH[:-1] - SH[1:], 0.)
    U = np.vstack([np.zeros((1, nc), np.int64), np.cumsum(drop - prints > TOL_SH, 0)])
    oc = np.array([cidx[(O['s'][o], O['k'][o])] if O['valid'][o] else -1 for o in range(O['H'])], int)
    return dict(c=c, nc=nc, SH=SH, U=U, oc=oc, q=q)


def _need(F, nc, start, end, c, q, m):
    """sum of q over frames [start, end] at cell c for the masked fills -> (F, nc)."""
    d = np.zeros((F + 1, nc))
    m = m & (start <= end) & (end >= 0) & (c >= 0)
    np.add.at(d, (start[m], c[m]), q[m]); np.add.at(d, (end[m] + 1, c[m]), -q[m])
    return np.cumsum(d[:F], 0)


def _count(F, nc, a, b, oc, m):
    """orders proven present per (frame, cell): +1 over each masked order's presence frames [a, b] (where its need is positive)."""
    d = np.zeros((F + 1, nc), np.int64)
    m = m & (a <= b) & (b >= 0) & (oc >= 0)
    np.add.at(d, (a[m], oc[m]), 1); np.add.at(d, (b[m] + 1, oc[m]), -1)
    return np.cumsum(d[:F], 0)


def _tol(n):
    """H3 display tolerance for n orders in the accounting: each may show up to TOL_SH short (+1e-9: exact 0.01 shortfalls in float)."""
    return TOL_SH * np.maximum(n, 1) + 1e-9


def _room(SH, nd, vm, nn):
    """room for an order not proven present: shown less the others' proven needs; NaN where the cell is unknown or the accounting is violated."""
    r = np.where(nd <= SH + _tol(nn), SH - nd, SH)
    r[vm] = np.nan
    return r


def _viol(SH, nd, nn):
    v = SH < nd - _tol(nn)
    return v, np.vstack([np.zeros((1, SH.shape[1]), np.int64), np.cumsum(v, 0)])


# ------------------------------------------------------------------ one scenario ------------------------------------------------------------------
def _nanmin(x):
    x = x[~np.isnan(x)]
    return float(x.min()) if len(x) else NAN


def _hard(X, O, B, K, we_ms):
    """Scenario-free consequences of H1-H3 (any fill time inside its EVIDENCE-FREE window (hi - MAX_LAG_MS, hi]; no priority): certain
    presence [min f_after, max fb_h], fills certainly after a frame (fb_h >= frame); capacity contradictions (then this order's capacity
    claims are void and the violated cells unknown for every other order), the size bound, the capacity-tight GONE bound, the capacity posting
    bound and the book removal bound. A class is constrained only when every scenario reports exactly these values."""
    F, t, gap = B['F'], B['t'], B['gap']; fb, fa, hi = X['fb_h'], X['fa'], X['hi']
    H, oid, c, q, SH = O['H'], O['oid'], K['c'], K['q'], K['SH']
    vf = O['valid'][oid]
    aw = np.full(H, F + 1, np.int64); np.minimum.at(aw, oid, fa)
    bw = np.full(H, -2, np.int64); np.maximum.at(bw, oid, fb)
    nd = _need(F, K['nc'], aw[oid], np.minimum(bw[oid], fb), c, q, vf & (fb >= 0))
    nn = _count(F, K['nc'], aw, bw, K['oc'], O['valid'])            # orders in the accounting: each may show TOL_SH short
    viol, P = _viol(SH, nd, nn)
    slack = np.where(viol, np.nan, SH - nd)
    room = _room(SH, nd, viol, nn)
    out = dict(viol=np.zeros(H, np.int64), size_ub=np.full(H, NAN), tight=np.zeros(H, bool), post=np.full(H, -1, np.int64),
               rhi=np.full(H, -1, np.int64), rm=[[] for _ in range(H)], miss=np.zeros(H, bool), viol_cells=int(viol.sum()), vmask=viol)
    for o in range(H):
        if not O['valid'][o]:
            continue
        fi = O['fills'][o]; co = K['oc'][o]; cum = O['cum'][o]; ao, bo = int(aw[o]), int(bw[o]); pres = ao <= bo
        nv = int(P[bo + 1, co] - P[ao, co]) if pres else 0
        out['viol'][o] = nv
        g0 = int(fa[fi].max())                       # the first frame certainly after every fill
        if not nv:                                   # a contradicted order gets no capacity claim at all (B1)
            minsl = _nanmin(slack[ao:bo + 1, co]) if pres else NAN
            out['tight'][o] = tight = bool(pres and minsl <= TOL_SH)    # no remainder after the last fill: GONE (exhausted or cancelled)
            out['size_ub'][o] = cum + max(minsl, 0.) if pres and not np.isnan(minsl) else NAN
            if ao > 0:                               # posting: frames before certain presence, against the fills certainly after them
                own = np.zeros(min(ao, F))
                for i in fi[fb[fi] >= 0]:
                    own[:min(int(fb[i]), len(own) - 1) + 1] += q[i]
                shv = np.where(viol[:len(own), co], np.nan, SH[:len(own), co])          # the single-order form too: never looser than it
                j = np.flatnonzero(((room[:len(own), co] < own - _tol(1 + nn[:len(own), co])) | (shv < own - _tol(1))) & (own > TOL_SH))
                if len(j):
                    out['post'][o] = int(t[j[-1]])
            if g0 < F:
                cands = [(int(t[g0]), 'capacity_tight')] if tight else []
                j = np.flatnonzero(room[g0:, co] <= TOL_SH)
                if len(j):
                    cands.append((int(t[g0 + j[0]]), 'book_gone'))
                if cands:
                    out['rhi'][o] = rh = min(x for x, _ in cands); out['rm'][o] = [m_ for x, m_ in cands if x == rh]
        rhi = out['rhi'][o]
        end_f = int(np.searchsorted(t, rhi, 'right')) - 1 if rhi >= 0 else F - 1
        span = slice(g0, end_f + 1)
        out['miss'][o] = (F == 0 or (g0 <= end_f and bool(gap[span].any() or np.isnan(room[span, co]).any()))
                          or (rhi < 0 and (g0 >= F or t[-1] < we_ms - GAP_MS)))
    return out


def _scenario(name, L, X, O, B, K, Hd, we_ms):
    sc = C.SCENARIOS[name]; prio = sc['priority'] == 'price_time'; loc = sc['fill_clock'] == 'lo'
    F, t, gap = B['F'], B['t'], B['gap']
    hi, lo, fb, fa, lo_h, fb_h = X['hi'], X['lo'], X['fb'], X['fa'], X['lo_h'], X['fb_h']
    pos_hi = fa - 1
    pos = np.where(fb >= 0, np.maximum(fb, fb_h), fb_h) if loc else pos_hi          # lo clock: the evidence lower end (B2 refinement)
    place = np.where(lo >= 0, np.maximum(lo, lo_h), lo_h) if loc else hi            # placement time of each fill (-1 = none)
    qb = np.where(fb >= 0, fb, pos)                                    # queue frame (contract: f_before; else the clock's frame)
    H, oid, c, q, SH, U = O['H'], O['oid'], K['c'], K['q'], K['SH'], K['U']
    N = len(L); vf = O['valid'][oid]
    a = np.full(H, F + 1, np.int64); np.minimum.at(a, oid, pos + 1)
    b = np.full(H, -2, np.int64); np.maximum.at(b, oid, pos)
    nd = _need(F, K['nc'], a[oid], np.minimum(b[oid], pos), c, q, vf)
    nn = _count(F, K['nc'], a, b, K['oc'], O['valid'])                # orders in the accounting: each may show TOL_SH short
    viol, P = _viol(SH, nd, nn)
    vm = viol | Hd['vmask']                                            # violated cells are unknown for every other order (B1)
    slack = np.where(vm, np.nan, SH - nd)
    room = _room(SH, nd, vm, nn)
    rank = X['rank']; seq = hi                     # event sequence: settle order; the fills of one settle ms are unordered, none 'later'
    T = Counter()
    exh = np.zeros(N, bool); exh_p = np.zeros(N, bool); alo = np.zeros(N); later = np.zeros(N, bool); tie = np.zeros(N, bool); tviol = set()
    unord = np.zeros(H, bool); np.logical_or.at(unord, oid, X['rcpt'])      # an order with a receipt-hi fill has no reliable sequence
    tconf = np.zeros(H, bool)
    RS = {}
    if prio:
        later_c = np.zeros(N, bool)
        for o in range(H):
            fi = O['fills'][o]
            if O['valid'][o] and len(fi) > 1:
                sqf = seq[fi]
                later[fi] = sqf < sqf.max()                                         # the order fills again STRICTLY later (a later settle ms)
                later_c[fi] = ((lo_h[fi][None, :] >= hi[fi][:, None]) & (lo_h[fi][None, :] >= 0)).any(1)   # certainly later (B2 ends)
                tie[fi] = (sqf[:, None] == sqf[None, :]).sum(1) >= 2              # another fill in the same settle ms (counted, never voids)
        exh_p = X['exh_p'] & vf                                                     # rule 1 (price priority)
        u1 = exh_p & ~unord[oid]
        T.update(T1_makers=int(u1.sum()), T1_violations=int((u1 & later).sum()), T1_violations_certain=int((u1 & later_c).sum()),
                 T1_ties=int((u1 & tie).sum()), T1_receipt_hi=int((exh_p & unord[oid]).sum()))
        exh |= exh_p & ~unord[oid]
        for mem in X['groups'].values():                                            # rule 2 (time priority at the worst price)
            if not vf[mem].all():
                T['T2_groups_invalid'] += 1; continue
            if unord[oid[mem]].any():
                T['T2_groups_receipt_hi'] += 1; continue
            A = mem[later[mem]]            # strictly later: a member whose only other fill shares this settle ms is not (FIFO puts that fill first)
            T['T2_groups'] += 1; T['T2_groups_tie'] += int(tie[mem].any())
            T['T2_violations'] += int(len(A) >= 2); T['T2_violations_certain'] += int(later_c[mem].sum() >= 2)
            if len(A) >= 2:
                tviol |= set(oid[A].tolist())
            elif len(A) == 1:
                oth = mem[mem != A[0]]; exh[oth] = True; alo[A[0]] = float(q[oth].sum()); T['T2_survivor_groups'] += 1
            else:
                T['T2_no_survivor_groups'] += 1; T['T2_min_exhausted'] += len(mem) - 1
        smax = np.full(H, -1, np.int64); np.maximum.at(smax, oid, seq)
        tconf = np.bincount(oid[exh & (seq == smax[oid])], minlength=H) >= 2     # exhausted in two records of its last settle ms: FIFO-inconsistent
        T['tie_conflicts'] = int(tconf.sum())
        r = X['rec'][X['rec'].side >= 0].copy()
        r['ts'] = np.where(r.lo >= 0, np.maximum(r.lo, r.lo_h), r.lo_h) if loc else r.hi   # placement (the clock) of each record
        r = r.sort_values(['hi', 'rank'], kind='stable')                                 # sequence (settle order)
        for sd in (0, 1):
            x = r[r.side == sd]
            RS[sd] = dict(ts=x.ts.to_numpy(np.int64), hi=x.hi.to_numpy(np.int64), w=x.worst.to_numpy(float), tk=x.tk.to_numpy(object))
    out = dict(size_exact=np.full(H, NAN), size_ub=np.full(H, NAN), post_lo=np.full(H, -1, np.int64), post_m=np.full(H, 'none', object),
               post_cap=np.full(H, -1, np.int64), post_fifo=np.full(H, -1, np.int64), alo=np.full(H, NAN), ahi=np.full(H, NAN),
               ahi_raw=np.full(H, NAN), rm_set=np.full(H, '|'.join(sorted(C.REMOVAL)), object), rm_hi=np.full(H, -1, np.int64),
               rm_m=np.full(H, 'none', object), clock_viol=np.zeros(H, np.int64), flags=[set() for _ in range(H)],
               f_ahi=np.full(N, NAN), f_alo=np.where(prio, alo, NAN),
               f_exh=exh & ~np.isin(oid, oid[exh_p & later]) & ~tconf[oid])
    for o in range(H):
        if not O['valid'][o]:
            continue
        fi = O['fills'][o]; co = K['oc'][o]; sd = O['s'][o]; p = O['k'][o] / 100.; g = O['sig'][o]; cum = O['cum'][o]; fl = out['flags'][o]
        sqf, posf = seq[fi], pos[fi]
        srt = fi[np.lexsort((rank[fi], sqf))]
        fF, fL = srt[0], srt[-1]
        ao, bo = int(a[o]), int(b[o]); pres = ao <= bo
        nv = int(P[bo + 1, co] - P[ao, co]) if pres else 0
        out['clock_viol'][o] = nv
        if nv and not Hd['viol'][o]:
            fl.add('clock_capacity_conflict')
        cap_ok = nv == 0 and not Hd['viol'][o]                           # a contradiction voids this scenario's capacity claims (B1)
        if not cap_ok:
            fl.add('capacity_bounds_void')
        minsl = _nanmin(slack[ao:bo + 1, co]) if (pres and cap_ok) else NAN
        tight_s = bool(pres and cap_ok and minsl <= TOL_SH)            # the book proves GONE only (integrator decision, two-history case)
        exhL = False; ekind = ''
        if prio:
            t1 = bool((exh_p[fi] & later[fi]).any())
            if t1:
                fl.add('fifo_reappearance')
            if o in tviol:
                fl.add('fifo_time_violation')
            if tconf[o]:
                fl.add('fifo_tie_conflict')
            xl = fi[exh[fi] & (sqf == sqf.max())]          # the exhausting fill among those of its last settle ms (FIFO orders the others first)
            exhL = len(xl) == 1 and not t1 and o not in tviol
            ekind = 'fifo_price' if (len(xl) and exh_p[xl[0]]) else 'fifo_time'
        ub = [x for x in ((cum + max(minsl, 0.)) if not np.isnan(minsl) else NAN, Hd['size_ub'][o]) if not np.isnan(x)]
        out['size_ub'][o] = cum if exhL else (min(ub) if ub else NAN)
        out['size_exact'][o] = cum if exhL else NAN                      # size_exact only from the S_FIFO price / survivor rules
        # posting: capacity (frames before presence, where all the order's fills are after the frame) and fifo_through
        j = (np.flatnonzero((room[:ao, co] < cum - _tol(1 + nn[:ao, co]))                                 # joint: this order + the others
                            | (np.where(vm[:ao, co], np.nan, SH[:ao, co]) < cum - _tol(1))) if (ao > 0 and cap_ok) else ())   # single-order form
        pcap = int(t[j[-1]]) if len(j) else -1
        pff = -1
        if prio:
            R_ = RS[sd]; i1 = int(np.searchsorted(R_['hi'], seq[fF], 'left'))
            thr = (R_['w'][:i1] > p + 1e-9) if sd == 1 else (R_['w'][:i1] < p - 1e-9)
            jj = np.flatnonzero(thr & (R_['tk'][:i1] != g))
            if len(jj):
                pff = int(R_['ts'][jj[-1]])
        pcap = max(pcap, int(Hd['post'][o]))
        out['post_cap'][o], out['post_fifo'][o] = pcap, pff
        out['post_lo'][o] = max(pcap, pff)
        out['post_m'][o] = 'capacity' if pcap >= 0 and pcap >= pff else ('fifo_through' if pff >= 0 else 'none')
        if place[fF] >= 0 and out['post_lo'][o] > place[fF]:
            fl.add('post_after_first_fill')                              # the posting bound contradicts this scenario's own fill placement
        # queue (price_time): reference = the first fill (min hi, then record order); every fill as x_ columns
        if prio:
            for i in fi:
                qi = int(qb[i])
                if qi < 0:
                    continue
                if ((sqf < seq[i]) & (posf + 1 <= qi)).any():               # at the front of its price since an earlier fill
                    out['f_ahi'][i] = 0.
                    if alo[i] > TOL_SH:
                        fl.add('fifo_front_conflict'); T['front_survivor_conflicts'] += 1
                    continue
                raw = float(SH[qi, co]) - float(q[fi][posf >= qi].sum())
                out['f_ahi'][i] = NAN if (np.isnan(raw) or raw < -TOL_SH) else max(raw, 0.)
            r0 = O['ref'][o]; qi = int(qb[r0])
            out['alo'][o] = float(alo[r0])
            if qi >= 0:
                raw = float(SH[qi, co]) - float(q[fi][posf >= qi].sum()); out['ahi_raw'][o] = raw
                if np.isnan(raw):
                    fl.add('ahead_hi_unknown_cell')
                elif raw < -TOL_SH:
                    fl.add('ahead_hi_negative')
                elif raw < 0:
                    fl.add('ahead_hi_clipped'); out['ahi'][o] = 0.
                else:
                    out['ahi'][o] = raw
            if not np.isnan(out['ahi'][o]) and out['alo'][o] > out['ahi'][o] + TOL_SH:
                fl.add('queue_contradiction')
        else:
            out['alo'][o] = 0.                     # no priority: only the trivial [0, NaN], the value a FIFO scenario reports without evidence
        # removal
        if exhL:
            rset = {'exhausted'}; rhi = int(hi[fL]); rm = [ekind]
        else:
            cands = [(int(t[bo + 1]), 'capacity_tight')] if (tight_s and bo + 1 < F) else []
            if bo + 1 < F and cap_ok:
                j = np.flatnonzero(room[bo + 1:, co] <= TOL_SH)
                if len(j):
                    cands.append((int(t[bo + 1 + j[0]]), 'book_gone'))
            if Hd['rhi'][o] >= 0:
                cands += [(int(Hd['rhi'][o]), m_) for m_ in Hd['rm'][o]]
            if prio:
                R_ = RS[sd]; i0 = int(np.searchsorted(R_['hi'], seq[fL], 'right'))
                reach = (R_['w'][i0:] >= p - 1e-9) if sd == 1 else (R_['w'][i0:] <= p + 1e-9)
                jj = np.flatnonzero(reach & (R_['tk'][i0:] != g))
                if len(jj):
                    cands.append((int(R_['hi'][i0 + jj[0]]), 'fifo_front'))
            rhi = min(x for x, _ in cands) if cands else -1
            rm = [m_ for x, m_ in cands if x == rhi]
            end_f = int(np.searchsorted(t, rhi, 'right')) - 1 if rhi >= 0 else F - 1
            if bo + 1 < F and SH[bo + 1, co] <= TOL_SH:
                rm.append('book_empty_after')
            if bo + 1 <= end_f and U[end_f + 1, co] - U[bo + 1, co] > 0:
                rm.append('unprinted_drop')
            g0 = int(fa[fi].max()); span = slice(g0, end_f + 1)          # missing-evidence span: frames certainly after every fill
            beyond = rhi >= 0 and (F == 0 or rhi > t[-1])                # only a priority (fifo_front) bound can lie past the book
            miss = (F == 0 or beyond or (g0 <= end_f and bool(gap[span].any() or np.isnan(room[span, co]).any()))
                    or (rhi < 0 and (g0 >= F or t[-1] < we_ms - GAP_MS)))
            rset = ({'exhausted', 'cancelled_or_expired'} | ({'end_censored'} if (rhi < 0 or beyond) else set())
                    | ({'missing_evidence'} if miss else set()))
            if rhi < 0:
                rm.append('end_open')
        need(rhi < 0 or O['rlo'][o] <= rhi, f'removal window inverted for {O["hash"][o]}')
        out['rm_set'][o] = '|'.join(sorted(rset)); out['rm_hi'][o] = rhi; out['rm_m'][o] = '+'.join(sorted(set(rm))) or 'none'
    out['tests'] = T
    out['viol_cells'] = int(viol.sum())
    return out


# ------------------------------------------------------------------ classes (compare the scenarios in code) ------------------------------------------------------------------
def _same(vals):
    def eq(x, y):
        if isinstance(x, tuple):
            return all(eq(u, w) for u, w in zip(x, y))
        if isinstance(x, float) and isinstance(y, float):
            return (np.isnan(x) and np.isnan(y)) or abs(x - y) <= 1e-9
        return x == y
    return all(eq(vals[0], v) for v in vals[1:])


def _classes(O, S, Hd):
    """constrained = every scenario reports exactly the scenario-free H1-H3 result (Hd) and it is non-trivial; a scenario that differs ->
    assumption_sensitive; all trivial -> unresolved (size: observed, its lower bound being read directly)."""
    H = O['H']; cl = {f: np.empty(H, object) for f in ('size', 'post', 'persist', 'queue', 'removal')}
    for o in range(H):
        span = C.OBSERVED if (O['n'][o] >= 2 and O['p_lo'][o] >= 0) else C.UNRESOLVED     # B3: a non-empty certain span (fills only)
        if not O['valid'][o]:
            cl['size'][o], cl['persist'][o] = C.OBSERVED, span          # no book / priority use; the span is -1 / -1 under NO_PRESENCE
            for f in ('post', 'queue', 'removal'):
                cl[f][o] = C.UNRESOLVED
            continue
        hz = (NAN, float(Hd['size_ub'][o]))
        sz = [hz] + [(float(S[s]['size_exact'][o]), float(S[s]['size_ub'][o])) for s in SCEN]
        cl['size'][o] = (C.OBSERVED if np.isnan(hz[0]) and np.isnan(hz[1]) else C.CONSTRAINED) if _same(sz) else C.SENSITIVE
        pl = [int(Hd['post'][o])] + [int(S[s]['post_lo'][o]) for s in SCEN]
        cl['post'][o] = (C.UNRESOLVED if pl[0] < 0 else C.CONSTRAINED) if _same(pl) else C.SENSITIVE
        cl['persist'][o] = span
        qu = [(float(S[s]['alo'][o]), float(S[s]['ahi'][o])) for s in SCEN]          # no scenario-free queue; S_NOPRIO = the trivial (0, NaN)
        cl['queue'][o] = C.UNRESOLVED if _same(qu) else C.SENSITIVE                    # any FIFO value beyond the trivial one differs from it
        rw = Hd['rhi'][o]
        hset = ({'exhausted', 'cancelled_or_expired'} | ({'end_censored'} if rw < 0 else set()) | ({'missing_evidence'} if Hd['miss'][o] else set()))
        rv = [('|'.join(sorted(hset)), int(O['rlo'][o]), int(rw))] + [(S[s]['rm_set'][o], int(O['rlo'][o]), int(S[s]['rm_hi'][o])) for s in SCEN]
        if _same(rv):
            cl['removal'][o] = C.UNRESOLVED if (rw < 0 and TRIVIAL_RM <= hset) else C.CONSTRAINED
        else:
            cl['removal'][o] = C.SENSITIVE
        if Hd['viol'][o] or any(S[s]['clock_viol'][o] for s in SCEN):      # B1: a capacity contradiction anywhere -> nothing constrained
            for f in ('size', 'post', 'removal'):
                if cl[f][o] == C.CONSTRAINED:
                    cl[f][o] = C.SENSITIVE
    return cl


# ------------------------------------------------------------------ links ------------------------------------------------------------------
def _links(Ot, mid):
    cols = ['mid', 'mk_signer', 'y_side', 'kind', 'prev_hash', 'next_hash', 'prev_ptick', 'next_ptick', 'prev_last_hi_ms', 'next_first_hi_ms',
            'gap_ms', 'x_latest', 'label', 'status']
    rows = []
    for (g, sd), G in Ot.groupby(['mk_signer', 'y_side'], sort=True):
        G = G.sort_values(['last_hi_ms', 'mk_hash'], kind='stable')
        last = G.last_hi_ms.to_numpy(np.int64); tick = G.y_ptick.to_numpy(int); hs = G.mk_hash.to_numpy(object); val = G.valid.to_numpy(bool)
        for h, tk_, f1, v in zip(G.mk_hash, G.y_ptick, G.first_hi_ms, G.valid):
            j = int(np.searchsorted(last, f1, 'left'))
            for kind, same in (('refill', True), ('reprice', False)):
                x = next((x for x in range(j - 1, -1, -1) if (tick[x] == tk_) == same), None)
                if x is not None:
                    rows.append(dict(mid=mid, mk_signer=g, y_side=sd, kind=kind, prev_hash=hs[x], next_hash=h, prev_ptick=int(tick[x]),
                                     next_ptick=int(tk_), prev_last_hi_ms=int(last[x]), next_first_hi_ms=int(f1), gap_ms=int(f1 - last[x]),
                                     x_latest=bool(x == j - 1), label='descriptive',
                                     status=C.STATUS_OK if (v and val[x]) else 'invalid_order'))
    D = pd.DataFrame(rows, columns=cols)
    return D.astype(dict(mid='int64', mk_signer='str', y_side='str', kind='str', prev_hash='str', next_hash='str', prev_ptick='int64',
                         next_ptick='int64', prev_last_hi_ms='int64', next_first_hi_ms='int64', gap_ms='int64', x_latest='bool',
                         label='str', status='str'))


# ------------------------------------------------------------------ the market ------------------------------------------------------------------
def run_market(legs, records, frames, levels, market):
    """-> (fills_df, orders_df, links_df, summary) for one market (contract section 5). Pure: no file is read or written."""
    mid = int(market['mid']); ws_ms, we_ms = int(market['ws']) * 1000, int(market['we']) * 1000
    need(we_ms > ws_ms, f'market {mid}: we <= ws')
    L, place, sub, xf = _prep(legs, records, mid)
    B = _book(frames, levels, mid, ws_ms, we_ms)
    X = _timing(L, records, B, place, sub)
    O = _orders(L, records, X, place, sub, xf)
    K = _cells(L, X, O, B, place, sub)
    F, H, N = B['F'], O['H'], len(L)
    oid = O['oid']
    Hd = _hard(X, O, B, K, we_ms); hard = Hd['viol']                     # scenario-free H1-H3 (window logic)
    for o in np.flatnonzero(hard > 0):
        O['flags'][o].add('capacity_contradiction')
    S = {s: _scenario(s, L, X, O, B, K, Hd, we_ms) for s in SCEN}
    PRIO = [s for s in SCEN if C.SCENARIOS[s]['priority'] == 'price_time']
    cl = _classes(O, S, Hd)
    # ---- fills table ----
    keep = ('identity_conflict', 'dup_hash_in_record', 'capacity_contradiction', 'unplaceable_leg', 'subcent_price', 'malformed_identity')
    fst = []
    for i in range(N):
        f = set(X['flags'][i]) | {x for x in O['flags'][oid[i]] if x in keep} | {f'leg_{x}' for x in xf[i]}
        if not place[i]:
            f.add('unplaceable')
        if sub[i]:
            f.add('subcent_price')
        fst.append(_flags(f))
    fid = B['fid']
    fills = pd.DataFrame(dict(mid=L.mid.to_numpy(), sid=L.sid.to_numpy(), mk_hash=L.mk_hash.to_numpy(), mk_idx=L.mk_idx.to_numpy(),
                              y_side=L.y_side.to_numpy(), y_ptick=L.y_ptick.to_numpy(), shares=L.shares.to_numpy(float), hi_ms=X['hi'], lo_ms=X['lo'],
                              t_method=X['meth'], f_before=np.where(X['fb'] >= 0, fid[np.maximum(X['fb'], 0)] if F else -1, -1),
                              f_after=np.where(X['fa'] < F, fid[np.minimum(X['fa'], F - 1)] if F else -1, -1), status=fst,
                              x_evidence_fid=X['e_fid'], x_drop_fid=X['d_fid'], x_lo_exact_ms=X['lo_x'], x_lo_hard_ms=X['lo_h'], x_rec_rank=X['rank'], x_exh_price=X['exh_p'], x_worst_k=X['wk'],
                              **{f'x_exh_{s}': S[s]['f_exh'] for s in PRIO}, **{f'x_ahead_lo_{s}': S[s]['f_alo'] for s in PRIO},
                              **{f'x_ahead_hi_{s}': S[s]['f_ahi'] for s in PRIO}))
    fills = _conform(fills, 'fills')
    # ---- orders table: one row per hash per scenario ----
    XO = ['x_size_ub', 'x_post_cap_ms', 'x_post_fifo_ms', 'x_ahead_hi_raw', 'x_n_lo_unbounded', 'x_as_taker', 'x_capacity_violation_frames',
          'x_clock_violation_frames']
    rows = []
    for s in SCEN:
        R = S[s]
        for o in range(H):
            rows.append(dict(scenario=s, mid=mid, mk_hash=O['hash'][o], mk_signer=O['sig'][o], y_side=O['ys'][o], y_ptick=int(O['k'][o]),
                             n_fills=int(O['n'][o]), cum_shares=float(O['cum'][o]), first_hi_ms=int(O['first_hi'][o]), last_hi_ms=int(O['last_hi'][o]),
                             size_lb=float(O['cum'][o]), size_exact=float(R['size_exact'][o]), size_class=cl['size'][o],
                             post_lo_ms=int(R['post_lo'][o]), post_hi_ms=int(O['first_hi'][o]), post_class=cl['post'][o], post_method=R['post_m'][o],
                             persist_lo_ms=int(O['p_lo'][o]), persist_hi_ms=int(O['p_hi'][o]), persist_class=cl['persist'][o],
                             ahead_lo=float(R['alo'][o]), ahead_hi=float(R['ahi'][o]), queue_class=cl['queue'][o],
                             removal_set=R['rm_set'][o], removal_lo_ms=int(O['rlo'][o]), removal_hi_ms=int(R['rm_hi'][o]), removal_class=cl['removal'][o],
                             removal_method=R['rm_m'][o], status=_flags(O['flags'][o] | R['flags'][o]),
                             x_size_ub=float(R['size_ub'][o]), x_post_cap_ms=int(R['post_cap'][o]), x_post_fifo_ms=int(R['post_fifo'][o]),
                             x_ahead_hi_raw=float(R['ahi_raw'][o]), x_n_lo_unbounded=int(O['n_unb'][o]), x_as_taker=int(O['as_taker'][o]),
                             x_capacity_violation_frames=int(hard[o]), x_clock_violation_frames=int(R['clock_viol'][o])))
    orders = _conform(pd.DataFrame(rows, columns=list(C.SCHEMAS['orders']) + XO), 'orders')
    Ot = orders[orders.scenario == SCEN[0]][['mk_hash', 'mk_signer', 'y_side', 'y_ptick', 'first_hi_ms', 'last_hi_ms']].copy()
    Ot['valid'] = O['valid']
    links = _links(Ot, mid)
    # ---- summary: counts and rates only (computed predicates, no verdicts) ----
    need(len(orders) == len(SCEN) * H and len(fills) == N, 'row counts do not reconcile')
    sh = L.shares.to_numpy(float)
    rate = lambda a, b: round(a / b, 6) if b else None  # noqa: E731
    cov = {}
    for f in ('size', 'post', 'persist', 'queue', 'removal'):
        cov[f] = {}
        for c_ in C.CLASSES:
            m = cl[f] == c_; mf = m[oid]
            cov[f][c_] = dict(orders=int(m.sum()), executions=int(mf.sum()), shares=round(float(np.nansum(sh[mf])), 6))
        need(sum(v['orders'] for v in cov[f].values()) == H and sum(v['executions'] for v in cov[f].values()) == N, f'coverage of {f} does not reconcile')
    flag_counts = lambda col: dict(sorted(Counter(x for v in col for x in v.split('+')).items()))  # noqa: E731
    pt = {}
    for s in PRIO:
        T = S[s]['tests']
        pt[s] = dict(
            T1_price=dict(makers=T['T1_makers'], violations=T['T1_violations'], rate=rate(T['T1_violations'], T['T1_makers']),
                          violations_certain=T['T1_violations_certain'], rate_certain=rate(T['T1_violations_certain'], T['T1_makers']), ties=T['T1_ties'],
                          receipt_hi_excluded=T['T1_receipt_hi'],
                          unit='maker legs strictly better than their record worst price (valid orders, none with a receipt-hi fill; tied legs '
                               'included); violation = the order fills again in a later settle ms'),
            T2_time=dict(groups=T['T2_groups'], violations=T['T2_violations'], rate=rate(T['T2_violations'], T['T2_groups']),
                         violations_certain=T['T2_violations_certain'], rate_certain=rate(T['T2_violations_certain'], T['T2_groups']),
                         survivor_groups=T['T2_survivor_groups'], no_survivor_groups=T['T2_no_survivor_groups'],
                         min_exhausted_in_no_survivor_groups=T['T2_min_exhausted'], groups_tie=T['T2_groups_tie'], groups_invalid=T['T2_groups_invalid'],
                         groups_receipt_hi=T['T2_groups_receipt_hi'],
                         unit='records with >= 2 makers at their worst price (valid orders, no receipt-hi fill; groups with a tied member included, '
                              'groups_tie of them); violation = >= 2 of them fill again in a later settle ms'),
            front_survivor_conflicts=T['front_survivor_conflicts'], tie_conflict_orders=T['tie_conflicts'],
            order='later = a strictly later settle ms (hi); fills of one settle ms are unordered, never later, and never void a rule; '
                  '*_certain only from disjoint windows (lo_j >= hi_i)')
    tt = X['tests']
    pt['makers_listing'] = dict(multi_price_records=tt['multi_price_records'], best_first=tt['listing_best_first'], worst_first=tt['listing_worst_first'],
                                neither=tt['listing_neither'], multi_maker_records_without_price_time=tt['multi_maker_records_without_price_time'],
                                note='descriptive only: mk_idx is data, never evidence (contract amendment 1)')
    summary = dict(
        version=C.VERSION, module='pf_mt_lifecycle', mid=mid, ws_ms=ws_ms, we_ms=we_ms, outcome=str(market.get('outcome')),
        market_status=str(market.get('status', '')),
        declared=dict(MAX_LAG_MS=C.MAX_LAG_MS, GAP_MS=GAP_MS, TOL_SH=TOL_SH, PEND_REL=PEND_REL, PEND_ABS=PEND_ABS, DROP_REL=DROP_REL,
                      DROP_ABS=DROP_ABS, USABLE_FRAME=sorted(USABLE_FRAME), scenarios=C.SCENARIOS,
                      unbounded='a negative int64 time is unbounded on that side; NaN float = no bound'),
        inputs=dict(legs=N, unplaceable_legs=int((~place).sum()), subcent_legs=int(sub.sum()), records=int(len(records)),
                    records_with_legs=X['n_records_with_legs'], records_without_legs=int(len(records)) - X['n_records_with_legs'],
                    frames=B['n_rows'], frames_usable=F, frames_excluded=B['excluded'], frame_gaps=B['n_gaps'], unknown_cells=B['unknown_cells'],
                    conn=B['n_conn'],
                    conn_switches=B['n_conn_switch'], depth_span_ms=B['span_ms'],
                    fills_before_first_frame=int((X['fa'] == 0).sum()) if F else N, fills_after_last_frame=int((X['fa'] == F).sum())),
        fills=dict(n=N, shares=round(float(np.nansum(sh)), 6), t_method=dict(sorted(Counter(X['meth'].tolist()).items())),
                   status=dict(sorted(Counter(fills.status).items())), status_flags=flag_counts(fills.status)),
        orders=dict(hashes=H, valid=int(O['valid'].sum()), **{s: dict(status_flags=flag_counts(orders.status[orders.scenario == s]),
                                                                       removal_sets=dict(sorted(Counter(S[s]['rm_set'].tolist()).items())),
                                                                       post_method=dict(sorted(Counter(S[s]['post_m'][O['valid']].tolist()).items())))
                                                            for s in SCEN}),
        capacity=dict(hard_violation_cell_frames=Hd['viol_cells'], orders_contradicted=int((hard > 0).sum()),
                      clock_violation_cell_frames={s: S[s]['viol_cells'] for s in SCEN},
                      orders_clock_conflict={s: int(sum('clock_capacity_conflict' in f for f in S[s]['flags'])) for s in SCEN}),
        priority_tests=pt, coverage=cov,
        links=dict(sorted(Counter(links.kind.tolist()).items())))
    return fills, orders, links, summary


# ------------------------------------------------------------------ CLI ------------------------------------------------------------------
def _empty(name):
    return pd.DataFrame({c: pd.Series(dtype=d) for c, d in C.SCHEMAS[name].items()})


def _sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for ch in iter(lambda: f.read(1 << 20), b''):
            h.update(ch)
    return h.hexdigest()


def _write_once(path, write):
    need(not path.exists(), f'write-once: {path} exists (refusing to overwrite)')
    tmp = path.with_name(f'.{path.name}.tmp{os.getpid()}')
    write(tmp)
    try:
        os.link(tmp, path)
    except FileExistsError:
        raise ValueError(f'write-once: {path} exists (refusing to overwrite)')
    finally:
        os.unlink(tmp)


def market_cli(day, mid, out=C.OUT):
    """Read this market's rows of OUT/{legs,records,markets}_<day>.parquet (pf_mt_extract) and OUT/depth/<day>_<mid>_{frames,levels}.parquet
    (both absent = no depth); run; write OUT/life/<day>_<mid>_{fills,orders,x_links}.parquet + <day>_<mid>.json once (the JSON last)."""
    need(day in C.ALLOWED_DAYS or day == C.NOV8, f'day {day} is not a consumed day (never opened)')
    need(str(mid).isdigit(), f'bad market id {mid!r}')
    mid = int(mid); out = Path(out); life = out / 'life'
    dst = {k: life / f'{day}_{mid}_{k}.parquet' for k in ('fills', 'orders', 'x_links')}; js = life / f'{day}_{mid}.json'
    for p in list(dst.values()) + [js]:
        need(not p.exists(), f'write-once: {p} exists (refusing to overwrite)')
    src = dict(legs=out / f'legs_{day}.parquet', records=out / f'records_{day}.parquet', markets=out / f'markets_{day}.parquet',
               frames=out / 'depth' / f'{day}_{mid}_frames.parquet', levels=out / 'depth' / f'{day}_{mid}_levels.parquet')
    flt = [('mid', '==', mid)]
    mk = pd.read_parquet(src['markets'], columns=['mid', 'slug', 'ws', 'we', 'outcome', 'status'], filters=flt)
    need(len(mk) == 1, f'market {mid}: {len(mk)} rows in markets_{day}.parquet')
    legs = pd.read_parquet(src['legs'], filters=flt); guard('legs')
    recs = pd.read_parquet(src['records'], filters=flt); guard('records')
    have = [src['frames'].exists(), src['levels'].exists()]
    need(have[0] == have[1], f'market {mid}: only one of the depth files exists')
    frames = pd.read_parquet(src['frames']) if have[0] else _empty('frames')
    levels = pd.read_parquet(src['levels']) if have[0] else _empty('levels')
    guard('depth')
    fills, orders, links, summ = run_market(legs, recs, frames, levels, mk.iloc[0].to_dict())
    m = guard('compute')
    summ['provenance'] = dict(day=day, depth=bool(have[0]), module=str(Path(__file__).resolve()), module_sha256=_sha(Path(__file__).resolve()),
                              inputs={k: (dict(path=str(p), bytes=p.stat().st_size, mtime_ns=p.stat().st_mtime_ns) if p.exists() else None)
                                      for k, p in src.items()}, vmhwm_mib=round(m, 1), guard_mib=C.GUARD_MIB)
    life.mkdir(parents=True, exist_ok=True)
    for k, df in (('fills', fills), ('orders', orders), ('x_links', links)):
        _write_once(dst[k], lambda p, df=df: df.to_parquet(p, index=False))
    _write_once(js, lambda p: p.write_text(json.dumps(summ, indent=1, default=str)))
    return summ


if __name__ == '__main__':
    need(len(sys.argv) == 4 and sys.argv[1] == 'market', 'usage: pf_mt_lifecycle.py market <day> <mid>')
    r = market_cli(sys.argv[2], sys.argv[3])
    print(json.dumps(dict(mid=r['mid'], inputs=r['inputs'], fills=r['fills'], priority_tests=r['priority_tests'],
                          vmhwm_mib=r['provenance']['vmhwm_mib']), indent=1, default=str))
