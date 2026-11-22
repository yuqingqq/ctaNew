"""MAKER-TRACE-RECONSTRUCTION-v1 ground truth (contract.md section 7; session aea5bfb9 development, 2026-11-22). The reconstruction's
real-data positive control: on Nov 8 our own engine traded live (runs v6 = tokyo-6 and basis = basis-tokyo-1, the 220 scope.json
windows), so for OUR orders the submit, activation, size, price, cancel and final state are known from the bot's own records. The
lifecycle runs blind on the GT markets; this module joins our orders to the public trades by hash and scores the reconstruction.

INPUTS - the export's derived files only (live-samewindow-20261108; the capture directory named in scope.json is never opened, no raw
file is read): scope.json (windows, cutoff_ms); live_vs_replay/kernel_orders_{run} (pf_live_vs_replay.kernel: per kernel order Control
Place / Exec Signed / Accepted / CancelPendingAudit / UserCancel, each the journal frame time 'at' = BOT clock), core_orders_{run}
(pf_live_vs_replay.core_orders: Request Place t = the core's decision tick, venue feedback known_ms, Request Cancel t - BOT clock; filled
= sum of Filled qty), kernel_fills_{run} (Exec Fill notices: at = bot receipt, match_ms = venue STATUS clock whole second per status
Matched / Confirmed / Failed, size, price = the bought token's price), matches_20261108 (public REST maker legs: maker signer, maker_hash,
yes_side, ptick, shares, exec_s = executedAt, the Confirmed second), who.json (our_address).
YES SPACE  every own order BUYS a token: up -> quote Bid on indexSet 1 at tick/100; down -> Bid on indexSet 2 at 1 - tick/100 (core tick
   is the YES-space price on both sides; evidence: every Matched notice's price equals that token price, status fill_price_mismatch
   otherwise); side / price via pf_mt_common.yes_space.
LINK (no hash in the bot export): own public legs (maker == our_address) per hash vs our filled orders on (market, YES side, YES tick,
   |filled - sum leg shares| <= TOL_SH); TIME = every leg's Confirmed second is among the order's Confirmed notice seconds and the legs'
   shares equal its Matched notice sizes as multisets. unique (one candidate both ways, time-consistent) | unique_by_time (several content
   candidates, one time-consistent both ways) | unique_time_inconsistent | ambiguous | missing; an own public hash with no order is a row
   with status public_hash_unlinked. Unfilled orders never print a hash: mk_hash '' (status unfilled).
CLOCK BAND  d = venue status clock - bot clock, ONE BAND PER RUN (v2, integration review: v6 = tokyo-6 and basis = basis-tokyo-1 may be
   different bot hosts / clocks; v1's pooled band, truth/own_orders.parquet, stays as provenance), one constant within a run (premise),
   bounded from the run's own notices as K7 of pf_keep_pull_exec_windows: a notice is received after its status second starts
   (d >= match_ms - at, Matched + Confirmed) and a match follows its placement and precedes its Matched second's end (d <= match_ms + 1000
   - t_place, Matched). Bot times enter the venue domain only as *_venue_ms = bot + [d_lo, d_hi] of the order's run (columns d_lo_ms /
   d_hi_ms; exact for interval tests: the offset shifts every truth time of a run rigidly). Premise, not established: the venue book /
   UUID / collector clocks of the lifecycle equal the status clock (FINDINGS C23 / K7). Every score carries it (scores caveats).
COLUMNS  registry COLS (name, dtype, clock, source) is written into the parquet metadata (key mt_truth) with the band; clocks: bot_ms (bot
   host), venue_status_ms (venue status stamps), venue_band_ms (bot + band); score() refuses a comparison of a lifecycle venue time with a
   truth column whose registered clock is not a venue domain (clock-domain mismatch) and any bounded lifecycle time that is not epoch ms.
SCORE per scenario x class (contract 5 classes; a failed observed / constrained claim is a DEFECT, an assumption_sensitive one a sensitive
   miss): coverage of our filled orders (orders, executions, shares); identity; posting (true activation hull inside / overlap / disjoint
   [post_lo, post_hi]); size (size_lb <= true size always; == size_exact when stated; <= the lifecycle's upper bound x_size_ub when stated,
   classed by size_class); removal (exhausted-only sets vs truly filled; the true cause's membership and its distribution per set; the
   window); persistence (proven presence inside the possible life); queue (ahead_hi == 0 at the order's first fill - min hi, then the record
   rank x_rec_rank, as the lifecycle - frame lo_ms against own orders certainly resting there at that price, ours and every other maker's
   front claims); fill windows of our legs - OBSERVED [x_lo_hard_ms = hi - MAX_LAG_MS (-1 receipt-hi), hi_ms] (a miss is a defect), refined
   [lo_ms, hi_ms] as a sensitivity count (a miss is a sensitive miss) - and for multi-fill orders the sorted x_lo_hard_ms (defect) and lo_ms
   (sensitivity) against the sorted per-notice match bounds (the i-th earliest match precedes the i-th smallest bound); unique (order,
   field) defects across scenarios. Lifecycle conventions
   (pf_mt_lifecycle): a negative int64 time is unbounded on that side, NaN float = no bound, removal_set '|'-joined, statuses '+'-joined.
COMPLETENESS  score refuses unless every GT window of extract_<day>.json ws_population.selected, less those whose markets_<day> status
   carries boundary_censored (excluded from scoring, counted), has the lifecycle's fills, orders and JSON receipt.
DRY GATE  dry rebuilds the table and REFUSES unless the written v2 table exists and this code reproduces it exactly (content + registry).
Usage (run_light, PF_SLOT=claude): pf_mt_truth.py dry | build | score (v2: OUT/truth/v2)     selftest: pf_mt_truth_selftest.py"""
import collections
import datetime as dt
import hashlib
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pf_mt_common as C  # noqa: E402

EXP = Path('/home/yuqing/pmmm-data/review-exports/live-samewindow-20261108')
LV = EXP / 'live_vs_replay'
TRUTH_V1 = C.OUT / 'truth'                 # v1 (pooled band): provenance only, never written again
TRUTH_DIR = C.OUT / 'truth' / 'v2'         # v2 (one band per run): the table score() and the dry gate use
RUNS = ('v6', 'basis')
CUTOFF_MS = 1794178800000.0            # scope.json cutoff_ms (2026-11-08 23:00Z), checked on load
TOL_SH = 1e-5                          # shares: 1e18-scaled REST amounts vs the bot's micro-shares (declared before any result)
MS_RANGE = (1.7e12, 1.9e12)            # every bounded lifecycle time must be epoch ms (ns / s / 0 refused)
INF = float('inf')
BOT, VST, VBAND, BANDC, NONE = 'bot_ms', 'venue_status_ms', 'venue_band_ms', 'offset_ms', '-'
META_KEY = b'mt_truth'
HASH_RE, ADDR_RE = re.compile(r'0x[0-9a-f]{64}'), re.compile(r'0x[0-9a-f]{40}')
SCORED = 'ok'                          # truth rows scored: linked, fill-consistent, removal known
PRESENT_OK = ('ok', 'unfilled')        # truth rows whose certain-presence interval may serve as queue evidence

COLS = [  # name, dtype, clock, source / meaning
    ('run', 'str', NONE, 'export run (v6 = tokyo-6, basis = basis-tokyo-1); scope.json windows.run of the market'),
    ('mid', 'int64', NONE, 'kernel_orders.market (== core_orders.market)'),
    ('cid', 'int64', NONE, 'kernel_orders.cid (bot client order id; -1 on a public_hash_unlinked row)'),
    ('core_id', 'int64', NONE, 'core_orders.id'),
    ('lane', 'int64', NONE, 'core_orders.lane (0 up touch, 1 down touch, 2 up rung, 3 down rung)'),
    ('mk_hash', 'str', NONE, 'order hash, matches_20261108.maker_hash (0x + 64 lowercase hex) via LINK; "" when unobservable'),
    ('link_status', 'str', NONE, 'unique | unique_by_time | unique_time_inconsistent | ambiguous | missing | none (unfilled) | public_unlinked'),
    ('x_link_candidates', 'str', NONE, 'content candidates (hashes of a cid, or cids of an unlinked hash), | joined'),
    ('mk_signer', 'str', NONE, 'who.json our_address (== matches.maker of every own leg)'),
    ('quote', 'str', NONE, 'Bid (every own order buys a token; kernel_orders.up)'),
    ('index_set', 'int64', NONE, '1 if kernel_orders.up (BuyUp) else 2 (BuyDown)'),
    ('token_price', 'float64', NONE, 'core_orders.tick / 100 (up) or 1 - tick / 100 (down); == every Matched notice price'),
    ('y_side', 'str', NONE, 'pf_mt_common.yes_space(quote, index_set, token_price)[0]'),
    ('y_price', 'float64', NONE, 'pf_mt_common.yes_space(...)[1]'),
    ('y_ptick', 'int64', NONE, 'round(y_price * 100) (== core_orders.tick)'),
    ('size_sh', 'float64', NONE, 'TRUE size: kernel_orders.size (== core_orders.size)'),
    ('filled_sh', 'float64', NONE, 'core_orders.filled (sum of Filled qty; == sum of kernel Matched notice sizes)'),
    ('n_legs', 'int64', NONE, 'public legs of mk_hash (matches_20261108)'),
    ('legs_sh', 'float64', NONE, 'sum of the public legs shares'),
    ('leg_shares', 'str', NONE, 'public legs shares in (exec_s, tx) order, | joined repr'),
    ('final_state', 'str', NONE, 'filled (filled >= size - TOL) | cancelled (CancelPendingAudit) | expired (accepted, no removal record) | other'),
    ('true_removal', 'str', NONE, 'exhausted (filled) | cancelled_or_expired | "" (never live / unknown)'),
    ('remaining_lb_sh', 'float64', NONE, 'max(size - filled, 0): a lower bound on the remaining at any time'),
    ('req_bot_ms', 'float64', BOT, 'core_orders.t_req (Request Place t: the decision tick)'),
    ('place_bot_ms', 'float64', BOT, 'kernel_orders.t_place (Control Place)'),
    ('signed_bot_ms', 'float64', BOT, 'kernel_orders.t_signed (Exec Signed: the earliest the order can exist at the venue)'),
    ('accepted_bot_ms', 'float64', BOT, 'kernel_orders.t_accepted, else core_orders.t_Accepted (receipt of the Accepted answer)'),
    ('accepted_src', 'str', NONE, 'kernel.t_accepted | core.t_Accepted | ""'),
    ('rejected_bot_ms', 'float64', BOT, 'core_orders.t_Rejected'),
    ('cancel_req_bot_ms', 'float64', BOT, 'core_orders.t_cancel (Request Cancel t)'),
    ('cancel_ack_bot_ms', 'float64', BOT, 'kernel_orders.t_CancelPendingAudit (first Canceled answer received: an upper bound on the effect)'),
    ('cancel_echo_bot_ms', 'float64', BOT, 'kernel_orders.t_UserCancel (user-stream cancellation echo)'),
    ('cancel_done_core_bot_ms', 'float64', BOT, 'core_orders.t_CancelDone (core receipt of a cancel answer, not-found answers included)'),
    ('notice_first_bot_ms', 'float64', BOT, 'kernel_fills Matched min(at): first fill notice received'),
    ('notice_last_bot_ms', 'float64', BOT, 'kernel_fills Matched max(at)'),
    ('matched_first_vstatus_ms', 'float64', VST, 'kernel_fills Matched min(match_ms) (whole second)'),
    ('matched_last_vstatus_ms', 'float64', VST, 'kernel_fills Matched max(match_ms)'),
    ('confirmed_first_vstatus_ms', 'float64', VST, 'matches_20261108 min(exec_s) * 1000 (executedAt = Confirmed second)'),
    ('confirmed_last_vstatus_ms', 'float64', VST, 'matches_20261108 max(exec_s) * 1000'),
    ('d_lo_ms', 'float64', BANDC, "the order's RUN band lower end (venue status - bot; own_orders.json clock_band.per_run[run])"),
    ('d_hi_ms', 'float64', BANDC, "the order's RUN band upper end"),
    ('act_lo_venue_ms', 'float64', VBAND, 'signed_bot_ms + d_lo: activation lower end'),
    ('act_hi_venue_ms', 'float64', VBAND, 'min(accepted_bot_ms + d_hi, match_hi_first_venue_ms): activation upper end'),
    ('match_hi_first_venue_ms', 'float64', VBAND, 'min over Matched notices of min(match_ms + 1000, at + d_hi): bounds the first match'),
    ('match_hi_last_venue_ms', 'float64', VBAND, 'max over Matched notices of min(match_ms + 1000, at + d_hi): bounds every match'),
    ('match_hi_list_venue_ms', 'str', VBAND, 'every Matched notice bound min(match_ms + 1000, at + d_hi), ascending, | joined repr: the i-th '
                                             'earliest match precedes the i-th smallest bound'),
    ('rm_lo_venue_ms', 'float64', VBAND, 'removal lower end: filled -> act_lo; cancelled -> cancel_req + d_lo (act_lo without a request)'),
    ('rm_hi_venue_ms', 'float64', VBAND, 'removal upper end: filled -> match_hi_last; cancelled -> cancel_ack + d_hi; else NaN (unbounded)'),
    ('present_lo_venue_ms', 'float64', VBAND, 'certain presence from act_hi (accepted, cancelled, cancel requested; else NaN)'),
    ('present_hi_venue_ms', 'float64', VBAND, 'certain presence until cancel_req + d_lo'),
    ('status', 'str', NONE, 'ok | unfilled | refused | rejected | not_accepted | unconfirmed_live | no_removal_record | link_ambiguous | '
                            'link_missing | link_time_inconsistent | fill_mismatch | fill_price_mismatch | no_signed_time | '
                            'notice_before_signing | after_cutoff | public_hash_unlinked'),
]
COLNAMES = [c[0] for c in COLS]
KO = ['cid', 'market', 'up', 'price', 'size', 't_place', 'refused', 't_signed', 't_accepted', 't_CancelPendingAudit', 't_UserCancel']
CO = ['market', 'id', 'lane', 'tick', 'size', 't_req', 't_Accepted', 't_CancelDone', 't_Rejected', 't_cancel', 'filled', 'cid', 'refused', 'up']
KF = ['cid', 'at', 'match_ms', 'size', 'price', 'status']
MT = ['mid', 'exec_s', 'tx', 'maker', 'maker_hash', 'yes_side', 'ptick', 'shares']


def refuse(ok, msg):
    C.need(bool(ok), 'REFUSED: ' + msg)


def vmhwm_mib():
    for line in open('/proc/self/status'):
        if line.startswith('VmHWM:'):
            return int(line.split()[1]) / 1024.
    return 0.


def guard(where):
    m = vmhwm_mib()
    if m > C.GUARD_MIB:
        raise MemoryError(f'self-guard: VmHWM {m:.0f} MiB > {C.GUARD_MIB:.0f} MiB at {where}')


def code_sha():
    return {Path(p).name: sha256(p) for p in (__file__, C.__file__)}


def sha256(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def now_utc():
    return dt.datetime.now(dt.timezone.utc).isoformat()


# ------------------------------------------------------------------ build ------------------------------------------------------------------
def load_export():
    sc = json.loads((EXP / 'scope.json').read_text())
    refuse(float(sc['cutoff_ms']) == CUTOFF_MS, f'scope cutoff_ms {sc["cutoff_ms"]} differs from the declared 23:00Z')
    files = ['scope.json', 'live_vs_replay/who.json', 'live_vs_replay/matches_20261108.parquet'] + \
            [f'live_vs_replay/{k}_{r}.parquet' for k in ('kernel_orders', 'core_orders', 'kernel_fills') for r in RUNS]
    src = dict(windows=pd.DataFrame(sc['windows'])[['market', 'ws', 'run']],
               our=str(json.loads((LV / 'who.json').read_text())['our_address']).lower(), sha={f: sha256(EXP / f) for f in files})
    for r in RUNS:
        src[f'kernel_orders_{r}'] = pd.read_parquet(LV / f'kernel_orders_{r}.parquet', columns=KO)
        src[f'core_orders_{r}'] = pd.read_parquet(LV / f'core_orders_{r}.parquet', columns=CO)
        src[f'kernel_fills_{r}'] = pd.read_parquet(LV / f'kernel_fills_{r}.parquet', columns=KF)
    src['matches'] = pd.read_parquet(LV / 'matches_20261108.parquet', columns=MT)
    guard('load_export')
    return src


def clock_band(src):
    """K7 per run from the run's own notices: d = venue status - bot in [max(match_ms - at), min(match_ms + 1000 - t_place)]. One band PER RUN
    (the runs may be different bot hosts / clocks); each order uses its run's band (v2)."""
    per = {}
    for r in RUNS:
        F = src[f'kernel_fills_{r}']; K = src[f'kernel_orders_{r}'][['cid', 't_place']]
        f = F[F.status.isin(['Matched', 'Confirmed'])].merge(K, on='cid', how='left', validate='many_to_one')
        refuse(f.t_place.notna().all(), f'{r}: a fill notice without its kernel order')
        m = (f.status == 'Matched').to_numpy()
        refuse(m.any(), f'{r}: no Matched notice: the clock band is unmeasured')
        lo = (f.match_ms - f['at']).to_numpy(); hi = (f.match_ms + 1000. - f.t_place).to_numpy()[m]
        il, ih = int(np.argmax(lo)), int(np.argmin(hi))
        per[r] = dict(d_lo_ms=float(lo[il]), d_hi_ms=float(hi[ih]), n_lo=int(len(lo)), n_hi=int(m.sum()),
                      binding_lo=dict(cid=int(f.cid.iloc[il]), status=str(f.status.iloc[il])), binding_hi=dict(cid=int(f.cid[m].iloc[ih])))
        refuse(per[r]['d_lo_ms'] <= per[r]['d_hi_ms'], f'clock band EMPTY for run {r}: d_lo {per[r]["d_lo_ms"]} > d_hi {per[r]["d_hi_ms"]}')
    return dict(per_run=per, clock='venue STATUS clock - bot clock (ms), one band per run',
                definition='per run: d >= match_ms - at (Matched, Confirmed notices); d <= match_ms + 1000 - t_place (Matched)',
                premises=['one constant bot-to-venue offset within each run (each run its own bot clock)', 'a notice is received after its status second starts',
                          'a match follows its placement and precedes the end of its Matched second', 'activation in [signed, Accepted answer received]',
                          'a cancel takes effect in [request, first Canceled answer received]',
                          'NOT ESTABLISHED: the lifecycle venue book / UUID / collector clocks equal the venue status clock'])


def _need_cols(df, cols, name):
    miss = [c for c in cols if c not in df.columns]
    refuse(not miss, f'{name}: missing columns {miss}')


def build_truth(src):
    """-> (truth DataFrame with COLS, receipt dict). Pure: reads only the src dict (load_export or a synthetic fixture)."""
    for r in RUNS:
        for k, cols in (('kernel_orders', KO), ('core_orders', CO), ('kernel_fills', KF)):
            _need_cols(src[f'{k}_{r}'], cols, f'{k}_{r}')
    _need_cols(src['matches'], MT, 'matches')
    our = src['our']; refuse(ADDR_RE.fullmatch(our), f'our_address {our!r} is not 0x + 40 lowercase hex')
    band = clock_band(src)
    W = src['windows']; refuse(W.market.is_unique, 'scope windows repeat a market'); run_of = dict(zip(W.market, W.run))
    parts, checks = [], {}
    for r in RUNS:
        K = src[f'kernel_orders_{r}'][KO]; P = src[f'core_orders_{r}'][CO]; F = src[f'kernel_fills_{r}'][KF]
        refuse(K.cid.is_unique and P.cid.is_unique and set(K.cid) == set(P.cid), f'{r}: kernel and core order sets differ')
        J = P.merge(K, on='cid', suffixes=('', '_k'), validate='one_to_one')
        agree = ((J.market == J.market_k) & (J.up == J.up_k) & (J.tick == J.price) & ((J['size'] - J.size_k).abs() < 1e-9)
                 & (J.refused.fillna('') == J.refused_k.fillna('')))
        refuse(agree.all(), f'{r}: kernel and core disagree on market / side / price / size / refusal for {int((~agree).sum())} orders')
        refuse(J.market.map(run_of).eq(r).all(), f"{r}: an order outside its run's scope windows")
        refuse(F.cid.isin(J.cid).all(), f'{r}: a fill notice for an unknown order')
        b = band['per_run'][r]; d0, d1 = b['d_lo_ms'], b['d_hi_ms']
        tp = np.where(J.up, J.tick / 100., 1. - J.tick / 100.).round(6)
        ys = [C.yes_space('Bid', 1 if u else 2, float(p)) for u, p in zip(J.up, tp)]
        Fm = F[F.status == 'Matched'].merge(J[['cid']].assign(tp=tp), on='cid')
        Fm = Fm.assign(mhi=np.minimum(Fm.match_ms + 1000., Fm['at'] + d1), pbad=(Fm.price - Fm.tp).abs() > 1e-9)
        g = Fm.groupby('cid')
        A = pd.DataFrame(dict(m_sh=g['size'].sum(), at0=g['at'].min(), at1=g['at'].max(), mm0=g.match_ms.min(), mm1=g.match_ms.max(),
                              mhi0=g.mhi.min(), mhi1=g.mhi.max(), pbad=g.pbad.any(), sizes=g['size'].apply(lambda s: tuple(sorted(s))),
                              mhl=g.mhi.apply(lambda s: '|'.join(repr(float(x)) for x in sorted(s)))))
        conf = F[F.status == 'Confirmed'].groupby('cid').match_ms.apply(lambda s: frozenset(int(x) // 1000 for x in s))
        acc = J.t_accepted.where(J.t_accepted.notna(), J.t_Accepted)
        D = pd.DataFrame(dict(run=r, mid=J.market.astype('int64'), cid=J.cid.astype('int64'), core_id=J['id'].astype('int64'),
                              lane=J.lane.astype('int64'), quote='Bid', index_set=np.where(J.up, 1, 2).astype('int64'), token_price=tp,
                              y_side=[s for s, _ in ys], y_price=[p for _, p in ys], size_sh=J['size'].astype(float), filled_sh=J.filled.astype(float),
                              req_bot_ms=J.t_req, place_bot_ms=J.t_place, signed_bot_ms=J.t_signed, accepted_bot_ms=acc,
                              accepted_src=np.where(J.t_accepted.notna(), 'kernel.t_accepted', np.where(J.t_Accepted.notna(), 'core.t_Accepted', '')),
                              rejected_bot_ms=J.t_Rejected, cancel_req_bot_ms=J.t_cancel, cancel_ack_bot_ms=J.t_CancelPendingAudit,
                              cancel_echo_bot_ms=J.t_UserCancel, cancel_done_core_bot_ms=J.t_CancelDone, x_refused=J.refused.notna().to_numpy(),
                              d_lo_ms=d0, d_hi_ms=d1))
        D = D.join(A, on='cid'); D['conf'] = D.cid.map(conf)
        checks[r] = dict(orders=len(D), core_filled_eq_matched=int(((D.filled_sh - D.m_sh.fillna(0.)).abs() <= TOL_SH).sum()),
                         matched_notices=len(Fm), matched_price_eq_token_price=int((~Fm.pbad).sum()),
                         accepted_from_core_only=int((D.accepted_src == 'core.t_Accepted').sum()))
        parts.append(D)
    D = pd.concat(parts, ignore_index=True)
    D0, D1 = D.d_lo_ms, D.d_hi_ms                 # per row: the order's run band
    D['y_ptick'] = np.rint(D.y_price * 100.).astype('int64')
    filled = D.filled_sh > TOL_SH

    # ---- own public legs and the link ----
    M = src['matches'][MT]; L = M[M.maker.str.lower() == our].copy()
    refuse(L.maker_hash.map(lambda h: bool(HASH_RE.fullmatch(h))).all(), 'an own leg hash is not 0x + 64 lowercase hex')
    refuse(not L.duplicated(['tx', 'maker_hash']).any(), 'duplicate own leg (tx, hash)')
    refuse(L.mid.isin(run_of).all(), 'an own public leg outside the scope windows')
    L = L.sort_values(['maker_hash', 'exec_s', 'tx'], kind='stable'); gl = L.groupby('maker_hash')
    H = pd.DataFrame(dict(mid=gl.mid.first(), n_mid=gl.mid.nunique(), y_side=gl.yes_side.first(), n_side=gl.yes_side.nunique(),
                          y_ptick=gl.ptick.first(), n_pt=gl.ptick.nunique(), n_legs=gl.size(), legs_sh=gl.shares.sum(), e0=gl.exec_s.min(),
                          e1=gl.exec_s.max(), secs=gl.exec_s.apply(lambda s: frozenset(int(x) for x in s)),
                          shs=gl.shares.apply(lambda s: tuple(sorted(s))), leg_shares=gl.shares.apply(lambda s: '|'.join(repr(float(x)) for x in s))))
    H.index.name = 'mk_hash'
    h_ok = (H.n_mid == 1) & (H.n_side == 1) & (H.n_pt == 1)
    cand = D[filled][['cid', 'mid', 'y_side', 'y_ptick', 'filled_sh', 'sizes', 'conf']].merge(
        H[h_ok][['mid', 'y_side', 'y_ptick', 'legs_sh', 'secs', 'shs']].reset_index(), on=['mid', 'y_side', 'y_ptick'])
    cand = cand[(cand.filled_sh - cand.legs_sh).abs() <= TOL_SH].copy()

    def time_ok(r):
        a, b = r.shs, r.sizes if isinstance(r.sizes, tuple) else ()
        return (isinstance(r.conf, frozenset) and r.secs <= r.conf and len(a) == len(b) and all(abs(x - y) <= TOL_SH for x, y in zip(a, b)))
    cand['tok'] = [time_ok(r) for r in cand.itertuples()]
    cc = cand.groupby('cid').mk_hash.apply(set).to_dict(); ch = cand.groupby('mk_hash').cid.apply(set).to_dict()
    tc = cand[cand.tok].groupby('cid').mk_hash.apply(set).to_dict(); th = cand[cand.tok].groupby('mk_hash').cid.apply(set).to_dict()
    link = {}
    for c, hs in cc.items():
        if len(hs) == 1 and len(ch[next(iter(hs))]) == 1:
            h = next(iter(hs)); link[c] = (h, 'unique' if c in th.get(h, set()) else 'unique_time_inconsistent')
        else:
            t = tc.get(c, set())
            link[c] = (next(iter(t)), 'unique_by_time') if len(t) == 1 and th.get(next(iter(t)), set()) == {c} else ('', 'ambiguous')
    lk = [h for h, s in link.values() if h]
    refuse(len(lk) == len(set(lk)), 'one public hash linked to two orders')
    nonu = {c for c, (_, s) in link.items() if s != 'unique'}
    link_evidence = [dict(cid=int(r.cid), mk_hash=r.mk_hash, link=link[r.cid][1], linked=link[r.cid][0] == r.mk_hash, leg_exec_s=sorted(r.secs),
                          confirmed_notice_s=sorted(r.conf) if isinstance(r.conf, frozenset) else [], leg_shares=list(r.shs),
                          matched_sizes=list(r.sizes) if isinstance(r.sizes, tuple) else [], time_consistent=bool(r.tok))
                     for r in cand[cand.cid.isin(nonu)].itertuples()]
    D['mk_hash'] = [link.get(c, ('', ''))[0] for c in D.cid]
    D['link_status'] = np.where(~filled, 'none', [link.get(c, ('', 'missing'))[1] for c in D.cid])
    D['x_link_candidates'] = ['|'.join(sorted(cc.get(c, ()))) for c in D.cid]
    Hj = H[['n_legs', 'legs_sh', 'leg_shares', 'e0', 'e1']]
    D = D.join(Hj, on='mk_hash')
    D['n_legs'] = D.n_legs.fillna(0).astype('int64'); D['legs_sh'] = D.legs_sh.fillna(0.); D['leg_shares'] = D.leg_shares.fillna('')
    D['confirmed_first_vstatus_ms'] = D.e0 * 1000.; D['confirmed_last_vstatus_ms'] = D.e1 * 1000.
    D['mk_signer'] = our

    # ---- final state, removal, venue-band times ----
    full = D.filled_sh >= D.size_sh - TOL_SH; ack = D.cancel_ack_bot_ms.notna(); accd = D.accepted_bot_ms.notna()
    D['final_state'] = np.select([full & filled, ack, accd & D.cancel_done_core_bot_ms.isna()], ['filled', 'cancelled', 'expired'], 'other')
    D['true_removal'] = np.select([D.final_state == 'filled', D.final_state.isin(['cancelled', 'expired'])], ['exhausted', 'cancelled_or_expired'], '')
    D['remaining_lb_sh'] = (D.size_sh - D.filled_sh).clip(lower=0.)
    D['notice_first_bot_ms'] = D.at0; D['notice_last_bot_ms'] = D.at1
    D['matched_first_vstatus_ms'] = D.mm0; D['matched_last_vstatus_ms'] = D.mm1
    D['match_hi_first_venue_ms'] = D.mhi0; D['match_hi_last_venue_ms'] = D.mhi1; D['match_hi_list_venue_ms'] = D.mhl
    D['act_lo_venue_ms'] = D.signed_bot_ms + D0
    D['act_hi_venue_ms'] = np.fmin((D.accepted_bot_ms + D1).to_numpy(), D.mhi0.to_numpy())
    canc = D.final_state == 'cancelled'
    D['rm_lo_venue_ms'] = np.select([D.final_state == 'filled', canc & D.cancel_req_bot_ms.notna(), canc | (D.final_state == 'expired')],
                                    [D.act_lo_venue_ms, D.cancel_req_bot_ms + D0, D.act_lo_venue_ms], np.nan)
    D['rm_hi_venue_ms'] = np.select([D.final_state == 'filled', canc], [D.mhi1, D.cancel_ack_bot_ms + D1], np.nan)
    plo = (D.accepted_bot_ms + D1).where(accd & canc & D.cancel_req_bot_ms.notna())
    plo = pd.Series(np.fmin(plo.to_numpy(), D.mhi0.to_numpy()), index=D.index).where(plo.notna())
    phi = (D.cancel_req_bot_ms + D0).where(plo.notna())
    keep = plo < phi
    D['present_lo_venue_ms'] = plo.where(keep); D['present_hi_venue_ms'] = phi.where(keep)
    tmax = D[['place_bot_ms', 'signed_bot_ms', 'accepted_bot_ms', 'cancel_req_bot_ms', 'cancel_ack_bot_ms', 'notice_last_bot_ms']].max(axis=1)
    lst = D.link_status
    D['status'] = np.select(
        [D.x_refused, filled & (lst == 'ambiguous'), filled & (lst == 'missing'), filled & (lst == 'unique_time_inconsistent'),
         filled & ((D.filled_sh - D.m_sh.fillna(0.)).abs() > TOL_SH), filled & D.pbad.fillna(False).astype(bool),
         filled & D.signed_bot_ms.isna(), filled & (D.notice_first_bot_ms < D.signed_bot_ms), tmax > CUTOFF_MS,
         filled & ~D.final_state.isin(['filled', 'cancelled']), filled,
         D.rejected_bot_ms.notna() & ~accd, ~accd & ack, ~accd, ~ack],
        ['refused', 'link_ambiguous', 'link_missing', 'link_time_inconsistent', 'fill_mismatch', 'fill_price_mismatch', 'no_signed_time',
         'notice_before_signing', 'after_cutoff', 'no_removal_record', 'ok', 'rejected', 'unconfirmed_live', 'not_accepted', 'no_removal_record'],
        'unfilled')
    # an own public hash with no order is a row, never a silent drop
    un = H[~H.index.isin(lk)]
    if len(un):
        U = pd.DataFrame(dict(run=un.mid.map(run_of).fillna('').astype(str), mid=un.mid.astype('int64'), cid=-1, core_id=-1, lane=-1,
                              mk_hash=un.index, link_status='public_unlinked', x_link_candidates=['|'.join(map(str, sorted(ch.get(h, ())))) for h in un.index],
                              mk_signer=our, n_legs=un.n_legs.astype('int64'), legs_sh=un.legs_sh, leg_shares=un.leg_shares, y_side=un.y_side,
                              y_ptick=un.y_ptick.astype('int64'), y_price=un.y_ptick / 100., confirmed_first_vstatus_ms=un.e0 * 1000.,
                              confirmed_last_vstatus_ms=un.e1 * 1000., final_state='other', true_removal='', status='public_hash_unlinked',
                              d_lo_ms=un.mid.map(run_of).map(lambda r: band['per_run'][r]['d_lo_ms']),
                              d_hi_ms=un.mid.map(run_of).map(lambda r: band['per_run'][r]['d_hi_ms'])))
        D = pd.concat([D, U], ignore_index=True)
    T = D[COLNAMES].copy()
    for n, t, _, _ in COLS:
        if t == 'int64':
            T[n] = T[n].fillna(-1).astype('int64')
        elif t == 'float64':
            T[n] = T[n].astype('float64')
        else:
            T[n] = T[n].fillna('').astype(str)
    T = T.sort_values(['mid', 'cid', 'mk_hash'], kind='stable').reset_index(drop=True)
    hx = T.mk_hash[T.mk_hash != '']
    refuse(hx.is_unique, 'duplicate order hash in truth')
    ok = T.status == SCORED
    mk = lambda s: sorted(int(x) for x in set(s))  # noqa: E731
    rec = dict(version=C.VERSION, module='pf_mt_truth', built_utc=now_utc(), code=code_sha(), inputs=src.get('sha', {}), clock_band=band, tol_sh=TOL_SH,
               cutoff_ms=CUTOFF_MS, signer=dict(our_address=our, signers_of_own_legs=sorted(L.maker.str.lower().unique().tolist())),
               own_public=dict(legs=int(len(L)), hashes=int(len(H)), shares=float(L.shares.sum()), hash_identity_conflicts=int((~h_ok).sum())),
               rows=int(len(T)), status=T.status.value_counts().to_dict(), final_state=T.final_state.value_counts().to_dict(),
               link_status=T.link_status.value_counts().to_dict(), checks=checks,
               scored=dict(orders=int(ok.sum()), legs=int(T.n_legs[ok].sum()), shares=float(T.legs_sh[ok].sum()),
                           exhausted=int((ok & (T.true_removal == 'exhausted')).sum()), cancelled=int((ok & (T.true_removal == 'cancelled_or_expired')).sum())),
               presence_rows=int(T.present_lo_venue_ms.notna().sum()), link_evidence=link_evidence,
               gt_markets=dict(scope=int(len(W)), with_own_orders=len(set(T.mid[T.cid >= 0])), with_accepted=len(set(T.mid[T.accepted_bot_ms.notna()])),
                               with_own_fills=len(set(T.mid[T.filled_sh > TOL_SH])), by_run={r: dict(scope=int((W.run == r).sum()),
                               with_own_fills=len(set(T.mid[(T.run == r) & (T.filled_sh > TOL_SH)]))) for r in RUNS},
                               own_fill_mids=mk(T.mid[(T.filled_sh > TOL_SH) | (T.status == 'public_hash_unlinked')])),
               columns={n: dict(dtype=t, clock=c, source=s) for n, t, c, s in COLS})
    return T, rec


def _write_once(path, write):
    refuse(not path.exists(), f'{path} exists (write-once)')
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f'.{path.name}.tmp{os.getpid()}')
    write(tmp)
    try:
        os.link(tmp, path)
    except FileExistsError:
        raise ValueError(f'REFUSED: {path} exists (write-once)')
    finally:
        os.unlink(tmp)


def _meta(rec):
    """The parquet registry (also what the dry gate compares): per-run clock bands, clock, premises, column registry."""
    return dict(version=C.VERSION, clock_band={k: rec['clock_band'][k] for k in ('per_run', 'clock', 'premises')}, columns=rec['columns'])


def write_truth(T, rec, out_dir):
    meta = dict(_meta(rec), built_utc=rec['built_utc'])
    tab = pa.Table.from_pandas(T, preserve_index=False)
    tab = tab.replace_schema_metadata({**(tab.schema.metadata or {}), META_KEY: json.dumps(meta).encode()})
    _write_once(out_dir / 'own_orders.parquet', lambda p: pq.write_table(tab, p))
    _write_once(out_dir / 'own_orders.json', lambda p: p.write_text(json.dumps(rec, indent=1, default=str)))


def read_truth(path):
    tab = pq.read_table(path)
    md = tab.schema.metadata or {}
    refuse(META_KEY in md, f'{path}: no {META_KEY.decode()} registry in the parquet metadata')
    T = tab.to_pandas(); T.attrs['mt_truth'] = json.loads(md[META_KEY])
    return T


# ------------------------------------------------------------------ score ------------------------------------------------------------------
COMPARE = [  # (lifecycle venue-clock field, the truth column it is tested against)
    ('orders.post_lo_ms', 'act_lo_venue_ms'), ('orders.post_hi_ms', 'act_hi_venue_ms'), ('orders.removal_lo_ms', 'rm_lo_venue_ms'),
    ('orders.removal_hi_ms', 'rm_hi_venue_ms'), ('orders.persist_lo_ms', 'act_lo_venue_ms'), ('orders.persist_hi_ms', 'rm_hi_venue_ms'),
    ('fills.x_lo_hard_ms', 'match_hi_last_venue_ms'), ('fills.lo_ms', 'match_hi_last_venue_ms'), ('fills.hi_ms', 'act_lo_venue_ms'),
    ('fills.lo_ms (queue frame)', 'present_lo_venue_ms'),
    ('fills.lo_ms (queue frame)', 'present_hi_venue_ms')]
ORD_TIMES = ['first_hi_ms', 'last_hi_ms', 'post_lo_ms', 'post_hi_ms', 'persist_lo_ms', 'persist_hi_ms', 'removal_lo_ms', 'removal_hi_ms']
CLS_COLS = ['size_class', 'post_class', 'persist_class', 'queue_class', 'removal_class']
UNINFORMATIVE = {'end_censored', 'missing_evidence'}


def _registry_check(T):
    """The truth's venue columns must be registered on a venue clock AND hold bot times converted with the registered band."""
    reg = T.attrs.get('mt_truth')
    refuse(isinstance(reg, dict) and 'columns' in reg and 'clock_band' in reg, 'truth carries no clock registry (read it with read_truth)')
    pr = reg['clock_band'].get('per_run')
    refuse(isinstance(pr, dict) and pr and all(np.isfinite(float(b['d_lo_ms'])) and np.isfinite(float(b['d_hi_ms'])) and float(b['d_lo_ms']) <= float(b['d_hi_ms'])
                                               for b in pr.values()), 'truth clock band missing or empty (one band per run)')
    for fld, col in COMPARE:
        clk = reg['columns'].get(col, {}).get('clock')
        refuse(clk in (VST, VBAND), f'clock-domain mismatch: truth column {col} is on {clk!r}, compared with the venue-clock lifecycle field {fld}')
    refuse(T.run.isin(list(pr)).all(), "a truth row's run has no registered clock band")
    rlo, rhi = T.run.map(lambda r: float(pr[r]['d_lo_ms'])), T.run.map(lambda r: float(pr[r]['d_hi_ms']))
    refuse(((T.d_lo_ms - rlo).abs() <= 1e-9).all() and ((T.d_hi_ms - rhi).abs() <= 1e-9).all(),
           "clock-domain mismatch: a row band differs from its run's registered band")
    for col, base in (('act_lo_venue_ms', 'signed_bot_ms'), ('present_hi_venue_ms', 'cancel_req_bot_ms')):
        x = T[[col, base, 'd_lo_ms']].dropna(); off = (x[col] - x[base] - x.d_lo_ms).abs()
        refuse(not (off > 1e-6).any(), f'clock-domain mismatch: {col} != {base} + band ({int((off > 1e-6).sum())} rows; bot times not converted)')
    x = T[['act_hi_venue_ms', 'accepted_bot_ms', 'd_hi_ms']].dropna()
    refuse(not (x.act_hi_venue_ms > x.accepted_bot_ms + x.d_hi_ms + 1e-6).any(), 'clock-domain mismatch: act_hi_venue_ms > accepted_bot_ms + d_hi')
    return reg


def _unit_check(df, cols, name):
    for c in cols:
        v = df[c].to_numpy(dtype=float); v = v[v >= 0]
        bad = (v < MS_RANGE[0]) | (v > MS_RANGE[1])
        refuse(not bad.any(), f'clock/unit mismatch: {name}.{c} holds {int(bad.sum())} bounded values outside epoch ms (e.g. {v[bad][:1].tolist()})')


def _rint(lo, hi):
    lo = np.asarray(lo, dtype=float); hi = np.asarray(hi, dtype=float)
    return np.where(lo < 0, -INF, lo), np.where(hi < 0, INF, hi)


def _tint(lo, hi):
    lo = np.asarray(lo, dtype=float); hi = np.asarray(hi, dtype=float)
    return np.where(np.isnan(lo), -INF, lo), np.where(np.isnan(hi), INF, hi)


def _rel(tlo, thi, rlo, rhi):
    """truth hull [tlo, thi] vs claim [rlo, rhi] -> (outcome, gap_ms): inside (every admissible truth in the claim) | disjoint | overlap."""
    dis = (thi < rlo) | (tlo > rhi); ins = (rlo <= tlo) & (thi <= rhi)
    gap = np.where(thi < rlo, rlo - thi, np.where(tlo > rhi, tlo - rhi, 0.))
    return np.select([dis, ins], ['disjoint', 'inside'], 'overlap'), gap


def _by_class(cls, outcome, fail, gap=None):
    d = pd.DataFrame(dict(cls=np.asarray(cls, dtype=object), out=np.asarray(outcome, dtype=object), fail=np.asarray(fail, dtype=bool)))
    if gap is not None:
        d['gap'] = np.asarray(gap, dtype=float)
    outs = sorted(set(d.out))
    res = {}
    for c in CLASSES:
        x = d[d.cls == c]; nf = int(x.fail.sum())
        res[c] = dict(n=int(len(x)), **{o: int((x.out == o).sum()) for o in outs}, failed=nf,
                      defects=nf if c in (C.OBSERVED, C.CONSTRAINED) else 0, sensitive_misses=nf if c == C.SENSITIVE else 0)
        if gap is not None and nf:
            res[c]['gap_ms_failed'] = dict(min=round(float(x.gap[x.fail].min()), 3), max=round(float(x.gap[x.fail].max()), 3))
    return res


def _defects(block):
    return sum(v['defects'] for v in block.values())


def _match_multiset(a, b):
    """greedy match of sorted share lists within TOL_SH -> (matched pairs, unmatched a, unmatched b)."""
    a, b = sorted(a), sorted(b); i = j = n = 0; ua, ub = [], []
    while i < len(a) and j < len(b):
        if abs(a[i] - b[j]) <= TOL_SH:
            n += 1; i += 1; j += 1
        elif a[i] < b[j]:
            ua.append(a[i]); i += 1
        else:
            ub.append(b[j]); j += 1
    return n, ua + a[i:], ub + b[j:]


CLASSES = C.CLASSES


def score(orders, fills, truth, public_hashes, gt_mids, censored_mids=()):
    """-> scores dict. orders / fills: the lifecycle tables (SCHEMAS + x_size_ub / x_rec_rank) of the GT markets; truth: read_truth();
    public_hashes: every maker hash of the public data of those markets; gt_mids: the scope markets; censored_mids: boundary_censored GT
    windows, excluded from scoring and counted. Refuses (ValueError 'REFUSED: ...') on any contract breach."""
    _need_cols(truth, COLNAMES, 'truth')
    reg = _registry_check(truth)
    _need_cols(orders, list(C.SCHEMAS['orders']) + ['x_size_ub'], 'lifecycle orders')
    _need_cols(fills, list(C.SCHEMAS['fills']) + ['x_rec_rank', 'x_lo_hard_ms'], 'lifecycle fills')
    hx = truth.mk_hash[truth.mk_hash != '']
    refuse(hx.is_unique, f'duplicate order hash in truth ({int(hx.duplicated().sum())} repeats)')
    sig = set(truth.mk_signer.astype(str).str.lower()); refuse(len(sig) == 1, f'truth has {len(sig)} signers, expected our one address'); OUR = sig.pop()  # addresses: lowercase (REST signers are EIP-55 mixed case)
    gt = set(int(m) for m in gt_mids); pub = set(public_hashes)
    refuse(set(orders.scenario) <= set(C.SCENARIOS), f'unknown scenario {sorted(set(orders.scenario) - set(C.SCENARIOS))}')
    for c in CLS_COLS:
        refuse(set(orders[c]) <= set(CLASSES), f'unknown class in {c}: {sorted(set(orders[c]) - set(CLASSES))}')
    rs = orders.removal_set.map(lambda s: frozenset(x for x in str(s).split('|') if x))
    refuse(rs.map(len).gt(0).all(), 'an empty removal_set')
    bad = set().union(*rs) - set(C.REMOVAL) if len(rs) else set()
    refuse(not bad, f'unknown removal member {sorted(bad)}')
    refuse(orders.mid.isin(gt).all() and fills.mid.isin(gt).all(), 'a lifecycle row outside the GT markets')
    for nm, df in (('orders', orders), ('fills', fills)):
        fmt = df.mk_hash.map(lambda h: bool(HASH_RE.fullmatch(str(h))))
        refuse(fmt.all(), f'lifecycle {nm} hash format: {int((~fmt).sum())} rows not 0x + 64 lowercase hex (e.g. {df.mk_hash[~fmt].head(1).tolist()})')
        miss = ~df.mk_hash.isin(pub)
        refuse(not miss.any(), f'lifecycle {nm} row for a hash not in the public data ({int(miss.sum())} rows, e.g. {df.mk_hash[miss].head(1).tolist()})')
    refuse(not orders.duplicated(['scenario', 'mid', 'mk_hash']).any(), 'duplicate (scenario, mid, mk_hash) in lifecycle orders')
    refuse(not fills.duplicated(['mid', 'sid', 'mk_hash', 'mk_idx']).any(), 'duplicate (mid, sid, mk_hash, mk_idx) in lifecycle fills')
    _unit_check(orders, ORD_TIMES, 'orders'); _unit_check(fills, ['hi_ms', 'lo_ms', 'x_lo_hard_ms'], 'fills')
    _unit_check(truth, [n for n, t, c, _ in COLS if c in (BOT, VST, VBAND) and t == 'float64'], 'truth')

    # ---- boundary_censored GT windows: excluded from scoring, counted ----
    cen = set(int(m) for m in censored_mids)
    T0 = truth[(truth.status == SCORED) & (truth.mk_hash != '')]; tc0 = T0.mid.isin(cen)
    excl = dict(windows=sorted(cen), lifecycle_orders_rows=int(orders.mid.isin(cen).sum()), lifecycle_fills_rows=int(fills.mid.isin(cen).sum()),
                own_orders=int(tc0.sum()), own_legs=int(T0.n_legs[tc0].sum()), own_shares=round(float(T0.legs_sh[tc0].sum()), 6))
    orders = orders[~orders.mid.isin(cen)]; fills = fills[~fills.mid.isin(cen)]
    T = T0[~T0.mid.isin(cen)].set_index('mk_hash')
    own_all = set(hx); hcid = dict(zip(truth.mk_hash[truth.mk_hash != ''], truth.cid[truth.mk_hash != '']))
    defects = []

    def note(sc, cat, rows, cls, detail, field=None):
        for (m, h), c, d in zip(rows, cls, detail):
            defects.append(dict(scenario=sc, category=cat, field=field or cat, mid=int(m), mk_hash=h, cls=c, detail=d))

    # ---- fills: execution / share coverage, identity, windows, multi-fill lo order (scenario-independent) ----
    Fo = fills[fills.mk_hash.isin(T.index)]; ex = dict(legs=0, matched=0, shares=0., matched_shares=0., phantom=0, ident_bad=0)
    lo_ord = dict(multi_fill_orders=0, compared=0, count_mismatch=0, violations=0, refined_violations=0)
    fdef = []
    for h, t in T.iterrows():
        f = Fo[Fo.mk_hash == h]; legs = [float(x) for x in t.leg_shares.split('|') if x]
        n, ua, ub = _match_multiset(legs, f.shares.tolist())
        ex['legs'] += len(legs); ex['matched'] += n; ex['shares'] += sum(legs); ex['matched_shares'] += sum(legs) - sum(ua); ex['phantom'] += len(ub)
        if ub:
            fdef.append(('fill_phantom', t.mid, h, f'{len(ub)} fills unmatched to own legs {ub}'))
        bi = (f.mid != t.mid) | (f.y_side != t.y_side) | (f.y_ptick != t.y_ptick)
        ex['ident_bad'] += int(bi.sum())
        if bi.any():
            fdef.append(('fill_identity', t.mid, h, f'{int(bi.sum())} fills off (mid, side, tick)'))
        if t.n_legs >= 2:                      # N4: the i-th smallest fill lo must precede the i-th smallest per-notice match bound
            lo_ord['multi_fill_orders'] += 1
            bnd = sorted(float(x) for x in t.match_hi_list_venue_ms.split('|') if x)
            if len(f) != len(bnd):
                lo_ord['count_mismatch'] += 1
            else:
                lo_ord['compared'] += 1                                         # defect on the OBSERVED lower ends x_lo_hard_ms
                los = sorted(float(x) for x in f.x_lo_hard_ms)
                viol = [i for i, (a, b) in enumerate(zip(los, bnd)) if a >= 0 and a > b]
                if viol:
                    lo_ord['violations'] += 1; i = viol[0]
                    fdef.append(('fill_lo_order', t.mid, h, f'sorted hard fill lo #{i + 1} is {los[i] - bnd[i]:.1f} ms after the #{i + 1} notice bound'))
                lor = sorted(float(x) for x in f.lo_ms)                         # sensitivity on the refined lower ends lo_ms
                lo_ord['refined_violations'] += int(any(x >= 0 and x > y for x, y in zip(lor, bnd)))
    tl = T.loc[Fo.mk_hash]
    tlo, thi = _tint(tl.act_lo_venue_ms, tl.match_hi_last_venue_ms)
    flo, fhi = _rint(Fo.x_lo_hard_ms, Fo.hi_ms)                                # OBSERVED window [x_lo_hard_ms, hi_ms]: a miss is a defect
    fo, fg = _rel(tlo, thi, flo, fhi)
    fw = _by_class([C.OBSERVED] * len(Fo), fo, fo == 'disjoint', fg)
    rlo_r, rhi_r = _rint(Fo.lo_ms, Fo.hi_ms)                                   # refined window [lo_ms, hi_ms]: a miss is a sensitive miss
    fo_r, fg_r = _rel(tlo, thi, rlo_r, rhi_r)
    fw_r = _by_class([C.SENSITIVE] * len(Fo), fo_r, fo_r == 'disjoint', fg_r)
    for m, h, g in zip(Fo.mid[fo == 'disjoint'], Fo.mk_hash[fo == 'disjoint'], fg[fo == 'disjoint']):
        fdef.append(('fill_window', m, h, f'observed fill window [x_lo_hard_ms, hi_ms] misses the true match bound by {g:.1f} ms'))
    for c, m, h, d in fdef:
        note('fills', 'fills', [(m, h)], [C.OBSERVED], [f'{c}: {d}'], field=f'fills.{c}')
    fills_block = dict(executions=dict(n=ex['legs'], covered=ex['matched'], rate=round(ex['matched'] / ex['legs'], 6) if ex['legs'] else None),
                       shares=dict(n=round(ex['shares'], 6), covered=round(ex['matched_shares'], 6),
                                   rate=round(ex['matched_shares'] / ex['shares'], 6) if ex['shares'] else None),
                       phantom_fills=ex['phantom'], identity_mismatches=ex['ident_bad'], window=fw, window_refined=fw_r, lo_order=lo_ord,
                       own_fill_status=Fo.status.value_counts().to_dict(), t_method=Fo.t_method.value_counts().to_dict(),
                       defects=len(fdef))

    # ---- queue evidence: own orders certainly resting (truth presence interval, remaining > 0) ----
    pres = truth[truth.status.isin(PRESENT_OK) & truth.present_lo_venue_ms.notna() & (truth.remaining_lb_sh > TOL_SH)]
    PI = {k: (g.present_lo_venue_ms.to_numpy(), g.present_hi_venue_ms.to_numpy(), g.cid.to_numpy()) for k, g in pres.groupby(['mid', 'y_side', 'y_ptick'])}
    ff = fills.sort_values(['mk_hash', 'hi_ms', 'x_rec_rank'], kind='stable').drop_duplicates('mk_hash').set_index('mk_hash')

    def evidence(O, self_cid):
        tref = O.mk_hash.map(ff.lo_ms).fillna(-1).to_numpy(dtype=float); out = []
        for (m, s, p), tr, sc in zip(zip(O.mid, O.y_side, O.y_ptick), tref, self_cid):
            if tr < 0:
                out.append('no_ref_time'); continue
            lo, hi, cid = PI.get((m, s, p), (np.array([]),) * 3)
            hit = (lo <= tr) & (tr <= hi) & (cid != sc)
            out.append('evidence_against' if hit.any() else 'no_evidence')
        return np.array(out, dtype=object)

    res = dict(version=C.VERSION, module='pf_mt_truth', scored_utc=now_utc(), clock_band=reg['clock_band'], tol_sh=TOL_SH,
               caveats=dict(outcome_selection='only filled orders print a public hash: every score is on FILLED own orders (outcome-selected); '
                                              'unfilled own orders enter only as queue-presence evidence',
                            clock_premise=list(reg['clock_band']['premises']),
                            clock_bands_ms={r: [b['d_lo_ms'], b['d_hi_ms']] for r, b in reg['clock_band']['per_run'].items()}),
               conventions=dict(unbounded='a negative lifecycle int64 time is unbounded on that side; a NaN truth bound is unbounded',
                                queue_reference='the order\'s first fill (min hi_ms, then the record rank x_rec_rank, as pf_mt_lifecycle); t_ref = its lo_ms (the f_before frame time)',
                                fill_window='observed = [x_lo_hard_ms (hi - MAX_LAG_MS; -1 receipt-hi), hi_ms]: a miss is a defect; refined = [lo_ms, hi_ms]: '
                                            'a miss is a sensitive miss (window_refined)',
                                lo_order='multi-fill own orders: sorted x_lo_hard_ms vs sorted per-notice match bounds, hard #i > bound #i is a defect; '
                                         'the same on lo_ms is counted as refined_violations (sensitive); count mismatch: skipped, counted',
                                size_ub='x_size_ub (NaN = not stated) is classed by size_class; true size > ub is a failed claim',
                                queue_evidence='an own order (truth status ok / unfilled) certainly resting at t_ref at that side / tick with remaining > 0: '
                                               'shown(p) at f_before then exceeds the claimed order\'s fills after f_before, so ahead_hi == 0 is '
                                               'contradicted - by a wrong shown size or frame, or by the claimed order not resting at f_before',
                                presence='[accepted + d_hi, cancel request + d_lo] of cancelled own orders (venue band, exact under one offset)',
                                defect='a failed observed / constrained claim; assumption_sensitive failures are sensitive misses'),
               population=dict(truth_rows=int(len(truth)), truth_status=truth.status.value_counts().to_dict(), scored_own_orders=int(len(T)),
                               scored_legs=int(T.n_legs.sum()), scored_shares=round(float(T.legs_sh.sum()), 6), gt_markets=len(gt),
                               lifecycle_markets=int(orders.mid.nunique()), own_fill_markets_without_lifecycle=sorted(int(m) for m in set(T.mid) - set(orders.mid)),
                               presence_orders=int(len(pres)), boundary_censored=excl),
               fills=fills_block, scenarios={})
    for sc in [s for s in C.SCENARIOS if s in set(orders.scenario)]:
        O = orders[orders.scenario == sc]; own = O[O.mk_hash.isin(T.index)]; t = T.loc[own.mk_hash]; R = {}
        keys = list(zip(own.mid, own.mk_hash))
        cov = T.index.isin(own.mk_hash)
        R['coverage'] = dict(orders=dict(n=int(len(T)), covered=int(cov.sum())), executions=dict(n=int(T.n_legs.sum()), covered=int(T.n_legs[cov].sum())),
                             shares=dict(n=round(float(T.legs_sh.sum()), 6), covered=round(float(T.legs_sh[cov].sum()), 6)))
        for v in R['coverage'].values():
            v['rate'] = round(v['covered'] / v['n'], 6) if v['n'] else None
        # identity (observed)
        idb = {'signer': own.mk_signer.astype(str).str.lower().to_numpy() != OUR, 'mid': own.mid.to_numpy() != t.mid.to_numpy(), 'y_side': own.y_side.to_numpy() != t.y_side.to_numpy(),
               'y_ptick': own.y_ptick.to_numpy() != t.y_ptick.to_numpy(), 'n_fills': own.n_fills.to_numpy() != t.n_legs.to_numpy(),
               'cum_shares': np.abs(own.cum_shares.to_numpy() - t.legs_sh.to_numpy()) > TOL_SH}
        for k, b in idb.items():
            note(sc, 'identity', [keys[i] for i in np.flatnonzero(b)], [C.OBSERVED] * int(b.sum()), [f'{k} differs from truth'] * int(b.sum()),
                 field=f'identity.{k}')
        unk = O[(O.mk_signer.astype(str).str.lower() == OUR) & ~O.mk_hash.isin(own_all)]
        note(sc, 'identity', list(zip(unk.mid, unk.mk_hash)), [C.OBSERVED] * len(unk), ['own signer, hash unknown to truth'] * len(unk),
             field='identity.own_signer_hash_unknown')
        R['identity'] = dict(n=int(len(own)), **{k: int(b.sum()) for k, b in idb.items()}, own_signer_hash_not_in_truth=int(len(unk)),
                             own_hash_excluded_from_truth_scoring=int(O.mk_hash.isin(own_all - set(T.index)).sum()),
                             defects=int(sum(int(b.sum()) for b in idb.values())) + int(len(unk)))
        # posting
        rlo, rhi = _rint(own.post_lo_ms, own.post_hi_ms); tlo, thi = _tint(t.act_lo_venue_ms, t.act_hi_venue_ms)
        po, pg = _rel(tlo, thi, rlo, rhi); po = np.where(np.isinf(rlo) & np.isinf(rhi), 'no_claim', po)
        R['posting'] = _by_class(own.post_class, po, po == 'disjoint', pg)
        R['posting_method'] = own.post_method.value_counts().to_dict()
        # size
        lbv = t.size_sh.to_numpy() < own.size_lb.to_numpy() - TOL_SH
        ex_st = own.size_exact.notna().to_numpy(); ex_eq = np.abs(own.size_exact.to_numpy() - t.size_sh.to_numpy()) <= TOL_SH
        so = np.where(~ex_st, 'not_stated', np.where(ex_eq, 'equal', 'unequal'))
        ub = own.x_size_ub.to_numpy(dtype=float); ub_st = np.isfinite(ub)
        ubv = ub_st & (t.size_sh.to_numpy() > ub + TOL_SH)
        uo = np.where(~ub_st, 'not_stated', np.where(ubv, 'violated', 'holds'))
        R['size'] = dict(lb=_by_class([C.OBSERVED] * len(own), np.where(lbv, 'violated', 'holds'), lbv),
                         exact=_by_class(own.size_class, so, so == 'unequal'), ub=_by_class(own.size_class, uo, ubv),
                         exact_stated_on_partial_truth=int((ex_st & (t.final_state.to_numpy() != 'filled')).sum()))
        # removal
        S = rs[own.index]; cause = t.true_removal.to_numpy()
        member = np.array([c in s for s, c in zip(S, cause)], dtype=bool); uninf = np.array([bool(s & UNINFORMATIVE) for s in S], dtype=bool)
        mo = np.select([member, uninf], ['member', 'uninformative'], 'contradicted')
        only_ex = np.array([s == frozenset({'exhausted'}) for s in S], dtype=bool); tf = t.final_state.to_numpy() == 'filled'
        eo = np.where(~only_ex, 'not_exhausted_only', np.where(tf, 'truly_filled', 'not_filled'))
        rlo2, rhi2 = _rint(own.removal_lo_ms, own.removal_hi_ms); tlo2, thi2 = _tint(t.rm_lo_venue_ms, t.rm_hi_venue_ms)
        wo, wg = _rel(tlo2, thi2, rlo2, rhi2); wo = np.where(np.isinf(rlo2) & np.isinf(rhi2), 'no_claim', wo)
        svc = collections.Counter(('|'.join(sorted(s)), c) for s, c in zip(S, cause)); sv = {}
        for (s, c), n in sorted(svc.items()):
            sv.setdefault(s, {})[c] = n
        R['removal'] = dict(exhausted_only=_by_class(own.removal_class, eo, eo == 'not_filled'), membership=_by_class(own.removal_class, mo, mo == 'contradicted'),
                            window=_by_class(own.removal_class, wo, wo == 'disjoint', wg), set_vs_true_cause=sv)
        # persistence: proven presence must lie inside the possible life [act_lo, rm_hi]
        plo, phi = _rint(own.persist_lo_ms, own.persist_hi_ms); llo, lhi = _tint(t.act_lo_venue_ms, t.rm_hi_venue_ms)
        claim = np.isfinite(plo) & np.isfinite(phi) & (plo <= phi)
        out = claim & ((plo < llo) | (phi > lhi))
        pgap = np.where(plo < llo, llo - plo, np.where(phi > lhi, phi - lhi, 0.))
        R['persistence'] = _by_class(own.persist_class, np.select([~claim, out], ['no_claim', 'outside_possible_life'], 'within'), out, pgap)
        # queue: front claims (ahead_hi == 0) vs own orders certainly resting at the reference frame
        fr = O[O.ahead_hi.notna() & (O.ahead_hi.abs() <= TOL_SH)]
        fo_own = fr[fr.mk_hash.isin(own_all)]; fo_oth = fr[~fr.mk_hash.isin(own_all)]
        qo1 = evidence(fo_own, fo_own.mk_hash.map(hcid).fillna(-1).to_numpy()); qo2 = evidence(fo_oth, np.full(len(fo_oth), -1))
        R['queue'] = dict(own_front_claims=_by_class(fo_own.queue_class, qo1, qo1 == 'evidence_against'),
                          other_makers_front_claims=_by_class(fo_oth.queue_class, qo2, qo2 == 'evidence_against'),
                          front_claims_by_status={f'{s}|{q}': int(n) for (s, q), n in collections.Counter(
                              zip(list(fo_own.status) + list(fo_oth.status), list(qo1) + list(qo2))).items()},
                          ahead_hi_negative=int((O.ahead_hi < -TOL_SH).sum()))
        # defect rows
        for cat, cls, failm, det in (
                ('posting', own.post_class, po == 'disjoint', [f'activation hull misses the window by {g:.1f} ms' for g in pg]),
                ('size_lb', [C.OBSERVED] * len(own), lbv, [f'true size {a} < size_lb {b}' for a, b in zip(t.size_sh, own.size_lb)]),
                ('size_exact', own.size_class, so == 'unequal', [f'true size {a} != size_exact {b}' for a, b in zip(t.size_sh, own.size_exact)]),
                ('size_ub', own.size_class, ubv, [f'true size {a} > size upper bound {b}' for a, b in zip(t.size_sh, ub)]),
                ('removal_exhausted', own.removal_class, eo == 'not_filled', [f'exhausted-only, truly {f}' for f in t.final_state]),
                ('removal_membership', own.removal_class, mo == 'contradicted', [f'true cause {c} not in the set' for c in cause]),
                ('removal_window', own.removal_class, wo == 'disjoint', [f'removal window misses the truth by {g:.1f} ms' for g in wg]),
                ('persistence', own.persist_class, out, [f'proven presence outside the possible life by {g:.1f} ms' for g in pgap])):
            cl = np.asarray(cls, dtype=object); sel = np.asarray(failm, dtype=bool) & np.isin(cl, [C.OBSERVED, C.CONSTRAINED])
            note(sc, cat, [keys[i] for i in np.flatnonzero(sel)], cl[sel].tolist(), [det[i] for i in np.flatnonzero(sel)])
        for Q, q, lab in ((fo_own, qo1, 'queue_own'), (fo_oth, qo2, 'queue_other')):
            cl = Q.queue_class.to_numpy(dtype=object); sel = (q == 'evidence_against') & np.isin(cl, [C.OBSERVED, C.CONSTRAINED])
            note(sc, lab, list(zip(Q.mid[sel], Q.mk_hash[sel])), cl[sel].tolist(), ['front claim vs an own order certainly resting there'] * int(sel.sum()))
        R['own_order_status'] = own.status.value_counts().to_dict()
        R['defects_total'] = (R['identity']['defects'] + _defects(R['posting']) + _defects(R['size']['lb']) + _defects(R['size']['exact'])
                              + _defects(R['size']['ub']) + _defects(R['removal']['exhausted_only']) + _defects(R['removal']['membership'])
                              + _defects(R['removal']['window']) + _defects(R['persistence']) + _defects(R['queue']['own_front_claims'])
                              + _defects(R['queue']['other_makers_front_claims']))
        R['sensitive_misses_total'] = sum(sum(v['sensitive_misses'] for v in b.values()) for b in (
            R['posting'], R['size']['exact'], R['size']['ub'], R['removal']['exhausted_only'], R['removal']['membership'], R['removal']['window'],
            R['persistence'], R['queue']['own_front_claims'], R['queue']['other_makers_front_claims']))
        res['scenarios'][sc] = R
    res['defect_rows'] = defects
    res['defects_total'] = dict(fills=fills_block['defects'], **{s: v['defects_total'] for s, v in res['scenarios'].items()})
    refuse(len(defects) == sum(res['defects_total'].values()), f'defect rows {len(defects)} != defect counts {res["defects_total"]}')
    u = {(d['mid'], d['mk_hash'], d['field']) for d in defects}
    res['unique_defects'] = dict(n=len(u), by_field=dict(sorted(collections.Counter(f for _, _, f in u).items())),
                                 note='distinct (order, field) pairs failing in at least one scenario; fills fields are scenario-independent')
    return res


# ------------------------------------------------------------------ CLI ------------------------------------------------------------------
def gt_markets():
    sc = json.loads((EXP / 'scope.json').read_text())
    return sorted(int(w['market']) for w in sc['windows'])


def dry_gate(T, rec, target_dir):
    """Rehearse the writer into a temporary directory (read back, registry-checked, equal), then REFUSE unless the written target table
    exists and this code reproduces it exactly (content and registry; receipt times excluded). The integration gate before any score."""
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        write_truth(T, rec, Path(d)); TT = read_truth(Path(d) / 'own_orders.parquet'); _registry_check(TT)
        refuse(TT.equals(T), 'dry run: the read-back truth differs from the built one')
    p = Path(target_dir) / 'own_orders.parquet'
    refuse(p.exists(), f'dry gate: the target truth table {p} is missing (build it first)')
    W0 = read_truth(p); m = _meta(rec)
    same, reg_same = bool(W0.equals(T)), all(W0.attrs['mt_truth'].get(k) == m[k] for k in ('clock_band', 'columns'))
    print('dry: reproduces the written truth table:', same, '| written registry identical:', reg_same)
    refuse(same and reg_same, 'dry gate: this code does not reproduce the written truth table (integration gate; rebuild needed)')
    return True


def gt_lifecycle_inputs(out, day):
    """B2: the GT windows scoring must cover = extract_<day>.json ws_population.selected, less those whose markets_<day> status carries
    boundary_censored (excluded from scoring, counted). REFUSES unless every remaining window has the lifecycle's orders, fills and JSON."""
    C.need(day == C.NOV8, 'GT day is Nov 8')
    out = Path(out); ex, mk = out / f'extract_{day}.json', out / f'markets_{day}.parquet'
    refuse(ex.exists() and mk.exists(), f'GT extract receipt or markets table missing ({ex.name}, {mk.name})')
    sel = (json.loads(ex.read_text()).get('ws_population') or {}).get('selected')
    refuse(isinstance(sel, list) and len(sel) > 0, f'{ex.name} has no ws_population.selected list')
    sel = sorted(int(m) for m in sel)
    M = pd.read_parquet(mk, columns=['mid', 'status']); st = dict(zip(M.mid.astype('int64'), M.status.astype(str)))
    refuse(all(m in st for m in sel), f'a selected GT window has no {mk.name} row')
    cen = [m for m in sel if 'boundary_censored' in st[m].split('+')]
    req = [m for m in sel if m not in set(cen)]
    life = out / 'life'
    miss = [(m, k) for m in req for k in (f'{day}_{m}_orders.parquet', f'{day}_{m}_fills.parquet', f'{day}_{m}.json') if not (life / k).exists()]
    refuse(not miss, f'incomplete lifecycle for the selected GT windows: {len(miss)} files missing over {len({m for m, _ in miss})} windows (e.g. {miss[:3]})')
    return dict(selected=sel, censored=cen, required=req, life=life)


def main_build(dry=False):
    src = load_export()
    T, rec = build_truth(src); guard('build')
    if dry:
        dry_gate(T, rec, TRUTH_DIR)
    else:
        write_truth(T, rec, TRUTH_DIR)
    print(json.dumps({k: rec[k] for k in ('clock_band', 'signer', 'own_public', 'rows', 'status', 'final_state', 'link_status', 'checks', 'scored',
                                          'presence_rows', 'link_evidence', 'gt_markets')}, indent=1, default=str))
    print('VmHWM_MiB', round(vmhwm_mib(), 1))
    return 0


def main_score():
    day = C.GT_DAY; C.need(day == C.NOV8, 'GT day is Nov 8')
    tp = TRUTH_DIR / 'own_orders.parquet'; refuse(tp.exists(), f'truth table {tp} missing: run build (then the dry gate)')
    T = read_truth(tp); mids = gt_markets()
    G = gt_lifecycle_inputs(C.OUT, day); LIFE, have = G['life'], G['required']
    O = pd.concat([pd.read_parquet(LIFE / f'{day}_{m}_orders.parquet') for m in have], ignore_index=True)
    F = pd.concat([pd.read_parquet(LIFE / f'{day}_{m}_fills.parquet') for m in have], ignore_index=True); guard('lifecycle')
    legs = C.OUT / f'legs_{day}.parquet'
    if legs.exists():
        pub, pub_src = set(pd.read_parquet(legs, columns=['mk_hash']).mk_hash), f'legs_{day}.parquet (pf_mt_extract)'
    else:
        pub, pub_src = set(pd.read_parquet(LV / 'matches_20261108.parquet', columns=['maker_hash']).maker_hash), 'export matches_20261108.parquet'
    res = score(O, F, T, pub, mids, censored_mids=G['censored']); guard('score')
    res['inputs'] = dict(code=code_sha(), truth=sha256(tp), truth_version='v2 (one clock band per run)', public_hashes=pub_src,
                         gt_selection=dict(selected=len(G['selected']), boundary_censored=G['censored'], scored_windows=len(have)),
                         lifecycle={m: [sha256(LIFE / k) for k in (f'{day}_{m}_orders.parquet', f'{day}_{m}_fills.parquet', f'{day}_{m}.json')] for m in have})
    _write_once(TRUTH_DIR / 'scores.json', lambda p: p.write_text(json.dumps(res, indent=1, default=str)))
    print(json.dumps({k: res[k] for k in ('population', 'defects_total', 'unique_defects')}, indent=1, default=str))
    print('VmHWM_MiB', round(vmhwm_mib(), 1))
    return 0


if __name__ == '__main__':
    sys.exit({'build': main_build, 'dry': lambda: main_build(dry=True), 'score': main_score}[sys.argv[1]]())
