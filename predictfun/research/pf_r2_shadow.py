"""R2 PREPARATION (plan v2 stage R2; review c68a072c's contract): THE SHADOW-PROBE / FEATURE / SUPPORT BUILDER AND ITS DETERMINISTIC MANIFEST
r2_shadow_sampling_v1 (P-2026-005; 2026-11-19, session aea5bfb9). USER 2026-11-19: "... continue the implementing and next priority loop"; review
c68a072c: "R2's first reviewable deliverable should be a versioned shadow-probe/feature/support builder and its deterministic manifest, verified
on synthetic fixtures and a fixed consumed Oct23 slice, before fitting PI0-50." Nothing is fitted or read from a model here.

POPULATION (fixed before any outcome): Oct 23 (consumed; the teacher day), the first 128 original opportunities per side by oid of the committed
 keep / pull run, of any status (no-path statuses stay in the census as rows).
MANIFEST r2_shadow_sampling_v1: phase(day, instrument, episode, side, head) = the first 8 bytes of SHA-256 of the UTF-8 key
 'r2_shadow_sampling_v1|day|instrument_id|original_episode_id|side|head', unsigned big-endian, mod 5 (instrument_id = the market id m; episode =
 the original opportunity's oid; head post / retain); never Python's hash(). Entry offsets: 0 and 50 k for k = 1..99 with k mod 5 = phase(post)
 (k = 99, 4,950 ms: status near_end, activation after END). Retained rows: every live client decision of a probe is a census row; it is
 SAMPLED when its episode-grid tick k has k mod 5 = phase(retain) or it is the probe's first eligible live review. Joint rows (below): absent
 states sampled with phase(post), live states with phase(retain).
SOURCES (labelled separately): entry (neutral entry shadows: one shadow order per offset, each an alternative world), retain (their live
 decisions), joint_post_keep / joint_chase (pf_branch_engine's fixed stand-ins on both sides; at a sampled feasible absent or live state of a
 side, the focal side's neutral continuation from that exact state - a fork).
NEUTRAL TARGET (J's single-order moments; never a paired quantity): the focal generation alone, held under the frozen client structural rule to
 the ORIGINAL END, never replaced: POST - its fills from activation; RETAIN - its fills at or after the decision + 37 ms; each fill at its own
 price, marked at END (pf_pipeline_export.affected, unchanged); q, value, qm = value - s (ref - p) q with ref = pf_mid at the decision (the
 export's identity), severe excess max(-value - 30, 0). Zero / no-fill / post-only refusal outcomes kept (a refusal is an outcome, not a status).
INPUTS (reused unchanged; the export's 91 in its order): the 71 market features (pf_pipeline_export.real_features at the decision's window grid
 index of its side; a decision off the window grid -> off_grid, a nonfinite pf_mid -> feature_unavailable), the 12 order inputs
 (pf_pipeline_export.known_steps -> FC.V.order_inputs with the generation's own submission, activation, price, fill history and the ORIGINAL
 terminal), the 6 fp_ extras and OWN (fair_edge = s (fair - p) from the keep / pull run's fair path at the episode step; t_to_end = the original
 END - decision); plus review c68a072c's 16 mandatory context fields from the branch engine's CLIENT state (acknowledged / reserved; never
 Engine.rem): ctx_has_generation, ctx_generation_age_ms, ctx_generation_ordinal, ctx_inventory_shares, ctx_bid_reserved_shares,
 ctx_ask_reserved_shares, ctx_self_state_{absent, post_pending, live, cancel_pending, terminal}, ctx_other_state_{same five}.
COSTS: the gross research convention, recorded as explicit zeros before any teacher fit (fee_bps 0, rebate_bps 0; unmodelled).
STATUSES (never dropped): no_path (the keep / pull run's own statuses), missing_mark, off_grid, post_pending, cancel_pending, cap_blocked,
 near_end, feature_unavailable, unsampled (a census row outside the manifest), ok.
SUPPORT (declared before the census is read; engineering floors, not confidence): per side x head x remaining-horizon bin ((0,100], (100,250],
 (250,500], (500,1000], (1000,2000], (2000,3500], (3500,5000] ms): sampled ok rows, distinct original episodes, episodes with a nonzero quantity;
 SUPPORTED iff >= 200 episodes and >= 20 nonzero episodes, else INSUFFICIENT_TRAINING_SUPPORT; category presence per side x head (remaining
 shares 0 / (0,10] / (10,20] / (20,30]; queue unknown / 0 / (0,30] / (30,150] / >150; generation 1 / 2 / >=3; inventory sign; other-side state;
 price [1,5] / (5,20] / (20,80] / (80,95] / (95,99]); the observed ranges of the numeric context.
CONTROLS (selftest, synthetic; each a valid case and a known-bad one): M1 the manifest phase against an independent SHA-256 computation, its
 range and stability (no hash()); later-birth (an offset probe joins the book known at its own decision), partial-fill and hidden-notice context
 (a fill not yet notified changes no input or context field, a notified one does), the queue inputs (the client's min-updated shown size),
 short horizon (t_to_end) and near_end; causality of the full row (inputs + context) under future frames / trades / receipt times; per-fill
 label reconciliation by an independent loop; the census complete (every candidate row once, statuses never dropped). Mutations:
 pf_r2_shadow_mutants.py.
Usage (run_light, PF_SLOT=claude): pf_r2_shadow.py selftest | census <day>
"""
import collections
import contextlib
import copy
import hashlib
import io
import json
import math
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pf_branch_engine as BE                             # noqa: E402
import pf_keep_pull as KP                                 # noqa: E402
import pf_pipeline_export as X                            # noqa: E402
import pf_queue_place_test as QP                          # noqa: E402

C, pd, I = KP.C, KP.pd, KP.I
FC = X.FC
MANIFEST = 'r2_shadow_sampling_v1'
OUT = Path(I.K.IO) / 'r2_shadow_20261119'
MARK = 'R2 SHADOW'
DAY = '20261023'                                           # the teacher day (consumed)
PER_SIDE = 128
STEP, END, J0 = KP.STEP, KP.H, KP.J0
COSTS = dict(convention='gross research value at the END mark', fee_bps=0., rebate_bps=0., unmodelled=['fees', 'rebates', 'latency costs'])
CTX = (['ctx_has_generation', 'ctx_generation_age_ms', 'ctx_generation_ordinal', 'ctx_inventory_shares', 'ctx_bid_reserved_shares', 'ctx_ask_reserved_shares']
       + [f'ctx_self_state_{x}' for x in ('absent', 'post_pending', 'live', 'cancel_pending', 'terminal')]
       + [f'ctx_other_state_{x}' for x in ('absent', 'post_pending', 'live', 'cancel_pending', 'terminal')])
INPUTS = list(X.INPUTS)                                    # the export's 91, in its order
HORIZON_BINS = (0., 100., 250., 500., 1000., 2000., 3500., 5000.)
FLOORS = dict(episodes=200, nonzero_episodes=20)
NEAR_END_K = 99                                            # the offset 4,950 ms: activation after END
CONTEXT_FROM_ENGINE = False                                # planted-error switch (selftest only): context from Engine.rem, not the client
CONTEXT_FUTURE = False                                     # planted-error switch (selftest only): the context reads the whole tape (a future leak)
LABEL_NUDGE = 0.                                           # planted-error switch (selftest only): added to every target quantity
ADAPT_CORRUPT = ''                                         # planted-error switch (selftest only): a stored field corrupted after the builders
PART_SIZE = 64                                             # review d10d3698: partitions of at most 64 original opportunities
DROP_ORIGINAL = None                                       # planted-error switch (selftest only): one original's rows never emitted
DUP_ORIGINAL = False                                       # planted-error switch (selftest only): a previous partition's original emitted again
FAIR_LEAK = False                                          # planted-error switch (selftest only): BOTH builders read the next step's archived fair
INV_TOL = 1e-9                                             # the inventory-sign tolerance (round-off is not a long / short category)
VENUE_DRAWS = 8                                            # deterministic seeds tried for a check's future-only world (a draw can leave no valid END mark)
VENUE_NO_PATH = 0                                          # planted-error switch (selftest only): 1 no future world builds, 2 the first draw of each check fails
CAUSAL_VERSION = 'r2_full_causal_v1'                       # review 4cd7b45f: the declared audit sample (causal_select)
CAUSAL_DIV = 64                                            # min(n, max(1, ceil(n / 64))) rows per original x side x source x head stratum
MAX_MATERIALIZE_ROWS = 300000                              # materialize(): a test / slice helper, refused above this (the full day never materializes)
CONTEXT_UNRECEIVED = ''                                    # planted-error switch (selftest only): 'frame' reads the venue frame at g (received or not),
                                                           # 'ahead' the open generation's engine-truth queue ahead
DUP_STATUS = False                                         # planted-error switch (selftest only): a status row emitted twice in one partition
DUP_EPISODE = False                                        # planted-error switch (selftest only): an episode row emitted twice in one partition


def phase(day, instrument, episode, side, head):
    """The manifest phase: the first 8 bytes of SHA-256 of the canonical UTF-8 key, unsigned big-endian, mod 5."""
    C.require(head in ('post', 'retain'), 'undeclared head')
    key = f'{MANIFEST}|{day}|{int(instrument)}|{int(episode)}|{int(side)}|{head}'.encode('utf-8')
    return int.from_bytes(hashlib.sha256(key).digest()[:8], 'big') % 5


def offsets(ph):
    """Entry offsets (ms): 0 and 50 k for k = 1..99 with k mod 5 = ph; k = 99 is kept as the near_end status."""
    return [0.] + [STEP * k for k in range(1, NEAR_END_K + 1) if k % 5 == ph]


def episode_ident(day, o):
    """Review efac0452 amendment 1: the identity bound into every branch of an original episode (label / join metadata, never an input)."""
    return dict(day=str(day), instrument_id=int(o.m), original_episode_id=int(o.oid), side=int(o.side), start_abs_ms=float(o.t), end_abs_ms=float(o.t) + END,
                mark_source='QP.build_path END mark of the M50 tape (the keep / pull run)')


def client_snapshot(B, g, s):
    """The 16 mandatory context fields of side s at g from the branch's CLIENT state (after pre_tick): acknowledged inventory, reservations,
    states, the generation's age and ordinal."""
    Ik, Bk, Ak = B.reserve(g); st, ot = B.status(s, g), B.status(1 - s, g); gid = B.open[s]
    n_side = sum(1 for o in B.E.o if o['side'] == s); o = B.E.o[gid] if gid is not None else None
    if CONTEXT_FROM_ENGINE and o is not None:                                               # planted: engine truth leaks into the context
        Ik = sum((1. if f_[2] == 0 else -1.) * f_[3] for f_ in B.E.fills if f_[0] <= g)
    if CONTEXT_FUTURE:                                                                      # planted: the whole tape's trade count leaks in
        Ik += 1e-3 * len(B.E.tr)
    if CONTEXT_UNRECEIVED == 'frame':                                                       # planted: the venue frame at g read directly (received or not)
        jv_ = int(np.searchsorted(B.E.f[:, 0], g, side='right') - 1); Ik += .01 * float(B.E.f[max(jv_, 0), 2])
    elif CONTEXT_UNRECEIVED == 'ahead' and o is not None:                                   # planted: the engine-truth queue ahead
        Ik += .01 * float(o['ahead'])
    x = dict(ctx_has_generation=float(gid is not None), ctx_generation_age_ms=float(g - o['submit']) if o is not None else 0.,
             ctx_generation_ordinal=float(n_side if gid is not None else n_side + 1), ctx_inventory_shares=float(Ik), ctx_bid_reserved_shares=float(Bk),
             ctx_ask_reserved_shares=float(Ak))
    for nm, v in (('self', st), ('other', ot)):
        for z in ('absent', 'post_pending', 'live', 'cancel_pending', 'terminal'):
            x[f'ctx_{nm}_state_{z}'] = float(v == z)
    return x, st, gid


class Single(BE.Policy):
    """The neutral world of one probe: POST once on the focal side at tick `k_post` (when absent), then hold (KEEP); never replaced; the other
    side never posts."""

    def __init__(self, side, k_post):
        self.side, self.k_post, self.posted = side, k_post, False
        super().__init__(self.fn, f'single_{side}_{k_post}')

    def fn(self, v):
        if v['side'] != self.side:
            return 'WAIT' if v['state'] == 'absent' else 'KEEP'
        if v['state'] == 'absent':
            if v['t'] == self.k_post * STEP and not self.posted:
                self.posted = True; return 'POST'
            return 'WAIT'
        return 'KEEP'


def hold_from(B, k, side):
    """The focal side's neutral continuation from the exact state of branch B at tick k (after pre_tick, before side decides): POST now when
    absent, KEEP otherwise; never replaced; the other side holds what it has and never posts. Returns (the focal gid, the finished ledger)."""
    Bx = BE.copy.deepcopy(B); g = k * STEP; v = Bx.view(g, side); C.require(side not in Bx.acted, 'hold_from: side already processed')
    pol = BE.Policy(lambda v_: ('WAIT' if v_['state'] == 'absent' else 'KEEP'), 'hold')
    BE.decide(Bx, g, side, pol, forced='POST' if v['state'] == 'absent' else 'KEEP')
    if (1 - side) not in Bx.acted:
        BE.decide(Bx, g, 1 - side, pol)
    gid = Bx.open[side]; nt = int(np.ceil(Bx.end / STEP)); gen = dict(Bx.E.o[gid]) if gid is not None else None
    for kk in range(k + 1, nt):
        BE.step(Bx, kk, pol)
    return gid, BE.finish(Bx), gen


def target(fills, gid, side, mark, eff, ref):
    """J's single-order neutral moments of generation gid: fills at or after eff (episode ms), each at its own price, marked at END
    (X.affected, unchanged) -> q, value, qm (vs ref), severe excess."""
    f = [x for x in fills if x[1] == gid]; sg = 1. if side == 0 else -1.
    ft = np.array([x[0] for x in f], float); fq = np.array([x[3] for x in f], float); fp = np.array([x[4] for x in f], float)
    val = fq * sg * (mark - fp); q_, v_ = X.affected(ft, fq, val, np.array([float(eff)]))
    q_, v_ = float(q_[0]) + LABEL_NUDGE, float(v_[0]); pq = float(fp[0]) if len(fp) else np.nan    # one generation: one price
    qm = v_ - (sg * (ref - pq) * q_ if q_ > 0 else 0.)
    return dict(q=q_, value=v_, qm=qm, severe=max(-v_ - KP.SEVERE, 0.))


def probes(path, o, day, src_rows):
    """Every candidate row of one original episode (identities, decision, the generation, its client context, the neutral target) for the
    entry / retain / joint sources; features and order inputs are added later per window."""
    s, oid, m = int(o.side), int(o.oid), int(o.m); ph_p, ph_r = phase(day, m, oid, s, 'post'), phase(day, m, oid, s, 'retain')
    base = dict(day=day, instrument_id=m, original_episode_id=oid, side=s, t0=float(o.t), phase_post=ph_p, phase_retain=ph_r)
    ident = episode_ident(day, o); mark = BE.canon_mark(path.get('mark')); base['end_mark'] = mark          # a label column, never an input
    base.update(end_abs_ms=ident['end_abs_ms'], mark_source=ident['mark_source'])
    for off in offsets(ph_p):                                                               # entry shadows and their retained decisions
        k_post = int(round(off / STEP)); r = dict(base, source='entry', head='post', offset_ms=off, decision_ms=off, generation=1)
        if k_post >= NEAR_END_K:
            src_rows.append(dict(r, status='near_end', sampled=True)); continue
        pol = Single(s, k_post); B = BE.Branch(path, ident=ident); nt = int(np.ceil(END / STEP)); snaps = {}; v_post = None
        for k in range(nt):
            g = k * STEP; BE.pre_tick(B, g); B.k = k
            ctx, st, gid = client_snapshot(B, g, s); snaps[k] = (ctx, st, gid, dict(B.E.o[gid]) if gid is not None else None)
            if k == k_post:
                v_post = B.view(g, s)
            for side_ in ((B.episode + k) % 2, 1 - (B.episode + k) % 2):
                BE.decide(B, g, side_, pol)
            B.check_position(g)
        L = BE.finish(B)
        if L['status'] != 'ok':
            src_rows.append(dict(r, status='missing_mark', sampled=True)); continue
        ctx0, st0, _, _ = snaps[k_post]; gen = [x for x in B.E.o if x['side'] == s]
        if not gen:                                                                         # the POST was infeasible at the decision (book / cap)
            src_rows.append(dict(r, status='feature_unavailable' if not BE.book_ok(v_post, s) else 'cap_blocked', sampled=True, **ctx0)); continue
        g0 = gen[0]; gid0 = g0['gid']
        src_rows.append(dict(r, status='ok', sampled=True, price=g0['price'], activation_ms=g0['act'], refused=bool(g0['refused']),
                             **ctx0, **target(L['fills'], gid0, s, mark, g0['act'], np.nan), effective_ms=g0['act'], gid=gid0, fills=_fl(L['fills'], gid0)))
        first_live = next((k for k in range(k_post + 1, nt) if snaps[k][1] == 'live' and snaps[k][2] == gid0), None)
        for k in range(k_post + 1, nt):                                                     # the probe's live decisions (census) and the sampled ones
            ctx, st, gid, og = snaps[k]
            if gid != gid0 or st not in ('live', 'cancel_pending', 'post_pending'):
                continue
            g = k * STEP; rr = dict(base, source='retain', head='retain', offset_ms=off, decision_ms=g, generation=1, gid=gid0, price=g0['price'], activation_ms=g0['act'])
            if st != 'live':
                src_rows.append(dict(rr, status=st, sampled=False, **ctx)); continue
            smp = (k % 5 == ph_r) or (k == first_live)
            if not smp:
                src_rows.append(dict(rr, status='unsampled', sampled=False, **ctx)); continue
            src_rows.append(dict(rr, status='ok', sampled=True, first_live=bool(k == first_live), **ctx, **target(L['fills'], gid0, s, mark, g + KP.CANCEL, np.nan),
                                 effective_ms=g + KP.CANCEL, fills=_fl(L['fills'], gid0)))
    for pname, pol in (('joint_post_keep', BE.POST_KEEP), ('joint_chase', BE.CHASE)):     # joint states: forks to the focal side's neutral hold
        B = BE.Branch(path, ident=ident); nt = int(np.ceil(END / STEP))
        for k in range(nt):
            g = k * STEP; BE.pre_tick(B, g); B.k = k; order = ((B.episode + k) % 2, 1 - (B.episode + k) % 2)
            for side_ in order:
                if side_ == s:
                    ctx, st, gid = client_snapshot(B, g, s); head = 'post' if st == 'absent' else 'retain'
                    rj = dict(base, source=pname, head=head, offset_ms=np.nan, decision_ms=g, generation=ctx['ctx_generation_ordinal'])
                    if st in ('absent', 'live'):
                        v = B.view(g, s); smp = k % 5 == (ph_p if head == 'post' else ph_r)
                        feas = (st == 'live') or (BE.book_ok(v, s) and v['room'] >= -1e-9 and g + KP.PLACE < END)
                        why = None if feas else ('near_end' if g + KP.PLACE >= END else 'cap_blocked' if v['room'] < -1e-9 else 'feature_unavailable')
                        if not smp or not feas:
                            src_rows.append(dict(rj, status=why or 'unsampled', sampled=False, **ctx))
                        else:
                            gid_f, Lf, gen_f = hold_from(B, k, s)
                            if Lf['status'] != 'ok' or gid_f is None:
                                src_rows.append(dict(rj, status='missing_mark' if Lf['status'] != 'ok' else 'cap_blocked', sampled=True, **ctx))
                            else:
                                pr, act = gen_f['price'], gen_f['act']; ref_kw = {'refused': bool(gen_f['refused'])} if head == 'post' else {}   # a POST outcome only
                                eff = act if head == 'post' else g + KP.CANCEL
                                src_rows.append(dict(rj, status='ok', sampled=True, gid=gid_f, price=pr, activation_ms=act, **ref_kw, **ctx,
                                                     **target(Lf['fills'], gid_f, s, mark, eff, np.nan), effective_ms=eff, fills=_fl(Lf['fills'], gid_f)))
                    else:
                        src_rows.append(dict(rj, status=st, sampled=False, **ctx))
                BE.decide(B, g, side_, pol)
            B.check_position(g)
    return src_rows


def _fl(fills, gid):
    return json.dumps([[float(x[0]), float(x[3]), float(x[4])] for x in fills if x[1] == gid])


def add_inputs(rows, W, o, fair_m, depth, feat_fn, ref_fair):
    """The 91 inputs for the sampled ok rows of one episode: market features at the decision's window grid index (off_grid / feature_unavailable
    statuses as the export), the order inputs (known_steps -> order_inputs with the generation's own clock and the ORIGINAL terminal), fp_
    extras, fair_edge and t_to_end; qm against ref = pf_mid at the decision."""
    s, t0 = int(o.side), float(o.t); w = W[s]; tg = np.asarray(w['tg'], float); sg = 1. if s == 0 else -1.
    want = [r for r in rows if r.get('sampled') and r.get('status') == 'ok']
    ks = {}
    for r in want:
        d = t0 + r['decision_ms']; kk = int(np.searchsorted(tg, d)); kk = min(kk, len(tg) - 1)
        ks[id(r)] = kk if abs(tg[kk] - d) <= 1e-6 else None
    chosen = {0: [], 1: []}; chosen[s] = sorted({k for k in ks.values() if k is not None})
    F = feat_fn(W, fair_m, depth, chosen) if chosen[s] else {s: {}}
    for r in want:
        kk = ks[id(r)]
        if kk is None:
            r['status'] = 'off_grid'; continue
        xm, scale = F[s][kk]; ref = float(xm[FC.MARKET.index('pf_mid')])
        if not np.isfinite(ref):
            r['status'] = 'feature_unavailable'; continue
        d = t0 + r['decision_ms']; ta = t0 + r['activation_ms']; fl = json.loads(r['fills'])
        kn = X.known_steps(w, s, ta, r['price'], (np.array([t0 + a for a, _, _ in fl], float), np.array([b for _, b, _ in fl], float)), np.array([d]), t0 + END)
        oi = FC.V.order_inputs(pd.DataFrame({f'x_{a}': kn[a] for a in X.KNOWN}), np.array([scale]), np.array([ref]))
        r.update({nm: float(xm[i_]) for i_, nm in enumerate(FC.MARKET)}); r.update({nm: float(np.asarray(oi[nm], float)[0]) for nm in FC.ORDER})
        k_ep = int(round(r['decision_ms'] / STEP)); fair = ref_fair(k_ep + int(FAIR_LEAK))
        r.update(fp_size=float(KP.SIZE), fp_remaining=float(kn['known_remaining'][0]), fp_queue=float(kn['queue_ahead_known'][0]), fp_shown=float(kn['shown_known'][0]),
                 fp_scale=float(scale), fp_phase=float(o.phase), fair_edge=sg * (fair - r['price']), t_to_end=END - r['decision_ms'], pf_mid_ref=ref)
        r['qm'] = r['value'] - (sg * (ref - r['price']) * r['q'] if r['q'] > 0 else 0.)
        if ADAPT_CORRUPT == 'fair_edge':                                                    # planted: the fair at END
            r['fair_edge'] = sg * (ref_fair(KP.NSTEP) - r['price'])
        elif ADAPT_CORRUPT == 't_to_end':
            r['t_to_end'] = END - r['decision_ms'] + 50.
        elif ADAPT_CORRUPT == 'fp_remaining':                                               # planted: every fill, notified or not
            r['fp_remaining'] = float(KP.SIZE) - sum(b for _, b, _ in fl)
        elif ADAPT_CORRUPT == 'market':
            r[FC.MARKET[0]] = r[FC.MARKET[0]] + 1.


SORT_KEY = ['day', 'instrument_id', 'original_episode_id', 'side', 'source', 'head', 'offset_ms', 'generation', 'decision_ms']   # c68a072c: the manifest's order


def canonical(D):
    """The census in the manifest's order: stable original identity, then source / head, offset, generation and decision (missing values first)."""
    return D.sort_values([c for c in SORT_KEY if c in D], kind='stable', na_position='first').reset_index(drop=True)


def frame_of(rows):
    """Rows -> a DataFrame with the inputs and context as float32 (the benchmark's dtype; labels stay float64)."""
    F_ = pd.DataFrame(rows)
    for c in INPUTS + CTX:
        if c in F_:
            F_[c] = F_[c].astype(np.float32)
    return F_


def weights(D):
    """Equal total training weight per original episode x side x head across its sampled ok rows and source families (c68a072c): w = 1 / n."""
    ok = (D.status == 'ok') & D.sampled.astype(bool); w = np.zeros(len(D))
    if ok.any():
        n = D[ok].groupby(['original_episode_id', 'side', 'head']).original_episode_id.transform('size').values; w[ok.values] = 1. / n
    return w


def resim(path_, o, day, r):
    """Re-simulate the row's own source on path_ up to its decision (pre_tick applied; the other side decided first when the joint order puts it
    first) -> the branch at the decision."""
    g = r['decision_ms']; s = int(o.side); k_dec = int(round(g / STEP)); B = BE.Branch(path_, ident=episode_ident(day, o))
    pol = Single(s, int(round(r['offset_ms'] / STEP))) if r['source'] in ('entry', 'retain') else (BE.POST_KEEP if r['source'] == 'joint_post_keep' else BE.CHASE)
    for k in range(k_dec + 1):
        gg = k * STEP; BE.pre_tick(B, gg); B.k = k
        if k == k_dec:
            if r['source'].startswith('joint') and (B.episode + k) % 2 != s:
                BE.decide(B, gg, 1 - s, pol)
            return B
        for side_ in ((B.episode + k) % 2, 1 - (B.episode + k) % 2):
            BE.decide(B, gg, side_, pol)


def market_at(W_, fair_m, depth, feat_fn, s, kk):
    """The market features and the W30 scale at one grid index, from the builder (which reads both sides' keys)."""
    ch = {0: [], 1: []}; ch[s] = [kk]                                                        # the real builder's contract: both sides' keys
    return feat_fn(W_, fair_m, depth, ch)[s][kk]


def rebuild_row(path_, W_, fv_, o, day, r, fair_m, depth, feat_fn, fill_change=None, B=None, market=None):
    """An INDEPENDENT per-row rebuild of the 107 stored inputs (the 91 + the 16 context fields): the row's source re-simulated to its decision
    (or B, a given decision state); the generation's price / activation / fills from that state (an absent decision: the JOIN at the view,
    activation + 58 ms, no fill); the market features at the decision's grid index (or `market`, those features already built), known_steps ->
    order_inputs, the fp_ extras, fair_edge from the archived fair path at the decision step, t_to_end to the original END.
    fill_change(times, qty, decision) perturbs the generation's fill list (private notices)."""
    s, t0 = int(o.side), float(o.t); g = r['decision_ms']; d = t0 + g; sg = 1. if s == 0 else -1.
    B = resim(path_, o, day, r) if B is None else B; ctx = client_snapshot(B, g, s)[0]; gid = B.open[s]
    if gid is None:
        v = B.view(g, s); price, act, ft, fq = (v['bid'] if s == 0 else v['ask']), g + KP.PLACE, np.zeros(0), np.zeros(0)
    else:
        og = B.E.o[gid]; price, act = og['price'], og['act']
        ff = [x for x in B.E.fills if x[1] == gid]; ft = np.array([t0 + x[0] for x in ff], float); fq = np.array([x[3] for x in ff], float)
    if fill_change is not None:
        ft, fq = fill_change(ft, fq, d)
    w = W_[s]; tg = np.asarray(w['tg'], float); kk = min(int(np.searchsorted(tg, d)), len(tg) - 1)
    if abs(tg[kk] - d) > 1e-6:
        return None
    xm, scale = market if market is not None else market_at(W_, fair_m, depth, feat_fn, s, kk); ref = float(xm[FC.MARKET.index('pf_mid')])
    kn = X.known_steps(w, s, t0 + act, price, (ft, fq), np.array([d]), t0 + END)
    oi = FC.V.order_inputs(pd.DataFrame({f'x_{a}': kn[a] for a in X.KNOWN}), np.array([scale]), np.array([ref]))
    out = {nm: float(xm[i_]) for i_, nm in enumerate(FC.MARKET)}; out.update({nm: float(np.asarray(oi[nm], float)[0]) for nm in FC.ORDER})
    k_ep = int(round(g / STEP))
    out.update(fp_size=float(KP.SIZE), fp_remaining=float(kn['known_remaining'][0]), fp_queue=float(kn['queue_ahead_known'][0]), fp_shown=float(kn['shown_known'][0]),
               fp_scale=float(scale), fp_phase=float(o.phase), fair_edge=sg * (float(fv_[min(k_ep + int(FAIR_LEAK), len(fv_) - 1)]) - price), t_to_end=END - g)
    out.update(ctx)
    return out


def same_f32(stored, rebuilt, cols):
    """Declared tolerance: equality after the float32 cast (the stored dtype), NaN equal to NaN."""
    return [c for c in cols if not ((np.isnan(np.float32(stored[c])) and np.isnan(np.float32(rebuilt[c]))) or np.float32(stored[c]) == np.float32(rebuilt[c]))]


def venue_windows(W, d, rng):
    """The windows' book frames after the last frame AT THE VENUE by d (tf <= d) perturbed as the builder's perturb_future perturbs them (P5 / S5 /
    opp). A frame received after the decision can precede it at the venue: the venue's past, whose own-order consequences (a post-only refusal,
    known at POST_NOTICE 0) the client is entitled to, so the engine's future-only world cuts on the venue clock, never on the receipt clock
    (rehearsal 2026-11-19 21:52Z: the receipt-clock cut rewrote the venue's past on 51 of 1,372 rows). The client's own visibility of the
    venue's past is checked separately (hidden_copy)."""
    out = {}
    for a, w in W.items():
        jv = int(np.searchsorted(np.asarray(w['tf'], float), d, side='right') - 1); w2 = dict(w); sl = slice(jv + 1, None)
        P5 = np.array(w['P5'], copy=True); S5 = np.array(w['S5'], dtype=float, copy=True); opp = np.array(w['opp'], copy=True)
        P5[sl] = np.where(P5[sl] >= 0, np.clip(P5[sl] + rng.integers(-3, 4, P5[sl].shape), 1, 99), P5[sl]); S5[sl] = S5[sl] * rng.uniform(.2, 3., S5[sl].shape)
        opp[sl] = np.where(opp[sl] >= 0, np.clip(opp[sl] + rng.integers(-3, 4, opp[sl].shape), 1, 99), opp[sl])
        w2.update(P5=P5, S5=S5, opp=opp, cache={}); out[a] = w2
    return out


def venue_future(W, o, g, rng, draw=0):
    """The engine's future-only world for a decision at g: the path rebuilt from venue_windows, with every trade whose fill would be unnotified at
    the decision (after g - FILL_NOTICE) or later halved -> (the path or None, the builder's status). A draw whose perturbed future leaves no
    valid END mark builds no path (census 2026-11-19 22:15Z: 2 of 2,765 checks, seed 95692 missing_endpoint; the next seeds build)."""
    if VENUE_NO_PATH == 1 or (VENUE_NO_PATH == 2 and draw == 0):                           # planted (selftest only)
        return None, 'planted_no_path'
    t0 = float(o.t); pth, st = QP.build_path(venue_windows(W, t0 + g, rng), t0, t0 + END)
    if pth is None:
        return None, st
    tr = np.array(pth['trades'], float, copy=True); tr[tr[:, 0] > g - BE.FILL_NOTICE, 3] *= .5
    return dict(pth, trades=tr), st


def hidden_copy(B, g, rng, delivered=False):
    """The client-visibility control (review 4cd7b45f P2): a copy of the decision state in which every engine field the client has NOT received
    is perturbed in place, WITHOUT re-running any execution: the book frames after the last received one (E.known), the own fills not yet
    notified (time + FILL_NOTICE > g), every order's engine-truth remaining and queue ahead, and the public trades after g. The delivered public
    history (the received frames) and the own-order notices (acceptance, refusal, notified fills, cancel acknowledgements) stay as they are.
    delivered=True perturbs the last RECEIVED frame instead: the positive control (the client's observation must change)."""
    B2 = copy.deepcopy(B); E = B2.E; f = np.array(E.f, float, copy=True)
    rows = np.arange(E.known, E.known + 1) if delivered else np.arange(E.known + 1, len(f))
    if len(rows):
        px, sz = [2, 3, 4, 5, 6, 12, 13, 14, 15, 16], [7, 8, 9, 10, 11, 17, 18, 19, 20, 21]
        p_ = f[np.ix_(rows, px)]; f[np.ix_(rows, px)] = np.where(p_ >= 0, np.clip(p_ + rng.choice([-3., -2., -1., 1., 2., 3.], p_.shape), 1., 99.), p_)
        f[np.ix_(rows, sz)] = f[np.ix_(rows, sz)] * rng.uniform(1.5, 3., (len(rows), len(sz))) + 1.
    E.f = f
    if not delivered:
        E.fills = [(t, i, s_, q * .5 if t + BE.FILL_NOTICE > g else q, p_, it) for t, i, s_, q, p_, it in E.fills]
        for o_ in E.o:
            o_['rem'] = o_['rem'] * .5 + 1.; o_['ahead'] = o_['ahead'] * 1.7 + 3.
        tr = np.array(E.tr, float, copy=True); fut = tr[:, 0] > g; tr[fut, 3] = tr[fut, 3] * .5 + 1.; E.tr = tr
    return B2


def observe(B, g, s):
    """The client's observation at g as canonical JSON: both sides' views and the context fields, read from a copy (status() closes
    generations)."""
    B_ = copy.deepcopy(B); x = dict(view=B_.view(g, s), other=B_.view(g, 1 - s), ctx=client_snapshot(B_, g, s)[0])
    return json.dumps(x, sort_keys=True, default=str)


def causal_key(r):
    """r2_full_causal_v1's canonical key (review 4cd7b45f): [version, day, instrument, original, side, source, head, offset ms or null, generation,
    decision ms] as compact UTF-8 JSON (the day a string; identifiers, generation and times integers in their units)."""
    off = r.get('offset_ms'); off = None if off is None or not np.isfinite(float(off)) else int(round(float(off)))
    return json.dumps([CAUSAL_VERSION, str(r['day']), int(r['instrument_id']), int(r['original_episode_id']), int(r['side']), str(r['source']), str(r['head']), off,
                       int(round(float(r['generation']))), int(round(float(r['decision_ms'])))], separators=(',', ':'), ensure_ascii=False)


def causal_select(ok, every=None):
    """The audit sample of the stored-row causality check. every=n: every n-th row (the slice rule before review 4cd7b45f). Otherwise
    r2_full_causal_v1: within each original x side x source x head stratum of n sampled ok rows, the min(n, max(1, ceil(n / 64))) rows with the
    smallest SHA-256 of their canonical key, ties by the key; no label, mark or outcome enters. A pure function of the keys: the same after a
    restart, a repartition or any iteration order. The training population and its weights are untouched."""
    if every:
        return ok[::int(every)]
    strata = {}
    for r in ok:
        strata.setdefault((int(r['original_episode_id']), int(r['side']), str(r['source']), str(r['head'])), []).append(r)
    out = []
    for k_ in sorted(strata):
        g_ = sorted(strata[k_], key=lambda r: (hashlib.sha256(causal_key(r).encode('utf-8')).hexdigest(), causal_key(r)))
        out.extend(g_[:min(len(g_), max(1, math.ceil(len(g_) / CAUSAL_DIV)))])
    return out


def causal_rows(rows, path, o, day, W, fair_m, depth, fv, every=None, feat=None, perturb=None, fire=None):
    """The stored-row causality check (reviews d10d3698, 4cd7b45f) on the declared audit sample (causal_select): (i) the STORED 107 inputs equal
    an independent rebuild at the same identity; (ii) they equal a rebuild in a future-only world: the market data received after the decision
    perturbed (the export's own perturbation, for the market and order inputs); the engine's tape rebuilt with the venue's frames after the
    decision perturbed and every trade whose fill would be unnotified at the decision halved (venue_future, for the client state); every fill not
    yet notified moved later / resized in the order inputs; the archived fair path after the decision step changed; (iii) they equal a rebuild
    from the decision state with every unreceived engine field perturbed and no execution re-run, and the client's observation is unchanged
    (hidden_copy: the client-visibility control); (iv) a same-time public input (the export's fire), a notified fill, the fair at the step and
    the last received frame (the observation) each change. A check whose future-only world builds in none of VENUE_DRAWS deterministic seeds is
    NOT EVALUABLE (counted; neither a pass nor a failure). -> (checked, failed, details, components, audit: one record per check)."""
    feat = feat or X.real_features; perturb = perturb or X.real_perturb; fire = fire or X.real_fire
    s, t0 = int(o.side), float(o.t); n_chk = n_bad = 0; det = []; audit = []; cols = INPUTS + CTX
    comp = dict(stored_vs_rebuild=0, stored_vs_perturbed=0, stored_vs_visibility=0, firing=0, not_evaluable=0, retried=0)
    for r in causal_select([x for x in rows if x.get('sampled') and x.get('status') == 'ok'], every):
        g = r['decision_ms']; d = t0 + g; tg = np.asarray(W[s]['tg'], float); kk = min(int(np.searchsorted(tg, d)), len(tg) - 1); k_ep = int(round(g / STEP))
        mk0 = market_at(W, fair_m, depth, feat, s, kk)
        base = rebuild_row(path, W, fv, o, day, r, fair_m, depth, feat, market=mk0); bad_i = same_f32(r, base, cols) if base is not None else ['no grid index']
        sm = int(d) % 100003; Wp, fair_p, dep_p = perturb(W, fair_m, depth, kk, d, np.random.default_rng(sm)); pth_p = None; draws = []
        for i_ in range(VENUE_DRAWS):                                                       # the next deterministic seed when a draw builds no path
            pth_p, st_ = venue_future(W, o, g, np.random.default_rng(int(d) % 100019 + i_), i_); draws.append([int(d) % 100019 + i_, str(st_)])
            if pth_p is not None:
                comp['retried'] += int(i_ > 0); break
        fvp = np.array(fv, float).copy(); fvp[k_ep + 1:] += 7.
        late = lambda ft_, fq_, d_: (np.where(ft_ + KP.NOTICE > d_, ft_ + 777., ft_), np.where(ft_ + KP.NOTICE > d_, fq_ * .5, fq_))
        pert = rebuild_row(pth_p, Wp, fvp, o, day, r, fair_p, dep_p, feat, late) if pth_p is not None else None
        bad_ii = same_f32(r, pert, cols) if pert is not None else None                    # None: not evaluable
        B0 = resim(path, o, day, r); ob0 = observe(B0, g, s); sh_ = int(d) % 100043
        Bh = hidden_copy(B0, g, np.random.default_rng(sh_)); ob_h = observe(Bh, g, s)
        vis = rebuild_row(path, W, fv, o, day, r, fair_m, depth, feat, B=Bh, market=mk0)
        bad_v = same_f32(r, vis, cols) + (['observation'] if ob_h != ob0 else [])
        f_del = observe(hidden_copy(B0, g, np.random.default_rng(sh_), delivered=True), g, s) != ob0
        fw = rebuild_row(path, fire(W, s, kk), fv, o, day, r, fair_m, depth, feat); f_mk = fw is not None and bool(same_f32(base, fw, FC.MARKET))
        fvf = np.array(fv, float).copy(); fvf[min(k_ep, len(fvf) - 1)] += 3.
        f_fair = rebuild_row(path, W, fvf, o, day, r, fair_m, depth, feat, market=mk0)['fair_edge'] != base['fair_edge']
        known = [x for x in json.loads(r['fills']) if t0 + x[0] + KP.NOTICE <= d]
        f_fill = True
        if known:
            nf = lambda ft_, fq_, d_: (ft_, np.where(ft_ + KP.NOTICE <= d_, fq_ * .5, fq_))
            f_fill = rebuild_row(path, W, fv, o, day, r, fair_m, depth, feat, nf, market=mk0)['fp_remaining'] != base['fp_remaining']
        fires = bool(f_mk and f_fair and f_fill and f_del)
        n_chk += 1; fail_ = bool(bad_i) or bool(bad_ii) or bool(bad_v) or not fires; n_bad += int(fail_)
        comp['stored_vs_rebuild'] += int(bool(bad_i)); comp['stored_vs_perturbed'] += int(bool(bad_ii)); comp['stored_vs_visibility'] += int(bool(bad_v))
        comp['firing'] += int(not fires); comp['not_evaluable'] += int(bad_ii is None)
        audit.append(dict(key=causal_key(r), seed_market=sm, draws=draws, seed_hidden=sh_, rebuild=not bad_i, perturbed=None if bad_ii is None else not bad_ii,
                          visibility=not bad_v, fires=fires))
        if fail_ or bad_ii is None:
            det.append(dict(oid=int(o.oid), side=s, source=r['source'], offset_ms=r.get('offset_ms'), decision_ms=g, stored_vs_rebuild=bad_i[:5],
                            stored_vs_perturbed=bad_ii[:5] if bad_ii is not None else 'not evaluable', stored_vs_visibility=bad_v[:5], fires_market=f_mk,
                            fires_fair=bool(f_fair), fires_notified_fill=bool(f_fill), fires_delivered=bool(f_del)))
    return n_chk, n_bad, det, comp, audit


def context_on_truncated(path, o, day, r, cut=True):
    """The client context of row r recomputed on the tape truncated after its decision (venue frames and trades after it removed; cut=False:
    the whole tape - the reference of the falsifier)."""
    g = r['decision_ms']; q = BE.truncate(path, g) if cut else path; s = int(o.side); k_dec = int(round(g / STEP)); ident = episode_ident(day, o)
    if r['source'] in ('entry', 'retain'):
        pol = Single(s, int(round(r['offset_ms'] / STEP))); B = BE.Branch(q, ident=ident)
    else:
        pol = BE.POST_KEEP if r['source'] == 'joint_post_keep' else BE.CHASE; B = BE.Branch(q, ident=ident)
    for k in range(k_dec + 1):
        gg = k * STEP; BE.pre_tick(B, gg); B.k = k
        if k == k_dec:
            if r['source'].startswith('joint') and (B.episode + k) % 2 != s:                    # the other side decided first at this tick
                BE.decide(B, gg, 1 - s, pol)
            return client_snapshot(B, gg, s)[0]
        for side_ in ((B.episode + k) % 2, 1 - (B.episode + k) % 2):
            BE.decide(B, gg, side_, pol)


DIMS = ('c_rem', 'c_queue', 'c_gen', 'c_inv', 'c_other', 'c_price')
DECLARED = dict(c_rem=['0', '(0,10]', '(10,20]', '(20,30]'], c_queue=['unknown', '0', '(0,30]', '(30,150]', '>150'], c_gen=['1', '2', '>=3'],
                c_inv=['long', 'flat', 'short'], c_other=['absent', 'post_pending', 'live', 'cancel_pending', 'terminal'],
                c_price=['[1,5]', '(5,20]', '(20,80]', '(80,95]', '(95,99]'])         # c68a072c's support categories
SIDE_HEADS = ('bid post', 'bid retain', 'ask post', 'ask retain')
FALLBACK = {'post': 'WAIT', 'retain': 'KEEP'}              # c68a072c: unsupported absent states WAIT, unsupported live states KEEP
NUMERIC = ('ctx_generation_age_ms', 'ctx_generation_ordinal', 'ctx_inventory_shares', 'ctx_bid_reserved_shares', 'ctx_ask_reserved_shares', 't_to_end')


def support_part(D):
    """The additive support tables of one partition's sampled ok rows (original episodes are partition-local, so distinct-original counts add):
    cells (side x head x horizon bin), categories and their cross-cells with the horizon bin (rows, distinct originals, nonzero originals),
    per side / head numeric ranges, and the coverage of delayed entry (offset > 0), offset-zero entry and later generations (>= 2); as JSON-able
    records (a part's completion record carries them; review 4cd7b45f)."""
    ok = D[(D.status == 'ok') & D.sampled.astype(bool)].copy()
    if not len(ok):
        return None
    ok['sh'] = np.where(ok.side == 0, 'bid', 'ask') + ' ' + ok['head'].astype(str)
    ok['hbin'] = pd.cut(END - ok.decision_ms.astype(float), HORIZON_BINS, right=True).astype(str); ok['nz'] = ok.q > 1e-12
    rem = ok.fp_remaining.astype(float); qu = ok.fp_queue.astype(float); inv = ok.ctx_inventory_shares.astype(float)
    ok['c_rem'] = np.select([rem <= 0, rem <= 10, rem <= 20, rem <= 30], ['0', '(0,10]', '(10,20]', '(20,30]'], 'n/a')
    ok['c_queue'] = np.select([~np.isfinite(qu), qu <= 0, qu <= 30, qu <= 150], ['unknown', '0', '(0,30]', '(30,150]'], '>150')
    ok['c_gen'] = np.select([ok.generation <= 1, ok.generation <= 2], ['1', '2'], '>=3')
    ok['c_inv'] = np.select([inv > INV_TOL, inv < -INV_TOL], ['long', 'short'], 'flat')
    ok['c_other'] = np.select([ok[f'ctx_other_state_{z}'] > 0 for z in ('absent', 'post_pending', 'live', 'cancel_pending', 'terminal')],
                              ['absent', 'post_pending', 'live', 'cancel_pending', 'terminal'], 'n/a')
    ok['c_price'] = np.select([ok.price <= 5, ok.price <= 20, ok.price <= 80, ok.price <= 95], ['[1,5]', '(5,20]', '(20,80]', '(80,95]'], '(95,99]')
    ok['cov'] = np.select([(ok.source == 'entry') & (ok.offset_ms > 0), (ok.source == 'entry') & (ok.offset_ms == 0), ok.generation >= 2],
                          ['delayed entry', 'offset-zero entry', 'later generation'], 'other')
    cnt = lambda g_: pd.Series(dict(rows=len(g_), originals=g_.original_episode_id.nunique(), nonzero_originals=g_[g_.nz].original_episode_id.nunique()))
    cells = ok.groupby(['sh', 'hbin']).apply(cnt).reset_index()
    cats = pd.concat([ok.groupby(['sh', d_]).apply(cnt).reset_index().rename(columns={d_: 'value'}).assign(dim=d_) for d_ in DIMS], ignore_index=True)
    cross = pd.concat([ok.groupby(['sh', 'hbin', d_]).apply(cnt).reset_index().rename(columns={d_: 'value'}).assign(dim=d_) for d_ in DIMS], ignore_index=True)
    cov = ok.groupby(['sh', 'cov']).apply(cnt).reset_index()
    rng = ok.groupby('sh')[list(NUMERIC)].agg(['min', 'max'])
    rec = lambda df: [{k: (int(v) if k in ('rows', 'originals', 'nonzero_originals') else str(v)) for k, v in r.items()} for r in df.to_dict('records')]
    rng_ = [dict(sh=str(sh_), field=c, lo=float(rng.loc[sh_, (c, 'min')]), hi=float(rng.loc[sh_, (c, 'max')])) for sh_ in rng.index for c in NUMERIC]
    return dict(cells=rec(cells), cats=rec(cats), cross=rec(cross), cov=rec(cov), rng=rng_)


def support_merge(parts):
    """The partition tables summed (counts) / merged (ranges) -> the support JSON (the floors applied to the merged cells)."""
    parts = [x for x in parts if x is not None and x.get('cells')]
    if not parts:
        return dict(cells={}, categories=[], cross_cells=[], coverage=[], ranges={}, presence={}, unsupported_state_category={}, fallback_rule={})
    F_ = lambda k: pd.DataFrame([r for x in parts for r in x[k]])
    cells = F_('cells').groupby(['sh', 'hbin'])[['rows', 'originals', 'nonzero_originals']].sum().reset_index()
    cats = F_('cats').groupby(['sh', 'dim', 'value'])[['rows', 'originals', 'nonzero_originals']].sum().reset_index()
    cross = F_('cross').groupby(['sh', 'hbin', 'dim', 'value'])[['rows', 'originals', 'nonzero_originals']].sum().reset_index()
    cov = F_('cov').groupby(['sh', 'cov'])[['rows', 'originals', 'nonzero_originals']].sum().reset_index()
    rg = F_('rng'); rng = {}
    for sh_, g_ in rg.groupby('sh'):
        rng[sh_] = {c: [float(g_[g_.field == c].lo.min()), float(g_[g_.field == c].hi.max())] for c in NUMERIC}
    got = {(r.sh, r.hbin): r for r in cells.itertuples()}; out = {}
    for sh_ in SIDE_HEADS:                                                                  # every declared cell, an unobserved one with zeros
        for iv in pd.IntervalIndex.from_breaks(HORIZON_BINS, closed='right'):
            r = got.get((sh_, str(iv))); n_, e_, z_ = (int(r.rows), int(r.originals), int(r.nonzero_originals)) if r is not None else (0, 0, 0)
            ok_ = e_ >= FLOORS['episodes'] and z_ >= FLOORS['nonzero_episodes']
            out[f'{sh_} {iv}'] = dict(rows=n_, episodes=e_, nonzero_episodes=z_, support='SUPPORTED' if ok_ else 'INSUFFICIENT_TRAINING_SUPPORT',
                                      fallback=None if ok_ else FALLBACK[sh_.split()[1]])
    pres = {sh_: {d_: {v_: 0 for v_ in DECLARED[d_]} for d_ in DIMS} for sh_ in SIDE_HEADS}
    for r in cats.itertuples():
        if r.value in pres[r.sh][r.dim]:
            pres[r.sh][r.dim][r.value] = int(r.originals)
    unseen = {sh_: [f'{d_}={v_}' for d_ in DIMS for v_ in DECLARED[d_] if pres[sh_][d_][v_] == 0] for sh_ in SIDE_HEADS}
    return dict(cells=out, categories=cats.to_dict('records'), cross_cells=cross.to_dict('records'), coverage=cov.to_dict('records'), ranges=rng,
                presence=pres, unsupported_state_category=unseen,
                fallback_rule=dict(statuses=['unsupported_horizon', 'unsupported_state_category', 'outside_training_range', 'insufficient_training_support'],
                                   absent='WAIT', live='KEEP', pending='process notices', ranges='observed per side / head; outside -> outside_training_range'))


def partitions(sel, size=None):
    """Deterministic partitions of the selected originals: the canonical order (market, original id), chunks of at most `size`."""
    size = size or PART_SIZE; s_ = sel.sort_values(['m', 'oid'], kind='stable').reset_index(drop=True)
    return [s_.iloc[i:i + size] for i in range(0, len(s_), size)]


def materialize(d, max_rows=None):
    """The ONLY all-part reader (review 4cd7b45f P2): the parts in their canonical order as one frame, for synthetic tests and the bounded slice
    receipts; refused above max_rows (default MAX_MATERIALIZE_ROWS) before any part is read. The census itself never calls it. Reads the
    runner's layout (plan.json, parts/part_NNNN.parquet with completion records) or the r2av2 / r2av3 slice layout (manifest.json)."""
    d = Path(d); lim = MAX_MATERIALIZE_ROWS if max_rows is None else int(max_rows)
    if (d / 'manifest.json').exists():
        man = json.loads((d / 'manifest.json').read_text()); files = [(d / x['file'], int(x['rows'])) for x in man['parts']]
    else:
        plan = json.loads((d / 'plan.json').read_text()); files = []
        for i in range(len(plan['partitions'])):
            rec = json.loads((d / 'parts' / f'part_{i:04d}.json').read_text()); files.append((d / 'parts' / rec['file'], int(rec['rows'])))
    n = sum(r for _, r in files); C.require(n <= lim, f'materialize: {n} rows above the bound {lim} (the full day is never materialized)')
    return pd.concat([pd.read_parquet(f) for f, _ in files], ignore_index=True)


read_parts = materialize                                   # the slice-era receipt scripts' name: the same bounded reader


def status_row(day, o, m, st):
    """The one row of an original without episode rows (no path, a missing source): a status, never a silent drop."""
    return dict(day=day, instrument_id=int(m), original_episode_id=int(o.oid), side=int(o.side), t0=float(o.t), source='original', head='post', status=st,
                sampled=True)


def process_partition(day, pi, P, listed, src, off, fair, depth, feat_fn, causal=True, causal_every=None, causal_feat=None, causal_perturb=None,
                      keep_fills=False, prev=None):
    """One partition P of selected originals (reviews d10d3698, 4cd7b45f): every candidate row with its inputs, labels and their per-fill
    reconciliation; the stored-row causality on the declared audit sample; the canonical order and the weights; and the population guards - a
    source pair yielded twice is refused, each original carries exactly one status row OR a set of episode rows (never both, never two status
    rows), canonical candidate identities are unique, and the represented originals equal the partition's. -> (D, info), info JSON-able (the
    part's completion record)."""
    rows, frames = [], []; mk = sorted(set(P.m.astype(int))); present = [m for m in mk if m in listed]
    recon = dict(checked=0, failed=0, no_fill=0); refusals = {}; caus = dict(checked=0, failed=0, details=[], audit=[])
    for o in P[~P.m.astype(int).isin(present)].itertuples():
        rows.append(status_row(day, o, o.m, 'missing_source:market'))
    yielded = set()
    if present:
        data = src.load(day, present, off)
        for W in src.pairs(data):
            m = int(W[0]['m'])
            if m not in present:
                continue
            C.require(m not in yielded, f'partition {pi}: the source yielded the pair of market {m} twice')
            yielded.add(m)
            for o in P[P.m.astype(int) == m].itertuples():
                if DROP_ORIGINAL is not None and int(o.oid) == DROP_ORIGINAL:
                    continue
                if str(o.status) not in ('eligible', 'refused_post_only'):
                    rows.append(status_row(day, o, m, f'no_path:{o.status}')); continue
                path, pst = QP.build_path(W, float(o.t), float(o.t) + END)
                if path is None:
                    rows.append(status_row(day, o, m, f'no_path:{pst}')); continue
                er = probes(path, o, day, [])
                fv = np.array([float(getattr(o, f'f{j:02d}')) for j in range(KP.NSTEP + 1)])
                add_inputs(er, W, o, fair.get(m), depth, feat_fn, lambda k, fv=fv: float(fv[min(k, len(fv) - 1)]))
                if causal:
                    nc_, nb_, dt_, cp_, au_ = causal_rows(er, path, o, day, W, fair.get(m), depth, fv, every=causal_every, feat=causal_feat,
                                                          perturb=(causal_perturb or (None, None))[0], fire=(causal_perturb or (None, None))[1])
                    caus['checked'] += nc_; caus['failed'] += nb_; caus['details'].extend(dt_[:3]); caus['audit'].extend(au_)
                    for k_, v_ in cp_.items():
                        caus[k_] = caus.get(k_, 0) + v_
                for r in er:                                                                # the per-fill reconciliation of every sampled ok row
                    if r.get('sampled') and r.get('status') == 'ok':
                        fl = json.loads(r['fills']); sg_ = 1. if r['side'] == 0 else -1.; eff = r['effective_ms']; mk_ = r['end_mark']
                        qq = sum(b for a, b, c in fl if a >= eff - 1e-9); vv = sum(b * sg_ * (mk_ - c) for a, b, c in fl if a >= eff - 1e-9)
                        idn = r['value'] - (sg_ * (r['pf_mid_ref'] - r['price']) * (r['q'] if r['q'] > 0 else 0.) + r['qm'])
                        recon['checked'] += 1; recon['failed'] += int(not (abs(qq - r['q']) <= 1e-9 and abs(vv - r['value']) <= 1e-9
                                                                              and abs(max(-vv - KP.SEVERE, 0.) - r['severe']) <= 1e-9 and abs(idn) <= 1e-9))
                        recon['no_fill'] += int(r['q'] <= 1e-12)
                        if r['head'] == 'post':
                            k_ = r['source']; refusals[k_] = refusals.get(k_, 0) + int(bool(r.get('refused')))
                    r['decision_abs_ms'] = float(o.t) + float(r['decision_ms']) if r.get('decision_ms') is not None else np.nan
                    if not (r.get('sampled') and r.get('status') == 'ok'):
                        for c in CTX:
                            r.pop(c, None)
                    if not keep_fills:
                        r.pop('fills', None)
                frames.append(frame_of(er)); er = None
        del data
    for o in P[P.m.astype(int).isin([m for m in present if m not in yielded])].itertuples():
        rows.append(status_row(day, o, o.m, 'missing_source:pair'))
    if DUP_ORIGINAL and prev is not None:                                                   # planted: the previous partition's first original again
        rows.append(status_row(day, prev, prev.m, 'no_path:duplicate'))
    if DUP_STATUS and rows:                                                                 # planted: a status row twice in this partition
        rows.append(dict(rows[0]))
    if rows:
        frames.append(frame_of(rows))
    D = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame({c: [] for c in SORT_KEY + ['status', 'sampled', 't0']}); frames = None
    if DUP_EPISODE and (D.source != 'original').any():                                     # planted: an episode row twice in this partition
        D = pd.concat([D, D[D.source != 'original'].head(1)], ignore_index=True)
    miss = [c for c in INPUTS + CTX + ['q', 'value', 'qm', 'severe', 'refused'] if c not in D]
    if miss:
        D = pd.concat([D, pd.DataFrame({c: np.full(len(D), np.nan, np.float32 if c in INPUTS + CTX else float) for c in miss}, index=D.index)], axis=1)
    D = canonical(D.copy())
    ks = list(zip(D.side.astype(int), D.original_episode_id.astype(int))); isst = (D.source == 'original').values      # the multiplicity guards
    n_st = collections.Counter(k for k, x in zip(ks, isst) if x); n_ep = collections.Counter(k for k, x in zip(ks, isst) if not x)
    want_ = set(zip(P.side.astype(int), P.oid.astype(int)))
    C.require(set(n_st) | set(n_ep) == want_, f'partition {pi}: originals represented {len(set(n_st) | set(n_ep))} vs the partition\'s {len(want_)}')
    C.require(all((n_st.get(k, 0) == 1 and n_ep.get(k, 0) == 0) or (n_st.get(k, 0) == 0 and n_ep.get(k, 0) > 0) for k in want_),
              f'partition {pi}: an original with two status rows, or with a status row and episode rows')
    C.require(not D.duplicated(subset=[c for c in SORT_KEY if c in D]).any(), f'partition {pi}: a duplicate canonical candidate identity')
    D['weight'] = weights(D); ok = (D.status == 'ok') & D.sampled.astype(bool)
    stc = {f'{a_} {b_} {c_}': int(v_) for (a_, b_, c_), v_ in D.groupby(['source', 'head', 'status']).size().items()}
    gw = D[ok].groupby(['original_episode_id', 'side', 'head']).weight.sum()
    info = dict(members=[list(x) for x in sorted(want_)], rows=int(len(D)), sampled_ok=int(ok.sum()), status_counts=stc, recon=recon, refusals=refusals,
                weight_groups=int(len(gw)), weight_groups_one=int(np.isclose(gw.values, 1., rtol=0, atol=1e-12).sum()),
                causal=dict({k_: v_ for k_, v_ in caus.items() if k_ not in ('details', 'audit')}, details=caus['details'][:10], audit=caus['audit']),
                support=support_part(D))
    return D, info


def census(day=DAY, per_side=PER_SIDE, out=None, src=None, kp=None, inputs=None, feat_fn=None, markets=None, causal=True, causal_every=None, causal_feat=None,
           causal_perturb=None, tag='', keep_fills=False, part_size=None, population='first_per_side', limits=None):
    """The census through the full-day runner in one call (pf_r2_runner: an immutable plan; parts written to temporary files, validated and
    promoted with completion records; resumable; the resource stop rule; finalized from the records alone) -> (the census directory, the
    support JSON, the readings). It never materializes the parts: materialize() (bounded) is explicit."""
    import pf_r2_runner as RN
    return RN.census(mod=sys.modules[__name__], day=day, per_side=per_side, out=out, src=src, kp=kp, inputs=inputs, feat_fn=feat_fn, markets=markets,
                     causal=causal, causal_every=causal_every, causal_feat=causal_feat, causal_perturb=causal_perturb, tag=tag, keep_fills=keep_fills,
                     part_size=part_size, population=population, limits=limits)


def selftest():  # noqa: C901
    """Synthetic only (see CONTROLS)."""
    global CONTEXT_FROM_ENGINE, CONTEXT_FUTURE, LABEL_NUDGE, ADAPT_CORRUPT, DROP_ORIGINAL, DUP_ORIGINAL, FAIR_LEAK, VENUE_NO_PATH, CONTEXT_UNRECEIVED
    global DUP_STATUS, DUP_EPISODE
    viol = []

    def strict(f):                                                                          # the real builder (feature_rows) reads chosen[0] and chosen[1]
        def g(W_, fa, de, ch):
            if set(ch) != {0, 1}:
                viol.append(sorted(ch)); ch = {0: ch.get(0, []), 1: ch.get(1, [])}
            return f(W_, fa, de, ch)
        return g
    bad = []
    # ---- M1 the manifest
    key = lambda d_, i_, e_, s_, h_: f'{MANIFEST}|{d_}|{i_}|{e_}|{s_}|{h_}'.encode('utf-8')
    for args in [('20261023', m_, e_, s_, h_) for m_ in range(5) for e_ in range(5) for s_ in (0, 1) for h_ in ('post',)]:   # 50 keys: no coincidence hides an error
        if phase(*args) != int(hashlib.sha256(key(*args)).hexdigest()[:16], 16) % 5:
            bad.append(f'M1: the phase of {args} differs from an independent hex-digest computation'); break
    if not (offsets(0) == [0.] + [50. * k for k in range(5, 100, 5)] and offsets(4)[-1] == 4950. and len(offsets(4)) == 21 and len(offsets(0)) == 20
            and all(0 <= phase('20261023', m_, e_, s_, h_) <= 4 for m_ in range(3) for e_ in range(5) for s_ in (0, 1) for h_ in ('post', 'retain'))):
        bad.append('M1: the entry offsets or the phase range')
    if 'hash(' in Path(__file__).read_text().split('def phase')[1].split('def offsets')[0].replace('hashlib', '').replace('.sha256(', ''):
        bad.append('M1: the phase function calls hash()')
    # ---- the engine-level cases: later birth, hidden notice context, short horizon / near_end
    tb = BE.tape([BE.frame(0, 0, 50, 52, 0, 0), BE.frame(120, 125, 51, 53, 0, 0)], [(200, 0, 51, 10), (2000, 0, 50, 30)], 52.)
    o_ = pd.DataFrame([dict(side=0, oid=1, m=7, t=0., phase=1)]).iloc[0]
    rows = probes(tb, o_, '20261023', [])
    late_ = [r for r in rows if r['source'] == 'entry' and r.get('status') == 'ok' and r['offset_ms'] >= 150.]
    e0 = [r for r in rows if r['source'] == 'entry' and r.get('offset_ms') == 0.]
    if not (e0 and e0[0]['price'] == 50. and late_ and all(r['price'] == 51. for r in late_)):
        bad.append(f'later birth: the offset probes do not join the book known at their own decision ({[(r.get("offset_ms"), r.get("price")) for r in rows if r["source"] == "entry"][:4]})')
    srcs = {(r['source'], r['head']) for r in rows if r.get('status') == 'ok' and r.get('sampled')}; cand = {(r['source'], r['head']) for r in rows}
    if not ({('entry', 'post'), ('retain', 'retain'), ('joint_post_keep', 'retain'), ('joint_chase', 'retain')} <= srcs
            and {('joint_post_keep', 'post'), ('joint_chase', 'post')} <= cand):
        bad.append(f'census: a source / head missing on the engine tape (sampled ok {sorted(srcs)}; candidates {sorted(cand)})')
    for r in rows:
        if r['source'] == 'entry' and r.get('status') == 'ok':
            lives = [x for x in rows if x['source'] == 'retain' and x.get('offset_ms') == r['offset_ms'] and x.get('status') in ('ok', 'unsampled')]
            if lives and not any(x.get('first_live') for x in lives):
                bad.append(f"sampling: the first live review of the probe at {r['offset_ms']} ms is not sampled"); break
        if r.get('status') == 'ok' and r.get('sampled') and json.loads(r['fills']) and any(c != r['price'] for _, _, c in json.loads(r['fills'])):
            bad.append(f"labels: a row's price differs from its fills' price ({r['source']} {r['decision_ms']})"); break
    B = BE.Branch(BE.tape([BE.frame(0, 0, 50, 52, 0, 0)], [(200, 0, 50, 10)], 51.)); pol = Single(0, 0)
    ctxs = {}
    for k in range(8):
        g = k * STEP; BE.pre_tick(B, g); B.k = k; ctxs[k] = client_snapshot(B, g, 0)[0]
        for sd in ((B.episode + k) % 2, 1 - (B.episode + k) % 2):
            BE.decide(B, g, sd, pol)
    if not (ctxs[5]['ctx_inventory_shares'] == 0. and ctxs[5]['ctx_bid_reserved_shares'] == 30. and ctxs[6]['ctx_inventory_shares'] == 10.
            and ctxs[6]['ctx_bid_reserved_shares'] == 20. and ctxs[6]['ctx_self_state_live'] == 1. and ctxs[0]['ctx_self_state_absent'] == 1.
            and ctxs[1]['ctx_self_state_post_pending'] == 1. and ctxs[6]['ctx_generation_age_ms'] == 300. and ctxs[0]['ctx_generation_ordinal'] == 1.
            and ctxs[6]['ctx_generation_ordinal'] == 1. and ctxs[6]['ctx_has_generation'] == 1. and ctxs[6]['ctx_other_state_absent'] == 1.):
        bad.append(f'hidden notice / context: a fill at 200 ms (notice 300) - context at 250 {ctxs[5]}, at 300 {ctxs[6]}')
    CONTEXT_FROM_ENGINE = True
    try:
        B2 = BE.Branch(BE.tape([BE.frame(0, 0, 50, 52, 0, 0)], [(200, 0, 50, 10)], 51.)); pol2 = Single(0, 0)
        for k in range(6):
            g = k * STEP; BE.pre_tick(B2, g); B2.k = k; c_ = client_snapshot(B2, g, 0)[0]
            for sd in ((B2.episode + k) % 2, 1 - (B2.episode + k) % 2):
                BE.decide(B2, g, sd, pol2)
        leak = c_['ctx_inventory_shares'] != 0.
    finally:
        CONTEXT_FROM_ENGINE = False
    if not leak:
        bad.append('hidden notice: the planted engine-truth context does not differ before the notice (the check cannot fire)')
    e4 = next((e_ for e_ in range(1, 500) if phase('20261023', 7, e_, 0, 'post') == 4), None)           # an episode whose entry phase reaches 4,950 ms
    if e4 is None:
        bad.append('M1: no episode reaches the entry phase 4 (the phase range)'); e4 = 1
    rows4 = probes(tb, pd.DataFrame([dict(side=0, oid=e4, m=7, t=0., phase=1)]).iloc[0], '20261023', [])
    ent4 = [r for r in rows4 if r['source'] == 'entry']; last = ent4[-1]; prev = ent4[-2]
    if not (last['offset_ms'] == 4950. and last['status'] == 'near_end' and prev['offset_ms'] == 4700. and prev['status'] == 'ok'):
        bad.append(f'short horizon: the last entry offsets ({prev.get("offset_ms")}, {prev.get("status")}), ({last.get("offset_ms")}, {last.get("status")})')
    e0r = next((e_ for e_ in range(1, 500) if phase('20261023', 7, e_, 0, 'retain') == 0 and phase('20261023', 7, e_, 0, 'post') != 0), None)
    if e0r is None:
        bad.append('M1: no episode with retain phase 0 and another post phase'); e0r = 1
    tc = BE.tape([BE.frame(0, 0, 50, 52, 0, 0)], [(1010, 0, 50, 5), (3000, 0, 50, 5)], 51.)                 # a fill 10 ms after a sampled review at 1,000 ms
    rc = probes(tc, pd.DataFrame([dict(side=0, oid=e0r, m=7, t=0., phase=1)]).iloc[0], '20261023', [])
    r1000 = [r for r in rc if r['source'] == 'retain' and r.get('offset_ms') == 0. and r['decision_ms'] == 1000.]
    if not (r1000 and r1000[0]['status'] == 'ok' and r1000[0]['q'] == 5. and r1000[0]['effective_ms'] == 1037.):
        bad.append(f"labels: the retained target at 1,000 ms must exclude the fill at 1,010 ms (before its landing at 1,037) ({r1000[:1]})")
    smp_ = [int(round(r['decision_ms'] / STEP)) for r in rc if r['source'] == 'retain' and r.get('offset_ms') == 0. and r.get('status') == 'ok' and not r.get('first_live')]
    if not (smp_ and all(k_ % 5 == 0 for k_ in smp_)):
        bad.append(f'sampling: retained rows outside the retain phase 0 ({smp_[:6]})')
    Bc = BE.Branch(tc); polc = Single(0, 0); n_open = 0
    for k in range(int(np.ceil(END / STEP))):
        g = k * STEP; BE.pre_tick(Bc, g); Bc.k = k; st_ = Bc.status(0, g)
        n_open += int(k > 0 and Bc.open[0] == 0 and st_ in ('post_pending', 'live', 'cancel_pending'))
        for sd in ((Bc.episode + k) % 2, 1 - (Bc.episode + k) % 2):
            BE.decide(Bc, g, sd, polc)
    n_rows = len([r for r in rc if r['source'] == 'retain' and r.get('offset_ms') == 0.])
    if n_rows != n_open:
        bad.append(f'census: {n_rows} retained candidate rows of the 0 ms probe, an independent count of its open ticks {n_open}')
    seen_ident = []; Branch0 = BE.Branch                                                     # amendment 1: every branch carries the episode identity
    class Spy(Branch0):
        def __init__(self, *a, **k):
            seen_ident.append(dict(k.get('ident') or {})); super().__init__(*a, **k)
    BE.Branch = Spy
    try:
        probes(tb, pd.DataFrame([dict(side=1, oid=3, m=9, t=1000., phase=1)]).iloc[0], '20261023', [])
    finally:
        BE.Branch = Branch0
    want_id = dict(day='20261023', instrument_id=9, original_episode_id=3, side=1, start_abs_ms=1000., end_abs_ms=1000. + END)
    if not (seen_ident and all(all(x.get(k_) == v_ for k_, v_ in want_id.items()) and x.get('mark_source') for x in seen_ident)):
        bad.append(f'identity: a branch without the episode identity ({seen_ident[:1]})')
    tq = BE.tape([BE.frame(0, 0, 50, 52, 0, 0)], [(200, 0, 50, 10)], 51.); oq = pd.DataFrame([dict(side=0, oid=2, m=7, t=0., phase=1)]).iloc[0]   # context truncation
    rq = dict(source='entry', offset_ms=0., decision_ms=400.); rq_ret = dict(source='retain', offset_ms=0., decision_ms=400.)
    c_full = context_on_truncated(tq, oq, '20261023', rq_ret); c_cut = context_on_truncated(BE.truncate(tq, 150.), oq, '20261023', rq_ret)
    if not (c_full['ctx_inventory_shares'] == 10. and c_cut['ctx_inventory_shares'] == 0. and context_on_truncated(BE.truncate(tq, 400.), oq, '20261023', rq_ret) == c_full):
        bad.append(f"causality: the truncated-tape context (known fill: {c_full.get('ctx_inventory_shares')}, cut before it: {c_cut.get('ctx_inventory_shares')})")
    CONTEXT_FUTURE = True                                                                   # the comparison fires on a planted future leak
    try:
        tq2 = BE.tape([BE.frame(0, 0, 50, 52, 0, 0)], [(200, 0, 50, 10), (3000, 0, 50, 5)], 51.)          # a trade after the decision for the leak to read
        fires_ = context_on_truncated(tq2, oq, '20261023', rq_ret, cut=False) != context_on_truncated(tq2, oq, '20261023', rq_ret)
    finally:
        CONTEXT_FUTURE = False
    if not fires_:
        bad.append('causality: the truncated-tape context does not expose a planted future leak (the truncation is not applied)')
    for mk_ in (float('nan'), None):                                                        # amendment 2: a missing mark is a status, never an abort
        try:
            rn = probes(BE.tape([BE.frame(0, 0, 50, 52, 0, 0)], [(200, 0, 50, 10)], mk_), o_, '20261023', [])
            if not (rn and all(r['status'] in ('missing_mark', 'near_end', 'unsampled', 'post_pending', 'cancel_pending', 'cap_blocked') for r in rn)
                    and any(r['status'] == 'missing_mark' for r in rn if r['source'] == 'entry') and any(r['status'] == 'missing_mark' for r in rn if r['source'].startswith('joint'))):
                bad.append(f'missing mark {mk_}: statuses {sorted({r["status"] for r in rn})}')
        except ValueError as e:
            bad.append(f'missing mark {mk_}: the probes raised ({str(e)[:80]})')
    e0p = next((e_ for e_ in range(1, 500) if phase('20261023', 7, e_, 0, 'post') == 0), None)   # review d10d3698 P3: joint POST refusals keep their outcome
    if e0p is None:
        bad.append('M1: no episode with post phase 0'); e0p = 1
    tx = BE.tape([BE.frame(0, 0, 50, 52, 0, 0), BE.frame(40, 41, 49, 50, 0, 0)], [(3000, 0, 49, 5)], 50.)    # the ask falls onto the bid's price before activation
    rx = probes(tx, pd.DataFrame([dict(side=0, oid=e0p, m=7, t=0., phase=1)]).iloc[0], '20261023', [])
    p0 = {r['source']: r for r in rx if r['head'] == 'post' and r['decision_ms'] == 0. and r.get('status') == 'ok'}
    rt_ = [r for r in rx if r['head'] == 'retain' and r['source'].startswith('joint') and r.get('status') == 'ok']
    if not (rt_ and not any('refused' in r for r in rx if r['head'] == 'retain')):
        bad.append(f"refusals: retain rows carry the POST refusal flag, or the fixture has no joint retain ok row ({len(rt_)})")
    if not (set(p0) == {'entry', 'joint_post_keep', 'joint_chase'} and all(p0[k_]['refused'] is True and p0[k_]['q'] == 0. for k_ in p0)):
        bad.append(f"refusals: a POST refused at activation must carry its outcome in every source ({ {k_: (v_.get('refused'), v_.get('q')) for k_, v_ in p0.items()} })")
    def srow(**k_):
        base_ = dict(day='d', instrument_id=1, original_episode_id=1, side=0, source='entry', head='post', offset_ms=0., generation=1, decision_ms=0., status='ok',
                     sampled=True, q=0., price=50., fp_remaining=30., fp_queue=10., ctx_inventory_shares=0., ctx_generation_age_ms=0., ctx_generation_ordinal=1.,
                     ctx_bid_reserved_shares=0., ctx_ask_reserved_shares=0., t_to_end=5000., **{f'ctx_other_state_{z}': 0. for z in ('absent', 'post_pending', 'live', 'cancel_pending', 'terminal')})
        base_['ctx_other_state_absent'] = 1.; base_.update(k_); return base_
    Ds = pd.DataFrame([srow(original_episode_id=1, ctx_inventory_shares=1e-15), srow(original_episode_id=2, ctx_inventory_shares=30., q=5., decision_ms=1000., t_to_end=4000.),
                       srow(original_episode_id=3, side=1, ctx_inventory_shares=-30., ctx_generation_age_ms=700., source='retain', head='retain')])
    Ss = support_merge([support_part(Ds)]); cat_inv = {(c_['sh'], c_['value']): c_['originals'] for c_ in Ss['categories'] if c_['dim'] == 'c_inv'}
    if not (cat_inv.get(('bid post', 'flat')) == 1 and cat_inv.get(('bid post', 'long')) == 1 and cat_inv.get(('ask retain', 'short')) == 1
            and Ss['ranges']['bid post']['ctx_generation_age_ms'] == [0., 0.] and Ss['ranges']['ask retain']['ctx_generation_age_ms'] == [700., 700.]
            and sum(c_['nonzero_originals'] for c_ in Ss['categories'] if c_['dim'] == 'c_inv' and c_['sh'] == 'bid post') == 1):
        bad.append(f'support: the inventory-sign tolerance, the per side / head ranges or the distinct-original counts ({cat_inv}, {Ss["ranges"]})')
    cl_ = Ss['cells']
    if not (len(cl_) == 28 and cl_.get('ask post (0.0, 100.0]', {}).get('rows') == 0 and all(v_['fallback'] == {'post': 'WAIT', 'retain': 'KEEP'}[k_.split()[1]] for k_, v_ in cl_.items())
            and 'c_other=terminal' in Ss['unsupported_state_category']['bid post'] and 'c_inv=long' not in Ss['unsupported_state_category']['bid post']
            and 'c_inv=long' in Ss['unsupported_state_category']['ask retain'] and Ss['presence']['ask retain']['c_inv']['short'] == 1):
        bad.append(f"support: the declared cells / fallbacks / unsupported categories ({len(cl_)}, {Ss['unsupported_state_category'].get('bid post')})")
    # ---- the client-visibility control (review 4cd7b45f P2): a venue frame at 0 ms received at 500 ms; an absent POST at 0 and a live retain row
    tv = BE.tape([BE.frame(0, 0, 50, 52), BE.frame(0, 500, 64, 66)], [], 50.); ov = pd.DataFrame([dict(side=0, oid=e0p, m=7, t=0., phase=1)]).iloc[0]

    def vis_rows(mode):
        global CONTEXT_UNRECEIVED
        CONTEXT_UNRECEIVED = mode
        try:
            res = []
            for r in [x for x in probes(tv, ov, '20261023', []) if x.get('sampled') and x.get('status') == 'ok' and x['source'] in ('entry', 'retain') and x['decision_ms'] < 500]:
                g = r['decision_ms']; B0 = resim(tv, ov, '20261023', r); Bh = hidden_copy(B0, g, np.random.default_rng(1))
                ch = client_snapshot(copy.deepcopy(Bh), g, 0)[0]
                res.append((r['head'], all(np.float32(ch[c]) == np.float32(r[c]) for c in CTX), observe(Bh, g, 0) == observe(B0, g, 0),
                            observe(hidden_copy(B0, g, np.random.default_rng(1), delivered=True), g, 0) != observe(B0, g, 0)))
            return res
        finally:
            CONTEXT_UNRECEIVED = ''
    v_clean, v_leak = vis_rows(''), vis_rows('frame')
    if not ({h for h, *_ in v_clean} == {'post', 'retain'} and all(a and b and c for _, a, b, c in v_clean)
            and {h for h, a, b, _ in v_leak if not (a and b)} == {'post', 'retain'}):
        bad.append(f'visibility: the unreceived-frame fixture (clean equal, planted read refused on the absent and the live row, delivered fires): {v_clean} / {v_leak}')
    # ---- the audit sample r2_full_causal_v1 (review 4cd7b45f): the canonical key, the per-stratum count, the smallest hashes, order-free
    rk = dict(day='20261023', instrument_id=7, original_episode_id=11, side=1, source='joint_chase', head='retain', offset_ms=np.nan, generation=2., decision_ms=1250.)
    if causal_key(rk) != '["r2_full_causal_v1","20261023",7,11,1,"joint_chase","retain",null,2,1250]':
        bad.append(f'audit sample: the canonical key {causal_key(rk)}')
    pool = [dict(rk, decision_ms=float(50 * i)) for i in range(130)] + [dict(rk, head='post', source='entry', offset_ms=float(50 * i), decision_ms=float(50 * i), generation=1.)
                                                                    for i in range(5)]
    hx = lambda r: hashlib.sha256(('["r2_full_causal_v1","20261023",7,11,1,"' + r['source'] + '","' + r['head'] + '",'
                                   + ('null' if r['source'] == 'joint_chase' else str(int(r['offset_ms']))) + ',' + str(int(r['generation'])) + ','
                                   + str(int(r['decision_ms'])) + ']').encode()).hexdigest()
    want = sorted(hx(r) for r in pool[:130])[:3] + sorted(hx(r) for r in pool[130:])[:1]
    got = causal_select(pool); got_r = causal_select(pool[::-1])
    if not (len(got) == 4 and sorted(hx(r) for r in got) == sorted(want) and [causal_key(r) for r in got] == [causal_key(r) for r in got_r]
            and causal_select(pool, every=17) == pool[::17]):
        bad.append(f'audit sample: the per-stratum min(n, max(1, ceil(n / 64))) smallest hashes, independent of order ({len(got)})')
    # ---- census complete, sampling rule, labels by an independent loop, the value identity (engine level)
    keys = [(r['source'], r['head'], r.get('offset_ms'), r['decision_ms']) for r in rows]
    if len(keys) != len(set(keys)):
        bad.append('census: a candidate row appears twice')
    for r in rows:
        if r['source'] == 'retain' and r['status'] == 'ok':
            k = int(round(r['decision_ms'] / STEP))
            if not (k % 5 == r['phase_retain'] or r.get('first_live')):
                bad.append(f'sampling: a retained row outside the manifest at tick {k}'); break
    tl = BE.tape([BE.frame(0, 0, 50, 52, 0, 0)], [(500, 0, 50, 30)], 48.)                          # a 60 c loss: the severe threshold matters
    rl_ = probes(tl, o_, '20261023', [])
    if not any(r.get('status') == 'ok' and r['source'] == 'entry' and r['severe'] == 30. for r in rl_):
        bad.append(f"labels: the severe excess of a 60 c loss is not 30 c ({[(r['source'], r.get('value'), r.get('severe')) for r in rl_ if r.get('status') == 'ok'][:2]})")
    for r in rows + [dict(x, _mark=48.) for x in rl_]:
        if r.get('status') == 'ok' and r.get('sampled'):
            fl = json.loads(r['fills']); q_ = sum(b for a, b, c in fl if a >= r['effective_ms'] - 1e-9); v_ = sum(b * (1. if r['side'] == 0 else -1.) * (r.get('_mark', tb['mark']) - c) for a, b, c in fl if a >= r['effective_ms'] - 1e-9)
            if not (abs(q_ - r['q']) <= 1e-9 and abs(v_ - r['value']) <= 1e-9 and abs(r['severe'] - max(-v_ - KP.SEVERE, 0.)) <= 1e-9):
                bad.append(f"labels: a row's q / value differ from a per-fill loop ({r['source']} {r['decision_ms']})"); break
    # ---- the synthetic day: features, order inputs, causality, the full census
    src = KP.Syn(); days = list(KP.DAYS); d0 = days[0]
    import pf_queue_place_ledger as QL
    with tempfile.TemporaryDirectory(dir=str(I.K.IO)) as tmp:
        tmp = Path(tmp)
        with QL._v1_on(src, tmp):
            for d in days:
                QP.day_run(d)
        with contextlib.redirect_stdout(io.StringIO()):
            for d in days:
                KP.day_run(d, src=src, v1=tmp / 'v1', out=tmp / 'kp')
            cd_, S, R = census(d0, per_side=3, out=tmp / 'r2', src=src, kp=tmp / 'kp', inputs=X.syn_inputs, feat_fn=strict(X.syn_features()),
                               causal_feat=strict(X.syn_features()), causal_perturb=(X.syn_perturb, X.syn_fire), keep_fills=True)
        D = materialize(cd_)
        if viol:
            bad.append(f'features: the builder called without both sides\' keys (the real builder reads both): {viol[:3]}')
        ok = D[(D.status == 'ok') & D.sampled.astype(bool)]
        if not (len(ok) > 0 and ok[INPUTS[:3]].notna().all().all() and ok[CTX].notna().all().all() and set(D.status.str.split(':').str[0]) <= {
                'ok', 'unsampled', 'near_end', 'post_pending', 'cancel_pending', 'cap_blocked', 'feature_unavailable', 'off_grid', 'missing_mark', 'no_path', 'missing_source'}):
            bad.append(f'census: sampled ok rows without inputs / context, or an undeclared status ({sorted(set(D.status))[:8]})')
        idn = ok.value - ((np.where(ok.side == 0, 1., -1.)) * (ok.pf_mid_ref - ok.price) * ok.q.where(ok.q > 0, 0.) + ok.qm)
        if not (np.abs(idn.values) <= 1e-9).all():
            bad.append('labels: value != s (ref - p) q + qm on a sampled row')
        if not (np.abs(ok.t_to_end - (END - ok.decision_ms)).max() <= 1e-9 and np.abs(ok.o_horizon.astype(float) * 1000. - (END - ok.decision_ms)).max() <= 1e-3):   # float32 inputs
            bad.append('inputs: t_to_end / o_horizon not measured to the ORIGINAL END')
        if not {'entry', 'retain', 'joint_post_keep', 'joint_chase'} <= set(ok.source):
            bad.append(f'census: a source without sampled ok rows on the synthetic day ({sorted(set(ok.source))})')
        gw = D[(D.status == 'ok') & D.sampled.astype(bool)].groupby(['original_episode_id', 'side', 'head']).weight.sum()
        if not (len(gw) and np.allclose(gw.values, 1., rtol=0, atol=1e-12) and (D.weight[~((D.status == 'ok') & D.sampled.astype(bool))] == 0).all()):
            bad.append('weights: not one per original episode x side x head over its sampled ok rows (zero elsewhere)')
        if not all(c_ in ok for c_ in ('decision_abs_ms', 'end_abs_ms', 'mark_source')) or not (ok.decision_abs_ms - ok.t0 - ok.decision_ms).abs().max() <= 1e-9 \
                or not (ok.end_abs_ms - ok.t0 - END).abs().max() <= 1e-9 or ok.mark_source.isna().any():
            bad.append('provenance: absolute decision / END times or the mark source of a row')
        if not (R['causality']['checked'] > 0 and R['causality']['failed'] == 0 and R['labels']['verdict'] == 'PASS' and R['labels']['checked'] == R['sampled_ok_rows']):
            bad.append(f"causality / labels: the real-row checks on the synthetic day ({R['causality']}, {R['labels']})")
        LABEL_NUDGE = 1.                                                                    # the reconciliation reading fires on a planted label error
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                Rn = census(d0, per_side=2, out=tmp / 'r2n', src=src, kp=tmp / 'kp', inputs=X.syn_inputs, feat_fn=X.syn_features(), causal=False, keep_fills=True)[2]
        finally:
            LABEL_NUDGE = 0.
        if not (Rn['labels']['failed'] > 0 and Rn['labels']['verdict'] == 'FAIL'):
            bad.append(f"labels: the census reconciliation does not fire on a planted label error ({Rn['labels']})")
        Ok_ = pd.read_parquet(tmp / 'kp' / d0 / 'orders.parquet'); n_leak = 0                    # the check fires on the planted look-ahead feature builder
        for W in src.pairs(src.load(d0, src.markets(d0), src.offset(d0))):
            for o2 in Ok_[(Ok_.m == int(W[0]['m'])) & Ok_.status.isin(('eligible',))].sort_values('oid').head(2).itertuples():
                pth, _ = QP.build_path(W, float(o2.t), float(o2.t) + END)
                if pth is None:
                    continue
                er2 = probes(pth, o2, d0, []); fv2 = np.array([float(getattr(o2, f'f{j:02d}')) for j in range(KP.NSTEP + 1)])
                add_inputs(er2, W, o2, None, None, X.syn_features(), lambda k, fv2=fv2: float(fv2[min(k, len(fv2) - 1)]))
                cr_ = causal_rows(er2, pth, o2, d0, W, None, None, fv2, every=5, feat=X.syn_features(leak=1), perturb=X.syn_perturb, fire=X.syn_fire)
                n_leak += cr_[1]; leak_comp = cr_[3]
            break
        if not (n_leak > 0 and leak_comp['stored_vs_perturbed'] > 0 and leak_comp['stored_vs_rebuild'] == 0):
            bad.append(f'causality: the real-row check on a planted look-ahead builder (a builder leak fails only the perturbed rebuild): {n_leak}, {leak_comp}')
        for mode in ('fair_edge', 't_to_end', 'fp_remaining', 'market'):
            ADAPT_CORRUPT = mode
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    Rc = census(d0, per_side=2, out=tmp / f'r2c_{mode}', src=src, kp=tmp / 'kp', inputs=X.syn_inputs, feat_fn=X.syn_features(), causal_feat=X.syn_features(),
                                causal_perturb=(X.syn_perturb, X.syn_fire), causal_every=1)[2]
            finally:
                ADAPT_CORRUPT = ''
            if not (Rc['causality']['verdict'] == 'FAIL' and Rc['causality']['components']['stored_vs_rebuild'] > 0):
                bad.append(f"causality: a stored {mode} corruption is not refused ({Rc['causality']['components']})")
        sel_m = sorted(set(Ok_[Ok_.oid.isin(Ok_.groupby('side').oid.apply(lambda x: sorted(x)[:2]).explode().astype(int))].m.astype(int)))   # no-drop population
        class Skip:
            def __init__(self, base, m):
                self.b, self.m = base, m
            def markets(self, d_):
                return self.b.markets(d_)
            def offset(self, d_):
                return self.b.offset(d_)
            def load(self, d_, b_, o_):
                return self.b.load(d_, b_, o_)
            def pairs(self, data_):
                return [W_ for W_ in self.b.pairs(data_) if int(W_[0]['m']) != self.m]
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                pm_, _, Rm = census(d0, per_side=2, out=tmp / 'r2m', src=src, kp=tmp / 'kp', inputs=X.syn_inputs, feat_fn=X.syn_features(), causal=False,
                                    markets=[m_ for m_ in src.markets(d0) if m_ != sel_m[0]])
                pp_, _, Rp = census(d0, per_side=2, out=tmp / 'r2p', src=Skip(src, sel_m[0]), kp=tmp / 'kp', inputs=X.syn_inputs, feat_fn=X.syn_features(), causal=False)
                pa_, _, Ra = census(d0, per_side=2, out=tmp / 'r2a', src=src, kp=tmp / 'kp', inputs=X.syn_inputs, feat_fn=X.syn_features(), causal=False)
            Dm, Dp, Da = materialize(pm_), materialize(pp_), materialize(pa_)
            if not ((Dm.status == 'missing_source:market').any() and (Dp.status == 'missing_source:pair').any() and not Da.status.str.startswith('missing_source').any()
                    and Rm['population']['represented'] == Rm['population']['selected'] == Rp['population']['represented'] == Ra['population']['represented']):
                bad.append(f"population: a missing market / pair is not a status, or an original is lost ({Rm['population'].get('represented')}, {Rp['population'].get('represented')})")
        except ValueError as e:
            bad.append(f'population: a census with a missing market / pair was refused instead of carrying a status ({str(e)[:120]})')
        DUP_ORIGINAL = True
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                census(d0, per_side=2, out=tmp / 'r2u', src=src, kp=tmp / 'kp', inputs=X.syn_inputs, feat_fn=X.syn_features(), causal=False, part_size=1)
            bad.append('population: an original emitted in two partitions was not refused')
        except ValueError:
            pass
        finally:
            DUP_ORIGINAL = False
        FAIR_LEAK = True                                                                    # a leak shared by both builders: only the perturbed rebuild sees it
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                Rf = census(d0, per_side=2, out=tmp / 'r2f', src=src, kp=tmp / 'kp', inputs=X.syn_inputs, feat_fn=X.syn_features(), causal_feat=X.syn_features(),
                            causal_perturb=(X.syn_perturb, X.syn_fire), causal_every=1)[2]
        finally:
            FAIR_LEAK = False
        if not (Rf['causality']['verdict'] == 'FAIL' and Rf['causality']['components']['stored_vs_perturbed'] > 0 and Rf['causality']['components']['stored_vs_rebuild'] == 0):
            bad.append(f"causality: a shared future-fair leak is not exposed by the perturbed rebuild ({Rf['causality']['components']})")
        CONTEXT_FROM_ENGINE = True                                                          # a shared engine-truth context: only the halved unnotified trades see it
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                Re = census(d0, per_side=3, out=tmp / 'r2e', src=src, kp=tmp / 'kp', inputs=X.syn_inputs, feat_fn=X.syn_features(), causal_feat=X.syn_features(),
                            causal_perturb=(X.syn_perturb, X.syn_fire), causal_every=1)[2]
        finally:
            CONTEXT_FROM_ENGINE = False
        if not (Re['causality']['components']['stored_vs_perturbed'] > 0 and Re['causality']['components']['stored_vs_rebuild'] == 0):
            bad.append(f"causality: an engine-truth context (unnotified fills) is not exposed by the halved unnotified trades ({Re['causality']['components']})")
        for mode_, want_ in ((1, 'NOT-EVALUABLE'), (2, 'PASS')):                            # an unbuildable future draw: never a pass, never a failure
            VENUE_NO_PATH = mode_
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    cz_ = census(d0, per_side=2, out=tmp / f'r2v{mode_}', src=src, kp=tmp / 'kp', inputs=X.syn_inputs, feat_fn=X.syn_features(),
                                 causal_feat=X.syn_features(), causal_perturb=(X.syn_perturb, X.syn_fire), causal_every=1)[2]['causality']
            finally:
                VENUE_NO_PATH = 0
            ne_ = cz_['components']['not_evaluable']
            if not (cz_['verdict'] == want_ and cz_['failed'] == 0 and cz_['checked'] > 0
                    and ((ne_ == cz_['checked'] and cz_['retried'] == 0) if mode_ == 1 else (ne_ == 0 and cz_['retried'] == cz_['checked']))):
                bad.append(f"causality: an unbuildable future draw (mode {mode_}) must read {want_} ({cz_['verdict']}, {cz_['components']}, retried {cz_['retried']})")
        CONTEXT_UNRECEIVED = 'ahead'                                                        # a hidden engine field read by the context: only visibility sees it
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                Rh = census(d0, per_side=3, out=tmp / 'r2h', src=src, kp=tmp / 'kp', inputs=X.syn_inputs, feat_fn=X.syn_features(), causal_feat=X.syn_features(),
                            causal_perturb=(X.syn_perturb, X.syn_fire), causal_every=1)[2]
        finally:
            CONTEXT_UNRECEIVED = ''
        if not (Rh['causality']['components']['stored_vs_visibility'] > 0 and Rh['causality']['components']['stored_vs_rebuild'] == 0 and Rh['causality']['verdict'] == 'FAIL'):
            bad.append(f"visibility: a context reading the engine-truth queue ahead is not refused by the client-visibility control ({Rh['causality']['components']})")
        class Twice(Skip):                                                                  # one source pair yielded twice
            def pairs(self, data_):
                ws = list(self.b.pairs(data_)); return ws + [W_ for W_ in ws if int(W_[0]['m']) == self.m]
        mult = {}
        for nm_, fn_ in (('repeated pair', lambda: census(d0, per_side=2, out=tmp / 'r2t', src=Twice(src, sel_m[0]), kp=tmp / 'kp', inputs=X.syn_inputs,
                                                          feat_fn=X.syn_features(), causal=False)),
                         ('repeated status', lambda: census(d0, per_side=2, out=tmp / 'r2ts', src=Skip(src, sel_m[0]), kp=tmp / 'kp', inputs=X.syn_inputs,
                                                            feat_fn=X.syn_features(), causal=False)),
                         ('repeated episode', lambda: census(d0, per_side=2, out=tmp / 'r2te', src=src, kp=tmp / 'kp', inputs=X.syn_inputs, feat_fn=X.syn_features(),
                                                             causal=False))):
            DUP_STATUS, DUP_EPISODE = nm_ == 'repeated status', nm_ == 'repeated episode'
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    fn_()
                mult[nm_] = 'accepted'
            except ValueError as e:
                mult[nm_] = str(e)
            finally:
                DUP_STATUS = DUP_EPISODE = False
        if not ('twice' in mult['repeated pair'] and 'two status rows' in mult['repeated status'] and 'duplicate canonical candidate identity' in mult['repeated episode']):
            bad.append(f'population: a repeated pair / status row / episode row within a partition was not refused ({mult})')
        def sentinel_(*a_, **k_):
            raise AssertionError('an all-part read')
        g_ = globals(); keep_m = (g_['materialize'], g_['read_parts'])
        try:
            g_['materialize'] = g_['read_parts'] = sentinel_
            with contextlib.redirect_stdout(io.StringIO()):
                Rs_ = census(d0, per_side=2, out=tmp / 'r2x', src=src, kp=tmp / 'kp', inputs=X.syn_inputs, feat_fn=X.syn_features(), causal=False)
            sent = Rs_[2]['state'] == 'COMPLETE'
        except AssertionError:
            sent = False
        finally:
            g_['materialize'], g_['read_parts'] = keep_m
        try:
            materialize(Rs_[0], max_rows=5); bounded = False
        except ValueError:
            bounded = True
        if not (sent and bounded):
            bad.append(f'materialize: the census read all parts, or the explicit reader is unbounded ({sent}, {bounded})')
        DROP_ORIGINAL = int(Ok_[Ok_.status == 'eligible'].sort_values('oid').oid.iloc[0])
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                census(d0, per_side=2, out=tmp / 'r2d', src=src, kp=tmp / 'kp', inputs=X.syn_inputs, feat_fn=X.syn_features(), causal=False)
            bad.append('population: a dropped original was not refused by the reconciliation')
        except ValueError:
            pass
        finally:
            DROP_ORIGINAL = None
        with contextlib.redirect_stdout(io.StringIO()):
            p1_, S1, _ = census(d0, per_side=3, out=tmp / 'r2s1', src=src, kp=tmp / 'kp', inputs=X.syn_inputs, feat_fn=X.syn_features(), causal=False, part_size=1)
            p9_, S9, _ = census(d0, per_side=3, out=tmp / 'r2s9', src=src, kp=tmp / 'kp', inputs=X.syn_inputs, feat_fn=X.syn_features(), causal=False, part_size=999)
        D1, D9 = materialize(p1_), materialize(p9_)
        if not (D1.equals(D9) and S1 == S9 and len(json.loads((p1_ / 'plan.json').read_text())['partitions']) == 6):
            bad.append('partitions: one-original partitions differ from a single partition (rows or support)')
        if not D[[c for c in SORT_KEY if c in D]].equals(canonical(D)[[c for c in SORT_KEY if c in D]]):
            bad.append('census: not in the manifest order (stable original identity, offset, generation, decision)')
        kk = D.groupby(['original_episode_id', 'source', 'head', 'offset_ms', 'decision_ms'], dropna=False).size()
        if (kk > 1).any():
            bad.append('census: duplicate candidate rows on the synthetic day')
        data = src.load(d0, src.markets(d0), src.offset(d0)); n_c = n_f = 0
        for W in src.pairs(data):
            for s_ in (0, 1):
                tg = np.asarray(W[s_]['tg'], float)
                for kq in range(0, len(tg), max(len(tg) // 3, 1))[:3]:
                    same, fired = X.causal_check(W, None, None, X.syn_features(), kq, s_, np.random.default_rng(kq), X.syn_perturb, X.syn_fire)
                    same_l, _ = X.causal_check(W, None, None, X.syn_features(leak=1), kq, s_, np.random.default_rng(kq), X.syn_perturb, X.syn_fire)
                    n_c += 1; n_f += int(not (same and fired)) + int(same_l)
            break
        Wv0 = None
        for W in src.pairs(data):
            Wv0 = W; break
        Wk = {a: dict(w, kf=np.asarray(w['tf'], float) + 30.) for a, w in Wv0.items()}; tf0 = np.asarray(Wk[0]['tf'], float); jm = len(tf0) // 2
        Vw = venue_windows(Wk, float(tf0[jm]) + 1., np.random.default_rng(5))               # frame jm: at the venue before d, received after it
        keep_ = all(np.array_equal(np.asarray(Vw[a][x])[:jm + 1], np.asarray(Wk[a][x])[:jm + 1]) for a in (0, 1) for x in ('P5', 'S5', 'opp'))
        moved_ = any(not np.array_equal(np.asarray(Vw[a][x])[jm + 1:], np.asarray(Wk[a][x])[jm + 1:]) for a in (0, 1) for x in ('P5', 'S5'))
        if not (keep_ and moved_):
            bad.append(f'causality: the engine world must keep every frame at the venue by the decision (received after it or not) and move later ones ({keep_}, {moved_})')
        if not (n_c > 0 and n_f == 0):
            bad.append(f'features: the export causal check (future inputs unchanged, a same-time input fires; the look-ahead stub caught) failed on {n_f} of {n_c}')
        w0 = None
        for W in src.pairs(data):
            w0 = W[0]; break
        fl_a = (np.array([1000., 3000.]), np.array([5., 7.])); fl_b = (np.array([1000., 3500.]), np.array([5., 9.]))        # a later fill changed
        kn_a = X.known_steps(w0, 0, 158., 50., fl_a, np.array([1200., 2500.]), 5100.); kn_b = X.known_steps(w0, 0, 158., 50., fl_b, np.array([1200., 2500.]), 5100.)
        kn_c = X.known_steps(w0, 0, 158., 50., (np.array([1000., 3000.]), np.array([6., 7.])), np.array([1200., 2500.]), 5100.)
        if not (all(np.array_equal(kn_a[x], kn_b[x], equal_nan=True) for x in X.KNOWN) and kn_c['known_filled'][1] != kn_a['known_filled'][1]):
            bad.append('order inputs: a fill after the decision changed them, or a notified fill did not')
    if bad:
        sys.exit(f'{MARK} SELFTEST FAILED: ' + ' | '.join(bad))
    print(f"synthetic census: {R['census_rows']} rows, {R['sampled_ok_rows']} sampled ok; by source / head / status {R['by_source_head_status']}", flush=True)
    print('r2 shadow selftests passed (synthetic only): M1 the manifest phase (independent digest, range, offsets, no hash()); later birth; hidden-notice '
          'context (and its engine-truth falsifier); short horizon / near_end; census unique rows and the sampling rule; labels by a per-fill loop and the '
          'value identity; the synthetic-day census with inputs and context; the export causal check on the features (the look-ahead stub caught); '
          'order inputs unchanged by a later fill, changed by a notified one; the stored-row rebuild (same identity; perturbed future inputs, '
          'fills and fair; firing) with four planted stored-field corruptions and a shared fair leak refused; the population reconciliation '
          '(missing market / pair statuses; a dropped and a duplicated original refused); one-original partitions = one partition; the joint POST '
          'refusal outcome (never on retain rows); the support cells, fallbacks and unsupported categories', flush=True)


if __name__ == '__main__':
    a = sys.argv[1:]
    if a == ['selftest']:
        selftest()
    elif len(a) in (2, 3) and a[0] == 'census':
        census(a[1], tag=a[2] if len(a) == 3 else '')
    else:
        sys.exit('usage: pf_r2_shadow.py selftest | census <day>')
