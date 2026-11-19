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
 END - decision); plus review c68a072c's 14 mandatory context fields from the branch engine's CLIENT state (acknowledged / reserved; never
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
import contextlib
import hashlib
import io
import json
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
    """The 14 mandatory context fields of side s at g from the branch's CLIENT state (after pre_tick): acknowledged inventory, reservations,
    states, the generation's age and ordinal."""
    Ik, Bk, Ak = B.reserve(g); st, ot = B.status(s, g), B.status(1 - s, g); gid = B.open[s]
    n_side = sum(1 for o in B.E.o if o['side'] == s); o = B.E.o[gid] if gid is not None else None
    if CONTEXT_FROM_ENGINE and o is not None:                                               # planted: engine truth leaks into the context
        Ik = sum((1. if f_[2] == 0 else -1.) * f_[3] for f_ in B.E.fills if f_[0] <= g)
    if CONTEXT_FUTURE:                                                                      # planted: the whole tape's trade count leaks in
        Ik += 1e-3 * len(B.E.tr)
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
                                pr, act = gen_f['price'], gen_f['act']                       # the engine's own generation
                                eff = act if head == 'post' else g + KP.CANCEL
                                src_rows.append(dict(rj, status='ok', sampled=True, gid=gid_f, price=pr, activation_ms=act, **ctx,
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
        k_ep = int(round(r['decision_ms'] / STEP)); fair = ref_fair(k_ep)
        r.update(fp_size=float(KP.SIZE), fp_remaining=float(kn['known_remaining'][0]), fp_queue=float(kn['queue_ahead_known'][0]), fp_shown=float(kn['shown_known'][0]),
                 fp_scale=float(scale), fp_phase=float(o.phase), fair_edge=sg * (fair - r['price']), t_to_end=END - r['decision_ms'], pf_mid_ref=ref)
        r['qm'] = r['value'] - (sg * (ref - r['price']) * r['q'] if r['q'] > 0 else 0.)


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


def causal_rows(rows, path, o, day, W, fair_m, depth, every=17, feat=None, perturb=None, fire=None):
    """The bounded real-row causality check on every `every`-th sampled ok row (deterministic): (a) the market features under the export's own
    X5 check (future public inputs and receipt times moved -> unchanged; a same-time input -> fires); (b) the order inputs with every fill not yet
    notified at the decision moved later / resized -> unchanged, a notified fill resized -> changes; (c) the client context recomputed on the tape
    truncated after the decision (frames of venue time after it and trades after it removed) -> unchanged. -> (checked, failures, details)."""
    s, t0 = int(o.side), float(o.t); w = W[s]; tg = np.asarray(w['tg'], float); n_chk = n_bad = 0; det = []
    want = [r for r in rows if r.get('sampled') and r.get('status') == 'ok']
    for r in want[::every]:
        d = t0 + r['decision_ms']; kk = int(np.searchsorted(tg, d)); kk = min(kk, len(tg) - 1)
        same, fired = X.causal_check(W, fair_m, depth, feat or X.real_features, kk, s, np.random.default_rng(int(d) % 100003), perturb or X.real_perturb, fire or X.real_fire)
        fl = json.loads(r['fills']); ft = np.array([t0 + a for a, _, _ in fl], float); fq = np.array([b for _, b, _ in fl], float)
        late = ft + KP.NOTICE > d; ta = t0 + r['activation_ms']
        ks = lambda ft_, fq_: X.known_steps(w, s, ta, r['price'], (ft_, fq_), np.array([d]), t0 + END)
        k0 = ks(ft, fq); k1 = ks(np.where(late, ft + 777., ft), np.where(late, fq * .5, fq))
        priv = all(np.array_equal(k0[x], k1[x], equal_nan=True) for x in X.KNOWN)
        fires = True
        if (~late).any():
            k2 = ks(ft, np.where(~late, fq * .5, fq)); fires = not np.array_equal(k0['known_filled'], k2['known_filled'])
        ctx_ok = context_on_truncated(path, o, day, r) == {c: r[c] for c in CTX}
        n_chk += 1; ok_ = same and fired and priv and fires and ctx_ok; n_bad += int(not ok_)
        if not ok_:
            det.append(dict(source=r['source'], decision_ms=r['decision_ms'], features_same=bool(same), features_fire=bool(fired), notices_private=bool(priv),
                            notice_fires=bool(fires), context_same=bool(ctx_ok)))
    return n_chk, n_bad, det


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


def support(D):
    """The declared support tables on the sampled ok rows (see SUPPORT)."""
    ok = D[(D.status == 'ok') & D.sampled.astype(bool)].copy(); out = {}
    if not len(ok):
        return dict(cells={}, categories={}, ranges={})
    ok['h_rem'] = END - ok.decision_ms.astype(float)
    ok['hbin'] = pd.cut(ok.h_rem, HORIZON_BINS, right=True).astype(str)
    for (s, h, b), g in ok.groupby(['side', 'head', 'hbin']):
        ep = g.original_episode_id.nunique(); nz = g[g.q > 1e-12].original_episode_id.nunique()
        out[f'{"bid" if s == 0 else "ask"} {h} {b}'] = dict(rows=int(len(g)), episodes=int(ep), nonzero_episodes=int(nz),
                                                            support='SUPPORTED' if (ep >= FLOORS['episodes'] and nz >= FLOORS['nonzero_episodes']) else 'INSUFFICIENT_TRAINING_SUPPORT')
    cat = {}
    rem = ok.get('fp_remaining', pd.Series(np.nan, index=ok.index)); qu = ok.get('fp_queue', pd.Series(np.nan, index=ok.index))
    ok['c_rem'] = np.select([rem <= 0, rem <= 10, rem <= 20, rem <= 30], ['0', '(0,10]', '(10,20]', '(20,30]'], 'n/a')
    ok['c_queue'] = np.select([~np.isfinite(qu), qu <= 0, qu <= 30, qu <= 150], ['unknown', '0', '(0,30]', '(30,150]'], '>150')
    ok['c_gen'] = np.select([ok.generation <= 1, ok.generation <= 2], ['1', '2'], '>=3')
    ok['c_inv'] = np.sign(ok.ctx_inventory_shares).astype(int).astype(str)
    ok['c_other'] = np.select([ok[f'ctx_other_state_{z}'] > 0 for z in ('absent', 'post_pending', 'live', 'cancel_pending', 'terminal')],
                              ['absent', 'post_pending', 'live', 'cancel_pending', 'terminal'], 'n/a')
    ok['c_price'] = np.select([ok.price <= 5, ok.price <= 20, ok.price <= 80, ok.price <= 95], ['[1,5]', '(5,20]', '(20,80]', '(80,95]'], '(95,99]')
    for (s, h), g in ok.groupby(['side', 'head']):
        cat[f'{"bid" if s == 0 else "ask"} {h}'] = {c: g[c].value_counts().to_dict() for c in ('c_rem', 'c_queue', 'c_gen', 'c_inv', 'c_other', 'c_price')}
    rng = {c: [float(ok[c].min()), float(ok[c].max())] for c in ('ctx_generation_age_ms', 'ctx_generation_ordinal', 'ctx_inventory_shares', 'ctx_bid_reserved_shares',
                                                               'ctx_ask_reserved_shares', 't_to_end') if c in ok}
    return dict(cells=out, categories=cat, ranges=rng)


def census(day=DAY, per_side=PER_SIDE, out=None, src=None, kp=None, inputs=None, feat_fn=None, markets=None, causal=True, causal_feat=None, causal_perturb=None,
           tag='', keep_fills=False):
    """The census of the fixed population (see POPULATION) -> census_<day>.parquet (every candidate row with its status) + support_<day>.json
    + the receipt."""
    start = time.time(); C.require(day in BE.SLICE_DAYS, 'consumed days only (Oct 23-27)'); out = Path(out or OUT); kp = Path(kp or KP.OUT)
    C.require(not (out / f'census_{day}{tag}.json').exists(), 'output exists'); src = src or KP.Tape(); feat_fn = feat_fn or X.real_features
    r_ = json.loads((kp / day / 'receipt.json').read_text()); C.require(all(C.sha(kp / day / x) == h for x, h in r_['outputs'].items()), 'keep-pull outputs changed since their receipt')
    O = pd.read_parquet(kp / day / 'orders.parquet')
    sel = pd.concat([O[O.side == s_].sort_values('oid').head(int(per_side)) for s_ in (0, 1)], ignore_index=True)
    mk = [m for m in (src.markets(day) if markets is None else markets) if int(m) in set(sel.m.astype(int))]
    fair, depth, hrs, tracked = (inputs or X.real_inputs)(day, mk); off = src.offset(day); rows = []; parts = []; peak = 0.; caus = dict(checked=0, failed=0, details=[])
    sel_ids = sorted((int(s_), int(x)) for s_, x in zip(sel.side, sel.oid)); sel_hash = hashlib.sha256(json.dumps(sel_ids).encode()).hexdigest()
    recon = dict(checked=0, failed=0, no_fill=0, refused=0)
    for b in range(0, len(mk), 24):
        data = src.load(day, mk[b:b + 24], off)
        for W in src.pairs(data):
            m = int(W[0]['m'])
            for o in sel[sel.m.astype(int) == m].itertuples():
                if str(o.status) not in ('eligible', 'refused_post_only'):
                    rows.append(dict(day=day, instrument_id=m, original_episode_id=int(o.oid), side=int(o.side), t0=float(o.t), source='original', head='post',
                                     status=f'no_path:{o.status}', sampled=True)); continue
                path, pst = QP.build_path(W, float(o.t), float(o.t) + END)
                if path is None:
                    rows.append(dict(day=day, instrument_id=m, original_episode_id=int(o.oid), side=int(o.side), t0=float(o.t), source='original', head='post',
                                     status=f'no_path:{pst}', sampled=True)); continue
                er = probes(path, o, day, [])
                fcols = [f'f{j:02d}' for j in range(KP.NSTEP + 1)]; fv = np.array([float(getattr(o, c)) for c in fcols])
                add_inputs(er, W, o, fair.get(m), depth, feat_fn, lambda k: float(fv[min(k, len(fv) - 1)]))
                if causal:
                    nc_, nb_, dt_ = causal_rows(er, path, o, day, W, fair.get(m), depth, feat=causal_feat, perturb=(causal_perturb or (None, None))[0],
                                                fire=(causal_perturb or (None, None))[1]); caus['checked'] += nc_; caus['failed'] += nb_; caus['details'].extend(dt_[:3])
                for r in er:                                                                # the per-fill reconciliation of every sampled ok row
                    if r.get('sampled') and r.get('status') == 'ok':
                        fl = json.loads(r['fills']); sg_ = 1. if r['side'] == 0 else -1.; eff = r['effective_ms']; mk_ = r['end_mark']
                        qq = sum(b for a, b, c in fl if a >= eff - 1e-9); vv = sum(b * sg_ * (mk_ - c) for a, b, c in fl if a >= eff - 1e-9)
                        idn = r['value'] - (sg_ * (r['pf_mid_ref'] - r['price']) * (r['q'] if r['q'] > 0 else 0.) + r['qm'])
                        recon['checked'] += 1; recon['failed'] += int(not (abs(qq - r['q']) <= 1e-9 and abs(vv - r['value']) <= 1e-9
                                                                              and abs(max(-vv - KP.SEVERE, 0.) - r['severe']) <= 1e-9 and abs(idn) <= 1e-9))
                        recon['no_fill'] += int(r['q'] <= 1e-12); recon['refused'] += int(bool(r.get('refused')))
                    r['decision_abs_ms'] = float(o.t) + float(r['decision_ms']) if r.get('decision_ms') is not None else np.nan
                    if not (r.get('sampled') and r.get('status') == 'ok'):
                        for c in CTX:                                                   # memory: context only on the sampled ok rows
                            r.pop(c, None)
                    if not keep_fills:
                        r.pop('fills', None)
                parts.append(frame_of(er)); er = None
        del data; C.guard('r2 shadow census'); peak = max(peak, I.K.rss_mib())
    if rows:
        parts.append(frame_of(rows))
    D = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(); parts = None
    miss = [c for c in INPUTS + CTX + ['q', 'value', 'qm', 'severe'] if c not in D]
    if miss:
        D = pd.concat([D, pd.DataFrame({c: np.full(len(D), np.nan, np.float32 if c in INPUTS + CTX else float) for c in miss}, index=D.index)], axis=1)
    D = canonical(D.copy()); D['weight'] = weights(D)
    S = support(D); out.mkdir(parents=True, exist_ok=True)
    D.drop(columns=['fills'], errors='ignore').to_parquet(out / f'census_{day}{tag}.parquet', index=False); C.dump(out / f'support_{day}{tag}.json', S)
    st = D.groupby(['source', 'head', 'status']).size().to_dict()
    R = dict(population=dict(per_side=int(per_side), selected=int(len(sel)), by_status=sel.status.astype(str).value_counts().to_dict(), selected_ids_sha256=sel_hash,
                             selected_ids=sel_ids),
             labels=dict(recon, verdict='PASS' if (recon['checked'] > 0 and recon['failed'] == 0) else 'FAIL'),
             causality=dict(checked=caus['checked'], failed=caus['failed'], details=caus['details'][:10],
                            verdict='PASS' if (caus['checked'] > 0 and caus['failed'] == 0) else ('NOT-RUN' if not causal else 'FAIL')),
             census_rows=int(len(D)), sampled_ok_rows=int(((D.status == 'ok') & D.sampled.astype(bool)).sum()),
             by_source_head_status={f'{a} {b} {c}': int(v) for (a, b, c), v in st.items()}, peak_rss_mib=float(peak))
    C.dump(out / f'census_{day}{tag}.json', dict(asof=C.stamp(), day=day, manifest=MANIFEST, costs=COSTS, readings=R, inputs_list_sha256=hashlib.sha256(json.dumps(INPUTS + CTX).encode()).hexdigest(),
                                           inputs=dict(kp_receipt=C.sha(kp / day / 'receipt.json'), hours=hrs, tracked=tracked),
                                           outputs={x: C.sha(out / x) for x in (f'census_{day}{tag}.parquet', f'support_{day}{tag}.json')},
                                           source_sha256=C.sha(Path(__file__).resolve()), engine_sha256=C.sha(Path(BE.__file__).resolve()),
                                           provenance=KP.provenance(), seconds=time.time() - start))
    print(json.dumps(R, indent=1, default=str), flush=True)
    return D, S, R


def selftest():  # noqa: C901
    """Synthetic only (see CONTROLS)."""
    global CONTEXT_FROM_ENGINE, CONTEXT_FUTURE, LABEL_NUDGE
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
            D, S, R = census(d0, per_side=3, out=tmp / 'r2', src=src, kp=tmp / 'kp', inputs=X.syn_inputs, feat_fn=X.syn_features(), causal_feat=X.syn_features(),
                             causal_perturb=(X.syn_perturb, X.syn_fire), keep_fills=True)
        ok = D[(D.status == 'ok') & D.sampled.astype(bool)]
        if not (len(ok) > 0 and ok[INPUTS[:3]].notna().all().all() and ok[CTX].notna().all().all() and set(D.status.str.split(':').str[0]) <= {
                'ok', 'unsampled', 'near_end', 'post_pending', 'cancel_pending', 'cap_blocked', 'feature_unavailable', 'off_grid', 'missing_mark', 'no_path'}):
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
                n_leak += causal_rows(er2, pth, o2, d0, W, None, None, every=5, feat=X.syn_features(leak=1), perturb=X.syn_perturb, fire=X.syn_fire)[1]
            break
        if n_leak == 0:
            bad.append('causality: the real-row check does not fire on the planted look-ahead feature builder')
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
          'order inputs unchanged by a later fill, changed by a notified one', flush=True)


if __name__ == '__main__':
    a = sys.argv[1:]
    if a == ['selftest']:
        selftest()
    elif len(a) in (2, 3) and a[0] == 'census':
        census(a[1], tag=a[2] if len(a) == 3 else '')
    else:
        sys.exit('usage: pf_r2_shadow.py selftest | census <day>')
