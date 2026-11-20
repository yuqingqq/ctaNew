"""THE ARM-BOUND PAIRED-LABEL ADAPTER, r2_labels_v1 (plan v2 stage R2-D; P-2026-005; 2026-11-20, session aea5bfb9).
USER 2026-11-19: "... continue the implementing and next priority loop"; review 8cc1dc49's next priority: "R2-D: build the arm-bound
candidate-label adapter and a bounded Oct24-25 integration receipt under the existing frozen pi0_50_v1." Declared before any paired outcome of
this module is read. Consumed Oct 24-25 only (the plan's candidate-training days); no teacher refit, no candidate fit, no tuning.

CONTRACT (paired_continuation_v2, unchanged), in whole-portfolio gross cents (zero fees / rebates):
 Y_post = R(POST now, then pi0) - R(WAIT this tick, then pi0); Y_keep = R(KEEP this tick, then pi0) - R(PULL now, then pi0).
 R = cash + inventory x the END mark, over every generation and both sides of the arm. Timing: the original END (+5,000 ms), the 50 ms grid,
 58 / 37 / 100 ms. The teacher's RETAIN forecasts are continuation inputs, never this target. A missing or infeasible pair is a status,
 never a zero label.
CONTINUATION pi0: the frozen pi0_50_v1 (freeze 769dff779d584f913aa9c142fb852a47eddbe01bdd6c0930c87b9d28dfa73294), verified by
 pf_r2_teacher.frozen and its model hashes; its library versions are ENFORCED here. This module is new orchestration outside the frozen
 closure; the teacher, engine and feature builders are used unchanged.
EXECUTION (review 8cc1dc49 P1): the saved fork context, validated by pf_branch_engine.validate_ctx before either arm runs, is built two ways:
 - for a stateless roll-in, by pf_branch_engine.fork_state;
 - for the pi0 source, by the same construction with a teacher bound to the roll-in branch.
 pair_bound mirrors pf_branch_engine.pair's control flow line for line (its WAIT_IS_NO_ENTRY / STALE_VIEW switches are asserted False), with
 one change: each arm is a deep copy of the fork state, and a policy factory gives that copy ITS OWN teacher, bound to it and to the world
 (each side's own original, the shared market features). The binding happens before any decision in the arm: the focal side's forced
 action, the remaining side's decision at the fork tick when the alternating order puts it second, and every later tick.
 Every teacher callback checks that the engine's view equals the bound branch's own view at that tick (the binding check). Arms are
 independent: executing them in reverse order changes nothing. The fork state is never mutated (its fingerprint is re-checked after the
 arms).
POPULATION (plan v2 section 5). The PREFIX is the first 32 shared (market, t0) worlds of each day in (m, t) order, which is the committed
 (m, original) order (checked). Per world and side, the candidate fork states of five declared sources, identified separately:
 - neutral: the original-time POST at k = 0 and the held order's live decisions (pf_r2_shadow.probes sources entry / retain at offset 0);
 - delayed: the entry probes at the manifest offsets and their live decisions (offset > 0);
 - joint_post_keep, joint_chase: the census's joint stand-in states (pf_r2_shadow.probes);
 - pi0: the frozen teacher's own two-sided rollout from t0, the focal side's state just before it decides. POST candidates at k = 0 and at
   k mod 5 = its post phase; KEEP candidates at k mod 5 = its retain phase. The phase is pf_r2_shadow.phase, unchanged.
 Statuses are kept for every candidate: the census's, plus pending / not-feasible / unsampled for pi0. Outcome fields are dropped from the
 census rows.
SELECTION (a bounded integration sample; not a training manifest, not a skill gate): per day, side and kind (post / keep), the first 16 ok
 (sampled, pair-feasible) candidates, in a declared order. The order is a round robin over the sources [neutral, delayed, joint_post_keep,
 joint_chase, pi0]: the i-th of every source before the (i+1)-th of any, each source in (world, decision ms, offset, generation) order.
 That gives at most 64 pairs a day, 128 in all. It is never expanded after an outcome is read. The manifest is written and hashed BEFORE any pair runs. It holds the candidate counts, the selection, each selected
 fork's fingerprint, and the sha256 of candidates.jsonl.gz: a deterministic gzip of every candidate row, streamed out world by world. The
 first build held every row in memory and in the manifest, and its resume pass stopped on the 1,536 MiB checkpoint. The label pass refuses a fork
 whose fingerprint differs from the manifest. For pi0, the roll-in fork must equal the census rollout's state, the exact context identity.
ROW KEY: day, instrument, original, side, source, kind, offset, generation, decision ms, the continuation (freeze sha), the scenario
 (settings), END and the fork fingerprint.
RECORDS: one JSON per pair, written to a temporary file and promoted by rename.
 - It carries: Y; per arm R, cash, position, the END mark, open exposure, generations, and the sha256 of its fills / decisions / requests.
 - Per-arm teacher statistics: calls by status and action, binding checks, and the causal probe (every 7th call: hidden_copy unchanged,
   the delivered control, parity with an independent rebuild).
 - Reconciliation: per arm value = cash + position x mark, with position and cash recomputed from the fills; every generation's fills <= its
   size; Y = R_a - R_b; the pre-fork fills identical; the fork unmutated.
 - Elapsed time and high-water.
 MEMORY: one day's inputs (~1 GB of depth arrays) are held at a time; the previous day's are released before the next day loads. The real
 run's high-water checks this; no synthetic check can.
 RESUME skips valid records. STOP RULE: a pair over PAIR_SECONDS or a high-water over 1,536 MiB latches STOP_RESOURCE. The watchdog
 (pf_r2_runner.Sampler, 2,048 MiB) writes STOP_HARD_RSS. Stops are archived (pf_r2_runner.archive_stops) and held until an
 identity-bound disposition of this manifest.
Usage (run_light): pf_r2_labels.py selftest | census <out> | pairs <out> [stop_after] | finalize <out>
"""
import collections
import copy
import gc
import gzip
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pf_branch_engine as BE                             # noqa: E402
import pf_r2_runner as RN                                 # noqa: E402
import pf_r2_shadow as R2                                 # noqa: E402
import pf_r2_teacher as T                                 # noqa: E402

C, pd, I, KP, X, QP = R2.C, R2.pd, R2.I, R2.KP, R2.X, R2.QP
VERSION, MARK = 'r2_labels_v1', 'R2 LABELS'
FREEZE_SHA = '769dff779d584f913aa9c142fb852a47eddbe01bdd6c0930c87b9d28dfa73294'
DAYS, PREFIX, CAP = ('20261024', '20261025'), 32, 16
SOURCES = ('neutral', 'delayed', 'joint_post_keep', 'joint_chase', 'pi0')
KINDS = ('post', 'keep')
PAIR_SECONDS, PEAK_MIB, GUARD_MIB, PROBE_EVERY = 120., 1536., 2048., 7
STEP, END, NT = float(BE.STEP), float(BE.END), int(np.ceil(float(BE.END) / float(BE.STEP)))
KEY = ('day', 'instrument_id', 'original_episode_id', 'side', 'source', 'kind', 'offset_ms', 'generation', 'decision_ms')
OUTCOMES = ('q', 'value', 'qm', 'severe', 'fills', 'effective_ms', 'gid', 'price', 'activation_ms', 'refused', 'end_mark')
STOP_AFTER = None                                          # planted interruption (resume tests / the identity gate)
DEPS = ('pf_r2_labels.py', 'pf_r2_teacher.py', 'pf_r2_shadow.py', 'pf_branch_engine.py', 'pf_r2_runner.py', 'pf_pipeline_bench.py', 'pf_pipeline_export.py',
        'pf_keep_pull.py', 'pf_queue_place_test.py')          # the manifest's declared source closure (fixed, never the process's import set)


def libraries_enforced(fz):
    """The freeze's library versions, enforced (review 8cc1dc49): a different LightGBM / numpy / pandas / pyarrow / Python refuses."""
    import lightgbm as lgb
    import pyarrow
    now = dict(lightgbm=lgb.__version__, numpy=np.__version__, pandas=pd.__version__, pyarrow=pyarrow.__version__, python=platform.python_version())
    C.require(now == fz['libraries'], f"library versions {now} differ from the freeze's {fz['libraries']}")
    return now


def teacher_assets(cfg=T.REAL, freeze_sha=FREEZE_SHA):
    """The frozen pi0_50_v1: freeze verified (sources, manifests, support), its sha pinned, libraries enforced, the 16 models by hash, the support
    adapter. (freeze_sha: a synthetic freeze in the selftest only.)"""
    fz = T.frozen(cfg); C.require(C.sha(cfg['out'] / 'freeze.json') == freeze_sha, 'not the frozen pi0_50_v1 (freeze sha)'); libraries_enforced(fz)
    return T.load_models(cfg, fz), T.Support(json.loads(Path(cfg['support']).read_text())), fz


class ArmTeacher(T.Teacher):
    """The frozen teacher with the binding check: at every callback the engine's view must equal the bound branch's own view at that tick."""

    def __init__(self, models, support, causal=None):
        super().__init__(models, support, causal=causal); self.bind_checked = 0; self.bind_bad = 0

    def fn(self, v):
        bv = self.B.view(float(v['t']), int(v['side'])); self.bind_checked += 1; self.bind_bad += int(bv != v)
        return super().fn(v)

    def stats(self):
        D = pd.DataFrame([{k: d_[k] for k in ('side', 'state', 'status', 'action')} for d_ in self.log]) if self.log else pd.DataFrame(columns=['side', 'state', 'status', 'action'])
        return dict(calls=len(self.log), by_status={str(k): int(v) for k, v in D.status.value_counts().items()},
                    by_action={str(k): int(v) for k, v in D.action.value_counts().items()}, binding_checked=self.bind_checked, binding_bad=self.bind_bad)


def teacher_factory(M, sup, world, every=PROBE_EVERY):
    """A policy factory: each branch gets its own ArmTeacher bound to it and to the world (and its own causal probe)."""
    made = []

    def make(B):
        probe, st = T.causal_probe(every); t_ = ArmTeacher(M, sup, causal=probe); t_.bind(B, world); made.append((t_, st)); return t_
    make.made = made
    return make


def standin_factory(policy):
    """A stateless stand-in: the same policy object for every branch (as pf_branch_engine.pair passes it)."""
    def make(B):
        return policy
    make.made = []
    return make


def ctx_of(B0, k, side, episode=0):
    """The saved fork context of branch B0 at tick k just before `side` decides (pf_branch_engine.fork_state's record)."""
    return dict(contract=BE.CONTRACT, state=B0, k=int(k), episode=int(episode), side=int(side), acted=sorted(B0.acted), tape_id=BE.tape_id(B0.E.f, B0.E.tr, B0.mark),
                end=B0.end, mark=B0.mark, settings=dict(B0.settings), ident=dict(B0.ident), fingerprint=BE.fingerprint(B0))


def bound_fork(path, factory, k, side, ident, episode=0):
    """pf_branch_engine.fork_state with the roll-in policy bound to the roll-in branch: steps 0..k-1, pre_tick at k, the other side first when
    the alternating order puts it first -> the context."""
    B0 = BE.Branch(path, episode, ident=ident); pol = factory(B0); C.require(getattr(pol, 'horizon', None) == END, 'the roll-in does not declare END')
    for kk in range(k):
        BE.step(B0, kk, pol)
    g = k * STEP; BE.pre_tick(B0, g); B0.k = k
    if (episode + k) % 2 != side:
        BE.decide(B0, g, 1 - side, pol)
    return ctx_of(B0, k, side, episode)


def hwm():
    """This process's never-reset high-water (VmHWM, MiB): the one reading of every checkpoint here (review 4393ad19 P2; the watchdog reads its
    own RSS)."""
    return RN.rss()[1]


def _h(x):
    return hashlib.sha256(json.dumps(x, sort_keys=True, default=str).encode()).hexdigest()


def reconcile_arm(L, B):
    """One arm's ledger checks: value = cash + position x mark, position and cash from the fills, every generation's fills <= its size."""
    sg = lambda s: 1. if s == 0 else -1.
    cash = -sum(sg(f[2]) * f[3] * f[4] for f in L['fills']); pos = sum(sg(f[2]) * f[3] for f in L['fills']); gen = {}
    for f in L['fills']:
        gen[f[1]] = gen.get(f[1], 0.) + f[3]
    return bool(abs(cash - L['cash']) <= 1e-6 and abs(pos - L['position']) <= 1e-9 and abs(L['cash'] + L['position'] * B.mark - L['value']) <= 1e-6
                and all(q_ <= B.E.o[g_]['size'] + 1e-9 for g_, q_ in gen.items()))


def pair_bound(ctx, factory, kind, reverse=False):
    """pf_branch_engine.pair with every arm's policy from factory(arm branch) (see the header) -> dict(status, Y, arms, checks)."""
    C.require(kind in ('post', 'keep'), 'undeclared pair kind'); C.require(not BE.WAIT_IS_NO_ENTRY and not BE.STALE_VIEW, 'an engine switch the mirror does not follow')
    k, side, episode = ctx['k'], ctx['side'], ctx['episode']
    BE.validate_ctx(ctx, None, k, side, episode, None)
    Bv = copy.deepcopy(ctx['state']); g = k * STEP; first = (episode + k) % 2; fp = ctx['fingerprint']
    v = Bv.view(g, side)
    if v['state'] != ('absent' if kind == 'post' else 'live'):
        return dict(status=f'not_{kind}_state', state=v['state'])
    if kind == 'post' and not (BE.book_ok(v, side) and v['room'] >= -1e-9 and g + BE.PLACE < Bv.end):
        return dict(status='post_not_feasible', state=v['state'])
    names = (('POST', 'POST'), ('WAIT', 'WAIT')) if kind == 'post' else (('KEEP', 'KEEP'), ('PULL', 'PULL'))
    arms, B_ = {}, {}
    for name, a in (names[::-1] if reverse else names):
        Bx = copy.deepcopy(Bv); pol = factory(Bx); C.require(getattr(pol, 'horizon', None) == END, 'the continuation does not declare END')
        BE.decide(Bx, g, side, pol, forced=a)
        if first == side:
            BE.decide(Bx, g, 1 - side, pol)
        Bx.check_position(g)
        for kk in range(k + 1, NT):
            BE.step(Bx, kk, pol)
        arms[name] = BE.finish(Bx); B_[name] = (Bx, pol)
    na, nb = names[0][0], names[1][0]
    C.require(BE.fingerprint(ctx['state']) == fp, 'the saved fork state changed while its arms ran')
    if any(x['status'] != 'ok' for x in arms.values()):
        return dict(status='missing_mark', fork_fingerprint=fp)
    common = BE.prefork_identical(arms[na]['fills'], arms[nb]['fills'], g)
    out = dict(status='ok', kind=kind, k=k, side=side, Y=arms[na]['value'] - arms[nb]['value'], common_prefork_identical=common, fork_fingerprint=fp,
               fork_known_filled=float(v['known_filled']), fork_first_side=int(first),
               settings=dict(ctx['settings']), mark=ctx['mark'], end=ctx['end'], tape_id=ctx['tape_id'], ident=dict(ctx['ident']), arms={})
    for name in (na, nb):
        L, (Bx, pol) = arms[name], B_[name]
        req = [e_ for e_ in L['requests']]; late = sum(1 for e_ in req for f in L['fills'] if f[1] == e_[2] and f[0] > e_[0])
        out['arms'][name] = dict(value=L['value'], cash=L['cash'], position=L['position'], mark=Bx.mark, open_at_end=L['open_at_end'], generations=L['generations'],
                                 gens_bid=sum(1 for o_ in L['orders9'] if o_[0] == 0), gens_ask=sum(1 for o_ in L['orders9'] if o_[0] == 1), fills_after_request=int(late),
                                 fills=len(L['fills']), fills_sha256=_h(L['fills']), decisions_sha256=_h(L['decisions']), requests_sha256=_h(L['requests']),
                                 reconciled=reconcile_arm(L, Bx), teacher=pol.stats() if isinstance(pol, ArmTeacher) else None)
    out['reconciled'] = bool(all(a_['reconciled'] for a_ in out['arms'].values()) and common
                             and abs(out['Y'] - (out['arms'][na]['value'] - out['arms'][nb]['value'])) <= 1e-9)
    return out


def same_pair(a, b):
    """Two pair results equal in everything but the teacher statistics' causal draws (the arms' ledgers, hashes and Y)."""
    strip = lambda r: {k: ({n: {kk: vv for kk, vv in arm.items() if kk != 'teacher'} for n, arm in v.items()} if k == 'arms' else v) for k, v in r.items()}
    return strip(a) == strip(b)


# ---------------------------------------------------------------- worlds, the prefix census and the selection
def day_orders(day, kp):
    O = pd.read_parquet(Path(kp) / day / 'orders.parquet'); return O


def worlds_of_day(day, kp, n=PREFIX):
    """The first n shared (market, t0) worlds of the day in (m, t) order with one original per side; the committed (m, original) order of their
    first originals is the same order (checked)."""
    O = day_orders(day, kp); W = []
    for (m, t), g in O.groupby(['m', 't'], sort=True):
        C.require(sorted(g.side.astype(int)) == [0, 1], f'{day} world {m} {t}: not one original per side')
        W.append(dict(m=int(m), t=float(t), o={int(r.side): r for r in g.itertuples()}))
    first = O.sort_values(['m', 'oid'], kind='stable').groupby(['m', 't'], sort=False).head(1)
    C.require([(int(a), float(b)) for a, b in zip(first.m, first.t)] == [(w['m'], w['t']) for w in W], f'{day}: the (m, t) order is not the committed (m, original) order')
    return W[:n]


def world_inputs(day, worlds, src=None, inputs=None, feat_fn=None):
    """The day's inputs for the prefix's markets -> (src, data pairs by market, fair, depth, feat_fn)."""
    src = src or KP.Tape(); mk = sorted({w['m'] for w in worlds}); fair, depth, _, _ = (inputs or X.real_inputs)(day, mk); feat_fn = feat_fn or X.real_features
    data = src.load(day, mk, src.offset(day)); return src, {int(W_[0]['m']): W_ for W_ in src.pairs(data)}, fair, depth, feat_fn


def world_bind(day, w, pairs, fair, depth, feat_fn):
    """A world's episode tape and the teacher's world binding (each side's own original, the shared market features), or a status."""
    W_ = pairs.get(w['m'])
    if W_ is None:
        return None, 'missing_source:pair'
    if any(str(w['o'][s].status) not in ('eligible', 'refused_post_only') for s in (0, 1)):
        return None, 'no_path:' + ','.join(sorted({str(w['o'][s].status) for s in (0, 1)}))
    path, pst = QP.build_path(W_, w['t'], w['t'] + END)
    if path is None:
        return None, f'no_path:{pst}'
    fv = np.array([float(getattr(w['o'][0], f'f{j:02d}')) for j in range(KP.NSTEP + 1)])
    chosen = {s: sorted({int(min(np.searchsorted(np.asarray(W_[s]['tg'], float), w['t'] + k * STEP), len(W_[s]['tg']) - 1)) for k in range(NT)}) for s in (0, 1)}
    F = feat_fn(W_, fair.get(w['m']), depth, chosen)
    return dict(path=path, W=W_, fv=fv, o=w['o'], day=day, fair_m=fair.get(w['m']), depth=depth, feat_fn=feat_fn, market={s: F.get(s, {}) for s in (0, 1)}), 'ok'


def census_world(day, wi, w, world, M, sup):
    """Every candidate of one world: the shadow sources (pf_r2_shadow.probes, outcome fields dropped) and the pi0 rollout's focal states, with
    statuses; the pi0 ok candidates carry the rollout state's fingerprint."""
    rows = []
    for s in (0, 1):
        o = w['o'][s]
        for r in R2.probes(world['path'], o, day, []):
            src_ = r['source']; off = float(r.get('offset_ms', np.nan))
            source = ('neutral' if off == 0 else 'delayed') if src_ in ('entry', 'retain') else src_
            kind = 'post' if r['head'] == 'post' else 'keep'
            rows.append(dict({k_: v_ for k_, v_ in r.items() if k_ not in OUTCOMES and not k_.startswith('ctx_')}, source=source, shadow_source=src_, kind=kind,
                             world=wi, m=w['m'], t=w['t'], status=r['status'], sampled=bool(r.get('sampled'))))
    fac = teacher_factory(M, sup, world, every=10 ** 9); B = BE.Branch(world['path'], 0, ident=R2.episode_ident(day, w['o'][0])); pol = fac(B)
    ph = {(s, hd): R2.phase(day, int(w['o'][s].m), int(w['o'][s].oid), s, hd) for s in (0, 1) for hd in ('post', 'retain')}
    for k in range(NT):
        g = k * STEP; BE.pre_tick(B, g); B.k = k
        for s in ((B.episode + k) % 2, 1 - (B.episode + k) % 2):
            v = B.view(g, s); st = v['state']; o = w['o'][s]
            base = dict(day=day, instrument_id=int(o.m), original_episode_id=int(o.oid), side=s, source='pi0', shadow_source='pi0', world=wi, m=w['m'], t=w['t'],
                        offset_ms=np.nan, decision_ms=g, generation=float(sum(1 for x in B.E.o if x['side'] == s) + (1 if st == 'absent' else 0)))
            if st in ('absent', 'live'):
                kind = 'post' if st == 'absent' else 'keep'; smp = (k == 0 or k % 5 == ph[(s, 'post')]) if kind == 'post' else (k % 5 == ph[(s, 'retain')])
                feas = st == 'live' or (BE.book_ok(v, s) and v['room'] >= -1e-9 and g + BE.PLACE < B.end)
                why = None if feas else ('near_end' if g + BE.PLACE >= B.end else 'cap_blocked' if v['room'] < -1e-9 else 'no_book')
                stt = 'ok' if (smp and feas) else (why or 'unsampled')
                rows.append(dict(base, kind=kind, status=stt, sampled=bool(smp), fingerprint=focal_fingerprint(B, day, o) if stt == 'ok' else None))
            else:
                rows.append(dict(base, kind='pending', status=st, sampled=False))
            BE.decide(B, g, s, pol)
        B.check_position(g)
    t_ = fac.made[0][0]
    return rows, dict(pi0_calls=t_.stats()['calls'], pi0_binding_bad=t_.bind_bad, pi0_binding_checked=t_.bind_checked)


def focal_fingerprint(B, day, o):
    """The fork fingerprint of the world rollout's state for the focal original o: the same state under o's identity fields (a shallow copy
    swaps only the metadata; a fork context carries its focal original's identity)."""
    Bc = copy.copy(B); Bc.ident = R2.episode_ident(day, o); return BE.fingerprint(Bc)


def select(rows, cap=CAP):
    """Per day, side and kind: the first `cap` ok candidates in the declared round-robin order over SOURCES (each source in (world, decision,
    offset, generation) order) -> the selected rows in that order, each with its rank."""
    out = []
    D = pd.DataFrame([r for r in rows if r['status'] == 'ok' and r['kind'] in KINDS])
    if not len(D):
        return out
    D['offset_ms'] = D.offset_ms.astype(float); D['_off'] = D.offset_ms.fillna(-1.)
    for (day, s, kind), g in D.groupby(['day', 'side', 'kind'], sort=True):
        g = g.sort_values(['world', 'decision_ms', '_off', 'generation'], kind='stable').copy()
        g['_rank'] = g.groupby('source').cumcount(); g['_src'] = g.source.map({x: i for i, x in enumerate(SOURCES)})
        g = g.sort_values(['_rank', '_src'], kind='stable').head(cap)
        for i, r in enumerate(g.drop(columns=['_off', '_src']).to_dict('records')):
            out.append(dict(r, order=i))
    return out


def fork_for(sel, world, M, sup):
    """The selected candidate's saved fork context: fork_state under its stateless roll-in, or the bound pi0 roll-in."""
    s, k, ident = int(sel['side']), int(round(float(sel['decision_ms']) / STEP)), R2.episode_ident(sel['day'], world['o'][int(sel['side'])])
    if sel['source'] == 'pi0':
        return bound_fork(world['path'], teacher_factory(M, sup, world, every=10 ** 9), k, s, ident)
    if sel['shadow_source'] in ('entry', 'retain'):
        rollin = R2.Single(s, int(round(float(sel['offset_ms']) / STEP)))
    else:
        rollin = BE.POST_KEEP if sel['shadow_source'] == 'joint_post_keep' else BE.CHASE
    return BE.fork_state(world['path'], rollin, k, s, 0, None, ident)


def census(out, days=DAYS, prefix=PREFIX, cap=CAP, cfg=T.REAL, src=None, inputs=None, feat_fn=None, kp=None, assets=None, freeze_sha=FREEZE_SHA):
    """Pass 1: the prefix census of every day, streamed to candidates.jsonl.gz (deterministic gzip, every candidate row) -> the selection, each
    selected fork's fingerprint and the candidate counts -> manifest.json (immutable; content-only). Only one day's inputs and its ok
    candidates are in memory at a time. (assets: stub models in the selftest only.)"""
    out = Path(out); out.mkdir(parents=True, exist_ok=True); mp, cp = out / 'manifest.json', out / 'candidates.jsonl.gz'
    C.require(not mp.exists() and not cp.exists(), 'the manifest exists (immutable)')
    RN.archive_stops(out); mon = RN.Sampler(out / 'census_status.json', GUARD_MIB); mon.start()        # the watchdog over loading and census (4393ad19 P2)
    try:
        return _census(out, mp, cp, days, prefix, cap, cfg, src, inputs, feat_fn, kp, assets, freeze_sha)
    finally:
        mon.live = False


def _census(out, mp, cp, days, prefix, cap, cfg, src, inputs, feat_fn, kp, assets, freeze_sha):
    M, sup, fz = assets or teacher_assets(cfg, freeze_sha); kp = Path(kp or json.loads(Path(cfg['population']).read_text())['inputs']['kp_dir'])
    wst, sel_all, info, counts = [], [], {}, collections.Counter()
    clean = lambda r: {k: (None if (isinstance(v, float) and not np.isfinite(v)) else v) for k, v in r.items()}
    raw = open(cp, 'wb'); gz = gzip.GzipFile(filename='', mode='wb', fileobj=raw, mtime=0); world = ctx = None   # deterministic bytes (A = B)
    for day in days:
        W = worlds_of_day(day, kp, prefix); src_, pairs, fair, depth, ff = world_inputs(day, W, src, inputs, feat_fn); worlds = {}; ok_rows = []
        for wi, w in enumerate(W):
            world, st = world_bind(day, w, pairs, fair, depth, ff); wst.append(dict(day=day, world=wi, m=w['m'], t=w['t'], status=st))
            if world is None:
                continue
            r_, inf = census_world(day, wi, w, world, M, sup); info[f'{day}/{wi}'] = inf; worlds[wi] = world
            for r in r_:                                                                    # every candidate streamed out; only the ok ones kept
                gz.write((json.dumps(clean(r), sort_keys=True, default=str) + '\n').encode()); counts[(day, int(r['side']), r['kind'], r['source'], r['status'])] += 1
                if r['status'] == 'ok' and r['kind'] in KINDS:
                    ok_rows.append(r)
            r_ = None
        sel = select(ok_rows, cap)
        for r in sel:
            ctx = fork_for(r, worlds[r['world']], M, sup); r['fork_fingerprint'] = ctx['fingerprint']
            r['fork_equals_census'] = (r['fingerprint'] == ctx['fingerprint']) if r['source'] == 'pi0' else None
        sel_all.extend(sel)
        del pairs, fair, depth, worlds, src_, ok_rows; world = ctx = None; gc.collect()     # one day's inputs at a time: no world / fork kept (4393ad19 P2)
        if hwm() > PEAK_MIB:                                                                # the checkpoint after every day's loading and census
            gz.close(); raw.close()
            RN.atomic_json(out / 'census_status.json', dict(state='STOP_RESOURCE', stage='census', day=day, hwm_mib=hwm(), peak_mib=PEAK_MIB, asof=C.stamp()))
            raise ValueError(f'census: the process high-water passed {PEAK_MIB} MiB (day {day}); no manifest is written')
    gz.close(); raw.close()
    man = dict(version=VERSION, freeze_sha256=freeze_sha, days=list(days), prefix=prefix, cap=cap, sources=list(SOURCES), settings=dict(BE.SETTINGS),
               worlds=wst, candidates_file=cp.name, candidates_sha256=C.sha(cp), candidate_rows=int(sum(counts.values())),
               candidate_counts=[dict(day=d_, side=s_, kind=k_, source=r_, status=t_, n=int(n_)) for (d_, s_, k_, r_, t_), n_ in sorted(counts.items())],
               selected=[clean(r) for r in sel_all], census_teacher=info, sources_sha256={n: C.sha(HERE / n) for n in DEPS})
    RN.atomic_json(mp, man); RN.atomic_json(out / 'manifest_meta.json', dict(asof=C.stamp(), manifest_sha256=C.sha(mp)))   # content-only manifest; time beside it
    return man


def load_candidates(out):
    """Every candidate row of a census, after checking candidates.jsonl.gz against the manifest's sha256 (audits only; the label pass never reads
    it)."""
    out = Path(out); man = json.loads((out / 'manifest.json').read_text()); cp = out / man['candidates_file']
    C.require(C.sha(cp) == man['candidates_sha256'], 'candidates.jsonl.gz differs from the manifest')
    with gzip.open(cp, 'rt') as f:
        return [json.loads(x) for x in f]


def pairs(out, cfg=T.REAL, src=None, inputs=None, feat_fn=None, kp=None, stop_after=None, assets=None):
    """Pass 2: every selected pair from the immutable manifest, one promoted record each; resume skips valid records; the stop rule latches
    STOP_RESOURCE; held stops need an identity-bound disposition -> the run state."""
    out = Path(out); man = json.loads((out / 'manifest.json').read_text()); msha = C.sha(out / 'manifest.json'); rdir = out / 'pairs'; rdir.mkdir(exist_ok=True)
    mon = RN.Sampler(out / 'label_status.json', GUARD_MIB); mon.start()
    try:
        held = held_stops(out, msha)
        if held['undisposed']:
            RN.atomic_json(out / 'label_status.json', dict(state='HELD', manifest_sha256=msha, held=held, asof=C.stamp())); return 'HELD'
        M, sup, fz = assets or teacher_assets(cfg, man['freeze_sha256']); C.require(assets is not None or man['freeze_sha256'] == FREEZE_SHA, 'a manifest of another teacher')
        kp = Path(kp or json.loads(Path(cfg['population']).read_text())['inputs']['kp_dir']); n_run = 0; state = 'PAIRS_DONE'
        for day in man['days']:
            todo = [r for r in man['selected'] if r['day'] == day and not valid_record(rdir, r, msha)]
            if not todo:
                continue
            W = worlds_of_day(day, kp, man['prefix']); src_, prs, fair, depth, ff = world_inputs(day, W, src, inputs, feat_fn); bound = {}
            for r in todo:
                if (stop_after if stop_after is not None else STOP_AFTER) is not None and n_run >= (stop_after if stop_after is not None else STOP_AFTER):
                    state = 'STOPPED'; break
                t0 = time.time(); mon.reset(); wi = int(r['world'])
                if wi not in bound:
                    bound[wi] = world_bind(day, W[wi], prs, fair, depth, ff)[0]
                world = bound[wi]; ctx = fork_for(r, world, M, sup)
                C.require(ctx['fingerprint'] == r['fork_fingerprint'], f"pair {rkey(r)}: the rebuilt fork differs from the manifest's")
                fac = teacher_factory(M, sup, world); res = pair_bound(ctx, fac, r['kind'])
                cz = [{k_: (v_ if k_ != 'details' else v_[:3]) for k_, v_ in st.items()} for _, st in fac.made]       # each arm's causal / parity probe
                el = time.time() - t0; hw = hwm(); breach = dict(seconds=bool(el > PAIR_SECONDS), rss=bool(hw > PEAK_MIB))
                rec = dict(manifest_sha256=msha, key=rkey(r), candidate=r, result=res, causal=cz, elapsed_s=el, hwm_mib=hw, breach=breach, asof=C.stamp())
                write_record(rdir, r, rec); n_run += 1
                if any(breach.values()):
                    RN.atomic_json(out / 'label_status.json', dict(state='STOP_RESOURCE', manifest_sha256=msha, pair=rkey(r), elapsed_s=el, hwm_mib=hw, **breach))
                    return 'STOP_RESOURCE'
            del prs, fair, depth, bound, src_; world = ctx = fac = res = None; gc.collect()  # one day's inputs at a time: no world / fork / factory kept
            if state == 'STOPPED':
                break
        RN.atomic_json(out / 'label_status.json', dict(state=state, manifest_sha256=msha, done=len([r for r in man['selected'] if valid_record(rdir, r, msha)]),
                                                       asof=C.stamp(), hwm_mib=hwm()))
        return state
    finally:
        mon.live = False


def rkey(r):
    return '|'.join(str(r.get(k)) for k in KEY)


def rname(r):
    return hashlib.sha256(rkey(r).encode()).hexdigest()[:20] + '.json'


def write_record(rdir, r, rec):
    p = rdir / rname(r); tmp = p.with_suffix('.tmp'); tmp.write_text(json.dumps(rec, sort_keys=True, default=str)); os.replace(tmp, p)


def valid_record(rdir, r, msha):
    p = rdir / rname(r)
    if not p.exists():
        return False
    rec = json.loads(p.read_text()); return rec.get('manifest_sha256') == msha and rec.get('key') == rkey(r)


def held_stops(out, msha):
    """Archived stops (pf_r2_runner.archive_stops) and breached pair records, each held until a disposition of THIS manifest names its identity."""
    hist = RN.archive_stops(out); br = sorted((p.name, C.sha(p)) for p in (Path(out) / 'pairs').glob('*.json') if any(json.loads(p.read_text()).get('breach', {}).values()))
    dp = Path(out) / 'disposition.json'; disp = json.loads(dp.read_text()) if dp.exists() else {}
    ok_ids = {e['id'] for e in disp.get('entries', []) if e.get('id') and str(e.get('reason', '')).strip()} if disp.get('plan_sha256') == msha else set()
    und = [f'pair {n} {h[:12]}' for n, h in br if h not in ok_ids] + [f"stop {e['file']} {e['id'][:12]}" for e in hist if e['id'] not in ok_ids]
    return dict(stops=[e['id'] for e in hist], breached=[n for n, _ in br], undisposed=und)


def finalize(out):
    """Every selected pair's record -> labels.csv (one row per pair, the key, Y, both arms' components, checks). Under the watchdog; refused if
    the process high-water passes the checkpoint (4393ad19 P2). The R2-D historical path: production labels use pf_r2_label_sample's
    shared validator."""
    out = Path(out); RN.archive_stops(out); mon = RN.Sampler(out / 'finalize_status.json', GUARD_MIB); mon.start()
    try:
        D = _finalize(out); C.require(hwm() <= PEAK_MIB, f'finalize: the process high-water passed {PEAK_MIB} MiB'); D.to_csv(out / 'labels.csv', index=False); return D
    finally:
        mon.live = False


def _finalize(out):
    man = json.loads((out / 'manifest.json').read_text()); msha = C.sha(out / 'manifest.json'); rdir = out / 'pairs'; rows = []
    C.require(not held_stops(out, msha)['undisposed'], 'finalize: an undisposed stop')                     # publication is held (4393ad19 P1)
    C.require(all(valid_record(rdir, r, msha) for r in man['selected']), 'finalize: a selected pair has no valid record')
    for r in man['selected']:
        rec = json.loads((rdir / rname(r)).read_text()); res = rec['result']; row = {k: r.get(k) for k in KEY}
        row.update(order=r['order'], world=r['world'], status=res['status'], fork_fingerprint=r['fork_fingerprint'], elapsed_s=rec['elapsed_s'], hwm_mib=rec['hwm_mib'],
                   record_sha256=C.sha(rdir / rname(r)))
        if res['status'] == 'ok':
            row.update(Y=res['Y'], reconciled=res['reconciled'], prefork_identical=res['common_prefork_identical'])
            for n, a in res['arms'].items():
                tch = a.get('teacher') or {}
                row.update({f'{n}_{k}': a[k] for k in ('value', 'cash', 'position', 'mark', 'generations', 'fills')}, **{f'{n}_open_bid': a['open_at_end'].get('0', a['open_at_end'].get(0)),
                            f'{n}_open_ask': a['open_at_end'].get('1', a['open_at_end'].get(1)), f'{n}_calls': tch.get('calls'), f'{n}_binding_bad': tch.get('binding_bad'),
                            f'{n}_binding_checked': tch.get('binding_checked')})
        rows.append(row)
    return pd.DataFrame(rows)


def main(argv, **kw):
    """The CLI (review 4393ad19 P3: the census print read a removed field and exited 1 after writing its outputs) -> the exit code. kw: the
    selftest's synthetic injections only."""
    a = list(argv)
    if a == ['selftest']:
        import pf_r2_labels_selftest as ST
        ST.run(sys.modules[__name__]); return 0
    if len(a) == 2 and a[0] == 'census':
        m_ = census(a[1], **kw); print(json.dumps(dict(candidates=m_['candidate_rows'], selected=len(m_['selected'])))); return 0
    if len(a) in (2, 3) and a[0] == 'pairs':
        st_ = pairs(a[1], stop_after=int(a[2]) if len(a) == 3 else None, **kw); print(st_); return 0 if st_ in ('PAIRS_DONE', 'STOPPED') else 2
    if len(a) == 2 and a[0] == 'finalize':
        print(len(finalize(a[1]))); return 0
    print('usage: pf_r2_labels.py selftest | census <out> | pairs <out> [stop_after] | finalize <out>', file=sys.stderr); return 1


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
