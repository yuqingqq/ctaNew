"""ENTRY-RETENTION-DISCRIMINATION-v1 (P-2026-005; 2026-11-21, session aea5bfb9). Review 67a0d9f7's one next priority: a bounded, no-fit
measurement of whether PI0's own frozen forecasts separate unattractive initial entries from subsequent deterioration, on ENTRY-EXPOSURE-v1's
768-fork manifest (consumed Oct 24-27). One diagnostic, zero fitted candidates; declared and frozen before any measurement.
POPULATION: exactly the exposure manifest (committed, byte-equal) and its statuses; per evaluable fork the saved exposure row (POST_PI0 and
 POST_KEEP_INITIAL ledgers, D_initial). Per fork three stages, at most one row each, evaluated separately:
 - entry: the shadow unforced PI0 decision on the exact absent fork state (a copy; the forced branch is untouched): admission would_POST /
   would_WAIT with its status (model, support, cap, near-END, feature_unavailable, incoherent) or engine_fallback (no book, no callback);
 - first_review: the first actual live PI0 callback on the forced initial gid in POST_PI0's prefix;
 - first_pull: its first discretionary PI0 PULL (same_as_review when the first review pulls).
 The POST_PI0 prefix runs as pf_r2_labels.pair_bound runs that arm, under a fresh bound teacher with the declared probe (every 35), and stops
 after the tick of the first pull of gid0, the initial generation's closure, or END. It must equal the saved POST_PI0 arm in decisions,
 requests and fills up to its stop tick (prefix_mismatch), with its true callback count (new decisions - the forced POST - engine fallbacks)
 and probe count ceil(callbacks / 35), zero binding / causal / parity mismatches (callback_mismatch). The fork's END must be 5,000 ms, its
 settings and mark those of the saved arms (horizon_mismatch). No three-arm continuation, no full replay.
RECORDED per stage row (captured in the teacher's fn, i.e. actual callbacks only; decide stays side-effect free): the decision (status,
 action, head), its time, the order's activation and the modeled effect of a PULL there (the engine's request rule), the 107 float32 inputs by
 name with their bit hash (units are not part of the frozen feature schema; its hash is recorded), the client view (book, sizes, received,
 own and other-side state, known filled, position, room, age), the venue order state (queue ahead, remaining; a diagnostic), the
 reservations (I, B, A), and the frozen predictor's outputs through pf_r2_teacher.predict_rows unchanged: P, E[q], E[qm], V, edge =
 sign x (pf_mid - price), q | fill = E[q] / P, qm | fill = E[qm] / P, the support flags; the logged decision's P / V must be reproduced and
 V = edge x E[q] + E[qm] (moments_not_reproduced). A hidden-notice copy (pf_r2_shadow.hidden_copy) of every stage state must give the same
 decision (causality_failed).
TARGETS (pf_r2_shadow.target / X.affected unchanged; END mark; gross): the neutral hold of the forced initial generation from the fork
 (pf_r2_shadow.hold_from), whose gid0 fills must equal the saved KEEP_INITIAL gid0 fills (target_not_equivalent; the structural rule reads
 only the order's own side and fills are allocated per side). entry: fills at or after activation; review / pull: at or after the stage's
 modeled cancellation effect; qm against the stage's own pf_mid; unavoidable = entry value - stage value. At the first pull the held value /
 quantity at or after its effect must equal D_initial / the KEEP - PI0 initial quantity, the POST_PI0 initial fills must be the held fills
 before the effect, and the prefix's actual removal time the modeled effect (pull_not_reconciled). The held initial fills' fill-time execution
 edge: sign x (venue mid at the fill - price) x q, with availability and quantity denominators; a fill's END value is named end_value.
MEASUREMENTS (measure(), declared here): per day x side x stage over the eligible rows (fork and stage ok, a model decision) with every
 exclusion count: fill rate, mean P, Brier of P and of the stage's pooled rate; mean E[q] / q and their MAE; mean V / value; MAE of V, of zero
 and of the mechanical edge x E[q]; among filled rows the q | fill and qm | fill errors. Probability bins PBINS. Per stage x V sign x entry
 admission: benefit / loss / zero counts and sums (floor 1e-6 c), severe (< -30 c) counts, the top-5 |value| share (unweighted rows), worlds.
 Selection (entry: would_WAIT; first_review: PULL) against 500 matched random draws (seed SEED + stage) of the same counts within day x side x
 UTC hour among the same eligible rows: the selected mean value's percentile (lower = the protective action selects worse exposure),
 degenerate strata counted, thin when the selected or unselected pooled count is below 30. Entry -> first_review: changed-input counts
 (float32 bits) per input group (group_of) and dP, dE[q], dE[qm], d edge, dV; the flip (V_entry > 0, V_review < 0) overall and where the
 market_flow and fair groups are unchanged.
PREDICATES (predicates()): per stage discriminates = not thin and percentile <= 0.05; value_fails = pooled MAE(V) >= MAE(0); flip_share = the
 share of forks with both stages modeled where V flips. Readings (each that holds is reported): entry_identifiable (entry discriminates);
 review_protective (first_review discriminates); forecasts_fail (value_fails at entry or first_review, or flip_share >= 0.5 without
 review_protective); not_distinguishable when none holds.
Usage (run_light): pf_entry_retention.py selftest | run <part> | measure"""
import copy
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pf_adv2_replay as AR                              # noqa: E402  frozen: mid_at
import pf_entry_exposure as EX                           # noqa: E402  frozen: the manifest and its machinery

C, pd, T, LS, L, R2, KP, BE, X = EX.C, EX.pd, EX.T, EX.LS, EX.L, EX.R2, EX.KP, EX.BE, EX.X
VERSION = 'entry_retention_v1'
OUT = EX.A.DATA / 'entry_retention_v1_20261121'
EXPO = EX.OUT
REPO = HERE.parents[1]
RECEIPT = 'predictfun/reviews/entry_retention_20261121'
STAGES = ('entry', 'first_review', 'first_pull')
AUDIT_EVERY, NT, STEP, END = 35, L.NT, float(L.STEP), float(L.END)
PBINS = (0., .01, .02, .05, .1, .2, .4, 1.)
NULL_DRAWS, SEED, THIN, PCT, MEANINGFUL, SEVERE, FLIP = 500, 20261121, 30, .05, 1e-6, 30., .5
FEATS = list(T.FEATS)
TIME_F = ('t_rel', 'act_rel', 't_to_end', 'fp_phase', 'o_age', 'o_horizon', 'ctx_generation_age_ms')
FAIR_F = ('fair_edge', 'dfv250', 'dfv1000', 'gap_adv', 'sens', 'pbn_dist', 'sigma')
VIEW_KEYS = ('t', 'state', 'other', 'gid', 'price', 'act', 'known_filled', 'position', 'room', 'bid', 'bid_size', 'ask', 'ask_size', 'received', 'age')
SOURCES = ('pf_entry_retention.py', 'pf_entry_exposure.py', 'pf_r2_labels.py', 'pf_r2_label_sample.py', 'pf_branch_engine.py', 'pf_r2_teacher.py', 'pf_r2_shadow.py',
           'pf_pipeline_bench.py', 'pf_adv2_replay.py')


def group_of(n):
    """The declared input groups: time, fair, order_state (o_*, fp_*, ctx_* not in time), market_flow (every other input)."""
    if n in TIME_F:
        return 'time'
    if n in FAIR_F:
        return 'fair'
    if n.startswith(('o_', 'fp_', 'ctx_')):
        return 'order_state'
    return 'market_flow'


GROUPS = ('market_flow', 'fair', 'order_state', 'time')


def sources():
    return {n: C.sha(HERE / n) for n in SOURCES}


def committed_equal(rel, local):
    r = subprocess.run(['git', '-C', str(REPO), 'show', f'HEAD:{rel}'], capture_output=True)
    return r.returncode == 0 and r.stdout == Path(local).read_bytes()


def jl(x):
    return json.loads(json.dumps(list(x), default=str))


def fork_seed(f):
    return int(hashlib.sha256(f"{f['day']}|{f['m']}|{f['t_ms']}|{f['side']}".encode()).hexdigest()[:8], 16)


def effect_of(o, g):
    """The modeled removal effect of a policy PULL requested at g (the engine's request rule)."""
    return (g + BE.CANCEL) if g >= o['act'] else max(g + BE.CANCEL, o['submit'] + BE.PENDING)


def moments(Mo, dec, s):
    """The frozen predictor on the decision's own inputs (pf_r2_teacher.predict_rows unchanged) -> the moments, the mechanical baseline and the
    reproduction of the logged P / V."""
    row = dec['row']; pr = T.predict_rows(Mo[(s, dec['head'])], pd.DataFrame([row]), s); f = lambda k: float(np.asarray(pr[k])[0]) if k in pr else None
    P, Eq, Eqm, V = f('P'), f('Eq'), f('Eqm'), f('V'); sg = 1. if s == 0 else -1.; edge = sg * (float(np.float32(row['pf_mid'])) - float(row['price']))
    rep = bool((dec.get('V') is None or dec['V'] == V) and (dec.get('P') is None or dec['P'] == P))
    return dict(P=P, Eq=Eq, Eqm=Eqm, V=V, edge=edge, mech=edge * Eq, q_fill=(Eq / P) if P > 0 else None, qm_fill=(Eqm / P) if P > 0 else None, S=f('S'),
                ok=bool(np.asarray(pr['ok'])[0]), zq=bool(np.asarray(pr['zq'])[0]) if 'zq' in pr else None, reproduced=rep,
                v_identity=bool(abs(edge * Eq + Eqm - V) <= 1e-9 * max(1., abs(V))))


def capture(t_, v, dec, stage, seed):
    """One stage row from an actual callback (header)."""
    B = t_.B; g, s = float(v['t']), int(v['side']); gid = v['gid']; o = B.E.o[gid] if gid is not None else None; row = dec.get('row')
    B2 = R2.hidden_copy(B, g, np.random.default_rng([seed, STAGES.index(stage)])); d2 = t_.decide(B2.view(g, s), B2)
    x = None if row is None else [float(np.float32(row[c])) for c in FEATS]
    return dict(stage=stage, status='ok', t=g, side=s, gid=gid, state=v['state'], dec_status=dec['status'], action=dec['action'], head=dec.get('head'),
                view={k: v.get(k) for k in VIEW_KEYS}, reserve=[float(z) for z in B.reserve(g)],
                venue=None if o is None else dict(ahead=float(o['ahead']), rem=float(o['rem']), act=float(o['act']), submit=float(o['submit']), price=float(o['price']),
                                                  remove=float(o['remove']), req=o['req']),
                effect=None if o is None else float(effect_of(o, g)), inputs=x, price=None if row is None else float(row['price']),
                inputs_sha256=None if x is None else hashlib.sha256(np.asarray(x, np.float32).tobytes()).hexdigest(),
                logged=dict(P=dec.get('P'), V=dec.get('V')), moments=None if row is None else moments(t_.M, dec, s), hidden_same=bool(T.same_decision(dec, d2)))


class RecTeacher(L.ArmTeacher):
    """The frozen teacher (decisions unchanged) recording the focal initial generation's stage callbacks in fn - actual callbacks only; the
    causal probe calls decide, which this class does not touch."""

    def __init__(self, models, support, causal=None):
        super().__init__(models, support, causal=causal); self.focal, self.gid0, self.seed, self.stage = None, None, 0, {}

    def fn(self, v):
        a = super().fn(v); dec = self.log[-1]
        if self.gid0 is not None and int(v['side']) == self.focal and v['state'] == 'live' and v['gid'] == self.gid0:
            first = 'first_review' not in self.stage
            if first:
                self.stage['first_review'] = capture(self, v, dec, 'first_review', self.seed)
            if a == 'PULL' and 'first_pull' not in self.stage:
                self.stage['first_pull'] = dict(self.stage['first_review'], stage='first_pull', same_as_review=True) if first else \
                    dict(capture(self, v, dec, 'first_pull', self.seed), same_as_review=False)
        return a


def entry_stage(ctx, world, Mo, sup, seed, cls=RecTeacher):
    """The shadow unforced PI0 decision on a copy of the exact absent fork state (header)."""
    Bc = copy.deepcopy(ctx['state']); k, s = int(ctx['k']), int(ctx['side']); g = k * STEP; v = Bc.view(g, s)
    if v['state'] != 'absent':
        return dict(stage='entry', status='not_absent')
    if not BE.book_ok(v, s):
        return dict(stage='entry', status='engine_fallback')
    t_ = cls(Mo, sup); t_.bind(Bc, world); dec = t_.decide(v, Bc); c = capture(t_, v, dec, 'entry', seed)
    return dict(c, admission='would_POST' if dec['action'] == 'POST' else 'would_WAIT')


def prefix(ctx, world, Mo, sup, seed, cls=RecTeacher):
    """POST_PI0's prefix from the saved fork (header) -> (branch, teacher, probe stats, gid0, stop tick ms, stop reason, decisions / fallbacks
    before the fork)."""
    k, s, ep = int(ctx['k']), int(ctx['side']), int(ctx['episode']); g = k * STEP; first = (ep + k) % 2
    Bx = copy.deepcopy(ctx['state']); probe, cst = T.causal_probe(AUDIT_EVERY); pol = cls(Mo, sup, causal=probe); pol.bind(Bx, world)
    C.require(getattr(pol, 'horizon', None) == END, 'the continuation does not declare END'); nd0, nf0 = len(Bx.decisions), len(Bx.fallbacks)
    BE.decide(Bx, g, s, pol, forced='POST'); gid0 = Bx.open[s]; pol.focal, pol.gid0, pol.seed = s, gid0, seed
    if first == s:
        BE.decide(Bx, g, 1 - s, pol)
    Bx.check_position(g); stop, why = g, 'end'
    for kk in range(k + 1, NT):
        BE.step(Bx, kk, pol); stop = kk * STEP
        if 'first_pull' in pol.stage:
            why = 'pull'; break
        if gid0 in Bx.closed:
            why = 'closed:' + str(Bx.closed[gid0][0]); break
    return Bx, pol, cst, gid0, stop, why, nd0, nf0


def prefix_check(Bx, pol, cst, stop, saved, nd0, nf0):
    """The prefix against the saved POST_PI0 arm up to the stop tick, and its true callback / probe denominators."""
    cut = lambda L_: [x for x in L_ if float(x[0]) <= stop]
    same = bool(cut(saved['decision_list']) == cut(jl(Bx.decisions)) and cut(saved['fill_list']) == cut(jl(Bx.E.fills))
                and cut(saved['requests']) == cut(jl([e for e in Bx.E.events if e[1].endswith('_request')])))
    calls = len(Bx.decisions) - nd0 - 1 - (len(Bx.fallbacks) - nf0)
    cb = bool(len(pol.log) == calls and cst['checked'] == -(-len(pol.log) // AUDIT_EVERY) and pol.bind_bad == 0 and pol.bind_checked == len(pol.log)
              and cst['mismatch'] == 0 and cst['parity_mismatch'] == 0)
    return dict(prefix_same=same, callbacks_ok=cb, callbacks=len(pol.log), expected_callbacks=int(calls), probe_checked=int(cst['checked']),
                decisions_to_stop=len(cut(jl(Bx.decisions))), fills_to_stop=len(cut(jl(Bx.E.fills))))


def unavoidable_of(te, ts):
    """The part of the entry value a decision at the stage cannot avoid: the entry target's value less the stage target's (fills before the effect)."""
    return te['value'] - ts['value']


def stage_target(fills, gid0, s, mark, eff, ref):
    t_ = R2.target(fills, gid0, s, mark, eff, ref); return dict(q=float(t_['q']), value=float(t_['value']), qm=float(t_['qm']), severe=float(t_['severe']), eff=float(eff))


def fork_stages(f, world, Mo, sup, saved):
    """One manifest fork -> its three stage rows, the held targets and every check (header)."""
    r = dict({k: f[k] for k in ('day', 'm', 't_ms', 'hour', 'side', 'oid')}, status=f['status'])
    if f['status'] != 'ok':
        return r
    if not saved or saved.get('status') != 'ok' or not saved.get('arms'):
        return dict(r, status='no_exposure_row')
    t0 = time.time(); s = int(f['side']); seed = fork_seed(f); ctx = L.fork_for(f['sel'], world, Mo, sup)
    if ctx['fingerprint'] != f['fork_fingerprint']:
        return dict(r, status='fork_mismatch')
    ap, ak = saved['arms']['POST_PI0'], saved['arms']['POST_KEEP_INITIAL']; mark = float(ap['mark'])
    horizon = bool(float(ctx['end']) == END == float(ap['end']) == float(ak['end']) and dict(ctx['settings']) == ap['settings'] == ak['settings']
                   and BE.canon_mark(ctx['mark']) == mark == float(ak['mark']))
    ent = entry_stage(ctx, world, Mo, sup, seed)
    Bx, pol, cst, gid0, stop, why, nd0, nf0 = prefix(ctx, world, Mo, sup, seed); pc = prefix_check(Bx, pol, cst, stop, ap, nd0, nf0)
    gid_h, Lh, gen = R2.hold_from(ctx['state'], int(ctx['k']), s); fh_all = jl(Lh['fills']); fh = [x for x in fh_all if int(x[1]) == int(gid0)]
    fk = [x for x in ak['fill_list'] if int(x[1]) == int(saved['init_KEEP']['gid0'])]
    equivalent = bool(gid_h == gid0 == int(saved['init_KEEP']['gid0']) == int(saved['init_PI0']['gid0']) and Lh['status'] == 'ok' and fh == fk)
    st = dict(entry=ent)
    for nm in ('first_review', 'first_pull'):
        st[nm] = pol.stage.get(nm) or dict(stage=nm, status=f"no_{'review' if nm == 'first_review' else 'pull'}:{why}")
    e_ref = (ent.get('inputs') or [None] * len(FEATS))[FEATS.index('pf_mid')]; act = float(gen['act'])
    tgt_e = stage_target(fh, gid0, s, mark, act, e_ref if e_ref is not None else np.nan)
    st['entry']['target'] = tgt_e
    for nm in ('first_review', 'first_pull'):
        if st[nm].get('status') == 'ok':
            ref = st[nm]['inputs'][FEATS.index('pf_mid')] if st[nm].get('inputs') else np.nan
            st[nm]['target'] = stage_target(fh, gid0, s, mark, st[nm]['effect'], ref); st[nm]['unavoidable'] = unavoidable_of(tgt_e, st[nm]['target'])
    rec = None
    if st['first_pull'].get('status') == 'ok':
        eff = st['first_pull']['effect']; tp = st['first_pull']['target']; fp_ = [x for x in ap['fill_list'] if int(x[1]) == int(gid0)]
        rec = bool(abs(tp['value'] - float(saved['D_initial'])) <= 1e-6 and abs(tp['q'] - (float(saved['init_KEEP']['filled']) - float(saved['init_PI0']['filled']))) <= 1e-9
                   and fp_ == [x for x in fh if float(x[0]) < eff] and float(Bx.E.o[gid0]['remove']) == eff)
    sg = 1. if s == 0 else -1.; fte = []
    for x in fh:
        m_ = AR.mid_at(ctx['state'].E.f, float(x[0])); fte.append(dict(t=float(x[0]), q=float(x[3]), price=float(x[4]), end_value=sg * (mark - float(x[4])) * float(x[3]),
                                                                        fill_time_edge=None if m_ is None else sg * (m_ - float(x[4])) * float(x[3])))
    rows_ = [x for x in (st['entry'], st['first_review'], st['first_pull']) if x.get('status') == 'ok']
    causal = all(x['hidden_same'] for x in rows_); mom = all(x['moments'] is None or (x['moments']['reproduced'] and x['moments']['v_identity']) for x in rows_)
    checks = dict(horizon_ok=horizon, **pc, equivalent=equivalent, pull_reconciled=rec, causal_ok=causal, moments_ok=mom)
    status = next((n for ok_, n in ((horizon, 'horizon_mismatch'), (pc['prefix_same'], 'prefix_mismatch'), (pc['callbacks_ok'], 'callback_mismatch'),
                                    (equivalent, 'target_not_equivalent'), (rec is not False, 'pull_not_reconciled'), (causal, 'causality_failed'),
                                    (mom, 'moments_not_reproduced')) if not ok_), 'ok')
    return dict(r, status=status, checks=checks, gid0=int(gid0), stop_ms=float(stop), stop_why=why, stages=st, held=dict(gid=int(gid_h), act=act, fills=fh, mark=mark),
                fills=fte, D_initial=float(saved['D_initial']), seconds=time.time() - t0)


def exposure_rows(k):
    """The saved exposure partition k (hash-bound to its record and the committed manifest) -> {fork key: row}."""
    rec = json.loads((EXPO / 'parts' / f'part_{k:03d}.json').read_text()); b = (EXPO / 'parts' / f'part_{k:03d}.jsonl.gz').read_bytes()
    C.require(hashlib.sha256(b).hexdigest() == rec['sha256'] and rec['manifest_sha256'] == C.sha(EXPO / 'manifest.json'), f'exposure part {k}: identity')
    return {(str(r['day']), int(r['m']), int(r['t_ms']), int(r['side'])): r for r in LS.gz_rows(b)}, rec['sha256']


def run_part(k, out=OUT):
    """One manifest partition -> out/parts/part_<k>.jsonl.gz + its record (header's identities checked first)."""
    out = Path(out); (out / 'parts').mkdir(parents=True, exist_ok=True); rp = out / 'parts' / f'part_{int(k):03d}.json'
    C.require(EX.committed_equal(EX.RECEIPT_REL, EXPO / 'manifest.json'), 'the exposure manifest is not the committed one')
    C.require(committed_equal('predictfun/research/pf_entry_retention.py', HERE / 'pf_entry_retention.py') and committed_equal(f'{RECEIPT}/declaration.md', REPO / RECEIPT / 'declaration.md'),
              'the module / declaration is not committed in HEAD: no measurement before the freeze')
    if rp.exists():
        rec = json.loads(rp.read_text()); C.require(rec.get('sources') == sources(), f'part {k}: an existing record of another identity'); return 'DONE'
    man = json.loads((EXPO / 'manifest.json').read_text()); pt = man['parts'][int(k)]; kind, day = pt['set'], pt['day']; S = man['sets'][kind]
    plan = json.loads((Path(EX.PLANS[kind]['dir']) / 'plan.json').read_text()); C.require(C.sha(Path(EX.PLANS[kind]['dir']) / 'plan.json') == S['plan_sha256'], 'not the manifest\'s plan')
    forks = {(f['day'], f['m'], f['t_ms'], f['side']): f for f in S['forks']}; members = pt['worlds']; saved, esha = exposure_rows(int(k))
    Mo, sup, _ = L.teacher_assets(); kp = Path(plan['kp_dir']); src = KP.Tape(); ff = EX.files_fn(kind)
    G = LS.Guard(600. + 120. * len(members), 120.); mon = LS.RN.Sampler(out / 'parts' / f'part_{int(k):03d}_status.json', LS.GUARD); mon.start()
    try:
        C.require(LS.identity_now(day, kp, src, ff, LS.hours_tracked) == plan['sample'][day]['inputs'], f'{day}: the inputs differ from the plan\'s')
        allw = {LS.ckey(LS.wkey(x)): x for x in LS.worlds_all(day, kp, keys=[(x['m'], x['t_ms']) for x in members])}
        C.require(sorted(allw) == sorted(LS.ckey(LS.wkey(x)) for x in members), 'a member world is not one of the day\'s worlds')
        mk = sorted({x['m'] for x in members}); fair, depth, hrs, trk = X.real_inputs(day, mk)
        C.require(LS.day_identity(day, kp, src, hrs, trk, ff) == plan['sample'][day]['inputs'], 'real_inputs\' hours / tracked differ from the plan')
        data = src.load(day, mk, src.offset(day)); prs = {int(W_[0]['m']): W_ for W_ in src.pairs(data)}; data = None; G.check('load'); rows = []
        for x in members:
            t0 = time.time(); aw = allw[LS.ckey(LS.wkey(x))]; world, wst = L.world_bind(day, dict(m=x['m'], t=aw['t'], o=aw['o']), prs, fair, depth, X.real_features)
            for s in (0, 1):
                f = forks[(x['day'], x['m'], x['t_ms'], s)]; key = (x['day'], int(x['m']), int(x['t_ms']), s)
                if world is None and f['status'] == 'ok':
                    rows.append(dict({k_: f[k_] for k_ in ('day', 'm', 't_ms', 'hour', 'side', 'oid')}, status=f'world:{wst}')); continue
                rows.append(fork_stages(f, world, Mo, sup, saved.get(key)) if f['status'] == 'ok' else fork_stages(f, None, Mo, sup, None))
            world = None; G.check('pair', time.time() - t0); G.check('world')
        b = LS.gz_bytes(rows); (out / 'parts' / f'part_{int(k):03d}.jsonl.gz').write_bytes(b)
        LS.RN.atomic_json(rp, dict(version=VERSION, part=int(k), set=kind, day=day, worlds=members, rows=len(rows), sha256=hashlib.sha256(b).hexdigest(), exposure_part_sha256=esha,
                                   manifest_sha256=C.sha(EXPO / 'manifest.json'), sources=sources(), elapsed_s=time.time() - G.t0, hwm_mib=LS.hwm(), timing=LS.timing(G.log), asof=C.stamp()))
        return 'PART_DONE'
    except LS._Latch as e:
        LS.RN.atomic_json(out / 'parts' / f'part_{int(k):03d}_status.json', dict(state='STOP_RESOURCE', part=int(k), latch=e.args[0], asof=C.stamp())); return 'STOP_RESOURCE'
    finally:
        mon.live = False


# ---------------------------------------------------------------------------------------------------------------------- the measurements
def stage_table(rows):
    """Fork rows -> one row per fork x stage (every status kept), with the entry admission, the targets and the entry -> review changes."""
    out = []
    for r in rows:
        base = {k: r.get(k) for k in ('day', 'm', 't_ms', 'hour', 'side')}; st = r.get('stages') or {}; ent = st.get('entry') or {}
        for nm in STAGES:
            x = st.get(nm) or {}; mo = x.get('moments') or {}; tg = x.get('target') or {}
            d = dict(base, stage=nm, fork_status=r['status'], stage_status=x.get('status', r['status']), dec_status=x.get('dec_status'), action=x.get('action'),
                     admission=ent.get('admission'), t=x.get('t'), effect=x.get('effect'), same_as_review=x.get('same_as_review'),
                     **{k: mo.get(k) for k in ('P', 'Eq', 'Eqm', 'V', 'edge', 'mech', 'q_fill', 'qm_fill', 'S', 'ok')}, q=tg.get('q'), value=tg.get('value'), qm=tg.get('qm'),
                     severe=tg.get('severe'), unavoidable=x.get('unavoidable'))
            if nm == 'first_review' and x.get('inputs') and ent.get('inputs'):
                a0, a1 = np.asarray(ent['inputs'], np.float32), np.asarray(x['inputs'], np.float32); ch = a0.view(np.uint32) != a1.view(np.uint32); m0 = ent.get('moments') or {}
                for gname in GROUPS:
                    d[f'chg_{gname}'] = int(sum(1 for i, n in enumerate(FEATS) if group_of(n) == gname and ch[i]))
                for k_ in ('P', 'Eq', 'Eqm', 'edge', 'V'):
                    d[f'd{k_}'] = (mo.get(k_) - m0[k_]) if (mo.get(k_) is not None and m0.get(k_) is not None) else None
                d['V_entry'] = m0.get('V'); d['flip'] = bool(m0.get('V') is not None and mo.get('V') is not None and m0['V'] > 0 and mo['V'] < 0)
            out.append(d)
    return pd.DataFrame(out)


def eligible(S, stage):
    return S[(S.stage == stage) & (S.fork_status == 'ok') & (S.stage_status == 'ok') & (S.dec_status == 'model') & S.V.notna()]


def calibration(S):
    out = []
    for nm in STAGES:
        E = eligible(S, nm); pool = float((E.q > 1e-12).mean()) if len(E) else None
        for (day, side), g in [(('all', 'all'), E)] + list(E.groupby(['day', 'side'], sort=True)):
            if not len(g):
                continue
            y = (g.q > 1e-12).astype(float); fl = g[g.q > 1e-12]
            out.append(dict(stage=nm, day=str(day), side=str(side), n=int(len(g)), filled=int(y.sum()), fill_rate=float(y.mean()), mean_P=float(g.P.mean()),
                            brier=float(((g.P - y) ** 2).mean()), brier_pooled_rate=float(((pool - y) ** 2).mean()), mean_Eq=float(g.Eq.mean()), mean_q=float(g.q.mean()),
                            mae_q=float((g.Eq - g.q).abs().mean()), mean_V=float(g.V.mean()), mean_value=float(g.value.mean()), mae_V=float((g.V - g.value).abs().mean()),
                            mae_zero=float(g.value.abs().mean()), mae_mech=float((g.mech - g.value).abs().mean()),
                            mae_q_fill=float((fl.q_fill - fl.q).abs().mean()) if len(fl) else None, mae_qm_fill=float((fl.qm_fill - fl.qm).abs().mean()) if len(fl) else None))
    return out


def bins(S):
    out = []
    for nm in STAGES:
        E = eligible(S, nm)
        for lo, hi in zip(PBINS[:-1], PBINS[1:]):
            g = E[(E.P >= lo) & ((E.P < hi) if hi < 1. else (E.P <= hi))]
            out.append(dict(stage=nm, p_lo=lo, p_hi=hi, n=int(len(g)), mean_P=float(g.P.mean()) if len(g) else None, fill_rate=float((g.q > 1e-12).mean()) if len(g) else None))
    return out


def decisions(S):
    out = []
    for nm in STAGES:
        E = S[(S.stage == nm) & (S.fork_status == 'ok') & (S.stage_status == 'ok') & S.value.notna()]
        E = E.assign(vsign=np.where(E.V.isna(), 'no_V', np.where(E.V.fillna(0) >= 0, 'V>=0', 'V<0')), adm=E.admission.fillna('none'))
        for (vs, adm), g in [(('all', 'all'), E)] + list(E.groupby(['vsign', 'adm'], sort=True)):
            v = g.value.astype(float); a = np.sort(v.abs().to_numpy())[::-1]
            out.append(dict(stage=nm, vsign=vs, admission=adm, n=int(len(g)), worlds=int(g[['day', 'm', 't_ms']].drop_duplicates().shape[0]), n_benefit=int((v > MEANINGFUL).sum()),
                            sum_benefit=float(v[v > MEANINGFUL].sum()), n_loss=int((v < -MEANINGFUL).sum()), sum_loss=float(v[v < -MEANINGFUL].sum()),
                            n_zero=int((v.abs() <= MEANINGFUL).sum()), n_severe=int((v < -SEVERE).sum()), top5_share_unweighted=float(a[:5].sum() / a.sum()) if a.sum() > 0 else None))
    return out


def selection_null(D, sel, draws=NULL_DRAWS, seed=SEED):
    """The selected rows' mean value against matched random draws of the same counts within day x side x UTC hour (header)."""
    rng = np.random.default_rng(seed); grp = [g for _, g in D.groupby(['day', 'side', 'hour'], sort=True)]
    n_sel = int(sel.sum()); ms = float(D.value[sel].mean()) if n_sel else None; ks = [int(sel[g.index].sum()) for g in grp]; vs = [g.value.to_numpy(float) for g in grp]
    deg = int(sum(1 for g, k_ in zip(grp, ks) if k_ in (0, len(g)))); null = np.empty(draws)
    for i in range(draws):
        null[i] = sum(v_[rng.choice(len(v_), k_, replace=False)].sum() for v_, k_ in zip(vs, ks) if k_) / n_sel if n_sel else np.nan
    return dict(n=int(len(D)), n_sel=n_sel, mean_sel=ms, mean_all=float(D.value.mean()) if len(D) else None, null_mean=float(np.nanmean(null)) if n_sel else None,
                null_p05=float(np.nanpercentile(null, 5)) if n_sel else None, null_p95=float(np.nanpercentile(null, 95)) if n_sel else None,
                percentile=float((null <= ms + 1e-12).mean()) if n_sel else None, strata=len(grp), degenerate_strata=deg, thin=bool(n_sel < THIN or len(D) - n_sel < THIN))


def selections(S):
    out = {}
    for i, (nm, act) in enumerate((('entry', 'WAIT'), ('first_review', 'PULL'))):
        D = eligible(S, nm).reset_index(drop=True); out[nm] = dict(selected_action=act, **selection_null(D, D.action == act, seed=SEED + i))
    return out


def changes(S):
    R = S[(S.stage == 'first_review') & (S.fork_status == 'ok') & (S.stage_status == 'ok')]
    R = R[R.get('chg_market_flow', pd.Series(dtype=float)).notna()] if 'chg_market_flow' in R else R.iloc[0:0]; out = []
    for (day, side), g in [(('all', 'all'), R)] + list(R.groupby(['day', 'side'], sort=True)):
        if not len(g):
            continue
        stable = g[(g.chg_market_flow == 0) & (g.chg_fair == 0)]
        out.append(dict(day=str(day), side=str(side), forks=int(len(g)), **{f'share_{gn}_changed': float((g[f'chg_{gn}'] > 0).mean()) for gn in GROUPS},
                        **{f'mean_changed_{gn}': float(g[f'chg_{gn}'].mean()) for gn in GROUPS}, mean_dV=float(g.dV.mean()), mean_V_entry=float(g.V_entry.mean()),
                        mean_V_review=float(g.V.mean()), flip_share=float(g.flip.mean()), stable_market_forks=int(len(stable)),
                        stable_market_flip_share=float(stable.flip.mean()) if len(stable) else None))
    return out


def predicates(S):
    """The declared predicates and readings (header)."""
    sel = selections(S); cal = {c['stage']: c for c in calibration(S) if c['day'] == 'all'}; ch = [c for c in changes(S) if c['day'] == 'all']
    disc = {nm: bool(sel[nm]['percentile'] is not None and not sel[nm]['thin'] and sel[nm]['percentile'] <= PCT) for nm in sel}
    fails = {nm: bool(nm in cal and cal[nm]['mae_V'] >= cal[nm]['mae_zero']) for nm in ('entry', 'first_review')}; flip = ch[0]['flip_share'] if ch else None
    rd = dict(entry_identifiable=disc['entry'], review_protective=disc['first_review'],
              forecasts_fail=bool(fails['entry'] or fails['first_review'] or (flip is not None and flip >= FLIP and not disc['first_review'])))
    rd['not_distinguishable'] = not any(rd.values())
    return dict(selection=sel, discriminates=disc, value_fails=fails, flip_share=flip, readings=rd)


def measure(out=OUT, man_path=EXPO / 'manifest.json'):
    out = Path(out); C.require(not (out / 'retention.json').exists(), 'the measurement exists (immutable)'); man = json.loads(Path(man_path).read_text()); rows = []
    for k, pt in enumerate(man['parts']):
        rec = json.loads((out / 'parts' / f'part_{k:03d}.json').read_text()); b = (out / 'parts' / f'part_{k:03d}.jsonl.gz').read_bytes()
        C.require(hashlib.sha256(b).hexdigest() == rec['sha256'] and rec['sources'] == sources() and rec['manifest_sha256'] == C.sha(man_path), f'part {k}: identity')
        rr = LS.gz_rows(b); C.require(len(rr) == 2 * len(pt['worlds']), f'part {k}: not two forks per world'); rows += rr
    S = stage_table(rows); S.to_csv(out / 'stages.csv', index=False)
    for name, fn in (('calibration', calibration), ('bins', bins), ('decisions', decisions), ('changes', changes)):
        pd.DataFrame(fn(S)).to_csv(out / f'{name}.csv', index=False)
    P = predicates(S); st = S.groupby(['stage', 'fork_status', 'stage_status']).size().rename('n').reset_index().to_dict('records')
    fs = pd.DataFrame([dict(day=r['day'], status=r['status']) for r in rows]).groupby(['day', 'status']).size().rename('n').reset_index().to_dict('records')
    js = dict(version=VERSION, forks=len(rows), fork_statuses=fs, stage_statuses=st, predicates=P, manifest_sha256=C.sha(man_path), sources=sources(), asof=C.stamp())
    (out / 'retention.json').write_text(json.dumps(js, indent=1, sort_keys=True, default=str) + '\n')
    print(json.dumps(dict(forks=len(rows), fork_statuses=fs, readings=P['readings'], discriminates=P['discriminates']), indent=1, default=str))


def selftest():
    import pf_entry_retention_selftest as ST
    ST.run(sys.modules[__name__])


if __name__ == '__main__':
    a = sys.argv[1:]
    if a == ['selftest']:
        selftest()
    elif a[:1] == ['run'] and len(a) == 2:
        print(run_part(int(a[1])))
    elif a == ['measure']:
        measure()
    else:
        sys.exit(__doc__)
