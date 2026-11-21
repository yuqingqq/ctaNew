"""ENTRY-EXPOSURE-v1: THE SAME INITIAL POST UNDER PI0 VERSUS KEEPING THAT INITIAL GENERATION (P-2026-005; 2026-11-21, session aea5bfb9). Review
3d8bf5ad's one next priority: does the current target / controller remove the initial order before useful execution develops, or does PI0's
early removal avoid adverse fills? A declared diagnostic continuation variant (multiplicity: one continuation contrast), on consumed dates only,
not a fitted candidate, a production minimum-hold timer or a profitable-policy claim. Declared and frozen before any branch result.
MANIFEST (build_manifest, frozen before simulation; its receipt copy predictfun/reviews/entry_exposure_20261121/manifest.json must be committed in
 HEAD and byte-equal to the data copy before any partition runs): Oct 24-25 from R2-F's accepted plan / dataset and Oct 26-27 from R3's (their
 recorded day identities: real_files for R2-F, the day-guarded files for R3). Per day and UTC hour, the plan's selected worlds ranked by
 sha256('entry_exposure_v1|' + the canonical JSON of {day, m, t_ms}), ties by that JSON; the four lowest, or all if fewer (at most 384 worlds,
 768 side forks). For each world and side the saved original-time neutral mandatory POST row (pf_r2_label_sample.is_mandatory): its fork key,
 fingerprint and stored arms; a missing world path, an absent mandatory row or any non-ok row is its own status. Per-hour population counts
 and inclusion fractions are recorded. No outcome, prediction or first-pull result enters the selection.
THREE BRANCHES from each rebuilt fork (pf_r2_labels.fork_for; its fingerprint must equal the saved one, else fork_mismatch):
 - WAIT_PI0 and POST_PI0: pf_r2_labels.pair_bound under the frozen pi0 (audit every 35). Both arms' fills / decisions hashes, values and Y
   must reproduce the stored label (fills / decisions / requests hashes, values, Y; else reproduction_failed; never skipped);
 - POST_KEEP_INITIAL: the identical forced POST, then the frozen PI0 everywhere except that a PULL of the forced initial generation (the live
   focal generation whose gid is the executed fork POST's) is overridden to KEEP; the shadow PI0 decision is logged with the override. The
   initial generation stays subject to post-only refusal, fills, the engine's client structural rule, pending / acknowledgement timing, caps
   and END; once it closes, PI0 governs focal re-entry and later generations, and the opposite side always. When no override happened the arm
   must equal POST_PI0 exactly (keep_baseline_mismatch otherwise); with an override, the decisions before the first override and the fills
   up to its tick must equal POST_PI0's, and the first difference must be that override (KEEP for PI0's PULL of gid0 at the same tick;
   prefix_mismatch otherwise). The KEEP arm's ledger reconciles (not_reconciled), its mark exists (missing_mark), its gid0 is POST_PI0's
   (initial_gid_mismatch) and the saved fork state is unchanged after it runs.
VALUES (common END mark, gross): Y_PI0 = R(POST_PI0) - R(WAIT_PI0) (the stored label); Y_KEEP_INITIAL = R(POST_KEEP_INITIAL) - R(WAIT_PI0);
 D_EXPOSURE = R(POST_KEEP_INITIAL) - R(POST_PI0) (the primary paired comparison); Y_KEEP_INITIAL = Y_PI0 + D_EXPOSURE is verified. Per POST arm
 the initial generation's evidence: activation, removal effect and its reason (policy / structural / refusal / END / full fill), active time,
 filled quantity, first and full fill times, the END value of its fills (V_initial) and of the rest, open inventory and the unfilled remainder at
 END; for each of its fills the execution edge sign x (mark - price) and the venue-mid markouts at +50 / +100 / +1,000 ms when the reference is
 inside the episode (else missing), with their quantity denominators; the initial order's request / effect times and its final queue-ahead
 (venue state, a diagnostic, never a policy input). Each POST arm's END value splits by fills into the initial generation, the focal side's
 other generations and the opposite side: D_EXPOSURE = D_initial + D_focal_rest + D_opposite (components_mismatch otherwise), D_rest =
 D_EXPOSURE - D_initial. Raw fills, decisions, requests and order metadata of the three arms are retained.
READINGS (summaries): per day / side, equal-world finite-sample means over common evaluable forks, with denominators and every status; no
 interval, no significance claim, no favorable-side selection. More fills alone is not success.
PREDICATES (predicates(), declared here): per day over both sides' common evaluable forks, the sign of mean D_EXPOSURE (|mean| <= 1e-9 is 0);
 additional-fill forks: KEEP's initial generation filled more than PI0's. diagnosis: incomplete when a declared day has no evaluable fork;
 not_identified_thin when the pooled additional-fill forks are fewer than 30; not_identified_days_disagree unless every day has the same nonzero
 sign; else keep_better (positive on each day) or pi0_better (negative on each). more_execution: the pooled initial fill probability KEEP -
 PI0 >= 0.05; initial_carries: |sum D_initial| >= |sum D_rest| pooled; extra_fills_harmful: the mean D_initial over the pooled additional-fill
 forks is negative.
Usage (run_light): pf_entry_exposure.py selftest | manifest | run <part> | summarize"""
import copy
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pf_adv2 as A                                      # noqa: E402  frozen: readers
import pf_adv2_entry_eval as NE                          # noqa: E402  frozen: R3's dataset id
import pf_adv2_eval as AE                                # noqa: E402  frozen: R3's evaluation plan
import pf_adv2_replay as AR                              # noqa: E402  frozen: mid_at
import pf_adv2_sample as AS                              # noqa: E402  frozen: the day-guarded files

C, pd, T, LS, L, R2, KP = A.C, A.pd, A.T, A.LS, A.L, A.R2, A.KP
BE, X = L.BE, LS.X
VERSION = 'entry_exposure_v1'
PREFIX, PER_HOUR, PART, AUDIT_EVERY, NT, END = 'entry_exposure_v1|', 4, 32, 35, L.NT, float(L.END)
MARKOUTS = (50., 100., 1000.)
DAYS = ('20261024', '20261025', '20261026', '20261027')
MORE_EXECUTION, SIGN_TOL, THIN, MEANINGFUL = .05, 1e-9, 30, 1e-6
OUT = A.DATA / 'entry_exposure_v1_20261121'
PLANS = dict(train=dict(dir=A.TRAIN['dir'], dataset=A.TRAIN['dataset_sha256'], files='real_files'),
             eval=dict(dir=AE.EVAL_DIR, dataset=NE.R3_DATASET, files='guarded_files'))
RECEIPT_REL = 'predictfun/reviews/entry_exposure_20261121/manifest.json'
SOURCES = ('pf_entry_exposure.py', 'pf_r2_labels.py', 'pf_r2_label_sample.py', 'pf_branch_engine.py', 'pf_r2_teacher.py', 'pf_r2_shadow.py', 'pf_adv2_sample.py')


def sources():
    return {n: C.sha(HERE / n) for n in SOURCES}


def committed_equal(rel, local):
    """HEAD holds rel (repo-relative) and it equals local byte for byte."""
    import subprocess
    r = subprocess.run(['git', '-C', str(HERE.parents[1]), 'show', f'HEAD:{rel}'], capture_output=True)
    return r.returncode == 0 and r.stdout == Path(local).read_bytes()


def files_fn(kind):
    return AS.guarded_files if PLANS[kind]['files'] == 'guarded_files' else LS.real_files


def rank_of(w):
    return hashlib.sha256((PREFIX + LS.ckey(dict(day=str(w['day']), m=int(w['m']), t_ms=int(w['t_ms'])))).encode()).hexdigest()


def select_worlds(plan, per_hour=PER_HOUR):
    """Per day and UTC hour the four lowest-ranked selected worlds of a plan -> (selected, per-hour population / inclusion)."""
    sel, pop = [], []
    for day in plan['days']:
        by = {}
        for w in plan['sample'][day]['selected']:
            by.setdefault(int(w['hour']), []).append(dict(w, day=str(day)))
        for h in sorted(by):
            ws = sorted(by[h], key=lambda w: (rank_of(w), LS.ckey(dict(day=w['day'], m=int(w['m']), t_ms=int(w['t_ms'])))))[:per_hour]
            sel += ws; pop.append(dict(day=str(day), hour=h, population=len(by[h]), selected=len(ws), inclusion=len(ws) / len(by[h])))
    return sel, pop


def build_manifest(out=OUT):
    """The frozen manifest (header) -> out/manifest.json. Reads each accepted dataset's mandatory POST rows by key; never an outcome to select."""
    out = Path(out); mp = out / 'manifest.json'; C.require(not mp.exists(), 'the manifest exists (immutable)'); out.mkdir(parents=True, exist_ok=True); man = dict(sets={})
    for kind, p in PLANS.items():
        plan = json.loads((Path(p['dir']) / 'plan.json').read_text()); sel, pop = select_worlds(plan)
        want = {(w['day'], int(w['m']), int(w['t_ms'])) for w in sel}; rows = {}
        for r in A.dataset_rows(p['dir'], p['dataset']):
            key = (str(r['day']), int(r['m']), int(r['t_ms']))
            if key in want and r.get('head') == 'post':                                     # only the fields the manifest records
                res = r.get('result') or {}; keep = ('value', 'fills_sha256', 'decisions_sha256', 'requests_sha256')
                rows.setdefault((key, int(r['side'])), []).append(dict({k_: r.get(k_) for k_ in ('status', 'pick', 'row_key', 'oid', 'fork_fingerprint')},
                                                                       result=dict(Y=res.get('Y'), arms={n: {h: a.get(h) for h in keep} for n, a in (res.get('arms') or {}).items()})))
        forks = []
        for w in sel:
            key = (w['day'], int(w['m']), int(w['t_ms']))
            for s in (0, 1):
                rs = rows.get((key, s), []); mall = [r for r in rs if r.get('pick') == 'mandatory']
                mand = [r for r in mall if r['status'] == 'ok' and LS.is_mandatory(r['row_key'])]
                f = dict(day=w['day'], m=int(w['m']), t_ms=int(w['t_ms']), hour=int(w['hour']), side=s, oid=None)
                if len(mand) == 1 and len(mall) == 1:
                    r = mand[0]; res = r['result']
                    f.update(status='ok', oid=int(r['oid']), sel=dict(r['row_key'], day=w['day'], side=s), fork_fingerprint=r['fork_fingerprint'], Y=res['Y'],
                             arms={n: dict(value=a['value'], fills_sha256=a['fills_sha256'], decisions_sha256=a['decisions_sha256'], requests_sha256=a['requests_sha256'])
                                   for n, a in res['arms'].items()})
                else:
                    st = sorted({str(r['status']) for r in rs}); wst = [x for x in st if x.startswith('world:')]
                    f.update(status=wst[0] if wst else 'no_mandatory_row' if not mall else 'mandatory_row_not_unique' if len(mall) > 1 else
                             'mandatory_key_mismatch' if mall[0]['status'] == 'ok' else f"mandatory_{mall[0]['status']}", row_statuses=st)
                forks.append(f)
        parts = []
        for day in plan['days']:
            ws = sorted([w for w in sel if w['day'] == str(day)], key=lambda w: (int(w['m']), int(w['t_ms'])))
            parts += [dict(day=str(day), worlds=[dict(day=w['day'], m=int(w['m']), t_ms=int(w['t_ms'])) for w in ws[i:i + PART]]) for i in range(0, len(ws), PART)]
        man['sets'][kind] = dict(plan_sha256=C.sha(Path(p['dir']) / 'plan.json'), dataset_sha256=p['dataset'], files=p['files'], population=pop, forks=forks, parts=parts)
    man['parts'] = [dict(set=k, **pt) for k in ('train', 'eval') for pt in man['sets'][k]['parts']]
    man.update(version=VERSION, per_hour=PER_HOUR, prefix=PREFIX, sources=sources(), asof=C.stamp())
    mp.write_text(json.dumps(man, sort_keys=True, separators=(',', ':'), default=str) + '\n')
    print(json.dumps({k: dict(worlds=len({(f['day'], f['m'], f['t_ms']) for f in v['forks']}), forks=len(v['forks']),
                              ok=sum(f['status'] == 'ok' for f in v['forks']), parts=len(v['parts'])) for k, v in man['sets'].items()}, indent=1))


class KeepInitial(L.ArmTeacher):
    """POST_KEEP_INITIAL's continuation (header): the inner policy's decision, except a PULL of the live focal generation gid0 becomes KEEP
    (logged with the shadow decision). gid0 is set from the executed fork POST; until then (None) nothing is overridden."""

    def __init__(self, inner, side, causal=None):
        super().__init__(None, None, causal=causal); self.inner, self.focal, self.gid0, self.overrides, self.name, self.ahead0 = inner, int(side), None, 0, 'KEEP_INITIAL', None

    def bind(self, B, world):
        super().bind(B, world); self.inner.bind(B, world)

    def decide(self, v, B):
        d = self.inner.decide(v, B)
        if self.gid0 is not None and v['state'] == 'live' and int(v['side']) == self.focal and v['gid'] == self.gid0 and d.get('action') == 'PULL':
            self.overrides += 1
            return dict(d, action='KEEP', status='keep_initial_override', shadow_action='PULL', shadow_status=d.get('status'))
        return d


def keep_factory(inner_factory, side, every=None):
    """A factory of KeepInitial policies around a fresh inner policy per branch (each bound to that branch)."""
    made = []

    def make(B):
        probe, st = T.causal_probe(every) if every else (None, None)
        inner = inner_factory(B); p = KeepInitial(inner, side, causal=probe); p.bind(B, getattr(inner, 'w', None)); made.append((p, st)); return p
    make.made = made
    return make


def run_arm(ctx, factory, keep=False):
    """One arm from a saved fork, as pf_r2_labels.pair_bound runs it (forced POST at k, the other side at k when it acts after, steps to END);
    with keep, the KeepInitial policy's gid0 is set from the executed fork POST -> (branch, ledger, policy)."""
    k, side, ep = int(ctx['k']), int(ctx['side']), int(ctx['episode']); g = k * L.STEP; first = (ep + k) % 2
    Bx = copy.deepcopy(ctx['state']); pol = factory(Bx); C.require(getattr(pol, 'horizon', None) == END, 'the continuation does not declare END')
    BE.decide(Bx, g, side, pol, forced='POST')
    if keep:
        pol.gid0 = Bx.open[side]; C.require(pol.gid0 is not None, 'the forced POST opened no generation'); pol.ahead0 = float(Bx.E.o[pol.gid0]['ahead'])
    if first == side:
        BE.decide(Bx, g, 1 - side, pol)
    Bx.check_position(g)
    for kk in range(k + 1, NT):
        BE.step(Bx, kk, pol)
    return Bx, BE.finish(Bx), pol


def ledger(B, vals, pol, cst):
    """An arm's retained evidence (header). vals: the arm's value / cash / position / open_at_end / hashes (pair_bound's arm or finish's ledger);
    the branch B is read, never finished again."""
    jl = lambda x: json.loads(json.dumps(list(x), default=str))
    return dict(fill_list=jl(B.E.fills), decision_list=jl(B.decisions), fallback_list=jl(B.fallbacks), requests=jl([e_ for e_ in B.E.events if e_[1].endswith('_request')]),
                value=float(vals['value']), cash=float(vals['cash']), position=float(vals['position']), mark=float(B.mark), end=float(B.end), settings=dict(B.settings),
                open_at_end=vals.get('open_at_end'),
                orders=[dict(gid=o['gid'], side=o['side'], price=o['price'], size=o['size'], rem=o['rem'], act=o['act'], remove=o['remove'], submit=o['submit'],
                             refused=bool(o['refused']), req=o['req'], why=o['why'], ahead=o['ahead']) for o in B.E.o],
                fills_sha256=vals['fills_sha256'], decisions_sha256=vals['decisions_sha256'], teacher=dict(calls=len(pol.log), binding_checked=pol.bind_checked, binding_bad=pol.bind_bad),
                causal={k_: v_ for k_, v_ in (cst or {}).items() if k_ not in ('details', 'n')})


def initial(B, gid0, side):
    """The forced initial generation's evidence in one POST arm (header)."""
    o = B.E.o[gid0]; mark = float(B.mark); sg = 1. if side == 0 else -1.; fl = [f for f in B.E.fills if int(f[1]) == gid0]; q = 0.; full = None
    for f in sorted(fl):
        q += float(f[3])
        if full is None and q >= float(o['size']) - 1e-9:
            full = float(f[0])
    rem_t = min(float(o['remove']), full if full is not None else np.inf, END)
    why = ('refused' if o['refused'] else 'full_fill' if full is not None and full <= float(o['remove']) and full <= END else
           (o['why'] or 'end') if float(o['remove']) <= END else 'end')
    fills = []
    for f in fl:
        t, p, qq = float(f[0]), float(f[4]), float(f[3]); mk = {}
        for h in MARKOUTS:
            m_ = AR.mid_at(B.E.f, t + h) if t + h <= END else None; mk[f'm{int(h)}'] = None if m_ is None else sg * (m_ - p) * qq
        fills.append(dict(t=t, q=qq, price=p, edge=sg * (mark - p) * qq, **mk))
    return dict(gid0=int(gid0), act=float(o['act']), req=None if o['req'] is None else float(o['req']), remove=float(o['remove']), why=why, refused=bool(o['refused']),
                ahead_final=float(o['ahead']), filled=q, size=float(o['size']),
                first_fill=min((float(f[0]) for f in fl), default=None), full_fill=full, active_ms=max(0., rem_t - float(o['act'])) if not o['refused'] else 0.,
                v_initial=sum(x['edge'] for x in fills), unfilled_at_end=float(o['rem']) if (float(o['remove']) > END and not o['refused']) else 0., fills=fills)


def components(B, gid0, side):
    """One POST arm's END value by its fills (sign x q x (mark - price)): the initial generation gid0, the focal side's other generations, the
    opposite side."""
    mark = float(B.mark); out = dict(v_initial=0., v_focal_rest=0., v_opposite=0.)
    for f in B.E.fills:
        out['v_initial' if int(f[1]) == int(gid0) else 'v_focal_rest' if int(f[2]) == int(side) else 'v_opposite'] += (1. if int(f[2]) == 0 else -1.) * float(f[3]) * (mark - float(f[4]))
    return out


def values(rw, rp, rk, cp, ck):
    """The declared contrasts from the arms' END values (WAIT_PI0, POST_PI0, POST_KEEP_INITIAL) and the two POST arms' components (cp in POST_PI0,
    ck in POST_KEEP_INITIAL): Y_PI0, Y_KEEP_INITIAL, D_EXPOSURE = D_initial + D_focal_rest + D_opposite, D_rest, and the identities."""
    y, yk, d = rp - rw, rk - rw, rk - rp; di, df, do = (ck[k] - cp[k] for k in ('v_initial', 'v_focal_rest', 'v_opposite'))
    return dict(Y_PI0=y, Y_KEEP_INITIAL=yk, D_EXPOSURE=d, D_initial=di, D_rest=d - di, D_focal_rest=df, D_opposite=do, identity_ok=bool(abs(yk - (y + d)) <= 1e-9),
                components_ok=bool(abs(d - (di + df + do)) <= 1e-6 and abs(rp - sum(cp.values())) <= 1e-6 and abs(rk - sum(ck.values())) <= 1e-6))


def prefix_ok(Bk, Bp, pk):
    """The KEEP arm equals POST_PI0 up to its first override (header); without an override, identical decisions and fills."""
    dk, dp = [list(d) for d in Bk.decisions], [list(d) for d in Bp.decisions]; ov = [d for d in pk.log if d.get('status') == 'keep_initial_override']
    if not ov:
        return bool(dk == dp and list(Bk.E.fills) == list(Bp.E.fills))
    i = next((j for j, (a, b) in enumerate(zip(dk, dp)) if a != b), None)
    if i is None:
        return False
    a, b, g = dk[i], dp[i], float(ov[0]['t'])
    return bool(a[0] == b[0] == g and a[1] == b[1] == pk.focal and a[2] == b[2] and a[3] == 'KEEP' and b[3] == 'PULL' and a[4] == b[4] == pk.gid0
                and [f for f in Bk.E.fills if f[0] <= g] == [f for f in Bp.E.fills if f[0] <= g])


def fork_package(f, world, M, sup):
    """One manifest fork -> its three branches and readings (header)."""
    r = dict({k: f[k] for k in ('day', 'm', 't_ms', 'hour', 'side', 'oid')}, status=f['status'])
    if f['status'] != 'ok':
        return r
    s = int(f['side']); ctx = L.fork_for(f['sel'], world, M, sup)
    if ctx['fingerprint'] != f['fork_fingerprint']:
        return dict(r, status='fork_mismatch')
    t0 = time.time(); fac = L.teacher_factory(M, sup, world, every=AUDIT_EVERY); res = L.pair_bound(ctx, fac, 'post')
    if res.get('status') != 'ok':
        return dict(r, status='pair_not_ok', pair_status=res.get('status'))
    LS.arm_ledgers(fac, res); Bp, Bw = fac.made[0][0].B, fac.made[1][0].B                  # the bound branches are the arms (else it raises)
    repro = bool(res['Y'] == f['Y'] and res['reconciled'] and all(res['arms'][n][h] == f['arms'][n][h] for n in ('POST', 'WAIT')
                                                                 for h in ('fills_sha256', 'decisions_sha256', 'requests_sha256', 'value')))

    def inner(B):
        t_ = L.ArmTeacher(M, sup); t_.bind(B, world); return t_
    kf = keep_factory(inner, s, every=AUDIT_EVERY); Bk, Lk, pk = run_arm(ctx, kf, keep=True)
    C.require(BE.fingerprint(ctx['state']) == ctx['fingerprint'], 'the saved fork state changed while the KEEP arm ran')
    ahead0 = kf.made[0][0].ahead0
    gid_p = [d for d in Bp.decisions if float(d[0]) == ctx['k'] * L.STEP and int(d[1]) == s and d[3] == 'POST'][0][4]
    if Lk.get('status') != 'ok':
        return dict(r, status='missing_mark', reproduced=repro)
    lk = ledger(Bk, dict(Lk, fills_sha256=L._h(Lk['fills']), decisions_sha256=L._h(Lk['decisions'])), pk, kf.made[0][1])
    lp = ledger(Bp, res['arms']['POST'], fac.made[0][0], fac.made[0][1]); lw = ledger(Bw, res['arms']['WAIT'], fac.made[1][0], fac.made[1][1])
    base_eq = bool(pk.overrides > 0 or (lk['fills_sha256'] == lp['fills_sha256'] and lk['decisions_sha256'] == lp['decisions_sha256'] and lk['value'] == lp['value']))
    checks = dict(reproduced=repro, keep_baseline_equal=base_eq, prefix_ok=prefix_ok(Bk, Bp, pk), keep_reconciled=bool(L.reconcile_arm(Lk, Bk)),
                  initial_gid_equal=bool(pk.gid0 == gid_p))
    st = next((n for k_, n in (('reproduced', 'reproduction_failed'), ('keep_baseline_equal', 'keep_baseline_mismatch'), ('prefix_ok', 'prefix_mismatch'),
                                ('keep_reconciled', 'not_reconciled'), ('initial_gid_equal', 'initial_gid_mismatch')) if not checks[k_]), 'ok')
    ip, ik = initial(Bp, int(gid_p), s), initial(Bk, int(pk.gid0), s); cp, ck = components(Bp, gid_p, s), components(Bk, pk.gid0, s)
    v = values(lw['value'], lp['value'], lk['value'], cp, ck)
    if st == 'ok' and not (v['components_ok'] and abs(cp['v_initial'] - ip['v_initial']) <= 1e-9 and abs(ck['v_initial'] - ik['v_initial']) <= 1e-9):
        st = 'components_mismatch'
    return dict(r, status=st, **checks, overrides=int(pk.overrides), Y_stored=f['Y'], **v, comp_PI0=cp, comp_KEEP=ck, init_PI0=ip, init_KEEP=ik,
                shadow=[dict(t=d['t'], gid=d['gid'], shadow_action=d['shadow_action'], shadow_status=d.get('shadow_status'), shadow_V=d.get('V'))
                        for d in pk.log if d.get('status') == 'keep_initial_override'],
                ahead_at_submit=ahead0, arms=dict(WAIT_PI0=lw, POST_PI0=lp, POST_KEEP_INITIAL=lk), seconds=time.time() - t0)


def run_part(k, out=OUT):
    """One manifest partition -> out/parts/part_<k>.jsonl.gz + its record (identity checked against the committed manifest and the plan)."""
    out = Path(out); C.require(committed_equal(RECEIPT_REL, out / 'manifest.json'), 'the manifest is not the one committed in HEAD: no simulation before the freeze')
    man = json.loads((out / 'manifest.json').read_text()); pt = man['parts'][int(k)]; kind = pt['set']; day = pt['day']; S = man['sets'][kind]
    (out / 'parts').mkdir(exist_ok=True); rp = out / 'parts' / f'part_{int(k):03d}.json'
    if rp.exists():
        rec = json.loads(rp.read_text()); C.require(rec.get('manifest_sha256') == C.sha(out / 'manifest.json') and rec.get('sources') == sources(), f'part {k}: an existing record of another identity')
        return 'DONE'
    plan = json.loads((Path(PLANS[kind]['dir']) / 'plan.json').read_text()); C.require(C.sha(Path(PLANS[kind]['dir']) / 'plan.json') == S['plan_sha256'], 'not the manifest\'s plan')
    C.require(kind != 'eval' or S['plan_sha256'] == AE.EVAL_PLAN_SHA, 'not the frozen evaluation plan')
    forks = {(f['day'], f['m'], f['t_ms'], f['side']): f for f in S['forks']}; members = pt['worlds']
    M, sup, _ = L.teacher_assets(); kp = Path(plan['kp_dir']); src = KP.Tape(); ff = files_fn(kind)
    G = LS.Guard(600. + 360. * len(members), 120.); mon = LS.RN.Sampler(out / 'parts' / f'part_{int(k):03d}_status.json', LS.GUARD); mon.start()
    try:
        C.require(LS.identity_now(day, kp, src, ff, LS.hours_tracked) == plan['sample'][day]['inputs'], f'{day}: the inputs differ from the plan\'s')
        allw = {LS.ckey(LS.wkey(x)): x for x in LS.worlds_all(day, kp, keys=[(x['m'], x['t_ms']) for x in members])}
        C.require(sorted(allw) == sorted(LS.ckey(LS.wkey(x)) for x in members), 'a member world is not one of the day\'s worlds')
        mk = sorted({x['m'] for x in members}); fair, depth, hrs, trk = X.real_inputs(day, mk)
        C.require(LS.day_identity(day, kp, src, hrs, trk, ff) == plan['sample'][day]['inputs'], 'real_inputs\' hours / tracked differ from the plan')
        data = src.load(day, mk, src.offset(day)); prs = {int(W_[0]['m']): W_ for W_ in src.pairs(data)}; data = None; G.check('load')
        rows = []
        for x in members:
            aw = allw[LS.ckey(LS.wkey(x))]; w = dict(m=x['m'], t=aw['t'], o=aw['o']); world, wst = L.world_bind(day, w, prs, fair, depth, X.real_features)
            for s in (0, 1):
                f = forks[(x['day'], x['m'], x['t_ms'], s)]
                if world is None and f['status'] == 'ok':
                    rows.append(dict({k_: f[k_] for k_ in ('day', 'm', 't_ms', 'hour', 'side', 'oid')}, status=f'world:{wst}')); continue
                t0 = time.time(); r = fork_package(f, world, M, sup); rows.append(r); G.check('pair', time.time() - t0)
            world = None; G.check('world')
        b = LS.gz_bytes(rows); (out / 'parts' / f'part_{int(k):03d}.jsonl.gz').write_bytes(b)
        LS.RN.atomic_json(rp, dict(version=VERSION, part=int(k), set=kind, day=day, worlds=members, rows=len(rows), sha256=hashlib.sha256(b).hexdigest(),
                                   manifest_sha256=C.sha(out / 'manifest.json'), sources=sources(), elapsed_s=time.time() - G.t0, hwm_mib=LS.hwm(), timing=LS.timing(G.log), asof=C.stamp()))
        return 'PART_DONE'
    except LS._Latch as e:
        LS.RN.atomic_json(out / 'parts' / f'part_{int(k):03d}_status.json', dict(state='STOP_RESOURCE', part=int(k), latch=e.args[0], asof=C.stamp())); return 'STOP_RESOURCE'
    finally:
        mon.live = False


def flat(r):
    """One fork row -> a flat reading row (the raw arms stay in the parts)."""
    out = {k: r.get(k) for k in ('day', 'm', 't_ms', 'hour', 'side', 'status', 'reproduced', 'keep_baseline_equal', 'prefix_ok', 'keep_reconciled', 'initial_gid_equal',
                                 'components_ok', 'overrides', 'Y_PI0', 'Y_stored', 'Y_KEEP_INITIAL', 'D_EXPOSURE', 'D_initial', 'D_rest', 'D_focal_rest', 'D_opposite',
                                 'identity_ok', 'ahead_at_submit')}
    for b in ('PI0', 'KEEP'):
        i = r.get(f'init_{b}') or {}
        out.update({f'{b}_{k}': i.get(k) for k in ('why', 'refused', 'filled', 'size', 'first_fill', 'full_fill', 'active_ms', 'v_initial', 'unfilled_at_end', 'act', 'req',
                                                    'remove', 'ahead_final')})
        fl = i.get('fills') or []
        for h in ('m50', 'm100', 'm1000'):
            vals = [x[h] for x in fl if x.get(h) is not None]; out[f'{b}_{h}_sum'] = float(sum(vals)) if vals else None; out[f'{b}_{h}_fills'] = len(vals)
            out[f'{b}_{h}_q'] = float(sum(x['q'] for x in fl if x.get(h) is not None))
        out[f'{b}_fills'] = len(fl)
    for n, a in (r.get('arms') or {}).items():
        out[f'{n}_value'] = a['value']; out[f'{n}_position'] = a['position']
    return out


def summaries(F):
    """Per day / side over the common evaluable forks (status ok): equal-world means and sums, both branches (header)."""
    ok = F[F.status == 'ok']; out = []
    for (day, s), g in ok.groupby(['day', 'side'], sort=True):
        d = g.D_EXPOSURE.astype(float); row = dict(day=day, side=int(s), forks=int(len(g)), overrides_forks=int((g.overrides > 0).sum()))
        row.update(mean_Y_PI0=float(g.Y_PI0.mean()), mean_Y_KEEP=float(g.Y_KEEP_INITIAL.mean()), sum_Y_PI0=float(g.Y_PI0.sum()), sum_Y_KEEP=float(g.Y_KEEP_INITIAL.sum()),
                   mean_D=float(d.mean()), sum_D=float(d.sum()), mean_D_initial=float(g.D_initial.mean()), mean_D_rest=float(g.D_rest.mean()),
                   sum_D_pos=float(d[d > 0].sum()), sum_D_neg=float(d[d < 0].sum()), n_D_pos=int((d > 1e-6).sum()), n_D_neg=int((d < -1e-6).sum()),
                   n_D_zero=int((d.abs() <= 1e-6).sum()), worst_D=float(d.min()), best_D=float(d.max()), n_abs_D_gt30=int((d.abs() > 30).sum()),
                   worst_Y_PI0=float(g.Y_PI0.min()), worst_Y_KEEP=float(g.Y_KEEP_INITIAL.min()), n_abs_Y_PI0_gt30=int((g.Y_PI0.abs() > 30).sum()),
                   n_abs_Y_KEEP_gt30=int((g.Y_KEEP_INITIAL.abs() > 30).sum()), n_meaningful_Y_PI0=int((g.Y_PI0.abs() > MEANINGFUL).sum()),
                   n_meaningful_Y_KEEP=int((g.Y_KEEP_INITIAL.abs() > MEANINGFUL).sum()), mean_D_focal_rest=float(g.D_focal_rest.mean()), mean_D_opposite=float(g.D_opposite.mean()))
        ad = g[g.KEEP_filled > g.PI0_filled + 1e-9]
        row.update(addl_fill_forks=int(len(ad)), mean_D_addl=float(ad.D_EXPOSURE.mean()) if len(ad) else None, mean_D_initial_addl=float(ad.D_initial.mean()) if len(ad) else None,
                   n_D_initial_neg_addl=int((ad.D_initial < -MEANINGFUL).sum()))
        for b, arm in (('PI0', 'POST_PI0'), ('KEEP', 'POST_KEEP_INITIAL')):
            filled = g[f'{b}_filled'] > 1e-9; ff = g.loc[filled, f'{b}_first_fill'].astype(float)
            row.update({f'fill_prob_{b}': float(filled.mean()), f'mean_filled_{b}': float(g[f'{b}_filled'].mean()), f'mean_active_{b}': float(g[f'{b}_active_ms'].mean()),
                        f'mean_first_fill_{b}': float(ff.mean()) if len(ff) else None, f'median_first_fill_{b}': float(ff.median()) if len(ff) else None,
                        f'mean_vinit_{b}': float(g[f'{b}_v_initial'].mean()), f'unfilled_end_{b}': float(g[f'{b}_unfilled_at_end'].mean()),
                        f'open_inv_{b}': float(g[f'{arm}_position'].abs().mean())})
            for h in ('m50', 'm100', 'm1000'):
                row.update({f'sum_{h}_{b}': float(g[f'{b}_{h}_sum'].fillna(0).sum()), f'{h}_fills_{b}': int(g[f'{b}_{h}_fills'].sum()), f'{h}_q_{b}': float(g[f'{b}_{h}_q'].sum())})
            row[f'fills_{b}'] = int(g[f'{b}_fills'].sum()); row[f'q_filled_{b}'] = float(g[f'{b}_filled'].sum())
        out.append(row)
    return out


def reasons(F):
    """Per day / side / branch: the initial generation's termination reasons over the evaluable forks."""
    ok = F[F.status == 'ok']; out = []
    for (day, s), g in ok.groupby(['day', 'side'], sort=True):
        for b in ('PI0', 'KEEP'):
            for why, n in sorted(g[f'{b}_why'].value_counts().items()):
                out.append(dict(day=day, side=int(s), branch=b, why=str(why), n=int(n), share=float(n / len(g))))
    return out


def predicates(F):
    """The declared predicates (header) from the flat fork rows."""
    ok = F[F.status == 'ok']; days = {}
    for day in DAYS:
        g = ok[ok.day.astype(str) == day]
        if not len(g):
            continue
        m = float(g.D_EXPOSURE.astype(float).mean())
        days[day] = dict(forks=int(len(g)), mean_D=m, sign=0 if abs(m) <= SIGN_TOL else (1 if m > 0 else -1), mean_D_initial=float(g.D_initial.mean()),
                         mean_D_rest=float(g.D_rest.mean()), fill_prob_PI0=float((g.PI0_filled > 1e-9).mean()), fill_prob_KEEP=float((g.KEEP_filled > 1e-9).mean()))
    sg = [days[d]['sign'] for d in DAYS if d in days]; ad = ok[ok.KEEP_filled > ok.PI0_filled + 1e-9]
    diag = ('incomplete' if len(days) != len(DAYS) else 'not_identified_thin' if len(ad) < THIN else 'keep_better' if all(x > 0 for x in sg) else
            'pi0_better' if all(x < 0 for x in sg) else 'not_identified_days_disagree')
    fp, fk = float((ok.PI0_filled > 1e-9).mean()) if len(ok) else None, float((ok.KEEP_filled > 1e-9).mean()) if len(ok) else None
    si, sr = float(ok.D_initial.sum()), float(ok.D_rest.sum())
    return dict(days=days, diagnosis=diag, fill_prob_PI0=fp, fill_prob_KEEP=fk, more_execution=bool(fp is not None and fk - fp >= MORE_EXECUTION),
                sum_D_initial=si, sum_D_rest=sr, initial_carries=bool(abs(si) >= abs(sr)), forks=int(len(ok)), additional_fill_forks=int(len(ad)),
                mean_D_initial_additional=float(ad.D_initial.mean()) if len(ad) else None, share_D_initial_neg_additional=float((ad.D_initial < -MEANINGFUL).mean()) if len(ad) else None,
                extra_fills_harmful=bool(len(ad) and float(ad.D_initial.mean()) < 0))


def summarize(out=OUT):
    out = Path(out); man = json.loads((out / 'manifest.json').read_text()); rows = []
    for k, pt in enumerate(man['parts']):
        rec = json.loads((out / 'parts' / f'part_{k:03d}.json').read_text()); b = (out / 'parts' / f'part_{k:03d}.jsonl.gz').read_bytes()
        C.require(hashlib.sha256(b).hexdigest() == rec['sha256'] and rec['manifest_sha256'] == C.sha(out / 'manifest.json') and rec['sources'] == sources(), f'part {k}: identity')
        rr = LS.gz_rows(b); C.require(len(rr) == 2 * len(pt['worlds']), f'part {k}: not two forks per world'); rows += [flat(r) for r in rr]
    F = pd.DataFrame(rows); F.to_csv(out / 'forks.csv', index=False); S = summaries(F); pd.DataFrame(S).to_csv(out / 'summary.csv', index=False)
    W = reasons(F); pd.DataFrame(W).to_csv(out / 'reasons.csv', index=False)
    st = F.groupby(['day', 'status']).size().rename('n').reset_index().to_dict('records')
    js = dict(version=VERSION, forks=int(len(F)), statuses=st, summary=S, reasons=W, predicates=predicates(F), reproduced_all=bool(F[F.status.isin(['ok', 'reproduction_failed', 'keep_baseline_mismatch'])].reproduced.all()),
              manifest_sha256=C.sha(out / 'manifest.json'), sources=sources(), asof=C.stamp())
    (out / 'exposure.json').write_text(json.dumps(js, indent=1, sort_keys=True, default=str) + '\n')
    print(json.dumps(dict(forks=js['forks'], statuses=st, reproduced_all=js['reproduced_all']), indent=1, default=str))


def selftest():
    import pf_entry_exposure_selftest as ST
    ST.run(sys.modules[__name__])


if __name__ == '__main__':
    a = sys.argv[1:]
    if a == ['selftest']:
        selftest()
    elif a == ['manifest']:
        build_manifest()
    elif a[:1] == ['run'] and len(a) == 2:
        print(run_part(int(a[1])))
    elif a == ['summarize']:
        summarize()
    else:
        sys.exit(__doc__)
