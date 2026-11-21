"""PI0-ADMISSION-EXPOSURE-v1 (P-2026-005; 2026-11-21, session aea5bfb9). Review 99fb165d's one next priority: the economic exposure comparison on
orders PI0 itself chose to post, with the corrected mean-forecast metrics, the existing exposure x admission cross-tab and the superseded
ENTRY-RETENTION reading, in one declared package. One diagnostic, zero fitted candidates, consumed Oct 26-27 only; frozen before any branch.
POPULATION (census() / select(), frozen in the manifest before simulation): the accepted R4E replay (adv2_entry_smooth_v1 replay_v1), each
 partition byte-equal to its committed record; its 384 worlds (380 ok, 4 world exclusions kept as statuses). A generation is a PI0 POST in a
 world's PI0 episode trace: a decision (t, side, 'absent', 'POST', gid) that created gid (689). Cell = the decision's UTC day x side x UTC
 hour (t_ms + t). Per cell the four lowest sha256(DOMAIN + canonical JSON of {day, m, t_ms, side, gid, decision_ms}), ties by that JSON,
 or all if fewer; inclusion weight N_cell / min(4, N_cell). No execution, outcome, review action or score enters the selection.
EXECUTION (per world, run_part): the factual PI0 episode (pf_adv2.Teacher0, the replay's own policy, probe every 35) is rerun step by step as
 pf_branch_engine.step runs it, snapshotting the branch just before each selected decision; its fills / decisions hashes, value and teacher
 calls must equal the saved trace (factual_not_reproduced). At each selected point PI0 must POST and create the selected gid (not_admitted /
 gid_mismatch). From the snapshot, POST_KEEP_SELECTED: PI0's own decision there, then PI0 except that a PULL of the live selected gid becomes
 KEEP (logged with its shadow; overrides counted from the log, decide side-effect free); other sides and generations PI0. Its decisions
 and fills before the first override must equal the factual's (prefix_mismatch); its ledger reconciles (not_reconciled). END 5,000 ms, the
 settings and the mark must be the saved trace's (horizon_mismatch). Every fork is related to its world; weights estimate the 689.
VALUES (END mark, gross): D_EXPOSURE = R(KEEP) - R(factual); D_selected (the selected gid's fills) + D_focal_rest + D_opposite =
 D_EXPOSURE (components_mismatch). The selected generation's activation, removal, lifetime, fills, shares and END value in both arms.
STAGES (as ENTRY-RETENTION, captured in fn, actual callbacks only): entry = PI0's own POST decision; first_review / first_pull of the selected
 gid in the factual run. Recorded: 107 inputs with their bit hash, the client view (PF receipt time), venue state, reservations, the frozen
 predictor's P / E[q] / E[qm] / V / edge (reproduced) and a hidden-notice copy. Targets: the neutral hold of the selected generation from the
 snapshot (pf_r2_shadow.hold_from), equal to the KEEP arm's selected-gid fills (target_not_equivalent); entry from activation, review / pull
 from the modeled effect; at the first pull the held value after the effect equals D_selected and the factual selected fills are the held
 ones before it (pull_not_reconciled).
MEASUREMENTS (measure()): weighted (inclusion) and unweighted. Economic per day x side and pooled: D_EXPOSURE and its components (means,
 sums, gains / losses / zero, severe < -30 c, top-5 |D| share, worlds), the selected generation's lifetime and fills in both arms.
 Forecasts per stage: MSE of V, of zero and of edge x E[q]; mean error; Brier of P and of the pooled rate; MAE as a secondary, median-type
 diagnostic; conditional-on-fill q / qm residuals; probability bins PBINS; per side and day. First review: PULL against 500 matched draws
 (seed SEED, counts within day x side x UTC hour, weights carried), with the strata and rows that can change assignment; thin below 30
 selected or unselected, degenerate strata counted - thin or degenerate is not identified. Entry -> first review (both modeled): input-group
 bit changes; new observations per source (pf_age, bn_age, bnd_age in ms: new when the age grew by less than the elapsed time minus 1 ms;
 the PF receipt time); dV with and without any new observation.
CROSS-TAB (crosstab(), adaptive accounting, declared here): the 768 ENTRY-EXPOSURE forks (v2) joined by exact key to ENTRY-RETENTION's entry
 admission: D_EXPOSURE, D_initial, D_rest per day x side x {would_POST, would_WAIT (model), fallback (not model), excluded}: n, worlds, means,
 sums, top-5 share. RETENTION v2 (retention_v2(), superseding the MAE-gated reading in band): per stage x side x day MSE of V / zero /
 edge x E[q], mean error, Brier; flips on both-modeled transitions; MAE kept as secondary.
PREDICATES (predicates()): per day the sign of the weighted mean D_EXPOSURE (|mean| <= 1e-9 is 0); thin when fewer than 30 selected
 generations gain selected-gid fills by keeping. admitted_longer_beneficial: both days positive, not thin; admitted_longer_harmful: both
 negative, not thin; not_identified otherwise. Per stage mse_beats_zero = weighted MSE(V) < MSE(0); review_discriminates = not thin, not all
 strata degenerate and percentile <= 0.05. RETENTION v2: entry_mse_beats_zero, review_mse_beats_zero (unweighted, as its population).
Usage (run_light): pf_pi0_admission.py selftest | manifest | run <part> | measure"""
import copy
import datetime
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pf_adv2 as A                                      # noqa: E402  frozen: Teacher0
import pf_adv2_entry_replay as ER                        # noqa: E402  frozen: the R4E replay
import pf_adv2_eval as AE                                # noqa: E402  frozen: R3's evaluation plan
import pf_adv2_sample as AS                              # noqa: E402  frozen: the day-guarded files
import pf_entry_exposure as EX                           # noqa: E402  frozen: components
import pf_entry_retention as RT                          # noqa: E402  frozen: capture, moments, targets

C, pd, T, LS, L, R2, KP, BE, X = RT.C, RT.pd, RT.T, RT.LS, RT.L, RT.R2, RT.KP, RT.BE, RT.X
VERSION = 'pi0_admission_exposure_v1'
DOMAIN, PER_CELL, AUDIT_EVERY, NT, STEP, END = 'pi0_admission_exposure_v1|', 4, 35, L.NT, float(L.STEP), float(L.END)
OUT = A.DATA / 'pi0_admission_v1_20261121'
REPO = HERE.parents[1]
RECEIPT = 'predictfun/reviews/pi0_admission_20261121'
R4E_DIR = Path(ER.OUT); R4E_REC = REPO / 'predictfun/reviews/r4e_entry_smooth_20261121/replay'
NULL_DRAWS, SEED, THIN, PCT, MEANINGFUL, SEVERE = 500, 20261121, 30, .05, 1e-6, 30.
AGES = ('pf_age', 'bn_age', 'bnd_age')
SOURCES = ('pf_pi0_admission.py', 'pf_entry_retention.py', 'pf_entry_exposure.py', 'pf_adv2.py', 'pf_r2_labels.py', 'pf_r2_label_sample.py', 'pf_branch_engine.py',
           'pf_r2_teacher.py', 'pf_r2_shadow.py', 'pf_pipeline_bench.py')


def sources():
    return {n: C.sha(HERE / n) for n in SOURCES}


def committed_equal(rel, local):
    r = subprocess.run(['git', '-C', str(REPO), 'show', f'HEAD:{rel}'], capture_output=True)
    return r.returncode == 0 and r.stdout == Path(local).read_bytes()


# ---------------------------------------------------------------------------------------------------------------------- the population
def r4e_parts(d=None, rec=None):
    """The R4E replay partitions, each byte-equal to its committed record -> [(k, record, rows)]."""
    d, rec = Path(d or R4E_DIR), Path(rec or R4E_REC); out = []
    for k in range(12):
        rc = json.loads((rec / f'part_{k:03d}.json').read_text()); b = (d / 'parts' / f'part_{k:03d}.jsonl.gz').read_bytes()
        C.require(hashlib.sha256(b).hexdigest() == rc['sha256'], f'R4E part {k}: its bytes differ from the committed record'); out.append((k, rc, LS.gz_rows(b)))
    return out


def key_of(g):
    return LS.ckey(dict(day=str(g['day']), m=int(g['m']), t_ms=int(g['t_ms']), side=int(g['side']), gid=int(g['gid']), decision_ms=int(g['decision_ms'])))


def rank_of(g):
    return hashlib.sha256((DOMAIN + key_of(g)).encode()).hexdigest()


def cell_of(t_ms, decision_ms, side):
    u = datetime.datetime.fromtimestamp((int(t_ms) + float(decision_ms)) / 1000., tz=datetime.timezone.utc)
    return (u.strftime('%Y%m%d'), int(side), int(u.hour))


def census(parts):
    """Every PI0 POST generation of the ok worlds' PI0 traces, and every world with its status."""
    gens, worlds = [], []
    for k, rc, rows in parts:
        for r in rows:
            worlds.append(dict(part=k, day=str(r['day']), m=int(r['m']), t_ms=int(r['t_ms']), hour=int(r['hour']), status=r['status']))
            if r['status'] != 'ok':
                continue
            for d in r['episodes']['PI0']['trace']['decision_list']:
                if d[2] == 'absent' and d[3] == 'POST' and d[4] is not None:
                    g = dict(part=k, day=str(r['day']), m=int(r['m']), t_ms=int(r['t_ms']), side=int(d[1]), gid=int(d[4]), decision_ms=int(round(float(d[0]))))
                    g['cell'] = list(cell_of(g['t_ms'], g['decision_ms'], g['side'])); gens.append(g)
    return gens, worlds


def select(gens, per_cell=PER_CELL):
    """Per cell the per_cell lowest-ranked generations, with the inclusion weights -> (selected, cells)."""
    by = {}
    for g in gens:
        by.setdefault(tuple(g['cell']), []).append(g)
    sel, cells = [], []
    for c in sorted(by):
        gs = sorted(by[c], key=lambda g: (rank_of(g), key_of(g))); n = len(gs); keep = gs[:per_cell]; w = n / min(per_cell, n)
        sel += [dict(g, weight=w, n_cell=n) for g in keep]; cells.append(dict(cell=list(c), n=n, selected=len(keep), weight=w))
    return sel, cells


def build_manifest(out=OUT, parts=None):
    out = Path(out); mp = out / 'manifest.json'; C.require(not mp.exists(), 'the manifest exists (immutable)'); out.mkdir(parents=True, exist_ok=True)
    parts = parts if parts is not None else r4e_parts(); gens, worlds = census(parts); sel, cells = select(gens)
    recs = {k: C.sha(R4E_REC / f'part_{k:03d}.json') for k, _, _ in parts} if parts and Path(R4E_REC).exists() else {}
    man = dict(version=VERSION, domain=DOMAIN, per_cell=PER_CELL, census=len(gens), selected=sel, cells=cells, worlds=worlds, r4e_records=recs,
               parts=sorted({g['part'] for g in sel}), sources=sources(), asof=C.stamp())
    mp.write_text(json.dumps(man, sort_keys=True, separators=(',', ':'), default=str) + '\n')
    print(json.dumps(dict(census=len(gens), selected=len(sel), cells=len(cells), worlds=len(worlds), parts=man['parts']), indent=1))


# ---------------------------------------------------------------------------------------------------------------------- the policies
class AdmRec(A.Teacher0):
    """PI0 (pf_adv2.Teacher0, decisions unchanged) recording, in fn, the selected generations' stages: entry at a selected (t, side) absent
    decision; the first live callback and the first PULL of a watched gid (registered from the executed POST by the episode loop)."""

    def __init__(self, models, support, causal=None, points=()):
        super().__init__(models, support); self.causal = causal; self.points = set(points); self.watch = {}; self.stage = {}

    def fn(self, v):
        a = super().fn(v); dec = self.log[-1]; s, t = int(v['side']), float(v['t']); key = (int(round(t)), s)
        if v['state'] == 'absent' and key in self.points and 'entry' not in self.stage.get(key, {}):
            self.stage.setdefault(key, {})['entry'] = dict(RT.capture(self, v, dec, 'entry', seed_of(key)), admission='would_POST' if a == 'POST' else 'would_WAIT')
        elif v['state'] == 'live' and v['gid'] in self.watch:
            st = self.stage[self.watch[v['gid']]]; first = 'first_review' not in st
            if first:
                st['first_review'] = RT.capture(self, v, dec, 'first_review', seed_of(self.watch[v['gid']]))
            if a == 'PULL' and 'first_pull' not in st:
                st['first_pull'] = dict(st['first_review'], stage='first_pull', same_as_review=True) if first else \
                    dict(RT.capture(self, v, dec, 'first_pull', seed_of(self.watch[v['gid']])), same_as_review=False)
        return a


class KeepSelected(A.Teacher0):
    """POST_KEEP_SELECTED's continuation: PI0, except that a PULL of the live selected gid on the focal side becomes KEEP, logged with its shadow
    (no counter; overrides are the log's)."""

    def __init__(self, models, support, causal=None):
        super().__init__(models, support); self.causal = causal; self.focal, self.gid0 = None, None

    def decide(self, v, B):
        d = super().decide(v, B)
        if self.gid0 is not None and v['state'] == 'live' and int(v['side']) == self.focal and v['gid'] == self.gid0 and d.get('action') == 'PULL':
            return dict(d, action='KEEP', status='keep_selected_override', shadow_action='PULL', shadow_status=d.get('status'))
        return d


def seed_of(key):
    return int(hashlib.sha256(f'{key[0]}|{key[1]}'.encode()).hexdigest()[:8], 16)


def factual(world, day, w, Mo, sup, points, cls=AdmRec):
    """The factual PI0 episode as pf_branch_engine.step runs it, snapshotting the branch before each selected decision -> (branch, ledger,
    policy, probe stats, snapshots {key: ctx}, posts {key: (decision, gid)})."""
    C.require(not BE.STALE_VIEW, 'an engine switch the episode loop does not follow')
    probe, cst = T.causal_probe(AUDIT_EVERY); pol = cls(Mo, sup, causal=probe, points=points); B = BE.Branch(world['path'], 0, ident=R2.episode_ident(day, w['o'][0]))
    pol.bind(B, world); snaps, posts = {}, {}
    for k in range(NT):
        g = k * STEP; BE.pre_tick(B, g); B.k = k
        for s in ((B.episode + k) % 2, 1 - (B.episode + k) % 2):
            key = (int(round(g)), s)
            if key in pol.points:
                snaps[key] = L.ctx_of(copy.deepcopy(B), k, s, B.episode)
            BE.decide(B, g, s, pol)
            if key in pol.points:
                posts[key] = (list(B.decisions[-1]) if B.decisions and float(B.decisions[-1][0]) == g and int(B.decisions[-1][1]) == s else None, B.open[s])
                if B.open[s] is not None and posts[key][0] is not None and posts[key][0][3] == 'POST':
                    pol.watch[B.open[s]] = key
        B.check_position(g)
    return B, BE.finish(B), pol, cst, snaps, posts


def keep_arm(ctx, world, Mo, sup, cls=KeepSelected):
    """POST_KEEP_SELECTED from a snapshot: PI0's own decision there, then the override on the gid it created -> (branch, ledger, policy, probe
    stats, the executed decision, gid0)."""
    k, s, ep = int(ctx['k']), int(ctx['side']), int(ctx['episode']); g = k * STEP; first = (ep + k) % 2
    Bx = copy.deepcopy(ctx['state']); probe, cst = T.causal_probe(AUDIT_EVERY); pol = cls(Mo, sup, causal=probe); pol.bind(Bx, world); pol.focal = s
    BE.decide(Bx, g, s, pol); d0 = list(Bx.decisions[-1]); gid0 = Bx.open[s] if d0[3] == 'POST' else None; pol.gid0 = gid0
    if first == s:
        BE.decide(Bx, g, 1 - s, pol)
    Bx.check_position(g)
    for kk in range(k + 1, NT):
        BE.step(Bx, kk, pol)
    return Bx, BE.finish(Bx), pol, cst, d0, gid0


def keep_prefix_ok(Bk, Bf, pk, stop_g):
    """The KEEP arm equals the factual up to its first override; without one, to the end (header)."""
    dk, df = RT.jl(Bk.decisions), RT.jl(Bf.decisions); ov = [d for d in pk.log if d.get('status') == 'keep_selected_override']
    if not ov:
        return bool(dk == df and RT.jl(Bk.E.fills) == RT.jl(Bf.E.fills))
    i = next((j for j, (a, b) in enumerate(zip(dk, df)) if a != b), None)
    if i is None:
        return False
    a, b, g = dk[i], df[i], float(ov[0]['t'])
    return bool(a[0] == b[0] == g and a[1] == b[1] == pk.focal and a[2] == b[2] == 'live' and a[3] == 'KEEP' and b[3] == 'PULL' and a[4] == b[4] == pk.gid0
                and [f for f in RT.jl(Bk.E.fills) if float(f[0]) <= g] == [f for f in RT.jl(Bf.E.fills) if float(f[0]) <= g])


def reconcile_pull(held_value, d_selected, factual_fills, held_fills, eff, removed_at):
    """The first-pull reconciliation (header): the held value at or after the effect is D_selected, the factual selected fills are the held ones
    before the effect, and the factual removal lands at the modeled effect."""
    return bool(abs(held_value - d_selected) <= 1e-6 and factual_fills == [x for x in held_fills if float(x[0]) < eff] and float(removed_at) == float(eff))


def gen_summary(B, gid, side):
    """The selected generation in one arm: activation, removal and its reason, lifetime, fills, shares, END value, unfilled at END."""
    o = B.E.o[gid]; mark = float(B.mark); sg = 1. if side == 0 else -1.; fl = sorted(f for f in B.E.fills if int(f[1]) == gid); q = 0.; full = None
    for f in fl:
        q += float(f[3])
        if full is None and q >= float(o['size']) - 1e-9:
            full = float(f[0])
    rem_t = min(float(o['remove']), full if full is not None else np.inf, END)
    why = 'refused' if o['refused'] else 'full_fill' if full is not None else ((o['why'] or 'end') if float(o['remove']) <= END else 'end')
    return dict(act=float(o['act']), remove=float(o['remove']), why=why, refused=bool(o['refused']), fills=len(fl), filled=q, first_fill=float(fl[0][0]) if fl else None,
                lifetime_ms=0. if o['refused'] else max(0., rem_t - float(o['act'])), end_value=sum(sg * float(f[3]) * (mark - float(f[4])) for f in fl),
                unfilled_at_end=float(o['rem']) if (float(o['remove']) > END and not o['refused']) else 0.)


def world_rows(day, w_row, world, w, Mo, sup, sel):
    """Every selected generation of one world -> its rows (header)."""
    saved = w_row['episodes']['PI0']['trace']; pts = {(int(g['decision_ms']), int(g['side'])): g for g in sel}; t0 = time.time()
    Bf, Lf, pf, cstf, snaps, posts = factual(world, day, w, Mo, sup, set(pts))
    fact = bool(Lf.get('status') == 'ok' and L._h(Lf['fills']) == saved['fills_sha256'] and L._h(Lf['decisions']) == saved['decisions_sha256']
                and float(Lf['value']) == float(saved['value']) and len(pf.log) == int(saved['teacher']['calls']))
    horizon = bool(float(Bf.end) == END == float(saved['end']) and dict(Bf.settings) == saved['settings'] and float(Bf.mark) == float(saved['mark']))
    out = []
    for key, g in sorted(pts.items()):
        r = dict({k: g[k] for k in ('day', 'm', 't_ms', 'side', 'gid', 'decision_ms', 'weight', 'n_cell')}, cell=g['cell'], status='ok')
        if not fact:
            out.append(dict(r, status='factual_not_reproduced')); continue
        if not horizon:
            out.append(dict(r, status='horizon_mismatch')); continue
        d0, gidf = posts.get(key, (None, None))
        if d0 is None or d0[3] != 'POST':
            out.append(dict(r, status='not_admitted')); continue
        if int(gidf) != int(g['gid']):
            out.append(dict(r, status='gid_mismatch')); continue
        ctx = snaps[key]; s = int(g['side']); gid = int(g['gid']); tg = time.time()
        Bk, Lk, pk, cstk, dk0, gidk = keep_arm(ctx, world, Mo, sup)
        if dk0[3] != 'POST' or gidk != gid:
            out.append(dict(r, status='keep_not_admitted')); continue
        gid_h, Lh, gen_h = R2.hold_from(ctx['state'], int(ctx['k']), s); fh = [x for x in RT.jl(Lh['fills']) if int(x[1]) == gid]
        fk = [x for x in RT.jl(Bk.E.fills) if int(x[1]) == gid]; equivalent = bool(gid_h == gid and Lh['status'] == 'ok' and fh == fk)
        st = pf.stage.get(key, {}); ent = st.get('entry') or dict(stage='entry', status='not_captured'); mark = float(Bf.mark)
        stages = dict(entry=ent)
        for nm in ('first_review', 'first_pull'):
            stages[nm] = st.get(nm) or dict(stage=nm, status=f"no_{'review' if nm == 'first_review' else 'pull'}")
        e_ref = (ent.get('inputs') or [None] * len(RT.FEATS))[RT.FEATS.index('pf_mid')]; te = RT.stage_target(fh, gid, s, mark, float(gen_h['act']), e_ref if e_ref is not None else np.nan)
        stages['entry']['target'] = te
        for nm in ('first_review', 'first_pull'):
            if stages[nm].get('status') == 'ok':
                ref = stages[nm]['inputs'][RT.FEATS.index('pf_mid')] if stages[nm].get('inputs') else np.nan
                stages[nm]['target'] = RT.stage_target(fh, gid, s, mark, stages[nm]['effect'], ref); stages[nm]['unavoidable'] = RT.unavoidable_of(te, stages[nm]['target'])
        cf, ck = EX.components(Bf, gid, s), EX.components(Bk, gid, s); rf, rk = float(Lf['value']), float(Lk['value']); dE = rk - rf
        dsel, dfr, dop = (ck[k_] - cf[k_] for k_ in ('v_initial', 'v_focal_rest', 'v_opposite'))
        comp_ok = bool(abs(dE - (dsel + dfr + dop)) <= 1e-6 and abs(rf - sum(cf.values())) <= 1e-6 and abs(rk - sum(ck.values())) <= 1e-6)
        rec = None
        if stages['first_pull'].get('status') == 'ok':
            eff = stages['first_pull']['effect']; tp = stages['first_pull']['target']; ffs = [x for x in RT.jl(Bf.E.fills) if int(x[1]) == gid]
            rec = reconcile_pull(tp['value'], dsel, ffs, fh, eff, Bf.E.o[gid]['remove'])
        ov = [d for d in pk.log if d.get('status') == 'keep_selected_override']
        rows_ = [x for x in stages.values() if x.get('status') == 'ok']
        checks = dict(factual_reproduced=fact, horizon_ok=horizon, prefix_ok=keep_prefix_ok(Bk, Bf, pk, None), keep_reconciled=bool(Lk.get('status') == 'ok' and L.reconcile_arm(Lk, Bk)),
                      equivalent=equivalent, components_ok=comp_ok, pull_reconciled=rec, causal_ok=all(x['hidden_same'] for x in rows_),
                      moments_ok=all(x['moments'] is None or (x['moments']['reproduced'] and x['moments']['v_identity']) for x in rows_),
                      binding_ok=bool(pk.bind_bad == 0 and pf.bind_bad == 0 and cstk['mismatch'] == 0 and cstf['mismatch'] == 0))
        status = next((n for k_, n in (('prefix_ok', 'prefix_mismatch'), ('keep_reconciled', 'not_reconciled'), ('equivalent', 'target_not_equivalent'),
                                      ('components_ok', 'components_mismatch'), ('causal_ok', 'causality_failed'), ('moments_ok', 'moments_not_reproduced'),
                                      ('binding_ok', 'binding_failed')) if not checks[k_]), 'ok')
        if status == 'ok' and rec is False:
            status = 'pull_not_reconciled'
        out.append(dict(r, status=status, checks=checks, R_factual=rf, R_keep=rk, D_EXPOSURE=dE, D_selected=dsel, D_focal_rest=dfr, D_opposite=dop, comp_factual=cf, comp_keep=ck,
                        gen_factual=gen_summary(Bf, gid, s), gen_keep=gen_summary(Bk, gid, s), overrides=len(ov), shadow=[dict(t=d['t'], shadow_action=d['shadow_action'],
                                                                                                                        shadow_V=d.get('V')) for d in ov],
                        stages=stages, held=dict(gid=int(gid_h), act=float(gen_h['act']), fills=fh, mark=mark),
                        keep_arm=dict(fill_list=RT.jl(Bk.E.fills), decision_list=RT.jl(Bk.decisions), value=rk, cash=float(Lk['cash']), position=float(Lk['position']),
                                      fills_sha256=L._h(Lk['fills']), decisions_sha256=L._h(Lk['decisions']), end=float(Bk.end), mark=float(Bk.mark),
                                      teacher=dict(calls=len(pk.log), binding_checked=pk.bind_checked, binding_bad=pk.bind_bad),
                                      causal={k_: v_ for k_, v_ in cstk.items() if k_ not in ('details', 'n')}),
                        seconds=time.time() - tg))
    return out, time.time() - t0


def run_part(k, out=OUT):
    """One R4E partition's selected generations -> out/parts/part_<k>.jsonl.gz + its record (identities first)."""
    out = Path(out); (out / 'parts').mkdir(parents=True, exist_ok=True); rp = out / 'parts' / f'part_{int(k):03d}.json'
    C.require(committed_equal(f'{RECEIPT}/manifest.json', out / 'manifest.json'), 'the manifest is not the committed one: no simulation before the freeze')
    C.require(committed_equal('predictfun/research/pf_pi0_admission.py', HERE / 'pf_pi0_admission.py'), 'the module is not committed in HEAD')
    if rp.exists():
        C.require(json.loads(rp.read_text()).get('sources') == sources(), f'part {k}: an existing record of another identity'); return 'DONE'
    man = json.loads((out / 'manifest.json').read_text()); sel = [g for g in man['selected'] if int(g['part']) == int(k)]
    parts = {k_: (rc, rows) for k_, rc, rows in r4e_parts() if k_ == int(k)}; rc, rows = parts[int(k)]; day = rc['day']
    plan = json.loads((AE.EVAL_DIR / 'plan.json').read_text()); C.require(C.sha(AE.EVAL_DIR / 'plan.json') == AE.EVAL_PLAN_SHA, 'not the frozen evaluation plan')
    byw = {}
    for g in sel:
        byw.setdefault((int(g['m']), int(g['t_ms'])), []).append(g)
    wrows = {(int(r['m']), int(r['t_ms'])): r for r in rows}; members = [dict(day=day, m=m, t_ms=t) for (m, t) in sorted(byw)]
    Mo, sup, _ = L.teacher_assets(); kp = Path(plan['kp_dir']); src = KP.Tape(); ff = AS.guarded_files
    G = LS.Guard(600. + 360. * max(1, len(members)), 120.); mon = LS.RN.Sampler(out / 'parts' / f'part_{int(k):03d}_status.json', LS.GUARD); mon.start(); res = []
    try:
        if members:
            C.require(LS.identity_now(day, kp, src, ff, LS.hours_tracked) == plan['sample'][day]['inputs'], f'{day}: the inputs differ from the plan\'s')
            allw = {LS.ckey(LS.wkey(x)): x for x in LS.worlds_all(day, kp, keys=[(x['m'], x['t_ms']) for x in members])}
            C.require(sorted(allw) == sorted(LS.ckey(LS.wkey(x)) for x in members), 'a member world is not one of the day\'s worlds')
            mk = sorted({x['m'] for x in members}); fair, depth, hrs, trk = X.real_inputs(day, mk)
            C.require(LS.day_identity(day, kp, src, hrs, trk, ff) == plan['sample'][day]['inputs'], 'real_inputs\' hours / tracked differ from the plan')
            data = src.load(day, mk, src.offset(day)); prs = {int(W_[0]['m']): W_ for W_ in src.pairs(data)}; data = None; G.check('load')
            for x in members:
                aw = allw[LS.ckey(LS.wkey(x))]; w = dict(m=x['m'], t=aw['t'], o=aw['o']); world, wst = L.world_bind(day, w, prs, fair, depth, X.real_features)
                gs = byw[(int(x['m']), int(x['t_ms']))]
                if world is None:
                    res += [dict({k_: g[k_] for k_ in ('day', 'm', 't_ms', 'side', 'gid', 'decision_ms', 'weight', 'n_cell')}, cell=g['cell'], status=f'world:{wst}') for g in gs]; continue
                rr, sec = world_rows(day, wrows[(int(x['m']), int(x['t_ms']))], world, w, Mo, sup, gs); res += rr; world = None
                G.check('pair', max([x_.get('seconds', 0.) for x_ in rr] + [0.])); G.check('world')
        b = LS.gz_bytes(res); (out / 'parts' / f'part_{int(k):03d}.jsonl.gz').write_bytes(b)
        LS.RN.atomic_json(rp, dict(version=VERSION, part=int(k), day=day, rows=len(res), sha256=hashlib.sha256(b).hexdigest(), manifest_sha256=C.sha(out / 'manifest.json'),
                                   r4e_record_sha256=C.sha(R4E_REC / f'part_{int(k):03d}.json'), sources=sources(), elapsed_s=time.time() - G.t0, hwm_mib=LS.hwm(),
                                   timing=LS.timing(G.log), asof=C.stamp()))
        return 'PART_DONE'
    except LS._Latch as e:
        LS.RN.atomic_json(out / 'parts' / f'part_{int(k):03d}_status.json', dict(state='STOP_RESOURCE', part=int(k), latch=e.args[0], asof=C.stamp())); return 'STOP_RESOURCE'
    finally:
        mon.live = False


# ---------------------------------------------------------------------------------------------------------------------- the measurements
def wmean(x, w):
    x, w = np.asarray(x, float), np.asarray(w, float); m = np.isfinite(x)
    return float((x[m] * w[m]).sum() / w[m].sum()) if m.any() and w[m].sum() > 0 else None


def gen_table(rows):
    out = []
    for r in rows:
        d = {k: r.get(k) for k in ('day', 'm', 't_ms', 'side', 'gid', 'decision_ms', 'weight', 'n_cell', 'status', 'D_EXPOSURE', 'D_selected', 'D_focal_rest', 'D_opposite',
                                   'R_factual', 'R_keep', 'overrides')}
        d['uday'], _, d['uhour'] = (r.get('cell') or [None, None, None])[:3]
        for arm in ('factual', 'keep'):
            gs = r.get(f'gen_{arm}') or {}
            d.update({f'{arm}_{k}': gs.get(k) for k in ('why', 'filled', 'fills', 'lifetime_ms', 'end_value', 'unfilled_at_end', 'first_fill')})
        out.append(d)
    return pd.DataFrame(out)


def economics(G):
    ok = G[G.status == 'ok']; out = []
    for (day, side), g in [(('all', 'all'), ok)] + list(ok.groupby(['day', 'side'], sort=True)):
        if not len(g):
            continue
        d = g.D_EXPOSURE.astype(float); w = g.weight.astype(float); a = np.sort(d.abs().to_numpy())[::-1]
        row = dict(day=str(day), side=str(side), n=int(len(g)), worlds=int(g[['day', 'm', 't_ms']].drop_duplicates().shape[0]), weight=float(w.sum()),
                   wmean_D=wmean(d, w), mean_D=float(d.mean()), sum_D=float(d.sum()), wmean_D_selected=wmean(g.D_selected, w), wmean_D_focal_rest=wmean(g.D_focal_rest, w),
                   wmean_D_opposite=wmean(g.D_opposite, w), n_pos=int((d > MEANINGFUL).sum()), n_neg=int((d < -MEANINGFUL).sum()), n_zero=int((d.abs() <= MEANINGFUL).sum()),
                   n_severe=int((d < -SEVERE).sum()), top5_share_unweighted=float(a[:5].sum() / a.sum()) if a.sum() > 0 else None,
                   gain_forks=int((g.keep_filled > g.factual_filled + 1e-9).sum()), overrides_forks=int((g.overrides > 0).sum()))
        for arm in ('factual', 'keep'):
            row.update({f'{arm}_wmean_lifetime': wmean(g[f'{arm}_lifetime_ms'], w), f'{arm}_wmean_filled': wmean(g[f'{arm}_filled'], w),
                        f'{arm}_fill_prob': wmean((g[f'{arm}_filled'] > 1e-9).astype(float), w), f'{arm}_wmean_end_value': wmean(g[f'{arm}_end_value'], w)})
        out.append(row)
    return out


def stage_rows(rows):
    """One row per selected generation x stage, with the weights, cells and the entry -> review transition fields."""
    out = []
    for r in rows:
        st = r.get('stages') or {}; ent = st.get('entry') or {}
        for nm in RT.STAGES:
            x = st.get(nm) or {}; mo = x.get('moments') or {}; tg = x.get('target') or {}
            d = dict(day=r['day'], m=r['m'], t_ms=r['t_ms'], side=r['side'], gid=r['gid'], weight=r.get('weight'), hour=(r.get('cell') or [0, 0, 0])[2], stage=nm,
                     fork_status=r['status'], stage_status=x.get('status', r['status']), dec_status=x.get('dec_status'), action=x.get('action'), t=x.get('t'),
                     **{k: mo.get(k) for k in ('P', 'Eq', 'Eqm', 'V', 'edge', 'mech', 'q_fill', 'qm_fill')}, q=tg.get('q'), value=tg.get('value'), qm=tg.get('qm'))
            if nm == 'first_review' and x.get('inputs') and ent.get('inputs') and x.get('dec_status') == 'model' and ent.get('dec_status') == 'model':
                a0, a1 = np.asarray(ent['inputs'], np.float32), np.asarray(x['inputs'], np.float32); ch = a0.view(np.uint32) != a1.view(np.uint32); dt = float(x['t']) - float(ent['t'])
                for gname in RT.GROUPS:
                    d[f'chg_{gname}'] = int(sum(1 for i, n in enumerate(RT.FEATS) if RT.group_of(n) == gname and ch[i]))
                for ag in AGES:
                    i = RT.FEATS.index(ag); d[f'new_{ag}'] = bool(float(a1[i]) < float(a0[i]) + dt - 1.)
                d['new_pf_receipt'] = bool((x.get('view') or {}).get('received') != (ent.get('view') or {}).get('received'))
                d['any_new'] = bool(any(d[f'new_{ag}'] for ag in AGES) or d['new_pf_receipt']); d['dV'] = mo['V'] - (ent.get('moments') or {})['V']
                d['V_entry'] = (ent.get('moments') or {}).get('V')
            out.append(d)
    return pd.DataFrame(out)


def forecast_metrics(S, weighted=True):
    out = []
    for nm in RT.STAGES:
        E = S[(S.stage == nm) & (S.fork_status == 'ok') & (S.stage_status == 'ok') & (S.dec_status == 'model') & S.V.notna()]
        pool = wmean((E.q > 1e-12).astype(float), E.weight if weighted else np.ones(len(E))) if len(E) else None
        for (day, side), g in [(('all', 'all'), E)] + list(E.groupby(['day', 'side'], sort=True)):
            if not len(g):
                continue
            w = g.weight.astype(float) if weighted else pd.Series(np.ones(len(g)), index=g.index); y = (g.q > 1e-12).astype(float); fl = g[g.q > 1e-12]; wf = fl.weight if weighted else np.ones(len(fl))
            out.append(dict(stage=nm, day=str(day), side=str(side), weighted=weighted, n=int(len(g)), fill_rate=wmean(y, w), mean_P=wmean(g.P, w), brier=wmean((g.P - y) ** 2, w),
                            brier_pooled_rate=wmean((pool - y) ** 2, w), mse_V=wmean((g.V - g.value) ** 2, w), mse_zero=wmean(g.value ** 2, w), mse_mech=wmean((g.mech - g.value) ** 2, w),
                            mean_error=wmean(g.V - g.value, w), mae_V=wmean((g.V - g.value).abs(), w), mae_zero=wmean(g.value.abs(), w), mean_V=wmean(g.V, w), mean_value=wmean(g.value, w),
                            q_fill_error=wmean(fl.q_fill - fl.q, wf) if len(fl) else None, qm_fill_error=wmean(fl.qm_fill - fl.qm, wf) if len(fl) else None))
    return out


def bins(S):
    out = []
    for nm in RT.STAGES:
        E = S[(S.stage == nm) & (S.fork_status == 'ok') & (S.stage_status == 'ok') & (S.dec_status == 'model') & S.V.notna()]
        for lo, hi in zip(RT.PBINS[:-1], RT.PBINS[1:]):
            g = E[(E.P >= lo) & ((E.P < hi) if hi < 1. else (E.P <= hi))]
            out.append(dict(stage=nm, p_lo=lo, p_hi=hi, n=int(len(g)), wmean_P=wmean(g.P, g.weight) if len(g) else None,
                            wfill_rate=wmean((g.q > 1e-12).astype(float), g.weight) if len(g) else None))
    return out


def selection(S, draws=NULL_DRAWS, seed=SEED):
    """First review: PULL against matched random draws of the same counts within day x side x UTC hour (weights carried) (header)."""
    D = S[(S.stage == 'first_review') & (S.fork_status == 'ok') & (S.stage_status == 'ok') & (S.dec_status == 'model') & S.V.notna()].reset_index(drop=True)
    sel = D.action == 'PULL'; rng = np.random.default_rng(seed); grp = [g for _, g in D.groupby(['day', 'side', 'hour'], sort=True)]
    n_sel = int(sel.sum()); ks = [int(sel[g.index].sum()) for g in grp]; free = [(g, k_) for g, k_ in zip(grp, ks) if 0 < k_ < len(g)]
    ms = wmean(D.value[sel], D.weight[sel]) if n_sel else None; null = np.full(draws, np.nan)
    for i in range(draws):
        idx = [g.index[rng.choice(len(g), k_, replace=False)] for g, k_ in zip(grp, ks) if k_]
        if idx:
            ii = np.concatenate(idx); null[i] = wmean(D.value[ii], D.weight[ii])
    return dict(n=int(len(D)), n_sel=n_sel, mean_sel=ms, mean_all=wmean(D.value, D.weight) if len(D) else None, null_mean=float(np.nanmean(null)) if n_sel else None,
                null_p05=float(np.nanpercentile(null, 5)) if n_sel else None, null_p95=float(np.nanpercentile(null, 95)) if n_sel else None,
                percentile=float((null <= ms + 1e-12).mean()) if n_sel else None, strata=len(grp), free_strata=len(free), free_rows=int(sum(len(g) for g, _ in free)),
                thin=bool(n_sel < THIN or len(D) - n_sel < THIN))


def transitions(S):
    R = S[(S.stage == 'first_review') & (S.fork_status == 'ok')]; out = []
    if 'chg_market_flow' not in R or not R['chg_market_flow'].notna().any():
        return [dict(subset=nm, n=0) for nm in ('all', 'any_new_observation', 'no_new_observation')]
    R = R[R['chg_market_flow'].notna()]
    for nm, g in (('all', R), ('any_new_observation', R[R.any_new == True]), ('no_new_observation', R[R.any_new == False])):     # noqa: E712
        if not len(g):
            out.append(dict(subset=nm, n=0)); continue
        out.append(dict(subset=nm, n=int(len(g)), **{f'share_{gn}_changed': float((g[f'chg_{gn}'] > 0).mean()) for gn in RT.GROUPS},
                        **{f'share_new_{ag}': float(g[f'new_{ag}'].mean()) for ag in AGES}, share_new_pf_receipt=float(g.new_pf_receipt.mean()),
                        wmean_dV=wmean(g.dV, g.weight), wmean_V_entry=wmean(g.V_entry, g.weight), wmean_V_review=wmean(g.V, g.weight),
                        flip_share=float(((g.V_entry > 0) & (g.V < 0)).mean())))
    return out


def crosstab(expo=EX.OUT, ret=RT.OUT):
    """The ENTRY-EXPOSURE forks (v2) by ENTRY-RETENTION's entry admission (header)."""
    F = pd.read_csv(Path(expo) / 'v2' / 'forks.csv', float_precision='round_trip'); F['day'] = F.day.astype(str)
    S = pd.read_csv(Path(ret) / 'stages.csv', float_precision='round_trip'); S['day'] = S.day.astype(str); E = S[S.stage == 'entry']
    J = F.merge(E[['day', 'm', 't_ms', 'side', 'admission', 'dec_status', 'stage_status']], on=['day', 'm', 't_ms', 'side'], how='left', validate='one_to_one')
    J['cls'] = np.where(J.status != 'ok', 'excluded', np.where(J.dec_status != 'model', 'fallback', np.where(J.admission == 'would_POST', 'would_POST', 'would_WAIT')))
    out = []
    for (day, side, cls), g in list(J.groupby(['day', 'side', 'cls'], sort=True)) + [(('all', 'all', c_), J[J.cls == c_]) for c_ in sorted(J.cls.unique())]:
        d = g.D_EXPOSURE.astype(float); a = np.sort(d.dropna().abs().to_numpy())[::-1]
        out.append(dict(day=str(day), side=str(side), cls=cls, n=int(len(g)), worlds=int(g[['day', 'm', 't_ms']].drop_duplicates().shape[0]),
                        mean_D=float(d.mean()) if d.notna().any() else None, sum_D=float(d.sum()) if d.notna().any() else None,
                        mean_D_initial=float(g.D_initial.mean()) if g.D_initial.notna().any() else None, mean_D_rest=float(g.D_rest.mean()) if g.D_rest.notna().any() else None,
                        top5_share_unweighted=float(a[:5].sum() / a.sum()) if len(a) and a.sum() > 0 else None))
    return out, int(len(J)), int(((J.status == 'ok') & J.admission.isna()).sum())


def retention_v2(ret=RT.OUT):
    """The superseding ENTRY-RETENTION reading (header): mean-forecast scores per stage x side x day and the both-modeled flips."""
    S = pd.read_csv(Path(ret) / 'stages.csv', float_precision='round_trip'); S['day'] = S.day.astype(str); out = []
    for nm in RT.STAGES:
        E = S[(S.stage == nm) & (S.fork_status == 'ok') & (S.stage_status == 'ok') & (S.dec_status == 'model') & S.V.notna()]
        for (day, side), g in [(('all', 'all'), E)] + list(E.groupby(['day', 'side'], sort=True)):
            y = (g.q > 1e-12).astype(float); pool = float((E.q > 1e-12).mean())
            out.append(dict(stage=nm, day=str(day), side=str(side), n=int(len(g)), mse_V=float(((g.V - g.value) ** 2).mean()), mse_zero=float((g.value ** 2).mean()),
                            mse_mech=float(((g.mech - g.value) ** 2).mean()), mean_error=float((g.V - g.value).mean()), mae_V=float((g.V - g.value).abs().mean()),
                            mae_zero=float(g.value.abs().mean()), brier=float(((g.P - y) ** 2).mean()), brier_pooled_rate=float(((pool - y) ** 2).mean())))
    en = S[(S.stage == 'entry') & (S.dec_status == 'model') & S.V.notna()][['day', 'm', 't_ms', 'side', 'V']].rename(columns={'V': 'V_entry'})
    rv = S[(S.stage == 'first_review') & (S.dec_status == 'model') & S.V.notna() & (S.stage_status == 'ok')][['day', 'm', 't_ms', 'side', 'V']]
    B = en.merge(rv, on=['day', 'm', 't_ms', 'side']); flips = int(((B.V_entry > 0) & (B.V < 0)).sum())
    return out, dict(both_modeled=int(len(B)), flips=flips, flip_share=float(flips / len(B)) if len(B) else None)


def predicates(G, S, ret_rows):
    """The declared predicates and readings (header)."""
    ok = G[G.status == 'ok']; days = {}
    for day, g in ok.groupby('day', sort=True):
        m = wmean(g.D_EXPOSURE, g.weight); days[str(day)] = dict(n=int(len(g)), wmean_D=m, sign=0 if (m is None or abs(m) <= 1e-9) else (1 if m > 0 else -1))
    gain = int((ok.keep_filled > ok.factual_filled + 1e-9).sum()); thin = gain < THIN; sg = [v['sign'] for v in days.values()]
    rd = dict(admitted_longer_beneficial=bool(len(days) == 2 and not thin and all(x > 0 for x in sg)), admitted_longer_harmful=bool(len(days) == 2 and not thin and all(x < 0 for x in sg)))
    rd['not_identified'] = not (rd['admitted_longer_beneficial'] or rd['admitted_longer_harmful'])
    fm = {f['stage']: f for f in forecast_metrics(S) if f['day'] == 'all'}; sel = selection(S)
    mse = {nm: bool(nm in fm and fm[nm]['mse_V'] is not None and fm[nm]['mse_V'] < fm[nm]['mse_zero']) for nm in RT.STAGES}
    disc = bool(sel['percentile'] is not None and not sel['thin'] and sel['free_strata'] > 0 and sel['percentile'] <= PCT)
    rv = {r['stage']: r for r in ret_rows if r['day'] == 'all'}
    ret = dict(entry_mse_beats_zero=bool(rv['entry']['mse_V'] < rv['entry']['mse_zero']), review_mse_beats_zero=bool(rv['first_review']['mse_V'] < rv['first_review']['mse_zero']))
    return dict(days=days, gain_forks=gain, thin=thin, readings=rd, mse_beats_zero=mse, review_selection=sel, review_discriminates=disc, retention_v2=ret)


def measure(out=OUT, man_path=None):
    out = Path(out); man_path = Path(man_path or out / 'manifest.json'); C.require(not (out / 'admission.json').exists(), 'the measurement exists (immutable)')
    man = json.loads(man_path.read_text()); rows = []
    for k in man['parts']:
        rec = json.loads((out / 'parts' / f'part_{k:03d}.json').read_text()); b = (out / 'parts' / f'part_{k:03d}.jsonl.gz').read_bytes()
        C.require(hashlib.sha256(b).hexdigest() == rec['sha256'] and rec['sources'] == sources() and rec['manifest_sha256'] == C.sha(man_path), f'part {k}: identity')
        rr = LS.gz_rows(b); C.require(len(rr) == sum(1 for g in man['selected'] if int(g['part']) == int(k)), f'part {k}: not its selected generations'); rows += rr
    G = gen_table(rows); S = stage_rows(rows); G.to_csv(out / 'generations.csv', index=False); S.to_csv(out / 'stages.csv', index=False)
    E = economics(G); FM = forecast_metrics(S) + forecast_metrics(S, weighted=False); BN = bins(S); TR = transitions(S)
    CT, nj, nmiss = crosstab(); RV, RF = retention_v2(); P = predicates(G, S, RV)
    for name, rows_ in (('economics', E), ('forecasts', FM), ('bins', BN), ('transitions', TR), ('crosstab', CT), ('retention_v2', RV)):
        pd.DataFrame(rows_).to_csv(out / f'{name}.csv', index=False)
    st = G.groupby(['day', 'status']).size().rename('n').reset_index().to_dict('records')
    js = dict(version=VERSION, generations=int(len(G)), statuses=st, predicates=P, crosstab=dict(joined=nj, missing_admission=nmiss), retention_v2_flips=RF,
              manifest_sha256=C.sha(man_path), sources=sources(), asof=C.stamp())
    (out / 'admission.json').write_text(json.dumps(js, indent=1, sort_keys=True, default=str) + '\n')
    print(json.dumps(dict(generations=len(G), statuses=st, readings=P['readings'], days=P['days'], mse=P['mse_beats_zero'], retention_v2=P['retention_v2']), indent=1, default=str))


def selftest():
    import pf_pi0_admission_selftest as ST
    ST.run(sys.modules[__name__])


if __name__ == '__main__':
    a = sys.argv[1:]
    if a == ['selftest']:
        selftest()
    elif a == ['manifest']:
        build_manifest()
    elif a[:1] == ['run'] and len(a) == 2:
        print(run_part(int(a[1])))
    elif a == ['measure']:
        measure()
    else:
        sys.exit(__doc__)
