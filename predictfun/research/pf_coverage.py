"""PI0-COVERAGE-v1 (P-2026-005; 2026-11-21, session aea5bfb9). Review 216ff603's next priority (next_priority.md), USER go 13:41Z ("Approve go")
and the preflight choice "Extend source, all 6": representative neutral-exposure labels on six more consumed dates under the SAME frozen PI0
(pi0_50_v1) and the SAME neutral-hold target, before any further model or target variant. One data / coverage round: no new predictor, target
horizon, model search, live order or untouched date; development data, never validation. The Oct 26-27 admitted cohort (PI0-ADMISSION) is a
separately identified reference, never pooled.
DATES: 20261024, 20261025 (world source keep_pull_20261117), 20261031, 20261101, 20261102, 20261103 (world source source_extend_20261121, the frozen
 builders run unchanged, the Oct 24 control reproduced). Oct 23 (the teacher day), the sealed Oct 28-30 block and the forward Nov 4 are never
 read: every tape reader goes through the guarded loader (part_allowed, fair_fix, hour_ok); input identities as pf_adv2_sample.guarded_files.
WORLDS (manifest, before any label): every world (m, t0) of the day with one original per side (pf_r2_label_sample.worlds_all on the day's
 source); per UTC hour of t0 the PER_HOUR lowest sha256(DOMAIN|world|canonical key), ties by the key; pi_world = n / N per hour. Statuses of
 selected worlds that do not bind are kept, never backfilled. Pilot: the PILOT_PER_DAY lowest-ranked selected worlds of each date. Partitions:
 one date x BLOCK_HOURS UTC hours (every cell lies inside one partition).
CENSUS (roll(), per selected world): the frozen PI0 factual episode (pf_adv2.Teacher0, causal probe every AUDIT_EVERY) both sides in the shared
 engine, step by step as pf_branch_engine.step; per (tick, side) the state, view (inventory, room, other side, reservations via the view's
 position / room), the teacher's status / action / V / P, the executed decision and generation ids. Categories (model decisions only):
 absent_POST, absent_WAIT (absent, a teacher call with status model: book_ok, room >= 0, not near END, supported); review_KEEP, review_PULL
 (the FIRST live decision of a generation, status model). Every other decision is a counted status (engine_fallback, cap_blocked, near_end,
 feature_unavailable, support refusals, incoherent_forecast, live_later, pending / terminal ticks).
QUERIES (sample_queries(), per partition after its census): cell = date x side x the world's stratum hour x category; the PER_CELL lowest
 sha256(DOMAIN|query|{day, m, t_ms, side, t, cat}), ties by the key; pi_cell = n / N; weight = 1 / (pi_world pi_cell). No fill, value or later
 duration enters selection. Queries of one world share its future; never independent market samples, never summed as P&L.
LABELS (labels(), per world with queries): the factual episode rerun (pf_pi0_admission.factual / AdmRec, frozen) with snapshots before every
 sampled absent decision and before the POST that created every sampled first review's generation; its fills / decisions must equal the
 census pass's (reroll_mismatch). From each snapshot the neutral hold pf_r2_shadow.hold_from (POST now and hold the generation to END under the
 structural rules; never replaced; the other side holds and never posts). Targets (pf_entry_retention.stage_target): absent queries from the
 hold's activation; first reviews from the review's modeled cancel effect. Reconciliation (reconcile()): the target recomputed from the hold
 ledger's own fills; the hold ledger's cash / position / value (pf_r2_labels.reconcile_arm); for PI0's own generations the hold gid equals the
 factual one and the factual fills before the factual removal equal the hold's; for a first PULL the factual removal is the modeled effect;
 for absent_WAIT the factual decision was WAIT. Moments: the captured stage (107 inputs, P, E[q], E[qm], V, edge reproduced, hidden-notice copy).
MEASUREMENT (measure(), descriptive; weights w; MAE secondary; no interval):
 coverage: per date x side x category x horizon bin: labeled rows, census N, fills, statuses;
 forecasts: per source / date / side / pooled: Brier of P and of the pooled rate, P bins, movement and value MSE and bias against zero and
   edge x E[q], conditional movement error on fills, severe (value < -SEVERE) totals, top-5 world / generation shares (weighted, unweighted);
 local actions on each source: none / all / PI0's sign rule (POST iff V > 0; KEEP iff V >= 0) - constant within a factual source (degenerate,
   shown); the combined entry population (absent_POST + absent_WAIT) and the combined review population (KEEP + PULL), each with its
   inclusion weights, J = sum(w chosen value) / sum(w), and a NULL_DRAWS matched null (seed SEED) within date x side x hour preserving the
   chosen count (chosen-weight ratio reported), free strata and rows.
READINGS (predicates(), computed): covered cells (>= MIN_ROWS labeled rows and >= MIN_FILLS fills per source x side x horizon bin); value error
 / severe losses concentrated (weighted top-5 world share >= 0.5); the unchanged rule selects positive local exposure (combined population:
 J_rule > 0, > J_all, > the null p95; INCONCLUSIVE below MIN_FREE_STRATA free strata or MIN_FREE_ROWS free rows); cells lacking support.
Usage (run_light): pf_coverage.py selftest | manifest | pilot <day> | run <part> | measure"""
import copy
import hashlib
import json
import subprocess
import sys
import time
import types
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pf_pi0_admission as PA                            # noqa: E402  frozen: factual / AdmRec, the loader pattern
import pf_adv2_sample as AS                              # noqa: E402  frozen: the day-guarded identity files
import pf_source_extend as SE                            # noqa: E402  the frozen source extension (its root)

C, pd, T, LS, L, R2, KP, BE, X, RT, A = PA.C, PA.pd, PA.T, PA.LS, PA.L, PA.R2, PA.KP, PA.BE, PA.X, PA.RT, PA.A
VERSION = 'pi0_coverage_v1'
DOMAIN = 'pi0_coverage_v1_20261121'
DAYS = ('20261024', '20261025', '20261031', '20261101', '20261102', '20261103')
KP_DIRS = {**{d: Path(KP.OUT) for d in ('20261024', '20261025')}, **{d: SE.OUT / 'keep_pull' for d in SE.NEW_DAYS}}
PER_HOUR, PER_CELL, BLOCK_HOURS, PILOT_PER_DAY, AUDIT_EVERY = 8, 4, 2, 4, 35
CATS = ('absent_POST', 'absent_WAIT', 'review_KEEP', 'review_PULL')
NT, STEP, END = L.NT, float(L.STEP), float(L.END)
MIN_ROWS, MIN_FILLS, MIN_FREE_STRATA, MIN_FREE_ROWS, NULL_DRAWS, SEED, SEVERE = 30, 10, 5, 30, 500, 20261121, 30.
HBINS = ((0., 100.), (100., 500.), (500., 2000.), (2000., 5000.))
OUT = A.DATA / 'pi0_coverage_v1_20261121'
REPO = HERE.parents[1]
RECEIPT = 'predictfun/reviews/coverage_20261121'
SOURCES = ('pf_coverage.py', 'pf_coverage_selftest.py', 'pf_pi0_admission.py', 'pf_entry_retention.py', 'pf_source_extend.py', 'pf_r2_shadow.py', 'pf_r2_labels.py',
           'pf_r2_label_sample.py', 'pf_branch_engine.py', 'pf_r2_teacher.py', 'pf_adv2.py', 'pf_adv2_sample.py')


def sources():
    return {n: C.sha(HERE / n) for n in SOURCES}


def committed_blob(rel):
    r = subprocess.run(['git', '-C', str(REPO), 'show', f'HEAD:{rel}'], capture_output=True)
    return r.stdout if r.returncode == 0 else None


def committed_equal(rel, local):
    b = committed_blob(rel)
    return b is not None and Path(local).exists() and b == Path(local).read_bytes()


def frozen():
    """The module, its selftest, the declaration and the manifest are the committed ones."""
    return bool(all(committed_equal(f'predictfun/research/{n}', HERE / n) for n in ('pf_coverage.py', 'pf_coverage_selftest.py'))
                and all(committed_equal(f'{RECEIPT}/{n}', REPO / RECEIPT / n) for n in ('declaration.md', 'manifest.json')))


# ---------------------------------------------------------------------------------------------------------------------- the population
def rank(kind, key):
    return hashlib.sha256(f'{DOMAIN}|{kind}|{LS.ckey(key)}'.encode()).hexdigest()


def sample_worlds(worlds, per_hour=PER_HOUR):
    """Per (day, UTC hour of t0) the per_hour lowest-ranked worlds -> (strata, selected in canonical (m, t0) order)."""
    by = {}
    for w in worlds:
        by.setdefault((w['day'], int(w['hour'])), []).append(w)
    strata, sel = [], []
    for (day, h), ws in sorted(by.items()):
        r = sorted(ws, key=lambda w: (rank('world', LS.wkey(w)), LS.ckey(LS.wkey(w)))); n = min(per_hour, len(r)); pi = n / len(r)
        strata.append(dict(day=day, hour=h, N=len(r), n=n, pi=pi))
        sel += [dict(LS.wkey(w), hour=h, rank=rank('world', LS.wkey(w)), pi_world=pi, oid0=int(w['o'][0].oid), oid1=int(w['o'][1].oid)) for w in r[:n]]
    sel.sort(key=lambda x: (x['day'], x['m'], x['t_ms']))
    return strata, sel


def pilot_of(sel, day, n=PILOT_PER_DAY):
    return sorted([x for x in sel if x['day'] == day], key=lambda x: (x['rank'], LS.ckey(LS.wkey(x))))[:n]


def partitions(sel):
    parts = []
    for day in DAYS:
        for h0 in range(0, 24, BLOCK_HOURS):
            mem = [LS.wkey(x) for x in sel if x['day'] == day and h0 <= int(x['hour']) < h0 + BLOCK_HOURS]
            if mem:
                parts.append(dict(part=len(parts), day=day, hours=[h0, h0 + BLOCK_HOURS - 1], members=mem))
    return parts


def build_manifest(path=None, worlds_fn=None, identity_fn=None):
    """The frozen population (header): per date the worlds, strata, selected worlds, pilot and input identity; the partitions. Write-once."""
    path = Path(path or REPO / RECEIPT / 'manifest.json'); C.require(not path.exists(), 'the manifest exists (write-once)')
    worlds_fn = worlds_fn or (lambda d: LS.worlds_all(d, KP_DIRS[d]))
    identity_fn = identity_fn or (lambda d: LS.identity_now(d, KP_DIRS[d], KP.Tape(), AS.guarded_files, LS.hours_tracked))
    sample, allsel = {}, []
    for day in DAYS:
        W = worlds_fn(day); strata, sel = sample_worlds(W); allsel += sel
        sample[day] = dict(N_worlds=len(W), strata=strata, selected=sel, selected_sha256=hashlib.sha256(LS.ckey(sel).encode()).hexdigest(), kp_dir=str(KP_DIRS[day]),
                           pilot=[LS.wkey(x) for x in pilot_of(sel, day)], inputs=identity_fn(day))
    man = dict(version=VERSION, domain=DOMAIN, days=list(DAYS), per_hour=PER_HOUR, per_cell=PER_CELL, block_hours=BLOCK_HOURS, pilot_per_day=PILOT_PER_DAY,
               audit_every=AUDIT_EVERY, cats=list(CATS), sample=sample, partitions=partitions(allsel), settings=dict(BE.SETTINGS),
               feature_schema=dict(n=len(T.FEATS), sha256=hashlib.sha256(LS.ckey(list(T.FEATS)).encode()).hexdigest()), asof=C.stamp())
    path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(json.loads(json.dumps(man, default=str)), indent=1, sort_keys=True) + '\n')
    return man


# ---------------------------------------------------------------------------------------------------------------------- the census
def categorize(st, dec, exe, first):
    """(category or None, status) of one (tick, side) decision (header)."""
    if st not in ('absent', 'live'):
        return None, st
    if st == 'live' and not first:
        return None, 'live_later'
    if dec is None:
        return None, 'engine_fallback'
    if dec['status'] != 'model':
        return None, str(dec['status'])
    if exe is None or exe[3] != dec['action']:
        return None, 'constrained'
    if st == 'absent':
        return ('absent_POST' if dec['action'] == 'POST' else 'absent_WAIT'), 'model'
    return ('review_KEEP' if dec['action'] == 'KEEP' else 'review_PULL'), 'model'


def roll(world, day, w, Mo, sup, cls=None):
    """The census pass: the factual PI0 episode as pf_branch_engine.step runs it, one row per (tick, side) -> (branch, ledger, policy, probe stats,
    rows)."""
    C.require(not BE.STALE_VIEW, 'an engine switch the episode loop does not follow')
    probe, cst = T.causal_probe(AUDIT_EVERY); pol = (cls or A.Teacher0)(Mo, sup); pol.causal = probe
    B = BE.Branch(world['path'], 0, ident=R2.episode_ident(day, w['o'][0])); pol.bind(B, world); rows, reviewed, created = [], set(), {}
    for k in range(NT):
        g = k * STEP; BE.pre_tick(B, g); B.k = k
        for s in ((B.episode + k) % 2, 1 - (B.episode + k) % 2):
            v = B.view(g, s); nl, nd = len(pol.log), len(B.decisions)
            BE.decide(B, g, s, pol)
            dec = pol.log[-1] if len(pol.log) > nl else None; exe = list(B.decisions[-1]) if len(B.decisions) > nd else None
            first = bool(v['state'] == 'live' and v['gid'] not in reviewed)
            if v['state'] == 'live':
                reviewed.add(v['gid'])
            cat, status = categorize(v['state'], dec, exe, first)
            if exe is not None and exe[3] == 'POST':
                created[int(exe[4])] = (int(round(g)), s)
            gid = int(v['gid']) if v['gid'] is not None else (int(exe[4]) if (exe is not None and exe[4] is not None) else None)
            rows.append(dict(t=int(round(g)), k=k, side=s, state=v['state'], other=v['other'], gid=gid, cat=cat, status=status, first_review=first,
                             book_ok=bool(BE.book_ok(v, s)), room=float(v['room']), position=float(v['position']), near_end=bool(g + BE.PLACE >= B.end),
                             action=dec['action'] if dec else None, V=dec.get('V') if dec else None, P=dec.get('P') if dec else None, executed=exe[3] if exe else None,
                             created_at=list(created[gid]) if (gid is not None and gid in created and v['state'] == 'live') else None))
        B.check_position(g)
    return B, BE.finish(B), pol, cst, rows


def hbin(t_to_end):
    for lo, hi in HBINS:
        if lo < t_to_end <= hi:
            return f'({lo:.0f},{hi:.0f}]'
    return 'outside'


def qbin(queue, remaining):
    if not (np.isfinite(queue) and np.isfinite(remaining)) or remaining <= 0 or queue < 0:
        return 'missing_or_invalid'
    r = queue / remaining
    return '0' if r == 0 else '(0,1]' if r <= 1 else '(1,3]' if r <= 3 else '>3'


def obin(o):
    return 'missing_or_invalid' if not np.isfinite(o) else '1' if o <= 1 else '2' if o <= 2 else '>=3'


def sample_queries(C_rows, sel_by_key):
    """Per cell (day, side, the world's stratum hour, category) the PER_CELL lowest-ranked queries (header) -> (queries, cells)."""
    by = {}
    for r in C_rows:
        if r['cat'] is not None:
            hour = int(sel_by_key[(r['day'], r['m'], r['t_ms'])]['hour'])
            by.setdefault((r['day'], int(r['side']), hour, r['cat']), []).append(r)
    qs, cells = [], []
    for cell, rs in sorted(by.items()):
        key = lambda r: dict(day=r['day'], m=int(r['m']), t_ms=int(r['t_ms']), side=int(r['side']), t=int(r['t']), cat=r['cat'])
        o = sorted(rs, key=lambda r: (rank('query', key(r)), LS.ckey(key(r)))); n = min(PER_CELL, len(o)); pi = n / len(o)
        cells.append(dict(day=cell[0], side=cell[1], hour=cell[2], cat=cell[3], N=len(o), n=n, pi_cell=pi))
        for r in o[:n]:
            pw = float(sel_by_key[(r['day'], r['m'], r['t_ms'])]['pi_world'])
            qs.append(dict(key(r), hour=cell[2], N_cell=len(o), n_cell=n, pi_cell=pi, pi_world=pw, weight=1. / (pw * pi), rank=rank('query', key(r)),
                           gid=r['gid'], created_at=r['created_at']))
    return qs, cells


# ---------------------------------------------------------------------------------------------------------------------- the labels
def ledger_ns(Lh, mark):
    """A branch-like view of a finished ledger for pf_r2_labels.reconcile_arm (its orders' sizes, the mark)."""
    return types.SimpleNamespace(mark=float(mark), E=types.SimpleNamespace(o=[dict(size=float(x[3])) for x in Lh['orders9']]))


def recompute(fills, gid, side, mark, eff, ref):
    """The target from a fill list, independently of pf_r2_shadow.target: fills of gid at or after eff."""
    sg = 1. if int(side) == 0 else -1.; f = [x for x in fills if int(x[1]) == int(gid) and float(x[0]) >= float(eff)]
    q = float(sum(float(x[3]) for x in f)); v = float(sum(sg * float(x[3]) * (float(mark) - float(x[4])) for x in f))
    qm = float(sum(sg * float(x[3]) * (float(mark) - float(ref)) for x in f))
    return dict(q=q, value=v, qm=qm)


def reconcile(target, fills, gid, side, mark, eff, ref):
    r = recompute(fills, gid, side, mark, eff, ref)
    return bool(abs(r['q'] - target['q']) <= 1e-9 and abs(r['value'] - target['value']) <= 1e-6 and abs(r['qm'] - target['qm']) <= 1e-6)


def labels(world, day, w, Mo, sup, qs, census_hashes):
    """The labeled rows of one world's queries (header)."""
    pts = set()
    for q in qs:
        pts.add((int(q['t']), int(q['side'])) if q['cat'].startswith('absent') else (int(q['created_at'][0]), int(q['created_at'][1])))
    Bf, Lf, pf, cst, snaps, posts = PA.factual(world, day, w, Mo, sup, pts)
    same = bool(Lf.get('status') == 'ok' and [L._h(Lf['fills']), L._h(Lf['decisions'])] == list(census_hashes)); mark = float(Bf.mark); out = []
    for q in qs:
        s = int(q['side']); base = dict(q, status='ok')
        if not same:
            out.append(dict(base, status='reroll_mismatch')); continue
        key = (int(q['t']), s) if q['cat'].startswith('absent') else (int(q['created_at'][0]), s)
        ctx = snaps.get(key)
        if ctx is None:
            out.append(dict(base, status='no_snapshot')); continue
        gid_h, Lh, gen_h = R2.hold_from(ctx['state'], int(ctx['k']), s)
        if Lh.get('status') != 'ok' or gid_h is None or gen_h is None:
            out.append(dict(base, status=f"hold_{Lh.get('status')}" if Lh.get('status') != 'ok' else 'hold_no_generation')); continue
        fh = [list(x) for x in RT.jl(Lh['fills']) if int(x[1]) == int(gid_h)]; d0, gidf = posts.get(key, (None, None)); checks = dict(ledger_ok=bool(L.reconcile_arm(Lh, ledger_ns(Lh, mark))))
        if q['cat'].startswith('absent'):
            st = (pf.stage.get(key) or {}).get('entry') or dict(stage='entry', status='not_captured'); eff = float(gen_h['act'])
            checks['factual'] = bool(d0 is not None and d0[3] == ('POST' if q['cat'] == 'absent_POST' else 'WAIT'))
            if q['cat'] == 'absent_POST':
                checks['gid'] = bool(gidf is not None and int(gidf) == int(gid_h))
                rem = float(Bf.E.o[int(gid_h)]['remove']) if checks['gid'] else np.nan
                checks['prefix'] = bool(checks['gid'] and [x for x in RT.jl(Bf.E.fills) if int(x[1]) == int(gid_h) and float(x[0]) < rem] == [x for x in fh if float(x[0]) < rem])
        else:
            st0 = pf.stage.get(key) or {}; st = st0.get('first_review') or dict(stage='first_review', status='not_captured')
            checks['factual'] = bool(st.get('status') == 'ok' and int(round(float(st['t']))) == int(q['t']) and st.get('dec_status') == 'model'
                                     and st.get('action') == ('KEEP' if q['cat'] == 'review_KEEP' else 'PULL'))
            checks['gid'] = bool(int(gid_h) == int(q['gid']) and gidf is not None and int(gidf) == int(q['gid']))
            eff = float(st['effect']) if st.get('effect') is not None else np.nan; rem = float(Bf.E.o[int(q['gid'])]['remove']) if checks['gid'] else np.nan
            checks['prefix'] = bool(checks['gid'] and [x for x in RT.jl(Bf.E.fills) if int(x[1]) == int(q['gid']) and float(x[0]) < rem] == [x for x in fh if float(x[0]) < rem])
            if q['cat'] == 'review_PULL':
                checks['pull_effect'] = bool(checks['gid'] and abs(rem - eff) <= 1e-9)
        if st.get('status') != 'ok' or not st.get('inputs'):
            out.append(dict(base, status='no_stage_capture', checks=checks)); continue
        ref = float(st['inputs'][RT.FEATS.index('pf_mid')]); tg = RT.stage_target(fh, int(gid_h), s, mark, eff, ref)
        checks['target_reconciled'] = reconcile(tg, fh, int(gid_h), s, mark, eff, ref)
        checks['moments_ok'] = bool(st.get('moments') is not None and st['moments']['reproduced'] and st['moments']['v_identity'])
        checks['causal_ok'] = bool(st.get('hidden_same'))
        status = next((f'{n}_failed' for n, ok in checks.items() if ok is False), 'ok')
        out.append(dict(base, status=status, checks=checks, stage=st, target=tg, hold=dict(gid=int(gid_h), act=float(gen_h['act']), remove=float(gen_h['remove']),
                                                                                      refused=bool(gen_h['refused']), fills=fh), mark=mark))
    return out, dict(same=same, calls=len(pf.log), binding_bad=pf.bind_bad, causal_mismatch=cst['mismatch'])


# ---------------------------------------------------------------------------------------------------------------------- the partitions
def _bind_all(day, members, man, tim=None):
    """The identity-checked loader of pf_pi0_admission.run_part for one date's members -> (worlds, pairs, fair, depth, feature fn); tim: the
    sub-phase seconds (fixed: identity, worlds, day identity; per market: real inputs, tape)."""
    kp = KP_DIRS[day]; src = KP.Tape(); ff = AS.guarded_files; tm = tim if tim is not None else {}; t = time.time()
    C.require(LS.identity_now(day, kp, src, ff, LS.hours_tracked) == man['sample'][day]['inputs'], f"{day}: the inputs differ from the manifest's"); tm['identity'] = time.time() - t; t = time.time()
    allw = {LS.ckey(LS.wkey(x)): x for x in LS.worlds_all(day, kp, keys=[(x['m'], x['t_ms']) for x in members])}
    C.require(sorted(allw) == sorted(LS.ckey(LS.wkey(x)) for x in members), "a member world is not one of the day's worlds"); tm['worlds'] = time.time() - t; t = time.time()
    mk = sorted({x['m'] for x in members}); fair, depth, hrs, trk = X.real_inputs(day, mk); tm['real_inputs'] = time.time() - t; t = time.time()
    C.require(LS.day_identity(day, kp, src, hrs, trk, ff) == man['sample'][day]['inputs'], "real_inputs' hours / tracked differ from the manifest"); tm['day_identity'] = time.time() - t
    t = time.time(); data = src.load(day, mk, src.offset(day)); prs = {int(W_[0]['m']): W_ for W_ in src.pairs(data)}; data = None; tm['tape'] = time.time() - t
    return allw, prs, fair, depth, X.real_features


def run_members(day, members, man, out_rec, tag, man_sha, binder=None, assets=None):
    """Census, query sampling and labels for one set of members -> the record (census rows and labeled rows written beside it). binder / assets:
    the synthetic world and stub forecasts in the selftest only."""
    sel = {(x['day'], int(x['m']), int(x['t_ms'])): x for x in man['sample'][day]['selected']}
    Mo, sup = (assets or (lambda: tuple(L.teacher_assets())[:2]))(); t0 = time.time(); G = LS.Guard(900. + 240. * max(1, len(members)), 240.)
    mon = LS.RN.Sampler(Path(str(out_rec) + '_status.json'), LS.GUARD); mon.start(); tim = dict(load=None, load_parts={}, census=[], labels=[], label_queries=[])
    try:
        allw, prs, fair, depth, ff = (binder or (lambda: _bind_all(day, members, man, tim['load_parts'])))(); tim['load'] = time.time() - t0; G.check('load')
        cens, wst, hashes, binds = [], {}, {}, {}
        for x in members:
            aw = allw[LS.ckey(LS.wkey(x))]; w = dict(m=x['m'], t=aw['t'], o=aw['o']); tw = time.time()
            world, st = L.world_bind(day, w, prs, fair, depth, ff); wk = (day, int(x['m']), int(x['t_ms']))
            if world is None:
                wst[wk] = f'world:{st}'; continue
            Bc, Lc, pc, cc, rows = roll(world, day, w, Mo, sup); wst[wk] = 'ok' if Lc.get('status') == 'ok' else f"ledger:{Lc.get('status')}"
            hashes[wk] = [L._h(Lc['fills']), L._h(Lc['decisions'])] if Lc.get('status') == 'ok' else None
            binds[wk] = dict(calls=len(pc.log), binding_bad=pc.bind_bad, causal_mismatch=cc['mismatch'])
            cens += [dict(r, day=day, m=int(x['m']), t_ms=int(x['t_ms'])) for r in rows]; world = Bc = None
            tim['census'].append(time.time() - tw); G.check('pair', time.time() - tw); G.check('world')
        ok_keys = {k for k, v in wst.items() if v == 'ok' and binds[k]['binding_bad'] == 0 and binds[k]['causal_mismatch'] == 0}
        qs, cells = sample_queries([r for r in cens if (r['day'], r['m'], r['t_ms']) in ok_keys], sel); rows_out = []
        for x in members:
            wk = (day, int(x['m']), int(x['t_ms'])); wq = [q for q in qs if (q['day'], q['m'], q['t_ms']) == wk]
            if not wq:
                continue
            aw = allw[LS.ckey(LS.wkey(x))]; w = dict(m=x['m'], t=aw['t'], o=aw['o']); tw = time.time()
            world, st = L.world_bind(day, w, prs, fair, depth, ff)
            if world is None:
                rows_out += [dict(q, status=f'world:{st}') for q in wq]; continue
            rr, info = labels(world, day, w, Mo, sup, wq, hashes[wk]); rows_out += [dict(r, label_info=info) for r in rr]; world = None
            tim['labels'].append(time.time() - tw); tim['label_queries'].append(len(wq)); G.check('pair', time.time() - tw); G.check('world')
        cb = LS.gz_bytes(cens); rb = LS.gz_bytes(rows_out)
        Path(str(out_rec) + '_census.jsonl.gz').write_bytes(cb); Path(str(out_rec) + '_rows.jsonl.gz').write_bytes(rb)
        rec = dict(version=VERSION, tag=tag, day=day, members=len(members), worlds={LS.ckey(list(k)): v for k, v in wst.items()}, binds={LS.ckey(list(k)): v for k, v in binds.items()},
                   cells=cells, queries=len(qs), rows=len(rows_out), census_rows=len(cens), census_sha256=hashlib.sha256(cb).hexdigest(), rows_sha256=hashlib.sha256(rb).hexdigest(),
                   manifest_sha256=man_sha, sources=sources(), timing=tim, elapsed_s=time.time() - t0, hwm_mib=LS.hwm(), asof=C.stamp())
        LS.RN.atomic_json(Path(str(out_rec) + '.json'), rec)
        return 'DONE'
    except LS._Latch as e:
        LS.RN.atomic_json(Path(str(out_rec) + '_status.json'), dict(state='STOP_RESOURCE', tag=tag, latch=e.args[0], asof=C.stamp())); return 'STOP_RESOURCE'
    finally:
        mon.live = False


def run_part(k, out=OUT):
    C.require(frozen(), 'not frozen: the module / selftest / declaration / manifest is not the committed one')
    man = json.loads((REPO / RECEIPT / 'manifest.json').read_text()); p = man['partitions'][int(k)]; d = Path(out) / 'parts'; d.mkdir(parents=True, exist_ok=True)
    rec = d / f'part_{int(k):03d}'
    if Path(str(rec) + '.json').exists():
        C.require(json.loads(Path(str(rec) + '.json').read_text()).get('sources') == sources(), f'part {k}: an existing record of another identity'); return 'EXISTS'
    pj = json.loads((REPO / RECEIPT / 'logs' / 'pilot_projection.json').read_text()) if (REPO / RECEIPT / 'logs' / 'pilot_projection.json').exists() else None
    C.require(pj is not None and pj.get('run_rest') is True, 'the write-once pilot projection does not allow the build')
    return run_members(p['day'], p['members'], man, rec, f'part_{int(k):03d}', C.sha(REPO / RECEIPT / 'manifest.json'))


def run_pilot(day, out=OUT):
    C.require(frozen(), 'not frozen: the module / selftest / declaration / manifest is not the committed one')
    man = json.loads((REPO / RECEIPT / 'manifest.json').read_text()); C.require(day in man['days'], f'{day}: not a manifest date')
    d = Path(out) / 'pilot'; d.mkdir(parents=True, exist_ok=True); rec = d / f'pilot_{day}'
    C.require(not Path(str(rec) + '.json').exists(), f'{day}: the pilot exists (write-once)')
    return run_members(day, man['sample'][day]['pilot'], man, rec, f'pilot_{day}', C.sha(REPO / RECEIPT / 'manifest.json'))


# ---------------------------------------------------------------------------------------------------------------------- the measurement
def wm(x, w):
    x, w = np.asarray(x, float), np.asarray(w, float); m = np.isfinite(x) & np.isfinite(w)
    return float((x[m] * w[m]).sum() / w[m].sum()) if m.any() and w[m].sum() > 0 else None


def kish(w):
    w = np.asarray(w, float); return float(w.sum() ** 2 / (w ** 2).sum()) if len(w) and (w ** 2).sum() > 0 else 0.


def frame(rows):
    """Labeled rows -> one record each (statuses kept; eligible = status ok with model moments)."""
    out = []
    for r in rows:
        st, tg, mo = r.get('stage') or {}, r.get('target') or {}, ((r.get('stage') or {}).get('moments') or {})
        a = st.get('inputs'); g = (lambda n: float(a[RT.FEATS.index(n)])) if a else (lambda n: np.nan)
        z = float(tg.get('q', 0.) > 1e-12) if tg else np.nan
        out.append(dict(day=r['day'], m=int(r['m']), t_ms=int(r['t_ms']), side=int(r['side']), t=int(r['t']), cat=r['cat'], hour=int(r['hour']), weight=float(r['weight']),
                        N_cell=int(r['N_cell']), n_cell=int(r['n_cell']), status=r['status'], gid=r.get('gid'), dec_status=st.get('dec_status'),
                        P=mo.get('P'), Eq=mo.get('Eq'), Eqm=mo.get('Eqm'), V=mo.get('V'), edge=mo.get('edge'), q=tg.get('q'), value=tg.get('value'),
                        M=(tg.get('qm') if z == 1. else 0.) if tg else np.nan, z=z, severe=tg.get('severe'), t_to_end=g('t_to_end'), hbin=hbin(g('t_to_end')) if a else 'none',
                        qbin=qbin(g('fp_queue'), g('fp_remaining')) if a else 'none', obin=obin(g('ctx_generation_ordinal')) if a else 'none'))
    D = pd.DataFrame(out)
    D['eligible'] = (D.status == 'ok') & D.V.notna() & (D.dec_status == 'model')
    return D


def slices(E):
    return ([('all', 'all', E)] + [(str(d), 'all', g) for d, g in E.groupby('day', sort=True)] + [('all', str(s), g) for s, g in E.groupby('side', sort=True)]
            + [(str(d), str(s), g) for (d, s), g in E.groupby(['day', 'side'], sort=True)])


def top5(vals, w, keys):
    c = np.asarray(vals, float) * np.asarray(w, float)
    if not len(c) or np.abs(c).sum() == 0:
        return None, None
    a = np.sort(np.abs(c))[::-1]; ws = {}
    for k, x in zip(keys, c):
        ws[k] = ws.get(k, 0.) + x
    aw = np.sort(np.abs(np.array(list(ws.values()))))[::-1]
    return float(a[:5].sum() / a.sum()), float(aw[:5].sum() / aw.sum())


def forecasts(D):
    out = []; E = D[D.eligible]
    for cat, g0 in E.groupby('cat', sort=True):
        pool = wm(g0.z, g0.weight)
        for day, side, g in slices(g0):
            w = g.weight; f = g[g.z > 0]; ev = g.V - g.value; em = g.Eqm - g.M; keys = list(zip(g.day, g.m, g.t_ms)); sv = g.value < -SEVERE
            gs, ws_ = top5(ev, w, keys); gsu, wsu = top5(ev, np.ones(len(g)), keys); vs, vws = top5(g.value.where(sv, 0.), w, keys)
            dsum = (w * ev).groupby(g.day).sum().abs(); dsev = (w * g.value.where(sv, 0.)).groupby(g.day).sum().abs()
            dmax = float(dsum.max() / dsum.sum()) if dsum.sum() > 0 else None; dsmax = float(dsev.max() / dsev.sum()) if dsev.sum() > 0 else None
            out.append(dict(cat=cat, day=day, side=side, n=int(len(g)), weight=float(w.sum()), ess=kish(w), worlds=len(set(keys)), fills=int(len(f)), weighted_fills=float(f.weight.sum()),
                            fill_rate=wm(g.z, w), mean_P=wm(g.P, w), brier=wm((g.P - g.z) ** 2, w), brier_pooled=wm((pool - g.z) ** 2, w), mean_V=wm(g.V, w),
                            mean_value=wm(g.value, w), bias_V=wm(ev, w), mse_V=wm(ev ** 2, w), mse_zero=wm(g.value ** 2, w), mse_mech=wm((g.edge * g.Eq - g.value) ** 2, w),
                            mae_V=wm(ev.abs(), w), mean_Eqm=wm(g.Eqm, w), mean_M=wm(g.M, w), bias_m=wm(em, w), mse_m=wm(em ** 2, w), mse_m_zero=wm(g.M ** 2, w),
                            c_m=wm(f.Eqm / f.P - f.M, f.weight) if len(f) else None, severe_n=int(sv.sum()), severe_sum=float((w[sv] * g.value[sv]).sum()),
                            top5_gen_err=gs, top5_world_err=ws_, top5_gen_err_unweighted=gsu, top5_world_err_unweighted=wsu, top5_world_severe=vws,
                            max_date_share_err=dmax, max_date_share_severe=dsmax))
    return out


def calibration(D):
    out = []; E = D[D.eligible]
    for cat, g0 in E.groupby('cat', sort=True):
        for lo, hi in zip(RT.PBINS[:-1], RT.PBINS[1:]):
            g = g0[(g0.P >= lo) & ((g0.P < hi) if hi < 1. else (g0.P <= hi))]
            out.append(dict(cat=cat, dim='P', cell=f'[{lo},{hi})', n=int(len(g)), weight=float(g.weight.sum()), mean_P=wm(g.P, g.weight), fill_rate=wm(g.z, g.weight)))
        for dim, col in (('horizon', 'hbin'), ('queue', 'qbin'), ('ordinal', 'obin')):
            for lab, g in g0.groupby(col, sort=True):
                out.append(dict(cat=cat, dim=dim, cell=str(lab), n=int(len(g)), weight=float(g.weight.sum()), mean_P=wm(g.P, g.weight), fill_rate=wm(g.z, g.weight),
                                fills=int((g.z > 0).sum()), mean_V=wm(g.V, g.weight), mean_value=wm(g.value, g.weight), mean_Eqm=wm(g.Eqm, g.weight), mean_M=wm(g.M, g.weight)))
    return out


def coverage(D):
    out = []
    for (day, side, cat, hb), g in D.groupby(['day', 'side', 'cat', 'hbin'], sort=True):
        out.append(dict(day=day, side=int(side), cat=cat, hbin=hb, rows=int(len(g)), eligible=int(g.eligible.sum()), fills=int((g.eligible & (g.z > 0)).sum()),
                        statuses=json.dumps(g.status.value_counts().to_dict(), sort_keys=True)))
    return out


def rule(cat, V):
    return (V > 0) if cat.startswith('absent') else (V >= 0)


def null(E, mask, draws=NULL_DRAWS, seed=SEED):
    w = E.weight.to_numpy(float); v = E.value.to_numpy(float); W = float(w.sum()); rng = np.random.default_rng(seed)
    keys = sorted(set(zip(E.day.astype(str), E.side.astype(int), E.hour.astype(int))), key=str)
    grp = [np.flatnonzero((E.day.astype(str) == d).to_numpy() & (E.side.astype(int) == s).to_numpy() & (E.hour.astype(int) == h).to_numpy()) for d, s, h in keys]
    ks = [int(mask[g].sum()) for g in grp]; free = [(g, k) for g, k in zip(grp, ks) if 0 < k < len(g)]; J = np.full(draws, np.nan); wr = np.full(draws, np.nan)
    Wc = float((w * mask).sum()); Jc = float((w * mask * v).sum() / W) if W > 0 else None
    for i in range(draws):
        m = np.zeros(len(E), bool)
        for g, k in zip(grp, ks):
            if k:
                m[rng.choice(g, k, replace=False)] = True
        J[i] = float((w * m * v).sum() / W) if W > 0 else np.nan; wr[i] = float((w * m).sum() / Wc) if Wc > 0 else np.nan
    return dict(strata=len(grp), free_strata=len(free), free_rows=int(sum(len(g) for g, _ in free)), chosen=int(mask.sum()), chosen_weight=Wc, J=Jc,
                null_mean=float(np.nanmean(J)), null_p95=float(np.nanpercentile(J, 95)), percentile=float((J <= Jc + 1e-12).mean()) if Jc is not None else None,
                chosen_weight_ratio_mean=float(np.nanmean(wr)) if Wc > 0 else None), J


def actions(D):
    """Per source (degenerate by construction) and per combined population: none / all / PI0's sign rule; the matched null on the combined."""
    out, nulls, draws = [], {}, []; E = D[D.eligible].reset_index(drop=True)
    pops = [(c, (c,)) for c in CATS] + [('entry_combined', ('absent_POST', 'absent_WAIT')), ('review_combined', ('review_KEEP', 'review_PULL'))]
    for name, cats in pops:
        g = E[E.cat.isin(cats)].reset_index(drop=True)
        if not len(g):
            continue
        w, v = g.weight.to_numpy(float), g.value.to_numpy(float); W = float(w.sum())
        mk = {'none': np.zeros(len(g), bool), 'all': np.ones(len(g), bool), 'pi0_rule': np.array([bool(rule(c, x)) for c, x in zip(g.cat, g.V.astype(float))])}
        for r, m in mk.items():
            out.append(dict(population=name, rule=r, n=int(len(g)), weight=W, chosen=int(m.sum()), chosen_weight=float((w * m).sum()), J=float((w * m * v).sum() / W),
                            mean_chosen=float((w * m * v).sum() / (w * m).sum()) if (w * m).sum() > 0 else None, degenerate=bool(m.all() or not m.any())))
        nu, J = null(g, mk['pi0_rule']); nulls[name] = nu; draws += [dict(population=name, draw=i, J=float(j)) for i, j in enumerate(J)]
    return out, nulls, draws


def predicates(FC, CV, AC, NU):
    """The declared descriptive readings (header), computed."""
    cov = [dict(cat=c['cat'], side=c['side'], hbin=c['hbin'], rows=c['rows'], fills=c['fills']) for c in CV]
    agg = {}
    for c in cov:
        k = (c['cat'], c['side'], c['hbin']); a = agg.setdefault(k, dict(rows=0, fills=0)); a['rows'] += c['rows']; a['fills'] += c['fills']
    covered = sorted(f'{k[0]}|{k[1]}|{k[2]}' for k, a in agg.items() if a['rows'] >= MIN_ROWS and a['fills'] >= MIN_FILLS)
    lacking = sorted(f'{k[0]}|{k[1]}|{k[2]}' for k, a in agg.items() if not (a['rows'] >= MIN_ROWS and a['fills'] >= MIN_FILLS))
    pooled = {r['cat']: r for r in FC if r['day'] == 'all' and r['side'] == 'all'}
    conc = {c: dict(value_error_concentrated=bool(r['top5_world_err'] is not None and r['top5_world_err'] >= .5),
                    severe_concentrated=bool(r['top5_world_severe'] is not None and r['top5_world_severe'] >= .5), top5_world_err=r['top5_world_err'],
                    top5_world_severe=r['top5_world_severe'], value_error_one_date=bool(r.get('max_date_share_err') is not None and r['max_date_share_err'] >= .5),
                    severe_one_date=bool(r.get('max_date_share_severe') is not None and r['max_date_share_severe'] >= .5),
                    max_date_share_err=r.get('max_date_share_err'), max_date_share_severe=r.get('max_date_share_severe')) for c, r in pooled.items()}
    sel = {}
    for name, nu in ((k_, v_) for k_, v_ in NU.items() if k_.endswith('combined')):
        a = {r['rule']: r for r in AC if r['population'] == name}; why = [n for n, b in (('inadequate_movable_support', not (nu['free_strata'] >= MIN_FREE_STRATA and nu['free_rows'] >= MIN_FREE_ROWS)),) if b]
        J = a['pi0_rule']['J']
        sel[name] = dict(answer='INCONCLUSIVE' if why else ('POSITIVE' if (J > 0 and J > a['all']['J'] and J > nu['null_p95']) else 'NOT_POSITIVE'), reasons=why, J_rule=J,
                         J_all=a['all']['J'], J_none=a['none']['J'], null_p95=nu['null_p95'], free_strata=nu['free_strata'], free_rows=nu['free_rows'])
    return dict(covered=covered, lacking_support=lacking, concentration=conc, selects_positive_exposure=sel)


REF_OPP_REL = 'predictfun/reviews/movement_audit_20261121/outputs/opportunities.csv.gz'


def reference():
    """The Oct 26-27 admitted cohort (PI0-ADMISSION, read from the movement audit's committed opportunities table) scored by forecasts() as
    ref_admitted_entry / ref_admitted_review - a separately identified reference, never pooled with this build."""
    import gzip
    import io
    b = committed_blob(REF_OPP_REL); C.require(b is not None, 'the reference table is not committed')
    A_ = pd.read_csv(io.BytesIO(gzip.decompress(b)), float_precision='round_trip', dtype={'day': str, 'key': str, 'acc_status': str, 'stage': str, 'cohort': str})
    A_ = A_[(A_.cohort == 'admitted') & (A_.acc_status == 'ok') & A_.stage.isin(['entry', 'first_review'])]
    R_ = pd.DataFrame(dict(day=A_.day, m=A_.m.astype(int), t_ms=A_.t_ms.astype(int), side=A_.side.astype(int), cat='ref_admitted_' + A_.stage.map({'entry': 'entry', 'first_review': 'review'}),
                           weight=A_.weight.astype(float), P=A_.P, Eq=A_.Eq, Eqm=A_.Eqm, V=A_.V, edge=A_.edge, q=A_.q, value=A_.value, M=A_.M, z=A_.z, eligible=True))
    return forecasts(R_.reset_index(drop=True)), hashlib.sha256(b).hexdigest()


def load_parts(out=OUT):
    man = json.loads((REPO / RECEIPT / 'manifest.json').read_text()); d = Path(out) / 'parts'; rows, cens, recs = [], [], {}
    for p in man['partitions']:
        k = int(p['part']); rp = d / f'part_{k:03d}.json'
        if not rp.exists():
            recs[k] = None; continue
        rec = json.loads(rp.read_text()); rb = (d / f'part_{k:03d}_rows.jsonl.gz').read_bytes(); cb = (d / f'part_{k:03d}_census.jsonl.gz').read_bytes()
        C.require(hashlib.sha256(rb).hexdigest() == rec['rows_sha256'] and hashlib.sha256(cb).hexdigest() == rec['census_sha256'] and rec['sources'] == sources()
                  and rec['manifest_sha256'] == C.sha(REPO / RECEIPT / 'manifest.json'), f'part {k}: identity')
        rows += LS.gz_rows(rb); cens += LS.gz_rows(cb); recs[k] = rec
    return man, rows, cens, recs


def measure(out=OUT):
    out = Path(out); C.require(not (out / 'audit.json').exists(), 'the measurement exists (immutable)')
    man, rows, cens, recs = load_parts(out); D = frame(rows); CE = pd.DataFrame(cens)
    FC, CA, CV = forecasts(D), calibration(D), coverage(D); AC, NU, DR = actions(D); P = predicates(FC, CV, AC, NU); REF, ref_sha = reference()
    cs = CE.groupby(['day', 'side', 'state', 'status']).size().rename('n').reset_index().to_dict('records') if len(CE) else []
    for name, rows_ in (('forecasts', FC), ('calibration', CA), ('coverage', CV), ('actions', AC), ('null_draws', DR), ('census_statuses', cs), ('reference', REF)):
        pd.DataFrame(rows_).to_csv(out / f'{name}.csv', index=False)
    D.to_csv(out / 'labels.csv', index=False)
    js = dict(version=VERSION, parts_done=sorted(k for k, v in recs.items() if v), parts_missing=sorted(k for k, v in recs.items() if not v), rows=int(len(D)),
              eligible=int(D.eligible.sum()), statuses=D.status.value_counts().to_dict(), null=NU, predicates=P, reference_sha256=ref_sha, manifest_sha256=C.sha(REPO / RECEIPT / 'manifest.json'),
              sources=sources(), asof=C.stamp())
    (out / 'audit.json').write_text(json.dumps(js, indent=1, sort_keys=True, default=str) + '\n'); print(json.dumps(dict(rows=js['rows'], eligible=js['eligible'], predicates=P), indent=1, default=str))


def selftest():
    import pf_coverage_selftest as ST_
    ST_.run(sys.modules[__name__])


if __name__ == '__main__':
    a = sys.argv[1:]
    if a == ['selftest']:
        selftest()
    elif a == ['manifest']:
        build_manifest()
    elif a[:1] == ['pilot'] and len(a) == 2:
        print(run_pilot(a[1]))
    elif a[:1] == ['run'] and len(a) == 2:
        print(run_part(int(a[1])))
    elif a == ['measure']:
        C.require(frozen(), 'not frozen'); measure()
    else:
        sys.exit(__doc__)
