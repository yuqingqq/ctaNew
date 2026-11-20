"""THE r2_label_sample_v2 PRODUCTION LABEL RUNNER (plan v2 stage R2; P-2026-005; 2026-11-20, session aea5bfb9).
Review 4393ad19's next priority, "production launch gate for r2_label_sample_v2": a committed sampled-label plan, corrected production
acceptance, synthetic controls and an hourly distributed integration receipt. Declared before any paired outcome of this module is read.
Consumed Oct 24-25 only; no teacher refit, no candidate fit. It replaces the integration sample's equal-source round robin (pf_r2_labels.select)
by the review's probability sample; pf_r2_labels (v1, the R2-D adapter) is imported unchanged.

TARGET (unchanged; pf_r2_labels.pair_bound, arm-bound, under the frozen pi0_50_v1):
 Y_post = R(POST now) - R(WAIT this tick); Y_keep = R(KEEP this tick) - R(PULL now). Continuation pi0 on both sides to the original END
 (+5,000 ms); the 50 ms grid; 58 / 37 / 100 ms; gross cents (zero fees). No shorter END, coarser grid, open-loop continuation or fill-selected
 rows.
POPULATION: every shared (day, market, t0) world of the day's keep / pull orders (both sides coupled; one original per side), t0 an integer
 millisecond inside its UTC day. A world without an episode is a status, never replaced.
WORLD SAMPLE (frozen in the plan before any pair):
 - strata = (day, UTC hour of t0); per stratum the min(64, N_h) worlds of lowest sha256('r2_label_sample_v2_20261120|world|' + key);
 - ties broken by the canonical key; processed in canonical (m, t0) order; pi_h = n_h / N_h;
 - canonical keys: sorted-key JSON, integer milliseconds and ids, explicit nulls.
ROW SAMPLE: per selected world and side, from that world's census (pf_r2_labels.census_world, unchanged: five roll-in sources, the
 every-fifth-tick rotating phase, original-time POST rows kept); eligible = census status ok; ranked by sha256('...|row|' + key):
 - POST: the neutral original-time POST (source neutral, entry, decision 0) if eligible, with q = 1, plus the lowest-ranked other eligible
   POST, with q = 1 / (N - 1). Without it, the lowest-ranked eligible POST, with q = 1 / N.
 - KEEP: the lowest-ranked eligible KEEP, with q = 1 / N.
 Empty pools are statuses (empty_pool, no_additional_post); at most 6 pairs a world. No outcome, fill, forecast or loss enters a selection.
WEIGHT: raw w = 1 / (pi_h N q), never renormalized; each selected nonempty original / head totals 1 / pi_h.
PERSISTED, per pick: the world (day, m, t0, hour, rank, pi_h), side, original id, head, pick, N, q, w, the row key / rank / source and the
 status. Per ok pair:
 - the fork fingerprint;
 - pair_bound's result with every arm's fills and order sizes;
 - the arms' causal / binding counts;
 - the ordered 107 causal inputs at the common fork BEFORE the forced action (pf_r2_teacher.Teacher.inputs, unchanged, on a copy of the fork
   state), with their sha256 and the support status;
 - a hidden-copy check (inputs unchanged), the delivered positive control, and an independent rebuild (pf_r2_shadow.rebuild_row with its
   own market features: inputs and price equal).
 Per world, side and head: the eligible pool's keys and the census status counts (the population and its exclusions).
IDENTITY (review 4393ad19 P1): the plan binds:
 - the source closure: every research file this module reaches by import statements, transitively, without the checks;
 - the frozen teacher (freeze sha, libraries);
 - the sample (every stratum count, the selected keys, ranks and probabilities);
 - per day, every producing input: the keep / pull receipt and outputs; the tape offset; the loaders' files (step1 registry / clock / legs /
   book parts, step5 preds / rows, the imputation-offset file, the flow-price receipt behind the market metadata); real_inputs' hours and
   tracked hour files.
 Before a part is skipped, run or accepted, the executing sources, the teacher (pf_r2_labels.teacher_assets) and the day's inputs
 (recomputed without arrays) must equal the plan's; after loading, real_inputs' own hours and tracked hashes must too. A changed source,
 input or plan needs a new plan and output namespace.
VALIDATION (review 4393ad19 P1): ONE validator for resume, promotion and acceptance:
 - the part / pool bytes = their record's hashes; the record's plan, part, sources, inputs and counts = the plan's; no breach in it;
 - every member world exactly once:
   * its pool lines: one per side and head, holding keys of this world's originals, with consistent counts;
   * its picks recomputed from the pool = the stored picks (key, N, q, w), with no extra, missing or duplicate row. Each nonempty original /
     head's total 1 / pi_h follows from the recomputed weights (the selftest's enumerated identity);
   * a world status covers all four of its picks;
 - every ok pair:
   * finite values and marks; arms exactly {POST, WAIT} / {KEEP, PULL};
   * the context = the pick (kind, side, decision, original, instrument, t0, settings);
   * Y = R_a - R_b and R = cash + position x mark;
   * cash and position recomputed from the stored fills, and every generation's fills <= its size (pf_r2_labels.reconcile_arm's
     tolerances); the pre-fork fills equal in both arms; the fills' hash and count;
   * the stored reconciliation and pre-fork flags true; the result's fork = the row's;
   * every arm's binding checked with 0 failures; its causal probe with 0 mismatches and 0 parity mismatches;
   * the 107 inputs complete with their hash; hidden-copy and rebuild equality.
 A failed invariant is a failure, never a status.
RESOURCES (review 4393ad19 P2): one process per partition (one day; at most 32 worlds, the pilot 8).
 - Stops are archived before any status writer, pf_r2_runner v5's watchdog included.
 - The 2,048 MiB watchdog runs from the start of every plan, run and finalize.
 - Checked after the loading, every census, every pair and every world:
   * the never-reset high-water <= 1,536 MiB;
   * every pair <= 120 s;
   * the process <= load_s + world_s x its worlds. The plan declares 600 s and 120 s, from R2-D's measured ~108 s a day of loading and
     6 pairs x 6.06 s mean / 8.96 s max a world.
 - A breach latches STOP_RESOURCE: nothing is promoted; the event is archived and holds every run of the output until an identity-bound
   disposition of this plan.
 - Each completion record carries its timing: the loading, every world's census and pairs, and the world (never in the rows, which stay
   byte-deterministic).
PILOT: the nested subset of the frozen full sample, its first-ranked world per (day, hour): at most 48 worlds and 288 pairs, in partitions of
 8. The weights refer to the full sample.
REVIEW 38febc0b (the production round):
 - Every picked row's status must equal its result's: 'pair_not_ok' iff the pair failed, and then only for a declared missing-data outcome
   ('missing_mark'); any other failed pair is a run failure, never an exclusion. 'feature_unavailable' rows pass every pair invariant
   except the fork inputs.
 - END = 5,000 ms (the result and the episode identity); both arms' mark = the result's common finite mark; every fill time within
   [0, END].
 - Each arm's teacher callbacks are reconciled with its decision log, hash-bound to pair_bound's own decisions hash. Counted: every
   decision after the fork, and the other side's at the fork tick when it acts after the forced one, less the engine's no-book
   fallbacks. Calls = binding checks = that count, so zero only when no callback was required (a final-tick fork); 0 binding failures.
 - The causal audit runs at the plan's declared rate (audit_every; production 35, the pilot 7): calls 1, 1 + every, ... Its checks must
   equal ceil(calls / every); 0 mismatches.
Usage (run_light): pf_r2_label_sample.py selftest | plan <out> [pilot] | run <out> <part> [stop_after_worlds] | finalize <out>
"""
import ast
import copy
import gzip
import hashlib
import io
import json
import math
import os
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pf_branch_engine as BE                             # noqa: E402
import pf_r2_labels as L                                  # noqa: E402
import pf_r2_runner as RN                                 # noqa: E402
import pf_r2_shadow as R2                                 # noqa: E402
import pf_r2_teacher as T                                 # noqa: E402

C, pd, KP, X = L.C, L.pd, L.KP, L.X
VERSION, DOMAIN = 'r2_label_sample_v2', 'r2_label_sample_v2_20261120'
DAYS, PER_HOUR, PART_WORLDS, PILOT_PART = ('20261024', '20261025'), 64, 32, 8
LIMITS = dict(load_s=600., world_s=120., pair_s=120.)    # declared time limits (header); a plan records its own
AUDIT_EVERY = 35                                         # the causal audit rate (review 38febc0b decision 2); a plan records its own
PAIR_DATA_STATUSES = ('missing_mark',)                   # the only failed pair outcome that is a missing-data exclusion
PEAK, GUARD = 1536., 2048.                               # fixed memory limits: a plan never changes them
TOL_V, TOL_P = 1e-6, 1e-9                                # pf_r2_labels.reconcile_arm's ledger tolerances (value / cash, position / quantity)
CORE = ('pf_r2_label_sample.py', 'pf_r2_labels.py', 'pf_r2_teacher.py', 'pf_r2_shadow.py', 'pf_branch_engine.py', 'pf_r2_runner.py', 'pf_pipeline_bench.py',
        'pf_pipeline_export.py', 'pf_keep_pull.py', 'pf_queue_place_test.py', 'pf_queue_place_ledger.py', 'pf_stop_ev_minutes_inputs.py', 'pf_joint_moment_data.py',
        'pf_flow_price_data.py', 'pf_io.py')
CHECKS = ('_selftest.py', '_mutants.py')
STATUSES = ('ok', 'pair_not_ok', 'feature_unavailable', 'empty_pool', 'no_additional_post')      # + 'world:<status>' (no episode)
PICKED = ('ok', 'pair_not_ok', 'feature_unavailable')
PICKS = {'post': ('mandatory', 'extra', 'only'), 'keep': ('keep',)}
NAMES = {'post': ('POST', 'WAIT'), 'keep': ('KEEP', 'PULL')}
DAY_MS, HOUR_MS = 86400000, 3600000
STRICT_DAY = True
STOP_AFTER = None
_CLOSURE = []


class _Latch(Exception):
    pass


def ckey(x):
    """Canonical JSON (sorted keys, explicit nulls, no spaces)."""
    return json.dumps(x, sort_keys=True, separators=(',', ':'), ensure_ascii=True)


def rank(kind, key):
    return hashlib.sha256(f'{DOMAIN}|{kind}|{ckey(key)}'.encode()).hexdigest()


def closure():
    """The source closure: every research file reachable from this module by import statements (module level and inside functions),
    transitively, except the checks (*_selftest.py, *_mutants.py) -> sorted names. Found once per process: a changed file changes its hash,
    and a changed import list changes the importing file's hash."""
    if not _CLOSURE:
        seen, todo = set(), ['pf_r2_label_sample.py']
        while todo:
            n = todo.pop()
            if n in seen or not (HERE / n).exists():
                continue
            seen.add(n)
            for node in ast.walk(ast.parse((HERE / n).read_text())):
                if isinstance(node, ast.Import):
                    todo += [a.name.split('.')[0] + '.py' for a in node.names]
                elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                    todo.append(node.module.split('.')[0] + '.py')
        _CLOSURE.extend(sorted(n for n in seen if not n.endswith(CHECKS)))
        C.require(set(CORE) <= set(_CLOSURE), 'the source closure misses a declared core module')
    return list(_CLOSURE)


def sources_now():
    return {n: C.sha(HERE / n) for n in closure()}


def hwm():
    """This process's never-reset high-water (VmHWM, MiB): the one reading of every checkpoint here (the watchdog reads its own RSS)."""
    return RN.rss()[1]


def utc_hour(day, t_ms):
    """The UTC hour of t0; t0 must lie inside its UTC day (STRICT_DAY)."""
    d0 = int(pd.Timestamp(f'{day[:4]}-{day[4:6]}-{day[6:]}', tz='UTC').value // 10 ** 6)
    C.require(not STRICT_DAY or d0 <= t_ms < d0 + DAY_MS, f'{day}: a world t0 {t_ms} outside its UTC day')
    return int(t_ms // HOUR_MS % 24)


def real_files(day):
    """The files the day's producing loaders read beyond the keep / pull outputs and real_inputs' tracked hours -> {path: sha256}:
    - pf_stop_ev_minutes_inputs.load_day: step1 registry / clock_rl_d2 / legs_timed2 / legs_tk / book parts, step5 preds_t60k / rows_rl_d2_pk;
    - the tape offset: the imputation-offset file;
    - the market metadata (pf_joint_moment_data.metadata): the flow-price receipt, which verifies its own raw inputs;
    - fair_fix: the clock, the registry and step5 rows_rl_d2_pk (above)."""
    io_ = Path(KP.I.IO); s1, s5 = io_ / 'step1' / day, io_ / 'step5' / day
    fs = [s1 / 'registry.parquet', s1 / 'clock_rl_d2.json', s1 / 'legs_timed2.parquet', s1 / 'legs_tk.parquet', s5 / 'preds_t60k.parquet', s5 / 'rows_rl_d2_pk.parquet',
          Path(KP.QL.C.V.E.OUTD) / f'day_{day}_imp.json', Path(X.QP.JD.C.OUT) / day / 'receipt.json'] + sorted((s1 / 'books').glob('part_*.parquet'))
    C.require(all(p.exists() for p in fs), f'{day}: a producing input file is missing')
    return {str(p): C.sha(p) for p in fs}


def hours_tracked(day):
    """real_inputs' hours and tracked hashes WITHOUT building its fair / depth arrays, by its rules (pf_pipeline_export.real_inputs). Every run also
    takes them from real_inputs itself and must match, so a divergence refuses, never passes. The first pilot plan called real_inputs for
    every selected market of both days in one process and peaked at 2,132 MiB (attempt 1, kept)."""
    import pf_io as IO
    hours = X.I.X5.depth_hours(day); C.require(all(X.hour_ok(h) for h in hours), f'{day}: a sealed or forward hour'); tracked = {}
    for h in hours:
        for p_ in (Path(IO.BN_DIR) / (h + '.csv.gz'), Path(IO.REF_RUN) / 'reference_prices' / (h + '.jsonl.gz'), Path(X.I.X5.DEPTH_DIR) / (h + '.csv.gz')):
            if p_.exists():
                tracked[str(p_)] = C.sha(p_)
    return hours, tracked


def day_identity(day, kp, src, hours, tracked, files_fn):
    """Every producing input of a day: the keep / pull receipt (its outputs verified), the tape offset, the loaders' files, real_inputs' hours and
    tracked hashes."""
    r_ = json.loads((Path(kp) / day / 'receipt.json').read_text())
    C.require(all(C.sha(Path(kp) / day / x) == h for x, h in r_['outputs'].items()), f'{day}: keep-pull outputs changed since their receipt')
    return json.loads(json.dumps(dict(kp_receipt=C.sha(Path(kp) / day / 'receipt.json'), kp_outputs=r_['outputs'], offset=src.offset(day), files=files_fn(day),
                                      hours=list(hours), tracked=dict(tracked)), default=str))


def identity_now(day, kp, src, files_fn, ht_fn):
    """The day's input identity recomputed now, without building any array."""
    hours, tracked = ht_fn(day)
    return day_identity(day, kp, src, hours, tracked, files_fn)


def worlds_all(day, kp, keys=None):
    """Every shared (m, t0) world of the day with its two originals (side -> row); hour = the UTC hour of t0. keys: only those (m, t0 ms) worlds
    are built (a partition's members). Attempt 2 built all 15,352 of a day in every partition, ~580 MiB held while loading, and latched the
    checkpoint (load_profile.py)."""
    O = L.day_orders(day, kp); out = []
    if keys is not None:
        want = {(int(m), int(t)) for m, t in keys}; O = O[[(int(m), int(round(float(t)))) in want for m, t in zip(O.m, O.t)]]
    for (m, t), g in O.groupby(['m', 't'], sort=True):
        C.require(sorted(g.side.astype(int)) == [0, 1], f'{day} world {m} {t}: not one original per side')
        t = float(t); C.require(t == round(t), f'{day} world {m} {t}: t0 is not an integer millisecond')
        out.append(dict(day=day, m=int(m), t=t, t_ms=int(round(t)), hour=utc_hour(day, int(round(t))), o={int(r.side): r for r in g.itertuples()}))
    return out


def wkey(w):
    return dict(day=str(w['day']), m=int(w['m']), t_ms=int(w['t_ms']))


def sample_worlds(worlds, per_hour=PER_HOUR):
    """The stratified world sample -> (strata, selected). selected: the chosen worlds in canonical (m, t0) order, each with its rank, pi_h, its
    two original ids and the pilot flag (its stratum's first-ranked world)."""
    by = {}
    for w in worlds:
        by.setdefault((w['day'], w['hour']), []).append(w)
    strata, sel = [], []
    for (day, h), ws in sorted(by.items()):
        ranked = sorted(ws, key=lambda w: (rank('world', wkey(w)), ckey(wkey(w)))); n = min(per_hour, len(ranked)); pi = n / len(ranked)
        strata.append(dict(day=day, hour=h, N=len(ranked), n=n, pi=pi))
        for i, w in enumerate(ranked[:n]):
            sel.append(dict(wkey(w), hour=h, rank=rank('world', wkey(w)), pi=pi, pilot=(i == 0), oid0=int(w['o'][0].oid), oid1=int(w['o'][1].oid)))
    sel.sort(key=lambda x: (x['day'], x['m'], x['t_ms']))
    return strata, sel


def row_key(r):
    """The canonical key of a census row (integers for ids and milliseconds, explicit nulls)."""
    fin = lambda v: v is not None and math.isfinite(float(v))
    return dict(day=str(r['day']), instrument_id=int(r['instrument_id']), original_episode_id=int(r['original_episode_id']), side=int(r['side']),
                source=str(r['source']), shadow_source=None if r.get('shadow_source') is None else str(r['shadow_source']), kind=str(r['kind']),
                offset_ms=int(round(float(r['offset_ms']))) if fin(r.get('offset_ms')) else None,
                generation=int(round(float(r['generation']))) if fin(r.get('generation')) else None, decision_ms=int(round(float(r['decision_ms']))))


def is_mandatory(k):
    return k['source'] == 'neutral' and k['shadow_source'] == 'entry' and k['decision_ms'] == 0


def picks_of(keys, head, pi):
    """The declared picks of one original and head from its eligible pool's canonical keys -> [(pick, key or None, N, q, w)]."""
    pool = sorted(keys, key=lambda k: (rank('row', k), ckey(k))); N = len(pool)
    C.require(len({ckey(k) for k in pool}) == N, 'two eligible census rows share one canonical key')
    if head == 'keep':
        out = [('keep', pool[0], N, 1. / N)] if pool else [('keep', None, 0, None)]
    else:
        mand = [k for k in pool if is_mandatory(k)]; C.require(len(mand) <= 1, 'two neutral original-time POST rows')
        if mand:
            rest = [k for k in pool if k is not mand[0]]
            out = [('mandatory', mand[0], N, 1.), ('extra', rest[0], N, 1. / (N - 1)) if rest else ('extra', None, N, None)]
        else:
            out = [('only', pool[0], N, 1. / N)] if pool else [('only', None, 0, None)]
    return [(p, k, n, q, (1. / (pi * n * q) if k is not None else None)) for p, k, n, q in out]


def fork_inputs(ctx, world, M, sup, head, seed):
    """The 107 inputs at the common fork before the forced action (Teacher.inputs on a copy of the fork state), the support status, the hidden
    copy's inputs (must be equal), the delivered copy's (the positive control) and an independent rebuild's (must be equal) -> dict."""
    s, g = int(ctx['side']), ctx['k'] * L.STEP; Bc = copy.deepcopy(ctx['state']); t_ = L.ArmTeacher(M, sup); t_.bind(Bc, world)
    row = t_.inputs(Bc.view(g, s), Bc, s, g)
    if row is None:
        return dict(status='feature_unavailable')
    vec = [float(row[c]) for c in T.FEATS]; rng = np.random.default_rng(seed)
    same = lambda a: a is not None and all(np.float32(a[c]).tobytes() == np.float32(row[c]).tobytes() for c in T.FEATS) and float(a['price']) == float(row['price'])
    other = []
    for dl in (False, True):
        B2 = R2.hidden_copy(Bc, g, rng, delivered=dl); t2 = L.ArmTeacher(M, sup); t2.bind(B2, world); other.append(t2.inputs(B2.view(g, s), B2, s, g))
    ind = R2.rebuild_row(world['path'], world['W'], world['fv'], world['o'][s], world['day'], {'decision_ms': g}, world['fair_m'], world['depth'], world['feat_fn'], B=Bc)
    if ind is not None:
        v = Bc.view(g, s); gid = Bc.open[s]; ind = dict(ind, price=float((v['bid'] if s == 0 else v['ask']) if gid is None else Bc.E.o[gid]['price']))
    C.require(BE.fingerprint(ctx['state']) == ctx['fingerprint'], 'the fork state changed while its inputs were built')
    return dict(status='ok', inputs=vec, inputs_sha256=hashlib.sha256(np.asarray(vec, np.float32).tobytes()).hexdigest(), price=float(row['price']),
                support=sup.check(s, 'post' if head == 'post' else 'retain', row) or 'supported', hidden_equal=bool(same(other[0])),
                delivered_changed=bool(not same(other[1])), rebuild_equal=bool(same(ind)))


def arm_ledgers(fac, res, reverse=False):
    """Every executed arm's fills and order sizes, read from the branch its teacher was bound to (pair_bound runs POST / KEEP first unless
    reversed; the factory is called once per arm, in that order), matched to the arm's own fills hash -> {arm: dict(fill_list, sizes)}."""
    names = NAMES[res['kind']][::-1] if reverse else NAMES[res['kind']]
    C.require(len(fac.made) == 2, 'not one teacher per executed arm'); out = {}
    for name, (t_, _) in zip(names, fac.made):
        fl = json.loads(json.dumps(list(t_.B.E.fills), default=str))
        C.require(L._h(fl) == res['arms'][name]['fills_sha256'], f'{name}: the bound branch is not the arm')
        dl = json.loads(json.dumps(list(t_.B.decisions), default=str))
        C.require(L._h(dl) == res['arms'][name]['decisions_sha256'], f'{name}: the bound branch decisions are not the arm\'s')
        out[name] = dict(fill_list=fl, sizes=[float(o['size']) for o in t_.B.E.o], decision_list=dl, fallback_list=json.loads(json.dumps(list(t_.B.fallbacks), default=str)))
    return out


class Guard:
    """The stop rule of one process (header): the never-reset high-water <= PEAK, every pair <= its limit, the process <= its bound; check()
    raises _Latch on a breach (the caller latches STOP_RESOURCE)."""

    def __init__(self, bound_s, pair_s):
        self.t0, self.bound_s, self.pair_s, self.worst, self.log = time.time(), float(bound_s), float(pair_s), 0., []

    def check(self, stage, pair_s=None):
        if pair_s is not None:
            self.worst = max(self.worst, float(pair_s))
        hw, el = hwm(), time.time() - self.t0; self.log.append((stage, el, pair_s))
        if self.worst > self.pair_s or hw > PEAK or el > self.bound_s:
            raise _Latch(dict(stage=stage, max_pair_s=self.worst, hwm_mib=hw, elapsed_s=el, bound_s=self.bound_s, pair_limit_s=self.pair_s, peak_mib=PEAK))


def timing(log):
    """The record's timing from a guard's check log: the loading; per world its census, its pairs' seconds and the world (seconds)."""
    out = dict(load_s=None, worlds=[]); prev = cur = None
    for stage, el, ps in log:
        if stage == 'load':
            out['load_s'] = prev = el; cur = dict(census_s=None, pair_s=[], world_s=None)
        elif stage == 'census':
            cur['census_s'] = el - prev
        elif stage == 'pair':
            cur['pair_s'].append(float(ps))
        elif stage == 'world':
            cur['world_s'] = el - prev; out['worlds'].append(cur); prev = el; cur = dict(census_s=None, pair_s=[], world_s=None)
    ps_ = [x for w in out['worlds'] for x in w['pair_s']]
    return dict(out, pairs=len(ps_), pair_mean_s=(sum(ps_) / len(ps_) if ps_ else None), pair_max_s=(max(ps_) if ps_ else None))


def world_rows(day, w, world, M, sup, guard, every=AUDIT_EVERY):
    """Every pick of one selected world -> (rows, pools); statuses included. guard.check after the census and after every pair."""
    base = dict(day=day, m=w['m'], t_ms=w['t_ms'], hour=w['hour'], world_rank=w['rank'], pi=w['pi'])
    if world is None:
        return [dict(base, side=s, oid=w[f'oid{s}'], head=h, pick=p, status=f"world:{w['status']}", N=None, q=None, w=None)
                for s in (0, 1) for h, p in (('post', 'only'), ('keep', 'keep'))], []
    cand, _ = L.census_world(day, 0, dict(m=w['m'], t=w['t'], o=world['o']), world, M, sup); guard.check('census')
    rows, pools = [], []
    for s in (0, 1):
        oid = int(world['o'][s].oid); kinds = dict(sorted(Counter(str(c['kind']) for c in cand if int(c['side']) == s).items()))
        for head in ('post', 'keep'):
            mine = [c for c in cand if int(c['side']) == s and c['kind'] == head]; elig = [c for c in mine if c['status'] == 'ok']
            keys = [row_key(c) for c in elig]; by = {ckey(k_): c for k_, c in zip(keys, elig)}
            pools.append(dict(day=day, m=w['m'], t_ms=w['t_ms'], side=s, head=head, keys=keys, counts=dict(sorted(Counter(str(c['status']) for c in mine).items())),
                              kinds=kinds))
            for pick, key, N, q, wt in picks_of(keys, head, w['pi']):
                r = dict(base, side=s, oid=oid, head=head, pick=pick, N=N, q=q, w=wt)
                if key is None:
                    rows.append(dict(r, status='no_additional_post' if pick == 'extra' else 'empty_pool')); continue
                r.update(source=key['source'], row_key=key, row_rank=rank('row', key)); t0 = time.time()
                ctx = L.fork_for(by[ckey(key)], world, M, sup); fi = fork_inputs(ctx, world, M, sup, head, int(r['row_rank'][:8], 16))
                fac = L.teacher_factory(M, sup, world, every=every); res = L.pair_bound(ctx, fac, head)
                if res.get('status') == 'ok':
                    for n_, v_ in arm_ledgers(fac, res).items():
                        res['arms'][n_].update(v_)
                cz = [{k_: v_ for k_, v_ in st.items() if k_ not in ('details', 'n')} for _, st in fac.made]
                st_ = 'pair_not_ok' if res.get('status') != 'ok' else ('feature_unavailable' if fi.get('status') != 'ok' else 'ok')
                rows.append(dict(r, status=st_, fork_fingerprint=ctx['fingerprint'], fork=fi, result=res, causal=cz))
                guard.check('pair', time.time() - t0)
    return rows, pools


# ---------------------------------------------------------------- the plan
def make_plan(out, days=DAYS, pilot=False, per_hour=PER_HOUR, kp=None, src=None, files_fn=None, ht_fn=None, cfg=T.REAL, freeze_sha=L.FREEZE_SHA, assets=None,
              part_size=None, limits=None, audit_every=None):
    """The immutable plan (canonical JSON): sources, teacher, sample, per-day input identity, partitions of the run set, limits. Under the
    watchdog; the high-water checkpoint after every day; refused in a directory with a plan or a stop history."""
    out = Path(out); out.mkdir(parents=True, exist_ok=True); pp = out / 'plan.json'; C.require(not pp.exists(), 'the plan exists (immutable)')
    C.require(not RN.archive_stops(out), 'a plan directory with a stop history: use a new namespace')
    mon = RN.Sampler(out / 'plan_status.json', GUARD); mon.start()
    try:
        return _make_plan(out, pp, days, pilot, per_hour, kp, src, files_fn, ht_fn, cfg, freeze_sha, assets, part_size, limits, audit_every)
    finally:
        mon.live = False


def _make_plan(out, pp, days, pilot, per_hour, kp, src, files_fn, ht_fn, cfg, freeze_sha, assets, part_size, limits, audit_every):
    fz = (assets or (lambda: L.teacher_assets(cfg, freeze_sha)))()[2] or {}
    kp = Path(kp or json.loads(Path(cfg['population']).read_text())['inputs']['kp_dir']); src = src or KP.Tape()
    files_fn, ht_fn = files_fn or real_files, ht_fn or hours_tracked; ps = int(part_size or (PILOT_PART if pilot else PART_WORLDS)); sample, parts = {}, []
    for day in days:
        W = worlds_all(day, kp); strata, sel = sample_worlds(W, per_hour)
        sample[day] = dict(N_worlds=len(W), strata=strata, selected=sel, selected_sha256=hashlib.sha256(ckey(sel).encode()).hexdigest(),
                           inputs=identity_now(day, kp, src, files_fn, ht_fn))
        if hwm() > PEAK:
            RN.atomic_json(out / 'plan_status.json', dict(state='STOP_RESOURCE', stage='plan', day=day, hwm_mib=hwm(), peak_mib=PEAK, asof=C.stamp()))
            raise ValueError(f'the plan process high-water passed {PEAK} MiB')
        run = [w for w in sel if (w['pilot'] or not pilot)]
        for i in range(0, len(run), ps):
            parts.append(dict(part=len(parts), day=day, members=[wkey(w) for w in run[i:i + ps]]))
    plan = dict(version=VERSION, domain=DOMAIN, days=list(days), per_hour=int(per_hour), run_set='pilot' if pilot else 'full', part_size=ps, strict_day=STRICT_DAY,
                freeze_sha256=freeze_sha, libraries=fz.get('libraries'), sources=sources_now(), settings=dict(BE.SETTINGS), kp_dir=str(kp),
                feature_schema=dict(n=len(T.FEATS), sha256=hashlib.sha256(ckey(list(T.FEATS)).encode()).hexdigest()),
                limits=dict(dict(limits or LIMITS), peak_mib=PEAK, guard_mib=GUARD), audit_every=int(audit_every or AUDIT_EVERY), sample=sample, partitions=parts)
    RN.atomic_json(pp, json.loads(json.dumps(plan, default=str))); return plan


def load_plan(out):
    return json.loads((Path(out) / 'plan.json').read_text()), C.sha(Path(out) / 'plan.json')


def verify_current(plan, days, kp, src, files_fn, ht_fn, assets, cfg):
    """Review 4393ad19 P1: the executing source closure, the frozen teacher (pf_r2_labels.teacher_assets: freeze, closure, libraries) and each
    day's inputs (recomputed now, without arrays) must equal the plan's -> (models, support). Before a part is skipped, run or accepted. The memory
    limits are this module's constants (in the source closure); a plan records them and never sets them."""
    C.require(sources_now() == plan['sources'], 'stale source: the source closure differs from the plan')
    C.require(plan['feature_schema'] == dict(n=len(T.FEATS), sha256=hashlib.sha256(ckey(list(T.FEATS)).encode()).hexdigest()), 'the feature schema differs from the plan')
    M, sup, _ = (assets or (lambda: L.teacher_assets(cfg, plan['freeze_sha256'])))()
    for d in days:
        C.require(identity_now(d, kp, src, files_fn, ht_fn) == plan['sample'][d]['inputs'], f'changed input: {d}\'s inputs differ from the plan')
    return M, sup


# ---------------------------------------------------------------- one validator
def _ledger_bad(A):
    """One arm's stored ledger against its stored fills (reconcile_arm's identities and tolerances) -> a failure or None."""
    fl, sz = A.get('fill_list'), A.get('sizes')
    if not (isinstance(fl, list) and isinstance(sz, list) and len(fl) == A.get('fills') and L._h(fl) == A.get('fills_sha256')):
        return 'the stored fills differ from their hash / count'
    sg = lambda s: 1. if int(float(s)) == 0 else -1.
    cash = -sum(sg(f[2]) * float(f[3]) * float(f[4]) for f in fl); pos = sum(sg(f[2]) * float(f[3]) for f in fl); gen = {}
    for f in fl:
        gen[int(float(f[1]))] = gen.get(int(float(f[1])), 0.) + float(f[3])
    if abs(cash - A['cash']) > TOL_V or abs(pos - A['position']) > TOL_P:
        return 'cash / position differ from the fills'
    if not all(0 <= g_ < len(sz) and q_ <= sz[g_] + TOL_P for g_, q_ in gen.items()):
        return 'a generation filled beyond its size'
    return None


def policy_calls(A, g, side, first):
    """The arm teacher's required callbacks, recomputed from the arm's decision log. pf_branch_engine.decide calls the policy only in an
    eligible state, without a forced action, with a book. Counted: every decision after the fork, and the other side's at the fork tick when
    it acts after the forced one, less the engine's no-book fallbacks."""
    fb = {(float(f[0]), int(f[1])) for f in A['fallback_list']}; n = 0
    for t, s, st, a, op in A['decision_list']:
        t, s = float(t), int(s)
        if t < g or (t == g and (s == side or first != side)) or (t, s) in fb:
            continue
        n += 1
    return n


def _arm_calls_bad(A, g, side, first):
    """One arm's callbacks against its decision log (review 38febc0b item 2) -> a failure or None."""
    dl, fb, t_ = A.get('decision_list'), A.get('fallback_list'), A.get('teacher') or {}
    if not (isinstance(dl, list) and isinstance(fb, list) and L._h(dl) == A.get('decisions_sha256')):
        return 'the stored decisions differ from their hash'
    n = policy_calls(A, g, side, first)
    if not (t_.get('calls') == n and t_.get('binding_checked') == n and t_.get('binding_bad') == 0):
        return f"the teacher's calls / binding checks ({t_.get('calls')} / {t_.get('binding_checked')}) are not the {n} required callbacks, or a binding failed"
    return None


def _pair_bad(plan, day, w, r, inputs_required=True):
    """Every invariant of one pair row whose pair ran ('ok'; 'feature_unavailable' without its inputs) -> the list of failures."""
    bad = []; res, fi, rk = r.get('result') or {}, r.get('fork') or {}, r.get('row_key') or {}
    names = NAMES[r['head']]; arms = res.get('arms') or {}
    vals = [res.get('Y')] + [arms[n].get(x) for n in names if n in arms for x in ('value', 'cash', 'position', 'mark')]
    if res.get('status') != 'ok' or set(arms) != set(names) or len(vals) != 9 or not all(isinstance(v, (int, float)) and math.isfinite(v) for v in vals):
        return ['arms / finite values']
    idn, oid = res.get('ident') or {}, w[f"oid{r['side']}"]
    end = float(plan['settings']['end_ms'])
    if not (float(res.get('end', -1.)) == end and abs(float(idn.get('end_abs_ms', -1.)) - float(idn.get('start_abs_ms', 0.)) - end) < 1e-6):
        bad.append('the END is not the declared 5,000 ms')
    if not (isinstance(res.get('mark'), (int, float)) and math.isfinite(res['mark']) and all(arms[n]['mark'] == res['mark'] for n in names)):
        bad.append("an arm's mark is not the common finite mark")
    if not all(isinstance(arms[n].get('fill_list'), list) and all(0. <= float(f[0]) <= end for f in arms[n]['fill_list']) for n in names):
        bad.append('a fill outside the episode')
    if not (res.get('kind') == r['head'] and res.get('side') == r['side'] and int(round(float(res.get('k', -1)) * L.STEP)) == rk.get('decision_ms')
            and res.get('settings') == plan['settings'] and idn.get('day') == day and idn.get('side') == r['side'] and idn.get('original_episode_id') == oid
            and idn.get('instrument_id') == r['m'] and abs(float(idn.get('start_abs_ms', -1.)) - r['t_ms']) < 1e-6):
        bad.append('the executed context differs from the pick')
    if abs(res['Y'] - (arms[names[0]]['value'] - arms[names[1]]['value'])) > TOL_V:
        bad.append('Y != R_a - R_b')
    if any(abs(arms[n]['cash'] + arms[n]['position'] * arms[n]['mark'] - arms[n]['value']) > TOL_V for n in names):
        bad.append('R != cash + position x mark')
    lb = [x for x in (_ledger_bad(arms[n]) for n in names) if x]
    bad += lb
    if not lb:
        g = float(res['k']) * L.STEP; pre = lambda fl: [f for f in fl if float(f[0]) <= g]
        if pre(arms[names[0]]['fill_list']) != pre(arms[names[1]]['fill_list']):
            bad.append('the pre-fork fills differ between the arms')
    if not (res.get('reconciled') is True and res.get('common_prefork_identical') is True and all(arms[n].get('reconciled') is True for n in names)):
        bad.append('a stored reconciliation or the pre-fork flag is false')
    if res.get('fork_fingerprint') != r.get('fork_fingerprint'):
        bad.append('the result belongs to another fork')
    g, first = float(res.get('k', -1)) * L.STEP, int(res.get('fork_first_side', -1))
    cb = [x for x in (_arm_calls_bad(arms[n], g, r['side'], first) for n in names) if x]
    bad += cb
    cz, ev = r.get('causal') or [], int(plan['audit_every'])
    if len(cz) != 2 or not all(c.get('mismatch', 1) == 0 and c.get('parity_mismatch', 1) == 0 for c in cz):
        bad.append('an arm causal / parity mismatch')
    elif not cb and not all(c.get('checked') == -(-int((arms[n].get('teacher') or {}).get('calls', 0)) // ev) for c, n in zip(cz, names)):
        bad.append(f'the causal audit did not run at its declared rate (every {ev})')
    if inputs_required:
        if not (fi.get('status') == 'ok' and isinstance(fi.get('inputs'), list) and len(fi['inputs']) == plan['feature_schema']['n'] and fi.get('hidden_equal') is True
                and fi.get('rebuild_equal') is True and hashlib.sha256(np.asarray(fi['inputs'], np.float32).tobytes()).hexdigest() == fi.get('inputs_sha256')):
            bad.append('the fork inputs are incomplete or not causal')
    elif fi.get('status') != 'feature_unavailable':
        bad.append('a feature exclusion without unavailable fork inputs')
    return bad


def validate(plan, psha, part, rows, pools, rec=None, raw=None, enforce_current=True):
    """THE shared validator (resume, promotion, acceptance; review 4393ad19 P1) -> the list of failures (empty: valid). raw: the (part, pool)
    bytes, bound to the record."""
    bad = []; day = part['day']
    if rec is not None:
        if raw is not None and (hashlib.sha256(raw[0]).hexdigest() != rec.get('sha256') or hashlib.sha256(raw[1]).hexdigest() != rec.get('pools_sha256')):
            bad.append('the part or pool bytes differ from the record')
        if rec.get('plan_sha256') != psha or rec.get('part') != part['part'] or rec.get('members') != part['members']:
            bad.append('the record belongs to another plan or part')
        if rec.get('sources') != plan['sources']:
            bad.append('the record was produced by other sources')
        if rec.get('inputs') != plan['sample'][day]['inputs']:
            bad.append('the record was produced from other inputs')
        if rec.get('breach') != dict(seconds=False, rss=False) or not (isinstance(rec.get('hwm_mib'), (int, float)) and rec['hwm_mib'] <= PEAK):
            bad.append('the record carries a resource breach')
        if rec.get('rows') != len(rows) or rec.get('pools') != len(pools):
            bad.append('the record counts differ from the part')
    if enforce_current and sources_now() != plan['sources']:
        bad.append('the executing sources differ from the plan')
    sel = {ckey(wkey(w)): w for w in plan['sample'][day]['selected']}; members = [ckey(m) for m in part['members']]
    if not all(m in sel for m in members):
        return bad + ['a member is not a selected world']
    got, pl = {}, {}
    for r in rows:
        wk = ckey(dict(day=r.get('day'), m=r.get('m'), t_ms=r.get('t_ms'))); key = (wk, r.get('side'), r.get('head'), r.get('pick'))
        if wk not in members or r.get('head') not in PICKS or r.get('pick') not in PICKS.get(r.get('head'), ()):
            bad.append(f'an extra row {key}'); continue
        if key in got:
            bad.append(f'a duplicate row {key}'); continue
        got[key] = r
    for p in pools:
        wk = ckey(dict(day=p.get('day'), m=p.get('m'), t_ms=p.get('t_ms'))); key = (wk, p.get('side'), p.get('head'))
        if wk not in members or key in pl:
            bad.append(f'an extra or duplicate pool line {key}'); continue
        pl[key] = p
    for wk in members:
        w = sel[wk]; mine = {k_: v_ for k_, v_ in got.items() if k_[0] == wk}
        wst = {v_['status'] for v_ in mine.values() if str(v_.get('status', '')).startswith('world:')}
        if wst:
            want = {(wk, s, h, p) for s in (0, 1) for h, p in (('post', 'only'), ('keep', 'keep'))}
            if (len(wst) != 1 or set(mine) != want or any(k_[0] == wk for k_ in pl) or any(v_.get('w') is not None or v_.get('pi') != w['pi'] or v_.get('world_rank')
                                                                                         != w['rank'] or v_.get('oid') != w[f"oid{v_['side']}"] for v_ in mine.values())):
                bad.append(f'{wk}: a world status does not cover exactly its four picks')
            continue
        for s in (0, 1):
            for h in ('post', 'keep'):
                p = pl.get((wk, s, h))
                if p is None:
                    bad.append(f'{wk} side {s} {h}: a missing pool line'); continue
                ks = p.get('keys') or []
                if not (all(k.get('day') == day and k.get('side') == s and k.get('kind') == h and k.get('original_episode_id') == w[f'oid{s}']
                            and k.get('instrument_id') == w['m'] for k in ks) and (p.get('counts') or {}).get('ok', 0) == len(ks)
                        and sum((p.get('counts') or {}).values()) == (p.get('kinds') or {}).get(h, 0)):
                    bad.append(f'{wk} side {s} {h}: the pool line is inconsistent'); continue
                try:
                    exp = picks_of(ks, h, w['pi'])
                except ValueError as e:
                    bad.append(f'{wk} side {s} {h}: {e}'); continue
                if {(wk, s, h, x[0]) for x in exp} != {k_ for k_ in mine if k_[1:3] == (s, h)}:
                    bad.append(f'{wk} side {s} {h}: the stored picks differ from the pool\'s'); continue
                for pick, key, N, q, wt in exp:
                    r = mine[(wk, s, h, pick)]; st = r.get('status')
                    same_w = (wt is None and r.get('w') is None and r.get('q') is None) or (wt is not None and isinstance(r.get('w'), float) and isinstance(r.get('q'), float)
                                                                                           and abs(r['w'] - wt) <= 1e-12 * wt and abs(r['q'] - q) <= 1e-15)
                    if not (r.get('pi') == w['pi'] and r.get('world_rank') == w['rank'] and r.get('oid') == w[f'oid{s}'] and r.get('N') == N and same_w):
                        bad.append(f'{wk} side {s} {pick}: N / q / w / world differ from the pool and the plan'); continue
                    if key is None:
                        if st != ('no_additional_post' if pick == 'extra' else 'empty_pool'):
                            bad.append(f'{wk} side {s} {pick}: an empty pick with status {st}')
                        continue
                    if st not in PICKED or r.get('row_key') != key or r.get('row_rank') != rank('row', key) or r.get('source') != key['source']:
                        bad.append(f'{wk} side {s} {pick}: the picked row differs from the pool\'s ({st})'); continue
                    res_, fi_ = r.get('result') or {}, r.get('fork') or {}
                    exp_st = 'pair_not_ok' if res_.get('status') != 'ok' else ('feature_unavailable' if fi_.get('status') != 'ok' else 'ok')
                    if st != exp_st:
                        bad.append(f'{wk} side {s} {pick}: the status {st} is not its result\'s ({exp_st})'); continue
                    if st == 'pair_not_ok':                                                 # a failed pair: only a declared missing-data outcome
                        if res_.get('status') not in PAIR_DATA_STATUSES or res_.get('fork_fingerprint') != r.get('fork_fingerprint'):
                            bad.append(f"{wk} side {s} {pick}: the pair outcome {res_.get('status')} is a failure, not an exclusion")
                        continue
                    bad += [f'{wk} side {s} {pick}: {x}' for x in _pair_bad(plan, day, w, r, inputs_required=(st == 'ok'))]
    return bad


def gz_bytes(rows):
    """Deterministic gzip of the rows (sorted-key JSON per line)."""
    buf = io.BytesIO()
    with gzip.GzipFile(filename='', mode='wb', fileobj=buf, mtime=0) as g:
        g.write(b''.join((json.dumps(r, sort_keys=True, default=str) + '\n').encode() for r in rows))
    return buf.getvalue()


def gz_rows(b):
    return [json.loads(x) for x in gzip.decompress(b).decode().splitlines() if x]


def held(out, psha):
    """Archived stops of this output, each held until a disposition of THIS plan names its identity (pf_r2_runner's rule) -> the undisposed."""
    hist = RN.archive_stops(out); dp = Path(out) / 'disposition.json'; disp = json.loads(dp.read_text()) if dp.exists() else {}
    ok_ids = {e['id'] for e in disp.get('entries', []) if e.get('id') and str(e.get('reason', '')).strip()} if disp.get('plan_sha256') == psha else set()
    return [f"stop {e['file']} {e['id'][:12]}" for e in hist if e['id'] not in ok_ids]


def part_paths(out, k):
    d = Path(out) / 'parts'
    return d / f'part_{k:03d}.json', d / f'part_{k:03d}.jsonl.gz', d / f'part_{k:03d}.pools.jsonl.gz'


def accepted_part(out, plan, psha, part):
    """A promoted part (its completion record is the commit point) that passes the validator -> its record; None if not promoted; refused if
    promoted and invalid."""
    rp, fp, pp = part_paths(out, part['part'])
    if not rp.exists():
        return None
    rec = json.loads(rp.read_text()); b, pb = fp.read_bytes(), pp.read_bytes()
    C.require(hashlib.sha256(b).hexdigest() == rec.get('sha256') and hashlib.sha256(pb).hexdigest() == rec.get('pools_sha256'),
              f"part {part['part']}: a promoted part fails validation (its bytes do not hash to the record)")              # before any byte is decoded
    bad = validate(plan, psha, part, gz_rows(b), gz_rows(pb), rec, (b, pb))
    C.require(not bad, f"part {part['part']}: a promoted part fails validation ({bad[:3]})")
    return rec


def run(out, k, stop_after=None, src=None, inputs=None, feat_fn=None, files_fn=None, ht_fn=None, assets=None, cfg=T.REAL):
    """One partition in this process (header) -> its state: DONE (already accepted), PART_DONE, STOPPED (planted), STOP_RESOURCE, HELD."""
    out = Path(out); plan, psha = load_plan(out); k = int(k); part = plan['partitions'][k]; day = part['day']; lim = plan['limits']
    stp = out / f'part_{k:03d}_status.json'; RN.archive_stops(out)                           # before any status writer (review 4393ad19 P2)
    mon = RN.Sampler(stp, GUARD); mon.start(); G = Guard(lim['load_s'] + lim['world_s'] * len(part['members']), lim['pair_s'])
    try:
        und = held(out, psha)
        if und:
            RN.atomic_json(stp, dict(state='HELD', plan_sha256=psha, undisposed=und, asof=C.stamp())); return 'HELD'
        kp = Path(plan['kp_dir']); src = src or KP.Tape(); inputs = inputs or X.real_inputs; feat_fn = feat_fn or X.real_features
        files_fn, ht_fn = files_fn or real_files, ht_fn or hours_tracked
        M, sup = verify_current(plan, [day], kp, src, files_fn, ht_fn, assets, cfg)        # before a part is skipped or run
        if accepted_part(out, plan, psha, part) is not None:
            return 'DONE'
        sel = {ckey(wkey(w)): w for w in plan['sample'][day]['selected']}; members = [sel[ckey(m)] for m in part['members']]
        allw = {ckey(wkey(w)): w for w in worlds_all(day, kp, keys=[(w['m'], w['t_ms']) for w in members])}
        C.require(sorted(allw) == sorted(ckey(wkey(w)) for w in members), 'a member world is not one of the day\'s worlds')
        mk = sorted({w['m'] for w in members}); fair, depth, hrs, trk = inputs(day, mk)
        C.require(day_identity(day, kp, src, hrs, trk, files_fn) == plan['sample'][day]['inputs'], 'changed input: real_inputs\' hours / tracked differ from the plan')
        data = src.load(day, mk, src.offset(day)); prs = {int(W_[0]['m']): W_ for W_ in src.pairs(data)}; data = None
        G.check('load'); rows, pools, n_w = [], [], 0; limit = stop_after if stop_after is not None else STOP_AFTER
        for w in members:
            if limit is not None and n_w >= limit:
                RN.atomic_json(stp, dict(state='STOPPED', plan_sha256=psha, part=k, worlds_done=n_w, asof=C.stamp())); return 'STOPPED'
            aw = allw[ckey(wkey(w))]; world, wst = L.world_bind(day, dict(m=w['m'], t=aw['t'], o=aw['o']), prs, fair, depth, feat_fn)
            wr, wp = world_rows(day, dict(w, t=aw['t'], status=wst), world, M, sup, G, plan['audit_every']); rows += wr; pools += wp; world = None; n_w += 1
            G.check('world')
        rp, fp, pp = part_paths(out, k); rp.parent.mkdir(exist_ok=True); b, pb = gz_bytes(rows), gz_bytes(pools)
        rec = dict(plan_sha256=psha, part=k, day=day, members=part['members'], file=fp.name, pools_file=pp.name, rows=len(rows), pools=len(pools),
                   statuses=dict(sorted(Counter(r['status'] for r in rows).items())), sources=sources_now(), inputs=plan['sample'][day]['inputs'],
                   elapsed_s=time.time() - G.t0, max_pair_s=G.worst, hwm_mib=hwm(), breach=dict(seconds=False, rss=False), timing=timing(G.log),
                   sha256=hashlib.sha256(b).hexdigest(), pools_sha256=hashlib.sha256(pb).hexdigest(), asof=C.stamp())
        bad = validate(plan, psha, part, gz_rows(b), gz_rows(pb), rec, (b, pb))
        C.require(not bad, f'part {k}: refused at promotion ({bad[:3]})')
        for p_, b_ in ((fp, b), (pp, pb)):
            p_.with_name(p_.name + '.tmp').write_bytes(b_); os.replace(p_.with_name(p_.name + '.tmp'), p_)
        RN.atomic_json(rp, rec); RN.atomic_json(stp, dict(state='PART_DONE', plan_sha256=psha, part=k, asof=C.stamp())); return 'PART_DONE'
    except _Latch as e:
        RN.atomic_json(stp, dict(state='STOP_RESOURCE', plan_sha256=psha, part=k, latch=e.args[0], asof=C.stamp())); return 'STOP_RESOURCE'
    finally:
        mon.live = False


def finalize(out, src=None, files_fn=None, ht_fn=None, assets=None, cfg=T.REAL):
    """Accepted publication: no undisposed stop; the executing sources, teacher and every day's inputs = the plan's; the partitions cover the run
    set exactly once; every partition promoted and valid (the shared validator); the high-water <= the checkpoint -> accepted.json (the
    accepted-output manifest); refused otherwise."""
    out = Path(out); plan, psha = load_plan(out); RN.archive_stops(out); mon = RN.Sampler(out / 'finalize_status.json', GUARD); mon.start()
    try:
        C.require(not held(out, psha), 'finalize: an undisposed stop')
        verify_current(plan, plan['days'], Path(plan['kp_dir']), src or KP.Tape(), files_fn or real_files, ht_fn or hours_tracked, assets, cfg)
        want = sorted(ckey(wkey(w)) for d in plan['days'] for w in plan['sample'][d]['selected'] if w['pilot'] or plan['run_set'] == 'full')
        C.require(sorted(ckey(m) for p in plan['partitions'] for m in p['members']) == want and all(m['day'] == p['day'] for p in plan['partitions'] for m in p['members']),
                  'finalize: the partitions do not cover the run set exactly once')
        parts, n, st, wt, pairs = [], 0, Counter(), Counter(), Counter()
        for part in plan['partitions']:
            rec = accepted_part(out, plan, psha, part); C.require(rec is not None, f"finalize: part {part['part']} is not promoted")
            rows = gz_rows(part_paths(out, part['part'])[1].read_bytes()); n += len(rows)
            parts.append(dict(part=part['part'], day=part['day'], sha256=rec['sha256'], pools_sha256=rec['pools_sha256'], rows=len(rows)))
            for r in rows:
                st[f"{r['day']} {r['head']} {r['status']}"] += 1
                if r['w'] is not None:
                    wt[f"{r['day']} {r['head']}"] += r['w']
                pairs[r['day']] += int(r['status'] in PICKED)
        C.require(hwm() <= PEAK, f'finalize: the process high-water passed {PEAK} MiB')
        acc = dict(version=VERSION, plan_sha256=psha, run_set=plan['run_set'], audit_every=plan['audit_every'], parts=parts, rows=n, pairs=dict(pairs), statuses=dict(sorted(st.items())),
                   weight_totals=dict(sorted(wt.items())), dataset_sha256=hashlib.sha256(ckey(parts).encode()).hexdigest(), current_sources_equal=True,
                   current_inputs_equal=True, teacher_verified=True, accepted=True, asof=C.stamp())
        RN.atomic_json(out / 'accepted.json', acc); return acc
    finally:
        mon.live = False


if __name__ == '__main__':
    a = sys.argv[1:]
    if a == ['selftest']:
        import pf_r2_label_sample_selftest as ST
        ST.run(sys.modules[__name__])
    elif a[:1] == ['plan'] and (len(a) == 2 or (len(a) == 3 and a[2] == 'pilot')):
        p_ = make_plan(a[1], pilot=len(a) == 3)
        print(json.dumps(dict(run_set=p_['run_set'], partitions=len(p_['partitions']), N_worlds={d: s['N_worlds'] for d, s in p_['sample'].items()},
                              selected={d: len(s['selected']) for d, s in p_['sample'].items()})))
    elif a[:1] == ['run'] and len(a) in (3, 4):
        st_ = run(a[1], int(a[2]), stop_after=int(a[3]) if len(a) == 4 else None); print(st_); sys.exit(0 if st_ in ('PART_DONE', 'DONE', 'STOPPED') else 2)
    elif a[:1] == ['finalize'] and len(a) == 2:
        print(json.dumps({k_: v_ for k_, v_ in finalize(a[1]).items() if k_ != 'parts'}))
    else:
        sys.exit('usage: pf_r2_label_sample.py selftest | plan <out> [pilot] | run <out> <part> [stop_after_worlds] | finalize <out>')
