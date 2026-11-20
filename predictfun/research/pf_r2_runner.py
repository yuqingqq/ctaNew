"""P-2026-005 R2-A full-day launch gate (review 4cd7b45f): a bounded, resumable census runner for pf_r2_shadow.

PLAN (immutable; its SHA-256 binds every record): the day; the population (every selected original id, its hash); the partition memberships
 (canonical (market, original id) chunks of at most part_size); the source hashes of the shadow module actually used, this runner and their
 dependencies; the keep / pull receipt and output hashes; the day's input hours and their file hashes; the market listing and the tape offset;
 the engine settings; the sampling versions (r2_shadow_sampling_v1 and the causal audit sample, r2_full_causal_v1 by default); the ordered
 107-input schema and dtype; the cost convention; the resource limits. A different plan for the same directory is refused.
RUN: each pending part is processed (pf_r2_shadow.process_partition, with its population guards), written to a temporary file, validated (the
 row count from the parquet metadata against the frame; then its hash) and promoted by an atomic rename; then its completion record (plan hash,
 members, file name and hash, row / key / status / weight counts, the support contribution, the label and causal readings with one audit record
 per check, elapsed seconds and memory) is written atomically. The record is the only completion marker: a promoted file without a record is
 unfinished, and is rerun.
RESUME: a record is accepted only if it binds this plan's hash, the plan's members for that part and the actual file's hash; any other record
 refuses the run (a conflict is never overwritten), as does an original in two accepted parts. A failed part keeps its failure record and is
 never rerun. The sources and the inputs must equal the plan's: a stale source or a changed input is refused before any part runs.
STOP RULE: after each part, partition elapsed > limits.partition_seconds or the process high-water RSS (VmHWM: loading, processing, sorting,
 writing; never reset, so a peak that recedes still counts) > limits.peak_rss_mib latches STOP_RESOURCE: that part's record is kept with its
 breach, no next part starts, and the state is never COMPLETE. The same high-water is checked after loading, before the first part. A sampler
 thread records each part's own peak; as a watchdog it ends the process (STOP_HARD_RSS, exit 3, nothing promoted) above
 limits.guard_rss_mib during work. run_light's 4 GiB address-space cap stays.
FINALIZE: from the records alone - never all parts in memory: every part accepted and its file re-hashed, the population exactly once, the
 readings and the support summed. The support and census JSONs are written last; the state is COMPLETE only if every part passed, the label
 and causal verdicts pass and finalization stayed within the memory limit.
SHARDS (USER 2026-11-20: "Yea 4 cores to make it faster"): run(shard=(k, n)) processes only the parts i with i % n == k, loads only their
 markets' inputs (the day's input hours and their hashes do not depend on the markets) and keeps its own status file; n shards of one plan
 run in parallel processes, one per research slot, and finalize runs once after all of them. Records, parts and readings do not depend on
 the shard that produced them.
Usage (run_light, PF_SLOT=claude..claude4): pf_r2_runner.py selftest | plan <day> <tag> [all|<per_side>] | run <day> <tag> [k/n] |
 finalize <day> <tag>
"""
import contextlib
import hashlib
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pf_r2_shadow as R2                                 # noqa: E402

BE, KP, X, QP, C, pd, I = R2.BE, R2.KP, R2.X, R2.QP, R2.C, R2.pd, R2.I
MARK = 'R2 RUNNER'
PLAN_VERSION = 'r2_full_plan_v1'
LIMITS = dict(partition_seconds=900., peak_rss_mib=1536., guard_rss_mib=2048., address_space_gib=4.)
DEPS = ('pf_branch_engine.py', 'pf_pipeline_export.py', 'pf_keep_pull.py', 'pf_queue_place_test.py', 'pf_flow_price_data.py', 'pf_stop_model.py',
        'pf_flow_price_common.py')
STOP_AFTER = None                                          # planted (selftest / the gate's interruption): stop after this many parts in one run
CRASH_AFTER_TMP = None                                     # planted (selftest only): raise in part i after its temporary file is written
WRITE_DROP = False                                         # planted (selftest only): the written part loses its last row
SPIKE_MIB = 0.                                             # planted (selftest only): a transient allocation inside each part (a peak that recedes)
WATCHDOG_EXIT = None                                       # the selftest injects a recorder; production: None (os._exit ends the process)


def rss():
    """(VmRSS, VmHWM) of this process in MiB; VmHWM is the kernel's high-water mark since the process started (never reset here)."""
    t = open('/proc/self/status').read()
    return int(re.search(r'VmRSS:\s+(\d+)', t).group(1)) / 1024., int(re.search(r'VmHWM:\s+(\d+)', t).group(1)) / 1024.


def canon(x):
    return json.dumps(x, sort_keys=True, separators=(',', ':'), default=str)


def atomic_json(path, obj):
    """Written to a temporary name, then renamed: a reader sees the old file or the new one, never a partial one."""
    path = Path(path); tmp = path.with_name(path.name + '.tmp'); tmp.write_text(json.dumps(obj, indent=1, default=str)); os.replace(tmp, path)


def sources(mod):
    """The source hashes a plan binds: the shadow module actually used (its own file), this runner and the dependencies."""
    out = {'pf_r2_shadow.py': C.sha(Path(mod.__file__).resolve()), 'pf_r2_runner.py': C.sha(Path(__file__).resolve())}
    out.update({n: C.sha(HERE / n) for n in DEPS})
    return out


class Sampler(threading.Thread):
    """VmRSS every `period` s: each part's own peak (attribution; the stop rule reads the never-reset VmHWM), and the hard watchdog - above
    hard_mib the run status STOP_HARD_RSS is written and the process ends (exit 3) with nothing promoted."""

    def __init__(self, status_path, hard_mib, period=.02, exit=None):
        super().__init__(daemon=True); self.status_path, self.hard, self.period, self.peak, self.live = Path(status_path), float(hard_mib), period, 0., True
        self.exit = os._exit if exit is None else exit                                       # the selftest injects a recorder; production ends the process

    def poll(self):
        """One sample -> True when the hard limit was breached (the status written, the exit called)."""
        r = rss()[0]; self.peak = max(self.peak, r)
        if r > self.hard:
            atomic_json(self.status_path, dict(state='STOP_HARD_RSS', rss_mib=r, guard_rss_mib=self.hard, asof=C.stamp())); self.exit(3); return True
        return False

    def run(self):
        while self.live:
            if self.poll():
                return
            time.sleep(self.period)

    def reset(self):
        self.peak = rss()[0]


def make_plan(d, mod, day, per_side, population, part_size, kp, src, markets, inputs, causal, causal_every, keep_fills, limits, tag):
    """The immutable plan -> (its path, the inputs loaded in this process). Refused if a different plan exists for the directory. The plan binds
    the content only, never the output location: `tag` is unused here (finalize derives the receipt names from the directory; the first gate run
    of 2026-11-20 01:32Z had two plans differing only in their tag, so a resumed copy of one content could not show the same plan hash)."""
    C.require(day in BE.SLICE_DAYS, 'consumed days only (Oct 23-27)'); d = Path(d); kp = Path(kp or KP.OUT); src = src or KP.Tape()
    d.mkdir(parents=True, exist_ok=True); lim0 = dict(limits or LIMITS)
    mon = Sampler(d / 'plan_status.json', lim0['guard_rss_mib'], exit=WATCHDOG_EXIT); mon.start()            # review e6b0efa4: the watchdog over loading
    try:
        return _make_plan(d, mod, day, per_side, population, part_size, kp, src, markets, inputs, causal, causal_every, keep_fills, limits)
    finally:
        mon.live = False


def _make_plan(d, mod, day, per_side, population, part_size, kp, src, markets, inputs, causal, causal_every, keep_fills, limits):
    r_ = json.loads((kp / day / 'receipt.json').read_text())
    C.require(all(C.sha(kp / day / x) == h for x, h in r_['outputs'].items()), 'keep-pull outputs changed since their receipt')
    O = pd.read_parquet(kp / day / 'orders.parquet'); C.require(population in ('first_per_side', 'all'), 'undeclared population')
    sel = O.copy() if population == 'all' else pd.concat([O[O.side == s_].sort_values('oid').head(int(per_side)) for s_ in (0, 1)], ignore_index=True)
    sel_ids = sorted([int(s_), int(x)] for s_, x in zip(sel.side, sel.oid))
    listed = sorted(set(int(m) for m in (src.markets(day) if markets is None else markets))); need = sorted(set(sel.m.astype(int)) & set(listed))
    loaded = (inputs or X.real_inputs)(day, need) if need else ({}, None, [], {})
    parts = [dict(part=i, members=sorted([int(a), int(b)] for a, b in zip(P.side, P.oid)), markets=sorted(set(int(m) for m in P.m)))
             for i, P in enumerate(mod.partitions(sel, part_size))]
    allm = [tuple(x) for p in parts for x in p['members']]
    C.require(len(allm) == len(set(allm)) and set(allm) == set(map(tuple, sel_ids)), 'plan: the partitions must hold every selected original exactly once')
    plan = dict(version=PLAN_VERSION, day=day, manifest=mod.MANIFEST,
                population=dict(kind=population, per_side=int(per_side), selected=len(sel_ids), selected_ids=sel_ids,
                                selected_ids_sha256=hashlib.sha256(json.dumps(sel_ids).encode()).hexdigest(),
                                by_status=sel.status.astype(str).value_counts().to_dict()),
                part_size=int(part_size), partitions=parts, sources=sources(mod),
                inputs=dict(kp_dir=str(kp), kp_receipt=C.sha(kp / day / 'receipt.json'), kp_outputs=r_['outputs'], hours=list(loaded[2]), tracked=loaded[3],
                            listed=listed, offset=src.offset(day)),
                engine_settings=BE.SETTINGS, schema=dict(inputs=mod.INPUTS, ctx=mod.CTX, dtype='float32', sort_key=mod.SORT_KEY), costs=mod.COSTS,
                causal=dict(on=bool(causal), every=int(causal_every) if causal_every else None,
                            sample=f'every_{int(causal_every)}' if causal_every else mod.CAUSAL_VERSION, draws=mod.VENUE_DRAWS),
                keep_fills=bool(keep_fills), limits=dict(limits or LIMITS))
    plan = json.loads(json.dumps(plan, default=str)); d.mkdir(parents=True, exist_ok=True); pp = d / 'plan.json'
    if pp.exists():
        C.require(canon(json.loads(pp.read_text())) == canon(plan), 'a different plan exists for this directory (a plan is immutable)')
    else:
        atomic_json(pp, plan)
    return pp, loaded


def accept(d, plan, psha, i):
    """Part i's completion record, accepted only if it binds this plan's hash, the plan's members for part i and the actual file's hash ->
    the record, None (no record: the part is pending), or a refusal (a conflict is never overwritten)."""
    rp = Path(d) / 'parts' / f'part_{i:04d}.json'
    if not rp.exists():
        return None
    rec = json.loads(rp.read_text())
    C.require(rec.get('plan_sha256') == psha, f'part {i}: a completion record of another plan')
    C.require(rec.get('part') == i and rec.get('members') == plan['partitions'][i]['members'], f'part {i}: the record\'s members differ from the plan\'s')
    f = Path(d) / 'parts' / str(rec.get('file'))
    C.require(f.exists() and C.sha(f) == rec.get('sha256'), f'part {i}: the part file is missing or its hash differs from the record')
    return rec


STOP_STATES = ('STOP_RESOURCE', 'STOP_HARD_RSS')


def archive_stops(d):
    """Review 8cc1dc49 P2: the stop HISTORY. Every status file of a process that stopped on a resource limit is appended once, by the sha256
    of its bytes, to the directory's append-only stops.jsonl. Every process does this before it writes a status, so a later overwrite of the
    same file cannot erase a stop -> the history (one event per distinct stop)."""
    hp = Path(d) / 'stops.jsonl'; hist = [json.loads(x) for x in hp.read_text().splitlines() if x.strip()] if hp.exists() else []; seen = {h['id'] for h in hist}
    for p in sorted(Path(d).glob('*status*.json')):
        raw = p.read_bytes(); st = json.loads(raw)
        if st.get('state') in STOP_STATES and hashlib.sha256(raw).hexdigest() not in seen:
            ev = dict(id=hashlib.sha256(raw).hexdigest(), file=p.name, state=st['state'], content=st, archived=C.stamp()); hist.append(ev); seen.add(ev['id'])
            with open(hp, 'a') as f:
                f.write(json.dumps(ev, sort_keys=True) + '\n')
    return hist


def held_breaches(d, accepted, psha):
    """Review e6b0efa4 finding 1, review 8cc1dc49 P2: the held breaches. These are every accepted record with a breach (identity: the sha256
    of its completion record) and every archived stop event (identity: the sha256 of the stop status's bytes). A breach is covered only by a
    disposition of THIS plan that names its identity with a reason: disposition.json = {plan_sha256, entries: [{id, name, reason}]}. A file
    name never covers a later event of the same name, and an older name-only disposition covers nothing -> dict(parts, stops, undisposed,
    disposition)."""
    hist = archive_stops(d)
    parts = sorted((int(r['part']), C.sha(Path(d) / 'parts' / f"part_{int(r['part']):04d}.json")) for r in accepted.values() if any(r.get('breach', {}).values()))
    dp = Path(d) / 'disposition.json'; disp = json.loads(dp.read_text()) if dp.exists() else {}
    ok_ids = {e['id'] for e in disp.get('entries', []) if e.get('id') and str(e.get('reason', '')).strip()} if disp.get('plan_sha256') == psha else set()
    undisposed = [f'part {i} {h[:12]}' for i, h in parts if h not in ok_ids] + [f"stop {e['file']} {e['id'][:12]}" for e in hist if e['id'] not in ok_ids]
    return dict(parts=[i for i, _ in parts], part_ids=[h for _, h in parts], stops=[dict(file=e['file'], id=e['id'], state=e['state']) for e in hist],
                undisposed=undisposed, disposition=disp or None)


def run(d, mod=None, src=None, inputs=None, loaded=None, feat_fn=None, causal_feat=None, causal_perturb=None, shard=None):
    """Process every pending part of the plan in directory d (resume: accepted parts are kept, unfinished ones rerun); shard=(k, n): only the
    parts i with i % n == k, with only their markets' inputs -> the state of those parts: PARTS_DONE, STOPPED (a planted interruption),
    STOP_RESOURCE (a latched breach in this process), HELD (an earlier breach or stop without its disposition: nothing run); refusals raise."""
    mod = mod or R2; d = Path(d); plan = json.loads((d / 'plan.json').read_text()); psha = C.sha(d / 'plan.json'); lim = plan['limits']; day = plan['day']
    st_name = 'run_status.json' if shard is None else f'run_status_{int(shard[0])}of{int(shard[1])}.json'
    mon = Sampler(d / st_name, lim['guard_rss_mib'], exit=WATCHDOG_EXIT); mon.start()                    # review e6b0efa4: from the start, loading included
    try:
        return _run(d, mod, plan, psha, lim, day, st_name, mon, src, inputs, loaded, feat_fn, causal_feat, causal_perturb, shard)
    finally:
        mon.live = False


def _run(d, mod, plan, psha, lim, day, st_name, mon, src, inputs, loaded, feat_fn, causal_feat, causal_perturb, shard):
    C.require(sources(mod) == plan['sources'], 'stale source: a module or a dependency changed since the plan')
    kp = Path(plan['inputs']['kp_dir'])
    C.require(C.sha(kp / day / 'receipt.json') == plan['inputs']['kp_receipt'] and all(C.sha(kp / day / x) == h for x, h in plan['inputs']['kp_outputs'].items()),
              'changed input: the keep-pull receipt or its outputs differ from the plan')
    src = src or KP.Tape(); C.require(canon(src.offset(day)) == canon(plan['inputs']['offset']), 'changed input: the tape offset differs from the plan')
    C.require(shard is None or (len(shard) == 2 and 0 <= int(shard[0]) < int(shard[1])), f'an invalid shard {shard}')
    mine = [i for i in range(len(plan['partitions'])) if shard is None or i % int(shard[1]) == int(shard[0])]
    listed = set(plan['inputs']['listed']); need = sorted(set(m for i in mine for m in plan['partitions'][i]['markets']) & listed)
    fair, depth, hrs, tracked = loaded if loaded is not None else ((inputs or X.real_inputs)(day, need) if need else ({}, None, [], {}))
    C.require(list(hrs) == plan['inputs']['hours'] and canon(tracked) == canon(plan['inputs']['tracked']), 'changed input: the day\'s input hours or their hashes differ from the plan')
    O = pd.read_parquet(kp / day / 'orders.parquet'); O['_k'] = O.side.astype(np.int64) * 10 ** 12 + O.oid.astype(np.int64)   # integer member keys
    key = lambda mm: {int(a_) * 10 ** 12 + int(b_) for a_, b_ in mm}
    pdir = d / 'parts'; pdir.mkdir(exist_ok=True); accepted = {}; seen = set()
    for i in range(len(plan['partitions'])):
        rec = accept(d, plan, psha, i)
        if rec is not None:
            mem = set(map(tuple, rec['members'])); C.require(not (mem & seen), f'part {i}: an original already in another accepted part'); seen |= mem; accepted[i] = rec
    pending = [i for i in mine if i not in accepted]
    status = d / st_name; latch = None; state = 'RUNNING'; n_run = 0
    held = held_breaches(d, accepted, psha)                                                 # reviews e6b0efa4 / 8cc1dc49: a resource stop is carried
    if held['undisposed'] and pending:
        latch = dict(reason='an earlier resource breach or stop without a recorded disposition', **held); state = 'HELD'   # no work, no new breach (8cc1dc49)
    try:
        hw0 = rss()[1]
        if latch is None and pending and hw0 > lim['peak_rss_mib']:
            latch = dict(reason='the high-water RSS after loading, before the first part', hwm_mib=hw0); state = 'STOP_RESOURCE'
        for i in (pending if latch is None else []):
            if STOP_AFTER is not None and n_run >= STOP_AFTER:
                state = 'STOPPED'; break
            P = O[O._k.isin(key(plan['partitions'][i]['members']))].sort_values(['m', 'oid'], kind='stable').drop(columns=['_k'])
            prev = None
            if i > 0:
                prev = next(O[O._k.isin(key(plan['partitions'][i - 1]['members']))].sort_values(['m', 'oid'], kind='stable').itertuples())
            t0 = time.time(); mon.reset()
            D, info = mod.process_partition(day, i, P, listed, src, src.offset(day), fair, depth, feat_fn or X.real_features, causal=plan['causal']['on'],
                                            causal_every=plan['causal']['every'], causal_feat=causal_feat, causal_perturb=causal_perturb,
                                            keep_fills=plan['keep_fills'], prev=prev)
            if SPIKE_MIB:                                                                   # planted: a peak that recedes before the end of the part
                z = np.ones(int(SPIKE_MIB * 2 ** 20 / 8)); z[::512] = 2.; time.sleep(.1); del z
            f_ = f'part_{i:04d}.parquet'; tmp = pdir / (f_ + '.tmp'); D = D.drop(columns=['fills'], errors='ignore')
            (D.iloc[:-1] if WRITE_DROP else D).to_parquet(tmp, index=False); n_rows = len(D); D = None
            if CRASH_AFTER_TMP is not None and i == CRASH_AFTER_TMP:
                raise RuntimeError(f'{MARK}: planted crash after the temporary write of part {i}')
            import pyarrow.parquet as pq
            C.require(pq.read_metadata(tmp).num_rows == n_rows == info['rows'], f'part {i}: the written part has {pq.read_metadata(tmp).num_rows} rows, the frame {n_rows}')
            sha = C.sha(tmp); os.replace(tmp, pdir / f_)
            el = time.time() - t0; hw = rss()[1]; breach = dict(seconds=bool(el > lim['partition_seconds']), rss=bool(hw > lim['peak_rss_mib']))
            cz = info['causal']; ok = info['recon']['failed'] == 0 and cz.get('failed', 0) == 0 and cz.get('not_evaluable', 0) == 0
            C.require(info['members'] == plan['partitions'][i]['members'], f'part {i}: the processed members differ from the plan\'s')
            rec = dict(plan_sha256=psha, part=i, members=info['members'], file=f_, sha256=sha, rows=info['rows'], keys=info['rows'], sampled_ok=info['sampled_ok'],
                       status_counts=info['status_counts'], weight_groups=info['weight_groups'], weight_groups_one=info['weight_groups_one'],
                       recon=info['recon'], refusals=info['refusals'], causal=cz, support=info['support'], elapsed_s=el, part_peak_rss_mib=mon.peak,
                       hwm_mib=hw, breach=breach, status='PASS' if ok else 'FAILED', asof=C.stamp())
            atomic_json(pdir / f'part_{i:04d}.json', rec); accepted[i] = rec; n_run += 1; C.guard('r2 runner')
            print(json.dumps(dict(partition=i, rows=rec['rows'], seconds=round(el, 1), part_peak_rss_mib=round(mon.peak), hwm_mib=round(hw), status=rec['status'])), flush=True)
            if breach['seconds'] or breach['rss']:
                latch = dict(part=i, elapsed_s=el, hwm_mib=hw, **breach); state = 'STOP_RESOURCE'; break
    finally:
        pass
    left = [i for i in mine if i not in accepted]
    if state == 'RUNNING':
        state = 'PARTS_DONE' if not left else 'STOPPED'
    atomic_json(status, dict(state=state, plan_sha256=psha, shard=list(shard) if shard is not None else None, done=sorted(i for i in accepted if i in mine),
                             pending=left, latch=latch, hwm_mib=rss()[1], asof=C.stamp()))
    return state


def finalize(d, mod=None):
    """The receipt from the completion records alone (never the parts' rows): every part accepted and its file re-hashed, the population
    exactly once, the readings and the support summed; the support and census JSONs written last -> (the directory, the support, the readings)."""
    mod = mod or R2; d = Path(d); plan = json.loads((d / 'plan.json').read_text()); psha = C.sha(d / 'plan.json'); lim = plan['limits']; day = plan['day']
    C.require(sources(mod) == plan['sources'], 'stale finalizer: the finalizing sources differ from the plan (review e6b0efa4 finding 2)')
    mon = Sampler(d / 'finalize_status.json', lim['guard_rss_mib'], exit=WATCHDOG_EXIT); mon.start()
    try:
        return _finalize(d, mod, plan, psha, lim, day)
    finally:
        mon.live = False


def _finalize(d, mod, plan, psha, lim, day):
    C.require(d.name.startswith(f'census_{day}'), f'finalize: the directory {d.name} is not a census of {day}'); tag = d.name[len(f'census_{day}'):]
    recs, seen = [], set()
    for i in range(len(plan['partitions'])):
        rec = accept(d, plan, psha, i); C.require(rec is not None, f'finalize: part {i} has no accepted completion record')
        mem = set(map(tuple, rec['members'])); C.require(not (mem & seen), f'finalize: part {i} repeats an original'); seen |= mem; recs.append(rec)
    sel = set(map(tuple, plan['population']['selected_ids']))
    C.require(seen == sel and sum(len(r['members']) for r in recs) == len(sel), 'finalize: the represented originals differ from the plan\'s selection')
    S = mod.support_merge([r['support'] for r in recs]); add = lambda k: sum(int(r[k]) for r in recs)
    recon = {k: sum(int(r['recon'][k]) for r in recs) for k in ('checked', 'failed', 'no_fill')}
    refusals, stc, comp = {}, {}, {}
    for r in recs:
        for k, v in r['refusals'].items():
            refusals[k] = refusals.get(k, 0) + int(v)
        for k, v in r['status_counts'].items():
            stc[k] = stc.get(k, 0) + int(v)
        for k in ('stored_vs_rebuild', 'stored_vs_perturbed', 'stored_vs_visibility', 'firing', 'not_evaluable', 'retried'):
            comp[k] = comp.get(k, 0) + int(r['causal'].get(k, 0))
    ck, cf = sum(int(r['causal'].get('checked', 0)) for r in recs), sum(int(r['causal'].get('failed', 0)) for r in recs)
    cverd = ('NOT-RUN' if not plan['causal']['on'] else 'FAIL' if (ck == 0 or cf > 0) else 'NOT-EVALUABLE' if comp['not_evaluable'] > 0 else 'PASS')
    lverd = 'PASS' if (recon['checked'] > 0 and recon['failed'] == 0) else 'FAIL'
    hw = rss()[1]; parts_ok = all(r['status'] == 'PASS' for r in recs); held = held_breaches(d, {r['part']: r for r in recs}, psha)
    state = ('STOP_RESOURCE' if (hw > lim['peak_rss_mib'] or held['undisposed']) else 'COMPLETE' if (parts_ok and lverd == 'PASS' and cverd in ('PASS', 'NOT-RUN'))
             else 'FAILED')
    pop = dict(plan['population'], represented=len(seen), reconciliation='EXACT')
    R = dict(state=state, population=pop,
             partitions=dict(count=len(recs), size=plan['part_size'], seconds=[r['elapsed_s'] for r in recs], part_peak_rss_mib=[r['part_peak_rss_mib'] for r in recs],
                             hwm_mib=[r['hwm_mib'] for r in recs], breaches=[r['part'] for r in recs if any(r['breach'].values())]),
             labels=dict(recon, verdict=lverd), refusals_by_source=refusals,
             causality=dict(checked=ck, failed=cf, sample=plan['causal']['sample'], details=[x for r in recs for x in r['causal'].get('details', [])][:10],
                            components={k: comp[k] for k in ('stored_vs_rebuild', 'stored_vs_perturbed', 'stored_vs_visibility', 'firing', 'not_evaluable')},
                            retried=comp['retried'], draws=plan['causal']['draws'], audited=sum(len(r['causal'].get('audit', [])) for r in recs), verdict=cverd),
             census_rows=add('rows'), sampled_ok_rows=add('sampled_ok'), by_source_head_status=stc,
             weights=dict(groups=add('weight_groups'), groups_summing_to_one=add('weight_groups_one')),
             peak_rss_mib=max([hw] + [float(r['hwm_mib']) for r in recs]), finalize_hwm_mib=hw, held_breaches=held)
    out = d.parent
    atomic_json(out / f'support_{day}{tag}.json', S)
    atomic_json(out / f'census_{day}{tag}.json', dict(asof=C.stamp(), day=day, manifest=plan['manifest'], state=state, costs=plan['costs'], readings=R, plan_sha256=psha,
                                                     inputs_list_sha256=hashlib.sha256(json.dumps(plan['schema']['inputs'] + plan['schema']['ctx']).encode()).hexdigest(),
                                                     inputs=plan['inputs'], sources=plan['sources'], finalizer_sources=sources(mod), provenance=KP.provenance(),
                                                     outputs={**{f'{d.name}/parts/{r["file"]}': r['sha256'] for r in recs}, f'{d.name}/plan.json': psha,
                                                              f'support_{day}{tag}.json': C.sha(out / f'support_{day}{tag}.json')}))
    print(json.dumps({k: v for k, v in R.items() if k != 'population'} | dict(population={k: v for k, v in pop.items() if k != 'selected_ids'}), indent=1, default=str), flush=True)
    return d, S, R


def census(mod=None, day=None, per_side=None, out=None, src=None, kp=None, inputs=None, feat_fn=None, markets=None, causal=True, causal_every=None,
           causal_feat=None, causal_perturb=None, tag='', keep_fills=False, part_size=None, population='first_per_side', limits=None):
    """plan -> run -> finalize in one call (a census directory already holding the same plan resumes) -> (directory, support, readings);
    (directory, None, {'state': ...}) when the run stopped."""
    mod = mod or R2; day = day or mod.DAY; per_side = mod.PER_SIDE if per_side is None else per_side; out = Path(out or mod.OUT)
    C.require(not (out / f'census_{day}{tag}.json').exists(), 'output exists'); d = out / f'census_{day}{tag}'
    _, loaded = make_plan(d, mod, day, per_side, population, part_size or mod.PART_SIZE, kp, src, markets, inputs, causal, causal_every, keep_fills, limits, tag)
    st = run(d, mod, src=src, inputs=inputs, loaded=loaded, feat_fn=feat_fn, causal_feat=causal_feat, causal_perturb=causal_perturb)
    if st != 'PARTS_DONE':
        return d, None, dict(state=st)
    return finalize(d, mod)


def selftest():  # noqa: C901
    """Synthetic only: resume identity (uninterrupted = interrupted + resumed = an interrupted write + resumed); refused resumes (a corrupt part,
    a forged / overlapping record, another plan's record, a stale source, a changed input, a different plan); the stop rule (time, a receding
    peak, loading) and the watchdog; no all-part read in census / finalize (a sentinel); finalize refusing without publishing; a failed part kept
    as a failure record, never COMPLETE, never rerun; the audit sample identical across partitionings; a short write refused."""
    global STOP_AFTER, CRASH_AFTER_TMP, SPIKE_MIB, WRITE_DROP, sources, Sampler
    bad = []; src = KP.Syn(); days = list(KP.DAYS); d0 = days[0]
    import pf_queue_place_ledger as QL
    with tempfile.TemporaryDirectory(dir=str(I.K.IO)) as tmp:
        tmp = Path(tmp)
        hits, quiet = [], []                                                               # first: the watchdog's breach action in this copy of the code
        fired = Sampler(tmp / 'wd_in.json', 1., exit=hits.append).poll(); calm = Sampler(tmp / 'wd_quiet.json', 1e9, exit=quiet.append).poll()
        wdi = json.loads((tmp / 'wd_in.json').read_text()) if (tmp / 'wd_in.json').exists() else {}
        if not (fired and hits == [3] and wdi.get('state') == 'STOP_HARD_RSS' and not calm and not quiet and not (tmp / 'wd_quiet.json').exists()):
            sys.exit(f'{MARK} SELFTEST FAILED: watchdog (a precondition of every run): a hard RSS breach did not write STOP_HARD_RSS and exit 3, or a '
                     f'quiet sample did ({fired}, {hits}, {wdi}, {calm}, {quiet})')
        with QL._v1_on(src, tmp):
            for d in days:
                QP.day_run(d)
        q = io.StringIO()
        kw = dict(src=src, kp=tmp / 'kp', inputs=X.syn_inputs, feat_fn=X.syn_features(), causal_feat=X.syn_features(), causal_perturb=(X.syn_perturb, X.syn_fire))
        with contextlib.redirect_stdout(q):
            for d in days:
                KP.day_run(d, src=src, v1=tmp / 'v1', out=tmp / 'kp')
            A = census(day=d0, per_side=2, out=tmp / 'A', part_size=1, **kw)
        strip = lambda R: {k: v for k, v in R.items() if k not in ('partitions', 'peak_rss_mib', 'finalize_hwm_mib')}
        parts = lambda dd: [(json.loads(p.read_text())['sha256'], json.loads(p.read_text())['causal'].get('audit')) for p in sorted((dd / 'parts').glob('part_*.json'))]
        with contextlib.redirect_stdout(q):                                                 # the plan binds the content, never the output location
            pl1, _ = make_plan(tmp / 'P1' / f'census_{d0}_one', R2, d0, 2, 'first_per_side', 1, tmp / 'kp', src, None, X.syn_inputs, True, None, False, None, '_one')
            pl2, _ = make_plan(tmp / 'P2' / f'census_{d0}_two', R2, d0, 2, 'first_per_side', 1, tmp / 'kp', src, None, X.syn_inputs, True, None, False, None, '_two')
        if not (C.sha(pl1) == C.sha(pl2) == C.sha(A[0] / 'plan.json')):
            bad.append('plan: two directories (tags) of the same content have different plans')
        if not (A[2]['state'] == 'COMPLETE' and A[2]['causality']['checked'] > 0 and len(list((A[0] / 'parts').glob('part_*.json'))) == 4):
            bad.append(f"census: the uninterrupted run is not COMPLETE with 4 parts ({A[2].get('state')})")
        try:                                                                                # interrupted after one part, resumed
            STOP_AFTER = 1
            with contextlib.redirect_stdout(q):
                b1 = census(day=d0, per_side=2, out=tmp / 'B', part_size=1, **kw)
        finally:
            STOP_AFTER = None
        with contextlib.redirect_stdout(q):
            B = census(day=d0, per_side=2, out=tmp / 'B', part_size=1, **kw)
        if not (b1[2] == dict(state='STOPPED') and parts(B[0]) == parts(A[0]) and strip(B[2]) == strip(A[2]) and B[1] == A[1]):
            bad.append(f"resume: interrupted + resumed differs from uninterrupted ({b1[2]}, {strip(B[2]) == strip(A[2])}, {parts(B[0]) == parts(A[0])})")
        try:                                                                                # an interrupted write: the temporary file, no record
            CRASH_AFTER_TMP = 1
            with contextlib.redirect_stdout(q):
                census(day=d0, per_side=2, out=tmp / 'W', part_size=1, **kw)
            bad.append('resume: the planted crash did not interrupt the run')
        except RuntimeError:
            pass
        finally:
            CRASH_AFTER_TMP = None
        w_tmp = (tmp / 'W' / f'census_{d0}' / 'parts' / 'part_0001.parquet.tmp').exists() and not (tmp / 'W' / f'census_{d0}' / 'parts' / 'part_0001.json').exists()
        with contextlib.redirect_stdout(q):
            Wr = census(day=d0, per_side=2, out=tmp / 'W', part_size=1, **kw)
        if not (w_tmp and parts(Wr[0]) == parts(A[0]) and strip(Wr[2]) == strip(A[2])):
            bad.append(f'resume: an interrupted write was not left unfinished and rerun to the same result ({w_tmp})')

        def refused(fn, why):
            try:
                with contextlib.redirect_stdout(q):
                    fn()
            except Exception as e:                                                          # the refusal's own message, whatever the exception
                return why in str(e)
            return False
        def copy_of(name):
            dd = tmp / name / f'census_{d0}'; shutil.copytree(A[0], dd); return dd
        c1 = copy_of('X1'); pf = c1 / 'parts' / 'part_0001.parquet'; b_ = bytearray(pf.read_bytes()); b_[len(b_) // 2] ^= 0xFF; pf.write_bytes(bytes(b_))
        c2 = copy_of('X2'); r2_ = json.loads((c2 / 'parts' / 'part_0001.json').read_text()); r2_['part'] = 2       # part 1's members claimed as part 2
        (c2 / 'parts' / 'part_0002.json').write_text(json.dumps(r2_))
        c3 = copy_of('X3'); r3 = json.loads((c3 / 'parts' / 'part_0002.json').read_text()); r3['plan_sha256'] = '0' * 64; R2_ = r3
        (c3 / 'parts' / 'part_0002.json').write_text(json.dumps(R2_))
        c4 = copy_of('X4'); os.remove(c4 / 'parts' / 'part_0003.json')
        c5 = copy_of('X5'); c6 = copy_of('X6')
        rr = dict(corrupt=refused(lambda: run(c1, src=src, inputs=X.syn_inputs, feat_fn=X.syn_features()), 'hash differs from the record'),
                  overlap=refused(lambda: run(c2, src=src, inputs=X.syn_inputs, feat_fn=X.syn_features()), 'members differ'),
                  other_plan=refused(lambda: run(c3, src=src, inputs=X.syn_inputs, feat_fn=X.syn_features()), 'another plan'),
                  changed_input=refused(lambda: run(c5, src=src, inputs=lambda day, mk: X.syn_inputs(day, mk)[:3] + ({'x': '0'},), feat_fn=X.syn_features()), 'changed input'),
                  different_plan=refused(lambda: make_plan(c6, R2, d0, 2, 'first_per_side', 2, tmp / 'kp', src, None, X.syn_inputs, True, None, False, None, ''),
                                         'a different plan exists'),
                  finalize_missing=refused(lambda: finalize(c4), 'no accepted completion record'))
        src_fn = sources
        try:
            sources = lambda mod: dict(src_fn(mod), **{'pf_r2_shadow.py': '0' * 64})
            rr['stale_source'] = refused(lambda: run(c5, src=src, inputs=X.syn_inputs, feat_fn=X.syn_features()), 'stale source')
        finally:
            sources = src_fn
        if not all(rr.values()):
            bad.append(f'resume: a refusal did not fire ({rr})')
        c8 = copy_of('X8')                                                                  # review e6b0efa4 finding 2: a stale finalizer is refused
        try:
            sources = lambda mod: dict(src_fn(mod), **{'pf_r2_runner.py': '0' * 64})
            stale_fin = refused(lambda: finalize(c8), 'stale finalizer')
        finally:
            sources = src_fn
        c9 = copy_of('X9'); r9 = json.loads((c9 / 'parts' / 'part_0001.json').read_text()); r9['breach'] = dict(seconds=True, rss=False)   # finding 1: breaches held
        (c9 / 'parts' / 'part_0001.json').write_text(json.dumps(r9)); ps9 = C.sha(c9 / 'plan.json'); id9 = C.sha(c9 / 'parts' / 'part_0001.json')
        disp = lambda d_, psha_, ids: (d_ / 'disposition.json').write_text(json.dumps(dict(plan_sha256=psha_, entries=[dict(id=x, name='selftest', reason='selftest: investigated') for x in ids])))
        with contextlib.redirect_stdout(q):
            f9a = finalize(c9)[2]
            (c9 / 'disposition.json').write_text(json.dumps(dict(parts=[1], statuses=[], reason='selftest: the old name-only form')))
            f9o = finalize(c9)[2]
            disp(c9, '0' * 64, [id9]); f9p = finalize(c9)[2]
            disp(c9, ps9, [id9]); f9b = finalize(c9)[2]
        c10 = copy_of('X10'); os.remove(c10 / 'parts' / 'part_0003.json'); (c10 / 'run_status.json').write_text(json.dumps(dict(state='STOP_RESOURCE', note='first')))
        ps10 = C.sha(c10 / 'plan.json'); id10 = hashlib.sha256((c10 / 'run_status.json').read_bytes()).hexdigest()
        rn = lambda: run(c10, src=src, inputs=X.syn_inputs, feat_fn=X.syn_features(), causal_feat=X.syn_features(), causal_perturb=(X.syn_perturb, X.syn_fire))
        with contextlib.redirect_stdout(q):
            r10a = rn(); gone10 = not (c10 / 'parts' / 'part_0003.json').exists()               # held: the stop has no disposition
            st10a = json.loads((c10 / 'run_status.json').read_text())                          # the held run overwrote the stop's file ...
            r10a2 = rn()                                                                        # ... yet its event is in the history: held again
            disp(c10, ps10, [id10]); r10b = rn(); back10 = (c10 / 'parts' / 'part_0003.json').exists()
            (c10 / 'parts' / 'part_0003.json').unlink(missing_ok=True); (c10 / 'run_status.json').write_text(json.dumps(dict(state='STOP_HARD_RSS', rss_mib=2100)))
            r10c = rn(); gone10c = not (c10 / 'parts' / 'part_0003.json').exists()             # review 8cc1dc49: a NEW stop, same file name
        hist10 = [json.loads(x) for x in (c10 / 'stops.jsonl').read_text().splitlines()] if (c10 / 'stops.jsonl').exists() else []
        held_ok = (stale_fin and f9a['state'] == 'STOP_RESOURCE' and len(f9a['held_breaches']['undisposed']) == 1 and f9a['held_breaches']['undisposed'][0].startswith('part 1 ')
                   and f9o['state'] == 'STOP_RESOURCE' and f9p['state'] == 'STOP_RESOURCE' and f9b['state'] == 'COMPLETE' and f9b['held_breaches']['undisposed'] == []
                   and r10a == 'HELD' and gone10 and st10a.get('state') == 'HELD' and r10a2 == 'HELD' and r10b == 'PARTS_DONE' and back10 and r10c == 'HELD' and gone10c
                   and [h['state'] for h in hist10] == ['STOP_RESOURCE', 'STOP_HARD_RSS'] and hist10[0]['id'] == id10)
        if not held_ok:
            bad.append(f"held breaches: a stale finalizer, a breached record or a resource stop not held until a disposition of ITS identity and plan "
                       f"({stale_fin}, {f9a['state']}, old form {f9o['state']}, other plan {f9p['state']}, {f9b['state']}, {r10a}, {gone10}, {st10a.get('state')}, {r10a2}, "
                       f"{r10b}, {back10}, new stop {r10c}, {gone10c}, history {[h['state'] for h in hist10]})")
        made, S0 = [], Sampler                                                              # the watchdog spans loading, the run and finalize

        class Rec(S0):
            def __init__(self, status_path, hard_mib, period=.02, exit=None):
                made.append(Path(status_path).name); super().__init__(status_path, hard_mib, period, exit)
        try:
            Sampler = Rec
            with contextlib.redirect_stdout(q):
                census(day=d0, per_side=1, out=tmp / 'WD', part_size=1, causal=False, **{k: v for k, v in kw.items() if k not in ('causal_feat', 'causal_perturb')})
        finally:
            Sampler = S0
        if not {'plan_status.json', 'run_status.json', 'finalize_status.json'} <= set(made):
            bad.append(f'watchdog: not alive over every stage (loading, run, finalize): {sorted(set(made))}')
        c7 = copy_of('X7'); (tmp / 'X7' / f'census_{d0}.json').unlink(missing_ok=True); pf7 = c7 / 'parts' / 'part_0000.parquet'
        pf7.write_bytes(pf7.read_bytes() + b'x')                                            # corrupted between run and finalize: no receipt
        ok7 = refused(lambda: finalize(c7), 'hash differs') and not (tmp / 'X7' / f'census_{d0}.json').exists() and not (tmp / 'X7' / f'support_{d0}.json').exists()
        if not ok7:
            bad.append('finalize: a corrupt part was not refused before any receipt was published')
        lim_t = dict(LIMITS, partition_seconds=0.)                                          # the stop rule: time
        with contextlib.redirect_stdout(q):
            Rt = census(day=d0, per_side=2, out=tmp / 'T', part_size=1, limits=lim_t, **kw)
        nt = len(list((Rt[0] / 'parts').glob('part_*.json'))); st_t = json.loads((Rt[0] / 'run_status.json').read_text())
        if not (Rt[2] == dict(state='STOP_RESOURCE') and nt == 1 and st_t['latch']['seconds'] is True and not (tmp / 'T' / f'census_{d0}.json').exists()):
            bad.append(f'stop rule: a partition over the time limit did not stop the run at its checkpoint ({Rt[2]}, {nt} parts)')
        r_now, h_now = rss(); lim_m = dict(LIMITS, peak_rss_mib=h_now + 64.)                 # a peak that recedes
        try:
            SPIKE_MIB = h_now - r_now + 160.
            with contextlib.redirect_stdout(q):
                Rm = census(day=d0, per_side=2, out=tmp / 'M', part_size=1, limits=lim_m, **kw)
        finally:
            SPIKE_MIB = 0.
        nm = len(list((Rm[0] / 'parts').glob('part_*.json'))); st_m = json.loads((Rm[0] / 'run_status.json').read_text())
        if not (Rm[2] == dict(state='STOP_RESOURCE') and nm == 1 and st_m['latch']['rss'] is True and rss()[0] < lim_m['peak_rss_mib']):
            bad.append(f"stop rule: a receding peak above the limit was not latched ({Rm[2]}, {nm} parts, RSS now {rss()[0]:.0f} vs {lim_m['peak_rss_mib']:.0f})")
        with contextlib.redirect_stdout(q):                                                 # loading above the limit: no part starts
            Rl = census(day=d0, per_side=2, out=tmp / 'L', part_size=1, limits=dict(LIMITS, peak_rss_mib=1.), **kw)
        if not (Rl[2] == dict(state='STOP_RESOURCE') and not list((Rl[0] / 'parts').glob('part_*.json'))):
            bad.append(f'stop rule: a high-water RSS above the limit before the first part did not stop the run ({Rl[2]})')
        wd = subprocess.run([sys.executable, str(Path(__file__).resolve()), 'watchdog', str(tmp / 'wd.json')], capture_output=True, text=True, timeout=120)
        wds = json.loads((tmp / 'wd.json').read_text()) if (tmp / 'wd.json').exists() else {}   # end to end: the real exit path of the file on disk
        if not (wd.returncode == 3 and wds.get('state') == 'STOP_HARD_RSS'):
            bad.append(f'watchdog: a hard RSS breach did not end the process with STOP_HARD_RSS ({wd.returncode}, {wds})')
        def sentinel(*a, **k):
            raise AssertionError('an all-part read during the census')
        m0, r0 = R2.materialize, R2.read_parts
        try:
            R2.materialize = R2.read_parts = sentinel
            with contextlib.redirect_stdout(q):
                Sx = census(day=d0, per_side=2, out=tmp / 'S', part_size=1, **kw)
            sent_ok = Sx[2]['state'] == 'COMPLETE'
        except AssertionError:
            sent_ok = False
        finally:
            R2.materialize, R2.read_parts = m0, r0
        if not (sent_ok and len(R2.materialize(Sx[0])) == Sx[2]['census_rows'] and refused(lambda: R2.materialize(Sx[0], max_rows=10), 'above the bound')):
            bad.append('materialize: the census read all parts, or the explicit reader is unbounded')
        kw_nc = {k: v for k, v in kw.items() if k not in ('causal_feat', 'causal_perturb')}
        try:                                                                                # a failed part: a failure record, never COMPLETE, never rerun
            R2.LABEL_NUDGE = 1.
            with contextlib.redirect_stdout(q):
                Fa = census(day=d0, per_side=2, out=tmp / 'F', part_size=1, causal=False, **kw_nc)
            fr0 = [json.loads(p.read_text()) for p in sorted((Fa[0] / 'parts').glob('part_*.json'))]
            with contextlib.redirect_stdout(q):
                st_f = run(Fa[0], src=src, inputs=X.syn_inputs, feat_fn=X.syn_features())
            fr1 = [json.loads(p.read_text()) for p in sorted((Fa[0] / 'parts').glob('part_*.json'))]
            f_ok = (Fa[2]['state'] == 'FAILED' and any(r['status'] == 'FAILED' for r in fr0) and st_f == 'PARTS_DONE' and [r['asof'] for r in fr0] == [r['asof'] for r in fr1])
            f_why = Fa[2]['state']
        except ValueError as e:                                                             # a refusal here is the check failing, not a pass
            f_ok, f_why = False, str(e)[:120]
        finally:
            R2.LABEL_NUDGE = 0.
        if not f_ok:
            bad.append(f'failure records: a failed part was COMPLETE, rerun or refused ({f_why})')
        with contextlib.redirect_stdout(q):                                                 # two shards of one plan, run in turn: each only its parts
            make_plan(tmp / 'SH' / f'census_{d0}', R2, d0, 2, 'first_per_side', 1, tmp / 'kp', src, None, X.syn_inputs, True, None, False, None, '')
            st0 = run(tmp / 'SH' / f'census_{d0}', src=src, inputs=X.syn_inputs, feat_fn=X.syn_features(), causal_feat=X.syn_features(),
                      causal_perturb=(X.syn_perturb, X.syn_fire), shard=(0, 2))
            after0 = sorted(p.name for p in (tmp / 'SH' / f'census_{d0}' / 'parts').glob('part_*.json'))
            st1 = run(tmp / 'SH' / f'census_{d0}', src=src, inputs=X.syn_inputs, feat_fn=X.syn_features(), causal_feat=X.syn_features(),
                      causal_perturb=(X.syn_perturb, X.syn_fire), shard=(1, 2))
            Sh = finalize(tmp / 'SH' / f'census_{d0}')
        if not (st0 == st1 == 'PARTS_DONE' and after0 == ['part_0000.json', 'part_0002.json'] and parts(Sh[0]) == parts(A[0]) and strip(Sh[2]) == strip(A[2])
                and Sh[1] == A[1] and (Sh[0] / 'run_status_0of2.json').exists() and (Sh[0] / 'run_status_1of2.json').exists()):
            bad.append(f'shards: two shards did not process only their parts, or their union differs from the unsharded run ({st0}, {st1}, {after0})')
        with contextlib.redirect_stdout(q):                                                 # the audit sample does not depend on the partitioning
            One = census(day=d0, per_side=2, out=tmp / 'O', part_size=999, **kw)
        ka = sorted(x['key'] for p in sorted((A[0] / 'parts').glob('part_*.json')) for x in json.loads(p.read_text())['causal']['audit'])
        ko = sorted(x['key'] for p in sorted((One[0] / 'parts').glob('part_*.json')) for x in json.loads(p.read_text())['causal']['audit'])
        if not (ka and ka == ko and strip(One[2])['causality'] == strip(A[2])['causality']):
            bad.append('audit sample: the selected keys depend on the partitioning')
        try:                                                                                # an unbuildable future world: never PASS, never COMPLETE
            R2.VENUE_NO_PATH = 1
            with contextlib.redirect_stdout(q):
                Ne = census(day=d0, per_side=2, out=tmp / 'N', part_size=1, causal_every=1, **kw)
            n_ok = Ne[2]['causality']['verdict'] == 'NOT-EVALUABLE' and Ne[2]['state'] == 'FAILED' and Ne[2]['causality']['components']['not_evaluable'] > 0
            n_why = (Ne[2]['causality']['verdict'], Ne[2]['state'])
        except ValueError as e:                                                             # a refusal here is the check failing
            n_ok, n_why = False, str(e)[:120]
        finally:
            R2.VENUE_NO_PATH = 0
        if not n_ok:
            bad.append(f'verdicts: an unbuildable future world was not NOT-EVALUABLE / FAILED ({n_why})')
        try:                                                                                # a short write is refused before promotion
            WRITE_DROP = True
            ok_w = refused(lambda: census(day=d0, per_side=2, out=tmp / 'V', part_size=1, **kw), 'the written part has')
        finally:
            WRITE_DROP = False
        if not (ok_w and not list((tmp / 'V' / f'census_{d0}' / 'parts').glob('part_*.json'))):
            bad.append('promotion: a part whose written rows differ from the frame was promoted')
    if bad:
        sys.exit(f'{MARK} SELFTEST FAILED: ' + ' | '.join(bad))
    print(f"r2 runner selftests passed (synthetic only): uninterrupted = interrupted + resumed = interrupted write + resumed ({A[2]['census_rows']} rows, "
          f"{A[2]['causality']['checked']} audited checks); refused resumes {sorted(rr)}; finalize refuses without publishing; the stop rule (time, a "
          f'receding peak, loading) and the watchdog; no all-part read (a sentinel); failure records kept, never rerun; the audit sample independent '
          f'of the partitioning; a short write refused', flush=True)


if __name__ == '__main__':
    a = sys.argv[1:]
    if a == ['selftest']:
        selftest()
    elif len(a) == 2 and a[0] == 'watchdog':                                                # the watchdog's firing control (a child process)
        m_ = Sampler(a[1], 1.); m_.start(); m_.join(5); sys.exit(0)
    elif len(a) in (3, 4) and a[0] == 'plan':
        pop = 'all' if (len(a) == 4 and a[3] == 'all') else 'first_per_side'; ps = int(a[3]) if (len(a) == 4 and a[3] != 'all') else R2.PER_SIDE
        make_plan(Path(R2.OUT) / f'census_{a[1]}{a[2]}', R2, a[1], ps, pop, R2.PART_SIZE, None, None, None, None, True, None, False, None, a[2])
    elif len(a) in (3, 4) and a[0] == 'run':
        sh_ = tuple(int(x) for x in a[3].split('/')) if len(a) == 4 else None
        print(run(Path(R2.OUT) / f'census_{a[1]}{a[2]}', shard=sh_), flush=True)
    elif len(a) == 3 and a[0] == 'finalize':
        finalize(Path(R2.OUT) / f'census_{a[1]}{a[2]}')
    else:
        sys.exit('usage: pf_r2_runner.py selftest | plan <day> <tag> [all|<per_side>] | run <day> <tag> [k/n] | finalize <day> <tag>')
