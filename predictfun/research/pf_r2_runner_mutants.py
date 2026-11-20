"""MUTATION CHECK of the R2-A full-day runner's selftest (review 4cd7b45f's launch gate; 2026-11-20, session aea5bfb9): a checker that cannot
fail proves nothing. Each entry plants one code error in a COPY of pf_r2_runner.py (text replacement, executed in a subprocess; the repo file is
never touched) and runs the copy's selftest (synthetic tapes only). Required: the unchanged copy passes, every mutant is caught BY A CHECK (the
selftest's own failure message; a crash is unexpected), and every replacement target occurs exactly as often as declared. No market data is read.
Usage (run_light, PF_SLOT=claude): pf_r2_runner_mutants.py [index | targets]
"""
import subprocess
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent / 'pf_r2_runner.py'
# (name, text in the source, replacement[, occurrences]) - entry 0 is the unchanged copy
MUT = [
    ('none (must pass)', None, None),
    # --- resume acceptance and the refused resumes
    ('a record of another plan accepted', "C.require(rec.get('plan_sha256') == psha, f'part {i}: a completion record of another plan')", "pass"),
    ('the part file\'s hash not checked', "f.exists() and C.sha(f) == rec.get('sha256')", "f.exists()"),
    ('the record\'s members not checked', "rec.get('part') == i and rec.get('members') == plan['partitions'][i]['members']", "rec.get('part') == i"),
    ('a stale source not refused', "C.require(sources(mod) == plan['sources'], 'stale source: a module or a dependency changed since the plan')", "pass"),
    ('changed input hours / hashes not refused', "C.require(list(hrs) == plan['inputs']['hours'] and canon(tracked) == canon(plan['inputs']['tracked']),",
     "C.require(True,"),
    ('a different plan overwritten', "C.require(canon(json.loads(pp.read_text())) == canon(plan), 'a different plan exists for this directory (a plan is immutable)')",
     "atomic_json(pp, plan)"),
    ('the plan binds the output location', "    plan = dict(version=PLAN_VERSION, day=day, manifest=mod.MANIFEST,",
     "    plan = dict(version=PLAN_VERSION, day=day, tag=d.name, manifest=mod.MANIFEST,"),
    ('a shard runs every part', "mine = [i for i in range(len(plan['partitions'])) if shard is None or i % int(shard[1]) == int(shard[0])]",
     "mine = [i for i in range(len(plan['partitions']))]"),
    ('a shard runs the other parts', "mine = [i for i in range(len(plan['partitions'])) if shard is None or i % int(shard[1]) == int(shard[0])]",
     "mine = [i for i in range(len(plan['partitions'])) if shard is None or i % int(shard[1]) != int(shard[0])]"),
    ('finalize without the source preflight (e6b0efa4)', "    C.require(sources(mod) == plan['sources'], 'stale finalizer: the finalizing sources differ from the plan (review e6b0efa4 finding 2)')",
     "    pass"),
    ('finalize ignores the held breaches (e6b0efa4)', "state = ('STOP_RESOURCE' if (hw > lim['peak_rss_mib'] or held['undisposed']) else", "state = ('STOP_RESOURCE' if (hw > lim['peak_rss_mib']) else"),
    ('a run through an earlier resource stop (e6b0efa4)', "    if held['undisposed'] and pending:", "    if False:"),
    ('a disposition never accepted (e6b0efa4)', "ok_ids = {e['id'] for e in disp.get('entries', []) if e.get('id') and str(e.get('reason', '')).strip()} if disp.get('plan_sha256') == psha else set()",
     "ok_ids = set()"),
    ('a disposition of another plan accepted (8cc1dc49)', "if e.get('id') and str(e.get('reason', '')).strip()} if disp.get('plan_sha256') == psha else set()",
     "if e.get('id') and str(e.get('reason', '')).strip()} if True else set()"),
    ('a stop matched by its file name (8cc1dc49)', "for e in hist if e['id'] not in ok_ids]", "for e in hist if e['file'] not in {h['file'] for h in hist if h['id'] in ok_ids}]"),
    ('no stop history (8cc1dc49)', "            with open(hp, 'a') as f:\n                f.write(json.dumps(ev, sort_keys=True) + '\\n')", "            pass"),
    ('a held run recorded as a resource stop (8cc1dc49)', "**held); state = 'HELD'", "**held); state = 'STOP_RESOURCE'"),
    ('no watchdog during loading (e6b0efa4)', "    mon = Sampler(d / 'plan_status.json', lim0['guard_rss_mib'], exit=WATCHDOG_EXIT); mon.start()", "    mon = threading.Thread()"),
    ('a failed part rerun on resume', "    return rec\n\n\nSTOP_STATES = ", "    return rec if rec.get('status') == 'PASS' else None\n\n\nSTOP_STATES = "),
    # --- the stop rule and the watchdog
    ('the stop rule reads the current RSS', "el = time.time() - t0; hw = rss()[1]", "el = time.time() - t0; hw = rss()[0]"),
    ('no stop after a latched breach', "latch = dict(part=i, elapsed_s=el, hwm_mib=hw, **breach); state = 'STOP_RESOURCE'; break",
     "latch = dict(part=i, elapsed_s=el, hwm_mib=hw, **breach); state = 'STOP_RESOURCE'"),
    ('the high-water after loading not checked', "if latch is None and pending and hw0 > lim['peak_rss_mib']:", "if False:"),
    ('the watchdog never ends the process', "guard_rss_mib=self.hard, asof=C.stamp())); self.exit(3); return True", "guard_rss_mib=self.hard, asof=C.stamp())); return True"),
    ('the watchdog fires below its limit', "        if r > self.hard:\n            atomic_json(self.status_path,", "        if r > 0:\n            atomic_json(self.status_path,"),
    # --- promotion and finalization
    ('a part promoted without its row check', "C.require(pq.read_metadata(tmp).num_rows == n_rows == info['rows'],", "C.require(True,"),
    ('finalize reads every part', "S = mod.support_merge([r['support'] for r in recs]); add =", "_all = mod.materialize(d); S = mod.support_merge([r['support'] for r in recs]); add ="),
    ('finalize without accepting the records', "rec = accept(d, plan, psha, i); C.require(rec is not None, f'finalize: part {i} has no accepted completion record')",
     "rec = json.loads((d / 'parts' / f'part_{i:04d}.json').read_text())"),
    ('COMPLETE despite a failed part', "else 'COMPLETE' if (parts_ok and lverd == 'PASS' and cverd in ('PASS', 'NOT-RUN'))\n             else 'FAILED')", "else 'COMPLETE')"),
    ('the verdict ignores the not-evaluable checks', "else 'NOT-EVALUABLE' if comp['not_evaluable'] > 0 else 'PASS')", "else 'PASS')"),
]


def source(i):
    _, a, b, *n = MUT[i]; s = SRC.read_text()
    if a is None:
        return s
    k = s.count(a); want = n[0] if n else 1
    if k != want:
        sys.exit(f'mutation target occurs {k} times, declared {want}: {MUT[i][0]}')
    return s.replace(a, b)


def one(i):
    import types
    mod = types.ModuleType('pf_r2_runner_mutant'); mod.__file__ = str(SRC); sys.modules['pf_r2_runner_mutant'] = mod
    exec(compile(source(i), str(SRC), 'exec'), mod.__dict__); mod.selftest()


def main():
    bad = 0
    for i, (name, *_) in enumerate(MUT):
        r = subprocess.run([sys.executable, __file__, str(i)], capture_output=True, text=True)
        tail = r.stderr.strip().splitlines() or r.stdout.strip().splitlines() or ['']
        caught = r.returncode == 1 and 'R2 RUNNER SELFTEST FAILED' in r.stderr           # caught BY A CHECK (the selftest's own exit message); a crash is unexpected
        ok = (r.returncode == 0) if i == 0 else caught
        bad += not ok
        first = next((x for x in r.stderr.splitlines() if 'SELFTEST FAILED' in x), None)
        how = 'passed' if r.returncode == 0 else ('caught: ' + first.strip()[:220]) if caught and first else ' / '.join(x.strip()[:220] for x in tail[-2:])
        print(f'{i:2d} {name}: rc {r.returncode} -> {how}{"" if ok else "   <<< UNEXPECTED"}', flush=True)
    if bad:
        sys.exit(f'R2 RUNNER MUTATION CHECK FAILED: {bad} unexpected outcomes of {len(MUT)}')
    print(f'r2 runner mutation check passed: the unchanged copy passes its selftest, {len(MUT) - 1} / {len(MUT) - 1} planted code errors are caught by a check', flush=True)


if __name__ == '__main__':
    if sys.argv[1:] == ['targets']:
        [source(i) for i in range(len(MUT))]; print(f'all {len(MUT) - 1} mutation targets occur as declared')
    elif len(sys.argv) > 1:
        one(int(sys.argv[1]))
    else:
        main()
