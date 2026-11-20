"""MUTATION CHECK of the R2-D label adapter's selftest (review 8cc1dc49's next priority; 2026-11-20, session aea5bfb9). Each planted error is a
textual replacement in a copy of pf_r2_labels.py (the file on disk is never touched). The copy is executed as the module under its real path,
in a fresh process, and its selftest (pf_r2_labels_selftest, synthetic only) is run.
Required:
 - the unchanged copy passes;
 - every mutant is caught BY A CHECK: the selftest's own failure message; a crash is not a catch;
 - every target occurs exactly once.
Three candidates are equivalent mutants and are not listed, because no valid run can distinguish them:
 - the deep copy of the fork state before its first view: pre_tick already evaluated both sides' status() at the fork tick, so that view
   cannot mutate the saved fork. The 8cc1dc49-era first run listed it and it passed; it is kept in that run's log;
 - the fork re-check after the arms: it can only fire when the fork copy itself is broken;
 - prefork_identical: it holds in every valid run.
Usage: pf_r2_labels_mutants.py [k/n] (shard k of n: mutants i with i mod n = k; the unchanged control runs in every shard)."""
import os
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE / 'pf_r2_labels.py'
FAIL = 'R2 LABELS SELFTEST FAILED'
MUTANTS = [
    ('the arm teacher bound to the fork copy', "Bx = copy.deepcopy(Bv); pol = factory(Bx);", "Bx = copy.deepcopy(Bv); pol = factory(Bv);"),
    ('the remaining side skipped at the fork tick', "        if first == side:\n            BE.decide(Bx, g, 1 - side, pol)\n", ""),
    ('the forced action dropped', "BE.decide(Bx, g, side, pol, forced=a)", "BE.decide(Bx, g, side, pol)"),
    ('arms share one branch', "Bx = copy.deepcopy(Bv); pol = factory(Bx);", "Bx = Bv; pol = factory(Bx);"),
    ('Y reversed', "Y=arms[na]['value'] - arms[nb]['value'], common_prefork_identical=common", "Y=arms[nb]['value'] - arms[na]['value'], common_prefork_identical=common"),
    ('the binding check disabled', "self.bind_bad += int(bv != v)", "self.bind_bad += 0"),
    ('reconcile without cash from the fills', "abs(cash - L['cash']) <= 1e-6 and ", ""),
    ('reconcile without position from the fills', "abs(pos - L['position']) <= 1e-9 and ", ""),
    ('reconcile without the value identity', " and abs(L['cash'] + L['position'] * B.mark - L['value']) <= 1e-6\n", "\n"),
    ('reconcile without the size guard', "\n                and all(q_ <= B.E.o[g_]['size'] + 1e-9 for g_, q_ in gen.items()))", ")"),
    ('the manifest fingerprint unchecked', "C.require(ctx['fingerprint'] == r['fork_fingerprint'], f\"pair {rkey(r)}: the rebuilt fork differs from the manifest's\")", "pass"),
    ('selection not round robin', "g = g.sort_values(['_rank', '_src'], kind='stable').head(cap)", "g = g.sort_values(['_src', '_rank'], kind='stable').head(cap)"),
    ('selection without the cap', ".head(cap)", ".head(10 ** 9)"),
    ('selection of non-ok candidates', "D = pd.DataFrame([r for r in rows if r['status'] == 'ok' and r['kind'] in KINDS])", "D = pd.DataFrame([r for r in rows if r['kind'] in KINDS])"),
    ('resume reruns finished pairs', "    rec = json.loads(p.read_text()); return rec.get('manifest_sha256') == msha and rec.get('key') == rkey(r)", "    return False"),
    ('held stops ignored', "        if held['undisposed']:\n            RN.atomic_json(out / 'label_status.json', dict(state='HELD'",
     "        if False:\n            RN.atomic_json(out / 'label_status.json', dict(state='HELD'"),
    ('libraries not enforced', "    C.require(now == fz['libraries'], f\"library versions {now} differ from the freeze's {fz['libraries']}\")", "    pass"),
    ('the freeze sha not pinned', "C.require(C.sha(cfg['out'] / 'freeze.json') == freeze_sha, 'not the frozen pi0_50_v1 (freeze sha)'); ", ""),
    ('the census fingerprint without the focal identity', "fingerprint=focal_fingerprint(B, day, o) if stt == 'ok' else None", "fingerprint=BE.fingerprint(B) if stt == 'ok' else None"),
    ('bound_fork without the other side first', "    if (episode + k) % 2 != side:\n        BE.decide(B0, g, 1 - side, pol)\n    return ctx_of(B0, k, side, episode)",
     "    return ctx_of(B0, k, side, episode)"),
    ('the manifest rewritten', "C.require(not mp.exists() and not cp.exists(), 'the manifest exists (immutable)')", "pass"),
    ('the candidates gzip not deterministic', "gz = gzip.GzipFile(filename='', mode='wb', fileobj=raw, mtime=0)", "gz = gzip.GzipFile(filename='', mode='wb', fileobj=raw, mtime=None); import time as _t; _t.sleep(1.1)"),
    ('only ok candidates streamed', "                gz.write((json.dumps(clean(r), sort_keys=True, default=str) + '\\n').encode()); counts[",
     "                (gz.write((json.dumps(clean(r), sort_keys=True, default=str) + '\\n').encode()) if r['status'] == 'ok' else None); counts["),
    ('the candidates hash unchecked', "    C.require(C.sha(cp) == man['candidates_sha256'], 'candidates.jsonl.gz differs from the manifest')", "    pass"),
    ('the pi0 sampling ignored', "smp = (k == 0 or k % 5 == ph[(s, 'post')]) if kind == 'post' else (k % 5 == ph[(s, 'retain')])", "smp = True"),
    ('offset 0 named delayed', "source = ('neutral' if off == 0 else 'delayed')", "source = ('delayed' if off == 0 else 'neutral')"),
    ('outcome fields kept', "if k_ not in OUTCOMES and not k_.startswith('ctx_')", "if not k_.startswith('ctx_')"),
    ('the arm probes not recorded', "rec = dict(manifest_sha256=msha, key=rkey(r), candidate=r, result=res, causal=cz,", "rec = dict(manifest_sha256=msha, key=rkey(r), candidate=r, result=res, causal=[],"),
    ('no arm causal probe', "probe, st = T.causal_probe(every); t_ = ArmTeacher(M, sup, causal=probe)", "probe, st = T.causal_probe(every); t_ = ArmTeacher(M, sup, causal=None)"),
]
RUN = '''import sys, types
sys.path.insert(0, {here!r})
mod = types.ModuleType('pf_r2_labels'); mod.__file__ = {src!r}; sys.modules['pf_r2_labels'] = mod
exec(compile(open({mut!r}).read(), {src!r}, 'exec'), mod.__dict__)
import pf_r2_labels_selftest as ST
ST.run(mod)
'''


def run_copy(text, tmp):
    p = Path(tmp) / 'mutant.py'; p.write_text(text)
    r = subprocess.run([sys.executable, '-c', RUN.format(here=str(HERE), src=str(SRC), mut=str(p))], capture_output=True, text=True, env=dict(os.environ))
    return r.returncode, r.stdout + r.stderr


k_, n_ = (int(x) for x in sys.argv[1].split('/')) if len(sys.argv) > 1 else (0, 1)
base = SRC.read_text()
bad = [n for n, old, _ in MUTANTS if base.count(old) != 1]
if bad:
    sys.exit(f'r2 labels mutation check REFUSED: targets not found exactly once: {bad}')
mine = [(i, m) for i, m in enumerate(MUTANTS) if i % n_ == k_]
with tempfile.TemporaryDirectory() as tmp:
    rc, out = run_copy(base, tmp)
    pre = rc == 0 and 'r2 labels selftests passed' in out
    print(f'shard {k_}/{n_}: unchanged copy passes its selftest: {pre}', flush=True)
    if not pre:
        sys.exit(f'r2 labels mutation check FAILED: the unchanged copy does not pass ({out[-400:]})')
    caught = []
    for i, (name, old, new) in mine:
        rc, out = run_copy(base.replace(old, new), tmp)
        hit = rc != 0 and FAIL in out
        why = out[out.index(FAIL):][:170].replace('\n', ' ') if hit else ('PASSED (not caught)' if rc == 0 else 'CRASHED: ' + out.strip().splitlines()[-1][:170])
        caught.append(hit); print(f'  {i} {name}: {"caught" if hit else "NOT CAUGHT"} - {why}', flush=True)
ok = all(caught)
print(f'r2 labels mutation check (shard {k_}/{n_}) {"passed" if ok else "FAILED"}: the unchanged copy passes its selftest, {sum(caught)} / {len(mine)} planted '
      f'code errors are caught by a check')
sys.exit(0 if ok else 1)
