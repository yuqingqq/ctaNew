"""MUTATION CHECK of the pi0-50 teacher's selftest (2026-11-20, session aea5bfb9): a checker that cannot fail is not a check. Each planted error
is a textual replacement in a copy of pf_r2_teacher.py (the file on disk is never touched). The copy is executed as the module under its real
path, in a fresh process, and its selftest (pf_r2_teacher_selftest, synthetic only) is run.
Required:
 - the unchanged copy passes;
 - every mutant is caught BY A CHECK: the selftest's own failure message; a crash is not a catch;
 - every replacement target occurs exactly once.
PF_TEACHER_SYN (the cached synthetic world) is passed through. Usage: pf_r2_teacher_mutants.py"""
import os
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE / 'pf_r2_teacher.py'
FAIL = 'R2 TEACHER SELFTEST FAILED'
MUTANTS = [
    ('weights dropped', "weight=norm_weights(w), feature_name=FEATS, free_raw_data=True)", "weight=None, feature_name=FEATS, free_raw_data=True)"),
    ('no normalization constant', "    return w * (len(w) / w.sum())", "    return w"),
    ('q on all rows', "q=(rf, qf, wf)", "q=(ra, qa, wa)"),
    ('qm on all rows', "qm=(rf, lab(d, 'qm', 'fil', nf), wf)", "qm=(ra, lab(d, 'qm', 'all', na), wa)"),
    ('p label not the fill indicator', "p=(ra, (qa > FILLED).astype(float), wa)", "p=(ra, (qa >= 0).astype(float), wa)"),
    ('severe on the filled rows', "severe=(ra, lab(d, 'severe', 'all', na), wa)", "severe=(rf, lab(d, 'severe', 'fil', nf), wf)"),
    ('a tie POSTs', "('POST' if V > 0 else 'WAIT')", "('POST' if V >= 0 else 'WAIT')"),
    ('a tie PULLs', "('PULL' if V < 0 else 'KEEP')", "('PULL' if V <= 0 else 'KEEP')"),
    ('support ignored', "        if ns is not None:\n            return dict(rec, status=ns", "        if False:\n            return dict(rec, status=ns"),
    ('coherence ignored', "if not (bool(pr['ok'][0]) and np.isfinite(V)):", "if not np.isfinite(V):"),
    ('cap ignored', "if st == 'absent' and v['room'] < -1e-9:", "if False:"),
    ('near END ignored', "if st == 'absent' and g + KP.PLACE >= B.end:", "if False:"),
    ('range without tolerance', "if not (lo - TOL <= x <= hi + TOL):", "if not (lo <= x <= hi):"),
    ('range ignored', "            if not (lo - TOL <= x <= hi + TOL):\n                return 'outside_training_range'",
     "            if False:\n                return 'outside_training_range'"),
    ('categories ignored', "if any(str(cats[d][0]) not in self.present.get(sh, {}).get(d, set()) for d in R2.DIMS):", "if False:"),
    ('inventory sign without tolerance', "c_inv=np.select([inv > TOL, inv < -TOL]", "c_inv=np.select([inv > 0, inv < 0]"),
    ('horizon outside ignored', "        if hb == 'nan':\n            return 'unsupported_horizon'", "        if False:\n            return 'unsupported_horizon'"),
    ('cell support ignored', "if self.cells.get(f'{sh} {hb}') != 'SUPPORTED':", "if False:"),
    ('horizon label from astype(str)', "return lab_[np.where(np.asarray(c.codes) >= 0, np.asarray(c.codes), len(lab_) - 1)]", "return np.asarray(c.astype(str))"),
    ('extract column order', "Xa = np.ascontiguousarray(G[FEATS].to_numpy(np.float32))", "Xa = np.ascontiguousarray(G[FEATS[::-1]].to_numpy(np.float32))"),
    ('extract without the status filter', "D = D[(D.status == 'ok') & D.sampled.astype(bool)]", "D = D[D.sampled.astype(bool)]"),
    ('extract without the sample filter', "D = D[(D.status == 'ok') & D.sampled.astype(bool)]", "D = D[(D.status == 'ok')]"),
    ('a batch shifted by one row', "np.frombuffer(os.pread(self.fd, (b - a) * self.rb, a * self.rb), np.float32)",
     "np.frombuffer(os.pread(self.fd, (b - a) * self.rb, max(a - 1, 0) * self.rb), np.float32)"),
    ('a single row shifted', "np.frombuffer(os.pread(self.fd, self.rb, i * self.rb), np.float32).astype(np.float64)",
     "np.frombuffer(os.pread(self.fd, self.rb, min(i + 1, self.n - 1) * self.rb), np.float32).astype(np.float64)"),
    ('a changed model accepted', "C.require(C.sha(p_) == freeze['models'][f'{s}_{h}_{head}'], f'model {p_.name} differs from the freeze')", "pass"),
    ('a changed support accepted', "C.require(C.sha(cfg['out'] / 'train' / 'manifest.json') == fz['train_manifest_sha256'] and C.sha(cfg['support']) == fz['support_sha256'],",
     "C.require(C.sha(cfg['out'] / 'train' / 'manifest.json') == fz['train_manifest_sha256'],"),
    ('a changed source accepted', "        C.require(C.sha(HERE / n_) == h_, f'source {n_} changed since the freeze')", "        pass"),
    ('a changed record accepted', "C.require(rp.name == f'part_{i:04d}.json' and C.sha(rp) == h, f'record {i} differs from the committed manifest')",
     "C.require(rp.name == f'part_{i:04d}.json', f'record {i} differs from the committed manifest')"),
    ('a changed parquet accepted', "C.require(rec['plan_sha256'] == cfg['plan_sha'] and rec['part'] == i and C.sha(f) == rec['sha256'], f'part {i}: record or parquet differs')",
     "C.require(rec['plan_sha256'] == cfg['plan_sha'] and rec['part'] == i, f'part {i}: record or parquet differs')"),
    ('a wrong sampled-ok count accepted', "C.require(len(D) == int(rec['sampled_ok']), f\"part {rec['part']}: {len(D)} sampled ok rows, the record says {rec['sampled_ok']}\")", "pass"),
    ('a sampled ok row outside the groups accepted', "            C.require(sum(row) == len(D), f\"part {rec['part']}: {len(D) - sum(row)} sampled ok rows in no side / head group\")\n", ""),
    ('extraction rewritten', "    C.require(not fin.exists(), 'the training files exist (immutable: never rewritten)')", "    import shutil; shutil.rmtree(fin, ignore_errors=True)"),
    ('the probe never compares', "ok_ = same_decision(dec, d2)", "ok_ = True"),
    ('decisions compared without inputs', "    return ra is None or all(np.float32(ra[c]).tobytes() == np.float32(rb[c]).tobytes() for c in FEATS) and ra['price'] == rb['price']",
     "    return True"),
    ('the delivered control never counted', "ch_ = not same_decision(dec, d3)", "ch_ = False"),
    ('the market one grid step ahead', "mk = w['market'][s].get(kk) if abs(tg[kk] - d) <= 1e-6 else None",
     "mk = w['market'][s].get(kk + 1, w['market'][s].get(kk)) if abs(tg[kk] - d) <= 1e-6 else None"),
    ('the other side\'s original', "o = w['o'][s]; tg = np.asarray(w['W'][s]['tg'], float)", "o = w['o'][1 - s]; tg = np.asarray(w['W'][s]['tg'], float)"),
    ('the other side\'s price', "price = (v['bid'] if s == 0 else v['ask']) if gid is None else B.E.o[gid]['price']        # rebuild_row",
     "price = (v['ask'] if s == 0 else v['bid']) if gid is None else B.E.o[gid]['price']        # rebuild_row"),
    ('decisions not logged', "dec = self.decide(v, self.B); self.log.append(dec)", "dec = self.decide(v, self.B)"),
    ('ledger cash unchecked', "ok = abs(cash - L['cash']) <= 1e-6 and abs(pos - L['position']) <= 1e-9", "ok = abs(pos - L['position']) <= 1e-9"),
    ('ledger generation size unchecked', "and abs(val - L['value']) <= 1e-6 and within and logged", "and abs(val - L['value']) <= 1e-6 and logged"),
    ('logging unchecked', " and logged == len(asked) - fb", ""),
    ('reconciliation always equal', "    return dict(categories_equal=got_c == want_c, cells_equal=got_l == want_l,", "    return dict(categories_equal=True, cells_equal=True,"),
    ('the fit loses its weights constant on q / qm', "    wa, wf = lab(d, 'weight', 'all', na), lab(d, 'weight', 'fil', nf)",
     "    wa, wf = lab(d, 'weight', 'all', na), np.ones(nf)"),
]
RUN = '''import sys, types
sys.path.insert(0, {here!r})
mod = types.ModuleType('pf_r2_teacher'); mod.__file__ = {src!r}; sys.modules['pf_r2_teacher'] = mod
exec(compile(open({mut!r}).read(), {src!r}, 'exec'), mod.__dict__)
mod.selftest()
'''


def run_copy(text, tmp):
    p = Path(tmp) / 'mutant.py'; p.write_text(text)
    r = subprocess.run([sys.executable, '-c', RUN.format(here=str(HERE), src=str(SRC), mut=str(p))], capture_output=True, text=True, env=dict(os.environ))
    return r.returncode, r.stdout + r.stderr


base = SRC.read_text()
bad = [n for n, old, _ in MUTANTS if base.count(old) != 1]
if bad:
    sys.exit(f'r2 teacher mutation check REFUSED: targets not found exactly once: {bad}')
with tempfile.TemporaryDirectory() as tmp:
    rc, out = run_copy(base, tmp)
    pre = rc == 0 and 'r2 teacher selftests passed' in out
    print(f'unchanged copy passes its selftest: {pre}', flush=True)
    if not pre:
        sys.exit(f'r2 teacher mutation check FAILED: the unchanged copy does not pass ({out[-400:]})')
    caught = []
    for name, old, new in MUTANTS:
        rc, out = run_copy(base.replace(old, new), tmp)
        hit = rc != 0 and FAIL in out
        why = out[out.index(FAIL):][:160].replace('\n', ' ') if hit else ('PASSED (not caught)' if rc == 0 else 'CRASHED: ' + out.strip().splitlines()[-1][:160])
        caught.append(hit); print(f'  {name}: {"caught" if hit else "NOT CAUGHT"} - {why}', flush=True)
ok = all(caught)
print(f'r2 teacher mutation check {"passed" if ok else "FAILED"}: the unchanged copy passes its selftest, {sum(caught)} / {len(MUTANTS)} planted code errors '
      f'are caught by a check')
sys.exit(0 if ok else 1)
