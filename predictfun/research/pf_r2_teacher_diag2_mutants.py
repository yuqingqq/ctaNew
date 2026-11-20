"""MUTATION CHECK of pf_r2_teacher_diag2's selftest (review 8cc1dc49 P3). Each planted error must be caught by the selftest's own failure
message (a crash is not a catch); the unchanged copy must pass; every target occurs exactly once. Usage: pf_r2_teacher_diag2_mutants.py"""
import subprocess
import sys
import tempfile
from pathlib import Path

SRC = Path(__file__).resolve().parent / 'pf_r2_teacher_diag2.py'
MUT = [('v1 mixed denominator (coherent sums / all weight)', "V_pred_coh_mean=float((w[ok] * vp[ok]).sum() / Wc) if Wc > 0 else np.nan",
        "V_pred_coh_mean=float((w[ok] * vp[ok]).sum() / W) if Wc > 0 else np.nan"),
       ('the realized coherent mean over all weight', "V_real_coh_mean=float((w[ok] * vr[ok]).sum() / Wc) if Wc > 0 else np.nan",
        "V_real_coh_mean=float((w[ok] * vr[ok]).sum() / W) if Wc > 0 else np.nan"),
       ('the all-row mean over coherent rows only', "V_real_all_mean=float((w * vr).sum() / W) if W > 0 else np.nan",
        "V_real_all_mean=float((w[ok] * vr[ok]).sum() / W) if W > 0 else np.nan"),
       ('incoherent rows counted coherent', "W, Wc = float(w.sum()), float(w[ok].sum())", "W, Wc = float(w.sum()), float(w.sum()); ok = np.ones_like(ok)")]
base = SRC.read_text(); assert all(base.count(o) == 1 for _, o, _ in MUT), 'a target not found exactly once'
caught = []
with tempfile.TemporaryDirectory() as t:
    for name, old, new in [('none (must pass)', None, None)] + MUT:
        p = Path(t) / 'm.py'; p.write_text(base if old is None else base.replace(old, new))
        r = subprocess.run([sys.executable, '-c', f"import sys; sys.argv=['x','selftest']; sys.path.insert(0, {str(SRC.parent)!r}); __file__={str(SRC)!r}; exec(compile(open({str(p)!r}).read(), {str(SRC)!r}, 'exec'))"],
                           capture_output=True, text=True)
        hit = r.returncode != 0 and 'R2 TEACHER DIAG2 SELFTEST FAILED' in (r.stdout + r.stderr)
        if old is None:
            pre = r.returncode == 0; print(f'unchanged copy passes: {pre}')
        else:
            caught.append(hit); print(f"  {name}: {'caught' if hit else 'NOT CAUGHT'}")
ok = pre and all(caught)
print(f"diag2 mutation check {'passed' if ok else 'FAILED'}: {sum(caught)} / {len(MUT)} planted errors caught by a check"); sys.exit(0 if ok else 1)
