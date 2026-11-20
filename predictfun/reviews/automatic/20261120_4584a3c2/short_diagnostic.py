"""Recompute only the predeclared <=100 ms bin from frozen models and saved training rows.
No fitting, new labels, tape or later day. Bounded at 200,000 selected rows total.
"""
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
sys.path.insert(0, '/tmp/pf-review-4584a3c2/predictfun/research')
import numpy as np
import pandas as pd
import pf_r2_teacher as T

data = Path('/home/yuqing/ctaNew/data/predict_fun/research/r2_teacher_20261120')
read = lambda p: json.loads(p.read_text())
man = read(data / 'train/manifest.json'); fz = read(data / 'freeze.json'); diag = read(data / 'diag/diag.json')
models = T.load_models(dict(T.REAL, out=data), fz)
out = []; ntotal = 0
for s, h in T.GROUPS:
    name = f'{s}_{h}'; n = man['groups'][name]['rows']; d = data / 'train' / name
    def field(c):
        return np.memmap(d / f'{c}_all.f64', dtype=np.float64, mode='r', shape=(n,))
    times = field('decision_ms'); ix = np.flatnonzero((5000 - times > 0) & (5000 - times <= 100)); ntotal += len(ix)
    assert ntotal <= 200000
    xx = np.memmap(d / 'X_all.f32', dtype=np.float32, mode='r', shape=(n, 107))[ix]
    X = pd.DataFrame(xx, columns=T.FEATS); X['price'] = field('price')[ix]
    q = field('q')[ix]; qm = field('qm')[ix]; weight = field('weight')[ix]; value = field('value')[ix]; severe = field('severe')[ix]
    pos = q > 1e-12; M = models[(s, h)]; P = M['p'].predict(xx, num_threads=1)
    qp = np.clip(M['q'].predict(xx, num_threads=1), 0, X.fp_remaining.to_numpy())
    mp = M['qm'].predict(xx, num_threads=1); sv = M['severe'].predict(xx, num_threads=1)
    V = (1 if s == 0 else -1) * (X.pf_mid.to_numpy().astype(float) - X.price.to_numpy()) * P * qp + P * mp
    mapped = T.predict_rows(M, X, s); assert np.allclose(V, mapped['V'], rtol=0, atol=1e-12)
    ok = mapped['ok']; avg = lambda z: float(np.average(z, weights=weight)); avf = lambda z: float(np.average(z[pos], weights=weight[pos]))
    got = dict(rows=len(ix), incoherent=int((~ok).sum()), pred_fill=avg(P), real_fill=avg(pos),
               cond_q_pred=avf(qp), cond_q_real=avf(q), cond_qm_pred=avf(mp), cond_qm_real=avf(qm),
               severe_pred=avg(sv), severe_real=avg(severe),
               V_pred_wmean_c=float(np.sum(weight[ok] * V[ok]) / weight.sum()),
               V_real_wmean_c=float(np.sum(weight[ok] * value[ok]) / weight.sum()))
    saved = next(a for a in diag['rows'] if a['side_head'] == T.sh_of(s, h) and a['horizon'] == '(0.0, 100.0]')
    assert all(np.isclose(v, saved[k], rtol=1e-10, atol=1e-9) for k, v in got.items())
    out.append(dict(group=name, reproduced=got, coherent_weight_fraction=float(weight[ok].sum()/weight.sum()),
                    corrected_coherent_V_pred_c=float(np.average(V[ok], weights=weight[ok])),
                    corrected_coherent_V_real_c=float(np.average(value[ok], weights=weight[ok])),
                    full_population_real_V_c=avg(value)))
result = dict(target='4584a3c2696625d27cf87f193a6260997348f85e', asof=datetime.now(timezone.utc).isoformat(),
              script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), rows=ntotal, freeze_sha256=diag['freeze_sha256'],
              groups=out, all_compared_saved_fields_match=True, scope='training diagnostics only; <=100 ms remaining; four frozen groups')
(Path(__file__).parent / 'short_diagnostic.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(dict(rows=ntotal, all_compared_saved_fields_match=True)))
