# The streamed (file-backed lgb.Sequence) vs in-memory LightGBM fit, measured before pf_r2_teacher was written; argument: the row count.
"""Feasibility: does a LightGBM Dataset built from a file-backed lgb.Sequence (pread batches) give the same model as one built from the
in-memory float32 array, with the pipeline's pinned parameters, weights, and more rows than bin_construct_sample_cnt (200,000)?"""
import os
import sys
import tempfile

import lightgbm as lgb
import numpy as np

sys.path.insert(0, '/home/yuqing/ctaNew/predictfun/research')
import pf_pipeline_bench as BX                                     # noqa: E402


class FileRows(lgb.Sequence):
    def __init__(self, path, n, ncol, batch_size=4096):
        self.fd, self.n, self.ncol, self.batch_size, self.rb = os.open(path, os.O_RDONLY), n, ncol, batch_size, ncol * 4

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            a, b, st = idx.indices(self.n); assert st == 1
            return np.frombuffer(os.pread(self.fd, (b - a) * self.rb, a * self.rb), np.float32).reshape(b - a, self.ncol)
        i = int(idx); i = i + self.n if i < 0 else i
        return np.frombuffer(os.pread(self.fd, self.rb, i * self.rb), np.float32).astype(np.float64)   # the sampler requires double (exact)


rng = np.random.default_rng(7); n, k = int(sys.argv[1]), 20
X = rng.normal(size=(n, k)).astype(np.float32); X[rng.random((n, k)) < .05] = np.nan
y = (X[:, 0] + .5 * np.nan_to_num(X[:, 1]) + rng.normal(size=n) > 0).astype(float); w = rng.uniform(.1, 3., n); w = w * n / w.sum()
names = [f'x{i}' for i in range(k)]
res = {}
with tempfile.TemporaryDirectory() as t:
    p = os.path.join(t, 'x.f32'); X.tofile(p)
    for obj, lab in (('binary', y), ('regression', X[:, 2].astype(float) * 2 + y)):
        prm = {**BX.PARAMS, 'objective': obj}
        b1 = lgb.train(prm, lgb.Dataset(X, label=lab, weight=w, feature_name=names), num_boost_round=30)
        b2 = lgb.train(prm, lgb.Dataset([FileRows(p, n, k)], label=lab, weight=w, feature_name=names), num_boost_round=30)
        s1, s2 = b1.model_to_string(), b2.model_to_string()
        trees = lambda s: s[s.index('Tree=0'):s.index('end of trees')]
        p1, p2 = b1.predict(X, num_threads=1), b2.predict(X, num_threads=1)
        res[obj] = dict(trees_equal=trees(s1) == trees(s2), max_pred_diff=float(np.max(np.abs(p1 - p2))))
print(res)
print('EQUAL' if all(v['trees_equal'] for v in res.values()) else 'DIFFERENT')
