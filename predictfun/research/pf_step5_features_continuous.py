"""Versioned continuous-basis variant of the pinned research feature builder.

The independent Python sample producer pf_basis_continuous supplies causal
ratios from the original known-time stream. The original builder still computes
all 32 features, including its median and gap formulas. Only its duplicated
hour-file ratio arrays are replaced. Writes *_basis_v1.parquet; never overwrites
frozen/default rows. Requires PF_BASIS_SAMPLES pointing to a verified artifact.
"""
import hashlib
import os
from pathlib import Path

BASE_SHA = '8bb95a0090f03196169d62e92a66b8a01bbdb3845021793655b08b4c0555d553'
OLD = '''rel_t = np.concatenate(rel_ratio_t) if rel_ratio_t else np.empty(0); rel_v = np.concatenate(rel_ratio_v) if rel_ratio_v else np.empty(0)
o = np.argsort(rel_t); rel_t = rel_t[o]; rel_v = rel_v[o]'''
NEW = '''from .pf_basis_continuous import load as _continuous_basis_load
rel_t, rel_v = _continuous_basis_load(os.environ['PF_BASIS_SAMPLES'])'''


def source():
    path = Path(__file__).with_name('pf_step5_features_rl.py')
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != BASE_SHA:
        raise ValueError('research feature source identity changed')
    text = raw.decode()
    if text.count(OLD) != 1:
        raise ValueError('expected one basis array construction')
    return text.replace(OLD, NEW, 1)


def run():
    if not os.environ.get('PF_BASIS_SAMPLES'):
        raise ValueError('PF_BASIS_SAMPLES is required')
    if os.environ.get('PF_ROWS_SUFFIX', '_basis_v1') != '_basis_v1':
        raise ValueError('continuous features require separate _basis_v1 output')
    os.environ['PF_ROWS_SUFFIX'] = '_basis_v1'
    exec(compile(source(), __file__, 'exec'), dict(__name__='__main__', __package__=__package__, __file__=__file__))


if __name__ == '__main__':
    run()
