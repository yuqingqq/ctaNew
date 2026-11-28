"""The trading test rerun on the final sources (after two selftest-only fixes) must reproduce run 1 exactly: every value, reading and
difference; only asof / seconds / source hashes may differ. Prints the predicate and the differing paths."""
import json
import sys
import numpy as np
R = sys.argv[1] if len(sys.argv) > 1 else 'data/predict_fun/research'
a = json.load(open(f'{R}/pipeline_policy_20261119_run1/policy_20261027.json')); b = json.load(open(f'{R}/pipeline_policy_20261119/policy_20261027.json'))
SKIP = {'asof', 'seconds', 'source_sha256', 'provenance', 'outputs'}


def differ(x, y, at=''):
    if isinstance(x, dict) and isinstance(y, dict):
        if set(x) != set(y):
            return [at + ' (keys)']
        return [d for k in x if k not in SKIP or at for d in differ(x[k], y[k], f'{at}.{k}')]
    if isinstance(x, (int, float)) and isinstance(y, (int, float)) and not isinstance(x, bool):
        return [] if (np.isnan(x) and np.isnan(y)) or x == y else [at]
    return [] if x == y else [at]


d = [p for p in differ({k: v for k, v in a.items() if k not in SKIP}, {k: v for k, v in b.items() if k not in SKIP})]
print(f'RERUN REPRODUCES RUN 1 EXACTLY: {not d}' + (f' - differs at {d[:10]}' if d else ''))
