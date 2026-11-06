"""Validate the isolated 8-window repair smoke; never reads raw market data."""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path('/home/yuqing/ctaNew')
sys.path.insert(0, str(ROOT))
from predictfun.research import pf_view_check as view
S4 = ROOT / 'data/predict_fun/research/step4'
DAY = '20261024'
TAG = 'repairloop_v5'
BASE = 'MICRO+G4+G2+G6+G5+G3+G1+V0+GS+XNA+XMIN1+XV+S60/mid@AV:5:25:0.06'
LADDER = BASE.replace('+XNA', '+LD2+XNA')


def stem(arm):
    return arm.replace('/', '__').replace('+', '_').replace('@', '_sz').replace(':', '-') + '.parquet'


def read(prefix, arm, engine):
    folder = S4 / f'{prefix}_tokyoreal_custom_e2_preds_t60k_{TAG}' / DAY
    if not (folder / 'summary.json').is_file():
        raise ValueError(f'REFUSED: missing smoke run {folder}')
    s = json.loads((folder / 'summary.json').read_text())
    assert s['day'] == DAY and s['env']['PF_MAX_WINDOWS'] == '8'
    digest = hashlib.sha256((ROOT / 'predictfun/research' / engine).read_bytes()).hexdigest()
    assert s['code_sha256'][engine] == digest, 'smoke was produced by different engine bytes'
    info = s['arms'][arm]
    w = pd.read_parquet(folder / 'win' / stem(arm))
    f = pd.read_parquet(folder / 'fills' / stem(arm))
    q = pd.read_parquet(folder / 'requests' / stem(arm))
    assert len(w) == info['windows'] == 8
    assert len(f) == info['fills'] and np.isclose(f.q.sum(), info['shares'], atol=.11)
    assert np.allclose(w.mtm_stop + w.carry_stop, w.pnl)
    assert np.allclose(f.groupby('mid').settle.sum().reindex(w.mid, fill_value=0), w.pnl)
    assert np.allclose(f.groupby('mid').q.sum().reindex(w.mid, fill_value=0), w.shares)
    assert info['live']['counters'].get('own_cross', 0) == 0
    assert info['live']['counters'].get('own_dup_price', 0) == 0
    return w, f, q, dict(windows=8, fills=len(f), requests=len(q), code_sha256=digest,
                         artifact=str(folder.relative_to(ROOT)))


if __name__ == '__main__':
    a = read('live', BASE, 'pf_step4_live.py')
    b = read('ladder', BASE, 'pf_step4_ladder.py')
    c = read('ladder', LADDER, 'pf_step4_ladder.py')
    assert len(view.read_run(f'live_tokyoreal_custom_e2_preds_t60k_{TAG}', DAY)) == 1
    assert len(view.read_run(f'ladder_tokyoreal_custom_e2_preds_t60k_{TAG}', DAY)) == 2
    pd.testing.assert_frame_equal(a[0], b[0])
    pd.testing.assert_frame_equal(a[1], b[1][a[1].columns])
    pd.testing.assert_frame_equal(a[2], b[2][a[2].columns])
    corrupt = b[0].copy(); corrupt.loc[0, 'pnl'] += 1
    try:
        pd.testing.assert_frame_equal(a[0], corrupt)
    except AssertionError:
        pass
    else:
        raise AssertionError('parity comparator did not detect corruption')
    try:
        read('missing', BASE, 'pf_step4_live.py')
    except ValueError:
        pass
    else:
        raise AssertionError('parity comparator accepted a missing run')
    result = dict(day=DAY, scope='integration smoke only, not performance evaluation',
                  live=a[3], ladder_base=b[3], ladder_engaged=c[3], parity='PASS',
                  corruption_control='PASS', missing_run_refusal='PASS', checker_runs_validated=2)
    Path(__file__).with_suffix('.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
