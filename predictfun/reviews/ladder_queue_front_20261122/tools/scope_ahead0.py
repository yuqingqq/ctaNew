"""SCOPE READ for the proposed QUEUE-PRIORITY-SENSITIVITY-v1 (USER plan 425e9fbc; session aea5bfb9, 2026-11-22), before any approval or run.
Counts only, from the accepted pressure arm (= FULL_11) of the pressure receipt's hash-bound day records: per accepted generation the
external queue ahead the engine set at activation (queue_venue ahead0 = [submit, ahead, rem]; refused generations have rem 0). These are
the generations FRONT_AT_ACTIVATION would change (ahead > 0) and by how much. No fill, value or economics is read."""
import gzip
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np

REPO = Path('/home/yuqing/ctaNew'); LED = REPO / 'data/predict_fun/research/ladder_pressure_v1_20261121/days'; PRS = 'predictfun/reviews/ladder_pressure_20261121/outputs/days'
out = {}
for d in ('20261024', '20261025', '20261026', '20261027'):
    rb = subprocess.run(['git', '-C', str(REPO), 'show', f'HEAD:{PRS}/{d}.json'], capture_output=True, check=True).stdout; gb = (LED / f'{d}.jsonl.gz').read_bytes()
    assert (LED / f'{d}.json').read_bytes() == rb and hashlib.sha256(gb).hexdigest() == json.loads(rb)['sha256']['gz'], d
    a, s = [], []
    for ln in gzip.decompress(gb).decode().splitlines():
        if ln:
            r = json.loads(ln)['pressure']; size = {int(o['gid']): float(o['size']) for o in r['orders']}
            for g, v in r['queue_venue']['ahead0'].items():
                if float(v[2]) > 0:
                    a.append(float(v[1])); s.append(size[int(g)])
    a, s = np.array(a), np.array(s)
    out[d] = dict(accepted=int(len(a)), ahead_zero=int((a == 0).sum()), ahead_pos=int((a > 0).sum()), median_ahead=float(np.median(a)), p75=float(np.percentile(a, 75)),
                  p90=float(np.percentile(a, 90)), median_ahead_over_size=float(np.median(a / s)), ahead_gt_size=int((a > s).sum()))
A = sum(v['accepted'] for v in out.values()); Z = sum(v['ahead_zero'] for v in out.values()); G = sum(v['ahead_gt_size'] for v in out.values())
print(json.dumps(dict(days=out, all=dict(accepted=A, ahead_zero=Z, share_already_front=Z / A, ahead_gt_size=G, share_ahead_gt_size=G / A)), indent=1))
