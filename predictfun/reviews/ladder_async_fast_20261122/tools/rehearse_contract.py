"""PRE-FREEZE REHEARSAL of fa_checks.contract_errors (session aea5bfb9, 2026-11-22): the execution-contract reconstruction on (1) the 38 reused real
baseline ledgers (QUEUE-PRIORITY-SENSITIVITY-v1's CURRENT_QUEUE / FRONT_AT_ACTIVATION, BASELINE profile) and (2) synthetic random-tape runs of the
async module under both profiles, with planted corruptions that must be caught. Counts only; no economics is read."""
import gzip
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path('/home/yuqing/ctaNew'); sys.path.insert(0, str(REPO / 'predictfun/reviews/ladder_async_fast_20261122')); sys.path.insert(0, str(REPO / 'predictfun/research'))
import fa_checks as FC                                    # noqa: E402
QD = REPO / 'data/predict_fun/research/ladder_queue_front_v1_20261122'; n = bad = 0
for part in ('pilot', 'rest'):
    for p in sorted((QD / part).glob('*.jsonl.gz')):
        for ln in gzip.decompress(p.read_bytes()).decode().splitlines():
            r = json.loads(ln); end = float(r['path_info']['end_rel'])
            for arm in ('CURRENT_QUEUE', 'FRONT_AT_ACTIVATION'):
                e = FC.contract_errors(r[arm], 'BASELINE', end); n += 1; bad += int(bool(e))
                if e:
                    print('REAL', p.name, r['m'], arm, e[:5])
print('real baseline ledgers checked', n, 'with violations', bad)
import pf_ladder_async_fast as X                          # noqa: E402
import pf_ladder_rest_selftest as V1                      # noqa: E402,F401
P, LB, BE, M = X.P, X.LB, X.BE, X.M; E_ = P.frozen_edges(); rng = np.random.default_rng(91); syn = sbad = 0; planted = []
z = lambda nt: dict(x={0: np.zeros(nt, np.float32), 1: np.zeros(nt, np.float32)}, age={0: np.full(nt, 5., np.float32), 1: np.full(nt, 5., np.float32)},
                    status={0: np.zeros(nt, np.int8), 1: np.zeros(nt, np.int8)})
for i in range(3):
    p = BE.rand_tape(rng, 300, 500, end=30000.)
    for prof in ('BASELINE', 'ASYNC_FAST10'):
        B = X.ProfileBranch(p, 30000., dict(end_rel=30000.), i == 1, prof); pol = P.PressureGuard(B, z(600), E_); LB.run_episode(B, pol); Lg = LB.finish(B); rec = M.ledger_record(B, Lg)
        e = FC.contract_errors(rec, prof, 30000.); syn += 1; sbad += int(bool(e))
        if e:
            print('SYN', i, prof, e[:5])
        r2 = json.loads(json.dumps(rec)); r2['orders'][0]['act'] = float(r2['orders'][0]['submit']) + 3.; r3 = json.loads(json.dumps(rec)); r3['orders'][1]['size'] = 400.
        planted.append((bool(FC.contract_errors(r2, prof, 30000.)), bool(FC.contract_errors(r3, prof, 30000.))))
print('synthetic runs checked', syn, 'with violations', sbad, 'planted caught', planted)
