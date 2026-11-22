"""ASYNC END IDENTITY (FEASIBLE-CANCEL-HEADROOM-v1 receipt; session aea5bfb9, 2026-11-22). Review 3e84ca6d P2: ASYNC report.md:43 called END minus
eligible +1 s "the movement from +1 s to END"; that difference also holds the END value of the late / unsupported +1 s fills. This tool computes, from
the ASYNC receipt's COMMITTED arms copy (its raw sha256 = its committed audit), per market and arm: eligible +1 s (all_id_mo1s), the subsequent
movement on that cohort (all_id_move), the late / unsupported fills at END (all_id_late) and END (value). Predicates: the identity holds on every
market (1e-6 c); every market x arm agrees with the review's independently rebuilt end_decomposition.csv (committed at 3e84ca6d) within 0.01 c.
falsify(): a perturbed movement must break the agreement and a perturbed late value the identity, each while the clean data passes.
Writes outputs/async_identity.csv (per arm totals) and outputs/async_identity.json (write-once). Usage: async_identity.py"""
import gzip
import hashlib
import io
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path('/home/yuqing/ctaNew'); AFR = 'predictfun/reviews/ladder_async_fast_20261122'; REV = 'predictfun/reviews/automatic/20261122_async_fast_review/end_decomposition.csv'
OUT = REPO / 'predictfun/reviews/ladder_cancel_headroom_20261122/outputs'
ARMS = ('CURRENT_BASELINE', 'FRONT_BASELINE', 'CURRENT_ASYNC_FAST10', 'FRONT_ASYNC_FAST10')


def blob(rel):
    r = subprocess.run(['git', '-C', str(REPO), 'show', f'HEAD:{rel}'], capture_output=True); assert r.returncode == 0, f'{rel} is not committed'; return r.stdout


def predicates(A, V):
    ident = bool(np.allclose(A.all_id_mo1s + A.all_id_move + A.all_id_late, A.value, rtol=0, atol=1e-6))
    J = A.merge(V, on=['day', 'm', 'arm'], how='inner')
    agree = bool(len(J) == len(A) == len(V) and all(np.allclose(J[a], J[b], rtol=0, atol=0.01) for a, b in (('all_id_mo1s', 'mo1s'), ('all_id_move', 'move_to_end'),
                                                                                                       ('all_id_late', 'late'), ('value', 'end'))))
    return dict(identity_every_market=ident, agrees_with_review_every_market=agree, markets=int(A.groupby(['day', 'm']).ngroups), rows=int(len(J)))


def main():
    for n in ('async_identity.csv', 'async_identity.json'):
        assert not (OUT / n).exists(), f'{n} exists (write-once)'
    raw = gzip.decompress(blob(f'{AFR}/outputs/arms.csv.gz')); assert hashlib.sha256(raw).hexdigest() == json.loads(blob(f'{AFR}/outputs/audit.json'))['outputs']['arms.csv'], 'arms copy != audit'
    A = pd.read_csv(io.BytesIO(raw), dtype={'day': str}); A = A[A.arm.isin(ARMS)][['day', 'm', 'arm', 'all_id_mo1s', 'all_id_move', 'all_id_late', 'value']].reset_index(drop=True)
    V = pd.read_csv(io.BytesIO(blob(REV)), dtype={'day': str}); V = V[V.arm.isin(ARMS)].reset_index(drop=True); pr = predicates(A, V)
    f1 = predicates(A, V.assign(move_to_end=V.move_to_end + np.where(V.index == 0, 1., 0.)))['agrees_with_review_every_market'] is False
    f2 = predicates(A.assign(all_id_late=A.all_id_late + np.where(A.index == 0, 1., 0.)), V)['identity_every_market'] is False
    T = A.groupby('arm')[['all_id_mo1s', 'all_id_move', 'all_id_late', 'value']].sum().reindex(list(ARMS)).reset_index()
    T.columns = ['arm', 'eligible_1s', 'subsequent_movement_on_that_cohort', 'late_unsupported_1s_fills_at_END', 'END']; T.to_csv(OUT / 'async_identity.csv', index=False)
    res = dict(predicates=pr, falsifier=dict(perturbed_movement_breaks_agreement=bool(f1), perturbed_late_breaks_identity=bool(f2)),
               falsifier_ok=bool(f1 and f2 and pr['identity_every_market'] and pr['agrees_with_review_every_market']), totals=T.round(2).to_dict('records'),
               sources=dict(arms_copy=hashlib.sha256(raw).hexdigest(), review_table=hashlib.sha256(blob(REV)).hexdigest()))
    (OUT / 'async_identity.json').write_text(json.dumps(res, indent=1) + '\n'); print(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()
