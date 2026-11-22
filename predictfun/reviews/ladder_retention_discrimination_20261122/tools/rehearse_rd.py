"""PRE-FREEZE REHEARSAL of the discrimination pipeline and rd_checks.py (LONG-VALUE-RETENTION-DISCRIMINATION-v1; session aea5bfb9, 2026-11-22). Two synthetic
'ASYNC' days (a train day and an evaluation day; random tapes run by the frozen client) go through the builder's census / manifest, run_part, measure
and then every checker family and falsifier case, the paths pointed at a temporary directory. No saved data is read. Required: every family passes
except A_committed; every falsifier case is refused. Usage: rehearse_rd.py"""
import gzip
import hashlib
import importlib.util
import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO = Path('/home/yuqing/ctaNew'); HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO / 'predictfun/research'))
import pf_ladder_retention_discrimination as X             # noqa: E402
import pf_ladder_long_continuation_selftest as T           # noqa: E402

spec = importlib.util.spec_from_file_location('rd_checks', HERE.parent / 'rd_checks.py'); CH = importlib.util.module_from_spec(spec); spec.loader.exec_module(CH)
DAYS = ('20990101', '20990102')


def main():
    LC, P, M, LB, BE, AF = X.LC, X.P, X.M, X.LB, X.BE, X.AF; E_ = P.frozen_edges(); nt = int(np.ceil(T.END / BE.STEP)) + 2
    tmp = Path(tempfile.mkdtemp(dir=str(X.OUT.parent))); sv = dict(OUT=AF.OUT, af=LC.af_manifest, mi=LC.market_inputs, tr=X.TRAIN_DAYS, ev=X.EVAL_DAYS, pc=X.PER_CELL)
    try:
        inputs = {}; (tmp / 'af' / 'pilot').mkdir(parents=True)
        for i, day in enumerate(DAYS):
            rng = np.random.default_rng(31 + i); path = BE.rand_tape(rng, 70, 160, end=T.END); vals = np.array([-0.5, 0., 0.24, 0.47, 1.0], np.float32)
            cache = T._cache(nt, rng.choice(vals, nt), rng.choice(vals, nt)); inputs[day] = (path, dict(end_rel=T.END), cache)
            B = AF.ProfileBranch(path, T.END, dict(end_rel=T.END), False, LC.PROFILE); pol = P.PressureGuard(B, cache, E_); obs = P.QueueObs(B, pol); LB.run_episode(B, pol, observe=obs)
            L = LB.finish(B); rec = M.ledger_record(B, L); rec['guard_log'] = [list(x) for x in pol.log]; cm = P.command_arrays(B)['commands']; af = tmp / 'af' / 'pilot'
            gz = gzip.compress(json.dumps({'m': 1, X.ARM: rec, 'path_info': {'end_rel': T.END}}).encode()); (af / f'{day}.jsonl.gz').write_bytes(gz)
            np.savez(af / f'{day}_arrays.npz', **{f'1__{X.ARM}__commands': cm}); nb = (af / f'{day}_arrays.npz').read_bytes()
            (af / f'{day}.json').write_text(json.dumps(dict(sha256=dict(gz=hashlib.sha256(gz).hexdigest(), npz=hashlib.sha256(nb).hexdigest()))))
        gm_af = dict(days=list(DAYS), parts=dict(pilot={d: [1] for d in DAYS}, rest={d: [] for d in DAYS}), parent=dict(day_records={}))
        AF.OUT = tmp / 'af'; LC.af_manifest = lambda: gm_af; LC.market_inputs = lambda day, mans, g: ({1: inputs[day]}, E_)
        X.TRAIN_DAYS, X.EVAL_DAYS, X.PER_CELL = (DAYS[0],), (DAYS[1],), 6
        rc = tmp / 'receipt'; (rc / 'logs').mkdir(parents=True); man = X.build_manifest(rc / 'manifest.json', rc / 'census.csv')
        man.update(parts=dict(prefix=list(DAYS), rest=[]), train_days=[DAYS[0]], eval_days=[DAYS[1]]); (rc / 'manifest.json').write_text(json.dumps(man))
        for day in DAYS:
            X.run_part('prefix', day, out=tmp / 'out', require_frozen=False, man_path=rc / 'manifest.json')
        X.measure(out=tmp / 'out', require_frozen=False, man_path=rc / 'manifest.json')
        CH.R, CH.D, CH.AF = rc, tmp / 'out', tmp / 'af'
        E = CH.load(); base = CH.run_checks(E); fz = CH.falsify(dict(E), dict(base, all_ok=all(v for k, v in base['checks'].items() if k != 'A_committed')))
        failed = [k for k, v in base['checks'].items() if not v and k != 'A_committed']; unref = [k for k, v in fz['cases'].items() if not v['refused']]
        res = dict(checks=base['checks'], failed_except_committed=failed, falsifier_unrefused=unref, pairs=len(E['pairs']), statuses=E['pairs'].status.value_counts().to_dict(),
                   reading=E['audit']['evaluation']['reading'], details=base['details'], rehearsal_ok=bool(not failed and not unref and len(E['pairs']) >= 8))
        print(json.dumps(res, indent=1, default=str)[:4000])
        if not res['rehearsal_ok']:
            raise SystemExit('REHEARSAL FAILED')
    finally:
        AF.OUT, LC.af_manifest, LC.market_inputs = sv['OUT'], sv['af'], sv['mi']; X.TRAIN_DAYS, X.EVAL_DAYS, X.PER_CELL = sv['tr'], sv['ev'], sv['pc']; shutil.rmtree(tmp, ignore_errors=True)


if __name__ == '__main__':
    main()
