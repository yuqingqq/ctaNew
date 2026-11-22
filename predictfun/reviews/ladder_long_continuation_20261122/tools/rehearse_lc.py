"""PRE-FREEZE REHEARSAL of the paired-continuation pipeline and lc_checks.py (LONG-CONTINUATION-PAIRED-VALUE-v1; session aea5bfb9, 2026-11-22). A synthetic
'ASYNC pilot' day (a random tape run by the frozen client, its record and command array) goes through the builder's census / manifest, run_part,
measure and then every checker family and falsifier case, with the paths pointed at a temporary directory. No saved data is read. Required:
every family passes except A_committed (nothing is committed before the freeze); every falsifier case is refused. Usage: rehearse_lc.py"""
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
import pf_ladder_long_continuation as X                   # noqa: E402
import pf_ladder_long_continuation_selftest as T          # noqa: E402

spec = importlib.util.spec_from_file_location('lc_checks', HERE.parent / 'lc_checks.py'); CH = importlib.util.module_from_spec(spec); spec.loader.exec_module(CH)
DAY = '20990101'


def main():
    P, M, LB, BE, AF = X.P, X.M, X.LB, X.BE, X.AF; E_ = P.frozen_edges(); nt = int(np.ceil(T.END / BE.STEP)) + 2
    tmp = Path(tempfile.mkdtemp(dir=str(X.OUT.parent))); saved = dict(OUT=AF.OUT, af_manifest=X.af_manifest, pilot_records=X.pilot_records, market_inputs=X.market_inputs)
    try:
        rng = np.random.default_rng(11); path = BE.rand_tape(rng, 70, 160, end=T.END); vals = np.array([-0.5, 0., 0.24, 0.47, 1.0], np.float32)
        cache = T._cache(nt, rng.choice(vals, nt), rng.choice(vals, nt))
        B = AF.ProfileBranch(path, T.END, dict(end_rel=T.END), False, X.PROFILE); pol = P.PressureGuard(B, cache, E_); obs = P.QueueObs(B, pol); LB.run_episode(B, pol, observe=obs)
        L = LB.finish(B); rec = M.ledger_record(B, L); rec['guard_log'] = [list(x) for x in pol.log]; cm = P.command_arrays(B)['commands']
        af = tmp / 'af' / 'pilot'; af.mkdir(parents=True); gz = gzip.compress(json.dumps({'m': 1, X.ARM: rec, 'path_info': {}}).encode()); (af / f'{DAY}.jsonl.gz').write_bytes(gz)
        np.savez(af / f'{DAY}_arrays.npz', **{f'1__{X.ARM}__commands': cm}); nb = (af / f'{DAY}_arrays.npz').read_bytes()
        (af / f'{DAY}.json').write_text(json.dumps(dict(sha256=dict(gz=hashlib.sha256(gz).hexdigest(), npz=hashlib.sha256(nb).hexdigest()))))
        gm_af = dict(days=[DAY], parts=dict(pilot={DAY: [1]}), parent=dict(day_records={}))
        AF.OUT = tmp / 'af'; X.af_manifest = lambda: gm_af; X.market_inputs = lambda day, mans, g: ({1: (path, dict(end_rel=T.END), cache)}, E_)
        rc = tmp / 'receipt'; (rc / 'logs').mkdir(parents=True); man = X.build_manifest(rc / 'manifest.json', rc / 'census.csv')
        man['parts'] = dict(subset=[DAY], rest=[]); (rc / 'manifest.json').write_text(json.dumps(man))
        X.run_part('subset', DAY, out=tmp / 'out', require_frozen=False, man_path=rc / 'manifest.json'); X.measure(out=tmp / 'out', require_frozen=False, man_path=rc / 'manifest.json')
        CH.R, CH.D, CH.AF = rc, tmp / 'out', tmp / 'af'
        E = CH.load(); E['my'] = CH.my_pairs(E); base = CH.run_checks(E); fz = CH.falsify(dict(E), dict(base, all_ok=all(v for k, v in base['checks'].items() if k != 'A_committed')))
        failed = [k for k, v in base['checks'].items() if not v and k != 'A_committed']; unref = [k for k, v in fz['cases'].items() if not v['refused']]
        res = dict(checks=base['checks'], failed_except_committed=failed, falsifier_unrefused=unref, pairs=len(E['pairs']), statuses=E['pairs'].status.value_counts().to_dict(),
                   reading=E['audit']['reading'], rehearsal_ok=bool(not failed and not unref and len(E['pairs']) >= 4))
        print(json.dumps(res, indent=1, default=str)[:4000])
        if not res['rehearsal_ok']:
            raise SystemExit('REHEARSAL FAILED')
    finally:
        AF.OUT = saved['OUT']; X.af_manifest, X.pilot_records, X.market_inputs = saved['af_manifest'], saved['pilot_records'], saved['market_inputs']; shutil.rmtree(tmp, ignore_errors=True)


if __name__ == '__main__':
    main()
