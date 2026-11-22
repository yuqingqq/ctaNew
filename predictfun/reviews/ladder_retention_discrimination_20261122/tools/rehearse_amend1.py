"""EQUIVALENCE REHEARSAL of amendment 1 (per-market loading; session aea5bfb9, 2026-11-22). One synthetic day with two 'pilot' markets and one 'rest'
market (random tapes run by the frozen client): the frozen run_part and the amendment's run_part_amend write their day outputs into separate
directories; their pairs / parity / ledgers bytes must be identical and the amendment must refuse a manifest that does not bind it. No saved data
is read. Usage: rehearse_amend1.py"""
import gzip
import hashlib
import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO = Path('/home/yuqing/ctaNew')
sys.path.insert(0, str(REPO / 'predictfun/research'))
import pf_ladder_retention_discrimination as X               # noqa: E402
import pf_ladder_retention_discrimination_amend1 as A1       # noqa: E402
import pf_ladder_long_continuation_selftest as T             # noqa: E402

DAY = '20990101'


def main():
    LC, P, M, LB, BE, AF = X.LC, X.P, X.M, X.LB, X.BE, X.AF; E_ = P.frozen_edges(); nt = int(np.ceil(T.END / BE.STEP)) + 2
    tmp = Path(tempfile.mkdtemp(dir=str(X.OUT.parent))); sv = dict(OUT=AF.OUT, af=LC.af_manifest, mi=LC.market_inputs, pc=X.PER_CELL)
    try:
        inputs, lines, arrs = {}, {'pilot': [], 'rest': []}, {'pilot': {}, 'rest': {}}
        for m, part in ((1, 'pilot'), (2, 'pilot'), (3, 'rest')):
            rng = np.random.default_rng(50 + m); path = BE.rand_tape(rng, 70, 160, end=T.END); vals = np.array([-0.5, 0., 0.24, 0.47, 1.0], np.float32)
            cache = T._cache(nt, rng.choice(vals, nt), rng.choice(vals, nt)); inputs[m] = (path, dict(end_rel=T.END), cache)
            B = AF.ProfileBranch(path, T.END, dict(end_rel=T.END), False, LC.PROFILE); pol = P.PressureGuard(B, cache, E_); obs = P.QueueObs(B, pol); LB.run_episode(B, pol, observe=obs)
            L = LB.finish(B); rec = M.ledger_record(B, L); rec['guard_log'] = [list(x) for x in pol.log]
            lines[part].append(json.dumps({'m': m, X.ARM: rec, 'path_info': {'end_rel': T.END}})); arrs[part][f'{m}__{X.ARM}__commands'] = P.command_arrays(B)['commands']
        for part in ('pilot', 'rest'):
            d = tmp / 'af' / part; d.mkdir(parents=True); gz = gzip.compress('\n'.join(lines[part]).encode()); (d / f'{DAY}.jsonl.gz').write_bytes(gz)
            np.savez(d / f'{DAY}_arrays.npz', **arrs[part]); nb = (d / f'{DAY}_arrays.npz').read_bytes()
            (d / f'{DAY}.json').write_text(json.dumps(dict(sha256=dict(gz=hashlib.sha256(gz).hexdigest(), npz=hashlib.sha256(nb).hexdigest()))))
        gm_af = dict(days=[DAY], parts=dict(pilot={DAY: [1, 2]}, rest={DAY: [3]}), parent=dict(day_records={}))
        AF.OUT = tmp / 'af'; LC.af_manifest = lambda: gm_af; LC.market_inputs = lambda day, mans, g: ({m: inputs[m] for m in mans}, E_); X.PER_CELL = 4
        rc = tmp / 'receipt'; (rc / 'logs').mkdir(parents=True); man = X.build_manifest(rc / 'manifest.json', rc / 'census.csv'); man['parts'] = dict(prefix=[DAY], rest=[])
        (rc / 'manifest.json').write_text(json.dumps(man)); gb = (rc / 'manifest.json').read_bytes()
        X.run_part('prefix', DAY, out=tmp / 'frozen', require_frozen=False, man_path=rc / 'manifest.json')
        am = dict(source=A1.own_sha(), manifest_sha256=hashlib.sha256(gb).hexdigest(), census_sha256=man['census_sha256'], applies_to=[f'prefix/{DAY}']); (rc / 'am.json').write_text(json.dumps(am))
        A1.run_part_amend('prefix', DAY, out=tmp / 'amend', require_frozen=False, man_path=rc / 'manifest.json', amend_path=rc / 'am.json')
        same = {k: (tmp / 'frozen' / 'prefix' / f'{DAY}{s}').read_bytes() == (tmp / 'amend' / 'prefix' / f'{DAY}{s}').read_bytes() for k, s in (('pairs', '_pairs.csv'), ('parity', '_parity.json'),
                                                                                                                                                  ('ledgers', '_ledgers.jsonl.gz'))}
        (rc / 'bad.json').write_text(json.dumps(dict(am, source='0' * 64)))
        try:
            A1.run_part_amend('prefix', DAY, out=tmp / 'amend2', require_frozen=False, man_path=rc / 'manifest.json', amend_path=rc / 'bad.json'); refused = False
        except ValueError as e:
            refused = 'does not bind' in str(e)
        n = len(json.loads((tmp / 'amend' / 'prefix' / f'{DAY}.json').read_text())['sha256'])
        res = dict(identical=same, refuses_unbound=refused, records=n, pairs=int(json.loads((tmp / 'frozen' / 'prefix' / f'{DAY}.json').read_text())['pairs']),
                   rehearsal_ok=bool(all(same.values()) and refused and json.loads((tmp / 'frozen' / 'prefix' / f'{DAY}.json').read_text())['pairs'] >= 6))
        print(json.dumps(res, indent=1))
        if not res['rehearsal_ok']:
            raise SystemExit('AMENDMENT REHEARSAL FAILED')
    finally:
        AF.OUT, LC.af_manifest, LC.market_inputs, X.PER_CELL = sv['OUT'], sv['af'], sv['mi'], sv['pc']; shutil.rmtree(tmp, ignore_errors=True)


if __name__ == '__main__':
    main()
