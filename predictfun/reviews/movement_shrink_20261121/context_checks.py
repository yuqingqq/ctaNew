"""PI0-MOVEMENT-SHRINK-v1 addendum (review 216ff603 item 2; 2026-11-21, session aea5bfb9). The frozen shrink_checks.py compares the eval rows'
numbers, not every decision-context column; this file closes that for the committed receipt without editing it. Every row of the committed
outputs/eval_rows.csv.gz (bytes bound by outputs/SHA256SUMS) is compared with its sources: the admission partitions (bytes bound to their records,
every record byte-equal to its committed receipt copy), the admission manifest's weights and the audit's committed opportunities table:
key identity (day, m, t_ms, side), head = stage, weight, the stage's own decision mid (its pf_mid input), the decision cell (cday, chour from the
stage's own decision time), status, and the horizon / queue / ordinal bins (equal to the audit table's and recomputed from the stage inputs).
FALSIFIER: the clean table passes; a mid +1 c, an hour +1, a flipped side, a changed horizon bin and a doubled weight each fail.
Writes context_checks.json. Usage (run_light): context_checks.py"""
import gzip
import hashlib
import io
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent; RES = Path('/home/yuqing/ctaNew/predictfun/research'); sys.path.insert(0, str(RES))
import pf_movement_shrink as M                           # noqa: E402  constants and committed_blob / committed_equal only

pd, RT, PA = M.pd, M.RT, M.PA
STR = {c: str for c in ('head', 'key', 'day', 'side', 'stage', 'cday', 'status', 'hbin', 'qbin', 'obin')}


def hb(t):
    return '(0,100]' if 0 < t <= 100 else '(100,500]' if 100 < t <= 500 else '(500,2000]' if 500 < t <= 2000 else '(2000,5000]' if 2000 < t <= 5000 else 'outside'


def qb(a, rem):
    if not (np.isfinite(a) and np.isfinite(rem)) or rem <= 0 or a < 0:
        return 'missing_or_invalid'
    x = a / rem
    return '0' if x == 0 else '(0,1]' if x <= 1 else '(1,3]' if x <= 3 else '>3'


def ob(o):
    return 'missing_or_invalid' if not np.isfinite(o) else '1' if o <= 1 else '2' if o <= 2 else '>=3'


def gkey(r):
    return json.dumps([str(r['day']), int(r['m']), int(r['t_ms']), int(r['side']), int(r['gid']), int(r['decision_ms'])])


def load():
    sums = {ln.split('  ')[1]: ln.split('  ')[0] for ln in (HERE / 'outputs' / 'SHA256SUMS').read_text().splitlines() if ln}
    b = (HERE / 'outputs' / 'eval_rows.csv.gz').read_bytes()
    M.C.require(hashlib.sha256(b).hexdigest() == sums['eval_rows.csv.gz'] and M.committed_equal(f'{M.RECEIPT}/outputs/eval_rows.csv.gz', HERE / 'outputs' / 'eval_rows.csv.gz'),
                'the eval rows are not the committed receipt bytes')
    R = pd.read_csv(io.BytesIO(gzip.decompress(b)), float_precision='round_trip', dtype=STR)
    adm = M.MA.ADM; am = json.loads((adm / 'manifest.json').read_text()); rows = {}
    M.C.require(M.MA.inputs_frozen(), 'a cohort manifest is not its committed copy')
    for k in am['parts']:
        pb = (adm / 'parts' / f'part_{int(k):03d}.jsonl.gz').read_bytes(); rec = json.loads((adm / 'parts' / f'part_{int(k):03d}.json').read_text())
        M.C.require(hashlib.sha256(pb).hexdigest() == rec['sha256'] and M.committed_equal(f'{M.ADM_REC_REL}/part_{int(k):03d}.json', adm / 'parts' / f'part_{int(k):03d}.json'),
                    f'admission part {k}: identity')
        rows.update({gkey(r): r for r in (json.loads(x) for x in gzip.decompress(pb).decode().splitlines() if x)})
    wsel = {gkey(g): float(g['weight']) for g in am['selected']}
    A = pd.read_csv(io.BytesIO(gzip.decompress(M.committed_blob(M.AUDIT_OPP_REL))), float_precision='round_trip', dtype=STR)
    A = A[A.cohort == 'admitted'].set_index(['key', 'stage'])
    return R, rows, wsel, A


def check(R, rows, wsel, A):
    bad = []
    for r in R.itertuples(index=False):
        src = rows.get(r.key)
        if src is None or (r.key, r.stage) not in A.index:
            bad.append(('no_source', r.key)); continue
        x = src['stages'][r.stage]; a = x['inputs']; g = lambda n: float(a[RT.FEATS.index(n)]); cell = PA.cell_of(int(src['t_ms']), float(x['t']), int(src['side']))
        au = A.loc[(r.key, r.stage)]
        exp = dict(head=r.stage, day=str(src['day']), m=int(src['m']), t_ms=int(src['t_ms']), side=str(int(src['side'])), weight=wsel.get(r.key), cday=str(cell[0]),
                   chour=int(cell[2]), mid=g('pf_mid'), status=str(au.acc_status), hbin=hb(g('t_to_end')), qbin=qb(g('fp_queue'), g('fp_remaining')), obin=ob(g('ctx_generation_ordinal')))
        got = dict(head=r.head, day=r.day, m=int(r.m), t_ms=int(r.t_ms), side=r.side, weight=float(r.weight), cday=r.cday, chour=int(float(r.chour)), mid=float(r.mid), status=r.status,
                   hbin=r.hbin, qbin=r.qbin, obin=r.obin)
        diff = [k for k, v in exp.items() if (abs(float(got[k]) - float(v)) > 1e-12 if isinstance(v, float) else got[k] != v)]
        diff += [k for k in ('hbin', 'qbin', 'obin', 'day') if str(au[k]) != str(got[k])] + (['status'] if str(au.acc_status) != 'ok' else [])
        if diff:
            bad.append((r.key, r.stage, sorted(set(diff))))
    return bad


def main():
    R, rows, wsel, A = load(); clean = check(R, rows, wsel, A); res = {}

    def corrupt(name, fn):
        X = R.copy(); fn(X); res[name] = dict(flagged=bool(check(X, rows, wsel, A)))
    corrupt('mid_plus_1c', lambda X: X.__setitem__('mid', X.mid.astype(float) + np.r_[1., np.zeros(len(X) - 1)]))
    corrupt('hour_plus_1', lambda X: X.__setitem__('chour', X.chour.astype(int) + np.r_[1, np.zeros(len(X) - 1, int)]))
    corrupt('side_flipped', lambda X: X.__setitem__('side', ['1' if (i == 0 and s == '0') else ('0' if (i == 0 and s == '1') else s) for i, s in enumerate(X.side)]))
    corrupt('horizon_bin_changed', lambda X: X.__setitem__('hbin', ['(0,100]' if (i == 0 and h != '(0,100]') else ('(100,500]' if i == 0 else h) for i, h in enumerate(X.hbin)]))
    corrupt('weight_doubled', lambda X: X.__setitem__('weight', X.weight.astype(float) * np.r_[2., np.ones(len(X) - 1)]))
    ok = bool(not clean and len(R) == 311 and all(v['flagged'] for v in res.values()))
    js = dict(rows=int(len(R)), heads={h: int((R['head'] == h).sum()) for h in M.HEADS}, clean_failures=clean[:5], n_clean_failures=len(clean), falsifier=res, all_ok=ok, asof=M.C.stamp())
    (HERE / 'context_checks.json').write_text(json.dumps(js, indent=1, sort_keys=True, default=str) + '\n'); print(json.dumps(js, indent=1, default=str))
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
