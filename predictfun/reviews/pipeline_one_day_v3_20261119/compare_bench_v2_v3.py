"""Benchmark v3 against v2 (review 1adf6a33 P2), declared before v3 ran. v3 changes only the coherence flag (the moments' round trip), so:
(1) every model file is byte-identical (same exports, rows, pinned parameters, seed, one thread);
(2) every prediction column the two share is identical, row by row, except the *_ok flags;
(3) J_ok(v3) = J_ok(v2) and not J_zq on every row, and J_zq holds exactly where E[q] = 0 with |E[q * move]| > 1e-9 (recomputed here);
(4) the changed rows, by kind / side / day, with their orders and the largest |E[q * move]|, beside the review's counts
    (Oct 26: 23 RETAIN rows / 9 orders, max 36.5011 c; Oct 27: 15 rows / 6 orders, max 5.2075 c; no POST row).
Reads one column at a time. Usage: compare_bench_v2_v3.py [research data dir]"""
import json
import sys
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
R = sys.argv[1] if len(sys.argv) > 1 else 'data/predict_fun/research'; A, B = f'{R}/pipeline_bench_v2_20261119', f'{R}/pipeline_bench_v3_20261119'
ja, jb = json.loads(open(f'{A}/bench.json').read()), json.loads(open(f'{B}/bench.json').read())
models = ja['models'] == jb['models'] and len(ja['models']) == 44
print(f"MODELS IDENTICAL (byte for byte, {len(ja['models'])} files): {models}")
review = {'20261026': (23, 9, 36.5011), '20261027': (15, 6, 5.2075)}; all_ok = models
for day in ('20261026', '20261027'):
    fa, fb = f'{A}/pred_{day}.parquet', f'{B}/pred_{day}.parquet'; ca, cb = pq.read_schema(fa).names, pq.read_schema(fb).names
    common = [c for c in ca if c in cb]; same = []; diff = []
    for c in common:
        x, y = pd.read_parquet(fa, columns=[c])[c].values, pd.read_parquet(fb, columns=[c])[c].values
        eq = len(x) == len(y) and (np.array_equal(x, y, equal_nan=True) if x.dtype.kind == 'f' else np.array_equal(x, y))
        (same if eq else diff).append(c)
    k = pd.read_parquet(fb, columns=['kind', 'side', 'oid', 'J_Eq', 'J_Eqm', 'J_zq']); oka, okb = pd.read_parquet(fa, columns=['J_ok']).J_ok.values.astype(bool), pd.read_parquet(fb, columns=['J_ok']).J_ok.values.astype(bool)
    zq = k.J_zq.values.astype(bool); zero = (k.J_Eq.values <= 0) & (np.abs(k.J_Eqm.values) > 1e-9 * np.maximum(1., np.abs(k.J_Eqm.values)))
    flag = bool(np.array_equal(okb, oka & ~zq)); where = bool(np.array_equal(zq, zero & oka)); ch = oka & ~okb
    only_ok = set(diff) <= {c for c in common if c.endswith('_ok')} and set(cb) - set(ca) == {f'{x}_zq' for x in ('J', 'M', 'F')}
    print(f'{day}: {len(common)} shared columns, {len(same)} identical; differing {diff}; new {sorted(set(cb) - set(ca))}; ONLY THE COHERENCE FLAGS DIFFER: {only_ok}')
    print(f'   J_ok(v3) = J_ok(v2) AND NOT J_zq ON EVERY ROW: {flag}; J_zq EXACTLY WHERE E[q] = 0 WITH E[q * move] != 0 (and v2 accepted): {where}')
    for kind in ('post', 'retain'):
        for s in (0, 1):
            m = ch & (k.kind.values == kind) & (k.side.values == s)
            print(f'   {kind} side {s}: {int(m.sum())} rows changed, {len(set(k.oid.values[m].astype(int)))} orders, '
                  f'max |E[q * move]| {float(np.abs(k.J_Eqm.values[m]).max()) if m.any() else 0.:.4f} c')
    mr, mp = ch & (k.kind.values == 'retain'), ch & (k.kind.values == 'post'); n, o, mx = review[day]
    got = (int(mr.sum()), len(set(k.oid.values[mr].astype(int))), round(float(np.abs(k.J_Eqm.values[mr]).max()) if mr.any() else 0., 4))
    print(f'   RETAIN rows / orders / max changed {got}, the review {(n, o, mx)}: SAME AS THE REVIEW: {got == (n, o, mx) and not mp.any()} (POST rows changed: {int(mp.sum())})')
    all_ok &= only_ok and flag and where
print(f'BENCH v3 = v2 EXCEPT THE ROUND-TRIP COHERENCE FLAG: {all_ok}')
