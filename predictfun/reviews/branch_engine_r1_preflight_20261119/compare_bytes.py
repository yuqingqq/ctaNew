"""Review efac0452 asks for unchanged valid ledgers after the engine-state guards: slice v3 (R1 v2, module 13239a24) against the same frozen selection
rerun with the preflight module. Declared before the rerun: the orders CSV, the forks CSV and the histories JSON byte-for-byte identical (SHA-256),
every reading identical, the module changed. Prints one predicate per artifact and ALL TRUE at the end."""
import hashlib
import json
import sys

D = '/home/yuqing/ctaNew/data/predict_fun/research/branch_engine_20261119'; day = '20261024'
sha = lambda p: hashlib.sha256(open(p, 'rb').read()).hexdigest()
res = {x: sha(f'{D}/slice_v3_{day}_{x}') == sha(f'{D}/slice_v3_{day}_preflight_{x}') for x in ('orders.csv', 'forks.csv', 'histories.json')}
ra, rb = json.load(open(f'{D}/slice_v3_{day}.json')), json.load(open(f'{D}/slice_v3_{day}_preflight.json'))
res['readings'] = ra['readings'] == rb['readings']; res['module changed'] = ra['source_sha256'] != rb['source_sha256']
print(' | '.join(f'{k}: {v}' for k, v in res.items()))
ok = all(res.values()); print(f'VALID LEDGERS BYTE-IDENTICAL AFTER THE ENGINE-STATE GUARDS: {ok}')
sys.exit(0 if ok else 1)
