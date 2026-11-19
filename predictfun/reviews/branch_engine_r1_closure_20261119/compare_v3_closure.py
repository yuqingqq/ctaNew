"""Review c68a072c asks for "unchanged valid-path ledgers": slice v3 (module 13239a24, fefa5aca) against the same frozen selection rerun with the
closure module (saved fork contexts, the timed pull-effect predicate). Declared before the rerun: the orders table, the forks table, the branch
histories and every reading identical (exact). Prints one predicate per artifact and ALL TRUE at the end."""
import json
import sys

import pandas as pd

D = '/home/yuqing/ctaNew/data/predict_fun/research/branch_engine_20261119'; day = '20261024'
a_o, b_o = pd.read_csv(f'{D}/slice_v3_{day}_orders.csv'), pd.read_csv(f'{D}/slice_v3_{day}_closure_orders.csv')
a_f, b_f = pd.read_csv(f'{D}/slice_v3_{day}_forks.csv'), pd.read_csv(f'{D}/slice_v3_{day}_closure_forks.csv')
same_o = a_o.equals(b_o); same_f = a_f.equals(b_f)
same_h = json.load(open(f'{D}/slice_v3_{day}_histories.json')) == json.load(open(f'{D}/slice_v3_{day}_closure_histories.json'))
ra, rb = json.load(open(f'{D}/slice_v3_{day}.json')), json.load(open(f'{D}/slice_v3_{day}_closure.json'))
same_r = ra['readings'] == rb['readings']; src = ra['source_sha256'] != rb['source_sha256']
print(f'orders table identical: {same_o} ({len(a_o)} rows); forks table identical: {same_f} ({len(a_f)} rows); histories identical: {same_h}; '
      f'readings identical: {same_r}; module changed: {src}')
ok = same_o and same_f and same_h and same_r and src
print(f'VALID-PATH LEDGERS UNCHANGED (orders, forks, histories, readings): {ok}')
sys.exit(0 if ok else 1)
