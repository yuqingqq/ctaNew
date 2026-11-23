"""SETUP of a downstream re-run root (amendment 3; session aea5bfb9, 2026-11-23). With MT_RUN=v<n> (n >= 2) pf_mt_common.OUT points at
maker_trace_<run>_20261123; this fills it with READ-ONLY symlinks to the shared v1 extract outputs (records / legs / tob / markets /
inventory / extract receipts, the depth/ directory) and to the v2 truth TABLE (truth/v2/own_orders.{parquet,json}) inside a real truth/v2/
directory, so the re-run's lifecycle / econ / truth scores land in new files and nothing under the v1 root can be written. Refuses if the
run root exists (write-once) or if a shared input is missing. Usage: MT_RUN=v2 setup_run.py"""
import os
import sys
from pathlib import Path

sys.path.insert(0, '/home/yuqing/ctaNew/predictfun/research')
import pf_mt_common as C   # noqa: E402

V1 = C.OUT_V1


def main():
    C.need(C.RUN != 'v1', 'set MT_RUN=v<n> (n >= 2)')
    C.need(not C.OUT.exists(), f'{C.OUT} exists (write-once)')
    days = (C.ID_DAY,) + C.MEASURE_DAYS + (C.GT_DAY,)
    files = [f'{k}_{d}.parquet' for d in days for k in ('records', 'legs', 'markets')] + [f'tob_{d}.parquet' for d in C.MEASURE_DAYS + (C.GT_DAY,)] + \
            [f'{k}_{d}.json' for d in days for k in ('inventory', 'extract')]
    miss = [f for f in files + ['depth', 'truth/v2/own_orders.parquet', 'truth/v2/own_orders.json'] if not (V1 / f).exists()]
    C.need(not miss, f'shared inputs missing under {V1}: {miss}')
    (C.OUT / 'truth' / 'v2').mkdir(parents=True)
    for f in files + ['depth']:
        os.symlink(V1 / f, C.OUT / f)
    for f in ('own_orders.parquet', 'own_orders.json'):
        os.symlink(V1 / 'truth' / 'v2' / f, C.OUT / 'truth' / 'v2' / f)
    C.RECEIPT.mkdir(parents=True, exist_ok=True)
    print(dict(run=C.RUN, root=str(C.OUT), links=len(files) + 3, receipt=str(C.RECEIPT)))


if __name__ == '__main__':
    main()
