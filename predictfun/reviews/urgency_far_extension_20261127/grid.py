"""Replicate the frozen BTC-only grid builder for the five extension dates."""
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parents[1]/'research'))
import pf_mt_btc as B
from pf_mt_decision import prev_day

DAYS=('20261023','20261031','20261101','20261102','20261103')
CONTROL='20261024'
OUT=B.C.REPO/'data/predict_fun/research/urgency_far_extension_20261127/btc'


def build(day):
    assert day in (*DAYS,CONTROL)
    OUT.mkdir(parents=True,exist_ok=True)
    opening=B.day_ms(day)
    roots=B.Roots()
    ledger,starts=B.ledger_info(roots.ledger,(opening-B.HOUR_MS-60000)*10**6,(opening+B.DAY_MS+60000)*10**6)
    pad=prev_day(day)
    stems=([pad+'_23'] if pad in DAYS else [])+[f'{day}_{h:02d}' for h in range(24)]
    ref=json.loads((HERE.parent/'take_exit_replay_20261125/v3_bn/bn_events.json').read_text())['files']
    src=B.ROOT_V2/'btc'
    if day==CONTROL:
        src_files=json.loads((src/f'btc_{day}.json').read_text())['bn']['files']
        ref={r['stem']:r for r in src_files}
    acc=B.BnDay(opening,starts)
    files=[]
    for i,stem in enumerate(stems):
        assert stem in ref
        rec,raw=B.load(roots.bn/f'{stem}.csv.gz')
        assert rec['state']=='read' and rec['sha256']==ref[stem]['sha256'],stem
        arrays,_=B.parse_bn(raw)
        del raw
        ok=np.isfinite(arrays['bid']) & np.isfinite(arrays['ask']) & (arrays['bid']>0) & (arrays['ask']>0)
        if not ok.all():
            arrays={k:v[ok] for k,v in arrays.items()}
        rec.update(acc.add(i,arrays))
        del arrays
        files.append(dict(stem=stem,**rec))
    holes=B.bn_holes(files,acc.starts)
    g,bid,ask,mid,updates,age,code=acc.finalize(holes)
    table=pa.Table.from_arrays([pa.array(v) for v in (g,bid,ask,mid,updates,age)]+[pc.take(pa.array(B.STATUS),pa.array(code))],schema=B.BTC_SCHEMA)
    assert table.num_rows==864000
    path=OUT/f'btc_{day}.parquet'
    pq.write_table(table,path)
    if day==CONTROL:
        pd.testing.assert_frame_equal(pq.read_table(path).to_pandas(),pq.read_table(src/path.name).to_pandas(),check_exact=True)
    record=dict(day=day,rows=len(g),holes=holes,ledger_sha=ledger['sha256'],inputs={r['stem']:r['sha256'] for r in files},
                status={s:int((code==k).sum()) for k,s in enumerate(B.STATUS)},sha256=B.sha_file(path),control_parity=day==CONTROL)
    (HERE/f'grid_{day}.json').write_text(json.dumps(record,indent=2)+'\n')
    print(day,'verified grid',record['status'],flush=True)


if __name__=='__main__':
    build(sys.argv[1])
