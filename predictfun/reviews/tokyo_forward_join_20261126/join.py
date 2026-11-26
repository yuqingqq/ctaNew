"""Join independently polled maker legs for Tokyo attempt 2. No trading or model fit."""
import collections
import csv
import datetime as dt
import gzip
import hashlib
import io
import json
import statistics
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
DATA = 'e773da51753bcac087dae50794f6838fc943d753'
CODE = 'd0440d63a190f627ae960a7e4cd7cf425fbde22e'
TAPE = ROOT/'data/predict_fun/runs/20261021T162823Z_0dd45f2d5a/rest/20261126_13.jsonl.gz'
CUT = '2026-11-26T13:50:00Z'
def ms(t):
    return int(dt.datetime.fromisoformat(t.replace('Z','+00:00')).timestamp()*1000)

def blob(rev,path):
    return subprocess.check_output(['git','show',rev+':'+path],cwd=ROOT/'pmmmrust')

def insert(rows,key,body,ambiguous):
    if key in rows and rows[key]!=body:
        ambiguous.add(key)
    rows.setdefault(key,body)

def main():
    test={}; conflicts=set()
    insert(test,1,'a',conflicts); insert(test,1,'a',conflicts)
    assert not conflicts
    insert(test,1,'b',conflicts); assert conflicts=={1}
    path='tokyo-forward-trade-clock-20261126/'
    manifest=blob(DATA,path+'MANIFEST.sha256')
    assert hashlib.sha256(manifest).hexdigest()=='13335aa732fa1d548bf6d1accdb9749fb67be45696dbf288e585355ecd97b83b'
    hashes={p:h for h,p in (x.split() for x in manifest.decode().splitlines())}
    def read(p):
        data=blob(DATA,path+p)
        assert hashlib.sha256(data).hexdigest()==hashes['./'+p]
        return gzip.decompress(data).decode()
    windows=list(csv.DictReader(blob(CODE,'docs/receipts/20261126_tokyo_forward_trade_clock_request/tokyo/attempt2_windows_declared.csv').decode().splitlines()))
    bounds={int(r['market']):(ms(r['utc_start']),ms(r['utc_end_exclusive'])) for r in windows}
    tok={}; res={}; amb=set(); aux={}
    for r in csv.DictReader(io.StringIO(read('attempt2/report/legs_dedup.csv.gz'))):
        m=int(r['market']); ex=ms(r['executedAt']); a,b=bounds[m]
        assert (a<=ex<b)==(r['in_window']=='True')
        if not a<=ex<b: continue
        key=(r['transactionHash'],r['maker_hash'],int(r['leg']))
        assert key not in tok and int(r['variants'])==1
        tok[key]=(m,r['maker_quoteType'],r['maker_outcome'],int(r['maker_indexSet']),int(r['maker_price_wei']),int(r['maker_amount_wei']),ex,r['settlementId'])
        aux[key]=(int(r['first_seen_ns'])/1e6,int(r['upspace_side']),int(r['upspace_tick']))
    digest=hashlib.sha256()
    with gzip.open(TAPE,'rb') as f:
        for line in f:
            d=json.loads(line)
            if d['recv_ns']>=ms(CUT)*1000000: break
            digest.update(line)
            if d['path']!='/v1/orders/matches' or d['status']!=200: continue
            m=int(d['params']['marketId'])
            if m not in bounds: continue
            a,b=bounds[m]
            for r in json.loads(d['raw'])['data']:
                ex=ms(r['executedAt'])
                if not a<=ex<b: continue
                assert int(r['market']['id'])==m
                for i,l in enumerate(r['makers']):
                    key=(r['transactionHash'],l['hash'],i)
                    body=(m,l['quoteType'],l['outcome']['name'],int(l['outcome']['indexSet']),int(l['price']),int(l['amount']),ex,r['settlementId'])
                    insert(res,key,body,amb)
        else: raise ValueError('capture does not reach fixed observation cutoff')
    pos={}; cancels=collections.defaultdict(list)
    for line in read('attempt2/shadow/run-1795698680767/outputs.jsonl.gz').splitlines():
        req=json.loads(line).get('Request',{})
        if 'Place' in req:
            v=req['Place'];pos[v['market'],v['id']]=v
        if 'Cancel' in req:
            v=req['Cancel'];p=pos[v['market'],v['id']]
            cancels[v['market'],p['lane']%2,p['tick']].append((p['t'],v['t']))
    markets={}; timing=collections.Counter()
    for m in bounds:
        local={k for k,v in res.items() if v[0]==m}; remote={k for k,v in tok.items() if v[0]==m}
        groups={'research_only':local-remote,'tokyo_only':remote-local,'ambiguous':(remote&local)&amb,
                'field_mismatch':{k for k in (local&remote)-amb if res[k]!=tok[k]}}
        groups['verified']=(remote&local)-groups['ambiguous']-groups['field_mismatch']
        markets[m]={g:{'legs':len(keys),'shares':sum((tok[k] if k in tok else res[k])[5] for k in keys)/1e18} for g,keys in groups.items()}
        delays=[]
        for k in groups['verified']:
            v=tok[k]; ex,uuid=v[-2:]; first,s,p=aux[k]; u=int(uuid.replace('-','')[:12],16)
            assert uuid[14]=='7'
            delays.append(first-u)
            cs=[c for t,c in cancels[m,s,p] if t-2000<=ex<=c+2000]
            if cs:
                timing['near_canceled_order']+=1
                timing['first_seen_before_all_cancels']+=first<min(cs)
                timing['uuid_pm50']+=all(abs(c-u)>50 for c in cs)
                timing['executed_pm1000']+=all(abs(c-ex)>1000 for c in cs)
        if delays: markets[m]['median_first_seen_minus_uuid_ms']=round(statistics.median(delays),3)
    out=dict(pinned_code=CODE,pinned_data=DATA,research_cutoff=CUT,research_prefix_sha256=digest.hexdigest(),markets=markets,timing=timing)
    (HERE/'result.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(out))
if __name__=='__main__': main()
