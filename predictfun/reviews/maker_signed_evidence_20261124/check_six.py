"""Independent six-chain integer recount directly from saved RPC receipt logs.

Verification only, declared by attach_contract. Does not import the attachment
builder or decode calldata. Hashes recomputed from signed tuples already bound
by the source gate; every value checked against original receipt logs.
"""
import copy
import csv
import hashlib
import json
from pathlib import Path
from Crypto.Hash import keccak

ROOT=Path('/home/yuqing/ctaNew'); HERE=Path(__file__).resolve().parent
OUT=ROOT/'data/predict_fun/research/maker_signed_six_20261124'
EX='0x8bc070bedab741406f4b1eb65a72bee27894b689'
FILL='0xd0a08e8c493f9c94f29311604c9de1b4e8c8d4c06bd0c789af57f2d65bfec0f6'


def require(ok,msg):
    if not ok: raise ValueError(msg)


def verify(table, events, manifest, responses):
    require(len(table)==6 and {r['order_hash'] for r in table}==set(manifest['order_transactions']),'six-order membership')
    require(len({(e['tx'],e['log_index']) for e in events})==len(events),'duplicate event')
    raw={}; wanted=set(manifest['order_transactions'])
    for rec in responses.values():
        if 'logs' not in rec: continue
        require(rec['status']=='0x1','failed receipt')
        for lg in rec['logs']:
            if lg['address'].lower()!=EX or not lg['topics'] or lg['topics'][0].lower()!=FILL or lg['topics'][1].lower() not in wanted:
                continue
            key=(rec['transactionHash'].lower(),int(lg['logIndex'],16))
            require(key not in raw,'duplicate raw')
            d=lg['data'][2:];words=[int(d[i:i+64],16) for i in range(0,len(d),64)]
            raw[key]=(lg,words)
    require(set(raw)=={(e['tx'],e['log_index']) for e in events},'raw membership')
    for e in events:
        lg,(ma,ta,mf,tf,fee)=raw[(e['tx'],e['log_index'])];v=e['values']
        require(e['order_hash']==lg['topics'][1].lower(),'event hash')
        require(e['making']==mf and e['fee']==fee and e['shares']==(tf if v[10]==0 else mf),'event accounting')
        require(int(lg['topics'][2],16)==v[1] and (ma,ta)==((0,v[4]) if v[10]==0 else (v[4],0)),'owner assets')
        h=keccak.new(digest_bits=256,data=bytes.fromhex('a852566c4e14d00869b6db0220888a9090a13eccdaea03713ff0a3d27bf9767c')+b''.join(x.to_bytes(32,'big') for x in v)).digest()
        h=keccak.new(digest_bits=256,data=b'\x19\x01'+bytes.fromhex('61e5a410200861954c73b7ec87a9054507b7b09d95035434808a719bf33bac8b')+h).hexdigest()
        require('0x'+h==e['order_hash'],'signed tuple')
    for r in table:
        es=[e for e in events if e['order_hash']==r['order_hash']];v=es[0]['values']
        require(all(e['values']==v for e in es),'terms invariant')
        mf=sum(e['making'] for e in es);q=sum(e['shares'] for e in es);txs=sorted({e['tx'] for e in es})
        require(txs==manifest['order_transactions'][r['order_hash']],'transaction population')
        require(int(r['original_making_wei'])==v[5] and int(r['observed_making_wei'])==mf,'original vs consumed')
        require(int(r['unseen_making_budget_upper_wei'])==v[5]-mf and mf<=v[5],'budget bound')
        require(float(r['signed_quote_shares'])==(v[6] if v[10]==0 else v[5])/10**18 and float(r['observed_shares'])==q/10**18,'size')
        require(r['budget_status']==('exhausted_by_observed_fills' if mf==v[5] else 'observed_budget_incomplete'),'classification')
        require(int(r['n_events'])==len(es) and int(r['n_transactions'])==len(txs),'repeats')
        require(all(r[k]=='False' for k in ('cancellation_identified','posting_time_identified','queue_front_identified')),'unsupported action')
    return {'chains':len(table),'raw_events':len(raw),'pass':True}


def main():
    manifest=json.loads((HERE/'attach_manifest.json').read_text());responses={}
    result=json.loads((HERE/'attach_result.json').read_text())
    for name,h in result['output_sha256'].items():
        require(hashlib.sha256((OUT/name).read_bytes()).hexdigest()==h,'file hash')
    for p in sorted(OUT.glob('batch_*.json')):
        for r in json.loads(p.read_text()):
            require(r['id'] not in responses,'RPC duplicate');responses[r['id']]=r['result']
    require(set(responses)==set(range(2*len(manifest['transactions']))),'RPC population')
    for i,tx in enumerate(manifest['transactions']):
        require(responses[2*i]['hash'].lower()==tx and responses[2*i+1]['transactionHash'].lower()==tx,'request binding')
    events=json.loads((OUT/'chain_events.json').read_text());table=list(csv.DictReader((HERE/'six_chains.csv').open()))
    out=verify(table,events,manifest,responses);bad={}
    for label,field,value in [('size','signed_quote_shares','99999'),('exhaustion','budget_status','observed_budget_incomplete'),
                              ('cancel','cancellation_identified','True')]:
        altered=copy.deepcopy(table);altered[0][field]=value
        try: verify(altered,events,manifest,responses)
        except ValueError as e: bad[label]=str(e)
        else: raise AssertionError('falsifier escaped')
    try: verify(table,events[1:],manifest,responses)
    except ValueError as e: bad['missing_event']=str(e)
    else: raise AssertionError('omission escaped')
    out['falsifiers']=bad
    with (HERE/'six_checks.json').open('x') as f:json.dump(out,f,indent=2);f.write('\n')
    print(json.dumps(out,indent=2))


if __name__=='__main__':main()
