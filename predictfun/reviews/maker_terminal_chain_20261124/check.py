"""Independent signed terminal recount from raw receipt logs; no builder import."""
import collections
import copy
import csv
import hashlib
import json
from pathlib import Path
from Crypto.Hash import keccak

ROOT=Path('/home/yuqing/ctaNew');HERE=Path(__file__).resolve().parent
OUT=ROOT/'data/predict_fun/research/maker_terminal_chain_20261124'
EX='0x8bc070bedab741406f4b1eb65a72bee27894b689'
FILLED='0xd0a08e8c493f9c94f29311604c9de1b4e8c8d4c06bd0c789af57f2d65bfec0f6'
def need(x,msg):
    if not x:raise ValueError(msg)
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def verify(rows,events,m,responses):
    selected={r['mk_hash']:r for r in m['selected']};wanted=set(selected)
    need(len(rows)==len(wanted) and {r['order_hash'] for r in rows}==wanted,'population')
    need(len(events)==len({(e['tx'],e['log_index']) for e in events}),'duplicate event')
    raw={};cancels=collections.Counter();ct='0x'+keccak.new(digest_bits=256,data=b'OrderCancelled(bytes32)').hexdigest()
    for r in responses.values():
        if 'logs' not in r:continue
        need(r['status']=='0x1','receipt success')
        for lg in r['logs']:
            if lg['address'].lower()!=EX or not lg['topics']:continue
            if lg['topics'][0].lower()==ct and lg['topics'][1].lower() in wanted:cancels[lg['topics'][1].lower()]+=1
            if lg['topics'][0].lower()!=FILLED or lg['topics'][1].lower() not in wanted:continue
            key=(r['transactionHash'].lower(),int(lg['logIndex'],16));need(key not in raw,'raw duplicate')
            d=lg['data'][2:];need(len(d)==320,'event ABI')
            raw[key]=(lg,[int(d[i:i+64],16) for i in range(0,320,64)])
    need(set(raw)=={(e['tx'],e['log_index']) for e in events},'raw event membership')
    per=collections.defaultdict(list)
    for e in events:
        lg,(ma,ta,mf,tf,fee)=raw[(e['tx'],e['log_index'])];v=e['values']
        h=keccak.new(digest_bits=256,data=bytes.fromhex('a852566c4e14d00869b6db0220888a9090a13eccdaea03713ff0a3d27bf9767c')+b''.join(x.to_bytes(32,'big') for x in v)).digest()
        h='0x'+keccak.new(digest_bits=256,data=b'\x19\x01'+bytes.fromhex('61e5a410200861954c73b7ec87a9054507b7b09d95035434808a719bf33bac8b')+h).hexdigest()
        need(h==e['order_hash']==lg['topics'][1].lower(),'signed event hash')
        need(int(lg['topics'][2],16)==v[1] and (ma,ta)==((0,v[4]) if v[10]==0 else (v[4],0)),'owner/assets')
        need(e['making']==mf and e['shares']==(tf if v[10]==0 else mf) and e['fee']==fee,'event accounting')
        need((int(lg['topics'][3],16)==int(EX,16))==(e['role']=='taker'),'role')
        per[h].append(e)
    for r in rows:
        h=r['order_hash'];es=per[h];need(bool(es),'missing order');v=es[0]['values'];o=selected[h]
        need(all(e['values']==v for e in es),'terms')
        making=sum(e['making'] for e in es);shares=sum(e['shares'] for e in es)
        need(sorted({e['tx'] for e in es})==m['order_transactions'][h],'tx membership')
        need(int(r['original_making_wei'])==v[5] and int(r['observed_making_wei'])==making and making<=v[5],'budget')
        need(int(r['remaining_upper_wei'])==v[5]-making,'remaining upper')
        status='exhausted_by_observed_fills' if making==v[5] else ('explicit_onchain_cancel' if cancels[h] else 'unresolved_remainder')
        need(r['status']==status,'terminal class')
        need(r['private_pull_identified']=='False' and r['queue_front_identified']=='False','unsupported action')
        need(float(r['observed_shares'])==shares/10**18 and float(r['signed_quote_shares'])==(v[6] if v[10]==0 else v[5])/10**18,'quantities')
        old=sum(x['shares'] for x in m['legs'] if x['mk_hash']==h)
        need(abs(old-sum(e['shares'] for e in es if e['role']=='maker')/10**18)<1e-6,'REST reconciliation')
        if o['successor']=='none' or not o['successor_hash']:sg='no_recorded_successor'
        elif o['successor_before_rhi'] in ('True','true','1'):sg='overlap_possible'
        elif status=='exhausted_by_observed_fills':sg='later_order_after_exhaustion_command_unidentified'
        else:sg='unresolved_replacement_or_parallel_order'
        need(r['successor_grade']==sg,'successor grade')
    return {'orders':len(rows),'raw_events':len(raw),'statuses':dict(collections.Counter(r['status'] for r in rows)),
            'successor_grades':dict(collections.Counter(r['successor_grade'] for r in rows)),'pass':True}


def main():
    m=json.loads((HERE/'manifest.json').read_text());s=json.loads((HERE/'summary.json').read_text())
    for name,h in s['outputs'].items():need(digest(OUT/name)==h,'output hash')
    need(digest(HERE/'orders.csv')==s['orders_sha256'],'orders identity')
    responses={}
    for p in sorted(OUT.glob('batch_*.json')):
        for r in json.loads(p.read_text()):need(r['id'] not in responses,'RPC duplicate');responses[r['id']]=r['result']
    need(set(responses)==set(range(2*len(m['transactions']))),'RPC population')
    for i,tx in enumerate(m['transactions']):
        need(responses[2*i]['hash'].lower()==tx==responses[2*i+1]['transactionHash'].lower(),'RPC identity')
        need(responses[2*i]['blockHash']==responses[2*i+1]['blockHash'],'block binding')
    rows=list(csv.DictReader((HERE/'orders.csv').open()));events=json.loads((OUT/'events.json').read_text())
    out=verify(rows,events,m,responses)
    need(out['statuses']==s['statuses'] and out['successor_grades']==s['successor_grades'],'summary')
    # Independently verify deterministic selection from the old population.
    oldpath=ROOT/'predictfun/reviews/maker_trace_20261122/v2/chains/pull/pulls.csv'
    addr_rank={r['addr'].lower():int(r['rank']) for r in m['selected']};pop=collections.defaultdict(list)
    for r in csv.DictReader(oldpath.open()):
        if r['view']!='S_FIFO':continue
        rank=addr_rank[r['addr'].lower()];key='terminal_signed_v1|'+r['day']+'|'+r['mid']+'|'+r['mk_hash']
        pop[(r['day'],rank)].append((hashlib.sha256(key.encode()).hexdigest(),r['mk_hash']))
    for cell in m['cells']:
        day,rank=cell['day'],cell['rank'];expected=[h for _,h in sorted(pop[(day,rank)])[:8]]
        actual=[r['mk_hash'] for r in m['selected'] if r['day']==day and r['rank']==rank]
        need(actual==expected and cell['population']==len(pop[(day,rank)]),'selection')
    bad={}
    for label,field,val in [('amount','original_making_wei','1'),('private_pull','private_pull_identified','True'),('queue','queue_front_identified','True')]:
        changed=copy.deepcopy(rows);changed[0][field]=val
        try:verify(changed,events,m,responses)
        except ValueError as e:bad[label]=str(e)
        else:raise AssertionError('falsifier escaped')
    try:verify(rows,events[1:],m,responses)
    except ValueError as e:bad['omitted_event']=str(e)
    else:raise AssertionError('omitted event escaped')
    state_files=sorted(OUT.glob('state_[0-9]*.json'));prepared=json.loads((OUT/'state_requests.json').read_text())
    out['state_prepared']=len(prepared['requests']);out['state_batches_saved']=len(state_files)
    out['state_requests_submitted']=min(16*len(state_files),out['state_prepared'])
    out['state_responses_returned']=sum(len(json.loads(p.read_text())) for p in state_files)
    out['state_errors']=dict(collections.Counter(x.get('error',{}).get('message','no_error') for p in state_files for x in json.loads(p.read_text())))
    out['falsifiers']=bad
    with (HERE/'checks.json').open('x') as f:json.dump(out,f,indent=2);f.write('\n')
    print(json.dumps(out,indent=2))


if __name__=='__main__':main()
