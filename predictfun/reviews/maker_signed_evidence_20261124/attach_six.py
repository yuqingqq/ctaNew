"""Attach signed terms to the six PREVIOUSLY selected chains; bounded public reads.

plan reads existing consumed-day metadata only. run performs <=64 transactions
and <=64 receipts via eth_getTransactionByHash/Receipt, never sends transactions.
See attach_contract.md; no new action timing, market feature or strategy test.
"""
import collections
import csv
import hashlib
import json
import resource
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path('/home/yuqing/ctaNew')
sys.path.insert(0, str(ROOT/'predictfun/research'))
import pf_mt_signed_evidence_v3 as S

HERE = Path(__file__).resolve().parent
OLD = ROOT/'predictfun/reviews/maker_trace_20261122/v2/chains/chains_v4/chain_links.csv'
OUT = ROOT/'data/predict_fun/research/maker_signed_six_20261124'
RPC = 'https://bsc-dataseed.bnbchain.org/'


def plan():
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq
    chains = {}
    for r in csv.DictReader(OLD.open()):
        if r['link'] != 'post':
            continue
        c = {k:r[k] for k in ('chain','rank','day','mid','addr','order_hash','size_lower_bound','post_grade','evidence_grade')}
        S.need(c['chain'] not in chains, 'duplicate chain post')
        S.need(c['day'] in ('20261024','20261025','20261026','20261027'), 'unconsumed day')
        chains[c['chain']] = c
    S.need(len(chains)==6 and len({c['order_hash'] for c in chains.values()})==6, 'fixed six-chain population')
    txs, inputs, legs = collections.defaultdict(set), {str(OLD.relative_to(ROOT)):S.sha(OLD)}, []
    for day in sorted({c['day'] for c in chains.values()}):
        p = ROOT/f'data/predict_fun/research/maker_trace_v1_20261122/legs_{day}.parquet'
        inputs[str(p.relative_to(ROOT))] = S.sha(p)
        target = {c['order_hash']:c for c in chains.values() if c['day']==day}
        cols = ['day','mid','sid','tx','mk_idx','mk_hash','tk_hash','mk_signer','shares','status']
        for b in pq.ParquetFile(p).iter_batches(columns=cols, batch_size=65536):
            keys = pa.array(list(target))
            mask = pc.or_(pc.is_in(b.column('mk_hash'),value_set=keys),pc.is_in(b.column('tk_hash'),value_set=keys))
            for r in b.filter(mask).to_pylist():
                for k in ('mk_hash','tk_hash'):
                    h=r[k]
                    if h in target:
                        S.need(int(r['mid'])==int(target[h]['mid']), 'market identity')
                        if k=='mk_hash':
                            S.need(r['mk_signer'].lower()==target[h]['addr'].lower(), 'owner identity')
                        txs[h].add(r['tx'].lower())
                legs.append(r)
    S.need(set(txs)=={c['order_hash'] for c in chains.values()}, 'missing chain fills')
    union=sorted(set().union(*txs.values()))
    S.need(0<len(union)<=64,'64 transaction cap')
    srcs=[Path(__file__),HERE/'attach_contract.md', ROOT/'predictfun/research/pf_mt_signed_evidence_v3.py']
    S.dump(HERE/'attach_manifest.json',dict(chains=sorted(chains.values(),key=lambda c:int(c['chain'])),
            transactions=union, order_transactions={h:sorted(x) for h,x in sorted(txs.items())},
            matched_old_legs=legs, input_sha256=inputs,
            source_sha256={str(p.relative_to(ROOT)):S.sha(p) for p in srcs}))
    print(json.dumps({'chains':len(chains),'transactions':len(union),'old_leg_rows':len(legs)}))


def response_map(res, requests):
    S.need(isinstance(res,list), 'RPC list')
    ids={r['id'] for r in requests}
    S.need(len(res)==len(ids) and {r.get('id') for r in res}==ids, 'RPC membership')
    out={}
    reqmap={r['id']:r for r in requests}
    for r in res:
        S.need('error' not in r and isinstance(r.get('result'),dict), 'RPC unavailable')
        p=r['result']; tx=reqmap[r['id']]['params'][0]
        field='hash' if reqmap[r['id']]['method']=='eth_getTransactionByHash' else 'transactionHash'
        S.need(p.get(field,'').lower()==tx, 'RPC tx identity')
        out[r['id']]=p
    return out


def account(inp, rec, tx, wanted):
    S.need(inp['hash'].lower()==tx and rec['transactionHash'].lower()==tx,'tx identity')
    S.need(inp['blockHash']==rec['blockHash'] and inp['blockNumber']==rec['blockNumber'],'block identity')
    S.need(rec['status']=='0x1','failed settlement')
    byhash={o['hash']:o for o in S.decode(inp['input'])}
    events=[]; seen=set()
    for lg in rec['logs']:
        if lg['address'].lower()!=S.EX or not lg['topics'] or lg['topics'][0].lower()!=S.FILLED:
            continue
        h=lg['topics'][1].lower()
        if h not in wanted:
            continue
        S.need(h in byhash,'signed order absent')
        key=(tx,int(lg['logIndex'],16));S.need(key not in seen,'duplicate event');seen.add(key)
        o=byhash[h];mf,tf,fee=S.bind_event(o,lg)
        events.append(dict(order_hash=h,tx=tx,log_index=key[1],role=o['role'],values=o['values'],
                           making=mf,shares=tf if o['values'][10]==0 else mf,fee=fee,
                           block_number=int(rec['blockNumber'],16),transaction_index=int(rec['transactionIndex'],16)))
    return events


def run():
    m=json.loads((HERE/'attach_manifest.json').read_text())
    for rel,h in {**m['source_sha256'],**m['input_sha256']}.items():
        S.need(S.sha(ROOT/rel)==h,'file identity '+rel)
    for rel in list(m['source_sha256'])+[str((HERE/'attach_manifest.json').relative_to(ROOT))]:
        S.need(subprocess.check_output(['git','-C',str(ROOT),'show','HEAD:'+rel])==(ROOT/rel).read_bytes(),'freeze '+rel)
    S.need(not OUT.exists(),'write-once');OUT.mkdir()
    start=time.monotonic();cpu=time.process_time();all_events=[];cache=[]
    requests=[]
    for tx in m['transactions']:
        for method in ('eth_getTransactionByHash','eth_getTransactionReceipt'):
            requests.append(dict(jsonrpc='2.0',id=len(requests),method=method,params=[tx]))
    responses={}; failure=''
    # Max 16 batches of8, no automatic retry or alternative provider after failure.
    for i in range(0,len(requests),8):
        batch=requests[i:i+8]
        try:
            req=urllib.request.Request(RPC,data=json.dumps(batch).encode(),headers={'Content-Type':'application/json','User-Agent':'pf-research-readonly/1'})
            with urllib.request.urlopen(req,timeout=10) as f:
                raw=f.read(8*1024*1024+1)
            S.need(len(raw)<=8*1024*1024,'RPC byte cap')
            res=json.loads(raw)
            S.dump(OUT/f'batch_{i//8:02d}.json',res)
            responses.update(response_map(res,batch))
            S.need(time.monotonic()-start<90 and time.process_time()-cpu<35,'runtime cap')
        except Exception as e:
            failure=type(e).__name__+': '+str(e)[:180]
            break
    if failure:
        S.dump(HERE/'attach_result.json',dict(status='source_unavailable',reason=failure,returned_requests=len(responses),required=len(requests)))
        print(failure);return
    wanted=set(m['order_transactions'])
    for j,tx in enumerate(m['transactions']):
        inp,rec=responses[2*j],responses[2*j+1]
        all_events.extend(account(inp,rec,tx,wanted))
    output=[]
    for c in m['chains']:
        es=[e for e in all_events if e['order_hash']==c['order_hash']]
        S.need(bool(es),'missing chain events')
        v=es[0]['values'];S.need(all(e['values']==v for e in es),'signed terms conflict')
        S.need('0x'+v[1].to_bytes(20,'big').hex()==c['addr'].lower(),'chain owner')
        actual=sorted({e['tx'] for e in es})
        S.need(actual==m['order_transactions'][c['order_hash']],'chain transaction membership')
        making=sum(e['making'] for e in es);shares=sum(e['shares'] for e in es)
        old_maker_shares=sum(r['shares'] for r in m['matched_old_legs'] if r['mk_hash']==c['order_hash'])
        chain_maker_shares=sum(e['shares'] for e in es if e['role']=='maker')/S.E18
        S.need(abs(old_maker_shares-chain_maker_shares)<1e-6,'REST vs chain maker shares')
        q=v[6] if v[10]==0 else v[5]
        output.append(dict(chain=c['chain'],rank=c['rank'],day=c['day'],market=c['mid'],order_hash=c['order_hash'],maker=c['addr'],
                           signed_quote_shares=q/S.E18,observed_shares=shares/S.E18,old_size_lower_bound=c['size_lower_bound'],
                           original_making_wei=str(v[5]),observed_making_wei=str(making),unseen_making_budget_upper_wei=str(v[5]-making),
                           expiration_s=v[7],n_events=len(es),n_transactions=len(actual),
                           **S.classify(v,making,len(es),len(actual))))
    with (HERE/'six_chains.csv').open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(output[0]));w.writeheader();w.writerows(output)
    S.dump(OUT/'chain_events.json',all_events)
    # Falsifiers against an actual returned record, no re-fetch.
    tx=m['transactions'][0];bad=dict(responses[0]);bad['hash']='0x'+'00'*32
    try: account(bad,responses[1],tx,wanted)
    except ValueError: refused_identity=True
    else: raise AssertionError('corrupted transaction accepted')
    result=dict(status='PASS',chains=len(output),transactions=len(m['transactions']),events=len(all_events),
                budget_status=dict(collections.Counter(r['budget_status'] for r in output)),
                rest_maker_shares_reconciled=True,identity_falsifier=refused_identity,network_scope='existing six chains on consumed days only',
                output_sha256={p.name:S.sha(p) for p in sorted(OUT.glob('*.json'))},
                six_chains_sha256=S.sha(HERE/'six_chains.csv'),cpu_seconds=time.process_time()-cpu)
    S.dump(HERE/'attach_result.json',result);print(json.dumps(result,indent=2))


def selftest():
    req=[dict(id=1,method='eth_getTransactionByHash',params=['0xaa'])]
    S.need(response_map([dict(id=1,result={'hash':'0xaa'})],req)=={1:{'hash':'0xaa'}},'positive')
    for res in ([],[dict(id=2,result={'hash':'0xaa'})],[dict(id=1,result={'hash':'0xbb'})],[dict(id=1,result=None)]):
        try: response_map(res,req)
        except ValueError: pass
        else: raise AssertionError('bad RPC result accepted')
    print('RPC identity positive + four refusals PASS')


if __name__=='__main__':
    {'plan':plan,'run':run,'selftest':selftest}[sys.argv[1]]()
