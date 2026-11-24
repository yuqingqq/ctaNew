"""Fixed terminal cohort; contract.md is the pre-run specification."""
import collections
import csv
import hashlib
import importlib.util
import json
import resource
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT=Path('/home/yuqing/ctaNew');HERE=Path(__file__).resolve().parent
PREV=ROOT/'predictfun/reviews/maker_signed_evidence_20261124'
OLD=ROOT/'predictfun/reviews/maker_trace_20261122/v2/chains'
OUT=ROOT/'data/predict_fun/research/maker_terminal_chain_20261124'
spec=importlib.util.spec_from_file_location('signed_attach',PREV/'attach_six.py')
A=importlib.util.module_from_spec(spec);spec.loader.exec_module(A);S=A.S
DAYS=('20261024','20261025','20261026','20261027')


def choose(rows,ranks):
    cells=collections.defaultdict(list);seen=set()
    for r in rows:
        if r['view']!='S_FIFO':continue
        S.need(r['day'] in DAYS,'unconsumed day')
        rank=ranks[r['addr'].lower()];key=(r['day'],r['mid'],r['mk_hash'])
        S.need(key not in seen,'duplicate order');seen.add(key)
        score=hashlib.sha256(('terminal_signed_v1|'+'|'.join(key)).encode()).hexdigest()
        cells[(r['day'],rank)].append((score,r['mk_hash'],r))
    selected=[];census=[]
    for day in DAYS:
        for rank in (1,2,3):
            pop=sorted(cells[(day,rank)],key=lambda x:x[:2]);pick=pop[:8]
            census.append(dict(day=day,rank=rank,population=len(pop),selected=len(pick)))
            selected.extend(dict(r,rank=rank,sampling_hash=score) for score,h,r in pick)
    return selected,census


def classify(original,making,cancel=False):
    S.need(0<making<=original,'making budget')
    if making==original:return 'exhausted_by_observed_fills'
    return 'explicit_onchain_cancel' if cancel else 'unresolved_remainder'


def state_grade(raw,original,making):
    if not isinstance(raw,str) or not raw.startswith('0x') or len(raw)!=130:return 'state_schema_unavailable',None,None
    closed,remaining=int(raw[2:66],16),int(raw[66:],16)
    S.need(closed in (0,1) and 0<=remaining<=original-making,'state budget discrepancy')
    if original==making:S.need(closed==1 and remaining==0,'exhaustion positive control')
    if closed:return 'closed_cause_not_identified',closed,remaining
    return ('onchain_remaining_at_block_end' if remaining>0 else 'open_zero_remaining_unresolved'),closed,remaining


def successor_grade(r,status):
    if r['successor']=='none' or not r['successor_hash']:return 'no_recorded_successor'
    if r['successor_before_rhi'] in ('True','true','1'):return 'overlap_possible'
    if status=='exhausted_by_observed_fills':return 'later_order_after_exhaustion_command_unidentified'
    return 'unresolved_replacement_or_parallel_order'


def source_files():
    return [Path(__file__),HERE/'contract.md',HERE/'selftest.py',PREV/'attach_six.py',ROOT/'predictfun/research/pf_mt_signed_evidence_v3.py']


def plan():
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq
    p=ROOT/'data/predict_fun/research/maker_trace_v1_20261122/econ/identities.json'
    ranks={r['addr'].lower():r['rank'] for r in json.loads(p.read_text())['ex_ante']['top10_shares'] if r['rank']<=3}
    selected,census=choose(list(csv.DictReader((OLD/'pull/pulls.csv').open())),ranks)
    inputs={str(p.relative_to(ROOT)):S.sha(p),str((OLD/'pull/pulls.csv').relative_to(ROOT)):S.sha(OLD/'pull/pulls.csv')}
    byhash={r['mk_hash']:r for r in selected};txs=collections.defaultdict(set);legs=[]
    S.need(len(byhash)==len(selected),'hash reused across markets')
    for day in DAYS:
        p=ROOT/f'data/predict_fun/research/maker_trace_v1_20261122/legs_{day}.parquet'
        inputs[str(p.relative_to(ROOT))]=S.sha(p)
        targets={h:r for h,r in byhash.items() if r['day']==day};keys=pa.array(list(targets))
        if not targets:continue
        cols=['day','mid','sid','tx','mk_idx','mk_hash','tk_hash','mk_signer','shares','status']
        for b in pq.ParquetFile(p).iter_batches(columns=cols,batch_size=65536):
            mask=pc.or_(pc.is_in(b.column('mk_hash'),value_set=keys),pc.is_in(b.column('tk_hash'),value_set=keys))
            for r in b.filter(mask).to_pylist():
                for k in ('mk_hash','tk_hash'):
                    if r[k] in targets:
                        c=targets[r[k]];S.need(int(c['mid'])==r['mid'],'market identity')
                        if k=='mk_hash':S.need(c['addr'].lower()==r['mk_signer'].lower(),'owner identity')
                        txs[r[k]].add(r['tx'].lower())
                legs.append(r)
    S.need(set(txs)==set(byhash),'missing selected fills')
    union=sorted(set().union(*txs.values()));S.need(len(union)<=384,'transaction projection cap')
    m=dict(selected=selected,cells=census,transactions=union,order_transactions={h:sorted(v) for h,v in sorted(txs.items())},legs=legs,
           input_sha256=inputs,source_sha256={str(p.relative_to(ROOT)):S.sha(p) for p in source_files()})
    S.dump(HERE/'manifest.json',m)
    print(json.dumps(dict(orders=len(selected),transactions=len(union),old_legs=len(legs),cells=census),indent=2))


def bound():
    m=json.loads((HERE/'manifest.json').read_text())
    for rel,h in {**m['source_sha256'],**m['input_sha256']}.items():S.need(S.sha(ROOT/rel)==h,'file identity '+rel)
    for rel in list(m['source_sha256'])+[str((HERE/'manifest.json').relative_to(ROOT))]:
        S.need(subprocess.check_output(['git','-C',str(ROOT),'show','HEAD:'+rel])==(ROOT/rel).read_bytes(),'freeze '+rel)
    return m


def rpc(batch):
    request=urllib.request.Request(A.RPC,data=json.dumps(batch).encode(),headers={'Content-Type':'application/json','User-Agent':'pf-research-readonly/1'})
    with urllib.request.urlopen(request,timeout=10) as f:raw=f.read(8*1024*1024+1)
    S.need(len(raw)<=8*1024*1024,'byte cap');return json.loads(raw)


def write_csv(p,rows):
    if not rows:return
    with p.open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)


def run():
    m=bound();S.need(not OUT.exists(),'write-once');OUT.mkdir()
    start=time.monotonic();cpu=time.process_time();requests=[];responses={};failure=''
    for tx in m['transactions']:
        for method in ('eth_getTransactionByHash','eth_getTransactionReceipt'):
            requests.append(dict(jsonrpc='2.0',id=len(requests),method=method,params=[tx]))
    for i in range(0,len(requests),16):
        try:
            batch=requests[i:i+16];res=rpc(batch);S.dump(OUT/f'batch_{i//16:03d}.json',res)
            responses.update(A.response_map(res,batch))
            S.need(time.monotonic()-start<180 and time.process_time()-cpu<35,'runtime cap')
            S.need(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss<1000*1024,'RSS cap')
        except Exception as e:
            failure=type(e).__name__+': '+str(e)[:180];break
    bytx={};errors={};events=[];cancels={};wanted=set(m['order_transactions'])
    for i,tx in enumerate(m['transactions']):
        if 2*i not in responses or 2*i+1 not in responses:errors[tx]='source_unavailable';continue
        try:
            inp,rec=responses[2*i],responses[2*i+1]
            es=A.account(inp,rec,tx,wanted);events.extend(es);bytx[tx]=rec
            for lg in rec['logs']:
                if lg['address'].lower()==S.EX and lg['topics'] and lg['topics'][0].lower()==S.CANCELLED:
                    h=lg['topics'][1].lower()
                    if h in wanted:cancels.setdefault(h,[]).append(dict(tx=tx,log_index=int(lg['logIndex'],16),block_number=int(rec['blockNumber'],16),transaction_index=int(rec['transactionIndex'],16)))
        except ValueError as e:errors[tx]='identity_refused:'+str(e)
    per=collections.defaultdict(list)
    for e in events:per[e['order_hash']].append(e)
    state_requests=[];state_orders=[]
    selector=S.kh(b'orderStatus(bytes32)')[:4].hex()
    for r in m['selected']:
        h=r['mk_hash']
        if any(tx in errors for tx in m['order_transactions'][h]) or not per[h]:continue
        es=per[h];last=max(es,key=lambda e:(e['block_number'],e['transaction_index'],e['log_index']))
        rec=bytx[last['tx']]
        state_requests.append(dict(jsonrpc='2.0',id=len(state_orders),method='eth_call',params=[{'to':S.EX,'data':'0x'+selector+h[2:]},rec['blockNumber']]))
        state_orders.append(dict(order_hash=h,block_number=rec['blockNumber'],block_hash=rec['blockHash']))
    states={};state_transport=''
    for i in range(0,len(state_requests),16):
        try:
            batch=state_requests[i:i+16];res=rpc(batch);S.dump(OUT/f'state_{i//16:03d}.json',res)
            S.need(isinstance(res,list) and len(res)==len(batch) and {r['id'] for r in res}=={r['id'] for r in batch},'state response membership')
            for x in res:states[state_orders[x['id']]['order_hash']]=x
            S.need(time.monotonic()-start<180 and time.process_time()-cpu<35,'runtime cap')
        except Exception as e:state_transport=type(e).__name__+': '+str(e)[:160];break
    S.dump(OUT/'state_requests.json',dict(requests=state_requests,orders=state_orders))
    out=[]
    for r in m['selected']:
        h=r['mk_hash'];es=per[h];txs=m['order_transactions'][h]
        row=dict(day=r['day'],rank=r['rank'],market=r['mid'],maker=r['addr'],order_hash=h,old_window_status=r['window_status'],
                 status='source_unavailable',reason='|'.join(errors[t] for t in txs if t in errors),signed_quote_shares='',observed_shares='',
                 original_making_wei='',observed_making_wei='',remaining_upper_wei='',expiration_s='',n_events=len(es),n_transactions=len(txs),
                 linked_cancel_events=len(cancels.get(h,[])),state_status='source_unavailable',closed_flag='',state_remaining_wei='',
                 successor=r['successor'],successor_hash=r['successor_hash'],successor_overlap=r['successor_before_rhi'],successor_grade='',
                 old_btc_against_1s_union=r['btc_against_1s_union'],old_inv_built_5s_union=r['inv_built_5s_union'],
                 old_removal_lo_ms=r['removal_lo_ms'],old_removal_hi_ms=r['removal_hi_ms'],private_pull_identified=False,queue_front_identified=False)
        if not row['reason']:
            try:
                S.need(bool(es),'no events');v=es[0]['values'];S.need(all(e['values']==v for e in es),'terms conflict')
                S.need('0x'+v[1].to_bytes(20,'big').hex()==r['addr'].lower(),'owner')
                S.need(sorted({e['tx'] for e in es})==txs,'transaction membership')
                making=sum(e['making'] for e in es);shares=sum(e['shares'] for e in es)
                old=sum(x['shares'] for x in m['legs'] if x['mk_hash']==h)
                actual=sum(e['shares'] for e in es if e['role']=='maker')/S.E18
                S.need(abs(old-actual)<1e-6,'REST maker quantity')
                row.update(status=classify(v[5],making,bool(cancels.get(h))),signed_quote_shares=(v[6] if v[10]==0 else v[5])/S.E18,
                           observed_shares=shares/S.E18,original_making_wei=str(v[5]),observed_making_wei=str(making),remaining_upper_wei=str(v[5]-making),expiration_s=v[7])
                x=states.get(h,{})
                if 'result' in x:
                    grade,closed,remain=state_grade(x['result'],v[5],making)
                    row.update(state_status=grade,closed_flag=closed,state_remaining_wei=str(remain) if remain is not None else '')
                else:row['state_status']='archive_or_rpc_unavailable'
            except ValueError as e:row.update(status='source_unavailable',reason=str(e))
        row['successor_grade']=successor_grade(r,row['status']);out.append(row)
    write_csv(HERE/'orders.csv',out);S.dump(OUT/'events.json',events);S.dump(OUT/'cancels.json',cancels)
    cells=[]
    for c in m['cells']:
        rs=[r for r in out if r['day']==c['day'] and r['rank']==c['rank']]
        cells.append(dict(c,statuses=dict(collections.Counter(r['status'] for r in rs))))
    tabs=[]
    for field in ('old_btc_against_1s_union','old_inv_built_5s_union'):
        for (status,value),n in sorted(collections.Counter((r['status'],r[field]) for r in out).items()):
            tabs.append(dict(field=field,status=status,old_window_class=value,n=n))
    summary=dict(as_of_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),source_commit=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD']).decode().strip(),
                 orders=len(out),transactions=len(m['transactions']),fill_events=len(events),cells=cells,
                 statuses=dict(collections.Counter(r['status'] for r in out)),state_statuses=dict(collections.Counter(r['state_status'] for r in out)),
                 successor_grades=dict(collections.Counter(r['successor_grade'] for r in out)),linked_cancels=sum(len(x) for x in cancels.values()),
                 transport_failure=failure,state_transport_failure=state_transport,gate_pass=all(r['status']!='source_unavailable' for r in out),
                 old_window_crosstabs=tabs,private_pull_identified=0,queue_front_identified=0,cpu_seconds=time.process_time()-cpu,
                 outputs={p.name:S.sha(p) for p in OUT.glob('*.json')},orders_sha256=S.sha(HERE/'orders.csv'))
    S.dump(HERE/'summary.json',summary);print(json.dumps({k:v for k,v in summary.items() if k not in ('outputs','old_window_crosstabs','cells')},indent=2))


if __name__=='__main__':{'plan':plan,'run':run}[sys.argv[1]]()
