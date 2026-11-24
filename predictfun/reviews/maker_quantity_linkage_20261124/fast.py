"""v2 performance adapter: cache repeated pandas columns; inference unchanged.
The original frozen modules and v1 failed resource gate stay intact. Dynamic
source transforms are exact, asserted, and both source bytes are bound.
"""
import inspect,json,sys,time,resource,subprocess
import build as B
H=B.HERE;DATA=B.DATA/'v2'
def install():
 src=inspect.getsource(B.LC._book)
 old='    pend = {}\n';new='    pend = {}\n    cached_pb = frames.pend_bids; cached_pa = frames.pend_asks\n'
 assert src.count(old)==1;src=src.replace(old,new)
 for old,new in [('frames.pend_bids.iat[i]','cached_pb.iat[i]'),('frames.pend_asks.iat[i]','cached_pa.iat[i]')]:assert src.count(old)==1;src=src.replace(old,new)
 ns=dict(B.LC.__dict__);exec(compile(src,'<bound fast _book>','exec'),ns);B.LC._book=ns['_book']
 src=inspect.getsource(B.normalized_book);old="bad=np.zeros(len(t),bool)";assert src.count(old)==1;src=src.replace(old,"bad=np.zeros(len(t),bool);cached_recv=fm.recv_ns.to_numpy(np.int64)")
 assert src.count('fm.recv_ns.iat[i]')==1;src=src.replace('fm.recv_ns.iat[i]','cached_recv[i]')
 ns=dict(B.__dict__);exec(compile(src,'<bound fast normalized_book>','exec'),ns);B.normalized_book=ns['normalized_book']
def main():
 start=time.process_time();resource.setrlimit(resource.RLIMIT_CPU,(16,18));B.bind()
 for p,h in json.loads((H/'fast_sources.json').read_text()).items():B.need(B.sha(B.ROOT/p)==h,'v2 source')
 install();inputs=B.input_tables();pairs=json.loads((H/'inputs.json').read_text())['pairs'];DATA.mkdir(exist_ok=True)
 if sys.argv[1]=='pilot':
  B.need(not (DATA/'pilot.json').exists(),'write once');p=B.clean(B.market(*pairs[0],inputs,DATA));old=json.loads((B.DATA/'pilot.json').read_text());a=json.loads(json.dumps(p));z=json.loads(json.dumps(old))
  for x in (a,z):
   for k in ('normalized_file','normalized_sha256','cpu_seconds'):x['normalization'].pop(k)
  B.need(a==z,'pilot inferred outputs parity')
  B.need(B.rows(B.ROOT/p['normalization']['normalized_file'])==B.rows(B.ROOT/old['normalization']['normalized_file']),'pilot normalized values parity')
  B.dump(DATA/'pilot.json',p);frames=[x for x in json.loads((H/'inputs.json').read_text())['parquets'] if x['kind']=='frames'];projection=(time.process_time()-start)+p['normalization']['cpu_seconds']*sum(x['rows'] for x in frames)/max(1,frames[0]['rows'])*2+8
  gate=dict(projected_remaining_all_in_cpu_s=projection,cap_cpu_s=54,run_full=projection<45,pilot_sha=B.sha(DATA/'pilot.json'),cpu_seconds=time.process_time()-start,pilot_content_equal=True,normalized_values_equal=True)
  B.dump(H/'pilot_gate_v2.json',gate);print(json.dumps(gate,indent=2));return
 gate=json.loads((H/'pilot_gate_v2.json').read_text());B.need(gate['run_full'] and B.sha(DATA/'pilot.json')==gate['pilot_sha'],'gate');out=H/'outputs';B.need(not out.exists(),'write once outputs');parts=[json.loads((DATA/'pilot.json').read_text())]
 for d,m in pairs[1:]:parts.append(B.clean(B.market(d,m,inputs,DATA)));B.guard()
 out.mkdir();names={'ledger':'order_ledger.csv','orders':'order_summary.csv','bounds':'chain_bounds.csv','relations':'relation_edges.csv.gz','book':'book_edges.csv.gz','joint':'joint_conflicts.csv'}
 for field,name in names.items():B.write(out/name,[r for p in parts for r in p[field]],['day','mid','fid','t_ms','side','tick','required_future_shares','capacity_upper','order_hashes'] if field=='joint' else None)
 B.dump(out/'normalization.json',[p['normalization'] for p in parts]);result=B.summarize(parts);result.update(as_of_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=B.ROOT,text=True).strip(),cpu_seconds=time.process_time()-start,hashes={p.name:B.sha(p) for p in sorted(out.iterdir())});B.dump(out/'result.json',result);print(json.dumps(result,indent=2))
if __name__=='__main__':main()
