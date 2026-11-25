"""Independent recomputation from saved fills/orders/commands; no policy helpers."""
import ast,collections,copy,gzip,json,math,resource,time
import numpy as np
def known(fills,t):
 lots=[];I=0.
 for ft,g,s,q,p,it in fills:
  if ft+100>t:break
  sign=1-2*s;I+=sign*q
  while q>1e-8 and lots and lots[0][0]!=sign:
   d=min(q,lots[0][1]);q-=d;lots[0][1]-=d
   if lots[0][1]<=1e-8:lots.pop(0)
  if q>1e-8:lots.append([sign,q,p])
 n=sum(l[1] for l in lots)
 return I,sum(l[1]*l[2] for l in lots)/n if n>1e-8 else None
def audit(rec,mode):
 fills=rec['fills'];orders={o['gid']:o for o in rec['orders']};cmd=rec['commands']
 post={c[3]:i for i,c in enumerate(cmd) if c[7]=='POST'};timed={};n=0
 for z in rec['exit_log']:
  t=z['t'];ci=z['command_index'];s=z['side'];q=z['q'];kind=z['kind'];p=z['price']
  assert 0<=ci<len(cmd) and cmd[ci][0]==t and cmd[ci][1]==s and cmd[ci][2]==kind and cmd[ci][5]==p
  if t not in timed:
   I,basis=known(fills,t);kf=collections.defaultdict(float)
   for ft,g,ss,qq,pp,it in fills:
    if ft+100<=t:kf[g]+=qq
   timed[t]=(I,basis,kf)
  I,basis,kf=timed[t];res=[0.,0.]
  for g,o in orders.items():
   if g not in post or post[g]>=ci:continue
   if o['act']<=t and (o['refused'] or kf[g]>=o['size']-1e-9 or (o['req'] is not None and o['remove']+100<=t)):continue
   res[o['side']]+=o['size']-kf[g]
  assert abs(z['I']-I)<1e-6 and abs(z['B']-res[0])<1e-6 and abs(z['A']-res[1])<1e-6
  assert z['basis']==basis or (z['basis'] is not None and basis is not None and abs(z['basis']-basis)<1e-7)
  own=0. if kind=='slot' else orders[z['gid']]['size']-kf[z['gid']]
  assert abs(z['own']-own)<1e-6
  if kind=='live':assert abs(q-own)<1e-6 and z['parent']==['PULL','pressure_pull']
  else:
   S=rec['settings']['inventory']['S'];C=rec['settings']['cap'];fac=min(1,max(0,1-I/S if s==0 else 1+I/S))
   proposed=math.floor(min(100*fac,max(0,200*fac-res[s]),max(0,C-I-res[s] if s==0 else C+I-res[s]))+1e-7)
   assert abs(q-proposed)<1e-6 and z['parent']==['WAIT',None,'pressure_withhold']
  room=max(0,(-I if s==0 else I)-max(0,res[s]-own));size=float(math.floor(min(q,room)+1e-8)) if kind=='slot' else q
  margin=None if basis is None else (basis-p if s==0 else p-basis)
  reducing=(I< -1e-8 if s==0 else I>1e-8) and size>1e-8 and size<=room+1e-8 and (kind!='slot' or size>=1)
  cost=margin is not None and margin>=1-1e-8;over=reducing and (mode=='reduce' or (mode=='cost' and cost))
  assert abs(z['room']-room)<1e-6 and abs(z['size']-size)<1e-6
  assert z['margin']==margin or (margin is not None and abs(z['margin']-margin)<1e-7)
  assert z['mode']==mode and z['reducing']==reducing and z['cost_ok']==cost and z['override']==over
  if over:
   assert cmd[ci][7:]==(['POST','exit_override'] if kind=='slot' else ['KEEP','exit_override'])
   if kind=='slot':assert cmd[ci][6]==size
  else:assert cmd[ci][7:]==(['WAIT','pressure_withhold'] if kind=='slot' else ['PULL','pressure_pull'])
  n+=1
 return n
def run(E):
 resource.setrlimit(resource.RLIMIT_CPU,(100,101));E.bound();start=time.process_time()
 summary=json.loads((E.OUT/'summary.json').read_text());man=json.loads((E.H/'manifest.json').read_text())
 expected={(d,m,a) for d,m,p in man['cases'] for a in E.ARMS};assert {(r['day'],r['mid'],r['arm']) for r in summary['rows']}==expected and len(summary['rows'])==len(expected)
 # Reuse the accepted independent ledger verifier, with its original config mapping.
 tr=ast.parse((E.CAP.H/'check.py').read_text());node=next(n for n in tr.body if isinstance(n,ast.FunctionDef) and n.name=='verify')
 ns=dict(np=np,collections=collections,math=math,E=E.CAP);exec(compile(ast.Module(body=[node],type_ignores=[]),'<accepted ledger verifier>','exec'),ns)
 output=[];calls=0;bad=0;sample=None
 for row in summary['rows']:
  path=E.OUT/f"{row['mid']}_{row['arm']}.json.gz";assert E.sha(path)==row['sha256'];rec=json.loads(gzip.decompress(path.read_bytes()))
  assert rec['manifest_sha256']==E.sha(E.H/'manifest.json') and rec['metrics']==row['metrics'] and rec['mid']==row['mid'] and rec['day']==row['day'] and rec['arm']==row['arm']
  ns['verify'](rec,'BASE' if rec['arm']=='BASE' else 'WIDE_PASSIVE');calls+=audit(rec,E.ARMS[rec['arm']][1])
  counts=collections.Counter((z['kind'],z['side']) for z in rec['exit_log'] if z['override']);affected={rec['commands'][z['command_index']][3] for z in rec['exit_log'] if z['override']}
  out=dict(day=row['day'],mid=row['mid'],arm=row['arm'],**rec['metrics'],overrides=sum(counts.values()),
   override_posts=sum(v for (k,s),v in counts.items() if k=='slot'),override_keeps=sum(v for (k,s),v in counts.items() if k=='live'),
   override_ask=sum(v for (k,s),v in counts.items() if s==1),override_bid=sum(v for (k,s),v in counts.items() if s==0),
   affected_gid_fills=sum(f[3] for f in rec['fills'] if f[1] in affected),pressure_decisions=len(rec['exit_log']))
  output.append(out)
  if sample is None and any(z['override'] for z in rec['exit_log']):sample=rec
 assert len(summary['parity'])==6 and all(all(x['same'].values()) for x in summary['parity'])
 assert sample is not None,'positive real exemption control absent'
 j=next(i for i,z in enumerate(sample['exit_log']) if z['override'])
 for field,delta in [('basis',1),('I',1),('A',1),('room',1),('override',None),('size',1),('margin',1)]:
  x=copy.deepcopy(sample)
  if delta is None:x['exit_log'][j][field]=not x['exit_log'][j][field]
  else:x['exit_log'][j][field]+=delta
  try:audit(x,E.ARMS[x['arm']][1])
  except AssertionError:bad+=1
  else:raise AssertionError('corruption escaped '+field)
 for field in ['net_value_c','cash']:
  x=copy.deepcopy(sample);x[field]+=1
  try:ns['verify'](x,'WIDE_PASSIVE')
  except AssertionError:bad+=1
  else:raise AssertionError('ledger corruption escaped')
 E.dump(E.H/'checks.json',dict(pass_all=True,records=len(output),pressure_decisions_checked=calls,prior_parity=6,corruptions_refused=bad,cpu_s=time.process_time()-start,rows=output))
 print('PASS',len(output),'ledgers',calls,'pressure decisions',bad,'corruptions',round(time.process_time()-start,2),'CPU s')
