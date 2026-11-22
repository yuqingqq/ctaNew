"""Small synthetic same-action fork parity and future-command causality controls."""
from pathlib import Path
import hashlib,json,sys
import numpy as np
PIN=Path('/tmp/pf-lc-review-6b806c19');OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(PIN/'predictfun/research'))
import pf_ladder_long_continuation as X
import pf_ladder_long_continuation_selftest as ST
end=6000.;rng=np.random.default_rng(7);path=X.BE.rand_tape(rng,70,160,end=end)
nt=int(np.ceil(end/X.BE.STEP))+2;v=np.array([-.5,0,.24,.47,1.],np.float32);cache=ST._cache(nt,rng.choice(v,nt),rng.choice(v,nt));edges=X.P.frozen_edges()
def make(p):
    B=X.AF.ProfileBranch(p,end,dict(end_rel=end),False,X.PROFILE);pol=X.LineageGuard(B,cache,edges);obs=X.P.QueueObs(B,pol);return B,pol,obs
B,p,o=make(path);X.LB.run_episode(B,p,observe=o);cm=X.P.command_arrays(B)['commands'];states=[]
for kind,mask in [('REVIEW',(cm[:,2]==0)&(cm[:,7]==2)&(cm[:,8]==1)),('ADMISSION',(cm[:,2]==1)&(cm[:,7]==0))]:
    r=cm[mask][0];states.append(dict(day='synthetic',m=1,kind=kind,side=int(r[1]),gid=int(r[3]),rung=float(r[4]),t=float(r[0]),key=kind))
def run(pth,st,action):
    b,p,o=make(pth);k=int(st['t']/X.BE.STEP);X.LB.run_episode(b,p,until=k,observe=o);ctx=(b,p,o)
    b,p,o=X.fork_copy(ctx);p.arm(st,action);X.LB.run_episode(b,p,k0=k,observe=o);r=X.arm_result(b,p,o,st,end);return b,p,r
sig=lambda v:hashlib.sha256(json.dumps(v,sort_keys=True,default=str).encode()).hexdigest()
same={}
for st in states:
    act=X.ACTIONS[st['kind']][0];b,p,r=run(path,st,act);b2,p2,r2=run(path,st,act)
    same[st['kind']]=bool(r['ledger']['fingerprint']==r2['ledger']['fingerprint'] and sig(b.cmd_log)==sig(b2.cmd_log) and r['post_raw']==r2['post_raw'] and r['Q']==r2['Q'])
assert all(same.values())
st=states[0];T=st['t']+1500.;q=dict(path,frames=path['frames'].copy(),trades=path['trades'].copy())
q['trades'][q['trades'][:,0]>T,3]*=2.;mask=q['frames'][:,0]>T;q['frames'][mask,7:12]*=1.5;q['frames'][mask,17:22]*=1.5
future={};changed=False
for act in X.ACTIONS['REVIEW']:
    b,p,r=run(path,st,act);b2,p2,r2=run(q,st,act)
    future[act]=bool(sig([c for c in b.cmd_log if c[0]<=T])==sig([c for c in b2.cmd_log if c[0]<=T]) and [f for f in b.E.fills if f[0]<=T]==[f for f in b2.E.fills if f[0]<=T])
    changed|=b.E.fills!=b2.E.fills
assert all(future.values()) and changed
out=dict(synthetic_only=True,same_action_parity=same,future_only_prefix_commands_and_fills=future,positive_control_changes_future_fills=changed)
(OUT/'controls.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
