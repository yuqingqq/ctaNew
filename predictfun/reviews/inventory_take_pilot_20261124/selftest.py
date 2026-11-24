import copy,resource
import numpy as np
import pilot as T
resource.setrlimit(resource.RLIMIT_CPU,(20,21))
def frame(t=0,recv=None,bid=40,ask=42,bq=100,aq=100):
    return [t,t if recv is None else recv,bid,-1,-1,-1,-1,bq,0,0,0,0,ask,-1,-1,-1,-1,aq,0,0,0,0]
def path(frames=None,trades=None):return dict(frames=np.array(frames if frames is not None else [frame()]),trades=np.array(trades if trades is not None else []).reshape(-1,4),mark=41.)
def eng(p=None):return T.IOCEngine(p or path(),1000.,200.)
def branch(p=None):return T.TakeBranch(p or path(),1000.,{},100.,200.,True)
def seed(B,q=150,s=0):
    gid=B.E.add9([s,0,40 if s==0 else 42,q,0,-100,-50,-110,0]);B.E.o[gid]['rem']=0
    B.E.fills.append((-100.,gid,s,q,40. if s==0 else 42.,-2));B.pre_tick(0)
groups=[]
def ok(name):groups.append(name)
assert T.requested_qty(99,0,0,100,100,50)==(None,0)
assert T.requested_qty(150,0,0,100,100,50)==(1,100)
assert T.requested_qty(-150,0,0,100,100,50)==(0,100)
assert T.requested_qty(150,0,340,100,100,50)==(1,10)
assert T.requested_qty(150,0,400,100,100,50)==(1,0);ok('both sides, threshold, target, reservation clipping')
E=eng();i=E.submit_ioc(0,0,60,42);E.advance(9);assert not E.fills;E.advance(10);assert E.fills==[(10.,i,0,60.,42.,-1)]
assert E.fees[0][2]==50.4;ok('effect latency, price, fee')
E=eng(path([frame(aq=30)]));i=E.submit_ioc(0,0,60,42);E.advance(10);assert E.exec_log[-1]['filled']==30
E.submit_ioc(50,0,60,42);E.advance(60);assert E.exec_log[-1]['filled']==0 and sum(f[3] for f in E.fills)==30;ok('partial and no reuse of same depth')
E=eng(path([frame(aq=30),frame(100,aq=50)]));E.submit_ioc(0,0,30,42);E.advance(10);E.submit_ioc(100,0,50,42);E.advance(110)
assert [r['filled'] for r in E.exec_log]==[30,20];ok('only net positive replenishment is reusable')
E=eng(path([frame()],[[5,1,42,80]]));E.submit_ioc(0,0,60,42);E.advance(10);assert E.exec_log[-1]['filled']==20;ok('public volume debit')
E=eng(path([frame(),frame(5,ask=43)]));E.submit_ioc(0,0,10,42);E.advance(10);assert E.exec_log[-1]['reason']=='limit_moved' and not E.fills;ok('limit refusal')
E=eng();E.add9([1,0,41,20,0,0,1000,0,0]);E.submit_ioc(0,0,10,42);E.advance(10);assert E.exec_log[-1]['reason']=='own_cross';ok('own-cross at effect')
B=branch();seed(B);B.take_step(0);assert B.reserve(0)==(150.,0.,100.) and len(B.take_open)==1
B.pre_tick(10);assert B.reserve(10)==(150.,0.,100.);assert B.E.exec_log[-1]['venue_after']==50
B.pre_tick(109);assert B.reserve(109)==(150.,0.,100.)
B.pre_tick(110);assert B.reserve(110)==(50.,0.,0.) and not B.take_open;B.check_position(110);ok('hidden fill and delayed result reservation')
B=branch();seed(B,s=1);B.take_step(0);B.pre_tick(10);assert B.E.exec_log[-1]['venue_after']==-50;ok('symmetric buy to reduce negative exposure')
B=branch(path([frame(0,recv=0),frame(500,recv=500,ask=80)]));seed(B);before=copy.deepcopy(B);B.take_step(0)
before.E.f[1,2:]=frame(500,recv=500,bid=10,ask=90)[2:];before.take_step(0)
assert B.take_decisions==before.take_decisions and B.E.iocs==before.E.iocs;ok('future-only frame cannot alter command')
B=branch(path([frame(bq=0)]));seed(B);B.take_step(0);B.pre_tick(110);B.take_step(150)
assert B.take_decisions[-1]['status']=='unchanged_state' and len(B.E.iocs)==1;ok('no fixed retry on unchanged state')
B=branch();seed(B);B.take_step(0);B.take_step(0);assert len(B.E.iocs)==1 and B.take_decisions[-1]['status']=='pending_ioc';ok('one pending IOC')
# A hidden opposite passive fill can invalidate reduction of the true position;
# the policy must still use known I, and the report must count the exception.
B=branch();seed(B);B.take_step(0);B.E.fills.append((5.,0,1,140.,42.,0));B.E.advance(10)
assert B.E.exec_log[-1]['venue_before']==10 and not B.E.exec_log[-1]['absolute_risk_reduced'];ok('known-risk vs realized-risk distinction')
E=eng(path([frame(-2000,recv=-2000)]));E.submit_ioc(0,0,10,42);E.advance(10);assert E.exec_log[-1]['reason']=='invalid_or_stale';ok('stale venue book refused')
f=[(0,0,0,10.,40.,0),(10,1,1,10.,42.,-1)]
v=T.X.fifo(f,41.);assert v['total_c']==20 and v['completed_c']==20 and v['residual_c']==0
assert v['total_c']-T.fee_c(42,10,200)==11.6;ok('whole book FIFO plus fee identity')
print('PASS',len(groups),'groups:',groups)
