import copy,types
import numpy as np
def run(E):
 def view(I,B=0,A=0,s=1):return dict(position=I,B=B,A=A,side=s)
 def f(v,p=42,q=100,own=0,b=40,kind='slot',mode='cost'):return E.eligibility(v,p,q,own,b,kind,mode)
 assert f(view(120))[0:2]==(100.,120.) and f(view(120))[3]
 assert not f(view(120),p=40.9)[3] and f(view(120),p=41)[3]
 assert f(view(120,A=100))[0]==20 and f(view(120,A=100))[3]
 assert not f(view(120,A=150))[3] and not f(view(0))[3] and not f(view(-20))[3]
 assert f(view(-120,s=0),p=38)[3] and not f(view(-120,s=0),p=40)[3]
 assert f(view(20,A=120),q=100,own=100,kind='live')[3] is False
 assert f(view(20,A=20),q=20,own=20,kind='live')[3]
 assert f(view(.5,A=.5),q=.5,own=.5,kind='live')[3]
 assert not f(view(.5),q=100)[3]
 assert f(view(120),p=10,mode='reduce')[3] and not f(view(120),mode='none')[3]
 assert not f(view(120),b=None)[3]
 fills=[[0,0,0,100,40,0],[20,1,0,100,50,1],[80,2,1,120,60,2],[1000,3,1,500,1,3]]
 k=E.KnownLots();assert k.view(fills,99)==(0,None)
 assert k.view(fills,100)==(100,40)
 assert k.view(fills,120)==(200,45)
 assert k.view(fills,180)==(80,50)
 future=copy.deepcopy(fills);future[-1]=[1000,3,0,999,99,3]
 assert E.KnownLots().view(future,180)==E.KnownLots().view(fills,180)
 assert E.KnownLots().view(future,1100)!=E.KnownLots().view(fills,1100)
 assert E.KnownLots().view([[0,0,1,100,60,0],[1,1,0,20,50,1]],200)==(-80,60)
 # Real callback fixtures use parent pressure/opening/inventory policies.
 frame=[0,0,40,-1,-1,-1,-1,100,0,0,0,0,42,-1,-1,-1,-1,100,0,0,0,0]
 path=dict(frames=np.array([frame],float),trades=np.empty((0,4)),mark=41.)
 b=E.CAP.Branch(path,1000,{},200,E.ARMS['WIDE'][0]);b.E.fills=[[0,0,0,100,40,0]]
 cache=dict(x={s:np.full(20,10.) for s in [0,1]},status={s:np.zeros(20,int) for s in [0,1]},age={s:np.zeros(20) for s in [0,1]})
 pol=E.ExitPolicy(b,cache,{s:[.1,.2,.3,.4] for s in [0,1]},mode='cost')
 v=dict(t=100.,side=1,position=100.,B=0.,A=0.,mode='UNRESTRICTED')
 assert pol.slot(v,0,42)==('POST',100.,'exit_override')
 v['A']=100.;assert pol.slot(v,1,43)[0]=='WAIT','including retiring reservations'
 v['A']=0.;assert pol.slot(v,0,40)[0]=='WAIT'
 cache['x'][1][:]=0;assert pol.slot(v,0,42)==('POST',100.,'post')
 # Opening gate with one preexisting order: ordinary post delegates parent.
 cache['x'][1][:]=10
 o=dict(gid=0,rung=0,size=100,known_filled=0,price=42)
 v['A']=100.;assert pol.live(v,o)==('KEEP','exit_override')
 v['mode']='SHORT_RESTRICTED';assert pol.live(v,o)==('PULL','inventory_pull')
 print('PASS: known FIFO, notice boundary, hidden/delivered fills, long/short, cost threshold, full remaining, reservations, parent callbacks and precedence')
