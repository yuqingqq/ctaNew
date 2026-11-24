import copy
import numpy as np
import experiment as E
import observe as A
T=E.T
def sizing(C,S,I,B,A_,s):
 p=object.__new__(E.LB.LadderInventory);p.cap=C;p.S=S;p.q0=100.;p.Qside=200.
 return p.sizing(dict(side=s,position=I,B=B,A=A_))[-1]
for I in range(-200,201,10):
 for B in range(0,401,20):
  for s in [0,1]:assert sizing(200,100,I,B,B,s)==sizing(1000,100,I,B,B,s)
assert sizing(200,100,100,0,0,0)==0 and sizing(1000,500,100,0,0,0)==80
assert E.qty(499,0,0,1000,500,250)==(None,0)
assert E.qty(550,0,0,1000,500,250)==(1,100)
assert E.qty(-550,0,0,1000,500,250)==(0,100)
assert E.qty(550,0,1540,1000,500,250)==(1,10)
assert E.qty(550,0,1700,1000,500,250)==(1,0)
assert A.inventory_class(-10)=='reducing' and A.inventory_class(10)=='adding'
assert A.label([1,np.nan],str)=='unknown' and A.label([-2,2],A.inventory_class)=='adding|reducing'
assert np.isnan(A.sqrms([1]*9)).all() and A.sqrms([1]*10)[-1]==1
z=A.sqrms([1]*10+[100]);assert z[9]==1 and z[10]>1
# Known-bad size/clock/sign substitutions must disagree, not silently pass.
assert sizing(1000,500,100,0,0,0)!=sizing(200,100,100,0,0,0)
assert E.qty(550,0,1540,1000,500,250)!=T.requested_qty(550,0,1540,100,500,250)
assert A.label([-1],A.inventory_class)!=A.label([1],A.inventory_class)
# Real branch ensures config is installed before the policy caches sizing fields.
f=[0,0,40,-1,-1,-1,-1,100,0,0,0,0,42,-1,-1,-1,-1,100,0,0,0,0]
p=dict(frames=np.array([f],float),trades=np.empty((0,4)),mark=41.)
b=E.Branch(p,1000,{},200,E.ARMS['WIDE_TAKE']);pol=E.LB.LadderInventory(b)
assert (pol.cap,pol.S,pol.Qside,b.inv['R'])==(1000,500,200,250)
assert b.E.fills==[] and b.E.o==[]
print('PASS cap-only sizing, independent wide-band positive control, cap-aware take room, signs, missing/future RMS, configuration integration')
