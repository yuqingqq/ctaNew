"""Strict array boundary for the conserved-volume branch kernel."""
import ctypes
import subprocess
import pf_flow_control_common as F
C,np=F.C,F.np
_LIB=None


def library():
    global _LIB
    if _LIB is None:
        src=F.HERE/'pf_flow_control_kernel.cpp';dest=F.OUT/'build'/('kernel_'+C.sha(src)[:16]+'.so')
        dest.parent.mkdir(parents=True,exist_ok=True)
        if not dest.exists():subprocess.run(['g++','-O3','-std=c++17','-shared','-fPIC',str(src),'-o',str(dest)],check=True)
        _LIB=ctypes.CDLL(str(dest));d=ctypes.POINTER(ctypes.c_double);i=ctypes.POINTER(ctypes.c_longlong)
        _LIB.evaluate.argtypes=[ctypes.c_int]*3+[i,i]+[d]*6+[ctypes.c_double]*7+[d,d]
        _LIB.evaluate.restype=ctypes.c_int
    return _LIB


def pack(paths,bid,horizon):
    frames=[p['frames'] for p in paths];trades=[p['trades'] for p in paths]
    C.require(horizon in (1000.,2000.),'undeclared horizon')
    for f,t in zip(frames,trades):
        C.require(f.ndim==2 and f.shape[1]==22 and len(f)>0 and t.ndim==2 and t.shape[1]==4,'path schema')
        C.require((np.diff(f[:,0])>=0).all() and (np.diff(f[:,1])>=0).all() and
                  (np.diff(t[:,0])>=0).all() and (t[:,0]>0).all() and (t[:,3]>=0).all(),'path chronology/volume')
    fo=np.r_[0,np.cumsum([len(x) for x in frames])].astype(np.int64)
    to=np.r_[0,np.cumsum([len(x) for x in trades])].astype(np.int64)
    f=np.ascontiguousarray(np.concatenate(frames),float);t=np.ascontiguousarray(np.concatenate(trades),float)
    delta=np.array([bid-p['bid'] for p in paths],float)
    # The initial received frame is absolute/current; later template frames translate.
    for j,d in zip(fo[:-1],delta):
        for cols in (slice(2,7),slice(12,17)):
            p=f[j,cols];f[j,cols]=np.where(p>=0,p+d,p)
    mark=np.array([p['marks'][int(horizon==2000.)]+d for p,d in zip(paths,delta)],float)
    C.require(np.isfinite(f).all() and np.isfinite(t).all() and ((mark>=0)&(mark<=100)).all(),'invalid path')
    return fo,to,f,t,delta,mark


def current_book(packed,prices,sizes):
    """Condition every path on the actual known starting depth, not its donor depth."""
    fo,to,f,t,delta,mark=packed
    for side in (0,1):
        p=np.asarray(prices[side],float);q=np.asarray(sizes[side],float)
        C.require(p.shape==(5,) and q.shape==(5,) and np.isfinite(p).all() and np.isfinite(q).all(),'invalid current depth')
        f[fo[:-1],2+side*10:7+side*10]=p
        f[fo[:-1],7+side*10:12+side*10]=q
    return packed


def evaluate(packed,orders,actions,position,mid,horizon=F.HORIZON):
    orders=np.ascontiguousarray(orders,float).reshape(-1,9)
    actions=np.ascontiguousarray(actions,float).reshape(-1,len(orders))
    C.require(len(orders)>0 and np.isfinite(orders).all() and np.isfinite(actions).all(),'invalid branch state')
    C.require(np.isin(actions,[0,1,2]).all() and (orders[:,3]>=0).all() and
              np.isin(orders[:,0],[0,1]).all() and ((orders[:,2]>=0)&(orders[:,2]<=100)).all(),'invalid branch action')
    fo,to,f,t,delta,mark=packed;ns=len(mark);nb=len(actions);no=len(orders)
    out=np.zeros((ns,nb,6));qty=np.zeros((ns,nb,no))
    dp=lambda a:a.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    ip=lambda a:a.ctypes.data_as(ctypes.POINTER(ctypes.c_longlong))
    rc=library().evaluate(ns,nb,no,ip(fo),ip(to),dp(f),dp(t),dp(delta),dp(mark),dp(orders),dp(actions),
        float(position),float(mid),float(horizon),58.,37.,95.,3.,dp(out),dp(qty))
    C.require(rc==0,'branch kernel refused '+str(rc))
    C.require(np.isfinite(out).all() and np.isfinite(qty).all() and (qty>=-1e-9).all()
              and (qty<=orders[None,None,:,3]+1e-7).all(),'branch quantity bounds')
    sign=1-2*orders[:,0]
    C.require(np.allclose(out[:,:,0],(qty*sign*(mark[:,None,None]-orders[None,None,:,2])).sum(2)+position*(mark[:,None]-mid),atol=1e-7),'branch cash/position/value reconciliation')
    C.require(np.allclose(out[:,:,2],position+(qty*sign).sum(2),atol=1e-7),'branch position reconciliation')
    return out,qty
