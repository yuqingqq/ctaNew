"""Clock, matching and accounting controls. Run without opening any market data."""
import numpy as np
from run import H, events, cancel_live, valued, exposures


def main():
    k=np.array([-300,0,12,40,80])
    mid=np.array([100.,100.,101.,100.,99.])
    ticks=np.array([0,12,13,40,50,80,90,100])
    flags=({0:np.array([False,False,True]),1:np.array([False,False,False])},None)
    edges={0:[.47],1:[.28]}
    assert events(ticks,k,mid,0,101,edges,flags)==[(12,1),(13,1),(80,0),(90,0)]
    bad=({0:flags[0][0],1:~flags[0][1]},None)
    try:
        events(ticks,k,mid,0,101,edges,bad)
    except AssertionError:
        pass
    else:
        raise ValueError('future / wrong grid accepted')
    f=np.array([[0.,0.,40,39,38,37,36,0,0,0,0,0,60,61,62,63,64,0,0,0,0,0]])
    tr=np.array([[t,1,60,1] for t in (72.,90.,108.,109.,110.)])
    path=dict(frames=f,trades=tr,mark=50.)
    b=H.U.UrgencyBranch(path,2000,{'opening_ms':0,'start_abs_ms':0})
    for side,price in [(0,40),(1,60)]:
        gid=b.E.add_order(side,price,0,50,0)
        b.unclosed[side].append(gid)
    assert not cancel_live(b,50,1)
    assert cancel_live(b,72,1)==[(1,72,1)]
    assert not cancel_live(b,80,1)
    assert b.E.o[0]['req'] is None
    b.E.advance(200)
    assert [r[0] for r in b.E.fills]==[72,90,108]
    pnl=valued(exposures(b.E.fills,path),path,2000,0)
    assert pnl['shares']==3 and np.isclose(pnl['favorable'],.30)
    assert valued(exposures(b.E.fills,path),path,1000,0)['censored']==3
    print('PASS: side, tie, reversal, grid parity falsifier, activation, duplicate, pending match, censor')

if __name__=='__main__':
    main()
