"""Re-run the frozen scorer on its complete written rows after its JSON writer refused a NumPy boolean."""
from role import R,S,HERE,BASE,gate
import math
import pandas as pd


def finish():
    df=pd.read_csv(HERE/'markets.csv',float_precision='round_trip')
    tr=pd.read_csv(HERE/'replacement.csv',float_precision='round_trip')
    previous=S.read(BASE/'result.json')
    old=pd.read_csv(BASE/'markets.csv')  # match the original runner's control loader
    assert R.sha(BASE/'markets.csv')==previous['market_hash']
    control=df[df.arm!='role'].reset_index(drop=True)
    pd.testing.assert_frame_equal(control,old,check_dtype=False,check_exact=True)
    candidate=df.query("arm=='role'").set_index('mid')
    assert candidate.index.is_unique and set(candidate.index)==set(old.mid)==set(tr.mid)
    assert len(df)==210 and len(tr)==len(candidate)==70
    reg=R.MM.registry(R.EVAL).set_index('mid')
    parity=[min(m for m in g.index if m in candidate.index) for _,g in reg.loc[sorted(candidate.index)].groupby('hour')]
    arms={}
    for arm,g in df.groupby('arm'):
        arms[arm]={k:float(g[k].sum()) for k in g.select_dtypes('number') if k not in ('mid','abs_end','end_pos','both')}
        arms[arm].update(mean_abs_end=g.abs_end.mean(),both=g.both.mean(),within_ten=int((g.abs_end<=10).sum()),
                         positive_markets=int((g.settle>0).sum()),c_per_matched=100*g.turnover.sum()/g.closed.sum())
    for arm in ('reference','retain'):
        assert all(math.isclose(arms[arm][k],v,abs_tol=1e-6) for k,v in previous['arms'][arm].items())
    tests=gate(arms['reference'],arms['role'])
    contrasts={}
    p=df.pivot(index='mid',columns='arm',values='settle')
    for arm in ('reference','retain'):
        delta=p.role-p[arm]
        contrasts[arm]=dict(delta=delta.sum(),better=int((delta>1e-6).sum()),worse=int((delta< -1e-6).sum()),
                           equal=int((delta.abs()<=1e-6).sum()),excluding_largest_gain=delta.sum()-delta.max())
    R.write(HERE/'result_v2.json',dict(arms=arms,criteria={k:bool(v) for k,v in tests.items()},pass_all=all(tests.values()),paired=contrasts,
          ceiling=arms['reference']['mean_abs_end']+10,trace=tr.drop(columns='mid').sum().to_dict(),
          source=R.sha(HERE/'role.py'),controls=previous['market_hash'],parity=parity,market_hash=R.sha(HERE/'markets.csv'),
          replacement_hash=R.sha(HERE/'replacement.csv')))

if __name__=="__main__":
    finish()
