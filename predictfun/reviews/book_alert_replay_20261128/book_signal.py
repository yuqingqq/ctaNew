"""Past-only transfer of PUBLIC_SCORE's ten best-level inputs and logistic model."""
from pathlib import Path
import hashlib
import json
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
DATA=ROOT/'data/predict_fun/research'
OUT=DATA/'book_alert_replay_20261128'
SOURCE=HERE.parent/'take_reprice_20261125'
RAW=ROOT/'data/mm_hf/raw/bookTicker/BTCUSDT'
TRAIN=('20261023','20261031','20261101','20261102')
DAYS=('20261124','20261125','20261126')
POINTS=DATA/'take_reprice_20261125/public_score_points.parquet'
INPUT_HASH='993fe91ed0748a47ca3c239dfe630427bf2a5dce5785091d604d5a115cdc3676'
FEATS='b_imb1 b_lfar1 b_lnear1 b_spread b_rfar b_rnear b_lchf b_lchn b_lupd b_against'.split()
DELAY=4
FRESH=300


def digest(path):
    return hashlib.file_digest(path.open('rb'),'sha256').hexdigest()


def raw_hour(stem,expected=None):
    path=RAW/(stem+'.csv.gz')
    if expected:
        assert digest(path)==expected
    cols=['recv','E','T','id','bid','bq','ask','aq']
    f=pd.read_csv(path,names=cols,usecols=['recv','E','bid','bq','ask','aq'])
    # Use the archived model's exchange-ms clock plus delivery delay; never local wall time.
    f=f[(f.recv>=1787579334881534478)&(f.bid>0)&(f.ask>f.bid)&np.isfinite(f.ask)]
    return f.drop(columns='recv').sort_values('E',kind='stable').drop_duplicates('E',keep='last')


def features(st,t,d):
    """Exact frozen best-level transforms at or before the exchange-time cutoff."""
    import public_score as ps
    dirs=np.full(len(t),d)
    h=ps.QB.hour_feats(st,t,dirs)
    prior,against=ps.QL.moves_flags(st,t,dirs)
    j=st[0].searchsorted(t,side='right')-1
    far,near=h['far1'],h['near1']
    return np.column_stack(((near-far)/(near+far),np.log(far),np.log(near),
                            np.rint((st[3][j]-st[1][j])/.1),
                            np.log(far/h['far0']),np.log(near/h['near0']),
                            np.log1p(h['churn_far']),np.log1p(h['churn_near']),
                            np.log1p(h['updates']),against)),prior


def arrays(f):
    return tuple(f[c].to_numpy() for c in ('E','bid','bq','ask','aq'))


def linear(x,m):
    x=np.clip(x,np.asarray(m['lo']),np.asarray(m['hi']))
    y=(x-np.asarray(m['mean']))/np.asarray(m['sd'])
    return y@np.asarray(m['coef'])+m['intercept']


def fit():
    import public_score as ps
    assert digest(POINTS)==INPUT_HASH and FEATS==ps.FAMILIES['best']
    z=pd.read_parquet(POINTS,columns=['day','kind','y','s_logit_best','t','d',*FEATS])
    assert set(z.day.astype(str))==set(TRAIN)
    valid=z.kind.ne('fire')
    x=z.loc[valid,FEATS].to_numpy(float)
    assert np.isfinite(x).all()
    a,_=ps.design(x,x[:2])
    model=LogisticRegression(C=1.,max_iter=3000).fit(a,z.loc[valid,'y'])
    lo,hi=np.quantile(x,[.005,.995],axis=0)
    clipped=np.clip(x,lo,hi)
    sd=clipped.std(0)
    sd[sd==0]=1
    threshold=np.quantile(z.loc[z.kind=='random','s_logit_best'],.99)
    m=dict(lo=lo.tolist(),hi=hi.tolist(),mean=clipped.mean(0).tolist(),sd=sd.tolist(),
           coef=model.coef_[0].tolist(),intercept=float(model.intercept_[0]),probability=float(threshold),
           cutoff=float(np.log(threshold/(1-threshold))),train_rows=len(x),input_hash=INPUT_HASH)
    np.testing.assert_allclose(linear(x,m),model.decision_function(a),atol=1e-10,rtol=0)
    # Rebuild both directions at historical points from one raw hour.
    st=arrays(raw_hour('20261101_12'))
    for direction in (1,-1):
        p=z[(z.d==direction)&z.t.between(st[0][0]+2000,st[0][-1]-2000)].iloc[::5]
        xx,prior=features(st,p.t.to_numpy(),direction)
        assert len(p)>15 and not prior.any()
        np.testing.assert_allclose(xx,p[FEATS],atol=1e-7,rtol=0)
        m[f'parity_{direction}']=len(p)
    (HERE/'model.json').write_text(json.dumps(m,indent=2)+'\n')
    print('fit and historical feature parity',m['train_rows'],threshold,flush=True)


def build(day):
    import public_score as ps
    m=json.loads((HERE/'model.json').read_text())
    base=DATA/'event_pressure_nov24_26'
    docs=json.loads((base/f'grid_{day}.json').read_text())['inputs']
    assert len(docs)==25 and DELAY==4
    chunks=[raw_hour(stem,docs[stem]) for stem in sorted(docs)]
    f=pd.concat(chunks).sort_values('E',kind='stable').drop_duplicates('E',keep='last')
    del chunks
    assert f.E.is_monotonic_increasing and (f.bq.gt(0)&f.aq.gt(0)).all()
    st=arrays(f)
    t=st[0]
    mask=np.zeros(len(t),dtype=np.uint8)
    # Process contiguous batches. A preceding full 300 ms observation window is required.
    for ix in range(t.searchsorted(t[0]+300),len(t),90000):
        end=min(len(t),ix+90000)
        cut=max(0,t.searchsorted(t[ix]-300,side='right')-1)
        sub=tuple(a[cut:end] for a in st)
        ts=t[ix:end]
        for side,direction in ((0,-1),(1,1)):
            x,prior=features(sub,ts,direction)
            on=(~prior)&(linear(x,m)>=m['cutoff'])
            mask[ix:end]|=on.astype(np.uint8)<<side
            # Positive/negative tie-clock cases: a future book suffix cannot affect a prefix.
            if ix==t.searchsorted(t[0]+300):
                stop=31
                short=tuple(a[:ix-cut+stop] for a in sub)
                xx,pp=features(short,ts[:stop],direction)
                np.testing.assert_allclose(xx,x[:stop],atol=1e-10,rtol=0)
                assert np.array_equal(pp,prior[:stop])
    dest=OUT/day
    dest.mkdir(parents=True,exist_ok=True)
    np.savez(dest/'alerts.npz',clock=t+DELAY,mask=mask)
    print(day,'clock rows',len(t),'directional update counts',np.bincount(mask,minlength=4),flush=True)

if __name__=='__main__':
    sys.path.insert(0,str(SOURCE))
    if sys.argv[1]=='fit':
        fit()
    else:
        assert sys.argv[1] in DAYS
        build(sys.argv[1])
