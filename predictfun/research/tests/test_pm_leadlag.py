import numpy as np
import pandas as pd
import pytest
from predictfun.research import pf_pm_leadlag_data as d
from predictfun.research import pf_pm_leadlag_stats as s


def test_causal_sampling_future_and_bad_book():
    a=np.array([[100,90,40,42],[200,190,60,62],[300,290,70,60]],float)
    z=d.sample(a,np.array([99,100,199,200,300]))
    assert np.isnan(z[0,0]) and list(z[1:4,0])==[41,41,61] and np.isnan(z[4,0])
    with pytest.raises(ValueError):d.sample(a[::-1],np.array([200]))
    a[0,0]=np.nan
    with pytest.raises(ValueError):d.sample(a,np.array([200]))


def test_market_and_token_identity():
    assert d.up_identity(np.int64(1)) and d.up_identity([1])
    assert not any(d.up_identity(x) for x in ([1,2],[0],True,'1',1.0))
    row=dict(slug='btc-updown-5m-300',window_start=300,window_end=600,recv_ns=299*10**9,
             outcomes=['Up','Down'],clobTokenIds=['up','down'])
    assert d.metadata([row],{row['slug']})[row['slug']]['up_token']=='up'
    with pytest.raises(ValueError):d.metadata([dict(row,outcomes=['Down','Up'])],{row['slug']})
    assert d.metadata([dict(row,recv_ns=301*10**9)],{row['slug']})=={}
    with pytest.raises(ValueError):d.day_ns('20261110')


def test_published_bbo_and_sentinels():
    e=dict(event_type='price_change',market='condition',timestamp='1000',price_changes=[dict(asset_id='up',best_bid='0.4',best_ask='0.42')])
    assert d.pm_events(e,'up','condition')==[(10**9,40.,42.)]
    assert d.pm_events(e,'down','condition')==[]
    with pytest.raises(ValueError):d.pm_events(e,'up','wrong')
    e['price_changes'][0].update(best_bid='0',best_ask='1')
    assert np.isnan(d.pm_events(e,'up','condition')[0][1:]).all()


def test_lag_orientation_planted_and_reversed():
    rng=np.random.default_rng(32);x=rng.normal(size=1000);y=np.r_[np.full(5,np.nan),x[:-5]]
    c=s.correlations(s.lag_stats(x,y));rev=s.correlations(s.lag_stats(y,x))
    assert d.LAGS[c.argmax()]==5 and c.max()>0.999
    assert d.LAGS[rev.argmax()]==-5 and rev.max()>0.999
    with pytest.raises(ValueError):s.pairs(x,y,1000)
    with pytest.raises(ValueError):s.shifted_null([],[],199)


def test_changes_do_not_wrap_windows():
    assert np.isnan(s.change(np.array([3.,5.,7.]))[0])
    assert s.change(np.array([3.,5.,7.]))[1:].tolist()==[2,2]
    with pytest.raises(ValueError):s.change(np.arange(5),0)


def test_chronological_model_and_bad_split():
    tr=pd.DataFrame(dict(day=['20261023']*400,b0=np.linspace(-1,1,400)))
    tr['target']=2*tr.b0
    te=pd.DataFrame(dict(day=['20261024']*50,b0=np.linspace(-1,1,50)))
    assert np.max(abs(s.predict(tr,te,['b0'])-2*te.b0))<.02
    with pytest.raises(ValueError):s.predict(te.assign(target=0),tr,['b0'])


def test_shift_null_positive_lead():
    rng=np.random.default_rng(92);segments=[];observed=[]
    for phase,interval in enumerate(d.PHASES):
        x=rng.normal(size=550);y=np.r_[np.full(5,np.nan),x[:-5]]
        c=s.correlations(s.lag_stats(x,y));segments.append((phase,x,y))
        observed.append(dict(clock='recv',max_age_ms=1000,phase=list(interval),positive_lag_asymmetry=s.directional(c).tolist()))
    result=s.shifted_null(segments,observed,200)
    index=list(d.LAGS[d.LAGS>0]).index(5)
    assert all(r['family_p'][index]<.01 for r in result['results'])
