"""Verify market membership and paired accounting, then summarize the frozen experiment."""
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from controller_module import controller

ARMS, CENSUS, EVAL, HERE, OUT = controller.ARMS, controller.CENSUS, controller.EVAL, controller.HERE, controller.OUT
forecast, sha, write = controller.forecast, controller.sha, controller.write


def aggregate(rows):
    a=np.asarray(rows,dtype=float).reshape(-1,5)
    cap,actual,level,exposure,unit=a.T
    residual=np.maximum(0,exposure-actual)*unit
    positive=cap>1e-8
    return dict(n=len(cap), forecast_positive=int(positive.sum()), covered=int((actual+1e-7>=cap).sum()),
                covered_positive=int((actual[positive]+1e-7>=cap[positive]).sum()), unsupported=int((level==3).sum()),
                mean_forecast=float(cap.mean()),mean_actual=float(actual.mean()),
                positive_mean_forecast=float(cap[positive].mean()) if positive.any() else None,
                positive_mean_actual=float(actual[positive].mean()) if positive.any() else None,
                all_fill_stress_over_50=int((residual>50+1e-7).sum()))


def main():
    fit=json.loads((HERE/'fit.json').read_text())
    assert sha(CENSUS)==fit['census_sha']
    for name,h in fit['inputs'].items():
        assert sha(OUT/'train'/name)==h
    cen=[r for r in json.loads(CENSUS.read_text())['rows'] if r['day']==EVAL]
    expected={r['mid'] for r in cen if r['status']=='complete'}
    rows=[]; anchors={a:[] for a in ARMS}; props=[]; drift={}; inputs={}
    for p in sorted((OUT/'eval').glob('*.json')):
        d=json.loads(p.read_text())
        assert d['day']==EVAL and d['source']==sha(HERE/'replay.py') and set(d['arms'])==set(ARMS)
        inputs[p.name]=sha(p)
        opened=[]
        for a,v in d['arms'].items():
            m=v['metrics']
            opened.append(m['first_s'])
            assert abs(m['settle']-m['turnover']-m['residual'])<1e-6
            assert m['peak']<=1000+1e-7 and m['posts']>=m['all_pulls']>=m['pressure_pulls']
            assert abs(m['shares']-2*m['matched']-abs(m['end_pos']))<1e-5
            assert (a!='A' or m['pressure_pulls']==0) and 0<=m['both']<=1
            rows.append(dict(mid=d['mid'],arm=a,**m))
            for key,actual,z in v['anchors']:
                cap,level=forecast(fit['cells'],key)
                item=[cap,actual,level,0.,0.]
                anchors[a].append(item)
                if a=='C':
                    band='missing' if z is None else ('near' if abs(z)<.5 else 'middle' if abs(z)<1.5 else 'far')
                    drift.setdefault(band,[]).append(item)
            if a=='C':
                assert len(v['proposals'])==m['posts']
                for cap,actual,lev,expo,unit in v['proposals']:
                    assert max(0.,expo-cap)*unit<=50+1e-6
                props.extend(v['proposals'])
        assert len(set(opened))==1
    frame=pd.DataFrame(rows)
    assert set(frame.mid)==expected and len(frame)==len(expected)*3 and not frame.duplicated(['mid','arm']).any()
    summary={}
    for a,g in frame.groupby('arm'):
        summary[a]={k:float(g[k].sum()) for k in ('settle','turnover','residual','marked','shares','matched','posts','pressure_pulls','all_pulls','risk_reduced_slots')}
        summary[a].update(both=float(g.both.mean()),max_peak=float(g.peak.max()),median_peak=float(g.peak.median()),mean_abs_end=float(g.end_pos.abs().mean()),
                          winners=int((g.settle>1e-7).sum()),first_delay_median=float(g.first_s.median()),no_open=int(g.first_s.isna().sum()))
    contrasts={}
    for treatment,base in [('B','A'),('C','B')]:
        delta=frame[frame.arm==treatment].set_index('mid').settle-frame[frame.arm==base].set_index('mid').settle
        contrasts[treatment+'-'+base]=dict(net=float(delta.sum()),better=int((delta>1e-7).sum()),worse=int((delta< -1e-7).sum()),equal=int((delta.abs()<=1e-7).sum()))
    frame.to_csv(HERE/'markets.csv',index=False)
    write(HERE/'results.json',dict(day=EVAL,status=Counter(r['status'] for r in cen),training_markets=len(fit['inputs']),fit_sha=sha(HERE/'fit.json'),
                                  rows=summary,contrasts=contrasts,capacity_on_clock={a:aggregate(data) for a,data in anchors.items()},
                                  capacity_at_C_submissions=aggregate(props),capacity_on_C_clock_by_abs_z={k:aggregate(v) for k,v in drift.items()},
                                  passes=summary['C']['settle']>0 and contrasts['C-B']['net']>0,inputs=inputs))


if __name__=='__main__':
    main()
