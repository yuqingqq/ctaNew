"""Post-hoc descriptive intersections on the unchanged baseline action states.

No policy replay, threshold selection or inferential claim. The main report's
overlapping off-touch/score/notional counts motivate checking whether removing
one refusal would actually leave an executable quote under the other checks.
"""
import datetime as dt
import json
from pathlib import Path
import subprocess
import numpy as np
import pandas as pd
from .pf_inventory_cost_study import DEST,HERE,DAYS,sha,write
from .pf_inventory_cost_run import folder

BLOCKS=frozenset(('invalid_price','off_touch','book_gap','binance','missing_score','minimum_order',
                 'rounded_minimum','room_below_minimum','no_book','invalid_book','score_line'))


def key(flags):
    if not isinstance(flags,(list,tuple,np.ndarray)) or any(not isinstance(x,str) for x in flags):
        raise ValueError('invalid episode flags')
    remaining=BLOCKS.intersection(flags)
    if not remaining:return 'eligible'
    if remaining=={'score_line'}:return 'score_only'
    if remaining=={'off_touch'}:return 'off_touch_only'
    if remaining=={'off_touch','score_line'}:return 'score_and_off_touch_only'
    return 'other_blockers'


def aggregate(a):
    if (a.end<a.start).any() or (a.start<60).any() or (a.end>300).any():
        raise ValueError('invalid episode interval')
    active=a[(a.net.abs()>1e-9)&(a.start<240)].copy()
    if (active.end>240+1e-9).any():raise ValueError('episode crosses closing cutoff')
    active['seconds']=active.end-active.start;active['category']=active['flags'].map(key)
    return {k:dict(episodes=len(g),seconds=float(g.seconds.sum())) for k,g in active.groupby('category')}


def controls():
    states=[[],['score_line'],['off_touch'],['score_line','off_touch'],['score_line','minimum_order'],['reserved_room']]
    expected=['eligible','score_only','off_touch_only','score_and_off_touch_only','other_blockers','eligible']
    if [key(x) for x in states]!=expected:raise ValueError('positive intersection control failed')
    a=pd.DataFrame(dict(start=[60.,70.],end=[70.,80.],net=[10.,0.],flags=[['off_touch'],[]]))
    if aggregate(a)!={'off_touch_only':dict(episodes=1,seconds=10.)}:raise ValueError('positive duration control failed')
    refused=0
    for bad in (None,'score_line',[1]):
        try:key(bad)
        except ValueError:refused+=1
    bad=a.copy();bad.loc[0,'end']=59.
    try:aggregate(bad)
    except ValueError:refused+=1
    if refused!=4:raise ValueError('intersection refusal controls failed')
    return dict(positive_intersections=True,positive_duration=True,known_bad_refusals=refused)


def main():
    control=controls();report_path=DEST/'report.json';r=json.loads(report_path.read_text());out=[];inputs={}
    verification=json.loads((DEST/'verification.json').read_text())
    if sha(report_path)!=verification['report_sha']:raise ValueError('unverified main report')
    for tm in ('J','X'):
        totals={};episodes=0;denominator=0.
        for day in DAYS:
            root=folder(day,tm);checks_path=root/'checks.json'
            if sha(checks_path)!=r['artifacts'][str(checks_path)]:raise ValueError('changed check artifact')
            checks=json.loads(checks_path.read_text());path=root/'exit_episodes.parquet'
            if sha(path)!=checks['artifacts']['exit_episodes.parquet']:raise ValueError('changed audit artifact')
            inputs[str(path)]=sha(path);a=pd.read_parquet(path);values=aggregate(a)
            ref=next(v for v in r['audits'] if (v['day'],v['timing'],v['mode'])==(day,tm,'main'))
            den=sum(v['seconds'] for v in values.values())
            if not np.isclose(den,ref['inventory_seconds'],rtol=0,atol=1e-7):raise ValueError('audit denominator mismatch')
            if not np.isclose(values.get('score_only',{}).get('seconds',0),ref['score_only_executable_seconds'],rtol=0,atol=1e-7):
                raise ValueError('score-only reconciliation')
            episodes+=sum(v['episodes'] for v in values.values());denominator+=den
            for k,v in values.items():
                t=totals.setdefault(k,dict(episodes=0,seconds=0.));t['episodes']+=v['episodes'];t['seconds']+=v['seconds']
        for v in totals.values():v['percent']=100*v['seconds']/denominator
        out.append(dict(timing=tm,days=9,markets=2592,active_episodes=episodes,inventory_seconds=denominator,categories=totals))
    result=dict(as_of=dt.datetime.now(dt.timezone.utc).isoformat(),kind='post-hoc descriptive baseline-state intersection audit',
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=HERE,text=True).strip(),source_sha=sha(__file__),
        report_sha=sha(report_path),controls=control,results=out,input_sha=inputs,
        limits='Decision-sampled durations, not independent actions or continuous executable availability; off-touch repair is not replayed; no prediction of added fills/PnL; reserved_room may allow a smaller legal quote.')
    write(DEST/'joint_audit.json',result);write(HERE.parent/'reviews/inventory_cost_20261110/joint_audit.json',result)
    print(json.dumps(dict(as_of=result['as_of'],results=out,controls=control),indent=2))


if __name__=='__main__':main()
