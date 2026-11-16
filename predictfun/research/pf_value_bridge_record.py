"""Publish small immutable review artifacts and computed interpretation predicates."""
import json
from pathlib import Path
import subprocess
import pf_value_bridge as V
C,pd,F=V.C,V.pd,V.F


def run():
    V.verify();C.require((V.OUT/'audit_v1.json').exists() and (V.OUT/'oracle_v1/audit_v1.json').exists(),'missing audits')
    files=['audit_v1.json']
    for stage in ['actions_20261026','actions_20261027','pilot','oracle_v1']:
        files += [stage+'/'+name for name in ['report.json','receipt.json']]
    files+=['oracle_v1/audit_v1.json']
    for name in files:
        dest=V.REVIEW/name;C.require(not dest.exists(),'review artifact exists '+name);dest.parent.mkdir(parents=True,exist_ok=True)
        dest.write_bytes((V.OUT/name).read_bytes())
    m=pd.read_parquet(V.OUT/'pilot/markets.parquet');a=m[m.case=='SOFT_JOINT'].set_index('m');b=m[m.case=='EXACT_JOINT'].set_index('m')
    d=pd.read_parquet(V.OUT/'pilot/decisions.parquet');predicates={}
    for key in ['terminal_pnl_usd','settlement_pnl_usd','pulls','both_quote_ms']:
        delta=a[key]-b[key];predicates[key]=dict(markets=len(delta),higher=int((delta>0).sum()),lower=int((delta<0).sum()),equal=int((delta==0).sum()),total_change=float(delta.sum()))
    churn={}
    for case,g in d[d.kind=='cancel'].groupby('case'):
        age=g.time-g.activation;churn[case]=dict(n=len(g),first_review_fraction=float((age<=50.+1e-7).mean()))
    base={}
    for day in ['20261026','20261027']:
        rows=pd.read_parquet(V.OUT/('actions_'+day)/'values.parquet');rows=rows[rows.case=='SOFT_JOINT']
        for kind,g in rows.groupby('kind'):base[day+':'+kind]=dict(n=len(g),positive_actual=int((g.actual>1e-9).sum()),zero_actual=int((g.actual.abs()<=1e-9).sum()),negative_actual=int((g.actual<-1e-9).sum()))
    report=json.loads((V.OUT/'pilot/report.json').read_text());rows={r['case']:r for r in report['rows']}
    predicates['soft_positive_one_second_markout']=rows['SOFT_JOINT']['markout_1s_cents_per_share']>0
    predicates['soft_positive_terminal_pnl']=rows['SOFT_JOINT']['terminal_pnl_usd']>0
    predicates['lower_pull_count']=rows['SOFT_JOINT']['pulls']<rows['EXACT_JOINT']['pulls']
    C.require(predicates['lower_pull_count']==(a.pulls.sum()<b.pulls.sum()),'interpretation differs from artifacts')
    result=dict(asof=C.stamp(),source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=F.ROOT,text=True).strip(),
        source_sha256=C.sha(Path(__file__)),candidate_source_commit=V.verify()['source_commit'],freeze_sha256=C.sha(V.FREEZE),
        outputs={name:C.sha(V.REVIEW/name) for name in files},interpretation_predicates=predicates,churn=churn,action_outcome_counts=base,
        scope='One frozen valuation candidate; consumed Oct26/27 action diagnostics and12-market Oct27 pilot. No policy threshold change, untouched validation or promotion. Future information restricted to retrospective diagnostic.')
    C.require(not (V.REVIEW/'index.json').exists(),'index exists');C.dump(V.REVIEW/'index.json',result)
    print(json.dumps(C.clean(dict(predicates=predicates,churn=churn,outcome_counts=base))),flush=True)


if __name__=='__main__':run()
