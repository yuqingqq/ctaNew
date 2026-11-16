"""Archive compact receipts and derive interpretation from audited artifacts."""
import json
from pathlib import Path
import subprocess
import pf_response_bridge_v2 as B
C,pd,F=B.C,B.pd,B.F


def run():
    B.verify();C.require((B.REVIEW/'audit_v2.json').exists(),'missing evaluation audit')
    files=['fit/receipt.json','prediction_report.json']
    for stage in ('actions_20261026','actions_20261027','pilot','oracle'):
        files += [stage+'/'+name for name in ('receipt.json','report.json')]
    for name in files:
        src=B.OUT/name;dest=B.REVIEW/name;C.require(src.exists() and not dest.exists(),'review source/destination '+name)
        dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(src.read_bytes())
    files+=['audit_v2.json'];p=json.loads((B.OUT/'pilot/report.json').read_text());rows={x['case']:x for x in p['rows']}
    a,b=rows['RICH_JOINT'],rows['SOFT_JOINT'];predicates={}
    for key in ('terminal_pnl_usd','settlement_pnl_usd','markout_1s_cents_per_share','markout_5s_cents_per_share','both_quote_percent','pulls'):
        predicates[key]=dict(baseline=b[key],candidate=a[key],delta=a[key]-b[key],higher=a[key]>b[key])
    for case,g in rows.items():predicates[case+'_first_review_cancel_fraction']=g['first_review_cancels']/g['model_cancels']
    action={};projection={}
    for day in (F.CAL,F.TEST):
        r=json.loads((B.OUT/('actions_'+day)/'report.json').read_text());m=r['metrics'];action[day]={}
        for kind in ('post','cancel'):
            a,b=m['RICH_JOINT:'+kind],m['SOFT_JOINT:'+kind]
            action[day][kind]=dict(n=a['n'],rmse_improved=a['rmse_cents']<b['rmse_cents'],rank_improved=a['spearman']>b['spearman'],
                zero_prediction_rmse_beaten=a['rmse_cents']<a['zero_prediction_rmse_cents'])
        values=pd.read_parquet(B.OUT/('actions_'+day)/'values.parquet')
        for case,g in values[values.evaluation_status=='eligible'].groupby('case'):
            projection[day+':'+case]=dict(n=len(g),effective_paths=float(g.effective_paths.mean()),
                coarse_l1=float(g.marginal_l1.mean()) if case=='SOFT_JOINT' else float(g.coarse_marginal_l1.mean()),
                rich_l1=None if case=='SOFT_JOINT' else float(g.rich_marginal_l1.mean()))
    C.dump(B.REVIEW/'index_v2.json',dict(asof=C.stamp(),source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=F.ROOT,text=True).strip(),
        source_sha256=C.sha(Path(__file__)),candidate_source_commit=B.verify()['source_commit'],freeze_sha256=C.sha(B.FREEZE),
        outputs={name:C.sha(B.REVIEW/name) for name in files},pilot_predicates=predicates,action_predicates=action,projection=projection,
        scope='One richer response candidate on consumed dates. Fixed bins/depths/parameters and old controller; separately declared empirical and future-signature diagnostics. No validation/promotion.'))
    print(json.dumps(C.clean(dict(pilot=predicates,actions=action))),flush=True)


if __name__=='__main__':run()
