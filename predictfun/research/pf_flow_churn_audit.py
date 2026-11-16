"""Independent identity, paired-value and attribution audit of churn diagnostics."""
from collections import Counter
import json
from pathlib import Path
import pickle
import subprocess
import pf_flow_churn_diagnostic as D
F,C,np,pd=D.F,D.C,D.np,D.pd


def same(a,b,name):
    C.require(np.allclose(a,b,atol=1e-7,rtol=1e-9),name)


def audit():
    F.verify();r=json.loads((D.OUT/'receipt.json').read_text())
    C.require(C.sha(D.FREEZE)==r['source_manifest'],'wrong diagnostic freeze')
    C.require(C.sha(Path(D.__file__))==r['source_sha256'],'wrong evaluator source')
    for name,h in r['outputs'].items():C.require(C.sha(D.OUT/name)==h,'changed output '+name)
    rows=pd.read_parquet(D.OUT/'pairs.parquet');entries=pd.read_parquet(D.OUT/'entries.parquet')
    a=pd.read_parquet(D.OUT/'attribution.parquet').set_index('uid')
    pairs=pickle.loads((D.OUT/'states.pkl').read_bytes());paths=D.PATH.load()
    C.require(len(a)==len(pairs) and not rows.uid.duplicated().any(),'duplicate action identities')
    expected=set()
    for _,g in rows.groupby(['m','side','cancel']):
        expected.update(sorted(g.uid,key=D.digest)[:D.PER_GROUP])
    C.require(set(a.index)==expected=={p['uid'] for p in pairs},'identity-hash sample mismatch')
    C.require(set(rows.uid)==set(entries.loc[entries.status=='paired_first_review','uid']),'entry/first-review status mismatch')
    saved=json.loads((D.OUT/'report.json').read_text())
    recomputed=D.report(rows,a.reset_index(),entries);recomputed['asof']=saved['asof']
    C.require(C.clean(recomputed)==saved,'report metrics or null changed')
    for p in pairs:
        x=a.loc[p['uid']];e=p['entry'];v=p['review'];s=v['state'];idx=v['index']
        ctx=D.context(paths,v['spec']);keep=v['prefix'].copy();keep[idx]=1.;cancel=keep.copy();cancel[idx]=2.
        out,q=D.K.evaluate(ctx[0],s['orders'],np.stack([keep,cancel]),s['position'],ctx[2],1000.)
        dv=ctx[1]@(out[:,0,0]-out[:,1,0]-F.RISK*(out[:,0,2]**2-out[:,1,2]**2))
        same(dv,x.current_keep,'actual keep/cancel value mismatch')
        same(x.current_wealth+x.current_risk,x.current_keep,'wealth/risk decomposition')
        if v['cancel']:same(-dv,v['selected_advantage'],'recorded cancellation step differs')
        es=e['state'];ei=e['index'];ctx=D.context(paths,e['spec']);post=e['prefix'].copy();post[ei]=1.;wait=post.copy();wait[ei]=0.
        eo,eq=D.K.evaluate(ctx[0],es['orders'],np.stack([post,wait]),es['position'],ctx[2],1000.)
        ev=ctx[1]@(eo[:,0,0]-eo[:,1,0]-F.RISK*(eo[:,0,2]**2-eo[:,1,2]**2))
        same(ev,x.entry_selected,'actual post/wait value mismatch');same(ev,e['selected_advantage'],'recorded entry step differs')
        changes=['entry_greedy_change','entry_queue_assumption_change','target_activation_change','absent_to_cancel_change']
        changes+=['shapley_'+f for f in D.FACTORS]
        same(x.entry_selected+sum(x[z] for z in changes),x.current_keep,'full attribution bridge')
    bad_refused=D.CHECK.refused(lambda:same(a.current_keep.iloc[0]+1.,a.current_keep.iloc[0],'corrupted advantage'),'corrupt advantage')
    result=dict(asof=C.stamp(),n=len(a),paired_actions=len(rows),hash_sample_identities=True,
        all_entry_and_cancel_values_recomputed=True,all_attribution_bridges_reconciled=True,
        report_and_200_draw_null_recomputed=True,corrupt_advantage_refused=bad_refused,
        report_sha256=C.sha(D.OUT/'report.json'),receipt_sha256=C.sha(D.OUT/'receipt.json'),
        source_sha256=C.sha(Path(__file__)),source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=F.ROOT,text=True).strip())
    C.require(not (D.OUT/'audit_v1.json').exists(),'audit exists');C.dump(D.OUT/'audit_v1.json',result)
    print(json.dumps(result),flush=True)


if __name__=='__main__':audit()
