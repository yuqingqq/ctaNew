"""Isolate old-vs-current empirical bank at the identical current state.

Before reading output: use the fixed384 saved pairs. Keep current book, bid
translation, forecast probabilities, queue, portfolio, horizon and action set.
Only substitute the entry bank's template identities. Refuse the intervention
if ANY old template violates current payoff bounds; retain a status. Unchanged
banks must reproduce the original value. This is a forecast-input sensitivity,
not proof the spread change is economically irrelevant or the old bank better.
"""
import json
from pathlib import Path
import pickle
import subprocess
import pf_flow_churn_diagnostic as D
C,np,pd,F=D.C,D.np,D.pd,D.F


def run():
    F.verify();receipt=json.loads((D.OUT/'receipt.json').read_text())
    for name,h in receipt['outputs'].items():C.require(C.sha(D.OUT/name)==h,'changed diagnostic artifact')
    pairs=pickle.loads((D.OUT/'states.pkl').read_bytes());paths=D.PATH.load();a=pd.read_parquet(D.OUT/'attribution.parquet').set_index('uid');rows=[]
    for pair in pairs:
        e,r=pair['entry'],pair['review'];sp=r['spec'];s=r['state'];i=r['index']
        old=D.PATH.select(paths,e['spec']['phase'],e['spec']['bid'],e['spec']['spread'])
        current=D.context(paths,sp);oldids={p['uid'] for p in old};newids={p['uid'] for p in current[3]}
        row=dict(uid=pair['uid'],m=pair['m'],side=int(s['orders'][i,0]),cancel=r['cancel'],bank_changed=oldids!=newids,current_keep=float(a.loc[pair['uid'],'current_keep']))
        admissible=all(p['min_price']+sp['bid']-p['bid']>=0 and p['max_price']+sp['bid']-p['bid']<=100 for p in old)
        if not admissible:row['status']='old_bank_outside_current_payoff_bounds';rows.append(row);continue
        packed=D.K.pack(old,sp['bid'],1000.);D.K.current_book(packed,sp['prices'],sp['sizes'])
        weights,diag=D.PATH.weights(old,sp['prob']);ctx=(packed,weights,current[2])
        value=D.advantage(s['orders'],r['prefix'],s['position'],i,ctx)['value']
        if oldids==newids:C.require(np.isclose(value,row['current_keep'],atol=1e-7),'unchanged bank moved value')
        row.update(status='eligible',old_bank_keep=value,bank_only_change_cents=row['current_keep']-value,
            old_bank_unsupported_mass=diag['unsupported_mass'],current_bank_unsupported_mass=current[4]['unsupported_mass'])
        rows.append(row)
    df=pd.DataFrame(rows);groups={}
    for name,cancel in [('cancel',True),('keep',False)]:
        g=df[df.cancel==cancel];z=g[(g.status=='eligible')&g.bank_changed]
        groups[name]=dict(n=len(g),statuses=g.status.value_counts().to_dict(),changed_bank_eligible_n=len(z),
            old_bank_keep_nonnegative=int((z.old_bank_keep>=-1e-9).sum()),
            mean_old_bank_keep_cents=float(z.old_bank_keep.mean()),mean_current_keep_cents=float(z.current_keep.mean()),
            mean_bank_only_change_cents=float(z.bank_only_change_cents.mean()),
            mean_old_bank_unsupported_mass=float(z.old_bank_unsupported_mass.mean()),
            mean_current_bank_unsupported_mass=float(z.current_bank_unsupported_mass.mean()))
    C.require(not (D.OUT/'bank_check_v1.json').exists(),'bank check exists')
    df.to_parquet(D.OUT/'bank_check_v1.parquet',index=False)
    result=dict(asof=C.stamp(),groups=groups,source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=F.ROOT,text=True).strip(),
        source_sha256=C.sha(Path(__file__)),input_receipt_sha256=C.sha(D.OUT/'receipt.json'),
        data_sha256=C.sha(D.OUT/'bank_check_v1.parquet'),
        controls=dict(unchanged_bank_values_match=int(((df.status=='eligible')&~df.bank_changed).sum()),
            corrupted_value_refused=D.CHECK.refused(lambda:C.require(np.isclose(1.,2.),'wrong value'),'bad value')),
        limits=['No policy change; frozen-bank alternative is a diagnostic, not a trading recommendation','Spread conditioning may legitimately alter future distributions; this test alone does not determine which bank is accurate','Out-of-bound counterfactuals remain statuses; no old path is silently dropped'])
    C.dump(D.OUT/'bank_check_v1.json',result);print(json.dumps(C.clean(result)),flush=True)


if __name__=='__main__':run()
