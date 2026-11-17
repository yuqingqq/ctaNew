"""Post-hoc ledger inspection of three preselected largest client-state residuals/day.

This verifies a concrete interpretation of the completed diagnostic, not a new
candidate, unseen-day test or parameter search. Never infer current remainder
from an order's final remainder; reconstruct it from actual fill timestamps.
"""
import json
from pathlib import Path
import subprocess
import pf_value_decomposition as D
F,C,np,pd,I,A,B=D.F,D.C,D.np,D.pd,D.I,D.A,D.B


def remaining(fills,t):
    actual=30.-sum(q for ft,q in fills if ft<=t)
    known=30.-sum(q for ft,q in fills if ft+100.<=t)
    C.require(-1e-7<=actual<=known+1e-7<=30.+2e-7,'remainder chronology')
    return actual,known


def run():
    D.verify();out=D.OUT/'state_check_v1';C.require(not out.exists(),'state check exists');out.mkdir()
    source=Path(__file__).resolve();rows=[];inputs={}
    C.require(remaining([(50.,10.),(100.,20.)],150.)==(0.,20.),'unreceived-fill positive control')
    controls=dict(unreceived_fill_detected=True,overfill_refused=D.CHECK.refused(lambda:remaining([(0.,31.)],200.),'overfill'))
    for day in (F.CAL,F.TEST):
        src=D.OUT/day;receipt=json.loads((src/'report_receipt.json').read_text())
        C.require(C.sha(src/'largest_errors.parquet')==receipt['outputs']['largest_errors.parquet'],'residual examples changed')
        inputs[str(src/'report_receipt.json')]=C.sha(src/'report_receipt.json')
        a=pd.read_parquet(src/'largest_errors.parquet')
        chosen=a[(a.channel=='actual_path')&(a.case=='ACTUAL_PATH_CLIENT')&(a.kind=='cancel')].sort_values('squared_error',ascending=False).head(3)
        C.require(len(chosen)==3,'residual population');markets=set(chosen.m.astype(int))
        raw=json.loads((C.OUT/day/'receipt.json').read_text())
        for p,h in raw['input_hashes'].items():C.require(C.sha(p)==h,'raw identity changed '+p)
        data=I.load_day(day,markets,raw['imputation_offset_ms']);seen=set()
        for W in I.pairs(data):
            m=int(W[0]['m']);seen.add(m)
            with A.CLOCK.clocks(58.,37.):r=A.AC.replay_neutral(W,size=30.,shift_ms=50.,queue='M',nr=6,cap=F.CAP,mode='hard')
            for row in chosen[chosen.m==m].to_dict('records'):
                t=row['decision_ms'];activation=t-row['x_age_ms']
                orders=[o for o in r['order_states'][int(row['side']),int(row['rung'])] if abs(o['ta']-activation)<.001]
                C.require(len(orders)==1 and orders[0]['pt']==row['price'],'residual order identity')
                actual,known=remaining(orders[0]['fl'],t)
                C.require(np.isclose(known,row['known_remaining'],rtol=0,atol=1e-7),'client remainder reconstruction')
                C.require(row['actual_affected_quantity']<=actual+1e-7,'actual effect exceeds real remainder')
                rows.append(dict(uid=row['uid'],day=day,m=m,side=int(row['side']),rung=int(row['rung']),decision_ms=t,
                    actual_remaining=actual,client_known_remaining=known,unreceived_filled_quantity=known-actual,
                    unreceived_fills=[dict(fill_ms=ft,quantity=q,notice_ms=ft+100.) for ft,q in orders[0]['fl'] if ft<=t<ft+100.],
                    predicted_affected_quantity=row['pred_affected_quantity'],actual_affected_quantity=row['actual_affected_quantity'],
                    predicted_value=row['predicted'],actual_value=row['actual'],
                    predicted_effect_exceeds_true_remainder=max(0.,row['pred_affected_quantity']-actual)))
        C.require(seen==markets,'missing residual market');C.guard('state inspection')
    result=dict(asof=C.stamp(),source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=F.ROOT,text=True).strip(),
        source_sha256=C.sha(source),inherited_freeze_sha256=C.sha(D.FREEZE),inputs=inputs,controls=controls,n=len(rows),rows=rows,
        selected_posthoc=True,selection='Exactly3 largest squared-error ACTUAL_PATH_CLIENT cancellation residuals per consumed Oct26/27, sorted from frozen largest_errors artifacts',
        limits=['Identity-matched descriptive inspection, not population-wide attribution or new model evidence',
            'Unreceived fills are unavailable to the client at decision time; using actual remainder in a model would leak',
            'Residual quantity differences not explained by remainder can involve queue/structural-state approximation; not resolved here'])
    C.dump(out/'report.json',result);dest=D.REVIEW/'state_check_v1.json';C.require(not dest.exists(),'state review exists');dest.write_bytes((out/'report.json').read_bytes())
    print(json.dumps(C.clean(result)),flush=True)


if __name__=='__main__':run()
