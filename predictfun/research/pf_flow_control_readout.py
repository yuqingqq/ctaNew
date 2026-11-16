"""Additional descriptive readout of the frozen controller pilot; no policy selection."""
from pathlib import Path
import json
import subprocess
import pf_flow_control_common as F
C,np,pd=F.C,F.np,F.pd


def neutral_inventory_support(probes):
    """Coverage diagnostic only; no outcomes are removed from the primary report."""
    import pf_flow_control_run as RUN
    r=json.loads((C.OUT/F.TEST/'receipt.json').read_text())
    mids=set(map(int,probes.m.unique()));data=F.I.load_day(F.TEST,mids,r['imputation_offset_ms']);records=[]
    for W in F.I.pairs(data):
        rows=probes[probes.m==W[0]['m']];bytime={float(t):g for t,g in rows.groupby('decision_ms')}
        def observe(k,t,states,position):
            for row in bytime.get(t,pd.DataFrame()).itertuples():
                C.require(int(row.generation)<len(states[int(row.side)]['orders'][int(row.rung)]),'probe generation missing')
                order=states[int(row.side)]['orders'][int(row.rung)][int(row.generation)]
                C.require(order['ta']<=t,'probe before activation')
                records.append(dict(m=int(row.m),side=int(row.side),rung=int(row.rung),generation=int(row.generation),
                                    decision_ms=float(t),known_inventory=position))
        RUN.run_lane(W,cap=None,observer=observe)
    result=pd.DataFrame(records);keys=['m','side','rung','generation','decision_ms']
    C.require(len(result)==len(probes) and not result.duplicated(keys).any(),'neutral inventory identity coverage')
    merged=probes.merge(result,on=keys,validate='one_to_one');outside=merged.known_inventory.abs()>F.CAP+1e-7
    summary=dict(n=len(merged),outside_controller_net_cap=int(outside.sum()),inside_controller_net_cap=int((~outside).sum()),
        max_abs_known_inventory=float(merged.known_inventory.abs().max()),
        note='All primary probe outcomes retained. Net position alone does not establish feasible outstanding-order reservations or full on-policy coverage.')
    return merged,summary


def run():
    F.verify();src=F.OUT/'pilot';receipt=json.loads((src/'receipt.json').read_text())
    for name,h in receipt['outputs'].items():C.require(C.sha(src/name)==h,'readout artifact changed '+name)
    audit=json.loads((src/'audit.json').read_text());report=json.loads((src/'report.json').read_text())
    for field in ('confirmed_partial_fill_front_preserved','pending_cache_deferred_until_activation','unknown_zero_queue_preserved'):
        C.require(audit['controls'][field] is True,'missing queue control '+field)
    m=pd.read_parquet(src/'markets.parquet');p=pd.read_parquet(src/'probes.parquet');d=pd.read_parquet(src/'decisions.parquet')
    inventory=pd.read_parquet(src/'inventory.parquet');f=pd.read_parquet(src/'fills.parquet')
    p,inventory_support=neutral_inventory_support(p)
    output=F.REVIEW/'pilot_v3';C.require(not output.exists(),'readout exists');output.mkdir(parents=True)
    sets={case:set(g.m) for case,g in m.groupby('case')}
    C.require(all(z==next(iter(sets.values())) for z in sets.values()),'unmatched replay markets')
    a=m[m.case=='JOINT'].set_index('m');b=m[m.case=='EMPIRICAL'].set_index('m')
    paired={field:dict(total_delta=float((a[field]-b[field]).sum()),positive_markets=int((a[field]>b[field]).sum()),markets=len(a))
            for field in ('terminal_pnl_usd','settlement_pnl_usd','shares','both_quote_ms','pulls')}
    churn={}
    for case,g in d.groupby('case'):
        cancel=g[g.kind=='cancel'];age=cancel.time-cancel.activation
        churn[case]=dict(selected_posts=int((g.kind=='post').sum()),selected_cancels=len(cancel),
            cancel_decision_age_ms_median=float(age.median()) if len(age) else None,
            cancel_at_first_review_fraction=float((age<=50.+1e-7).mean()) if len(age) else None,
            cancel_below_250ms_fraction=float((age<250.).mean()) if len(age) else None,
            note='Decision age, not actual lifetime: add cancellation latency, and earlier fills may end the order. Excludes final forced shutdown.')
    ranking={}
    for name,column in [('joint','predicted_advantage_cents'),('empirical','empirical_advantage_cents')]:
        pred=p[column];actual=p.realized_advantage_cents;selected=pred>0
        groups=pd.qcut(pred.rank(method='first'),5,labels=False)
        bins=[]
        for label in range(5):
            g=p[groups==label];bins.append(dict(bin=label,n=len(g),predicted_mean_cents=float(g[column].mean()),actual_mean_cents=float(g.realized_advantage_cents.mean())))
        ranking[name]=dict(n=len(p),spearman=float(pred.corr(actual,method='spearman')),
            predicted_mean_cents=float(pred.mean()),actual_mean_cents=float(actual.mean()),
            positive_predictions=int(selected.sum()),positive_predicted_mean_cents=float(pred[selected].mean()),
            positive_realized_mean_cents=float(actual[selected].mean()),quintiles=bins)
    phase=[]
    for (case,minute),g in inventory.groupby(['case','minute']):
        phase.append(dict(case=case,minute=int(minute),markets=len(g),
            quote_both_percent=float(g.quote_both_ms.sum()/g.observed_ms.sum()*100),
            quote_one_percent=float((g.quote_any_ms-g.quote_both_ms).sum()/g.observed_ms.sum()*100),
            incremental_marked_pnl_usd=float(g.incremental_marked_pnl_usd.sum()),
            mean_abs_ending_inventory=float(g.abs_position_end.mean())))
    byphase=[]
    for (case,minute),g in f.groupby(['case','minute']):
        ok=g.markout_1000_status=='ok';shares=float(g.loc[ok,'shares'].sum())
        byphase.append(dict(case=case,minute=int(minute),fills=len(g),shares=float(g.shares.sum()),
            markout_1s_covered_shares=shares,markout_1s_cents_per_share=float(g.loc[ok,'markout_1000_usd'].sum()*100/shares) if shares else None))
    interpretation=dict(asof=C.stamp(),test_day=F.TEST,markets=len(a),neutral_actions=len(p),
        paired_joint_minus_empirical=paired,churn=churn,neutral_advantage_prediction=ranking,neutral_inventory_support=inventory_support,
        phase_inventory=phase,phase_markouts=byphase,
        predicates=dict(joint_positive_terminal=bool(a.terminal_pnl_usd.sum()>0),
            joint_positive_settlement=bool(a.settlement_pnl_usd.sum()>0),
            joint_higher_terminal_than_empirical=bool(a.terminal_pnl_usd.sum()>b.terminal_pnl_usd.sum()),
            joint_higher_both_quote_time=bool(a.both_quote_ms.sum()>b.both_quote_ms.sum()),
            joint_fewer_pulls=bool(a.pulls.sum()<b.pulls.sum())),
        limits=['One consumed UTC day, point estimates only; no validation/promotion or confidence intervals',
            'Different policy exposures; raw P&L deltas alone do not establish information value',
            'Matched neutral branch diagnostic covers resting cancellation, not fresh-entry calibration',
            'Net inventory/reservation bounds are not a complete gross-collateral or inventory liquidation model',
            'Finite-horizon zero continuation and empirical path transport remain approximations'],
        evaluator_source_sha256=C.sha(Path(__file__)),evaluator_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=F.ROOT,text=True).strip(),
        input_hashes={p.name:C.sha(p) for p in src.iterdir() if p.is_file()})
    C.dump(output/'interpretation.json',interpretation)
    p.to_csv(output/'neutral_probe_inventory.csv',index=False)
    for name in ('report.json','receipt.json','audit.json'):(output/name).write_bytes((src/name).read_bytes())
    chart(report,output/'comparison.png')
    C.dump(output/'readout_receipt.json',dict(asof=C.stamp(),source_sha256=C.sha(Path(__file__)),
        outputs={p.name:C.sha(p) for p in output.iterdir() if p.is_file()}))
    print(json.dumps(C.clean(dict(rows=report['rows'],interpretation=interpretation,matched_null=report['matched_null']))),flush=True)


def chart(report,path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows=report['rows'];names=[r['case'].replace('_','\n') for r in rows]
    fig,axes=plt.subplots(1,3,figsize=(13,4.4))
    fields=[('markout_1s_cents_per_share','1-second markout','cents / covered filled share'),
            ('both_quote_percent','Both sides quoting','% of full contract time'),
            ('terminal_pnl_usd','Terminal marked value','USD across pilot markets')]
    for ax,(key,title,unit) in zip(axes,fields):
        values=[r[key] for r in rows];ax.bar(names,values,color=['#68737d','#80a6c2','#146c94','#87a86b'])
        ax.axhline(0,color='#444',lw=.7);ax.set_title(title);ax.set_ylabel(unit);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    fig.suptitle('Oct 27 controller development pilot — 12 markets, one consumed day',fontsize=12)
    fig.tight_layout();fig.savefig(path,dpi=150);plt.close(fig)


if __name__=='__main__':run()
