"""Readable comparison on common observable windows; preserve raw experiment receipts."""
import datetime as dt
import json
from pathlib import Path
import numpy as np
import pandas as pd
from . import pf_queue_model_spec as S
from . import pf_queue_age_control as A
from . import pf_queue_accounting_v2 as C


def run():
    raw=json.loads((C.OUT/'report.json').read_text());age=json.loads((A.OUT/'report.json').read_text())
    ver=json.loads((C.OUT/'verification.json').read_text())
    if ver['status']!='PASS':raise ValueError('unverified original result')
    parts=[]
    for folder,report in [(C.OUT,raw),(A.OUT,age)]:
        path=folder/'portfolio_windows.parquet'
        if S.sha(path)!=report['portfolio_sha']:raise ValueError('portfolio hash mismatch')
        parts.append(pd.read_parquet(path))
    frame=pd.concat(parts,ignore_index=True)
    if frame.duplicated(['day','mid','config','arm']).any():raise ValueError('duplicate portfolio window')
    output=[];comparisons=[]
    for config,g in frame.groupby('config',sort=False):
        table=g.pivot(index=['day','mid'],columns='arm',values='stop_value_c')
        common=table.index[table.notna().all(axis=1)]
        for arm,z in g.groupby('arm',sort=False):
            on=z.set_index(['day','mid']).loc[common]
            output.append(dict(config=config,arm=arm,days=int(z.day.nunique()),windows=len(z),common_mark_windows=len(on),
                marked_usd=float(on.stop_value_c.sum()/100),common_settlement_usd=float(on.settle_value_c.sum()/100),
                common_carry_usd=float((on.settle_value_c-on.stop_value_c).sum()/100),
                full_settlement_usd=float(z.settle_value_c.sum()/100) if z.settle_value_c.notna().all() else None,
                missing_stop=int(z.stop_value_c.isna().sum()),missing_settlement=int(z.settle_value_c.isna().sum()),
                spread_usd=float(z.spread_c.sum()/100),mo1_usd=float(z.mo1_c.sum()/100),mo60_usd=float(z.mo60_c.sum()/100),
                missing_mark_groups=int(z.missing_marks.sum()),
                inventory_rms=float(np.sqrt(z.inventory_square_s.sum()/(len(z)*240))),
                inventory_square_s=float(z.inventory_square_s.sum()),terminal_abs_mean=float(z.terminal_abs.mean()),
                shares=float(z.shares.sum()),orders=int(z.orders.sum()),fills=int(z.fills.sum())))
        candidate=next(x for x in output if x['config']==config and x['arm']=='queue')
        for arm in ('market','market_age','always_keep','always_cancel'):
            ref=next(x for x in output if x['config']==config and x['arm']==arm)
            comparisons.append(dict(config=config,ref=arm,common_mark_windows=len(common),
                marked_delta_usd=candidate['marked_usd']-ref['marked_usd'],
                common_settlement_delta_usd=candidate['common_settlement_usd']-ref['common_settlement_usd'],
                full_settlement_delta_usd=candidate['full_settlement_usd']-ref['full_settlement_usd'] if candidate['full_settlement_usd'] is not None and ref['full_settlement_usd'] is not None else None,
                inventory_square_change_pct=100*(candidate['inventory_square_s']/ref['inventory_square_s']-1) if ref['inventory_square_s'] else None,
                terminal_abs_change_pct=100*(candidate['terminal_abs_mean']/ref['terminal_abs_mean']-1) if ref['terminal_abs_mean'] else None))
    decision_checks=[]
    for config in S.CONFIGS:
        mark=next(x for x in age['deltas'] if x['config']==config and x['metric']=='stop_value_c')
        inv=next(x for x in age['deltas'] if x['config']==config and x['metric']=='inventory_square_s')
        interval=mark.get('interval')
        decision_checks.append(dict(config=config,complete_mark_comparison='sum' in mark,
            marked_interval_positive=bool(interval and interval['low']>0),marked_point_positive=bool(mark.get('sum',float('-inf'))>0),
            inventory_square_nonincreasing=bool(inv.get('sum',float('inf'))<=0)))
    primary=next(x for x in decision_checks if x['config']=='L50')
    supported=primary['marked_interval_positive'] and all(x['complete_mark_comparison'] and x['marked_point_positive'] and x['inventory_square_nonincreasing'] for x in decision_checks)
    result=dict(as_of=dt.datetime.now(dt.timezone.utc).isoformat(),tables=output,comparisons=comparisons,
        decision_checks=decision_checks,supports_incremental_queue_improvement=supported,
        original_sha=S.sha(C.OUT/'report.json'),age_sha=S.sha(A.OUT/'report.json'),
        note='Common-mark tables are a reporting clarification, not a new policy or validation. Full marked PnL is unavailable when stop marks are missing; original partial sums are preserved. No decision interval is asserted for the common-mark subset. Settlement is shown separately.')
    S.write(S.OUT/'comparison.json',result);S.write(S.REVIEW/'comparison.json',result)
    rows=['# Queue continuation model: controlled research comparison','',f"As of {result['as_of']}. All dates were previously consumed; exploratory results.",'',
        'Built KEEP/CANCEL/same-price REJOIN value models using market conditions, known inventory and remaining size, with or without causal queue estimates. Orders are fixed at5 shares on a shared entry schedule. Cancellation/acknowledgment/fill-notice latency and pending exposure are included. Training expands chronologically from Oct23–25 to six test days: Oct26/27/31 and Nov1/2/3.', '',
        'The additional market_age baseline includes order age, isolating queue-size/progress information from remaining lifetime. Its source and design were committed before inspecting portfolio returns. Frozen v3 and live code are unchanged.','',
        '## Primary conservative queue assumption (L50)','',
        '| Policy | Marked PnL, common windows | Settlement PnL, common windows | Inventory RMS, all windows | Mean terminal absolute inventory | Orders |',
        '|---|---:|---:|---:|---:|---:|']
    for arm in ('always_keep','always_cancel','legacy_score','market','market_age','queue'):
        z=next(x for x in output if x['config']=='L50' and x['arm']==arm)
        rows.append(f"| {arm} | ${z['marked_usd']:.2f} | ${z['common_settlement_usd']:.2f} | {z['inventory_rms']:.3f} sh | {z['terminal_abs_mean']:.3f} sh | {z['orders']:,} |")
    z=next(x for x in output if x['config']=='L50')
    rows+=['',f"Population: {z['common_mark_windows']:,} common observable stop-mark windows out of {z['windows']:,} scheduled test windows across {z['days']} complete UTC days. Inventory and order counts cover all scheduled windows. Raw full-settlement totals, spread/markouts, all scenarios, and exclusion counts are in comparison.json and the original report.json.",'',
           '## Queue model versus age-matched market baseline','',
           '| Assumption | Common marked delta | Full settlement delta | Inventory-square time change |',
           '|---|---:|---:|---:|']
    for c in comparisons:
        if c['ref']!='market_age':continue
        settle=f"${c['full_settlement_delta_usd']:+.2f}" if c['full_settlement_delta_usd'] is not None else 'Unavailable'
        rows.append(f"| {c['config']} | ${c['marked_delta_usd']:+.2f} | {settle} | {c['inventory_square_change_pct']:+.2f}% |")
    rows+=['',f"Computed evidence rule supports incremental queue improvement: **{supported}**. See decision_checks in comparison.json; missing complete marked values prevent a positive primary decision.",'',
        '## Checks and limits','',
        f"15 synthetic controls passed. Original saved-artifact verifier reconciled {ver['portfolio_windows_checked']:,} portfolio windows and {ver['selected_fill_records_checked']:,} selected fills; the age baseline independently reconciled {age['verified_windows']:,} more windows. Nulls use500 action-matched shuffles; intervals are clustered on UTC day.",'',
        'Accounting v2 supersedes a floating-point bug: a roughly1e-15-share residue made an economically flat portfolio lose its known cash mark when the book mark was missing. Stable summation and1e-9-share zero tolerance restore those marks. Original model decisions, labels, fills, failed verifier and report are preserved. All comparisons here use the superseding receipt.', '',
        'This is a fixed-entry continuation experiment, not a full-v3 comparison. Queue position and fill timing remain uncalibrated against own live fills. RuleL credits trades only; ruleM additionally caps queue ahead by observed same-price size. Slower timings are a separate sensitivity. Maker fees were verified zero in input registry; rebates are excluded. Legacy-score baseline uses prior LODO predictions and is diagnostic only. Markout/spread sums with missing-mark groups are partial, not complete trading PnL.', '',
        'Reproduction: source68769426; original freeze9f8b09a8; age-control source recorded in its declaration. Run run_queue_model.sh controls, declare, commit declaration, all; then pf_queue_age_control declare/run and pf_queue_report. Existing output directories intentionally refuse overwrites.','']
    path=Path(__file__).resolve().parents[1]/'handoffs/20261111_queue_policy_results.md'
    with path.open('x') as f:f.write('\n'.join(rows))
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':run()
