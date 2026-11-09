"""Publish the loop comparison only from linked, verified round artifacts."""
import datetime as dt
import json
import subprocess
from . import pf_inventory_grace_study as s


def verified_link(expected,verification):
    if verification['receipt_sha256']!=expected:raise ValueError('verification belongs to another receipt')
    if not verification['native_inventory_identity_reconciled'] or not verification['old_baseline_and_unrestricted_arm_reconciled']:raise ValueError('reconciliation failed')
    if not all(verification['controls'].values()):raise ValueError('verifier controls failed')


def main():
    a=s.a;outpath=s.REVIEW/'findings.json'
    if outpath.exists():raise ValueError('findings exist; use a superseding version')
    control=dict(receipt_sha256='good',native_inventory_identity_reconciled=True,old_baseline_and_unrestricted_arm_reconciled=True,controls=dict(positive=True,negative=True))
    verified_link('good',control)
    try:verified_link('bad',control)
    except ValueError:refused=True
    else:raise ValueError('wrong receipt accepted')
    rounds={};hashes={}
    for round_name in s.ROUNDS:
        dest=s.DEST/round_name;r=json.loads((dest/'receipt.json').read_text());v=json.loads((dest/'verification.json').read_text())
        verified_link(a.sha(dest/'receipt.json'),v)
        for p,h in r['artifacts'].items():
            if a.sha(dest/p)!=h:raise ValueError('reported artifact changed')
        rounds[round_name]=r;hashes[round_name]=dict(receipt=a.sha(dest/'receipt.json'),verification=a.sha(dest/'verification.json'))
    final=rounds['duration_latency'];by=final['summary'];comparisons={}
    for arm in s.ROUNDS['duration_latency']['arms']:
        cc={}
        for latency,bb in by.items():
            q=bb[arm];b=bb['reference'];allq=bb['all100'];gain=q['aligned_lane_ms']-b['aligned_lane_ms'];allgain=allq['aligned_lane_ms']-b['aligned_lane_ms']
            cc[latency]=dict(requests=q['requests'],request_change_pct=q['request_change_pct'],aligned_change_pct=q['aligned_change_pct'],
                coverage_pct=100*q['aligned_lane_ms']/q['desired_lane_ms'],aligned_gain_ms=gain,
                fraction_of_unrestricted_coverage_gain=gain/allgain if allgain>0 else None,
                extra_off_target_lane_ms=q['elsewhere_lane_ms']-b['elsewhere_lane_ms'],
                off_other_share_ms_change_pct=q['off_other_share_ms_change_pct'],
                active_range_extra_share_ms_change_pct=q['active_range_extra_share_ms_change_pct'],
                reserved_range_extra_share_ms_change_pct=q['reserved_range_extra_share_ms_change_pct'],
                retained_actions=q['retained_actions'],same_price_reopens=q['same_price_reopens'],
                fewer_requests_and_more_aligned=q['requests']<b['requests'] and gain>0,
                less_off_other_than_unrestricted=q['off_other_share_ms']<allq['off_other_share_ms'])
        comparisons[arm]=dict(by_latency=cc,all_scenarios_fewer_requests_and_more_aligned=all(x['fewer_requests_and_more_aligned'] for x in cc.values()))
    out=dict(as_of=dt.datetime.now(dt.timezone.utc).isoformat(),source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=a.HERE,text=True).strip(),
        report_source_sha256=a.sha(a.HERE/'pf_optimization_report.py'),population=final['population'],round_hashes=hashes,
        comparisons=comparisons,controls=dict(correct_link_accepted=True,wrong_receipt_refused=refused),
        limitations=final['limitations']+['one consumed partial day; latency scenarios are assumptions, not independent statistical validation'],
        next_evidence=['Rust A2 paired-order parity/integration (B4)','calibrated exact fills or separately authorized forward capture before economic selection','inventory-range rather than strict no-flip restriction is a possible separately declared follow-up'])
    a.write(outpath,out);a.write(s.DEST/'findings.json',out)
    lines=['# Optimization loop — 2026-11-09','',f"Verified comparison as-of **{out['as_of']}**. **220 consumed Nov8 windows**, **279,059 decisions**, **13,077 neutral touch-pull actions** per scenario/arm; zero excluded windows. One partial day; no confidence interval or independent validation.",'',
        'The loop completed two predeclared rounds: inventory restrictions, then shorter grace and latency stress. These results concern conditional quote mechanics and exposure. They do not estimate trading P&L, realized inventory variance, or settlement gains.','',
        'All arms use the same 0.01-cent soft exit except the reference. `all100/50/25` allow an accepted small touch order up to the named milliseconds. `individual100` also requires that order alone to reduce recorded inventory without crossing zero. `reserved100` requires enough reducing room for **all** same-side reservations, including pending creates/cancels and their grace. Flat inventory is refused by both inventory variants.','',
        'The extra inventory observer captures partial/full fill notices at their native knowledge time. Original native decisions, requests, books and due-event probes remain exact. Hypothetical orders never consume captured fills.','',
        '| Latency | Policy | Requests | Request change | Original aligned time change | Off-target non-reducing share-time change | Active inventory-range expansion change |',
        '|---|---|---:|---:|---:|---:|---:|']
    for latency in s.LATENCIES:
        for arm in s.ROUNDS['duration_latency']['arms']:
            q=comparisons[arm]['by_latency'][latency]
            lines.append(f"| {latency} | {arm} | {q['requests']:,} | {q['request_change_pct']:+.2f}% | {q['aligned_change_pct']:+.2f}% | {q['off_other_share_ms_change_pct']:+.2f}% | {q['active_range_extra_share_ms_change_pct']:+.2f}% |")
    lines+=['','Latency assumptions (create / accepted-cancel / pending-create-cancel): base **40/22/95 ms**, faster **20/10/50 ms**, slower **80/50/160 ms**. Cancellation reservation grace remains 1 second. Each row compares with its own latency reference. These are stress scenarios, not fitted venue quantiles.','',
        '“Non-reducing” includes flat inventory. “Inventory-range expansion” is the maximum absolute endpoint of `[recorded net − down orders, recorded net + up orders]` above `abs(recorded net)`, integrated over time. It does not net opposite orders and is not a realized risk or variance estimate.','',
        '## Computed comparisons','']
    for arm in ['individual100','reserved100','all50','all25']:
        c=comparisons[arm];q=c['by_latency']['base'];f=q['fraction_of_unrestricted_coverage_gain']
        lines.append(f"- `{arm}`: retains {100*f:.2f}% of unrestricted100ms coverage gain at base latency; {int(q['retained_actions']):,} retained admissions / {int(q['same_price_reopens']):,} same-price reopens. Fewer requests and more original aligned time in all three scenarios: **{c['all_scenarios_fewer_requests_and_more_aligned']}**.")
    lines+=['','Full action statuses, refusal reasons, inventory-check identities, source hashes and artifact hashes are in `reviews/optimization_loop_20261109/{inventory,duration_latency}/`. The duration round reconciles all four earlier base-latency arms. Both old reference and unrestricted100ms policy match the preceding soft-pull study.','',
        '## Limits and next work','',
        '- The factual inventory path is exogenous. A changed fill path would change it. Positive exposure mechanics cannot establish lower realized inventory or preserve trading profits.',
        '- The historical fill-clock gate remains closed; the observer uses exact notice time, never an inferred match timestamp. Protected Nov9–13 remain unread.',
        '- The separately implemented Rust A2 paired-order engine is not the order manager in this test. B4 parity/integration remains necessary.',
        '- Calibrated fills or a controlled forward experiment are needed before economic selection or deployment. No policy is promoted by this report.',
        '- A softer inventory-range restriction is a possible next separately declared experiment if strict no-flip gating sacrifices too much quote continuity.','',
        'Research branch `research/predict-opt-loop-20261109`. Reproduction: `pf_native_inventory_events` observer, `pf_inventory_grace_study declare/run ROUND`, `pf_inventory_grace_verify ROUND`, and `pf_optimization_report`, serially under `run_light.sh`. Declarations are committed before outcomes; completed output paths refuse overwrite.']
    path=a.HERE.parent/'handoffs/20261109_optimization_loop.md';path.write_text('\n'.join(lines)+'\n')
    plan=json.loads((s.REVIEW/'plan.json').read_text());plan['status']='completed_pass';plan['as_of']=out['as_of'];plan['findings']='findings.json'
    for row in plan['loop']:row['status']='complete';row['receipt']=f"{row['round']}/receipt.json";row['verification']=f"{row['round']}/verification.json"
    # The register is mutable; immutable receipts use a.write(exclusive create).
    (s.REVIEW/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    print(json.dumps(dict(population=out['population'],comparisons=comparisons),indent=2))


if __name__=='__main__':main()
