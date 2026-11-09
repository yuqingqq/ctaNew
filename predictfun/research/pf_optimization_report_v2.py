"""Superseding loop findings including the exploratory deadline-order round."""
import datetime as dt
import json
import subprocess
from . import pf_inventory_grace_study as s
from . import pf_deadline_grace_study as t
from .pf_optimization_report import verified_link


def verify_deadline(expected,v):
    if v['receipt_sha256']!=expected or v['maximum_grace_ms']!=50 or not v['original_reference_reconciled_all_latencies'] or not all(v['controls'].values()):raise ValueError('deadline verification fails')


def main():
    a=s.a;outpath=s.REVIEW/'findings_v2.json'
    if outpath.exists():raise ValueError('findings_v2 exists')
    old=json.loads((s.REVIEW/'findings.json').read_text())
    parent=json.loads((s.DEST/'duration_latency/receipt.json').read_text())
    for round_name in s.ROUNDS:
        dest=s.DEST/round_name;r=json.loads((dest/'receipt.json').read_text());v=json.loads((dest/'verification.json').read_text());verified_link(a.sha(dest/'receipt.json'),v)
        for p,h in r['artifacts'].items():
            if a.sha(dest/p)!=h:raise ValueError('previous artifact differs')
    new=json.loads((t.DEST/'receipt.json').read_text());v=json.loads((t.DEST/'verification.json').read_text());verify_deadline(a.sha(t.DEST/'receipt.json'),v)
    for p,h in new['artifacts'].items():
        if a.sha(t.DEST/p)!=h:raise ValueError('deadline artifact differs')
    control=dict(receipt_sha256='good',maximum_grace_ms=50,original_reference_reconciled_all_latencies=True,controls={'positive':True})
    verify_deadline('good',control)
    try:verify_deadline('wrong',control)
    except ValueError:refused=True
    else:raise ValueError('wrong-receipt control accepted')
    comparisons={}
    for latency,bb in new['summary'].items():
        q=bb['update_first50'];base=bb['reference'];u=parent['summary'][latency]['all100'];timer=parent['summary'][latency]['all50']
        oldgain=u['aligned_lane_ms']-base['aligned_lane_ms'];oldextra=u['elsewhere_lane_ms']-base['elsewhere_lane_ms']
        comparisons[latency]=dict(requests=q['requests'],request_change_pct=q['request_change_pct'],aligned_change_pct=q['aligned_change_pct'],
            off_target_change_pct=q['off_target_change_pct'],active_shares_change_pct=q['active_shares_change_pct'],reserved_change_pct=q['reserved_change_pct'],
            coverage_gain_retained_vs100_pct=100*(q['aligned_lane_ms']-base['aligned_lane_ms'])/oldgain,
            extra_off_target_reduction_vs100_pct=100*(1-(q['elsewhere_lane_ms']-base['elsewhere_lane_ms'])/oldextra),
            equal_clock_reopens=new['equal_clock_reopens'][latency],same_price_reopens=q['same_price_reopens'],retained_actions=q['retained_actions'],
            fewer_requests_and_more_aligned=q['requests']<base['requests'] and q['aligned_lane_ms']>base['aligned_lane_ms'],
            fewer_requests_than_timer_first50=q['requests']<timer['requests'],
            more_aligned_than_timer_first50=q['aligned_lane_ms']>timer['aligned_lane_ms'],
            lower_off_target_than100=q['elsewhere_lane_ms']<u['elsewhere_lane_ms'],
            active_range_extra_share_ms_change_pct=q['active_range_extra_share_ms_change_pct'])
    out=dict(as_of=dt.datetime.now(dt.timezone.utc).isoformat(),supersedes='findings.json',supersedes_sha256=a.sha(s.REVIEW/'findings.json'),
        reason='add exploratory update-before-expiry round; retain unchanged first-two-round numbers; correct mutable register publication after exclusive-create status-write failure',
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=a.HERE,text=True).strip(),source_sha256=a.sha(a.HERE/'pf_optimization_report_v2.py'),
        population=new['population'],prior_findings=old['comparisons'],deadline_order=comparisons,
        all_latencies_fewer_requests_more_aligned_and_less_off_target_than100=all(q['fewer_requests_and_more_aligned'] and q['lower_off_target_than100'] for q in comparisons.values()),
        receipt_sha256=a.sha(t.DEST/'receipt.json'),verification_sha256=a.sha(t.DEST/'verification.json'),
        controls=dict(correct_verification_accepted=True,wrong_receipt_refused=refused),multiplicity=21,
        limitations=new['limitations'],next=['A2/B4 native paired-order integration and parity','calibrated fills/forward observations for economic valuation','inventory-range follow-up is untested; no live promotion'])
    a.write(outpath,out);a.write(s.DEST/'findings_v2.json',out)
    q=comparisons['base']
    lines=['# Optimization loop, superseding findings v2 — 2026-11-09','',
        f"As-of **{out['as_of']}**. **220 consumed Nov8 windows**, **279,059 original decisions**, **13,077 neutral touch-pull actions** per arm/scenario; zero excluded windows. One partial day, no confidence interval or independent validation.",'',
        'The loop completed three declared rounds. The main new candidate is a **50 ms maximum grace that processes updates at the exact deadline before expiring**. If the quote remains off, cancellation still occurs at50ms; a later event cannot undo it. This changes event priority, not timestamps or the maximum grace.','',
        '| Policy, base latency | Request change | Original aligned time change | Off-target time change | Active share-time change |',
        '|---|---:|---:|---:|---:|']
    base_rows=[('Reference',parent['summary']['base']['reference']),('100 ms unrestricted',parent['summary']['base']['all100']),
        ('100 ms individually reducing',parent['summary']['base']['individual100']),('100 ms full reservation room',parent['summary']['base']['reserved100']),
        ('50 ms timer first',parent['summary']['base']['all50']),('25 ms timer first',parent['summary']['base']['all25']),('50 ms updates first',new['summary']['base']['update_first50'])]
    for name,z in base_rows:lines.append(f"| {name} | {z['request_change_pct']:+.2f}% | {z['aligned_change_pct']:+.2f}% | {z['off_target_change_pct']:+.2f}% | {z['active_shares_change_pct']:+.2f}% |")
    lines+=['',
        f"At base latency, updates-first50ms retains **{q['coverage_gain_retained_vs100_pct']:.2f}%** of the100ms coverage gain while reducing its **additional off-target quote time by {q['extra_off_target_reduction_vs100_pct']:.2f}%**. It bridges **{q['equal_clock_reopens']:,}** same-price reopens stamped exactly at the deadline. No claim of higher trading P&L follows from these metrics.",'',
        '## What each loop round found','',
        '1. **Inventory gating is restrictive.** Individually reducing grace retained4.86% of unrestricted coverage gain (208 admissions); requiring enough room for all same-side reservations retained0.68% (29). Both added no non-reducing share-time at base latency. This protects the conditional exposure measure but sacrifices most quote-continuity benefit.',
        '2. **Duration alone was misleading.** Timer-first50ms had zero same-price bridges in all three latency scenarios; timer-first25ms added exposure with essentially no base-latency coverage gain. The guard updates on a50ms grid, so expiry priority needed a separate experiment.',
        '3. **Equal-clock ordering is material.** The follow-up processes every recorded event with the exact deadline timestamp, then cancels any order still under grace. Events after the deadline are never used to extend it. This is an exploratory policy discovered from round2, not an independently validated result.','',
        '| Assumed latency | Request change | Original aligned time change | Off-target time change | 100ms coverage gain retained | Extra off-target time removed vs100ms |',
        '|---|---:|---:|---:|---:|---:|']
    for latency,z in comparisons.items():lines.append(f"| {latency} | {z['request_change_pct']:+.2f}% | {z['aligned_change_pct']:+.2f}% | {z['off_target_change_pct']:+.2f}% | {z['coverage_gain_retained_vs100_pct']:.2f}% | {z['extra_off_target_reduction_vs100_pct']:.2f}% |")
    lines+=['',
        'Latency assumptions are create / accepted-cancel / pending-create-cancel: base40/22/95ms, faster20/10/50ms, slower80/50/160ms, all with1s reservation grace. Each scenario has its own paired reference. These are assumed stress cases, not venue quantiles or separate days.',
        f"Computed predicate—fewer requests and more original aligned time versus reference, plus less off-target time than100ms, in all three cases: **{out['all_latencies_fewer_requests_more_aligned_and_less_off_target_than100']}**.",'',
        '## Verification and remaining limits','',
        '-37 combined controls pass, including equal-clock reopen, hard/off cancellation at deadline, rejection of a later reopen, inventory reversal/overshoot, pending cancellation reservations and exact exposure accounting.',
        '-Native observer supplement captures93 fill-notice states at0–65s; original decisions/requests/books/grid probes remain exact. Actual fill notices update only the factual inventory path, never hypothetical orders.',
        '-Artifact checks reconcile prior baselines, all first-two-round overlap, action identities, exact native inventory references, maximum50ms grace, coverage gain/loss and summaries. Corrupted counts and mismatched verification links are refused.',
        '-The factual inventory path is exogenous and historical match timestamps remain inadequate for calibrated fill/P&L tests. Active exposure still increases; this does not prove lower inventory variance, favorable settlement inventory, or preserved trading profits.',
        '-The model uses the earlier research sizing planner, not the new Rust A2 paired-order implementation. A2/B4 integration and calibrated fills/forward evidence remain necessary. No live code/deployment, frozen-engine, loss-budget or protected Nov9–13 changes.',
        '-Cumulative distinct policies21. The first two rounds were declared before outcomes; deadline ordering was separately declared after observing round2. No candidate is promoted automatically.',
        '-This v2 supersedes the earlier two-round findings. Their numbers remain valid. An exclusive-create helper refused the mutable register update after the first report had been written; that publication issue is corrected here and the old log/outputs remain preserved.','',
        'Register and receipts: `reviews/optimization_loop_20261109/plan.json`, `findings_v2.json`, and `{native_inventory,inventory,duration_latency,deadline_order}/`. Full first-two-round table: `handoffs/20261109_optimization_loop.md`. Research branch: `research/predict-opt-loop-20261109`.']
    (a.HERE.parent/'handoffs/20261109_optimization_loop_v2.md').write_text('\n'.join(lines)+'\n')
    plan=json.loads((s.REVIEW/'plan.json').read_text());plan.update(status='completed_pass',as_of=out['as_of'],findings='findings_v2.json')
    for row in plan['loop']:row.update(status='complete',receipt=f"{row['round']}/receipt.json",verification=f"{row['round']}/verification.json")
    plan['next_evidence']=out['next'];(s.REVIEW/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    print(json.dumps(dict(comparisons=comparisons,predicate=out['all_latencies_fewer_requests_more_aligned_and_less_off_target_than100']),indent=2))


if __name__=='__main__':main()
