"""Publish the frozen opening-only replay; no strategy or threshold changes."""
import hashlib,json,re,shutil
from pathlib import Path
H=Path(__file__).resolve().parent;R=H.parents[2];D=R/'data/predict_fun/research/opening_entry_pilot_20261124'
c=json.loads((H/'checks.json').read_text());s=json.loads((D/'summary.json').read_text());rows=c['rows'];tab=[];opening=[];detail=[]
for r in rows:
 m=r['m'];a=r['baseline'];b=r['candidate'];t=r['opening_batch_ms']
 tab.append(f"|{m}|{a['total_c']/100:+.2f}|{b['total_c']/100:+.2f}|{r['delta_c']/100:+.2f}|")
 opening.append(f"|{m}|{t/1000 if t is not None else 'no submission'}|{(t+50)/1000 if t is not None else 'not reached'}|{b['first_post_sides']}|{a['first_minute']['total_c']/100:+.2f}|{b['first_minute']['total_c']/100:+.2f}|")
 detail.append(f"|{m}|{a['completed_c']/100:+.2f} / {b['completed_c']/100:+.2f}|{a['residual_c']/100:+.2f} / {b['residual_c']/100:+.2f}|{a['filled_shares']:.1f} / {b['filled_shares']:.1f}|{100*a['both_active_grid_share']:.1f}% / {100*b['both_active_grid_share']:.1f}%|{a['posts']} / {b['posts']}|{a['pulls']} / {b['pulls']}|")
matched=sum(r['candidate']['completed_c']-r['baseline']['completed_c'] for r in rows);residual=sum(r['candidate']['residual_c']-r['baseline']['residual_c'] for r in rows)
a_q=sum(r['baseline']['filled_shares'] for r in rows);b_q=sum(r['candidate']['filled_shares'] for r in rows)
selected=next(r for r in rows if r['m']==3789484)
text=f'''# Opening-entry-only replay

2026-11-24, frozen before replay at b1898f0b4. USER correction: change only
market-open entries, not all later posting decisions. Three consumed examples,
no new day, fitted model, threshold change or validation claim.

## Exact scope

Apply the same directional gate while waiting for the first actual submission,
and throughout that submission's50ms decision tick (both sides and rungs).
From the next tick, permanently use the original PressureGuard entry rule.
The filter never rearms after a fill, cancellation, refusal or empty book of
our orders. A submitted-but-refused first batch still ends the filter. There
is no fixed waiting period. If no first batch arrives, keep the no-entry case.
All three cases here submit, as the table shows.

The inventory sign-crossing correction is identical in baseline and candidate.
Live-order cancellation, quote prices/levels, sizing, caps, notices, latency,
queue and tape assumptions remain the same. Later **rules** are identical;
later **actions and fills** can differ because the opening changed inventory,
reservations and queue history. This is a whole-policy feedback contrast.

## Results

USD, full supported interval to opening+298s, identical within-market mark,
zero maker fees/rebates as in the parent simulator. Not binary settlement P&L.

|Market|Fixed inventory, default opening|Opening-only gate|Difference|
|---|---:|---:|---:|
{chr(10).join(tab)}

Pooled value: ${c['pooled_baseline_c']/100:+.2f} ->
${c['pooled_candidate_c']/100:+.2f}, difference
${c['pooled_delta_c']/100:+.2f}. Better in{c['better_markets']}/{c['markets']}
markets; positive absolute value in{c['positive_markets']}/{c['markets']}.
The pooled strategy still loses. Completed-turnover delta is
${matched/100:+.2f}, residual-value delta${residual/100:+.2f}.
Filled shares:{a_q:.1f} -> {b_q:.1f} ({100*(b_q/a_q-1):+.1f}%).

|Market|First batch seconds|Original entries resume seconds|First sides (0 bid,1 ask)|Default first-minute USD|Opening-only first-minute USD|
|---|---:|---:|---|---:|---:|
{chr(10).join(opening)}

In the user-selected3789484 example, only the opening filter lasts through
{selected['opening_batch_ms']}ms; the original entries resume at
{selected['opening_batch_ms']+50}ms. The full-market difference is
${selected['delta_c']/100:+.2f}. No maker3 decision policy is inferred.

The earlier all-entry candidate's pooled+$21.27 is retained in its original
receipt. It answers a different question and cannot be cited as the result
of changing the opening alone. That restriction is now superseded in the
active handoff; the earlier frozen source and results are unchanged.

Detailed totals below show default / opening-only:

|Market|Completed FIFO USD|Residual USD|Filled shares|Both sides active,50ms grid|Posts|Pulls|
|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(detail)}

FIFO uses all fills from the common interval, with no resets at the first
minute.50ms/1s/5s/30s markout values and native/common-cutoff share denominators
are retained in the summary; these do not replace whole-turnover accounting.

## Checks and limitations

Exact saved tape fingerprints required before each run. Source/input hashes
include all reused FIXED_DEFAULT result/array bytes. The disabled opening
wrapper reproduces the fixed baseline on market3702960, including fingerprint,
fills, orders, requests, commands, notices, guard log and value. The other
checked baselines are reused unchanged, not recomputed just for repetition.

All three new arms pass the parent accounting, quantity, known-inventory,
reservation and opening-gate checks. Every one of{c['later_slot_calls_verified']}
post-opening slot callbacks was independently checked against the original
inventory-sizing/pressure rule. Four deliberately corrupted scope/action/
cutoff/log cases are refused. Synthetic controls cover both opening rungs,
next-tick entry with a flat signal, no rearming after refusal/closure and
unchanged live handling. No code or checker amendment occurred after freeze.

Selected consumed examples with point estimates only. No matched-activity
null, calibrated prediction probability, causal maker-behaviour or robust
profitability claim. M50 execution and median-delay information clocks remain
assumptions. The new state can alter many later fills even though their policy
is unchanged; do not attribute every P&L difference to the first fill alone.

## Next priority

Inspect the first completed inventory cycles and subsequent turnover in the
remaining losing market, using these saved opening-only versus fixed-default
ledgers. Separate the benefit of avoiding the opening sweep from losses that
persist under the unchanged continuation. Keep this opening scope fixed;
do not reinstate an all-entry gate or start threshold tuning. Saved-evidence
inspection may use only the remaining allowance; another replay needs a
separately bounded resource decision.
'''
(H/'report.md').write_text(text);shutil.copyfile(D/'summary.json',H/'summary.json');shutil.copyfile(D/'disabled_parity.json',H/'disabled_parity.json')
(H/'SHA256SUMS').write_text(''.join(hashlib.sha256(p.read_bytes()).hexdigest()+'  '+str(p.relative_to(R))+'\n' for p in sorted(D.glob('*')) if p.is_file()))
tags={'opening_prepare','opening_run','opening_check'};measured=0.;rr=[]
for p in H.glob('*.log'):
 for line in p.read_text().splitlines():
  m=re.search(r'tag=(\S+).*cpu_user_s=([\d.]+)',line)
  if m and m[1] in tags:measured+=float(m[2]);rr.append(dict(tag=m[1],cpu_s=float(m[2])))
budget=dict(previous_packet_upper_cpu_s=139.43,measured_cpu_s=measured,publication_reserve_s=2,packet_upper_cpu_s=139.43+measured+2,packet_cap_cpu_s=180,runs=rr)
assert budget['packet_upper_cpu_s']<=180
(H/'resources.json').write_text(json.dumps(budget,indent=2)+'\n')
print('\n'.join(tab));print('\n'.join(opening));print('\n'.join(detail));print(json.dumps(budget))
