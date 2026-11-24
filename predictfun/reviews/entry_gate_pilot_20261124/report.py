"""Descriptive publication after the frozen replay/check; no rule changes."""
import json,shutil,hashlib
from pathlib import Path
H=Path(__file__).resolve().parent;R=H.parents[2];O=R/'data/predict_fun/research/entry_gate_pilot_20261124'
s=json.loads((O/'summary.json').read_text());checks=json.loads((H/'checks.json').read_text())
rows=s['rows'];table=[];detail=[]
for mid in [3702960,3789484,3877472]:
 d={r['arm']:r['metrics'] for r in rows if r['m']==mid};a=d['ORIGINAL'];b=d['FIXED_DEFAULT'];c=d['DIRECTIONAL_ENTRY']
 table.append(f"|{mid}|{a['total_c']/100:+.2f}|{b['total_c']/100:+.2f}|{c['total_c']/100:+.2f}|{(c['total_c']-b['total_c'])/100:+.2f}|")
 for name in ['FIXED_DEFAULT','DIRECTIONAL_ENTRY']:
  v=d[name];detail.append(f"|{mid}|{name}|{v['completed_c']/100:+.2f}|{v['residual_c']/100:+.2f}|{v['filled_shares']:.1f}|{v['peak_abs_inventory']:.1f}|{v['both_active_grid_share']*100:.1f}%|{v['posts']}|{v['pulls']}|{v['first_post_ms']} / {v['first_post_sides']}|")
pooled={}
for arm in ['ORIGINAL','FIXED_DEFAULT','DIRECTIONAL_ENTRY']:
 rr=[x['metrics'] for x in rows if x['arm']==arm]
 pooled[arm]={k:sum(r[k] for r in rr) for k in ['total_c','completed_c','residual_c','filled_shares','posts','pulls']}
a=pooled['FIXED_DEFAULT'];b=pooled['DIRECTIONAL_ENTRY'];fraction=1-b['filled_shares']/a['filled_shares']
n=len(checks['contrasts']);better=sum(x['entry_delta_c']>0 for x in checks['contrasts']);positive=sum(x['entry_total_c']>0 for x in checks['contrasts'])
case={r['arm']:r['metrics'] for r in rows if r['m']==3789484}
report=f'''# Directional-entry replay: three consumed dashboard markets

2026-11-24. Frozen implementation/manifest/declaration/checker: fa9d035d2.
One new unfitted candidate; nine full-market arms, no new days or model fit.
All9 saved results pass independent checks; the ORIGINAL arm reproduces the
saved policy on all3 markets. Seven deliberate corruptions are refused.

## Result

All amounts below are USD, pre-expiry marked value, zero maker fees/rebates
as in the original simulator. The same END mark and full supported interval
apply within each market. These are three selected illustrations, not validation.

|Market|Original as saved|Fixed inventory, default entry|Fixed inventory, directional entry|Entry delta vs fixed default|
|---|---:|---:|---:|---:|
{chr(10).join(table)}

Pooled fixed-default value is${a['total_c']/100:+.2f}; directional entry is
${b['total_c']/100:+.2f}, a${(b['total_c']-a['total_c'])/100:+.2f} difference.
The gate is better in{better}/{n} markets; {positive}/{n} candidate markets has positive
absolute value. The positive pooled result is concentrated in market3789484.
Filled shares fall from{a['filled_shares']:.1f} to{b['filled_shares']:.1f}
({100*fraction:.1f}% lower). No matched participation null was run, so this
does not establish selection skill beyond reducing exposure.

|Market|Arm|Completed FIFO USD|Residual USD|Filled shares|Peak absolute inventory|Both sides active,50ms grid|Posts|Pulls|First post ms / sides(0 bid,1 ask)|
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
{chr(10).join(detail)}

Completed FIFO delta is${(b['completed_c']-a['completed_c'])/100:+.2f}; residual
delta is${(b['residual_c']-a['residual_c'])/100:+.2f}. The whole-policy gain
must not be attributed entirely to an initial-fill prediction. The accounting
is whole-path FIFO, not claimed intent; residual exposure and turnover remain
separate. Completed matched shares are counted once per buy/sell pair.

## The user-selected opening example

Market3789484: initial entry moves from0ms/both sides to
{case['DIRECTIONAL_ENTRY']['first_post_ms']}ms/sides{case['DIRECTIONAL_ENTRY']['first_post_sides']}.
First-minute common-mark value is
${case['FIXED_DEFAULT']['first_minute']['total_c']/100:+.2f} for fixed default,
${case['DIRECTIONAL_ENTRY']['first_minute']['total_c']/100:+.2f} for the gate.
Full-market values are${case['FIXED_DEFAULT']['total_c']/100:+.2f} and
${case['DIRECTIONAL_ENTRY']['total_c']/100:+.2f}. This avoids quoting both sides
at tick0. Its completed first-minute matched volume is only
{case['DIRECTIONAL_ENTRY']['first_minute']['matched_shares']:.2f} shares,
against{case['FIXED_DEFAULT']['first_minute']['matched_shares']:.2f} for the
fixed default. It does not reproduce maker3's private rule or prove its decisions
used this signal. Other gate/post/inventory decisions also change later.

## What actually changed

An existing favourable Binance-pressure condition permits a new risk-increasing
quote. Otherwise the proposed size is restricted to unreserved reduction of
known inventory; flat/missing signals do not authorize new risk. No probability
or expected-value model is fitted. Thresholds/window are the accepted guard's,
not chosen from this replay. Existing live-order cancellations are unchanged:
a flat entry signal alone never cancels a resting order.

Both primary comparison arms fix the known skipped-deadband inventory bug.
The original stopped quoting for long intervals after sign-crossing fills;
restoring trading can expose more losing fills. A worse value after this fix
is not a reason to retain incorrect inventory logic. Its contribution is
explicit in checks.json, separate from the entry contrast.

## Reliability and limits

The original loader, tape/mark identity, current queue, received signal cache,
10ms activation/cancel,100ms notices,50ms grid and M50 assumed execution clock
are unchanged. Local wrappers leave frozen implementation files untouched.
The input/source manifest was committed before any replay. Checks reconcile
cash, inventory, END value, fills, per-order quantities, entry tick and side,
known inventory from noticed fills, and reservations from prior commands and
closures; corrected modes never block the wrong inventory side. All three
original fingerprints/fills/orders/requests/values match the saved controls.

The initial preparation refused on a wrong imputation-file path before a
manifest or replay existed. The adapter was corrected to the same Tape.offset
source and committed in the freeze; both logs remain. No run amendment or
economic rule correction was made after the freeze.

The checker does not estimate sampling uncertainty, calibrate execution
conditional value, or isolate signal selection from reduced participation.
An empirical significance null is intentionally absent: this is a bounded
feedback-policy diagnostic, with no predictive-skill claim. Markout values
at50ms/1s/5s/30s and valid share denominators are retained in summary.json,
with the declared shared END−30s cutoff; these are not the primary score.

## Next priority

Keep this exact rule frozen. Before tuning or expanding it, compare it with
a declared activity-matched entry control on the same markets, using complete
turnover and residual value. This separates useful directional timing from
the benefit of doing less. The present replay justifies that question; it
does not justify deployment, more aggressive size, or a claim of profitability.
Any further replay must fit a separately bounded resource decision; no extra
candidate is launched by this receipt.
'''
(H/'report.md').write_text(report)
shutil.copyfile(O/'summary.json',H/'summary.json')
(H/'SHA256SUMS').write_text(''.join(hashlib.sha256(p.read_bytes()).hexdigest()+'  '+str(p.relative_to(R))+'\n' for p in sorted(O.glob('*')) if p.is_file()))
tags=['entry_gate_prepare','entry_gate_prepare2','entry_gate_run','entry_gate_check']
import re
charged=0.;logs=[]
for p in sorted(H.glob('*.log')):
 for line in p.read_text().splitlines():
  m=re.search(r'tag=(\S+).*cpu_user_s=([\d.]+)',line)
  if m and m[1] in tags:charged+=float(m[2]);logs.append(dict(tag=m[1],cpu_user_s=float(m[2])))
resources=dict(previous_packet_upper_cpu_s=87.11,measured_cpu_s=charged,publication_and_operating_reserve_s=3.,
 packet_upper_cpu_s=87.11+charged+3,packet_cap_cpu_s=180,runs=logs,new_strategy_replay_authorized_by_latest_user=True,new_days=False)
(H/'resources.json').write_text(json.dumps(resources,indent=2)+'\n')
print('\n'.join(table));print(json.dumps(dict(pooled=pooled,resources=resources,first_minute_3789484={k:v['first_minute'] for k,v in case.items()}),indent=2))
