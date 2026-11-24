import hashlib,json,re,subprocess
from pathlib import Path
H=Path(__file__).resolve().parent;R=H.parents[2];A=R/'predictfun/reviews/maker3_size_inventory_20261124'
def usage(folder):
    wall=user=0.
    for p in folder.glob('*.log'):
        for line in p.read_text().splitlines():
            if 'slot=codex' in line and 'wall_s=' in line:
                wall+=float(re.search(r'wall_s=([\d.]+)',line).group(1));user+=float(re.search(r'cpu_user_s=([\d.]+)',line).group(1))
    return dict(one_core_wall_s=wall,cpu_user_s=user)
ru=usage(H);au=usage(A);assert ru['one_core_wall_s']<105 and au['one_core_wall_s']<8
for folder,u,upper,cap in [(H,ru,110,720),(A,au,10,60)]:
    u.update(charged_upper_cpu_s=upper,cap_cpu_s=cap,includes='all attempts plus final publication reserve')
    (folder/'resources.json').write_text(json.dumps(u,indent=2)+'\n')
    p=folder/'report.md';p.write_text(p.read_text()+f'\nResource charge upper: {upper}CPU seconds of {cap}; all failures and checks included. Standing USER small-CPU authorization; no push or live use.\n')
    files={str(p.relative_to(folder)):hashlib.sha256(p.read_bytes()).hexdigest() for p in folder.iterdir() if p.is_file() and p.name not in ['SHA256SUMS','finalize.log']}
    (folder/'SHA256SUMS').write_text(''.join(f'{v}  {k}\n' for k,v in sorted(files.items())))
body='''## 2026-11-24 — final-minute quoting verified; simple active inventory reduction loses

[Maker3 size/position audit](reviews/maker3_size_inventory_20261124/report.md); [frozen IOC pilot](reviews/inventory_take_pilot_20261124/report.md), freeze7626e320f. USER asks to quote/accumulate/take like maker3. In3789484 OPENING_ONLY already quotes either side53.25/58final seconds, posts199 and fills1635.35sh. No scheduled final-minute cutoff. The historical ORIGINAL has an inventory-mode bug and must stay distinguished from corrected active baselines.

Maker3 signed size known only33/61373filled hashes:16at10/17at220, selected sample; one220hash in3789484. Per-level aggregate and private cap unknown. Example captured flow−562→+934, take states−549to+622, maximum across934markets~3000 but only one>=2990 (not a proven cap). At82.838s maker3 cuts positive exposure while we are short: copying its action would add risk for us.

Implemented/tested one unfitted IOC rule at our soft inventory thresholds, with matched100/220size arms and proportional caps200/440 (soft100/220, release50/110); existing opening-only entry unchanged. Causal known-state inputs,10ms effect/100ms notice, reservations/own-cross, depth-limited IOC and2%*min(p,1-p) cash-equivalent fee. Reducing resting orders preserved. Same3consumed markets; baseline parity3/3,12ledger checks,124IOC requests independently reconciled,15synthetic groups and12corruptions. External flow remains exogenous; conservative net-depth budget is a sensitivity, not measured achievable execution. One independent checker mistake (refused orders treated as live) corrected with raw evidence; replay unchanged.

END net USD pooled: PASSIVE100−121.32 /TAKE100−171.75 /PASSIVE220−161.00 /TAKE220−318.15. Mean absolute inventory79.1→53.6 and149.7→115.3; whole-book gross completed turnover worsens58.24/153.65USD, residual improves22.85/14.51, fees15.03/18.00. Both active variants worse before fees too. All89filled takes reduce true inventory in these runs; hidden-fill counterexample tested. Reject this simple threshold rule on these selected examples; do NOT infer selective maker3 taking cannot work or promote220 from two positive examples. All pooled negative; no validation/tuning.

ONE next priority: saved-ledger attribution of costly active closures versus passive continuation, comparing maker3's selective small trims with our automatic soft-limit-to-release actions. Decide economic worth of active reduction before adding a new threshold or expanding risk. Opening first-quote vs first-position scope stays unchanged. Resource upper audit10CPU +pilot110CPU; standing-approved task cap2.35/line5.65/.05reserve, cumulative charged~2.05489/5.36559. No frozen core source edits or push; other-session changes preserved.
'''
blocks={
'predictfun/HANDOFF.md':'<!-- INVENTORY_TAKE_PILOT_20261124_CODEX -->\n'+body+'<!-- /INVENTORY_TAKE_PILOT_20261124_CODEX -->\n\n',
'predictfun/FINDINGS.md':'<!-- INVENTORY_TAKE_PILOT_20261124_CODEX -->\n'+body+'<!-- /INVENTORY_TAKE_PILOT_20261124_CODEX -->\n\n',
'predictfun/STATUS.yml':'''inventory_take_pilot_20261124_codex:
  report: predictfun/reviews/inventory_take_pilot_20261124/report.md
  size_inventory_audit: predictfun/reviews/maker3_size_inventory_20261124/report.md
  freeze: 7626e320f
  stage: simple_inventory_take_rule_rejected_on_selected_consumed_examples
  example_late_quoting_seconds: 53.25
  example_late_supported_seconds: 58
  maker3_private_cap_identified: false
  source_signed_size_sample: 33
  four_arms_three_markets: true
  passive_100_parity_markets: 3
  ioc_requests_independently_checked: 124
  pooled_net_usd: [-121.315946, -171.746941, -161.002615, -318.152009]
  arm_order: [PASSIVE_100, TAKE_100, PASSIVE_220, TAKE_220]
  validation: false
  audit_upper_cpu_s: 10
  pilot_upper_cpu_s: 110
  task_cap_coreh: 2.35
  line_cap_coreh: 5.65
'''}
for name,block in blocks.items():
    p=R/name;old=p.read_text();assert 'INVENTORY_TAKE_PILOT_20261124_CODEX' not in old and 'inventory_take_pilot_20261124_codex:' not in old
    base=subprocess.check_output(['git','show','HEAD:'+name],cwd=R).decode();p.write_text(block+old)
    blob=subprocess.check_output(['git','hash-object','-w','--stdin'],cwd=R,input=(block+base).encode()).decode().strip()
    subprocess.run(['git','update-index','--cacheinfo','100644',blob,name],cwd=R,check=True)
print('Usage:',ru,au,'; owned state prefixes staged; pre-existing dirty changes preserved')
