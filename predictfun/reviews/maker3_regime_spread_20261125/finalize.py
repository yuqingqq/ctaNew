"""Publish only owned state prefixes, preserving shared unstaged work."""
import hashlib,json,re,subprocess
from pathlib import Path
H=Path(__file__).resolve().parent;R=H.parents[2]
lines=[x for x in (R/'predictfun/research/run_light.log').read_text().splitlines() if x.startswith('2026-11-25') and 'tag=mrs_' in x and 'slot=codex' in x]
(H/'resource.log').write_text('\n'.join(lines)+'\n');cpu=sum(float(re.search(r'cpu_user_s=([\d.]+)',x)[1]) for x in lines);assert cpu<50
resource=dict(asof='2026-11-25',packet_cap_cpu_s=360,charged_upper_cpu_s=60,wrapper_user_s=cpu,peak_rss_mib=max(int(re.search(r'peak_rss_mib=(\d+)',x)[1]) for x in lines),task_coreh=2.104886666666667+60/3600,line_coreh=5.415586666666667+60/3600,task_cap=2.35,line_cap=5.65,reserve_coreh=.05)
(H/'resources.json').write_text(json.dumps(resource,indent=2)+'\n')
note='''<!-- MAKER3_REGIME_SPREAD_20261125_CODEX -->
## 2026-11-25 — volatile exits earn turnover; volatile entries leave costly residuals

[Receipt](reviews/maker3_regime_spread_20261125/report.md); [figure](reviews/maker3_regime_spread_20261125/regimes.png), served localhost8765/maker3-regimes.png. USER requests calm/volatile and narrow/wide spread attribution. Frozen0d620f34e; neutral Oct24–25 q25/q75 bounds c7abba308; phase6fdfce670 before reading economics. Same19 complete consumed markets,17 active on3days;2,759 legs/2,510 FIFO allocations. Predict10s RMS of1s mid changes: calm<=0.9487c,volatile>2.6458c; BTC separate. No new strategy/model/days.

Primary maker->maker EXIT regime: calm−$50/−1.49c per matched share,typical+$208/+1.81c,volatile+$752/+9.98c. Volatile exits59.6%positive passive value,32.5%matchedshares; majority on all3clock assumptions. Volatile/narrow exits alone+$395;wide+$83 on288matchedshares is thin (23hashes). ENTRY regime lacks stable calm-vs-volatile ranking across clocks. Full-book volatile-entry completed+$290 but residual−$706 ->−$416; negative also upper−$713/lag1−$516. FIFO entry is an earlier fill,not posting; no private long-rest/queue policy recovered.

Critical concentration: all-role volatile-exit net Oct24−$85.81,Oct25−$10.32,Oct26+$882.24;Oct27 no maker closes here.3789484+$587.49 and3803131+$268.21 drive it; excluding both−$69.59. Thus useful illustration,not general volatility edge. Early/late results retained; no blanket volatility quote/pull rule. Completed and residual reconcile to existing+$859.50/−$1210.32/−$350.81.

Checks5700feature rows,allparentlegs/closes,3917groupedcells,373phasecells,4corruptions,causal future-perturbation+positivecontrols,long/short/fee identities. No output-driven correction. Conservative60CPU sec; cumulative~2.12155/5.43225,caps2.35/5.65,.05reserve. Parent frozen artifacts and shared dirty state preserved; no push.

ONE next priority: unchanged WIDE_PASSIVE vs opening-only baseline on19consumed complete markets; predeclare attribution by adding/reducing orders and these frozen regimes to determine whether extra inventory room preserves valuable closes or only enlarges residual risk. No new tuned filter or take threshold from these descriptive cells.
<!-- /MAKER3_REGIME_SPREAD_20261125_CODEX -->

'''
status='''maker3_regime_spread_20261125_codex:
  report: predictfun/reviews/maker3_regime_spread_20261125/report.md
  figure: predictfun/reviews/maker3_regime_spread_20261125/regimes.png
  freeze: 0d620f34e
  bounds_freeze: c7abba308
  complete_markets: 19
  active_markets: 17
  observed_active_days: 3
  parent_actor_legs: 2759
  parent_fifo_allocations: 2510
  passive_volatile_exit_net_c_per_matched_share: 9.97606
  passive_calm_exit_net_c_per_matched_share: -1.48687
  volatile_exit_positive_value_majority_all_clocks: true
  volatile_exit_net_excluding_top_two_markets_usd: -69.59368
  volatile_entry_marked_negative_all_clocks: true
  validation: false
  causal_quote_rule_identified: false
  packet_charged_upper_cpu_s: 60
  task_coreh: 2.12155
  line_coreh: 5.43225
  next: fixed_wider_band_19_market_replay_with_adding_vs_closing_attribution
'''
assert not subprocess.check_output(['git','diff','--cached','--name-only'],cwd=R).strip()
for name,prefix in [('predictfun/HANDOFF.md',note),('predictfun/FINDINGS.md',note),('predictfun/STATUS.yml',status)]:
 p=R/name;old=p.read_text();assert prefix.splitlines()[0] not in old;p.write_text(prefix+old)
 head=subprocess.check_output(['git','show','HEAD:'+name],cwd=R);blob=subprocess.check_output(['git','hash-object','-w','--stdin'],input=prefix.encode()+head,cwd=R).decode().strip()
 subprocess.run(['git','update-index','--cacheinfo','100644',blob,name],cwd=R,check=True)
files=[p for p in H.iterdir() if p.is_file() and p.name!='SHA256SUMS']
(H/'SHA256SUMS').write_text(''.join(hashlib.sha256(p.read_bytes()).hexdigest()+'  '+str(p.relative_to(R))+'\n' for p in sorted(files)))
print(json.dumps(resource))
