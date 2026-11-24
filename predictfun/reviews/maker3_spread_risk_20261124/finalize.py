"""Local receipt publication; preserve non-owned state text and index hunks."""
import hashlib,json,re,subprocess
from pathlib import Path
H=Path(__file__).resolve().parent;R=H.parents[2]
lines=[s for s in (R/'predictfun/research/run_light.log').read_text().splitlines() if s.startswith('2026-11-24') and 'tag=msr_' in s and 'slot=codex' in s]
(H/'resource.log').write_text('\n'.join(lines)+'\n')
cpu=sum(float(re.search(r'cpu_user_s=([\d.]+)',x)[1]) for x in lines)
assert cpu<45,cpu
resources=dict(asof='2026-11-24',packet_cap_cpu_s=540,charged_upper_cpu_s=60,wrapper_user_s=cpu,peak_rss_mib=max(int(re.search(r'peak_rss_mib=(\d+)',x)[1]) for x in lines),task_coreh=2.08822+60/3600,line_coreh=5.39892+60/3600,task_cap=2.35,line_cap=5.65,reserve_coreh=.05,browser_resource='browser_resource_v2.log',validation=False)
(H/'resources.json').write_text(json.dumps(resources,indent=2)+'\n')
note='''<!-- MAKER3_SPREAD_RISK_20261124_CODEX -->
## 2026-11-24 — maker3's late turnover, spread and risk chart

[Receipt](reviews/maker3_spread_risk_20261124/report.md); [chart](reviews/maker_dashboard_20261124/maker3-research.html), existing localhost8765 tunnel. USER asks inventory rules, trending fills, late/wide-spread profits, sizing/skew/takes. All19 consumed complete markets, 2,759 actor legs. Frozen2b540b096; canonical mixed-case address adapter41b546811, after initial build refused before outputs. Completed turnover+$859.50; residual less remaining fees−$1,210.32; END−$350.81. Late exits+$953.10 vs early−$93.59, entries may precede180s. In3789484 late closes+$564.07 vs early−$55.44. Both legs>4c explain only0.41% pre-window/5.31% upper of positive completed value: mainly-wide hypothesis not supported. Edge-versus-midmove attribution is highly clock-sensitive.

Public-price-only trend3838531,245–275s,+71c:76legs,15.38buys/1,621.36YES-equivalent sells, recorded net−64.85→−1,670.83; two NO-buy takes add negative recorded exposure. Not every take reduces. Full-market net still+$113.34; never equate the window with a losing whole market. Native/YES prices, bounded times and book ages explicit. Our baseline/wider overlays exist on3prior markets only.

5,700neutral seconds,5predeclared taker conditions x D0/lag1,500raw-side shifts. No stable positive trigger: no-passive-reduction and executable-profit screens have counts but no positive association; burst/BTC-adverse/wide thin. No inferred private skew/threshold or sizing rule. Prior33signed quotes/28usable windows remain the size evidence, both10/220 on adding/reducing sides. Our wider S500/R250/C1000 uses quantity skew, no price skew; soft restriction and outstanding-reservation hard cap separate; wider IOC did not trigger in prior3 paths.

Checks2,759raw legs/156cells/10screens/4corruptions. Desktop/mobile chart pass after UI overflow correction; scientific data unchanged. No fit/replay/new days/validation. Upper charge60CPU sec; cumulative task~2.10489/line5.41559, caps2.35/5.65, .05reserve. Local only; other-session edits preserved.

ONE next priority remains unchanged WIDE_PASSIVE vs opening-only baseline across all19consumed complete markets, paired turnover/residual/risk/churn, before adding a sizing switch or taker trigger. Standing bounded CPU authorization suffices; no new days/live.
<!-- /MAKER3_SPREAD_RISK_20261124_CODEX -->

'''
state='''maker3_spread_risk_20261124_codex:
  report: predictfun/reviews/maker3_spread_risk_20261124/report.md
  chart: predictfun/reviews/maker_dashboard_20261124/maker3-research.html
  freeze: 2b540b096
  adapter: 41b546811
  consumed_complete_markets: 19
  raw_actor_legs: 2759
  completed_net_usd: 859.5039737
  residual_less_unclosed_fees_usd: -1210.3151442
  marked_total_usd: -350.8111704
  mainly_wide_spreads_supported: false
  stable_positive_active_take_trigger_identified: false
  private_size_skew_cap_identified: false
  validation: false
  new_replay: false
  packet_charged_upper_cpu_s: 60
  task_coreh: 2.10489
  line_coreh: 5.41559
  task_cap_coreh: 2.35
  line_cap_coreh: 5.65
  next: fixed_wider_band_paired_replay_on_19_consumed_complete_markets
'''
assert not subprocess.check_output(['git','diff','--cached','--name-only'],cwd=R).strip(),'pre-existing staged edits'
for p,prefix in [('predictfun/HANDOFF.md',note),('predictfun/FINDINGS.md',note),('predictfun/STATUS.yml',state)]:
 path=R/p;old=path.read_text();assert prefix.splitlines()[0] not in old
 path.write_text(prefix+old)
 base=subprocess.check_output(['git','show','HEAD:'+p],cwd=R)
 blob=subprocess.check_output(['git','hash-object','-w','--stdin'],input=prefix.encode()+base,cwd=R).decode().strip()
 subprocess.run(['git','update-index','--cacheinfo','100644',blob,p],cwd=R,check=True)
# Output hashes are receipt-owned; input identities remain in the frozen manifest.
files=[p for p in H.iterdir() if p.is_file() and p.name!='SHA256SUMS']+[R/'predictfun/reviews/maker_dashboard_20261124/maker3-research.html']
(H/'SHA256SUMS').write_text(''.join(hashlib.sha256(p.read_bytes()).hexdigest()+'  '+str(p.relative_to(R))+'\n' for p in sorted(files)))
print(json.dumps(resources))
