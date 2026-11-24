"""Save presentation evidence and only this task's state hunks."""
import hashlib,json,re,shutil,subprocess
from pathlib import Path
H=Path(__file__).resolve().parent;R=H.parents[2]
for fn in ['sizes.csv','take_events.csv']:
 shutil.copyfile(R/'data/predict_fun/research/maker3_cap_skew_20261124'/fn,H/fn)
lines=[x for x in (R/'predictfun/research/run_light.log').read_text().splitlines() if 'tag=cs_' in x and x.startswith('2026-11-24T') and 'slot=codex' in x]
(H/'resource.log').write_text('\n'.join(lines)+'\n')
user=sum(float(re.search(r'cpu_user_s=([0-9.]+)',x)[1]) for x in lines)
wall=sum(float(re.search(r'wall_s=([0-9.]+)',x)[1]) for x in lines)
assert wall+2<120 and user<120
(H/'resources.json').write_text(json.dumps(dict(stages=len(lines),cpu_user_s=user,wall_serial_upper_s=wall,charged_upper_s=120,packet_cap_s=540,
 task_approx_coreh=2.05489+120/3600,line_approx_coreh=5.36559+120/3600,task_cap=2.35,line_cap=5.65,standing_reserve_coreh=.05),indent=2)+'\n')
block='''<!-- CAP_MAKER_SIZING_20261124_CODEX -->
## 2026-11-24 — raising the soft inventory band changes turnover; cap alone does not

[Receipt](reviews/maker3_cap_skew_20261124/report.md), freeze787d1a0f6. USER asks higher cap,10/220 sizing vs conditions, skew and selective taking. Same3consumed examples, maximum100shares x2levels. C200→1000 alone EXACT fills/commands/grid parity3/3. Widening S100→500/R50→250 (Qside200 unchanged) gives END USD:3702960 +5.41→+24.38;3789484 +9.73→+248.80;3877472 −136.46→−225.39. Combined−121.32→+47.79; completed turnover−100.27→+100.93, residual−21.05→−53.14. Both-side grid uptime36.7→88.4%; meanabsI79→236; peak496. More generations/pulls: churn not solved. Wider TAKE identical because no threshold crossing; synthetic550-share control fires correctly. No claim taker performance or validated profitability.

Source34signed rank3hashes =33maker (16x10/17x220) +1incomplete taker45.2805 unknown.28maker posting windows usable. Small quotes6alwaysadding/5reducing/4mixed;large3/7/3. Volatility-size rule not established (selected,day-confounded,uncertainposting); private cap/two-sided quote skew unknown. Full5,700neutral seconds/19markets:52opposing-take seconds; no passive reduction prior5s46.2% vsmatched53.0%, prior30s7.7% vs5.9%. Takes often follow passive fills too. In3789484,660same-direction shares before112s and220before200s precede partial trims; compatible with accumulation pressure,not identified motive/skew failure. Native/YES direction and timing variants retained.

Checks12ledgers,3parentparities,68state rows/independentpriorcontext,9corruptions,positiveIOC control. Observation unknown-mid and CSVmissing/None instrument failures preserved/superseded, no replay change. Packetchargedupper120CPU; cumulative~2.08822/5.39892,caps2.35/5.65/.05reserve. No push; all other-session edits preserved.

ONE next priority: fixed WIDE_PASSIVE vs opening-only baseline across all19already-consumed complete markets; same100size,C1000,S500,R250,full-END turnover/residual/risk/participation, no tuning. Determine generality beyond selected3 before a new volatility-sizing/taker trigger. Standing small-CPU authorization covers a bounded declaration; no new days/live.
<!-- /CAP_MAKER_SIZING_20261124_CODEX -->

'''
status='''cap_maker_sizing_20261124_codex:
  report: predictfun/reviews/maker3_cap_skew_20261124/report.md
  freeze: 787d1a0f6
  cap_only_exact_parity: 3
  order_max_shares: 100
  levels: 2
  wider_hard_cap: 1000
  wider_soft_band: 500
  pooled_end_usd: [-121.315946, -121.315946, 47.787734, 47.787734]
  arms: [BASE, CAP_ONLY, WIDE_PASSIVE, WIDE_TAKE]
  wider_take_activated: false
  validation: false
  maker_signed_quote_hashes: 33
  maker_quote_contexts_usable: 28
  maker_private_skew_or_cap_identified: false
  neutral_maker_seconds: 5700
  matched_opposing_take_seconds: 52
  packet_charged_upper_cpu_s: 120
  next: fixed_wider_band_paired_replay_on_19_consumed_complete_markets
'''
for rel,txt in [('predictfun/HANDOFF.md',block),('predictfun/FINDINGS.md',block),('predictfun/STATUS.yml',status)]:
 p=R/rel;current=p.read_text();assert 'CAP_MAKER_SIZING_20261124_CODEX' not in current and 'cap_maker_sizing_20261124_codex:' not in current
 p.write_text(txt+current)
 base=subprocess.check_output(['git','show','HEAD:'+rel],cwd=R)
 blob=subprocess.check_output(['git','hash-object','-w','--stdin'],cwd=R,input=txt.encode()+base).decode().strip()
 subprocess.check_call(['git','update-index','--cacheinfo','100644',blob,rel],cwd=R)
print('Saved evidence and staged only new owned state prefixes. Research CPU user',user,'serial wall upper',wall,'charge120.')
