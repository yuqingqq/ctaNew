"""Publish accepted outputs and review-owned prefixes; never stage other-session hunks."""
import hashlib,json,math,re,shutil,subprocess
from pathlib import Path
H=Path(__file__).resolve().parent;R=H.parents[2];D=R/'data/predict_fun/research/inventory_exit_exception_20261125'
def git(*args,input=None):return subprocess.check_output(['git',*args],cwd=R,input=input)
def run():
 assert not git('diff','--cached','--name-only').strip(),'another staged change exists'
 checks=json.loads((H/'checks.json').read_text());assert checks['pass_all']
 claims=json.loads((H/'claims.json').read_text());example=json.loads((H/'example.json').read_text())
 sm=json.loads((D/'summary.json').read_text());assert len(sm['rows'])==76
 for row in sm['rows']:
  p=D/f"{row['mid']}_{row['arm']}.json.gz";assert hashlib.sha256(p.read_bytes()).hexdigest()==row['sha256']
 shutil.copyfile(D/'summary.json',H/'summary.json')
 logs=[l for l in (R/'predictfun/research/run_light.log').read_text().splitlines() if l.startswith('2026-11-25') and 'tag=ix_' in l and 'slot=codex' in l]
 (H/'resources.log').write_text('\n'.join(logs)+'\n')
 wall=sum(float(re.search(r'wall_s=([0-9.]+)',l)[1]) for l in logs)
 cpu=sum(float(re.search(r'cpu_user_s=([0-9.]+)',l)[1]) for l in logs)
 # Round charged wall time up, plus publication margin; wall upper-bounds this serial one-core research work.
 charged=math.ceil((wall+10)/30)*30;assert charged<=720
 task=2.12155+charged/3600;line=5.43225+charged/3600
 assert task+.05<=2.40 and line+.05<=5.70
 res=dict(measured_wall_s=wall,measured_user_cpu_s=cpu,charged_cpu_s=charged,task_coreh=task,line_coreh=line,task_cap=2.40,line_cap=5.70,reserve_coreh=.05)
 (H/'resources.json').write_text(json.dumps(res,indent=2)+'\n')
 freeze=git('rev-parse','2273fa902').decode().strip();amend=git('rev-parse','d47ff7a67').decode().strip()
 ex=example['BASE']['first_eligible_ask'];text=(H/'report.md').read_text()
 assert '## Accepted interpretation' not in text
 text+='\n## Accepted interpretation\n\n'
 text+=f"The observed blockage exists. At{ex['t']/1000:.1f}s in3789484, BASE knows{ex['I']:.2f}long shares, average remaining basis{ex['basis']:.0f}c, no other asks reserved, and a60c ask is withheld by pressure. The cost exception permits a bounded reducing quote under this type of state. This is an observable example, not an identified Maker3 posting decision.\n\n"
 text+='The proposed cost-based exemption does not earn its place in this strategy on these consumed markets: compared with unchanged WIDE it reduces END value on all four days. Its remaining-inventory value improves, but its completed turnover deteriorates more. The cost gate improves on exempting all reducing orders; both exceptions are inferior to retaining the original guard. This does not identify which downstream trade caused the loss: complete feedback paths change subsequent orders and fills as well as the immediate exit. Do not call the difference solely selling too early.\n\n'
 text+='The full19 cap comparison also qualifies the earlier three-market result: WIDE improves the pooled total by only$63.72 with mixed day/market signs; removing3789484 reverses that to−$175.35. Both full19 BASE and WIDE lose. More participation and completed turnover do not by themselves establish profitable market making.\n\n'
 text+='ONE next priority: saved-path attribution of WIDE_COST minus WIDE on the19markets, starting with3789484’s first divergence, separating changed exit proceeds from new/changed re-entry turnover and residual value. Reconcile the full difference before proposing another exit rule; do not tune the one-cent cushion or launch another cap variant. Saved-data bounded work fits existing authorization; no new days or live use.\n\n'
 text+=f"## Provenance and resources\n\nPolicy/source freeze `{freeze}`; resource-only completion `{amend}`. The first projection correctly stopped and its files/log remain. No policy changed afterward; first-market reuse was byte/hash-bound. All76 pressure-log memberships are exact. The first launcher’s pre-freeze module-name collision and successful correction are disclosed in resource_amendment.md. Final accepted outputs are report.md, claims.json, checks.json, per_market.csv, example.json and summary.json.\n\n"
 text+=f"Measured before publication: {cpu:.2f}user-CPU s / {wall:.2f}wall s. Charge{charged}CPU s including publication margin, below720; cumulative task{task:.5f}/line{line:.5f}, ceilings2.40/5.70 with0.05reserve. Peak903MiB. Larger allowance used the standing small-CPU authorization; no data/strategy expansion. Frozen parent files and other sessions’ unstaged changes preserved; local commits only. Inventory mean/uptime/peak statistics use the inherited50ms grid.\n"
 (H/'report.md').write_text(text)
 key='INVENTORY_EXIT_EXCEPTION_20261125_CODEX';link='reviews/inventory_exit_exception_20261125/report.md'
 common=f'''## 2026-11-25 — acquisition-cost exit exception loses; full19 cap benefit is mixed

[Receipt]({link}). USER proposes allowing reducing asks above acquisition cost in3789484. Frozen2273fa902; resource-only amendmentd47ff7a67 after first timing projection refused. Same19 complete consumed Oct24–27 markets, full298s, four arms;100max/two levels/opening gate/ASYNC_FAST10 unchanged. BASE C200/S100/R50, others C1000/S500/R250; REDUCE exempts pressure-blocked risk-reducing orders, COST adds>=1c over known remaining FIFO average basis (short side symmetric). Unclosed/cancel-pending reservations retained; no use of hidden fills; no taker/new days/fit/live.

END USD BASE−916.05, WIDE−852.33, REDUCE−1624.34, COST−1153.19. WIDE−BASE+63.72 MIXED,9better/10worse; excluding3789484−175.35. Thus prior3 positive combined result does NOT generalize to this19. COST−WIDE−300.86,negative all4days; REDUCE−WIDE−772.01,negative all4. COST beats REDUCE+471.14 all4, but neither earns adoption. COST completed−936.11 versus WIDE−512.63; residual−217.09 versus−339.71. Both-side grid uptime89.5/89.0%; meanabsI183/197. Selling profitably by historical cost is insufficient for improved full-book value; immediate exit vs subsequent re-entry feedback not yet separated.

3789484 WIDE+248.80 ->COST+173.29, REDUCE+109.74. BASE first qualifying ask at8.1s: known50.87shares at59c, ask60c, pressure withholds. This verifies the user's blockage observation; no claim Maker3’s posting clock/logic identified.76ledgers/16023pressure decisions independently checked,6prior controls exact,9corruptions refused; original projection and pre-freeze launcher failure retained/disclosed.

Charge{charged}CPU s; task{task:.5f}/line{line:.5f}, caps2.40/5.70 with.05reserve under standing CPU top-ups. No push; other-session files/hunks preserved.

ONE next priority: saved-path attribution COST−WIDE on19markets, starting3789484’s first divergence, separating changed exits from subsequent re-entry turnover and residual value. Full difference must reconcile before another exit candidate; no cushion/cap tuning. No new evidence needed for that bounded step.
'''
 prefixes={
 'predictfun/HANDOFF.md':f'<!-- {key} -->\n'+common+f'<!-- /{key} -->\n\n',
 'predictfun/FINDINGS.md':f'<!-- {key} -->\n'+common+f'<!-- /{key} -->\n\n',
 'predictfun/STATUS.yml':f'''inventory_exit_exception_20261125_codex:
  report: predictfun/{link}
  freeze: {freeze}
  population_markets: 19
  consumed_days: 4
  arms: [BASE, WIDE, WIDE_REDUCE, WIDE_COST]
  net_usd: {{BASE: -916.05, WIDE: -852.33, WIDE_REDUCE: -1624.34, WIDE_COST: -1153.19}}
  cost_vs_wide_negative_every_day: true
  reduce_vs_wide_negative_every_day: true
  wide_vs_base_reading: MIXED
  adopted: false
  validation: false
  ledgers_checked: 76
  pressure_decisions_checked: 16023
  parity_controls: 6
  falsifiers_refused: 9
  charged_cpu_s: {charged}
  task_coreh: {task:.5f}
  line_coreh: {line:.5f}
  task_cap_coreh: 2.40
  line_cap_coreh: 5.70
  next: attribute_changed_exit_proceeds_vs_reentry_and_residual_from_saved_paths_before_new_rule
'''}
 for path,prefix in prefixes.items():
  p=R/path;current=p.read_text();assert key not in current if path.endswith('.md') else 'inventory_exit_exception_20261125_codex:' not in current
  p.write_text(prefix+current)
  # Only prefix against committed bytes enters the index; preserve all existing unstaged additions.
  original=git('show','HEAD:'+path);blob=git('hash-object','-w','--stdin',input=prefix.encode()+original).decode().strip()
  subprocess.run(['git','update-index','--cacheinfo','100644',blob,path],cwd=R,check=True)
 files=['report.md','claims.json','checks.json','summary.json','per_market.csv','example.json','resources.json','manifest.json','resume_manifest.json']
 (H/'SHA256SUMS').write_text(''.join(hashlib.sha256((H/f).read_bytes()).hexdigest()+'  '+f+'\n' for f in files))
 print(json.dumps(res));print('Owned state prefixes staged; all pre-existing unstaged content retained.')
if __name__=='__main__':run()
