import hashlib,json,re,subprocess
from pathlib import Path
H=Path(__file__).resolve().parent;R=H.parents[2]
def git(*args,input=None):return subprocess.check_output(['git',*args],cwd=R,input=input)
def run():
 assert not git('diff','--cached','--name-only').strip()
 x=json.loads((H/'components.json').read_text());c=json.loads((H/'checks.json').read_text());assert c['pass_all'] and c['falsifiers']==4
 named=next(z for z in x['markets'] if z['mid']==3789484);f=next(z for z in json.loads((H/'first_divergences.json').read_text()) if z['mid']==3789484)
 p={z['category']:z for z in x['pooled']};n={z['category']:z for z in named['components']}
 entry=sum(p[k]['delta_c'] for k in ['new_entry','foregone_entry','new_role','foregone_role']);ne=sum(n[k]['delta_c'] for k in ['new_entry','foregone_entry','new_role','foregone_role'])
 text=(H/'report.md').read_text();assert '## Interpretation' not in text
 text+='\n## Interpretation\n\n'
 text+=f"On common entry units closed in both arms ({p['common_closed']['shares']:.2f}shares), the exception exits{-p['common_closed']['hold_ms_diff']/1000:.2f}s earlier on average, at{p['common_closed']['signed_exit_price_diff']:.3f}c/share worse directional exit value. This accounts for${p['common_closed']['delta_c']/100:.2f}. Close-versus-END changes contribute${p['common_close_vs_residual']['delta_c']/100:+.2f}; changed entry/turnover chains contribute${entry/100:+.2f}. Total${x['total_c']/100:+.2f}. These are quantity-weighted descriptions, not a claim every earlier exit is harmful.\n\n"
 text+=f"In3789484, common closed entries are{n['common_closed']['shares']:.2f}shares: exits{-n['common_closed']['hold_ms_diff']/1000:.2f}s earlier and{n['common_closed']['signed_exit_price_diff']:.3f}c/share worse, a${n['common_closed']['delta_c']/100:.2f}contribution. Close-versus-residual contributes${n['common_close_vs_residual']['delta_c']/100:+.2f}; changed entry/turnover chains${ne/100:+.2f}; total${named['delta_c']/100:+.2f}.\n\n"
 text+=f"First divergence is{f['t']/1000:.1f}s: WIDE waits; COST posts{f['cost'][6]:.0f}asks at{f['cost'][5]:.0f}c with{f['context']['I']:.2f}known long shares at{f['context']['basis']:.0f}c average basis. Both paths' fills and commands before that decision agree. This identifies the policy divergence, not a Maker3 action or a guaranteed fill of that quote.\n\n"
 text+='The acquisition-cost test is insufficient for overriding the guard at the existing quote prices. It sometimes locks in a positive accounting margin while giving up better exits, and it also changes later exposure. Do not adopt either exit exception or tune the cost threshold. No new strategy is promoted.\n\n'
 text+='ONE next priority: a saved-data reducing-side quote-price and exit-timing audit on the same19markets—at pressure-blocked, inventory-reducing states, compare our intended price/depth and already available market pressure with the later observed exit prices; use Maker3 fill/price bounds only as a separately graded contextual comparison. This should establish whether the next useful lever is quote placement rather than acquisition-cost permission. Do not turn future exit prices into live inputs or infer exact Maker3 posting prices/times. No new replay/days are needed for that diagnostic.\n\n'
 freeze=git('rev-parse','ca16030cd').decode().strip()
 logs=[l for l in (R/'predictfun/research/run_light.log').read_text().splitlines() if l.startswith('2026-11-25') and 'tag=ia_' in l and 'slot=codex' in l]
 (H/'resources.log').write_text('\n'.join(logs)+'\n');cpu=sum(float(re.search(r'cpu_user_s=([0-9.]+)',l)[1]) for l in logs)
 charged=30;task=2.2882166666666666+charged/3600;line=5.598916666666667+charged/3600
 assert cpu<charged and task+.05<2.4 and line+.05<5.7
 text+=f"Freeze `{freeze}`. Measured{cpu:.2f}CPU s before publication; charge30s including publication, below72. Cumulative task{task:.5f}/line{line:.5f}, caps2.40/5.70 with.05reserve unchanged. Parent ledgers frozen and hash-verified; local commits only.\n"
 (H/'report.md').write_text(text)
 (H/'resources.json').write_text(json.dumps(dict(measured_cpu_s=cpu,charged_cpu_s=charged,task_coreh=task,line_coreh=line,task_cap=2.4,line_cap=5.7,reserve=.05),indent=2)+'\n')
 key='INVENTORY_EXIT_ATTRIBUTION_20261125_CODEX';link='reviews/inventory_exit_attribution_20261125/report.md'
 content=f'''## 2026-11-25 — cost-based exits mostly lose through worse exit prices

[Receipt]({link}); frozen{freeze}. USER Continue; saved WIDE_COST−WIDE paths only, no replay/fit/new days.19markets,38ledgers,17424FIFO/event-unit interval pieces, common END. Common event quantities matched by prefix (canonical fungible-unit attribution, not physical-lot/motive identification). All deltas reconcile to accepted parent,4real corruptions refused; synthetic long/short/partial/residual/new-role controls pass.

All19 delta−300.86USD = same-entry closed exits−232.09 +same-entry close-vs-residual+64.24 +changed entry/turnover chains−133.01. Common closed entries43472.26shares: exit2.27s earlier on average, directional exit price0.534c/share worse. In3789484−75.51 = −62.15+0.41−13.77; common1739.81shares exit7.75s earlier/3.572c worse. Thus giving up better exits is evidenced for common entries, while re-entry/role feedback also matters; this is accounting attribution, not an isolated causal exit treatment.

First3789484 divergence8.1s: pressure WAIT becomesPOST23asks at60c, knownlong23.87shares at59c average basis; both paths' prior fills/commands identical. The trade can be positive vs acquisition cost yet inferior to holding. No exit exception adopted; no cost-threshold tuning. No identified private Maker3 post decision or unseen-day claim.

Charge30CPU s; cumulative task{task:.5f}/line{line:.5f}, caps2.40/5.70 with.05reserve. ONE next priority: same19 saved-data reducing-side quote-price/depth and exit-timing audit at pressure-blocked states, with Maker3 fill/price bounds kept as separate contextual evidence. Establish whether price placement is the next lever; future prices remain outcomes, not live inputs. No replay/new days needed.
'''
 prefixes={p:f'<!-- {key} -->\n'+content+f'<!-- /{key} -->\n\n' for p in ['predictfun/HANDOFF.md','predictfun/FINDINGS.md']}
 prefixes['predictfun/STATUS.yml']=f'''inventory_exit_attribution_20261125_codex:
  report: predictfun/{link}
  freeze: {freeze}
  markets: 19
  ledgers_checked: 38
  interval_pieces_checked: 17424
  falsifiers_refused: 4
  total_delta_usd: -300.86
  same_entry_closed_exit_delta_usd: -232.09
  same_entry_close_residual_delta_usd: 64.24
  changed_entry_chains_delta_usd: -133.01
  common_exit_seconds_earlier: 2.27
  common_exit_cents_worse: 0.534
  named_market_exit_seconds_earlier: 7.75
  named_market_exit_cents_worse: 3.572
  strategy_adopted: false
  validation: false
  charged_cpu_s: 30
  task_coreh: {task:.5f}
  line_coreh: {line:.5f}
  next: saved_reducing_quote_price_and_exit_timing_audit_before_new_variant
'''
 for path,prefix in prefixes.items():
  file=R/path;current=file.read_text();assert key not in current
  file.write_text(prefix+current);base=git('show','HEAD:'+path)
  blob=git('hash-object','-w','--stdin',input=prefix.encode()+base).decode().strip();subprocess.run(['git','update-index','--cacheinfo','100644',blob,path],cwd=R,check=True)
 fs=['report.md','components.json','components.csv','first_divergences.json','checks.json','manifest.json','resources.json']
 (H/'SHA256SUMS').write_text(''.join(hashlib.sha256((H/f).read_bytes()).hexdigest()+'  '+f+'\n' for f in fs))
 print('Published and staged owned prefixes only;',task,line)
if __name__=='__main__':run()
