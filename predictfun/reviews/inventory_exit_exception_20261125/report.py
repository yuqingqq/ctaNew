import collections,csv,json
def run(E):
 E.bound();checks=json.loads((E.H/'checks.json').read_text());assert checks['pass_all'];rows=checks['rows'];lookup={(x['mid'],x['arm']):x for x in rows}
 numeric=['net_c','completed_c','residual_c','maker_shares','passive_generations','passive_pulls','overrides','override_posts','override_keeps','override_ask','override_bid','affected_gid_fills','pressure_decisions']
 def total(rs):
  o={k:sum(r[k] for r in rs) for k in numeric};o.update(peak_inventory=max(r['peak_inventory'] for r in rs),mean_abs_inventory=sum(r['mean_abs_inventory'] for r in rs)/len(rs),both_grid_fraction=sum(r['both_grid_fraction'] for r in rs)/len(rs));return o
 totals={a:total([r for r in rows if r['arm']==a]) for a in E.ARMS};contrasts={}
 for a,b in [('BASE','WIDE'),('WIDE','WIDE_REDUCE'),('WIDE','WIDE_COST'),('WIDE_REDUCE','WIDE_COST')]:
  diff=[dict(day=x['day'],mid=x['mid'],delta=lookup[x['mid'],b]['net_c']-x['net_c']) for x in rows if x['arm']==a]
  days={d:sum(r['delta'] for r in diff if r['day']==d) for d in sorted({r['day'] for r in diff})};den=sum(abs(x['delta']) for x in diff)
  reading='POSITIVE_EVERY_DAY' if all(v>1e-6 for v in days.values()) else 'NEGATIVE_EVERY_DAY' if all(v< -1e-6 for v in days.values()) else 'MIXED'
  contrasts[b+' minus '+a]=dict(total=sum(x['delta'] for x in diff),days=days,better=sum(x['delta']>1e-6 for x in diff),worse=sum(x['delta']< -1e-6 for x in diff),
   excluding_3789484=sum(x['delta'] for x in diff if x['mid']!=3789484),top3_abs_share=sum(sorted([abs(x['delta']) for x in diff],reverse=True)[:3])/den if den else None,reading=reading,markets=diff)
 E.dump(E.H/'claims.json',dict(totals=totals,contrasts=contrasts,markets=19,days=4,validation=False))
 with (E.H/'per_market.csv').open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 lines=['# Inventory-reducing pressure exception — 2026-11-25','',
 'Full19 consumed BOOK markets, Oct24–27; four fixed arms, full supported298s. These are complete simulated feedback paths, not identified Maker3 posting decisions. Net is cash plus inventory at common END, not settlement. Passive fees/rebates remain zero as in the parent.','',
 '| Arm | Net USD | Completed USD | Residual USD | Both sides live | Mean abs inventory | Pulls |',
 '|---|---:|---:|---:|---:|---:|---:|']
 for a,t in totals.items():lines.append(f"|{a}|{t['net_c']/100:+.2f}|{t['completed_c']/100:+.2f}|{t['residual_c']/100:+.2f}|{100*t['both_grid_fraction']:.1f}%|{t['mean_abs_inventory']:.1f}|{t['passive_pulls']}|")
 lines+=['','BASE uses C200/S100/R50. All others C1000/S500/R250. REDUCE permits otherwise pressure-blocked inventory-reducing quotes; COST additionally requires one-cent margin against known remaining average cost. Same100max/two levels and all parent protections.','']
 for name,c in contrasts.items():lines.append(f"- {name}: ${c['total']/100:+.2f}; {c['reading']}; {c['better']} better/{c['worse']} worse markets; excluding3789484 ${c['excluding_3789484']/100:+.2f}. Day USD: "+', '.join(f'{d} {v/100:+.2f}' for d,v in c['days'].items()))
 lines+=['','## Market3789484','', '|Arm|Net USD|Completed USD|Residual USD|New exception posts|Exception keeps (decisions)|','|---|---:|---:|---:|---:|---:|']
 for a in E.ARMS:
  x=lookup[3789484,a];lines.append(f"|{a}|{x['net_c']/100:+.2f}|{x['completed_c']/100:+.2f}|{x['residual_c']/100:+.2f}|{x['override_posts']}|{x['override_keeps']}|")
 lines+=['','## Integrity and limits','',f"All{checks['records']} ledgers independently recomputed; {checks['pressure_decisions_checked']} pressure decisions checked against notified FIFO lots, order reservations and resulting commands; prior controls6/6 exact; corruptions refused{checks['corruptions_refused']}/9. Details checks.json and contract.md.",
 'Historical cost does not estimate the opportunity cost of selling now or future market movement. Cost qualifying uses average remaining FIFO basis; it does not guarantee every individual closed lot is profitable. Retained orders can later lose their exemption; churn remains measured. Guard logs contain parent counterfactual proposals; commands/exit_log determine executed actions. Override KEEP counts are decision ticks, not distinct orders. Affected-gid fills include all that gid’s fills and are descriptive, not avoided-loss attribution.',
 'Four consumed days, no intervals or forward validation. The19complete markets exclude earlier missing/stale paths; their historical external flow is fixed. Gains cannot be claimed for unseen markets or inferred Maker3 policy.']
 (E.H/'report.md').write_text('\n'.join(lines)+'\n');print(json.dumps(dict(totals=totals,contrasts={k:{q:v for q,v in c.items() if q!='markets'} for k,c in contrasts.items()}),indent=2))
