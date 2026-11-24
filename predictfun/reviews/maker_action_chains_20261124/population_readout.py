"""Explicitly post-run descriptive grouping, all88; no inference test or fitted rule."""
import csv,json,collections
from pathlib import Path
H=Path(__file__).resolve().parent;O=H/'outputs';a=list(csv.DictReader((O/'chains.csv').open()));need=lambda x:None
keys=('group','successor_relation','successor_kind','successor_direction')
c=collections.Counter(tuple(x[k] for k in keys) for x in a)
with (H/'population_chain_counts.csv').open('x') as f:
 w=csv.DictWriter(f,fieldnames=list(keys)+['orders'],lineterminator='\n');w.writeheader();w.writerows(dict(zip(keys,k),orders=n) for k,n in sorted(c.items()))
assert sum(c.values())==88
rows=[]
for group in ('exhausted','conditional_withdrawal','unresolved'):
 for scope in ('all_existing_links','ordered_after_only'):
  z=[x for x in a if x['group']==group and x['successor_hash'] and (scope=='all_existing_links' or x['successor_relation']=='sequential_after_under_bounds')]
  rows.append(dict(group=group,scope=scope,orders=len(z),direction=dict(collections.Counter(x['successor_direction'] for x in z)),kind=dict(collections.Counter(x['successor_kind'] for x in z)),successor_timing=dict(collections.Counter(x['successor_relation'] for x in z))))
(H/'population_readout.json').write_text(json.dumps(rows,indent=2)+'\n')
p=H/'logic_chains.md';text=p.read_text();old='No continuously-available-priority contradiction is found **between** the repeated fills; the exact between-fill coverage is in readout.json and must accompany that zero. It does not establish continuous rest.'
assert old in text;text=text.replace(old,'The between-fill test has **zero timing-separated opportunities**: no competing record has a hard window strictly between two own fill windows. Its zero contradictions therefore says nothing about continuous queue retention.')
text+='\n## All successor links, including ambiguous timing\n\nPost-run descriptive grouping of the existing88 rows; no new test or selected threshold. Every cell, including no successor, is in [population_chain_counts.csv](population_chain_counts.csv).\n\n| Ending group | Scope | Orders | Same price | Toward spread | Away from spread | Unknown |\n|---|---|---:|---:|---:|---:|---:|\n'
for x in rows:
 d=x['direction'];text+='| '+x['group']+' | '+x['scope']+' | '+str(x['orders'])+' | '+str(d.get('same_price',0))+' | '+str(d.get('toward_spread',0))+' | '+str(d.get('away_from_spread',0))+' | '+str(d.get('unknown',0))+' |\n'
text+='\nA direction is observable for the linked price; timing and replacement intent remain separately graded. This table does not convert ambiguous successors into post-ending commands. The seven ordered cases suggest candidate branches to test—replenish/reprice after depletion and retreat after a conditional ending—but the sample is too thin to establish a general maker rule.\n'
p.write_text(text);print(json.dumps(rows,indent=2))
