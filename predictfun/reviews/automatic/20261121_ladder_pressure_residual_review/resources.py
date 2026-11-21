"""Saved mutation records, budget and descriptive denominators; no experiment."""
from pathlib import Path
import json,re,sys
R=Path('/home/yuqing/ctaNew');P=Path('/tmp/pf-residual-review-2063d31a');O=Path(__file__).resolve().parent
sys.path.insert(0,str(P/'predictfun/research'))
import pf_mutant_summary as MS
import pf_mutant_one as MO
import pf_ladder_pressure_residual_mutants as H
receipt=P/'predictfun/reviews/ladder_pressure_residual_20261121';d=receipt/'logs/mutants_try1'
recs={'control':MS.last_record(d/'lrs_control.log')};recs.update({i:MS.last_record(d/f'lrs_{i}.log') for i in range(len(H.MUTANTS))})
okay,bad=MS.judge(len(H.MUTANTS),recs,MO.sources(H),[m[0] for m in H.MUTANTS]);assert okay,bad
rx=re.compile(r'^(\S+) tag=(\S+) rc=(\d+) peak_rss_mib=(\d+) wall_s=([\d.]+) cpu_user_s=([\d.]+)')
names=['ladder_rest','ladder_audit','ladder_guard','ladder_pressure','ladder_pressure_lag','ladder_pressure_ablation','ladder_pressure_residual'];out={}
for name in names:
 rows=[m for p in (R/'predictfun/reviews'/f'{name}_20261121'/'logs').rglob('*.log') for ln in p.read_text().splitlines() if (m:=rx.match(ln))]
 out[name]=dict(cpu_hours=sum(float(m[6]) for m in rows)/3600,peak_mib=max([int(m[4]) for m in rows] or [0]),records=len(rows))
rows=[m for name in names for p in (R/'predictfun/reviews/automatic'/f'20261121_{name}_review').glob('*.log') for ln in p.read_text().splitlines() if (m:=rx.match(ln))]
out['available_review_logs']=dict(cpu_hours=sum(float(m[6]) for m in rows)/3600,records=len(rows))
out['total_recorded_hours']=sum(v['cpu_hours'] for v in out.values());out['remaining_hours']=2-out['total_recorded_hours']
out['proposed_next_cap_hours']=.14;out['next_round_within_original_two_hours']=out['total_recorded_hours']+.14<=2
out['mutant_summary_rejudged']=dict(control='PASSED',caught=len(H.MUTANTS),sources=MO.sources(H))
rd=json.loads((O/'evidence.json').read_text())['reading']['by_day']['all']
out['descriptive_denominators']={}
for name in ('timing','unflagged'):
 z=rd[name];out['descriptive_denominators'][name]=dict(valid_q_fraction=z['q']/rd['total']['q'],positive_value_fraction=z['pos']/rd['total']['pos'],
 negative_cents_per_valid_share=z['neg']/z['q'],net_cents_per_valid_share=(z['neg']+z['pos'])/z['q'])
(O/'resources.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
