"""Charge full one-core renderer wall bound; release unused preview reserve."""
import json,re,subprocess
from pathlib import Path
H=Path(__file__).resolve().parent;R=H.parents[2]
budget=json.loads((H/'resources.json').read_text());stages=[]
for p in H.glob('*.log'):
 v=re.findall(r'wall_s=([0-9.]+)',p.read_text(errors='replace'))
 if v:stages.append(dict(log=p.name,wall_upper_s=float(v[-1])))
upper=round(sum(z['wall_upper_s'] for z in stages)+25+1,2)
assert upper<=60
budget.update(stages=stages,render_cpu_charge_s=25,render_charge_basis='full25s timeout wall; one-core renderer; time recorder terminated before footer',publication_and_final_browser_reserve_s=1,packet_upper_cpu_s=upper)
(H/'resources.json').write_text(json.dumps(budget,indent=2)+'\n')
p=R/'predictfun/STATUS.yml';old=p.read_text();before='  packet_upper_cpu_s: 42.50\n  packet_cap_cpu_s: 60\n';after=f'  packet_upper_cpu_s: {upper:.2f}\n  packet_cap_cpu_s: 60\n';assert before in old;p.write_text(old.replace(before,after,1))
staged=subprocess.check_output(['git','show',':predictfun/STATUS.yml'],cwd=R).decode();assert before in staged
oid=subprocess.check_output(['git','hash-object','-w','--stdin'],cwd=R,input=staged.replace(before,after,1).encode()).decode().strip();subprocess.run(['git','update-index','--cacheinfo','100644',oid,'predictfun/STATUS.yml'],cwd=R,check=True)
note=f'\nFinal resource upper: {upper:.2f}/60CPU seconds. The renderer time footer was terminated with its25s timeout, so the full one-core25s wall is charged. Remaining publication reserve1s; no further browser run. Screenshots precede the final first-quote/first-fill text clarification; final JavaScript syntax and served page were checked.\n'
for path in [H/'report.md',R/'predictfun/reviews/maker_dashboard_20261124/opening_report.md']:path.write_text(path.read_text()+note)
print('Conservative final bound',upper)
