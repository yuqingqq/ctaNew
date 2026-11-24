import json,re
from pathlib import Path
h=Path(__file__).resolve().parent;root=h.parents[3]
log=(root/'predictfun/research/run_light.log').read_text();runs=[]
for line in log.splitlines():
 if 'tag=mt_dash_inline_' not in line:continue
 d=dict(re.findall(r'(\w+)=([^ ]+)',line));runs.append(dict(tag=d['tag'],rc=int(d['rc']),cpu_user_s=float(d['cpu_user_s'])))
previous=json.loads((h.parent/'resources.json').read_text());charge=sum(r['cpu_user_s'] for r in runs)+20+1
r=dict(runs=runs,additional_browser_conservative_s=20,finalizer_reserve_s=1,additional_charge_s=charge,total_packet_charge_s=previous['packet_charged_cpu_s']+charge,packet_cap_s=180,task_upper_coreh=previous['task_upper_coreh']+charge/3600,line_upper_coreh=previous['line_upper_coreh']+charge/3600,task_cap_coreh=2.05,line_cap_coreh=5.35,standing_reserve_coreh=.05)
assert r['total_packet_charge_s']<=180 and r['task_upper_coreh']+.05<=2.05 and r['line_upper_coreh']+.05<=5.35
(h/'resources.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r))
