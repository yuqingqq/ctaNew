"""Bounded resource accounting for the original ladder allocation, no experiment."""
from pathlib import Path
import re,json
R=Path('/home/yuqing/ctaNew');O=Path(__file__).resolve().parent
rx=re.compile(r'^(\S+) tag=(\S+) rc=(\d+) peak_rss_mib=(\d+) wall_s=([\d.]+) cpu_user_s=([\d.]+)')
names=['ladder_rest_20261121','ladder_audit_20261121','ladder_guard_20261121','ladder_pressure_20261121','ladder_pressure_lag_20261121']
out={}
for name in names:
 rows=[m for p in (R/'predictfun/reviews'/name/'logs').rglob('*.log') for ln in p.read_text().splitlines() if (m:=rx.match(ln))]
 out[name]=dict(cpu_hours=sum(float(m[6]) for m in rows)/3600,peak_mib=max([int(m[4]) for m in rows] or [0]),records=len(rows))
review_names=['20261121_ladder_rest_review','20261121_ladder_audit_review','20261121_ladder_guard_review','20261121_ladder_pressure_review','20261121_ladder_pressure_lag_review']
rows=[m for name in review_names for p in (R/'predictfun/reviews/automatic'/name).glob('*.log') for ln in p.read_text().splitlines() if (m:=rx.match(ln))]
out['available_review_logs']=dict(cpu_hours=sum(float(m[6]) for m in rows)/3600,records=len(rows))
out['total_recorded_hours']=sum(v['cpu_hours'] for v in out.values());out['proposed_next_cap_hours']=.35
out['within_original_two_hours']=out['total_recorded_hours']+.35<=2
(O/'resources.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
