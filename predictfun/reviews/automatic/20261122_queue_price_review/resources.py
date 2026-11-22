from pathlib import Path
import json,re
O=Path(__file__).resolve().parent
R=Path('/home/yuqing/ctaNew')
e=json.loads((O/'evidence.json').read_text())
rx=re.compile(r'cpu_user_s=([0-9.]+)')
sec=sum(float(m.group(1)) for p in O.glob('*.log') for m in rx.finditer(p.read_text()))
base=e['checks']['details']['I']['line_core_hours']
projection=json.loads((R/'predictfun/reviews/ladder_pressure_positive_20261121/logs/pilot_projection.json').read_text())
remaining=projection['projected_core_hours']-projection['so_far_core_hours']
out=dict(review_cpu_seconds_before_this_small_summary=sec,cumulative_before_summary=base+sec/3600,
         remaining_original_allocation_before_summary=2-base-sec/3600,
         prior_positive_guard_projected_remaining=remaining,
         cumulative_projected_if_completed=base+sec/3600+remaining,
         proposed_positive_round_ceiling=0.16,proposed_line_ceiling=2.12,
         proposed_ceilings_are_not_authorized=True)
(O/'resources.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
