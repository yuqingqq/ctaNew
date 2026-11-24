"""Measured all-in resource gate, before full-cohort inference."""
import inspect,json,re,sys
import fast_v3 as F
import build as B
H=B.HERE
p=H/'resource_gate.json';B.need(not p.exists(),'write once resource gate')
spent=0.
for line in (B.ROOT/'predictfun/research/run_light.log').read_text().splitlines():
 if re.search(r' tag=mt_qty_\S+ ',line):
  match=re.search(r' cpu_user_s=([\d.]+)',line);B.need(match,'missing CPU charge');spent+=float(match.group(1))
gate=json.loads((H/'pilot_gate_v3.json').read_text());allowed=spent+gate['projected_remaining_all_in_cpu_s']+.25<=54
res=dict(previous_cpu_s=spent,projected_remaining_cpu_s=gate['projected_remaining_all_in_cpu_s'],launch_margin_s=.25,all_in_cap_s=54,allowed=allowed,basis='measured_prior_charge_replaces_fixed_9_second_reserve',pilot_parity=gate['pilot_content_equal'] and gate['normalized_values_equal'])
B.dump(p,res);print(json.dumps(res),flush=True);B.need(allowed and res['pilot_parity'],'all-in gate')
src=inspect.getsource(F.main)
for old,new in [("resource.setrlimit(resource.RLIMIT_CPU,(16,18))","resource.setrlimit(resource.RLIMIT_CPU,(20,22))"),("gate['run_full'] and B.sha(DATA/'pilot.json')==gate['pilot_sha']","B.sha(DATA/'pilot.json')==gate['pilot_sha']")]:B.need(src.count(old)==1,'exact runtime adapter');src=src.replace(old,new)
ns=dict(F.__dict__);exec(compile(src,'<measured resource launch>','exec'),ns)
sys.argv=['resource_launch.py','full'];ns['main']()
