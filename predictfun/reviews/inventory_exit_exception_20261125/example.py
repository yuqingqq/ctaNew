"""Pre-specified illustration: earliest eligible BASE slot in3789484; no fill-outcome selection.
Presentation only, after full checks. Counts are decision ticks, not orders.
"""
import gzip,json
import launch
E=launch.E
def run():
 checks=json.loads((E.H/'checks.json').read_text());assert checks['pass_all'];sm=json.loads((E.OUT/'summary.json').read_text())
 out={}
 for arm in E.ARMS:
  row=next(x for x in sm['rows'] if x['mid']==3789484 and x['arm']==arm);p=E.OUT/f'3789484_{arm}.json.gz';assert E.sha(p)==row['sha256']
  r=json.loads(gzip.decompress(p.read_bytes()));eligible=[z for z in r['exit_log'] if z['kind']=='slot' and z['side']==1 and z['reducing'] and z['cost_ok']]
  out[arm]=dict(eligible_ask_slot_decisions=len(eligible),first_eligible_ask=eligible[0] if eligible else None)
 E.dump(E.H/'example.json',out);print(json.dumps(out,indent=2))
if __name__=='__main__':run()
