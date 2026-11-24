"""Owned handoff blocks, resource receipt and index-only owned state hunks."""
import json,re,subprocess
from pathlib import Path
H=Path(__file__).resolve().parent;R=H.parents[2]
logs=list(H.glob('*.log'));wall=0.;user=0.
for p in logs:
    for line in p.read_text().splitlines():
        if 'tag=maker3_late_' in line and 'wall_s=' in line:
            wall+=float(re.search(r'wall_s=([\d.]+)',line).group(1));user+=float(re.search(r'cpu_user_s=([\d.]+)',line).group(1))
assert wall<44
receipt=dict(cap_cpu_s=90,charged_upper_cpu_s=45,measured_one_core_wall_s=wall,measured_user_cpu_s=user,
             upper_includes='all attempted slot runs, static JS syntax checks and final publication; no browser process',
             task_cap_h=2.10,line_cap_h=5.40,standing_reserve_h=.05,
             task_charged_h=2.0090588889+45/3600,line_charged_h=5.3197588889+45/3600)
(H/'resources.json').write_text(json.dumps(receipt,indent=2)+'\n')
note='\nUI update: both existing self-contained charts show native BUY YES / BUY NO and the recorded-flow change when a maker3 taker marker is selected.13 annotations across the chart markets; all7 example takes match. Both decoded research payloads remain byte-identical, JavaScript syntax passes, and served HTML equals local bytes. This text-only annotation update did not rerun a browser rendering test. Packet charged conservatively45/90CPU seconds (all failed attempts included), task cap2.10/line5.40 under standing authorization.\n'
p=H/'report.md';p.write_text(p.read_text()+note)
(R/'predictfun/reviews/maker_dashboard_20261124/maker3_late_report.md').write_text(p.read_text())
body='''## 2026-11-24 — Maker3 late activity and native take direction verified

[Receipt](reviews/maker3_late_activity_20261124/report.md). In3789484, last120s carries52.8% shares and89.3% taker shares; mostly the final minute.319 legs/8,142.595shares, seven takes/205.964shares; passive volume97.47%. All seven native actions BUY (5NO/2YES), not native YES sells. At202.668–202.705s, BUY NO90c109.387sh is chart ask10c, trimming recorded net+553.4→+444.0;248.630s BUY NO84c74.535sh trims+621.5→+547.0. All seven oppose/reduce the absolute captured trade-flow proxy; this is NOT private inventory or identified motive. Captured record order is not exact execution/decision order.

Broader934 observed markets: last2m only30.2% total shares/37.0% taker shares;70/934 have majority volume there. Example late concentration is not a general market-entry timing rule.134 source-status legs/36self legs excluded; registry coverage statuses retained; Oct27 activity stop remains relevant. Checks:319raw/chart legs,9562native takes,60minute cells,4corruptions; chart13native annotations, unchanged embedded research payloads. No replay/model/newdays. All attempts charged45/90CPU seconds; standing-approved cap task2.10/line5.40/.05reserve, approximate usage2.02156/5.33226.

Next priority: align these already observed takes with our contemporaneous inventory and executable prices in this market, before declaring an active-rebalancing rule; use whole-book longer-horizon value. First-quote versus first-position opening scope remains unresolved and unchanged. Do not add a last-two-minute rule or copy all NO buys from this descriptive audit.
'''
blocks={
'predictfun/HANDOFF.md':'<!-- MAKER3_LATE_ACTIVITY_20261124_CODEX -->\n'+body+'<!-- /MAKER3_LATE_ACTIVITY_20261124_CODEX -->\n\n',
'predictfun/FINDINGS.md':'<!-- MAKER3_LATE_ACTIVITY_20261124_CODEX -->\n'+body+'<!-- /MAKER3_LATE_ACTIVITY_20261124_CODEX -->\n\n',
'predictfun/STATUS.yml':'''maker3_late_activity_20261124_codex:
  report: predictfun/reviews/maker3_late_activity_20261124/report.md
  example_market: 3789484
  last_120s_share_fraction: 0.5280906534
  pooled_last_120s_share_fraction: 0.3017198738
  example_takes: 7
  example_native_actions: "5 BUY NO, 2 BUY YES"
  example_takes_reduce_recorded_net_absolute: 7
  private_inventory_or_motive_identified: false
  new_replay: false
  packet_charged_upper_cpu_s: 45
  packet_cap_cpu_s: 90
'''}
for name,block in blocks.items():
    path=R/name;current=path.read_text();assert 'MAKER3_LATE_ACTIVITY_20261124_CODEX' not in current and 'maker3_late_activity_20261124_codex:' not in current
    base=subprocess.check_output(['git','show','HEAD:'+name],cwd=R).decode()
    path.write_text(block+current)
    blob=subprocess.check_output(['git','hash-object','-w','--stdin'],input=(block+base).encode(),cwd=R).decode().strip()
    subprocess.run(['git','update-index','--cacheinfo','100644',blob,name],cwd=R,check=True)
print(json.dumps(receipt,indent=2));print('Only new owned prefixes indexed; all pre-existing working-tree edits preserved')
