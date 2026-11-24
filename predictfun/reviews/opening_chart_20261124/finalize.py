"""Publish verified chart receipt, clarify first quote versus first fill."""
import gzip,json,re,subprocess
from pathlib import Path
H=Path(__file__).resolve().parent;R=H.parents[2];UI=R/'predictfun/reviews/maker_dashboard_20261124'
x=json.loads(gzip.decompress((H/'data.json.gz').read_bytes()));checks=json.loads((H/'checks.json').read_text());browser=json.loads((H/'browser.json').read_text())
assert checks['passed'] and not browser['errors'] and not browser['mobile']['overflow']
m=next(m for m in x['cases'] if m['id']==3789484);b=m['arms']['OPENING_ONLY'];f=b['fills'][0]
assert (f['t'],f['side'],f['price'],f['q'])==(.951,1,55.,5.)
initial_ids={o['gid'] for o in b['first_orders']}
assert not any(f['gid'] in initial_ids for f in b['fills'])
js=(UI/'opening.js').read_text()
old='<p>First fill ${num(s.first_fill)} s · ${s.first_orders.map(o=>`${o.side===0?\'bid\':\'ask\'} ${num(o.price,0)}¢`).join(\' / \')}</p>'
new='<p>First quotes: ${s.first_orders.map(o=>`${o.side===0?\'bid\':\'ask\'} ${num(o.price,0)}¢`).join(\' / \')}</p><p>First fill ${num(s.first_fill)} s: ${s.fills[0]?.side===0?\'bid\':\'ask\'} ${num(s.fills[0]?.price,0)}¢ · ${num(s.fills[0]?.q,2)} sh</p>'
assert old in js;js=js.replace(old,new)
old2='The gate never rearms. Later orders still follow the original pressure and inventory rules.'
new2='This gates the first quotes, not the first filled position. The gate never rearms; later orders follow the original pressure and inventory rules.'
assert old2 in js;js=js.replace(old2,new2);(UI/'opening.js').write_text(js)
index=(UI/'index.html').read_text();needle='<section class="controls" aria-label="Chart controls">'
assert needle in index and 'opening.html?market=3789484' not in index
index=index.replace(needle,'<p><a href="opening.html?market=3789484">New: compare the opening-only modification with default and maker #3</a></p>\n'+needle);(UI/'index.html').write_text(index)
stages=[]
for file in H.glob('*.log'):
 text=file.read_text(errors='replace');v=re.findall(r'wall_s=([0-9.]+)',text)
 if v:stages.append(dict(log=file.name,wall_upper_s=float(v[-1])))
render_text=(H/'browser_process.log').read_text(errors='replace')
render=re.findall(r'render_cpu_user_s=([0-9.]+) render_cpu_sys_s=([0-9.]+)',render_text)
render_charge=sum(map(float,render[-1])) if render else 25.0
upper=sum(z['wall_upper_s'] for z in stages)+render_charge+1.0
assert upper<60,upper
budget=dict(packet_cap_cpu_s=60,packet_upper_cpu_s=round(upper,2),stages=stages,render_cpu_charge_s=render_charge,publication_and_final_browser_reserve_s=1,task_cap_coreh=2.075,line_cap_coreh=5.375,standing_reserve_coreh=.05,prior_entry_side_packet_upper_cpu_s=17.2,prior_dashboard_packet_upper_cpu_s=179.5)
(H/'resources.json').write_text(json.dumps(budget,indent=2)+'\n')
report='''# Opening rule and chart — 2026-11-24

The modified rule waits for the available250ms Binance return to reach the fixed favorable-side pressure threshold before the first submitted batch. It is not a fixed waiting period, a fair-value forecast, or a probability>50 rule. For a bid, the signed BTC return must reach the frozen ask-adverse q5 edge+0.285456bps; for an ask, the downward move must reach the frozen bid-adverse edge0.466673bps. Actual float32/cache/bin behavior is inherited unchanged. The available-side current guard/inventory constraints also apply.

Only the first actual submission tick is gated. In3789484: no orders at0; signal+0.32485bps at550ms; bids49/48c submitted550ms, active560ms; original entry rules resume600ms. **These initial bids never fill.** A later ask55c is submitted900ms under the original rules and sells5shares at951ms. The extra gate waits for information before first quoting; it does not enforce an information-aligned first filled position or maintain one-sided entry until a fill.

Default fills191.24shares of asks50/51c at448ms. Avoiding that opening exposure changes the whole subsequent path: first-minute marked value-22.08USD→+2.08; full-market-17.40→+9.73. This is not proof the modified rule matches maker3's entry logic. Maker3's first passive record upper bound1.755s is a buy at56c, and its quote time remains uncertain. Simulated and recorded clocks differ. Both comparison arms carry the same corrected inventory logic; the historical original dashboard remains intact.

Other cases:3702960 submits asks at29.05s (normal entries resume29.10), first fill29.718s versus default0.881s;3877472 submits asks0.85s (resume0.90), both first fill1.448s. Maker3 has no recorded activity in3877472, and the page reports that rather than drawing an invented trace. Three-market pooled value remains-121.32USD versus-149.58 default. No strategy rerun, tuning or new data in this chart packet.

Local page: http://127.0.0.1:8765/opening.html?market=3789484 (existing SSH tunnel). Compare price/live-quote/fill paths, actual signed250ms cached signal, inventory and marked value. Blue shade covers the opening-gate period; separate boundaries show first submission and next-tick resumption. All three markets,3/15/60/full windows, signal jump, series toggles, scrub and event inspection. Maker markers stay settlement upper bounds; post windows preserve uncertainty. The page is self-contained; old data and frozen research files are untouched.

Checks:6arms /2,227 simulated fills /2,135orders /406 maker3 legs. All fills, order clocks/lifetimes, displayed inventory/value curves, first-batch scope and logged entry signal reconcile with saved evidence;6 corruption refusals. Original replay hashes bind reused controls and candidate arrays; inputs.json records every chart source. Browser: all3market selectors, zoom/jump, series toggle, event detail and keyboard cursor work; no JS errors or desktop/mobile overflow. The render process uses already installed browser libraries, outside run_light's4GiB virtual-address cap because Chromium cannot start within it; actual browser automation remains in the codex slot. Renderer is one-core/time-bounded and charged. No dependency install or external hosting.

Instrument history retained: first projection refused a differently named baseline array-hash field; corrected using the frozen opening manifest. The second projection completed all data then its summary failed on the genuinely empty maker3 population in3877472; summary-only recovery retains null, data unchanged. Initial checker used rounded displayed fill quantities to infer exact full-fill time; corrected to exact source quantities. The next attempt hit its12CPU-second guard because NPZ arrays were repeatedly decompressed; v3 loads each array once and passes unchanged checks. Two browser launch refusals (existing shared-library path, then virtual-address limit) precede the successful bounded CDP preview. These affect instruments, not simulated results; all attempts counted.

Next priority: explicitly resolve whether the desired opening unit is the first submitted batch or the first acquired position before any new replay. Current authorized implementation is the former; charts now expose the difference. Preserve the accepted opening-only result and avoid silently extending the entry filter. No claim that delayed fill times alone prove maker3 waits for information.
'''
(H/'report.md').write_text(report);(UI/'opening_report.md').write_text(report)
subprocess.run(['git','diff','--cached','--quiet'],cwd=R,check=True)
block='''<!-- OPENING_CHART_20261124_CODEX -->
## 2026-11-24 — opening comparison visible; first quote differs from first position

[Chart receipt](reviews/opening_chart_20261124/report.md); [self-contained page](reviews/maker_dashboard_20261124/opening.html), existing localhost8765 tunnel. Saved FIXED_DEFAULT vs OPENING_ONLY, same inventory correction, maker3;3market/3–298s views, actual cached signal, quotes/fills/inventory/value.6arms2227fills2135orders406maker legs verified,6 corruptions refused;desktop/mobile controls pass. No new replay.

Important scope:3789484 gate triggers550ms(+0.32485bps), initial bids49/48 activate560ms, original entries resume600ms. Those initial bids NEVER FILL. The first acquired position is a later default-rule ask55c fill5sh at951ms, from a900ms submission. Thus information gates first quoting, not first-position direction. It avoids default191.24sh ask50/51 fills448ms; first-minute value-22.08→+2.08USD, full-17.40→+9.73. Pooled remainsnegative. Maker3 first passive upper1.755s is not a known private entry time.

ONE next priority: clarify desired opening unit (first submitted batch vs first acquired position) before any new replay; do not silently broaden the gate. Resource packet<=60CPU seconds under standing short-CPU approval, prior charges intact, task2.075/line5.375/.05reserve; failures and browser renderer counted in receipt.
<!-- /OPENING_CHART_20261124_CODEX -->

'''
state=f'''opening_chart_20261124_codex:
  report: predictfun/reviews/opening_chart_20261124/report.md
  chart: predictfun/reviews/maker_dashboard_20261124/opening.html
  new_replay: false
  first_quote_is_not_first_position: true
  example_initial_bids_filled: false
  example_first_fill: "951ms ask55c 5shares; default-rule continuation"
  packet_upper_cpu_s: {upper:.2f}
  packet_cap_cpu_s: 60
'''
for rel,text in [('predictfun/HANDOFF.md',block),('predictfun/FINDINGS.md',block),('predictfun/STATUS.yml',state)]:
 p=R/rel;working=p.read_text();base=subprocess.check_output(['git','show','HEAD:'+rel],cwd=R).decode()
 assert 'OPENING_CHART_20261124_CODEX' not in working and 'opening_chart_20261124_codex:' not in working
 p.write_text(text+working);oid=subprocess.check_output(['git','hash-object','-w','--stdin'],cwd=R,input=(text+base).encode()).decode().strip();subprocess.run(['git','update-index','--cacheinfo','100644',oid,rel],cwd=R,check=True)
print('Published; first acquired position clarified; resource upper',upper)
