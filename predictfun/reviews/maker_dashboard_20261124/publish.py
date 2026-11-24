"""Verify copied extras, publish evidence and add only owned state blocks to the index."""
from pathlib import Path
import collections,hashlib,json,re,subprocess,gzip
import extract_v2 as E
H=E.H;D=json.loads(gzip.decompress((H/'data.json.gz').read_bytes()));extra=json.loads((H/'extra.js').read_text().removeprefix('window.MM_EXTRA=').rstrip(';\n'));R=E.load_records()
for m in D['cases']:
 for arm in E.ARMS:assert extra[str(m['id'])][arm]['structural']==R[m['id']][arm]['struct_log']
orders={r['order_hash']:r for r in E.read(E.R/'predictfun/reviews/maker_action_population_20261124/outputs/orders.csv')}
for h,row in extra['signed_orders'].items():assert row==orders[h]
assert extra['inventory_diagnosis']==json.loads((H/'inventory_diagnosis.json').read_text())
# Positive identity and known-bad copy refusal for the supplementary structural fields.
a=extra['3702960'][E.ARMS[0]]['structural'];bad=json.loads(json.dumps(a));bad[0]['gid']+=999;assert bad!=R[3702960][E.ARMS[0]]['struct_log']
(H/'extra_checks.json').write_text(json.dumps(dict(passed=True,arms=6,signed_orders=len(extra['signed_orders']),structural_identity_falsifier=True))+'\n')
rows=[]
for m in D['cases']:
 s=m['ours'][E.ARMS[0]]['summary'];f=m['ours'][E.ARMS[1]]['summary'];rows.append(f"| {m['day']} / {m['id']} | {s['pnl_usd']:+.2f} | {sum(m['maker_totals'].values()):+.2f} | {f['pnl_usd']:+.2f} | {s['generations']} / {s['cancels']} |")
diag=json.loads((H/'inventory_diagnosis.json').read_text());dr=[]
for x in diag:
 if x['arm']!=E.ARMS[0]:continue
 dr.append(f"| {x['m']} | {x['duration_s']:.2f} | {sum(z['no_live_ticks'] for z in x['segments'])*.05:.2f} | "+'; '.join(f"{z['start_s']:.2f}–{z['end_s']:.2f}: {z['position_before']:+.2f} → {z['position_start']:+.2f}" for z in x['segments'])+' |')
report='''# Market microscope — three saved market comparisons

USER task: compare our actions, market conditions and top-maker behavior in a usable frontend, to investigate queue/churn and why we lose. This packet adds no strategy replay, fit, market day, validation or live activity. Three known examples were fixed before extraction. As of 2026-11-24.

Open [the portable interactive dashboard](dashboard.html) (works directly from disk, no network), or serve this directory and open `dashboard.html?market=3789484`. [Desktop preview](desktop.png). Select a market/maker/side, switch current/front assumptions, zoom, click fills/orders, and follow linked maker-order candidates. Marker hard windows, inferred post bounds, original sizes where available, our received-book structural reasons, queue samples, exact commands and inventory/value paths are retained. Histories are not turned into identified maker cancels.

Source HEAD: e8c3f3a70c8c246ecf4fc30b4a5e7ed68ca1c307. Initial extraction freeze fbce28223d06d9640193ad8a0ec06bdf57f75058; superseding pre-output adapter freeze d436b11d87d85b98c083d952657aad15d5ed200f. All underlying inputs have hashes in manifest_v2.json; standalone decoded data equals the checked projection. No external messaging or push.

## Material finding: inventory hysteresis blocks the wrong side

**P1 — outcome interpretation affected; fix before drawing stronger queue conclusions.** `pf_branch_ladder.py:55–65` resets only when `abs(I) <= R`, unless the opposite S threshold is reached. A batch of fills can jump entirely over that reset band. Old LONG_RESTRICTED can persist while inventory is short (or vice versa). `blocked` at lines 287–289 then prevents the inventory-reducing side; sizing at lines 294–307 can reduce the other side below the one-share minimum. This can leave neither side quoting. No implementation was edited.

`check_inventory.py` independently sums all notified fills at every stored tick in all six arms and reconciles the saved known position. It executes only the source's pure next_mode function: LONG_RESTRICTED with I=-99.92717 stays LONG; controls I=-30 reset and I=-101 switch. This is a post-hoc saved-evidence diagnosis, not a new strategy candidate or a claim about the corrected P&L.

Current arm, measured 50 ms grids (durations stop at supported END=298s):

| Market | Wrong-side restriction seconds | Of those, no live quotes seconds | Entry jump, shares; interval seconds |
|---|---:|---:|---|
'''+ '\n'.join(dr)+'''

For 3702960, the known inventory jumps +102.07283 → -97.92717 at 29.85s, then reaches -99.92717; LONG_RESTRICTED persists through END. For 3877472 the same trap persists from132.35s toEND. The front-at-activation arms have different trap durations (268.15 /0 /15.35s), so their feedback-policy contrasts combine queue differences with this controller defect. Accounting remains the arithmetic of the policy that actually ran; it is not an estimate of the intended corrected policy.

## What the three markets show

USD, same supported numerical interval and common END mark. Recorded top-three includes maker+taker legs and self-trade fees, zero-start view. Our fills are simulated, maker fills recorded with different clock certainty. These are marked values, not settlement P&L or spread-only profit.

| Day / market | Our current END | Top 3 recorded END | Our hypothetical front END | Our generations / cancel requests |
|---|---:|---:|---:|---:|
'''+ '\n'.join(rows)+'''

- **3702960:** a profitable own replay, useful as a counterexample to “we always lose because of churn.” Maker #3 sells4sh at24¢ in hard window56.303–58.303s. Its10s and60s markouts are both+2.5¢/share. Our same/equal-better ask is absent throughout that window. Inventory is about-99.93 while the stale LONG mode remains; ask slot reasons are zero_desired. This missed overlap is not simply a queue race. G4 estimated posting window57.653–58.303s is a scenario, not proof of650ms continuous resting or priority.
- **3789484:** 554 generations and482 cancel requests (316 structural,161 pressure,5 inventory). Maker #1's10sh46¢ bid fill has hard window269.035–271.035s and G4 estimated post269.998–271.035s. Our bids traverse61/60/59/53/52/49¢ during that bound; no46¢ order overlaps. Our equal-or-better bid is present only part of the bound. The viewer links the observed32¢ successor candidate, without asserting an identified cancellation. Queue-front sensitivity improves this market, but includes different inventory/controller paths.
- **3877472:** maker #1's59¢ ask trades50.187169sh by29.176s, with further same-hash fills. Our own59¢ ask (gid49) is already live24.71–37.51s, covering the entire27.176–29.176s hard match window; the sidebar exposes its seeded/first-review queue. This is compatible with queue disadvantage, but public data do not identify the maker's true priority. Merely assuming front priority worsens this market's END value; it is not sufficient for profitability.

The examples are illustrative and outcome-aware navigation, not a representative or matched selection test. The whole supported market trace remains available. Missing marks remain missing. Good10/60s fills are not proof of profitable complete turnover. Maker zero-start flow is not actual wallet inventory.

## Verification and limitations

Independent artifact check: six arms,2,297 own fills,1,819 own generations,2,198 recorded top-maker legs;18 planted wrong-side/price/time/value/front/queue corruptions refused. Source check: all2,198 maker legs,1,685 hard maker match windows,1,280 enabled scenario post windows and450 public-book samples;three altered source markouts refused. Source counts reconcile (10 recorded legs outside the shown interval on Oct27). Extra structural fields equal all six raw logs;18 signed-order context records equal the hash-bound source. Raw known inventory is independently reconciled at every stored grid. Browser desktop/mobile tests exercise all markets, arms, examples, zoom, side controls, event drilldown and keyboard; no JS errors or horizontal overflow. Portable HTML makes no network requests.

Preserved failures: the first extraction refused before output because it excluded self-trade fees included by the accepted totals. Amendment v2 retains flagged self legs and handles verified empty ranks. The v2/v3 checker failures came from summing six-decimal display quantities with exact tolerances/full-fill thresholds. v4 uses explicit rounding bounds for sums and raw exact fills for completion; projection unchanged. Logs and old source are kept. These checks establish faithful display, not predictive skill or profitability.

Browser environment: Chromium cannot initialize within run_light's4GiB virtual-address limit. Research extraction/checks all used PF_SLOT=codex/run_light. Rendering-only Chromium used core8 separately with timed RSS and no new research calculation, while the Playwright driver stayed in run_light. A conservative browser charge includes its renderers; this is recorded in resources.json. No resource limits were changed on research jobs.

Largest limitation: the saved controller's sign-crossing restriction defect materially affects participation and feedback; in addition, maker private cancellations and continuous queue positions remain unidentified. The chart shows source facts/bounds, not a proven maker decision policy.

**ONE next priority:** the development session should fix sign-crossing hysteresis release, add a batch-fill control that jumps across the entire deadband, and rerun exactly these three fixed markets for CURRENT_ASYNC_FAST10 and FRONT_ASYNC_FAST10 with the same ledgers/metrics. Compare participation, quotes on the inventory-reducing side, turnover/residual and common-END value before interpreting a queue benefit. This is a new strategy replay and needs a separately bounded authorization; the current approved0.05h packet is saved-data analysis only. Do not tune thresholds or expand days to hide the failure. The dashboard and diagnosis are complete; no replay is launched here.
'''
(H/'report.md').write_text(report)
# Charge every recorded tag once; preserve previous conservative charges.
log=(E.R/'predictfun/research/run_light.log').read_text();runs=[]
for line in log.splitlines():
 if 'tag=mt_dash_' not in line:continue
 fields=dict(re.findall(r'(\w+)=([^ ]+)',line));runs.append(dict(tag=fields['tag'],rc=int(fields['rc']),cpu_user_s=float(fields['cpu_user_s'])))
previous=json.loads((E.R/'predictfun/reviews/maker_fill_age_20261124/resources.json').read_text());seconds=sum(x['cpu_user_s'] for x in runs);charge=seconds+15+2 # all browser processes + final publication / remaining tiny checks reserve
resources=dict(runs=runs,measured_research_cpu_s=seconds,browser_conservative_cpu_s=15,publication_reserve_s=2,packet_charged_cpu_s=charge,packet_cap_s=180,task_upper_coreh=previous['task_upper_coreh']+charge/3600,line_upper_coreh=previous['line_upper_coreh']+charge/3600,standing_reserve_coreh=.05,task_cap_coreh=2.05,line_cap_coreh=5.35)
assert charge<=180 and resources['task_upper_coreh']+.05<=2.05 and resources['line_upper_coreh']+.05<=5.35
(H/'resources.json').write_text(json.dumps(resources,indent=2)+'\n')
# Stage additive owned blocks independently of other sessions' unstaged content.
blocks={
'predictfun/HANDOFF.md':'''<!-- MARKET_DASHBOARD_20261124_CODEX -->
## 2026-11-24 — three-market microscope and material inventory finding

[Interactive dashboard](reviews/maker_dashboard_20261124/dashboard.html); [report](reviews/maker_dashboard_20261124/report.md). USER-approved <=0.05coreh saved-data packet, no replay/new days. Six saved arms/three markets,2,297 own fills and2,198 maker legs; source/accounting and desktop/mobile checks pass. Prices,our orders/queue samples,cancel causes,inventory and maker hard/inferred windows are aligned; no private maker pull/queue rank claimed.

P1: pf_branch_ladder.next_mode retains the old restriction when a fill jumps over abs(I)<=50 but does not reach the opposite100 threshold. Current traces have wrong-side restriction for268.15/27.85/165.65s (markets3702960/3789484/3877472), independently reconciled to noticed fills. At-99.93sh LONG_RESTRICTED blocks bids while sizing makes asks<1share: near-total inactivity. Our current END values+29.42/-17.90/-6.54USD and front+29.19/+47.90/-50.80 are as-ran,not corrected-controller values. Queue contrasts include this path-dependent controller trap.

ONE next priority: development session fixes sign-crossing hysteresis release with a skipped-deadband batch-fill control,then reruns these exactthree CURRENT/FRONT pairs before further queue inference. No threshold tuning/day expansion. New replay needs its own bounded authorization; this packet only approved saved-data analysis. Resource receipt preserves old charges; task cap2.05,line5.35,standing.05reserve. No push/external messaging/implementation edit.
<!-- /MARKET_DASHBOARD_20261124_CODEX -->

''',
'predictfun/FINDINGS.md':'''<!-- MARKET_DASHBOARD_20261124_CODEX -->
## 2026-11-24 — dashboard exposes sign-crossing inventory restriction trap

[Evidence and source lines](reviews/maker_dashboard_20261124/report.md), [viewer](reviews/maker_dashboard_20261124/dashboard.html). P1: next_mode's absolute reset band can be jumped by grouped fills; old LONG_RESTRICTED survives short inventory between-50and-100 (symmetric SHORT case). Current saved traces:268.15/27.85/165.65s on the wrong side. Every known inventory grid reconciles to raw noticed fills; pure-function positive/control cases reproduce. This affects participation and the interpretation of prior queue/front feedback contrasts; accounting still describes the executed policy. No implementation or replay changed. Next priority: fix release semantics and repeat the same three pairs under a separately bounded replay authorization.
<!-- /MARKET_DASHBOARD_20261124_CODEX -->

''',
'predictfun/STATUS.yml':'''market_dashboard_20261124_codex:
  report: predictfun/reviews/maker_dashboard_20261124/report.md
  dashboard: predictfun/reviews/maker_dashboard_20261124/dashboard.html
  stage: saved_trace_dashboard_verified_material_controller_issue
  markets: [3702960, 3789484, 3877472]
  own_fills_six_arms: 2297
  recorded_maker_legs: 2198
  implementation_changed: false
  sign_crossing_hysteresis_defect: true
  current_wrong_side_restriction_seconds: [268.15, 27.85, 165.65]
  next_priority: fix_skipped_deadband_release_then_same_three_market_pairs
  new_replay_requires_bounded_authorization: true
  packet_cap_coreh: 0.05
  task_cap_coreh: 2.05
  line_cap_coreh: 5.35
  standing_reserve_coreh: 0.05
'''}
for name,block in blocks.items():
 p=E.R/name;current=p.read_text();head=subprocess.check_output(['git','show','HEAD:'+name],cwd=E.R).decode();assert 'market_dashboard_20261124_codex:' not in head and '<!-- MARKET_DASHBOARD_20261124_CODEX -->' not in head
 p.write_text(block+current);blob=subprocess.check_output(['git','hash-object','-w','--stdin'],input=(block+head).encode(),cwd=E.R).decode().strip();subprocess.run(['git','update-index','--cacheinfo','100644,'+blob+','+name],cwd=E.R,check=True)
print(json.dumps(resources))
