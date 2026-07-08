export const meta = {
  name: 'v4-tuning-validate-optimize',
  description: 'From vanilla v4: validate each production tuning-logic against the data (check->validate-issue->apply->review), apply only what is genuinely needed, build the optimized stack, then optimize the side-sign lever',
  phases: [
    { title: 'CheckValidate', detail: 'per-logic: what it does + is the issue real in vanilla v4' },
    { title: 'Apply', detail: 'run vanilla+logic through the bot (OOS+in-sample)' },
    { title: 'Review', detail: 'issue-real AND helps-OOS => APPLY else REJECT' },
    { title: 'Stack', detail: 'cumulatively apply the APPLY logics, test the stack' },
    { title: 'SideSign', detail: 'the validated lever: PIT side-short sign detection' },
    { title: 'Synthesize', detail: 'optimized stack + honest verdict' },
  ],
}

const CTX = `
CONVEXITY v4 optimization from VANILLA. Vanilla v4 = raw two-book per-symbol RidgeCV (residual target), gates OFF,
K_long=1/K_short=2, inv-vol sizing, ALL regimes traded L/S uniformly. Through the real bot (fee 4.5 + funding + depth
slippage ~14.5 bps/fill) vanilla nets ~-1.30 Sharpe OOS 2023-25 / ~+3 in-sample 2025-10+.

VALIDATED ROOT CAUSE (deep_validate_regime.py, raw gross alpha per year, no cost/gates):
  2023 +56 | 2024 -9 | 2025H1 -67 | 2025H2 +136 | 2026 +110.  Issue REAL (gross alpha itself flips negative).
  Mechanism: the TIP inverts (avg MR-IC stays +0.01..0.02 every year; dispersion did NOT collapse). SIDE-DRIVEN:
  BEAR tip +every period (+211/+109/+79/+46 robust anchor); BULL -every period (robust avoid); SIDE swings sign
  (+62/-4/-37/+229 = the regime-fragility source).

THE HARNESS (locked protocol; agents cannot alter): live/tuning_harness.py <config.json> <out.json>
  config.json = {"name": str, "env": {"KEY":"VAL",...}}  -> runs VANILLA + those env overrides through the bot on
  BOTH windows, returns {oos:{sharpe,pnl,maxdd,stop_pct,per_regime,per_year}, ins:{...}}. ~3-4 min/config (2 bot runs).
VANILLA baseline metrics are in live/state/longtail/tune/_cfg/vanilla_m.json (read it for the reference numbers).

RULE: a tuning-logic is APPLIED only if (1) the ISSUE it targets is really present in vanilla v4 AND (2) it improves
OOS net (not just in-sample) without wrecking maxDD. In-sample-only improvement = REJECT (overfit). Prior session found
most fine-knobs are neutral-to-overfit; be adversarial.
`

// ---- the production tuning-logics to validate 1-by-1, each with the issue it targets ----
const LOGICS = [
  { id:'regime_gate', targets:'trailing-drawdown regimes (2024/2025H1 are gross-negative) need de-grossing',
    env:{REGIME_GATE:'1',REGIME_GATE_W:'180',REGIME_GATE_FLOOR:'0.0',REGIME_GATE_K:'2',REGIME_GATE_MINHIST:'60',REGIME_GATE_MODE:'binary',REGIME_GATE_UNIV:'full'} },
  { id:'bull_hedge', targets:'bull is a persistent loser (-48/-187/-190) -> sit-out longs + BTC hedge',
    env:{BULL_MODE:'sidealpha',BULL_GROSS_MULT:'1',BULL_LONG_MULT:'0.25',BULL_LONG_INSTRUMENT:'btc',BTC_HEDGE_COST_BPS:'2',BULL_K:'2',STRAT_HOLD_BULL:'1',BULL_SHORT_RANK:'return_1d',BULL_DEEP_THR:'0.15'} },
  { id:'bear_ramp', targets:'shallow-bear short is anti-alpha; scale bear gross with drawdown depth',
    env:{BEAR_DEPTH_RAMP:'1',BEAR_DEPTH_D0:'0.10',BEAR_DEPTH_D1:'0.30',BEAR_DEPTH_FLOOR:'0.0'} },
  { id:'bear_k2', targets:'bear wants more depth than K=1/2 (bear tip robust, farm wider)', env:{BEAR_K:'2'} },
  { id:'dd_stop', targets:'runaway drawdowns in side/bull need a DD stop', env:{STOP_SKIP_REGIMES:'bear',STOP_K_SIGMA:'2.0'} },
  { id:'conc_cap', targets:'over-concentration in a single name', env:{CONC_CAP:'0.40'} },
  { id:'short_filter', targets:'shorting already-crashed names (ret3d very negative) loses', env:{SHORT_MIN_RET3D:'-0.20'} },
  { id:'kshort3', targets:'the short is a broad rank-2..6 band; K_short=3 diversifies', env:{STRAT_K:'3'} },
]

phase('CheckValidate')
const checked = await parallel(LOGICS.map(L => () =>
  agent(`${CTX}\n\nLOGIC "${L.id}" targets: ${L.targets}\nenv = ${JSON.stringify(L.env)}\n
Do TWO things: (1) CHECK — in 1-2 sentences, what does this knob mechanically do in the bot. (2) VALIDATE THE ISSUE —
using the validated per-year/per-regime data in CTX (and, if useful, read vanilla_m.json), state whether the issue it
targets is REALLY present in vanilla v4, with the specific numbers. Return {id, mechanism, issue_present:boolean, evidence}.`,
    { phase:'CheckValidate', label:`check:${L.id}`, schema:{ type:'object', properties:{
      id:{type:'string'}, mechanism:{type:'string'}, issue_present:{type:'boolean'}, evidence:{type:'string'} },
      required:['id','issue_present','evidence'] } })
)).then(r=>r.filter(Boolean))

phase('Apply')
// SEQUENTIAL bot runs (memory). Run every logic (even if issue-doubtful) so review has full data.
const applied = []
for (const L of LOGICS) {
  const r = await agent(
    `Run the harness on ONE tuning-logic vs vanilla. Steps EXACTLY:
1. Write to live/state/longtail/tune/_cfg/${L.id}.json :
${JSON.stringify({ name: L.id, env: L.env })}
2. Run: cd /home/yuqing/ctaNew && ulimit -v 20000000 && OMP_NUM_THREADS=1 python3 live/tuning_harness.py live/state/longtail/tune/_cfg/${L.id}.json live/state/longtail/tune/_cfg/${L.id}_m.json
3. Read live/state/longtail/tune/_cfg/${L.id}_m.json and return {name, oos_sharpe, oos_pnl, oos_maxdd, ins_sharpe, ins_pnl, oos_per_regime}. On error/timeout return oos_sharpe null.`,
    { phase:'Apply', label:`apply:${L.id}`, schema:{ type:'object', properties:{
      name:{type:'string'}, oos_sharpe:{type:['number','null']}, oos_pnl:{type:['number','null']},
      oos_maxdd:{type:['number','null']}, ins_sharpe:{type:['number','null']}, ins_pnl:{type:['number','null']},
      oos_per_regime:{type:'object'} }, required:['name'] } })
  if (r) applied.push(r); log(`applied ${L.id}: OOS Sh ${r?.oos_sharpe}`)
}

phase('Review')
const reviewed = await parallel(LOGICS.map(L => () => {
  const chk = checked.find(c=>c.id===L.id) || {}
  const app = applied.find(a=>a.name===L.id) || {}
  return agent(`${CTX}\n\nADVERSARIAL REVIEW of tuning-logic "${L.id}".
Issue check: ${JSON.stringify(chk)}\nApply result (vs vanilla OOS ~-1.30): ${JSON.stringify(app)}\n
Decide verdict. APPLY only if BOTH: issue_present==true AND it improves OOS Sharpe/PnL vs vanilla (read
vanilla_m.json for exact vanilla OOS numbers) without materially worsening maxDD. If it only helps in-sample -> REJECT
(overfit). If the issue is NOT present -> REJECT (solves a non-problem). Capital-preservation logics may keep Sharpe ~flat
but must cut OOS loss/maxDD to APPLY. Return {id, verdict:"APPLY"|"REJECT", reason, keep_env}.`,
    { phase:'Review', label:`review:${L.id}`, schema:{ type:'object', properties:{
      id:{type:'string'}, verdict:{type:'string'}, reason:{type:'string'}, keep_env:{type:'object'} },
      required:['id','verdict','reason'] } })
})).then(r=>r.filter(Boolean))

phase('Stack')
// cumulatively apply APPLY-verdict logics, test the combined stack OOS+in-sample
const keeps = reviewed.filter(r=>r.verdict==='APPLY')
let stackEnv = {}
for (const k of keeps) { const L=LOGICS.find(x=>x.id===k.id); if(L) stackEnv={...stackEnv,...L.env} }
let stackRes = null
if (Object.keys(stackEnv).length) {
  stackRes = await agent(
    `Run the harness on the CUMULATIVE optimized stack (vanilla + all APPLY logics). Steps:
1. Write to live/state/longtail/tune/_cfg/STACK.json : ${JSON.stringify({ name:'STACK', env: stackEnv })}
2. Run: cd /home/yuqing/ctaNew && ulimit -v 20000000 && OMP_NUM_THREADS=1 python3 live/tuning_harness.py live/state/longtail/tune/_cfg/STACK.json live/state/longtail/tune/_cfg/STACK_m.json
3. Read STACK_m.json, return {oos_sharpe,oos_pnl,oos_maxdd,ins_sharpe,ins_pnl,oos_per_regime,oos_per_year}.`,
    { phase:'Stack', label:'stack', schema:{ type:'object', properties:{
      oos_sharpe:{type:['number','null']}, oos_pnl:{type:['number','null']}, oos_maxdd:{type:['number','null']},
      ins_sharpe:{type:['number','null']}, ins_pnl:{type:['number','null']}, oos_per_regime:{type:'object'}, oos_per_year:{type:'object'} } } })
  log(`STACK OOS Sh ${stackRes?.oos_sharpe} ins ${stackRes?.ins_sharpe}`)
}

phase('SideSign')
// the validated optimization lever: is the side-short sign PIT-detectable? Propose + test detectors as REGIME_GATE-like side gates.
const detectors = [
  { id:'ss_trailedge', targets:'gate SIDE gross on trailing realized side-short edge (reactive persistence)', env:{REGIME_GATE:'1',REGIME_GATE_UNIV:'side',REGIME_GATE_W:'90',REGIME_GATE_K:'2',REGIME_GATE_MINHIST:'40'} },
  { id:'ss_trailedge_long', targets:'same but longer 180-window trailing side edge', env:{REGIME_GATE:'1',REGIME_GATE_UNIV:'side',REGIME_GATE_W:'180',REGIME_GATE_K:'2',REGIME_GATE_MINHIST:'60'} },
]
const ssResults = []
for (const D of detectors) {
  const r = await agent(
    `Test a PIT SIDE-SIGN detector (the validated lever). It gates SIDE gross on a trailing signal (${D.targets}).
Steps: write live/state/longtail/tune/_cfg/${D.id}.json = ${JSON.stringify({name:D.id,env:{...stackEnv,...D.env}})}
(NOTE: layered on the optimized STACK). Run: cd /home/yuqing/ctaNew && ulimit -v 20000000 && OMP_NUM_THREADS=1 python3 live/tuning_harness.py live/state/longtail/tune/_cfg/${D.id}.json live/state/longtail/tune/_cfg/${D.id}_m.json
Read the metrics json, return {id, oos_sharpe, oos_pnl, oos_maxdd, ins_sharpe, oos_per_year}.`,
    { phase:'SideSign', label:`ss:${D.id}`, schema:{ type:'object', properties:{
      id:{type:'string'}, oos_sharpe:{type:['number','null']}, oos_pnl:{type:['number','null']},
      oos_maxdd:{type:['number','null']}, ins_sharpe:{type:['number','null']}, oos_per_year:{type:'object'} }, required:['id'] } })
  if (r) ssResults.push(r); log(`sidesign ${D.id}: OOS Sh ${r?.oos_sharpe}`)
}

phase('Synthesize')
const summary = await agent(
  `${CTX}\n\nSynthesize the full vanilla->optimized run.\n
Per-logic review: ${JSON.stringify(reviewed,null,1)}\n
Apply metrics: ${JSON.stringify(applied,null,1)}\n
Optimized STACK: ${JSON.stringify(stackRes,null,1)}\n
Side-sign detectors: ${JSON.stringify(ssResults,null,1)}\n
Write a concise honest verdict (<=250 words): (1) which tuning-logics validated (issue real AND helps OOS) vs which
were rejected (non-issue or in-sample-only overfit); (2) the optimized-stack OOS+in-sample numbers vs vanilla (-1.30 OOS
/ ~+3 ins); (3) did any side-sign detector lift OOS (the key lever)? (4) is the strategy now OOS-viable or still
regime-confined? Be specific with numbers, no hype.`,
  { phase:'Synthesize', label:'synthesize' })

return { vanilla_oos:-1.30, reviewed, applied, stack_env:stackEnv, stack:stackRes, sidesign:ssResults, verdict:summary }
