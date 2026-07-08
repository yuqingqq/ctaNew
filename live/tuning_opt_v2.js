export const meta = {
  name: 'v4-full-v3-logics-optimize',
  description: 'Validate ALL v3 production logics 1-by-1 on vanilla v4 (regime-modes/gates/filters/sizing/gross), then greedy-forward-optimize the config (recent-primary objective, OOS guardrail)',
  phases: [
    { title: 'Validate', detail: 'apply each new v3 logic on vanilla v4, both windows' },
    { title: 'Review', detail: 'classify APPLY/NEUTRAL/REJECT (recent-primary, OOS guardrail)' },
    { title: 'Optimize', detail: 'greedy forward selection, iterative rounds' },
    { title: 'Synthesize', detail: 'optimized config + honest verdict' },
  ],
}

const CTX = `
CONVEXITY v4 optimization from VANILLA (raw two-book, residual target, gates OFF, K_long=1/K_short=2, inv_sqrt_vol).
Through the real bot (fee4.5 + funding + depth-slippage ~14.5 bps/fill) vanilla = recent(2025-10..2026-06, per-month
rolling WF) +1.263 Sharpe / backward-OOS(2023-25) -1.301.
POLICY (user): RECENT 10-12mo is the primary performance reference (backward-OOS is survivorship-distorted + regime-
changed, so it's a GUARDRAIL not a target). A logic is good if it lifts RECENT without materially breaking OOS
(OOS-degrade flags overfit). Honest recent ref: bootstrap median +2.68, 90% CI [+0.83,+4.51] IF regime holds; October
is ~55% of recent gross (concentration risk).
VALIDATED root cause (deep_validate_regime.py): raw gross alpha flips negative 2024/2025H1; the TIP inverts, SIDE-driven
(BEAR + every period = robust anchor +211/+109/+79/+46; BULL - every period; SIDE swings sign). Prior loop: only
regime_gate validated (capital preservation); bull_hedge/dd_stop overfit; side-sign trailing detector FAILED. KEY
UNTESTED: the bear anchor is UNFARMED (vanilla BEAR_MODE=flat) + the whole SIZING/GROSS layer.

HARNESS (locked): live/tuning_harness.py <config.json> <out.json>; config={"name":str,"env":{K:V}} -> VANILLA+overrides
through the bot on BOTH windows -> {oos:{sharpe,pnl,maxdd,per_regime,per_year}, ins:{...}}. ~4 min/config.
`

// Complete v3 + sizing/gross logic catalog (the NEW / re-contextualized ones; regime_gate carried as known-APPLY)
const LOGICS = [
  { id:'bear_mode_equal', targets:'FARM THE ROBUST BEAR ANCHOR — vanilla BEAR_MODE=flat leaves bear unfarmed; bear is + every period', env:{BEAR_MODE:'equal'} },
  { id:'bear_mode_equal_k2', targets:'bear anchor at depth 2 (BEAR_K activates only under BEAR_MODE=equal)', env:{BEAR_MODE:'equal',BEAR_K:'2'} },
  { id:'bear_ramp_on_equal', targets:'shallow-bear short anti-alpha; depth-ramp bear gross (needs BEAR_MODE=equal)', env:{BEAR_MODE:'equal',BEAR_DEPTH_RAMP:'1',BEAR_DEPTH_D0:'0.10',BEAR_DEPTH_D1:'0.30',BEAR_DEPTH_FLOOR:'0.0'} },
  { id:'side_beta_neut', targets:'strip residual BTC-beta from the side book', env:{SIDE_BETA_NEUT:'1'} },
  { id:'sizing_equal', targets:'VALIDATE inv_sqrt_vol vs equal-weight (inv_sqrt_vol was baked in, never tested)', env:{SIZING_MODE:'equal'} },
  { id:'sizing_inv_vol', targets:'stronger vol down-weighting than inv_sqrt', env:{SIZING_MODE:'inv_vol'} },
  { id:'sizing_volcap', targets:'cap high-vol name weight', env:{SIZING_MODE:'volcap'} },
  { id:'short_conv_tilt', targets:'conviction-weighted short sizing (deeper pred -> larger)', env:{SHORT_CONV_TILT:'0.5'} },
  { id:'vol_target', targets:'TOTAL-GROSS vol-targeting: de-gross in high-vol cycles (gross*=clip(150/trailvol,0.3,1))', env:{VOL_TARGET:'150',VOL_TARGET_WIN:'30',VOL_TARGET_FLOOR:'0.30',VOL_TARGET_CAP:'1.00'} },
  { id:'auto_sizer', targets:'PIT adaptive regime-bucket gross throttle (cuts a btc-return bucket after its trailing PnL goes neg) — a leading side-sign proxy', env:{AUTO_SIZER:'1',AUTO_BINW:'0.04',AUTO_MINLOOK:'20',AUTO_THROTTLE:'0.5'} },
  { id:'regime_gate', targets:'trailing-edge de-gross (known-APPLY, capital preservation)', env:{REGIME_GATE:'1',REGIME_GATE_W:'180',REGIME_GATE_K:'2',REGIME_GATE_MINHIST:'60',REGIME_GATE_MODE:'binary',REGIME_GATE_UNIV:'full'} },
]

phase('Validate')
const applied = []
for (const L of LOGICS) {
  const r = await agent(
    `${CTX}\n\nValidate ONE v3 logic on vanilla v4. LOGIC "${L.id}" targets: ${L.targets}. env=${JSON.stringify(L.env)}
Steps EXACTLY:
1. Write live/state/longtail/tune/_cfg/v2_${L.id}.json = ${JSON.stringify({ name:`v2_${L.id}`, env:L.env })}
2. Run: cd /home/yuqing/ctaNew && ulimit -v 20000000 && OMP_NUM_THREADS=1 python3 live/tuning_harness.py live/state/longtail/tune/_cfg/v2_${L.id}.json live/state/longtail/tune/_cfg/v2_${L.id}_m.json
3. Read v2_${L.id}_m.json, return {id:"${L.id}", oos_sharpe,oos_pnl,oos_maxdd, ins_sharpe,ins_pnl, oos_per_regime, ins_per_regime}. On error return oos_sharpe null.`,
    { phase:'Validate', label:`val:${L.id}`, schema:{ type:'object', properties:{
      id:{type:'string'}, oos_sharpe:{type:['number','null']}, oos_pnl:{type:['number','null']}, oos_maxdd:{type:['number','null']},
      ins_sharpe:{type:['number','null']}, ins_pnl:{type:['number','null']}, oos_per_regime:{type:'object'}, ins_per_regime:{type:'object'} },
      required:['id'] } })
  if (r) applied.push(r); log(`validated ${L.id}: recent ${r?.ins_sharpe} / OOS ${r?.oos_sharpe}`)
}

phase('Review')
const reviewed = await parallel(applied.map(a => () =>
  agent(`${CTX}\n\nReview logic "${a.id}" (targets: ${LOGICS.find(x=>x.id===a.id)?.targets}).
Result vs vanilla (recent +1.263 / OOS -1.301): ${JSON.stringify(a)}
Verdict under RECENT-PRIMARY / OOS-GUARDRAIL: APPLY if it lifts RECENT Sharpe meaningfully (>~+0.15) AND OOS does not
materially degrade (OOS Sharpe not worse than ~-1.6 and OOS PnL not much worse). NEUTRAL if ~flat or no-op. REJECT if it
lifts recent only by breaking OOS (overfit) or hurts recent. Note if it targets the robust BEAR anchor (special interest).
Return {id, verdict:"APPLY"|"NEUTRAL"|"REJECT", reason, recent_delta:number}.`,
    { phase:'Review', label:`rev:${a.id}`, schema:{ type:'object', properties:{
      id:{type:'string'}, verdict:{type:'string'}, reason:{type:'string'}, recent_delta:{type:['number','null']} }, required:['id','verdict'] } })
)).then(r=>r.filter(Boolean))

phase('Optimize')
// greedy forward selection over non-REJECT candidates; objective = recent Sharpe, guardrail = OOS not < -1.65
const pool = LOGICS.filter(L => (reviewed.find(r=>r.id===L.id)?.verdict) !== 'REJECT')
log(`greedy pool (${pool.length}): ${pool.map(l=>l.id).join(', ')}`)
let stackEnv = {}, chosen = [], curRecent = 1.263, curOOS = -1.301
const trace = [{ round:0, stack:[], recent:curRecent, oos:curOOS }]
const MAXR = 5
for (let round=1; round<=MAXR; round++) {
  const cands = pool.filter(L => !chosen.includes(L.id))
  if (!cands.length) break
  const results = []
  for (const L of cands) {
    const env = { ...stackEnv, ...L.env }
    const nm = `v2_greedy_r${round}_${L.id}`
    const r = await agent(
      `Greedy round ${round}: test adding "${L.id}" to the current stack.
1. Write live/state/longtail/tune/_cfg/${nm}.json = ${JSON.stringify({ name:nm, env })}
2. Run: cd /home/yuqing/ctaNew && ulimit -v 20000000 && OMP_NUM_THREADS=1 python3 live/tuning_harness.py live/state/longtail/tune/_cfg/${nm}.json live/state/longtail/tune/_cfg/${nm}_m.json
3. Read ${nm}_m.json, return {id:"${L.id}", oos_sharpe, ins_sharpe, oos_pnl, ins_pnl, oos_maxdd}.`,
      { phase:'Optimize', label:`r${round}:${L.id}`, schema:{ type:'object', properties:{
        id:{type:'string'}, oos_sharpe:{type:['number','null']}, ins_sharpe:{type:['number','null']},
        oos_pnl:{type:['number','null']}, ins_pnl:{type:['number','null']}, oos_maxdd:{type:['number','null']} }, required:['id'] } })
    if (r && r.ins_sharpe!=null) results.push({ ...r, env:L.env })
  }
  // pick best by recent Sharpe among those passing OOS guardrail (OOS not worse than -1.65)
  const ok = results.filter(r => (r.oos_sharpe ?? -99) >= -1.65)
  const cand = (ok.length ? ok : results).sort((a,b)=>(b.ins_sharpe??-99)-(a.ins_sharpe??-99))[0]
  if (!cand || (cand.ins_sharpe ?? -99) <= curRecent + 0.05) { log(`round ${round}: no improvement (best ${cand?.id} recent ${cand?.ins_sharpe}) -> stop`); break }
  stackEnv = { ...stackEnv, ...cand.env }; chosen.push(cand.id)
  curRecent = cand.ins_sharpe; curOOS = cand.oos_sharpe
  trace.push({ round, added:cand.id, stack:[...chosen], recent:curRecent, oos:curOOS })
  log(`round ${round}: ADDED ${cand.id} -> recent ${curRecent} / OOS ${curOOS}`)
}

phase('Synthesize')
const summary = await agent(
  `${CTX}\n\nSynthesize the full v3-logics validation + greedy optimization from vanilla v4.
Individual review verdicts: ${JSON.stringify(reviewed,null,1)}
Greedy trace (recent-primary forward selection): ${JSON.stringify(trace,null,1)}
Final optimized stack env: ${JSON.stringify(stackEnv)}
Write a concise honest verdict (<=260 words): (1) which v3 logics validated vs rejected on v4, with special attention to
the BEAR-ANCHOR (bear_mode_equal) and the SIZING/GROSS layer (sizing modes, vol_target, auto_sizer) that were untested
before; (2) the greedy-optimized config's recent + OOS numbers vs vanilla (+1.263 / -1.301); (3) did farming the bear
anchor or any sizing/gross knob finally lift performance robustly, or is it still regime-confined? (4) the recommended
deployable config. Be specific with numbers, no hype.`,
  { phase:'Synthesize', label:'synthesize' })

return { reviewed, applied, greedy_trace:trace, optimized_env:stackEnv, final_recent:curRecent, final_oos:curOOS, verdict:summary }
