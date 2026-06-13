export const meta = {
  name: 'convexity-opt-pass',
  description: 'Anti-overfit optimization of convexity v2: enumerate optimizable levers (data-driven), test each through the real bot replay with honest gates, synthesize. Rejects hand-tuned thresholds.',
  phases: [
    { title: 'Enumerate', detail: '5 agents propose parameter-free / nested-validatable candidate levers from distinct angles' },
    { title: 'Test', detail: 'each env-testable candidate run through the real --replay-all via opt_eval, honest gates' },
    { title: 'Synthesize', detail: 'adversarial judge: overfit-penalized ranking + honest recommendation' },
  ],
}

const SHARED = `
CONTEXT — convexity v2 strategy (Binance low-vol USDM perps, cross-sectional mean-reversion):
- dual-pred (long=resid-rev model, short=base V0), K=3 long / K=3 short, inv_vol sizing, equal-weight legs
  (SIDE_BETA_NEUT=0), 24h hold via 6 overlapping sleeves, regime gate (BTC 30d return: >+10% bull / <-10% bear /
  else side), bear traded equal-weight K=2, plus the LIVE long-winner gate LONG_MAX_RET3D=0.20.
- BASELINE = Sharpe +4.22 / maxDD -2777 / totPnL +16731 (full OOS 2025-10-04..2026-06-04), cycles at
  live/state/v3loop/iter5_tilt0/cycles.csv. Per-leg: SHORT carries it (+20200, Sh +3.57); LONG tradeable -1758
  (but alpha +7148 — load-bearing). maxDD is all bear-regime.
- The bot is live/convexity_paper_bot.py; its levers are env vars (grep 'os.environ' to enumerate them:
  SIZING_MODE, SIDE_MODE, STRAT_K/STRAT_K_LONG/STRAT_K_SHORT, BULL_MODE, BEAR_MODE, BEAR_K, SLEEVE_DECAY_TAU,
  ENTRY_HOUR_SCALE/SKIP_ENTRY_HOURS, DISP_GATE, MIN_HISTORY_DAYS, LIQ_FLOOR_DOLLAR_VOL_30D, VOLCAP_PCTILE, etc.).
- Full prior-work ledger: docs/convexity_v3_loop.md (read it — it documents what's been tried).

THE EVALUATOR (use EXACTLY this; ~5 min per run, apples-to-apples):
  python3 live/opt_eval.py <label> "ENV1=v1 ENV2=v2"
  -> prints one JSON line: {sharpe, maxdd, totpnl, lift (vs +4.22 baseline), maxdd_change_pct, folds_positive (of 9
     vs baseline per-fold PnL), per_fold_delta}. Empty override string "" = baseline reproduction.

HARD ANTI-OVERFIT MANDATE (the user explicitly rejected hand-tuned thresholds like the mid-bear band -0.22/-0.13):
- A candidate is ONLY acceptable if it is PARAMETER-FREE (a discrete/structural choice, or a principled continuous
  rule with NO cutoff fit to the sample) OR its single parameter is validated nested-OOS (calibrate on past folds,
  apply forward). Any "magic number picked because it works on this sample" is DISQUALIFIED as overfit.
- HONEST GATES for a real win: lift >= +0.30 Sharpe AND folds_positive >= 6/9 AND not concentrated in 1-2 folds
  AND not a directional/market-beta artifact. Risk-only levers judged on the Sharpe-maxDD frontier (must also not
  be reproducible by random throttling).

ALREADY TESTED & REJECTED — do NOT re-propose these (documented closed in the ledger):
  weekly-retrain / recency-half-life / training-window / fit-cutoff-ensemble (freshness all exhausted);
  conviction-weighted/|pred|-tilt sizing; asymmetric K (K_short!=K_long); vol-targeting de-gross; beta-neutralizing
  legs (SIDE_BETA_NEUT=1); short_btc_hedge / long_basket_hedge / long-leg restructures; in-model regime features;
  auto-adaptive bucket sizer; fixed mid-bear band (BEAR_GROSS_MULT/BEAR_MID); pred_disp/disp gate; GBM or pooled
  model; sector/cluster features; falling-knife / idio-vol long skip; entry-hour gate. K=3 and inv_vol are optimal.
`

const ENUM_SCHEMA = {
  type: 'object', additionalProperties: false,
  properties: { candidates: { type: 'array', items: {
    type: 'object', additionalProperties: false,
    properties: {
      name: { type: 'string' },
      rationale: { type: 'string', description: 'data-driven reason + what evidence you checked' },
      env_spec: { type: 'string', description: 'EXACT env overrides for opt_eval (e.g. "STRAT_K=4"); "" if needs code' },
      needs_code: { type: 'boolean' },
      param_free: { type: 'boolean', description: 'true if no sample-fit threshold; false = overfit risk' },
      expected: { type: 'string' },
    }, required: ['name','rationale','env_spec','needs_code','param_free','expected'],
  } } },
  required: ['candidates'],
}

const VERDICT_SCHEMA = {
  type: 'object', additionalProperties: false,
  properties: {
    name: { type: 'string' }, env_spec: { type: 'string' },
    sharpe: { type: 'number' }, maxdd: { type: 'number' }, lift: { type: 'number' },
    folds_positive: { type: 'integer' },
    passes_gates: { type: 'boolean' }, overfit_risk: { type: 'boolean' },
    verdict_note: { type: 'string', description: 'honest read incl per-fold concentration / directional check' },
  }, required: ['name','env_spec','sharpe','maxdd','lift','folds_positive','passes_gates','overfit_risk','verdict_note'],
}

const SYNTH_SCHEMA = {
  type: 'object', additionalProperties: false,
  properties: {
    winners: { type: 'array', items: { type: 'string' }, description: 'candidates passing honest gates AND param-free' },
    rejected: { type: 'array', items: { type: 'string' } },
    needs_code_followups: { type: 'array', items: { type: 'string' } },
    recommendation: { type: 'string' },
    honest_summary: { type: 'string', description: 'plain verdict; if nothing robust beats baseline, say so' },
  }, required: ['winners','rejected','needs_code_followups','recommendation','honest_summary'],
}

const ANGLES = [
  { k: 'pnl-regime', p: `ANGLE: PnL & regime decomposition. Read the ledger + iter5_tilt0/cycles.csv (it has long_ret_bps, short_ret_bps, regime, btc_ret_30d, pred_disp, turnover). Find where PnL leaks and propose 2-3 PARAMETER-FREE structural levers (discrete construction changes, not tuned thresholds) that could lift risk-adjusted return. Map each to env_spec where possible.` },
  { k: 'cost-hold', p: `ANGLE: cost / turnover / hold-structure. The book holds 24h via 6 sleeves; cost is ~1711 bps of the +16731. Propose 2-3 PARAMETER-FREE levers around hold horizon / sleeve count / rebalancing (discrete integer choices, e.g. via SLEEVE_DECAY_TAU or K) that reduce cost or smooth turnover without a fitted threshold.` },
  { k: 'sizing-risk', p: `ANGLE: sizing & risk (PARAMETER-FREE ONLY — vol-targeting/conviction/beta-neut are REJECTED). Propose 2-3 risk-budgeting levers that need no sample-fit number (e.g. equal-risk contribution, a discrete sizing-feature swap via SIZING_FEAT, volcap as a structural cap). A risk improvement at flat Sharpe counts if maxDD drops and it beats random.` },
  { k: 'model-target', p: `ANGLE: model / target / IC (PARAMETER-FREE — GBM/pooled/regime-features/ensembles are REJECTED). Quick-check per-cycle IC structure. Propose 2-3 param-free model/target ideas (e.g. seed/feature-subset averaging that needs no tuning, target winsorization at a principled level, rank vs z target) — each must be nested-OOS-honest, not sample-fit.` },
  { k: 'robustness', p: `ANGLE: robustness / universe / concentration (top-10 names = 77% of net PnL). Propose 2-3 levers that make the strategy MORE ROBUST (less overfit / less concentration-fragile) — discrete, parameter-free (e.g. universe breadth via MIN_HISTORY_DAYS or LIQ_FLOOR as principled floors, dedup). Flat-Sharpe-but-more-robust is a valid win here. Avoid sample-fit cutoffs.` },
]

function testPrompt(c) {
  return `TEST this candidate through the real bot replay and judge it honestly.
Candidate: ${c.name}
Rationale: ${c.rationale}
env_spec: ${c.env_spec}
Run EXACTLY: python3 live/opt_eval.py ${c.name.replace(/[^a-zA-Z0-9]/g,'_').slice(0,24)} "${c.env_spec}"
It takes ~5 min — run it, wait for completion, parse the final JSON line. Then judge against the HONEST GATES:
passes only if lift>=+0.30 AND folds_positive>=6/9 AND per_fold_delta is NOT concentrated in 1-2 folds (look at the
array) AND it's not a directional/beta artifact. Set overfit_risk=true if the candidate relied on any sample-fit
number. Return the metrics from the JSON verbatim + your honest verdict_note (mention per-fold concentration explicitly).
If opt_eval returns an error, set passes_gates=false and note the error.`
}

function synthPrompt(verdicts, allCands) {
  return `You are the adversarial judge. Here are the tested candidates' verdicts (JSON):
${JSON.stringify(verdicts, null, 1)}

And the full candidate list proposed (incl needs_code ones): ${JSON.stringify(allCands.map(c => ({name:c.name, needs_code:c.needs_code, param_free:c.param_free})), null, 1)}

Produce the honest synthesis. RULES: (1) a winner must pass the gates AND be param-free (no sample-fit threshold) —
penalize/reject anything with overfit_risk=true even if its number looks good; (2) reject fold-concentrated lifts;
(3) if NOTHING robustly beats the +4.22 baseline, say so plainly — that is the honest answer the user wants, not a
manufactured win. List needs_code candidates worth a careful separate implementation. Give a crisp recommendation.`
}

phase('Enumerate')
const enr = (await parallel(ANGLES.map(a => () =>
  agent(SHARED + '\n' + a.p, { schema: ENUM_SCHEMA, label: 'enum:' + a.k, phase: 'Enumerate' })
))).filter(Boolean)

const all = []
for (const e of enr) for (const c of (e.candidates || [])) all.push(c)
const seen = {}; const testable = []
for (const c of all) {
  const key = (c.name || '').toLowerCase().replace(/\s+/g, ' ').trim()
  if (!key || seen[key]) continue
  seen[key] = 1
  if (c.env_spec && c.env_spec.trim() && !c.needs_code) testable.push(c)
}
const picked = testable.slice(0, 7)
log(`enumerated ${all.length} candidates (${testable.length} env-testable); testing ${picked.length}`)

phase('Test')
const verdicts = (await pipeline(picked, c =>
  agent(SHARED + '\n' + testPrompt(c), { schema: VERDICT_SCHEMA, label: 'test:' + (c.name||'').slice(0,18), phase: 'Test' })
)).filter(Boolean)

phase('Synthesize')
const synth = await agent(SHARED + '\n' + synthPrompt(verdicts, all), { schema: SYNTH_SCHEMA, label: 'synthesize' })
return { tested: verdicts, synthesis: synth }
