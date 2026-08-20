# HANDOFF — P-2026-003 Polymarket Crypto 5-min

Updated: 2026-08-19 ~14:35 UTC (session 1 — program created, collector live).

## Done

- **Program record** (this dir): PROGRAM.md (idea = price the 300s binary via
  P(up)=Φ((F−K+μ̂τ)/(σ̂√τ)) from the P-2026-002 Binance fair-value stack; three
  edge hypotheses: stale-quote taker / calibration / binary MM; kill gates
  G1-G2 sketched, MUST be properly pre-registered before the first experiment).
- **API probes (all verified live 2026-08-19):** slugs
  `{btc,eth,sol,xrp}-updown-5m-<unix>` on 300s boundaries, pre-created ~5 min
  ahead; Gamma needs non-python UA (Cloudflare 403); CLOB WS market channel
  gives book / price_change / last_trade_price (incl. `fee_rate_bps` — fee
  schedule to be MEASURED from data); stale never-closed markets from Dec-2025
  exist in Gamma — always filter by window time, never by active/closed alone.
- **Collector LIVE** since ~14:32 UTC: `live/pm_research/collect_pm.py` →
  `data/pm_5min/` (markets.jsonl, resolutions.jsonl, raw/<day>/<slug>.jsonl.gz,
  one WS conn per market, resume-safe). Smoke: 12 markets, 50.6k msgs / 75s.
  Restart: `nohup python3 live/pm_research/collect_pm.py >
  data/pm_5min/collector.log 2>&1 &`
- Signal side already covered: P-2026-002's collect_hf.py streams Binance
  bookTicker/depth20/trade for the same 4 coins since 12:45 UTC.

## In progress / verify next session

- Resolution-poller verification (first closed windows resolve ~120s after
  window end; check resolutions.jsonl non-empty and outcomePrices populated).
- Message-volume check after 24h (~50k/75s smoke rate would be ~5-6 GB/day
  raw if sustained — likely lower off-peak; confirm disk burn acceptable,
  else trim book snapshots).

## STRUCTURE (2026-08-20) — `PM_ARCHITECTURE.md` is now the entry point

Modular architecture replacing the ORGANISATION of PM_MM_PLAN (17 appended
sections; the plan stays as the findings archive). 5 planes, ~20 modules, each
with an interface and honest status. Four structural rules encode the failures
we actually suffered: R-SSOT (one owner per quantity — 3 live latency values),
R-EXPLICIT (no hidden inputs — the siting/fair-value contradiction),
R-ONCE (no double-counted risk — 3 instances), R-PROVENANCE (assumed params
may not gate decisions).
KEY PROPERTY: **Option A vs Option B is a configuration choice**, not an
architectural one — B is this system with simpler impls of B3/B2/C1. So the
structure can be built before the A/B decision is made.
Build order: K1+K2 (param store, latency budget) → D2+D3 (normalizer,
point-in-time state) → E1/E2 measurements → E-X1/E-X2 (decide strategy
identity) → C4+K3 (risk, kill switch) → module details.

## ⛔ DECISION POINT (2026-08-20) — READ PM_MM_PLAN §17 FIRST

Deep adversarial review (`PM_DEEP_REVIEW.md`, all findings MEASURED on data
already on disk) returned **3 FATAL + 10 MAJOR; not sound enough to execute as
written**. Nothing further should be built until the user chooses:

- **Option A** — fix and continue the mechanism program (6-item minimum §17.6).
- **Option B (reviewer + data recommend)** — re-scope to *"harvest the
  favourite–longshot bias as a passive two-sided maker at moderate moneyness,
  with a loss-given-adverse cap, and measure the markout"*. Needs almost none
  of the current apparatus.

FATAL summary: (1) latency budget 9× too small — measured 1,700 ms of which
1,440 ms is PM-side publish delay; ATM never quotable anywhere; **SITING
DECISION SUSPENDED**; §12.1's premise also contradicts §2. (2) the
participation frontier is a safety constraint promoted to a policy with no
revenue term; addresses only 5–10% of window-time; `Q_max` one exponent wrong.
(3) no STOP condition exists — the §9 scope change deleted the only kill gates,
and E-M5 reads AFTER the subsidy it tests expires.

POSITIVE, and previously uncomputed: maker side is **+95 bps of notional gross
(+136 with rebate)**; book over-dispersed at every decision time; FLB
**+3.6 c/share at p ∈ [0.15,0.35)**, stable. That is the re-scope's prior.

STANDING RULE ADOPTED: no design decision that a measurement on existing data
could settle may be recorded as settled until that measurement is run.

## CURRENT STATE (2026-08-20) — read this first

Program is **mechanism-first, no PnL/capacity estimation** (user, 2026-08-20).
Live docs, in reading order:
1. `live/pm_research/PM_MM_PLAN.md` — §9 mechanism inventory M-1..M-8 is the
   organizing frame; §10 records that §3 is superseded and §5 replaced.
2. `PM_THEORY_CHECK_ORCHESTRATOR.md` — **the model is now re-based on
   arXiv 2607.17991 (Optimal MM in Prediction Markets)**, our exact setting.
3. `PM_MECHANISM_EXPERIMENTS.md` — the live ladder (E0 → E-M6 → M1/M2 → M3 →
   M7 → M8 → M4 → M5). **E-M6 settlement truth is the foundation gate**;
   earliest read 08-25. Nothing downstream is valid until it passes.
4. `PM_MECHANISM_THEORY.md` — theory map per mechanism, ADOPTED; its three
   model changes are recorded in PM_MM_PLAN §11.

Three model changes now binding (§11): (a) rewards = constraint with a shadow
price, occupy iff R/X ≥ c(|d|,r); Tullock contest ⇒ the band only pays a maker
with differentially lower pickoff cost; (b) **pull-on-burst is not a defence**
at our latency — use an ex-ante participation region `m/φ(d) ≥ k√(3L/r)` with
size as the risk knob, and note pickoff exposure is VOLATILITY-FREE; the moat
far from the money is the TICK not the fee, which makes the 0.01→0.001 tick
change a first-order economic parameter; (c) λ_fill needs a queue coordinate
(Q_ahead, with the bracket on Q_ahead not λ), plus q_up/q_down split for
pair-harvest and Q_max = κ/(γ·p̂(1−p̂)).

NOTE for iteration 3: my `PM_THEORY_CHECK_ORCHESTRATOR.md` §2 claim that
"MM-under-obligation theory doesn't exist" was a SEARCH FAILURE — the correct
body is principal–agent MM contracts (El Euch et al.; Baldacci et al.). That
section is marked superseded; don't cite it.

Open model questions now explicit: w may be 300 s not 60 s (testable — we
record twap_thirty too); K likely NOT known at t=0 (stream lag p50 ≈1.7 s);
tick regime changes away from ATM (0.01 → 0.001, 328 events/130 windows).

Collector fix applied 2026-08-20: WS `1013 slow consumer` drops on BTC were
load-correlated loss on ~85% of notional — now max_queue=2^16 + batched writes.

## Review loop state (PM-E0.5 — runs before ANY experiment)

Iter 1 DONE (PM_REVIEW_ITER1.md; log in PM_REVIEW_LOOP.md): settlement stream
FOUND on the public live-data WS (`crypto_prices_twap_sixty` — Chainlink RTDS
relay; `crypto_prices` itself is just a Binance-spot mirror); collector
restart-loss bug fixed (in-flight resume + numbered gz shards); TWAP topics
subscribed ~15:45 UTC 2026-08-19 — **no replay exists, K/X_T truth starts
accumulating from then**; doge/bnb/hype coins added; CLOB fee fields captured.
KNOWN DUP for iter-2's gap scan: a duplicate collector pair ran ~30 s at
2026-08-20 00:46 UTC (bad pgrep pattern → false "collector down" → duplicate
launch; killed immediately). Effect: 225 duplicated (timestamp, symbol)
records in crypto_prices_twap_sixty spanning epoch-ms 1787186761000–
1787186791000, plus possible duplicate lines in the then-in-flight market
raw files. **All consumers must dedup by (timestamp, symbol) for prices and
by (recv-independent) message identity for raw — recv_ns differs per process
so exact-line dedup does NOT catch it.** Standing rule, not a one-off.
CHECK COLLECTORS WITH: `ps -eo pid,etimes,cmd | grep live/pm_research`
(pgrep patterns must include the pm_research/ path segment).

KNOWN GAP for iter-2's gap scan: market-side collection down ~15:36–15:52 UTC
2026-08-19 (verification harness killed between its pkill and relaunch —
~3 windows/coin lost on the market side; TWAP + price feeds unaffected).
**Iter 2 due after ≥12 h post-fix data**: verify fixes on real data (restart
test, shard concat, TWAP continuity/gaps), resolve the fee-source conflict
(docs 7% vs CLOB base_fee vs fee_rate_bps=0 observed on ALL trades so far),
pin the settlement boundary convention (X_T≥X_0 vs full-window TWAP reading —
decide empirically: recorded TWAP stream vs CLOB winners on our own windows).
Convergence: two consecutive clean iterations, then PM-E1.

## Next steps (MM-FIRST — user decision 2026-08-19; plan = PM_MM_PLAN.md)

1. **pm_backfill.py**: deterministic historical slugs → Gamma metadata +
   resolutions + Data-API trades (+ CLOB prices-history). Months of windows
   available without waiting — the biggest lever; PM-E1/E2 run on it.
2. **pm_e1_anatomy.py** (PM-E1): fee empirics from fee_rate_bps; settlement
   verification (oracle, TWAP, K convention, ties); calibration curve;
   flow timing within window; spread/tick structure. At its end: FREEZE
   PM-E2/E3 gate numbers (prereg) before any edge computation.
3. PM-E2 model contest (p̂=Φ(d) TWAP-aware vs book mid; log-loss/Brier OOS,
   gate G2), then PM-E3 MM replay on ≥2 wks forward books (gates G3a
   no-rewards / G3b with-rewards, queue bracket, notional-weighted).
4. Rewards/fee program docs snapshot → data/pm_5min/docs/ (rules are
   versioned; H-PM4).

## Watch out for

- **resolutions.jsonl contains 8 GARBAGE rows from the first hour** (bug found
  + fixed 2026-08-19: Gamma populates outcomePrices continuously for OPEN
  markets — presence ≠ resolution). Consumers MUST filter rows through
  `is_final` and dedupe by slug keeping the final row.
- **Gamma slug queries return EMPTY shortly after a 5-min market resolves**
  (probe-verified) — Gamma can NEVER confirm outcomes for these markets. The
  durable resolution source is `clob.polymarket.com/markets/<conditionId>`
  (closed=True + per-token winner flags); resolver switched to it ~14:55 UTC.
  New resolution rows: {slug, conditionId, closed, winners:{Up,Down}, source}.
  IMPLICATION FOR BACKFILL (pm_backfill.py design): historical slugs canNOT be
  resolved via Gamma; enumerate via paginated CLOB GET /markets (includes
  closed; next_cursor) or gamma ?closed=true list queries — PROBE FIRST. The
  "Polymarket-v1 Database" paper (arxiv 2606.04217) may provide seed history.
- Thin windows exist (a BTC window with volumeNum=2 shares traded) — anatomy
  must stratify by volume; never average across dead and live windows.

- Trading access (US status, CLOB auth) is a DEPLOYMENT question — out of
  scope until G1/G2 pass; don't let it leak into research design.
- fee_rate_bps must be read from trades, not assumed zero.
- Resolution near ties: oracle/timestamp convention risk — quantify from
  recorded resolutions before trusting the Φ model at |P−0.5| small.
- The 4 coins are the liquid set; do NOT generalize conclusions beyond them.
