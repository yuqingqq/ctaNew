# PM_MECHANISM_EXPERIMENTS — mechanism-truth ladder (replaces PM_MM_PLAN §5)

> **⚠ For current state read [`FLOW_MODEL_STATE.md`](FLOW_MODEL_STATE.md).** This
> document is **provenance** — correct about its own moment, not a statement of
> current belief. Where it conflicts with `FLOW_MODEL_STATE.md`, that page wins.


Program P-2026-003. Written for sketch-review iteration 2 under the user's
2026-08-20 scope change (`PM_MM_PLAN.md` §9): **no PnL, no capacity, no sizing.**
Every experiment asks one question — *does this venue behave the way the model
assumes?* — and every experiment must change the model on at least one of its
outcomes, or it is cut.

Organizing frame: the M-1..M-8 mechanism inventory (§9). One experiment per
mechanism, plus a cross-cutting data-integrity precondition E0.

Statistical discipline is inherited **verbatim** from `PM_SKETCH_REVIEW_ITER1_S.md`
and is not re-litigated here:

- gate statistics are **paired** wherever two things are compared (never
  difference-in-significance);
- **window = inferential unit**; SEs **day-clustered**; stationary **block
  bootstrap over UTC days, B = 2000**;
- **strata frozen before data is read** and **model-free** (time-in-window;
  `|book_mid − 0.5|` — never model `|d|`, which lets the model grade its own exam);
- **Holm** across each experiment's own family of tests; families enumerated below;
- **notional-weighted** wherever a weighting exists, with **ex-ante** weights
  (trailing same-coin median per-window notional, frozen), never the scored
  window's own volume;
- universe **frozen** at 7 coins {btc, eth, sol, xrp, doge, bnb, hype} — 6 for
  any experiment needing a Binance leg (**HYPEUSDT is absent from `data/mm_hf/`**);
  no coin enters mid-sample (the expanding-survivor artifact killed the OB-timing
  signal in this repo);
- **fails-final, passes-provisional**; freeze artifact is `PM_PREREG.md`, which
  must carry the constants below and be cited by every harness (no inline numbers).

**Design-freeze date: 2026-08-21 UTC.** Gate-bearing data is admitted from
**2026-08-20 01:03 UTC** (first uninterrupted market-side capture after the
restart gap found during this design pass) except where a longer history is
explicitly allowed.

### Measurements taken during design (prereg-safe: coverage/variance/schema only, no outcome-dependent quantity read)

| fact | value | source |
|---|---|---|
| raw line format | `recv_ns \t <json>`; payload `timestamp` in ms | `data/pm_5min/raw/*/*.jsonl*.gz` |
| event types | `book`, `price_change`, `last_trade_price`, `tick_size_change` | ditto |
| `last_trade_price` carries `transaction_hash` | yes → on-chain join available | ditto |
| **`tick_size_change` 0.01 → 0.001** | **328 events / 130 windows (2026-08-20)**, never seen reverting in-sample | ditto |
| window capture span | p50 −287 s to +364 s rel. window open | ditto |
| full [0, 300 s] coverage | **103/130 windows = 79%** on 2026-08-20 | ditto |
| BTC events/window (p50) | 155.7k (149.7k price_change, 4.1k book, 1.9k ltp) | ditto |
| price_change recv−payload (quiet BTC window) | p50 57 ms, p90 189 ms, p99 696 ms | ditto |
| **`1013 slow consumer` WS disconnects on BTC** | present, recurring (01:15Z, 01:18Z 2026-08-20) | `data/pm_5min/collector.log` |
| **new gap, not in HANDOFF** | market-side collection down **~00:46–01:03 UTC 2026-08-20** (restart) | file mtimes + collector pid age |
| `last_trade_price` tx → asset_ids | **668/668 tx touched exactly ONE token** (one BTC window) | raw |
| price_change payload-ms mod 10 | uniform (7,132–7,813 per residue over 76k events) | raw |
| price_change msgs sharing one ms | mean 1.98, p99 8 | raw |
| TWAP60 stream | 1 s cadence, 8 symbols, **71 gaps > 2 s per ~10 h, max gap 44 s** | `prices/crypto_prices_twap_sixty/` |
| **TWAP60 recv − payload** | **p50 ≈ 1.70 s** (all coins) | ditto |
| TWAP30 stream also collected | yes (`crypto_prices_twap_thirty`) — window length is testable | ditto |
| resolutions | 871 final (dedupe by slug, `winners` present) of 892 markets | `resolutions.jsonl` |
| resolved windows with full TWAP over [t0−60, T] | **114** (stream started 2026-08-19 15:45) | joined |
| rewards | `rewards_authoritative: null`, `rewards_registry_n: 16196`, Gamma band 4.5 c / min 50 | `markets.jsonl` |
| market params | `orderPriceMinTickSize` 0.01, `orderMinSize` 5, `neg_risk` false | ditto |

---

## Summary ladder

| # | ID | mechanism | what it settles | depends on | earliest read |
|---|---|---|---|---|---|
| 0 | **E0** | — (cross-cutting) | admissible-window universe; gaps/dedupe/shards; whether exclusions are load-correlated | — | **2026-08-22** |
| 1 | **E-M6** | M-6 settlement | **FOUNDATION GATE** — does our stream + rule reproduce the actual winners? boundary/tie/strike convention; is K knowable at t=0? | E0 | **2026-08-25** |
| 2 | **E-M1** | M-1 matching/queue | is the feed replayable? trade-vs-cancel attribution; price priority; batching signature; operator latency | E0 | 2026-08-23 |
| 3 | **E-M2** | M-2 lifecycle | tick regime (0.01 vs 0.001), spread in ticks, quote half-life → is §3's 4-action set well-posed? | E0 | 2026-08-23 |
| 4 | **E-M3** | M-3 CTF pair | is the book unified (2 sides, not 4)? are mint-crossings real? intra-window merge usage/cost | E0 | 2026-08-22 / 08-24 |
| 5 | **E-M7** | M-7 latency/sniping | PM book's reaction lag to Binance; pickoff intensity + cost in c/share; **can a slow maker survive?** | E0, E-M1(a) | 2026-08-30 |
| 6 | **E-M8** | M-8 expiry | does inventory really self-liquidate into a Bernoulli payoff? resolution completeness, flips, terminal pinning | E0, E-M6 | 2026-08-27 |
| 7 | **E-M4** | M-4 fees | fee formula confirmatory; **is the maker rebate actually paid?** | E-M3(b) | 2026-08-24 / 09-03 |
| 8 | **E-M5** | M-5 rewards | is there ANY behavioural evidence these markets pay a band? | E0, E-M2 (tick) | **2026-09-03** |

**Dependency logic.** E0 gates everything (it defines the admissible set).
E-M6 is the **foundation gate**: it fixes the *target* of p̂. Any experiment that
strata­fies or conditions on the model is void until E-M6 resolves — hence the
standing rule that until E-M6 passes, **all conditioning is on model-free
`|book_mid − 0.5|`**, never on `|d|`. E-M1/E-M2/E-M3/E-M5/E-M7 are
book-and-trade experiments and are *not* blocked by E-M6; they may run in
parallel. E-M1(a) (feed replayability) is a second-order structural gate: if it
fails, every replay-shaped arm (E-M7(b,c), any future fill work) is deleted
rather than degraded.

---

## E0 — data integrity (precondition)

**Proposition.** The recorded capture is a faithful, gap-characterised,
deduplicated view of the venue, and the windows we admit are not a
market-condition-selected subsample.

**Observables / estimators.**

1. *Shard concatenation.* For slug `S`, read `S.jsonl.gz`, `S.jsonl.1.gz`,
   `S.jsonl.2.gz`, … in numeric order, then the bare in-flight `S.jsonl`; sort by
   `recv_ns`. 22/130 slugs on 2026-08-20 have >1 shard, so this path is
   load-bearing. Verify: (i) `recv_ns` monotone within a shard; (ii) shard
   boundaries do not overlap in payload time; (iii) after concat, duplicate rate
   under the identity key below.
2. *Dedupe keys* (standing rule — `recv_ns` differs per process, so line-level
   dedupe does **not** catch duplicate collectors):
   - raw: `(event_type, asset_id, payload.timestamp, hash)` for `book` /
     `price_change`; `(transaction_hash, asset_id, payload.timestamp)` for
     `last_trade_price`.
   - prices: `(topic, payload.symbol, payload.timestamp)`.
   - resolutions: dedupe by `slug`, keep last, **filter `closed==True` and
     `winners` present** (the first-hour rows are garbage: `closed:false` with
     populated `outcomePrices`).
3. *Admissibility.* Window `w` is ADMISSIBLE iff, after concat+dedupe:
   contiguous market-side capture over `[t0 − 60 s, T + 90 s]`; max intra-window
   inter-arrival gap ≤ 5 s; no `1013 slow consumer` disconnect logged for that
   `(slug, interval)`; and the coin's TWAP60 stream has no gap > 5 s intersecting
   `[t0 − 60 s, T]` (else the window is separately flagged SETTLEMENT-BLIND and
   excluded from E-M6/E-M8 only).
4. *Frozen blacklist* (goes into `PM_PREREG.md` verbatim):
   market-side outage 2026-08-19 15:36–15:52 UTC; **collector restart gap
   2026-08-20 00:46–01:03 UTC** (found in this design pass; add to HANDOFF);
   duplicate-collector price rows at epoch-ms 1787186761000–1787186791000;
   **all data before 2026-08-07 is a different settlement rule and is never
   pooled with anything** (it may be used for liquidity anatomy only, labelled).
5. *Load-correlation of exclusions.* Per coin-hour, count admissibility failures
   `x_h` and compute Binance 1-min realised vol `RV_h` from
   `data/mm_hf/raw/bookTicker/<SYM>/`. Estimate Spearman ρ(x_h, RV_h),
   day-clustered, block-bootstrap CI.

**Decision rule.**
- **PASS**: ≥ 90% of windows admissible per coin over the freeze period; zero
  dedupe-key collisions surviving concat; ρ 95% CI contains 0 **or** |ρ| < 0.2.
- **CONDITIONAL**: 70–90% admissible → proceed on the frozen blacklist, and every
  downstream result carries the excluded fraction and the *direction* of the bias
  in its results table. (Current measurement: **79%** → CONDITIONAL is the
  expected state.)
- **FAIL**: < 70% admissible on BTC (85% of notional), or ρ ≥ 0.4 → the collector
  is redesigned (per-coin processes / larger recv buffer / drop `book` snapshots
  and reconstruct from `price_change`) and re-collection starts before any
  mechanism experiment is read. **The `1013 slow consumer` disconnects are a live
  defect on exactly the coin that carries the notional and must be fixed or
  bounded before E-M7 is believed.**
- **UNDERPOWERED**: < 5 UTC days.

**Prereqs / read date.** Nothing beyond the running collectors. **2026-08-22**
(≥ 2 clean UTC days after the 08-20 01:03 restart).

**Model consequence.** Defines the admissible universe used by every other
experiment and freezes it. If exclusions are load-correlated, *every* mechanism
number in this program is a calm-market number and must be stated as such —
outages cluster with bursts, i.e. with exactly the toxic states the MM model
exists to handle. A FAIL prevents spending four weeks accumulating unusable data.

---

## E-M6 — settlement truth *(FOUNDATION GATE)*

**Mechanism.** M-6: Chainlink 60 s TWAP; Up iff `X_T ≥ X_0`; tie → Up; `r³`
pinning inside the averaging window.

**Proposition (what the model assumes).** `X_t` = the Chainlink 60 s TWAP stream
reading; `X_0` = the reading at window open `t0`; `X_T` = the reading at window
end `T`; ties resolve Up; and `X_0` is **known at `t = 0`** so `p̂` has a fixed
strike from the first tick.

**Why this is the foundation.** §2's entire variance law
(`Var_t[X_T] = σ²(τ − 40 s)`, then `σ²r³/(3w²)`), §3's `(|d|, r)` pull surface,
and the definition of the unquotable near-tie zone are all consequences of this
one convention. If the target is wrong, every `p̂`-conditioned result is void.

**The ambiguity is real and documented in our own data.** `markets.jsonl`
`description` reads: *"…the time-weighted average price (TWAP) … **of the time
range specified in the title** is greater than or equal to **the price** at the
beginning of that range"* — which admits (i) a TWAP over the whole 300 s range,
not a 60 s reading at `T`, and (ii) a strike that is *"the price"*, not
necessarily a TWAP. Meanwhile `resolutionSource` points at the **60 s** stream,
and we independently record `crypto_prices_twap_thirty`, so the averaging window
itself is testable.

**Estimator.** Frozen convention grid, evaluated by exact winner reproduction.

- `X_T ∈ {S60(T), S30(T), mean_{[t0,T]} S60, mean_{[t0,T]} spot-mirror}` (4)
- `X_0 ∈ {S60(t0), S30(t0), spot-mirror(t0)}` (3)
- boundary rule: **last payload timestamp ≤ boundary** (primary; the other two —
  first ≥, nearest — enter only the sensitivity sub-grid on the winning cell)
- tie rule: `≥ → Up` primary; `> → Up` in the sensitivity sub-grid

⇒ **12 preregistered primary cells.** For convention `c` and window `w`:
`Ŷ_c(w) = 1{X_T ≥ X_0}`. Score `A_c` = notional-weighted agreement with the CLOB
winner (`resolutions.jsonl`, filtered/deduped per E0). Also compute the margin
`m_w = 1e4·(X_T − X_0)/X_0` in bps and report `A_c` conditional on `|m_w|`.
Sources: `data/pm_5min/prices/crypto_prices_twap_sixty|thirty/<hour>.csv[.gz]`
(`payload.symbol`, `payload.timestamp`, `payload.full_accuracy_value` — use the
1e18-scaled integer, not the float `value`) and
`data/pm_5min/prices/crypto_prices/` for the spot mirror.

**Decision rule (numbers).**
- **CONFIRM**: the best cell reproduces **≥ 99.0%** of admissible resolved
  windows pooled (notional-weighted), **≥ 99.5%** on `|m_w| > 0.5 bps`, and beats
  the runner-up on a **paired McNemar** exact-binomial test over discordant
  windows, Holm-corrected across the 12 cells at α = 0.05. Under confirmation,
  the residual disagreements define the **jitter band**
  `δ_jit = p99 of |m_w| among disagreements` → this is the numeric width of the
  **do-not-quote near-tie zone** (H-PM1/H-PM1c), a deliverable in bps and in
  probability-cents.
- **REFUTE (model rewrite mandatory)**: no cell exceeds 99.0%, **or** the winner
  is anything other than `{S60(T), S60(t0), ≥}`. Consequences are not cosmetic:
  a full-range TWAP has `Var_t[X_T] ∝ σ²·T/3` with **no `r³` pinning** — the
  endgame is a completely different object and §3's pull surface is re-derived
  from scratch; a 30 s window changes `w` from 60 → 30 in every §2 formula.
- **UNDERPOWERED**: < 400 admissible windows with full TWAP coverage over
  `[t0 − 5 s, T + 5 s]`, **or** < 20 discordant windows between the top two cells
  (McNemar has nothing to test). Do not adopt; extend collection.

**E-M6b — is the strike knowable at `t = 0`?** *(sub-experiment, own decision)*
Estimator: per window, `Δ_K = recv_ns/1e6 − t0` of the stream tick that defines
`X_0` under the winning convention. **Decision: if p95(Δ_K) > 3 s, the model's
"K KNOWN at window open" claim is FALSE** and §2/§3 must carry a strike-uncertainty
term over `[t0, t0 + Δ_K]` or forbid quoting there. *Design-pass indication:
TWAP60 `recv − payload` p50 ≈ 1.70 s on a 1 s cadence ⇒ Δ_K ≈ 1.7–2.7 s, so this
is very likely to fire.* That is already a model change, independent of the
convention outcome.

**E-M6c — manipulation-successor screen** *(H-PM1b, folded in here because it is
a settlement-mechanism question).* Per window, `z_max` = max standardised
divergence between the TWAP60 stream and a Binance-synthetic 60 s TWAP inside
`[T − 60 s, T]`, standardised on the trailing 24 h of the same coin. Report the
flagged fraction at `z_max > 4` and the winner-flip rate of flagged windows
(would the outcome differ under the un-diverged synthetic?). Decision: flagged
fraction > 1% with a flip rate > 20% ⇒ oracle-pushing is live, and flagged
windows become a hard do-not-quote screen in §3 and a toxicity feature.
Otherwise it is recorded as a standing monitor with a measured null.

**Prereqs / read date.** E0. TWAP stream exists only from **2026-08-19 15:45 UTC**
— of 871 currently-final resolutions, only **114** have full `[t0 − 60, T]` TWAP
coverage, so essentially all usable data is forward. Yield at ~288 windows/day ×
7 coins × 0.79 admissible ≈ **1,590 windows/day**, so the 400-window bar clears in
under a day; the binding constraint is **≥ 5 distinct UTC days** (so the boundary
rule is exercised across day boundaries and any operator clock drift) plus the
20-discordant-window bar. **Earliest read 2026-08-25**; declare provisional if the
discordant count is short, and extend to 2026-09-01.
Weighting/clustering: agreement is near-deterministic, so report per-day `A_c`,
a day-clustered CI on `(1 − A_c)`, and block-bootstrap the `δ_jit` estimate.

**Model consequence.** Fixes the target of `p̂` — `X_T`, `X_0`, `w`, the tie rule
— and therefore §2's variance law, §3's pull surface, the near-tie unquotable
band `δ_jit`, and whether HYPE (no Binance leg) is quotable at all. Every
`p̂`-conditioned stratification in the program is provisional until this reads.

---

## E-M1 — matching and queue reconstruction

**Mechanism.** M-1: off-chain operator matching, price-time priority, on-chain
settlement; batching and matching latency unverified.

**Propositions.**
- **P1** The public feed (`book` snapshots + `price_change` level totals +
  `last_trade_price`) is sufficient to reconstruct a level's displayed total and
  to attribute decrements to trades vs cancels — i.e. a price-time queue position
  can be *bracketed*.
- **P2** Matching respects **price** priority and is **continuous** (not a
  discrete batch auction).

**Estimators.**

**(a) Feed replayability.** Maintain a ladder from `book` snapshots; apply each
`price_change` as `L[price] := size` (level totals — verified). At every
subsequent `book` snapshot compare: `R_exact` = fraction of snapshots matching on
all levels to 1e−6; `R_touch` = fraction matching on best bid/ask. Free internal
control: every `price_change` also carries `best_bid`/`best_ask` — cross-check
against the reconstruction.
*Decision:* `R_touch ≥ 0.99` **and** `R_exact ≥ 0.95` (per coin, window-weighted
by ex-ante notional) ⇒ replay sound. `R_touch ∈ [0.90, 0.99)` ⇒ sound **at the
touch only**; all queue work is restricted to the touch. `R_touch < 0.90` ⇒ the
feed is not replayable and **every replay-shaped arm in this ladder is deleted**,
not degraded.

**(b) Trade-vs-cancel attribution.** For each level decrement `Δ < 0` at price
`P`, payload ms `t`, search `last_trade_price` prints at `P` with payload ms in
`[t − 250, t + 250]` (Δt = 250 ms frozen). Define
`α_trade = Σ min(|Δ|, matched print size) / Σ |Δ|`.
Report also the cancel-to-trade ratio and the **touch half-life**: median time
until the touch level total changes by > 50%.
*Decision:* `α_trade ≥ 0.60` ⇒ the pessimistic queue rule (trades-only decrement,
join-back) is implementable with a measured cancel share `(1 − α_trade)`.
`α_trade < 0.30` ⇒ level totals are churn-dominated, queue-ahead is unknowable,
the bracket widens to [front-of-queue, never-fill] and any fill-rate statement is
vacuous ⇒ **fill modelling is dropped from §3** and only quote-level markout
(E-M7) survives. Separately: **touch half-life < 250 ms ⇒ "queue position" is not
a stable state variable at our latency**, and §3's default action moves from
"join best" to "rest 1–2 back" regardless of `α_trade`.

**(c) Price-priority violations.** For each print at price `P` on token `X`,
inspect the reconstructed opposite side at that instant: a BUY print at `P` while
an ask exists at `P′ < P` with size > 0 is a violation. Report violation rate `v`
and the median staleness of the violating book state.
*Decision:* `v ≤ 0.005` with median staleness < 100 ms ⇒ **price priority
CONFIRMED**. `v > 0.05` ⇒ matching is not simple price-priority (operator
batching/internalisation) and §3's per-level EV must be replaced by an
empirically-estimated fill probability (regression of fill on level and displayed
depth), not a queue model.

**(d) Batching signature.** Distribution of `price_change` payload-ms modulo
{1, 5, 10, 25, 50, 100} ms; χ² uniformity per modulus, day-clustered, Holm across
the 6. Also the count of distinct `hash` values sharing one payload ms.
*Design-pass indication:* mod-10 uniform (7,132–7,813 per residue over 76k
events); 1.98 changes/ms mean, p99 = 8 ⇒ continuous matching with multi-leg
emission, no time grid.
*Decision:* all moduli χ² p > 0.01 after Holm ⇒ **continuous matching CONFIRMED**
⇒ finite-horizon continuous MM theory applies as §3 assumes. Any modulus with
> 5% excess mass ⇒ the venue runs discrete batches of that period, i.e. a
**Budish–Cramton–Shim frequent-batch auction**, which changes the model class
entirely: queue position stops mattering and the sniping race becomes a
within-batch tie. This is the single highest-leverage binary in E-M1.

**(e) Operator matching latency.** For prints with `transaction_hash`, join the
Polygon receipt: report `block_ts − payload_ts` and `payload_ts − triggering
Binance event ts` (p50/p90). Descriptive; enters the model as the
settlement-confirmation lag, not the quoting lag.

**Harness negative controls** (mandatory, from S4): a replay quoter that never
quotes must record exactly zero fills; a quoter bidding 0.99 must fill on every
downward print. Either failing ⇒ harness bug, results void.

**Prereqs / read date.** E0. Uses `book` + `price_change` + `last_trade_price`
only; (e) needs public Polygon RPC. No dependency on E-M6. 3 UTC days of
admissible BTC+ETH windows (~1,300 windows) suffices for (a)–(d).
**2026-08-23.**

**Model consequence.** (a) decides whether replay exists at all; (b) sets the
bracket ends and, via touch half-life, decides whether queue position is a state
variable in §3 or is struck from it; (c)/(d) **select the model class**
(continuous price-time MM vs batch auction vs black-box fill probability).

---

## E-M2 — order lifecycle: the quoting grid the model assumes

**Mechanism.** M-2: tick $0.01, min size 5, rewards-min 50, cancel/replace
latency, rate limits.

**Proposition.** The quoting grid is $0.01, books are 2–4 ticks wide, and §3's
discrete per-level EV over the action set {join best, improve 1 tick, rest 1–2
back, stay out} is therefore well-posed.

**Already contradicted in-sample.** `tick_size_change` fires **328 times across
130 windows** on 2026-08-20, all `0.01 → 0.001`, i.e. **the tick is
state-dependent** (consistent with the documented finer tick outside
`[0.10, 0.90]`). §2's "2–4 ticks wide" holds at best ATM. This is not a
descriptive detail: the entire "1 bp of BTC ≈ 3 ticks" arithmetic (M5) is 10×
off in the 0.001 regime.

**Estimators** (per admissible window; pooled notional-weighted; strata =
time-in-window × `|book_mid − 0.5|`, both model-free).
1. **Tick-regime occupancy**: fraction of window-time at 0.01 vs 0.001; empirical
   CDF of book mid at the transition timestamp; reversion rate.
2. **Spread** in ticks and in cents on a 1 s grid, split by tick regime;
   p10/p50/p90.
3. **Touch depth**: displayed size at best and best ± 1 tick (shares and USDC);
   and cumulative depth within 1.0/1.5/3.0/4.5 c of mid (this feeds E-M5).
4. **Quote-life**: survival curve of the touch price; distinct best-bid values per
   minute = the requote cadence the field actually runs.
5. **Effective action-set size**: number of price levels between mid and mid ± 3 c
   carrying ≥ `orderMinSize` (5).

**Decision rule.**
- **CONFIRM §3's action set** if, in the 0.01 regime, median spread ∈ [2, 4] ticks
  **and** median level count within 3 c ≥ 3.
- **REFUTE / model change** if median spread ≥ 6 ticks in any frozen stratum
  (continuous-δ quoting returns for that stratum), **or** if **≥ 20% of
  window-time sits at tick 0.001** (the action set must be redefined in *cents*,
  the 0.001 regime becomes a separate quoting mode, and all tick-denominated
  thresholds in §3 are re-expressed).
- **Independent refutation:** touch quote-life **p50 < 250 ms** ⇒ a maker at
  150–250 ms requote latency cannot hold a touch quote at all ⇒ §3's default
  action becomes "rest 1–2 back", not "join best".
- **UNDERPOWERED**: < 200 admissible windows/coin, or < 20 tick transitions.

**Prereqs / read date.** E0. 3 days → **2026-08-23**.

**Model consequence.** Fixes the action set and the grid in §3; a material
0.001 regime forces a second quoting mode with different rewards arithmetic and
a 10×-finer adverse-selection scale.

---

## E-M3 — CTF pair mechanics: is the book unified?

**Mechanism.** M-3: split/merge/redeem, Up + Down = $1, unified book with
mint/merge at match time; intra-window merge cost/latency unverified.

**Propositions.**
- **P1** The two token books are one unified book: `ask(Up, q) ≡ bid(Down, 1 − q)`
  at all times ⇒ a maker quotes **two** economic sides, **not four** (§2.3 and §3
  both say four).
- **P2** Intra-window pair acquisition is real and free: two maker fills at
  combined cost < $1 are minted by the operator with no extra transaction and no
  extra fee.
- **P3** Intra-window `mergePositions` is a capital-velocity option and is (or is
  not) actually used.

**Estimators.**

**(a) Mirror-identity violation.** On every `book` pair with identical payload
`timestamp`, and on the `best_bid`/`best_ask` carried in every `price_change`,
compute `V = 1{∃ q : |size_ask(Up, q) − size_bid(Down, 1 − q)| > 1e−6}` with
**tick-aware price keys** (tick may be 0.001 — E-M2; a 2-decimal key silently
manufactures violations). Report violation rate, duration to re-identity, and
magnitude in shares.
*Decision:* `V ≤ 0.01` with median duration < 100 ms ⇒ **unified book CONFIRMED**
⇒ §2.3/§3 are rewritten from "quote FOUR sides" to "quote two sides of one
book", "pair-harvest" is re-labelled as ordinary two-sided market making with
spread `p_ask − p_bid`, and E-M1's de-duplication rule becomes trivially correct
(2 queue positions, not 4 — the double-fill bug MF-6 warns about cannot occur).
`V > 0.10`, or persistent violations > 1 s ⇒ the tokens are **separate** books
with an arbitrage relation; four sides are real; report the distribution of the
arbitrage gap `1 − (bid_Up + bid_Down)` as a newly-quotable object.

**(b) Pair-trade observability.** Group `last_trade_price` by `transaction_hash`;
count distinct `asset_id` per tx. *Design-pass indication: 668/668 tx touched
exactly ONE token in one BTC window* ⇒ the WS print does not expose a mint leg,
so the on-chain leg is required. For a frozen random sample of **≥ 500 tx**
(stratified by coin × time-in-window), decode the Polygon receipt: count
`OrderFilled` legs, CTF `PositionSplit` / `PositionsMerge` events, and the fee
word.
*Decision:* ≥ 20% of taker fills mint a complementary pair ⇒ the mint-crossing
mechanism is CONFIRMED live. < 5% ⇒ minting is documented-but-unused and
pair-harvest must be modelled as **two independent one-sided fills with leg
risk** — which the plan under-states as a merge-cost problem when it is a
sequencing problem.

**(c) Intra-window merge usage and cost.** Scan Polygon logs on each sampled
market's `conditionId` for `PositionsMerge` between `t0` and `T`: count, size,
gas.
*Decision:* count ≈ 0 across ≥ 500 windows ⇒ nobody merges intra-window; §2.3
states plainly that holding to resolution *is* the exit and drops merge from the
model. Non-zero ⇒ record median gas and fill→merge latency.

**(d) Redemption latency.** `resolutions.jsonl` `recv_ns` − `window_end` over
≥ 500 windows; p50/p95 (ITER1 measured ≈ 85 s). Feeds E-M8's capital-cycle
statement — **not** a PnL quantity.

**Prereqs / read date.** E0; (b)/(c) need public Polygon RPC (500 receipts is
minutes; the M lens already demonstrated the decode path and the `OrderFilled`
topic0). Independent of E-M6. **(a) 2026-08-22; (b)(c) 2026-08-24.**

**Model consequence.** Collapses (or confirms) four sides to two — a structural
simplification of §3's quoting problem; decides whether pair-harvest is a
distinct strategy or a renaming of two-sided MM; and moves the binding constraint
from merge cost to leg risk (or refutes that).

---

## E-M7 — latency topology and sniping exposure

**Mechanism.** M-7: PM matching in London, Binance in Tokyo, our box in
us-east-1; `1 bp of BTC ≈ 2.6–3.7 c of probability` ATM.

**Propositions.**
- **P1** The PM book reacts to Binance with a measurable lag `L`, and a maker at
  requote latency `L_us ≈ 150–250 ms` is exposed on the residual — so `δ_tox` must
  be sized off measured `L`, not an assumed reaction time.
- **P2 (the one that decides the track)** **A slow maker can survive**: resting
  quotes are more often *withdrawn* than *run over* after a Binance burst.

**Clock discipline (mandatory; a conclusion that holds on only one clock is
recorded INCONCLUSIVE).** Both feeds land on the same box with the same
`recv_ns`, so no clock-sync assumption is needed — but two distinct quantities
must be reported side by side:
- **transport-inclusive lag** — `recv_ns` on both sides = what our stack sees;
- **venue-intrinsic lag** — PM `payload.timestamp` vs Binance `E` = the venue's
  own reaction, free of our WS backlog.
The gap between them is our transport cost: measured p50 57 ms / p90 189 ms /
p99 696 ms on a quiet BTC window, and *seconds* under `1013 slow consumer`
conditions. **Windows E0-flagged for slow-consumer disconnects are excluded from
this experiment**, and their excluded fraction is reported (they are the burst
windows — the exclusion is load-correlated by construction).

**Estimators.**

**(a) Lead–lag response function.** 100 ms grid over admissible windows.
`Δm^PM_t` = change in Up-token book mid in probability-cents; `ΔF_t` = Binance
log-return in bps from `data/mm_hf/raw/bookTicker/<SYM>/`. Fit
`Δm^PM_t = Σ_{k=−10}^{+30} β_k ΔF_{t−k} + ε` with day-clustered SEs; report the
lag profile. `L_50` = smallest `k ≥ 0` with `Σ_{j≤k} β_j ≥ 0.5 Σβ`;
block-bootstrap over days (B = 2000) for its CI.
*Free specification test of §2:* compare `Σ_k β_k` against the theoretical binary
delta `φ(d)/σ_eff` per time-in-window stratum. Ratio ≈ 1 ⇒ the book is efficient
in the Binance information set (a pure speed/subsidy game). Ratio ≪ 1 ⇒ the book
is **structurally under-reactive** — free money for the fast, and a *forecastable*
mid for us, which is a different (and better) strategy than defensive quoting.

**(b) Pickoff intensity and cost** — the direct measurement of `δ_tox`.
Define a burst as `|ΔF|` over 200 ms exceeding **θ = 2 bps** (frozen; ≈ 5–7
probability-cents ATM). For each burst measure
`λ_snipe(Δ) = P(a trade prints on PM on the wrong side of post-burst fair value
within Δ ms)` and `c_snipe(Δ)` = mean adverse displacement in c/share,
notional-weighted, for `Δ ∈ {100, 250, 500, 1000, 2000} ms` (Holm across the 5).
`c_snipe(L_us)` **is** the empirical adverse-selection floor for a maker with
reaction time `L_us`.

**(c) Stale-quote survival — the slow-maker test.** Conditional on a burst,
`P(the pre-burst touch level is cancelled before it is traded)` as a function of
Δ.

**Decision rule (numbers).**
- **EXPOSED (mechanism confirmed)** if `c_snipe(250 ms) ≥ 1.0 c` with
  day-clustered block-bootstrap 95% CI-lo > 0.5 c. ⇒ §3's `δ_tox` floor is set to
  the measured `c_snipe(L_us)`, the `(|d|, r)` pull surface is parameterised by
  it, and an ATM quote must rest `≥ ⌈c_snipe⌉` ticks back.
- **SLOW-MAKER-SURVIVES** if additionally cancel-before-trade `≥ 0.8` at
  Δ = 250 ms. If `≤ 0.5`, the venue is a pure speed game at our latency and the MM
  track must either colocate (London — **a restricted jurisdiction**, M15) or
  restrict quoting to the `|d|` region where `φ(d)` collapses. That is a
  deployment-shaped conclusion reached from mechanism data alone.
- **REFUTE the latency framing** if venue-intrinsic `L_50 ≥ 2 s`: PM would then be
  so slow that Binance is a **forecast**, not a defence, and the model flips from
  defensive quoting to informed quoting — a different strategy with a different
  literature.
- **UNDERPOWERED**: < 200 bursts at θ = 2 bps across < 10 distinct UTC days.

**Prereqs / read date.** E0 PASS/CONDITIONAL + E-M1(a) `R_touch ≥ 0.90`.
Binance leg required ⇒ 6-coin universe, **HYPE excluded**. Bursts are frequent;
the binding constraint is **≥ 10 day-clusters** ⇒ **2026-08-30**.
Not blocked by E-M6 (it uses book mid, not `p̂`) — but its conditioning is on
`|book_mid − 0.5|` until E-M6 reads.

**Model consequence.** Sets `δ_tox` and the pull surface numerically; decides
whether pull-on-burst is sufficient or merely necessary; determines whether
deployment requires London proximity; and, via the `Σβ` ratio, tells us whether
the book is structurally stale (an informational opportunity the plan does not
currently contemplate) or efficient.

---

## E-M8 — expiry: does inventory really self-liquidate?

**Mechanism.** M-8: inventory expires into a Bernoulli payoff; no liquidation
leg. This is what makes the 350 bps taker fee survivable and is the reason §3's
inventory penalty is `q²·p̂(1 − p̂)` rather than a diffusion term.

**Propositions.** (i) Every admissible window reaches a CLOB-confirmed final
resolution; (ii) the resolution lag is bounded; (iii) there is no void /
re-resolution / winner-flip state; (iv) the terminal payoff is $1 per winning
share with no haircut; (v) the outcome variance conditional on the book mid is
actually `mid(1 − mid)`, which is what the penalty assumes.

**Estimators.**
1. **Completeness**: fraction of admissible windows reaching `closed=True` with a
   winner; distribution of `resolution recv_ns − window_end`. *(Current inventory:
   871 final of 892 markets = 97.6% — must be re-measured on the admissible set,
   with the missing ones' cause identified.)*
2. **Flip rate**: any slug whose `winners` changes across polls (dedupe-keep-last
   silently hides this — count it explicitly).
3. **Terminal pinning**: distribution of book mid at `T − 5 s` and `T − 1 s`;
   fraction of windows with mid ∈ {0.01, 0.99} at `T − 1 s`; and the disagreement
   rate between a pinned late mid and the realised winner.
4. **Reliability of the venue** (not of our model): bin windows by book mid at
   `t ∈ {60, 150, 240} s`, compare realised winner frequency to the bin mid,
   day-clustered, notional-weighted, on model-free bins.

**Decision rule.** Completeness ≥ 99%, **zero** winner flips, resolution-lag
p95 < 300 s ⇒ **self-liquidation CONFIRMED**; §3's terminal-payoff formulation and
the "never unwind" doctrine stand. Completeness < 95%, **or any** winner flip,
**or any** void ⇒ a residual settlement-risk term is added to `v(t)` and
inventory can no longer be treated as self-liquidating — which reintroduces an
unwind leg at 350 bps and materially damages the whole thesis.
Separately: if realised outcome variance given mid is materially below
`mid(1 − mid)` (paired test on binned residuals, day-clustered, Holm across the
3 decision times), the inventory penalty is **mis-sized** and `γ`'s calibration
recipe changes.
**UNDERPOWERED**: < 2,000 admissible windows or < 7 UTC days.

**Prereqs / read date.** E0 + **E-M6** (the near-tie subset — the windows where
resolution risk actually lives — is only definable once the convention and
`δ_jit` are known). **2026-08-27.**

**Model consequence.** Validates or kills §2.2/§3's Bernoulli terminal penalty
and the "never unwind" doctrine that the punitive taker fee makes load-bearing.

---

## E-M4 — fee mechanics (confirmatory + the unverified rebate)

**Mechanism.** M-4: `fee = C × 0.07 × p(1 − p)` on the taker leg, maker legs zero,
maker rebate ≈ 20% of collected taker fees ≈ 70 bps of notional per fill.

**Status.** The *formula* is already resolved on-chain to the cent (M lens: two
exact matches). What remains open, and is decision-relevant, is narrower:
(i) is the maker leg **always** zero, including in mint-crossings; (ii) does
`p` in the formula mean the **trade price** or the **mid**; (iii) **is the maker
rebate actually paid?** — the same evidentiary question as M-5, and it concerns
the *larger* of the two claimed subsidies (M lens: rebate pool $19–38k/day vs
rewards $18.3k/day).

**Estimator.** On the frozen ≥ 500-tx on-chain sample from E-M3(b): regress the
observed fee word on `C · 0.07 · p(1 − p)` with `p` = trade price and, separately,
`p` = book mid at the print. Report slope, R², and max |residual| in µUSDC.
Rebate: search daily USDC transfers at 00:00 UTC from the exchange/distributor
contract to the top-10 maker wallets identified in the M-lens census.

**Decision rule.** Slope ∈ [0.999, 1.001], max residual < 10 µUSDC with
`p` = trade price ⇒ formula CONFIRMED and frozen into the cost model; the mid
variant is recorded as refuted. **Any** maker leg with fee > 0 ⇒ maker-fee risk is
live and enters as a standing sensitivity. Rebate: **≥ 1 identified payout in
14 days ⇒ CONFIRMED**; **zero payouts in 14 days ⇒ the 70 bps/fill rebate line is
struck from §1** on exactly the same evidentiary standard applied to rewards.
Never read fees from the WS `fee_rate_bps` field — it is unpopulated
(`"0"` on every observed trade, including a print we verified against on-chain
value).

**Prereqs / read date.** E-M3(b) sample. Formula **2026-08-24**; rebate
**2026-09-03**.

**Model consequence.** Freezes the fee arithmetic; and, if the rebate is
unverified, changes what "unsubsidised maker" even means — because the rebate,
not the rewards program, is the larger claimed subsidy in §1.

---

## E-M5 — rewards: is there ANY observable evidence these markets pay?

**Mechanism.** M-5: a rewards band is a quoting obligation with parameters we
cannot verify.

**Status — contradictory, and this is the point.** The M lens matched our
`conditionId`s to `GET /rewards/markets/current` and read
`rate_per_day = $10,000`, band 1.5 c. The orchestrator's later verification
exhausted 33 pages / 16,172 rows and found our condition_ids **absent**, and
`GET /rewards/markets/<cid>` empty. Our collector records the miss:
`rewards_authoritative: null`, `rewards_registry_n: 16196`. Gamma still serves
`rewardsMaxSpread: 4.5`, `rewardsMinSize: 50`, and `clob.rewards.rates: null`.
So the mechanism is **unverified from every registry we have**, and a PnL-shaped
G3b is un-preregisterable regardless of scope.

**Proposition (behavioural, so it does not depend on any registry).** If these
markets pay a band-riding reward with band `B` and min size `Q`, resting depth
exhibits (i) a discontinuity in the depth-vs-distance-from-mid profile at exactly
`mid ± B`, (ii) a mass point at `Q` shares, (iii) a regime change in
two-sidedness at the `0.10 / 0.90` mid boundaries where two-sided quoting is
documented as compulsory. If none of these exist, the markets are not rewarded —
or the reward does not shape behaviour, which for modelling purposes is the same
thing.

**Estimators.**

**(a) Band discontinuity (RD).** Pool admissible windows; on a 1 s book grid
compute displayed size vs signed distance from mid `d = |price − mid|` in **0.5 c
bins** (tick-aware — E-M2). Local-linear RD at candidate bands
`B ∈ {1.0, 1.5, 2.0, 3.0, 4.5} c`: estimate the jump
`J(B) = lim_{d↑B} size − lim_{d↓B} size`, day-clustered SEs, **Holm across the
5 candidates**.
*Direction matters and is already suggestive:* the M lens measured BTC depth
within 1 c of mid at 138 shares vs 1,290 within 4.5 c — depth sits **outside**
1.5 c, which is evidence *against* a 1.5 c band being farmed and is consistent
with the registry miss.

**(b) Size mass point.** Individual order sizes are not observable (level totals
only), but level-total histograms will show excess mass at multiples of 50 if a
min-qualifying size binds. Metric: excess mass at {50, 100, 150} vs a smooth
kernel fit, day-clustered.

**(c) Two-sidedness regime change.** Fraction of 1 s samples with ≥ 50 shares on
both sides within `B` of mid, conditional on mid ∈ [0.10, 0.90] vs outside. A
break at exactly 0.10/0.90 is a strong signature of the documented scoring rule
and is *identified independently of the band*.

**(d) Band re-cut natural experiment.** If a re-cut occurs (the M lens claims
4.5 → 1.5 on 2026-08-20), test for a break in (a)'s jump location across the
re-cut timestamp via a day-clustered difference-in-differences on the depth
profile. This is the cleanest identification available and costs nothing — but it
requires the event to be captured, so **the collector must keep snapshotting the
registry and Gamma continuously** (it already records
`rewards_authoritative` / `rewards_registry_n` per market).

**(e) Primary-source resolution** (not a data experiment, but part of the ladder,
and available immediately): version the August program announcement and the
rewards/fee docs into `data/pm_5min/docs/`; and attempt to observe an **actual
payout on-chain** — a daily 00:00 UTC USDC transfer from the distributor to a
wallet whose activity is 5-min crypto quoting. One observed payout settles
eligibility definitively.

**Decision rule.**
- **ELIGIBLE-CONFIRMED**: (e) observes a payout, **or** (a) identifies a single
  candidate band with |t| ≥ 3 after Holm that matches a registry/docs value. ⇒ the
  band constraint enters §3 and the `(|d|, τ)`-conditional occupancy rule becomes
  the primary quoting refinement.
- **NOT-ELIGIBLE (refuted)**: no significant discontinuity at any candidate across
  **≥ 14 days**, **and** the registry miss persists, **and** (e) finds no payout.
  ⇒ **rewards are removed from the model entirely**: §3 drops the band
  constraint, **M-5 is struck from the mechanism inventory**, and H-PM4's
  program-level time-box collapses — the program never had the subsidy, and the
  question simplifies to "is there an unsubsidised maker mechanism?"
- **UNDERPOWERED**: < 14 days, or 0.5 c bins with < 100 observations.

**Prereqs / read date.** E0 + E-M2 (tick-aware binning). Independent of E-M6.
RD needs day-clusters ⇒ **2026-09-03**; (e) is available immediately and should
be attempted first, because a single observed payout makes (a)–(d) confirmatory
rather than decisive.

**Model consequence.** A binary switch on whether §3's objective has a band
constraint at all — it installs or deletes an entire term, and it decides whether
the program is time-boxed by an August subsidy.

---

## Cut list — old §5 items dropped, and why

| dropped | why |
|---|---|
| **PM-E3 MM economics replay + gates G3a / G3b** (net PnL, day-clustered t ≥ 2, bootstrap CI-lo > 0) | Out of scope under §9 — these are profit gates. The *mechanism* content is preserved and re-homed: fill bracketing → E-M1(a,b) as a measurement of bracket **width** (an epistemic quantity, not a PnL); adverse selection → E-M7(b) in probability-cents per fill. Note G3a was already invalid as a counterfactual (M1) and G3b was already un-preregisterable (M2 correction) — the scope change resolves both by deletion rather than repair. |
| **Capacity / PnL(α) participation curve, rewards-share dilution** (S4d, SF-7) | Explicitly out of scope (sizing). The M-lens $10–30k/mo figure survives only as a recorded aside. |
| **PM-E4 live paper / min-size probes** | Deployment, not research. The one deployment fact that *is* mechanism-derived — London proximity vs a restricted jurisdiction — is produced by E-M7 and recorded there. |
| **PM-E2 model contest (G2: Brier/log-loss, p̂ vs book mid)** | **Demoted from gate to deferred diagnostic.** It is a forecasting-skill question, not a mechanism question, and it is structurally void until E-M6 fixes `p̂`'s target. The full S-lens spec (MF-1..MF-4, MF-8: paired ΔBrier, local-recv clock both sides, 9-point grid, window = unit, paired-complete, fit ≤ d−1, pooled-notional + 3 frozen model-free strata + Holm, ≥ 28 days, isotonic-recalibrated-book secondary baseline) is retained **verbatim** in `PM_PREREG.md` for the day it is run. |
| **PM-E2.5 shadow-quote markout anatomy** (S4a / SF-4) | **Absorbed** into E-M7(b)(c). The mechanism core — how badly does a resting quote get run over after a Binance burst — survives; the PnL wrapper does not. |
| **Rewards-band occupancy feasibility** (S4b / SF-5) | **Absorbed** into E-M5(a)(c), and re-posed as an *identification* question (does the band exist?) rather than a feasibility question (can we afford it?), because occupancy cost is a PnL quantity. |
| **H-PM1b manipulation screen** (S4c / SF-6) | **Retained, re-homed** to E-M6c — it is a settlement-mechanism question, not a standalone experiment. |
| **Cross-venue Binance-perp hedge ablation** (§3) | A PnL ablation with a cost threshold. Cut. |
| **Fee-regime sensitivity rows** (SF-1) | **Absorbed** into E-M4; with the formula resolved on-chain, sensitivity rows are replaced by a direct confirmatory regression plus the (open) rebate-payment question. |
| **PM-E1 anatomy as a standalone rung** | Dissolved into the mechanism experiments that each of its tables served: spread/tick → E-M2; depth profile → E-M5; flow timing and calibration → E-M8; fee empirics → E-M4; settlement reconstruction → E-M6. The "freeze the gate numbers at the end of E1" step is replaced by the design-freeze date above. |
| **Naive-quoter baseline arm** (SF-3) | Kept only as E-M1's **harness negative control** (never-quote ⇒ zero fills; 0.99-bid ⇒ fills on every print). Its original role was to bound PnL; that role is gone. |

---

## What goes into `PM_PREREG.md`

Owner: orchestrator. Frozen **2026-08-21**, covering: the admissible-window
definition and the blacklist (E0.4); the 7-coin universe and the ex-ante notional
weight vector; the 3 model-free strata; the E-M6 12-cell convention grid; every
threshold appearing in bold above (99.0 / 99.5 / 0.99 / 0.95 / 0.60 / 0.30 /
0.005 / 0.05 / 0.01 / 250 ms / θ = 2 bps / 1.0 c / 0.8 / 0.5 / 20% / 3 s);
the Holm families as enumerated per experiment; B = 2000 block bootstrap over UTC
days; the minimum read dates in the ladder table; and the amendment rule (any
post-freeze change is logged with rationale and date, and re-reads restart from
the amendment date). Harnesses read constants from this file, never inline.
