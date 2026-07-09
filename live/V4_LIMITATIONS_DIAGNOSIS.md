# v4 production-config limitations diagnosis (2026-07-09)

Config-aware diagnosis of what production v4 ACTUALLY trades (not the vanilla book the variant
cells used). Purpose: locate the real limitations before optimizing. Grounded in `run_convexity_v4_live.sh`
+ this session's validated findings + a per-regime book-net attribution (residual-alpha AND naked
frames, defined below).

## Production v4 = a regime-SWITCHED strategy

Per `run_convexity_v4_live.sh`, the model preds feed a regime switch (btc_ret_30d):

| regime (btc30) | what production does | share (rec/oos) |
|---|---|---|
| **side** (−0.10..+0.10) | model 1L/2S, REGIME_GATE, inv_sqrt_vol | 54% / 55% |
| **bear** (< −0.10) | equal-weight 1L/2S (BEAR_MODE=equal); **DD-stop OFF** (STOP_SKIP_REGIMES=bear) | 36% / 10% |
| **bull** (0.10..0.15) | **FLAT — gated** (BULL_GROSS_MULT=0) | 7% / 12% |
| **deep bull** (≥ 0.15) | **mom1d long-only** (long top-2 by return_1d, ½ gross) | 3% / 23% |

Plus GLOBAL_GROSS_MULT=0.5 (live cap), kill-switch.

## Per-regime book-net attribution (net, book 0.5/0.5, 9-bps RT convention, daily Sharpe)

**FRAME (read carefully):** these are per-cycle BOOK NET in bps for the STANDARD beta-neutral 1L/2S
MODEL book (long top-K / short bottom-K by pred), bucketed by regime — i.e. "what the model book
does in each regime" — shown in BOTH the RESIDUAL-alpha frame (`alpha_vs_btc` — the v4 TARGET, where
the model's skill lives) and the NAKED realized-return frame (`return_pct` — what you'd actually
book). They are within a few bps of each other because the book is ~beta-neutral, so the diagnosis
is frame-independent. **These are NOT the production +2.22 Sharpe:** they are PRE-GATE (no
REGIME_GATE/DD-stop → actual production per-regime net is ≥ these) and per-regime CONDITIONAL
Sharpes (not comparable to the all-regime, post-overlay canonical number). **Crucially, the
deep-bull row is this beta-neutral MODEL book restricted to deep-bull cycles — the COUNTERFACTUAL
production abandons — NOT the mom1d long-only overlay production actually runs there** (that overlay
is long-only, not beta-neutral; its real performance is §6.1/Q3, cross-referenced in limitation #3).
Small-sample flags noted.

| regime | REC residual / naked (net/Sh) | OOS residual / naked (net/Sh) | production action |
|---|---|---|---|
| side | **+18.7/+4.09 ; +21.3/+4.61** | **−0.1/−0.04 ; +0.6/+0.17** | traded |
| bear | −4.9/−0.88 ; −2.5/−0.44 | **+17.3/+4.46 ; +14.8/+3.94** | traded |
| bull (mild) | +45.2/+4.36 ; +47.6/+4.58 (n=114, short-driven, era-fit) | −7.0/−1.47 ; −5.3/−1.19 | GATED |
| deep bull | −32.3/−4.55 ; −31.1/−4.40 (n=47) | −0.9/−0.22 ; −1.4/−0.32 | mom1d long-only (row = model counterfactual, NOT the patch — see #3) |

(Format: `residual net / residual Sharpe ; naked net / naked Sharpe`. Residual = v4 alpha target;
naked = realized. Diagnosis below holds in both frames.)

## The core limitations (what this reveals)

1. **NO regime has a consistent both-era edge — every edge is era-dependent.** Side carries
   RECENT (+4.09) but is FLAT OOS (−0.04). Bear is the anchor OOS (+4.46) but flat-within-noise
   recent (−0.88 Sh over ~97 bear-days ⇒ t ≈ −0.45 — NOT a significant drag; recent simply can't
   speak to it). Mild-bull would've earned recent (+4.36) but loses OOS (−1.47). The strategy is a
   *collection of era-specific regime edges*, and the config (gate bull, trade bear) is implicitly
   a bet on which era you're in. This is the deepest limitation and the mechanism behind the 2022
   holdout FAIL + the 0.5× cap. (Best-powered evidence is the SIDE bucket — 54% of cycles, +4.09
   rec / −0.04 OOS = the same event-concentration as #5; the thin bear/bull buckets corroborate the
   pattern but carry sampling noise.)
2. **The bull gate is blunt — but deliberately so.** The two bull sub-regimes are handled
   separately: in mild-bull (0.10-0.15) BULL_GROSS_MULT=0 zeroes the model book → FLAT; in deep-bull
   (≥0.15) the model book is likewise zeroed but the mom1d long-only overlay runs instead (#3). What
   the mild-bull gate "gives up" is a RECENT-ONLY bucket that was strongly positive (short-driven
   "pump topping reverts", +4.36) but −1.47 OOS — an era-fit bucket (n=114, short-driven, +4.36 rec
   / −1.47 OOS = the #1 pattern), NOT a durable edge. So the gate's bluntness is the correct
   conservative choice, not a real cost. (See "config choice" below: the mild-bull split is the
   era-trap, not a lever; the deep-bull lottery is the only live bull question.)
3. **In deep-bull the model's cross-sectional book is weak** (beta-neutral counterfactual −32 rec
   small-n / −0.9 OOS) — which is WHY production abandons it for the mom1d long-only overlay. But
   that overlay is NOT a dead patch: it EARNS, via generic long-alt BETA — §6.1/Q3: OOS signal
   +62k gross, essentially matched by random-alt picks (placebo median +54k gross) and ~2× a
   BTC-long (+27k). What's unproven is the return_1d RANKING (Q3 p=0.215), not the PnL. So deep-bull
   handling is an UNVALIDATED DIRECTIONAL BETA LOTTERY (§5: deep-bull long median −73/−107, top-3
   cycles ≈ 97-106% of totals), not a broken selection leg. The real question is whether a
   ~beta-neutral strategy should hold that lottery bet, not whether it "earns."
4. **Bear is traded unconditionally** (BEAR_MODE=equal, DD-stop OFF in bear) — a bet that bear
   reverts to its OOS-anchor behavior. The recent bear number (−0.88) is within noise (t ≈ −0.45),
   so it neither confirms nor refutes the bet. The squeeze tail (short median +42 but mean
   tail-gutted, skew −2.45) is the risk, and it is unhedged (SQ1 crowding signal exists but is too
   weak/non-stationary on free data — SK1 failed the recent holdout).
5. **The workhorse (side) alpha is thin and event-concentrated** — 76% of side net from 2
   dispersion months, 5/9 months positive, flat OOS. No config fixes this; it is a signal/data
   limit.
6. **The regime label is a 30-day-LAGGING classifier.** Every switch keys off `btc_ret_30d`, so
   "which regime am I in" is itself a noisy, trailing estimate — worst in fast transitions (a
   2022-style cascade, a sharp bull→bear flip), where production can apply "bear" rules to the
   start of a recovery (or "side/bull" rules into a breakdown). Regime is not a clean partition; it
   is a lagging guess at one, which compounds the era-dependence in #1.

## Optimization map (limitation → lever, with honest priors)

| limitation | candidate lever | prior / status |
|---|---|---|
| era-dependent regime edges | finer/adaptive regime detection | LOW — adaptive-timing failed 7× (dynamic K, gates, rvol-scaling); "trailing estimators lag era" |
| blunt bull gate | split gate: trade mild-bull (0.10-0.15), gate deep-bull (≥0.15) | LOW — this IS the #1 era-trap (mild-bull +4.36 rec / −1.47 OOS is the textbook era-dependent bucket); uniform bull0 is correct BECAUSE it refuses the era bet. Do not test. |
| deep-bull = directional beta lottery | pre-registered {mom1d vs flat vs alt-rule} in deep bull | worth a test — but as KEEP/DROP a high-variance long-alt beta bet (it earns via beta, §6.1), NOT as "remove a dead patch"; era-neutral (bad-to-flat both eras) so simplifying is not an era bet |
| regime label is 30d-lagging (#6) | (structural) faster/robust regime estimator | LOW — same adaptive-timing failure class; naming it, not proposing to chase it |
| bear unconditional / squeeze tail | CUSUM gross-throttle (regime-level); liquidation data (SQ1) | CUSUM wired-not-lever; liquidation = the paid-data route |
| thin event-concentrated side alpha | dispersion timing (FAILED); new data | dispersion timing dead; execution/cost is the real lever |
| era-fragility (2022 FAIL, 0.5× cap) | forward ledger (release cap); universe-portable build | operational; the binding path forward |

## The config choice worth re-examining (NEW, actionable)

Unlike the model/feature axis (exhausted, 0 promotions), ONE config choice has a live question the
config-aware attribution surfaces cleanly:
- **The deep-bull mom1d overlay** — a pre-registered {mom1d vs flat vs alt-rule} test in deep bull.
  Framing (corrected from an earlier "it doesn't earn" reading — see #3): the overlay DOES earn, but
  via generic long-alt BETA (§6.1/Q3: OOS +54k, matched by random alts), with the return_1d RANKING
  unproven — so it is a high-variance DIRECTIONAL beta lottery. The test is a KEEP/DROP on whether a
  ~beta-neutral strategy should hold that bet; it is era-neutral (bad-to-flat both eras), so it is
  NOT the #1 era-trap.

**NOT worth re-examining: the bull-gate split** (trade mild-bull). It is the single clearest
instance of the #1 era-trap — mild-bull is +4.36 rec / −1.47 OOS, and splitting the gate to harvest
the recent side is exactly the era bet the diagnosis warns against (adaptive-timing has failed 7×).
Uniform bull0 is correct because it refuses to classify the era.

The deep-bull test is an OVERLAY/config test (not a model change), so it needs the full-stack replay
discipline (faithful cost, no path-coupled variant noise — the estimator law), NOT the vanilla-book
cell machinery. Everything else points back to the standing levers: forward ledger, execution/cost,
paid positioning-depth data.

## Caveats on the numbers

Pre-gate book-net attribution of the STANDARD beta-neutral 1L/2S model book, bucketed by regime, in
BOTH the residual-alpha (v4 target) and naked (realized) frames — within a few bps of each other
because the book is ~beta-neutral. It is the pre-gate regime EDGE (the right lens for "where is the
model's signal"), NOT production PnL: REGIME_GATE, DD-stop, and inv_sqrt_vol sizing are not applied,
so actual production per-regime net is ≥ these, and the per-regime conditional Sharpes are NOT
comparable to the all-regime post-overlay canonical +2.22. **The deep-bull row is the beta-neutral
MODEL counterfactual (what the model book would do there), NOT the mom1d long-only overlay
production runs — see limitation #3 and §6.1/Q3 for the overlay's real numbers.** Recent bull
(n=114) and deep-bull (n=47) are small-sample; recent bear (−0.88 Sh) is within noise (t ≈ −0.45).
Cost = 9-bps RT per §8 convention; Sharpe = daily-aggregated then ×√365.
