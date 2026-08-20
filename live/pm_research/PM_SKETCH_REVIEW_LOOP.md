# PM_SKETCH_REVIEW_LOOP — MM experimental-sketch review loop (P-2026-003)

Charter (user-initiated 2026-08-19): **adversarially review the MM experiment
sketch — `PM_MM_PLAN.md` (model §2–§3, ladder §5, hazards §6) — until it is
sound enough to pre-register against.** Runs alongside the data loop
(PM_REVIEW_LOOP.md); this loop owns THEORY + DESIGN, that one owns DATA.
Convergence: two consecutive iterations with zero MUST-FIX.

Iteration protocol: three independent reviewers with distinct lenses, then
orchestrator triage → plan amendments (logged in the plan) → next iteration
re-reviews amended plan.

| Lens | Owns |
|---|---|
| T (theory/math) | every formula re-derived; model assumptions; binary-GLFT coherence; probability- vs price-space consistency |
| S (statistics/design) | gates measurable + powered; scoring rules; cluster units; multiple testing; prereg-freeze quality; kill sharpness |
| M (mechanism/competition) | who is the counterparty; matching/microstructure reality; rewards-game equilibrium; capacity; operational frictions |

Reviewers write PM_SKETCH_REVIEW_ITER<n>_{T,S,M}.md; triage table per
iteration recorded here.

| iter | date | MUST-FIX (T/S/M) | status |
|---|---|---|---|
| 1 | 2026-08-19/20 | T:4, S:9, M:5 (=18) | T+M applied inline; S adopted into PM_PREREG.md (to write) |

**M-lens headlines** (full list in plan §8): fees are REAL and resolved
on-chain (taker ≈3.5% of notional ATM; maker rebate ≈70 bps/fill; the WS
`fee_rate_bps=0` was an unpopulated field, not zero fees) → §1 rewritten;
rewards pool is $550k/**month** and August-only → program time-box; the
rewards band must come from the CLOB registry, not Gamma (collector PATCHED —
Gamma served a stale band after a 2026-08-20 re-cut); **G3a as written is not
a valid counterfactual** (rewards and adverse selection are coupled through
the band constraint — the no-rewards arm must re-optimise, not subtract);
latency ~120–250 ms Binance→PM is first-order because 1 bp of BTC ≈ 3 ticks
of binary ATM. Capacity ceiling: net **$10–30k/mo** (plausibly $0) on
$25–100k capital.

**Iteration 1 outcome.** 13 MUST-FIX from two lenses (M was interrupted by a
session limit and resumed to write its deliverable). Model changes applied to
PM_MM_PLAN §2/§3 and logged in its §8:
- fair value is now **stream-anchored** (synthetic-Binance X̂ is noise inside
  r ≈ 17–26 s — the single most consequential finding; retroactively justifies
  the data loop's TWAP-topic subscription);
- quoting restated as **discrete per-level EV** (continuous-δ GLFT is
  meaningless on a $0.01 grid with 2–4-tick books);
- v(t) is a **min-structure** (p̂(1−p̂) already IS remaining QV — the old sum
  double-counted);
- pull rule is a **(|d|, r) surface** (ATM λ_bin diverges as 1/√r).
All §2 variance/expectation laws re-derived and MC-verified CORRECT.
Statistics lens: G2 must be a **paired** ΔBrier test (difference-in-
significance is this repo's historical killer), matching is **price-time**
(pro-rata was mislabeled), and **read dates moved out** — G2 ≥28 d, G3a 4–6 wk,
because effective breadth is ≈1.4 coins (BTC ≈85% of notional).

**Iteration 2 agenda** (after M lands + PM_PREREG.md is written): re-review the
amended §2/§3 for internal consistency; verify the S spec items were adopted
faithfully (not paraphrased into weakness); apply the 6 queued T SHOULD-FIXes;
check the new arms (PM-E2.5 shadow quotes, capacity curve) are specified well
enough to implement; confirm no gate reads a pre-2026-08-07 era.
