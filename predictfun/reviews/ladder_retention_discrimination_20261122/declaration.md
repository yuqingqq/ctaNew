# LONG-VALUE-RETENTION-DISCRIMINATION-v1 — declaration (frozen before any label or fit)

2026-11-22, session aea5bfb9.
- **Specified by:** review `838f4be2`'s ONE next priority.
- **Approval:** the USER, via AskUserQuestion, approved this new label-and-discrimination scope at **≤ 1.25 additional core-hours**, including a 0.05 h review reserve, with a **ladder line ceiling of 4.00**.
- **Builder:** `predictfun/research/pf_ladder_retention_discrimination.py`. Its docstring is the full rule.

## Question and target

Can information available at an order's **first review** identify orders whose **full joint continuation value** is better if retained than if cancelled and re-entered?

**Target:** the accepted paired full-book Y to the common supported END: the pilot's frozen fork, focal-lineage continuation and gates (`pf_ladder_long_continuation`, frozen `7407bdf1`). First-fill loss, 1 s markouts and FIFO component signs never replace it.

**What it is not:** a strategy, a composed policy replay, a validation or a profitability claim.

## Population (census frozen here)

- **Arm and markets:** CURRENT_ASYNC_FAST10 on the **19 complete consumed markets**, Oct 24–27 (ASYNC pilot + rest records, hash-bound).
- **REVIEW states:** each generation's first **policy-consulted** recorded live KEEP. The census has every eligible state (`census.csv`, its sha256 in `manifest.json`).
- **Sample:** per day × side, the **32** states with the smallest sha256(arm|day|m|REVIEW|side|gid), the pilot's key, giving up to **256 forks**. Inclusion probability is 32 / n(day, side).
- **Overlap:** the exact overlap with the pilot's REVIEW labels is recorded.

No outcome or branch fill enters the selection. ADMISSION stays deferred.

## Inputs (causal; recorded at the callback, identical in both arms)

17 features, from the client view, the order view, the **received** book and the guard's **receipt-clocked** pressure cache:

| Group | Features |
|---|---|
| Order | age, rung, distance from the received best (ticks), size, client-known remaining |
| Book | received spread, received size at the order's price |
| Inventory and risk | client inventory toward the order's side, side / opposite reservations, unclosed orders on each side, mode restricted |
| Signals | pressure value and its ok flag |
| Time and flow | time left to END, received mid change over 1 s toward the side |

No venue queue and no future fill information is used.

## ONE candidate (fixed here)

**Ridge on the 17 features:**
- each feature winsorized at its training 1st / 99th percentile and standardized (sd 0 → 1);
- target Y winsorized at its training 5th / 95th percentile;
- **λ = 1.0 × n_train**;
- intercept = the clipped training mean.

**Protocol:**
- Train on the **ok** pairs of **Oct 24–25**; score the ok pairs of **Oct 26–27 once**.
- Decision: **retain iff predicted Y > 0**.
- Count: one new adaptive development candidate. No threshold search, feature race or tuning after scoring.

## Measurement and reading

**On the evaluation days:**
- **Rule value:** V_rule = Σ Y over retained opportunities (zero for the alternative), also per day, side and market and inclusion-weighted.
- **Controls:** always-retain, always-alternative (0), and the training constant (retain all iff the training mean Y > 0).
- **Matched draws:** **1,000** (seed 20261122), keeping the rule's retain count per evaluation day, giving a one-sided p.
- **Diagnostics:** MSE vs zero / constant; calibration by predicted quartile; the selected negative tail and top-5 concentration; movable (non-tie) support.

**Reading:**

| Reading | Condition |
|---|---|
| `INSUFFICIENT` | training or evaluation has < 30 non-tie ok pairs, or the rule retains < 5 or > n − 5 |
| `useful_local_discrimination` | V_rule > 0, V_rule > every control and p ≤ 0.05 pooled, **and** on each evaluation day V_rule_day > 0 and above that day's controls |
| `not_reliable` | the base parity fails |
| `no_useful_discrimination` | otherwise |

Pairs failing a gate keep their status and are excluded, with counts.

**What the readings permit:** these are independent fork decisions, not a composed policy. `useful_local_discrimination` would motivate a policy design on untouched data, which needs the USER. It is not promotion.

## Gates (per pair; a failure is a status)

- the pilot's gates;
- the **expected callback** tuple (live, t, gid, KEEP / PULL) once per arm at equal contexts;
- features recorded and identical in both arms;
- the unused-exemption control against the saved ASYNC record;
- base parity per market.

## Run and resources

**Parts:**
1. **prefix:** Oct 24, 7 markets, 64 forks.
2. **Write-once projection:** spent + 3.3 × the prefix + a 120 s tail + the 0.05 reserve ≤ 1.25. The line + those must be ≤ 4.00, and the peak < 1,536 MiB.
3. **rest:** Oct 25 / 26 / 27 in parallel.
4. measure, then checks.

If the projection does not fit, stop for a revised resource decision.

## Verification before the freeze

- **Selftest:** 6 groups:
  - features by definition;
  - run_market gates on a synthetic market;
  - known-bad controls: features differing between arms, an unexpected callback, a retain arm unequal to the saved record;
  - the candidate reproduced by hand;
  - evaluate with all readings, including a one-day-fails case;
  - the guards.
- **Mutation:** 26 / 26 planted errors caught (`logs/mutants_try2`).
- **Checker:** `rd_checks.py`, families A / B / Y / G / P / M / I:
  - an independent census, Q / Y, callback tuples and features across arms with sanity;
  - an independent ridge refit, the metrics and draws, and the reading;
  - a 12-case falsifier.
- **Rehearsal:** `tools/rehearse_rd.py` ran census → manifest → run_part → measure → checks on two synthetic days. All families passed except `A_committed`, and all cases were refused.
