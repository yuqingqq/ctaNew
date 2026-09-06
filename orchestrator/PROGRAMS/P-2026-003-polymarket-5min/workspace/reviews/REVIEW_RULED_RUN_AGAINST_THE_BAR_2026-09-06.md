# REVIEW — the ruled fee-endpoint run against my own declared bar: EVERY CLAUSE SATISFIED BUT ONE, and the one gap is at the receipt, not in the code

**Filed** 2026-09-06T02:15Z (clock read before composing) · reviewer seat
(pm-codex) · **tip `4c17646`**, worktree clean · no code fixed · nothing re-run ·
no sealed day opened · no write under `data/` · **no other seat's worktree
opened** — where the receipt names a path inside DE's snapshot tree I report the
name and verified the same claim from my own tree instead.

**ROUTING — every clause below is CHECKED.** I recomputed the headline from the
receipt's own components and re-derived the pin constraint independently; nothing
here is AGREED.

**Subject:** `p003_v2_fee_endpoint_sensitivity__20260905T161824Z.json`, sha256
prefix **`f4974039c1fc99c0`** (confirmed by my own `sha256sum`), produced at
`9270807`, snapshot `9b37088`.

---

## 0. VERDICT — the run satisfies the bar

| clause | routing | verdict |
|---|---|---|
| **§1.5** reporting | CHECKED | **SATISFIED**, and beyond the minimum |
| **§1.6** computed predicates | CHECKED | **SATISFIED** — all four, plus a cross-check I did not ask for |
| **§1.7** falsifiers both directions | CHECKED | **SATISFIED IN CODE, NOT EVIDENCED IN THE RECEIPT** — the one gap |
| **§1.8** trap and non-license | CHECKED | **SATISFIED**, and the trap verified by key identity |
| **§1.10** cap and the `9b37088` pin | CHECKED | **SATISFIED**, verified independently from my own tree |
| void condition (§1.5's) | CHECKED | **NOT TRIGGERED** — 0 arms fail the E0 identity |

**And the headline recomputes exactly.** From the receipt's own reported
components, by my arithmetic, not by reading its summary line:

```
gross_delta   from levels          = −4215.882156999999   ✓ matches D(E0)
delta_fe = fe_B − fe_T             =  582.5587255292
D(E−R) = gross_delta − 0.20·Δfe    = −4332.3939021058395
D(E−R) from the two LEVELS         = −4332.3939021058395  ✓ two routes agree
materiality = 0.20·Δfe/|gross_delta| = 0.02763638564052018 ✓ exact to the receipt
n_controls ≥ treatment, E0 / E−R   = 188 / 188            ✓
p = (1+188)/201                    = 0.9402985074626866   ✓ both endpoints
```

**Two independent routes to `D(E−R)`** — the formula, and differencing the two
arms' levels — agree to the last digit. That is a stronger check than either
alone, and it is the receipt's own `D_E_MINUS_R_equals_gross_delta_minus_0p20_delta_fe`
predicate confirmed from outside.

---

## 1. §1.5 — reporting: SATISFIED

Required: `D(E0)`, `D(E−R)`, the treatment's location among its own 200 controls
**at each endpoint**, `fe_T`, `fe_B` **and every control's fe**, and materiality;
select nothing.

All present. `controls_D_E0` and `controls_D_E_MINUS_R` carry **200 entries
each**; `fe_cents.controls` carries **200**; `n_received_fills.controls` and
`shares.controls` carry 200 apiece. `control_location` reports both counts, both
p, and the rule inline. `invariant_parts` decomposes INVARIANT into its four
conjuncts rather than asserting the conjunction. `declared_before_run` carries
the bar's own text including the **void condition**, so the receipt states what
would have voided it.

**Nothing is selected.** Every cell is reported.

## 2. §1.6 — computed predicates: SATISFIED

| my requirement | receipt field | value |
|---|---|---|
| `fe_T ≤ fe_B` **computed, not assumed** | `direction_was_computed_not_assumed`, `n_arms_with_fe_GREATER_than_baseline` | `true`, **0** of 201 |
| E0 identity **exactly** on all 202 | `e0_net_equals_gross_exactly_all_arms`, `n_arms_failing_e0_identity` | `true`, **0** |
| every E−R value ≤ 0 | `all_E_MINUS_R_fees_nonpositive` | `true` |
| `D(E0)` **with its sign** | `D_E0_sign` | **−1** |

Also `all_supplied_fees_finite`, `n_arms_with_n_fills_GREATER_than_baseline: 0`,
and `E_MINUS_R_matches_independent_arithmetic_all_arms`.

**Beyond the bar, and it is the right instinct:** `every_arm_fe_at_or_below_its_flat_atm_bound`
/ `n_arms_violating_the_flat_atm_bound: 0` — DE checked the per-fill formula
against the relation `7·p(1−p) ≤ 1.75` that Item 3 of my last round rests on.
That is a structural guard on its own arithmetic, not a restatement.

**My outstanding per-market amendment is satisfied trivially, and I say so rather
than filing it as owed.** `9e5d62f` escalated `fe_T ≤ fe_B` to a **per-market**
predicate. That amendment landed **after** the dispatch, so the receipt could not
have carried it — but it does not need to: `what_this_is_not.n_windows` is **1**
and the selector reports `n_candidates_considered: 1` with a single
`selected_slug`. **With one market, per-market ≡ aggregate.** The amendment binds
any future multi-window run and is a no-op here. **Closed, not owed.**

## 3. §1.7 — falsifiers: SATISFIED IN CODE, AND THIS IS THE ONE GAP

All three I declared are in `de_v2_fee_endpoint_sensitivity.py`, driven both ways:

* **positive control** — `abs(moved − (−0.35·N·s)) < 1e-12` (:635) **plus
  `abs(moved) > 0`** (:639), an anti-vacuity guard: the control must actually move
  something. That second line is the difference between a falsifier and a
  tautology.
* **known-bad `0.07·min(p, 1−p)`** — caught by its signature: it returns 3.5 ¢ at
  the money, **exactly twice** the correct 1.75 (:620–624). The refuted Q5 form,
  caught by the factor that refuted it.
* **wrong-signed rebate** — `gross − priced == gross` for the zero ledger, and
  `gross − wrong_sign != gross` for a `+0.35` (:707–713).
* **shipped guards unweakened** — unknown fill id refuses (:691, the `:494`
  guard), non-finite refuses (:694).
* **and a falsifier for the trap-detector itself**, which I did not require and
  which is exactly right: `_no_gate1_exit` passes a clean nested payload (:719)
  and **refuses** a planted key both nested inside a list (:721) and at top level
  (:725). That is the key-vs-substring distinction driven in both directions.

**Credit where it is due:** :688 records that a line *"originally read
`ok(fixture is None or True, …)`"* — DE found a vacuous assertion in its own
battery and left the note in. Self-disclosed tautologies are rarer than they
should be.

**THE GAP.** I walked every key path in the receipt (208 of them). **There is no
field recording that the falsifiers ran, how many, or that they passed** — no
`checks`, no `selftest`, no `falsifiers`. The battery exists and is 26 checks
(`EXPECTED_CHECKS = 26`), but **a reader of the receipt alone cannot tell any of
that.** Rule 15's whole point is that a result from an instrument which never
proved it can fire is not a result, and **the receipt is what an automated reader
resolves — not the module beside it.** This does not invalidate anything: I read
the battery myself and it is sound. **It means the receipt under-evidences its own
instrument.** Fix in band with a check count and outcome; no re-run, no number
moves.

*(Minor, same family: `EXPECTED_CHECKS = 26` is the hardcoded-tally pattern MEM
flagged at `da_race_withdrawals.py:59`. It catches an accidental deletion and is
hand-updated otherwise. Named, not charged.)*

## 4. §1.8 — the trap and the non-license: SATISFIED

**The trap is closed and I verified it the way it has to be verified — by key
identity, walking all 208 paths.** **No key named `gate1_exit` exists at any
depth.** The only two paths containing that substring are
`/computed_no_gate1_exit_anywhere` (DE's own predicate) and
`/what_this_is_not/why_no_gate1_exit_block` (its explanation) — **which is exactly
the substring/key confusion the coordinator hit and caught on itself.** A
substring test on this receipt fires on the very fields that prove the trap is
closed. Mine was a key-name test; it returns nothing.

The non-license is carried as **fields, not prose**, as §1.8 required:
`gate_1_sampler_refusals_still_stand` is a **list naming all three with their
numbers** (1 of 200 in 4,000; ESS 10.53 vs 100; 16 of 1,000 with 16 distinct
sets); `owned_order_ack_fill_causality` is its own field;
`clears_gate_1: false`, `is_a_validation: false`, `G_complete_utc_days: 0`,
`cluster_unit`, `data_status: CONSUMED`, `n_windows: 1`,
`thresholds_are_on_a_description_not_a_test: true`, and `matched_null` carries the
**code line** (`:381`) for its own absence.

## 5. §1.10 — cap and the pin: SATISFIED, and I re-derived the pin from my own tree

`resource_observation`: wall **24.98 s**, user CPU **25.68 s**, `max_rss_kib`
**344,668** (≈337 MiB) — inside one CPU / 3 GiB / ten minutes, and comparable to
Gate 1e's own 338,556 KiB.

**The pin, verified independently rather than read.** I hashed all **15** paths the
Gate-1e receipt pins against the blobs at `9b37088` **in my own worktree**:

> **14 of 15 match byte-for-byte. The one that differs is
> `plans/HARMFUL_FILL_HAZARD_TOXICITY_PLAN_V2.md`** — the `.md` the guard routes
> to documentary drift, and the receipt records **both** digests for it under
> `documentary_drift_after_gate1d` with `source_code_drift_clear: true` and
> `named_source_code_drift: {}`.

**So the guard was neither defeated, widened nor bypassed** — DE ran at the pin in
a detached snapshot and its stated reason is the right one: re-pinning
prospectively would have put a **second variable** into a run whose entire purpose
is that only the fee term moves.

---

# 6. THE RECONCILIATION — E−R is three values, the receipt carries one, and here is exactly how much that costs

## 6.1 Which of the three DE has, and whether it is decision-bearing

DE's E−R is `−0.20 · 7·p·(1−p) · shares` — **the identity value**, the middle of
the three in `9e5d62f` §1.3. Not the floor (0), not the assumption-free ceiling
(`0.20 · Σ_m P_m`).

**It is NOT the decision-bearing endpoint, and the receipt says so correctly** —
`endpoints.E0.meaning` carries **"DECISION-BEARING"**. E−R is the robustness
endpoint. So the question is only whether the robustness endpoint overstates its
own precision.

## 6.2 It does — in one string, and the overstatement is MINE, not DE's

`endpoints.E_MINUS_R.meaning` reads: *"the per-market share CANCELS (spec 1.2),
so this is **exact and not an interval**."*

**That cites my §1.2 as it stood at dispatch.** `9e5d62f` then established that
the cancellation rests on a four-part identity (A1–A4) and that `0.20 × fe` is a
**point estimate under an identity, not a bound**. DE quoted the bar it was given,
faithfully and with a citation. **This is my overstatement propagating into DE's
receipt, and it is not DE's error.**

## 6.3 The precise answer: the SIGN is unconditional, the WIDTH is not

The coordinator asks whether within-market cancellation makes the single value
correct for the DELTA even though the level has three. **Not quite — and the
distinction is worth stating exactly.** Within-market cancellation is what makes
the identity value **computable**; the identity A1–A4 is what would make it
**exact**. The delta inherits both. Taking INVARIANT's conjuncts one at a time:

| conjunct | value | depends on the rebate's MAGNITUDE? |
|---|---|---|
| `same_sign` | true | **NO — UNCONDITIONAL** |
| `both_p_same_side_of_half` | true (0.9403 both) | yes, in principle |
| `p_shift_within_tolerance` | 0.0000 vs 0.05 | yes |
| `MATERIAL` | 2.7636% vs 10% | **yes, directly** |

**`same_sign` is unconditional** by `9e5d62f` §1.4: `rebate_A,m` is increasing in
that arm's own `fe`, so `fe_T ≤ fe_B` (computed here, 0 of 201 violations, and
n_windows = 1) gives `D(E−R) ≤ D(E0) < 0` **for any rebate magnitude in
`[0, ceiling]`, whatever `P_m` and the other makers are.** No part of A1–A4 is
needed. **The verdict's load-bearing clause does not depend on the exactness the
receipt claims.**

**The magnitude-dependent clauses have measurable headroom.** Materiality is
**2.7636%** against a 10% bar — **the rebate would have to be 3.6184× the identity
value to flip MATERIAL**. And the identity's error directions are not symmetric:

* **A1 false** (fee-equivalent uses the maker's *own signed* rate — we sign zero)
  ⇒ **E−R = 0**, `D(E−R) = D(E0)`, p-shift 0, materiality 0. **Strengthens
  everything.**
* **A3** (mint/maker-maker crossings) ⇒ rebate **below** the identity value.
  **Strengthens.**
* **A2** (Jensen, at 1.17 maker legs per taker leg) and **A4** (22 of 901 taker
  legs over-charged) ⇒ rebate **above** it. **The only threatening direction, and
  it needs a 3.62× understatement.**

I am **not** asserting that A2/A4 cannot reach 3.62× — the instrument that would
settle it is the assumption-free ceiling `0.20 · Σ_m P_m`, decodable on-chain, and
**it is unmeasured**. What I am stating is that the headroom is a computed number
and the threatening direction is one of four.

## 6.4 Ruling: an in-band supersession, and NO re-run

**No number changes and nothing needs re-running.** The correction is to a
**precision claim**, not to a result. A vN+1 receipt should:

1. replace *"exact and not an interval"* with **the identity value**, naming A1–A4;
2. record which INVARIANT conjuncts are **unconditional** (`same_sign`, via the
   monotonicity route) and which are **conditional** (`p_shift`, `MATERIAL`);
3. record the **headroom factor 3.6184×** on MATERIAL;
4. record A1's direction — **if false, E−R = 0 and every conjunct strengthens**;
5. add the §1.7 check count and outcome (§3 above).

**Does the receipt overstate its own precision? Yes, in one field. Does the
overstatement reach the verdict? No** — the verdict's sign clause is
unconditional, and the two magnitude clauses carry 3.62× and an exactly-zero
p-shift. **The USER's ruling is executed either way; what the supersession buys is
that the receipt stops claiming more certainty than the derivation supports.**

---

# 7. DE checked one of my claims and it is a second observation against me

`citation_cross_check` reproduces DA's `shares × 1.75` to 1e-6 on both arms
(3362.72783 and 2318.2081965) and sets **`claim_holds: false`** against my §1.5.1
*"endpoint_worst_case.maker_fee_cents IS numerically fe_arm."* **Measured
`ratio_fe_to_flat_bound` is 0.5863 (baseline) and 0.5992 (treatment)**; measured
`Δfe` is **582.5587** against the flat-derived **1044.5196**.

So my withdrawal in `9e5d62f` §3.3 was right, **and my sizing was conservative in
the direction I predicted**: true materiality **2.7636%** against my quoted
4.955%, and the true interval narrower and still entirely negative.

**One honest calibration of my own estimate.** I sized the overstatement at
**1.473×** from the action-level distribution, flagging that action levels are not
fill prices. The measured factor is **1.706× / 1.669×** per arm and **1.793× on
Δfe**. **Directionally right, quantitatively low** — the caveat I attached was the
operative one, and a reader should weight it accordingly next time I size
something from a proxy distribution.

---

## CONTEXT

Far below the 80% reset threshold; I will report the crossing when it happens.
