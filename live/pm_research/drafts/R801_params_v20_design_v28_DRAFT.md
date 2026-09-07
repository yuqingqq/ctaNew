# DRAFT — params v20 / design v28 under R-801. **NOT A DECLARATION.**

**This file is a DRAFT and is not resolvable by any chain.** It is `.md`, it is not
named `<family>_v<N>.json`, and it does not live in the declarations directory —
so `declaration_chain.resolve_head` cannot see it and no run can pin it. It exists
because DE 136 was told to draft the text and NOT to land the versions: the R-775
freeze holds until GO #8's receipt lands, and REV 104 reads this beside the code.

Written 2026-09-07 by DE (DE 136). Every digest below is a placeholder marked
`<measure at write time>`: R-761 requires the CAS write to re-read the head, and
REV 96 §5 requires the cascade pins to be RE-MEASURED, not copied from here.

---

## 1. What changes, and what does not

**The USER, 2026-09-07 (R-801), verbatim:** *"the pnls are from trades and remaining
position's settlement p&l, need to calculate this correctly"*.

The **estimand** changes. The **test does not**: `per_day_location` and
`per_day_standardised_excess` are the design's own and are reused unchanged, so the
two endpoints stay comparable and no second statistic enters by the back door.

| | before | after |
|---|---|---|
| PRIMARY endpoint | `D_E0` — the 5-second markout excess | **`D_E_settle`** — the ruled P&L excess |
| `D_E0` | the result | a **DIAGNOSTIC** of short-horizon adverse selection, emitted beside the primary, never quoted as the result |
| the null | 500 matched random-cancel draws, valued at 5 s | the **SAME** 500 draws, valued **both** ways, inline at draw time |

## 2. The primary endpoint, named and defined

**`D_E_settle` = (arm path's ruled P&L) − (0-cancel baseline's ruled P&L)**, cents.

A path's ruled P&L on a slug:

```
trades leg   = sum(sells px x size) - sum(buys px x size)
residual leg = net_shares x settle          settle = 100c iff Up won, else 0
total        = trades leg + residual leg
             = sum over fills of  sgn x (settle - px) x size
```

and the day's P&L is the sum over slugs. The last line is an algebraic identity,
so the code ASSERTS it (`SETTLEMENT_LEGS_DO_NOT_RECONCILE` at 1e-9) rather than
trusting it: the legs are a decomposition of the ruled quantity, not a second
definition of it.

**Emitted:** per arm-day `economic_settlement` (D_E_settle, Z, p_location,
null_mean, null_sd, n draws, the arm's and the baseline's own totals, both legs);
per slug, in the decision ledger, rows `SETTLEMENT_SCALARS` and `SETTLEMENT_SLUG`
for BOTH books — so a reader holding the ledger and no receipt re-forms the P&L
slug by slug without re-running the day.

## 3. The winner source, and what is NOT yet verified

- **Source of record today:** the venue's `data/pm_5min/resolutions.jsonl` — the
  `winners` dict on records with `closed: true`; the UP share pays 100c iff
  `winners["Up"] is True`. The method is **BE 98's** (Q-BE-341), not this seat's.
- **The receipt carries** the source's path, sha256, slug count, the method in
  words, and a per-slug verification status.
- **What is NOT established:** this file's own `source` is `"clob"`. The market
  text names Chainlink as the resolution source; **BE 98 states it verified no
  winner against a Chainlink read, and BE 99 owns that method** (R-801 (4)). Until
  a verification source is supplied, every value carries
  `chainlink_verification.status = NOT_VERIFIED_AGAINST_CHAINLINK` and
  `is_final_for_quotation = false`, and a caller that asks for a quotable value is
  REFUSED (`SETTLEMENT_WINNERS_NOT_VERIFIED`).
- **BE 99 HAS LANDED (R-803) and the verification is wired** — see §3a. The
  endpoint is no longer provisional on the venue record alone: it is the venue's
  winner CHECKED per slug against the Chainlink stream, and a day that fails the
  check is not quotable.
  CLAUDE.md rule 9's parenthetical is now NARROWER, not resolved: on 09-05 and
  09-06 one form reproduces every recorded winner (576/576, reproduced twice —
  BE 99 and DE 137). Whether it does so on every day is a PER-DAY CHECK, which
  is why the check runs per day and refuses rather than being asserted once.

## 3a. The winner is VERIFIED, and the convention is pinned (R-803, BE 99)

**Updated by DE 137.** BE 99 (Q-BE-342 `9e1d943`) measured the pre-registered
grid in `exp_m6_settlement.py` against the venue's recorded winner:

| convention | slugs disagreeing, 09-05 / 09-06 |
|---|---|
| **S60(T) ≥ S60(t0)** | **0 / 0 — 288 of 288 on both days** |
| S30(T) vs S30(t0) | 23 / 17 |
| S60(T) vs S30(t0) | 15 / 10 |
| mean S60[t0,T] vs S60(t0) | 44 / 43 |

Flipping a convention's disagreeing slugs moves a day total by as much as
−42,454 c. **So the winner is NOT convention-free at the slug level**, and v20
pins exactly one:

```
name             S60(T) >= S60(t0)
X_T              the 60-second Chainlink TWAP at the window's close
X_0              the same stream at the window's open
boundary reader  last sample at or before the boundary
tie              X_T >= X_0 -> Up
readers          exp_m6_settlement.load_streams / read_at, imported, not re-implemented
provenance       BE 99, Q-BE-342, R-803
```

The venue record is the **join**; the stream is the **check** (rule 9's door).
Per slug the receipt carries `VERIFIED_AGREE` / `DISAGREE` /
`CHAINLINK_UNAVAILABLE` / `VENUE_UNRESOLVED`. A day with any DISAGREE or
UNAVAILABLE slug may still be COMPUTED and LABELLED, and it is REFUSED BY NAME
on the quotable path (`SETTLEMENT_WINNER_DISAGREES_WITH_CHAINLINK`,
`SETTLEMENT_CHAINLINK_UNAVAILABLE`). `is_final_for_quotation` is true only when
every slug agrees. **v20 records the convention block verbatim** so a later
reader resolves the rule and not a sentence.

## 3b. Placement latency — a DECLARED parameter of the reference (R-803, BE 100)

**BE 100 (Q-BE-343 `a8d0ad4`) established that the reference has no placement
latency at all.** `_qr_spec(QR_SKEW, latency_ms=0, cancel=False)` binds
`latency_ms` to `cancel_latency_ms` with cancelling disabled; `grep -niE latency
policy_optimizer_queue_realistic.py` returns only `cancel_latency_ms`. **Quotes
were placed instantaneously BY OMISSION, not by a declared choice.** And it is
not a small omission: 48–56 % of every path's fills land within 250 ms of their
generation's start, carrying **98 %** of the baseline's settlement P&L on 09-05
(79,264 of 81,238 c) and **55 %** on 09-06 (25,812 of 46,562).

`build_reference` now takes `placement_latency_ms`, with the semantics of the
cancel latency mirrored: *a generation's quote is not resting until t0 + L_place;
a fill before that is not ours* — dropped and COUNTED under
`TRANCHE_BEFORE_PLACEMENT_LATENCY`, never silently either way.

**PROPOSED for v20: `placement_latency_ms = 250`** — the arms' own cancel
latency. **The code's default is 0.0 and stays there until v20 lands**, because
changing the reference's fills under the freeze would move every future day
silently, which is the choosing-after-seeing the declaration exists to prevent.
The parameter and the value it ran with are recorded in every reference's own
output.

Two consequences v20 must state: the four consumed days may be RE-VALUED under a
non-zero L_place as **DESIGN data** by a later GO (the landed artifacts are never
edited, rule 13); and the predicate is `>= L` on floats — a fill at t0 + 100 ms
measures 99.99999999999964 ms, so **no declared L may sit on a value a fill lands
exactly on**.

## 4. The admissible days — rule 11, in the declaration

- **09-03 … 09-06 are DESIGN data.** The USER's early read (R-754) consumed them;
  they may be valued under this endpoint and every such value is LABELLED
  `DESIGN_DATA`. They may not validate it.
- **09-07 and later are the validation population.** `settlement_endpoint.admissible_days`
  in v20 is the list; the estimator REFUSES `SETTLEMENT_DAY_NOT_ADMISSIBLE` for any
  day that is neither a design day nor in that list, and the set is READ from the
  declaration — never typed in the code.
- **Nobody values 09-07 before v20/v28 land.** GO #8 runs tonight under the current
  bytes; its ledger is the input the new estimator reads afterwards.
- **Multiplicity (rule 12):** two endpoints now exist on one race. v20 records the
  count of endpoints in the forward race at freeze time; the primary is
  `D_E_settle` and `D_E0` is not a second bite.

## 5. The null, and its memory budget

The SAME draws, same seed, valued twice — never a second null (a separate loop
would be "a different null wearing the same seed", which the existing cross-check
against BE's `draw_null` exists to forbid).

Design v27's R11 budget is why `draw_null` discards each draw's fills. The second
valuation therefore runs **inline at draw time** and keeps **one float per draw**:
measured at 500 draws, **0 fills retained** and a scalar list of a few hundred
bytes, with `rss_mb_before_the_draws` and `peak_rss_mb_during_draws` recorded in
`second_valuation_residency`. **v27's R11 text needs one sentence added:** the
draws are valued under BOTH declared endpoints before being dropped.

## 6. The pins, to be RE-MEASURED (REV 96 §5, R-761)

Not copied from here. At write time:

- `de_phase4_diag_runner.py` — the cascade pin moved twice since v19 (DE 129, DE 130)
  and again with DE 134/136; **re-measure both axes** (draw-path functions changed,
  top-level defs, module-level assignments) against the then-current digest and
  record `changed_by_commit`.
- the ten `be_cascade` entries — re-derive from the import closure, assert each
  equals its HEAD blob before writing.
- `de_multiday_gate1_runner.py`, `de_decision_ledger.py`, `de_early_read.py` — the
  bytes that carry this endpoint.
- The CAS: re-read the head AFTER the copy and before the commit; a landing whose
  diff shows `M` on an existing version is a refusal.

## 7. What this draft does not decide

- The **inventory-leg ruling of R-795** is answered for this endpoint (the residual
  IS in the P&L, marked at settlement) and remains open for any other use.
- **Position caps** and the quoter's placement (R-800 (iii)) are untouched.
- **Fees** are not folded in: maker fee zero is the E0 convention; the rewards
  registry exists and stays out unless the user rules otherwise.
- Whether the four design days' values are ever published as a table is the
  coordinator's and the user's, not this declaration's.
