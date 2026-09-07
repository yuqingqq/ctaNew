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
- **v20 must therefore carry** the verification source by pair once BE 99 lands it,
  or state in the declaration that the endpoint is provisional on the venue record.
  CLAUDE.md rule 9's parenthetical (settlement = Chainlink) is asserted by the
  market text, not by this file, and this declaration must not restate it as fact.

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
