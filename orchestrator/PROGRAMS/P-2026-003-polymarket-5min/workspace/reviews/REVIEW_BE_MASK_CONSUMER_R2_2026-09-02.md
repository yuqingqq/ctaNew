# Review — BE mask-consumer batch, round 2 (RR8 closures + CO-1)
reviewer: claude (pm-codex seat) · round opened by the coordinator (pm-co)

**Pinned tip executed: `d337b73`** (commits `791b53b`, `c200262`, `d151afa`, `d337b73`;
one file: `live/pm_research/harmful_forward_scorer.py`).
**Request of record:** `REQUEST_BE_MASK_CONSUMER_R2_2026-09-02.md`.
**Composed 2026-09-02T10:32:17Z.** One filing, per R-377.
`da_midnight_verify.sh` was **not** run this round, per the dispatch.

Executed in `~/ctaNew-wt-rev` at `--detach d337b73`; filing lands in the shared tree
under R-387's pathspec discipline.

**Inputs I read, with their identity — and one changed mid-review.** At **10:27:30Z**:
`da_blackout_mask_20260901.json` sha `61ae2782d8a3dcfa`; `da_dayverdict_20260901.json`
sha `d071030deb8328f0` (as-of 10:16:16Z); `da_dayverdict_20260902.json` sha
`71001e120e3a6974` (as-of 10:16:17Z); `da_blackout_mask_20260902.json` **absent**.
At **10:29:25Z DA's round-7 restoration landed** and both verdicts became sha
`c087d507fe433210` / `09a14a7392abe224`, **as-of 00:06:01Z / 00:06:03Z** — the original
bytes, recovered. Every conclusion below is reported against the reading it came from,
and where the answer moved with the input I say so rather than calling it a finding.

---

## Verdict

### RELEASED. RR8-1, RR8-2, RR8-3 and CO-1 are closed, verified against the real committed artifact and both launchers.

One finding, low: **RR10-1** — one control is anchored to a live artifact rather than a
fixture, and today's two artifact changes made it insensitive.

---

## 1. RR8-1 — the producer's artifact IS the contract

Loading the **real committed** `da_blackout_mask_20260901.json` through
`load_blackout_mask`:

- **141 windows**, equal to the artifact's own `total_masked_windows`;
- per-coin counts **asserted against the artifact's own `n_masked` values**, not
  literals: bnb 22, btc 23, doge 22, eth 22, hype 9, sol 23, xrp 20 — all agree.

Five drift controls, each refusing **by name**:

| drift | result |
|---|---|
| `artifact` renamed | REFUSED — *"declares artifact 'something_else', not 'da_blackout_mask_v1'"* |
| `coins` → `per_coin` | REFUSED — *"carries no `coins` block"* |
| `n_masked` ≠ `len(list)` | REFUSED — *"says n_masked=999 with 23 windows listed"* |
| `total_masked_windows` off by one | REFUSED — *"declares 142 while its own per-coin lists hold 141"* |
| wrong day | REFUSED — *"is for day '20260831', not '20260901'"* |

**The question the request asked about DA's round 7: the adapter accepts the extended
envelope without a change.** I added `carrying_commit` and a `module_sha256_prefix`
inside `detector` to the real artifact and re-loaded: **ACCEPTED**. The adapter asserts
the keys it needs and ignores additive ones, so round 7's provenance fields will land
without touching this file.

## 2. The trigger, as amended by R-411 and R-412

`EFFECTIVE_FROM_DAY` is **imported** from the frozen rule — `S.EFFECTIVE_FROM_DAY ==
CLR.EFFECTIVE_FROM_DAY == 20260902`, not restated.

| case | result |
|---|---|
| 09-01 pre-governed, UNRESOLVED, no mask | **SCORED** — basis states neither trigger fired |
| 09-02 governed, no mask, liveness UNRESOLVED | **REFUSED** — *"is GOVERNED … REQUIRES a mask artifact, EMPTY PERMITTED (R-410): absence means the producer did not run, never that nothing was thin"* |
| 09-02 governed, no mask, liveness **LIVE** | **REFUSED** — governance requires regardless of the reading |
| governed + mask **present** + UNRESOLVED | **REFUSED** — *"This is TEMPORARY: retry when the verdict lands. No disposition is implied and none is decided here."* |
| governed + mask present + LIVE / THIN | **SCORED** — *"a mask is PRESENT and is CONSUMED for any day (R-411)"* |
| UNJUDGEABLE, **either** day | **REFUSED** — *"which is PERMANENT — no later data makes it judgeable"*, carrying `routed_to: "frozen rule §7 — coordinator exclusion with a stated reason"` as text, deciding nothing |

That is R-411's *presence consumes, governance requires* and R-412's *not-yet vs
cannot* implemented exactly, including the retry remedy on the temporary branch and the
rule-14 discipline on the permanent one.

**The mutant that collapses the two statuses** (removing the UNJUDGEABLE branch so it
falls through to the UNRESOLVED handling): **KILLED** — *"UNJUDGEABLE must REFUSE"*.

## 3. RR8-3 — `day_closed_calendar`, both directions

Against the real 09-01 mask with the flag flipped: `True` → **SCORED**; `False` →
**REFUSED by name** (*"has day_closed_calendar=False. A PARTIAL mask lists only the
windows…"*). Both directions, on the committed artifact rather than a fixture.

## 4. CO-1 — the launcher gap

| check | result |
|---|---|
| `python3 -m live.pm_research.harmful_forward_scorer --selftest` | rc 0, **60 checks** |
| script-directory launch | rc 0, **60 checks** |
| repo-root path launch | rc 0, **60 checks** |
| **both** import mechanisms disabled | **`GoverningRuleUnreadable`** raised at the point of use — *"the governing day could not be read from `da_content_liveness_rule.EFFECTIVE_FROM_DAY`"*, never a silent `governed=False` |
| the launch-invariance check made to contribute 0 checks | **KILLED** — *"contributed NO checks, so it did not run. Green under one launcher is not green"* |

BE's two accepted survivors are accepted for the right reason: either mechanism alone
suffices, so neither is individually load-bearing; **removing both is killed**, which is
the mutation that matters. The coverage guard means the launch check cannot be deleted
without the suite noticing — the defect class that produced CO-1 in the first place.

## 5. BE's two self-found defective controls

**(a) The dual-module refusal control — fixed structurally.** It now mutates
`globals()` of the **running** module, with the hazard named in place: *"Under
`__main__` that import creates a SECOND module object, so setting its attribute leaves
the running one untouched and the control passes while testing nothing."* Using
`globals()` makes the right object a property of the code rather than of the launcher,
so this cannot regress by being run differently.

**(b) The pre-governed control — see RR10-1.**

### RR10-1 — LOW — a control anchored to a live artifact stopped discriminating today

The control scores **08-27** and asserts rc 0, with the comment that *"09-01 no longer
serves here — DA's rule block has LANDED and 09-01 now reads CONTENT_THIN."* That was
true when BE wrote it. **It is not true now:** DA's 10:29Z restoration returned 09-01 to
its 00:06Z bytes, which predate the R-402 wiring and carry **no** `content_liveness_rule`
block, so 09-01 reads UNRESOLVED and is pre-governed.

**Executed:** swapping the control's day back to `20260901` leaves the suite **60/60
green**. The control is not wrong — both days satisfy *pre-governed, no thin signal, no
mask → scores* — but it no longer tests the distinction it was rewritten for, and its
verdict depends on bytes outside the repo that changed **twice today**.

**Closure:** anchor it to a fixture verdict written into the tmpdir (as the neighbouring
controls already do) so it asserts the behaviour regardless of what DA's artifacts say
this hour. One fixture, no new semantics.

## 6. The live case, reported against both readings

| reading | 09-02 liveness | mask | `score_day` |
|---|---|---|---|
| 10:27Z, verdict sha `71001e12…` (as-of 10:16Z) | `CONTENT_LIVENESS_UNRESOLVED` (legacy key) | absent | **REFUSED** — governed, mask absent |
| 10:31Z, verdict sha `09a14a73…` (as-of **00:06Z**, restored) | `CONTENT_LIVENESS_UNRESOLVED` (legacy key; no rule block in these bytes) | absent | **REFUSED** — governed, mask absent |

BE's stated premise for this case — *"09-02 … reads CONTENT_THIN"* — does **not** hold
against the restored bytes, and the control handles that correctly: its `else` branch
asserts the scorer responded consistently with what is on disk rather than failing on a
changed input.

**The substantive point is that the day refuses either way, and for the stronger
reason.** Under the old trigger 09-02 scored whole because liveness read UNRESOLVED;
under R-410/R-412 it refuses on **governance**, which does not depend on the liveness
read at all. The escalation BE filed last round is closed by construction, and I
verified it on the artifact as it actually stands rather than as the filing describes
it.

---

## Executed evidence

At `d337b73`, 2026-09-02T10:27–10:32Z:

| check | result |
|---|---|
| three launch modes | rc 0, **60 checks** each |
| real 09-01 mask through the adapter | **141 windows**, per-coin counts match the artifact's own `n_masked` |
| five envelope/consistency drifts | **all REFUSED by name** |
| round-7 additive keys | **ACCEPTED** — no adapter change needed |
| `EFFECTIVE_FROM_DAY` | **imported** from the frozen rule, equal to it |
| trigger matrix (6 cases) | as ruled — presence consumes, governance requires, temporary vs permanent distinguished |
| status-collapse mutant | **KILLED** |
| `day_closed_calendar` True / False | **SCORED / REFUSED by name** |
| both import mechanisms removed | **`GoverningRuleUnreadable`** |
| launch check contributing 0 checks | **KILLED** |
| dual-module control | fixed via `globals()` of the running module |
| 08-27 control's day swapped to 09-01 | **survives** — RR10-1 |
| live 09-02, both readings | **REFUSED** on governance, both times |
| mutants executed | **11 — 10 behaved as designed, 1 survivor (RR10-1)** |
| worktree after the review | clean |

---

## Disposition

- **RELEASED:** RR8-1 (the producer's envelope is the contract, and the adapter reads
  the real committed artifact), RR8-2 (temporary and permanent are distinguished, with
  the remedy on one and `routed_to` on the other), RR8-3 (`day_closed_calendar` both
  directions), CO-1 (both launchers, the unreadable-rule refusal, and a coverage guard
  on the launch check itself). **No hold from this seat.**
- **FILED, not holding:** RR10-1 — anchor the pre-governed control to a fixture; today's
  artifact churn made it insensitive.
- **Recorded:** DA's round-7 restoration landed **mid-review** and returned both
  verdicts to their 00:06Z bytes (as-of 00:06:01Z / 00:06:03Z). My own incident of the
  previous round is therefore fully undone at the artifacts, and I confirmed it by
  reading them rather than by accepting the register. The one place it still shows is
  RR10-1, where a control's premise moved with those bytes — which is itself the
  argument for fixtures.
