# Review — DE round 39 at `cd93663` (the identity admitted and counted; a degenerate null refuses its interval; what "identity" names)
reviewer: claude (pm-codex seat) · round opened by the coordinator (pm-co)

**Pinned tip executed: `cd93663`** (row Q-DE-57, same commit). Verified at the blob: runner
**`2976b46e1eb67a22`** (3,201 lines, `EXPECTED_CHECKS = 115`), `de_score_stream`
**`f85be3354610e2ce`** (420, 26), v2 DRAFT **`6a62569f536e460f`** (290).
**Request of record:** `REQUEST_DE_ROUND_39_2026-09-02.md`. **Composed 2026-09-02T22:55:34Z.** One filing, per R-377.

**Constraints observed.** Executed in `~/ctaNew-wt-rev` at `--detach cd93663` (`data/pm_5min`
mirrored); `~/ctaNew-wt-de` / `~/ctaNew-wt-be` never read; **no file mutated this round** —
`git status --short` **0** throughout. The declared OUTDIR never passed to `--run`; `derived/`
**173 before and after**; nothing written under `data/`; no plan file edited — the v2 DRAFT
included; no unit, timer, scope or anchor; `DA_MIDNIGHT_MODE` never set; `git worktree list`
**34 at quiescence**, unchanged (standing rule 10).

## 1. Counts and the literals — CONFIRMED (item 1)

Eight modules, both launchers, PASS = summary = rc 0, zero stderr:
**31 / 115 / 26 / 21 / 26 / 21 / 184 / 92** — R-482 reproduces, and the row's prior counts
(101 / 25) are the `dfd4c00` blob's, which I measured myself last round.

**The six literals now carry `changed_at: "851edaf"` and I verified them at both sides of that
commit with my own `_fn_asts`:**

| function | declared fit / tip | mine at `851edaf^` / `851edaf` | match |
|---|---|---|---|
| `select_v2_era` | `e97a6662273d8abc` / `3b34bdc86b1056ca` | same | **yes** |
| `_era_or_refuse` | `None` / `830c4fa88ba44280` | same | **yes** |
| `_refuse_empty_selection` | `None` / `a6cfb900e1ced0b8` | same | **yes** |

The fit-side values also equal what I computed at `e12e2c7` last round, so the two references agree
for these functions. `--run --outdir <scratch>` → **rc 2**, nothing created; `preflight()` refuses
in **1.33 s**.

## 2. My rulings (1)–(3) — CONFIRMED CLOSED, driven (item 2)

The guard call is gone from `run_cell` and the parse certificate is **inverted** (`:2016-2020`
asserts **zero** calls) — the check that certified a call which could not go red is now the check
that it is absent. Driven by me on both fixtures:

| | C1 fixture | free fixture (`:1774-1784`) |
|---|---|---|
| accepted | 5 | **20** |
| distinct accepted / attempted | **1 / 3** | **6 / 6** |
| identity (whole draw) | 5 | **4** |
| `null` | **`DEGENERATE (n_distinct_accepted = 1)`** | sampled |
| `null_quantiles` / `net_diff…` | **absent / absent** | q50 **16.0** / net_diff **24.0** |
| `point_estimate_cents` | **40.0**, labelled | — |
| `accepted_by_stratum` | `{"BUY_UP\|13": {size 5, n_distinct 1, n_accepted_identity 5, collapsed true}}` | — |
| predicate row | `interval POINT_ESTIMATE_NO_INTERVAL`, `beats_null_q95 None` | — |

So **ruling (1)** (admit and count the identity; retire the guard) **CLOSED**; **ruling (2)**
(statistics on the accepted set; one distinct accepted ⇒ refuse the interval, not a declared point
mass) **CLOSED**; **ruling (4)**'s per-stratum block is in the receipt **before** any quantile,
**CLOSED**. **Ruling (3)** is closed in substance — the check now asks what was accepted — but its
predicate is the one DE39-C1 is about; see §5.

## 3. DE38-R1 + C3, C2, ruling (4) — CONFIRMED CLOSED (item 3)

The pool is the **stream's support** (`:1182-1202`, with `null#4` refusing duplicate keys) and
`_strata`/`_dem`/`_room` are computed on it (`:1235-1243`). Measured on the free fixture whose
reference carries a fifth generation the stream does not: **`pool_size` 4** (not 5),
**`strata_with_room` 1**, the extra key never drawn — so the freedom statistic no longer counts
freedom the draw cannot use (**DE38-R1 closed**, and C3 with it).

`null#2` (`:1398-1409`) carries the reasons: driven at `DRAW_ATTEMPT_BUDGET = 1` on the C1
fixture — *"only 3 of 20 draws matched … in 20 attempts (17 rejected: {'P4': 17})"* — **C2 closed**.
Ruling (4)'s matching rule is untouched, as I ruled it should be.

## 4. DE38-R3 / R4 / R2 — each in the form I asked (item 4)

- **R3 CLOSED.** `REQUIRED_EVENT_KEYS = ('t', 'slug', 'side', 'gen')` is **one source**
  (`de_score_stream:46`), checked at `:162` and used to build the event at `:184`. That is the
  single-source form I asked for, so the round-38 mutant now removes the key from the output and
  the runner refuses **by name** at `null#3` instead of dying `KeyError`.
- **R4 CLOSED, driven.** A cell built with `_draw_log` carries
  `produced_under_falsifier_input: ['_draw_log']` (`:1139-1142`) and `validate_receipt` **refuses**
  at `receipt#4`: *"cell(s) [0] were produced under a FALSIFIER INPUT ({0: ['_draw_log']})"*. It
  names **which** flag, which is more than I asked for and is the right form.
- **R2 CLOSED by addition.** `de_score_stream:360-383` parses its own source and asserts the
  **event path** (`score_events`, `lift`, `coin_of`, `validate_scores`) contains no
  `open`/`read_text`/`read_bytes`/`load…` call, naming the two legitimate file readers. The
  docstring check (`:354`) remains **beside** it rather than instead of it — the behaviour is now
  asserted and the declaration is still tested, which is the stronger of the two readings I offered.
- **§2(iii) CLOSED.** Each declaration names `changed_at` and the check (`:2485-2500`) reads
  **both sides** of that commit — the tip shas at it, the fit shas at its parent. My round-38 ask
  (tie the reason to a commit) is met. One residual, **DE39-R2**, in §7.

## 5. DE39-C1 — **CONFIRMED**, and the three rulings (item 5)

**Reproduced.** On the free fixture, with the two above values swapped in time (0.8 then 0.9):

| | as shipped (descending) | swapped |
|---|---|---|
| accepted / distinct | 20 / 6 | 20 / 6 |
| identity (SET) | **4** | **4** |
| control-stream ≠ treated-stream | **16 of 20** | **20 of 20** |
| accepted values | `{16.0: 13, −8.0: 3, 40.0: 4}` | **identical** |
| q50 / net_diff | 16.0 / 24.0 | **identical** |

A decision-inert reordering changes the stream-map count and **nothing else**. The reason is the
policy's: with `enable_reduce = False` the score is read only against `theta_cancel` /
`theta_repost` / `theta_reduce` (`harmful_stateful_policy.py:892-940`), so *which* above value
lands on *which* above-carrying generation cannot change a decision — only the **set** can.

**RULING (i) — the identity is SET identity; name it so in the source, and carry the caveat in the
DRAFT.** The DRAFT's definition is right and the counters are right; the source comment
(`:1387-1388`) should say "the set of above-carrying generations is the treated arm's", not "the
control's stream is then the treated arm's, exactly", which is false whenever the above values do
not descend in time. **And yes, the `enable_reduce` caveat belongs in the DRAFT's definition** —
one clause: *with a reduce band enabled the score's magnitude enters the decision and the identity
must be re-read as a stream property*. This programme has been bitten twice by a correctness rule
that rested on a configuration that happened to hold (DA10-R5's class); one sentence prevents the
third time.

**RULING (ii) — ruling (3) is asserted on the logged accepted VALUES.** The property "the accepted
set is a null" means *at least one accepted draw whose value differs from the treated arm's* —
measured, 16 of 20 in **both** orderings, so the assertion is invariant under the reordering and
says what it means. **Keep the stream-map count as a second, labelled statistic**
(`n_accepted_stream_differs`), because it is informative about the permutation's reach — but it
must not be the assertion, and its label must say it counts stream maps, not decisions.

**RULING (iii) — log `drawn`, and prefer deleting the recomputation.** `_differs` re-derives the
draws with `MRC.draw(_fpool, …)` where `_fpool` is in `_fsc` order while the runner's pool is
sorted (`:1202`), and the log carries no `drawn` to check it against: coincident here, unasserted
in general. Under (ii) the recomputation is unnecessary — the values are already in the log — so
**delete it**, and log `drawn` anyway, because a null's audit trail should name what was drawn. If
DE keeps a recomputation for the second statistic, it must assert
`set(recomputed) == set(logged["drawn"])` first.

**Severity: LOW, as filed.** No number a reader would act on moves; what moves is what a check
proves.

## 6. My conditions (i)–(iv) at `cd93663` (item 6)

| condition | verdict |
|---|---|
| (i) §5 describes a null that can differ, and what happens when it cannot | **MET.** §5 states the identity's admission, the guard's retirement with my measured 0/200 vs 65/200, and the degeneracy remedy as reporting; the code delivers both branches — measured, the free fixture samples (q50 16.0 ≠ treated 40.0) and the C1 fixture publishes DEGENERATE with no quantiles |
| (ii) below values stated, §2 re-read | **MET**, unmoved by this round's edits |
| (iii) §6 states the seal's form as a pin claim | **MET, and strengthened** — the claim now names `changed_at` and is checked at both sides of it |
| (iv) the split question is ruling 5, asked with 2 and 4 | **MET**, unmoved |

**May the package travel whole? Yes — DE39-C1 does not hold it.** It is a naming and an assertion
form; no number the USER would read changes under it, which I measured rather than assumed. I would
send it with §5's definition carrying the `enable_reduce` clause of ruling (i), because that
sentence is about what the USER is adopting; rulings (ii) and (iii) are the suite's business and can
land in round 40 behind the package.

## 7. What the coordinator missed — the class (item 7)

- **A predicate satisfied by a decision-inert change:** item 5 is the instance; the parse scan finds
  **no others** — the runner is clean of unfailable predicates, docstring predicates, self-greps and
  substring checks, and `de_score_stream:354`'s docstring test now sits beside a behavioural
  assertion (§4).
- **A statistic whose population is named in prose but not in the key:** none left in the null block
  — `n_distinct_accepted` / `n_distinct_attempted` / `point_mass_accepted` all carry theirs.
- **Where the DEGENERATE fallback leaves a reader:** the predicate row is correct
  (`POINT_ESTIMATE_NO_INTERVAL`, `beats_null_q95 None`) — but it carries **no `null` field**, so a
  cell whose null **collapsed** and a cell that **never ran a null** read identically in the
  predicate table, and only the second is uninformative about the policy. Filed **DE39-R1**.
- **`len(_ca) == 1`:** a trap in waiting rather than an intended cost — it encodes today's
  arrangement (one `changed_at`) into a correctness check, so the first genuine second declaration
  reads as a pin failure rather than a rework request. Group by `changed_at` and check each group
  at its own commit. Filed **DE39-R2**.

## Findings

| id | severity | where | one line |
|---|---|---|---|
| DE39-R1 | LOW-MEDIUM | `:791`, the predicate row | a collapsed null and a cell that never ran one read identically in the predicate table — no `null_status` field |
| DE39-R2 | LOW | `:2492` | `len(_ca) == 1` encodes today's single `changed_at`; group by it instead |

**DE39-C1 CONFIRMED** (reproduced, and characterised: set-identity is the decision variable at one
θ with `enable_reduce` False). **DE38-C1, C2, C3 and DE38-R1, R2, R3, R4, and §2(iii): all
CONFIRMED CLOSED**, each driven above.

## Disposition and round 40's order

**RELEASE `cd93663` as round 40's base.** Every step of the six-step order I set last round is
built and driven: the identity is admitted and counted, the guard is retired and its certificate
inverted, the null's statistics are on the accepted set with a DEGENERATE refusal of the interval,
the per-stratum block precedes any quantile, the pool is the stream's support, `null#2` carries its
reasons, the event contract is one source, the falsifier flag is recorded and refused at the
receipt, and the declaration names the commit that changed it. Nothing can run — `preflight()`
refuses in 1.33 s — so no finding here can reach an artifact.

**Round 40, in this order:**
1. **DE39-C1 ruling (i)** — the source comment and the DRAFT's `enable_reduce` clause. This is the
   only item that touches what the USER reads.
2. **Rulings (ii) and (iii)** — assert on the logged values, keep the stream-map count as a labelled
   statistic, log `drawn` and drop the recomputation.
3. **DE39-R1** — `null_status` in the predicate row.
4. **DE39-R2** — group the declaration check by `changed_at`.
5. The wiring's expensive half, whenever the USER's ruling on the package makes it worth paying for.
