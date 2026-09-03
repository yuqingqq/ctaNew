# Review — DE round 41 at `8479b67` (the grouping driven on two real groups; `null_status` from the request; the invariance over the whole block)
reviewer: claude (pm-codex seat) · round opened by the coordinator (pm-co)

**Pinned tip executed: `8479b67`** (row Q-DE-59, same commit) on base **`35452c0`** (my RELEASE at
`c3f8743`). Verified at the blob: runner **`384c64dee08de879`**, **3,455 lines**,
`EXPECTED_CHECKS = 124`; it is the only code file changed. The v2 DRAFT is **not edited**
(`cb693000880c3d94`, 307 lines — the USER's).
**Request of record:** `REQUEST_DE_ROUND_41_2026-09-03.md`. **Composed 2026-09-03T00:56:31Z.** One filing, per R-377.
**Behind the package**: nothing here re-opens the forwarding.

**Constraints observed.** Executed in `~/ctaNew-wt-rev` at `--detach 8479b67` (`data/pm_5min`
mirrored **including `raw/`**); `~/ctaNew-wt-da`, `~/ctaNew-wt-be`, `~/ctaNew-wt-de` never read
(BE 12 and DA 20 are open there); `be_forward_day.py` never run. Five mutants applied to my
worktree copy and **restored byte-identical** — `git status --short` **0** after each. The declared
Phase-4 OUTDIR never passed to `--run`; `derived/` **178 before and after**; nothing written under
`data/`; no unit, timer, scope or anchor; `DA_MIDNIGHT_MODE` never set; `git worktree list`
**34 at quiescence**.

## 1. Counts — CONFIRMED (item 1)

Eight modules, both launchers, PASS = summary = rc 0, zero stderr:
**31 / 124 / 26 / 21 / 26 / 21 / 184 / 92** — R-487 (A) reproduces, **including
`de_admissible_windows` 92**, which needs `raw/` and which DE could not measure in its own
worktree. The row's prior counts (119 / 26) are the `35452c0` blob's, which I measured last round.
`--run --outdir <scratch>` → **rc 2**, no traceback, nothing created; `preflight()` refuses at the
scorer.

## 2. DE40-R1 — **CLOSED**, and the second group is the right one (item 2)

`declaration_groups(declared=None)` (`:507-537`) groups by `(changed_at, file)` and checks each
group at **its own commit and parent**; `declared` is injectable and the docstring says why (the
fixture needs two groups; the run passes the real map).

**Is `46ab455` the right second group? Yes — I verified it at the blobs.** Walking the commits that
touch `harmful_exposure_rows.py` and hashing `label_rows`' AST at each:

| commit | `label_rows` AST sha |
|---|---|
| `851edaf` | `905975dceed925f0` |
| **`46ab455`** | **`905975dceed925f0`** ← the change lands here |
| `f30cf26` (its parent) | **`4a4403ee715d88f7`** |

So `46ab455` is exactly where `label_rows` changed, and the fixture's declared pair
(`905975dceed925f0` / `4a4403ee715d88f7`) is what the blobs carry — a **real** second declaring
commit on the same file, which is the strongest fixture available.

**Is the falsifier the one I asked for? Yes — driven.** A `break` after the first group is now
**red by name**: *"DE40-R1, DRIVEN ON TWO GROUPS: a fixture adding a real second declaring commit
yields **1 groups** — [('46ab455', ['label_rows'])]"* (rc 1, 94 PASS). At `35452c0` the same
mutation survived the whole suite. The second-group known-bad is present too and names the group
alone (`['label_rows@46ab455^']`) while the first stays green.

## 3. DE40-R2 — **CLOSED**, and my item-5 question is answered by the code (item 3)

`run_cell` records `out["n_draws_requested"] = int(draws)`, and `_null_status` (`:804-832`) reads
**what the cell asked for**: 0 → `NO_NULL_REQUESTED`; > 0 with DEGENERATE → `NULL_COLLAPSED`;
> 0 with quantiles → `NULL_SAMPLED`; the field absent → refuse at **`pred#1`**; > 0 with neither →
refuse at **`pred#2`**.

**My question — a cell that requested draws and accepted none — is answered, and the answer is
better than a fourth state.** Such a cell **cannot reach the receipt**: `null#2` refuses the run
when fewer draws are accepted than requested. The tip does not lean on that (which was my finding);
it *computes* the impossibility and refuses at `pred#2` if it ever appears, with the reason in the
message — *"inventing a fourth label for it would report a state nobody computed"*. A cell that
requested draws, accepted some, and collapsed reads `NULL_COLLAPSED`, which is the state the reader
needs. **No fourth state is missing.**

One measured residual, **DE41-R1 (LOW)**: I mutated the guard alone
(`if "n_draws_requested" not in c:` → `if False:`). The suite goes **red** — so the guard-removal
is caught — but it dies with **`KeyError: 'n_draws_requested'` at `:820`**, raised *inside*
`_null_status` while the `pred#1` known-bad is being driven, so the line a maintainer meets is a
traceback rather than a named failure. (The coordinator's variant — the whole round-40 derivation
restored — never indexes the field and does die at the known-bad by name; both mutants are
legitimate and they expose different halves.) One-line closure: have the `refuses(...)` helper
report a wrong exception type by name, or read the field with `.get()` after the guard so its
removal surfaces at the known-bad's own message.

## 4. DE40-R3 — **CLOSED**, and the label is worth a line (item 4)

The invariance (`:1974-2024`) compares **every** field of `null_population` with
`n_accepted_stream_differs` excluded **by name** (`_ORDER_DEPENDENT`), compares the key sets, and
asserts the excluded field **differs** (16 → 20). Two mutants driven:

| mutant | result |
|---|---|
| `n_draws_requested` not recorded | **red at DE40-R3**, through the predicate branch |
| a 22nd order-dependent field added to the block | **red with the assertion unedited** — the block grows to 22 |

**The LOW observation is real and I would file it rather than wave it — DE41-R2.** Under the second
mutant the failure line reads *"…leaves **ALL 21 of the 22** `null_population` fields identical"*:
the count is computed on the **filtered** block before the comparison, so a **red** line asserts the
invariant holds. It is small, but it is a message contradicting its own verdict on the one path
where the message is load-bearing — the same class this programme has closed four times. Round 42,
one line: report the number of fields that **differ**, and name them.

## 5. What the coordinator missed — the class (item 5)

- **Order-dependence outside `null_population`: none.** I compared the **whole cell** (32 fields,
  `null_population` flattened) and the **predicate row** (11 fields) across the DESC/ASC pair. The
  only field that moves is `null_population.n_accepted_stream_differs`. The invariance's scope is
  right, and it is now the block's own scope rather than an enumeration.
- **Can the grouping key merge two declaring commits touching the same function?** No. The key is
  `(changed_at, file)`, so two commits are two groups; a function declared under two commits would
  appear in both and each is checked at its own pair. The reverse risk — an edit landing *after* the
  declaring commit — is not the grouping's job and is caught elsewhere: `pin_statuses` compares the
  **current** file's function AST against the declared tip sha, which I drove to BLOCKING in round
  39.
- **Does any closure rest on a single falsifier?** DE40-R1 has three (the two-group positive, the
  second-group known-bad, the `break` mutant); DE40-R3 has four. **`pred#2` has none that can be
  driven**, and cannot: it names a state `null#2` makes unreachable. That is the honest form — the
  module says so in the refusal itself — and I would not manufacture a fixture to reach it.

## Findings

| id | severity | where | one line |
|---|---|---|---|
| DE41-R1 | LOW | `:815-820` | removing `pred#1`'s guard is caught, but by `KeyError` inside `_null_status` rather than by the known-bad's own name |
| DE41-R2 | LOW | `:1974-2024` | the DE40-R3 failure line reports the invariant as **holding** ("ALL 21 of the 22 … identical") because the count is taken on the filtered block |

**DE40-R1, DE40-R2 and DE40-R3: all CLOSED**, each driven at the tip.

## 6. Round 42 (item 6)

**RELEASE `8479b67` as round 42's base.** All three of my round-40 findings are closed with
falsifiers that fire, the counts reproduce on both launchers including the one DE could not measure,
nothing can run (`preflight()` refuses at the scorer), and the DRAFT is untouched in the USER's
hands.

**Does my DE-40 item-6 answer on ask (5) still hold? Line for line, yes — with one line cheaper.**

- **MECHANICS on the consumed population:** unchanged — the per-cell **computed** `train`/`score`
  composition, its refusal, and its falsifier. Nothing in round 41 added or moved that field.
- **The `score`-split restriction:** unchanged in structure — the tape index must run **before**
  `build_reference` (inverting `preflight → feed → score`), the receipt must carry both the
  declared §3 population and the scored subset, and **§a must travel back to the USER** because it
  declares the frozen population "exactly". One clause is now cheaper: I wrote that the shrunken
  pool would make the `NULL_COLLAPSED` branch load-bearing; at this tip that branch is derived from
  `n_draws_requested` and reported per stratum, so the shrunken-population case **reports itself**
  instead of having to be inferred from the absence of quantiles. The cost of that answer is
  therefore the feed inversion and the re-declaration, not the reporting.

**Round 42's order (independent of the USER):** DE41-R2, then DE41-R1 — both one-liners, both in
failure paths, neither touching what the USER reads.
