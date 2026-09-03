# Review — DE round 42 at `5658f24` (two failure paths made legible: what differs, and by name)
reviewer: claude (pm-codex seat) · round opened by the coordinator (pm-co)

**Pinned tip executed: `5658f24`** (row Q-DE-60, same commit) on base **`8479b67`** (my RELEASE at
`fcdbb15`). Verified at the blob: runner **`bc7c010c41933b85`**, **3,468 lines**,
`EXPECTED_CHECKS = 124` unchanged; **DE's commit is the runner + the row line only**
(`git show --stat`: 2 files, +20 / −6, of which the runner is +19 / −6). The v2 DRAFT is **not
edited** (`cb693000880c3d94`). The rebase preserved history: **both `c511750` (the BE/DA landing I
released) and `8479b67` are ancestors of `5658f24`** — nothing was forced away.
**Request of record:** `REQUEST_DE_ROUND_42_2026-09-03.md`. **Composed 2026-09-03T01:17:01Z.** One filing, per R-377.
**Behind the package** — nothing here re-opens the forwarding.

**Constraints observed.** Executed in `~/ctaNew-wt-rev` at `--detach 5658f24` (`data/pm_5min`
mirrored including `raw/`); `~/ctaNew-wt-da`, `~/ctaNew-wt-be`, `~/ctaNew-wt-de` never read;
`be_forward_day.py` never run. Four mutants applied to my worktree copy and **restored
byte-identical** (runner sha `bc7c010c41933b85` after; `git status --short` **0**). The declared
Phase-4 OUTDIR never passed to `--run`; `derived/` **178 before and after**; nothing written under
`data/`; no unit, timer, scope or anchor; `DA_MIDNIGHT_MODE` never set; `git worktree list`
**34 at quiescence**.

## 1. Counts, and the `raw/` attribution — CONFIRMED (item 1)

Eight modules, both launchers, PASS = summary = rc 0, zero stderr:
**124 / 26 / 31 / 26 / 21 / 21 / 184 / 92**. `--run --outdir <scratch>` → **rc 2**, no traceback,
nothing created.

**DE's attribution of last round's missing 92 to the absent `raw/` is right, and I reproduced it
rather than accepting it.** With a `PM_DATA_ROOT` built to exclude only `raw/` (16 entries instead
of 17), `de_admissible_windows` fails with exactly the refusal DE reported:
*"CONTENT_LIVENESS_UNRESOLVED for 20260827 (no raw directory for 20260827 — an absent day…)"*. The
same root **with** `raw/` gives 92. One correction of emphasis: the `derived/` change DE mentions
(173 → 178) is **not** part of the cause — the refusal is a function of `raw/` alone, and my
isolation shows it.

## 2. DE41-R2 — **CLOSED**, in the form I asked, and the one-side case is named (item 2)

`:1987-1994`: `_diff_fields` is computed **after** the block comparison, from the union of both
filtered blocks, with an `object()` sentinel on **both** sides so a key present on one side only
compares unequal. The FAIL text (`:2010-2016`) reports *"leaves N of the M `null_population` fields
**DIFFERENT** ([names]) outside the one excluded BY NAME"* plus whether the predicate row differs.

Driven, two mutants, both red, **zero tracebacks**:

| mutant | failure line |
|---|---|
| a 22nd order-dependent field, **both** sides, differing | *"leaves **1** of the 21 … **DIFFERENT** (['probe_22nd_field'])"* |
| a key present on **one side only** | *"leaves **1** of the 21 … **DIFFERENT** (['only_on_the_swapped_side'])"* |

So a key-set difference is **named, not absorbed** — that was the half I most wanted to see. The
must-differ clause on `n_accepted_stream_differs` is **unchanged** (`:2031-2032`), and
`_ORDER_DEPENDENT` (`:1983`) is still the one-tuple.

One explanatory line so two numbers are not read as a discrepancy: the denominator is
`len(_np_free)` — the **unfiltered** block — so my injections (into the filtered blocks) print
"of the 21" while the coordinator's (into the block itself) print "of the 22". Both are correct for
what they measured; the numerator, which is the number the line is now about, is right in both.

## 3. DE41-R1 — **CLOSED**, and `.get()` moves no state (item 3)

`:813-824`: `pred#1` is **unchanged**; the read below it is now `c.get("n_draws_requested")`, with
a comment naming why. **Driven:** the guard-only mutant (`if "n_draws_requested" not in c:` →
`if False:`) now dies at
*"**FAIL (no refusal): KNOWN-BAD: a cell with no `n_draws_requested` REFUSES at `pred#1`** …"* —
rc 1 after 37 PASS, **zero tracebacks**. That is exactly the named failure I asked for.

**Does `.get()` move any state for a present field? No — measured across the value classes:**

| value | `_null_status` | `c[k] is c.get(k)` |
|---|---|---|
| `0`, `None`, `''`, `False` | `NO_NULL_REQUESTED` | True |
| `20`, `'20'` | refuses at `pred#2` (no quantiles, not DEGENERATE) | True |
| **absent** | **refuses at `pred#1`** | — |

**And the guard is still the only thing between an absent field and `NO_NULL_REQUESTED`** — that is
precisely why the known-bad now fires by name: with the guard gone, `.get()` returns `None` and the
cell would be **mislabelled** rather than crash, which is the failure the known-bad exists to catch.
The change trades a crash for a mislabel and then asserts the guard that prevents it; that is the
right way round, because the assertion is driven.

## 4. The class — one instance of each shape, both accounted for (item 4)

**Method.** Two AST scans over the module. (a) For the DE41-R2 shape: every `ok(...)` whose message
interpolates a name that does **not** appear in its condition, filtered to messages carrying a
verdict word (`identical`, `ALL `, `holds`, `unchanged`, `equal`). (b) For the DE41-R1 shape:
every `if "K" not in <obj>:` whose body raises, then any `Subscript` of the same key later in the
same function.

**(a) — no other instance.** Four `ok()` sites match the filter and I read each: `:2004` is
DE40-R3's own line (now correct); `:1920` interpolates `_nerr`, the caught exception in the
"REFUSED INSTEAD" idiom, which reports what happened when the condition is False rather than
asserting a state; `:2420` and `:2457` interpolate values that also appear in their conditions.
**None computes a verdict before its comparison.**

**(b) — one site, and it is safe.** `_null_status:813`'s guard is followed by an index of the same
key at **`:832`** — inside `pred#2`'s **refusal message** (`f"a cell requested
{c['n_draws_requested']} draws …"`). The only path to `:832` runs through the guard, which has
already established the key is present, so it cannot raise; and it is a *report* of the value, not a
read that decides. Named here so the scan's one hit is on the record rather than left for the next
reader to re-find.

## 5. Findings

**None.** Both closures are complete as measured, the two shapes are accounted for, and I found
nothing new at this tip. The two observations worth carrying forward are in §2 (the denominator's
provenance) and §4(b) (the safe index), neither of which is a defect.

## 6. Disposition (item 5)

**RELEASE `5658f24` as the runner's resting tip while the package waits on the USER.** The tip is
green on both launchers at 124, the two failure paths this round touched are the only ones it
changed, both are driven red **by name** with zero tracebacks, nothing can run
(`preflight()` refuses at the scorer, `--run` into any other outdir is rc 2), the DRAFT is
untouched, and the rebase preserved both my last RELEASE and the BE/DA landing as ancestors.

**Does my DE-40 item-6 answer on ask (5) still hold at `5658f24`? Yes — unchanged, including the
one clause round 41 made cheaper.** Round 42's +19 / −6 touch two failure paths and nothing else:
no split composition field, no change to the feed order, no change to the population statement. So:

- **MECHANICS on the consumed population** — one **computed** per-cell `train`/`score` composition,
  its refusal, its falsifier;
- **the `score`-split restriction** — the tape index must run **before** `build_reference`, the
  receipt must carry both the declared §3 population and the scored subset, and **§a must travel
  back to the USER**; the shrunken-population case still **reports itself** through
  `n_draws_requested` → `NULL_COLLAPSED` per stratum rather than being inferred.

Round 43 has nothing waiting from me: the queue is the USER's ruling on the five asks.
