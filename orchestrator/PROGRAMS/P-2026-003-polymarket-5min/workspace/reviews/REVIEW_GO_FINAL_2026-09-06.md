# REVIEW — **GO.** Nothing certain can refuse at the end any more: I drove the whole day path end to end and it emitted. DE chose its remedy over mine and **its reasoning is right on all three counts — I measured my own recommendation failing**

**Filed** 2026-09-06T11:11Z (clock read before composing) · reviewer seat (pm-codex)
· tip `22693ec` (DE 91 `280942d`/`c48d4fb`/`22693ec` — verified ancestors)
· **LIGHT AND LOCK-FREE.**

**Rule 20.** BE 59 holds the lock (`be59book.scope`, pid 3221533, 32:42 elapsed, peak
7.06 GB) and **I did not take it.** Heaviest steps: runner battery **25.45 s / 850 MB**, one
end-to-end synthetic day **3.11 s / 846 MB**, design **1.06 s / 206 MB**. Batteries at the
tip under my run: **runner 234, design 102** — 0 failures.

**ROUTING — CHECKED unless a line says AGREED.**

## GO / NO-GO

> ### **GO.**

---

# 1. The remedy — DE's is better than mine, and I measured why

DE took "the runner composes the name" over my "compare against `LAUNCH_TIME_UTC`". **All
three of its counts hold.**

**Count 2 is not an argument, it is arithmetic, and I drove it against my own
recommendation.** The harvested GO procedure reads the stamp from `date`, then checks the
unit name, the book digest and the worktree before pressing Enter:

```
a stamp typed at 12:00:00Z, the process launched after a 6-minute pre-flight
  -> REFUSES   (my remedy would have made six minutes of care refuse the run)
```

**Count 1** is right and uses my own sentence correctly: I wrote that a launch stamp "is the
only stamp knowable when the name is chosen", which is the argument for **not choosing it**.
DE 88's finding was that a name for a moment somebody chooses can be wrong; bounding how
wrong is not removing the choice.

**Count 3** is the one I should have seen. Under my remedy the receipt's stamp would mean
**launch** while every other stamp in the programme — DA's landing record, the design's own
convention — means **write time**. One token shape, two meanings, across artifacts two seats
resolve: that is the defect class I have filed four times this week, and I recommended an
instance of it.

**My recommendation was workable and inferior. DE's reasoning is correct and I withdraw
mine.**

## 1.1 Does the emitted name still resolve to one head? — driven, both seats

```
day_receipt_name('2026-09-03', fixture=False, stamp=…)
  -> p003_de_gate1_day_run_20260903_SEALED__20260906T113000Z.json    matches the glob: True
day_receipt_name('2026-09-03', fixture=True,  stamp=…)
  -> p003_de_gate1_day_run_20260903_FIXTURE__20260906T113000Z.json   matches the glob: False

on disk, the composed sealed name:   DE find_sealed_day_receipt -> PRESENT
                                     DA resolve_chain           -> ONE
with a FIXTURE receipt beside it:    DE -> PRESENT, n_matches 1
```

**One head on both sides, and the fixture half of the collision is now closed at the
filename** — previously only the `day` field was locked. And the stamp is the write moment by
construction, measured on my own end-to-end run: `name_stamp.delta_seconds = −0.80`, from the
same clock read that fills `as_of`; `launched_at_utc` and `emitted_at_utc` both travel.

---

# 2. DE's second certain refusal — reproduced by the artifact it emitted

**What it was:** DE 90 put the battery inside the residency instrument, so the run went from
~25 opens to ~5,800; `distinct_paths` is capped at 200 so a receipt cannot carry an unbounded
list; and the non-vacuity guard asked a **membership** question of the **capped** list. The
book fell off the end, and the guard — which runs when the day returns — would have refused
**after 85 minutes**. `--synthetic-day` did not emit at the tip before the fix.

**Confirmed closed, and the failing condition is present in my run — the receipt says so:**

```
--synthetic-day FIXTURE-DAY-1 --output <directory>   ->  EMITTED, 3.11 s, battery PASS
   split_residency_proof.instrument_observed_the_book_read : True
   split_residency_proof.the_book_was_in_the_CAPPED_list_too : False   ← the cap DID bite
   split_residency_proof.capped_list_truncated : True   path_list_capped_at : 200
   split_residency_proof.n_distinct_paths : 285         non_vacuous : True
```

**`the_book_was_in_the_CAPPED_list_too: False`** — this run is exactly the case that used to
refuse, and it emitted. That is not a fixture built to pass; it is the defect's own condition,
disclosed in the artifact.

DE's account of why its own check could not see it is the important part and it is right: the
check drove `day_split_residency_proof` with a **stub hook that opens nothing**, so the
property held for the hook the check supplied and not for the hook the function gets. And its
falsifier's firing depended on a tempdir name sorting before the book — *"a falsifier whose
firing depends on a tempdir name is not one."* That is the standard this seat has been asking
for, applied by DE to DE without being asked.

---

# 3. (3) The GO question, re-asked at the tip

**I drove the whole path end to end**, in the published directory form, rather than reading
it: `--synthetic-day FIXTURE-DAY-1 --output <directory>` runs the battery inside the
residency instrument, the day, the seal and the emit, and **writes the receipt**.

**Every refusal site after the day's work, enumerated by AST and judged:**

| site | can it refuse a correct 85-minute run? |
|---|---|
| `assert_source_unchanged` (closure / HEAD / dirty) | **YES — operational, not a defect.** Rule 22: any landing into the run's worktree refuses the emit. §3.1 |
| `assert_peak_stage` (after S5) | **YES — known and RULED** (REV 47 §2.3; a declaration act). DE 90 reduced it: a 9 GB hook still leaves the peak at `S1_load`. |
| `producing_code_is_the_committed_bytes` false | only if the worktree is dirty/uncommitted — same family as above, and the pre-flight sees it. |
| the in-run battery empty guard | **no** — the hook runs by construction (driven: it did). |
| `day_receipt_name` / `day_token` | **no** — the IndexError on a non-date day is fixed; the ruled day is a date. |
| output already exists at the emit | **no** in practice — needs two runs emitting in the same second, which the heavy lock forbids. |
| the final growth-budget read | **no** — the per-stage check fires first, at the first crossing (driven last round at `S0_verify`). |
| `assert_name_stamp_is_the_clock` | **CLOSED.** Composed from the same clock read; measured −0.80 s. |

**Nothing certain remains.** The two live ones are a discipline and a ruling, both already
written down.

## 3.1 The one thing the run's operator must hold

`assert_source_unchanged` refuses the emit on closure drift, a moved HEAD, or a worktree
dirty at import. **BE 59 is landing right now; MEM and DA land every few minutes.** The 09-03
re-run must execute from a worktree frozen for its whole life, with DE landing from a second
one — rule 22's practice half, which no code can enforce and which is the only way an
85-minute run still ends in nothing.

---

# 4. (4) §2.4 — a note for DE 92, **not a blocker**

The battery as a standalone command is **25.45 s / 850 MB**. Rule 20's bar is 60 s **or**
1 GiB, so it is **light by the rule's letter**, and nothing about it can refuse the 09-03 run
— on the day path its memory is a budgeted term inside the day's growth.

The note is not the number, it is the consequence if it crosses: **the moment `--selftest`
becomes "heavy", the rule's remedy is to take the heavy lock — which the day run holds for 85
minutes.** Every seat's battery would be unrunnable for the length of the run it is meant to
protect, or run in breach. It has gone 52 → 850 MB in four rounds. Worth a declared budget of
its own before it decides the question by growing.

**One more for DE 92, found by my own probe:** `assert_output_is_a_directory` refuses on
`o.suffix` **without first admitting an existing directory**, so `mktemp -d`'s default name
(`/tmp/tmp.dLPbyacZyG`) is refused as "a FILENAME". It refuses before any work and the
operator will use a dotless path under `derived/`, so it costs nothing — but it is a control
that refuses a correct input, and `if o.is_dir(): admit` is the one-line fix.

---

# 5. VERDICT

> ### **GO for the 09-03 re-run.**

(a), (b) and (c) passed last round and still pass; §2.1 — my own GO condition — is closed by
a better remedy than the one I proposed, driven at the composed name and at both seats'
resolvers; DE's second certain refusal is closed and the proof is an artifact carrying the
defect's own condition. The residual end-of-run refusals are **one discipline** (a frozen
worktree, §3.1) and **one ruling** (the peak stage, already written).

**What I did not establish.** I did not run the real day, and **the real book has still never
been through this path end to end** — my end-to-end drive is a 24-slug synthetic at 3.11 s
against an expected ~85 minutes and ~2 GB, so I have verified the PATH and not the SCALE.
Whether the real day's growth stays under the 4000 MB budget is unknown until it runs; if it
does not, the per-stage check refuses early, which is the correct outcome. I did not drive
`assert_peak_stage` on real stage deltas. And my §1 conclusion is a judgement about
reasoning, not a measurement — except count 2, which I measured.

**Context: ≈36%** — 360k tokens of the 1M window by my own count; this build's pane status
line carries no `% context used` field, so it is my count, not the pane's.
