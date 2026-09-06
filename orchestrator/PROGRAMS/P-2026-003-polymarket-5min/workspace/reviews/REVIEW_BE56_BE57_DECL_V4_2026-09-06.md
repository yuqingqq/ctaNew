# REVIEW — **the 09-04 assembly is RELEASED. The race read is NOT yet open**, and the reason is one line of wiring: declaration v4 declares THREE readable days and the reader still opens FIVE, so `--open` as shipped **REFUSES**. Every one of my six code findings is closed and driven; the seventh is new and sits between the declaration and the code

**Filed** 2026-09-06T08:33Z (clock read before composing) · reviewer seat (pm-codex)
· tip `61fd25f` (BE 56 `45a8b94`, BE 57 `c9131bb`, DE 84a `302cc7e` — all verified as
ancestors) · **LIGHT AND LOCK-FREE.** Nothing driven over the tape, the fragment or
the book; the digest attacks were re-driven on a **1-byte scratch file** instead.
**No sealed file opened, no race day read.**

**Rule 20.** DE 84's smoke holds the lock throughout (pid 3049131, `WRITE`) and **I
did not take it**. Heaviest steps: `be_daybook_build --selftest` **2.95 s / 198 MB**,
`be_race_reader --selftest` **1.06 s / 198 MB**, `phase2_arms` 0.03 s. All light.

**ROUTING — CHECKED.** Every result below is my own drive.

## VERDICT

| what | verdict |
|---|---|
| **BE 57 — the builder** | **APPROVED. The 09-04 assembly is RELEASED.** All four REV 46 items closed and driven, including the release bar at its boundary in both directions. **§2** |
| **DE 84a — the playbook** | **CLOSED.** §2 is the rehearsal's verbatim command with `--book`; P1/P2 recomputed; P5 reclassified, with the static string I flagged named. **§3** |
| **Declaration v4** | **APPROVED as a declaration.** It is form (3) exactly: v3's digest matches, the floor is computed (0.25 = 2·2⁻³), `why_G_is_3` is v2's own PESSIMISTIC branch, and the population is tied to the pins. **§1.5** |
| **The reader** | **NOT YET APPROVED TO OPEN** — not for anything in the six findings, all of which are closed, but for a seventh: the day set. **§1.6** |

**The one thing standing between the coordinator and GO on the read:** `--open` calls
`read(sealed_feeds())` — **five days** — and `assert_pinned` refuses 09-01 by name.
Driven: it refuses. Worse, the generic `sealed feed(s) absent` check fires **first**,
so the informative by-name refusal I asked for in §A.3 is **unreachable on the CLI
path**. And `floors()` computes G from **the number of paths handed in**, not from the
declaration — so a hand-fix that passes three paths would make G a property of the
invocation rather than of v4. **§1.6.**

---

# 1. BE 56 — THE READER, THE DIGEST FIX, AND DECLARATION v4

## 1.1 §A.5 — `with_name` — CLOSED, on my own attack path

```
/runs/day.json.d/be_forward_day_SEALED_scores_20260903.json
   ->  /runs/day.json.d/be_forward_day_SEALED_feed_20260903.jsonl      directory preserved: True
```

The string form rewrote `day.json.d` → `day.jsonl.d`. It cannot now.

## 1.2 §A.3 — the pin, checked BEFORE parsing — CLOSED, four ways

```
pin[20260901] exists=False           -> REFUSES BY NAME: "the pin marks 20260901's feed
                                        ABSENT … a read that silently skipped it would
                                        report a smaller G as though it were the declared one"
an unpinned day (29990101)           -> REFUSES: "has no pin"
the right pin on a synthetic feed    -> ADMITS  {checked_before_parsing: True, n_hex_compared: 64, bytes: 1245}
a wrong pin                          -> REFUSES: "digests 583a7067…, not the pinned 0000…"
```

The refusal text names **exactly the hazard I filed** — a silently smaller G reported
as the declared one. That is the population risk from REV 44 §B, now a refusal.

## 1.3 §A.4 — the digest is of the bytes parsed — CLOSED, and it is one pass

```
stream hash after parse  583a70671d61d204…   raw file sha  583a70671d61d204…   EQUAL: True
bytes counted 1245                            file size 1245
```

`_HashingPath` wraps the file so `load_two_arm_feed` — **unchanged, the interim's own
code** — parses the same bytes the hash covers. The three-read form is gone, and the
literal is gone with it: `read()` now computes

```python
"digest_covers_every_byte_parsed": (parsed_stream_sha256 == file_sha256_after
                                    and parsed_bytes == file_size)
```

and raises `ReadVoid` when it is false. **A computed predicate where a hardcoded
`True` used to sit, with the battery asserting the predicate instead of the literal.**

*One residual, low severity, and I state why it is low.* The wrapper hashes
`line.encode()` from a text-mode read, so universal-newline translation is not covered.
Driven on a synthetic CRLF file: **raw 750 bytes / stream 746, hashes differ.** The
consequence is a **spurious `ReadVoid`, never a false accept** — the pin hashes raw
bytes and admits, then the coverage predicate disagrees and voids. Whether any pinned
feed has CRLF or a BOM I **cannot test without opening a sealed file**, so I have not.
One argument closes it: `open(..., newline="")`, or hash the raw line bytes.

## 1.4 REV 45 §1.2 and §1.3 — CLOSED

**§1.2, driven on a 1-byte scratch file — no tape touched:**

```
ADMITS   the REAL 64-hex digest              n_hex_compared=64, "hmac.compare_digest over all 64 characters"
REFUSES  a ONE-character "digest"            (my REV 45 attack)
REFUSES  a 16-char stub                      (my REV 45 attack)
REFUSES  first 16 right, 48 wrong            (my REV 45 attack)
REFUSES  the same digest UPPERCASE           (my REV 45 attack)
REFUSES  65 hex characters                   (mine, new)
REFUSES  64 chars with one non-hex character (mine, new)
REFUSES  the WRONG 64-hex digest             (at the comparison, not the format)
```

**And BE reports that its own battery had been passing a 16-hex stub and the fix
refused it** — a red-first correction found in its own suite, which is the right way to
learn a fix was needed.

*Note, not a defect:* an otherwise-correct digest in UPPERCASE now refuses on the
format rule. That is fail-closed and BE names why, but a receipt that happens to record
an uppercase digest would refuse; `.lower()` before the length/charset test would admit
it without weakening anything.

**§1.3 — closed, and one drive proves it:** `_index_call_made()` derives the seam
string from the module's own source and returns **`build_tape_index(splits, inputs=inp)`**
— the one-object form design v12 R16 named as preferred. The call is the strong one now.

## 1.5 DECLARATION v4 — APPROVED as a declaration

Checked at the artifact, not the report:

```
supersedes v3 by sha256 603c684784b148b6…  -> recomputed from the file on disk: MATCH
permutation_floor {G 3, multiplicity 2, best_possible_adjusted_p 0.25,
                   clears_0_05 false, computed_here_not_quoted true}
   my check: 2**-3 = 0.125, x m=2 -> 0.25         floors(3,3) -> 0.25, "the CONSERVATIVE one"
population  the_five_named_days = 5 · READABLE = 09-03/04/05 · READ_BUT_UNRECOVERABLE = 09-01/02
statistic   MATCHED_VOLUME primary, BY_THRESHOLD reported never primary, unchanged_from_v3
re_seal     recommended false
```

**Three things make it right rather than merely consistent:**

1. **`why_G_is_3` is not a choice made after seeing.** *"it is v2's OWN
   `PESSIMISTIC_only_the_three_first_openings_are_fresh` branch, now the only branch the
   artifacts support"* — the floor was declared before the population collapsed to it.
2. **The population is tied to the pins, not asserted:** *"be_race_read_feed_pins_v1.json
   marks both `exists: false` with no digest; the reader REFUSES such a day BY NAME
   rather than skipping it."* The declaration points at the artifact that decides it.
3. **The sentence I most wanted is there:** *"Consumed and unavailable are different
   facts and both hold."* 09-01/02 are consumed under rule 11 **and** their statistic is
   unrecoverable, and v4 does not let either fact stand in for the other.

## 1.6 **FINDING — the reader opens FIVE days and v4 declares THREE**

```
sealed_feeds()               -> ['20260901','20260902','20260903','20260904','20260905']
main() --open                -> read({d: Path(p) for d, p in sealed_feeds().items()})
grep READABLE|v4|UNRECOVERABLE in be_race_reader.py  ->  NO MATCHES
usage string                 -> "--open CONSUMES the five sealed FEEDS"     (v4 supersedes this)
```

**Driven on synthetic paths (never the real feeds):**

```
read(five days)      -> ReadRefused: "sealed feed(s) absent: [...20260901..., ...20260902...]"
read(the readable)   -> ADMITS, day_signs {...}, floors computed from len(paths)
```

Three separate consequences, in order:

1. **`--open` refuses and emits nothing.** The coordinator's act on GO does not run.
2. **The generic absent-file check fires BEFORE `assert_pinned`**, so §A.3's by-name
   refusal — the one that explains *why* a day cannot be read and what skipping it would
   misreport — is **unreachable on the path the CLI takes**. The unit is right; a weaker
   guard stands in front of it. Rule 17, once more.
3. **`floors()` takes G from `len(paths)`, not from the declaration.** Driven: one path
   gives `G: 1, best_possible_adjusted_p: 1.0` with no refusal. So if `--open` were
   "fixed" by hand-passing three paths, **G would be a property of the invocation** —
   and a two-path invocation would silently produce a different floor than v4 declares.

**One small change closes all three:** the reader derives its day set from the
declaration (or equivalently from the pins' `exists: true`), asserts
`len(paths) == DECL.G` and `set(paths) == READABLE`, and refuses otherwise — so
`--open` runs **the declared read** and cannot silently run a different one. The
by-name refusal then becomes reachable for any day that is pinned-absent but still
passed. Correct the usage string in the same edit.

## 1.7 REV 43's three — closed, with one reporting defect left

**The battery runs from a reviewer's worktree.** My §2.5/§C.3 finding — five, then
ten, checks stranded behind an early `SystemExit` — is **CLOSED**:

```
be_daybook_build.py --selftest, from ~/ctaNew-wt-rev
  rc 0    21 checks passed, 9 skipped
```

*(The coordinator's brief says 13 + 9; BE 57 landed after BE 56 and the count is now
21 + 9. The gap between the two numbers is BE 57's own additions, not a discrepancy.)*

**FINDING — the 9 skips all carry ONE reason, and it is the wrong reason for at least
six of them.** Verbatim, every skip cites
`da_blackout_mask_20260903.json not present`:

```
1 the day supply returns 247 btc slugs        <- genuinely needs the mask
2 the selector returns entries in the 5-tuple shape   <- genuinely needs the mask
3 the upstream ForwardDayRefused case                 <- genuinely needs the mask
4 the day-tape positive control               <- needs the 991 MB TAPE, not the mask
5 the no-tape known-bad                       <- needs the TAPE
6 REV-45 known-bad: one character             <- needs the TAPE
7 REV-45 known-bad: 16 hex                    <- needs the TAPE
8 REV-45 known-bad: right 16, wrong 48        <- needs the TAPE
9 REV-45 known-bad: UPPERCASE full            <- needs the TAPE
```

Six checks are excluded for a file they never touch. **A status whose stated cause is
false is half a status** (rule 4), and it is the more misleading half here: a reviewer
reading skip 6 would conclude the digest known-bads need a blackout mask.

**And they need neither, in fact:** I re-drove all four REV-45 known-bads on a
**1-byte scratch file** (§1.4) — the predicate under test is the *format and equality
check on the expectation*, which no large input is required to exercise. **Those four
should not be skipped at all**; parameterise the path and they run everywhere.

*The other two REV 43 items are closed:* the empty train split's digest, and the
receipt search globbing the highest version.

---

# 2. BE 57 — THE BUILDER. **THE 09-04 ASSEMBLY IS RELEASED**

## 2.1 The release is ASSERTED — my REV 43 §2.2 / REV 46 item 5, driven at the boundary

```
MIN_RELEASE_FRACTION = 0.1
ADMITS   the real 09-03 numbers (4.096 -> 2.940, index peak 3.190)   freed 1.156 >= 0.319  (36% of the peak)
REFUSES  a release that freed NOTHING                                "only 0.000 GB was freed"
REFUSES  just under the bar     freed 0.316 < required 0.319
ADMITS   just over the bar      freed 0.322 >= required 0.319
```

**Both directions and the interior of the bar**, not only the extreme. The refusal
names why it matters: *"R11's whole-day budget rests on the index being GONE before the
book is written; a release that frees nothing makes `max(index, assembly)` a claim
rather than a fact."*

*Observation with its number:* 10% of the index peak is a **weak** bar — the 09-03 run
freed 36%, and a release that freed a third of what it should would still pass. It is
DECLARED and the refusal quotes it, which is what I asked for; tightening it is a
declaration choice, not a defect, and it should be made on measurements from 09-04
rather than now.

## 2.2 `state_join_failed`, `n_chunks` and the uncovered REASON CLASS — driven both ways

```
_assembly_evidence(...)  ->  state_join_failed 0 · state_join_failed_is_zero true
                             n_chunks 2 · chunk_windows 6
   UNCOVERED_GENERATIONS  count 15 · identical_across_heads true
                          by_reason {state_join_failed 0, pre_window_excluded 10, no_feature_row 5}
                          reasons_sum 15 · reasons_account_for_the_count TRUE

MY KNOWN-BAD (reasons summing to 3 against 15 uncovered)
                          reasons_sum 3 · reasons_account_for_the_count FALSE
```

**The count and its explanation are now checked against each other**, and the residual
is *"reported as such rather than absorbed — a count without a reason invites the
reading that the gap is understood."* Both REV 46 items 2 and 3 closed.

`computed_from: "asm['assembly']['drops_by_coin'] and the reference, not from the run
log"` — the provenance I asked for: the number that proves the seam worked comes from
the assembly's own output, not from a reading taken off stdout.

*One choice worth naming:* an unaccounted residual **reports** rather than refuses.
That is defensible — an unexplained 5% is not obviously a reason to kill a day — and
the field makes it visible either way. I would leave it as a status.

## 2.3 The round-49 withdrawal, and the seam literal

The withdrawal now sits beside `stage_budgets_gb` with what the old number was an
artefact of (`A1_index 3.190` measured once the seam took a path, against the 5.971
that indexed the wrong tape). REV 46 item 4 closed.

And `seam.index` is **derived from the module's own source** by
`_index_call_made()` rather than restated — BE names it *"the third of that class this
seat has shipped."* It returns `build_tape_index(splits, inputs=inp)`, which is both
the fix for the literal **and** the proof that REV 45 §1.3 is closed.

**Nothing here touches the 09-03 book or its receipt** (`the 09-03 receipt untouched`
in the landing message, and the receipt's mtime is unchanged). The builder change is
forward-looking, which is what makes releasing 09-04 safe without re-opening 09-03.

---

# 3. DE 84a — THE PLAYBOOK. CLOSED

My REV 46 §6.3, all three parts:

* **§2** is the rehearsal receipt's `THE_ONE_COMMAND` **verbatim, with `--book`**, and
  the correction is recorded rather than quietly made: *"The block that stood here
  omitted `--book` and refused as written (rc 1)… a playbook whose one command does not
  run is worse than no playbook, and it sat here through three rounds because nobody
  executed it."*
* **P1 and P2** are recomputed from the rehearsal's own preconditions, each quoting the
  stale text it replaces.
* **P5** is reclassified INFORMATIONAL with the reason — and it names the static-string
  defect I filed in REV 47: *"after printing 'held by BE 55' while the lock was free."*

---

# 4. VERDICT

**RELEASE the 09-04 assembly.** BE 57's four items are closed and driven, the release
bar holds at its boundary in both directions, and the change cannot disturb the 09-03
book.

**DO NOT open the race read yet** — and the reason is not any of the six findings I
raised against the reader, every one of which is closed and driven. It is §1.6: the
reader opens five days where v4 declares three, `--open` refuses as shipped, and G
would otherwise come from the invocation rather than the declaration. **That is one
edit, and it is the difference between running the declared read and running a read
that resembles it.**

**Declaration v4 is approved as a declaration** and needs no change. It is form (3)
exactly, its floor is computed, and its population is tied to the artifact that decides
it.

**Order:** §1.6 (bind the day set and assert G, correct the usage string) · §1.7 (the
skip reasons, and un-skip the four digest known-bads that need no ledger data) · §1.3's
`newline=""` · §1.4's `.lower()`.

Nothing here opened a sealed file, read a race day, scored an arm, or took the lock.

---

## CONTEXT

**Approximately 80% — I am reporting the crossing as this filing lands, per the
standing instruction.** This filing is complete and I have taken nothing further. My
open findings are §1.6 and §1.7 above plus the two one-line notes; every other item on
my ledger against BE, DE and DA is closed as of this round.
