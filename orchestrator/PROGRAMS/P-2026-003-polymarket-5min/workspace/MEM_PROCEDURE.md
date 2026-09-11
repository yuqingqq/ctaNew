# MEM seat — the per-round procedure

**Harvested verbatim in substance from the MEM seat's stop answer at 2026-09-07T16:15Z,
before its context was cleared (R-812). None of it was in a file; all of it was in one
context. MEM maintains this file from here on — it is MEM's, not the coordinator's.**

## Sweep order, per round

1. `git fetch origin`.
2. Read the clock: `date -u '+%Y-%m-%dT%H:%M:%SZ'`. **Never write a time not read from a
   clock.**
3. Establish the window **bounded on the tip you last READ, never on your own last
   commit** — seat commits land between a read and a write and become ancestors of yours
   (rounds 265, 267, 279, 282 each caught a landing in that gap).
4. Read the named R-entries out of `COORDINATION.md` with
   `awk '/^### R-NNN/{f=1} f&&/^### R-NNN+1/{exit} f'`.
5. **Verify every claim at the artifact, never from the entry.** Prefer DRIVING code
   (import and call) over reading it.
6. **Every probe carries a falsifier** — a known-positive that must appear, or a control
   that must be absent. *An empty probe result is a broken instrument until proven
   otherwise* (this caught a bad pathspec at round 266 and a TSV-read-as-CSV at 280).

## Counts, and how they are measured

- Flags and provenance by `yaml.safe_load`, **never by regex** — a line-regex over-counted
  by 1 and 15 at round 265 and wrong numbers were published before the parse corrected
  them.
- Classification counts, findings, `orphan_entries`, `checked_artifact_missing`,
  `window_generations`, `window_over_by`, `new_flags_without_provenance` all come from
  `live/pm_research/mem_flag_provenance.py --audit --json`.
  **`--selftest` (31 checks) runs first, every round.**
- Batch number = `len(re.findall(r'^## Batch', archive, re.M)) + 1`.
- Last landed (round 284, 2026-09-07): flags 2289, provenance 1834,
  **1,544 CHECKED / 285 RELAYED / 5 MALFORMED / 455 UNMARKED** (160 consecutive rounds
  unchanged on UNMARKED), findings 183, orphans 0, missing-artifact 178, window 3/3.

## Classification rule (from `classify()`)

- `prov: CHECKED` needs `artifact:` **and** a non-empty, non-bare `said:` — a bare
  confirmation is MALFORMED.
- `prov: RELAYED` needs a `from:` key naming the row or entry — **not** `artifact:`. One
  filed with `artifact:` only at round 266 landed as MALFORMED (5→6) until fixed.
- Anything else is MALFORMED.
- **Only CHECKED is authoritative**; RELAYED and UNMARKED are equally non-authoritative.
- `artifact_exists()` returns *undecidable* for `git:` / `http:` / `https:` prefixes and
  resolves everything else as a path — so an artifact field must be **one resolvable path
  or a `git:` ref, never compound prose** (17 compound fields pushed missing 174→191 at
  round 265).
- **Never cite `/run/user/.../transient/<unit>.service`** — it dies with the unit and
  expired four flags at once (round 268). Cite the run journal
  `p003_de_gate1_run_journal_*` instead, and put live systemd readings in the prose with
  their clock.

## Duplicate-name gate — run BEFORE any write

`set(re.findall(r'^  ([A-Za-z0-9_]+):', txt, re.M))` over the whole file; refuse on
collision. It has caught this seat twice (rounds 252, 275); it over-matches, so it can
only over-refuse.

## State-file structure

`STATUS.yml`, in order: `program`, `name`, `phase`, `status`, `branch`, `updated:`
(folded `>-`, **exactly three generations, newest first**, each beginning
`  <ISO>Z (MEM ROUND N --`), `previous_updates:`, `flags:` (flat mapping, newest block
first under a `# --- MEM round N ... ---` comment), `focus:`, `flag_provenance:`
(`prov` / `artifact` / `as_of` / `said`), `standing_rules:`, `tasks:`.

- Generation boundary regex: `^  \d{4}-\d{2}-\d{2}T.*Z \(MEM ROUND`.
- The `updated:` block is a **folded scalar** — line anchors vanish after parsing, which is
  why the instrument matches raw text.
- Rotation: prepend the new generation, keep two, archive the overflow **verbatim** to
  `STATUS_UPDATED_ARCHIVE.md` as `## Batch N — archived <clock> (1 entry, rolling-window
  overflow)` plus a fenced block.
- `HANDOFF.md`: round blocks prepended, `# READ FIRST — round N (MEM, <clock>, tip
  ` + backtick-sha + `)`.
- **Escape literal braces in f-string generation text** (`{{path, sha256}}`) — this raised
  at round 280 before any write.
- **Escape literal `%` in `%`-formatted generation text** (`97.5694 %%`) — raised at round 287, but only AFTER
  `STATUS.yml` and the archive had been written, because the `HANDOFF.md` block was built last: the round's STATUS
  write had already landed and re-running the script would have double-applied it. Two rules from one failure —
  **order the blocks so a later failure cannot orphan an earlier write**, and prefer a placeholder +
  `.replace()` template over `%`/f-string formatting for any text carrying numbers with units. A body full of
  `%`, `{` and `}` is DATA, not a format string.

## Landing

- Register rows go **only** through the locked insertion form:
  `bash scripts/land_register_row.sh --row <rowfile> 'Q-MEM-N' <msgfile>`, dry-run first.
  The legacy hand-edit is RETIRED (R-784) — two seats' uncommitted rows deadlock each
  other.
- The row must be one line beginning `| Q-MEM-N |`.
- On `HELD REGISTER_DIRTY`: wait and re-run. **Never withdraw a row that is not yours.**
- State files commit **separately, by explicit pathspec**, then push with the status
  captured.
- **Never `git add` under `data/` or another seat's files.**

## Closed: the `?? data` observation

MEM's stop answer left one open observation: wt-de has 345 tracked files under `data/`
with sparse-checkout false, yet `git status --short` there reports only `?? data`, and
four checks offered no mechanism. **Closed by the coordinator at 2026-09-07T16:1xZ:**
`git -C /home/yuqing/ctaNew-wt-de ls-files -v data` returns **345 paths, every one flagged
`S` (skip-worktree)** — set deliberately by the worktree rule (R-554) so git stops
reporting the tracked data files as deleted behind the symlink. `data` itself lists as
untracked because no tracked path is literally `data`. So `?? data` **is** the expected
clean state of a seat worktree, which is what REV 103 §5's P10 condition reads: P10 stays
meaningful for anything *else* that appears in that output.

## Round 334 — guard the OPERATION, not the line that broke last time

Round 332: unquoted colon-space in an `artifact:` line broke STATUS.yml. I added
a guard for `artifact:` lines. Round 334: the same break, in **prose inside
eleven flag values**. The guard must quote **any flag scalar containing ": "** —
that is the operation; `artifact:` was only where it first appeared.

Same round, same shape: ten `%%` escapes reached the file because values
substituted INTO a format string are not themselves format strings. Escape at
the point of formatting, or use `.replace()` throughout (round 320's lesson) —
and then **grep the written file for `%%` before validating**, because YAML
parses `%%` happily.

Standing pre-commit checks, in order: (1) collision gate, (2) quote every flag
scalar with ": ", (3) no `__TIP__`/`__CLOCK__` left, (4) no `%%` left,
(5) `yaml.safe_load`, (6) `--selftest`, (7) `--audit` and read
`checked_artifact_missing` for THIS round's keys — a CHECKED entry whose
artifact does not resolve is a filing error, and it caught two of mine here.

## Round 343 — a missing artifact can be the finding, not a filing error

Three flags cited a params file that does not exist in this tree, because the
commit carrying it is staged in another worktree and is not an ancestor of HEAD.
I read it with `git show <sha>:<path>`, so the flags are legitimately CHECKED.

**Do not repoint such an artifact to something that happens to resolve.** Leave
it at the in-commit path and record why: when the commit lands, the path
resolves and `missing-artifact` returns to its baseline **on its own**. That
turns the audit into a landing detector. If the count does not come back, the
merge did not happen.

Baseline to watch: **185**. This round it is **188**.

## Round 349 — a delimiter defined by SPELLING cut a generation in half

The rotation split `updated:` on `\n(?=  \d{4}-\d\d-\d\dT)` — any line beginning
with an ISO date at column 2. Round 348's own prose contained
`  2026-09-13T00:00Z, ONE LOOK, NO EXTENSION …`, so that continuation line was
counted as a generation, round 348 was cut in half, and **a real generation
(round 347) was rotated out one round early.** The audit caught it as
`window 2/3`; the content was recoverable from the archive.

**The splitter now requires the structure, not the prefix:**

```python
GEN_START = r"\n(?=  \d{4}-\d\d-\d\dT[\d:.]+Z \(MEM ROUND )"
```

This is rule 32 — enumerate by operation, not spelling — applied to my own
rotation for the third time (round 332 colons, round 336 single-spelling counts,
now this). **Add to the pre-commit list: after rotation, assert the number of
generations is exactly 3 AND that each one's first line contains `(MEM ROUND`.**
A date can appear in prose. `(MEM ROUND ` is a delimiter I control.

## Round 351 — LoadState is required on every unit-state reading

`systemctl --user show <unit> -p ActiveState -p Result` returns
`Result=success`, `ActiveState=inactive`, `SubState=dead` for a unit that
**does not exist**. No error. I proved it by inventing a unit name.

I used exactly that command to report unit state at rounds 338, 339, 345 and
349. Those readings were correct — the units were real — but **the instrument
could not have told me otherwise.**

**Required from now on:** every unit-state reading carries `LoadState`
(`loaded` vs `not-found`), and a reading whose `LoadState` is not `loaded` is
**not a reading**.

Third instrument of mine to fail by satisfying a word: colons (332), a
single-spelling count (336), a rotation delimiter (349), and now this. **The fix
is never "be more careful with the old field" — it is to add the field that can
say no.**

## Round 353 — COMMIT BEFORE VALIDATING, not after

My round-353 STATUS.yml write was **destroyed while uncommitted**. I wrote it at
~03:15, ran the selftest and audit at ~03:17, and the audit reported the
*round-352* counts: another seat's tooling had restored STATUS.yml from git in
that window, and `git status` showed the file CLEAN — the edit was simply gone.
Recovered by re-running the write script, which is why the script is written to
be re-runnable from a clean file.

**New ordering, and it is rule 31's ("commit as soon as it parses") applied to
my own sequence:**

1. write STATUS.yml → `yaml.safe_load` → **`git add` + commit IMMEDIATELY**
2. *then* HANDOFF, procedure, archive
3. *then* selftest / audit / register row, amending the commit if a repair is needed

**Never leave a state-file edit uncommitted across another command.** This is a
shared tree with several seats' tooling running in it; an uncommitted edit is not
state, it is a gamble.

## Round 357 — check the register BEFORE re-deriving, and date every convergence

DA 221: **a filed result does not reach the seat's own next claim, including when
the seat is the author** — because filing discharges the feeling of having
handled it, and the artifact then sits unconsulted. Better indexing does not fix
this.

**I am an instance.** At round 354 I presented my own 2-of-10 hashing as
verification without recording that **R-888 had held that result since the
previous morning** (`0da40bb`, 09-10 07:17). At rounds 344 and 352 I claimed
convergence without stating which leg was filed first.

**Required from now on, both halves:**

1. **Before re-deriving anything, grep `COORDINATION.md` for it.** If the
   register already holds it, say so and state what my instrument adds.
2. **Every convergence claim carries the filing order** — when each leg was
   first filed, and whether the later party had access. Differing instruments is
   necessary and insufficient.

## Round 359 — quote EVERY flag value, and NEVER repair by slicing to a section end

Two failures in one round, the second worse than the first.

**(1)** A flag value **beginning with a single quote** — `'By luck again, not by
design.' Day one's validity…` — is parsed by YAML as a *quoted scalar* and breaks
at the next apostrophe. My `q()` quoted only values containing `": "`. **It now
quotes every value unconditionally.** Third time the same lesson: guard the
operation, not the spelling that broke last time.

**(2)** The repair was far worse than the fault. I sliced from the round's marker
to `\nflag_provenance:` — **that is the WHOLE flags section, every round's** —
and re-quoted **1752 values across the entire history**. It did not fix the file
and it rewrote thousands of lines I had no business touching.

**Rules now:**
- **A repair is bounded by the round's own block: marker to the NEXT `  # --- MEM
  round` marker, never to a section end.**
- **If a write fails validation, RESTORE the file from the last valid commit and
  re-run the write.** Do not patch a broken file in place.
- **Check `$?` after the write before committing.** The broken file was committed
  because the shell ran on past a Python traceback.

## Round 368 — a specification is not an implementation; grep before you file

I recorded the v12 tripwire (`CONCENTRATION_FINDING`, thresholds, firing
conditions) at rounds 365 and 367 **in the same voice I use for things I have
driven**, and never asked whether the emit existed. It did not: **zero `.py`
files, in the shared tree and in `wt-deval`.** The token lived only in three
REVIEWs and in my own HANDOFF.

Worse, I closed round 367 praising the batch for being "a field or a drive
rather than a claim in prose" — **about an item that was a claim in prose.**
*An endorsement is harder for the next reader to question than a specification.*

**Required from now on:** when a dispatch describes an instrument by NAME —
a refusal token, a check, an emit — **grep for that token in `.py` before
filing it**, and record which of these it is:

- `SPECIFIED` — named in a declaration or review, no code found
- `PRESENT` — token found in code
- `DRIVEN` — token found *and* a drive/falsifier reported with it

**Never let a SPECIFIED item inherit the voice of a DRIVEN one.**

---

## Round 382 — **run `--audit` before the round ends, not as the report's last line**

Round 353 put the commit immediately after the parse, and that stands: *in a
shared tree an uncommitted edit is not state, it is a gamble.* Round 359 added
*check `$?` before committing.* **Both were satisfied this round and both were
insufficient.** The write parsed, the commit landed, and **all twelve of my new
provenance entries were `MALFORMED`** — I had invented `checked:`/`artifact:`/
`by:` where the instrument enforces:

```yaml
  <flag_key>:
    prov: CHECKED          # or RELAYED
    artifact: <path or git:<ref>:<path>>
    said: "what the artifact said -- NOT a bare confirmation"
```
```yaml
  <flag_key>:
    prov: RELAYED
    from: <the row or entry it came from>
```

A `said:` that normalises to "verified", "checked", "confirmed", "correct",
"as stated" is **refused by name**: it *"would read identically had the check
not happened."*

**Required from now on:** after the parse-and-commit, run
`mem_flag_provenance.py --audit` **in the same tool call block**, read the
class counts, and land the repair as a numbered follow-up commit (`1b`, `1c`)
**before writing HANDOFF**. *The class counts are part of the write, not part
of the report.*

### And the refusal that was not cosmetic

The audit also reported one entry as **`CHECKED, but its artifact is not on
disk`**. It was right: I had cited `live/pm_research/de_preflight_matrix.py` as
a path, and that file **is absent from `mm-research`** — it exists only on
`origin/be-build-runner` and `origin/de-freeze-chain-v2`.

> ***The schema violation was mine; the absent-artifact refusal was a finding.***
> It located a second instance of the very thing this round's headline names —
> `PENDING_ORIGIN_MAIN` reaching the code — **that I had not noticed while
> writing the flag about it.**

**So: cite off-disk evidence as `git:<ref>:<path>` deliberately, and when the
audit says a path is absent, ask whether the absence is the finding** before
reaching for the `git:` form. Here it was. (Amended 2026-09-11T12:39:51Z.)

---

## Round 393 — **the audit must GATE the commit, not merely precede it**

Round 382 required `--audit` in the same tool-call block as the write. This
round satisfied that and still committed a bad citation: the block ran

```bash
python3 …/mem_flag_provenance.py --audit | grep r393 …   # printed a FINDING
git add … && git commit …                                # ran anyway
```

***An audit whose output nothing branches on is a report, not a gate.***

**Required from now on:**

```bash
A=$(python3 live/pm_research/mem_flag_provenance.py --audit 2>&1 \
      | grep "rNNN" | grep -v "^  CHECKED")
if [ -z "$A" ]; then git add … && git commit … ; else echo "$A"; fi
```

### And the repair failed too, for the reason round 359 already named

My first fix asserted `s.count(artifact_line) == 1` and found **seven** — six
earlier rounds cite the same path. The assert stopped it, so nothing was
damaged, but the repair I reached for was **unbounded**.

> **Bound every repair to the round's own key**, by regex anchored on the flag
> name, and assert that the total occurrence count is **unchanged** afterwards:
>
> ```python
> pat = re.compile(r"(^  %s:\n    prov: CHECKED\n    artifact: )<old>(\n)" % re.escape(KEY), re.M)
> assert pat.search(s); s2 = pat.sub(r"\1<new>\2", s, count=1)
> assert s2 != s and s2.count("<old-basename>") == s.count("<old-basename>")
> ```
>
> *Round 359 said repairs are bounded by the round's own block. This is the same
> rule one level finer: bounded by the round's own KEY.* (Amended 2026-09-11T16:39:21Z.)

---

## Round 397 — **when ABSENCE is the finding, the search must be able to fail visibly**

Promoted from REV via R-922, and the first thing I did with it was turn it on my
own round-396 report that *no file asserts "can pass a 69-fold screen."*

**The conclusion survived. The method did not.** I had run a bare
`grep -rniE` over three trees. ***Had my path been wrong, my regex malformed or
my tree the wrong one, the command would have returned nothing and I would have
reported the same absence with the same confidence.***

**Required from now on — three searches, in this order:**

```bash
# (a) POSITIVE CONTROL: a token I know is present, same command shape
grep -rniE "<token-I-planted>" <trees> | wc -l      # MUST be > 0
# (b) KNOWN-BAD: a token that cannot exist
grep -rniE "zz_impossible_token" <trees> | wc -l    # MUST be 0
# (c) THE REAL QUERY, only then
```

*(a) proves the search can speak; (b) proves it is not matching everything.
An absence reported without (a) is a zero from an instrument that never showed
it can fire — SEAT_PROTOCOL rule 15, in a shell.*

### And the second defect was SCOPE, which is the reusable half

My exclusion filter was **line-scoped** (`grep -v corrected|first reading`)
while the markers identifying a correction live in the **flag key** and
elsewhere in the sentence — often outside the matched line. **Three candidates
were dropped before I could see them**; the answer happened to be right.

> ***A filter that removes candidates before a human sees them must be scoped to
> the same unit as the thing it filters on.*** When in doubt, do not filter —
> print the context and read the hits. (Amended 2026-09-11T19:17:35Z.)
