# Review — DE round 23 (DE21-R1: the block's extent pinned by the walk's own stop)
reviewer: claude (pm-codex seat) · round opened by the coordinator (pm-co)

**Pinned tip executed: `a83083a`** (Q-DE-41 row at `40806b7`).
**Request of record:** `REQUEST_DE_ROUND_23_2026-09-02.md` (at `22baa54`).
**Composed 2026-09-02T15:31:34Z.** One filing, per R-377.

**How the suite was run, since the request asks:** in `~/ctaNew-wt-rev` at `--detach a83083a`,
**without** a scratch `PM_DATA_ROOT` — this worktree carries `data/pm_5min/raw` as a symlink to
the canonical tape (`data/pm_5min/raw -> /home/yuqing/ctaNew/data/pm_5min/raw`), so the resolver
takes branch 2 and the admissible suite runs in place. Read-only under `data/`; every mutant
applied to the worktree copy with `__pycache__` cleared before each run (R-446) and restored —
worktree clean after.

Scope confirmed: `de_admissible_windows.py` only, **+90/−4**, one file in the commit. Suites at
the tip: admissible **87** both launchers, ratification 160, seam 69, rc 0.

---

## Verdict

### RELEASE for `a83083a`. DE21-R1 is closed for both shapes it named, and the predicate is prose-blind as claimed.

Two findings, both LOW, and they are two halves of the same observation: **the predicate tests
`#:`-ness where it means "the comment run was not cut"** — three ordinary comment edits truncate
47% of the block with `stopped_at_a_real_boundary` reported **True** — and **the known-bad that
drives it encodes today's layout**, so the next contiguous paragraph above the head turns it red
with nothing wrong. The round replaced anchors that go stale with a predicate that cannot; its
own falsifier still can.

---

## 1. Ruling — the right shape, one token short of the right predicate

`declared_limit_boundary()` (`:265`) walks up from the anchor while lines start with `#:`, skips
the blank gap, and asserts the first non-blank line above it is not another `#:` line. Intact:
`first_read_line 139`, `above_line 137`, `above = '_REBOUND = "<rebound-import>"'`,
`stopped_at_a_real_boundary True`.

**Both shapes DE21-R1 named are caught** (driven through the reader, my round-21 standard):

| cut | chars read | boundary |
|---|---|---|
| a paragraph above the head with a blank between | 3,752 (identical to intact) | **False** |
| a blank inside the block, above OVER-CAUGHT | 1,975 (−1,777) | **False** |

**What the predicate accepts that it should not** — the question asked. Four one-line
interruptions inside the comment run truncate the block by the same 1,777 characters (47%) and
report `stopped_at_a_real_boundary` **True**:

| the line inserted above OVER-CAUGHT | chars read | boundary |
|---|---|---|
| `# a plain comment, not #:` | 1,975 | **True** |
| `  #: an indented continuation` | 1,975 | **True** |
| `#` (bare) | 1,975 | **True** |
| `X = 1` (code) | 1,975 | **True** |

So the predicate answers "the line above the gap is not a `#:` line" when the property it wants is
"the comment run was not cut in half". A blank is caught; a `#` is not — and a stray `# TODO`, a
`# noqa`, or a reflow that indents a continuation are ordinary edits, more ordinary than a blank.

**Ruled: the shape is right and the predicate is one token short.** Structural, prose-blind and
covering both named shapes is the correct design — it is what I asked for at round 21 and it is
strictly better than the anchors. The fix is `not above.lstrip().startswith("#")`, which closes
the three comment shapes at once. The code-insertion case is the honest residual: code above the
block is exactly what the intact file looks like (`_REBOUND = …`), so a mid-run statement cannot
be told from a real boundary without more structure — worth stating in the docstring rather than
chasing. **DE23-R1.**

## 2. The predicate knows no prose — and the claim about upward growth is false, for a different reason

Grepped: `declared_limit_boundary`'s body contains **zero** anchor tokens (`DECLARED`, `REFUSED`,
`OVER-CAUGHT`, `runpy`, `builtins`) — it cannot go stale as the prose grows, which is the property
the round claims and it holds for the predicate.

But the request's own test — *"a fourth upward growth that keeps the block contiguous stays
green"* — **fails**. I added a contiguous `#:` paragraph above the head (no blank) and the suite
went **red**, not at the extent check but at its known-bad. That is **DE23-R2**, and the predicate
is not at fault: the known-bad is.

The blank-below variant of the same paragraph goes red at the extent check itself, naming the
first line read — as designed.

## 3. The strengthening, taken — and the round-21 ruling stands, refined

`_declaration_holds(text)` (`:1155`) is the check's own predicate, called by the check and by both
known-bads. Driven:

| conjunct dropped from `_declaration_holds` | result |
|---|---|
| the **phrase** conjunct | **red** at the `_cut2` known-bad |
| the **OVER-CAUGHT** conjunct | **red** at the `_cut` known-bad |

Both edits were invisible before; each is now caught by the known-bad that drives that half.

**Agreed, with a refinement rather than an overturn.** My round-21 ruling was about the general
case: a suite cannot falsify the deletion of its own assertion, and requiring assertion-mutants
as a standard would test the harness rather than the code. What the lift does is **shrink the
surface on which that is true**: by naming the predicate and having the known-bads call it, an
edit *inside the predicate* stops being an assertion-edit and becomes a change to a subject two
known-bads exercise. What remains un-falsifiable is an edit to the `ok(...)` line itself
(`ok(_declaration_holds(_limit) or True)`) — still invisible, and still rightly so. So the ruling
holds and its scope is now smaller by exactly the amount DE lifted.

## 4. The mutants

| mutant | result |
|---|---|
| the boundary predicate hardcoded `True` | **red** (at the above-head known-bad) |
| `_declaration_holds` loses the phrase conjunct | **red** at `_cut2` |
| `_declaration_holds` loses the OVER-CAUGHT conjunct | **red** at `_cut` |
| a paragraph above the head with a blank | **red** at the extent check |
| a contiguous paragraph above the head | **red** — DE23-R2, and it should not be |

All run with `__pycache__` cleared before each execution, per R-446.

## 5. Surface, and nothing pinned

`_BLIND_HEAD` and `_declaration_holds` are suite locals (`hasattr(daw, …)` **False** for both).
`declared_limit_boundary` **is** new module surface — deliberately, as a reader beside
`declared_limit_text`, and it is the round's subject rather than a constant. The block length
(**3,752**) is printed and asserted nowhere. Supply `n_supplied_total` **1,875**; the seam emits
**1,875**; `seam.daw is de_admissible_windows` **True**; R-419 on the real register
`verified_for_new_run True, unverifiable [], superseded_by []`; `_g_no_decision_field` untouched;
`decides: "nothing -- this reports…"`.

## 6. Counts and census — nothing removed

84 → **87**; `EXPECTED_CHECKS = 87`. AST census `0255b60` → `a83083a`: `ok` 64 → **67**,
`refuses` 15 → 15. One `ok(` line appears removed — `ok("OVER-CAUGHT" in _limit and _phrase in
_limit,` — and it is the declaration check **rewritten in place** to call `_declaration_holds`;
its subject is unchanged. The DATA_ROOT split is not in this round: the `ROOT` and `MASK_DIR`
lines are byte-identical to `0255b60` (zero diff hits), so it stays behind DA's landing.

## 7. Discipline

The commit touches **one file** (`live/pm_research/de_admissible_windows.py`, +90/−4) — a
pathspec commit, confirmed from `git show --stat`. Nothing under `data/`. Line numbers in the
request resolve at the pin (`:265` the boundary function, `:1155` `_declaration_holds`, `:1243`
and `:1252` the two known-bads).

---

## Findings

### DE23-R1 — LOW — the predicate tests `#:`-ness where it means "the run was not cut"

`declared_limit_boundary` returns `stopped_at_a_real_boundary = not above.startswith("#:")`
(`:281`). Measured on copies, each a single inserted line above the OVER-CAUGHT paragraph:

```
# a plain comment, not `#:`     -> 1,975 of 3,752 chars read (47% unread), boundary True
  #: an indented continuation   -> 1,975,                                   boundary True
#                                -> 1,975,                                  boundary True
X = 1                            -> 1,975,                                  boundary True
```

Each is the same truncation the round exists to catch, and each passes. The first three are
edits a person makes without thinking about this predicate at all — a note, a lint pragma, a
reflow — while the shape that *is* caught (a blank) is the one an editor inserts by accident.

**Closure:** `not above.lstrip().startswith("#")` — one token, still prose-blind, and it covers
`#`, `#:` and the indented form together. The code-line case cannot be separated from a real
boundary by this method (the intact file's boundary *is* a code line), so it belongs in the
docstring as the stated limit rather than in the predicate — the module's own idiom for a limit
it cannot close.

### DE23-R2 — LOW — the extent known-bad encodes today's layout, so the next legitimate growth turns it red

`:1234-1248` asserts three things about the cut copy: the boundary is False, all three anchors
survive, **and `len(declared_limit_text(_above_head)) == len(_limit)`** — the cut text is exactly
as long as the intact block.

That equality is a fact about *today's file*: it holds because the head is the run's topmost
line, so cutting above the head removes nothing. Add a contiguous paragraph above the head — the
upward growth DE cites as the reason for a structural predicate, and which the predicate itself
handles fine — and the intact block grows while the cut copy still stops at the mutant's blank.
Driven: the suite goes **red at this known-bad**, with the predicate answering correctly
throughout.

So the round replaced an anchor set that goes stale as the block grows with a predicate that
cannot, and left its falsifier holding the same assumption the anchors did. It is the mirror of
DE21-R1, one artefact over.

**Closure:** drop the length conjunct — the property being falsified is *the boundary predicate
fires and the anchors do not*, and the length equality is a **reason** the old anchors were
fooled, not part of the claim. If the comparison is wanted, compute it against the same source
the cut was made from (`declared_limit_text(src_without_the_blank)`), which is growth-proof.

---

## Executed evidence

At `a83083a`, 2026-09-02T15:29–15:31Z, in `~/ctaNew-wt-rev` (which carries `data/pm_5min/raw`):

| check | result |
|---|---|
| scope | one file, **+90/−4**; suites 87 / 160 / 69, rc 0 both launchers |
| the intact boundary | `first_read_line 139`, `above_line 137`, `above = _REBOUND = …`, boundary **True** |
| the two DE21-R1 cuts | boundary **False** for both (3,752 identical / 1,975 truncated) |
| **four other interruptions** | 47% unread each, boundary **True** — DE23-R1 |
| prose-blindness | zero anchor tokens in the predicate's body |
| **contiguous upward growth** | suite **red** at the extent known-bad — DE23-R2 |
| the blank-below variant | red at the extent check, naming the first line read |
| hardcoded `True` | red |
| both conjunct drops | red at `_cut2` and `_cut` respectively |
| surface | `_BLIND_HEAD` / `_declaration_holds` suite-local; `declared_limit_boundary` module-level by design |
| census | `ok` 64 → 67, `refuses` 15 → 15; one `-` line, rewritten in place |
| unchanged | supply/seam **1,875**, `daw is` True, R-419 `True, [], []`, roots byte-identical to `0255b60` |
| worktree | clean at `a83083a` after every mutant; cache cleared before each run |

---

## Disposition

- **RELEASE** for `a83083a`. DE21-R1 is closed: both cut shapes now fail a predicate that knows
  nothing about the prose, and the `_declaration_holds` lift makes an edit inside the predicate
  visible to the known-bads that drive it. **No hold.**
- **RULED (item 1):** the shape is right — structural, prose-blind, both shapes covered — and the
  predicate is one token short: `#:`-ness is not "the run was not cut", and three ordinary comment
  edits pass while dropping 47% of the block.
- **RULED (item 3):** my round-21 ruling stands, refined — lifting the predicate converts
  "assertion" into "subject" for everything inside it, so the un-falsifiable surface is now the
  `ok(...)` line alone, which is where it belongs.
- **FILED:** **DE23-R1** (the predicate accepts three comment shapes it should refuse) and
  **DE23-R2** (the extent known-bad asserts a property of today's layout and will go red on the
  next legitimate growth). Both are one line each, and they touch adjacent lines.
