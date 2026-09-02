# Review — DE round 25 (DE23-R1: the boundary tests the run; DE23-R2: the falsifier stops encoding today's layout)
reviewer: claude (pm-codex seat) · round opened by the coordinator (pm-co)

**Pinned tip executed: `50a9113`** (Q-DE-43 row at `bcf0959`).
**Request of record:** `REQUEST_DE_ROUND_25_2026-09-02.md` (at `dc83580`).
**Composed 2026-09-02T15:59:00Z.** One filing, per R-377.

Executed in `~/ctaNew-wt-rev` at `--detach 50a9113` — the worktree's `data/pm_5min/raw` symlink
lets the admissible suite run in place, read-only under `data/`. Every mutant applied to the
worktree copy with `__pycache__` cleared **before** each execution (R-446) and restored; worktree
clean after. Register untouched.

Scope confirmed: `de_admissible_windows.py` only, **+90/−8**, one file in the commit;
`de_ratification_check.py` byte-identical to `e0d1e9f`. Suites: admissible **91** both launchers,
ratification 168, seam 69, rc 0.

---

## Verdict

### RELEASE for `50a9113`. Both findings close, and the two checks now cover each other's blind spots.

One finding, LOW: **the declared limit names one shape the predicate cannot see and not the
other.** An earlier line matching the anchor literal makes the reader return **zero characters**
while the predicate reports `stopped_at_a_real_boundary True` — its most misleading possible
answer. The anchors catch it, so nothing is exposed; the docstring's list is short by one.

---

## 1. DE23-R1 — the one token, taken

`:301` is now `not above.lstrip().startswith("#")`. Measured on copies of the pinned file, each a
single line inserted above the OVER-CAUGHT paragraph:

| inserted | chars read | boundary (was at `a83083a`) |
|---|---|---|
| `# a plain comment` | 1,975 | **False** (True) |
| `  #: an indented continuation` | 1,975 | **False** (True) |
| `#` (bare) | 1,975 | **False** (True) |
| `X = 1` | 1,975 | True — the declared limit |

Intact: 3,752 chars, `first_read_line 139`, `above_line 137`, above `_REBOUND = "<rebound-import>"`,
boundary True. The predicate remains prose-blind — its body contains no anchor text.

**The false-refusal question, ruled: the right conservative answer, not a finding.** I drove the
case the request names — the boundary line itself commented out (`# _REBOUND = …`) — and the
predicate reports `boundary False` with the block **intact** (3,752 chars, all three anchors
present). That is a refusal where nothing was lost. It is still right: with that line commented,
the comment run genuinely extends past the block's head, so the reader is stopping inside a
comment region and the block's extent is no longer determined by the file's structure. The
message names the exact line, the direction is safe (a red on a legitimate edit, never a green on
a cut), and the author's fix is one line. A predicate that answered "fine" there would be
asserting something it cannot know.

## 2. Ruling — the declared limit is honest, real, and one shape short

**Honest and real:** `X = 1` inserted into the run reads 1,975 chars with `boundary True`. The
docstring says so plainly and says why (the intact boundary *is* a code line), rather than
chasing it.

**And it costs nothing today, which is the part worth measuring.** I drove the code-line cut in
three positions:

| where the code line lands | chars | boundary | anchors missing |
|---|---|---|---|
| above the head | 3,752 (nothing lost) | True | 0/3 |
| above `REFUSED` (mid-block) | 3,621 | True | **1/3** |
| above OVER-CAUGHT | 1,975 | True | **3/3** |

So every position in which a code line actually removes content loses at least one anchor, and
the anchors check fires; the only position the extent predicate misses removes nothing. DE's
sentence — *"a code line that cut the run above the head would leave the anchors intact and this
limit is real"* — is exactly right, and the composition is stronger than it claims: the extent
predicate covers comment cuts, the anchors cover code cuts, and between them no cut that loses
content is silent.

**The completion, and it is the finding.** That composition holds because all three anchors sit at
the top of the block — `THE DECLARED LIMIT` is its first line. A future section added *above* the
head would sit above every anchor, and a code line cutting between the two would be invisible to
both checks. Worth a sentence in the same docstring, alongside the shape below.

**A shape the limit does not name — DE25-R1.** The reader anchors on the first line starting with
`DECLARED_BLIND_SHAPES = (`. A line matching that literal placed **earlier** in the file (I put
one at column 0 at the end of the module docstring) redirects the walk: the reader returns
**0 chars** and the predicate reports **`boundary True`**. The anchors check catches it (3/3
missing), so the suite is red and nothing is exposed — but the predicate's answer on a read of
nothing is "a real boundary", which is the one answer a reader must not take at face value.

## 3. DE23-R2 — the length conjunct is gone, and growth is controlled both ways on one source

`== len(_limit)` appears **zero** times in the module; `len(_limit)` survives only in printed
messages. The positive control at `:1297` uses one source and drives both directions — reproduced:

```
a contiguous paragraph above the head : 3,797 chars (intact 3,752), boundary True   -> GREEN
the same copy with a blank between    : 3,752 chars,                boundary False  -> refuses
```

The property falsified is now "the boundary fires and the anchors do not", with no equality
against today's length anywhere — which is what DE23-R2 asked for, and the round-23 known-bad
that went red on legitimate growth is gone.

## 4. Six mutants — and the ordering hides nothing

| mutant | result |
|---|---|
| the **old** token restored (`startswith("#:")`) | **red** at the new DE23-R1 known-bad, naming the plain-`#` shape |
| the predicate hardcoded `True` | **red** at the extent known-bad |
| a plain-comment cut applied to the **file** | **red at the ANCHORS check**, which comes first |
| the same cut with the **anchors check neutralised** | **still red** — at a known-bad whose own fixture the cut has broken |
| a blank above the head | **red** at the extent check, naming line 141 |
| **contiguous growth (the control)** | **green at 91** |

**Ruled: the ordering hides nothing.** Both checks see those cuts independently — I measured
`boundary False` for the plain-comment and indented-continuation cuts directly, so the extent
check's verdict on the damaged file is a refusal in its own right; the anchors simply appear
earlier in the suite. And with the anchors check neutralised the suite is still red. DE naming
which check catches which is the honest description, not a hedge.

## 5. Counts and census — nothing removed, nothing pinned

87 → **91**, `EXPECTED_CHECKS = 91`. AST census `a83083a` → `50a9113`: `ok` 67 → **69**,
`refuses` 15 → 15, and **zero** check-call lines removed — the length conjunct came out of an
existing check rewritten in place. No block length is asserted anywhere.

`ROOT` / `MASK_DIR` untouched (zero diff hits), so the DATA_ROOT split stays behind DA's landing;
supply **1,875**; seam **1,875**; `seam.daw is de_admissible_windows` True; R-419 on the real
register `verified_for_new_run True, unverifiable [], superseded_by []`; `decides: "nothing --
this reports…"`.

## 6. Discipline

One file in the commit (`live/pm_research/de_admissible_windows.py`, +90/−8) — a pathspec commit,
confirmed from `git show --stat`. Nothing under `data/`. The line numbers the request cites
resolve at the pin (`:279` the declared limit, `:301` the token, `:1297` the growth control,
`:1323` the three-shape loop, `:839` `EXPECTED_CHECKS`).

---

## Findings

### DE25-R1 — LOW — the declared limit names the code-line shape and not the anchor collision

`declared_limit_boundary` locates the block by `next(n for n, ln in enumerate(lines) if
ln.startswith(_LIMIT_ANCHOR))` — the **first** line starting with `DECLARED_BLIND_SHAPES = (`.
Driven: with such a line inserted at column 0 earlier in the file (inside the module docstring),

```
declared_limit_text  -> 0 chars
declared_limit_boundary -> stopped_at_a_real_boundary: True, first_read: 'DECLARED_BLIND_SHAPES = (  # quoted …'
```

The predicate certifies a real boundary on a block it did not read. The anchors check catches the
consequence (all three missing), so the suite goes red and no reader is misled in practice — this
is a completeness point about the stated limit, not an exposure.

But the docstring's limit paragraph is what a future maintainer will read before trusting the
predicate, and it names one blind shape (`X = 1`) while this one is worse in kind: the code-line
case returns a truncated but real block, this one returns nothing while answering True.

**Closure:** name it in the same paragraph — the reader anchors on the first match of a literal,
so a same-text line earlier in the file redirects it — and, if a check is wanted rather than a
sentence, have the boundary report refuse an empty read (`first_read_line == the anchor's line`
means the walk read nothing), which is one comparison and needs no knowledge of the prose. The
same paragraph is also the right place for the composition's own condition: the anchors cover
code cuts only while they remain the block's topmost content.

---

## Executed evidence

At `50a9113`, 2026-09-02T15:56–15:59Z:

| check | result |
|---|---|
| scope | one file, **+90/−8**; ratification byte-identical to `e0d1e9f`; suites 91 / 168 / 69 |
| the three comment shapes | boundary **False** for all three (True at `a83083a`), 1,975 chars each |
| `X = 1` | boundary True — the declared limit, reproduced |
| the boundary line commented out | boundary **False** with the block **intact** — the conservative refusal |
| code-line cuts, three positions | anchors missing 0 / 1 / 3 — every content-losing position is caught |
| **the anchor collision** | **0 chars read, boundary True** — DE25-R1 |
| the growth control | 3,797 chars boundary True (green) / 3,752 boundary False (refuses), one source |
| `== len(_limit)` | **zero** occurrences |
| six mutants | old token, hardcoded True, plain-comment cut, cut with anchors neutralised, blank above head → red; contiguous growth → **green at 91** |
| census | `ok` 67 → 69, `refuses` 15 → 15, **0 removed** |
| unchanged | roots untouched, supply/seam **1,875**, `daw is` True, R-419 `True, [], []`, `decides: nothing` |
| worktree | clean at `50a9113`; cache cleared before every mutant |

---

## Disposition

- **RELEASE** for `50a9113`. DE23-R1 closes with the one token, and the three comment shapes that
  passed at `a83083a` now refuse; DE23-R2 closes with the length conjunct gone and growth
  controlled both ways from one source, so the falsifier no longer encodes today's layout. **No
  hold.**
- **RULED (item 1):** the new token's false refusal — a commented-out boundary line reads as a cut
  — is the right conservative answer, not a finding: the comment run genuinely extends past the
  block's head there, the message names the line, and the error direction is safe.
- **RULED (item 2):** the declared limit is honest and real, and it costs nothing today — every
  code-line cut that removes content loses an anchor. It is **one shape short** (DE25-R1), and its
  composition with the anchors holds only while the anchors remain the block's topmost content.
- **RULED (item 4):** the ordering hides nothing — both checks see those cuts independently, and
  with the anchors check neutralised the suite is still red.
- **FILED:** **DE25-R1** (the anchor collision, unnamed in the limit).
