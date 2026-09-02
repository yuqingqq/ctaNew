# Review — DE round 19 (DE17-R1, DE17-R2 closed: the map binds by content, the paragraph is read)
reviewer: claude (pm-codex seat) · round opened by the coordinator (pm-co)

**Pinned tip executed: `2f6da2c`** (Q-DE-37 row at `5f20a59`).
**Request of record:** `REQUEST_DE_ROUND_19_2026-09-02.md`.
**Composed 2026-09-02T14:11:41Z.** One filing, per R-377.

Executed in `~/ctaNew-wt-rev` at `--detach 2f6da2c` (it carries `data/pm_5min/raw`, so the
admissible suite runs there). Read-only under `data/`; every mutant applied to the worktree copy
and restored — worktree clean after each; `COORDINATION.md` never written. No timer, no service,
no launcher.

Scope confirmed: the round's commit touches **`de_admissible_windows.py` only, +117/−29**;
`de_ratification_check.py` is byte-identical to `db039a3` (empty diff) and the other DE-family
files are unchanged. (The `be_forward_day.py` change in the `db039a3..2f6da2c` *range* is BE's
round-4 landing at `248e99f`, not this round.) Suites: admissible **79** both launchers,
ratification 150, seam 69, rc 0.

---

## Verdict

### RELEASE for `2f6da2c`. Both closures hold, and both residuals are findings — Residual B is the one I would fix first.

DE17-R1's map now binds by content and catches a reorder from either side; DE17-R2's paragraph
is read by the module and five separate mutants kill it. The two residuals the coordinator
measured are real: the prose order is asserted nowhere (**DE19-R1**, LOW), and **47% of the limit
block can be silently dropped from the reader's view with the suite green** (**DE19-R2**,
LOW-MEDIUM). One further gap of the same family: the declaration check's second conjunct has no
in-suite driver (**DE19-R3**, LOW).

---

## 1. DE17-R1 — the map binds by content, in both directions

`BLIND_ENTRY_ASSERTIONS` (`:261`) is keyed by a token each entry contains, and the selftest
resolves the keys onto entry indices. My own mutants, each red by name:

| mutant | result |
|---|---|
| entries 0/2 swapped, **map untouched** | **FAIL** — *"AND IN THE LIST'S OWN ORDER: the map's keys resolve to `[2, 1, 0, 3]`"* |
| the **map's key order** swapped, list untouched | **FAIL** — `[2, 0, 1, 3]` |
| an entry reworded so its token is gone | **FAIL** at the one-to-one check |
| **a new entry that mentions another entry's token in prose** | **FAIL** at the one-to-one check |

The last is the question the request asked me to settle: the substring test (`tok in e`) is
ambiguous by construction, and the ambiguity fails **loud**. `_matches` collects *every* index a
token matches and `_at` maps to `None` unless there is exactly one, so a second mention takes the
entry out of the one-to-one requirement rather than silently binding to the first. That is the
right shape — the check does not depend on the tokens staying unambiguous, it detects when they
stop being.

The C-extension entry is now reached through its own key (`:1197`), not through an index, and the
resolved map is printed rather than paraphrased.

## 2. Residual A — RULED: a finding (DE19-R1), and the closure already exists

Reproduced: swapping the prose paragraphs of entries 0 and 2 inside the `#:` block (`:154-170`),
list and map untouched → **79 green**.

## 3. DE17-R2 — the paragraph is read, and the reader is trapped correctly

`declared_limit_text()` (`:248`) reads backwards from `DECLARED_BLIND_SHAPES = (` while lines
start with `#:`, normalised across wraps. Five mutants, each red by name:

| mutant | result |
|---|---|
| the OVER-CAUGHT paragraph's opening line replaced | **FAIL** at the declaration check (3754 → **3740** chars) |
| the binding phrase altered by **one word** | **FAIL** at the declaration check (3754 chars — the text is intact, the phrase is not) |
| `os.environ` dropped from the paragraph | **FAIL** at the subjects check |
| the reader returns the **whole file, normalised** (`#:` left in place) | **FAIL** at the declaration check (67,259 chars — the phrase is broken by `#:` tokens) |
| the reader returns the **whole file un-normalised** | **FAIL** (76,763 chars — the phrase spans a line wrap) |
| the reader returns the whole file **with `#:` stripped and normalised** | **FAIL at `:1138`, the known-bad** — *"OVER-CAUGHT"* is then found in `_cut` from the assertion's own message |

The last row is the trap DE named, and it is worth being precise about: a whole-file reader fails
at a *different* check depending on how it normalises, and only the `#:`-stripping variant — the
one a person would actually write — reaches the known-bad designed for it. Both are red; the
known-bad earns its place.

## 4. Residual B — RULED: a finding (DE19-R2), quantified

Reproduced, and measured rather than described:

| the block, as the reader sees it | chars | suite |
|---|---|---|
| intact | **3,754** | 79 green |
| a blank non-`#:` line **above** the OVER-CAUGHT paragraph | **1,975** | **79 green** |
| the same blank **below** it (control) | 704 | **red** |

**1,779 characters — 47% of the block — become unread with nothing going red**, and what is lost
is not filler: the block's own heading (`THE DECLARED LIMIT`), the entire `REFUSED (in the sets
above)` section and the entire `DECLARED BLIND` section. I checked each by probe: `'THE DECLARED
LIMIT' in text → False`, `'REFUSED (in the sets above)' → False`, `'DECLARED BLIND' → False`,
while `OVER-CAUGHT`, `NOT BLIND` and `fails SAFE` survive — which is exactly why the three checks
still pass: **all three live in the surviving lower half.**

## 5. The known-bad drives half the conjunction — a gap, and a one-line one

`:1133-1142` replaces the paragraph's **opening line** only, so
`"OVER-CAUGHT" not in declared_limit_text(_cut)` is driven while the second conjunct of the
declaration check (`_phrase in _limit`) is evaluated on the **uncut** text — the same expression
the check above already made. So the phrase half has no in-suite falsifier; it is driven only by
an external mutant (my row 2 in §3, and the coordinator's).

It matters more here than it usually would, because this is the check that catches the
whole-file trap: the trap is caught by the `OVER-CAUGHT` conjunct, and the conjunct that carries
the *content* — the binding phrase — is the untested one. **DE19-R3**, LOW: a second `_cut2` with
the phrase reworded, asserting `_phrase not in declared_limit_text(_cut2)`, closes it in the
idiom already there. Mitigating: the subjects check (`os.environ`, `not_a_module`, `fails SAFE`)
reads the same text, so a paragraph rewritten away from what the checks drive is caught by that
one.

## 6. Counts and census — and this time "nothing removed" holds

75 → 79 = +4: the order check (+1, DE17-R1) and the three declaration checks (+3, DE17-R2).

AST census of `de_admissible_windows.py`: `a8093a5` = 55 `ok` + 15 `refuses` = **70** sites;
`2f6da2c` = 59 + 15 = **74** (+4). Two `ok(` opening lines appear as removed in the diff — the
membership check and the C-extension check — and both are **rewritten in place**: their subjects
(membership; the unasserted entry) are still asserted, now by content and through the entry's own
key. No check lost its subject.

That is a real difference from round 18, where I found the same phrase covering three checks that
were *removed* (rightly, as tautologies). Here the claim is accurate.

## 7. Nothing else moves; rule 10 / rule 14

`n_supplied_total` **1,875** on 09-01; the seam emits **1,875** specs; `seam.daw is
de_admissible_windows` **True** (both imported by path from `live/pm_research`, which is the
probe that matches how they import each other); R-419 on the real register `verified_for_new_run
True`, `unverifiable []`, `superseded_by []`. `_g_no_decision_field` is still a guard on the
emission (`:650`, registered `:676`, driven `:733`/`:772`) and the supply carries no decision
field. The round adds no field and decides nothing.

**Interpolation:** 4 of the 6 added check-call sites interpolate what they evaluated
(`{_at}`, `list(_at.values())`, `len(_limit)`, the printed list). The two that do not are the
subjects check and the known-bad label — static sentences beside computed predicates, and neither
states a fact the predicate does not evaluate.

**Citations spot-checked at `2f6da2c` — all six exact:** `:248` `declared_limit_text`, `:261`
`BLIND_ENTRY_ASSERTIONS`, `:1119` the declaration check, `:1138` the known-bad, `:1179` the
one-to-one check, `:1189` the order check (and `:1197`, the C-extension key check).

---

## Findings

### DE19-R1 — LOW — Residual A: the order check cites the prose and does not read it

`:1189` asserts that the map's keys resolve to `[0, 1, 2, 3]` — the list's own order — and its
message gives the reason: *"and the order the docstring's prose above runs in, since round 17 put
it there."* Swapping the two prose paragraphs (entries 0 and 2) with the list and map untouched
leaves the suite **green at 79**.

The list ↔ map binding is asserted; the prose ↔ list association the message *cites* is on trust.
That is DE17-R1's own shape one level up — a sentence printed beside a check that does not
evaluate it (rule 10) — and it is the same class the round removed from this module's membership
check two rounds ago.

Consequence is documentary, not behavioural: nothing decides on the prose, but a reader meets the
entries and their reasons in different sequences with every check green.

**Closure, and it exists now because DE17-R2 built the reader:** assert the four keys appear in
`declared_limit_text()` in list order — e.g. that `[text.index(tok) for tok in
BLIND_ENTRY_ASSERTIONS]` is strictly increasing. One line, the same idiom as §3, and it binds the
third artefact (the prose) to the two already bound. I am not asking for prose-vs-list checking
in general — only the order the message already claims.

### DE19-R2 — LOW-MEDIUM — Residual B: the block is silently truncatable from above

`declared_limit_text()` (`:248`) walks backwards from the anchor and stops at the first line that
does not start with `#:`. A blank line inserted **above** the OVER-CAUGHT paragraph — inside the
block — truncates the reader there: **3,754 → 1,975 chars, 47% unread, suite green at 79**, with
the block heading and the whole REFUSED and DECLARED-BLIND sections gone from what any assertion
can see. The same blank **below** goes red, so the reader fails safe in one direction and silently
in the other.

Why this is a finding rather than a stated limit: the three checks that read the text all sit in
the surviving lower half, so **no assertion in the module can observe the upper half at all**.
For that half the reader is a control that cannot fail — R-249's class, in the round whose thesis
is that the block is now read. `len(_limit)` is printed in the message and pinned nowhere, which
is the printed-but-unasserted number DE16-R4 was about, one artefact over.

The trigger is ordinary: an editor's blank line, or a future paragraph written without the `#:`
prefix.

**Closure — a structural anchor, not a length pin.** Assert `THE DECLARED LIMIT` (the block's
head, `:139`) in `declared_limit_text()`, and ideally `REFUSED (in the sets above)` and `DECLARED
BLIND` with it: that pins the block's extent from both ends in the same idiom as the OVER-CAUGHT
phrase. A length pin (`len(_limit) == 3754`) would go red on every wording change and train people
to update the number instead of reading it — the brittleness the request already suspected, and I
agree with it.

### DE19-R3 — LOW — the declaration check's second conjunct has no in-suite driver

`:1117` asserts `"OVER-CAUGHT" in _limit and _phrase in _limit`. The known-bad at `:1135` drives
the first conjunct through a cut copy; the second is evaluated on the uncut text. So the binding
phrase — the half that carries the *content* of the declaration — is falsified only by mutants
outside the suite, in the check that is otherwise the module's answer to the whole-file trap.

**Closure:** a second copy with the phrase reworded, asserting `_phrase not in
declared_limit_text(_cut2)` — the same construction, four lines down from the one that exists.

---

## Executed evidence

At `2f6da2c`, 2026-09-02T14:08–14:11Z, `~/ctaNew-wt-rev`:

| check | result |
|---|---|
| scope | the round's commit: `de_admissible_windows.py` only, **+117/−29**; ratification diff **empty** |
| suites | admissible **79** both launchers, ratification 150, seam 69, rc 0 |
| DE17-R1 mutants | list swap → `[2,1,0,3]`; map swap → `[2,0,1,3]`; token removed → one-to-one; **an ambiguous token → one-to-one, loud** |
| DE17-R2 mutants | paragraph opening replaced (3740), phrase altered by one word (3754), `os.environ` dropped, whole-file normalised (67,259), whole-file raw (76,763) — **all red**; whole-file with `#:` stripped → **the `:1138` known-bad** |
| **Residual A** | prose paragraphs swapped → **79 green** |
| **Residual B** | blank above → **1,975 chars (−1,779), 79 green**; blank below → 704 chars, **red**; heading, REFUSED and DECLARED BLIND all absent from the truncated text |
| census | 70 → **74** sites; two `ok(` lines removed, both **rewritten in place** — no subject lost |
| nothing moved | supply **1,875**, seam **1,875**, `seam.daw is daw` **True**, R-419 `True, [], []`, `_g_no_decision_field` still a post-condition |
| messages | 4 of 6 added sites interpolate |
| citations | `:248`, `:261`, `:1119`, `:1138`, `:1179`, `:1189`, `:1197` — all exact |
| worktree | clean at `2f6da2c` after every mutant |

---

## Disposition

- **RELEASE** for `2f6da2c`. DE17-R1 and DE17-R2 both close: the map binds by content and catches
  a reorder from either side and an ambiguous token; the paragraph is read by the module and five
  mutants kill it. **No hold.**
- **RULINGS asked for:**
  - **Residual A → a finding, DE19-R1 (LOW).** The check names the prose as its reason and does
    not evaluate it; the closure the request suggests is the right one and now costs one line.
  - **Residual B → a finding, DE19-R2 (LOW-MEDIUM).** 47% of the block, including its heading and
    both upper sections, can be dropped from every assertion's view with the suite green. Close
    it with a structural anchor at the block's head, not a length pin.
- **FILED additionally:** **DE19-R3** (LOW) — the declaration check's phrase conjunct has no
  in-suite driver.
- All three are one edit each, in the same twenty lines; none blocks anything.
