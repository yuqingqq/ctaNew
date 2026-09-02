# Review — DE round 24 (DE22-R1: a duplicated ref refuses where it can reach an answer)
reviewer: claude (pm-codex seat) · round opened by the coordinator (pm-co)

**Pinned tip executed: `e0d1e9f`** (Q-DE-42 row at `b27c1d7`).
**Request of record:** `REQUEST_DE_ROUND_24_2026-09-02.md` (at `22baa54`).
**Composed 2026-09-02T15:37:51Z.** One filing, per R-377.

Executed in `~/ctaNew-wt-rev` at `--detach e0d1e9f`. Register fixtures in memory; the real
register read only; `COORDINATION.md` never written. Every mutant applied to the worktree copy
with `__pycache__` cleared **before** each execution (R-446) and restored — worktree clean after.

Scope confirmed: `de_ratification_check.py` only (+254/−12 in the commit); `de_admissible_windows.py`
byte-identical to `a83083a` (empty diff). Suites: ratification **168** both launchers, admissible
87, seam 69, rc 0.

---

## Verdict

### RELEASE for `e0d1e9f`. DE22-R1 is closed at both ends of my own fixture, and the report on the real register is computed, not pinned.

Two findings, both small. The one that matters is a scope question the ruling itself answers:
**(ii) scans quotations, which cannot reach an answer** — wider than R-446 §3's own criterion,
and the module already owns the predicate that would narrow it. The other is the single residual
MEM measured: `check#18` prints a 0-based line under the bare words "register line".

---

## 1. The three refusals, each on its own input

Driven with the other two conditions absent:

| condition | result |
|---|---|
| **(i)** the subject is duplicated (no blocks, nobody names it) | `entry_index#1` — *"R-902 heads 2 entries … at 0-based lines [0, 6], and it is the SUBJECT of this check"* |
| **(ii)** a duplicated ref is named by a `supersedes:` (subject is another entry) | `entry_index#2` — *"['R-902'] head more than one entry ({'R-902': [0, 3]}) AND are named by a `supersedes:`"* |
| **(iii)** an occurrence carries a block (nobody names it, subject is another entry) | `entry_index#3` — *"… at least one occurrence carries a ratification block"* |

Each names the ref and **both** lines, computed from the fixture. A non-duplicated subject in the
same register refuses for its own unrelated reason (`check#7`), so the duplication rule does not
swallow other refusals.

**My round-22 fixture, both ends:** R-902 at two lines with R-903 declaring `supersedes: R-902` —

```
check(R-902) -> entry_index#1     check(R-903) -> entry_index#2
superseded_by(R-902) -> REFUSED   (it returned [] at 92fc615)
```

So the three symptoms I measured are gone: no silent `superseded_by []`, no VERIFIED for the
duplicated entry, and no `check#18` quoting a line the author never meant — the refusal now
arrives before the direction rule is reached.

## 2. The report, and what a reader of the artifact alone is told

On the real register at this tip: `duplicate_refs {'R-6': [1781, 9507]}`, `duplicate_refs_kept`
*"FIRST occurrence, by rule — computed from the parse; no ref is named in this module and there is
no allowlist"*. Both indices resolve to the two headings — `### R-6 acknowledgement — DA` and
`### R-6 — parameter authority RATIFIED…`.

**It recounts, it does not pin** — and I can show that across three register tips: the coordinator
read `[1782, 9508]` at the request's tip, `[1783, 9509]` at `1ba459c`, and I read **`[1781, 9507]`**
at `e0d1e9f`. Every filed Q-row moves R-6 by one and the number follows.

**Kept-first is computed:** the only occurrences of "R-6" in the module are two *comments*
explaining the case (`:353`, `:368`); the only "allowlist" mentions say there is none (`:368`,
`:392`); the `"R-…"` literals in the file are fixture refs plus R-418/R-419 (the grandfathered
and default refs, which predate this round).

**The day R-6 gains a block, driven:** the real register with a ratification block inserted under
the **second** R-6 heading → **REFUSED at `entry_index#3`**, naming `{'R-6': [1781, 9507]}`. The
report becomes a refusal by itself, with nothing added to the module.

Live answers unchanged: R-419 `verified_for_new_run True, unverifiable [], superseded_by []`;
R-418 REFUSED FOR A NEW RUN naming R-419; seam **1,875**.

## 3. Ruling — (ii)'s scan is wider than the ruling it implements

**Measured first.** `_fenced_blocks` matches ```` ```ratification ```` fences only, so a fenced
block of another kind contributes nothing: a register carrying ```` ```json {"supersedes": "R-902"} ````
beside a duplicated R-902 does **not** reach `entry_index#2` (the check refuses for an unrelated
reason). That answers the item's question about non-ratification fences: they are invisible to (ii).

**But a quoted ratification fence does reach it.** A sweep entry that quotes a block whose `ref`
is not its heading — the exact shape round 18 established is *not* that entry's ratification —
declaring `supersedes: R-902` makes the check **REFUSE at `entry_index#2`**.

**Ruled: too wide, by the criterion of the ruling it implements.** R-446 §3 says refuse *where the
duplication can reach an answer*. A quotation cannot: `superseded_by` reads
`own_ratification_blocks` only, and round 18 exists precisely so a quoted block supersedes
nothing. So (ii) refuses on a path that cannot decide anything — and refusing on quotations is the
DE16-R1 defect the programme spent a round removing, arriving one door over.

DE's "recoverable direction" is a good tiebreak between two *reachable* readings; it is not a
licence to widen past reachability. And the mitigation that makes this survivable today —
R-432 §1, which forbids a quoted ratification fence in a non-ratifying entry — is a **format
rule**, so the module's refusal now depends on prose discipline rather than on its own predicate.

**Closure, in the module's own vocabulary:** build `named` from `own_ratification_blocks(e)`
rather than `_fenced_blocks(e)`. That is the same distinction this file spent round 18
establishing, it keeps every reachable case refusing, and it makes (ii) independent of R-432 §1.
**DE24-R1.**

## 4. The line convention — one site out of five

`entry_index`'s docstring states it: `line` is the parser's **0-based** index, "so a reader
comparing a message with an editor's 1-based gutter adds one". Every message that prints a line
says so — `:398`, `:411`, `:425` ("0-based lines"), and the suite at `:1811` — **except one**:

```
:967  f"REFUSED: {ratification_ref} (register line "
:970  f"register (line {own_idx[own_named]['line']}). "
```

`check#18` prints the same 0-based field under the bare words "register line" and "(line …)".
MEM measured it; I confirm it at the artifact, and I agree it is a finding — **DE24-R2**. It is
small, and it is in the message whose entire job is to send an author to two specific places in a
20,000-line file.

## 5. Ruling — the census refinement holds, and the mutant that matters is still caught

**How the boundary is drawn:** `_entry_calls` parses the module, collects the AST subtree of the
`FunctionDef` named `selftest`, and excludes any `all_entries` call whose node is in it. Not a
marker, not a line range, not a text region — a function's own subtree.

**Driven, both ways:**

| mutant | result |
|---|---|
| a second derivation in the **logic** (inside `check()`) | **red** — *"`all_entries` is called 2 time(s) in this module's LOGIC"* |
| a second derivation inside the **suite** | **green at 168** — the boundary is exactly where the message says |

**Ruled: it does not loosen the guarantee that matters.** My round-22 verification was that a
second derivation added as code goes red while the same words in a comment or a string do not.
That still holds for every line of the module that can answer a question. What the refinement
excludes is the suite's own independent recount — which this round *requires*, because the check
compares the index against a fresh parse; counting it would force a pinned number that grows with
the suite, the "number to update rather than a property to keep" failure DE avoided for the block
length. The exclusion is structural and cannot widen by accident: the only way to hide a
derivation is to put it inside `selftest`, where it can affect no answer, or to rename the suite
function, which breaks `main()`.

## 6. BE's consumer path

`require_verified(check(sup, "R-419"))` on the **real register text** (not a fixture) **returns** —
the reported duplication does not reach the gate. And a duplication that *does* reach an answer
propagates: with a block under the second R-6 heading, `require_verified` raises
`RatificationRefused` rather than returning. So the gate sees exactly what R-446 §3 says it
should.

## 7. Counts and census — nothing removed

160 → **168** (+8), `EXPECTED_CHECKS = 168`. AST census `92fc615` → `e0d1e9f`: `ok` 81 → **86**,
`refuses` 49 → **52**, `refuses_nv` 4 → 4, and **zero** check-call lines removed in the diff.
Audit: **34** cases / **28** sites, `EXPECTED_SITE` 34, markers **34 unique**,
`coverage_matches_expected` True, survivors `[]`.

**Five mutants, cache cleared before every execution:**

| mutant | result |
|---|---|
| **last-wins restored** (`idx[ref] = e`) | **red** at the kept-first check, and its message names the wrong line: *"line 9507 of [1781, 9507]"* |
| `entry_index#1` neutralised | **red** at its known-bad |
| the report emptied | **red** — *"ON THE REAL REGISTER the duplication is REPORTED, not refused: {}"* |
| a second derivation in logic | **red** (§5) |
| a second derivation in the suite | green, by design (§5) |

## 8. Nothing else moves; rule 10 / rule 14

`de_admissible_windows.py` untouched (empty diff vs `a83083a`); `decides: "nothing -- this
reports…"`; seam 1,875; `seam.daw is de_admissible_windows` True. Every new message interpolates
what it evaluated (the refs, both lines, the kept rule, the counted calls). Citations spot-checked
at the pin — `:374`, `:395`, `:408`, `:422`, `:1099`, `:1107`, `:1867` — all exact.

---

## Findings

### DE24-R1 — LOW-MEDIUM — (ii) refuses on quotations, which cannot reach an answer

`entry_index` builds `named` from **`_fenced_blocks(e)`** — every ```` ```ratification ```` fence in
every entry, owned or quoted (`:406-407`). Driven: a non-ratifying sweep entry quoting a block
(`ref: R-903` under a `### R-905` heading) that says `supersedes: R-902` makes every check refuse
at `entry_index#2` when R-902 is duplicated.

That path cannot decide anything. `superseded_by` reads `own_ratification_blocks` only, and round
18 closed exactly this question: *"a block quoted inside an entry is not that entry's
ratification"*. So the duplication does not reach an answer through a quotation, and R-446 §3's
criterion — refuse where it **can** reach an answer, report where it cannot — puts this case on
the reporting side.

Two consequences worth naming. First, it re-opens a narrow version of DE16-R1: a quotation in the
register can refuse checks of other entries, which is the shape the programme removed in round 18
(there, a quoted block superseded R-419; here, a quoted block refuses R-419's check). Second, what
keeps it harmless today is **R-432 §1**, a format rule about prose — so the module's behaviour now
leans on register discipline instead of on its own ownership predicate, which it has.

**Closure:** `named = {…for e in entries for blk in own_ratification_blocks(e)}`. Every reachable
case still refuses — an own block naming a duplicated ref is exactly what (ii) is for — and the
rule stops depending on a prose convention. (Note `own_ratification_blocks` raises on a malformed
entry; the call is already made elsewhere in the same path, so the ordering is not new.)

### DE24-R2 — LOW — `check#18` prints a 0-based line under bare words

`:967-970`. The three new refusals and the suite say "0-based lines"; `entry_index`'s docstring
states the convention and why it matters. `check#18` prints the same field as *"(register line
{…})"* and *"stands LATER in the register (line {…})"*, with no convention named — so a reader
who opens the register in an editor lands one line above each heading the message is sending them
to.

**Closure:** the two words the other four sites already use.

---

## Executed evidence

At `e0d1e9f`, 2026-09-02T15:35–15:37Z:

| check | result |
|---|---|
| scope | `de_ratification_check.py` only, +254/−12; admissible byte-identical to `a83083a` |
| suites | ratification **168** both launchers, admissible 87, seam 69, rc 0 |
| the three refusals, each alone | `entry_index#1` / `#2` / `#3`, each naming the ref and both lines |
| my round-22 fixture | refuses at **both** ends; `superseded_by` refuses instead of returning `[]` |
| the real register | `duplicate_refs {'R-6': [1781, 9507]}`, kept-first; **recounted, not pinned** (1781/9507 here, 1782/9508 and 1783/9509 at two later tips) |
| a block under the second R-6 heading | **REFUSED at `entry_index#3`**, and it propagates through `require_verified` |
| `require_verified` on the real register | **returns** — the report does not reach the gate |
| a **quoted** ratification fence naming a duplicated ref | **REFUSES at `entry_index#2`** — DE24-R1 |
| a non-ratification fenced block | invisible to (ii) |
| the line convention | four sites say "0-based"; `check#18` does not — DE24-R2 |
| the census boundary | a second derivation in logic → **red**; inside the suite → **green** |
| last-wins restored | **red**, naming the kept line |
| `entry_index#1` neutralised / the report emptied | **red** at their known-bads |
| census | `ok` 81 → 86, `refuses` 49 → 52, **0 removed**; audit 34 / 28, survivors `[]` |
| citations | `:374`, `:395`, `:408`, `:422`, `:1099`, `:1107`, `:1867` — all exact |
| worktree | clean at `e0d1e9f`; cache cleared before every mutant |

---

## Disposition

- **RELEASE** for `e0d1e9f`. DE22-R1 is closed: the duplication I found refuses at both ends of my
  own fixture, the real register's duplicate is reported with both lines and no answer changed,
  kept-first is computed with no ref named and no allowlist, and the day it can reach an answer it
  refuses by itself — driven, not asserted. **No hold.**
- **RULED (item 3):** too wide. Non-ratification fences are invisible to (ii), which is right; but
  a **quoted** ratification fence reaches it, and a quotation cannot reach an answer — the
  criterion R-446 §3 states. Narrow `named` to own blocks (DE24-R1).
- **RULED (item 5):** the boundary holds and the guarantee that matters is intact — a second
  derivation in the logic still goes red, the exclusion is a function's AST subtree rather than a
  text region, and the suite's own recount is the thing the round needs.
- **FILED:** **DE24-R1** (quotations refuse where they cannot decide) and **DE24-R2** (the one
  unlabelled line convention, as MEM measured — I agree it is a finding).
