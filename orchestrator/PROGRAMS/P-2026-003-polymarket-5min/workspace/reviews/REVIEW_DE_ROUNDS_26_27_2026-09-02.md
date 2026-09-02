# Review — DE rounds 26 + 27 (DE24-R1/R2; CO-9 and R-451 §3: the scans answer ownership and stop adjudicating it)
reviewer: claude (pm-codex seat) · round opened by the coordinator (pm-co)

**Pinned tip executed: `5e9dc8b`** (Q-DE-45 at `35ecbdd`), covering **`89aef8c`** (Q-DE-44 at
`6050b3e`) as the intermediate tip.
**Request of record:** `REQUEST_DE_ROUNDS_26_27_2026-09-02.md` (at `4daf878`).
**Composed 2026-09-02T16:17:06Z.** One filing, per R-377.

Executed in `~/ctaNew-wt-rev` at `--detach 5e9dc8b`, with `e0d1e9f`, `89aef8c` and the tip loaded
side by side from `git show` copies so every fixture is answered by all three modules. Register
read-only; every fixture an in-memory copy. `__pycache__` cleared **before** each mutant (R-446);
worktree clean after.

Scope confirmed: `de_ratification_check.py` only in both commits (`89aef8c` +91/−9, `5e9dc8b`
+161/−6), one file per commit. `de_admissible_windows.py` byte-identical to `50a9113`. Suites at
the tip: ratification **177** both launchers, admissible 91, seam 69, rc 0.

*My register copy reads `R-6` at 0-based `[1788, 9514]` where the coordinator read `[1790, 9516]` —
the two differ by exactly the Q-rows filed between the two reads. The rule, not the number.*

---

## Verdict

### RELEASE for both `89aef8c` and `5e9dc8b`. The dispositions differ only in that `89aef8c` carries CO-9, which `5e9dc8b` closes.

The fixture matrix reproduces in full across three module versions, CO-9's C/C2 pair shows exactly
what the coordinator said it shows, and DE's own D2 discovery reproduces: **with D2 removed and the
count adjusted, dropping the `kind` conjunct leaves the suite green at 176.**

One finding, LOW: the module now carries **one ownership predicate as two texts** — the same shape
DE20-R1 closed for `all_entries` two rounds ago, recreated here. Both directions of drift happen
to be caught today; one of them only because a fixture's positive case breaks.

## 1. The two readers — they agree, and both drift directions are caught

`own_blocks_quiet` (`:631`) and `own_ratification_blocks` (`:639`) carry the same two conjuncts as
separate text: `ref == heading` and `kind == "R-ADMISS"`. Read side by side, they agree exactly.

Driven, all four drops:

| conjunct dropped from | caught by |
|---|---|
| the **quiet** filter — `kind` | **D2** (*"POSITIVE CONTROL D2: a block under its OWN heading that does NOT declare itself R-ADMISS…"*) |
| the **quiet** filter — `ref` | the DE24-R1 positive control (**A**) |
| the **adjudicating** reader — `ref` | the DE16-R1 real-register control |
| the **adjudicating** reader — `kind` | red, but as an **uncaught `RatificationRefused`** inside a fixture: *"REFUSED FOR A NEW RUN: R-419 is SUPERSEDED by R-999"* |

**Ruled: no direction is uncovered, so this is not the DE16-R4 shape the request feared** — but
the coverage is uneven. Three drops die at a named control; the fourth dies because a positive
control's own subject stops verifying. That is a real red and it is loud, and it is not a control
aimed at the conjunct. The finding below is about the duplication itself rather than about a gap.

## 2. CO-9 — the C / C2 pair, reproduced across three modules

| fixture | `e0d1e9f` | `89aef8c` | tip |
|---|---|---|---|
| **A** — a non-ratifying entry QUOTING `ref: R-903` / `supersedes: R-6` | REFUSED `#2` | RETURNS + report | **RETURNS + report** |
| **B** — an OWN block naming `R-6` | REFUSED `#2` | REFUSED `#2` | **REFUSED `#2`** |
| **C** — `R-99900` with TWO own blocks, EARLIER than R-419 | RETURNS | **REFUSED `own_ratification_blocks#1`** | **RETURNS + report** |
| **C2** — C with no duplicate anywhere in the register | RETURNS | **RETURNS** | RETURNS |
| **C3** — the same malformed entry, LATER than R-419 | REFUSED | REFUSED | **REFUSED** |
| **D** — duplicated heading, 2nd carries a QUOTED block | REFUSED `#3` | REFUSED `#3` | **RETURNS + report** |
| **D2** — duplicated heading, 2nd own-ref, `kind: R-OTHER` | REFUSED `#3` | REFUSED `#3` | **RETURNS + report** |
| **E** — duplicated heading, 2nd an OWN block `supersedes: R-419` | REFUSED `#3` | REFUSED `#3` | **REFUSED `#3`** |

**C and C2 are the finding, and the pair is what makes it one.** At `89aef8c` the same malformed
entry refuses when the register happens to contain an unrelated duplicate (`R-6`) and returns when
it does not. Whether `R-419`'s check refused depended on a fact about a different ref entirely —
which is precisely "an answer that depends on something the question is not about".

**Ruled on DE's own sentence.** Q-DE-45 says round 26's note — *"nothing is refused earlier than
before"* — was *"true of ORDER and silent on the SET"*. I agree, and C3 is why: the same entry on
the **later** side of the subject refuses at all three tips, via `superseded_by`, because there the
adjudicating reader is on the path by design. So round 26 changed **which entries the scan
adjudicates**, not the order in which it does so, and an ordering note could not have covered it.
The suite now carries a fixture on each side of the subject — C earlier (returns) and C3 later
(refuses) — which is the shape the finding demands.

## 3. Ruling — (iii) on OWN blocks, and no quotation reaches a verification

`_fenced_blocks` (`:584`) has exactly three callers: the quiet filter (`:634`), the adjudicating
reader (`:644`) — both of which filter to own — and `bind_from_block` (`:676`), which reads the
**subject's first fence** unfiltered. Driven at the tip:

| the subject's first fence | outcome |
|---|---|
| a QUOTED block (`ref: R-903` under `### R-999`) | REFUSED at `check#8` |
| a **self-quotation** (`ref == heading`, twice) | REFUSED at `own_ratification_blocks#1` |
| a block with `kind != R-ADMISS` | REFUSED at `check#10` |

So the only unfiltered reader produces **refusals and never a verification** from a quoted block.
DE's reasoning is complete for the property that matters: a quotation cannot reach an answer that
admits anything.

**On the stated definition — a self-quotation under a duplicated heading counts as own and
refuses, fail-closed: agreed.** The case where it matters is an entry that quotes its own
ratification (a sweep restating the block under the same ref): it is counted as own and refuses as
"two ratifications under one heading". That is the right answer for the reason the module already
gives — taking the first is how a corrected block gets shadowed by the one it corrects — and it
does not depend on knowing which of the two was "meant" as a quotation, which nothing can know.

## 4. D2 — DE's discovery, reproduced exactly

| run | result |
|---|---|
| D2 present, `kind` dropped from the quiet filter | **red at D2**, by name |
| D2 removed, `kind` dropped | red at the **count assertion** (`176 == 177`) — a vanished check, not the loosened predicate |
| **D2 removed, count adjusted to 176, `kind` dropped** | **GREEN at 176** |
| D2 removed, count adjusted, predicate intact | green at 176 (the control for the control) |

So before D2 existed nothing in the suite turned on the `kind` conjunct: **D exercises the `ref`
conjunct alone.** DE found that by running the mutant rather than by reasoning about it, and
disclosed it. That is the best evidence in this batch that the controls discriminate, and it
belongs in the record exactly as it came out.

## 5. The four mutants

| mutant | result |
|---|---|
| the quiet filter replaced by the raising reader | **red at C** (the CO-9 control) |
| (ii) back to `_fenced_blocks` (the `ref` drop) | **red at A**, round 26's positive control |
| the `kind` conjunct dropped | **red at D2** |
| (iii) back to `_fenced_blocks` | covered by the D/D2 pair above |

The `#3` message now reads *"…at least one occurrence carries an OWN ratification block"* and says
what a quoted block is. Read as a maintainer: it names the ref, both 0-based lines, and why a
quotation is not the thing being refused — enough to act on without opening the module.

## 6. DE24-R2 — `check#18` labelled at `89aef8c`

At that tip `:987` prints *"0-based register line"* and `:990` *"0-based line"*, and a suite check
(`:1913`) asserts both strings are present, recording that this was *"the one message of five
printing a 0-based field"* without saying so. Six line-printing sites in total — the three
`entry_index` refusals (`:398`, `:429`, `:445`), the suite's own line (`:1832`), and the two in
`check#18` — all now name the convention.

## 7. Counts and census — nothing removed in either commit

168 → **171** → **177** under both launchers; `EXPECTED_CHECKS = 177`. AST census: `ok` 86 → 89 →
**93**, `refuses` 52 → 52 → **54**, `refuses_nv` 4 throughout; **zero** check-call sites removed
across either step. Audit **34 cases / 28 sites**, 34 unique markers, survivors `[]`. R-419 on the
real register `True, [], []` with `duplicate_refs {'R-6': [1788, 9514]}` reported; seam **1,875**;
`de_admissible_windows.py` byte-identical to `50a9113`.

## 8. Discipline

One pathspec commit of one file per round (`91/9` and `161/6` on `de_ratification_check.py`);
nothing under `data/`; the line numbers cited resolve at the tip named in each item.

---

## Findings

### DE27-R1 — LOW — one ownership predicate, two texts

`own_blocks_quiet` (`:631`) and `own_ratification_blocks` (`:639`) each spell out
`str(blk.get("ref","")).strip() == ref and str(blk.get("kind","")).strip() == "R-ADMISS"`. They
agree today — I read both and drove all four conjunct drops — and every drop is red. So this is
not an uncovered gap; it is a **drift surface**, and it is the shape this module removed two rounds
ago for a different predicate: DE20-R1's *"one implementation of 'an entry exists'"*, asserted from
the AST because a predicate stated twice drifts without either copy noticing.

Two things make it worth naming now rather than later. First, the two texts are edited under
different pressures — the quiet filter is tuned by the scans, the adjudicating reader by the path —
so an edit to one is exactly the situation the AST census exists to catch elsewhere. Second, the
coverage is uneven: three of the four drops die at a control written for that conjunct, while the
adjudicating reader's `kind` drop dies as an **uncaught refusal inside a positive control**
(*"R-419 is SUPERSEDED by R-999"*) — red and loud, but not a control aimed at it, so a future edit
that changes which fixture breaks could turn a named red into a puzzling one.

**Closure:** one text. The shared part is the two-conjunct filter over `_fenced_blocks`; the
adjudicating reader adds only the two raises. Have `own_blocks_quiet` return the `(blk, dups)`
pairs and let `own_ratification_blocks` consume its output, so the conjuncts exist once — and, if
DE wants the same guarantee it gave `all_entries`, the AST census idiom already in this file can
assert that the filter is written in one place.

---

## Executed evidence

At `5e9dc8b` (and `89aef8c`, `e0d1e9f` side by side), 2026-09-02T16:13–16:17Z:

| check | result |
|---|---|
| scope | one file per commit; admissible byte-identical to `50a9113`; suites 177 / 91 / 69 |
| the fixture matrix | eight fixtures × three modules — reproduces the coordinator's table exactly |
| **CO-9** | C refuses at `89aef8c` and **C2 returns** — the refusal depended on `R-6`; both return at the tip; **C3 refuses at all three** |
| DE24-R1 | A returns at `89aef8c` and the tip, refused at `e0d1e9f` |
| the (iii) refinement | D and D2 report at the tip (refused before); **E still refuses** |
| the four conjunct drops | quiet `kind` → D2; quiet `ref` → A; adjudicating `ref` → DE16-R1; adjudicating `kind` → an uncaught refusal in a fixture |
| **D2 removed + count adjusted + `kind` dropped** | **GREEN at 176** — DE's discovery, reproduced |
| `_fenced_blocks` callers | three; the only unfiltered one reads the subject's first fence, and every quoted-fence outcome is a refusal (`check#8`, `own_ratification_blocks#1`, `check#10`) |
| `check#18` at `89aef8c` | *"0-based register line"* / *"0-based line"*, asserted by the suite |
| census | `ok` 86 → 89 → 93, `refuses` 52 → 52 → 54, **0 removed** in either step |
| audit | 34 / 28, 34 unique markers, survivors `[]` |
| unchanged | R-419 `True, [], []`, `duplicate_refs` reported, seam **1,875** |
| worktree | clean at `5e9dc8b`; cache cleared before every mutant; register untouched |

---

## Disposition

- **RELEASE** for **`5e9dc8b`**, and **RELEASE** for **`89aef8c`** as an intermediate tip — with
  the record that `89aef8c` is where CO-9 lived (C refuses, C2 returns) and `5e9dc8b` is where it
  closes. They differ in that respect and in no other.
- **RULED (item 2):** DE's own sentence is right — the round-26 note was true of order and silent
  on the set; C3 is the case an ordering note could not reach, and the suite now has a fixture on
  each side of the subject.
- **RULED (item 3):** the reasoning is complete. `bind_from_block` is the only unfiltered reader,
  it reads the subject's first fence, and every quotation-driven outcome is a refusal — no path
  admits anything through a quoted block. The self-quotation definition is right, and the case
  where it matters (an entry restating its own block) refuses for the reason the module already
  gives.
- **RULED (item 1):** no drift direction is uncovered, so the DE16-R4 shape does not apply — but
  the predicate is written twice, which is **DE27-R1**.
- **FILED:** **DE27-R1** (LOW — one ownership predicate, two texts; three of four drops die at a
  named control and the fourth at a broken positive control).
