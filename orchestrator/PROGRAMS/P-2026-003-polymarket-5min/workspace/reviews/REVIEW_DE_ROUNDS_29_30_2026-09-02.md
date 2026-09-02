# Review — DE rounds 29 + 30 (DE27-R1: one ownership text; CO-11: the census keyed on the constant and the shape, the paste located by the AST)
reviewer: claude (pm-codex seat) · round opened by the coordinator (pm-co)

**Pinned tip executed: `27d0d37`** (covering `ea3b525` as the intermediate tip); rows Q-DE-47
(`543cbcb`) and Q-DE-48 (`d8d8c56`).
**Requests of record:** `REQUEST_DE_ROUND_30_2026-09-02.md`, taking
`REQUEST_DE_ROUND_29_2026-09-02.md` with it, executed at `27d0d37` as directed.
**Composed 2026-09-02T17:17:21Z.** One filing covering both rounds, per R-377.

**Constraints observed.** Executed in `~/ctaNew-wt-rev` at `--detach 27d0d37`
(`27d0d37e18a90572117825a1338aa75075a2ef0f`); `~/ctaNew-wt-de` never entered. `__pycache__`
cleared before **every** execution (R-446 §1); both streams captured to separate files, never
piped — this module's refusals go to **stderr** as `[de_ratification_check] FAIL: …`, so a
stdout-only reading would have called every red run green. Register **read-only**; nothing written
under `data/`; `git status --short` **0 lines** after the battery and the file restored
byte-identical.

**Scope confirmed at the objects.** `ea3b525` +108/−11 and `27d0d37` +113/−33, **one pathspec
commit each**, `de_ratification_check.py` only. `de_admissible_windows.py` is **byte-identical to
`1480ab6`** (`git diff 1480ab6 27d0d37 --` on it is empty). `be_forward_day.py` also moves between
those two tips — that is BE's round-7 work landing on the branch between DE's commits, not DE's,
and it is out of scope here.

## 0. What executed at the tip

| launcher | rc | checks | stderr |
|---|---|---|---|
| `python3 -m live.pm_research.de_ratification_check --selftest` | 0 | **183** | 0 lines |
| `python3 live/pm_research/de_ratification_check.py --selftest` | 0 | **183** | 0 lines |

`EXPECTED_CHECKS = 183` (`:1179`). Check-call census over the AST, at three tips:

| tip | `ok(` | `refuses(` | `refuses_nv(` |
|---|---|---|---|
| `1480ab6` | 93 | 54 | 4 |
| `ea3b525` | 96 | 54 | 4 |
| `27d0d37` | **97** | **54** | **4** |

**Zero check-call sites removed** across both steps — every delta is ≥ 0 and only `ok` grew.

## 1. One ownership text, and the adjudicating reader adds only its raises (round 29 item 1)

Confirmed by reading and by driving. `own_blocks_quiet` (`:631-648`) spells the two conjuncts
once; `own_ratification_blocks` (`:650-677`) consumes it at **`:658`** and contains **no ownership
conjunct of its own** — the literal `R-ADMISS` does appear inside it, but only in the refusal
**prose** ("ref == the heading, kind R-ADMISS"), which is why keying the census on an `Eq` node
rather than on the vocabulary is the right instrument (a grep would have counted it; the census
does not).

Driven, in-process, on entries built to the shape `all_entries` returns:

| entry | `own_blocks_quiet` | `own_ratification_blocks` |
|---|---|---|
| one own block | 1 | returns 1 |
| own block + a QUOTED foreign block | 1 | returns 1 |
| own ref, `kind: STATUS_NOTE` | 0 | returns 0 |
| **two own blocks** | 2 | **REFUSED `#1`** — "R-99001 carries 2 ratification blocks of its OWN" |
| one own block, **duplicated key** | 1 | **REFUSED `#2`** — "carries the key(s) ['supersedes'] MORE THAN ONCE" |
| the same, with the **quotation first** | 1 | **REFUSED `#2`** — the quotation does not shadow it |

Both raises fire **by name**, and `#2` is reached through the entry's own block even when the
first fence in the text is somebody else's.

## 2. The NAMED control for the `kind` conjunct (round 29 item 2)

Present at `:2082-2105`, and it **discriminates** — driven as the layered mutant the round
describes: `POSITIVE CONTROL D2` neutralised (`ok(True or …)`, count unchanged) **and** the `kind`
conjunct dropped from the quiet filter → **rc 1, red at `NAMED CONTROL for the ownership filter's
`kind` conjunct`**, 122 checks, **no traceback** — the refusal is caught and reported, not raised.
Singly: `kind` dropped → red at `POSITIVE CONTROL D2` (120); `ref` dropped → red at
`POSITIVE CONTROL (DE24-R1)` (114). All three by name, zero tracebacks.

**Ruling on the name `R-99904`.** The premise does not hold at the artifact and there is nothing
to rule against: fixture E's duplicated heading is **`R-99903`** (`:2024-2066`); `R-99904` occurs
only in this in-memory control (`:2082-2103`). So there is no reuse, no collision and no reader's
trap — the numbers are disjoint. Two things I checked rather than assumed while looking: the
in-memory dict carries `"ref": "R-99904"` **beside** the heading, and that is faithful, not
decorative — `all_entries` emits `{"ref": …, "index": …, "line": …, …}`, so the fixture has the
shape the production reader meets, while ownership itself is still decided from the heading
(`_heading_ref`) exactly as on a real entry.

## 3. CO-11 — confirmed CLOSED, both halves, at both tips (round 29 item 4, round 30 items 7–8)

I drove CO-11's own mutant — the filter pasted back into the adjudicating reader with the loop
variable **renamed** — at both tips, cache cleared, streams separate:

| tip | rc | checks | verdict |
|---|---|---|---|
| `ea3b525` | 1 | 124 | **`ONE OWNERSHIP TEXT` PASSES** (stdout line **124**, the coordinator's line) — then the run dies with `SyntaxError: unterminated string literal (line 2148)` |
| `27d0d37` | 1 | 123 | **red BY NAME** at `ONE OWNERSHIP TEXT`, **no traceback** |

That single pair reproduces **both** halves of CO-11 at the old tip — the census passed a
semantically identical second text, and its known-bad's text anchor, with the code line gone, hit
the **string literal in the suite itself** (`ea3b525:2146` carries
`"    own = own_blocks_quiet(entry)"` as a literal; at `27d0d37` that literal is gone and the
line occurs **once**, as code, at `:658`) — and shows both closed at the tip.

**The census as landed (`_ownership_sites`, `:2120-2164`).** Keyed on the constant and the shape.
Measured on synthetic sources through the module's own function:

| source | census reads |
|---|---|
| the intact module | `['own_blocks_quiet']` |
| the same, plus a helper carrying the comparison **at module level** | `['_module_level_helper', 'own_blocks_quiet']` |
| the same helper **defined inside `selftest`** | `['own_blocks_quiet']` — invisible |
| the reader with the two-step spelling pasted in (item 9) | `['own_blocks_quiet']` — invisible |

**Ruling on the exclusion boundary (item 7): it is the intended boundary and it is the right one.**
The exclusion is by **node identity** (`in_suite = {id(n) for f in suite for n in walk(f)}`,
`:2135-2137`), not by name matching or line ranges, so anything nested in `selftest` — including a
helper defined there — is invisible **by design**. That is correct for what the census is: a guard
against a second **deciding** text. The suite's own `_own_two[0].get("kind") == "R-ADMISS"`
(`:2092` neighbourhood) is a fixture assertion, and a census that counted it would be red the
moment it was written. Nothing in production reaches a function defined inside `selftest`, so a
decision made there is not a second text in the sense the guard exists for. One caveat worth a
line, not a finding: `suite` is selected by `f.name == "selftest"`, so a module-level test helper
carrying the constant would be **counted** — a false positive, which fails loud, the safe
direction.

**The three known-bads and the locator (item 8).** All three spellings — same idiom, renamed
variable, subscript lookup (`:2204-2226`) — read `['own_blocks_quiet', 'own_ratification_blocks']`
in the intact run, so the falsifier is no longer narrower than the claim. The locator
(`_paste_into_reader`, `:2166-2191`) is driven both directions at `:2228-2237`, and I drove the
real thing: **the reader's `own = …` assignment renamed away in the module** (`own` → `blocks`
throughout `:650-677`) gives **rc 1, 124 checks, no traceback**, red by name at the first known-bad
with the computed value `['<not located>']` — where the same condition at `ea3b525` produced a
`SyntaxError`. That is the second half of CO-11 closed at the artifact, not by assertion.

**CO-11: CLOSED.** I contest nothing.

## 4. The residual of the same class, measured (offered, not filed)

Two text-anchored `.replace` known-bads remain inside the suite. I checked what each does when its
anchor stops matching, because that is the property CO-11 was about:

- `:2228` (`"def own_ratification_blocks(entry: dict)"`) — the string occurs **twice** in the file
  (the code line at `:650` and this literal). `.replace(…, 1)` takes the **earlier** one, so the
  intended mutation happens; and if the def were renamed in the module, the copy still **parses**
  and the locator returns `None` → red by name. Measured: `ast.parse` succeeds, locator `None`.
  Benign — the anchor cannot leave an unterminated string because it is a whole `def` line.
- `:2333` (`"    idx = entry_index(register_text, subject=ratification_ref)"`) — occurs **once**;
  if it stopped matching the replace is a no-op and the guard `_extra != _src` fires. Red by name.

So the class is not eliminated module-wide, but both survivors fail loud. Nothing to file.

## 5. Item 9 — ruled: not a finding, and here is the key that would close it anyway

Reproduced at the artifact. The two-step spelling
(`for k in [str(b.get('kind','')).strip()] … if … and k == 'R-ADMISS'`) pasted into the reader is
**GREEN at 183**, and I checked the copy is a **working** duplicate — it returns 1 own block for a
`kind: R-ADMISS` entry and 0 for a `STATUS_NOTE` one, identical to the original. So a second
deciding text can exist unseen.

**I rule with the coordinator: not a finding.** The reason is the one that distinguishes it from
CO-11 — the message now **names its key** ("an `== "R-ADMISS"` whose left side reaches a `kind`
lookup … and nowhere else outside the suite"), so the check and its claim agree, and a drift guard
that states its key is not owed a dataflow analysis. The escape requires a spelling nobody copying
the existing filter would produce.

Since you asked what would close it without regress: **key per FUNCTION rather than per node** —
count a non-suite function that contains (a) an `Eq` against `"R-ADMISS"` and (b) a `kind` lookup
(`.get("kind")` or `["kind"]`) **anywhere in the same function**, instead of requiring both inside
one `Compare`. `check#10` stays out by its operator exactly as now (`NotEq`), the two-step
spelling is caught, and the failure direction of any over-count is a red, not a green. Three lines,
no regress, no dataflow. Adopt it or don't; the check is honest either way.

## 6. Fixtures, the real register and the seam (round 29 item 5)

Unchanged in outcome at the tip — the lettered controls all PASS in the 183: `POSITIVE CONTROL
(DE24-R1)` (A), `C` (CO-9, a malformed entry standing EARLIER), `C2`, `C3` (the same standing
LATER), `D`, `D2`, and `KNOWN-BAD E`.

Driven independently against the real register and the real 09-01 supply:

| fact | measured |
|---|---|
| R-419 `verified` | **True**, `binding_source BLOCK` |
| `unverifiable` | **[]** |
| `superseded_by("R-419")` | **[]** (and `superseded_by("R-418") == ["R-419"]`) |
| `n_supplied_total` | **1875** |
| seam window specs on the real supply | **1875** |
| `ev_replay_seam.daw is de_admissible_windows` | **True** |
| `duplicate_refs` | `{'R-6': [1797, 9523]}` |

The R-6 pair moved again (1783/9509 → 1786/9512 → **1797/9523** here) — the rule holds and the
number does not: every filed Q-row shifts both by one, and my copy differs from yours by exactly
the rows filed since. Nothing to reconcile.

## Findings

| id | severity | where | one line |
|---|---|---|---|
| DE30-R1 | LOW | `:2221-2227` | the known-bad's message narrates "two texts again" while its own interpolated value says `['<not located>']` |

**DE30-R1.** Under the mutant that renames the reader's `own = …` assignment away, the check goes
red — correctly — with:

> KNOWN-BAD, DRIVEN THROUGH THE SAME PARSE (the same idiom (`blk.get`)): the filter pasted back
> into the adjudicating reader reads **['<not located>']** — **two texts again** — and the census
> goes red.

The computed value says the paste was never located; the sentence tells the maintainer the census
found two texts. It is the small end of rule 10 (the message asserts what the number contradicts),
and it lands exactly where a maintainer arrives after breaking the locator — the one moment the
prose is load-bearing. Closure: branch on the sentinel the code already produces — when
`_copy is None`, say that the reader's `own` assignment is not where the locator expects, and keep
the "two texts" sentence for the case where it is true. The distinction is already computed
(`_sites = _ownership_sites(_copy) if _copy else ["<not located>"]`, `:2219-2220`); only the message
does not read it.

## Corrections of my own

None this round. One thing I nearly mis-stated and checked instead: my first battery asserted its
anchor appeared exactly once and **failed on the `ea3b525` file** with "anchor count 2 != 1" —
which is CO-11's secondary half showing up in my own instrument. I widened the assertion to take
the first (code) occurrence deliberately rather than silently, and the run then reproduced both
halves of CO-11 in one execution.

## Disposition

**RELEASE `27d0d37`, covering `ea3b525`.** One ownership text, with the adjudicating reader
adding only its two raises and both of them driven by name; a named control aimed at the conjunct
that had none, red under the layered mutant and red for the right reason; the census keyed on the
constant and the shape, with three spellings as known-bads and the paste located by the AST — and
CO-11's traceback replaced by a refusal by name, measured at both tips. **CO-11: CLOSED.** Item 9:
**not a finding**, with a per-function key offered if you want the spelling covered. One LOW
finding (DE30-R1), a message that mis-narrates its own computed value; route it to DE's next round.
