# Review — DE round 21 (DE19-R1..R3: the prose is read, the block is pinned, the phrase has its known-bad)
reviewer: claude (pm-codex seat) · round opened by the coordinator (pm-co)

**Pinned tip executed: `0255b60`** (Q-DE-39 row at `0c7ab78`).
**Request of record:** `REQUEST_DE_ROUND_21_2026-09-02.md` (at `0004252`).
**Composed 2026-09-02T14:42:27Z.** One filing, per R-377.

Executed in `~/ctaNew-wt-rev` at `--detach 0255b60`. Read-only under `data/`; every mutant applied
to the worktree copy and restored — worktree clean after; `COORDINATION.md` never written. No
timer, no service, no launcher.

Scope confirmed: the round's commit touches **`de_admissible_windows.py` only, +107/−3** (the
request's +110/−2 is the `--stat` split). `de_ratification_check.py` is **byte-identical to
`0778918`**, the tip of round 20, which I released separately. Suites: admissible **84** both
launchers, ratification 155, seam 69, rc 0.

---

## Verdict

### RELEASE for `0255b60`. All three closures hold, and the item-1 measurement says the anchor set is the right extent — for the block as it stands.

One finding, LOW: the pin is by **content**, so it holds because `THE DECLARED LIMIT` happens to
be the block's topmost line. A paragraph added *above* the head, with a blank between, is unread
with all three anchors intact — and one structural predicate closes the whole class.

And the ruling asked for in item 3: **a subject-mutant through the reader is the standard**, the
assertion-mutant staying green is expected, and DE19-R3 is closed.

---

## 1. DE19-R2 — pinned from both ends, and the gaps measured

Three anchors: `THE DECLARED LIMIT`, `REFUSED (in the sets above)`, and the **full** heading
`DECLARED BLIND -- seen, named, and NOT refused`. The full heading is necessary, not stylistic —
I confirmed the plain token `DECLARED BLIND` also occurs above, in the REFUSED section's
cross-reference, so a short anchor would match twice and could survive a cut that removed the
section it names.

**Every inter-section gap, driven on a copy** (a blank non-`#:` line inserted at each):

| where the blank lands | chars read | unread | verdict |
|---|---|---|---|
| head ▸ REFUSED | 3,621 | 3% | **RED** (1 of 3 anchors gone) |
| REFUSED ▸ DECLARED BLIND | 3,071 | 18% | **RED** (2 of 3) |
| inside BLIND, above the builtins entry | 2,808 | 25% | **RED** (3 of 3) |
| inside BLIND, above the getattr entry | 2,450 | 35% | **RED** |
| inside BLIND, above the C-extension entry | 2,268 | 40% | **RED** |
| above OVER-CAUGHT | (the module's own known-bad) | 47% | **RED** |
| above NOT BLIND | 1,298 | 65% | **RED** |
| above the closing paragraph | 704 | 81% | **RED** |

**No gap is green.** The reason is structural: the reader walks up and stops at the first
non-`#:` line, so any blank inside the block drops everything above it — and `THE DECLARED LIMIT`
is the block's *first* line, so every cut takes at least one anchor with it. The anchor set is
therefore the right extent for this block, which is what item 1 asked. The residual is DE21-R1.

I also drove the vacuity guard: `_anchors = ()` with `len(_anchors) == 3` kept → **red**, message
printing `[]`. DE built that guard out of the fix for a control that could not fail, and it works.

No length pin: `len(_limit)` is printed (3,752 after the reword) and asserted nowhere — the right
call, and the reason is written where the number is.

## 2. DE19-R1 — the prose order is read, and the window is load-bearing

`_key_order` (`:1185`) locates the four map keys inside `[_BLIND_HEAD, OVER-CAUGHT)` and the check
asserts they are strictly increasing, distinct, and all present, naming any key the prose does not
mention. Resolved at this tip: **[64, 263, 621, 803]**.

**The window is necessary.** I checked each key against the text *above* the DECLARED BLIND
heading: `builtins.exec` **does** occur there (the REFUSED section's cross-reference explaining
that the attribute form is declared blind below). Without the window that key would resolve to the
earlier mention and the order could be satisfied by a sentence in the wrong section. So "a key
mentioned in the REFUSED section would not count, by design" — yes, and by necessity.

**Three mutants of my own, each red by name:**

| mutant | result |
|---|---|
| the runpy and getattr **paragraphs swapped in the real file** | **RED** at the order check |
| a key's prose mention **removed** (runpy reworded away) | **RED**, same check — the "does not name" branch |
| a key mentioned a **second time, earlier in the window** | **RED** — an earlier duplicate necessarily disorders the sequence, so the ambiguity class the membership check handles loudly is handled here too |

**The reword at `:166-167`.** Old: *"a module imported by a C extension or an import hook …
outside the source entirely. NOT ASSERTABLE IN-PROCESS — there is no source …"*; new:
*"C extensions and import hooks …… a module imported by one is outside the source entirely. NOT
ASSERTABLE IN-PROCESS — no source …"*. The limit is unchanged — same shape, same
not-assertable-in-process decision — said in the list's own words so the entry carries its key.
That is the minimum edit the order check needs and it does not weaken the declaration.

## 3. DE19-R3 — the ruling: a subject-mutant through the reader is the standard, and this is closed

`_cut2` (`:1235`) rewords the binding phrase on a copy and asserts the reader returns text with
`OVER-CAUGHT` present and the phrase absent — the same structure `_cut` has for the first
conjunct.

**That is what DE19-R3 meant, and I rule it closed.** My finding was an asymmetry: `_cut` falsified
the OVER-CAUGHT conjunct by moving its subject, and the phrase conjunct was evaluated only on the
uncut text, so nothing showed the phrase could go missing. `_cut2` supplies exactly the missing
half, in the module's own idiom.

**On the assertion-mutant that stayed green — expected, and not a gap.** Editing `ok(... and True)`
mutates the *test*, not the subject. No suite detects the deletion of its own assertion: that is
what the count assertion is for, and here `EXPECTED_CHECKS` cannot see it either, because the
check is still present and still runs — one conjunct fewer. Requiring assertion-mutants as the
standard would mean every conjunction carries a per-conjunct switch, which tests the harness
rather than the code, and this programme has consistently asked for the opposite: R-249's rule is
about controls that cannot fail *on their subject*, and rule 15 asks a checker to ship a
known-bad *input*.

**The honest residual, which I record rather than file:** neither `_cut` nor `_cut2` runs the
declaration check's own expression — both call `declared_limit_text` directly and assert on the
text. So what is proved is "the reader distinguishes the mutated subject", not "the check would
fail on it". The distance is small (the check is a two-term `in` over that same output, and both
terms are driven), and it closes completely if the predicate is lifted into a helper —
`_declaration_holds(text) -> bool`, asserted True on the real text and False on `_cut` and
`_cut2`. That is a strengthening worth taking when the lines are next touched, not a condition of
release.

## 4. Counts and census — nothing removed, literally

79 → **84** = +2 (DE19-R2) / +2 (DE19-R1) / +1 (DE19-R3), `EXPECTED_CHECKS = 84` at `:782`.

AST census `2f6da2c` → `0255b60`: `ok` 59 → **64**, `refuses` 15 → 15, total 74 → **79**, and
**zero** `ok(`/`refuses(` lines removed in the diff. Unlike round 18 — and like rounds 19 and 20 —
the claim holds without qualification.

The block length moved 3,754 → **3,752** with the reword, printed and asserted nowhere.

## 5. Nothing else moves; rule 10 / rule 14

Supply **1,875**; the seam emits **1,875** specs; `seam.daw is de_admissible_windows` **True**
(both imported by path); R-419 `verified_for_new_run True, unverifiable [], superseded_by []`;
`decides: "nothing -- this reports…"`; `_g_no_decision_field` untouched. `_BLIND_HEAD` is a
**local of the suite**, not a module constant — confirmed by attribute probe
(`hasattr(daw, "_BLIND_HEAD") is False`), so no new module surface.

**Interpolation:** 4 of the 5 added check-call sites interpolate what they evaluated
(`{_order}`, `{_sorder}`, `{list(_anchors)}`, `{len(_trunc)} of {len(_limit)}`,
`{_unnamed or 'none missing'}` — the fifth is a known-bad label beside a computed predicate).

**Citations spot-checked at `0255b60` — all seven exact:** `:782` (`EXPECTED_CHECKS = 84`), `:1149`
(`_BLIND_HEAD`), `:1157` (the anchor check's message), `:1185` (`_key_order`), `:1196` (the order
check), `:1212` (its known-bad), `:1235` (`_cut2`).

## 6. The round-22 boundary — nothing reaches into the DATA_ROOT split

`de_admissible_windows.py` still resolves its roots exactly as at `2f6da2c`: `ROOT =
Path(__file__).resolve().parents[2]` and `MASK_DIR = ROOT / "data/pm_5min/derived"` — the
code-root-derived data path, untouched by this round. The split stays behind DA's landing after
the 00:14Z read, as dispatched.

---

## Findings

### DE21-R1 — LOW — the block is pinned by content, so the pin holds only while the head is the topmost line

Every blank inside the block today takes an anchor with it (§1) **because** `THE DECLARED LIMIT`
is the first line of the `#:` run — the line above it is `_REBOUND = "<rebound-import>"`. Add a
paragraph above the head and a blank between the two, and the new content is unread with nothing
red:

```
block with a new paragraph above the head + a blank below it:
    3,752 chars read (identical to the intact block), all three anchors present, suite green at 84
```

The anchors pin *content that is known today*; they do not pin *extent*. The failure they were
written for returns the moment the block grows upward — which is exactly how the block has grown
in each of the last three rounds.

**Closure — one structural predicate, and it subsumes the anchors' job.** `declared_limit_text`
stops at the first non-`#:` line walking up; assert that the line it stopped at is a **real
boundary**, i.e. the line immediately above the first line it read is not itself a `#:` line. That
is False exactly when a blank cut a comment run in half:

- intact: the walk stops at the blank after `_REBOUND = …`, whose predecessor is code → holds
- blank above OVER-CAUGHT: the walk stops inside the run, predecessor is `#: …` → **fails**
- a new paragraph above the head with a blank below it: same → **fails**

It needs no knowledge of the block's contents, so it cannot go stale as the prose grows. Keep the
three anchors for what they are good at — naming *which* sections must be present — and add this
for extent.

---

## Executed evidence

At `0255b60`, 2026-09-02T14:39–14:42Z:

| check | result |
|---|---|
| scope | `de_admissible_windows.py` only, **+107/−3**; `de_ratification_check.py` identical to `0778918` |
| suites | admissible **84** both launchers, ratification 155, seam 69, rc 0 |
| **eight inter-section blanks** | **all RED**, 3% → 81% unread; no gap green |
| `DECLARED BLIND` short token | occurs above the section too — the full heading is required |
| `_anchors = ()` with the length guard | **RED**, printing `[]` |
| prose order | keys at **[64, 263, 621, 803]**, strictly increasing |
| my three order mutants | paragraphs swapped / a key's mention removed / a key named twice earlier → **all RED by name** |
| the C-extension reword | same limit, same NOT-ASSERTABLE decision, said in the list's words |
| **the residual** | a paragraph above the head + a blank → 3,752 chars, anchors intact, **green** — DE21-R1 |
| census | `ok` 59 → 64, `refuses` 15 → 15, **0 removed** |
| unchanged | supply/seam **1,875**, `daw is` True, R-419 `True, [], []`, `decides: nothing`, roots as at `2f6da2c` |
| `_BLIND_HEAD` | a suite local, not a module constant |
| messages | 4 of 5 added sites interpolate |
| citations | `:782`, `:1149`, `:1157`, `:1185`, `:1196`, `:1212`, `:1235` — all exact |
| worktree | clean at `0255b60` after every mutant |

---

## Disposition

- **RELEASE** for `0255b60`. DE19-R1 (the prose order is read, in a window that is load-bearing),
  DE19-R2 (the block pinned from both ends, with every inter-section gap measured red) and
  DE19-R3 (the phrase conjunct's known-bad, in the module's own idiom) all close. **No hold.**
- **RULING (item 3):** a **subject-mutant through the reader** is the standard DE19-R3 meant, and
  it is the standard `_cut` already set; the assertion-mutant staying green is expected and is not
  evidence of a gap, because a suite cannot falsify the deletion of its own assertion. Recorded,
  not filed: neither known-bad runs the declaration check's own expression, and lifting the
  predicate into `_declaration_holds(text)` would close that last inch.
- **FILED:** **DE21-R1** (LOW — the pin is by content; one structural predicate on the walk's stop
  pins extent and cannot go stale).
