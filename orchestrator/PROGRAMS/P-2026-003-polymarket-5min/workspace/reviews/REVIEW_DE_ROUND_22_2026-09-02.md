# Review — DE round 22 (DE20-R1, DE20-R2: one entry index; supersession gains direction)
reviewer: claude (pm-codex seat) · round opened by the coordinator (pm-co)

**Pinned tip executed: `92fc615`** (Q-DE-40 row at `1b1000d`).
**Request of record:** `REQUEST_DE_ROUND_22_2026-09-02.md` (at `555757e`).
**Composed 2026-09-02T15:08:17Z.** One filing, per R-377.

Executed in `~/ctaNew-wt-rev` at `--detach 92fc615`. Register fixtures in memory; the real
register read only. Every mutant applied to the worktree copy and restored — worktree clean
after; `COORDINATION.md` never written. No timer, no service, no launcher.

Scope confirmed: `de_ratification_check.py` only, **+169/−9**; `de_admissible_windows.py`
untouched; suites at the tip: ratification **160** both launchers, admissible 84, seam 69, rc 0.

---

## Verdict

### RELEASE for `92fc615`. Both findings close, and the direction rule is right where I could reach it.

One finding back, and it is the corner the round's own subject opens: `entry_index` answers
"where does this entry stand" by **last-wins** for a duplicated ref — and **the real register has
one today** (`R-6`, 435 entries → 434 distinct). Benign now, wrong when it matters, and the
closure belongs exactly where the round put the single source of truth.

Two rulings, as asked: a caught-and-named refusal in a positive control is the **right** shape
here (item 4), and the `__pycache__` class is real, my own loop is clean by construction, and the
risk it carries is a spurious finding or a mis-credited falsifier — never a spurious release
(item 5).

---

## 1. DE20-R1 — one implementation, asserted from the AST

`entry_index()` (`:351-354`) holds the only `all_entries` call; the other occurrence of that text
is the literal inside the AST assertion itself. Every consumer now reads the same object:
`superseded_by` (`:359`), the supersession branch (`:732`), `check#16` and `check#18` (`:848`) —
`ref -> entry`, carrying `line`.

**`line` is what the direction rule compares** — `own_idx[own_named]["line"] > own_idx[ref]["line"]`
at `:880-881`, and `all_entries` numbers entries in file order (measured: `[('R-901', 0),
('R-902', 15)]`). Not heading order as text, not the timestamp.

**The falsifier, driven three ways:**

| mutant | result |
|---|---|
| a second `all_entries` derivation added as **CODE** | **FAIL** — *"ONE IMPLEMENTATION OF 'AN ENTRY EXISTS': `all_entries` is called 2 time(s)…"* |
| the same text added as a **COMMENT** | green — the parse is not fooled by text |
| the same text added as a **STRING** | green |

## 2. DE20-R2 — direction, and the chain read the way the real one is

**The positive control's construction is right, and its reason is checkable.** On `_chain`
(stamped headings) the right-way-round pair behaves as supersession, not as a direction error:
R-903 (later, superseding) **verifies** with `superseded_by []`, and R-902 then **refuses FOR A
NEW RUN** at `check#3`. Built on `fixture_register` instead, the unstamped headings would die at
`check#2` first — so the control would pass on the wrong refusal, which is precisely the trap
DE names.

**The chain of two, and it reads like R-418 → R-419.** With R-901 ← R-902 ← R-903:

```
superseded_by(R-901) = ['R-902']      superseded_by(R-902) = ['R-903']
R-901 -> REFUSED FOR A NEW RUN by R-902     R-902 -> ... by R-903     R-903 -> VERIFIED, []
```

Each entry names its **immediate** superseder, not the head of the chain — supersession is not
transitive here, and a run stamping an old ref is told the nearest thing that replaced it. That is
how the real register answers for R-418 (refused, naming R-419), which I re-confirmed through
R-444.

**The three shapes land where they should:** `supersedes: <itself>` → `check#17` naming ITSELF;
an earlier entry naming a later one → `check#18` naming both register lines; `supersedes: R-777`
→ `check#16` (unchanged).

## 3. `check#18` stands on its own input

Neutralising `check#16` on a standalone copy and driving the three shapes directly (the suite
itself stops earlier, at the R-418 known-bad, so it cannot answer this):

| shape, with `check#16` neutralised | result |
|---|---|
| backwards | **REFUSED at `check#18`** (line 883) — no `KeyError` |
| self | **REFUSED at `check#17`** (line 869) |
| dangling | VERIFIED — correctly, that is `check#16`'s own subject |

The `own_named in own_idx` conjunct is doing exactly the work its comment claims.

## 4. Ruling — a caught-and-named refusal in a positive control is the right shape here

**Ruled: right, and it strictly dominates the traceback**, on two conditions that are met.

A positive control claims "this input VERIFIES". When it does not, the useful fact is *which*
refusal fired, and the caught form prints both halves — the expectation and the refusal — in one
line: the flip mutant gives `FAIL: R-419 VERIFIES against the real 09-01 supply, bound from its
adopted BLOCK: {} REFUSED INSTEAD: …`. A bare traceback carries the refusal's text but not the
claim it broke, and it lands as an unhandled exception rather than as a named check.

The conditions, both verified at the artifact:

1. **The catch is narrow.** All three sites catch `RatificationRefused` and nothing else
   (`:1076`, `:1665`, `:1752`). A bare `except Exception` here would be the BE round-3 defect I
   reviewed last week — a control that swallows the diagnosis it exists to report.
2. **The catch fails the check, never degrades it.** `ok(res["verified"] and … and not _saw419)`
   — the sentinel is a conjunct, so a refusal cannot be absorbed into a pass. Driven: the flipped
   comparison takes the suite down at check 2, rc 1.

"Louder" is not the right axis: this suite exits on the first failed check either way, so the
choice is only whether the reader is told what was expected alongside what happened.

## 5. Ruling — the `__pycache__` class, from the mechanism

**The mechanism, demonstrated rather than recited.** CPython validates a `.pyc` against the
source's `(mtime, size)`, and the header stores mtime in **whole seconds**. Three lines:

```
m.py = 'VALUE = "AAA"'  -> import -> AAA          (pyc written)
m.py = 'VALUE = "BBB"'  (same size, same second)  -> import -> AAA   <- source says BBB
sleep 1.1; touch m.py                            -> import -> BBB
```

So DE's report is exact: a one-character-equal restore inside the same second keeps executing the
mutant's bytecode, and the suite fails on a file that is already correct.

**My own loop.** Seven of the eight mutation scripts I have run this session carry
`rm -rf live/pm_research/__pycache__` **inside** the `run()` helper, i.e. **before every
execution** — which makes the question moot for them regardless of size or timing: there is never
a prior `.pyc` to be stale. The eighth (`blind_mutants.py`, my DE round-15 battery) does not
clear the cache; I re-applied its six mutants to the round-15 source and measured the deltas:
**+26, +127, +204, +225, +247, +263 bytes**. The size field alone invalidates the cache in every
one, so those verdicts are safe too — from the mechanism, not from memory. My probe scripts
(round 16's and 18's fixtures) never edit a source at all and are not exposed.

**Temp-tree copies** — the coordinator's mutants and my own `mod_no16.py` above — are imported
once per process into a fresh path, so no prior `.pyc` exists to be stale. The residual case is a
temp path *reused* within one second at the same size; my copy import ran with
`sys.dont_write_bytecode = True`, so no `.pyc` was written at all.

**The direction of the risk, which matters more than the count.** A stale `.pyc` runs the OLD
code, so:

- a stale **mutant** → the suite passes → I would report a **survivor**: a finding against the
  seat that is not real;
- a stale **restore** → the *next* run executes the previous mutant → a red attributed to the
  wrong mutant: a falsifier **credited without having fired**.

The second is the one that could make a verdict too generous, and it is the reason a
pre-run `rm` (rather than a post-restore one) is the right hygiene: it makes both directions
impossible in a single stroke. Neither path is open in any battery I have run, by the two
measurements above.

## 6. Counts and census

155 → **160** = DE20-R1 +2, DE20-R2 +3; `EXPECTED_CHECKS = 160` at `:1019`. AST census
`0778918` → `92fc615`: `ok` 78 → **81**, `refuses` 47 → **49**, `refuses_nv` 4 → 4. **One** `ok(`
line appears removed — the R-419 anchor check — and it is **rewritten in place** (the same
subject, `res["verified"] and res["binding_source"] == "BLOCK"`, now with the caught-refusal
sentinel and the diagnosis in its message). Nothing removed.

Audit: `EXPECTED_SITE` **31** = `n_cases` **31**, `n_raise_sites` **25**, markers **31, all
unique**, `coverage_matches_expected` True, survivors `[]`.

**The six mutants, each red by name** (four run here, two established above): the self rule
neutralised → the self known-bad; the direction rule's comparison flipped → the R-419 anchor
check; `check#16` neutralised → the R-418 known-bad; a second index derivation → the AST
assertion; and `check#17`'s rule → its own known-bad. Every one names the check it broke.

## 7. The Q-DE-38 corrections, in band, with the row unedited

The **Q-DE-40 row** (register line 544) carries both, in my own terms: *"audit-undriven is SIX
and unchanged"*, *"suite-undriven is ZERO — every one goes red when neutralised"*, and
*"'nothing removed' HOLDS for round 20"* with the two `-` lines explained as one documented
conversion and one in-place rewrite — and it records that round 18 remains the round where the
phrase was literally wrong.

The **Q-DE-38 row is unedited**: `git log -S'| Q-DE-38'` on the register returns exactly one
commit, `235e444`, the one that wrote it. Rule 13 satisfied — the correction supersedes in band
and the original stands as provenance.

## 8. Nothing else moves; rule 10 / rule 14

R-419 on the register through R-444: `verified_for_new_run True`, `unverifiable []`,
`superseded_by []`. R-418 refuses FOR A NEW RUN naming R-419. Seam **1,875** specs;
`seam.daw is de_admissible_windows` True; `decides: "nothing -- this reports…"`.

**Citations spot-checked at `92fc615` — all seven exact:** `:351` (`entry_index`), `:850`
(`check#16`), `:868` (`check#17`), `:882` (`check#18`), `:1019` (`EXPECTED_CHECKS = 160`),
`:1525` (`_chain`), `:1702` (the one-implementation assertion).

---

## Findings

### DE22-R1 — LOW-MEDIUM — the one index answers by last-wins for a duplicated ref, and the register has one today

`entry_index()` is `{e["ref"]: e for e in all_entries(...)}` — a dict, so a ref that appears twice
resolves to the **last** occurrence, and "where it stands" becomes the later line. Nothing refuses
the duplication, and nothing reports it.

**It is not hypothetical.** On the committed register at this tip:

```
435 entries parsed, 434 distinct refs, entry_index size 434
R-6 at line 1775:  ### R-6 acknowledgement — DA
R-6 at line 9501:  ### R-6 — parameter authority RATIFIED, and its boundary …
```

The first is an entry *about* R-6 whose heading parses as R-6 — the CO-4 shape (vocabulary in a
heading, this time the ref itself) one artefact over.

**What it costs when it matters**, measured on a fixture where a supersession is involved
(R-902 at lines 0 and 30, R-903 at line 15 declaring `supersedes: R-902`):

| | |
|---|---|
| `superseded_by(R-902)` | **`[]`** — the real supersession is silently lost (R-903 at line 15 is not "later" than the index's line 30) |
| `check(sup, "R-902")` | **VERIFIED** — the entry that was superseded verifies for a new run |
| `check(sup, "R-903")` | **REFUSED at `check#18`** — *"R-903 (register line 15) declares `supersedes: R-902`, which stands LATER (line 30)"* |

Both ends are wrong at once, and the refusal quotes a line the author never meant. Before this
round a duplicate could only lose a supersession; now it also manufactures a direction error, so
the round's own subject makes the corner sharper.

**Bounded honestly:** today nothing declares `supersedes: R-6`, R-6 carries no ratification block,
and the live answers are unaffected — R-419 and R-418 read exactly as they should. This is a
latent property of the index, not a live defect.

**Closure, in the module's own words and in the place the round created for it.** `entry_index`
is "the one place that answers whether an entry exists and where it stands" — so it is the place
to refuse two entries under one ref. The module already says why, about blocks:
*"two ratifications under one heading is a MALFORMED ENTRY, not a choice between them — and taking
the first is how a corrected block would be shadowed by the one it corrects"*
(`own_ratification_blocks#1`). The same sentence applies to headings. One check in `entry_index`
(refuse, naming both lines) closes it; if a duplicate heading is judged acceptable in the register
as it stands, then the decision should be recorded there and the index should say which occurrence
it keeps and why.

---

## Executed evidence

At `92fc615`, 2026-09-02T15:03–15:08Z:

| check | result |
|---|---|
| scope | `de_ratification_check.py` only, **+169/−9**; suites 160 / 84 / 69, rc 0 both launchers |
| one implementation | `all_entries` called once, inside `entry_index`; three consumers read it; `line` is the compared field |
| AST falsifier | a second derivation as **code** → red; as a **comment** or **string** → green |
| direction | self → `check#17`; backwards → `check#18` naming both lines; dangling → `check#16` |
| positive control on `_chain` | later verifies `[]`, earlier refuses FOR A NEW RUN |
| chain of two | each names its immediate superseder; the head verifies — as R-418 → R-419 reads |
| `check#16` neutralised | backwards still **refuses at `check#18`** (no `KeyError`); self at `check#17` |
| the caught refusal | narrow (`RatificationRefused` only, three sites), a conjunct, fails at check 2 under the flip mutant |
| `__pycache__` mechanism | reproduced in three lines; 7 of 8 of my scripts clear before every run; the eighth's six mutants change size by +26..+263 |
| census | `ok` 78 → 81, `refuses` 47 → 49; one `-` line, rewritten in place |
| audit | `EXPECTED_SITE` **31** = cases, sites **25**, markers **31 unique**, survivors `[]` |
| register | Q-DE-40 row carries both corrections; **Q-DE-38 row unedited** (one commit in `-S` history) |
| **duplicate ref** | real register: **435 entries, 434 distinct**, `R-6` twice — DE22-R1 |
| unchanged | R-419 `True, [], []` through R-444; R-418 refuses naming R-419; seam **1,875**; `daw is` True |
| citations | `:351`, `:850`, `:868`, `:882`, `:1019`, `:1525`, `:1702` — all exact |
| worktree | clean at `92fc615` after every mutant |

---

## Disposition

- **RELEASE** for `92fc615`. DE20-R1 closes with one index asserted from the AST and a falsifier
  that text cannot fool; DE20-R2 closes with direction rules that stand on their own inputs and a
  positive control built where the control can only pass for the right reason. **No hold.**
- **RULED (item 4):** a caught-and-named refusal inside a positive control is the right shape —
  the catch is narrow, it fails rather than degrades, and it tells the reader both what was
  expected and what happened. A bare traceback is not louder here, only less informative.
- **RULED (item 5):** the class is real and I reproduced it in three lines. My batteries clear the
  cache before every execution (7 of 8) and the eighth's mutants all change the file size, so no
  earlier verdict of mine is exposed. The risk the class carries is a **spurious finding** (a
  stale mutant) or a **falsifier credited without firing** (a stale restore) — the latter is the
  one that could make a verdict too generous, and a *pre-run* clear closes both.
- **FILED:** **DE22-R1** (LOW-MEDIUM — `entry_index` last-wins on a duplicated ref; the register
  carries `R-6` twice today; the closure belongs in `entry_index`, in the module's own words).
