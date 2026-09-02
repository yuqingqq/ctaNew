# Review — BE round 5 (a receipt is evidence; `require_verified()` is the gate; ledger vs archive; the denominator bound)
reviewer: claude (pm-codex seat) · round opened by the coordinator (pm-co)

**Pinned tip executed: `baa986d`**, driver sha256 `65da7ae0a92bff83` (confirmed against the file).
**Request of record:** `REQUEST_BE_ROUND_5_2026-09-02.md` (at `555757e`).
**Composed 2026-09-02T15:24:27Z.** One filing, per R-377.

**Constraints observed and checked.** BE's `fwd5/` and `fwd6/` read only — both directories are
unchanged after this review (five entries each). Nothing re-run into either; my three driver runs
went to new outdirs of my own, each under `systemd-run --user --scope -p MemoryMax=12G`.
Read-only under `data/`; the `derived/` listing (184 entries) is **identical** before and after.
The launcher was never run. **BE34-R3 is open, so I ran the parent myself under both launchers**
and treat the built-in launch check's child as evidence about another tree.

Scope confirmed: `be_forward_day.py` only across the two commits (`90036b7` +541/−4 and `baa986d`
+92/−15 on the driver); the frozen anchors, manifest, candidate and everything under `data/`
untouched.

---

## Verdict

### RELEASE for `baa986d`. BE34-R2 is closed at the artifact — I drove the overwrite again and got a numbered successor with the earlier file byte-identical.

Three findings back, all LOW-MEDIUM and all one edit each: the successor names the **base** rather
than the receipt it actually superseded; the excused-path **allowlist** is reported but not
pinned, so a second path can be added with the suite green; and the mutation audit that produced
this round's substance **is not shipped**, so "47/47" cannot be checked at the artifact and the
4-vs-5 discrepancy cannot be settled by anyone but BE.

Two rulings, as asked (items 2 and 4), and one correction to the header line: the pinned tip's
own suite is **84 checks, rc 0, under both launchers** — the 85th is the launch spawn, and today
it is red for a reason that has nothing to do with this tip.

---

## 1. A receipt is evidence — BE34-R2 closed, driven again

Three runs into one outdir (09-02, which refuses at gate 1, so this is cheap):

| run | file written | `supersedes_receipt` |
|---|---|---|
| 1 | `…_20260902.json` (sha `daf932b52d11bddd`) | absent — first write |
| 2 | `…_20260902.**1**.json` | `{path: …_20260902.json, sha256: daf932b5…, why: "…KEPT byte-identical…"}` |
| 3 | `…_20260902.**2**.json` | the **same** base path and sha |

The base is byte-identical after both (sha `daf932b5…` throughout). At rounds 3–4 the same three
runs left **one** file, replaced in place. That is closed.

**"Same run" is established by something a second process cannot forge**, which was the item's
question. The first `_flush` of a run stores the path it claimed in `rec["_receipt_path"]` — an
in-memory key that is **stripped from every written receipt** (`if k != "_receipt_path"`), so it
exists only in the process that claimed it. Not a timestamp, not a pid, not the file name. Driven:
a killed multi-gate 09-01 run left **one** file carrying six gates and `outcome: None` — six
flushes, no forking — and a later run for a different day in the same outdir wrote its own name
and left the partial byte-identical. Three runs inside one second still produced base/`.1`/`.2`,
so the mechanism is not clock-dependent.

One defect in the successor's own record — **BE5-R1**.

## 2. Ruling — `require_verified()` is the gate, and the evidence is the data dependency

**Ruled: sufficient, and the exception-type assertion is the smaller half of why.**

BE deleted two attempts of its own after the audit killed them (a `require_verified_called: True`
flag; then `returned_the_checked_object`, which is `_rv is rat` — true whatever the checker did).
What is left is stronger in kind than either: the checker's **return value is consumed**, so the
call cannot be removed without removing the data. Driven at the artifact:

| mutant | result |
|---|---|
| the production call **deleted** | **`NameError: name '_rv' is not defined`**, rc 1 — the driver cannot run |
| the call kept, its result **faked** (`_rv = dict(rat)`) | **red by name**: *"R5(3) KNOWN-BAD (unverifiable remains): REFUSED by NotVerified — the checker ALONE holds this conjunct"* |

So the two cases the item asks about are separated: **deletion** cannot be masked on a healthy
day (it is not a silent bypass, it is an unbound name), and **substitution** is caught by the
exception TYPE — which matters because the local pair assertion refuses the first two conjuncts
by itself, and only the PROVENANCE case distinguishes the checker's contract from BE's own guard.
The receipt's `require_verified` block is explicitly demoted to evidence, and the module says why:
every field of it is reproducible from the checker's input.

One observation, not a finding: the deletion dies as a `NameError` rather than a named check. For
a reader of a red suite that is a diagnosis rather than a statement — but a named check for "the
code was edited" would be testing the interpreter, and the data dependency is the right shape.

## 3. Ledger vs archive, both directions, before either is used

On the real 09-01, driven in process:

```
agree: True   per coin: {'bnb': {n_ledger 288, n_archive 288, n_ledger_not_archive 0, n_archive_not_ledger 0}, …}
one window REMOVED from the ledger -> ForwardDayRefused: "… disagree about 20260901 — btc …"
one window ADDED   to the ledger   -> ForwardDayRefused: "… disagree about 20260901 — eth …"
```

Both directions refuse, naming the coin and carrying both counts per coin. `archive_index_source`
names what was actually compared — *"flow_intensity (tree copy; the frozen anchors are imported
later in the sequence)"* — which is the honest statement: the comparison happens before the
frozen import, so the index it read is the tree's, and the receipt says so rather than implying
the frozen one.

## 4. Ruling — the ONE excused path

**Ruled: the shape is right.** A path-bound, string-typed, receipt-reported exemption is
materially better than the alternative BE avoided (narrowing the vocabulary after seeing the
failure, which is rule 11). Driven — five falsifiers, all refusing:

| emission | result |
|---|---|
| a string at `gates[].gate` (the real shape) | **passes**, `excused_paths ['gates[].gate']` |
| a **boolean** at the excused path | REFUSED — and the hit carries the value |
| an entitlement nested **inside** the excused block (`gates[].eligible`) | REFUSED |
| the same word at **another** path (`population.gate`) | REFUSED |
| the word at **top level** | REFUSED |
| (mine) a decision word with a *string* value elsewhere | REFUSED |

The vocabulary is borrowed by value from `de_admissible_windows.DECISION_VOCAB` (18 words here),
so the two cannot drift.

**The "invitation to grow" is real and measurable** — BE5-R2: I added a second path to
`DECISION_ALLOWLIST` and the suite stayed **green at 84**, because `excused_paths` reports what
was *excused in this emission*, not what *may be*. The receipt already carries the whole
`allowlist`; pinning its membership is one assertion.

## 5. The denominator is bound, not restated

`calendar_windows_per_day` reads `da_blackout_mask.WINDOWS_PER_DAY` and the receipt names the
source (`calendar_windows_per_day_source`). **There is no numeric `288` literal anywhere in the
driver** — an AST scan for the constant returns none; every textual 288 is inside a message or a
comment quoting R-411(i).

## 6. The `fwd6/` reproduction, at the artifacts

Receipt sha `a568346660a3b4db`, `outcome SCORED`, ten gates, `frozen_commit 1b53929`, manifest
`eb8733da2c8e2126`, `ledger_vs_archive.agree True`, `decision_field_check` 208 keys with
`excused_paths ['gates[].gate']`, `supersedes_receipt` absent (first write). **The sealed scores
are `cmp`-identical to `fwd5/`'s** — the 09-01 score of record reproduced by a later driver, which
is the claim that matters and it holds byte for byte.

**The diff `90036b7 → baa986d`, checked by the diff rather than by BE's word.** Its 92 added lines
sit in `selftest` (78), `population` (11) and `run_forward_day` (3). Reading the 14 outside the
suite: all are comments **except one removal** — the emitted key
`"returned_the_checked_object": _rv is rat` is deleted. So the diff **touches no emitted value**,
and it removes exactly one emitted **key** — the tautology the audit killed. Confirmed at the
artifact: `fwd6/`'s `require_verified` block still carries `returned_the_checked_object`, so a
receipt written by `baa986d` differs from `fwd6/`'s by that key's absence and by nothing else.

**What `carrying_commit 0c7ab780` + `working_tree_dirty true` means for a reader:** the driver
**sha** binds the code that ran (`4c0425c5…` = `90036b7`'s file, verifiable by `git show`), while
`carrying_commit` names the tree the process stood in and `working_tree_dirty` says other files in
that tree were uncommitted at the time. The pair is honest precisely because it does not claim the
tree was clean: the sha is the evidence about the code, the commit is context about the run.

## 7. The audit, its process error, and the 4-vs-5

**The final pass does not stand on its own at the artifact, because the audit is not in it.**
`be_forward_day.py` ships **no** mutation-audit function — the module has none, and nothing in the
85 checks re-runs the mutants. So "47/47 killed at `65da7ae0…`" is a report in a filing, not a
fact a reader can reproduce; that is BE5-R3, and it is the same rule-15 point the sibling seat's
modules satisfy by shipping `mutation_audit()` with `survivors == []` asserted in the suite.

**On the discrepancy, my independent reading:** it cannot be settled from the artifacts by anyone
but BE, and the two numbers are not necessarily inconsistent. BE disclosed that the harness and an
edit raced on the same file — the class DE hit with `__pycache__` and the coordinator has now made
programme standard (a pre-run clear). A race of that kind produces *apparent* survivors: a mutant
that never took effect, or a restore that did not, is counted as "survived" while the code was
never actually mutated. So "5 survived" (the parenthetical) and "four call-site survivors" (the
body) reconcile most naturally as **five apparent survivors of which four were real**, the fifth
being the harness artifact BE disclosed. I offer that as the reading to check, not as a finding —
and the check on it is BE5-R3: with the audit shipped, the question answers itself on the next
run.

The two-way rebuild BE describes pins **`90036b7`'s** bytes (`4c0425c578e36b2a`); the 47/47 is
reported at **`baa986d`'s** (`65da7ae0a92bff83`), which is the sha I confirmed at the pinned tip.
Those are different files, and the rebuild evidence does not carry across.

## 8. Register discipline — stated, not adjudicated

`768465a` is **one insertion and one deletion** on a single line of `COORDINATION.md`: the
Q-BE-230 row rewritten in place, and the only content difference is the mutation-audit
parenthetical (`47/47 killed at the committed bytes …`). **Nothing else in the row moved** — the
coordinator's finding reproduces at the diff. That an evidence row was rewritten in place rather
than superseded, and that its disposition column still calls the freeze disposition OPEN after
R-442 ruled it at 14:37Z, are both recorded here and left to the coordinator and BE's round 6.

## 9. Nothing else moves — with one correction to the header line

**The pinned tip's own suite is 84 checks, rc 0, under BOTH launchers** (`BE_FORWARD_LAUNCH_CHECK=1`,
which skips only the spawn). The 85th check is the launch spawn, and **today it is red**: run
under `-m`, the parent failed at that check with the child's tail showing
`/home/yuqing/ctaNew/live/pm_research/be_forward_day.py` failing its own gate-2 known-bad. That
child is the **shared tree's** file — dirty at the time (`M`, sha `0a1f7609…` ≠ the pinned
`65da7ae0…`), carrying BE's round-6 work. Minutes earlier, by path, the same check passed. So
"85 both launchers" is not reproducible at this tip and cannot be while BE34-R3 is open; the
number that is reproducible is 84. This is not a new finding — it is BE34-R3 doing what I said it
would, now producing a red rather than a stale green.

| | |
|---|---|
| 09-02 | REFUSED at gate 1 **by name** (`day_closed_calendar=False`), receipt written |
| `derived/` | listing identical before and after; nothing written by the driver |
| BE's artifacts | `fwd5/` and `fwd6/` unchanged, five entries each |
| BE34-R1 | untouched — `build_and_score` is called only from the production gate (`:1147`); `score_rows` (`:853`) is still defined and called nowhere |
| BE34-R3 | untouched — `REPO` hardcoded (`:35`), `cwd=str(REPO)` in the spawn (`:1957`) |
| BE34-R4 | untouched — the usage path still `return 0` |
| BE34-R5 | untouched — `not_frozen_in_closure` still the static closure (`:258`) |

All four are round 6's, as dispatched; none was touched here.

---

## Findings

### BE5-R1 — LOW-MEDIUM — the successor names the base, not the receipt it superseded

`_flush` computes the next free `.N` name, then records `supersedes_receipt` pointing at **`p`** —
the canonical base path — whatever N is. Driven: with base, `.1` and `.2` present, **both**
successors record `path: …_20260902.json` and the base's sha `daf932b5…`. Nothing records that
`.1` was superseded by `.2`.

Consequences for a reader: two receipts claim to supersede the same one; the supersession graph is
a star rather than a chain; and "which receipt is current" is answerable only by sorting
filenames. In my three same-second runs `.1` and `.2` are byte-identical (same `as_of`, same
content), so the filename is the **only** discriminator. The `why` text — *"an earlier run's
receipt was already here"* — is true but names the wrong earlier one.

The evidence is not lost (that was BE34-R2 and it is closed); what is wrong is the record of what
superseded what, in a programme whose rule 13 is precisely about that.

**Closure:** point `supersedes_receipt` at the highest-numbered existing receipt for that day —
the one this run actually stands after — or record the full chain (`supersedes_receipt` plus
`prior_receipts: [...]`). Either makes the order readable from the receipts rather than from `ls`.

### BE5-R2 — LOW-MEDIUM — the excused-path allowlist is reported but not pinned

`assert_no_decision_field` returns `excused_paths` (what this emission actually used) and
`allowlist` (the whole table), and the suite asserts the former equals `["gates[].gate"]`. Adding
a second entry to `DECISION_ALLOWLIST` — I used `"population.gate"` — leaves the suite **green at
84**, because the real receipt has no key at that path, so nothing is excused there and
`excused_paths` does not change.

So the allowlist can grow with no check firing until something in the emission happens to use the
new path — at which point the growth is already in the artifact. For an exemption from a rule-14
post-condition, growth should be a visible act.

**Closure:** assert the allowlist's **membership**, not just its use —
`set(DECISION_ALLOWLIST) == {"gates[].gate"}` — the idiom `de_admissible_windows` uses for
`BLIND_ENTRY_ASSERTIONS` and `de_ratification_check` for `SCOPE_OPEN_TOKENS`. The receipt already
carries `allowlist`, so the reader and the check would then be looking at the same thing.

### BE5-R3 — LOW-MEDIUM — the mutation audit is not shipped, so its result is a report rather than an artifact

The audit is the substance of this round: it killed two of BE's own evidence attempts, found four
call-site survivors, and drove the shape of `require_verified`'s closure. None of it is in the
module — `be_forward_day.py` defines no audit function and none of the 85 checks re-runs a mutant.
"47/47 killed at the committed bytes" is therefore unverifiable at the artifact, and the 4-vs-5
discrepancy is unresolvable by any reader.

This is rule 15 at the level of the harness: *a checker ships its falsifier*. The sibling seat's
modules do exactly this — `de_ratification_check.mutation_audit()` drives 31 cases against a
producer-recorded `EXPECTED_SITE` and asserts `survivors == []` inside the suite, which is why I
can re-run it every round and did again this week.

**Closure:** ship the mutant table (name → the edit and the check that must go red) and assert
`survivors == []` in the selftest, with the pre-run cache clear now standard (R-446). The audit's
own process error would then be a suite failure rather than a disclosure.

---

## Executed evidence

At `baa986d`, 2026-09-02T15:13–15:24Z:

| check | result |
|---|---|
| driver identity | `65da7ae0a92bff83` at the pinned tip |
| suite | **84 checks, rc 0, both launchers** (spawn skipped); the 85th (the spawn) red under `-m` today — BE34-R3 |
| BE34-R2 | three runs → base + `.1` + `.2`; base byte-identical; successors carry `supersedes_receipt` |
| same-run identity | `rec["_receipt_path"]`, in memory, stripped from every written receipt — unforgeable by another process |
| durable flush | a killed multi-gate run left ONE file, six gates, `outcome: None` |
| `require_verified` | call deleted → `NameError`; result faked → red by name at the PROVENANCE conjunct |
| ledger vs archive | agree True per coin; one removed → refuses naming btc; one added → naming eth; source string names the tree copy |
| the excused path | five falsifiers refuse; the real shape passes with `excused_paths ['gates[].gate']` |
| **a second excused path** | suite **green at 84** — BE5-R2 |
| the denominator | bound to `da_blackout_mask.WINDOWS_PER_DAY`; **no numeric 288 in the AST** |
| `fwd6/` | receipt `a568346660a3b4db`, SCORED, 10 gates; sealed scores **`cmp`-identical to `fwd5/`** |
| the survivor-fix diff | 78 lines in `selftest`, 14 outside — all comments **except** the removal of `returned_the_checked_object`; no emitted value changes |
| the audit | **no audit function in the module** — BE5-R3 |
| `768465a` | 1 insertion / 1 deletion; only the mutation parenthetical moved |
| BE34-R1/R3/R4/R5 | all four untouched, as dispatched |
| guards | `derived/` identical; `fwd5/`/`fwd6/` unchanged; launcher never run; three runs under `MemoryMax=12G` into my own outdirs |

---

## Disposition

- **RELEASE** for `baa986d`. **BE34-R2 is closed** — I drove the same three runs that lost the
  12:49 receipt and got a numbered successor with the earlier file intact, and "same run" is
  established by something on-disk cannot imitate. `require_verified` is the gate on evidence that
  survives its own audit; the ledger/archive check discriminates both ways; the denominator is
  bound; and the 09-01 score of record reproduces byte for byte under a later driver. **No hold.**
- **RULED (item 2):** sufficient — the data dependency makes deletion loud, and the exception-type
  assertion covers the one conjunct BE's own pair logic cannot. A healthy-day mask would take a
  rewrite, not a deletion, and no suite defends against a reimplementation.
- **RULED (item 4):** the shape is right — path-bound, string-typed, reported, with the vocabulary
  borrowed by value. Its one weakness is governance, not design: pin the allowlist's membership
  and growth becomes a deliberate, visible act (BE5-R2).
- **FILED:** **BE5-R1** (the successor names the base), **BE5-R2** (the allowlist unpinned),
  **BE5-R3** (the audit unshipped).
- **Stated, not adjudicated (item 8):** `768465a` moved only the mutation parenthetical; the row's
  stale disposition column and the in-place rewrite are the coordinator's and BE round 6's.
- **Correction to the round's header:** the reproducible figure at this tip is **84 both
  launchers**, not 85 — the 85th check's subject is another tree's file, and today that file is
  red. BE34-R3, already open.
