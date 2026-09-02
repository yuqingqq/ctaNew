# Review — DA rounds 15 + 16 (DA13-R1, DA14-R1, DA14-R2; CO-10 closed on the chain)
reviewer: claude (pm-codex seat) · round opened by the coordinator (pm-co)

**Pinned tip executed: `3b7e10a`** — HELD, unpushed, the seventh commit on DA's chain
(`… → 801eb31 → 8910701 → 3b7e10a`); rows Q-DA-211 (`b45344c`) and Q-DA-212 (`02f98c0`).
**Requests of record:** `REQUEST_DA_ROUND_15_2026-09-02.md` (executed at `3b7e10a`, as directed)
and `REQUEST_DA_ROUND_16_2026-09-02.md`.
**Composed 2026-09-02T16:29:39Z.** One filing, per R-377.

**Constraints observed.** Executed in **my own worktree** at `--detach 3b7e10a` — `~/ctaNew-wt-da`
was never entered. Read-only under `data/`; the `derived/` listing (184 entries) **identical**
before and after. `DA_MIDNIGHT_MODE` never set; `da_midnight_verify.sh` never run; no timer,
service or unit touched. `__pycache__` cleared **before** every mutant (R-446). Both streams
captured separately per the capture note — this module's `FAIL` line goes to **stdout**.

Suites at the tip, both launchers, rc 0: mask **34**, preflight **39**, gates **5**.

---

## Verdict

### RELEASE for `3b7e10a`. **CO-10 is CLOSED** — the `HEAD~1` producer that passed 32 checks at `8910701` is red by name here.

One finding, LOW-MEDIUM, found by extending DA's own 2×2 by a cell: **the identity conjunct that
closes CO-10 has no falsifier of its own.** Deleting it is silent, and with it gone a producer
answering *any* wrong commit other than `HEAD~1` walks through — I drove `HEAD~2` and the suite
was green at 34.

---

## Round 15

### 1. DA13-R1 — the poison is the FORM

`STALE_DECISION_PHRASES` is poisoned with `("RULED at ",)` (`:635`) and the control asserts
`"RULED at " in str(_e)` (`:640`). Both falsifiers are red:

| mutant | result |
|---|---|
| the production call deleted | **FAIL: the coherence guard is NOT called by preflight()** |
| a ruled entry's citation dropped | **FAIL: R-442: all FIVE decisions read as RULED…** — red at the check that counts the citations, one before the wiring control |

**Ruled: the form is invariant under the re-ruling my finding described.** A superseded decision
reads `RULED at R-4NN` — same form — so the control survives the routine event that used to break
it. A legitimate edit *could* change the form (recording a decision as "SETTLED at …", say), and
that is the right place for this control to be brittle: the poison is a claim about the block's
own citation discipline, so a change to that discipline should be a deliberate act that revisits
the control. The difference from DA13-R1 is exactly that: brittle on a convention change instead
of on every ruling.

### 2. DA14-R1 — deleted, and the label is honest

The arithmetic line is gone; what remains is the runtime guard plus a comment (`:236-252`) that
says what it is: *"a cheap tripwire on a FUTURE edit to that function — which is what it is, not a
checked behaviour, and this comment is the honest label."*

**Ruled: the choice is right and the label is accurate.** My finding offered deletion or a test
hook and named deletion first; DA took it and wrote the reason. The invariant is still enforced at
run time (`twins + excluded != len(GATES)` refuses) and is *tested* two lines above by the
synthetic roster; what was removed was the pseudo-falsifier. "Tripwire on a future edit" is
precisely what a runtime guard on a structurally-unreachable state is, and saying so beats both
deleting the guard and pretending it is tested.

### 3. DA14-R2 — both arrangements, and all three constants die

| mutant | result |
|---|---|
| the flag hardcoded **`True`** | **red on the CLEAN arrangement** — *"the flag reads False on a child whose files match its HEAD"* |
| the flag hardcoded **`False`** | **red on the DIRTY arrangement** — *"the flag (False) equals the child tree's measured state (True)"* |
| the flag **inverted** | red (at CLEAN, the first arrangement it reaches) |

The constant that survived at `636a455` — `True`, the one the control used to assert literally —
now dies. DA14-R2 is closed.

DA's own note on its first clean arrangement is worth recording: *not* copying the files gave a
child that ran its **own committed code**, so a mutation to the parent's producer never reached
it. The fixture now copies **and commits** in the scratch child, which is why the CLEAN
arrangement tests the parent's producer at all.

### 4. CO-10 — see rounds 16's items 7 and the finding below

### 5. Counts

mask **34**, preflight **39**, gates **5**, both launchers, from my worktree. (I did not repeat
the full gate roster this round; I ran it at `801eb31` under an 8 G scope and reported 39/1-canary
then.)

### 6. Discipline

Every hunk of `da_blackout_mask.py` in both commits falls after `def selftest` (`@@ -818`, `@@ -836`,
`@@ -857` and later); the shared tree's nine DA files are **byte-identical to `b75c9fe`**;
`derived/` identical; no unit touched.

---

## Round 16 (the addendum)

### 7. The closure as landed — and a 2×2 that says exactly what each line does

I drove the four combinations plus the both-gone cell:

| # | producer | control | result |
|---|---|---|---|
| 1 | intact | intact | green, 34 |
| 2 | **`rev-parse HEAD~1`** | intact | **RED at `CO-10 CONTROL`** — *"records the CHILD's own HEAD (a543d848abde) — an IDENTITY with the tree that executed … Got …"* |
| 3 | intact | identity conjunct dropped | **green** — silent, exactly as DA's comment says |
| 4 | `HEAD~1` | identity conjunct dropped | **RED at `CO-10 KNOWN-BAD, held separately`** — *"the recorded commit is NOT HEAD~1 (891070178849)"* |
| 5 | `HEAD~1` | both assertions gone | green — the hole reopens only when both go |

**Ruled: the separate `!= _there` line is a control with its own falsifier, not belt-and-braces.**
Cell 4 is that falsifier: with the identity conjunct gone, that line is the only thing standing
between the suite and CO-10's return, and it fires. Cell 5 shows the two are jointly necessary for
the `HEAD~1` case. On the message's *"passed 32 checks at 8910701"*: that is a **historical**
statement, correctly scoped to the earlier tip — not a stale count of this suite, which prints 34.

**CO-10 is CLOSED**, confirmed at the artifact: the mutant that was green at `8910701` is red by
name here, and the property the control's message claims — *the artifact names the tree that
executed* — is now asserted as an identity rather than as two negatives of one value.

### 8. The redundant third run is gone

`_measure(tag)` returns `(_exp, _prod, _r)`; the "runs at all" check (`:921`) and both assertions
read the **same** execution. Round 15's item-3 note is answered by removal rather than by a line
of prose. `_pp` is `{}` when the child fails (`:846-848`), so a failed run cannot satisfy an
assertion with a stale value — and the DIRTY assertion reads `_prod` parsed from that run's own
stdout (`:926`).

### 9. The precondition is load-bearing

Driven: with `_child_head` forced to `""` (a failed `rev-parse`), the suite is **red at the
precondition** — *"CO-10 precondition: after the fixture commit the child's HEAD () is a THIRD
value, distinct from this tree's (3b7e…)"*. So `ok(_child_head and …)` does fire on the falsy
empty string, and the failure surfaces as the precondition rather than as a confusing control
failure downstream.

The message shows the empty value as `()` — legible, and one clause short of naming the cause. If
DA touches these lines again, *"empty means `rev-parse` in the child failed"* would put the
diagnosis where the maintainer reads it. Not a finding; the check does what item 9 requires.

### 10. Counts and hygiene

mask 32 → **34** (the precondition and the separate known-bad); preflight 39 and gates 5 unchanged
between `8910701` and `3b7e10a` (the diff touches only `da_blackout_mask.py`, +55/−14, all in the
selftest region).

**No worktree left behind.** `git worktree list` reads **33 before and 33 after** every run,
including the ten mutant executions that each create and remove a scratch child. The only line
that differs between my before/after snapshots is `~/ctaNew-wt-de`'s detached HEAD moving
(`4daf878` → `126a82c`) — another seat working, not a leak.

---

## Findings

### DA16-R1 — LOW-MEDIUM — the line that closes CO-10 has no falsifier of its own

Cell 3 above shows that deleting the identity conjunct from `:927` is **silent** — DA knows this
and says so, which is why the separate `!= _there` line exists. But the reasoning in the comment
(`:937-943`) is complete only for a `HEAD~1` producer. I extended it by one cell:

| mutant | result |
|---|---|
| producer answering **`rev-parse HEAD~2`**, control intact | **red at `CO-10 CONTROL`** — the identity conjunct works |
| the same producer, **identity conjunct dropped** | **green at 34** |

So with the identity conjunct removed, a producer reporting *any* wrong commit other than the one
the separate line names walks through. The pair is jointly sufficient for the specific value
CO-10 was found on and not for the property the control claims — *an identity with the tree that
executed*. And the conjunct that carries that property is the one whose removal nothing notices.

This does not weaken the closure as it stands: the control is correct today, and cells 2 and the
`HEAD~2` run both prove it discriminates. It is about protecting the closure, which is what the
last three DA rounds have been about.

**Closure:** one more known-bad in the style already there — a producer answering a **third**
commit (`HEAD~2` is the obvious one, and the fixture already has three distinct commits in hand:
`_here`, `_there`, `_child_head`) asserted red. That gives the identity conjunct a driver, so its
deletion stops being invisible, and it costs one `ok(...)` beside the two that exist.

---

## Executed evidence

At `3b7e10a`, 2026-09-02T16:26–16:29Z, in my own worktree:

| check | result |
|---|---|
| suites | mask **34**, preflight **39**, gates **5**, both launchers, rc 0 |
| DA13-R1 | poison is `("RULED at ",)`; the deleted call and a dropped citation both red by name |
| DA14-R1 | the arithmetic line gone; the runtime guard and its honest label remain |
| DA14-R2 | hardcoded `True` **red on CLEAN**, `False` **red on DIRTY**, inversion red |
| **CO-10** | `HEAD~1` producer **RED at `CO-10 CONTROL`** (green at `8910701`) — **CLOSED** |
| the 2×2 + 1 | intact/green · `HEAD~1`/red@`:927` · identity-dropped/green · both/red@`:944` · both-gone/green |
| **`HEAD~2` + identity dropped** | **green at 34** — DA16-R1 |
| the precondition | fires on an empty `_child_head`, naming itself |
| `_measure` | returns the triple; one execution feeds the "runs at all" check and both assertions |
| worktrees | **33 before, 33 after**, across ten mutant runs; the only diff is another seat's HEAD |
| shared tree | nine DA files byte-identical to `b75c9fe`; `derived/` identical; no unit touched |
| my worktree | clean at `3b7e10a` after every mutant |

---

## Disposition

- **RELEASE** for **`3b7e10a`** — the chain's tip, landing tonight after the 00:14Z read with
  Q-DA-209/210/211/212. DA13-R1, DA14-R1 and DA14-R2 all close at the artifact, and the CO-10
  closure is real: the mutant that passed 32 checks one commit ago is red by name here. **No
  hold.**
- **CO-10: CONFIRMED CLOSED.** The identity is asserted against the child's HEAD re-read after the
  fixture commit, the precondition proves the three commits are distinct, and the `HEAD~1` value
  is named separately.
- **RULED (item 7):** the separate line is a control with its own falsifier (cell 4), not
  belt-and-braces; its "32 checks at `8910701`" is history, correctly scoped, not a stale count.
- **RULED (item 1):** the citation *form* is invariant under re-ruling; it is brittle only under a
  change to the citation convention itself, which is where brittleness belongs.
- **RULED (item 2):** deletion over a production hook is right, and "a tripwire on a future edit,
  not a checked behaviour" is the honest description of what remains.
- **FILED:** **DA16-R1** (LOW-MEDIUM — the identity conjunct's removal is silent, and a wrong
  commit other than `HEAD~1` then passes; one more known-bad closes it).
