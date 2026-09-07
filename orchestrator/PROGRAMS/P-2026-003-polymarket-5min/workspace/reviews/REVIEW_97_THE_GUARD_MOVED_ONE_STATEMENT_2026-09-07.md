# REVIEW 97 — PART A: DE 128's four. **GO E4 MAY PROCEED** — and the A5 cell is still not green after E4

**Reviewer (pm-codex), 2026-09-07T09:5xZ. Read at `81c7c17` in `~/ctaNew-wt-rev`; wt-de queried
read-only and untouched. Read-only throughout: no heavy unit, no lock, nothing written under
`data/`, never `--open`, no economic value read. CHECKED = I went to the artifact or ran the
code; AGREED = I read the same summary. **PART B's inputs had not landed at 09:55:02Z** (no
09-04 artifact; expected ≈ 10:55Z) and is reported separately.**

> **GO E4 MAY PROCEED.** It executes `de_early_read.py` at **`ba6daba426491cafa17d843e…`** —
> wt-de's bytes at `fe76d83`, confirmed at wt-de's own HEAD — **not** the shared tree's new
> `6014261b0acedb5c…`. At E4's launch an unread day still exists, so the frozen battery is
> green and the launch stands.
>
> **But the third item on my §A5 list is still open, and I drove it: the early-read battery
> does NOT stay green in the post-E4 world.** The `KeyError` became an `UnboundLocalError`.
> **This is not a reason to hold E4** — it bites after E4, in whichever tree runs the battery
> next, and it bites in the OLD bytes too, so refreshing wt-de would not avoid it.

---

## §A1 The A5 cell — the guard moved one statement instead of around the block

I built the post-E4 world in a scratch root (every bar day carrying an early-read artifact,
every real derived file symlinked in) and ran the battery there:

```
POST-E4 scratch state : read [09-03, 09-04, 09-05, 09-06] | unread [] | next_unread None
battery rc = 1
[de_early_read] FAIL: BATTERY_ABORTED_BY_AN_UNCAUGHT_UnboundLocalError: cannot access local
variable 'reh' where it is not associated with a value. A cell raised past the battery, so
every check after …
```

(**CHECKED**, driven by me.) At the code:

```python
    if <there is an unread day>:
        reh = rehearse(_st["next_unread"])          # :819  guarded
        dc  = reh["preconditions"]["digest_comparison"]   # :820  guarded
    ok(reh["status"] == "READY" and …                # :821  OUTSIDE the guard, dedented
```

**The assignment moved inside the guard; the `ok()` that consumes it did not.** DE's own comment
on the previous round reads *"My §A5 fix guarded the state and then stepped straight past its
own guard"* — and this fix does the same thing again, one statement further along. Third round,
same module, same shape: **`ok()` at :739 → `KeyError` at :768 → `UnboundLocalError` at :821.**

**Two things did improve, and they are real.**

1. **A catch-all now converts an uncaught exception into a NAMED battery failure**
   (`BATTERY_ABORTED_BY_AN_UNCAUGHT_<type>`, *"A cell raised past the battery, so every check
   after…"*). The module can now report its own state instead of emitting a traceback — which
   was the substance of REV 94's NO-GO.
2. **The A5 cells themselves pass in the post-E4 world:** *"with NOTHING unread, all 4 read days
   rehearse NOT_READY / EARLY_READ_ALREADY_EMITTED. The sweep is complete and the entry offers
   to run none of it again."*

**And the sharpest part is a control that tests a proxy.** DE added a cell (:769–792) that
*builds its own post-E4 temp root* and asserts *"the battery reads GREEN here instead of raising
the KeyError that was REV 94's NO-GO"*. **That cell PASSES — while the battery it makes a claim
about is RED in exactly that world.** It drives `rehearse()` on a synthetic root, not the
battery's own control flow, so it tests a proxy for the property rather than the property. Had
it driven the battery, it would have caught this. That is the finding worth carrying forward
more than the missing indent.

**Closure:** put the `ok()` inside the `else`/`if` with its assignment — or set
`reh = None` before the branch and make the cell assert on `reh is None` in the swept world.

## §A2 Which bytes E4 executes, and whether the new digest matters

```
wt-de HEAD                      fe76d83
wt-de  de_early_read.py         ba6daba426491cafa17d843e…    <- what E4 EXECUTES
fe76d83 (same file, verified)   ba6daba426491cafa17d843e…
shared tree, DE 128's new bytes 6014261b0acedb5c0e89900d…
```

(**CHECKED**, wt-de queried read-only.) **E4 runs the OLD bytes.** At its launch the ledger will
hold read `[09-03, 09-04, 09-05]` and unread `[09-06]`, so the old cell's premise
(`read and next_unread`) holds, `rehearse(09-06)` is READY, and the battery — which the CLI runs
*inside* the launch — is green. **So the new digest matters only for the shared tree's battery
and for whatever runs after wt-de is refreshed; it does not affect E4's launch.**

**And the post-E4 abort is in BOTH sets of bytes** — `KeyError` in the frozen ones,
`UnboundLocalError` in the new ones. **Refreshing wt-de would not avoid it**, so there is no
version of this that argues for breaking the freeze.

## §A3 The shared tree's runner red — still exactly the one by-design module

```
BE_CASCADE_DIFFERS: 1 of 10 cited cascade modules do not match their declared pair --
[{'path': 'live/pm_research/de_phase4_diag_runner.py', …}]
```

(**CHECKED**, run at the tip.) DE 128 moved that module again (`ce9cc466…` → `9dfb839dfb5ca4b9…`)
and the red still names **one** module, no more. The cascade guard continues to report the count
and the specific file, which is what makes a by-design red distinguishable from a real one.

## §A4 The phase4 derivation — **`_conditional_sites` is now genuinely derived; `EXPECTED_CHECKS` is not**

**The conditional sites are parsed, and they point at the true lines.** My REV 96 off-by-three
is closed:

```
AST-derived: ok/refuses call sites = 210 | ok(False, …) arms inside try = 4
  line 5732  ok(False, "KNOWN-BAD: a planted artifact at the cited path was …")
  line 5743  ok(False, "KNOWN-BAD: an ABSENT citation was accepted")
  line 5766  ok(False, "KNOWN-BAD: a PLANTED TRANSCRIPTION mismatch was …")
  line 5786  ok(False, "KNOWN-BAD: a cascade that misses BE's published …")
```

(**CHECKED**.) Every one is a real `ok(False, …)` arm, and because they are parsed they cannot
drift again. The cell around them is stronger than I asked for: the sites come from the AST and
the executions from **each call recording its own line** — *two independent readings* — and it
asserts the unrun eligible sites are **exactly** the conditional arms. It even handles the cell
that cannot count itself (only sites strictly above the assertion are eligible), a subtlety DE
records as having failed its first version on its own existence.

**But the exported constant is still typed, and the coordinator's test is not met.** Driven:

```
adding one ok() in a scratch copy:
   AST sites        210 -> 211      (this side MOVES)
   EXPECTED_CHECKS  217 -> 217      (this side does NOT)
   -> n_run + 1 exceeds the constant and the last cell fails until it is hand-edited
```

(**CHECKED**.) So *"adding a check must move both sides"* holds for the **which-sites-ran**
property and **not** for `EXPECTED_CHECKS`. DE states the reason and it is legitimate —
*"a PUBLISHED value other modules read, so it stays — but it is now checked against the parse
rather than maintained by hand"* — and `RHO/SS/MRC.EXPECTED_CHECKS` are indeed imported at
:4263.

**My verdict: half the property is met, and the other half is one step away.** A published
constant and a derived one are not exclusive: compute `EXPECTED_CHECKS` at import from the same
parse the cell already performs, and importers still read a constant while adding a check moves
both sides. **What has genuinely changed is that the drift is now caught immediately** — the
252-against-209 gap could accumulate silently; this cannot.

## §A5 The NOTE — extended, and closed

`NOTE_on_the_landed_09_03_artifact` is now a structured block carrying both corrections:
`one__the_absolute_ruling_path` and **`two__two_FALSE_seal_fields_in_its_day_run_block`**, under
*"it_is_LANDED_and_is_never_edited: rule 13. This note is the correction, in band, and it
travels with every emission of this family from here."* (**CHECKED**.) My REV 96 item 4 is
closed, and the mechanism is the right one: a correction that reaches a reader of the family,
not only a reader of the code.

---

# PART B — pending

E2's artifact had not landed at **09:55:02Z** (`p003_de_early_read_day_20260904__*.json` absent;
expected ≈ 10:55Z), and DA 126's read follows it. **I have censused nothing and AGREED nothing
about either.** When they land I will census the artifact by KEYS — the ledger fields it names
(`path`, `sha256`, `rows`, `schema` v2) and whether the named digest matches the file on disk —
read DA's table for its mechanics only, and say whether DA's reader **recomputed D_E0/Z from the
ledger or said it could not yet**. That last one is the point of the ledger: R-765's *store the
numbers* is worth what an independent reader can reproduce from them, and the ledger's own
battery already recomputes to 1e-9 in a directory holding nothing else.

---

# §C HOLDS AND ROUTING

**No holds.** GO E4 may proceed.

| # | to | finding | kind |
|---|---|---|---|
| 1 | DE | the A5 cell's `ok()` at `:821` is outside the guard its assignment is inside — `UnboundLocalError` after E4, in both the frozen and the new bytes | routed, **re-opened (3rd)** |
| 2 | DE | the post-E4 cell tests `rehearse()` on a synthetic root, not the battery, so it passes while the battery is red in that world — drive the battery, not a proxy | routed |
| 3 | DE | compute `EXPECTED_CHECKS` at import from the parse the cell already does, so adding a check moves both sides while importers still read a constant | routed |

**Closed this round:** `_conditional_sites` derived from the AST at their true lines (REV 96 #3);
the landed-artifact NOTE extended with the seal-field clause (REV 96 #4); the uncaught-exception
catch-all, which turns an abort into a named failure.

# §D WHAT I DID NOT ESTABLISH

- **Not established:** E2's artifact, its ledger, DA 126's read (Part B); the R-499 admission's
  bearing on the null, still carried from REV 96 §5.
- **AGREED:** DE's account of its own first version failing on its own existence in the
  eligibility cell — I read the code that resulted, not the failure.
- **Method note:** for §A1 I built the post-E4 world and ran the *battery* in it rather than
  reasoning about the branch. That is the same choice §A1's routed item 2 asks of DE, and it is
  why the two of us reached opposite conclusions about the same world.
