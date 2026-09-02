# Review — DE rounds 7–9 (the bridge, the ratification predicate, the consumed block)
reviewer: claude (pm-codex seat) · round opened by the coordinator (pm-co)

**Pinned tip executed: `b98421d`** (rounds 7/8/9: `79c6249`, `575f076`, `b98421d`).
**Request of record:** `REQUEST_DE_ROUNDS_7-9_2026-09-02.md` (at `fb57b8b`).
**Composed 2026-09-02T11:29:42Z.** One filing, per R-377.

Executed in `~/ctaNew-wt-rev` at `--detach b98421d`. Read-only under `data/`; every
register known-bad was built on a **copy in a temp dir** — `COORDINATION.md` was never
written. No timer, no service, no launcher.

Nothing below is accepted from a filing, from R-419 or from R-421 — including one claim
in R-421 §6 that I could not reproduce.

---

## Verdict

### RELEASED. The bridge, the binding and the block-consumption all do what rounds 7–9 claim.

Four findings, none holding — **DE-R1** (the bridge carries an identity it never checks),
**DE-R2** (the reads-no-verdict predicate is defeated by a dynamic import, and the module
now contains that mechanism), **DE-R3** (four block fields are silently optional,
including the one CO-R1 is about), **DE-R4** (one refusal path has no control) — plus a
**correction to the register**: the claim that the driver refuses ledger-only windows is
not true at the artifact, and it bears directly on item 4's answer.

Both judgements the dispatch asked for are given, and I disagree in part with the
dispatched disposition for CO-R1.

---

## 1. The bridge (round 7) — it never chooses, and it carries one thing it does not verify

On the real 09-01 supply: **1,875 specs, in the supply's order**, each carrying the
supply's identity (`mask_identity_hash`, `supply_protocol`, `day`) beside the seam's
`slug` and `inputs_hash`. First three specs equal the first three supplied windows.

**Plane order holds.** `de_admissible_windows`'s parsed imports are stdlib plus
`da_blackout_mask`, `da_content_liveness_rule`, `importlib` — no seam. Its single textual
mention of `ev_replay_seam` is a **filename inside a list of strings** (line 844), not an
import. DE does not import EV.

| red-first | result |
|---|---|
| supply count disagrees with its windows | **REFUSED** by name |
| one window dropped from a coin | **REFUSED** by name |
| both identity fields absent | **REFUSED** by name |
| **`mask_identity_hash` tampered, identity block intact** | **ACCEPTED** |
| **identity block tampered, hash intact** | **ACCEPTED** |

### DE-R1 — MEDIUM — the bridge stamps an identity it never checks against itself

The supply carries **both** `mask_identity` (the block) and `mask_identity_hash`, and I
verified the hash is exactly `de_admissible_windows._sha(mask_identity)`. The bridge
copies both into 1,875 specs and never compares them, so either half can be replaced and
every spec then carries an identity that does not identify its own population. The
request lists "an identity hash that does not match" among the refusals this bridge must
have; it is the one that is missing.

**Closure:** one line at the bridge — `_sha(sup["mask_identity"]) == sup["mask_identity_hash"]`,
refusing by name otherwise. Both halves are already in hand; nothing new is read.

## 2. The binding (rounds 8–9) — identity, not vocabulary

Six known-bads, each on a copy of the register:

| known-bad | emission |
|---|---|
| (a) the block **quoted in prose**, unfenced | **REFUSED** — *"carries no ratification block; prose binding is not admissible after R-419 … an entry that merely QUOTES a ratification carries all of its vocabulary"* |
| (b) block `ref: R-419` under heading `R-9102` | **REFUSED** — *"the block declares ref 'R-419' while the entry heading is 'R-9102'"* |
| (c) `population: SAMPLED` | **REFUSED** — names a population it will not evaluate as full |
| (d) `sampling: STRATIFIED` under a FULL population | **REFUSED** — *"ratifies FULL_SUPPLIED_COMPLEMENT but declares sampling='STRATIFIED'"* |
| (e) a later entry with **no block** | **REFUSED** by name |
| (f) block missing `scope_to` (**CO-5**) | `verified True`, `binding_source BLOCK`, `unverifiable ['day_in_scope']` — **reproduced exactly** |

This is the CO-4 hole closed properly: the coordinator's own recap-entry fixture now
refuses, and it refuses for the right reason (no fenced block), not because of a
vocabulary test.

### DE-R3 — MEDIUM — four fields are silently optional, including the one CO-R1 turns on

The request asks whether any **other** missing field is undecided-but-verified. I removed
each field in turn:

| missing field | result |
|---|---|
| `kind`, `population`, `sampling` | **REFUSED** by name |
| `scope_from`, `scope_to` | `verified True`, `unverifiable ['day_in_scope']` — honest |
| **`present_source`** | **`verified True`, `unverifiable []`** |
| `scope_days` | `verified True`, `unverifiable []` |
| `revocable_by` | `verified True`, `unverifiable []` |
| `supersedes` | `verified True`, `unverifiable []` (defensible: absence = supersedes nothing) |

`present_source` is the field that says **where `present` comes from** — the whole subject
of CO-R1 — and a ratification without it verifies clean. Worse, a ratification **with a
nonsense one** verifies clean too: `present_source: /etc/passwd` → `verified True`,
`unverifiable []`. `scope_days` absent means a ratification that does not say which days
it covers verifies for any day, and `revocable_by` absent removes the record that the USER
can revoke it — both silently.

**Closure:** treat the field list as declared-and-required (refuse a missing
`present_source`/`scope_days`/`revocable_by` by name, as `kind`/`population`/`sampling`
already are), or at minimum surface them in `unverifiable` the way `scope_from` is. The
distinction that matters is between *a field the checker cannot decide* and *a field
nobody supplied* — today those look identical in the emission.

## 3. Supersession — directional, and my answer on the stamp

`check(sup, "R-418")` refuses: *"REFUSED FOR A NEW RUN: R-418 is SUPERSEDED by R-419. A
receipt already carrying R-418 is provenance and stays valid — this refusal is about
starting a run under a ratification that is no longer the one in force, not about
rewriting history."*

**Directionality verified:** I built a register in which an **earlier** entry's block
claims `supersedes: R-419`; R-419 still verifies. The scan is bounded to later entries.

**CO-R3, and my answer to "which stamp".** The message states the right rule and the API
cannot express it: `check(sup, "R-418")` refuses whether the caller is starting a run or
re-verifying a receipt written this morning. So the fix has two parts, and the stamp is
the smaller one.

**The right stamp is the receipt's own `as_of_utc`, compared against the superseding
entry's heading timestamp.** Not the harness commit, not the register's line at that
commit:

- the receipt's as-of is **inside the receipt**, so a consumer needs only the receipt and
  the register — no git, no commit resolution, no working tree;
- a harness commit answers *what code ran*, which is a different question from *which
  ratification was in force*; the register is the authority on the second;
- "the register's line at that commit" reintroduces git and is ambiguous while entries
  are amended in band;
- both entries already carry a heading timestamp (`10:53Z`, `11:03Z`), so the comparison
  is `receipt.as_of_utc < superseder.entry_time → the ref was in force`.

The caveat to state in the code: the receipt's as-of is **self-reported**, so it is
provenance-grade, not adversarial-grade. That is the right trade here, and it is the field
the programme already uses for this purpose (`supersedes.as_of_utc`).

**The other part, which matters more:** `check()` needs a way to be told it is verifying
an **existing** receipt — e.g. `receipt_as_of=None` meaning new-run semantics (refuse if
superseded), and a supplied value meaning in-force-at-T. Without that parameter no stamp
choice helps, because the caller cannot say which question it is asking.

## 4. `day_in_scope`, and my judgement on CO-R1 — the checker predicate does not suffice

`day_in_scope` decides as specified (`null` = open; `day <= scope_to`; a day before
`scope_from` is False), and R-419 verifies with `unverifiable []`.

On CO-R1 the dispatch says the disposition is a `day_closed` predicate in the checker
rather than a restatement of R-419. **I disagree in part, and one premise does not hold.**

**The premise.** R-421 §6 records that *"the driver already refuses open days and
ledger-only windows"*. The first half is true: `be_forward_day.assert_day_closed_and_attributed`
refuses when `day_closed_calendar` is not true. **The second half is not.** I looked for
any tape read in the chain:

| module | references to `scan_day` / `raw/` |
|---|---|
| `be_forward_day.py` | **0** |
| `de_admissible_windows.py` | **0** |
| `ev_replay_seam.py` | **0** |
| (`da_blackout_mask.py`, for contrast) | 2 |

`present_from_ledger` reads the ledger and hands it straight to `supply()`. Executed: a
`present` carrying **one window the tape does not have** is supplied as admissible —
`n_supplied_total` 1,876, and the tape-less window is emitted in the specs.

**So my judgement.** A `day_closed` predicate in the checker is **necessary but not
sufficient**, and the textual fix is not the primary one either:

- the checker *verifies a supply against a ratification*; it does not gate the
  **production** of `present`, so a `day_closed` predicate tells a reader afterwards
  what the driver already refused beforehand;
- on a **closed** day the ledger and the tape agreed exactly on 09-01 (2,016 = 2,016) —
  but that is an empirical agreement, not a checked one, and **nothing in the chain would
  notice if they diverged on a closed day**: a market that existed while its window file
  was lost or never written is precisely the blackout shape this programme keeps finding;
- therefore the missing predicate belongs in the **producer of `present`**, not only in
  the checker.

**The cheapest complete fix, and it needs no USER restatement:** have
`present_from_ledger` compare the ledger's windows against the day's tape window set and
either intersect (with the difference reported as a named count) or refuse. R-419's text
stays true — `present_source` is still the ledger — while "population" becomes
well-defined. If instead the text is changed to name tape-backed windows, that is R-419's
substance and the USER restates it; I do not think that is necessary to make the
population well-defined, and it costs a USER turn.

Add the checker's `day_closed` predicate too: it is cheap, and it makes the closure
visible in the emission rather than only in the driver's refusal.

## 5. The grandfather — pinned in both directions

`GRANDFATHERED_PROSE_REFS == ("R-418",)`.

| mutant | result |
|---|---|
| a **second** ref added to the tuple | **KILLED** — *"prose survives for EXACTLY ONE grandfathered ref, named with its reason"* |
| the tuple **emptied** | **KILLED**, same check |

The control pins the tuple's contents, not merely its non-emptiness, so neither widening
nor removing it passes.

## 6. The suite tests what it says

42 / 47 / 64 checks (`de_ratification_check`, `de_admissible_windows`, `ev_replay_seam`)
under **both** launchers, rc 0.

**Refusal-path coverage, measured by deleting each branch in turn** (10 raise sites):

- **9 KILLED**, each by a *named* check — matching Q-DE-27's count of nine refusal paths;
- **1 SURVIVED**: the `population not in KNOWN_POPULATIONS` branch (line 360). It is
  reachable — `population: SOMETHING_NEW` refuses by name when I drive it — but **no
  control exercises it**. DE's "nine" describes its coverage, not its code.

### DE-R4 — LOW — one refusal path exists, works, and is proven by nothing

**Closure:** one known-bad with an unrecognised population name; the branch already
refuses correctly, so this is a control, not a fix.

### DE-R2 — MEDIUM — `reads_no_verdict` is defeated by a dynamic import, and the module now contains the mechanism

`imported_modules` walks the AST for `Import` / `ImportFrom` — strictly better than a
grep, and I confirmed last round that a verdict producer named in a **string** does not
trip it. But:

```
static  : "import da_forward_day_verify"                       -> imports {da_forward_day_verify}, reads_no_verdict False
DYNAMIC : "importlib.import_module('da_forward_day_verify')"   -> imports {importlib},            reads_no_verdict TRUE
```

And `de_admissible_windows` itself now calls `importlib.import_module(...)` (line 805,
for the dual-module-identity probe). So the evasion mechanism is already in the file that
the predicate is supposed to certify, and the certification cannot see it. Not a live
defect — the dynamic import there names the liveness rule, not a verdict producer — but
the guarantee reads stronger than it is.

**Closure:** extend the predicate to `importlib.import_module(...)` / `__import__(...)`
with a literal argument, and treat a **non-literal** argument as unresolvable — refusing
rather than passing, since "I cannot tell what this imports" is not "it imports nothing".

## Rule 10 / rule 14

`decides: "nothing -- this reports; admission is the coordinator's …"` is emitted **in
the artifact** (not only in the docstring) and is asserted by its own check, so a consumer
reads it where they read the result. The module docstring states it performs no
ratification and admits no day. No printed conclusion beside a predicate. `verified` is
consistency, and the emission says so.

---

## Executed evidence

At `b98421d`, 2026-09-02T11:22–11:29Z:

| check | result |
|---|---|
| three suites, both launchers | **42 / 47 / 64**, rc 0 |
| bridge on the real supply | **1,875 specs**, supply order, both identities carried |
| bridge red-first | count / dropped / both-absent refuse; **hash-vs-block never compared** — DE-R1 |
| plane order | supplier imports no seam (the one mention is a filename string) |
| six register known-bads | (a)–(e) **refuse by name**; (f) CO-5 reproduced exactly |
| every other field removed in turn | 3 refuse; 2 honest; **4 silently verified** — DE-R3 |
| `present_source: /etc/passwd` | `verified True`, `unverifiable []` |
| supersession directionality | an earlier block claiming to supersede R-419 leaves it verified |
| R-418 | refuses **for a new run**, naming R-419 |
| grandfather tuple, widened / emptied | **both KILLED** |
| refusal-path deletion, 10 sites | **9 killed by name, 1 survivor** — DE-R4 |
| `reads_no_verdict` vs a dynamic import | **passes** — DE-R2 |
| tape reads in driver / supplier / seam | **0 / 0 / 0**; a tape-less window is supplied (1,876) |
| ledger vs tape, 09-01 (closed) | 2,016 = 2,016, exact |
| register file | never written; all known-bads on temp copies |

---

## Disposition

- **RELEASED:** DE rounds 7, 8 and 9. **No hold.**
- **FILED:** DE-R1 (verify the identity the bridge stamps), DE-R2 (dynamic imports evade
  the no-verdict predicate), DE-R3 (four silently optional fields, `present_source` among
  them), DE-R4 (an uncontrolled refusal path).
- **Judgement, item 3:** the receipt's own `as_of_utc` against the superseding entry's
  timestamp — and `check()` needs a parameter that says which question is being asked.
- **Judgement, item 4 — a partial disagreement with the dispatched design:** the checker
  predicate does not suffice, and the fix belongs in `present_from_ledger` rather than in
  R-419's text. R-421 §6's "the driver already refuses … ledger-only windows" does not
  hold at the artifact: no layer in the chain reads the tape, and a tape-less window is
  supplied as admissible.
