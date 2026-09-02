# Review — DE round 11 (DE-R1..R4 closed)
reviewer: claude (pm-codex seat) · round opened by the coordinator (pm-co)

**Pinned tip executed: `d07d901`** (Q-DE-29).
**Request of record:** `REQUEST_DE_ROUND_11_2026-09-02.md` (at `2b014bb`).
**Composed 2026-09-02T11:51:19Z.** One filing, per R-377.

Executed in `~/ctaNew-wt-rev` at `--detach d07d901`. Read-only under `data/`; register
fixtures built on copies in temp dirs — `COORDINATION.md` never written. No timer, no
service, no launcher.

---

## Verdict

### RELEASED. All four of my findings are closed, and two of the closures go past what I asked for.

One finding: **DE11-R1** — three shapes remain **blind-and-passes** in the import
predicate. Everything else in my sweep either resolves or refuses.

Both judgements the request asked for are given: the two separations in item 3 are right
(with evidence, not taste), and **the audit does not launder defence-in-depth into
coverage** — I tested that directly rather than reading the claim.

---

## 1. DE-R1 — the bridge compares the identity it copies

`daw` **is** the supplier module itself (`ev_replay_seam.daw is de_admissible_windows` →
True), so `_sha` is imported, not restated — the docstring's reason is the right one: a
second implementation "would agree by construction with nothing".

| tamper | result |
|---|---|
| hash zeroed, identity intact | **REFUSED** — names the digest it computed |
| one window flipped **inside** `mask_identity`, hash left | **REFUSED**, same guard |
| hash present, `mask_identity` removed | **REFUSED** — MISSING, named |
| identity present, hash removed | **REFUSED** — MISSING, named |

**The refusal fires before any spec exists:** all `BRIDGE_GUARDS` run over `ctx` before
the emission loop begins, so a refusal returns nothing rather than a partial list.

**The spec carries the copied value** (`supplied["mask_identity_hash"]`), not the
recomputed one — which is correct *because* the guard has just proved them equal, and it
keeps the spec a faithful record of what the supply declared. The distinction only exists
while the guard does, which is why its own known-bad matters.

## 2. DE-R2 — dynamic imports resolve or refuse, and what is still blind

My own sweep of twelve shapes:

| shape | parsed | verdict |
|---|---|---|
| `import X` / `from X import y` | X | **caught** |
| `importlib.import_module('X')` | X | **caught** |
| `__import__('X')` | X | **caught** |
| dotted literal `live.pm_research.X` | `live` **and** X | **caught** (first and last segment — DE's own hole, closed) |
| **`from importlib import import_module`** then `import_module('X')` | X | **caught** |
| **aliased** `import importlib as il` then `il.import_module('X')` | X | **caught** |
| non-literal argument | `<non-literal>` | **REFUSES** (`ImportsUnresolvable`) |
| literal built by concatenation | `<non-literal>` | **REFUSES** |
| **`exec('import X')`** | `[]` | **PASSES — blind** |
| **`eval("__import__('X')")`** | `[]` | **PASSES — blind** |
| **`f = __import__; f('X')`** | `[]` | **PASSES — blind** |

Two of the caught rows — the `from importlib import import_module` form and the aliased
module — are beyond what my DE-R2 asked for. And the refusal text states the rule I
wanted in the module's own words: *"'Cannot tell what this imports' is not 'imports
nothing', and answering True here is what DE-R2 found."*

The supplier's own dynamic import resolves: its parsed set contains
`da_content_liveness_rule` (and `live` from the dotted spelling), and
`reads_no_verdict` is True.

### DE11-R1 — LOW/MEDIUM — three shapes are blind-and-passes

`exec`, `eval` and a rebound `__import__` all yield an empty import set, so
`reads_no_verdict` returns **True** — the category the request defines as a finding
rather than a limit.

Weighed honestly: none of these appears by accident, and the predicate certifies a seat's
own discipline rather than defending against a hostile author — so this is not urgent.
But the fix is cheap and already has a home: treat the **presence** of `exec`, `eval`, or
a bare `__import__` *reference* (not called with a literal) as unresolvable and refuse,
exactly as a non-literal argument already does. That converts three blind-and-passes into
blind-and-refuses, which is the category this module handles correctly.

## 3. DE-R3 — VALUE, MISSING, and undecidable are three different complaints

| input | emission |
|---|---|
| `present_source: /etc/passwd` | **REFUSED [VALUE]** — names the field and the value |
| `present_source` absent | **REFUSED [MISSING]** |
| `scope_days: WHENEVER` | **REFUSED [VALUE]** |
| `revocable_by: nobody` | **REFUSED [VALUE]** |
| `population: SAMPLED` | **REFUSED [VALUE]** |
| `sampling: STRATIFIED` | **REFUSED** — self-contradiction, not the VALUE loop |
| `kind: SOMETHING_ELSE` | **REFUSED** — "does not declare itself an R-ADMISS ratification" |

Both halves of DE-R3 are closed: absence says MISSING, a wrong value says VALUE.

**The two separations are right, and the second is not a matter of taste.**
`FIELD_VOCABULARY['sampling'] == ('NONE','STRATIFIED','CAPPED')` — **`STRATIFIED` is a
legal value**. It is wrong only *in combination* with `population:
FULL_SUPPLIED_COMPLEMENT`, and a vocabulary check cannot express a cross-field
contradiction: folding it in would make the message name one field when the defect is in
the pair. Keeping `kind` out is right for the other reason — a wrong `kind` means the
entry is not the sort of thing being asked about, so evaluating its population fields
would answer a question nobody asked. Answering the narrower complaint first is the
correct order in both cases.

**Confirmed unchanged and NOT re-filed** (out of scope, dispatched as DE round 12):
`scope_to: not-a-date` still yields `verified True, unverifiable []`, and
`now_utc='zzzz'` still reads as closed. DE10-R1 is unchanged at this tip.

One consequence worth stating: `present_source`'s vocabulary is a **one-element tuple**
naming the ledger path. That is the strongest form of DE-R3's closure, and it means a
future ratification naming a different source refuses until the checker is taught — which
is correct (a new source is a new thing to verify), and worth knowing before it happens.

## 4. DE-R4 — the audit at 14 paths, and it does not launder

**The seam's bridge audit uses `skip_guard`, which is the uniqueness test.** All nine
guards report `refuses_when_live: true` **and `refuses_when_disabled: false`** — if
another guard covered the same case, disabling the one under test would still refuse and
that field would read True. `survivors: []`, `all_load_bearing: true`. Three guards
additionally disclose `crash_when_disabled` (`AttributeError`/`KeyError`) rather than
counting a crash as a refusal — the honest treatment.

**The `check()` audit has no `skip_guard` and says so**, driving a bad input against a
control input instead. So I tested the laundering question directly: I disabled **only**
the MISSING refusal and re-ran each missing-field case.

| missing field, MISSING guard disabled | outcome |
|---|---|
| `population`, `sampling` | **still refused** by a second guard — genuinely double-covered |
| `present_source`, `scope_days`, `revocable_by`, `scope_from`, **`scope_to`** | **accepted** — only the MISSING guard owned them |

**The audit's own known-bad for `malformed_block_missing_field` removes `scope_to`** — a
case in the second row. So the guard is driven on a case it **uniquely owns**, and the
double-covered fields are not used as its case. That is the opposite of laundering, and
it is verified rather than asserted.

## 5. Nothing under review moved

`R-419` on 09-01: `verified_for_new_run True`, `provenance False`. `R-418` stamped
10:30Z: `provenance True`, `verified True`, `superseded_by ['R-419']`. The bridge emits
**1,875** specs on the real 09-01 supply. **66 / 53 / 69** checks under both launchers
from the repo root, rc 0.

## 6. Rule 10 / rule 14

Refusal messages compute what they state — the identity refusal prints the digest it
computed, the VALUE refusals name the field and the value found. The only `PASS` prints
are the selftests' own reporters. `decides: "nothing -- this reports; admission is the
coordinator's act and accrual is decided elsewhere"` is in the emission. `reads_no_verdict`
is documented as a predicate over imports that **refuses** rather than answering when it
cannot see, which is the distinction between a predicate and a verdict.

---

## Executed evidence

At `d07d901`, 2026-09-02T11:47–11:51Z:

| check | result |
|---|---|
| suites, both launchers | **66 / 53 / 69**, rc 0 |
| `SEAM.daw is de_admissible_windows` | **True** — `_sha` imported, not restated |
| four identity tampers | **all REFUSE by name**; guards run before emission |
| twelve import shapes | 8 caught, 2 refuse as unresolvable, **3 blind-and-pass** — DE11-R1 |
| supplier's own dynamic import | resolves to `da_content_liveness_rule` |
| VALUE / MISSING / undecidable | all three distinguished by name |
| `sampling` vocabulary | `('NONE','STRATIFIED','CAPPED')` — the separation is required |
| DE10-R1 at this tip | **unchanged** (`scope_to: not-a-date` verifies clean) — not re-filed |
| bridge audit | 9 guards, `refuses_when_disabled: false` for **all**, 3 crashes disclosed |
| MISSING guard disabled, per field | 2 double-covered, **5 uniquely owned** including the audit's own case |
| R-419 / R-418@10:30Z | `for_new_run True` / `provenance True` |
| bridge on the real supply | **1,875** specs |
| register file | never written; worktree clean after |

---

## Disposition

- **RELEASED:** DE round 11. DE-R1, DE-R2, DE-R3 and DE-R4 are closed, two of them past
  the ask (the from-import and aliased-module forms; `present_source` pinned to a
  one-element vocabulary). **No hold.**
- **FILED:** DE11-R1 — `exec`, `eval` and a rebound `__import__` are blind-and-passes;
  make them unresolvable-and-refuse, as the module already does for a non-literal
  argument.
- **Judgements:** item 3's two separations are right, and the sampling one is required by
  the vocabulary rather than merely tidy; item 4's audit does not launder — the seam's
  skip-based test *is* the uniqueness proof, and `check()`'s MISSING guard is driven on a
  case I confirmed no other guard covers.
