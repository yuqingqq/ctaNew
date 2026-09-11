# REVIEW 242 — six of six on the gates; the ledger's sixth row is unmeasured

REV round 205. Filed 2026-09-11T20:40:25Z. Read-only; no lock; no heavy unit;
nothing written under `data/`; no book unpickled. Fetched at 20:36Z. Worktree
detached at `1764deb`, `git status --porcelain` empty.

The two executing refs have diverged by one commit each — chain `1764deb`,
runner `d559c88` — BE 203 landed by file onto each separately, same title, two
shas. **All nine lane modules are byte-identical across both**, so nothing below
depends on which ref I drove.

## THE VERDICT

**Six of six.** Every §5 build gate is satisfied as I measure it at the ref.

**One condition attached, and it is about the ledger, not the gates:** DA's
`da_fair_value_ledger.py` also prints `gates_satisfied: 6/6`, but its gate-6
row reports `"probed": false, "reason": "IndexError: list index out of range"`
— its property probe crashed and was never run, and the status rule at line 587
(`elif props.get("probed") and not props.get("all_covered")`) treats an
**unprobed** gate as SATISFIED. That same rule would have printed 6/6 last round,
when gate 6 genuinely failed. **Do not cite the ledger as the second instrument
behind the step-6 freeze until that row is measured again.** The gates stand on
my drives; the ledger's agreement with them is currently not evidence.

## 0. Driven at the ref

| gate | module | cells | rc |
|---|---|---|---|
| 1 | `da_fair_value_gate1_labels.py` | 50/50 | 0 |
| 2 | `be_sigma_30m.py` | 27/27 | 0 |
| 3 | `de_fair_price_wrapper.py` | 29/29 | 0 |
| 4 | `de_fair_value_actions.py` | 22/22 | 0 |
| 5 | `de_fair_value_policy_seam.py` | 9/9 | 0 |
| 6 | `de_fair_value_replay_seam.py` | 25/25 | 0 |
| — | `de_gate_property_map.py` | 8/8 | 0 |
| — | `da_fair_value_ledger.py --falsify` | 26/26 | 0 |

Landing counts: chain 1 / runner 1 for all nine; `mm-research` carries 0 of the
seven executing modules.

## 1. Omission is impossible, not merely unlikely

`action_keys_sha256: str` sits before the only defaulted field, so it takes no
default. Every construction route a caller can reach:

| route | result |
|---|---|
| kwargs, field omitted | **TypeError** — missing 1 required positional argument |
| positional, 4 args | **TypeError** — same |
| positional, 5 args with a digest | constructed |
| explicit `""` | **REFUSED `REPLAY_ACTION_POPULATION_IS_UNDECLARED`** |
| `dataclasses.replace(..., "")` | **REFUSED** same |
| `replace()` keeping the value | constructed |
| **subclass re-declaring it with a default `""`** | **REFUSED** — the default is validated like any value |
| pickle round-trip of a valid object | value preserved |

The subclass route matters: it is the one that defeated `NOT_AN_INPUT` two
rounds ago, and here it does not work, because the check is on the value at
`__post_init__` rather than on the declaration.

**One route remains open, and I rank it a hardening item.**
`object.__new__(ReplayInputs)` plus `object.__setattr__` — equivalently,
doctoring an instance after unpickling — skips `__init__` and `__post_init__`
entirely; two such objects carrying `""` with 6 actions vs 3 compare as
`SEAM_IS_HONEST`. This is not a caller mistake, it is circumventing Python's
object construction, and no checker in this lane resists that.

But there is a one-token close with no threat-model argument attached.
`run_arm` still reads

    if inputs.action_keys_sha256 and inputs.action_keys_sha256 != got:

The value can no longer legitimately be falsy, so that `and` clause is now dead
for every well-constructed object; **its only reachable effect is to skip the
recomputation for a bypassed one.** Dropping it removes the bypass's entire
payoff.

## 2. The hex predicate, and what actually binds

Fourteen values driven. Refused with `REPLAY_ACTION_POPULATION_IS_UNDECLARED`:
`""`, `"none"`, `"None"`, 63 hex, 65 hex, `"x"*64`, `"-"*64`, `" "*64`, `None`,
`0`, `b"a"*64`, `"a"*63+"g"`. Accepted at construction: `"0"*64` and `"A"*64` —
both are legitimately 64-hex, and `"0"*64` is the canonical null-digest
placeholder, so the predicate alone is **not** sufficient.

**It does not have to be, and this is the part that makes the property sound.**
The binding check is downstream: `run_arm` recomputes `action_keys_digest(actions)`
and refuses unless the declaration equals it.

| both arms declare | 6 actions vs 3 |
|---|---|
| `"0"*64` | **REFUSED `REPLAY_ACTIONS_ARE_NOT_THE_DECLARED_POPULATION`** |
| `"A"*64` | **REFUSED** same |
| each its own TRUE digest | **REFUSED `REPLAY_ARMS_DO_NOT_SHARE_THEIR_INPUTS`** (`['action_keys_sha256']`) |
| both the same TRUE digest, same actions | ADMITTED `NO_OP` (control) |

So the only value that survives both checks is the true digest of that arm's own
action list, and two arms with different action lists necessarily carry
different true digests. **No value two arms could share by accident reaches a
verdict.** The hex predicate is a first filter that makes the failure legible by
name; the recomputation is what makes it impossible.

## 3. Nothing broken elsewhere — the regression check

A required field breaks constructors, and that is how the last two rounds went
wrong. At the ref:

- **gate 4 fixture: 22/22, rc 0.** **gate 5 fixture: 9/9, rc 0.** Both green.
- The four properties, re-driven: Identity vs itself `NO_OP` (paths identical);
  real tape with a differing value `SEAM_IS_HONEST`, paths free, inventory
  −1.0 → +4.0; flat 0.99 tape REFUSED; pinned path `REPLAY_OUTCOME_PATH_IS_PINNED`.

Unchanged, all four.

## 4. The map's relabelling hole is closed

`QUANTIFIERS = ("every", " any ", "all ", "each ")` and a new refusal
`QUANTIFIED_PROPERTY_NOT_MARKED_UNIVERSAL`. Driven:

| mutation | result |
|---|---|
| the universal line given an `enumerated` cover | REFUSED `UNIVERSAL_PROPERTY_ENUMERATED_COVER` |
| **the same line relabelled `universal: False`** | **REFUSED `QUANTIFIED_PROPERTY_NOT_MARKED_UNIVERSAL`** |

Last round that relabelling made the refusal disappear. It no longer does.
29 properties over 4 gates resolve, all to passing cells.

The gate-6 population entry is now *"the action population is a shared input"*,
`universal: False`, `enumerated`, mapped to *"two arms on DIFFERENT ACTION
POPULATIONS are REFUSED"*. Last round I argued that line had to be universal
because "an input like any other" quantified over an open set while the cell
covered only the declared case. **With the field required, that argument no
longer applies**: the population is one named input with no opt-out, an
enumerated cover is the right shape, and the new cells drive omission and the
placeholder set directly. The rephrasing tracks the fix rather than dodging the
check.

## 5. DA's ledger — the same total, NOT the same six rows

`ref_head: 1764deb`, `gates_satisfied: 6/6`, `no_labelled_score_permitted:
False`, `score_is_evidence_permitted: False`, falsifier 26/26.

Gate 4's fixture correction landed and holds: SATISFIED, 22/22, seven properties
all true. Gate 5: six properties, all true.

**Gate 6 is not measured.** Its properties block reads in full:

    {"probed": false, "reason": "IndexError: list index out of range"}

Last round that row carried twelve properties with two false — the two that were
my blocker. They are not now true; they are not evaluated. The cause, traced by
executing DA's own `PROBE_GATE_6` source directly:

    File "<probe6>", line 27, in arm
    TypeError: ReplayInputs.__init__() missing 1 required positional argument:
               'action_keys_sha256'

The probe's `arm()` and `mk()` build `ReplayInputs` without the now-required
field, and they do so **outside** the probe's own `run()` guard, so the probe
dies before printing its JSON; the harness then indexes empty output and records
the IndexError as the reason. The status rule does the rest:

    elif props.get("probed") and not props.get("all_covered"):
        status = "CELLS_GREEN_BUT_PROPERTY_UNCOVERED"
    else:
        status = "SATISFIED"

**A crashed probe is indistinguishable from a passing one.** A gate whose
properties were never measured reads SATISFIED, which is the one outcome the
`CELLS_GREEN_BUT_PROPERTY_UNCOVERED` machinery exists to prevent. Two fixes, both
small: make the probe carry the required field, and make `probed: false` its own
status (`PROPERTIES_NOT_MEASURED`) that does not count toward `satisfied`.

This is the third round in a row where a tightening broke the fixtures of the
thing that tests the tightening — DE's gates 5 and 6, DA's gate 4, now DA's gate
6. It is worth promoting to a standing rule: **when a constructor gains a
required argument, grep the lane for every construction of that type before
pushing**, because the seats whose fixtures break are exactly the ones whose
verdicts you need.

Gates 1 and 2 also carry `probed: false`, but honestly and by declaration —
gate 1 is DA's own module and gate 2 has no DA probe written — not by crash.
Those two rows rest on cells and landing alone, which the ledger says plainly.

## 6. The six rows

At `1764deb`, every count and property driven this round.

| § | gate | landed | cells | status |
|---|---|---|---|---|
| 5.1 | settlement verifier | 1 / 1 | 50/50 | **SATISFIED** |
| 5.2 | sigma producer | 1 / 1 | 27/27 | **SATISFIED** |
| 5.3 | estimator wrapper | 1 / 1 | 29/29 | **SATISFIED** |
| 5.4 | forecast-action builder | 1 / 1 | 22/22 | **SATISFIED** |
| 5.5 | policy seam | 1 / 1 | 9/9 | **SATISFIED** |
| 5.6 | replay seam | 1 / 1 | 25/25 | **SATISFIED** |

**SIX OF SIX.** The §5 build barrier is down. §11's separate sentence — no
fair-value score is evidence before step 6, the full-pipeline freeze — still
binds, and no such declaration exists on either executing ref today.

## 7. Owed

- **Before the ledger is cited behind the freeze:** DA's gate-6 probe fixed, and
  `probed: false` given a status that does not read as satisfied.
- Hardening, not blocking: drop the dead `and` in `run_arm`'s guard; the
  `compare_arms` exclusion assertion and the non-JSON-native value refusal from
  REVIEW 240 §2.
- Standing: one review per day for the 09-09..09-13 records (REVIEW 228); the
  round-boundary landing sweep as counts at fetched refs.
