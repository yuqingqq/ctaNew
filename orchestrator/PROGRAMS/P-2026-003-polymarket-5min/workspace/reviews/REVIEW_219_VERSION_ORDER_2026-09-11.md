# REVIEW 219 — `b34ed9f`: version-ordered supersession is right and records the whole chain; the declaration is a commit behind for the fifth time, and that is now a structural fact

**REV, 2026-09-11T13:39Z.** Read-only: no lock, no heavy unit, nothing written under
`data/`, no book unpickled. Drives on scratch and in my own worktree (restored).

`b34ed9f`, 2026-09-11T13:36:33Z, on **both refs**. Two files — `de_multiday_gate1_runner.py`
(+83/−59) and the cells (+70/−59). **Nothing beyond the walker changed** ✓. Comparator
**`a455191d6bceec7e`** ✓. Both refs' `live/pm_research` trees are byte-identical
(`31e3c153…`) at their tips.

## VERDICT

| asked | answer |
|---|---|
| version-ordered supersession | **works** — `code_freeze_declaration` resolves to v4 via amendment **v16** |
| the full pin chain recorded | **yes, all three hops**, each with `named_by` and its sha |
| the within-amendment conflict cell | **fires by name** — the one refusal a version order cannot settle |
| the prior cells | **20/20 pass, `rc = 4`** — main() is still `INPUT_ABSENT` |
| comparator / scope | **both clean** |
| licensing | **NOT LICENSED** — and the valuation **still refuses at import** |

---

## 1. MY OWN RULE WAS WRONG, AND DA 251 IS RIGHT

I asked for agree-or-refuse — REVIEW 211 §3 ("agree-or-refuse, never overwrite") and
REVIEW 212 §2, where I called last-wins a defect and the refusal an improvement. **DA 251
measured that it makes supersession impossible**, and that is correct.

**Why I was wrong, precisely**: rule 13 makes supersession the *norm*, and the **params pin in
the very same resolver already superseded by version order** — `resolve_frozen_params_pin`
takes the highest amendment that pins params, and has since the chain existed. I endorsed a
second, contradictory convention for declaration pins **inside a function that already had
one**. The hazard I was pointing at was real, but it is a *silence* hazard, not an *overwrite*
hazard, and the fix is to record the move rather than forbid it. **`b34ed9f` does exactly
that**, which is why it is a better answer than either of the two rules that preceded it.

*(DE 328's intermediate form — a later amendment may move a pin only by naming what it
replaces — would have worked too, but it is stricter than the ruling and it would have required
DA's v15 to carry a `supersedes` field. I pre-verified both arms on a fixture before `b34ed9f`
landed: with the field the pin moved, without it the chain still refused. Version order removes
that coupling entirely, and it is the right call.)*

## 2. THE SUPERSESSION, DRIVEN

```
resolve_declaration_pins(...)["code_freeze_declaration"]:
  path      da_code_freeze_declaration_v4.json      named_by  de_arm_freeze_v16_amendment.json
  version   16
  chain     [ {version 14, de_arm_freeze_v14_amendment.json, …_v2.json, f8dd3f43…},
              {version 15, de_arm_freeze_v15_amendment.json, …_v3.json, 87b0c6a3…},
              {version 16, de_arm_freeze_v16_amendment.json, …_v4.json, 280e6180…} ]
```

**Every hop is recorded with its amendment and its sha.** That is what my original objection
actually wanted — a reader can see the pin moved, when, and by which amendment — and it is
strictly more than DE 328's single-predecessor record. The provenance requirement is met.

**And the conflict that version order cannot settle still refuses:**

```
[PASS] ONE amendment pinning the same key twice still REFUSES by name
       REFUSED DECLARATION_PIN_CONFLICT: de_arm_freeze_v16_amendment…
[PASS] a CONFLICT propagates out of the reader instead of being swallowed as absence
```

Both arms of the property, and the second is the one that keeps a refusal from being read as
an absence — the class from REVIEW 212 §2.

**20/20 cells pass, `rc = 4`.** The one non-pass is honest and now carries two reasons:

```
[INPUT_ABSENT] the production entry point, end to end
               DECLARATION_DOES_NOT_YET_NAME_THIS_COMMIT,HEAVY_RUN_LOCK_HELD
```

**main() is still not reached by a cell** — for the fifth round running — and the second reason
is BE's rebuild holding the heavy lock, which is correct behaviour to report rather than wait on.

**My independent sweep** (compile-based, 15 modules, known-bad proven): **0 unbound names**.
Converges with DE's again.

## 3. THE FIFTH MISS, IN TIMESTAMPS — AND IT IS NO LONGER A MISTAKE

```
13:35:05Z  d0bb179  DE 328   walker: named-supersession
13:36:33Z  b34ed9f  DE 329   walker: VERSION-ORDER supersession     <- the code is here
13:37:03Z  1d06392  DA 252   v15, code-freeze v4, v16, freeze v12
                             code-freeze v4 names d0bb179e00198800  <- DE 328, not DE 329
```

Driven at the chain tip:

```
import de_forward_value_day -> REFUSED VALUATION_COMPUTING_MODULES_ARE_NOT_AT_THE_PIPELINE_COMMIT:
                               right tree, WRONG BYTES in ['de_multiday_gate1_runner.py']
```

**DA wrote v4 against the tip as it stood when they started; DE's `b34ed9f` landed 30 seconds
before DA's commit.** Nobody made an error. **Five commits have now been declared the freeze
and five have been overtaken** — `8afbd1a`, `d095c5a`, `6d22d78`, `bacb4e3`, and now
`b34ed9f`/`d0bb179`.

**The mechanism is fully diagnosed and it is structural**: the declaration names a *point*, it
can only be written *after* that point exists, and any code commit in the gap invalidates it.
At the current cadence the gap is tens of seconds. No further commit fixes this; the shape of
the pin does.

**The exit, and it is one the programme already owns**: the **build** pin does not name a
point — `_builder_commit_admissible` admits *a descendant of BUILD_PIN whose declared digests
are identical*, and I drove that arm green in REVIEW 214. **The valuation pin should take the
same form.** Then a code commit that does not move a computing module is admissible without a
new declaration, and one that does move a computing module refuses by digest — which is the
property the freeze is actually about. The alternative is to land code and declaration in **one
commit**, which needs one seat to write both.

## 4. LICENSING — **NOT LICENSED**

At 13:39Z: **no artifact anywhere carries `admitted_by`**; two research units are running
(BE's rebuild, holding the heavy lock, due ~14:00Z); and the valuation **refuses at import** at
the tip (§3).

The criterion is unchanged and DA-declared: **`cells[<arm>].book_receipt.admitted_by ==
"DESCENDANT"`** in a receipt from a valuation that ran through the production chain on a book
whose `builder_commit` descends from the build pin.

**What has to be true before I can read it:** the declaration names the commit the code is at
(§3), `wt-deval` is at that commit, and the end-to-end runs. Nothing in the code stands in the
way any more — the walker is right, the cells are right, the sweep is clean, the comparator is
intact. **What stands in the way is a declaration that cannot catch a moving tip**, and one
ruling on the pin's shape ends it.

## SCOPE

Closed over: `b34ed9f` read and diffed; both refs' tips compared by tree; the walker driven at
the chain tip; the resolved pin's full chain read; all 20 cells run; my own sweep over 15
modules; the import driven at the tip; DA 252's v4 read for the commit it names; the timestamps
of the last three commits. **Not closed over:** BE's rebuild, which is mid-flight; DA's freeze
v12 and v16's other contents, which I read only as filenames and for the pin; and (6), which
cannot start while the lock is held.

## ROUTED

1. **Coordinator — rule on the pin's SHAPE, not on the next commit** (§3). Five freezes, five
   overtaken, ~30 s apart. The build pin's descendant-plus-digests form already exists in this
   codebase and is celled.
2. **DE/DA — or land code and declaration as one commit.** Either ends the loop; nothing else
   will.
3. **Me — the agree-or-refuse rule was mine and it was wrong** (§1). Recorded here so the next
   reader of REVIEW 211/212 finds the correction beside the recommendation.
