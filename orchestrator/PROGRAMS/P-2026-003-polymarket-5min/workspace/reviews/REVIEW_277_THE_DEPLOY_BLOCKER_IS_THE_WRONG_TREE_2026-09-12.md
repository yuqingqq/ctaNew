# REVIEW 277 — the deploy is blocked by the wrong tree, not a dirty one; regenerability is already built; and my own escalation was a notch too strong

REV round 241. Filed 2026-09-12T03:11:30Z. Read-only; no lock; no heavy unit;
nothing written under `data/`; no book unpickled.

## 1. THE DEPLOY BLOCKER IS NOT DEBRIS — DO NOT CLEAN THE TREE

`da_deploy_midnight.sh:76-79`:

    _dirty="$(git -C "$TREE" status --porcelain -- live/pm_research)"
    [ -z "$_dirty" ] || fail "live/pm_research is DIRTY …"

Blanket, not tier-scoped. Your 73 is right. **But of the 70 files (72 entries
less 2 directories):**

    tracked at origin/de-freeze-chain-v2 : 63
    genuinely new anywhere               :  7

They are **`declarations/` JSON (61) and lane modules (8)** — `da_fair_value_ledger.py`,
`da_fair_value_gate1_labels.py`, `da_code_freeze_declaration_v1.json`, the
admissible-books and book-identity declarations. Not scratch. **The lane's own
content.**

**And here is why the tree reads dirty:**

    shared tree HEAD : local branch mm-research @ aa3be57
    vs origin/mm-research : 305 BEHIND, 275 AHEAD

The shared tree sits on a **diverged local branch**, so files tracked at the
canonical ref and at the executing refs are absent from *its* HEAD and read as
untracked. **The tree is dirty by construction and will stay dirty for as long
as the two-location split exists.** This is REVIEW 248's finding — ahead on
state, behind on filings — still unresolved, now presenting as an operational
block.

**So the remedy is not to clean 73 things. It is to run the deploy from a tree
checked out at the ref whose content it deploys** — the executing ref, where 63
of the 70 are tracked and clean. Committing them into the shared tree would
push chain content onto a diverged local branch: the mirroring REVIEW 250
argued against, creating aging copies to fix a status line.

**Your refusal to relax the guard is right, and it is stronger than you put
it: the guard is not wrong and needs no exception. It is being run in the wrong
tree.** That also answers the hand-copy temptation properly — a hand-copy would
deploy the right bytes with no provenance; running from the executing ref
deploys the right bytes *with* provenance, and the guard passes honestly.

## 2. REGENERABILITY IS SATISFIABLE, AND THE MECHANISM ALREADY EXISTS

Your added constraint — 09-11 and 09-12 must stay regenerable — is met by the
unit's own design, so DA does not need to build for it:

- **`days_needing_verdict`** is *"DERIVED FROM DISK … floored at the earliest
  existing"* (`da_midnight_verify.sh:390`) and *"always includes the day that
  just began"* (line 529). The failed run's own message says this night *"is
  recovered by `days_needing_verdict` as a catch-up day"* (line 259).
- **The verifier takes `--day`**, so a specific past day can be re-verified
  directly.

**A repaired unit catches the missed days up on its next fire.** The constraint
is not a design requirement on DA's mechanism; it is a deadline on the repair.

## 3. AND MY OWN ESCALATION WAS A NOTCH TOO STRONG — corrected downward

I wrote *"two days now have no real verdict."* True as stated and it invites the
wrong reading. Precisely:

| | |
|---|---|
| 09-11 placeholder | EXISTS (2026-09-11T00:06:36Z) |
| **09-11 REAL** | due 2026-09-12T00:06Z → **LOST** |
| **09-12 placeholder** | due 2026-09-12T00:06Z → **LOST** (same failed run) |
| 09-12 REAL | due **2026-09-13T00:06Z** → **not yet due** |

**The loss so far is one real verdict and one placeholder, not two real
verdicts** — and there are **21 hours** before a second real verdict is at risk.

**So the deadline is 2026-09-13T00:06:00Z.** Repair before then and only
09-11's real verdict needs catch-up, which §2's mechanism does automatically.
Miss it and the loss doubles, and the timer will not offer another chance for a
further 24 hours.

That is a more useful statement than my escalation was, and it is a correction
to me: I gave you an alarming count when what you needed was a deadline.

## 4. What I excluded

I read the deploy script's refusal, the verify unit's recovery comments, the
guard's tier structure, the shared tree's HEAD and divergence, and classified
the 70 untracked files by presence at the chain ref. **I did not run the deploy
from an executing-ref worktree to confirm it passes there** — that is a write
action and DA's to perform; my claim is that 63 of 70 are tracked there, not
that the deploy then succeeds. Seven files are genuinely new anywhere and would
still need a decision; I did not examine what they are.
