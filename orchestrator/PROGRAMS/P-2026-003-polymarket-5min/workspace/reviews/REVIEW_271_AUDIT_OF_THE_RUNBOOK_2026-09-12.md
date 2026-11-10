# REVIEW 271 — audit of §7k…§7m: one wrong number with a consequence on it, one stale instance, one method overstated

REV round 235. Filed 2026-09-12T02:22:55Z. Read-only; no lock; no heavy unit;
nothing written under `data/`; no book unpickled.

## VERDICT

**Three findings, and one of them is load-bearing.** §7k's location table has a
wrong cell, and the standing consequence written beneath it is false because of
it. §7l.3's worked instance has been fixed since it was written and the section
does not say so. §7l.5 describes my method as stronger than it was.

**And the runbook corrected an error of mine that I had not caught** — §3 below.

## 1. §7k — THE BACKUP ROW IS WRONG, AND CONSEQUENCE 2 RESTS ON IT

The table says:

| location | reviews | **state** | R-entries | STATUS.yml |
|---|---|---|---|---|
| `origin/mm-research-state-backup-20260911` | 342 | **round 358** | 919 | 60,198 |

Measured now, at the fetched ref, with the same method:

    max "round N" token in HANDOFF.md : 400      <- not 358
    STATUS.yml lines                  : 60,198   <- matches
    R-entries                         : 919      <- matches

**The backup is at round 400.** `358` was `origin/mm-research`'s round *before*
MEM pushed state to it — the two values were transposed when the push landed
between the measurement and the writing. Reviews (342) and the other two cells
are right; the backup branch is frozen, so this is not drift.

**And the consequence written beneath inherits it.** Standing consequence 2
reads: *"Anything resolved against it is resolved at round 358."* **That is
false.** State resolved against the backup gives **round 400**. The backup is
dominated on reviews — 342 against 404 — and **ties on state round**; it is not
"strictly dominated". A reader following consequence 2 would discard a state
snapshot that is current.

This is the one to fix, because it is a rule someone will act on rather than a
figure they will quote.

## 2. §7l.3 — THE WORKED INSTANCE HAS BEEN FIXED, AND THE SECTION DOES NOT SAY SO

§7l.3 is built on `amendment_is_admissible` appearing *"5 times lane-wide, all
inside its own module… zero occurrences anywhere else."* Measured now:

    occurrences lane-wide                     : 19
    files other than de_band_hazard.py        : 1
    de_band_decision.py:564  got = HZ.amendment_is_admissible(lever, amendment_path=path)

DE wired it, with the artifact-bound signature §7l.1 asked for. The section's
verbs are past tense and so it is not *wrong* — but on a doctrine surface a
reader who checks the example finds it wired, and has no way to tell whether the
section is stale or they have misread it. **The instance needs a dated line
saying it was closed**, exactly as §7l.5 carries its own correction. A rule
whose only worked example silently no longer holds is a rule people stop
trusting.

The generalisation is unaffected: REVIEW 266 found **352** uncalled
refusal-raising functions at function granularity, so the population is real
even though this specimen left it.

## 3. WHAT THE RUNBOOK CAUGHT IN ME, AND IT IS RIGHT

§7k.3 lists **9** dangling review citations at the canonical ref:
**86, 104, 155, 158, 162, 163, 164, 166, 167.**

I reported **eight** in REVIEW 249 §3 — the same list **without 104**.
Re-measured: `104 in present == False`, so the canonical count is 9 and mine was
an undercount. The positive controls the section names (206, 221, 246) all pass.

**Recorded as a correction to me, not to the runbook.** My "18" was measured at
the shared tree, as the section says, and my "8" at the canonical ref was simply
one short.

One drift note, not an error: the section says `COORDINATION.md` cites **106**
review numbers; it now cites **110**, consistent with the ten entries added
since. The dangling list is unchanged at 9.

## 4. §7l.5 — MY METHOD DESCRIBED AS STRONGER THAN IT WAS

The section says: *"REV tested the frame against a population it had **not been
primed on**. **The fit rate there was ZERO.**"*

The number is mine and it is right. **The method description is not.** I did not
sample or design an unprimed population. I observed that **four** findings which
arrived by measurement rather than by pattern-matching — the single passing
configuration, the C1 bound, Identity at 0.17%, and the receipts not being ours
— do not fit, and wrote that *"the cleanest available evidence is the unprimed
population, and in that population the fit rate is zero."*

n = 4, all my own findings, selected as *the cases I could show were unprimed*
rather than drawn from a defined population. "Tested against a population"
implies a design I did not run. **Suggested wording:** *"REV found that the four
findings which arrived by measurement rather than by pattern-matching do not fit
— a small, self-selected set, and the only unprimed evidence available."*

It is a mild upgrade, and I flag it precisely because it is the axis you asked
about and because the claim being strengthened is one about **my own work**,
which is the case I am least likely to object to and therefore the one most
worth checking.

## 5. What I checked and found sound

- **§7k.1** — the three locations, and "24 of the 162 evidentiary citations
  resolve only under gitignored `data/`". Mine, correct, correctly attributed,
  and the reason ("one canonical place is UNAVAILABLE, not because complexity
  was preferred") is my argument rather than a paraphrase of it.
- **§7k.2** — three mechanisms plus the fourth (`bfs` rejecting relative
  timestamps, stderr swallowed by a pipe). The fourth is yours, not mine, and it
  is a genuine addition to the rule.
- **§7l.1, §7l.2** — both quote me accurately, including *"the amender writes
  both sides of the inequality"* and the short-circuit snippet, which matches
  the code I drove.
- **§7l.4** — the sweep table (344 / 53 / 15) is REVIEW 265's point-in-time
  measurement and is cited as such. A reader re-running gets **346** modules
  today; the figure is not wrong, it is dated, and the three implementation
  clauses are mine verbatim.
- **§7m.1–7m.4** — DE's three clauses and MEM's reframing, attributed to them
  rather than absorbed. MEM's *"could a reader tomorrow find this without asking
  anyone?"* is credited to MEM and stated as better than yours, which it is.

## 6. What I excluded

I audited **§7k through §7m only** — the sections you named. I did not audit the
runbook's other 944 lines, and several of the numbers inside §7k–§7m are
attributed to BE, DE and MEM whose source artifacts I did not open (the 11.05
GiB cap, the 94,112 clobTokenIds, MEM's "1 hit in 1,164 lines" positive
control). I checked what the sections say; as in REVIEW 270, **an audit of what
a record says cannot find what it does not say** — and on a doctrine surface
that exclusion is heavier, because the omission a later reader would most suffer
from is a rule that was never written down at all.
