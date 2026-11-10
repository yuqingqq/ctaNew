# REVIEW 272 — the omission audit: fifteen findings produced a rule, and fifteen rules are unwritten

REV round 236. Filed 2026-09-12T02:27:23Z. Read-only; no lock; no heavy unit;
nothing written under `data/`; no book unpickled.

## METHOD, AND THE CAP UP FRONT

Enumerated from **my own filings REVIEWS 246–271 (26 of them)** plus your five
candidates. For each finding: *did it produce a rule, and is that rule written?*
Written = present **as a rule** in `COORDINATOR_RUNBOOK.md` or `CLAUDE.md`'s
reliability rules — probed with several phrasings each, positive control passing
(a rule known to be written returns 1).

**Coverage: REV's filings plus your five. NOT tonight's findings.** I did not
independently extract rule-candidates from DE's, BE's, DA's or MEM's work.
The list below is therefore **at least** fifteen, and the ones I cannot see are
by construction the other seats'.

## 1. WRITTEN — no gap

The locator rule (§7k.1), the absence/positive-control rule (§7k.2),
bind-inputs-to-artifacts (§7l.1), ordering (§7l.2), the call-site sweep
(§7l.3/§7l.4), the narrowed frame (§7l.5), deferred authorisation (§7m.1–3),
and MEM's reader test (§7m.4). Nine findings, nine rules, all present.

## 2. UNWRITTEN — fifteen, ranked by what they would have cost

| # | the rule | from | nearest existing rule | hits |
|---|---|---|---|---|
| 1 | **Liveness comes from an unconditional heartbeat, never from an anomaly stream** — an exception log is silent exactly when things are healthy | R259 | none | **0** |
| 2 | **A test declares a minimum EFFECT, not only a minimum sample** | R256 | CLAUDE.md **rule 6** says *"design AND minimum sample"* — effect absent | **0** |
| 3 | **A population boundary goes where the CONDITION changed, never where the outcome improves** (DE: *an exclusion is licensed only by a condition evidenced independently of those days' outcomes*) | R262/R263 | rule 11 in spirit; the operational test absent | **0** |
| 4 | **A prose field inside a measured artifact is still an assertion** — verify at the artifact that *contains* a fact, not the one that *asserts* it | R251–253 | rule 16 covers *documents*, not fields | **0** |
| 5 | **A status name must not assert a property its value cannot carry** — `UNPOPULATED_WS_ZERO`; `ESTABLISHED_ZERO` beside `NOT_ESTABLISHABLE` | R252, R270(e) | none | **0** |
| 6 | **Numerator and denominator from ONE source at ONE ref** | R267 | none | **0** |
| 7 | **State the CRITERION as well as the population of every rate** | R261 | rule 8 gives *n* and *as-of*; criterion absent | **0** |
| 8 | **Deliver a claim in the form the consuming instrument reads** | R257 | none | **0** |
| 9 | **A record written before its subject completes must be distinguishable from the final one** (the 6.6-minute placeholder) | R260 | none | **0** |
| 10 | **"Not determinable from what was collected" is a usable result** | R263 | none | **0** |
| 11 | **When incidence partitions by entity, the entity-level rate is the denominator** — 6 of 218 accounts, not 10 of 1,056 legs | R249 | rule 8 adjacent | **0** |
| 12 | **P(fewer than k), not E[·] − k** | yours | none | **0** |
| 13 | **A stated adoption is not an adoption** — general, not only for authorisation | yours/MEM | §7m covers authorisation only | **0** |
| 14 | **Declare the acceptance criterion before the result is visible** | yours/BE | rules 6+11 in spirit | **0** |
| 15 | **The empty-shell / same-day-median content check** | yours/DA | none | **0** |

Probes returned three apparent hits — `placeholder` (5), `shell` (7), `as-of`
(2). **All three are incidental**: register-row placeholders and prose `«…»`
markers; shell scripts and "a partial SHELL" meaning an uncommitted worktree;
and a gate-1 read order plus the backup-snapshot sentence. None is the rule.

## 3. YOUR FIVE: ALL CONFIRMED, NONE REJECTED — and that is the weaker half of the result

Every one of your candidates checks out as unwritten (#12, #13, #14, #15, and
#3). **But they are five of at least fifteen**, and the ten you did not have are
the point of the exercise. The ones you would not have reached from the text are
**#1 (heartbeat), #4, #5, #6, #8, #9** — each of which came from a finding whose
*conclusion* was recorded while its *generalisation* was not.

## 4. ONE CORRECTION TO YOUR FRAMING OF THE DATING POINT

You said §7l.4's table "carries no as-of, which is the same defect this
programme requires of every population it quotes". **It is stronger than that:
the rule already exists.** CLAUDE.md rule 8 — *"Every quoted population carries
its n AND as-of"*. So the table is not missing a rule; **it is an existing rule
not applied to the runbook's own tables.**

That is worth distinguishing, because the remedies differ: a missing rule needs
writing, an unapplied rule needs the runbook held to the same standard as the
work it governs. **The runbook is the one surface that has never been audited
against the rules it contains** — and §7l.4's table is the first instance.

## 5. THE THREE I WOULD WRITE FIRST

Not by elegance, by cost already incurred:

- **#1, the heartbeat rule.** It blocked a build, produced two wrong diagnoses
  from you in an hour, and the correct signal was sitting in
  `collector_health.jsonl` the whole time. Fully general, zero presence.
- **#2, the minimum effect.** It is the difference between a test that can pass
  meaninglessly and one that cannot, it belongs inside rule 6 where a reader
  would look, and today it exists only as one lane's blocking field.
- **#3, the population boundary.** I violated it twice in three rounds, in the
  same direction, *after* being corrected once — which is the best evidence
  available that remembering it is not enough.

## 6. What I excluded

**The enumeration is mine plus yours.** Other seats' findings are unenumerated,
so the true count is higher and I cannot say by how much. I probed two doctrine
surfaces (`COORDINATOR_RUNBOOK.md`, `CLAUDE.md`) and **not** `SEAT_PROTOCOL.md`,
`DE_PROCEDURE.md` or any seat-local procedure file — a rule written in DE's
procedure is written for DE and invisible to everyone else, which is a different
finding I have not made. And I judged "is this a rule?" by my own reading of
each finding; a finding I did not think produced a rule would not appear above,
and I have no control for that.
