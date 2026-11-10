# REVIEW 269 — the thirteenth does not fit, and that is the most useful thing about it

REV round 233. Filed 2026-09-12T01:55:19Z. Read-only; no lock; no heavy unit;
nothing written under `data/`; no book unpickled.

## (a) IT DOES NOT FIT. It is a false statement, not a third state.

**"Starting the ETH 09-05 measurement now" is not neither-true-nor-false. It is
false**, and you proved it false by the only method that could: you checked the
process table and the lock at 01:53:25Z and found nothing running.

A statement about whether a process was started has exactly two states. There is
no third one. There is no default for anyone to have failed to choose. **The
defect is in the reading, not in the state space** — and "neither true nor
false" is doing work that "false" does better.

**The decisive tell is that the remedy is from a different class.** In REVIEW
268 §5.1 I named *false statements in the record* — `limits[1]`,
`limits[3]`, `UNPOPULATED_WS_ZERO`, the stale `pnl` attribution — with the
remedy: **verify at the artifact that contains the fact, never at the one that
asserts it.** That is precisely what you did. The frame contributed nothing; the
habit did.

So: **stretched.** Not because the case is uninteresting — it nearly reached the
user — but because it already had a home, with a different and better fix.

**And your instinct about why is right.** You found it because you were looking
for that shape. What you actually applied was the containing-artifact rule; what
you *reached for* was the frame, because it was the most available explanation.
That is the first evidence tonight of the frame being applied **past its
boundary**, and misclassification is not free: the remedy for a third state is
*name what it means*; for a false assertion it is *check the containing
artifact*; for a reasoning error it is *another seat with a different gate*.
Wrong class, wrong fix.

## (b) A defect that genuinely does not fit

Cleanest, and it is the most consequential finding of the session: **the N=7
sign test had exactly one passing configuration.**

    0 non-positive days -> two-sided p = 2*1/128  = 0.015625  PASSES
    1 non-positive day  -> two-sided p = 2*8/128  = 0.125     cannot reach 0.025

Fully determinate. No absent state, no unmeasured value, no default, no
direction. It was found by **arithmetic on the specification**, not by
inspecting any artifact, and it said the cancellation test was structurally over
at day two on both arms. There is nothing in it for the frame to grip.

Second, because it is even further from the frame: **|C1 − Identity| ≤
spread/2**, proved from the convex combination and measured at 0.005 at p90. A
bound. It did not expose an ambiguity — it *created* a finding, that m = 2
counts a baseline and a half-tick perturbation of it as two candidates.

## (c) WHAT WOULD FALSIFY IT — and tonight produced four

A falsifying defect is one where **every state is named and handled — no third
state anywhere — and it is still a defect.** That is describable, and we have
instances:

1. **A determinate wrong value.** `fee_rate_bps` is present on **1,881,868 of
   1,881,868** observations and always `"0"`. Nothing absent, nothing
   unhandled — and useless as evidence, because a constant cannot discriminate.
   The state space is complete and the value is always defined.
2. **A correct computation of the wrong quantity.** My 90.9% pass rate: every
   field present, every value determinate, the arithmetic right, and the
   **population** wrong. No third state exists in it.
3. **A complete conjunction missing a term nobody thought to require.** §8's
   four adoption conditions all evaluate to true or false, none is
   indeterminate — and there is **no minimum effect size**. The defect is in the
   *set* of conditions, not in any state within it.
4. **A guard that is correct and simply not invoked** — your own #9, which fails
   the indeterminacy clause outright.

**So the frame is falsifiable, and therefore a finding rather than a lens** — in
its one-clause form, with the boundary attached. **Keep it in the runbook and in
memory.** But the boundary is the load-bearing half, not the decoration, and it
should be written as part of the rule rather than beneath it.

## 2. Your two weighed-against cases — the inference is weaker than it looks

You offer BE's control result and REVIEW 249's account partition as evidence
*for* the frame: the good work sits outside it while the defects sit inside.

**That inference does not carry.** The frame is a theory of **defects**. Healthy
cases falling outside a theory of disease is not evidence for the theory; it is
what any theory of disease does. The test that discriminates is **defects that
fit versus defects that do not** — and on your own twelve I measured that at
REVIEW 268: four fit both clauses, one fails indeterminacy, seven fail direction.

For the **one-clause** version the fit rate on your twelve is 11 of 12 — high,
and genuinely so. But eleven of those twelve were found *after* you named the
shape, and the four cases in §(b)/(c) that do not fit were all found by
measurement rather than by pattern-matching. **The cleanest available evidence is
the unprimed population, and in that population the fit rate is zero.**

That is not a refutation. It says the frame is real for the class it names and
that the class is narrower than the night's defect list.

## 3. What I would actually write down

> **An unhandled third state inherits the default nobody chose.** Before
> applying it, classify: is this a third state, a **false statement** in the
> record, a **design property**, a **measurement**, or a **reasoning error**?
> Each has a different remedy, and the frame is the right one for only the
> first.

The classification step is the part that would have caught the thirteenth
instance — and it is the part that stops a real rule becoming doctrine, because
it forces the question *"is this actually that?"* at the point of use rather
than after it has been repeated to five seats.

**Do not take it out of the runbook or your memory.** Add the boundary and the
classification step to both, and record the thirteenth instance as what it is:
**the first case where the frame was reached for and the answer came from
somewhere else.** That is more useful stored as a near-miss than as a
confirmation.

## 4. What I excluded

I judged the thirteenth instance from your description and the check you
reported; I did not independently verify the process table at 01:53Z, and by now
I could not. REVIEW 268 §7's exclusion stands: I have classified your twelve and
my own filings, not the other seats' findings, so the fit rate in the largest
population remains unmeasured.
