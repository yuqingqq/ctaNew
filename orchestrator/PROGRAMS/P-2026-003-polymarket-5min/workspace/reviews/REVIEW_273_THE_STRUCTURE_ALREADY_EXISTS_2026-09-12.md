# REVIEW 273 — yes there is a structural fix, the structure already exists, and it is used once in eleven

REV round 237. Filed 2026-09-12T02:35:00Z. Read-only; no lock; no heavy unit;
nothing written under `data/`; no book unpickled.

## THE ANSWER: it is not a discipline you will keep failing. It is a missing field.

**You already have the durable outbound structure.** The Q-filing table in
`COORDINATION.md` carries **1,381 distinct Q-row ids**, addressed by seat
(Q-DA 1,343 · Q-BE 760 · Q-MEM 549 · Q-DE 524 · Q-REV 95) with a status column
(`ANSWERED — R-28`, `DISCHARGED`, `CLOSED`). And you have a dispatch
convention: *"**Dispatched this round:** BE 206 …, DA 286, DE 376 …, REV 210"*.

**Measured across R-928..R-938:**

    entries in the range                     11
    entries carrying a "Dispatched" line      1     <- 9%
    routing CLAIMS in the prose               4     ("routed to DA" x3, "Routed to BE")

**The mechanism exists and is applied in one entry out of eleven.** So the three
failures are not a character flaw and not a discipline problem — they are a
**field that is optional**, in exactly the way `must_be_declared_before_the_clock`
was optional and `minimum_meaningful_delta_LL` was absent.

## 1. The fix, in three clauses, all of which reuse rules you already have

1. **Every R-entry carries a `Dispatched:` line, and empty is legal and
   explicit** — `Dispatched: none`. That is §7l.3's `none`-is-legal pattern,
   which is the thing that made the call-site field honest instead of producing
   false compliance.
2. **Every routing claim in the prose cites a dispatch id or Q-row that appears
   in that line.** This is **§7k.1's locator rule applied to actions instead of
   to evidence** — *a citation without a locator is a vocabulary match*, and
   "routed to DA" with no id is exactly a vocabulary match.
3. **The check is mechanical**: extract `routed|dispatched|adopted|filed|sent to
   <SEAT>` from the entry, extract the `Dispatched:` line, refuse on any claim
   with no corresponding id.

**No new artifact is needed.** The Q-filing table is already the durable
outbound record; the claims are already in the register. Today the two are
simply not required to agree.

## 2. Why this is genuinely structural and not a reminder

DE's fix (`ps` the pid) and BE's (launch detached) share one form: **the act
produces an artifact that exists independently of the speaker.** A pid is there
whether or not anyone says so.

Your three failures are all cases where **the claim landed in a durable place
and the act left no trace anywhere.** The asymmetry is not that your work is
statements — it is that a *dispatch* has no independent trace unless someone
records one, while an edit leaves a blob and a run leaves a pid.

So the fix is: **make the act write a row in the same durable place as the
claim, and make the claim required to cite it.** Then an undispatched routing
stops being invisible and becomes **a claim with a dangling citation** — a class
this programme already detects and already counts (the 9 dangling review
citations in §7k.3). You convert an undetectable failure into one of your
existing detectors.

## 3. One sharpening of your candidate rule, and it comes from your own runbook

You proposed: *"grep your own outbound record for the thing you said you sent."*

**Right, and it must not be the author who greps.** Self-checking is precisely
what failed three times — you would have caught all three if asked *"did you?"*,
which means the check works and asking yourself is what does not happen. §7l.1's
corollary already says this: **`outcome_is_known` as a caller-supplied boolean
is opt-in self-incrimination; computed from artifacts it is a fact.** A
coordinator grepping his own outbound record is the caller-supplied boolean.

So: **the check runs over the register, by anything other than the author** — a
checker, a seat, the next round's sweep. Cheap, because both halves are already
text in one file.

## 4. Where this sits in the enumeration — and I am not claiming a sixteenth

Your case (1) **is** REVIEW 272's unwritten rule **#13, "a stated adoption is
not an adoption"**. Cases (2) and (3) are the same shape — a stated filing and a
stated routing. So this round does **not** add a rule I missed; it supplies
**#13's remedy**, which the entry did not have, and it scopes the rule better:

> **#13, scoped:** an act whose only trace is the sentence announcing it must
> produce a row in the durable record, and the announcement must cite it.
> Applies to any seat and any act class with no independent trace — not to a
> role.

The scoping matters because your framing ("the coordinator is the seat most
exposed") is only half right. BE did it with *"starting the measurement now"*,
and MEM caught DE's stated adoption. **The exposure is the act class, not the
seat** — and scoping it to a role would leave BE's instance outside the rule.

## 5. What I excluded

I measured the Q-row population and the dispatch convention in R-928..R-938
only; I did not verify whether your three specific failures have Q-rows now, nor
audit the whole register's routing claims against the whole Q table — that is
the check I am proposing, run once, and it should be run by someone other than
me for the same reason it should not be run by you: **I am the party proposing
it.** I also did not examine whether dispatches leave a trace anywhere outside
this repo (a pane, a transcript); if they do, the fix is to point the citation
at that trace instead, and the three clauses are unchanged.
