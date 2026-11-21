# Entry horizon direction — USER clarification, 2026-11-21

USER: “Yea prediction should be based on longer term holding period”.

Interpret this in the current entry discussion as valuing a posted order over
a meaningful resting lifetime, allowing queue progress and fills, with its
post-fill valuation horizon stated separately. The training label and the
action controller must describe the same continuation policy.

The current paired label already uses a common five-second END. Its initial
POST-versus-WAIT deviation lasts one decision tick before PI0 resumes; PI0
can then cancel the posted generation quickly. Moving the END mark farther
out alone would not change that resting policy. Distinguish three clocks:

- decision/update interval: when new information can change an action;
- resting lifetime: how long the order remains available to execute;
- valuation horizon: when fills and inventory are valued.

The immediate next measurement remains **ENTRY-EXPOSURE-v1**, as specified in
[the accepted review contract](../20261121_37300cc9/report.md): compare the
same initial order under PI0 and KEEP_INITIAL, with WAIT as the reference,
original END and structural safety rules retained. This isolates longer
resting exposure. The freeze now present at
`6ca42d256fe5cea1b91bdf12c36c93d7e931f090` is not reviewed by this note; its
declared experiment is not amended or restarted by this clarification.

The desired model direction is to estimate entry value for that declared
longer-lived action. A later trained candidate must use labels and a
controller with matching retention assumptions. Fast updates and cancellation
when updated risk makes keeping unattractive remain compatible with a longer
valuation horizon. The KEEP_INITIAL diagnostic measures the effect of
exposure; it does not establish a deployed mandatory holding timer.

Judge any benefit on all sampled opportunities, including no fills and
exclusions, using common-horizon value, adverse fills, inventory and the
declared controls. Additional fills alone do not establish improvement.
Unfilled queue position beyond END remains unvalued in this probe. No new
horizon, conditional-model fit or broader data build is selected here.

Development retains implementation ownership. This is a review-owned record
of the user's direction, with no new numerical result, implementation
change, new-date access or expansion of the accepted resource budget.
