# Can receipt-matched book frames explain the November 8 false exits?

November 26, 2026 UTC. Across the 13 markets and 15 orders from the fixed-order trace, 20 fill allocations accounted for 119.486 disputed closing shares. All had zero wallet fills. The control cohort comprised all 99 confirmed wallet maker legs (849.867 shares) from both archived runs. Each row's status is retained.

| Quantity | Disputed closing shares | Wallet-confirmed shares |
|---|---:|---:|
| With a uniquely associated pending increment and matching Tokyo book frame | 19.00 | 16.55 |
| Without that evidence | 100.49 | 833.32 |
| Paired shares whose Tokyo frame arrived before or at the cancel request | 0 | 0 |
| Paired shares excluded only under a maximum 50 ms match-to-receipt delay | 19.00 | 0 |
| Paired shares excluded with a 250 or 500 ms maximum delay | 0 | 0 |

The 19 shares reduce to two distinct book frames in two markets. Five allocations are not five independent measurements. All four matched wallet controls received their corresponding frame after the cancel request despite actually trading. This directly demonstrates why discarding fills based solely on post-request frame receipt would throw away real executions. The matching rule uses market, time, price, quantity and reciprocal uniqueness, plus frame content. It cannot turn an aggregated pending increment into a trade with a venue-issued identity.

A frame showing a trade provides an upper bound on its execution time, conditional on correct association. No paired disputed receipt preceded submission, and a receipt after cancel acknowledgment can still describe an earlier execution. With an assumed 50 ms maximum match-to-Tokyo-receipt delay all 19 disputed shares fall after the cancel acknowledgement; with 250 ms they all overlap the order lifetime. The archive supports neither a universal 50 ms maximum nor a profitable replay correction. Keep all 119.49 unresolved as to cause. The 100.49 remaining disputed shares did not have an admissible individual increment; they were not zero-valued or counted as proof of a timing error.

The first checker stopped before producing results on an integer-encoded Step 1 signer field. The completed run used the prior wallet-clock audit's eight shared identity fields with a unique-row requirement. Known-bad controls reject repeated book-frame identities and a perturbed increment. Protocol and checker commits were 45f10d1d0 and 12e4200ca.

## Decision

Close this archive timing investigation with the uncertainty recorded. The next economic gate is a single paired sensitivity on the remaining previously studied dates: freeze 50 shares per level and the 150-share reservation cap; compare the current execution assumptions and a predeclared conservative order lifetime, applied to both sides through the full inventory path. Include eligible zero-fill markets. Report daily FIFO turnover, residual at payout, settlement, matched volume and concentration. If the nominal result already loses or the sign depends on uncertain exit credit, stop the configuration. A pass would justify considering a separate prospective validation; it would not establish the simulator's real-fill accuracy. No further Tokyo capture or live trading is required for this gate.

[Per-allocation and control evidence](rows.csv) · [Frozen method](protocol.md)
