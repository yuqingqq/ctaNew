# RECEIPT-LEAD-TIME-AUDIT-v1 with the corrected ladder diagnostics — report

2026-11-21 17:54Z, session aea5bfb9. Review `0065191f`'s ONE next priority on its accepted baseline (`869fa85f`): the same 19 complete markets and 5,288 generations; all 96 statuses kept (complete 19, incomplete:missing_endpoint 54, incomplete:stale_future_book 23).
Freeze `b1167651` before any saved evidence was read; checker supersessions `b8642e6d` (v2) and `349d2f32` (v3), each committed before it ran. No fit, guard, profit or avoided-loss claim; no replay.

## Corrected diagnostics against the review's independent numbers (checks_v3.json C_*)

| item | audit | review | equal |
|---|---|---|---|
| F1 fills in [pending_first, live_first) (submit + 95 .. + 137 ms) | 235 fills / 93 generations / 2,175.7 shares / +32,481.2 c | 235 / 93 / 2,175.7 / +32,481.2 c | equal |
| F2 sweep fill rows: public-trade events / timestamp groups / timestamp-only | 388 / 2,132 / 1,744 | 388 / 2,132 / 1,744 | equal |
| F3 sampled live states: n / lifetime any-fill / future-window any-fill | 4,840 / 2,074 / 1,702 | 4,840 / 2,074 / 1,702 | equal |
| F3 first reviews: n / lifetime any-fill / future-window any-fill | 4,840 / 2,074 / 1,986 | 4,840 / 2,074 / 1,986 | equal |
| F4 requested by the first live tick (structural / policy); policy-only first-review pulls | 147 (143 / 4); 4 | 147 (143 / 4); 4 | equal |
| F4 restriction episodes: recovered / reversal / censored | 252 / 23 / 12 (recovered median 7,275.0 ms) | 252 / 23 / — | equal |
| fills before any request / cancel-pending: fills, shares, +1 s (per fill, valid) | 8,647, 92,497.6, -173,618.2 c / 559, 4,493.5, -29,465.7 c | 8,647, 92,497.6, -173,618.2 c / 559, 4,493.5, -29,465.7 c | equal |
| F8 both-side venue-active share: exact (grid) | 40.70 % (40.31 % grid mean of markets) | 40.70 % (40.31 %) | equal |
| equal-population edge / +50 ms / +1 s (all-horizon common subset) | 7,796 fills: +35,625.2 c / -65,535.8 c / -161,824.3 c | 7,796: +35,625.2 c / -65,535.8 c / -161,824.3 c | equal |
| F7 net top-5 losing-market share (losing markets) | 41.49 % (19) | 41.49 % | equal |

- Sweep tranches: the 388 public-trade event rows carry END -48,305.7 c and +1 s -22,605.9 c; the former table summed every fill of any ever-flagged generation.
- The audit's own fill-timing aggregate censors a whole generation when one fill lacks its +1 s mid (-173,449.7 c before a request); the per-fill valid sum above is the review's convention. Both are checked.
- Generation tails: worst END -8,550.0 c, END severe sum -1,884,949.6 c; worst +1 s -5,250.0 c, +1 s below -30 c sum -259,294.9 c. END legs of flat round trips carry opposite tails (selftest control: +3,000 / -2,900 c for a +100 c round trip).

## Lead time: where the +1 s adverse value sits relative to the earliest effective command (states.csv; one window per snapshot, kinds never pooled)

| kind | snapshots | censored +1 s | window any-fill | lifetime any-fill | window +1 s | fills before the effect +1 s | cancel-pending window +1 s | held inventory move to +1 s | decision -> effect | Predict obs. age | Binance obs. age |
|---|---|---|---|---|---|---|---|---|---|---|---|
| entry | 5,288 | 13 | 2,080 | 2,080 | -202,915.4 c | +0.0 c | -29,465.7 c | -54,662.5 c | 58.0 ms | 28.0 ms | 19.5 ms |
| first | 4,840 | 13 | 1,986 | 2,074 | -185,438.2 c | -4,130.1 c | -29,152.1 c | -43,742.1 c | 37.0 ms | 29.0 ms | 21.5 ms |
| live | 4,840 | 9 | 1,702 | 2,074 | -179,659.5 c | -8,669.9 c | -29,152.1 c | -31,023.3 c | 37.0 ms | 30.6 ms | 22.5 ms |

- Entry: withholding the POST avoids every fill of the generation (the window); the effect is its activation (58 ms). First / live: a PULL at d lands at d + 37 ms; fills in [d, d + 37) cannot be prevented by it.
- The held-inventory move is the client-known inventory at d times the venue-mid change to d + 1 s: inventory already acquired, apart from the order's future fills. Windows of different kinds overlap; never add across kinds.

## Do receipt-known states separate bad future fills? (the declared lead rule on the frozen Oct 24-25 bins)

**Computed reading:** LEAD — kind `live`, input `ctx_bid_reserved_shares`, sign -1, pooled q5 - q1 -23.1 c per state; 1 lead(s) of 78 (kind, input) pairs.

| kind | input | lead (all 8 cells one sign) | pooled q5 - q1 (+1 s per state) |
|---|---|---|---|
| first | adv_bn_r250 | — | -110.2 c |
| live | adv_bn_r250 | — | -93.1 c |
| live | adv_d_bnimb250 | — | -30.4 c |
| live | r_s1 | — | -28.7 c |
| live | r_own_pend | — | +27.0 c |
| first | bn_ntr1000 | — | +23.8 c |
| live | ctx_bid_reserved_shares | LEAD | -23.1 c |
| entry | adv_bn_r250 | — | -21.5 c |
| first | r_own_pend | — | +20.8 c |
| live | adv_bn_r1000 | — | -20.5 c |

**Post-hoc descriptive view (computed here from the frozen bins.csv after the reading; the rule and the reading are unchanged):** the +1 s outcome is a SUM over the window's shares, so an input correlated with the order's own remaining quantity can move it by size alone. Per window share:

| day | side | q1 states | q5 states | q1 mean +1 s | q5 mean +1 s | q1 per share | q5 per share | per-share sign as the lead |
|---|---|---|---|---|---|---|---|---|
| 20261024 | bid | 195 | 480 | -13.5 c | -21.3 c | -1.257 c / share | -1.351 c / share | yes |
| 20261024 | ask | 117 | 189 | -10.1 c | -68.2 c | -0.674 c / share | -5.421 c / share | yes |
| 20261025 | bid | 100 | 325 | -49.1 c | -62.8 c | -4.367 c / share | -3.624 c / share | no |
| 20261025 | ask | 67 | 109 | -14.0 c | -61.1 c | -0.855 c / share | -3.640 c / share | yes |
| 20261026 | bid | 163 | 331 | -29.9 c | -30.3 c | -3.332 c / share | -2.156 c / share | no |
| 20261026 | ask | 121 | 162 | -3.7 c | -41.3 c | -0.402 c / share | -3.920 c / share | yes |
| 20261027 | bid | 39 | 122 | -7.5 c | -54.8 c | -2.418 c / share | -2.327 c / share | no |
| 20261027 | ask | 39 | 57 | -49.2 c | -66.1 c | -2.998 c / share | -4.682 c / share | yes |

- The per-share difference has the lead's sign on 4 of 4 days on the ask side and 1 of 4 on the bid side.

## Causal audit of the reconstructed inputs (16 hash-selected snapshots a day)

| day | snapshots | market inputs unchanged (future perturbed) | market control fired | order inputs unchanged | order control eligible | order control fired |
|---|---|---|---|---|---|---|
| 20261024 | 16 | 16 | 16 | 16 | 14 | 14 |
| 20261025 | 16 | 16 | 16 | 16 | 15 | 15 |
| 20261026 | 16 | 16 | 16 | 16 | 14 | 14 |
| 20261027 | 16 | 16 | 16 | 16 | 16 | 16 |

## Validity

- Checks v3: 23 / 23 true; falsifier 7 / 7. Checker v1 (frozen in `b1167651`) and v2 (`b8642e6d`) crashed before computing any check (a list / Series filter; the day column read as integers); both logs are kept, each superseded by a version committed before it ran.
- Mutation suites: audit 16 / 16, safe resume 6 / 6; the frozen baseline harnesses under selftest v2: client 24 / 24, runner 20 / 20. The audit suite's first run (14 / 16) is kept: one vacuous assertion (precedence) and a flat-mid fixture.
- The frozen baseline selftest v1 fails after the commit as the review found (logs/ladder_selftest_v1_now.log); v2 makes the guard conditions explicit.
- A `sed` edit of the frozen launcher made after every audit stage had run was reverted by `git checkout` before any further stage (frozen() true again); no output came from the edited file.
- Cost: 0.196 core-hours of the 0.5 cap; peak 1,252 MiB.

## Limits

- One fixed policy on 19 of 96 selected markets: associations are leads for the next experiment, not evidence that a loss was predictable or avoidable, nor guard value.
- The lead rule compares quintile tails of one input at a time on sums of +1 s markouts; a later candidate must price lost good fills, queue / re-entry and inventory effects.
- M50 trade timing remains the declared assumption; the snapshot inputs were built in the run on the receipt clock.
