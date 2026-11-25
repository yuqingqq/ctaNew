# Inventory risk near the opening level, November 25, 2026

The BTC distance identifies contested markets. Adding its probability estimate to the Predict midpoint did not improve the adverse-move forecast. Applying the declared $50 budget also gave up more completed turnover than it saved in marked residual inventory on the three dashboard markets.

## Risk measurement

At 150 seconds, among valid market observations, the mean one-sided adverse move from midpoint to actual resolution was 25.31¢ long / 22.26¢ short for the 288 observations within half an estimated remaining BTC standard deviation of the open. Beyond 1.5 standard deviations it was 7.24¢ / 7.49¢ in 335 markets. Each side includes zero losses in favourable outcomes. These are losses on hypothetical shares carried to settlement. They do not describe filled-order P&L.

The forecast compared the current YES midpoint against a 50/50 blend with the existing BTC probability. Both scored identical market-clock observations. The blend made adverse squared-move prediction error and Brier score worse on each of the four consumed days. Of 9,216 scheduled observations, 7,616 were usable, 1,529 lacked an eligible two-sided book, 63 lacked a usable BTC feature and eight lacked a usable resolution. The forecast used futures distance from its own opening mid and a modeled one-second information delay. Basis relative to the settlement reference, tails in BTC returns and book availability matter. Directional trend was outside this candidate.

## Policy pilot

Three selected markets, matched full-feedback paths and the same 298-second END mark. WIDE used its existing pressure, structural, opening and inventory controls. BOOK and BTC additionally sized new risk-adding posts to a fixed $50 directional RMS downside allowance. All prior passive protections continued; the overlay had no authority to cancel an existing order. Market and share caps continued to apply. Missing BTC reverted to the book. The first second had unavailable BTC in each replay. The comparable book-only sizing policy isolates the incremental use of BTC.

| Policy | Completed turnover, USD | Residual at END, USD | Net at END, USD |
|---|---:|---:|---:|
| WIDE | +100.93 | −53.14 | +47.79 |
| BOOK | −19.67 | −27.69 | −47.36 |
| BTC | −26.01 | −22.96 | −48.98 |

Market 3789484 drove the lost opportunity. Peak absolute shares fell from 464 to 356 under BTC, while completed turnover fell from +$253.82 to +$122.14. The residual improved by $10.15. In 3877472 the same policy reduced the END loss by $58.65. Peak shares rose slightly in 3702960, consistent with a prospective admission rule on a feedback path.

The prespecified two-day forecast gate failed, so the extension to 19 markets was stopped. These results do not establish an optimal risk appetite or measure an actual maximum loss. One score overstates the cost of holding lots that would have closed before resolution. The next useful measurement is the amount of inventory that stays unmatched to expiry versus the spread earned on lots closed earlier in the same conditions. This would establish the horizon that a risk budget should price before trying another cap.

## Verification

The WIDE replay reproduced all stored fills, END values and both-sided active fractions on the three markets. Independent checks enumerated reservation-room outcomes, rejected an omitted-reservations implementation and verified both sides' squared-downside identity and the nine full-book accounting decompositions. The initial replay stopped before results because the occupancy observer iterated dictionary keys; the corrected observer reproduced the saved control. Design freeze 7f648960b, observer correction 2524d2e0b. All runs used the codex slot. No fit, untouched day or live change. The [forecast](measure.log), [pilot](pilot.csv), [verification](verify.log) and [declaration](contract.md) are saved alongside this report.
