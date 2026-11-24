# Directional-entry replay: three consumed dashboard markets

2026-11-24. Frozen implementation/manifest/declaration/checker: fa9d035d2.
One new unfitted candidate; nine full-market arms, no new days or model fit.
All9 saved results pass independent checks; the ORIGINAL arm reproduces the
saved policy on all3 markets. Seven deliberate corruptions are refused.

## Result

All amounts below are USD, pre-expiry marked value, zero maker fees/rebates
as in the original simulator. The same END mark and full supported interval
apply within each market. These are three selected illustrations, not validation.

|Market|Original as saved|Fixed inventory, default entry|Fixed inventory, directional entry|Entry delta vs fixed default|
|---|---:|---:|---:|---:|
|3702960|+29.42|+4.28|-2.99|-7.26|
|3789484|-17.90|-17.40|+31.92|+49.32|
|3877472|-6.54|-136.46|-7.66|+128.79|

Pooled fixed-default value is$-149.58; directional entry is
$+21.27, a$+170.85 difference.
The gate is better in2/3 markets; 1/3 candidate markets has positive
absolute value. The positive pooled result is concentrated in market3789484.
Filled shares fall from11548.7 to2358.5
(79.6% lower). No matched participation null was run, so this
does not establish selection skill beyond reducing exposure.

|Market|Arm|Completed FIFO USD|Residual USD|Filled shares|Peak absolute inventory|Both sides active,50ms grid|Posts|Pulls|First post ms / sides(0 bid,1 ask)|
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
|3702960|FIXED_DEFAULT|+5.68|-1.40|2714.1|127.8|50.9%|141|20|0.0 / [0, 1]|
|3702960|DIRECTIONAL_ENTRY|-0.35|-2.64|125.7|108.2|0.0%|27|1|29050.0 / [1]|
|3789484|FIXED_DEFAULT|-10.32|-7.08|3859.7|191.2|43.2%|598|180|0.0 / [0, 1]|
|3789484|DIRECTIONAL_ENTRY|+25.18|+6.75|1175.6|101.7|8.8%|292|59|550.0 / [0]|
|3877472|FIXED_DEFAULT|-124.68|-11.78|4974.9|199.8|32.8%|349|52|0.0 / [0, 1]|
|3877472|DIRECTIONAL_ENTRY|-7.68|+0.02|1057.2|168.8|0.5%|126|11|850.0 / [1]|

Completed FIFO delta is$+146.47; residual
delta is$+24.39. The whole-policy gain
must not be attributed entirely to an initial-fill prediction. The accounting
is whole-path FIFO, not claimed intent; residual exposure and turnover remain
separate. Completed matched shares are counted once per buy/sell pair.

## The user-selected opening example

Market3789484: initial entry moves from0ms/both sides to
550.0ms/sides[0].
First-minute common-mark value is
$-22.08 for fixed default,
$+0.05 for the gate.
Full-market values are$-17.40 and
$+31.92. This avoids quoting both sides
at tick0. Its completed first-minute matched volume is only
12.00 shares,
against322.12 for the
fixed default. It does not reproduce maker3's private rule or prove its decisions
used this signal. Other gate/post/inventory decisions also change later.

## What actually changed

An existing favourable Binance-pressure condition permits a new risk-increasing
quote. Otherwise the proposed size is restricted to unreserved reduction of
known inventory; flat/missing signals do not authorize new risk. No probability
or expected-value model is fitted. Thresholds/window are the accepted guard's,
not chosen from this replay. Existing live-order cancellations are unchanged:
a flat entry signal alone never cancels a resting order.

Both primary comparison arms fix the known skipped-deadband inventory bug.
The original stopped quoting for long intervals after sign-crossing fills;
restoring trading can expose more losing fills. A worse value after this fix
is not a reason to retain incorrect inventory logic. Its contribution is
explicit in checks.json, separate from the entry contrast.

## Reliability and limits

The original loader, tape/mark identity, current queue, received signal cache,
10ms activation/cancel,100ms notices,50ms grid and M50 assumed execution clock
are unchanged. Local wrappers leave frozen implementation files untouched.
The input/source manifest was committed before any replay. Checks reconcile
cash, inventory, END value, fills, per-order quantities, entry tick and side,
known inventory from noticed fills, and reservations from prior commands and
closures; corrected modes never block the wrong inventory side. All three
original fingerprints/fills/orders/requests/values match the saved controls.

The initial preparation refused on a wrong imputation-file path before a
manifest or replay existed. The adapter was corrected to the same Tape.offset
source and committed in the freeze; both logs remain. No run amendment or
economic rule correction was made after the freeze.

The checker does not estimate sampling uncertainty, calibrate execution
conditional value, or isolate signal selection from reduced participation.
An empirical significance null is intentionally absent: this is a bounded
feedback-policy diagnostic, with no predictive-skill claim. Markout values
at50ms/1s/5s/30s and valid share denominators are retained in summary.json,
with the declared shared END−30s cutoff; these are not the primary score.

## Next priority

Keep this exact rule frozen. Before tuning or expanding it, compare it with
a declared activity-matched entry control on the same markets, using complete
turnover and residual value. This separates useful directional timing from
the benefit of doing less. The present replay justifies that question; it
does not justify deployment, more aggressive size, or a claim of profitability.
Any further replay must fit a separately bounded resource decision; no extra
candidate is launched by this receipt.
