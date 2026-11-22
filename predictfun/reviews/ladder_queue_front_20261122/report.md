# QUEUE-PRIORITY-SENSITIVITY-v1 — receipt

2026-11-22, session aea5bfb9. The USER's plan `425e9fbc`, approved with its caps and pilot.

| Step | Commit / result |
|---|---|
| Freeze, before any replay | `b4ecbc34` |
| Checker v2, committed before it ran | `580411db` |
| Run | 4 pilot dates in parallel → write-once projection (run_rest true) → 3 rest dates → measurement → checks |

Scope:
- 19 complete markets, consumed Oct 24–27;
- **a hypothetical execution sensitivity**, not an executable policy, an upper bound, a validation or a profitability claim;
- four days, so point estimates only.

## Integrity

| Item | Result |
|---|---|
| Control | CURRENT_QUEUE = the accepted pressure arm on **19 / 19** markets: fingerprint, fills, orders, requests, value, guard log, command array, queue records |
| Treatment | FRONT set the queue ahead to 0 for **4,874 / 4,874** accepted generations. CURRENT's 4,535 accepted generations were untreated; 4,152 of them start behind external queue |
| Independent checks | `fq_checks_v2.py`: A / B / K / T / F / M / P / I all pass. The falsifier refused 10 / 10 corruptions |
| v1 checker | `fq_checks.py`'s falsifier helper crashed before any verdict (a tuple-keyed `**` spread; `logs/qf_checks.log`). v2 changes only that helper |
| Mutation | 26 / 26 caught |
| Resources | Batch **0.196 core-hours** (cap 0.25; the projection allowed 0.212). Line **2.172 of 2.25**. Peak 1,186 MiB |

## Result (FRONT_AT_ACTIVATION − CURRENT_QUEUE)

| | BASELINE | CURRENT_QUEUE | FRONT_AT_ACTIVATION | FRONT − CURRENT |
|---|---:|---:|---:|---:|
| END (19 markets, c) | −209,558 | −86,110 | −65,937 | **+20,173** (recovers 23.4 % of CURRENT's deficit) |
| END by day (Oct 24 / 25 / 26 / 27) | | −33,011 / −29,622 / −9,275 / −14,202 | −9,119 / −28,490 / −12,231 / −16,097 | +23,892 / +1,131 / **−2,956 / −1,895** |
| Eligible +1 s total (c) | −203,084 | −85,164 | −61,812 | +23,352 (by day +12,188 / +9,342 / +7,504 / **−5,682**) |
| +1 s c / share | −2.10 | −1.08 | −0.53 | +0.55 (better on every day) |
| Filled shares | 96,991 | 79,399 | 117,506 | +48 % |

Markets: 13 better, 6 worse.

**Computed readings:**
- END is MIXED and +1 s is MIXED, so **economic = MIXED**.
- adds_fills = true.
- **reading = `mixed`** by the declared map. The pooled END improves, but not on every day, and neither does the +1 s total.
- **Every arm's END is negative on every day.**

**Where the +1 s change comes from** (the declared mix split, eligible +1 s):

| Class | CURRENT shares | CURRENT c / share | FRONT shares | FRONT c / share |
|---|---:|---:|---:|---:|
| same_price (queue matters) | 37,422 | +0.05 | 85,088 | +0.25 |
| through (sweeps fill regardless of queue) | 41,632 | **−2.09** | 31,905 | **−2.59** |

- The **volume term is +22,562 c**: mostly fewer through-price shares (+20,310) plus more same-price shares (+2,251).
- The **value term is +790 c**: same-price per-share up (+16,863), through-price per-share down (−16,072).
- So the improvement is mostly that front orders fill out at the same price before sweeps reach them. The sweeps that remain are worse per share.

**The 1-second loss is almost all sweep fills** in both arms: −86,932 c of CURRENT's −85,164 c. Same-price fills are about break-even at 1 s even at the front, and drift negative at longer marks: −0.30 / −0.63 c/share at 5 / 30 s for CURRENT, −0.09 / 0.00 for FRONT.

**At the END mark the class picture flips.** CURRENT's same-price fills carry −95,333 c (−2.55 c/share), while its through fills carry +9,223 c. END is the pre-expiry mid, and it includes the inventory's directional exposure to each market's finish. A class attribution of END is therefore not stable, and its day signs vary.

**Participation** (FRONT vs CURRENT):

| Measure | CURRENT | FRONT |
|---|---:|---:|
| Share of time quoting both sides | 0.402 | 0.356 |
| Generations | 4,811 | 5,196 |
| Pressure pulls / withholds | 433 / 1,289 | 560 / 1,708 |
| First-review pulls | 25 | 51 |
| Fills | 7,395 | 11,595 |

**Lost / added fills** (multiset convention):
- lost: 4,274 fills / 50,508 shares, END −42,982;
- added: 8,474 fills / 88,614 shares, END −22,810.

## What this shows and does not show

**The USER's hypothesis — queue disadvantage explains most of the losses — is not supported on this evidence.**
- Giving every order perfect front priority, an extreme and non-executable treatment, recovers 23 % of the pooled END deficit.
- That gain is inconsistent by day: 2 of 4 days are worse.
- It leaves the policy losing on every day: −65.9 k c, −0.53 c/share at 1 s.
- Queue priority only changes same-price fills, which are about break-even at 1 s. The short-horizon loss is in through-price sweeps, which fill an order wherever it sits in the queue.

Limits:
- The engine fills through-price trades without the queue.
- External flow is held unchanged, so diverted fills would change real makers' behaviour.
- END mixes in directional exposure.
- Four consumed days.

Per the plan, no tuned intermediate queue advantage or automatic variant follows.

**Largest evidenced limitation:** the residual loss is sweep (through-price) adverse selection plus directional inventory exposure. Priority does not address either. Any further economic gain would have to come from avoiding or pricing sweeps at decision time, or from controlling one-way inventory, not from queue position.

Files:
- `logs/` holds the projection and every job log;
- `checks_v2.json` and `checks_falsifier_v2.json`;
- the measurement outputs (`arms`, `control`, `lost_added`, `classes`, `treatment` .csv and `audit.json`) are copied to `outputs/`.
