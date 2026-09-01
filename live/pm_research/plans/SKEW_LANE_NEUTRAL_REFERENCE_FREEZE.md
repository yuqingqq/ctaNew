# QR_SKEW_ONLY as the NEUTRAL INTEGRATION REFERENCE — freeze draft

**STATUS: FROZEN — USER RULING 2026-09-01 ("release and do skew freeze,
proceed"). No fit, no score, no code change: this freeze RECORDS committed
semantics and introduces nothing.**

**Q1–Q3 of §7 were NOT answered in that ruling, and are resolved below by the
COORDINATOR under its authorization — recorded as coordinator resolutions, not
as user rulings, because this programme attributes rulings precisely and an
unattributed resolution is how an implementation choice becomes a silent design
decision (§A1.0 of the iteration-011 amendment).** All three take the same
conservative form: **record what committed code contains, invent nothing** —
which is this draft's own stated principle (§9: "It introduces no new number").
**Each is CORRECTABLE by a one-line user ruling, superseding in-band (rule 13);
none is load-bearing for any number, because nothing is fitted or scored here.**
**Scope:** hazard plan §10 item 5 / STATEFUL_HARMFUL_CANCEL_TODO §5.3 Phase 2C.
**Drafted by:** BE. **Frozen by:** the USER, after the reviewer has seen it.

Every semantic below is **cited to code** (rule 16), not recalled. Where the
code is ambiguous the draft **states the ambiguity as a question** rather than
resolving it — the R-237 / A1 pattern. **No number appears here that is not
already in committed code.**

---

## 1. What QR_SKEW_ONLY is, as implemented

Declared by `_qr_spec` in `policy_optimizer_queue_realistic.py:45-61`, and
instantiated for the neutral arm at `:66` with `cancel=False`:

| field | value for QR_SKEW_ONLY | source |
|---|---|---|
| `placement` | `QUEUE_REALISTIC_SKEW` | `policy_optimizer_queue_realistic.py:48` |
| `skew` | `True` | `:49` |
| `skew_band_shares` | `placement_skew.SKEW_BAND_SHARES` | `:50` |
| `front_on_repost` | `False` | `:51` |
| `queue_realistic` | `True` | `:52` |
| `cancel` | **`False`** | `:66` passes `cancel=False` |
| `cancel_latency_ms` | the run's `latency_ms` | `:54` |
| `cancel_join_only` | `True` | `:55` |
| `cancel_hold` | `False` (= `cancel`) | `:56` |
| `protect_reducing_side` | `False` (= `cancel`) | `:57` |
| `r_cut` | `0` | `:58` |
| `size` | `5.0` | `:59` |

`SKEW_BAND_SHARES = QUOTE_SIZE` (`placement_skew.py:46`), and
`QUOTE_SIZE = fd.ACTION_SIZE` (`inventory_walk.py:34`).

**The neutral reference does not cancel.** `cancel=False` makes `cancel_hold`
and `protect_reducing_side` false by construction, so the cancel-side fields
above are inert for this arm. That is what makes it the no-cancel shadow the
whole harmful-fill line is measured against.

## 2. Placement semantics

`placement_skew._target_front(net, band)` (`placement_skew.py:64-80`) returns
`(buy_front, sell_front)`:

```text
net >  band  -> (False, True)    long Up  -> reducing side SELL_UP goes FRONT
net < -band  -> (True,  False)   short Up -> reducing side BUY_UP  goes FRONT
otherwise    -> (False, False)   near flat -> both join the BACK (baseline)
```

`SimArm.apply_skew_intent` (`policy_optimizer.py:104-117`) applies this and its
docstring fixes the queue semantics exactly:

> *"Change placement intent, never current queue position. `SKEW_LB` receives
> the new intent only on the next genuine touch formation. Its BoundedSide also
> rejoins behind displayed depth after a full lift, which is the pessimistic
> queue rule frozen for Stage B."*

Three consequences worth freezing explicitly, all read from that method:
1. skew changes **intent**, never an order's existing queue position;
2. the new intent takes effect **only at the next genuine touch formation**;
3. after a full lift the side **rejoins behind displayed depth** — the
   pessimistic rule.

`skew_intent_flips` is counted per arm (`:114-115`) and surfaced as a
diagnostic (`policy_optimizer_queue_realistic.py:515-516`).

## 3. What the state machine consumes from it

`harmful_stateful_policy.py:10-25` names three inputs, **none optional**. On the
reference trajectory it is explicit:

> *"a REFERENCE TRAJECTORY: the QR_SKEW_ONLY no-cancel shadow, generation-native
> ... The skew rules' placement choices live entirely in this input; the
> predictor NEVER chooses placement (TODO section 2.2)."*

And on repost (`:43-45`):

> *"the join takes the CURRENT reference generation's side/level/size —
> ordinary skew rules, never the predictor."*

**This is the §2.2 fence in operational form:** placement is the skew lane's,
scoring is the predictor's, and the predictor cannot reach placement even on a
repost.

## 4. Policy interface (TODO §5.3)

The four quantities the policy layer is entitled to, and where each already
exists in code:

| interface quantity | in code today | citation |
|---|---|---|
| inventory-increasing / reducing status | `_is_reducing(side, net)` | `harmful_stateful_policy.py:544-548` |
| allowed placement | `_target_front(net, band)` → front/back per side | `placement_skew.py:64-80` |
| desired exposure | **not a single named object**; expressed jointly by `size` (5.0) and the front/back intent | see §7 Q1 |
| marginal inventory-risk value | **not present as a returned quantity** | see §7 Q2 |

`_is_reducing` fixes the flat case explicitly:

> *"net == 0 means NEITHER side reduces (both grow |net|), so it classifies as
> increasing."*

## 5. Declared ablation set

`PROTECTION_MODES = ("REDUCING_SIDE_PROTECTION", "ALL_ORDERS_OVERRIDE")`
(`harmful_stateful_policy.py:114`), validated at `:239-244` — an unrecognised
mode is `InvalidParameter`, not a default.

The reduce lane is an **explicit ablation flag, default OFF**
(`harmful_stateful_policy.py:24`, `:257`): `enable_reduce` defaults to `False`,
and if set `True` it **requires** `theta_reduce` and `reduce_remaining_fraction`
to be declared (`:260-264`) — an inert declaration is refused, not ignored
(`:281`).

## 6. Parity requirement

DA's battery pins the canon `replay_traj_canon_v1`
(`da_replay_parity_battery.py:52`) and requires **bit-identical, not
within-tolerance** agreement (`:395`), with the reason stated there: *a
tolerance would hide* the class of defect the check exists to catch.

Two conditions the battery already asserts against QR_SKEW_ONLY:
- predictor **disabled** → bit-identical to the reference (`:400`, `:410-411`);
- an **infinite** cancel threshold → bit-identical to QR_SKEW_ONLY (`:912`).

Freezing this lane means: any seven-arm replay must reproduce the QR_SKEW_ONLY
digest under those two conditions, or the lane has moved.

## 7. AMBIGUITIES — questions for the USER, not resolved here

**Q1 — "desired exposure" has no single representation in code.** §5.3 names it
as an interface quantity, but the implementation expresses it jointly through
`size = 5.0` and the front/back intent. Freezing an interface that names an
object the code does not have would either invent it or quietly redefine it as
"size". *Should the freeze name the pair, or should a single `desired_exposure`
be introduced later as a declared change?*

> **RESOLVED 2026-09-01 (COORDINATOR, under the USER's freeze authorization;
> CORRECTABLE): NAME THE PAIR.** Desired exposure is frozen as the two
> quantities the code actually has — `size` (5.0, `policy_optimizer_queue_
> realistic.py:59`) together with the per-side front/back intent from
> `_target_front` — and NOT as a single object. Introducing a
> `desired_exposure` scalar now would either invent a quantity no caller
> computes or silently redefine it as `size`, and the two are not the same:
> `size` is how much, the intent is where. A single object remains available
> later as a DECLARED change with its own preregistration, which is the only
> route that keeps it from being back-fitted to whatever the code then does.

**Q2 — "marginal inventory-risk value" is not a returned quantity.** The
inversion it rests on is documented in `_target_front`'s docstring (NEW_BBO's
"~9.4x inventory risk"), but that is a rationale in prose, not a computed value
any caller receives. *Does the freeze record it as NOT-PRESENT (my
recommendation, since the alternative is to freeze a number nobody computes), or
is a implementation expected before the freeze?*

> **RESOLVED 2026-09-01 (COORDINATOR, under the USER's freeze authorization;
> CORRECTABLE): NOT-PRESENT, as BE recommended.** The interface records that no
> marginal inventory-risk value is returned by any code path. The "~9.4x
> inventory risk" in `_target_front`'s docstring is a RATIONALE IN PROSE and is
> explicitly NOT frozen as a quantity — freezing it would put a number into the
> interface that nothing computes and nothing can falsify, which is the precise
> failure this draft's §9 forbids. A consumer needing it must first implement
> and validate it as a declared change; until then, callers get the front/back
> intent, which is the decision that rationale supports.

**Q3 — the code already declares one ambiguity and refuses to guess.**
`harmful_stateful_policy.py:46-49`: whether a repost landing exactly on a
reference generation start is charged the queue-reset cost is itself a declared
parameter, `charge_reset_cost_at_generation_start`, because *"the spec is
ambiguous there and this module refuses to guess"*. That ambiguity is in the
policy layer rather than the skew lane, but it touches the reference boundary.
*Should the skew-lane freeze cite it as out-of-scope, or resolve it?*

> **RESOLVED 2026-09-01 (COORDINATOR, under the USER's freeze authorization;
> CORRECTABLE): OUT OF SCOPE, cited not resolved.**
> `charge_reset_cost_at_generation_start` is a POLICY-LAYER declared parameter
> that already refuses to guess, which is the correct handling of a genuinely
> ambiguous spec (rule 14: decisions live in the policy layer with their own
> priced trade-offs). The skew lane defines the reference trajectory; what a
> repost is CHARGED on landing is the policy layer's question. Resolving it
> here would move a priced decision into an interface freeze and would settle
> it without the trade-off ever being priced. It is cited so the boundary is
> visible, and it must be ruled before any lifecycle-economics number is
> claimed — it is a live obligation, not a closed one.

BE answered none of these; answering Q1 or Q2 in the DRAFT would have
introduced a semantic that no committed code contains. The resolutions above
introduce none either: each records absence or an existing pair, and each is a
coordinator resolution the user can overturn in one line.

## 8. NO-SELECTION CLAUSE

**No band, hysteresis or skew threshold may be chosen on 2026-08-20..25.** Those
days are consumed for the harmful-fill line (CLAUDE.md rule 11); 08-26 is ruled
FAIL (Q-DA-72) and 08-27 EXCLUDED (R-222). `SKEW_BAND_SHARES` is frozen at
`QUOTE_SIZE` **as already committed** — this draft does not select it, it
records it.

Any future change to the band, to `_target_front`'s thresholds, or to the
protection modes is a **new selection** requiring its own preregistration on
untouched days, not an adjustment to this freeze.

## 9. What this freeze does NOT do

- It changes no code in QR_SKEW_ONLY or anything it calls.
- It introduces **no new number**. Every value cited exists in committed code.
- It does not put inventory, skew, cooldown or lifecycle into the harm
  predictor. Those remain **policy-layer inputs only** (§2.2), and §3 shows the
  code already enforces that.
- It does not resolve Q1–Q3.
