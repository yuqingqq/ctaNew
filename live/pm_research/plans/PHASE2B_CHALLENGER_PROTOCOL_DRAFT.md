# Phase 2B — fair-price challenger protocol

**STATUS: DRAFT-FOR-USER-FREEZE. NOTHING IS SCORED UNDER THIS DOCUMENT.**
TODO §10 gates *all* 2B scoring on this protocol being recorded and frozen, so
until the user freezes it no challenger may be evaluated — not exploratorily,
not "just to look".
**Owner:** DA. **Authorised as a draft by** the user's plan `d506a06`
(§10 item 4 / TODO §5.2), whose operative constraint is *nothing frozen or
scored*. **Baseline artifact:** `da_fair_price_identity.py` (this batch).

## 1. What is being asked

Whether any predeclared challenger beats **`Identity`** — the executable-book
price — as an estimate of `E[Y | state]`. Not whether it beats chance.

## 2. The closed set: at most TWO challengers

1. **PM microprice** (size-weighted book price).
2. **At most one cross-venue forecast.**

Declared **before** any comparison. **The set is closed at freeze**: adding a
third later is a new family and restarts multiplicity, because choosing what to
compare after seeing how the first comparison went is selection (rule 11).

## 3. Admissibility — inherited, not restated

A challenger produces the same typed record as `Identity` and is bound by the
same rules: **both timestamps or INADMISSIBLE** (never degraded, never a
default), strictly-as-of consumption, statuses instead of silent zeros.
**A challenger that cannot state when it could first have been known is not a
slower challenger — it is an unusable one.**

## 4. The score: PROPER, and INCREMENTAL TO IDENTITY

- **Proper score:** Brier on the settled binary outcome, `(p - y)²`. Proper, so
  the honest forecast maximises the expected score; a non-proper score rewards
  hedging and would make a shaded challenger look skilful.
- **Reported as skill vs `Identity`**, per day:
  `skill_d = 1 - BS(challenger)_d / BS(Identity)_d`.
- **THE INCREMENT IS THE ESTIMAND, AND THIS IS RULE 9.** PM binaries settle on
  a Binance-derived price, so a challenger reading that same price scores well
  against *chance* while adding nothing to what the book already says. **Skill
  against a base rate is meaningless here; only skill incremental to `Identity`
  is a result.**
- **Paired on identical decision instants.** Both estimators are scored on the
  same admissible records, matched by `(coin, window, outcome, decision time)`.
  A challenger scored on a different or larger population is not a comparison.

## 5. PIT parity — the challenger must not read further ahead

For every scored instant both estimators satisfy
`local_knowledge_timestamp <= decision_time`, and **the paired instants are
identical**. A challenger with a systematically later local-knowledge time is
not better, it is later — and would win by reading more of the future.
**Report the per-estimator freshness distribution beside the score**, so a skill
difference explained by a latency difference is visible rather than inferred.

## 6. Cluster unit and reporting

- **Unit: the UTC day** (rule 8). Below **G = 5 complete days**: point estimate,
  no interval, and say so.
- Per-day skill, `n` admissible paired instants, and the **status tally on both
  sides** (rule 4) — a challenger that wins by being admissible less often has
  not won.
- **Multiplicity recorded at freeze**: candidates in the race, budgets, and any
  earlier look. Two challengers × the declared budgets is the family; the joint
  reading is Holm across it, and a single uncorrected cell is not a result.

## 7. Declared before the data: what would make a challenger PASS

Stated now so it cannot be chosen afterwards. A challenger is adopted only if
**all** hold:
1. Positive `Identity`-incremental skill, **Holm-corrected across the family**.
2. **≥5 complete UTC days**, all after the protocol freeze; consumed days stay
   consumed.
3. Freshness distribution **not systematically later** than `Identity`'s (§5).
4. Admissible-instant count **not materially below** `Identity`'s (§6).

**A failed challenger does not block anything.** The full policy runs with
`Identity`. That asymmetry is deliberate — it removes any incentive to keep
looking until one passes.

## 8. Falsifiers required before any scoring run (rule 15)

1. A challenger missing `local_knowledge_timestamp` is **REFUSED**, with a
   positive control that a complete record is accepted.
2. Scoring on **unpaired** populations **REFUSES** — a synthetic case where the
   challenger has extra instants must not silently score.
3. `Identity` **versus itself** yields skill exactly **0** — the null the whole
   comparison rests on; if it is nonzero the pairing or the score is wrong.
4. A challenger that is `Identity` **plus a constant lag** shows **negative or
   zero** skill, never positive: a pure-latency "improvement" must not read as
   one.
5. An **empty** or all-inadmissible day reports **NOT EVALUABLE**, never a
   passing skill of 0.

## 9. What this document does not do

No scoring, no fitting, no promotion, no forward clock. Adoption of a passing
challenger remains a policy decision with its own priced trade-offs (rule 14):
this protocol estimates, it never decides.
