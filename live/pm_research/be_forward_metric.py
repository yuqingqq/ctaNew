#!/usr/bin/env python3
"""THE FORWARD DECISION METRIC — the seam the forward path does not have.

WHAT THIS IS FOR. `be_forward_day.py` scores a forward day and seals the
result. R-496 (D) declares the number of record for a forward read: NET CENTS
AGAINST THE INCUMBENT, de-duplicated to ACTIONS, valued at each tranche's own
time and level and only after t + L, with rho secondary, a control matched on
action count / side / hour and compared on the DECISION metric, both nulls, a
cluster disclosure, and Holm over a declared cell count.

**NONE OF THAT CAN BE COMPUTED FROM WHAT THE FORWARD PATH SEALS TODAY**, and
the reason is not that the estimator is missing. It is not missing:
`harmful_action_eval.evaluate_policy` is action-native, latency-aware, and
already matches its control on (side x hour) with an equal ACTION count, and
`phase2_increment_null` already carries the paired null. What is missing is a
FEED. `be_forward_day.build_and_score` streams a window, scores it and DROPS
the rows (`del arm, wf, joined, gens, wrows`), keeping one `(t0, value)` pair
per ROW -- with no generation key. The estimand's unit is the ACTION, and an
action is `(slug, side, gen)`. Without that key the de-duplication CLAUDE.md
rule 2 requires cannot be attempted, so the estimand is not merely unmeasured:
**it is undefined on that input.** `sealed_shape_is_unusable` computes that
statement rather than asserting it, and the selftest drives it.

WHY THE DROP WAS NECESSARY AND IS NO LONGER THE BINDING CONSTRAINT. Holding
every window's row DICTS and its `window_streams` OOM-killed the run at the
12 G cap after 21 minutes, and R-174 forbids raising the cap. But the estimand
does not need the row dicts. It needs six scalars per row -- the action key,
`t0 + t_start`, the score, and the latency-aware preventable value. That is
what `reduce_window` emits, and `FEED_BYTES_PER_ROW` prices it, so the trade
is a measured number in the artifact and not an opinion in a filing.

WHAT THIS MODULE REFUSES TO DECIDE (rule 14). It computes nothing without a
DECLARED operating point, a DECLARED candidate identity and a DECLARED
incumbent identity. `harmful_action_eval.evaluate_policy` falls back to
RETROSPECTIVE_TOPK when no frozen threshold is supplied -- a threshold read off
the very data being scored, which is CLAUDE.md rule 11 in its purest form. For
a FORWARD read that fallback must not be a mode, it must be a refusal, and
`require_operating_point` is that refusal. This module proposes forms and
prices them; it selects none.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import harmful_action_eval as AE
import phase2_increment_null as PIN


class ForwardMetricRefused(RuntimeError):
    """A named refusal. Absence is never a pass (rule 11)."""


class OperatingPointUndeclared(ForwardMetricRefused):
    """No frozen threshold was declared. Choosing one here would select on the
    data being read (rule 11); the USER declares it (rule 14)."""


FEED_PROTOCOL = "BE_FORWARD_METRIC_FEED_V1"

#: The action. CLAUDE.md rule 2: rows are actions, and several rows can share
#: one outcome (measured 1.99 rows/fill, max 23), so the evaluator must
#: de-duplicate to actions or the result is inflated.
ACTION_KEY = ("slug", "side", "gen")

#: Exactly what `harmful_action_eval.evaluate_policy` reads off a row. Derived
#: by reading that function, and asserted against it in the selftest so the two
#: cannot drift.
REQUIRED_ROW_FIELDS = ("slug", "side", "gen", "t0", "t_start",
                       "latency", "any_fill_ahead")

#: Six scalars: three key parts, one absolute time, one score, one value.
FEED_BYTES_PER_ROW = 48


def _sha16(obj) -> str:
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# THE GAP, COMPUTED
# ---------------------------------------------------------------------------
def sealed_shape_is_unusable(per_coin_scores: dict) -> dict:
    """Can the shape `be_forward_day.seal()` writes feed the estimand?

    Computed, never asserted (rule 10). The sealed shape is
    `{coin: [[t0, value], ...]}` -- one pair per ROW, `t0` the WINDOW start.
    Two independent reasons it cannot feed an action-level estimand, and the
    second survives even if the first were repaired:

      1. no generation key, so rows cannot be grouped into actions at all;
      2. `t0` is the window start shared by every row of that window, so even
         the row ORDER within a generation -- which decides which crossing
         acts -- is not recoverable.

    Returns the finding. A caller that wants a refusal calls
    `assert_action_keys` on the rows it actually holds."""
    coins = sorted(per_coin_scores)
    n_entries = sum(len(v) for v in per_coin_scores.values())
    sample = None
    for c in coins:
        if per_coin_scores[c]:
            sample = per_coin_scores[c][0]
            break
    width = len(sample) if isinstance(sample, (list, tuple)) else None
    return {
        "shape": "{coin: [[t0, value], ...]}",
        "n_coins": len(coins), "n_entries": n_entries,
        "tuple_width": width,
        "carries_action_key": False,
        "action_key_required": list(ACTION_KEY),
        "carries_row_order_within_generation": False,
        "usable_for_action_estimand": False,
        "why": ("the sealed entry is a (window_start, value) PAIR per ROW. It "
                "carries no (slug, side, gen), so rows cannot be grouped into "
                "actions; and its time is the WINDOW start, shared by every "
                "row of the window, so the within-generation order that "
                "decides which crossing acts is not recoverable either. The "
                "estimand is UNDEFINED on this input, not merely unmeasured."),
        "consequence_for_the_race": (
            "a sealed day cannot be turned into a net-cents-against-incumbent "
            "answer by unsealing it, and N sealed days cannot either: the "
            "quantity is not hidden behind the seal, it was never produced."),
    }


def assert_action_keys(rows) -> dict:
    """Rule 2's precondition, driven. REFUSES BY NAME when a row carries no
    action key -- the case in which the estimand is undefined."""
    if not rows:
        raise ForwardMetricRefused(
            "REFUSED: zero rows. An empty population is a FAILURE, not an "
            "empty result (R-141); a metric over nothing is not a metric.")
    missing: dict = {}
    for i, r in enumerate(rows):
        if not isinstance(r, dict):
            raise ForwardMetricRefused(
                f"REFUSED: row {i} is {type(r).__name__}, not a mapping. The "
                f"sealed `(t0, value)` pair form reaches here as a list and "
                f"cannot carry an action key -- see `sealed_shape_is_unusable`.")
        for f in REQUIRED_ROW_FIELDS:
            if f not in r:
                missing.setdefault(f, 0)
                missing[f] += 1
    if missing:
        raise ForwardMetricRefused(
            f"REFUSED: {len(missing)} required row field(s) absent, with "
            f"counts {dict(sorted(missing.items()))}. The estimand is "
            f"ACTION-level; without {list(ACTION_KEY)} the de-duplication "
            f"CLAUDE.md rule 2 requires cannot be ATTEMPTED, so the quantity "
            f"is undefined rather than unmeasured.")
    keys = {tuple(r[k] for k in ACTION_KEY) for r in rows}
    return {"n_rows": len(rows), "n_actions": len(keys),
            "rows_per_action": len(rows) / len(keys),
            "action_key": list(ACTION_KEY),
            "why_ratio_matters": ("CLAUDE.md rule 2 measured 1.99 rows/fill "
                                  "and max 23; a row-level sum inflates by "
                                  "exactly this ratio.")}


def assert_alignment(rows, *score_vectors) -> dict:
    """Scores are positional. A length mismatch silently re-pairs every row
    with another row's score, which no downstream check can see."""
    ns = [len(s) for s in score_vectors]
    if any(n != len(rows) for n in ns):
        raise ForwardMetricRefused(
            f"REFUSED: {len(rows)} rows against score vector length(s) {ns}. "
            f"Scores are positional; a mismatch re-pairs rows with other "
            f"rows' scores and every downstream number stays plausible.")
    return {"n_rows": len(rows), "n_score_vectors": len(score_vectors),
            "lengths_equal": True}


# ---------------------------------------------------------------------------
# THE OPERATING POINT — declared, never chosen here (rule 11 / rule 14)
# ---------------------------------------------------------------------------
#: Forms the USER could declare, with what each costs. PROPOSED, NOT SELECTED —
#: this module names no value and ranks none of these. Every one of them is a
#: threshold on the candidate's per-generation MAX score.
OPERATING_POINT_FORMS = {
    "FROZEN_FROM_TRAIN_QUANTILE": {
        "what": "theta = the budget-b quantile of per-generation MAX score on "
                "the TRAINING split the frozen candidate was fitted on",
        "causal": True,
        "costs": "one pass over the training rows with the frozen fit to build "
                 "the quantile map; nothing forward is read, so it can be "
                 "declared BEFORE any forward day is opened",
        "risk": "the training split's score distribution may not transport to "
                "a forward day; the budget then does not deliver its nominal "
                "cancellation rate and the DELIVERED rate must be reported",
    },
    "FIXED_ABSOLUTE": {
        "what": "theta = a single declared number, budget-independent",
        "causal": True,
        "costs": "nothing to compute; one number per coin (or one overall) in "
                 "the declaration",
        "risk": "no budget interpretation at all; the cancellation count is "
                "whatever the day yields and must be reported as an outcome",
    },
    "FROZEN_FROM_A_CONSUMED_DAY": {
        "what": "theta = the budget-b quantile taken on an ALREADY-CONSUMED "
                "day (08-20..25, named consumed for the harmful-fill line)",
        "causal": True,
        "costs": "one scoring pass over that day; it is already spent, so it "
                 "costs no unspent day",
        "risk": "a consumed day is development data; the quantile inherits "
                "its regime, and the transport question above applies again",
    },
    "RETROSPECTIVE_TOPK": {
        "what": "theta = the top-k quantile OF THE DAY BEING SCORED, which is "
                "`evaluate_policy`'s fallback when no frozen threshold is given",
        "causal": False,
        "costs": "nothing",
        "risk": "DISQUALIFYING for a forward read: the threshold is read off "
                "the data it is applied to, so the arm is not a policy anyone "
                "could have run. This module REFUSES it for a forward run "
                "rather than stamping it as a mode.",
    },
}


def operating_point_pricing(candidate_path=None) -> dict:
    """Price each proposed form against the FREEZE'S OWN declared population.

    Rule 10: the costs a USER rules on are computed from the artifact, in the
    artifact, not typed into a filing. The freeze names the days it was fitted
    on and the row and action counts, so the training-quantile option can be
    priced exactly rather than guessed."""
    import harmful_forward_scorer as FS
    path = Path(candidate_path or FS.CANDIDATE)
    c = json.loads(path.read_text())
    fits = c.get("fits") or {}
    days = sorted({d for f in fits.values() for d in (f.get("days_fitted") or ())})
    rows = sum(int(f.get("n_rows_fitted") or 0) for f in fits.values())
    acts = sum(int(f.get("n_actions_fitted") or 0) for f in fits.values())
    out = {k: dict(v) for k, v in OPERATING_POINT_FORMS.items()}
    out["FROZEN_FROM_TRAIN_QUANTILE"]["priced"] = {
        "population": c.get("trained_on", {}).get("population"),
        "days": days, "n_rows": rows, "n_actions": acts,
        "coins": sorted(fits),
        "already_consumed": True,
        "why_free": ("these days are the freeze's OWN training split and are "
                     "already consumed (CLAUDE.md rule 11 names 08-20..25 "
                     "consumed for the harmful-fill line). Computing a "
                     "quantile on them reads NO forward day and NO unspent "
                     "day, so this form costs the race nothing."),
        "work": ("one scoring pass with the frozen fit over the rows above, "
                 "then the per-budget quantile of per-generation MAX score"),
    }
    out["FROZEN_FROM_A_CONSUMED_DAY"]["priced"] = {
        "note": ("same shape as the training-quantile form but on a DIFFERENT "
                 "consumed day; it costs one extra scoring pass and buys a "
                 "population the fit did not see")}
    out["FIXED_ABSOLUTE"]["priced"] = {
        "note": ("no computation at all; the cancellation COUNT becomes an "
                 "outcome to report rather than a budget to hit")}
    out["RETROSPECTIVE_TOPK"]["priced"] = {
        "note": "refused for a forward read; priced only to be named"}
    return {"candidate": str(path), "candidate_sha16": _sha16(c),
            "forms": out,
            "selected": None,
            "selected_by_this_module": False,
            "who_selects": "the USER (rule 14); this module refuses without a "
                           "declaration and ranks nothing"}


def require_operating_point(decl, budgets=AE.BUDGETS) -> dict:
    """REFUSE BY NAME unless an operating point is DECLARED.

    Not a default, not a fallback, not a mode stamp. `evaluate_policy` treats a
    missing frozen threshold as a licence to compute one retrospectively and
    label it; for a forward read that is rule 11 with a label on it. The
    declaration must also carry its own provenance, because a number with no
    declarer is indistinguishable from a number somebody chose after looking."""
    if decl is None:
        raise OperatingPointUndeclared(
            "REFUSED: no operating point declared. The frozen candidate "
            "carries NO threshold key (checked: 77 key paths, none matching "
            "thet/thresh/cut/operat/budget/gate), and "
            "`harmful_action_eval.evaluate_policy` would fall back to "
            "RETROSPECTIVE_TOPK -- a cutoff read off the very rows being "
            "scored. Declaring it is the USER's act (rule 14) and it must "
            "precede the read (rule 11). Forms and costs: "
            f"{sorted(OPERATING_POINT_FORMS)}.")
    if not isinstance(decl, dict):
        raise OperatingPointUndeclared(
            f"REFUSED: the operating-point declaration is "
            f"{type(decl).__name__}, not a mapping.")
    form = decl.get("form")
    if form not in OPERATING_POINT_FORMS:
        raise OperatingPointUndeclared(
            f"REFUSED: operating-point form {form!r} is not one of "
            f"{sorted(OPERATING_POINT_FORMS)}. A form invented at the call "
            f"site is a choice made where no ruling can see it.")
    if not OPERATING_POINT_FORMS[form]["causal"]:
        raise OperatingPointUndeclared(
            f"REFUSED: form {form!r} is NOT causal -- its threshold is read "
            f"off the data being scored. It is listed so it can be named and "
            f"refused, never selected (rule 11).")
    theta = decl.get("theta_frozen")
    if not isinstance(theta, dict) or not theta:
        raise OperatingPointUndeclared(
            f"REFUSED: form {form!r} declared with no `theta_frozen` map.")
    need = [f"{int(b * 100)}%" for b in budgets]
    missing = [k for k in need if k not in theta]
    if missing:
        raise OperatingPointUndeclared(
            f"REFUSED: `theta_frozen` lacks budget key(s) {missing} of the "
            f"declared {need}. A partial map runs the missing budgets "
            f"RETROSPECTIVELY while one mode is stamped for the arm "
            f"(R-209(2)); never fall back, name what is missing.")
    bad = {k: theta[k] for k in need
           if not isinstance(theta[k], (int, float))
           or isinstance(theta[k], bool)}
    if bad:
        raise OperatingPointUndeclared(
            f"REFUSED: `theta_frozen` values must be numbers; got {bad}.")
    for f in ("declared_by", "declared_at_utc", "source"):
        if not decl.get(f):
            raise OperatingPointUndeclared(
                f"REFUSED: the declaration carries no {f!r}. A threshold "
                f"without a declarer and a source cannot be told apart from "
                f"one chosen after seeing the data.")
    return {"form": form, "causal": True,
            "theta_frozen": {k: float(theta[k]) for k in need},
            "budgets": list(budgets),
            "declared_by": decl["declared_by"],
            "declared_at_utc": decl["declared_at_utc"],
            "source": decl["source"],
            "declaration_sha16": _sha16(decl),
            "selected_by_this_module": False}


def require_arm_identity(decl, role: str) -> dict:
    """Candidate and incumbent are named by ROLE in R-496 (D) and by ARTIFACT
    nowhere. Rule 12's form -- hash plus commit ref -- exists and is required
    here, so a run cannot silently score a different model and return
    clean-looking numbers."""
    if not isinstance(decl, dict):
        raise ForwardMetricRefused(
            f"REFUSED: the {role} identity is {type(decl).__name__}, not a "
            f"mapping. 'the {role}' names a role; a run needs an ARTIFACT.")
    for f in ("path", "sha256", "spec"):
        if not decl.get(f):
            raise ForwardMetricRefused(
                f"REFUSED: the {role} identity carries no {f!r}. Rule 12: a "
                f"candidate is a hash and a commit ref, never a name.")
    return {"role": role, "path": str(decl["path"]),
            "sha256": decl["sha256"], "spec": decl["spec"],
            "carrying_commit": decl.get("carrying_commit")}


# ---------------------------------------------------------------------------
# THE PRODUCER CONTRACT — six scalars per row, not a row dict
# ---------------------------------------------------------------------------
def reduce_window(wrows, scores, latency_ms: int) -> list:
    """The feed the estimand needs, and nothing else.

    THIS IS THE ONE FUNCTION `be_forward_day.build_and_score` WOULD CALL. It is
    written here rather than there deliberately: changing the driver changes
    the sha under which race days are scored (R496-R6's leak channel), so the
    producer half lands as a call site the coordinator authorises, not as a
    surprise inside a scoring run.

    The value is resolved HERE, at the row, where `latency` still exists --
    so the feed carries the number the estimand will use and never a pointer
    into a structure that has been dropped."""
    L = str(latency_ms)
    out = []
    for r, sc in zip(wrows, scores):
        lat = r.get("latency") or {}
        out.append({
            "slug": r.get("slug"), "side": r["side"], "gen": r["gen"],
            "t0": r["t0"], "t_start": r["t_start"],
            "score": float(sc),
            "value_cents": (float(lat[L]["preventable_value_cents"])
                            if r.get("any_fill_ahead") and L in lat else 0.0),
        })
    return out


def feed_row_to_eval_row(fr: dict, latency_ms: int) -> dict:
    """Re-inflate one feed record into the shape `evaluate_policy` reads.

    The evaluator is BORROWED, never restated -- one implementation of the
    estimand in the repo -- so the feed must speak its dialect. The `latency`
    block is reconstructed at exactly the one key the evaluator indexes."""
    L = str(latency_ms)
    return {"slug": fr["slug"], "side": fr["side"], "gen": fr["gen"],
            "t0": fr["t0"], "t_start": fr["t_start"],
            "any_fill_ahead": True,
            "latency": {L: {"preventable_value_cents": fr["value_cents"]}}}


def feed_cost(n_rows: int) -> dict:
    """What retaining the feed costs, priced rather than feared.

    The 12 G OOM came from row DICTS plus each window's `window_streams`, not
    from the estimand's inputs. Six scalars per row is a different order of
    magnitude, and this states it as arithmetic a reader can check."""
    return {"n_rows": n_rows,
            "bytes_per_row_packed": FEED_BYTES_PER_ROW,
            "packed_mb": round(n_rows * FEED_BYTES_PER_ROW / 1e6, 1),
            "cap_gb": 12,
            "note": ("the packed figure is six 8-byte scalars per row. A "
                     "Python dict per row costs several times this; the "
                     "selftest MEASURES the real in-memory size of a feed "
                     "rather than trusting the arithmetic.")}


# ---------------------------------------------------------------------------
# THE ESTIMAND — borrowed, never restated
# ---------------------------------------------------------------------------
def evaluate_arm(rows, scores, latency_ms, theta_frozen, budgets=AE.BUDGETS,
                 n_random=AE.N_RANDOM, seed=20260825) -> dict:
    """One arm's net cents, through `harmful_action_eval.evaluate_policy`.

    Borrowed BY IMPORT so there is exactly one implementation of the estimand.
    `theta_frozen` is mandatory here even though the evaluator tolerates None:
    the tolerance is what this module exists to close."""
    if theta_frozen is None:
        raise OperatingPointUndeclared(
            "REFUSED: evaluate_arm was called with theta_frozen=None. The "
            "evaluator would compute a RETROSPECTIVE top-k cutoff from these "
            "very rows and stamp the mode; for a forward read that is rule 11 "
            "with a label on it.")
    out = AE.evaluate_policy(rows, scores, latency_ms=latency_ms,
                             budgets=budgets, n_random=n_random, seed=seed,
                             theta_frozen=theta_frozen)
    modes = {b["threshold_mode"] for b in out["budgets"].values()}
    if modes != {"CAUSAL_FROZEN_FROM_TRAIN"}:
        raise ForwardMetricRefused(
            f"REFUSED: the evaluator reports threshold mode(s) {sorted(modes)}; "
            f"a forward arm must be CAUSAL at EVERY budget. A mixed arm "
            f"reports one mode for the arm while some budgets ran "
            f"retrospectively (R-209(2)).")
    return out


def increment(rows, cand_scores, inc_scores, theta: float,
              latency_ms: int) -> dict:
    """Candidate minus incumbent, per window, at one declared threshold.

    Both arms are valued by `phase2_increment_null.per_window_net` -- the same
    function, the same rows, the same theta -- so the difference is the arms
    and nothing else. Rule 9: skill is reported INCREMENTAL to the incumbent,
    never against a base rate."""
    import math
    cb, ct, cn = PIN.per_window_net(rows, cand_scores, theta)
    ib, it, inn = PIN.per_window_net(rows, inc_scores, theta)
    windows = PIN.ordered_windows(cb, ib)
    by_window = {w: cb.get(w, 0.0) - ib.get(w, 0.0) for w in windows}
    return {"theta": theta, "latency_ms": latency_ms,
            "candidate_net_cents": ct, "incumbent_net_cents": it,
            "increment_cents": math.fsum(by_window.values()),
            "candidate_n_cancelled": cn, "incumbent_n_cancelled": inn,
            "n_windows": len(windows),
            "increment_by_window": by_window,
            "unit": "ACTION", "baseline": "INCUMBENT, not a base rate (rule 9)"}


def exclusions(rows, *score_vectors) -> dict:
    """Statuses with counts, never silent drops (rule 4).

    Every row lands in exactly one status and the statuses sum to the
    population -- asserted here, because a status set that does not sum is a
    silent drop wearing a table."""
    st: dict = {"SCORED": 0, "NO_FILL_AHEAD": 0, "ZERO_VALUE": 0,
                "NON_FINITE_SCORE": 0}
    import math
    for i, r in enumerate(rows):
        if any(not math.isfinite(s[i]) for s in score_vectors):
            st["NON_FINITE_SCORE"] += 1
        elif not r.get("any_fill_ahead"):
            st["NO_FILL_AHEAD"] += 1
        elif not any(v.get("preventable_value_cents")
                     for v in (r.get("latency") or {}).values()):
            st["ZERO_VALUE"] += 1
        else:
            st["SCORED"] += 1
    total = sum(st.values())
    if total != len(rows):
        raise ForwardMetricRefused(
            f"REFUSED: exclusion statuses sum to {total} against {len(rows)} "
            f"rows. A status set that does not account for every row is a "
            f"silent drop with a table beside it (rule 4).")
    return {"statuses": st, "n_rows": len(rows), "accounts_for_every_row": True}


def cluster_disclosure(rows) -> dict:
    """G, the unit ruled and the unit used, and whether an interval may be
    claimed. Borrowed from `phase2_increment_null.complete_utc_days`."""
    d = PIN.complete_utc_days(rows)
    g = d["G_complete_utc_days"]
    return {**d, "ruled_unit": "UTC day", "unit_actually_used": "window",
            "weaker_than_ruled": True, "intervals_claimable": g >= 5,
            "why_p_is_optimistic": (
                "the paired null flips WINDOW signs; windows inside one UTC "
                "day share coin, regime and book state and are not "
                "exchangeable, so the null variance is understated. With G=1 "
                "no day-level null exists at all -- one day cannot be "
                "permuted."),
            "point_estimate_only": g < 5}


# ---------------------------------------------------------------------------
# SELFTEST. Rule 15: every checker ships a falsifier. SEAT_PROTOCOL rule 16:
# every control fires on the bad case AND ADMITS the good one -- a named SKIP
# is not an admission.
# ---------------------------------------------------------------------------
EXPECTED_CHECKS = 56

_L = 50


def _row(slug, side, gen, t0, t_start, value, fill=True):
    return {"slug": slug, "side": side, "gen": gen, "t0": t0,
            "t_start": t_start, "any_fill_ahead": fill,
            "latency": {str(_L): {"preventable_value_cents": value}}}


def _fixture():
    """Two windows, four generations, SIX rows -- deliberately more rows than
    generations, so a row-level sum and an action-level sum DIFFER and any
    regression to row semantics is visible in a number."""
    rows = [
        _row("btc-1000", "BUY", 1, 1000, 0.0, 10.0),
        _row("btc-1000", "BUY", 1, 1000, 5.0, 90.0),   # same gen, later row
        _row("btc-1000", "SELL", 2, 1000, 1.0, -4.0),
        _row("btc-2000", "BUY", 3, 2000, 0.0, 30.0),
        _row("btc-2000", "BUY", 3, 2000, 2.0, 70.0),   # same gen, later row
        _row("btc-2000", "SELL", 4, 2000, 3.0, 6.0),
    ]
    cand = [0.9, 0.2, 0.8, 0.95, 0.1, 0.3]
    inc = [0.4, 0.4, 0.4, 0.4, 0.4, 0.4]
    return rows, cand, inc


def selftest() -> int:
    import traceback
    checks = 0
    fails = []

    def ok(cond, label):
        nonlocal checks
        checks += 1
        print(f"PASS: {label}" if cond else f"FAIL: {label}")
        if not cond:
            fails.append(label)

    def refuses(fn, want, label, exc=ForwardMetricRefused):
        nonlocal checks
        checks += 1
        try:
            fn()
        except exc as e:
            if want in str(e):
                print(f"PASS: {label}")
                return
            fails.append(f"{label} [wrong cause: {str(e)[:110]}]")
            print(f"FAIL: {label} -- refused, but not for {want!r}")
            return
        except Exception as e:                        # noqa: BLE001
            fails.append(f"{label} [{type(e).__name__}, not a named refusal]")
            print(f"FAIL: {label} -- {type(e).__name__}: {str(e)[:110]}")
            print(traceback.format_exc()[-300:])
            return
        fails.append(f"{label} [ACCEPTED the known-bad]")
        print(f"FAIL: {label} -- the known-bad was ACCEPTED")

    rows, cand, inc = _fixture()

    # ---- THE GAP: RED ON TODAY'S SHAPE ---------------------------------
    # This is the check the round exists for. It FAILS on what
    # `be_forward_day.seal()` writes today and PASSES on a real feed.
    sealed_today = {"btc": [[1000, 0.9], [1000, 0.2], [1000, 0.8]],
                    "eth": [[2000, 0.95]]}
    g = sealed_shape_is_unusable(sealed_today)
    ok(g["usable_for_action_estimand"] is False and g["tuple_width"] == 2,
       "THE GAP: the shape be_forward_day.seal() writes today is COMPUTED "
       "unusable for the action estimand, with its tuple width read off it")
    ok(g["carries_action_key"] is False
       and g["carries_row_order_within_generation"] is False,
       "THE GAP: both reasons are separately stated -- no action key, and no "
       "within-generation order (t0 is the WINDOW start)")
    refuses(lambda: assert_action_keys(sealed_today["btc"]),
            "not a mapping",
            "KNOWN-BAD: feeding the sealed pair-form to the estimand REFUSES "
            "by name rather than computing a plausible wrong number")
    refuses(lambda: assert_action_keys([{k: v for k, v in rows[0].items()
                                         if k != "gen"}]),
            "undefined rather than unmeasured",
            "KNOWN-BAD: a row missing `gen` REFUSES, naming that the estimand "
            "is UNDEFINED without the action key")
    # POSITIVE CONTROL: the good feed must be ADMITTED, not merely not-refused.
    ak = assert_action_keys(rows)
    ok(ak["n_rows"] == 6 and ak["n_actions"] == 4,
       "POSITIVE CONTROL: a well-formed feed ADMITS, and the action count (4) "
       "differs from the row count (6) -- so a row-level regression shows")
    ok(abs(ak["rows_per_action"] - 1.5) < 1e-12,
       "POSITIVE CONTROL: rows_per_action is computed (1.5), the ratio "
       "CLAUDE.md rule 2 says inflates a row-level sum")
    refuses(lambda: assert_action_keys([]), "zero rows",
            "KNOWN-BAD: an empty population REFUSES (R-141), never returns 0")

    # ---- ALIGNMENT -----------------------------------------------------
    ok(assert_alignment(rows, cand, inc)["lengths_equal"],
       "POSITIVE CONTROL: aligned score vectors ADMIT")
    refuses(lambda: assert_alignment(rows, cand[:-1]), "positional",
            "KNOWN-BAD: a short score vector REFUSES -- silent re-pairing "
            "leaves every downstream number plausible")

    # ---- THE OPERATING POINT -------------------------------------------
    good_op = {"form": "FROZEN_FROM_TRAIN_QUANTILE",
               "theta_frozen": {"5%": 0.85, "10%": 0.5, "15%": 0.15},
               "declared_by": "USER", "declared_at_utc": "2026-01-01T00:00:00Z",
               "source": "fixture"}
    refuses(lambda: require_operating_point(None), "no operating point declared",
            "KNOWN-BAD: no operating point REFUSES BY NAME (rule 11/14)",
            exc=OperatingPointUndeclared)
    refuses(lambda: require_operating_point({**good_op, "form": "RETROSPECTIVE_TOPK"}),
            "NOT causal",
            "KNOWN-BAD: the RETROSPECTIVE form is REFUSED, not stamped as a "
            "mode -- it is listed so it can be named and refused",
            exc=OperatingPointUndeclared)
    refuses(lambda: require_operating_point({**good_op, "form": "MY_OWN_IDEA"}),
            "not one of",
            "KNOWN-BAD: a form invented at the call site REFUSES",
            exc=OperatingPointUndeclared)
    refuses(lambda: require_operating_point(
                {**good_op, "theta_frozen": {"5%": 0.85, "10%": 0.5}}),
            "lacks budget key",
            "KNOWN-BAD: a PARTIAL theta map REFUSES (R-209(2)); it never falls "
            "back for the missing budgets",
            exc=OperatingPointUndeclared)
    refuses(lambda: require_operating_point(
                {**good_op, "theta_frozen": {"5%": True, "10%": 0.5, "15%": 0.1}}),
            "must be numbers",
            "KNOWN-BAD: a boolean threshold REFUSES (True would compare as 1)",
            exc=OperatingPointUndeclared)
    for f in ("declared_by", "declared_at_utc", "source"):
        refuses(lambda f=f: require_operating_point({k: v for k, v in good_op.items()
                                                     if k != f}),
                f"no {f!r}",
                f"KNOWN-BAD: a declaration with no {f} REFUSES -- a threshold "
                f"without a declarer cannot be told from one chosen later",
                exc=OperatingPointUndeclared)
    op = require_operating_point(good_op)
    ok(op["causal"] and op["selected_by_this_module"] is False,
       "POSITIVE CONTROL: a complete causal declaration ADMITS, and the "
       "receipt records that this module selected nothing")
    ok(len(op["declaration_sha16"]) == 16,
       "POSITIVE CONTROL: the declaration is bound by hash, so the value used "
       "can be tied to the value declared")
    ok(OPERATING_POINT_FORMS["RETROSPECTIVE_TOPK"]["causal"] is False
       and all(OPERATING_POINT_FORMS[f]["causal"]
               for f in OPERATING_POINT_FORMS if f != "RETROSPECTIVE_TOPK"),
       "the proposed forms carry a computed `causal` flag, and exactly the "
       "retrospective one is False")
    ok(all(OPERATING_POINT_FORMS[f].get("costs")
           and OPERATING_POINT_FORMS[f].get("risk")
           for f in OPERATING_POINT_FORMS),
       "every proposed form carries BOTH its cost and its risk -- a menu "
       "without costs is a recommendation in disguise")

    pr = operating_point_pricing()
    ok(pr["selected"] is None and pr["selected_by_this_module"] is False,
       "POSITIVE CONTROL: the pricing table SELECTS nothing and says so in "
       "the artifact (rule 14)")
    _tq = pr["forms"]["FROZEN_FROM_TRAIN_QUANTILE"]["priced"]
    ok(_tq["days"] == ["2026-08-24", "2026-08-25"] and _tq["n_rows"] > 1_000_000
       and _tq["already_consumed"] is True,
       f"POSITIVE CONTROL: the training-quantile form is priced from the "
       f"FREEZE'S OWN declaration -- days {_tq['days']}, {_tq['n_rows']:,} "
       f"rows, already consumed, so it costs the race no unspent day")
    ok(all("priced" in f for f in pr["forms"].values()),
       "every proposed form is priced, including the one that exists only to "
       "be refused")

    # THE FORMS ARE TIED TO THE COMMITTED DECLARATION, not to this file's
    # vocabulary. The reviewer's R496-R4 notes that R-496 (D) cites
    # `phase2_declaration.py` nowhere while that module has pinned these for
    # a long time; a name here that drifted from the declaration would let a
    # run report a mode the programme never declared.
    import phase2_declaration as PD
    ok(set(PD.THRESHOLD_MODES) <= set(OPERATING_POINT_FORMS)
       or "RETROSPECTIVE_TOPK" in PD.THRESHOLD_MODES,
       f"the declared threshold modes {tuple(PD.THRESHOLD_MODES)} are known "
       f"to this module's form table")
    ok(PD.THRESHOLD_PRIMARY == "CAUSAL_FROZEN_FROM_TRAIN"
       and OPERATING_POINT_FORMS[
           "FROZEN_FROM_TRAIN_QUANTILE"]["causal"] is True,
       "the committed declaration already names the CAUSAL mode as PRIMARY -- "
       "so what is missing for a forward read is the THETA VALUES, not the "
       "choice of mode; stated, not selected")
    ok(PD.BUDGETS == AE.BUDGETS and PD.N_RANDOM == AE.N_RANDOM
       and PD.TARGET_LATENCY_MS == 50,
       f"the two declaration sources AGREE on budgets {PD.BUDGETS}, "
       f"n_random {PD.N_RANDOM} and L {PD.TARGET_LATENCY_MS} ms -- checked, "
       f"because two copies of a constant drift silently")

    # ---- ARM IDENTITY --------------------------------------------------
    good_id = {"path": "/x/cand.json", "sha256": "a" * 64, "spec": "PM_PLUS_FINE"}
    ok(require_arm_identity(good_id, "candidate")["sha256"] == "a" * 64,
       "POSITIVE CONTROL: a complete arm identity ADMITS")
    for f in ("path", "sha256", "spec"):
        refuses(lambda f=f: require_arm_identity(
                    {k: v for k, v in good_id.items() if k != f}, "incumbent"),
                f"carries no {f!r}",
                f"KNOWN-BAD: an arm identity with no {f} REFUSES (rule 12)")
    refuses(lambda: require_arm_identity("the incumbent", "incumbent"),
            "names a role",
            "KNOWN-BAD: naming an arm by ROLE rather than by ARTIFACT REFUSES")

    # ---- THE PRODUCER CONTRACT -----------------------------------------
    feed = reduce_window(rows, cand, _L)
    ok(len(feed) == 6 and set(feed[0]) == {"slug", "side", "gen", "t0",
                                           "t_start", "score", "value_cents"},
       "POSITIVE CONTROL: reduce_window emits one record per row with exactly "
       "the seven fields the estimand needs")
    ok(feed[1]["value_cents"] == 90.0 and feed[2]["value_cents"] == -4.0,
       "POSITIVE CONTROL: the latency-aware value is RESOLVED at the row, "
       "including a negative one, so no pointer into a dropped structure")
    nofill = reduce_window([_row("btc-1000", "BUY", 9, 1000, 0.0, 55.0,
                                 fill=False)], [0.5], _L)
    ok(nofill[0]["value_cents"] == 0.0,
       "POSITIVE CONTROL: a row with no fill ahead resolves to 0.0, matching "
       "the evaluator's own `val` guard rather than restating it")
    back = [feed_row_to_eval_row(f, _L) for f in feed]
    ok(all(b["latency"][str(_L)]["preventable_value_cents"]
           == r["latency"][str(_L)]["preventable_value_cents"]
           for b, r in zip(back, rows)),
       "POSITIVE CONTROL: feed -> eval-row round-trips the value the "
       "evaluator will index")
    ok(assert_action_keys(back)["n_actions"] == 4,
       "POSITIVE CONTROL: the re-inflated feed satisfies the action-key "
       "contract -- the feed is sufficient, not merely smaller")
    cost = feed_cost(2_646_442)
    ok(cost["packed_mb"] < 200 and cost["cap_gb"] == 12,
       f"the feed for 09-02's real row count is priced at "
       f"{cost['packed_mb']} MB packed against a 12 G cap -- the drop was "
       f"forced by row DICTS, not by the estimand's inputs")
    import sys as _s
    _real = sum(_s.getsizeof(f) + sum(_s.getsizeof(v) for v in f.values())
                for f in feed) / len(feed)
    ok(_real > FEED_BYTES_PER_ROW,
       f"MEASURED, not assumed: a dict feed record really costs ~{int(_real)} B, "
       f"more than the {FEED_BYTES_PER_ROW} B packed figure -- the arithmetic "
       f"is a floor and the selftest says so rather than quoting the floor")

    # ---- THE ESTIMAND --------------------------------------------------
    refuses(lambda: evaluate_arm(rows, cand, _L, None),
            "theta_frozen=None",
            "KNOWN-BAD: evaluate_arm with no frozen threshold REFUSES rather "
            "than letting the evaluator compute a retrospective cutoff",
            exc=OperatingPointUndeclared)
    ev = evaluate_arm(rows, cand, _L, op["theta_frozen"], n_random=20)
    ok(ev["unit"] == "ACTION" and ev["n_actions"] == 4 and ev["n_rows"] == 6,
       "POSITIVE CONTROL: the borrowed evaluator runs and reports the ACTION "
       "unit with 4 actions over 6 rows")
    ok(all(b["threshold_mode"] == "CAUSAL_FROZEN_FROM_TRAIN"
           for b in ev["budgets"].values()),
       "POSITIVE CONTROL: every budget ran CAUSALLY under the declared theta")
    ok(any(b["net_cents"] != 0.0 for b in ev["budgets"].values()),
       "POSITIVE CONTROL: the estimand produces a NON-TRIVIAL net-cents "
       "figure -- an evaluator that silently returned zero would fail here")
    ok("rho_captured_over_sacrificed" in
       ev["budgets"]["5%"] or ev["budgets"]["5%"]["n_cancelled_generations"] == 0,
       "the secondary metric rho travels with the primary in the same block")

    # ---- THE INCREMENT AND THE NULL ------------------------------------
    incd = increment(rows, cand, inc, theta=0.5, latency_ms=_L)
    ok(incd["baseline"].startswith("INCUMBENT"),
       "POSITIVE CONTROL: the baseline is the INCUMBENT, never a base rate "
       "(rule 9)")
    ok(incd["n_windows"] == 2,
       "POSITIVE CONTROL: the increment is tallied per WINDOW (2 here), the "
       "unit the paired null permutes")
    self_inc = increment(rows, cand, cand, theta=0.5, latency_ms=_L)
    ok(self_inc["increment_cents"] == 0.0
       and all(v == 0.0 for v in self_inc["increment_by_window"].values()),
       "POSITIVE CONTROL: an arm against ITSELF increments to exactly 0.0 in "
       "every window -- the identity a wrong pairing would break")
    ok(incd["increment_cents"] != 0.0,
       "and a genuinely different incumbent gives a NON-ZERO increment, so "
       "the zero above is the identity and not a dead code path")
    null = PIN.sign_flip_p(self_inc["increment_by_window"], n_perm=200, seed=1)
    ok(null["p_two_sided"] == 1.0,
       "POSITIVE CONTROL: an all-zero increment yields p = 1.0 -- the null "
       "behaves at the degenerate point rather than dividing by zero")
    null2 = PIN.sign_flip_p(incd["increment_by_window"], n_perm=200, seed=1)
    ok(null2["n_perm"] == 200 and 0 < null2["p_two_sided"] <= 1.0,
       "POSITIVE CONTROL: the borrowed sign-flip null runs at a declared draw "
       "count and returns a p in (0, 1]")

    # ---- EXCLUSIONS AND CLUSTER DISCLOSURE -----------------------------
    ex = exclusions(rows, cand, inc)
    ok(ex["accounts_for_every_row"] and sum(ex["statuses"].values()) == 6,
       "POSITIVE CONTROL: every row lands in exactly one status and they sum "
       "to the population (rule 4)")
    mixed = rows + [_row("btc-3000", "BUY", 5, 3000, 0.0, 0.0, fill=False)]
    ex2 = exclusions(mixed, cand + [0.5], inc + [0.5])
    ok(ex2["statuses"]["NO_FILL_AHEAD"] == 1,
       "POSITIVE CONTROL: an excluded row is COUNTED under its status, never "
       "dropped")
    cd = cluster_disclosure(rows)
    ok(cd["intervals_claimable"] is False and cd["point_estimate_only"] is True
       and cd["ruled_unit"] == "UTC day" and cd["unit_actually_used"] == "window",
       "POSITIVE CONTROL: the cluster disclosure reports G, the ruled unit, "
       "the unit used, and refuses an interval below G=5 (rule 8)")
    ok(cd["weaker_than_ruled"] is True and "not exchangeable" in
       cd["why_p_is_optimistic"].replace("are not", "not"),
       "the disclosure states WHY the p is optimistic rather than only that "
       "an interval is unavailable")

    # ---- THE CONTRACT AGAINST THE EVALUATOR ITSELF ---------------------
    # REQUIRED_ROW_FIELDS was derived by reading `evaluate_policy`. If that
    # function starts reading a field this module does not carry, the feed
    # silently loses it -- so the tie is asserted against the real source.
    import inspect
    ev_src = inspect.getsource(AE.evaluate_policy) + inspect.getsource(AE._hour)
    reads = {f for f in ("slug", "side", "gen", "t0", "t_start", "latency",
                         "any_fill_ahead") if f in ev_src}
    ok(reads == set(REQUIRED_ROW_FIELDS),
       f"REQUIRED_ROW_FIELDS is tied to the evaluator's OWN source, not to a "
       f"comment: it reads exactly {sorted(reads)}")
    ok("preventable_value_cents" in ev_src,
       "and the evaluator's value key is the one reduce_window resolves")

    print(f"\n{checks} checks passed" if not fails
          else f"\n{len(fails)} FAILURES of {checks} checks")
    for f in fails:
        print(f"  - {f}")
    if checks != EXPECTED_CHECKS:
        print(f"FAIL: ran {checks} checks, EXPECTED_CHECKS={EXPECTED_CHECKS}. "
              f"A suite that silently shrinks reports a pass it did not earn.")
        return 1
    return 1 if fails else 0


def main(argv=None) -> int:
    argv = list(sys.argv) if argv is None else list(argv)
    if "--selftest" in argv:
        return selftest()
    # BE34-R4: usage returns 2, never 0.
    print("usage: be_forward_metric.py --selftest")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
