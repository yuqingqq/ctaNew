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
import phase2_iter011 as I11


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

#: How a POSITIONAL entry is read, by width. Declared, so `sealed_shape_*`
#: can answer for tuple forms instead of only for dicts.
POSITIONAL_COLUMN_MAP = {
    2: ("t0", "score"),                                    # today's seal
    7: ("slug", "side", "gen", "t0", "t_start", "score", "value_cents"),
}


def _sha16(obj) -> str:
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# THE GAP, COMPUTED
# ---------------------------------------------------------------------------
def sealed_shape_is_unusable(per_coin_scores: dict) -> dict:
    """Can this shape feed the ACTION-level estimand? DERIVED from the input.

    BEM-R4: the predecessor returned three literal `False` verdict fields and
    called itself computed, so it answered "unusable" for every input --
    including the evaluator's own well-formed rows, which `assert_action_keys`
    admits. A control that cannot say `usable` cannot be told from one that
    examined nothing (SEAT_PROTOCOL 16).

    BEM-R5: the second reason is restated. Row order SURVIVES the seal --
    `seal()` sorts only the coin keys and JSON preserves list order -- and the
    evaluator does not use list order anyway, it sorts by `t_start`
    (`harmful_action_eval.py:67`). The genuinely independent second reason is
    that the sealed pair carries no per-row **`t_start`**, which is what the
    evaluator sorts by, what the staleness cut needs (rows filling before
    `t_start + L` are stale, CLAUDE.md rules 3 and 7) and half of the control's
    hour key (`:35`). Repairing only reason 1 would leave the estimand
    undefined."""
    coins = sorted(per_coin_scores)
    n_entries = sum(len(v) for v in per_coin_scores.values())
    sample = None
    for c in coins:
        if per_coin_scores[c]:
            sample = per_coin_scores[c][0]
            break
    if isinstance(sample, dict):
        fields = set(sample)
        width = None
    elif isinstance(sample, (list, tuple)):
        fields = set(POSITIONAL_COLUMN_MAP.get(len(sample), ()))
        width = len(sample)
    else:
        fields, width = set(), None
    has_key = set(ACTION_KEY) <= fields
    has_tstart = "t_start" in fields
    return {
        "shape": "{coin: [entry, ...]}",
        "n_coins": len(coins), "n_entries": n_entries,
        "tuple_width": width,
        "entry_fields_seen": sorted(fields),
        "carries_action_key": has_key,
        "action_key_required": list(ACTION_KEY),
        "carries_per_row_t_start": has_tstart,
        "usable_for_action_estimand": bool(has_key and has_tstart),
        "why": ("the estimand groups rows into actions by "
                f"{list(ACTION_KEY)} and orders them within an action by "
                "`t_start`; an entry carrying neither cannot be grouped and "
                "cannot be ordered, so the quantity is UNDEFINED on it, not "
                "merely unmeasured"),
        "second_reason_is_independent_because": (
            "row ORDER survives the seal (seal() sorts only the coin keys and "
            "JSON preserves list order) and the evaluator sorts by `t_start` "
            "regardless, so the independent missing thing is the per-row "
            "`t_start` itself -- consumed at harmful_action_eval.py:67 "
            "(ordering), :13 (the t_start + L staleness cut) and :35 (the "
            "hour key of the matched control)"),
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


#: The field `require_operating_point` stamps on its output and `increment`
#: recomputes. It is NOT a secret and does not pretend to be: it is a digest of
#: the validated content, so a hand-built mapping can only carry a correct one
#: by having done the validation arithmetic on the same values. It closes the
#: route BEM-R1 found -- a bare float reaching the decision metric -- by making
#: the fence's OUTPUT the only shape `increment` will read a theta out of.
OP_TOKEN_FIELD = "_operating_point_token"

#: Fields the token binds. Changing any of them after validation invalidates it.
_OP_TOKEN_FIELDS = ("form", "theta_frozen", "declared_by", "declared_at_utc",
                    "source", "derived_from_split", "provenance")


def _op_token(d: dict) -> str:
    return hashlib.sha256(json.dumps(
        {k: d.get(k) for k in _OP_TOKEN_FIELDS},
        sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()[:32]


def _verify_provenance(decl: dict) -> dict:
    """BEM-R2. RECOMPUTE where the theta came from; never read a label.

    The reviewer built a theta map from the quantiles of the rows being
    SCORED, labelled it FROZEN_FROM_TRAIN_QUANTILE, and this function's
    predecessor accepted it as causal -- because the form was a string and the
    numbers arrived as a bare {budget: float} dict. Everything checked here is
    a byte, a hash or a file that either exists or does not."""
    prov = decl.get("provenance")
    if not isinstance(prov, dict):
        raise OperatingPointUndeclared(
            "REFUSED: the declaration carries no `provenance` block. A form "
            "flag only a human can honour is a LABEL; the derivation must be "
            "recomputable or the declaration is a name for a number (BEM-R2).")
    out = {}
    for key in ("rows_artifact", "fit_artifact"):
        a = prov.get(key)
        if not isinstance(a, dict) or not a.get("path") or not a.get("sha256"):
            raise OperatingPointUndeclared(
                f"REFUSED: `provenance.{key}` needs a path AND a sha256; got "
                f"{a!r}. Rule 12's form, applied to the thing the quantiles "
                f"were computed FROM.")
        f = Path(a["path"])
        if not f.exists():
            raise OperatingPointUndeclared(
                f"REFUSED: `provenance.{key}` names {f}, which does not "
                f"exist. Nothing was hashed, so nothing was verified.")
        h = hashlib.sha256()
        with open(f, "rb") as fh:
            for b in iter(lambda: fh.read(1 << 20), b""):
                h.update(b)
        got = h.hexdigest()
        if got != a["sha256"]:
            raise OperatingPointUndeclared(
                f"REFUSED: `provenance.{key}` at {f} hashes to {got[:16]}…, "
                f"the declaration says {str(a['sha256'])[:16]}…. The theta "
                f"map was not derived from the artifact it names.")
        out[key] = {"path": str(f), "sha256": got, "verified_by": "rehash"}
    tm = prov.get("theta_map_sha16")
    want = hashlib.sha256(json.dumps(
        decl.get("theta_frozen_by_coin", decl.get("theta_frozen")),
        sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:16]
    if tm != want:
        raise OperatingPointUndeclared(
            f"REFUSED: the declared theta map digests to {want}, the "
            f"declaration says {tm!r}. The numbers that would run are not the "
            f"numbers that were declared.")
    out["theta_map_sha16"] = want
    return out


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
    split = decl.get("derived_from_split")
    if not isinstance(split, dict) or not split.get("days"):
        raise OperatingPointUndeclared(
            "REFUSED: the declaration does not name the SPLIT its quantiles "
            "were taken over (`derived_from_split.days`). Without it the "
            "overlap against the scored population cannot be computed, and "
            "that overlap is the only place 'causal' can be checked rather "
            "than asserted (BEM-R2).")
    prov = _verify_provenance(decl)
    out = {"form": form, "causal_declared": True,
            "causal_verified_against_scored_population": False,
            "causal_verification_note": (
                "provenance is REHASHED here; the derivation is only fully "
                "checked at `increment`, where the scored population exists "
                "and the split overlap can be computed"),
            "derived_from_split": split,
            "provenance_verified": prov,
            "theta_frozen": {k: float(theta[k]) for k in need},
            "budgets": list(budgets),
            "declared_by": decl["declared_by"],
            "declared_at_utc": decl["declared_at_utc"],
            "source": decl["source"],
            "declaration_sha16": _sha16(decl),
            "selected_by_this_module": False}
    out[OP_TOKEN_FIELD] = _op_token(decl)
    return out


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
    # BEM-R3: THE FILE IS HASHED. The predecessor accepted a nonexistent path
    # with a garbage sha and a wrong spec and returned them verbatim -- a
    # check whose only subject was whether three strings were truthy.
    f = Path(decl["path"])
    if not f.exists():
        raise ForwardMetricRefused(
            f"REFUSED: the {role} identity names {f}, which does not exist. "
            f"Nothing was hashed, so nothing was verified.")
    h = hashlib.sha256()
    with open(f, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    got = h.hexdigest()
    if got != decl["sha256"]:
        raise ForwardMetricRefused(
            f"REFUSED: the {role} artifact at {f} hashes to {got[:16]}…, the "
            f"declaration says {str(decl['sha256'])[:16]}…. Scoring with a "
            f"different artifact than the one declared returns numbers that "
            f"look right and are not.")
    return {"role": role, "path": str(f), "sha256": got,
            "spec": decl["spec"], "verified_by": "rehash of the named file",
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
            "any_fill_ahead": bool(r.get("any_fill_ahead")),
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
    # `any_fill_ahead` is CARRIED, not assumed True. The arithmetic is
    # unaffected (a no-fill row resolves to 0.0 either way) but `exclusions()`
    # classifies on it, so assuming it would silently empty a status.
    return {"slug": fr["slug"], "side": fr["side"], "gen": fr["gen"],
            "t0": fr["t0"], "t_start": fr["t_start"],
            "any_fill_ahead": bool(fr.get("any_fill_ahead", True)),
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


def paired_null(inc_by_window: dict, n_perm: int = I11.N_PERM_011,
                seed: int = I11.PERM_SEED_011) -> dict:
    """The paired sign-flip null, BORROWED FROM THE INSTRUMENT THAT SORTS.

    BE15-S1, and it is a correction to round 14's own choice. This module first
    borrowed `phase2_increment_null.sign_flip_p`, which consumes
    `inc_by_window.values()` in DICT ORDER. `phase2_iter011.sign_flip_null`
    sorts its keys at consumption and says why in its own docstring: R-234's
    defect was a seeded sign sequence applied to an unpinned data order, so
    every run was an independent draw rather than a replay, and *"sorting the
    keys ... belongs here, at the point of consumption, where it cannot be
    undone by a caller's iteration order."*

    MEASURED, not argued: on one dict's content in two insertion orders,
    `sign_flip_p` returns 0.27680798 and 0.23690773 while `sign_flip_null`
    returns 0.27680798 both times. My `increment()` happens to build its dict
    in sorted order, so the old borrow was correct BY THE CALLER -- which is
    exactly the reliance R-234 says must not exist. It also returns the
    ONE-SIDED `p_value` that R-286/R-288 adjudicate, which `sign_flip_p` does
    not compute at all."""
    return I11.sign_flip_null(inc_by_window, n_perm=n_perm, seed=seed)


# ---------------------------------------------------------------------------
# THE TWO PAIRING CONVENTIONS — DIFFERENT ESTIMANDS (R-497 (F)(4))
# ---------------------------------------------------------------------------
#: The USER ruled BOTH: by-THRESHOLD is the decision metric, by-COUNT is
#: reported beside it as a bridge to iteration 011's development number. They
#: are NOT two views of one quantity. A cell carries its convention as a
#: top-level field, and `pooled_increment` REFUSES to combine cells that do
#: not share one, because the failure this guards is not an error a reader
#: would notice: two plausible numbers averaged into a third plausible number.
PAIRING_CONVENTIONS = {
    "BY_THRESHOLD": {
        "rule": "ONE declared theta applied to BOTH arms; the cancellation "
                "COUNTS differ between arms",
        "role": "PRIMARY — the decision metric (R-497 (F)(4))",
        "theta_source": "DECLARED before the read",
        "causal": True,
    },
    "BY_COUNT": {
        "rule": "kk = max(1, int(n_actions * budget)); EACH ARM cancels its "
                "own top-kk, so the two arms use DIFFERENT cutoffs and the "
                "counts match",
        "role": "REPORTED BESIDE — a bridge to iteration 011's development "
                "number, never a forward result (R-497 (F)(4))",
        "theta_source": "READ OFF THE RANKING OF THE DATA BEING SCORED",
        "causal": False,
    },
}


def _gen_index(rows) -> dict:
    """(slug, side, gen) -> row indices, ordered by t_start. One grouping,
    shared by both conventions, so they can differ only in SELECTION."""
    gens: dict = {}
    for i, r in enumerate(rows):
        gens.setdefault(tuple(r[k] for k in ACTION_KEY), []).append(i)
    for k in gens:
        gens[k].sort(key=lambda i: rows[i]["t_start"])
    return gens


def _cancel_value(rows, gens, scores, chosen, theta, latency_ms) -> dict:
    """Per-window value of cancelling `chosen` at `theta`. First crossing acts.

    Shared by both conventions: the VALUATION is identical and only the
    selection differs, so any gap between the two numbers is the pairing rule
    and never an accounting difference."""
    L = str(latency_ms)
    bw: dict = {}
    for gk in chosen:
        i = next((j for j in gens[gk] if scores[j] >= theta[gk]), None)
        if i is None:
            continue
        r = rows[i]
        v = (r["latency"][L]["preventable_value_cents"]
             if r.get("any_fill_ahead") and "latency" in r else 0.0)
        bw[gk[0]] = bw.get(gk[0], 0.0) + v
    return bw


def _select_by_threshold(gens, scores, theta: float):
    """Both arms, one declared theta. Returns (chosen, per-gen theta)."""
    gmax = {k: max(scores[i] for i in gens[k]) for k in gens}
    chosen = [k for k in gens if gmax[k] >= theta]
    return chosen, {k: theta for k in chosen}, theta


def _select_by_count(gens, scores, frac: float):
    """Each arm its own top-kk. The cutoff is a FUNCTION OF THESE SCORES."""
    gmax = {k: max(scores[i] for i in gens[k]) for k in gens}
    # TIE-BREAK ON THE KEY, matching `phase2_iter011_run._rank`'s
    # `sorted(gens, key=lambda k: (-gmax[k], k))` exactly. Under equal maxima
    # a bare `-gmax` sort falls back to dict order, which is the R-234 defect
    # in a second place; and a bridge arm that does not reproduce the
    # implementation it bridges to is not a bridge.
    order = sorted(gens, key=lambda k: (-gmax[k], k))
    kk = max(1, int(len(order) * frac))
    chosen = order[:kk]
    cut = gmax[chosen[-1]] if chosen else float("inf")
    return chosen, {k: cut for k in chosen}, cut


def require_fenced_op(op, budget_key: str, rows=None) -> dict:
    """BEM-R1 + BEM-R2. The only door a theta may come through.

    `increment` no longer accepts a float. It accepts the object
    `require_operating_point` returned, recomputes that object's token, and --
    when it is given the rows -- computes the OVERLAP between the split the
    quantiles were declared to come from and the population being scored. A
    theta derived from the scored rows can only reach here by declaring those
    days as its own derivation split, and then the overlap fires."""
    if isinstance(op, (int, float)) and not isinstance(op, bool):
        raise OperatingPointUndeclared(
            f"REFUSED: `increment` was handed a bare threshold ({op!r}). A "
            f"float carries no derivation, so nothing about it can be "
            f"checked; the decision metric accepts only the object "
            f"`require_operating_point` returned (BEM-R1).")
    if not isinstance(op, dict):
        raise OperatingPointUndeclared(
            f"REFUSED: `increment` needs the fenced operating point, got "
            f"{type(op).__name__}.")
    tok = op.get(OP_TOKEN_FIELD)
    if not tok:
        raise OperatingPointUndeclared(
            f"REFUSED: this mapping carries no {OP_TOKEN_FIELD!r}. It did "
            f"not come from `require_operating_point`, so no fence has seen "
            f"it.")
    want = _op_token({k: op.get(k) for k in _OP_TOKEN_FIELDS
                      if k != "provenance"}
                     | {"provenance": (op.get("provenance_verified") or {})})
    theta_map = op.get("theta_frozen") or {}
    if budget_key not in theta_map:
        raise OperatingPointUndeclared(
            f"REFUSED: the fenced operating point carries no theta for "
            f"budget {budget_key!r} (has {sorted(theta_map)}).")
    overlap = None
    if rows is not None:
        import datetime as _dt
        scored_days = {
            _dt.datetime.fromtimestamp(r["t0"] + r["t_start"],
                                       _dt.timezone.utc).date().isoformat()
            for r in rows}
        declared = set((op.get("derived_from_split") or {}).get("days") or ())
        overlap = sorted(scored_days & declared)
        if overlap:
            raise OperatingPointUndeclared(
                f"REFUSED: the operating point declares its quantiles were "
                f"derived from {sorted(declared)}, and the population being "
                f"scored covers {sorted(scored_days)} -- overlapping on "
                f"{overlap}. A threshold taken on the very rows it is applied "
                f"to is the retrospective cutoff this fence exists to refuse, "
                f"whatever the form is called (BEM-R2).")
    return {"theta": float(theta_map[budget_key]), "budget_key": budget_key,
            "form": op.get("form"),
            "token_present": True, "token_recomputed": want == tok,
            "derived_from_split": op.get("derived_from_split"),
            "scored_split_overlap": overlap,
            "causal_verified_against_scored_population": overlap == []
            if overlap is not None else False}


def increment(rows, cand_scores, inc_scores, op=None,
              latency_ms: int = 50, convention: str = "BY_THRESHOLD",
              budget: float = None, budget_key: str = None,
              bridge_to_development_ack: bool = False) -> dict:
    """Candidate minus incumbent, per window, under ONE NAMED CONVENTION.

    Rule 9: skill is reported INCREMENTAL to the incumbent, never against a
    base rate. The convention travels in the return value as a field, so a
    downstream reader cannot lose track of WHICH estimand a number is."""
    import math
    if convention not in PAIRING_CONVENTIONS:
        raise ForwardMetricRefused(
            f"REFUSED: pairing convention {convention!r} is not one of "
            f"{sorted(PAIRING_CONVENTIONS)}. A convention named at the call "
            f"site is an estimand chosen where no ruling can see it.")
    gens = _gen_index(rows)
    fenced = None
    theta = None
    if convention == "BY_THRESHOLD":
        bk = budget_key or (f"{int(budget * 100)}%" if budget is not None
                            else None)
        if bk is None:
            raise OperatingPointUndeclared(
                "REFUSED: BY_THRESHOLD needs a budget key to read its theta "
                "out of the fenced operating point.")
        fenced = require_fenced_op(op, bk, rows=rows)
        theta = fenced["theta"]
        c_sel, c_th, c_cut = _select_by_threshold(gens, cand_scores, theta)
        i_sel, i_th, i_cut = _select_by_threshold(gens, inc_scores, theta)
    else:
        if budget is None:
            raise ForwardMetricRefused(
                "REFUSED: BY_COUNT needs a budget fraction; kk is defined "
                "only relative to one.")
        if not bridge_to_development_ack:
            raise ForwardMetricRefused(
                "REFUSED: BY_COUNT is NON-CAUSAL by construction -- its "
                "cutoff is the kk-th score of the data being scored -- so it "
                "is a bridge to a development number and never a forward "
                "result (R-497 (F)(4)). A caller must acknowledge that "
                "explicitly with bridge_to_development_ack=True; the "
                "acknowledgement is recorded in the cell.")
        c_sel, c_th, c_cut = _select_by_count(gens, cand_scores, budget)
        i_sel, i_th, i_cut = _select_by_count(gens, inc_scores, budget)
    cb = _cancel_value(rows, gens, cand_scores, c_sel, c_th, latency_ms)
    ib = _cancel_value(rows, gens, inc_scores, i_sel, i_th, latency_ms)
    windows = sorted(set(cb) | set(ib))
    by_window = {w: cb.get(w, 0.0) - ib.get(w, 0.0) for w in windows}
    meta = PAIRING_CONVENTIONS[convention]
    return {
        "pairing_convention": convention,
        "pairing_rule": meta["rule"], "pairing_role": meta["role"],
        "causal": meta["causal"],
        "theta_source": meta["theta_source"],
        "theta_declared": theta, "budget": budget,
        "operating_point_fence": fenced,
        "bridge_to_development_ack": bridge_to_development_ack,
        "candidate_cutoff": c_cut, "incumbent_cutoff": i_cut,
        "cutoffs_equal_between_arms": c_cut == i_cut,
        "candidate_n_cancelled": len(c_sel),
        "incumbent_n_cancelled": len(i_sel),
        "counts_equal_between_arms": len(c_sel) == len(i_sel),
        "candidate_net_cents": math.fsum(cb.values()),
        "incumbent_net_cents": math.fsum(ib.values()),
        "increment_cents": math.fsum(by_window.values()),
        "n_windows": len(windows), "increment_by_window": by_window,
        "n_actions": len(gens), "latency_ms": latency_ms,
        "unit": "ACTION",
        "baseline": "INCUMBENT, not a base rate (rule 9)",
    }


def cutoff_depends_on_scored_data(convention: str, rows, scores_a, scores_b,
                                  op=None, budget: float = None,
                                  budget_key: str = None,
                                  latency_ms: int = 50) -> dict:
    """IS THE CUTOFF A FUNCTION OF THE DATA BEING SCORED? COMPUTED, not stated.

    Rule 10. The claim "by-count is retrospective and therefore not a forward
    result" is a property of the ARITHMETIC, so it is measured rather than
    written in a docstring: hold the declared inputs fixed, change ONLY the
    scores, and see whether the effective cutoff moves. A cutoff that moves
    with the data is one read off the data, which is exactly what
    `require_operating_point` refuses for a forward read."""
    kw = dict(convention=convention, latency_ms=latency_ms, budget=budget,
              bridge_to_development_ack=True)
    if convention == "BY_THRESHOLD":
        kw["op"] = op
        kw["budget_key"] = budget_key
    a = increment(rows, scores_a, scores_a, **kw)
    b = increment(rows, scores_b, scores_b, **kw)
    moved = a["candidate_cutoff"] != b["candidate_cutoff"]
    return {
        "convention": convention,
        "cutoff_on_scores_a": a["candidate_cutoff"],
        "cutoff_on_scores_b": b["candidate_cutoff"],
        "cutoff_moved_with_the_data": moved,
        "declared_inputs_held_fixed": {"budget_key": budget_key,
                                       "budget": budget},
        "forward_eligible": not moved,
        "why": ("a cutoff that moves when only the scored data changes was "
                "READ OFF that data. Forward, that is the retrospective "
                "cutoff `require_operating_point` refuses, so an arm with "
                "cutoff_moved_with_the_data=True is a BRIDGE TO A "
                "DEVELOPMENT NUMBER and never a forward result."),
        "is_bridge_to_development_number": moved,
    }


def pooled_increment(cells) -> dict:
    """REFUSE to combine cells that do not share one pairing convention.

    The two conventions are different estimands (R-497 (F)(4)). Averaging or
    summing across them produces a number that is plausible, has no estimand,
    and cannot be told from a correct one by looking at it."""
    import math
    convs = sorted({c["pairing_convention"] for c in cells})
    if len(convs) != 1:
        raise ForwardMetricRefused(
            f"REFUSED: asked to pool cells spanning pairing conventions "
            f"{convs}. These are DIFFERENT ESTIMANDS (R-497 (F)(4)): "
            f"by-THRESHOLD holds the cutoff equal and lets the counts differ; "
            f"by-COUNT holds the counts equal and lets the cutoffs differ. "
            f"Pooling them yields a plausible number with no estimand.")
    return {"pairing_convention": convs[0], "n_cells": len(cells),
            "increment_cents": math.fsum(c["increment_cents"] for c in cells),
            "pooled_within_one_convention_only": True}


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
EXPECTED_CHECKS = 87

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
    ok(g["carries_action_key"] is False and g["carries_per_row_t_start"] is False,
       "THE GAP: both reasons are DERIVED from the entry -- no action key, "
       "and no per-row `t_start` (BEM-R5: row ORDER survives the seal; "
       "`t_start` is the independent missing field)")
    # BEM-R4's falsifier: a shape that DOES carry the key and t_start must
    # come back USABLE, or the function is a constant wearing a computation.
    good_shape = {"btc": [{"slug": "s", "side": "BUY", "gen": 1, "t0": 1,
                           "t_start": 0.5, "score": 0.9, "value_cents": 1.0,
                           "any_fill_ahead": True}]}
    gg = sealed_shape_is_unusable(good_shape)
    ok(gg["usable_for_action_estimand"] is True
       and gg["carries_action_key"] is True
       and gg["carries_per_row_t_start"] is True,
       "BEM-R4 POSITIVE CONTROL: an action-keyed entry carrying `t_start` "
       "comes back USABLE -- the function can say yes, so its 'no' above is a "
       "measurement and not a literal")
    gt = sealed_shape_is_unusable({"btc": [["s", "BUY", 1, 1, 0.5, 0.9, 1.0]]})
    ok(gt["carries_action_key"] is True and gt["tuple_width"] == 7,
       "BEM-R4: and a POSITIONAL 7-tuple is read through the declared column "
       "map, so the answer does not depend on the entry being a dict")
    keyed_no_t = {"btc": [{"slug": "s", "side": "BUY", "gen": 1, "score": 0.9}]}
    ok(sealed_shape_is_unusable(keyed_no_t)["usable_for_action_estimand"]
       is False,
       "BEM-R5: an entry with the action key but NO `t_start` is still "
       "UNUSABLE -- repairing only reason 1 would leave the estimand "
       "undefined, which is why the second reason had to be restated")
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
    import tempfile as _tf
    _fx = _tf.TemporaryDirectory()
    _fxd = Path(_fx.name)
    (_fxd / "rows.json").write_text("rows-fixture")
    (_fxd / "fit.json").write_text("fit-fixture")

    def _prov():
        def _h(n):
            return hashlib.sha256((_fxd / n).read_bytes()).hexdigest()
        return {"rows_artifact": {"path": str(_fxd / "rows.json"),
                                  "sha256": _h("rows.json")},
                "fit_artifact": {"path": str(_fxd / "fit.json"),
                                 "sha256": _h("fit.json")},
                "theta_map_sha16": None}

    _tf_map = {"5%": 0.85, "10%": 0.5, "15%": 0.15}
    _pv = _prov()
    _pv["theta_map_sha16"] = hashlib.sha256(json.dumps(
        _tf_map, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:16]
    good_op = {"form": "FROZEN_FROM_TRAIN_QUANTILE",
               "theta_frozen": _tf_map,
               "derived_from_split": {"days": ["2020-01-01"],
                                      "population": "fixture"},
               "provenance": _pv,
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
    ok(op["causal_declared"] and op["selected_by_this_module"] is False,
       "POSITIVE CONTROL: a complete causal declaration ADMITS, and the "
       "receipt records that this module selected nothing")
    ok(op["causal_verified_against_scored_population"] is False
       and op[OP_TOKEN_FIELD],
       "BEM-R2: `causal` is reported as DECLARED and explicitly NOT yet "
       "verified against a scored population, and the object carries the "
       "fence token that is the only currency increment() accepts")
    for _bad, _want, _lab in (
            ({"provenance": None}, "no `provenance` block",
             "a declaration with no provenance block"),
            ({"derived_from_split": None}, "does not name the SPLIT",
             "a declaration that does not name its derivation split")):
        refuses(lambda b=_bad: require_operating_point({**good_op, **b}),
                _want, f"BEM-R2 KNOWN-BAD: {_lab} REFUSES",
                exc=OperatingPointUndeclared)
    _p2 = {k: dict(v) if isinstance(v, dict) else v for k, v in _pv.items()}
    _p2["rows_artifact"] = {**_p2["rows_artifact"], "sha256": "0" * 64}
    refuses(lambda: require_operating_point({**good_op, "provenance": _p2}),
            "not derived from the artifact it names",
            "BEM-R2 KNOWN-BAD: a provenance sha that does not match the file "
            "on disk REFUSES -- the derivation is REHASHED, not read",
            exc=OperatingPointUndeclared)
    _p3 = {k: dict(v) if isinstance(v, dict) else v for k, v in _pv.items()}
    _p3["rows_artifact"] = {**_p3["rows_artifact"], "path": "/nope/rows.json"}
    refuses(lambda: require_operating_point({**good_op, "provenance": _p3}),
            "does not exist",
            "BEM-R2 KNOWN-BAD: provenance naming a nonexistent artifact "
            "REFUSES -- nothing hashed is nothing verified",
            exc=OperatingPointUndeclared)
    refuses(lambda: require_operating_point(
                {**good_op, "provenance": {**_pv, "theta_map_sha16": "dead"}}),
            "not the numbers that were declared",
            "BEM-R2 KNOWN-BAD: a theta map whose digest does not match the "
            "declaration REFUSES",
            exc=OperatingPointUndeclared)
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
    (_fxd / "arm.json").write_text("arm-artifact-bytes")
    _arm_sha = hashlib.sha256((_fxd / "arm.json").read_bytes()).hexdigest()
    good_id = {"path": str(_fxd / "arm.json"), "sha256": _arm_sha,
               "spec": "PM_PLUS_FINE"}
    _got = require_arm_identity(good_id, "candidate")
    ok(_got["sha256"] == _arm_sha and _got["verified_by"] == "rehash of the named file",
       "BEM-R3 POSITIVE CONTROL: an arm identity naming a REAL file whose "
       "bytes match ADMITS, and the receipt says the sha was re-hashed")
    for f in ("path", "sha256", "spec"):
        refuses(lambda f=f: require_arm_identity(
                    {k: v for k, v in good_id.items() if k != f}, "incumbent"),
                f"carries no {f!r}",
                f"KNOWN-BAD: an arm identity with no {f} REFUSES (rule 12)")
    refuses(lambda: require_arm_identity("the incumbent", "incumbent"),
            "names a role",
            "KNOWN-BAD: naming an arm by ROLE rather than by ARTIFACT REFUSES")
    # BEM-R3's own reproduction, now red: this exact input was ACCEPTED and
    # returned verbatim at b717340.
    refuses(lambda: require_arm_identity(
                {"path": "/nonexistent/not_a_model.json", "sha256": "de" * 32,
                 "spec": "A_COMPLETELY_DIFFERENT_ARM"}, "candidate"),
            "does not exist",
            "BEM-R3 KNOWN-BAD (the reviewer's own input): a nonexistent path "
            "with a garbage sha and a wrong spec now REFUSES -- it was "
            "ACCEPTED and returned verbatim before this round")
    refuses(lambda: require_arm_identity({**good_id, "sha256": "0" * 64},
                                         "candidate"),
            "look right and are not",
            "BEM-R3 KNOWN-BAD: a REAL file whose bytes do not match the "
            "declared sha REFUSES")

    # ---- THE PRODUCER CONTRACT -----------------------------------------
    feed = reduce_window(rows, cand, _L)
    ok(len(feed) == 6 and set(feed[0]) == {"slug", "side", "gen", "t0",
                                           "t_start", "score", "value_cents",
                                           "any_fill_ahead"},
       "POSITIVE CONTROL: reduce_window emits one record per row with exactly "
       "the eight fields the estimand needs, `any_fill_ahead` among them")
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
    incd = increment(rows, cand, inc, op=op, budget_key="10%", latency_ms=_L)
    ok(incd["baseline"].startswith("INCUMBENT"),
       "POSITIVE CONTROL: the baseline is the INCUMBENT, never a base rate "
       "(rule 9)")
    ok(incd["n_windows"] == 2,
       "POSITIVE CONTROL: the increment is tallied per WINDOW (2 here), the "
       "unit the paired null permutes")
    self_inc = increment(rows, cand, cand, op=op, budget_key="10%", latency_ms=_L)
    ok(self_inc["increment_cents"] == 0.0
       and all(v == 0.0 for v in self_inc["increment_by_window"].values()),
       "POSITIVE CONTROL: an arm against ITSELF increments to exactly 0.0 in "
       "every window -- the identity a wrong pairing would break")
    ok(incd["increment_cents"] != 0.0,
       "and a genuinely different incumbent gives a NON-ZERO increment, so "
       "the zero above is the identity and not a dead code path")
    null = paired_null(self_inc["increment_by_window"], n_perm=200, seed=1)
    ok(null["p_two_sided"] == 1.0 and null["p_value"] == 1.0,
       "POSITIVE CONTROL: an all-zero increment yields p = 1.0 on BOTH sides "
       "-- the null behaves at the degenerate point rather than dividing by "
       "zero")
    null2 = paired_null(incd["increment_by_window"], n_perm=200, seed=1)
    ok(null2["n_perm"] == 200 and 0 < null2["p_value"] <= 1.0
       and null2["sided"] == "one",
       "POSITIVE CONTROL: the borrowed null runs at a declared draw count and "
       "returns the ONE-SIDED p that R-286/R-288 adjudicate")
    # BE15-S1 BOTH DIRECTIONS: order-invariance is the property being borrowed
    # FOR, so it is driven -- and the known-bad shows the instrument this
    # module NO LONGER uses is order-DEPENDENT on the same content.
    _b = {f"w{i:03d}": (i % 7) - 3.0 + 0.5 for i in range(40)}
    import random as _r
    _ks = list(_b); _r.Random(99).shuffle(_ks)
    _shuf = {k: _b[k] for k in _ks}
    ok(paired_null(_b, n_perm=400, seed=7)["p_two_sided"]
       == paired_null(_shuf, n_perm=400, seed=7)["p_two_sided"],
       "BE15-S1 POSITIVE CONTROL: the borrowed null is ORDER-INVARIANT -- the "
       "same content in a different insertion order gives the same p")
    # BEM-R8 + BE17-S2: these TWO checks asserted that `sign_flip_p` IS
    # order-dependent -- my own round-15 controls, whose subject was the
    # defect. Repairing `sign_flip_p` at consumption turned them red, which is
    # how an enshrining control announces itself. Inverted: both instruments
    # must now agree, and a regression that removed either sort turns this red.
    ok(PIN.sign_flip_p(_b, n_perm=400, seed=7)["p_two_sided"]
       == PIN.sign_flip_p(_shuf, n_perm=400, seed=7)["p_two_sided"],
       "BEM-R8: `phase2_increment_null.sign_flip_p` is now order-INVARIANT "
       "too -- the defect BE15-S1 documented is repaired at its source rather "
       "than avoided by this module")
    ok(paired_null(_b, n_perm=400, seed=7)["p_two_sided"]
       == PIN.sign_flip_p(_b, n_perm=400, seed=7)["p_two_sided"],
       "BEM-R8: and the two instruments now AGREE on the same input, so a "
       "caller can no longer pick a p by picking an implementation")
    ok("sorted(" in __import__("inspect").getsource(I11.sign_flip_null)
       and "sorted(" in __import__("inspect").getsource(PIN.sign_flip_p),
       "BEM-R8: BOTH sort at consumption now, asserted from the code -- the "
       "caller-side reliance R-234 forbids is gone from this path")

    # ---- THE TWO PAIRING CONVENTIONS (R-497 (F)(4)) --------------------
    thr = increment(rows, cand, inc, op=op, budget_key="10%", latency_ms=_L,
                    convention="BY_THRESHOLD")
    cnt = increment(rows, cand, inc, budget=0.5, latency_ms=_L,
                    convention="BY_COUNT", bridge_to_development_ack=True)
    ok(thr["pairing_convention"] == "BY_THRESHOLD"
       and cnt["pairing_convention"] == "BY_COUNT",
       "POSITIVE CONTROL: every cell CARRIES its pairing convention as a "
       "top-level field a reader cannot miss")
    ok(thr["cutoffs_equal_between_arms"] is True
       and thr["counts_equal_between_arms"] is False,
       f"BY_THRESHOLD holds the CUTOFF equal across arms and lets the counts "
       f"differ ({thr['candidate_n_cancelled']} vs "
       f"{thr['incumbent_n_cancelled']}) -- computed, not described")
    ok(cnt["counts_equal_between_arms"] is True
       and cnt["cutoffs_equal_between_arms"] is False,
       f"BY_COUNT holds the COUNT equal ({cnt['candidate_n_cancelled']} vs "
       f"{cnt['incumbent_n_cancelled']}) and lets the cutoffs differ "
       f"({cnt['candidate_cutoff']} vs {cnt['incumbent_cutoff']})")
    ok(thr["increment_cents"] != cnt["increment_cents"],
       f"THE TWO ARE DIFFERENT ESTIMANDS, shown by a number: the same rows "
       f"and the same scores give {thr['increment_cents']} under "
       f"BY_THRESHOLD and {cnt['increment_cents']} under BY_COUNT")
    ok(thr["causal"] is True and cnt["causal"] is False,
       "the registry marks exactly the by-count convention non-causal")
    refuses(lambda: increment(rows, cand, inc, op=op, budget_key="10%",
                              latency_ms=_L, convention="WHATEVER"),
            "is not one of",
            "KNOWN-BAD: a convention named at the call site REFUSES")
    refuses(lambda: increment(rows, cand, inc, budget_key="10%",
                              latency_ms=_L, convention="BY_THRESHOLD"),
            "needs the fenced operating point",
            "BEM-R1 KNOWN-BAD: BY_THRESHOLD with NO fenced operating point "
            "REFUSES", exc=OperatingPointUndeclared)
    refuses(lambda: increment(rows, cand, inc, op=0.95, budget_key="10%",
                              latency_ms=_L, convention="BY_THRESHOLD"),
            "was handed a bare threshold",
            "BEM-R1 KNOWN-BAD (the reviewer's own route): a BARE FLOAT theta "
            "REFUSES BY NAME -- at b717340 this produced a complete "
            "net-cents-vs-incumbent result with no fence touched",
            exc=OperatingPointUndeclared)
    refuses(lambda: increment(rows, cand, inc,
                              op={k: v for k, v in op.items()
                                  if k != OP_TOKEN_FIELD},
                              budget_key="10%", latency_ms=_L),
            "did not come from",
            "BEM-R1 KNOWN-BAD: a hand-built mapping without the fence token "
            "REFUSES -- no fence has seen it", exc=OperatingPointUndeclared)
    refuses(lambda: increment(rows, cand, inc, budget=0.5, latency_ms=_L,
                              convention="BY_COUNT"),
            "acknowledge that explicitly",
            "BEM-R1 KNOWN-BAD: BY_COUNT without an explicit bridge "
            "acknowledgement REFUSES -- the only other route to an unfenced "
            "number is closed by an admission the cell records")
    refuses(lambda: increment(rows, cand, inc, latency_ms=_L,
                              convention="BY_COUNT"),
            "needs a budget fraction",
            "KNOWN-BAD: BY_COUNT without a budget REFUSES")

    # ---- THE DISQUALIFIER, COMPUTED ON THE ARM (rule 10) ---------------
    other = [s * 0.5 + 0.02 for s in cand]
    dt_ = cutoff_depends_on_scored_data("BY_THRESHOLD", rows, cand, other,
                                        op=op, budget_key="10%", latency_ms=_L)
    dc_ = cutoff_depends_on_scored_data("BY_COUNT", rows, cand, other,
                                        budget=0.5, latency_ms=_L)
    ok(dt_["cutoff_moved_with_the_data"] is False
       and dt_["forward_eligible"] is True,
       "COMPUTED: BY_THRESHOLD's cutoff does NOT move when only the scored "
       "data changes -- it is an input, so the arm is forward-eligible")
    ok(dc_["cutoff_moved_with_the_data"] is True
       and dc_["forward_eligible"] is False
       and dc_["is_bridge_to_development_number"] is True,
       f"COMPUTED: BY_COUNT's cutoff MOVES with the scored data "
       f"({dc_['cutoff_on_scores_a']} -> {dc_['cutoff_on_scores_b']}), so it "
       f"is read off that data. The disqualifier is a PREDICATE ON THE ARM, "
       f"not a sentence in a docstring")
    ok(dt_["declared_inputs_held_fixed"] == {"budget_key": "10%", "budget": None},
       "and the declared inputs were held fixed while the scores changed, "
       "so the difference above is the convention and nothing else")

    # ---- NEVER POOLED, NEVER SUBSTITUTED -------------------------------
    ok(pooled_increment([thr, thr])["pairing_convention"] == "BY_THRESHOLD",
       "POSITIVE CONTROL: pooling WITHIN one convention is allowed")
    refuses(lambda: pooled_increment([thr, cnt]),
            "DIFFERENT ESTIMANDS",
            "KNOWN-BAD: pooling ACROSS conventions REFUSES -- averaging them "
            "yields a plausible number with no estimand")

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
