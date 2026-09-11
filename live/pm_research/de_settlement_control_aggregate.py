"""THE VERDICT FOR PLAN v2 STEP 2 -- WRITTEN BEFORE THE DRAWS LAND.

REV 165: **"A VERDICT FUNCTION THAT HAS NEVER PRODUCED A FAILURE VERDICT IS
THE SAME CLASS OF INSTRUMENT AS A NULL THAT HAS NEVER FIRED."**

Two reasons this is written now rather than at 18:50Z:

  * PLAN STEP 3 -- "if neither arm beats matched random on settlement P&L
    under the pre-declared two-arm correction, do NOT spend untouched days
    on these frozen arms" -- IS THE BRANCH THAT PROTECTS THE VALIDATION
    SET. If the verdict function cannot return failure, step 3 is
    decoration, and we would find that out at the moment we needed it.
  * Writing the verdict rule AFTER seeing the draws is writing it on seen
    data, which is the rule that voided this programme's last results.

The combination rule is NOT invented here. It is read from
`de_settlement_control_declaration_v1/v2.json`, committed at `b72e329` /
`a52baec` before any draw:

    D_arm  = the SUM of that arm's per-day D over the pool, in cents
             (a SUM and not a weighted mean because the endpoint is CASH
             and cash ADDS; summing also removes the weight choice, so no
             weighting can be revisited now the numbers exist)
    null   = for draw index i, the SUM of each day's i-th matched draw's D
             -- pooling DRAWS, not p-values, which keeps the day structure
    p      = (1 + #{|D_null| >= |D_arm|}) / (1 + n)      two-sided
    family = the TWO arm-level tests, HOLM, alpha 0.05

PRIMARY is 09-04/05/06. **09-03 IS COMPANION-ONLY AND IS NEVER PROMOTED** --
fixed before the draws, because it was already excluded before the
asymmetry draws and cannot be promoted after being seen to change a
verdict.
"""
from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

PRIMARY = ("2026-09-04", "2026-09-05", "2026-09-06")
COMPANION_ONLY = ("2026-09-03",)
ARMS = ("CONDVALUE_X_SKEW", "HAZARD_OVER_SKEWED_REF")
DECLARED_N = 500
ALPHA = 0.05

WRONG_N = "SETTLEMENT_CONTROL_CELL_N_DIFFERS_FROM_THE_DECLARED_N"
DUP = "SETTLEMENT_CONTROL_CHECKPOINT_DOUBLE_COUNTED_A_DRAW"
GAP = "SETTLEMENT_CONTROL_CHECKPOINT_HAS_A_GAP"
PROMOTED = "SETTLEMENT_CONTROL_COMPANION_DAY_PROMOTED_TO_PRIMARY"
PUBLICATION = "SETTLEMENT_CONTROL_PUBLICATION_CLAUSE_UNSATISFIED"
BAD_CHECKPOINT = "SETTLEMENT_CONTROL_CHECKPOINT_MALFORMED"
BAD_CELL = "SETTLEMENT_CONTROL_RESULT_IDENTITY_MISMATCH"
BAD_RECONCILIATION = "SETTLEMENT_CONTROL_RESULT_DOES_NOT_RECONCILE"

VERDICT_BOTH = "SETTLEMENT_SKILL_SUPPORTED_ON_THE_DEVELOPMENT_SET"
VERDICT_ONE = "SUPPORTED_FOR_THAT_ARM_ONLY"
VERDICT_NONE = "NO_SETTLEMENT_SKILL_OVER_MATCHED_RANDOM"
VERDICT_SPLIT = "NOT_ONE_MECHANISM"


class AggregateRefused(RuntimeError):
    """A named refusal."""


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _finite_number(value) -> bool:
    return (isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(float(value)))


def _read_json_object(path: Path, refusal: str) -> tuple[dict, dict]:
    """Parse and digest the same bytes; provenance cannot race the parse."""
    path = Path(path)
    try:
        payload = path.read_bytes()
        document = json.loads(payload)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AggregateRefused(
            f"REFUSED {refusal}: {path}: {type(exc).__name__}: {exc}") \
            from None
    if not isinstance(document, dict):
        raise AggregateRefused(
            f"REFUSED {refusal}: {path} must contain one JSON object.")
    return document, {"path": str(path), "sha256": _sha256(payload)}


def read_cell_checkpoint(ckpt: Path) -> dict:
    """Read one uniquely headed, gapless checkpoint without de-duplication."""
    ckpt = Path(ckpt)
    try:
        payload = ckpt.read_bytes()
        rows = []
        for line_number, line in enumerate(payload.decode().splitlines(), 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"line {line_number} is not an object")
            rows.append(row)
    except (OSError, UnicodeDecodeError, ValueError,
            json.JSONDecodeError) as exc:
        raise AggregateRefused(
            f"REFUSED {BAD_CHECKPOINT}: {ckpt}: "
            f"{type(exc).__name__}: {exc}") from None

    headers = [row for row in rows if row.get("kind") == "HEADER"]
    if len(headers) != 1 or not rows or rows[0] is not headers[0]:
        raise AggregateRefused(
            f"REFUSED {BAD_CHECKPOINT}: {ckpt.name} must begin with exactly "
            f"one HEADER row.")
    header = headers[0]
    header_n = header.get("n_draws")
    if (not isinstance(header_n, int) or isinstance(header_n, bool)
            or header_n <= 0):
        raise AggregateRefused(
            f"REFUSED {BAD_CHECKPOINT}: {ckpt.name} has invalid n_draws "
            f"{header_n!r}.")

    seen = {}
    for row in rows[1:]:
        if row.get("kind") == "HEADER" or "i" not in row:
            raise AggregateRefused(
                f"REFUSED {BAD_CHECKPOINT}: {ckpt.name} has a non-draw row "
                f"after its header.")
        index = row["i"]
        if not isinstance(index, int) or isinstance(index, bool):
            raise AggregateRefused(
                f"REFUSED {BAD_CHECKPOINT}: draw index {index!r} is not an "
                f"integer.")
        if index in seen:
            raise AggregateRefused(
                f"REFUSED {DUP}: draw {index} appears twice in {ckpt.name}; "
                f"even an identical duplicate is not a second sample.")
        if not 0 <= index < header_n:
            raise AggregateRefused(
                f"REFUSED {BAD_CHECKPOINT}: draw {index} is outside "
                f"0..{header_n - 1} in {ckpt.name}.")
        if not _finite_number(row.get("D")):
            raise AggregateRefused(
                f"REFUSED {BAD_CHECKPOINT}: draw {index} has non-finite D "
                f"{row.get('D')!r}.")
        seen[index] = row
    indices = sorted(seen)
    if indices != list(range(len(indices))):
        raise AggregateRefused(
            f"REFUSED {GAP}: {ckpt.name} holds indices up to {indices[-1]} "
            f"with {len(indices)} distinct -- a hole in the draw sequence.")
    return {
        "header": header,
        "draws": [seen[index] for index in indices],
        "source": {"path": str(ckpt), "sha256": _sha256(payload)},
    }


def read_cell_draws(ckpt: Path) -> list:
    """Compatibility wrapper returning strictly read checkpoint draws."""
    return read_cell_checkpoint(ckpt)["draws"]


def _require_equal(actual, expected, label: str) -> None:
    if actual != expected:
        raise AggregateRefused(
            f"REFUSED {BAD_RECONCILIATION}: {label} is {actual!r}; expected "
            f"{expected!r} from the checkpoint/result arithmetic.")


def _verify_forward_cell(result: dict, checkpoint: dict, *, day: str,
                         arm: str, result_path: Path,
                         checkpoint_path: Path) -> dict:
    """Bind a V2 runner result to its exact checkpoint and recompute it."""
    import de_multiday_gate1_runner as R
    import de_settlement_control_run as SC

    book_sha = result.get("book_sha256")
    if not isinstance(book_sha, str) or len(book_sha) != 64:
        raise AggregateRefused(
            f"REFUSED {BAD_CELL}: {day}/{arm} has no 64-character book "
            f"digest.")
    winner_source = result.get("winner_source") or {}
    winner_sha = winner_source.get("sha256")
    if not isinstance(winner_sha, str) or len(winner_sha) != 64:
        raise AggregateRefused(
            f"REFUSED {BAD_CELL}: {day}/{arm} has no 64-character winner "
            f"source digest.")
    derived_seed = R.seed_for(book_sha, arm)
    expected_identity = SC.run_identity(
        day, arm, book_sha, SC.DECLARED_N, derived_seed, winner_sha)
    expected_result = {
        "protocol": SC.PROTOCOL,
        "day": day,
        "arm": arm,
        "n_draws": SC.DECLARED_N,
        "seed": derived_seed,
        "primary_fill_assumption": "REFERENCE_FILLS",
    }
    wrong_result = {
        key: {"read": result.get(key), "expected": expected}
        for key, expected in expected_result.items()
        if result.get(key) != expected
    }
    declared_checkpoint = result.get("checkpoint")
    if (not isinstance(declared_checkpoint, str)
            or Path(declared_checkpoint).resolve()
            != checkpoint_path.resolve()):
        wrong_result["checkpoint"] = {
            "read": declared_checkpoint, "expected": str(checkpoint_path)}
    expected_producer = {
        "path": str(Path(SC.__file__).resolve()),
        "sha256": _sha256(Path(SC.__file__).read_bytes()),
    }
    if result.get("producer") != expected_producer:
        wrong_result["producer"] = {
            "read": result.get("producer"), "expected": expected_producer}
    if result.get("checkpoint_sha256") != checkpoint["source"]["sha256"]:
        wrong_result["checkpoint_sha256"] = {
            "read": result.get("checkpoint_sha256"),
            "expected": checkpoint["source"]["sha256"],
        }
    receipt = result.get("book_receipt") or {}
    params_pin = result.get("params_pin") or {}
    input_verification = result.get("input_verification") or {}
    certifications = result.get("score_neutrality_certifications") or []
    score_bounds = result.get("score_delta_max_certified") or {}
    margin_guard = result.get("forward_book_margin_guard") or {}
    evidence_errors = []
    if (receipt.get("book_sha256") != book_sha
            or receipt.get("builder_commit") != SC.PIPELINE_COMMIT
            or not isinstance(receipt.get("book_scoring_code"), dict)
            or not isinstance(receipt.get("sha256"), str)
            or len(receipt["sha256"]) != 64):
        evidence_errors.append("book_receipt")
    if (not isinstance(params_pin.get("sha256"), str)
            or len(params_pin["sha256"]) != 64):
        evidence_errors.append("params_pin")
    if (not isinstance(input_verification, dict)
            or not all(isinstance(input_verification.get(key), dict)
                       for key in ("be_module", "models", "thetas"))):
        evidence_errors.append("input_verification")
    if (not isinstance(certifications, list) or not certifications
            or not all(isinstance(row, dict)
                       and isinstance(row.get("sha256"), str)
                       and len(row["sha256"]) == 64
                       for row in certifications)):
        evidence_errors.append("score_neutrality_certifications")
    frozen_arms = set(SC.BEN.arm_heads())
    if (set(score_bounds) != frozen_arms
            or not all(_finite_number(value) and value >= 0
                       for value in score_bounds.values())):
        evidence_errors.append("score_delta_max_certified")
    if margin_guard.get("passes") is not True:
        evidence_errors.append("forward_book_margin_guard")
    if (not isinstance(result.get("be_module"), str)
            or len(result["be_module"]) != 64):
        evidence_errors.append("be_module")
    if evidence_errors:
        wrong_result["required_guard_evidence"] = evidence_errors
    if wrong_result:
        raise AggregateRefused(
            f"REFUSED {BAD_CELL}: {day}/{arm} result {result_path.name} "
            f"does not identify this exact run: {wrong_result}.")

    baseline = result.get("zero_model_cancel_baseline_total_cents")
    arm_total = result.get("arm_settled_total_cents")
    observed = result.get("observed_D_cents")
    if not all(_finite_number(value)
               for value in (baseline, arm_total, observed)):
        raise AggregateRefused(
            f"REFUSED {BAD_RECONCILIATION}: {day}/{arm} primary P&L fields "
            f"must be finite numbers.")
    _require_equal(observed, arm_total - baseline, "observed_D_cents")

    header = checkpoint["header"]
    expected_header = {
        "kind": "HEADER", "identity": expected_identity,
        "protocol": SC.PROTOCOL, "n_draws": SC.DECLARED_N,
        "seed": derived_seed, "day": day, "arm": arm,
        "winner_source_sha256": winner_sha,
        "baseline_total_cents": baseline,
    }
    wrong_header = {
        key: {"read": header.get(key), "expected": expected}
        for key, expected in expected_header.items()
        if header.get(key) != expected
    }
    if wrong_header:
        raise AggregateRefused(
            f"REFUSED {BAD_CELL}: checkpoint {checkpoint_path.name} is not "
            f"the checkpoint for {day}/{arm}: {wrong_header}.")

    draws = checkpoint["draws"]
    if len(draws) != SC.DECLARED_N:
        raise AggregateRefused(
            f"REFUSED {WRONG_N}: {day}/{arm} contributes {len(draws)} "
            f"draws and the V2 runner declaration says {SC.DECLARED_N}.")
    for index, draw in enumerate(draws):
        settled = draw.get("settled_total_cents")
        if (draw.get("seed") != derived_seed + index
                or not _finite_number(settled)):
            raise AggregateRefused(
                f"REFUSED {BAD_RECONCILIATION}: draw {index} in "
                f"{checkpoint_path.name} has the wrong seed or a non-finite "
                f"settled total.")
        _require_equal(draw["D"], settled - baseline,
                       f"draw {index} D")

    deltas = [draw["D"] for draw in draws]
    n_at_or_beyond = sum(
        1 for value in deltas if abs(value) >= abs(observed))
    expected_null = {
        "n": len(deltas), "min_D": min(deltas), "max_D": max(deltas),
        "mean_D": sum(deltas) / len(deltas),
        "n_at_or_beyond_two_sided": n_at_or_beyond,
    }
    _require_equal(result.get("null"), expected_null, "null summary")
    _require_equal(
        result.get("p_two_sided"),
        (1 + n_at_or_beyond) / (1 + len(deltas)), "p_two_sided")

    robust = result.get("robustness_leg")
    if not isinstance(robust, dict):
        raise AggregateRefused(
            f"REFUSED {BAD_RECONCILIATION}: {day}/{arm} has no robustness "
            f"leg.")
    robust_base = robust.get("zero_model_cancel_baseline_total_cents")
    robust_arm = robust.get("arm_settled_total_cents")
    robust_D = robust.get("observed_D_cents")
    if not all(_finite_number(value)
               for value in (robust_base, robust_arm, robust_D)):
        raise AggregateRefused(
            f"REFUSED {BAD_RECONCILIATION}: {day}/{arm} robustness P&L "
            f"fields must be finite numbers.")
    expected_reversal = ((observed > 0) - (observed < 0)
                         != (robust_D > 0) - (robust_D < 0))
    expected_robust = {
        "label": "NO_FILLS_UNTIL_NEXT_GENERATION",
        "observed_D_cents": robust_arm - robust_base,
        "primary_label": "REFERENCE_FILLS",
        "primary_observed_D_cents": observed,
        "sign_reversal": expected_reversal,
        "a_sign_reversal_blocks_promotion": True,
    }
    wrong_robust = {
        key: {"read": robust.get(key), "expected": expected}
        for key, expected in expected_robust.items()
        if robust.get(key) != expected
    }
    if wrong_robust:
        raise AggregateRefused(
            f"REFUSED {BAD_RECONCILIATION}: {day}/{arm} robustness leg does "
            f"not reconcile: {wrong_robust}.")
    return {
        "status": "RESULT_AND_CHECKPOINT_RECONCILED",
        "run_identity": expected_identity,
        "derived_seed": derived_seed,
        "result": str(result_path),
        "producer": expected_producer,
        "checkpoint": checkpoint["source"],
    }


def load_cell(root: Path, day: str, arm: str, *,
              strict_forward: bool = False) -> dict:
    """Load one cell, enforcing n and optionally the V2 forward contract."""
    root = Path(root)
    c = day.replace("-", "")
    result_path = root / f"de_settle_result_{c}_{arm}.json"
    checkpoint_path = root / f"de_settle_ckpt_{day}_{arm}.jsonl"
    res, result_source = _read_json_object(result_path, BAD_CELL)
    checkpoint = read_cell_checkpoint(checkpoint_path)
    draws = checkpoint["draws"]
    if len(draws) != DECLARED_N:
        raise AggregateRefused(
            f"REFUSED {WRONG_N}: {day}/{arm} contributes {len(draws)} "
            f"draws and the declaration says {DECLARED_N}. The floor is a "
            f"MINIMUM, not a licence to report a different n than "
            f"declared.")
    validation = (_verify_forward_cell(
        res, checkpoint, day=day, arm=arm, result_path=result_path,
        checkpoint_path=checkpoint_path) if strict_forward else
        {"status": "LEGACY_AGGREGATE_N_ONLY"})
    return {"day": day, "arm": arm, "result": res,
            "D": res["observed_D_cents"],
            "baseline_total_cents":
                res["zero_model_cancel_baseline_total_cents"],
            "arm_total_cents": res["arm_settled_total_cents"],
            "draw_D": [d["D"] for d in draws],
            "n": len(draws), "result_source": result_source,
            "checkpoint_source": checkpoint["source"],
            "cell_validation": validation}


def pooled(cells: dict, days, arm: str) -> dict:
    """D_arm and its null, by the declared rule: SUM across days."""
    picked = [cells[(d, arm)] for d in days]
    D_arm = sum(c["D"] for c in picked)
    n = min(c["n"] for c in picked)
    null = [sum(c["draw_D"][i] for c in picked) for i in range(n)]
    n_ge = sum(1 for v in null if abs(v) >= abs(D_arm))
    return {"arm": arm, "days": list(days), "D_arm_cents": D_arm,
            "n_draws": n, "n_at_or_beyond_two_sided": n_ge,
            "p_two_sided": (1 + n_ge) / (1 + n),
            "null_min": min(null), "null_max": max(null),
            "null_mean": sum(null) / len(null),
            "per_day_D_cents": {c["day"]: c["D"] for c in picked}}


def holm(ps, alpha: float = ALPHA) -> list:
    """Step-down Holm over the family. Once one fails, the rest fail."""
    order = sorted(range(len(ps)), key=lambda k: ps[k])
    m, out, failed = len(ps), [None] * len(ps), False
    for r, k in enumerate(order):
        thr = alpha / (m - r)
        ok = (ps[k] <= thr) and not failed
        if not ok:
            failed = True
        out[k] = {"p": ps[k], "threshold": thr, "passes": ok}
    return out


def verdict_for(pools: list, holms: list) -> dict:
    """The four declared cases. NOTHING here is decided by the code's mood."""
    passes = [h["passes"] for h in holms]
    signs = [p["D_arm_cents"] > 0 for p in pools]
    n_pass = sum(passes)
    if n_pass == 2 and len(set(signs)) > 1:
        v, why = VERDICT_SPLIT, ("two opposite significant effects are not "
                                 "one mechanism; this OVERRIDES any pass")
    elif n_pass == 2:
        v, why = VERDICT_BOTH, "both arms beat matched random under Holm"
    elif n_pass == 1:
        v, why = VERDICT_ONE, ("one candidate clearing its own control is "
                               "NOT the programme clearing it")
    else:
        v, why = VERDICT_NONE, ("neither arm beats matched random on the "
                                "decision metric under the pre-declared "
                                "two-arm correction")
    out = {"verdict": v, "why": why,
           "n_arms_passing": n_pass,
           "arms_passing": [p["arm"] for p, ok in zip(pools, passes) if ok]}
    if v == VERDICT_NONE:
        out["triggers_plan_step_3"] = True
        out["plan_step_3"] = (
            "DO NOT SPEND UNTOUCHED DAYS ON THESE FROZEN ARMS. QR_SKEW_ONLY "
            "remains the reference. Any redesign uses CONSUMED data only "
            "and starts a new freeze and a new validation clock.")
        out["this_is_the_outcome_that_saves_the_validation_set"] = True
    else:
        out["triggers_plan_step_3"] = False
    return out


def publication_block(days, receipts_dir: Path, cells: dict) -> dict:
    """STEP 2's FOUR ELEMENTS, TOGETHER. Absence of any REFUSES.

    The plan requires the zero-model-cancel baseline, EVERY random-control
    distribution, coverage/exclusion counts and settlement finality to be
    published TOGETHER -- not as four artifacts a reader must assemble.
    09-03's 40 masked windows reach the number here: the point estimate
    carried only the post-mask `n_supplied` 247, so a reader saw a
    247-window day without being told 40 were removed (BE 156)."""
    out, missing = {}, []
    for day in days:
        c = day.replace("-", "")
        rev = "EV22" if day == "2026-09-03" else "EV21"
        rp = receipts_dir / f"be_daybook_receipt_{c}_btc__L250ms__{rev}.json"
        if not rp.is_file():
            missing.append(f"{day}:receipt")
            continue
        r = json.loads(rp.read_text())
        sel = r.get("selection") or {}
        mask = sel.get("mask") or {}
        ev = (r.get("assembly_evidence") or {}).get(
            "UNCOVERED_GENERATIONS") or {}
        st = ((r.get("reference") or {}).get("statuses")) or {}
        for k in ("n_present", "n_masked", "n_supplied",
                  "mask_identity_hash"):
            if mask.get(k) in (None, ""):
                missing.append(f"{day}:mask.{k}")
        if ev.get("coverage") is None:
            missing.append(f"{day}:coverage")
        base = {a: cells[(day, a)]["baseline_total_cents"]
                for a in ARMS if (day, a) in cells}
        out[day] = {
            "zero_model_cancel_baseline_total_cents": base,
            "coverage": ev.get("coverage"),
            "n_uncovered": ev.get("count"),
            "n_reference_generations": ev.get("n_reference_generations"),
            "windows_masked": {"n_present": mask.get("n_present"),
                               "n_masked": mask.get("n_masked"),
                               "n_supplied": mask.get("n_supplied"),
                               "mask_identity_hash":
                                   mask.get("mask_identity_hash")},
            "exclusions": {k: v for k, v in st.items()
                           if k.startswith(("TRANCHE", "TERMINAL",
                                            "BINANCE", "NO_REPLAY",
                                            "ADMITTED", "RECONCIL"))},
            "settlement_finality": (
                ((cells.get((day, ARMS[0])) or {}).get("result") or {})
                .get("settlement_finality")
                or "see the day artifact's winner_source."
                   "is_final_for_quotation"),
            "era": sel.get("era"),
            "n_gap_bearing_windows": sel.get("n_gap_bearing_windows"),
        }
    if missing:
        raise AggregateRefused(
            f"REFUSED {PUBLICATION}: step 2 requires the baseline, every "
            f"control distribution, coverage/exclusion counts and "
            f"settlement finality PUBLISHED TOGETHER, and these are "
            f"absent: {missing}. A number a reader cannot place is not "
            f"published.")
    return out


def aggregate(root: Path, receipts_dir: Path) -> dict:
    """The whole verdict, both pools, with everything step 2 requires."""
    days = list(PRIMARY) + list(COMPANION_ONLY)
    cells = {(d, a): load_cell(root, d, a) for d in days for a in ARMS}
    pools = {}
    for name, dd in (("PRIMARY", PRIMARY),
                     ("COMPANION_ALL_FOUR", tuple(days))):
        ps = [pooled(cells, dd, a) for a in ARMS]
        hs = holm([p["p_two_sided"] for p in ps])
        v = verdict_for(ps, hs)
        pools[name] = {"days": list(dd), "arms": {
            p["arm"]: {**p, "holm": h} for p, h in zip(ps, hs)}, **v}
    if set(pools["PRIMARY"]["days"]) & set(COMPANION_ONLY):
        raise AggregateRefused(
            f"REFUSED {PROMOTED}: {COMPANION_ONLY} is COMPANION-ONLY and "
            f"appears in the PRIMARY pool. It was fixed as companion-only "
            f"before any draw precisely so it could not be promoted after "
            f"being seen to change a verdict.")
    return {
        "protocol": "P003_DE_SETTLEMENT_CONTROL_VERDICT_V1",
        "declaration": ["de_settlement_control_declaration_v1.json (b72e329)",
                        "de_settlement_control_declaration_v2.json (a52baec)"],
        "THE_VERDICT": pools["PRIMARY"]["verdict"],
        "the_verdict_is_the_PRIMARY_pool": (
            "09-04, 09-05, 09-06. The companion pool of all four is "
            "reported BESIDE it and never instead of it."),
        "pools": pools,
        "matched_count_field_path": (
            "bk['rows'] -> arm_stream -> above-theta subset -> distinct "
            "(slug, side, int(gen)) grouped by (side, utc_hour). NOT the "
            "fr.reference count, which this runner never reads (a52baec)"),
        "n_per_cell_enforced": DECLARED_N,
        "publication": publication_block(days, receipts_dir, cells),
        "per_cell_descriptive_only": {
            f"{d}|{a}": {"D_cents": cells[(d, a)]["D"],
                         "p_two_sided": cells[(d, a)]["result"]["p_two_sided"],
                         "n": cells[(d, a)]["n"]}
            for d in days for a in ARMS},
        "per_cell_values_carry_no_verdict": (
            "descriptive only; the verdict is the two arm-level pooled "
            "tests and nothing else"),
        "what_this_cannot_establish": {
            "effective_independent_units": (
                "FOUR days, THREE in PRIMARY -- not eight cells. The two "
                "arms within a day share the day, the book, the reference "
                "path and a BITWISE IDENTICAL baseline."),
            "no_interval": "rule 8 forbids one below five complete days",
            "validation": ("CONSUMED days -- development evidence, never "
                           "validation"),
            "matching_bias": (
                "distinct-generation matching UNDER-matches raw exposure "
                "(max 23 cancels on one generation); THE BIAS RUNS TOWARD "
                "FINDING THE ARM SKILFUL, so any advantage is an UPPER "
                "BOUND"),
        },
    }


# --------------------------------------------------------------------------
# THE FALSIFIER -- THE VERDICT FUNCTION MUST BE ABLE TO FAIL
# --------------------------------------------------------------------------
def falsify() -> int:                                        # noqa: C901
    fails = []

    def ok(cond, label):
        print(f"  {'ok  ' if cond else 'FAIL'}  {label}")
        if not cond:
            fails.append(label)

    def refuses(fn, needle, label):
        try:
            fn()
        except AggregateRefused as e:
            ok(needle in str(e), f"{label} -- refuses {needle}")
        except Exception as e:                               # noqa: BLE001
            ok(False, f"{label} -- raised {type(e).__name__}: {e}")
        else:
            ok(False, f"{label} -- DID NOT REFUSE")

    print("[de_settlement_control_aggregate] falsifier")

    def synth(D_by_arm, null_scale=1000.0, n=500):
        """Cells whose arm delta is D and whose null is symmetric noise."""
        import random
        cells = {}
        for arm, Dtot in D_by_arm.items():
            per = Dtot / len(PRIMARY)
            for k, d in enumerate(PRIMARY):
                rr = random.Random(hash((arm, d)) & 0xffff)
                cells[(d, arm)] = {
                    "day": d, "arm": arm, "D": per,
                    "baseline_total_cents": 1000.0,
                    "arm_total_cents": 1000.0 + per,
                    "draw_D": [rr.gauss(0, null_scale) for _ in range(n)],
                    "n": n, "result": {"p_two_sided": 0.5}}
        return cells

    # (a) NEITHER ARM BEATS THE CONTROL -> FAILURE, and step 3 named.
    weak = synth({"CONDVALUE_X_SKEW": 300.0,
                  "HAZARD_OVER_SKEWED_REF": -200.0})
    ps = [pooled(weak, PRIMARY, a) for a in ARMS]
    hs = holm([p["p_two_sided"] for p in ps])
    v = verdict_for(ps, hs)
    ok(v["verdict"] == VERDICT_NONE and v["triggers_plan_step_3"] is True
       and "DO NOT SPEND UNTOUCHED DAYS" in v["plan_step_3"],
       f"(a) THE FAILURE VERDICT EXISTS AND FIRES: neither arm beating "
       f"matched random returns `{v['verdict']}` and TRIGGERS PLAN STEP 3 "
       f"by name. This is the branch that protects the validation set, and "
       f"a verdict function that had only ever passed could not reach it")

    # (b) BOTH BEAT IT -> PASS.
    strong = synth({"CONDVALUE_X_SKEW": 900000.0,
                    "HAZARD_OVER_SKEWED_REF": 800000.0})
    ps2 = [pooled(strong, PRIMARY, a) for a in ARMS]
    hs2 = holm([p["p_two_sided"] for p in ps2])
    v2 = verdict_for(ps2, hs2)
    ok(v2["verdict"] == VERDICT_BOTH and v2["n_arms_passing"] == 2
       and v2["triggers_plan_step_3"] is False,
       f"(b) AND IT CAN PASS: both arms far beyond the null return "
       f"`{v2['verdict']}`")

    # (c) MIXED -> resolved by the DECLARED rule, not by the code's mood.
    mixed = synth({"CONDVALUE_X_SKEW": 900000.0,
                   "HAZARD_OVER_SKEWED_REF": 100.0})
    ps3 = [pooled(mixed, PRIMARY, a) for a in ARMS]
    hs3 = holm([p["p_two_sided"] for p in ps3])
    v3 = verdict_for(ps3, hs3)
    ok(v3["verdict"] == VERDICT_ONE and v3["n_arms_passing"] == 1
       and v3["arms_passing"] == ["CONDVALUE_X_SKEW"]
       and "NOT the programme clearing it" in v3["why"],
       f"(c) MIXED RESOLVES BY THE DECLARED RULE: one arm clearing gives "
       f"`{v3['verdict']}` and says in the artifact that it is NOT a "
       f"programme-level pass")

    # (d) BOTH PASS WITH OPPOSITE SIGNS -> the override.
    split = synth({"CONDVALUE_X_SKEW": 900000.0,
                   "HAZARD_OVER_SKEWED_REF": -900000.0})
    ps4 = [pooled(split, PRIMARY, a) for a in ARMS]
    hs4 = holm([p["p_two_sided"] for p in ps4])
    v4 = verdict_for(ps4, hs4)
    ok(v4["verdict"] == VERDICT_SPLIT,
       f"(d) OPPOSITE SIGNS OVERRIDE A DOUBLE PASS: `{v4['verdict']}` -- "
       f"two opposite significant effects are not one mechanism")

    # (e) HOLM IS STEP-DOWN, not two independent thresholds.
    h = holm([0.026, 0.030])
    ok(h[0]["threshold"] == 0.025 and h[0]["passes"] is False
       and h[1]["passes"] is False,
       f"(e) HOLM IS STEP-DOWN: p=0.026 misses 0.025 and the SECOND test "
       f"fails with it even though 0.030 < 0.05 -- which is exactly the "
       f"shape the asymmetry PRIMARY produced")

    # (f) checkpoint identity, gaps and duplicate refusal.
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        ck = root / "de_settle_ckpt_2026-09-04_X.jsonl"
        header = json.dumps({"kind": "HEADER", "n_draws": 3})
        ck.write_text("\n".join(
            [header]
            + [json.dumps({"i": i, "D": float(i)}) for i in range(3)]))
        ok(len(read_cell_draws(ck)) == 3, "(f) a clean checkpoint reads")
        ck.write_text("\n".join(
            [header]
            + [json.dumps({"i": i, "D": float(i)}) for i in range(3)]
            + [json.dumps({"i": 1, "D": 1.0})]))
        refuses(lambda: read_cell_draws(ck), DUP,
                "(f) KNOWN-BAD: even an identical repeated draw")
        ck.write_text("\n".join(
            [header]
            + [json.dumps({"i": i, "D": float(i)}) for i in range(3)]
            + [json.dumps({"i": 1, "D": 99.0})]))
        refuses(lambda: read_cell_draws(ck), DUP,
                "(f) KNOWN-BAD: the same index with a DIFFERENT value")
        ck.write_text("\n".join(
            [header]
            + [json.dumps({"i": i, "D": float(i)})
               for i in (0, 1)]
            + [json.dumps({"i": 3, "D": 3.0})]))
        refuses(lambda: read_cell_draws(ck), BAD_CHECKPOINT,
                "(f) KNOWN-BAD: an out-of-range draw index")
        ck.write_text("\n".join(
            [header]
            + [json.dumps({"i": i, "D": float(i)})
               for i in (0, 2)]))
        refuses(lambda: read_cell_draws(ck), GAP,
                "(f) KNOWN-BAD: a hole in the draw indices")

    # (g) the forward reader binds result, checkpoint and recomputed money.
    with tempfile.TemporaryDirectory() as td:
        import de_multiday_gate1_runner as R
        import de_settlement_control_run as SC

        root = Path(td)
        day, arm, book_sha = "2026-09-07", ARMS[0], "a" * 64
        seed = R.seed_for(book_sha, arm)
        winner_sha = "f" * 64
        identity = SC.run_identity(
            day, arm, book_sha, DECLARED_N, seed, winner_sha)
        checkpoint = root / f"de_settle_ckpt_{day}_{arm}.jsonl"
        checkpoint.write_text("\n".join(
            [json.dumps({
                "kind": "HEADER", "identity": identity,
                "protocol": SC.PROTOCOL, "n_draws": DECLARED_N,
                "seed": seed, "day": day, "arm": arm,
                "winner_source_sha256": winner_sha,
                "baseline_total_cents": 10.0})]
            + [json.dumps({
                "i": index, "seed": seed + index,
                "settled_total_cents": 10.0, "D": 0.0})
               for index in range(DECLARED_N)]))
        result_path = root / f"de_settle_result_20260907_{arm}.json"
        good_result = {
            "protocol": SC.PROTOCOL, "day": day, "arm": arm,
            "book_sha256": book_sha,
            "winner_source": {"path": "resolutions.jsonl",
                              "sha256": winner_sha},
            "book_receipt": {
                "sha256": "b" * 64, "book_sha256": book_sha,
                "builder_commit": SC.PIPELINE_COMMIT,
                "book_scoring_code": {}},
            "params_pin": {"path": "params.json", "sha256": "c" * 64},
            "input_verification": {
                "be_module": {}, "models": {}, "thetas": {}},
            "score_neutrality_certifications": [
                {"path": "cert.json", "sha256": "d" * 64}],
            "score_delta_max_certified": {
                frozen_arm: 0.25 for frozen_arm in SC.BEN.arm_heads()},
            "forward_book_margin_guard": {"passes": True},
            "primary_fill_assumption": "REFERENCE_FILLS",
            "zero_model_cancel_baseline_total_cents": 10.0,
            "arm_settled_total_cents": 11.0,
            "observed_D_cents": 1.0,
            "robustness_leg": {
                "label": "NO_FILLS_UNTIL_NEXT_GENERATION",
                "zero_model_cancel_baseline_total_cents": 10.0,
                "arm_settled_total_cents": 11.0,
                "observed_D_cents": 1.0,
                "primary_label": "REFERENCE_FILLS",
                "primary_observed_D_cents": 1.0,
                "sign_reversal": False,
                "a_sign_reversal_blocks_promotion": True},
            "n_draws": DECLARED_N, "seed": seed,
            "null": {"n": DECLARED_N, "min_D": 0.0,
                     "max_D": 0.0, "mean_D": 0.0,
                     "n_at_or_beyond_two_sided": 0},
            "p_two_sided": 1 / (1 + DECLARED_N),
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": _sha256(checkpoint.read_bytes()),
            "be_module": "e" * 64,
            "producer": {
                "path": str(Path(SC.__file__).resolve()),
                "sha256": _sha256(Path(SC.__file__).read_bytes()),
            },
        }
        result_path.write_text(json.dumps(good_result))
        loaded = load_cell(root, day, arm, strict_forward=True)
        ok(loaded["cell_validation"]["status"]
           == "RESULT_AND_CHECKPOINT_RECONCILED",
           "(g) a V2 result is accepted only after exact reconciliation")

        bad_identity = dict(good_result, day="2026-09-08")
        result_path.write_text(json.dumps(bad_identity))
        refuses(lambda: load_cell(root, day, arm, strict_forward=True),
                BAD_CELL,
                "(g) KNOWN-BAD: a result copied from another cell")

        bad_summary = json.loads(json.dumps(good_result))
        bad_summary["null"]["n_at_or_beyond_two_sided"] = 1
        result_path.write_text(json.dumps(bad_summary))
        refuses(lambda: load_cell(root, day, arm, strict_forward=True),
                BAD_RECONCILIATION,
                "(g) KNOWN-BAD: reported null differs from checkpoint draws")

        result_path.write_text(json.dumps(good_result))
        checkpoint.write_text(checkpoint.read_text() + "\n")
        refuses(lambda: load_cell(root, day, arm, strict_forward=True),
                BAD_CELL,
                "(g) KNOWN-BAD: checkpoint bytes changed after result emit")

    ok(set(PRIMARY).isdisjoint(COMPANION_ONLY),
       f"(h) 09-03 IS COMPANION-ONLY BY CONSTRUCTION: PRIMARY {PRIMARY} "
       f"and COMPANION {COMPANION_ONLY} are disjoint, and `aggregate` "
       f"REFUSES {PROMOTED} if they ever overlap")

    print(f"[de_settlement_control_aggregate] "
          f"{'PASS' if not fails else 'FAIL'} -- {len(fails)} failing")
    for f in fails:
        print(f"    FAILED: {f}")
    return 1 if fails else 0


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--falsify" in argv:
        return falsify()
    root = Path("/home/yuqing/ctaNew/data/pm_5min/derived/settle")
    rec = Path("/home/yuqing/ctaNew/data/pm_5min/derived")
    print(json.dumps(aggregate(root, rec), indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
