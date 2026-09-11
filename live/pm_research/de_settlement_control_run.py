"""Run one forward settlement-control cell, fail-closed and resumable.

The endpoint is R-801 settlement P&L. The matched random control is drawn
on distinct reference generations, side and UTC hour. This runner also
produces the declared NO_FILLS_UNTIL_NEXT_GENERATION robustness leg and
verifies the full frozen input cascade before the first replay.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import be_score_neutrality as BEN          # noqa: E402
import de_asymmetry_null_run as ASYM       # noqa: E402
import de_matched_cancel_control as MCC    # noqa: E402
import de_multiday_gate1_runner as R       # noqa: E402

PROTOCOL = "P003_DE_SETTLEMENT_CONTROL_RUN_V2"
PIPELINE_COMMIT = "7ed5a9015f75de64feeeeaad21d97e4eecc2b15c"
CERTIFICATION_DAY = "20260903"
DECL = HERE / "declarations" / "de_settlement_control_declaration_v1.json"
_DECLARATION = json.loads(DECL.read_text())
DECLARED_N = int(_DECLARATION["control"]["n_draws"])
FLOOR = int(_DECLARATION["control"]["floor"])

WRONG_N = "SETTLEMENT_CONTROL_CELL_N_DIFFERS_FROM_THE_DECLARED_N"
UNDER_SAMPLED = "SETTLEMENT_CONTROL_BELOW_THE_FAIL_CLOSED_FLOOR"
PARAMS_NOT_FROZEN = "SETTLEMENT_CONTROL_PARAMS_ARE_NOT_THE_FROZEN_PARAMS"
NO_RECEIPT = "SETTLEMENT_CONTROL_HAS_NO_BOOK_RECEIPT"
BOOK_MISMATCH = "SETTLEMENT_CONTROL_BOOK_AND_RECEIPT_DISAGREE"
NO_CERTIFICATION = "SETTLEMENT_CONTROL_HAS_NO_SCORE_NEUTRALITY_CERTIFICATION"
BAD_CERTIFICATION = "SETTLEMENT_CONTROL_SCORE_NEUTRALITY_NOT_CERTIFIED"
BAD_CHECKPOINT = "SETTLEMENT_CONTROL_CHECKPOINT_MALFORMED"
OUTPUT_EXISTS = "SETTLEMENT_CONTROL_RESULT_ALREADY_EXISTS"
UNKNOWN_ARM = "SETTLEMENT_CONTROL_ARM_NOT_IN_THE_FREEZE"
SEED_NOT_DERIVED = "SETTLEMENT_CONTROL_SEED_NOT_DERIVED_FROM_BOOK_AND_ARM"


class SettlementControlRefused(RuntimeError):
    """A named refusal."""


def _sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def settled_total(fills, winners) -> float:
    """R-801's settled total in cents, through its instrument of record."""
    legs = R.settlement_legs_by_slug(fills, winners)
    return float(legs["total_cents"])


def frozen_params(params=None) -> tuple[dict, dict]:
    """Load the params named by the freeze; a caller cannot replace them."""
    frozen, path, pin = BEN.frozen_params()
    frozen = R.load_params(path)
    if params is not None and params != frozen:
        raise SettlementControlRefused(
            f"REFUSED {PARAMS_NOT_FROZEN}: the supplied params do not equal "
            f"{path.name}, whose digest is fixed by {BEN.FREEZE_REL}.")
    return frozen, {"path": str(path), "sha256": pin["sha256"]}


def run_identity(day: str, arm: str, book_sha: str, n_draws: int,
                 seed: int, winner_sha: str) -> str:
    """Bind a checkpoint to endpoint, cell, book, winners, count and seed."""
    return hashlib.sha256(
        f"{PROTOCOL}|{day}|{arm}|{book_sha}|{winner_sha}|"
        f"{n_draws}|{seed}".encode()
    ).hexdigest()


def read_checkpoint(path: Path, identity: str, *, day: str, arm: str,
                    n_draws: int, seed: int, winner_sha: str,
                    baseline_total_cents: float) -> dict:
    """Read only a uniquely headed, gapless prefix for this exact run."""
    path = Path(path)
    if not path.exists():
        return {"draws": {}, "n": 0, "resumed": False}
    rows = []
    try:
        for line in path.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError("checkpoint row is not an object")
                rows.append(row)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise SettlementControlRefused(
            f"REFUSED {BAD_CHECKPOINT}: {path}: "
            f"{type(exc).__name__}: {exc}") from None
    headers = [row for row in rows if row.get("kind") == "HEADER"]
    expected = {"identity": identity, "protocol": PROTOCOL,
                "n_draws": n_draws, "seed": seed,
                "day": day, "arm": arm,
                "winner_source_sha256": winner_sha,
                "baseline_total_cents": baseline_total_cents}
    if len(headers) != 1 or not rows or rows[0] is not headers[0] or any(
            headers[0].get(key) != value for key, value in expected.items()):
        raise SettlementControlRefused(
            f"REFUSED {BAD_CHECKPOINT}: {path} must begin with exactly one "
            f"header for this protocol, cell, count, seed and identity.")
    draws = {}
    for row in rows[1:]:
        if row.get("kind") == "HEADER" or "i" not in row:
            raise SettlementControlRefused(
                f"REFUSED {BAD_CHECKPOINT}: non-draw row after the header.")
        try:
            index = int(row["i"])
        except (TypeError, ValueError):
            raise SettlementControlRefused(
                f"REFUSED {BAD_CHECKPOINT}: draw index {row['i']!r} is "
                f"not an integer.") from None
        if index in draws or not 0 <= index < n_draws:
            raise SettlementControlRefused(
                f"REFUSED {BAD_CHECKPOINT}: draw index {index} is duplicate "
                f"or outside 0..{n_draws - 1}.")
        settled = row.get("settled_total_cents")
        delta = row.get("D")
        if (row.get("seed") != seed + index
                or not isinstance(settled, (int, float))
                or isinstance(settled, bool)
                or not math.isfinite(float(settled))
                or not isinstance(delta, (int, float))
                or isinstance(delta, bool)
                or not math.isfinite(float(delta))
                or delta != settled - baseline_total_cents):
            raise SettlementControlRefused(
                f"REFUSED {BAD_CHECKPOINT}: draw {index} does not reconcile "
                f"to seed {seed + index} and baseline "
                f"{baseline_total_cents}.")
        draws[index] = row
    if draws and sorted(draws) != list(range(max(draws) + 1)):
        raise SettlementControlRefused(
            f"REFUSED {BAD_CHECKPOINT}: completed draw indices are not a "
            f"gapless prefix: {sorted(draws)[:8]}.")
    return {"draws": draws, "n": len(draws), "resumed": bool(draws)}


def _load_certifications(sources) -> tuple[list[dict], list[dict]]:
    if not sources:
        raise SettlementControlRefused(
            f"REFUSED {NO_CERTIFICATION}: the scoring path moved and no "
            f"score-neutrality result was supplied.")
    documents, provenance = [], []
    for source in sources:
        path = Path(source)
        if not path.is_file():
            raise SettlementControlRefused(
                f"REFUSED {NO_CERTIFICATION}: no certification at {path}.")
        try:
            payload = path.read_bytes()
            documents.append(json.loads(payload))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise SettlementControlRefused(
                f"REFUSED {BAD_CERTIFICATION}: {path}: "
                f"{type(exc).__name__}: {exc}") from None
        provenance.append({
            "path": str(path),
            "sha256": hashlib.sha256(payload).hexdigest(),
            "digest_is_of_the_parsed_buffer": True,
        })
    return documents, provenance


def certified_delta_bounds(documents: list[dict]) -> dict:
    """Take the worst certified perturbation per frozen arm."""
    if not documents:
        raise SettlementControlRefused(
            f"REFUSED {NO_CERTIFICATION}: no certification documents.")
    arms = BEN.arm_heads()
    bounds = {arm: [] for arm in arms}
    for document in documents:
        identity_day = str((document.get("identity") or {}).get("day") or "")
        compared = document.get("compared_books") or {}
        source_digests = [
            (compared.get(side) or {}).get("sha256")
            for side in ("old", "new")]
        receipts = document.get("comparison_receipts") or {}
        receipt_evidence = [receipts.get(side) or {}
                            for side in ("old", "new")]
        producer_sha = (document.get("producer") or {}).get("sha256")
        expected_producer_sha = _sha(HERE / "be_score_neutrality.py")
        if (document.get("protocol")
                != "BE_SCORE_NEUTRALITY_V2_REVIEW173_BAR"
                or document.get("verdict") != "SUPPORTED_ON_THIS_DAY"
                or identity_day.replace("-", "") != CERTIFICATION_DAY
                or document.get("n_flips_overall") != 0
                or not all(isinstance(value, str) and len(value) == 64
                           for value in source_digests)
                or source_digests[0] == source_digests[1]
                or [row.get("builder_commit") for row in receipt_evidence]
                   != [BEN.CERTIFICATION_OLD_COMMIT,
                       BEN.CERTIFICATION_NEW_COMMIT]
                or [row.get("book_sha256") for row in receipt_evidence]
                   != source_digests
                or not all(isinstance(row.get("sha256"), str)
                           and len(row["sha256"]) == 64
                           for row in receipt_evidence)
                or producer_sha != expected_producer_sha):
            raise SettlementControlRefused(
                f"REFUSED {BAD_CERTIFICATION}: expected the 09-03 "
                f"SUPPORTED_ON_THIS_DAY certificate, zero flips, two source "
                f"book digests and producer digest {expected_producer_sha}; "
                f"read protocol={document.get('protocol')!r}, "
                f"verdict={document.get('verdict')!r}, day={identity_day!r}, "
                f"n_flips={document.get('n_flips_overall')!r}.")
        rows = document.get("per_arm") or {}
        for arm in arms:
            row = rows.get(arm) or {}
            value = row.get("B_delta_max_abs")
            if (not isinstance(value, (int, float))
                    or isinstance(value, bool)
                    or not math.isfinite(float(value)) or value < 0
                    or row.get("A_n_flips") != 0
                    or row.get("D_reconciles") is not True
                    or row.get("strength") in
                       (None, "REFUTED", "LUCK_NOT_CERTIFICATION")):
                raise SettlementControlRefused(
                    f"REFUSED {BAD_CERTIFICATION}: {arm} does not carry a "
                    f"reconciled, zero-flip, non-luck perturbation bound.")
            bounds[arm].append(float(value))
    return {arm: max(values) for arm, values in bounds.items()}


def _day_key(value) -> str:
    return str(value or "").replace("-", "")


def verify_book_receipt(receipt_path, book_sha: str | None, day: str,
                        *, book_path=None) -> dict:
    """Bind the loaded book to its receipt and the current scoring bytes."""
    path = Path(receipt_path) if receipt_path else None
    if path is None or not path.is_file():
        raise SettlementControlRefused(
            f"REFUSED {NO_RECEIPT}: {receipt_path!r}.")
    try:
        payload = path.read_bytes()
        receipt = json.loads(payload)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SettlementControlRefused(
            f"REFUSED {NO_RECEIPT}: {path}: "
            f"{type(exc).__name__}: {exc}") from None
    declared = (receipt.get("book") or {}).get("sha256")
    declared_path = (receipt.get("book") or {}).get("path")
    builder_commit = (receipt.get("producing_code") or {}).get(
        "builder_commit")
    digest_valid = isinstance(declared, str) and len(declared) == 64
    digest_matches = book_sha is None or declared == book_sha
    path_matches = (book_path is None
                    or (isinstance(declared_path, str)
                        and Path(declared_path).resolve()
                        == Path(book_path).resolve()))
    if (not digest_valid or not digest_matches
            or _day_key(receipt.get("day")) != _day_key(day)
            or builder_commit != PIPELINE_COMMIT or not path_matches):
        raise SettlementControlRefused(
            f"REFUSED {BOOK_MISMATCH}: loaded day/digest is "
            f"{day}/{str(book_sha)[:16]}, receipt says "
            f"{receipt.get('day')}/{str(declared)[:16]} from builder "
            f"{builder_commit} at {declared_path}; frozen pipeline is "
            f"{PIPELINE_COMMIT}, requested path is {book_path}.")
    scoring = R.assert_book_scoring_code(
        receipt, where=f"the settlement-control path for {day}")
    if not scoring.get("is_a_match"):
        unnamed = set(scoring.get(
            "in_the_set_and_NOT_named_by_the_receipt") or [])
        lazy = R.ruled_lazy_exemption(HERE)
        beyond = sorted(unnamed - set(lazy["exempt"]))
        if beyond:
            raise SettlementControlRefused(
                f"REFUSED {R.LAZY_ONLY_EXEMPTION}: receipt omits {beyond}, "
                f"outside the ruled lazy-import set {lazy['exempt']}.")
        scoring["ruled_lazy_exemption"] = {
            "ruling": "USER, DE 174 (2)", "unnamed": sorted(unnamed),
            "lazy_only_set": lazy["exempt"], "satisfied": True}
    return {"path": str(path),
            "sha256": hashlib.sha256(payload).hexdigest(),
            "digest_is_of_the_parsed_buffer": True,
            "book_sha256": declared, "book_path": declared_path,
            "receipt_day": receipt.get("day"),
            "builder_commit": builder_commit,
            "book_scoring_code": scoring}


def replay_with_fill_model(module, book: dict, scores: list, theta: float,
                           fill_model: str) -> dict:
    """Replay through the same engine with only the declared fill leg moved."""
    if fill_model == "REFERENCE_FILLS":
        return module.replay(book, scores, theta)
    import de_phase4_diag_runner as DR
    import harmful_stateful_policy as HSP
    if fill_model not in HSP.REPOST_FILL_MODELS:
        raise SettlementControlRefused(
            f"REFUSED: unknown repost fill model {fill_model!r}.")
    policy_params = module.params_for(theta)
    policy_params["repost_fill_model"] = fill_model
    HSP.validate_scores(scores)
    replay = HSP.replay_policy(book["ref"], scores, policy_params)
    fills = DR.received_fills(
        replay, book["ref"], DR._decision_times(scores))
    return {"cancels_issued": int(
                replay["counters"].get("cancels_issued", 0)),
            "fills": fills, "n_fills": len(fills)}


def robustness_result(primary_D: float, robust_base: float,
                      robust_arm: float) -> dict:
    robust_D = robust_arm - robust_base
    sign = lambda value: 1 if value > 0 else -1 if value < 0 else 0
    return {
        "label": "NO_FILLS_UNTIL_NEXT_GENERATION",
        "zero_model_cancel_baseline_total_cents": robust_base,
        "arm_settled_total_cents": robust_arm,
        "observed_D_cents": robust_D,
        "primary_label": "REFERENCE_FILLS",
        "primary_observed_D_cents": primary_D,
        "sign_reversal": sign(primary_D) != sign(robust_D),
        "a_sign_reversal_blocks_promotion": True,
    }


def run_one_day_arm(day: str, book_path, arm: str, *, n_draws: int,
                    seed: int | None = None, out_dir: Path, book_receipt,
                    score_certifications, params=None) -> dict:
    """Value one arm and its matched null after every input guard passes."""
    if n_draws != DECLARED_N:
        raise SettlementControlRefused(
            f"REFUSED {WRONG_N}: requested {n_draws}, declaration says "
            f"{DECLARED_N}; the floor {FLOOR} is not an alternate count.")
    params, params_pin = frozen_params(params)
    if arm not in params["arms"]:
        raise SettlementControlRefused(
            f"REFUSED {UNKNOWN_ARM}: {arm!r} not in {sorted(params['arms'])}.")
    cert_docs, cert_provenance = _load_certifications(score_certifications)
    delta_bounds = certified_delta_bounds(cert_docs)
    input_verification = R.verify_run_inputs(params)
    module, cite = R.import_be_cascade(params)
    book = module.load(Path(book_path))
    book_sha = book["source_sha256"]
    derived_seed = R.seed_for(book_sha, arm)
    if seed is not None and seed != derived_seed:
        raise SettlementControlRefused(
            f"REFUSED {SEED_NOT_DERIVED}: supplied {seed}, derived "
            f"{derived_seed} from the loaded book digest and arm.")
    seed = derived_seed
    receipt_evidence = verify_book_receipt(book_receipt, book_sha, day)
    per_book = BEN.per_book_guard(book, delta_bounds)

    spec = params["arms"][arm]
    theta = spec["theta"]
    rows = module.arm_stream(book, spec["head"])
    baseline_scores = module.flagged_stream(book["rows"], [])
    base_replay = replay_with_fill_model(
        module, book, baseline_scores, 0.5, "REFERENCE_FILLS")
    arm_replay = replay_with_fill_model(
        module, book, rows, theta, "REFERENCE_FILLS")
    robust_base_replay = replay_with_fill_model(
        module, book, baseline_scores, 0.5,
        "NO_FILLS_UNTIL_NEXT_GENERATION")
    robust_arm_replay = replay_with_fill_model(
        module, book, rows, theta, "NO_FILLS_UNTIL_NEXT_GENERATION")

    slugs = sorted({row["slug"] for row in book["rows"]})
    winner_source = R.winner_source(required_slugs=slugs)
    winners = winner_source["winners"]
    base_total = settled_total(base_replay["fills"], winners)
    arm_total = settled_total(arm_replay["fills"], winners)
    observed_D = arm_total - base_total
    robust = robustness_result(
        observed_D,
        settled_total(robust_base_replay["fills"], winners),
        settled_total(robust_arm_replay["fills"], winners))

    above = [row for row in rows if float(row["score"]) >= theta]
    arm_cancels = [
        {"slug": row["slug"], "side": row["side"],
         "t": float(row["t"]), "gen": row.get("gen"),
         "ref_gen": int(float(row["gen"]))}
        for row in above]
    demand = ASYM.demand_distinct_reference_generations(arm_cancels)
    pool = MCC.build_pool_from_rows(rows)
    row_index = {(row["slug"], row["side"], float(row["t"])): index
                 for index, row in enumerate(rows)}

    identity = run_identity(
        day, arm, book_sha, n_draws, seed, winner_source["sha256"])
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = out_dir / f"de_settle_ckpt_{day}_{arm}.jsonl"
    state = read_checkpoint(
        checkpoint, identity, day=day, arm=arm,
        n_draws=n_draws, seed=seed,
        winner_sha=winner_source["sha256"],
        baseline_total_cents=base_total)
    if not checkpoint.exists():
        ASYM.append_draw(
            checkpoint,
            {"kind": "HEADER", "identity": identity,
             "protocol": PROTOCOL, "n_draws": n_draws, "seed": seed,
             "day": day, "arm": arm,
             "winner_source_sha256": winner_source["sha256"],
             "baseline_total_cents": base_total})
    done = state["draws"]
    started = time.time()
    for index in range(n_draws):
        if index in done:
            continue
        drawn = MCC.draw_one(pool, demand, random.Random(seed + index))
        flags = MCC.flags_for(drawn, row_index)
        replay = module.replay(
            book, module.flagged_stream(rows, flags), 0.5)
        total = settled_total(replay["fills"], winners)
        draw = {"i": index, "seed": seed + index,
                "n_generations_cancelled": len(drawn),
                "settled_total_cents": total,
                "D": total - base_total, "at_utc": time.time()}
        ASYM.append_draw(checkpoint, draw)
        done[index] = draw

    draws = [done[index] for index in range(n_draws) if index in done]
    if len(draws) < FLOOR:
        raise SettlementControlRefused(
            f"REFUSED {UNDER_SAMPLED}: {len(draws)} draws completed and "
            f"the declared floor is {FLOOR}.")
    if len(draws) != DECLARED_N:
        raise SettlementControlRefused(
            f"REFUSED {WRONG_N}: completed {len(draws)}, declaration says "
            f"{DECLARED_N}.")
    deltas = [draw["D"] for draw in draws]
    n_at_or_beyond = sum(
        1 for value in deltas if abs(value) >= abs(observed_D))
    return {
        "protocol": PROTOCOL, "declaration": DECL.name,
        "day": day, "arm": arm, "book": str(book_path),
        "book_sha256": book_sha, "book_receipt": receipt_evidence,
        "winner_source": {
            key: winner_source.get(key)
            for key in ("path", "sha256", "n_records", "n_closed_records",
                        "n_slugs", "settle_up_cents", "method",
                        "is_final_for_quotation")},
        "params_pin": params_pin, "input_verification": input_verification,
        "score_neutrality_certifications": cert_provenance,
        "score_delta_max_certified": delta_bounds,
        "forward_book_margin_guard": per_book,
        "statistic": (
            "D = settle(arm) - settle(zero-model-cancel baseline), cents"),
        "matched_on": ["distinct_reference_generation_count", "side",
                       "utc_hour"],
        "matching_unit": "DISTINCT_REFERENCE_GENERATIONS",
        "primary_fill_assumption": "REFERENCE_FILLS",
        "zero_model_cancel_baseline_total_cents": base_total,
        "arm_settled_total_cents": arm_total,
        "observed_D_cents": observed_D,
        "robustness_leg": robust,
        "n_generations_cancelled_by_the_arm": sum(demand.values()),
        "n_draws": len(draws), "resumed_from_draw": state["n"],
        "seed": seed,
        "seed_derivation": "de_multiday_gate1_runner.seed_for(book_sha, arm)",
        "null": {"n": len(deltas), "min_D": min(deltas),
                 "max_D": max(deltas),
                 "mean_D": sum(deltas) / len(deltas),
                 "n_at_or_beyond_two_sided": n_at_or_beyond},
        "p_two_sided": (1 + n_at_or_beyond) / (1 + len(deltas)),
        "interval": None, "checkpoint": str(checkpoint),
        "checkpoint_sha256": _sha(checkpoint),
        "elapsed_s": round(time.time() - started, 1),
        "be_module": cite.get("sha256"),
        "producer": {
            "path": str(Path(__file__).resolve()),
            "sha256": _sha(Path(__file__)),
        },
    }


def result_path(out_dir: Path, day: str, arm: str) -> Path:
    return Path(out_dir) / f"de_settle_result_{day.replace('-', '')}_{arm}.json"


def write_result_exclusive(path: Path, result: dict) -> None:
    """Publish a complete result without replacing an earlier result."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("x") as handle:
            handle.write(json.dumps(result, indent=1, default=str) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            raise SettlementControlRefused(
                f"REFUSED {OUTPUT_EXISTS}: {path}.") from None
    finally:
        temporary.unlink(missing_ok=True)


def selftest() -> int:
    import tempfile
    failures = []

    def check(label, condition):
        print(f"  {'PASS' if condition else 'FAIL'}  {label}")
        if not condition:
            failures.append(label)

    good = {
        "protocol": "BE_SCORE_NEUTRALITY_V2_REVIEW173_BAR",
        "verdict": "SUPPORTED_ON_THIS_DAY",
        "identity": {"day": CERTIFICATION_DAY},
        "compared_books": {
            "old": {"sha256": "a" * 64},
            "new": {"sha256": "b" * 64}},
        "comparison_receipts": {
            "old": {"sha256": "c" * 64, "book_sha256": "a" * 64,
                    "builder_commit": BEN.CERTIFICATION_OLD_COMMIT},
            "new": {"sha256": "d" * 64, "book_sha256": "b" * 64,
                    "builder_commit": BEN.CERTIFICATION_NEW_COMMIT}},
        "producer": {"sha256": _sha(HERE / "be_score_neutrality.py")},
        "n_flips_overall": 0,
        "per_arm": {
            arm: {"B_delta_max_abs": index + 0.25,
                  "A_n_flips": 0, "D_reconciles": True,
                  "strength": "STRONG_DENSE_OCCUPANCY_NO_FLIP"}
            for index, arm in enumerate(BEN.arm_heads())}}
    bounds = certified_delta_bounds([good])
    check("certified bounds are read for every frozen arm",
          set(bounds) == set(BEN.arm_heads()))
    bad = json.loads(json.dumps(good))
    bad["verdict"] = "NOT_CERTIFIED_ON_THIS_DAY"
    try:
        certified_delta_bounds([bad])
        check("a non-certification refuses", False)
    except SettlementControlRefused as exc:
        check("a non-certification refuses", BAD_CERTIFICATION in str(exc))
    robust = robustness_result(1.0, 10.0, 9.0)
    check("the declared robustness sign reversal is computed",
          robust["observed_D_cents"] == -1.0
          and robust["sign_reversal"] is True)

    with tempfile.TemporaryDirectory() as temporary_dir:
        root = Path(temporary_dir)
        winner_sha = "b" * 64
        identity = run_identity(
            "2026-09-07", "ARM", "a" * 64, 3, 9, winner_sha)
        checkpoint = root / "checkpoint.jsonl"
        ASYM.append_draw(checkpoint, {"kind": "HEADER",
                         "identity": identity, "protocol": PROTOCOL,
                         "n_draws": 3, "seed": 9,
                         "day": "2026-09-07", "arm": "ARM",
                         "winner_source_sha256": winner_sha,
                         "baseline_total_cents": 10.0})
        ASYM.append_draw(checkpoint, {
            "i": 0, "seed": 9, "settled_total_cents": 11.0, "D": 1.0})
        state = read_checkpoint(
            checkpoint, identity, day="2026-09-07", arm="ARM",
            n_draws=3, seed=9, winner_sha=winner_sha,
            baseline_total_cents=10.0)
        check("a valid checkpoint resumes at its gapless prefix",
              state["n"] == 1 and state["resumed"] is True)
        try:
            read_checkpoint(checkpoint, identity, day="2026-09-07",
                            arm="ARM", n_draws=3, seed=9,
                            winner_sha=winner_sha,
                            baseline_total_cents=12.0)
            check("a resumed checkpoint with a moved baseline refuses", False)
        except SettlementControlRefused as exc:
            check("a resumed checkpoint with a moved baseline refuses",
                  BAD_CHECKPOINT in str(exc))
        try:
            read_checkpoint(checkpoint, identity, day="2026-09-07",
                            arm="ARM", n_draws=3, seed=9,
                            winner_sha="c" * 64,
                            baseline_total_cents=10.0)
            check("a resumed checkpoint with moved winners refuses", False)
        except SettlementControlRefused as exc:
            check("a resumed checkpoint with moved winners refuses",
                  BAD_CHECKPOINT in str(exc))
        check("winner bytes participate in the run identity",
              identity != run_identity(
                  "2026-09-07", "ARM", "a" * 64, 3, 9, "c" * 64))
        ASYM.append_draw(checkpoint, {
            "i": 0, "seed": 9, "settled_total_cents": 11.0, "D": 1.0})
        try:
            read_checkpoint(checkpoint, identity, day="2026-09-07",
                            arm="ARM", n_draws=3, seed=9,
                            winner_sha=winner_sha,
                            baseline_total_cents=10.0)
            check("a duplicate checkpoint draw refuses", False)
        except SettlementControlRefused as exc:
            check("a duplicate checkpoint draw refuses",
                  BAD_CHECKPOINT in str(exc))

        output = root / "result.json"
        write_result_exclusive(output, {"ok": True})
        try:
            write_result_exclusive(output, {"ok": False})
            check("an existing final result cannot be overwritten", False)
        except SettlementControlRefused as exc:
            check("an existing final result cannot be overwritten",
                  OUTPUT_EXISTS in str(exc))

    check("draw count and floor come from the declaration",
          DECLARED_N == _DECLARATION["control"]["n_draws"]
          and FLOOR == _DECLARATION["control"]["floor"])
    check("hyphenated and compact day identities are equivalent",
          _day_key("2026-09-07") == _day_key("20260907"))
    print(f"[de_settlement_control_run] "
          f"{len(failures)} failing checks")
    return 1 if failures else 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selftest", action="store_true")
    parser.add_argument("--day")
    parser.add_argument("--book")
    parser.add_argument("--book-receipt")
    parser.add_argument("--arm", choices=tuple(BEN.arm_heads()))
    parser.add_argument("--n-draws", type=int, default=DECLARED_N)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--out-dir")
    parser.add_argument("--score-certification", action="append")
    parser.add_argument("--params")
    args = parser.parse_args(argv)
    if args.selftest:
        return selftest()
    required = {"day": args.day, "book": args.book,
                "book_receipt": args.book_receipt, "arm": args.arm,
                "out_dir": args.out_dir,
                "score_certification": args.score_certification}
    missing = [name for name, value in required.items() if value is None]
    if missing:
        parser.error("required for a run: " + ", ".join(missing))
    destination = result_path(Path(args.out_dir), args.day, args.arm)
    if destination.exists():
        print(json.dumps({"refused": f"REFUSED {OUTPUT_EXISTS}: "
                                      f"{destination}"}, indent=1))
        return 3
    try:
        supplied_params = (R.load_params(Path(args.params))
                           if args.params else None)
        result = run_one_day_arm(
            args.day, args.book, args.arm, n_draws=args.n_draws,
            seed=args.seed, out_dir=Path(args.out_dir),
            book_receipt=args.book_receipt,
            score_certifications=args.score_certification,
            params=supplied_params)
        write_result_exclusive(destination, result)
    except (RuntimeError, OSError, json.JSONDecodeError) as exc:
        print(json.dumps({"refused": str(exc)}, indent=1))
        return 3
    print(json.dumps({"written": str(destination),
                      "D": result["observed_D_cents"],
                      "p_two_sided": result["p_two_sided"],
                      "robustness_D": result["robustness_leg"]
                          ["observed_D_cents"]}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
