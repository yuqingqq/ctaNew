#!/usr/bin/env python3
"""Emit one P003 settlement point estimate without drawing the null."""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import math
import os
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import de_data_root as DR  # noqa: E402
import de_multiday_gate1_runner as R  # noqa: E402


PROTOCOL = "P003_DE_POINT_ESTIMATE_DAY_V1"
FAMILY = "p003_de_point_estimate_day"
DRIVER_SHA256_AT_IMPORT = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
RESULT_CONTRACT_REFUSAL = "POINT_ESTIMATE_RESULT_CONTRACT_VIOLATION"
NULL_FIELDS = ("Z", "p_location", "null_mean", "null_sd")


def _stamp(when: datetime.datetime) -> str:
    return when.strftime("%Y%m%dT%H%M%SZ")


def _latency_tag(value_ms: float) -> str:
    value = float(value_ms)
    text = str(int(value)) if value.is_integer() else format(value, ".12g")
    return f"L{text.replace('.', 'p')}ms"


def artifact_name(day: str, value_ms: float, when: datetime.datetime) -> str:
    return (f"{FAMILY}_{day.replace('-', '')}_{_latency_tag(value_ms)}__"
            f"{_stamp(when)}.json")


def _sha(path: Path) -> str:
    return R.sha256_streamed(Path(path))


def _refusal_name(exc: Exception) -> str:
    return str(exc).split(":")[0].replace("REFUSED ", "")


def prior_artifacts(output_dir: Path, day: str, value_ms: float) -> list[dict]:
    """Resolve every prior artifact for the same (day, latency)."""
    pattern = (f"{FAMILY}_{day.replace('-', '')}_{_latency_tag(value_ms)}__"
               "*.json")
    candidates = []
    for path in sorted(Path(output_dir).glob(pattern)):
        try:
            doc = json.loads(path.read_text())
        except (OSError, ValueError) as exc:
            raise R.RunnerRefused(
                f"REFUSED POINT_ESTIMATE_PRIOR_UNREADABLE: {path.name}: "
                f"{exc}") from exc
        if doc.get("protocol") != PROTOCOL or doc.get("day") != day:
            raise R.RunnerRefused(
                f"REFUSED POINT_ESTIMATE_PRIOR_IDENTITY_MISMATCH: "
                f"{path.name} matches the family name but declares protocol "
                f"{doc.get('protocol')!r} and day {doc.get('day')!r}.")
        top_latency = ((doc.get("placement_latency") or {})
                       .get("L_place_ms"))
        if (isinstance(top_latency, bool)
                or not isinstance(top_latency, (int, float))
                or not math.isfinite(float(top_latency))
                or float(top_latency) != float(value_ms)):
            raise R.RunnerRefused(
                f"REFUSED POINT_ESTIMATE_PRIOR_LATENCY_MISMATCH: "
                f"{path.name} is in the {_latency_tag(value_ms)} family but "
                f"its top-level L_place_ms is {top_latency!r}.")
        consistency = "PLACEMENT_LATENCY_AGREES"
        try:
            R.assert_one_placement_latency(doc)
        except R.RunnerRefused as exc:
            consistency = _refusal_name(exc)
        candidates.append({
            "path": str(path),
            "sha256": _sha(path),
            "day": day,
            "L_place_ms": float(value_ms),
            "target_placement_latency_consistency": consistency,
        })
    return candidates


def prior_artifact(output_dir: Path, day: str, value_ms: float) -> dict | None:
    """Return the newest prior, while the emitter absorbs every sibling."""
    candidates = prior_artifacts(output_dir, day, value_ms)
    return candidates[-1] if candidates else None


def assert_driver_source(where: str) -> dict:
    """Bind the outer artifact to this committed driver and the runner."""
    path = Path(__file__).resolve()
    now = hashlib.sha256(path.read_bytes()).hexdigest()
    if now != DRIVER_SHA256_AT_IMPORT:
        raise R.RunnerRefused(
            f"REFUSED POINT_ESTIMATE_DRIVER_CHANGED_DURING_RUN at {where}: "
            f"imported {DRIVER_SHA256_AT_IMPORT[:16]} and now "
            f"{now[:16]}.")
    committed = R.carrying_commit_block(path)
    if not committed["producing_code_is_the_committed_bytes"]:
        raise R.RunnerRefused(
            f"REFUSED POINT_ESTIMATE_DRIVER_NOT_COMMITTED at {where}: "
            "the point-estimate artifact builder is not the file held by "
            "HEAD. Commit the driver before emitting a result-bearing day.")
    runner = R.assert_source_unchanged(where, fixture=False)
    return {
        "driver": {
            **committed,
            "sha256_at_import": DRIVER_SHA256_AT_IMPORT,
            "sha256_at_emit": now,
            "unchanged_during_run": now == DRIVER_SHA256_AT_IMPORT,
        },
        "runner": runner,
    }


def assert_point_estimate_result(result: dict) -> dict:
    """Require settlement values and named no-null statuses before emit."""
    problems = []
    if result.get("run_mode") != "POINT_ESTIMATE":
        problems.append(f"run_mode={result.get('run_mode')!r}")
    try:
        R.assert_one_placement_latency(result)
    except R.RunnerRefused as exc:
        problems.append(_refusal_name(exc))
    ledger = result.get("decision_ledger") or {}
    if ledger.get("run_mode") != "POINT_ESTIMATE":
        problems.append(f"decision_ledger.run_mode={ledger.get('run_mode')!r}")

    valued_arms = []
    for arm_day in result.get("per_day_sealed_artifacts") or []:
        if arm_day.get("status") != "OK_POINT_ESTIMATE":
            continue
        arm = arm_day.get("arm")
        economic = arm_day.get("economic") or {}
        settlement = arm_day.get("economic_settlement") or {}
        d_settle = settlement.get("D_E_settle")
        if (isinstance(d_settle, bool)
                or not isinstance(d_settle, (int, float))
                or not math.isfinite(float(d_settle))):
            problems.append(f"{arm}.economic_settlement.D_E_settle absent")
            continue
        for label, block in (("economic", economic),
                             ("economic_settlement", settlement)):
            for field in NULL_FIELDS:
                if block.get(field) != R.NULL_NOT_DRAWN_STATUS:
                    problems.append(
                        f"{arm}.{label}.{field}={block.get(field)!r}")
            draws = block.get("null_draws_summary") or {}
            if draws.get("n") != R.NULL_NOT_DRAWN_STATUS:
                problems.append(
                    f"{arm}.{label}.null_draws_summary.n="
                    f"{draws.get('n')!r}")
        valued_arms.append(arm)
    if not valued_arms:
        problems.append("no OK_POINT_ESTIMATE arm carries ruled settlement P&L")
    n_admissible = result.get("n_admissible_arms")
    if n_admissible != len(valued_arms):
        problems.append(
            f"n_admissible_arms={n_admissible!r} but valued arms="
            f"{valued_arms}")
    if problems:
        raise R.RunnerRefused(
            f"REFUSED {RESULT_CONTRACT_REFUSAL}: {problems}. The point-"
            "estimate family claims ruled settlement P&L with no null; it "
            "must not publish a diagnostic-only day or a partial statistics "
            "block.")
    return {"status": "POINT_ESTIMATE_RESULT_CONTRACT_SATISFIED",
            "valued_arms": valued_arms,
            "null_status": R.NULL_NOT_DRAWN_STATUS}


def write_artifact(path: Path, payload: dict) -> dict:
    """Validate the serialized document before atomically publishing it."""
    path = Path(path)
    if path.exists():
        raise R.RunnerRefused(
            f"REFUSED POINT_ESTIMATE_OUTPUT_EXISTS: {path} already exists; "
            "a result artifact is never overwritten.")
    payload["placement_latency_consistency"] = (
        R.assert_one_placement_latency(payload))
    encoded = json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n"
    parsed = json.loads(encoded)
    R.assert_one_placement_latency(parsed)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        mode="w", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp",
        delete=False)
    temporary = Path(handle.name)
    try:
        with handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        reread = json.loads(temporary.read_text())
        R.assert_one_placement_latency(reread)
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return {"path": str(path), "sha256": _sha(path),
            "bytes": path.stat().st_size}


def run(day: str, book: Path, output_dir: Path) -> dict:
    started = time.time()
    book = Path(book).resolve()
    output_dir = Path(output_dir).resolve()
    if not output_dir.is_dir():
        raise R.RunnerRefused(
            f"REFUSED POINT_ESTIMATE_OUTPUT_NOT_DIRECTORY: {output_dir}")

    # Refuse uncommitted or drifting producers before paying for the replay,
    # then check the same identities again immediately before publication.
    assert_driver_source("the point-estimate preflight")
    params = R.load_params()
    result = R.run_day(
        day, book, params=params, fixture=False,
        n_days_complete=R.days_complete_now(params)["n_days_complete"],
        point_estimate=True, ledger_anchor=output_dir,
        before_work=lambda: R.selftest(quiet=True, offline=False),
    )
    result_contract = assert_point_estimate_result(result)
    placement = result.get("placement_latency") or {}
    value_ms = placement.get("L_place_ms")
    if value_ms is None:
        raise R.RunnerRefused(
            "REFUSED POINT_ESTIMATE_PLACEMENT_LATENCY_ABSENT: run_day "
            "returned no L_place_ms.")

    emitted_at = datetime.datetime.now(datetime.timezone.utc)
    path = output_dir / artifact_name(day, value_ms, emitted_at)
    prior = prior_artifacts(output_dir, day, value_ms)
    supersedes = prior[-1] if prior else None
    also_supersedes = prior[:-1]
    source_identity = assert_driver_source("the point-estimate emit")
    receipt_path = Path(result["reference_book"]["builder_receipt"])
    payload = {
        "protocol": PROTOCOL,
        "what_this_is": (
            "the ruled settlement P&L at the book's declared placement "
            "latency, without a null: S4 was skipped, so no Z, p or "
            "interval exists and nothing here implies significance"),
        "status": result.get("status"),
        "day": day,
        "run_mode": result["run_mode"],
        "result_contract": result_contract,
        "placement_latency": placement,
        "book": {
            "path": str(book),
            "sha256": result["reference_book"]["sha256"],
            "builder_receipt": {
                "path": str(receipt_path),
                "sha256": _sha(receipt_path),
            },
        },
        "supersedes": supersedes,
        "also_supersedes": also_supersedes,
        "supersession_scope": {
            "n_prior_artifacts_for_day_and_latency": len(prior),
            "all_prior_artifacts_are_named": len(prior)
            == (1 if supersedes else 0) + len(also_supersedes),
        },
        "supersession_note": (
            "every prior artifact remains unedited; supersedes names the "
            "newest pair and also_supersedes absorbs any earlier unlinked "
            "sibling, including an abandoned attempt" if supersedes else
            "no prior artifact exists for this (day, L_place_ms) family"),
        "source_identity": source_identity,
        "invocation": {
            "argv": list(sys.argv),
            "interpreter": sys.executable,
            "exclusive_lock_verified_by": (
                "run_day.assert_lock_form_at_runtime and "
                "assert_real_day_has_the_lock"),
        },
        "day_run": result,
        "as_of": emitted_at.isoformat(),
        "elapsed_seconds": time.time() - started,
    }
    artifact = write_artifact(path, payload)
    return {"artifact": artifact, "day": day, "L_place_ms": value_ms,
            "status": result.get("status"), "supersedes": supersedes}


def selftest() -> int:
    with tempfile.TemporaryDirectory(prefix="de_point_estimate_") as tmp:
        root = Path(tmp)
        old = root / artifact_name(
            "2026-09-05", 250.0,
            datetime.datetime(2026, 9, 8, 3, 47, 39,
                              tzinfo=datetime.timezone.utc))
        old_payload = {
            "protocol": PROTOCOL, "day": "2026-09-05",
            "placement_latency": {"L_place_ms": 250.0},
            "day_run": {"placement_latency": {"L_place_ms": 0.0}},
        }
        old.write_text(json.dumps(old_payload))
        prior = prior_artifact(root, "2026-09-05", 250.0)
        if (prior is None or prior["sha256"] != _sha(old)
                or prior["target_placement_latency_consistency"]
                != R.PLACEMENT_LATENCY_DISAGREES):
            raise SystemExit("point-estimate selftest: prior resolution failed")

        emitted = root / artifact_name(
            "2026-09-05", 250.0,
            datetime.datetime(2026, 9, 8, 6, 30,
                              tzinfo=datetime.timezone.utc))
        good = {
            "protocol": PROTOCOL, "day": "2026-09-05",
            "placement_latency": {"L_place_ms": 250.0},
            "day_run": {"placement_latency": {"L_place_ms": 250.0}},
            "supersedes": prior,
        }
        wrote = write_artifact(emitted, good)
        if wrote["sha256"] != _sha(emitted):
            raise SystemExit("point-estimate selftest: emitted digest mismatch")

        bad = root / "bad.json"
        inconsistent = json.loads(json.dumps(good))
        inconsistent["day_run"]["placement_latency"]["L_place_ms"] = 0.0
        refusal = None
        try:
            write_artifact(bad, inconsistent)
        except R.RunnerRefused as exc:
            refusal = _refusal_name(exc)
        if refusal != R.PLACEMENT_LATENCY_DISAGREES or bad.exists():
            raise SystemExit(
                "point-estimate selftest: inconsistent artifact was not refused")

        status = R.NULL_NOT_DRAWN_STATUS
        result_fixture = {
            "run_mode": "POINT_ESTIMATE",
            "n_admissible_arms": 1,
            "placement_latency": {"L_place_ms": 250.0},
            "decision_ledger": {"run_mode": "POINT_ESTIMATE"},
            "per_day_sealed_artifacts": [{
                "arm": "A", "status": "OK_POINT_ESTIMATE",
                "economic": {
                    **{field: status for field in NULL_FIELDS},
                    "null_draws_summary": {"n": status},
                },
                "economic_settlement": {
                    "D_E_settle": 12.5,
                    **{field: status for field in NULL_FIELDS},
                    "null_draws_summary": {"n": status},
                },
            }],
        }
        asserted = assert_point_estimate_result(result_fixture)
        missing = json.loads(json.dumps(result_fixture))
        missing["per_day_sealed_artifacts"][0].pop("economic_settlement")
        refusal = None
        try:
            assert_point_estimate_result(missing)
        except R.RunnerRefused as exc:
            refusal = _refusal_name(exc)
        if (asserted["valued_arms"] != ["A"]
                or refusal != RESULT_CONTRACT_REFUSAL):
            raise SystemExit(
                "point-estimate selftest: result contract did not fire")
    print("[de_point_estimate_day] PASS -- 4 checks")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("day", nargs="?")
    parser.add_argument("book", nargs="?", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args()
    if args.selftest:
        return selftest()
    if not args.day or args.book is None:
        parser.error("day and book are required unless --selftest is used")
    output_dir = args.output_dir or (
        Path(DR.resolve()["data_root"]) / "pm_5min" / "derived")
    print(json.dumps(run(args.day, args.book, output_dir), indent=2,
                     sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
