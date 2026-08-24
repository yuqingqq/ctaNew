"""Point-in-time Tier-1 loader and the deliberately leaky replay canary.

This is research validation infrastructure.  Normal consumers receive only a
``StateView``.  The paired ``EventTimeView`` is available through the explicit
canary harness so every replay can prove that knowledge-time truncation is
actually wired.

Run the focused checks with::

    python3 -m live.pm_research.replay_canary --selftest

Run a completed Tier-1 day with::

    python3 -m live.pm_research.replay_canary \
        --day 2026-08-20 --coin btc
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from dataclasses import asdict, dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from live.pm_research.coverage_ledger import (
    SourceRegistry,
    canonical_json,
    load_source_registry,
)
from live.pm_research.da_state import (
    DAState,
    Duration,
    Known,
    KnownFactory,
    KnowledgeTime,
    SourceProfile,
    Transport,
    Unavailable,
)
from live.pm_research.tier1_pipeline import (
    DISTILLER_VERSION,
    COIN_SYMBOL,
    DEFAULT_OUTPUT_ROOT,
    read_partition,
    write_partition,
)


CANARY_VERSION = "leak_canary_v2_r7"
REFERENCE_DELTA_PP = 0.5

# --- R-7 licence: VACATED (R-89), RETAINED UNDER ADJUDICATION ----------------
# THIS DICT IS NOT LIVE.  R-7's licence is dead and nothing may cite it as
# authority.  The amendment it once licensed SURVIVES on a different foundation
# (R-94, ordering -- see `classify`), which is not distributional and cannot
# drift, so NOTHING below is the reason the amendment is correct any more.
#
# It is not deleted because its ONLY consumer is `r7_drift_check`, and whether
# that check is extended, narrowed or retired is before DE (the coordinator
# recused: R-87 ordered it extended, R-89 killed its subject, both theirs).
# Deleting this dict would retire the check, i.e. would decide DE's question.
# So the basis is marked dead and left standing. Vacating a rule is not
# self-executing -- this annotation IS the sweep that R-94's class demands.
R7_LICENSE_STATUS = "VACATED_R89_NOT_LIVE_PENDING_DE_ON_R87"
R7_LICENSE = {
    "ruling": "R-7",
    "granted": "2026-08-23",
    "statistic": "decision_disagreements per coin-day",
    "fit": "Poisson",
    "lambda": 1.857,
    "variance_observed": 1.363,
    "n_coin_days": 14,
    "source": "2026-08-20 and 2026-08-21, all seven coins, scratch-root enumeration",
    "p_zero": 0.156,
    "expected_invalid_coin_days_per_7_coin_day": 1.09,
}
# Escalation thresholds for condition 4.  These BOUND the licence; they are not
# a verdict bar on any research hypothesis.
# R-99: BOTH distributional arms are RETIRED. They policed the Poisson fit
# (lambda 1.857, variance 1.363) that R-89 VACATED, so they tested a bar with
# no force. `R7_DRIFT_MIN_COIN_DAYS` went with them: it existed so a
# DISTRIBUTIONAL check would abstain rather than guess at small n. The
# surviving arm is an INVARIANT, not an estimate -- one reclassified coin-day
# carrying a nonzero delta is a construction violation at n=1 -- so there is
# nothing left to abstain from.


def _sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _manifest_path(root: Path, dataset: str, day: date, coin: str) -> Path:
    # Partition addresses carry the distiller generation (R-10/R-12); this
    # duplicated the old layout and silently pointed at the superseded one.
    return (
        root
        / dataset
        / f"day={day.isoformat()}"
        / f"coin={coin}"
        / f"distiller={DISTILLER_VERSION}"
        / "manifest.json"
    )


def _load_manifest(root: Path, dataset: str, day: date, coin: str) -> dict[str, Any]:
    path = _manifest_path(root, dataset, day, coin)
    doc = json.loads(path.read_text())
    declared = str(doc.get("manifest_hash", ""))
    payload = dict(doc)
    payload.pop("manifest_hash", None)
    actual = _sha256_bytes(canonical_json(payload))
    if not declared or declared != actual:
        raise RuntimeError(f"manifest hash mismatch {path}")
    return doc


def _factory(registry: SourceRegistry) -> KnownFactory:
    profiles = [
        SourceProfile(
            source=profile.source,
            has_recv_ns=profile.has_recv_ns,
            transport=Transport(profile.transport),
            clock_err=Duration(profile.clock_err_ns),
        )
        for profile in registry.profiles.values()
    ]
    return KnownFactory(profiles)


def _observed_row(
    row: Mapping[str, Any],
    *,
    value: Any,
    event_ns: int,
    registry: SourceRegistry,
    factory: KnownFactory,
) -> Known[Any]:
    source = str(row["source"])
    profile = registry.profile(source)
    if row.get("t_known_prov") != "OBSERVED":
        raise RuntimeError(f"Tier-1 row is not OBSERVED: {source}")
    if row.get("source_profile_hash") != registry.registry_hash:
        raise RuntimeError(f"source-profile hash mismatch on {source} row")
    if int(row["t_known_err_ns"]) != profile.clock_err_ns:
        raise RuntimeError(f"clock-error mismatch on {source} row")
    return factory.observed(
        source,
        value,
        event_ns=event_ns,
        recv_ns=int(row["t_known_ns"]),
        seq=int(row["seq"]),
        provenance={
            "source_profile_hash": registry.registry_hash,
            "source_file_id": row.get("source_file_id"),
        },
    )


@dataclass(frozen=True, slots=True)
class LoadedReplay:
    state: DAState
    windows: tuple[dict[str, Any], ...]
    input_manifests: tuple[str, ...]
    source_profile_hash: str


def load_replay(
    *,
    output_root: Path,
    day: date,
    coin: str,
    allow_partial: bool = False,
) -> LoadedReplay:
    """Load adjacent TWAP, target windows and coverage through verified manifests."""
    if coin not in COIN_SYMBOL:
        raise ValueError(f"unsupported coin {coin!r}")
    registry = load_source_registry()
    factory = _factory(registry)
    state = DAState()
    manifest_hashes: list[str] = []

    for selected_day in (day - timedelta(days=1), day, day + timedelta(days=1)):
        manifest = _load_manifest(output_root, "twap", selected_day, coin)
        if manifest.get("partial") and not allow_partial:
            raise RuntimeError("replay canary refuses partial TWAP input")
        manifest_hashes.append(str(manifest["manifest_hash"]))
        table = read_partition(
            output_root,
            "twap",
            selected_day,
            coin,
            allow_partial=allow_partial,
        )
        for row in table.to_pylist():
            item = _observed_row(
                row,
                value=float(row["value"]),
                event_ns=int(row["t_event_ms"]) * 1_000_000,
                registry=registry,
                factory=factory,
            )
            state.ingest(f"twap{int(row['window_s'])}[{row['symbol']}]", item)

    windows_manifest = _load_manifest(output_root, "windows", day, coin)
    coverage_manifest = _load_manifest(output_root, "coverage", day, coin)
    for manifest in (windows_manifest, coverage_manifest):
        if manifest.get("partial") and not allow_partial:
            raise RuntimeError("replay canary refuses partial target input")
        manifest_hashes.append(str(manifest["manifest_hash"]))

    windows = tuple(
        read_partition(
            output_root, "windows", day, coin, allow_partial=allow_partial
        ).to_pylist()
    )
    coverage_rows = read_partition(
        output_root, "coverage", day, coin, allow_partial=allow_partial
    ).to_pylist()
    for row in coverage_rows:
        item = _observed_row(
            row,
            value=dict(row),
            # Coverage cannot exist before the interval it describes has ended.
            event_ns=int(row["target_end_ns"]),
            registry=registry,
            factory=factory,
        )
        state.ingest_coverage(str(row["field"]), item)

    return LoadedReplay(
        state=state,
        windows=windows,
        input_manifests=tuple(manifest_hashes),
        source_profile_hash=registry.registry_hash,
    )


@dataclass(frozen=True, slots=True)
class LeakCanary:
    metric: str
    value_knowledge_time: float
    value_event_time: float
    delta: float
    n: int
    knowledge_hits: int
    event_hits: int
    decision_disagreements: int
    event_only_boundary_reads: int
    skipped: int
    status: str
    reference_delta_pp: float
    refusal_counts: Mapping[str, Mapping[str, int]]
    r7_reclassified: bool = False


def _known_value(item: Known[Any] | Unavailable) -> float | None:
    return float(item.value) if isinstance(item, Known) else None


# R-95 — ONE NAME COVERED TWO MECHANISMS AND THAT IS WHAT LET A WRONG MODEL FORM.
# `INVALID_UNBOUND_GUARD` was returned by two arms that fail for unrelated
# reasons: a guard that was never WIRED, and counters that CONTRADICT each other.
# The coordinator's rationale for R-7 attached itself to the first when the
# amendment actually touched neither, and the conflation survived several rulings
# because the status could be discussed as though it had a single cause.
#
# The old name is retained as a READ alias only: receipts already on disk carry
# it, they are immutable under R-28, and a reader that cannot parse them would
# turn a naming fix into data loss. Nothing EMITS it.
INVALID_UNWIRED_GUARD = "INVALID_UNWIRED_GUARD"
INVALID_COUNTER_INCONSISTENT = "INVALID_COUNTER_INCONSISTENT"
LEGACY_INVALID = "INVALID_UNBOUND_GUARD"          # read-only; pre-R-95 receipts
INVALID_STATUSES = frozenset(
    {INVALID_UNWIRED_GUARD, INVALID_COUNTER_INCONSISTENT, LEGACY_INVALID})


def classify(
    event_only: int, disagreements: int, delta: float
) -> tuple[str, bool]:
    """The canary's status rule. Returns (status, reclassified).

    The amendment stands on R-94 (ordering). R-7's licence is VACATED (R-89)
    and is not authority for anything here.

    Pure and separate from the measurement so the rule can be tested directly,
    including the branch that the measurement cannot construct.

    `event_only == 0` means the leaky twin never read past a knowledge boundary,
    i.e. the guard is not wired.  That arm is UNCHANGED by R-7 and stays fatal.

    `disagreements == 0` used to be fatal too, and it pre-empted the branch
    below that already handled the case correctly.  The reason is ORDERING, not
    frequency (R-94; the distributional basis R-7 granted it on is VACATED and
    may not be cited): pre-amendment, zero disagreements with zero harm was
    INVALID while five disagreements with the SAME zero harm was fine, so the
    strictly safer observation was punished more harshly.  The rule was
    non-monotone in its own evidence.  That is a defect in the rule's shape, no
    distribution enters it, and G=2 cannot break it.

    Conditions 1 and 2 are why `delta` is re-checked here rather than
    assumed: the status must assert a MEASURED zero, and any nonzero delta stays
    INVALID.  `disagreements == 0` forces knowledge_hits == event_hits and hence
    delta == 0 exactly, so that second branch is a fail-closed consistency check
    which should never fire; if it does, the counters disagree with each other
    and INVALID is the right answer.
    """
    if event_only == 0:
        return "INVALID_UNWIRED_GUARD", False
    if disagreements == 0:
        if math.isclose(delta, 0.0, abs_tol=1e-15):
            return "BOUND_ZERO_SCORE_DELTA", True
        return "INVALID_COUNTER_INCONSISTENT", False
    if math.isclose(delta, 0.0, abs_tol=1e-15):
        # A score delta can cancel even when the selected states differ.  That
        # is a review flag, while event-only reads prove the guard is wired.
        return "BOUND_ZERO_SCORE_DELTA", False
    return "VALID_GUARD_BITES", False


def measure_leak_canary(
    state: DAState,
    windows: Iterable[Mapping[str, Any]],
    *,
    coin: str,
    spec_snapshot: str,
    refuse_k: float = 1.0,
) -> LeakCanary:
    """Twin-run S60(T) >= S60(t0) through knowledge and event-time views."""
    field = f"twap60[{COIN_SYMBOL[coin]}]"
    harness = state.canary_harness(
        refuse_k=refuse_k, spec_snapshot=spec_snapshot
    )
    n = knowledge_hits = event_hits = disagreements = event_only = skipped = 0
    for window in sorted(windows, key=lambda row: int(row["window_start_s"])):
        if not window.get("closed") or window.get("winner_up") is None:
            skipped += 1
            continue
        t0_ns = int(window["window_start_s"]) * 1_000_000_000
        end_ns = int(window["window_end_s"]) * 1_000_000_000
        kt0_view, et0_view = harness.views_at(t0_ns)
        kt_end_view, et_end_view = harness.views_at(end_ns)
        kt0_item = kt0_view.get(field)
        kt_end_item = kt_end_view.get(field)
        et0_item = et0_view.get(field)
        et_end_item = et_end_view.get(field)
        values = tuple(
            _known_value(item)
            for item in (kt0_item, kt_end_item, et0_item, et_end_item)
        )
        if any(value is None for value in values):
            skipped += 1
            continue
        kt0, kt_end, et0, et_end = values
        assert kt0 is not None and kt_end is not None
        assert et0 is not None and et_end is not None
        pred_knowledge = kt_end >= kt0
        pred_event = et_end >= et0
        winner_up = bool(window["winner_up"])
        n += 1
        knowledge_hits += int(pred_knowledge == winner_up)
        event_hits += int(pred_event == winner_up)
        disagreements += int(pred_knowledge != pred_event)
        for boundary_ns, selected in (
            (t0_ns, et0_item),
            (end_ns, et_end_item),
        ):
            if isinstance(selected, Known) and selected.known_hi_ns(refuse_k) > boundary_ns:
                event_only += 1

    if n == 0:
        raise RuntimeError("leak canary has no paired resolved windows")
    value_knowledge = knowledge_hits / n
    value_event = event_hits / n
    delta = value_event - value_knowledge
    # --- status, as amended by R-7 -----------------------------------------
    # `event_only == 0` still means the leaky twin never read past a knowledge
    # boundary, i.e. the guard is not wired.  That arm is UNCHANGED and fatal.
    #
    # `disagreements == 0` used to be fatal too, and it pre-empted the branch
    # below that already handled it correctly.  The reason is ORDERING, not
    # frequency (R-94; R-7's distributional basis is VACATED and uncitable):
    # zero disagreements with zero harm was INVALID while five disagreements
    # with the SAME zero harm was fine -- the strictly safer observation was
    # punished more harshly, so the rule was non-monotone in its own evidence.
    # No distribution enters that, and G=2 cannot break it.
    #
    # Conditions 1 and 2 are why the delta is re-checked here rather than
    # assumed: the status must assert a MEASURED zero, and any nonzero delta
    # stays INVALID.  Note that `disagreements == 0` forces
    # knowledge_hits == event_hits and therefore delta == 0 exactly, so the
    # second branch below is a fail-closed consistency check that should never
    # fire -- if it ever does, the counters disagree with each other and INVALID
    # is the right answer.
    status, r7_reclassified = classify(event_only, disagreements, delta)
    refusals = {
        name: {
            "n_admitted": count.n_admitted,
            "n_refused_within_err": count.n_refused_within_err,
            "n_unavailable_upstream": count.n_unavailable_upstream,
            "worst_case_peek_blocked_ns": count.worst_case_peek_blocked.ns,
        }
        for name, count in sorted(
            state.view(
                # The ledger is shared; the timestamp does not affect snapshot.
                # Use zero instead of consulting a wall clock.
                KnowledgeTime(0)
            ).refusals().items()
        )
    }
    return LeakCanary(
        metric="winner_reproduction:S60(T)>=S60(t0)",
        value_knowledge_time=value_knowledge,
        value_event_time=value_event,
        delta=delta,
        n=n,
        knowledge_hits=knowledge_hits,
        event_hits=event_hits,
        decision_disagreements=disagreements,
        event_only_boundary_reads=event_only,
        skipped=skipped,
        status=status,
        reference_delta_pp=REFERENCE_DELTA_PP,
        refusal_counts=refusals,
        r7_reclassified=r7_reclassified,
    )


def r7_drift_check(
    observations: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Runtime witness for the amendment's CONSTRUCTION (R-99, one arm).

    **This no longer checks a distribution.**  R-7 licensed the amendment on a
    Poisson fit; R-89 VACATED that licence, and R-94 re-founded the amendment on
    an ORDERING argument that uses no distribution and cannot be broken at G=2.
    Two arms policed the dead fit -- lambda tolerance and variance/mean
    Poisson-likeness -- and R-99 retired both: a check against a vacated bar
    tests nothing and, worse, mints the claim of an authority that no longer
    exists.

    What survives is the arm that polices the CONSTRUCTION: a reclassified
    coin-day must carry a MEASURED zero delta.  That is an invariant, not a
    tolerance, and it holds at any n.

    The rule's ORDERING property -- the thing R-94 actually re-founded it on --
    cannot drift with data but CAN be silently reintroduced by an edit, so it is
    policed statically by `selftest()` rather than here (R-99, DE's reasoning).

    Each observation needs `decision_disagreements`, `delta` and `status`.
    """
    counts = [int(row["decision_disagreements"]) for row in observations]
    n = len(counts)
    reasons: list[str] = []
    # An invariant, not a tolerance: a reclassified coin-day with
    # a nonzero delta would mean impact was forgiven, which the amendment
    # forbids.  It cannot happen by construction, so if it appears the
    # construction is wrong and the coordinator must hear about it.
    forgiven = [
        row for row in observations
        if row.get("r7_reclassified") and not math.isclose(
            float(row.get("delta", 0.0)), 0.0, abs_tol=1e-15
        )
    ]
    if forgiven:
        reasons.append(
            f"{len(forgiven)} reclassified coin-day(s) carry a NONZERO delta -- "
            f"R-7 condition 2 violated"
        )
    return {
        # NOT "WITHIN_LICENCE": there is no live licence to be within, and the
        # old string minted a claim of vacated authority into every receipt.
        "verdict": "ESCALATE_TO_COORDINATOR" if reasons else "CONSTRUCTION_INTACT",
        "polices": "construction only (R-99): reclassified coin-days carry a measured zero delta",
        "ordering_property": "policed statically by selftest(), not by this function",
        "n_coin_days": n,
        "disagreements_observed": counts,
        "reclassified_coin_days": sum(
            1 for row in observations if row.get("r7_reclassified")
        ),
        "reasons": reasons,
        "licence_status": R7_LICENSE_STATUS,
    }


def _report_dir(root: Path, day: date, coin: str, *, partial: bool) -> Path:
    """Path carries the canary VERSION, mirroring the health record's layout.

    The report binds `canary_code_sha256` into its own `source_digest` while
    living at a fixed path, so before R-7 any edit to this file made every
    already-written report raise `canary merge-never-overwrite` for ever -- the
    defect class recorded in DA_PIPELINE_OUTAGE_DIAGNOSIS_2026-08-23.md 1.4a.
    Versioning the path means a semantics change produces a NEW generation of
    artifacts and leaves the old ones valid and readable, which is also what
    R-7 condition 3 wants: the pre-amendment verdicts stay on disk as the other
    arm rather than being overwritten by the amended ones.
    """
    dataset = "canary_partial" if partial else "canary"
    return (
        root
        / dataset
        / f"day={day.isoformat()}"
        / f"coin={coin}"
        / f"canary={CANARY_VERSION}"
    )


def write_report(
    *,
    output_root: Path,
    day: date,
    coin: str,
    canary: LeakCanary,
    input_manifests: Sequence[str],
    source_profile_hash: str,
    partial: bool = False,
) -> dict[str, Any]:
    source = {
        "canary_version": CANARY_VERSION,
        "day": day.isoformat(),
        "coin": coin,
        "input_manifests": list(input_manifests),
        "source_profile_hash": source_profile_hash,
        "partial": partial,
        "canary_code_sha256": _sha256_file(Path(__file__)),
        "da_state_code_sha256": _sha256_file(
            Path(__file__).with_name("da_state.py")
        ),
    }
    source_digest = _sha256_bytes(canonical_json(source))
    # R-10/R-16: address by the SOURCE DIGEST, which covers both the canary code
    # and its inputs.  Versioning by CANARY_VERSION alone was not enough and I
    # proved it the hard way: editing this file mid-run left an existing report
    # whose source_digest no longer matched, and the run died on
    # `canary merge-never-overwrite` at a hand-maintained version directory.  A
    # hand-maintained version is a promise to remember; a content address is not.
    path = _report_dir(output_root, day, coin, partial=partial) / (
        f"report={source_digest}.json"
    )
    report: dict[str, Any] = {
        **source,
        "source_digest": source_digest,
        "canary": asdict(canary),
    }
    report["report_hash"] = _sha256_bytes(canonical_json(report))
    if path.exists():
        existing = json.loads(path.read_text())
        if existing.get("source_digest") != source_digest:
            raise RuntimeError(f"canary merge-never-overwrite at {path}")
        declared = str(existing.get("report_hash", ""))
        payload = dict(existing)
        payload.pop("report_hash", None)
        if declared != _sha256_bytes(canonical_json(payload)):
            raise RuntimeError(f"canary report hash mismatch at {path}")
        return existing
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix="report-", suffix=".json.tmp", dir=path.parent, delete=False
        ) as handle:
            temporary = Path(handle.name)
            handle.write(json.dumps(report, indent=2, sort_keys=True).encode())
            handle.write(b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    return report


def run_from_partitions(
    *,
    output_root: Path,
    day: date,
    coin: str,
    allow_partial: bool = False,
) -> dict[str, Any]:
    loaded = load_replay(
        output_root=output_root,
        day=day,
        coin=coin,
        allow_partial=allow_partial,
    )
    spec_snapshot = _sha256_bytes(
        canonical_json(
            {
                "source_profile_hash": loaded.source_profile_hash,
                "input_manifests": loaded.input_manifests,
            }
        )
    )
    canary = measure_leak_canary(
        loaded.state,
        loaded.windows,
        coin=coin,
        spec_snapshot=spec_snapshot,
    )
    report = write_report(
        output_root=output_root,
        day=day,
        coin=coin,
        canary=canary,
        input_manifests=loaded.input_manifests,
        source_profile_hash=loaded.source_profile_hash,
        partial=allow_partial,
    )
    if canary.status in INVALID_STATUSES:
        raise RuntimeError(
            f"leak canary refused: {canary.status} — "
            + ("the guard was never wired (event_only == 0), so the run proves "
               "nothing about knowledge-time truncation"
               if canary.status == "INVALID_UNWIRED_GUARD" else
               "the counters contradict each other (disagreements == 0 with a "
               "nonzero score delta); fail closed"))
    return report


def _r7_selftest() -> None:
    """The four R-7 conditions, each tested against the branch it constrains."""
    # The arm R-7 did NOT touch: no event-only read means the guard is unwired.
    assert classify(0, 0, 0.0) == ("INVALID_UNWIRED_GUARD", False)
    assert classify(0, 5, 0.02) == ("INVALID_UNWIRED_GUARD", False)
    print("  PASS  R-7 leaves the unwired-guard arm fatal")

    # Condition 1: a wired guard with zero disagreements and a MEASURED zero
    # delta is bounded, not invalid, and is marked as reclassified.
    assert classify(568, 0, 0.0) == ("BOUND_ZERO_SCORE_DELTA", True)
    print("  PASS  R-7 condition 1: measured zero delta bounds the guard")

    # Condition 2: impact is never forgiven, even with zero disagreements.
    assert classify(568, 0, 0.007) == ("INVALID_COUNTER_INCONSISTENT", False)
    assert classify(568, 0, -1e-9) == ("INVALID_COUNTER_INCONSISTENT", False)
    print("  PASS  R-7 condition 2: any nonzero delta stays INVALID")

    # Pre-existing behaviour is unchanged where R-7 does not reach.
    assert classify(568, 4, 0.0139) == ("VALID_GUARD_BITES", False)
    assert classify(568, 2, 0.0) == ("BOUND_ZERO_SCORE_DELTA", False)
    print("  PASS  R-7 does not disturb the bites or cancelling-delta arms")

    # R-99: the drift check now polices CONSTRUCTION ONLY.
    sample = [
        {"decision_disagreements": c, "delta": 0.0, "r7_reclassified": c == 0}
        for c in (4, 3, 3, 2, 0, 2, 1, 2, 1, 0, 1, 2, 3, 2)
    ]
    intact = r7_drift_check(sample)
    assert intact["verdict"] == "CONSTRUCTION_INTACT", intact
    assert intact["reclassified_coin_days"] == 2
    print("  PASS  R-99: construction intact on the old licensing sample")

    # The retired arms must STAY retired. A collapsed disagreement rate used to
    # escalate on lambda tolerance; with no distribution it is unremarkable, and
    # an all-zero sample is the SAFEST possible observation.
    collapsed = [
        {"decision_disagreements": 0, "delta": 0.0, "r7_reclassified": True}
    ] * 14
    assert r7_drift_check(collapsed)["verdict"] == "CONSTRUCTION_INTACT"
    print("  PASS  R-99: a collapsed rate no longer escalates (lambda arm retired)")

    # No abstention floor survives: the invariant holds at n=1.
    assert r7_drift_check(sample[:1])["verdict"] == "CONSTRUCTION_INTACT"
    print("  PASS  R-99: no coin-day floor -- an invariant needs no power")

    # No receipt may mint a claim of the vacated licence.
    assert "WITHIN_LICENCE" not in str(intact)
    print("  PASS  R-99: the vacated-licence verdict string is gone")

    # The surviving arm still bites.
    forgiven = list(sample)
    forgiven[4] = {"decision_disagreements": 0, "delta": 0.01, "r7_reclassified": True}
    result = r7_drift_check(forgiven)
    assert result["verdict"] == "ESCALATE_TO_COORDINATOR"
    assert any("condition 2" in reason for reason in result["reasons"])
    print("  PASS  R-99: the surviving construction arm still bites")

    _classify_monotonicity_selftest()


def _classify_monotonicity_selftest() -> None:
    """R-99's commissioned instrument: a STATIC sweep of `classify()`.

    DE's reasoning: a rule-ORDERING property cannot drift with data, but it CAN
    be silently reintroduced by a code change. So it is policed here, over the
    rule's input lattice, and runs with every canary.

    The defect R-94 named: pre-amendment, `disagreements == 0` with zero harm was
    INVALID while `disagreements == 5` with the SAME zero harm was fine -- the
    strictly safer observation punished more harshly.
    """
    K = 12
    fatal = lambda st: st.startswith("INVALID")

    # P1 -- the R-94 property itself.
    bad = [d for d in range(K + 1) if fatal(classify(1, d, 0.0)[0])]
    assert not bad, f"zero-harm wired guard fatal at disagreements={bad}"
    # P2 -- no strictly safer input punished harder.
    pairs = [(a, b) for a in range(K + 1) for b in range(a + 1, K + 1)
             if fatal(classify(1, a, 0.0)[0]) and not fatal(classify(1, b, 0.0)[0])]
    assert not pairs, f"non-monotone in its own evidence at {pairs[:3]}"
    print("  PASS  R-99 monotonicity: zero harm is never punished for being safer")

    # P3 -- the unwired arm is untouched by the amendment, everywhere.
    for d in range(K + 1):
        for x in (0.0, 0.25):
            assert classify(0, d, x) == ("INVALID_UNWIRED_GUARD", False)
    print("  PASS  R-99 monotonicity: the unwired-guard arm stays fatal everywhere")

    # P4/P5 -- reclassification only ever on a MEASURED zero.
    assert fatal(classify(1, 0, 0.25)[0])
    assert not any(classify(1, d, 0.25)[1] for d in range(K + 1))
    print("  PASS  R-99 monotonicity: reclassify only on a measured zero delta")

    # NEGATIVE CONTROL -- the properties must FAIL on the pre-amendment rule.
    def pre(event_only, disagreements, delta):
        if event_only == 0:
            return "INVALID_UNWIRED_GUARD", False
        if disagreements == 0:
            return "INVALID_UNBOUND_GUARD", False
        if math.isclose(delta, 0.0, abs_tol=1e-15):
            return "BOUND_ZERO_SCORE_DELTA", False
        return "VALID_GUARD_BITES", False
    assert fatal(pre(1, 0, 0.0)[0]) and not fatal(pre(1, 5, 0.0)[0]), (
        "the negative control no longer reproduces the R-94 defect, so this "
        "test can no longer prove it would catch a reintroduction"
    )
    print("  PASS  R-99 monotonicity: negative control still detects the R-94 defect")


def selftest() -> None:
    _r7_selftest()
    profile = SourceProfile(
        "pm_twap60_ws", True, Transport.LOCAL_WS, Duration(1)
    )
    factory = KnownFactory([profile])
    state = DAState()
    field = "twap60[btc/usd]"
    for event_s, known_s, value, seq in (
        (90, 91, 11.0, 0),
        (100, 102, 10.0, 1),
        (190, 191, 10.0, 2),
        (200, 202, 12.0, 3),
    ):
        state.ingest(
            field,
            factory.observed(
                profile.source,
                value,
                event_ns=event_s * 1_000_000_000,
                recv_ns=known_s * 1_000_000_000,
                seq=seq,
            ),
        )
    result = measure_leak_canary(
        state,
        [
            {
                "window_start_s": 100,
                "window_end_s": 200,
                "closed": True,
                "winner_up": True,
            }
        ],
        coin="btc",
        spec_snapshot="test",
    )
    assert result.value_knowledge_time == 0.0
    assert result.value_event_time == 1.0 and result.delta == 1.0
    assert result.decision_disagreements == 1
    assert result.event_only_boundary_reads == 2
    assert result.status == "VALID_GUARD_BITES"
    print("  PASS  twin run measures the event-time look-ahead bite")

    with tempfile.TemporaryDirectory(prefix="pm-canary-test-") as tmp:
        root = Path(tmp)
        first = write_report(
            output_root=root,
            day=date(1970, 1, 1),
            coin="btc",
            canary=result,
            input_manifests=("a", "b"),
            source_profile_hash="profiles",
        )
        second = write_report(
            output_root=root,
            day=date(1970, 1, 1),
            coin="btc",
            canary=result,
            input_manifests=("a", "b"),
            source_profile_hash="profiles",
        )
        assert first == second
        print("  PASS  canary report is atomic and idempotent")

    with tempfile.TemporaryDirectory(prefix="pm-canary-loader-test-") as tmp:
        root = Path(tmp)
        selected_day = date(1970, 1, 2)
        start_s = 86_400
        end_s = start_s + 300

        def twap_row(event_s: int, known_s: int, value: float, seq: int) -> dict[str, Any]:
            return {
                "t_known_ns": known_s * 1_000_000_000,
                "t_known_err_ns": 1_000_000,
                "t_known_prov": "OBSERVED",
                "t_event_ms": event_s * 1_000,
                "t_publish_ms": event_s * 1_000,
                "symbol": "btc/usd",
                "coin": "btc",
                "topic": "crypto_prices_twap_sixty",
                "window_s": 60,
                "value": value,
                "full_accuracy_value": str(value),
                "seq": seq,
                "duplicate_count": 0,
                "source": "pm_twap60_ws",
                "source_profile_hash": load_source_registry().registry_hash,
                "collector_version": "test",
                "collector_era_coverage": "test",
                "source_file_id": "test",
            }

        twap_by_day = {
            selected_day - timedelta(days=1): [
                twap_row(start_s - 10, start_s - 9, 11.0, 0)
            ],
            selected_day: [
                twap_row(start_s, start_s + 2, 10.0, 1),
                twap_row(end_s - 10, end_s - 9, 10.0, 2),
                twap_row(end_s, end_s + 2, 12.0, 3),
            ],
            selected_day + timedelta(days=1): [],
        }
        for partition_day, rows in twap_by_day.items():
            write_partition(
                root,
                "twap",
                partition_day,
                "btc",
                rows,
                [],
                partial=False,
                diagnostics={"test": True},
            )
        write_partition(
            root,
            "windows",
            selected_day,
            "btc",
            [
                {
                    "t_known_ns": (end_s + 3) * 1_000_000_000,
                    "t_known_err_ns": 1_000_000,
                    "t_known_prov": "OBSERVED",
                    "t_event_ms": start_s * 1_000,
                    "slug": f"btc-updown-5m-{start_s}",
                    "coin": "btc",
                    "window_start_s": start_s,
                    "window_end_s": end_s,
                    "closed": True,
                    "winner_up": True,
                    "seq": 0,
                    "source": "pm_resolution_poll",
                    "source_profile_hash": load_source_registry().registry_hash,
                    "source_file_id": "test",
                }
            ],
            [],
            partial=False,
            diagnostics={"test": True},
        )
        coverage_known_ns = (end_s + 7) * 1_000_000_000
        write_partition(
            root,
            "coverage",
            selected_day,
            "btc",
            [
                {
                    "t_known_ns": coverage_known_ns,
                    "t_known_err_ns": 1_000_000,
                    "t_known_prov": "OBSERVED",
                    "t_event_ms": start_s * 1_000,
                    "slug": f"btc-updown-5m-{start_s}",
                    "coin": "btc",
                    "symbol": "btc/usd",
                    "field": "twap60[btc/usd]",
                    "target_start_ns": (start_s - 5) * 1_000_000_000,
                    "target_end_ns": (end_s + 5) * 1_000_000_000,
                    "coverage_hash": "test-coverage",
                    "rule_id": "A-TWAP-1",
                    "rule_hash": "test-rule",
                    "admissible": True,
                    "evaluated_at_ns": coverage_known_ns,
                    "seq": 0,
                    "source": "pm_twap60_ws",
                    "source_profile_hash": load_source_registry().registry_hash,
                }
            ],
            [],
            partial=False,
            diagnostics={"test": True},
        )
        loaded = load_replay(
            output_root=root,
            day=selected_day,
            coin="btc",
        )
        loaded_result = measure_leak_canary(
            loaded.state,
            loaded.windows,
            coin="btc",
            spec_snapshot="loader-test",
        )
        assert loaded_result.status == "VALID_GUARD_BITES"
        before = loaded.state.view(KnowledgeTime(coverage_known_ns)).coverage(
            "twap60[btc/usd]", Duration(400 * 1_000_000_000)
        )
        after = loaded.state.view(
            KnowledgeTime(coverage_known_ns + 1_000_000)
        ).coverage("twap60[btc/usd]", Duration(400 * 1_000_000_000))
        assert isinstance(before, Unavailable) and isinstance(after, Known)
        print("  PASS  verified Tier-1 loader preserves point-in-time coverage")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selftest", action="store_true")
    parser.add_argument("--day", help="completed target UTC day YYYY-MM-DD")
    parser.add_argument("--coin", choices=tuple(COIN_SYMBOL))
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--allow-partial-smoke",
        action="store_true",
        help="diagnostic only; a partial report is never an eligible result",
    )
    args = parser.parse_args()
    if args.selftest:
        selftest()
    if args.day:
        if not args.coin:
            parser.error("--day requires --coin")
        report = run_from_partitions(
            output_root=args.output_root,
            day=date.fromisoformat(args.day),
            coin=args.coin,
            allow_partial=args.allow_partial_smoke,
        )
        item = report["canary"]
        print(
            f"CANARY day={args.day} coin={args.coin} n={item['n']} "
            f"knowledge={item['value_knowledge_time']:.4%} "
            f"event={item['value_event_time']:.4%} "
            f"delta_pp={100 * item['delta']:.3f} status={item['status']}"
        )
    if not args.selftest and not args.day:
        parser.error("choose --selftest and/or --day")


if __name__ == "__main__":
    main()
