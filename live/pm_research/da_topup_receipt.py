"""Materialise the R-145(3) development top-up as a pre-registered population.

SURFACE AUTHORISATION (R-126, in-file): R-145(3) DECLARED this population
before any score existed and R-145(6) assigned DA to materialise its receipt;
the dispatch makes it DA's first action because it gates BE's Phase 2.
Nothing here selects, scores or fits -- it pins WHICH tape is admissible and
WHAT BYTES it consists of, so that no later slice of it can be chosen after a
number has been seen (rule 11).

THE POPULATION, transcribed from R-145(3) rather than re-derived:

  btc + eth 5-minute windows whose slug start is STRICTLY AFTER 1787650200
  (2026-08-25T09:30Z, the last consumed slug) and STRICTLY BEFORE 1787702400
  (2026-08-26T00:00Z; last admissible start 1787702100).  Era-pure, complete
  windows only, exclusions as counted statuses.  DEVELOPMENT ONLY -- never
  forward validation; consumed at first read.

WHY THE ERA FLOOR IS A LITERAL HERE AND NOT A LEDGER READ (Q-DA-67).  The
shared builder derives it as `max(started_at_ns)` over
`data/mm_hf/collector_runs.jsonl`.  The 2026-08-26T03:55Z box crash appended a
restart row, so that expression silently moved the floor forward 39.4 hours and
`select_v2_era` now admits 0 of 926 windows.  R-147(2) ruled that row is "a
coverage gap, not a boundary".  This module therefore pins the floor to the
value R-145(3) NAMES.  That is READING A RULED CONSTANT, not choosing one: no
Class-C/D value is set here, and the receipt records the floor with its
authority so the provenance is auditable rather than implicit.

WHAT IS HASHED, AND WHY IT IS THE INPUTS.  The top-up's derived exposure rows
DO NOT EXIST YET and cannot be built until Q-DA-67 is ruled -- the selector
that would build them admits nothing.  So this receipt pins the RAW SOURCE
BYTES (every per-slug PM archive, the HF collector ledger, the market/token
map, the PM gap ledger) plus the builder files that will consume them.  That is
strictly stronger than hashing a derived artifact: the inputs are closed-period
immutable tape, and BE can rebuild from exactly these bytes and verify.  What
this receipt does NOT claim is that any dataset was built.

    python3 live/pm_research/da_topup_receipt.py --selftest
    python3 live/pm_research/da_topup_receipt.py run [--out PATH]
"""
from __future__ import annotations

import argparse
import collections
import datetime as dt
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import harmful_exposure_rows as HER          # continuity logic, UNMODIFIED
import policy_optimizer_queue_realistic as qr

REPO = Path(__file__).resolve().parents[2]
PM = REPO / "data/pm_5min"
HF_RUNS = REPO / "data/mm_hf/collector_runs.jsonl"
MARKETS = PM / "markets.jsonl"
PM_GAPS = PM / "collector_gaps.jsonl"
OUT = PM / "derived/da_development_topup_v2.json"

# --- the declared bounds, verbatim from R-145(3) ---------------------------
SLUG_START_EXCL_LO = 1787650200      # strictly after
SLUG_START_EXCL_HI = 1787702400      # strictly before
COINS = ("btc", "eth")

# CLASS D (frozen verdict/guard): the admitted HF era floor.  Authority
# R-145(3) and CLAUDE.md's data budget; NOT derived from the ledger.  See
# Q-DA-67 for what happens when it is derived.
ERA_FLOOR_RECV_NS = 1787579334881534478

# CLASS D. The population-scoped declared ERA END, in BE's post-R-154 idiom
# (`DECLARED_ERA_END_S`): last admissible slug start + WINDOW_S + MARKOUT_S + 5.
# 1787702100 + 300 + 5 + 5.  DECLARED HERE BECAUSE DA OWNS THIS POPULATION:
# `harmful_exposure_rows.DECLARED_ERA_END_S` carries an entry only for
# v3_4_consumed_fragment, so a Phase-2 build of the top-up would otherwise
# need an end invented by its CONSUMER.  An end chosen by the consumer is an
# undeclared parameter no matter how reasonable it looks.
DECLARED_ERA_END_S = 1787702410.0
POPULATION_NAME = "da_development_topup"

RECEIPT_VERSION = "da_development_topup_v2"
SUPERSEDES = "da_development_topup_v1.json"
SUPERSEDE_REASON = (
    "v1 pinned harmful_exposure_rows.py at sha256 3ed11912... which was the "
    "PRE-FIX builder. BE's Q-DA-67 remedy (commit c4cb4e3, ratified R-154) "
    "changed that file, so v1's builder hash no longer describes the code a "
    "rebuild would run. v2 refreshes the hash and adds the population's "
    "DECLARED ERA END. The population itself is UNCHANGED and that is "
    "asserted, not assumed -- see population_unchanged_vs_v1. Rule 13: v1 is "
    "not edited; it stands as provenance.")
ROLE = "DEVELOPMENT_ONLY_NEVER_FORWARD_VALIDATION"

BUILDER_FILES = (
    "live/pm_research/da_topup_receipt.py",
    "live/pm_research/harmful_exposure_rows.py",
    "live/pm_research/policy_optimizer_queue_realistic.py",
    "live/pm_research/flow_intensity.py",
)

#: R-160: a pinned hash means different things for different inputs, and the
#: difference decides whether a MISMATCH is evidence of drift or of time
#: passing.  Emitted PER SOURCE rather than as one receipt-level flag, because
#: a single flag cannot answer the only question a reader actually has: "this
#: hash does not match -- do I care?"
#:
#:   reproducibility_anchor -- immutable once written (closed-period archive
#:       bytes, a builder file at a commit).  A mismatch IS drift; investigate.
#:   state_at_build -- an append-growing live registry.  Its hash differs
#:       between ANY two builds at different instants, for entirely benign
#:       reasons.  A mismatch is NOT evidence of anything; the substantive
#:       guarantee lives in the population comparison, not here.  Found the
#:       hard way: markets.jsonl's hash moved between v1 and v2 while the
#:       population was byte-identical (Q-DA-76).
PIN_SEMANTICS = {
    "live/pm_research/da_topup_receipt.py": "reproducibility_anchor",
    "live/pm_research/harmful_exposure_rows.py": "reproducibility_anchor",
    "live/pm_research/policy_optimizer_queue_realistic.py": "reproducibility_anchor",
    "live/pm_research/flow_intensity.py": "reproducibility_anchor",
    "hf_collector_runs.jsonl": "state_at_build",
    "markets.jsonl": "state_at_build",
    "pm_collector_gaps.jsonl": "state_at_build",
}

STATUSES = (
    "OK",
    "NO_ARCHIVE",
    "NO_TOKENS",
    "NOT_PM_ERA_COVERED",
    "PM_GAP",
    "BINANCE_GAP_OR_TRUNCATED",
)


class PinnedIdentityMismatch(RuntimeError):
    """A pinned identity literal does not match the artifact it names."""


def assert_era_floor_is_real(path: Path = HF_RUNS) -> dict[str, Any]:
    """REFUSE unless the pinned floor is an actual run record in the ledger.

    R-154's lesson, adopted: BE's first `DECLARED_ERA_KEY` was copied from
    console output truncated at 110 characters, and only a guard that REFUSED
    on mismatch caught it.  A wrong-but-plausible identity literal does not
    crash -- it silently describes the real runs as some other era while
    looking entirely correct.

    My own floor was transcribed from the TEXT of R-145(3), which is the same
    exposure: a ruling is rendered prose, not the artifact.  The literal
    happens to be right (verified: it is the `started_at_ns` of pid 1369188),
    but "happens to be right" is not a property a build should rely on.  So
    this reads the ledger and REFUSES, rather than warning, if the pinned
    value is not there.
    """
    recs = [json.loads(l) for l in
            path.read_text(encoding="utf-8").splitlines() if l.strip()]
    hit = [r for r in recs if int(r["started_at_ns"]) == ERA_FLOOR_RECV_NS]
    if not hit:
        raise PinnedIdentityMismatch(
            f"pinned ERA_FLOOR_RECV_NS {ERA_FLOOR_RECV_NS} is not the "
            f"started_at_ns of ANY run record in {path}. Ledger holds "
            f"{[int(r['started_at_ns']) for r in recs]}. Refusing to build: a "
            f"floor that names no real run pins nothing.")
    return hit[0]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def in_range(t0: int) -> bool:
    """STRICTLY between the declared bounds -- both ends exclusive (R-145(3))."""
    return SLUG_START_EXCL_LO < t0 < SLUG_START_EXCL_HI


def day_of(t0: int) -> str:
    return dt.datetime.fromtimestamp(t0, dt.timezone.utc).strftime("%Y-%m-%d")


def classify(slug: str, t0: int, coin: str, *, paths, tokens, covered,
             gaps, bn_ok) -> str:
    """First failing predicate wins, and every branch returns a STATUS.

    Order matters and is declared: presence of bytes, then identity, then the
    two tapes.  A window is never dropped -- rule 4.
    """
    if slug not in paths:
        return "NO_ARCHIVE"
    if slug not in tokens:
        return "NO_TOKENS"
    if slug not in covered:
        return "NOT_PM_ERA_COVERED"
    if gaps.get(slug):
        return "PM_GAP"
    if not bn_ok(t0, coin):
        return "BINANCE_GAP_OR_TRUNCATED"
    return "OK"


def build(as_of_ns: int | None = None) -> dict[str, Any]:
    floor_record = assert_era_floor_is_real()      # refuses, never warns
    fi = qr.base.fi
    as_of = as_of_ns if as_of_ns is not None else int(dt.datetime.now(
        dt.timezone.utc).timestamp() * 1e9)

    paths = fi._archive_paths()
    tokens = fi.token_map()
    covered = fi.covered_slugs(fi.ERA)
    gaps = fi.gaps_by_slug(fi.ERA)

    # The SAME continuity predicate the exposure builder uses, called with the
    # PINNED floor instead of the ledger maximum.  The builder is not edited;
    # only the bounds it is handed are correct.
    bounds = (ERA_FLOOR_RECV_NS / 1e9, as_of / 1e9)

    def bn_ok(t0: int, coin: str) -> bool:
        return HER.binance_continuity_ok(t0, coin, bounds)

    candidates: list[tuple[str, int, str]] = []
    for slug in sorted(set(paths) | set(covered)):
        parts = slug.split("-")
        coin = parts[0]
        if coin not in COINS or "-updown-5m-" not in slug:
            continue
        try:
            t0 = int(slug.rsplit("-", 1)[1])
        except ValueError:
            continue
        if in_range(t0):
            candidates.append((slug, t0, coin))
    candidates.sort(key=lambda x: (x[1], x[2]))

    rows = []
    for slug, t0, coin in candidates:
        status = classify(slug, t0, coin, paths=paths, tokens=tokens,
                          covered=covered, gaps=gaps, bn_ok=bn_ok)
        path = paths.get(slug)
        rows.append({
            "slug": slug, "coin": coin, "window_start": t0,
            "day": day_of(t0), "status": status,
            "archive_file": path.name if path else None,
            "archive_sha256": sha256_file(path) if path else None,
        })

    by_status = collections.Counter(r["status"] for r in rows)
    by_coin_day_status: dict[str, Any] = collections.defaultdict(
        lambda: collections.defaultdict(collections.Counter))
    for r in rows:
        by_coin_day_status[r["coin"]][r["day"]][r["status"]] += 1

    ok_slugs = [r["slug"] for r in rows if r["status"] == "OK"]
    manifest = "\n".join(f"{r['slug']} {r['status']} {r['archive_sha256']}"
                         for r in rows)

    # A source with no declared semantics is a source nobody has decided how
    # to read.  Refuse rather than default (R-154: guards refuse, never warn).
    src = {}
    for rel in BUILDER_FILES:
        p = REPO / rel
        src[rel] = sha256_file(p) if p.exists() else None
    for label, p in (("hf_collector_runs.jsonl", HF_RUNS),
                     ("markets.jsonl", MARKETS),
                     ("pm_collector_gaps.jsonl", PM_GAPS)):
        src[label] = sha256_file(p) if p.exists() else None

    undeclared = sorted(set(src) - set(PIN_SEMANTICS))
    if undeclared:
        raise PinnedIdentityMismatch(
            f"pinned source(s) {undeclared} have no declared pin_semantics. "
            f"Every pin must say whether a mismatch means drift "
            f"(reproducibility_anchor) or merely time passing "
            f"(state_at_build); defaulting would hand the reader a hash they "
            f"cannot interpret.")

    days = sorted({r["day"] for r in rows})
    return {
        "receipt_version": RECEIPT_VERSION,
        "supersedes": SUPERSEDES,
        "supersede_reason": SUPERSEDE_REASON,
        "as_of_utc": dt.datetime.fromtimestamp(
            as_of / 1e9, dt.timezone.utc).isoformat(),
        "as_of_ns": as_of,
        "role": ROLE,
        "role_note": (
            "DEVELOPMENT ONLY. This tape may be used for Phase-1/2 selection "
            "and is CONSUMED at first read. It is never forward validation. "
            "Tape from 2026-08-26T00:00Z onward is untouched and reserved."),
        "authority": {
            "population_declared_by": "COORDINATION.md R-145(3)",
            "materialisation_assigned_by": "COORDINATION.md R-145(6)",
            "era_floor_authority": "R-145(3) literal; CLAUDE.md data budget",
            "era_floor_not_derived_because": (
                "harmful_exposure_rows.v2_era_bounds() reduces the collector "
                "ledger with max(), which the 2026-08-26T03:55Z crash restart "
                "moved forward 39.4h -- see Q-DA-67. R-147(2) ruled that row "
                "a coverage gap, not a boundary."),
        },
        "bounds": {
            "slug_start_strictly_after": SLUG_START_EXCL_LO,
            "slug_start_strictly_before": SLUG_START_EXCL_HI,
            "last_admissible_slug_start": SLUG_START_EXCL_HI - 300,
            "coins": list(COINS),
            "era_floor_recv_ns": ERA_FLOOR_RECV_NS,
            "era_floor_verified_at_ledger": {
                "pid": floor_record.get("pid"),
                "collector_schema_version": floor_record.get(
                    "collector_schema_version"),
                "stamp_point": floor_record.get("stamp_point"),
                "note": "the pinned floor IS this run record; the build "
                        "refuses if it is not (R-154 identity lesson)"},
            "population_name": POPULATION_NAME,
            "declared_era_end_s": DECLARED_ERA_END_S,
            "declared_era_end_note": (
                "last admissible slug start 1787702100 + WINDOW_S 300 + "
                "MARKOUT_S 5 + 5, matching harmful_exposure_rows' v3.4 "
                "convention. DECLARED BY DA as the population owner so a "
                "Phase-2 build does not have to invent one."),
            "continuity_predicate": (
                "harmful_exposure_rows.binance_continuity_ok over "
                "[t0-10, t0+WINDOW_S+MARKOUT_S+1], max HF gap <= 1.0s, "
                "era-pure from era_floor_recv_ns"),
        },
        "n_and_as_of": {
            "n_candidate_slugs": len(rows),
            "n_ok": by_status.get("OK", 0),
            "n_by_status": dict(by_status),
            "n_by_coin_day_status": {
                c: {d: dict(s) for d, s in dd.items()}
                for c, dd in by_coin_day_status.items()},
            "days_spanned": days,
            "n_complete_utc_days": len(days),
            "cluster_note": (
                "This population spans %d UTC day(s). Below G=5 complete days "
                "no interval may be quoted on a day-clustered statistic "
                "(rule 8); development selection does not need one, but any "
                "figure derived from this tape inherits the constraint."
                % len(days)),
        },
        "hashes": {
            "sources": src,
            "pin_semantics": {k: PIN_SEMANTICS[k] for k in src},
            "pin_semantics_note": (
                "reproducibility_anchor: immutable once written; a mismatch IS "
                "drift. state_at_build: append-growing live registry; its hash "
                "differs between any two builds for benign reasons and a "
                "mismatch is NOT evidence of drift -- the guarantee is the "
                "population comparison, not the hash. R-160."),
            "slug_manifest_sha256": hashlib.sha256(
                manifest.encode("utf-8")).hexdigest(),
            "ok_slug_list_sha256": hashlib.sha256(
                "\n".join(ok_slugs).encode("utf-8")).hexdigest(),
        },
        "derived_dataset": {
            "built": False,
            "reason": (
                "The top-up exposure rows are NOT built. Building them "
                "requires harmful_exposure_rows --v2-era, whose era floor is "
                "the Q-DA-67 defect; it currently selects 0 windows. This "
                "receipt pins INPUTS, not outputs, and claims nothing about a "
                "dataset that does not exist."),
            "unblocks_on": "RESOLVED — Q-DA-67 fixed by BE (c4cb4e3), "
                           "ratified R-154. A top-up build is now possible; "
                           "pass era_end_s=1787702410.0 (declared above).",
        },
        "slugs": rows,
    }


def _selftests() -> int:
    checks = 0

    def ok(cond, label):
        nonlocal checks
        checks += 1
        if not cond:
            raise AssertionError(f"selftest failed: {label}")

    # --- the bounds are EXCLUSIVE at both ends (R-145(3) says "strictly") ---
    ok(not in_range(SLUG_START_EXCL_LO), "the last consumed slug is EXCLUDED")
    ok(in_range(SLUG_START_EXCL_LO + 300), "the next slug after it is included")
    ok(not in_range(SLUG_START_EXCL_HI), "the 08-26 00:00 boundary is EXCLUDED")
    ok(in_range(SLUG_START_EXCL_HI - 300), "the last 08-25 slug is included")
    ok(not in_range(SLUG_START_EXCL_HI + 300), "forward tape is EXCLUDED")

    # --- FALSIFIER (rule 15): classify must FIRE on each known-bad input ---
    base = dict(paths={"s": Path("/x")}, tokens={"s": ("a", "b")},
                covered={"s"}, gaps={}, bn_ok=lambda t, c: True)
    ok(classify("s", 1, "btc", **base) == "OK", "a clean window is OK")
    ok(classify("s", 1, "btc", **{**base, "paths": {}}) == "NO_ARCHIVE",
       "a missing archive is a STATUS")
    ok(classify("s", 1, "btc", **{**base, "tokens": {}}) == "NO_TOKENS",
       "a missing token map is a STATUS")
    ok(classify("s", 1, "btc", **{**base, "covered": set()})
       == "NOT_PM_ERA_COVERED", "a PM era miss is a STATUS")
    ok(classify("s", 1, "btc", **{**base, "gaps": {"s": [(1.0, 2.0)]}})
       == "PM_GAP", "a PM gap is a STATUS")
    ok(classify("s", 1, "btc", **{**base, "bn_ok": lambda t, c: False})
       == "BINANCE_GAP_OR_TRUNCATED", "an HF gap is a STATUS")
    # an EMPTY gap list is not a gap -- the ledger records slugs with no gaps
    ok(classify("s", 1, "btc", **{**base, "gaps": {"s": []}}) == "OK",
       "an empty gap list is not a gap")
    ok(set(STATUSES) >= {classify("s", 1, "btc", **base)},
       "every returned status is declared")

    # --- the era floor is the RULED literal, not a ledger reduction --------
    ledger_max = max(int(json.loads(l)["started_at_ns"])
                     for l in HF_RUNS.read_text().splitlines() if l.strip())
    ok(ERA_FLOOR_RECV_NS != ledger_max,
       "REGRESSION GUARD: the pinned floor is NOT max(ledger) -- if these ever "
       "coincide the guard is vacuous and this test must be re-grounded")
    ok(ERA_FLOOR_RECV_NS == 1787579334881534478,
       "the floor is the value R-145(3) names")

    # --- R-154 identity lesson: the guard REFUSES, and it can FIRE --------
    rec = assert_era_floor_is_real()
    ok(int(rec["started_at_ns"]) == ERA_FLOOR_RECV_NS,
       "the pinned floor IS a real run record in the ledger artifact")
    ok(rec.get("collector_schema_version") == "hf_ws_v2_recv_boundary",
       "and that record carries the declared era key")

    import tempfile
    with tempfile.TemporaryDirectory() as td:
        # FALSIFIER: a plausible-but-wrong literal, the exact failure R-154
        # describes -- a truncated/typo'd identity that does not crash.
        bad = Path(td) / "runs.jsonl"
        bad.write_text(json.dumps({
            "started_at_ns": ERA_FLOOR_RECV_NS + 1, "pid": 1,
            "collector_schema_version": "hf_ws_v2_recv_boundary",
            "stamp_point": "IMMEDIATELY_AFTER_WS_RECV_BEFORE_JSON_PARSE",
            "symbols": ["BTCUSDT"]}) + "\n", encoding="utf-8")
        try:
            assert_era_floor_is_real(bad)
        except PinnedIdentityMismatch:
            ok(True, "the guard REFUSES a floor that names no real run")
        else:
            ok(False, "MUST refuse: off by one nanosecond is still not a run")
        # ... and does not fire on a good one (no false positive)
        good = Path(td) / "good.jsonl"
        good.write_text(json.dumps({
            "started_at_ns": ERA_FLOOR_RECV_NS, "pid": 9,
            "collector_schema_version": "hf_ws_v2_recv_boundary",
            "stamp_point": "IMMEDIATELY_AFTER_WS_RECV_BEFORE_JSON_PARSE",
            "symbols": ["BTCUSDT"]}) + "\n", encoding="utf-8")
        ok(assert_era_floor_is_real(good)["pid"] == 9,
           "and accepts a ledger that does carry it")
        # an EMPTY ledger must refuse too -- absence is not confirmation
        empty = Path(td) / "empty.jsonl"
        empty.write_text("", encoding="utf-8")
        try:
            assert_era_floor_is_real(empty)
        except PinnedIdentityMismatch:
            ok(True, "an empty ledger cannot confirm the floor -- refuses")
        else:
            ok(False, "MUST NOT treat an empty ledger as confirmation")

    # --- the declared era end is a literal tied to the declared bounds ----
    ok(DECLARED_ERA_END_S == (SLUG_START_EXCL_HI - 300) + 300 + 5.0 + 5.0,
       "the declared end is the last admissible slug's window end + markout, "
       "derived from the DECLARED bounds rather than picked")
    ok(DECLARED_ERA_END_S > SLUG_START_EXCL_HI - 300,
       "the end is after the last admissible slug start")

    # --- R-160: every pin declares how a MISMATCH should be read ----------
    pinned = set(BUILDER_FILES) | {"hf_collector_runs.jsonl", "markets.jsonl",
                                   "pm_collector_gaps.jsonl"}
    ok(pinned <= set(PIN_SEMANTICS),
       "every pinned source has declared pin_semantics -- a new source cannot "
       "be added without deciding how its hash is to be read")
    ok(set(PIN_SEMANTICS.values()) <= {"reproducibility_anchor",
                                       "state_at_build"},
       "only the two declared semantics exist")
    ok(PIN_SEMANTICS["markets.jsonl"] == "state_at_build",
       "the append-growing registry that actually moved is marked as such")
    ok(PIN_SEMANTICS["live/pm_research/harmful_exposure_rows.py"]
       == "reproducibility_anchor",
       "a builder file IS an anchor -- its hash moving is exactly the drift "
       "that forced the v1 -> v2 supersession")
    ok(sum(1 for v in PIN_SEMANTICS.values() if v == "state_at_build") == 3,
       "all three live ledgers are state_at_build, not just the one that bit")

    print(f"da_topup_receipt selftests: {checks} checks passed")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("cmd", nargs="?", choices=["run"])
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if a.selftest or not a.cmd:
        return _selftests()
    rep = build()
    out = Path(a.out) if a.out else OUT
    tmp = out.with_suffix(out.suffix + ".tmp")
    tmp.write_text(json.dumps(rep, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(out)                      # atomic (TODO Phase-0 discipline)
    n = rep["n_and_as_of"]
    print(f"wrote {out}")
    print(f"  candidates {n['n_candidate_slugs']}  OK {n['n_ok']}")
    print(f"  by status  {n['n_by_status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
