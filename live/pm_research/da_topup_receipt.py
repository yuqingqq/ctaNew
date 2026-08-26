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
OUT = PM / "derived/da_development_topup_v1.json"

# --- the declared bounds, verbatim from R-145(3) ---------------------------
SLUG_START_EXCL_LO = 1787650200      # strictly after
SLUG_START_EXCL_HI = 1787702400      # strictly before
COINS = ("btc", "eth")

# CLASS D (frozen verdict/guard): the admitted HF era floor.  Authority
# R-145(3) and CLAUDE.md's data budget; NOT derived from the ledger.  See
# Q-DA-67 for what happens when it is derived.
ERA_FLOOR_RECV_NS = 1787579334881534478

RECEIPT_VERSION = "da_development_topup_v1"
ROLE = "DEVELOPMENT_ONLY_NEVER_FORWARD_VALIDATION"

BUILDER_FILES = (
    "live/pm_research/da_topup_receipt.py",
    "live/pm_research/harmful_exposure_rows.py",
    "live/pm_research/policy_optimizer_queue_realistic.py",
    "live/pm_research/flow_intensity.py",
)

STATUSES = (
    "OK",
    "NO_ARCHIVE",
    "NO_TOKENS",
    "NOT_PM_ERA_COVERED",
    "PM_GAP",
    "BINANCE_GAP_OR_TRUNCATED",
)


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

    src = {}
    for rel in BUILDER_FILES:
        p = REPO / rel
        src[rel] = sha256_file(p) if p.exists() else None
    for label, p in (("hf_collector_runs.jsonl", HF_RUNS),
                     ("markets.jsonl", MARKETS),
                     ("pm_collector_gaps.jsonl", PM_GAPS)):
        src[label] = sha256_file(p) if p.exists() else None

    days = sorted({r["day"] for r in rows})
    return {
        "receipt_version": RECEIPT_VERSION,
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
            "unblocks_on": "Q-DA-67 ruling",
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
