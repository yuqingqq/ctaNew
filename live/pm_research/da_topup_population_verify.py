"""Independently verify BE's built top-up rows against DA's OWN v2 receipt.

SURFACE AUTHORISATION (R-126, in-file): R-171 assigns DA to verify the built
population before the Phase-2 comparison receipt finalizes. "A mismatch stops
the receipt, not the runs."

PRE-REGISTERED: every predicate below was written and its falsifiers RUN while
`harmful_exposure_rows_v3_topup.json` DID NOT YET EXIST. A verifier authored
after seeing the artifact can always be shaped, however honestly, to the thing
it is looking at; this one could not be. The checks are fixed first.

INDEPENDENCE IS THE POINT. Every expectation is taken from
`da_development_topup_v2.json` -- DA's own pinned slug manifest -- and NEVER
from the built dataset's own summary fields. A dataset agreeing with its own
header proves nothing (rule 16: verify at the artifact a claim names, and know
what KIND of document you are reading).

WHAT WOULD BE WORST, and is therefore checked explicitly: RESERVED FORWARD
TAPE LEAKING INTO A DEVELOPMENT DATASET. Slug starts at or after
2026-08-26T00:00Z are reserved for forward validation. If any reached a
development population, the forward reservation is broken for every line at
once and no later test can repair it -- the tape would be consumed.

    python3 live/pm_research/da_topup_population_verify.py --selftest
    python3 live/pm_research/da_topup_population_verify.py verify
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path
from typing import Any, Iterator

REPO = Path(__file__).resolve().parents[2]
DERIVED = REPO / "data/pm_5min/derived"
def _receipt_version_key(path: Path) -> tuple[int, ...]:
    """Sort `..._v2.json`, `..._v2.1.json`, `..._v10.json` correctly.

    Tuple comparison, not integer: a resolver that assumed integers once could
    not parse `v2.1` and CRASHED rather than silently falling back to v2 --
    fail-closed, but still incomplete. Dotted versions sort properly here.
    """
    stem = path.stem.rsplit("_v", 1)[-1]
    try:
        return tuple(int(x) for x in stem.split("."))
    except ValueError:
        return (-1,)


def latest_receipt(derived: Path = DERIVED) -> Path:
    """The HIGHEST-version topup receipt, resolved at run time.

    Hardcoding v2 was correct until R-173 forced a v3 re-base; a verifier
    pinned to a superseded receipt would keep checking the built population
    against a manifest the programme had already overruled, and would report
    a confident MISMATCH that was purely its own staleness. Resolving forward
    is the same discipline the freeze receipts use.
    """
    cands = [q for q in derived.glob("da_development_topup_v*.json")
             if _receipt_version_key(q) != (-1,)]
    if not cands:
        raise VerificationFailed(
            f"no da_development_topup_v*.json receipt in {derived}; refusing "
            f"to verify against no manifest at all")
    return max(cands, key=_receipt_version_key)


RECEIPT = DERIVED / "da_development_topup_v2.json"   # default; see latest_receipt()
BUILT = DERIVED / "harmful_exposure_rows_v3_topup.json"

# The declared bounds, re-read from DA's receipt at run time rather than
# retyped here -- a literal transcribed twice is a literal that can disagree
# with itself (R-154).
CHUNK = 1 << 23


class VerificationFailed(RuntimeError):
    """A predicate failed. The Phase-2 receipt does not finalize."""


def iter_rows(path: Path) -> Iterator[dict[str, Any]]:
    """Stream a single-line multi-hundred-MB JSON without materialising it.

    Same quote-aware brace scanner used for the v3.4 audit, where it was
    validated element-for-element against `json.load` on a 21 MB file
    (19,723 rows, `streamed == truth`). R-148: a whole-file parse of an
    artifact this size is the allocation burst that killed the box.
    """
    buf = ""
    pos = keep = depth = 0
    start = None
    in_str = esc = started = False
    with path.open("r", encoding="utf-8") as fh:
        while True:
            chunk = fh.read(CHUNK)
            if not chunk:
                break
            if keep:
                buf = buf[keep:]
                pos -= keep
                if start is not None:
                    start -= keep
                keep = 0
            buf += chunk
            n = len(buf)
            while pos < n:
                c = buf[pos]
                if in_str:
                    if esc:
                        esc = False
                    elif c == "\\":
                        esc = True
                    elif c == '"':
                        in_str = False
                elif c == '"':
                    in_str = True
                elif not started:
                    if c == "[":
                        started = True
                elif c == "{":
                    if depth == 0:
                        start = pos
                    depth += 1
                elif c == "}":
                    depth -= 1
                    if depth == 0 and start is not None:
                        yield json.loads(buf[start:pos + 1])
                        keep = pos + 1
                        start = None
                elif c == "]" and depth == 0:
                    return
                pos += 1


def expectations(receipt_path: Path | None = None) -> dict[str, Any]:
    """DA's own pinned manifest -- the ONLY source of expected values."""
    receipt_path = receipt_path or latest_receipt()
    r = json.loads(receipt_path.read_text(encoding="utf-8"))
    b = r["bounds"]
    ok_slugs = {row["slug"] for row in r["slugs"] if row["status"] == "OK"}
    return {
        "receipt_version": r["receipt_version"],
        "ok_slugs": ok_slugs,
        "n_ok": r["n_and_as_of"]["n_ok"],
        "lo_excl": b["slug_start_strictly_after"],
        "hi_excl": b["slug_start_strictly_before"],
        "era_floor_recv_ns": b["era_floor_recv_ns"],
        "declared_era_end_s": b["declared_era_end_s"],
        "population_name": b["population_name"],
        "by_coin_day_status": r["n_and_as_of"]["n_by_coin_day_status"],
    }


def check(built_slugs: set[str], t0s: dict[str, int], exp: dict[str, Any],
          n_rows: int) -> list[dict[str, Any]]:
    """Every predicate COMPUTED, never a printed conclusion (rule 10)."""
    out: list[dict[str, Any]] = []

    def p(name, ok, detail):
        out.append({"predicate": name, "pass": bool(ok), "detail": detail})

    exp_slugs = exp["ok_slugs"]
    missing = sorted(exp_slugs - built_slugs)
    extra = sorted(built_slugs - exp_slugs)

    p("slug_set_identity", not missing and not extra,
      f"expected {len(exp_slugs)}, built {len(built_slugs)}, "
      f"missing {len(missing)}, extra {len(extra)}"
      + (f"; first missing {missing[:3]}" if missing else "")
      + (f"; first extra {extra[:3]}" if extra else ""))

    p("n_ok_matches_receipt", len(built_slugs) == exp["n_ok"],
      f"receipt n_ok {exp['n_ok']}, built distinct slugs {len(built_slugs)}")

    # THE ONE THAT MATTERS MOST: reserved forward tape must not appear.
    leaked = sorted(s for s, t in t0s.items() if t >= exp["hi_excl"])
    p("no_reserved_forward_tape", not leaked,
      f"slugs at/after {exp['hi_excl']} (2026-08-26T00:00Z): {len(leaked)}"
      + (f" -> {leaked[:5]}" if leaked else " (reservation intact)"))

    # ... and the consumed fragment must not be re-entered either.
    consumed = sorted(s for s, t in t0s.items() if t <= exp["lo_excl"])
    p("no_consumed_fragment", not consumed,
      f"slugs at/before {exp['lo_excl']}: {len(consumed)}"
      + (f" -> {consumed[:5]}" if consumed else " (fragment not re-entered)"))

    if t0s:
        lo, hi = min(t0s.values()), max(t0s.values())
        p("t0_within_declared_bounds",
          lo > exp["lo_excl"] and hi < exp["hi_excl"],
          f"t0 range [{lo}, {hi}] vs declared open interval "
          f"({exp['lo_excl']}, {exp['hi_excl']})")
        # the declared era end must cover the last window's full span+markout
        need = hi + 300 + 5.0 + 5.0
        p("declared_era_end_covers_last_window",
          need <= exp["declared_era_end_s"],
          f"last window needs {need}, declared end "
          f"{exp['declared_era_end_s']}")
    else:
        p("t0_within_declared_bounds", False, "NO ROWS -- nothing to bound")
        p("declared_era_end_covers_last_window", False, "NO ROWS")

    p("dataset_non_empty", n_rows > 0 and bool(built_slugs),
      f"{n_rows} rows across {len(built_slugs)} slugs")
    return out


def verify(built_path: Path = BUILT,
           receipt_path: Path | None = None) -> dict[str, Any]:
    if not built_path.exists():
        raise VerificationFailed(
            f"{built_path} does not exist. Refusing to report a verdict on an "
            f"absent artifact -- an absent dataset is not a passing one.")
    exp = expectations(receipt_path)
    built_slugs: set[str] = set()
    t0s: dict[str, int] = {}
    by = collections.defaultdict(lambda: collections.defaultdict(
        collections.Counter))
    statuses: collections.Counter = collections.Counter()
    n_rows = 0
    for r in iter_rows(built_path):
        n_rows += 1
        slug = r.get("slug")
        if slug is None:
            statuses["__NO_SLUG__"] += 1
            continue
        built_slugs.add(slug)
        if slug not in t0s:
            try:
                t0s[slug] = int(slug.rsplit("-", 1)[1])
            except (IndexError, ValueError):
                statuses["__BAD_SLUG__"] += 1
        st = r.get("status", "__MISSING__")
        statuses[st] += 1
        by[r.get("coin")][r.get("day")][st] += 1
    preds = check(built_slugs, t0s, exp, n_rows)
    return {
        "built": str(built_path), "receipt": exp["receipt_version"],
        "n_rows": n_rows, "n_slugs": len(built_slugs),
        "status_counts": dict(statuses),
        "by_coin_day_status": {c: {d: dict(s) for d, s in dd.items()}
                               for c, dd in by.items()},
        "predicates": preds,
        "all_pass": all(p["pass"] for p in preds),
    }


def _selftests() -> int:
    import tempfile
    checks = 0

    def ok(cond, label):
        nonlocal checks
        checks += 1
        if not cond:
            raise AssertionError(f"selftest failed: {label}")

    exp = {"ok_slugs": {"btc-updown-5m-1787650500",
                        "eth-updown-5m-1787650500"},
           "n_ok": 2, "lo_excl": 1787650200, "hi_excl": 1787702400,
           "era_floor_recv_ns": 1787579334881534478,
           "declared_era_end_s": 1787702410.0}
    good = {"btc-updown-5m-1787650500": 1787650500,
            "eth-updown-5m-1787650500": 1787650500}

    def run(slugs, t0s, n_rows=10):
        return {p["predicate"]: p["pass"]
                for p in check(set(slugs), t0s, exp, n_rows)}

    # POSITIVE CONTROL: a correct build must pass everything.
    r = run(good, good)
    ok(all(r.values()), f"a correct build passes every predicate ({r})")

    # FALSIFIERS -- each must FIRE, or the checker is decoration (rule 15).
    r = run({k: v for k, v in list(good.items())[:1]},
            {k: v for k, v in list(good.items())[:1]})
    ok(not r["slug_set_identity"], "a MISSING slug is caught")
    ok(not r["n_ok_matches_receipt"], "and the count check catches it too")

    leak = dict(good); leak["btc-updown-5m-1787702400"] = 1787702400
    r = run(leak, leak)
    ok(not r["no_reserved_forward_tape"],
       "RESERVED FORWARD TAPE at exactly 08-26T00:00Z is caught")
    ok(not r["slug_set_identity"], "and it also shows up as an extra slug")
    leak2 = dict(good); leak2["btc-updown-5m-1787702700"] = 1787702700
    ok(not run(leak2, leak2)["no_reserved_forward_tape"],
       "forward tape past the boundary is caught")

    cons = dict(good); cons["btc-updown-5m-1787650200"] = 1787650200
    ok(not run(cons, cons)["no_consumed_fragment"],
       "re-entering the consumed fragment at its last slug is caught")

    ok(not run({}, {}, n_rows=0)["dataset_non_empty"],
       "an EMPTY dataset FAILS -- it never silently passes")
    ok(not run({}, {}, n_rows=0)["t0_within_declared_bounds"],
       "and an empty dataset cannot satisfy the bounds check either")

    # boundary exactness: the receipt's interval is OPEN at both ends
    edge_lo = {"btc-updown-5m-1787650500": 1787650500}
    ok(run(edge_lo, edge_lo)["no_consumed_fragment"],
       "the first admissible slug is NOT flagged as consumed")
    last = {"btc-updown-5m-1787702100": 1787702100}
    ok(run(last, last)["no_reserved_forward_tape"],
       "the last admissible slug is NOT flagged as forward tape")
    ok(run(last, last)["declared_era_end_covers_last_window"],
       "the declared end exactly covers the last admissible window "
       "(1787702100 + 310 = 1787702410)")
    over = {"btc-updown-5m-1787702400": 1787702400}
    ok(not run(over, over)["declared_era_end_covers_last_window"],
       "a window needing more than the declared end is caught")

    # refuse on an absent artifact rather than reporting a pass
    try:
        verify(Path("/nonexistent/topup.json"))
    except VerificationFailed:
        ok(True, "an ABSENT artifact is refused, never reported as passing")
    else:
        ok(False, "MUST refuse to verify a dataset that does not exist")

    # --- the resolver follows the ruling forward, not a hardcoded version --
    ok(_receipt_version_key(Path("x_v3.json")) >
       _receipt_version_key(Path("x_v2.json")), "v3 sorts above v2")
    ok(_receipt_version_key(Path("x_v2.1.json")) >
       _receipt_version_key(Path("x_v2.json")), "dotted v2.1 sorts above v2")
    ok(_receipt_version_key(Path("x_v10.json")) >
       _receipt_version_key(Path("x_v9.json")), "v10 sorts above v9, not below")
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        for v in ("v1", "v2", "v10", "v2.1"):
            (d / f"da_development_topup_{v}.json").write_text("{}", encoding="utf-8")
        ok(latest_receipt(d).name == "da_development_topup_v10.json",
           "the highest version wins among mixed integer and dotted versions")
        try:
            latest_receipt(Path(td) / "empty")
        except VerificationFailed:
            ok(True, "no receipt at all is REFUSED, not defaulted")
        else:
            ok(False, "MUST refuse when no manifest exists")

    # the streaming scanner round-trips
    with tempfile.TemporaryDirectory() as td:
        f = Path(td) / "rows.json"
        rows = [{"slug": "btc-updown-5m-1787650500", "coin": "btc",
                 "day": "2026-08-25", "status": "OK", "note": "a } brace"},
                {"slug": "eth-updown-5m-1787650800", "coin": "eth",
                 "day": "2026-08-25", "status": "OK",
                 "note": 'quote " here'}]
        f.write_text(json.dumps({"rows": rows, "n_windows": 2}),
                     encoding="utf-8")
        got = list(iter_rows(f))
        ok(got == rows,
           "the scanner round-trips rows containing braces and quotes")

    print(f"da_topup_population_verify selftests: {checks} checks passed")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("cmd", nargs="?", choices=["verify"])
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--built", default=None)
    a = ap.parse_args()
    if a.selftest or not a.cmd:
        return _selftests()
    rep = verify(Path(a.built) if a.built else BUILT)
    print(json.dumps({k: v for k, v in rep.items()
                      if k != "predicates"}, indent=2, sort_keys=True))
    print("\nPREDICATES")
    for p in rep["predicates"]:
        print(f"  [{'PASS' if p['pass'] else 'FAIL'}] {p['predicate']}: "
              f"{p['detail']}")
    print(f"\nALL PASS: {rep['all_pass']}")
    return 0 if rep["all_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
