#!/usr/bin/env python3
"""Emit a machine-readable contamination record for a collector incident.

Rule 4: exclusions are STATUSES with counts, never silent drops. A day whose
tape is partly unusable must say so in a form a reader resolves, not in prose
beside the table. This enumerates every affected file with its MEASURED
retention so a consumer can censor an interval rather than quietly average
over a hole.

WHAT IT DOES NOT DO: it does not take the incident's scope on report. The
affected windows are DISCOVERED from the collector's own gzip failures -- the
process's declaration that another writer finalized its file -- because the
first account of this incident named one window and the log named three. An
instrument that is told what to find cannot correct the person telling it.

BASELINE INTEGRITY IS THE LOAD-BEARING PART. Retention is measured against
clean neighbouring windows, so a baseline that is ITSELF damaged would make the
damage read as ~100% and vanish -- the one arithmetic that silently exonerates
an incident. The guard cannot be "is the baseline in the affected list": every
baseline lies before the earliest affected window, so that test can never fire.
It is whether a baseline file was FINALIZED DURING THE INCIDENT, which is
observable from its mtime and does not depend on the collector having noticed.
"""
from __future__ import annotations

import argparse
import gzip
import json
import re
import statistics
import sys
from pathlib import Path

SCHEMA = "pm5_contamination_v1"
GZFAIL = re.compile(r"gzip ([a-z0-9]+)-updown-5m-(\d+): \[Errno 2\]")
WINDOW_S = 300


class Refused(Exception):
    """The record will not be written from evidence it cannot stand behind."""


def affected_from_log(log_text: str) -> dict[int, set[str]]:
    """Windows another process finalized, per the collector's OWN failures."""
    out: dict[int, set[str]] = {}
    for coin, win in GZFAIL.findall(log_text):
        out.setdefault(int(win), set()).add(coin)
    return out


def window_lines(raw_dir: Path, coin: str, window: int) -> int | None:
    gz = raw_dir / f"{coin}-updown-5m-{window}.jsonl.gz"
    if gz.exists():
        with gzip.open(gz, "rb") as fh:
            return sum(1 for _ in fh)
    plain = raw_dir / f"{coin}-updown-5m-{window}.jsonl"
    if plain.exists():
        with plain.open("rb") as fh:
            return sum(1 for _ in fh)
    return None


def build_record(raw_dir: Path, log_text: str, incident: dict,
                 n_baseline: int = 3, as_of: str = "",
                 span_epoch: tuple[float, float] | None = None) -> dict:
    affected = affected_from_log(log_text)
    if not affected:
        raise Refused(
            "REFUSED: no gzip-race evidence in the log. A record asserting "
            "'nothing was contaminated' from an instrument that found nothing "
            "is a zero from a checker that never proved it can fire (rule 15). "
            "If the incident left no trace here, say so in the register.")
    if span_epoch is None:
        raise Refused(
            "REFUSED: no incident span given, so no baseline window can be "
            "shown to predate the incident. Retention against an unvalidated "
            "baseline is not a measurement.")
    first = min(affected)
    base_windows = [first - WINDOW_S * k for k in range(n_baseline, 0, -1)]
    lo, hi = span_epoch
    tainted = []
    for w in base_windows:
        for coin in {c for cs in affected.values() for c in cs}:
            for suf in (".jsonl.gz", ".jsonl"):
                f = raw_dir / f"{coin}-updown-5m-{w}{suf}"
                if f.exists() and lo <= f.stat().st_mtime <= hi:
                    tainted.append(f.name)
    if tainted:
        raise Refused(
            f"REFUSED: baseline file(s) {sorted(tainted)[:4]} were FINALIZED "
            f"DURING the incident. Measuring retention against them would make "
            f"the damage read as ~100% and vanish. This is checked from mtimes "
            f"rather than from the collector's error list, because a baseline "
            f"damaged WITHOUT the collector noticing is exactly the case the "
            f"error list cannot report.")
    coins = sorted({c for cs in affected.values() for c in cs})
    files, base = [], {}
    for coin in coins:
        obs = [n for n in (window_lines(raw_dir, coin, w)
                           for w in base_windows) if n]
        if not obs:
            raise Refused(
                f"REFUSED: no readable baseline window for {coin!r}; its "
                f"retention would be unmeasurable and reporting it as 0 or "
                f"100 would both be inventions")
        base[coin] = statistics.median(obs)
    for window in sorted(affected):
        for coin in sorted(affected[window]):
            n = window_lines(raw_dir, coin, window)
            pct = round(100.0 * n / base[coin], 2) if n is not None else None
            files.append({
                "coin": coin, "window": window,
                "lines_captured": n,
                "baseline_median_lines": base[coin],
                "retention_pct": pct,
                "status": ("UNMEASURABLE" if pct is None
                           else "INFLATED_DUPLICATE_ROWS" if pct > 100.0
                           else "TRUNCATED_BY_FOREIGN_PROCESS"),
            })
    lost = sum(max(0, r["baseline_median_lines"] - (r["lines_captured"] or 0))
               for r in files)
    return {
        "schema": SCHEMA,
        "as_of": as_of,
        "incident": incident,
        "method": {
            "scope_discovery": "collector's own gzip [Errno 2] failures",
            "baseline": f"per-coin median of the {n_baseline} windows "
                        f"immediately preceding the earliest affected window",
            "retention": "gz/plain line count vs that baseline",
            "caveat": "line volume varies with market activity; observed "
                      "baseline spread is roughly +/-20%, so retention near "
                      "100% is not evidence of no damage, and only large "
                      "deficits are load-bearing",
        },
        "windows_affected": sorted(affected),
        "coins": coins,
        "files_affected": len(files),
        "estimated_lines_never_captured": int(lost),
        "recoverable": False,
        "recovery_note": "the foreign process finalized the file; the "
                         "collector's handle then referred to an unlinked "
                         "inode, so subsequent rows were written nowhere",
        "files": files,
    }


def _selftests() -> int:
    import tempfile
    checks = 0

    def ok(cond, label):
        nonlocal checks
        if not cond:
            raise AssertionError(f"selftest failed: {label}")
        checks += 1

    def mkgz(d: Path, coin: str, w: int, n: int):
        with gzip.open(d / f"{coin}-updown-5m-{w}.jsonl.gz", "wt") as fh:
            for i in range(n):
                fh.write(f"{i}\t[]\n")

    W = 1788153900
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        for k in (3, 2, 1):
            mkgz(d, "btc", W - WINDOW_S * k, 1000)
        mkgz(d, "btc", W, 5)
        log = f"[pm] gzip btc-updown-5m-{W}: [Errno 2] No such file\n"
        r = build_record(d, log, {"cause": "test"}, as_of="T",
                         span_epoch=(0.0, 1.0))
        ok(r["windows_affected"] == [W] and r["files_affected"] == 1,
           "scope is DISCOVERED from the collector's own gzip failures, not "
           "supplied -- an instrument told what to find cannot correct the "
           "account that told it")
        f0 = r["files"][0]
        ok(f0["retention_pct"] == 0.5
           and f0["status"] == "TRUNCATED_BY_FOREIGN_PROCESS"
           and r["estimated_lines_never_captured"] == 995,
           "POSITIVE CONTROL: a window cut to 5 of a 1000-line baseline is "
           "flagged at 0.5% retention with the deficit counted")

        mkgz(d, "eth", W, 1400)
        for k in (3, 2, 1):
            mkgz(d, "eth", W - WINDOW_S * k, 1000)
        log2 = log + f"[pm] gzip eth-updown-5m-{W}: [Errno 2] No such file\n"
        r2 = build_record(d, log2, {"cause": "test"}, as_of="T",
                          span_epoch=(0.0, 1.0))
        ok(any(x["status"] == "INFLATED_DUPLICATE_ROWS" and x["coin"] == "eth"
               for x in r2["files"]),
           "and an OVER-full window is a DIFFERENT status -- duplication and "
           "truncation are opposite failures and must not share a label")

        import os
        bfile = d / f"btc-updown-5m-{W - WINDOW_S}.jsonl.gz"
        os.utime(bfile, (500.0, 500.0))
        refused = ""
        try:
            build_record(d, log2, {"cause": "test"}, as_of="T",
                         span_epoch=(400.0, 600.0))
        except Refused as e:
            refused = str(e)
        ok("FINALIZED DURING the incident" in refused,
           "KNOWN-BAD: a baseline file finalized DURING the incident REFUSES. "
           "Checked from MTIME, not from the collector's error list -- a "
           "baseline damaged without the collector noticing is precisely what "
           "the error list cannot report. (My first version of this guard "
           "asked whether a baseline was in the affected list, which can NEVER "
           "fire: every baseline lies before the earliest affected window)")
        nospan = ""
        try:
            build_record(d, log, {"cause": "test"}, as_of="T")
        except Refused as e:
            nospan = str(e)
        ok("no incident span given" in nospan,
           "and omitting the span REFUSES -- an unvalidated baseline is not a "
           "measurement")
        empty = ""
        try:
            build_record(d, "no errors here", {"cause": "test"}, as_of="T",
                         span_epoch=(0.0, 1.0))
        except Refused as e:
            empty = str(e)
        ok("never proved it can fire" in empty,
           "KNOWN-BAD: no evidence in the log REFUSES rather than emitting a "
           "record that asserts cleanliness (rule 15)")
        miss = ""
        try:
            build_record(d, log2.replace("eth", "sol"), {"cause": "test"},
                         as_of="T", span_epoch=(0.0, 1.0))
        except Refused as e:
            miss = str(e)
        ok("no readable baseline" in miss,
           "and a coin with no baseline REFUSES -- reporting it as 0% or 100% "
           "would both be inventions")
    print(f"da_contamination_record selftests: {checks} checks passed")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw-dir", type=Path)
    ap.add_argument("--log", type=Path)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--as-of", default="")
    ap.add_argument("--incident-json", default="{}")
    ap.add_argument("--span-start-epoch", type=float)
    ap.add_argument("--span-end-epoch", type=float)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return _selftests()
    if not (a.raw_dir and a.log and a.out and a.as_of):
        ap.error("--raw-dir, --log, --out and --as-of are required")
    try:
        rec = build_record(a.raw_dir, a.log.read_text(errors="replace"),
                           json.loads(a.incident_json), as_of=a.as_of,
                           span_epoch=(a.span_start_epoch, a.span_end_epoch)
                           if a.span_start_epoch and a.span_end_epoch else None)
    except Refused as e:
        print(e, file=sys.stderr)
        return 2
    a.out.write_text(json.dumps(rec, indent=2, sort_keys=True) + "\n")
    print(f"{a.out}: {rec['files_affected']} files across "
          f"{len(rec['windows_affected'])} windows, "
          f"~{rec['estimated_lines_never_captured']:,} lines never captured")
    return 0


if __name__ == "__main__":
    sys.exit(main())
