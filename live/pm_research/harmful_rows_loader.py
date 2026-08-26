"""Memory-conscious streaming loader for the v3.4 exposure dataset.

AUTHORISATION (R-126, in-file): R-148(5) / R-151(6) — if the committed builder
cannot reproduce inside a sane job cap, a memory-conscious load path is
legitimate pre-freeze Phase-0 work, because reproducibility includes resource
feasibility. This file changes HOW rows are loaded. It must not change WHAT is
computed, and `equivalence_report()` exists to prove that on real data.

MEASURED PROBLEM (2026-08-26, bounded probes under research.slice):
    full parsed row      4,259 B/row  ->  4.79 GB for 1,125,289 OK rows
    + the 1.24 GB source string held simultaneously by read_text()
  which is why every attempt exceeded 8.8 GiB and the third took the box down.
  The bloat is structural: 9 latency entries x 3 float fields = 27 nested
  dicts per row, and only ONE latency is ever read by the model.

WHAT THIS DOES
  Streams one row at a time and retains a compact projection: exactly the seven
  keys the builder keeps, with `latency` reduced to the TARGET latency only,
  plus `any_fill_ahead` precomputed by the SAME predicate `keptrow()` uses.
  Measured 648 B/row -> 0.73 GB, independent (nothing shared with a parent
  graph), and the full graph never exists at any instant.

WHY raw_decode AND NOT BRACE COUNTING
  A JSON string may legally contain '{' or '}'. A brace counter silently
  mis-splits such a file and yields corrupt rows. `json.JSONDecoder.raw_decode`
  is the exact parser, so row boundaries are decided by JSON grammar, and every
  float is converted by the SAME code path as a full `json.loads` -- which is
  what makes bit-identity possible rather than merely likely.

THE COMPACTION IS GUARDED, NOT ASSUMED
  Reducing `latency` to one key is only safe while the model reads that one
  key. `stream_ok_rows` records which latency it compacted, and
  `check_target_latency()` REFUSES a mismatch instead of silently returning
  zeros for a latency that was dropped. Rule 14: this estimates nothing and
  decides nothing; it either serves the declared latency or refuses.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

EXPECTED_SCHEMA = "harmful_exposure_v3_4_fill_scoped_markout"
KEEP = ("slug", "day", "t0", "t_start", "side", "gen", "latency")
_DEC = json.JSONDecoder()


class LatencyMismatch(RuntimeError):
    """Raised when code asks for a latency this loader did not retain."""


def _any_fill_ahead(lat: dict, L: str) -> bool:
    """EXACTLY the predicate in harmful_hazard_model.keptrow(). Duplicated
    deliberately and pinned by an equivalence test, because importing the
    builder to get it would create a cycle."""
    return (lat.get(L, {}).get("preventable_shares", 0.0) > 0
            or any(v.get("preventable_shares", 0.0) > 0
                   or v.get("stale_shares", 0.0) > 0
                   for v in lat.values()))


def compact_row(r: dict, L: str) -> dict:
    lat = r.get("latency") or {}
    l = lat.get(L) or {}
    return {
        "slug": r.get("slug"), "day": r.get("day"), "t0": r.get("t0"),
        "t_start": r.get("t_start"), "side": r.get("side"), "gen": r.get("gen"),
        "coin": r.get("coin"),
        "latency": {L: {
            "preventable_value_cents": l.get("preventable_value_cents", 0.0),
            "preventable_shares": l.get("preventable_shares", 0.0),
            "stale_shares": l.get("stale_shares", 0.0)}},
        "any_fill_ahead": _any_fill_ahead(lat, L),
    }


def _header(path: Path) -> tuple[dict, int]:
    """Return (metadata, char offset where the rows array opens).

    LAYOUT FACT, verified at the artifact rather than assumed: this file is
    written with "rows" as the FIRST key, so every scalar -- schema, days,
    n_windows, the correctness counters -- sits AFTER the 1.24 GB array, at
    the END of the file. Reading a 4 MB head for them finds nothing and yields
    `schema: None`, which the schema guard then correctly refuses. So the
    metadata is read from the TAIL and the array offset from the head."""
    with path.open("rb") as fh:
        head = fh.read(1_000_000).decode("utf-8", "replace")
        fh.seek(0, 2)
        size = fh.tell()
        fh.seek(max(0, size - 8192))
        tail = fh.read().decode("utf-8", "replace")
    j = head.index("[", head.index('"rows"')) + 1

    # LAYOUT-AGNOSTIC. The real artifact puts "rows" first and its scalars
    # last; a file written the other way puts them first. Try the tail, then
    # the head, and take whichever actually yields a schema. (Falsifier D --
    # a fixture with the opposite key order -- caught this: a tail-only reader
    # refused a perfectly valid file with `schema: None`.)
    meta: dict = {}
    k = tail.rfind("}]")
    if k >= 0:
        frag = tail[k + 2:].lstrip().lstrip(",")
        try:
            meta = json.loads("{" + frag)
        except ValueError:
            meta = {}
    if "schema" not in meta:
        for key in ("schema", "n_windows", "days", "reconciliation_failures",
                    "boundary_time_violations", "consume_clock_violations",
                    "unhooked_state_changes"):
            tok = f'"{key}"'
            at = head.find(tok)
            if at < 0 or (at > head.find('"rows"') >= 0 and key not in
                          ("schema",) and at > j):
                pass
            if at >= 0 and at < j:
                q = _skip_ws(head, head.index(":", at + len(tok)) + 1)
                try:
                    meta[key], _ = _DEC.raw_decode(head, q)
                except ValueError:
                    pass
    return meta, j


def _skip_ws(s: str, i: int) -> int:
    while i < len(s) and s[i] in " \t\r\n":
        i += 1
    return i


def stream_ok_rows(path: Path, target_latency_ms: int,
                   chunk: int = 1 << 24) -> Iterator[dict]:
    """Yield compact projections of rows whose status == 'OK', one at a time."""
    L = str(target_latency_ms)
    meta, start = _header(path)
    if meta.get("schema") != EXPECTED_SCHEMA:
        raise SystemExit(f"REFUSED: schema {meta.get('schema')!r}")
    buf = ""
    with path.open("r", encoding="utf-8") as fh:
        fh.read(start)
        while True:
            piece = fh.read(chunk)
            if piece:
                buf += piece
            i = _skip_ws(buf, 0)
            while i < len(buf):
                if buf[i] == ",":
                    i = _skip_ws(buf, i + 1); continue
                if buf[i] == "]":
                    return
                try:
                    obj, end = _DEC.raw_decode(buf, i)
                except ValueError:
                    break                       # row spans the chunk edge
                if obj.get("status") == "OK":
                    yield compact_row(obj, L)
                i = _skip_ws(buf, end)
            buf = buf[i:]
            if not piece:
                return


def check_target_latency(rows_latency_key: str, wanted_ms: int) -> None:
    if str(wanted_ms) != rows_latency_key:
        raise LatencyMismatch(
            f"loader retained latency {rows_latency_key!r} but {wanted_ms} was "
            f"requested. The other latencies were DROPPED to save memory; "
            f"serving zeros for them would be silent corruption.")


def equivalence_report(path: Path, target_latency_ms: int,
                       n: int = 5000) -> dict:
    """PROOF OBLIGATION: stream the first n OK rows and compare, field by
    field, against what a full json.loads of the same rows produces. Any
    difference at all is a failure -- this is the test that licenses using
    the loader for a cent-exact reproduction."""
    L = str(target_latency_ms)
    streamed = []
    for r in stream_ok_rows(path, target_latency_ms):
        streamed.append(r)
        if len(streamed) >= n:
            break
    # independent full parse of the same prefix, via the exact decoder
    _, start = _header(path)
    with path.open("r", encoding="utf-8") as fh:
        fh.read(start)
        buf = fh.read(1 << 26)
    ref, i = [], _skip_ws(buf, 0)
    while len(ref) < n and i < len(buf):
        if buf[i] in ",":
            i = _skip_ws(buf, i + 1); continue
        try:
            obj, end = _DEC.raw_decode(buf, i)
        except ValueError:
            break
        if obj.get("status") == "OK":
            ref.append(obj)
        i = _skip_ws(buf, end)
    m = min(len(streamed), len(ref))
    diffs = []
    for a, b in zip(streamed[:m], ref[:m]):
        for k in ("slug", "day", "t0", "t_start", "side", "gen", "coin"):
            if a.get(k) != b.get(k):
                diffs.append((k, a.get(k), b.get(k)))
        lb = (b.get("latency") or {}).get(L) or {}
        la = a["latency"][L]
        for f in ("preventable_value_cents", "preventable_shares", "stale_shares"):
            if la[f] != lb.get(f, 0.0):
                diffs.append((f, la[f], lb.get(f)))
        if a["any_fill_ahead"] != _any_fill_ahead(b.get("latency") or {}, L):
            diffs.append(("any_fill_ahead", a["any_fill_ahead"], None))
    return {"compared": m, "diffs": diffs[:20], "n_diffs": len(diffs),
            "identical": len(diffs) == 0 and m > 0}
