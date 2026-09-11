"""AS-OF ASSERTION for inputs too large to copy.

`raw/<day>/` is 4.4 GB and 2,016 files for one day, so the snapshot cannot
freeze it by copying. What it can do is ASSERT it: record every file of the
DAY'S SLICE with its size and sha256 at launch, and re-verify the same
listing when the valuation exits.

Growth OUTSIDE the day's slice is allowed and counted -- the collector is
still writing later windows and that cannot touch a closed day. A change
INSIDE the slice refuses RUN_INPUT_MOVED_DURING_RUN:<file>.
"""
from __future__ import annotations
import hashlib, json, sys
from pathlib import Path

MOVED = "RUN_INPUT_MOVED_DURING_RUN"


class AsOfRefused(RuntimeError):
    """A named refusal."""


def listing(root: Path, day: str) -> dict:
    """The day's slice of raw/, file by file."""
    d = Path(root) / "data" / "pm_5min" / "raw" / day.replace("-", "")
    files = {}
    if d.is_dir():
        for f in sorted(d.iterdir()):
            if f.is_file():
                files[f.name] = {
                    "size": f.stat().st_size,
                    "sha256": hashlib.sha256(f.read_bytes()).hexdigest()}
    return {"dir": str(d), "n_files": len(files),
            "total_bytes": sum(v["size"] for v in files.values()),
            "files": files}


def verify(before: dict, root: Path, day: str) -> dict:
    """Re-verify at exit. Growth outside the slice is not this listing."""
    after = listing(root, day)
    changed, vanished, appeared = [], [], []
    for name, v in before["files"].items():
        w = after["files"].get(name)
        if w is None:
            vanished.append(name)
        elif w != v:
            changed.append(name)
    for name in after["files"]:
        if name not in before["files"]:
            appeared.append(name)
    bad = changed + vanished
    out = {"n_before": before["n_files"], "n_after": after["n_files"],
           "changed": changed, "vanished": vanished,
           "appeared_in_the_slice": appeared,
           "growth_outside_the_slice_is_allowed_and_counted": True}
    if bad:
        raise AsOfRefused(
            f"REFUSED {MOVED}:{bad[0]} -- {len(bad)} file(s) of the day's "
            f"own slice changed or vanished while the run was reading "
            f"them. A closed day's inputs must not move.")
    return {**out, "status": "PASS"}


def falsify() -> int:
    import tempfile, os
    cells = ok = 0

    def ck(n, c):
        nonlocal cells, ok
        cells += 1
        ok += bool(c)
        print(f"  [{'PASS' if c else 'FAIL'}] {n}")

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        d = root / "data" / "pm_5min" / "raw" / "20260908"
        d.mkdir(parents=True)
        for i in range(4):
            (d / f"btc-{i}.jsonl.gz").write_bytes(b"x" * (10 + i))
        b = listing(root, "2026-09-08")
        ck("listing sees the day's files", b["n_files"] == 4)
        ck("unchanged -> PASS", verify(b, root, "2026-09-08")["status"] == "PASS")
        (root / "data" / "pm_5min" / "raw" / "20260909").mkdir()
        (root / "data" / "pm_5min" / "raw" / "20260909" / "x.gz").write_bytes(b"y")
        ck("growth OUTSIDE the slice is allowed",
           verify(b, root, "2026-09-08")["status"] == "PASS")
        (d / "btc-1.jsonl.gz").write_bytes(b"CHANGED")
        try:
            verify(b, root, "2026-09-08")
            ck("touching a file INSIDE the slice REFUSES by name", False)
        except AsOfRefused as e:
            ck("touching a file INSIDE the slice REFUSES by name",
               f"{MOVED}:btc-1.jsonl.gz" in str(e))
    print(f"\n{ok}/{cells} cells pass")
    return 0 if ok == cells else 1


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--falsify" in argv:
        return falsify()
    mode, root, day = argv[0], Path(argv[1]), argv[2]
    if mode == "listing":
        print(json.dumps(listing(root, day)))
        return 0
    before = json.loads(Path(argv[3]).read_text())
    try:
        print(json.dumps(verify(before, root, day)))
        return 0
    except AsOfRefused as exc:
        print(str(exc))
        return 8


if __name__ == "__main__":
    raise SystemExit(main())
