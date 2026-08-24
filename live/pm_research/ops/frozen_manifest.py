"""Make R-28's append-only rule ENFORCEABLE instead of declared.

R-28 says frozen documents are append-only: corrections are annotations BESIDE
the original, never edits to it. **Nothing enforced that.** 15 artifacts declare
`FROZEN` and 5 of them are not even tracked by git, so a silent edit to a frozen
protocol would leave no evidence anywhere.

R-57 condition 4 protects frozen artifacts from being broken by a contract
change. This protects them from being changed at all — the complementary hole,
and the one nobody was watching.

**It encodes R-28's actual semantics, not a blanket checksum.** A whole-file hash
would flag a LEGAL annotation as a violation, which would get the check muted.
So the seal records the file's length and the hash of exactly that prefix:

    prefix unchanged, file longer   -> APPEND      (legal under R-28)
    prefix unchanged, same length   -> UNCHANGED
    prefix changed, or shorter      -> EDITED      (R-28 VIOLATION)

**Honest limit, stated because it cannot be fixed retroactively:** sealing today
establishes a baseline from today. It cannot prove nothing was edited BEFORE the
seal. It makes future drift detectable, which is all a seal can ever do.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

REPO = Path("/home/yuqing/ctaNew")
SEARCH = [REPO / "live/pm_research", REPO / "live/pm_research/plans"]
MANIFEST = REPO / "data/pm_5min/ops/frozen_manifest.json"
FROZEN_RE = re.compile(r"status.{0,40}frozen", re.I)


def frozen_artifacts() -> list[Path]:
    out: list[Path] = []
    for d in SEARCH:
        if not d.is_dir():
            continue
        for f in sorted(d.iterdir()):
            if f.suffix not in (".md", ".yaml") or not f.is_file():
                continue
            try:
                head = f.read_text(errors="replace")[:4000]
            except OSError:
                continue
            if FROZEN_RE.search(head):
                out.append(f)
    return out


def prefix_digest(path: Path, n: int) -> str | None:
    try:
        with path.open("rb") as fh:
            return hashlib.sha256(fh.read(n)).hexdigest()
    except OSError:
        return None


def seal() -> int:
    entries = {}
    for f in frozen_artifacts():
        size = f.stat().st_size
        entries[str(f.relative_to(REPO))] = {
            "sealed_len": size,
            "prefix_sha256": prefix_digest(f, size),
        }
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps({"artifacts": entries}, indent=2, sort_keys=True) + "\n")
    print(f"  sealed {len(entries)} frozen artifacts -> {MANIFEST}")
    print("  NOTE: baseline starts NOW; this cannot prove anything about edits before the seal.")
    return 0


def verify() -> int:
    if not MANIFEST.exists():
        print("  FAIL  no manifest — run --seal first")
        return 2
    sealed = json.loads(MANIFEST.read_text())["artifacts"]
    rows, worst = [], 0
    for rel, rec in sorted(sealed.items()):
        f = REPO / rel
        if not f.is_file():
            rows.append(("MISSING", rel, "sealed artifact is gone")); worst = 2; continue
        size = f.stat().st_size
        got = prefix_digest(f, rec["sealed_len"])
        if got is None:
            rows.append(("FAIL", rel, "unreadable — an error is never a skip")); worst = 2
        elif got != rec["prefix_sha256"]:
            rows.append(("EDITED", rel, "R-28 VIOLATION: sealed prefix changed")); worst = 2
        elif size > rec["sealed_len"]:
            rows.append(("APPEND", rel, f"+{size - rec['sealed_len']} bytes appended (legal)"))
        elif size < rec["sealed_len"]:
            rows.append(("EDITED", rel, "R-28 VIOLATION: file shortened")); worst = 2
        else:
            rows.append(("OK", rel, ""))
    # a newly frozen artifact that is not in the manifest is a gap, not a pass
    current = {str(f.relative_to(REPO)) for f in frozen_artifacts()}
    for rel in sorted(current - set(sealed)):
        rows.append(("UNSEALED", rel, "declares FROZEN but is not in the manifest"))
        worst = max(worst, 1)
    width = max(len(r[1]) for r in rows) if rows else 10
    for status, rel, note in rows:
        print(f"  {status:9s} {rel:<{width}}  {note}")
    ok = sum(1 for r in rows if r[0] in ("OK", "APPEND"))
    print(f"  {ok}/{len(rows)} frozen artifacts intact under R-28")
    return worst


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--seal", action="store_true")
    g.add_argument("--verify", action="store_true")
    a = ap.parse_args()
    return seal() if a.seal else verify()


if __name__ == "__main__":
    raise SystemExit(main())
