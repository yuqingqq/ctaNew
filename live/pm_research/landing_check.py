"""Landing-evidence checks that fail CLOSED.

A grep returning zero matches is AMBIGUOUS. The text may genuinely be absent,
or the instrument may be broken: a phrase split across a line wrap, markdown
emphasis inside the phrase, a curly quote, a shell variable lost across a `cd`,
a path that does not resolve. Every one of those reports ABSENT for text that
is PRESENT.

That is the fail-open direction, and it is the dangerous one, because a false
ABSENT looks like diligence catching something. It is the exact mirror of a
gate that cannot fire: both report the safe-looking answer regardless of the
world. This programme has now produced four of them -- two by the coordinator
(a grep that matched the comment DOCUMENTING a removal, and a shell variable
lost across a `cd`) and two by DA (a pattern broken by a line wrap, another by
shell escaping).

The rule this module enforces:

    NEVER REPORT ABSENT UNLESS A POSITIVE CONTROL ON THE SAME FILE, THROUGH
    THE SAME CODE PATH, HAS JUST SUCCEEDED.

Two controls run before any ABSENT verdict:

  1. SELF-SLICE -- a slice of the file's own normalised text is searched for
     using the same matcher. If that fails, the matcher is broken, and the
     verdict is INSTRUMENT_BROKEN rather than ABSENT.
  2. IDENTITY (optional, recommended) -- a phrase the caller asserts identifies
     this document. If it fails, you are reading the wrong file, and the
     verdict is INSTRUMENT_BROKEN rather than ABSENT.

And the second failure mode, the fail-CLOSED-looking one that is really
fail-open in the other direction: matching the PROSE ABOUT a thing instead of
the thing. `reject_lines_matching` splits matches into accepted and rejected,
and every probe returns the MATCHING LINES, never just a count -- so the reader
can see whether what matched was the fact or the talk about the fact.

DA-plane tooling. Additive, sets no CHOSEN value, touches no contract.
Offered to every plane; nothing depends on it.
"""

from __future__ import annotations

import bisect
import re
import sys
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

PRESENT = "PRESENT"
ABSENT = "ABSENT"
INSTRUMENT_BROKEN = "INSTRUMENT_BROKEN"

# Emphasis and code markers are stripped: a phrase quoted from rendered text
# will not carry the `**` that surrounds it in source.
_EMPHASIS = re.compile(r"[*_`]")
# Curly quotes and non-breaking spaces are folded to their ASCII forms.
_FOLD = {
    "‘": "'", "’": "'", "“": '"', "”": '"',
    " ": " ", " ": " ", " ": " ",
}
# Dashes are deliberately NOT folded. Em dash and hyphen are different
# characters that mean different things in these documents, and folding them
# would trade a false ABSENT for a false PRESENT.


def normalise(text: str) -> str:
    """Fold the differences that break greps but not meaning."""
    text = unicodedata.normalize("NFC", text)
    for src, dst in _FOLD.items():
        text = text.replace(src, dst)
    text = _EMPHASIS.sub("", text)
    return " ".join(text.split())


@dataclass(frozen=True)
class Probe:
    outcome: str
    phrase: str
    path: str
    accepted: tuple[tuple[int, str], ...] = ()
    rejected: tuple[tuple[int, str], ...] = ()
    controls: tuple[tuple[str, bool, str], ...] = ()
    reason: str = ""

    @property
    def count(self) -> int:
        return len(self.accepted)

    def render(self) -> str:
        head = f"{self.outcome}  n={self.count}  {self.phrase!r}  in {self.path}"
        out = [head]
        if self.reason:
            out.append(f"    reason: {self.reason}")
        for lineno, line in self.accepted[:5]:
            out.append(f"    L{lineno}: {line.strip()[:100]}")
        for lineno, line in self.rejected[:5]:
            out.append(f"    L{lineno}: [REJECTED] {line.strip()[:100]}")
        for name, ok, detail in self.controls:
            out.append(f"    control {name}: {'pass' if ok else 'FAIL'} {detail}")
        return "\n".join(out)


class _Indexed:
    """Whitespace-collapsed document text with an offset -> line-number map.

    Collapsing is what lets a phrase match across a line wrap; the map is what
    lets the match still be reported at a line number a human can open.
    """

    def __init__(self, raw_lines: list[str]) -> None:
        self.raw_lines = raw_lines
        chunks: list[str] = []
        starts: list[int] = []
        linenos: list[int] = []
        cursor = 0
        for lineno, raw in enumerate(raw_lines, start=1):
            piece = normalise(raw)
            if not piece:
                continue
            if chunks:
                chunks.append(" ")
                cursor += 1
            starts.append(cursor)
            linenos.append(lineno)
            chunks.append(piece)
            cursor += len(piece)
        self.text = "".join(chunks)
        self._starts = starts
        self._linenos = linenos

    def line_of(self, offset: int) -> int:
        if not self._starts:
            return 0
        idx = bisect.bisect_right(self._starts, offset) - 1
        return self._linenos[max(idx, 0)]

    def find_all(self, needle: str) -> list[int]:
        hits: list[int] = []
        if not needle:
            return hits
        start = 0
        while True:
            found = self.text.find(needle, start)
            if found < 0:
                return hits
            hits.append(found)
            start = found + 1


def probe(
    path: str | Path,
    phrase: str,
    *,
    identity: str | None = None,
    identity_within: int = 40,
    reject_lines_matching: str | None = None,
) -> Probe:
    """Search `path` for `phrase`, refusing to say ABSENT on a broken instrument.

    `identity` is a phrase you assert identifies this document. It is searched
    ONLY in the first `identity_within` lines -- the title/header region. An
    anchor matched anywhere in the body does not discriminate: 'SP_PLANE_PLAN'
    appears on 21 lines of COORDINATION.md, so a whole-file anchor would have
    passed on the wrong document and licensed a false ABSENT. If the anchor is
    not found in the header, the verdict is INSTRUMENT_BROKEN, not ABSENT.

    `reject_lines_matching` is a regex applied to the RAW matching line. Lines
    it matches are reported separately and do not count toward PRESENT -- use
    it to stop a comment about a removal from reading as the thing itself.
    """
    p = Path(path)
    controls: list[tuple[str, bool, str]] = []
    target = normalise(phrase)

    if not target:
        return Probe(INSTRUMENT_BROKEN, phrase, str(p),
                     reason="phrase is empty after normalisation")
    try:
        raw = p.read_text(encoding="utf-8")
    except OSError as exc:
        return Probe(INSTRUMENT_BROKEN, phrase, str(p),
                     controls=(("readable", False, str(exc)),),
                     reason=f"cannot read {p}")
    controls.append(("readable", True, f"{len(raw)} bytes"))

    doc = _Indexed(raw.splitlines())
    if not doc.text:
        return Probe(INSTRUMENT_BROKEN, phrase, str(p),
                     controls=tuple(controls) + (("non_empty", False, "0 chars"),),
                     reason="file has no text after normalisation")
    controls.append(("non_empty", True, f"{len(doc.text)} chars normalised"))

    # Control 1: a slice of the document's own text, searched the same way.
    # If this cannot be found, the matcher itself is broken.
    mid = len(doc.text) // 2
    slice_len = min(40, len(doc.text))
    self_slice = doc.text[mid:mid + slice_len] or doc.text[:slice_len]
    slice_ok = bool(doc.find_all(self_slice))
    controls.append(("self_slice", slice_ok, repr(self_slice[:32])))

    # Control 2: is this even the right document?
    identity_ok = True
    if identity is not None:
        head = _Indexed(raw.splitlines()[:identity_within])
        identity_ok = bool(head.find_all(normalise(identity)))
        controls.append(("identity", identity_ok,
                         f"{identity[:40]!r} in first {identity_within} lines"))

    offsets = doc.find_all(target)
    accepted: list[tuple[int, str]] = []
    rejected: list[tuple[int, str]] = []
    pattern = re.compile(reject_lines_matching) if reject_lines_matching else None
    seen: set[int] = set()
    for off in offsets:
        lineno = doc.line_of(off)
        if lineno in seen:
            continue
        seen.add(lineno)
        raw_line = doc.raw_lines[lineno - 1] if 0 < lineno <= len(doc.raw_lines) else ""
        if pattern is not None and pattern.search(raw_line):
            rejected.append((lineno, raw_line))
        else:
            accepted.append((lineno, raw_line))

    if accepted:
        # A hit is a hit: controls cannot make a found thing unfound. They only
        # gate the ABSENT verdict, which is the one that fails open.
        return Probe(PRESENT, phrase, str(p), tuple(accepted), tuple(rejected),
                     tuple(controls))

    if not slice_ok or not identity_ok:
        broken = "matcher failed its own self-slice" if not slice_ok else \
                 (f"identity anchor absent from the first {identity_within} "
                  "lines -- wrong file?")
        return Probe(INSTRUMENT_BROKEN, phrase, str(p), (), tuple(rejected),
                     tuple(controls), reason=broken)

    reason = ""
    if rejected:
        reason = (f"{len(rejected)} match(es) found but all on rejected lines "
                  f"-- the prose ABOUT the thing, not the thing")
    return Probe(ABSENT, phrase, str(p), (), tuple(rejected), tuple(controls),
                 reason=reason)


def require(path: str | Path, phrase: str, **kw) -> Probe:
    """probe(), but raise unless the verdict is PRESENT.

    For landing evidence under R-36: a check that must be demonstrated passing.
    INSTRUMENT_BROKEN raises with a distinct message so a broken check is never
    mistaken for a failed one.
    """
    result = probe(path, phrase, **kw)
    if result.outcome == INSTRUMENT_BROKEN:
        raise RuntimeError("LANDING CHECK INSTRUMENT BROKEN, verdict unusable:\n"
                           + result.render())
    if result.outcome != PRESENT:
        raise AssertionError("LANDING CHECK FAILED:\n" + result.render())
    return result


# ---------------------------------------------------------------------------
# selftests
# ---------------------------------------------------------------------------

def _selftests() -> int:
    import tempfile

    checks = 0

    def ok(cond, label):
        nonlocal checks
        checks += 1
        if not cond:
            raise AssertionError(f"selftest failed: {label}")

    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)

        # A document reproducing every real failure this module exists for.
        doc = d / "doc.md"
        doc.write_text(
            "# SP_PLANE_PLAN Revision 12\n"
            "\n"
            "CLAUSE (E): a vacated verdict carries its provenance permanently\n"
            "and reads VACATED -- was DEAD under bar X.\n"
            "\n"
            "The **refuse_k** guard is Class D under R-20.\n"
            "He said “the bar is frozen” and meant it.\n",
            encoding="utf-8",
        )

        # 1. phrase split across a line wrap -- the failure that broke DA twice
        r = probe(doc, "carries its provenance permanently and reads VACATED")
        ok(r.outcome == PRESENT, "line-wrapped phrase must be found")
        ok(r.accepted[0][0] == 3, "wrapped match reports its START line")

        # 2. markdown emphasis inside the phrase
        r = probe(doc, "The refuse_k guard is Class D")
        ok(r.outcome == PRESENT, "emphasis markers must not block a match")

        # 3. curly quotes and a non-breaking space
        r = probe(doc, 'He said "the bar is frozen" and meant it.')
        ok(r.outcome == PRESENT, "curly quotes and NBSP must fold")

        # 4. a genuinely absent phrase, with controls passing, IS reportable
        r = probe(doc, "CLAUSE (F): verdicts may be revised at will")
        ok(r.outcome == ABSENT, "true absence with good controls reports ABSENT")
        ok(all(passed for _, passed, _ in r.controls), "controls all pass on true absence")

        # 5. wrong file -> INSTRUMENT_BROKEN, never ABSENT
        r = probe(doc, "CLAUSE (F)", identity="MEASUREMENT_PLAN Revision 4")
        ok(r.outcome == INSTRUMENT_BROKEN, "bad identity anchor must not report ABSENT")
        ok("wrong file" in r.reason, "broken-instrument reason names the cause")

        # 6. right file -> identity control passes and ABSENT survives
        r = probe(doc, "CLAUSE (F)", identity="SP_PLANE_PLAN Revision 12")
        ok(r.outcome == ABSENT, "good identity anchor permits an ABSENT verdict")

        # 7. missing file -> INSTRUMENT_BROKEN, never ABSENT
        r = probe(d / "nope.md", "anything")
        ok(r.outcome == INSTRUMENT_BROKEN, "unreadable file must not report ABSENT")

        # 8. empty file -> INSTRUMENT_BROKEN, never ABSENT
        empty = d / "empty.md"
        empty.write_text("", encoding="utf-8")
        r = probe(empty, "anything")
        ok(r.outcome == INSTRUMENT_BROKEN, "empty file must not report ABSENT")

        # 9. the coordinator's failure: matching the comment that DOCUMENTS a
        #    removal, and reading it as the thing still being present
        code = d / "mod.py"
        code.write_text(
            "# the legacy_gate check was REMOVED here, see R-31\n"
            "def run():\n"
            "    return 1\n",
            encoding="utf-8",
        )
        naive = probe(code, "legacy_gate")
        ok(naive.outcome == PRESENT, "without a reject rule the comment matches")
        guarded = probe(code, "legacy_gate", reject_lines_matching=r"^\s*#")
        ok(guarded.outcome == ABSENT, "comment-only match must not read as PRESENT")
        ok(len(guarded.rejected) == 1, "the rejected match is still reported, not hidden")
        ok("prose ABOUT the thing" in guarded.reason, "reason explains the rejection")

        # 9b. an anchor that matches only in the BODY must not license ABSENT.
        #     This is the defect the first version of this module shipped with:
        #     'SP_PLANE_PLAN' matches 21 body lines of COORDINATION.md, so a
        #     whole-file anchor passed on the wrong document.
        other = d / "other.md"
        other.write_text(
            "# COORDINATION ledger\n" + "filler\n" * 60
            + "we discussed SP_PLANE_PLAN Revision 12 at length\n",
            encoding="utf-8",
        )
        r = probe(other, "CLAUSE (E)", identity="SP_PLANE_PLAN Revision 12")
        ok(r.outcome == INSTRUMENT_BROKEN, "body-only anchor must not license ABSENT")
        ok("wrong file" in r.reason, "reason names the wrong-file cause")
        r = probe(other, "CLAUSE (E)", identity="COORDINATION ledger")
        ok(r.outcome == ABSENT, "a header anchor does license ABSENT")
        r = probe(other, "CLAUSE (E)", identity="SP_PLANE_PLAN Revision 12",
                  identity_within=200)
        ok(r.outcome == ABSENT, "the window is caller-adjustable when body is meant")

        # 10. a hit stays a hit even if the identity anchor is wrong: controls
        #     gate ABSENT only, they never unfind a found thing
        r = probe(doc, "CLAUSE (E)", identity="not this document at all")
        ok(r.outcome == PRESENT, "controls must not suppress a genuine match")

        # 11. every probe reports lines, never a bare count
        r = probe(doc, "Class D")
        ok(r.accepted and isinstance(r.accepted[0][1], str), "matches carry their text")

        # 12. dashes are NOT folded -- leniency there would buy a false PRESENT
        r = probe(doc, "was DEAD under bar X")
        ok(r.outcome == PRESENT, "em-dash text matches when quoted faithfully")
        r = probe(doc, "reads VACATED - was DEAD")
        ok(r.outcome == ABSENT, "a hyphen must not silently match an em dash")

        # 13. require() distinguishes a FAILED check from a BROKEN one
        try:
            require(doc, "CLAUSE (F)", identity="SP_PLANE_PLAN Revision 12")
        except AssertionError as exc:
            ok("LANDING CHECK FAILED" in str(exc), "genuine failure raises AssertionError")
        else:
            ok(False, "require must raise on a failed check")
        try:
            require(d / "nope.md", "anything")
        except RuntimeError as exc:
            ok("INSTRUMENT BROKEN" in str(exc), "broken instrument raises RuntimeError")
        else:
            ok(False, "require must raise on a broken instrument")
        ok(require(doc, "CLAUSE (E)").outcome == PRESENT, "require returns on success")

    print(f"landing_check selftests: {checks} checks passed")
    return 0


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--selftest":
        raise SystemExit(_selftests())
    if len(sys.argv) < 3:
        print("usage: landing_check.py <file> <phrase> [identity]", file=sys.stderr)
        raise SystemExit(2)
    ident = sys.argv[3] if len(sys.argv) > 3 else None
    res = probe(sys.argv[1], sys.argv[2], identity=ident)
    print(res.render())
    raise SystemExit(0 if res.outcome == PRESENT else 1)
