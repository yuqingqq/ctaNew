"""A FALSIFIER PROVES A MODULE CAN FIRE. IT SAYS NOTHING ABOUT WHETHER
ANYTHING ASKS IT TO.

REVIEW 265's rule, adopted here as an instrument. Two independent
properties, and the lane has been measuring only the first -- which is
exactly how the anti-amendment guard came out 37/37 green with no call
site at all.

So every refusal-raising module declares a `CALL_SITE`, and this audits
the declaration rather than trusting it:

  IMPORTED_BY            another lane module names it -- VERIFIED here
  UNIT / LAUNCHER        a .sh or systemd unit names it -- VERIFIED here
  ARTIFACT_MEDIATED      it gates the artifact it emits, and the consumer
                         reads that artifact (REVIEW 265 named this as
                         correct wiring, and counting it as an orphan as
                         its own false positive)
  DELIBERATE_INVOCATION  it is meant to be run by a person and read; it
                         GATES NOTHING, and saying so is the point
  REQUIRED_BUT_ABSENT    it exists to gate and nothing calls it yet --
                         reported as OPEN, never as a pass

The distinction that matters: DELIBERATE_INVOCATION is a claim that no
protection is being asserted. If a declaration elsewhere says such a
module enforces something, the DECLARATION is the defect -- a receipt
asserting a guard that never runs is paper protection.

Usage:  de_call_site_audit.py --falsify
        de_call_site_audit.py --audit
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROTOCOL = "P003_DE_CALL_SITE_AUDIT_V1"

NOT_DECLARED = "CALL_SITE_NOT_DECLARED"
DOES_NOT_NAME = "CALL_SITE_DOES_NOT_NAME_THE_MODULE"
UNKNOWN_KIND = "CALL_SITE_KIND_NOT_RECOGNISED"

KINDS = ("IMPORTED_BY", "UNIT", "LAUNCHER", "ARTIFACT_MEDIATED",
         "DELIBERATE_INVOCATION", "REQUIRED_BUT_ABSENT")
VERIFIABLE = ("IMPORTED_BY", "UNIT", "LAUNCHER")


class CallSiteRefused(ValueError):
    """The call-site declaration cannot be verified as written."""


def raises_a_named_refusal(src: str) -> bool:
    return bool(re.search(r'REFUSED \{?[A-Z_]{6,}', src))


def declared_call_site(src: str):
    m = re.search(r"^CALL_SITE\s*=\s*(\{.*?\n\})", src,
                  re.S | re.M)
    if not m:
        return None
    # NO QUOTE REWRITING: an apostrophe inside a declared sentence
    # ("the launcher's watcher") would be turned into a quote and the
    # block would stop parsing -- a corrupted declaration reading as an
    # absent one.
    body = m.group(1)
    # A TRAILING COMMA IS PYTHON AND NOT JSON, and a declaration this
    # parser silently dropped would read as "no call site declared" --
    # the exact false negative this audit exists to avoid.
    body = re.sub(r",(\s*[}\]])", r"\1", body)
    try:
        return json.loads(body)
    except Exception:                                       # noqa: BLE001
        raise CallSiteRefused(
            f"REFUSED {NOT_DECLARED}: a CALL_SITE block is present but "
            f"does not parse, which would otherwise read as no "
            f"declaration at all.") from None


def verify(module: str, site: dict, root: Path = HERE) -> dict:
    kind = site.get("kind")
    if kind not in KINDS:
        raise CallSiteRefused(
            f"REFUSED {UNKNOWN_KIND}: {module} declares kind {kind!r}, "
            f"not one of {list(KINDS)}.")
    row = {"module": module, "kind": kind, "by": site.get("by"),
           "gates": site.get("gates"), "verified": None}
    if kind in VERIFIABLE:
        by = site.get("by") or ""
        target = root / by if not by.startswith("/") else Path(by)
        if not target.is_file():
            raise CallSiteRefused(
                f"REFUSED {DOES_NOT_NAME}: {module} names {by!r} as its "
                f"call site and that file does not exist.")
        if module not in target.read_text():
            raise CallSiteRefused(
                f"REFUSED {DOES_NOT_NAME}: {by} does not mention "
                f"{module}. A declared call site that does not name the "
                f"module is the paper protection this audit exists to "
                f"find.")
        row["verified"] = True
    elif kind == "ARTIFACT_MEDIATED":
        row["verified"] = bool(site.get("artifact"))
        row["artifact"] = site.get("artifact")
    elif kind == "DELIBERATE_INVOCATION":
        row["verified"] = True
        row["ASSERTS_NO_PROTECTION"] = True
    else:                                    # REQUIRED_BUT_ABSENT
        row["verified"] = False
        row["OPEN"] = site.get("why") or "no call site yet"
    return row


def audit(root: Path = HERE, only=None) -> dict:
    rows, missing = [], []
    for f in sorted(root.glob("*.py")):
        if only and f.stem not in only:
            continue
        src = f.read_text()
        if not raises_a_named_refusal(src):
            continue
        try:
            site = declared_call_site(src)
        except CallSiteRefused as exc:
            rows.append({"module": f.stem, "REFUSED": str(exc)[:160]})
            continue
        if site is None:
            missing.append(f.stem)
            continue
        try:
            rows.append(verify(f.stem, site, root))
        except CallSiteRefused as exc:
            rows.append({"module": f.stem, "REFUSED": str(exc)[:160]})
    return {"protocol": PROTOCOL,
            "n_refusal_raising_with_a_declaration": len(rows),
            "n_refusal_raising_without_one": len(missing),
            "undeclared": missing,
            "open_sites": [r for r in rows if r.get("OPEN")],
            "assert_no_protection": [r["module"] for r in rows
                                     if r.get("ASSERTS_NO_PROTECTION")],
            "rows": rows,
            "the_rule": "a falsifier proves a module CAN fire; a call "
                        "site is whether anything asks it to"}


def falsify() -> int:
    import tempfile
    n = ok = 0

    def ck(name, cond, note=""):
        nonlocal n, ok
        n += 1
        ok += bool(cond)
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}"
              + (f"  {note}" if note else ""))

    tmp = Path(tempfile.mkdtemp(prefix="de_callsite_"))
    (tmp / "caller.py").write_text("import guarded_mod\n")
    (tmp / "guarded_mod.py").write_text(
        'CALL_SITE = {\n "kind": "IMPORTED_BY",\n "by": "caller.py"\n}\n'
        'raise X(f"REFUSED SOMETHING_NAMED: x")\n')
    (tmp / "liar_mod.py").write_text(
        'CALL_SITE = {\n "kind": "IMPORTED_BY",\n "by": "caller.py"\n}\n'
        'raise X(f"REFUSED SOMETHING_NAMED: x")\n')
    (tmp / "orphan_mod.py").write_text(
        'raise X(f"REFUSED SOMETHING_NAMED: x")\n')
    (tmp / "weird_mod.py").write_text(
        'CALL_SITE = {\n "kind": "VIBES"\n}\n'
        'raise X(f"REFUSED SOMETHING_NAMED: x")\n')

    print("== the audit verifies the declaration, never trusts it ==")
    got = audit(tmp)
    by_mod = {r["module"]: r for r in got["rows"]}
    ck("a module whose declared caller DOES name it verifies",
       by_mod["guarded_mod"].get("verified") is True)
    ck("  and one whose declared caller does NOT name it is REFUSED -- "
       "positive control",
       DOES_NOT_NAME in by_mod["liar_mod"].get("REFUSED", ""),
       "the paper-protection shape")
    ck("a refusal-raising module with NO declaration is listed undeclared",
       "orphan_mod" in got["undeclared"])
    ck("an unrecognised kind refuses rather than passing",
       UNKNOWN_KIND in by_mod["weird_mod"].get("REFUSED", ""))

    print("== the lane's own refusal-raising modules ==")
    mine = audit(HERE, only={"de_fair_value_plumbing_run",
                             "de_band_decision", "de_band_hazard",
                             "de_unit_verdict", "de_gate_property_map"})
    rows = {r["module"]: r for r in mine["rows"]}
    ck("the plumbing run declares DELIBERATE_INVOCATION and asserts NO "
       "protection",
       rows["de_fair_value_plumbing_run"].get("ASSERTS_NO_PROTECTION")
       is True,
       "a diagnostic that must be invoked; it gates nothing")
    ck("de_band_hazard's declared importer really does name it",
       rows["de_band_hazard"].get("verified") is True,
       rows["de_band_hazard"].get("by"))
    ck("de_band_decision is ARTIFACT_MEDIATED and names its artifact",
       rows["de_band_decision"]["kind"] == "ARTIFACT_MEDIATED"
       and bool(rows["de_band_decision"].get("artifact")))
    ck("de_unit_verdict is reported OPEN -- it exists to gate and "
       "NOTHING calls it",
       rows["de_unit_verdict"].get("OPEN"),
       rows["de_unit_verdict"].get("OPEN"))
    ck("de_gate_property_map is reported OPEN for the same reason",
       rows["de_gate_property_map"].get("OPEN"))
    ck("  and OPEN sites are never counted as passes",
       all(r.get("verified") is False for r in mine["open_sites"]),
       f"{len(mine['open_sites'])} open")

    print(f"\n{ok}/{n} cells pass")
    return 0 if ok == n else 1


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--falsify" in argv:
        return falsify()
    if "--audit" in argv:
        print(json.dumps(audit(), indent=2, default=str))
        return 0
    print(json.dumps({"protocol": PROTOCOL, "kinds": list(KINDS)},
                     indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
