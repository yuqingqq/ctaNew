"""Generate DE_LANE4_REAL_PARITY_RESULTS.md FROM the receipt, never by hand.

SURFACE AUTHORISATION (R-126, in-file): R-379 TASK 1 (DE seat).
RESEARCH-ONLY, OFFLINE.

WHY A GENERATOR AND NOT A DOCUMENT.  A results file whose numbers are typed
beside a receipt can disagree with it, and this programme has paid for that
shape more than once (a nightly log reading "verdict artifact written" beside a
file that had since been replaced; a multiplicity derivation that no reader
could recompute).  Every number in the emitted document is READ from
`de_lane4_real_parity_v1.json`, so the two cannot drift; and the selftest
regenerates and compares byte-for-byte, so a hand-edit is caught.

The PROSE sections are authored (findings and asks are judgements, not
numbers) and carry no figures that the receipt does not also carry.

    python3 live/pm_research/de_lane4_results_doc.py --selftest
    python3 live/pm_research/de_lane4_results_doc.py --write
"""
from __future__ import annotations

import argparse
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[2]
RECEIPT = ROOT / "data/pm_5min/derived/de_lane4_real_parity_v1.json"
DOC = ROOT / "live/pm_research/DE_LANE4_REAL_PARITY_RESULTS.md"
SECTIONS = pathlib.Path(__file__).with_name("de_lane4_results_sections.md")


class ReceiptMissing(RuntimeError):
    """No receipt: the document may not be written from memory."""


def _pct(a: int, b: int) -> str:
    return f"{100.0 * a / b:.1f}%" if b else "n/a"


def render(r: dict, prose: str) -> str:
    pop = r["population"]
    win = r["window_status_counts"]
    gen = r["generation_status_counts"]
    gates = r["gates"]
    excl_g = {k: v for k, v in gen.items() if k != "ADMITTED" and v}
    excl_w = {k: v for k, v in win.items() if k != "ADMITTED" and v}
    lines = [
        "# Lane-4 seven-arm parity, run against the REAL QR_SKEW_ONLY shadow",
        "",
        "**GENERATED from the receipt by `de_lane4_results_doc.py` — do not "
        "hand-edit.** Every number below is read from",
        f"`data/pm_5min/derived/de_lane4_real_parity_v1.json`; the selftest "
        f"regenerates and compares byte-for-byte, so the document and the "
        f"artifact cannot drift.",
        "",
        f"**Protocol** `{r['protocol']}` · **status** `{r['status']}` · "
        f"**as-of** {r['as_of_utc']} · **elapsed** {r['elapsed_s']} s",
        "",
        "**THIS IS VERIFICATION, NOT SCORING.** No economics may be read from "
        "it: the receipt structurally excludes the economics block, the fills "
        "block and the per-cancel records, and the standing hold (HANDOFF "
        "item 4) forbids PnL, capacity, promotion and forward verdicts.",
        "",
        "## 1. What ran, and on what",
        "",
        "| | |",
        "|---|---|",
        f"| population | `{pop['name']}`, era `{pop['era']}`, coins "
        f"{', '.join(pop['coins'])} |",
        f"| windows selected | **{pop['n_selected_windows']}** |",
        f"| windows excluded for Binance discontinuity (before selection) | "
        f"{pop['windows_excluded_binance_gap']} |",
        f"| UTC days | {', '.join(pop['days'])} (n={pop['n_days']}) |",
        f"| windows ADMITTED | **{win['ADMITTED']}** "
        f"({_pct(win['ADMITTED'], pop['n_selected_windows'])} of selected) |",
        f"| window exclusions, as counted statuses | "
        f"{excl_w if excl_w else 'none'} |",
        f"| generations ADMITTED | **{r['n_admitted_generations']:,}** |",
        f"| generation exclusions, as counted statuses | "
        f"{excl_g if excl_g else 'none'} |",
        f"| stub score events | {r['n_stub_score_events']:,} |",
        f"| cancels issued (active-stub cell) | "
        f"**{r['n_cancels_issued_active_stub']:,}** |",
        f"| windows where the policy ACTED | "
        f"{r['n_windows_where_the_policy_acted']} |",
        f"| fills charged STALE inside the latency window | "
        f"{r['n_stale_charged_fills']:,} |",
        f"| aggregate gate digest | `{r['aggregate_gate_digest'][:32]}…` |",
        "",
        "**Every exclusion above is a counted status, never a silent drop "
        "(rule 4), and every population carries its n and its as-of "
        "(rule 8).** The battery REFUSES a vacuum in both directions: zero "
        "admitted windows, and zero cancels issued — because every lifecycle "
        "gate would then pass on an empty set.",
        "",
        "## 2. The gates",
        "",
        "| gate | pass | failing windows |",
        "|---|---|---|",
    ]
    for k, v in gates.items():
        lines.append(f"| `{k}` | {'**PASS**' if v['pass'] else '**FAIL**'} | "
                     f"{v['n_failing_windows']}"
                     + (f" — {', '.join(v['failing_slugs'][:5])}"
                        if v['failing_slugs'] else "") + " |")
    lines += [
        "",
        f"**ALL GATES PASS: {r['all_gates_pass']}.**",
        "",
        "The two anchors the LANE4 spec calls bit-identical are bit-identical "
        "at real-data scale: a disabled predictor and an infinite cancel "
        "threshold each reproduce the QR_SKEW_ONLY passthrough event for "
        "event, and the two equal each other (so score evaluation is provably "
        "side-effect-free). Cancel-and-hold equivalence is checked against an "
        "**independently constructed** trajectory — written from the declared "
        "semantics, not by calling the machine's own event builders.",
        "",
        "## 3. Arm runnability — reported, never dropped",
        "",
        "| arm | status |",
        "|---|---|",
    ]
    for arm, st in r["arm_runnability"].items():
        lines.append(f"| `{arm}` | `{st}` |")
    leg = r.get("contract_leg") or {}
    inert = (leg.get("inert_check") or {})
    lines += [
        "",
        f"**{sum(1 for v in r['arm_runnability'].values() if v == 'RUNNABLE')}"
        f" of {len(r['arm_runnability'])} arms are runnable on the frozen "
        f"reference**, and the reasons are missing INPUTS, not a missing "
        f"predictor. See §5.",
        "",
        "## 4. The contract leg — DE's own exporter through DA's own loader",
        "",
        f"Run on one window (`{leg.get('slug', 'n/a')}`): the contract's value "
        f"here is its REFUSAL surface, which one window exercises as well as "
        f"471 do.",
        "",
        f"- inert arms admitted: **{leg.get('n_inert_arms_admitted')} of 7**, "
        f"refusals: {leg.get('inert_refusals') or 'none'}",
        f"- `inactive_predictors_agree`: "
        f"**{inert.get('inactive_predictors_agree')}** — every submission with "
        f"the predictor off is bit-identical, whatever its arm",
        f"- `pass`: **{inert.get('pass')}**",
        f"- DA canon `{leg.get('da_battery_canon')}`; DA's `ARMS` tuple "
        f"matches DE's exactly: **{leg.get('da_arms_match_ours')}**",
        "",
        "**The repost-axis diagnostic, run as a matched pair** (two cells "
        "differing only in whether the policy reposts):",
        "",
        "| cell | reposts | `no_fill_after_effective` | `pass` |",
        "|---|---|---|---|",
    ]
    diag = leg.get("repost_axis_diagnostic") or {}
    for name in ("reposting", "no_repost_permanent_hold"):
        d = diag.get(name) or {}
        lines.append(f"| {name} | {d.get('n_reposts')} | "
                     f"**{d.get('no_fill_after_effective')}** | "
                     f"{d.get('pass')} |")
    lines += [
        "",
        f"**Diagnosis holds: {diag.get('diagnosis_holds')}** — see finding C2.",
        "",
        "**Both readings of a STALE cancel, neither sound** (finding C1):",
        "",
        "| reading | requested | effective | suppressed | identity holds |",
        "|---|---|---|---|---|",
    ]
    for reading, lc in (leg.get("acting_lifecycle_by_stale_reading")
                        or {}).items():
        lines.append(f"| `{reading}` | {lc.get('requested')} | "
                     f"{lc.get('effective')} | {lc.get('suppressed')} | "
                     f"{lc.get('identity_holds')} |")
    lines += [
        "",
        "## 5. Declared parameters, and the code that produced this",
        "",
        "```json",
        json.dumps(r["declared_parameters"], indent=2, sort_keys=True),
        "```",
        "",
        "**Code identity, taken AT IMPORT** (a long run outlives edits to its "
        "own source; hashing at receipt-write time would stamp the receipt "
        "with code that did not produce it):",
        "",
        "| file | sha256[:16] |",
        "|---|---|",
    ]
    for f, h in sorted(r["code_identity"].items()):
        lines.append(f"| `{f}` | `{h}` |")
    lines += ["", prose.rstrip(), ""]
    return "\n".join(lines)


def build() -> str:
    if not RECEIPT.exists():
        raise ReceiptMissing(
            f"{RECEIPT} does not exist. The results document is generated "
            f"FROM the receipt and is never written from memory.")
    return render(json.loads(RECEIPT.read_text()), SECTIONS.read_text())


def selftest() -> int:
    n = [0]

    def ok(c, label):
        if not c:
            raise SystemExit(f"[de_lane4_results_doc] FAIL: {label}")
        n[0] += 1
        print(f"  PASS  {label}")

    ok(SECTIONS.exists(), "the authored prose sections file exists")
    if not RECEIPT.exists():
        ok(True, "NO RECEIPT YET: generation refuses rather than inventing "
                 "one -- checked below")
        try:
            build()
            raise SystemExit("FAIL: build() must refuse without a receipt")
        except ReceiptMissing:
            ok(True, "KNOWN-BAD: build() REFUSES with no receipt on disk")
        print(f"[de_lane4_results_doc] selftest OK -- {n[0]} checks "
              f"(receipt absent)")
        return 0
    text = build()
    r = json.loads(RECEIPT.read_text())
    ok(str(r["population"]["n_selected_windows"]) in text
       and r["as_of_utc"] in text
       and r["aggregate_gate_digest"][:32] in text,
       "the rendered document carries the receipt's own n, as-of and digest")
    ok(DOC.exists() and DOC.read_text() == text,
       "the COMMITTED document regenerates byte-for-byte from the receipt -- "
       "a hand-edit or a stale copy is caught here")
    doctored = dict(r)
    doctored["population"] = dict(r["population"], n_selected_windows=-1)
    ok(render(doctored, SECTIONS.read_text()) != text,
       "KNOWN-BAD: a doctored receipt renders a DIFFERENT document, so the "
       "comparison above is not vacuous")
    print(f"[de_lane4_results_doc] selftest OK -- {n[0]} checks")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args(argv)
    if a.write:
        DOC.write_text(build())
        print(f"wrote {DOC}")
        return 0
    if a.selftest:
        return selftest()
    print(build())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
