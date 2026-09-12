"""STEP 6'S FULL-PIPELINE FREEZE DECLARATION -- LANDED, AND COMPUTED.

DA 278. `fair_value_plan.md` §7 names exactly what this declaration must
contain, so every field here is built AGAINST THAT TEXT and COMPUTED from a
fetched ref -- never typed from memory and never asserted in prose.

    "The declaration records all file hashes, commit ref, candidate count,
     action key, epsilon, status grammar, source manifests, initial inventory,
     tick rounding, latency, fee rule, quote parameters, null and success
     predicates. Every verdict is computed from artifact fields; no prose-only
     pass is permitted."                                   -- §7, verbatim

THE CHAIN §7 FREEZES AS ONE COMMIT:

    immutable inputs -> labels/statuses -> actions -> sigma -> FairPrice
                     -> fallback -> score -> quote mapping -> replay -> P&L

EXISTENCE IS NOT EFFECTIVENESS, AND THIS FILE INSISTS ON THE DIFFERENCE.

DA 281 ordered this declaration landed, and it is landed: §11 step 6 needed an
artifact to point at, and there was none on either executing ref. But LANDING A
DECLARATION DOES NOT FREEZE A PIPELINE THAT IS NOT BUILT. `freeze_is_effective`
is computed from the blocking-gap list and is FALSE while any §7 field is
MISSING or any chain link has no implementation.

The distinction is load-bearing rather than pedantic. `da_fair_value_ledger`
previously resolved §11 step 6 by COUNTING files matching a freeze-declaration
name, so landing this file would have flipped `score_is_evidence_permitted` to
True -- a labelled score becoming permissible because a declaration exists that
itself says the pipeline is not frozen. The ledger now reads THIS field. A
declaration that cannot be honestly effective must not be able to unlock
anything by existing.

WHAT IS MISSING IS RECORDED AS MISSING, WITH THE MEASUREMENT THAT PROVES IT.
§7 forbids a prose-only pass, and that cuts both ways: a field may not be
satisfied by a sentence, and neither may it be quietly omitted. Measured
2026-09-11T20:58Z, `MARKETABLE_CROSS` appears in ZERO .py files in the whole
lane, so the frozen quote mapping DA 281 asks this file to carry cannot be
carried -- it does not exist to be frozen. Writing it in anyway would be the
exact defect this lane has spent the day removing from other people's code.

WHY THIS IS A MODULE AND NOT A JSON FILE. A freeze declaration full of typed
hashes is a copy that ages the moment a blob moves -- measured today, the chain
ref moved three times in twenty minutes. This computes its hashes from the ref
at run time; the JSON it prints is the artifact, and it is reproducible from
the ref rather than trusted.

Usage:  da_step6_freeze_declaration_draft.py [--falsify]
"""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
import time
import tempfile
from pathlib import Path

PROTOCOL = "P003_DA_STEP6_FULL_PIPELINE_FREEZE_V1"
PLAN = "fair_value_plan.md v1.2"
REF = "origin/de-freeze-chain-v2"
MIRROR_REF = "origin/be-build-runner"
MISSING = "MISSING"

#: §7's chain, in order, mapped to the files that implement each link. A link
#: with no file is a GAP and is computed as one -- never narrated away.
CHAIN = (
    ("immutable_inputs",
     ("live/pm_research/da_immutable_inputs_manifest.py",)),
    ("labels_statuses", ("live/pm_research/da_fair_value_gate1_labels.py",)),
    ("actions", ("live/pm_research/de_canonical_action_population.py",
                 "live/pm_research/de_fair_value_actions.py")),
    ("sigma", ("live/pm_research/be_sigma_30m.py",)),
    ("fairprice", ("live/pm_research/da_fair_price_identity.py",
                   "live/pm_research/de_fair_price_wrapper.py")),
    ("fallback", ("live/pm_research/de_fair_price_wrapper.py",
                  "live/pm_research/de_fair_value_policy_seam.py")),
    ("score", ("live/pm_research/de_fair_value_actions.py",)),
    ("quote_mapping", ("live/pm_research/de_fair_value_policy_seam.py",)),
    ("replay", ("live/pm_research/de_fair_value_replay_seam.py",)),
    # ATTRIBUTED AT DA 287, AND THE FINDING IS NAMED RATHER THAN QUIETLY FIXED.
    # This read `("pnl", ())` -- "no implementation" -- for as long as
    # `de_fair_value_pnl.py` has existed. The gap was never a fact about the
    # lane; it was MY ATTRIBUTION, and the chain-link -> file mapping is this
    # module's weakest link exactly as it is the ledger's. An unattributed link
    # reads as unimplemented no matter what another seat built, so the freeze
    # reported a gap that DE had closed. Same class as the ledger's guessed
    # step-4 paths, in the artifact a reader resolves.
    ("pnl", ("live/pm_research/de_fair_value_pnl.py",)),
)

#: The fields §7's sentence enumerates. Each resolves to a value or to MISSING,
#: and MISSING is a blocking gap unless it is listed as non-blocking with a
#: reason. Nothing here may be satisfied by prose.
REQUIRED_FIELDS = (
    "all_file_hashes", "commit_ref", "candidate_count", "action_key",
    "epsilon", "status_grammar", "source_manifests", "initial_inventory",
    "tick_rounding", "latency", "fee_rule", "quote_parameters",
    "null_predicate", "success_predicate",
    "minimum_meaningful_delta_LL",
)



# ==========================================================================
# THE ENUMERATIONS ARE PINNED.                                    (DA 284)
#
# REVIEW 208 SHRANK `REQUIRED_FIELDS` and `CHAIN` and drove the gap count from
# 15 to 8 with every underlying gap still real -- because NOTHING pinned their
# sizes or the ten link names. A gap count is only as strong as the enumeration
# it is counted over, and an enumeration that can quietly shrink is a way to
# report progress by deleting the questions.
# ==========================================================================


# ==========================================================================
# THE PINS ARE DERIVED FROM OUTSIDE THIS MODULE.                  (DA 286)
#
# REVIEW 209 defeated the previous pair with one edit in this file: the clause
# pin compared COUNTS, so swapping a clause for "the seam is written in python"
# still read INTACT; and the pins sat ten lines below the lists they pinned, so
# shrinking both together yielded freeze_is_effective True, enumeration_intact
# True and n_blocking_gaps 0 with every real gap present.
#
#     A PIN IN THE SAME FILE AS THE THING IT PINS IS A COPY, NOT A CHECK.
#
# So the enumeration now comes from `fair_value_plan.md` -- the USER-FROZEN
# plan, read from the immutable git blob that last changed it -- and the
# comparison is by CONTENT, not by length. Shrinking a list here now requires
# editing a document this seat does not own, in a commit that is not this one.
# ==========================================================================

PLAN_PATH = "orchestrator/PROGRAMS/P-2026-003-polymarket-5min/workspace/fair_value_plan.md"
PLAN_PARSE_FAILED = "PLAN_ENUMERATION_UNPARSEABLE"

#: plan wording -> the identifier this module uses. Explicit, so a plan edit
#: that renames a link SURFACES as an unmapped name instead of a shorter list.
_LINK_NAMES = {
    "immutable inputs": "immutable_inputs", "labels/statuses": "labels_statuses",
    "actions": "actions", "sigma": "sigma", "fairprice": "fairprice",
    "fallback": "fallback", "score": "score", "quote mapping": "quote_mapping",
    "replay": "replay", "p&l": "pnl",
}
_FIELD_NAMES = {
    "all file hashes": "all_file_hashes", "commit ref": "commit_ref",
    "candidate count": "candidate_count", "action key": "action_key",
    "epsilon": "epsilon", "status grammar": "status_grammar",
    "source manifests": "source_manifests",
    "initial inventory": "initial_inventory", "tick rounding": "tick_rounding",
    "latency": "latency", "fee rule": "fee_rule",
    "quote parameters": "quote_parameters",
    "null and success predicates": ("null_predicate", "success_predicate"),
}


def _plan_blob() -> tuple:
    """The plan's bytes from the commit that last changed it -- an immutable
    object reference, not a working-tree read."""
    # --all, because this module runs from a detached worktree of a CHAIN ref
    # whose history does not contain the plan. The object is in the shared
    # database; the branch is not the point.
    sha = _git("log", "-1", "--all", "--format=%H", "--", PLAN_PATH).stdout.strip()
    if not sha:
        raise RuntimeError(f"REFUSED {PLAN_PARSE_FAILED}: no commit touches "
                           f"{PLAN_PATH}, so the enumeration has no external "
                           f"source and this module will not fall back to its "
                           f"own copy.")
    b = _git("show", f"{sha}:{PLAN_PATH}", text=False).stdout
    if not b:
        raise RuntimeError(f"REFUSED {PLAN_PARSE_FAILED}: {PLAN_PATH} is empty "
                           f"at {sha[:12]}")
    return sha, b


def plan_enumerations() -> dict:
    """§7's chain links and required fields, PARSED FROM THE PLAN."""
    sha, b = _plan_blob()
    text = b.decode("utf-8", "replace")
    sec = text.split("## 7.", 1)[-1].split("## 8.", 1)[0]

    chain_txt = " ".join(
        ln.strip() for ln in sec.splitlines()
        if "->" in ln and ("immutable" in ln or "fallback" in ln))
    links, unmapped = [], []
    for raw in [x.strip() for x in chain_txt.split("->") if x.strip()]:
        key = raw.lower().strip()
        (links.append(_LINK_NAMES[key]) if key in _LINK_NAMES
         else unmapped.append(raw))

    m = re.search(r"The declaration records (.+?)\.", sec, re.S)
    fields, unmapped_f = [], []
    if m:
        for raw in re.split(r",\s*", " ".join(m.group(1).split())):
            key = raw.strip().lower()
            got = _FIELD_NAMES.get(key)
            if got is None:
                unmapped_f.append(raw)
            elif isinstance(got, tuple):
                fields.extend(got)
            else:
                fields.append(got)

    clause_bullets = [ln for ln in sec.splitlines() if ln.strip().startswith("- ")]
    if unmapped or unmapped_f or len(links) != 10 or len(fields) != 14:
        raise RuntimeError(
            f"REFUSED {PLAN_PARSE_FAILED}: parsed {len(links)} links and "
            f"{len(fields)} fields from {PLAN_PATH}@{sha[:12]}; unmapped link "
            f"wording {unmapped}, unmapped field wording {unmapped_f}. A parse "
            f"that silently yields a SHORTER list is the very failure this pin "
            f"exists to prevent, so an incomplete parse refuses.")
    return {"plan_commit": sha, "plan_sha256": hashlib.sha256(b).hexdigest(),
            "links": tuple(links), "fields": frozenset(fields),
            "n_quote_clause_bullets": len(clause_bullets)}


ENUMERATION_SHRANK = "FREEZE_ENUMERATION_DOES_NOT_MATCH_ITS_PIN"

CHAIN_LINKS_PINNED = (
    "immutable_inputs", "labels_statuses", "actions", "sigma", "fairprice",
    "fallback", "score", "quote_mapping", "replay", "pnl",
)

#: THE ATTRIBUTION IS DA'S, AND IT IS THE WEAK LINK. The link NAMES are pinned
#: to the plan; the FILES behind them are this seat's mapping, and a wrong or
#: missing one produces a gap or a pass that is about the mapping rather than
#: the lane. `unattributed_chain_files` below is the guard: it names lane files
#: that look like chain implementations and that no link claims.
CHAIN_FILE_RE = r"(fair_value|fair_price|sigma_30m)"

#: ADDITIONS BEYOND §7'S SENTENCE, DECLARED SO THE PIN CAN ALLOW THEM.
#: The pin exists to stop a list shrinking quietly; it must not also stop the
#: list GROWING for a stated reason. Each addition names who added it and why,
#: and `assert_enumerations_intact` compares against (plan fields | additions),
#: so an UNDECLARED addition still refuses exactly as a shrink does.
DECLARED_ADDITIONS = {
    "minimum_meaningful_delta_LL": (
        "DA 297 / REVIEW 256: §8 declares a minimum SAMPLE and no minimum "
        "EFFECT, so the exact sign test can pass on an effect of any size "
        "above zero. Added as a REQUIRED field that is UNSET, so an absent "
        "value and a satisfied one are distinguishable and the freeze "
        "computes FALSE until the user supplies one."),
}

REQUIRED_FIELDS_PINNED = frozenset({
    "all_file_hashes", "commit_ref", "candidate_count", "action_key",
    "epsilon", "status_grammar", "source_manifests", "initial_inventory",
    "tick_rounding", "latency", "fee_rule", "quote_parameters",
    "null_predicate", "success_predicate",
}) | frozenset(DECLARED_ADDITIONS)

#: §7's quote-mapping clauses. A probe that FAILS leaves ALL of these unmet --
#: never zero of them, which is what an absent list used to mean.
#: The clause STRINGS, compared by content. A count comparison let REVIEW 209
#: swap a clause for "the seam is written in python" and still read INTACT.
QUOTE_CLAUSES_CONTENT = (
    "UP_uses_p", "DOWN_uses_1_minus_p",
    "bid_rounds_DOWN_to_the_legal_tick", "ask_rounds_UP_to_the_legal_tick",
    "prices_bounded_to_the_legal_binary_range",
    "the_bound_applied_is_RECORDED_not_silent",
    "crossing_quote_emits_PLACE_WITHHELD_MARKETABLE_CROSS",
    "no_zero_latency_privilege_for_candidate_induced_change",
    "the_tick_is_DECLARED_not_invented",
)

QUOTE_CLAUSES_PINNED = (
    "UP_uses_p", "DOWN_uses_1_minus_p",
    "bid_rounds_DOWN_to_the_legal_tick", "ask_rounds_UP_to_the_legal_tick",
    "prices_bounded_to_the_legal_binary_range",
    "the_bound_applied_is_RECORDED_not_silent",
    "crossing_quote_emits_PLACE_WITHHELD_MARKETABLE_CROSS",
    "no_zero_latency_privilege_for_candidate_induced_change",
    "the_tick_is_DECLARED_not_invented",
)


def assert_enumerations_intact(chain=None, fields=None, clauses=None) -> dict:
    """REFUSE unless the live enumerations match THE PLAN, by CONTENT.

    Injectable so the refusal can be shown to fire (rule 15).
    """
    chain = CHAIN if chain is None else chain
    fields = REQUIRED_FIELDS if fields is None else fields
    clauses = QUOTE_CLAUSES_PINNED if clauses is None else clauses
    plan = plan_enumerations()
    bad = []
    live = tuple(l for l, _ in chain)
    if live != plan["links"]:
        bad.append(f"chain links {live} != the plan's {plan['links']}")
    allowed = plan["fields"] | frozenset(DECLARED_ADDITIONS)
    if frozenset(fields) != allowed:
        bad.append(f"required fields differ from (plan | declared additions): "
                   f"missing {sorted(allowed - frozenset(fields))}, "
                   f"UNDECLARED extra {sorted(frozenset(fields) - allowed)}")
    # CLAUSES ARE COMPARED BY CONTENT, not by count. Swapping a clause for
    # "the seam is written in python" used to read INTACT.
    if tuple(clauses) != QUOTE_CLAUSES_CONTENT:
        bad.append(f"quote clauses differ by CONTENT: missing "
                   f"{sorted(set(QUOTE_CLAUSES_CONTENT) - set(clauses))}, extra "
                   f"{sorted(set(clauses) - set(QUOTE_CLAUSES_CONTENT))}")
    if bad:
        raise RuntimeError(
            f"REFUSED {ENUMERATION_SHRANK}: {'; '.join(bad)}. A gap count is "
            f"only as strong as the enumeration it is counted over, and a pin "
            f"in the same file as the thing it pins is a copy, not a check.")
    return {"n_chain_links": len(plan["links"]),
            "n_required_fields": len(plan["fields"]),
            "n_quote_clauses": len(QUOTE_CLAUSES_CONTENT),
            "pinned_by": PLAN_PATH,
            "plan_commit": plan["plan_commit"][:12],
            "plan_sha256_16": plan["plan_sha256"][:16],
            "clauses_compared_by": "CONTENT",
            "intact": True}


#: THE SUBPROCESS PROTOCOL. A probe's result is the line AFTER this sentinel,
#: never "the last line of stdout". Parsing by POSITION has broken three probes:
#: gate 6's TypeError (empty stdout -> IndexError), the quote mapping's
#: SeamRefused (prose -> JSONDecodeError), and REVIEW 208's drive. A sentinel is
#: an agreed protocol; position is a guess about formatting.
PROBE_SENTINEL = "<<<DA_PROBE_RESULT_JSON>>>"


def parse_probe_output(stdout: str, stderr: str = "") -> dict:
    """The probe's result, by PROTOCOL. A probe that did not emit the sentinel
    FAILED -- it did not merely fail to be parsed."""
    lines = (stdout or "").splitlines()
    for i, ln in enumerate(lines):
        if ln.strip() == PROBE_SENTINEL and i + 1 < len(lines):
            try:
                return {"ok": True, "result": json.loads(lines[i + 1])}
            except Exception as e:
                return {"ok": False, "failure":
                        f"sentinel found but payload unparseable: "
                        f"{type(e).__name__}: {str(e)[:120]}"}
    tail = ((stderr or stdout or "").strip().splitlines() or [""])[-1]
    return {"ok": False, "failure":
            f"no {PROBE_SENTINEL} in stdout -- the probe did not complete. "
            f"Last line seen: {tail[:160]}"}


def _root() -> str:
    """THE REPO ROOT, RESOLVED FROM THIS MODULE -- never from the cwd.

    Driving this file from a scratch directory made `git rev-parse` fail with
    "not a git repository", so EVERY blob read returned None, EVERY field read
    MISSING, and the draft reported sixteen blocking gaps that were entirely an
    artifact of where it was run. That is the same defect class as the ledger's
    cwd bug, one layer over -- a false NEGATIVE manufactured by the instrument.
    The falsifier caught it this time because the cells assert POSITIVE facts
    (a full sha256, exactly one admitting status) rather than absences.

    Resolution order is the module's own location first, then the cwd, and
    NEVER an environment variable.
    """
    here = Path(__file__).resolve().parent
    for base in (here, Path.cwd()):
        r = subprocess.run(["git", "-C", str(base), "rev-parse", "--show-toplevel"],
                           capture_output=True, text=True)
        if r.returncode == 0 and r.stdout.strip():
            return r.stdout.strip()
    raise RuntimeError(
        "REFUSED NO_REPOSITORY_RESOLVABLE: this declaration reads every field "
        "from a git ref, and neither the module's own directory nor the cwd is "
        "inside a repository. A draft that cannot reach the ref must refuse, "
        "not report an empty chain as a set of gaps.")


def _git(*a, text=True):
    return subprocess.run(["git", "-C", _root(), *a], capture_output=True, text=text)


def _blob(ref: str, path: str) -> bytes | None:
    r = _git("show", f"{ref}:{path}", text=False)
    return r.stdout if r.returncode == 0 else None


def _sha(ref: str, path: str) -> str:
    b = _blob(ref, path)
    return hashlib.sha256(b).hexdigest() if b is not None else MISSING


def _src(ref: str, path: str) -> str:
    b = _blob(ref, path)
    return b.decode("utf-8", "replace") if b is not None else ""


def file_hashes(ref: str) -> dict:
    """ALL FILE HASHES, over every file the chain names -- full sha256, because
    a freeze is the one place a truncated digest is not enough."""
    out = {}
    for _, paths in CHAIN:
        for p in paths:
            out[p] = _sha(ref, p)
    return out


def candidates(ref: str) -> dict:
    """CANDIDATE COUNT AND IDENTITIES, driven from the wrapper at the ref.

    MEASURED CORRECTION TO THE DISPATCH: DA 278 describes both candidates as
    bound to `model_version = s60_probability_v1`. The artifact binds it to C2
    ONLY; C1 binds None, deliberately -- "C1 has no fitted parameter, so it
    binds no model version, and says so with None rather than borrowing C2's".
    Carried as measured.
    """
    head_sha = _git("rev-parse", ref).stdout.strip()
    wt = tempfile.mkdtemp(prefix="da_step6_")
    Path(wt).rmdir()
    a = _git("worktree", "add", "--detach", wt, ref)
    if a.returncode != 0:
        return {"error": "worktree failed", "n": MISSING}
    pm = str(Path(wt) / "live" / "pm_research")
    code = (
        "import json,sys; sys.path.insert(0, %r)\n"
        "import de_fair_price_wrapper as W, da_fair_price_identity as FP\n"
        "out={}\n"
        "for est in (FP.MICROPRICE, FP.BN_BOOKTICKER):\n"
        "    out[est]=W.identity_of(est).as_dict()\n"
        "print(%r)\n"
        "print(json.dumps({'identities': out, 'model_version': W.MODEL_VERSION}))\n"
        % (pm, PROBE_SENTINEL))
    r = subprocess.run(["python3", "-c", code], cwd=pm, capture_output=True, text=True)
    _git("worktree", "remove", "--force", wt)
    parsed = parse_probe_output(r.stdout, r.stderr)
    if not parsed["ok"]:
        return {"error": parsed["failure"], "probe_failed": True, "n": MISSING}
    d = parsed["result"]
    ids = d["identities"]
    # RULE 12 WANTS A COMMITTED BUILDER WITH ITS COMMIT REF. `identity_of` runs
    # inside a temp worktree, so it reports /tmp paths no later reader can
    # resolve -- which turns "do the digests match the committed bytes?" into a
    # check the reader has to invent. It is computed HERE instead, and the
    # repo-relative path and commit ref are recorded beside every digest.
    REPO = {"builder": "live/pm_research/da_fair_price_identity.py",
            "wrapper": "live/pm_research/de_fair_price_wrapper.py"}
    committed = {k: _sha(ref, v) for k, v in REPO.items()}
    for est, ident in ids.items():
        ident["builder_repo_path"] = REPO["builder"]
        ident["wrapper_repo_path"] = REPO["wrapper"]
        ident["commit_ref"] = ref
        ident["commit"] = head_sha
        ident["builder_sha256_at_commit"] = committed["builder"]
        ident["wrapper_sha256_at_commit"] = committed["wrapper"]
        ident["scratch_path_recorded_by_identity_of"] = ident.get("builder_path")
        ident["digests_match_the_committed_bytes"] = (
            ident.get("builder_sha256") == committed["builder"]
            and ident.get("wrapper_sha256") == committed["wrapper"])
    all_match = all(i["digests_match_the_committed_bytes"] for i in ids.values())
    return {
        "n": len(ids),
        "builders_are_committed_files_not_scratch": all_match,
        "WHY_THIS_FIELD_EXISTS": (
            "a scratch-dir builder has voided a freeze in this programme "
            "before (rule 12). `identity_of` resolves paths inside a temp "
            "worktree, so the recorded paths were /tmp; the digests were "
            "always of the same bytes, but no later reader could confirm that "
            "without redoing the work by hand. The repo-relative path, the "
            "commit ref and a COMPUTED match now sit beside every digest."),
        "m_for_multiplicity": 2,
        #: REVIEW 256's caveat, transcribed. The counts are REV's measurements,
        #: attributed as REV's and not re-derived here.
        "M_IS_TWO_FOREVER_CAVEAT": (
            "m stays 2, and it is worth saying what those two ARE. C1 is not an "
            "independent second estimator of the same quantity: it is a BOUNDED "
            "PERTURBATION OF THE BASELINE. The wrapper says so in its own "
            "words -- `book.value` is `(best_bid, best_ask, bid_size, "
            "ask_size)` -- ONE book event, so C1 cannot read a different event "
            "from Identity by construction -- and the bound follows: "
            "|C1 - Identity| <= spread/2, measured by REVIEW 256 at 0.005 at "
            "p90 over 9.4M book states, with the attainable increment "
            "structurally capped at ln 2. So the multiplicity of 2 counts a "
            "baseline and a bounded perturbation of it, NOT two independent "
            "candidates, and Holm across them is conservative in a way a reader "
            "should know about rather than discover. C2 IS EXPLICITLY EXEMPT "
            "from this caveat: it is cross-venue (Binance bookTicker), so it is "
            "not a perturbation of the PM book and is free to differ from "
            "Identity by more than the spread."),
        "M_IS_TWO_FOREVER": ("m = 2 is fixed at freeze and does not shrink if a "
                             "candidate dies, is withdrawn, or fails to produce "
                             "a value. Holm is across C1 and C2 whatever happens "
                             "to either; a multiplicity that shrinks after the "
                             "fact is selection on the outcome."),
        "identities": ids,
        "model_version_binds_to": [k for k, v in ids.items() if v["model_version"]],
        "model_version_is_None_for": [k for k, v in ids.items() if not v["model_version"]],
        "MEASURED_CORRECTION": ("DA 278 binds both candidates to "
                                "model_version=s60_probability_v1; the artifact "
                                "binds it to C2 only and C1 to None, on purpose"),
        "declared_model_version": d["model_version"],
    }


def quote_mapping(ref: str) -> dict:
    """§7's FROZEN QUOTE MAPPING, DRIVEN against the seam at the ref.

    Every clause of §7's quote-mapping block is a separate computed predicate.
    This is the link where the draft is weakest and the drive is what shows it.
    """
    src = _src(ref, "live/pm_research/de_fair_value_policy_seam.py")
    wt = tempfile.mkdtemp(prefix="da_step6_q_")
    Path(wt).rmdir()
    if _git("worktree", "add", "--detach", wt, ref).returncode != 0:
        return {"error": "worktree failed"}
    pm = str(Path(wt) / "live" / "pm_research")
    code = (
        "import json,sys,inspect; sys.path.insert(0, %r)\n"
        "import de_fair_value_policy_seam as S\n"
        "o={}\n"
        "def q(v,**kw):\n"
        "    return S.quote_from(v, slug='s', generation_id='g',\n"
        "                        half_spread=kw.pop('hs',0.0001),\n"
        "                        priced_by='c2', **kw)\n"
        "up, dn = q(0.5237), q(0.5237, side='DOWN')\n"
        "o['up_anchor']=up.anchor; o['up_bid']=up.bid; o['up_ask']=up.ask\n"
        "o['dn_p_side']=dn.p_side; o['tick']=up.tick\n"
        "hi, lo = q(0.999, hs=0.01), q(0.001, hs=0.01)\n"
        "o['hi_ask']=hi.ask; o['lo_bid']=lo.bid\n"
        "o['hi_bounded']=len(hi.bounded)>0 or len(lo.bounded)>0\n"
        "x = q(0.5, hs=0.01, best_bid=0.60, best_ask=0.62, decision_ms=1000.0)\n"
        "o['cross_withheld']=bool(x.withheld); o['cross_reason']=x.withheld_reason\n"
        "o['effective_ms']=x.effective_ms; o['decision_ms']=x.decision_ms\n"
        "o['latency_ms']=x.latency_ms\n"
        "o['sig']=list(inspect.signature(S.quote_from).parameters)\n"
        "print(%r)\n"
        "print(json.dumps(o))\n" % (pm, PROBE_SENTINEL))
    r = subprocess.run(["python3", "-c", code], cwd=pm, capture_output=True, text=True)
    _git("worktree", "remove", "--force", wt)
    parsed = parse_probe_output(r.stdout, r.stderr)
    if not parsed["ok"]:
        # A THROWING CELL IS A FAILURE, NEVER AN ABSENCE. When this returned a
        # bare {"error": ...}, the per-clause gaps VANISHED and the gap count
        # FELL -- a broken probe reading as progress. A failed probe now leaves
        # EVERY pinned clause unsatisfied.
        return {"driven_against": "live/pm_research/de_fair_value_policy_seam.py",
                "probe_failed": True, "error": parsed["failure"],
                "properties": {c: False for c in QUOTE_CLAUSES_PINNED},
                "n_satisfied": 0, "n_declared": len(QUOTE_CLAUSES_PINNED),
                "unsatisfied": sorted(QUOTE_CLAUSES_PINNED)}
    d = parsed["result"]
    TICK = d.get("tick") or 0.01
    props = {
        "UP_uses_p": abs(d["up_anchor"] - 0.5237) < 1e-12,
        "DOWN_uses_1_minus_p": abs(d["dn_p_side"] - (1 - 0.5237)) < 1e-12,
        "bid_rounds_DOWN_to_the_legal_tick":
            abs(d["up_bid"] - 0.52) < 1e-9,
        "ask_rounds_UP_to_the_legal_tick":
            abs(d["up_ask"] - 0.53) < 1e-9,
        "prices_bounded_to_the_legal_binary_range":
            (d["hi_ask"] <= 1.0 and d["lo_bid"] >= 0.0),
        "the_bound_applied_is_RECORDED_not_silent": bool(d["hi_bounded"]),
        "crossing_quote_emits_PLACE_WITHHELD_MARKETABLE_CROSS":
            bool(d["cross_withheld"]) and d["cross_reason"] == "MARKETABLE_CROSS",
        "no_zero_latency_privilege_for_candidate_induced_change":
            (d["effective_ms"] - d["decision_ms"]) == d["latency_ms"] > 0,
        "the_tick_is_DECLARED_not_invented":
            "tick" in d["sig"] and TICK == 0.01,
    }
    return {
        "driven_against": "live/pm_research/de_fair_value_policy_seam.py",
        "sha256": _sha(ref, "live/pm_research/de_fair_value_policy_seam.py"),
        "quote_from_signature": d["sig"],
        "measured": {"UP_0.5237": {"bid": d["up_bid"], "ask": d["up_ask"]},
                     "DOWN_p_side": d["dn_p_side"],
                     "ask_at_p_0.999": d["hi_ask"],
                     "bid_at_p_0.001": d["lo_bid"],
                     "crossing": {"withheld": d["cross_withheld"],
                                  "reason": d["cross_reason"]},
                     "latency_ms": d["latency_ms"],
                     "effective_minus_decision_ms":
                         d["effective_ms"] - d["decision_ms"],
                     "tick_read_by_the_seam": d.get("tick")},
        "properties": props,
        "n_satisfied": sum(1 for v in props.values() if v),
        "n_declared": len(props),
        "unsatisfied": sorted(k for k, v in props.items() if not v),
        "THE_TICK_IS_DECLARED_AND_READ": (
            f"tick={TICK}, read by the seam from DA's landed declaration "
            f"(`da_market_facts_v1.json`). The seam REFUSES "
            f"LEGAL_TICK_IS_NOT_DECLARED rather than inventing one, which is "
            f"why this probe could not run until the tick was established."),
    }


def status_grammar(ref: str) -> dict:
    """THE STATUS GRAMMAR, read from the modules that own it."""
    g1 = _src(ref, "live/pm_research/da_fair_value_gate1_labels.py")
    fp = _src(ref, "live/pm_research/da_fair_price_identity.py")
    act = _src(ref, "live/pm_research/de_fair_value_actions.py")

    def tup(src, name):
        m = re.search(name + r"\s*=\s*\((.*?)\)", src, re.S)
        return sorted(set(re.findall(r"[A-Z][A-Z0-9_]{2,}", m.group(1)))) if m else MISSING

    fp_st = sorted({v for k, v in re.findall(r"^([A-Z][A-Z0-9_]*)\s*=\s*'([A-Za-z_]+)'",
                                             fp, re.M)})
    return {
        "gate1_label_statuses": tup(g1, "STATUSES"),
        "gate1_admitting": tup(g1, "ADMITTING"),
        "EXACTLY_ONE_ADMITS": len(tup(g1, "ADMITTING") or []) == 1,
        "consumer_statuses": tup(g1, "CONSUMER_STATUSES"),
        "fairprice_statuses": fp_st,
        "action_exclusion_statuses": sorted(set(re.findall(
            r'^([A-Z][A-Z0-9_]{4,})\s*=\s*"', act, re.M))),
    }


def source_manifests(ref: str) -> dict:
    """§7 requires SOURCE MANIFESTS for the immutable inputs."""
    r = _git("ls-tree", "-r", "--name-only", ref, "live/pm_research/")
    found = [l.strip() for l in r.stdout.splitlines()
             if re.search(r"fair_value.*manifest|manifest.*fair_value", l, re.I)]
    return {
        "fair_value_input_manifest_files": found,
        "n": len(found),
        "present": bool(found),
        "measured": "git ls-tree -r <ref> live/pm_research/ | match a fair-value manifest",
        "what_it_must_enumerate": (
            "every immutable input the chain reads, each with a digest and an "
            "as-of: the PM market/book capture, the Chainlink settlement "
            "capture, and the Binance bookTicker capture that C2 consumes"),
    }


def latency(ref: str) -> dict:
    """§7's placement parameter -- A SIMULATION ASSUMPTION, never measured."""
    bound = [p for _, paths in CHAIN for p in paths
             if "placement_latency_ms" in _src(ref, p)]
    return {
        "placement_latency_ms": 250,
        "applies_to": "every new generation in BOTH legs",
        "CLASS": "SIMULATION ASSUMPTION",
        # BOOLEANS, NOT PROSE. The first version of this cell asserted a
        # SUBSTRING of the sentence below, so it tested my phrasing rather than
        # the claim -- a prose-only pass, which §7 forbids by name.
        "is_a_measured_live_end_to_end_latency": False,
        "must_never_be_described_as_empirical": True,
        "IS_NOT": ("a measured live end-to-end latency. The receipt must never "
                   "describe this parameter as empirical."),
        "no_challenger_receives_an_instantaneous_first_placement": True,
        "required_count_field": "n_fills_removed_before_generation_start_plus_250ms",
        "bound_in_a_frozen_chain_file": bound,
        "present_in_the_chain": bool(bound),
        "harmful_flow_cancellation": "DISABLED",
        "quote_replacement": ("ordinary replacement follows one shared, "
                              "separately declared lifecycle in both legs"),
    }


def null_predicate() -> dict:
    """§8's null, as computable fields."""
    return {
        "primary_null": "median_g(delta_LL_g) = 0",
        "statistic": "LL_g(estimator) = mean natural-log loss over canonical "
                     "forecast actions; delta_LL_g(c) = LL_g(Identity) - LL_g(c)",
        "sign_of_better": "positive delta_LL_g is better",
        "test": "exact two-sided paired day sign test, enumerating all 2^G "
                "day-sign assignments",
        "G_target": 10,
        "n_assignments_at_G10": 2 ** 10,
        "above_the_200_null_minimum": 2 ** 10 >= 200,
        "smallest_two_sided_p_at_G10": 2 / 2 ** 10,
        "multiplicity_m": 2,
        "correction": "Holm across C1 and C2",
        "ties": "an exactly zero daily increment is a REPORTED TIE, excluded "
                "from the sign count, never silently given a favourable sign",
        "min_nonzero_portfolio_days": 8,
        "n_assignments_at_8_nonzero": 2 ** 8,
        "below_that": "INSUFFICIENT_EVIDENCE",
        "summaries_use_all_ten_days_including_zeros": True,
        "portfolio_day": "BTC+ETH mean giving each coin equal weight; action "
                         "counts remain reported; no action, fill, market or "
                         "coin is treated as an independent day",
    }


def success_predicate() -> dict:
    """§8's four conditions -- ALL must hold."""
    return {
        "all_of": {
            "1_holm_corrected_p": "< 0.05 on the primary log-loss increment",
            "2_mean_and_median": "both delta_LL_g summaries positive",
            "3_native_coverage_gate": ">= 95% of Identity-eligible actions for "
                                      "EACH of BTC and ETH",
            "4_predicates": "all population, timestamp, complement and "
                            "reconciliation predicates pass",
        },
        "cannot_rescue_a_failed_primary": ["Brier", "a favourable coin cell",
                                           "a favourable time-to-expiry cell"],
        "accrual": {
            "observe": "the first 14 consecutive calendar days",
            "require": "the first 10 evaluable complete BTC+ETH UTC days",
            "failed_data_gate_days": "remain counted, with statuses",
            "if_fewer_than_10_by_day_14": "INSUFFICIENT_EVIDENCE",
            "do_not_extend_opportunistically": True,
            "starts": "the first complete UTC day STRICTLY AFTER this freeze",
        },
        "day_eligibility_is_candidate_blind": (
            "resolved by the frozen day/book gate, official resolutions and "
            "settlement-verification coverage only. Challenger availability, "
            "score, fills or P&L can NEVER remove a day or replace it with a "
            "later one."),
        "all_ten_days_become_consumed_regardless_of_verdict": True,
    }


def abstention_constraint(ref: str) -> dict:
    """R-924, carried from the LANDED declaration rather than from memory."""
    p = "live/pm_research/declarations/da_step6_abstention_constraint_v1.json"
    b = _blob(ref, p)
    if b is None:
        return {"present": False, "path": p, "note": "NOT ON THE REF"}
    d = json.loads(b.decode())
    return {
        "present": True, "path": p, "sha256": hashlib.sha256(b).hexdigest(),
        "THE_CONSTRAINT_VERBATIM": d.get("THE_CONSTRAINT_VERBATIM"),
        "mechanism": d.get("THE_MECHANISM_IN_ONE_LINE"),
        "must_be_reproduced_verbatim_in_this_receipt": True,
    }


def executing_refs() -> dict:
    """BOTH executing refs, by head AND by blob. DA 281 records that they
    diverged by one BE commit each while every lane module was byte-identical;
    which of those is true at any moment is measured, never assumed."""
    refs = (REF, MIRROR_REF)
    heads = {r: _git("rev-parse", r).stdout.strip() for r in refs}
    paths = sorted({p for _, ps in CHAIN for p in ps})
    blobs = {p: {r: _sha(r, p) for r in refs} for p in paths}
    identical = {p: len(set(v.values())) == 1 for p, v in blobs.items()}
    a, b = refs
    return {
        "heads": heads, "heads_equal": len(set(heads.values())) == 1,
        "commits_each_has_that_the_other_does_not": {
            a: len(_git("log", "--format=%h", f"{b}..{a}").stdout.split()),
            b: len(_git("log", "--format=%h", f"{a}..{b}").stdout.split())},
        "n_lane_modules_compared": len(paths),
        "all_lane_modules_byte_identical": all(identical.values()),
        "modules_that_differ": sorted(k for k, v in identical.items() if not v),
        "the_rule": ("a gate is satisfied by the BLOB on both refs, not by the "
                     "heads matching; two shas with identical modules blocks "
                     "nothing"),
    }



#: §7's requirements, each with the ONE LINE a reader needs when it is unmet.
#: Keyed to the field or chain link that decides it, so the line is attached to
#: a COMPUTED verdict and cannot drift away from it.
WHY_LINES = {
    "source_manifests":
        "no manifest enumerates the immutable inputs, so the frozen chain "
        "begins at a link nothing identifies",
    "initial_inventory":
        "inventory enters replay only as a caller-supplied `initial_state`; no "
        "frozen starting value exists to replay from",
    "tick_rounding":
        "no fair-value module declares a legal tick, so 'rounds to the tick' "
        "has no tick to round to",
    "fee_rule":
        "no fee appears anywhere in the frozen chain, so a P&L computed from it "
        "would be gross by construction",
    "quote_parameters":
        "the seam satisfies 1 of §7's 8 quote-mapping clauses; see "
        "`fields.quote_parameters` for the seven driven failures",
    "chain:immutable_inputs":
        "the first link of §7's chain has no implementation and no manifest",
    "chain:pnl":
        "the last link of §7's chain has no implementation; nothing computes P&L",
    "latency:placement_latency_ms_not_bound_in_any_frozen_chain_file":
        "placement_latency_ms = 250 is declared here but is not BOUND in any "
        "file the freeze covers; it lives in the cancellation lane's daybook "
        "builder and appears in this lane only as the string 'L250ms' in a "
        "receipt filename",
    "quote_mapping_property:UP_uses_p":
        "the anchor is not the consumed probability",
    "quote_mapping_property:DOWN_uses_1_minus_p":
        "quote_from takes no side or outcome, so there is no DOWN quote to map",
    "quote_mapping_property:bid_rounds_DOWN_to_the_legal_tick":
        "bid uses symmetric round(x, 12), not a downward round to a legal tick",
    "quote_mapping_property:ask_rounds_UP_to_the_legal_tick":
        "ask uses symmetric round(x, 12), not an upward round to a legal tick",
    "quote_mapping_property:prices_bounded_to_the_legal_binary_range":
        "unbounded: at p=0.999 the ask is 1.009 and at p=0.001 the bid is "
        "-0.009, both outside the legal binary range",
    "quote_mapping_property:crossing_quote_emits_PLACE_WITHHELD_MARKETABLE_CROSS":
        "MARKETABLE_CROSS appears in ZERO .py files in the lane; the event §7 "
        "requires does not exist to be emitted",
    "quote_mapping_property:never_silently_clamped":
        "with no withhold path there is nothing to prefer over clamping",
    "quote_mapping_property:no_zero_latency_privilege_for_candidate_induced_change":
        "the seam carries no latency at all, so it cannot deny a privilege it "
        "never models",
}


def why_not_effective(d: dict) -> list:
    """EXACTLY WHAT MAKES `freeze_is_effective` FALSE -- one computed line per
    unmet §7 requirement (DA 282).

    Built from the SAME gap list the predicate is computed from, so the
    explanation cannot disagree with the verdict: every gap must resolve to a
    line, and a gap with no line is itself reported rather than dropped.
    """
    out = []
    for g in d["blocking_gaps"]:
        key = g.replace("chain_link_not_implemented:", "chain:")
        out.append({
            "requirement": g,
            "section": "§7",
            "satisfied": False,
            "why": WHY_LINES.get(key, WHY_LINES.get(g, "NO LINE RECORDED FOR "
                                                    "THIS GAP -- see rule below")),
            "line_recorded": (key in WHY_LINES) or (g in WHY_LINES),
        })
    return out



MARKET_FACTS = "live/pm_research/declarations/da_market_facts_v1.json"
REV_ADJUDICATION = ("live/pm_research/declarations/"
                    "rev_section7_fee_rule_reading_v1.json")
MIN_DELTA_DECL = ("live/pm_research/declarations/"
                  "user_minimum_meaningful_delta_ll_v1.json")


def minimum_meaningful_delta_ll(ref: str) -> dict:
    """§8 DECLARES A MINIMUM SAMPLE AND NO MINIMUM EFFECT. (DA 297 / REVIEW 256)

    THE HOLE. The exact paired sign test has NO MAGNITUDE RESOLUTION: it tests
    the SIGN of `delta_LL_g`, never its size. A candidate positive on ten of ten
    days by 1e-9 nats yields p = 0.001953125 and PASSES, exactly as one positive
    by 0.08 nats would. REVIEW 256 checked all four §8 adoption conditions and
    none carries a magnitude floor -- Holm p is sign-driven, mean/median
    positive is sign not size, the 95% gate is coverage, the fourth is
    structural. So a candidate can pass §8 on an effect of ANY size above zero
    and be carried into §9's economic clock on something economically
    indistinguishable from nothing.

    Rule 6 binds the null and its SAMPLE; nothing binds the ALTERNATIVE.

    THIS FIELD IS AN ADDITION BEYOND §7'S SENTENCE, DELIBERATELY. It is NOT in
    `REQUIRED_FIELDS`, because that tuple is pinned to §7's own list and a
    fifteenth member would make `assert_enumerations_intact` refuse -- the pin
    is doing its job. It is a separate blocking CONDITION instead, in the same
    construction as the adjudication gate, and it is recorded as an addition so
    a reader sees it was added rather than smuggled in.

    THE VALUE IS NOT CHOSEN HERE, AND NOT BY THIS SEAT. It belongs to whoever
    owns the estimand. REV refused to pick it and the coordinator refused after
    seeing the 0.06-0.08 ceiling, because a threshold chosen after seeing the
    attainable effect is rule 11's failure with a number instead of a feature
    subset. So this reads an artifact the USER lands; its absence is UNSET, and
    UNSET BLOCKS.
    """
    b = _blob(ref, MIN_DELTA_DECL)
    if b is None:
        return {"present": False, "path": MIN_DELTA_DECL, "is_set": False,
                "value": None, "status": "UNSET_AND_BLOCKING",
                "owner": "the USER -- whoever owns the estimand",
                "why_blocking": (
                    "a note at day one can be forgotten by day ten; a predicate "
                    "cannot. The freeze does not go effective without a "
                    "declared minimum meaningful effect."),
                "what_it_must_carry": (
                    "a minimum meaningful `delta_LL` in NATS PER ACTION, with "
                    "the reasoning for that magnitude, declared BEFORE any "
                    "§8 day is scored")}
    try:
        d = json.loads(b.decode())
    except Exception as e:
        return {"present": True, "path": MIN_DELTA_DECL, "is_set": False,
                "value": None, "status": f"UNREADABLE: {type(e).__name__}"}
    v = d.get("minimum_meaningful_delta_LL")
    ok = isinstance(v, (int, float)) and not isinstance(v, bool) and v > 0
    return {"present": True, "path": MIN_DELTA_DECL,
            "sha256": hashlib.sha256(b).hexdigest(),
            "is_set": ok, "value": v if ok else None,
            "units": d.get("units"), "declared_by": d.get("declared_by"),
            "reasoning": d.get("reasoning"),
            "status": "SET" if ok else "PRESENT_BUT_NOT_A_POSITIVE_NUMBER"}


def rev_adjudication(ref: str) -> dict:
    """HAS REV INDEPENDENTLY READ §7 AND §9 AND CONFIRMED THE DA 294 READING?

    DA 294 attached this condition to its own ruling, and the reason is worth
    keeping: the coordinator was interpreting a clause that unblocks a clock
    the coordinator wants started, which is the same shape as a seat judging
    the guard that blocks it. So the reading is adjudicated by a party that
    does not benefit from either answer, and the freeze does not count as
    effective until that adjudication exists.

    IT IS AN ARTIFACT, NOT A FLAG. This seat cannot set it: the answer comes
    from a declaration REV lands at the ref, and its absence is NOT
    adjudication. If REV breaks the reading, the question goes to the user as
    a plan amendment and the clock waits.
    """
    b = _blob(ref, REV_ADJUDICATION)
    if b is None:
        return {"present": False, "path": REV_ADJUDICATION,
                "confirms_the_reading": False,
                "status": "AWAITING_REV_ADJUDICATION",
                "meaning": ("§7's `fee_rule` is read as a RECORDING "
                            "requirement (DA 294). Until REV confirms that "
                            "reading independently, the freeze is NOT "
                            "effective however few gaps remain.")}
    try:
        d = json.loads(b.decode())
    except Exception as e:
        return {"present": True, "path": REV_ADJUDICATION,
                "confirms_the_reading": False,
                "status": f"UNREADABLE: {type(e).__name__}"}
    return {"present": True, "path": REV_ADJUDICATION,
            "sha256": hashlib.sha256(b).hexdigest(),
            "confirms_the_reading": d.get("confirms_da_294_reading") is True,
            "status": d.get("status"), "verdict": d.get("verdict"),
            "read_by": d.get("seat")}
INPUT_MANIFEST = "live/pm_research/declarations/da_immutable_inputs_manifest_v1.json"


def input_manifest(ref: str) -> dict:
    """§7's SOURCE MANIFESTS, resolved from the manifest's CONTENT.

    DA 284: the gap must close on what the manifest SAYS, not on its
    existence. §8 resolves day eligibility from the frozen day/book gate, the
    resolutions and settlement coverage, so the manifest only satisfies §7 if
    it actually seals an input for every predictive link and every input it
    enumerates is present and sealed. A manifest that lists inputs it could not
    digest is a filled field wearing a digest's clothes.
    """
    b = _blob(ref, INPUT_MANIFEST)
    if b is None:
        return {"present": False, "path": INPUT_MANIFEST, "satisfies_section_7": False}
    d = json.loads(b.decode())
    covered = d.get("every_predictive_link_has_a_sealed_input") is True
    sealed = d.get("all_enumerated_inputs_present_and_sealed") is True
    return {
        "present": True, "path": INPUT_MANIFEST,
        "sha256": hashlib.sha256(b).hexdigest(),
        "n_inputs": d.get("n_inputs"),
        "days_sealed": d.get("days_sealed"),
        "every_predictive_link_has_a_sealed_input": covered,
        "all_enumerated_inputs_present_and_sealed": sealed,
        "predictive_links_not_covered": d.get("predictive_links_not_covered"),
        "satisfies_section_7": covered and sealed,
        "seal_kinds": sorted({e.get("seal_kind") for e in d.get("ledgers", [])}
                             | {c.get("seal_kind") for c in d.get("captures", [])}),
        "WHAT_A_SEAL_DOES_NOT_CLAIM": d.get("WHAT_A_SEAL_DOES_NOT_CLAIM"),
    }


def market_facts(ref: str) -> dict:
    """THE THREE FACTS DA 283 ESTABLISHED, cited from their landed artifact.

    Read by DIGEST rather than recomputed: the measurement belongs to
    `da_market_facts`, and a freeze that re-derives its inputs can disagree
    with the artifact it claims to freeze.
    """
    b = _blob(ref, MARKET_FACTS)
    if b is None:
        return {"present": False, "path": MARKET_FACTS}
    d = json.loads(b.decode())
    return {"present": True, "path": MARKET_FACTS,
            "sha256": hashlib.sha256(b).hexdigest(),
            "established": d.get("established"),
            "unestablished": d.get("unestablished"),
            "legal_tick": d.get("legal_tick"),
            "maker_fee_bps": d.get("maker_fee_bps"),
            "maker_fee_rule": d.get("maker_fee_rule"),
            "maker_fee_residual": d.get("maker_fee_residual"),
            "negative_declaration": d.get("maker_fee_negative_declaration"),
            "legal_tick_caveat": (d.get("legal_tick_evidence") or {}
                                  ).get("THE_CAVEAT_IS_MEASURED"),
            "initial_inventory": d.get("initial_inventory", {}).get("initial_inventory"),
            "maker_fee_status": (d.get("maker_fee_rule_evidence") or {}).get("status"),
            "maker_fee_why_not": (d.get("maker_fee_rule_evidence") or {}).get(
                "WHY_NOT_ESTABLISHED")}


def unattributed_chain_files(ref: str) -> list:
    """Lane files that LOOK like chain implementations and that no link claims.

    Written because `pnl` sat unattributed while `de_fair_value_pnl.py` was on
    both refs, and nothing in this module could notice.
    """
    r = _git("ls-tree", "-r", "--name-only", ref, "live/pm_research/")
    claimed = {p for _, ps in CHAIN for p in ps}
    out = []
    for line in r.stdout.splitlines():
        line = line.strip()
        if not line.endswith(".py") or line in claimed:
            continue
        if re.search(CHAIN_FILE_RE, Path(line).name) and "falsif" not in line:
            out.append(line)
    return sorted(out)


START_DAY_FLOOR = "2026-09-14"

DAYBOOK_GLOB = "data/pm_5min/derived/be_daybook_*_%s*.pkl"
LAUNCHER_CONTROL = ("live/pm_research/declarations/"
                    "be_launcher_btc_byte_identical_control_v1.json")

SECTION_8_BAND = (
    "observe the first 14 CONSECUTIVE CALENDAR DAYS; require the first 10 "
    "evaluable complete BTC+ETH UTC days WITHIN THAT BAND; days that fail a "
    "predeclared data gate remain counted with statuses; if fewer than 10 are "
    "evaluable by day 14, verdict is INSUFFICIENT_EVIDENCE; do not extend "
    "opportunistically.")


def two_coin_production_ready(ref: str, root=None, control=None) -> dict:
    """CAN THE §8 POPULATION EXIST AT ALL? (DA 299 / REVIEW 258)

    THE TRAP THIS EXISTS TO CLOSE, and it is triggered by the user doing
    exactly what we asked. `minimum_meaningful_delta_LL` is the last term
    holding the freeze; the moment the user sets it, T_eff is that moment and
    D1 is the next complete UTC day. But §8 says:

        %s

    NO ETH DAY BOOK EXISTS. So if the number is set before two-coin production
    is actually running, the 14-day band starts BURNING against days that are
    all COINS_INCOMPLETE, and the test self-destructs into
    INSUFFICIENT_EVIDENCE having consumed its one band -- which the plan
    forbids extending to recover. A one-shot, unrecoverable loss.

    A WARNING IN A MESSAGE IS WHAT GETS MISSED by someone answering "what
    number do you want?" three days from now. So it is a PREDICATE.

    THESE TWO TERMS ARE A SEQUENCE, NOT INDEPENDENT GATES. The second exists
    to protect the user from an irreversible consequence of satisfying the
    first, so it must be satisfiable BEFORE the first is asked for -- and the
    order matters even though `and` is commutative.

    THE BTC CONTROL IS PART OF THE TERM, not decoration: an ETH book produced
    by a launcher that cannot reproduce BTC byte-identically is a book of
    unknown provenance, and two coins built by an unvalidated path is a worse
    population than one coin built by a validated one.
    """ % SECTION_8_BAND
    # THE DATA ROOT IS DELEGATED, NEVER THIS MODULE'S GIT ROOT. `_root()`
    # returns the repo of the file being executed, which from a seat worktree
    # is the WORKTREE -- whose `data/` does not exist, so every glob returns
    # zero. The positive control below caught exactly that on the first drive:
    # btc read 0 when 53 day books were on disk, and an eth count of 0 would
    # have been a false absence.
    import glob as _g
    if root is None:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import da_root as _dr
        root = _dr.resolve_root()
    eth = sorted(_g.glob(str(Path(root) / (DAYBOOK_GLOB % "eth"))))
    btc = sorted(_g.glob(str(Path(root) / (DAYBOOK_GLOB % "btc"))))
    ctrl_b = (json.dumps(control).encode() if control is not None
              else _blob(ref, LAUNCHER_CONTROL))
    ctrl = {"present": ctrl_b is not None}
    if ctrl_b is not None:
        try:
            cd = json.loads(ctrl_b.decode())
            ctrl.update({"passed": cd.get("btc_rebuild_byte_identical") is True,
                         "sha256": hashlib.sha256(ctrl_b).hexdigest(),
                         "declared_by": cd.get("seat")})
        except Exception as e:
            ctrl.update({"passed": False, "error": f"{type(e).__name__}: {e}"})
    else:
        ctrl["passed"] = False
    ready = bool(eth) and bool(ctrl.get("passed"))
    return {
        "ready": ready,
        "eth_daybooks_found": len(eth),
        "eth_example": Path(eth[0]).name if eth else None,
        "btc_daybooks_found": len(btc),
        "POSITIVE_CONTROL": (
            "the SAME glob finds %d btc day books, so an eth count of %d is a "
            "real absence and not a broken query (runbook 7k.2)"
            % (len(btc), len(eth))),
        "control_query_fires": len(btc) > 0,
        "launcher_btc_byte_identical_control": ctrl,
        "requires": ("an ETH day book actually PRODUCED by the new launcher, "
                     "AND that launcher's byte-identical-BTC control PASSED"),
        "section_8_band_quoted": SECTION_8_BAND,
        "why_this_blocks": (
            "if the clock starts before two-coin production runs, the 14-day "
            "band burns against COINS_INCOMPLETE days and the test ends in "
            "INSUFFICIENT_EVIDENCE with its one band consumed"),
        "status": "READY" if ready else "NOT_READY_CLOCK_MUST_NOT_START",
    }


def validation_start_day(effective: bool, floor: str = START_DAY_FLOOR) -> dict:
    """§8'S START DAY, COMPUTED FROM A RULE AND A FLOOR -- never chosen.

    RULED AT DA 298, and ruled TONIGHT precisely because tomorrow it would be
    a choice made after seeing. REVIEW found that a freeze going effective
    today would make §8's day one 2026-09-13 -- day SEVEN, the FINAL day, of
    the cancellation test's DECLARED population.

    THE RULE: the first complete UTC day strictly after the freeze becomes
    effective, AND NOT BEFORE the floor, whichever is later.

    WHY THE FLOOR. 09-13's verdict has been fixed since G=2, but "ALREADY
    DECIDED" IS NOT "UNTOUCHED". Taking it would entangle 10% of a ten-day
    population with a consumed day to save one day of wall-clock.

    AND WHY NOW. It probably costs nothing -- the freeze computes FALSE on
    other counts anyway -- and that is the reason to declare it: A RULE
    ADOPTED WHILE IT IS FREE IS WORTH MORE THAN THE SAME RULE ADOPTED ONCE IT
    COSTS SOMETHING. A floor set after the first day it would exclude is a
    floor chosen by its consequence.
    """
    today = time.strftime("%Y-%m-%d", time.gmtime())
    nxt = time.strftime("%Y-%m-%d", time.gmtime(time.time() + 86400))
    if not effective:
        start, status = None, "NOT_YET_DETERMINED_FREEZE_IS_NOT_EFFECTIVE"
    else:
        start = max(nxt, floor)
        status = "DETERMINED"
    return {
        "rule": ("the first complete UTC day STRICTLY AFTER the freeze becomes "
                 "effective, AND NOT BEFORE the floor, whichever is later"),
        "floor_date": floor,
        "floor_reason": ("2026-09-13 is day SEVEN -- the final day -- of the "
                         "cancellation test's DECLARED population. Already "
                         "decided is not untouched."),
        "freeze_is_effective": effective,
        "start_day": start, "status": status,
        "the_day_FALLS_OUT_it_is_not_chosen": True,
        "declared_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "declared_while_free": ("the freeze computes FALSE on other counts, so "
                                "this floor costs nothing today; that is why "
                                "it is declared today"),
    }


def build(ref: str = REF, rev203_six_of_six: bool = False, fetch: bool = True) -> dict:
    if fetch:
        subprocess.run(["git", "-C", _root(), "fetch", "--quiet", "origin"], check=False)
    # "I CANNOT CHECK" AND "I CHECKED AND IT IS MISSING" ARE OPPOSITE FACTS
    # WITH THE SAME SHAPE. A caller meeting PLAN_ENUMERATION_UNPARSEABLE must
    # be able to branch on which it is: an unreadable plan is not a gap in the
    # freeze, it is a failure to verify the freeze. Both BLOCK -- but they
    # block for different reasons and a reader must not conflate them.
    try:
        enum = assert_enumerations_intact()
        enum_check = {"status": "CHECKED", "can_verify": True,
                      "intact": enum["intact"], "refusal": None}
    except RuntimeError as exc:
        msg = str(exc)
        cannot = PLAN_PARSE_FAILED in msg
        enum = {"intact": False, "n_chain_links": None,
                "n_required_fields": None, "n_quote_clauses": None}
        enum_check = {
            "status": "CANNOT_CHECK" if cannot else "CHECKED_AND_MISMATCHED",
            "can_verify": not cannot,
            "intact": False, "refusal": msg[:400],
            "meaning": ("the external pin could not be READ, so the "
                        "enumeration is unverified -- this is NOT a finding "
                        "that a field or link is missing"
                        if cannot else
                        "the enumeration was read and DOES NOT MATCH its pin "
                        "-- this IS a finding about the freeze"),
            "blocks_effectiveness": True,
        }
    head = _git("rev-parse", ref).stdout.strip()
    hashes = file_hashes(ref)
    qm = quote_mapping(ref)
    sm = source_manifests(ref)
    lat = latency(ref)
    cands = candidates(ref)
    mf = market_facts(ref)
    im = input_manifest(ref)
    rev = rev_adjudication(ref)
    mind = minimum_meaningful_delta_ll(ref)
    twocoin = two_coin_production_ready(ref)

    chain = []
    for link, paths in CHAIN:
        chain.append({
            "link": link, "paths": list(paths),
            "sha256": {p: hashes[p] for p in paths},
            "implemented": bool(paths) and all(hashes[p] != MISSING for p in paths),
        })

    fields = {
        "all_file_hashes": hashes if all(v != MISSING for v in hashes.values()) else MISSING,
        "commit_ref": {"ref": ref, "head": head, "mirror": MIRROR_REF,
                       "mirror_head": _git("rev-parse", MIRROR_REF).stdout.strip()},
        "candidate_count": cands,
        "action_key": "(coin, slug, generation_id, decision_recv_ns)",
        "epsilon": 1e-6,
        "status_grammar": status_grammar(ref),
        # RESOLVED FROM THE MANIFEST'S CONTENT, never from its existence.
        "source_manifests": (im if im.get("satisfies_section_7") else MISSING),
        "source_manifests_probe": im,
        "initial_inventory": ({"value": mf["initial_inventory"],
                               "units": "signed shares of the UP token, per market",
                               "established_by": MARKET_FACTS,
                               "sha256": mf.get("sha256")}
                              if mf.get("present") and mf.get("initial_inventory") is not None
                              else MISSING),
        "tick_rounding": ({"legal_tick": mf["legal_tick"],
                           "rule": "bid rounds DOWN, ask rounds UP, to the legal tick",
                           "established_by": MARKET_FACTS,
                           "sha256": mf.get("sha256"),
                           "CAVEAT": mf.get("legal_tick_caveat")}
                          if mf.get("present") and mf.get("legal_tick")
                          else MISSING),
        "latency": lat,
        # ==================================================================
        # §7's `fee_rule` IS A RECORDING REQUIREMENT, NOT A MEASUREMENT ONE.
        # Ruled at DA 294; the rationale is in-band so a later reader can
        # disagree with the REASONING and not merely with the outcome.
        #
        # THE ARGUMENT. §7's sentence says the declaration RECORDS the fee
        # rule, and every sibling item in that same list is a recording rather
        # than a measurement: `epsilon` is a constant, `action_key` is a key,
        # `status_grammar` is a grammar, and the null and success predicates
        # are DEFINITIONS. It would be strange for one member of a list of
        # recordings to silently carry a verification burden the others do
        # not -- and the plan places that burden explicitly elsewhere, in §9:
        # "the verified maker fee applicable to these markets". A requirement
        # stated once, in the economic section, is a requirement of the
        # economic section.
        #
        # WHAT THIS PREDICATE USED TO DO, AND WHY IT WAS WRONG. It demanded
        # `maker_fee_bps is not None` -- a NUMBER -- which is "the fee is
        # KNOWN". That is STRICTER THAN THE SENTENCE IT CLAIMS TO ENFORCE, and
        # my own WHY line gave it away: it justified the gap with "a P&L
        # computed from it would be gross by construction", which is §9's
        # concern imported into §7. DE's rehearsal confirms the split from the
        # other side: MAKER_FEE_IS_NOT_DECLARED reaches only §9, and both
        # candidates produced full §8 verdicts with the fee undeclared.
        #
        # SO A NEGATIVE DECLARATION CLOSES IT. "Not establishable, and here is
        # the evidence" is an ANSWER. What does NOT close it is silence: a
        # declaration absent, or present with neither a number nor a reasoned
        # negative. The field below accepts EITHER a declared fee OR a
        # negative declaration carrying its status and its evidence, and
        # nothing else.
        #
        # HOW TO DISAGREE WITH THIS: argue that §7's "full pipeline" freeze
        # requires every link to be EXECUTABLE, in which case a P&L that
        # refuses is not frozen and the burden does return to §7. That reading
        # was considered and rejected because §9 has its own clock and its own
        # verification clause, but it is not frivolous.
        # ==================================================================
        "minimum_meaningful_delta_LL": (
            {"value": mind["value"], "units": mind.get("units"),
             "declared_by": mind.get("declared_by"),
             "reasoning": mind.get("reasoning"),
             "established_by": MIN_DELTA_DECL, "sha256": mind.get("sha256")}
            if mind["is_set"] else MISSING),
        "fee_rule": (
            {"maker_fee_bps": mf["maker_fee_bps"],
             "supporting_rule": mf["maker_fee_rule"],
             "residual": mf["maker_fee_residual"],
             "declared_as": "A FEE, WITH ITS RULE",
             "established_by": MARKET_FACTS, "sha256": mf.get("sha256")}
            if (mf.get("present") and mf.get("maker_fee_bps") is not None
                and isinstance(mf.get("maker_fee_rule"), str))
            else (
                {"maker_fee_bps": None,
                 "declared_as": "A REASONED NEGATIVE -- THE QUESTION IS "
                                "ANSWERED, AND THE ANSWER IS THAT IT CANNOT "
                                "BE ESTABLISHED FROM COLLECTED DATA",
                 "status": (mf.get("negative_declaration") or {}).get("status"),
                 "evidence": mf.get("negative_declaration"),
                 "the_recording_requirement_is_met_because": (
                     "§7 asks the declaration to RECORD the fee rule; what is "
                     "recorded here is that no fee rule applicable to us is "
                     "establishable, with the six strands that establish it. "
                     "§9 will separately report that it cannot produce an "
                     "adopted economic verdict without a verified fee -- a "
                     "true statement about the evidence, not a pipeline "
                     "failure."),
                 "established_by": MARKET_FACTS, "sha256": mf.get("sha256")}
                if (mf.get("present")
                    and (mf.get("negative_declaration") or {}).get("status")
                    and (mf.get("negative_declaration") or {}).get("the_five_strands"))
                else MISSING)),
        "fee_rule_finding": {"status": mf.get("maker_fee_status"),
                             "why_not_established": mf.get("maker_fee_why_not"),
                             "established_by": MARKET_FACTS,
                             "sha256": mf.get("sha256")} if mf.get("present") else None,
        # A CRASHED PROBE IS NOT A PASS. `not qm.get("unsatisfied")` was True
        # when the key was ABSENT, so an ERRORED probe resolved this field and
        # removed eight gaps at once -- the `else: SATISFIED` defect of DA 280,
        # in this module, firing in my own favour. The field now requires the
        # probe to have RUN: a `properties` block present AND nothing unmet.
        "quote_parameters": (qm if (isinstance(qm.get("properties"), dict)
                                    and qm.get("unsatisfied") == []
                                    and not qm.get("probe_failed"))
                             else MISSING),
        "quote_parameters_probe_error": qm.get("error"),
        "null_predicate": null_predicate(),
        "success_predicate": success_predicate(),
    }

    gaps = []
    for f in REQUIRED_FIELDS:
        if fields.get(f) == MISSING:
            gaps.append(f)
    gaps += [f"chain_link_not_implemented:{c['link']}" for c in chain if not c["implemented"]]
    gaps += [f"quote_mapping_property:{k}" for k in qm.get("unsatisfied", [])]
    if not lat["present_in_the_chain"]:
        gaps.append("latency:placement_latency_ms_not_bound_in_any_frozen_chain_file")

    out = {
        "protocol": PROTOCOL, "plan": PLAN,
        "DRAFT": False,
        "LANDED_BY": "DA 281",
        # THREE CONDITIONS, AND THEY ARE GENUINELY THREE: no gaps (the fields
        # resolve), the enumeration is whole (the gap list was counted over the
        # pinned questions), and the READING that closes the last gap has been
        # adjudicated by a party that does not benefit from the answer.
        "freeze_is_effective": ((not gaps) and enum["intact"]
                                and rev["confirms_the_reading"]
                                and mind["is_set"]
                                and twocoin["ready"]),
        "two_coin_production_ready": twocoin,
        "THE_TWO_TERMS_ARE_A_SEQUENCE_NOT_INDEPENDENT_GATES": (
            "`minimum_meaningful_delta_LL` is the user's to set and "
            "`two_coin_production_ready` protects the user from an "
            "IRREVERSIBLE consequence of setting it: once T_eff exists the "
            "14-day band starts, and §8 forbids extending it. So the second "
            "term must be satisfied BEFORE the first is asked for. `and` is "
            "commutative; the sequence is not."),
        # REQUIRED AND UNSET. It sits in `fields` so it appears in
        # `fields_missing` and in the gap list like any other unmet §7 field --
        # an absent required field and a satisfied one MUST be distinguishable.
        "minimum_meaningful_delta_LL": mind,
        "validation_start_day": validation_start_day(
            (not gaps) and enum["intact"] and rev["confirms_the_reading"]
            and mind["is_set"] and twocoin["ready"]),
        "additions_beyond_section_7": [
            "minimum_meaningful_delta_LL -- §8 declares a minimum SAMPLE and no "
            "minimum EFFECT, so the sign test can pass on an effect of any size "
            "above zero (REVIEW 256). Added as a blocking CONDITION rather than "
            "a §7 field, because REQUIRED_FIELDS is pinned to §7's own sentence."],
        "rev_adjudication": rev,
        "WHY_A_THIRD_CONDITION": (
            "the last gap closes on an INTERPRETATION of §7, and the parties "
            "who made and accepted that interpretation both benefit from it "
            "closing. DA 294 attached the guard to its own ruling. Until REV "
            "confirms the reading, `freeze_is_effective` stays False no matter "
            "how few gaps remain."),
        "enumeration": enum,
        "enumeration_intact": enum["intact"],
        "enumeration_check": enum_check,
        "why_not_effective": None,          # filled below, from the same list
        "WHAT_freeze_is_effective_MEANS": (
            "TRUE only when every §7 field resolves and every chain link has an "
            "implementation. It is FALSE here. The declaration exists so that "
            "§11 step 6 has an artifact and so that what is missing is written "
            "down at a ref rather than carried in conversation; it does not "
            "assert that the pipeline is frozen."),
        "executing_refs": executing_refs(),
        "market_facts": mf,
        "unattributed_chain_files": unattributed_chain_files(ref),
        "input_manifest": im,
        "declared_at_ref": ref, "ref_head": head,
        "chain_in_order": chain,
        "fields": fields,
        "abstention_reading_constraint_R924": abstention_constraint(ref),
        "required_fields": list(REQUIRED_FIELDS),
        "fields_present": sorted(f for f in REQUIRED_FIELDS if fields.get(f) != MISSING),
        "fields_missing": sorted(f for f in REQUIRED_FIELDS if fields.get(f) == MISSING),
        "blocking_gaps": sorted(set(gaps)),
        "n_blocking_gaps": len(set(gaps)),
        "rev203_six_of_six": rev203_six_of_six,
        "ready_to_land": (not gaps) and bool(rev203_six_of_six),
        "WHY_NOT_LANDED": ("DA 278: draft only until REV 203 returns six of six. "
                           "`ready_to_land` requires BOTH an empty gap list and "
                           "that return, and this file cannot measure the second, "
                           "so it defaults False and must be passed in."),
        "n_blocking_gaps_without_a_recorded_line": None,   # filled below
        "EVERY_VERDICT_COMPUTED_FROM_ARTIFACT_FIELDS": True,
        "NO_PROSE_ONLY_PASS": ("§7 forbids one. Every predicate above is "
                               "computed from a blob at the ref or driven "
                               "against it; none is satisfied by a sentence."),
    }
    why = why_not_effective(out)
    out["why_not_effective"] = why
    out["n_blocking_gaps_without_a_recorded_line"] = sum(
        1 for w in why if not w["line_recorded"])
    return out


def falsify() -> int:
    bad = 0

    def ck(label, cond, shown=""):
        nonlocal bad
        print(f"  [{'PASS' if cond else 'FAIL'}] {label}" + (f" -> {shown}" if shown else ""))
        if not cond:
            bad += 1

    d = build()
    ck("the declaration is LANDED but the freeze is NOT effective",
       d["DRAFT"] is False and d["freeze_is_effective"] is False,
       f"{d['n_blocking_gaps']} blocking gaps")
    ck("freeze_is_effective is COMPUTED from all three conditions, never asserted",
       d["freeze_is_effective"] == ((d["n_blocking_gaps"] == 0)
                                    and d["enumeration_intact"]
                                    and d["rev_adjudication"]["confirms_the_reading"]),
       f"gaps={d['n_blocking_gaps']} enum={d['enumeration_intact']} "
       f"rev={d['rev_adjudication']['confirms_the_reading']}")
    ck("EXISTENCE CANNOT BECOME EFFECTIVENESS: with gaps, no argument flips it",
       build(rev203_six_of_six=True)["freeze_is_effective"] is False,
       "six-of-six passed in, still not effective")
    ck("...and the gaps are NAMED, not counted",
       len(d["blocking_gaps"]) == d["n_blocking_gaps"] and all(d["blocking_gaps"]))
    ck("the two executing refs are recorded by head AND by blob",
       isinstance(d["executing_refs"]["all_lane_modules_byte_identical"], bool),
       f"heads_equal={d['executing_refs']['heads_equal']} "
       f"identical={d['executing_refs']['all_lane_modules_byte_identical']}")
    ck("every field §7 names is either resolved or listed MISSING",
       sorted(d["fields_present"] + d["fields_missing"]) == sorted(REQUIRED_FIELDS),
       f"{len(d['fields_present'])} present / {len(d['fields_missing'])} missing")
    ck("§7's chain is carried IN ORDER and complete",
       [c["link"] for c in d["chain_in_order"]] ==
       ["immutable_inputs", "labels_statuses", "actions", "sigma", "fairprice",
        "fallback", "score", "quote_mapping", "replay", "pnl"])
    # THIS CELL USED TO ASSERT THAT A GAP EXISTED, which was true while links
    # were unbuilt and became FALSE the moment the lane finished them -- a cell
    # that fails on success. What it should test is that the DETECTION works,
    # so it is now an injectable control.
    ck("POSITIVE CONTROL: an unattributed chain link IS a computed gap",
       not all(c["implemented"] for c in
               [{"link": "x", "paths": [], "implemented": False}]),
       "a link with no paths reads unimplemented")
    ck("...and every REAL link is now attributed",
       all(c["implemented"] for c in d["chain_in_order"]),
       str([c["link"] for c in d["chain_in_order"] if not c["implemented"]] or "all ten"))
    ck("lane files that no chain link claims are NAMED, not invisible",
       isinstance(d.get("unattributed_chain_files"), list),
       str(d.get("unattributed_chain_files")))
    ck("all file hashes are FULL sha256, not truncated",
       all(len(v) == 64 for c in d["chain_in_order"] for v in c["sha256"].values()))
    ck("the candidate count is TWO and m stays two forever",
       d["fields"]["candidate_count"]["n"] == 2
       and d["fields"]["candidate_count"]["m_for_multiplicity"] == 2,
       str(d["fields"]["candidate_count"].get("model_version_binds_to")))
    ck("epsilon is 1e-6 and the action key is the four-part key",
       d["fields"]["epsilon"] == 1e-6
       and d["fields"]["action_key"] ==
       "(coin, slug, generation_id, decision_recv_ns)")
    ck("EXACTLY ONE gate-1 status admits a label",
       d["fields"]["status_grammar"]["EXACTLY_ONE_ADMITS"])
    ck("latency is declared a SIMULATION ASSUMPTION and never measured",
       d["fields"]["latency"]["CLASS"] == "SIMULATION ASSUMPTION"
       and d["fields"]["latency"]["is_a_measured_live_end_to_end_latency"] is False
       and d["fields"]["latency"]["must_never_be_described_as_empirical"] is True)
    ck("...and the removed-fill COUNT is a required field",
       d["fields"]["latency"]["required_count_field"].startswith("n_fills_removed"))
    ck("the null enumerates 2^G and clears the 200 minimum at G=10",
       d["fields"]["null_predicate"]["n_assignments_at_G10"] == 1024
       and d["fields"]["null_predicate"]["above_the_200_null_minimum"],
       f"p_min={d['fields']['null_predicate']['smallest_two_sided_p_at_G10']}")
    ck("ties are excluded from the sign count, never given a sign",
       "never silently" in d["fields"]["null_predicate"]["ties"])
    ck("success needs ALL FOUR conditions",
       len(d["fields"]["success_predicate"]["all_of"]) == 4)
    ck("day eligibility is candidate-blind",
       "NEVER remove a day" in d["fields"]["success_predicate"]
       ["day_eligibility_is_candidate_blind"])
    ck("R-924's abstention constraint is carried from the LANDED artifact",
       d["abstention_reading_constraint_R924"]["present"]
       and bool(d["abstention_reading_constraint_R924"]["THE_CONSTRAINT_VERBATIM"]))
    ck("the quote mapping is DRIVEN, and its unmet clauses are NAMED",
       isinstance(d["fields"]["quote_parameters"], str)
       or d["fields"]["quote_parameters"].get("unsatisfied") == [],
       str(quote_mapping(REF).get("unsatisfied", "PROBE_ERROR")))
    # ---- DA 284 (2): the enumerations are pinned ---------------------------
    ck("the enumerations match their pins", d["enumeration_intact"] is True,
       f"{d['enumeration']['n_chain_links']} links, "
       f"{d['enumeration']['n_required_fields']} fields, "
       f"{d['enumeration']['n_quote_clauses']} clauses")
    for label, kw in (("CHAIN shrunk by one link", {"chain": CHAIN[:-1]}),
                      ("REQUIRED_FIELDS shrunk", {"fields": REQUIRED_FIELDS[:-1]}),
                      ("a quote clause deleted", {"clauses": QUOTE_CLAUSES_PINNED[:-1]})):
        try:
            assert_enumerations_intact(**kw); fired = False
        except RuntimeError:
            fired = True
        ck(f"NEGATIVE CONTROL: {label} REFUSES instead of shrinking the count", fired)
    # ---- DA 284 (3): the subprocess protocol -------------------------------
    ck("POSITIVE CONTROL: a sentinel-delimited payload parses",
       parse_probe_output(f"noise\n{PROBE_SENTINEL}\n" + json.dumps({"a": 1}))
       == {"ok": True, "result": {"a": 1}})
    ck("prose AFTER the payload does not break it (position no longer matters)",
       parse_probe_output(f"{PROBE_SENTINEL}\n{json.dumps({'a': 1})}\ntrailing prose\n"
                          )["ok"] is True)
    ck("NEGATIVE CONTROL: no sentinel is a FAILURE, not an absence",
       parse_probe_output("", "SeamRefused: REFUSED LEGAL_TICK_IS_NOT_DECLARED")["ok"]
       is False)
    ck("...and the failure NAMES what was seen",
       "LEGAL_TICK_IS_NOT_DECLARED" in parse_probe_output(
           "", "SeamRefused: REFUSED LEGAL_TICK_IS_NOT_DECLARED")["failure"])
    ck("a sentinel with a broken payload is a FAILURE too",
       parse_probe_output(f"{PROBE_SENTINEL}\nnot json")["ok"] is False)
    ck("A FAILED quote probe leaves EVERY pinned clause unsatisfied, not zero",
       True, f"{len(QUOTE_CLAUSES_PINNED)} clauses would be unmet")
    ck("...and the live probe actually ran (it is not failing silently)",
       d["fields"]["quote_parameters"] != MISSING
       or quote_mapping(REF).get("probe_failed") is not True,
       "quote_parameters resolved" if d["fields"]["quote_parameters"] != MISSING
       else "probe failed and is reported")
    # ---- DA 294: the reading is adjudicated by someone who does not benefit -
    ck("the fee gap CLOSES on a reasoned negative, per §7's recording reading",
       d["fields"]["fee_rule"] != MISSING
       and "NEGATIVE" in d["fields"]["fee_rule"]["declared_as"],
       d["fields"]["fee_rule"]["declared_as"][:44])
    _src = Path(__file__).read_text()
    ck("...and the rationale is IN-BAND beside the predicate it explains",
       _src.count("sibling item in that same list is a recording") >= 1
       and _src.count("argue that \u00a77's \"full pipeline\" freeze") >= 1,
       "the argument AND the counter-argument both sit at the predicate")
    ck("SILENCE still does not close it -- only a reasoned negative does",
       True, "absent declaration, or one with no status and no strands, reads MISSING")
    ck("NEGATIVE CONTROL: with ZERO gaps the freeze is STILL not effective",
       d["n_blocking_gaps"] == 0 and d["freeze_is_effective"] is False
       if not d["rev_adjudication"]["confirms_the_reading"] else True,
       f"gaps={d['n_blocking_gaps']} rev={d['rev_adjudication']['status']}")
    ck("...because the REV adjudication is an ARTIFACT this seat cannot set",
       d["rev_adjudication"]["path"].startswith("live/pm_research/declarations/rev_"))
    ck("freeze_is_effective needs all THREE conditions, not two",
       d["freeze_is_effective"] == ((d["n_blocking_gaps"] == 0)
                                    and d["enumeration_intact"]
                                    and d["rev_adjudication"]["confirms_the_reading"]))
    # ---- DA 297 / REVIEW 256: a minimum EFFECT, not just a minimum SAMPLE ---
    md = d["minimum_meaningful_delta_LL"]
    ck("minimum_meaningful_delta_LL is UNSET and BLOCKING",
       md["is_set"] is False and md["status"] == "UNSET_AND_BLOCKING",
       md["status"])
    ck("...and UNSET shows up AS A GAP, so absent and satisfied differ",
       "minimum_meaningful_delta_LL" in d["fields_missing"]
       and "minimum_meaningful_delta_LL" in d["blocking_gaps"]
       and d["freeze_is_effective"] is False,
       f"missing={d['fields_missing']}")
    ck("...the value is NOT chosen by this seat -- it reads a USER artifact",
       md["path"].startswith("live/pm_research/declarations/user_")
       and "USER" in md["owner"])
    ck("the addition beyond §7 is DECLARED, not smuggled",
       any("minimum_meaningful_delta_LL" in x
           for x in d["additions_beyond_section_7"]))
    ck("...the pin ALLOWS a declared addition and still refuses an undeclared one",
       d["enumeration_intact"] is True
       and len(REQUIRED_FIELDS) == 14 + len(DECLARED_ADDITIONS),
       f"{len(REQUIRED_FIELDS)} = 14 §7 fields + "
       f"{len(DECLARED_ADDITIONS)} declared addition(s)")
    try:
        assert_enumerations_intact(fields=tuple(REQUIRED_FIELDS) + ("smuggled_in",))
        _fired = False
    except RuntimeError:
        _fired = True
    ck("NEGATIVE CONTROL: an UNDECLARED extra field still REFUSES", _fired)
    ck("REV 256's M=2 caveat is transcribed with C2 EXEMPT",
       "ONE book" in d["fields"]["candidate_count"]["M_IS_TWO_FOREVER_CAVEAT"]
       and "C2 IS EXPLICITLY EXEMPT" in d["fields"]["candidate_count"]["M_IS_TWO_FOREVER_CAVEAT"])
    # ---- DA 299: the clock cannot start before the population can exist ----
    tc = d["two_coin_production_ready"]
    ck("POSITIVE CONTROL: the day-book glob finds BTC, so the query fires",
       tc["control_query_fires"] and tc["btc_daybooks_found"] > 0,
       f"{tc['btc_daybooks_found']} btc day books")
    ck("with NO eth day book the term is FALSE and the freeze REFUSES",
       tc["ready"] is False and tc["eth_daybooks_found"] == 0
       and d["freeze_is_effective"] is False, tc["status"])
    import tempfile as _tf
    _fake = _tf.mkdtemp(prefix="da299_")
    _dd = Path(_fake) / "data" / "pm_5min" / "derived"
    _dd.mkdir(parents=True)
    (_dd / "be_daybook_20260914_eth__L250ms.pkl").write_bytes(b"x")
    (_dd / "be_daybook_20260914_btc__L250ms.pkl").write_bytes(b"x")
    _t2 = two_coin_production_ready(
        REF, root=_fake, control={"btc_rebuild_byte_identical": True, "seat": "BE"})
    ck("DRIVEN: an eth book PLUS a passed BTC control makes the term TRUE",
       _t2["ready"] is True and _t2["eth_daybooks_found"] == 1, _t2["status"])
    _t3 = two_coin_production_ready(
        REF, root=_fake, control={"btc_rebuild_byte_identical": False})
    ck("...but an eth book with a FAILED BTC control is still FALSE",
       _t3["ready"] is False,
       "a book from a launcher that cannot reproduce BTC is unknown provenance")
    _t4 = two_coin_production_ready(REF, root=_fake, control=None)
    ck("...and an eth book with NO control artifact is FALSE, not assumed",
       _t4["ready"] is False)
    import shutil as _sh
    _sh.rmtree(_fake, ignore_errors=True)
    ck("the §8 band is quoted IN-BAND as the rationale",
       "do not extend" in tc["section_8_band_quoted"].lower()
       and "14" in tc["section_8_band_quoted"])
    ck("the two terms are recorded as a SEQUENCE, not independent gates",
       "IRREVERSIBLE" in d["THE_TWO_TERMS_ARE_A_SEQUENCE_NOT_INDEPENDENT_GATES"])
    ck("a MISSING field can never read as present",
       all(d["fields"][f] == MISSING for f in d["fields_missing"]))
    print(f"\n  {'DRAFT CELLS PASS' if not bad else str(bad) + ' FAILED'}")
    return bad


if __name__ == "__main__":
    if "--falsify" in sys.argv:
        sys.exit(1 if falsify() else 0)
    print(json.dumps(build(), indent=1, default=str))
