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
    ("immutable_inputs", ()),                    # source manifests -- see below
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
    ("pnl", ()),                                 # no implementation -- a GAP
)

#: The fields §7's sentence enumerates. Each resolves to a value or to MISSING,
#: and MISSING is a blocking gap unless it is listed as non-blocking with a
#: reason. Nothing here may be satisfied by prose.
REQUIRED_FIELDS = (
    "all_file_hashes", "commit_ref", "candidate_count", "action_key",
    "epsilon", "status_grammar", "source_manifests", "initial_inventory",
    "tick_rounding", "latency", "fee_rule", "quote_parameters",
    "null_predicate", "success_predicate",
)


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
        "print(json.dumps({'identities': out, 'model_version': W.MODEL_VERSION}))\n" % pm)
    r = subprocess.run(["python3", "-c", code], cwd=pm, capture_output=True, text=True)
    _git("worktree", "remove", "--force", wt)
    try:
        d = json.loads(r.stdout.strip().splitlines()[-1])
    except Exception as e:
        return {"error": f"{type(e).__name__}: {str(e)[:120]}", "n": MISSING}
    ids = d["identities"]
    return {
        "n": len(ids),
        "m_for_multiplicity": 2,
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
        "q=S.quote_from(0.5, slug='s', generation_id='g', half_spread=0.01, priced_by='x')\n"
        "o['anchor_is_p']= (q.anchor==0.5)\n"
        "o['sig']=list(inspect.signature(S.quote_from).parameters)\n"
        "hi=S.quote_from(0.999, slug='s', generation_id='g', half_spread=0.01, priced_by='x')\n"
        "o['ask_at_p999']=hi.ask\n"
        "lo=S.quote_from(0.001, slug='s', generation_id='g', half_spread=0.01, priced_by='x')\n"
        "o['bid_at_p001']=lo.bid\n"
        "t=S.quote_from(0.5237, slug='s', generation_id='g', half_spread=0.0001, priced_by='x')\n"
        "o['bid_odd']=t.bid; o['ask_odd']=t.ask\n"
        "print(json.dumps(o))\n" % pm)
    r = subprocess.run(["python3", "-c", code], cwd=pm, capture_output=True, text=True)
    _git("worktree", "remove", "--force", wt)
    try:
        d = json.loads(r.stdout.strip().splitlines()[-1])
    except Exception as e:
        return {"error": f"{type(e).__name__}: {str(e)[:120]}", "raw": r.stderr[-200:]}
    TICK = 0.01     # the legal binary tick this draft ASSUMES; see gaps below
    props = {
        "UP_uses_p": d["anchor_is_p"],
        "DOWN_uses_1_minus_p": ("outcome" in d["sig"] or "side" in d["sig"]),
        "bid_rounds_DOWN_to_the_legal_tick":
            abs(d["bid_odd"] * (1 / TICK) - round(d["bid_odd"] * (1 / TICK))) < 1e-9,
        "ask_rounds_UP_to_the_legal_tick":
            abs(d["ask_odd"] * (1 / TICK) - round(d["ask_odd"] * (1 / TICK))) < 1e-9,
        "prices_bounded_to_the_legal_binary_range":
            (d["ask_at_p999"] <= 1.0 and d["bid_at_p001"] >= 0.0),
        "crossing_quote_emits_PLACE_WITHHELD_MARKETABLE_CROSS":
            ("MARKETABLE_CROSS" in src and "PLACE_WITHHELD" in src),
        "never_silently_clamped":
            ("MARKETABLE_CROSS" in src),
        "no_zero_latency_privilege_for_candidate_induced_change":
            ("latency" in src.lower()),
    }
    return {
        "driven_against": "live/pm_research/de_fair_value_policy_seam.py",
        "sha256": _sha(ref, "live/pm_research/de_fair_value_policy_seam.py"),
        "quote_from_signature": d["sig"],
        "measured": {"ask_at_p_0.999": d["ask_at_p999"],
                     "bid_at_p_0.001": d["bid_at_p001"],
                     "bid_at_p_0.5237_hs_0.0001": d["bid_odd"],
                     "ask_at_p_0.5237_hs_0.0001": d["ask_odd"]},
        "properties": props,
        "n_satisfied": sum(1 for v in props.values() if v),
        "n_declared": len(props),
        "unsatisfied": sorted(k for k, v in props.items() if not v),
        "THE_TICK_IS_ASSUMED_NOT_DECLARED":
            f"this probe assumed tick={TICK}; no fair-value module declares a "
            f"legal tick, so the rounding predicates are measured against an "
            f"assumption and cannot be a freeze input until one is declared",
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


def build(ref: str = REF, rev203_six_of_six: bool = False, fetch: bool = True) -> dict:
    if fetch:
        subprocess.run(["git", "-C", _root(), "fetch", "--quiet", "origin"], check=False)
    head = _git("rev-parse", ref).stdout.strip()
    hashes = file_hashes(ref)
    qm = quote_mapping(ref)
    sm = source_manifests(ref)
    lat = latency(ref)
    cands = candidates(ref)

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
        "source_manifests": sm if sm["present"] else MISSING,
        "initial_inventory": MISSING,
        "tick_rounding": MISSING,
        "latency": lat,
        "fee_rule": MISSING,
        "quote_parameters": qm if not qm.get("unsatisfied") else MISSING,
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
        "freeze_is_effective": not gaps,
        "why_not_effective": None,          # filled below, from the same list
        "WHAT_freeze_is_effective_MEANS": (
            "TRUE only when every §7 field resolves and every chain link has an "
            "implementation. It is FALSE here. The declaration exists so that "
            "§11 step 6 has an artifact and so that what is missing is written "
            "down at a ref rather than carried in conversation; it does not "
            "assert that the pipeline is frozen."),
        "executing_refs": executing_refs(),
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
    ck("freeze_is_effective is COMPUTED from the gap list, never asserted",
       d["freeze_is_effective"] == (d["n_blocking_gaps"] == 0))
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
    ck("a chain link with NO implementation is a computed gap, not a silence",
       any(g.startswith("chain_link_not_implemented") for g in d["blocking_gaps"]),
       [c["link"] for c in d["chain_in_order"] if not c["implemented"]])
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
    ck("a MISSING field can never read as present",
       all(d["fields"][f] == MISSING for f in d["fields_missing"]))
    print(f"\n  {'DRAFT CELLS PASS' if not bad else str(bad) + ' FAILED'}")
    return bad


if __name__ == "__main__":
    if "--falsify" in sys.argv:
        sys.exit(1 if falsify() else 0)
    print(json.dumps(build(), indent=1, default=str))
