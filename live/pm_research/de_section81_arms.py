"""§8.1 integration ablation -- the runnable arms, on the real
QR_SKEW_ONLY opportunity population, with real predictors.

Landed from the scratch harness that produced Q-DE-70 so the
numbers have a committed producer. The archive-root injection at
the top is the two-root property (round 45): `phase2_arms.DERIVED`
is hardcoded to the fit tree and `flow_intensity.PM` is
`__file__`-relative, so a run from a worktree must point them at
one tree or select zero windows.
"""
"""§8.1 arms on the real population with REAL PREDICTORS and LEGIBLE ARM
IDENTITY.

Round 52's run was refused as arm results: arms 2 and 5 were the same
computation, and every arm used `_pricing_scorer` -- a declared synthetic.
Here each arm records WHICH predictor it loaded, by artifact and sha, so
"is this arm real or a stub" is answered by evidence in the emission
rather than by trusting the name."""
import ast as _ast
import hashlib, json, resource, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import flow_intensity as fi
fi.PM = Path("/home/yuqing/ctaNew/data/pm_5min"); fi.RAW = fi.PM / "raw"
fi.MARKETS = fi.PM / "markets.jsonl"; fi.GAPS = fi.PM / "collector_gaps.jsonl"
fi.DAYS = fi._discover_days()
import de_phase4_diag_runner as R
import de_lane4_real_parity as L
import harmful_stateful_policy as HSP
import de_rho_estimator as RHO
import de_matched_random_control as MRC
import de_score_stream as SS
import harmful_hazard_model as hm

def gb(): return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20, 2)


#: THE CONTROL PREDICATES, named once. `VALID_AS_A_CONTROL` is DERIVED
#: from these and is never a literal: round 53 emitted the flag as a
#: hardcoded True beside booleans that said otherwise, and a note beside
#: it asserting the opposite of its own fields. A verdict a reader can
#: disagree with its own data is worse than no verdict (rule 10).
CONTROL_PREDICATES = ("P1_key_multisets_equal",
                      "P2_stratum_score_multisets_equal",
                      "P3_drawn_carry_above_and_only_drawn",
                      "P4_realised_action_counts_equal")


#: THE FLOOR IS UNAVAILABLE, AND WHY IT CANNOT YET BE DIAGNOSED.
#:
#: DA REFUSED TO CONFIRM THE STRUCTURAL CLAIM FROM THIS ARTIFACT, and DA
#: is right. What the emission carried was a COUNT -- attempts 20,
#: rejections {P4: 20} -- and a count cannot separate the two readings
#: that decide the question:
#:
#:   (a) the ACHIEVABLE SET of realised counts EXCLUDES the target, in
#:       which case no budget ever helps and the floor is unavailable BY
#:       CONSTRUCTION; or
#:   (b) it BRACKETS the target and twenty draws missed, in which case
#:       the refusal is a BUDGET FACT.
#:
#: The 333-vs-496 figures that were carried as decisive came from a
#: SINGLE SEED recorded in a source comment: one-directionality rested on
#: n=1, not on emitted evidence.
#:
#: SO THE REASON IS NOT WRITTEN AS SETTLED. The grade is DA's:
#: REFUSAL_LOCALISED_BUT_NOT_DIAGNOSABLE -- the refusal is localised to
#: P4 (P1-P3 pass on every draw) and the mechanism is not yet
#: distinguishable from a budget miss.
#:
#: WHAT WOULD SETTLE IT, and it is emitted below: for EVERY draw that
#: reaches P4, the REALISED VALUE, the SIGNED GAP against the treated
#: arm, and the PER-STRATUM BREAKDOWN. P4 is a per-stratum equality, so a
#: near-miss on ONE stratum is a different fact from a miss on ALL of
#: them, and only the signed gap shows whether the misses are
#: one-directional or scattered.
#:
#: THE HONEST LIMIT, kept in the emission: even a clean answer would
#: evidence UNREACHABLE FOR THIS CONSTRUCTION, never NEVER.
#:
#: WHAT IS ESTABLISHED, and is not in question: DE36-R3 showed cancel-set
#: IDENTITY cannot be matched, and its remedy was to match the
#: per-stratum REALISED COUNT instead. This round finds that remedy's own
#: criterion failing too. That is a step BEYOND DE36-R3, not a
#: restatement -- the finding was not previously recorded. The DE36-R3
#: sentence sits inside the assertion message of P4's own positive
#: control (`de_phase4_diag_runner.py:4322-4326`).
MATCHED_FLOOR_STATE = {
    "floor": "UNAVAILABLE",
    "grade": "REFUSAL_LOCALISED_BUT_NOT_DIAGNOSABLE",
    "localised_to": "P4_realised_action_counts_equal",
    "not_yet_distinguishable": [
        "achievable set EXCLUDES the target (unavailable by construction)",
        "achievable set BRACKETS the target and the budget missed it"],
    "why_a_count_cannot_decide": "attempts and a rejection tally carry no "
                                 "realised VALUES, so neither reading can "
                                 "be excluded",
    "one_directionality_evidence": "n=1, from a single seed in a source "
                                   "comment -- NOT emitted evidence, and "
                                   "not sufficient",
    "what_would_settle_it": "per-draw realised value, signed gap vs the "
                            "treated arm, and per-stratum breakdown, over "
                            "several seeds",
    "honest_limit": "even a clean answer evidences UNREACHABLE FOR THIS "
                    "CONSTRUCTION, never NEVER",
    "established_and_not_in_question": "DE36-R3 -- cancel-set IDENTITY "
                                       "cannot be matched; its remedy was "
                                       "the per-stratum realised count, "
                                       "whose own criterion this round "
                                       "finds failing. A step BEYOND "
                                       "DE36-R3, not a restatement",
}


def control_is_valid(predicates: dict) -> bool:
    """A control is valid when EVERY declared predicate is True.

    A None (undecided) predicate is NOT True: P4 reads null until the
    control has been replayed, and a control whose match was never
    checked is not a matched control."""
    return all(predicates.get(k) is True for k in CONTROL_PREDICATES)


#: The protocol name DA's `provenance` resolves the producer from: it
#: strips a trailing _vN and appends .py, so this must name this file.
PROTOCOL = "de_section81_arms_v1"

#: Every file whose bytes decide what these numbers are. Recomputed by
#: the instrument from the tree; a DIFFER means the artifact was made by
#: a program that is no longer there.
IDENTITY_FILES = ("de_section81_arms.py", "de_phase4_diag_runner.py",
                  "harmful_stateful_policy.py", "de_matched_random_control.py",
                  "de_rho_estimator.py", "de_score_stream.py",
                  "de_lane4_real_parity.py")


def code_identity() -> dict:
    """`{file: sha16}` in the instrument's own scheme."""
    d = Path(__file__).resolve().parent
    return {f: hashlib.sha256((d / f).read_bytes()).hexdigest()[:16]
            for f in IDENTITY_FILES if (d / f).is_file()}


def emit(doc: dict, dst: Path) -> Path:
    """THE EMITTING ENTRY POINT -- a TOP-LEVEL function that serialises.

    DA's `_emitting_entry_points` finds the emitter BY SHAPE (`json.dump`
    / `write_text`) among top-level functions, then censuses its call
    sites. This module wrote its artifact at module level, so the census
    found no entry point at all and could not ask the question it exists
    to ask. The write lives here and is called from the run body, which
    is a PRODUCTION call site and not a selftest one."""
    dst = Path(dst)
    dst.write_text(json.dumps(doc, indent=1, sort_keys=True, default=str))
    return dst


def provenance() -> dict:
    """WHO PRODUCED THIS EMISSION -- so a provenance census finds it.

    Round 53's arm results lived in a 14 kB scratchpad file naming no
    producer: no `produced_by`, no `producing_code`, no
    `carrying_commit`. DA's census over committed modules returned NONE
    for every field the filing quoted, and the only hit was a CONSUMER.
    An emission that cannot say what made it is not a result."""
    import subprocess
    me = Path(__file__).resolve()
    def _git(*a):
        try:
            r = subprocess.run(("git",) + a, cwd=str(me.parents[2]),
                               capture_output=True, text=True, timeout=20)
            return r.stdout.strip() if r.returncode == 0 else None
        except Exception:                            # noqa: BLE001
            return None
    return {
        "protocol": PROTOCOL,
        "code_identity": code_identity(),
        "produced_by": me.name,
        "producing_code": hashlib.sha256(me.read_bytes()).hexdigest()[:16],
        "producing_code_path": str(me),
        "carrying_commit": _git("rev-parse", "HEAD"),
        "carrying_commit_short": _git("rev-parse", "--short", "HEAD"),
        "working_tree_clean_for_this_file": (
            _git("status", "--porcelain", "--", str(me)) == ""),
        "spec": "§8.1 integration ablation",
    }
if "--selftest" in sys.argv:
    pass
EXPECTED_CHECKS = 13


def selftest() -> int:
    """Falsifiers for the two things round 53 got wrong: a validity flag
    that was a literal, and an emission that named no producer."""
    n = [0]

    def ok(cond, label):
        if not cond:
            raise SystemExit(f"[de_section81_arms] FAIL: {label}")
        n[0] += 1
        print(f"  PASS  {label}")

    allp = {k: True for k in CONTROL_PREDICATES}
    ok(control_is_valid(allp) is True,
       f"POSITIVE CONTROL: every declared predicate True -> the control "
       f"is VALID ({list(CONTROL_PREDICATES)}). The admitting direction, "
       f"which a refusal-only check never proves")
    for k in CONTROL_PREDICATES:
        bad = dict(allp, **{k: False})
        ok(control_is_valid(bad) is False,
           f"KNOWN-BAD: {k} False -> the control is NOT valid. Each "
           f"predicate is load-bearing on its own; round 53 emitted "
           f"VALID_AS_A_CONTROL as a hardcoded True beside booleans that "
           f"said otherwise")
    ok(control_is_valid(dict(allp, P4_realised_action_counts_equal=None))
       is False,
       "and a NULL predicate is not True: P4 reads null until the control "
       "has been replayed, and a control whose match was never checked is "
       "not a matched control")
    ok(control_is_valid({}) is False,
       "and an EMPTY predicate set is not valid -- a refused draw has no "
       "predicates and must not inherit a passing flag")
    _mfu = MATCHED_FLOOR_STATE
    ok(_mfu["floor"] == "UNAVAILABLE"
       and _mfu["grade"] == "REFUSAL_LOCALISED_BUT_NOT_DIAGNOSABLE"
       and len(_mfu["not_yet_distinguishable"]) == 2
       and "n=1" in _mfu["one_directionality_evidence"]
       and "never NEVER" in _mfu["honest_limit"]
       and "step BEYOND" in _mfu["established_and_not_in_question"],
       f"THE FLOOR IS UNAVAILABLE AND THE REASON IS NOT WRITTEN AS "
       f"SETTLED: grade {_mfu['grade']}, localised to "
       f"{_mfu['localised_to']}, with TWO readings still "
       f"indistinguishable ({len(_mfu['not_yet_distinguishable'])}). A "
       f"COUNT cannot separate them, and the one-directionality that was "
       f"carried as decisive rests on {_mfu['one_directionality_evidence'][:24]}. "
       f"What IS established stands: {_mfu['established_and_not_in_question'][:70]}")
    _fa = Path(__file__).read_text()
    _key = "floor_" + "available"
    ok(f'"{_key}": True' not in _fa and f'"{_key}": False' not in _fa,
       f"and `{_key}` is DERIVED from the predicates, never a literal -- "
       f"read from this file's own source, the same guard the validity "
       f"flag carries")
    pr = provenance()
    ok(pr["produced_by"] == "de_section81_arms.py"
       and len(pr["producing_code"]) == 16
       and pr["carrying_commit"] is not None,
       f"DE53-R2: the emission NAMES ITS PRODUCER -- produced_by "
       f"{pr['produced_by']}, producing_code {pr['producing_code']}, "
       f"carrying_commit {pr['carrying_commit_short']}. Round 53's arm "
       f"results lived in a scratchpad file naming none of these, so a "
       f"provenance census over committed modules returned NOTHING for "
       f"every field the filing quoted")
    _src = Path(__file__).read_text()
    # DE55-R1: THE FIRST VERSION WAS A TWO-SPELLING SUBSTRING and four
    # other forms passed it -- `d["VALID_AS_A_CONTROL"] = True`, a dict
    # built with `dict(...)`, single quotes, whitespace variants. A needle
    # is a guess about syntax; the PARSE is the syntax. This walks the
    # AST for ANY constant True/False bound to the flag, by any form:
    # dict literal key, subscript assignment, or keyword argument.
    # (The name is still BUILT rather than spelt, so the check cannot
    # match itself -- a falsifier that cannot pass is as useless as one
    # that cannot fail.)
    _flag = "VALID_AS" + "_A_CONTROL"
    _bad_nodes = []
    for _n in _ast.walk(_ast.parse(_src)):
        if isinstance(_n, _ast.Dict):
            for _k, _v in zip(_n.keys, _n.values):
                if (isinstance(_k, _ast.Constant) and _k.value == _flag
                        and isinstance(_v, _ast.Constant)
                        and isinstance(_v.value, bool)):
                    _bad_nodes.append(("dict-literal", _n.lineno))
        elif isinstance(_n, _ast.Assign) and isinstance(
                _n.value, _ast.Constant) and isinstance(_n.value.value, bool):
            for _t in _n.targets:
                if (isinstance(_t, _ast.Subscript)
                        and isinstance(_t.slice, _ast.Constant)
                        and _t.slice.value == _flag):
                    _bad_nodes.append(("subscript-assign", _n.lineno))
        elif isinstance(_n, _ast.keyword) and _n.arg == _flag and isinstance(
                _n.value, _ast.Constant) and isinstance(_n.value.value, bool):
            _bad_nodes.append(("keyword-arg", _n.lineno))
    ok(not _bad_nodes,
       f"DE55-R1 CLOSED: the validity flag is never a literal by ANY "
       f"syntactic form -- dict-literal key, subscript assignment or "
       f"keyword argument -- checked against the PARSE rather than a "
       f"substring ({_bad_nodes} found). The first version was a "
       f"two-spelling needle and `d[flag] = True` walked straight past "
       f"it: a needle is a guess about syntax, and the parse IS the "
       f"syntax")
    _me = Path(__file__).read_text()
    _tree = _ast.parse(_me)
    # `sys.path.insert` is the import shim every module in this package
    # carries and is not work; anything ELSE calling at module scope is.
    _toplevel_calls = [
        _ast.unparse(_n) for _n in _tree.body
        if isinstance(_n, _ast.Expr) and isinstance(_n.value, _ast.Call)
        and not _ast.unparse(_n).startswith("sys.path.insert")]
    _guarded = any(
        isinstance(_n, _ast.If) and _ast.unparse(_n.test).startswith(
            "__name__ ==") for _n in _tree.body)
    # AND THE EMIT DEFAULT IS CHECKED AT THE PARSE, NOT AS A SUBSTRING --
    # the third time today a needle of mine matched its own explanatory
    # prose. The docstring below MENTIONS `Path(__file__).parent` to say
    # it was the defect; a substring check reads that as the defect.
    _scratch_defaults = [
        _ast.unparse(_n.value) for _n in _ast.walk(_tree)
        if isinstance(_n, _ast.Assign)
        and any(getattr(_t, "id", "") == "SCRATCH" for _t in _n.targets)]
    _emits_beside_source = any("__file__).parent" in d.replace(" ", "")
                               for d in _scratch_defaults)
    ok(_guarded and not _toplevel_calls and not _emits_beside_source
       and any("derived" in d for d in _scratch_defaults),
       f"DE57: THE RUN IS BEHIND A `__main__` GUARD and no bare call runs "
       f"at module scope ({len(_toplevel_calls)} found). Until round 57 "
       f"this body ran 60 SEEDS ON IMPORT and emitted beside its own "
       f"source, because SCRATCH defaulted to the source directory -- "
       f"where `live/pm_research/*.json` is not gitignored and nothing "
       f"ever committed it. A module that works on import and writes into "
       f"its own directory is why its artifacts keep vanishing. The "
       f"emit default is read from the ASSIGNMENT "
       f"({[d[:60] for d in _scratch_defaults]}), not from a substring "
       f"that the docstring explaining the fix would itself match")
    ok(n[0] + 1 == EXPECTED_CHECKS,
       f"check count asserted at run time: {n[0] + 1} == {EXPECTED_CHECKS}")
    print(f"[de_section81_arms] selftest OK -- {n[0]} checks")
    return 0


if "--selftest" in sys.argv:
    raise SystemExit(selftest())


def run_arms(argv=None):
    """THE RUN, behind a `__main__` guard.

    Until round 57 this body executed AT MODULE SCOPE: importing the
    module ran 60 seeds and wrote an artifact, and `SCRATCH` defaulted to
    `Path(__file__).parent` -- so an import from anywhere emitted a file
    beside the source, where `live/pm_research/*.json` is not gitignored
    and nothing ever committed it. A module that does heavy work on
    import and emits into its own source directory is why its artifacts
    keep vanishing. Both halves are fixed here: the work is behind a
    guard, and the emission defaults to the derived directory DA reads."""
    global SCRATCH, OUT, RUN_ID, PROV
    sys.argv = list(argv) if argv is not None else sys.argv
    COIN = "btc"; LIMIT = (int(sys.argv[1])
                           if len(sys.argv) > 1 and sys.argv[1].isdigit() else 12)
    BUDGET, LAT, NO_CANCEL_THETA = 0.10, 250, 2.0
    T0 = time.time()

    import pickle
    # NEVER `Path(__file__).parent`: artifacts beside the source are
    # transient and were never committed. `data/pm_5min/derived/` is
    # where DA reads and where the artifact is committed from.
    SCRATCH = (Path(sys.argv[2]) if len(sys.argv) > 2
               else Path(__file__).resolve().parents[2]
               / "data/pm_5min/derived")
    # DE59: VERSIONED. The cached `fr` gained `terminal_marks`, so a v1
    # cache would silently feed an arms run a reference with no
    # window-end mark and every fill would read NO_TERMINAL_MARK. A
    # schema change that reuses its predecessor's filename is a
    # stale-cache bug that looks like a measurement.
    CACHE = SCRATCH / f"de_section81_cache_v2_{LIMIT}.pkl"
    if CACHE.exists():
        _c = pickle.loads(CACHE.read_bytes())
        fr, asm = _c["fr"], _c["asm"]
        print(json.dumps({"cache": "HIT", "path": str(CACHE)}), flush=True)
    else:
        _c = None
    fr = fr if _c else R.build_reference(COIN, limit=LIMIT)
    ref = fr["reference"]
    POP = {"coin": COIN, "windows": len(ref),
           "generations": sum(len(s[x]) for s in ref.values() for x in HSP.SIDES),
           "statuses": fr["statuses"], "feed_wall_s": round(time.time()-T0, 1),
           "as_of": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    print(json.dumps({"population": POP}), flush=True)

    # ---- REAL head scores for THIS population -------------------------------
    splits = R.DECLARED_SPLIT_SETS[R.RULED_SPLIT_SET]
    if _c is None:
      t = time.time(); tape = R.build_tape_index(splits)
      print(json.dumps({"tape_index_s": round(time.time()-t,1),
                        "rows": tape["n_tape_rows"], "peak_gb": gb()}), flush=True)
      frag = SCRATCH / "de_section81_frag.json"
      R.fragment_slice(frag, n_windows=LIMIT, only_slugs=list(ref))
      t = time.time()
      asm = R.assemble_streaming({COIN: ref}, splits=splits, coins=(COIN,),
                                 chunk_windows=6, source=frag, tape=tape)
      print(json.dumps({"assembly_s": round(time.time()-t,1),
                        "kept": asm["assembly"]["kept_by_coin"],
                        "n_scores": {f"{k[1]}": len(v[0])
                                     for k, v in asm["by_arm"].items()},
                        "peak_gb": gb()}), flush=True)
      CACHE.write_bytes(pickle.dumps({"fr": fr, "asm": asm}))

    def stream(head):
        """The event stream for one head, through the manifest-bound adapter.

        THE POPULATION IS FILTERED TO GENERATIONS THE ASSEMBLY SCORED. The
        scorer refuses a generation whose rows the feature pass dropped --
        correctly, since scoring it would be scoring from nothing -- so the
        exclusion happens HERE, before scoring, and is COUNTED (rule 4)."""
        gen_scores = asm["by_arm"][(COIN, head)][0]
        rows, dropped = [], 0
        for s_, sides in sorted(ref.items()):
            for sd in HSP.SIDES:
                for g in sides[sd]:
                    if (s_, sd, float(g["t0"])) in gen_scores:
                        rows.append({"t": g["t0"], "slug": s_, "side": sd,
                                     "gen": g["gen"]})
                    else:
                        dropped += 1
        EXCL[head] = {"scored_generations": len(rows),
                      "excluded_no_assembled_score": dropped,
                      "reference_generations": len(rows) + dropped,
                      "excluded_fraction": round(dropped/max(len(rows)+dropped,1), 4),
                      "reason": "the feature pass dropped every row of these "
                                "generations; scoring them would be scoring "
                                "from nothing (rule 4: counted, never dropped)"}
        v = SS.verify_head(head, COIN)
        return SS.score_events(rows, head=head, coin=COIN,
                               scorer=R._head_scorer(head, COIN, gen_scores),
                               verified=v), v

    EXCL = {}


    def ident(head, verified):
        """ARM IDENTITY, as evidence rather than a name: which predictor was
        loaded, by artifact and sha."""
        return {"predictor": head, "artifacts": verified,
                "n_artifacts": len(verified),
                "source": "de_score_stream.verify_head -> the fit manifest"}

    def fields(res, rho_out, mp_arm, inv_arm):
        e, c = res["economics"], res["counters"]
        inv = e.get("inventory", {}) or {}
        out = {}
        for f, spec in L.SECTION_8_1_FIELDS.items():
            src = spec.get("source")
            if src is None:
                out[f] = {"status": "NOT_AVAILABLE", "reason": spec["why"]}
                # DE58: an UNRULED field must carry what the ruling is
                # BETWEEN, or the reader is told only that it is missing.
                for k in ("candidates", "also_unruled"):
                    if spec.get(k) is not None:
                        out[f][k] = spec[k]
            elif src == "de_phase4_diag_runner.inventory_pnl":
                # DE59: BOTH RULINGS EMITTED, NEITHER CHOSEN. The status
                # is neither OK nor NOT_AVAILABLE, because a single value
                # cannot honestly be quoted until the ruling lands -- and
                # a field that reports one reading while another is live
                # is the field a reader would cite without knowing what
                # it excluded.
                out[f] = {
                    # RULED DE59: (C'). The held-forward mark is the
                    # PRIMARY reading; the gap refusal is a SECOND
                    # COMPUTABLE VIEW, not a branch. The status is OK
                    # because a value can now be quoted -- and it names
                    # its ruling, because a value quoted without one is
                    # a value whose exclusion rule nobody declared.
                    "status": "OK",
                    "value": inv_arm["primary_inventory_loss_cents"],
                    "ruling": inv_arm["primary_ruling"],
                    "ruling_provenance": inv_arm["ruling_provenance"],
                    "terminal_indexes": inv_arm["terminal_indexes"],
                    "second_view_disagreement_cents":
                        inv_arm["second_view_disagreement_cents"],
                    "second_view_disagreement_share":
                        inv_arm["second_view_disagreement_share"],
                    "views_disagree_materially":
                        inv_arm["views_disagree_materially"],
                    "concentration": inv_arm["concentration"],
                    "interval": inv_arm["interval"],
                    "rulings_available": list(spec["ruling_required"]),
                    "by_ruling": {
                        k: {kk: vv for kk, vv in v.items()
                            if kk != "ruling_text"}
                        for k, v in inv_arm["by_ruling"].items()},
                    "ruling_texts": {k: v["ruling_text"] for k, v
                                     in inv_arm["by_ruling"].items()},
                    "unit": spec["unit"],
                    "per_slug": inv_arm["per_slug"],
                    "summed_terminal_net_shares":
                        inv_arm["summed_terminal_net_shares"],
                    "summed_terminal_net_shares_status":
                        inv_arm["summed_terminal_net_shares_status"],
                    "fills_leg_cents": inv_arm["fills_leg_cents"],
                    "total_to_terminal_cents":
                        inv_arm["total_to_terminal_cents"],
                    "identity": spec["identity"],
                    "identity_holds": inv_arm["identity_holds"],
                    "identity_residual_cents":
                        inv_arm["identity_residual_cents"],
                    "fill_statuses": inv_arm["fill_statuses"],
                    "found_in": "de_phase4_diag_runner.inventory_pnl"}
            elif src.startswith("de_phase4_diag_runner.maker_pnl_from_fills."):
                # DE58: the per-arm maker-P&L decomposition, over the
                # fills THIS ARM received. `maker_pnl(reference)` would
                # be the same number for every arm -- the baseline under
                # four names -- so the per-arm construction is used and
                # the leg it was computed over travels with it.
                leaf = src.rsplit(".", 1)[1]
                out[f] = {"status": "OK", "value": mp_arm[leaf],
                          "found_in": "de_phase4_diag_runner."
                                      "maker_pnl_from_fills",
                          "n_fills": mp_arm["decomposition_n_fills"],
                          "shares": mp_arm["decomposition_shares"],
                          "fill_statuses": mp_arm["fill_statuses"],
                          "identity_holds": mp_arm["identity_holds"],
                          "identity_residual_cents":
                              mp_arm["identity_residual_cents"]}
                for extra in spec.get("also", ()):
                    if extra.startswith("de_phase4_diag_runner."):
                        out[f].setdefault("also", {})[
                            extra.rsplit(".", 1)[1]] = mp_arm[
                                extra.rsplit(".", 1)[1]]
                if spec.get("equals"):
                    out[f]["equals"] = spec["equals"]
            elif src == "de_rho_estimator.rho":
                out[f] = {"status": "OK", "value": rho_out["rho"],
                          "statuses": rho_out["statuses"]}
            elif src.startswith("the caller"):
                out[f] = {"status": "NOT_RUN_THIS_ROUND",
                          "reason": "needs the 9-rung latency axis; one rung run"}
            else:
                leaf = src.split(".")[-1]
                # ROUND 52 LOOKED IN THE WRONG CONTAINER for four fields.
                # Search all three and RECORD WHICH ONE ANSWERED, so a value
                # carries where it came from.
                for cname, blk in (("holds", res.get("holds") or {}),
                                   ("inventory", res.get("inventory") or {}),
                                   ("economics.inventory", inv),
                                   ("counters", c), ("economics", e),
                                   ("result", res)):
                    if leaf in blk:
                        out[f] = {"status": "OK", "value": blk[leaf],
                                  "found_in": cname}
                        break
                else:
                    out[f] = {"status": "MISSING_AT_RUNTIME",
                              "reason": f"{leaf!r} absent from economics, "
                                        f"counters, economics.inventory and "
                                        f"the result",
                              "searched": ["economics.inventory", "counters",
                                           "economics", "result"]}
        return out

    # DE59-C3: an arm's received fills, kept so the CROSS-ARM set
    # differences can be taken. "Which fills did this arm decline" is a
    # set difference against the baseline and cannot be recovered from
    # any arm's own summary.
    FILLS = {}
    ARM_FILLS = {}

    def replay(scores, *, cancel, theta):
        params = R.cell_params(
            {"coin": COIN, "latency_ms": LAT, "budget": BUDGET,
             "enable_reduce": False,
             "charge_reset_cost_at_generation_start": False},
            theta_cancel=(theta if cancel else NO_CANCEL_THETA),
            protection_mode=HSP.PROTECTION_MODES[0],
            repost_fill_model=HSP.REPOST_FILL_MODELS[0])
        res = HSP.replay_policy(ref, scores, params)
        f = R.received_fills(res, ref, R._decision_times(scores))
        FILLS[id(res)] = f
        return (res, RHO.rho(f, LAT,
                             proxy={"rho_captured_over_sacrificed": None}),
                R.maker_pnl_from_fills(f),
                R.inventory_pnl(f, fr.get("terminal_marks") or {}))

    PROV = provenance()
    RUN_ID = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    OUT = {"protocol": PROTOCOL, "code_identity": code_identity(),
           "provenance": PROV, "run_id": RUN_ID,
           "population": POP, "arms": {}, "spec": "§8.1", "coin": COIN,
           "latency_ms": LAT, "budget": BUDGET,
           "evidence_class": "DEVELOPMENT_EVIDENCE_NOT_A_VALIDATION"}

    CV = "q1_arrival_composed_lgbm"; HZ = "incumbent_linear_d"
    cv_ev, cv_v = stream(CV)
    hz_ev, hz_v = stream(HZ)
    theta_cv = R.theta_for(COIN, CV, BUDGET)
    theta_hz = R.theta_for(COIN, HZ, BUDGET)
    _mx = max((float(e["score"]) for e in cv_ev), default=0.0)
    assert _mx < NO_CANCEL_THETA, f"arm 1 would cancel: max score {_mx}"

    def add(name, res, rho_out, mp_arm, inv_arm, identity, **extra):
        ARM_FILLS[name] = FILLS.get(id(res), [])
        OUT["arms"][name] = dict(
            {"arm": name, "n_cancels": res["counters"].get("cancels_issued", 0),
             "identity": identity,
             "fields": fields(res, rho_out, mp_arm, inv_arm),
             # DE58: the reconciliation, RUN AGAINST THIS REAL REPLAY and
             # not against a fixture. It is DIRECTIONAL -- |replay| <=
             # |reference| -- and degenerates to an EQUALITY on the
             # no-cancel arm, which is the strongest form it takes.
             "maker_pnl_reconciliation": R.reconcile_maker_pnl(
                 mp_arm, res)}, **extra)
        print(json.dumps({"arm": name,
                          "n_cancels": res["counters"].get("cancels_issued", 0),
                          "predictor": identity["predictor"],
                          "rho": rho_out["rho"]}), flush=True)

    r1, h1, m1, i1 = replay(cv_ev, cancel=False, theta=theta_cv)
    add("QR_SKEW_ONLY", r1, h1, m1, i1,
        {"predictor": "NONE", "artifacts": {},
         "note": "the neutral opportunity population; no cancellation, "
                 "asserted n_cancels==0"})
    assert OUT["arms"]["QR_SKEW_ONLY"]["n_cancels"] == 0

    r5, h5, m5, i5 = replay(cv_ev, cancel=True, theta=theta_cv)
    add("CONDVALUE_X_SKEW", r5, h5, m5, i5, ident(CV, cv_v),
        semantics=L.ARM_X_SKEW_SEMANTICS, theta=theta_cv)

    rH, hH, mH, iH = replay(hz_ev, cancel=True, theta=theta_hz)
    add("HAZARD_OVER_SKEWED_REF", rH, hH, mH, iH, ident(HZ, hz_v),
        note="NOT §8.1 arm 3 -- arm 3 requires NEUTRAL placement, which is "
             "ABSENT. This is the hazard head over the SKEWED reference and "
             "is named so it cannot be read as arm 3", theta=theta_hz)

    gidx = R._gen_index(ref)
    # THE DEMAND IS OVER ABOVE-THRESHOLD EVENTS, NEVER OVER REALISED ACTIONS.
    # I built it from CANCEL_ISSUED trajectory entries -- i.e. actions -- and
    # that is EXACTLY round 37's defect, which DE37-R2 closed inside
    # `run_cell` and marked as the falsifier's input (`_known_bad_demand`,
    # "no run passes it"). The guard protects `run_cell`'s path; this script
    # bypasses `run_cell` and so reintroduced the known-bad from outside it.
    #
    # WHY IT BREAKS P2 AND P3, in the code's own words: "Every action is an
    # above event and not conversely -- the policy is stateful, a HELD side
    # suppresses later crossings -- so in any stratum with a non-acting above
    # event the demand was too small, `permuted_stream` returned ok=False
    # with a TRUNCATED-ZIP stream (above values dropped, below values
    # duplicated)". Dropped above values are why the stratum score multisets
    # differ (P2) and why the drawn set does not carry exactly the above
    # values (P3).
    treated = [{"slug": f"{e['slug']}|{e['side']}|{e['gen']}"}
               for e in cv_ev if float(e["score"]) >= theta_cv]
    _actions = sum(1 for c in r5["trajectory"] if c.get("kind") == "CANCEL_ISSUED")
    DEMAND = {"above_threshold_events": len(treated),
              "realised_actions": _actions,
              "non_acting_above_events": len(treated) - _actions,
              "why": "the demand is over ABOVE events; the ACTION count keeps "
                     "its own job at P4, matched AFTER the replay (DE37-R2)"}
    print(json.dumps({"demand": DEMAND}), flush=True)
    pbk = {}
    for e in cv_ev:
        k = f"{e['slug']}|{e['side']}|{e['gen']}"
        pbk[k] = {"slug": k, "side": e["side"], "hour": R._hour_of(e["slug"])}
    pool = [pbk[k] for k in sorted(pbk)]
    # P4 IS MATCHED AFTER THE REPLAY, AND A DRAW THAT FAILS IT IS REJECTED
    # AND REDRAWN -- not reported. §8.1 arm 7 is matched on ACTION COUNT, and
    # a stateful policy cannot be made to cancel exactly the drawn set, so
    # the realised count is only knowable once the control has been replayed.
    # One seed gave 496 realised actions against the treated arm's 333; that
    # is a reshuffle, not a matched control. The attempt budget is the frozen
    # one and EXHAUSTING IT REFUSES rather than accepting a mismatch.
    def realised(res):
        """Per-stratum realised action count, straight off ONE replay.

        `_realised_by_stratum` reads a four-leg arm dict at its reported leg;
        this script replays one leg, so the map is built directly rather than
        by faking an arm shape -- summing across legs of one and reading one
        leg of the other compares two different quantities, which the helper's
        own docstring calls "a 100% rejection rate wearing a matching rule's
        name"."""
        out = {}
        for c in res["trajectory"]:
            if c.get("kind") != "CANCEL_ISSUED":
                continue
            st = (c["side"], R._hour_of(c["slug"]))
            out[st] = out.get(st, 0) + 1
        return out

    rc_treated = realised(r5)
    attempts = accepted = 0
    rej = {"PERM_NOT_OK": 0, "P1": 0, "P2": 0, "P3": 0, "P4": 0}
    chosen = None
    # DA's specification: a COUNT cannot separate "the achievable set
    # excludes the target" from "it brackets the target and the budget
    # missed". Every draw that REACHES P4 records its realised value, the
    # SIGNED gap against the treated arm, and the per-stratum breakdown --
    # because P4 is a per-stratum equality and a near-miss on one stratum is
    # a different fact from a miss on all of them.
    P4_OBSERVATIONS = []
    # THE MECHANISM'S OWN VARIABLE, per stratum: how many above-threshold
    # events there were, so the suppression rate can be COMPUTED and the
    # mechanism TESTED rather than assumed.
    ABOVE_BY_ST = {}
    for _e in treated:
        _sl, _sd, _gn = _e["slug"].split("|")
        _k = (_sd, R._hour_of(_sl))
        ABOVE_BY_ST[_k] = ABOVE_BY_ST.get(_k, 0) + 1
    N_SEEDS = int(sys.argv[3]) if len(sys.argv) > 3 else 60
    for seed in range(N_SEEDS):
        attempts += 1
        d_ = MRC.draw(pool, treated, seed=seed)
        c_, ok_ = R.permuted_stream(cv_ev, d_, theta_cv, gidx)
        if not ok_:
            rej["PERM_NOT_OK"] += 1; continue
        pre = R.stream_predicates(cv_ev, c_, d_, theta_cv, gidx)
        bad = [k for k in ("P1_key_multisets_equal",
                           "P2_stratum_score_multisets_equal",
                           "P3_drawn_carry_above_and_only_drawn")
               if not pre[k]]
        if bad:
            rej[bad[0][:2]] += 1; continue
        r_, h_, m_, i_ = replay(c_, cancel=True, theta=theta_cv)
        rc_ctrl = realised(r_)
        post = R.stream_predicates(cv_ev, c_, d_, theta_cv, gidx,
                                   rc_treated=rc_treated, rc_control=rc_ctrl)
        _strata = sorted(set(rc_treated) | set(rc_ctrl) | set(ABOVE_BY_ST))
        _per = {f"{st[0]}|{st[1]}": {
            "treated": rc_treated.get(st, 0), "control": rc_ctrl.get(st, 0),
            "signed_gap": rc_ctrl.get(st, 0) - rc_treated.get(st, 0),
            "above_events": ABOVE_BY_ST.get(st, 0),
            # THE MECHANISM'S OWN VARIABLE: above events the treated arm
            # did NOT act on, because a HELD side suppressed the crossing.
            "suppressed": ABOVE_BY_ST.get(st, 0) - rc_treated.get(st, 0),
            "suppression_rate": (
                (ABOVE_BY_ST.get(st, 0) - rc_treated.get(st, 0))
                / ABOVE_BY_ST[st]) if ABOVE_BY_ST.get(st) else None}
            for st in _strata}
        _obs = {"seed": seed,
                "realised_total_control": sum(rc_ctrl.values()),
                "realised_total_treated": sum(rc_treated.values()),
                "signed_gap_total": sum(rc_ctrl.values()) - sum(rc_treated.values()),
                "n_strata": len(_strata),
                "n_strata_matching": sum(1 for v in _per.values()
                                         if v["signed_gap"] == 0),
                "n_strata_control_over": sum(1 for v in _per.values()
                                             if v["signed_gap"] > 0),
                "n_strata_control_under": sum(1 for v in _per.values()
                                              if v["signed_gap"] < 0),
                "per_stratum": _per,
                "P4": bool(post["P4_realised_action_counts_equal"])}
        P4_OBSERVATIONS.append(_obs)
        if not post["P4_realised_action_counts_equal"]:
            rej["P4"] += 1; continue
        accepted += 1; chosen = (d_, c_, r_, h_, m_, i_, post, seed); break
    if chosen is None:
        print(json.dumps({"RANDOM_MATCHED": "REFUSED",
                          "reason": "no draw satisfied P1-P4 within the "
                                    "attempt budget; a control that cannot be "
                                    "matched is refused, never reported as a "
                                    "weaker control",
                          "attempts": attempts, "rejections": rej}), flush=True)
            # THE MECHANISM TEST. Prediction: strata with a HIGHER suppression
        # rate (above events the treated arm did not act on) show a LARGER
        # control-over gap. Averaged per stratum across all draws, then
        # ranked -- Spearman on ranks, because the prediction is monotone and
        # not linear. A FAILURE here is the more important result and is
        # reported as such rather than buried.
        _by_st: dict = {}
        for _o in P4_OBSERVATIONS:
            for _k, _v in _o["per_stratum"].items():
                _b = _by_st.setdefault(_k, {"gaps": [], "supp": None,
                                            "above": None, "treated": None})
                _b["gaps"].append(_v["signed_gap"])
                _b["supp"] = _v["suppression_rate"]
                _b["above"] = _v["above_events"]
                _b["treated"] = _v["treated"]
        _rows = [{"stratum": k,
                  "suppression_rate": v["supp"],
                  "above_events": v["above"],
                  "treated_realised": v["treated"],
                  "mean_control_over": (sum(v["gaps"]) / len(v["gaps"])
                                        if v["gaps"] else None)}
                 for k, v in sorted(_by_st.items())
                 if v["supp"] is not None]

        def _spearman(xs, ys):
            n = len(xs)
            if n < 3:
                return None
            def _rank(a):
                order = sorted(range(n), key=lambda i: a[i])
                r = [0.0] * n
                i = 0
                while i < n:
                    j = i
                    while j + 1 < n and a[order[j + 1]] == a[order[i]]:
                        j += 1
                    avg = (i + j) / 2.0 + 1.0
                    for k in range(i, j + 1):
                        r[order[k]] = avg
                    i = j + 1
                return r
            rx, ry = _rank(xs), _rank(ys)
            mx, my = sum(rx) / n, sum(ry) / n
            num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
            dx = sum((a - mx) ** 2 for a in rx) ** 0.5
            dy = sum((b - my) ** 2 for b in ry) ** 0.5
            return (num / (dx * dy)) if dx and dy else None

        _rho = _spearman([r["suppression_rate"] for r in _rows],
                         [r["mean_control_over"] for r in _rows])
        _mech = {
            "prediction": "strata with a HIGHER suppression rate show a "
                          "LARGER control-over gap",
            "why": "a random draw does not reproduce the treated arm's "
                   "suppression clustering, so the control realises a "
                   "systematically higher fraction of above events",
            "n_strata": len(_rows),
            "per_stratum": _rows,
            "spearman_rho": _rho,
            "statistic": "Spearman on ranks -- the prediction is MONOTONE, "
                         "not linear",
            "verdict": ("UNDECIDABLE_TOO_FEW_STRATA" if _rho is None
                        else "PREDICTION_HELD" if _rho > 0
                        else "PREDICTION_FAILED"),
            "if_failed": "a failure is the MORE IMPORTANT result: it would "
                         "mean the all-positive sign is a repeated "
                         "observation without the mechanism that licenses "
                         "reading it as structural",
            "power_note": "with few strata this statistic is weak; "
                          "UNDECIDABLE is reported rather than resolved",
        }
        _tot = [o["signed_gap_total"] for o in P4_OBSERVATIONS]
        _allpos = all(g > 0 for g in _tot) if _tot else None
        _allneg = all(g < 0 for g in _tot) if _tot else None
        OUT["arms"]["RANDOM_MATCHED"] = {
            "arm": "RANDOM_MATCHED", "status": "REFUSED_NO_MATCHED_DRAW",
            "attempts": attempts, "rejections": rej,
            "predicates_last_seen": None,
            "VALID_AS_A_CONTROL": control_is_valid({}),
            "floor_state": dict(MATCHED_FLOOR_STATE),
            "p4_observations": P4_OBSERVATIONS,
            "p4_summary": {
            # THE HEADLINE IS THE PER-STRATUM SIGN, not the aggregate.
            # An aggregate gap can be produced by a few strata; a
            # UNIVERSAL per-stratum sign cannot. Reporting the aggregate
            # as the finding and per-stratum as detail is backwards.
            "HEADLINE_per_stratum_sign": {
                "n_draws": len(P4_OBSERVATIONS),
                "n_strata_per_draw": (P4_OBSERVATIONS[0]["n_strata"]
                                      if P4_OBSERVATIONS else None),
                # THE UNIVERSAL FACT, and control-over is NOT it. DA
                # recomputed from the raw draws: control-over on EVERY
                # stratum holds in only 35 of 60 draws, with 27 negative
                # per-stratum gaps, 22 of them in one stratum. What IS
                # universal is that NO STRATUM EVER MATCHED EXACTLY --
                # and P4 requires equality on ALL FOUR SIMULTANEOUSLY.
                "stratum_observations_total": sum(
                    o["n_strata"] for o in P4_OBSERVATIONS),
                "stratum_observations_exactly_matching": sum(
                    1 for o in P4_OBSERVATIONS
                    for v in o["per_stratum"].values()
                    if v["signed_gap"] == 0),
                "draws_with_all_strata_matching": sum(
                    1 for o in P4_OBSERVATIONS
                    if all(v["signed_gap"] == 0
                           for v in o["per_stratum"].values())),
                "why_this_is_the_headline": (
                    "NO STRATUM EVER MATCHED EXACTLY, and P4 requires "
                    "equality on ALL strata SIMULTANEOUSLY. That is the "
                    "universal fact. Control-over per stratum is NOT "
                    "universal and must not be reported as though it "
                    "were"),
                "control_over_is_not_universal": {
                    "draws_with_every_stratum_control_over": sum(
                        1 for o in P4_OBSERVATIONS
                        if o["n_strata_control_over"] == o["n_strata"]),
                    "draws_with_any_stratum_control_under": sum(
                        1 for o in P4_OBSERVATIONS
                        if o["n_strata_control_under"] > 0),
                    "negative_per_stratum_gaps": sum(
                        1 for o in P4_OBSERVATIONS
                        for v in o["per_stratum"].values()
                        if v["signed_gap"] < 0),
                    "negative_gaps_by_stratum": {
                        k: sum(1 for o in P4_OBSERVATIONS
                               if o["per_stratum"].get(k, {}).get(
                                   "signed_gap", 0) < 0)
                        for k in (P4_OBSERVATIONS[0]["per_stratum"]
                                  if P4_OBSERVATIONS else {})}}},
            # THE MECHANISM, TESTED RATHER THAN ASSUMED. If the control's
            # over-realisation is because a random draw does not
            # reproduce the treated arm's suppression clustering, then
            # strata with MORE non-acting above events must show LARGER
            # control-over. That is a computable prediction, and a
            # FAILURE here is the more important result.
            "MECHANISM_TEST_gap_vs_suppression": _mech,
            "aggregate_observations": {
                "treated_realised": (
                    P4_OBSERVATIONS[0]["realised_total_treated"]
                    if P4_OBSERVATIONS else None),
                # DA's correction: the fields formerly called
                # `control_realised_min/max` carried the signed-GAP
                # extremes, not the control's realisation. Both are
                # emitted now, each under the name of what it is.
                "signed_gap_min": (min(_tot) if _tot else None),
                "signed_gap_max": (max(_tot) if _tot else None),
                "control_realised_min": (
                    min(o["realised_total_control"]
                        for o in P4_OBSERVATIONS) if P4_OBSERVATIONS
                    else None),
                "control_realised_max": (
                    max(o["realised_total_control"]
                        for o in P4_OBSERVATIONS) if P4_OBSERVATIONS
                    else None),
                "signed_gap_totals": _tot,
                "all_gaps_positive": _allpos,
                "all_gaps_negative": _allneg,
                # AN OBSERVATION, NOT A CONCLUSION -- and named so the
                # field cannot carry what the count will not support.
                "no_draw_in_n_bracketed_the_target": (
                    (min(_tot) > 0 or max(_tot) < 0) if _tot else None),
                "n_draws": len(_tot),
                "rule_of_three_upper_bound_per_draw": (
                    round(3.0 / len(_tot), 4) if _tot else None),
                "what_the_count_supports": (
                    "0 of %d draws at or below the target bounds the "
                    "PER-DRAW probability at ~%.1f%% (rule of three, "
                    "one-sided 95%%). That is a statement about DRAWS, "
                    "not about the achievable SET, and it is not "
                    "exclusion." % (len(_tot), 300.0 / len(_tot))
                    if _tot else None),
                "extrapolation_required_to_bracket": (
                    "target %s, observed minimum %s, observed range %s "
                    "wide -- bracketing needs a draw about %.1f range-"
                    "widths below the smallest of %d. Not absurd; not "
                    "established." % (
                        P4_OBSERVATIONS[0]["realised_total_treated"],
                        min(o["realised_total_control"]
                            for o in P4_OBSERVATIONS),
                        max(o["realised_total_control"]
                            for o in P4_OBSERVATIONS)
                        - min(o["realised_total_control"]
                              for o in P4_OBSERVATIONS),
                        (min(o["realised_total_control"]
                             for o in P4_OBSERVATIONS)
                         - P4_OBSERVATIONS[0]["realised_total_treated"])
                        / max(max(o["realised_total_control"]
                                  for o in P4_OBSERVATIONS)
                              - min(o["realised_total_control"]
                                    for o in P4_OBSERVATIONS), 1),
                        len(P4_OBSERVATIONS))
                    if P4_OBSERVATIONS else None)},
            "what_licenses_the_strong_reading": (
                "the MECHANISM, not the count: a random draw does not "
                "reproduce the treated arm's suppression clustering, so "
                "the control realises a systematically higher fraction of "
                "above events. Under that mechanism an all-positive sign "
                "is CONFIRMATION OF A PREDICTED SIGN and the count is "
                "corroboration, not the argument"),
            "DA_LIMIT_QUOTED": (
                "UNREACHABLE FOR THIS CONSTRUCTION, NEVER NEVER"),
            "conditional_on": (
                "the draw rule, the stratification, and the frozen "
                "protocol's single matched quantity. Without this field a "
                "reader inherits 'no matched floor exists' when what was "
                "shown is 'no matched floor exists UNDER THIS DRAW'"),
            "honest_limit": MATCHED_FLOOR_STATE["honest_limit"]}}
    else:
        drawn, ctrl, r7, h7, m7, i7, post, seed_used = chosen
        add("RANDOM_MATCHED", r7, h7, m7, i7,
            {"predictor": "MATCHED_RANDOM_PERMUTATION", "artifacts": cv_v,
             "note": "the CONDVALUE stream's above-threshold values permuted "
                     "within (side, hour); shares CONDVALUE's artifacts BY "
                     "CONSTRUCTION"},
            matched={"above_threshold_demand": len(treated), "drawn": len(drawn),
                     "seed_accepted": seed_used, "attempts": attempts,
                     "rejections": rej, "predicates": post,
                     "VALID_AS_A_CONTROL": control_is_valid(post),
                     "predicates_required": list(CONTROL_PREDICATES),
                     "demand": DEMAND})

    # DISTINCTNESS, ASSERTED: identical outputs across arms is what round 52
    # shipped. The emission carries the check rather than a reader making it.
    # A REFUSED ARM HAS NO SIGNATURE AND MUST STILL BE REPRESENTABLE. The
    # first version indexed `n_cancels` on every arm and died when the control
    # refused -- a distinctness check that cannot describe a refusal is a
    # check that only works when nothing went wrong.
    sig = {a: ("REFUSED:" + v.get("status", "?") if "fields" not in v
               else (v["n_cancels"], v["fields"]["rho"].get("value"),
                     v["fields"]["post_fill_markout_cents"].get("value")))
           for a, v in OUT["arms"].items()}
    OUT["arm_distinctness"] = {
        "signatures": sig,
        "n_arms": len(sig), "n_distinct": len(set(map(str, sig.values()))),
        "all_distinct": len(set(map(str, sig.values()))) == len(sig),
        "predictors": {a: v.get("identity", {}).get("predictor", "NONE_REFUSED")
                       for a, v in OUT["arms"].items()},
        "n_distinct_predictors": len({
            v.get("identity", {}).get("predictor", "NONE_REFUSED")
            for v in OUT["arms"].values()}),
        "arms_with_values": sorted(a for a, v in OUT["arms"].items()
                                   if "fields" in v),
        "arms_refused": sorted(a for a, v in OUT["arms"].items()
                               if "fields" not in v),
        # DERIVED from the predicates, never a constant: a floor exists only
        # if some arm is a control whose four predicates all hold.
        "floor_available": any(
            control_is_valid((v.get("matched") or {}).get("predicates") or {})
            for v in OUT["arms"].values()),
        "no_arm_vs_floor_comparison_available": not any(
            control_is_valid((v.get("matched") or {}).get("predicates") or {})
            for v in OUT["arms"].values()),
        "why_no_floor": dict(MATCHED_FLOOR_STATE)}
    # ---- DE59-C3: THE TAIL, THE BREAK-EVEN RATIO, AND THE CASCADE -----
    # THE QUESTION THAT DECIDES THE OVERLAY CASE, AND IT IS A SET
    # DIFFERENCE RATHER THAN AN INFERENCE. The book's maker P&L is
    # carried by its extreme fills; if the body sums AGAINST the net then
    # `r = adverse/spread` is already past any plausible break-even on
    # 99% of the book, and the overlay pays IFF IT DECLINES THE BODY
    # WITHOUT DECLINING THE TAIL. Whether each arm does that was never
    # measured -- it was inferred from the aggregate ("it evidently did
    # not remove many, or the delta would be more negative"). An
    # inference from an aggregate is not a count.
    _bl = ARM_FILLS.get("QR_SKEW_ONLY") or []
    if not _bl:
        OUT["tail_and_cascade"] = {"status": "NO_BASELINE_FILLS"}
    else:
        _acting = {a: ARM_FILLS[a] for a in ARM_FILLS
                   if a != "QR_SKEW_ONLY" and ARM_FILLS[a]}
        OUT["tail_and_cascade"] = {
            # BOTH RANKINGS. The answer to "did it decline the tail?"
            # must not depend on which tail we mean, and on this book
            # the two tails carry 1.13 and 0.10 of the net.
            "tail_decline": {b: R.tail_decline(_bl, _acting, by=b)
                             for b in ("abs", "signed")},
            "adverse_over_spread": {
                a: R.adverse_over_spread(f)
                for a, f in [("QR_SKEW_ONLY", _bl)] + sorted(_acting.items())},
            "cancel_mechanics": R.cancel_mechanics(
                _bl,
                {a: (f, OUT["arms"][a]["n_cancels"])
                 for a, f in _acting.items()},
                R.generations_with_fills(ref)),
            "why_these_three_together": (
                "the tail says WHERE the book's P&L lives, `r` says what "
                "an overlay must beat and what the body already exceeds, "
                "and the cascade says whether a cancel's cost is the "
                "ranker's PICK or the number of fills the picked "
                "generation held. Separately each invites a wrong "
                "reading; together they are the specification"),
            "decides_nothing": "REPORTED (rule 14).",
        }

    # ---- DE59: WHAT THE POLICY DOES TO THE RESIDUAL -------------------
    # RULED INTO THE EMISSION whatever the inventory leg turns out to be,
    # because it characterises what the policy DOES rather than what it
    # EARNS -- and the two are not the same claim. Read off the terminal
    # net BEFORE any P&L: an arm that CUTS the residual's magnitude is
    # reducing risk; an arm that FLIPS ITS SIGN at the same magnitude is
    # taking a DIRECTIONAL BET, whose payoff depends on the realised path
    # and not on the mechanism. A number that improves because of the
    # second is not evidence the policy manages inventory.
    _bn = (OUT["arms"].get("QR_SKEW_ONLY", {}).get("fields", {})
           .get("terminal_inventory", {}).get("value"))
    if _bn is None:
        OUT["residual_policy_effect"] = {
            "status": "NO_BASELINE_TERMINAL_NET"}
    else:
        _re = {}
        for _a, _v in OUT["arms"].items():
            if _a == "QR_SKEW_ONLY" or "fields" not in _v:
                continue
            _n = _v["fields"].get("terminal_inventory", {}).get("value")
            if _n is None:
                _re[_a] = {"status": "NO_TERMINAL_NET"}
                continue
            _re[_a] = {
                "n_cancels": _v["n_cancels"],
                "terminal_net_shares": _n,
                "magnitude_ratio_to_baseline": (abs(_n) / abs(_bn)
                                                if abs(_bn) > 1e-12
                                                else None),
                "reduces_magnitude": abs(_n) < abs(_bn),
                "reverses_sign": (_n * _bn) < 0,
                "reading": ("RISK REDUCTION requires the magnitude to "
                            "fall; a SIGN FLIP at equal magnitude is a "
                            "DIRECTIONAL BET, not risk reduction"),
            }
        OUT["residual_policy_effect"] = {
            "baseline_arm": "QR_SKEW_ONLY",
            "baseline_terminal_net_shares": _bn,
            "arms": _re,
            "why_this_is_emitted_regardless": (
                "it is read off the terminal net BEFORE any P&L, so it "
                "characterises the policy independently of what the "
                "inventory leg turns out to be"),
            "decides_nothing": "REPORTED (rule 14).",
        }

    # ---- DE58: IS CANCELLING WORTH DOING AT ALL? ----------------------
    # §8.1 closes "`net_cancel_cents` alone is not a strategy-P&L
    # verdict." Until this round every field an arm filled was a
    # CANCELLATION quantity -- markout, retention, rho, cancel counts --
    # and QR_SKEW_ONLY at 0 cancels was called the baseline while
    # producing no P&L, so nothing here measured whether cancelling was
    # worth doing. It does now, and only because the maker-P&L
    # decomposition landed: an arm that cancels gives up ENTRY EDGE on
    # the fills it declines and saves ADVERSE SELECTION on the same
    # fills, and those two are separable only once the markout is split.
    #
    # THE PREDICATE IS COMPUTED, NEVER PRINTED (rule 10). The identity
    # net == saved - forgone is CHECKED, not asserted.
    _base = OUT["arms"].get("QR_SKEW_ONLY", {}).get("fields")
    if _base is None:
        OUT["cancellation_economics"] = {
            "status": "NO_BASELINE",
            "why": "the 0-cancel arm produced no fields, so there is "
                   "nothing to price cancellation against"}
    else:
        _b_sp = _base["spread_capture_cents"]["value"]
        _b_ad = _base["maker_pnl_cents"]["also"]["adverse_selection_cents"]
        _b_pl = _base["maker_pnl_cents"]["value"]
        _rows = {}
        for _a, _v in OUT["arms"].items():
            if _a == "QR_SKEW_ONLY" or "fields" not in _v:
                continue
            _sp = _v["fields"]["spread_capture_cents"]["value"]
            _ad = _v["fields"]["maker_pnl_cents"]["also"][
                "adverse_selection_cents"]
            _pl = _v["fields"]["maker_pnl_cents"]["value"]
            _forgone, _saved, _net = _b_sp - _sp, _ad - _b_ad, _pl - _b_pl
            _rows[_a] = {
                "n_cancels": _v["n_cancels"],
                "spread_forgone_cents": _forgone,
                "adverse_saved_cents": _saved,
                "net_pnl_delta_cents": _net,
                "adverse_saved_exceeds_spread_forgone": _saved > _forgone,
                "identity_residual_cents": abs(_saved - _forgone - _net),
                "identity_holds":
                    abs(_saved - _forgone - _net) <= 1e-6,
                "cents_per_cancel": (_net / _v["n_cancels"]
                                     if _v["n_cancels"] else None),
            }
        OUT["cancellation_economics"] = {
            "baseline_arm": "QR_SKEW_ONLY",
            "baseline": {"n_cancels": 0, "spread_capture_cents": _b_sp,
                         "adverse_selection_cents": _b_ad,
                         "maker_pnl_cents": _b_pl},
            "arms": _rows,
            "predicate": "an arm pays SPREAD FORGONE on the fills it "
                         "declines and is paid ADVERSE SAVED on the same "
                         "fills; cancelling adds value iff saved > forgone",
            "identity": "net_pnl_delta == adverse_saved - spread_forgone "
                        "(COMPUTED per arm, never asserted)",
            "scope_and_limits": (
                "THE FILLS LEG ONLY, valued at t + MARKOUT_S over "
                "n=%d windows of %s. It EXCLUDES the residual position "
                "at window end (`inventory_loss_cents`, unruled) and any "
                "value of avoiding a fill whose harm lands beyond the "
                "markout horizon. DEVELOPMENT EVIDENCE, not a validation: "
                "one coin, one latency rung, no forward day, no interval "
                "-- 12 windows is below the 5-complete-day cluster floor, "
                "so a POINT ESTIMATE AND NO INTERVAL (rule 8)."
                % (POP["windows"], POP["as_of"])),
            "decides_nothing": "REPORTED. Whether the trade-off is worth "
                               "taking is the policy layer's (rule 14).",
        }
    OUT["population_exclusions"] = EXCL
    OUT["peak_rss_gb"] = gb(); OUT["total_wall_s"] = round(time.time()-T0, 1)
    # IMMUTABLE OUTPUT NAME. Round 53 wrote every run to one filename, so a
    # later run overwrote the one that had been filed from and the two could
    # not be told apart afterwards -- which is why "the artifact on disk" and
    # "the artifact I filed from" were different runs.
    _dst = emit(OUT, SCRATCH / f"de_section81_arms__{RUN_ID}.json")
    print(json.dumps({"emitted": str(_dst), "run_id": RUN_ID,
                      "produced_by": PROV["produced_by"],
                      "producing_code": PROV["producing_code"],
                      "carrying_commit": PROV["carrying_commit_short"]}),
          flush=True)
    print(json.dumps(OUT["arm_distinctness"]), flush=True)
    print("ARMS53 COMPLETE peak_gb=%s wall=%ss" % (OUT["peak_rss_gb"], OUT["total_wall_s"]), flush=True)

    return OUT


if __name__ == "__main__":
    run_arms()
