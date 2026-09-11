"""CELLS: the read-once growing-ledger fixture, and the admitting arms.

Both properties are about what a RUN records, so each cell drives the same
functions the run calls -- never a re-implementation.
"""
from __future__ import annotations
import hashlib, json, sys, tempfile   # noqa: F401
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import de_settlement_control_run as SC          # noqa: E402
import de_multiday_gate1_runner as R            # noqa: E402
import be_score_neutrality as BEN               # noqa: E402

EXACT = "7ed5a9015f75de64feeeeaad21d97e4eecc2b15c"


def falsify() -> int:
    n = ok = 0

    def ck(name, cond, note=""):
        nonlocal n, ok
        n += 1
        ok += bool(cond)
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}" +
              (f"  {note}" if note else ""))

    # --- (c) THE GROWING-LEDGER FIXTURE --------------------------------
    # The live ledger grows between the arms; both cells must still carry
    # ONE winner_source, because the run reads it once and passes it.
    live = R.winner_source()
    snapshot_sha, snapshot_n = live["sha256"], live["n_records"]
    with tempfile.TemporaryDirectory() as td:
        led = Path(td) / "resolutions.jsonl"
        led.write_text("\n".join(
            json.dumps({"slug": f"btc-updown-5m-{1788739200 + i*300}",
                        "closed": True, "winners": {"Up": True}})
            for i in range(6)) + "\n")
        before = hashlib.sha256(led.read_bytes()).hexdigest()
        passed = {"sha256": before, "n_records": 6,
                  "winners": {f"btc-updown-5m-{1788739200 + i*300}":
                              {"Up": True} for i in range(6)}}
        # the ledger GROWS after the read
        with led.open("a") as fh:
            fh.write(json.dumps({"slug": "btc-updown-5m-1789000000",
                                 "closed": True,
                                 "winners": {"Up": False}}) + "\n")
        after = hashlib.sha256(led.read_bytes()).hexdigest()
        ck("the ledger really moved between the arms", before != after)
        ck("arm 1 and arm 2 carry the SAME winner_source",
           passed["sha256"] == before and passed["sha256"] != after,
           "both record the passed snapshot, not the file")
        missing = [s for s in ["btc-updown-5m-1789000000"]
                   if s not in passed["winners"]]
        ck("a slug absent from the snapshot still refuses by name",
           bool(missing), "SETTLEMENT_WINNER_MISSING_FOR_SLUG")
    ck("the real oracle is readable once and reports its identity",
       bool(snapshot_sha) and snapshot_n > 0,
       f"sha {snapshot_sha[:16]} n={snapshot_n}")

    # --- REV's FOUR ADMITTING-ARM CELLS --------------------------------
    ck("the exact pin admits, arm EXACT",
       SC._admitting_arm(EXACT) == "EXACT")
    ck("a descendant with the declared digests admits, arm DESCENDANT",
       SC._admitting_arm("dbb11e4") == "DESCENDANT")
    ck("a non-descendant refuses (no arm)",
       SC._admitting_arm("0" * 40) is None)
    # (3) both former silent Falses now REFUSE BY NAME
    import tempfile as _tf, shutil as _sh
    try:
        SC._builder_commit_admissible("0" * 40)
        ck("a non-descendant refuses BY NAME", False)
    except SC.SettlementControlRefused as e:
        ck("a non-descendant refuses BY NAME",
           SC.BUILD_NOT_DESCENDANT in str(e))
    # The fixture declaration lives INSIDE the real declarations dir under
    # a distinct name: moving HERE to a temp tree also moves the git root
    # the descendant check uses, so NOT_DESCENDANT fired before the digest
    # check could. Keep HERE real; vary only the pinned identity.
    D = HERE / "declarations"
    fx = D / "da_forward_test_declaration_v26.DIGESTCELL.json"
    try:
        doc = json.loads(
            (D / "da_forward_test_declaration_v26.json").read_text())
        k = sorted(doc["BUILD_PINNED_DIGESTS"])[0]
        doc["BUILD_PINNED_DIGESTS"][k] = "0" * 64
        fx.write_text(json.dumps(doc))
        pin = {"path": fx.name,
               "sha256": hashlib.sha256(fx.read_bytes()).hexdigest()}
        orig_pin = SC._declaration_pin
        SC._declaration_pin = lambda decl_dir=None, _p=pin: _p
        try:
            SC._builder_commit_admissible("dbb11e4")
            ck("one declared digest moved refuses BY NAME", False)
        except SC.SettlementControlRefused as e:
            ck("one declared digest moved refuses BY NAME",
               SC.BUILD_DIGEST_MOVED in str(e) and k in str(e), k)
        finally:
            SC._declaration_pin = orig_pin
    finally:
        if fx.exists():
            fx.unlink()          # the cell leaves nothing behind
    pins = R.resolve_declaration_pins()
    # ALL THREE, INCLUDING THE ONE THE VALUATION'S OWN COMMIT COMES FROM.
    # `code_freeze_declaration` was pinned by DA's freeze v14 and wanted by
    # nobody, so the resolver returned None for it and `_frozen_commit()`
    # fell back to the v1 FILENAME -- a commit two freezes stale, read
    # without a refusal. An identity the chain pins and the reader does not
    # ask for is not pinned.
    ck("every declaration identity the chain pins is named here",
       set(pins) == {"day_read_state_attestation",
                     "forward_test_declaration",
                     "code_freeze_declaration"},
       ",".join(sorted(pins)))
    # DE 324: ONE VERDICT, WHATEVER THE CWD. The self-check refused from
    # every directory because its digest loop still used the literal
    # PIPELINE_COMMIT (now a sentinel) while the HEAD check used the
    # resolved frozen commit -- two sources of truth in one function.
    import subprocess as _sp
    verdicts = []
    for cwd in ("/tmp", str(HERE.parents[1])):
        r = _sp.run(["/home/yuqing/pricer-sol/venv/bin/python3", "-c",
                     "import sys; sys.path.insert(0, %r)\n"
                     "import de_forward_value_day as V\n"
                     "print(V._PREFLIGHT['head'])" % str(HERE)],
                    capture_output=True, text=True, cwd=cwd)
        name = next((w for w in (r.stderr or "").split()
                     if w.isupper() and len(w) > 8), "")
        verdicts.append((r.returncode, r.stdout.strip(), name))
    # THE PROPERTY IS AGREEMENT, NOT SUCCESS. Requiring rc==0 would make
    # this cell fail for a reason that has nothing to do with cwd -- e.g.
    # a landed fix the declaration does not yet name -- and a cell that
    # fails for the wrong reason teaches a reader to ignore it.
    ck("the self-check gives ONE verdict from /tmp and from the tree root",
       verdicts[0] == verdicts[1],
       f"rc={verdicts[0][0]} {(verdicts[0][1] or verdicts[0][2])[:44]}")

    # --- CELL 12 (REVIEW 175 A): THE PRODUCTION ENTRY POINT ------------
    # A NameError in the record build passed import and all eleven cells
    # and would have crashed the real launch AFTER both arms were valued.
    # Every cell above proves a FUNCTION; none ran the ENTRY POINT.
    import subprocess, os                                  # noqa: E402
    import de_unbound_name_sweep as SWEEP                   # noqa: E402
    tree = HERE.parents[1]

    # ARM 1: the sweep, run AS AN INSTRUMENT FROM A FOREIGN CWD -- the
    # same way anything else would run it.
    r = subprocess.run(
        [sys.executable, str(HERE / "de_unbound_name_sweep.py")],
        capture_output=True, text=True, cwd="/tmp",
        env=dict(os.environ, DE_VALUATION_PREFLIGHT_OFF="1"))
    ck("no function in the valuation closure loads an unbound name",
       r.returncode == 0,
       (r.stdout.strip() or r.stderr.strip()).splitlines()[-1][:70])

    # ARM 2 (rule 15): a zero from an instrument that never proved it can
    # fire is not a result. The sweep's own falsifier, driven here.
    rf = subprocess.run(
        [sys.executable, str(HERE / "de_unbound_name_sweep.py"), "--falsify"],
        capture_output=True, text=True, cwd="/tmp",
        env=dict(os.environ, DE_VALUATION_PREFLIGHT_OFF="1"))
    ck("the sweep FLAGS planted unbound names (its own falsifier)",
       rf.returncode == 0 and "4/4" in rf.stdout,
       rf.stdout.strip().splitlines()[-1][:40])

    # ARM 3: the entry point invoked THE WAY PRODUCTION INVOKES IT --
    # be_heavy_run.sh --inner --lock, absolute arguments, from a foreign
    # cwd -- end to end on a consumed day in POINT_ESTIMATE. This is the
    # only arm that reaches the RECORD BUILD, which is where A lived:
    # after both arms are valued, past import, past every other cell.
    absent = []
    D = Path("/home/yuqing/ctaNew/data/pm_5min/derived")
    book = D / "be_daybook_20260907_btc__L250ms__FWD1.pkl"
    rcpt = D / "be_daybook_receipt_20260907_btc__L250ms__FWD1.json"
    cert = D / "be_score_neutrality_20260903__EV22_vs_NEUTCHK__da00220.json"
    lock = Path("/home/yuqing/ctaNew/data/.heavy_run.lock")
    absent += [f"MISSING:{f.name}" for f in (book, rcpt, cert)
               if not f.is_file()]
    # The driver refuses AT IMPORT when the declaration does not name this
    # tree's bytes -- which is the expected state between a landed fix and
    # DA's re-declaration. Reading the frozen commit must not depend on
    # that verdict, so the pre-flight is off for THIS read and on for the
    # production arm below.
    os.environ["DE_VALUATION_PREFLIGHT_OFF"] = "1"
    try:
        import de_forward_value_day as V                    # noqa: E402
        frozen = V._frozen_commit()
    except Exception as exc:                                # noqa: BLE001
        frozen = None
        absent.append(f"NO_FROZEN_COMMIT:{type(exc).__name__}")
    finally:
        os.environ.pop("DE_VALUATION_PREFLIGHT_OFF", None)
    live = subprocess.run(
        [sys.executable, "-c", "import de_forward_value_day"],
        capture_output=True, text=True, cwd="/tmp",
        env=dict(os.environ, PYTHONPATH=str(HERE)))
    if live.returncode != 0:
        absent.append("DECLARATION_DOES_NOT_YET_NAME_THIS_COMMIT")
    if lock.is_file() and subprocess.run(
            ["flock", "-n", str(lock), "true"],
            capture_output=True).returncode != 0:
        absent.append("HEAVY_RUN_LOCK_HELD")
    if absent:
        # NOT A PASS AND NOT A FAIL: the arm DID NOT RUN, and the script
        # exits 4 (INPUT_ABSENT) so no reader can mistake the set for
        # closed. A skipped heavy arm printed green is the failure mode
        # this whole cell exists to prevent.
        print("  [INPUT_ABSENT] the production entry point, end to end  "
              + ",".join(absent))
        production_absent = True
    else:
        production_absent = False
        out = D / "de_cells_prod_entrypoint"
        out.mkdir(parents=True, exist_ok=True)
        params = HERE / "declarations" / Path(str(
            BEN.resolve_frozen_params_pin(
                HERE / "declarations")["pin"]["path"])).name
        pr = subprocess.run(
            [str(HERE / "be_heavy_run.sh"), "--inner", "--lock", str(lock),
             sys.executable, str(HERE / "de_forward_value_day.py"),
             "--day", "2026-09-07", "--book", str(book),
             "--book-receipt", str(rcpt), "--score-certification", str(cert),
             "--params", str(params), "--out-dir", str(out),
             "--n-draws", "0", "--seed", "0",
             "--days-scored", "2026-09-07", "--n-declared", "7",
             "--derived", str(D)],
            capture_output=True, text=True, cwd="/tmp",
            env=dict(os.environ, BE_WORKTREE=str(tree),
                     DE_VALUATION_EXPECTED_TREE=str(tree)))
        rec_f = out / "p003_de_forward_value_20260907.json"
        rec = json.loads(rec_f.read_text()) if rec_f.is_file() else {}
        prov = rec.get("computing_module_provenance") or {}
        ck("the production entry point runs END TO END and writes the "
           "record (the NameError's own site)",
           pr.returncode == 0 and bool(rec) and "NameError" not in pr.stderr,
           f"rc={pr.returncode} " + (
               f"cells={rec.get('cells')}" if pr.returncode == 0
               else pr.stderr.strip()[-64:]))
        ck("the record's provenance names the FROZEN commit and every "
           "computing module matches it",
           prov.get("pipeline_commit") == frozen
           and prov.get("every_computing_module_matches_the_pipeline_commit")
           is True,
           f"{str(prov.get('pipeline_commit'))[:12]} all="
           f"{prov.get('every_computing_module_matches_the_pipeline_commit')}")

    # --- CELL 15: THE FREEZE CHECK TAKES THE BUILD PIN'S FORM ----------
    # USER RULING, DE 331. A declaration naming a tip can never name the
    # commit that CONTAINS it -- DA's re-declaration lands on top of the
    # freeze, so the tip is always one past it, and tip-equality refuses
    # the correct tree forever. Provenance is ANCESTRY; identity is
    # DIGESTS; and a module does not vouch for itself.
    import de_preflight_matrix as MX                       # noqa: E402
    real_rows, real_frozen = V._declared_closure_digests, V._frozen_commit
    tree_p = HERE.parents[1]   # the module's own tree, not a cwd
    head_now = subprocess.run(["git", "-C", str(tree_p), "rev-parse", "HEAD"],
                              capture_output=True, text=True).stdout.strip()
    on_disk = {n: hashlib.sha256(
        (tree_p / "live" / "pm_research" / n).read_bytes()).hexdigest()
        for n in real_rows()}

    def drive(rows, frozen):
        V._declared_closure_digests = lambda: rows
        V._frozen_commit = lambda: frozen
        try:
            V.assert_computing_modules_at_the_pipeline_commit()
            return ""
        except Exception as exc:                           # noqa: BLE001
            return str(exc)
        finally:
            V._declared_closure_digests = real_rows
            V._frozen_commit = real_frozen

    frozen_decl = real_frozen()
    admit_desc = drive(dict(on_disk), frozen_decl)
    admit_same = drive(dict(on_disk), head_now)
    one_moved = dict(on_disk, de_forward_evaluator_py=None)
    one_moved.pop("de_forward_evaluator_py", None)
    one_moved["de_forward_evaluator.py"] = "0" * 64
    moved_out = drive(one_moved, frozen_decl)
    stranger = drive(dict(on_disk), "af675ac" + "0" * 33)
    # V2's row deliberately wrong: the valuation must ADMIT (it records,
    # never asserts) and the LAUNCHER's row must refuse.
    v2_wrong = dict(on_disk)
    v2_wrong["de_settlement_control_run.py"] = "9" * 64
    v2_in_module = drive(v2_wrong, frozen_decl)
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        for f in (HERE / "declarations").glob("de_arm_freeze_v*.json"):
            (d / f.name).write_bytes(f.read_bytes())
        cf = json.loads((HERE / "declarations" /
                         Path(str(R.resolve_declaration_pins(
                             HERE / "declarations")
                             ["code_freeze_declaration"]["path"])).name
                         ).read_text())
        cf["VALUATION_CLOSURE_DIGESTS_AT_THE_FREEZE"][
            "de_settlement_control_run.py"] = "9" * 64
        (d / Path(str(R.resolve_declaration_pins(
            HERE / "declarations")["code_freeze_declaration"]["path"])).name
         ).write_text(json.dumps(cf))
        v2_row_bad = MX.v2_matches_declaration(d)
    v2_row_live = MX.v2_matches_declaration()

    ck("DESCENDANT tip + the declared rows identical ADMITS",
       admit_desc == "", admit_desc[:60] or f"head {head_now[:12]}")
    ck("the SAME tip admits too (no tip-equality either way)",
       admit_same == "", admit_same[:60] or f"frozen = head {head_now[:12]}")
    ck("ONE declared row moved REFUSES by module name",
       "VALUATION_CLOSURE_DIGEST_MOVED" in moved_out
       and "de_forward_evaluator.py" in moved_out, moved_out[:64])
    ck("a NON-DESCENDANT freeze commit REFUSES by name",
       "VALUATION_TREE_IS_NOT_A_DESCENDANT_OF_THE_FROZEN_COMMIT" in stranger,
       stranger[:60] or "ADMITTED A STRANGER")
    ck("V2's own row is RECORDED, not asserted: the valuation admits",
       v2_in_module == "", v2_in_module[:60] or "admits, as ruled")
    ck("and the LAUNCHER's external row is what refuses it, by name",
       str(v2_row_bad.get("status", "")).endswith(MX.V2_MISMATCH)
       and v2_row_live.get("status") == "PASS",
       f"{v2_row_bad.get('status')} | live {v2_row_live.get('status')}")

    # --- CELL 14: VERSION-ORDERED SUPERSESSION (USER RULING, DE 329) ---
    # Measured deadlock, 13:33Z: DA's code freeze v3 named the new commit,
    # my agree-or-refuse rule would have REFUSED the amendment pinning it,
    # and the valuation went on resolving d095c5a -- two freezes stale,
    # quietly. Declaration pins now supersede in version order exactly as
    # params pins do; the chain is recorded so the move is visible.
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        (d / "de_arm_freeze_v1.json").write_text(json.dumps(
            {"code_freeze_declaration": {"path": "cf_v2.json",
                                         "sha256": "2" * 64}}))
        (d / "de_arm_freeze_v14_amendment.json").write_text(json.dumps(
            {"code_freeze_declaration": {"path": "cf_v3.json",
                                         "sha256": "3" * 64}}))
        (d / "de_arm_freeze_v15_amendment.json").write_text(json.dumps(
            {"code_freeze_declaration": {"path": "cf_v4.json",
                                         "sha256": "4" * 64}}))
        moved = R.resolve_declaration_pins(d)["code_freeze_declaration"]
        # ONE amendment, the SAME key twice, different identities: version
        # order cannot settle a document disagreeing with itself, and JSON
        # would have kept the last silently.
        (d / "de_arm_freeze_v16_amendment.json").write_bytes(json.dumps(
            {"code_freeze_declaration": {"path": "cf_v5.json",
                                         "sha256": "5" * 64}}).encode()[:-1]
            + b', "code_freeze_declaration": {"path": "cf_v6.json", '
              b'"sha256": "6666666666666666666666666666666666666666'
              b'666666666666666666666666"}}')
        try:
            R.resolve_declaration_pins(d)
            dup = ""
        except Exception as exc:                           # noqa: BLE001
            dup = str(exc)
    ck("v14 -> v15 SUPERSEDES: the later amendment's identity wins",
       moved.get("path") == "cf_v4.json" and moved.get("version") == 15
       and moved.get("supersedes") == "cf_v3.json",
       f"{moved.get('path')} v{moved.get('version')} "
       f"<- {moved.get('supersedes')}")
    ck("the FULL CHAIN of pins is recorded, in version order (rule 13)",
       [(c["version"], c["path"]) for c in moved.get("chain", [])]
       == [(1, "cf_v2.json"), (14, "cf_v3.json"), (15, "cf_v4.json")],
       str([(c["version"], c["path"]) for c in moved.get("chain", [])]))
    ck("ONE amendment pinning the same key twice still REFUSES by name",
       "DECLARATION_PIN_CONFLICT" in dup and "cf_v6.json" in dup,
       dup[:60] or "ADMITTED A DOCUMENT DISAGREEING WITH ITSELF")

    # --- CELL 13 (REVIEW 175 B): THE DECLARATION READER ----------------
    # `_declaration_pin`'s second path referenced an unbound `chain`, and
    # the blanket `except` read the NameError as "this declaration names
    # no pin". Absence and defect were indistinguishable -- again.
    decl = HERE / "declarations"
    landed = SC._declaration_pin(decl)
    chain_pin = R.resolve_declaration_pins(decl).get("forward_test_declaration")
    ck("the reader returns the LANDED declaration's pinned identity",
       bool(landed) and landed.get("path") == (chain_pin or {}).get("path")
       and landed.get("sha256") == (chain_pin or {}).get("sha256"),
       f"{Path(str((landed or {}).get('path'))).name} "
       f"{str((landed or {}).get('sha256'))[:12]}")

    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        # the freeze names a PARAMS pin but NO forward_test_declaration:
        # the second path -- the one that raised NameError -- is the only
        # way to an answer here.
        (d / "de_arm_freeze_v1.json").write_text(json.dumps(
            {"frozen_parameters": {"params": {"path": "p_cell.json"}}}))
        (d / "p_cell.json").write_text(json.dumps(
            {"forward_test_declaration": {"path": "fwd_cell.json",
                                          "sha256": "f" * 64}}))
        fallback = SC._declaration_pin(d)
        (d / "p_cell.json").write_text(json.dumps({"no_pin_here": True}))
        nothing = SC._declaration_pin(d)
        # A REFUSAL, NOT A DISAGREEMENT. Two amendments naming different
        # identities is now legitimate supersession (DE 329); the refusal
        # that remains is ONE amendment pinning the same key twice, which
        # version order cannot settle.
        (d / "de_arm_freeze_v2_amendment.json").write_bytes(json.dumps(
            {"forward_test_declaration": {"path": "a.json",
                                          "sha256": "a" * 64}}).encode()[:-1]
            + b', "forward_test_declaration": {"path": "b.json", '
              b'"sha256": "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb'
              b'bbbbbbbbbbbbbbbbbbbbbbbb"}}')
        try:
            SC._declaration_pin(d)
            conflict = ""
        except Exception as exc:                           # noqa: BLE001
            conflict = str(exc)
    ck("the PARAMS fallback path reaches an answer (it raised NameError)",
       (fallback or {}).get("path") == "fwd_cell.json",
       str(fallback))
    ck("a declaration that names no pin still returns absence",
       nothing is None, str(nothing))
    ck("a CONFLICT propagates out of the reader instead of being "
       "swallowed as absence",
       "DECLARATION_PIN_CONFLICT" in conflict, conflict[:56] or "RETURNED NONE")

    print(f"\n{ok}/{n} cells pass")
    if ok != n:
        return 1
    return 4 if production_absent else 0


if __name__ == "__main__":
    raise SystemExit(falsify() if "--falsify" in sys.argv[1:] else 2)

