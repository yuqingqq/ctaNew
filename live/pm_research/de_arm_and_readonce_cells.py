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
    ck("both declaration identities are pinned and named",
       set(pins) == {"day_read_state_attestation",
                     "forward_test_declaration"},
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
        verdicts.append((r.returncode, r.stdout.strip()))
    ck("the self-check gives ONE verdict from /tmp and from the tree root",
       verdicts[0] == verdicts[1] and verdicts[0][0] == 0,
       f"{verdicts[0][1][:12]}")
    print(f"\n{ok}/{n} cells pass")
    return 0 if ok == n else 1


if __name__ == "__main__":
    raise SystemExit(falsify() if "--falsify" in sys.argv[1:] else 2)
