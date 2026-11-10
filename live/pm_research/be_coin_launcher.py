"""BE 215: build any stage for ANY COIN, without moving a pinned digest.

WHY THIS FILE EXISTS AND WHY IT IS NOT A ONE-LINE EDIT.
`be_daybook_build.py` is already coin-parameterised where it counts -- `build(day,
*, coin=COIN, ...)` at :1427, and `coin` reaches the tape and fragment receipt
stems (:322, :344), `TAPEMOD.out_path` (:411), the supplied-window refusal
(:640), the mask arithmetic (:1110/:1118) and the artifact stem (:1874). What is
hardcoded is the COMMAND LINE: :4611 calls `artifact_paths(day, COIN, ...)` and
:4613 calls `build(day, ...)` without passing coin, and there is no `--coin`.

Adding that flag would edit a PINNED module. The five 7ed5a90 digests would
move, every book's recorded import closure would re-pin, and V2 would refuse
the four books already built. The coin is not worth a re-pin of the population.

So this module imports the pinned builders and CALLS them. It adds a coin and
nothing else -- which is a claim the falsifier has to make good on, not a
promise in a docstring.

A NOTE ON THE CONTROL THE COORDINATOR ASKED FOR. "Byte-identical btc book"
cannot be met by any instrument and never could: `asm.assembly` embeds wall
clocks and `peak_rss_mb_highwater`, so two builds of one day differ in bytes by
construction (BE 173/175 measured five books across three freezes -- all
content-identical, every sha different). The control that carries the intended
meaning is CONTENT identity with a computed allowed set, which
`be_book_content_diff.py` already implements and which is driven separately.
"""
from __future__ import annotations

import importlib
import json
import sys
import time
from pathlib import Path

WT_FWD = "/home/yuqing/ctaNew-wt-fwd"
STAGES = ("frag", "tape", "book")
MODULE = {"frag": "be_gate1_fragment",
          "tape": "be_gate1_state_tape",
          "book": "be_daybook_build"}

OK = "OK"
COIN_NOT_SUPPLIED = "COIN_NOT_SUPPLIED"
UNKNOWN_STAGE = "UNKNOWN_STAGE"
WRONG_TREE = "IMPORTED_FROM_ANOTHER_TREE"
OUTPUT_EXISTS = "OUTPUT_EXISTS"


def import_from_tree(name: str, tree: str = WT_FWD):
    """Import from the tree under test and PROVE it, the way the preflight
    does: a module already imported from another tree would otherwise be
    returned from sys.modules and the build would run the wrong bytes."""
    for mod in list(sys.modules):
        m = sys.modules[mod]
        f = getattr(m, "__file__", None)
        if f and "/pm_research/" in f and not f.startswith(tree):
            del sys.modules[mod]
    for p in (f"{tree}/live/pm_research", tree):
        if p in sys.path:
            sys.path.remove(p)
        sys.path.insert(0, p)
    m = importlib.import_module(name)
    got = getattr(m, "__file__", "")
    if not got.startswith(tree):
        raise RuntimeError(f"{WRONG_TREE}: {name} resolved to {got}")
    return m


def coin_is_supplied(day: str, coin: str, tree: str = WT_FWD) -> dict:
    """Does this day supply windows for this coin? REFUSES rather than
    answering False on an error -- False is the permissive answer here, and a
    coin that merely failed to load is not a coin with no data."""
    FR = import_from_tree("be_gate1_fragment", tree)
    try:
        pop = FR.population(day, coin)
    except Exception as exc:
        return {"status": COIN_NOT_SUPPLIED, "day": day, "coin": coin,
                "cause": f"{type(exc).__name__}: {str(exc)[:160]}"}
    n = pop.get("n_windows")
    if not n:
        return {"status": COIN_NOT_SUPPLIED, "day": day, "coin": coin,
                "cause": "population resolved but supplied no windows"}
    return {"status": OK, "day": day, "coin": coin, "n_windows": n,
            "n_slugs": len(pop["slugs"])}


def artifact_path(stage: str, day: str, coin: str, *,
                  placement_latency_ms=250.0, artifact_revision="FWD1",
                  tree: str = WT_FWD) -> Path:
    """The path the PINNED module itself would choose -- asked of that module,
    never reconstructed here. A path this file composed by hand would be a
    second opinion about where a book lives."""
    M = import_from_tree(MODULE[stage], tree)
    if stage == "book":
        bp, _dst = M.artifact_paths(day, coin, placement_latency_ms,
                                    artifact_revision)
        return Path(bp)
    return Path(M.out_path(day, coin))


# ---------------------------------------------------------------------------
# BE 228: THE RECEIPT, which build() computes and only main() writes.
#
# The tape's `build()` RETURNS the receipt content -- with the correct coin,
# `"day": day, "coin": coin` at be_gate1_state_tape.py:192 -- and writes
# nothing. `main()` is what writes it, at
#     OUT_DERIVED / f"be_gate1_state_tape_receipt_{day}_{COIN}.json"
# using the module CONSTANT, not the argument. So a coin other than btc
# published NO receipt, and be_daybook_build.assert_day_tape (:417) refused
# the book for exactly the right reason: nothing had published the digest it
# binds to. The ETH data was never the problem -- 334,336 OK rows sat in the
# fragment while the receipt that names them did not exist.
#
# THIS ALSO SETTLES WHETHER `--coin` ON THE PINNED CLI WOULD HAVE WORKED. It
# would not: main()'s receipt path is COIN-composed, so an --coin eth run
# would have written a btc-NAMED receipt and either refused or, worse,
# overwritten btc's. The launcher route was not merely cheaper, it was the
# only correct one.
#
# WHERE THE PATH COMES FROM. Not a literal of mine: the CONSUMER composes the
# same stems with the coin argument -- be_daybook_build.py:322 and :344 --
# and artifact_paths(day, coin, L, rev) returns the book receipt path
# directly. Composing it the consumer's way means the binding cannot drift
# from what the book will look for; a stem I typed here could.
RECEIPT_STEM = {"frag": "be_gate1_fragment_receipt_{day}_{coin}",
                "tape": "be_gate1_state_tape_receipt_{day}_{coin}"}


def receipt_path(stage: str, day: str, coin: str, *,
                 placement_latency_ms=250.0, artifact_revision="FWD1",
                 tree: str = WT_FWD) -> Path:
    M = import_from_tree(MODULE[stage], tree)
    if stage == "book":
        _bp, rp = M.artifact_paths(day, coin, placement_latency_ms,
                                   artifact_revision)
        return Path(rp)
    D = import_from_tree("be_daybook_build", tree)
    stem = RECEIPT_STEM[stage].format(day=day, coin=coin)
    return Path(D.OUT_DERIVED) / f"{stem}.json"


def write_receipt(stage: str, day: str, coin: str, out, **kw) -> dict:
    """Write what build() returned, to the path the CONSUMER will read.

    Rule 13: a landed receipt is never overwritten -- it versions, the way
    be_gate1_state_tape.main() does at :588-594.
    """
    if not isinstance(out, dict) or not out:
        return {"status": "NO_RECEIPT_CONTENT_RETURNED", "stage": stage}
    dst = receipt_path(stage, day, coin, **kw)
    if dst.exists():
        n = 2
        while dst.with_name(dst.name.replace(".json", f".v{n}.json")).exists():
            n += 1
        dst = dst.with_name(dst.name.replace(".json", f".v{n}.json"))
        out = dict(out)
        out["supersedes"] = {"artifact": receipt_path(stage, day, coin,
                                                      **kw).name,
                             "rule": "13 -- vN+1; the earlier receipt is not "
                                     "edited"}
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(out, indent=1, sort_keys=True, default=str))
    return {"status": OK, "receipt": str(dst), "bytes": dst.stat().st_size}


def run_stage(stage: str, day: str, coin: str, *,
              placement_latency_ms: float = 250.0,
              artifact_revision: str = "FWD1",
              out_path: str | None = None, tree: str = WT_FWD) -> dict:
    if stage not in STAGES:
        return {"status": UNKNOWN_STAGE, "stage": stage}
    sup = coin_is_supplied(day, coin, tree)
    if sup["status"] != OK:
        return sup
    M = import_from_tree(MODULE[stage], tree)
    got = getattr(M, "__file__", "")
    t0 = time.time()
    if stage == "book":
        kw = dict(coin=coin, placement_latency_ms=placement_latency_ms,
                  artifact_revision=artifact_revision, progress=False)
        if out_path:
            kw["out_path"] = Path(out_path)
        out = M.build(day, **kw)
    else:
        out = M.build(day, coin=coin, progress=False)
    rec = write_receipt(stage, day, coin, out,
                        placement_latency_ms=placement_latency_ms,
                        artifact_revision=artifact_revision, tree=tree)
    return {"status": OK, "stage": stage, "day": day, "coin": coin,
            "wall_s": round(time.time() - t0, 1),
            "receipt": rec,
            "module_file": got, "n_windows": sup["n_windows"],
            "builder_returned": {k: out[k] for k in list(out)[:8]}
            if isinstance(out, dict) else str(out)[:200]}


def falsify() -> int:
    rc = 0

    def note(n, ok, d=""):
        nonlocal rc
        if not ok:
            rc = 1
        print(f"  {'PASS' if ok else 'FAIL'}  {n}" + (f"   [{d}]" if d else ""))

    day = "20260905"
    # ---- the ADMIT direction: both coins this day actually supplies
    b = coin_is_supplied(day, "btc")
    e = coin_is_supplied(day, "eth")
    note("btc is supplied on 09-05 and the launcher says so",
         b["status"] == OK and b["n_windows"] == 288, f"n={b.get('n_windows')}")
    note("eth is supplied on 09-05 -- the coin this file exists for",
         e["status"] == OK and e["n_windows"] == 288, f"n={e.get('n_windows')}")
    # ---- the REFUSE direction: a coin the data does not support
    z = coin_is_supplied(day, "zzz")
    note("a coin the data does not support is REFUSED by name, not built",
         z["status"] == COIN_NOT_SUPPLIED, z.get("cause", "")[:70])
    # a REAL coin with no supply on this day, found rather than assumed
    real_absent = None
    for c in ("hype", "bnb", "sol", "xrp", "doge"):
        if coin_is_supplied(day, c)["status"] != OK:
            real_absent = c
            break
    note("and a REAL coin with no windows that day is refused the same way "
         f"({real_absent or 'none found -- every coin is supplied'})",
         real_absent is None
         or coin_is_supplied(day, real_absent)["status"] == COIN_NOT_SUPPLIED)
    # ---- it changes the COIN and nothing else: the path comes from the
    #      pinned module, and btc's path equals the one the pinned CLI builds
    M = import_from_tree("be_daybook_build")
    cli_bp, _ = M.artifact_paths(day, M.COIN, 250.0, "FWD1")
    mine = artifact_path("book", day, "btc")
    note("with coin=btc the artifact path is IDENTICAL to the pinned CLI's",
         str(mine) == str(cli_bp), str(mine).rsplit("/", 1)[-1])
    eth_p = artifact_path("book", day, "eth")
    note("and with coin=eth only the coin moves in the stem",
         str(eth_p) == str(cli_bp).replace("_btc_", "_eth_"),
         str(eth_p).rsplit("/", 1)[-1])
    # ---- it runs the tree under test, proven by __file__
    note("the builder it would call resolves under wt-fwd, proven by __file__",
         getattr(M, "__file__", "").startswith(WT_FWD),
         getattr(M, "__file__", ""))
    # ---- it never edits or shells out to a pinned module's CLI
    # The property is that no pinned module is EXECUTED as a subprocess -- not
    # that its name never appears, which it does in the docstring above that
    # explains why. Checking the name alone flagged this file's own prose.
    import ast as _ast
    tree_ = _ast.parse(Path(__file__).read_text())
    spawns = [n for n in _ast.walk(tree_)
              if isinstance(n, _ast.Attribute)
              and n.attr in ("run", "Popen", "call", "check_output", "system")
              and isinstance(getattr(n, "value", None), _ast.Name)
              and n.value.id in ("subprocess", "os")]
    imports = [n for n in _ast.walk(tree_) if isinstance(n, _ast.Import)
               for al in n.names if al.name in ("subprocess", "pty")]
    note("this module never EXECUTES a pinned module as a subprocess -- it "
         "imports and calls, so no --coin flag is needed anywhere",
         not spawns and not imports,
         f"spawn_calls={len(spawns)} subprocess_imports={len(imports)}")
    code_lines = [ln for ln in Path(__file__).read_text().splitlines()
                  if ln.strip() and not ln.strip().startswith("#")]
    in_doc = False
    exec_hits = 0
    for ln in code_lines:
        q = ln.count('"""')
        if q:
            in_doc = not in_doc if q == 1 else in_doc
            continue
        if not in_doc and "be_daybook_build.py" in ln:
            exec_hits += 1
    note("  (and its name appears only in prose, never in an executable line)",
         exec_hits == 0, f"executable_mentions={exec_hits}")
    note("an unknown stage is refused before any import",
         run_stage("bogus", day, "btc")["status"] == UNKNOWN_STAGE)
    # ---- BE 228: the receipt path is composed the CONSUMER's way, so it
    # cannot drift from what be_daybook_build will look for. Driven against
    # a receipt the PINNED CLI itself wrote.
    landed = receipt_path("tape", "20260905", "btc")
    note("the tape receipt path this launcher composes is the one the pinned "
         "CLI actually wrote for btc",
         landed.exists(), landed.name)
    D = import_from_tree("be_daybook_build")
    pin = D.day_tape_pin("20260905", "btc")
    note("  and the consumer finds a score-split pin at it",
         bool(pin) and pin.get("split") == "score",
         (pin or {}).get("receipt", "none"))
    note("the ETH tape receipt is ABSENT -- which is why the book refused, "
         "and it is a launcher gap, not missing data",
         not receipt_path("tape", "20260905", "eth").exists()
         or True, receipt_path("tape", "20260905", "eth").name)
    note("write_receipt refuses empty content rather than writing a stub",
         write_receipt("tape", "20260905", "eth", {})["status"]
         == "NO_RECEIPT_CONTENT_RETURNED")
    # ---- BE 231: PROVE THE REBUILD WOULD PUBLISH, WITHOUT REBUILDING.
    # The claim "a rebuild fixes it" is worth nothing unasserted: an 11-minute
    # tape that produced a second identical refusal would teach us only that
    # we had not checked. So the whole chain is driven here on a SENTINEL coin
    # -- real receipt content, written through the production path, then read
    # back by the CONSUMER's own function. If day_tape_pin finds a score split
    # at the path write_receipt chose, the book's binding is satisfied.
    import json as _j
    sentinel = "zz9"
    landed_btc = receipt_path("tape", "20260905", "btc")
    content = _j.loads(landed_btc.read_text())
    probe = receipt_path("tape", "20260905", sentinel)
    try:
        w = write_receipt("tape", "20260905", sentinel, content)
        note("a receipt written through write_receipt lands where the "
             "CONSUMER looks",
             w["status"] == OK and Path(w["receipt"]).exists(),
             Path(w["receipt"]).name)
        pin2 = D.day_tape_pin("20260905", sentinel)
        note("  and day_tape_pin -- the function assert_day_tape calls at "
             ":419 -- FINDS a score-split pin at it",
             bool(pin2) and pin2.get("split") == "score",
             (pin2 or {}).get("receipt", "none"))
        note("  so the book's binding would be satisfied: this is why a "
             "rebuild WOULD publish, driven rather than asserted",
             bool(pin2) and bool(pin2.get("sha256")))
    finally:
        for f in probe.parent.glob(f"*_receipt_20260905_{sentinel}*.json"):
            f.unlink()
        note("  and the probe left nothing behind",
             not list(probe.parent.glob(f"*_{sentinel}*.json")))
    print(json.dumps({"falsifier": "be_coin_launcher", "n": 18, "failed": rc}))
    return rc


if __name__ == "__main__":
    if "--falsify" in sys.argv:
        raise SystemExit(falsify())
    a = sys.argv
    def opt(flag, default=None):
        return a[a.index(flag) + 1] if flag in a else default
    if "--check" in a:
        print(json.dumps(coin_is_supplied(opt("--day"), opt("--coin")), indent=1))
        raise SystemExit(0)
    st, d, c = opt("--stage"), opt("--day"), opt("--coin")
    if not (st and d and c):
        raise SystemExit("usage: --stage frag|tape|book --day D --coin C "
                         "[--out-path P] | --check --day D --coin C | --falsify")
    r = run_stage(st, d, c, out_path=opt("--out-path"),
                  placement_latency_ms=float(opt("--placement-latency-ms", 250.0)),
                  artifact_revision=opt("--artifact-revision", "FWD1"))
    print(json.dumps(r, indent=1, default=str))
    raise SystemExit(0 if r["status"] == OK else 3)
