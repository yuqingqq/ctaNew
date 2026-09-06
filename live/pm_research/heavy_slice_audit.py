#!/usr/bin/env python3
"""Rule 20 / R-575(C): is anything heavy running WITHOUT the lock?

At 05:54Z on 2026-09-06 two heavy scopes ran at once, one holding
`data/.heavy_run.lock` and one not. Rule 20 has no instrument behind it: every
seat is trusted to remember, and the one time it was checked the check was a
person reading `ps`. This is the check.

WHAT IT DOES
  * lists every transient scope in `research.slice` with its RSS, elapsed
    wall time and command line;
  * names the holder of `/home/yuqing/ctaNew/data/.heavy_run.lock` (via
    `fuser`, falling back to `/proc/locks`);
  * REFUSES -- rc 2, with the offenders named -- when a scope over the heavy
    bar (1 GiB RSS or 60 s wall) is running and the lock is NOT held by that
    scope's own process tree.

WHY IT KEYS ON THE SLICE AND THE PROCESS TREE
  The calling shell is itself inside a transient scope (`app.slice`), so
  "am I in a scope" is true either way and is not a test. `research.slice` is
  what the rule-20 wrapper adds and nothing else does. And a lock held by
  SOME process is not a lock held by THIS one: the holder's pid must be an
  ancestor-or-self of the heavy scope's processes, or two heavy runs are
  overlapping with one of them merely coincident with a lock.

    python3 live/pm_research/heavy_slice_audit.py            # audit, rc 0/2
    python3 live/pm_research/heavy_slice_audit.py --json
    python3 live/pm_research/heavy_slice_audit.py --selftest # falsifiers
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

LOCK = Path("/home/yuqing/ctaNew/data/.heavy_run.lock")
SLICE = "research.slice"
HEAVY_RSS_KIB = 1024 * 1024          # 1 GiB, rule 20
HEAVY_WALL_S = 60.0                  # 60 s, rule 20
RC_OK, RC_REFUSE, RC_ERROR = 0, 2, 3


def _run(cmd: list[str]) -> str:
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=20)
        return r.stdout
    except Exception:                                         # noqa: BLE001
        return ""


def scope_pids() -> dict[str, list[int]]:
    """pids grouped by the transient scope they sit in, from their cgroups.

    Read from /proc rather than from `systemctl`, so a scope whose unit has
    already been reaped but whose processes are alive is still seen.
    """
    out: dict[str, list[int]] = {}
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            cg = (entry / "cgroup").read_text()
        except OSError:
            continue
        m = re.search(r"/([^/\n]+\.slice)/([^/\n]+\.scope)", cg)
        if not m:
            continue
        if m.group(1) != SLICE:
            continue
        out.setdefault(m.group(2), []).append(int(entry.name))
    return out


def proc_info(pid: int) -> dict:
    try:
        status = (Path("/proc") / str(pid) / "status").read_text()
        rss = 0
        for line in status.splitlines():
            if line.startswith("VmRSS:"):
                rss = int(line.split()[1])
        cmd = (Path("/proc") / str(pid) / "cmdline").read_bytes()
        cmd = cmd.replace(b"\0", b" ").decode(errors="replace").strip()
        st = (Path("/proc") / str(pid) / "stat").read_text().rsplit(") ", 1)[1]
        starttime = int(st.split()[19]) / os.sysconf("SC_CLK_TCK")
        uptime = float(Path("/proc/uptime").read_text().split()[0])
        return {"pid": pid, "rss_kib": rss, "cmd": cmd,
                "elapsed_s": round(uptime - starttime, 1)}
    except (OSError, IndexError, ValueError):
        return {"pid": pid, "rss_kib": 0, "cmd": "<gone>", "elapsed_s": 0.0}


def lock_holders(lock: Path = LOCK) -> list[int]:
    """Pids holding the heavy-run lock. `fuser` first, /proc/locks as the
    fallback so the instrument does not depend on psmisc being installed."""
    txt = _run(["fuser", str(lock)])
    pids = [int(x) for x in re.findall(r"\d+", txt)]
    if pids:
        return sorted(set(pids))
    try:
        st = lock.stat()
    except OSError:
        return []
    want = f"{os.major(st.st_dev):02x}:{os.minor(st.st_dev):02x}:{st.st_ino}"
    out = []
    try:
        for line in Path("/proc/locks").read_text().splitlines():
            f = line.split()
            if len(f) >= 6 and f[5] == want:
                out.append(int(f[4]))
    except OSError:
        pass
    return sorted(set(out))


def ancestors(pid: int) -> set[int]:
    seen, cur = set(), pid
    while cur and cur not in seen:
        seen.add(cur)
        try:
            st = (Path("/proc") / str(cur) / "stat").read_text()
            cur = int(st.rsplit(") ", 1)[1].split()[1])
        except (OSError, IndexError, ValueError):
            break
    return seen


def audit(heavy_rss_kib: int = HEAVY_RSS_KIB,
          heavy_wall_s: float = HEAVY_WALL_S,
          lock: Path = LOCK) -> dict:
    holders = lock_holders(lock)
    holder_trees: set[int] = set()
    for h in holders:
        holder_trees |= ancestors(h)
    scopes = []
    offenders = []
    for scope, pids in sorted(scope_pids().items()):
        infos = [proc_info(p) for p in pids]
        rss = max((i["rss_kib"] for i in infos), default=0)
        wall = max((i["elapsed_s"] for i in infos), default=0.0)
        heavy = rss >= heavy_rss_kib or wall >= heavy_wall_s
        # The lock must be held BY THIS SCOPE's own tree, not by anyone.
        locked = any(p in holder_trees or (set(ancestors(p)) & set(holders))
                     for p in pids)
        rec = {"scope": scope, "pids": pids, "max_rss_kib": rss,
               "max_elapsed_s": wall, "is_heavy": heavy,
               "lock_held_by_this_scope": bool(locked),
               "cmds": [i["cmd"][:200] for i in infos]}
        scopes.append(rec)
        if heavy and not locked:
            offenders.append(rec)
    refused = bool(offenders)
    return {
        "instrument": "live/pm_research/heavy_slice_audit.py",
        "rule": "SEAT_PROTOCOL rule 20 / R-575(C)",
        "as_of_unix": time.time(),
        "slice": SLICE,
        "heavy_bar": {"rss_kib": heavy_rss_kib, "wall_s": heavy_wall_s},
        "lock_path": str(lock),
        "lock_holders": holders,
        "n_scopes_in_slice": len(scopes),
        "scopes": scopes,
        "refused": refused,
        "offenders": offenders,
        "verdict": ("REFUSED: "
                    + "; ".join(f"{o['scope']} is heavy "
                                f"({o['max_rss_kib']} KiB, "
                                f"{o['max_elapsed_s']:.0f} s) and does NOT "
                                f"hold {lock}" for o in offenders)
                    if refused else
                    f"ok: {len(scopes)} scope(s) in {SLICE}, none heavy "
                    f"without the lock"),
    }


# --------------------------------------------------------------------------
def selftest() -> int:                                        # noqa: C901
    """Both directions, on PLANTED scopes -- and every assertion is scoped to
    the scope this test planted, BY NAME.

    The first version of this selftest asserted on the audit's global
    `refused` flag. It "passed" its known-bad by firing on ANOTHER SEAT's
    real heavy run that happened to be in the slice, and then failed its
    positive control for the same reason. A control that passes because of
    something it did not plant is rule 16's shape exactly, so each planted
    scope is now given an explicit `--unit` and the assertions name it.
    """
    fails = []

    def ok(c, m):
        print(("ok   " if c else "FAIL ") + m)
        if not c:
            fails.append(m)

    tag = f"hsaself{os.getpid()}"
    tmp = Path("/tmp") / f"{tag}.lock"
    tmp.touch()

    def plant(unit: str, with_lock: bool, hold_s: int = 25):
        inner = (f"exec flock -n {tmp} sleep {hold_s}" if with_lock
                 else f"exec sleep {hold_s}")
        p = subprocess.Popen(
            ["systemd-run", "--user", "--scope", f"--slice={SLICE}",
             f"--unit={unit}", "-p", "MemoryMax=8G", "-p", "CPUQuota=100%",
             "bash", "-c", inner],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(2.0)
        return p

    def unplant(p, unit, timeout=15.0):
        """Stopping the CLIENT is not stopping the SCOPE. `systemd-run
        --scope` leaves the transient unit and its children running when the
        launching process is terminated; the first version of this teardown
        did exactly that and two later controls failed because the planted
        scope was still there holding the lock."""
        subprocess.run(["systemctl", "--user", "stop", f"{unit}.scope"],
                       capture_output=True, timeout=timeout)
        try:
            p.terminate(); p.wait(timeout=5)
        except Exception:                                     # noqa: BLE001
            pass
        deadline = time.time() + timeout
        while time.time() < deadline:
            if not [x for x in scope_pids() if x.startswith(unit)]:
                return True
            time.sleep(0.5)
        return False

    def mine(rep, unit):
        return [o for o in rep["offenders"] if o["scope"].startswith(unit)]

    def seen(rep, unit):
        return [s for s in rep["scopes"] if s["scope"].startswith(unit)]

    # Planted scopes are young and tiny, so the bar is LOWERED for the
    # controls (rule 8 permits lowering for attributability; never raising).
    bar_w, bar_r = 0.5, 0

    u_bad = f"{tag}bad"
    p_bad = plant(u_bad, with_lock=False)
    a_bad = audit(heavy_rss_kib=bar_r, heavy_wall_s=bar_w, lock=tmp)
    ok(len(mine(a_bad, u_bad)) == 1 and a_bad["refused"],
       f"KNOWN-BAD: the scope I planted WITHOUT the lock is named as an "
       f"offender -- {[o['scope'] for o in mine(a_bad, u_bad)]} "
       f"(total offenders {len(a_bad['offenders'])}, which may include "
       f"another seat's run; this assertion is on MINE)")
    gone_bad = unplant(p_bad, u_bad)

    u_good = f"{tag}good"
    p_good = plant(u_good, with_lock=True)
    a_good = audit(heavy_rss_kib=bar_r, heavy_wall_s=bar_w, lock=tmp)
    g_seen, g_off = seen(a_good, u_good), mine(a_good, u_good)
    ok(len(g_seen) == 1 and len(g_off) == 0
       and g_seen[0]["lock_held_by_this_scope"] and a_good["lock_holders"],
       f"POSITIVE CONTROL: the scope I planted HOLDING the lock is seen "
       f"({len(g_seen)}), is heavy under the lowered bar "
       f"({g_seen[0]['is_heavy'] if g_seen else None}) and is NOT an "
       f"offender -- so the audit can ADMIT, holders "
       f"{a_good['lock_holders']}")
    gone_good = unplant(p_good, u_good)

    a_idle = audit(heavy_rss_kib=bar_r, heavy_wall_s=bar_w, lock=tmp)
    ok(gone_bad and gone_good and not seen(a_idle, tag),
       f"POSITIVE CONTROL: once my scopes are STOPPED the audit no longer "
       f"sees them -- {len(a_idle['scopes'])} scope(s) remain in the slice "
       f"and {len(seen(a_idle, tag))} of them are mine")

    ok(lock_holders(tmp) == [],
       "LOCK READ: an unheld lock reports NO holders -- read, not assumed")
    ok(audit(lock=Path("/nonexistent/x.lock"))["lock_holders"] == [],
       "LOCK READ: a lock path that does not exist reports no holders "
       "rather than raising")
    ok(RC_REFUSE != RC_OK and RC_REFUSE == 2,
       f"THE REFUSAL IS A NAMED RETURN CODE ({RC_REFUSE}), not a printed "
       f"word: a caller can gate on it")
    try:
        tmp.unlink()
    except OSError:
        pass
    print(f"\n{'selftest OK' if not fails else 'SELFTEST FAILED'} -- "
          f"{len(fails)} failure(s)")
    return 1 if fails else 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    try:
        rep = audit()
    except Exception as exc:                                  # noqa: BLE001
        print(f"AUDIT ERROR: {exc}", file=sys.stderr)
        return RC_ERROR
    if a.json:
        print(json.dumps(rep, indent=2, sort_keys=True))
    else:
        print(f"lock {rep['lock_path']}  holders={rep['lock_holders']}")
        for sc in rep["scopes"]:
            print(f"  {sc['scope']}  rss={sc['max_rss_kib']} KiB  "
                  f"{sc['max_elapsed_s']:.0f} s  heavy={sc['is_heavy']}  "
                  f"locked={sc['lock_held_by_this_scope']}")
            for c in sc["cmds"][:2]:
                print(f"      {c[:150]}")
        print(rep["verdict"])
    return RC_REFUSE if rep["refused"] else RC_OK


if __name__ == "__main__":
    raise SystemExit(main())
