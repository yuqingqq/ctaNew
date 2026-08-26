"""Sample a cgroup's memory + system-wide pressure to disk, once a second.

WHY THIS EXISTS. The 2026-08-26 03:55 UTC box stop could not be attributed,
because the only memory evidence was a human polling `systemctl show` by hand
and the kernel log did not survive the reboot. This writes the trajectory to
DISK AS IT GOES (line-buffered, fsync every 10 lines), so the record survives
a hard stop. It samples from OUTSIDE the measured job and touches nothing the
job does -- a reproduction stays bit-identical while being watched.

It records what would have settled the question: cgroup current/peak, the
system MemAvailable, and /proc/pressure/memory (PSI). PSI `some avg10` rising
toward 100 is the swapless-livelock signature; a clean kill with flat PSI is
an ordinary cgroup OOM; both flat right up to the last line points OUTSIDE
the box entirely.
"""
from __future__ import annotations

import sys, time
from pathlib import Path

CG = Path("/sys/fs/cgroup")


def read(p: Path) -> str:
    try:
        return p.read_text().strip()
    except OSError:
        return ""


def cgroup_dir(unit: str) -> Path | None:
    uid = os.getuid() if (os := __import__("os")) else 1000
    for c in (CG / f"user.slice/user-{uid}.slice/user@{uid}.service/app.slice/{unit}.service",
              CG / f"user.slice/user-{uid}.slice/{unit}.service"):
        if c.exists():
            return c
    return None


def main() -> int:
    if len(sys.argv) < 3:
        print("usage: mem_trace.py <unit-name> <out.tsv> [interval_s]")
        return 2
    unit, out = sys.argv[1], Path(sys.argv[2])
    iv = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0
    cg = cgroup_dir(unit)
    fh = out.open("w", buffering=1)
    fh.write("#ts\tcg_current\tcg_peak\tmem_available_kb\tpsi_some_avg10\tpsi_full_avg10\n")
    n = 0
    while True:
        cur = read(cg / "memory.current") if cg else ""
        pk = read(cg / "memory.peak") if cg else ""
        avail = ""
        for line in read(Path("/proc/meminfo")).splitlines():
            if line.startswith("MemAvailable:"):
                avail = line.split()[1]; break
        some = full = ""
        for line in read(Path("/proc/pressure/memory")).splitlines():
            if line.startswith("some"):
                some = line.split("avg10=")[1].split()[0]
            elif line.startswith("full"):
                full = line.split("avg10=")[1].split()[0]
        fh.write(f"{time.time():.1f}\t{cur}\t{pk}\t{avail}\t{some}\t{full}\n")
        n += 1
        if n % 10 == 0:
            import os as _os
            fh.flush(); _os.fsync(fh.fileno())
        if cg and not cg.exists():
            fh.write(f"# cgroup vanished at {time.time():.1f} (job ended)\n"); break
        time.sleep(iv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
