#!/usr/bin/env python3
"""Contract-inventory diff for PM_ARCHITECTURE.md (§13).

v7's tool extracted every CamelCase identifier from the WHOLE document, which is
insufficient (review SHOULD-FIX 1): old identifiers linger in explanatory prose,
so a contract can vanish from the canonical schema while the diff still reports
it "present". It also mis-dated the records-block drop as v5->v6 when it was
v4->v5.

This version inventories ONLY canonical fenced blocks -- the schemas, keys and
rule table -- and ignores prose entirely.

  python3 contract_inventory.py <git-ref> [<git-ref-or-worktree>]
"""
import re, subprocess, sys

PATH = "live/pm_research/PM_ARCHITECTURE.md"

def load(ref):
    if ref in ("", "WORKTREE"):
        return open(PATH).read()
    return subprocess.run(["git", "show", f"{ref}:{PATH}"],
                          capture_output=True, text=True).stdout

def inventory(text):
    """Canonical surface = fenced code blocks + the R-* rule table rows."""
    body = re.sub(r'^## 13\..*?(?=^## \d+\.)', '', text, flags=re.S | re.M)
    fenced = "\n".join(re.findall(r'```(.*?)```', body, flags=re.S))
    fenced = re.sub(r'#.*', '', fenced)                    # strip block comments
    out = set()
    # type / record declarations:  Name{ ... }   or   Name = ...   or  Name -> ...
    out |= {f"type:{m}" for m in re.findall(r'\b([A-Z][A-Za-z0-9_-]*[A-Za-z0-9_])\s*[\{=]', fenced)}
    out |= {f"type:{m}" for m in re.findall(r'\b([A-Z][A-Za-z0-9_]*)\s*->', fenced)}
    # module identifiers
    out |= {f"mod:{a}-{b}" for a, b in re.findall(r'\b(SP|DA|BE|DE|EV|OP)-([A-Za-z]+)', fenced)}
    # fields declared inside braces
    out |= {f"field:{m}" for m in re.findall(r'[\{,]\s*([a-z_][a-z0-9_]*)\s*:', fenced)}
    # rules from the table (canonical, not prose)
    out |= {f"rule:{m}" for m in re.findall(r'\*\*(R-[A-Z]+)\*\*', text)}
    return out

if __name__ == "__main__":
    a = inventory(load(sys.argv[1]))
    b = inventory(load(sys.argv[2] if len(sys.argv) > 2 else "WORKTREE"))
    drop, add = sorted(a - b), sorted(b - a)
    print(f"DROPPED ({len(drop)}):")
    print("  " + ("\n  ".join(drop) if drop else "none"))
    print(f"ADDED ({len(add)}):")
    print("  " + ("\n  ".join(add) if add else "none"))
    sys.exit(1 if drop else 0)
