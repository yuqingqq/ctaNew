import ast, subprocess, sys, pathlib
SRC = pathlib.Path('/home/yuqing/ctaNew/live/pm_research/v5_boundary_preflight.py')
src = SRC.read_text()
tree = ast.parse(src)
CHECKERS = {"check_boundary_current", "installed_mode", "classify_era_row",
            "current_era_and_open_v5", "check_pre_arm", "check_post_restart",
            "check_counters", "check_post_rollback",
            "check_runbook_consistency"}
raises = []
for fn in ast.walk(tree):
    if isinstance(fn, ast.FunctionDef) and fn.name in CHECKERS:
        for node in ast.walk(fn):
            if isinstance(node, ast.Raise):
                raises.append((fn.name, node.lineno, node.end_lineno))
print(f"{len(raises)} refusal sites in {len(CHECKERS)} checkers")
lines = src.splitlines()
survivors = []
for name, lo, hi in raises:
    mutated = lines[:]
    indent = len(lines[lo-1]) - len(lines[lo-1].lstrip())
    for i in range(lo-1, hi):
        mutated[i] = " " * indent + "pass  # MUTATED"
    scratch = pathlib.Path('mut_preflight.py')
    scratch.write_text("\n".join(mutated))
    r = subprocess.run([sys.executable, str(scratch), "--selftest"],
                       capture_output=True, text=True, timeout=120)
    if r.returncode == 0:
        survivors.append((name, lo))
        print(f"  SURVIVOR: {name}:{lo} — suite GREEN with this refusal blanked")
print(f"killed {len(raises)-len(survivors)}/{len(raises)}; survivors: {survivors or 'NONE'}")
