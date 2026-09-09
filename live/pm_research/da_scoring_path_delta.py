"""The payload a refusal should carry, timed end to end."""
import ast, hashlib, pathlib, subprocess, sys, time, re
from collections import deque
t0 = time.time()
ROOT = pathlib.Path("/home/yuqing/ctaNew/live/pm_research")
MOD, BOOK_REF = "de_multiday_gate1_runner", "69123fd"
sys.path.insert(0, str(ROOT))
import be_producing_closure as PC
# 1. the reachable scoring set
sc = sorted(PC.expected_set_from_disk(ROOT, PC.SCORING_ENTRY_POINTS)["modules"])
# 2. within the offending module, the functions the scoring path can reach
src = (ROOT / (MOD + ".py")).read_text()
tree = ast.parse(src)
defs = {}
for n in ast.walk(tree):
    if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
        defs.setdefault(n.name, n)
# entry into this module, found by operation: what the front door calls on it
fd = (ROOT / "de_phase4_diag_runner.py").read_text()
entries = {n.attr for n in ast.walk(ast.parse(fd))
           if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name)
           and n.value.id in ("_G",)}
seen = set(entries); q = deque(entries)
while q:
    m = q.popleft()
    for x in ast.walk(defs.get(m) or ast.Module(body=[], type_ignores=[])):
        if isinstance(x, ast.Call):
            f = x.func
            c = f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", None)
            if c in defs and c not in seen:
                seen.add(c); q.append(c)
# 3. byte-identity of each on-path function against the book's version
old = subprocess.run(["git","-C","/home/yuqing/ctaNew","show",
                      f"{BOOK_REF}:live/pm_research/{MOD}.py"],
                     capture_output=True, text=True).stdout
def fn_src(s, name):
    t = ast.parse(s); L = s.split("\n")
    for n in ast.walk(t):
        if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.name==name:
            return "\n".join(L[n.lineno-1:n.end_lineno])
ident = {f: (hashlib.sha256((fn_src(old,f) or "").encode()).hexdigest()
             == hashlib.sha256((fn_src(src,f) or "").encode()).hexdigest())
         for f in sorted(seen)}
el = time.time() - t0
print(f"scoring set: {len(sc)} modules")
print(f"on-path functions in {MOD}: {sorted(seen)}")
print(f"byte-identical vs the book's version: {ident}")
print(f"ALL ON-PATH FUNCTIONS UNCHANGED: {all(ident.values())}")
print(f"ELAPSED: {el:.3f} s")
