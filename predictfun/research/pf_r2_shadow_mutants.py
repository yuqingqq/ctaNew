"""MUTATION CHECK of the R2 shadow / feature / support adapter's selftest (plan v2 R2-A; 2026-11-19, session aea5bfb9): a checker that cannot fail proves nothing. Each
entry plants one code error in a COPY of pf_r2_shadow.py (text replacement, executed in a subprocess; the repo file is never touched) and runs
the copy's selftest (synthetic tapes only). Required: the unchanged copy passes, every mutant is caught BY A CHECK (the selftest's own failure
message; a crash is unexpected), and every replacement target occurs exactly as often as declared (a stale target is a failure). No market data
is read. Usage (run_light, PF_SLOT=claude): pf_branch_engine_mutants.py [index | targets]
"""
import subprocess
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent / 'pf_r2_shadow.py'
# (name, text in the source, replacement[, occurrences]) - entry 0 is the unchanged copy
MUT = [
    ('none (must pass)', None, None),
    # --- the manifest
    ('the phase from the last 8 digest bytes', "return int.from_bytes(hashlib.sha256(key).digest()[:8], 'big') % 5", "return int.from_bytes(hashlib.sha256(key).digest()[-8:], 'big') % 5"),
    ('the phase mod 4', "return int.from_bytes(hashlib.sha256(key).digest()[:8], 'big') % 5", "return int.from_bytes(hashlib.sha256(key).digest()[:8], 'big') % 4"),
    # (the byte order is NOT a mutant: 256 = 1 mod 5, so an 8-byte integer mod 5 is its byte sum mod 5 in either order - an equivalent mutant)
    ('the phase from the first 4 digest bytes', "return int.from_bytes(hashlib.sha256(key).digest()[:8], 'big') % 5", "return int.from_bytes(hashlib.sha256(key).digest()[:4], 'big') % 5"),
    ('the entry offsets ignore the phase', "return [0.] + [STEP * k for k in range(1, NEAR_END_K + 1) if k % 5 == ph]", "return [0.] + [STEP * k for k in range(1, NEAR_END_K + 1)]"),
    ('the 4,950 ms offset run, not near_end', "        if k_post >= NEAR_END_K:", "        if False:"),
    # --- the context
    ('the context from Engine truth (switch)', "CONTEXT_FROM_ENGINE = False ", "CONTEXT_FROM_ENGINE = True "),
    ('the reservations swapped', "ctx_bid_reserved_shares=float(Bk),\n             ctx_ask_reserved_shares=float(Ak))", "ctx_bid_reserved_shares=float(Ak),\n             ctx_ask_reserved_shares=float(Bk))"),
    ('the generation age from activation', "ctx_generation_age_ms=float(g - o['submit'])", "ctx_generation_age_ms=float(g - o['act'])"),
    ('the generation ordinal off by one', "ctx_generation_ordinal=float(n_side if gid is not None else n_side + 1)", "ctx_generation_ordinal=float(n_side + 1)"),
    ('the branches without the episode identity', "B = BE.Branch(path, ident=ident); nt = int(np.ceil(END / STEP)); snaps = {}; v_post = None", "B = BE.Branch(path); nt = int(np.ceil(END / STEP)); snaps = {}; v_post = None"),
    # --- sampling and the census
    ('the retain sampling with the post phase', "smp = (k % 5 == ph_r) or (k == first_live)", "smp = (k % 5 == ph_p) or (k == first_live)"),
    ('the first live review not sampled', "smp = (k % 5 == ph_r) or (k == first_live)", "smp = (k % 5 == ph_r)"),
    ('unsampled retained rows dropped', "src_rows.append(dict(rr, status='unsampled', sampled=False, **ctx)); continue", "continue"),
    ('a missing mark aborts the probes', "            src_rows.append(dict(r, status='missing_mark', sampled=True)); continue", "            raise ValueError('missing mark')"),
    # --- labels and inputs
    ('the retained target without the cancel latency', "**target(L['fills'], gid0, s, mark, g + KP.CANCEL, np.nan),", "**target(L['fills'], gid0, s, mark, g, np.nan),"),
    ('qm against the reversed reference', "r['qm'] = r['value'] - (sg * (ref - r['price']) * r['q'] if r['q'] > 0 else 0.)", "r['qm'] = r['value'] - (sg * (r['price'] - ref) * r['q'] if r['q'] > 0 else 0.)"),
    ('t_to_end restarted at the generation', "t_to_end=END - r['decision_ms']", "t_to_end=END - (r['decision_ms'] - r['activation_ms'] + KP.PLACE)"),
    ('the order inputs with a restarted terminal', "np.array([b for _, b, _ in fl], float)), np.array([d]), t0 + END)", "np.array([b for _, b, _ in fl], float)), np.array([d]), d + END)"),
    ('the weights not balanced per episode / side / head', "w[ok.values] = 1. / n", "w[ok.values] = 1."),
    ('the causal check ignores the feature comparison', "ok_ = same and fired and priv and fires and ctx_ok", "ok_ = fired and priv and fires and ctx_ok"),
    ('the truncated-tape context without the truncation', "q = BE.truncate(path, g) if cut else path;", "q = path;"),
    ('the provenance columns dropped', "    base.update(end_abs_ms=ident['end_abs_ms'], mark_source=ident['mark_source'])\n", ""),
    ('the census label reconciliation never fails', "recon['failed'] += int(not (abs(qq - r['q']) <= 1e-9", "recon['failed'] += int(False and (abs(qq - r['q']) <= 1e-9"),
    ('the census not in the manifest order', "    D = canonical(D.copy()); D['weight'] = weights(D)", "    D = D.copy(); D['weight'] = weights(D)"),
    ('the severe excess without its threshold', "return dict(q=q_, value=v_, qm=qm, severe=max(-v_ - KP.SEVERE, 0.))", "return dict(q=q_, value=v_, qm=qm, severe=max(-v_, 0.))"),
]
STALE = 97                                                # exit code of a replacement target that does not occur as declared


def source(i):
    name, a, b = MUT[i][:3]; n = (MUT[i] + (1,))[3]; s = SRC.read_text()
    if a is not None:
        if s.count(a) != n:
            print(f'mutation target occurs {s.count(a)} times, declared {n}: {name}', file=sys.stderr); sys.exit(STALE)
        s = s.replace(a, b)
    return s


def one(i):
    import types
    mod = types.ModuleType('pf_r2_shadow_mutant'); mod.__file__ = str(SRC); sys.modules['pf_r2_shadow_mutant'] = mod   # importable: the selftest pickles contexts
    exec(compile(source(i), str(SRC), 'exec'), mod.__dict__); mod.selftest()


def main():
    bad = 0
    for i, (name, *_) in enumerate(MUT):
        r = subprocess.run([sys.executable, __file__, str(i)], capture_output=True, text=True)
        tail = r.stderr.strip().splitlines() or r.stdout.strip().splitlines() or ['']
        caught = r.returncode == 1 and 'R2 SHADOW SELFTEST FAILED' in r.stderr          # caught BY A CHECK (the selftest's own exit message); a crash is unexpected
        ok = (r.returncode == 0) if i == 0 else caught
        bad += not ok
        first = next((x for x in r.stderr.splitlines() if 'SELFTEST FAILED' in x), None)
        how = 'passed' if r.returncode == 0 else ('caught: ' + first.strip()[:220]) if caught and first else ' / '.join(x.strip()[:220] for x in tail[-2:])
        print(f'{i:2d} {name}: rc {r.returncode} -> {how}{"" if ok else "   <<< UNEXPECTED"}', flush=True)
    if bad:
        sys.exit(f'R2 SHADOW MUTATION CHECK FAILED: {bad} unexpected outcomes of {len(MUT)}')
    print(f'r2 shadow mutation check passed: the unchanged copy passes its selftest, {len(MUT) - 1} / {len(MUT) - 1} planted code errors are caught by a check', flush=True)


if __name__ == '__main__':
    if sys.argv[1:] == ['targets']:
        [source(i) for i in range(len(MUT))]; print(f'all {len(MUT) - 1} mutation targets occur as declared')
    elif len(sys.argv) > 1:
        one(int(sys.argv[1]))
    else:
        main()
