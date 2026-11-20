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
    ('the stored row never compared with a same-identity rebuild (d10d3698)',
     "base = rebuild_row(path, W, fv, o, day, r, fair_m, depth, feat, market=mk0); bad_i = same_f32(r, base, cols) if base is not None else ['no grid index']",
     "base = rebuild_row(path, W, fv, o, day, r, fair_m, depth, feat, market=mk0); bad_i = []"),
    # --- review 4cd7b45f: the client-visibility control, the audit sample, the multiplicity guards, the bounded reader
    ('the visibility component disabled (4cd7b45f)', "bad_v = same_f32(r, vis, cols) + (['observation'] if ob_h != ob0 else [])", "bad_v = []"),
    ('no unreceived frame perturbed (4cd7b45f)', "rows = np.arange(E.known, E.known + 1) if delivered else np.arange(E.known + 1, len(f))",
     "rows = np.arange(E.known, E.known + 1) if delivered else np.arange(0)"),
    ('the engine-truth queue ahead left as it is (4cd7b45f)', "o_['rem'] = o_['rem'] * .5 + 1.; o_['ahead'] = o_['ahead'] * 1.7 + 3.", "o_['rem'] = o_['rem'] * .5 + 1."),
    ('the delivered positive control perturbs nothing (4cd7b45f)', "rows = np.arange(E.known, E.known + 1) if delivered else np.arange(E.known + 1, len(f))",
     "rows = np.arange(0) if delivered else np.arange(E.known + 1, len(f))"),
    ('the audit count ceil(n / 32) (4cd7b45f)', "out.extend(g_[:min(len(g_), max(1, math.ceil(len(g_) / CAUSAL_DIV)))])", "out.extend(g_[:min(len(g_), max(1, math.ceil(len(g_) / 32)))])"),
    ('the audit sample in row order, not by hash (4cd7b45f)',
     "g_ = sorted(strata[k_], key=lambda r: (hashlib.sha256(causal_key(r).encode('utf-8')).hexdigest(), causal_key(r)))", "g_ = list(strata[k_])"),
    ('the audit key without its version (4cd7b45f)', "return json.dumps([CAUSAL_VERSION, str(r['day']),", "return json.dumps([str(r['day']),"),
    ('a repeated source pair accepted (4cd7b45f)', "C.require(m not in yielded, f'partition {pi}: the source yielded the pair of market {m} twice')", "pass"),
    ('two status rows for one original accepted (4cd7b45f)', "C.require(all((n_st.get(k, 0) == 1 and n_ep.get(k, 0) == 0) or (n_st.get(k, 0) == 0 and n_ep.get(k, 0) > 0) for k in want_),",
     "C.require(True or all((n_st.get(k, 0) == 1 and n_ep.get(k, 0) == 0) or (n_st.get(k, 0) == 0 and n_ep.get(k, 0) > 0) for k in want_),"),
    ('duplicate canonical identities accepted (4cd7b45f)', "C.require(not D.duplicated(subset=[c for c in SORT_KEY if c in D]).any(),", "C.require(True,"),
    ('the all-part reader unbounded (4cd7b45f)', "n = sum(r for _, r in files); C.require(n <= lim,", "n = 0; C.require(n <= lim,"),
    ('the perturbed rebuild skipped (d10d3698)', "bad_ii = same_f32(r, pert, cols) if pert is not None else None", "bad_ii = []"),
    ('an unbuildable future counted as passing (census 22:15Z)', "bad_ii = same_f32(r, pert, cols) if pert is not None else None",
     "bad_ii = same_f32(r, pert, cols) if pert is not None else []"),
    ('no retry of an unbuildable future draw (census 22:15Z)', "for i_ in range(VENUE_DRAWS):", "for i_ in range(1):"),
    ('the future fair path not perturbed (d10d3698)', "fvp = np.array(fv, float).copy(); fvp[k_ep + 1:] += 7.", "fvp = np.array(fv, float).copy()"),
    ('the float32 comparison ignores the inputs (d10d3698)', "    return [c for c in cols if not ((np.isnan(np.float32(stored[c])) and np.isnan(np.float32(rebuilt[c]))) or np.float32(stored[c]) == np.float32(rebuilt[c]))]",
     "    return [c for c in cols if c in CTX and not ((np.isnan(np.float32(stored[c])) and np.isnan(np.float32(rebuilt[c]))) or np.float32(stored[c]) == np.float32(rebuilt[c]))]"),
    ('the missing-market status not emitted (d10d3698)', "        rows.append(status_row(day, o, o.m, 'missing_source:market'))", "        pass"),
    ('the missing-pair status not emitted (d10d3698)', "        rows.append(status_row(day, o, o.m, 'missing_source:pair'))", "        pass"),
    ('the partition reconciliation not enforced (d10d3698)', "    C.require(set(n_st) | set(n_ep) == want_, f'partition {pi}:", "    C.require(True, f'partition {pi}:"),
    ('the rebuild calls the builder with the focal side only (rehearsal 21:35Z)', "    ch = {0: [], 1: []}; ch[s] = [kk]", "    ch = {s: [kk]}"),
    ('the engine world cut on the receipt clock (rehearsal 21:52Z)', "jv = int(np.searchsorted(np.asarray(w['tf'], float), d, side='right') - 1)",
     "jv = int(np.searchsorted(np.asarray(w['kf'], float), d, side='right') - 1)"),
    ('the unnotified trades not halved (rehearsal 21:52Z)', "tr[tr[:, 0] > g - BE.FILL_NOTICE, 3] *= .5", "tr[tr[:, 0] > g + 1e12, 3] *= .5"),
    ('only the trades after the decision halved (rehearsal 21:52Z)', "tr[tr[:, 0] > g - BE.FILL_NOTICE, 3] *= .5", "tr[tr[:, 0] > g, 3] *= .5"),
    ('the joint POST refusal outcome dropped (d10d3698)', "ref_kw = {'refused': bool(gen_f['refused'])} if head == 'post' else {}", "ref_kw = {'refused': False} if head == 'post' else {}"),
    ('retain rows carry the POST refusal flag (d10d3698)', "ref_kw = {'refused': bool(gen_f['refused'])} if head == 'post' else {}", "ref_kw = {'refused': bool(gen_f['refused'])}"),
    ('unseen declared categories never listed (d10d3698)', "unseen = {sh_: [f'{d_}={v_}' for d_ in DIMS for v_ in DECLARED[d_] if pres[sh_][d_][v_] == 0] for sh_ in SIDE_HEADS}",
     "unseen = {sh_: [] for sh_ in SIDE_HEADS}"),
    ('the live fallback WAIT (d10d3698)', "FALLBACK = {'post': 'WAIT', 'retain': 'KEEP'}", "FALLBACK = {'post': 'WAIT', 'retain': 'WAIT'}"),
    ('unobserved declared cells omitted (d10d3698)',
     "            r = got.get((sh_, str(iv))); n_, e_, z_ = (int(r.rows), int(r.originals), int(r.nonzero_originals)) if r is not None else (0, 0, 0)",
     "            r = got.get((sh_, str(iv)))\n            if r is None:\n                continue\n            n_, e_, z_ = (int(r.rows), int(r.originals), int(r.nonzero_originals))"),
    ('the inventory sign without its tolerance (d10d3698)', "ok['c_inv'] = np.select([inv > INV_TOL, inv < -INV_TOL], ['long', 'short'], 'flat')", "ok['c_inv'] = np.select([inv > 0, inv < 0], ['long', 'short'], 'flat')"),
    ('the support ranges pooled across sides / heads (d10d3698)', "        rng[sh_] = {c: [float(g_[g_.field == c].lo.min()), float(g_[g_.field == c].hi.max())] for c in NUMERIC}",
     "        rng[sh_] = {c: [float(rg[rg.field == c].lo.min()), float(rg[rg.field == c].hi.max())] for c in NUMERIC}"),
    ('the weights over the whole census instead of per episode (d10d3698)', "n = D[ok].groupby(['original_episode_id', 'side', 'head']).original_episode_id.transform('size').values", "n = np.full(int(ok.sum()), float(ok.sum()))"),
    ('the truncated-tape context without the truncation', "q = BE.truncate(path, g) if cut else path;", "q = path;"),
    ('the provenance columns dropped', "    base.update(end_abs_ms=ident['end_abs_ms'], mark_source=ident['mark_source'])\n", ""),
    ('the census label reconciliation never fails', "recon['failed'] += int(not (abs(qq - r['q']) <= 1e-9", "recon['failed'] += int(False and (abs(qq - r['q']) <= 1e-9"),
    ('the census not in the manifest order', "    D = canonical(D.copy())\n    ks = list(", "    D = D.copy()\n    ks = list("),
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
