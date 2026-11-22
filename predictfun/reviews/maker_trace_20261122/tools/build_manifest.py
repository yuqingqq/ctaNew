"""FREEZE MANIFEST of MAKER-TRACE-RECONSTRUCTION-v1 (session aea5bfb9, 2026-11-22): sha256 of every source the run executes or reads as a
rule (modules, selftests, common, contract, declaration, launcher, tools, checker), of the input manifests (BOOK list, GT scope, step1
registries Oct 23-27, the pre-built truth table), the USER approvals, populations, scenarios and the declared reading thresholds. Write-once.
Usage: build_manifest.py"""
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, '/home/yuqing/ctaNew/predictfun/research')
import pf_mt_common as C   # noqa: E402

R = C.REPO; D = C.RECEIPT; OUTP = D / 'manifest.json'
SOURCES = ['predictfun/research/pf_mt_common.py'] + [f'predictfun/research/pf_mt_{m}{s}.py' for m in ('extract', 'lifecycle', 'econ', 'truth', 'report') for s in ('', '_selftest')] + \
    [f'predictfun/reviews/maker_trace_20261122/{f}' for f in ('contract.md', 'declaration.md', 'launch_mt.sh', 'mt_checks.py', 'tools/rehearse_checks.py',
                                                                'tools/cpu_ledger.py', 'tools/project.py', 'tools/build_manifest.py', 'tools/rehearse_e2e.py')]


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    C.need(not OUTP.exists(), f'{OUTP} exists (write-once)')
    book_p = R / 'predictfun/reviews/ladder_async_fast_20261122/manifest.json'; scope_p = Path('/home/yuqing/pmmm-data/review-exports/live-samewindow-20261108/scope.json')
    book = json.loads(book_p.read_text())['complete']
    man = dict(version=C.VERSION, task='MAKER-TRACE-RECONSTRUCTION-v1', plan=['8b39f9e8', '9ff1e197'], contract_commit='5927971d',
               sources={s: sha(R / s) for s in SOURCES},
               inputs={'book_manifest': dict(path=str(book_p.relative_to(R)), sha256=sha(book_p)), 'gt_scope': dict(path=str(scope_p), sha256=sha(scope_p)),
                       **{f'registry_{d}': dict(path=str((C.STEP1 / d / 'registry.parquet').relative_to(R)), sha256=sha(C.STEP1 / d / 'registry.parquet')) for d in (C.ID_DAY,) + C.MEASURE_DAYS},
                       'truth_table': dict(path=str((C.OUT / 'truth' / 'v2' / 'own_orders.parquet').relative_to(R)), sha256=sha(C.OUT / 'truth' / 'v2' / 'own_orders.parquet'),
                                           note='v2 (per-run clock bands), built before the freeze; the run refuses unless the frozen code reproduces it (dry gate)'),
                       'truth_table_v1_provenance': dict(path=str((C.OUT / 'truth' / 'own_orders.parquet').relative_to(R)), sha256=sha(C.OUT / 'truth' / 'own_orders.parquet'),
                                                         note='v1 (pooled clock band), superseded by v2, kept unedited')},
               populations=dict(ID=C.ID_DAY, FILL=list(C.MEASURE_DAYS), BOOK={d: sorted(int(m) for m in v) for d, v in book.items()},
                                GT=dict(day=C.GT_DAY, rule='scope windows with a maker leg of our signer 0xba14f115ccf30e976f7639594fdee7b36c85ced8 (public legs)')),
               scenarios=C.SCENARIOS, classes=list(C.CLASSES), removal=list(C.REMOVAL),
               readings=dict(R1=dict(tests=['T1_price_priority', 'T2_time_priority'], min_denominator=200, supported_max_rate=0.01),
                             R2=dict(defect='a failed observed / constrained claim', reading=['no_defect', 'defective']),
                             R3='coverage fractions, no threshold', R4=dict(no_verdict=True, sign_consistent='same sign on all 4 days', intervals='none (4 < 5 days)')),
               approval=dict(first=dict(core_hours=0.90, line_ceiling=4.40), topup=dict(core_hours=1.35, line_ceiling=4.85), reserve_h=0.05, via='AskUserQuestion 2026-11-22'),
               caps=dict(guard_mib=C.GUARD_MIB, checkpoint_mib=1536, hard_mib=2048), multiplicity=0)
    OUTP.write_text(json.dumps(man, indent=1, sort_keys=True) + '\n'); print(json.dumps(dict(sources=len(man['sources']), book=sum(len(v) for v in book.values())), indent=1))


if __name__ == '__main__':
    main()
