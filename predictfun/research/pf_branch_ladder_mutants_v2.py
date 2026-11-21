"""MUTATION CHECK v2 of pf_branch_ladder (review 0065191f finding 6): the frozen v1 harness's planted errors, unchanged, judged by pf_ladder_rest_selftest_v2
(the frozen suite under explicit guard conditions). Required: the control passes; every mutant is caught BY A CHECK."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pf_branch_ladder_mutants as V1                                # noqa: E402  the frozen harness (unchanged)

SRC, MUTANTS, FAIL = V1.SRC, V1.MUTANTS, V1.FAIL
SELFTEST = 'pf_ladder_rest_selftest_v2'
