# Instrument amendment1, before maker observation outputs

The frozen observe.py stopped before writing any outputs: some of the33 signed maker3 orders lack a posting-census market. Casting that unknown mid to int raised ValueError (observe.log retained). observe_v2.py retains mid=None and the existing unknown status, preserving every order and both clocks. Nothing about the population, labels, rules or replay changes. selftest_v2 adds the missing-market case. check_v2 runs the frozen checker with this corrected observation function and adds reconciliation to the independently saved prior state map. The replay's12 arms were already generated and are unchanged.

All new files bound in amendment.json and committed before v2 observations. No threshold or output selection. Previous logs counted. Source v1 and manifest remain untouched.
