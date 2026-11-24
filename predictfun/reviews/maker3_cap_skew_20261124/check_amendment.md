# Checker CSV amendment, before acceptance

check_v2 stopped after12 successful ledger checks because pandas compares CSV missing clean=NaN to in-memory clean=None as unequal. check_v3 canonically compares missing with missing and explicitly parses boolean strings; unknown is not False. Same all-field equality and1e-12 absolute numeric tolerance. Seven hand assertions include a missing-vs-False falsifier. Other invariants and prior independent context reconciliation unchanged. Original check.log retained; no observed data or replay changes.

Also record a source-population clarification discovered by retaining unknowns: the source contains34 rank3 signed hashes, including one incomplete taker-only45.2805-share hash outside the posting census. The previously described33 are pure-maker quote hashes (16x10/17x220). The additional hash remains unknown in the outputs and is NOT evidence of a45.2805-share passive quote. No data-selected exclusion or new candidate.
