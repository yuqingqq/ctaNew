# Exact full-fill endpoint in the checker

v3 subsequently refused the order lifetime: it reconstructed exact full-fill completion from six-decimal display quantities and a 1e-8-share threshold. v4 computes this exact event from the unrounded raw ledger, then compares the displayed endpoint (six-decimal seconds). Per-fill quantities still reconcile individually. Other source checks/falsifiers are unchanged; neither the extractor nor the data changes. This is a precision boundary in the checker, not a new outcome exclusion. Source/lifecycle evidence remains authoritative.
