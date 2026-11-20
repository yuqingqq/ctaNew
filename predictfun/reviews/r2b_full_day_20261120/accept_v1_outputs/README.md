# v1 acceptance outputs, kept for the byte-equality claim (added 2026-11-20 with the pi0-50 receipt; review 3d7eae6a's provenance note)

**What these files are.** The nine CSV / `records_full.sha256` outputs that `accept_full_v1.py` wrote on its full run. That run started at
09:44:05Z and lasted 963 s, ending about 10:00:08Z.

**When they were copied.** At 10:03:48Z, into the session scratchpad, before v2 started at 10:04:57Z (the times are the run logs' footers and
the copies' modification times, kept by `cp -p`). They are unchanged since.

**Why they are here.** The R2-B receipt claimed v1 and v2 wrote byte-identical files. Review 3d7eae6a could not reproduce that claim, because
only the final copies were committed. With these copies the claim can be checked: `cmp` each file here against its namesake one directory up.
