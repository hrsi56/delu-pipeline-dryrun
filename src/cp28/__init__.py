"""CP-28: capstone v21-r17 §24.8 on CP-26's stored forecasts.

Seven arms against the control (nominal v5), under the outcome-blind frozen protocol
`docs/track-b/evidence/cp-26/cp28-frozen-protocol-v21-r14.md`, after the r17 descriptive GFS note
(`docs/track-b/evidence/cp-26/cp28-gfs-postresults-addendum-v21-r17.md`). NumPy only: no new
dependency, member fit, search, source or download. CP-26's stored outputs are read through
`cp28.reader`, which verifies every byte against the committed manifests; `src/cp26` is never
imported by the run.
"""
