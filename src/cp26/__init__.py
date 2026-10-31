"""CP-26: the re-baseline on every development day (capstone v21-r13 §24.7).

A harness around the landed member code: B0, v1-R and v2-v5 refit daily on every eligible
development day to 2026-09-06, the forecast ledger, the scores and CP-28's frozen protocol.
The landed member code is imported unchanged; the harness replaces only what §24.7 lists
(data identity, boundary, weather source, issue-time label and identity pins), and logs each
replacement (`cp26.harness.HARNESS_LOG`).
"""
