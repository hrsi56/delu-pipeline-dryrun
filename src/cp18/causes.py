"""The causes of a delivery day that is not issued (r19 §24.13, "For CP-18"), as the record names them.

A day is issued or not issued; there is no "late" status (decision 22). A day that is not issued carries exactly one
of these five causes in the store's record:

- `CHAIN`: the frozen chain ran and issued nothing (`cp17.issue.decide`'s failed issuance; its artifact is kept);
- `H2`: the run's checks before its first issuance failed (`cp17.hostcheck`), so the run issued nothing;
- `INPUTS`: the day's inputs could not be shown to precede its cutoff (§24.13, the publication time);
- `NO_RUN`: no run reached the day within decision 23's lookback;
- `S14`: the search state that serves the day (S_14, H4) is not in place.

A day attempted by a run that could not issue it stays open while it is inside the lookback, so a later run may
still issue it. It is closed as not issued, with the cause of its last attempt (or `NO_RUN` if nothing attempted it),
once a later day is issued or it leaves the lookback, whichever comes first. A chain failure closes the day at once:
the chain ran, and its record is the day's issuance record.
"""
from __future__ import annotations

CHAIN = 'chain'
H2 = 'h2'
INPUTS = 'inputs'
NO_RUN = 'no_run'
S14 = 's14'

TEXT = {
    CHAIN: 'a failed issuance by the chain',
    H2: 'a failed H2 check',
    INPUTS: "a live day's inputs that cannot be shown to precede its cutoff",
    NO_RUN: 'no run within the lookback',
    S14: "S_14's state not in place",
}
ORDER = (CHAIN, H2, INPUTS, NO_RUN, S14)


def text(code: str) -> str:
    if code not in TEXT:
        raise ValueError(f'{code!r} is not one of the five causes of a day not issued')
    return TEXT[code]
