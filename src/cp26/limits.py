"""Finite CP-26 operational limits; calibration supersedes historical r14 values."""
# Prior measured three-worker tail0.645392s;3s permits >4.6x that observation.
# The short real-ledger calibration must validate these choices before model fits.
LEDGER_TIMEOUT_SECONDS = 3.0
HEARTBEAT_TIMEOUT_SECONDS = 8.0
STARTUP_GRACE_SECONDS = 14.0
POLL_SECONDS = 0.2
TERMINATION_GRACE_SECONDS = 5.0
KILL_CONFIRM_SECONDS = 5.0


def shutdown_exposure_seconds(workers: int) -> float:
    """Uncharged upper tail: detection, lock/poll jitter, termination and closure."""
    return workers * (HEARTBEAT_TIMEOUT_SECONDS + 2 * LEDGER_TIMEOUT_SECONDS +
                      2 * POLL_SECONDS + TERMINATION_GRACE_SECONDS +
                      KILL_CONFIRM_SECONDS + STARTUP_GRACE_SECONDS)
