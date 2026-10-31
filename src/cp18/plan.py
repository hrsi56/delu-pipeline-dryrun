"""When a run issues what: the cutoff, the current target day, decision 23's catch-up, and the red line.

- **The cutoff** is D−1 11:00 Europe/Berlin (§24.4), `cp25.guard.issue_time`, unchanged by decision 22. A run may
  happen at any time after it; there is no deadline and no "late" label.
- **The current target day** of a run started at wall time `now` is the latest delivery day whose cutoff has passed:
  the day after today from 11:00 Berlin on, today before it. It is judged by the run's start, once per run.
- **The catch-up (decision 23).** A run issues, oldest first, the delivery days of the three-day lookback before the
  current target day that are neither issued nor closed, then the current target day: at most four issuances. A day
  older than the lookback is never issued. Days before the store's first day, and days not after the last issued or
  closed day (days go forward), are never planned.
- **The red line (§24.2, decision 7).** Until CP-18's activation brief retires the boundary guard after 4.7T, every
  planned day passes `cp25.guard.admit_delivery_day`: a plan holding any day after 2026-09-06 is refused whole, before
  anything is retrieved or checked. The retirement is a one-line, reviewed change to `cp25.guard`, not a flag here.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

from cp25 import guard

LOOKBACK_DAYS = 3
MAX_ISSUANCES = LOOKBACK_DAYS + 1


def cutoff(day: date) -> datetime:
    """D−1 11:00 Europe/Berlin as an aware UTC instant (§24.4)."""
    return guard.issue_time(day)


def target_day(now: datetime) -> date:
    """The latest delivery day whose cutoff is at or before `now` (an aware instant)."""
    if now.tzinfo is None:
        raise ValueError('a naive wall clock: the target day is undefined without a timezone')
    day = now.astimezone(guard.BERLIN).date() + timedelta(days=1)
    while cutoff(day) > now:
        day -= timedelta(days=1)
    return day


def lookback(target: date) -> list[date]:
    return [target - timedelta(days=i) for i in range(LOOKBACK_DAYS, 0, -1)]


def plan(target: date, *, first_day: date, done: set[str], last_closed: date | None) -> list[date]:
    """The days this run issues, oldest first: the open days of the lookback, then the target day (at most four)."""
    days = [d for d in lookback(target) + [target]
            if d >= first_day and str(d) not in done and (last_closed is None or d > last_closed)]
    return days[:MAX_ISSUANCES]


def to_close(target: date, *, first_day: date, done: set[str], last_closed: date | None) -> list[date]:
    """The days older than the lookback that are neither issued nor closed: they are closed as not issued."""
    start = lookback(target)[0]
    first = first_day if last_closed is None else max(first_day, last_closed + timedelta(days=1))
    return [first + timedelta(days=i) for i in range((start - first).days) if str(first + timedelta(days=i)) not in done]


def admit(days: list[date]) -> None:
    """The red line: refuse the whole plan if any day is after the development boundary (`cp25.guard`)."""
    for day in days:
        guard.admit_delivery_day(day)


def parse_now(value: str | None) -> datetime:
    """The run's wall clock: the real one, or a simulated one for a dry run on development days."""
    if not value:
        return datetime.now(timezone.utc)
    now = datetime.fromisoformat(value)
    if now.tzinfo is None:
        raise ValueError('--now needs an explicit offset, e.g. 2026-09-02T12:00:00+02:00')
    return now
