"""The boundary of the live pipeline: the development guard retired for the live pipeline only (r20 §24.14 item 2).

**Why a route and not an edit.** `src/cp25/guard.py` is bound by CP-17c's freeze manifest (its `code` hashes), so its
`BOUNDARY` and `GFS_RUN_CUTOFF` cannot change without a new freeze. CP-18c retires the guard for the live pipeline by
routing around it here: every boundary check on the live pipeline's path (`cp18.plan`, `cp18.live.transport`,
`cp18.live.archive`, `cp18.live.history`, `cp18.live.build`, `cp18.live_inputs`, `cp18.outcomes`, `cp18.bundle`) calls
this module instead of `cp25.guard`.

**Two modes.**
- **Development** (the default, and the only mode outside a `live()` block): every check is `cp25.guard`'s, unchanged:
  any delivery day, row or GFS run after 2026-09-06 is refused. CP-18a's and CP-18b's dry runs, CP-25's development
  downloaders, CP-17c's development builder (`cp17.tools.payloads`) and every development report never enter
  `live()`, and most never import this module at all.
- **Live** (inside `with boundary.live(now=...)`, entered only by the live commands: `python -m cp18 run --mode live`,
  `python -m cp18 acquire`, `python -m cp18.outcomes`'s live route and the live bundle): the boundary is each
  delivery day's own cutoff, through the present.
  - **Through the present.** A delivery day, a row or a request's first day is admitted up to the next delivery day
    whose cutoff is still ahead of the present (the furthest day the pipeline can acquire for or issue), and a GFS
    run only if it was initialized at or before the present. Nothing on the live path admits the future.
  - **Each day's own cutoff.** What a day's build may use is judged by publication time against that day's cutoff
    (`cp18.live.availability`, unchanged), and D's own price is never an input (the target-day embargo). Those are
    the live red line, and they stay in force.
  - **The floor and the URL rules stay.** 2019-01-01, the 00 UTC cycle, §15.2's f021–f048, the two anonymous GFS
    archives' URL shapes and the credential refusal are `cp25.guard`'s own functions, called unchanged.

**Ruling 1's scope** (`cp18.live.availability.RULING_1_LAST_DAY`, 2026-09-06) is not this boundary and is not touched:
the recorded exception for the tier-3 load forecast covers the stored development history only, in either mode.
"""
from __future__ import annotations

from contextlib import contextmanager
import contextvars
from datetime import date, datetime, timedelta, timezone
import urllib.parse

from cp25 import guard

UTC = timezone.utc
BoundaryRefused = guard.BoundaryRefused
MODE_DEVELOPMENT, MODE_LIVE = 'development', 'live'

_LIVE: contextvars.ContextVar = contextvars.ContextVar('cp18_live_boundary', default=None)


@contextmanager
def live(now: datetime | None = None):
    """The live pipeline's boundary for the block. `now` is the present (an aware instant); the real clock if None,
    read at each check."""
    if now is not None and now.tzinfo is None:
        raise ValueError('a naive present: the live boundary needs an aware instant')
    token = _LIVE.set({'now': now})
    try:
        yield
    finally:
        _LIVE.reset(token)


def mode() -> str:
    return MODE_LIVE if _LIVE.get() is not None else MODE_DEVELOPMENT


def is_live() -> bool:
    return _LIVE.get() is not None


def present() -> datetime:
    value = _LIVE.get()
    now = value.get('now') if value else None
    return (now or datetime.now(UTC)).astimezone(UTC)


def horizon() -> date:
    """The last delivery day the live pipeline admits: the next one whose cutoff is still ahead of the present (the
    day after the present's Europe/Berlin day before 11:00 Berlin, the day after that from 11:00 on)."""
    now = present()
    day = now.astimezone(guard.BERLIN).date() + timedelta(days=1)
    while guard.issue_time(day) > now:
        day -= timedelta(days=1)
    return day + timedelta(days=1)


def describe() -> dict:
    if not is_live():
        return {'mode': MODE_DEVELOPMENT, 'boundary': str(guard.BOUNDARY), 'gfs_run_cutoff': guard.GFS_RUN_CUTOFF.isoformat(),
                'rule': 'cp25.guard (§24.2): nothing after 2026-09-06'}
    return {'mode': MODE_LIVE, 'present_utc': present().strftime('%Y-%m-%dT%H:%M:%SZ'), 'horizon': str(horizon()),
            'rule': ("r20 §24.14 item 2: the guard retired for the live pipeline only; each delivery day's own cutoff "
                     '(cp18.live.availability) through the present')}


# ------------------------------------------------------------------------------------------------ days and rows
def admit_delivery_day(day: date) -> date:
    if not is_live():
        return guard.admit_delivery_day(day)
    if day < guard.FLOOR:
        raise BoundaryRefused(f'delivery day {day} is before the 2019 input floor')
    if day > horizon():
        raise BoundaryRefused(f'delivery day {day} is after the present ({horizon()} is the last admitted day)')
    return day


def admit_row(instant: datetime) -> date:
    if not is_live():
        return guard.admit_row(instant)
    return admit_delivery_day(guard.delivery_day(instant))


def admit_rows(instants) -> int:
    if not is_live():
        return guard.admit_rows(instants)
    n = 0
    for instant in instants:
        admit_row(instant if isinstance(instant, datetime) else instant.to_pydatetime())
        n += 1
    return n


def admit_scope(days) -> None:
    """A market request's scope: development, every day by the guard; live, the request may not start after the
    present (a source's week or month bucket reaches past it, and holds only what is published)."""
    days = list(days)
    if not days:
        raise BoundaryRefused('a market-data request must name its delivery days')
    if not is_live():
        for d in days:
            guard.admit_delivery_day(d)
        return
    if min(days) < guard.FLOOR:
        raise BoundaryRefused(f'a request from {min(days)}, before the 2019 input floor')
    if min(days) > horizon():
        raise BoundaryRefused(f'a request from {min(days)}, after the present ({horizon()})')


# ------------------------------------------------------------------------------------------------ GFS
def admit_gfs(init: datetime, lead: int) -> None:
    if not is_live():
        return guard.admit_gfs(init, lead)
    if init.tzinfo is None:
        raise BoundaryRefused('naive GFS initialization')
    if init.astimezone(UTC) > present():
        raise BoundaryRefused(f'GFS run {init.astimezone(UTC):%Y-%m-%dT%HZ} is initialized after the present')
    if init.astimezone(UTC) < guard.run_init(guard.FLOOR):
        raise BoundaryRefused(f'GFS run {init:%Y-%m-%d} is before the 2019 input floor')
    if not isinstance(lead, int) or not guard.LEAD_MIN <= lead <= guard.LEAD_MAX:
        raise BoundaryRefused(f'GFS lead f{lead} is outside f{guard.LEAD_MIN:03d}-f{guard.LEAD_MAX:03d}')


def admit_url(url: str) -> dict:
    """`cp25.guard.admit_url`'s parse, with the run admitted by this module's mode. Fails closed."""
    if not is_live():
        return guard.admit_url(url)
    parts = guard.refuse_credentials(url)
    if parts.scheme != 'https':
        raise BoundaryRefused('only https GFS requests are admitted')
    query = dict(urllib.parse.parse_qsl(parts.query, keep_blank_values=True))
    if parts.netloc != guard.AWS_HOST:
        raise BoundaryRefused(f'host {parts.netloc!r}: the live pipeline reads GFS from the AWS archive only')
    if parts.path == '/' and set(query) == {'list-type', 'prefix'} and query['list-type'] == '2':
        m = guard._AWS_PREFIX.match(query['prefix'])   # noqa: SLF001 - the frozen guard's own URL shapes
        if not m:
            raise BoundaryRefused('AWS listing prefix is not one exact (run, lead) object stem')
        ymd, cyc, tcyc, lead = m.groups()
        kind = 'aws_listing'
    else:
        if query:
            raise BoundaryRefused('unexpected AWS query')
        m = guard._AWS_OBJECT.match(parts.path)        # noqa: SLF001
        if not m:
            raise BoundaryRefused('unrecognised AWS object path')
        ymd, cyc, tcyc, lead, idx = m.groups()
        kind = 'aws_index' if idx else 'aws_object'
    if cyc != '00' or tcyc != '00':
        raise BoundaryRefused('only the 00 UTC cycle is admitted')
    run, lead = guard._day(ymd), int(lead)              # noqa: SLF001
    admit_gfs(guard.run_init(run), lead)
    return {'endpoint': 'aws', 'kind': kind, 'run': run, 'lead': lead}
