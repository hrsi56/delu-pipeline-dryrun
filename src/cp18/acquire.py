"""The acquisition before the cutoff: the next delivery day's inputs retrieved, archived and published (CP-18c).

**Why it exists.** The day-ahead load forecast is tier 3 (the tier register, ruling 1): only a retrieval, or a
document's own time, strictly before D's cutoff (D−1 11:00 Europe/Berlin) shows it, and no source preserves its
vintage. A run after the cutoff can therefore never show D's load forecast; so D's inputs are acquired before it,
by this command, and the issuance runs after it (`python -m cp18 run --mode live`). TSOs publish the forecast by
D−1 10:00 local (Reg. 543/2013 Art. 6), on ENTSO-E's Transparency Platform (A65, process A01), so the window is
about an hour (`cp18.live.acquire`).

**One acquisition, in order** (inside `cp18.boundary.live()`):
1. the claim: fetch `main` and check it out (`cp18.claim`), so the archive is the published one;
2. D: the given day, or the next delivery day whose cutoff is still ahead;
3. every input of archive day D still missing (`cp18.live.acquire.acquire_day`): the load forecast of D, the prices of
   D−3..D−1, the GFS run that serves D; then, while D's load forecast is not yet shown (not published, or incomplete),
   again every `poll_seconds`, until `margin_seconds` before the cutoff;
4. the archive day published to `main` through the claim (a data-only commit of `live/inputs/`), whatever was found:
   a retrieval is evidence either way.

Exit: 0 D's load forecast is shown before its cutoff and published; 1 it is not (D will not be issued, with its
cause, by the run); 2 refused (the claim or the publication). An exception never escapes: it is the record's outcome.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
import time
import traceback

from cp25 import guard

from . import boundary, plan
from .claim import ClaimError
from .store import INPUTS_PATH

SCHEMA = 'cp18c-acquire-v1'
UTC = timezone.utc


def utc_now() -> str:
    return datetime.now(UTC).strftime('%Y-%m-%dT%H:%M:%SZ')


def next_day(now: datetime) -> date:
    """The next delivery day whose cutoff is still ahead of `now`."""
    return plan.target_day(now) + timedelta(days=1)


def run(root, *, claim, day: str | None = None, poll_seconds: int = 300, margin_seconds: int = 120,
        window_hours: float | None = 3.0, transport=None, clock=None, sleep=time.sleep) -> dict:
    """One acquisition. With `window_hours`, a run that finds the next cutoff further away than that does nothing
    (exit 0, "not yet"): a schedule that fires twice a day, for winter and summer time, never polls for a day."""
    from .live import acquire as AQ
    from .live.archive import Archive
    from .live.transport import HttpTransport
    fixed = clock is not None              # a given clock is also the live boundary's present (tests, rehearsals)
    clock = clock or (lambda: datetime.now(UTC))
    record = {'schema': SCHEMA, 'started_utc': utc_now(), 'day': None, 'cutoff_utc': None, 'attempts': [],
              'load_forecast': None, 'publication': None, 'outcome': None, 'exit': None}
    try:
        with boundary.live(now=clock() if fixed else None):
            record['boundary'] = boundary.describe()
            claim.sync()
            target = date.fromisoformat(day) if day else next_day(clock())
            boundary.admit_delivery_day(target)
            cutoff = plan.cutoff(target)
            record.update(day=str(target), cutoff_utc=cutoff.strftime('%Y-%m-%dT%H:%M:%SZ'))
            if day is None and window_hours is not None and cutoff - clock() > timedelta(hours=window_hours):
                record.update(outcome=f'not yet: {target}\'s cutoff is more than {window_hours:g} h away', exit=0,
                              finished_utc=utc_now())
                return record
            archive = Archive(root / INPUTS_PATH)
            transport = transport or HttpTransport()
            while True:
                result = AQ.acquire_day(archive, target, transport)
                record['attempts'].append({k: v for k, v in result.items() if k != 'items'} |
                                          {'items': [{k: v for k, v in i.items() if k != 'traceback'}
                                                     for i in result['items']]})
                shown = AQ.load_forecast_shown(archive, target)
                record['load_forecast'] = shown
                if shown.get('shown'):
                    break
                if clock() + timedelta(seconds=poll_seconds + margin_seconds) >= cutoff:
                    break
                sleep(poll_seconds)
            published = claim.publish([INPUTS_PATH], f'live: inputs for {target} acquired before its cutoff\n\n'
                                                     f'Delu-Daily-Inputs: {target}\n')
            record['publication'] = published
            if published.get('lost'):
                record.update(outcome='main moved while acquiring; this acquisition was discarded', exit=2)
            elif record['load_forecast'].get('shown'):
                record.update(outcome=f"{target}: the load forecast is shown before the cutoff "
                                      f"({record['load_forecast'].get('publication_utc')})", exit=0)
            else:
                record.update(outcome=f"{target}: the load forecast is not shown before the cutoff "
                                      f"({record['load_forecast'].get('reason')}); the day will not be issued", exit=1)
    except ClaimError as exc:
        record.update(outcome=f'refused: {guard.redact(str(exc))}', exit=2)
    except Exception as exc:  # noqa: BLE001 - N5: never an escaped run
        record.update(outcome=f'failed: {guard.redact(f"{type(exc).__name__}: {exc}")}', exit=2,
                      traceback=guard.redact(traceback.format_exc(limit=6)))
    record['finished_utc'] = utc_now()
    return record
