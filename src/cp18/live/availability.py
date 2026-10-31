"""The availability judgement: each input by its publication time against D's cutoff, never by the run time.

**The cutoff** is D-1 11:00 Europe/Berlin, strictly before (`cp25.guard.issue_time` / `strictly_before`,
capstone §24.4, §24.13 "The publication time"): 10:00 UTC when D-1 is on CET, 09:00 UTC on CEST, D-1's own
offset at 11:00 on transition days.

**What each input may rest on** (the tier register, `reports/development-inputs/tier-register.md`, and the
Owner's rulings of 2026-10-12, `docs/track-b/cp-25-rulings-2026-10-12.md`):

| Input | Tier | Accepted publication bases |
|---|---|---|
| day-ahead price of P (lags, statistics, training targets) | 2 (ruling 2) | the documented schedule: P's SDAC results by 13:00 local on P-1; or a document time / retrieval time |
| day-ahead load forecast of P | 3 (ruling 1) | retrieval and archive strictly before the cutoff; or the document's own time strictly before it; or, for the stored development history only (P <= 2026-09-06), ruling 1's recorded exception |
| GFS D-1 00 UTC run (the flag) | 2 (rulings 3, 4) | AWS `LastModified` of all 10 objects and 10 indexes strictly before the cutoff (ruling 4's criterion); where no listing exists, the documented schedule (CP-25's reconstruction of NCEP's 48-hour-products completion) |

D's own price is never an input (the target-day embargo, §24.4).

**Which cutoff.** The price and the load forecast of every history day P that enters D's frame are judged
against D's cutoff. A GFS run is judged against the cutoff of the day it serves: a run late for its own day is
missing in every later frame too, as `cp26.harness._features` treats ruling 4's two runs.

**Ruling 1's exception** admits the tier-3 load forecast of the stored development history (the snapshot pulled
2026-09-08) and is refused for any day after 2026-09-06, whatever the boundary guard says: the exception covers
the development history, not live days. `RULING_1_LAST_DAY` is that scope, not a red line.

**The GFS flag** is True only if the run is shown published strictly before the cutoff of the day it serves,
all 50 messages and the decoded run are in the archive, and the converted vector (CP-20's frozen §15.2 recipe)
is valid on every canonical hour. Otherwise it is False and the day's three weather columns are set missing
under §15.3, on every hour.
"""
from __future__ import annotations

from datetime import date, datetime, time, timedelta, timezone

from cp25 import guard

UTC = timezone.utc
RULING_1_LAST_DAY = date(2026, 9, 6)        # ruling 1's scope: the stored development history (not the guard)
RULINGS = 'docs/track-b/cp-25-rulings-2026-10-12.md'
TIER_REGISTER = 'reports/development-inputs/tier-register.md'

BASES = {
    'documented_schedule': 'the documented schedule of a tier-2 input',
    'aws_last_modified': "AWS LastModified of the run's 10 objects and 10 indexes (ruling 4's criterion)",
    'document_time': "the document's own creation time",
    'retrieval_time': 'retrieved and archived by the pipeline at this instant',
    'ruling_1_exception': ("Owner's ruling 1 (2026-10-12, " + RULINGS + "): recorded exception for the stored "
                           "development history of the tier-3 load forecast; development data only "
                           "(delivery days to 2026-09-06)"),
}
ACCEPTED = {
    'price': ('documented_schedule', 'document_time', 'retrieval_time'),
    'load_forecast': ('document_time', 'retrieval_time', 'ruling_1_exception'),
    'gfs': ('aws_last_modified', 'documented_schedule'),
}
TIERS = {'price': 2, 'load_forecast': 3, 'gfs': 2}
PRICE_BASIS = ('tier 2 (ruling 2): SDAC results for delivery day P are published no later than one hour after '
               'the 12:00 gate on P-1, so by 13:00 Europe/Berlin (Reg. 543/2013 Art. 12(1)(d); CACM Arts. 47-48; '
               'the tier register)')
GFS_SCHEDULE_BASIS = ("tier 2: CP-25's reconstruction of NCEP's 00 UTC 48-hour-products completion from the "
                      'bracketing dated captures (reports/weather-admission dossier §6; cp25.runs)')


def iso(instant: datetime | None) -> str | None:
    return None if instant is None else instant.astimezone(UTC).strftime('%Y-%m-%dT%H:%M:%SZ')


def parse_iso(text: str | None) -> datetime | None:
    if not text:
        return None
    text = text.replace('Z', '+00:00')
    value = datetime.fromisoformat(text)
    if value.tzinfo is None:
        raise ValueError('naive instant in an archive record')
    return value.astimezone(UTC)


def cutoff(day: date) -> datetime:
    """D-1 11:00 Europe/Berlin as UTC (`cp25.guard.issue_time`)."""
    return guard.issue_time(day)


# ------------------------------------------------------------------------------------------------ evidence
def price_publication(price_day: date) -> dict:
    """The documented publication of delivery day P's prices: 13:00 Europe/Berlin on P-1."""
    at = datetime.combine(price_day - timedelta(days=1), time(13, 0), tzinfo=guard.BERLIN)
    return {'utc': iso(at), 'basis_id': 'documented_schedule', 'basis': PRICE_BASIS}


def load_publication(day: date, *, provenance: str, retrieved_utc: str | None, document_time_utc: str | None,
                     development_snapshot: bool) -> dict:
    """The evidence an archived load-forecast response carries for delivery day P (= `day`)."""
    if development_snapshot:
        if day > RULING_1_LAST_DAY:
            raise ValueError("ruling 1's exception covers the stored development history only")
        return {'utc': None, 'basis_id': 'ruling_1_exception', 'basis': BASES['ruling_1_exception']}
    if provenance != 'live':
        return {'utc': None, 'basis_id': 'none',
                'basis': f'no publication evidence: a {provenance} response that is not the development snapshot'}
    doc, got = parse_iso(document_time_utc), parse_iso(retrieved_utc)
    if doc is not None and (got is None or doc <= got):
        return {'utc': iso(doc), 'basis_id': 'document_time', 'basis': BASES['document_time']}
    return {'utc': iso(got), 'basis_id': 'retrieval_time', 'basis': BASES['retrieval_time']}


def gfs_publication(listing: dict | None, reconstructed: dict | None) -> dict:
    """The run's publication: the latest LastModified of its 20 listed objects (ruling 4), else the schedule."""
    stamps, missing = [], []
    for lead in sorted(listing or {}):
        for kind in ('object', 'index'):
            item = (listing[lead] or {}).get(kind)
            if item and item.get('last_modified'):
                stamps.append(parse_iso(item['last_modified']))
            else:
                missing.append(f'{lead} {kind}')
    if listing and len(stamps) == 20 and not missing:
        return {'utc': iso(max(stamps)), 'basis_id': 'aws_last_modified', 'basis': BASES['aws_last_modified'],
                'evidence': {'objects_listed': 20}}
    if reconstructed and reconstructed.get('bracketed'):
        return {'utc': reconstructed['available_utc'], 'basis_id': 'documented_schedule', 'basis': GFS_SCHEDULE_BASIS,
                'evidence': {k: reconstructed[k] for k in ('capture_before', 'capture_after', 'end_48h_before',
                                                           'end_48h_after')},
                'listing_incomplete': missing if listing else 'no listing recorded'}
    return {'utc': None, 'basis_id': 'none', 'basis': 'no listing and no documented-schedule reconstruction',
            'listing_incomplete': missing if listing else 'no listing recorded'}


# ------------------------------------------------------------------------------------------------ judgement
def judge(input_name: str, for_day: date, data_day: date, publication: dict, record: dict | None,
          archive_day: date | None = None) -> dict:
    """One input of one data day, for D = `for_day`: admitted or not, and why. Pure, from the record alone."""
    if input_name == 'gfs':
        limit = cutoff(data_day)           # a run is judged against the day it serves
    else:
        limit = cutoff(for_day)
    out = {'input': input_name, 'tier': TIERS[input_name], 'data_day': data_day.isoformat(),
           'for_day': for_day.isoformat(), 'cutoff_utc': iso(limit),
           'archive_day': archive_day.isoformat() if archive_day else None,
           'record': record.get('id') if record else None, 'sha256': record.get('sha256') if record else None,
           'provenance': record.get('provenance') if record else None,
           'publication_utc': publication.get('utc'), 'basis_id': publication.get('basis_id'),
           'basis': publication.get('basis'), 'by_exception': False, 'admitted': False, 'reason': ''}
    if input_name == 'price' and data_day >= for_day:
        out['reason'] = "the target-day embargo: D's own price is never an input"
        return out
    basis = publication.get('basis_id')
    if basis not in ACCEPTED[input_name]:
        out['reason'] = f'basis {basis!r} is not accepted for a tier-{TIERS[input_name]} {input_name}'
        return out
    if basis == 'ruling_1_exception':
        if data_day > RULING_1_LAST_DAY:
            out['reason'] = "ruling 1's exception covers the stored development history only, not this day"
            return out
        if record is not None and record.get('provenance') not in ('rendered', 'recorded'):
            out['reason'] = "ruling 1's exception covers the stored development history, not a live retrieval"
            return out
        out.update(admitted=True, by_exception=True,
                   reason="admitted under ruling 1's recorded exception (development data only)")
        return out
    at = parse_iso(publication.get('utc'))
    if at is None:
        out['reason'] = 'no publication time'
        return out
    if guard.strictly_before(at, data_day if input_name == 'gfs' else for_day):
        out.update(admitted=True, reason=f'published {iso(at)}, strictly before the cutoff {iso(limit)}')
    else:
        out['reason'] = f'published {iso(at)}, not strictly before the cutoff {iso(limit)}'
    return out
