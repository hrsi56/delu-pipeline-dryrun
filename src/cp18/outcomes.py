"""The outcomes' reconciliation: each issued day's published prices, per UTC key, with source, publication and revision.

**As the live pipeline runs it.** Each run reconciles every issued delivery day of the store against the input archive
(`live/inputs/days/<x>/`, CP-18a's), at the run's own wall clock `now`:

- **Where a price comes from.** Day X's prices are archived for the build of a later day x > X (SMARD `chart_data`,
  filter 4169, DE-LU: the hourly product before 2025-10-01, the mean of the four quarter-hour prices from then on),
  each record with its retrieval time, SHA-256, provenance and the publication of each data day it covers. The
  reconciler reads the earliest such record (`cp18.live.history`'s reader) and re-parses its stored body.
- **Published or pending.** A day's prices are published once their publication time is at or before `now`. The
  publication time is the source document's own when the response carries one; otherwise the tier register's
  documented schedule (P's prices by 13:00 Europe/Berlin on P−1, ruling 2). Until then the day is **pending**: every
  price is `null` and nothing is scored. A key the record holds no value for (a partial quarter-hour bin fails closed)
  stays `null`, so its day is **partial**.
- **Revisions.** A later archived record that covers X with other values is a revision: the new value is reconciled,
  with its own publication or retrieval time as `revised_utc`; the earlier observation stays in the log.
- **A later run reconciles the days whose outcomes have since been published.** Every reconciliation appends one
  entry to `live/outcomes/` (`cp18.records.RecordLog`) holding the days that changed: newly published, newly partial or
  revised. Nothing is rewritten. The current view folds the entries in order.

**This round** reads development days only, from the archive CP-18a's recorded responses rebuild (SMARD values pulled
2026-09-08, rendered in the source's documented response shape and labelled `rendered`), never a fresh download. A
rendered response carries no document time, so its publication time is the documented schedule's, and the record
says so.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from cp25 import guard

from . import boundary

from .records import OUTCOMES_PATH, RecordLog

SCHEMA = 'cp18b-outcomes-reconciliation-v1'
SOURCE = 'SMARD chart_data filter 4169 (day-ahead price, DE-LU), via the input archive'
SERIES = 'SMARD chart_data filter 4169 (day-ahead price, DE-LU)'
WHERE = {'input archive': 'live/inputs/days/{x}/', 'outcome retrieval': 'live/outcomes/archive/days/{x}/'}


def source_label(observed: dict) -> str:
    """O18's N1: the archive and the record that served a price, never a generic label."""
    where = WHERE.get(observed.get('archive'), '{x}').format(x=observed.get('archive_day'))
    return f"{SERIES}, {observed.get('archive')} {where} record {observed.get('record_id')}"
UNIT = 'EUR/MWh'


def _utc(text: str | None) -> datetime | None:
    if not text:
        return None
    value = datetime.fromisoformat(text.replace('Z', '+00:00'))
    return value if value.tzinfo else value.replace(tzinfo=timezone.utc)


def _iso(value: datetime | None) -> str:
    return '' if value is None else value.astimezone(timezone.utc).isoformat(timespec='seconds')


def canonical_keys(day: date) -> list[str]:
    from cp17.issue import canonical_hours
    return canonical_hours(day)


def _candidates(archive, day: date, own=None) -> list[tuple[str, object, date, dict]]:
    """Every archived price record covering `day`: the input archive's later days (`x > day`, written for those days'
    builds), then the reconciler's own retrievals (`own`, an archive whose day `day` holds them)."""
    found = []
    if archive is not None:
        for x in archive.days():
            if x <= day:
                continue
            for r in archive.records(x, 'price'):
                if day.isoformat() in r.get('covers', []) and r.get('status') == 200 and r.get('body'):
                    found.append(('input archive', archive, x, r))
    if own is not None and own.has(day):
        for r in own.records(day, 'price'):
            if day.isoformat() in r.get('covers', []) and r.get('status') == 200 and r.get('body'):
                found.append(('outcome retrieval', own, day, r))
    return found


def observe(archive, day: date, now: datetime, own=None) -> dict:
    """Day `day`'s prices as the archives show them at `now`: the earliest admissible record's values and evidence,
    every later record that disagrees (a revision), or pending."""
    from .live import history as H
    boundary.admit_delivery_day(day)
    keys = canonical_keys(day)
    observed = []
    for where, arch, x, record in _candidates(archive, day, own):
        pub = (record.get('publication') or {}).get(day.isoformat(), {})
        document = record.get('document_time_utc')
        published = _utc(document) or _utc(pub.get('utc'))
        basis = 'the source document time' if document else (pub.get('basis_id') or 'unknown')
        if published is None or published > now:
            continue
        body = arch.body(x, record)
        values = H.VALUES[record['source']]('price', record, body,
                                            [date.fromisoformat(c) for c in record['covers']])[day]
        observed.append({'archive': where, 'archive_day': str(x), 'record_id': record['id'],
                         'record_sha256': record['sha256'], 'provenance': record.get('provenance'),
                         'retrieved_utc': record.get('retrieved_utc') or '', 'published_utc': _iso(published),
                         'publication_basis': basis, 'publication_basis_text': pub.get('basis'),
                         'prices': [None if not np.isfinite(v) else float(v) for v in values]})
    if not observed:
        return {'status': 'pending', 'reason': 'not yet published at the run time, or not yet archived', 'keys': keys}
    first = observed[0]
    current, revised = list(first['prices']), [''] * len(keys)
    revisions = []
    for later in observed[1:]:
        changed = [i for i, (a, b) in enumerate(zip(current, later['prices'])) if a != b and b is not None]
        if changed:
            stamp = later['published_utc'] if later['publication_basis'] == 'the source document time' else later['retrieved_utc']
            for i in changed:
                current[i], revised[i] = later['prices'][i], stamp
            revisions.append({**{k: later[k] for k in ('archive', 'archive_day', 'record_id', 'record_sha256',
                                                        'retrieved_utc', 'published_utc', 'publication_basis')},
                              'keys': [keys[i] for i in changed]})
    missing = [keys[i] for i, v in enumerate(current) if v is None]
    return {'status': 'published' if not missing else 'partial', 'keys': keys, 'prices': current,
            'revised_utc': revised, 'missing_keys': missing, 'source': SOURCE,
            'first': {k: v for k, v in first.items() if k != 'prices'}, 'revisions': revisions}


def _signature(item: dict) -> tuple:
    return (item['status'], tuple(item.get('prices') or ()), tuple(item.get('revised_utc') or ()))


class Outcomes:
    """The reconciliation log under `live/outcomes/` of a checkout (or any folder)."""

    def __init__(self, folder):
        self.log = RecordLog(folder)

    @classmethod
    def at(cls, root) -> 'Outcomes':
        return cls(Path(root) / OUTCOMES_PATH)

    def view(self) -> dict[str, dict]:
        """Every reconciled day's current observation, folding the entries in order."""
        out: dict[str, dict] = {}
        for entry in self.log.verify():
            for day, item in self.log.read(entry)['days'].items():
                out[day] = item
        return out

    @property
    def own(self):
        """The reconciler's own retrievals (`live/outcomes/archive/`), in CP-18a's archive format."""
        from .live.archive import Archive
        return Archive(self.log.folder / 'archive')

    def retrieve(self, transport, day: date, now: datetime) -> dict | None:
        """Retrieve `day`'s prices through the SMARD client once they are published (the documented schedule) and
        nothing archived covers them yet; the response is archived with its evidence like an input."""
        from .live import archive as AR, smard
        from .live.availability import price_publication
        if _utc(price_publication(day)['utc']) > now or self.own.has(day):
            return None
        fetched = smard.fetch(transport, 'price', [day])
        if not any(ex.role == 'bucket' and ex.response is not None and ex.response.status == 200
                   for ex in fetched.exchanges):
            return {'day': str(day), 'records': [], 'problems': fetched.problems or ['nothing retrieved'],
                    'archived': False}       # a failed retrieval archives nothing, so a later run retries it
        rendered = [ex.response.provenance == 'rendered' for ex in fetched.exchanges if ex.response is not None]
        records = AR.store_market(self.own, day, fetched, development_snapshot=bool(rendered) and all(rendered))
        return {'day': str(day), 'records': [r['id'] for r in records], 'problems': fetched.problems}

    def reconcile(self, archive, days: list[date], now: datetime, run_id: str, transport=None) -> dict | None:
        """One reconciliation at `now` of the issued `days`: a published day nothing archived covers is retrieved
        through `transport` (when given); appends an entry with the days whose observation changed and returns it
        (None when nothing changed)."""
        current = self.view()
        changed, pending, retrieved = {}, [], []
        for day in sorted(days):
            item = observe(archive, day, now, self.own)
            if item['status'] == 'pending' and transport is not None:
                got = self.retrieve(transport, day, now)
                if got:
                    retrieved.append(got)
                    item = observe(archive, day, now, self.own)
            if item['status'] == 'pending':
                pending.append(str(day))
                if str(day) in current and current[str(day)]['status'] != 'pending':
                    raise ValueError(f'{day}: an earlier reconciliation published it; a price is never withdrawn')
                continue
            if str(day) not in current or _signature(current[str(day)]) != _signature(item):
                changed[str(day)] = item
        value = {'schema': SCHEMA, 'reconciled_utc': _iso(now), 'run_id': run_id, 'unit': UNIT, 'days': changed,
                 'pending': pending, 'retrieved': retrieved}
        if not changed:
            return None
        n = len(self.log.entries())
        return self.log.append('reconciliation', f'entries/{n:05d}-{_iso(now)[:10]}.json', value,
                               reconciled_utc=_iso(now), run_id=run_id, days=sorted(changed))


def named_sources(item: dict) -> list[str]:
    """Per key, the archive and record that served its current price: the first observation, or the revision that
    changed it (O18's N1)."""
    names = [source_label(item['first'])] * len(item['keys'])
    for revision in item.get('revisions') or []:
        label = source_label(revision)
        for k in revision.get('keys') or []:
            names[item['keys'].index(k)] = label
    return names


def bundle_outcomes(days: list[date], view: dict[str, dict], named: bool = False) -> dict:
    """The bundle's `outcomes.json` (cp17-outcomes-v1): every canonical key of every issued day, its price or null.
    `named` (the live product, CP-18c): each price names the archive and record that served it (O18's N1); a dry run's
    bundle keeps CP-18b's label, so its committed records reproduce byte for byte."""
    prices = []
    for day in sorted(days):
        item = view.get(str(day))
        keys = canonical_keys(day)
        if item is None or item['status'] == 'pending':
            prices += [{'timestamp_utc': k, 'delivery_day': str(day), 'price': None, 'source': SOURCE,
                        'published_utc': '', 'revised_utc': ''} for k in keys]
            continue
        if item['keys'] != keys:
            raise ValueError(f'{day}: the reconciled keys are not the canonical hours')
        labels = named_sources(item) if named else [SOURCE] * len(keys)
        for k, v, revised, source in zip(keys, item['prices'], item['revised_utc'], labels):
            prices.append({'timestamp_utc': k, 'delivery_day': str(day), 'price': v, 'source': source,
                           'published_utc': item['first']['published_utc'] if v is not None else '',
                           'revised_utc': revised if v is not None else ''})
    return {'schema': 'cp17-outcomes-v1', 'unit': UNIT, 'prices': prices}


def outcome_map(outcomes: dict) -> dict:
    """outcomes.json as `cp17.scoring`'s {key: price}, NaN where null."""
    return {p['timestamp_utc']: (np.nan if p['price'] is None else float(p['price'])) for p in outcomes['prices']}


def day_status(day: date, outcomes: dict) -> dict:
    keys = canonical_keys(day)
    by_key = {p['timestamp_utc']: p for p in outcomes['prices'] if p['delivery_day'] == str(day)}
    published = [k for k in keys if by_key.get(k, {}).get('price') is not None]
    state = 'pending' if not published else ('partial' if len(published) < len(keys) else 'published')
    return {'day': str(day), 'state': state, 'published_hours': len(published), 'canonical_hours': len(keys)}


__all__ = ['Outcomes', 'observe', 'bundle_outcomes', 'outcome_map', 'day_status', 'SOURCE', 'canonical_keys',
]
