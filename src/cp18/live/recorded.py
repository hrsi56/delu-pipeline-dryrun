"""Responses for development days, and the checks that the clients reproduce the recorded data.

**What is truly recorded and what is rendered.**

* **SMARD and ENTSO-E responses were never recorded raw.** The snapshot (`data/snapshot.parquet`) holds the
  hourly values SMARD returned on 2026-09-08 (pull time 2026-09-08T13:34:33Z), not the responses. This module
  **renders** them from those values in each source's documented response shape - SMARD `chart_data` weekly
  buckets and their index; ENTSO-E A44 and A65 XML documents - and says so in every replay entry and archive
  record (`provenance: rendered`). The quarter-hour values are not recoverable from hourly values, so each hour
  is rendered as four equal quarters: the price as v four times (mean v), SMARD's load as v/4 four times (sum v),
  ENTSO-E's load as v four times (mean v). `verify` shows that the clients' parse, with the snapshot's own
  aggregation rules, gives back every snapshot row bit for bit. A rendered document carries no source document
  time (SMARD's `meta_data.created` is null; ENTSO-E's `createdDateTime` is the pull time and is never used as
  evidence).
* **GFS is truly recorded:** CP-20's and CP-25's decoded runs (`.local/artifacts/cp-{20,25}/weather/runs/`),
  each message's idx line, byte range, bytes, SHA-256 and decoded metadata; CP-25's per-lead listing values
  (size, ETag, `LastModified`); and, for seven CP-25 runs including 2026-09-05, the raw GRIB2 messages. The
  listing XML and the full `.idx` texts were not kept: `gfs-client` renders them from the recorded values
  (the index's successor lines and its Size are rendered and labelled so) and serves the recorded raw messages.

Subcommands (`python -m cp18.live.recorded <cmd> ...`):

* `render --replay DIR` - SMARD buckets and indexes for every development week, ENTSO-E documents for every
  development day, into `DIR/smard` and `DIR/entsoe`;
* `verify --replay DIR --report FILE` - parse every development day through both clients and compare with the
  snapshot rows (bit-exact floats; NaN where the snapshot is null);
* `archive --replay DIR --archive DIR --days A..B[,C..D] [--source smard|entsoe]` - development archive days:
  prices of D-3..D-1 and the load forecast of D through the client (rendered responses), and the D-1 00 UTC run
  from the recorded decoded runs;
* `dry-run-inputs --days A..B --out DIR` - one self-contained folder for given development days: the rendered
  market responses those archive days need (`DIR/replay/{smard,entsoe}`) and copies of the truly recorded GFS
  runs and listing values (`DIR/gfs`), with a manifest of hashes and sizes. `archive --replay DIR/replay
  --gfs-sources DIR/gfs` then needs no `.local/` (for a runner that has only committed files);
* `gfs-client --run 2026-09-05 --work DIR --report FILE` - the GFS client end to end on the recorded material,
  decoded in the ecCodes environment and compared with the recorded run.
"""
from __future__ import annotations

import argparse
from datetime import date, timedelta
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

from cp15.data import day_hours
from delu_forecast.ingest import BERLIN, TRANSITION_LOCAL_DATE
from cp25 import guard

from . import entsoe, gfs, smard
from . import archive as AR
from .transport import ReplayFolder, ReplayTransport, Response

PROJECT = gfs.PROJECT
CP20_WEATHER = PROJECT / '.local/artifacts/cp-20/weather'
CP25_WEATHER = PROJECT / '.local/artifacts/cp-25/weather'
GFS_SOURCES = (CP25_WEATHER, CP20_WEATHER)
FLOOR = date(2019, 1, 1)
RENDERED_SMARD = ('rendered from data/snapshot.parquet (SMARD values pulled 2026-09-08T13:34:33Z) in SMARD '
                  "chart_data's documented shape; not a recorded raw response; quarter-hour values rendered as four "
                  'equal quarters (price v, load v/4); meta_data.created is null')
RENDERED_ENTSOE = ('rendered from data/snapshot.parquet (SMARD values pulled 2026-09-08T13:34:33Z) in ENTSO-E\'s '
                   'documented A44/A65 XML shape; not a recorded raw response (the API was unavailable at the pull); '
                   'quarter-hour values rendered as four equal quarters (v); createdDateTime is the pull time, not a '
                   'source document time')


def snapshot(root: Path) -> pd.DataFrame:
    frame = pd.read_parquet(Path(root) / 'data/snapshot.parquet',
                            columns=['timestamp_utc', 'delivery_date', 'price_eur_mwh', 'load_forecast_mw'],
                            filters=[('delivery_date', '>=', FLOOR), ('delivery_date', '<=', guard.BOUNDARY)])
    guard.admit_delivery_day(max(frame.delivery_date))
    guard.admit_rows(pd.DatetimeIndex(frame.timestamp_utc).tz_convert('UTC'))
    frame['timestamp_utc'] = pd.DatetimeIndex(frame.timestamp_utc).tz_convert('UTC').as_unit('ns')
    return frame.set_index('timestamp_utc')


def _ms(ts: pd.Timestamp) -> int:
    return int(ts.value // 1_000_000)


def week_start(day: date) -> pd.Timestamp:
    monday = day - timedelta(days=day.weekday())
    return pd.Timestamp(monday, tz=BERLIN).tz_convert('UTC')


def _next(start: pd.Timestamp) -> pd.Timestamp:
    local = start.tz_convert(BERLIN).date() + timedelta(days=7)
    return pd.Timestamp(local, tz=BERLIN).tz_convert('UTC')


def _smard_series(snap: pd.DataFrame, input_name: str, res: str, start: pd.Timestamp) -> list:
    end = _next(start)
    step = '1h' if res == 'hour' else '15min'
    instants = pd.date_range(start, end, freq=step, inclusive='left', tz='UTC')
    hours = instants.floor('h')
    col = 'price_eur_mwh' if input_name == 'price' else 'load_forecast_mw'
    values = snap[col].reindex(hours).to_numpy(float)
    local_day = np.array(hours.tz_convert(BERLIN).date)
    if input_name == 'price':
        era_ok = (local_day < TRANSITION_LOCAL_DATE) if res == 'hour' else (local_day >= TRANSITION_LOCAL_DATE)
        values = np.where(era_ok, values, np.nan)
    else:
        values = values / 4.0
    return [[_ms(t), None if np.isnan(v) else float(v)] for t, v in zip(instants, values)]


def render_smard(folder: Path, snap: pd.DataFrame, days) -> int:
    """The index and every bucket holding one of `days`, for both inputs and the resolutions they use."""
    replay = ReplayFolder(folder)
    last = max(snap.delivery_date)
    starts_all = []
    s = week_start(FLOOR)
    while s.tz_convert(BERLIN).date() <= last:
        starts_all.append(s)
        s = _next(s)
    if _next(starts_all[-1]).tz_convert(BERLIN).date() - timedelta(days=1) > last:
        raise ValueError('the last rendered week would reach past the snapshot')
    n = 0
    for input_name in ('price', 'load_forecast'):
        need: dict[str, set] = {}
        for d in days:
            need.setdefault(smard.resolution(input_name, d), set()).add(week_start(d))
        for res, starts in need.items():
            body = json.dumps({'timestamps': [_ms(t) for t in starts_all]}).encode()
            replay.add(smard.index_request(input_name, res),
                       Response(200, {'Content-Type': 'application/json'}, body, AR.SNAPSHOT_PULLED_UTC, 'rendered',
                                RENDERED_SMARD))
            for start in sorted(starts):
                body = json.dumps({'meta_data': {'version': 1, 'created': None},
                                   'series': _smard_series(snap, input_name, res, start)}).encode()
                replay.add(smard.bucket_request(input_name, res, _ms(start)),
                           Response(200, {'Content-Type': 'application/json'}, body, AR.SNAPSHOT_PULLED_UTC, 'rendered',
                                    RENDERED_SMARD))
                n += 1
    replay.save()
    return n


def _t(ts: pd.Timestamp) -> str:
    return ts.tz_convert('UTC').strftime('%Y-%m-%dT%H:%MZ')


def render_entsoe_day(snap: pd.DataFrame, input_name: str, day: date) -> bytes:
    hours = day_hours(day)
    start, end = hours[0], hours[-1] + pd.Timedelta(hours=1)
    col = 'price_eur_mwh' if input_name == 'price' else 'load_forecast_mw'
    values = snap[col].reindex(hours).to_numpy(float)
    if input_name == 'price' and day < TRANSITION_LOCAL_DATE:
        res, per = 'PT60M', 1
    else:
        res, per = 'PT15M', 4
    points = []
    for i, v in enumerate(values):
        if np.isnan(v):
            continue
        for q in range(per):
            pos = i * per + q + 1
            tag = 'price.amount' if input_name == 'price' else 'quantity'
            points.append(f'<Point><position>{pos}</position><{tag}>{repr(float(v))}</{tag}></Point>')
    interval = f'<timeInterval><start>{_t(start)}</start><end>{_t(end)}</end></timeInterval>'
    if input_name == 'price':
        head = ('<Publication_MarketDocument xmlns="urn:iec62325.351:tc57wg16:451-3:publicationdocument:7:3">'
                f'<mRID>cp18-rendered-A44-{day:%Y%m%d}</mRID><revisionNumber>1</revisionNumber><type>A44</type>'
                '<sender_MarketParticipant.mRID codingScheme="A01">10X1001A1001A450</sender_MarketParticipant.mRID>'
                '<sender_MarketParticipant.marketRole.type>A32</sender_MarketParticipant.marketRole.type>'
                '<receiver_MarketParticipant.mRID codingScheme="A01">10X1001A1001A450</receiver_MarketParticipant.mRID>'
                '<receiver_MarketParticipant.marketRole.type>A33</receiver_MarketParticipant.marketRole.type>'
                f'<createdDateTime>{AR.SNAPSHOT_PULLED_UTC}</createdDateTime>'
                f'<period.timeInterval><start>{_t(start)}</start><end>{_t(end)}</end></period.timeInterval>'
                '<TimeSeries><mRID>1</mRID><auction.type>A01</auction.type><businessType>A62</businessType>'
                f'<in_Domain.mRID codingScheme="A01">{entsoe.DELU}</in_Domain.mRID>'
                f'<out_Domain.mRID codingScheme="A01">{entsoe.DELU}</out_Domain.mRID>'
                '<contract_MarketAgreement.type>A01</contract_MarketAgreement.type>'
                '<currency_Unit.name>EUR</currency_Unit.name><price_Measure_Unit.name>MWH</price_Measure_Unit.name>'
                '<curveType>A01</curveType>')
        tail = '</TimeSeries></Publication_MarketDocument>'
    else:
        head = ('<GL_MarketDocument xmlns="urn:iec62325.351:tc57wg16:451-6:generationloaddocument:3:0">'
                f'<mRID>cp18-rendered-A65-{day:%Y%m%d}</mRID><revisionNumber>1</revisionNumber><type>A65</type>'
                '<process.processType>A01</process.processType>'
                '<sender_MarketParticipant.mRID codingScheme="A01">10X1001A1001A450</sender_MarketParticipant.mRID>'
                '<sender_MarketParticipant.marketRole.type>A32</sender_MarketParticipant.marketRole.type>'
                '<receiver_MarketParticipant.mRID codingScheme="A01">10X1001A1001A450</receiver_MarketParticipant.mRID>'
                '<receiver_MarketParticipant.marketRole.type>A33</receiver_MarketParticipant.marketRole.type>'
                f'<createdDateTime>{AR.SNAPSHOT_PULLED_UTC}</createdDateTime>'
                f'<time_Period.timeInterval><start>{_t(start)}</start><end>{_t(end)}</end></time_Period.timeInterval>'
                '<TimeSeries><mRID>1</mRID><businessType>A04</businessType><objectAggregation>A01</objectAggregation>'
                f'<outBiddingZone_Domain.mRID codingScheme="A01">{entsoe.DELU}</outBiddingZone_Domain.mRID>'
                '<quantity_Measure_Unit.name>MAW</quantity_Measure_Unit.name><curveType>A01</curveType>')
        tail = '</TimeSeries></GL_MarketDocument>'
    body = (f'<?xml version="1.0" encoding="utf-8"?>{head}<Period>{interval}<resolution>{res}</resolution>'
            + ''.join(points) + f'</Period>{tail}')
    return body.encode()


def render_entsoe(folder: Path, snap: pd.DataFrame, days) -> int:
    replay = ReplayFolder(folder)
    n = 0
    for d in sorted(set(days)):
        for input_name in ('price', 'load_forecast'):
            replay.add(entsoe.request(input_name, [d]),
                       Response(200, {'Content-Type': 'text/xml'}, render_entsoe_day(snap, input_name, d),
                                AR.SNAPSHOT_PULLED_UTC, 'rendered', RENDERED_ENTSOE))
            n += 1
    replay.save()
    return n


# ------------------------------------------------------------------------------------------------ verify
def _bits(a, b) -> np.ndarray:
    a, b = np.asarray(a, float), np.asarray(b, float)
    return (a.view(np.uint64) == b.view(np.uint64)) | (np.isnan(a) & np.isnan(b))


def verify(replay_root: Path, snap: pd.DataFrame, days) -> dict:
    """Every day through both clients against the snapshot rows, bit for bit."""
    days = sorted(set(days))
    out = {'days': len(days), 'first': str(days[0]), 'last': str(days[-1]), 'sources': {}}
    for source in ('smard', 'entsoe'):
        tr = ReplayTransport(Path(replay_root) / source)
        client = smard if source == 'smard' else entsoe
        stats = {'hours': 0, 'mismatched_hours': {'price': 0, 'load_forecast': 0}, 'null_hours': {'price': 0,
                 'load_forecast': 0}, 'problems': 0, 'mismatch_examples': [], 'days_23h': 0, 'days_25h': 0,
                 'quarter_hour_price_days': 0}
        groups = {}
        for d in days:
            groups.setdefault(week_start(d) if source == 'smard' else d, []).append(d)
        for _, group in sorted(groups.items()):
            for input_name in ('price', 'load_forecast'):
                fetched = client.fetch(tr, input_name, group)
                stats['problems'] += len([p for p in fetched.problems if 'hours without a value' not in p])
                col = 'price_eur_mwh' if input_name == 'price' else 'load_forecast_mw'
                for d in group:
                    hours = day_hours(d)
                    want = snap[col].reindex(hours).to_numpy(float)
                    got = fetched.values[d]
                    ok = _bits(got, want)
                    stats['mismatched_hours'][input_name] += int((~ok).sum())
                    stats['null_hours'][input_name] += int(np.isnan(want).sum())
                    if (~ok).any() and len(stats['mismatch_examples']) < 5:
                        stats['mismatch_examples'].append(f'{input_name} {d}')
        for d in days:
            n = len(day_hours(d))
            stats['hours'] += n
            stats['days_23h'] += n == 23
            stats['days_25h'] += n == 25
            stats['quarter_hour_price_days'] += d >= TRANSITION_LOCAL_DATE
        stats['bit_exact'] = not any(stats['mismatched_hours'].values())
        out['sources'][source] = stats
    out['bit_exact'] = all(s['bit_exact'] for s in out['sources'].values())
    return out


# ------------------------------------------------------------------------------------------------ archive
def _listing_dict(entry: dict | None) -> dict | None:
    if not entry:
        return None
    return {k: {**v, 'retrieved_utc': entry.get('listed_utc'),
                'note': 'listing values recorded by CP-25 (listing.jsonl)'} for k, v in entry['leads'].items()}


def archive_gfs_recorded(archive: AR.Archive, day: date, root: Path, sources=GFS_SOURCES) -> dict:
    from cp25 import runs as R
    run = day - timedelta(days=1)
    found = gfs.recorded_run_files(run, sources)
    if found is None:
        raise FileNotFoundError(f'no recorded decoded run {run}')
    js, npz = found
    listing = _listing_dict(gfs.recorded_listing(run, [js.parents[1]]))
    reconstructed = None if listing else R.reconstructed_availability(run, R.capture_table(root))
    labels = js.parents[1] / 'origin.json'
    label = (json.loads(labels.read_text()).get(run.isoformat()) if labels.exists() else None) \
        or f'{js.parents[1].parent.name}/weather/runs/{run}'
    origin = (f'recorded decoded run {label} (CP-20 decoder output, '
              'retrieved by CP-20/CP-25)' + ('; listing values from CP-25 listing.jsonl' if listing else
                                             '; no AWS listing recorded for this run'))
    return AR.store_gfs(archive, day, run=run, run_json=js.read_bytes(), run_npz=npz.read_bytes(), listing=listing,
                        reconstructed=reconstructed, provenance='recorded', origin=origin)


def archive_day(archive: AR.Archive, replay_root: Path, day: date, root: Path, source: str = 'smard',
                gfs_sources=GFS_SOURCES) -> dict:
    """One development archive day: rendered market responses through the client, recorded GFS."""
    client = smard if source == 'smard' else entsoe
    tr = ReplayTransport(Path(replay_root) / source)
    price_days = [day - timedelta(days=k) for k in (3, 2, 1) if day - timedelta(days=k) >= FLOOR]
    fp = client.fetch(tr, 'price', price_days)
    AR.store_market(archive, day, fp, development_snapshot=True)
    fl = client.fetch(tr, 'load_forecast', [day])
    AR.store_market(archive, day, fl, development_snapshot=True)
    archive_gfs_recorded(archive, day, root, gfs_sources)
    return {'day': day.isoformat(), 'problems': fp.problems + fl.problems, 'size': archive.size(day),
            'verify': archive.verify(day)}


def market_days(days) -> list[date]:
    """The market data days an archive day needs: the prices of D-3..D-1 and the load forecast of D."""
    out = set()
    for d in days:
        out |= {d - timedelta(days=k) for k in (3, 2, 1, 0) if d - timedelta(days=k) >= FLOOR}
    return sorted(out)


def dry_run_inputs(out: Path, root: Path, days, sources=GFS_SOURCES) -> dict:
    """Everything `archive` needs for `days`, in one folder that holds only repository-independent copies:
    `replay/smard` and `replay/entsoe` (rendered from the committed snapshot) and `gfs/runs/<run>.{json,npz}`
    plus `gfs/listing.jsonl` (the truly recorded decoded runs and CP-25's listing values, copied from `.local`).
    With it, `archive --replay out/replay --gfs-sources out/gfs` runs where no `.local/` exists."""
    import hashlib
    import shutil
    out = Path(out)
    days = sorted(set(days))
    snap = snapshot(root)
    mdays = market_days(days)
    render_smard(out / 'replay' / 'smard', snap, mdays)
    render_entsoe(out / 'replay' / 'entsoe', snap, mdays)
    (out / 'gfs' / 'runs').mkdir(parents=True, exist_ok=True)
    listing_lines, labels = [], {}
    for d in days:
        run = d - timedelta(days=1)
        found = gfs.recorded_run_files(run, sources)
        if found is None:
            raise FileNotFoundError(f'no recorded decoded run {run}')
        for f in found:
            shutil.copyfile(f, out / 'gfs' / 'runs' / f.name)
        labels[run.isoformat()] = f'{found[0].parents[1].parent.name}/weather/runs/{run}'
        entry = gfs.recorded_listing(run, [found[0].parents[1]])
        if entry:
            listing_lines.append(json.dumps(entry, sort_keys=True))
    (out / 'gfs' / 'listing.jsonl').write_text(''.join(line + '\n' for line in listing_lines))
    (out / 'gfs' / 'origin.json').write_text(json.dumps(labels, sort_keys=True, indent=1) + '\n')
    files = {str(p.relative_to(out)): {'sha256': hashlib.sha256(p.read_bytes()).hexdigest(), 'bytes': p.stat().st_size}
             for p in sorted(out.rglob('*')) if p.is_file() and p.name != 'manifest.json'}
    total = sum(v['bytes'] for v in files.values())
    manifest = {'schema': 'cp18a-live-dry-run-inputs-v1', 'days': [d.isoformat() for d in days],
                'market_days': [d.isoformat() for d in mdays], 'files': files, 'bytes_total': total,
                'bytes_per_day': round(total / len(days)),
                'provenance': {'replay': RENDERED_SMARD + ' | ' + RENDERED_ENTSOE,
                               'gfs': ('truly recorded: CP-20/CP-25 decoded runs (CP-20 decoder output) and CP-25 '
                                       'listing values, copied byte for byte from .local/artifacts')},
                'use': ('python -m cp18.live.recorded archive --replay <this>/replay --gfs-sources <this>/gfs '
                        '--archive <archive> --days <days> [--source smard|entsoe]')}
    (out / 'manifest.json').write_text(json.dumps(manifest, sort_keys=True, indent=1) + '\n')
    return manifest


def parse_days(text: str) -> list[date]:
    out = []
    for part in text.split(','):
        a, _, b = part.partition('..')
        a = date.fromisoformat(a)
        b = date.fromisoformat(b) if b else a
        out += [a + timedelta(days=i) for i in range((b - a).days + 1)]
    for d in out:
        guard.admit_delivery_day(d)
    return sorted(set(out))


# ------------------------------------------------------------------------------------------------ GFS client
def render_idx(run: date, lead: int, messages: list, size: int) -> bytes:
    lines = {}
    starts = {m['byte_range'][0] for m in messages}
    for m in messages:
        n, off = (int(x) for x in m['idx_line'].split(':')[:2])
        if off != m['byte_range'][0]:
            raise ValueError('recorded idx line and byte range disagree')
        lines[off] = m['idx_line']
        nxt = m['byte_range'][1] + 1
        if nxt not in starts and nxt < size:
            lines.setdefault(nxt, f'{n + 1}:{nxt}:d={run:%Y%m%d}00:RENDERED:successor of record {n} (rendered; '
                                  'the recorded index was not kept):')
    return ('\n'.join(lines[k] for k in sorted(lines)) + '\n').encode()


def render_listing(run: date, lead: int, entry: dict, idx_size: int) -> bytes:
    key = gfs.stem(run, lead)
    obj, idx = entry['object'], entry['index']

    def contents(k, item, size):
        return (f'<Contents><Key>{k}</Key><LastModified>{item["last_modified"]}</LastModified>'
                f'<ETag>{item["etag"]}</ETag><Size>{size}</Size><StorageClass>STANDARD</StorageClass></Contents>')
    return ('<?xml version="1.0" encoding="UTF-8"?><ListBucketResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/">'
            f'<Name>noaa-gfs-bdp-pds</Name><Prefix>{key}</Prefix><KeyCount>2</KeyCount><MaxKeys>1000</MaxKeys>'
            '<IsTruncated>false</IsTruncated>' + contents(key, obj, obj['size']) + contents(key + '.idx', idx, idx_size)
            + '</ListBucketResult>').encode()


def gfs_client(run: date, work: Path, root: Path, python: Path = gfs.WX_PYTHON) -> dict:
    """The client end to end on the recorded run: rendered listing and index, recorded raw messages."""
    from . import history as Hs
    t0 = time.perf_counter()
    day = run + timedelta(days=1)
    js, npz = gfs.recorded_run_files(run, (CP25_WEATHER,))
    record = json.loads(js.read_text())
    entry = gfs.recorded_listing(run, (CP25_WEATHER,))
    raw_src = CP25_WEATHER / 'raw' / run.isoformat()
    work = Path(work)
    replay = ReplayFolder(work / 'replay')
    rendered_idx_sizes = {}
    for lead in gfs.LEADS:
        key = f'f{lead:03d}'
        msgs = [m for m in record['messages'] if m['lead'] == lead]
        size = entry['leads'][key]['object']['size']
        idx = render_idx(run, lead, msgs, size)
        rendered_idx_sizes[key] = {'rendered': len(idx), 'recorded': entry['leads'][key]['index']['size']}
        replay.add(gfs.listing_request(run, lead),
                   Response(200, {'Content-Type': 'application/xml'}, render_listing(run, lead, entry['leads'][key], len(idx)),
                            entry['listed_utc'], 'rendered',
                            "rendered from CP-25's recorded listing values (keys, sizes, ETags, LastModified); the "
                            "index's Size is the rendered index's length"))
        replay.add(gfs.idx_request(run, lead),
                   Response(200, {'Content-Type': 'text/plain'}, idx, msgs[0]['retrieved_utc'], 'rendered',
                            'rendered from the recorded idx lines and byte ranges; successor lines rendered'))
        for m in msgs:
            body = (raw_src / f'{key}_{m["field"]}_aws.grib2').read_bytes()
            if gfs.sha(body) != m['sha256']:
                raise ValueError(f'{key} {m["field"]}: the retained raw message differs from its record')
            a, b = m['byte_range']
            replay.add(gfs.message_request(run, lead, a, b, None),
                       Response(206, {'Content-Range': f'bytes {a}-{b}/{size}'}, body, m['retrieved_utc'], 'recorded',
                                'the raw GRIB2 message CP-25 retrieved and retained'))
    replay.save()
    retrieval = gfs.retrieve_run(ReplayTransport(work / 'replay'), run, work / 'raw')
    want = {(m['lead'], m['field']): m for m in record['messages']}
    got = {(m['lead'], m['field']): m for m in retrieval.messages}
    fields = ('byte_range', 'idx_line', 'bytes', 'sha256', 'url')
    message_diff = [f'{k}: {f}' for k in sorted(want) for f in fields if k not in got or got[k][f] != want[k][f]]
    out = {'run': run.isoformat(), 'delivery_day': day.isoformat(), 'retrieval_problems': retrieval.problems,
           'messages_retrieved': len(retrieval.messages), 'messages_equal_to_record': not message_diff,
           'message_differences': message_diff[:10], 'rendered_index_sizes': rendered_idx_sizes,
           'recorded': {'run_json': str(js), 'npz_sha256': record['npz_sha256']}}
    if not Path(python).exists():
        out['decode'] = {'skipped': f'the ecCodes environment is absent ({python})'}
        return out
    decoded = gfs.decode_run(run, day, retrieval.messages, work / 'raw', work / 'decoded', python)
    new = json.loads(Path(decoded['json']).read_text())
    with np.load(decoded['npz']) as a, np.load(npz) as b:
        arrays_equal = {k: bool(np.array_equal(a[k], b[k], equal_nan=True) if a[k].dtype.kind == 'f'
                                else np.array_equal(a[k], b[k])) for k in ('data', 'lats', 'lons', 'fields', 'leads')}
        data_bits = bool(np.array_equal(a['data'].view(np.uint64), b['data'].view(np.uint64)))
    nm = {(m['lead'], m['field']): m for m in new['messages']}
    meta_diff = [f'{k}' for k in sorted(want) if nm[k]['meta'] != want[k]['meta']
                 or nm[k]['packing_quantum'] != want[k]['packing_quantum']]
    out['decode'] = {'decoder': decoded['decoder'], 'arrays_equal': arrays_equal, 'data_bitwise_equal': data_bits,
                     'metadata_equal': not meta_diff, 'metadata_differences': meta_diff[:10],
                     'npz_file_sha256_equal': new['npz_sha256'] == record['npz_sha256']}
    arch = AR.Archive(work / 'archive')
    AR.store_gfs_retrieval(arch, day, retrieval, decoded)
    g = Hs.gfs_for_day(arch, day)
    committed = Hs.base_weather(root, day)
    committed = committed.loc[committed.delivery_date == day].sort_values('timestamp_utc')
    mine = g['features'].sort_values('timestamp_utc')
    from cp20.weather import COLUMNS
    out['archive'] = {'verify': arch.verify(day), 'size': arch.size(day), 'gfs_available': g['available'],
                      'judgement': g['judgement'],
                      'features_bitwise_equal_to_committed': bool(np.array_equal(
                          mine[list(COLUMNS)].to_numpy(float).view(np.uint64),
                          committed[list(COLUMNS)].to_numpy(float).view(np.uint64)))}
    out['seconds'] = round(time.perf_counter() - t0, 1)
    out['pass'] = bool(not retrieval.problems and not message_diff and data_bits and not meta_diff
                       and all(arrays_equal.values()) and g['available'] and out['archive']['features_bitwise_equal_to_committed']
                       and not out['archive']['verify'])
    return out


# ------------------------------------------------------------------------------------------------ CLI
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog='python -m cp18.live.recorded')
    sub = ap.add_subparsers(dest='cmd', required=True)
    r = sub.add_parser('render')
    r.add_argument('--replay', required=True)
    r.add_argument('--days', default=None, help='A..B[,C..D]; default every development day')
    v = sub.add_parser('verify')
    v.add_argument('--replay', required=True)
    v.add_argument('--report', required=True)
    v.add_argument('--days', default=None)
    a = sub.add_parser('archive')
    a.add_argument('--replay', required=True)
    a.add_argument('--archive', required=True)
    a.add_argument('--days', required=True)
    a.add_argument('--source', choices=('smard', 'entsoe'), default='smard')
    a.add_argument('--gfs-sources', default=None,
                   help='comma-separated folders holding runs/<run>.{json,npz} [and listing.jsonl]; '
                        'default CP-25 then CP-20 under .local/artifacts')
    dr = sub.add_parser('dry-run-inputs')
    dr.add_argument('--days', required=True)
    dr.add_argument('--out', required=True)
    g = sub.add_parser('gfs-client')
    g.add_argument('--run', required=True)
    g.add_argument('--work', required=True)
    g.add_argument('--report', required=True)
    for p in (r, v, a, dr, g):
        p.add_argument('--root', default='.')
    args = ap.parse_args(argv)
    root = Path(args.root)
    t0 = time.perf_counter()
    if args.cmd in ('render', 'verify'):
        snap = snapshot(root)
        days = parse_days(args.days) if args.days else sorted(set(snap.delivery_date))
        if args.cmd == 'render':
            n1 = render_smard(Path(args.replay) / 'smard', snap, days)
            n2 = render_entsoe(Path(args.replay) / 'entsoe', snap, days)
            print(json.dumps({'smard_buckets': n1, 'entsoe_documents': n2, 'seconds': round(time.perf_counter() - t0, 1)}))
            return 0
        result = verify(Path(args.replay), snap, days)
        result.update(schema='cp18a-live-recorded-parse-v1', seconds=round(time.perf_counter() - t0, 1),
                      rendered=RENDERED_SMARD + ' | ' + RENDERED_ENTSOE)
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report).write_text(json.dumps(result, sort_keys=True, indent=1) + '\n')
        print(json.dumps({'bit_exact': result['bit_exact'], 'days': result['days'], 'seconds': result['seconds']}))
        return 0 if result['bit_exact'] else 1
    if args.cmd == 'archive':
        arch = AR.Archive(Path(args.archive))
        bad = 0
        sources = [Path(x) for x in args.gfs_sources.split(',')] if args.gfs_sources else GFS_SOURCES
        for d in parse_days(args.days):
            item = archive_day(arch, Path(args.replay), d, root, args.source, sources)
            bad += bool(item['verify'])
            print(json.dumps(item), flush=True)
        return 1 if bad else 0
    if args.cmd == 'dry-run-inputs':
        manifest = dry_run_inputs(Path(args.out), root, parse_days(args.days))
        print(json.dumps({k: manifest[k] for k in ('days', 'bytes_total', 'bytes_per_day')}))
        return 0
    result = gfs_client(date.fromisoformat(args.run), Path(args.work), root)
    result['schema'] = 'cp18a-live-gfs-client-v1'
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(result, sort_keys=True, indent=1, default=str) + '\n')
    print(json.dumps({'pass': result.get('pass'), 'seconds': result.get('seconds')}))
    return 0 if result.get('pass') else 1


if __name__ == '__main__':
    sys.exit(main())
