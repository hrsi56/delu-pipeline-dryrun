"""The input archive: per delivery day D, every response retrieved for D's build, with its evidence.

Layout (`root/days/<D>/`):

* `manifest.json` - the day's records, in the order written. Each record carries the source, the request
  (redacted: never a credential), the retrieval time UTC, the source publication time UTC and its basis (per
  covered data day), the SHA-256 and byte count of the body as served, its provenance (`live`, `recorded` or
  `rendered`) and origin, and where the body is kept.
* `bodies/<sha256>.gz` - each stored body, gzip-compressed (mtime 0); the recorded hash is the body's own.
* `gfs/runs/<run>.json`, `gfs/runs/<run>.npz` - the decoded D-1 00 UTC run in CP-20's layout.
* Raw GRIB messages are never stored here: their records keep the hash and size only (`local_only`).

The archive is append-only: a record id is written once; writing the same id again with other content is
refused. `verify(day)` re-hashes every stored body and file. Reading a day goes through
`cp25.guard.admit_delivery_day`.

The writers below turn the clients' results (`smard.fetch`, `entsoe.fetch`, `gfs.retrieve_run` + `decode_run`)
and the recorded development material into records; the publication evidence comes from `availability`.
"""
from __future__ import annotations

from datetime import date, timedelta
import gzip
import hashlib
import json
from pathlib import Path

from cp25 import guard

from . import availability as A

SCHEMA = 'cp18-live-archive-day-v1'
SNAPSHOT_ORIGIN = ('development snapshot data/snapshot.parquet (SMARD values pulled 2026-09-08T13:34:33Z), rendered '
                   "in the source's documented response shape; not a recorded raw response")
SNAPSHOT_PULLED_UTC = '2026-09-08T13:34:33Z'


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class ArchiveError(ValueError):
    pass


class Archive:
    def __init__(self, root: Path):
        self.root = Path(root)

    # ------------------------------------------------------------------ storage
    def day_dir(self, day: date) -> Path:
        guard.admit_delivery_day(day)
        return self.root / 'days' / day.isoformat()

    def has(self, day: date) -> bool:
        return (self.day_dir(day) / 'manifest.json').exists()

    def days(self) -> list[date]:
        base = self.root / 'days'
        if not base.exists():
            return []
        out = []
        for p in sorted(base.iterdir()):
            if (p / 'manifest.json').exists():
                d = date.fromisoformat(p.name)
                guard.admit_delivery_day(d)
                out.append(d)
        return out

    def manifest(self, day: date) -> dict:
        path = self.day_dir(day) / 'manifest.json'
        if not path.exists():
            return {'schema': SCHEMA, 'day': day.isoformat(), 'records': []}
        value = json.loads(path.read_text())
        if value.get('schema') != SCHEMA or value.get('day') != day.isoformat():
            raise ArchiveError(f'{path}: not an archive manifest for {day}')
        return value

    def records(self, day: date, input_name: str | None = None) -> list[dict]:
        recs = self.manifest(day)['records']
        return [r for r in recs if input_name is None or r['input'] == input_name]

    def _write(self, day: date, manifest: dict) -> None:
        d = self.day_dir(day)
        d.mkdir(parents=True, exist_ok=True)
        tmp = d / 'manifest.json.tmp'
        tmp.write_text(json.dumps(manifest, sort_keys=True, indent=1, allow_nan=False) + '\n')
        tmp.replace(d / 'manifest.json')

    def add(self, day: date, record: dict, body: bytes | None = None, files: dict | None = None) -> dict:
        """Append one record; `body` is stored compressed, `files` ({relative path: bytes}) as they are."""
        manifest = self.manifest(day)
        record = dict(record)
        for d in record.get('covers', []):
            guard.admit_delivery_day(date.fromisoformat(d))
        d = self.day_dir(day)
        if body is not None:
            record['sha256'], record['bytes'] = sha(body), len(body)
            name = f'bodies/{record["sha256"]}.gz'
            target = d / name
            if not target.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(gzip.compress(body, compresslevel=6, mtime=0))
            record['body'] = name
        else:
            record.setdefault('body', None)
        if files:
            record['files'] = {}
            for rel, data in sorted(files.items()):
                target = d / rel
                target.parent.mkdir(parents=True, exist_ok=True)
                if target.exists() and target.read_bytes() != data:
                    raise ArchiveError(f'{day} {rel}: a different file is already archived')
                target.write_bytes(data)
                record['files'][rel] = {'sha256': sha(data), 'bytes': len(data)}
        for existing in manifest['records']:
            if existing['id'] == record['id']:
                if json.dumps(existing, sort_keys=True) == json.dumps(record, sort_keys=True):
                    return existing
                raise ArchiveError(f'{day} {record["id"]}: the archive is append-only')
        manifest['records'].append(record)
        self._write(day, manifest)
        return record

    def body(self, day: date, record: dict) -> bytes:
        if not record.get('body'):
            raise ArchiveError(f'{day} {record["id"]}: no stored body')
        data = gzip.decompress((self.day_dir(day) / record['body']).read_bytes())
        if sha(data) != record['sha256'] or len(data) != record['bytes']:
            raise ArchiveError(f'{day} {record["id"]}: the stored body differs from its SHA-256')
        return data

    def file(self, day: date, record: dict, rel: str) -> Path:
        path = self.day_dir(day) / rel
        if sha(path.read_bytes()) != record['files'][rel]['sha256']:
            raise ArchiveError(f'{day} {record["id"]}: {rel} differs from its SHA-256')
        return path

    def verify(self, day: date) -> list[str]:
        """Every stored body and file re-hashed; ids unique; covers admitted. Returns the problems found."""
        problems = []
        try:
            manifest = self.manifest(day)
        except Exception as exc:  # noqa: BLE001
            return [f'{day}: manifest unreadable ({type(exc).__name__}: {exc})']
        ids = [r['id'] for r in manifest['records']]
        if len(ids) != len(set(ids)):
            problems.append(f'{day}: duplicate record ids')
        for r in manifest['records']:
            try:
                for c in r.get('covers', []):
                    guard.admit_delivery_day(date.fromisoformat(c))
                if r.get('body'):
                    self.body(day, r)
                for rel in r.get('files', {}):
                    self.file(day, r, rel)
            except Exception as exc:  # noqa: BLE001 - each failing record is reported
                problems.append(f'{day} {r["id"]}: {type(exc).__name__}: {exc}')
        return problems

    def size(self, day: date) -> dict:
        d = self.day_dir(day)
        total = sum(p.stat().st_size for p in d.rglob('*') if p.is_file())
        return {'bytes': total, 'files': sum(1 for p in d.rglob('*') if p.is_file())}


# ------------------------------------------------------------------------------------------------ writers
def _base(record_id, input_name, source, request, response, *, origin, provenance=None, covers=(), extra=None):
    out = {'id': record_id, 'input': input_name, 'source': source,
           'provenance': provenance or (response.provenance if response else 'live'), 'origin': origin,
           'request': request.redacted(), 'status': response.status if response else None,
           'retrieved_utc': response.retrieved_utc if response else None,
           'covers': [d.isoformat() for d in covers], 'publication': {}, 'document_time_utc': None,
           'note': response.note if response else ''}
    out.update(extra or {})
    return out


def store_market(archive: Archive, day: date, fetched, *, development_snapshot: bool = False) -> list[dict]:
    """Archive a SMARD or ENTSO-E fetch (`smard.Fetched` / `entsoe.Fetched`) under delivery day `day`."""
    out = []
    origin = SNAPSHOT_ORIGIN if development_snapshot else f'{fetched.source} retrieval'
    for i, ex in enumerate(fetched.exchanges):
        rid = f'{fetched.input}-{fetched.source}-{ex.role}-{i:02d}'
        resp = ex.response
        record = _base(rid, fetched.input if ex.role != 'index' else f'{fetched.input}_index', fetched.source,
                       ex.request, resp, origin=origin, covers=ex.covers,
                       extra={'resolution': ex.resolution, 'role': ex.role, 'error': ex.error})
        if resp is not None and development_snapshot and resp.provenance != 'rendered':
            raise ArchiveError('a development-snapshot record must be a rendered response')
        doc_time = ex.document_time_utc if (resp is not None and resp.provenance != 'rendered') else None
        record['document_time_utc'] = doc_time
        if resp is not None and ex.role != 'index':
            for d in ex.covers:
                if fetched.input == 'price':
                    record['publication'][d.isoformat()] = A.price_publication(d)
                else:
                    record['publication'][d.isoformat()] = A.load_publication(
                        d, provenance=resp.provenance, retrieved_utc=resp.retrieved_utc, document_time_utc=doc_time,
                        development_snapshot=development_snapshot)
        out.append(archive.add(day, record, body=resp.body if resp is not None else None))
    return out


def _listing_records(archive, day, run, listing: dict, provenance: str, origin: str, bodies: dict | None = None):
    from .gfs import listing_request
    out = []
    for key in sorted(listing):
        lead = int(key[1:])
        entry = listing[key]
        record = {'id': f'gfs-listing-{key}', 'input': 'gfs_listing', 'source': 'aws', 'provenance': provenance,
                  'origin': origin, 'request': listing_request(run, lead).redacted(), 'status': 200,
                  'retrieved_utc': entry.get('retrieved_utc'), 'covers': [(run + timedelta(days=1)).isoformat()],
                  'publication': {k: {'utc': A.iso(A.parse_iso(entry[k]['last_modified'])),
                                      'basis_id': 'aws_last_modified', 'basis': 'AWS LastModified (ListObjectsV2)',
                                      'size': entry[k]['size'], 'etag': entry[k].get('etag')}
                                  for k in ('object', 'index') if entry.get(k)},
                  'document_time_utc': None, 'note': entry.get('note', '')}
        body = (bodies or {}).get(key)
        if body is None:
            body = json.dumps({k: entry.get(k) for k in ('object', 'index', 'listed_keys', 'truncated')},
                              sort_keys=True).encode()
            record['note'] = (record['note'] + ' ' if record['note'] else '') + \
                'body: the parsed listing values as recorded (the raw ListObjectsV2 XML was not kept)'
        out.append(archive.add(day, record, body=body))
    return out


def store_gfs(archive: Archive, day: date, *, run: date, run_json: bytes, run_npz: bytes, listing: dict | None,
              reconstructed: dict | None, provenance: str, origin: str, index_bodies: dict | None = None,
              listing_bodies: dict | None = None, problems: list | None = None) -> dict:
    """Archive the D-1 00 UTC run serving `day`: listing evidence, index and message records, the decoded run."""
    from .gfs import idx_request, message_request
    guard.admit_gfs(guard.run_init(run), 21)
    if day != run + timedelta(days=1):
        raise ArchiveError('the archived run must be the D-1 00 UTC run of the day')
    record = json.loads(run_json)
    if record.get('status') != 'complete' or sha(run_npz) != record['npz_sha256'] or record['run_00z'] != run.isoformat():
        raise ArchiveError(f'{run}: the decoded run is incomplete or its arrays differ from their hash')
    if listing:
        _listing_records(archive, day, run, listing, provenance, origin, listing_bodies)
    by_lead = {}
    for m in record['messages']:
        by_lead.setdefault(m['lead'], []).append(m)
    for lead in sorted(by_lead):
        key = f'f{lead:03d}'
        first = by_lead[lead][0]
        obj = (listing or {}).get(key, {}).get('object') if listing else None
        idx_pub = (listing or {}).get(key, {}).get('index') if listing else None
        irec = {'id': f'gfs-index-{key}', 'input': 'gfs_index', 'source': 'aws', 'provenance': provenance,
                'origin': origin, 'request': idx_request(run, lead).redacted(), 'status': 200,
                'retrieved_utc': first.get('retrieved_utc'), 'covers': [day.isoformat()],
                'publication': ({'index': {'utc': A.iso(A.parse_iso(idx_pub['last_modified'])),
                                           'basis_id': 'aws_last_modified', 'basis': 'AWS LastModified (ListObjectsV2)'}}
                                if idx_pub else {}),
                'document_time_utc': None, 'target_lines': sorted({m['idx_line'] for m in by_lead[lead] if m.get('idx_line')})}
        body = (index_bodies or {}).get(key)
        if body is None:
            irec.update(sha256=first.get('idx_sha256'), bytes=idx_pub['size'] if idx_pub else None,
                        note='the index body was not recorded; its SHA-256 and target lines are from the decoded run record')
            archive.add(day, irec)
        else:
            archive.add(day, irec, body=body)
        for m in sorted(by_lead[lead], key=lambda x: x['field']):
            a, b = m['byte_range']
            mrec = {'id': f'gfs-message-{key}-{m["field"]}', 'input': 'gfs_message', 'source': m.get('endpoint', 'aws'),
                    'provenance': provenance, 'origin': origin,
                    'request': message_request(run, lead, a, b, None).redacted() if m.get('endpoint', 'aws') == 'aws'
                    else {'method': 'GET', 'url': m['url'], 'range': [a, b]},
                    'status': 206, 'retrieved_utc': m.get('retrieved_utc'), 'covers': [day.isoformat()],
                    'publication': ({'object': {'utc': A.iso(A.parse_iso(obj['last_modified'])),
                                                'basis_id': 'aws_last_modified', 'basis': 'AWS LastModified (ListObjectsV2)'}}
                                    if obj else {}),
                    'document_time_utc': None, 'sha256': m['sha256'], 'bytes': m['bytes'], 'local_only': True,
                    'packing_quantum': m.get('packing_quantum'),
                    'note': 'raw GRIB2 message kept outside the archive; hash and size only'}
            archive.add(day, mrec)
    pub = A.gfs_publication(listing, reconstructed)
    rel_json, rel_npz = f'gfs/runs/{run}.json', f'gfs/runs/{run}.npz'
    run_record = {'id': f'gfs-run-{run}', 'input': 'gfs_run', 'source': 'aws', 'provenance': provenance,
                  'origin': origin, 'request': {'run': run.isoformat(), 'leads': [21, 48], 'fields': 5},
                  'status': None, 'retrieved_utc': max((m.get('retrieved_utc') or '') for m in record['messages']) or None,
                  'covers': [day.isoformat()], 'run': run.isoformat(), 'publication': {day.isoformat(): pub},
                  'document_time_utc': None, 'sha256': record['npz_sha256'], 'bytes': len(run_npz),
                  'checks': {'messages': len(record['messages']), 'decoded': True,
                             'listed': sum(1 for v in (listing or {}).values() for k in ('object', 'index') if v.get(k)),
                             'version': record.get('version')},
                  'problems': list(problems or []), 'note': ''}
    return archive.add(day, run_record, files={rel_json: run_json, rel_npz: run_npz})


def store_gfs_retrieval(archive: Archive, day: date, retrieval, decoded: dict) -> dict:
    """Archive a live client retrieval (`gfs.retrieve_run`) and its decode (`gfs.decode_run`)."""
    listing_bodies, index_bodies = {}, {}
    for role, lead, _f, _req, resp, _err in retrieval.exchanges:
        if resp is None:
            continue
        if role == 'listing':
            listing_bodies[f'f{lead:03d}'] = resp.body
        elif role == 'index':
            index_bodies[f'f{lead:03d}'] = resp.body
    listing = {k: {**v, 'retrieved_utc': next((resp.retrieved_utc for role, lead, _f, _r, resp, _e in retrieval.exchanges
                                               if role == 'listing' and resp and f'f{lead:03d}' == k), None)}
               for k, v in retrieval.listing.items()}
    run_json = Path(decoded['json']).read_bytes()
    run_npz = Path(decoded['npz']).read_bytes()
    prov = {r.provenance for *_x, r, _e in retrieval.exchanges if r is not None}
    return store_gfs(archive, day, run=retrieval.run, run_json=run_json, run_npz=run_npz, listing=listing,
                     reconstructed=None, provenance=prov.pop() if len(prov) == 1 else 'mixed',
                     origin='GFS client retrieval (listing, index, byte ranges) and CP-20 decoder',
                     index_bodies=index_bodies, listing_bodies=listing_bodies, problems=retrieval.problems)
