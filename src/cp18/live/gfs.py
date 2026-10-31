"""GFS 0.25 deg D-1 00 UTC client: AWS listing, `.idx`, byte ranges, checks, and CP-20's decoder in a subprocess.

For the run serving delivery day D (initialized D-1 00 UTC) and each of §15.2's leads f021..f048 (every 3 h):

1. **The listing** (ListObjectsV2, one exact prefix per lead) gives each GRIB object's and index's size, ETag
   and `LastModified`: the per-run publication evidence of ruling 4 (AWS objects written after the issue time
   make the run late). Parsed with CP-25's landed `parse_listing`.
2. **The index** (`.idx`) locates CP-20's five fields (u10, v10, u100, v100, DSWRF) with CP-20's landed
   `parse_idx`, which demands exactly one record per field with the exact inventory text.
3. **The messages** are fetched by byte range, each with `If-Match` on the listed ETag, and checked: HTTP status,
   the `Content-Range` total against the listed size, the byte count, GRIB2 framing (`cp20.gfs.check_frame`) and
   the SHA-256 recorded. Raw messages stay local; the archive keeps their hashes and sizes.
4. **Decoding** runs CP-20's decoder (`cp20.gfs.decode`, field-by-field metadata validation, and
   `version_evidence`) in the ecCodes environment as a subprocess (`cp18.live.decode_worker`), writing the
   run in CP-20's layout (`runs/<run>.npz` + `runs/<run>.json`), which `cp20.weather.load_run` reads.

Every request is a `gfs`-scoped `Request`, so `cp25.guard.admit_url` refuses a run initialized at or after the
boundary, or a lead outside f021..f048, before it is sent or served. The URLs are anonymous.

`import_recorded_run` brings CP-20's or CP-25's truly recorded decoded runs (and CP-25's listing evidence) into
the archive for development days; their raw messages are not present except for CP-25's seven retained runs.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

from cp20.gfs import FIELDS, LEADS, V16_FIRST, Contradiction, IntegrityError, aws_url, check_frame, parse_idx
from cp25 import guard
from cp25.extract import parse_listing

from .transport import ReplayMiss, Request, Response, TransportError

AWS = f'https://{guard.AWS_HOST}'
PROJECT = Path(os.environ.get('CP26_PROJECT_ROOT', '/Users/djourno/Downloads/PJM'))
WX_PYTHON = PROJECT / '.local/artifacts/cp-20/wx-venv/bin/python'


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def stem(run: date, lead: int) -> str:
    sub = 'atmos/' if run >= V16_FIRST else ''
    return f'gfs.{run:%Y%m%d}/00/{sub}gfs.t00z.pgrb2.0p25.f{lead:03d}'


def listing_request(run: date, lead: int) -> Request:
    return Request('GET', AWS + '/', params=(('list-type', '2'), ('prefix', stem(run, lead))), scope=('gfs',))


def idx_request(run: date, lead: int) -> Request:
    return Request('GET', aws_url(run, lead) + '.idx', scope=('gfs',))


def message_request(run: date, lead: int, start: int, end: int, etag: str | None) -> Request:
    headers = (('If-Match', etag),) if etag else ()
    return Request('GET', aws_url(run, lead), byte_range=(start, end), headers=headers, scope=('gfs',))


def _etag(value: str | None) -> str | None:
    return value.replace('&quot;', '"') if value else None


@dataclass
class Retrieval:
    run: date
    exchanges: list = field(default_factory=list)     # (role, lead, field, Request, Response | None, error)
    listing: dict = field(default_factory=dict)       # 'fNNN' -> {'object': {...}, 'index': {...}}
    messages: list = field(default_factory=list)      # CP-20 layout provenance, plus 'raw_file'
    problems: list = field(default_factory=list)

    @property
    def complete(self) -> bool:
        return len(self.messages) == len(FIELDS) * len(LEADS) and not self.problems


def retrieve_run(transport, run: date, raw_dir: Path) -> Retrieval:
    """Listing, index and the 50 target messages of one run; raw messages written under `raw_dir/<run>/`."""
    out = Retrieval(run)
    raw = Path(raw_dir) / run.isoformat()
    raw.mkdir(parents=True, exist_ok=True)
    for lead in LEADS:
        guard.admit_gfs(guard.run_init(run), lead)
        key = f'f{lead:03d}'
        req = listing_request(run, lead)
        try:
            resp = transport.send(req)
        except (ReplayMiss, TransportError) as exc:
            out.exchanges.append(('listing', lead, None, req, None, str(exc)))
            out.problems.append(f'{key}: listing not retrieved ({exc})')
            continue
        out.exchanges.append(('listing', lead, None, req, resp, None))
        parsed = parse_listing(resp.body)
        obj, idx = parsed['objects'].get(stem(run, lead)), parsed['objects'].get(stem(run, lead) + '.idx')
        out.listing[key] = {'object': obj, 'index': idx, 'truncated': parsed['truncated'],
                            'listed_keys': sorted(parsed['objects'])}
        if resp.status != 200 or not obj or not idx or parsed['truncated']:
            out.problems.append(f'{key}: the listing does not show both the object and its index')
            continue
        req = idx_request(run, lead)
        try:
            resp = transport.send(req)
        except (ReplayMiss, TransportError) as exc:
            out.exchanges.append(('index', lead, None, req, None, str(exc)))
            out.problems.append(f'{key}: index not retrieved ({exc})')
            continue
        out.exchanges.append(('index', lead, None, req, resp, None))
        if resp.status != 200 or len(resp.body) != idx['size']:
            out.problems.append(f'{key}: index HTTP {resp.status}, {len(resp.body)} bytes, listed {idx["size"]}')
            continue
        try:
            found = parse_idx(resp.body.decode('utf8', 'replace'), run, lead, obj['size'])
        except (Contradiction, IntegrityError) as exc:
            out.problems.append(f'{key}: {type(exc).__name__}: {exc}')
            continue
        idx_sha = sha(resp.body)
        for f in FIELDS:
            start, end = found[f]['offset'], found[f]['end']
            req = message_request(run, lead, start, end, _etag(obj.get('etag')))
            try:
                resp = transport.send(req)
            except (ReplayMiss, TransportError) as exc:
                out.exchanges.append(('message', lead, f, req, None, str(exc)))
                out.problems.append(f'{key} {f}: message not retrieved ({exc})')
                continue
            out.exchanges.append(('message', lead, f, req, resp, None))
            total = (resp.headers.get('Content-Range') or '').rpartition('/')[2]
            try:
                if resp.status not in (200, 206):
                    raise IntegrityError(f'HTTP {resp.status}')
                if total and total != str(obj['size']):
                    raise IntegrityError(f'object size {total} differs from the listed {obj["size"]}')
                check_frame(resp.body, end - start + 1)
            except IntegrityError as exc:
                out.problems.append(f'{key} {f}: {exc}')
                continue
            name = f'{key}_{f}_aws.grib2'
            (raw / name).write_bytes(resp.body)
            out.messages.append({'field': f, 'lead': lead, 'endpoint': 'aws', 'url': aws_url(run, lead),
                                 'byte_range': [start, end], 'idx_line': found[f]['idx_line'], 'idx_sha256': idx_sha,
                                 'bytes': len(resp.body), 'sha256': sha(resp.body),
                                 'retrieved_utc': resp.retrieved_utc, 'purpose': 'cp18-live', 'raw_file': name})
    return out


def _env() -> dict:
    """The decoder's environment: no credential variable is passed to the subprocess."""
    keep = ('PATH', 'HOME', 'TMPDIR', 'LANG', 'LC_ALL', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS',
            'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'ECCODES_DEFINITION_PATH',
            'ECCODES_SAMPLES_PATH')
    env = {k: os.environ.get(k) for k in keep if os.environ.get(k)}
    env['PYTHONPATH'] = str(Path(__file__).resolve().parents[2])
    return env


def decode_run(run: date, delivery: date, messages: list, raw_dir: Path, out_dir: Path,
               python: Path = WX_PYTHON, timeout: float = 900.0) -> dict:
    """Decode the run's messages with CP-20's decoder in the ecCodes environment; CP-20's run layout out."""
    guard.admit_delivery_day(delivery)
    if delivery != run + timedelta(days=1):
        raise ValueError('the D-1 00 UTC run serves the delivery day after its run date')
    if not Path(python).exists():
        raise FileNotFoundError(f'the ecCodes environment is absent: {python}')
    out_dir = Path(out_dir)
    (out_dir / 'runs').mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        spec = Path(tmp) / 'spec.json'
        spec.write_text(json.dumps({'run': run.isoformat(), 'delivery_day': delivery.isoformat(),
                                    'raw_dir': str(Path(raw_dir) / run.isoformat()), 'out': str(out_dir),
                                    'messages': messages}))
        proc = subprocess.run([str(python), '-m', 'cp18.live.decode_worker', str(spec)], capture_output=True,
                              text=True, timeout=timeout, env=_env())
    if proc.returncode != 0:
        raise RuntimeError(guard.redact(f'decoder exited {proc.returncode}: {proc.stderr[-2000:]}'))
    return json.loads(proc.stdout.strip().splitlines()[-1])


# ------------------------------------------------------------------------------------------------ recorded
def recorded_run_files(run: date, sources) -> tuple[Path, Path] | None:
    """The first recorded decoded run among `sources` (CP-25's, then CP-20's weather folders)."""
    for base in sources:
        js, npz = Path(base) / 'runs' / f'{run}.json', Path(base) / 'runs' / f'{run}.npz'
        if js.exists() and npz.exists():
            return js, npz
    return None


def recorded_listing(run: date, sources) -> dict | None:
    for base in sources:
        path = Path(base) / 'listing.jsonl'
        if not path.exists():
            continue
        for line in path.read_text().splitlines():
            e = json.loads(line)
            if e['run'] == run.isoformat():
                return e
    return None


def copy_run(js: Path, npz: Path, out_dir: Path) -> tuple[Path, Path]:
    """Copy a recorded decoded run (byte for byte) into CP-20's layout under `out_dir/runs/`."""
    record = json.loads(js.read_text())
    if record.get('status') != 'complete' or sha(npz.read_bytes()) != record['npz_sha256']:
        raise ValueError(f'{js.stem}: the recorded run is incomplete or its arrays differ from their hash')
    target = Path(out_dir) / 'runs'
    target.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(js, target / js.name)
    shutil.copyfile(npz, target / npz.name)
    return target / js.name, target / npz.name
