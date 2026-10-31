"""GFS 0.25 deg D-1 00 UTC message location, integrity, decoding and metadata validation.

Runs in the pinned extraction environment (ecCodes). Only whole target GRIB2 messages
are fetched by byte range; no whole global run file is ever downloaded. AWS objects are
located through the per-file ``.idx``; NCAR d084001 has no index, so each target is
located by an adaptive jump: one resynchronisation window placed from the offsets
learned on nearby runs, local header chaining inside that window, then 320-byte header
hops. Every message is checked for GRIB2 framing, hashed, decoded locally and validated
field by field before its regional box is kept. The admission locator (read-only in
``reports/weather-admission/scripts/gfs_sample.py``) informed the header parsing.
"""
from __future__ import annotations

from dataclasses import dataclass
import datetime as dt
import hashlib
import struct
import threading

import numpy as np

LEADS = (21, 24, 27, 30, 33, 36, 39, 42, 45, 48)
FIELDS = ('u10', 'v10', 'u100', 'v100', 'dswrf')
# Fixed rectangular proxy 47-55.25 N, 5.5-15.5 E, inclusive 0.25 deg grid centres.
ROW0, ROW1 = 139, 173   # latitude 55.25 .. 47.00 (north to south, GRIB scan order)
COL0, COL1 = 22, 63     # longitude 5.50 .. 15.50
BOX_LATS = 90.0 - 0.25 * np.arange(ROW0, ROW1)
BOX_LONS = 0.25 * np.arange(COL0, COL1)
V15_FIRST, V16_FIRST = dt.date(2019, 6, 13), dt.date(2021, 3, 23)
AWS_FIRST = dt.date(2021, 1, 1)
HEADER_BYTES = 320
FIELD_ORDER = {'u10': 0, 'v10': 1, 'dswrf': 2, 'u100': 3, 'v100': 4}  # observed in-file order
GROUPS = (('u10', 'v10'), ('dswrf',), ('u100', 'v100'))


class IntegrityError(RuntimeError):
    """Bytes are not an intact target message (retryable / alternate endpoint)."""


class Contradiction(RuntimeError):
    """An intact message contradicts admission (units, level, step, grid, version...): stop."""


def version(run: dt.date) -> str:
    return 'v14' if run < V15_FIRST else ('v15.1' if run < V16_FIRST else 'v16')


def dswrf_bounds(lead: int) -> tuple[int, int]:
    """Documented six-hour-reset averaging window of the DSWRF message at this lead."""
    return (lead - 3, lead) if lead % 6 == 3 else (lead - 6, lead)


def aws_url(run: dt.date, lead: int) -> str:
    ymd = run.strftime('%Y%m%d')
    sub = 'atmos/' if run >= V16_FIRST else ''
    return f'https://noaa-gfs-bdp-pds.s3.amazonaws.com/gfs.{ymd}/00/{sub}gfs.t00z.pgrb2.0p25.f{lead:03d}'


def ncar_url(run: dt.date, lead: int) -> str:
    ymd = run.strftime('%Y%m%d')
    return (f'https://tds.gdex.ucar.edu/thredds/fileServer/files/g/d084001/{run.year}/{ymd}/'
            f'gfs.0p25.{ymd}00.f{lead:03d}.grib2')


def idx_record(field: str, lead: int) -> str:
    """Exact inventory record text; anything else is not the target."""
    if field == 'dswrf':
        a, b = dswrf_bounds(lead)
        return f':DSWRF:surface:{a}-{b} hour ave fcst:'
    name = 'UGRD' if field[0] == 'u' else 'VGRD'
    height = field[1:]
    return f':{name}:{height} m above ground:{lead} hour fcst:'


def parse_idx(text: str, run: dt.date, lead: int, size: int | None = None) -> dict:
    lines = [l for l in text.splitlines() if l.strip()]
    starts = [int(l.split(':')[1]) for l in lines]
    found = {}
    for field in FIELDS:
        want = f':d={run.strftime("%Y%m%d")}00{idx_record(field, lead)}'
        hits = [i for i, l in enumerate(lines) if l[l.index(':', l.index(':') + 1):] == want]
        if len(hits) != 1:
            raise Contradiction(f'idx has {len(hits)} records for {field} f{lead:03d}: {want}')
        i = hits[0]
        end = starts[i + 1] - 1 if i + 1 < len(lines) else (size - 1 if size else None)
        if end is None:
            raise IntegrityError('last idx record needs the object size')
        found[field] = {'offset': starts[i], 'end': end, 'idx_line': lines[i]}
    return found


def parse_header(buf: bytes) -> dict | None:
    """GRIB2 sections 0, 1, (2), 3 and 4 from the first bytes of a message."""
    if len(buf) < 16 or buf[:4] != b'GRIB' or buf[7] != 2:
        return None
    info = {'discipline': buf[6], 'length': struct.unpack('>Q', buf[8:16])[0]}
    p = 16
    while p + 5 <= len(buf):
        slen, snum = struct.unpack('>I', buf[p:p + 4])[0], buf[p + 4]
        if slen < 5:
            return None
        if snum == 1:
            if p + 17 > len(buf):
                return info
            year = struct.unpack('>H', buf[p + 12:p + 14])[0]
            info['ref'] = (year, buf[p + 14], buf[p + 15], buf[p + 16])
        elif snum == 4:
            if p + 28 > len(buf):
                return info
            info['template'] = struct.unpack('>H', buf[p + 7:p + 9])[0]
            info['cat'], info['num'] = buf[p + 9], buf[p + 10]
            info['time_unit'] = buf[p + 17]
            info['ftime'] = struct.unpack('>I', buf[p + 18:p + 22])[0]
            info['surf'] = buf[p + 22]
            scale = struct.unpack('>b', buf[p + 23:p + 24])[0]
            value = struct.unpack('>I', buf[p + 24:p + 28])[0]
            info['level'] = value / 10 ** scale if info['surf'] == 103 else None
            return info
        elif snum not in (2, 3):
            return info
        p += slen
    return info


MIN_GRIB2_BYTES = 100      # sections 0-8 of the smallest complete GRIB2 message exceed this
MIN_SCAN_START_BYTES = 2000


def plausible(info: dict | None, run: dt.date, *, chained: bool = False) -> bool:
    """A real message start: edition 2, bounded length, reference time = the run at 00 UTC.

    A header found by *scanning* a window for the magic ``GRIB`` must also exceed 2,000 bytes
    (a stray pattern inside packed data could otherwise resynchronise the chain). A header
    reached by *following* the length chain from a validated header needs only a valid
    structure: near-constant fields such as categorical ice pellets (0/1/194) are 179-656
    bytes and are genuine messages (r10; rejecting them was the "broken GRIB chain" defect).
    """
    floor = MIN_GRIB2_BYTES if chained else MIN_SCAN_START_BYTES
    return bool(info and floor < info['length'] < 20_000_000 and info.get('ref') == (run.year, run.month, run.day, 0)
                and 'template' in info)


def identify(info: dict, lead: int) -> str | None:
    d, c, n, s, t = (info.get(k) for k in ('discipline', 'cat', 'num', 'surf', 'template'))
    if (d, c, s, t) == (0, 2, 103, 0) and n in (2, 3) and info.get('level') in (10.0, 100.0) \
            and info.get('time_unit') == 1 and info.get('ftime') == lead:
        return ('u' if n == 2 else 'v') + ('10' if info['level'] == 10.0 else '100')
    if (d, c, s, t) == (0, 4, 1, 8) and n in (192, 7) and info.get('time_unit') == 1 \
            and info.get('ftime') == dswrf_bounds(lead)[0]:
        return 'dswrf'
    return None


def check_frame(msg: bytes, expected_length: int | None = None) -> None:
    if len(msg) < 16 or msg[:4] != b'GRIB' or msg[7] != 2:
        raise IntegrityError('missing GRIB2 section 0')
    total = struct.unpack('>Q', msg[8:16])[0]
    if total != len(msg) or (expected_length is not None and total != expected_length):
        raise IntegrityError(f'GRIB length {total} != bytes {len(msg)}')
    if msg[-4:] != b'7777':
        raise IntegrityError('missing end section 7777')


META_KEYS = ('shortName', 'units', 'typeOfLevel', 'level', 'stepType', 'startStep', 'endStep', 'stepUnits',
             'dataDate', 'dataTime', 'validityDate', 'validityTime', 'gridType', 'Ni', 'Nj',
             'latitudeOfFirstGridPointInDegrees', 'longitudeOfFirstGridPointInDegrees',
             'latitudeOfLastGridPointInDegrees', 'longitudeOfLastGridPointInDegrees',
             'iDirectionIncrementInDegrees', 'jDirectionIncrementInDegrees', 'jScansPositively',
             'iScansNegatively', 'discipline', 'parameterCategory', 'parameterNumber',
             'productDefinitionTemplateNumber', 'typeOfFirstFixedSurface', 'packingType', 'bitsPerValue',
             'binaryScaleFactor', 'decimalScaleFactor', 'referenceValue', 'numberOfMissing',
             'numberOfDataPoints', 'centre', 'subCentre', 'generatingProcessIdentifier',
             'typeOfGeneratingProcess', 'significanceOfReferenceTime', 'productionStatusOfProcessedData',
             'typeOfProcessedData', 'editionNumber')
PDT8_KEYS = ('typeOfStatisticalProcessing', 'lengthOfTimeRange', 'indicatorOfUnitForTimeRange')
MISSING = 1.0e20


def expected_meta(run: dt.date, lead: int, field: str) -> dict:
    valid = dt.datetime(run.year, run.month, run.day) + dt.timedelta(hours=lead)
    common = {'dataDate': int(run.strftime('%Y%m%d')), 'dataTime': 0, 'stepUnits': 1,
              'validityDate': int(valid.strftime('%Y%m%d')), 'validityTime': valid.hour * 100,
              'gridType': 'regular_ll', 'Ni': 1440, 'Nj': 721, 'latitudeOfFirstGridPointInDegrees': 90.0,
              'longitudeOfFirstGridPointInDegrees': 0.0, 'latitudeOfLastGridPointInDegrees': -90.0,
              'longitudeOfLastGridPointInDegrees': 359.75, 'iDirectionIncrementInDegrees': 0.25,
              'jDirectionIncrementInDegrees': 0.25, 'jScansPositively': 0, 'iScansNegatively': 0,
              'discipline': 0, 'centre': 'kwbc', 'generatingProcessIdentifier': 96,
              'typeOfGeneratingProcess': 2, 'significanceOfReferenceTime': 1,
              'productionStatusOfProcessedData': 0, 'editionNumber': 2, 'numberOfDataPoints': 1440 * 721}
    if field == 'dswrf':
        a, b = dswrf_bounds(lead)
        return {**common, 'parameterCategory': 4, 'productDefinitionTemplateNumber': 8,
                'typeOfFirstFixedSurface': 1, 'typeOfLevel': 'surface', 'units': 'W m**-2', 'stepType': 'avg',
                'startStep': a, 'endStep': b, 'typeOfStatisticalProcessing': 0, 'lengthOfTimeRange': b - a,
                'indicatorOfUnitForTimeRange': 1}
    return {**common, 'parameterCategory': 2, 'parameterNumber': 2 if field[0] == 'u' else 3,
            'productDefinitionTemplateNumber': 0, 'typeOfFirstFixedSurface': 103,
            'typeOfLevel': 'heightAboveGround', 'level': int(field[1:]), 'units': 'm s**-1',
            'stepType': 'instant', 'startStep': lead, 'endStep': lead}


def validate(meta: dict, run: dt.date, lead: int, field: str) -> None:
    """Refuse any decoded message whose identity, units, level, time or grid is not the target."""
    want = expected_meta(run, lead, field)
    wrong = {k: (meta.get(k), v) for k, v in want.items() if meta.get(k) != v}
    if field == 'dswrf' and meta.get('parameterNumber') not in (192, 7):
        wrong['parameterNumber'] = (meta.get('parameterNumber'), '192 or 7')
    if meta.get('typeOfProcessedData') not in (1, None):  # forecast product (1) when encoded
        wrong['typeOfProcessedData'] = (meta.get('typeOfProcessedData'), 1)
    if wrong:
        raise Contradiction(f'{run} f{lead:03d} {field} metadata contradicts admission: {wrong}')


# Code-table keys are compared as WMO/NCEP integer codes; ecCodes' native string form is an
# abbreviation (e.g. typeOfFirstFixedSurface 103 -> 'sfc', typeOfProcessedData 1 -> 'fc').
INT_KEYS = frozenset({'typeOfFirstFixedSurface', 'typeOfProcessedData', 'stepUnits', 'discipline',
                      'parameterCategory', 'parameterNumber', 'productDefinitionTemplateNumber',
                      'productionStatusOfProcessedData', 'significanceOfReferenceTime', 'typeOfGeneratingProcess',
                      'generatingProcessIdentifier', 'typeOfStatisticalProcessing', 'indicatorOfUnitForTimeRange',
                      'jScansPositively', 'iScansNegatively', 'editionNumber', 'subCentre'})
ABBREVIATED = ('typeOfFirstFixedSurface', 'typeOfProcessedData', 'centre')


def read_meta(gid, field: str) -> dict:
    import eccodes
    meta = {}
    for key in META_KEYS + (PDT8_KEYS if field == 'dswrf' else ()):
        try:
            meta[key] = eccodes.codes_get(gid, key, ktype=int) if key in INT_KEYS else eccodes.codes_get(gid, key)
        except eccodes.KeyValueNotFoundError:
            meta[key] = None
    for key in ABBREVIATED:
        try:
            meta[f'{key}_native'] = eccodes.codes_get(gid, key)
        except eccodes.KeyValueNotFoundError:
            meta[f'{key}_native'] = None
    return meta


def decode(msg: bytes, run: dt.date, lead: int, field: str):
    """Decode locally; validate every expected key; return (metadata, 34x41 float64 box, quantum)."""
    import eccodes
    try:
        gid = eccodes.codes_new_from_message(msg)
    except Exception as exc:  # noqa: BLE001
        raise IntegrityError(f'ecCodes cannot open message: {exc}') from exc
    try:
        meta = read_meta(gid, field)
        eccodes.codes_set(gid, 'missingValue', MISSING)
        values = eccodes.codes_get_values(gid)
    except Contradiction:
        raise
    except Exception as exc:  # noqa: BLE001
        raise IntegrityError(f'decode failure: {exc}') from exc
    finally:
        eccodes.codes_release(gid)
    validate(meta, run, lead, field)
    if values.shape != (1440 * 721,):
        raise Contradiction('unexpected grid size')
    grid = values.reshape(721, 1440)
    box = np.array(grid[ROW0:ROW1, COL0:COL1], dtype=np.float64)
    box[box == MISSING] = np.nan
    e, d = meta['binaryScaleFactor'], meta['decimalScaleFactor']
    quantum = float(2.0 ** e / 10.0 ** d)
    meta['packing_quantum'] = quantum
    meta['box_nonfinite'] = int((~np.isfinite(box)).sum())
    return meta, box, quantum


def version_evidence(run: dt.date, field: str, meta: dict, endpoint: str, url: str) -> dict:
    """Checks that tie the message to the documented model version by date."""
    v = version(run)
    checks = {'expected_version': v}
    if endpoint == 'aws':
        checks['aws_path_matches_version'] = ('/atmos/' in url) == (v == 'v16')
    if field == 'dswrf':
        coarse = meta['packing_quantum'] >= 1.0
        checks['dswrf_precision_matches_version'] = coarse == (v != 'v16')
    if not all(x for k, x in checks.items() if k != 'expected_version'):
        raise Contradiction(f'{run} {field} version evidence contradicts {v}: {checks}')
    return checks


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# ---------------------------------------------------------------- NCAR adaptive locator
@dataclass
class Located:
    offset: int
    length: int
    header: dict


class OffsetModel:
    """Learned byte fractions per (version, lead, group anchor), shared by worker threads."""

    def __init__(self, priors: dict[tuple, list[tuple[dt.date, float]]], learned=None):
        self._prior = {k: list(v) for k, v in priors.items()}
        self._obs = {k: list(v) for k, v in (learned or {}).items()}
        self._err = {}
        self._lock = threading.Lock()

    def predict(self, v, lead, anchor, run):
        """Nearest-date learned fraction for this version/lead/anchor, else the admission prior."""
        with self._lock:
            obs = self._obs.get((v, lead, anchor))
            source = 'learned'
            if not obs:
                source = 'prior'
                obs = self._prior.get((v, lead, anchor)) or [o for (vv, _, a), lst in self._prior.items()
                                                             if vv == v and a == anchor for o in lst]
            if not obs:
                return None, None, None
            nearest = min(obs, key=lambda o: abs((o[0] - run).days))
            errs = self._err.get((v, lead, anchor), [])
            return nearest[1], (max(errs[-8:]) if errs and source == 'learned' else None), source

    def learn(self, v, lead, anchor, run, frac, error_bytes, source):
        with self._lock:
            self._obs.setdefault((v, lead, anchor), []).append((run, frac))
            if source == 'learned':
                self._err.setdefault((v, lead, anchor), []).append(abs(error_bytes))

    def observations(self):
        with self._lock:
            return [{'key': list(k), 'run': r.isoformat(), 'frac': f} for k, lst in self._obs.items() for r, f in lst]


class NcarLocator:
    MIN_BACK, MAX_BACK, FORWARD, MAX_HOPS, WINDOW = 300_000, 1_000_000, 300_000, 40, 1_150_000
    EXTEND, MAX_EXTEND = 600_000, 3

    def __init__(self, fetcher, model: OffsetModel):
        self.fetcher, self.model = fetcher, model

    def _get(self, url, start, end, size, purpose):
        _, headers, body = self.fetcher.get(url, purpose, byte_range=(start, end))
        total = headers.get('Content-Range', '').rpartition('/')[2]
        if total != str(size):
            raise Contradiction(f'{url} size {total} differs from the admission inventory size {size}')
        return body

    def _window(self, url, run, start, length, size, purpose):
        """Read a resynchronisation window, extending it only while it holds no message start."""
        end = min(size - 1, start + length - 1)
        body = self._get(url, start, end, size, purpose)
        for _ in range(self.MAX_EXTEND):
            if self._first_header(start, body, run) is not None or start + len(body) >= size:
                break
            nxt = start + len(body)
            body += self._get(url, nxt, min(size - 1, nxt + self.EXTEND - 1), size, purpose + ' extension')
        return start, body

    def _first_header(self, base, body, run):
        i = 0
        while True:
            j = body.find(b'GRIB', i)
            if j < 0 or j + HEADER_BYTES > len(body):
                return None
            if plausible(parse_header(body[j:j + HEADER_BYTES]), run):
                return base + j
            i = j + 1

    def find_group(self, url, run, lead, size, group, trace):
        v, anchor = version(run), group[0]
        frac, err, source = self.model.predict(v, lead, anchor, run)
        pred = int(frac * size) if frac is not None else None
        if err is not None:  # learned with an error history: a bounded window around the prediction
            back = min(self.MAX_BACK, max(self.MIN_BACK, int(1.25 * err)))
            first_length = back + self.FORWARD
        else:
            back, first_length = 2_500_000, self.WINDOW
        attempts = [(back, first_length), (4_000_000, self.WINDOW + 300_000), (9_000_000, self.WINDOW + 300_000)] \
            if pred is not None else []
        for k, (margin, length) in enumerate(attempts):
            start = max(0, pred - margin)
            base, win = self._window(url, run, start, length, size, f'ncar window {run} f{lead:03d} {anchor}')
            off = self._first_header(base, win, run)
            hops, found, overshoot = 0, {}, False
            while off is not None and off < size:
                rel = off - base
                if 0 <= rel and rel + HEADER_BYTES <= len(win):
                    info = parse_header(win[rel:rel + HEADER_BYTES])
                else:
                    if hops >= self.MAX_HOPS:
                        break
                    _, _, hb = self.fetcher.get(url, f'ncar hop {run} f{lead:03d} {anchor}',
                                                byte_range=(off, min(size - 1, off + HEADER_BYTES - 1)))
                    info = parse_header(hb)
                    hops += 1
                if not plausible(info, run, chained=True):
                    raise IntegrityError(f'broken GRIB chain at {off}')
                name = identify(info, lead)
                if name is not None and name not in group and FIELD_ORDER[name] > FIELD_ORDER[group[-1]]:
                    overshoot = True
                    break
                if name in group and name not in found:
                    if name != group[len(found)]:
                        overshoot = True  # partner seen before anchor
                        break
                    found[name] = Located(off, info['length'], info)
                    if len(found) == len(group):
                        break
                off += info['length']
            trace.append({'group': list(group), 'try': k, 'window_start': base, 'window_bytes': len(win),
                          'predicted': pred, 'prediction_source': source, 'hops': hops, 'found': list(found),
                          'overshoot': overshoot,
                          'error_bytes': found[anchor].offset - pred if anchor in found else None,
                          'contiguous': (len(found) < 2 or found[group[0]].offset + found[group[0]].length
                                         == found[group[1]].offset)})
            if len(found) == len(group):
                error = found[anchor].offset - pred
                self.model.learn(v, lead, anchor, run, found[anchor].offset / size, error, source)
                return found, (base, win)
        raise IntegrityError(f'{run} f{lead:03d}: could not locate {group} on NCAR')

    def payload(self, url, run, lead, located, window, purpose):
        """Fetch the contiguous group, reusing any bytes already held in the window."""
        first = min(x.offset for x in located.values())
        last = max(x.offset + x.length - 1 for x in located.values())
        base, win = window
        held = b''
        if base <= first < base + len(win):
            held = win[first - base:min(len(win), last - base + 1)]
        rest = b''
        if first + len(held) <= last:
            _, _, rest = self.fetcher.get(url, purpose, byte_range=(first + len(held), last))
        blob = held + rest
        if len(blob) != last - first + 1:
            raise IntegrityError('group payload length mismatch')
        return first, blob, len(held)
