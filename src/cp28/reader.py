"""Read-only access to CP-26's stored outputs, bound by hash to the committed manifests.

Nothing here writes to CP-26's paths or calls `src/cp26` (whose live identity guards refuse
under r17 by design). Every stored file is checked against `scores-manifest.json` (itself pinned
below) before a byte of it is used: the forecast ledger file by file, and the local-only analysis
outputs by the hashes the brief and the manifest record. The member arithmetic is CP-26's own
(`cp26.replay._learning_members`): c_HG = A1_w/2 + B2_w/2, L = L-N/2 + L-R/2, D2 its median.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd

from cp25 import guard

RUN = 'reports/rebaseline/cold-rerun-2026-10-15'
SCORES_MANIFEST = f'{RUN}/scores-manifest.json'
SCORES_MANIFEST_SHA256 = 'b9f10917ab471177eda1b5f8ef735ddca5a651fa2e9bb47844497f3a6deaf84d'
REPORT_JSON_SHA256 = '1b1aa805a066cda2da142cb9039fe63dd7a02c9b859f8129bd29b943ec1756d8'
SNAPSHOT = 'data/snapshot.parquet'
SNAPSHOT_SHA256 = '7dd2dc73407706ca6bd3c1ad51d201ac0de35eec5ca129ce320cca639f697f00'
CP25_MANIFEST = 'reports/development-inputs/manifest.json'
CP25_MANIFEST_SHA256 = 'f3e8ca15e4466d371de5b6db9c181f77adaa8823ca2e7e0d325182120db32fd9'
# Where the brief puts the extracted ledger and CP-26's local analysis outputs, under `.local/`.
INPUTS = 'artifacts/cp-28/inputs'
LOCAL_ANALYSIS = 'artifacts/cp-26/cold-rerun-2026-10-15/r16/final-analysis-inputs'
LEVELS = np.array([.025, .10, .25, .50, .75, .90, .975])
MEMBERS = ('c_HG', 'L', 'D2')
SOURCE_MEMBERS = ('A1_w', 'B2_w', 'L-N', 'L-R', 'D2')


def sha256(path) -> str:
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def content_digest(value) -> str:
    """CP-26's `state.digest`: canonical sorted JSON, no NaN."""
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def local_root(root) -> Path | None:
    """The `.local/` that holds the extracted inputs: `$CP28_LOCAL_ROOT`, this tree's, or the primary checkout's."""
    if os.environ.get('CP28_LOCAL_ROOT'):
        path = Path(os.environ['CP28_LOCAL_ROOT'])
        return path if (path / INPUTS).exists() else None
    own = Path(root) / '.local'
    if (own / INPUTS).exists():
        return own
    try:
        common = subprocess.run(['git', '-C', str(root), 'rev-parse', '--path-format=absolute', '--git-common-dir'],
                                capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None
    primary = Path(common).parent / '.local'
    return primary if (primary / INPUTS).exists() else None


def require(path, expected: str, what: str) -> Path:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f'{what} is absent: {path}')
    actual = sha256(path)
    if actual != expected:
        raise ValueError(f'{what} hash mismatch: {actual} != {expected}')
    return path


def scores_manifest(root) -> dict:
    return json.loads(require(Path(root) / SCORES_MANIFEST, SCORES_MANIFEST_SHA256, 'scores manifest').read_text())


def table_path(root, local, name: str) -> Path:
    """A CP-26 score table, committed or local-only, verified against the manifest."""
    entry = scores_manifest(root)['tables'][name]
    committed = Path(root) / entry['path']
    path = committed if committed.exists() else Path(local) / LOCAL_ANALYSIS / entry['path']
    return require(path, entry['sha256'], f'CP-26 table {name}')


def read_table(root, local, name: str) -> pd.DataFrame:
    frame = pd.read_parquet(table_path(root, local, name))
    if 'delivery_date' in frame and len(frame) and isinstance(frame.delivery_date.iloc[0], str):
        frame['delivery_date'] = [date.fromisoformat(json.loads(v)['__date__'] if v.startswith('{') else v)
                                  for v in frame.delivery_date]
    return frame


def report_json(local) -> dict:
    path = Path(local) / LOCAL_ANALYSIS / RUN / 'report.json'
    return json.loads(require(path, REPORT_JSON_SHA256, 'CP-26 report.json').read_text())


def cp25_file(root, relative: str) -> Path:
    manifest = json.loads(require(Path(root) / CP25_MANIFEST, CP25_MANIFEST_SHA256, 'CP-25 manifest').read_text())
    return require(Path(root) / relative, manifest['outputs_sha256'][relative], f'CP-25 {relative}')


# ----------------------------------------------------------------------------- the forecast ledger

def ledger_entries(root) -> list[dict]:
    entries = scores_manifest(root)['forecast_ledger']
    days = [entry['day'] for entry in entries]
    if days != sorted(set(days)):
        raise ValueError('forecast ledger entries must be unique and chronological')
    return entries


def verify_ledger(root, local, *, git_tag: str | None = None) -> dict:
    """Bind every stored ledger file at entry (protocol §1); optionally against the tag's tree too."""
    entries = ledger_entries(root)
    base = Path(local) / INPUTS
    blobs = None
    if git_tag is not None:
        listing = subprocess.run(['git', '-C', str(root), 'ls-tree', '-r', git_tag, f'{RUN}/forecast-ledger/'],
                                 capture_output=True, text=True, check=True).stdout.split('\n')
        blobs = {line.split('\t')[1]: line.split()[2] for line in listing if line.strip()}
        if set(blobs) != {entry['path'] for entry in entries}:
            raise ValueError('the tag tree and the scores manifest list different ledger files')
    total, rows = 0, []
    for entry in entries:
        path = base / entry['path']
        data = path.read_bytes()
        if hashlib.sha256(data).hexdigest() != entry['sha256']:
            raise ValueError(f'ledger file hash mismatch: {entry["path"]}')
        if blobs is not None:
            blob = hashlib.sha1(b'blob %d\0' % len(data) + data).hexdigest()
            if blob != blobs[entry['path']]:
                raise ValueError(f'ledger file differs from the tag tree: {entry["path"]}')
        value = json.loads(data)
        if value['content_sha256'] != entry['content_sha256'] or value['day'] != entry['day'] \
                or content_digest({k: v for k, v in value.items() if k != 'content_sha256'}) != value['content_sha256']:
            raise ValueError(f'ledger content identity mismatch: {entry["path"]}')
        total += len(data)
        rows.append(entry['sha256'])
    return {'files': len(entries), 'bytes': total, 'first_day': entries[0]['day'], 'last_day': entries[-1]['day'],
            'manifest': SCORES_MANIFEST, 'manifest_sha256': SCORES_MANIFEST_SHA256,
            'file_hashes_sha256': hashlib.sha256('\n'.join(rows).encode()).hexdigest(),
            'git_tag_checked': git_tag}


def _vector(value, n):
    if value is None:
        return None
    array = np.asarray(value, dtype=float)
    return array if array.shape == (n,) and np.isfinite(array).all() else None


def _quantiles(value, n):
    if value is None:
        return None
    array = np.asarray(value, dtype=float)
    if array.shape != (n, 7) or not np.isfinite(array).all() or (np.diff(array, axis=1) < 0).any():
        return None
    return array


def _keys(values) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(pd.to_datetime(values, utc=True)).as_unit('ns')


def canonical_hours(day: date) -> pd.DatetimeIndex:
    return pd.date_range(pd.Timestamp(day, tz='Europe/Berlin'), pd.Timestamp(day, tz='Europe/Berlin') + pd.offsets.Day(1),
                         freq='h', inclusive='left').tz_convert('UTC').as_unit('ns')


@dataclass
class Day:
    """One delivery date of CP-26's nominal ledger, reduced to what CP-28 may use."""
    day: date
    issue_utc: str
    index: pd.DatetimeIndex               # the nominal issuance keys
    scoring_index: pd.DatetimeIndex       # CP-25's frozen eligible keys (may be partial or empty)
    scored: bool
    complete: bool                        # issuance keys equal the canonical 23/24/25 hours
    members: dict = field(default_factory=dict)     # c_HG, L, D2 -> vector or None
    member_status: dict = field(default_factory=dict)
    d2_quantiles: np.ndarray | None = None          # D2's own Johnson SU seven quantiles, native
    scale: np.ndarray | None = None                 # the A1 price scale issued with this date
    v5_source: str | None = None
    v5_quantiles: np.ndarray | None = None          # v5's emitted quantiles on scoring keys
    b0_source: str | None = None
    b0_quantiles: np.ndarray | None = None          # B0's emitted quantiles on scoring keys
    identity_ok: bool = True

    @property
    def all_members_valid(self) -> bool:
        return self.identity_ok and all(self.members.get(name) is not None for name in MEMBERS)


def parse_day(value: dict, protocol_identity: dict) -> Day:
    day = date.fromisoformat(value['day'])
    guard.admit_delivery_day(day)
    issue = guard.issue_time(day).isoformat()
    nominal = value['policies']['nominal/v5']
    index, scoring = _keys(nominal['index']), _keys(nominal['scoring_index'])
    n = len(index)
    lineage_issue = value['lineage']['nominal']['issue_utc']
    identity_ok = (pd.Timestamp(lineage_issue) == pd.Timestamp(issue)
                   and value['protocol_identity'] == protocol_identity
                   and value['lineage']['nominal']['target_day_embargo'] == str(day))
    sources, status = {}, {}
    for name in SOURCE_MEMBERS:
        item = value['members']['nominal'].get(name)
        ok = (item is not None and item.get('status') == 'ok'
              and pd.Timestamp(item['lineage']['issue_utc']) == pd.Timestamp(issue))
        sources[name] = _vector(item['central'], n) if ok else None
        status[name] = item.get('status') if item is not None else 'absent'
    members = {
        'c_HG': None if sources['A1_w'] is None or sources['B2_w'] is None else sources['A1_w'] / 2 + sources['B2_w'] / 2,
        'L': None if sources['L-N'] is None or sources['L-R'] is None else sources['L-N'] / 2 + sources['L-R'] / 2,
        'D2': sources['D2'],
    }
    d2 = value['members']['nominal'].get('D2')
    d2_q = _quantiles(d2.get('quantiles'), n) if d2 is not None and sources['D2'] is not None else None
    if d2_q is not None and not np.array_equal(d2_q[:, 3], sources['D2']):
        d2_q = None
    # CP-26's own v3 central is c_HG bit for bit; the scale is issued with every valid H central.
    scale = None
    for policy in ('nominal/v3', 'nominal/v5', 'nominal/v4', 'nominal/v2'):
        record = value['policies'][policy]
        if record.get('native_scale_valid') and record.get('issued_scale') is not None:
            scale = _vector(record['issued_scale'], n)
            if scale is not None and (scale < 1).any():
                scale = None
            if scale is not None:
                break
    v3 = value['policies']['nominal/v3']
    if members['c_HG'] is not None and v3.get('native_central_valid') and not np.array_equal(v3['native_central'], members['c_HG']):
        raise ValueError(f'{day}: c_HG differs from CP-26 v3 native central')
    b0 = value['policies']['nominal/B0']
    m = len(scoring)
    return Day(day=day, issue_utc=issue, index=index, scoring_index=scoring, scored=bool(nominal['scored']),
               complete=index.equals(canonical_hours(day)), members=members, member_status=status, d2_quantiles=d2_q,
               scale=scale, v5_source=nominal['output_source'], v5_quantiles=_quantiles(nominal['emitted_quantiles'], m),
               b0_source=b0['output_source'], b0_quantiles=_quantiles(b0['emitted_quantiles'], m), identity_ok=identity_ok)


def load_days(root, local) -> list[Day]:
    """Every ledger date, each file re-verified against the manifest as it is read."""
    base = Path(local) / INPUTS
    manifest = scores_manifest(root)
    identity = None
    result = []
    for entry in ledger_entries(root):
        data = (base / entry['path']).read_bytes()
        if hashlib.sha256(data).hexdigest() != entry['sha256']:
            raise ValueError(f'ledger file hash mismatch: {entry["path"]}')
        value = json.loads(data)
        if value['execution_manifest_sha256'] != manifest['execution_manifest_sha256']:
            raise ValueError(f'{entry["day"]}: ledger file belongs to another execution')
        identity = value['protocol_identity'] if identity is None else identity
        result.append(parse_day(value, identity))
    return result


# ----------------------------------------------------------------------------------------- prices

def prices(root) -> pd.Series:
    """Development prices behind the boundary guard, indexed by canonical UTC key."""
    path = require(Path(root) / SNAPSHOT, SNAPSHOT_SHA256, 'snapshot')
    frame = pd.read_parquet(path, columns=['timestamp_utc', 'delivery_date', 'price_eur_mwh'],
                            filters=[('delivery_date', '>=', guard.FLOOR), ('delivery_date', '<=', guard.BOUNDARY)])
    guard.admit_rows(pd.DatetimeIndex(frame.timestamp_utc).to_pydatetime())
    series = pd.Series(frame.price_eur_mwh.to_numpy(float), index=_keys(frame.timestamp_utc), name='price')
    if series.index.has_duplicates:
        raise ValueError('duplicate price keys')
    return series.sort_index()


def truth_for(series: pd.Series, index: pd.DatetimeIndex) -> np.ndarray:
    return series.reindex(index).to_numpy(float)
