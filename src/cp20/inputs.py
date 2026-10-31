"""Identity-bound inputs for CP-20: issued v21-r4 documents, inherited CP-15/CP-16 artifacts,
partition filtering before materialisation, and verified H0 component caches."""
from __future__ import annotations

from datetime import date
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd

from cp15.data import RAW_COLUMNS, array_hash, origin_utc, prepare, sha
from cp16.inputs import SavedComponents
from delu_forecast.folds import load_partition_spec

ISSUED = {
    'capstone_v21.md': '150bd53fa15067b1d138c95d0912f86a1e92ce74092059be09034758c1926167',
    'docs/track-b/capstone_v21-r3-to-v21-r4-amendments.md': '3e0bdf6a343d5336f406f0eeb726b1b65c0f767ec8d9418a7d8118902e1ace55',
    'docs/track-b/cp-20-direct-weather-brief.md': '28a4e195fa7f66ff3270d5d54e477478e90aa42124bceb1ee529f95f5761883e'}
# CP-15 protocol's rulebook/anchor identities live at the historical CP-15 evidence commit.
CP15_EVIDENCE = '1bdc75b8ab943092bb8de6ba893defb9e12250d8'
CP16_MANIFEST = 'reports/v2-causal/artifact-manifest.json'
CP16_REQUIRED = ('reports/v2-causal/predictions.parquet', 'reports/v2-causal/lineage.json',
                 'reports/v2-causal/input-manifest.json', 'reports/v2-causal/protocol.json',
                 'src/cp16/residuals.py', 'src/cp16/inputs.py', 'src/cp16/scoring.py')
ORIGINS = 638
FOLD_COUNTS = [2160, 2159, 2112, 2160, 2156]


def identities(root: Path) -> tuple[dict, dict]:
    root = Path(root)
    result = {}
    for name, expected in ISSUED.items():
        if sha(root / name) != expected:
            raise ValueError(f'issued CP-20 identity mismatch: {name}')
        result[name] = expected
    p = json.loads((root / 'reports/cp15/protocol.json').read_text())
    for name, expected in p['input_sha256'].items():
        if name in ('AGENTS.md', 'capstone_v21.md'):
            blob = subprocess.check_output(['git', 'show', f'{CP15_EVIDENCE}:{name}'], cwd=root)
            if hashlib.sha256(blob).hexdigest() != expected:
                raise ValueError(f'inherited historical identity mismatch: {name}')
            result[f'{CP15_EVIDENCE}:{name}'] = expected
        elif sha(root / name) != expected:
            raise ValueError(f'inherited CP-15 input mismatch: {name}')
        else:
            result[name] = expected
    cp15 = json.loads((root / 'reports/cp15/artifact-manifest.json').read_text())['artifact_sha256']
    names = ['src/cp15/data.py', 'src/cp15/models.py', 'reports/cp15/protocol.json', 'reports/cp15/predictions.parquet']
    names += [f'reports/cp15/folds/fold_{f}-{s}' for f in range(1, 6) for s in ('issued.parquet', 'fits.parquet', 'run.json')]
    for name in names:
        if sha(root / name) != cp15[name]:
            raise ValueError(f'saved CP-15 artifact identity mismatch: {name}')
        result[name] = cp15[name]
    cp16 = json.loads((root / CP16_MANIFEST).read_text())['artifact_sha256']
    for name in CP16_REQUIRED:
        if sha(root / name) != cp16[name]:
            raise ValueError(f'accepted CP-16 artifact identity mismatch: {name}')
        result[name] = cp16[name]
    return p, result


def load(root: Path, before: date | None = None):
    """Permitted partitions only, filtered before materialisation (never past the EDA cutoff)."""
    p, hashes = identities(root)
    spec = load_partition_spec(root / 'data/partitions.json')
    filters = [('delivery_date', '>=', date(2019, 1, 1)), ('delivery_date', '<=', spec.eda_cutoff)]
    if before is not None:
        filters.append(('delivery_date', '<', before))
    frame = pd.read_parquet(root / 'data/snapshot.parquet', columns=RAW_COLUMNS, filters=filters)
    return prepare(frame, p, spec), hashes


def origin_manifest(root: Path) -> dict:
    m = json.loads((Path(root) / 'reports/v2-causal/input-manifest.json').read_text())
    if sum(f['date_count'] for f in m['folds']) != ORIGINS:
        raise ValueError('frozen CP-16 origin manifest changed')
    if [len(f['original_target_keys']) for f in m['folds']] != FOLD_COUNTS:
        raise ValueError('original target key counts changed')
    return m


class H0Components:
    """Identity-verified no-weather A1/B2 central forecasts: CP-15 saved components first,
    then CP-16's content-hashed new components. H0 never fits inside the replay."""

    def __init__(self, root: Path, fold: str, data):
        self.fold, self.data = fold, data
        self.saved = SavedComponents(root, fold, data)
        lineage = json.loads((Path(root) / 'reports/v2-causal/lineage.json').read_text())
        self.protocol, self.fingerprint = lineage['protocol_sha256'], lineage['input_fingerprint']
        self.new = {k: v for k, v in lineage['new_components'].items() if v['fold'] == fold}

    @staticmethod
    def digest(item):
        return hashlib.sha256(json.dumps({k: v for k, v in item.items() if k != 'content_sha256'}, sort_keys=True,
                                         allow_nan=False, default=str).encode()).hexdigest()

    def get(self, day: date):
        rows = self.data.rows(day)
        if not len(rows):
            return rows, {'A1': np.array([]), 'B2': np.array([])}, 'original_no_eligible_hours'
        saved = self.saved.get(day)
        if saved is not None:
            return saved[0], saved[1], 'verified_cp15_cache'
        item = self.new.get(f'{self.fold}:{day}')
        if item is None:
            raise ValueError(f'H0 cache miss {self.fold} {day}: no refit permitted inside replay')
        if item['content_sha256'] != self.digest(item) or item['protocol_sha256'] != self.protocol \
                or item['input_fingerprint'] != self.fingerprint or item['day'] != str(day) \
                or item['origin_utc'] != str(origin_utc(day).tz_convert('UTC')):
            raise ValueError(f'CP-16 component identity mismatch {self.fold} {day}')
        if item['timestamp_utc'] != list(map(str, self.data.index[rows])) or item['scale_sha256'] != array_hash(self.data.scale[rows]):
            raise ValueError(f'CP-16 component row/scale identity mismatch {self.fold} {day}')
        centers = {p: np.asarray(item['central'][p], float) for p in ('A1', 'B2')}
        if any(v.shape != (len(rows),) or not np.isfinite(v).all() for v in centers.values()):
            raise ValueError('invalid cached CP-16 central vector')
        return rows, centers, 'verified_cp16_cache'
