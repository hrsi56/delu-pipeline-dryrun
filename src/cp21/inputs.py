"""Identity-bound CP-21 inputs (capstone v21-r6 §17.4, §17.10 items 1-2).

* The issued v21-r6 documents are verified in the working tree; historical anchors that
  inherited recipes pin (v21-r1 for CP-15, v21-r4 for CP-20) are verified as Git objects at the
  evidence tags that preserve them. CP-16's and CP-20's own identity functions pin the older
  anchor *bytes in the tree* and cannot run against v21-r6, so they are reproduced here from
  those preserved objects, never relaxed.
* CP-20's HG component-cache identity is recomputed exactly as CP-20 computed it, so every cached
  A1_w/B2_w entry is checked against its CP-20 fingerprint before reuse.
* Permitted partitions are filtered before materialisation; nothing dated after 2026-04-07 is
  read (the boundary guard refuses it, with a positive control in the tests).
"""
from __future__ import annotations

from datetime import date
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd

from cp15.data import RAW_COLUMNS, prepare, sha
from cp20.components import fingerprint
from cp20.weather import COLUMNS as WEATHER_COLUMNS, WeatherDesign
from delu_forecast.folds import load_partition_spec

BOUNDARY = date(2026, 4, 7)
FLOOR = date(2019, 1, 1)
ORIGINS = 638
FOLD_COUNTS = (2160, 2159, 2112, 2160, 2156)
KEYS = 10747

#: The issued CP-21 documents (brief §Target, §Publication packet) and the anchors it pins.
ISSUED = {
    'capstone_v21.md': 'ee402c4703d176f8181ed82966c978ad84c3c73a0cdb1d3567871f58bc867344',
    'docs/track-b/capstone_v21-r5-to-v21-r6-amendments.md': 'a9086fa1d70ea9cfd9c7fde3733f631bff7b8e372a8990e6cc7a52be3650bedd',
    'docs/track-b/evidence/cp-21/issued-brief.md': '813fb8a476a7b6d10ecf0a5519e97ec8efc4ff36e0f13868676ad34534fc020f',
    'docs/PUBLISH_RULES.md': '91eea445434718a163f03bcfd82e1db275a9d98311e76eb6cd1365f3707584f3',
    'docs/track-b/publication-standard-v1.md': '01d721c2ba6316ca9a2f707e79da229bef732e91bad22f6ce457fb01e69478cc',
    'docs/track-b/presentation-and-tracking-plan-2026-09-24.md': '281193740a256a99a1676d53ef61c9babb5f744a7d0fb038927e305d1149812c',
    'docs/track-b/publication-packet-template.md': '4efb01855ae2d9a6f54b5624607a93046b1fffa881e34af0c6798c513c7829cf',
}
#: CP-15's protocol pins the rulebook/anchor at the CP-15 evidence commit.
CP15_EVIDENCE = '1bdc75b8ab943092bb8de6ba893defb9e12250d8'
#: CP-20's issued documents, preserved at evidence/cp-20 (its own identity function pinned them).
CP20_EVIDENCE = 'a7a9b2e3a4d0147a0c82835a853277e9c81c7945'
CP20_ISSUED = {
    'capstone_v21.md': '150bd53fa15067b1d138c95d0912f86a1e92ce74092059be09034758c1926167',
    'docs/track-b/capstone_v21-r3-to-v21-r4-amendments.md': '3e0bdf6a343d5336f406f0eeb726b1b65c0f767ec8d9418a7d8118902e1ace55',
    'docs/track-b/cp-20-direct-weather-brief.md': '28a4e195fa7f66ff3270d5d54e477478e90aa42124bceb1ee529f95f5761883e',
}
CP16_REQUIRED = ('reports/v2-causal/predictions.parquet', 'reports/v2-causal/lineage.json',
                 'reports/v2-causal/input-manifest.json', 'reports/v2-causal/protocol.json',
                 'src/cp16/residuals.py', 'src/cp16/inputs.py', 'src/cp16/scoring.py')
#: The accepted CP-20 artifacts CP-21 reads (its manifest pins them).
CP20_REQUIRED = ('reports/weather-ablation/protocol.json', 'reports/weather-ablation/predictions.parquet',
                 'reports/weather-ablation/lineage.json', 'reports/weather-ablation/weather-features.parquet',
                 'reports/weather-ablation/run-manifest.json', 'reports/weather-ablation/extraction-summary.json',
                 'reports/weather-ablation/metrics.csv', 'reports/weather-ablation/uncertainty.csv',
                 'reports/weather-ablation/criteria.csv', 'reports/weather-ablation/missingness.csv')
WEATHER_DESIGN_SHA256 = 'f5a9c6ed7dae18e711017e9f2887e3c72f64c286137bd7cabfb3469cd10274c0'
CP20_HG_FINGERPRINT = '132369044ef45e3af3205de9ee3831a40e579070e7f16a96061d1708dc37a591'


class BoundaryViolation(ValueError):
    """An input or outcome dated after 2026-04-07 (or before 2019-01-01) was materialised."""


def _git_blob_sha256(root: Path, rev: str, name: str) -> str:
    blob = subprocess.check_output(['git', 'show', f'{rev}:{name}'], cwd=root)
    return hashlib.sha256(blob).hexdigest()


def guard_boundary(frame: pd.DataFrame) -> pd.DataFrame:
    """Refuse any row dated after 2026-04-07 or before 2019-01-01 (§17.1, §17.7 boundary guard)."""
    days = pd.Index(frame['delivery_date'])
    if len(days) and (max(days) > BOUNDARY or min(days) < FLOOR):
        raise BoundaryViolation(f'materialised delivery dates {min(days)}..{max(days)} outside {FLOOR}..{BOUNDARY}')
    return frame


def cp20_identities(root: Path) -> dict:
    """CP-20's `identities()` result, reproduced from preserved objects (its anchor is no longer
    the tree's). Every value is re-verified; nothing is copied without a hash check."""
    root = Path(root)
    result = {}
    for name, expected in CP20_ISSUED.items():
        if _git_blob_sha256(root, CP20_EVIDENCE, name) != expected:
            raise ValueError(f'CP-20 issued identity mismatch at evidence/cp-20: {name}')
        result[name] = expected
    p = json.loads((root / 'reports/cp15/protocol.json').read_text())
    for name, expected in p['input_sha256'].items():
        if name in ('AGENTS.md', 'capstone_v21.md'):
            if _git_blob_sha256(root, CP15_EVIDENCE, name) != expected:
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
    cp16 = json.loads((root / 'reports/v2-causal/artifact-manifest.json').read_text())['artifact_sha256']
    for name in CP16_REQUIRED:
        if sha(root / name) != cp16[name]:
            raise ValueError(f'accepted CP-16 artifact identity mismatch: {name}')
        result[name] = cp16[name]
    return result


def identities(root: Path) -> dict:
    """Every identity CP-21 relies on: issued v21-r6 documents, inherited CP-15/16/20 artifacts."""
    root = Path(root)
    result = {}
    for name, expected in ISSUED.items():
        if sha(root / name) != expected:
            raise ValueError(f'issued CP-21 identity mismatch: {name}')
        result[name] = expected
    result.update(cp20_identities(root))
    cp20 = json.loads((root / 'reports/weather-ablation/artifact-manifest.json').read_text())['artifact_sha256']
    for name in CP20_REQUIRED:
        if sha(root / name) != cp20[name]:
            raise ValueError(f'accepted CP-20 artifact identity mismatch: {name}')
        result[name] = cp20[name]
    return result


def hg_identity(root: Path) -> dict:
    """CP-20's HG component identity, recomputed exactly as `cp20.execution.hg_identity` did."""
    root = Path(root)
    protocol_sha = sha(root / 'reports/weather-ablation/protocol.json')
    ident = {'input_fingerprint': fingerprint(cp20_identities(root), WEATHER_DESIGN_SHA256, protocol_sha),
             'weather_design_sha256': WEATHER_DESIGN_SHA256, 'protocol_sha256': protocol_sha}
    lineage = json.loads((root / 'reports/weather-ablation/lineage.json').read_text())
    if ident != lineage['hg_identity'] or ident['input_fingerprint'] != CP20_HG_FINGERPRINT:
        raise ValueError('recomputed CP-20 HG identity differs from the accepted lineage')
    return ident


def load(root: Path, before: date | None = None):
    """Permitted partitions only, filtered before materialisation, never past 2026-04-07."""
    root = Path(root)
    p = json.loads((root / 'reports/cp15/protocol.json').read_text())
    spec = load_partition_spec(root / 'data/partitions.json')
    if spec.eda_cutoff != BOUNDARY:
        raise ValueError('EDA cutoff is not the 2026-04-07 development boundary')
    cutoff = BOUNDARY if before is None else min(BOUNDARY, before - pd.Timedelta(days=1).to_pytimedelta())
    filters = [('delivery_date', '>=', FLOOR), ('delivery_date', '<=', cutoff)]
    frame = pd.read_parquet(root / 'data/snapshot.parquet', columns=RAW_COLUMNS, filters=filters)
    guard_boundary(frame)
    return prepare(frame, p, spec)


def origin_manifest(root: Path) -> dict:
    m = json.loads((Path(root) / 'reports/v2-causal/input-manifest.json').read_text())
    if sum(f['date_count'] for f in m['folds']) != ORIGINS:
        raise ValueError('frozen CP-16/CP-20 origin manifest changed')
    if tuple(len(f['original_target_keys']) for f in m['folds']) != FOLD_COUNTS:
        raise ValueError('original target key counts changed')
    return m


def weather_design(root: Path) -> WeatherDesign:
    features = pd.read_parquet(Path(root) / 'reports/weather-ablation/weather-features.parquet')
    design = WeatherDesign.from_features(features)
    if design.sha256 != WEATHER_DESIGN_SHA256:
        raise ValueError('weather design differs from CP-20\'s frozen design')
    return design


def weather_matrix(design: WeatherDesign, data) -> tuple[np.ndarray, np.ndarray]:
    """The three same-target-local-hour weather columns for every modelling row (NaN = missing,
    imputed later on each fit's own training partition), and whether the row has a frozen weather
    record at all. The extraction covers every CP-20 training window, not every calendar date, so
    each fit checks that all its training and forecast rows have a record (as CP-20's HG did)."""
    present = design.present(data.dates, data.hours)
    return design.matrix(data.dates, data.hours, required=np.zeros(len(data.dates), bool)), present


__all__ = ['BOUNDARY', 'BoundaryViolation', 'ISSUED', 'KEYS', 'ORIGINS', 'WEATHER_COLUMNS', 'cp20_identities',
           'guard_boundary', 'hg_identity', 'identities', 'load', 'origin_manifest', 'weather_design', 'weather_matrix']
