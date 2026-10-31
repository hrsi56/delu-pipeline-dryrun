"""Identity-bound CP-24 inputs (capstone v21-r11 §23.8, §23.10, §23.13 items 1-2).

* **The issued v21-r11 documents** are verified in the working tree: the anchor, its amendment
  record, the packaged brief, the pinned publication rules and packet template (§23.12, which the
  brief also records), the incorporated publication standard and presentation plan, and CP-23's
  test-only PyTorch reference lock (§23.7).
* **The inherited CP-15, CP-16 and CP-20 identities** are CP-21's `cp20_identities()`, unchanged
  (the root `uv.lock` that CP-15's protocol binds is byte-identical).
* **CP-20's, CP-21's and CP-22's committed evidence** is verified against each artifact manifest and
  against the blob at each evidence tag.
* **CP-23's identities.** `cp23.inputs.identities()` pins the v21-r10 anchor *bytes in the tree* and
  raises on v21-r11. Its result is reproduced here from the objects preserved at `evidence/cp-23`
  (as `cp22.inputs.cp21_identities` does for CP-21): every document it pinned is checked as a blob
  at the tag, and every CP-23 artifact CP-24 reads -- CP-23's D among them -- against CP-23's
  artifact manifest and its blob at the tag. Never relaxed.
* **Permitted partitions** are filtered before materialisation, and nothing dated after 2026-04-07 is
  read (`cp21.inputs.load` and its boundary guard, reused unchanged).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from cp15.data import sha
from cp21 import inputs as I21
from cp21.inputs import (BOUNDARY, FLOOR, KEYS, ORIGINS, BoundaryViolation, guard_boundary, hg_identity, load,  # noqa: F401
                         origin_manifest, weather_design, weather_matrix)
from cp22.inputs import CP21_EVIDENCE, CP21_REQUIRED, cp21_fit_identity, git_blob_sha256  # noqa: F401
from cp23 import inputs as I23

#: The issued CP-24 documents and the identities the brief and §23.7/§23.12 pin.
ISSUED = {
    'capstone_v21.md': '11068e3f57bd9277be50db62d109f3e7d8ea7b8fdfa042886ccc4ae5ab1ed2d2',
    'docs/track-b/capstone_v21-r10-to-v21-r11-amendments.md': '474017e1c8e8584e9c2956f40a872aa8917a6e110cbc58dc7a4110c9c802d782',
    'docs/track-b/evidence/cp-24/issued-brief.md': 'a3f11470dde0a36e9bdf84fbf02de0597ae3d0eca4c4c75f18e79ab173c34bd0',
    'docs/PUBLISH_RULES.md': '5a660864f8b82943174741320c71087a3d0508f707edeb997cb2c1395c1c73b4',
    'docs/track-b/publication-packet-template.md': '4efb01855ae2d9a6f54b5624607a93046b1fffa881e34af0c6798c513c7829cf',
    'docs/track-b/publication-standard-v1.md': '01d721c2ba6316ca9a2f707e79da229bef732e91bad22f6ce457fb01e69478cc',
    'docs/track-b/presentation-and-tracking-plan-2026-09-24.md': '281193740a256a99a1676d53ef61c9babb5f744a7d0fb038927e305d1149812c',
    'tests/cp23/torch-reference/uv.lock': 'b3164a375e871396e685d0e887972b092fcd0c24a7fe0047167e8fc41c25dfed',
}
CP23_EVIDENCE = '928bc1389627499fba14222e552feb62492f6e2d'   # evidence/cp-23
CP23_LAND = '03c5b6433d2045276780cae6ab0a0aaa6e7aa370'       # land/cp-23
CP22_EVIDENCE = I23.CP22_EVIDENCE
CP20_EVIDENCE = I21.CP20_EVIDENCE
CP22_REQUIRED = I23.CP22_REQUIRED
CP20_REQUIRED = I21.CP20_REQUIRED
#: The committed CP-23 evidence CP-24 reads: D's vectors, the members table (A1_w, B2_w), the record
#: CP-24's 4.6L' extends, the reference record, and the diagnostics CP-24 reports beside its own.
CP23_REQUIRED = ('reports/distribution-challenger/predictions.parquet', 'reports/distribution-challenger/members.parquet',
                 'reports/distribution-challenger/lineage.json', 'reports/distribution-challenger/protocol.json',
                 'reports/distribution-challenger/metrics.csv', 'reports/distribution-challenger/criteria.csv',
                 'reports/distribution-challenger/uncertainty.csv', 'reports/distribution-challenger/adoption.json',
                 'reports/distribution-challenger/licence-admission.md', 'reports/distribution-challenger/reference-checks.json',
                 'reports/distribution-challenger/resource-admission.json', 'reports/distribution-challenger/fit-cost.json',
                 'reports/distribution-challenger/daily-cycle.json',
                 'reports/distribution-challenger/diagnostics/extrapolation.csv',
                 'reports/distribution-challenger/diagnostics/calibration-by-level.csv',
                 'reports/distribution-challenger/diagnostics/pit-histogram.csv',
                 'reports/distribution-challenger/diagnostics/seed-stability.csv',
                 'tests/cp23/torch_reference_checks.py', 'tests/cp23/torch-reference/pyproject.toml',
                 'tests/cp23/torch-reference/uv.lock', 'src/cp23/ddnn.py', 'src/cp23/audit.py', 'src/cp23/scoring.py',
                 'scripts/mlflow_export.py', 'docs/track-b/evidence/cp-23/publication-packet.md',
                 'docs/track-b/research-content/cp23-claims.md')


def _manifest(root: Path, path: str) -> dict:
    return json.loads((root / path).read_text())['artifact_sha256']


def cp23_identities(root: Path) -> dict:
    """`cp23.inputs.identities()`'s documents, reproduced from the blobs preserved at evidence/cp-23."""
    root = Path(root)
    result = {}
    for name, expected in I23.ISSUED.items():
        if git_blob_sha256(root, CP23_EVIDENCE, name) != expected:
            raise ValueError(f'CP-23 issued identity mismatch at evidence/cp-23: {name}')
        result[f'{CP23_EVIDENCE}:{name}'] = expected
    return result


def identities(root: Path) -> dict:
    """Every identity CP-24 relies on, each recomputed; any mismatch raises."""
    root = Path(root)
    result = {}
    for name, expected in ISSUED.items():
        if sha(root / name) != expected:
            raise ValueError(f'issued CP-24 identity mismatch: {name}')
        result[name] = expected
    result.update(I21.cp20_identities(root))
    result.update(cp23_identities(root))
    cp20 = _manifest(root, 'reports/weather-ablation/artifact-manifest.json')
    if git_blob_sha256(root, CP20_EVIDENCE, 'reports/weather-ablation/artifact-manifest.json') \
            != sha(root / 'reports/weather-ablation/artifact-manifest.json'):
        raise ValueError('CP-20 artifact manifest differs from evidence/cp-20')
    for name in CP20_REQUIRED:
        digest = sha(root / name)
        if digest != cp20[name] or git_blob_sha256(root, CP20_EVIDENCE, name) != digest:
            raise ValueError(f'accepted CP-20 artifact identity mismatch: {name}')
        result[name] = digest
    for label, manifest, tag, required in (
            ('CP-21', 'reports/block-challenger/artifact-manifest.json', CP21_EVIDENCE, CP21_REQUIRED),
            ('CP-22', 'reports/v4-revision/artifact-manifest.json', CP22_EVIDENCE, CP22_REQUIRED),
            ('CP-23', 'reports/distribution-challenger/artifact-manifest.json', CP23_EVIDENCE, CP23_REQUIRED)):
        if git_blob_sha256(root, tag, manifest) != sha(root / manifest):
            raise ValueError(f'{label} artifact manifest differs from its evidence tag')
        pinned = _manifest(root, manifest)
        for name in required:
            digest = sha(root / name)
            if digest != pinned[name] or git_blob_sha256(root, tag, name) != digest:
                raise ValueError(f'accepted {label} artifact identity mismatch: {name}')
            result[name] = digest
    return result


def input_fingerprint(root: Path) -> str:
    return hashlib.sha256(json.dumps(identities(root), sort_keys=True).encode()).hexdigest()


__all__ = ['BOUNDARY', 'BoundaryViolation', 'CP21_REQUIRED', 'CP22_REQUIRED', 'CP23_REQUIRED', 'ISSUED', 'KEYS', 'ORIGINS',
           'cp21_fit_identity', 'cp23_identities', 'git_blob_sha256', 'guard_boundary', 'hg_identity', 'identities',
           'input_fingerprint', 'load', 'origin_manifest', 'weather_design', 'weather_matrix']
