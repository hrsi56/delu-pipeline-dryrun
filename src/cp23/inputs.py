"""Identity-bound CP-23 inputs (capstone v21-r10 §21.5, §21.7, §21.10 items 1 and 6).

* **The issued v21-r10 documents** are verified in the working tree: the anchor, its amendment
  record, the packaged brief, and the publication rules and standards the packet pins. PUBLISH_RULES
  1.3's hash is the Owner's ruling of 2026-10-04: the brief omitted it, and 1.3 has had a single
  identity since `940eb98`.
* **The inherited CP-15, CP-16 and CP-20 identities** are CP-21's `cp20_identities()`, unchanged. It
  runs as is because the root `uv.lock`, which CP-15's protocol binds, is byte-identical: the PyTorch
  reference is pinned in its own test-only lock.
* **CP-20's, CP-21's and CP-22's committed evidence** is verified against each artifact manifest and
  against the blob at each evidence tag, so every vector CP-23 reuses is the committed one (§21.7:
  "reused HGL, HG, A1 and B2 vectors match their committed blobs").
* **Permitted partitions** are filtered before materialisation, and nothing dated after 2026-04-07 is
  read (`cp21.inputs.load` and its boundary guard, reused unchanged).
"""
from __future__ import annotations

import json
from pathlib import Path

from cp15.data import sha
from cp21 import inputs as I21
from cp21.inputs import (BOUNDARY, FLOOR, KEYS, ORIGINS, BoundaryViolation, guard_boundary, hg_identity, load,  # noqa: F401
                         origin_manifest, weather_design, weather_matrix)
from cp22.inputs import CP21_EVIDENCE, CP21_REQUIRED, cp21_fit_identity, git_blob_sha256  # noqa: F401

#: The issued CP-23 documents and the publication documents the packet pins.
ISSUED = {
    'capstone_v21.md': '6873c2501067c63692ec7cb79dfd4073164a8a014edbd6ebc23169e7e8ff6709',
    'docs/track-b/capstone_v21-r9-to-v21-r10-amendments.md': '453a66a107e6b597cecc80c02e04995b4de4a0935960638df220c5722afa5199',
    'docs/track-b/evidence/cp-23/issued-brief.md': '33f6b41202586b2fc4f1d87624c5978e1fd5bde1185ac05271429bf719502d0c',
    'docs/PUBLISH_RULES.md': '5a660864f8b82943174741320c71087a3d0508f707edeb997cb2c1395c1c73b4',
    'docs/track-b/publication-standard-v1.md': '01d721c2ba6316ca9a2f707e79da229bef732e91bad22f6ce457fb01e69478cc',
    'docs/track-b/presentation-and-tracking-plan-2026-09-24.md': '281193740a256a99a1676d53ef61c9babb5f744a7d0fb038927e305d1149812c',
    'docs/track-b/publication-packet-template.md': '4efb01855ae2d9a6f54b5624607a93046b1fffa881e34af0c6798c513c7829cf',
}
CP22_EVIDENCE = '8daf7d0775404611e9f7ef4844fc602f1aa67b26'   # evidence/cp-22
CP22_LAND = 'ebb7d421957d2a55ae13039955135ff299145390'       # land/cp-22
CP20_EVIDENCE = I21.CP20_EVIDENCE
#: The committed CP-22 evidence CP-23 reads: the origin enumeration and the diagnostics it sits beside.
CP22_REQUIRED = ('reports/v4-revision/preflight/e1.json', 'reports/v4-revision/investigation/extrapolation.csv',
                 'reports/v4-revision/investigation/capacity-stability.csv', 'reports/v4-revision/fit-cost.json',
                 'reports/v4-revision/daily-cycle.json', 'reports/v4-revision/decisions.json')
CP20_REQUIRED = I21.CP20_REQUIRED + ('reports/weather-ablation/artifact-manifest.json',)


def _manifest(root: Path, path: str) -> dict:
    return json.loads((root / path).read_text())['artifact_sha256']


def identities(root: Path) -> dict:
    """Every identity CP-23 relies on, each recomputed; any mismatch raises."""
    root = Path(root)
    result = {}
    for name, expected in ISSUED.items():
        if sha(root / name) != expected:
            raise ValueError(f'issued CP-23 identity mismatch: {name}')
        result[name] = expected
    result.update(I21.cp20_identities(root))
    cp20 = _manifest(root, 'reports/weather-ablation/artifact-manifest.json')
    for name in I21.CP20_REQUIRED:
        digest = sha(root / name)
        if digest != cp20[name] or git_blob_sha256(root, CP20_EVIDENCE, name) != digest:
            raise ValueError(f'accepted CP-20 artifact identity mismatch: {name}')
        result[name] = digest
    for label, manifest, tag, required in (
            ('CP-21', 'reports/block-challenger/artifact-manifest.json', CP21_EVIDENCE, CP21_REQUIRED),
            ('CP-22', 'reports/v4-revision/artifact-manifest.json', CP22_EVIDENCE, CP22_REQUIRED)):
        if git_blob_sha256(root, tag, manifest) != sha(root / manifest):
            raise ValueError(f'{label} artifact manifest differs from its evidence tag')
        pinned = _manifest(root, manifest)
        for name in required:
            digest = sha(root / name)
            if digest != pinned[name] or git_blob_sha256(root, tag, name) != digest:
                raise ValueError(f'accepted {label} artifact identity mismatch: {name}')
            result[name] = digest
    return result


__all__ = ['BOUNDARY', 'BoundaryViolation', 'CP21_REQUIRED', 'CP22_REQUIRED', 'ISSUED', 'KEYS', 'ORIGINS',
           'cp21_fit_identity', 'git_blob_sha256', 'guard_boundary', 'hg_identity', 'identities', 'load',
           'origin_manifest', 'weather_design', 'weather_matrix']
