"""Identity-bound CP-22 inputs (capstone v21-r9 §20.4, §20.7, §20.10 items 1-2).

* The issued v21-r9 documents are verified in the working tree.
* CP-21's own `identities()` pins the v21-r6 anchor *bytes in the tree* and cannot run against
  v21-r9; its result is reproduced here from the Git objects preserved at `evidence/cp-21`, never
  relaxed. Its SHA-256 must equal the `cp21_input_fingerprint` that CP-21's committed lineage
  binds to every fit-cache entry, so CP-21's retained fits are reusable only with that identity.
* CP-20's and CP-21's committed evidence files are verified against their own artifact manifests
  *and* against the blobs at their evidence tags, so the vectors CP-22 reuses are the committed
  ones (§20.7 saved-vector identity).
* Permitted partitions are filtered before materialisation; nothing dated after 2026-04-07 is read
  (`cp21.inputs.load` and its boundary guard, reused unchanged).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess

from cp15.data import sha
from cp21 import inputs as I21
from cp21.inputs import (BOUNDARY, FLOOR, KEYS, ORIGINS, BoundaryViolation, guard_boundary, hg_identity, load,  # noqa: F401
                         origin_manifest, weather_design, weather_matrix)

#: The issued CP-22 documents (brief §Target) and the publication rules they pin.
ISSUED = {
    'capstone_v21.md': '5fc9c6862aa9f623af29db295e79456ecd94c60e213f8f286cdca97153e09175',
    'docs/track-b/capstone_v21-r8-to-v21-r9-amendments.md': '912eb98ffceb1b7b31e0e14da6cffc8725c3af7b227cbbcef4cea62841f5d52a',
    'docs/PUBLISH_RULES.md': '5a660864f8b82943174741320c71087a3d0508f707edeb997cb2c1395c1c73b4',
    'docs/track-b/evidence/cp-22/issued-brief.md': '563f64f3602848808ff11a7d74da8218b16d51846f34e90ad754c36ec2461fb3',
    'docs/track-b/publication-standard-v1.md': '01d721c2ba6316ca9a2f707e79da229bef732e91bad22f6ce457fb01e69478cc',
    'docs/track-b/presentation-and-tracking-plan-2026-09-24.md': '281193740a256a99a1676d53ef61c9babb5f744a7d0fb038927e305d1149812c',
    'docs/track-b/publication-packet-template.md': '4efb01855ae2d9a6f54b5624607a93046b1fffa881e34af0c6798c513c7829cf',
}
CP21_EVIDENCE = '1d13f99b1ab12f64719b439d0a9d4703bc7e1bdb'   # evidence/cp-21
CP21_LAND = '4e37cf7926c87744188b679cef5a7d29e9ef6499'       # land/cp-21
CP20_EVIDENCE = I21.CP20_EVIDENCE
#: CP-21's binding fit identity (its committed lineage records it; every retained fit carries it).
CP21_INPUT_FINGERPRINT = '83765c0a713a17f621d263c3c8d89c08df7824a23dd814a3cfee8ce622d2d5f8'
CP21_PROTOCOL_SHA256 = 'a51cfbbeb7bc7861c78e1fe0916807d7ff5e0f45be495285959ff4ac52030db1'
#: The committed CP-21 evidence CP-22 reads (its artifact manifest pins each).
CP21_REQUIRED = ('reports/block-challenger/predictions.parquet', 'reports/block-challenger/lineage.json',
                 'reports/block-challenger/protocol.json', 'reports/block-challenger/fits.parquet',
                 'reports/block-challenger/metrics.csv', 'reports/block-challenger/criteria.csv',
                 'reports/block-challenger/uncertainty.csv', 'reports/block-challenger/adoption.json',
                 'reports/block-challenger/fit-cost-by-origin.csv', 'reports/block-challenger/fit-cost.json',
                 'reports/block-challenger/daily-cycle.json', 'reports/block-challenger/draft-registry.json',
                 'reports/block-challenger/replicates.parquet', 'reports/block-challenger/diagnostics.csv',
                 'docs/track-b/evidence/cp-21/publication-packet.md', 'docs/track-b/research-content/cp21-claims.md')
CP20_REQUIRED = I21.CP20_REQUIRED + ('reports/weather-ablation/artifact-manifest.json',)


def git_blob_sha256(root: Path, rev: str, name: str) -> str:
    blob = subprocess.check_output(['git', 'show', f'{rev}:{name}'], cwd=root)
    return hashlib.sha256(blob).hexdigest()


def cp21_identities(root: Path) -> dict:
    """CP-21's `identities()` result, reproduced from the objects preserved at evidence/cp-21."""
    root = Path(root)
    result = {}
    for name, expected in I21.ISSUED.items():
        if git_blob_sha256(root, CP21_EVIDENCE, name) != expected:
            raise ValueError(f'CP-21 issued identity mismatch at evidence/cp-21: {name}')
        result[name] = expected
    result.update(I21.cp20_identities(root))
    cp20 = json.loads((root / 'reports/weather-ablation/artifact-manifest.json').read_text())['artifact_sha256']
    for name in I21.CP20_REQUIRED:
        if sha(root / name) != cp20[name]:
            raise ValueError(f'accepted CP-20 artifact identity mismatch: {name}')
        result[name] = cp20[name]
    return result


def cp21_fit_identity(root: Path) -> dict:
    """The identity CP-21 bound to each retained fit-cache entry, recomputed and checked."""
    root = Path(root)
    ident = {'cp21_input_fingerprint': hashlib.sha256(json.dumps(cp21_identities(root), sort_keys=True).encode()).hexdigest(),
             'cp21_protocol_sha256': sha(root / 'reports/block-challenger/protocol.json'),
             'weather_design_sha256': weather_design(root).sha256}
    lineage = json.loads((root / 'reports/block-challenger/lineage.json').read_text())
    if ident != lineage['fit_identity'] or ident['cp21_input_fingerprint'] != CP21_INPUT_FINGERPRINT \
            or ident['cp21_protocol_sha256'] != CP21_PROTOCOL_SHA256:
        raise ValueError('recomputed CP-21 fit identity differs from its committed lineage')
    return ident


def identities(root: Path) -> dict:
    """Every identity CP-22 relies on: the issued v21-r9 documents, the inherited CP-15/16/20
    artifacts, and CP-20's and CP-21's committed evidence (manifest and evidence-tag blob)."""
    root = Path(root)
    result = {}
    for name, expected in ISSUED.items():
        if sha(root / name) != expected:
            raise ValueError(f'issued CP-22 identity mismatch: {name}')
        result[name] = expected
    result.update(I21.cp20_identities(root))
    cp20 = json.loads((root / 'reports/weather-ablation/artifact-manifest.json').read_text())['artifact_sha256']
    for name in I21.CP20_REQUIRED:
        digest = sha(root / name)
        if digest != cp20[name] or git_blob_sha256(root, CP20_EVIDENCE, name) != digest:
            raise ValueError(f'accepted CP-20 artifact identity mismatch: {name}')
        result[name] = digest
    cp21 = json.loads((root / 'reports/block-challenger/artifact-manifest.json').read_text())['artifact_sha256']
    if git_blob_sha256(root, CP21_EVIDENCE, 'reports/block-challenger/artifact-manifest.json') \
            != sha(root / 'reports/block-challenger/artifact-manifest.json'):
        raise ValueError('CP-21 artifact manifest differs from evidence/cp-21')
    for name in CP21_REQUIRED:
        digest = sha(root / name)
        if digest != cp21[name] or git_blob_sha256(root, CP21_EVIDENCE, name) != digest:
            raise ValueError(f'accepted CP-21 artifact identity mismatch: {name}')
        result[name] = digest
    return result


__all__ = ['BOUNDARY', 'BoundaryViolation', 'CP21_REQUIRED', 'ISSUED', 'KEYS', 'ORIGINS', 'cp21_fit_identity',
           'cp21_identities', 'git_blob_sha256', 'guard_boundary', 'hg_identity', 'identities', 'load', 'origin_manifest',
           'weather_design', 'weather_matrix']
