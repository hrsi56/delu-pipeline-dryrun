"""CP-17b's re-freeze: the policy identity and freeze manifest that supersede CP-17's, and their verifier.

**What is pinned, and how** (the Owner's decision 19; brief item 4):

- **Governance documents** are bound by frozen copies under `docs/track-b/evidence/cp-17b/pinned/`, each with its
  SHA-256: the anchor at v21-r17, CP-17's two briefs and CP-17b's, both decisions records, and the CP-25 rulings
  and the CP-26 GFS ruling. `verify` checks the copies, never the live files, so an anchor amendment (r18) or a
  new decisions record leaves it green.
- **Computational inputs and code** are verified live: the snapshot, CP-25's and CP-26's manifests, CP-28's
  selection, the tier register, the partitions, the margin tables, the weather features and their gaps,
  CP-17-probe's runtime retrieval manifest, CP-15's protocol, `uv.lock`, `pyproject.toml`; every package file,
  the runners and the issuance driver (`cp17.daily`); the package's deterministic zip, rebuilt; the payload
  schemas, against the package's own.
- **Local runtimes and state** are verified live: the 23 Pyodide files, the Node.js binary, and A1's search state
  (`k13.json` in the `evidence/cp-26` extract), its file hash and content digest. A file present is always checked.
  On the issuing machine (one holding the project's pinned Pyodide or the extract) every one is required; elsewhere
  (CI) an absent one is reported as unchecked, never passed silently.
- **CP-17's freeze is superseded, not edited:** its evidence files are checked unchanged, by the hashes the brief
  cites. CP-17's own verifier (`cp17.tools.freeze`) stays as landed. It pins governance documents live, so it is run
  on `cp17_view`, whose governance documents are CP-17b's frozen copies; there it reports exactly the package files
  CP-17b changed (`tests/cp17/test_records.py`), and an amendment leaves it green.

`COVERAGE` states how each hash the manifest records is checked.

    PYTHONPATH=src python -m cp17.tools.refreeze pin
    PYTHONPATH=src python -m cp17.tools.refreeze identity --bundle-manifest <src.json> --protocol-from <v2 payload json> \\
        --browser <browser-results.json> --out <policy-identity.json>
    PYTHONPATH=src python -m cp17.tools.refreeze manifest --identity ... --issuance ... --bundle ... --states ... --out ...
    PYTHONPATH=src python -m cp17.tools.refreeze verify docs/track-b/evidence/cp-17b/freeze-manifest.json

`verify` dispatches on the manifest's schema: CP-17c's manifest, which supersedes CP-17b's, is verified by
`cp17.tools.refreeze17c`, with the same output.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import tempfile
import zlib

from .. import issue, retrain, schema, scoring, shadow
from . import freeze, ledger

EVIDENCE = 'docs/track-b/evidence/cp-17b'
PINNED = f'{EVIDENCE}/pinned'
SCHEMAS_DIR = f'{EVIDENCE}/schemas'
ARTIFACT_PATH = f'{EVIDENCE}/artifact-2026-09-06.json'
#: name: (the source at the freeze, its frozen copy's file name, its SHA-256 as the brief and CP-17 cite it)
GOVERNANCE = {
    'anchor_v21_r17': ('capstone_v21.md', 'anchor-capstone-v21-r17.md',
                       '3124451da84c3cd99fae728c371207aace6ae537474f7769d0ff6333ee53e4fc'),
    'cp17_brief_9e522b9a': ('docs/track-b/evidence/cp-17/issued-brief-v1-9e522b9a.md', 'cp-17-brief-9e522b9a.md',
                            '9e522b9a42ea3c02574bd2cde4fad97266bdfd8dc38a104c13566e05c80164fa'),
    'cp17_brief_9f235cf8': ('docs/track-b/evidence/cp-17/issued-brief.md', 'cp-17-brief-9f235cf8.md',
                            '9f235cf84876db4975930f1fce8fb62dbe188b8be4130e692eda9b5b7e1b1a73'),
    'cp17b_brief_b9aeeee4': ('docs/track-b/evidence/cp-17b/issued-brief.md', 'cp-17b-brief-b9aeeee4.md',
                             'b9aeeee4131cadd060d35e05d8d0c04f114a8ec764b5c141dc893863fdd75034'),
    'decisions_1_12': ('docs/track-b/pre-cp-17-rulings-2026-10-25.md', 'pre-cp-17-rulings-2026-10-25.md',
                       'd29b0abc6a17dcb6bca2cbd464c1dcc5858c50e2b653a42e646d3dc66ea8f6b0'),
    'decisions_13_19': ('docs/track-b/cp-17-owner-decisions-2026-10-28.md', 'cp-17-owner-decisions-2026-10-28.md',
                        '4d049770e51a37a806c52a4ae681f70aae62ac1847a67c7c0c3109385da3ac37'),
    'cp25_rulings': ('docs/track-b/cp-25-rulings-2026-10-12.md', 'cp-25-rulings-2026-10-12.md',
                     'c7af67f88157277aafd8dd2b1b49e12a1d9ff4b747ca5bdf178fac664fba53c1'),
    'cp26_gfs_ruling': ('docs/track-b/cp-26-gfs-ruling-2026-10-19.md', 'cp-26-gfs-ruling-2026-10-19.md',
                        '3b97ff6abdf9a30ddd8c950cf3c459301d757aa1af08f6d8121bc73f92ba2373'),
}
COMPUTATIONAL = ('data/snapshot.parquet', 'reports/development-inputs/manifest.json',
                 'reports/rebaseline/cold-rerun-2026-10-15/scores-manifest.json', 'reports/cp28/results/selection.json',
                 'reports/development-inputs/tier-register.json', 'reports/development-inputs/partitions-development.json',
                 'reports/development-inputs/margin-days.csv', 'reports/development-inputs/margin-table.csv',
                 'reports/weather-ablation/weather-features.parquet',
                 'reports/development-inputs/weather-features-gaps.parquet', freeze.PROBE_RETRIEVAL,
                 'reports/cp15/protocol.json', 'uv.lock', 'pyproject.toml')
PROTOCOL = 'reports/cp15/protocol.json'
#: CP-17's evidence, superseded and unchanged (the brief's citations).
CP17 = {'freeze_manifest': ('docs/track-b/evidence/cp-17/freeze-manifest.json',
                            '4af62410c2d08442b85cd11cd271f0eb367bfb019196784d1e048a46a908a544'),
        'freeze_record': ('docs/track-b/evidence/cp-17/freeze-record.md',
                          '4a47623718441cca1eb7210dfa3b47dcfcc5c421b8e3a57b858282281b5e0cec'),
        'protocol': ('docs/track-b/evidence/cp-17/4-7t-frozen-protocol.md',
                     '337a102d6bbbf6d9335fe6ed701a4c80ea1b9172344ffc0f63909b155b567c58'),
        'integration': ('docs/track-b/evidence/cp-17/integration.md',
                        '7083bfb7176a9b7fbb4bfc4e03d00317f0c5601ec5360c006de1a2b4e066319d')}
CP17_POLICY_IDENTITY = 'b304d6479a8628a26014422c6682f1e6543bd0d9ff656ad34b2d87b163bc254e'
#: How `verify` checks every hash the manifest records (N4).
COVERAGE = {
    'policy_identity.governance': 'each frozen copy hashed, and the set of pins is the frozen set',
    'policy_identity.computational_sha256': 'each file hashed live',
    'policy_identity.code.files_sha256 / runner_files_sha256 / pipeline_files_sha256': 'each file hashed live',
    'policy_identity.code.package_sha256 / zip_sha256': 'the package rebuilt from src/ (cp17.tools.bundle) and compared; '
                                                        'the zip where the zlib runtime equals zip_zlib_runtime, else '
                                                        'reported unchecked',
    'policy_identity.schemas_sha256': "each schema file hashed, and equal to the package's own schema",
    'policy_identity.hyperparameters.cp15_protocol_sha256': 'recomputed from reports/cp15/protocol.json',
    'policy_identity.hyperparameters.A1_search': "k13.json hashed and its content digest read (local; required where "
                                                 "any local runtime is present)",
    'policy_identity.issuing_host': 'the 23 Pyodide files and the Node.js binary hashed (local; as above); the '
                                    'retrieval manifest hashed live',
    'policy_identity.supersedes.records': "CP-17's evidence files hashed live",
    'policy_identity_sha256 / content_sha256': "the manifest's own digests recomputed",
    'freeze_origin.artifact_file_sha256 / artifact_sha256': 'the committed artifact hashed, its identity recomputed, '
                                                            'and `cp17.issue.verify_artifact` and the schema applied',
    'freeze_origin.artifact_arrays_sha256': "local reproduction material (the issued fit's exported arrays); not "
                                            'recomputed by verify',
    'reference_bundle_manifest_sha256': 'local reproduction material; `cp17.schema.verify_bundle` checks the bundle',
    'initial_state': "CP-17's state, carried by reference (rebuilt from CP-26's ledger by cp17.tools.states and "
                     'compared by the shadow run, reports/cp17b/shadow-run-*.json); not recomputed by verify'}


def pin(root: Path) -> dict:
    """Copy each governance document, at its pinned identity, into `PINNED`; returns the record."""
    root = Path(root)
    (root / PINNED).mkdir(parents=True, exist_ok=True)
    out = {}
    for name, (source, copy_name, expected) in GOVERNANCE.items():
        copy = f'{PINNED}/{copy_name}'
        target = root / copy
        data = (target if target.exists() else root / source).read_bytes()     # a copy, once made, is never rewritten
        if _sha(data) != expected:
            raise ValueError(f'{copy if target.exists() else source}: {_sha(data)} is not the pinned {expected}')
        if not target.exists():
            target.write_bytes(data)
        out[name] = {'source_at_freeze': source, 'copy': copy, 'sha256': expected}
    return out


def _sha(data: bytes) -> str:
    import hashlib
    return hashlib.sha256(data).hexdigest()


def export_schemas(root: Path) -> dict:
    folder = Path(root) / SCHEMAS_DIR
    folder.mkdir(parents=True, exist_ok=True)
    out = {}
    for kind, value in schema.SCHEMAS.items():
        path = folder / f'{kind}.schema.json'
        path.write_text(json.dumps(value, indent=1, sort_keys=True, ensure_ascii=False) + '\n')
        out[f'{SCHEMAS_DIR}/{kind}.schema.json'] = freeze.sha256_file(path)
    return out


def reference_runtime(browser_results: Path | None) -> dict:
    """The reference runtime with its browser part (CP-17's review, N6), checked against a measured run."""
    out = {'pyodide': '0.28.0', 'node': 'v26.7.0', 'chromium': '153.0.8010.12', 'webkit': '26.6', 'playwright': '1.63.0',
           'machine_class': 'Apple M3, 8 cores, 16 GB, macOS 26.5.2'}
    if browser_results is not None:
        record = json.loads(Path(browser_results).read_text())
        seen = {r['browser']: r.get('browser_version') for r in record['runs'] if not r.get('errors')}
        expected = {k: out[k] for k in seen}
        if not seen or seen != expected or record['playwright'] != out['playwright']:
            raise ValueError(f'the measured browsers differ from the reference runtime: {seen}, {record["playwright"]}')
        out['measured'] = {'browsers': seen, 'platform': record['platform'], 'machine': record['machine'],
                           'cpu_count': record['cpu_count']}
    return out


def policy_identity(root: Path, bundle_manifest: dict, protocol: dict, governance: dict, schemas_sha256: dict,
                    runtime: dict) -> dict:
    base = freeze.policy_identity(root, bundle_manifest, protocol)
    if base['hyperparameters']['cp15_protocol_sha256'] != freeze.digest(json.loads((Path(root) / PROTOCOL).read_text())):
        raise ValueError("the payloads' CP-15 protocol is not reports/cp15/protocol.json")
    base.pop('pinned_files')
    base.pop('contracts_sha256')
    base['schema'] = 'cp17b-policy-identity-v1'
    base['governance'] = governance
    base['computational_sha256'] = {p: freeze.sha256_file(Path(root) / p) for p in COMPUTATIONAL}
    base['code']['pipeline_files_sha256'] = bundle_manifest['pipeline_files_sha256']
    base['code']['zip_zlib_runtime'] = zlib.ZLIB_RUNTIME_VERSION      # the zip's bytes depend on it
    base['schemas_sha256'] = schemas_sha256
    base['reference_runtime'] = runtime
    base['supersedes'] = {'checkpoint': 'CP-17', 'policy_identity_sha256': CP17_POLICY_IDENTITY,
                          'records': {k: {'path': p, 'sha256': h} for k, (p, h) in CP17.items()},
                          'note': "CP-17's freeze is superseded, not edited; its records stay as committed"}
    base['success_definition'] = {
        'wall_seconds': issue.WALL_LIMIT_SECONDS, 'peak_bytes': issue.MEMORY_LIMIT_BYTES,
        'criteria': ['inputs: the payload identity, the embargo, released errors <= D-2, training targets before D; a '
                     'day with no forecastable hour fails here for every policy, so it is a failed issuance (decision 13)',
                     'training: the frozen rules, no guard refusal, no non-finite value',
                     'emission: seven finite, ordered quantiles on every hour of D the frozen design can forecast '
                     '(cp15.data.Inputs.rows); fewer than the day\'s hours is successful on those hours, displayed as '
                     'partial with the missing hours named (decision 13)',
                     'artifact: written and verified inside the chain, before it moves on (N3)',
                     'resources: per attempt, in its own fresh issuing-host process, <= 900 s wall and <= 4 GiB peak; '
                     'an issued artifact carries the measurement, and the native comparison never issues (N2)'],
        'labels': 'released errors from complete delivery days <= D-2 (§2); training targets the delivery days before D '
                  'published strictly before the issue time, so D-1 is one (decision 16, C1)',
        'driver': 'cp17.daily: one fresh process per attempt, a fresh output path, a 900 s timeout; a killed, timed-out, '
                  'non-zero, missing, unreadable or foreign record is a failed fit, and the chain moves on (N1)'}
    base['failure_policy'].update({
        'rulings': 'decisions 11 and 12, kept by decision 15',
        'gfs_missing_in_training_window': 'decision 17 (G2): A1\'s fit fails, v4 fails for the same reason, v2 issues',
        'partial_days': 'decision 13 (C2)',
        'gfs_late_disclosure': "CP-17's item 1: v2 +2.27% MAE and +6.68% WIS against A1's GFS +6 h route on 1,354 "
                               'paired affected development dates'})
    base['shadow'] = {'decision': 14, 'policies': list(shadow.POLICIES),
                      'rule': 'every delivery day, whether or not A1 succeeds, v4, v2 and B0 are fitted under the issuing '
                              'host as shadow issuances, never published, displayed or scored as the product\'s forecast; '
                              'each central and scale is registered before the outcome, and each policy\'s own errors '
                              'from complete delivery days enter its own buffer once, after D-2\'s release (CP-26\'s '
                              'intake: a complete canonical issuance with a valid central and scale); a fallback the '
                              'chain fitted is not fitted twice; v4\'s shadow is skipped when GFS is late or missing',
                      'b0': 'B0 is shadow-issued too: its buffer rebuilt from published prices alone does not equal '
                            'CP-26\'s (reports/cp17b/b0-rebuild.json)',
                      'initial_state': "CP-17's freeze-origin state (CP-26's final replay state), carried by reference"}
    base['disclosures'] = {'F1': 'at the DST origin 2025-10-26, A1 under the issuing host differs from native by up to '
                                 '0.028 EUR/MWh (p50) with every selection identical; tau_eq stays 0.01 as the visitor\'s '
                                 'tolerance against the issued artifact (decision 18)'}
    base['retrain'] = {'tau_eq_eur': retrain.TAU_EQ, 'status': list(retrain.STATUS),
                       'binding': "the retrain must run on the issued fit's training window (payload identity) first"}
    base['definitions'] = {'tau_eur': scoring.TAU_EUR, 'rolling_days': scoring.ROLLING_DAYS,
                           'recompute_relative_tolerance': scoring.RELATIVE_TOLERANCE, 'levels': list(scoring.LEVELS)}
    base['registry']['state'] = 'local names and versions only; nothing is uploaded or registered at CP-17b'
    return base


def manifest(identity: dict, issuance: dict, bundle_manifest_path: Path | None, states_record: dict) -> dict:
    out = freeze.freeze_manifest(identity, issuance, bundle_manifest_path, states_record)
    out.pop('content_sha256')
    out['schema'] = 'cp17b-freeze-manifest-v1'
    out['freeze_origin']['artifact_path'] = ARTIFACT_PATH
    out['verification'] = COVERAGE
    out['content_sha256'] = freeze.digest({k: v for k, v in out.items() if k != 'content_sha256'})
    return out


def cp17_view(root: Path, dest: Path) -> Path:
    """CP-17's superseded freeze as its own verifier (`cp17.tools.freeze.verify`) can still read it after an amendment.

    CP-17's manifest pins governance documents live: the anchor, CP-17's brief, decisions 1-12 and the CP-25 and
    CP-26 rulings. Decision 19 binds them by frozen copies instead. This view holds every file CP-17's manifest
    records, with those documents taken from CP-17b's frozen copies and everything else (computational inputs, code,
    runners, the freeze-origin artifact) taken live, so that an amendment such as r18 leaves CP-17's check green and a
    changed input or code file still fails it. A file absent from `root` is absent from the view."""
    root, dest = Path(root), Path(dest)
    manifest = json.loads((root / CP17['freeze_manifest'][0]).read_text())
    identity = manifest['policy_identity']
    copies = {source: f'{PINNED}/{name}' for source, name, _ in GOVERNANCE.values()}
    rels = [item['path'] for item in identity['pinned_files'].values()] + list(identity['contracts_sha256'])
    rels += [f'src/{rel}' for rel in {**identity['code']['files_sha256'], **identity['code']['runner_files_sha256']}]
    rels.append(manifest['freeze_origin']['artifact_path'])
    for rel in rels:
        source = root / copies.get(rel, rel)
        if source.exists():
            (dest / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, dest / rel)
    return dest


def cp17_governance_live(root: Path) -> list[str]:
    """The documents CP-17's manifest pins that are governance, not computation: each must have a frozen copy."""
    identity = json.loads((Path(root) / CP17['freeze_manifest'][0]).read_text())['policy_identity']
    return sorted(p for p in [i['path'] for i in identity['pinned_files'].values()] + list(identity['contracts_sha256'])
                  if p.endswith('.md'))


def _local(project: Path, identity: dict) -> dict[str, str]:
    """The local-only files the identity records: path -> expected SHA-256."""
    out = {}
    for name, h in identity['issuing_host']['pyodide']['files_sha256'].items():
        out[str(project / freeze.RUNTIME_DIR / name)] = h
    out[identity['issuing_host']['node']['path']] = identity['issuing_host']['node']['sha256']
    search = identity['hyperparameters']['A1_search']
    out[str(project / '.local' / ledger.INPUTS / search['file'])] = search['file_sha256']
    return out


def verify(manifest_value: dict, root: Path, project: Path | None = None, require_local: bool | None = None,
           rebuild: bool = True) -> dict:
    """{'problems': [...], 'unchecked': [...]}: every hash `COVERAGE` names, recomputed (N4)."""
    root = Path(root)
    project = ledger.project_root(root) if project is None else Path(project)
    problems, unchecked = [], []
    if manifest_value.get('schema') != 'cp17b-freeze-manifest-v1':
        problems.append('not a CP-17b freeze manifest')
    if manifest_value.get('content_sha256') != freeze.digest({k: v for k, v in manifest_value.items() if k != 'content_sha256'}):
        problems.append('the manifest\'s own content digest does not hold')
    identity = manifest_value.get('policy_identity', {})
    if manifest_value.get('policy_identity_sha256') != freeze.digest(identity):
        problems.append('the policy identity differs from its recorded digest')
    registry = identity.get('registry', {})
    if registry.get('version') != 1 or registry.get('registered_model') != issue.PRODUCT:
        problems.append('wrong or missing registry name or version')
    governance = identity.get('governance', {})
    if set(governance) != set(GOVERNANCE) or any(governance[n]['sha256'] != GOVERNANCE[n][2] for n in governance):
        problems.append('the governance pins are not the frozen set')
    for item in governance.values():
        path = root / item['copy']
        if not path.exists() or freeze.sha256_file(path) != item['sha256']:
            problems.append(f'{item["copy"]}: the frozen governance copy is missing or altered')
    checks = {**identity.get('computational_sha256', {}), **identity.get('schemas_sha256', {}),
              **{r['path']: r['sha256'] for r in identity.get('supersedes', {}).get('records', {}).values()}}
    code = identity.get('code', {})
    for group in ('files_sha256', 'runner_files_sha256', 'pipeline_files_sha256'):
        if not code.get(group):
            problems.append(f'the code group {group} is empty')
        checks.update({f'src/{rel}': h for rel, h in code.get(group, {}).items()})
    for rel, expected in checks.items():
        path = root / rel
        if not path.exists() or freeze.sha256_file(path) != expected:
            problems.append(f'{rel}: missing or altered')
    for kind, value in schema.SCHEMAS.items():
        rel = f'{SCHEMAS_DIR}/{kind}.schema.json'
        if rel not in identity.get('schemas_sha256', {}) or not (root / rel).exists() \
                or json.loads((root / rel).read_text()) != json.loads(json.dumps(value)):
            problems.append(f'{rel}: does not hold the package\'s schema')
    if (root / PROTOCOL).exists() and identity.get('hyperparameters', {}).get('cp15_protocol_sha256') \
            != freeze.digest(json.loads((root / PROTOCOL).read_text())):
        problems.append(f'{PROTOCOL}: its digest is not the recorded protocol digest')
    if rebuild and code.get('files_sha256'):
        rebuilt, zip_unchecked = _rebuilt(root, code)
        problems += rebuilt
        unchecked += zip_unchecked
    local = _local(project, identity) if identity.get('issuing_host') else {}
    # The issuing machine is one holding the project's pinned runtime or the extract; a system Node.js alone is not.
    issuing = (project / freeze.RUNTIME_DIR).exists() or (project / '.local' / ledger.INPUTS).exists()
    required = issuing if require_local is None else require_local
    for path, expected in local.items():
        if not Path(path).exists():
            (problems if required else unchecked).append(f'{path}: absent')
        elif freeze.sha256_file(Path(path)) != expected:
            problems.append(f'{path}: altered')
    search = identity.get('hyperparameters', {}).get('A1_search', {})
    k13 = project / '.local' / ledger.INPUTS / search.get('file', 'absent')
    if k13.exists() and json.loads(k13.read_text()).get('content_sha256') != search.get('content_sha256'):
        problems.append(f'{k13}: its content digest is not the recorded search state')
    origin = manifest_value.get('freeze_origin', {})
    path = root / origin.get('artifact_path', ARTIFACT_PATH)
    if not path.exists() or freeze.sha256_file(path) != origin.get('artifact_file_sha256'):
        problems.append(f'{origin.get("artifact_path", ARTIFACT_PATH)}: the freeze-origin artifact is missing or altered')
    else:
        art = json.loads(path.read_text())
        if art.get('artifact_sha256') != origin.get('artifact_sha256') or issue.verify_artifact(art) \
                or schema.validate(art, schema.SCHEMAS['fitted_artifact']) \
                or art.get('lineage', {}).get('policy_identity_sha256') != manifest_value.get('policy_identity_sha256') \
                or art.get('lineage', {}).get('code_sha256') != code.get('package_sha256'):
            problems.append('the freeze-origin artifact does not verify against the manifest')
    return {'problems': problems, 'unchecked': unchecked}


def _rebuilt(root: Path, code: dict) -> tuple[list[str], list[str]]:
    """The package rebuilt from `root/src` (its import closure, measured), against the recorded files, identity and
    zip. The zip's compressed bytes depend on the zlib runtime, so it is compared where that equals the recorded
    one, and reported unchecked elsewhere; the files and the package identity never depend on it."""
    from . import bundle
    with tempfile.TemporaryDirectory() as folder:
        try:
            built = bundle.build(root / 'src', Path(folder) / 'src.zip')
        except Exception as exc:     # a package that cannot be rebuilt is a refusal
            return [f'the package could not be rebuilt ({type(exc).__name__}: {exc})'], []
    out, unchecked = [], []
    if built['files_sha256'] != code['files_sha256']:
        out.append('the rebuilt package\'s files differ from the recorded ones')
    if built['package_sha256'] != code['package_sha256']:
        out.append('the rebuilt package\'s identity differs from the recorded one')
    if zlib.ZLIB_RUNTIME_VERSION != code.get('zip_zlib_runtime'):
        unchecked.append(f"the package zip: zlib {zlib.ZLIB_RUNTIME_VERSION} here, {code.get('zip_zlib_runtime')} at the "
                         'freeze (its files and identity are checked)')
    elif built['zip_sha256'] != code['zip_sha256']:
        out.append('the rebuilt package\'s zip differs from the recorded one')
    return out, unchecked


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    sub.add_parser('pin').add_argument('--root', default='.', type=Path)
    a = sub.add_parser('identity')
    a.add_argument('--root', default='.', type=Path)
    a.add_argument('--bundle-manifest', required=True, type=Path)
    a.add_argument('--protocol-from', required=True, type=Path)
    a.add_argument('--browser', type=Path)
    a.add_argument('--out', required=True, type=Path)
    b = sub.add_parser('manifest')
    b.add_argument('--identity', required=True, type=Path)
    b.add_argument('--issuance', required=True, type=Path)
    b.add_argument('--bundle', type=Path)
    b.add_argument('--states', required=True, type=Path)
    b.add_argument('--out', required=True, type=Path)
    c = sub.add_parser('verify')
    c.add_argument('manifest', type=Path)
    c.add_argument('--root', default='.', type=Path)
    args = ap.parse_args(argv)
    if args.cmd == 'pin':
        print(json.dumps(pin(args.root), indent=1))
    elif args.cmd == 'identity':
        governance = pin(args.root)
        value = policy_identity(args.root, json.loads(args.bundle_manifest.read_text()),
                                json.loads(args.protocol_from.read_text())['protocol'], governance,
                                export_schemas(args.root), reference_runtime(args.browser))
        args.out.write_text(json.dumps(value, indent=1, sort_keys=True) + '\n')
        print('policy identity', freeze.digest(value))
    elif args.cmd == 'manifest':
        value = manifest(json.loads(args.identity.read_text()), json.loads(args.issuance.read_text()), args.bundle,
                         json.loads(args.states.read_text()))
        args.out.write_text(json.dumps(value, indent=1, sort_keys=True) + '\n')
        print('freeze manifest', value['content_sha256'])
    else:
        value = json.loads(args.manifest.read_text())
        if value.get('schema') == 'cp17c-freeze-manifest-v1':     # CP-17c's re-freeze, which supersedes this one
            from . import refreeze17c
            result = refreeze17c.verify(value, args.root)
        else:
            result = verify(value, args.root)
        if result['problems']:
            print('\n'.join(result['problems']))
        elif result['unchecked']:
            print(f"verified, except {len(result['unchecked'])} local runtime or state files absent on this machine "
                  '(unchecked, not passed):')
            print('\n'.join(result['unchecked']))
        else:
            print('verified')
        return 1 if result['problems'] else 0
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
