"""CP-17c's re-freeze: A1 under the issuing host r19 §24.13 pins, superseding CP-17b's freeze, and its verifier.

**What changes from CP-17b's identity** (`CHANGES`; r19 §24.13's list "What CP-17c changes"):
- **the issuing host:** the official Node.js v26.7.0 darwin-arm64 binary, by its tarball and `bin/node` hashes,
  retrieved on each run with the release's `SHASUMS256.txt` (`cp17.runtime.retrieve`), at a project-relative path in
  place of Homebrew's; Pyodide's 23 files at their project-relative folder; one CPython patch version and its zlib
  runtime; the machines (`macos-26` runners and the Owner's Apple-silicon Mac), recorded and not pinned;
- **the checks before each run's first issuance (H2),** `cp17.hostcheck`, with the canary's frozen quantile hash;
- **the states' route:** A1's `k13.json`, the freeze-origin states of v4, v2 and B0 and the canary's payload, each
  by hash at a project-relative path (`cp17.runtime`); the shadows' buffers and the issued records by a state store
  (`cp17c.store`, outside the identity);
- **R7's `schema.py` defect, fixed on its error path only (H3);**
- **the code:** the package (`schema.py`), the driver (`daily.py`'s default binary) and the two new pipeline modules;
- **governance** pinned by frozen copies under `docs/track-b/evidence/cp-17c/pinned/` (the anchor at r19, CP-17c's
  brief, decisions 20-21 and 22-25), beside CP-17b's own frozen copies, which stay where they are.

Every other term is CP-17b's, carried unchanged from its committed identity (`carried`).

**Verifying** recomputes every hash the manifest records, as CP-17b's verifier does, with these local checks on the
issuing machine (one holding the pinned Pyodide folder, the binary's folder, the state folder or the CP-26 extract):
the 23 Pyodide files, `bin/node`, `k13.json` and its content digest, the freeze-origin state, the canary payload, and
the CPython patch version. Each is required there; elsewhere (CI) it is reported unchecked, never passed. The package
zip is compared where the zlib runtime equals the recorded one, else reported unchecked. The H2 pins check passes
only on the single line `verified`.

    PYTHONPATH=src python -m cp17.tools.refreeze17c pin
    PYTHONPATH=src python -m cp17.tools.refreeze17c identity --bundle-manifest <src.json> --browser <browser.json> --out <file>
    PYTHONPATH=src python -m cp17.tools.refreeze17c manifest --identity ... --issuance ... --bundle ... --out ...
    PYTHONPATH=src python -m cp17.tools.refreeze verify docs/track-b/evidence/cp-17c/freeze-manifest.json
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import platform
import zlib

from .. import dailyfit, hostcheck, issue, runtime, schema
from . import freeze, ledger, refreeze

EVIDENCE = 'docs/track-b/evidence/cp-17c'
PINNED = f'{EVIDENCE}/pinned'
ARTIFACT_PATH = f'{EVIDENCE}/artifact-2026-09-06.json'
MANIFEST_SCHEMA = 'cp17c-freeze-manifest-v1'
IDENTITY_SCHEMA = 'cp17c-policy-identity-v1'
CP17B_MANIFEST = 'docs/track-b/evidence/cp-17b/freeze-manifest.json'
CP17B_POLICY_IDENTITY = '439115fb984359c0c68f79b77a6da4b85a5e75677cfbeb24e043876793ca9893'
#: name: (the source at the freeze, its frozen copy under `PINNED`, its SHA-256 as the brief cites it)
GOVERNANCE = {
    'anchor_v21_r19': ('capstone_v21.md', 'anchor-capstone-v21-r19.md',
                       'acdeeab18a17ecdb83ade83d369a5031f3180f8fdb618a23af5b37836663996e'),
    'cp17c_brief_5f34f3ab': ('docs/track-b/evidence/cp-17c/issued-brief.md', 'cp-17c-brief-5f34f3ab.md',
                             '5f34f3ab298cfa619ddefaa9b1dcfd5df207c73a0ae06cbf8ed30454ac559c7d'),
    'decisions_20_21': ('docs/track-b/owner-decisions-2026-10-29.md', 'owner-decisions-2026-10-29.md',
                        '5bf496341549025c98d42047b3f5c2056ba30dc36df6a036f82e9a118a41aa3f'),
    'decisions_22_25': ('docs/track-b/owner-decisions-2026-10-30.md', 'owner-decisions-2026-10-30.md',
                        '1286548362b5618e3fb01c616d673c0af83dd79effeaf102bb9727ac219aabab'),
}
#: CP-17b's evidence, superseded and unchanged (the brief's citations).
CP17B = {'freeze_manifest': (CP17B_MANIFEST, '1fe07bddda14035ff44e9a8e6571c12bd58320d501395cbacc185d902f855502'),
         'freeze_record': ('docs/track-b/evidence/cp-17b/freeze-record.md',
                           'aaaa93047dece943db8f5d19f78927ac2f1dfb2da94136425570506e3da1eaca'),
         'protocol': ('docs/track-b/evidence/cp-17b/4-7t-frozen-protocol.md',
                      '04aac76a1fc1166a367eb212c19f802abf2b2ba46f7e63345ed72e46508ae22a'),
         'artifact': ('docs/track-b/evidence/cp-17b/artifact-2026-09-06.json',
                      'c1c3dab694ac828867087d2028c0f5aaeda21b00d58e1e7bf35b2c7c7bbc273d'),
         'integration': ('docs/track-b/evidence/cp-17b/integration.md',
                         'a19c4502479d60998d677d587d123cbeaac0654479d6af7b35884d647f1716c1'),
         'landing_record': ('docs/track-b/cp-17b-landing-2026-10-29.md',
                            '2b64f54261339337c1e372c1ee28a2ad442f14dec8fb99746a4c640ac3bfa72f'),
         'probe_landing_record': ('docs/track-b/cp-17c-probe-landing-2026-10-30.md',
                                  '350ff44cc1f72f7f1c18873d8d0cc5a4e4df70ccf9cf3b7881911f10ce15aeff')}
CHANGES = [
    'issuing_host: the official Node.js v26.7.0 darwin-arm64 binary (tarball and bin/node hashes, retrieved each run '
    'with SHASUMS256.txt) at a project-relative path, in place of Homebrew\'s (r19 §24.13, H1)',
    'issuing_host.cpython: one CPython patch version and its zlib runtime, pinned outside uv.lock and pyproject.toml',
    'issuing_host.machines: macos-26 runners and the Owner\'s Apple-silicon Mac; image version and CPU recorded, not pinned',
    'issuing_host.checks: the checks before each run\'s first issuance (H2), cp17.hostcheck, with the canary hash',
    'state_route: k13.json, the freeze-origin states of v4, v2 and B0, and the canary payload, by hash at '
    'project-relative paths; the shadows\' buffers and issued records by a state store outside the identity',
    'code: schema.py (R7, H3: a malformed issued-forecasts day is listed, not raised), daily.py (the default '
    'binary), the pipeline modules runtime.py and hostcheck.py; the package zip therefore changed',
    'governance: the anchor at r19, CP-17c\'s brief and decisions 20-21 and 22-25 pinned by frozen copies, beside '
    'CP-17b\'s',
    'disclosures.F1: r18\'s largest observed figure, 0.082 EUR/MWh, an observation and not a bound (§24.12, R3)',
    'reference_runtime: the issuing host above; the browsers measured again, since the zip changed',
    'supersedes: CP-17b\'s freeze, whose records stay as committed']
#: How `verify` checks every hash the manifest records.
COVERAGE = {
    **{k: v for k, v in refreeze.COVERAGE.items() if k not in ('policy_identity.issuing_host',
                                                               'policy_identity.hyperparameters.A1_search')},
    'policy_identity.governance': "each frozen copy hashed (CP-17c's under its pinned/, CP-17b's under its own), and "
                                  'the set of pins is the frozen set',
    'policy_identity.issuing_host': 'the 23 Pyodide files, bin/node and the CPython patch version (local; required on '
                                    'the issuing machine, else unchecked); the tarball and SHASUMS256.txt by each run\'s '
                                    'retrieval and H2\'s binary check',
    'policy_identity.state_route': 'k13.json (file hash and content digest), the freeze-origin state and the canary '
                                   'payload, each hashed (local; as above)',
    'policy_identity.supersedes.records': "CP-17b's evidence files hashed live",
    'freeze_origin.canary_quantiles_sha256': "the committed artifact's quantile hash, equal to H2's canary hash"}


def pin(root: Path) -> dict:
    """Copy each new governance document, at its pinned identity, into `PINNED`, once; returns the record, with
    CP-17b's frozen copies carried where they are."""
    root = Path(root)
    (root / PINNED).mkdir(parents=True, exist_ok=True)
    out = {}
    for name, (source, copy_name, expected) in GOVERNANCE.items():
        target = root / PINNED / copy_name
        data = (target if target.exists() else root / source).read_bytes()
        if refreeze._sha(data) != expected:
            raise ValueError(f'{target if target.exists() else source}: {refreeze._sha(data)} is not the pinned {expected}')
        if not target.exists():
            target.write_bytes(data)
        out[name] = {'source_at_freeze': source, 'copy': f'{PINNED}/{copy_name}', 'sha256': expected}
    for name, (source, copy_name, expected) in refreeze.GOVERNANCE.items():
        if freeze.sha256_file(root / refreeze.PINNED / copy_name) != expected:
            raise ValueError(f'{refreeze.PINNED}/{copy_name}: not CP-17b\'s frozen copy')
        out[name] = {'source_at_freeze': source, 'copy': f'{refreeze.PINNED}/{copy_name}', 'sha256': expected}
    return out


def _local_pins(project: Path) -> dict:
    """The state route's and the host's local files, as recorded at the freeze."""
    project = Path(project)
    search = json.loads((project / runtime.K13).read_text())
    canary = project / runtime.CANARY_DIR
    _, meta = dailyfit.load(canary / f'{hostcheck.CANARY_STEM}.npz', canary / f'{hostcheck.CANARY_STEM}.json')
    state = json.loads((project / runtime.FREEZE_STATE).read_text())
    from ..shadow import ShadowStates
    return {
        'k13': {'path': runtime.K13, 'file_sha256': freeze.sha256_file(project / runtime.K13),
                'content_sha256': search['content_sha256'], 'k': search['k'], 'search_day': search['search_day']},
        'freeze_origin_state': {'path': runtime.FREEZE_STATE, 'file_sha256': freeze.sha256_file(project / runtime.FREEZE_STATE),
                                'buffers_sha256': ShadowStates.from_dict(state).digest(),
                                'pending': {p: [i['day'] for i in v['pending']] for p, v in state['h'].items()}
                                | {'B0': [i['day'] for i in state['b0']['pending']]}},
        'canary_payload': {'stem': hostcheck.CANARY_STEM, 'payload_sha256': meta['payload_sha256'],
                           'files_sha256': {f'{runtime.CANARY_DIR}/{p.name}': freeze.sha256_file(p)
                                            for p in sorted(canary.iterdir()) if p.is_file()}}}


def policy_identity(root: Path, project: Path, bundle_manifest: dict, governance: dict, runtime_ref: dict) -> dict:
    root, project = Path(root), Path(project)
    base = json.loads((root / CP17B_MANIFEST).read_text())
    if freeze.sha256_file(root / CP17B_MANIFEST) != CP17B['freeze_manifest'][1]:
        raise ValueError(f'{CP17B_MANIFEST}: not CP-17b\'s committed manifest')
    out = copy.deepcopy(base['policy_identity'])
    out['schema'] = IDENTITY_SCHEMA
    out['governance'] = governance
    out['computational_sha256'] = {p: freeze.sha256_file(root / p) for p in refreeze.COMPUTATIONAL}
    out['code'] = {k: bundle_manifest[k] for k in ('package_sha256', 'zip_sha256', 'files_sha256', 'entry_modules',
                                                   'runner_files_sha256', 'pyodide_packages', 'pipeline_files_sha256')}
    out['code']['zip_zlib_runtime'] = zlib.ZLIB_RUNTIME_VERSION
    if zlib.ZLIB_RUNTIME_VERSION != runtime.ZLIB_RUNTIME or platform.python_version() != runtime.CPYTHON:
        raise ValueError(f'freeze under the pinned CPython {runtime.CPYTHON} (zlib {runtime.ZLIB_RUNTIME}), not '
                         f'{platform.python_version()} (zlib {zlib.ZLIB_RUNTIME_VERSION})')
    for name, expected in out['issuing_host']['pyodide']['files_sha256'].items():
        if freeze.sha256_file(project / runtime.PYODIDE_DIR / name) != expected:
            raise ValueError(f'{runtime.PYODIDE_DIR}/{name}: not the pinned Pyodide file')
    for problem in runtime.check_binary(project)['problems']:
        raise ValueError(problem)
    out['issuing_host'] = {
        'runtime': 'Pyodide 0.28.0 in the official Node.js v26.7.0 darwin-arm64 binary, one worker (r19 §24.13, H1)',
        'pyodide': {**out['issuing_host']['pyodide'], 'folder': runtime.PYODIDE_DIR},
        'node': {'version': runtime.NODE_VERSION, 'path': runtime.NODE_RELATIVE, 'sha256': runtime.BINARY['sha256'],
                 'bytes': runtime.BINARY['bytes'],
                 'tarball': {**runtime.TARBALL, 'url': runtime.NODE_DIST + runtime.TARBALL['name'],
                             'path': runtime.TARBALL_RELATIVE},
                 'shasums': {**runtime.SHASUMS, 'url': runtime.NODE_DIST + runtime.SHASUMS['name']},
                 'retrieval': 'each run retrieves the tarball with SHASUMS256.txt and checks all three by hash '
                              '(cp17.runtime.retrieve); a failed retrieval or hash issues nothing',
                 'superseded': {'path': '/opt/homebrew/Cellar/node/26.7.0/bin/node',
                                'sha256': '1ef99ea25fe70c9b67e7efe768ef8ee22148d3cabc703db6131b57aeb617d040'}},
        'workers': 1,
        'cpython': {'version': runtime.CPYTHON, 'build': runtime.CPYTHON_BUILD, 'zlib_runtime': runtime.ZLIB_RUNTIME,
                    'scope': "the chain's judgement and the shadow states, in cp17.daily, in the environment uv.lock pins"},
        'machines': {'daily_product': f"a GitHub-hosted standard runner, label {runtime.MACHINE['runner_label']} (Apple "
                                      f"silicon), ImageOS {runtime.MACHINE['image_os']}",
                     'local': "the Owner's Apple-silicon Mac (4.7T and any local run)",
                     'recorded_not_pinned': 'the image version and the CPU model, in each run\'s check record '
                                            '(cp17.hostcheck; CP-18\'s records)',
                     'rule': 'a machine that passes the checks is not a runtime change; any other binary, runtime '
                             'file, operating system or architecture is (§16.2)'},
        'checks': {'module': 'cp17.hostcheck', 'order': list(hostcheck.ORDER), 'manifest': hostcheck.MANIFEST,
                   'canary': {'stem': hostcheck.CANARY_STEM, 'quantiles_sha256': hostcheck.CANARY_QUANTILES_SHA256},
                   'rule': 'once per run before its first issuance, in order, and again when the machine or runtime '
                           'changes within a run; any failure issues nothing in the run; for 4.7T a failure is a '
                           'blocker, never a day\'s outcome'}}
    local = _local_pins(project)
    search = out['hyperparameters']['A1_search']
    if local['k13']['file_sha256'] != search['file_sha256'] or local['k13']['content_sha256'] != search['content_sha256']:
        raise ValueError(f'{runtime.K13}: not the frozen search state S_13')
    out['hyperparameters']['A1_search'] = {**search, 'file': runtime.K13}
    cp17_state = json.loads((root / refreeze.CP17['freeze_manifest'][0]).read_text())['initial_state']['buffers_sha256']
    if any(local['freeze_origin_state']['buffers_sha256'][p] != cp17_state[p] for p in ('v4', 'v2', 'B0')):
        raise ValueError(f'{runtime.FREEZE_STATE}: not CP-17\'s freeze-origin state')
    if local['canary_payload']['payload_sha256'] != base['freeze_origin']['payloads']['A1']:
        raise ValueError('the canary payload is not the freeze-origin issuance\'s')
    out['state_route'] = {
        **local, 'store': {'shadow_buffers_and_issued_records': 'a state store per run (cp17c.store): the shadow '
                                                                 'states after each day, the issued artifacts, and a '
                                                                 'hash chain; CP-18 builds its persistence'},
        'reaches_the_runner': 'every file at the same project-relative path, checked by hash before the first fit'}
    out['reference_runtime'] = runtime_ref
    out['supersedes'] = {'checkpoint': 'CP-17b', 'policy_identity_sha256': CP17B_POLICY_IDENTITY,
                         'records': {k: {'path': p, 'sha256': h} for k, (p, h) in CP17B.items()},
                         'chain': base['policy_identity']['supersedes'],
                         'note': "CP-17b's freeze is superseded, not edited; its records stay as committed"}
    out['disclosures'] = {**base['policy_identity']['disclosures'],
                          'F1_largest_observed': 'A1 under the issuing host differs from native by up to 0.082 EUR/MWh '
                                                 '(p2.5, 2026-03-30; 0.028 at its p50), with every selection identical: '
                                                 'an observation, not a bound (§24.12, R3)'}
    out['defects_fixed'] = {'R7': 'cp17.schema.verify_bundle lists a structurally malformed day of a re-hashed '
                                  'issued-forecasts record as a fault instead of raising (r19 §24.13, H3); every fit, '
                                  'quantile and artifact field is unchanged'}
    out['registry'] = {**out['registry'], 'state': 'local names and versions only; nothing is uploaded or registered '
                                                   'at CP-17c'}
    out['changes_from_cp17b'] = CHANGES
    return out


def manifest(identity: dict, issuance: dict, bundle_manifest_path: Path | None, root: Path) -> dict:
    base = json.loads((Path(root) / CP17B_MANIFEST).read_text())
    out = freeze.freeze_manifest(identity, issuance, bundle_manifest_path, {
        **base['initial_state'], 'carried': "CP-17's freeze-origin state, carried by reference through CP-17b's "
                                            f"manifest ({CP17B['freeze_manifest'][1][:8]}...); its file on the state "
                                            f"route is {runtime.FREEZE_STATE}, by hash in the identity"})
    out.pop('content_sha256')
    out['schema'] = MANIFEST_SCHEMA
    out['freeze_origin']['artifact_path'] = ARTIFACT_PATH
    out['freeze_origin']['canary_quantiles_sha256'] = issuance['artifact']['quantiles_sha256']
    out['verification'] = COVERAGE
    out['content_sha256'] = freeze.digest({k: v for k, v in out.items() if k != 'content_sha256'})
    return out


def _governance_problems(identity: dict, root: Path) -> list[str]:
    governance, problems = identity.get('governance', {}), []
    frozen = {**{n: v[2] for n, v in GOVERNANCE.items()}, **{n: v[2] for n, v in refreeze.GOVERNANCE.items()}}
    if set(governance) != set(frozen) or any(governance[n]['sha256'] != frozen[n] for n in governance):
        problems.append('the governance pins are not the frozen set')
    for item in governance.values():
        path = root / item['copy']
        if not path.exists() or freeze.sha256_file(path) != item['sha256']:
            problems.append(f'{item["copy"]}: the frozen governance copy is missing or altered')
    return problems


def _local(project: Path, identity: dict) -> dict[str, str]:
    """The local-only files the identity records: project-relative path -> expected SHA-256."""
    host, route = identity.get('issuing_host', {}), identity.get('state_route', {})
    out = {f"{host.get('pyodide', {}).get('folder', runtime.PYODIDE_DIR)}/{name}": h
           for name, h in host.get('pyodide', {}).get('files_sha256', {}).items()}
    if host.get('node'):
        out[host['node']['path']] = host['node']['sha256']
    for key in ('k13', 'freeze_origin_state'):
        if key in route:
            out[route[key]['path']] = route[key]['file_sha256']
    out.update(route.get('canary_payload', {}).get('files_sha256', {}))
    return out


def issuing_machine(project: Path) -> bool:
    """One holding the pinned runtime or the states; a system Node.js alone is not."""
    project = Path(project)
    return any((project / p).exists() for p in (runtime.PYODIDE_DIR, runtime.NODE_DIR, runtime.STATE_DIR,
                                                f'.local/{ledger.INPUTS}'))


def verify(manifest_value: dict, root: Path, project: Path | None = None, require_local: bool | None = None,
           rebuild: bool = True) -> dict:
    """{'problems': [...], 'unchecked': [...]}: every hash `COVERAGE` names, recomputed."""
    root = Path(root)
    project = ledger.project_root(root) if project is None else Path(project)
    problems, unchecked = [], []
    if manifest_value.get('schema') != MANIFEST_SCHEMA:
        problems.append('not a CP-17c freeze manifest')
    if manifest_value.get('content_sha256') != freeze.digest({k: v for k, v in manifest_value.items() if k != 'content_sha256'}):
        problems.append("the manifest's own content digest does not hold")
    identity = manifest_value.get('policy_identity', {})
    if manifest_value.get('policy_identity_sha256') != freeze.digest(identity) or identity.get('schema') != IDENTITY_SCHEMA:
        problems.append('the policy identity differs from its recorded digest')
    registry = identity.get('registry', {})
    if registry.get('version') != 1 or registry.get('registered_model') != issue.PRODUCT:
        problems.append('wrong or missing registry name or version')
    problems += _governance_problems(identity, root)
    records = identity.get('supersedes', {}).get('records', {})
    if {k: (r['path'], r['sha256']) for k, r in records.items()} != CP17B:
        problems.append("the superseded records are not CP-17b's")
    checks = {**identity.get('computational_sha256', {}), **identity.get('schemas_sha256', {}),
              **{r['path']: r['sha256'] for r in records.values()}}
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
        rel = f'{refreeze.SCHEMAS_DIR}/{kind}.schema.json'
        if rel not in identity.get('schemas_sha256', {}) or not (root / rel).exists() \
                or json.loads((root / rel).read_text()) != json.loads(json.dumps(value)):
            problems.append(f"{rel}: does not hold the package's schema")
    if (root / refreeze.PROTOCOL).exists() and identity.get('hyperparameters', {}).get('cp15_protocol_sha256') \
            != freeze.digest(json.loads((root / refreeze.PROTOCOL).read_text())):
        problems.append(f'{refreeze.PROTOCOL}: its digest is not the recorded protocol digest')
    if rebuild and code.get('files_sha256'):
        rebuilt, zip_unchecked = refreeze._rebuilt(root, code)
        problems += rebuilt
        unchecked += zip_unchecked
    required = issuing_machine(project) if require_local is None else require_local
    for rel, expected in _local(project, identity).items():
        path = project / rel
        if not path.exists():
            (problems if required else unchecked).append(f'{rel}: absent')
        elif freeze.sha256_file(path) != expected:
            problems.append(f'{rel}: altered')
    cpython = identity.get('issuing_host', {}).get('cpython', {}).get('version')
    if platform.python_version() != cpython:
        (problems if required else unchecked).append(f'CPython {platform.python_version()} here, {cpython} pinned')
    k13 = identity.get('state_route', {}).get('k13', {})
    if k13 and (project / k13['path']).exists():
        try:
            content = json.loads((project / k13['path']).read_text()).get('content_sha256')
        except (ValueError, OSError, AttributeError):       # an unreadable search state is a refusal, never a crash
            content = None
        if content != k13.get('content_sha256'):
            problems.append(f"{k13['path']}: its content digest is not the recorded search state")
    origin = manifest_value.get('freeze_origin', {})
    path = root / origin.get('artifact_path', ARTIFACT_PATH)
    if not path.exists() or freeze.sha256_file(path) != origin.get('artifact_file_sha256'):
        problems.append(f'{origin.get("artifact_path", ARTIFACT_PATH)}: the freeze-origin artifact is missing or altered')
    else:
        art = json.loads(path.read_text())
        if art.get('artifact_sha256') != origin.get('artifact_sha256') or issue.verify_artifact(art) \
                or schema.validate(art, schema.SCHEMAS['fitted_artifact']) \
                or art.get('lineage', {}).get('policy_identity_sha256') != manifest_value.get('policy_identity_sha256') \
                or art.get('lineage', {}).get('code_sha256') != code.get('package_sha256') \
                or art.get('quantiles_sha256') != origin.get('canary_quantiles_sha256') \
                or origin.get('canary_quantiles_sha256') != identity.get('issuing_host', {}).get('checks', {}) \
                .get('canary', {}).get('quantiles_sha256'):
            problems.append('the freeze-origin artifact does not verify against the manifest')
    return {'problems': problems, 'unchecked': unchecked}


def cp17b_superseded(root: Path, project: Path | None = None) -> list[str]:
    """CP-17b's own verifier on the live tree: it must report exactly the package and pipeline files CP-17c changed,
    and the rebuilt package that follows from them, and nothing else."""
    manifest_value = json.loads((Path(root) / CP17B_MANIFEST).read_text())
    return refreeze.verify(manifest_value, root, project)['problems']


#: What CP-17b's verifier reports on CP-17c's tree, by design (`tests/cp17c/test_refreeze17c.py`).
CP17B_EXPECTED = ['src/cp17/schema.py: missing or altered', 'src/cp17/daily.py: missing or altered',
                  "the rebuilt package's files differ from the recorded ones",
                  "the rebuilt package's identity differs from the recorded one",
                  "the rebuilt package's zip differs from the recorded one"]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    sub.add_parser('pin').add_argument('--root', default='.', type=Path)
    a = sub.add_parser('identity')
    a.add_argument('--root', default='.', type=Path)
    a.add_argument('--bundle-manifest', required=True, type=Path)
    a.add_argument('--browser', type=Path)
    a.add_argument('--out', required=True, type=Path)
    b = sub.add_parser('manifest')
    b.add_argument('--root', default='.', type=Path)
    b.add_argument('--identity', required=True, type=Path)
    b.add_argument('--issuance', required=True, type=Path)
    b.add_argument('--bundle', type=Path)
    b.add_argument('--out', required=True, type=Path)
    c = sub.add_parser('verify')
    c.add_argument('manifest', type=Path)
    c.add_argument('--root', default='.', type=Path)
    args = ap.parse_args(argv)
    if args.cmd == 'pin':
        print(json.dumps(pin(args.root), indent=1))
    elif args.cmd == 'identity':
        ref = refreeze.reference_runtime(args.browser)
        ref['issuing_host'] = 'the issuing host above (r19 §24.13)'
        ref['node'] = f'{runtime.NODE_VERSION}, the official darwin-arm64 binary {runtime.BINARY["sha256"][:8]}…'
        value = policy_identity(args.root, ledger.project_root(args.root), json.loads(args.bundle_manifest.read_text()),
                                pin(args.root), ref)
        args.out.write_text(json.dumps(value, indent=1, sort_keys=True) + '\n')
        print('policy identity', freeze.digest(value))
    elif args.cmd == 'manifest':
        value = manifest(json.loads(args.identity.read_text()), json.loads(args.issuance.read_text()), args.bundle,
                         args.root)
        args.out.write_text(json.dumps(value, indent=1, sort_keys=True) + '\n')
        print('freeze manifest', value['content_sha256'])
    else:
        return refreeze.main(['verify', str(args.manifest), '--root', str(args.root)])
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
