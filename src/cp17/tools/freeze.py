"""Item 4's machine-readable freeze: the policy identity, the freeze manifest, and its verifier.

**Two levels.** The *policy identity* is everything that defines the frozen product policy: code and
dependencies, the issuing host, the data and availability contracts, hyperparameters and search state,
windows and label rules, seeds, the refit schedule, the uncertainty layers, the failure policy, the success
definition, §19.5's definitions, §19.3's status rule and τ_eq, and the registry names. Its SHA-256 is cited by
every issued artifact. The *freeze manifest* is the policy identity plus the initial artifact and state at the
freeze origin (delivery 2026-09-06, issued under the issuing host) and the reference bundle's manifest.

**Verification** (§9: "verify fingerprints before opening outcome data, and refuse wrong/missing versions or
altered artifacts"): `verify` recomputes every recorded file hash from the files themselves and refuses on any
difference, a missing file, a wrong version, or a manifest whose own digests do not hold.

    PYTHONPATH=src python -m cp17.tools.freeze identity --bundle-manifest <src.json> --out <policy-identity.json>
    PYTHONPATH=src python -m cp17.tools.freeze manifest --identity <policy-identity.json> --issuance <issuance.json> \\
        --bundle <reference-bundle-manifest.json> --out docs/track-b/evidence/cp-17/freeze-manifest.json
    PYTHONPATH=src python -m cp17.tools.freeze verify docs/track-b/evidence/cp-17/freeze-manifest.json
"""
from __future__ import annotations

import argparse
from datetime import date, timedelta
import hashlib
import json
from pathlib import Path

from .. import dailyfit, issue, retrain, scoring
from . import ledger, states

ROOT_FILES = {
    'anchor': ('capstone_v21.md', '3124451da84c3cd99fae728c371207aace6ae537474f7769d0ff6333ee53e4fc'),
    'brief': ('docs/track-b/evidence/cp-17/issued-brief.md', '9f235cf84876db4975930f1fce8fb62dbe188b8be4130e692eda9b5b7e1b1a73'),
    'rulings': ('docs/track-b/pre-cp-17-rulings-2026-10-25.md', 'd29b0abc6a17dcb6bca2cbd464c1dcc5858c50e2b653a42e646d3dc66ea8f6b0'),
    'cp28_selection': ('reports/cp28/results/selection.json', '22a3dacfc06303a42db132ae3ba8ee8fc7e82606a886585fc40d02b82ae930b0'),
    'uv_lock': ('uv.lock', '5d2aecc3d173fef6191cb2bb9d14cd008c0e9cc764700f1352b81d1924465a94'),
    'pyproject': ('pyproject.toml', 'aa35326abbb13dfa4ea8ba1b4b367343b77f895e7724d5dc94e0bdb7637b4e4a'),
    'snapshot': ('data/snapshot.parquet', '7dd2dc73407706ca6bd3c1ad51d201ac0de35eec5ca129ce320cca639f697f00'),
    'cp25_manifest': ('reports/development-inputs/manifest.json', 'f3e8ca15e4466d371de5b6db9c181f77adaa8823ca2e7e0d325182120db32fd9'),
    'cp26_scores_manifest': ('reports/rebaseline/cold-rerun-2026-10-15/scores-manifest.json',
                             'b9f10917ab471177eda1b5f8ef735ddca5a651fa2e9bb47844497f3a6deaf84d'),
}
CONTRACT_FILES = ('reports/development-inputs/tier-register.json', 'reports/development-inputs/partitions-development.json',
                  'reports/development-inputs/margin-days.csv', 'reports/development-inputs/margin-table.csv',
                  'docs/track-b/cp-25-rulings-2026-10-12.md', 'docs/track-b/cp-26-gfs-ruling-2026-10-19.md',
                  'reports/weather-ablation/weather-features.parquet',
                  'reports/development-inputs/weather-features-gaps.parquet')
RUNTIME_DIR = '.local/tools/pyodide-0.28.0'
PROBE_RETRIEVAL = 'reports/cp17-probe/retrieval-manifest.json'
SEARCH = 'reports/rebaseline/cold-rerun-2026-10-15/search-results/nominal/k13.json'
FREEZE_ORIGIN = date(2026, 9, 6)
ARTIFACT_PATH = 'docs/track-b/evidence/cp-17/artifact-2026-09-06.json'
SEARCH_START, SEARCH_SPACING = date(2020, 1, 15), 182


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def digest(value) -> str:
    return hashlib.sha256(dailyfit.canonical(value)).hexdigest()


def search_day(k: int) -> date:
    return SEARCH_START + timedelta(days=SEARCH_SPACING * k)


def _pyodide(root: Path) -> dict:
    """The pinned runtime by its file hashes, checked against CP-17-probe's committed retrieval manifest."""
    probe = json.loads((Path(root) / PROBE_RETRIEVAL).read_text())
    project = ledger.project_root(root)
    files = {}
    for item in probe['pyodide']['files']:
        actual = sha256_file(project / item['path'])
        if actual != item['sha256']:
            raise ValueError(f"{item['path']}: not the retrieved runtime file")
        files[Path(item['path']).name] = actual
    return {'version': probe['pyodide']['version'], 'lock_info': probe['pyodide']['lock_info'], 'files_sha256': files,
            'retrieval_manifest': PROBE_RETRIEVAL, 'retrieval_manifest_sha256': sha256_file(Path(root) / PROBE_RETRIEVAL)}


def policy_identity(root: Path, bundle_manifest: dict, protocol: dict) -> dict:
    """`protocol` is CP-15's recipe as the payloads carry it (`meta['protocol']` of a v2 or v4 payload)."""
    from cp21.lgbm import GRID
    from .hosts import node_identity
    root = Path(root)
    pinned = {}
    for name, (path, expected) in ROOT_FILES.items():
        actual = sha256_file(root / path)
        if actual != expected:
            raise ValueError(f'{path}: {actual} is not the pinned {expected}')
        pinned[name] = {'path': path, 'sha256': actual}
    contracts = {p: sha256_file(root / p) for p in CONTRACT_FILES}
    search = json.loads((ledger.inputs_dir(root) / SEARCH).read_text())
    protocol_sha = digest(protocol)
    return {
        'schema': 'cp17-policy-identity-v1', 'product': issue.PRODUCT, 'designated_policy': issue.DESIGNATED,
        'designation': "A1 (DDNN-2 alone, CP-26's nominal D2 with its own Johnson SU quantiles), designated the final "
                       "version pending 4.7T (§16.1; decision 1); v1 remains the product until CP-19",
        'pinned_files': pinned, 'contracts_sha256': contracts,
        'code': {'package_sha256': bundle_manifest['package_sha256'], 'zip_sha256': bundle_manifest['zip_sha256'],
                 'files_sha256': bundle_manifest['files_sha256'], 'entry_modules': bundle_manifest['entry_modules'],
                 'runner_files_sha256': bundle_manifest['runner_files_sha256'],
                 'pyodide_packages': bundle_manifest['pyodide_packages']},
        'issuing_host': {'runtime': 'Pyodide in Node.js (decision 10)', 'pyodide': _pyodide(root), 'node': node_identity(),
                         'workers': 1},
        'availability': {'issue_time': 'D-1 11:00 Europe/Berlin (§24.4): 10:00 UTC on CET, 09:00 UTC on CEST, by D-1\'s '
                                       'own offset at 11:00', 'strictly_before': True,
                         'target_day_embargo': "no price of D in any zone, nor a transformation revealing it",
                         'tiers': 'reports/development-inputs/tier-register.json with the Owner\'s CP-25 rulings and the '
                                  'GFS ruling of 2026-10-19 (§24.5)'},
        'hyperparameters': {
            'A1_search': {'k': search['k'], 'search_day': search['search_day'], 'route': 'nominal',
                          'file': SEARCH, 'file_sha256': sha256_file(ledger.inputs_dir(root) / SEARCH),
                          'content_sha256': search['content_sha256'],
                          'members': [{'rank': m['rank'], 'repeat': m['repeat'], 'seed': m['seed'],
                                       'config_id': m['config']['id']} for m in search['members']]},
            'cp15_protocol_sha256': protocol_sha, 'lightgbm_grid': list(GRID),
            'v4_blend': 'A1_w/3 + B2_w/3 + L-N/6 + L-R/6 (cp21.lgbm.hgl_central)', 'v2_blend': 'A1/2 + B2/2 (hg_central)',
            'v5_blend_control': 'A1_w/3 + B2_w/3 + L-N/12 + L-R/12 + D2/6 (cp24.execution.v5_central)'},
        'windows': {'A1': '[max(2019-01-01, D-728), D) training days, with lags reaching D-735; held-out weeks and '
                          'early stopping per member (cp24.member.fit_member)',
                    'LEAR': 'cp15.models.history_start by member, to D exclusive; chronological inner validation on the '
                            'last 28 days', 'LightGBM': '[max(2019-01-01, D-728), D); capacity on the window minus its last '
                                                        '28 days (cp21.lgbm)',
                    'training_targets': 'delivery days < D published strictly before the issue time (D-1 is published '
                                        'on D-2)', 'released_errors': 'complete delivery days <= D-2, consumed once (§2)'},
        'seeds': {'A1_members': [int(m['seed']) for m in search['members']], 'lightgbm_random_state': 42, 'lear': None,
                  'search_seeds': 'trial/batch 26_000_000 + k*100_000 + trial*100 + batch; members '
                                  '26_000_000 + k*100_000 + 90_000 + 10*rank + repeat; sampler SeedSequence([26, k, 0x5A3])'},
        'refit_schedule': {'daily': 'every delivery day, including DST days', 'search': 'S_k = 2020-01-15 + 182k',
                           'current': f"S_13 = {search_day(13)}", 'next': f"S_14 = {search_day(14)}",
                           'note': 'protocol dates only, never permission to read future data'},
        'uncertainty': {'A1': "DDNN-2's own Johnson SU quantiles, per-level median of the eight members",
                        'v4_v2': 'the H layer (cp16.residuals): 28 released standardized errors, hour shrinkage n/(n+56) '
                                 'from 14 distinct days', 'B0': '28 released raw errors, pooled linear quantiles',
                        'v5_control': "v5's own H layer"},
        'failure_policy': {'gfs_unavailable': list(issue.CHAINS['gfs_unavailable']),
                           'a1_failed': list(issue.CHAINS['a1_failed']), 'rulings': 'decisions 11 and 12, 2026-10-28',
                           'display': 'a failed or skipped fit is displayed as such, even when a fallback issues'},
        'success_definition': {'wall_seconds': issue.WALL_LIMIT_SECONDS, 'peak_bytes': issue.MEMORY_LIMIT_BYTES,
                               'criteria': ['inputs', 'training', 'emission (every delivery hour of D)', 'artifact',
                                            'resources']},
        'definitions': {'tau_eur': scoring.TAU_EUR, 'rolling_days': scoring.ROLLING_DAYS,
                        'recompute_relative_tolerance': scoring.RELATIVE_TOLERANCE, 'levels': list(scoring.LEVELS)},
        'retrain': {'tau_eq_eur': retrain.TAU_EQ, 'status': list(retrain.STATUS)},
        'registry': {'registered_model': 'delu-day-ahead-policy', 'version': 1, 'experiment_live': 'delu-live',
                     'state': 'local names and versions only; nothing is uploaded or registered at CP-17'},
        'lock_price_policy': 'not adopted (decision 3)'}


def freeze_manifest(identity: dict, issuance: dict, bundle_manifest_path: Path | None, states_record: dict) -> dict:
    out = {'schema': 'cp17-freeze-manifest-v1', 'policy_identity_sha256': digest(identity), 'policy_identity': identity,
           'freeze_origin': {'delivery_day': str(FREEZE_ORIGIN), 'issue_utc': issue.issue_utc(FREEZE_ORIGIN),
                             'artifact_sha256': issuance['artifact']['artifact_sha256'],
                             'issued_policy': issuance['artifact']['issued_policy'],
                             'display': issuance['artifact']['display'],
                             'artifact_path': ARTIFACT_PATH,
                             'artifact_file_sha256': issuance.get('artifact_file_sha256'),
                             'artifact_arrays_sha256': issuance.get('artifact_arrays_sha256'),
                             'payloads': issuance['artifact']['lineage']['inputs'],
                             'host': issuance['artifact']['lineage']['host']},
           'initial_state': states_record}
    if bundle_manifest_path is not None:
        out['reference_bundle_manifest_sha256'] = sha256_file(bundle_manifest_path)
    out['content_sha256'] = digest({k: v for k, v in out.items() if k != 'content_sha256'})
    return out


def verify(manifest: dict, root: Path) -> list[str]:
    """Refuse a manifest whose digests do not hold, or any recorded file that is missing or altered."""
    root = Path(root)
    problems = []
    body = {k: v for k, v in manifest.items() if k != 'content_sha256'}
    if manifest.get('content_sha256') != digest(body):
        problems.append('the manifest\'s own content digest does not hold')
    identity = manifest.get('policy_identity', {})
    if manifest.get('policy_identity_sha256') != digest(identity):
        problems.append('the policy identity differs from its recorded digest')
    if identity.get('registry', {}).get('version') != 1 or identity.get('registry', {}).get('registered_model') != issue.PRODUCT:
        problems.append('wrong or missing registry name or version')
    for name, item in identity.get('pinned_files', {}).items():
        path = root / item['path']
        if not path.exists():
            problems.append(f'{item["path"]}: missing')
        elif sha256_file(path) != item['sha256']:
            problems.append(f'{item["path"]}: altered')
    for path, expected in identity.get('contracts_sha256', {}).items():
        if not (root / path).exists() or sha256_file(root / path) != expected:
            problems.append(f'{path}: missing or altered')
    code = identity.get('code', {})
    for rel, expected in {**code.get('files_sha256', {}), **code.get('runner_files_sha256', {})}.items():
        path = root / 'src' / rel
        if not path.exists() or sha256_file(path) != expected:
            problems.append(f'src/{rel}: missing or altered')
    if not code.get('runner_files_sha256'):
        problems.append('the runners that load the package are not frozen')
    origin = manifest.get('freeze_origin', {})
    path = root / origin.get('artifact_path', ARTIFACT_PATH)
    if not path.exists() or sha256_file(path) != origin.get('artifact_file_sha256'):
        problems.append(f'{origin.get("artifact_path", ARTIFACT_PATH)}: the freeze-origin artifact is missing or altered')
    else:
        art = json.loads(path.read_text())
        if art.get('artifact_sha256') != origin.get('artifact_sha256') or issue.verify_artifact(art) \
                or art.get('lineage', {}).get('policy_identity_sha256') != manifest.get('policy_identity_sha256'):
            problems.append('the freeze-origin artifact does not verify against the manifest')
    return problems


def initial_state(root: Path, ctx=None) -> dict:
    """The fallbacks' interval states at the freeze origin (CP-26's, rebuilt and checked), by digest."""
    from . import payloads
    ctx = ctx or payloads.context(root)
    rebuilt = states.rebuild(root, ctx, [FREEZE_ORIGIN])
    return {'origin': str(FREEZE_ORIGIN), 'source': "CP-26's nominal ledger, rebuilt and checked against its final "
                                                    "replay state's consumption trace",
            'replay_state_sha256': rebuilt['replay_state_sha256'],
            'buffers_sha256': {p: states.buffer_sha256(rebuilt['buffers'][p][FREEZE_ORIGIN]) for p in states.POLICIES},
            'buffer_days': {p: [b['day'] for b in rebuilt['buffers'][p][FREEZE_ORIGIN]][::27] for p in states.POLICIES}}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    a = sub.add_parser('identity')
    a.add_argument('--root', default='.', type=Path)
    a.add_argument('--bundle-manifest', required=True, type=Path)
    a.add_argument('--protocol-from', required=True, type=Path, help='a v2 or v4 payload meta file')
    a.add_argument('--out', required=True, type=Path)
    b = sub.add_parser('manifest')
    b.add_argument('--root', default='.', type=Path)
    b.add_argument('--identity', required=True, type=Path)
    b.add_argument('--issuance', required=True, type=Path)
    b.add_argument('--bundle', type=Path)
    b.add_argument('--states', required=True, type=Path)
    b.add_argument('--out', required=True, type=Path)
    c = sub.add_parser('verify')
    c.add_argument('manifest', type=Path)
    c.add_argument('--root', default='.', type=Path)
    s = sub.add_parser('states')
    s.add_argument('--root', default='.', type=Path)
    s.add_argument('--out', required=True, type=Path)
    args = ap.parse_args(argv)
    if args.cmd == 'identity':
        value = policy_identity(args.root, json.loads(args.bundle_manifest.read_text()),
                                json.loads(args.protocol_from.read_text())['protocol'])
        args.out.write_text(json.dumps(value, indent=1, sort_keys=True) + '\n')
        print('policy identity', digest(value))
    elif args.cmd == 'states':
        args.out.write_text(json.dumps(initial_state(args.root), indent=1, sort_keys=True) + '\n')
    elif args.cmd == 'manifest':
        value = freeze_manifest(json.loads(args.identity.read_text()), json.loads(args.issuance.read_text()), args.bundle,
                                json.loads(args.states.read_text()))
        args.out.write_text(json.dumps(value, indent=1, sort_keys=True) + '\n')
        print('freeze manifest', value['content_sha256'])
    else:
        problems = verify(json.loads(args.manifest.read_text()), args.root)
        print('verified' if not problems else '\n'.join(problems))
        return 1 if problems else 0
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
