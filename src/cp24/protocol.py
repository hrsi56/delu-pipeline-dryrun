"""The frozen protocol of scored attempt k (capstone v21-r11 §23.7, §23.6, §23.10 pre-registration).

Written and committed after the S1 answer that freezes a round's design, and before any of the
attempt's warm-up or evaluation fits. It records the representation, the input groups and their
encoding; the search (space, priors, sampler, trial count, seeds, batch dates, ranking metric); every
fold's chosen ensemble with its validation scores; the gate's dates, its excluded training days and its
results; the training recipe, the guards and the emission; the reference tolerances; the arms and the
weight; the attempt number, the level and the rule's text, quoted verbatim from the ratified anchor;
4.6R′'s fixed values; and the budget accounting.

`check_protocol` is called by every fitting and scoring entry point of the attempt: it refuses unless
the protocol is committed at HEAD, its frozen implementation and inputs are unchanged, and the protocol
commit is an ancestor of HEAD.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import sys

from cp15.data import sha
from . import ddnn2 as M
from . import design as G
from .budget import CAPS, TIMEBOX_ACTIVE_SECONDS, atomic, effective_caps, ledger
from .inputs import ISSUED, identities, origin_manifest
from .jobs import stamp
from .reference import RECORD, REFERENCE_TOLERANCES, TRAINING_LOOP_EPOCHS, TRAJECTORY_STEPS, require_passing_record
from .rounds import RECIPE

OUT = Path('reports/ddnn2')
IMPLEMENTATION = [
    'src/cp24/__init__.py', 'src/cp24/budget.py', 'src/cp24/inputs.py', 'src/cp24/ddnn2.py', 'src/cp24/design.py',
    'src/cp24/member.py', 'src/cp24/sampler.py', 'src/cp24/audit.py', 'src/cp24/reference.py', 'src/cp24/admission.py',
    'src/cp24/search.py', 'src/cp24/gate.py', 'src/cp24/v4members.py', 'src/cp24/rounds.py', 'src/cp24/preflight.py',
    'src/cp24/protocol.py', 'src/cp24/execution.py', 'src/cp24/scoring.py', 'src/cp24/evaluate.py', 'src/cp24/jobs.py',
    'scripts/cp24_ddnn2.py',
    'tests/cp24/test_numpy_only.py', 'tests/cp24/test_ddnn2_gradients.py', 'tests/cp24/test_reference_record.py',
    'tests/cp24/torch_reference_checks.py', 'tests/cp23/torch-reference/pyproject.toml', 'tests/cp23/torch-reference/uv.lock',
    'src/cp15/data.py', 'src/cp15/models.py', 'src/cp15/scoring.py', 'src/cp16/residuals.py', 'src/cp20/components.py',
    'src/cp20/weather.py', 'src/cp20/scoring.py', 'src/cp20/budget.py', 'src/cp21/lgbm.py', 'src/cp21/inputs.py',
    'src/cp21/components.py', 'src/cp21/execution.py', 'src/cp22/execution.py', 'src/cp22/inputs.py', 'src/cp22/pn.py',
    'src/cp22/scoring.py', 'src/cp23/inputs.py', 'src/cp23/audit.py', 'src/cp23/scoring.py',
    'src/delu_forecast/features.py', 'src/delu_forecast/folds.py', 'src/delu_forecast/baselines.py',
    'src/delu_forecast/ingest.py']
FROZEN = ['data/snapshot.parquet', 'data/partitions.json', 'uv.lock', 'pyproject.toml', 'reports/cp15/protocol.json',
          'reports/cp15/predictions.parquet', 'reports/v2-causal/input-manifest.json', 'reports/weather-ablation/protocol.json',
          'reports/weather-ablation/lineage.json', 'reports/weather-ablation/weather-features.parquet',
          'reports/weather-ablation/predictions.parquet', 'reports/weather-ablation/criteria.csv',
          'reports/block-challenger/predictions.parquet', 'reports/block-challenger/lineage.json',
          'reports/block-challenger/criteria.csv', 'reports/block-challenger/uncertainty.csv',
          'reports/distribution-challenger/predictions.parquet', 'reports/distribution-challenger/members.parquet',
          'reports/ddnn2/licence-admission.md', 'reports/ddnn2/resource-admission.json', 'reports/ddnn2/resource-admission.md',
          'reports/ddnn2/reference-checks.json', 'reports/ddnn2/v4-parity.json',
          'reports/ddnn2/preflight/input-verification.json', 'reports/ddnn2/preflight/weather-regeneration.json']
LEVEL = {1: 0.975, 2: 0.975}


def recorded_modules(root: Path) -> dict:
    """Every other CP-24 module, test and script at the freeze, recorded but not enforced by
    `check_protocol`: they run after scoring (controls, diagnostics, the daily cycle, fit cost, export,
    packet, claims, report, review, finalise, tracking) or never touch an attempt's forecast (the base-tree
    record, round 1's search-ledger repair). A later change is listed in `defects-and-repairs.md`, and none
    may change a frozen element (§23.6)."""
    root = Path(root)
    names = sorted({str(q.relative_to(root)) for pattern in ('src/cp24/*.py', 'tests/cp24/*.py', 'scripts/cp24_*.py')
                    for q in root.glob(pattern)} - set(IMPLEMENTATION))
    return {name: sha(root / name) for name in names}


def attempt_dir(root: Path, k: int) -> Path:
    return Path(root) / OUT / f'attempt-{k}'


def section(root: Path, heading: str, until: str) -> str:
    """A ratified section, verbatim, from its heading line to the next named heading."""
    text = (Path(root) / 'capstone_v21.md').read_text()
    start = text.index(heading)
    end = text.index(until, start + len(heading))
    return text[start:end].rstrip() + '\n'


def git(root: Path, *args: str) -> str:
    return subprocess.check_output(['git', *args], cwd=root, text=True).strip()


def build(root: Path, k: int, round_number: int, steering: str) -> dict:
    root = Path(root)
    ids = identities(root)
    reference = require_passing_record(root)
    rd = root / OUT / 'rounds' / f'round-{round_number}'
    design = json.loads((rd / 'design.json').read_text())
    search = json.loads((rd / 'search-ledger.json').read_text())
    ens = json.loads((rd / 'ensembles.json').read_text())
    gate = json.loads((rd / 'gate.json').read_text())
    if not gate['passed']:
        raise ValueError(f'round {round_number}\'s gate did not pass: no freeze (§23.6)')
    for name in ('design.json', 'search-ledger.json', 'ensembles.json', 'gate.json', 'report.md'):
        committed = subprocess.check_output(['git', 'show', f'HEAD:{(rd / name).relative_to(root)}'], cwd=root)
        if committed != (rd / name).read_bytes():
            raise ValueError(f'round {round_number} {name} is not committed at HEAD')
    steer = root / steering
    if subprocess.check_output(['git', 'show', f'HEAD:{steering}'], cwd=root) != steer.read_bytes():
        raise ValueError('the freezing S1 answer is not committed at HEAD')
    adm = json.loads((root / OUT / 'resource-admission.json').read_text())
    m = origin_manifest(root)
    state = ledger().read()
    folds = {}
    for fold, f in search['folds'].items():
        folds[fold] = {'d0': f['d0'], 'batches': f['batches'], 'B': f['B'], 'sampler_seed': f['sampler_seed'],
                       'ensemble': ens['folds'][fold]['members'],
                       'gate': gate['by_fold'][fold]}
    return {
        'schema': 'cp24-protocol-v1', 'checkpoint': 'CP-24', 'attempt': k, 'level': LEVEL[k],
        'anchor': 'capstone_v21.md v21-r11 §23 (ratified 2026-10-05)', 'evidence_class': 'development_post_selection',
        'status': f'frozen after round {round_number}\'s gate passed and S1 froze its design; before any warm-up or evaluation '
                  f'fit of attempt {k}',
        'freezing_steering_answer': {'path': steering, 'sha256': sha(steer)},
        'round': round_number, 'round_files_sha256': {name: sha(rd / name) for name in
                                                      ('design.json', 'search-ledger.json', 'ensembles.json', 'gate.json', 'report.md')},
        'issued_and_inherited_sha256': ids,
        'issued_brief': {'path': 'docs/track-b/evidence/cp-24/issued-brief.md', 'sha256': ISSUED['docs/track-b/evidence/cp-24/issued-brief.md']},
        'representation': {
            'rows': 'one row per delivery day D, whole-day inputs by Europe/Berlin local hour (src/cp15/data.py convention: a '
                    'repeated local hour averages its two observations; a missing local hour stays missing until training-only '
                    'imputation)',
            'always': list(G.ALWAYS), 'optional_groups': list(G.OPTIONAL),
            'encoding': 'cp24.design (module docstring): price curves and price-location statistics as the transform of '
                        '(value - c_d)/s_d; SD statistics as value/s_d; the negative-price count as is; load and weather in '
                        'their units; weekday dummies; the calendar group as 0/1 flags and one-hot day type and month; weather '
                        'missing indicators',
            'target': 'z = (y - c_d)/s_d with the row\'s own origin statistics (s4: §4 level and scale; mad: 168-h median '
                      'and 1.4826 x MAD, floor 1), then z or asinh(z); quantiles inverted with the forecast origin\'s '
                      'statistics',
            'outputs': '24 local-hour slots x 4 Johnson SU parameters (CP-23\'s parameterisation)',
            'preprocessing': 'winsorise continuous inputs at the training rows\' 0.5/99.5% quantiles; training medians; '
                             'StandardScaler on the training rows'},
        'search': {'design': design, 'trials_per_fold': design['trials_per_fold'], 'space': design['space'],
                   'sampler': design['sampler'], 'ranking_metric': 'mean seven-level pinball in EUR/MWh after inversion, pooled '
                   'over the scored hours of the batches each trial ran', 'halving': 'min(4, B_f) most recent batches, then the '
                   'best ceil(N/3) on all B_f', 'failed_fits': search['failed_fits']},
        'folds': folds,
        'gate': {'passed': gate['passed'], 'conditions': gate['conditions'], 'pooled': gate['pooled'],
                 'excluded_training_days': gate['excluded_training_days']},
        'training_recipe': RECIPE,
        'model_constants': {'warm_epochs': M.WARM_EPOCHS, 'patience': M.PATIENCE, 'max_epochs': M.MAX_EPOCHS,
                            'cap_multiple': M.CAP_MULTIPLE, 'grid': list(M.GRID), 'levels': list(M.LEVELS),
                            'lambda_floor': M.LAMBDA_FLOOR, 'delta_floor': M.DELTA_FLOOR, 'adam': M.ADAM,
                            'holdout_share': G.HOLDOUT_SHARE, 'recent_excluded_days': G.RECENT_EXCLUDED_DAYS,
                            'winsor': list(G.WINSOR), 'min_train_days': G.MIN_TRAIN_DAYS, 'min_hold_days': G.MIN_HOLD_DAYS},
        'attempt_fits': {'origins': 'every warm-up and evaluation origin of the frozen manifest with eligible hours',
                         'window': '[max(2019-01-01, D-728), D); no warm-up or evaluation window changes (§23.6); a window '
                                   'day without a frozen weather record fails the fit',
                         'members': 'the fold\'s eight frozen members (configuration, seed), trained fresh at every origin',
                         'data': 'warm-up origins: materialised before the fold\'s first evaluation day; evaluation origins: '
                                 'all data through 2026-04-07'},
        'reference_tolerances': REFERENCE_TOLERANCES, 'reference_trajectory_steps': TRAJECTORY_STEPS,
        'reference_training_loop_epochs': TRAINING_LOOP_EPOCHS, 'reference_record_sha256': sha(root / RECORD),
        'reference_passed': reference['passed'],
        'arms': {'v5': {'role': 'the single eligible candidate', 'central': 'A1_w/3 + B2_w/3 + L-N/12 + L-R/12 + D2/6 = '
                        '(2/3)c_HG + (1/6)L + (1/6)D2', 'layer': 'HG\'s H layer re-estimated on v5\'s own errors'},
                 'D2': {'role': 'study arm, never eligible', 'central': 'the ensemble median', 'layer': 'its own Johnson SU '
                        'quantiles; p50 = central'},
                 'v3+D2': {'role': 'attribution, never eligible', 'central': 'A1_w/3 + B2_w/3 + D2/3', 'layer': 'H on its own errors'},
                 'references': {'HGL': 'v4, the comparator (CP-21 saved)', 'HG': 'v3 (CP-20 saved)', 'A1': 'saved',
                                'B2': 'saved', 'D': 'CP-23\'s DDNN (saved, evidence/cp-23)', 'v3+D': 'CP-23 saved, beside v3+D2'}},
        'weight': '1/6 for DDNN-2 inside LightGBM\'s third; fixed before any fit, never estimated',
        'rule': 'cp24-adoption', 'rule_verbatim': section(root, '### 23.9 Pre-registered rule', '### 23.10 '),
        'arms_verbatim': section(root, '### 23.5 The candidate, the arms and the references', '### 23.6 '),
        'resource_admission': {'verdict': adm['verdict'], 'fixed': adm['fixed']},
        'population': {'keys': 10747, 'fold_counts': [2160, 2159, 2112, 2160, 2156], 'origins': sum(f['date_count'] for f in m['folds'])},
        'budget': {'caps': CAPS, 'effective_caps': effective_caps(state), 'raises': state.get('raises', []),
                   'spent_at_freeze': state['counts'], 'timebox_active_seconds': TIMEBOX_ACTIVE_SECONDS},
        'protocol_change_rule': 'no parameter, configuration, seed, policy, weight, tolerance, diagnostic definition or rule '
                                'change informed by outcomes; a repair after scoring that changes any frozen element is a new '
                                'attempt (§23.6)',
        'python': sys.version.split()[0], 'platform': platform.platform(),
        'dependencies': {name: importlib.metadata.version(name) for name in ('numpy', 'pandas', 'lightgbm', 'scikit-learn', 'pyarrow')},
        'implementation_sha256': {name: sha(root / name) for name in IMPLEMENTATION if (root / name).exists()},
        'frozen_inputs_sha256': {name: sha(root / name) for name in FROZEN},
        'recorded_not_enforced_sha256': recorded_modules(root),
        'written_utc': stamp(),
    }


def job_protocol(root: Path, rest) -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--attempt', type=int, required=True, choices=(1, 2))
    ap.add_argument('--round', type=int, required=True)
    ap.add_argument('--steering', required=True)
    args = ap.parse_args(rest)
    root = Path(root)
    path = attempt_dir(root, args.attempt) / 'protocol.json'
    if path.exists():
        raise ValueError('a protocol exists for this attempt; it is frozen once')
    protocol = build(root, args.attempt, args.round, args.steering)
    atomic(path, protocol)
    print(json.dumps({'attempt': args.attempt, 'protocol_sha256': sha(path)}), flush=True)
    return 0


def check_protocol(root: Path, k: int) -> dict:
    """Attempt k's committed protocol must be at HEAD, its implementation and inputs unchanged, and its
    commit an ancestor of HEAD (§23.6, §23.10)."""
    root = Path(root)
    path = attempt_dir(root, k) / 'protocol.json'
    if not path.exists():
        raise ValueError(f'attempt {k} has no frozen protocol: no warm-up or evaluation fit (§23.6)')
    p = json.loads(path.read_text())
    for name, value in p['implementation_sha256'].items():
        if sha(root / name) != value:
            raise ValueError(f'implementation changed since attempt {k}\'s freeze: {name}')
    for name, value in p['frozen_inputs_sha256'].items():
        if sha(root / name) != value:
            raise ValueError(f'frozen input changed since attempt {k}\'s freeze: {name}')
    rel = str(path.relative_to(root))
    if subprocess.check_output(['git', 'show', f'HEAD:{rel}'], cwd=root) != path.read_bytes():
        raise ValueError(f'attempt {k}\'s protocol is not committed at HEAD')
    commit = git(root, 'log', '-1', '--format=%H', '--', rel)
    if subprocess.run(['git', 'merge-base', '--is-ancestor', commit, 'HEAD'], cwd=root).returncode != 0:
        raise ValueError('the protocol commit is not an ancestor of HEAD')
    p['_protocol_commit'] = commit
    return p
