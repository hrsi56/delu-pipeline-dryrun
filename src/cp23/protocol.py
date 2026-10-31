"""The frozen CP-23 pre-run protocol (capstone v21-r10 §21.4, §21.10 item 5; §14.6 E1-E4 for CP-23).

It is written and committed after 4.6L, the §21.3 checks and 4.6R, and before any main-run DDNN fit
(the per-fold configuration choices, the warm-up and evaluation fits), any admission and any outer
scoring. It binds:

* DDNN's representation, architecture family, configuration set and selection rule;
* the early-stopping rule, the ensemble size, the seeds and the quantile construction;
* the reference tolerances;
* the arms and their composites;
* budget accounting;
* the rule `cp23-adoption`, quoted verbatim from the ratified anchor;
* the definition of every §21.5 diagnostic, so that none is chosen after outcomes are seen.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys

from cp15.data import sha
from . import ddnn as D
from . import scoring as S
from .budget import CAPS, TIMEBOX_ACTIVE_SECONDS, atomic, ledger
from .execution import COMPOSITE_TOLERANCE, CYCLE_OFFSETS, LAYER, NEW_POLICIES, OUT
from .features import CATEGORICAL, MIN_HOLDOUT_ROWS, MIN_ROWS, SELECTION_DAYS
from .inputs import BOUNDARY, ISSUED, cp21_fit_identity, hg_identity, identities, origin_manifest
from .jobs import art, stamp
from .reference import RECORD, REFERENCE_TOLERANCES, TRAINING_LOOP_EPOCHS, TRAJECTORY_STEPS, require_passing_record

IMPLEMENTATION = [
    'src/cp23/__init__.py', 'src/cp23/budget.py', 'src/cp23/inputs.py', 'src/cp23/ddnn.py', 'src/cp23/features.py',
    'src/cp23/member.py', 'src/cp23/audit.py', 'src/cp23/reference.py', 'src/cp23/admission.py', 'src/cp23/execution.py',
    'src/cp23/jobs.py', 'src/cp23/preflight.py', 'src/cp23/scoring.py', 'src/cp23/evaluate.py', 'src/cp23/protocol.py',
    'scripts/cp23_ddnn.py',
    'tests/cp23/test_numpy_only.py', 'tests/cp23/test_ddnn_gradients.py', 'tests/cp23/test_reference_record.py',
    'tests/cp23/test_composites_and_rule.py', 'tests/cp23/test_member_paths.py', 'tests/cp23/torch_reference_checks.py',
    'tests/cp23/torch-reference/pyproject.toml', 'tests/cp23/torch-reference/uv.lock',
    'src/cp15/data.py', 'src/cp15/models.py', 'src/cp15/scoring.py', 'src/cp16/residuals.py', 'src/cp20/components.py',
    'src/cp20/weather.py', 'src/cp20/scoring.py', 'src/cp20/budget.py', 'src/cp21/lgbm.py', 'src/cp21/inputs.py',
    'src/cp21/execution.py', 'src/cp21/jobs.py', 'src/cp21/budget.py', 'src/cp22/execution.py', 'src/cp22/inputs.py',
    'src/cp22/pn.py', 'src/cp22/dl.py', 'src/cp22/scoring.py', 'src/cp22/budget.py', 'src/cp22/jobs.py',
    'src/delu_forecast/features.py', 'src/delu_forecast/folds.py', 'src/delu_forecast/baselines.py',
    'src/delu_forecast/ingest.py']
FROZEN = ['data/snapshot.parquet', 'data/partitions.json', 'uv.lock', 'pyproject.toml', 'reports/cp15/protocol.json',
          'reports/cp15/predictions.parquet', 'reports/v2-causal/input-manifest.json', 'reports/weather-ablation/protocol.json',
          'reports/weather-ablation/lineage.json', 'reports/weather-ablation/weather-features.parquet',
          'reports/weather-ablation/predictions.parquet', 'reports/weather-ablation/run-manifest.json',
          'reports/weather-ablation/criteria.csv', 'reports/weather-ablation/uncertainty.csv',
          'reports/block-challenger/predictions.parquet', 'reports/block-challenger/lineage.json',
          'reports/block-challenger/protocol.json', 'reports/block-challenger/criteria.csv',
          'reports/block-challenger/uncertainty.csv', 'reports/v4-revision/preflight/e1.json',
          'reports/distribution-challenger/licence-admission.md', 'reports/distribution-challenger/resource-admission.md',
          'reports/distribution-challenger/resource-admission.json', 'reports/distribution-challenger/reference-checks.json']
PREFLIGHT = ('input-verification.json', 'weather-regeneration.json', 'e1.json')

#: §21.5's diagnostics, defined before any outcome is seen. Descriptive; they choose nothing.
DIAGNOSTICS = {
    'ddnn_calibration': {
        'coverage_by_level': 'for D, per fold and pooled: the share of actual prices at or below each of the seven emitted '
                             'quantiles (nominal 0.025 ... 0.975), and the 50/80/95% central-interval coverage with mean, '
                             'median and 95th-percentile width',
        'pit_histogram': 'for D: PIT_t = F_ens(z_t), the CDF of the quantile-averaged ensemble -- the inverse in p of the mean '
                         'of the four members\' Johnson SU quantile functions, solved by bisection to |dp| < 1e-12 -- at the '
                         'realised normalised price z_t = (y_t - level_t) / scale_t; 20 equal bins on [0, 1], per fold and '
                         'pooled, with counts and shares (uniform = calibrated)'},
    'extrapolation': {
        'extreme_days': 'evaluation delivery days whose maximum actual hourly price exceeds the maximum eligible actual price '
                        'in that origin\'s training window [max(2019-01-01, D-728), D); and, separately, each fold\'s top 5% of '
                        'represented days by maximum actual hourly price (ceil) -- CP-22\'s definition, unchanged',
        'report': 'per extreme day: the window maximum, the actual maximum, and the maximum central forecast of D, v5, v3+D, '
                  'v4 and v3 and the maximum emitted p975 of D; the share of hours each forecasts above the window maximum; '
                  'beside CP-22\'s committed tree record (reports/v4-revision/investigation/extrapolation.csv: L-P, L-N, L-R, '
                  'PN) on the same days'},
    'peak_and_fold_4': {
        'scopes': 'the 2022-08-15..31 peak (408 hours, 17 days, descriptive) and fold 4 (2025-05-01..07-29), the fold where '
                  'CP-22\'s candidates were decisively worse than v4',
        'report': 'MAE, WIS, bias, 50/80/95% coverage with mean width for every policy; every contrast\'s fold-4 paired '
                  'daily-loss interval'},
    'ensemble_and_seed_stability': {
        'per_seed': 'each seed\'s own median forecast (inverted to EUR/MWh) scored by MAE on the evaluation keys, against the '
                    'ensemble median\'s MAE, per fold and pooled',
        'spread': 'the mean absolute deviation of the four member medians around the ensemble median (EUR/MWh), per fold',
        'straddle': 'the share of hours whose member medians lie on both sides of the actual price',
        'epochs': 'the best-epoch and epochs-run distributions per seed and per configuration'},
    'configuration_by_fold': 'selection.json: the configuration chosen in each fold, each configuration\'s holdout MAE, ties '
                             'and the winner\'s relative margin',
    'fit_cost_and_daily_cycle': {
        'fit_cost': 'every main-run DDNN member fit (selection, warm-up, evaluation): rows, configuration, seed, epochs run, '
                    'best epoch, wall and CPU seconds; per origin (four-seed ensemble) and per fold',
        'daily_cycle': 'cold on the M3 with 4 worker processes at 25 evaluation origins (five per fold at offsets '
                       f'{list(CYCLE_OFFSETS)} days into the window, the next day with eligible hours if one has none): a '
                       'fresh pool loads the data and features through delivery day D; the four seeds\' fits run one per '
                       'worker; the parent forms the ensemble and D, v5\'s central from v4\'s verified cached members, loads '
                       'v5\'s H-layer state persisted that morning, releases, predicts and issues. The fits must equal the main '
                       'run\'s bit for bit and the issued v5 vector the committed one. No LightGBM or LEAR component is '
                       'refitted (§21.8 caps only DDNN fits): v4\'s own cold cycle is CP-21\'s committed measurement '
                       '(reports/block-challenger/daily-cycle.json), shown beside it',
        'role': 'diagnostic, not a criterion (§17.5 D3)'},
}


def section(root: Path, heading: str, until: str) -> str:
    """A ratified section, verbatim, from its heading line to the next named heading."""
    text = (root / 'capstone_v21.md').read_text()
    start = text.index(heading)
    end = text.index(until, start + len(heading))
    return text[start:end].rstrip() + '\n'


def fixtures(root: Path) -> dict:
    """Run the synthetic CP-23 default-suite tests (no research data) and record their outcome."""
    run = subprocess.run([sys.executable, '-m', 'pytest', '-q', '-p', 'no:cacheprovider', 'tests/cp23'], cwd=root,
                         capture_output=True, text=True)
    tail = run.stdout.strip().splitlines()[-1] if run.stdout.strip() else ''
    return {'command': 'python -m pytest -q -p no:cacheprovider tests/cp23', 'exit_code': run.returncode, 'summary': tail,
            'files_sha256': {f'tests/cp23/{p.name}': sha(p) for p in sorted((root / 'tests/cp23').glob('test_*.py'))}}


def build(root: Path) -> dict:
    root = Path(root)
    ids = identities(root)
    reference = require_passing_record(root)
    m = origin_manifest(root)
    pre = {name: json.loads((art() / 'preflight' / name).read_text()) for name in PREFLIGHT}
    e1 = pre['e1.json']
    admission46r = json.loads((root / OUT / 'resource-admission.json').read_text())
    if admission46r['verdict'] != 'PASS':
        raise ValueError('4.6R did not pass: no protocol, no comparison (§21.4)')
    counts = ledger().read()['counts']
    n_inputs = admission46r['origins'][0]['configs']['C1']['n_inputs']
    origins = [{'fold': o['fold'], 'day': o['day'], 'phase': o['phase'], 'n_forecast': o['n_forecast']} for o in e1['origins_detail']]
    fx = fixtures(root)
    if fx['exit_code'] != 0:
        raise ValueError(f'fixtures fail: {fx["summary"]}')
    return {
        'schema': 'cp23-protocol-v1', 'checkpoint': 'CP-23', 'revision': 'cp23-prerun-1',
        'anchor': 'capstone_v21.md v21-r10 §21 (ratified 2026-10-04)', 'evidence_class': 'development_post_selection',
        'status': 'frozen after 4.6L, the §21.3 checks and 4.6R, and before any main-run DDNN fit, admission or outer scoring; '
                  'only the 32 accuracy-blind 4.6R fits on training-only data preceded it',
        'issued_and_inherited_sha256': ids,
        'issued_brief': {'path': 'docs/track-b/evidence/cp-23/issued-brief.md', 'sha256': ISSUED['docs/track-b/evidence/cp-23/issued-brief.md'],
                         'canonical_copy': '.local/artifacts/cp-23/issued-brief.md', 'identical_to_pasted_brief': True},
        'owner_rulings': {
            'publish_rules_pin': 'PUBLISH_RULES 1.3 at SHA-256 5a660864f8b82943174741320c71087a3d0508f707edeb997cb2c1395c1c73b4 '
                                 '(the brief omitted the hash §21.9 says it records; 1.3 has one identity; Owner, 2026-10-04)',
            'torch_pin': 'PyTorch pinned in the separate test-only uv project tests/cp23/torch-reference/ (pyproject.toml, uv.lock), '
                         'not the root uv.lock, whose SHA-256 CP-10\'s lineage and CP-15\'s protocol bind (Owner, 2026-10-04)'},
        'hg_identity': hg_identity(root), 'cp21_fit_identity': cp21_fit_identity(root),
        'entry_gates': {
            '4.6L': {'record': 'reports/distribution-challenger/licence-admission.md',
                     'sha256': sha(root / 'reports/distribution-challenger/licence-admission.md'),
                     'result': 'research use and retention permitted; hosted service and product unresolved (carried to 4.7/4.10)'},
            'section_21_3': {'import_audit': 'tests/cp23/test_numpy_only.py (src/cp23/audit.py)',
                             'gradient_checks': 'tests/cp23/test_ddnn_gradients.py',
                             'reference_record': str(RECORD), 'reference_sha256': sha(root / RECORD),
                             'reference_passed': reference['passed'], 'reference_counts': reference['counts'],
                             'reference_versions': reference['versions']},
            '4.6R': {'record': 'reports/distribution-challenger/resource-admission.md', 'verdict': admission46r['verdict'],
                     'sha256': sha(root / 'reports/distribution-challenger/resource-admission.json')}},
        'ddnn': {
            'implementation': 'src/cp23/ddnn.py (NumPy and the standard library only), sha256 ' + sha(root / 'src/cp23/ddnn.py'),
            'representation': {
                'rows': 'one row per delivery hour, pooled over the 24 local hours (PN\'s and L-P\'s rows)',
                'inputs': 'CP-15\'s 23 normalised LightGBM features (data.lgbm_normalized) with the categoricals '
                          f'{sorted(CATEGORICAL)} one-hot encoded, plus the three frozen GFS columns and three missing '
                          f'indicators: {n_inputs} inputs',
                'categorical_levels': {k: list(v) for k, v in CATEGORICAL.items()},
                'preprocessing': '§15.3 on each fit\'s own training rows: weather training medians (all-null -> 0), one '
                                 'indicator per weather column, then a StandardScaler over every column (zero variance -> 1)',
                'target': '§4: z = (y - level) / scale with each row\'s own origin statistics; quantiles inverted with the '
                          'origin\'s level and scale',
                'information': 'exactly v4\'s (§17.3): no other feature, lag, cross-hour expansion or source'},
            'architecture': {'family': 'feed-forward network, ELU hidden activations, four linear outputs per row',
                             'head': 'Johnson SU: xi = o1, lambda = softplus(o2) + 1e-3, gamma = o3, delta = softplus(o4) + '
                                     '0.05; Y = xi + lambda*sinh((Z - gamma)/delta), Z ~ N(0,1)',
                             'lambda_floor': D.LAMBDA_FLOOR, 'delta_floor': D.DELTA_FLOOR,
                             'initialisation': 'Glorot uniform weights, zero biases, output biases at lambda = delta = 1, '
                                               'PCG64(seed)'},
            'configurations': [{**c, 'hidden': list(c['hidden']), 'n_parameters': D.n_parameters(n_inputs, c['hidden'])}
                               for c in D.CONFIGS],
            'configuration_order': 'strictly increasing parameter count; the smaller configuration is the earlier one',
            'selection': {'when': 'once per fold, before its first origin D0 (its genuine warm-up start), from data before D0 '
                                  'only; fixed for every origin of the fold',
                          'rows': f'train [max(2019-01-01, D0-728), D0-{2 * SELECTION_DAYS}); early stopping '
                                  f'[D0-{2 * SELECTION_DAYS}, D0-{SELECTION_DAYS}); holdout [D0-{SELECTION_DAYS}, D0)',
                          'criterion': 'the lowest holdout MAE (EUR/MWh) of each configuration\'s four-seed ensemble median',
                          'tie': 'an exact tie goes to the smaller configuration', 'fits': len(m['folds']) * len(D.CONFIGS) * len(D.SEEDS)},
            'training': {**D.TRAINING, 'loss': 'mean Johnson SU negative log-likelihood + l2 * sum(W^2) over weight matrices',
                         'optimizer': 'Adam in PyTorch\'s update order (no weight decay, no amsgrad)',
                         'batches': 'a fresh PCG64 permutation each epoch, after initialisation, from the member\'s seed'},
            'early_stopping': {'rows': 'the window\'s last 28 calendar delivery days [D-28, D), training-only inner validation',
                               'criterion': 'validation negative log-likelihood without the penalty, after every epoch',
                               'kept': 'the weights of the best epoch (strict improvement)', 'patience': D.TRAINING['patience'],
                               'max_epochs': D.TRAINING['max_epochs']},
            'history': {'window': '[max(2019-01-01, D-728 calendar days), D), eligible rows; fresh fit at every origin',
                        'minimum_training_rows': MIN_ROWS, 'minimum_early_stopping_rows': MIN_HOLDOUT_ROWS},
            'ensemble': {'seeds': list(D.SEEDS), 'size': len(D.SEEDS), 'combination': 'quantile averaging at each level'},
            'emission': {'levels': list(D.LEVELS), 'quantile_function': 'xi + lambda*sinh((Phi^-1(q) - gamma)/delta) per member',
                         'p50_and_central': 'the ensemble median (the average of the members\' medians); D\'s emitted p50 '
                                            'equals its central forecast by construction',
                         'rearrangement': 'sort a row\'s seven quantiles if any cross; every case recorded (averaging '
                                          'ordered quantile functions cannot cross)'},
            'cache': '.local/artifacts/cp-23/fits/<fold>/<day>/DDNN.json bound to cp23_input_fingerprint, cp23_protocol_sha256 '
                     '(this file) and the weather design; a stale or wrong entry is refitted and charged, never reused'},
        'reference_tolerances': REFERENCE_TOLERANCES, 'reference_trajectory_steps': TRAJECTORY_STEPS,
        'reference_training_loop_epochs': TRAINING_LOOP_EPOCHS,
        'policies': {
            'v5': {'role': 'the single eligible candidate', 'central': '2*c_v4/3 + D/3, c_v4 = A1/3 + B2/3 + L-N/6 + L-R/6',
                   'layer': 'H (HG\'s, on v5\'s own errors)'},
            'v3+D': {'role': 'attribution: DDNN as v3\'s third member, never eligible', 'central': 'A1/3 + B2/3 + D/3',
                     'layer': 'H (on its own errors)'},
            'D': {'role': 'study arm: DDNN alone, never eligible', 'central': 'the ensemble median',
                  'layer': 'its own Johnson SU quantiles and p50'},
            'HGL': {'role': 'comparator v4 (CP-21\'s saved vectors)'}, 'HG': {'role': 'reference v3 (CP-20\'s saved vectors)'},
            'A1': {'role': 'reference (saved)'}, 'B2': {'role': 'reference (saved)'},
            'B0, B1, B3': {'role': 'the B0 normaliser and the saved section-8 comparators, metric only'}},
        'composite_parity_tolerance_eur_mwh': COMPOSITE_TOLERANCE, 'member_weight': '1/3, fixed', 'layer_by_policy': LAYER,
        'h_layer': 'section 14.2 via cp16.residuals.SharedResidualState (V2-H), one state per policy, central passed twice',
        'sequence': 'protocol freeze (committed); the five folds\' configuration choices (committed); warm-up fits; '
                    'training-only admission of v5, v3+D and D (committed); evaluation fits; comparison replay; vectors '
                    'committed; one scoring pass and cp23-adoption',
        'fixtures': fx,
        'population': {'keys': 10747, 'fold_counts': [2160, 2159, 2112, 2160, 2156], 'represented_days': 448,
                       'origins': sum(f['date_count'] for f in m['folds']), 'origins_with_eligible_hours': e1['origins_with_eligible_hours'],
                       'origins_without_eligible_hours': e1['origins_without_eligible_hours'], 'admission_dates': 35,
                       'warmup_starts': {f['fold']: f['warmup_start'] for f in m['folds']}, 'origin_table': origins,
                       'keys_sha256': hashlib.sha256('\n'.join(k for f in m['folds'] for k in f['original_target_keys']).encode()).hexdigest()},
        'chronology': {'origin': 'D-1 11:00 UTC (12:00 fixed UTC+01:00)', 'delivery_calendar': 'Europe/Berlin',
                       'errors': 'complete released delivery days <= D-2, each consumed once', 'boundary': str(BOUNDARY)},
        'weather': {'source': 'CP-20 retained decoded grids through the frozen conversion; no retrieval',
                    'regenerated_from_grids': pre['weather-regeneration.json']},
        'cache_identities': {'hg_components': '.local/artifacts/cp-20/hg-components (every entry verified)',
                             'cp21_fits': '.local/artifacts/cp-21/fits (CP-21 identity recomputed from evidence/cp-21 objects; every '
                                          'central and v4 composite equal to CP-21\'s committed lineage hash)',
                             'verification': pre['input-verification.json']['tally']},
        'scores': {'point': 'emitted-p50 MAE', 'interval': 'seven-quantile WIS', 'primary': 'S_MAE, S_WIS equal-fold ratios to B0',
                   'secondary': 'pooled observation-weighted scores', 'saved_references': list(S.SAVED), 'display': S.DISPLAY},
        'uncertainty': {'seed': S.BOOTSTRAP_SEED, 'replicates': S.BOOTSTRAP_REPLICATES, 'block_days': S.BLOCK_DAYS,
                        'index_set': 'cp20.scoring._indices (sha256 ' + S.CP20_INDEX_SHA256 + ')',
                        'contrasts': [f'{c}-{b} ({r})' for c, b, r in S.contrasts()],
                        'reading': 'observed joint improvement iff upper dS_WIS < 0 and upper dS_MAE <= 0; observed joint '
                                   'worsening iff lower dS_WIS > 0 and lower dS_MAE >= 0; otherwise no demonstrated joint preference',
                        'per_fold': 'paired difference of each fold\'s mean daily loss, the same 7-day block index set'},
        'rules': S.RULES, 'rules_application': S.APPLICATION,
        'rule_verbatim': section(root, '### 21.6 Pre-registered rule', '### 21.7 '),
        'arms_verbatim': section(root, '### 21.2 DDNN, the candidate and the arms', '### 21.3 '),
        'diagnostics': DIAGNOSTICS,
        'support_rule': f'at least {S.SUPPORT_DATES} represented dates per fold for block/hour statements',
        'caps': CAPS, 'timebox_active_seconds': TIMEBOX_ACTIVE_SECONDS,
        'E1': {'origins': e1['origins'], 'with_eligible_hours': e1['origins_with_eligible_hours'], 'fits': e1['fits'],
               'main_fit_cap': CAPS['main_ddnn_fits'], 'total_fit_cap': CAPS['ddnn_fits'],
               'spent_before_freeze': {k: v for k, v in counts.items() if k.startswith('ddnn')},
               'planned_non_main_fits': admission46r['projection']['planned_non_main_fits'],
               'policy_days': admission46r['projection']['policy_days_planned'],
               'passes': {'reference': 'scoring + independent review = 2 of 3', 'bootstrap': 'scoring + independent review = 2 of 3'},
               'machine_hours_projection': admission46r['projection']['projected_machine_hours'],
               'sufficiency_problems': e1['sufficiency_problems'], 'feasible': not e1['sufficiency_problems']},
        'E2': {'reproduction': 'reports/distribution-challenger/reproduce.md', 'output_schema': {
            'predictions.parquet': 'fold, policy, timestamp_utc, delivery_date, origin_utc, y_true, central, scale, level, '
                                   'evidence_class, p025..p975 (v5, v3+D, D: 32,241 rows)',
            'members.parquet': 'fold, timestamp_utc, delivery_date, config, D, A1, B2, L-N, L-R, HG, HGL, c_v5, c_v3+D (10,747 rows)'}},
        'E3': {'ledger': '.local/artifacts/cp-23/ledger/budget.json (cumulative, never reset; counters reserved before use)',
               'monitor': 'scripts/cp23_ddnn.py monitor: wall x declared workers, process-tree and aggregate RSS, added disk, '
                          'duplicate/5th-worker refusal, markers',
               'preflight_disk_free_bytes': shutil.disk_usage(str(art())).free},
        'E4': {'lead': 'CP-23 Track B Engineering Lead (this session)',
               'reviewer': 'one fresh independent Integration Critic on a clean detached checkout of the exact final candidate, '
                           'launched with scripts/gauntlet.py critic-open, critic-brief and critic-close',
               'final_candidate_sha': 'recorded at review'},
        'protocol_change_rule': 'no parameter, configuration, seed, policy, weight, tolerance, diagnostic definition or rule '
                                'change informed by outcomes; implementation defects are corrected with explicit records and '
                                'reruns, never hidden tuning',
        'python': sys.version.split()[0], 'platform': platform.platform(),
        'dependencies': {name: importlib.metadata.version(name) for name in ('numpy', 'pandas', 'lightgbm', 'scikit-learn', 'pyarrow')},
        'implementation_sha256': {name: sha(root / name) for name in IMPLEMENTATION},
        'frozen_inputs_sha256': {name: sha(root / name) for name in FROZEN},
        'preflight_sha256': {f'{OUT}/preflight/{name}': None for name in PREFLIGHT},
        'written_utc': stamp(),
    }


def job_protocol(root: Path, rest) -> int:
    root = Path(root)
    out = root / OUT
    if (out / 'protocol.json').exists():
        raise ValueError('a protocol exists; it is frozen once')
    dest = out / 'preflight'
    dest.mkdir(parents=True, exist_ok=True)
    for name in PREFLIGHT:
        shutil.copyfile(art() / 'preflight' / name, dest / name)
    protocol = build(root)
    protocol['preflight_sha256'] = {f'{OUT}/preflight/{name}': sha(dest / name) for name in PREFLIGHT}
    atomic(out / 'protocol.json', protocol)
    print(json.dumps({'protocol_sha256': sha(out / 'protocol.json'), 'fixtures': protocol['fixtures']['summary'],
                      'E1': protocol['E1']['fits']}), flush=True)
    return 0
