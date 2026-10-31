"""The frozen CP-21 pre-run protocol (capstone v21-r6 §17.10 item 1; §14.6 E1-E4 for CP-21).

Written and committed before any main fit, admission or outer scoring. It binds the arms, blend
weights, block map, feature list and missing rule; the capacity grid, inner split and tie rule;
fixtures, seeds, the key/origin manifest and cache identities; budget accounting; and the §17.6
rule text, quoted verbatim from the ratified anchor.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import shutil
import sys

from cp15.data import sha
from .budget import CAPS, TIMEBOX_ACTIVE_SECONDS, atomic, ledger
from .inputs import BOUNDARY, ISSUED, hg_identity, identities, origin_manifest
from .jobs import art, stamp
from .lgbm import ARMS, BLOCKS, FITS_PER_MODEL, GRID, INNER_DAYS, MIN_TRAIN_ROWS, MIN_VALIDATION_ROWS
from .scoring import ADOPTION_RULE, BLOCK_DAYS, BOOTSTRAP_REPLICATES, BOOTSTRAP_SEED, CONTRASTS, CP20_INDEX_SHA256, SUPPORT_DATES

OUT = Path('reports/block-challenger')
IMPLEMENTATION = [
    'src/cp21/__init__.py', 'src/cp21/budget.py', 'src/cp21/inputs.py', 'src/cp21/lgbm.py', 'src/cp21/execution.py',
    'src/cp21/jobs.py', 'src/cp21/preflight.py', 'src/cp21/scoring.py', 'src/cp21/protocol.py', 'scripts/cp21_blocks.py',
    'src/cp15/data.py', 'src/cp15/models.py', 'src/cp15/scoring.py', 'src/cp16/residuals.py', 'src/cp20/components.py',
    'src/cp20/weather.py', 'src/cp20/scoring.py', 'src/cp20/budget.py', 'src/delu_forecast/features.py',
    'src/delu_forecast/folds.py', 'src/delu_forecast/baselines.py', 'src/delu_forecast/ingest.py']
FROZEN = ['data/snapshot.parquet', 'data/partitions.json', 'reports/cp15/protocol.json', 'reports/cp15/predictions.parquet',
          'reports/v2-causal/input-manifest.json', 'reports/weather-ablation/protocol.json',
          'reports/weather-ablation/lineage.json', 'reports/weather-ablation/weather-features.parquet',
          'reports/weather-ablation/predictions.parquet', 'reports/weather-ablation/run-manifest.json']
PREFLIGHT = ('input-verification.json', 'weather-regeneration.json', 'benchmark.json', 'thread-determinism.json', 'e1.json')


def section(root: Path, heading: str, until: str) -> str:
    """A ratified section, verbatim, from its heading line to the next named heading."""
    text = (root / 'capstone_v21.md').read_text()
    start = text.index(heading)
    end = text.index(until, start + len(heading))
    return text[start:end].rstrip() + '\n'


def build(root: Path) -> dict:
    root = Path(root)
    ids = identities(root)
    m = origin_manifest(root)
    p15 = json.loads((root / 'reports/cp15/protocol.json').read_text())
    pre = {name: json.loads((art() / 'preflight' / name).read_text()) for name in PREFLIGHT}
    e1 = pre['e1.json']
    bench = pre['benchmark.json']
    threads = pre['thread-determinism.json']
    counts = ledger().read()['counts']
    per_origin = bench['per_origin_single_thread_wall_seconds_all_three_arms']
    origins = [{'fold': o['fold'], 'day': o['day'], 'phase': o['phase'], 'n_forecast': o['n_forecast']} for o in e1['origins_detail']]
    main_fits = e1['main_fits']
    estimate_machine_hours = {
        'main_fits': round(per_origin * e1['origins_with_eligible_hours'] * 1.5 / 3600, 2),
        'controls_daily_cycle_review_tests_scoring': 12.0,
        'basis': 'benchmark single-thread wall per origin (all three LightGBM arms) x origins with eligible hours x 1.5 '
                 '(process start, data load, scheduling); charged as wall-clock x declared workers'}
    return {
        'schema': 'cp21-protocol-v1', 'checkpoint': 'CP-21', 'revision': 'cp21-prerun-1',
        'anchor': 'capstone_v21.md v21-r6 §17 (ratified 2026-09-29)', 'evidence_class': 'development_post_selection',
        'status': 'frozen before any main fit, admission or outer scoring; only accuracy-blind timing and thread-determinism '
                  'fits on training-only data preceded it (reports/block-challenger/preflight/)',
        'issued_and_inherited_sha256': ids,
        'issued_brief': {'path': 'docs/track-b/evidence/cp-21/issued-brief.md', 'sha256': ISSUED['docs/track-b/evidence/cp-21/issued-brief.md'],
                         'canonical_copy': '.local/artifacts/cp-21/issued-brief.md', 'identical_to_pasted_brief': True},
        'hg_identity': hg_identity(root),
        'arms': {
            'HG': 'comparator (v3), saved: CP-20 HG exactly; its A1_w/B2_w reused only from the identity-verified CP-20 cache',
            'L-P': 'study arm: one pooled LightGBM for all 24 local hours, raw target, HG information, own H layer; never adoption-eligible',
            'L-R': 'study arm: three block LightGBM models (night/solar/shoulder), raw target, own H layer; never adoption-eligible',
            'L-N': 'study arm: L-R with the section-4 normalised target and CP-15 price-valued centre/scale features; never adoption-eligible',
            'HGL': 'sole adoption candidate: c = A1_w/3 + B2_w/3 + L-N/6 + L-R/6 = (2/3)c_HG + (1/3)mean(L-N, L-R); HG H layer re-estimated on HGL own issued errors'},
        'saved_references': ['B0', 'B1', 'B2', 'B3', 'A1', 'H0', 'HG'], 'scored_policies': 11, 'new_policies': 4,
        'blend': {'weights': {'A1_w': '1/3', 'B2_w': '1/3', 'L-N': '1/6', 'L-R': '1/6'},
                  'float_expression': 'A1/3 + B2/3 + LN/6 + LR/6, float64, left to right (cp21.lgbm.hgl_central)',
                  'parity_tolerance': 'max |c_HGL - (2/3)c_HG - (1/3)mean(L-N,L-R)| <= 1e-9 EUR/MWh', 'tuned': False},
        'blocks': {name: list(hours) for name, hours in BLOCKS.items()},
        'block_rules': 'Europe/Berlin local hours; exhaustive and disjoint; on a 25-hour day both canonical local-hour-2 '
                       'observations enter the night block (and L-P rows); a missing spring hour stays absent',
        'features': {'lgbm_raw': p15['lgbm']['features'], 'lgbm_normalized_center_and_scale': p15['normalization']['lgbm_center_and_scale'],
                     'lgbm_normalized_scale_only': p15['normalization']['lgbm_scale_only'],
                     'weather': ['wx_wind10_mean', 'wx_wind100_mean', 'wx_dswrf_mean'],
                     'weather_source': 'CP-20 frozen same-target-local-hour design (repeated autumn hour averages its two canonical vectors)',
                     'missing_indicators': ['wx_wind10_mean_missing', 'wx_wind100_mean_missing', 'wx_dswrf_mean_missing'],
                     'order': '23 CP-15 features, then 3 imputed weather columns, then 3 indicators (29 columns)'},
        'missing_rule': 'section 15.3 on each fit own training partition (each inner-training split and each final window): '
                        'per-column training median, all-null column -> 0, one missing indicator per column; no interpolation, '
                        'forward fill or zero shortcut; LightGBM native missing routing never used for weather. The section 15.3 '
                        'StandardScaler is a strictly increasing per-column map to which tree splits are invariant; it is not '
                        'applied to tree inputs. E1: no training window of any origin contains a missing weather row.',
        'lgbm': {'inherited_parameters': p15['lgbm']['parameters'], 'objective': 'quantile', 'alpha': 0.5, 'seed': p15['seed'],
                 'changed': 'capacity only (n_estimators, num_leaves) from the frozen grid',
                 'execution_threads': {'n_jobs': 1, 'processes': 'up to 4 (threads x processes <= 4)',
                                       'evidence': 'reports/block-challenger/preflight/thread-determinism.json',
                                       'finding': 'with deterministic row-wise training, n_jobs 1 and 4 give identical trees and '
                                                  'bitwise-identical forecasts in all {} real-data cases; the model string differs '
                                                  'only in the recorded [num_threads] parameter line'.format(len(threads['cases'])),
                                       'all_identical': threads['all_thread_counts_identical']},
                 'model_hash': 'SHA-256 of the model string through its end-of-trees marker'},
        'capacity_grid': list(GRID), 'grid_order': 'strictly increasing trees x leaves; the smaller configuration is the earlier one',
        'selection': {'inner_split': f'train on the origin window minus its last {INNER_DAYS} calendar delivery days; validate on those days',
                      'metric': 'validation MAE in EUR/MWh (L-N inverted with each row own origin level/scale first)',
                      'tie': 'exact tie -> the smaller configuration', 'refit': 'selected configuration refitted on the whole window',
                      'per_model': 'every block model and L-P pooled model, at every origin', 'fits_per_model': FITS_PER_MODEL,
                      'outcomes': 'training data only; outer outcomes never enter selection'},
        'history': {'window': '[max(2019-01-01, D-728 calendar days), D), eligible rows, subject to each origin availability filter',
                    'minimum_window_rows': MIN_TRAIN_ROWS, 'minimum_inner_rows': MIN_TRAIN_ROWS,
                    'minimum_validation_rows': MIN_VALIDATION_ROWS,
                    'rule': 'L-P keeps the inherited 8,760-row minimum; blocks 365 x block hours; the inner-training split must '
                            'meet the same minimum; validation needs 14 rows per block hour'},
        'cadence': 'fresh daily fits with selection at every origin; a cached fit is reused only with verified input, protocol and origin identity; a cache miss is a budgeted fit',
        'h_layer': {'recipe': 'section 14.2 H: price-only A1 scale s_t, 28 complete released days <= D-2, w_h=n_h/(n_h+56), '
                              'w_h=0 below 14 distinct days, seven linear empirical quantiles, section 14.2 failure rule',
                    'code_path': 'cp16.residuals.SharedResidualState (V2-H output) for HG, HGL, L-P, L-R and L-N; one state per '
                                 'arm; one central vector passed as both blend inputs (c/2 + c/2 == c exactly, asserted)',
                    'residuals_shared': False},
        'population': {'keys': 10747, 'fold_counts': [2160, 2159, 2112, 2160, 2156], 'represented_days': 448,
                       'origins': sum(f['date_count'] for f in m['folds']), 'origins_with_eligible_hours': e1['origins_with_eligible_hours'],
                       'origins_without_eligible_hours': e1['origins_without_eligible_hours'],
                       'admission_dates': 35, 'warmup_starts': {f['fold']: f['warmup_start'] for f in m['folds']},
                       'new_policy_target_rows': 42988, 'scored_rows': 118217, 'origin_table': origins,
                       'keys_sha256': hashlib.sha256('\n'.join(k for f in m['folds'] for k in f['original_target_keys']).encode()).hexdigest()},
        'chronology': {'origin': 'D-1 11:00 UTC (12:00 fixed UTC+01:00)', 'delivery_calendar': 'Europe/Berlin',
                       'errors': 'complete released delivery days <= D-2, each consumed once', 'boundary': str(BOUNDARY),
                       'staging': 'warm-up/admission fits on data materialised before each fold first evaluation day; training-only '
                                  'admission replay; admission freeze committed; then evaluation fits and the comparison replay'},
        'weather': {'source': 'CP-20 retained decoded grids through the frozen conversion; no retrieval',
                    'regenerated_from_grids': pre['weather-regeneration.json'], 'design_sha256': pre['weather-regeneration.json']['committed_design_sha256']},
        'cache_identities': {'hg_components': '.local/artifacts/cp-20/hg-components (CP-20 cache; every entry verified)',
                             'hg_verification': pre['input-verification.json']['hg_cache'],
                             'cp21_fits': '.local/artifacts/cp-21/fits/<fold>/<day>/<arm>.json bound to cp21_input_fingerprint, '
                                          'cp21_protocol_sha256 (this file) and the weather design'},
        'scores': {'point': 'emitted-p50 MAE', 'interval': 'seven-quantile WIS (alpha/2 interval weights, median weight 1/2, /3.5)',
                   'primary': 'S_MAE, S_WIS: equal-fold ratios to B0', 'secondary': 'pooled observation-weighted scores'},
        'uncertainty': {'seed': BOOTSTRAP_SEED, 'replicates': BOOTSTRAP_REPLICATES, 'block_days': BLOCK_DAYS,
                        'index_sets_per_pass': 1, 'generator': 'cp20.scoring._indices (CP-20 index set; sha256 ' + CP20_INDEX_SHA256 + ')',
                        'contrasts': [f'{c}-{b}' for c, b in CONTRASTS], 'primary': 'HGL-HG',
                        'ratio': 'R_b = S_policy,b / S_comparator,b - 1 per replicate; 2.5/97.5 percentiles; point = full-sample ratio - 1',
                        'per_fold': 'paired difference of each fold mean daily loss, same index set within the fold',
                        'stored': 'every replicate difference and ratio (reports/block-challenger/replicates.parquet)',
                        'block_split_reading': 'L-R - L-P: observed joint improvement iff upper dS_WIS < 0 and upper dS_MAE <= 0; observed '
                                               'joint worsening iff lower dS_WIS > 0 and lower dS_MAE >= 0; otherwise no demonstrated joint preference'},
        'diagnostics': {'support_rule': f'at least {SUPPORT_DATES} represented dates per fold for block/hour statements',
                        'stress': 'fold 3, 2022-07-01..09-28 (2,112 hours / 88 days); peak 2022-08-15..31 (408 hours) descriptive',
                        'section8': 'all six original section-8 diagnostics for every new arm (and HG), saved B0-B3 comparators'},
        'adoption_rule': ADOPTION_RULE,
        'adoption_rule_verbatim': section(root, '### 17.6 Pre-registered adoption rule', '### 17.7 '),
        'block_models_verbatim': section(root, '### 17.3 Block models', '### 17.4 '),
        'caps': CAPS, 'timebox_active_seconds': TIMEBOX_ACTIVE_SECONDS,
        'E1': {'origins': e1['origins'], 'with_eligible_hours': e1['origins_with_eligible_hours'], 'main_fits': main_fits,
               'main_fit_cap': CAPS['main_lgbm_fits'], 'total_fit_cap': CAPS['lgbm_fits'],
               'planned_non_main_fits': {'benchmark_and_determinism_spent': counts.get('lgbm_fits_benchmark', 0),
                                         'controls': 1400, 'daily_cycle': 750, 'review': 600, 'repair_reserve': 1500},
               'policy_days': {'main_replay': e1['policy_days_main_replay'], 'hg_parity': e1['policy_days_hg_parity_replay'],
                               'daily_cycle': 25, 'controls': 200, 'review': 800, 'cap': CAPS['policy_days']},
               'hg_components': {'daily_cycle': 50, 'controls': 16, 'review': 10, 'cap_component_days': CAPS['component_attempts'],
                                 'lasso_per_component_day': 120, 'cap_lasso': CAPS['primitive_fits']},
               'passes': {'reference': 'scoring 1 + review 1 of 3', 'bootstrap': 'scoring 1 + review 1 of 3'},
               'machine_hours_estimate': estimate_machine_hours, 'sufficiency_problems': e1['sufficiency_problems'],
               'feasible': True},
        'E2': {'implementation_note': 'implementation_sha256 below; inherited modules imported unchanged',
               'fixtures': {'quantiles': p15['quantiles']['fixtures'], 'blend_parity_tolerance_eur_mwh': 1e-9,
                            'hg_parity': 'bitwise on all 10,747 keys', 'daily_cycle_components': 'bitwise against the CP-20 cache',
                            'daily_cycle_block_fits': 'bitwise against the main-run fits', 'thread_count': 'identical trees and forecasts'},
               'output_schema': {'predictions.parquet': 'fold, policy, timestamp_utc, delivery_date, origin_utc, y_true, central, scale, '
                                                        'level, evidence_class, p025..p975 (four new arms, 42,988 rows)'},
               'reproduction': 'reports/block-challenger/reproduce.md'},
        'E3': {'ledger': '.local/artifacts/cp-21/ledger/budget.json (cumulative, never reset; counters reserved before use)',
               'monitor': 'scripts/cp21_blocks.py monitor: wall x declared workers, process-tree and aggregate RSS, added disk, '
                          'duplicate/5th-worker refusal, completion markers',
               'resumability': 'fits and states are written atomically, so a stopped job resumes from its cache',
               'preflight_disk_free_bytes': shutil.disk_usage(str(art())).free},
        'E4': {'lead': 'CP-21 Track B Engineering Lead (this session)', 'reviewer': 'one fresh independent Integration Critic, '
               'launched by the Lead on a clean detached checkout of the exact final candidate under .local/worktrees/cp-21/critic',
               'final_candidate_sha': 'recorded at review'},
        'protocol_change_rule': 'no parameter, arm, grid, weight or rule change informed by outcomes; implementation defects are '
                                'corrected with explicit records and reruns, never hidden tuning',
        'python': sys.version.split()[0], 'platform': platform.platform(),
        'dependencies': {name: importlib.metadata.version(name) for name in ('lightgbm', 'numpy', 'pandas', 'scikit-learn', 'pyarrow')},
        'implementation_sha256': {name: sha(root / name) for name in IMPLEMENTATION},
        'frozen_inputs_sha256': {name: sha(root / name) for name in FROZEN},
        'preflight_sha256': {f'reports/block-challenger/preflight/{name}': None for name in PREFLIGHT},
        'written_utc': stamp(),
    }


def job_protocol(root: Path, rest) -> int:
    root = Path(root)
    out = root / OUT
    dest = out / 'preflight'
    dest.mkdir(parents=True, exist_ok=True)
    for name in PREFLIGHT:
        shutil.copyfile(art() / 'preflight' / name, dest / name)
    protocol = build(root)
    protocol['preflight_sha256'] = {f'reports/block-challenger/preflight/{name}': sha(dest / name) for name in PREFLIGHT}
    atomic(out / 'protocol.json', protocol)
    print(json.dumps({'protocol_sha256': sha(out / 'protocol.json'), 'E1': {k: v for k, v in protocol['E1'].items()}}), flush=True)
    return 0
