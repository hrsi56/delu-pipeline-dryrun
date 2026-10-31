"""The frozen CP-22 pre-run protocol (capstone v21-r9 §20.10 item 1; §14.6 E1-E4 for CP-22).

Written and committed before any main fit, admission or outer scoring. It binds the arms and
members; the grid, inner split and tie rule; DL's half-life, gamma, clipping and rearrangement and
the fast component's half-life and share, with their fixtures; seeds, the manifest and cache
identities; budget accounting; the three §20.6 rules, quoted verbatim from the ratified anchor;
and the definitions of every §20.5 diagnostic, so that none is chosen after outcomes are seen.
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
from cp21.lgbm import GRID, INNER_DAYS, MIN_TRAIN_ROWS, MIN_VALIDATION_ROWS, POOLED
from .budget import CAPS, TIMEBOX_ACTIVE_SECONDS, atomic, ledger
from .dl import ALPHAS, BUFFER_DAYS, FAST_HALF_LIFE_DAYS, FAST_SHARE, GAMMA, HALF_LIFE_DAYS, MIN_HOUR_DAYS, SHRINK, VARIANTS, clip_bounds
from .execution import CYCLE_OFFSETS, FIXED_NEW, LAYER, OUT, W_POLICIES
from .inputs import BOUNDARY, ISSUED, cp21_fit_identity, hg_identity, identities, origin_manifest
from .jobs import art, stamp
from .pn import AVERAGING_TOLERANCE, FITS_PER_ORIGIN
from . import scoring as S

IMPLEMENTATION = [
    'src/cp22/__init__.py', 'src/cp22/budget.py', 'src/cp22/inputs.py', 'src/cp22/pn.py', 'src/cp22/dl.py',
    'src/cp22/execution.py', 'src/cp22/jobs.py', 'src/cp22/preflight.py', 'src/cp22/scoring.py', 'src/cp22/evaluate.py',
    'src/cp22/protocol.py', 'scripts/cp22_revision.py',
    'tests/cp22/test_dynamic_layer.py', 'tests/cp22/test_pn_and_composites.py', 'tests/cp22/test_scoring_rules.py',
    'src/cp15/data.py', 'src/cp15/models.py', 'src/cp15/scoring.py', 'src/cp16/residuals.py', 'src/cp20/components.py',
    'src/cp20/weather.py', 'src/cp20/scoring.py', 'src/cp20/budget.py', 'src/cp21/lgbm.py', 'src/cp21/inputs.py',
    'src/cp21/execution.py', 'src/cp21/jobs.py', 'src/cp21/budget.py', 'src/delu_forecast/features.py',
    'src/delu_forecast/folds.py', 'src/delu_forecast/baselines.py', 'src/delu_forecast/ingest.py']
FROZEN = ['data/snapshot.parquet', 'data/partitions.json', 'reports/cp15/protocol.json', 'reports/cp15/predictions.parquet',
          'reports/v2-causal/input-manifest.json', 'reports/weather-ablation/protocol.json',
          'reports/weather-ablation/lineage.json', 'reports/weather-ablation/weather-features.parquet',
          'reports/weather-ablation/predictions.parquet', 'reports/weather-ablation/run-manifest.json',
          'reports/weather-ablation/criteria.csv', 'reports/weather-ablation/uncertainty.csv',
          'reports/block-challenger/predictions.parquet', 'reports/block-challenger/lineage.json',
          'reports/block-challenger/protocol.json', 'reports/block-challenger/fits.parquet',
          'reports/block-challenger/criteria.csv', 'reports/block-challenger/uncertainty.csv']
PREFLIGHT = ('input-verification.json', 'weather-regeneration.json', 'benchmark.json', 'e1.json')

#: The definitions of §20.5's diagnostics and the Owner's investigation, frozen before scoring.
INVESTIGATION = {
    'ladder_decomposition': {
        'steps': [['HGL', 'M', 'the split removed'], ['M', 'A-PN-sel', 'the raw half dropped'],
                  ['A-PN-sel', 'R', 'averaging over capacities instead of daily selection'], ['HG', 'R', 'v3 to R'],
                  ['HG', 'HGL', 'v3 to v4 (CP-21)']],
        'identity': 'S(v4) - S(v3) = [S(v4) - S(M)] + [S(M) - S(A-PN-sel)] + [S(A-PN-sel) - S(R)] + [S(R) - S(v3)], '
                    'for S_MAE and S_WIS, from the final scores; each bracket with its paired interval where it is a contrast',
        'member_weight_curve': {'weights': [round(0.05 * i, 2) for i in range(21)],
                                'members': {'v4 member mean(L-N, L-R)': ['L-N', 'L-R'], 'PN-avg': ['PN-avg'], 'PN-sel': ['PN-sel'],
                                            'mean(PN-sel, L-P)': ['PN-sel', 'L-P'], 'L-P': ['L-P'], 'L-N': ['L-N']},
                                'curve': '(1-w)*c_HG + w*member, central forecast before any interval layer; equal-fold mean '
                                         'of per-fold central MAE divided by B0\'s per-fold MAE',
                                'label': 'oracle, not selectable'}},
    'capacity_selection_stability': {
        'models': 'PN (this checkpoint); CP-21 L-P (pooled), L-R and L-N (each block) from reports/block-challenger/fits.parquet',
        'flip_rate': 'share of consecutive origins with forecasts, within a fold and a model, whose selected configuration differs',
        'winner_margin': '(runner-up inner validation MAE - winner MAE) / winner MAE, per origin; median and quartiles',
        'distribution': 'share of origins selecting each of G1-G4', 'scope': 'evaluation origins; warm-up shown separately'},
    'extrapolation': {
        'extreme_days': 'evaluation delivery days whose maximum actual hourly price exceeds the maximum eligible actual price '
                        'in that origin\'s training window [max(2019-01-01, D-728), D); and, separately, each fold\'s top 5% '
                        'of represented days by maximum actual hourly price (ceil)',
        'report': 'per extreme day: window maximum, actual maximum, and each member\'s and policy\'s maximum central forecast; '
                  'share of hours forecast above the window maximum'},
    'coverage_breakdown': {
        'hour_and_block': 'from the per-fold hour/block diagnostics (56-date support rule)',
        'regime': 'three regimes by the issued price-only A1 scale s_t (168-hour price SD at the origin): terciles over all '
                  '10,747 keys (low / medium / high volatility); coverage at 50/80/95 with mean width per policy and regime',
        'alpha_path': 'every DL policy\'s alpha_t per nominal alpha at every emission, per fold, from the lineage',
        'reaction_after_peak_start': {
            'aci': 'days from 2022-08-15 to the first emission whose alpha_t(0.05) is lower than at the 2022-08-15 emission',
            'width': 'days from 2022-08-15 to the first day whose mean 95% width exceeds 1.25 x the policy\'s mean 95% width '
                     'over 2022-08-01..14',
            'policies': 'W, W+ACI, W+DL, W+DLF, v4, v4+DL, v3, v3+DL (those that exist)'}},
    'shock_days': {
        'set': f'in each fold, the ceil({S.SHOCK_SHARE:.0%} x represented days) delivery days with the largest absolute change '
               'in daily mean actual price from the previous calendar day (all canonical hours of both days); and the peak\'s '
               f'first {S.PEAK_FIRST_DAYS} days, 2022-08-15..24',
        'window': f'each shock day and the {S.SHOCK_AFTER_DAYS} days after it, within the fold window',
        'metrics': 'MAE, WIS and 50/80/95 coverage with mean width',
        'policies': 'W, W+DL, W+DLF (and, descriptively, v4, v4+DL, v3, v3+DL, W+ACI)'},
    'lear_penalty_stability': {
        'source': 'logged selections in CP-20\'s verified HG component cache (selected_relative_alpha and validation MAE by '
                  'alpha, per component A1_w/B2_w, local hour and origin); no LEAR refit',
        'flip_rate': 'per component and hour, share of consecutive origins whose selected penalty differs',
        'margin': 'relative validation-MAE gap between the selected penalty and the runner-up'},
    'fit_cost_and_daily_cycle': {
        'fit_cost': 'every PN fit: rows, configuration, inner and full-window wall/CPU seconds, worker peak memory; PN inner and '
                    'final fits against CP-21 L-P\'s (reports/block-challenger/fits.parquet)',
        'daily_cycle': 'cold on the M3 with 4 worker processes, at 25 evaluation origins (five per fold at offsets '
                       f'{list(CYCLE_OFFSETS)} days into the window, the next day with eligible hours if one has none): data and '
                       'features, the A1_w/B2_w refits, the member fits the replacement needs (R: PN\'s four full-window fits; '
                       'M: PN-sel\'s selection and refit and L-P\'s), the interval layer from the state persisted that morning, '
                       'and issuance; refitted components must equal CP-20\'s and every issued vector the main run\'s, bit for bit',
        'which': 'the replacement W (with its adopted layer); with no replacement, R and M each, descriptively',
        'role': 'diagnostic, not a criterion (§17.5 D3)'},
}


def section(root: Path, heading: str, until: str) -> str:
    """A ratified section, verbatim, from its heading line to the next named heading."""
    text = (root / 'capstone_v21.md').read_text()
    start = text.index(heading)
    end = text.index(until, start + len(heading))
    return text[start:end].rstrip() + '\n'


def fixtures(root: Path) -> dict:
    """Run the synthetic CP-22 fixtures (no research data) and record their outcome."""
    run = subprocess.run([sys.executable, '-m', 'pytest', '-q', '-p', 'no:cacheprovider', 'tests/cp22'], cwd=root,
                         capture_output=True, text=True)
    tail = run.stdout.strip().splitlines()[-1] if run.stdout.strip() else ''
    return {'command': 'python -m pytest -q -p no:cacheprovider tests/cp22', 'exit_code': run.returncode, 'summary': tail,
            'files_sha256': {f'tests/cp22/{p.name}': sha(p) for p in sorted((root / 'tests/cp22').glob('test_*.py'))}}


def build(root: Path) -> dict:
    root = Path(root)
    ids = identities(root)
    m = origin_manifest(root)
    pre = {name: json.loads((art() / 'preflight' / name).read_text()) for name in PREFLIGHT}
    e1, bench = pre['e1.json'], pre['benchmark.json']
    counts = ledger().read()['counts']
    origins = [{'fold': o['fold'], 'day': o['day'], 'phase': o['phase'], 'n_forecast': o['n_forecast']} for o in e1['origins_detail']]
    fx = fixtures(root)
    if fx['exit_code'] != 0:
        raise ValueError(f'fixtures fail: {fx["summary"]}')
    main_hours = bench['per_origin_single_thread_wall_seconds'] * e1['origins_with_eligible_hours'] * 1.5 / 3600
    return {
        'schema': 'cp22-protocol-v1', 'checkpoint': 'CP-22', 'revision': 'cp22-prerun-1',
        'anchor': 'capstone_v21.md v21-r9 §20 (ratified 2026-10-01)', 'evidence_class': 'development_post_selection',
        'status': 'frozen before any main fit, admission or outer scoring; only accuracy-blind timing and thread-determinism fits '
                  'on training-only data preceded it (reports/v4-revision/preflight/)',
        'issued_and_inherited_sha256': ids,
        'issued_brief': {'path': 'docs/track-b/evidence/cp-22/issued-brief.md', 'sha256': ISSUED['docs/track-b/evidence/cp-22/issued-brief.md'],
                         'canonical_copy': '.local/artifacts/cp-22/issued-brief.md', 'identical_to_pasted_brief': True},
        'hg_identity': hg_identity(root), 'cp21_fit_identity': cp21_fit_identity(root),
        'members': {
            'PN': 'one pooled LightGBM for all 24 local hours on the section-4 normalised target: L-N\'s representation (23 '
                  'normalised CP-15 features, 3 weather columns, 3 indicators) fitted as L-P\'s pooled model; HG\'s and v4\'s information',
            'PN-avg': 'equal mean of the G1-G4 full-window forecasts, each inverted to EUR/MWh with the origin\'s common level and '
                      f'scale: (G1 + G2 + G3 + G4) / 4 in float64; averaging after inversion equals averaging before within {AVERAGING_TOLERANCE} EUR/MWh',
            'PN-sel': 'section 17.3 selection: inner fits on the window minus its last 28 calendar delivery days, lowest validation '
                      'MAE in EUR/MWh after inversion, exact tie to the smaller configuration; the selected configuration\'s '
                      'full-window fit, the same fitted model (bit for bit)',
            'reused': {'L-P': 'CP-21 retained fit cache, identity-verified', 'L-N': 'idem', 'L-R': 'idem (inside v4)',
                       'A1_w, B2_w': 'CP-20 HG component cache, identity-verified',
                       'HGL (v4)': 'CP-21 committed vectors', 'HG (v3)': 'CP-20 committed vectors'}},
        'policies': {
            'R': {'role': 'eligible, tried first', 'central': 'A1/3 + B2/3 + PN-avg/3', 'layer': 'H'},
            'M': {'role': 'eligible, tried second', 'central': 'A1/3 + B2/3 + PN-sel/6 + L-P/6', 'layer': 'H'},
            'A-PN-sel': {'role': 'attribution', 'central': 'A1/3 + B2/3 + PN-sel/3', 'layer': 'H'},
            'A-LP': {'role': 'attribution: v3 + pooled raw', 'central': 'A1/3 + B2/3 + L-P/3', 'layer': 'H'},
            'A-LN': {'role': 'attribution, descriptive', 'central': 'A1/3 + B2/3 + L-N/3', 'layer': 'H'},
            'W+ACI': {'role': 'attribution: adaptive coverage on the unweighted 28-day buffer', 'central': 'W', 'layer': 'ACI'},
            'W+DL': {'role': 'eligible for the layer decision only', 'central': 'W', 'layer': 'DL'},
            'W+DLF': {'role': 'eligible only as an add-on to an adopted W+DL', 'central': 'W', 'layer': 'DLF'},
            'v4+DL': {'role': 'attribution, descriptive', 'central': 'v4 (HGL)', 'layer': 'DL'},
            'v3+DL': {'role': 'attribution, descriptive', 'central': 'v3 (HG)', 'layer': 'DL'}},
        'composite_float_expression': 'cp22.pn.composite: A1/3 + B2/3 + sum(member/k), left to right in float64 (v4\'s own '
                                      'expression A1/3 + B2/3 + L-N/6 + L-R/6 reproduced bit for bit)',
        'composite_parity_tolerance_eur_mwh': 1e-9, 'member_weight': '1/3, fixed',
        'sequence': {'fixed_new_policies': list(FIXED_NEW), 'w_policies': list(W_POLICIES),
                     'order': 'protocol freeze; warm-up PN fits; training-only admission of the seven fixed new policies; '
                              'admission freeze committed; evaluation PN fits; comparison replay; vectors committed; pass 1 '
                              'and cp22-replacement; replacement.json committed; only if W exists: W-arm admission (committed), '
                              'W-arm comparison (committed), pass 2, cp22-dynamic-layer, then cp22-fast-component'},
        'capacity_grid': list(GRID), 'grid_order': 'strictly increasing trees x leaves; the smaller configuration is the earlier one',
        'selection': {'inner_split': f'window minus its last {INNER_DAYS} calendar delivery days', 'metric': 'validation MAE in '
                      'EUR/MWh after inversion with each row\'s own origin level and scale', 'tie': 'exact tie -> smaller configuration',
                      'outcomes': 'training data only'},
        'history': {'window': '[max(2019-01-01, D-728 calendar days), D), eligible rows', 'minimum_window_rows': MIN_TRAIN_ROWS[POOLED],
                    'minimum_inner_rows': MIN_TRAIN_ROWS[POOLED], 'minimum_validation_rows': MIN_VALIDATION_ROWS[POOLED]},
        'lgbm': {'parameters': 'inherited CP-15 parameters, capacity only from the grid (cp21.lgbm.parameters)', 'seed': 42,
                 'objective': 'quantile alpha 0.5', 'deterministic': True, 'n_jobs': 1, 'processes': 'up to 4',
                 'thread_determinism': bench['all_thread_counts_identical'], 'fits_per_origin': FITS_PER_ORIGIN},
        'interval_layers': {
            'H': 'section 14.2 via cp16.residuals.SharedResidualState (V2-H), one state per policy, central passed twice',
            'DL': {'class': 'cp22.dl.DynamicResidualState (subclass of SharedResidualState)', 'buffer_days': BUFFER_DAYS,
                   'recency_weight': f'2^(-a/{HALF_LIFE_DAYS:g}), a = age in calendar days from the newest buffer day (age 0), '
                                     'normalised to sum to one; each observation carries its day\'s weight',
                   'half_life_days': HALF_LIFE_DAYS, 'n_h': '(sum w)^2 / sum w^2', 'shrinkage': f'w_h = n_h/(n_h + {SHRINK}); 0 below '
                   f'{MIN_HOUR_DAYS} distinct days', 'p50': 'weighted median',
                   'weighted_quantile': 'p_k = (S_k - w_k/2 - w_1/2) / (1 - w_1/2 - w_n/2) over sorted values, linear '
                                        'interpolation in p; equal weights reproduce numpy.quantile(method=linear)',
                   'aci': {'convention': 'Gibbs-Candes; alpha is a miscoverage', 'alphas': list(ALPHAS), 'gamma_per_day': GAMMA,
                           'start': 'alpha at each fold\'s genuine warm-up start; the warm-up updates it',
                           'update': 'after each released complete day that carried an emitted interval: alpha_{t+1} = '
                                     'clip(alpha_t + gamma*(alpha - m_t)); releases in delivery-day order',
                           'm_t': 'fraction of canonical hours outside the emitted (rearranged) interval at alpha_t',
                           'clip': {str(a): list(clip_bounds(a)) for a in ALPHAS},
                           'levels': 'alpha_t/2 and 1 - alpha_t/2', 'emission': 'every day the buffer holds 28 complete days'},
                   'rearrangement': 'sort each row\'s seven quantiles (fixed monotone rearrangement)'},
            'ACI': 'DL with equal weights (numpy.quantile itself) and ACI: W+ACI differs from W only by ACI',
            'DLF': {'weights': f'(2/3)*k7/sum k7 + (1/3)*k1/sum k1, k1(a) = 2^(-a/{FAST_HALF_LIFE_DAYS:g})',
                    'fast_half_life_days': FAST_HALF_LIFE_DAYS, 'fast_share': FAST_SHARE, 'otherwise': 'identical to DL, ACI included'},
            'variants': VARIANTS, 'layer_by_policy': LAYER},
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
                                          'central equal to CP-21\'s committed lineage hash)',
                             'cp22_fits': '.local/artifacts/cp-22/fits/<fold>/<day>/PN.json bound to cp22_input_fingerprint, '
                                          'cp22_protocol_sha256 (this file) and the weather design',
                             'verification': pre['input-verification.json']['tally']},
        'scores': {'point': 'emitted-p50 MAE', 'interval': 'seven-quantile WIS', 'primary': 'S_MAE, S_WIS equal-fold ratios to B0',
                   'secondary': 'pooled observation-weighted scores', 'saved_references': list(S.SAVED), 'display': S.DISPLAY},
        'uncertainty': {'seed': S.BOOTSTRAP_SEED, 'replicates': S.BOOTSTRAP_REPLICATES, 'block_days': S.BLOCK_DAYS,
                        'index_set': 'cp20.scoring._indices (sha256 ' + S.CP20_INDEX_SHA256 + ')',
                        'contrasts_without_w': [f'{c}-{b} ({r})' for c, b, r in S.contrasts(None)],
                        'contrasts_with_w': [f'{c}-{b} ({r})' for c, b, r in S.contrasts('W')],
                        'reading': 'observed joint improvement iff upper dS_WIS < 0 and upper dS_MAE <= 0; observed joint worsening iff '
                                   'lower dS_WIS > 0 and lower dS_MAE >= 0; otherwise no demonstrated joint preference'},
        'rules': S.RULES, 'rules_application': S.APPLICATION,
        'rules_verbatim': section(root, '### 20.6 Pre-registered rules', '### 20.7 '),
        'arms_verbatim': section(root, '### 20.2 Arms, comparator', '### 20.3 '),
        'dynamic_layer_verbatim': section(root, '### 20.3 The dynamic interval layer', '### 20.4 '),
        'investigation': INVESTIGATION,
        'diagnostics': {'support_rule': f'at least {S.SUPPORT_DATES} represented dates per fold for block/hour statements',
                        'stress': 'fold 3, 2022-07-01..09-28 (2,112 hours / 88 days); peak 2022-08-15..31 (408 hours) descriptive',
                        'section8': 'all six original section-8 diagnostics for every new policy (and v3, v4), saved B0-B3 comparators'},
        'caps': CAPS, 'timebox_active_seconds': TIMEBOX_ACTIVE_SECONDS,
        'E1': {'origins': e1['origins'], 'with_eligible_hours': e1['origins_with_eligible_hours'], 'main_fits': e1['main_fits'],
               'main_fit_cap': CAPS['main_lgbm_fits'], 'total_fit_cap': CAPS['lgbm_fits'],
               'planned_non_main_fits': {'benchmark_spent': counts.get('lgbm_fits_benchmark', 0), 'controls': 300,
                                         'reproduction': 80, 'daily_cycle': 450, 'review': 250, 'repair_reserve': 1500},
               'policy_days': {**e1['policy_days'], 'controls': 300, 'daily_cycle': 100, 'review': 1500, 'cap': CAPS['policy_days']},
               'hg_components': {'reproduction': 4, 'daily_cycle': 50, 'controls': 8, 'review': 6,
                                 'lasso_per_component_day': 120, 'cap_component_days': CAPS['component_attempts'],
                                 'cap_lasso': CAPS['primitive_fits']},
               'passes': {'reference': 'pass 1 + pass 2 (only if W) + review = 3 of 3',
                          'bootstrap': 'pass 1 + pass 2 (only if W) + review = 3 of 3'},
               'machine_hours_estimate': {'main_fits': round(main_hours, 2), 'replay_scoring_controls_cycle_review_tests': 6.0,
                                          'basis': 'benchmark single-thread wall per origin x origins x 1.5; charged as wall x '
                                                   'declared workers'},
               'sufficiency_problems': e1['sufficiency_problems'], 'feasible': True},
        'E2': {'reproduction': 'reports/v4-revision/reproduce.md', 'output_schema': {
            'predictions.parquet': 'fold, policy, timestamp_utc, delivery_date, origin_utc, y_true, central, scale, level, '
                                   'evidence_class, p025..p975 (seven fixed new policies, 75,229 rows)',
            'predictions-w.parquet': 'the same for W+ACI, W+DL, W+DLF (32,241 rows), only if W exists',
            'members.parquet': 'fold, timestamp_utc, delivery_date, PN-avg, PN-sel, L-P, L-N, L-R, A1, B2, HG, HGL (10,747 rows)'}},
        'E3': {'ledger': '.local/artifacts/cp-22/ledger/budget.json (cumulative, never reset; counters reserved before use)',
               'monitor': 'scripts/cp22_revision.py monitor: wall x declared workers, process-tree and aggregate RSS, added disk, '
                          'duplicate/5th-worker refusal, markers',
               'preflight_disk_free_bytes': shutil.disk_usage(str(art())).free},
        'E4': {'lead': 'CP-22 Track B Engineering Lead (this session)',
               'reviewer': 'one fresh independent Integration Critic on a clean detached checkout of the exact final candidate '
                           'under .local/worktrees/cp-22/critic', 'final_candidate_sha': 'recorded at review'},
        'protocol_change_rule': 'no parameter, policy, grid, weight, layer setting, diagnostic definition or rule change informed '
                                'by outcomes; implementation defects are corrected with explicit records and reruns, never hidden tuning',
        'python': sys.version.split()[0], 'platform': platform.platform(),
        'dependencies': {name: importlib.metadata.version(name) for name in ('lightgbm', 'numpy', 'pandas', 'scikit-learn', 'pyarrow')},
        'implementation_sha256': {name: sha(root / name) for name in IMPLEMENTATION},
        'frozen_inputs_sha256': {name: sha(root / name) for name in FROZEN},
        'preflight_sha256': {f'reports/v4-revision/preflight/{name}': None for name in PREFLIGHT},
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
    protocol['preflight_sha256'] = {f'reports/v4-revision/preflight/{name}': sha(dest / name) for name in PREFLIGHT}
    atomic(out / 'protocol.json', protocol)
    print(json.dumps({'protocol_sha256': sha(out / 'protocol.json'), 'fixtures': protocol['fixtures']['summary'],
                      'E1': {k: v for k, v in protocol['E1'].items()}}), flush=True)
    return 0
