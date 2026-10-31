"""Pre-fit CP-20 protocol (E1-E4): frozen before any HG fit, admission or outer scoring."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys

from .budget import CAPS, TIMEBOX_ACTIVE_SECONDS, atomic
from .inputs import identities, origin_manifest

IMPLEMENTATION = ['src/cp20/__init__.py', 'src/cp20/budget.py', 'src/cp20/net.py', 'src/cp20/gfs.py',
                  'src/cp20/extract.py', 'src/cp20/plan.py', 'src/cp20/weather.py', 'src/cp20/assemble.py',
                  'src/cp20/inputs.py', 'src/cp20/components.py', 'src/cp20/execution.py', 'src/cp20/scoring.py',
                  'src/cp20/controls.py', 'src/cp20/protocol.py', 'src/cp20/report.py', 'src/cp20/feasibility.py',
                  'src/cp20/probe.py', 'src/cp20/chain.py', 'scripts/cp20_weather.py',
                  'src/cp15/data.py', 'src/cp15/models.py', 'src/cp15/scoring.py', 'src/cp16/residuals.py',
                  'src/cp16/inputs.py', 'src/cp16/scoring.py', 'src/delu_forecast/features.py']
FROZEN = ['reports/weather-ablation/run-manifest.json', 'reports/weather-ablation/extraction-protocol.json',
          'reports/weather-ablation/extraction-repairs.json', 'reports/weather-ablation/weather-features.parquet',
          'reports/weather-ablation/messages.parquet', 'reports/weather-ablation/runs.csv',
          'reports/weather-ablation/missingness.csv', 'reports/weather-ablation/extended-inventory.csv',
          'reports/weather-ablation/sample-hash-comparison.csv', 'reports/weather-ablation/radiation-clipping.csv',
          'reports/weather-ablation/extraction-summary.json', 'reports/weather-ablation/feasibility.json',
          'reports/weather-ablation/prerun-attempts.csv', 'reports/weather-ablation/throughput-probe.json',
          'reports/weather-ablation/h0-dryrun.json', 'reports/weather-ablation/chain-code-versions.json',
          'reports/weather-ablation/locator-fix-verification.json']


def sha(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build(root: Path) -> dict:
    root = Path(root)
    _, ids = identities(root)
    m = origin_manifest(root)
    summary = json.loads((root / 'reports/weather-ablation/extraction-summary.json').read_text())
    lear = json.loads((root / 'reports/cp15/protocol.json').read_text())['lear']
    folds = [{'fold': f['fold'], 'warmup_start': f['warmup_start'], 'evaluation_start': f['evaluation_start'],
              'evaluation_end': f['evaluation_end'], 'origins': f['date_count'],
              'admission_dates': [a['day'] for a in f['admission']]} for f in m['folds']]
    return {
        'schema': 'cp20-protocol-v1', 'checkpoint': 'CP-20', 'anchor': 'capstone_v21.md v21-r4 section 15',
        'evidence_class': 'development_post_selection', 'status': 'frozen before any HG fit, admission or outer scoring',
        'issued_and_inherited_sha256': ids,
        'arms': {'H0': 'no-weather V2-H exactly as section 14.2: A1/B2 50/50 central blend and frozen H layer; identity-verified CP-15/CP-16 component caches only',
                 'HG': 'the same A1/B2 recipes with only three same-target-local-hour weather columns appended to each hourly LEAR design (raw and normalised); same blend, price-only A1 scale, H recipe, failure rules; own issued errors'},
        'weather_recipe': {
            'product': 'GFS 0.25 deg operational, D-1 00 UTC run, leads f021..f048 (3-hourly)',
            'native_fields': ['u10', 'v10', 'u100', 'v100', 'dswrf'],
            'wind': 'per-cell linear interpolation of each component between bracketing 3-hour endpoints to canonical hour starts h22..h46; speed=sqrt(u^2+v^2) per cell at 10 m and 100 m; then weighted mean',
            'radiation': 'per cell 3-hour block means: A(L-3,L) at L=3 mod 6, 2A(L-6,L)-A(L-6,L-3) at L=0 mod 6, validated decoded bounds; block mean to each constituent hour; negative >= -3q clipped to 0 and logged (q=max packing quantum of contributing messages); below -3q or unknown q/bounds = invalid support',
            'aggregation': 'fixed 47-55.25 N x 5.5-15.5 E inclusive 0.25 deg centres (34x41), cos(latitude) weights normalised over the fixed grid, no renormalisation over missing cells; nonfinite required support -> vector missing',
            'design_columns': ['wx_wind10_mean', 'wx_wind100_mean', 'wx_dswrf_mean'],
            'repeated_local_hour': 'average both canonical vectors; require both; spring gap absent',
            'missing': 'all three columns NaN together; inherited CP-15 LEAR imputer: training-partition median, all-null->0, indicator per feature, training-only StandardScaler, fitted separately in each inner-training split and final window; no interpolation across dates, forward fill or zero-radiation shortcut',
            'label': 'regional weather proxy, not an exact DE-LU polygon or power-generation forecast; no power curve, capacity weights, VRE conversion or feature search'},
        'weather_design_sha256': summary['weather_design_sha256'],
        'missingness': {'classes': summary['feature_status'], 'structural_missing_days': summary['structural_missing_days'],
                        'classified_missing_runs': summary['classified_missing_runs'],
                        'unfinished_extraction': 'none (assembly refuses otherwise; BLOCKED, never imputed)'},
        'admission_disclosures': ['NCAR-only 2019-2020 and the 2021-02-02 NCAR fill: field presence at admission was inferred; CP-20 decoded and validated every extracted message',
                                  'public availability before D-1 11:00 UTC is reconstructed from dated NCEP production-status averages; dissemination lag assumed, not evidenced; no per-day delivery guarantee',
                                  'delivery 2019-01-01 weather structurally missing (pre-2019 run)',
                                  'radiation-precision criterion was defined after admission sampling (post-sample disclosure); v14/v15 DSWRF packed at 10 (or 1) W m-2',
                                  'known endpoint defects: AWS lacks 2021-02-02 (NCAR used); NCAR 2024-09-15 f039 truncated (AWS used)',
                                  'ICON not admitted and excluded; no reanalysis, hindcast, later cycle or pre-2019 input'],
        'folds': folds, 'origins_per_arm': sum(f['origins'] for f in folds), 'admission_dates_per_arm': 35,
        'original_target_keys': 10747, 'contrast_rows': 21494, 'scored_rows': 75229,
        'component_recipe': {'inherited': 'reports/cp15/protocol.json lear', 'relative_alpha_grid': lear['relative_alpha_grid'],
                             'selection': lear['selection'], 'max_iter': lear['max_iter'], 'tol': lear['tol'],
                             'inner_validation': lear['inner_validation'], 'seed': 42,
                             'history': '[max(2019-01-01,D-728),D) per origin availability filter'},
        'residual_recipe': 'section 14.2 H: 28 complete released days <= D-2, w_h=n_h/(n_h+56), w_h=0 below 14 distinct days, linear empirical quantiles; cp16.residuals.SharedResidualState (V2-H output only), one state per arm',
        'comparison': {'primary': 'HG-H0 paired equal-fold B0-normalised score differences', 'bootstrap_seed': 15042,
                       'replicates': 2000, 'block_days': 7, 'index_sets_per_pass': 1,
                       'joint_rule': 'observed joint improvement iff upper 95% CI(dS_WIS)<0 AND upper 95% CI(dS_MAE)<=0; otherwise no demonstrated joint preference (not equivalence, absence of benefit or harm)',
                       'descriptive_order': 'lower S_WIS, then S_MAE, H0 on exact ties; not promotion',
                       'section8': 'all six original criteria for both arms with saved B0-B3 comparators; diagnostics only'},
        'E1_preflight': json.loads((root / 'reports/weather-ablation/feasibility.json').read_text()),
        'E2_reproduction': 'reports/weather-ablation/reproduce.md; fixtures tests/cp20; tolerances: H0 bitwise equal to accepted CP-16 V2-H; component reproduction atol 1e-8 rtol 1e-10 with fingerprint equality',
        'E3_instrumentation': {'ledger': '.local/artifacts/cp-20/ledger/budget.json (cumulative, flock, reserve-before-use)',
                               'machine_time': 'monitor charges wall-clock x declared workers per job', 'caps': CAPS,
                               'timebox_active_seconds': TIMEBOX_ACTIVE_SECONDS,
                               'blas_threads': 1, 'workers_max': 4, 'atomic_saves': 'temp + os.replace'},
        'E4_review_identity': {'executor': 'CP-20 Engineering Lead (this session)',
                               'reviewer': 'fresh independent CP-20 Integration Critic subagent on a clean detached worktree of the exact final candidate'},
        'frozen_inputs_sha256': {f: sha(root / f) for f in FROZEN},
        'implementation_sha256': {f: sha(root / f) for f in IMPLEMENTATION},
        'dependencies': {n: importlib.metadata.version(n) for n in ['numpy', 'pandas', 'pyarrow', 'scikit-learn', 'psutil', 'pytest']},
        'python': sys.version}


def write(root: Path):
    atomic(Path(root) / 'reports/weather-ablation/protocol.json', build(root))
