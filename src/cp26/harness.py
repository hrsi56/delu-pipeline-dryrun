"""The CP-26 harness (capstone v21-r15 §24.7, "The harness and the parity gate").

The landed code refuses any snapshot, boundary or anchor other than its own (`cp15.data.load_inputs`,
`cp21.inputs.load`, `cp24.inputs.identities`). The harness therefore calls the landed *primitives*
directly -- `cp15.data.prepare`, `cp15.models.fit_day`, `cp20.components.augment`,
`cp21.lgbm.fit_arm`, `cp24.design.build`, `cp24.member.fit_member`, `cp24.ddnn2.ensemble_median`,
`cp16.residuals.SharedResidualState`, v1's heads, CQR and isotonic -- and replaces only:

* **the data identity:** CP-25's partition specification and manifest (new settings), or
  `data/partitions.json` (old settings);
* **the boundary:** 2026-09-06 through `cp25.guard` (new), or 2026-04-07 (old);
* **the weather source:** CP-20's committed features plus CP-25's gap features (new), or CP-20's
  alone (old);
* **the issue-time label:** D-1 11:00 Europe/Berlin (`cp25.guard.issue_time`, new), or the
  landed D-1 11:00 UTC (`cp15.data.origin_utc`, old);
* **the identity pins:** the protocol revision, the anchor and issued-document hashes and the
  lineage checks.

Every replacement is listed in `HARNESS_LOG`, which the `harness-log` job writes to
`reports/rebaseline/harness-log.json`.

**The Owner's rulings of 2026-10-12** (`docs/track-b/cp-25-rulings-2026-10-12.md`) enter here:
ruling 4's two late GFS runs are set missing (§15.3, all three columns, indicators set); ruling 3's
GFS +6 h delay run sets every affected delivery day's weather missing (`weather_variant='gfs6'`);
ruling 1's no-load variant removes the load columns (`without_load`, `ddnn2_without_load`), and with
`restore=True` the same code path is the identity.
"""
from __future__ import annotations

import contextlib
from dataclasses import dataclass, replace
from datetime import date, timedelta
import hashlib
import json
import os
from pathlib import Path
import subprocess
from functools import lru_cache

import numpy as np
import pandas as pd

BLAS = ('OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS')

from cp15.data import RAW_COLUMNS, prepare, sha  # noqa: E402
from cp20.weather import COLUMNS as WEATHER_COLUMNS, WeatherDesign  # noqa: E402
from cp25 import guard as G25  # noqa: E402
from delu_forecast.folds import load_partition_spec  # noqa: E402

FLOOR = date(2019, 1, 1)
OLD_BOUNDARY = date(2026, 4, 7)
NEW_BOUNDARY = G25.BOUNDARY                     # 2026-09-06
WARMUP_START = date(2020, 1, 1)
SCORED_START = date(2020, 12, 29)
BASE_COMMIT = '647d6eb81b2cf6e0cfaf4152f659d689ff973578'   # main at CP-26's issue
SCIENTIFIC_ANCHOR_COMMIT = '332baeb861cec5a58516c58a29a1a3afe1eec909'
SCIENTIFIC_ANCHOR_SHA256 = '3be866bd10af02cc2b0b956ed91be660dca29a62c0b23b00997bf0b415d100c5'
SCIENTIFIC_AUTHORITY_PATH = 'docs/track-b/evidence/cp-26/continuation-v21-r14/authority.json'
SCIENTIFIC_BRIEF = 'docs/track-b/evidence/cp-26/issued-continuation-brief-v21-r14.md'
SCIENTIFIC_BASE_COMMIT = '1e7eb029f1cbbec6483aea4a7dda0754fd7adf0b'
ANCHOR_COMMIT = '4766a61cb0c836573406f5a9729997d1b6ee9862'
AUTHORITY_COMMIT = 'b06114f1d06cadb38beebff56e6869270d25e797'
AUTHORITY_PATH = 'docs/track-b/evidence/cp-26/cold-rerun-2026-10-15/authority-v21-r16.json'
CONTINUATION_BRIEF = 'docs/track-b/evidence/cp-26/cold-rerun-2026-10-15/issued-continuation-v21-r16.md'
RUNTIME_LINEAGE = 'docs/track-b/evidence/cp-26/cold-rerun-2026-10-15/runtime-code-v21-r16.json'

#: Every pinned identity of the brief (Target; Applicable constraints), recomputed by `identities()`.
PINS = {
    SCIENTIFIC_AUTHORITY_PATH: '6d00862075102c9c9ea40dc024650a07c0c199d7c78e9c10c38cbdfa68d591db',
    SCIENTIFIC_BRIEF: '3e689e0cc3dda0cacf21103ce76da86ffa429af2c84ed8ce90c2e5417baa5c0c',
    'docs/track-b/evidence/cp-26/repair-v21-r14/active-documents.json': 'aa29af82181769afefdb0a4446057dd08f7ebafaac27a398aa2679397a3cadfe',
    'capstone_v21.md': '27bdc6046117a543d4bfcaa439651e7bf2a8fa4d0c58af9631b606a358c10fd1',
    'docs/track-b/capstone_v21-r13-to-v21-r14-amendments.md': '554a3aa4bf81f3fc1cc2319e624295570f919ff1fd4a6d9e2c9f56f8dcf6b79a',
    'docs/track-b/cp-25-rulings-2026-10-12.md': 'c7af67f88157277aafd8dd2b1b49e12a1d9ff4b747ca5bdf178fac664fba53c1',
    'docs/PUBLISH_RULES.md': '1bc0c52351656c5a499291bf15ccb82b0ae8b4e6475d1879b0a4d307daaef94e',
    'docs/track-b/publication-packet-template.md': '4049c3e1f695034ec45fa185d465fce05ab1183cbded72188306579d7287d01a',
    'docs/track-b/capstone_v21-r12-to-v21-r13-amendments.md': '0f3b54ea21fef1d21782cdd40993f23b5d792cc45f638ad15fbdc27b97285e59',
    'docs/track-b/cp-25-landing-2026-10-12.md': 'f33f58985a85803aed04ef48bf9c55db0d7738fac0a7d81db2a9c02d31b11b56',
    'docs/track-b/exposure-ledger.md': '375fef5d200c400b79840dcc9977ab3d63a95ed85af603c8add3b9a9210d98a6',
    'docs/track-b/evidence/cp-26/issued-brief.md': '5e16a134a9dd798733b0393f4f8e2570d03bf251471503d9292966e67e05214b',
    'docs/track-b/evidence/cp-26/issued-repair-brief-v21-r14.md': '5b6a0cdfad870da6b59a03ff29576ff6ec98891e941d56986516e6f3f743ff84',
    'docs/track-b/evidence/cp-26/cp28-frozen-protocol-v21-r14.md': '2b9286b0e948f59f97de33315a8b4fb3ef121003fc4ba82355b1ccf6585497a8',
    'reports/development-inputs/manifest.json': 'f3e8ca15e4466d371de5b6db9c181f77adaa8823ca2e7e0d325182120db32fd9',
    'data/snapshot.parquet': '7dd2dc73407706ca6bd3c1ad51d201ac0de35eec5ca129ce320cca639f697f00',
    'data/partitions.json': '6bee065368b721ae8092ba287e04bb6dda0d4bde80909e72380503ba61e0b090',
    'reports/weather-ablation/weather-features.parquet': '66b8264bb6e124ae0cc540a40ff00d0e9f625cabee6e0ebb6e4605c3e7ea1b58',
    'reports/cp15/protocol.json': '051e4b7cdd1d5db68b26e6d6bc17333c4055c4bcbe5ea693d1d647fe355ccdd6',
    'pyproject.toml': 'aa35326abbb13dfa4ea8ba1b4b367343b77f895e7724d5dc94e0bdb7637b4e4a',
    'uv.lock': '5d2aecc3d173fef6191cb2bb9d14cd008c0e9cc764700f1352b81d1924465a94',
}
# This run's original scientific identity is retained, rather than relabelling
# its signed r14 probes/plans. Active r16 authority is checked independently.
SCIENTIFIC_PINS = {**PINS, 'capstone_v21.md': SCIENTIFIC_ANCHOR_SHA256}
PINS.update({
    AUTHORITY_PATH: 'e2780c59078722d660e27d1606110a4e54ccf3a47bc354c022514768ef92d367',
    CONTINUATION_BRIEF: '4fe30292467596ad65d9391829d6849401ede1cc6be3b7bc9789c47dece60107',
    'docs/track-b/capstone_v21-r14-to-v21-r15-amendments.md':
        '9e7e87361a7030c14158bb880370acaef52940cd2c76f699a59f47d19c03dc41',
    'docs/track-b/capstone_v21-r15-to-v21-r16-amendments.md':
        '36b104634d22093990aa2641a644572e8f892c9547118247023fc4267d5aae72',
})
#: The CP-25 outputs CP-26 reads, each bound by CP-25's manifest (verified in `identities()`).
CP25_READ = ('reports/development-inputs/partitions-development.json',
             'reports/development-inputs/eligibility-calendar.csv',
             'reports/development-inputs/record-masks.json', 'reports/development-inputs/record-masks.csv',
             'reports/development-inputs/tier-register.json', 'reports/development-inputs/margin-days.csv',
             'reports/development-inputs/margin-table.csv',
             'reports/development-inputs/weather-features-gaps.parquet')
#: CP-20's frozen weather design (cp21.inputs.WEATHER_DESIGN_SHA256): the old settings' design.
CP20_DESIGN_SHA256 = 'f5a9c6ed7dae18e711017e9f2887e3c72f64c286137bd7cabfb3469cd10274c0'
#: Ruling 4: the two GFS runs treated as late under §15.3 (delivery days).
LATE_GFS_DELIVERY = (date(2022, 12, 1), date(2023, 1, 21))
#: The landed member code, imported unchanged (checked against the brief's base commit).
LANDED_CODE = ('src/delu_forecast', 'src/cp15', 'src/cp16', 'src/cp20', 'src/cp21', 'src/cp22', 'src/cp24', 'src/cp25')

#: §24.7's harness log: every replaced item, where the landed code pins it, and what replaces it.
HARNESS_LOG = [
    {'item': 'r16 operational authority', 'landed': 'r14 scientific identity retained in signed plans/probes',
     'harness': 'Active r16 anchor/Owner authority and440h named-run admission/500h ceiling are verified separately; current runtime hashes accompany new artifacts. Only operational source/provenance changed; original recipes, seeds and input identities remain frozen.'},
    {'item': 'identity pin: Owner continuation authority',
     'landed': 'r14 repair record marked pending ratification; experiment stopped after repair',
     'harness': 'Owner delivery ratifies the exact r14 anchor/amendment/CP28 bytes; '
                'new hash-bound continuation authority and brief supersede the repair-only execution boundary; '
                'historical records retain their original bytes and statuses'},
    {'item': 'data identity: partition specification', 'landed': 'data/partitions.json via cp15.data.load_inputs, '
     'cp16.inputs.load, cp20.inputs.load, cp21.inputs.load (eda_cutoff 2026-04-07)',
     'harness': 'reports/development-inputs/partitions-development.json (CP-25; development to 2026-09-06), passed to '
                'cp15.data.prepare as the spec whose eda_cutoff is 2026-09-06'},
    {'item': 'data identity: manifest', 'landed': 'reports/cp15/protocol.json input_sha256 (cp15.data.protocol), '
     'cp20/cp21/cp24 identities()', 'harness': 'reports/development-inputs/manifest.json (f3e8ca15...) and every '
     'CP-25 output CP-26 reads, verified against it; data/snapshot.parquet verified at 7dd2dc73...'},
    {'item': 'boundary', 'landed': 'cp21.inputs.BOUNDARY = 2026-04-07 and guard_boundary; cp15.data.prepare refuses '
     'days after spec.eda_cutoff', 'harness': 'cp25.guard (2026-09-06): parquet filters before materialisation, then '
     'cp25.guard.admit_delivery_day and admit_rows on every loaded row'},
    {'item': 'weather source', 'landed': 'cp21.inputs.weather_design: CP-20 features only, design sha f5a9c6ed...',
     'harness': 'CP-20 features (66b8264b...) plus CP-25 gap features (fbc2517e...); ruling 4 sets delivery days '
                '2022-12-01 and 2023-01-21 missing (documented_lateness); the gfs6 delay run sets every affected day '
                'missing (margin-days.csv)'},
    {'item': 'issue-time label', 'landed': 'cp15.data.origin_utc (D-1 11:00 UTC), imported by cp15.models, '
     'cp21.lgbm and cp24.member for their fit records', 'harness': 'cp25.guard.issue_time (D-1 11:00 Europe/Berlin), '
     'rebound in cp15.models, cp21.lgbm and cp24.member under the new settings; the label enters records only'},
    {'item': 'identity pin: protocol revision', 'landed': "cp15.data.protocol requires revision 'v21-r1-comparison-1' "
     'and its input_sha256 (the v21-r1 anchor at CP-15 evidence)', 'harness': 'reports/cp15/protocol.json is read for '
     'its recipe sections only, verified at 051e4b7c... (CP-15 artifact manifest); CP-26 protocol revision '
     'cp26-v21-r14, exact research bytes ratified in the committed Owner continuation authority'},
    {'item': 'identity pin: anchor and issued documents', 'landed': 'cp20/cp21/cp22/cp23/cp24 ISSUED (v21-r4..r11 '
     'anchors, briefs, PUBLISH_RULES 1.x)', 'harness': 'v21-r14 candidate capstone_v21.md (3be866bd...), the rulings '
     '(c7af67f8...), PUBLISH_RULES 1.5 (1bc0c523...), the packet template (4049c3e1...), the amendment record '
     '(0f3b54ea...), r14 amendment (554a3aa4...), CP-25 landing record (f33f5898...), '
     'historical brief (5e16a134...), repair brief, versioned CP-28 protocol and historical active-document manifest; '
     'Owner continuation brief (3e689e0c...) and authority (6d008620...) added by the continuation (all verified by PINS)'},
    {'item': 'identity pin: lineage checks', 'landed': 'cache lineage of CP-15/16/20/21/24 (SavedComponents, '
     'HGComponents, D2Cache, check_protocol per attempt)', 'harness': 'CP-26 fits every member fresh at every origin; '
     'its own cache is keyed by run, member, origin and an identity hash (cp26.fits); the parity gate reads the '
     'committed vectors under reports/{cp15,v2-causal,weather-ablation,block-challenger,ddnn2}'},
    {'item': 'DDNN-2 search schedule', 'landed': 'cp24.sampler.batches (per fold, from D0-57, blocks from '
     '2020-01-01), sampler_seed([24, r, f, 0x5A3]), fit_seed, member_seeds', 'harness': '§24.7: S_k = 2020-01-15 + 182k, '
     'k = 0..13; blocks tile back from S_k - 1, newest up to11 support-eligible from 2019-07-01; SeedSequence([26, k, 0x5A3]); '
     'CP-26 fit and member seeds (cp26.search); successive halving, ranking and ties are cp24.sampler/cp24.search'},
]


class HarnessError(ValueError):
    pass


def project() -> Path:
    return Path(os.environ.get('CP26_PROJECT_ROOT', '/Users/djourno/Downloads/PJM'))


def art() -> Path:
    return project() / '.local' / 'artifacts' / 'cp-26'


def check_blas() -> None:
    bad = [n for n in BLAS if os.environ.get(n) != '1']
    if bad:
        raise HarnessError(f'BLAS threads must be 1 before numpy loads: {bad}')


def _git(root: Path, *args) -> str:
    return subprocess.check_output(['git', *args], cwd=root, text=True)


def git_blob_sha256(root: Path, rev: str, name: str) -> str:
    blob = subprocess.check_output(['git', 'show', f'{rev}:{name}'], cwd=root)
    return hashlib.sha256(blob).hexdigest()


def landed_code(root: Path) -> dict:
    """Every landed module file, with its SHA-256; each must equal the brief's base commit."""
    root = Path(root)
    changed = _git(root, 'diff', '--name-only', BASE_COMMIT, '--', *LANDED_CODE).split()
    if changed:
        raise HarnessError(f'landed member code changed since {BASE_COMMIT[:12]}: {changed}')
    untracked = _git(root, 'ls-files', '--others', '--exclude-standard', '--', *LANDED_CODE).split()
    if untracked:
        raise HarnessError(f'untracked files inside landed code: {untracked}')
    out = {}
    for base in LANDED_CODE:
        for path in sorted((root / base).rglob('*.py')):
            out[str(path.relative_to(root))] = sha(path)
    return out


def identities(root: Path) -> dict:
    """Every identity CP-26 relies on, recomputed; any mismatch raises."""
    root = Path(root)
    G25.admit_delivery_day(NEW_BOUNDARY)
    out = {}
    for name, expected in PINS.items():
        actual = sha(root / name)
        if actual != expected:
            raise HarnessError(f'pinned identity mismatch: {name} {actual} != {expected}')
        out[name] = actual
    if git_blob_sha256(root, ANCHOR_COMMIT, 'capstone_v21.md') != PINS['capstone_v21.md']:
        raise HarnessError('anchor differs from its recorded candidate commit')
    manifest = json.loads((root / 'reports/development-inputs/manifest.json').read_text())['outputs_sha256']
    for name in CP25_READ:
        actual = sha(root / name)
        if manifest.get(name) != actual:
            raise HarnessError(f'CP-25 output differs from its manifest: {name}')
        out[name] = actual
    cp15 = json.loads((root / 'reports/cp15/artifact-manifest.json').read_text())['artifact_sha256']
    if cp15['reports/cp15/protocol.json'] != PINS['reports/cp15/protocol.json']:
        raise HarnessError('CP-15 protocol differs from the CP-15 artifact manifest')
    return out


def continuation_authority(root: Path) -> dict:
    """Verify the recorded Owner grant; an environment flag cannot grant authority."""
    root = Path(root)
    for name in (AUTHORITY_PATH, CONTINUATION_BRIEF):
        if sha(root / name) != PINS[name] or git_blob_sha256(root, AUTHORITY_COMMIT, name) != PINS[name]:
            raise HarnessError('continuation authority differs from its committed Owner-delivery record')
    authority = json.loads((root / AUTHORITY_PATH).read_text())
    if (authority.get('status') != 'owner_ratified_and_delivered'
            or authority.get('checkpoint') != 'cp-26'
            or authority.get('brief_sha256') != PINS[CONTINUATION_BRIEF]
            or authority.get('anchor_commit') != ANCHOR_COMMIT
            or not authority.get('no_new_lockdown_suspension')
            or not authority.get('no_cp28_execution') or not authority.get('no_publication')):
        raise HarnessError('continuation authority does not grant this bounded CP-26 run')
    for item in authority['ratified_documents']:
        if PINS.get(item['path']) != item['sha256'] or sha(root / item['path']) != item['sha256']:
            raise HarnessError('ratified continuation document identity changed')
    return authority


def input_fingerprint(root: Path) -> str:
    current = identities(root)
    scientific = {name: current[name] for name in SCIENTIFIC_PINS}
    if git_blob_sha256(root, SCIENTIFIC_ANCHOR_COMMIT, 'capstone_v21.md') != SCIENTIFIC_ANCHOR_SHA256:
        raise HarnessError('original scientific anchor is not preserved')
    scientific['capstone_v21.md'] = SCIENTIFIC_ANCHOR_SHA256
    scientific.update({name: current[name] for name in CP25_READ})
    ident = {'identities': scientific, 'landed_code': landed_code(root)}
    return hashlib.sha256(json.dumps(ident, sort_keys=True).encode()).hexdigest()


@lru_cache(maxsize=4)
def _runtime_record(root: Path) -> dict:
    path = Path(root) / RUNTIME_LINEAGE
    raw = path.read_bytes()
    if raw != subprocess.check_output(['git', 'show', f'HEAD:{RUNTIME_LINEAGE}'], cwd=root):
        raise HarnessError('r16 operational lineage must be committed')
    record = json.loads(raw)
    if record.get('scientific_base_commit') != SCIENTIFIC_BASE_COMMIT:
        raise HarnessError('r15 lineage names a different scientific base')
    return record


def recipe_code_sha256(root: Path, name: str) -> str:
    """Historical recipe identity for three explicitly checked operational edits.

    Actual executed source hashes are retained separately in runtime_lineage;
    existing probe receipts are never rewritten to imply execution of current code.
    """
    actual = sha(Path(root) / name)
    if name not in ('src/cp26/harness.py', 'src/cp26/search.py', 'src/cp26/parity.py'):
        return actual
    record = _runtime_record(Path(root))
    if record['actual_code_sha256'].get(name) != actual:
        raise HarnessError('operational code changed after compatibility validation: ' + name)
    return record['original_recipe_code_sha256'][name]


def runtime_lineage(root: Path) -> dict:
    record = _runtime_record(Path(root))
    for name, expected in record['actual_code_sha256'].items():
        if sha(Path(root) / name) != expected:
            raise HarnessError('operational source identity changed: ' + name)
    return {'authority_document': AUTHORITY_PATH, 'authority_sha256': PINS[AUTHORITY_PATH],
            'active_anchor_sha256': PINS['capstone_v21.md'],
            'actual_code_sha256': record['actual_code_sha256'],
            'scientific_base_commit': SCIENTIFIC_BASE_COMMIT}


# ------------------------------------------------------------------ settings
@dataclass(frozen=True)
class Spec:
    """What cp15.data.prepare reads from a partition spec: the last admissible delivery day."""
    eda_cutoff: date
    source: str


@dataclass(frozen=True)
class Settings:
    name: str                 # 'old' (the parity gate) or 'new' (CP-26)
    boundary: date
    weather: str              # 'cp20' (old), 'nominal' or 'gfs6' (new)

    @property
    def issue_label(self) -> str:
        return 'D-1 11:00 UTC (landed, fixed +01:00)' if self.name == 'old' else 'D-1 11:00 Europe/Berlin (§24.4)'


OLD = Settings('old', OLD_BOUNDARY, 'cp20')
NEW = Settings('new', NEW_BOUNDARY, 'nominal')
GFS6 = Settings('new', NEW_BOUNDARY, 'gfs6')


def spec_for(root: Path, settings: Settings):
    root = Path(root)
    if settings.name == 'old':
        spec = load_partition_spec(root / 'data/partitions.json')
        if spec.eda_cutoff != OLD_BOUNDARY:
            raise HarnessError('old partition spec moved')
        return spec
    p = json.loads((root / 'reports/development-inputs/partitions-development.json').read_text())
    last = date.fromisoformat(p['development']['last_delivery_day'])
    if last != NEW_BOUNDARY or p['schema'] != 'cp25-partition-spec-v1':
        raise HarnessError('CP-25 partition specification differs from the boundary')
    return Spec(last, 'reports/development-inputs/partitions-development.json')


def issue_instant(day: date, settings: Settings) -> pd.Timestamp:
    """The forecast's issue instant (UTC) under the settings."""
    if settings.name == 'old':
        from cp15.data import origin_utc
        return origin_utc(day).tz_convert('UTC')
    return pd.Timestamp(G25.issue_time(day)).tz_convert('UTC')


def install_issue_label(settings: Settings) -> None:
    """Rebind the landed records' issue-time label (records only) in this process."""
    import cp15.models
    import cp21.lgbm
    import cp24.member
    from cp15.data import origin_utc

    def label(day):
        return pd.Timestamp(G25.issue_time(day))
    for module in (cp15.models, cp21.lgbm, cp24.member):
        module.origin_utc = origin_utc if settings.name == 'old' else label


# ------------------------------------------------------------------ data
def load_frame(root: Path, settings: Settings, before: date | None = None) -> pd.DataFrame:
    """Raw snapshot rows through the boundary (filters before materialisation, then the guard)."""
    cutoff = settings.boundary if before is None else min(settings.boundary, before - timedelta(days=1))
    frame = pd.read_parquet(Path(root) / 'data/snapshot.parquet', columns=RAW_COLUMNS,
                            filters=[('delivery_date', '>=', FLOOR), ('delivery_date', '<=', cutoff)])
    days = pd.Index(frame['delivery_date'])
    if len(days):
        G25.admit_delivery_day(max(days))
        if max(days) > settings.boundary or min(days) < FLOOR:
            raise HarnessError(f'rows outside {FLOOR}..{settings.boundary}')
    G25.admit_rows(pd.DatetimeIndex(frame['timestamp_utc']).tz_convert('UTC'))
    return frame


def protocol_recipe(root: Path) -> dict:
    p = json.loads((Path(root) / 'reports/cp15/protocol.json').read_text())
    if sha(Path(root) / 'reports/cp15/protocol.json') != PINS['reports/cp15/protocol.json']:
        raise HarnessError('CP-15 recipe changed')
    return p


def load(root: Path, settings: Settings, before: date | None = None):
    """CP-15's prepared inputs (the landed `prepare`) under the settings' spec and boundary."""
    frame = load_frame(root, settings, before)
    return prepare(frame, protocol_recipe(root), spec_for(root, settings))


# ------------------------------------------------------------------ weather
def _features(root: Path, settings: Settings) -> pd.DataFrame:
    root = Path(root)
    cp20 = guarded_weather(root / 'reports/weather-ablation/weather-features.parquet', settings.boundary)
    if settings.weather == 'cp20':
        return cp20
    gaps = guarded_weather(root / 'reports/development-inputs/weather-features-gaps.parquet', settings.boundary)
    if set(cp20.delivery_date) & set(gaps.delivery_date):
        raise HarnessError('CP-20 and CP-25 weather features overlap')
    features = pd.concat([cp20, gaps], ignore_index=True)
    features['timestamp_utc'] = pd.to_datetime(features.timestamp_utc, utc=True)
    if features.timestamp_utc.duplicated().any():
        raise HarnessError('duplicate canonical weather hour')
    if max(features.delivery_date) > NEW_BOUNDARY:
        raise HarnessError('weather after the boundary')
    late = features.delivery_date.isin(LATE_GFS_DELIVERY)
    if late.sum() != 48:
        raise HarnessError('ruling 4: the two late runs do not cover 48 canonical hours')
    features.loc[late, list(WEATHER_COLUMNS)] = np.nan
    features.loc[late, 'status'] = 'documented_lateness'
    features.loc[late, 'reason'] = "Owner's ruling 4 (2026-10-12): AWS objects written after the issue time"
    if settings.weather == 'gfs6':
        affected = gfs6_affected_days(root)
        mask = features.delivery_date.isin(affected)
        features.loc[mask, list(WEATHER_COLUMNS)] = np.nan
        features.loc[mask, 'status'] = 'documented_lateness'
        features.loc[mask, 'reason'] = "delay run: GFS +6 h is not strictly before the issue time (ruling 3)"
    return features.sort_values('timestamp_utc').reset_index(drop=True)


def guarded_weather(path: Path, boundary: date) -> pd.DataFrame:
    """Filter before reading weather rows, then verify canonical delivery dates through CP-25."""
    G25.admit_delivery_day(boundary)
    frame = pd.read_parquet(path, filters=[('delivery_date', '<=', boundary)])
    instants = pd.DatetimeIndex(pd.to_datetime(frame.timestamp_utc, utc=True))
    G25.admit_rows(instants)
    if not np.array_equal(instants.tz_convert('Europe/Berlin').date, frame.delivery_date.to_numpy()):
        raise HarnessError('weather delivery dates disagree with canonical timestamps')
    return frame


def gfs6_affected_days(root: Path) -> set:
    m = pd.read_csv(Path(root) / 'reports/development-inputs/margin-days.csv')
    m = m.loc[m.group.eq('gfs')]
    days = set(pd.to_datetime(m.loc[m.affected_plus_6h.astype(str).eq('True'), 'delivery_date']).dt.date)
    for day in days:
        G25.admit_delivery_day(day)
    if len(days) != 1783:
        raise HarnessError(f'ruling 3: GFS +6 h affects {len(days)} days, not 1,783')
    return days


def weather_design(root: Path, settings: Settings) -> WeatherDesign:
    design = WeatherDesign.from_features(_features(root, settings))
    if settings.weather == 'cp20' and design.sha256 != CP20_DESIGN_SHA256:
        raise HarnessError('old weather design differs from CP-20\'s frozen design')
    return design


def weather_matrix(design: WeatherDesign, data) -> tuple[np.ndarray, np.ndarray]:
    """As `cp21.inputs.weather_matrix`: three same-local-hour columns (NaN = missing) and presence."""
    present = design.present(data.dates, data.hours)
    return design.matrix(data.dates, data.hours, required=np.zeros(len(data.dates), bool)), present


# ------------------------------------------------------------------ the no-load variant (ruling 1)
def load_columns(p: dict) -> tuple[list[int], list[int]]:
    lear = [i for i, n in enumerate(p['lear']['features']) if n.startswith('load_')]
    lgbm = [p['lgbm']['features'].index(n) for n in ('load_forecast_mw', 'load_forecast_day_mean_mw')]
    if lear != list(range(96, 168)) or lgbm != [10, 11]:
        raise HarnessError('the landed load-column layout changed')
    return lear, lgbm


def without_load(data, *, restore: bool = False):
    """Ruling 1, LEAR and LightGBM: the load columns (LEAR's D, D-1 and D-7 load curves; LightGBM's
    `load_forecast_mw` and `load_forecast_day_mean_mw`) removed from the input matrices. With
    `restore=True` the inputs are returned unchanged (the variant's parity check)."""
    if restore:
        return data
    lear, lgbm = load_columns(data.p)
    keep_lear = np.setdiff1d(np.arange(data.lear_raw.shape[1]), lear)
    keep_lgbm = np.setdiff1d(np.arange(data.lgbm_raw.shape[1]), lgbm)
    return replace(data, lear_raw=data.lear_raw[:, keep_lear], lear_normalized=data.lear_normalized[:, keep_lear],
                   lgbm_raw=data.lgbm_raw[:, keep_lgbm], lgbm_normalized=data.lgbm_normalized[:, keep_lgbm])


@contextlib.contextmanager
def ddnn2_without_load(active: bool):
    """Ruling 1, DDNN-2: the always-included `load_d0` and the optional `load_d1`/`load_d7` columns are
    removed from every input block `cp24.design.raw_inputs` builds (training, held-out and forecast rows
    alike). Inactive, it is the identity."""
    import cp24.design as G
    original = G.raw_inputs

    def raw_inputs(dd, ix, groups, form):
        C, B, cnames, bnames = original(dd, ix, groups, form)
        keep = [i for i, n in enumerate(cnames) if not n.startswith('load_')]
        return C[:, keep], B, [cnames[i] for i in keep], bnames
    if active:
        G.raw_inputs = raw_inputs
    try:
        yield
    finally:
        G.raw_inputs = original


NO_LOAD_FLAGS = ('price_d2', 'price_d3', 'price_d7', 'gfs', 'stats', 'calendar')
