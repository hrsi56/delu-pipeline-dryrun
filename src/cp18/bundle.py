"""The dated bundle: the seven kinds of `docs/track-b/evidence/cp-17/payload-schema.md`, built from the live store.

**What goes in** (`build`), for a run at wall clock `now`:

| Kind | File(s) | From |
|---|---|---|
| `manifest` | `manifest.json` | the served identity: the product, the frozen policy identity and package, the issuing host, the daily artifact, its three times, the freshness state, and every other file with its SHA-256, size and kind |
| `issued_forecasts` | `issued-forecasts.json` | every issued day of the store, from its own artifact (a failed issuance issues nothing) |
| `outcomes` | `outcomes.json` | the reconciled outcomes of every issued day (`cp18.outcomes`); pending days are `null` |
| `evaluation` | `evaluation.json` | the static development records, then the day's recomputed windows (`cp18.evaluation`) |
| `fitted_artifact` | `artifact.json`, `artifact-arrays.npz` | the manifest's artifact from the store; its fit's exported arrays (the issuing run's attempt folder, or a refit under the issuing host, checked against the artifact) |
| `training_window` | `A1-<day>.{json,npz}`, `B0-<day>.{json,npz}` | A1's payload for the artifact's day ("Train it yourself"); B0's payload for every issued day (the context the Space recomputes); each checked against the payload hash its day's record holds |
| `series` | `series.json` | hourly prices through the last day published at `now`, no later than the development boundary before the final test |

**Freshness** (the manifest's, from the store and `now`): the target day is the latest delivery day whose cutoff has
passed (`cp18.plan.target_day`). `fresh` when it is issued; `failed` when it is decided as not issued (any of the five
causes, a failed issuance included); `stale` when it is not decided yet. The as-of day is `now`'s Europe/Berlin date,
and the latest issued day is the store's. A day that is not issued stays shown as not issued, with its cause. Nothing
is labelled late or timely (decision 22).

**The gates before publication** (`gate`): `cp18.r7.verify` (R7: a raised error is a refused bundle), then
`cp18.spaceglue.recompute` natively, the same file the Space runs: any mismatch blocks the day's publication as a
product failure. Both are recorded in the bundle's build record, with the size budget: the first-visit bytes (the
manifest and the records the Space loads at start: the issued forecasts, the outcomes, the evaluation records and the
artifact it names) against the training-window bytes and the other on-request files.
"""
from __future__ import annotations

import argparse
from datetime import date, datetime, timedelta
import hashlib
import json
from pathlib import Path
import shutil
import tempfile

import numpy as np

from cp25 import guard

from . import evaluation as EV, outcomes as OC, plan, spaceglue as G
from .host import MANIFEST
from .records import BUNDLES_PATH, RecordLog
from .store import LiveStore

SCHEMA = 'cp18b-bundle-build-v1'
ISSUING_HOST = 'Pyodide 0.28.0 in Node.js v26.7.0'
EVIDENCE_LABEL = ('A1 · final candidate · pending 4.7T; development data through 2026-09-06 only; the issued days are '
                  'a development rehearsal (CP-18 dry runs), not prospective and not 4.7T')
#: The live product's label (CP-18c, r20 §24.14 item 5; O13): a store whose first day is after the development boundary.
LIVE_EVIDENCE_LABEL = ("A1 · final candidate · pending 4.7T; the live product's own issued days from {first} (the Owner's "
                       'decision 28), prospective evaluation in progress, never 4.7T\'s issuance or score; the '
                       'development-period displays stay on days through 2026-09-06')
START_FILES = ('manifest.json', 'issued-forecasts.json', 'outcomes.json', 'evaluation.json', 'artifact.json')
SERIES_FIRST_DAY = date(2019, 1, 1)


def dump(path: Path, value) -> None:
    """A served JSON file: sorted keys, no spaces, a final newline (as CP-17's reference bundle)."""
    path.write_text(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n', encoding='utf-8')


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read(folder: Path) -> dict[str, bytes]:
    return {p.name: p.read_bytes() for p in sorted(Path(folder).iterdir()) if p.is_file() and p.name != '.DS_Store'}


# ------------------------------------------------------------------ the store's days
def issued_days(store: LiveStore) -> list[dict]:
    """Each issuance entry with its artifact and record, each checked against the ledger's file hashes."""
    out = []
    for entry in store.entries():
        if entry['kind'] != 'issuance':
            continue
        art = (store.folder / entry['artifact_file']).read_bytes()
        rec = (store.folder / entry['record_file']).read_bytes()
        if sha(art) != entry['artifact_file_sha256'] or sha(rec) != entry['record_file_sha256']:
            raise ValueError(f"{entry['day']}: the store's artifact or record differs from its ledger")
        out.append({'day': entry['day'], 'entry': entry, 'artifact': json.loads(art), 'record': json.loads(rec)})
    return out


def freshness(store: LiveStore, now: datetime) -> dict:
    days = store.days()
    target = plan.target_day(now)
    issued = sorted(d for d, v in days.items() if v['status'] == 'issued')
    if not issued:
        raise ValueError('the store has no issued day: there is no artifact to serve')
    item = days.get(str(target))
    if item and item['status'] == 'issued':
        state = 'fresh'
    elif item and item['status'] == 'not issued':
        state = 'failed'
    else:
        state = 'stale'
    return {'state': state, 'as_of_day': str(now.astimezone(guard.BERLIN).date()), 'latest_issued_day': issued[-1],
            'target_day': str(target), 'target_status': (item or {}).get('status') or 'not decided yet',
            'target_cause': (item or {}).get('cause')}


def not_issued(store: LiveStore, fresh: dict) -> list[dict]:
    """Every recorded day that is not issued, within the issued record and up to the target day, with its cause."""
    from . import causes
    out = []
    for day, item in sorted(store.days().items()):
        if item['status'] == 'issued':
            continue
        if item['status'] == 'outside':
            text = 'outside the issued record (' + item['note'] + ')'
        elif item.get('cause'):
            text = causes.text(item['cause'])
        else:
            text = 'cause not recorded'
        out.append({'day': day, 'status': item['status'], 'cause': item.get('cause'), 'text': text})
    if fresh['state'] == 'stale':
        out.append({'day': fresh['target_day'], 'status': 'not decided yet', 'cause': None,
                    'text': 'not issued yet: no run has decided it'})
    return out


def live_store(store: LiveStore) -> bool:
    """A live product's store: its first issuable day after the development boundary (the carry's first live day)."""
    first = store.first_day()
    carried = [e for e in store.entries() if e['kind'] == 'carry']
    if carried:
        first = date.fromisoformat(carried[-1]['first_live_day'])
    return first > guard.BOUNDARY


def first_live_day(store: LiveStore) -> str:
    carried = [e for e in store.entries() if e['kind'] == 'carry']
    return carried[-1]['first_live_day'] if carried else str(store.first_day())


def evidence_label(gaps: list[dict], live_from: str | None = None) -> str:
    """The manifest's evidence label, then each day not issued with its cause: the frozen manifest's one free-text
    field (a dedicated field would change the frozen schema). A live store's outside days (before its first live day)
    are named once, as a range, not day by day."""
    base = EVIDENCE_LABEL if live_from is None else LIVE_EVIDENCE_LABEL.format(first=live_from)
    if live_from is not None:
        gaps = [g for g in gaps if g['status'] != 'outside']
    if not gaps:
        return base
    return base + '; not issued: ' + '; '.join(f"{g['day']}, {g['text']}" for g in gaps)


def manifest_times(times: dict) -> dict:
    """The artifact's three times in the frozen manifest's notation: an instant written `...Z` (the first live artifact's
    data refresh, from the archive's retrieval times) is restated as the same instant `...+00:00`; the artifact is
    served as issued."""
    return {k: (v[:-1] + '+00:00' if isinstance(v, str) and v.endswith('Z') else v) for k, v in times.items()}


def served_artifact(entries: list[dict], fresh: dict) -> dict:
    """The manifest's artifact: the target day's failed issuance when its chain failed, else the latest issued day's."""
    by_day = {e['day']: e for e in entries}
    target = by_day.get(fresh['target_day'])
    if fresh['state'] == 'failed' and target is not None:
        return target
    return by_day[fresh['latest_issued_day']]


# ------------------------------------------------------------------ payloads, checked against the day's record
class Payloads:
    """Each day's served payload, by the payload hash its day record holds: from a cache folder when one holds it,
    else rebuilt by the live-input builder (`cp18.live_inputs`) from the store's state before that day."""

    def __init__(self, store: LiveStore, entries: list[dict], *, caches=(), root=None, project=None, recorded=None,
                 work=None, archive=None):
        self.store, self.entries = store, {e['day']: e for e in entries}
        self.archive = archive
        self.caches = [Path(c) for c in caches]
        self.root, self.project, self.recorded = root, project, recorded
        self.work = Path(work or tempfile.mkdtemp(prefix='cp18b-payloads-'))
        self.used: dict[str, str] = {}

    def expected(self, day: str, policy: str) -> str:
        return self.entries[day]['record']['inputs']['payload_sha256'][policy]

    def _from(self, folder: Path, day: str, policy: str) -> tuple[bytes, bytes] | None:
        meta, npz = folder / f'{policy}-{day}.json', folder / f'{policy}-{day}.npz'
        if not meta.is_file():
            return None
        try:
            value = json.loads(meta.read_text())
        except ValueError:
            return None
        if value.get('payload_sha256') != self.expected(day, policy):
            return None
        return meta.read_bytes(), (npz.read_bytes() if npz.is_file() else b'')

    def get(self, day: str, policy: str) -> tuple[bytes, bytes]:
        for cache in self.caches:
            for folder in (cache, cache / day):
                found = self._from(folder, day, policy)
                if found:
                    self.used[f'{policy}-{day}'] = f'cache {folder}'
                    return self._checked(found, day, policy)
        built = self.rebuild(day)
        found = self._from(built, day, policy)
        if not found:
            raise ValueError(f'{policy}-{day}: the rebuilt payload is not the issued one')
        self.used[f'{policy}-{day}'] = 'rebuilt by the live-input builder'
        return self._checked(found, day, policy)

    def _checked(self, found, day, policy):
        from cp17 import dailyfit
        meta_bytes, npz_bytes = found
        meta = json.loads(meta_bytes)
        folder = Path(tempfile.mkdtemp(prefix='check-', dir=self.work))
        (folder / 'm.json').write_bytes(meta_bytes)
        arrays = {}
        if npz_bytes:
            (folder / 'a.npz').write_bytes(npz_bytes)
            arrays = dailyfit.load_npz(folder / 'a.npz')
        if dailyfit.payload_digest(arrays, {k: v for k, v in meta.items() if k != 'payload_sha256'}) != self.expected(day, policy):
            raise ValueError(f'{policy}-{day}: the payload differs from its identity')
        return meta_bytes, npz_bytes

    def rebuild(self, day: str) -> Path:
        from . import boundary
        if self.root is None or (self.recorded is None and not boundary.is_live()):
            raise FileNotFoundError(f'{day}: no cached payload and no recorded inputs to rebuild it from')
        from cp17.shadow import ShadowStates
        from cp26.search import serving_search
        from .live_inputs import LiveInputs
        target = self.work / 'rebuilt' / day
        if (target / f'A1-{day}.json').exists():
            return target
        ledger = self.store.entries()
        i = next(k for k, e in enumerate(ledger) if e['kind'] == 'issuance' and e['day'] == day)
        state = ShadowStates.from_dict(json.loads((self.store.folder / ledger[i - 1]['state_file']).read_text()))
        if self.recorded is None:              # live (CP-18c): the committed archive; anything missing is retrieved
            from .live.transport import HttpTransport
            inputs = LiveInputs(self.root, self.project, None, transport=HttpTransport())
        else:
            inputs = LiveInputs(self.root, self.project, self.recorded)
        if self.archive is not None:
            inputs = _ArchivedInputs(inputs, Path(self.archive))
        d = date.fromisoformat(day)
        state.release(d, inputs.truth(d))
        search = inputs.search_file(serving_search(d))
        built = inputs.build(d, buffers={p: state.buffer(p) for p in ('v4', 'v2', 'B0')}, search_file=Path(search),
                             out=target)
        if not built.get('shown'):
            raise ValueError(f'{day}: the live-input builder did not rebuild the day ({built.get("reasons")})')
        return target


def _ArchivedInputs(inputs, archive: Path):
    """The live-input builder with its archive at `archive` (a dry run's live folder) instead of `root/live/inputs`."""
    from .live_inputs import LiveInputs

    class Inputs(LiveInputs):
        @property
        def archive_root(self) -> Path:
            return archive
    new = Inputs.__new__(Inputs)
    new.__dict__.update(inputs.__dict__)
    return new


# ------------------------------------------------------------------ the fitted arrays
def check_arrays(arrays: bytes, files: dict) -> str:
    """The served arrays reproduce the issued artifact (`spaceglue.rebuild_a1`); returns how exactly."""
    trial = dict(files, **{'artifact-arrays.npz': arrays})
    return G.rebuild_a1(trial)['reproduced']


def refit_arrays(root, project, payload_dir: Path, day: str, out: Path, artifact: dict | None = None) -> Path:
    """A1's fit of `day` again under the issuing host with the export on. The fit is deterministic on the host: with
    `artifact`, its quantiles and every member's weights' hash must equal the issued artifact's, and `check_arrays`
    then binds the arrays to it again."""
    from cp17 import daily, runtime
    from .host import PACKAGE_ZIP
    host = daily.Host(Path(project) / runtime.PYODIDE_DIR, Path(project) / PACKAGE_ZIP)
    run = host.run(payload_dir, f'A1-{day}', out)
    if run.get('errors'):
        raise RuntimeError(f'the refit failed: {run["errors"][0][-400:]}')
    if artifact is not None:
        result = run['python']['result']
        members = [m['params_sha256'] for m in artifact['lineage']['training']['members']]
        if result['quantiles_sha256'] != artifact['quantiles_sha256'] or \
                [m['params_sha256'] for m in result['members']] != members:
            raise RuntimeError("the refit does not reproduce the issued artifact's quantiles and weights")
    return Path(run['folder']) / 'run-arrays.npz'


# ------------------------------------------------------------------ the series
def series(root, last: date) -> dict:
    from cp28 import reader
    from cp17.issue import canonical_hours
    prices = reader.prices(root)
    keys, values = [], []
    d = SERIES_FIRST_DAY
    while d <= last:
        for k in canonical_hours(d):
            v = prices.get(k)
            keys.append(k)
            values.append(None if v is None or not np.isfinite(v) else float(v))
        d += timedelta(days=1)
    return {'schema': 'cp17-series-v1', 'unit': 'EUR/MWh', 'first_day': str(SERIES_FIRST_DAY), 'last_day': str(last),
            'timestamp_utc': keys, 'price': values}


def last_published(now: datetime) -> date:
    """The last delivery day whose prices are published at `now` (the documented schedule), within the boundary."""
    from .live.availability import price_publication
    d = min(now.astimezone(guard.BERLIN).date() + timedelta(days=1), guard.BOUNDARY)
    while datetime.fromisoformat(price_publication(d)['utc'].replace('Z', '+00:00')) > now:
        d -= timedelta(days=1)
    return d


# ------------------------------------------------------------------ the build
def build(*, root, store_dir, archive, now: datetime, out, development_file=EV.DEVELOPMENT_FILE, arrays=None,
          payloads: Payloads | None = None, run_id='dry-run', outcomes_dir=None, caches=(), project=None,
          recorded=None, transport=None) -> dict:
    """Build the dated bundle in `out` (emptied first) and return its build record (not yet gated)."""
    root, out = Path(root), Path(out)
    store = LiveStore(store_dir)
    store.verify()
    entries = issued_days(store)
    fresh = freshness(store, now)
    gaps = not_issued(store, fresh)
    chosen = served_artifact(entries, fresh)
    artifact = chosen['artifact']
    out.mkdir(parents=True, exist_ok=True)
    for old in out.iterdir():
        old.unlink()
    issued = [e for e in entries if e['artifact']['issued_policy']]
    forecasts = {'schema': 'cp17b-issued-forecasts-v2', 'levels': [0.025, 0.1, 0.25, 0.5, 0.75, 0.9, 0.975],
                 'days': [{'delivery_day': e['day'], 'issue_utc': e['artifact']['issue_utc'],
                           'issuing_artifact_sha256': e['artifact']['artifact_sha256'],
                           'issued_policy': e['artifact']['issued_policy'], 'display': e['artifact']['display'],
                           'keys': e['artifact']['keys'], 'quantiles': e['artifact']['quantiles'],
                           'partial': e['artifact']['partial']} for e in issued]}
    dump(out / 'issued-forecasts.json', forecasts)
    dump(out / 'artifact.json', artifact)
    # outcomes, reconciled at `now`
    log = OC.Outcomes(outcomes_dir) if outcomes_dir else OC.Outcomes.at(root)
    from .live.archive import Archive
    if transport is None and recorded is not None:
        from .live.transport import ReplayTransport
        transport = ReplayTransport(Path(recorded) / 'replay' / 'smard')
    reconciliation = log.reconcile(Archive(Path(archive)), [date.fromisoformat(e['day']) for e in issued], now, run_id,
                                   transport=transport)
    view = log.view()
    outcome_value = OC.bundle_outcomes([date.fromisoformat(e['day']) for e in issued], view, named=live_store(store))
    dump(out / 'outcomes.json', outcome_value)
    # training windows: B0 for every issued day, A1 for the served artifact's day
    payloads = payloads or Payloads(store, entries, caches=caches, root=root, project=project, recorded=recorded,
                                    archive=archive)
    for e in issued:
        meta, npz = payloads.get(e['day'], 'B0')
        (out / f"B0-{e['day']}.json").write_bytes(meta)
        (out / f"B0-{e['day']}.npz").write_bytes(npz)
    if artifact['issued_policy']:
        policy, day = artifact['issued_policy'], artifact['delivery_day']
        meta, npz = payloads.get(day, policy)
        (out / f'{policy}-{day}.json').write_bytes(meta)
        (out / f'{policy}-{day}.npz').write_bytes(npz)
    # B0's context forecasts, natively, each equal to the issuing host's shadow
    b0, absent = G.b0_items(read(out), [e['day'] for e in issued])
    shadows = {e['day']: e['record']['shadows']['B0']['quantiles_sha256'] for e in issued}
    b0_check = {it['day']: it['quantiles_sha256'] == shadows[it['day']] for it in b0}
    if absent or not all(b0_check.values()):
        raise ValueError(f'B0 context: absent {absent}, unequal to the shadow {[d for d, ok in b0_check.items() if not ok]}')
    static, static_sha = EV.load_development(root, development_file)
    daily = EV.daily_windows(forecasts, outcome_value, b0)
    dump(out / 'evaluation.json', EV.evaluation_bundle(static, daily))
    # the fitted arrays
    arrays_how = None
    if artifact['issued_policy']:
        if arrays is None:
            raise ValueError('the fitted arrays of the served artifact are required (--arrays or a refit)')
        data = Path(arrays).read_bytes()
        arrays_how = check_arrays(data, read(out))
        (out / 'artifact-arrays.npz').write_bytes(data)
    last = last_published(now)
    dump(out / 'series.json', series(root, last))
    files = []
    for path in sorted(out.iterdir()):
        from cp17 import schema
        kind = schema.FILE_KIND.get(path.name) or ('training_window' if schema.TRAINING_FILE.match(path.name) else None)
        files.append({'path': path.name, 'sha256': sha(path.read_bytes()), 'bytes': path.stat().st_size, 'kind': kind})
    identity = json.loads((root / MANIFEST).read_text())
    manifest = {'schema': 'cp17-manifest-v1',
                'product': {'name': 'delu-day-ahead-policy', 'designated_policy': 'A1',
                            'evidence_label': evidence_label(gaps, first_live_day(store) if live_store(store) else None)},
                'policy': {'policy_identity_sha256': identity['policy_identity_sha256'],
                           'package_sha256': identity['policy_identity']['code']['package_sha256'],
                           'issuing_host': ISSUING_HOST},
                'artifact': {'artifact_sha256': artifact['artifact_sha256'], 'delivery_day': artifact['delivery_day'],
                             'issued_policy': artifact['issued_policy'], 'display': artifact['display']},
                'times': manifest_times(artifact['times']),
                'freshness': {k: fresh[k] for k in ('state', 'as_of_day', 'latest_issued_day')}, 'files': files}
    dump(out / 'manifest.json', manifest)
    return {'schema': SCHEMA, 'run_id': run_id, 'now': now.isoformat(), 'out': str(out),
            'manifest_sha256': sha((out / 'manifest.json').read_bytes()), 'freshness': fresh, 'not_issued': gaps,
            'store_head': store.verify()['entry_sha256'], 'days': store.days(),
            'served_artifact': {'day': artifact['delivery_day'], 'sha256': artifact['artifact_sha256'],
                                'arrays': arrays_how},
            'payload_sources': payloads.used, 'b0_equal_to_shadow': b0_check,
            'reconciliation': reconciliation, 'static_development_sha256': static_sha,
            'series_last_day': str(last), 'daily_windows': daily, 'size': size(read(out))}


def size(files: dict[str, bytes]) -> dict:
    """The size budget (A9, "Loading and network"): what loads at start against what loads on request."""
    first = {k: len(v) for k, v in files.items() if k in START_FILES}
    training = {k: len(v) for k, v in files.items() if k.startswith('A1-')}
    request = {k: len(v) for k, v in files.items() if k not in START_FILES and not k.startswith('A1-')}
    return {'first_visit_bundle_bytes': sum(first.values()), 'first_visit_files': first,
            'training_window_bytes': sum(training.values()), 'training_window_files': training,
            'on_request_bytes': sum(request.values()), 'on_request_files': request,
            'total_bytes': sum(len(v) for v in files.values())}


def identity_problems(files: dict[str, bytes], root=None) -> list[str]:
    """The served identity against the frozen one (§16.4's altered identities): the manifest's policy identity and
    package, and the served artifact's lineage, must be CP-17c's; and before the final test no served delivery day lies
    after the development boundary (§24.2)."""
    from delu_forecast.claims import REPO_ROOT
    frozen = json.loads((Path(root or REPO_ROOT) / MANIFEST).read_text())
    problems = []
    try:
        manifest = json.loads(files['manifest.json'])
        artifact = json.loads(files['artifact.json'])
        forecasts = json.loads(files['issued-forecasts.json'])
        outcomes = json.loads(files['outcomes.json'])
        served = json.loads(files['series.json']) if 'series.json' in files else None
    except (KeyError, ValueError) as exc:
        return [f'the identity check could not read the bundle ({type(exc).__name__})']
    if manifest['policy']['policy_identity_sha256'] != frozen['policy_identity_sha256']:
        problems.append("the manifest's policy identity is not the frozen one")
    if manifest['policy']['package_sha256'] != frozen['policy_identity']['code']['package_sha256']:
        problems.append("the manifest's package is not the frozen one")
    if manifest['policy']['issuing_host'] != ISSUING_HOST:
        problems.append("the manifest's issuing host is not the frozen one")
    lineage = artifact.get('lineage') or {}
    policy = artifact.get('issued_policy')
    if policy:
        stem = f"{policy}-{artifact['delivery_day']}.json"
        meta = json.loads(files[stem]) if stem in files else {}
        if meta.get('payload_sha256') != (lineage.get('inputs') or {}).get(policy):
            problems.append("the served training window is not the issued fit's (its payload identity differs)")
    if lineage.get('policy_identity_sha256') != frozen['policy_identity_sha256'] or \
            lineage.get('code_sha256') != frozen['policy_identity']['code']['package_sha256']:
        problems.append("the served artifact's lineage is not the frozen policy's")
    # The red line applies to the research windows only (CP-18c; r20 §24.14 items 2 and 5): the served series and the
    # development evaluation windows stay on days through 2026-09-06; the live record's issued days, outcomes and the
    # prospective window lie after it by design. No window labelled 4.7T may hold a day before 4.7T is scored, and the
    # prospective window holds no development day: the classes never pool.
    research = [served['last_day']] if served is not None else []
    try:
        windows = json.loads(files['evaluation.json'])['windows']
    except (KeyError, ValueError, TypeError):
        windows = []
    for w in windows if isinstance(windows, list) else []:
        days_scored = [d.get('day') for d in ((w.get('record') or {}).get('days') or []) if isinstance(d, dict)]
        if w.get('window') == 'development':
            research += [d for d in days_scored if d]
        elif w.get('window') == '4.7T' and days_scored:
            problems.append("a window labelled 4.7T holds scored days before 4.7T is scored")
        elif w.get('window') == 'prospective' and any(d and date.fromisoformat(d) <= guard.BOUNDARY for d in days_scored):
            problems.append('the prospective window holds a development day (the classes never pool)')
    if any(date.fromisoformat(d) > guard.BOUNDARY for d in research):
        problems.append('a research window lies after the development boundary (the red line, §24.2)')
    return problems


def gate(files: dict[str, bytes], root=None) -> dict:
    """Before any publication: R7's verification, the frozen identity and the red line, then the native
    recomputation check. Fails closed."""
    from . import r7
    verified = r7.verify(files)
    if not verified['accepted']:
        return {'publishable': False, 'verify': verified, 'recompute': None, 'identity': None,
                'reason': 'the bundle does not verify (the frozen verifier through cp18.r7)'}
    identity = identity_problems(files, root)
    if identity:
        return {'publishable': False, 'verify': verified, 'recompute': None, 'identity': identity,
                'reason': 'the served identity is not the frozen one, or the red line is crossed'}
    try:
        check = G.recompute(files)
    except Exception as exc:        # a recomputation that cannot run blocks, never passes
        return {'publishable': False, 'verify': verified, 'recompute': {'error': f'{type(exc).__name__}: {exc}'},
                'identity': [], 'reason': 'the recomputation could not run: a product failure'}
    ok = check['match'] and not check['blocks_publication']
    return {'publishable': ok, 'verify': verified, 'recompute': check, 'identity': [],
            'reason': None if ok else 'the recomputation does not match the committed records: a product failure'}


def record(live, value: dict, verdict: dict) -> dict:
    """The day's bundle record under `<live>/bundles/` (`live/bundles/` on `main`): the manifest and the gate's
    verdict, once per as-of day."""
    manifest = json.loads(Path(value['out'], 'manifest.json').read_text())
    as_of = value['freshness']['as_of_day']
    return RecordLog(Path(live) / Path(BUNDLES_PATH).name).append(
        'bundle', f"{as_of}/bundle.json",
        {'schema': 'cp18b-bundle-record-v1', 'as_of_day': as_of, 'run_id': value['run_id'], 'manifest': manifest,
         'manifest_sha256': value['manifest_sha256'], 'publishable': verdict['publishable'],
         'verify_problems': verdict['verify']['problems'], 'recompute_match': (verdict['recompute'] or {}).get('match'),
         'size': value['size']},
        as_of_day=as_of, manifest_sha256=value['manifest_sha256'], publishable=verdict['publishable'])


def refit_served(root, project, live, now, caches, recorded, work) -> Path:
    """The served artifact's arrays by a refit under the issuing host, on its own payload (from a cache or rebuilt)."""
    store = LiveStore(live / 'store')
    store.verify()
    entries = issued_days(store)
    chosen = served_artifact(entries, freshness(store, now))
    artifact = chosen['artifact']
    payloads = Payloads(store, entries, caches=caches, root=root, project=project, recorded=recorded,
                        archive=live / 'inputs')
    folder = Path(tempfile.mkdtemp(prefix='refit-', dir=work))
    policy, day = artifact['issued_policy'], artifact['delivery_day']
    meta, npz = payloads.get(day, policy)
    (folder / f'{policy}-{day}.json').write_bytes(meta)
    (folder / f'{policy}-{day}.npz').write_bytes(npz)
    return refit_arrays(root, project, folder, day, folder / 'runs', artifact)


def _live_transport():
    """Live (CP-18c): the outcomes a later archive day does not hold yet are retrieved once published."""
    from . import boundary
    if not boundary.is_live():
        return None
    from .live.transport import HttpTransport
    return HttpTransport()


class _NoRetrieval:
    """A transport that retrieves nothing (the pending control): every request is a miss."""

    def send(self, request):
        from .live.transport import ReplayMiss
        raise ReplayMiss('retrieval disabled for this control')


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog='python -m cp18.bundle')
    ap.add_argument('--root', default='.', type=Path, help='the checkout holding the live store and the input archive')
    ap.add_argument('--project', default='.', type=Path)
    ap.add_argument('--live', type=Path, help="the data-only folder (default <root>/live): store/, inputs/, outcomes/, "
                                             "evaluation/, bundles/")
    ap.add_argument('--recorded', type=Path, help="recorded responses to acquire or rebuild archive days from")
    ap.add_argument('--now', help="the run's wall clock (a dry run's simulated one; a live run's default is the real clock)")
    ap.add_argument('--mode', default='dry-run', choices=('dry-run', 'live'),
                    help='live: the live record inside cp18.boundary.live() (CP-18c)')
    ap.add_argument('--out', required=True, type=Path)
    ap.add_argument('--arrays', type=Path, help="the served artifact's exported arrays (the issuing attempt's run-arrays.npz)")
    ap.add_argument('--refit', action='store_true', help="no arrays at hand (a run that issued nothing): refit the served "
                                                         "artifact's fit under the issuing host, then check it")
    ap.add_argument('--cache', action='append', default=[], type=Path, help='folders holding payloads already built')
    ap.add_argument('--run-id', default='dry-run')
    ap.add_argument('--record', type=Path, help='write the build record and the gate here')
    ap.add_argument('--no-retrieval', action='store_true', help='reconcile from the archives only (a control: no '
                                                                     'outcome retrieval, so unarchived days stay pending)')
    ap.add_argument('--commit-records', action='store_true', help='append to live/evaluation and live/bundles')
    args = ap.parse_args(argv)
    from contextlib import nullcontext
    from . import boundary
    if args.mode == 'dry-run' and not args.now:
        raise SystemExit('a dry run needs --now, its simulated wall clock')
    now = plan.parse_now(args.now)
    with (boundary.live(now=now) if args.mode == 'live' else nullcontext()):
        return _main(args, now)


def _main(args, now) -> int:
    root = args.root.resolve()
    live = (args.live or root / 'live').resolve()
    arrays = args.arrays
    if arrays is None and args.refit:
        arrays = refit_served(root, args.project.resolve(), live, now, args.cache, args.recorded, args.out.parent)
    value = build(root=root, store_dir=live / 'store', archive=live / 'inputs', outcomes_dir=live / 'outcomes', now=now,
                  out=args.out, arrays=arrays, run_id=args.run_id, caches=args.cache,
                  project=args.project.resolve(), recorded=args.recorded,
                  transport=_NoRetrieval() if args.no_retrieval else _live_transport())
    verdict = gate(read(args.out), root)
    value['gate'] = {k: v for k, v in verdict.items()}
    if args.commit_records:
        EV.commit_daily(live, date.fromisoformat(value['freshness']['as_of_day']), value['daily_windows'],
                        value['static_development_sha256'], args.run_id)
        value['bundle_record'] = record(live, value, verdict)
    if args.record:
        args.record.parent.mkdir(parents=True, exist_ok=True)
        args.record.write_text(json.dumps(value, indent=1, sort_keys=True, default=str) + '\n')
    print('bundle', value['manifest_sha256'][:16], value['freshness']['state'], 'as of', value['freshness']['as_of_day'],
          '| latest issued', value['freshness']['latest_issued_day'], '| publishable' if verdict['publishable'] else
          f"| REFUSED: {verdict['reason']}", '| first visit', value['size']['first_visit_bundle_bytes'], 'bytes')
    return 0 if verdict['publishable'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
