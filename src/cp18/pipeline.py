"""One run of the daily pipeline: the H2 gate once, decision 23's catch-up, each issuance through `cp17.daily`.

**A run, in order.**
1. **The plan,** from the run's wall clock (`cp18.plan`): the current target day and the open days of its three-day
   lookback, oldest first, at most four. **The red line** refuses a plan holding any day after 2026-09-06 before
   anything is synchronized, checked or retrieved (`cp25.guard`), and the refusal names no such day.
2. **The claim** (`cp18.claim`): fetch `main`, check it out, load the store and verify it (`cp18.store.LiveStore`,
   CP-17c's checks first). A store that does not verify is refused: nothing is issued or written.
3. **Days that left the lookback** unissued are closed as not issued, with their last attempt's cause, or
   "no run within the lookback" when no run attempted them.
4. **The H2 gate** (`cp17.hostcheck`): pins, binary, machine, canary, once, before the first issuance. Any failure
   issues nothing in this run; each planned day records the failed check as its attempt's cause.
5. **Each planned day,** oldest first: fetch again (a day another host decided meanwhile is never issued); load the
   store's state and release D−2 with the published prices; ask the inputs for the day's payloads, built with the
   state's buffers. The serving search's state not in place (S_14) or inputs not shown to precede the cutoff leave
   the day open with that cause. Otherwise `cp17.daily.issue_day` issues it, with its shadows registered in the state;
   any earlier day of this run left open is closed first (days go forward); the day's artifact, record and state are
   appended to the store, committed and pushed. A lost push race discards this host's issuance and ends the run.
6. **The run's record** is appended to the store and pushed, whatever the run did: the store lists every run.

**No deadline, no "late" label (decision 22).** Availability is the inputs' business, judged by publication time
against each day's own cutoff and recorded with each input's publication time, retrieval time and hash. The run's
own times (its start, each issuance's start and end) are recorded in the day's record beside the artifact, its
lineage, and never relabel anything.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timezone
import json
import os
from pathlib import Path
import platform
import time
import uuid

from cp25 import guard

from . import causes, plan
from .claim import Claim, ClaimError
from .store import INPUTS_PATH, STORE_PATH, LiveStore, StoreRefused

SCHEMA = 'cp18a-run-v1'
RECORD_SCHEMA = 'cp18a-day-record-v1'
MANIFEST = 'docs/track-b/evidence/cp-17c/freeze-manifest.json'
PACKAGE_ZIP = '.local/artifacts/cp-17c/bundle/src.zip'
RED_LINE = ('refused: the plan holds a delivery day after the development boundary (cp25.guard, §24.2); '
            'nothing was synchronized, checked, retrieved or issued')


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


def dump(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=1, sort_keys=True, allow_nan=False, default=str) + '\n')


@dataclass
class Settings:
    """Where a run works. `root` is the pipeline's checkout of the product repository (the store at
    `root/live/store`); `project` holds `.local/` (the pinned host files); `out` is the run's scratch folder."""
    root: Path
    project: Path
    out: Path
    host_kind: str = 'runner'             # 'runner' or 'local'
    mode: str = 'dry-run'
    run_label: str = ''
    extra: dict = field(default_factory=dict)


def frozen_identity(root: Path) -> dict:
    value = json.loads((Path(root) / MANIFEST).read_text())
    return {'policy_identity_sha256': value['policy_identity_sha256'],
            'package_sha256': value['policy_identity']['code']['package_sha256']}


def frozen_host(root: Path, project: Path, timeout: float | None = None):
    """The issuing host: Pyodide's pinned folder and the frozen package zip, checked against the manifest."""
    from cp17 import daily, runtime
    zip_path = Path(project) / PACKAGE_ZIP
    expected = json.loads((Path(root) / MANIFEST).read_text())['policy_identity']['code']['zip_sha256']
    if not zip_path.exists() or runtime.sha256_file(zip_path) != expected:
        raise RuntimeError(f'{PACKAGE_ZIP}: not the frozen package zip')
    return daily.Host(Path(project) / runtime.PYODIDE_DIR, zip_path, **({} if timeout is None else {'timeout': timeout}))


def h2_checks(settings: Settings) -> dict:
    """The four checks of `cp17.hostcheck`, in its order, for this checkout and project."""
    from cp17 import hostcheck, runtime
    root, project, out = settings.root, settings.project, settings.out
    return {'pins': lambda: hostcheck.pins(root),
            'binary': lambda: hostcheck.binary(project),
            'machine': lambda: hostcheck.machine(),
            'canary': lambda: hostcheck.canary(frozen_host(root, project), project / runtime.CANARY_DIR, out / 'canary')}


def default_issue(day: date, gfs_available: bool, payloads: Path, *, settings: Settings, parent_sha256, data_refresh_utc,
                  states) -> dict:
    from cp17 import daily
    return daily.issue_day(day, gfs_available, payloads, host=frozen_host(settings.root, settings.project),
                           out=settings.out / 'runs', identity=frozen_identity(settings.root),
                           parent_sha256=parent_sha256, data_refresh_utc=data_refresh_utc, states=states)


def _attempts(store: LiveStore) -> dict[str, list[dict]]:
    """Each still-open day's attempts, from the store's run records, oldest first."""
    out: dict[str, list[dict]] = {}
    for run in store.runs():
        for day, item in (run.get('days') or {}).items():
            if item.get('status') == 'open':
                out.setdefault(day, []).append({'run_id': run['run_id'], 'cause': item['cause']})
    return out


def _close(store: LiveStore, day: date, record: dict) -> None:
    attempts = _attempts(store).get(str(day), [])
    this = record['days'].get(str(day))
    if this and this.get('status') == 'open':
        attempts = attempts + [{'run_id': record['run_id'], 'cause': this['cause']}]
    cause = attempts[-1]['cause'] if attempts else causes.NO_RUN
    store.record_not_issued(day, cause, [a['run_id'] for a in attempts])
    record['closed'].append({'day': str(day), 'cause': cause, 'cause_text': causes.text(cause),
                             'attempts': [a['run_id'] for a in attempts]})


def run(settings: Settings, now: datetime, *, claim: Claim, inputs, checks: dict | None = None,
        issue=default_issue, serving=None) -> dict:
    """One run. `inputs` provides each day's payloads (`cp18.inputs`); `checks` replaces the H2 checks (tests and
    forced failures only); `issue` is `cp17.daily.issue_day` behind `default_issue`."""
    from cp17 import hostcheck
    from cp26.search import serving_search
    serving = serving or serving_search
    started = time.perf_counter()
    run_id = settings.run_label or f"{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-{settings.host_kind}-{uuid.uuid4().hex[:8]}"
    target = plan.target_day(now)
    record = {'schema': SCHEMA, 'run_id': run_id, 'mode': settings.mode, 'host_kind': settings.host_kind,
              'started_utc': utc_now(), 'wall_clock': now.isoformat(), 'target_day': None, 'lookback': None,
              'closed': [], 'planned': [], 'gate': None, 'days': {}, 'outcome': None,
              'machine': {'system': platform.system(), 'machine': platform.machine(),
                          'runner_image': os.environ.get('ImageVersion'), 'github_run_id': os.environ.get('GITHUB_RUN_ID')}}
    try:
        plan.admit(plan.lookback(target) + [target])
    except guard.BoundaryRefused:
        return {**record, 'outcome': RED_LINE, 'exit': 3}
    record.update(target_day=str(target), lookback=[str(d) for d in plan.lookback(target)])
    store_dir = settings.root / STORE_PATH
    try:
        claim.sync()
        store = LiveStore(store_dir)
        store.verify()
    except (ClaimError, StoreRefused, OSError, ValueError) as exc:
        return {**record, 'outcome': f'refused before any issuance: {type(exc).__name__}: {exc}', 'exit': 2}

    def publish(paths, message) -> dict:
        result = claim.publish(paths, message + f'\n\nDelu-Daily-Run: {run_id}\n')
        record.setdefault('commits', []).append({k: v for k, v in result.items()})
        return result

    first, done = store.first_day(), set(store.days())
    for day in plan.to_close(target, first_day=first, done=done, last_closed=store.last_closed()):
        _close(store, day, record)
    days = plan.plan(target, first_day=first, done=set(store.days()), last_closed=store.last_closed())
    record['planned'] = [str(d) for d in days]
    if days:
        gate = hostcheck.gate(checks or h2_checks(settings))
        record['gate'] = gate
        if not gate['passed']:
            for day in days:
                record['days'][str(day)] = {'status': 'open', 'cause': causes.H2, 'cause_text': causes.text(causes.H2),
                                            'check': gate['failed']}
            days = []
    lost = None
    for i, day in enumerate(days):
        item = record['days'][str(day)] = {'status': None, 'started_utc': utc_now()}
        t0 = time.perf_counter()
        if claim.moved():
            claim.sync()
            store = LiveStore(store_dir)
            store.verify()
            record.setdefault('resynchronized', []).append(str(day))
        if str(day) in store.days() or (store.last_closed() and day <= store.last_closed()):
            item.update(status='decided elsewhere', note='another host decided this day first; nothing issued here')
            continue
        state, _ = store.state()
        parent_state = state.digest()
        state.release(day, inputs.truth(day))
        k = serving(day)
        search = store.search_file(k) if k is not None and k >= 14 else inputs.search_file(k)
        if search is None or not Path(search).is_file():
            item.update(status='open', cause=causes.S14, cause_text=causes.text(causes.S14), search=k,
                        note='the search state that serves this day is not in place; the day waits for it')
            continue
        built = inputs.build(day, buffers={p: state.buffer(p) for p in ('v4', 'v2', 'B0')}, search_file=Path(search),
                             out=settings.out / 'payloads' / str(day))
        item['inputs'] = {k2: v for k2, v in built.items() if k2 not in ('payloads',)}
        if not built.get('shown'):
            item.update(status='open', cause=causes.INPUTS, cause_text=causes.text(causes.INPUTS))
            continue
        for earlier in days[:i]:
            if record['days'].get(str(earlier), {}).get('status') == 'open':
                _close(store, earlier, record)
        issuance_started = utc_now()
        result = issue(day, bool(built['gfs_available']), Path(built['payloads']), settings=settings,
                       parent_sha256=store.last_artifact_sha256(), data_refresh_utc=built.get('data_refresh_utc', ''),
                       states=state)
        art = result['artifact']
        from cp17 import issue as cp17issue
        day_record = {'schema': RECORD_SCHEMA, 'day': str(day), 'run_id': run_id, 'host_kind': settings.host_kind,
                      'run_started_utc': record['started_utc'], 'issuance_started_utc': issuance_started,
                      'issuance_finished_utc': utc_now(), 'cutoff_utc': plan.cutoff(day).isoformat(),
                      'gfs_available': bool(built['gfs_available']), 'inputs': item['inputs'],
                      'gate': {'passed': record['gate']['passed'], 'started_utc': record['gate']['started_utc'],
                               'machine': next((c.get('described') for c in record['gate']['checks']
                                                if c['check'] == 'machine'), None)},
                      'path': [(a['policy'], a['status']) for a in result['decision']['attempts']],
                      'causes': {a['policy']: a.get('cause') for a in result['decision']['attempts'] if a['status'] != 'ok'},
                      'display': result['decision']['display'], 'issued_policy': art['issued_policy'],
                      'artifact_sha256': art['artifact_sha256'], 'artifact_verifies': cp17issue.verify_artifact(art) == [],
                      'shadows': result['shadows'], 'process_seconds': result['process_seconds'],
                      'process_seconds_sum': result['process_seconds_sum'], 'issue_day_seconds': result['total_seconds'],
                      'blocked_non_local_requests': result['blocked_non_local_requests']}
        store.record_day(day, state, art, day_record, parent_state)
        status = 'issued' if art['issued_policy'] else 'not issued'
        item.update(status=status, policy=art['issued_policy'], display=art['display'],
                    artifact_sha256=art['artifact_sha256'], seconds=time.perf_counter() - t0,
                    process_seconds_sum=result['process_seconds_sum'],
                    cause=None if art['issued_policy'] else causes.CHAIN,
                    cause_text=None if art['issued_policy'] else causes.text(causes.CHAIN))
        paths = [STORE_PATH] + ([f'{INPUTS_PATH}/{day}'] if (settings.root / INPUTS_PATH / str(day)).exists() else [])
        try:
            pushed = publish(paths, f'live: {day} {status} ({art["display"]})\n\nDelu-Daily: {day}')
        except ClaimError as exc:
            item['publish_error'] = str(exc)
            record['outcome'] = f'publication refused: {exc}'
            record['exit'] = 2
            break
        if pushed.get('lost'):
            item.update(status='lost the claim', note='another host issued this day first; this issuance is discarded')
            lost = str(day)
            break
    unissued = [d for d, v in record['days'].items() if v.get('status') not in ('issued', 'decided elsewhere')]
    if lost is None and record.get('exit') is None:
        store = LiveStore(store_dir)
        store.verify()
        record['finished_utc'] = utc_now()
        record['seconds'] = time.perf_counter() - started
        record['outcome'] = ('nothing to issue' if not record['planned'] else
                             'done' if not unissued else f"done; not issued: {', '.join(unissued)}")
        record['exit'] = 1 if unissued else 0
        store.record_run(run_id, record)
        try:
            publish([STORE_PATH], f'live: run {run_id} ({record["outcome"]})')
        except ClaimError as exc:
            record['outcome'] += f'; the run record was not published: {exc}'
            record['exit'] = 2
    elif lost is not None:
        record['outcome'] = f'lost the claim at {lost}: the other host\'s record stands; this run stopped'
        record['exit'] = 0
    record.setdefault('finished_utc', utc_now())
    record.setdefault('seconds', time.perf_counter() - started)
    return record
