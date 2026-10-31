"""Prospective CP-26 execution gates; command flags never change the research recipe."""
from __future__ import annotations

import hashlib
import copy
import json
import math
import os
from pathlib import Path
import subprocess

from . import harness as H
from .budget import Budget, CapExceeded, projected_cost, require_compute_ready
from .state import digest

RUN_ID = 'cp26-cold-rerun-2026-10-15-01'
OUTPUT = Path('reports/rebaseline/cold-rerun-2026-10-15')
LOCAL_NAME = 'cold-rerun-2026-10-15'
PROJECTION_NAME = 'cost-projection-v21-r16.json'
RUN_EVIDENCE = Path('docs/track-b/evidence/cp-26') / LOCAL_NAME
PROTOCOL = Path('docs/track-b/evidence/cp-26/continuation-v21-r14/execution-protocol.json')


def code_identity(root: Path) -> dict:
    root = Path(root)
    paths = [*sorted((root / 'src/cp26').glob('*.py')),
             *sorted((root / 'scripts').glob('cp26*.py')),
             *sorted((root / 'tests').glob('test_cp26*.py'))]
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def protocol_identity(root: Path) -> dict:
    root = Path(root)
    raw = (root / PROTOCOL).read_bytes()
    protocol = json.loads(raw)
    H.continuation_authority(root)
    if protocol.get('schema') != 'cp26-execution-protocol-v21-r14' or protocol.get('authority_sha256') != H.PINS[H.SCIENTIFIC_AUTHORITY_PATH]:
        raise RuntimeError('execution protocol does not bind the Owner continuation authority')
    committed = subprocess.check_output(['git', 'show', f'HEAD:{PROTOCOL}'], cwd=root)
    if raw != committed:
        raise RuntimeError('execution details must be frozen in a local commit before compute')
    if protocol.get('research_sha256') != H.SCIENTIFIC_ANCHOR_SHA256:
        raise RuntimeError('execution protocol names a different research anchor')
    registration_path = RUN_EVIDENCE / 'registration.json'
    registration_raw = (root / registration_path).read_bytes()
    registration = json.loads(registration_raw)
    brief_path = RUN_EVIDENCE / 'issued-brief.md'
    for path in (registration_path, brief_path):
        if subprocess.check_output(['git', 'show', f'HEAD:{path}'], cwd=root) != (root / path).read_bytes():
            raise RuntimeError('cold-run authority must be committed before compute')
    if (registration['run_id'] != RUN_ID or registration['output'] != str(OUTPUT)
            or registration['local'] != str(H.art() / LOCAL_NAME)
            or registration['initial_run_machine_hours'] != 0
            or registration['reuse_old_scientific_state'] is not False
            or hashlib.sha256((root / brief_path).read_bytes()).hexdigest() != registration['issued_brief_sha256']):
        raise RuntimeError('cold-run authority/namespace identity differs')
    return {'schema': protocol['schema'], 'document': str(PROTOCOL),
            'document_sha256': hashlib.sha256(raw).hexdigest(),
            'authority_sha256': H.PINS[H.SCIENTIFIC_AUTHORITY_PATH],
            'research_sha256': protocol['research_sha256'],
            'cp28_protocol_sha256': H.PINS['docs/track-b/evidence/cp-26/cp28-frozen-protocol-v21-r14.md'],
            'run_id': RUN_ID, 'run_authority_document': str(registration_path),
            'run_authority_sha256': hashlib.sha256(registration_raw).hexdigest()}


class ExecutionGate:
    def __init__(self, root: Path, *, budget=None):
        self.root = Path(root)
        self.budget = budget or Budget()

    def __call__(self, stage: str) -> None:
        if stage not in ('prepare', 'pilot', 'search', 'forecast', 'score', 'report'):
            raise ValueError('unregistered CP-26 execution stage')
        H.continuation_authority(self.root)
        H.check_blas()
        if os.environ.get('CP26_RUN_ID') != RUN_ID:
            raise RuntimeError('cold execution requires its registered CP26_RUN_ID')
        if not os.environ.get('CP26_MONITORED_JOB'):
            raise RuntimeError('CP-26 execution stages require cumulative supervision')
        from .supervisor import reconcile_running, LEDGER_TIMEOUT_SECONDS
        def admit_resources(state):
            reconcile_running(state)
            require_compute_ready(state)
            if not state.get('continuation_scope') or state['continuation_scope']['authority_sha256'] != H.PINS[H.SCIENTIFIC_AUTHORITY_PATH]:
                raise RuntimeError('continuation accounting is not bound to the Owner grant')
        self.budget.resource_update(admit_resources, timeout_seconds=LEDGER_TIMEOUT_SECONDS)
        H.runtime_lineage(self.root)
        protocol_identity(self.root)
        if stage == 'prepare':
            return
        tests_path = self.root / OUTPUT / 'preparation-tests-v21-r16.json'
        tested = json.loads(tests_path.read_text())
        if (tested.get('status') != 'PASS' or tested.get('code_sha256') != code_identity(self.root)
                or tested.get('protocol') != protocol_identity(self.root)):
            raise RuntimeError('preparation tests are missing or stale for this code/protocol')
        job_index = tested['job_index']
        state = self.budget.read()
        job = state['jobs'][job_index]
        if job.get('job_id') != tested['job_id'] or job['running'] or job.get('exit_code') != 0 or job.get('abort_reason'):
            raise RuntimeError('preparation test receipt is not a successful supervised completion')
        from .parity import require_pass
        require_pass(self.root, output=OUTPUT / 'parity.json')
        if stage == 'pilot':
            return
        projection = json.loads((self.root / OUTPUT / PROJECTION_NAME).read_text())
        from .costing import _verify, SCHEMA
        _verify(projection, SCHEMA)
        from .budget import total_seconds
        baseline = projection.get('ledger_seconds_at_pricing')
        if (not isinstance(baseline, (int, float)) or isinstance(baseline, bool)
                or not math.isfinite(baseline) or baseline < 0 or total_seconds(self.budget.read()) < baseline):
            raise RuntimeError('projection ledger baseline regressed or is invalid')
        if projection.get('protocol') != protocol_identity(self.root) or projection.get('code_sha256') != code_identity(self.root):
            raise RuntimeError('pilot projection identity changed; refuse full execution')
        pilot = json.loads((self.root / OUTPUT / 'timing-pilot.json').read_text())
        if projection.get('pilot_sha256') != digest(pilot):
            raise RuntimeError('projection does not bind the retained timing pilot')
        reconcile_terminal_evidence(self.root, self.root / OUTPUT, projection['protocol'], self.budget)
        # Reprice against the live cumulative ledger, including all charges since
        # the pilot. This executable stop applies before searches and forecasts.
        searches, remaining = outstanding_projection(projection, self.root / OUTPUT, self.budget.read())
        current = projected_cost(self.budget.read(), searches, remaining, run_id=RUN_ID)
        if not current['within_pilot_stop']:
            raise CapExceeded(f"CP-26 projection {current['projected_total_hours']:.6f} hours exceeds "
                              'the 440-hour cumulative admission line; Owner resource/protocol decision required')
        if stage in ('score', 'report'):
            from .execution import _read, Paths, calendar, manifest_for, _validate_state_calendar
            from .state import ReplayState
            completion = _read(self.root / OUTPUT / 'forecast-completion.json')
            shared = protocol_identity(self.root)
            paths = Paths.build(self.root)
            manifest = manifest_for(paths, shared)
            saved = ReplayState.load(self.root / OUTPUT / 'replay-state.json', shared)
            _validate_state_calendar(saved)
            inventory = {p.stem for p in (self.root / OUTPUT / 'forecast-ledger').glob('*.json')}
            if (completion.get('schema') != 'cp26-forecast-completion-v1'
                    or completion.get('protocol') != shared or completion.get('status') != 'COMPLETE'
                    or completion.get('origin_count') != 2441 or completion.get('last_origin') != str(H.NEW_BOUNDARY)
                    or saved.last_origin != H.NEW_BOUNDARY
                    or completion.get('replay_state_sha256') != saved.to_dict()['content_sha256']
                    or completion.get('execution_manifest_sha256') != manifest['content_sha256']
                    or inventory != {str(d) for d in calendar()}
                    or remaining['daily_fits'].get('completed_origins') != 2441):
                raise RuntimeError('scoring requires the complete frozen forecast ledger')


def authorize_consuming_process(root: Path) -> dict:
    authority = H.continuation_authority(root)
    os.environ['CP26_EXPERIMENT_AUTHORIZED'] = '1'
    return authority


def outstanding_projection(projection: dict, output: Path, state: dict) -> tuple[list, dict]:
    """Keep cumulative actual charges and remove only evidenced completed estimates.

    A completed search still appears in the 42-row identity inventory, carrying
    its original positive price and an explicit completed marker. The pricing
    helper recognizes that marker rather than removing a row or charging twice.
    """
    searches = copy.deepcopy(projection['searches'])
    from .execution import Paths, manifest_for, terminal_search, _read, DAY_SCHEMA, calendar
    from .members import RUN_MEMBERS
    from .replay import POLICIES
    paths = Paths.build(Path(output).parents[2], output)
    manifest = manifest_for(paths, projection['protocol'])
    tasks = {task['id']: task for task in state.get('execution_tasks', [])}
    for row in searches:
        result_path = Path(output) / 'search-results' / row['run'] / f"k{row['k']:02d}.json"
        if result_path.exists():
            item = terminal_search(paths, row['run'], row['k'], manifest)
            if item.get('plan_sha256') != row.get('plan_sha256'):
                raise RuntimeError('completed search cost has no valid terminal evidence')
            task_id = f"search/{digest(projection['protocol'])}/{row['run']}/{row['k']}"
            task = tasks.get(task_id)
            if task and task['evidence'] != {'result_sha256': item['content_sha256'], 'status': item['status']}:
                raise RuntimeError('completed search cost disagrees with its ledger task')
            if task:
                row['completed_evidence'] = {'result_sha256': item['content_sha256'], 'path': str(result_path),
                                             'task_id': task_id}
    remaining = copy.deepcopy(projection['remaining'])
    ledger = Path(output) / 'forecast-ledger'
    completed_days = 0
    if ledger.exists():
        permitted = {str(day) for day in calendar()}
        expected_policies = {f'{r}/{p}' for r in POLICIES for p in POLICIES[r]}
        for path in sorted(ledger.glob('*.json')):
            item = _read(path)
            if (item.get('schema') != DAY_SCHEMA or item.get('day') != path.stem or path.stem not in permitted
                    or item.get('protocol') != projection['protocol']
                    or item.get('execution_manifest_sha256') != manifest['content_sha256']
                    or set(item.get('policies', {})) != expected_policies
                    or set(item.get('members', {})) != set(RUN_MEMBERS)
                    or any(set(item['members'][r]) != set(ms) for r, ms in RUN_MEMBERS.items())):
                raise RuntimeError('forecast progress is incomplete or belongs to a different protocol/calendar')
            task_id = f"issuance/{digest(projection['protocol'])}/{path.stem}"
            task = tasks.get(task_id)
            if task and task['evidence'] != {'ledger_sha256': item['content_sha256']}:
                raise RuntimeError('completed forecast cost disagrees with its ledger task')
            if task:
                completed_days += 1
    daily = remaining['daily_fits']
    if completed_days:
        if completed_days > daily['total_origins']:
            raise RuntimeError('completed forecast days exceed the frozen calendar')
        daily['machine_seconds'] *= (daily['total_origins'] - completed_days) / daily['total_origins']
        daily['completed_origins'] = completed_days
    _discount_completed_analysis(remaining, projection, paths.root, output, state)
    return searches, remaining


def _discount_completed_analysis(remaining, projection, root, output, state):
    """Actual analysis duration stays charged; only proven future work is removed."""
    from . import analysis as A
    tasks = {task['id']: task for task in state.get('execution_tasks', [])}
    code = code_identity(root)
    for stage, category, schema in (('inference', 'bootstrap', A.INFERENCE_SCHEMA),
                                    ('diagnostics', 'diagnostics', A.DIAGNOSTIC_SCHEMA)):
        path = Path(output) / f'{stage}-manifest.json'
        task_id = f"analysis/{digest(projection['protocol'])}/{stage}"
        task = tasks.get(task_id)
        if not path.exists():
            if task:
                raise RuntimeError('completed analysis task lost its manifest')
            continue
        value = A._read(path, schema)
        if (value.get('status') != 'COMPLETE' or value.get('protocol') != projection['protocol']
                or value.get('code_sha256') != code):
            raise RuntimeError('completed analysis cost has stale protocol/code evidence')
        scores = A._read(Path(output) / 'scores-manifest.json', A.SCORE_SCHEMA)
        if (scores.get('protocol') != projection['protocol'] or scores.get('code_sha256') != code
                or value.get('scores_manifest_sha256') != scores['content_sha256']):
            raise RuntimeError('completed analysis cost has stale scores evidence')
        evidence = {'manifest_sha256': value['content_sha256'], 'status': value['status'],
                    'code_sha256': digest(code), 'scores_manifest_sha256': scores['content_sha256']}
        if task and task.get('evidence') != evidence:
            raise RuntimeError('completed analysis cost disagrees with its ledger task')
        if task:
            remaining[category]['machine_seconds'] = 0.0
            remaining[category]['completed_evidence'] = {'task_id': task_id, **evidence}


def reconcile_terminal_evidence(root, output, protocol, budget):
    """Close artifact-before-counter crash seams without fitting or scoring."""
    from . import analysis as A, execution as E
    from .fits import FitCache
    from .state import ReplayState
    paths = E.Paths.build(root, output)
    manifest = E.manifest_for(paths, protocol)
    recorded = {row['id'] for row in budget.read().get('execution_tasks', [])}
    for k in range(14):
        for route in ('nominal', 'noload', 'gfs6'):
            task_id = f'search/{digest(protocol)}/{route}/{k}'
            if paths.result(route, k).exists() and task_id not in recorded:
                result = E.terminal_search(paths, route, k, manifest)
                E.record_evidence_once(budget, task_id, {'result_sha256': result['content_sha256'],
                    'status': result['status']}, {'completed_search_results': 1,
                    'unavailable_search_results': int(result['status'] == 'unavailable')})
    state_path = paths.state(protocol)
    if not state_path.exists():
        state_path = Path(output) / 'replay-state.json'
    if state_path.exists():
        saved = ReplayState.load(state_path, protocol)
        E._validate_state_calendar(saved)
        for path in sorted((Path(output) / 'forecast-ledger').glob('*.json')):
            task_id = f'issuance/{digest(protocol)}/{path.stem}'
            if task_id in recorded:
                continue
            value = E._read(path)
            if (value.get('schema') != E.DAY_SCHEMA or value.get('day') != path.stem
                    or value.get('protocol') != protocol or value.get('execution_manifest_sha256') != manifest['content_sha256']
                    or value.get('policies') != {key: policy.records.get(path.stem) for key, policy in saved.policies.items()}):
                raise RuntimeError('unrecorded forecast evidence differs from saved issuance')
            day = H.date.fromisoformat(path.stem)
            for route, members in E.RUN_MEMBERS.items():
                row = value['policies'][f'{route}/{E.POLICIES[route][0]}']
                cache = FitCache(paths.local / 'fits', route, value['lineage'][route]['fit_identity'])
                k = E.S.serving_search(day)
                ensemble = E.terminal_search(paths, route, k, manifest) if k is not None else None
                for member in members:
                    result = cache.load(day, member, row['index'], row['issue_utc'], ensemble=ensemble)
                    if (result is None or E._plain(result) != value['members'][route][member]
                            or result['cache_content_sha256'] != row['provenance']['member_cache_sha256'][member]):
                        raise RuntimeError('unrecorded forecast native evidence differs from its cache')
            E.record_evidence_once(budget, task_id, {'ledger_sha256': value['content_sha256']},
                                   {'completed_daily_issuances': 1})
    code = code_identity(root)
    for stage, schema in (('inference', A.INFERENCE_SCHEMA), ('diagnostics', A.DIAGNOSTIC_SCHEMA)):
        path = Path(output) / f'{stage}-manifest.json'
        task_id = f'analysis/{digest(protocol)}/{stage}'
        if path.exists() and task_id not in recorded:
            scores = A._read(Path(output) / 'scores-manifest.json', A.SCORE_SCHEMA)
            value = A._validate_completed(root, stage, A._read(path, schema), protocol, scores, code)
            E.record_evidence_once(budget, task_id, {'manifest_sha256': value['content_sha256'],
                'status': value['status'], 'code_sha256': digest(code),
                'scores_manifest_sha256': scores['content_sha256']})
