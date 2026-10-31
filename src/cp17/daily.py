"""The daily issuance on the issuing host: the frozen chain, the shadow issuances, and the issued artifact.

This is the driver CP-18's pipeline and 4.7T's product run call once per delivery day, after the day's payloads
are built. It runs natively, because it starts the issuing host, and it is part of the frozen policy identity
(CP-17's review, N1): its hash is frozen beside the package's (`cp17.tools.bundle.PIPELINE`).

- **Each attempt** runs the package's own fit (`host/driver.py`, fit mode) in a fresh Pyodide process under the
  pinned Node.js, with a fresh output path, so no earlier record can be read as its result, and a
  timeout equal to the success definition's wall-time ceiling. A process that is killed, times out, exits
  non-zero, dies without a record, or leaves an unreadable record or one of another run is a failed fit, and the
  chain moves on. The process's own wall time and peak resident memory judge criterion 5 (`cp17.issue.attempt`).
- **The chain** is `cp17.issue.decide`, with the artifact built and verified inside it (criterion 4, N3). A
  failed issuance is written and verified too.
- **The shadows** (decision 14). After the chain, v4, v2 and B0 are fitted for the same day under the issuing
  host, whether or not A1 succeeded; a fallback the chain already fitted is not fitted twice. A shadow passes the
  same input checks as an attempt (criterion 1); a day with no forecastable hour fits nothing, and on a day whose
  GFS is late or missing at the issue time v4's shadow is skipped, since its inputs are unavailable. Each fit
  is registered in its own state before the outcome (`cp17.shadow.ShadowStates.issue`), under CP-26's intake
  rule. A shadow is never published, displayed or scored as the product's forecast.
- **The native comparison** (`comparison_day`) runs the same chain in-process, with criterion 5 not judged. It is
  labelled as the comparison and can never issue (N2).

**The binary (CP-17c, r19 §24.13).** The default is the official Node.js v26.7.0 darwin-arm64 binary at its
project-relative path (`cp17.runtime.NODE_RELATIVE`), resolved against the project root when a `Host` is made, in
place of Homebrew's. A run retrieves it and checks it by hash (`cp17.runtime.retrieve`), and checks the host before
its first issuance (`cp17.hostcheck`).
"""
from __future__ import annotations

from datetime import date
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time

import numpy as np

from . import dailyfit, issue, runtime
from .shadow import POLICIES as SHADOW_POLICIES

NODE = runtime.NODE_RELATIVE
RUNNER = Path(__file__).resolve().parent / 'host' / 'node_run.mjs'
#: The pinned Pyodide 0.28.0 packages the package's import closure needs (the lock adds their dependencies).
PACKAGES = ('numpy', 'pandas', 'scikit-learn', 'lightgbm', 'requests', 'tzdata')
TIMEOUT_SECONDS = issue.WALL_LIMIT_SECONDS
SCHEMA = 'cp17b-daily-issuance-v1'


def _failed(stem: str, errors: list[str], started: float, folder: Path, returncode=None) -> dict:
    return {'target': stem, 'errors': errors, 'total_ms': (time.perf_counter() - started) * 1000,
            'process_rss_peak_bytes': None, 'folder': str(folder), 'returncode': returncode,
            'blocked_non_local_requests': []}


class Host:
    """The issuing host: the pinned Node.js, the pinned Pyodide folder and the frozen package's zip."""

    def __init__(self, pyodide: Path, package_zip: Path, node: str = NODE, timeout: float = TIMEOUT_SECONDS,
                 runner: Path = RUNNER):
        self.pyodide, self.package_zip = Path(pyodide), Path(package_zip)
        self.node = node if Path(node).is_absolute() else str(runtime.project_root() / node)
        self.timeout, self.runner = timeout, Path(runner)

    def command(self, payloads: Path, stem: str, target: Path) -> list[str]:
        return [self.node, str(self.runner), 'fit', str(self.pyodide), str(self.package_zip), str(payloads), stem, '1',
                ','.join(PACKAGES), str(target)]

    def run(self, payloads: Path, stem: str, out: Path) -> dict:
        """One attempt in a fresh process, with a fresh output path. Never raises: a failure is in `errors`."""
        out.mkdir(parents=True, exist_ok=True)
        folder = Path(tempfile.mkdtemp(prefix=f'{stem}-', dir=out))
        target = folder / 'run.json'
        started = time.perf_counter()
        try:
            proc = subprocess.run(self.command(Path(payloads), stem, target), capture_output=True, text=True,
                                  timeout=self.timeout, env={**os.environ, 'CP17_EXPORT': '1'})
        except subprocess.TimeoutExpired:
            return _failed(stem, [f'the issuing-host process exceeded the {self.timeout} s timeout and was killed'],
                           started, folder)
        except OSError as exc:
            return _failed(stem, [f'the issuing-host process could not start ({type(exc).__name__})'], started, folder)
        if not target.exists():
            return _failed(stem, [f'the issuing-host process left no record (exit {proc.returncode})'], started, folder,
                           proc.returncode)
        try:
            run = json.loads(target.read_text())
        except (OSError, ValueError) as exc:
            return _failed(stem, [f'the issuing-host process left an unreadable record ({type(exc).__name__})'], started,
                           folder, proc.returncode)
        problems = record_problems(run, stem)
        if not problems and proc.returncode != 0 and not run.get('errors'):
            problems = [f'the issuing-host process exited {proc.returncode}']
        if problems:
            return _failed(stem, problems, started, folder, proc.returncode)
        run.update(folder=str(folder), returncode=proc.returncode,
                   driver_wall_ms=(time.perf_counter() - started) * 1000)
        return run


def record_problems(run, stem: str) -> list[str]:
    """Whether a record is this attempt's own, complete record (an attempt's errors are its own, not these)."""
    if not isinstance(run, dict):
        return ['the issuing-host record is not an object']
    if run.get('target') != stem or run.get('mode') != 'fit':
        return ['the issuing-host record is of another run']
    if not isinstance(run.get('total_ms'), (int, float)) or not isinstance(run.get('process_rss_peak_bytes'), int):
        return ['the issuing-host record lacks its measurement']
    if not run.get('errors'):
        result = (run.get('python') or {}).get('result')
        policy, day = stem.split('-', 1)
        if not isinstance(result, dict) or result.get('policy') != policy or result.get('day') != day:
            return ["the issuing-host record does not hold this attempt's fit"]
    return []


def _paths(payloads: Path, policy: str, day: date) -> tuple[Path | None, Path]:
    npz = payloads / f'{policy}-{day}.npz'
    return (npz if npz.exists() else None), payloads / f'{policy}-{day}.json'


def _hash(values) -> str | None:
    return None if values is None else dailyfit.array_hash(np.asarray(values, float))


def issue_day(day: date, gfs_available: bool, payloads: Path, *, host: Host, out: Path, identity: dict,
              parent_sha256: str | None = None, data_refresh_utc: str = '', states=None) -> dict:
    """The day's issuance under the issuing host, then the shadows, registered in `states` if given.

    `identity` is {'policy_identity_sha256', 'package_sha256'}, the frozen identity the artifact cites."""
    payloads, out = Path(payloads), Path(out)
    started = time.perf_counter()
    runs, inputs, loaded = {}, {}, {}

    def load(policy):
        if policy not in loaded:
            npz, meta = _paths(payloads, policy, day)
            loaded[policy] = dailyfit.load(npz, meta)
            inputs[policy] = {'payload_sha256': loaded[policy][1].get('payload_sha256')}
        return loaded[policy]

    def host_run(policy):
        if policy not in runs:
            runs[policy] = host.run(payloads, f'{policy}-{day}', out)
        return runs[policy]

    def measured(policy):
        run = runs[policy]
        return {'wall_seconds': run['total_ms'] / 1000, 'peak_bytes': int(run['process_rss_peak_bytes'])}

    def attempts(policy):
        arrays, value = load(policy)

        def fit():
            run = host_run(policy)
            if run.get('errors'):
                raise dailyfit.FitFailure(run['errors'][0][-2000:])
            return run['python']['result']
        return issue.attempt(policy, fit, day, arrays, value, resources=lambda: measured(policy))

    def finalize(decision):
        policy = decision['issued_policy']
        run = runs.get(policy) or next(iter(runs.values()), {})
        host_identity = {'runtime': 'pyodide', 'pyodide': run.get('pyodide_version'), 'node': run.get('node'),
                         'process': 'one fresh process per attempt'}
        return issue.artifact(decision, identity=identity, payloads=inputs, parent_sha256=parent_sha256,
                              data_refresh_utc=data_refresh_utc, host=host_identity)

    decision = issue.decide(day, gfs_available, attempts, finalize)
    issued_seconds = time.perf_counter() - started
    shadows = {}
    for policy in SHADOW_POLICIES:
        shadow = {'role': 'shadow: never published, displayed or scored as the product\'s forecast'}
        chained = next((a for a in decision['attempts'] if a['policy'] == policy and a['status'] != 'skipped'), None)
        if policy == 'v4' and not gfs_available:
            shadow.update(status='skipped', cause='GFS late or missing at the issue time', result=None)
        elif not _paths(payloads, policy, day)[1].exists():
            shadow.update(status='failed', cause='inputs: no payload', result=None)
        else:
            arrays, value = load(policy)
            problems = issue.check_inputs(policy, arrays, value)
            if problems:
                shadow.update(status='failed', cause='inputs: ' + '; '.join(problems), result=None)
            else:
                run = host_run(policy)
                result = None if run.get('errors') else run['python']['result']
                shadow.update(status='failed' if result is None else 'ok', cause=(run.get('errors') or [None])[0],
                              result=result, wall_seconds=run['total_ms'] / 1000,
                              peak_bytes=run.get('process_rss_peak_bytes'))
        shadow['fitted_by'] = 'chain' if chained is not None and policy in runs else 'shadow'
        result = shadow.pop('result')
        if result is not None:
            shadow.update(keys=len(result['index']), quantiles_sha256=result['quantiles_sha256'],
                          central_sha256=_hash(result['central']), scale_sha256=_hash(result.get('scale')))
        if states is not None and shadow['status'] != 'skipped':
            shadow['intake'] = states.issue(policy, day, result)
        shadows[policy] = shadow
    total = time.perf_counter() - started
    process_seconds = {p: r['total_ms'] / 1000 for p, r in runs.items()}
    return {'schema': SCHEMA, 'day': str(day), 'gfs_available': gfs_available, 'decision': _strip(decision),
            'artifact': decision.get('artifact'), 'shadows': shadows, 'runs': runs,
            'issued_seconds': issued_seconds, 'total_seconds': total, 'process_seconds': process_seconds,
            'process_seconds_sum': sum(process_seconds.values()),
            'blocked_non_local_requests': sorted({u for r in runs.values() for u in r.get('blocked_non_local_requests') or []})}


def _strip(decision: dict) -> dict:
    return {**{k: v for k, v in decision.items() if k not in ('attempts', 'artifact')},
            'attempts': [{k: v for k, v in a.items() if k not in ('result', 'traceback')} for a in decision['attempts']]}


def comparison_day(day: date, gfs_available: bool, payloads: Path, states=None, attribution: bool = True) -> dict:
    """The same chain natively, in-process, with criterion 5 not judged: labelled as the comparison, it never
    issues. With `states`, the shadows are fitted natively too and registered there."""
    payloads = Path(payloads)
    results = {}

    def native(policy):
        if policy not in results:
            npz, meta = _paths(payloads, policy, day)
            try:
                results[policy] = dailyfit.run(policy, npz, meta, attribution=attribution)
            except Exception as exc:     # recorded as the fit's cause, as `issue.attempt` records it
                results[policy] = exc
        if isinstance(results[policy], Exception):
            raise results[policy]
        return results[policy]

    def attempts(policy):
        npz, meta = _paths(payloads, policy, day)
        arrays, value = dailyfit.load(npz, meta)
        return issue.attempt(policy, lambda: native(policy), day, arrays, value)
    started = time.perf_counter()
    decision = issue.decide(day, gfs_available, attempts)
    out = {'comparison': 'native, in-process; criterion 5 not judged; never issues', 'decision': decision,
           'seconds': time.perf_counter() - started}
    if states is not None:
        shadows = {}
        for policy in SHADOW_POLICIES:
            if policy == 'v4' and not gfs_available:
                shadows[policy] = {'status': 'skipped', 'cause': 'GFS late or missing at the issue time'}
                continue
            npz, meta = _paths(payloads, policy, day)
            arrays, value = dailyfit.load(npz, meta)
            problems = issue.check_inputs(policy, arrays, value)
            result = None
            if not problems:
                try:
                    result = native(policy)
                except Exception as exc:
                    problems = [f'{type(exc).__name__}: {exc}']
            shadows[policy] = {'status': 'ok' if result is not None else 'failed', 'cause': '; '.join(problems) or None,
                               'quantiles_sha256': result['quantiles_sha256'] if result else None,
                               'central_sha256': _hash(result['central']) if result else None,
                               'intake': states.issue(policy, day, result)}
        out['shadows'] = shadows
    out['results'] = results
    return out
