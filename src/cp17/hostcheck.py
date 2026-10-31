"""The checks before each run's first issuance (r19 §24.13, H2), frozen with the policy identity.

They run once per run, on the runner or on the Owner's Mac, in this order, and again whenever the machine or the
runtime changes within a run:

1. **The pins.** `python -m cp17.tools.refreeze verify` against CP-17c's manifest must report no problem and no
   unchecked item. It exits 0 in both cases, so its output is read: only the single line `verified` passes.
2. **The binary.** The tarball and `bin/node` must match their pinned bytes and hashes (`cp17.runtime`).
3. **The machine.** macOS on ARM64. On a GitHub-hosted runner, `ImageOS` must be `macos26`. The image version, CPU
   model, cores, memory and OS version are recorded, never pinned. A machine that passes is not a runtime change.
4. **A canary fit.** A1 at the freeze origin, 2026-09-06, reissued under the host in its own fresh process and judged
   as any attempt (`cp17.issue.attempt`). Its quantiles must reproduce the frozen SHA-256 `CANARY_QUANTILES_SHA256`.

**Any failure means nothing in the run is issued.** The first failed check stops the gate, no later check runs,
and `run` never calls the issuance. For 4.7T, a failure is a blocker, never a day's outcome.

The record of each run's checks holds the machine's image version and CPU model, and is the run's lineage record
beside its issued artifacts (§24.13: "in the artifact's lineage, or in CP-18's records").
"""
from __future__ import annotations

from datetime import date, datetime, timezone
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

from . import runtime

ORDER = ('pins', 'binary', 'machine', 'canary')
MANIFEST = 'docs/track-b/evidence/cp-17c/freeze-manifest.json'
CANARY_DAY = date(2026, 9, 6)
CANARY_STEM = f'A1-{CANARY_DAY}'
CANARY_QUANTILES_SHA256 = '4e6abb02fbb02aa8ef202207f21b76bc33b34c37d025a0a93a491135ea43cea5'
SCHEMA = 'cp17c-host-checks-v1'
RUNNER_ENV = ('ImageOS', 'ImageVersion', 'RUNNER_OS', 'RUNNER_ARCH', 'RUNNER_NAME', 'RUNNER_ENVIRONMENT',
              'GITHUB_ACTIONS', 'GITHUB_RUN_ID', 'GITHUB_RUN_ATTEMPT', 'GITHUB_JOB', 'GITHUB_SHA', 'GITHUB_REPOSITORY')


def _result(name: str, problems: list[str], started: float, **record) -> dict:
    return {'check': name, 'passed': not problems, 'problems': problems, 'seconds': time.perf_counter() - started,
            **record}


def pins(root, manifest: str = MANIFEST, python: str = sys.executable) -> dict:
    """Check 1: `refreeze verify`'s output, read; anything but `verified` fails."""
    started = time.perf_counter()
    env = {**os.environ, 'PYTHONPATH': str(Path(root) / 'src')}
    try:
        done = subprocess.run([python, '-m', 'cp17.tools.refreeze', 'verify', manifest, '--root', str(root)],
                              capture_output=True, text=True, cwd=root, env=env, timeout=900)
        output, code = done.stdout.strip(), done.returncode
    except (OSError, subprocess.SubprocessError) as exc:
        output, code = f'{type(exc).__name__}: {exc}', None
    problems = [] if code == 0 and output == 'verified' else \
        [f'refreeze verify (exit {code}) did not report exactly "verified": ' + ' | '.join(output.splitlines()[:20])]
    return _result('pins', problems, started, manifest=manifest, exit=code, output=output.splitlines()[:50])


def binary(project) -> dict:
    """Check 2: the tarball and `bin/node` by their pinned bytes and hashes."""
    started = time.perf_counter()
    found = runtime.check_binary(project)
    return _result('binary', found['problems'], started, tarball=found['tarball'], node=found['node'],
                   pins={'tarball': runtime.TARBALL, 'binary': runtime.BINARY})


def _sysctl(name: str) -> str:
    try:
        return subprocess.run(['sysctl', '-n', name], capture_output=True, text=True, timeout=30).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ''


def describe(env=None) -> dict:
    """The machine as recorded (not pinned): OS, architecture, CPU model, cores, memory, and a runner's image."""
    env = os.environ if env is None else env
    out = {'system': platform.system(), 'machine': platform.machine(), 'platform': platform.platform(),
           'cpu_count': os.cpu_count(), 'runner_env': {n: env.get(n) for n in RUNNER_ENV}}
    if out['system'] == 'Darwin':
        out.update(cpu_model=_sysctl('machdep.cpu.brand_string'), hardware_model=_sysctl('hw.model'),
                   physical_cores=int(_sysctl('hw.physicalcpu') or 0), memory_bytes=int(_sysctl('hw.memsize') or 0))
        try:
            out['os_version'] = subprocess.run(['sw_vers', '-productVersion'], capture_output=True, text=True,
                                               timeout=30).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            out['os_version'] = ''
    return out


def machine(env=None, described: dict | None = None) -> dict:
    """Check 3: macOS on ARM64; on a GitHub-hosted runner, `ImageOS` is `macos26`."""
    started = time.perf_counter()
    env = os.environ if env is None else env
    found = describe(env) if described is None else described
    problems = []
    if found['system'] != runtime.MACHINE['system'] or found['machine'] != runtime.MACHINE['machine']:
        problems.append(f"the machine is {found['system']} on {found['machine']}, not macOS on ARM64")
    runner = env.get('GITHUB_ACTIONS') == 'true'
    if runner and env.get('ImageOS') != runtime.MACHINE['image_os']:
        problems.append(f"the runner's ImageOS is {env.get('ImageOS')!r}, not {runtime.MACHINE['image_os']!r}")
    return _result('machine', problems, started, kind='GitHub-hosted runner' if runner else 'local Mac',
                   image_version=env.get('ImageVersion') if runner else None, described=found)


def canary(host, payloads, out, expected: str = CANARY_QUANTILES_SHA256) -> dict:
    """Check 4: A1 at the freeze origin under the host, judged as an attempt, its quantiles equal to the frozen
    hash. `host` is a `cp17.daily.Host`; `payloads` holds the canary's payload."""
    from . import dailyfit, issue
    started = time.perf_counter()
    payloads = Path(payloads)
    npz, meta = payloads / f'{CANARY_STEM}.npz', payloads / f'{CANARY_STEM}.json'
    runs = {}
    try:
        arrays, value = dailyfit.load(npz if npz.exists() else None, meta)
    except Exception as exc:            # an absent or unreadable canary payload fails the check
        return _result('canary', [f'the canary payload is unreadable ({type(exc).__name__}: {exc})'], started,
                       expected_quantiles_sha256=expected)

    def fit():
        run = runs['run'] = host.run(payloads, CANARY_STEM, Path(out))
        if run.get('errors'):
            raise dailyfit.FitFailure(run['errors'][0][-2000:])
        return run['python']['result']

    def measured():
        return {'wall_seconds': runs['run']['total_ms'] / 1000, 'peak_bytes': int(runs['run']['process_rss_peak_bytes'])}
    item = issue.attempt('A1', fit, CANARY_DAY, arrays, value, resources=measured)
    result = item.pop('result', None) or {}
    item.pop('traceback', None)
    actual = result.get('quantiles_sha256')
    problems = []
    if item['status'] != 'ok':
        problems.append(f"the canary fit failed: {item['cause']}")
    elif actual != expected:
        problems.append(f'the canary quantiles are {actual}, not the frozen {expected}')
    run = runs.get('run', {})
    return _result('canary', problems, started, expected_quantiles_sha256=expected, quantiles_sha256=actual,
                   payload_sha256=value.get('payload_sha256'), attempt=item, node=run.get('node'),
                   node_execpath=run.get('node_execpath'), pyodide=run.get('pyodide_version'),
                   executed_files=(run.get('python') or {}).get('source_sha256'), folder=run.get('folder'))


def gate(checks: dict) -> dict:
    """Run `checks[name]()` in `ORDER`, stopping at the first failure: the later checks never run."""
    started = datetime.now(timezone.utc).isoformat(timespec='seconds')
    done = []
    for name in ORDER:
        try:
            result = checks[name]()
        except Exception as exc:        # a check that cannot complete has failed
            result = {'check': name, 'passed': False, 'problems': [f'{type(exc).__name__}: {exc}']}
        done.append(result)
        if not result['passed']:
            break
    failed = next((r['check'] for r in done if not r['passed']), None)
    return {'schema': SCHEMA, 'started_utc': started, 'order': list(ORDER), 'checks': done, 'passed': failed is None,
            'failed': failed, 'not_run': list(ORDER[len(done):]), 'python': platform.python_version(),
            'consequence': 'the run may issue' if failed is None else 'nothing in this run is issued'}


def run(root, project, host, canary_payloads, out, issue_run, env=None, manifest: str = MANIFEST) -> dict:
    """The checks, then `issue_run()` only if every one passed. Returns {'checks', 'issued'}."""
    checks = {'pins': lambda: pins(root, manifest),
              'binary': lambda: binary(project),
              'machine': lambda: machine(env),
              'canary': lambda: canary(host, canary_payloads, out)}
    record = gate(checks)
    return {'checks': record, 'issued': issue_run() if record['passed'] else None}
