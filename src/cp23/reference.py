"""The PyTorch correctness reference for DDNN (capstone v21-r10 §21.3, §18.3), and its recorded run.

**Tolerances.** They are frozen here, committed before the first comparison run, and copied into the
pre-run protocol. A tolerance changes only through a recorded protocol change, never silently after a
failure (§18.3). Every comparison is elementwise, `|numpy - torch| <= atol + rtol * |torch|`, in
float64.

**The checks** (`tests/cp23/torch_reference_checks.py`, on fixed synthetic inputs and seeds, for every
frozen configuration):

* the forward pass: raw outputs and the four Johnson SU parameters;
* the JSU log-likelihood, from an independent change-of-variables density whose Jacobian comes from
  autograd, and its partial derivatives;
* the full-loss gradient of every weight and bias against autograd;
* the quantile function against `torch.special.ndtri`, and the normal CDF at each quantile;
* short optimizer trajectories: 25 Adam steps on fixed mini-batches against `torch.optim.Adam`, and
  three epochs of `train_member` (initialisation, permutations, early-stopping bookkeeping) against a
  PyTorch replica.

**How it runs, so that it cannot be skipped silently.**

* PyTorch is pinned in the separate test-only lock `tests/cp23/torch-reference/`. It is never a
  runtime dependency.
* The check file's name does not match pytest's `test_*.py`, so the default suite never collects it.
  It imports `torch` at module level, with no skip logic: a missing PyTorch is a collection error.
* The `reference-checks` job runs it under the monitor and records the run, the versions and the
  result in the ledger. The step fails unless every expected check passed, with zero skipped,
  deselected or errored.
* `require_passing_record()` refuses every DDNN fit unless a passing record exists for the current
  `src/cp23/ddnn.py` bytes. The default suite asserts the committed record
  (`tests/cp23/test_reference_record.py`).
"""
from __future__ import annotations

import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys

from cp15.data import sha
from .budget import atomic, ledger
from .jobs import art, stamp

REFERENCE_TOLERANCES = {
    'forward_outputs': {'rtol': 1e-10, 'atol': 1e-12},
    'jsu_parameters': {'rtol': 1e-10, 'atol': 1e-12},
    'log_likelihood': {'rtol': 1e-10, 'atol': 1e-12},
    'log_likelihood_partials': {'rtol': 1e-9, 'atol': 1e-12},
    'network_gradients': {'rtol': 1e-8, 'atol': 1e-11},
    'quantiles': {'rtol': 1e-10, 'atol': 1e-12},
    'cdf_at_quantiles': {'rtol': 0.0, 'atol': 1e-12},
    'adam_trajectory': {'rtol': 1e-7, 'atol': 1e-10},
    'training_loop_trajectory': {'rtol': 1e-7, 'atol': 1e-10},
}
TRAJECTORY_STEPS = 25
TRAINING_LOOP_EPOCHS = 3
CHECK_FILE = 'tests/cp23/torch_reference_checks.py'
DDNN_FILE = 'src/cp23/ddnn.py'
RECORD = Path('reports/distribution-challenger/reference-checks.json')
#: The checks the file must report as passed: 6 per frozen configuration plus 2 configuration-free.
EXPECTED_PASSED = 6 * 4 + 2


def versions() -> dict:
    out = {'python': sys.version.split()[0], 'platform': platform.platform()}
    for name in ('torch', 'numpy'):
        try:
            out[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            out[name] = None
    return out


def _counts(text: str) -> dict:
    tail = text.strip().splitlines()[-1] if text.strip() else ''
    found = {k: int(v) for v, k in re.findall(r'(\d+) (passed|failed|skipped|deselected|errors?|xfailed|xpassed)', tail)}
    if 'error' in found:
        found['errors'] = found.pop('error')
    return {'summary': tail, **{k: found.get(k, 0) for k in ('passed', 'failed', 'skipped', 'deselected', 'errors',
                                                              'xfailed', 'xpassed')}}


def job_reference_checks(root: Path, rest) -> int:
    root = Path(root)
    budget = ledger()
    budget.reserve(reference_check_runs=1)
    errors_path = art() / 'reference' / f'max-errors-{stamp()}.json'
    env = {**os.environ, 'CP23_REFERENCE_ERRORS': str(errors_path)}
    command = [sys.executable, '-m', 'pytest', '-q', '-p', 'no:cacheprovider', '-rA', CHECK_FILE]
    run = subprocess.run(command, cwd=root, capture_output=True, text=True, env=env)
    counts = _counts(run.stdout)
    max_errors = json.loads(errors_path.read_text()) if errors_path.exists() else None
    passed = bool(run.returncode == 0 and counts['passed'] == EXPECTED_PASSED and not any(
        counts[k] for k in ('failed', 'skipped', 'deselected', 'errors', 'xfailed', 'xpassed')))
    record = {'schema': 'cp23-reference-checks-v1', 'written_utc': stamp(), 'passed': passed,
              'command': 'python -m pytest -q -p no:cacheprovider -rA ' + CHECK_FILE, 'exit_code': run.returncode,
              'counts': counts, 'expected_passed': EXPECTED_PASSED, 'versions': versions(),
              'ddnn_sha256': sha(root / DDNN_FILE), 'check_file_sha256': sha(root / CHECK_FILE),
              'reference_module_sha256': sha(root / 'src/cp23/reference.py'),
              'torch_lock_sha256': sha(root / 'tests/cp23/torch-reference/uv.lock'),
              'tolerances': REFERENCE_TOLERANCES, 'trajectory_steps': TRAJECTORY_STEPS,
              'training_loop_epochs': TRAINING_LOOP_EPOCHS, 'max_observed_errors': max_errors,
              'device': 'cpu (float64); no GPU or MPS',
              'stdout_tail': run.stdout.strip().splitlines()[-40:], 'stderr_tail': run.stderr.strip().splitlines()[-20:]}
    history = art() / 'reference' / 'runs.jsonl'
    history.parent.mkdir(parents=True, exist_ok=True)
    with history.open('a') as handle:
        handle.write(json.dumps(record, sort_keys=True) + '\n')
    budget.event('reference_checks', passed=passed, counts=counts, versions=record['versions'],
                 ddnn_sha256=record['ddnn_sha256'], check_file_sha256=record['check_file_sha256'])
    atomic(root / RECORD, record)
    print(json.dumps({k: record[k] for k in ('passed', 'counts', 'versions', 'ddnn_sha256')}), flush=True)
    return 0 if passed else 8


def require_passing_record(root: Path) -> dict:
    """Refuse DDNN results unless the reference passed for the current DDNN implementation (§21.3)."""
    root = Path(root)
    path = root / RECORD
    if not path.exists():
        raise RuntimeError('no PyTorch reference record: run the reference-checks job first (§21.3)')
    record = json.loads(path.read_text())
    if not record.get('passed'):
        raise RuntimeError('the PyTorch reference checks failed: every DDNN result is blocked until fixed (§18.3)')
    if record['ddnn_sha256'] != sha(root / DDNN_FILE):
        raise RuntimeError('src/cp23/ddnn.py changed since its passing reference record: rerun the reference checks')
    if record['tolerances'] != REFERENCE_TOLERANCES:
        raise RuntimeError('reference tolerances differ from the recorded run')
    return record
