"""The PyTorch correctness reference for DDNN-2 (capstone v21-r11 §23.7; §18.3, §21.3), and its run.

**Tolerances.** Frozen here, committed before the first comparison run, and repeated in every frozen
protocol. A tolerance changes only through a recorded protocol change, never silently after a failure
(§18.3). Every comparison is elementwise, `|numpy - torch| <= atol + rtol * |torch|`, in float64.

**The checks** (`tests/cp24/torch_reference_checks.py`, on fixed synthetic inputs and seeds):

* the forward pass, for every activation in the space, one and two hidden layers, with fixed
  input-dropout masks: the 96 raw outputs and the 24 x 4 Johnson SU parameters;
* the masked, recency-weighted 24-slot Johnson SU loss, from an independent change-of-variables
  density with an autograd Jacobian, and its gradient for every weight and bias against autograd;
* the pinball loss over the 19-level grid through the Johnson SU quantile function, and its gradient;
* the full loss -- kappa * NLL + (1 - kappa) * pinball + L1 + L2, with dropout -- and its gradient,
  for every activation and every kappa in the space;
* the quantile function at the seven levels against `torch.special.ndtri`, every target transform's
  inverse, and the cap;
* short optimizer trajectories: 25 Adam steps on fixed mini-batches against `torch.optim.Adam`, and
  four epochs of `cp24.ddnn2.train` (initialisation, permutations, dropout masks, the warm start,
  the stopping metric and the best-epoch bookkeeping) against a PyTorch replica.

**How it runs, so that it cannot be skipped silently.** PyTorch is CP-23's test-only reference lock
`tests/cp23/torch-reference/` (SHA-256 b3164a37...), never a runtime dependency; the root
`pyproject.toml` and `uv.lock` never change (§23.7). The check file's name does not match pytest's
`test_*.py`, so the default suite (and CI, which installs only the root lock) never collects it; it
imports `torch` at module level with no skip logic, so a missing PyTorch is a collection error. The
`reference-checks` job runs it under the monitor and records the run, the versions and the result in
the ledger and in `reports/ddnn2/reference-checks.json`; the step fails unless every expected check
passed, with zero skipped, deselected or errored. `require_passing_record()` refuses every DDNN-2 fit
unless a passing record exists for the current `src/cp24/ddnn2.py` bytes, and the default suite
asserts the committed record (`tests/cp24/test_reference_record.py`).
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
    'nll_loss': {'rtol': 1e-10, 'atol': 1e-12},
    'pinball_loss': {'rtol': 1e-10, 'atol': 1e-12},
    'full_loss': {'rtol': 1e-10, 'atol': 1e-12},
    'loss_gradients': {'rtol': 1e-8, 'atol': 1e-11},
    'quantiles': {'rtol': 1e-10, 'atol': 1e-12},
    'adam_trajectory': {'rtol': 1e-7, 'atol': 1e-10},
    'training_loop_trajectory': {'rtol': 1e-7, 'atol': 1e-10},
    'stopping_metric': {'rtol': 1e-8, 'atol': 1e-10},
}
TRAJECTORY_STEPS = 25
TRAINING_LOOP_EPOCHS = 4
CHECK_FILE = 'tests/cp24/torch_reference_checks.py'
MODEL_FILE = 'src/cp24/ddnn2.py'
LOCK_FILE = 'tests/cp23/torch-reference/uv.lock'
LOCK_SHA256 = 'b3164a375e871396e685d0e887972b092fcd0c24a7fe0047167e8fc41c25dfed'
PINNED_TORCH = '2.14.1'
RECORD = Path('reports/ddnn2/reference-checks.json')
#: The checks the file must report as passed (counted from its parametrisation).
EXPECTED_PASSED = 4 * 2 + 4 + 4 + 4 * 3 + 1 + 1 + 1 + 1


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
    env = {**os.environ, 'CP24_REFERENCE_ERRORS': str(errors_path)}
    command = [sys.executable, '-m', 'pytest', '-q', '-p', 'no:cacheprovider', '-rA', CHECK_FILE]
    run = subprocess.run(command, cwd=root, capture_output=True, text=True, env=env)
    counts = _counts(run.stdout)
    max_errors = json.loads(errors_path.read_text()) if errors_path.exists() else None
    passed = bool(run.returncode == 0 and counts['passed'] == EXPECTED_PASSED and not any(
        counts[k] for k in ('failed', 'skipped', 'deselected', 'errors', 'xfailed', 'xpassed')))
    record = {'schema': 'cp24-reference-checks-v1', 'written_utc': stamp(), 'passed': passed,
              'command': 'python -m pytest -q -p no:cacheprovider -rA ' + CHECK_FILE, 'exit_code': run.returncode,
              'counts': counts, 'expected_passed': EXPECTED_PASSED, 'versions': versions(),
              'model_sha256': sha(root / MODEL_FILE), 'check_file_sha256': sha(root / CHECK_FILE),
              'reference_module_sha256': sha(root / 'src/cp24/reference.py'), 'torch_lock_sha256': sha(root / LOCK_FILE),
              'tolerances': REFERENCE_TOLERANCES, 'trajectory_steps': TRAJECTORY_STEPS,
              'training_loop_epochs': TRAINING_LOOP_EPOCHS, 'max_observed_errors': max_errors,
              'device': 'cpu (float64); no GPU or MPS',
              'stdout_tail': run.stdout.strip().splitlines()[-60:], 'stderr_tail': run.stderr.strip().splitlines()[-20:]}
    history = art() / 'reference' / 'runs.jsonl'
    history.parent.mkdir(parents=True, exist_ok=True)
    with history.open('a') as handle:
        handle.write(json.dumps(record, sort_keys=True) + '\n')
    budget.event('reference_checks', passed=passed, counts=counts, versions=record['versions'],
                 model_sha256=record['model_sha256'], check_file_sha256=record['check_file_sha256'])
    atomic(root / RECORD, record)
    print(json.dumps({k: record[k] for k in ('passed', 'counts', 'versions', 'model_sha256')}), flush=True)
    return 0 if passed else 8


def require_passing_record(root: Path) -> dict:
    """Refuse DDNN-2 results unless the reference passed for the current model code (§23.7, §18.3)."""
    root = Path(root)
    path = root / RECORD
    if not path.exists():
        raise RuntimeError('no PyTorch reference record: run the reference-checks job first (§23.7)')
    record = json.loads(path.read_text())
    if not record.get('passed'):
        raise RuntimeError('the PyTorch reference checks failed: every DDNN-2 result is blocked until fixed (§18.3)')
    if record['model_sha256'] != sha(root / MODEL_FILE):
        raise RuntimeError('src/cp24/ddnn2.py changed since its passing reference record: rerun the reference checks')
    if record['tolerances'] != REFERENCE_TOLERANCES:
        raise RuntimeError('reference tolerances differ from the recorded run')
    return record
