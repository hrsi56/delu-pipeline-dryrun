"""CP-21 research jobs, dispatched by `scripts/cp21_blocks.py job <name>` under the monitor.

The job table names every module up front, so modules written after the pre-run freeze
(controls, the fit-cost and daily-cycle diagnostic, the report, local tracking) never change a
frozen file. None of those later modules produces or alters a forecast.
"""
from __future__ import annotations

import importlib
import json
import os
from pathlib import Path
import time

JOBS = {
    'verify-inputs': ('preflight', 'verify_inputs'),
    'verify-weather': ('preflight', 'verify_weather'),
    'benchmark': ('preflight', 'benchmark'),
    'determinism': ('preflight', 'determinism'),
    'e1': ('preflight', 'enumerate_e1'),
    'protocol': ('protocol', 'job_protocol'),
    'fits': ('execution', 'job_fits'),
    'admission': ('execution', 'job_admission'),
    'comparison': ('execution', 'job_comparison'),
    'hg-parity': ('execution', 'job_hg_parity'),
    'score': ('evaluate', 'job_score'),
    'controls': ('controls', 'job_controls'),
    'daily-cycle': ('daily', 'job_daily_cycle'),
    'fit-cost': ('daily', 'job_fit_cost'),
    'finalise': ('finalise', 'job_finalise'),
    'mlflow-local': ('tracking', 'job_mlflow_local'),
}


def _project() -> Path:
    return Path(os.environ.get('CP21_PROJECT_ROOT', '/Users/djourno/Downloads/PJM'))


def art() -> Path:
    return _project() / '.local' / 'artifacts' / 'cp-21'


def dispatch(root: Path, name: str, rest: list[str]) -> int:
    if name not in JOBS:
        raise SystemExit(f'unknown CP-21 job {name!r}')
    module, function = JOBS[name]
    return getattr(importlib.import_module(f'cp21.{module}'), function)(Path(root), rest)


def write_json(path: Path, obj) -> None:
    from .budget import atomic
    atomic(path, obj)


def stamp() -> str:
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())


def dumps(obj) -> str:
    return json.dumps(obj, indent=1, sort_keys=True, allow_nan=False, default=str)
