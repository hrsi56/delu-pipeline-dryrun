"""CP-23 research jobs, dispatched by `scripts/cp23_ddnn.py job <name>` under the monitor.

The job table names every module up front, so modules written after the pre-run freeze (controls,
diagnostics, the report, local tracking, the packet) never change a frozen file. None of those
later modules produces or alters a forecast.
"""
from __future__ import annotations

import importlib
import json
import os
from pathlib import Path
import time

JOBS = {
    'reference-checks': ('reference', 'job_reference_checks'),
    'resource-admission': ('admission', 'job_resource_admission'),
    'verify-inputs': ('preflight', 'verify_inputs'),
    'verify-weather': ('preflight', 'verify_weather'),
    'e1': ('preflight', 'enumerate_e1'),
    'protocol': ('protocol', 'job_protocol'),
    'select': ('execution', 'job_select'),
    'fits': ('execution', 'job_fits'),
    'admission': ('execution', 'job_admission'),
    'comparison': ('execution', 'job_comparison'),
    'parity': ('execution', 'job_parity'),
    'score': ('evaluate', 'job_score'),
    'controls': ('controls', 'job_controls'),
    'reproduce': ('controls', 'job_reproduce'),
    'daily-cycle': ('daily', 'job_daily_cycle'),
    'fit-cost': ('daily', 'job_fit_cost'),
    'diagnostics': ('diagnostics', 'job_diagnostics'),
    'finalise': ('finalise', 'job_finalise'),
    'mlflow-local': ('tracking', 'job_mlflow_local'),
}


def _project() -> Path:
    return Path(os.environ.get('CP23_PROJECT_ROOT', '/Users/djourno/Downloads/PJM'))


def art() -> Path:
    return _project() / '.local' / 'artifacts' / 'cp-23'


def cp21_art() -> Path:
    return _project() / '.local' / 'artifacts' / 'cp-21'


def cp20_art() -> Path:
    return _project() / '.local' / 'artifacts' / 'cp-20'


def dispatch(root: Path, name: str, rest: list[str]) -> int:
    if name not in JOBS:
        raise SystemExit(f'unknown CP-23 job {name!r}')
    module, function = JOBS[name]
    return getattr(importlib.import_module(f'cp23.{module}'), function)(Path(root), rest)


def write_json(path: Path, obj) -> None:
    from .budget import atomic
    atomic(path, obj)


def stamp() -> str:
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
