"""CP-24 research jobs, dispatched by `scripts/cp24_ddnn2.py job <name>` under the monitor.

The job table names every module up front. Modules written after an attempt's freeze (controls,
diagnostics, the report, local tracking, the packet) never change a frozen file, and none of them
produces or alters a forecast.
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
    'reference-checks': ('reference', 'job_reference_checks'),
    'resource-admission': ('admission', 'job_resource_admission'),
    'search': ('search', 'job_search'),
    'search-ledger': ('searchledger', 'job_search_ledger'),
    'v4-gate': ('gate', 'job_v4_gate'),
    'v4-parity': ('gate', 'job_v4_parity'),
    'gate': ('gate', 'job_gate'),
    'round-design': ('rounds', 'job_round_design'),
    'round-report': ('rounds', 'job_round_report'),
    'protocol': ('protocol', 'job_protocol'),
    'fits': ('execution', 'job_fits'),
    'admission': ('execution', 'job_admission'),
    'comparison': ('execution', 'job_comparison'),
    'score': ('evaluate', 'job_score'),
    'controls': ('controls', 'job_controls'),
    'daily-cycle': ('daily', 'job_daily_cycle'),
    'fit-cost': ('daily', 'job_fit_cost'),
    'diagnostics': ('diagnostics', 'job_diagnostics'),
    'finalise': ('finalise', 'job_finalise'),
    'mlflow-local': ('tracking', 'job_mlflow_local'),
}


def _project() -> Path:
    return Path(os.environ.get('CP24_PROJECT_ROOT', '/Users/djourno/Downloads/PJM'))


def art() -> Path:
    return _project() / '.local' / 'artifacts' / 'cp-24'


def cp23_art() -> Path:
    return _project() / '.local' / 'artifacts' / 'cp-23'


def cp21_art() -> Path:
    return _project() / '.local' / 'artifacts' / 'cp-21'


def cp20_art() -> Path:
    return _project() / '.local' / 'artifacts' / 'cp-20'


def dispatch(root: Path, name: str, rest: list[str]) -> int:
    if name not in JOBS:
        raise SystemExit(f'unknown CP-24 job {name!r}')
    module, function = JOBS[name]
    return getattr(importlib.import_module(f'cp24.{module}'), function)(Path(root), rest)


def write_json(path: Path, obj) -> None:
    from .budget import atomic
    atomic(path, obj)


def stamp() -> str:
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())


def read_json(path: Path):
    return json.loads(Path(path).read_text())
