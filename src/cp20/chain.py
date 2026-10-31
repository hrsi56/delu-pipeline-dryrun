"""Unattended CP-20 extraction chain (Owner standing instruction S1, 2026-09-23).

Phases run strictly one after another, each a monitored single-endpoint job (plan r11,
Owner 2026-09-24; the first chain ran NCAR 3 workers then AWS 1 worker):
  1. AWS alone, all runs except the days that had already failed, 4 workers;
  2. retry AWS: every still-incomplete AWS run (the earlier failed days included), 4 workers;
  3. NCAR re-run: every still-incomplete NCAR run (locator-defect days, verified on five
     days first by the extractor), 4 workers.
Resumable: completed runs are revalidated and reused by every phase.
A phase continues the chain only on exit 0 (complete) or 5 (some runs failed after their
bounded counted attempts). Any other end (stop request, contradiction, 60 minutes without
data, a cap or stop line) stops the chain and is written to ``chain-status.json``. Network
problems are handled inside the fetcher and never stop the chain.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
P = Path('/Users/djourno/Downloads/PJM')
A = P / '.local/artifacts/cp-20'
PY, WX = P / '.venv/bin/python', A / 'wx-venv/bin/python'
CONTINUE = {0, 5}
STOP_MACHINE_HOURS = '100'  # Owner 2026-09-24: extraction stop line 100 within the 120 cap


def status(**fields):
    path = A / 'chain-status.json'
    state = json.loads(path.read_text()) if path.exists() else {'phases': []}
    state.update(fields)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(state, indent=1))
    os.replace(tmp, path)
    return state


def phase(name, endpoint, workers, select='all', exclude=''):
    cmd = [str(PY), 'scripts/cp20_weather.py', '--monitor', '--name', 'gfs-extract', '--workers', str(workers),
           '--stop-machine-hours', STOP_MACHINE_HOURS, '--log', str(A / 'logs/gfs-extract.log'), '--',
           str(WX), '-u', '-m', 'cp20.extract', '--manifest', 'reports/weather-ablation/run-manifest.json',
           '--out', str(A / 'weather'), '--admission', 'reports/weather-admission', '--select', select,
           '--endpoint', endpoint, '--workers', str(workers), '--stop-transfer-gib', '150', '--inventory-added']
    if exclude:
        cmd += ['--exclude', exclude]
    began = time.time()
    state = status(current=name)
    state['phases'].append({'phase': name, 'endpoint': endpoint, 'workers': workers, 'start_epoch': began})
    status(phases=state['phases'])
    code = subprocess.call(cmd, cwd=ROOT)
    marker = json.loads((A / 'markers' / ('gfs-extract.DONE.json' if code == 0 else 'gfs-extract.FAILED.json')).read_text())
    versions = A / 'weather/job-code-versions.jsonl'
    ran = [json.loads(l) for l in versions.read_text().splitlines()] if versions.exists() else []
    ran = [r for r in ran if r['start_epoch'] >= began]
    state = status()
    state['phases'][-1].update(end_epoch=time.time(), monitor_exit=code, marker=marker,
                               code_version=ran[0]['code_version'] if ran else 'not recorded (pre-r10 extractor)')
    status(phases=state['phases'])
    return code, marker


def failed_days():
    path = A / 'weather/failures.jsonl'
    if not path.exists():
        return []
    runs = {json.loads(l)['run'] for l in path.read_text().splitlines()}
    return sorted(r for r in runs if not (A / f'weather/runs/{r}.json').exists())


def main():
    caffeinate = subprocess.Popen(['/usr/bin/caffeinate', '-i', '-w', str(os.getpid())])
    previous = A / 'chain-status.json'
    if previous.exists():  # keep every earlier chain's record
        k = 1
        while (A / f'chain-status.run{k}.json').exists():
            k += 1
        previous.rename(A / f'chain-status.run{k}.json')
    deferred = ','.join(failed_days())
    aws_deferred = ','.join(d for d in deferred.split(',') if d >= '2021-01-01' and d != '2021-02-02')
    status(started_epoch=time.time(), pid=os.getpid(), deferred_failed_days=aws_deferred, result=None, plan='r11')
    plan = [('aws-all', 'aws', 4, 'all', aws_deferred), ('retry-aws', 'aws', 4, 'all', ''),
            ('retry-ncar', 'ncar', 4, 'all', '')]
    for name, endpoint, workers, select, exclude in plan:
        code, marker = phase(name, endpoint, workers, select, exclude)
        child = (marker.get('abort_reason'), marker.get('exit_code'))
        if code not in CONTINUE and marker.get('exit_code') not in CONTINUE:
            status(result='stopped', stopped_in=name, reason=child, end_epoch=time.time())
            caffeinate.terminate()
            return 1
    remaining = [r for r in json.loads((ROOT / 'reports/weather-ablation/run-manifest.json').read_text())['runs']
                 if not (A / f'weather/runs/{r["run_00z"]}.json').exists()]
    status(result='complete' if not remaining else 'finished_with_incomplete_runs',
           incomplete_runs=[r['run_00z'] for r in remaining], end_epoch=time.time())
    caffeinate.terminate()
    return 0


if __name__ == '__main__':
    sys.exit(main())
