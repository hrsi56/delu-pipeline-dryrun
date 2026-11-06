"""The local fallback on the Owner's Mac: the same commands as the runner, the same binary, checks and claim rule.

    # before D's cutoff (the agent com.delu.daily-acquire): D's inputs, its tier-3 load forecast above all
    PYTHONPATH=<clone>/src .venv/bin/python -m cp18.local acquire --clone <clone> --project /Users/djourno/Downloads/PJM \
        --out <dir> [--window-hours 3]
    # after the cutoff (the agent com.delu.daily-run): the live run, then the day's bundle, Space, publication, records
    PYTHONPATH=<clone>/src .venv/bin/python -m cp18.local run --clone <clone> --project /Users/djourno/Downloads/PJM \
        --out <dir> --mode live
    # a dry run on development days (CP-18a's), with the post-run sequence against a local static target if asked
    PYTHONPATH=<clone>/src .venv/bin/python -m cp18.local run --clone <clone> --project <folder holding .local> \
        --out <dir> --now <ISO> --inputs <spec> [--publish local:<folder>]

- **The same binary:** the official Node.js v26.7.0 at `.local/tools/node-v26.7.0-official/` in the project
  (`cp17.runtime.NODE_RELATIVE`), checked by hash in place; retrieved by hash only if it is absent.
- **The same checks:** the package zip is rebuilt from `src/` and compared with the manifest first; the H2 gate
  (`cp17.hostcheck`: pins, binary, machine, canary) runs in the pipeline before the first issuance. The project is named
  to every stage explicitly: `CP17_PROJECT_ROOT` (the pins check's `.local/`) and `CP26_PROJECT_ROOT` (`cp18.live.gfs`'s
  ecCodes environment, `<project>/.local/artifacts/cp-20/wx-venv`) are exported before anything runs, because the
  pipeline's checkout is the dedicated clone, not the project.
- **The same claim rule:** the clone is the pipeline's own checkout of the product repository, with `core.hooksPath`
  set to its `.githooks`; the day's record on `main` is the claim (`cp18.claim`), so the runner and the fallback never
  issue the same day twice, and the loser of a race discards its own issuance.
- **The same sequence as the runner's workflow** (`.github/workflows/daily.yml`), one stage per process:
  1. `python -m cp18 run --mode live --inputs live-http` (or a dry run's `--now` and `--inputs`);
  2. if the as-of day has no bundle record yet and the store has an artifact to serve (`bundle_plan`): the dated
     bundle `python -m cp18.bundle` with the served artifact's arrays from this run's attempt folder, or `--refit`;
  3. if the bundle is publishable: the Space build `scripts/build_product_space.py`, then the publication
     `python -m cp18.publish` (to the Hugging Face Space in a live run; to a local static target in a rehearsal);
  4. the day's data-only records (`live/outcomes/`, `live/evaluation/`, `live/bundles/`, and `live/store/` and
     `live/inputs/` if anything is left) committed and pushed through the claim (`records`).
- **Credentials, per stage, as on the runner:** each stage is a child process whose environment holds only the
  credential it uses: `ENTSOE_API_TOKEN` for the run and the acquisition, `HF_TOKEN` for the publication, none for the
  bundle, the Space build or the records. A LaunchAgent inherits the launchd session's environment, where the Owner's
  `~/.zshrc` exports are mirrored (AGENTS.md § Credentials); nothing here prints, logs or writes a value.
- **The clone** is dedicated to the pipeline (never the Owner's working checkout): created by `git clone` if absent,
  and never written by anything else. Its uncommitted changes refuse the run (`cp18.claim`).

The agents (`reports/cp18c/launchd/`) are written, not installed: installing them is the Orchestrator's step after the
LAND, if the runner cannot run.

**Helpers the runner's workflow calls too**, so both hosts decide the same way:

    python -m cp18.local bundle-plan --root . --runs <run out>/runs --now <ISO>        # key=value lines
    python -m cp18.local records --root . --remote origin --run-id <id> --out <file>
    python -m cp18.local wx-env --python <wx-venv>/bin/python --freeze <requirements.freeze.txt> --out <file>
    python -m cp18.local decode-check --run <run day> --python <wx python> --archived <decoded runs> --work <dir> \
        --out <file>
    python -m cp18.local scan --name ENTSOE_API_TOKEN --name HF_TOKEN --out <file> <results>   # by value, names only
"""
from __future__ import annotations

import argparse
from datetime import date, datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

PRODUCT_URL = 'https://github.com/hrsi56/delu-day-ahead-forecast.git'
HUB = 'Yarden-Viktor/delu-day-ahead-forecast'
WX_PYTHON = '.local/artifacts/cp-20/wx-venv/bin/python'          # cp18.live.gfs.WX_PYTHON, relative to the project
WX_FREEZE = 'reports/weather-admission/scripts/requirements.freeze.txt'
SPACE_HISTORY = 'reports/cp18c/history-v1/files'                # v1's deployed files, served unchanged at /history/
DATA_PATHS = ('live/store', 'live/inputs', 'live/outcomes', 'live/evaluation', 'live/bundles')
#: The stored credentials (AGENTS.md § Credentials); any other variable named like one is dropped from a stage too.
CREDENTIALS = ('ENTSOE_API_TOKEN', 'HF_TOKEN', 'DAGSHUB_USER_TOKEN', 'MLFLOW_TRACKING_USERNAME',
               'MLFLOW_TRACKING_PASSWORD')
CREDENTIAL_NAME = re.compile(r'TOKEN|SECRET|PASSWORD|PASSWD|API_?KEY|ACCESS_KEY|PRIVATE_KEY', re.I)
SRC = Path(__file__).resolve().parents[1]
UTC = timezone.utc


def utc_now() -> str:
    return datetime.now(UTC).strftime('%Y-%m-%dT%H:%M:%SZ')


def dump(path: Path, value) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=1, sort_keys=True, default=str) + '\n')


def pin_project(project: Path) -> dict:
    """Name the project to every stage: `CP17_PROJECT_ROOT` for the H2 pins check and the issuing host's `.local/`,
    `CP26_PROJECT_ROOT` for `cp18.live.gfs`'s ecCodes environment. Set before anything runs, unconditionally."""
    os.environ['CP17_PROJECT_ROOT'] = str(project)
    os.environ['CP26_PROJECT_ROOT'] = str(project)
    return {'CP17_PROJECT_ROOT': str(project), 'CP26_PROJECT_ROOT': str(project)}


def stage_env(*keep: str) -> dict:
    """This process's environment for one stage: every credential dropped except the ones named in `keep`, and the
    same code on `PYTHONPATH` as this command's (the clone's `src/` under the agents)."""
    env = {k: v for k, v in os.environ.items()
           if k in keep or (k not in CREDENTIALS and not CREDENTIAL_NAME.search(k))}
    env['PYTHONPATH'] = str(SRC)
    return env


def stage(name: str, args: list[str], *, keep: tuple[str, ...] = (), cwd: Path | None = None) -> dict:
    """One stage, a child process of this interpreter; its output goes to this command's own (the agent's log)."""
    started = time.perf_counter()
    print(f'local fallback: {name}', flush=True)
    done = subprocess.run([sys.executable, *args], env=stage_env(*keep), cwd=str(cwd) if cwd else None)
    return {'stage': name, 'exit': done.returncode, 'seconds': round(time.perf_counter() - started, 1),
            'credentials': sorted(keep)}


def ensure_clone(clone: Path, url: str | None) -> dict:
    """The pipeline's own clone, with the repository's hooks active."""
    created = False
    if not (clone / '.git').exists():
        if not url:
            raise SystemExit(f'{clone}: no clone there, and no --url to create it from')
        subprocess.run(['git', 'clone', '--quiet', url, str(clone)], check=True)
        created = True
    subprocess.run(['git', '-C', str(clone), 'config', 'core.hooksPath', '.githooks'], check=True)
    return {'clone': str(clone), 'created': created}


# ------------------------------------------------------------------------------------------------ the ecCodes env
_WX_PROBE = r'''
import hashlib, importlib.metadata as md, json, platform, sys
skip = {"RECORD", "INSTALLER", "REQUESTED", "direct_url.json", "uv_cache.json"}
dists = {}
for d in md.distributions():
    name = d.metadata["Name"].lower().replace("_", "-")
    lines = sorted(f"{f.as_posix()},{f.hash.value if f.hash else ''}" for f in (d.files or ())
                   if not f.as_posix().startswith("..") and "__pycache__" not in f.as_posix() and f.name not in skip
                   and not f.as_posix().endswith(".pyc"))
    dists[name] = {"version": d.version, "files": len(lines),
                   "content_sha256": hashlib.sha256("\n".join(lines).encode()).hexdigest()}
out = {"python": platform.python_version(), "implementation": sys.implementation.name, "platform": platform.platform(),
       "machine": platform.machine(), "distributions": dict(sorted(dists.items()))}
try:
    import eccodes, findlibs
    lib = findlibs.find("eccodes")
    out["eccodes"] = {"module_version": eccodes.__version__, "api_version": eccodes.codes_get_api_version(),
                      "library": lib, "library_sha256": hashlib.sha256(open(lib, "rb").read()).hexdigest() if lib else None}
except Exception as exc:
    out["eccodes"] = {"error": f"{type(exc).__name__}: {exc}"}
print(json.dumps(out, sort_keys=True))
'''


def pins(freeze: Path) -> dict[str, str]:
    """{name: version} from a `name==version` requirements file (CP-20's `requirements.freeze.txt`)."""
    out = {}
    for line in Path(freeze).read_text().splitlines():
        line = line.split('#', 1)[0].strip()
        if line:
            name, _, version = line.partition('==')
            out[name.strip().lower().replace('_', '-')] = version.strip()
    return out


def wx_environment(python: Path, freeze: Path | None = None) -> dict:
    """The ecCodes environment `cp18.live.gfs` decodes in: each distribution's version and a content digest (its
    installed files' own hashes, from RECORD, outside the venv's scripts), the ecCodes library's SHA-256, and the
    comparison with the pinned versions. The probe runs with an environment built from scratch (no credential)."""
    python = Path(python)
    record = {'python_path': str(python), 'present': python.exists(), 'freeze': str(freeze) if freeze else None}
    if not python.exists():
        record['problems'] = [f'the ecCodes environment is absent: {python}']
        return record
    env = {'PATH': '/usr/bin:/bin', 'HOME': os.environ.get('HOME', '/tmp'), 'PYTHONNOUSERSITE': '1'}
    done = subprocess.run([str(python), '-I', '-c', _WX_PROBE], capture_output=True, text=True, env=env, timeout=300)
    if done.returncode != 0:
        record['problems'] = [f'the probe failed (exit {done.returncode}): {done.stderr.strip()[-400:]}']
        return record
    record.update(json.loads(done.stdout.strip().splitlines()[-1]))
    problems = []
    if freeze:
        wanted, have = pins(freeze), {k: v['version'] for k, v in record['distributions'].items()}
        record['pinned'] = wanted
        record['missing'] = sorted(k for k in wanted if k not in have)
        record['different'] = {k: {'pinned': wanted[k], 'installed': have[k]} for k in sorted(wanted)
                               if k in have and have[k] != wanted[k]}
        record['extra'] = sorted(k for k in have if k not in wanted)
        problems += [f'{k}: pinned, not installed' for k in record['missing']]
        problems += [f"{k}: {v['installed']} installed, {v['pinned']} pinned" for k, v in record['different'].items()]
    if 'error' in record.get('eccodes', {}):
        problems.append(f"ecCodes does not load: {record['eccodes']['error']}")
    record['problems'] = problems
    return record


# ------------------------------------------------------------------------------------------------ the decode check
def decode_check(run: date, python: Path, archived: Path, work: Path, transport=None) -> dict:
    """One development-day GFS run retrieved from the AWS archive through `cp18.live.gfs` (anonymous, under the
    development guard: the run serves a day through 2026-09-06), decoded by CP-20's decoder in the ecCodes environment
    at `python`, and compared with the archived decode of the same run (`<archived>/runs/<run>.{json,npz}`): every
    message's bytes by SHA-256 and byte range, its decoded metadata, and the decoded arrays exactly."""
    import numpy as np
    from cp25 import guard
    from .live import gfs
    from .live.transport import HttpTransport
    work = Path(work)
    record = {'schema': 'cp18c-decode-check-v1', 'run': str(run), 'delivery_day': str(run + timedelta(days=1)),
              'python': str(python), 'started_utc': utc_now(), 'archived': str(archived)}
    try:
        guard.admit_delivery_day(run + timedelta(days=1))          # the development guard, never the live route
        got = gfs.retrieve_run(transport or HttpTransport(), run, work / 'raw')
        record['retrieval'] = {'messages': len(got.messages), 'problems': got.problems, 'complete': got.complete,
                               'bytes': sum(m['bytes'] for m in got.messages),
                               'listing': {k: {'object_last_modified': (v.get('object') or {}).get('last_modified'),
                                               'object_size': (v.get('object') or {}).get('size')}
                                           for k, v in sorted(got.listing.items())}}
        if not got.complete:
            record.update(equal=False, outcome='the run could not be retrieved completely')
            return record
        summary = gfs.decode_run(run, run + timedelta(days=1), got.messages, work / 'raw', work / 'decoded',
                                 python=Path(python))
        record['decoder'] = summary.get('decoder')
        mine_js = json.loads((work / 'decoded/runs' / f'{run}.json').read_text())
        ref_js = json.loads((Path(archived) / 'runs' / f'{run}.json').read_text())
        key = lambda m: (m['lead'], m['field'])                     # noqa: E731
        mine_m, ref_m = {key(m): m for m in mine_js['messages']}, {key(m): m for m in ref_js['messages']}
        messages = {'compared': len(ref_m), 'same_set': set(mine_m) == set(ref_m),
                    'sha256_equal': sum(mine_m[k]['sha256'] == ref_m[k]['sha256'] for k in ref_m if k in mine_m),
                    'byte_range_equal': sum(mine_m[k]['byte_range'] == ref_m[k]['byte_range'] for k in ref_m if k in mine_m),
                    'meta_equal': sum(mine_m[k].get('meta') == ref_m[k].get('meta') for k in ref_m if k in mine_m),
                    'packing_quantum_equal': sum(mine_m[k].get('packing_quantum') == ref_m[k].get('packing_quantum')
                                                 for k in ref_m if k in mine_m)}
        with np.load(work / 'decoded/runs' / f'{run}.npz') as a, np.load(Path(archived) / 'runs' / f'{run}.npz') as b:
            arrays = {name: bool(np.array_equal(a[name], b[name], equal_nan=a[name].dtype.kind == 'f'))
                      for name in ('data', 'lats', 'lons', 'fields', 'leads')}
            diff = float(np.nanmax(np.abs(a['data'] - b['data']))) if a['data'].shape == b['data'].shape else None
        record.update(messages=messages, arrays=arrays, max_abs_difference=diff,
                      npz_sha256={'decoded': mine_js['npz_sha256'], 'archived': ref_js['npz_sha256'],
                                  'equal': mine_js['npz_sha256'] == ref_js['npz_sha256']},
                      status={'decoded': mine_js['status'], 'archived': ref_js['status']})
        equal = (messages['same_set'] and all(messages[k] == messages['compared'] for k in
                                              ('sha256_equal', 'byte_range_equal', 'meta_equal', 'packing_quantum_equal'))
                 and all(arrays.values()))
        record.update(equal=equal, outcome=('the decode equals the archived decode: every message byte for byte, its '
                                            'metadata, and the arrays exactly') if equal else
                      'the decode differs from the archived decode')
    except Exception as exc:  # noqa: BLE001 - the check's outcome, never an escaped run
        record.update(equal=False, outcome=f'failed: {guard.redact(f"{type(exc).__name__}: {exc}")}')
    record['finished_utc'] = utc_now()
    return record


# ------------------------------------------------------------------------------------------------ the scan
def scan(names: list[str], paths: list[Path]) -> dict:
    """Which credentials' values occur in which result files, by value (raw, and URL- and base64-encoded): names and
    paths only. Each value is read with `os.environ.get` in this process and never printed; a value shorter than 12
    characters is not a credential and is skipped, as the secret guard skips it."""
    import base64
    import urllib.parse
    values = {}
    for name in names:
        value = os.environ.get(name) or ''
        if len(value) >= 12:
            values[name] = {value.encode(), urllib.parse.quote(value, safe='').encode(),
                            base64.b64encode(value.encode()).rstrip(b'=')}
    found, files = {}, 0
    for base in paths:
        base = Path(base)
        for path in ([base] if base.is_file() else sorted(p for p in base.rglob('*') if p.is_file())):
            files += 1
            blob = path.read_bytes()
            for name, forms in values.items():
                if any(form in blob for form in forms):
                    found.setdefault(name, []).append(str(path))
    return {'schema': 'cp18c-scan-v1', 'names_set': sorted(values), 'names_unset': sorted(set(names) - set(values)),
            'files_scanned': files, 'found': found, 'clean': not found}


# ------------------------------------------------------------------------------------------------ after the run
def bundle_plan(root: Path, runs: Path, now: datetime, mode: str = 'dry-run') -> dict:
    """Whether the day's bundle is built, and how its arrays are supplied. The bundle and its records are written once
    per as-of day (`cp18.records`), so a later run on the same as-of day (the runner's retry, the fallback after the
    runner) builds nothing. The arrays are the served artifact's A1 fit from this run's own attempt folder
    (`<runs>/A1-<day>-*/run-arrays.npz`, before the results step deletes them), else a refit."""
    from cp25 import guard
    from . import bundle
    from .records import BUNDLES_PATH
    from .store import STORE_PATH, LiveStore
    as_of = now.astimezone(guard.BERLIN).date()
    plan = {'now': now.isoformat(), 'mode': mode, 'as_of': str(as_of), 'build': False, 'how': 'none', 'arrays': '', 'served_day': '',
            'reason': ''}
    if (Path(root) / BUNDLES_PATH / str(as_of) / 'bundle.json').exists():
        return {**plan, 'already': True, 'reason': f'the bundle of as-of day {as_of} is already recorded (once per day)'}
    plan['already'] = False
    from contextlib import nullcontext
    from . import boundary
    store = LiveStore(Path(root) / STORE_PATH)
    try:
        with boundary.live(now=now) if mode == 'live' else nullcontext():
            store.verify()
            entries = bundle.issued_days(store)
            chosen = bundle.served_artifact(entries, bundle.freshness(store, now))
    except Exception as exc:  # noqa: BLE001 - no artifact to serve is a recorded reason, never an escaped run
        return {**plan, 'reason': f'no bundle: {type(exc).__name__}: {exc}'}
    plan.update(build=True, served_day=chosen['day'])
    if not chosen['artifact']['issued_policy']:
        return {**plan, 'how': 'none', 'reason': 'the served artifact is a failed issuance: no fitted arrays'}
    found = sorted(Path(runs).glob(f"A1-{chosen['day']}-*/run-arrays.npz"), key=lambda p: p.stat().st_mtime)
    if found:
        return {**plan, 'how': 'arrays', 'arrays': str(found[-1]), 'reason': "this run's own A1 fit of the served day"}
    return {**plan, 'how': 'refit', 'reason': "the served day's A1 fit is not in this run's results: a refit"}


def records(root: Path, *, remote: str = 'origin', branch: str = 'main', run_id: str = '') -> dict:
    """The day's data-only records committed and pushed through the claim, on the commit the run left synchronized.
    A lost race leaves the other host's records standing; a refusal (a guard, the network) publishes nothing."""
    from cp25 import guard
    from .claim import Claim, ClaimError
    paths = [p for p in DATA_PATHS if (Path(root) / p).exists()]
    record = {'schema': 'cp18c-records-v1', 'run_id': run_id, 'paths': paths, 'started_utc': utc_now()}
    try:
        result = Claim(root, remote=remote, branch=branch).publish(
            paths, f'live: the records of run {run_id} (outcomes, evaluation, bundle, publication)\n\n'
                   f'Delu-Daily-Run: {run_id}\n')
    except ClaimError as exc:
        return {**record, 'outcome': f'refused: {guard.redact(str(exc))}', 'exit': 2, 'finished_utc': utc_now()}
    if result.get('lost'):
        outcome = "main moved first: the other host's records stand; these were discarded"
    elif result.get('nothing'):
        outcome = 'nothing to commit'
    else:
        outcome = f"pushed {result['pushed']}"
    return {**record, 'result': result, 'outcome': outcome, 'exit': 0, 'finished_utc': utc_now()}


def after_run(clone: Path, project: Path, out: Path, *, now: datetime, run_id: str, remote: str, publish: str,
              mode: str, recorded: str | None = None) -> list[dict]:
    """Steps 2-4 of the sequence, as the workflow runs them. `publish` is `hub` or `local:<folder>`."""
    steps = []
    plan = bundle_plan(clone, out / 'runs', now, mode)
    dump(out / 'bundle-plan.json', plan)
    steps.append({'stage': 'bundle-plan', **{k: plan[k] for k in ('build', 'how', 'as_of', 'reason')}})
    if plan['build']:
        how = {'arrays': ['--arrays', plan['arrays']], 'refit': ['--refit'], 'none': []}[plan['how']]
        caches = ['--cache', str(out / 'payloads')] if (out / 'payloads').is_dir() else []
        bundled = stage('the dated bundle', [
            '-m', 'cp18.bundle', '--root', str(clone), '--project', str(project), '--mode', mode, '--now', plan['now'],
            '--out', str(out / 'bundle'), *how, *caches, *(['--recorded', recorded] if recorded else []),
            '--record', str(out / 'bundle-record.json'), '--commit-records', '--run-id', run_id], cwd=clone)
        steps.append(bundled)
        if bundled['exit'] == 0:
            script = clone / 'scripts/build_product_space.py'
            live_flags = []
            if publish == 'hub' or _supports(script, '--live'):
                live_flags += ['--live', str(clone / 'live')]
            if publish == 'hub' or (clone / SPACE_HISTORY).is_dir():
                live_flags += ['--history', str(clone / SPACE_HISTORY)]
            built = stage('the Space build', [str(script), '--bundle', str(out / 'bundle'), '--out', str(out / 'space'),
                                              *live_flags], cwd=clone)
            steps.append(built)
            if built['exit'] == 0:
                target = (['--hub', HUB] if publish == 'hub' else
                          ['--target', str(Path(publish.removeprefix('local:')).resolve())])
                steps.append(stage('the publication', [
                    '-m', 'cp18.publish', '--build', str(out / 'space'), '--bundle', str(out / 'bundle'), *target,
                    '--root', str(clone), '--live', str(clone / 'live'), '--record', str(out / 'publication.json')],
                    keep=('HF_TOKEN',) if publish == 'hub' else (), cwd=clone))
    value = records(clone, remote=remote, run_id=run_id)
    dump(out / 'records.json', value)
    steps.append({'stage': 'the records', 'exit': value['exit'], 'outcome': value['outcome']})
    return steps


def _supports(script: Path, flag: str) -> bool:
    try:
        return f"'{flag}'" in Path(script).read_text()
    except OSError:
        return False


def _prepare(args, issuing: bool) -> tuple[Path, Path, dict]:
    """The project named, the clone in place with its hooks, the ecCodes environment recorded; for an issuing run, the
    official binary checked in place (retrieved by hash only if absent) and the package zip rebuilt and compared."""
    clone, project = args.clone.resolve(), args.project.resolve()
    record = {'project': pin_project(project), 'clone': ensure_clone(clone, args.url),
              'wx_env': wx_environment(project / WX_PYTHON)}
    if issuing:
        from cp17 import runtime
        from . import host
        record['node'] = host.node(project, retrieve=not (project / runtime.NODE_RELATIVE).exists())
        record['package'] = host.package(clone, project)
    return clone, project, record


# ------------------------------------------------------------------------------------------------ commands
def results_folder(args, label: str) -> Path:
    """`--out` as given, or a new folder `<--out-root>/<label>` (refused if it exists: a run never writes over another)."""
    if args.out is not None:
        out = args.out.resolve()
        out.mkdir(parents=True, exist_ok=True)
        return out
    out = args.out_root.resolve() / label
    out.mkdir(parents=True, exist_ok=False)
    return out


def cmd_run(args) -> int:
    if args.mode == 'live':
        if args.now:
            raise SystemExit('a live run reads the real clock; --now is for dry runs only')
        inputs, publish = args.inputs or 'live-http', args.publish or 'hub'
    elif args.mode == 'dry-run':
        if not args.now or not args.inputs:
            raise SystemExit('a dry run needs --now and --inputs')
        inputs, publish = args.inputs, args.publish or 'none'
    else:
        raise SystemExit(f'--mode {args.mode!r}: expected dry-run or live')
    if publish != 'none' and publish != 'hub' and not publish.startswith('local:'):
        raise SystemExit(f'--publish {publish!r}: expected hub, local:<folder> or none')
    if args.mode == 'dry-run' and publish == 'hub':
        raise SystemExit('a dry run never publishes to the Hugging Face Space')
    label = args.run_label or f"local-{datetime.now(UTC):%Y%m%dT%H%M%SZ}"
    clone, project, record = _prepare(args, issuing=True)
    out = results_folder(args, label)
    dump(out / 'local-prepare.json', record)
    if not record['package']['equal_to_manifest']:
        print('local fallback: the rebuilt package zip is not the frozen one; nothing runs', file=sys.stderr)
        return 2
    run = stage('the pipeline', ['-m', 'cp18', 'run', '--root', str(clone), '--project', str(project), '--out', str(out),
                                 '--host', 'local', '--mode', args.mode, *(['--now', args.now] if args.now else []),
                                 '--inputs', inputs, '--remote', args.remote, '--run-label', label],
                keep=('ENTSOE_API_TOKEN',) if args.mode == 'live' else (), cwd=clone)
    steps, code = [run], run['exit']
    if publish != 'none' and code in (0, 1) and (out / 'run.json').exists():
        from . import plan
        value = json.loads((out / 'run.json').read_text())
        now = plan.parse_now(args.now) if args.now else datetime.now(UTC).replace(microsecond=0)
        recorded = inputs.partition(':')[2] if inputs.startswith('live:') else None
        steps += after_run(clone, project, out, now=now, run_id=value['run_id'], remote=args.remote, publish=publish,
                           mode=args.mode, recorded=recorded)
    dump(out / 'local-run.json', {'schema': 'cp18c-local-run-v1', 'mode': args.mode, 'publish': publish,
                                  'run_label': label, 'steps': steps, 'finished_utc': utc_now()})
    worst = max([s['exit'] for s in steps[1:] if s.get('exit')] or [0])
    return code if code else (2 if worst else 0)


def cmd_acquire(args) -> int:
    clone, project, record = _prepare(args, issuing=False)
    out = results_folder(args, f"acquire-{datetime.now(UTC):%Y%m%dT%H%M%SZ}")
    dump(out / 'local-prepare.json', record)
    done = stage('the acquisition', ['-m', 'cp18', 'acquire', '--root', str(clone), '--out', str(out / 'acquire.json'),
                                     '--remote', args.remote, '--window-hours', str(args.window_hours),
                                     *(['--day', args.day] if args.day else [])],
                 keep=('ENTSOE_API_TOKEN',), cwd=clone)
    return done['exit']


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog='python -m cp18.local')
    sub = ap.add_subparsers(dest='cmd', required=True)
    for name in ('run', 'acquire'):
        p = sub.add_parser(name)
        p.add_argument('--clone', required=True, type=Path)
        p.add_argument('--url', help='the product repository, to create the clone if it is absent')
        p.add_argument('--project', required=True, type=Path, help='the folder that holds .local/ (the pinned host '
                                                                   'files and the ecCodes environment)')
        where = p.add_mutually_exclusive_group(required=True)
        where.add_argument('--out', type=Path, help='the run\'s results folder')
        where.add_argument('--out-root', type=Path, help='a folder under which each run gets its own new results folder, '
                                                        'named by its label (the agents: never a stale file in a bundle)')
        p.add_argument('--remote', default='origin')
        if name == 'run':
            p.add_argument('--mode', default='dry-run', choices=('dry-run', 'live'))
            p.add_argument('--now', help='a dry run\'s simulated wall clock (never in a live run)')
            p.add_argument('--inputs', help='a dry run\'s inputs; a live run takes live-http')
            p.add_argument('--publish', help='hub (a live run\'s default), local:<folder>, or none (a dry run\'s default)')
            p.add_argument('--run-label')
        else:
            p.add_argument('--day')
            p.add_argument('--window-hours', type=float, default=3.0)
    b = sub.add_parser('bundle-plan')
    b.add_argument('--mode', default='dry-run', choices=('dry-run', 'live'))
    b.add_argument('--root', default=Path('.'), type=Path)
    b.add_argument('--runs', required=True, type=Path)
    b.add_argument('--now', help='the wall clock (default: now)')
    r = sub.add_parser('records')
    r.add_argument('--root', default=Path('.'), type=Path)
    r.add_argument('--remote', default='origin')
    r.add_argument('--branch', default='main')
    r.add_argument('--run-id', required=True)
    r.add_argument('--out', required=True, type=Path)
    w = sub.add_parser('wx-env')
    w.add_argument('--python', required=True, type=Path)
    w.add_argument('--freeze', type=Path)
    w.add_argument('--out', required=True, type=Path)
    s_ = sub.add_parser('scan')
    s_.add_argument('--name', action='append', required=True, help='a credential variable to scan for, by value')
    s_.add_argument('--out', required=True, type=Path)
    s_.add_argument('paths', nargs='+', type=Path)
    d = sub.add_parser('decode-check')
    d.add_argument('--run', required=True, help='the GFS run day (00 UTC); it serves the next delivery day')
    d.add_argument('--python', required=True, type=Path)
    d.add_argument('--archived', required=True, type=Path, help='a folder holding runs/<run>.{json,npz}')
    d.add_argument('--work', required=True, type=Path)
    d.add_argument('--out', required=True, type=Path)
    return ap


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    if args.cmd == 'run':
        return cmd_run(args)
    if args.cmd == 'acquire':
        return cmd_acquire(args)
    if args.cmd == 'bundle-plan':
        from . import plan
        now = plan.parse_now(args.now) if args.now else datetime.now(UTC).replace(microsecond=0)
        for k, v in bundle_plan(args.root.resolve(), args.runs, now, args.mode).items():
            print(f"{k}={str(v).lower() if isinstance(v, bool) else str(v).replace(chr(10), ' ')}")
        return 0
    if args.cmd == 'records':
        value = records(args.root.resolve(), remote=args.remote, branch=args.branch, run_id=args.run_id)
        dump(args.out, value)
        print('records |', value['outcome'], flush=True)
        return value['exit']
    if args.cmd == 'scan':
        value = scan(args.name, args.paths)
        dump(args.out, value)
        print('scan |', value['files_scanned'], 'files |', 'set:', ', '.join(value['names_set']) or 'none', '|',
              'clean' if value['clean'] else 'FOUND: ' + ', '.join(f'{k} in {len(v)} file(s)' for k, v in value['found'].items()),
              flush=True)
        return 0 if value['clean'] else 1
    if args.cmd == 'wx-env':
        value = wx_environment(args.python, args.freeze)
        dump(args.out, value)
        ecc = value.get('eccodes') or {}
        print('ecCodes environment |', value.get('python'), '| eccodes', ecc.get('module_version'), 'api',
              ecc.get('api_version'), '|', 'pinned versions installed' if not value['problems'] else
              '; '.join(value['problems']), flush=True)
        return 0 if not value['problems'] else 1
    value = decode_check(date.fromisoformat(args.run), args.python, args.archived, args.work)
    dump(args.out, value)
    print('decode check', value['run'], '|', value['outcome'], flush=True)
    return 0 if value.get('equal') else 1


if __name__ == '__main__':
    sys.exit(main())
