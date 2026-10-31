"""The frozen daily chain under one Node.js binary on one machine: a GitHub-hosted runner, or the local control.

Nothing frozen changes. Every fit is `cp17.daily.Host.run`, the chain's own attempt runner, with the binary passed
in; every shadow day is `cp17.daily.issue_day`; the shadow states are `cp17.shadow.ShadowStates`, in CPython. The
inputs are the reference runs' own: the package zip, the payloads (checked by their identity), the shadow states
and the published prices their release reads (`cp17cprobe.inputs`). The comparison with the frozen host is
`cp17cprobe.compare`, offline.

- **verify:** every shipped input against its recorded hash.
- **node:** retrieve the official Node.js tarball for this platform, verified against the release folder's
  `SHASUMS256.txt`, itself pinned by its SHA-256; record the URL, bytes and hashes.
- **host:** the machine and the runtime: image label and version, CPU, cores, memory, the binary, Pyodide's files.
- **parity:** one fresh issuing-host process per stem, in order, one at a time (as `reports/cp17b/host-parity.json`
  was made), each judged by `cp17.issue.attempt` on the success definition's criteria 1, 2, 3 and 5.
- **shadow:** consecutive delivery days from a given state: release D-2, check the buffers against the ones the
  shipped payloads carry, issue the day (A1 and the three shadows), register the shadows. Two days run at once
  (as `reports/cp17b/shadow-run.json` was made); a day needs only D-2's registrations.

    OPENBLAS_NUM_THREADS=1 ... PYTHONPATH=src python -m cp17cprobe.run parity --node <bin> --pyodide <dir> \\
        --zip <src.zip> --payloads <dir> --stems A1-2025-10-26,... --out <dir>
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tarfile
import time
import urllib.request

import numpy as np

BLAS = ('OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS')
NO_IDENTITY = {'policy_identity_sha256': None, 'package_sha256': None}
NODE_VERSION = 'v26.7.0'
NODE_DIST = f'https://nodejs.org/dist/{NODE_VERSION}/'
#: `SHASUMS256.txt` of the release folder, as retrieved for this probe on 2026-10-30 (2,943 bytes).
SHASUMS_SHA256 = '4533f0a43b9ba7f78a48230a0511b9dd5c931f20c3b3cac281ff9b7a2080fb2e'
TARBALLS = {('Darwin', 'arm64'): f'node-{NODE_VERSION}-darwin-arm64.tar.gz',
            ('Linux', 'x86_64'): f'node-{NODE_VERSION}-linux-x64.tar.gz'}
#: The runner image's own variables, read by name (none is a credential).
RUNNER_ENV = ('ImageOS', 'ImageVersion', 'RUNNER_OS', 'RUNNER_ARCH', 'RUNNER_NAME', 'RUNNER_ENVIRONMENT',
              'GITHUB_RUN_ID', 'GITHUB_RUN_ATTEMPT', 'GITHUB_JOB', 'GITHUB_SHA', 'GITHUB_REPOSITORY')


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def buffer_sha256(buffer: list[dict]) -> str:
    """`cp17.tools.states.buffer_sha256`, restated: the tools are not shipped to a runner."""
    return hashlib.sha256(json.dumps(buffer, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def dump(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=1, sort_keys=True, allow_nan=False, default=str) + '\n')


def check_blas() -> None:
    if any(os.environ.get(n) != '1' for n in BLAS):
        raise SystemExit('export the five *_NUM_THREADS=1 before NumPy loads, as CP-17b ran')


# --- node ---------------------------------------------------------------------------------------------------------

def _context():
    """TLS against certifi's roots where the environment has it (a runner's Python may ship without a CA store)."""
    import ssl
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


def _get(url: str, target: Path) -> dict:
    started = time.perf_counter()
    with urllib.request.urlopen(url, timeout=300, context=_context()) as response, open(target, 'wb') as handle:
        while block := response.read(1 << 20):
            handle.write(block)
    return {'url': url, 'bytes': target.stat().st_size, 'sha256': sha256_file(target),
            'seconds': time.perf_counter() - started}


def fetch_node(dest: Path) -> dict:
    """The official tarball for this platform, verified against `SHASUMS256.txt` (pinned), extracted to `dest`."""
    name = TARBALLS[(platform.system(), platform.machine())]
    dest.mkdir(parents=True, exist_ok=True)
    sums = _get(NODE_DIST + 'SHASUMS256.txt', dest / 'SHASUMS256.txt')
    if sums['sha256'] != SHASUMS_SHA256:
        raise SystemExit(f"SHASUMS256.txt is not the pinned one: {sums['sha256']}")
    listed = {line.split()[1]: line.split()[0] for line in (dest / 'SHASUMS256.txt').read_text().splitlines() if line.strip()}
    tarball = _get(NODE_DIST + name, dest / name)
    if tarball['sha256'] != listed.get(name):
        raise SystemExit(f'{name} does not match SHASUMS256.txt')
    with tarfile.open(dest / name) as archive:
        archive.extractall(dest, filter='data')
    binary = dest / name.removesuffix('.tar.gz') / 'bin' / 'node'
    return {'version': NODE_VERSION, 'shasums': sums, 'tarball': {**tarball, 'listed_sha256': listed[name]},
            'binary': {'path': str(binary), 'sha256': sha256_file(binary), 'bytes': binary.stat().st_size}}


def verify_inputs(tree: Path, files: dict) -> dict:
    """Every shipped input against the hash `cp17cprobe.inputs` recorded when it was taken from the reference."""
    problems = [rel for rel, expected in sorted(files.items())
                if not (tree / rel).is_file() or sha256_file(tree / rel) != expected]
    return {'files': len(files), 'problems': problems, 'verified': not problems}


# --- host ---------------------------------------------------------------------------------------------------------

def _run(*command: str) -> str:
    try:
        return subprocess.run(command, capture_output=True, text=True, timeout=60).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ''


def machine() -> dict:
    """The machine as a runner reports it: CPU model, logical cores, memory, OS."""
    out = {'platform': platform.platform(), 'system': platform.system(), 'machine': platform.machine(),
           'release': platform.release(), 'cpu_count': os.cpu_count(),
           'runner_env': {n: os.environ.get(n) for n in RUNNER_ENV}}
    if hasattr(os, 'sched_getaffinity'):
        out['cpu_affinity'] = len(os.sched_getaffinity(0))
    if platform.system() == 'Darwin':
        out.update(cpu_model=_run('sysctl', '-n', 'machdep.cpu.brand_string'),
                   memory_bytes=int(_run('sysctl', '-n', 'hw.memsize') or 0),
                   physical_cores=int(_run('sysctl', '-n', 'hw.physicalcpu') or 0),
                   os_version=_run('sw_vers', '-productVersion'), os_build=_run('sw_vers', '-buildVersion'),
                   hardware_model=_run('sysctl', '-n', 'hw.model'))
    else:
        info = Path('/proc/cpuinfo').read_text() if Path('/proc/cpuinfo').exists() else ''
        models = sorted({line.split(':', 1)[1].strip() for line in info.splitlines() if line.startswith('model name')})
        mem = next((line for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemTotal')), '')
        release = Path('/etc/os-release').read_text() if Path('/etc/os-release').exists() else ''
        out.update(cpu_model='; '.join(models), memory_bytes=int(mem.split()[1]) * 1024 if mem else 0,
                   os_version=next((line.split('=', 1)[1].strip('"') for line in release.splitlines()
                                    if line.startswith('PRETTY_NAME=')), ''))
    return out


def host_identity(node: Path, pyodide: Path, expected_pyodide: dict | None, label: str | None) -> dict:
    versions = json.loads(_run(str(node), '-p', 'JSON.stringify(process.versions)') or '{}')
    files = {p.name: sha256_file(p) for p in sorted(Path(pyodide).iterdir()) if p.is_file()}
    out = {'label': label, 'machine': machine(), 'python': sys.version.split()[0], 'numpy': np.__version__,
           'node': {'path': str(node), 'sha256': sha256_file(node), 'version': versions.get('node'),
                    'v8': versions.get('v8'), 'uv': versions.get('uv')},
           'pyodide': {'files_sha256': files, 'files': len(files)}}
    if expected_pyodide is not None:
        out['pyodide']['equal_to_freeze_manifest'] = files == expected_pyodide
    return out


# --- parity -------------------------------------------------------------------------------------------------------

def _record(run: dict, out: Path) -> dict:
    """What a host record shows, with the record file's own path and hash so that a comparison can verify it."""
    folder = Path(run['folder']) if run.get('folder') else None
    path = folder / 'run.json' if folder else None
    arrays = folder / 'run-arrays.npz' if folder else None
    python = run.get('python') or {}
    result = python.get('result') or {}
    return {'record': str(path.relative_to(out)) if path and path.exists() else None,
            'record_sha256': sha256_file(path) if path and path.exists() else None,
            'arrays_sha256': sha256_file(arrays) if arrays and arrays.exists() else None,
            'errors': run.get('errors'), 'returncode': run.get('returncode'),
            'quantiles_sha256': result.get('quantiles_sha256'), 'result_payload_sha256': result.get('payload_sha256'),
            'total_ms': run.get('total_ms'), 'process_rss_peak_bytes': run.get('process_rss_peak_bytes'),
            'wasm_memory_bytes': run.get('wasm_memory_bytes'), 'driver_wall_ms': run.get('driver_wall_ms'),
            'node': run.get('node'), 'node_execpath': run.get('node_execpath'), 'pyodide': run.get('pyodide_version'),
            'blocked_non_local_requests': run.get('blocked_non_local_requests'),
            'executed_files': python.get('source_sha256'), 'stubs': python.get('stubs'),
            'runtime_python': python.get('python'), 'runtime_numpy': python.get('numpy')}


def attempt(host, folder: Path, stem: str, out: Path) -> dict:
    """One stem's fit in a fresh issuing-host process, judged by `cp17.issue.attempt` as the chain judges it."""
    from cp17 import daily, dailyfit, issue
    policy, day = stem.split('-', 1)[0], date.fromisoformat(stem.split('-', 1)[1])
    npz, meta_path = daily._paths(folder, policy, day)
    arrays, meta = dailyfit.load(npz, meta_path)
    runs = {}

    def fit():
        run = runs['run'] = host.run(folder, stem, out / 'runs')
        if run.get('errors'):
            raise dailyfit.FitFailure(run['errors'][0][-2000:])
        return run['python']['result']

    def measured():
        run = runs['run']
        return {'wall_seconds': run['total_ms'] / 1000, 'peak_bytes': int(run['process_rss_peak_bytes'])}
    started = time.perf_counter()
    item = issue.attempt(policy, fit, day, arrays, meta, resources=measured)
    item.pop('result', None)
    item.pop('traceback', None)
    return {'stem': stem, 'policy': policy, 'day': str(day), 'payload_sha256': meta.get('payload_sha256'),
            'attempt': item, 'seconds': time.perf_counter() - started, **_record(runs.get('run', {}), out)}


def parity(args) -> int:
    from cp17 import daily
    check_blas()
    host = daily.Host(args.pyodide, args.zip, node=str(args.node))
    record = {'schema': 'cp17c-probe-parity-v1', 'node': str(args.node), 'zip_sha256': sha256_file(args.zip),
              'payloads': str(args.payloads), 'stems': [], 'concurrency': 1}
    for stem in args.stems.split(','):
        row = attempt(host, args.payloads, stem, args.out)
        record['stems'].append(row)
        dump(args.out / 'parity.json', record)
        print(stem, row['attempt']['status'], f"{(row['total_ms'] or 0) / 1000:.1f}s",
              f"{(row['process_rss_peak_bytes'] or 0) >> 20} MiB", (row['quantiles_sha256'] or '')[:16], flush=True)
    return 0 if all(r['attempt']['status'] == 'ok' for r in record['stems']) else 1


# --- shadow -------------------------------------------------------------------------------------------------------

def truth_from(path: Path):
    """The published prices the release reads, as `cp17.tools.exercise.truth` serves them: NaN unless every key
    is present."""
    import pandas as pd
    value = json.loads(Path(path).read_text())
    index = pd.DatetimeIndex(pd.to_datetime(value['keys'], utc=True)).as_unit('ns')
    y = np.asarray(value['y'], float)

    def at(keys):
        pos = index.get_indexer(pd.DatetimeIndex(pd.to_datetime(keys, utc=True)).as_unit('ns'))
        if (pos < 0).any():
            return np.full(len(keys), np.nan)
        return np.asarray(y[pos], float)
    return at


def shipped_buffers(payloads: Path, day: date) -> dict:
    """The buffer each shipped payload of the day carries (the one its builder read from the reference's states)."""
    from cp17.shadow import POLICIES
    out = {}
    for p in POLICIES:
        meta = json.loads((payloads / f'{p}-{day}.json').read_text())
        out[p] = buffer_sha256(meta['b0_buffer' if p == 'B0' else 'h_buffer'])
    return out


def _day_summary(item: dict, out: Path) -> dict:
    from cp17 import issue
    art = item['artifact']
    return {'path': [(a['policy'], a['status']) for a in item['decision']['attempts']],
            'causes': {a['policy']: a.get('cause') for a in item['decision']['attempts'] if a['status'] != 'ok'},
            'resources': {a['policy']: a.get('resources') for a in item['decision']['attempts']},
            'display': item['decision']['display'], 'issued_policy': art['issued_policy'],
            'issued_quantiles_sha256': art.get('quantiles_sha256'), 'artifact': art,
            'artifact_verifies': issue.verify_artifact(art) == [], 'shadows': item['shadows'],
            'process_seconds': item['process_seconds'], 'total_seconds': item['process_seconds_sum'],
            'driver_seconds': item['total_seconds'], 'within_ceiling': item['process_seconds_sum'] <= issue.WALL_LIMIT_SECONDS,
            'blocked_non_local_requests': item['blocked_non_local_requests'],
            'runs': {p: _record(r, out) for p, r in item['runs'].items()}}


def _result(item: dict, policy: str):
    run = item['runs'].get(policy)
    if run is None or run.get('errors'):
        return None
    return run['python']['result']


def shadow(args) -> int:
    from cp17 import daily
    from cp17.shadow import POLICIES, ShadowStates
    check_blas()
    started = time.perf_counter()
    state = ShadowStates.from_dict(json.loads(args.state.read_text()))
    at = truth_from(args.truth)
    host = daily.Host(args.pyodide, args.zip, node=str(args.node))
    first, last = date.fromisoformat(args.first), date.fromisoformat(args.last)
    days = [first + timedelta(days=i) for i in range((last - first).days + 1)]
    groups = [days[i:i + args.concurrency] for i in range(0, len(days), args.concurrency)]
    record = {'schema': 'cp17c-probe-shadow-v1', 'node': str(args.node), 'zip_sha256': sha256_file(args.zip),
              'first': str(first), 'last': str(last), 'concurrency': args.concurrency,
              'start': {'state_file_sha256': sha256_file(args.state), 'digest': state.digest()},
              'truth_sha256': sha256_file(args.truth), 'days': {}}
    for group in groups:
        prepared = {}
        for day in group:
            state.release(day, at)
            mine = {p: buffer_sha256(state.buffer(p)) for p in POLICIES}
            theirs = shipped_buffers(args.payloads, day)
            prepared[day] = {'buffers_sha256': mine, 'payload_buffers_sha256': theirs,
                             'buffers_equal_payload': {p: mine[p] == theirs[p] for p in POLICIES}}
        with ThreadPoolExecutor(max_workers=len(group)) as pool:
            done = dict(zip(group, pool.map(lambda d: daily.issue_day(
                d, True, args.payloads, host=host, out=args.out / 'runs', identity=NO_IDENTITY), group)))
        for day in group:
            item = done[day]
            prepared[day]['intake'] = {p: state.issue(p, day, _result(item, p)) for p in POLICIES}
            prepared[day].update(_day_summary(item, args.out))
            record['days'][str(day)] = prepared[day]
            d = prepared[day]
            print(day, d['path'], d['intake'], d['buffers_equal_payload'], f"{d['total_seconds']:.0f}s", flush=True)
        dump(args.out / 'shadow.json', record)
    final = state.to_dict()
    dump(args.out / 'final-state.json', final)
    record['final'] = {'digest': state.digest(), 'state_file_sha256': sha256_file(args.out / 'final-state.json')}
    record['seconds'] = time.perf_counter() - started
    dump(args.out / 'shadow.json', record)
    ok = all(d['artifact_verifies'] and d['issued_policy'] for d in record['days'].values())
    return 0 if ok else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    v = sub.add_parser('verify')
    v.add_argument('--tree', default='.', type=Path)
    v.add_argument('--files', required=True, type=Path, help='a JSON {path: sha256}')
    v.add_argument('--out', required=True, type=Path)
    n = sub.add_parser('node')
    n.add_argument('--dest', required=True, type=Path)
    n.add_argument('--out', required=True, type=Path)
    h = sub.add_parser('host')
    h.add_argument('--node', required=True, type=Path)
    h.add_argument('--pyodide', required=True, type=Path)
    h.add_argument('--expected-pyodide', type=Path, help='a JSON {file: sha256} (the freeze manifest\'s 23)')
    h.add_argument('--label', help="the runner's image label, as the workflow names it")
    h.add_argument('--out', required=True, type=Path)
    for name in ('parity', 'shadow'):
        s = sub.add_parser(name)
        s.add_argument('--node', required=True, type=Path)
        s.add_argument('--pyodide', required=True, type=Path)
        s.add_argument('--zip', required=True, type=Path)
        s.add_argument('--payloads', required=True, type=Path)
        s.add_argument('--out', required=True, type=Path)
        if name == 'parity':
            s.add_argument('--stems', required=True)
        else:
            s.add_argument('--first', required=True)
            s.add_argument('--last', required=True)
            s.add_argument('--state', required=True, type=Path)
            s.add_argument('--truth', required=True, type=Path)
            s.add_argument('--concurrency', type=int, default=2)
    args = ap.parse_args(argv)
    if args.cmd == 'verify':
        value = verify_inputs(args.tree, json.loads(args.files.read_text()))
        dump(args.out, value)
        print(value['files'], 'inputs', 'verified' if value['verified'] else f"NOT verified: {value['problems'][:5]}")
        return 0 if value['verified'] else 1
    if args.cmd == 'node':
        value = fetch_node(args.dest)
        dump(args.out, value)
        print(value['binary']['path'], value['binary']['sha256'])
        return 0
    if args.cmd == 'host':
        expected = json.loads(args.expected_pyodide.read_text()) if args.expected_pyodide else None
        value = host_identity(args.node, args.pyodide, expected, args.label)
        dump(args.out, value)
        print(json.dumps({k: value['machine'].get(k) for k in ('cpu_model', 'cpu_count', 'memory_bytes', 'os_version')}),
              value['node']['sha256'], value['pyodide'].get('equal_to_freeze_manifest'))
        return 0 if value['pyodide'].get('equal_to_freeze_manifest', True) else 1
    # Absolute, because the host's fresh run folders are (`tempfile.mkdtemp`) and the records cite them relative to it.
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=True)
    return parity(args) if args.cmd == 'parity' else shadow(args)


if __name__ == '__main__':
    raise SystemExit(main())
