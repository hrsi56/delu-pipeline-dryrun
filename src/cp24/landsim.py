"""Simulated LAND: does CI stay green on `main` after the Owner squashes the candidate onto it?

The Owner lands by hand with `git merge --squash gauntlet/cp-24` onto `main`. The candidate adds files only under
CP-24's write paths. So when `main` touches none of those paths, the squash tree is `main`'s tree plus the
candidate's CP-24 paths, and the module checks both conditions first. It reads the project repository only
(`git archive`, `git diff`, `git ls-tree`), and builds the squash tree in a fresh scratch repository with one
commit, no history and no tags, as CI's shallow checkout sees it. There it runs the steps of
`.github/workflows/tests.yml`: the browser payload, the full suite, the CQR fixture, `verify_release.py`, the WASM
equivalence gate and the publication guard.

`--later-edits` appends a line to living files that authorized work on `main` changes after the LAND: the closure
records `progress.md` and `docs/track-b/cp-0-defects.md`, `AGENTS.md`, `capstone_v21.md`, `README.md`,
`docs/index.html`, `scripts/mlflow_export.py` and CP-23's test-only lock. It then commits them as a second scratch
commit and runs the same steps.

Run under the monitor:

    python scripts/cp24_ddnn2.py monitor --name landsim-<label> --workers 1 --log <log> -- \
        <python> -m cp24.landsim --main <rev> --candidate <rev> --out <dir outside the checkout> [--later-edits]

It writes `<out>/land-simulation.json` and one log per step. Nothing is written in the project repository.
"""
from __future__ import annotations

import argparse
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import time

ROOT = Path(__file__).resolve().parents[2]
CP24_PATHS = ('src/cp24/', 'tests/cp24/', 'scripts/cp24_', 'reports/ddnn2/', 'docs/track-b/evidence/cp-24/',
              'docs/track-b/research-content/cp24-claims.md')
BASE = '522d7ea722b5d215d827d7aed9c56b7a9dec066e'
LIVING = ('progress.md', 'docs/track-b/cp-0-defects.md', 'AGENTS.md', 'capstone_v21.md', 'README.md', 'docs/index.html',
          'scripts/mlflow_export.py', 'tests/cp23/torch-reference/uv.lock')
SCRATCH_GIT = ['git', '-c', 'user.name=land-sim', '-c', 'user.email=land-sim@localhost', '-c', 'commit.gpgsign=false',
               '-c', 'core.hooksPath=/dev/null']
CHECKPOINT_VARIABLES = ('CP16_LEDGER', 'CP20_LEDGER', 'CP21_LEDGER', 'CP22_LEDGER', 'CP23_LEDGER', 'CP24_LEDGER',
                        'CP24_JOB_INDEX', 'CP24_JOB_NAME', 'CP24_WORKERS', 'CP21_PROJECT_ROOT', 'CP22_PROJECT_ROOT',
                        'CP23_PROJECT_ROOT', 'CP24_PROJECT_ROOT')


def _git(*args: str, cwd: Path = ROOT) -> str:
    return subprocess.check_output(['git', *args], cwd=cwd, text=True)


def _names(diff: str) -> list[tuple[str, str]]:
    return [tuple(line.split('\t', 1)) for line in diff.splitlines() if line.strip()]


def preconditions(main: str, candidate: str) -> dict:
    """The candidate only adds files under CP-24's paths, and `main` touches none of them since the base."""
    added = _names(_git('diff', '--name-status', '--no-renames', BASE, candidate))
    moved = _names(_git('diff', '--name-status', '--no-renames', BASE, main))
    return {'candidate_changes': len(added),
            'candidate_only_adds_under_cp24_paths': all(s == 'A' and n.startswith(CP24_PATHS) for s, n in added),
            'main_changes_since_base': len(moved),
            'main_touches_no_cp24_path': not any(n.startswith(CP24_PATHS) for _, n in moved),
            'base_is_an_ancestor_of_main': subprocess.run(['git', 'merge-base', '--is-ancestor', BASE, main],
                                                          cwd=ROOT).returncode == 0}


def _extract(rev: str, paths: list[str], into: Path) -> None:
    data = subprocess.run(['git', 'archive', '--format=tar', rev, *paths], cwd=ROOT, capture_output=True, check=True).stdout
    with tarfile.open(fileobj=io.BytesIO(data)) as tar:
        tar.extractall(into, filter='tar')


def materialise(main: str, candidate: str, tree: Path) -> dict:
    if tree.exists():
        shutil.rmtree(tree)
    tree.mkdir(parents=True)
    _extract(main, [], tree)
    ours = sorted(n for n in _git('ls-tree', '-r', '--name-only', candidate).split('\n') if n.startswith(CP24_PATHS))
    _extract(candidate, ours, tree)
    subprocess.run([*SCRATCH_GIT, 'init', '-q'], cwd=tree, check=True)
    subprocess.run([*SCRATCH_GIT, 'add', '-A'], cwd=tree, check=True)
    subprocess.run([*SCRATCH_GIT, 'commit', '-q', '-m', f'simulated squash of {candidate} onto {main}'], cwd=tree, check=True)
    return {'scratch_tree': _git('rev-parse', 'HEAD^{tree}', cwd=tree).strip(), 'cp24_files': len(ours)}


def later_edits(tree: Path) -> list[str]:
    for name in LIVING:
        with (tree / name).open('ab') as handle:
            handle.write(b'\n# simulated later authorized change after the CP-24 LAND\n' if name.endswith(('.py', '.lock'))
                         else b'\n<!-- simulated later authorized change after the CP-24 LAND -->\n')
    subprocess.run([*SCRATCH_GIT, 'commit', '-q', '-am', 'simulated later authorized edits'], cwd=tree, check=True)
    return list(LIVING)


def run_steps(tree: Path, out: Path, python: str) -> list[dict]:
    env = {k: v for k, v in os.environ.items() if k not in CHECKPOINT_VARIABLES}
    env.update(PYTHONPATH=str(tree / 'src'), MLFLOW_DISABLE_TELEMETRY='true', DO_NOT_TRACK='1', MLFLOW_DISABLE_AGENT_HINT='1',
               PYTHONDONTWRITEBYTECODE='1')
    steps = [('payload', [python, 'scripts/build_wasm_payload.py']),
             ('suite', [python, '-m', 'pytest', '-q', '-p', 'no:cacheprovider', '-rfEs']),
             ('cqr', [python, '-m', 'pytest', '-q', '-p', 'no:cacheprovider', 'tests/test_10_cqr_order_statistic.py']),
             ('verify-release', [python, 'scripts/verify_release.py']),
             ('wasm', [python, '-m', 'pytest', '-q', '-p', 'no:cacheprovider', 'tests/test_22_wasm_equivalence.py']),
             ('publication-guard', ['python3', 'scripts/publication_guard.py', 'tree'])]
    results = []
    for name, command in steps:
        began = time.time()
        run = subprocess.run(command, cwd=tree, env=env, capture_output=True, text=True)
        (out / f'{name}.log').write_text(run.stdout + run.stderr)
        lines = [line for line in (run.stdout + run.stderr).splitlines() if line.strip()]
        failed = [line for line in lines if line.startswith(('FAILED ', 'ERROR '))]
        results.append({'step': name, 'exit_code': run.returncode, 'seconds': round(time.time() - began, 1),
                        'summary': lines[-1] if lines else '', 'failed': failed,
                        'tracked_changes_after': len(_git('status', '--porcelain=v1', '--untracked-files=no', cwd=tree).splitlines())})
    return results


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--main', required=True)
    ap.add_argument('--candidate', required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--later-edits', action='store_true')
    ap.add_argument('--python', default=sys.executable)
    args = ap.parse_args(argv)
    out = args.out.resolve()
    if out == ROOT or ROOT in out.parents:
        raise SystemExit('--out must lie outside this checkout')
    main_sha = _git('rev-parse', '--verify', f'{args.main}^{{commit}}').strip()
    cand_sha = _git('rev-parse', '--verify', f'{args.candidate}^{{commit}}').strip()
    out.mkdir(parents=True, exist_ok=True)
    pre = preconditions(main_sha, cand_sha)
    record = {'schema': 'cp24-land-simulation-v1', 'main': main_sha, 'candidate': cand_sha, 'preconditions': pre,
              'later_edits': None}
    if not all(v for k, v in pre.items() if isinstance(v, bool)):
        record['result'] = 'not simulated: the squash tree is not main plus the candidate\'s CP-24 paths'
        (out / 'land-simulation.json').write_text(json.dumps(record, indent=1) + '\n')
        print(json.dumps(record))
        return 2
    tree = out / 'tree'
    record.update(materialise(main_sha, cand_sha, tree))
    if args.later_edits:
        record['later_edits'] = later_edits(tree)
    record['steps'] = run_steps(tree, out, args.python)
    record['green'] = all(s['exit_code'] == 0 for s in record['steps'])
    record['written_utc'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    (out / 'land-simulation.json').write_text(json.dumps(record, indent=1) + '\n')
    print(json.dumps({k: v for k, v in record.items() if k != 'steps'}), flush=True)
    for s in record['steps']:
        print(json.dumps(s)[:600], flush=True)
    return 0 if record['green'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
