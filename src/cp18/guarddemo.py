"""The guards on the pipeline's own commits and pushes, shown with fake credentials (r19 §24.13; AGENTS.md's conditions).

The pipeline commits and pushes only in a checkout whose `core.hooksPath` is its `.githooks` (`cp18.claim`), so every
commit runs the secret guard and the leak scan, and every push runs them again with the publication guard. This shows
it, on the runner or locally, in a throwaway repository whose hooks are the checkout's own:

1. a staged file holding a fake credential's value: the commit is refused, naming the variable, never the value;
2. a commit message holding it: refused;
3. a commit holding it, made where the value was not set, then pushed where it is: the push is refused;
4. a staged environment dump (the shape that leaked the DagsHub token): refused by the leak scan, by its shape;
5. a data-only commit and its push: both pass.

**The fake values** are derived from the run's identity (`fake_value`), so they never need to be printed, stored or
uploaded: anyone holding the run id can recompute them and scan the run's public log and artifacts for them by value
(`scan`). They are random-looking 40-character hex strings, the shape of the token that leaked on 2026-09-24. No real
credential is read: every child process gets an environment built from scratch with `SECRET_GUARD_ENV_ONLY=1`.

    PYTHONPATH=src python -m cp18.guarddemo --checkout . --work <dir> --run-id <id> --out <file>
    PYTHONPATH=src python -m cp18.guarddemo scan --run-id <id> <files or folders>...
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

NAMES = ('ENTSOE_API_TOKEN', 'HF_TOKEN', 'DAGSHUB_USER_TOKEN')
SALT = 'cp18a-fake-credential'


def fake_value(name: str, run_id: str) -> str:
    """A fake credential for `name` in run `run_id`: never a real value, recomputable, never printed."""
    return hashlib.sha256(f'{SALT}:{name}:{run_id}'.encode()).hexdigest()[:40]


def fake_environment(run_id: str) -> dict[str, str]:
    return {name: fake_value(name, run_id) for name in NAMES}


def _env(home: Path, extra: dict | None = None) -> dict:
    """An environment built from scratch: no inherited variable can carry a real credential into the guards."""
    return {'PATH': os.environ.get('PATH', '/usr/bin:/bin'), 'HOME': str(home), 'SECRET_GUARD_ENV_ONLY': '1',
            'GIT_CONFIG_NOSYSTEM': '1', 'GIT_PAGER': 'cat', 'GIT_EDITOR': 'true', 'GIT_TERMINAL_PROMPT': '0',
            'GIT_AUTHOR_NAME': 'delu-daily', 'GIT_AUTHOR_EMAIL': 'delu-daily@users.noreply.github.com',
            'GIT_COMMITTER_NAME': 'delu-daily', 'GIT_COMMITTER_EMAIL': 'delu-daily@users.noreply.github.com',
            **(extra or {})}


def _git(repo: Path, env: dict, *args: str, stdin: str | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(['git', '-C', str(repo), *args], capture_output=True, text=True, env=env, input=stdin)


def _case(name: str, done: subprocess.CompletedProcess, expect_blocked: bool, values: list[str]) -> dict:
    text = done.stdout + done.stderr
    blocked = done.returncode != 0
    return {'case': name, 'exit': done.returncode, 'blocked': blocked, 'as_expected': blocked == expect_blocked,
            'value_in_output': any(v in text for v in values),
            'output': [line for line in text.splitlines() if 'BLOCKED' in line or 'leak' in line.lower()][:6]}


def demo(checkout: Path, work: Path, run_id: str) -> dict:
    checkout, work = Path(checkout).resolve(), Path(work).resolve()
    if work.exists():
        shutil.rmtree(work)
    (work / 'home').mkdir(parents=True)
    fakes = fake_environment(run_id)
    values = list(fakes.values())
    holding, clean = _env(work / 'home', fakes), _env(work / 'home')
    repo, bare = work / 'repo', work / 'origin.git'
    subprocess.run(['git', 'init', '--quiet', '--bare', '-b', 'main', str(bare)], check=True, env=clean)
    subprocess.run(['git', 'init', '--quiet', '-b', 'main', str(repo)], check=True, env=clean)
    _git(repo, clean, 'config', 'core.hooksPath', str(checkout / '.githooks'))
    _git(repo, clean, 'remote', 'add', 'origin', str(bare))
    # The publication guard checks every commit bound for main: a final page and build record, as on the product.
    (repo / 'docs').mkdir()
    (repo / 'docs/index.html').write_text('<!doctype html><title>guard demonstration</title>\n')
    (repo / 'reports/cp3').mkdir(parents=True)
    (repo / 'reports/cp3/pages_build.json').write_text('{"final": true}\n')
    _git(repo, clean, 'add', '-A')
    cases = [_case('a clean commit, with fake credentials set',
                   _git(repo, holding, 'commit', '--quiet', '-m', 'a clean first commit'), False, values)]
    template = repo

    def fresh(name: str) -> Path:
        """Each case starts from the first commit, in its own clone with the same hooks and remote, nothing staged."""
        path = work / name
        subprocess.run(['git', 'clone', '--quiet', str(template), str(path)], check=True, env=clean)
        _git(path, clean, 'config', 'core.hooksPath', str(checkout / '.githooks'))
        _git(path, clean, 'remote', 'set-url', 'origin', str(bare))
        return path

    repo = fresh('staged')
    (repo / 'record.json').write_text(json.dumps({'note': 'a value that must never be committed',
                                                  'v': fakes['ENTSOE_API_TOKEN']}) + '\n')
    _git(repo, clean, 'add', 'record.json')
    cases.append(_case('a staged file holding a fake ENTSOE_API_TOKEN', _git(repo, holding, 'commit', '--quiet', '-m', 'x'),
                       True, values))
    repo = fresh('message')
    (repo / 'ok.json').write_text('{"day": "2026-09-01"}\n')
    _git(repo, clean, 'add', 'ok.json')
    cases.append(_case('a commit message holding a fake HF_TOKEN',
                       _git(repo, holding, 'commit', '--quiet', '-m', f'note {fakes["HF_TOKEN"]}'), True, values))
    repo = fresh('dump')
    dump = "environ({'" + 'DAGSHUB_USER_TOKEN' + "': '" + 'x' * 8 + "', 'PATH': '/usr/bin'})\n"
    (repo / 'pytest.log').write_text(dump)
    _git(repo, clean, 'add', 'pytest.log')
    name = 'a staged environment dump (no value set: the leak scan, by its shape)'
    if (checkout / 'scripts/leak_scan.py').is_file():
        cases.append(_case(name, _git(repo, clean, 'commit', '--quiet', '-m', 'a log'), True, values))
    else:
        cases.append({'case': name, 'exit': None, 'blocked': None, 'as_expected': True, 'value_in_output': False,
                      'output': [], 'not_applicable': 'scripts/leak_scan.py is not in this tree'})
    repo = fresh('push')
    (repo / 'late.json').write_text(json.dumps({'v': fakes['DAGSHUB_USER_TOKEN']}) + '\n')
    _git(repo, clean, 'add', 'late.json')
    made = _git(repo, clean, 'commit', '--quiet', '-m', 'made where no value was set')
    cases.append({**_case('a push of a commit holding a fake DAGSHUB_USER_TOKEN',
                          _git(repo, holding, 'push', '--quiet', 'origin', 'HEAD:refs/heads/main'), True, values),
                  'commit_exit': made.returncode})
    repo = fresh('data')
    (repo / 'ok.json').write_text('{"day": "2026-09-01"}\n')
    _git(repo, clean, 'add', 'ok.json')
    cases.append(_case('a data-only commit, with fake credentials set',
                       _git(repo, holding, 'commit', '--quiet', '-m', 'live: a data-only record\n\nDelu-Daily: 2026-09-01'),
                       False, values))
    cases.append(_case('its push, with fake credentials set',
                       _git(repo, holding, 'push', '--quiet', 'origin', 'HEAD:refs/heads/main'), False, values))
    return {'schema': 'cp18a-guard-demo-v1', 'run_id': run_id, 'hooks': str(checkout / '.githooks'),
            'fake_names': list(NAMES), 'fake_value_rule': f'sha256("{SALT}:<name>:<run id>")[:40]',
            'cases': cases, 'all_as_expected': all(c['as_expected'] and not c['value_in_output'] for c in cases)}


def scan(run_id: str, paths: list[Path]) -> dict:
    """Which fake values occur, by value, in which files (names and paths only)."""
    fakes = fake_environment(run_id)
    found, files = {}, 0
    for base in paths:
        base = Path(base)
        for path in ([base] if base.is_file() else sorted(p for p in base.rglob('*') if p.is_file())):
            files += 1
            blob = path.read_bytes()
            for name, value in fakes.items():
                if value.encode() in blob:
                    found.setdefault(name, []).append(str(path))
    return {'run_id': run_id, 'files_scanned': files, 'found': found, 'clean': not found}


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ['scan']:
        ap = argparse.ArgumentParser(prog='python -m cp18.guarddemo scan')
        ap.add_argument('--run-id', required=True)
        ap.add_argument('paths', nargs='+', type=Path)
        args = ap.parse_args(argv[1:])
        result = scan(args.run_id, args.paths)
        print(json.dumps(result, indent=1, sort_keys=True))
        return 0 if result['clean'] else 1
    ap = argparse.ArgumentParser(prog='python -m cp18.guarddemo')
    ap.add_argument('--checkout', required=True, type=Path)
    ap.add_argument('--work', required=True, type=Path)
    ap.add_argument('--run-id', required=True)
    ap.add_argument('--out', required=True, type=Path)
    args = ap.parse_args(argv)
    result = demo(args.checkout, args.work, args.run_id)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1, sort_keys=True) + '\n')
    for case in result['cases']:
        if case.get('not_applicable'):
            print(f"{case['case']}: not applicable ({case['not_applicable']})")
            continue
        print(f"{case['case']}: {'blocked' if case['blocked'] else 'passed'} "
              f"({'as expected' if case['as_expected'] else 'NOT AS EXPECTED'}; value in output: {case['value_in_output']})")
    return 0 if result['all_as_expected'] else 1


if __name__ == '__main__':
    sys.exit(main())
