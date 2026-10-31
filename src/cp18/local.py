"""The local fallback on the Owner's Mac: one command, the same binary, the same checks, the same claim rule.

    PYTHONPATH=<clone>/src .venv/bin/python -m cp18.local run --clone <clone> --project /Users/djourno/Downloads/PJM \
        --now <ISO> --inputs <spec> [--remote origin]

- **The same binary:** the official Node.js v26.7.0 at `.local/tools/node-v26.7.0-official/` in the project
  (`cp17.runtime.NODE_RELATIVE`), checked by hash in place; retrieved by hash only if it is absent.
- **The same checks:** the H2 gate (`cp17.hostcheck`) in the pipeline, before the first issuance; the package zip is
  rebuilt from the clone's `src/` and compared with the manifest first.
- **The same claim rule:** the clone is the pipeline's own checkout of the product repository, with
  `core.hooksPath` set to its `.githooks`; the day's record on `main` is the claim (`cp18.claim`), so the runner and
  the fallback never issue the same day twice, and the loser of a race discards its own issuance.
- **The clone** is dedicated to the pipeline (never the Owner's working checkout): created by `git clone` if absent,
  and never written by anything else. Its uncommitted changes refuse the run.

The proposed `launchd` agent (`reports/cp18a/launchd/`) runs this command once a day, after the runner's usual time;
when the runner has already issued, the fallback finds nothing to issue and records its run. It is written, not
installed: installing it is CP-18's activation. This round accepts `--mode dry-run` only (development days).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys


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


def main(argv=None) -> int:
    from cp17 import runtime
    from . import __main__ as cli
    from . import host
    ap = argparse.ArgumentParser(prog='python -m cp18.local')
    sub = ap.add_subparsers(dest='cmd', required=True)
    r = sub.add_parser('run')
    r.add_argument('--clone', required=True, type=Path)
    r.add_argument('--url', help='the product repository, to create the clone if it is absent')
    r.add_argument('--project', required=True, type=Path, help='the folder that holds .local/ (the pinned host files)')
    r.add_argument('--out', required=True, type=Path)
    r.add_argument('--now', required=True)
    r.add_argument('--inputs', required=True)
    r.add_argument('--remote', default='origin')
    r.add_argument('--mode', default='dry-run')
    r.add_argument('--run-label')
    args = ap.parse_args(argv)
    clone, project = args.clone.resolve(), args.project.resolve()
    record = {'clone': ensure_clone(clone, args.url)}
    record['node'] = host.node(project, retrieve=not (project / runtime.NODE_RELATIVE).exists())
    record['package'] = host.package(clone, project)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / 'local-prepare.json').write_text(json.dumps(record, indent=1, sort_keys=True, default=str) + '\n')
    if not record['package']['equal_to_manifest']:
        print('local fallback: the rebuilt package zip is not the frozen one; nothing runs', file=sys.stderr)
        return 2
    return cli.main(['run', '--root', str(clone), '--project', str(project), '--out', str(args.out), '--host', 'local',
                     '--mode', args.mode, '--now', args.now, '--inputs', args.inputs, '--remote', args.remote,
                     *(['--run-label', args.run_label] if args.run_label else [])])


if __name__ == '__main__':
    sys.exit(main())
