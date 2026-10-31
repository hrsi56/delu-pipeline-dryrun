"""Reproduce the pinned probe in a new scratch directory, preserving saved evidence."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--python', type=Path, required=True, help='Python from the pinned isolated probe environment')
    parser.add_argument('--scratch', type=Path, required=True, help='New directory outside the checkout')
    parser.add_argument('--allow-download', action='store_true')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    scratch = args.scratch.resolve()
    if scratch == root or root in scratch.parents or scratch.exists():
        raise ValueError('scratch must be a new directory outside the checkout')
    scratch.mkdir(parents=True)
    copies = ['src/cp15/feasibility_probe.py', 'reports/cp15/protocol.json',
              'reports/cp15/feasibility/model_verification.json',
              'reports/cp15/feasibility/requirements.freeze.txt', 'data/partitions.json']
    for name in copies:
        target = scratch / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root / name, target)
        assert digest(target) == digest(root / name)
    (scratch / 'data/snapshot.parquet').symlink_to(root / 'data/snapshot.parquet')
    cache = root / 'data/cp15-probe-cache/huggingface'
    if cache.exists():
        (scratch / 'data/cp15-probe-cache').mkdir()
        (scratch / 'data/cp15-probe-cache/huggingface').symlink_to(cache)
    elif not args.allow_download:
        raise FileNotFoundError('copy the documented ignored Hugging Face cache first')
    command = [str(args.python.absolute()), 'src/cp15/feasibility_probe.py']
    if not args.allow_download:
        command.append('--offline')
    run = subprocess.run(command, cwd=scratch, check=False)
    if run.returncode:
        raise SystemExit(run.returncode)
    names = ['context.csv', 'future_load.csv', 'input_manifest.json', 'probe_predictions_unscored.csv']
    comparisons = {}
    for name in names:
        relative = Path('reports/cp15/feasibility') / name
        actual, expected = digest(scratch / relative), digest(root / relative)
        comparisons[name] = {'expected_sha256': expected, 'actual_sha256': actual, 'identical': actual == expected}
    report = {'command': command, 'cwd': str(scratch), 'exit_code': run.returncode,
              'comparisons': comparisons, 'all_identical': all(x['identical'] for x in comparisons.values())}
    (scratch / 'reproduction-comparison.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
    if not report['all_identical']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
