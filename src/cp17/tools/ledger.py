"""Read-only access to CP-26's stored forecast ledger and replay state, bound by hash.

The inputs are the extract of the local-only tag `evidence/cp-26` in `.local/artifacts/cp-17/inputs/`.
Every ledger file is checked against CP-26's committed `scores-manifest.json` (SHA-256 pinned in
`cp28.reader`) as it is read, so a stored byte that differs from CP-26's record is refused. Nothing here
calls `src/cp26` or writes to CP-26's paths.
"""
from __future__ import annotations

from datetime import date
import hashlib
import json
import os
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd

from cp28 import reader

RUN = reader.RUN
INPUTS = 'artifacts/cp-17/inputs'
REPLAY_STATE = f'{RUN}/replay-state.json'   # bound by the tag tree's blob (`verify_tag_file`)


def project_root(root) -> Path:
    """The primary checkout that holds `.local/`: `$CP17_PROJECT_ROOT`, or the git common dir's parent."""
    if os.environ.get('CP17_PROJECT_ROOT'):
        return Path(os.environ['CP17_PROJECT_ROOT'])
    try:
        common = subprocess.run(['git', '-C', str(root), 'rev-parse', '--path-format=absolute', '--git-common-dir'],
                                capture_output=True, text=True, check=True).stdout.strip()
        return Path(common).parent
    except (OSError, subprocess.CalledProcessError):
        return Path(root)


def inputs_dir(root) -> Path:
    return project_root(root) / '.local' / INPUTS


def available(root) -> bool:
    return (inputs_dir(root) / RUN / 'forecast-ledger').exists()


def verify_tag_file(root, relative: str) -> str:
    """The extract's file equals the tag tree's blob (git SHA-1); returns its SHA-256."""
    path = inputs_dir(root) / relative
    data = path.read_bytes()
    listing = subprocess.run(['git', '-C', str(project_root(root)), 'ls-tree', 'evidence/cp-26', relative],
                             capture_output=True, text=True, check=True).stdout.split()
    blob = hashlib.sha1(b'blob %d\0' % len(data) + data).hexdigest()
    if not listing or listing[2] != blob:
        raise ValueError(f'{relative} differs from the evidence/cp-26 tree')
    return hashlib.sha256(data).hexdigest()


def entries(root) -> dict:
    return {e['day']: e for e in reader.ledger_entries(root)}


def day(root, when: date | str, _entries=None) -> dict:
    """One ledger day, its bytes checked against the scores manifest and its content digest."""
    e = (_entries or entries(root))[str(when)]
    data = (inputs_dir(root) / e['path']).read_bytes()
    if hashlib.sha256(data).hexdigest() != e['sha256']:
        raise ValueError(f'ledger file hash mismatch: {e["path"]}')
    value = json.loads(data)
    if value['content_sha256'] != e['content_sha256'] or \
            reader.content_digest({k: v for k, v in value.items() if k != 'content_sha256'}) != value['content_sha256']:
        raise ValueError(f'ledger content identity mismatch: {e["path"]}')
    value['_file_sha256'] = e['sha256']
    return value


def iter_days(root, first: date | None = None, last: date | None = None):
    es = entries(root)
    for key in sorted(es):
        d = date.fromisoformat(key)
        if (first is None or d >= first) and (last is None or d <= last):
            yield d, day(root, d, es)


def keys(values) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(pd.to_datetime(values, utc=True)).as_unit('ns')


def on_keys(q, index, scoring) -> np.ndarray | None:
    """A (n, 7) quantile matrix on `index`, restricted to `scoring` keys; None if absent or invalid."""
    if q is None:
        return None
    q = np.asarray(q, float)
    index, scoring = keys(index), keys(scoring)
    if q.shape != (len(index), 7) or not np.isfinite(q).all() or (np.diff(q, axis=1) < 0).any():
        return None
    pos = index.get_indexer(scoring)
    if (pos < 0).any():
        return None
    return q[pos]


def load_replay_state(root) -> dict:
    """CP-26's final replay state (357 MB), checked against the tag tree, with its content digest."""
    verify_tag_file(root, REPLAY_STATE)
    value = json.loads((inputs_dir(root) / REPLAY_STATE).read_text())
    if value.get('content_sha256') != reader.content_digest({k: v for k, v in value.items() if k != 'content_sha256'}):
        raise ValueError('replay state content identity mismatch')
    return value
