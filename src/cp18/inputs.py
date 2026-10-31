"""What a run asks of its inputs, and the two providers: the live builder (`cp18.live`) and prebuilt payloads.

The pipeline asks three things for each delivery day D:
- `truth(D)`: the published prices the shadow states' release reads (`cp17.shadow.ShadowStates.release`), a function
  of the requested keys that is NaN unless every key is present, holding delivery days up to D−1 only (published
  before D's cutoff);
- `search_file(k)`: A1's search state for search k when it is a pinned file (S_13's `k13.json` at
  `cp17.runtime.K13`); S_14 and later come from the store's `search` entries, never from here;
- `build(D, buffers, search_file, out)`: the day's four payloads (A1, v4, v2, B0) in `out`, built with the buffers the
  store's state holds after D−2's release, and the judgement of the day's inputs: a dict with `payloads` (the folder),
  `gfs_available`, `shown` (every required input shown to precede D's cutoff) with its reasons, `availability` (each
  input's source, publication time and basis, retrieval time and SHA-256), `payload_sha256` and `data_refresh_utc`.

`PrebuiltInputs` serves payloads the development builder already made (`cp17.tools.payloads`, CP-17b's runs), for
rehearsals: it checks that each payload's buffers equal the state's, so a store that drifted from the payloads is
refused. Its days are development days, admitted under the tier register and the Owner's ruling 1 (the load
forecast's recorded exception for the stored development history), which its availability record says.
"""
from __future__ import annotations

from datetime import date, timedelta
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np

from cp25 import guard

POLICIES = ('A1', 'v4', 'v2', 'B0')
DEVELOPMENT_BASIS = ("development day built by cp17.tools.payloads from the recorded snapshot (pulled 2026-09-08): "
                     "admitted under the tier register and the Owner's ruling 1 (the load forecast's recorded "
                     "exception for the stored development history); development data only")


def buffer_sha256(buffer: list[dict]) -> str:
    return hashlib.sha256(json.dumps(buffer, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


class SnapshotTruth:
    """Published prices from the committed snapshot, through the boundary guard, for delivery days <= D−1."""

    def __init__(self, root):
        import pandas as pd
        frame = pd.read_parquet(Path(root) / 'data/snapshot.parquet', columns=['timestamp_utc', 'delivery_date', 'price_eur_mwh'],
                                filters=[('delivery_date', '<=', guard.BOUNDARY)])
        index = pd.DatetimeIndex(pd.to_datetime(frame['timestamp_utc'], utc=True)).as_unit('ns')
        guard.admit_rows(index)
        self.index, self.days = index, np.asarray(pd.Index(frame['delivery_date']).astype(str))
        self.y = np.asarray(frame['price_eur_mwh'], float)

    def __call__(self, day: date):
        import pandas as pd
        guard.admit_delivery_day(day)
        keep = self.days <= str(day - timedelta(days=1))
        index, y = self.index[keep], self.y[keep]

        def at(keys):
            pos = index.get_indexer(pd.DatetimeIndex(pd.to_datetime(keys, utc=True)).as_unit('ns'))
            if (pos < 0).any():
                return np.full(len(keys), np.nan)
            return np.asarray(y[pos], float)
        return at


def pinned_search(project: Path, k: int | None) -> Path | None:
    """A1's pinned search state: S_13 at `cp17.runtime.K13`; earlier searches (development days only) beside it."""
    from cp17 import runtime
    if k is None or k >= 14:
        return None
    path = Path(project) / runtime.K13
    return path if k == 13 else path.parent / f'k{k:02d}.json'


class PrebuiltInputs:
    def __init__(self, root, project, payloads, gfs_late: tuple = ()):
        self.root, self.project, self.payloads = Path(root), Path(project), Path(payloads)
        self.truth = SnapshotTruth(root)
        self.gfs_late = {str(d) for d in gfs_late}

    def search_file(self, k):
        return pinned_search(self.project, k)

    def build(self, day: date, *, buffers: dict, search_file: Path, out: Path) -> dict:
        from cp17 import dailyfit
        guard.admit_delivery_day(day)
        out = Path(out)
        out.mkdir(parents=True, exist_ok=True)
        shas, problems = {}, []
        for policy in POLICIES:
            meta_path = self.payloads / f'{policy}-{day}.json'
            npz = self.payloads / f'{policy}-{day}.npz'
            arrays, meta = dailyfit.load(npz if npz.exists() else None, meta_path)
            body = {k: v for k, v in meta.items() if k != 'payload_sha256'}
            if meta.get('payload_sha256') != dailyfit.payload_digest(arrays, body):
                problems.append(f'{policy}: the payload\'s identity does not verify')
            if policy != 'A1':
                carried = meta['b0_buffer' if policy == 'B0' else 'h_buffer']
                if buffer_sha256(carried) != buffer_sha256(buffers[policy]):
                    problems.append(f'{policy}: the payload\'s buffer is not the store state\'s after the release')
            for path in (meta_path, npz):
                if path.exists():
                    shutil.copyfile(path, out / path.name)
            shas[policy] = meta['payload_sha256']
        if problems:
            raise ValueError('the prebuilt payloads do not match the store: ' + '; '.join(problems))
        cutoff = guard.issue_time(day).isoformat()
        return {'payloads': str(out), 'payload_sha256': shas, 'gfs_available': str(day) not in self.gfs_late,
                'shown': True, 'reasons': [], 'data_refresh_utc': '',
                'availability': [{'input': 'all', 'basis': DEVELOPMENT_BASIS, 'cutoff_utc': cutoff, 'admitted': True}],
                'source': 'prebuilt development payloads', 'search_file': str(search_file)}
