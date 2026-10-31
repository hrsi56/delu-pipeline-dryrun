"""The live-input payload builder (`cp18.live`) behind the pipeline's inputs interface (`cp18.inputs`).

**Acquisition.** Before a day is built, every archive day from the day after the base's end through D must be in the
day's input archive (`live/inputs/days/<day>/`, committed with the day's records). A missing archive day is acquired:
in this round only by replaying recorded responses (`cp18.live.recorded.archive_day`, through the same retrieval
clients and archive writers the live route uses); CP-18's activation adds the live transport. Nothing is retrieved
from the network here.

**The day.** `build` calls `cp18.live.build.build_day` with the store state's buffers after D−2's release and maps its
result onto the pipeline's: `built` gives the payloads; `not_shown` and `failed` leave the day open with the cause
"a live day's inputs that cannot be shown to precede its cutoff", the second because no payload could be built from
them (the build's own cause is kept in the record); `search_state_missing` leaves it open with "S_14's state not in
place". The day's `data_refresh_utc` is the latest retrieval time among its archive records.

**The base.** `base_end` is the last day of the committed history. Live operation uses 2026-09-06, the snapshot's
end; a dry run on development days uses the day before the recorded responses' first day.
"""
from __future__ import annotations

from datetime import date, timedelta
import json
from pathlib import Path

from cp25 import guard

from .inputs import pinned_search
from .store import INPUTS_PATH

GFS_SOURCE = 'gfs'
REPLAY = 'replay'


class LiveInputs:
    def __init__(self, root, project, recorded, base_end: date | None = None, source: str = 'smard'):
        self.root, self.project, self.recorded = Path(root), Path(project), Path(recorded)
        manifest = json.loads((self.recorded / 'manifest.json').read_text())
        days = sorted(date.fromisoformat(d) for d in manifest['days'])
        self.recorded_days = set(days)
        self.base_end = base_end or (days[0] - timedelta(days=1))
        guard.admit_delivery_day(self.base_end)
        self.source = source
        self.acquired: list[str] = []

    @property
    def archive_root(self) -> Path:
        return self.root / INPUTS_PATH

    def acquire(self, day: date) -> None:
        """Every archive day in (base_end, day], replayed from the recorded responses where it is missing."""
        from .live import archive as AR, recorded as R
        guard.admit_delivery_day(day)
        archive = AR.Archive(self.archive_root)
        x = self.base_end + timedelta(days=1)
        while x <= day:
            if not archive.has(x):
                if x not in self.recorded_days:
                    raise FileNotFoundError(f'no recorded responses for archive day {x}')
                item = R.archive_day(archive, self.recorded / REPLAY, x, self.root, self.source,
                                     [self.recorded / GFS_SOURCE])
                if item['verify']:
                    raise ValueError(f'archive day {x} does not verify: {item["verify"]}')
                self.acquired.append(str(x))
            x += timedelta(days=1)

    def truth(self, day: date):
        from .live.build import published_prices
        if day - timedelta(days=1) > self.base_end:
            self.acquire(day - timedelta(days=1))
        return published_prices(day, archive=self.archive_root, base_end=self.base_end, root=self.root)

    def search_file(self, k):
        return pinned_search(self.project, k)

    def build(self, day: date, *, buffers: dict, search_file: Path, out: Path) -> dict:
        from .live.archive import Archive
        from .live.build import build_day
        self.acquire(day)
        result = build_day(day, archive=self.archive_root, base_end=self.base_end, buffers=buffers,
                           search_dir=Path(search_file).parent, out=out, root=self.root)
        status = result['status']
        archive = Archive(self.archive_root)
        retrieved = [r.get('retrieved_utc') or '' for r in archive.records(day)]
        out_value = {'payloads': str(out), 'source': 'cp18.live (replayed recorded responses)', 'build_status': status,
                     'build_cause': result.get('cause'), 'gfs_available': bool(result.get('gfs_available')),
                     'gfs': result.get('gfs'), 'availability': result.get('availability'), 'search': result.get('search'),
                     'history': result.get('history'), 'problems': result.get('problems'), 'timings': result.get('timings'),
                     'payload_sha256': {p: v['payload_sha256'] for p, v in (result.get('payloads') or {}).items()},
                     'data_refresh_utc': max((r for r in retrieved if r), default=''),
                     'archive_days': sorted(str(d) for d in archive.days() if self.base_end < d <= day)}
        out_value['shown'] = status == 'built'
        out_value['reasons'] = [] if status == 'built' else [result.get('cause') or status]
        if status == 'search_state_missing':
            out_value['search_missing'] = True
        return out_value
