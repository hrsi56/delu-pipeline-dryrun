"""Decode one GFS run's 50 messages with CP-20's landed decoder (ecCodes environment only).

    <wx-venv>/bin/python -m cp18.live.decode_worker <spec.json>

The spec names the run, its delivery day, the raw-message folder, the output folder and each message's
provenance (field, lead, url, byte range, idx line and hash, bytes, SHA-256, retrieval time, raw file). Each
message is re-hashed, framed (`cp20.gfs.check_frame`), decoded and validated field by field
(`cp20.gfs.decode`), and tied to its model version (`version_evidence`). The run is written in CP-20's layout,
`runs/<run>.npz` (data, lats, lons, fields, leads) and `runs/<run>.json` (status, npz SHA-256, messages with
their decoded metadata), only if all 50 succeed. The last stdout line is a JSON summary. Imports numpy and
`cp20.gfs` only, so it runs in the extraction environment.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

from cp20.gfs import (BOX_LATS, BOX_LONS, FIELDS, LEADS, check_frame, decode, version, version_evidence)


def _default(value):
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    raise TypeError(type(value).__name__)


def main(spec_path: str) -> int:
    spec = json.loads(Path(spec_path).read_text())
    run = dt.date.fromisoformat(spec['run'])
    raw, out = Path(spec['raw_dir']), Path(spec['out'])
    data = np.full((len(FIELDS), len(LEADS), len(BOX_LATS), len(BOX_LONS)), np.nan)
    records, seen = [], set()
    for m in spec['messages']:
        msg = (raw / m['raw_file']).read_bytes()
        if hashlib.sha256(msg).hexdigest() != m['sha256'] or len(msg) != m['bytes']:
            raise SystemExit(f'{m["raw_file"]}: bytes differ from the retrieval record')
        check_frame(msg, m['bytes'])
        meta, box, quantum = decode(msg, run, m['lead'], m['field'])
        evidence = version_evidence(run, m['field'], meta, m.get('endpoint', 'aws'), m['url'])
        data[FIELDS.index(m['field']), LEADS.index(m['lead'])] = box
        seen.add((m['lead'], m['field']))
        record = {k: v for k, v in m.items() if k != 'raw_file'}
        record.update(packing_quantum=quantum, version_evidence=evidence,
                      meta={k: v for k, v in meta.items() if v is not None})
        records.append(record)
    if seen != {(lead, f) for lead in LEADS for f in FIELDS}:
        raise SystemExit(f'{run}: {len(seen)} of {len(LEADS) * len(FIELDS)} target messages decoded')
    (out / 'runs').mkdir(parents=True, exist_ok=True)
    npz = out / 'runs' / f'{run}.npz'
    tmp = out / 'runs' / f'{run}.tmp.npz'
    np.savez_compressed(tmp, data=data, lats=BOX_LATS, lons=BOX_LONS, fields=np.array(FIELDS), leads=np.array(LEADS))
    os.replace(tmp, npz)
    import eccodes
    record = {'status': 'complete', 'run_00z': run.isoformat(), 'delivery_day': spec['delivery_day'],
              'version': version(run), 'npz_sha256': hashlib.sha256(npz.read_bytes()).hexdigest(),
              'messages': records, 'failed_attempts': [], 'locate_traces': {},
              'nonfinite_box_values': int((~np.isfinite(data)).sum()),
              'decoder': {'module': 'cp20.gfs.decode', 'eccodes': eccodes.__version__, 'numpy': np.__version__,
                          'python': sys.version.split()[0]},
              'written_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}
    js = out / 'runs' / f'{run}.json'
    js.write_text(json.dumps(record, sort_keys=True, default=_default))
    print(json.dumps({'run': run.isoformat(), 'json': str(js), 'npz': str(npz), 'npz_sha256': record['npz_sha256'],
                      'messages': len(records), 'decoder': record['decoder']}))
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1]))
