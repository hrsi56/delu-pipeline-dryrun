"""The states' route, built once on the Owner's Mac from CP-26's ledger and CP-17b's payloads (run locally).

- **The freeze-origin states of v4, v2 and B0** (`cp17.runtime.FREEZE_STATE`): CP-26's state at 2026-09-06, after
  its release (the buffers CP-17 recorded), with CP-26's own registrations of 2026-09-05 and 2026-09-06 pending.
  That is CP-17's freeze-origin state, by which 4.7T's shadows continue on 2026-09-07. It is checked against CP-17's
  recorded buffers and against the final state of CP-17b's native shadow run.
- **The canary's payload** (`cp17.runtime.CANARY_DIR`): A1's payload for 2026-09-06, the freeze-origin issuance's,
  checked by its identity against CP-17b's freeze-origin artifact.
- **A1's search state** `k13.json` stays where the extract holds it (`cp17.runtime.K13`).

    OPENBLAS_NUM_THREADS=1 ... PYTHONPATH=src python -m cp17c.state --record <file>
"""
from __future__ import annotations

import argparse
from datetime import date, timedelta
import json
from pathlib import Path
import shutil

from cp17 import dailyfit, hostcheck, runtime

FREEZE_ORIGIN = date(2026, 9, 6)
CP17B_ARTIFACT = 'docs/track-b/evidence/cp-17b/artifact-2026-09-06.json'
CP17_MANIFEST = 'docs/track-b/evidence/cp-17/freeze-manifest.json'
CANARY_SOURCE = '.local/artifacts/cp-17b/runs/native-parity/payloads'
NATIVE_SHADOW_RUN = '.local/artifacts/cp-17b/runs/shadow-native/shadow-run.json'


def dump(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=1, sort_keys=True, allow_nan=False) + '\n')


def freeze_origin_state(root: Path):
    """CP-26's state at the freeze origin, from its ledger: `cp17.tools.exercise.start` (2026-09-05 pending), with
    2026-09-06's registration added as `start` adds the day before's."""
    from cp17.shadow import POLICIES, ShadowStates
    from cp17.tools import exercise, ledger, payloads
    ctx = payloads.context(root)
    state, record = exercise.start(root, ctx, FREEZE_ORIGIN)
    value = state.to_dict()
    day = ledger.day(root, FREEZE_ORIGIN)
    for p in POLICIES:
        item = day['policies'][f'nominal/{p}']
        if not item['error_eligible']:
            continue
        entry = {'day': str(FREEZE_ORIGIN), 'index': list(item['index']), 'central': list(item['native_central'])}
        if p != 'B0':
            entry['scale'] = list(item['issued_scale'])
        (value['b0'] if p == 'B0' else value['h'][p])['pending'].append(entry)
    return ShadowStates.from_dict(json.loads(json.dumps(value, allow_nan=False))), record


def _carried(value: dict) -> dict:
    """What a state carries forward: each policy's released buffer, pending registrations and last origin. The
    bookkeeping lists `consumed` and `trace` record history (the native run's go back to 2026-08-24) and are not
    compared."""
    keep = ('buffer', 'pending', 'last_origin')
    return {'h': {p: {k: v[k] for k in keep} for p, v in value['h'].items()}, 'b0': {k: value['b0'][k] for k in keep}}


def build(root: Path, project: Path) -> dict:
    from cp17.shadow import POLICIES, ShadowStates
    root, project = Path(root), Path(project)
    state, start_record = freeze_origin_state(root)
    recorded = json.loads((root / CP17_MANIFEST).read_text())['initial_state']['buffers_sha256']
    digest = state.digest()
    if any(digest[p] != recorded[p] for p in POLICIES):
        raise ValueError(f"the rebuilt buffers are not CP-17's recorded freeze-origin state: {digest}")
    native = json.loads((project / NATIVE_SHADOW_RUN).read_text())['state']
    equal_native = _carried(json.loads(json.dumps(state.to_dict(), allow_nan=False))) == _carried(native)
    dump(project / runtime.FREEZE_STATE, state.to_dict())
    canary = project / runtime.CANARY_DIR
    canary.mkdir(parents=True, exist_ok=True)
    for suffix in ('.json', '.npz'):
        shutil.copyfile(project / CANARY_SOURCE / f'{hostcheck.CANARY_STEM}{suffix}', canary / f'{hostcheck.CANARY_STEM}{suffix}')
    arrays, meta = dailyfit.load(canary / f'{hostcheck.CANARY_STEM}.npz', canary / f'{hostcheck.CANARY_STEM}.json')
    body = {k: v for k, v in meta.items() if k != 'payload_sha256'}
    expected = json.loads((root / CP17B_ARTIFACT).read_text())['lineage']['inputs']['A1']
    if meta['payload_sha256'] != dailyfit.payload_digest(arrays, body) or meta['payload_sha256'] != expected:
        raise ValueError('the canary payload is not the freeze-origin issuance\'s')
    if not equal_native:
        raise ValueError("the rebuilt state's buffers, pending registrations or origin differ from CP-17b's native "
                         "shadow run's final state")
    return {'schema': 'cp17c-state-route-v1', 'freeze_origin_state': {
                'path': runtime.FREEZE_STATE, 'file_sha256': runtime.sha256_file(project / runtime.FREEZE_STATE),
                'buffers_sha256': digest, 'equal_to_cp17_recorded': True,
                'equal_to_cp17b_native_shadow_run_final_state': equal_native,
                'compared': 'buffers, pending registrations and last origin (consumed and trace are history)',
                'pending': {p: [i['day'] for i in (state.to_dict()['b0'] if p == 'B0' else state.to_dict()['h'][p])['pending']]
                            for p in POLICIES},
                'source': "CP-26's nominal ledger (cp17.tools.exercise.start at 2026-09-06, and 2026-09-06's registration)",
                'start': start_record},
            'canary_payload': {'path': runtime.CANARY_DIR, 'payload_sha256': meta['payload_sha256'],
                               'equal_to_cp17b_freeze_origin_input': True, 'source': CANARY_SOURCE,
                               'files_sha256': {p.name: runtime.sha256_file(p) for p in sorted(canary.iterdir())}},
            'k13': {'path': runtime.K13, 'file_sha256': runtime.sha256_file(project / runtime.K13)}}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='.', type=Path)
    ap.add_argument('--record', required=True, type=Path)
    args = ap.parse_args(argv)
    from cp17.tools import ledger
    value = build(args.root, ledger.project_root(args.root))
    dump(args.record, value)
    print(json.dumps(value['freeze_origin_state']['buffers_sha256']), value['canary_payload']['payload_sha256'][:12])
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
