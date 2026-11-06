"""CP-18b's controls on a built bundle: every verification rule, the gate, and §16.4's positive controls.

**Negative controls** (`verify_controls`): one tampered copy per rule the frozen bundle verifier checks (the rules of
`docs/track-b/evidence/cp-17/payload-schema.md`), plus R7's raised error, the frozen-identity and red-line checks and
the recomputation. Each must be refused by the gate (`cp18.bundle.gate`): refused means nothing is published. Where
a control must reach a later rule, the tampered file is re-hashed into the manifest, so only that rule can refuse it.

**Variants for the Space** (`variants`): bundles the Space must show in a given state, built through the real build
path where the state can arise there (pending, stale, failed), and by a re-hashed edit where it cannot (a tampered
file, a mismatched committed record, a changed seed, missing/zero/negative prices).

    PYTHONPATH=src python -m cp18.controls --bundle <dir> --record <file> [--variants <dir> --live <store copy> ...]
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import io
import json
from pathlib import Path

import numpy as np

from .bundle import dump, gate, read


def _bytes(value) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n').encode()


def rehash(files: dict, changed: dict[str, bytes], drop=(), add_listed: dict | None = None) -> dict:
    """`files` with `changed` replaced and re-hashed into the manifest; `drop` removed from both."""
    out = {k: v for k, v in files.items() if k not in drop}
    out.update(changed)
    manifest = json.loads(files['manifest.json'])
    manifest['files'] = [f for f in manifest['files'] if f['path'] not in drop]
    for f in manifest['files']:
        if f['path'] in changed:
            f.update(sha256=hashlib.sha256(changed[f['path']]).hexdigest(), bytes=len(changed[f['path']]))
    for path, kind in (add_listed or {}).items():
        manifest['files'].append({'path': path, 'sha256': hashlib.sha256(out[path]).hexdigest(), 'bytes': len(out[path]),
                                  'kind': kind})
    out['manifest.json'] = _bytes(manifest)
    return out


def _flip(data: bytes, at: int | None = None) -> bytes:
    raw = bytearray(data)
    i = len(raw) // 2 if at is None else at
    raw[i] ^= 0x01
    return bytes(raw)


def _npz(data: bytes) -> dict:
    with np.load(io.BytesIO(data), allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def _npz_bytes(arrays: dict) -> bytes:
    from cp17.dailyfit import save_npz
    buffer = io.BytesIO()
    save_npz(buffer, arrays)
    return buffer.getvalue()


def _repayload(meta: dict, arrays: dict) -> tuple[bytes, bytes]:
    """A training window whose own identity holds again after an edit (so only a later rule can refuse it)."""
    from cp17.dailyfit import array_hash, payload_digest
    meta = dict(meta, array_sha256={k: array_hash(v) for k, v in arrays.items()})
    meta['payload_sha256'] = payload_digest(arrays, {k: v for k, v in meta.items() if k != 'payload_sha256'})
    return json.dumps(meta, sort_keys=True).encode(), _npz_bytes(arrays)


def _stem(files: dict, policy: str = 'A1') -> str:
    return next(k[:-5] for k in sorted(files) if k.startswith(f'{policy}-') and k.endswith('.json'))


def _rehearsal_index(files: dict) -> int:
    """The recomputed product window a control alters: the development rehearsal's (a dry run's bundle) or, when it
    scores no day, the prospective record's (the live product's bundle, CP-18c)."""
    ev = json.loads(files['evaluation.json'])
    for window in ('rehearsal', 'prospective'):
        for i, w in enumerate(ev['windows']):
            hit = (w['evidence_class'].startswith('development rehearsal') if window == 'rehearsal'
                   else w['window'] == 'prospective')
            if hit and w['policy'] == 'product' and w['record']['pooled'].get('scored_days'):
                return i
    return next(i for i, w in enumerate(ev['windows'])
                if w['evidence_class'].startswith('development rehearsal') and w['policy'] == 'product')


def verify_controls(files: dict, root=None) -> dict:
    """Every negative control on the gate; each entry records whether it was refused and the first reasons."""
    from . import r7
    c = {}
    manifest = json.loads(files['manifest.json'])
    # rule 1: the manifest validates
    m = copy.deepcopy(manifest)
    m['freshness']['state'] = 'late'
    c['1_manifest_invalid_state_late'] = dict(files, **{'manifest.json': _bytes(m)})
    # rule 2: every listed file exists, with its hash and size; an unreadable file is a refusal
    c['2_listed_file_missing'] = {k: v for k, v in files.items() if k != 'outcomes.json'}
    c['2_one_byte_changed_unhashed'] = dict(files, **{'issued-forecasts.json': _flip(files['issued-forecasts.json'])})
    c['2_unreadable_json_rehashed'] = rehash(files, {'outcomes.json': b'{"schema": '})
    c['2_unreadable_npz_rehashed'] = rehash(files, {f'{_stem(files)}.npz': b'PK\x03\x04 not an archive'})
    # rule 3: nothing served unlisted
    c['3_unlisted_file_served'] = dict(files, **{'unlisted.json': b'{}'})
    # rule 4: each file is its listed kind and validates
    m = copy.deepcopy(manifest)
    for f in m['files']:
        if f['path'] == 'artifact-arrays.npz':
            f['kind'] = 'series'
    c['4_wrong_kind'] = dict(files, **{'manifest.json': _bytes(m)})
    out = json.loads(files['outcomes.json'])
    out['prices'][0]['extra'] = 1
    c['4_outcomes_fail_their_schema'] = rehash(files, {'outcomes.json': _bytes(out)})
    # rule 5: a training window's identity holds
    stem = _stem(files)
    c['5_training_window_bit_flipped'] = rehash(files, {f'{stem}.npz': _flip(files[f'{stem}.npz'])})
    meta = json.loads(files[f'{stem}.json'])
    meta['members'][0]['seed'] += 1
    c['5_training_window_meta_changed'] = rehash(files, {f'{stem}.json': json.dumps(meta, sort_keys=True).encode()})
    # rule 6: the embargo (future-label leakage), with the window's own identity made to hold again
    arrays = _npz(files[f'{stem}.npz'])
    arrays['price'] = arrays['price'].copy()
    arrays['price'][-1] = 50.0
    jb, nb = _repayload(json.loads(files[f'{stem}.json']), arrays)
    c['6_future_label_leak_target_day_price'] = rehash(files, {f'{stem}.json': jb, f'{stem}.npz': nb})
    b0 = _stem(files, 'B0')
    meta = json.loads(files[f'{b0}.json'])
    if meta.get('b0_buffer'):
        meta['b0_buffer'][-1]['day'] = meta['day']
        jb, nb = _repayload(meta, _npz(files[f'{b0}.npz']))
        c['6_released_error_after_d_minus_2'] = rehash(files, {f'{b0}.json': jb, f'{b0}.npz': nb})
    # rule 7: the artifact verifies, the manifest names it, and its day's forecast is its own
    art = json.loads(files['artifact.json'])
    art['display'] = 'edited'
    c['7_artifact_identity_altered'] = rehash(files, {'artifact.json': _bytes(art)})
    m = copy.deepcopy(manifest)
    m['artifact']['artifact_sha256'] = '0' * 64
    c['7_manifest_names_another_artifact'] = dict(files, **{'manifest.json': _bytes(m)})
    fc = json.loads(files['issued-forecasts.json'])
    day = json.loads(files['artifact.json'])['delivery_day']
    target = next(d for d in fc['days'] if d['delivery_day'] == day)
    target['quantiles'][0][3] += 0.5
    c['7_issued_forecast_not_the_artifacts'] = rehash(files, {'issued-forecasts.json': _bytes(fc)})
    # R7: a raised error is a refused record
    c['r7_verifier_raises'] = 'raises'
    # §16.4: altered identities, and the red line
    m = copy.deepcopy(manifest)
    m['policy']['policy_identity_sha256'] = 'f' * 64
    c['identity_policy_altered'] = dict(files, **{'manifest.json': _bytes(m)})
    m = copy.deepcopy(manifest)
    m['policy']['package_sha256'] = 'e' * 64
    c['identity_package_altered'] = dict(files, **{'manifest.json': _bytes(m)})
    series = json.loads(files['series.json'])
    series['last_day'] = '2026-09-07'          # a calendar date one day past the boundary, never a served price
    c['red_line_series_past_the_boundary'] = rehash(files, {'series.json': _bytes(series)})
    # §19.5: a mismatched recomputation blocks publication
    c['recompute_mismatch'] = mismatch(files)
    record = {}
    for name, variant in c.items():
        if variant == 'raises':
            def boom(_files):
                raise KeyError('malformed')
            verdict = r7.verify(files, verifier=boom)
            record[name] = {'refused': not verdict['accepted'], 'reasons': verdict['problems'][:2]}
            continue
        verdict = gate(variant, root)
        reasons = (verdict['verify']['problems'][:2] if not verdict['verify']['accepted'] else
                   (verdict.get('identity') or [])[:2] or [verdict['reason']])
        record[name] = {'refused': not verdict['publishable'], 'reasons': reasons}
    return {'controls': record, 'all_refused': all(v['refused'] for v in record.values()),
            'served_bundle': gate(files, root)['publishable']}


def mismatch(files: dict) -> dict:
    """The rehearsal record's pooled MAE moved by 0.01 EUR/MWh and re-hashed: it verifies, and the recomputation ✗."""
    ev = json.loads(files['evaluation.json'])
    i = _rehearsal_index(files)
    ev['windows'][i]['record']['pooled']['mae_eur'] += 0.01
    return rehash(files, {'evaluation.json': _bytes(ev)})


def seed_variant(files: dict) -> dict:
    """Member 0's seed changed in the served training window, its identity re-made: a retrain on it is "different"."""
    stem = _stem(files)
    meta = json.loads(files[f'{stem}.json'])
    meta['members'][0]['seed'] += 1
    jb, nb = _repayload(meta, _npz(files[f'{stem}.npz']))
    return rehash(files, {f'{stem}.json': jb, f'{stem}.npz': nb})


def prices_variant(files: dict) -> dict:
    """§16.4's missing, zero and negative prices: three of the artifact day's prices made null, 0 and −5 EUR/MWh,
    the day's committed records recomputed natively (`cp18.evaluation`), so the bundle stays publishable and the Space
    shows the day as partial."""
    from . import evaluation as EV, spaceglue as G
    out = json.loads(files['outcomes.json'])
    day = json.loads(files['artifact.json'])['delivery_day']
    rows = [p for p in out['prices'] if p['delivery_day'] == day]
    rows[0].update(price=None, published_utc='', revised_utc='')
    rows[1]['price'] = 0.0
    rows[2]['price'] = -5.0
    forecasts = json.loads(files['issued-forecasts.json'])
    b0, _ = G.b0_items(files, [d['delivery_day'] for d in forecasts['days']])
    ev = json.loads(files['evaluation.json'])
    static = [w for w in ev['windows'] if not G.recomputed(w)]
    ev2 = EV.evaluation_bundle(static, EV.daily_windows(forecasts, out, b0))
    return rehash(files, {'outcomes.json': _bytes(out), 'evaluation.json': _bytes(ev2)})


def write(files: dict, folder: Path) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    for old in folder.iterdir():
        old.unlink()
    for name, data in files.items():
        (folder / name).write_bytes(data)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog='python -m cp18.controls')
    ap.add_argument('--bundle', required=True, type=Path)
    ap.add_argument('--record', required=True, type=Path)
    ap.add_argument('--root', default='.', type=Path)
    ap.add_argument('--variants', type=Path, help='write the Space variants (tampered, mismatch, seed, prices) here')
    args = ap.parse_args(argv)
    files = read(args.bundle)
    value = verify_controls(files, args.root)
    if args.variants:
        made = {'tampered': dict(files, **{'issued-forecasts.json': _flip(files['issued-forecasts.json'])}),
                'mismatch': mismatch(files), 'seed': seed_variant(files), 'prices': prices_variant(files)}
        value['variants'] = {}
        for name, variant in made.items():
            write(variant, args.variants / name)
            v = gate(variant, args.root)
            value['variants'][name] = {'publishable': v['publishable'], 'reason': v['reason']}
    args.record.parent.mkdir(parents=True, exist_ok=True)
    args.record.write_text(json.dumps(value, indent=1, sort_keys=True, default=str) + '\n')
    print('controls', len(value['controls']), 'all refused' if value['all_refused'] else 'NOT ALL REFUSED',
          '| served bundle publishable:', value['served_bundle'], '|', value.get('variants', ''))
    return 0 if value['all_refused'] and value['served_bundle'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
