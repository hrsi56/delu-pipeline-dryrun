"""Item 6's negative controls for the retrain comparison, on the real freeze-origin payloads.

1. **A changed seed reports "different".** A1's payload with member 0's seed changed, refitted natively and
   compared with the issued artifact's result by `cp17.retrain.compare`.
2. **A changed training-window value reports "different".** A1's payload with one training day's price
   changed (2026-08-20 hour 12, +1 EUR/MWh), refitted and compared. The record keeps `measured_status`, what the
   quantiles and selections alone say: CP-17's first run found it "within ±0.01 EUR/MWh" (the change moved the
   quantiles by under 0.01 with every selection matched), which is why the comparison binds the training window's
   identity first.
3. **A changed outcome of the target day leaves the payload unchanged.** The A1, v2, v4 and B0 payloads are
   rebuilt from inputs whose target-day prices are all changed (+50 EUR/MWh); every payload digest and every
   file byte must equal the original's (the embargo).

    OPENBLAS_NUM_THREADS=1 ... PYTHONPATH=src python -m cp17.tools.controls --payloads <dir> --issued <result.json> --out <json>
"""
from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import date
import json
from pathlib import Path
import tempfile

import numpy as np

from .. import dailyfit, retrain
from . import ledger, payloads, states

DAY = date(2026, 9, 6)
CHANGED_DAY = date(2026, 8, 20)


def refit(arrays: dict, meta: dict) -> dict:
    body = {k: v for k, v in meta.items() if k not in ('payload_sha256', 'array_sha256')}
    body['array_sha256'] = {name: dailyfit.array_hash(arrays[name]) for name in sorted(arrays)}
    body['payload_sha256'] = dailyfit.payload_digest(arrays, body)
    return dailyfit.fit_a1(arrays, body)


def seed_control(arrays: dict, meta: dict, issued: dict) -> dict:
    changed = json.loads(json.dumps(meta))
    changed['members'][0]['seed'] = int(changed['members'][0]['seed']) + 1
    result = refit(arrays, changed)
    cmp = retrain.compare('A1', issued, result)
    return {'change': f"member 0 seed {meta['members'][0]['seed']} -> {changed['members'][0]['seed']}",
            'status': cmp['status'], 'measured_status': cmp['measured_status'], 'selections_match': cmp['selections_match'],
            'max_abs_eur': cmp['deviation'].get('max_abs_eur'), 'reports_different': cmp['status'] == 'different'}


def window_control(arrays: dict, meta: dict, issued: dict) -> dict:
    changed = {k: v.copy() for k, v in arrays.items()}
    row = int((np.datetime64(CHANGED_DAY) - np.datetime64(meta['first'])).astype(int))
    if int(changed['days'][row]) != (CHANGED_DAY - date(1970, 1, 1)).days:
        raise ValueError('the changed day is not where the day table puts it')
    changed['price'][row, 12] += 1.0
    result = refit(changed, meta)
    cmp = retrain.compare('A1', issued, result)
    return {'change': f'price of {CHANGED_DAY} slot 12 +1 EUR/MWh', 'status': cmp['status'],
            'measured_status': cmp['measured_status'], 'payload_sha256_equal': cmp['payload_sha256_equal'],
            'selections_match': cmp['selections_match'], 'max_abs_eur': cmp['deviation'].get('max_abs_eur'),
            'reports_different': cmp['status'] == 'different'}


def changed_context(root: Path, ctx):
    """The inputs rebuilt end to end, as `cp26.members.Context.build` builds them, from a snapshot frame whose
    target-day prices are all +50 EUR/MWh: CP-15's preparation, the weather designs and DDNN-2's day table."""
    from cp15.data import prepare
    from cp20.components import augment
    from cp24 import design as G
    from cp24.member import Keys
    from cp26 import harness as H
    frame = H.load_frame(root, H.NEW)
    on_day = frame['delivery_date'] == DAY
    if on_day.sum() != 24:
        raise ValueError('the target day is not 24 rows in the frame')
    frame.loc[on_day, 'price_eur_mwh'] += 50.0
    data = prepare(frame, H.protocol_recipe(root), H.spec_for(root, H.NEW))
    data_aug, present = augment(data, ctx.design)
    wx, present_w = H.weather_matrix(ctx.design, data)
    dd = G.build(data, ctx.design)
    return replace(ctx, data=data, data_m=data, data_aug=data_aug, wx=wx, present=present_w, dd=dd,
                   keys=Keys.from_data(data, dd))


def embargo_control(root: Path, folder: Path) -> dict:
    ctx = payloads.context(root)
    rebuilt = states.rebuild(root, ctx, [DAY])['buffers']
    changed = changed_context(root, ctx)
    if np.array_equal(changed.data.y, ctx.data.y, equal_nan=True):
        raise ValueError('the control did not change the inputs')
    out = {}
    with tempfile.TemporaryDirectory() as tmp:
        for policy, build in (('A1', lambda c: payloads.a1_payload(c, DAY, payloads.ensemble(ledger.inputs_dir(root), DAY))),
                              ('v2', lambda c: payloads.v2_payload(c, DAY, rebuilt['v2'][DAY])),
                              ('v4', lambda c: payloads.v4_payload(c, DAY, rebuilt['v4'][DAY])),
                              ('B0', lambda c: payloads.b0_payload(c, DAY, rebuilt['B0'][DAY]))):
            arrays, meta = build(changed)
            written = payloads.write(arrays, meta, Path(tmp), f'{policy}-{DAY}')
            same_files = all(Path(written[k]).read_bytes() == (folder / Path(written[k]).name).read_bytes()
                             for k in ('npz', 'meta') if k in written)
            original = json.loads((folder / f'{policy}-{DAY}.json').read_text())
            out[policy] = {'payload_sha256_equal': meta['payload_sha256'] == original['payload_sha256'],
                           'file_bytes_equal': same_files}
    out['target_day_changed'] = f'every price of {DAY} +50 EUR/MWh'
    out['unchanged'] = all(v['payload_sha256_equal'] and v['file_bytes_equal'] for k, v in out.items() if isinstance(v, dict))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='.', type=Path)
    ap.add_argument('--payloads', required=True, type=Path)
    ap.add_argument('--issued', required=True, type=Path, help='the issued A1 result for 2026-09-06 (JSON)')
    ap.add_argument('--out', required=True, type=Path)
    args = ap.parse_args(argv)
    arrays, meta = dailyfit.load(args.payloads / f'A1-{DAY}.npz', args.payloads / f'A1-{DAY}.json')
    issued = json.loads(args.issued.read_text())
    record = {'schema': 'cp17-retrain-controls-v1', 'issued_quantiles_sha256': issued['quantiles_sha256'],
              'seed': seed_control(arrays, meta, issued), 'training_window': window_control(arrays, meta, issued),
              'target_day_outcome': embargo_control(args.root, args.payloads)}
    record['all_pass'] = bool(record['seed']['reports_different'] and record['training_window']['reports_different']
                              and record['target_day_outcome']['unchanged'])
    args.out.write_text(json.dumps(record, indent=1, sort_keys=True) + '\n')
    print(record['seed']['status'], record['training_window']['status'], record['target_day_outcome']['unchanged'])
    return 0 if record['all_pass'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
