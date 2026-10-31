"""Uncached production-parameter fit reproduction and optional origin controls."""
from __future__ import annotations

import argparse
from datetime import date, timedelta
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from cp15.data import FIT_POLICIES, load_inputs, prepare
from cp15.models import fit_day


def fit(data, day, rows):
    central = {'B0': data.naive[rows]}
    logs = []
    for policy in FIT_POLICIES:
        central[policy], records = fit_day(data, day, policy, rows=rows)
        logs.extend(records)
    central['A3'] = (central['A1'] + central['A2']) / 2
    central['A5'] = (central['A1'] + central['A2'] + central['A4']) / 3
    return central, logs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--day', type=date.fromisoformat, required=True)
    parser.add_argument('--fold', choices=[f'fold_{i}' for i in range(1, 6)], required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--causal-controls', action='store_true')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    output = args.output.resolve()
    if output == root or root in output.parents:
        raise ValueError('reproduction output must be outside the reviewed checkout')
    if output.exists():
        raise FileExistsError(output)
    started = time.monotonic()
    data = load_inputs(root)
    rows = data.rows(args.day)
    original, logs = fit(data, args.day, rows)
    folder = root / 'reports/cp15/folds'
    issued = pd.read_parquet(folder / f'{args.fold}-issued.parquet')
    saved_fits = pd.read_parquet(folder / f'{args.fold}-fits.parquet')
    issued = issued.loc[issued.delivery_date.astype(str).eq(str(args.day))]
    saved_fits = saved_fits.loc[saved_fits.delivery_date.astype(str).eq(str(args.day))]
    comparisons = {}
    for policy, values in original.items():
        saved = issued.loc[issued.policy.eq(policy)].sort_values('timestamp_utc')
        assert saved.timestamp_utc.tolist() == data.index[rows].tolist(), policy
        np.testing.assert_allclose(values, saved.central, rtol=1e-10, atol=1e-8)
        comparisons[policy] = {'hours': len(values), 'maximum_absolute_difference': float(np.max(np.abs(values - saved.central.to_numpy())))}
    identity_fields = ['history_start', 'history_end_exclusive', 'n_train', 'train_rows_sha256',
                       'train_target_sha256', 'normalization_rows_sha256', 'model_sha256']
    for record in logs:
        match = saved_fits.loc[saved_fits.policy.eq(record['policy']) & saved_fits.local_hour.eq(record['local_hour'])]
        assert len(match) == 1
        for name in identity_fields:
            assert record[name] == match.iloc[0][name], (record['policy'], record['local_hour'], name)
    result = {'fold': args.fold, 'day': str(args.day), 'production_protocol_unchanged': True,
              'uncached': True, 'comparisons': comparisons, 'model_records_matched': len(logs),
              'identity_fields_matched': identity_fields}
    if args.causal_controls:
        masked = data.frame.copy()
        future = masked.delivery_date >= args.day
        masked.loc[future, 'price_eur_mwh'] = np.nan
        masked.loc[masked.delivery_date > args.day, 'load_forecast_mw'] = 1e8
        masked_values, masked_logs = fit(prepare(masked, data.p), args.day, rows)
        for policy in original:
            np.testing.assert_array_equal(original[policy], masked_values[policy])
        assert [x['model_sha256'] for x in logs] == [x['model_sha256'] for x in masked_logs]
        changed = data.frame.copy()
        changed.loc[changed.delivery_date.eq(args.day - timedelta(days=1)), 'price_eur_mwh'] += 500
        positive, _ = fit(prepare(changed, data.p), args.day, rows)
        deltas = {p: float(np.max(np.abs(positive[p] - original[p]))) for p in FIT_POLICIES}
        assert all(value > 0 for value in deltas.values()), deltas
        result['causal_controls'] = {'D_and_future_target_mask_maximum_delta': 0,
            'future_load_changed_after_D': True, 'all_model_hashes_identical_under_mask': True,
            'positive_D_minus_1_price_add_500_maximum_deltas': deltas}
    result['elapsed_seconds'] = time.monotonic() - started
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
