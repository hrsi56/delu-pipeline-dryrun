"""E1 metadata/training-only feasibility; no fits or comparison scores."""
from __future__ import annotations
from datetime import timedelta
import importlib.metadata
import json
import shutil
import sys
import numpy as np
import pandas as pd
import psutil
from cp15.data import sha, day_hours, origin_utc, history_start
from .inputs import load, expected, identities
from .budget import atomic, CAPS


def run(root, output):
    keys = expected(root)
    counts = keys.groupby('fold').size().tolist()
    if counts != [2160,2159,2112,2160,2156]: raise ValueError('original target count mismatch')
    _, hashes = identities(root)
    from delu_forecast.folds import load_partition_spec
    spec = load_partition_spec(root/'data/partitions.json')
    folds = []; all_missing = 0; total_dates = 0; support_failures = []
    for f in spec.development_folds:
        first = f.evaluation.start
        data, _ = load(root,before=first)
        admission = list(pd.date_range(first-timedelta(days=8),first-timedelta(days=2)).date)
        allowable = list(pd.date_range(first-timedelta(days=60),first-timedelta(days=10)).date)
        complete = [d for d in allowable if data.complete_day(d)]
        if len(complete)<28: raise ValueError('insufficient warmup support within 60-day cap')
        start = complete[-28]
        dates = list(pd.date_range(start,f.evaluation.end).date)
        saved = pd.read_parquet(root/f'reports/cp15/folds/{f.name}-issued.parquet',
                                columns=['delivery_date','policy'],filters=[('policy','==','A1')])
        cached_dates = set(saved.delivery_date)
        missing = [d for d in dates if d not in cached_dates]
        # Missing component dates all precede D0, so inspect only permitted training data.
        for d in missing:
            rows = data.rows(d)
            train = (data.dates>=np.datetime64(history_start(d,'A1'))) & (data.dates<np.datetime64(d)) & data.eligible
            for h in np.unique(data.hours[rows]):
                tr = train & (data.hours==h)
                inner = tr & (data.dates<np.datetime64(d-timedelta(days=28)))
                val = tr & ~inner
                if tr.sum()<365 or inner.sum()<20 or val.sum()<14:
                    support_failures.append({'fold':f.name,'day':str(d),'hour':int(h),
                        'train':int(tr.sum()),'inner':int(inner.sum()),'validation':int(val.sum())})
        required = []
        for d in admission:
            available = [x for x in pd.date_range(start,d-timedelta(days=2)).date if data.complete_day(x)][-28:]
            if len(available)!=28: support_failures.append({'fold':f.name,'day':str(d),'buffer_days':len(available)})
            required.append({'day':str(d),'eligible_hours':len(data.rows(d)),
                'canonical_hours':len(day_hours(d)), 'complete_input_day':data.complete_day(d),
                'buffer_dates':list(map(str,available))})
        total_dates += len(dates); all_missing += len(missing)
        folds.append({'fold':f.name,'evaluation_start':str(first),'evaluation_end':str(f.evaluation.end),
            'warmup_start':str(start),'admission':required,'origins':[{'day':str(d),
            'origin_utc':str(origin_utc(d).tz_convert('UTC')),'cache_present':d in cached_dates,
            'history_start':str(history_start(d,'A1'))} for d in dates],
            'uncached_dates':list(map(str,missing)), 'date_count':len(dates),
            'original_target_keys':[str(x) for x in keys.loc[keys.fold.eq(f.name),'timestamp_utc']]})
        print(f.name,'start',start,'dates',len(dates),'cache misses',len(missing),flush=True)
    estimates = {'date_fold_keys':total_dates,'new_main_component_attempts':2*all_missing,
                 'nominal_main_primitive_fits':2*all_missing*120,
                 'one_full_residual_pass_policy_days':2*total_dates,
                 'independent_component_reproduction_reserved_attempts':20,
                 'control_component_attempts_reserved':8,
                 'solver_continuations':'each actual Lasso.fit consumes primitive and inner/final allowance before call; abort at cap'}
    if total_dates>750 or 2*all_missing>1500: raise ValueError('preflight allowance insufficient')
    manifest = {'input_sha256':hashes,'folds':folds,'estimates':estimates,'support_failures':support_failures,
                'cache_status':'identity-bound saved CP-15 component artifacts; representative reproduction still required before reuse',
                'partition_filter':'2019-01-01 <= delivery_date < fold D0 for E1; never beyond 2026-04-07 thereafter',
                'dependencies':{n:importlib.metadata.version(n) for n in ['numpy','pandas','pyarrow','scikit-learn','lightgbm','psutil','pytest']},
                'python':sys.version, 'resource_preflight':{'disk_free_bytes':shutil.disk_usage(root).free,
                'memory_available_bytes':psutil.virtual_memory().available,'physical_memory_bytes':psutil.virtual_memory().total},
                'caps':CAPS}
    atomic(output/'input-manifest.json',manifest)
    if support_failures: raise ValueError('training/admission support deficiency; no recipe substitution')
    return manifest
