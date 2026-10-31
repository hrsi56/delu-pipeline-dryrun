"""Admissible data and per-origin representations for the preregistered CP-15 run."""
from __future__ import annotations
import hashlib
import json
from dataclasses import dataclass
from datetime import date, timedelta, timezone
from pathlib import Path
import numpy as np
import pandas as pd
from delu_forecast.features import build_base_features
from delu_forecast.baselines import similar_day_naive
from delu_forecast.folds import load_partition_spec
from delu_forecast.ingest import BERLIN

RAW_COLUMNS = ['timestamp_utc', 'delivery_date', 'price_eur_mwh', 'load_forecast_mw']
POLICIES = ('B0','B1','B2','B3','A1','A2','A3','A4','A5')
FIT_POLICIES = ('B2','B3','A1','A2','A4')
NORMALIZED = frozenset(('A1','A2','A3','A4','A5'))
LEVELS = np.array([.025,.10,.25,.50,.75,.90,.975])
LABELS = ['p025','p10','p25','p50','p75','p90','p975']


def sha(path: Path) -> str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def array_hash(a) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def protocol(root: Path, verify=True) -> dict:
    p=json.loads((root/'reports/cp15/protocol.json').read_text())
    if p['revision']!='v21-r1-comparison-1':raise ValueError('wrong comparison protocol')
    if verify:
        for name,expected in p['input_sha256'].items():
            if sha(root/name)!=expected:raise ValueError(f'input hash mismatch: {name}')
    return p


def day_hours(d: date) -> pd.DatetimeIndex:
    return pd.date_range(pd.Timestamp(d,tz=BERLIN),pd.Timestamp(d+timedelta(days=1),tz=BERLIN),freq='h',inclusive='left').tz_convert('UTC')


def origin_utc(d: date) -> pd.Timestamp:
    return pd.Timestamp(d-timedelta(days=1),tz=timezone(timedelta(hours=1)))+pd.Timedelta(hours=12)


def history_start(d: date, policy: str) -> date:
    if policy not in FIT_POLICIES:raise ValueError('not a fitted policy')
    lower=d-timedelta(days=84 if policy=='A4' else 728)
    if policy!='A4':lower=max(date(2019,1,1),lower)
    if lower<date(2019,1,1):raise ValueError('pre-2019 history forbidden')
    return lower


@dataclass
class Inputs:
    frame: pd.DataFrame
    spec: object
    index: pd.DatetimeIndex
    dates: np.ndarray
    hours: np.ndarray
    y: np.ndarray
    eligible: np.ndarray
    feature_valid: np.ndarray
    level: np.ndarray
    scale: np.ndarray
    lgbm_raw: np.ndarray
    lgbm_normalized: np.ndarray
    lear_raw: np.ndarray
    lear_normalized: np.ndarray
    naive: np.ndarray
    p: dict
    reference: pd.DataFrame | None = None

    def rows(self,d: date,eligible=True):
        return np.flatnonzero((self.dates==np.datetime64(d)) & (self.feature_valid if eligible else True))

    def complete_day(self,d: date) -> bool:
        ix=self.rows(d,eligible=False)
        return self.index[ix].equals(day_hours(d)) and bool(self.eligible[ix].all())

    def warmup_start(self,first: date) -> date:
        dates=pd.date_range(date(2019,1,1),first-timedelta(days=2),freq='D').date
        complete=[d for d in dates if self.complete_day(d)]
        if len(complete)<28:raise ValueError('insufficient genuine warmup days')
        return complete[-28]


def prepare(frame: pd.DataFrame, p: dict, spec=None) -> Inputs:
    if list(frame.columns)!=RAW_COLUMNS:raise ValueError('strict raw schema: A69/actual/extra/reordered columns refused')
    frame=frame.copy()
    frame['timestamp_utc']=pd.to_datetime(frame.timestamp_utc).dt.tz_convert('UTC')
    index=pd.DatetimeIndex(frame.timestamp_utc).tz_convert('UTC')
    if index.has_duplicates or not index.is_monotonic_increasing or not index.equals(index.floor('h')):raise ValueError('invalid canonical timestamps')
    days=pd.Index(frame.delivery_date)
    if any(days<date(2019,1,1)):raise ValueError('pre-2019 input refused')
    if not np.array_equal(index.tz_convert(BERLIN).date,days.to_numpy()):raise ValueError('delivery calendar mismatch')
    if spec is not None and any(days>spec.eda_cutoff):raise ValueError('reserved outcome input refused')
    base=build_base_features(frame)
    valid=base.notna().all(axis=1).to_numpy()
    level=base.price_roll_mean_168h.to_numpy(float)
    scale=np.maximum(base.price_roll_std_168h.to_numpy(float),1.)
    x=base[p['lgbm']['features']].copy()
    z=x.copy()
    for c in p['normalization']['lgbm_center_and_scale']:z[c]=(x[c].to_numpy()-level)/scale
    for c in p['normalization']['lgbm_scale_only']:z[c]=x[c].to_numpy()/scale
    # Build calendar vectors without fixed 24-row shifts or target-price inputs.
    local=index.tz_convert(BERLIN)
    table=pd.DataFrame({'day':days,'hour':local.hour,'price':frame.price_eur_mwh.to_numpy(),'load':frame.load_forecast_mw.to_numpy()})
    calendar=pd.Index(pd.date_range(min(days),max(days),freq='D').date)
    vectors={c:table.pivot_table(index='day',columns='hour',values=c,aggfunc='mean',dropna=False).reindex(index=calendar,columns=range(24)) for c in ('price','load')}
    parts=[]
    for c,lags in [('price',[1,2,3,7]),('load',[0,1,7])]:
        for lag in lags:
            a=vectors[c].shift(lag).reindex(days).to_numpy(float)
            parts.append(a)
    parts.append(np.eye(7)[local.dayofweek])
    lear=np.column_stack(parts)
    ln=lear.copy();ln[:,:96]=(ln[:,:96]-level[:,None])/scale[:,None]
    y=frame.price_eur_mwh.to_numpy(float)
    return Inputs(frame,spec,index,np.array(days,dtype='datetime64[D]'),local.hour.to_numpy(),y,
                  valid & np.isfinite(y),valid,level,scale,x.to_numpy(float),z.to_numpy(float),lear,ln,
                  similar_day_naive(frame).to_numpy(float),p)


def load_inputs(root: Path) -> Inputs:
    p=protocol(root)
    spec=load_partition_spec(root/'data/partitions.json')
    frame=pd.read_parquet(root/'data/snapshot.parquet',columns=RAW_COLUMNS,
                          filters=[('delivery_date','>=',date(2019,1,1)),('delivery_date','<=',spec.eda_cutoff)])
    data=prepare(frame,p,spec)
    original=pd.read_parquet(root/'reports/cp2/development_predictions.parquet')
    original=original.loc[original.arm.eq('base')]
    refs=[]
    for fold in spec.development_folds:
        mask=(data.dates>=np.datetime64(fold.evaluation.start)) & (data.dates<=np.datetime64(fold.evaluation.end)) & data.eligible
        ref=original.loc[original.fold.eq(fold.name)].copy().reset_index(drop=True)
        if len(ref)!=mask.sum() or not np.array_equal(ref.y_true.to_numpy(),data.y[mask]) or not np.array_equal(np.array(ref.delivery_date,dtype='datetime64[D]'),data.dates[mask]):raise ValueError('original eligible row lineage mismatch')
        ref['timestamp_utc']=data.index[mask];ref['row_index']=np.flatnonzero(mask);refs.append(ref)
    data.reference=pd.concat(refs,ignore_index=True)
    return data
