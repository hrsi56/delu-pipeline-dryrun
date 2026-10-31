"""Prospectively frozen, descriptive CP-26 paired calendar-block inference.

The exact run, HAC and degeneracy rules come from frozen CP-28 protocol §9;
this module executes no CP-28 arm or adoption endpoint. Results retain block
starts, replicate estimates and studentized pivots for durable serialization.
Large arrays are processed in bounded chunks, never discarded selectively.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import math
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from cp25 import guard
from .scoring import MASKS, RESEARCH_STREAMS, paired_losses

SEED = 20261012
REPLICATES = 10_000
HAC_LAG = 14
VARIANTS = (('primary14', 14, True), ('sensitivity7', 7, True),
            ('sensitivity28', 28, True), ('unstratified14', 14, False))
CHUNK_REPLICATES = 256


def prospective_specification():
    """Outcome-free method metadata for the Lead's committed execution freeze."""
    return {'schema': 'cp26-inference-semantics-v1', 'seed': SEED, 'replicates': REPLICATES, 'HAC_lag': HAC_LAG,
            'HAC_centering': 'global endpoint mean; specific frozen protocol section9',
            'HAC_divisor': 'n squared', 'HAC_pairs': 'within actual contiguous run, exact calendar lag; resample pseudo-day lag',
            'variants': [{'name': name, 'block_days': length, 'stratified': stratified} for name, length, stratified in VARIANTS],
            'strata': '2020-12-29..31 joined to2021; remaining calendar years separate; real gaps always split',
            'block_sampling': 'uniform full noncircular starts per run, whole short run, fixed original run count, truncate final right end',
            'RNG_within_endpoint': 'replicate then original run then appended block',
            'RNG_endpoint_order': 'comparison registry; masks; overall/year2020..26/DJF,MAM,JJA,SON; MAE,WIS; own raw streams same cells; secondary stream,mask,season/year,MAE/WIS',
            'contrast_test': 'centered bootstrap-t lower tail including equality, +1 rule',
            'CI95': 'point minus linear .975/.025 pivot quantiles times observed SE',
            'ratio_influence': '(a-R*b)/mean(b), recomputed in each paired replicate',
            'secondary_influence': 'n/(J*n_g)*(a-R_g*b)/mean_g(b), J4 pooledseasons or J5 completeyears2021..25; labels travel with rows; all groups required',
            'degeneracy': 'no floor/drop/redraw; undefined endpoint p1, undefined CI; preserve all draw/statistic records',
            'DM': 'two-sided asymptotic normal, calendar Bartlett14',
            'GW': 'five instruments constant,calendar-lag1 differential,weekend,negative-risk,high-volatility; chi-square5; unknown high0 retained/disclosed; calendar-lag1 missing excluded',
            'MCS': '90percent HLN T_R, primary14 same paired draws, observed Bartlett14 pair SE; centered maxabs replicate statistic; eliminate max_i max_j mean(d_ij)/SE_ij lexical tie; stop p>=.10; undefined pair retains remaining set',
            'MCS_rng': 'restart default_rng20261012 each mask/metric cell; lexical stream columns; retain elimination history and replicates',
            'claim': 'descriptive CP26 control audit; no CP28 execution or adoption'}


def endpoint_identity(result):
    fields = ('kind', 'variant', 'left', 'right', 'stream', 'mask', 'group', 'value', 'metric', 'identity')
    identity = {name: result[name] for name in fields if name in result}
    return hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def write_result_artifact(root, relative_stem, result):
    """Persist every numeric array with hash-bound JSON; refuse historical overwrite.

    Output is confined to a new path under reports/rebaseline. The caller owns
    monitored resource accounting and the final inference/diagnostic manifest.
    """
    root, relative = Path(root), Path(relative_stem)
    if relative.is_absolute() or relative.parts[:2] != ('reports', 'rebaseline') or '..' in relative.parts:
        raise ValueError('numeric artifact must use a project-local reports/rebaseline path')
    stem = root / relative
    metadata_path, arrays_path = stem.with_suffix('.json'), stem.with_suffix('.npz')
    if metadata_path.exists() or arrays_path.exists():
        raise ValueError('refuse overwriting an existing numeric artifact')
    arrays = {}

    def pack(value, key='root'):
        if isinstance(value, np.ndarray):
            if value.dtype.hasobject:
                raise ValueError('numeric artifacts never pickle object arrays')
            name = f'array_{len(arrays):04d}'
            arrays[name] = value
            return {'array': name, 'shape': list(value.shape), 'dtype': str(value.dtype), 'field': key}
        if isinstance(value, dict):
            return {str(name): pack(item, f'{key}.{name}') for name, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [pack(item, f'{key}.{i}') for i, item in enumerate(value)]
        if isinstance(value, np.generic):
            value = value.item()
        if isinstance(value, float) and not np.isfinite(value):
            return None
        if isinstance(value, (date, pd.Timestamp)):
            return str(value)
        return value

    payload = pack(result)
    stem.parent.mkdir(parents=True, exist_ok=True)
    temporary_arrays = arrays_path.with_name(arrays_path.name+'.tmp')
    with temporary_arrays.open('xb') as handle:
        np.savez_compressed(handle, **arrays)
    arrays_sha = hashlib.sha256(temporary_arrays.read_bytes()).hexdigest()
    metadata = {'schema': 'cp26-numeric-artifact-v1', 'endpoint_id': endpoint_identity(result),
                'arrays_file': arrays_path.name, 'arrays_sha256': arrays_sha, 'result': payload}
    temporary_metadata = metadata_path.with_name(metadata_path.name+'.tmp')
    temporary_metadata.write_text(json.dumps(metadata, sort_keys=True, allow_nan=False)+'\n', encoding='utf-8')
    temporary_arrays.replace(arrays_path)
    temporary_metadata.replace(metadata_path)
    metadata_sha = hashlib.sha256(metadata_path.read_bytes()).hexdigest()
    compact = {key: value for key, value in payload.items()
               if not isinstance(value, dict) or 'array' not in value}
    # MCS history statistics remain in JSON array references; the manifest uses
    # compact elimination details while the complete record stays hash-bound.
    return {'endpoint_id': metadata['endpoint_id'], 'metadata_path': str(metadata_path.relative_to(root)),
            'metadata_sha256': metadata_sha, 'arrays_path': str(arrays_path.relative_to(root)),
            'arrays_sha256': arrays_sha, **compact}


def calendar_runs(dates, *, stratified=True):
    """Contiguous actual-calendar runs; eligible 2020 joins 2021's stratum."""
    days = [pd.Timestamp(day).date() for day in dates]
    if days != sorted(set(days)):
        raise ValueError('paired dates must be unique and chronological')
    for day in days:
        guard.admit_delivery_day(day)
    runs = []
    start, prior_stratum = 0, None
    for i, day in enumerate(days):
        stratum = max(2021, day.year) if stratified else 0
        if i and ((day - days[i-1]).days != 1 or stratum != prior_stratum):
            runs.append((start, i, prior_stratum))
            start = i
        prior_stratum = stratum
    if days:
        runs.append((start, len(days), prior_stratum))
    return tuple(runs)


@dataclass(frozen=True)
class DrawPlan:
    """Per-replicate starts in original-run-relative coordinates."""
    runs: tuple
    block_days: int
    starts: np.ndarray
    start_columns: tuple
    dates: tuple
    stratified: bool

    @property
    def n(self):
        return len(self.dates)

    def indices(self, first=0, last=None):
        last = len(self.starts) if last is None else last
        result = np.empty((last-first, self.n), dtype=np.int32)
        for (begin, end, _), (col_start, col_end) in zip(self.runs, self.start_columns):
            length = min(self.block_days, end-begin)
            for j, col in enumerate(range(col_start, col_end)):
                count = min(length, end-begin-j*length)
                destination = begin+j*length
                result[:, destination:destination+count] = (begin+self.starts[first:last, col, None]
                                                           + np.arange(count)[None, :])
        return result

    def metadata(self):
        return {'block_days': self.block_days, 'stratified': self.stratified,
                'dates': list(self.dates), 'runs': [list(row) for row in self.runs],
                'start_columns': [list(row) for row in self.start_columns],
                'replicates': len(self.starts), 'start_coordinate': 'zero-based within original run',
                'rng_order': 'replicate then original run then appended block',
                'short_run_rule': 'whole run; truncate only final block right end'}


def draw_plan(dates, *, block_days=14, stratified=True, rng=None, replicates=REPLICATES):
    if block_days not in (7, 14, 28) or replicates < 1:
        raise ValueError('invalid registered block length or replicate count')
    runs = calendar_runs(dates, stratified=stratified)
    columns, highs, col = [], [], 0
    for begin, end, _ in runs:
        n = end-begin
        length = min(block_days, n)
        blocks = math.ceil(n/length)
        columns.append((col, col+blocks))
        highs.extend([n-length+1]*blocks)
        col += blocks
    rng = np.random.default_rng(SEED) if rng is None else rng
    starts = (rng.integers(0, np.array(highs, dtype=np.int64), size=(replicates, col), dtype=np.int32)
              if col else np.empty((replicates, 0), np.int32))
    return DrawPlan(runs, block_days, starts, tuple(columns),
                    tuple(str(pd.Timestamp(day).date()) for day in dates), stratified)


def hac_mean_covariance(values, runs, *, centering='global'):
    """Bartlett14 covariance of mean, divisor n², no shortened lag or floor.

    ``values`` has shape (n,p) or (replicate,n,p). Global centering is the
    specific frozen protocol §9 convention; stratum centering is reserved for
    the memo's vector moment diagnostic. Only within-run calendar pairs count.
    """
    x = np.asarray(values, dtype=float)
    if x.ndim == 1:
        x = x[:, None]
    batched = x.ndim == 3
    if x.ndim not in (2, 3) or centering not in ('global', 'stratum'):
        raise ValueError('invalid HAC input or centering convention')
    if not batched:
        x = x[None, :, :]
    n = x.shape[1]
    if n < 2:
        return np.full((x.shape[0], x.shape[2], x.shape[2]), np.nan) if batched else np.full((x.shape[2], x.shape[2]), np.nan)
    covered = [i for begin, end, _ in runs for i in range(begin, end)]
    if covered != list(range(n)):
        raise ValueError('HAC run boundaries must partition the retained sample')
    z = x.copy()
    if centering == 'global':
        z -= x.mean(axis=1, keepdims=True)
    else:
        for stratum in sorted({r[2] for r in runs}):
            positions = np.concatenate([np.arange(begin, end) for begin, end, s in runs if s == stratum])
            z[:, positions] -= x[:, positions].mean(axis=1, keepdims=True)
    covariance = np.einsum('bnp,bnq->bpq', z, z)
    for lag in range(1, HAC_LAG+1):
        for begin, end, _ in runs:
            if end-begin > lag:
                term = np.einsum('bnp,bnq->bpq', z[:, begin+lag:end], z[:, begin:end-lag])
                covariance += (1-lag/(HAC_LAG+1)) * (term+term.transpose(0, 2, 1))
    covariance /= n*n
    return covariance if batched else covariance[0]


def hac_variance_mean(values, runs):
    x = np.asarray(values, float)
    return hac_mean_covariance(x[:, :, None] if x.ndim == 2 else x, runs)[..., 0, 0]


def _valid_variance(value):
    return np.isfinite(value) & (value > 0)


def _interval(point, standard_error, pivots, valid):
    if not valid:
        return (np.nan, np.nan)
    lower, upper = np.quantile(pivots, [.025, .975], method='linear')
    return (float(point-upper*standard_error), float(point-lower*standard_error))


def paired_bootstrap(pair, *, block_days=14, stratified=True, rng=None,
                     replicates=REPLICATES, production=True, include_raw=True):
    """Shared resamples for paired differences and ratios, with bootstrap-t CIs.

    All original pairs are fixed before values are read. Uncomputable endpoints
    consume no additional draws; a valid raw-loss endpoint can still require
    the shared samples when difference/ratio is degenerate. A bad replicate
    invalidates its whole endpoint and remains stored, with no variance floor.
    """
    if production and replicates != REPLICATES:
        raise ValueError('production inference requires exactly 10,000 replicates')
    if not {'delivery_date', 'a', 'b'}.issubset(pair):
        raise ValueError('paired input requires delivery_date, a, b')
    a, b = pair.a.to_numpy(float), pair.b.to_numpy(float)
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('paired loss mask must be fixed before bootstrap')
    runs = calendar_runs(pair.delivery_date, stratified=stratified)
    n = len(a)
    mean_a, mean_b = (float(a.mean()), float(b.mean())) if n else (np.nan, np.nan)
    difference = mean_a-mean_b
    ratio_ok = np.isfinite(mean_b) and mean_b != 0
    ratio = mean_a/mean_b if ratio_ok else np.nan
    difference_variance = float(hac_variance_mean(a-b, runs)) if n >= 2 else np.nan
    influence = (a-ratio*b)/mean_b if ratio_ok else np.full(n, np.nan)
    ratio_variance = float(hac_variance_mean(influence, runs)) if n >= 2 and ratio_ok else np.nan
    diff_valid, ratio_valid = bool(_valid_variance(difference_variance)), bool(_valid_variance(ratio_variance))
    raw_a_variance = float(hac_variance_mean(a, runs)) if n >= 2 else np.nan
    raw_b_variance = float(hac_variance_mean(b, runs)) if n >= 2 else np.nan
    raw_a_valid = include_raw and bool(_valid_variance(raw_a_variance))
    raw_b_valid = include_raw and bool(_valid_variance(raw_b_variance))
    result = {'n_dates': n, 'difference': difference, 'ratio': ratio,
        'difference_variance': difference_variance, 'ratio_variance': ratio_variance,
        'difference_status': 'computable' if diff_valid else 'nonpositive_or_nonfinite_observed_variance',
        'ratio_status': 'computable' if ratio_valid else ('zero_or_nonfinite_denominator' if not ratio_ok else 'nonpositive_or_nonfinite_observed_variance'),
        'difference_p_value': 1., 'difference_ci95': (np.nan, np.nan), 'ratio_ci95': (np.nan, np.nan),
        'replicate_differences': np.empty(0), 'replicate_ratios': np.empty(0),
        'replicate_left_losses': np.empty(0), 'replicate_right_losses': np.empty(0),
        'raw_left_loss': mean_a, 'raw_right_loss': mean_b,
        'raw_left_variance': raw_a_variance, 'raw_right_variance': raw_b_variance,
        'raw_left_ci95': (np.nan, np.nan), 'raw_right_ci95': (np.nan, np.nan),
        'raw_left_pivots': np.empty(0), 'raw_right_pivots': np.empty(0),
        'raw_left_status': 'computable' if raw_a_valid else 'nonpositive_or_nonfinite_observed_variance',
        'raw_right_status': 'computable' if raw_b_valid else 'nonpositive_or_nonfinite_observed_variance',
        'difference_pivots': np.empty(0), 'ratio_pivots': np.empty(0), 'draw_starts': np.empty((0, 0), np.int32),
        'draw_metadata': None, 'replicates_required': replicates,
        'interpretation': 'descriptive; no adoption, replacement or demotion'}
    if not any((diff_valid, ratio_valid, raw_a_valid, raw_b_valid)):
        return result
    plan = draw_plan(pair.delivery_date, block_days=block_days, stratified=stratified, rng=rng, replicates=replicates)
    delta_draws, ratio_draws, delta_pivots, ratio_pivots, left_draws, right_draws, left_pivots, right_pivots = [np.full(replicates, np.nan) for _ in range(8)]
    for first in range(0, replicates, CHUNK_REPLICATES):
        last = min(first+CHUNK_REPLICATES, replicates)
        indices = plan.indices(first, last)
        ar, br = a[indices], b[indices]
        am, bm = ar.mean(axis=1), br.mean(axis=1)
        left_draws[first:last], right_draws[first:last] = am, bm
        for values, mean_draw, original_mean, pivots, valid_observed in (
                (ar, am, mean_a, left_pivots, raw_a_valid), (br, bm, mean_b, right_pivots, raw_b_valid)):
            if valid_observed:
                variance = hac_variance_mean(values, runs)
                valid = _valid_variance(variance)
                pivots[first:last][valid] = (mean_draw[valid]-original_mean)/np.sqrt(variance[valid])
        dr = ar-br
        delta_draws[first:last] = dr.mean(axis=1)
        if diff_valid:
            variance = hac_variance_mean(dr, runs)
            valid = _valid_variance(variance)
            delta_pivots[first:last][valid] = (delta_draws[first:last][valid]-difference)/np.sqrt(variance[valid])
        if ratio_valid:
            valid_denominator = np.isfinite(bm) & (bm != 0)
            rr = np.divide(am, bm, out=np.full(len(am), np.nan), where=valid_denominator)
            ratio_draws[first:last] = rr
            ir = np.divide(ar-rr[:, None]*br, bm[:, None], out=np.full(ar.shape, np.nan), where=valid_denominator[:, None])
            variance = hac_variance_mean(ir, runs)
            valid = valid_denominator & _valid_variance(variance)
            ratio_pivots[first:last][valid] = (rr[valid]-ratio)/np.sqrt(variance[valid])
    if diff_valid and not np.isfinite(delta_pivots).all():
        diff_valid = False
        result['difference_status'] = 'nonfinite_or_nonpositive_replicate_variance_or_statistic'
    if ratio_valid and not np.isfinite(ratio_pivots).all():
        ratio_valid = False
        result['ratio_status'] = 'undefined_replicate_denominator_variance_or_statistic'
    if raw_a_valid and not np.isfinite(left_pivots).all():
        raw_a_valid = False
        result['raw_left_status'] = 'nonfinite_or_nonpositive_replicate_variance_or_statistic'
    if raw_b_valid and not np.isfinite(right_pivots).all():
        raw_b_valid = False
        result['raw_right_status'] = 'nonfinite_or_nonpositive_replicate_variance_or_statistic'
    result.update(replicate_differences=delta_draws, replicate_ratios=ratio_draws,
        replicate_left_losses=left_draws, replicate_right_losses=right_draws,
        raw_left_pivots=left_pivots, raw_right_pivots=right_pivots,
        raw_left_ci95=_interval(mean_a, np.sqrt(raw_a_variance) if raw_a_valid else np.nan, left_pivots, raw_a_valid),
        raw_right_ci95=_interval(mean_b, np.sqrt(raw_b_variance) if raw_b_valid else np.nan, right_pivots, raw_b_valid),
        difference_pivots=delta_pivots, ratio_pivots=ratio_pivots,
        draw_starts=plan.starts, draw_metadata=plan.metadata(),
        difference_ci95=_interval(difference, np.sqrt(difference_variance) if diff_valid else np.nan, delta_pivots, diff_valid),
        ratio_ci95=_interval(ratio, np.sqrt(ratio_variance) if ratio_valid else np.nan, ratio_pivots, ratio_valid))
    if diff_valid:
        observed = difference/np.sqrt(difference_variance)
        count = int((delta_pivots <= observed).sum())
        result.update(difference_statistic=observed, difference_p_value=(1+count)/(replicates+1),
                      difference_lower_tail_count=count)
    return result


def secondary_ratio_bootstrap(pair, *, group, block_days=14, stratified=True, rng=None,
                              replicates=REPLICATES, production=True):
    """Bootstrap-t for mean four-season or five-complete-year paired B0 ratios.

    The Lead froze this delta influence prospectively: for i in group g,
    psi_i=n/(J*n_g)*(a_i-R_g*b_i)/mean_g(b). Recompute every group count,
    denominator, ratio and influence on each shared calendar-block sample.
    Source-row group labels travel with blocks. No empty or undefined group is
    omitted or redrawn; any such replicate invalidates the whole interval.
    """
    if production and replicates != REPLICATES:
        raise ValueError('production secondary inference requires 10,000 replicates')
    if group not in ('season', 'year'):
        raise ValueError('secondary ratio groups are exactly season or complete year')
    dates = pd.to_datetime(pair.delivery_date)
    if group == 'year':
        pair = pair.loc[dates.dt.year.between(2021, 2025)].reset_index(drop=True)
        dates = pd.to_datetime(pair.delivery_date)
        levels = tuple(range(2021, 2026))
        labels = dates.dt.year.to_numpy()
    else:
        levels = ('DJF', 'MAM', 'JJA', 'SON')
        labels = dates.dt.month.map({12:'DJF', 1:'DJF', 2:'DJF', 3:'MAM', 4:'MAM', 5:'MAM',
                                    6:'JJA', 7:'JJA', 8:'JJA', 9:'SON', 10:'SON', 11:'SON'}).to_numpy()
    group_ids = np.array([levels.index(value) for value in labels], dtype=np.int32)
    a, b = pair.a.to_numpy(float), pair.b.to_numpy(float)
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('secondary paired availability must be fixed before inference')
    n, j = len(a), len(levels)

    def calculate(ar, br, gr):
        count = np.stack([(gr == k).sum(axis=1) for k in range(j)], axis=1)
        am = np.stack([np.sum(np.where(gr == k, ar, 0), axis=1) for k in range(j)], axis=1)
        bm = np.stack([np.sum(np.where(gr == k, br, 0), axis=1) for k in range(j)], axis=1)
        denominator = np.divide(bm, count, out=np.full(bm.shape, np.nan), where=count > 0)
        ratio = np.divide(am, bm, out=np.full(am.shape, np.nan), where=(count > 0) & (bm != 0))
        row = np.arange(len(ar))[:, None]
        influence = np.divide(n*(ar-ratio[row, gr]*br), j*count[row, gr]*denominator[row, gr],
                              out=np.full(ar.shape, np.nan), where=(count[row, gr] > 0) & (denominator[row, gr] != 0))
        return ratio.mean(axis=1), influence, count, denominator, ratio

    result = {'group': group, 'groups': list(levels), 'n_dates': n, 'ratio': np.nan,
              'ci95': (np.nan, np.nan), 'status': 'missing_or_undefined_registered_group',
              'draw_starts': np.empty((0, 0), np.int32), 'replicate_ratios': np.empty(0),
              'pivots': np.empty(0), 'replicate_group_counts': np.empty((0, j), np.int32),
              'replicate_group_denominators': np.empty((0, j)), 'replicate_group_ratios': np.empty((0, j)),
              'draw_metadata': None, 'replicates_required': replicates, 'evidence_class': 'descriptive'}
    if not n:
        return result
    estimate, influence, counts, denominator, ratios = calculate(a[None, :], b[None, :], group_ids[None, :])
    result.update(ratio=float(estimate[0]), group_counts=counts[0], group_denominators=denominator[0], group_ratios=ratios[0])
    if not np.isfinite(estimate[0]) or not np.isfinite(influence).all():
        return result
    runs = calendar_runs(pair.delivery_date, stratified=stratified)
    variance = float(hac_variance_mean(influence[0], runs)) if n >= 2 else np.nan
    result['variance'] = variance
    if not _valid_variance(variance):
        return {**result, 'status': 'nonpositive_or_nonfinite_observed_variance'}
    plan = draw_plan(pair.delivery_date, block_days=block_days, stratified=stratified, rng=rng, replicates=replicates)
    estimates, pivots = np.full(replicates, np.nan), np.full(replicates, np.nan)
    all_counts = np.empty((replicates, j), np.int32)
    all_denominators, all_ratios = np.empty((replicates, j)), np.empty((replicates, j))
    for first in range(0, replicates, CHUNK_REPLICATES):
        last = min(first+CHUNK_REPLICATES, replicates)
        index = plan.indices(first, last)
        estimate_b, influence_b, count_b, denominator_b, ratio_b = calculate(a[index], b[index], group_ids[index])
        estimates[first:last], all_counts[first:last] = estimate_b, count_b
        all_denominators[first:last], all_ratios[first:last] = denominator_b, ratio_b
        variance_b = hac_variance_mean(influence_b, runs)
        valid = np.isfinite(estimate_b) & _valid_variance(variance_b)
        pivots[first:last][valid] = (estimate_b[valid]-estimate[0])/np.sqrt(variance_b[valid])
    valid = np.isfinite(pivots).all()
    return {**result, 'status': 'computable' if valid else 'undefined_replicate_group_denominator_variance_or_statistic',
            'ci95': _interval(float(estimate[0]), np.sqrt(variance), pivots, valid),
            'draw_starts': plan.starts, 'draw_metadata': plan.metadata(), 'replicate_ratios': estimates,
            'pivots': pivots, 'replicate_group_counts': all_counts,
            'replicate_group_denominators': all_denominators, 'replicate_group_ratios': all_ratios}


def iter_mcs(daily, *, streams=None, replicates=REPLICATES, production=True):
    """One four-mask/metric MCS cell with explicit common-population counts."""
    streams = tuple(sorted(RESEARCH_STREAMS if production else daily.stream.unique())) if streams is None else tuple(sorted(streams))
    if production and streams != tuple(sorted(RESEARCH_STREAMS)):
        raise ValueError('production MCS universe is exactly 13 CP-26 research streams')
    for mask in MASKS:
        for metric in ('MAE', 'WIS'):
            selected = daily[daily[mask] & daily.stream.isin(streams)]
            losses = selected.pivot(index='delivery_date', columns='stream', values=metric).reindex(columns=streams)
            valid = np.isfinite(losses.to_numpy(float)).all(axis=1)
            common = losses.loc[valid]
            yield {**model_confidence_set(common, mask=mask, metric=metric,
                                          replicates=replicates, production=production),
                   'calendar_dates': len(losses), 'incomplete_common_dates': int((~valid).sum()),
                   'stream_missing_dates': {stream: int(losses[stream].isna().sum()) for stream in streams}}


def comparison_registry(streams):
    """Frozen lexical stream order; B0 comparisons, then generation transitions."""
    streams = tuple(sorted(set(streams)))
    if 'nominal/B0' not in streams:
        raise ValueError('nominal B0 is required')
    rows = [(stream, 'nominal/B0') for stream in streams if stream != 'nominal/B0']
    for route in ('nominal', 'noload', 'gfs6'):
        for left, right in (('v3', 'v2'), ('v4', 'v3'), ('v5', 'v4')):
            contrast = (f'{route}/{left}', f'{route}/{right}')
            if set(contrast).issubset(streams) and contrast not in rows:
                rows.append(contrast)
    for route in ('noload', 'gfs6'):
        contrast = (f'{route}/v5', 'nominal/v5')
        if set(contrast).issubset(streams) and contrast not in rows:
            rows.append(contrast)
    for reference in ('nominal/D2', 'nominal/released-v1'):
        for policy in ('v1-R', 'v2', 'v3', 'v4', 'v5'):
            contrast = (reference, f'nominal/{policy}')
            if set(contrast).issubset(streams) and contrast not in rows:
                rows.append(contrast)
    return tuple(rows)


def iter_inference(daily, *, comparisons=None, replicates=REPLICATES, production=True):
    """Yield a serializable endpoint at a time for hash-bound .npz storage.

    RNG restarts for each variant, then visits registered comparisons, four
    masks, overall/years/seasons, and MAE/WIS; then lexical own-loss streams in
    the same cell order; then lexical stream/B0 secondary season/year ratios.
    Arrays can be persisted and released before advancing the iterator.
    """
    comparisons = comparison_registry(daily.stream.unique()) if comparisons is None else tuple(comparisons)
    cells = (None,) + tuple(('year', year) for year in range(2020, 2027)) + tuple(('season', s) for s in ('DJF', 'MAM', 'JJA', 'SON'))
    for variant, block_days, stratified in VARIANTS:
        rng = np.random.default_rng(SEED)
        for left, right in comparisons:
            for mask in MASKS:
                for cell in cells:
                    for metric in ('MAE', 'WIS'):
                        pair, counts = paired_losses(daily, left, right, metric=metric, mask=mask, group=cell)
                        result = paired_bootstrap(pair, block_days=block_days, stratified=stratified,
                                                  rng=rng, replicates=replicates, production=production)
                        yield {'kind': 'paired_comparison', 'variant': variant, 'left': left, 'right': right, 'mask': mask, 'metric': metric,
                               'group': cell[0] if cell else 'overall', 'value': cell[1] if cell else 'overall',
                               **counts, **result}
        for stream in sorted(daily.stream.unique()):
            for mask in MASKS:
                for cell in cells:
                    for metric in ('MAE', 'WIS'):
                        pair, counts = paired_losses(daily, stream, stream, metric=metric, mask=mask, group=cell)
                        result = paired_bootstrap(pair, block_days=block_days, stratified=stratified,
                                                  rng=rng, replicates=replicates, production=production)
                        yield {'kind': 'own_raw_loss', 'variant': variant, 'stream': stream, 'mask': mask, 'metric': metric,
                               'group': cell[0] if cell else 'overall', 'value': cell[1] if cell else 'overall',
                               **counts, **result}
        for stream in sorted(daily.stream.unique()):
            for mask in MASKS:
                for group in ('season', 'year'):
                    for metric in ('MAE', 'WIS'):
                        pair, counts = paired_losses(daily, stream, 'nominal/B0', metric=metric, mask=mask)
                        result = secondary_ratio_bootstrap(pair, group=group, block_days=block_days, stratified=stratified,
                                                           rng=rng, replicates=replicates, production=production)
                        yield {'kind': 'secondary_mean_ratio', 'variant': variant, 'stream': stream,
                               'mask': mask, 'metric': metric, **counts, **result}


def diebold_mariano(pair, *, stratified=True):
    """Descriptive, two-sided normal DM; exact calendar-run Bartlett14 SE."""
    d = pair.a.to_numpy(float)-pair.b.to_numpy(float)
    variance = float(hac_variance_mean(d, calendar_runs(pair.delivery_date, stratified=stratified))) if len(d) >= 2 else np.nan
    if not _valid_variance(variance):
        return {'status': 'unavailable_variance_or_support', 'p_value': 1., 'n_dates': len(d)}
    statistic = float(d.mean()/np.sqrt(variance))
    return {'status': 'computable', 'n_dates': len(d), 'statistic': statistic,
            'p_value': float(2*stats.norm.sf(abs(statistic))), 'HAC_lag': HAC_LAG,
            'evidence_class': 'descriptive', 'mean_differential': float(d.mean())}


def giacomini_white(pair, known_states):
    """Descriptive five-instrument Wald with true calendar-lag-1 differential.

    The frozen startup high-volatility flag is zero with unknown support
    disclosed. A missing actual lag-1 pair cannot supply that instrument.
    """
    p = pair.copy()
    p['delivery_date'] = pd.to_datetime(p.delivery_date).dt.date
    known_states = known_states.copy()
    known_states['delivery_date'] = pd.to_datetime(known_states.delivery_date).dt.date
    if p.delivery_date.duplicated().any() or known_states.delivery_date.duplicated().any():
        raise ValueError('GW dates and frozen states must be unique')
    p = p.set_index('delivery_date').sort_index()
    p['d'] = p.a-p.b
    states = known_states.set_index('delivery_date')
    vectors, dates, unknown_volatility = [], [], 0
    for day, row in p.iterrows():
        day = pd.Timestamp(day).date()
        lag = day-pd.Timedelta(days=1).to_pytimedelta()
        if lag not in p.index or day not in states.index:
            continue
        negative, volatility = states.loc[day, ['negative_price_risk', 'high_volatility']]
        if pd.isna(negative) or pd.isna(volatility):
            raise ValueError('frozen causal state flags must use their registered startup encoding')
        if 'volatility_unknown' in states and states.at[day, 'volatility_unknown']:
            unknown_volatility += 1
        instruments = np.array([1., p.at[lag, 'd'], float(day.weekday() >= 5), float(negative), float(volatility)])
        vectors.append(row.d*instruments)
        dates.append(day)
    g = np.asarray(vectors, float).reshape(-1, 5)
    covariance = hac_mean_covariance(g, calendar_runs(dates)) if len(g) >= 2 else np.full((5, 5), np.nan)
    result = {'n_dates': len(g), 'paired_dates': len(pair), 'excluded_instrument_dates': len(pair)-len(g),
              'instruments': ['constant', 'calendar_lag1_differential', 'weekend', 'negative_price_risk', 'high_volatility'],
              'HAC_lag': HAC_LAG, 'evidence_class': 'descriptive', 'p_value': 1.}
    result['volatility_unknown_dates'] = unknown_volatility
    try:
        if not np.isfinite(covariance).all() or np.linalg.eigvalsh(covariance).min() <= 0:
            raise np.linalg.LinAlgError('nonpositive covariance')
        statistic = float(g.mean(axis=0) @ np.linalg.solve(covariance, g.mean(axis=0)))
    except np.linalg.LinAlgError:
        return {**result, 'status': 'unavailable_covariance_or_support'}
    return {**result, 'status': 'computable', 'statistic': statistic, 'p_value': float(stats.chi2.sf(statistic, 5))}


def model_confidence_set(losses, *, mask, metric, replicates=REPLICATES, production=True):
    """90% HLN range-statistic MCS on a frozen complete common-date population.

    Losses are wide (delivery_date index, lexical stream columns). Each call
    restarts seed20261012. The same draws serve every elimination; zero or bad
    pair variance makes the remaining set unavailable rather than deleting an
    arbitrary policy. Every elimination and replicate statistic is retained.
    """
    if production and replicates != REPLICATES:
        raise ValueError('production MCS requires exactly 10,000 replicates')
    if metric not in ('MAE', 'WIS') or mask not in MASKS:
        raise ValueError('unregistered MCS cell')
    losses = losses.sort_index().sort_index(axis=1)
    x = losses.to_numpy(float)
    names = list(losses.columns)
    result = {'mask': mask, 'metric': metric, 'confidence': .90, 'streams': names,
              'n_common_dates': len(x), 'history': [], 'remaining': names, 'evidence_class': 'descriptive'}
    if len(x) < 2 or len(names) < 2 or not np.isfinite(x).all():
        return {**result, 'status': 'unavailable_support_or_common_loss'}
    plan = draw_plan(losses.index, replicates=replicates)
    means = np.empty((replicates, len(names)))
    for first in range(0, replicates, CHUNK_REPLICATES):
        last = min(first+CHUNK_REPLICATES, replicates)
        means[first:last] = x[plan.indices(first, last)].mean(axis=1)
    result.update(draw_starts=plan.starts, draw_metadata=plan.metadata(), replicate_means=means)
    active = list(range(len(names)))
    original = x.mean(axis=0)
    while len(active) > 1:
        statistics, bootstrap, worst = [], [], {i: -np.inf for i in active}
        bad_pair = None
        for ii, i in enumerate(active):
            for j in active[ii+1:]:
                variance = float(hac_variance_mean(x[:, i]-x[:, j], plan.runs))
                if not _valid_variance(variance):
                    bad_pair = (names[i], names[j])
                    break
                se = np.sqrt(variance)
                observed = (original[i]-original[j])/se
                statistics.append(abs(observed))
                bootstrap.append(np.abs((means[:, i]-means[:, j]-(original[i]-original[j]))/se))
                worst[i], worst[j] = max(worst[i], observed), max(worst[j], -observed)
            if bad_pair is not None:
                break
        if bad_pair is not None:
            result.update(status='unavailable_pair_variance', undefined_pair=bad_pair, remaining=[names[i] for i in active])
            return result
        observed = float(max(statistics))
        replicate_statistics = np.max(np.asarray(bootstrap), axis=0)
        p = (1+int((replicate_statistics >= observed).sum()))/(replicates+1)
        removed = min(active, key=lambda i: (-worst[i], names[i])) if p < .10 else None
        result['history'].append({'active': [names[i] for i in active], 'T_R': observed, 'p_value': p,
                                  'removed': names[removed] if removed is not None else None,
                                  'replicate_statistics': replicate_statistics})
        if removed is None:
            break
        active.remove(removed)
    result.update(status='computable', remaining=[names[i] for i in active])
    return result


def holm(p_values, *, alpha=.05):
    """Full-family Holm; lexical original identities break equal p-values."""
    if any(not np.isfinite(p) or p < 0 or p > 1 for p in p_values.values()):
        raise ValueError('Holm p-values must be finite in [0,1]')
    order = sorted(p_values, key=lambda name: (p_values[name], name))
    m, maximum, rejecting, rows = len(order), 0., True, {}
    for rank, name in enumerate(order, start=1):
        p = p_values[name]
        maximum = max(maximum, (m-rank+1)*p)
        threshold = alpha/(m-rank+1)
        rejecting = rejecting and p <= threshold
        rows[name] = {'rank': rank, 'p_value': p, 'adjusted_p_value': min(1., maximum),
                      'threshold': threshold, 'rejected': rejecting, 'family_size': m}
    return rows
