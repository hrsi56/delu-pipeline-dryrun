"""CP-26 production analysis orchestration and outcome-free resource fixtures.

Scientific methods live in scoring, inference and diagnostics. This module owns
guarded input assembly, complete inventories, durable artifacts and RNG-safe
restart. Its public jobs expose no method, replicate-count or population flags.
"""
from __future__ import annotations

from datetime import date
from itertools import product
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import shutil
import sys
import time
import zipfile

import numpy as np
import pandas as pd

from .budget import Budget, atomic

from .gates import OUTPUT, LOCAL_NAME
SCORE_SCHEMA = 'cp26-scores-manifest-v1'
INFERENCE_SCHEMA = 'cp26-inference-manifest-v1'
DIAGNOSTIC_SCHEMA = 'cp26-diagnostics-manifest-v1'
BENCHMARK_SCHEMA = 'cp26-fixture-benchmark-v1'


def _plain(value):
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, np.ndarray):
        return [_plain(v) for v in value.tolist()]
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(v) for v in value]
    if isinstance(value, pd.Timestamp):
        return {'__timestamp__': value.isoformat()}
    if isinstance(value, date):
        return {'__date__': value.isoformat()}
    if value is pd.NA or isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _restore(value):
    if isinstance(value, dict):
        if set(value) == {'__date__'}:
            return date.fromisoformat(value['__date__'])
        if set(value) == {'__timestamp__'}:
            return pd.Timestamp(value['__timestamp__'])
        return {k: _restore(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_restore(v) for v in value]
    return value


def _digest(value):
    raw = json.dumps(_plain(value), sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    return hashlib.sha256(raw).hexdigest()


def _sha(path):
    hasher = hashlib.sha256()
    with Path(path).open('rb') as handle:
        while block := handle.read(1024 * 1024):
            hasher.update(block)
    return hasher.hexdigest()


def _signed(value):
    value = _plain(value)
    return {**value, 'content_sha256': _digest(value)}


def _read(path, schema=None):
    value = json.loads(Path(path).read_text())
    unsigned = {k: v for k, v in value.items() if k != 'content_sha256'}
    if value.get('content_sha256') != _digest(unsigned) or schema and value.get('schema') != schema:
        raise ValueError(f'corrupt analysis artifact: {Path(path).name}')
    return value


def _immutable(path, value):
    path = Path(path)
    if path.exists():
        if _read(path) != value:
            raise ValueError(f'analysis artifact differs; preserve original {path.name}')
    else:
        atomic(path, value)
    return value


def _output(root):
    return Path(root).resolve() / OUTPUT


def _code(root):
    from .gates import code_identity
    return code_identity(Path(root))


def _gate(root, gate, stage='score'):
    from .execution import _gate as execution_gate
    execution_gate(gate, stage)


def registered_table_keys(name):
    """Fixed semantic identities; support never determines table membership."""
    keys = {'summaries': ('mask', 'stream', 'metric', 'group', 'value'),
            'local_hour_summaries': ('mask', 'stream', 'local_hour', 'metric'),
            'coverage': ('mask', 'stream', 'group', 'value', 'interval'),
            'quantile_hits': ('mask', 'stream', 'group', 'value', 'quantile'),
            'histogram': ('mask', 'stream', 'group', 'value', 'bin')}
    if name not in keys:
        raise ValueError('unregistered descriptive inventory table')
    return keys[name]


def registered_table_identities(name):
    """Outcome-free expected identity set for the 13 production streams."""
    from . import scoring as S
    keys = registered_table_keys(name)
    dimensions = S.registered_breakdown_cells(include_local_hours=name != 'summaries')
    identities = set()
    for mask, stream in product(S.MASKS, S.RESEARCH_STREAMS):
        if name == 'local_hour_summaries':
            rows = ((mask, stream, hour, metric) for hour, metric in product(range(24), ('MAE', 'WIS')))
        elif name == 'summaries':
            rows = ((mask, stream, metric, group, value)
                    for metric, (group, value) in product(('MAE', 'WIS'), dimensions))
        else:
            tails = {'coverage': tuple(level for level, *_ in S.INTERVALS),
                     'quantile_hits': S.QUANTILES, 'histogram': tuple(range(8))}[name]
            rows = ((mask, stream, group, value, tail) for (group, value), tail in product(dimensions, tails))
        identities.update(_digest(dict(zip(keys, row))) for row in rows)
    return identities


def validate_registered_table(name, table):
    """Reject omitted, additional or duplicated cells without evaluating a loss."""
    frame = table if isinstance(table, pd.DataFrame) else pd.DataFrame(table)
    keys = registered_table_keys(name)
    if not set(keys).issubset(frame):
        raise ValueError(f'registered {name} inventory lacks identity columns')
    observed = [_digest(dict(zip(keys, row))) for row in frame[list(keys)].itertuples(index=False, name=None)]
    expected, unique = registered_table_identities(name), set(observed)
    if len(observed) != len(unique) or unique != expected:
        raise ValueError(f'registered {name} inventory differs: missing={len(expected - unique)}, '
                         f'unexpected={len(unique - expected)}, duplicates={len(observed) - len(unique)}')
    return {'cells': len(expected), 'identities_sha256': _digest(sorted(expected))}


def validate_score_tables(tables):
    required = ('summaries', 'local_hour_summaries')
    if not set(required).issubset(tables):
        raise ValueError('registered score inventory lacks required tables')
    return {name: validate_registered_table(name, tables[name]) for name in required}


def validate_calibration_tables(tables):
    required = ('coverage', 'quantile_hits', 'histogram')
    if not set(required).issubset(tables):
        raise ValueError('registered calibration inventory lacks required tables')
    return {name: validate_registered_table(name, tables[name]) for name in required}


def _contained(root, relative):
    root, relative = Path(root).resolve(), Path(relative)
    if relative.is_absolute() or '..' in relative.parts:
        raise ValueError('analysis references must be project-relative')
    path = (root / relative).resolve()
    if not path.is_relative_to(root / 'reports/rebaseline'):
        raise ValueError('analysis artifacts must remain in reports/rebaseline')
    return path


def _decode_table(path, json_columns=None):
    table = pd.read_parquet(path)
    columns = table.attrs.get('cp26_json_columns', []) if json_columns is None else json_columns
    for name in columns:
        if name not in table:
            raise ValueError('encoded analysis table column is missing')
        table[name] = table[name].map(lambda value: _restore(json.loads(value)))
    table.attrs = {}
    return table


def _table(root, name, frame, *, keys):
    """Parquet tables carry semantic row IDs; manifest binds their ordered digest."""
    frame = frame.copy().reset_index(drop=True)
    if len(frame) and not set(keys).issubset(frame):
        raise ValueError(f'{name} lacks its registered row identity')
    identities = [_digest({'table': name, 'keys': dict(zip(keys, row))})
                  for row in frame[list(keys)].itertuples(index=False, name=None)] if len(frame) else []
    if len(set(identities)) != len(identities):
        raise ValueError(f'{name} has duplicate row identities')
    frame['row_id'] = identities
    json_columns = [str(column) for column in frame if frame[column].dtype == object and column != 'row_id']
    for name_column in json_columns:
        frame[name_column] = frame[name_column].map(lambda value: _restore(_plain(value)))
    encoded = frame.copy()
    for name_column in json_columns:
        encoded[name_column] = encoded[name_column].map(lambda value: json.dumps(
            _plain(value), sort_keys=True, separators=(',', ':'), allow_nan=False))
    encoded.attrs = {'cp26_json_columns': json_columns}
    relative = OUTPUT / 'scores' / f'{name}.parquet'
    path = _contained(root, relative)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        observed = _decode_table(path, json_columns)
        if not observed.equals(frame):
            raise ValueError(f'analysis table differs; preserve original {name}')
    else:
        temporary = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
        encoded.to_parquet(temporary, index=False)
        try:
            os.link(temporary, path)
        except FileExistsError:
            if not _decode_table(path, json_columns).equals(frame):
                raise ValueError(f'concurrent analysis table differs: {name}')
        finally:
            temporary.unlink(missing_ok=True)
    return {'path': str(relative), 'sha256': _sha(path), 'rows': len(frame),
            'row_ids_sha256': _digest(identities), 'row_id_columns': list(keys), 'json_columns': json_columns}


def _load_table(root, reference):
    from cp25 import guard
    guard.admit_delivery_day(guard.BOUNDARY)
    path = _contained(root, reference['path'])
    if _sha(path) != reference['sha256']:
        raise ValueError('analysis table content hash changed')
    frame = _decode_table(path, reference.get('json_columns', []))
    if (len(frame) != reference['rows'] or 'row_id' not in frame or frame.row_id.duplicated().any()
            or _digest(frame.row_id.tolist()) != reference['row_ids_sha256']):
        raise ValueError('analysis table row inventory changed')
    keys = reference['row_id_columns']
    expected = [_digest({'table': path.stem, 'keys': dict(zip(keys, row))})
                for row in frame[list(keys)].itertuples(index=False, name=None)] if len(frame) else []
    if expected != frame.row_id.tolist():
        raise ValueError('analysis table semantic row identities changed')
    if 'delivery_date' in frame:
        for day in frame.delivery_date.drop_duplicates():
            guard.admit_delivery_day(pd.Timestamp(day).date())
    if 'timestamp_utc' in frame and len(frame):
        guard.admit_rows(pd.DatetimeIndex(frame.timestamp_utc))
    return frame


def _ledger(root, protocol, population):
    from .execution import Paths, calendar, manifest_for, _read as read_execution, DAY_SCHEMA
    from .scoring import RESEARCH_STREAMS
    from .members import RUN_MEMBERS
    from .state import ReplayState
    paths = Paths.build(root)
    manifest = manifest_for(paths, protocol)
    saved = ReplayState.load(paths.output / 'replay-state.json', protocol)
    days = calendar()
    if len(days) != 2441 or {path.stem for path in (paths.output / 'forecast-ledger').glob('*.json')} != {str(d) for d in days}:
        raise ValueError('scoring requires exactly all 2441 frozen forecast days')
    expected, records, inventory, supplied_keys = set(RESEARCH_STREAMS), [], [], {}
    measurement = set(population.calendar.delivery_date)
    for day in days:
        path = paths.day(day)
        value = read_execution(path)
        if (value.get('schema') != DAY_SCHEMA or value.get('day') != str(day)
                or value.get('protocol') != protocol or value.get('protocol_identity') != protocol
                or value.get('execution_manifest_sha256') != manifest['content_sha256']
                or set(value.get('policies', {})) != expected or set(value.get('members', {})) != set(RUN_MEMBERS)
                or any(set(value['members'][route]) != set(members) for route, members in RUN_MEMBERS.items())):
            raise ValueError('forecast day protocol/policy/member inventory differs')
        if value['policies'] != {key: policy.records.get(str(day)) for key, policy in saved.policies.items()}:
            raise ValueError('forecast day differs from original saved policy issuance')
        for stream, record in value['policies'].items():
            if record.get('day') != str(day) or f"{record.get('route')}/{record.get('policy')}" != stream:
                raise ValueError('forecast day contains an unbound policy record')
            if day in measurement:
                records.append(record)
        if day in measurement:
            supplied_keys[day] = pd.DatetimeIndex(pd.to_datetime(value['policies']['nominal/B0']['scoring_index'], utc=True))
        inventory.append({'day': str(day), 'path': str(path.relative_to(paths.root)),
                          'sha256': _sha(path), 'content_sha256': value['content_sha256']})
    population.verify_key_mapping(supplied_keys)
    return records, inventory, manifest


def _truth_and_states(root, population):
    from cp25.calendar import load_snapshot
    from cp25 import guard
    from . import diagnostics as D
    snapshot = load_snapshot(Path(root))
    guard.admit_rows(snapshot.timestamp_utc)
    history = snapshot[['timestamp_utc', 'delivery_date', 'price_eur_mwh']].copy()
    history = D.attach_registered_price_availability(history, root)
    truth = population.keys.merge(history[['timestamp_utc', 'delivery_date', 'price_eur_mwh']],
                                  on=['timestamp_utc', 'delivery_date'], how='left', validate='one_to_one')
    truth = truth.rename(columns={'price_eur_mwh': 'y_true'})
    population.verify_keys(truth)
    if not np.isfinite(truth.y_true.to_numpy(float)).all():
        raise ValueError('guarded snapshot lacks finite truth for frozen scoring keys')
    states = D.known_price_states(history, population.calendar.delivery_date.tolist())
    return truth, states, history.attrs.get('price_availability', {})


def _masked_fallbacks(daily):
    from .scoring import MASKS, fallback_summary
    rows = []
    for mask in MASKS:
        frame = fallback_summary(daily[daily[mask]])
        frame['mask'] = mask
        rows.append(frame)
    return pd.concat(rows, ignore_index=True)


def run_scores(root, protocol_identity, *, gate):
    """Score only after the complete forecast gate, with a separate released reference."""
    _gate(root, gate)
    from . import scoring as S
    population = S.load_population(root)
    records, inventory, execution = _ledger(root, protocol_identity, population)
    target = _output(root) / 'scores-manifest.json'
    if target.exists():
        existing = _read(target, SCORE_SCHEMA)
        if existing.get('forecast_ledger') != inventory:
            raise ValueError('scored forecast inventory changed')
        load_scores(root, protocol_identity)
        return existing
    truth, states, availability = _truth_and_states(root, population)
    result = S.score_records(records, truth, population, streams=S.RESEARCH_STREAMS,
                             known_states=states, production=True)
    registered_inventory = validate_score_tables(result)
    reference = S.score_released_reference(root, population, truth, known_states=states)
    tables = {}
    keys = {'daily': ('stream', 'delivery_date'), 'hourly': ('stream', 'timestamp_utc'),
            'summaries': ('mask', 'stream', 'metric', 'group', 'value'),
            'secondary_summaries': ('mask', 'stream', 'metric', 'summary'),
            'local_hour_summaries': ('mask', 'stream', 'local_hour', 'metric'), 'fallbacks': ('stream',)}
    for name, columns in keys.items():
        tables[name] = _table(root, name, result[name], keys=columns)
    tables['known_price_states'] = _table(root, 'known_price_states', states, keys=('delivery_date',))
    tables['fallbacks_by_mask'] = _table(root, 'fallbacks_by_mask', _masked_fallbacks(result['daily']), keys=('mask', 'stream'))
    references = {name: _table(root, f'released_{name}', reference[name], keys=columns)
                  for name, columns in {'daily': ('stream', 'delivery_date'), 'hourly': ('stream', 'timestamp_utc'),
                                        'key_differences': ('delivery_date',)}.items()}
    holdout = result['daily'][result['daily'].delivery_date.between(date(2026, 6, 9), date(2026, 9, 6)) & result['daily'].stream.eq('nominal/B0')]
    released = reference['daily'][reference['daily'].delivery_date.between(date(2026, 6, 9), date(2026, 9, 6))]
    descriptive = S.summaries(pd.concat([holdout, released], ignore_index=True))
    references['summaries'] = _table(root, 'released_summaries', descriptive[descriptive.stream.eq('nominal/released-v1')],
                                    keys=('mask', 'stream', 'metric', 'group', 'value'))
    value = _signed({'schema': SCORE_SCHEMA, 'status': 'COMPLETE', 'protocol': protocol_identity,
        'manifest': str(OUTPUT / 'scores-manifest.json'),
        'code_sha256': _code(root), 'streams': list(S.RESEARCH_STREAMS), 'masks': list(S.MASKS),
        'measurement_dates': len(population.calendar), 'forecast_days': 2441,
        'execution_manifest_sha256': execution['content_sha256'], 'forecast_ledger': inventory,
        'population_evidence': result['population_evidence'], 'price_availability': availability,
        'registered_score_inventory': registered_inventory,
        'tables': tables, 'released_reference': {'tables': references, 'metadata': reference['reference_metadata']},
        'interpretation': 'development control audit; released v1 descriptive on original holdout only'})
    return _immutable(target, value)


def load_scores(root, protocol_identity):
    """Read verified stored rows for inference/reporting; compute no new metric."""
    from .scoring import RESEARCH_STREAMS, MASKS
    manifest = _read(_output(root) / 'scores-manifest.json', SCORE_SCHEMA)
    if (manifest.get('status') != 'COMPLETE' or manifest.get('protocol') != protocol_identity
            or manifest.get('code_sha256') != _code(root) or manifest.get('streams') != list(RESEARCH_STREAMS)
            or manifest.get('masks') != list(MASKS) or manifest.get('forecast_days') != 2441):
        raise ValueError('scores manifest belongs to a different or incomplete execution')
    result = {name: _load_table(root, reference) for name, reference in manifest['tables'].items()}
    registered_inventory = validate_score_tables(result)
    if manifest.get('registered_score_inventory') != registered_inventory:
        raise ValueError('stored registered score inventory differs from its manifest')
    daily = result['daily']
    expected_dates = {day.date() for day in pd.date_range('2020-12-29', '2026-09-06', freq='D')}
    if (set(daily.stream) != set(RESEARCH_STREAMS) or len(daily) != len(expected_dates) * 13
            or any(set(cell.delivery_date) != expected_dates for _, cell in daily.groupby('stream'))
            or daily[['stream', 'delivery_date']].duplicated().any()):
        raise ValueError('stored score rows omit a stream or measurement day')
    result['population_evidence'] = manifest['population_evidence']
    reference = manifest['released_reference']
    result['released_reference'] = {name: _load_table(root, ref) for name, ref in reference['tables'].items()}
    result['released_reference']['metadata'] = reference['metadata']
    result['manifest'] = manifest
    return result


def _scalar(value):
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, (date, pd.Timestamp)):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _pack_numeric(result):
    arrays = {}
    def pack(value, field='root'):
        if isinstance(value, np.ndarray):
            if value.dtype.hasobject:
                raise ValueError('analysis numeric artifacts never pickle object arrays')
            name = f'array_{len(arrays):04d}'
            arrays[name] = value
            return {'array': name, 'shape': list(value.shape), 'dtype': str(value.dtype), 'field': field}
        if isinstance(value, dict):
            return {str(k): pack(v, f'{field}.{k}') for k, v in value.items()}
        if isinstance(value, (tuple, list)):
            return [pack(v, f'{field}.{i}') for i, v in enumerate(value)]
        return _scalar(value)
    return pack(result), arrays


def _validate_numeric(payload, result, arrays):
    seen = set()
    def check(saved, expected):
        if isinstance(expected, np.ndarray):
            if (not isinstance(saved, dict) or set(saved) != {'array', 'shape', 'dtype', 'field'}
                    or saved['array'] not in arrays or saved['shape'] != list(expected.shape)
                    or saved['dtype'] != str(expected.dtype)):
                raise ValueError('numeric array descriptor differs on deterministic replay')
            observed = arrays[saved['array']]
            if observed.dtype != expected.dtype or observed.shape != expected.shape or not np.array_equal(observed, expected, equal_nan=True):
                raise ValueError('numeric replicate array differs on deterministic replay')
            seen.add(saved['array'])
        elif isinstance(expected, dict):
            if not isinstance(saved, dict) or set(saved) != set(expected):
                raise ValueError('numeric result fields differ on deterministic replay')
            for key in expected:
                check(saved[key], expected[key])
        elif isinstance(expected, (tuple, list)):
            if not isinstance(saved, list) or len(saved) != len(expected):
                raise ValueError('numeric result sequence differs on deterministic replay')
            for a, b in zip(saved, expected):
                check(a, b)
        elif saved != _scalar(expected):
            raise ValueError('numeric result scalar differs on deterministic replay')
    check(payload, result)
    if seen != set(arrays.files):
        raise ValueError('numeric artifact contains unbound arrays')


def _compact(metadata, root, path, arrays_path):
    excluded = {'moment_dates', 'draw_metadata', 'protocol_sha256', 'analysis_code_sha256', 'source_scores_sha256'}
    compact = {key: value for key, value in metadata['result'].items()
               if key not in excluded and (not isinstance(value, dict) or 'array' not in value)}
    return {'endpoint_id': metadata['endpoint_id'], 'metadata_path': str(path.relative_to(root)),
            'metadata_sha256': _sha(path), 'arrays_path': str(arrays_path.relative_to(root)),
            'arrays_sha256': _sha(arrays_path), **compact}


def _numeric(root, relative_stem, result):
    """Reuse only fully regenerated scalars/arrays, retaining every RNG advance."""
    from . import inference as I
    root = Path(root).resolve()
    stem = _contained(root, relative_stem)
    metadata_path, arrays_path = stem.with_suffix('.json'), stem.with_suffix('.npz')
    result = {**result, 'row_id': I.endpoint_identity(result)}
    if not metadata_path.exists() and not arrays_path.exists():
        pending_arrays = arrays_path.with_name(arrays_path.name + '.tmp')
        pending_metadata = metadata_path.with_name(metadata_path.name + '.tmp')
        if pending_arrays.exists():
            try:
                with np.load(pending_arrays, allow_pickle=False) as arrays:
                    for name in arrays.files:
                        arrays[name]
            except (OSError, EOFError, ValueError, zipfile.BadZipFile):
                from .harness import art
                recovery = art() / LOCAL_NAME / 'analysis-recovery'
                recovery.mkdir(parents=True, exist_ok=True)
                for partial in (pending_arrays, pending_metadata):
                    if partial.exists():
                        saved = recovery / f'{stem.name}.{_sha(partial)}.{partial.name[-8:]}'
                        if saved.exists() and _sha(saved) != _sha(partial):
                            raise ValueError('partial analysis recovery identity changed')
                        if not saved.exists():
                            shutil.copy2(partial, saved)
                        if _sha(saved) != _sha(partial):
                            raise ValueError('partial analysis recovery copy failed')
                        partial.unlink()
        if pending_arrays.exists() and pending_metadata.exists():
            metadata = json.loads(pending_metadata.read_text())
            if metadata.get('arrays_sha256') != _sha(pending_arrays):
                raise ValueError('interrupted numeric artifact has unbound partial arrays')
            with np.load(pending_arrays, allow_pickle=False) as arrays:
                _validate_numeric(metadata['result'], result, arrays)
            pending_arrays.replace(arrays_path)
            pending_metadata.replace(metadata_path)
        elif pending_arrays.exists():
            payload, _ = _pack_numeric(result)
            with np.load(pending_arrays, allow_pickle=False) as arrays:
                _validate_numeric(payload, result, arrays)
            atomic(pending_metadata, {'schema': 'cp26-numeric-artifact-v1', 'endpoint_id': I.endpoint_identity(result),
                'arrays_file': arrays_path.name, 'arrays_sha256': _sha(pending_arrays), 'result': payload})
            pending_arrays.replace(arrays_path)
            pending_metadata.replace(metadata_path)
        elif pending_metadata.exists():
            raise ValueError('interrupted numeric artifact is incomplete; preserve recovery material')
        else:
            I.write_result_artifact(root, relative_stem, result)
    if arrays_path.exists() and not metadata_path.exists():
        # Crash after the writer's first rename: the regenerated result proves
        # the orphan NPZ, so creating its missing JSON does not overwrite history.
        payload, _ = _pack_numeric(result)
        with np.load(arrays_path, allow_pickle=False) as arrays:
            _validate_numeric(payload, result, arrays)
        atomic(metadata_path, {'schema': 'cp26-numeric-artifact-v1', 'endpoint_id': I.endpoint_identity(result),
                               'arrays_file': arrays_path.name, 'arrays_sha256': _sha(arrays_path), 'result': payload})
    if not metadata_path.exists() or not arrays_path.exists():
        raise ValueError('numeric artifact lost one of its bound files')
    metadata = json.loads(metadata_path.read_text())
    if (metadata.get('schema') != 'cp26-numeric-artifact-v1' or metadata.get('endpoint_id') != I.endpoint_identity(result)
            or metadata.get('arrays_file') != arrays_path.name or metadata.get('arrays_sha256') != _sha(arrays_path)):
        raise ValueError('numeric endpoint/hash identity changed')
    with np.load(arrays_path, allow_pickle=False) as arrays:
        _validate_numeric(metadata['result'], result, arrays)
    return _compact(metadata, root, metadata_path, arrays_path)


def _bound_result(result, protocol, score_manifest, code):
    return {**result, 'protocol_sha256': _digest(protocol), 'source_scores_sha256': score_manifest['content_sha256'],
            'analysis_code_sha256': _digest(code)}


def workload_counts():
    from . import scoring as S, inference as I
    streams, comparisons = len(S.RESEARCH_STREAMS), len(I.comparison_registry(S.RESEARCH_STREAMS))
    return {'research_streams': streams, 'comparisons': comparisons,
            'bootstrap_endpoints': len(I.VARIANTS) * ((comparisons + streams) * 4 * 12 * 2 + streams * 4 * 2 * 2),
            'mcs_cells': 4 * 2, 'secondary_DM_GW_entries': comparisons * 4 * 2, 'moment_cells': streams * 24 * 3}


def _record_analysis_completion(root, stage, value):
    from .execution import record_evidence_once
    record_evidence_once(Budget(), f"analysis/{_digest(value['protocol'])}/{stage}", {
        'manifest_sha256': value['content_sha256'], 'status': value['status'],
        'code_sha256': _digest(value['code_sha256']),
        'scores_manifest_sha256': value['scores_manifest_sha256']})
    return value


def _validate_completed(root, stage, value, protocol, scores, code):
    if (value.get('status') != 'COMPLETE' or value.get('protocol') != protocol
            or value.get('code_sha256') != code
            or value.get('scores_manifest_sha256') != scores['content_sha256']):
        raise ValueError('completed analysis protocol/code/score identity changed')
    counts = workload_counts()
    if stage == 'inference':
        if (value.get('bootstrap_endpoint_count') != counts['bootstrap_endpoints']
                or len(value.get('secondary_tests', [])) != counts['secondary_DM_GW_entries']
                or len(value.get('mcs', {}).get('cells', [])) != counts['mcs_cells']):
            raise ValueError('completed inference inventory changed')
        references = value['endpoints']
        if len(references) != sum(counts[k] for k in ('bootstrap_endpoints', 'secondary_DM_GW_entries', 'mcs_cells')):
            raise ValueError('completed inference endpoint count changed')
    else:
        references = value.get('moment_family', {}).get('cells', [])
        if len(references) != counts['moment_cells'] or len(value['moment_family'].get('holm', [])) != counts['moment_cells']:
            raise ValueError('completed diagnostic inventory changed')
        tables = {name: _load_table(root, reference) for name, reference in value.get('tables', {}).items()}
        if value.get('registered_calibration_inventory') != validate_calibration_tables(tables):
            raise ValueError('completed registered calibration inventory differs from its manifest')
    if len({row['endpoint_id'] for row in references}) != len(references):
        raise ValueError('completed analysis has duplicate endpoint identities')
    for row in references:
        metadata_path, arrays_path = _contained(root, row['metadata_path']), _contained(root, row['arrays_path'])
        if _sha(metadata_path) != row['metadata_sha256'] or _sha(arrays_path) != row['arrays_sha256']:
            raise ValueError('completed analysis artifact content changed')
        metadata = json.loads(metadata_path.read_text())
        bound = metadata['result']
        if (metadata.get('schema') != 'cp26-numeric-artifact-v1' or metadata.get('endpoint_id') != row['endpoint_id']
                or metadata.get('arrays_file') != arrays_path.name or metadata.get('arrays_sha256') != row['arrays_sha256']
                or bound.get('protocol_sha256') != _digest(protocol)
                or bound.get('source_scores_sha256') != scores['content_sha256']
                or bound.get('analysis_code_sha256') != _digest(code)
                or _compact(metadata, Path(root).resolve(), metadata_path, arrays_path) != row):
            raise ValueError('completed analysis endpoint binding changed')
    return value


def run_inference(root, protocol_identity, *, gate):
    _gate(root, gate)
    from . import inference as I, scoring as S
    scores = load_scores(root, protocol_identity)
    daily, states, code = scores['daily'], scores['known_price_states'], _code(root)
    target = _output(root) / 'inference-manifest.json'
    if target.exists():
        value = _validate_completed(root, 'inference', _read(target, INFERENCE_SCHEMA),
                                    protocol_identity, scores['manifest'], code)
        return _record_analysis_completion(root, 'inference', value)
    comparisons = I.comparison_registry(S.RESEARCH_STREAMS)
    references = []
    # Always regenerate the full iterator, including earlier completed yields.
    # Skipping writes is safe; skipping a yield would change the next RNG draw.
    for result in I.iter_inference(daily, comparisons=comparisons, production=True):
        bound = _bound_result(result, protocol_identity, scores['manifest'], code)
        stem = OUTPUT / 'inference' / I.endpoint_identity(bound)
        references.append(_numeric(root, stem, bound))
    if len(references) != workload_counts()['bootstrap_endpoints']:
        raise ValueError('inference iterator omitted a frozen endpoint')
    mcs = []
    for result in I.iter_mcs(daily, streams=S.RESEARCH_STREAMS, production=True):
        bound = _bound_result({'kind': 'model_confidence_set', **result}, protocol_identity, scores['manifest'], code)
        mcs.append(_numeric(root, OUTPUT / 'inference' / I.endpoint_identity(bound), bound))
    secondary = []
    for left, right in comparisons:
        for mask in S.MASKS:
            for metric in ('MAE', 'WIS'):
                pair, counts = S.paired_losses(daily, left, right, metric=metric, mask=mask)
                result = {'kind': 'secondary_DM_GW', 'left': left, 'right': right, 'mask': mask, 'metric': metric,
                          **counts, 'DM': I.diebold_mariano(pair), 'GW': I.giacomini_white(pair, states)}
                bound = _bound_result(result, protocol_identity, scores['manifest'], code)
                secondary.append(_numeric(root, OUTPUT / 'inference' / I.endpoint_identity(bound), bound))
    counts = workload_counts()
    if len(mcs) != counts['mcs_cells'] or len(secondary) != counts['secondary_DM_GW_entries']:
        raise ValueError('secondary inference inventory is incomplete')
    value = _signed({'schema': INFERENCE_SCHEMA, 'status': 'COMPLETE', 'protocol': protocol_identity,
        'manifest': str(OUTPUT / 'inference-manifest.json'),
        'code_sha256': code, 'scores_manifest_sha256': scores['manifest']['content_sha256'],
        'workload': counts, 'specification': I.prospective_specification(), 'comparisons': list(comparisons),
        'endpoints': references + secondary + mcs, 'bootstrap_endpoint_count': len(references),
        'secondary_tests': secondary, 'mcs': {'cells': mcs}, 'interpretation': 'descriptive control audit only'})
    return _record_analysis_completion(root, 'inference',
        _immutable(_output(root) / 'inference-manifest.json', value))


def run_diagnostics(root, protocol_identity, *, gate):
    _gate(root, gate)
    from . import diagnostics as D, inference as I, scoring as S
    scores = load_scores(root, protocol_identity)
    daily, hourly, code = scores['daily'], scores['hourly'], _code(root)
    target = _output(root) / 'diagnostics-manifest.json'
    if target.exists():
        value = _validate_completed(root, 'diagnostics', _read(target, DIAGNOSTIC_SCHEMA),
                                    protocol_identity, scores['manifest'], code)
        return _record_analysis_completion(root, 'diagnostics', value)
    tables = {}
    calibration = D.calibration_tables(hourly, daily)
    registered_inventory = validate_calibration_tables(calibration)
    for name, tail in (('coverage', 'interval'), ('quantile_hits', 'quantile'), ('histogram', 'bin')):
        tables[name] = _table(root, name, calibration[name], keys=('mask', 'stream', 'group', 'value', tail))
    tables['christoffersen'] = _table(root, 'christoffersen', D.christoffersen(hourly, daily), keys=('mask', 'stream', 'interval'))
    realized = D.realized_outcome_slices(hourly, daily)
    tables['slices'] = _table(root, 'realized_slices', realized['slices'], keys=('mask', 'stream', 'slice', 'metric'))
    tables['thresholds'] = _table(root, 'realized_thresholds', realized['thresholds'], keys=('stream',))
    tables['known_price_states'] = scores['manifest']['tables']['known_price_states']
    tables['fallbacks_by_mask'] = scores['manifest']['tables']['fallbacks_by_mask']
    cells = []
    for result in D.iter_moment_family(hourly, daily, streams=S.RESEARCH_STREAMS, production=True):
        bound = _bound_result({'kind': 'moment_test', **result}, protocol_identity, scores['manifest'], code)
        cells.append(_numeric(root, OUTPUT / 'diagnostics' / I.endpoint_identity(bound), bound))
    adjustment = D.moment_holm(cells, streams=S.RESEARCH_STREAMS, production=True)
    holm = [{'identity': identity, **row} for identity, row in sorted(adjustment.items())]
    tables['moment_holm'] = _table(root, 'moment_holm', pd.DataFrame(holm), keys=('identity',))
    inference = _read(_output(root) / 'inference-manifest.json', INFERENCE_SCHEMA)
    if (inference.get('status') != 'COMPLETE' or inference.get('protocol') != protocol_identity
            or inference.get('code_sha256') != code or inference.get('scores_manifest_sha256') != scores['manifest']['content_sha256']):
        raise ValueError('diagnostic MCS references do not bind the completed inference')
    if len(cells) != workload_counts()['moment_cells']:
        raise ValueError('moment family must retain exactly 936 registered cells')
    value = _signed({'schema': DIAGNOSTIC_SCHEMA, 'status': 'COMPLETE', 'protocol': protocol_identity,
        'manifest': str(OUTPUT / 'diagnostics-manifest.json'),
        'code_sha256': code, 'scores_manifest_sha256': scores['manifest']['content_sha256'],
        'specification': D.prospective_specification(), 'tables': tables,
        'registered_calibration_inventory': registered_inventory,
        'moment_family': {'cells': cells, 'holm': holm}, 'mcs': inference['mcs'],
        'inference_manifest_sha256': inference['content_sha256'], 'interpretation': 'diagnostic/descriptive only'})
    return _record_analysis_completion(root, 'diagnostics',
        _immutable(_output(root) / 'diagnostics-manifest.json', value))


def _peak_rss():
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == 'darwin' else value * 1024)


def _measure(name, run):
    started_epoch, began = time.time(), time.perf_counter()
    result = run()
    seconds = time.perf_counter() - began
    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError('fixture timing must be finite and positive')
    return result, {'name': name, 'started_epoch': started_epoch, 'ended_epoch': time.time(),
                    'wall_seconds': seconds, 'peak_rss_bytes': _peak_rss()}


def _prior_costs(state):
    """Price future tests/parity from retained supervised durations, never reserves."""
    estimates = {}
    for category, fragments in (('tests', ('test', 'review', 'preparation-suite')), ('parity', ('parity',))):
        observed = []
        for job in state['jobs']:
            if job.get('running') or not any(word in str(job.get('name', '')).lower() for word in fragments):
                continue
            wall, workers = job.get('wall_seconds'), job.get('workers')
            if (isinstance(wall, bool) or not isinstance(wall, (float, int)) or not math.isfinite(wall)
                    or wall <= 0 or type(workers) is not int or not 1 <= workers <= 4):
                continue
            observed.append({'job_id': job.get('job_id'), 'name': job['name'], 'wall_seconds': wall,
                             'workers': workers, 'machine_seconds': wall * workers,
                             'completion_clock': job.get('completion_clock'), 'ended_epoch': job.get('ended_epoch'),
                             'abort_reason': job.get('abort_reason')})
        if not observed:
            raise ValueError(f'future {category} cost has no identified supervised timing basis')
        maximum = max(observed, key=lambda item: item['machine_seconds'])
        estimates[category] = {'classification': 'conservative', 'machine_seconds': maximum['machine_seconds'] * 2,
            'basis': f'two future complete {category}/independent-review equivalents at maximum retained '
                     'CP-26 supervised machine duration, including unsuccessful completed jobs',
            'evidence_sha256': _digest(observed), 'source_jobs': observed, 'maximum_job': maximum}
    return estimates


def _calibration_fixture(dates):
    from cp25.calendar import expected_hours
    from . import scoring as S
    rng = np.random.default_rng(26000001)
    keys = pd.DatetimeIndex([timestamp for day in dates for timestamp in expected_hours(day)])
    delivery = keys.tz_convert('Europe/Berlin').date
    calendar = pd.DataFrame({'delivery_date': dates, 'canonical_hours': [len(expected_hours(day)) for day in dates],
                             'eligible': True})
    flags = pd.DataFrame({'delivery_date': dates, 'negative_price_risk': np.arange(len(dates)) % 2,
                          'high_volatility': (np.arange(len(dates)) // 2) % 2})
    groups = S.calendar_groups(calendar, flags)
    hourly, daily = [], []
    for stream in S.RESEARCH_STREAMS:
        truth = rng.normal(size=len(keys))
        q = rng.normal(scale=.2, size=(len(keys), 1)) + np.array([-3., -2., -1., 0., 1., 2., 3.])[None, :]
        h = pd.DataFrame({'timestamp_utc': keys, 'delivery_date': delivery, 'y_true': truth,
                          'stream': stream, 'local_hour': keys.tz_convert('Europe/Berlin').hour})
        for name, value in S.hourly_losses(truth, q).items():
            h[name] = value
        for index, name in enumerate(S.QUANTILES):
            h[name] = q[:, index]
        hourly.append(h)
        d = groups.copy()
        d['stream'], d['available'] = stream, True
        d['MAE'] = rng.uniform(.5, 1.5, len(d))
        d['WIS'] = rng.uniform(.5, 1.5, len(d))
        for mask in S.MASKS:
            d[mask] = True  # A synthetic computational upper envelope for all four masks.
        daily.append(d)
    return pd.concat(hourly, ignore_index=True), pd.concat(daily, ignore_index=True)


def benchmark(root):
    """Time full registered algorithms on synthetic values; emit no metric values.

    This job is prospectively available before the training pilot. Its only
    persisted observations are time/RSS and identities; it reads no market data.
    Additional prices are estimates, never resource charges or new reserves.
    """
    from . import harness as H, inference as I, diagnostics as D, scoring as S
    from cp25.calendar import expected_hours
    if not os.environ.get('CP26_MONITORED_JOB'):
        raise RuntimeError('fixture benchmark requires the cumulative CP-26 monitor')
    H.continuation_authority(Path(root))
    H.check_blas()
    code = _code(root)
    target = _output(root) / 'fixture-benchmark.json'
    if target.exists():
        value = _read(target, BENCHMARK_SCHEMA)
        if value.get('code_sha256') != code:
            raise ValueError('fixture benchmark code identity changed; retain original evidence')
        return value
    prices = _prior_costs(Budget().read())
    dates = [value.date() for value in pd.date_range('2020-12-29', '2026-09-06', freq='D')]
    rng = np.random.default_rng(26000002)
    pair = pd.DataFrame({'delivery_date': dates, 'a': rng.uniform(.5, 2.5, len(dates)),
                         'b': rng.uniform(.5, 2.5, len(dates))})
    timings = []
    for variant, block_days, stratified in I.VARIANTS:
        result, timing = _measure(f'paired_{variant}_10000', lambda block_days=block_days, stratified=stratified:
                                  I.paired_bootstrap(pair, block_days=block_days, stratified=stratified, production=True))
        if len(result['replicate_differences']) != 10000:
            raise ValueError('fixture did not exercise the complete paired replicate workload')
        timings.append(timing)
        del result
    for group in ('season', 'year'):
        result, timing = _measure(f'secondary_{group}_10000', lambda group=group:
                                  I.secondary_ratio_bootstrap(pair, group=group, block_days=7, production=True))
        if len(result['replicate_ratios']) != 10000:
            raise ValueError('fixture did not exercise the complete secondary replicate workload')
        timings.append(timing)
        del result
    # Large distinct offsets force the full elimination chain, rather than
    # pricing all production MCS calls from an early-stopping equal-mean fixture.
    losses = pd.DataFrame(rng.uniform(.5, 2.5, size=(len(dates), 13)) + 5 * np.arange(13)[None, :],
                          index=dates, columns=sorted(S.RESEARCH_STREAMS))
    result, timing = _measure('mcs_13streams_10000', lambda:
                              I.model_confidence_set(losses, mask='all_eligible', metric='MAE', production=True))
    if len(result.get('replicate_means', [])) != 10000:
        raise ValueError('fixture did not exercise the complete MCS replicate workload')
    if result.get('status') != 'computable' or len(result.get('history', [])) != 12 or len(result.get('remaining', [])) != 1:
        raise ValueError('fixture did not exercise all 12 MCS elimination steps')
    timings.append(timing)
    del result, losses
    flags = pd.DataFrame({'delivery_date': dates, 'negative_price_risk': rng.integers(0, 2, len(dates)),
                          'high_volatility': rng.integers(0, 2, len(dates)), 'volatility_unknown': 0})
    _, timing = _measure('secondary_DM_GW', lambda: (I.diebold_mariano(pair), I.giacomini_white(pair, flags)))
    timings.append(timing)
    moment_dates = [day for day in dates if len(expected_hours(day)) == 24]
    hits = rng.integers(0, 2, len(moment_dates)).astype(bool)
    cell = pd.DataFrame({'delivery_date': moment_dates, 'y_true': np.where(hits, 0., 3.), 'coverage50': hits})
    for name, value in zip(S.QUANTILES, (-4., -3., -1., 0., 1., 4., 5.)):
        cell[name] = value
    result, timing = _measure('moment_cell_100000', lambda: D.moment_test(cell, interval=50, production=True))
    if len(result['replicate_statistics']) != 100000:
        raise ValueError('fixture did not exercise the complete moment replicate workload')
    timings.append(timing)
    del result, cell, pair
    def calibration():
        hourly, daily = _calibration_fixture(dates)
        D.calibration_tables(hourly, daily)
        D.christoffersen(hourly, daily)
        D.realized_outcome_slices(hourly, daily)
    _, timing = _measure('calibration_13streams_four_masks', calibration)
    timings.append(timing)
    counts = workload_counts()
    measured = {row['name']: row['wall_seconds'] for row in timings}
    expensive_endpoint = max(seconds for name, seconds in measured.items()
                             if name.startswith(('paired_', 'secondary_season_', 'secondary_year_')))
    bootstrap_seconds = 1.5 * (expensive_endpoint * counts['bootstrap_endpoints']
        + measured['mcs_13streams_10000'] * counts['mcs_cells']
        + measured['secondary_DM_GW'] * counts['secondary_DM_GW_entries'])
    diagnostics_seconds = 1.5 * (measured['moment_cell_100000'] * counts['moment_cells']
                                 + measured['calibration_13streams_four_masks'])
    evidence_sha = _digest({'timings': timings, 'fixture_dates': [str(dates[0]), str(dates[-1])],
                            'fixture_seed': [26000001, 26000002], 'code_sha256': code})
    prices.update(bootstrap={'classification': 'conservative', 'machine_seconds': bootstrap_seconds,
        'evidence_sha256': evidence_sha,
        'basis': '1.5×maximum full-support paired/season/year endpoint wall time×15808, plus1.5×full13-stream '
                 'MCS elimination-chain wall time×8 and1.5×DM/GW wall time×208; four paired variants and '
                 'secondary7-day stress, complete10000 draws; margin includes storage/validation; synthetic values only'},
        diagnostics={'classification': 'conservative', 'machine_seconds': diagnostics_seconds,
        'evidence_sha256': evidence_sha,
        'basis': '1.5×full100000-replicate moment-cell wall time×936, plus1.5×complete13-stream/four-mask '
                 'calibration/Christoffersen/realized-slice wall time; synthetic values only'})
    prices['failed_jobs_and_resume'] = {'classification': 'conservative', 'machine_seconds': .1 * sum(
        row['machine_seconds'] for row in prices.values()), 'evidence_sha256': evidence_sha,
        'basis': '10% of priced analysis/tests/parity subtotal for failed jobs and deterministic RNG replay; '
                 'project job must extend this to include its pilot-priced daily fits and searches',
        'extend_with_pilot_fit_and_search_prices': True}
    value = _signed({'schema': BENCHMARK_SCHEMA, 'status': 'COMPLETE', 'fixture_only': True,
        'manifest': str(OUTPUT / 'fixture-benchmark.json'),
        'evaluation_scores_computed': False, 'market_data_read': False, 'code_sha256': code,
        'fixture_identity_sha256': _digest({'dates': [str(d) for d in dates], 'seeds': [26000001, 26000002]}),
        'timings': timings, 'workload': counts, 'additional_costs': prices})
    return _immutable(target, value)
