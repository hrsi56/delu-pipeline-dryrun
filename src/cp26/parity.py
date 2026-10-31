"""Fail-closed, bitwise old-settings gate (capstone v21-r13 §24.7 and §24.5).

Only committed pre-boundary central vectors are read. Four origins, fixed without inspecting
losses, cover old folds 2–5. Every member is fitted fresh through ``members.Context`` under
``harness.OLD``; CP-24's frozen ensembles are used here solely as parity fixtures. The same
gate also fits the no-load variant with its inputs restored. No score or residual replay is
computed. A passing report contains every expected check; a failure is saved before raising.

The caller must run this job under CP-26's resource monitor (one worker, BLAS 1, CPU only).
``--inspect`` verifies reference custody and their internal central identities without fitting;
it cannot create a passing gate report.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import date, datetime, timezone
import json
from pathlib import Path
import subprocess
import time

import numpy as np
import pandas as pd

from cp15.data import array_hash, sha
from cp20.budget import atomic
from cp21.lgbm import hg_central, hgl_central
from cp24.execution import v5_central
from cp25 import guard
from . import harness as H
from . import members as M

OUT = Path('reports/rebaseline/continuation-v21-r14-accounting-2/parity.json')
SCHEMA = 'cp26-old-settings-parity-v1'
SAMPLES = (
    ('fold_2', date(2021, 4, 1)),
    ('fold_3', date(2022, 7, 1)),
    ('fold_4', date(2025, 5, 1)),
    ('fold_5', date(2026, 1, 8)),
)
MEMBERS = ('B0', 'A1', 'B2', 'A1_w', 'B2_w', 'L-N', 'L-R', 'D2')
POLICIES = ('v2', 'v3', 'v4', 'v5')
PATHS = ('nominal', 'noload_restored')


@dataclass(frozen=True)
class Source:
    evidence_commit: str
    manifest: str
    files: tuple[str, ...]


SOURCES = {
    'cp15': Source('1bdc75b8ab943092bb8de6ba893defb9e12250d8',
                  'reports/cp15/artifact-manifest.json', ('reports/cp15/predictions.parquet',)),
    'cp16': Source('5ec8a92a4032569b31a1a4f0bb3c512793d15a78',
                  'reports/v2-causal/artifact-manifest.json', ('reports/v2-causal/predictions.parquet',)),
    'cp20': Source('a7a9b2e3a4d0147a0c82835a853277e9c81c7945',
                  'reports/weather-ablation/artifact-manifest.json',
                  ('reports/weather-ablation/predictions.parquet',)),
    'cp21': Source('1d13f99b1ab12f64719b439d0a9d4703bc7e1bdb',
                  'reports/block-challenger/artifact-manifest.json',
                  ('reports/block-challenger/predictions.parquet',)),
    'cp23': Source('928bc1389627499fba14222e552feb62492f6e2d',
                  'reports/distribution-challenger/artifact-manifest.json',
                  ('reports/distribution-challenger/members.parquet',)),
    'cp24': Source('05b7e618de8de2f3f59c3e62aaf1664807bf397e',
                  'reports/ddnn2/artifact-manifest.json',
                  ('reports/ddnn2/attempt-1/members.parquet',
                   'reports/ddnn2/attempt-1/predictions.parquet',
                   'reports/ddnn2/rounds/round-1/ensembles.json')),
}


class ParityFailed(H.HarnessError):
    """The gate failed. Its partial, exact evidence is in ``report`` and at ``OUT``."""

    def __init__(self, message, report):
        super().__init__(message)
        self.report = report


def _stamp():
    return datetime.now(timezone.utc).isoformat()


def verify_sources(root: Path) -> dict:
    """Bind each reference to both its historical committed manifest and its immutable Git blob."""
    root = Path(root)
    result = {}
    # These are designated old-development fixtures; row materialisation is guarded below.
    guard.admit_delivery_day(H.OLD_BOUNDARY)
    for label, source in SOURCES.items():
        manifest_hash = sha(root / source.manifest)
        if H.git_blob_sha256(root, source.evidence_commit, source.manifest) != manifest_hash:
            raise H.HarnessError(f'{label} parity manifest differs from its evidence commit')
        manifest = json.loads((root / source.manifest).read_text())['artifact_sha256']
        files = {}
        for name in source.files:
            actual = sha(root / name)
            if manifest.get(name) != actual or H.git_blob_sha256(root, source.evidence_commit, name) != actual:
                raise H.HarnessError(f'parity reference is not its committed, manifest-bound artifact: {name}')
            files[name] = actual
        result[label] = {'evidence_commit': source.evidence_commit, 'manifest': source.manifest,
                         'manifest_sha256': manifest_hash, 'files_sha256': files}
    return result


def _day(root: Path, name: str, fold: str, day: date, values: tuple[str, ...], policy=None) -> pd.DataFrame:
    """Read only the exact sampled origin and needed central columns, never truths or losses."""
    guard.admit_delivery_day(day)
    if day < H.FLOOR or day > H.OLD_BOUNDARY:
        raise H.HarnessError(f'parity origin outside the old settings: {day}')
    filters = [('delivery_date', '==', day), ('fold', '==', fold)]
    columns = ['fold', 'timestamp_utc', 'delivery_date', *values]
    if policy is not None:
        filters.append(('policy', '==', policy))
        columns.append('policy')
    frame = pd.read_parquet(Path(root) / name, columns=columns, filters=filters)
    if frame.empty:
        raise H.HarnessError(f'empty committed parity vector: {name} {fold} {day} {policy}')
    for d in frame.delivery_date.unique():
        guard.admit_delivery_day(d)
        if d != day:
            raise H.HarnessError('parity reference filter returned another day')
    ix = pd.DatetimeIndex(pd.to_datetime(frame.timestamp_utc, utc=True))
    guard.admit_rows(ix)
    if not np.array_equal(ix.tz_convert('Europe/Berlin').date, frame.delivery_date.to_numpy()):
        raise H.HarnessError('parity reference delivery day differs from its timestamp')
    if ix.has_duplicates or not frame.fold.eq(fold).all():
        raise H.HarnessError('duplicate or wrong-fold parity reference keys')
    frame['timestamp_utc'] = ix
    return frame.sort_values('timestamp_utc').reset_index(drop=True)


def _vector(frame, column):
    result = frame[column].to_numpy(dtype='<f8', copy=True)
    if result.ndim != 1 or not np.isfinite(result).all():
        raise H.HarnessError(f'invalid committed central vector: {column}')
    return np.ascontiguousarray(result)


def _keys(frame):
    return pd.DatetimeIndex(frame.timestamp_utc).tz_convert('UTC').as_unit('ns')


def bitwise_equal(left, right):
    """Exact float64 bytes, including the sign of zero; NaNs are inadmissible."""
    a, b = np.asarray(left), np.asarray(right)
    if a.dtype.kind != 'f' or b.dtype.kind != 'f' or a.dtype.itemsize != 8 or b.dtype.itemsize != 8:
        return False
    a, b = np.ascontiguousarray(a, dtype='<f8'), np.ascontiguousarray(b, dtype='<f8')
    return a.shape == b.shape and bool(np.isfinite(a).all() and np.isfinite(b).all()) and a.tobytes() == b.tobytes()


def assemble(central: dict) -> dict:
    """The landed arithmetic order; reassociating these blends changes float bits."""
    return {'v2': hg_central(central['A1'], central['B2']),
            'v3': hg_central(central['A1_w'], central['B2_w']),
            'v4': hgl_central(central['A1_w'], central['B2_w'], central['L-N'], central['L-R']),
            'v5': v5_central(central['A1_w'], central['B2_w'], central['L-N'], central['L-R'], central['D2'])}


def load_references(root: Path) -> tuple[dict, dict]:
    """Verify custody before reading any vector; corroborate inherited members across releases."""
    root = Path(root)
    custody = verify_sources(root)
    frozen = json.loads((root / SOURCES['cp24'].files[2]).read_text())
    refs = {}
    for fold, day in SAMPLES:
        cp15 = {n: _day(root, SOURCES['cp15'].files[0], fold, day, ('central',), n)
                for n in ('B0', 'A1', 'B2')}
        cp16 = _day(root, SOURCES['cp16'].files[0], fold, day, ('central',), 'V2-H')
        cp20 = _day(root, SOURCES['cp20'].files[0], fold, day, ('central',), 'HG')
        cp21 = {n: _day(root, SOURCES['cp21'].files[0], fold, day, ('central',), n)
                for n in ('L-N', 'L-R', 'HGL')}
        cp23 = _day(root, SOURCES['cp23'].files[0], fold, day, ('A1', 'B2', 'L-N', 'L-R', 'HG', 'HGL'))
        cp24 = _day(root, SOURCES['cp24'].files[0], fold, day, ('A1', 'B2', 'L-N', 'L-R', 'D2', 'c_v5'))
        d2 = _day(root, SOURCES['cp24'].files[1], fold, day, ('central',), 'D2')
        v5 = _day(root, SOURCES['cp24'].files[1], fold, day, ('central',), 'v5')
        keys = _keys(cp24)
        frames = [*cp15.values(), cp16, cp20, *cp21.values(), cp23, d2, v5]
        if any(not _keys(f).equals(keys) for f in frames):
            raise H.HarnessError(f'committed parity references have unequal keys: {fold} {day}')
        expected = {n: _vector(f, 'central') for n, f in cp15.items()}
        expected.update({'A1_w': _vector(cp24, 'A1'), 'B2_w': _vector(cp24, 'B2'),
                         'L-N': _vector(cp21['L-N'], 'central'), 'L-R': _vector(cp21['L-R'], 'central'),
                         'D2': _vector(d2, 'central'), 'v2': _vector(cp16, 'central'),
                         'v3': _vector(cp20, 'central'), 'v4': _vector(cp21['HGL'], 'central'),
                         'v5': _vector(v5, 'central')})
        corroboration = [('A1_w', cp23, 'A1'), ('B2_w', cp23, 'B2'),
                         ('L-N', cp23, 'L-N'), ('L-R', cp23, 'L-R'),
                         ('L-N', cp24, 'L-N'), ('L-R', cp24, 'L-R'),
                         ('v3', cp23, 'HG'), ('v4', cp23, 'HGL'),
                         ('D2', cp24, 'D2'), ('v5', cp24, 'c_v5')]
        if any(not bitwise_equal(expected[n], _vector(f, c)) for n, f, c in corroboration):
            raise H.HarnessError(f'committed parity references disagree across releases: {fold} {day}')
        if any(not bitwise_equal(v, expected[n]) for n, v in assemble(expected).items()):
            raise H.HarnessError(f'committed members do not assemble their policy vectors: {fold} {day}')
        ensemble = frozen['folds'][fold]
        guard.admit_delivery_day(date.fromisoformat(ensemble['d0']))
        if ensemble.get('complete') is not True or len(ensemble.get('members', [])) != 8:
            raise H.HarnessError(f'CP-24 parity ensemble incomplete: {fold}')
        # Historical validation losses are not used, retained, or recomputed by the gate.
        parity_ensemble = {'complete': True, 'members': [{k: m[k] for k in ('config', 'rank', 'seed')}
                                                       for m in ensemble['members']]}
        refs[(fold, day)] = {'keys': keys, 'central': expected, 'ensemble': parity_ensemble}
    return refs, custody


def _bits(values):
    return [f'{int(v):016x}' for v in np.ascontiguousarray(values, dtype='<f8').view('<u8').ravel()]


def compare(expected, actual, expected_keys, actual_keys) -> dict:
    a, b = np.ascontiguousarray(expected, dtype='<f8'), np.asarray(actual)
    keys_equal = expected_keys.equals(actual_keys)
    valid = b.dtype.kind == 'f' and b.dtype.itemsize == 8 and b.ndim == 1 and np.isfinite(b).all()
    converted = np.ascontiguousarray(b, dtype='<f8') if valid else None
    equal = bool(keys_equal and valid and bitwise_equal(a, converted))
    return {'keys_equal': keys_equal, 'bitwise_equal': equal, 'actual_valid_float64': bool(valid),
            'actual_timestamp_utc': [str(k) for k in actual_keys],
            'expected_shape': list(a.shape), 'actual_shape': list(b.shape), 'actual_dtype': str(b.dtype),
            'expected_sha256': array_hash(a), 'actual_sha256': array_hash(converted) if valid else None,
            'expected_float64_bits': _bits(a), 'actual_float64_bits': _bits(converted) if valid else None,
            'max_abs_difference': float(np.max(np.abs(a - converted)))
            if valid and a.shape == converted.shape else None}


def _required():
    return {(path, fold, str(day), name) for path in PATHS for fold, day in SAMPLES for name in (*MEMBERS, *POLICIES)}


def validate_report(report: dict) -> None:
    """A consumer cannot start the full run with a partial, duplicated, or failed gate."""
    rows = report.get('checks', [])
    seen = [(r.get('path'), r.get('fold'), r.get('day'), r.get('name')) for r in rows]
    if report.get('schema') != SCHEMA or report.get('status') != 'PASS' or report.get('settings') != 'old' \
            or report.get('issue_time') != 'D-1 11:00 UTC' or report.get('boundary') != str(H.OLD_BOUNDARY) \
            or len(seen) != len(set(seen)) or set(seen) != _required() \
            or not all(r.get('bitwise_equal') is True and r.get('keys_equal') is True
                       and r.get('actual_valid_float64') is True
                       and r.get('expected_sha256') == r.get('actual_sha256')
                       and r.get('expected_float64_bits') == r.get('actual_float64_bits')
                       and bool(r.get('expected_float64_bits'))
                       and r.get('timestamp_utc') == r.get('actual_timestamp_utc') for r in rows):
        raise H.HarnessError('parity gate is failed, incomplete, or not the required old-settings gate')


def require_pass(root: Path, *, output: Path | None = None) -> dict:
    report = json.loads((Path(root) / (OUT if output is None else output)).read_text())
    validate_report(report)
    current = {name: H.recipe_code_sha256(Path(root), name) for name in report['code_sha256']}
    if any(report['code_sha256'][name] not in (value, sha(Path(root) / name))
           for name, value in current.items()) or report.get('input_fingerprint') != H.input_fingerprint(Path(root)):
        raise H.HarnessError('parity report is stale for the current code or inputs')
    if report.get('reference_custody') != verify_sources(Path(root)):
        raise H.HarnessError('parity references changed after the gate')
    return report


def run(root: Path, *, output: Path | None = None) -> dict:
    """Run all fresh fits serially, save exact evidence, and stop immediately on any mismatch."""
    root = Path(root)
    output = OUT if output is None else Path(output)
    if output.is_absolute() or '..' in output.parts or not output.is_relative_to('reports/rebaseline'):
        raise H.HarnessError('parity output must be a repository-local CP-26 report')
    started = time.perf_counter()
    report = {'schema': SCHEMA, 'status': 'RUNNING', 'written_utc': _stamp(), 'settings': 'old',
              'issue_time': 'D-1 11:00 UTC', 'boundary': str(H.OLD_BOUNDARY),
              'sample_rule': 'first evaluation origin in each old fold 2–5, fixed before fits or scoring',
              'samples': [{'fold': f, 'day': str(d)} for f, d in SAMPLES], 'checks': [],
              'ddnn2_ensembles': 'CP-24 round 1, parity fixtures only; never CP-26 forecasts',
              'bit_representation': 'each float64 as its uint64 IEEE-754 bits, hexadecimal, little endian bytes',
              'fit_records': [], 'workers': 1, 'scores_computed': 0}
    try:
        H.check_blas()
        report['input_fingerprint'] = H.input_fingerprint(root)
        report['candidate_sha'] = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()
        report['code_sha256'] = {name: sha(root / name)
                                 for name in ('src/cp26/harness.py', 'src/cp26/members.py', 'src/cp26/parity.py')}
        refs, report['reference_custody'] = load_references(root)
        for path in PATHS:
            restored = path == 'noload_restored'
            ctx = M.Context.build(root, H.OLD, 'noload' if restored else 'nominal', restore=restored)
            if ctx.settings != H.OLD or ctx.noload_active:
                raise H.HarnessError('old-settings restored-input parity context changed its method')
            for fold, day in SAMPLES:
                ref = refs[(fold, day)]
                keys = ctx.data.index[ctx.data.rows(day)].tz_convert('UTC').as_unit('ns')
                central = {}
                for name in MEMBERS:
                    began = time.perf_counter()
                    out = M.FITTERS[name](ctx, day, ensemble=ref['ensemble'], k=None)
                    if out.get('status') != 'ok':
                        raise H.HarnessError(f'parity fit failed: {path} {fold} {day} {name}')
                    central[name] = np.asarray(out['central'])
                    check = {'path': path, 'fold': fold, 'day': str(day), 'name': name,
                             'timestamp_utc': [str(k) for k in ref['keys']],
                             **compare(ref['central'][name], central[name], ref['keys'], keys)}
                    report['checks'].append(check)
                    report['fit_records'].append({'path': path, 'fold': fold, 'day': str(day), 'member': name,
                                                  'seconds': time.perf_counter() - began,
                                                  'fits': out.get('fits', {}), 'records': out.get('records', []),
                                                  **({'selection': out['selection']} if 'selection' in out else {})})
                    atomic(root / output, report)
                    if not check['bitwise_equal']:
                        raise H.HarnessError(f'bitwise parity mismatch: {path} {fold} {day} {name}')
                for name, actual in assemble(central).items():
                    check = {'path': path, 'fold': fold, 'day': str(day), 'name': name,
                             'timestamp_utc': [str(k) for k in ref['keys']],
                             **compare(ref['central'][name], actual, ref['keys'], keys)}
                    report['checks'].append(check)
                    if not check['bitwise_equal']:
                        raise H.HarnessError(f'bitwise parity mismatch: {path} {fold} {day} {name}')
            del ctx
        report['status'] = 'PASS'
        validate_report(report)
    except Exception as exc:
        report['status'] = 'FAIL'
        report['cause'] = guard.redact(f'{type(exc).__name__}: {exc}')
        report['seconds'] = time.perf_counter() - started
        report['written_utc'] = _stamp()
        atomic(root / output, report)
        raise ParityFailed(report['cause'], report) from None
    report['seconds'] = time.perf_counter() - started
    report['written_utc'] = _stamp()
    atomic(root / output, report)
    return report


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=Path.cwd())
    p.add_argument('--inspect', action='store_true', help='verify committed references only; no fits or gate PASS')
    args = p.parse_args(argv)
    if args.inspect:
        refs, custody = load_references(args.root)
        print(json.dumps({'status': 'REFERENCES_VERIFIED_ONLY', 'samples': len(refs),
                          'references': custody, 'fits_run': 0, 'scores_computed': 0}, sort_keys=True))
        return 0
    try:
        report = run(args.root)
    except ParityFailed as exc:
        print(json.dumps({'status': 'FAIL', 'cause': exc.report['cause'], 'report': str(args.root / OUT)}))
        return 6
    print(json.dumps({'status': report['status'], 'checks': len(report['checks']), 'report': str(args.root / OUT)}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
