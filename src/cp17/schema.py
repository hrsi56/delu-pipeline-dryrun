"""The payload schema and manifest (Space plan §4.12 and §7 step 3), frozen at CP-17 and re-frozen at CP-17b.

CP-17b changes two kinds and nothing else: `issued_forecasts` v2 admits a partial day (its forecastable hours
only, with the missing hours named; the Owner's decision 13), and `fitted_artifact` v2 takes one of two forms,
issued (keys, quantiles, the partial record and its fit's own resource measurement within the ceiling) or a
failed issuance (none of them), so the schema itself refuses an issued artifact without a measurement (CP-17's
review, N2).

CP-17c changes one error path and nothing else (r19 §24.13, H3, fixing §24.12's R7): `verify_bundle` lists a
structurally malformed day of a re-hashed `issued-forecasts.json` as a fault and reads no further into it, where it
used to raise. A record that verifies is judged exactly as before.

Seven payload kinds make up a dated bundle. Each JSON file has a JSON Schema (draft 2020-12, the subset
`validate` implements); each NumPy file declares its arrays in its JSON companion:

| Kind | File | Loaded |
|---|---|---|
| `manifest` | `manifest.json`: every served file with its SHA-256 and kind; the product and policy identity; the daily artifact's identity; the three times; the freshness state | at start |
| `issued_forecasts` | `issued-forecasts.json`: every issued day's keys and quantiles, issue time and issuing artifact | at start |
| `outcomes` | `outcomes.json`: published prices with source, publication and revision times | at start |
| `evaluation` | `evaluation.json`: the committed metric records the recomputation must match (`cp17.scoring`) | at start |
| `fitted_artifact` | `artifact.json` (`cp17.issue.artifact`) and `artifact-arrays.npz` (the fitted parameters) | on request |
| `training_window` | `<policy>-<day>.json` and `.npz`: the fit's shipped payload (`cp17.dailyfit`) | on "Train it yourself" |
| `series` | `series.json`: hourly prices for the Data view | for Data |

**The embargo** is part of the schema: a training window never carries its target day's outcome
(`check_training_window`), and every released-error buffer ends at D-2 or earlier.

`verify_bundle` checks a bundle as the Space does at start: the manifest validates, every listed file exists
and hashes as listed, no unlisted file is served, every JSON file validates against its kind's schema, and
the cross-references hold, and the required kinds (issued forecasts, outcomes, evaluation records and the
artifact the manifest names) are all served. Any failure is listed, never raised; a tampered file fails.
"""
from __future__ import annotations

from datetime import date, timedelta
import hashlib
import json
import re

SHA = {'type': 'string', 'pattern': '^[0-9a-f]{64}$'}
DAY = {'type': 'string', 'pattern': r'^\d{4}-\d{2}-\d{2}$'}
UTC = {'type': 'string', 'pattern': r'^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d+)?\+00:00$'}
UTC_OR_EMPTY = {'type': 'string', 'pattern': r'^(|\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d+)?\+00:00)$'}
POLICY = {'enum': ['A1', 'v4', 'v2', 'B0']}
#: One to 25 hours: a partial day issues its forecastable hours only (decision 13).
QUANTILES = {'type': 'array', 'minItems': 1, 'maxItems': 25,
             'items': {'type': 'array', 'minItems': 7, 'maxItems': 7, 'items': {'type': 'number'}}}
KEYS = {'type': 'array', 'minItems': 1, 'maxItems': 25, 'items': UTC}
#: An issued fit's own resource measurement on the issuing host, within the success definition's ceiling (N2);
#: a failed issuance has none.
MEASUREMENT = {'type': 'object', 'required': ['wall_seconds', 'peak_bytes', 'limits'],
               'properties': {'wall_seconds': {'type': 'number', 'minimum': 0, 'maximum': 900},
                              'peak_bytes': {'type': 'integer', 'minimum': 0, 'maximum': 4 * 1024 ** 3},
                              'limits': {'type': 'object', 'required': ['wall_seconds', 'peak_bytes'], 'additionalProperties': False,
                                         'properties': {'wall_seconds': {'const': 900}, 'peak_bytes': {'const': 4 * 1024 ** 3}}}}}
PARTIAL = {'type': 'object', 'required': ['issued_hours', 'canonical_hours', 'missing_hours'], 'additionalProperties': False,
           'properties': {'issued_hours': {'type': 'integer', 'minimum': 1}, 'canonical_hours': {'type': 'integer'},
                          'missing_hours': {'type': 'array', 'minItems': 1, 'items': {'type': 'string'}}}}
#: The two forms of an issued artifact: a policy issued (keys, quantiles and its fit's measurement), or a failed
#: issuance (none of them). The schema itself refuses an issued artifact without a measurement (N2).
ISSUED_FORM = {'properties': {'issued_policy': POLICY, 'keys': KEYS, 'quantiles': QUANTILES, 'quantiles_sha256': SHA,
                              'resources': MEASUREMENT, 'partial': {'anyOf': [PARTIAL, {'type': 'null'}]}}}
FAILED_FORM = {'properties': {'issued_policy': {'type': 'null'}, 'keys': {'type': 'null'}, 'quantiles': {'type': 'null'},
                              'quantiles_sha256': {'type': 'null'}, 'resources': {'type': 'null'}, 'partial': {'type': 'null'}}}
ARTIFACT_SCHEMA = 'cp17b-issued-artifact-v2'
ISSUED_FORECASTS_SCHEMA = 'cp17b-issued-forecasts-v2'
KINDS = ('manifest', 'issued_forecasts', 'outcomes', 'evaluation', 'fitted_artifact', 'training_window', 'series')

SCHEMAS = {
    'manifest': {
        '$schema': 'https://json-schema.org/draft/2020-12/schema', '$id': 'cp17-manifest-v1', 'type': 'object',
        'required': ['schema', 'product', 'policy', 'artifact', 'times', 'freshness', 'files'],
        'additionalProperties': False,
        'properties': {
            'schema': {'const': 'cp17-manifest-v1'},
            'product': {'type': 'object', 'required': ['name', 'designated_policy', 'evidence_label'],
                        'additionalProperties': False,
                        'properties': {'name': {'const': 'delu-day-ahead-policy'}, 'designated_policy': {'const': 'A1'},
                                       'evidence_label': {'type': 'string'}}},
            'policy': {'type': 'object', 'required': ['policy_identity_sha256', 'package_sha256', 'issuing_host'],
                       'additionalProperties': False,
                       'properties': {'policy_identity_sha256': SHA, 'package_sha256': SHA,
                                      'issuing_host': {'type': 'string'}}},
            'artifact': {'type': 'object', 'required': ['artifact_sha256', 'delivery_day', 'issued_policy', 'display'],
                         'additionalProperties': False,
                         'properties': {'artifact_sha256': SHA, 'delivery_day': DAY,
                                        'issued_policy': {'anyOf': [POLICY, {'type': 'null'}]},
                                        'display': {'type': 'string'}}},
            'times': {'type': 'object', 'required': ['last_training_utc', 'data_refresh_utc', 'issuance_utc'],
                      'additionalProperties': False,
                      'properties': {'last_training_utc': UTC, 'data_refresh_utc': UTC_OR_EMPTY, 'issuance_utc': UTC}},
            'freshness': {'type': 'object', 'required': ['state', 'as_of_day', 'latest_issued_day'],
                          'additionalProperties': False,
                          'properties': {'state': {'enum': ['fresh', 'stale', 'failed']}, 'as_of_day': DAY,
                                         'latest_issued_day': DAY}},
            'files': {'type': 'array', 'minItems': 1, 'items': {
                'type': 'object', 'required': ['path', 'sha256', 'bytes', 'kind'], 'additionalProperties': False,
                'properties': {'path': {'type': 'string', 'pattern': r'^[A-Za-z0-9._-]+$'}, 'sha256': SHA,
                               'bytes': {'type': 'integer', 'minimum': 0}, 'kind': {'enum': list(KINDS[1:])}}}}}},
    'issued_forecasts': {
        '$schema': 'https://json-schema.org/draft/2020-12/schema', '$id': ISSUED_FORECASTS_SCHEMA, 'type': 'object',
        'required': ['schema', 'levels', 'days'], 'additionalProperties': False,
        'properties': {
            'schema': {'const': ISSUED_FORECASTS_SCHEMA},
            'levels': {'const': [0.025, 0.1, 0.25, 0.5, 0.75, 0.9, 0.975]},
            'days': {'type': 'array', 'items': {
                'type': 'object', 'additionalProperties': False,
                'required': ['delivery_day', 'issue_utc', 'issuing_artifact_sha256', 'issued_policy', 'display', 'keys',
                             'quantiles', 'partial'],
                'properties': {'delivery_day': DAY, 'issue_utc': UTC, 'issuing_artifact_sha256': SHA,
                               'issued_policy': POLICY, 'display': {'type': 'string'}, 'keys': KEYS,
                               'quantiles': QUANTILES, 'partial': {'anyOf': [PARTIAL, {'type': 'null'}]}}}}}},
    'outcomes': {
        '$schema': 'https://json-schema.org/draft/2020-12/schema', '$id': 'cp17-outcomes-v1', 'type': 'object',
        'required': ['schema', 'unit', 'prices'], 'additionalProperties': False,
        'properties': {
            'schema': {'const': 'cp17-outcomes-v1'}, 'unit': {'const': 'EUR/MWh'},
            'prices': {'type': 'array', 'items': {
                'type': 'object', 'additionalProperties': False,
                'required': ['timestamp_utc', 'delivery_day', 'price', 'source', 'published_utc', 'revised_utc'],
                'properties': {'timestamp_utc': UTC, 'delivery_day': DAY, 'price': {'type': ['number', 'null']},
                               'source': {'type': 'string'}, 'published_utc': UTC_OR_EMPTY,
                               'revised_utc': UTC_OR_EMPTY}}}}},
    'evaluation': {
        '$schema': 'https://json-schema.org/draft/2020-12/schema', '$id': 'cp17-evaluation-bundle-v1', 'type': 'object',
        'required': ['schema', 'windows'], 'additionalProperties': False,
        'properties': {
            'schema': {'const': 'cp17-evaluation-bundle-v1'},
            'windows': {'type': 'array', 'items': {
                'type': 'object', 'required': ['window', 'evidence_class', 'policy', 'record'], 'additionalProperties': False,
                'properties': {'window': {'enum': ['development', '4.7T', 'prospective']},
                               'evidence_class': {'type': 'string'}, 'policy': {'enum': ['product', 'B0']},
                               'record': {'type': 'object', 'required': ['schema', 'tau_eur', 'rolling_days', 'pooled',
                                                                         'days', 'rolling'],
                                          'properties': {'schema': {'const': 'cp17-evaluation-v1'},
                                                         'tau_eur': {'const': 10.0}, 'rolling_days': {'const': 28}}}}}}}},
    'fitted_artifact': {
        '$schema': 'https://json-schema.org/draft/2020-12/schema', '$id': ARTIFACT_SCHEMA, 'type': 'object',
        'required': ['schema', 'product', 'designated_policy', 'delivery_day', 'issue_utc', 'issued_policy', 'route',
                     'display', 'attempts', 'keys', 'quantiles', 'quantiles_sha256', 'times', 'lineage', 'resources',
                     'partial', 'artifact_sha256'],
        'anyOf': [ISSUED_FORM, FAILED_FORM],
        'properties': {'schema': {'const': ARTIFACT_SCHEMA}, 'product': {'const': 'delu-day-ahead-policy'},
                       'designated_policy': {'const': 'A1'}, 'delivery_day': DAY, 'issue_utc': UTC,
                       'issued_policy': {'anyOf': [POLICY, {'type': 'null'}]},
                       'route': {'enum': ['designated', 'gfs_unavailable', 'a1_failed']},
                       'attempts': {'type': 'array', 'minItems': 1, 'items': {
                           'type': 'object', 'required': ['policy', 'status'],
                           'properties': {'policy': POLICY, 'status': {'enum': ['ok', 'failed', 'skipped']}}}},
                       'keys': {'anyOf': [KEYS, {'type': 'null'}]}, 'quantiles': {'anyOf': [QUANTILES, {'type': 'null'}]},
                       'resources': {'anyOf': [MEASUREMENT, {'type': 'null'}]},
                       'partial': {'anyOf': [PARTIAL, {'type': 'null'}]},
                       'artifact_sha256': SHA,
                       'lineage': {'type': 'object', 'required': ['parent_artifact_sha256', 'policy_identity_sha256',
                                                                  'code_sha256', 'host', 'inputs', 'training', 'state']}}},
    'training_window': {
        '$schema': 'https://json-schema.org/draft/2020-12/schema', '$id': 'cp17-training-window-v1', 'type': 'object',
        'required': ['schema', 'policy', 'day', 'embargo', 'array_sha256', 'payload_sha256'],
        'properties': {'schema': {'enum': ['cp17-a1-payload-v1', 'cp17-v2-payload-v1', 'cp17-v4-payload-v1',
                                           'cp17-b0-payload-v1']},
                       'policy': POLICY, 'day': DAY, 'embargo': {'type': 'string'},
                       'array_sha256': {'type': 'object', 'additionalProperties': SHA}, 'payload_sha256': SHA}},
    'series': {
        '$schema': 'https://json-schema.org/draft/2020-12/schema', '$id': 'cp17-series-v1', 'type': 'object',
        'required': ['schema', 'unit', 'first_day', 'last_day', 'timestamp_utc', 'price'], 'additionalProperties': False,
        'properties': {'schema': {'const': 'cp17-series-v1'}, 'unit': {'const': 'EUR/MWh'}, 'first_day': DAY,
                       'last_day': DAY, 'timestamp_utc': {'type': 'array', 'items': UTC},
                       'price': {'type': 'array', 'items': {'type': ['number', 'null']}}}},
}
#: One issued day, as `issued_forecasts` lists it; `verify_bundle` reads no further into a day that fails it (R7).
DAY_ITEM = SCHEMAS['issued_forecasts']['properties']['days']['items']
FILE_KIND = {'manifest.json': 'manifest', 'issued-forecasts.json': 'issued_forecasts', 'outcomes.json': 'outcomes',
             'evaluation.json': 'evaluation', 'artifact.json': 'fitted_artifact', 'artifact-arrays.npz': 'fitted_artifact',
             'series.json': 'series'}
TRAINING_FILE = re.compile(r'^(A1|v4|v2|B0)-(\d{4}-\d{2}-\d{2})\.(json|npz)$')
#: What every bundle serves: the files the Space loads at start, and the artifact the manifest names.
REQUIRED = ('issued-forecasts.json', 'outcomes.json', 'evaluation.json', 'artifact.json')

_TYPES = {'object': dict, 'array': list, 'string': str, 'boolean': bool, 'null': type(None)}


def _is(value, name: str) -> bool:
    if name == 'integer':
        return isinstance(value, int) and not isinstance(value, bool)
    if name == 'number':
        return isinstance(value, (int, float)) and not isinstance(value, bool) and value == value \
            and value not in (float('inf'), float('-inf'))
    return isinstance(value, _TYPES[name])


def validate(value, schema: dict, path: str = '$') -> list[str]:
    """The JSON Schema subset these schemas use: type, const, enum, anyOf, required, properties,
    additionalProperties, items, minItems, maxItems, pattern, minimum, maximum. Returns every violation."""
    errors = []
    if 'type' in schema:
        names = schema['type'] if isinstance(schema['type'], list) else [schema['type']]
        if not any(_is(value, n) for n in names):
            return [f'{path}: expected {"/".join(names)}']
    if 'const' in schema and value != schema['const']:
        errors.append(f'{path}: expected {schema["const"]!r}')
    if 'enum' in schema and value not in schema['enum']:
        errors.append(f'{path}: {value!r} not in {schema["enum"]}')
    if 'anyOf' in schema and not any(not validate(value, s, path) for s in schema['anyOf']):
        errors.append(f'{path}: matches none of the allowed forms')
    if isinstance(value, dict):
        for key in schema.get('required', ()):
            if key not in value:
                errors.append(f'{path}: missing {key}')
        props = schema.get('properties', {})
        extra = schema.get('additionalProperties', True)
        for key, item in value.items():
            if key in props:
                errors += validate(item, props[key], f'{path}.{key}')
            elif extra is False:
                errors.append(f'{path}: unexpected {key}')
            elif isinstance(extra, dict):
                errors += validate(item, extra, f'{path}.{key}')
    if isinstance(value, list):
        if 'minItems' in schema and len(value) < schema['minItems']:
            errors.append(f'{path}: fewer than {schema["minItems"]} items')
        if 'maxItems' in schema and len(value) > schema['maxItems']:
            errors.append(f'{path}: more than {schema["maxItems"]} items')
        if 'items' in schema:
            for i, item in enumerate(value):
                errors += validate(item, schema['items'], f'{path}[{i}]')
    if isinstance(value, str) and 'pattern' in schema and not re.search(schema['pattern'], value):
        errors.append(f'{path}: does not match {schema["pattern"]}')
    if _is(value, 'number') and 'minimum' in schema and value < schema['minimum']:
        errors.append(f'{path}: below {schema["minimum"]}')
    if _is(value, 'number') and 'maximum' in schema and value > schema['maximum']:
        errors.append(f'{path}: above {schema["maximum"]}')
    return errors


def check_training_window(meta: dict, arrays: dict) -> list[str]:
    """The embargo and the payload's own identity, on a training-window payload."""
    import numpy as np
    from .dailyfit import array_hash, payload_digest
    errors = validate(meta, SCHEMAS['training_window'], meta.get('policy', '?'))
    if errors:
        return errors
    day = date.fromisoformat(meta['day'])
    body = {k: v for k, v in meta.items() if k != 'payload_sha256'}
    if payload_digest(arrays, body) != meta['payload_sha256']:
        errors.append(f"{meta['policy']}: the payload differs from its recorded identity")
    if sorted(arrays) != sorted(meta['array_sha256']) or any(array_hash(arrays[k]) != h for k, h in meta['array_sha256'].items()):
        errors.append(f"{meta['policy']}: the arrays differ from their declared hashes")
    epoch = (day - date(1970, 1, 1)).days
    policy = meta['policy']
    if policy == 'A1' and (np.isfinite(arrays['price'][-1]).any() or np.isfinite(arrays['y'][-1]).any()
                           or arrays['mask'][-1].any() or int(arrays['days'][-1]) != epoch):
        errors.append("A1: the target day's outcome is present")
    if policy in ('v2', 'v4') and np.isfinite(arrays['y'][arrays['dates'] == epoch]).any():
        errors.append(f"{policy}: the target day's outcome is present")
    if policy == 'B0' and np.isfinite(arrays['price'][np.isin(arrays['price_ns'], arrays['key_ns'])]).any():
        errors.append("B0: the target day's prices are present")
    for key in ('h_buffer', 'b0_buffer'):
        if any(date.fromisoformat(b['day']) > day - timedelta(days=2) for b in meta.get(key, ())):
            errors.append(f'{policy}: {key} holds a day not released under D-2')
    return errors


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def verify_bundle(files: dict[str, bytes]) -> list[str]:
    """A dated bundle, as {path: bytes}. Returns every failure; empty when the bundle verifies."""
    import io
    import numpy as np
    from .issue import verify_artifact
    if 'manifest.json' not in files:
        return ['no manifest.json']
    manifest = json.loads(files['manifest.json'])
    errors = validate(manifest, SCHEMAS['manifest'], 'manifest')
    if errors:
        return errors
    listed = {f['path']: f for f in manifest['files']}
    for path in REQUIRED:
        if path not in listed:
            errors.append(f'{path}: required, but not listed in the manifest')
    for path in sorted(set(files) - set(listed) - {'manifest.json'}):
        errors.append(f'{path}: served but not listed in the manifest')
    loaded, verified = {}, set()
    for path, item in listed.items():
        if path not in files:
            errors.append(f'{path}: listed but absent')
            continue
        if sha256(files[path]) != item['sha256'] or len(files[path]) != item['bytes']:
            errors.append(f'{path}: its SHA-256 or size differs from the manifest')
            continue
        kind = FILE_KIND.get(path) or ('training_window' if TRAINING_FILE.match(path) else None)
        if kind != item['kind']:
            errors.append(f'{path}: listed as {item["kind"]}, expected {kind}')
            continue
        verified.add(path)
        if path.endswith('.json'):
            try:
                value = json.loads(files[path])
            except ValueError as exc:
                errors.append(f'{path}: not readable JSON ({exc})')
                continue
            errors += validate(value, SCHEMAS[kind], path)
            loaded[path] = value
    for path in [p for p in listed if TRAINING_FILE.match(p) and p.endswith('.json') and p in loaded]:
        npz = path[:-5] + '.npz'
        arrays = {}
        if npz in listed and npz not in verified:
            continue          # already refused by its hash; a file that failed it is never opened
        if npz in verified:
            try:
                with np.load(io.BytesIO(files[npz]), allow_pickle=False) as z:
                    arrays = {k: z[k] for k in z.files}
            except Exception as exc:   # an unreadable archive is a refusal, never a crash
                errors.append(f'{npz}: not readable ({type(exc).__name__})')
                continue
        errors += check_training_window(loaded[path], arrays)
    from .issue import partial
    issued = loaded.get('issued-forecasts.json', {})
    forecasts = []           # the well-formed days; a malformed one is listed and never read further (R7)
    for i, d in enumerate(issued.get('days', []) if isinstance(issued, dict) and isinstance(issued.get('days'), list)
                          else []):
        try:
            if validate(d, DAY_ITEM):
                raise ValueError('not a well-formed day')
            expected = partial(date.fromisoformat(d['delivery_day']), d['keys'])
        except (ValueError, TypeError, KeyError, AttributeError):
            errors.append(f'issued-forecasts.json: day {i} is malformed, so it is listed and not read (R7)')
            continue
        forecasts.append(d)
        if d.get('partial') != expected:
            errors.append(f'issued-forecasts.json: day {i} does not name exactly its missing hours (decision 13)')
    artifact = loaded.get('artifact.json')
    if artifact is not None:
        errors += [f'artifact.json: {e}' for e in verify_artifact(artifact)]
        if artifact['artifact_sha256'] != manifest['artifact']['artifact_sha256']:
            errors.append('the manifest names another artifact')
        same = [d for d in forecasts if d['delivery_day'] == artifact['delivery_day']]
        if artifact['issued_policy'] and (len(same) != 1 or same[0]['issuing_artifact_sha256'] != artifact['artifact_sha256']
                                          or same[0]['quantiles'] != artifact['quantiles']):
            errors.append("the issued forecast of the artifact's day is not the artifact's")
    return errors
