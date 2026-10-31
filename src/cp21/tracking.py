"""CP-21 local MLflow tracking (capstone v21-r6 §17.9): experiment `delu-generations` in
`.local/mlruns/cp21` only -- parent `cp21` and one child per new policy, with the names, tags,
parameters, metric histories and artifacts of the committed draft export.

Local only: the tracking URI is a SQLite file under `.local/mlruns/cp21`, never an environment
value; MLflow telemetry must be disabled (MLFLOW_DISABLE_TELEMETRY=true and DO_NOT_TRACK=true) or
the job refuses to start; nothing is uploaded and no network call is made. Idempotent by
`delu.run_key`; every history and artifact is read back and compared with the export.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from .budget import atomic
from .execution import OUT
from .jobs import _project, stamp

DRAFT = OUT / 'mlflow-export-draft' / 'cp21.json'


def store() -> Path:
    return _project() / '.local' / 'mlruns' / 'cp21'


def _client():
    if os.environ.get('MLFLOW_DISABLE_TELEMETRY', '').lower() != 'true' or os.environ.get('DO_NOT_TRACK', '').lower() != 'true':
        raise SystemExit('refused: set MLFLOW_DISABLE_TELEMETRY=true and DO_NOT_TRACK=true (no network call is permitted)')
    os.environ.pop('MLFLOW_TRACKING_URI', None)
    import mlflow
    from mlflow.tracking import MlflowClient
    store().mkdir(parents=True, exist_ok=True)
    uri = f'sqlite:///{store() / "mlflow.db"}'
    mlflow.set_tracking_uri(uri)
    return MlflowClient(tracking_uri=uri), uri


def job_mlflow_local(root: Path, rest) -> int:
    from mlflow.entities import Metric, Param, RunTag
    export = json.loads((Path(root) / DRAFT).read_text())
    client, uri = _client()
    experiment = client.get_experiment_by_name(export['experiment'])
    if experiment is None:
        experiment_id = client.create_experiment(export['experiment'], artifact_location=(store() / 'artifacts').as_uri(),
                                                 tags={'delu.local_draft': 'CP-21 local tracking of the draft export; never uploaded'})
    else:
        experiment_id = experiment.experiment_id
    ids, verified = {}, {}
    for run in export['runs']:  # parent first
        found = client.search_runs([experiment_id], filter_string=f"tags.`delu.run_key` = '{run['run_key']}'")
        if len(found) > 1:
            raise ValueError(f'two local runs carry {run["run_key"]}')
        if found and found[0].data.tags.get('delu.local_state') == 'complete':
            ids[run['run_key']] = found[0].info.run_id
        else:
            if found:
                client.delete_run(found[0].info.run_id)  # an incomplete local run is rebuilt, never merged
            tags = dict(run['tags'])
            if run['parent']:
                tags['mlflow.parentRunId'] = ids[run['parent']]
            created = client.create_run(experiment_id, run_name=run['run_name'], tags=tags)
            rid = created.info.run_id
            client.log_batch(rid, params=[Param(k, str(v)) for k, v in run['params'].items()])
            points = [Metric(key, p['value'], p['timestamp'], p['step']) for key, hist in run['metrics'].items() for p in hist]
            for i in range(0, len(points), 900):
                client.log_batch(rid, metrics=points[i:i + 900])
            for artifact in run['artifacts']:
                client.log_text(rid, artifact['content'], artifact['path'])
            client.set_tag(rid, 'delu.local_state', 'complete')
            client.set_terminated(rid)
            ids[run['run_key']] = rid
        rid = ids[run['run_key']]
        got = client.get_run(rid)
        histories = {key: [(m.step, m.timestamp, m.value) for m in sorted(client.get_metric_history(rid, key), key=lambda m: (m.step, m.timestamp))]
                     for key in run['metrics']}
        expected = {key: sorted((p['step'], p['timestamp'], p['value']) for p in hist) for key, hist in run['metrics'].items()}
        texts = {a['path']: client.download_artifacts(rid, a['path'], str(store() / 'readback' / rid)) for a in run['artifacts']}
        artifact_ok = all(hashlib.sha256(Path(texts[a['path']]).read_bytes()).hexdigest() == a['sha256'] for a in run['artifacts'])
        verified[run['run_key']] = {
            'run_id': rid, 'run_name_equal': got.info.run_name == run['run_name'],
            'tags_equal': all(got.data.tags.get(k) == v for k, v in run['tags'].items()),
            'params_equal': got.data.params == {k: str(v) for k, v in run['params'].items()},
            'histories_equal': histories == expected, 'artifacts_equal': artifact_ok,
            'parent_equal': (got.data.tags.get('mlflow.parentRunId') == ids.get(run['parent'])) if run['parent'] else True}
    ok = all(all(v for k, v in rec.items() if k != 'run_id') for rec in verified.values())
    record = {'schema': 'cp21-mlflow-local-v1', 'written_utc': stamp(), 'tracking': 'local SQLite store .local/mlruns/cp21 (ignored)',
              'experiment': export['experiment'], 'runs': verified, 'all_read_back_equal': ok,
              'network': 'none; telemetry disabled; no upload', 'draft_export_sha256': hashlib.sha256((Path(root) / DRAFT).read_bytes()).hexdigest()}
    atomic(Path(root) / OUT / 'mlflow-local.json', record)
    print(json.dumps({'all_read_back_equal': ok, 'runs': {k: v['run_id'] for k, v in verified.items()}}), flush=True)
    return 0 if ok else 9
