"""CP-24's draft MLflow export (capstone v21-r11 §23.12): `reports/ddnn2/mlflow-export-draft/cp24.json`.

Built by this CP-24 module by importing `scripts/mlflow_export.py`'s functions unchanged -- the draft-entry
reader's types, `DraftBuilder`, the evidence layer's record builders, run naming, statuses, descriptions,
datasets, metric units, artifacts and the outbound-secret scan -- with CP-24's paths: the scored attempts'
committed rows under `reports/ddnn2/attempt-<k>/` and the packet's draft entries
(`reports/ddnn2/draft-registry.json`). Nothing is uploaded; the published export set
(`reports/presentation/mlflow-export/`) is not touched. Run as ``python -m cp24.export`` (``--check``
verifies the committed file).
"""
from __future__ import annotations

import dataclasses
import importlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
DRAFT_DIR = ROOT / 'reports' / 'ddnn2' / 'mlflow-export-draft'
DRAFT_REGISTRY = 'reports/ddnn2/draft-registry.json'


def _module():
    for p in (str(ROOT / 'src'), str(ROOT / 'scripts')):
        if p not in sys.path:
            sys.path.insert(0, p)
    return importlib.import_module('mlflow_export')


def paths(k: int) -> dict:
    base = f'reports/ddnn2/attempt-{k}'
    return {name: f'{base}/{file}' for name, file in (('metrics', 'metrics.csv'), ('uncertainty', 'uncertainty.csv'),
                                                      ('criteria', 'criteria.csv'), ('diagnostics', 'diagnostics.csv'),
                                                      ('protocol', 'protocol.json'), ('decisions', 'decisions.json'))}


#: Each CP-24 run's contrasts (§23.8): (candidate, base code, metric slug).
CONTRASTS = {
    'v5': (('v5', 'HGL', 'v4'), ('v5', 'HG', 'v3'), ('v5', 'v3+D2', 'v3_d2')),
    'D2': (('D2', 'HGL', 'v4'), ('D2', 'HG', 'v3'), ('D2', 'D', 'd')),
    'v3+D2': (('v3+D2', 'HG', 'v3'), ('v3+D2', 'HGL', 'v4')),
}


def draft_registry():
    X = _module()
    G = X.G
    spec = json.loads((ROOT / DRAFT_REGISTRY).read_text())

    def entry(raw: dict):
        fields = dict(raw)
        fields['codes'] = tuple(G.Code(**code) for code in raw['codes'])
        fields['statuses'] = tuple(G.StatusEvent(**event) for event in raw['statuses'])
        for key in ('rules', 'sources', 'run_keys'):
            fields[key] = tuple(raw[key])
        return G.Entry(**fields)

    entries = {raw['id']: entry(raw) for raw in spec['entries']}
    checkpoint = G.Checkpoint(**{**spec['checkpoint'], 'children': tuple(spec['checkpoint']['children'])})
    return spec, entries, checkpoint


def records(k: int) -> dict:
    X = _module()
    R = X.R
    P = paths(k)
    settings = json.loads((ROOT / P['protocol']).read_text())
    interval = R.Interval(method='paired noncircular moving-block percentile bootstrap', level=0.95, seed=15042,
                          replicates=2000, block_days=7)
    windows = R._fold_windows(P['metrics'])
    built = (R._metric_records('cp24', 'CP-24', P['metrics'])
             + [dataclasses.replace(r, interval=interval) for r in R._uncertainty_records('cp24', 'CP-24', P['uncertainty'], windows)]
             + R._criteria_records('cp24', 'CP-24', P['criteria'], windows)
             + R._diagnostic_records('cp24', 'CP-24', P['diagnostics']))
    for line, row in R._rows(P['uncertainty']):
        if row['scope'] != 'equal_fold':
            continue
        built.append(R.EvidenceRecord(
            record_id=f"cp24.ratio.{row['candidate']}-{row['baseline']}.{row['metric']}", checkpoint='CP-24',
            generation=None, policy_code=row['candidate'], source_path=P['uncertainty'],
            selector=(('scope', 'equal_fold'), ('candidate', row['candidate']), ('baseline', row['baseline']),
                      ('metric', row['metric'])),
            field='ratio', metric=f"ratio_S_{row['metric']}", unit=X.UNIT_RATIO_CHANGE, aggregation='equal_fold_ratio',
            population_id='common-10747h', comparator=row['baseline'], evidence_class='development_post_selection',
            window=windows['all'], display_precision=4, raw=row['ratio'], source_line=line, interval=interval,
            ci_low_raw=row['ratio_ci_lower'], ci_high_raw=row['ratio_ci_upper']))
    out = {}
    for record in built:
        if record.record_id in out:
            raise R.EvidenceError(f'duplicate draft record id {record.record_id}')
        out[record.record_id] = record
    del settings
    return out


def scores(builder, k: int, code: str) -> None:
    equal, pooled = f'cp24.metrics.{code}.equal_fold', f'cp24.metrics.{code}.pooled'
    fold, peak = f'cp24.metrics.{code}.{{fold}}', f'cp24.diagnostics.{code}.peak'
    builder.single('s_mae', f'{equal}.S_MAE')
    builder.single('s_wis', f'{equal}.S_WIS')
    for key, column in (('pooled_mae_eur', 'MAE'), ('pooled_wis_eur', 'WIS'), ('pooled_rmse_eur', 'RMSE'),
                        ('pooled_bias_eur', 'bias'), ('pooled_coverage95', 'coverage95')):
        builder.single(key, f'{pooled}.{column}')
    for key, column in (('fold_mae_eur', 'MAE'), ('fold_wis_eur', 'WIS'), ('fold_coverage50', 'coverage50'),
                        ('fold_coverage80', 'coverage80'), ('fold_coverage95', 'coverage95'),
                        ('fold_mean_width95_eur', 'mean_width95')):
        builder.per_fold(key, f'{fold}.{column}')
    for key, column in (('peak_mae_eur', 'MAE'), ('peak_wis_eur', 'WIS'), ('peak_coverage95', 'coverage95'),
                        ('peak_hits95', 'hit_count95')):
        builder.single(key, f'{peak}.{column}')
    builder.daily(paths(k)['diagnostics'], code)
    for candidate, base, slug in CONTRASTS[code]:
        prefix = f'cp24.uncertainty.{candidate}-{base}'
        for metric, score in (('MAE', 'mae'), ('WIS', 'wis')):
            eq = f'{prefix}.equal_fold.{metric}'
            builder.single(f'delta_s_{score}_vs_{slug}', eq)
            builder.single(f'delta_s_{score}_vs_{slug}_ci_low', eq, 'ci_low')
            builder.single(f'delta_s_{score}_vs_{slug}_ci_high', eq, 'ci_high')
            ratio = f'cp24.ratio.{candidate}-{base}.{metric}'
            builder.single(f'ratio_s_{score}_vs_{slug}', ratio)
            builder.single(f'ratio_s_{score}_vs_{slug}_ci_low', ratio, 'ci_low')
            builder.single(f'ratio_s_{score}_vs_{slug}_ci_high', ratio, 'ci_high')
            fp = f'{prefix}.{{fold}}.{metric}'
            builder.per_fold(f'delta_fold_{score}_eur_vs_{slug}', fp)
            builder.per_fold(f'delta_fold_{score}_eur_vs_{slug}_ci_low', fp, 'ci_low')
            builder.per_fold(f'delta_fold_{score}_eur_vs_{slug}_ci_high', fp, 'ci_high')


def params(k: int, code: str | None) -> dict:
    X = _module()
    P = paths(k)
    protocol = json.loads((ROOT / P['protocol']).read_text())
    cp15 = json.loads((ROOT / 'reports/cp15/protocol.json').read_text())
    out = {'anchor_version': 'capstone_v21.md v21-r11 §23', 'protocol_sha256': X._file_sha256(P['protocol']),
           'scored_attempt': str(k)}
    if code is None:
        return out
    arm = protocol['arms'][code]
    out['quantile_set'] = ','.join(str(level) for level in cp15['quantiles']['levels'])
    out['policy_definition'] = f"{arm['role']}; central {arm['central']}"
    out['interval_method'] = arm['layer']
    out['member_weight'] = protocol['weight']
    out['ddnn2'] = ('day-level feed-forward network, 24 Johnson SU heads; ' + protocol['training_recipe']['loss'])
    out['ddnn2_search'] = (f"{protocol['search']['trials_per_fold']} trials per fold; " + protocol['search']['halving'] + '; '
                           + protocol['search']['ranking_metric'])
    out['ddnn2_ensemble'] = protocol['training_recipe']['ensemble']
    out['history_window'] = protocol['attempt_fits']['window']
    out['weather_features'] = "v4's three GFS columns plus missing indicators (an optional input group)"
    return {key: value[:500] for key, value in out.items()}


def readme(X, run_name, checkpoint, spec, blobs) -> str:
    tag = checkpoint.evidence_tag
    lines = [f'# {run_name}', '',
             f"Tracked by CP-24 from its committed evidence (draft export; evidence tag `{tag}` and its commit are "
             f"{spec['pending']}). Development evidence; nothing here is a live or confirmatory result.", '',
             f'- Report: {X.GITHUB_URL}/blob/{tag}/{checkpoint.report}',
             f'- Independent Integration review: {X.GITHUB_URL}/blob/{tag}/{checkpoint.verdict}',
             f'- Landing record: {checkpoint.landing}', f'- Presentation: {X.PAGES_URL}', '',
             'Source rows at the evidence tag:', '']
    lines += [f'- {X.GITHUB_URL}/blob/{tag}/{path} (blob {blob})' for path, blob in blobs.items()]
    return '\n'.join(lines) + '\n'


def runs(k: int) -> list[dict]:
    X = _module()
    G, R = X.G, X.R
    spec, entries, checkpoint = draft_registry()
    P = paths(k)
    recs = records(k)
    group = X.comparability()['common']
    pending, note = spec['pending'], spec['note']
    owner = entries[checkpoint.owner]
    parent_tags = {
        'delu.run_key': checkpoint.run_key, 'delu.checkpoint': checkpoint.code, 'delu.registry_id': owner.id,
        'delu.kind': owner.kind, 'delu.public_name': owner.name, 'delu.status': X._draft_status_text(owner, pending),
        'delu.role': 'checkpoint', 'delu.evidence_class': 'development_post_selection',
        'delu.population_id': group['population_id'], 'delu.comparability_id': group['comparability_id'],
        'delu.model_code_sha': spec['model_code_sha'], 'delu.evidence_ref': spec['evidence_ref'],
        'delu.backfilled': 'false', 'delu.original_completed_utc': pending,
        'delu.children': ','.join(f'{checkpoint.run_key}/{code}' for code in checkpoint.children),
        'mlflow.note.content': X._draft_description(owner, note, pending),
    }
    parent = {'run_key': checkpoint.run_key, 'parent': None,
              'run_name': X.draft_run_name(checkpoint.run_key, entries, checkpoint),
              'params': params(k, None), 'tags': parent_tags, 'metrics': {}, 'metric_provenance': {},
              'metric_units': {}, 'inputs': [], 'comparability': dict(group)}
    parent['artifacts'] = [
        X._artifact('summary.json', X._canonical({x: v for x, v in parent.items() if x != 'metric_provenance'}) + '\n'),
        X._artifact('README.md', readme(X, parent['run_name'], checkpoint, spec,
                                        {P['protocol']: R.blob_sha(R.source_bytes(P['protocol']))})),
    ]
    out = [parent]
    for code in checkpoint.children:
        run_key = f'{checkpoint.run_key}/{code}'
        entry = X._draft_entry_for(run_key, entries)
        builder = X.DraftBuilder(recs)
        scores(builder, k, code)
        blobs = X._draft_blobs(builder.provenance, recs)
        run_name = X.draft_run_name(run_key, entries, checkpoint)
        comparator = (G.get(entry.comparator).name if entry.comparator in {e.id for e in G.entries()}
                      else entries[entry.comparator].name)
        tags = {
            'delu.run_key': run_key, 'delu.checkpoint': checkpoint.code, 'delu.registry_id': entry.id,
            'delu.kind': entry.kind, 'delu.generation': entry.version or 'none', 'delu.policy_code': code,
            'delu.public_name': entry.name, 'delu.status': X._draft_status_text(entry, pending), 'delu.comparator': comparator,
            'delu.role': 'candidate', 'delu.adopted': G.adopted_flag(entry),
            'delu.evidence_class': 'development_post_selection',
            'delu.population_id': group['population_id'], 'delu.comparability_id': group['comparability_id'],
            'delu.model_code_sha': spec['model_code_sha'], 'delu.evidence_ref': spec['evidence_ref'],
            'delu.source_blobs': X._canonical(blobs), 'delu.backfilled': 'false', 'delu.original_completed_utc': pending,
            'mlflow.note.content': X._draft_description(entry, note, pending, code=code, role='candidate'),
        }
        inputs = X.datasets('cp24', code, group, weather=True)
        tags['delu.datasets'] = X._canonical({d['name']: d['sha256'] for d in inputs})
        run = {'run_key': run_key, 'parent': checkpoint.run_key, 'run_name': run_name, 'params': params(k, code), 'tags': tags,
               'metrics': builder.metrics, 'metric_provenance': builder.provenance,
               'metric_units': {key: X.metric_unit(key) for key in builder.metrics}, 'inputs': inputs}
        summary = X._canonical({x: v for x, v in run.items() if x != 'metric_provenance'})
        run['artifacts'] = [X._artifact('summary.json', summary + '\n'),
                            X._artifact('README.md', readme(X, run_name, checkpoint, spec, blobs))]
        out.append(run)
    return out


def build(k: int) -> dict:
    X = _module()
    spec, entries, checkpoint = draft_registry()
    return {'experiment': X.EXPERIMENT, 'checkpoint': checkpoint.code, 'status': 'draft', 'scored_attempt': k,
            'pending_fields': spec['pending_fields'], 'draft_registry': DRAFT_REGISTRY, 'revisions': spec.get('revisions'),
            'runs': runs(k)}


def problems(draft: dict) -> list[str]:
    X = _module()
    _, entries, checkpoint = draft_registry()
    keys = [run['run_key'] for run in draft['runs']]
    expected = [checkpoint.run_key] + [f'{checkpoint.run_key}/{code}' for code in checkpoint.children]
    out = []
    if sorted(keys) != sorted(expected) or len(keys) != len(set(keys)):
        out.append(f'run keys differ from the draft checkpoint: {keys}')
    for run in draft['runs']:
        if run['run_name'] != X.draft_run_name(run['run_key'], entries, checkpoint):
            out.append(f"{run['run_key']}: run name is not the draft entry's")
        want = None if '/' not in run['run_key'] else checkpoint.run_key
        if run['parent'] != want:
            out.append(f"{run['run_key']}: parent {run['parent']!r}")
    return out


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--attempt', type=int, required=True)
    ap.add_argument('--check', action='store_true')
    args = ap.parse_args()
    X = _module()
    draft = build(args.attempt)
    found = problems(draft)
    if found:
        raise SystemExit('the draft export does not match its draft entries: ' + '; '.join(found))
    findings = X.outbound_findings(X.outbound_strings({'cp24': draft}), X.local_secrets())
    if findings:
        for finding in dict.fromkeys(findings):
            print(f'cp24 export: BLOCKED - {finding}', file=sys.stderr)
        return 1
    text = json.dumps(draft, indent=1, sort_keys=True, ensure_ascii=False) + '\n'
    path = DRAFT_DIR / 'cp24.json'
    if args.check:
        same = path.exists() and path.read_text() == text
        print('the committed draft export is current' if same else f'the committed draft export is stale: {path}')
        return 0 if same else 1
    DRAFT_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    print(f'wrote {path.relative_to(ROOT)}: {len(draft["runs"])} runs (draft; pending fields {sorted(draft["pending_fields"])})')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
