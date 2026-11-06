"""The live claim builder: what the final-product Space says, from the claim layer and the committed records.

`tests/test_24_live_namespace_is_walled_off.py` keeps the wall: every claim built here has a `live_` key, the record
claim set (`delu_forecast.claims.build_claims`) holds none of them, and no surface that speaks for the model of
record renders one. The Space renders its Limitations view, its product status, its badges and its fixed wording from
this builder only, so there is no hand-written limitation copy and no hand-typed number.

Two kinds of entry:

- **`values`** (`live_*`): the live product's own claims. Bare figures the Space serves in its records (τ, the rolling
  window's length, τ_eq) are read there, not repeated here. Every figure is read from a committed record (the CP-17c
  freeze manifest, CP-17's frozen protocol and report, `cp17.scoring` and `cp17.attribution`'s frozen labels); the
  sentences around them are fixed here.
- **`shared`**: record-layer claims that apply unchanged to the live product, rendered verbatim by their record key
  (`build_claims()`): the data attribution, the GFS attribution and the quarter-hour averaging; the Data view takes the price floor's definition from the record sentence's first clause.

**O13** (the Orchestrator's ruling under decision 27, adopting CP-17's protocol §12): A1 is "final candidate · pending
4.7T" until 4.7T is scored; it becomes "live" only at CP-18's activation, which requires a passed 4.7T; on a fail, v1
stays the released product. **O14:** CP-17's proposed visitor wording for "Train it yourself", plus one sentence that
states F1 at its largest observed figure, as §24.12 (R3) records it.
"""
from __future__ import annotations

import json
from pathlib import Path
import re

MANIFEST = 'docs/track-b/evidence/cp-17c/freeze-manifest.json'
PROTOCOL = 'docs/track-b/evidence/cp-17/4-7t-frozen-protocol.md'
CP17_REPORT = 'reports/cp17/report.md'
SHARED = ('attribution', 'limitation_mtu_averaging')
#: The claim layer's constants that are not record claims of their own (`delu_forecast.claims`), shared verbatim.
SHARED_CONSTANTS = {'gfs_attribution': 'GFS_ATTRIBUTION'}
STATES = ('pending', 'passed', 'failed')
LIMITATION_ORDER = ('evidence', 'rehearsal', 'f1', 'retrain_selections', 'inputs', 'rendered', 'operation')


def _read(root, rel) -> str:
    return (Path(root) / rel).read_text(encoding='utf-8')


def _quoted(text: str, after: str) -> str:
    """The first double-quoted passage after the marker `after`, with its line breaks joined."""
    i = text.index(after)
    m = re.search(r'"(.+?)"', text[i:], re.S)
    return ' '.join(m.group(1).split())


def _f1(root) -> tuple[str, str, str]:
    """F1's largest observed figure, its quantile level and its day, from the freeze manifest's disclosure."""
    identity = json.loads(_read(root, MANIFEST))['policy_identity']
    m = re.search(r'by up to ([0-9.]+) EUR/MWh \((p[0-9.]+), (\d{4}-\d{2}-\d{2})', identity['disclosures']['F1_largest_observed'])
    return m.group(1), m.group(2), m.group(3)


def build_live_claims(root=None, fresh_test: str = 'pending', rehearsal_days: tuple[str, ...] = ()) -> dict:
    """{'values': {live_key: text}, 'shared': {record_key: text}, 'limitations': [(label, key)]}."""
    from cp17 import scoring
    from cp17.attribution import WORDING
    import delu_forecast.claims as record
    from delu_forecast.claims import REPO_ROOT, build_claims
    root = Path(root or REPO_ROOT)
    if fresh_test not in STATES:
        raise ValueError(f'fresh_test: one of {STATES}')
    identity = json.loads(_read(root, MANIFEST))
    pi = identity['policy_identity']
    protocol = _read(root, PROTOCOL)
    report = _read(root, CP17_REPORT)
    f1, f1_level, f1_day = _f1(root)
    f1_where = f"A1's {f1_level} on {f1_day}"
    status = {'pending': 'final candidate · pending 4.7T', 'passed': 'live',
              'failed': 'tested candidate · did not pass 4.7T'}[fresh_test]
    first, last = (rehearsal_days[0], rehearsal_days[-1]) if rehearsal_days else ('', '')
    values = {
        'live_product_name': 'A1 · DDNN-2 alone',
        'live_product_designation': pi['designation'],
        'live_product_status': status,
        'live_released_product': ('v1 stays the released product until CP-18\'s activation, which requires a passed '
                                  '4.7T' if fresh_test != 'failed' else 'v1 stays the released product'),
        'live_hero_rule': ("A1 is shown as \"final candidate · pending 4.7T\" until 4.7T is scored. It becomes \"live\" "
                           "only at CP-18's activation, which requires a passed 4.7T. On a fail, v1 stays the released "
                           "product."),
        'live_fresh_test_badge': _quoted(protocol, '**The badge:**'),
        'live_fresh_test_wording': _quoted(protocol, '**The wording:**'),
        'live_development_badge': 'development · post-selection',
        'live_prospective_label': 'prospective evaluation in progress',
        'live_rehearsal_badge': ('development rehearsal · CP-18 dry runs on development days'
                                 + (f' {first} to {last}' if first else '') + ' · not prospective and not 4.7T'),
        'live_percentage_label': scoring.PERCENTAGE_LABEL,
        'live_coverage_wording': 'nominal {level}% interval; measured coverage {share} over {window}; mean width {width} EUR/MWh',
        'live_retrained_label': 'your retrained model',
        'live_counterfactual_label': 'counterfactual scenario — not an issued forecast',
        'live_attribution_wording': WORDING,
        'live_explain_wording': 'contributed {value} EUR/MWh to this forecast, relative to the model\'s baseline',
        'live_features_wording': 'how much this fitted model relies on each input',
        'live_retrain_wording': (_quoted(report, '**Proposed wording**') + ' The issued values were computed in the same '
                                 'WebAssembly runtime the visitor runs. On development days that runtime\'s results '
                                 f'differed from native arithmetic by up to {f1} EUR/MWh ({f1_where}), the largest '
                                 'difference observed, not a bound.'),
        'live_policy_identity_sha256': identity['policy_identity_sha256'],
        'live_package_sha256': pi['code']['package_sha256'],
        'live_issuing_host': 'Pyodide 0.28.0 in Node.js v26.7.0',
        'live_reference_runtime': (f"Pyodide {pi['reference_runtime']['pyodide']}; Chromium "
                                   f"{pi['reference_runtime']['chromium']} and WebKit {pi['reference_runtime']['webkit']} "
                                   f"through Playwright {pi['reference_runtime']['playwright']}; "
                                   f"{pi['reference_runtime']['machine_class']}"),
        'live_cutoff': 'D−1 11:00 Europe/Berlin, strictly before',
        # the Data view's definition, from the claim layer's sentence (its first clause; the rest is v1's)
        'live_data_floor': record.FLOOR_CHANGE.split(',')[0] + '.',
        # the limitations, generated
        'live_limitation_evidence': (
            'Every result shown is development evidence on data through 2026-09-06, post-selection and descriptive. '
            'A1 has not yet been tested on fresh data: its one-shot test (4.7T) is registered and not yet scored, and '
            'the prospective record is "prospective evaluation in progress" until CP-19.'),
        'live_limitation_rehearsal': (
            'The issued days in this Space are a development rehearsal: CP-18\'s dry runs of the daily pipeline on '
            + (f'development days {first} to {last}, ' if first else 'development days, ') +
            'with a simulated wall clock. They are not prospective forecasts and not 4.7T, and they are never scored '
            'as either.'),
        'live_limitation_f1': (
            'The issued forecasts come from the frozen package running in WebAssembly (Pyodide), the runtime a '
            'visitor\'s browser runs. On development days its results differed from native arithmetic by up to '
            f'{f1} EUR/MWh ({f1_where}), the largest difference observed and not a bound; a visitor\'s retrain is compared with the '
            'issued artifact, not with native arithmetic.'),
        'live_limitation_retrain_selections': (
            '"Train it yourself" compares the data-driven selections the issued artifact records for each of the '
            'eight members (the seed, the training and held-out day sets and the latest training day) and each '
            'member\'s weights\' hash; the artifact does not record the other selections (early stopping\'s epochs, '
            'the winsorisation and cap activations), so those are not compared.'),
        'live_limitation_inputs': (
            'A live day is issued only if each input can be shown to precede its cutoff (D−1 11:00 Europe/Berlin). '
            'The day-ahead load forecast is admitted by its retrieval before the cutoff; its stored development '
            'history rests on the Owner\'s recorded exception (ruling 1), which covers development days only.'),
        'live_limitation_rendered': (
            'On development days the market inputs and outcomes are the development snapshot\'s values (SMARD, pulled '
            '2026-09-08) rendered in each source\'s response shape, not recorded raw responses, so each price\'s '
            'publication time is the documented schedule\'s (13:00 Europe/Berlin on the day before delivery).'),
        'live_limitation_operation': (
            'Daily operation starts only at CP-18\'s activation. Until then nothing here is published, and no '
            'forecast is claimed to have been published before the auction or before its outcome.'),
    }
    labels = {'evidence': 'Development evidence only', 'rehearsal': 'A development rehearsal, not a live record',
              'f1': 'The issuing runtime against native arithmetic', 'retrain_selections': 'What a retrain compares',
              'inputs': 'Inputs shown to precede the cutoff', 'rendered': 'Rendered development responses',
              'operation': 'Not yet operating'}
    claims = build_claims()
    shared = {k: claims[k] for k in SHARED}
    shared.update({k: getattr(record, name) for k, name in SHARED_CONSTANTS.items()})
    limitations = [(labels[t], f'live_limitation_{t}') for t in LIMITATION_ORDER] + [
        ('The 15-minute MTU averaging choice', 'limitation_mtu_averaging'),
        ('Data attribution', 'attribution'),
        ('Weather data attribution', 'gfs_attribution')]
    return {'values': values, 'shared': shared, 'limitations': limitations}


def text(claims: dict, key: str) -> str:
    return claims['values'][key] if key.startswith('live_') else claims['shared'][key]
