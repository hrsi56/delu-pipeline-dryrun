"""The publication packet's draft registry entries (capstone v21-r6 §17.9), derived mechanically
from the committed §17.6 verdict. Run under the monitor as ``python -m cp21.packet``.

Nothing is registered in CP-21: the entries are drafts, statuses are dated at landing, and every
identity that exists only at landing is an explicit pending field. Names follow the plan's naming
rule (§17.6; PUBLISH_RULES 1.1 §4): a version number only on adoption; otherwise the branch
"Three-block LightGBM on v3" with HGL, L-P, L-R and L-N as descriptive study arms.
"""
from __future__ import annotations

import json
from pathlib import Path

from .budget import atomic

OUT = Path('reports/block-challenger')
PENDING = 'pending-at-landing'
CLAIMS = 'docs/track-b/research-content/cp21-claims.md'
PLAN = 'capstone_v21.md §17 (v21-r6)'
SOURCES = ('reports/block-challenger/metrics.csv', 'reports/block-challenger/uncertainty.csv',
           'reports/block-challenger/criteria.csv', 'reports/block-challenger/adoption.json')
CONDITION_WORDS = {1: 'joint improvement over v3', 2: 'no regression on the original section-8 screen',
                   3: 'a complete, valid evaluation', 4: 'no resolved per-fold degradation'}
ARMS = {
    'L-P': ('pooled-lightgbm-weather', 'Pooled LightGBM with weather', 'One 24-hour LightGBM on v3 information', 'Pooled LightGBM'),
    'L-R': ('block-lightgbm', 'Three-block LightGBM', 'Night, solar and peak models, raw price', 'Block LightGBM'),
    'L-N': ('normalized-block-lightgbm', 'Normalized three-block LightGBM', 'The block models on a normalized price',
            'Normalized block LightGBM'),
    'HGL': ('blend-with-block-lightgbm', 'Blend with block LightGBM', 'v3 LEAR forecasts plus a LightGBM member',
            'Blend + block LightGBM'),
}


def _entry(ident, name, subtitle, kind, code, status, reason, *, run_key, style, short, anchor=None, after=None,
           question='', predecessor=None):
    return {'id': ident, 'name': name, 'subtitle': subtitle, 'kind': kind,
            'codes': [{'experiment': 'programme' if kind == 'branch' else 'CP-21', 'code': code,
                       'population': 'common-10747h', 'note': ''}],
            'statuses': [{'status': status, 'date': PENDING, 'source': f'{PENDING}: the CP-21 landing record',
                          'reason': reason}],
            'comparator': 'v3', 'population': 'common-10747h', 'evidence_class': 'development_post_selection',
            'plan': PLAN, 'rules': ['cp21-adoption'], 'sources': list(SOURCES), 'claim_map': CLAIMS,
            'run_keys': [run_key], 'style': style, 'anchor': anchor, 'checkpoint': 'CP-21', 'after': after,
            'question': question, 'informed': None, 'short': short, 'predecessor': predecessor}


def build(root: Path) -> dict:
    adoption = json.loads((root / OUT / 'adoption.json').read_text())
    adopted = adoption['verdict'] == 'v4'
    first = adoption['first_unmet_condition']
    reason = '' if adopted else f'condition {first} of rule cp21-adoption, {CONDITION_WORDS[first]}, was not met'
    status = 'adopted in research' if adopted else 'not adopted'
    split = adoption['block_split']['reading']
    entries = []
    if adopted:
        owner = 'v4'
        entries.append(_entry('v4', 'v4 · three-block LightGBM added', 'v3 plus a three-block LightGBM member', 'generation',
                              'HGL', status, reason, run_key='cp21/HGL', style='v4', short='v4', anchor='#v4',
                              predecessor='v3'))
        entries[-1]['run_keys'] = ['cp21', 'cp21/HGL']
    else:
        owner = 'block-lightgbm-on-v3'
        entries.append(_entry(owner, 'Three-block LightGBM on v3', 'A nonlinear block member added to v3', 'branch',
                              'CP-21', status, reason, run_key='cp21', style='branch', short='Three-block LightGBM',
                              anchor='#branch-block-lightgbm-on-v3', after='v3',
                              question='Does adding a three-block LightGBM member to v3 improve both of its error scores?'))
    for code in ('HGL', 'L-P', 'L-R', 'L-N'):
        if adopted and code == 'HGL':
            continue
        ident, name, subtitle, short = ARMS[code]
        entries.append(_entry(ident, name, subtitle, 'study arm', code, 'not adopted',
                              'a study arm for attribution; never eligible for adoption' if code != 'HGL' else reason,
                              run_key=f'cp21/{code}', style='study', short=short))
    verdict_words = ('HGL met the four conditions of rule cp21-adoption, the third being this checkpoint\'s independent '
                     'Integration PASS, and was adopted in research as v4' if adopted else f'HGL was not adopted: {reason}')
    return {
        'schema': 'cp21-draft-registry-v1',
        'status': 'draft registry entries for the publication packet; CP-21 registers nothing; the publication block '
                  'registers them after the Owner\'s landing',
        'outcome': 'adopted' if adopted else 'not_adopted', 'verdict_source': 'reports/block-challenger/adoption.json',
        'pending': PENDING,
        'pending_fields': {
            'statuses[].date': 'the adoption or not-adoption date, set at landing',
            'statuses[].source': 'the CP-21 landing record, written at landing',
            'checkpoint.evidence_sha': 'the evidence/cp-21 tag commit, created by the Owner at landing',
            'checkpoint.frozen_on': 'the evidence tag freeze date',
            'checkpoint.landing': 'the landing record path',
            'model_code_sha': 'the final reviewed candidate SHA: a commit cannot contain its own SHA; it is recorded in '
                              'the CP-21 return and filled by the publication block',
            'evidence_ref': 'evidence/cp-21 at its commit',
            'original_completed_utc': 'the terminal return time'},
        'model_code_sha': PENDING, 'evidence_ref': f'evidence/cp-21@{PENDING}',
        'note': ('CP-21 added a three-block LightGBM member to v3\'s blend of two LEAR forecasts (HGL) and compared it '
                 'with v3 under the pre-registered rule cp21-adoption; pooled, block and normalized-block LightGBM '
                 f'study arms attribute the change. {verdict_words}. The block split (block against pooled LightGBM) '
                 f'showed {split}. development_post_selection.'),
        'checkpoint': {'code': 'CP-21', 'run_key': 'cp21', 'owner': owner, 'evidence_tag': 'evidence/cp-21',
                       'evidence_sha': PENDING, 'frozen_on': PENDING, 'report': 'reports/block-challenger/report.md',
                       'verdict': 'docs/track-b/evidence/cp-21/integration.md', 'landing': PENDING,
                       'children': ['HGL', 'L-P', 'L-R', 'L-N']},
        'entries': entries,
        'code_additions_to_existing_entries': [
            {'entry': 'v3', 'code': {'experiment': 'CP-21', 'code': 'HG', 'population': 'common-10747h', 'note': 'comparator, saved'}},
            {'entry': 'v2', 'code': {'experiment': 'CP-21', 'code': 'H0', 'population': 'common-10747h', 'note': 'saved reference'}},
            {'entry': 'naive', 'code': {'experiment': 'CP-21', 'code': 'B0', 'population': 'common-10747h', 'note': 'normalizer'}},
            {'entry': 'v1', 'code': {'experiment': 'CP-21', 'code': 'B1', 'population': 'common-10747h', 'note': 'development replay'}},
            {'entry': 'daily-lear', 'code': {'experiment': 'CP-21', 'code': 'B2', 'population': 'common-10747h', 'note': ''}},
            {'entry': 'daily-lightgbm', 'code': {'experiment': 'CP-21', 'code': 'B3', 'population': 'common-10747h', 'note': ''}},
            {'entry': 'normalized-lear', 'code': {'experiment': 'CP-21', 'code': 'A1', 'population': 'common-10747h', 'note': ''}}],
        'rule': {'id': 'cp21-adoption', 'plan': 'capstone_v21.md §17.6 (v21-r6)', 'set_on': '2026-09-29',
                 'words': 'HGL becomes v4 only if both paired differences against v3 improve (the interval score\'s upper '
                          '95% endpoint below zero, the point error\'s at or below zero), it meets all six original '
                          'screening diagnostics, the evaluation is complete and valid, and no fold is decisively worse',
                 'provenance': ['capstone_v21.md v21-r6 §17.6 (ratified 2026-09-29)',
                                'reports/block-challenger/protocol.json adoption_rule_verbatim (frozen before scoring)']},
        'transition': ({'title': 'From v3 to v4: adding a three-block LightGBM', 'predecessor': 'v3', 'comparator': 'v3',
                        'comparator_is_predecessor': True} if adopted else None),
    }


def main() -> int:
    import sys
    root = Path.cwd()
    registry = build(root)
    if '--check' in sys.argv[1:]:
        same = json.loads((root / OUT / 'draft-registry.json').read_text()) == registry
        print(json.dumps({'draft_registry_identical_to_committed': same}), flush=True)
        return 0 if same else 11
    atomic(root / OUT / 'draft-registry.json', registry)
    print(json.dumps({'outcome': registry['outcome'], 'owner': registry['checkpoint']['owner'],
                      'entries': [e['id'] for e in registry['entries']]}), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
