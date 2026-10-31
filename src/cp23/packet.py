"""The publication packet's draft registry entries (capstone v21-r10 §21.9), derived mechanically from the
committed `cp23-adoption` verdict. Run as ``python -m cp23.packet`` (``--check`` verifies the committed file).

Nothing is registered in CP-23. The entries are drafts, statuses are dated at landing, and every identity
that exists only at landing is an explicit pending field. Names follow §21.6's proposal:

* **Adopted:** "v5 · DDNN member added", a generation with predecessor v4. D and v3+D are study arms.
* **Not adopted:** the branch "DDNN member on v4", attached after v4, with the first unmet condition and
  its values as the reason. D and v3+D are study arms. The Owner decides whether the branch is
  published (§21.9).
"""
from __future__ import annotations

import json
from pathlib import Path

from .budget import atomic
from .execution import OUT

PENDING = 'pending-at-landing'
CLAIMS = 'docs/track-b/research-content/cp23-claims.md'
PLAN = 'capstone_v21.md §21 (v21-r10)'
SOURCES = ('reports/distribution-challenger/metrics.csv', 'reports/distribution-challenger/uncertainty.csv',
           'reports/distribution-challenger/criteria.csv', 'reports/distribution-challenger/decisions.json')
RULE_WORDS = {1: 'a joint improvement over v4, with the upper 95% endpoint of the interval-score difference below zero and '
                 'that of the point-error difference at or below zero',
              2: 'all six original screening diagnostics', 3: 'a complete, valid evaluation',
              4: 'no resolved per-fold degradation against v4'}
#: (registry id, name, subtitle, short) for each study arm.
ARMS = {
    'D': ('ddnn-alone', 'DDNN alone', 'A distributional neural network with a Johnson SU head, with its own quantiles',
          'DDNN alone'),
    'v3+D': ('v3-plus-ddnn', 'v3 plus a DDNN member', 'DDNN as v3\'s third member, in LightGBM\'s place', 'v3 + DDNN'),
}


def _entry(ident, name, subtitle, kind, code, status, reason, *, run_keys, style, short, anchor=None, after=None,
           question='', predecessor=None, comparator='v4'):
    return {'id': ident, 'name': name, 'subtitle': subtitle, 'kind': kind,
            'codes': [{'experiment': 'CP-23', 'code': code, 'population': 'common-10747h', 'note': ''}],
            'statuses': [{'status': status, 'date': PENDING, 'source': f'{PENDING}: the CP-23 landing record', 'reason': reason}],
            'comparator': comparator, 'population': 'common-10747h', 'evidence_class': 'development_post_selection',
            'plan': PLAN, 'rules': ['cp23-adoption'], 'sources': list(SOURCES), 'claim_map': CLAIMS, 'run_keys': list(run_keys),
            'style': style, 'anchor': anchor, 'checkpoint': 'CP-23', 'after': after, 'question': question, 'informed': None,
            'short': short, 'predecessor': predecessor}


def unmet_words(decision: dict) -> str:
    first = decision['first_unmet_condition']
    return f'condition {first} of rule cp23-adoption, {RULE_WORDS[int(first)]}, was not met'


def build(root: Path) -> dict:
    decisions = json.loads((root / OUT / 'decisions.json').read_text())
    decision = decisions['adoption']
    adopted = bool(decision['adopted'])
    policies = ['v5', 'D', 'v3+D']
    entries = []
    if adopted:
        owner = 'v5'
        entries.append(_entry('v5', 'v5 · DDNN member added', 'v4 plus a DDNN member', 'generation', 'v5', 'adopted in research',
                              '', run_keys=['cp23', 'cp23/v5'], style='v5', short='v5', anchor='#v5', predecessor='v4'))
    else:
        owner = 'ddnn-member-on-v4'
        entries.append(_entry(owner, 'DDNN member on v4', 'v4 plus a DDNN member, under rule cp23-adoption', 'branch', 'v5',
                              'not adopted', unmet_words(decision), run_keys=['cp23', 'cp23/v5'], style='branch',
                              short='DDNN member on v4', anchor='#branch-ddnn-member-on-v4', after='v4',
                              question='Does a distributional neural network, added to v4 as a one-third member, improve on v4 '
                                       'jointly in point and interval accuracy?'))
    for code in ('D', 'v3+D'):
        ident, name, subtitle, short = ARMS[code]
        entries.append(_entry(ident, name, subtitle, 'study arm', code, 'not adopted', 'a study arm for attribution; never eligible',
                              run_keys=[f'cp23/{code}'], style='study', short=short,
                              comparator='v4' if code == 'D' else 'v3'))
    first = decision['first_unmet_condition']
    note = ('CP-23 added a NumPy-only distributional neural network (DDNN, Johnson SU head) to v4 as a fixed one-third '
            'member and tested v5 = (2/3)·v4 + (1/3)·DDNN under the pre-registered rule cp23-adoption (joint improvement over '
            'v4, all six original screening diagnostics, a complete evaluation, no resolved per-fold degradation): '
            + ('v5 met all four conditions and is adopted in research.' if adopted else
               f'v5 is not adopted; the first unmet condition is {first} ({RULE_WORDS[int(first)]}).')
            + ' development_post_selection.')
    return {
        'schema': 'cp23-draft-registry-v1',
        'status': 'draft registry entries for the publication packet; CP-23 registers nothing; the next publication registers '
                  'them after the Owner\'s landing' + ('' if adopted else ', and only if the Owner decides to publish the branch'),
        'outcome': {'adopted': adopted, 'first_unmet_condition': first, 'unmet_conditions': decision['unmet_conditions']},
        'verdict_source': 'reports/distribution-challenger/decisions.json', 'pending': PENDING,
        'pending_fields': {
            'statuses[].date': 'the adoption or not-adoption date, set at landing',
            'statuses[].source': 'the CP-23 landing record, written at landing',
            'checkpoint.evidence_sha': 'the evidence/cp-23 tag commit, created by the Owner at landing',
            'checkpoint.frozen_on': 'the evidence tag freeze date', 'checkpoint.landing': 'the landing record path',
            'model_code_sha': 'the final reviewed candidate SHA: a commit cannot contain its own SHA; it is recorded in the CP-23 '
                              'return and filled by the publication block',
            'evidence_ref': 'evidence/cp-23 at its commit', 'original_completed_utc': 'the terminal return time'},
        'model_code_sha': PENDING, 'evidence_ref': f'evidence/cp-23@{PENDING}', 'note': note,
        'checkpoint': {'code': 'CP-23', 'run_key': 'cp23', 'owner': owner, 'evidence_tag': 'evidence/cp-23',
                       'evidence_sha': PENDING, 'frozen_on': PENDING, 'report': 'reports/distribution-challenger/report.md',
                       'verdict': 'docs/track-b/evidence/cp-23/integration.md', 'landing': PENDING, 'children': policies},
        'entries': entries, 'revisions': None,
        'code_additions_to_existing_entries': [
            {'entry': 'v4', 'code': {'experiment': 'CP-23', 'code': 'HGL', 'population': 'common-10747h', 'note': 'comparator, saved'}},
            {'entry': 'v3', 'code': {'experiment': 'CP-23', 'code': 'HG', 'population': 'common-10747h', 'note': 'reference, saved'}},
            {'entry': 'naive', 'code': {'experiment': 'CP-23', 'code': 'B0', 'population': 'common-10747h', 'note': 'normalizer'}},
            {'entry': 'v1', 'code': {'experiment': 'CP-23', 'code': 'B1', 'population': 'common-10747h', 'note': 'development replay'}},
            {'entry': 'daily-lear', 'code': {'experiment': 'CP-23', 'code': 'B2', 'population': 'common-10747h', 'note': 'reference'}},
            {'entry': 'daily-lightgbm', 'code': {'experiment': 'CP-23', 'code': 'B3', 'population': 'common-10747h', 'note': 'section-8 comparator'}},
            {'entry': 'normalized-lear', 'code': {'experiment': 'CP-23', 'code': 'A1', 'population': 'common-10747h', 'note': 'reference'}}],
        'rules': [{'id': 'cp23-adoption', 'plan': 'capstone_v21.md §21.6 (v21-r10)', 'set_on': '2026-10-04',
                   'provenance': ['capstone_v21.md v21-r10 §21.6 (ratified 2026-10-04)',
                                  'reports/distribution-challenger/protocol.json rule_verbatim (frozen before scoring)']}],
        'transition': ({'title': 'From v4 to v5: adding a distributional neural network', 'predecessor': 'v4',
                        'comparator_is_predecessor': True} if adopted else None),
        'encoding': {'v5': 'set by the Owner before the publication brief (PUBLISH_RULES §14; §21.9)'} if adopted else
                    {'note': 'no new generation; the branch, if published, uses the branch style'},
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
