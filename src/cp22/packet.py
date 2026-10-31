"""The publication packet's draft registry entries (capstone v21-r9 §20.9), derived mechanically from
the committed §20.6 verdicts. Run as ``python -m cp22.packet`` (``--check`` verifies the committed file).

Nothing is registered in CP-22: the entries are drafts, statuses are dated at landing, and every
identity that exists only at landing is an explicit pending field. Names follow the plan's naming
rule (§20.6; PUBLISH_RULES 1.3 A10):

* **A replacement (R or M):** v4 keeps its number and gains a dated revision. Its current
  revision -- W, or W with the dynamic layer if adopted -- is v4's current construction, proposed
  as "v4 · LightGBM member added"; CP-21's three-block construction (HGL) becomes the superseded
  revision, labelled "three-block LightGBM added", with its evidence, records and result unchanged.
  Every other new policy is a study arm.
* **No replacement:** v4 is unchanged; CP-22 is drafted as a descriptive branch after v4 whose
  status the Owner decides first (§20.6); every new policy is a study arm.
"""
from __future__ import annotations

import json
from pathlib import Path

from .budget import atomic
from .execution import OUT

PENDING = 'pending-at-landing'
CLAIMS = 'docs/track-b/research-content/cp22-claims.md'
PLAN = 'capstone_v21.md §20 (v21-r9)'
SOURCES = ('reports/v4-revision/metrics.csv', 'reports/v4-revision/uncertainty.csv',
           'reports/v4-revision/criteria.csv', 'reports/v4-revision/decisions.json')
RULE_WORDS = {
    'cp22-replacement': {1: 'non-inferiority against v4 on both scores', 2: 'all six original screening diagnostics',
                         3: 'a complete, valid evaluation', 4: 'no resolved per-fold degradation'},
    'cp22-dynamic-layer': {1: 'an interval-score gain whose upper 95% endpoint is below zero',
                           2: 'no point-error interval lying entirely above zero', 3: 'all six original screening diagnostics',
                           4: 'pooled 95% coverage closer to 0.95', 5: 'no resolved per-fold degradation'},
    'cp22-fast-component': {1: 'an interval-score gain whose upper 95% endpoint is below zero',
                            2: 'no point-error interval lying entirely above zero', 3: 'all six original screening diagnostics',
                            4: 'no resolved per-fold degradation'},
}
#: (registry id, name, subtitle, short) for each new policy when it is a study arm.
ARMS = {
    'R': ('v3-plus-pooled-normalized-lightgbm-averaged', 'v3 plus a capacity-averaged pooled LightGBM',
          'v3 with a normalized pooled LightGBM, capacities averaged', 'Pooled member, averaged'),
    'M': ('v3-plus-pooled-lightgbm-pair', 'v3 plus pooled normalized and raw LightGBM',
          'v4 with the block split removed', 'Pooled pair'),
    'A-PN-sel': ('v3-plus-pooled-normalized-lightgbm-selected', 'v3 plus a pooled normalized LightGBM',
                 'The raw half dropped; daily capacity selection', 'Pooled normalized member'),
    'A-LP': ('v3-plus-pooled-raw-lightgbm', 'v3 plus a pooled raw LightGBM', 'v3 with the pooled raw LightGBM member',
             'Pooled raw member'),
    'A-LN': ('v3-plus-normalized-block-lightgbm', 'v3 plus normalized block LightGBM',
             'v3 with the normalized three-block member', 'Normalized block member'),
    'v4+DL': ('v4-with-dynamic-intervals', 'v4 with dynamic intervals', 'Three-block v4 with the dynamic interval layer',
              'v4 + dynamic layer'),
    'v3+DL': ('v3-with-dynamic-intervals', 'v3 with dynamic intervals', 'v3 with the dynamic interval layer',
              'v3 + dynamic layer'),
    'W+ACI': ('replacement-with-adaptive-coverage', 'Replacement with adaptive coverage',
              'Adaptive coverage on the unweighted 28-day buffer', 'Adaptive coverage'),
    'W+DL': ('replacement-with-dynamic-intervals', 'Replacement with dynamic intervals',
             '7-day recency weights and adaptive coverage', 'Dynamic layer'),
    'W+DLF': ('replacement-with-fast-dynamic-intervals', 'Replacement with a fast dynamic layer',
              'The dynamic layer plus a one-day kernel', 'Dynamic layer + fast'),
}
CURRENT_CHANGE = {'R': 'one pooled LightGBM member, averaged over four capacities',
                  'M': 'pooled normalized and raw LightGBM members, the block split removed'}


def _entry(ident, name, subtitle, kind, code, status, reason, *, run_keys, style, short, anchor=None, after=None,
           question='', predecessor=None, codes=None, statuses=None, comparator='v4', rules=('cp22-replacement',)):
    return {'id': ident, 'name': name, 'subtitle': subtitle, 'kind': kind,
            'codes': codes or [{'experiment': 'CP-22', 'code': code, 'population': 'common-10747h', 'note': ''}],
            'statuses': statuses or [{'status': status, 'date': PENDING, 'source': f'{PENDING}: the CP-22 landing record',
                                      'reason': reason}],
            'comparator': comparator, 'population': 'common-10747h', 'evidence_class': 'development_post_selection',
            'plan': PLAN, 'rules': list(rules), 'sources': list(SOURCES), 'claim_map': CLAIMS, 'run_keys': list(run_keys),
            'style': style, 'anchor': anchor, 'checkpoint': 'CP-22', 'after': after, 'question': question, 'informed': None,
            'short': short, 'predecessor': predecessor}


def _unmet(rule: str, decision: dict) -> str:
    first = decision['first_unmet_condition']
    return f'condition {first} of rule {rule}, {RULE_WORDS[rule][int(first)]}, was not met'


def build(root: Path) -> dict:
    decisions = json.loads((root / OUT / 'decisions.json').read_text())
    rep, dl, fast = decisions['replacement'], decisions['dynamic_layer'], decisions['fast_component']
    w = rep['winner']
    current = None
    if w:
        current = 'W+DLF' if fast.get('adopted') else ('W+DL' if dl.get('adopted') else w)
    policies = ['R', 'M', 'A-PN-sel', 'A-LP', 'A-LN', 'v4+DL', 'v3+DL'] + (['W+ACI', 'W+DL', 'W+DLF'] if w else [])
    entries, revisions = [], None
    if w:
        owner = 'v4'
        layer_words = {'W+DLF': ' with the dynamic interval layer and its fast component', 'W+DL': ' with the dynamic interval layer'}
        change = CURRENT_CHANGE[w] + layer_words.get(current, '')
        entries.append({
            'id': 'v4', 'name': 'v4 · LightGBM member added', 'subtitle': 'v3 plus one pooled LightGBM member',
            'kind': 'generation',
            'codes': [{'experiment': 'CP-21', 'code': 'HGL', 'population': 'common-10747h', 'note': 'superseded revision'},
                      {'experiment': 'CP-22', 'code': current, 'population': 'common-10747h', 'note': 'current revision'},
                      {'experiment': 'CP-22', 'code': 'HGL', 'population': 'common-10747h', 'note': 'comparator, saved'}],
            'statuses': [{'status': 'adopted in research', 'date': '2026-09-30', 'source': 'docs/track-b/cp-21-landing-2026-09-30.md',
                          'reason': ''},
                         {'status': 'adopted in research', 'date': PENDING, 'source': f'{PENDING}: the CP-22 landing record',
                          'reason': f'revised in place (PUBLISH_RULES 1.3 A10): {change}'}],
            'comparator': 'v3', 'population': 'common-10747h', 'evidence_class': 'development_post_selection',
            'plan': PLAN, 'rules': ['cp22-replacement', 'cp22-dynamic-layer', 'cp22-fast-component'], 'sources': list(SOURCES),
            'claim_map': CLAIMS, 'run_keys': ['cp22', f'cp22/{current}'], 'style': 'v4', 'anchor': '#v4', 'checkpoint': 'CP-22',
            'after': None, 'question': '', 'informed': None, 'short': 'v4', 'predecessor': 'v3'})
        revisions = [
            {'revision': 1, 'label': 'three-block LightGBM added', 'state': 'superseded revision', 'checkpoint': 'CP-21',
             'code': 'HGL', 'run_keys': ['cp21', 'cp21/HGL'], 'rule': 'cp21-adoption', 'adopted': '2026-09-30',
             'superseded': PENDING, 'evidence': 'evidence/cp-21 (1d13f99); reports/block-challenger/',
             'note': 'kept unchanged and visible: evidence, MLflow records, comparator v3, result and limitations (A10)'},
            {'revision': 2, 'label': 'LightGBM member added', 'state': 'current revision', 'checkpoint': 'CP-22', 'code': current,
             'run_keys': ['cp22', f'cp22/{current}'], 'rule': 'cp22-replacement' + (', cp22-dynamic-layer' if dl.get('adopted') else '')
             + (', cp22-fast-component' if fast.get('adopted') else ''), 'adopted': PENDING, 'evidence': 'evidence/cp-22 (pending)',
             'change': change, 'construction': {'member': w, 'interval_layer': {'W+DLF': 'DLF', 'W+DL': 'DL'}.get(current, 'H')}}]
        for code in policies:
            if code == current:
                continue
            ident, name, subtitle, short = ARMS[code]
            if code in ('R', 'M'):
                reason = ('the replacement member; its interval layer was superseded by the adopted dynamic layer'
                          if code == w else ('tried after R: not reached, R replaced v4' if code == 'M' and w == 'R'
                                             else _unmet('cp22-replacement', rep['candidates'][code])))
            elif code == 'W+DL':
                reason = _unmet('cp22-dynamic-layer', dl) if dl.get('applies') and not dl.get('adopted') else 'superseded by the adopted fast component'
            elif code == 'W+DLF':
                reason = (_unmet('cp22-fast-component', fast) if fast.get('applies') else
                          'decided only as an add-on to an adopted dynamic layer; descriptive here')
            else:
                reason = 'a study arm for attribution; never eligible'
            entries.append(_entry(ident, name, subtitle, 'study arm', code, 'not adopted', reason, run_keys=[f'cp22/{code}'],
                                  style='study', short=short))
    else:
        owner = 'pooled-lightgbm-member-for-v4'
        reason = (f'cp22-replacement yielded no replacement: for R, {_unmet("cp22-replacement", rep["candidates"]["R"])}; '
                  f'for M, {_unmet("cp22-replacement", rep["candidates"]["M"])}; the Owner decides first (§20.6)')
        entries.append(_entry(owner, 'Pooled LightGBM member for v4', 'The block split removed, under non-inferiority',
                              'branch', 'CP-22', 'not adopted', reason, run_keys=['cp22'], style='branch',
                              short='Pooled member for v4', anchor='#branch-pooled-lightgbm-member-for-v4', after='v4',
                              question='Can one pooled LightGBM member replace v4\'s three-block member without losing accuracy?'))
        for code in policies:
            ident, name, subtitle, short = ARMS[code]
            why = (_unmet('cp22-replacement', rep['candidates'][code]) if code in ('R', 'M')
                   else 'a study arm for attribution; never eligible')
            entries.append(_entry(ident, name, subtitle, 'study arm', code, 'not adopted', why, run_keys=[f'cp22/{code}'],
                                  style='study', short=short))
    words = {'R': 'R, the capacity-averaged pooled member, replaced v4\'s three-block member',
             'M': 'M, the pooled pair with the split removed, replaced v4\'s three-block member', None: 'neither R nor M met the rule'}[w]
    layer = (f'; the dynamic interval layer was {"adopted" if dl.get("adopted") else "not adopted"}' if dl.get('applies') else '')
    layer += (f'; the fast component was {"adopted" if fast.get("adopted") else "not adopted"}' if fast.get('applies') else '')
    return {
        'schema': 'cp22-draft-registry-v1',
        'status': 'draft registry entries for the publication packet; CP-22 registers nothing; PRES-4 registers them after '
                  'the Owner\'s landing' + ('' if w else ', and only after the Owner decides (no replacement)'),
        'outcome': {'replacement': w, 'current_revision_code': current, 'dynamic_layer_adopted': bool(dl.get('adopted')),
                    'fast_component_adopted': bool(fast.get('adopted'))},
        'verdict_source': 'reports/v4-revision/decisions.json', 'pending': PENDING,
        'pending_fields': {
            'statuses[].date': 'the revision or not-adoption date, set at landing',
            'statuses[].source': 'the CP-22 landing record, written at landing',
            'checkpoint.evidence_sha': 'the evidence/cp-22 tag commit, created by the Owner at landing',
            'checkpoint.frozen_on': 'the evidence tag freeze date', 'checkpoint.landing': 'the landing record path',
            'model_code_sha': 'the final reviewed candidate SHA: a commit cannot contain its own SHA; it is recorded in the CP-22 '
                              'return and filled by the publication block',
            'evidence_ref': 'evidence/cp-22 at its commit', 'original_completed_utc': 'the terminal return time',
            'revisions[].adopted / superseded': 'the revision date, set at landing'},
        'model_code_sha': PENDING, 'evidence_ref': f'evidence/cp-22@{PENDING}',
        'note': (f'CP-22 re-examined v4 under the pre-registered rule cp22-replacement (non-inferiority against v4 on both scores and '
                 f'per fold, with all six original screening diagnostics): {words}{layer}. development_post_selection.'),
        'checkpoint': {'code': 'CP-22', 'run_key': 'cp22', 'owner': owner, 'evidence_tag': 'evidence/cp-22',
                       'evidence_sha': PENDING, 'frozen_on': PENDING, 'report': 'reports/v4-revision/report.md',
                       'verdict': 'docs/track-b/evidence/cp-22/integration.md', 'landing': PENDING, 'children': policies},
        'entries': entries, 'revisions': revisions,
        'code_additions_to_existing_entries': [
            {'entry': 'v4', 'code': {'experiment': 'CP-22', 'code': 'HGL', 'population': 'common-10747h', 'note': 'comparator, saved'}},
            {'entry': 'v3', 'code': {'experiment': 'CP-22', 'code': 'HG', 'population': 'common-10747h', 'note': 'reference, saved'}},
            {'entry': 'v2', 'code': {'experiment': 'CP-22', 'code': 'H0', 'population': 'common-10747h', 'note': 'saved reference'}},
            {'entry': 'naive', 'code': {'experiment': 'CP-22', 'code': 'B0', 'population': 'common-10747h', 'note': 'normalizer'}},
            {'entry': 'v1', 'code': {'experiment': 'CP-22', 'code': 'B1', 'population': 'common-10747h', 'note': 'development replay'}},
            {'entry': 'daily-lear', 'code': {'experiment': 'CP-22', 'code': 'B2', 'population': 'common-10747h', 'note': ''}},
            {'entry': 'daily-lightgbm', 'code': {'experiment': 'CP-22', 'code': 'B3', 'population': 'common-10747h', 'note': ''}},
            {'entry': 'normalized-lear', 'code': {'experiment': 'CP-22', 'code': 'A1', 'population': 'common-10747h', 'note': ''}}],
        'rules': [{'id': rid, 'plan': 'capstone_v21.md §20.6 (v21-r9)', 'set_on': '2026-10-01',
                   'provenance': ['capstone_v21.md v21-r9 §20.6 (ratified 2026-10-01)',
                                  'reports/v4-revision/protocol.json rules_verbatim (frozen before scoring)']}
                  for rid in ('cp22-replacement', 'cp22-dynamic-layer', 'cp22-fast-component')],
        'transition': ({'title': 'From v3 to v4: adding a LightGBM member', 'revision': 'revision 2 (CP-22), current',
                        'predecessor': 'v3', 'comparator_of_the_revision_rule': 'v4 (three-block revision)',
                        'comparator_is_predecessor': False, 'revision_note': 'dated note with the measured difference against '
                        'the three-block revision and its interval (A10)'} if w else None),
        'encoding': {'v4': 'amber #B45309, filled diamond, direct label "v4" (D6, unchanged)'},
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
