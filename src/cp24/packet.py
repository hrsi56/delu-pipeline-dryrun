"""The publication packet's draft registry entries (capstone v21-r11 §23.12), derived mechanically from the
committed `cp24-adoption` verdicts of the scored attempts, or from the stop. Run as ``python -m cp24.packet``
(``--check`` verifies the committed file).

Nothing is registered in CP-24. The entries are drafts; statuses are dated at landing, and every identity
that exists only at landing is an explicit pending field. Names follow §23.9's proposal:

* **Adopted in attempt k:** "v5 · DDNN-2 member added", a generation with predecessor v4; D2 and v3+D2 are
  study arms.
* **Not adopted:** the branch "DDNN-2 member on v4", attached after v4, with each scored attempt's first
  unmet condition and its values as the reason; D2 and v3+D2 are study arms.
* **Stopped at the pre-fold gate** (no scored attempt): no draft entry; the packet records the stop.

The export's children are the deciding attempt's policies (the adopted attempt, or the last scored one),
with their committed codes `v5`, `D2` and `v3+D2`; an earlier scored attempt's entries carry the codes
`v5-a<k>`, `D2-a<k>` and `v3+D2-a<k>` and stay in the registry without export runs.
"""
from __future__ import annotations

import json
from pathlib import Path

from .budget import atomic

OUT = Path('reports/ddnn2')
PENDING = 'pending-at-landing'
CLAIMS = 'docs/track-b/research-content/cp24-claims.md'
PLAN = 'capstone_v21.md §23 (v21-r11)'
RULE_WORDS = {1: 'a joint improvement over v4 at the attempts-adjusted 97.5% level, with the upper endpoint of the '
                 'interval-score difference below zero and that of the point-error difference at or below zero',
              2: 'all six original screening diagnostics', 3: 'a complete, valid evaluation with every guard activation '
                                                              'reported', 4: 'no resolved per-fold degradation against v4',
              5: 'a practical size: both point estimates at least 0.5% better than v4'}
ARMS = {
    'D2': ('ddnn2-alone', 'DDNN-2 alone', 'A day-level distributional network with its own quantiles', 'DDNN-2 alone'),
    'v3+D2': ('v3-plus-ddnn2', 'v3 plus a DDNN-2 member', 'DDNN-2 as v3\'s third member, in LightGBM\'s place', 'v3 + DDNN-2'),
}


def attempts(root: Path) -> list[int]:
    return sorted(int(p.name.split('-')[1]) for p in (Path(root) / OUT).glob('attempt-*')
                  if (p / 'decisions.json').exists())


def rounds(root: Path) -> list[int]:
    return sorted(int(p.name.split('-')[1]) for p in (Path(root) / OUT / 'rounds').glob('round-*') if (p / 'gate.json').exists())


def outcome(root: Path) -> dict:
    ks = attempts(root)
    decisions = {k: json.loads((Path(root) / OUT / f'attempt-{k}' / 'decisions.json').read_text())['adoption'] for k in ks}
    adopted = [k for k in ks if decisions[k]['adopted']]
    if adopted:
        kind, deciding = 'adopted', adopted[0]
    elif ks:
        kind, deciding = 'branch', ks[-1]
    else:
        kind, deciding = 'stopped_at_the_pre_fold_gate', None
    gates = {r: json.loads((Path(root) / OUT / 'rounds' / f'round-{r}' / 'gate.json').read_text())['passed'] for r in rounds(root)}
    return {'kind': kind, 'deciding_attempt': deciding, 'scored_attempts': ks, 'rounds': rounds(root), 'gates_passed': gates,
            'decisions': {k: {'adopted': d['adopted'], 'first_unmet_condition': d['first_unmet_condition'],
                              'unmet_conditions': d['unmet_conditions']} for k, d in decisions.items()}}


def _entry(ident, name, subtitle, kind, codes, status, reason, *, run_keys, style, short, anchor=None, after=None,
           question='', predecessor=None, comparator='v4', sources=()):
    return {'id': ident, 'name': name, 'subtitle': subtitle, 'kind': kind,
            'codes': [{'experiment': 'CP-24', 'code': c, 'population': 'common-10747h', 'note': n} for c, n in codes],
            'statuses': [{'status': status, 'date': PENDING, 'source': f'{PENDING}: the CP-24 landing record', 'reason': reason}],
            'comparator': comparator, 'population': 'common-10747h', 'evidence_class': 'development_post_selection',
            'plan': PLAN, 'rules': ['cp24-adoption'], 'sources': list(sources), 'claim_map': CLAIMS, 'run_keys': list(run_keys),
            'style': style, 'anchor': anchor, 'checkpoint': 'CP-24', 'after': after, 'question': question, 'informed': None,
            'short': short, 'predecessor': predecessor}


def unmet_words(decision: dict, k: int) -> str:
    first = decision['first_unmet_condition']
    return f'scored attempt {k}: condition {first} of rule cp24-adoption, {RULE_WORDS[int(first)]}, was not met'


def build(root: Path) -> dict:
    root = Path(root)
    o = outcome(root)
    entries, children = [], []
    question = ('Does a literature-faithful distributional neural network, added to v4 inside LightGBM\'s third at a fixed '
                'one-sixth weight, improve on v4 jointly in point and interval accuracy?')
    for k in o['scored_attempts']:
        decision = json.loads((root / OUT / f'attempt-{k}' / 'decisions.json').read_text())['adoption']
        deciding = k == o['deciding_attempt']
        suffix = '' if deciding else f'-a{k}'
        sources = [f'reports/ddnn2/attempt-{k}/{n}' for n in ('metrics.csv', 'uncertainty.csv', 'criteria.csv', 'decisions.json')]
        note = f'scored attempt {k}'
        if decision['adopted']:
            entries.append(_entry('v5', 'v5 · DDNN-2 member added', 'v4 plus a DDNN-2 member', 'generation', [('v5', note)],
                                  'adopted in research', '', run_keys=['cp24', 'cp24/v5'], style='v5', short='v5', anchor='#v5',
                                  predecessor='v4', sources=sources))
        else:
            ident = 'ddnn2-member-on-v4' + ('' if deciding else f'-attempt-{k}')
            entries.append(_entry(ident, 'DDNN-2 member on v4' + ('' if deciding else f' (attempt {k})'),
                                  'v4 plus a DDNN-2 member, under rule cp24-adoption', 'branch', [(f'v5{suffix}', note)],
                                  'not adopted', unmet_words(decision, k),
                                  run_keys=(['cp24', 'cp24/v5'] if deciding else []), style='branch',
                                  short='DDNN-2 member on v4', anchor='#branch-ddnn2-member-on-v4' if deciding else None,
                                  after='v4', question=question, sources=sources))
        for code in ('D2', 'v3+D2'):
            ident, name, subtitle, short = ARMS[code]
            entries.append(_entry(ident + ('' if deciding else f'-attempt-{k}'), name + ('' if deciding else f' (attempt {k})'),
                                  subtitle, 'study arm', [(f'{code}{suffix}', note)], 'not adopted',
                                  'a study arm for attribution; never eligible',
                                  run_keys=([f'cp24/{code}'] if deciding else []), style='study', short=short,
                                  comparator='v4' if code == 'D2' else 'v3', sources=sources))
        if deciding:
            children = ['v5', 'D2', 'v3+D2']
    if o['kind'] == 'adopted':
        owner = 'v5'
    elif o['kind'] == 'branch':
        owner = 'ddnn2-member-on-v4'
    else:
        owner = None
    k = o['deciding_attempt']
    if o['kind'] == 'stopped_at_the_pre_fold_gate':
        note = (f'CP-24 searched a day-level, literature-faithful DDNN (DDNN-2) per fold on training data only and ran {len(o["rounds"])} '
                'pre-fold round(s), each ending in a gate on 280 pre-fold days; no round\'s gate passed, so no scored attempt ran and the '
                'five folds were not looked at. development_post_selection.')
    else:
        d = o['decisions'][k]
        note = ('CP-24 added DDNN-2, a NumPy-only day-level distributional network chosen per fold by a training-only random search, '
                'to v4 inside LightGBM\'s third: v5 = (2/3)·HG + (1/6)·L + (1/6)·DDNN-2, under the pre-registered rule cp24-adoption '
                f'(97.5% joint improvement over v4, the six original screening diagnostics, a complete evaluation, no resolved '
                f'per-fold degradation, a 0.5% practical size), in {len(o["scored_attempts"])} scored attempt(s) after '
                f'{len(o["rounds"])} pre-fold round(s): '
                + (f'v5 met all five conditions in attempt {k} and is adopted in research.' if o['kind'] == 'adopted' else
                   f'v5 is not adopted; in attempt {k} the first unmet condition is {d["first_unmet_condition"]} '
                   f'({RULE_WORDS[int(d["first_unmet_condition"])]}).') + ' development_post_selection.')
    return {
        'schema': 'cp24-draft-registry-v1',
        'status': ('draft registry entries for the publication packet; CP-24 registers nothing; publication follows the Owner\'s '
                   'landing and the Owner\'s decisions (§23.12)'),
        'outcome': o, 'verdict_sources': [f'reports/ddnn2/attempt-{x}/decisions.json' for x in o['scored_attempts']],
        'pending': PENDING,
        'pending_fields': {
            'statuses[].date': 'the adoption or not-adoption date, set at landing',
            'statuses[].source': 'the CP-24 landing record, written at landing',
            'checkpoint.evidence_sha': 'the evidence/cp-24 tag commit, created by the Owner at landing',
            'checkpoint.frozen_on': 'the evidence tag freeze date', 'checkpoint.landing': 'the landing record path',
            'model_code_sha': 'the final reviewed candidate SHA: a commit cannot contain its own SHA; it is recorded in the CP-24 '
                              'return and filled by the publication block',
            'evidence_ref': 'evidence/cp-24 at its commit', 'original_completed_utc': 'the terminal return time'},
        'model_code_sha': PENDING, 'evidence_ref': f'evidence/cp-24@{PENDING}', 'note': note,
        'checkpoint': ({'code': 'CP-24', 'run_key': 'cp24', 'owner': owner, 'evidence_tag': 'evidence/cp-24', 'evidence_sha': PENDING,
                        'frozen_on': PENDING, 'report': 'reports/ddnn2/report.md',
                        'verdict': 'docs/track-b/evidence/cp-24/integration.md', 'landing': PENDING, 'children': children}
                       if owner else None),
        'entries': entries, 'revisions': None,
        'code_additions_to_existing_entries': ([
            {'entry': 'v4', 'code': {'experiment': 'CP-24', 'code': 'HGL', 'population': 'common-10747h', 'note': 'comparator, saved'}},
            {'entry': 'v3', 'code': {'experiment': 'CP-24', 'code': 'HG', 'population': 'common-10747h', 'note': 'reference, saved'}},
            {'entry': 'naive', 'code': {'experiment': 'CP-24', 'code': 'B0', 'population': 'common-10747h', 'note': 'normalizer'}},
            {'entry': 'v1', 'code': {'experiment': 'CP-24', 'code': 'B1', 'population': 'common-10747h', 'note': 'development replay'}},
            {'entry': 'daily-lear', 'code': {'experiment': 'CP-24', 'code': 'B2', 'population': 'common-10747h', 'note': 'reference'}},
            {'entry': 'daily-lightgbm', 'code': {'experiment': 'CP-24', 'code': 'B3', 'population': 'common-10747h', 'note': 'section-8 comparator'}},
            {'entry': 'normalized-lear', 'code': {'experiment': 'CP-24', 'code': 'A1', 'population': 'common-10747h', 'note': 'reference'}},
            {'entry': 'ddnn-alone', 'code': {'experiment': 'CP-24', 'code': 'D', 'population': 'common-10747h', 'note': 'CP-23\'s DDNN, saved'}}]
            if owner else []),
        'rules': [{'id': 'cp24-adoption', 'plan': 'capstone_v21.md §23.9 (v21-r11)', 'set_on': '2026-10-05',
                   'provenance': ['capstone_v21.md v21-r11 §23.9 (ratified 2026-10-05)'] +
                                 [f'reports/ddnn2/attempt-{x}/protocol.json rule_verbatim (frozen before its fits)' for x in o['scored_attempts']]}],
        'transition': ({'title': 'From v4 to v5: adding a distributional neural network', 'predecessor': 'v4',
                        'comparator_is_predecessor': True} if o['kind'] == 'adopted' else None),
        'encoding': ({'v5': 'set by the Owner before the publication brief (PUBLISH_RULES §14; §23.12)'} if o['kind'] == 'adopted' else
                     {'note': 'no new generation; a branch, if published, uses the branch style'}),
    }


def main() -> int:
    import sys
    root = Path.cwd()
    registry = build(root)
    if '--check' in sys.argv[1:]:
        # compare as stored: JSON turns the integer round and attempt keys into strings
        same = json.loads((root / OUT / 'draft-registry.json').read_text()) == json.loads(json.dumps(registry))
        print(json.dumps({'draft_registry_identical_to_committed': same}), flush=True)
        return 0 if same else 11
    atomic(root / OUT / 'draft-registry.json', registry)
    print(json.dumps({'outcome': registry['outcome']['kind'], 'entries': [e['id'] for e in registry['entries']]}), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
