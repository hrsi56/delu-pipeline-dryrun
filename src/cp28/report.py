"""CP-28's report and draft publication packet, generated only from committed result records.

    python -m cp28.report <root>

Every number is derived from `reports/cp28/forecast/` and `reports/cp28/results/` (hash-bound by
their manifests) and from the committed r17 GFS table; nothing is typed. The packet is a local
draft under PUBLISH_RULES 1.5 §11: it grants no publication, MLflow upload or product authority.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

from . import inference, reader
from .pipeline import FORECAST, OUT, RESULTS, plain, write_json

BRIEF = 'docs/track-b/evidence/cp-28/issued-brief.md'
BRIEF_SHA256 = '57cb6d53519ede4b709face3000caca13f7358f349c9827773871ec987fb6ba3'
ANCHOR = ('capstone_v21.md', 'v21-r17', '3124451da84c3cd99fae728c371207aace6ae537474f7769d0ff6333ee53e4fc',
          '5eaaa7c8ee418e1b4dc0029ccb66c862f3b6bc7fca1bbc451c4850bd39d84d26')
PROTOCOL = ('docs/track-b/evidence/cp-26/cp28-frozen-protocol-v21-r14.md', '2b9286b0e948f59f97de33315a8b4fb3ef121003fc4ba82355b1ccf6585497a8')
SUPPLEMENT = ('docs/track-b/evidence/cp-26/cp28-gfs-postresults-addendum-v21-r17.md',
              '39f322afa9717c5cfb0b626a1a8430fa9f5fcc0dfdd159123b9d4b48e7db03b4')
RULINGS = {'docs/track-b/cp-25-rulings-2026-10-12.md': 'c7af67f88157277aafd8dd2b1b49e12a1d9ff4b747ca5bdf178fac664fba53c1',
           'docs/track-b/cp-26-gfs-ruling-2026-10-19.md': '3b97ff6abdf9a30ddd8c950cf3c459301d757aa1af08f6d8121bc73f92ba2373'}
PUBLISH_RULES = ('docs/PUBLISH_RULES.md', '1.5', '1bc0c52351656c5a499291bf15ccb82b0ae8b4e6475d1879b0a4d307daaef94e')
TEMPLATE = ('docs/track-b/publication-packet-template.md', '4049c3e1f695034ec45fa185d465fce05ab1183cbded72188306579d7287d01a')
RULES = {'A1': 'DDNN-2 alone, with its own Johnson SU quantiles',
         'A2': 'equal weights (1/3, 1/3, 1/3), its own H layer',
         'A3': 'global simplex-grid MAE fit on the latest ≤728 dates (min 364), deployed 0.5·fit + 0.5·w0',
         'A4': 'per-block grid fits (min 728), deployed 0.25·block + 0.75·A3 deployed; before 728 = A3',
         'A5': 'NumPy 16→8→3 meta-learner (min 364), deployed 0.5·network + 0.5·w0',
         'A6': 'A3\'s fit unshrunk (min 364)',
         'A7': 'A5\'s network unshrunk (min 364)'}


def load(root: Path):
    r = lambda p: json.loads((root / p).read_text())
    for directory in (FORECAST, RESULTS):
        manifest = r(f'{directory}/manifest.json')
        for entry in manifest['files']:
            if reader.sha256(root / entry['path']) != entry['sha256']:
                raise SystemExit(f'{entry["path"]} differs from its manifest')
    return {'root': root, 'forecast_manifest': r(f'{FORECAST}/manifest.json'), 'results_manifest': r(f'{RESULTS}/manifest.json'),
            'cross': r(f'{FORECAST}/cross-checks.json'), 'records': pd.read_parquet(root / FORECAST / 'arm-days.parquet'),
            'fits': pd.read_parquet(root / FORECAST / 'fits.parquet'), 'selection': r(f'{RESULTS}/selection.json'),
            'compound': r(f'{RESULTS}/compound.json'), 'holm': pd.read_csv(root / RESULTS / 'holm.csv'),
            'endpoints': pd.read_parquet(root / RESULTS / 'endpoints.parquet'), 'descriptive': r(f'{RESULTS}/descriptive.json'),
            'bulk': r(f'{RESULTS}/bulk-manifest.json'), 'daily': pd.read_parquet(root / RESULTS / 'daily-losses.parquet'),
            'gfs': pd.read_csv(root / 'docs/track-b/evidence/cp-28/gfs-postresults/table.csv')}


def pct(ratio):
    return f'{(ratio - 1) * 100:+.1f}%'


def ci_pct(ci):
    if not ci or ci.get('status') != 'computable':
        return 'undefined'
    return f'{(ci["ci95"][0] - 1) * 100:+.1f}% to {(ci["ci95"][1] - 1) * 100:+.1f}%'


def pval(p):
    return '1' if p >= 1 else (f'{p:.4f}' if p >= 1e-4 else f'{p:.1e}')


def startup(fits: pd.DataFrame) -> dict:
    """Observed learning support: the first origin with ≥364 and ≥728 released learning dates."""
    out = {}
    for label, k in (('minimum_364', 364), ('full_728', 728)):
        rows = fits[fits.learning_dates >= k]
        if len(rows):
            first = rows.iloc[0]
            out[label] = {'first_origin': first.day, 'learning_oldest': first.learning_oldest, 'learning_latest': first.learning_latest,
                          'calendar_span_days': int(first.learning_span_days)}
        else:
            out[label] = None
    out['first_learning_date'] = fits[fits.learning_dates > 0].iloc[0].learning_oldest if (fits.learning_dates > 0).any() else None
    last = fits.iloc[-1]
    out['final_window'] = {'origin': last.day, 'dates': int(last.learning_dates), 'oldest': last.learning_oldest,
                           'latest': last.learning_latest, 'calendar_span_days': int(last.learning_span_days)}
    return out


def summary(d) -> dict:
    sel, desc = d['selection'], d['descriptive']
    selected = sel['selected']
    gfs = d['gfs']
    g = gfs[(gfs.section == 'differential') & (gfs.model == 'v5') & (gfs.population == 'all_eligible') & (gfs.slice == 'overall')]
    gfs_head = {m: {'nominal': float(r.nominal_mean), 'gfs6': float(r.gfs6_mean), 'change_percent': float(r.change_percent),
                    'ci95_ratio': [float(r.ratio_ci95_low), float(r.ratio_ci95_high)], 'n': int(r.n_pairs)}
                for m, r in zip(g.metric, g.itertuples())}
    holm = d['holm']
    decisions = {v: {row.arm: {'compound_p': float(row.p), 'adjusted_p': float(row.adjusted_p), 'rejected': bool(row.rejected)}
                     for row in holm[holm.variant == v].itertuples()} for v in holm.variant.unique()}
    fallback = {arm: {'scored_dates': f['scored_dates'], 'fallback_dates': f['fallback_dates'], 'by_source': f['by_source']}
                for arm, f in desc['fallbacks'].items()}
    candidate = None
    if selected != 'control':
        a = desc['arms'][selected]
        candidate = {m: {'arm': a[m]['arm'], 'control': a[m]['control'], 'B0': a[m]['B0'],
                         'ratio_to_control': a[m]['ratio_to_control'], 'ratio_to_control_ci95': a[m].get('ratio_to_control_ci95'),
                         'ratio_to_B0': a[m]['ratio_to_B0'], 'ratio_to_B0_ci95': a[m].get('ratio_to_B0_ci95')} for m in ('MAE', 'WIS')}
    any_arm = next(iter(desc['arms'].values()))
    control = {m: {'control': any_arm[m]['control'], 'B0': any_arm[m]['B0'], 'control_over_B0': any_arm[m]['control'] / any_arm[m]['B0'],
                   'paired_dates': any_arm['paired_dates']} for m in ('MAE', 'WIS')}
    return {'selected': selected, 'selection': {k: sel[k] for k in ('tie_set', 'best_arm', 'best_wis', 'reason')},
            'candidate_scores': candidate, 'control_scores_on_A1_pairs': control,
            'passes_all_four': sel['passes_all_four'], 'primary_rejections': sel['primary_rejections'],
            'inconclusive': sel['inconclusive_primary_pass_failed_variant'], 'holm': decisions,
            'startup': startup(d['fits']), 'fallbacks': fallback, 'gfs_note_headline': gfs_head}


def render(d) -> str:
    s = summary(d)
    desc, sel = d['descriptive'], d['selection']
    holm, endpoints = d['holm'], d['endpoints']
    fm, rm = d['forecast_manifest'], d['results_manifest']
    L = ['# CP-28 report — 4.8 on stored forecasts: seven arms against the control', '',
         f'**Evidence class:** `development_post_selection`; 4.7T carries the protection. **Authority:** `{ANCHOR[0]}` {ANCHOR[1]} '
         f'§24.8 (anchor `{ANCHOR[2]}`, section `{ANCHOR[3]}`), the frozen protocol `{PROTOCOL[0]}` (`{PROTOCOL[1]}`), the issued '
         f'brief `{BRIEF}` (`{BRIEF_SHA256}`). CP-28\'s Holm control is conditional on the Owner\'s decision to run it, which '
         'followed CP-26\'s report (which already showed DDNN-2 alone, A1\'s forecasts, on this calendar).', '']
    L += ['## The decision', '']
    if s['selected'] == 'control':
        L.append('**No arm passes. The control, nominal v5 (with its load forecast), is the candidate for the freeze.**')
    else:
        L.append(f'**{s["selected"]} is selected** ({RULES[s["selected"]]}): {s["selection"]["reason"]}.')
    L += ['', f'Passing all four analyses: {", ".join(a for a, v in s["passes_all_four"].items() if v) or "none"}. '
          f'Primary Holm rejections: {", ".join(a for a, v in s["primary_rejections"].items() if v) or "none"}. '
          f'Inconclusive (primary pass, failed a variant): {", ".join(s["inconclusive"]) or "none"}.', '']
    L += ['## Losses against the control and B0 (paired dates; descriptive intervals)', '',
          '| arm | rule | dates | MAE | vs control | 95% | vs B0 | WIS | vs control | 95% | vs B0 |', '|---|---|---|---|---|---|---|---|---|---|---|']
    for arm in inference.ARMS:
        a = desc['arms'][arm]
        L.append(f'| {arm} | {RULES[arm]} | {a["paired_dates"]} | {a["MAE"]["arm"]:.3f} | {pct(a["MAE"]["ratio_to_control"])} | '
                 f'{ci_pct(a["MAE"].get("ratio_to_control_ci95"))} | {pct(a["MAE"]["ratio_to_B0"])} | {a["WIS"]["arm"]:.3f} | '
                 f'{pct(a["WIS"]["ratio_to_control"])} | {ci_pct(a["WIS"].get("ratio_to_control_ci95"))} | {pct(a["WIS"]["ratio_to_B0"])} |')
    c = s['control_scores_on_A1_pairs']
    L += ['', f'Control on the same {c["MAE"]["paired_dates"]} dates: MAE {c["MAE"]["control"]:.3f} EUR/MWh '
          f'({pct(c["MAE"]["control_over_B0"])} vs B0 {c["MAE"]["B0"]:.3f}); WIS {c["WIS"]["control"]:.3f} '
          f'({pct(c["WIS"]["control_over_B0"])} vs B0 {c["WIS"]["B0"]:.3f}). Intervals are studentized, from each arm\'s own '
          'primary overall-endpoint resamples; they are descriptive, never a substitute for the registered 0.995/1.02 tests.', '']
    L += ['## The compound test and Holm, in all four analyses', '',
          '| analysis | ' + ' | '.join(inference.ARMS) + ' |', '|---|' + '---|' * len(inference.ARMS)]
    for variant in holm.variant.unique():
        part = holm[holm.variant == variant].set_index('arm')
        L.append(f'| {variant} | ' + ' | '.join(f'p={pval(part.loc[a, "p"])}, adj {pval(part.loc[a, "adjusted_p"])}'
                                                  f'{" **rejected**" if part.loc[a, "rejected"] else ""}' for a in inference.ARMS) + ' |')
    L += ['', 'Holm sorts the seven compound p-values (ties by arm number) and compares position j with 0.05/(8−j), stopping at '
          'the first failure; failed or unexecuted arms stay in the family at p = 1. An arm passes only if Holm rejects it in the '
          'primary 14-day analysis and in each of the 7-day, 28-day and unstratified 14-day variants.', '']
    L += ['## The twenty endpoints (primary analysis)', '',
          '| arm | compound p | first endpoint at the maximum | ' + ' | '.join(e['name'] for e in inference.endpoints()) + ' |',
          '|---|---|---|' + '---|' * 20]
    prim = endpoints[endpoints.variant == 'primary14']
    for arm in inference.ARMS:
        part = prim[prim.arm == arm].sort_values('endpoint')
        compound = float(part.p.max())
        first = part[part.p == compound].iloc[0]['name']
        L.append(f'| {arm} | {pval(compound)} | {first} | ' + ' | '.join(pval(p) for p in part.p) + ' |')
    status = endpoints.groupby(['variant', 'status']).size().reset_index(name='n')
    L += ['', 'Endpoint statuses across all four analyses: ' + '; '.join(f'{r.variant} {r.status} {r.n}' for r in status.itertuples()) + '.',
          'Each endpoint\'s mean differential d = L_arm − 0.995·L_v5 (improvement) or L_arm − 1.02·L_v5 (non-inferiority) is in '
          '`results/endpoints.parquet`, with n, HAC variance, statistic, lower-tail count and status.', '']
    L += ['## Selection', '', f'Passing arms\' paired WIS: ' + (', '.join(f'{a} {sel["wis_paired_with_control"][a]:.4f}' for a in inference.ARMS
                                                                       if sel['passes_all_four'][a]) or 'none') + '.',
          f'Best: {sel["best_arm"]}; tie bound {sel.get("tie_bound")}; tie set {sel["tie_set"]}; selected **{sel["selected"]}** '
          '(simplicity order A1, A2, A3, A6, A4, A5, A7).', '']
    st = s['startup']
    L += ['## Startup, support and equivalence to v5', '',
          f'- First CP-28 learning date (first complete all-member-valid released date): {st["first_learning_date"]}.',
          f'- 364 released learning dates first reached at origin {st["minimum_364"]["first_origin"] if st["minimum_364"] else "never"}'
          + (f' (window {st["minimum_364"]["learning_oldest"]}..{st["minimum_364"]["learning_latest"]}, span {st["minimum_364"]["calendar_span_days"]} days)' if st['minimum_364'] else '')
          + '. Before it, A3, A5, A6 and A7 issue with v5\'s weights (their own H layers), and A4 equals A3.',
          f'- 728 first reached at origin {st["full_728"]["first_origin"] if st["full_728"] else "never"}'
          + (f' (window {st["full_728"]["learning_oldest"]}..{st["full_728"]["learning_latest"]}, span {st["full_728"]["calendar_span_days"]} days)' if st['full_728'] else '')
          + '. Until then A4 equals A3\'s deployed weights, so A4\'s early cells test A3\'s weights to that extent.',
          f'- Final window at {st["final_window"]["origin"]}: {st["final_window"]["dates"]} dates, {st["final_window"]["oldest"]}..{st["final_window"]["latest"]} '
          f'({st["final_window"]["calendar_span_days"]} days). Activation follows actual support; no date was promised.', '']
    scored_startup = d['records'][d['records'].scored & (d['records'].scoring_keys > 0)]
    L += ['| arm | scored dates at v5\'s startup weights | of which native | ' + 'weight rules on scored dates |', '|---|---|---|---|']
    for arm in inference.ARMS:
        part = scored_startup[scored_startup.arm == arm]
        startup_rows = part[part.weight_rule == 'startup_w0']
        L.append(f'| {arm} | {len(startup_rows)} | {int((startup_rows.output_source == "native").sum())} | '
                 + ', '.join(f'{k} {v}' for k, v in part.weight_rule.value_counts().items()) + ' |')
    L += ['', '## Fallbacks, by arm, source and reason (scored dates)', '', '| arm | scored | fallback | by source | by reason |', '|---|---|---|---|---|']
    for arm in inference.ARMS:
        f = desc['fallbacks'][arm]
        L.append(f'| {arm} | {f["scored_dates"]} | {f["fallback_dates"]} | {f["by_source"]} | {f["by_reason"]} |')
    fallback_days = sorted(set(d['records'][d['records'].fallback & d['records'].scored].day))
    if (d['root'] / reader.RUN / 'scores/daily.parquet').exists():
        cp26 = reader.read_table(d['root'], None, 'daily')
        v5 = cp26[cp26.stream == 'nominal/v5'].set_index('delivery_date')
        own = v5.loc[[pd.Timestamp(x).date() for x in fallback_days], 'output_source'].value_counts().to_dict() if fallback_days else {}
    else:
        own = 'unavailable: CP-26\'s committed daily table is absent from this tree'
    L += ['', f'On the {len(fallback_days)} fallback dates ({fallback_days[0] if fallback_days else "—"} … {fallback_days[-1] if fallback_days else "—"}), '
          f'the control\'s own emitted forecast, as CP-26 scored it, was: {own} (CP-26 `scores/daily.parquet`). The copies are scored as emitted.']
    L += ['', 'A missing member makes every arm use the control\'s complete forecast that day (then B0\'s); a solver failure uses A2\'s '
          'complete forecast; an H-history shortage, missing scale or failed H construction uses the control, then B0. Copied '
          'forecasts are scored as emitted and never enter an H buffer or the learning ledger.', '']
    L += ['## Descriptive breakdowns (point estimates; no gate)', '']
    for metric in ('MAE', 'WIS'):
        L += [f'**{metric}, ratio to the control by mask, year and season.**', '',
              '| arm | ' + ' | '.join(['all', 'never scored', 'old folds', 'newly opened']) + ' | '
              + ' | '.join(sorted(desc['arms']['A1'][metric]['year'])) + ' | ' + ' | '.join(['DJF', 'MAM', 'JJA', 'SON']) + ' |',
              '|---|' + '---|' * (4 + len(desc['arms']['A1'][metric]['year']) + 4)]
        for arm in inference.ARMS:
            a = desc['arms'][arm][metric]
            masks = [a['masks'][m]['ratio_to_control'] for m in ('all_eligible', 'never_scored', 'old_fold_2_5', 'newly_opened')]
            years = [a['year'][y]['ratio_to_control'] for y in sorted(a['year'])]
            seasons = [a['season'].get(x, {}).get('ratio_to_control') for x in ('DJF', 'MAM', 'JJA', 'SON')]
            L.append(f'| {arm} | ' + ' | '.join(pct(v) if v is not None else '—' for v in masks + years + seasons) + ' |')
        L.append('')
    daily = d['daily']
    cov = daily[daily.stream.isin(inference.ARMS)].groupby('stream')[['coverage50', 'coverage80', 'coverage95', 'width50', 'width80', 'width95']].mean()
    L += ['**Calibration (mean daily coverage and width, EUR/MWh; descriptive).**', '',
          '| arm | cov 50 | cov 80 | cov 95 | width 50 | width 80 | width 95 |', '|---|---|---|---|---|---|---|']
    for arm in inference.ARMS:
        r = cov.loc[arm]
        L.append(f'| {arm} | {r.coverage50:.3f} | {r.coverage80:.3f} | {r.coverage95:.3f} | {r.width50:.1f} | {r.width80:.1f} | {r.width95:.1f} |')
    gh = s['gfs_note_headline']
    L += ['', '## The r17 GFS note (separate; outside this family)', '',
          f'Committed before any arm ran (`docs/track-b/evidence/cp-28/gfs-postresults/`). On {gh["MAE"]["n"]} paired dates GFS+6 worsens nominal '
          f'v5 MAE {gh["MAE"]["change_percent"]:+.1f}% (registered 95% {pct(gh["MAE"]["ci95_ratio"][0])} to {pct(gh["MAE"]["ci95_ratio"][1])}) and WIS '
          f'{gh["WIS"]["change_percent"]:+.1f}% ({pct(gh["WIS"]["ci95_ratio"][0])} to {pct(gh["WIS"]["ci95_ratio"][1])}). The weather-independent v2 is about '
          'level with the delayed-weather route on CP-25\'s affected dates and far behind the on-time route; the note does not '
          'recommend an availability-switching experiment on present evidence. It changes no arm, control or rule here.', '']
    cross = d['cross']
    L += ['## Disclosures', '',
          '- **Evidence class.** Development, post-selection. CP-28 cannot establish performance on unseen data; 4.7T carries the '
          'protection. A pass is a research selection, not a confirmation.',
          '- **The control** is nominal v5 as landed, including its day-ahead load forecast: tier 3 under the Owner\'s recorded '
          'exception (CP-25 rulings §1), with the historical-revision risk disclosed. The target price is tier 2 with SMARD\'s '
          '"update possible" caveat (§2). Prices +24 h, GFS +24 h and load +1 h are accepted structural timing risks (§3), to be '
          'measured in CP-18. GFS+6 is **fragile** (Owner ruling 2026-10-19).',
          '- **A6 and A7 are unshrunk** (Owner, 2026-10-12): they may place any weight, up to 100%, on one member, and are the arms '
          'most exposed to overfitting and to lag at regime changes. The gates, the simplicity order and 4.7T are their protection.',
          '- **Minimum-history equivalence.** Learned arms equal v5\'s weights until 364 released learning dates exist, and A4 equals '
          'A3 until 728. These are native rules, labelled `startup_w0`/`a3_deployed`, never fallbacks.',
          f'- **Identity cross-checks.** CP-28\'s learning dates equal CP-26\'s stored learning ledger: {cross.get("learning_rows")}. '
          f'Issue-time states against CP-26\'s table: {cross["issue_states"]}.',
          '- **Bootstrap degeneracies and missing cells** are listed by status above; a deficient or uncomputable endpoint counts '
          'with p = 1 and consumes no draw.',
          '- **No later look.** Nothing was rerun, re-specified or added after the first score.', '']
    L += ['## Reproduction and retention', '',
          f'- Forecast stage: `python -m cp28.pipeline forecast <root>` (manifest `reports/cp28/forecast/manifest.json`, {fm["origins"]} origins, '
          f'{fm["issues"]} arm-days, wall {fm["wall_seconds"]} s).',
          f'- Score stage: `python -m cp28.pipeline score <root>` (manifest `reports/cp28/results/manifest.json`, wall {rm["wall_seconds"]} s).',
          f'- Bulk retention, local only: {len(d["bulk"]["files"])} files under `{d["bulk"]["location"]}`, each listed with size and SHA-256 in '
          '`reports/cp28/results/bulk-manifest.json` (every computable endpoint\'s block starts and replicate statistics; the descriptive ratio pivots).',
          '- Origin-level weights, shared-fit identifiers, window dates, scaler statistics and parameters, feature support, issue-time '
          'states, native centrals, emitted quantiles, statuses and H-buffer provenance are in `reports/cp28/forecast/`.', '']
    L += ['## Resources', '', 'Machine time, peak memory and every ledgered job are in `docs/track-b/evidence/cp-28/resources.json` '
          '(CP-28\'s 40-machine-hour ceiling; one worker per job, BLAS 1, CPU only, no cloud or GPU).', '']
    return '\n'.join(L)


def packet(d) -> tuple[str, dict]:
    s = summary(d)
    desc = d['descriptive']
    registry = []
    for arm in inference.ARMS:
        registry.append({'id': f'cp28-{arm.lower()}', 'name': f'{arm} · {RULES[arm]}', 'kind': 'study arm',
                         'codes': [f'cp-28: {arm} (cp26-nominal-scored-calendar)'],
                         'statuses': [f'{"selected" if s["selected"] == arm else ("passed, not selected" if s["passes_all_four"][arm] else "not adopted")}, '
                                      'CP-28 one look, reports/cp28/results/selection.json'],
                         'comparator': 'nominal v5 (the control)', 'population': 'cp26-nominal-scored-calendar (CP-25 frozen eligible keys)',
                         'evidence_class': 'development_post_selection', 'plan': 'capstone v21-r17 §24.8', 'rules': 'cp28-stored-forecasts-v21-r14',
                         'sources': ['reports/cp28/results/descriptive.json', 'reports/cp28/results/holm.csv', 'reports/cp28/results/endpoints.parquet'],
                         'claim_map': 'reports/cp28/publication-packet.json#claims', 'run_keys': [f'cp-28/{arm}'],
                         'style': 'Owner decision', 'checkpoint': 'CP-28', 'predecessor': None})
    registry.append({'id': 'cp26-nominal-v5', 'name': 'nominal v5 (control)', 'kind': 'control', 'evidence_class': 'development_post_selection',
                     'statuses': ['retained control' if s['selected'] == 'control' else 'comparator']})
    registry.append({'id': 'cp26-nominal-b0', 'name': 'B0 similar-day naive', 'kind': 'reference', 'evidence_class': 'development_post_selection'})
    permitted = ['On CP-26\'s development calendar, under a one-look protocol registered before CP-26\'s first score, '
                 + ('no recombination arm met both the improvement and every non-inferiority endpoint after Holm in all four analyses; '
                    'the control is retained as the freeze candidate.' if s['selected'] == 'control' else
                    f'{s["selected"]} passed the compound gates after Holm in all four analyses and was selected.'),
                 'Every arm\'s losses, ratios to the control and to B0, with descriptive intervals, are reported for the same paired dates.']
    withheld = ['performance on unseen data or live eligibility (development, post-selection)', 'a causal contribution of any member',
                'equivalence from a null result', 'economic or trading value', 'robustness to late weather (GFS+6 is fragile)']
    headline = {'verdict': 'met' if s['selected'] != 'control' else 'not met by any arm',
                'rule': '§24.8 gates: improvement 0.995 on MAE and WIS; non-inferiority 1.02 in every complete year 2021–2025 and pooled season; '
                        'compound max; Holm across seven at 0.05; all three sensitivity variants. Registered 2026-10-12 (anchor) and in '
                        'CP-28\'s frozen protocol before CP-26\'s first score.',
                'eligible_policy_count': 7, 'point_comparison_label_needed': False,
                'comparator': 'nominal v5', 'interval_method': 'studentized paired non-circular moving blocks, 14 days, 10,000 replicates, '
                                                               'seed 20261012, Bartlett HAC 14, from CP-28\'s own draws'}
    arms_table = {arm: {m: {k: desc['arms'][arm][m][k] for k in ('arm', 'control', 'B0', 'ratio_to_control', 'ratio_to_B0')}
                        | {'ratio_to_control_ci95': (desc['arms'][arm][m].get('ratio_to_control_ci95') or {}).get('ci95')}
                        for m in ('MAE', 'WIS')} for arm in inference.ARMS}
    value = {'status': 'DRAFT_LOCAL', 'authority': 'none: no publication, MLflow upload or product activation',
             'identity': {'checkpoint': 'CP-28', 'brief': [BRIEF, BRIEF_SHA256], 'plan': list(ANCHOR), 'protocol': list(PROTOCOL),
                          'supplement': list(SUPPLEMENT), 'rulings': RULINGS, 'publish_rules': list(PUBLISH_RULES), 'template': list(TEMPLATE),
                          'evidence_tag': 'awaits the Owner\'s LAND', 'final_candidate': 'recorded in the checkpoint return'},
             'registry': registry, 'claims': {'permitted': permitted, 'withheld': withheld}, 'headline': headline,
             'arms': arms_table, 'owner_summary': s,
             'slots': {'question': 'Does a registered recombination of v5\'s three members beat v5 on both primary scores without '
                                   'losing more than 2% in any complete year or season?',
                       'change': 'Seven pre-registered weightings of c_HG, L and D2 (or D2 alone), each with its own H interval layer.',
                       'headline': 'Paired differences against the control on MAE and WIS with 95% intervals, per arm.',
                       'reading': permitted[0],
                       'does_not_establish': withheld[:3],
                       'decision': f'{s["selected"]} is the CP-17 freeze candidate (CP-28 one look).'},
             'transition': 'none: CP-28 adopts no generation; the selection is a freeze candidate for CP-17',
             'product_documentation': 'inapplicable: the released model does not change in CP-28',
             'chart_routes': 'none drafted: publication is a separate decision',
             'final_product_fields': 'unmet trigger: no final-product designation in CP-28',
             'mlflow': {'export': 'reports/cp28/MLflow-export-draft.json', 'status': 'local draft, never uploaded'},
             'checks': ['tests/cp28 (causal controls, units, pipeline, GFS note)', 'scorer reproduces CP-26 v5/B0 losses',
                        'forecast and results manifests hash-checked'],
             'completion': 'INCOMPLETE: editorial, fresh-reader and independent publication review not performed; no public surface changed'}
    md = ['# CP-28 draft publication packet', '', 'DRAFT_LOCAL. This packet grants no publication, MLflow upload or product activation authority. '
          f'It follows `{TEMPLATE[0]}` (`{TEMPLATE[1]}`) under PUBLISH_RULES {PUBLISH_RULES[1]} (`{PUBLISH_RULES[2]}`) §11.', '',
          '## Identity', '', f'- Checkpoint CP-28, brief `{BRIEF}` (`{BRIEF_SHA256}`).', f'- Plan `{ANCHOR[0]}` {ANCHOR[1]} §24.8.',
          '- Evidence tag and final candidate: recorded in the checkpoint return; the tag awaits the Owner\'s LAND.', '',
          '## Registry (draft)', '', '| id | kind | name | status |', '|---|---|---|---|']
    for r in registry:
        md.append(f'| {r["id"]} | {r["kind"]} | {r["name"]} | {"; ".join(r.get("statuses", [])) or "—"} |')
    md += ['', '## Claims', '', '**Permitted:**', *[f'- {c}' for c in permitted], '', '**Withheld:**', *[f'- {c}' for c in withheld], '',
           '## Headline quantities', '', f'- Verdict: {headline["verdict"]}. Eligible policies under the rule: 7. Comparator: nominal v5.',
           f'- Rule: {headline["rule"]}', '', '| arm | MAE vs control | 95% | WIS vs control | 95% |', '|---|---|---|---|---|']
    for arm in inference.ARMS:
        a = desc['arms'][arm]
        md.append(f'| {arm} | {(a["MAE"]["ratio_to_control"] - 1) * 100:+.0f}% | {ci_whole(a["MAE"].get("ratio_to_control_ci95"))} | '
                  f'{(a["WIS"]["ratio_to_control"] - 1) * 100:+.0f}% | {ci_whole(a["WIS"].get("ratio_to_control_ci95"))} |')
    md += ['', 'Percentages are whole numbers and EUR/MWh one decimal on any reading path (PUBLISH_RULES §3.4); full precision is in '
           '`publication-packet.json` and `reports/cp28/report.md`.', '',
           '## Slots, transition and product fields', '', f'- Question: {value["slots"]["question"]}', f'- Decision: {value["slots"]["decision"]}',
           f'- Transition: {value["transition"]}.', f'- Product documentation: {value["product_documentation"]}.',
           f'- Final-product fields: {value["final_product_fields"]}.', '',
           '## MLflow and completion', '', f'- Export: `{value["mlflow"]["export"]}`, {value["mlflow"]["status"]}.', f'- {value["completion"]}.', '']
    return '\n'.join(md), value


def ci_whole(ci):
    if not ci or ci.get('status') != 'computable':
        return 'undefined'
    return f'{(ci["ci95"][0] - 1) * 100:+.0f}% to {(ci["ci95"][1] - 1) * 100:+.0f}%'


def mlflow_export(d) -> dict:
    """Deterministic local draft: one run per arm with its rule and descriptive metrics. Never uploaded."""
    desc = d['descriptive']
    runs = []
    for arm in inference.ARMS:
        a = desc['arms'][arm]
        runs.append({'run_key': f'cp-28/{arm}', 'params': {'rule': RULES[arm], 'protocol': PROTOCOL[1]},
                     'metrics': {f'{m}_{k}': a[m][k] for m in ('MAE', 'WIS') for k in ('arm', 'ratio_to_control', 'ratio_to_B0')},
                     'tags': {'checkpoint': 'CP-28', 'evidence_class': 'development_post_selection',
                              'passes_all_four': str(d['selection']['passes_all_four'][arm]), 'selected': str(d['selection']['selected'] == arm)}})
    return {'schema': 'cp28-mlflow-export-draft-v1', 'status': 'local draft, never uploaded', 'experiment': 'cp-28-stored-forecasts', 'runs': runs}


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    root = Path(argv[0]).resolve()
    d = load(root)
    (root / OUT / 'report.md').write_text(render(d), encoding='utf-8')
    md, value = packet(d)
    (root / OUT / 'publication-packet.md').write_text(md, encoding='utf-8')
    write_json(root / OUT / 'publication-packet.json', value)
    write_json(root / OUT / 'MLflow-export-draft.json', mlflow_export(d))
    write_json(root / OUT / 'owner-summary.json', summary(d))
    print('wrote report, packet, export draft and owner summary')


if __name__ == '__main__':
    main()
