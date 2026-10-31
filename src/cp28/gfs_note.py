"""The r17 post-results GFS note: descriptive, from CP-26's stored outputs only.

Contract: `docs/track-b/evidence/cp-26/cp28-gfs-postresults-addendum-v21-r17.md` (39f322af...),
under the Owner's ruling `docs/track-b/cp-26-gfs-ruling-2026-10-19.md`. No fit, search, replay,
recalibration, new bootstrap or test: losses come from CP-26's daily and hourly score tables, and
uncertainty only from CP-26's registered GFS+6 v5 endpoints where the comparison and population
are identical. It runs before any CP-28 arm is fit or scored and is outside the seven-arm family.

    python -m cp28.gfs_note <repo-root> <output-dir>
"""
from __future__ import annotations

from datetime import date
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

from . import reader

MASKS = ('all_eligible', 'never_scored', 'old_fold_2_5', 'newly_opened')
METRICS = ('MAE', 'WIS')
MODELS = ('v4', 'v5')
MIXTURE_P = (0.0, 0.01, 0.05, 0.10, 0.20, 0.50, 1.0)
BLOCKS = {'night': (22, 23, 0, 1, 2, 3, 4, 5), 'solar': tuple(range(10, 17)),
          'shoulder_peak': (6, 7, 8, 9, 17, 18, 19, 20, 21)}
SUPPLEMENT = 'docs/track-b/evidence/cp-26/cp28-gfs-postresults-addendum-v21-r17.md'
SUPPLEMENT_SHA256 = '39f322afa9717c5cfb0b626a1a8430fa9f5fcc0dfdd159123b9d4b48e7db03b4'
RULING = 'docs/track-b/cp-26-gfs-ruling-2026-10-19.md'
RULING_SHA256 = '3b97ff6abdf9a30ddd8c950cf3c459301d757aa1af08f6d8121bc73f92ba2373'
MARGIN_DAYS = 'reports/development-inputs/margin-days.csv'


def block_of(hour: int) -> str:
    for name, hours in BLOCKS.items():
        if hour in hours:
            return name
    raise ValueError(f'local hour {hour} has no block')


def load(root: Path, local: Path):
    reader.require(root / SUPPLEMENT, SUPPLEMENT_SHA256, 'GFS supplement')
    reader.require(root / RULING, RULING_SHA256, 'GFS ruling')
    daily = reader.read_table(root, local, 'daily')
    states = reader.read_table(root, local, 'known_price_states')[['delivery_date', 'volatility_unknown']]
    hourly = reader.read_table(root, local, 'hourly')
    margins = pd.read_csv(reader.cp25_file(root, MARGIN_DAYS))
    gfs = margins[margins.group == 'gfs'][['delivery_date', 'affected_plus_6h']].copy()
    gfs['delivery_date'] = [date.fromisoformat(v) for v in gfs.delivery_date]
    if gfs.delivery_date.duplicated().any():
        raise ValueError('duplicate GFS classification dates')
    report = reader.report_json(local)
    return daily, states, hourly, gfs, report


def wide(daily: pd.DataFrame, states: pd.DataFrame, gfs: pd.DataFrame) -> pd.DataFrame:
    """One row per delivery date; loss columns per stream; NaN where a stream has no scored loss."""
    keep = ['nominal/v2', 'nominal/v4', 'nominal/v5', 'gfs6/v4', 'gfs6/v5']
    rows = daily[daily.stream.isin(keep)]
    calendar = rows[rows.stream == 'nominal/v5'][['delivery_date', 'year', 'season', 'negative_price_risk',
                                                 'high_volatility', *MASKS]].set_index('delivery_date')
    for stream in keep:
        part = rows[rows.stream == stream].set_index('delivery_date')
        usable = part.eligible & part.available & ~part.failed
        for metric in METRICS:
            calendar[f'{stream}:{metric}'] = part[metric].where(usable)
        calendar[f'{stream}:source'] = part.output_source
    calendar = calendar.join(states.set_index('delivery_date'), how='left')
    calendar = calendar.join(gfs.set_index('delivery_date'), how='left')
    if calendar.affected_plus_6h.isna().any():
        raise ValueError('a scored date lacks its CP-25 GFS+6 classification')
    calendar['volatility_state'] = np.where(calendar.volatility_unknown == 1, 'unknown',
                                            np.where(calendar.high_volatility == 1, 'high', 'not_high'))
    calendar['gfs6_affected'] = np.where(calendar.affected_plus_6h.astype(bool), 'affected', 'unaffected')
    return calendar.sort_index()


def check_hourly_aggregation(daily: pd.DataFrame, hourly: pd.DataFrame) -> dict:
    """The stored daily losses must be the equal-hour means of the stored hourly losses."""
    keep = ['nominal/v2', 'nominal/v4', 'nominal/v5', 'gfs6/v4', 'gfs6/v5']
    h = hourly[hourly.stream.isin(keep)]
    means = h.groupby(['stream', 'delivery_date'])[list(METRICS)].mean()
    counts = h.groupby(['stream', 'delivery_date']).size()
    d = daily[daily.stream.isin(keep) & daily.eligible & daily.available & ~daily.failed].set_index(['stream', 'delivery_date'])
    joined = d[[*METRICS, 'scoring_hours']].join(means, rsuffix='_hourly', how='left')
    joined['hours'] = counts.reindex(joined.index)
    worst = {m: float(np.nanmax(np.abs(joined[m] - joined[f'{m}_hourly']) / np.maximum(1.0, np.abs(joined[m])))) for m in METRICS}
    if joined[[f'{m}_hourly' for m in METRICS]].isna().any().any() or (joined.hours != joined.scoring_hours).any() \
            or max(worst.values()) > 1e-12:
        raise ValueError(f'hourly scores do not aggregate to the daily table: {worst}')
    return {'rows_checked': int(len(joined)), 'max_relative_difference': worst, 'hours_match_scoring_hours': True}


def block_losses(hourly: pd.DataFrame, stream: str) -> pd.DataFrame:
    part = hourly[hourly.stream == stream].copy()
    part['block'] = [block_of(int(h)) for h in part.local_hour]
    return part.groupby(['delivery_date', 'block'])[list(METRICS)].mean()


def summary(left: pd.Series, right: pd.Series) -> dict:
    """Paired mean losses: left = nominal, right = GFS+6 (or any reference)."""
    pair = pd.concat([left.rename('N'), right.rename('S')], axis=1)
    complete = pair.dropna()
    n, missing = len(complete), int(len(pair) - len(complete))
    if not n:
        return {'n_pairs': 0, 'missing_pairs': missing}
    N, S = float(complete.N.mean()), float(complete.S.mean())
    return {'n_pairs': n, 'missing_pairs': missing, 'nominal_mean': N, 'gfs6_mean': S, 'difference': S - N,
            'ratio': S / N if N else np.nan, 'change_percent': (S / N - 1) * 100 if N else np.nan}


def registered_ci(report: dict) -> dict:
    """CP-26's registered primary-14 GFS+6 v5 endpoints, keyed by (mask, group, value, metric)."""
    found = {}
    for row in report['inference_endpoints']:
        if row.get('kind') == 'paired_comparison' and row.get('left') == 'gfs6/v5' and row.get('right') == 'nominal/v5' \
                and row.get('variant') == 'primary14':
            found[(row['mask'], row['group'], str(row['value']), row['metric'])] = row
    return found


def differentials(calendar, hourly, report):
    ci = registered_ci(report)
    rows = []

    def add(model, metric, population, kind, value, frame, cell=None):
        stats = summary(frame[f'nominal/{model}:{metric}'], frame[f'gfs6/{model}:{metric}'])
        row = {'section': 'differential', 'model': model, 'metric': metric, 'population': population,
               'slice': kind, 'value': value, **stats, 'uncertainty': 'point only'}
        if model == 'v5' and cell is not None and cell in ci:
            reg = ci[cell]
            if reg['paired_dates'] != stats['n_pairs'] or not np.isclose(reg['difference'], stats['difference'], rtol=0, atol=1e-9):
                raise ValueError(f'registered endpoint {cell} differs from the stored daily table')
            row.update(uncertainty='CP-26 registered primary14 bootstrap-t (same pairs)',
                       ratio_ci95_low=reg['ratio_ci95'][0], ratio_ci95_high=reg['ratio_ci95'][1],
                       difference_ci95_low=reg['difference_ci95'][0], difference_ci95_high=reg['difference_ci95'][1])
        rows.append(row)

    for model in MODELS:
        for metric in METRICS:
            for mask in MASKS:
                frame = calendar[calendar[mask]]
                add(model, metric, mask, 'overall', 'overall', frame, (mask, 'overall', 'overall', metric))
                for value in ('affected', 'unaffected'):
                    add(model, metric, mask, 'gfs6_affected', value, frame[frame.gfs6_affected == value])
            frame = calendar[calendar.all_eligible]
            for year in sorted(frame.year.unique()):
                add(model, metric, 'all_eligible', 'year', str(year), frame[frame.year == year],
                    ('all_eligible', 'year', str(year), metric))
            for season in ('DJF', 'MAM', 'JJA', 'SON'):
                add(model, metric, 'all_eligible', 'season', season, frame[frame.season == season],
                    ('all_eligible', 'season', season, metric))
            for value in (0, 1):
                add(model, metric, 'all_eligible', 'negative_price_risk', str(value), frame[frame.negative_price_risk == value])
            for value in ('high', 'not_high', 'unknown'):
                add(model, metric, 'all_eligible', 'volatility_state', value, frame[frame.volatility_state == value])
        nominal, shifted = block_losses(hourly, f'nominal/{model}'), block_losses(hourly, f'gfs6/{model}')
        eligible = set(calendar.index[calendar.all_eligible])
        for metric in METRICS:
            for block in BLOCKS:
                left = nominal[metric].xs(block, level='block')
                right = shifted[metric].xs(block, level='block')
                left = left[left.index.isin(eligible)]
                right = right[right.index.isin(eligible)]
                stats = summary(left, right)
                rows.append({'section': 'differential', 'model': model, 'metric': metric, 'population': 'all_eligible',
                             'slice': 'hour_block', 'value': block, **stats, 'uncertainty': 'point only'})
    return rows


def v2_reference(calendar):
    rows = []
    for model in MODELS:
        for metric in METRICS:
            populations = [(mask, 'overall', 'overall', calendar[calendar[mask]]) for mask in MASKS]
            frame = calendar[calendar.all_eligible]
            populations += [('all_eligible', 'gfs6_affected', v, frame[frame.gfs6_affected == v]) for v in ('affected', 'unaffected')]
            for population, kind, value, part in populations:
                cols = [f'nominal/v2:{metric}', f'nominal/{model}:{metric}', f'gfs6/{model}:{metric}']
                complete = part[cols].dropna()
                B, N, S = (float(complete[c].mean()) for c in cols)
                rows.append({'section': 'v2_reference', 'model': model, 'metric': metric, 'population': population,
                             'slice': kind, 'value': value, 'n_pairs': len(complete), 'missing_pairs': len(part) - len(complete),
                             'nominal_mean': N, 'gfs6_mean': S, 'v2_mean': B,
                             'v2_minus_nominal': B - N, 'v2_over_nominal': B / N, 'v2_minus_gfs6': B - S, 'v2_over_gfs6': B / S,
                             'uncertainty': 'point only'})
    return rows


def break_even(N: float, S: float, B: float) -> tuple[float | None, str]:
    """p* = (B-N)/(S-N) only when S > N and it lies in [0, 1]."""
    if S > N:
        p = (B - N) / (S - N)
        return (p, 'crossing') if 0 <= p <= 1 else (None, 'no crossing in [0,1]')
    if S == N:
        return None, 'S equals N: L(p) is flat at N; no break-even'
    return None, 'S below N: the GFS+6 route is not worse; no break-even'


def mixture(v2_rows):
    rows = []
    for row in v2_rows:
        N, S, B = row['nominal_mean'], row['gfs6_mean'], row['v2_mean']
        p_star, status = break_even(N, S, B)
        out = {'section': 'mixture', 'model': row['model'], 'metric': row['metric'], 'population': row['population'],
               'slice': row['slice'], 'value': row['value'], 'n_pairs': row['n_pairs'], 'nominal_mean': N, 'gfs6_mean': S,
               'v2_mean': B, 'p_star': p_star, 'p_star_status': status, 'uncertainty': 'point only; p is a scenario weight'}
        for p in MIXTURE_P:
            out[f'L_p{p:g}'] = (1 - p) * N + p * S
        if not np.isclose(out['L_p0'], N, rtol=0, atol=0) or not np.isclose(out['L_p1'], S, rtol=0, atol=1e-12):
            raise ValueError('mixture endpoints must reproduce N and S')
        if p_star is not None and not np.isclose((1 - p_star) * N + p_star * S, B, rtol=0, atol=1e-9):
            raise ValueError('break-even does not reproduce B')
        rows.append(out)
    return rows


def fallback_status(calendar):
    out = {}
    for stream in ('nominal/v2', 'nominal/v4', 'nominal/v5', 'gfs6/v4', 'gfs6/v5'):
        frame = calendar[calendar.all_eligible]
        out[stream] = {str(k): int(v) for k, v in frame[f'{stream}:source'].value_counts().items()}
    return out


def fmt(value, digits=2):
    return '—' if value is None or (isinstance(value, float) and not np.isfinite(value)) else f'{value:.{digits}f}'


def render(rows, checks, fallback, sources) -> str:
    table = pd.DataFrame(rows)
    d = table[table.section == 'differential']
    v2 = table[table.section == 'v2_reference']
    mx = table[table.section == 'mixture']

    def cell(model, metric, population, kind, value):
        r = d[(d.model == model) & (d.metric == metric) & (d.population == population) & (d.slice == kind) & (d.value == value)].iloc[0]
        ci = ''
        if isinstance(r.get('uncertainty'), str) and r['uncertainty'].startswith('CP-26'):
            ci = f' (95% {(r.ratio_ci95_low - 1) * 100:+.1f}% to {(r.ratio_ci95_high - 1) * 100:+.1f}%)'
        return f'{r.nominal_mean:.2f} → {r.gfs6_mean:.2f} ({r.difference:+.2f}; {r.change_percent:+.1f}%{ci}; n={int(r.n_pairs)})'

    lines = ['# CP-28 r17 GFS note — descriptive, before the seven-arm experiment', '',
             '**Identity:** `cp28-gfs-postresults-v21-r17`, under the supplement '
             f'`{SUPPLEMENT}` (`{SUPPLEMENT_SHA256}`) and the Owner\'s ruling `{RULING}` (`{RULING_SHA256}`).',
             '**Evidence class:** `development_post_selection`; post-results exploration written after CP-26\'s outcomes '
             'were inspected. It is outside CP-28\'s seven-arm experiment, its multiplicity family and its selection. '
             'No fit, search, replay, H-buffer replay, recalibration, new bootstrap or test, download or dependency change '
             'was made. It adds no gate and changes no arm, input or control.', '',
             '## Headline', '']
    a5 = d[(d.model == 'v5') & (d.population == 'all_eligible') & (d.slice == 'overall')]
    lines.append(f'- On all {int(a5.iloc[0].n_pairs)} paired eligible dates, GFS+6 worsens nominal v5 MAE '
                 f'{cell("v5", "MAE", "all_eligible", "overall", "overall")} and WIS {cell("v5", "WIS", "all_eligible", "overall", "overall")} (EUR/MWh).')
    lines.append(f'- v4, point only: MAE {cell("v4", "MAE", "all_eligible", "overall", "overall")}; WIS {cell("v4", "WIS", "all_eligible", "overall", "overall")}.')
    for model in MODELS:
        aff = {v: cell(model, 'MAE', 'all_eligible', 'gfs6_affected', v) for v in ('affected', 'unaffected')}
        lines.append(f'- {model} MAE on CP-25\'s GFS+6-affected dates: {aff["affected"]}; on unaffected dates: {aff["unaffected"]}. '
                     'Unaffected dates still move because the delayed route has its own training, search and residual history.')
    for model in MODELS:
        for metric in METRICS:
            r = v2[(v2.model == model) & (v2.metric == metric) & (v2.population == 'all_eligible') & (v2.slice == 'overall')].iloc[0]
            m = mx[(mx.model == model) & (mx.metric == metric) & (mx.population == 'all_eligible') & (mx.slice == 'overall')].iloc[0]
            ps = f'p* = {m.p_star:.3f}' if m.p_star is not None and np.isfinite(m.p_star) else m.p_star_status
            lines.append(f'- Weather-independent v2 against {model}, {metric}: v2 {r.v2_mean:.2f} vs nominal {r.nominal_mean:.2f} '
                         f'({(r.v2_over_nominal - 1) * 100:+.1f}%) and GFS+6 {r.gfs6_mean:.2f} ({(r.v2_over_gfs6 - 1) * 100:+.1f}%); '
                         f'planning break-even {ps}.')
    lines += ['', '## Sources, bound by hash', '']
    for name, value in sources.items():
        lines.append(f'- `{name}`: `{value}`')
    lines += ['', '## Populations, definitions and calculations', '',
              '- **Population.** CP-26\'s frozen scored calendar, 2020-12-29..2026-09-06 (2,078 dates), on CP-25\'s frozen '
              'eligible keys; a date enters a comparison only where both streams have a scored loss. No date is removed for its loss. '
              'Fallback losses stay in as scored (see the fallback status below). Actual DST hours; equal date weights.',
              '- **Daily differential.** Per date, `loss(GFS+6) − loss(nominal)` for MAE and WIS separately, in EUR/MWh; '
              'reported as the mean difference and the ratio of paired mean daily losses. No daily percentage is averaged.',
              '- **Breakdowns.** The four CP-26 masks; year (2020 and 2026 partial); pooled season; the three CP-21 hour blocks '
              '(night 22–05, solar 10–16, shoulder/peak 06–09 and 17–21, from the stored hourly losses: block mean within date, then '
              'dates equally); the two issue-time-known states (negative-price risk; high volatility, with its unknown startup state); '
              'and CP-25\'s frozen GFS+6 affected/unaffected classification joined by delivery date (`' + MARGIN_DAYS + '`). '
              'That classification is the documented-schedule stress test, not measured live failures.',
              '- **Uncertainty.** Only CP-26\'s registered primary-14 bootstrap-t intervals for GFS+6 v5 against nominal v5 are reused, '
              'and only where their mask/year/season population is the same set of paired dates (checked: equal paired count and '
              'difference within 1e-9). Every other figure is a point summary.',
              '- **Planning mixture.** For each compatible population and metric, with N the nominal mean, S the GFS+6 mean and B '
              'v2\'s mean on the same dates: `L(p) = (1−p)·N + p·S` at p = 0, .01, .05, .10, .20, .50, 1, B beside it. '
              '**p is a scenario weight, not a measured late-day frequency,** and this is not a chronological replay. '
              'Where S > N, `p* = (B−N)/(S−N)` is shown only if it lies in [0, 1].', '',
              '## Checks against the small source tables', '',
              f'- Hourly-to-daily aggregation: {checks["rows_checked"]} stream-dates; the stored daily MAE/WIS equal the equal-hour '
              f'means of the stored hourly losses (max relative difference {checks["max_relative_difference"]}), hour counts equal '
              'the scoring hours.',
              '- Every reused interval matched its paired count and point difference. The mixture reproduces N at p=0 and S at p=1, '
              'and each shown p* reproduces B.', '',
              '## Fallback status on all eligible dates', '']
    for stream, counts in fallback.items():
        lines.append(f'- `{stream}`: {counts}')
    lines += ['', '## Daily differential, GFS+6 minus nominal', '',
              '| model | metric | population | slice | value | n | nominal | GFS+6 | diff | change | registered 95% ratio CI |',
              '|---|---|---|---|---|---|---|---|---|---|---|']
    for _, r in d.iterrows():
        ci = (f'{(r.ratio_ci95_low - 1) * 100:+.1f}% to {(r.ratio_ci95_high - 1) * 100:+.1f}%'
              if isinstance(r.get('uncertainty'), str) and r['uncertainty'].startswith('CP-26') else 'point only')
        if not r.n_pairs:
            lines.append(f'| {r.model} | {r.metric} | {r.population} | {r.slice} | {r.value} | 0 | — | — | — | — | no paired dates |')
            continue
        lines.append(f'| {r.model} | {r.metric} | {r.population} | {r.slice} | {r.value} | {int(r.n_pairs)} | {fmt(r.nominal_mean)} | '
                     f'{fmt(r.gfs6_mean)} | {r.difference:+.2f} | {r.change_percent:+.1f}% | {ci} |')
    lines += ['', '## Weather-independent v2 on the same dates', '',
              '| model | metric | population | slice | value | n | v2 | nominal | GFS+6 | v2 vs nominal | v2 vs GFS+6 |',
              '|---|---|---|---|---|---|---|---|---|---|---|']
    for _, r in v2.iterrows():
        lines.append(f'| {r.model} | {r.metric} | {r.population} | {r.slice} | {r.value} | {int(r.n_pairs)} | {r.v2_mean:.2f} | '
                     f'{r.nominal_mean:.2f} | {r.gfs6_mean:.2f} | {(r.v2_over_nominal - 1) * 100:+.1f}% | {(r.v2_over_gfs6 - 1) * 100:+.1f}% |')
    lines += ['', '## Planning scenarios: L(p) = (1−p)·N + p·S, with v2 (B) beside it', '',
              '| model | metric | population | slice | value | ' + ' | '.join(f'p={p:g}' for p in MIXTURE_P) + ' | B (v2) | break-even |',
              '|---|---|---|---|---|' + '---|' * len(MIXTURE_P) + '---|---|']
    for _, r in mx.iterrows():
        ps = f'{r.p_star:.3f}' if r.p_star is not None and np.isfinite(r.p_star) else r.p_star_status
        lines.append(f'| {r.model} | {r.metric} | {r.population} | {r.slice} | {r.value} | '
                     + ' | '.join(f'{r[f"L_p{p:g}"]:.2f}' for p in MIXTURE_P) + f' | {r.v2_mean:.2f} | {ps} |')
    lines += ['', '## What this cannot say', '',
              '- **Mechanism.** The GFS+6 route differs on unaffected dates too: it has its own 14 searches, training history and '
              'residual buffer. These stored outputs cannot separate direct input unavailability from propagated history effects; '
              'that would need another experiment.',
              '- **Arrival probabilities.** p is a planning weight. Real late-run frequencies are not estimated from CP-25\'s shifted '
              'schedule and await CP-18\'s planned live availability measurement. Lateness is unlikely to be random, so realized costs '
              'depend on which dates are hit; the affected-date rows are shown separately for that reason, and the all-calendar '
              'deterioration is not a per-late-day penalty.',
              '- **Uncertainty beyond v5\'s registered cells.** v4, v2, hour-block, state, affected/unaffected and every mixture or p* '
              'figure are point summaries; no interval exists for them in CP-26\'s stored inference, and none was computed.',
              '- **Components.** CP-26 stores no scored HG/L/D2 component losses for the GFS+6 route; no component quantile stream '
              'was manufactured from central values, and no component central MAE stands in for an emitted p50 MAE.',
              '- **Oracle envelope.** Not shown: an ex-post best-model envelope is not implementable and adds nothing to the decision below.',
              '- **Money.** These are accuracy costs, not trading profit or business value, and not proof that v2 is the best fallback.', '',
              '## Decision-policy recommendation, without fitting one', '',
              _recommendation(v2, mx), '',
              'Any such rule changes its own error history and interval calibration, so it needs a separate prospective registration, '
              'causal evaluation and Owner authority. It is not an eighth CP-28 arm, it cannot replace nominal v5 through this note, '
              'and it does not authorize waiting past the issue time or a previous-run weather fallback.', '',
              '## Boundary', '',
              'CP-28\'s members, control (nominal v5, with its load forecast), seven arms, sixteen inputs, learning rules, H layers, '
              'one-look rule, twenty endpoints, four Holm families and selection are unchanged by this note. The GFS+6 status stays '
              '**fragile**, accepted by the Owner as a disclosed research-continuation risk; no robustness, publication or live claim follows.', '']
    return '\n'.join(lines)


def _recommendation(v2, mx) -> str:
    def row(model, metric, value):
        return v2[(v2.model == model) & (v2.metric == metric) & (v2.population == 'all_eligible')
                  & (v2.slice == 'gfs6_affected') & (v2.value == value)].iloc[0]
    parts = []
    for model in MODELS:
        for metric in METRICS:
            a = row(model, metric, 'affected')
            parts.append(f'{model} {metric}: v2 {a.v2_mean:.2f}, GFS+6 {a.gfs6_mean:.2f} ({(a.v2_over_gfs6 - 1) * 100:+.1f}%), '
                         f'nominal {a.nominal_mean:.2f} ({(a.v2_over_nominal - 1) * 100:+.1f}%)')
    regular = row('v5', 'MAE', 'unaffected')
    control_gain = all(row('v5', metric, 'affected').v2_mean < row('v5', metric, 'affected').gfs6_mean for metric in METRICS)
    text = ('On CP-25\'s 1,358 GFS+6-affected dates the weather-independent v2 sits close to the delayed-weather route and far '
            'behind the nominal route: ' + '; '.join(parts) + '. On unaffected dates v2 costs '
            f'{(regular.v2_over_nominal - 1) * 100:+.1f}% MAE against nominal v5. ')
    if control_gain:
        text += ('For the control, the weather-independent forecast is descriptively more accurate than the delayed-weather '
                 'route on the stress dates on both metrics, so **a later simple causal rule merits a registered experiment**: '
                 'use the weather-based control when its required GFS inputs were published strictly before the issue time, and a '
                 'separately validated weather-independent alternative when they were not, using only what is observable at '
                 'issuance (actual availability and age, released historical errors), never same-day realized losses or '
                 'hindsight knowledge of the best model.')
    else:
        text += ('For the control (v5), switching to v2 when weather is late would not, descriptively, beat the registered '
                 'delayed-weather handling on the stress dates on both metrics, and against v4 the gain is about one percent. '
                 '**This note therefore does not recommend an availability-switching experiment on present evidence.** The '
                 'material cost of late weather is the gap to the on-time nominal route, so the decision-relevant unknown is how '
                 'often the D−1 00 UTC run actually arrives before the issue time: CP-18\'s planned live availability '
                 'measurement. If a rule is studied later, it should use only what is observable at issuance (actual '
                 'availability and age, released historical errors). The decision is the Owner\'s.')
    return text


def build(root: Path, local: Path) -> tuple[list[dict], str, dict]:
    daily, states, hourly, gfs, report = load(root, local)
    checks = check_hourly_aggregation(daily, hourly)
    calendar = wide(daily, states, gfs)
    rows = differentials(calendar, hourly, report)
    v2_rows = v2_reference(calendar)
    rows += v2_rows + mixture(v2_rows)
    entries = reader.scores_manifest(root)['tables']
    sources = {reader.SCORES_MANIFEST: reader.SCORES_MANIFEST_SHA256,
               entries['daily']['path']: entries['daily']['sha256'],
               entries['known_price_states']['path']: entries['known_price_states']['sha256'],
               entries['hourly']['path'] + ' (local only, `.local/' + reader.LOCAL_ANALYSIS + '/…`)': entries['hourly']['sha256'],
               reader.RUN + '/report.json (local only; inference_endpoints, gfs6)': reader.REPORT_JSON_SHA256,
               MARGIN_DAYS + ' (bound by CP-25 manifest ' + reader.CP25_MANIFEST_SHA256[:12] + '…)': reader.sha256(root / MARGIN_DAYS),
               SUPPLEMENT: SUPPLEMENT_SHA256, RULING: RULING_SHA256}
    note = render(rows, checks, fallback_status(calendar), sources)
    return rows, note, {'checks': checks, 'sources': sources}


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    root, out = Path(argv[0]).resolve(), Path(argv[1])
    local = reader.local_root(root)
    if local is None:
        raise SystemExit('the local CP-26 analysis outputs are absent; the GFS note cannot be built')
    rows, note, meta = build(root, local)
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / 'table.csv', index=False, float_format='%.10g')
    (out / 'note.md').write_text(note, encoding='utf-8')
    (out / 'provenance.json').write_text(json.dumps({'identity': 'cp28-gfs-postresults-v21-r17', **meta},
                                                    indent=2, sort_keys=True, default=str) + '\n', encoding='utf-8')
    print(f'wrote {len(rows)} rows to {out}')


if __name__ == '__main__':
    main()
