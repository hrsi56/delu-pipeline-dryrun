// Inline SVG charts for the final-product Space. Each chart is drawn at its container's CSS width, so the text
// sizes below are rendered sizes (never under 12 px); below 620 px a chart is laid out again for a phone, not
// shrunk. Meaning never rests on colour alone: every series has a direct label, a marker shape or a line style,
// and every reference line is named as a reference, not a target. The table alternative with the same values is
// built beside each chart by main.js.
(function () {
  'use strict';
  const NS = (() => { const d = document.createElement('div'); d.innerHTML = '<svg></svg>'; return d.firstChild.namespaceURI; })();
  const C = {
    text: '#18181B', secondary: '#52525B', grid: '#E4E4E7', accent: '#1D4ED8', issued: '#18181B',
    band: ['rgba(29,78,216,0.10)', 'rgba(29,78,216,0.18)', 'rgba(29,78,216,0.30)'], bandEdge: '#1D4ED8',
    outcome: '#18181B', cf: '#A21CAF', ref: '#52525B', pending: '#92400E',
    series: ['#18181B', '#1D4ED8', '#0F766E', '#92400E', '#A21CAF', '#52525B'],
  };
  const FS = 13;
  const MOBILE = 620;

  function el(tag, attrs, parent) {
    const e = document.createElementNS(NS, tag);
    for (const [k, v] of Object.entries(attrs || {})) if (v !== undefined && v !== null) e.setAttribute(k, String(v));
    if (parent) parent.appendChild(e);
    return e;
  }
  function text(parent, x, y, str, attrs) {
    const t = el('text', Object.assign({x, y, 'font-size': FS, fill: C.text}, attrs || {}), parent);
    t.textContent = str;
    return t;
  }
  function svg(width, height, label) {
    const s = el('svg', {width, height, viewBox: `0 0 ${width} ${height}`, role: 'img', 'aria-label': label,
                         'font-family': 'system-ui, -apple-system, Segoe UI, Roboto, sans-serif'});
    const title = el('title', {}, s);
    title.textContent = label;
    return s;
  }
  function widthOf(container) { return Math.max(200, Math.floor(container.clientWidth || container.getBoundingClientRect().width || 600)); }
  function isMobile(w) { return w < MOBILE; }
  function scale(d0, d1, r0, r1) { const k = (r1 - r0) / ((d1 - d0) || 1); return v => r0 + (v - d0) * k; }
  function nice(span, count) {
    const raw = span / Math.max(1, count);
    const p = Math.pow(10, Math.floor(Math.log10(raw || 1)));
    const m = raw / p;
    return (m <= 1 ? 1 : m <= 2 ? 2 : m <= 5 ? 5 : 10) * p;
  }
  function ticks(min, max, count) {
    const step = nice(max - min, count);
    const out = [];
    for (let v = Math.ceil(min / step) * step; v <= max + step * 1e-9; v += step) out.push(+v.toFixed(10));
    return out;
  }
  function fmt(v, d) {
    if (v === null || v === undefined || !isFinite(v)) return '—';
    const s = Math.abs(v).toFixed(d === undefined ? 0 : d);
    return (v < 0 && Number(s) !== 0 ? '−' : '') + s;
  }
  function yAxis(g, y, tks, x0, x1, unit, d) {
    for (const t of tks) {
      el('line', {x1: x0, x2: x1, y1: y(t), y2: y(t), stroke: t === 0 ? C.secondary : C.grid, 'stroke-width': 1}, g);
      text(g, x0 - 6, y(t) + 4, fmt(t, d), {'text-anchor': 'end', fill: C.secondary});
    }
    if (unit) text(g, x0 - 6, y(tks[tks.length - 1]) - 12, unit, {'text-anchor': 'end', fill: C.secondary});
  }
  function figure(container, s, caption) {
    const f = document.createElement('figure');
    f.className = 'chart';
    f.appendChild(s);
    if (caption) { const c = document.createElement('figcaption'); c.textContent = caption; f.appendChild(c); }
    container.appendChild(f);
    return f;
  }
  function legend(container, items) {
    const div = document.createElement('div');
    div.className = 'legend';
    for (const it of items) {
      const span = document.createElement('span');
      const s = svg(28, 16, it.label);
      s.removeAttribute('role'); s.setAttribute('aria-hidden', 'true');
      if (it.kind === 'area') el('rect', {x: 2, y: 3, width: 24, height: 10, fill: it.fill, stroke: it.stroke || 'none'}, s);
      else el('line', {x1: 2, x2: 26, y1: 8, y2: 8, stroke: it.stroke, 'stroke-width': it.width || 2, 'stroke-dasharray': it.dash || null}, s);
      if (it.marker) marker(s, 14, 8, it.marker, it.markerColor || it.stroke);
      span.appendChild(s);
      span.appendChild(document.createTextNode(it.label));
      div.appendChild(span);
    }
    container.appendChild(div);
    return div;
  }
  function marker(g, x, y, shape, color, attrs) {
    const a = Object.assign({'data-mark': '1'}, attrs || {});
    if (shape === 'circle') return el('circle', Object.assign({cx: x, cy: y, r: 3.5, fill: color}, a), g);
    if (shape === 'ring') return el('circle', Object.assign({cx: x, cy: y, r: 4, fill: '#FFFFFF', stroke: color, 'stroke-width': 1.75}, a), g);
    if (shape === 'square') return el('rect', Object.assign({x: x - 3.5, y: y - 3.5, width: 7, height: 7, fill: color}, a), g);
    if (shape === 'diamond') return el('path', Object.assign({d: `M${x} ${y - 5}L${x + 5} ${y}L${x} ${y + 5}L${x - 5} ${y}Z`, fill: '#FFFFFF', stroke: color, 'stroke-width': 1.75}, a), g);
    if (shape === 'triangle') return el('path', Object.assign({d: `M${x} ${y - 5}L${x + 5} ${y + 4}L${x - 5} ${y + 4}Z`, fill: color}, a), g);
    return null;
  }
  function polyline(g, pts, attrs) {
    const d = [];
    let pen = false;
    for (const p of pts) {
      if (p === null) { pen = false; continue; }
      d.push((pen ? 'L' : 'M') + p[0].toFixed(1) + ' ' + p[1].toFixed(1));
      pen = true;
    }
    return el('path', Object.assign({d: d.join(''), fill: 'none'}, attrs), g);
  }

  // ---------------------------------------------------------------- the hourly forecast with its nominal bands
  // opts: {labels[], q[][7], y[] (null = pending), title, cf[] (optional counterfactual p50), cfLabel, issuedLabel}
  function hourly(container, opts) {
    const w = widthOf(container), mob = isMobile(w);
    const n = opts.q.length;
    const h = mob ? 300 : 340;
    const m = {l: mob ? 46 : 60, r: mob ? 12 : 150, t: 28, b: mob ? 52 : 44};
    const vals = [];
    opts.q.forEach(r => vals.push(r[0], r[6]));
    (opts.y || []).forEach(v => { if (v !== null && v !== undefined && isFinite(v)) vals.push(v); });
    (opts.cf || []).forEach(v => { if (isFinite(v)) vals.push(v); });
    let lo = Math.min(...vals), hi = Math.max(...vals);
    const pad = (hi - lo) * 0.06 || 5;
    lo -= pad; hi += pad;
    if (lo > 0 && lo < (hi - lo) * 0.3) lo = 0;
    const s = svg(w, h, opts.title);
    const x = scale(-0.5, n - 0.5, m.l, w - m.r);
    const y = scale(lo, hi, h - m.b, m.t);
    const g = el('g', {}, s);
    const tks = ticks(lo, hi, mob ? 4 : 6);
    yAxis(g, y, tks.filter(t => t >= lo && t <= hi), m.l, w - m.r, 'EUR/MWh', 0);
    // pending hours: shaded columns and a named marker
    (opts.y || []).forEach((v, i) => {
      if (v === null || v === undefined) {
        el('rect', {x: x(i - 0.5), y: m.t, width: x(i + 0.5) - x(i - 0.5), height: h - m.b - m.t, fill: '#FEF3C7', 'data-pending': '1'}, g);
        marker(g, x(i), h - m.b - 6, 'triangle', C.pending, {'data-role': 'pending'});
      }
    });
    const bands = [[0, 6], [1, 5], [2, 4]];
    bands.forEach(([a, b], k) => {
      const top = opts.q.map((r, i) => [x(i), y(r[b])]);
      const bottom = opts.q.map((r, i) => [x(i), y(r[a])]).reverse();
      el('path', {d: 'M' + top.concat(bottom).map(p => p[0].toFixed(1) + ' ' + p[1].toFixed(1)).join('L') + 'Z', fill: C.band[k], 'data-band': ['95', '80', '50'][k]}, g);
    });
    polyline(g, opts.q.map((r, i) => [x(i), y(r[0])]), {stroke: C.bandEdge, 'stroke-width': 1, 'stroke-dasharray': '2 3', 'data-mark': '1'});
    polyline(g, opts.q.map((r, i) => [x(i), y(r[6])]), {stroke: C.bandEdge, 'stroke-width': 1, 'stroke-dasharray': '2 3', 'data-mark': '1'});
    polyline(g, opts.q.map((r, i) => [x(i), y(r[3])]), {stroke: C.issued, 'stroke-width': 2.25, 'data-mark': '1', 'data-series': 'issued-p50', class: 'issued'});
    if (opts.cf) {
      polyline(g, opts.cf.map((v, i) => [x(i), y(v)]), {stroke: C.cf, 'stroke-width': 2.25, 'stroke-dasharray': '6 4', 'data-mark': '1', 'data-series': 'counterfactual', class: 'counterfactual'});
      opts.cf.forEach((v, i) => marker(g, x(i), y(v), 'diamond', C.cf, {'data-series': 'counterfactual'}));
    }
    if (opts.y) {
      opts.y.forEach((v, i) => { if (v !== null && v !== undefined) marker(g, x(i), y(v), 'circle', C.outcome, {'data-series': 'outcome'}); });
    }
    // x labels: every hour on desktop, every 3rd or 6th on a phone
    const every = mob ? (n > 12 ? 6 : 3) : (w < 900 ? 2 : 1);
    opts.labels.forEach((lab, i) => {
      if (i % every === 0) text(g, x(i), h - m.b + 18, lab, {'text-anchor': 'middle', fill: C.secondary});
    });
    text(g, (m.l + w - m.r) / 2, h - 6, opts.xTitle || 'delivery hour, Europe/Berlin', {'text-anchor': 'middle', fill: C.secondary});
    if (!mob) {
      const last = n - 1, lx = w - m.r + 8;
      const labels = [[opts.q[last][3], opts.issuedLabel || 'issued p50'], [opts.q[last][6], '95% band (nominal)'],
                      [opts.q[last][5], '80% band (nominal)'], [opts.q[last][4], '50% band (nominal)']];
      if (opts.cf) labels.push([opts.cf[last], 'counterfactual']);
      labels.sort((a, b) => y(a[0]) - y(b[0]));
      let prev = -1e9;
      for (const [v, lab] of labels) {
        const yy = Math.max(y(v) + 4, prev + 15);
        text(g, lx, yy, lab, {fill: lab === 'counterfactual' ? C.cf : C.text});
        prev = yy;
      }
    }
    const fig = figure(container, s, opts.caption);
    const items = [{label: opts.issuedLabel || 'issued p50', stroke: C.issued, width: 2.25},
                   {label: '50% / 80% / 95% nominal bands (darker is narrower)', kind: 'area', fill: C.band[1]},
                   {label: '2.5% and 97.5% quantiles', stroke: C.bandEdge, width: 1, dash: '2 3'}];
    if (opts.y) items.push({label: 'published price', stroke: 'transparent', marker: 'circle', markerColor: C.outcome});
    if (opts.y && opts.y.some(v => v === null || v === undefined)) items.push({label: 'pending hour (not published, unscored)', stroke: 'transparent', marker: 'triangle', markerColor: C.pending});
    if (opts.cf) items.push({label: opts.cfLabel, stroke: C.cf, dash: '6 4', marker: 'diamond'});
    legend(fig, items);
    return fig;
  }

  // ---------------------------------------------------------------- reliability diagram: points with counts
  function reliability(container, opts) {
    const w = Math.min(widthOf(container), 560), mob = isMobile(w);
    const h = mob ? w : Math.min(w, 460);
    const m = {l: 52, r: 16, t: 20, b: 56};
    const s = svg(w, h, opts.title);
    const x = scale(0, 1, m.l, w - m.r), y = scale(0, 1, h - m.b, m.t);
    const g = el('g', {}, s);
    for (const t of [0, 0.25, 0.5, 0.75, 1]) {
      el('line', {x1: x(0), x2: x(1), y1: y(t), y2: y(t), stroke: C.grid}, g);
      text(g, m.l - 6, y(t) + 4, Math.round(t * 100) + '%', {'text-anchor': 'end', fill: C.secondary});
      text(g, x(t), h - m.b + 18, Math.round(t * 100) + '%', {'text-anchor': 'middle', fill: C.secondary});
    }
    el('line', {x1: x(0), y1: y(0), x2: x(1), y2: y(1), stroke: C.ref, 'stroke-width': 1.5, 'stroke-dasharray': '5 4', 'data-mark': '1', 'data-ref': 'diagonal'}, g);
    text(g, x(0.04), y(0.04) - 30, 'diagonal: nominal level', {fill: C.secondary, transform: `rotate(-45 ${x(0.04)} ${y(0.04) - 30})`});
    text(g, x(0.10), y(0.10) - 30, '(a reference, not a target)', {fill: C.secondary, transform: `rotate(-45 ${x(0.10)} ${y(0.10) - 30})`});
    for (const p of opts.points) {
      marker(g, x(p.level), y(p.share), 'circle', C.text);
      if (!mob) text(g, x(p.level) + 7, y(p.share) - 6, `${p.name} · ${p.at_or_below}/${p.scored_hours}`, {fill: C.text});
    }
    text(g, (m.l + w - m.r) / 2, h - 18, 'quantile level q (nominal)', {'text-anchor': 'middle', fill: C.secondary});
    text(g, 6, m.t - 6, 'observed share at or below the q forecast', {fill: C.secondary});
    return figure(container, s, opts.caption);
  }

  // ---------------------------------------------------------------- rolling coverage, with width beside it
  const BAND_STYLE = {'50': {dash: null, marker: 'circle'}, '80': {dash: '6 3', marker: 'square'}, '95': {dash: '2 3', marker: 'triangle'}};
  function rolling(container, opts) {
    const w = widthOf(container), mob = isMobile(w);
    const n = opts.series.length;
    const m = {l: 52, r: mob ? 16 : 120, t: 24, b: 40};
    const out = [];
    for (const kind of ['coverage', 'width']) {
      const h = mob ? 220 : 240;
      const title = kind === 'coverage' ? opts.title : opts.widthTitle;
      const s = svg(w, h, title);
      const g = el('g', {}, s);
      const x = scale(-0.5, n - 0.5, m.l, w - m.r);
      let lo = 0, hi = 1;
      if (kind === 'width') {
        hi = Math.max(...opts.series.map(r => r.width95_eur)) * 1.1 || 1;
      }
      const y = scale(lo, hi, h - m.b, m.t);
      const tks = kind === 'coverage' ? [0, 0.25, 0.5, 0.75, 1] : ticks(0, hi, 4);
      for (const t of tks) {
        el('line', {x1: m.l, x2: w - m.r, y1: y(t), y2: y(t), stroke: C.grid}, g);
        text(g, m.l - 6, y(t) + 4, kind === 'coverage' ? Math.round(t * 100) + '%' : fmt(t, 0), {'text-anchor': 'end', fill: C.secondary});
      }
      text(g, m.l - 6, m.t - 10, kind === 'coverage' ? 'coverage' : 'EUR/MWh', {'text-anchor': 'end', fill: C.secondary});
      for (const b of ['50', '80', '95']) {
        const st = BAND_STYLE[b];
        if (kind === 'coverage') {
          el('line', {x1: m.l, x2: w - m.r, y1: y(Number(b) / 100), y2: y(Number(b) / 100), stroke: C.ref, 'stroke-dasharray': '1 4', 'stroke-width': 1.5, 'data-mark': '1', 'data-ref': 'nominal-' + b}, g);
        }
        const key = kind === 'coverage' ? 'coverage' + b : 'width' + b + '_eur';
        polyline(g, opts.series.map((r, i) => [x(i), y(r[key])]), {stroke: C.text, 'stroke-width': 1.75, 'stroke-dasharray': st.dash, 'data-mark': '1'});
        if (n <= 60) opts.series.forEach((r, i) => marker(g, x(i), y(r[key]), st.marker, C.text));
        const last = opts.series[n - 1];
        if (!mob) text(g, w - m.r + 8, y(last[key]) + 4 + (b === '80' && kind === 'coverage' ? 0 : 0), `${b}% band` + (kind === 'coverage' ? ` (nominal ${b}%: reference)` : ''), {fill: C.text});
      }
      const step = Math.max(1, Math.ceil(n / (mob ? 3 : 6)));
      opts.series.forEach((r, i) => { if (i % step === 0 || i === n - 1) text(g, x(i), h - m.b + 18, r.day, {'text-anchor': i === n - 1 ? 'end' : 'middle', fill: C.secondary}); });
      text(g, (m.l + w - m.r) / 2, h - 4, 'scored delivery day (window ends here)', {'text-anchor': 'middle', fill: C.secondary});
      const fig = figure(container, s, kind === 'coverage' ? opts.caption : opts.widthCaption);
      legend(fig, ['50', '80', '95'].map(b => ({label: `${b}% band`, stroke: C.text, dash: BAND_STYLE[b].dash, marker: BAND_STYLE[b].marker}))
        .concat(kind === 'coverage' ? [{label: 'nominal level (a reference, not a target)', stroke: C.ref, dash: '1 4', width: 1.5}] : []));
      out.push(fig);
    }
    return out;
  }

  // ---------------------------------------------------------------- bars (vertical; horizontal on a phone)
  // items: [{label, value, note, nominal}] ; opts: {title, unit, refLabel, signed, horizontal, digits}
  function bars(container, items, opts) {
    const w = widthOf(container), mob = isMobile(w);
    const horizontal = opts.horizontal || mob;
    const vals = items.map(i => i.value).concat(items.filter(i => i.nominal !== undefined).map(i => i.nominal));
    let lo = Math.min(0, ...vals), hi = Math.max(0, ...vals);
    if (hi === lo) hi = lo + 1;
    const d = opts.digits === undefined ? 1 : opts.digits;
    if (horizontal) {
      // room for the value labels beside the bars' ends
      const span = hi - lo;
      if (lo < 0) lo -= span * 0.16;
      if (hi > 0) hi += span * 0.16;
      const row = 28, labW = Math.min(mob ? 120 : 220, Math.floor(w * 0.38));
      const m = {l: labW + 8, r: 16, t: 26, b: 28};
      const h = m.t + m.b + row * items.length;
      const s = svg(w, h, opts.title);
      const g = el('g', {}, s);
      const x = scale(lo, hi, m.l, w - m.r);
      for (const t of ticks(lo, hi, mob ? 3 : 5).filter(t => t >= lo && t <= hi)) {
        el('line', {x1: x(t), x2: x(t), y1: m.t - 4, y2: h - m.b, stroke: t === 0 ? C.secondary : C.grid}, g);
        text(g, x(t), h - m.b + 16, opts.pct ? Math.round(t * 100) + '%' : fmt(t, opts.tickDigits || 0), {'text-anchor': 'middle', fill: C.secondary});
      }
      text(g, w - m.r, m.t - 10, opts.unit || '', {'text-anchor': 'end', fill: C.secondary});
      items.forEach((it, i) => {
        const yy = m.t + i * row;
        text(g, m.l - 8, yy + 18, it.label, {'text-anchor': 'end'});
        const x0 = x(Math.min(0, it.value)), x1 = x(Math.max(0, it.value));
        el('rect', {x: x0, y: yy + 6, width: Math.max(1, x1 - x0), height: row - 12, fill: it.color || C.text, 'data-mark': '1'}, g);
        if (it.nominal !== undefined) el('line', {x1: x(it.nominal), x2: x(it.nominal), y1: yy + 2, y2: yy + row - 2, stroke: C.accent, 'stroke-width': 2.5, 'data-mark': '1', 'data-ref': 'nominal'}, g);
        const tv = it.text !== undefined ? it.text : fmt(it.value, d);
        text(g, (it.value < 0 ? x0 - 4 : x1 + 4), yy + 18, tv, {'text-anchor': it.value < 0 ? 'end' : 'start'});
      });
      const fig = figure(container, s, opts.caption);
      if (opts.refLabel) legend(fig, [{label: opts.refLabel, stroke: C.accent, width: 2.5}]);
      return fig;
    }
    const h = opts.height || 300;
    const m = {l: 56, r: 16, t: 26, b: opts.xTitle ? 64 : 48};
    const s = svg(w, h, opts.title);
    const g = el('g', {}, s);
    const n = items.length;
    const x = scale(0, n, m.l, w - m.r);
    const y = scale(lo, hi * 1.05, h - m.b, m.t);
    const tks = ticks(lo, hi, 5).filter(t => t >= lo && t <= hi * 1.05);
    for (const t of tks) {
      el('line', {x1: m.l, x2: w - m.r, y1: y(t), y2: y(t), stroke: t === 0 ? C.secondary : C.grid}, g);
      text(g, m.l - 6, y(t) + 4, opts.pct ? Math.round(t * 100) + '%' : fmt(t, opts.tickDigits || 0), {'text-anchor': 'end', fill: C.secondary});
    }
    text(g, m.l - 6, m.t - 10, opts.unit || '', {'text-anchor': 'end', fill: C.secondary});
    const bw = Math.max(1, (x(1) - x(0)) * 0.72);
    const every = Math.max(1, Math.ceil(n / Math.floor((w - m.l - m.r) / 64)));
    items.forEach((it, i) => {
      const cx = (x(i) + x(i + 1)) / 2;
      const y0 = y(Math.max(0, it.value)), y1 = y(Math.min(0, it.value));
      el('rect', {x: cx - bw / 2, y: y0, width: bw, height: Math.max(0.5, y1 - y0), fill: it.color || C.text, 'data-mark': '1'}, g);
      if (it.nominal !== undefined) el('line', {x1: cx - bw / 2 - 2, x2: cx + bw / 2 + 2, y1: y(it.nominal), y2: y(it.nominal), stroke: C.accent, 'stroke-width': 2.5, 'data-mark': '1', 'data-ref': 'nominal'}, g);
      if (opts.valueLabels) text(g, cx, y0 - 4, it.text !== undefined ? it.text : fmt(it.value, d), {'text-anchor': 'middle'});
      if (i % every === 0) text(g, cx, h - m.b + 18, it.label, {'text-anchor': 'middle', fill: C.secondary});
    });
    for (const r of opts.refs || []) {
      const i = r.at;
      const cx = x(i);
      el('line', {x1: cx, x2: cx, y1: m.t, y2: h - m.b, stroke: r.color || C.secondary, 'stroke-width': 1.5, 'stroke-dasharray': '4 3', 'data-mark': '1', 'data-ref': r.name}, g);
      text(g, cx + (r.anchor === 'end' ? -4 : 4), m.t + 12 + (r.dy || 0), r.label, {'text-anchor': r.anchor || 'start', fill: C.text});
    }
    if (opts.xTitle) text(g, (m.l + w - m.r) / 2, h - 8, opts.xTitle, {'text-anchor': 'middle', fill: C.secondary});
    const fig = figure(container, s, opts.caption);
    if (opts.refLabel) legend(fig, [{label: opts.refLabel, stroke: C.accent, width: 2.5}]);
    return fig;
  }

  // ---------------------------------------------------------------- lines over categories (the hourly profile)
  function lines(container, opts) {
    const w = widthOf(container), mob = isMobile(w);
    const n = opts.labels.length;
    const h = mob ? 280 : 320;
    const m = {l: 56, r: mob ? 16 : 110, t: 24, b: 48};
    const vals = [];
    opts.series.forEach(sr => sr.values.forEach(v => { if (isFinite(v)) vals.push(v); }));
    let lo = Math.min(0, ...vals), hi = Math.max(...vals);
    hi += (hi - lo) * 0.05;
    const s = svg(w, h, opts.title);
    const g = el('g', {}, s);
    const x = scale(-0.5, n - 0.5, m.l, w - m.r), y = scale(lo, hi, h - m.b, m.t);
    yAxis(g, y, ticks(lo, hi, 5).filter(t => t >= lo && t <= hi), m.l, w - m.r, opts.unit, 0);
    const shapes = ['circle', 'square', 'triangle', 'diamond'];
    const dashes = [null, '6 3', '2 3', '8 3 2 3'];
    const style = (sr, k) => ({color: sr.color || C.series[k % C.series.length], dash: sr.dash !== undefined ? sr.dash : dashes[k % 4],
                               shape: sr.marker || shapes[k % 4]});
    const placed = [];
    opts.series.forEach((sr, k) => {
      const st = style(sr, k);
      polyline(g, sr.values.map((v, i) => isFinite(v) ? [x(i), y(v)] : null),
               {stroke: st.color, 'stroke-width': 2.25, 'stroke-dasharray': st.dash, 'data-mark': '1', 'data-series': sr.id || null, class: sr.cls || null});
      sr.values.forEach((v, i) => { if (isFinite(v) && (n <= 25 || i % 2 === 0)) marker(g, x(i), y(v), st.shape, st.color, {'data-series': sr.id || null}); });
      if (!mob) {
        let yy = y(sr.values[n - 1]) + 4;
        for (const p of placed) if (Math.abs(p - yy) < 15) yy = p + 15;
        placed.push(yy);
        text(g, w - m.r + 8, yy, sr.short || sr.name, {fill: st.color});
      }
    });
    const every = mob ? 6 : (w < 900 ? 3 : (n <= 25 ? 2 : 3));
    opts.labels.forEach((lab, i) => { if (i % every === 0) text(g, x(i), h - m.b + 18, lab, {'text-anchor': 'middle', fill: C.secondary}); });
    if (opts.xTitle) text(g, (m.l + w - m.r) / 2, h - 8, opts.xTitle, {'text-anchor': 'middle', fill: C.secondary});
    const fig = figure(container, s, opts.caption);
    legend(fig, opts.series.map((sr, k) => { const st = style(sr, k); return {label: sr.name, stroke: st.color, dash: st.dash, marker: st.shape}; }));
    return fig;
  }

  window.Charts = {hourly, reliability, rolling, bars, lines, isMobile, widthOf, MOBILE, colours: C, fmt};
})();
