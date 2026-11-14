/* Shared helpers for the QCNN dashboard. Global: window.QCNNUI. No modules, works from file://. */
(function () {
  'use strict';
  const C = {
    conv: '#2f7fd0', pool: '#f08a24', fc: '#7a3fb0', err: '#d1342c',
    spt: '#e9c46a', triv: '#2a9d8f', afm: '#c0728f',
    ink: '#e8e4da', muted: '#9aa3ad', rule: '#2a2f37', bg: '#0e1116',
  };
  const MONO = '"JetBrains Mono", Consolas, monospace';
  const SVGNS = 'http://www.w3.org/2000/svg';

  // ---- small DOM helpers (also used by section files) ----
  function h(tag, cls, html, parent) {
    const e = document.createElement(tag);
    if (cls) e.className = cls;
    if (html != null) e.innerHTML = html;
    if (parent) parent.appendChild(e);
    return e;
  }
  function svgEl(tag, attrs, parent) {
    const e = document.createElementNS(SVGNS, tag);
    for (const k in (attrs || {})) {
      if (k === 'text') e.textContent = attrs[k];
      else e.setAttribute(k, attrs[k]);
    }
    if (parent) parent.appendChild(e);
    return e;
  }

  function isObj(x) { return x && typeof x === 'object' && !Array.isArray(x); }
  function merge(a, b) {
    const out = Object.assign({}, a);
    for (const k in (b || {})) out[k] = (isObj(out[k]) && isObj(b[k])) ? merge(out[k], b[k]) : b[k];
    return out;
  }

  // ---- Plotly ----
  function layout(over) {
    const ax = {
      gridcolor: 'rgba(232,228,218,0.07)', zerolinecolor: 'rgba(232,228,218,0.18)',
      linecolor: C.rule, tickcolor: C.rule, color: C.muted,
      tickfont: { family: MONO, size: 11, color: C.muted },
      title: { font: { family: MONO, size: 12, color: C.muted }, standoff: 10 },
    };
    const base = {
      paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)',
      font: { family: MONO, size: 11, color: C.ink },
      margin: { l: 56, r: 16, t: 24, b: 48 },
      xaxis: ax, yaxis: ax,
      legend: { font: { family: MONO, size: 11, color: C.ink }, bgcolor: 'rgba(0,0,0,0)' },
      hoverlabel: { bgcolor: '#161a21', bordercolor: C.rule, font: { family: MONO, size: 11, color: C.ink } },
      colorway: [C.conv, C.pool, C.fc, C.spt, C.triv, C.err],
    };
    return merge(base, over);
  }
  function plot(div, traces, over) {
    if (typeof Plotly === 'undefined') throw new Error('Plotly did not load (needs the CDN; check the network connection).');
    return Plotly.newPlot(div, traces, layout(over), { displaylogo: false, displayModeBar: false, responsive: true });
  }
  // Plotly colorscale for QCNN outputs in [0, 1]
  const scale = [[0, '#1b3a4b'], [0.5, '#2a9d8f'], [1, '#e9c46a']];

  // ---- data guards ----
  function needData(el, key, stage) {
    if (window.QCNN && window.QCNN[key] != null) return true;
    h('div', 'notice',
      'Data for this figure has not been generated yet. Run <code>.venv/Scripts/python build_data.py --stage ' + stage + '</code>, then reload.', el);
    return false;
  }
  function quickBadge(el, obj) {
    if (obj && obj.quick) h('div', 'qbadge', 'preview data (quick build)', el);
  }

  // ---- layout pieces ----
  // Renders the heading block into `el`. Returns the content container (class sec-body).
  // `kicker` (optional) is a small mono line under the title, e.g. the figure reference.
  function head(el, num, title, kicker) {
    const hd = h('header', 'sec-head', null, el);
    h('div', 'sec-num', '§' + num, hd);
    h('h2', 'sec-title', title, hd);
    if (kicker) h('p', 'sec-kicker', kicker, hd);
    return h('div', 'sec-body', null, el);
  }
  function note(el, html) { return h('aside', 'note', html, el); }

  function slider(el, o) {
    const f = o.fmt || (v => String(v));
    const wrap = h('label', 'ctl', null, el);
    h('span', null, o.label, wrap);
    const inp = h('input', null, null, wrap);
    inp.type = 'range'; inp.min = o.min; inp.max = o.max; inp.step = o.step; inp.value = o.value;
    inp.setAttribute('aria-label', o.label);
    const out = h('output', null, f(+o.value), wrap);
    inp.addEventListener('input', () => { out.textContent = f(+inp.value); if (o.onInput) o.onInput(+inp.value); });
    inp.set = v => { inp.value = v; out.textContent = f(+inp.value); }; // programmatic update, no callback
    return inp;
  }

  // options: array of strings or {value, label}. opts.multi -> value/onChange use arrays.
  function toggle(el, options, value, onChange, opts) {
    const multi = !!(opts && opts.multi);
    const grp = h('div', 'pills', null, el);
    grp.setAttribute('role', 'group');
    let cur = multi ? (value || []).slice() : value;
    const btns = options.map(o => {
      const v = (o && o.value !== undefined) ? o.value : o;
      const b = h('button', 'pill', (o && o.label !== undefined) ? o.label : String(o), grp);
      b.type = 'button'; b._v = v;
      b.addEventListener('click', () => {
        if (multi) cur = cur.includes(v) ? cur.filter(x => x !== v) : cur.concat([v]);
        else cur = v;
        paint(); if (onChange) onChange(multi ? cur.slice() : cur);
      });
      return b;
    });
    function paint() { btns.forEach(b => b.setAttribute('aria-pressed', String(multi ? cur.includes(b._v) : cur === b._v))); }
    paint();
    grp.set = v => { cur = multi ? v.slice() : v; paint(); }; // programmatic update, no callback
    Object.defineProperty(grp, 'value', { get: () => (multi ? cur.slice() : cur) });
    return grp;
  }

  function katex(el) {
    if (typeof renderMathInElement !== 'function') return;
    renderMathInElement(el, {
      delimiters: [{ left: '$$', right: '$$', display: true }, { left: '$', right: '$', display: false }],
      throwOnError: false,
    });
  }
  function fmt(x, digits) {
    if (x == null || (typeof x === 'number' && !isFinite(x))) return '—';
    const d = digits == null ? 3 : digits;
    const a = Math.abs(x);
    return (a !== 0 && (a < 1e-3 || a >= 1e5)) ? x.toExponential(Math.max(d - 1, 0)) : x.toFixed(d);
  }

  // ---- registry: every renderer is isolated by try/catch ----
  const reg = [];
  let started = false;
  function run(r) {
    const el = document.getElementById('sec-' + r.id);
    if (!el) return;
    try {
      r.fn(el, window.QCNN || {});
      try { katex(el); } catch (e) { console.warn('[' + r.id + '] KaTeX:', e); }
    } catch (e) {
      console.error('[' + r.id + ']', e);
      h('div', 'errcard', '<b>This section failed to render.</b><pre></pre>', el).querySelector('pre').textContent = String(e && e.stack || e).split('\n').slice(0, 4).join('\n');
    }
  }
  function register(id, fn) {
    const r = { id, fn };
    reg.push(r);
    if (started) run(r);
  }

  // ---- rail: scroll spy + reading progress ----
  function initRail() {
    const links = [...document.querySelectorAll('.rail a')];
    const by = {};
    links.forEach(a => { by[a.getAttribute('href').slice(1)] = a; });
    if ('IntersectionObserver' in window) {
      const vis = new Set();
      const io = new IntersectionObserver(es => {
        es.forEach(e => (e.isIntersecting ? vis.add(e.target.id) : vis.delete(e.target.id)));
        const first = links.map(a => a.getAttribute('href').slice(1)).find(id => vis.has(id));
        if (first) links.forEach(a => a.classList.toggle('on', a === by[first]));
      }, { rootMargin: '-20% 0px -60% 0px' });
      document.querySelectorAll('.sec').forEach(s => io.observe(s));
    }
    const prog = () => {
      const m = document.documentElement.scrollHeight - innerHeight;
      document.documentElement.style.setProperty('--p', m > 0 ? Math.min(1, Math.max(0, scrollY / m)) : 0);
    };
    addEventListener('scroll', prog, { passive: true }); addEventListener('resize', prog); prog();
  }

  function start() {
    started = true;
    try { initRail(); } catch (e) { console.error('[rail]', e); }
    reg.forEach(run);
    // Late-loading CDN scripts: typeset once more if auto-render arrived after us.
    window.addEventListener('load', () => document.querySelectorAll('.sec').forEach(s => { try { katex(s); } catch (e) { /* ignore */ } }));
  }
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', start);
  else start();

  window.QCNNUI = { C, register, layout, plot, scale, needData, quickBadge, head, note, slider, toggle, katex, fmt, h, svgEl, merge };
})();
