/* Sections: phase (§5), depth (§6), samples (§7), rgflow (§8). Data: QCNN.phase_ed, phase_mps, cut. */
(function () {
  'use strict';
  const U = window.QCNNUI;
  if (!U) return;
  const { C, h } = U;
  const MONO = '"JetBrains Mono", Consolas, monospace';
  const MINUS = '−';

  // ---------- helpers ----------
  const keysOf = o => Object.keys(o || {}).sort((a, b) => a - b);               // "1","2",... numeric order
  const sgn = v => (v < 0 ? MINUS : '') + Math.abs(v);                           // typographic minus
  const hexRGB = c => [1, 3, 5].map(i => parseInt(c.slice(i, i + 2), 16));
  function ramp(c0, c1, n) {                                                     // light -> dark; one entry -> dark end
    const a = hexRGB(c0), b = hexRGB(c1);
    return Array.from({ length: n }, (_, i) => {
      const t = n > 1 ? i / (n - 1) : 1;
      return '#' + a.map((v, k) => Math.round(v + (b[k] - v) * t).toString(16).padStart(2, '0')).join('');
    });
  }
  const BLUES = n => ramp('#a9cff7', '#1d5fb0', n);                              // QCNN depth family
  const REDS = n => ramp('#f4b3ad', '#b8372d', n);                               // SOP length family
  const closest = (arr, v) => arr.reduce((bi, x, i) => (Math.abs(x - v) < Math.abs(arr[bi] - v) ? i : bi), 0);
  const bdColor = { upper: C.conv, lower: '#e0564c' };                           // boundary diamonds, as in the paper

  function plotDiv(parent, height) {
    const d = h('div', 'ph-plot', null, parent);
    d.style.height = height + 'px';
    return d;
  }
  const sub = (parent, html) => h('p', 'ph-sub', html, parent);
  const axTitle = t => ({ text: t });

  function boundaryTraces(mps) {
    const out = [];
    ['upper', 'lower'].forEach(k => {
      const pts = (mps && mps.boundary && mps.boundary[k]) || [];
      if (!pts.length) return;
      out.push({
        type: 'scatter', mode: 'lines+markers', name: k + ' boundary',
        x: pts.map(p => p[0]), y: pts.map(p => p[1]),
        marker: { symbol: 'diamond', size: 9, color: bdColor[k], line: { color: C.ink, width: 1 } },
        line: { color: bdColor[k], width: 1, dash: 'dot' },
        hovertemplate: k + ' boundary<br>h₁ = %{x:.2f}<br>h₂ = %{y:.3f}<extra></extra>',
      });
    });
    return out;
  }

  const cbar = title => ({
    thickness: 10, outlinewidth: 0, len: 0.92, tickfont: { family: MONO, size: 10, color: C.muted },
    title: { text: title, font: { family: MONO, size: 11, color: C.muted } },
  });

  // =====================================================================
  // §5 phase
  // =====================================================================
  U.register('phase', (el, D) => {
    const b = U.head(el, 5, 'The phase diagram, read by a circuit', 'Fig. 2a');
    const ok = [U.needData(b, 'phase_ed', 'phase_ed'), U.needData(b, 'phase_mps', 'phase_mps')];
    const ed = ok[0] ? D.phase_ed : null, mps = ok[1] ? D.phase_mps : null;
    const sources = [];
    if (ed) sources.push({ value: 'ed1', label: `N=${ed.N} exact diag · d=1`, N: ed.N, d: 1, h1: ed.h1, h2: ed.h2, z: ed.qcnn['1'], e: ed.energy.map(r => r.map(v => v / ed.N)), quick: ed.quick });
    if (mps) keysOf(mps.qcnn).forEach(k => sources.push({ value: 'mps' + k, label: `N=${mps.N} MPS · d=${k}`, N: mps.N, d: +k, h1: mps.h1, h2: mps.h2, z: mps.qcnn[k], e: mps.energy_density, quick: mps.quick }));
    if (!sources.length) return;
    U.quickBadge(b, ed || mps);

    h('p', null,
      'This is where the circuit of §3 earns its keep. We fix it once, with no training and no fitting, and give it the ground state of $H$ at every point of the $(h_1,h_2)$ plane. ' +
      'The output is the expectation value of one measured qubit, $\\langle O\\rangle$. Deep inside the SPT phase it is close to $1$; everywhere else it is close to $0$.', b);
    U.note(b, 'The circuit was designed for the cluster state alone, at $h_1=h_2=0$. That it still <b>sees the phase</b> far from there is the whole point: its error correction absorbs the perturbation.');

    const fig = h('div', 'fig', null, b);
    const row = h('div', 'row', null, fig);
    h('span', 'ctl', 'source', row);
    const tg = U.toggle(row, sources.map(s => ({ value: s.value, label: s.label })), sources[sources.length - 1].value, draw);
    const pair = h('div', 'ph-pair', null, fig);
    const left = h('div', null, null, pair), right = h('div', null, null, pair);
    sub(left, 'QCNN output $\\langle O\\rangle$');
    const dA = plotDiv(left, 440);
    sub(right, 'energy density $e=E/N$');
    const dB = plotDiv(right, 440);
    const cap = h('p', 'caption', '', fig);

    let first = true;
    function draw(val) {
      const s = sources.find(x => x.value === val) || sources[0];
      const x1 = Math.max.apply(null, s.h1), x0 = Math.min.apply(null, s.h1), y1 = Math.max.apply(null, s.h2);
      const ann = (t, y, color) => ({ x: x0 + 0.04 * (x1 - x0 || 1), y, xref: 'x', yref: 'y', text: t, showarrow: false, xanchor: 'left', font: { family: MONO, size: 12, color }, bgcolor: 'rgba(14,17,22,0.72)', borderpad: 3 });
      const A = [traceHm(s, s.z, 'output', U.scale, 0, 1, '⟨O⟩')].concat(boundaryTraces(mps));
      const over = {
        margin: { l: 56, r: 10, t: 34, b: 48 },
        xaxis: { title: axTitle('h₁ / J') }, yaxis: { title: axTitle('h₂ / J') },
        legend: { orientation: 'h', x: 0, y: 1.1, yanchor: 'bottom' },
        annotations: [ann('SPT', -0.25 * y1, C.spt), ann('Paramagnet', 0.62 * y1, C.triv), ann('Antiferromagnet', -0.82 * y1, C.afm)],
      };
      // e-panel: single hue, light = lowest energy
      const eMin = Math.min.apply(null, s.e.flat()), eMax = Math.max.apply(null, s.e.flat());
      const B = [traceHm(s, s.e, 'energy density', [[0, '#d7e7fa'], [1, '#12263d']], eMin, eMax, 'e')];
      const overB = { margin: { l: 56, r: 10, t: 34, b: 48 }, xaxis: { title: axTitle('h₁ / J') }, yaxis: { title: axTitle('h₂ / J') } };
      if (first) { U.plot(dA, A, over); U.plot(dB, B, overB); first = false; }
      else { Plotly.react(dA, A, U.layout(over)); Plotly.react(dB, B, U.layout(overB)); }
      cap.innerHTML = `Fig. 2a — reproduced: QCNN output over the $(h_1,h_2)$ plane from ${s.label.replace(' · ', ', ')}` + (s.quick ? ' (coarse preview grid)' : '') +
        '. Diamonds mark the phase boundaries, taken from the peaks of $-\\partial^2 e/\\partial h_2^2$ of the DMRG energy density ' +
        (mps ? `(N=${mps.N})` : '') + ': blue for the upper, red for the lower. Right: the energy density of the same ground states.';
      U.katex(cap);
    }
    function traceHm(s, z, name, cs, zmin, zmax, bar) {
      return {
        type: 'heatmap', name, x: s.h1, y: s.h2, z, colorscale: cs, zmin, zmax, zsmooth: false, colorbar: cbar(bar),
        hovertemplate: 'h₁ = %{x}<br>h₂ = %{y}<br>' + (name === 'output' ? 'output = %{z:.3f}' : 'e = %{z:.3f}') + '<extra></extra>',
      };
    }
    draw(tg.value);

    h('p', null,
      'Read the left panel in three pieces. The warm band in the middle is the SPT phase, where the circuit outputs a number near $1$. Above it, a large $h_2$ gives the paramagnet, and below it a negative $h_2$ gives the antiferromagnet. ' +
      'Both read near $0$, and the circuit never needs to know which one it is looking at. ' +
      'Now switch from $d=1$ to $d=2$. Each extra layer is another round of error correction, so the output is pushed harder towards $1$ inside the phase and towards $0$ outside, and the crossover narrows around the true boundary.', b);
    h('p', null,
      'The right panel shows why this is not a trivial feat. The energy density is smooth and shows no sign of a transition to the eye. The boundaries live in its second derivative, and the QCNN finds them from the state alone.', b);
  });

  // =====================================================================
  // §6 depth
  // =====================================================================
  U.register('depth', (el, D) => {
    const b = U.head(el, 6, 'Deeper is sharper', 'Fig. 2c');
    if (!U.needData(b, 'cut', 'cut')) return;
    const cut = D.cut, dk = keysOf(cut.qcnn);
    U.quickBadge(b, cut);
    const cols = BLUES(dk.length);
    const crit = cut.critical_h2_paper;

    h('p', null,
      'One layer of this circuit is a modest error corrector: it repairs a single flipped spin per block and nothing more. Stack layers and the repairs compound, so a state that sits close to the SPT fixed point is pulled onto it, and one that does not is pushed away. ' +
      'Let us watch that on a line through the phase diagram, at fixed $h_1=' + cut.h1 + '$, and vary the depth.', b);

    const fig = h('div', 'fig', null, b);
    const row = h('div', 'row', null, fig);
    const read = h('span', 'ctl ph-read', '', row);
    const div = plotDiv(fig, 400);
    const bandLo = 0.30, bandHi = 0.46;
    const traces = dk.map((k, i) => ({
      type: 'scatter', mode: 'lines+markers', name: 'd = ' + k, x: cut.h2, y: cut.qcnn[k],
      line: { color: cols[i], width: 1.6 }, marker: { color: cols[i], size: 4 }, opacity: 1,
      hovertemplate: 'output = %{y:.3f}<extra>d = ' + k + '</extra>',
    }));
    const sl = U.slider(row, {
      label: 'depth', min: 1, max: dk.length, step: 1, value: dk.length, fmt: v => 'd = ' + dk[v - 1],
      onInput: v => highlight(v - 1),
    });
    row.insertBefore(sl.parentNode, read);
    const h2a = Math.min.apply(null, cut.h2), h2b = Math.max.apply(null, cut.h2);
    U.plot(div, traces, {
      margin: { l: 56, r: 16, t: 36, b: 48 }, hovermode: 'x unified',
      xaxis: { title: axTitle('h₂ / J'), range: [h2a, h2b] },
      yaxis: { title: axTitle('QCNN output ⟨O⟩'), range: [-0.03, 1.06] },
      legend: { orientation: 'h', x: 0, y: 1.02, yanchor: 'bottom' },
      shapes: [
        { type: 'rect', xref: 'x', yref: 'paper', x0: bandLo, x1: bandHi, y0: 0, y1: 1, fillcolor: 'rgba(232,228,218,0.09)', line: { width: 0 }, layer: 'below' },
        { type: 'line', xref: 'x', yref: 'paper', x0: crit, x1: crit, y0: 0, y1: 1, line: { color: C.ink, width: 1, dash: 'dash' } },
      ],
      annotations: [
        { x: crit, y: 1, xref: 'x', yref: 'paper', xanchor: 'left', yanchor: 'bottom', showarrow: false, xshift: 4, text: 'paper iDMRG critical point, h₂ = ' + crit, font: { family: MONO, size: 11, color: C.ink } },
        { x: bandLo, y: 0.05, xref: 'x', yref: 'paper', xanchor: 'right', yanchor: 'bottom', showarrow: false, xshift: -4, text: 'zoom in §7', font: { family: MONO, size: 11, color: C.muted } },
      ],
    });

    // upper edge of the plateau: first downward 0.5-crossing right of the maximum
    function edge(y) {
      let m = 0; y.forEach((v, i) => { if (v != null && v > (y[m] == null ? -1 : y[m])) m = i; });
      for (let i = m; i < y.length - 1; i++) {
        if (y[i] != null && y[i + 1] != null && y[i] >= 0.5 && y[i + 1] < 0.5) return cut.h2[i] + (y[i] - 0.5) / (y[i] - y[i + 1]) * (cut.h2[i + 1] - cut.h2[i]);
      }
      return null;
    }
    function highlight(i) {
      Plotly.restyle(div, { 'line.width': dk.map((_, j) => (j === i ? 3.6 : 1.4)), opacity: dk.map((_, j) => (j === i ? 1 : 0.3)), 'marker.size': dk.map((_, j) => (j === i ? 6 : 3)) });
      const e = edge(cut.qcnn[dk[i]]);
      read.innerHTML = 'output crosses 0.5 at h₂ = <b>' + (e == null ? '—' : U.fmt(e, 3)) + '</b> (paper ' + crit + ', linear interpolation)';
    }
    highlight(dk.length - 1);
    h('p', 'caption', `Fig. 2c — reproduced: QCNN output along the cut $h_1=${cut.h1}$ for N=${cut.N}, depth by depth (light to dark blue = shallow to deep). ` +
      'The dashed line is the paper’s iDMRG critical point; the band is the window we magnify in §7.' + (cut.quick ? ' Preview grid: only a few h₂ values.' : ''), fig);

    U.note(b, 'A finite chain cannot show a true jump. The output is a smooth step, and the sharpening with depth is how we know the circuit is <b>approaching</b> a sharp order parameter.');
    h('p', null,
      'Drag the depth slider. The plateau in the middle is the SPT phase. On both sides the output is pushed down towards $0$, and the deeper the circuit, the steeper the drop. ' +
      'The readout gives the $h_2$ where the output crosses one half. As the depth grows, this crossing should settle towards the critical point from the paper’s iDMRG calculation. ' +
      'The remaining gap is finite-size effect (our chain has N=' + cut.N + ' sites) and the coarse step between grid points, not a flaw in the circuit.', b);
  });

  // =====================================================================
  // §7 samples
  // =====================================================================
  U.register('samples', (el, D) => {
    const b = U.head(el, 7, 'How many copies do we need?', 'Eq. 4 and Fig. 2d');
    if (!U.needData(b, 'cut', 'cut')) return;
    const cut = D.cut, sc = cut.sample_complexity;
    if (!sc) { U.needData(b, 'cut.sample_complexity', 'cut'); return; }
    U.quickBadge(b, cut);
    const qk = keysOf(sc.qcnn), sk = keysOf(sc.sop);
    const qc = BLUES(qk.length), scol = REDS(sk.length);
    const CAP = 100, crit = cut.critical_h2_paper;

    h('p', null,
      'Recognising a phase is a statistical decision. Every measurement of the output (or of a string order parameter, SOP) is a coin flip that gives $+1$ with probability $p=(\\langle O\\rangle+1)/2$. ' +
      'To decide with 95% confidence that the state really is in the phase, that is, that the coin favours heads, we need at least', b);
    h('div', null, String.raw`$$M\;\ge\;\frac{1.96^{2}}{\left(\arcsin\sqrt{p}-\arcsin\sqrt{p_0}\right)^{2}},\qquad p_0=\tfrac12 \tag{4}$$`, b);
    U.note(b, 'The printed Eq. 4 has $\\sqrt{\\arcsin p_0}$. We read it as <b>$\\arcsin\\sqrt{p_0}$</b>, so both terms are the same variance-stabilising transform of a coin bias. For $p\\le p_0$ the formula means nothing, and we plot a gap.');
    h('p', null,
      'Near the critical point the SOP expectation $\\langle S\\rangle$ falls towards $0$, so the coin becomes fair, $p\\to\\tfrac12$, and the denominator collapses: $M$ blows up. ' +
      'The QCNN does something clever with the very same state. Its error correction pushes $p$ towards $1$, so far fewer copies are enough.', b);

    // ---- data window ----
    const lo = 0.30, hi = 0.46;
    let idx = cut.h2.map((v, i) => (v >= lo - 1e-9 && v <= hi + 1e-9 ? i : -1)).filter(i => i >= 0);
    const fallback = idx.length < 2;
    if (fallback) idx = cut.h2.map((_, i) => i);
    const xs = idx.map(i => cut.h2[i]);

    const fig = h('div', 'fig', null, b);
    const pair = h('div', 'ph-pair ph-pair-s', null, fig);
    const left = h('div', null, null, pair), right = h('div', null, null, pair);
    sub(left, 'copies $M$ needed, 95% confidence' + (fallback ? ' (whole cut: this grid has no points in 0.30–0.46)' : ''));
    const dM = plotDiv(left, 420);
    sub(right, 'advantage: best SOP $M$ over QCNN $M$');
    const dI = plotDiv(right, 420);

    const traces = [], capped = [];
    function series(label, arr, color, group) {
      const y = idx.map(i => arr[i]);
      traces.push({
        type: 'scatter', mode: 'lines+markers', name: label, legendgroup: group, x: xs,
        y: y.map(v => (v == null ? null : Math.min(v, CAP))), customdata: y,
        line: { color, width: 1.8 }, marker: { color, size: 5 }, connectgaps: false,
        hovertemplate: 'M = %{customdata:.3g}<extra>' + label + '</extra>',
      });
      const cx = [], cy = [];
      y.forEach((v, j) => { if (v != null && v > CAP) { cx.push(xs[j]); cy.push(CAP); } });
      if (cx.length) capped.push({ type: 'scatter', mode: 'markers', x: cx, y: cy, legendgroup: group, showlegend: false, hoverinfo: 'skip', marker: { symbol: 'triangle-up', size: 9, color, line: { color: C.bg, width: 1 } } });
    }
    qk.forEach((k, i) => series('QCNN d=' + k, sc.qcnn[k], qc[i], 'q'));
    sk.forEach((k, i) => series('SOP L=' + k, sc.sop[k], scol[i], 's'));
    const shapes = (crit >= xs[0] && crit <= xs[xs.length - 1])
      ? [{ type: 'line', xref: 'x', yref: 'paper', x0: crit, x1: crit, y0: 0, y1: 1, line: { color: C.ink, width: 1, dash: 'dash' } }] : [];
    U.plot(dM, traces.concat(capped), {
      margin: { l: 56, r: 12, t: 12, b: 48 }, hovermode: 'closest',
      xaxis: { title: axTitle('h₂ / J'), range: [xs[0] - 0.004 * (xs[xs.length - 1] - xs[0]), xs[xs.length - 1] + 0.004 * (xs[xs.length - 1] - xs[0])] },
      yaxis: { title: axTitle('M (capped at ' + CAP + ')'), range: [0, CAP * 1.07] },
      legend: { x: 0.01, y: 0.99, xanchor: 'left', yanchor: 'top', bgcolor: 'rgba(14,17,22,0.7)' },
      shapes,
    });

    // ---- inset: ratio vs depth ----
    const ins = cut.inset;
    if (ins && ins.depth && ins.depth.length) {
      const pts = ins.depth.map((d, i) => [d, ins.ratio[i]]).filter(p => p[1] != null && p[1] > 0);
      const d0 = Math.min.apply(null, ins.depth) - 0.25, d1 = Math.max.apply(null, ins.depth) + 0.25;
      const grid = Array.from({ length: 25 }, (_, i) => d0 + (d1 - d0) * i / 24);
      const fit = ins.fit && ins.fit.a != null ? ins.fit : null;
      const tr = [
        { type: 'scatter', mode: 'lines', name: 'paper 1.73·e^(0.28 d)', x: grid, y: grid.map(d => 1.73 * Math.exp(0.28 * d)), line: { color: C.ink, width: 1.4, dash: 'dash' }, hoverinfo: 'skip' },
      ];
      if (fit) tr.push({ type: 'scatter', mode: 'lines', name: 'our fit', x: grid, y: grid.map(d => fit.a * Math.exp(fit.b * d)), line: { color: C.conv, width: 2 }, hoverinfo: 'skip' });
      tr.push({ type: 'scatter', mode: 'markers', name: 'ours', x: pts.map(p => p[0]), y: pts.map(p => p[1]), marker: { color: C.conv, size: 9, line: { color: C.ink, width: 1 } }, hovertemplate: 'd = %{x}<br>ratio = %{y:.3g}<extra></extra>' });
      U.plot(dI, tr, {
        margin: { l: 56, r: 12, t: 12, b: 48 },
        xaxis: { title: axTitle('depth d'), dtick: 1, range: [d0, d1] },
        yaxis: { title: axTitle('M_SOP,best / M_QCNN'), type: 'log' },
        legend: { x: 0.02, y: 0.98, xanchor: 'left', yanchor: 'top', bgcolor: 'rgba(14,17,22,0.7)' },
        annotations: fit ? [{ xref: 'paper', yref: 'paper', x: 0.02, y: 0.04, xanchor: 'left', yanchor: 'bottom', showarrow: false, align: 'left',
          text: 'fit  a = ' + U.fmt(fit.a, 2) + ', b = ' + U.fmt(fit.b, 3) + '<br>at h₂ = ' + sgn(ins.h2), font: { family: MONO, size: 11, color: C.conv } }] : [],
      });
    } else {
      h('div', 'notice', 'No inset data in this build.', dI);
    }

    const insOk = ins && ins.fit && ins.fit.b > 0;
    h('p', 'caption', 'Fig. 2d — ' + (insOk ? 'reproduced' : 'main panel reproduced, inset not') + ': copies needed to certify the SPT phase near criticality. Blue: QCNN at each depth. Red: string order parameters of length L. ' +
      'Gaps mean the state is not recognised as SPT there (p ≤ ½). Triangles mark values above ' + CAP + ' (hover for the true number). Dashed line: critical point. ' +
      'Right: ratio of the best SOP to the QCNN at h₂ = ' + sgn(ins ? ins.h2 : 0) + '; solid is our fit a·e^(b d), dashed is the paper’s.' + (cut.quick ? ' Preview build (N=' + cut.N + ').' : ''), fig);

    const fitTxt = ins && ins.fit && ins.fit.a != null ? `Our fit gives $a=${U.fmt(ins.fit.a, 2)}$ and $b=${U.fmt(ins.fit.b, 3)}$.` : '';
    h('p', null,
      'The effect grows with depth. Each extra layer corrects more errors, so $p$ for the QCNN moves further from $\\tfrac12$ while the SOPs are stuck with whatever the state gives them. ' +
      'The paper fits the advantage as $M_\\mathrm{SOP}/M_\\mathrm{QCNN}\\approx 1.73\\,e^{0.28\\,d}$, an exponential improvement. ' + fitTxt +
      (cut.quick ? ' With only a coarse preview grid and depths up to ' + qk[qk.length - 1] + ', the fit is not yet a meaningful test of that exponent.' : '') +
      (!cut.quick && !insOk ? ' Here our chain is too short to confirm it. ' +
        'At $N=' + cut.N + '$ the depth-' + qk[qk.length - 1] + ' circuit ends with only three sites, so its measured operator spans the whole open chain and feels both edges, and its output starts to fall before the true transition. ' +
        'The QCNN still beats every string order parameter at every depth over most of the window, and $d=2$ beats $d=1$. But the last depth is finite-size limited. The paper flags exactly this regime (the shaded band of its Fig. 2d) and used $N=135$. Testing the exponent properly would need $N=243$, which we did not run.' : ''), b);
  });

  // =====================================================================
  // §8 rgflow
  // =====================================================================
  U.register('rgflow', (el, D) => {
    const b = U.head(el, 8, 'Watching renormalisation happen', 'syndromes level by level');
    const ok = [U.needData(b, 'cut', 'cut'), U.needData(b, 'phase_mps', 'phase_mps')];
    if (!ok[0] && !ok[1]) return;
    U.quickBadge(b, ok[0] ? D.cut : D.phase_mps);

    h('p', null,
      'Every pooling layer measures the qubits it throws away, and each measurement is a syndrome: a flag that says whether the neighbourhood looks like the perfect cluster state or not. ' +
      'The fraction of flags that fire at level $\\ell$ is the <b>syndrome density</b>. It tells us how far the coarse-grained state still is from the fixed point, so how it changes from level to level is a renormalisation-group flow.', b);
    U.note(b, 'Compare with Fig. 10c of the paper: for a concatenated code below its threshold, the logical error rate shrinks every level; above it, errors grow. Here the <b>phase</b> plays the role of the noise level.');

    // ---- (a) line cut ----
    if (ok[0]) {
      const cut = D.cut, lv = keysOf(cut.syndrome_density), lx = lv.map(Number);
      const targets = [-1.2, -0.4, 0.2, 0.38, 0.42, 0.6];
      const PAL = [C.afm, '#b48a22', '#3f8fd8', '#2fa07d', '#d9712c', '#9a7fd1'];   // validated dark-mode categorical
      const sel = [];
      targets.forEach((t, ti) => { const i = closest(cut.h2, t); if (!sel.some(s => s.i === i)) sel.push({ i, v: cut.h2[i], c: PAL[ti] }); });

      const fig = h('div', 'fig', null, b);
      const row = h('div', 'row', null, fig);
      h('span', 'ctl', 'h₂ values', row);
      const tg = U.toggle(row, sel.map(s => ({ value: s.i, label: 'h₂ = ' + sgn(s.v) })), sel.map(s => s.i), () => { stop(); draw(); }, { multi: true });
      tg.classList.add('ph-swatch');
      [...tg.children].forEach((btn, k) => btn.style.setProperty('--c', sel[k].c));
      const play = h('button', 'pill', 'play', row); play.type = 'button';
      const div = plotDiv(fig, 380);
      let raf = null, chosen = [];

      function yAt(s, x) {                                    // interpolate the line of one h2 value
        const ys = lv.map(k => cut.syndrome_density[k][s.i]);
        for (let j = 0; j < lx.length - 1; j++) {
          if (x <= lx[j + 1] + 1e-9) { const t = (x - lx[j]) / (lx[j + 1] - lx[j]); return ys[j] == null || ys[j + 1] == null ? null : ys[j] + t * (ys[j + 1] - ys[j]); }
        }
        return ys[ys.length - 1];
      }
      function draw() {
        const on = tg.value;
        chosen = sel.filter(s => on.includes(s.i));
        const lines = chosen.map(s => ({
          type: 'scatter', mode: 'lines+markers', name: 'h₂ = ' + sgn(s.v), x: lx, y: lv.map(k => cut.syndrome_density[k][s.i]),
          line: { color: s.c, width: 2 }, marker: { color: s.c, size: 6 }, hovertemplate: 'level %{x}: density %{y:.3f}<extra>h₂ = ' + sgn(s.v) + '</extra>',
        }));
        const dots = chosen.map(s => ({
          type: 'scatter', mode: 'markers', x: [lx[0]], y: [yAt(s, lx[0])], showlegend: false, hoverinfo: 'skip',
          marker: { color: s.c, size: 14, line: { color: C.ink, width: 2 } },
        }));
        const over = {
          margin: { l: 56, r: 12, t: 12, b: 48 }, hovermode: 'closest',
          xaxis: { title: axTitle('RG level ℓ (number of layers applied)'), dtick: 1, range: [lx[0] - 0.2, lx[lx.length - 1] + 0.2] },
          yaxis: { title: axTitle('syndrome density'), range: [-0.02, 0.52] },
          legend: { x: 1.01, y: 1, xanchor: 'left' },
          shapes: [{ type: 'line', xref: 'paper', yref: 'y', x0: 0, x1: 1, y0: 0, y1: 0, line: { color: C.triv, width: 1, dash: 'dot' } }],
          annotations: [{ xref: 'paper', yref: 'y', x: 0.01, y: 0, xanchor: 'left', yanchor: 'bottom', showarrow: false, text: 'fixed point: no errors', font: { family: MONO, size: 10, color: C.triv } }],
        };
        if (!div.data) U.plot(div, lines.concat(dots), over); else Plotly.react(div, lines.concat(dots), U.layout(over));
      }
      function stop() { if (raf) cancelAnimationFrame(raf); raf = null; play.textContent = 'play'; }
      function run() {
        if (!chosen.length) return;
        stop(); play.textContent = 'stop';
        const t0 = performance.now(), dur = 3200, n = chosen.length;
        const idxs = chosen.map((_, j) => n + j);
        const step = now => {
          const t = Math.min(1, (now - t0) / dur), e = t < 0.5 ? 2 * t * t : 1 - Math.pow(-2 * t + 2, 2) / 2;   // ease in-out
          const x = lx[0] + e * (lx[lx.length - 1] - lx[0]);
          Plotly.restyle(div, { x: chosen.map(() => [x]), y: chosen.map(s => [yAt(s, x)]) }, idxs);
          if (t < 1) raf = requestAnimationFrame(step); else { raf = null; play.textContent = 'play'; }
        };
        raf = requestAnimationFrame(step);
      }
      play.addEventListener('click', () => (raf ? (stop(), draw()) : (draw(), run())));
      draw();
      h('p', 'caption', `Syndrome density per level along the cut $h_1=${cut.h1}$ (N=${cut.N}). Press play to carry a marker through the levels. ` +
        'The dotted line at zero is the fixed point. Pills double as a legend: click to add or remove a curve.' + (cut.quick ? ' Preview build: the nearest available grid values replace the requested ones.' : ''), fig);
    }

    h('p', null,
      'Pick a value of $h_2$ deep in the SPT phase and press play. The error density drops with each level, towards zero. Zero is the fixed point of this flow, the perfect cluster state. ' +
      'Outside the phase there is no such fixed point, so the density stays finite and drifts up towards the value $\\tfrac12$ of a random state. Close to the critical point the flow hesitates: it neither collapses nor grows quickly, which is the signature of criticality seen from the circuit.', b);

    // ---- (b) phase plane ----
    if (ok[1]) {
      const mps = D.phase_mps, lv = keysOf(mps.syndrome_density);
      const fig = h('div', 'fig', null, b);
      const row = h('div', 'row', null, fig);
      h('span', 'ctl', 'level', row);
      const tg = U.toggle(row, lv.map(k => ({ value: k, label: 'level ' + k })), lv[0], draw);
      const div = plotDiv(fig, 420);
      let first = true;
      function draw(k) {
        const z = mps.syndrome_density[k];
        const t = [{
          type: 'heatmap', name: 'syndrome density', x: mps.h1, y: mps.h2, z, zmin: 0, zmax: 0.5, zsmooth: false,
          colorscale: [[0, '#14202a'], [0.5, '#7a3a3a'], [1, '#e0564c']], colorbar: cbar('density'),
          hovertemplate: 'h₁ = %{x}<br>h₂ = %{y}<br>density = %{z:.3f}<extra></extra>',
        }].concat(boundaryTraces(mps));
        const over = { margin: { l: 56, r: 10, t: 34, b: 48 }, xaxis: { title: axTitle('h₁ / J') }, yaxis: { title: axTitle('h₂ / J') }, legend: { orientation: 'h', x: 0, y: 1.06, yanchor: 'bottom' } };
        if (first) { U.plot(div, t, over); first = false; } else Plotly.react(div, t, U.layout(over));
      }
      draw(tg.value);
      h('p', 'caption', `Syndrome density over the whole phase plane (N=${mps.N} MPS). Switch the level and watch the SPT region empty out while the rest stays red. ` +
        'The colour scale is fixed at 0–0.5, so the two levels compare honestly.', fig);
      h('p', null,
        'The same flow is visible over the whole plane. At level 1 the SPT region is already quiet; at the next level it is quieter still, while the paramagnet and antiferromagnet stay loud. ' +
        'The boundary between quiet and loud sharpens as we coarse-grain, the picture of a phase boundary as an unstable fixed point of the RG.', b);
    }
  });
})();
