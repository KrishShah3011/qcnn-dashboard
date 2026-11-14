/* Section: learning (§9) — a parameterised QCNN trained on the solvable line (Fig. 4, Fig. 9). */
(function () {
  'use strict';
  const U = window.QCNNUI;
  if (!U) return;
  const { C, h, svgEl: S, fmt } = U;

  const LAYERS = ['C1', 'C2', 'C3', 'C4', 'V1', 'V2', 'FC'];
  const COLOR = { C1: C.conv, C2: C.conv, C3: C.conv, C4: C.conv, V1: C.pool, V2: C.pool, FC: C.fc };
  const FILL = { conv: 'rgba(47,127,208,.28)', pool: 'rgba(240,138,36,.28)', fc: 'rgba(122,63,176,.42)' };

  U.register('learning', (el, D) => {
    const b = U.head(el, 9, 'Learning a QCNN from 40 solvable points', 'Fig. 4 and Fig. 9');
    if (!U.needData(b, 'trained', 'trained')) return;
    const T = D.trained;
    const P = T.params;
    const exact = D.phase_ed && D.phase_ed.qcnn && D.phase_ed.qcnn['1'] ? D.phase_ed : null;
    U.quickBadge(b, T);

    U.note(b, '<b>Why not just read the circuit off Section 3?</b> Because on a real device we would not know the ' +
      'Hamiltonian\'s symmetry in advance. Learning asks less of us: label some states, let the optimiser do the rest.');
    h('p', null,
      'Everything so far used a circuit we designed by hand. Now we hand the optimiser a circuit with the same <em>shape</em>, ' +
      'but every gate is a free unitary, initialised at random, and we ask it to find the phase boundary from labelled examples. ' +
      'The training signal is the mean squared error over the training set $S$ with labels $y_i\\in\\{0,1\\}$ ' +
      '(1 in the SPT phase), minimised over all ' + P.total + ' parameters $c$:', b);
    h('p', null, '$$\\mathrm{MSE}(c)=\\frac{1}{2|S|}\\sum_{i\\in S}\\bigl(y_i-f_c(\\psi_i)\\bigr)^2,\\qquad f_c(\\psi)=\\tfrac12\\bigl(1+\\langle X_6\\rangle\\bigr).\\tag{1}$$', b);

    // ---------------- Fig. 9: circuit + parameter bar ----------------
    const f1 = h('div', 'fig', null, b);
    const top = h('div', 'sl-top', null, f1);
    const left = h('div', 'sl-circ', null, top);
    const svg = circuit(left, P);
    const ro = h('p', 'caption sl-ro', '', left);
    const barDiv = h('div', 'sl-bar', null, top);
    wireHover(svg, ro, P);
    h('p', 'caption', 'Fig. 9 — reproduced: the parameterised QCNN for N = 15 (n = 3, d = 1). Each box is a free unitary, shared by all blocks of its layer. ' +
      'Hover a layer for its parameter count. Right: all ' + P.total + ' parameters, with the fully connected layer F holding the great majority. ' +
      'The paper quotes 1309; this layout counts ' + P.total + '.', f1);
    barChart(barDiv, P);

    h('p', null,
      'Count the parameters layer by layer. A general unitary on $m$ qubits has $4^m-1$ real parameters, so a three-qubit block costs $63$ and the pooling gates ' +
      '$V_1,V_2$ cost $3$ each. The four-qubit $U_1$ is a product of six two-qubit unitaries with $15$ parameters apiece. The only large item is the last box: ' +
      'a $32\\times32$ unitary on five qubits, $32^2-1=1023$ parameters. Crucially, <b>only that box grows with the number of survivors</b>; the convolution and pooling layers ' +
      'are translation invariant, so adding more sites adds more blocks but not more numbers. In the paper\'s scaling this is the $O(\\log N)$ parameter count, ' +
      'and it is why this model has little room to overfit.', b);

    // ---------------- training + player ----------------
    U.note(b, '<b>Optimiser.</b> The paper estimates gradients by finite differences and steps with a bold-driver rule. ' +
      'We use autograd with Adam (' + (T.optimizer || 'lr = 0.01') + '). The gradient is the same up to $O(\\epsilon^2)$ ' +
      'and far cheaper; the paper\'s own outlook suggests backpropagation. Initialisation follows the paper: coefficients uniform in $[0,2\\pi)$.');
    h('p', null,
      'Here is the surprising part. We train on <b>one line only</b>: ' + T.train.h1.length + ' ground states along $h_2=0$, where the Hamiltonian is solvable by a Jordan–Wigner transformation ' +
      'and the transition sits at $h_1=1$. That line never crosses the lower boundary of the SPT phase. And yet, once trained, the circuit labels the whole $(h_1,h_2)$ plane ' +
      'in rough agreement with the exact QCNN of Section 5. Press play to watch the picture sharpen.', b);

    const f2 = h('div', 'fig', null, b);
    const ctl = h('div', 'row', null, f2);
    const play = h('button', 'sl-play', 'Play', ctl);
    play.type = 'button';
    const snaps = T.snapshots || [];
    const hist = T.history || { iter: [], mse: [] };
    const mseAt = it => { let k = -1; hist.iter.forEach((x, i) => { if (x <= it) k = i; }); return k < 0 ? null : hist.mse[k]; };
    const sl = U.slider(ctl, { label: 'snapshot', min: 0, max: Math.max(0, snaps.length - 1), step: 1, value: 0,
      fmt: i => 'iter ' + (snaps[i] ? snaps[i].iter : '—'), onInput: i => { stop(); show(i); } });
    const stat = h('span', 'sl-stat', '', ctl);

    // training curve
    const curveDiv = h('div', 'sl-curve', null, f2);
    const snapPts = snaps.map(s => ({ x: s.iter, y: mseAt(s.iter) })).filter(p => p.y != null);
    U.plot(curveDiv, [
      { x: hist.iter, y: hist.mse, mode: 'lines', name: 'MSE', line: { color: C.conv, width: 1.8 }, hovertemplate: 'iter %{x}<br>MSE %{y:.4f}<extra></extra>' },
      { x: snapPts.map(p => p.x), y: snapPts.map(p => p.y), mode: 'markers', name: 'snapshots', marker: { size: 7, color: C.bg, line: { color: C.ink, width: 1.4 } },
        hovertemplate: 'snapshot, iter %{x}<br>MSE %{y:.4f}<extra></extra>' },
      { x: [null], y: [null], mode: 'markers', name: 'shown below', marker: { size: 12, color: C.pool, line: { color: C.bg, width: 2 } }, hoverinfo: 'skip' },
    ], {
      height: 230, margin: { l: 64, r: 16, t: 10, b: 44 }, showlegend: true,
      legend: { orientation: 'h', x: 1, xanchor: 'right', y: 1.12 },
      xaxis: { title: { text: 'iteration' } }, yaxis: { type: 'log', title: { text: 'MSE (Eq. 1)' } },
    });

    // heatmaps
    const cols = h('div', 'grid2 sl-heat', null, f2);
    const mk = (title) => { const c = h('div', null, null, cols); const t = h('div', 'sl-sub', title, c); const d = h('div', 'sl-hm', null, c); return { t, d }; };
    const A = mk(''); const B = exact ? mk('exact QCNN, N = 15, d = 1 (Section 5)') : null;
    if (!exact) h('div', 'notice', 'The exact QCNN grid (<code>phase_ed</code>) is missing, so only the trained prediction is shown. Run <code>.venv/Scripts/python build_data.py --stage phase_ed</code>.', cols);

    const xmax = Math.max.apply(null, T.h1);
    const tx = [], tyv = [];
    T.train.h1.forEach((v, i) => { if (v <= xmax + 1e-9) { tx.push(v); tyv.push(T.train.y[i]); } });
    const heat = (x, y, z, showscale) => ({
      type: 'heatmap', x, y, z, zmin: 0, zmax: 1, colorscale: U.scale, showscale,
      colorbar: { thickness: 10, len: 0.9, outlinewidth: 0, tickfont: { family: '"JetBrains Mono", monospace', size: 10, color: C.muted }, title: { text: 'f', font: { size: 11, color: C.muted } } },
      hovertemplate: 'h₁ = %{x}<br>h₂ = %{y}<br>f = %{z:.3f}<extra></extra>',
    });
    const dots = {
      type: 'scatter', mode: 'markers', name: 'training points', x: tx, y: tx.map(() => 0), customdata: tyv,
      marker: { size: 7, color: '#aab1ba', line: { color: C.bg, width: 1.5 } },
      hovertemplate: 'training point<br>h₁ = %{x:.2f}, label y = %{customdata}<extra></extra>',
    };
    const hmLay = U.layout({
      height: 300, margin: { l: 52, r: 56, t: 6, b: 46 }, showlegend: false,
      xaxis: { title: { text: 'h₁' } }, yaxis: { title: { text: 'h₂' } },
    });
    const diffNow = z => {
      if (!exact) return null;
      const E = exact.qcnn['1']; let s = 0, n = 0;
      E.forEach((row, i) => row.forEach((v, j) => { if (z[i] && z[i][j] != null && v != null) { s += Math.abs(z[i][j] - v); n++; } }));
      return n ? s / n : null;
    };
    const first = snaps.length ? snaps[0] : { iter: 0, grid: T.final.grid };
    U.plot(A.d, [heat(T.h1, T.h2, first.grid, !exact), dots], hmLay);
    if (exact) U.plot(B.d, [heat(exact.h1, exact.h2, exact.qcnn['1'], true), dots], hmLay);

    function show(i) {
      const s = snaps[i]; if (!s) return;
      sl.set(i);
      const m = mseAt(s.iter), d = diffNow(s.grid);
      A.t.textContent = 'trained circuit, iteration ' + s.iter;
      stat.textContent = 'MSE ' + fmt(m, 4) + (d != null ? ' · mean |f − exact| over the grid ' + fmt(d, 3) : '');
      Plotly.react(A.d, [heat(T.h1, T.h2, s.grid, !exact), dots], hmLay);
      Plotly.restyle(curveDiv, { x: [[s.iter]], y: [[m]] }, [2]);
    }
    let timer = null, cur = 0;
    function stop() { if (timer) { clearInterval(timer); timer = null; } play.textContent = 'Play'; }
    play.addEventListener('click', () => {
      if (timer) { stop(); return; }
      if (snaps.length < 2) return;
      cur = +sl.value; if (cur >= snaps.length - 1) cur = -1;
      play.textContent = 'Pause';
      const step = () => { cur++; show(cur); if (cur >= snaps.length - 1) stop(); };
      step(); timer = setInterval(step, 800);
    });
    // initial state once the Plotly curve exists
    Promise.resolve().then(() => setTimeout(() => show(0), 0));
    A.t.textContent = 'trained circuit, iteration ' + first.iter;

    h('p', 'caption', 'Fig. 4 — reproduced: the circuit is trained only on the ' + T.train.h1.length + ' points along h₂ = 0 (gray dots; points beyond h₁ = ' + fmt(xmax, 1) +
      ' are off the plotted grid). Left: the QCNN output f on the full grid at the chosen snapshot, the orange marker locating that snapshot on the training curve. ' +
      'Right: the hand-built circuit of Section 5 for comparison. Output f = 1 is read as SPT (yellow), f = 0 as trivial (dark teal).', f2);

    // ---------------- final fit along the training line ----------------
    h('p', null,
      'Finally, the training line itself. A perfect classifier would trace the step: output 1 for $h_1<1$, output 0 beyond. With so few parameters and so few points we do not expect a perfect step, ' +
      'but the trained outputs should fall as $h_1$ crosses the solvable transition, and they should follow the hand-built circuit rather than the labels, which tells us the optimiser found the same physics.', b);
    const f3 = h('div', 'fig', null, b);
    const lineDiv = h('div', 'sl-line', null, f3);
    const stepY = T.train.y, stepX = T.train.h1;
    const tr = [
      { x: stepX, y: stepY, mode: 'markers', marker: { symbol: 'x-thin', size: 9, color: C.ink, line: { color: C.ink, width: 1.5 } }, name: 'training label y', hovertemplate: 'h₁ = %{x:.2f}<br>label %{y}<extra></extra>' },
      { x: stepX, y: T.final.train_pred, mode: 'lines+markers', line: { color: C.conv, width: 2 }, marker: { size: 7, color: C.conv, line: { color: C.bg, width: 1.5 } }, name: 'trained f', hovertemplate: 'h₁ = %{x:.2f}<br>trained f = %{y:.3f}<extra></extra>' },
    ];
    if (exact) {
      let r0 = 0; exact.h2.forEach((v, i) => { if (Math.abs(v) < Math.abs(exact.h2[r0])) r0 = i; });
      tr.push({ x: exact.h1, y: exact.qcnn['1'][r0], mode: 'lines+markers', line: { color: C.spt, width: 2, dash: 'dash' }, marker: { size: 6, symbol: 'diamond', color: C.spt }, name: 'exact QCNN, h₂ = 0', hovertemplate: 'h₁ = %{x:.2f}<br>exact f = %{y:.3f}<extra></extra>' });
    }
    U.plot(lineDiv, tr, {
      height: 320, margin: { l: 56, r: 16, t: 34, b: 46 },
      legend: { orientation: 'h', x: 0, y: 1.22 },
      xaxis: { title: { text: 'h₁   (h₂ = 0)' } }, yaxis: { title: { text: 'output f' }, range: [-0.05, 1.12] },
      shapes: [{ type: 'line', x0: 1, x1: 1, y0: -0.05, y1: 1.05, line: { color: C.rule, width: 1.5, dash: 'dash' } }],
      annotations: [{ x: 1, y: -0.05, xanchor: 'left', yanchor: 'bottom', xshift: 6, text: 'h₁ = 1', showarrow: false, font: { size: 10, color: C.muted } }],
    });
    h('p', 'caption', 'Fig. 4 (line cut) — reproduced: final outputs on the training points against the target labels (crosses: 1 in the SPT phase, 0 beyond)' + (exact ? ' and the exact QCNN along the same line' : '') +
      '. The dashed vertical line marks the solvable transition at h₁ = 1.', f3);

    h('p', null,
      'What should we take from this? Forty labelled points on a line that never touches the lower phase boundary were enough to teach a ' + P.total +
      '-parameter circuit a two-dimensional phase diagram, because the symmetry and locality built into the circuit leave the optimiser very few ways to be wrong.', b);
  });

  // ---------------- Fig. 9 circuit drawing ----------------
  function circuit(parent, P) {
    const Y = i => 50 + i * 21;
    const svg = S('svg', { viewBox: '0 0 560 372', role: 'img', 'aria-label': 'Parameterised QCNN circuit for 15 qubits: convolution layers C1 to C4, pooling V1 and V2 with X measurements, and one fully connected 5-qubit gate.' }, parent);
    const xs = { c1a: 66, c1b: 102, c2: 146, c3: 186, c4: 226, v1: 280, v2: 320, met: 358, fc: 410 };
    const fcL = 396, fcR = 426, outX = 470;
    const g = {}; LAYERS.forEach(k => { g[k] = S('g', { class: 'lay', 'data-layer': k, tabindex: 0 }, svg); });
    const wires = S('g', {}, svg);
    svg.insertBefore(wires, svg.firstChild);
    for (let q = 0; q < 15; q++) {
      const surv = q % 3 === 0;
      S('line', { x1: 30, x2: surv ? fcL : xs.met, y1: Y(q), y2: Y(q), stroke: '#4a525c', 'stroke-width': 1 }, wires);
      S('text', { x: 24, y: Y(q) + 3, 'text-anchor': 'end', 'font-size': 9, text: q }, wires);
    }
    S('line', { x1: fcR, x2: outX - 9, y1: Y(6), y2: Y(6), stroke: '#4a525c' }, wires);

    const box = (grp, cx, a, bq, w, color, fill, label) => {
      S('rect', { x: cx - w / 2, y: Y(a) - 8, width: w, height: Y(bq) - Y(a) + 16, rx: 2, fill, stroke: color, 'stroke-width': 1.2 }, grp);
      S('text', { x: cx, y: (Y(a) + Y(bq)) / 2 + 3, 'text-anchor': 'middle', 'font-size': 9.5, fill: C.ink, text: label }, grp);
    };
    // C1: slice A (k = 0, 2) then slice B (k = 1, 3), 4 qubits each
    [0, 2].forEach(k => box(g.C1, xs.c1a, 3 * k, 3 * k + 3, 22, C.conv, FILL.conv, 'U₁'));
    [1, 3].forEach(k => box(g.C1, xs.c1b, 3 * k, 3 * k + 3, 22, C.conv, FILL.conv, 'U₁'));
    for (let k = 0; k < 4; k++) box(g.C2, xs.c2, 3 * k + 2, 3 * k + 4, 22, C.conv, FILL.conv, 'U₂');
    for (let k = 0; k < 5; k++) box(g.C3, xs.c3, 3 * k, 3 * k + 2, 22, C.conv, FILL.conv, 'U₃');
    for (let k = 0; k < 4; k++) box(g.C4, xs.c4, 3 * k + 1, 3 * k + 3, 22, C.conv, FILL.conv, 'U₄');

    // pooling: survivors r = 3k. X-measured r+1 controls V1 on r; X-measured r+2 controls V2 on r+3.
    const control = (grp, x, from, to) => {
      S('line', { x1: x, x2: x, y1: Y(from), y2: Y(to), stroke: C.pool, 'stroke-width': 1.2 }, grp);
      S('circle', { cx: x, cy: Y(from), r: 3, fill: C.pool }, grp);
    };
    const meter = (grp, x, q) => {
      S('rect', { x: x - 8, y: Y(q) - 7, width: 16, height: 14, rx: 2, fill: C.bg, stroke: C.pool }, grp);
      S('text', { x, y: Y(q) + 3.5, 'text-anchor': 'middle', 'font-size': 9, fill: C.pool, text: 'X' }, grp);
    };
    const small = (grp, x, q, label) => {
      S('rect', { x: x - 11, y: Y(q) - 8, width: 22, height: 16, rx: 2, fill: FILL.pool, stroke: C.pool, 'stroke-width': 1.2 }, grp);
      S('text', { x, y: Y(q) + 3, 'text-anchor': 'middle', 'font-size': 9, fill: C.ink, text: label }, grp);
    };
    for (let k = 0; k < 5; k++) {
      const r = 3 * k;
      control(g.V1, xs.v1, r + 1, r); small(g.V1, xs.v1, r, 'V₁'); meter(g.V1, xs.met, r + 1);
      if (r + 3 < 15) { control(g.V2, xs.v2, r + 2, r + 3); small(g.V2, xs.v2, r + 3, 'V₂'); }
      meter(g.V2, xs.met, r + 2);
    }
    // wires pass behind the control dots; V2 meters sit on the wire r+2 which also ends at xs.met

    // FC on survivors 0,3,6,9,12
    S('rect', { x: fcL, y: Y(0) - 10, width: fcR - fcL, height: Y(12) - Y(0) + 20, rx: 2, fill: FILL.fc, stroke: C.fc, 'stroke-width': 1.4 }, g.FC);
    S('text', { x: xs.fc, y: Y(3.6), 'text-anchor': 'middle', 'font-size': 13, fill: C.ink, text: 'F' }, g.FC);
    S('text', { x: xs.fc, y: Y(4.5) + 3, 'text-anchor': 'middle', 'font-size': 8.5, text: '5 qubits' }, g.FC);
    [0, 3, 6, 9, 12].forEach(q => S('circle', { cx: fcL, cy: Y(q), r: 2.3, fill: C.fc }, g.FC));

    // output meter
    S('rect', { x: outX - 9, y: Y(6) - 8, width: 18, height: 16, rx: 2, fill: C.bg, stroke: C.fc, 'stroke-width': 1.2 }, svg);
    S('text', { x: outX, y: Y(6) + 3.5, 'text-anchor': 'middle', 'font-size': 9, fill: C.fc, text: '⟨X⟩' }, svg);
    S('text', { x: 556, y: Y(6) + 26, 'text-anchor': 'end', 'font-size': 10, fill: C.ink, text: 'f = (1 + ⟨X₆⟩)/2' }, svg);

    // column headers (part of each layer's hover group)
    const head = (k, x, name) => {
      S('text', { x, y: 13, 'text-anchor': 'middle', 'font-size': 11, fill: COLOR[k], 'font-weight': 500, text: name }, g[k]);
      S('text', { x, y: 25, 'text-anchor': 'middle', 'font-size': 9, text: P[k] }, g[k]);
    };
    head('C1', (xs.c1a + xs.c1b) / 2, 'C1'); head('C2', xs.c2, 'C2'); head('C3', xs.c3, 'C3'); head('C4', xs.c4, 'C4');
    head('V1', xs.v1, 'V1'); head('V2', xs.v2, 'V2'); head('FC', xs.fc, 'FC');
    S('text', { x: xs.met, y: 13, 'text-anchor': 'middle', 'font-size': 9, fill: C.pool, text: 'measure' }, svg);
    S('text', { x: xs.met, y: 25, 'text-anchor': 'middle', 'font-size': 9, fill: C.pool, text: 'in X' }, svg);
    return svg;
  }

  function wireHover(svg, ro, P) {
    const text = {
      C1: n => `C1 · ${n} parameters. U₁ = U(23)U(24)U(13)U(14)U(12)U(34): six two-qubit unitaries × 15. One U₁ is shared by all four blocks, applied in two staggered slices (k = 0, 2, then k = 1, 3).`,
      C2: n => `C2 · ${n} parameters. One general three-qubit unitary, 2⁶ − 1 = 63, shared by four blocks on (3k+2 … 3k+4).`,
      C3: n => `C3 · ${n} parameters. One three-qubit unitary shared by five blocks on (3k … 3k+2).`,
      C4: n => `C4 · ${n} parameters. One three-qubit unitary shared by four blocks on (3k+1 … 3k+3).`,
      V1: n => `V1 · ${n} parameters. A single-qubit unitary (2² − 1 = 3) on survivor 3k, applied if the X measurement of qubit 3k+1 reads −1.`,
      V2: n => `V2 · ${n} parameters. A single-qubit unitary on survivor 3k+3, controlled by the X measurement of qubit 3k+2.`,
      FC: n => `FC · ${n} parameters. One 32 × 32 unitary on the five survivors (0, 3, 6, 9, 12): 32² − 1 = 1023 = ${(100 * n / P.total).toFixed(0)}% of all parameters.`,
    };
    const dflt = 'Hover a layer: its parameter count appears here. Parameters are shared inside each layer (translation invariance).';
    ro.textContent = dflt;
    const groups = [...svg.querySelectorAll('.lay')];
    const on = g => {
      groups.forEach(o => o.classList.toggle('dim', o !== g));
      const k = g.getAttribute('data-layer'); ro.textContent = text[k](P[k]);
    };
    const off = () => { groups.forEach(o => o.classList.remove('dim')); ro.textContent = dflt; };
    groups.forEach(g => {
      g.addEventListener('mouseenter', () => on(g)); g.addEventListener('mouseleave', off);
      g.addEventListener('focus', () => on(g)); g.addEventListener('blur', off);
    });
  }

  function barChart(div, P) {
    const traces = LAYERS.map(k => ({
      type: 'bar', x: [0], y: [P[k]], name: k, width: 0.5, marker: { color: COLOR[k], line: { color: C.bg, width: 1.5 } },
      hovertemplate: k + ': %{y} parameters<extra></extra>',
    }));
    const mid = {}; let cum = 0; LAYERS.forEach(k => { mid[k] = cum + P[k] / 2; cum += P[k]; });
    const ann = (y, txt, extra) => Object.assign({ xref: 'paper', x: 0.78, y, xanchor: 'left', yanchor: 'middle', showarrow: false, text: txt, font: { size: 10, color: C.ink } }, extra);
    const annotations = [
      ann(mid.C1, 'C1 · ' + P.C1), ann(mid.C2, 'C2 · ' + P.C2), ann(mid.C3, 'C3 · ' + P.C3), ann(mid.C4, 'C4 · ' + P.C4, { yshift: 0 }),
      ann(mid.V1 + 1.5, 'V1, V2 · ' + (P.V1 + P.V2), { yshift: 13 }),
      ann(mid.FC, 'FC · ' + P.FC + '<br>' + (100 * P.FC / P.total).toFixed(0) + '% of all', { font: { size: 11, color: C.fc } }),
    ];
    U.plot(div, traces, {
      height: 470, barmode: 'stack', showlegend: false, margin: { l: 56, r: 8, t: 14, b: 20 },
      xaxis: { visible: false, range: [-0.55, 0.55], fixedrange: true },
      yaxis: { title: { text: 'parameters' }, range: [0, P.total * 1.02], fixedrange: true },
      annotations,
    });
  }

  // CSS for this section lives in styles.css under "sec-learn.js".
})();
