/* Sections: hero, idea (§1), physics (§2). */
(function () {
  'use strict';
  const U = window.QCNNUI;
  if (!U) return;
  const { C, h, svgEl: S } = U;
  const R = String.raw;

  // ======================= hero =======================
  U.register('hero', (el) => {
    const w = h('div', 'hero', null, el);
    h('h1', null, 'Quantum Convolutional Neural Networks', w);
    h('p', 'hero-sub', 'Cong · Choi · Lukin — Nature Physics 15, 1273 (2019) · arXiv:1810.03787', w);

    const cols = h('div', 'hero-cols', null, w);
    h('p', null,
      'We take a quantum circuit that is shaped like a convolutional neural network: translationally invariant convolutions, ' +
      'pooling that shrinks the system, and one fully connected layer at the end. It needs only <b>O(log N)</b> parameters, ' +
      'it is a MERA tensor network run backwards, and it carries the logic of quantum error correction inside it. ' +
      'We use it for two jobs. First, to recognise a symmetry-protected topological phase with fewer samples than ' +
      'string order parameters need (the paper finds the gap growing exponentially with depth; §7 shows how far our N = 81 chain gets). Second, to learn error-correcting codes tailored to the noise at hand.', cols);

    // tapering wire diagram: 27 -> 9 -> 3 -> 1 qubits (draws itself once on load)
    const svg = S('svg', { viewBox: '0 0 420 250', class: 'hero-taper', role: 'img', 'aria-label': 'A circuit whose width shrinks from 27 wires to 9, 3 and 1.' }, cols);
    const y = i => 10 + i * (230 / 26);
    const ends = [110, 230, 330, 410];            // x where each stage's wires stop
    const starts = [0, 110, 230, 330];
    const idx = [Array.from({ length: 27 }, (_, i) => i), [], [], [13]];
    idx[1] = Array.from({ length: 9 }, (_, j) => 9 * 0 + 3 * j + 1);          // centres of stage 1
    idx[2] = [4, 13, 22];
    const surv = [idx[1], idx[2], idx[3]];
    for (let s = 0; s < 4; s++) {
      idx[s].forEach(i => {
        const inNext = s < 3 && surv[s].includes(i);
        const x2 = (s === 3 || inNext) ? ends[s] : ends[s];
        S('line', { x1: starts[s], x2, y1: y(i), y2: y(i), pathLength: 1, class: 'taper-wire', stroke: C.muted, 'stroke-opacity': 0.55, 'stroke-width': 1.1, style: `animation-delay:${0.15 + s * 0.35}s` }, svg);
      });
    }
    // conv blocks, pooling meters, FC box
    const bits = [];
    [[idx[0], 3, 45, 110], [idx[1], 3, 160, 230], [idx[2], 3, 262, 330]].forEach(([wires, k, cx, px], s) => {
      for (let b = 0; b < wires.length; b += k) {
        const a = y(wires[b]) - 3, c = y(wires[b + k - 1]) + 3;
        bits.push(S('rect', { x: cx - 7, y: a, width: 14, height: c - a, rx: 2, fill: 'rgba(47,127,208,.28)', stroke: C.conv, 'stroke-width': 1, class: 'taper-bit', style: `animation-delay:${0.5 + s * 0.35}s` }, svg));
        [0, 2].forEach(o => bits.push(S('circle', { cx: px, cy: y(wires[b + o]), r: 2.6, fill: C.pool, class: 'taper-bit', style: `animation-delay:${0.8 + s * 0.35}s` }, svg)));
      }
    });
    bits.push(S('rect', { x: 345, y: y(13) - 9, width: 22, height: 18, rx: 2, fill: 'rgba(122,63,176,.4)', stroke: C.fc, class: 'taper-bit', style: 'animation-delay:1.6s' }, svg));
    bits.push(S('rect', { x: 394, y: y(13) - 7, width: 14, height: 14, rx: 2, fill: C.bg, stroke: C.fc, class: 'taper-bit', style: 'animation-delay:1.8s' }, svg));

    const stats = h('div', 'stats', null, w);
    [['O(log N)', 'parameters', C.conv], ['1308', 'trained parameters (paper: 1309)', C.fc],
     ['126', 'QEC parameters', C.pool], ['d = 3', 'depth at N = 81 simulated', C.triv]]
      .forEach(([n, l, c]) => { const t = h('div', 'stat', `<b>${n}</b><span>${l}</span>`, stats); t.style.setProperty('--c', c); });

    h('p', 'howto',
      'How to read this page. Sections run in order and each opens with why it matters. Colours follow the paper: ' +
      `<span class="sw" style="background:${C.conv}"></span>convolution, <span class="sw" style="background:${C.pool}"></span>pooling, ` +
      `<span class="sw" style="background:${C.fc}"></span>fully connected, <span class="sw" style="background:${C.err}"></span>error, ` +
      `<span class="sw" style="background:${C.spt}"></span>SPT phase, <span class="sw" style="background:${C.triv}"></span>trivial phase. ` +
      'Sliders, toggles and charts are live; hover a chart for exact values.', w);
  });

  // ======================= idea (§1) =======================
  U.register('idea', (el) => {
    const b = U.head(el, 1, 'From CNN to QCNN', 'Fig. 1 of the paper');
    h('p', null,
      'A convolutional network gets its power from three moves repeated: look locally with the same filter everywhere, ' +
      'shrink the picture, then decide. We build a circuit with exactly those three moves. That is why the number of ' +
      'parameters can stay tiny: a filter shared across the whole system costs the same whether the system has 9 qubits or 900.', b);

    const fig = h('div', 'fig', null, b);
    const ctl = h('div', 'row', null, fig);
    h('span', 'ctl', 'direction of the circuit', ctl);
    const tg = U.toggle(ctl, [{ value: 'qcnn', label: 'QCNN' }, { value: 'mera', label: 'MERA' }], 'qcnn', setMode);

    const OX = 100, WS = 18;
    const svg = S('svg', { viewBox: '0 0 1000 690', role: 'img', 'aria-label': 'Three rows: a CNN, a QCNN and a MERA.' }, fig);
    const defs = S('defs', {}, svg);
    const mk = S('marker', { id: 'ah', viewBox: '0 0 8 8', refX: 7, refY: 4, markerWidth: 7, markerHeight: 7, orient: 'auto-start-reverse' }, defs);
    S('path', { d: 'M0 0 L8 4 L0 8 z', fill: C.muted }, mk);

    const title = (t, y) => S('text', { x: 10, y, 'font-size': 18, 'font-style': 'italic', style: `font-family:Fraunces,Georgia,serif;fill:${C.ink}`, text: t }, svg);
    const lab = (g, x, y, t, o) => S('text', Object.assign({ x, y, 'font-size': 11, 'text-anchor': 'middle', text: t }, o || {}), g);
    const arrow = (g, x1, y1, x2, y2) => S('line', { x1, y1, x2, y2, stroke: C.muted, 'stroke-width': 1.2, 'marker-end': 'url(#ah)' }, g);

    // ---- row 1: CNN ----
    title('CNN', 90);
    const g1 = S('g', {}, svg);
    for (let r = 0; r < 5; r++) for (let c = 0; c < 5; c++)
      S('rect', { x: 130 + c * 15, y: 52 + r * 15, width: 13, height: 13, fill: ((r * 5 + c * 3) % 4 === 0) ? 'rgba(232,228,218,.45)' : 'rgba(232,228,218,.12)' }, g1);
    S('rect', { x: 160, y: 67, width: 45, height: 45, fill: 'none', stroke: C.conv, 'stroke-width': 1.5, 'stroke-dasharray': '3 2' }, g1); // kernel window
    arrow(g1, 220, 90, 275, 90);
    for (let k = 2; k >= 0; k--) S('rect', { x: 290 + k * 9, y: 62 + k * 7 - 12, width: 46, height: 52, fill: 'rgba(47,127,208,.2)', stroke: C.conv }, g1);
    arrow(g1, 360, 90, 420, 90);
    for (let k = 2; k >= 0; k--) S('rect', { x: 435 + k * 9, y: 72 + k * 6 - 10, width: 28, height: 32, fill: 'rgba(240,138,36,.2)', stroke: C.pool }, g1);
    arrow(g1, 495, 90, 560, 90);
    [0, 1, 2, 3, 4].forEach(i => S('circle', { cx: 585, cy: 56 + i * 17, r: 5.5, fill: 'rgba(122,63,176,.35)', stroke: C.fc }, g1));
    [0, 1, 2].forEach(o => [0, 1, 2, 3, 4].forEach(i => S('line', { x1: 590, y1: 56 + i * 17, x2: 700, y2: 64 + o * 26, stroke: C.fc, 'stroke-opacity': 0.25 }, g1)));
    [[0.9, C.spt], [0.3, C.muted], [0.15, C.muted]].forEach(([v, c], o) => S('rect', { x: 700, y: 54 + o * 26, width: 12 + v * 80, height: 14, fill: c, 'fill-opacity': 0.7 }, g1));
    [['input', 170], ['convolution', 328], ['pooling', 470], ['fully connected', 585], ['output', 745]].forEach(([t, x]) => lab(g1, x, 142, t));

    // ---- rows 2 and 3: the same circuit, N = 9 ----
    const layers = { conv1: 215, pool1: 300, conv2: 385, pool2: 470, fc: 555, meter: 640 }; // before OX shift
    const X = k => layers[k] + OX;
    function circuit(parent, y0) {
      const y = i => y0 + i * WS;
      const g = S('g', {}, parent);
      S('rect', { x: 240, y: y0 - 6, width: 520, height: 8 * WS + 12, fill: 'none', stroke: 'none' }, g); // fixes the bbox so the flip pivots on x = 500
      const wire = (i, x2) => S('line', { x1: 240, x2, y1: y(i), y2: y(i), stroke: C.muted, 'stroke-opacity': 0.55, 'stroke-width': 1.2 }, g);
      const cen = [1, 4, 7];
      for (let i = 0; i < 9; i++) wire(i, cen.includes(i) ? (i === 4 ? 760 : X('pool2')) : X('pool1'));
      const conv = (k, a, c) => S('rect', { x: X(k) - 12, y: y(a) - 5, width: 24, height: y(c) - y(a) + 10, rx: 2, fill: 'rgba(47,127,208,.28)', stroke: C.conv, 'stroke-width': 1.3 }, g);
      [0, 3, 6].forEach(a => conv('conv1', a, a + 2));
      conv('conv2', 1, 7);
      const meter = (x, yy, col) => {
        S('rect', { x: x - 6.5, y: yy - 5, width: 13, height: 10, rx: 2, fill: C.bg, stroke: col, 'stroke-width': 1.3 }, g);
        S('path', { d: `M${x - 3.5} ${yy + 2.5} A3.8 3.8 0 0 1 ${x + 3.5} ${yy + 2.5}`, fill: 'none', stroke: col }, g);
        S('line', { x1: x, y1: yy + 2.5, x2: x + 2.5, y2: yy - 2, stroke: col }, g);
      };
      const pool = (k, drop, cE) => drop.forEach(i => {
        S('line', { x1: X(k), x2: X(k), y1: y(i), y2: y(cE(i)), stroke: C.pool, 'stroke-width': 1.1 }, g);
        S('circle', { cx: X(k), cy: y(cE(i)), r: 2.8, fill: C.pool }, g);
        meter(X(k), y(i), C.pool);
      });
      pool('pool1', [0, 2, 3, 5, 6, 8], i => Math.floor(i / 3) * 3 + 1);
      pool('pool2', [1, 7], () => 4);
      S('rect', { x: X('fc') - 14, y: y(4) - 8, width: 28, height: 16, rx: 2, fill: 'rgba(122,63,176,.4)', stroke: C.fc, 'stroke-width': 1.3 }, g);
      meter(X('meter'), y(4), C.fc);
      return g;
    }
    const Y2 = 215, Y3 = 490;
    title('QCNN', Y2 + 4 * WS + 5);
    const t2 = svg.lastChild;
    title('MERA', Y3 + 4 * WS + 5);
    const flipper = circuit(svg, Y2); flipper.id = 'flipper';
    const g3 = circuit(svg, Y3); g3.setAttribute('transform', 'translate(1000 0) scale(-1 1)');

    // direction arrows in time + the "run it backwards" arrow
    [Y2, Y3].forEach(y0 => { arrow(svg, 240, y0 - 18, 760, y0 - 18); S('text', { x: 770, y: y0 - 14, 'font-size': 10, text: 'time' }, svg); });
    const inv = S('g', { id: 'invarrow' }, svg);
    S('line', { x1: 140, y1: 462, x2: 140, y2: 410, stroke: C.pool, 'stroke-width': 1.6, 'marker-end': 'url(#ah)' }, inv);
    const same = S('g', { class: 'fade', opacity: 0 }, svg);
    [Y2, Y3].forEach(y0 => S('rect', { x: 236, y: y0 - 10, width: 528, height: 8 * WS + 20, rx: 4, fill: 'none', stroke: C.conv, 'stroke-dasharray': '4 4' }, same));

    // text labels (re-drawn halfway through each flip)
    const L = S('g', { class: 'fade' }, svg);
    const mir = x => 1000 - x;
    function drawLabels(mode) {
      L.replaceChildren();
      const flipped = mode === 'mera';
      const px = x => flipped ? mir(x) : x;
      const names = flipped
        ? { conv1: 'disentangle', pool1: 'new |0⟩', conv2: 'disentangle', pool2: 'new |0⟩', fc: 'F⁻¹', meter: 'seed' }
        : { conv1: 'convolve U', pool1: 'measure + V', conv2: 'convolve U', pool2: 'measure + V', fc: 'F', meter: 'output' };
      Object.keys(names).forEach(k => lab(L, px(X(k)), Y2 + 8 * WS + 28, names[k], { fill: k === 'fc' || k === 'meter' ? C.fc : k.startsWith('conv') ? C.conv : C.pool }));
      const left = flipped ? '1 qubit' : '9 qubits in', right = flipped ? '9 qubits out' : 'one bit out';
      S('text', { x: 230, y: Y2 + 4 * WS + 4, 'font-size': 11, 'text-anchor': 'end', text: left }, L);
      S('text', { x: 780, y: Y2 + 4 * WS + 4, 'font-size': 11, text: right }, L);
      t2.textContent = flipped ? 'QCNN, mirrored' : 'QCNN';
      // MERA reference row (always mirrored)
      const m = { conv1: 'disentangle', pool1: 'new |0⟩', conv2: 'disentangle', pool2: 'new |0⟩', fc: 'F⁻¹', meter: 'seed' };
      Object.keys(m).forEach(k => lab(L, mir(X(k)), Y3 + 8 * WS + 28, m[k], { fill: k === 'fc' || k === 'meter' ? C.fc : k.startsWith('conv') ? C.conv : C.pool }));
      S('text', { x: 230, y: Y3 + 4 * WS + 4, 'font-size': 11, 'text-anchor': 'end', text: '1 qubit' }, L);
      S('text', { x: 780, y: Y3 + 4 * WS + 4, 'font-size': 11, text: '9 qubits out' }, L);
      S('text', { x: 160, y: 440, 'font-size': 11, fill: C.pool, text: flipped ? 'reverse the gates again: back to the QCNN' : 'reverse every gate: a MERA' }, L);
    }
    drawLabels('qcnn');

    let timer;
    function setMode(m) {
      if (flipper.classList.contains('flipped') === (m === 'mera')) return;
      L.style.opacity = 0;
      flipper.classList.toggle('flipped', m === 'mera');
      inv.classList.toggle('flipped', m === 'mera');
      same.setAttribute('opacity', m === 'mera' ? 0.8 : 0);
      clearTimeout(timer);
      timer = setTimeout(() => { drawLabels(m); L.style.opacity = 1; }, 420);
      readout.innerHTML = m === 'mera' ? readMera : readQcnn;
    }
    const readQcnn = '<b>QCNN.</b> Start with N qubits and end with one. Each pooling step <i>measures</i> two of every three qubits, and those outcomes are error syndromes.';
    const readMera = '<b>MERA.</b> Same gates, opposite direction. Start with one qubit and keep adding fresh ones, entangling each layer. Flipping the picture is all it takes.';
    const readout = h('p', 'caption', readQcnn, fig);
    readout.style.maxWidth = 'none';
    readout.setAttribute('aria-live', 'polite');

    h('p', null,
      '<b>Convolution</b> is a quasi-local unitary $U_i$ acting on a small block of neighbouring qubits, the same $U$ on every block. ' +
      '<b>Pooling</b> measures a subset of qubits and, depending on the outcome, applies a rotation $V_j$ to a neighbour that survives. ' +
      'This is where the nonlinearity comes from: a unitary cannot be nonlinear in the state, but measure-then-correct can. ' +
      'The <b>fully connected layer</b> is one last unitary $F$ on whatever is left, followed by a measurement whose average is the network output $f$.', b);
    U.note(b, 'Pooling is the only nonlinear step. The measured bit decides which correction $V_j$ is applied, and that is a branch, not a rotation.');
    h('p', null, 'Training minimises the mean squared error over a set $S$ of labelled states $|\\psi_i\\rangle$ with labels $y_i$:', b);
    h('div', null, R`$$\mathrm{MSE}=\frac{1}{2|S|}\sum_{i\in S}\bigl(y_i-f_{\{U,V,F\}}(|\psi_i\rangle)\bigr)^2 \tag{1}$$`, b);
    h('p', null,
      'Because the filters are shared, depth grows like $\\log N$ and so does the parameter count. ' +
      'The picture above also hides a second idea. A MERA, the multiscale entanglement renormalisation ansatz, builds a many-body state ' +
      'from a few qubits by layers of disentanglers and isometries. Run it backwards and you get a QCNN. ' +
      'The extra qubits that a MERA would have inserted are exactly the ones the QCNN measures, and in the QEC picture ' +
      'those measurements are syndromes that tell us which error occurred.', b);
  });

  // ======================= physics (§2) =======================
  U.register('physics', (el) => {
    const b = U.head(el, 2, 'The phase we want to recognise', 'Eq. 2 and Fig. 2a');
    h('p', null,
      'Before we can ask a circuit to recognise a phase we need a phase worth recognising. We pick one with no local order parameter at all: ' +
      'a symmetry-protected topological (SPT) phase of a spin chain. Its Hamiltonian has three competing terms:', b);
    const cJ = C.spt, c1 = C.triv, c2 = C.afm;
    h('div', null,
      String.raw`$$H=\textcolor{${cJ}}{-J\sum_i Z_{i-1}X_iZ_{i+1}}\;\textcolor{${c1}}{-\,h_1\sum_i X_i}\;\textcolor{${c2}}{-\,h_2\sum_i X_iX_{i+1}} \tag{2}$$`, b);
    U.note(b, 'Open boundaries give the SPT phase a 4-fold quasi-degenerate ground space (edge spins); our QCNN only measures bulk operators, so it does not care.');
    h('p', null,
      `The <span style="color:${cJ}">cluster term</span> wants the SPT phase, the <span style="color:${c1}">field</span> $h_1$ wants a trivial paramagnet, ` +
      `and the <span style="color:${c2}">coupling</span> $h_2$, when negative, wants an antiferromagnet. ` +
      'All three terms commute with two symmetries, $\\prod_{i\\,\\mathrm{even}}X_i$ and $\\prod_{i\\,\\mathrm{odd}}X_i$, which together form $\\mathbb Z_2\\times\\mathbb Z_2$.', b);

    // ---- the chain ----
    const N = 15, X0 = 60, DX = 60, YS = 110;
    const sx = i => X0 + i * DX;
    const st = { hover: null, pin: null, mode: 'inspect', err: new Set() };
    const fig = h('div', 'fig', null, b);
    const ctl = h('div', 'row', null, fig);
    U.toggle(ctl, [{ value: 'inspect', label: 'inspect stabilisers' }, { value: 'inject', label: 'inject X error' }], 'inspect', v => { st.mode = v; draw(); });
    const clr = h('button', 'pill', 'clear errors', ctl); clr.type = 'button';
    clr.style.border = '1px solid #2a2f37';
    clr.onclick = () => { st.err.clear(); draw(); };

    const svg = S('svg', { viewBox: '0 40 960 170', role: 'group', 'aria-label': 'Chain of 15 spins with their 13 cluster stabilisers' }, fig);
    S('line', { x1: sx(0), x2: sx(N - 1), y1: YS, y2: YS, stroke: C.rule, 'stroke-width': 2 }, svg);
    const brace = S('path', { fill: 'none', 'stroke-width': 1.6 }, svg);
    const sites = [], letters = [], idxT = [], xmark = [];
    for (let i = 0; i < N; i++) {
      const c = S('circle', { cx: sx(i), cy: YS, r: 18, fill: C.bg, 'stroke-width': 1.6, class: 'chain-site', tabindex: 0, role: 'button', 'aria-label': 'site ' + i }, svg);
      c.addEventListener('mouseenter', () => { st.hover = i; draw(); });
      c.addEventListener('mouseleave', () => { st.hover = null; draw(); });
      c.addEventListener('focus', () => { st.hover = i; draw(); });
      c.addEventListener('blur', () => { st.hover = null; draw(); });
      const act = () => {
        if (st.mode === 'inject') { st.err.has(i) ? st.err.delete(i) : st.err.add(i); st.pin = null; }
        else st.pin = st.pin === i ? null : i;
        draw();
      };
      c.addEventListener('click', act);
      c.addEventListener('keydown', e => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); act(); } });
      sites.push(c);
      letters.push(S('text', { x: sx(i), y: YS - 34, 'text-anchor': 'middle', 'font-size': 17, 'font-weight': 500 }, svg));
      xmark.push(S('text', { x: sx(i), y: YS + 5, 'text-anchor': 'middle', 'font-size': 14, 'pointer-events': 'none' }, svg));
      idxT.push(S('text', { x: sx(i), y: YS + 40, 'text-anchor': 'middle', 'font-size': 11, text: String(i) }, svg));
    }
    // stabiliser chips K_1..K_13
    const chips = [];
    for (let i = 1; i < N - 1; i++) {
      const g = S('g', { style: 'cursor:pointer' }, svg);
      const r = S('rect', { x: sx(i) - 20, y: 176, width: 40, height: 26, rx: 3, 'stroke-width': 1.3 }, g);
      S('text', { x: sx(i), y: 193, 'text-anchor': 'middle', 'font-size': 12, 'pointer-events': 'none', text: 'K' + i }, g);
      g.addEventListener('mouseenter', () => { st.hover = i; draw(); });
      g.addEventListener('mouseleave', () => { st.hover = null; draw(); });
      chips[i] = r;
    }

    h('p', 'caption', 'K_i = Z_{i-1} X_i Z_{i+1}. A chip filled red means that stabiliser is violated (reads −1).', fig);
    const status = h('p', 'caption', '', fig);
    status.setAttribute('aria-live', 'polite'); status.style.maxWidth = 'none'; status.style.minHeight = '3em';

    const viol = i => ((st.err.has(i - 1) ? 1 : 0) + (st.err.has(i + 1) ? 1 : 0)) % 2 === 1; // X_j anticommutes with Z_j in neighbouring K
    function draw() {
      const A = st.hover != null ? st.hover : st.pin;
      const act = A != null && A >= 1 && A <= N - 2;
      const bad = [];
      for (let i = 1; i < N - 1; i++) if (viol(i)) bad.push(i);
      const col = act ? (viol(A) ? C.err : C.conv) : C.conv;
      for (let i = 0; i < N; i++) {
        const inK = act && Math.abs(i - A) <= 1, e = st.err.has(i);
        sites[i].setAttribute('stroke', e ? C.err : inK ? col : (A === i ? C.muted : '#5a626d'));
        sites[i].setAttribute('fill', e ? 'rgba(209,52,44,.22)' : inK ? (viol(A) ? 'rgba(209,52,44,.12)' : 'rgba(47,127,208,.16)') : C.bg);
        letters[i].textContent = inK ? (i === A ? 'X' : 'Z') : '';
        letters[i].setAttribute('fill', col);
        xmark[i].textContent = e ? 'X' : '';
        xmark[i].setAttribute('fill', C.err);
        idxT[i].setAttribute('fill', A === i ? C.ink : C.muted);
      }
      if (act) {
        brace.setAttribute('d', `M${sx(A - 1)} ${YS - 56} V${YS - 50} H${sx(A + 1)} V${YS - 56}`);
        brace.setAttribute('stroke', col);
      } else brace.setAttribute('d', '');
      for (let i = 1; i < N - 1; i++) {
        const v = viol(i), on = act && A === i;
        chips[i].setAttribute('fill', v ? 'rgba(209,52,44,.5)' : on ? 'rgba(47,127,208,.28)' : 'rgba(47,127,208,.08)');
        chips[i].setAttribute('stroke', v ? C.err : C.conv);
        chips[i].setAttribute('stroke-width', on ? 2.4 : 1.3);
      }
      // sentence under the figure
      const errs = [...st.err].sort((a, b) => a - b);
      let s;
      if (!errs.length) {
        s = 'No errors: every stabiliser reads +1, because the cluster state is their common +1 eigenstate. Hover a site to see the three spins its stabiliser touches' +
          (st.mode === 'inject' ? ', and click sites to flip them.' : '; switch to "inject X error" and click.');
      } else {
        s = 'X error on site' + (errs.length > 1 ? 's ' : ' ') + errs.join(', ') + '. Violated: ' + (bad.length ? bad.map(i => 'K' + i).join(', ') : 'none') + '. ';
        if (errs.length === 1) {
          const e = errs[0];
          s += e === 0 || e === N - 1
            ? 'At the edge there is only one neighbouring stabiliser, so one chip turns red.'
            : `X${e} anticommutes with the Z in K${e - 1} and K${e + 1}, so exactly those two flip. It commutes with K${e}, which has X${e} in the middle.`;
        } else s += errs.some((e, k) => k && e - errs[k - 1] === 2)
          ? 'Two errors two sites apart share a neighbour, and the shared stabiliser flips twice, so it reads +1 again.'
          : 'Each error flips its own two neighbouring stabilisers.';
      }
      if (A != null && !act) s += ` Site ${A} is a boundary spin: it has no stabiliser of its own.`;
      status.textContent = s;
    }
    draw();

    h('p', null,
      'When $h_2=0$ the model maps by a Jordan–Wigner transformation to free fermions, so it is exactly solvable, and the SPT-to-paramagnet transition sits at $h_1=J$. ' +
      'The three phases are the SPT (small $h_1$, $|h_2|$), the paramagnet (large $h_1$), and the antiferromagnet (large negative $h_2$), which breaks the symmetry spontaneously. ' +
      'The SPT phase has no local order parameter, but it does have a nonlocal one: the string $Z_aX_{a+1}X_{a+3}\\cdots X_{b-1}Z_b$, a product of stabilisers. ' +
      'It is non-zero only in this phase, and the QCNN will turn out to measure a clever weighted sum of such strings.', b);
  });
})();
