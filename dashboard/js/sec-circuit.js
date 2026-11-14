/* Sections: circuit (§3) and multiscale (§4). */
(function () {
  'use strict';
  const U = window.QCNNUI;
  if (!U) return;
  const { C, h, svgEl: S } = U;
  const R = String.raw;
  const MINUS = '−';
  const EPS = 1e-6;

  // diverging scale for expectation values: -1 error red, 0 neutral grey, +1 convolution blue
  const NEUTRAL = [93, 101, 112];
  const hex2rgb = s => [1, 3, 5].map(i => parseInt(s.slice(i, i + 2), 16));
  function divColor(v) {
    const t = Math.max(-1, Math.min(1, v));
    const end = hex2rgb(t < 0 ? C.err : C.conv), a = Math.abs(t);
    return 'rgb(' + NEUTRAL.map((n, i) => Math.round(n + (end[i] - n) * a)).join(',') + ')';
  }
  const sgn = v => (Math.abs(v) < 0.005 ? '0' : (v < 0 ? MINUS : '+') + (Math.abs(v) > 0.995 ? '1' : Math.abs(v).toFixed(1).replace(/^0/, '')));
  const num = (v, d) => (Math.abs(v) < 0.5 * Math.pow(10, -(d || 2)) ? 0 : v).toFixed(d || 2).replace('-', MINUS);
  const wrap = (s, n) => {
    const out = []; let cur = '';
    s.split(' ').forEach(w => { if ((cur + ' ' + w).trim().length > n && cur) { out.push(cur); cur = w; } else cur = (cur + ' ' + w).trim(); });
    out.push(cur); return out;
  };

  // ======================= circuit (§3) =======================
  U.register('circuit', (el, D) => {
    const body = U.head(el, 3, 'An exact QCNN for the cluster-state phase', 'Fig. 2b · Fig. 3');
    if (!U.needData(body, 'circuit_demo', 'circuit_demo')) return;
    const cd = D.circuit_demo;
    U.quickBadge(body, cd);

    const NQ = cd.N, nL = cd.layers.length;
    const live = new Set(cd.live), fine = cd.fine;
    const thetas = cd.thetas, thMax = thetas[thetas.length - 1], thStep = thetas.length > 1 ? thetas[1] - thetas[0] : 1;
    const sites = cd.runs.map(r => r.site).filter(s => s != null);

    h('p', null,
      R`The cleanest way to see what a QCNN does is to build one we can check by hand. Take $N=9$ qubits in the cluster state, the SPT ground state at $h_1=h_2=0$. Its stabilisers $K_i = Z_{i-1}X_iZ_{i+1}$ all equal $+1$. Four layers shrink nine qubits to the three block centres $1,4,7$, and a last layer asks one question of them: is $\langle Z_1X_4Z_7\rangle = +1$? That number is the network's output.`, body);
    U.note(body, R`<b>Edge sites 0 and 8 are boundary modes,</b> not errors the bulk code can see. Try them: nothing fires, and nothing breaks either.`);
    h('p', null,
      R`Below is the whole circuit. Drag the stage slider to follow the state layer by layer: each bead is $\langle X_i\rangle$ on that wire. Then inject an error of strength $\theta$ on any site and watch which fine qubits flip.`, body);

    // ---- controls ----
    const fig = h('div', 'fig circ', null, body);
    const r1 = h('div', 'row', null, fig);
    h('span', 'ctl-lab', 'inject X error on site', r1);
    const st = { site: 3, theta: thMax, stage: 1 };
    const sitePills = U.toggle(r1, [{ value: -1, label: 'none' }].concat(sites.map(s => ({ value: s, label: String(s) }))), st.site, v => { st.site = v; render(); });
    const r2 = h('div', 'row', null, fig);
    const thSl = U.slider(r2, { label: R`rotation θ, error = exp(−iθX)`, min: 0, max: thMax, step: thStep, value: st.theta, fmt: v => (v / Math.PI).toFixed(2) + 'π', onInput: v => { st.theta = v; render(); } });
    const stageFmt = s => (s === 0 ? 'input state' : 'after layer ' + s);
    const stSl = U.slider(r2, { label: 'stage', min: 0, max: nL, step: 1, value: st.stage, fmt: stageFmt, onInput: v => { st.stage = v; render(); } });
    const play = h('button', 'circ-play', 'step through', r2);
    play.type = 'button';
    let timer = null;
    function stop() { if (timer) { clearInterval(timer); timer = null; play.textContent = 'step through'; } }
    play.addEventListener('click', () => {
      if (timer) return stop();
      if (st.stage >= nL) { st.stage = 0; stSl.set(0); render(); }
      play.textContent = 'stop';
      timer = setInterval(() => { st.stage += 1; stSl.set(st.stage); render(); if (st.stage >= nL) stop(); }, 1100);
    });
    stSl.addEventListener('input', stop);

    // ---- circuit geometry ----
    const GAP = 36, TOP = 78, SUB = 60, PADL = 40, X0 = 64;
    const wy = i => TOP + i * GAP;
    const H = wy(NQ - 1) + 52;
    // assign sub-columns so gates in one layer never overlap vertically
    const layers = cd.layers.map(L => {
      const cols = [];
      const gates = L.gates.map(g => {
        const lo = Math.min(...g.q), hi = Math.max(...g.q);
        let s = cols.findIndex(c => c.every(([a, b]) => hi < a || lo > b));
        if (s < 0) { s = cols.length; cols.push([]); }
        cols[s].push([lo, hi]);
        return Object.assign({}, g, { sub: s });
      });
      return Object.assign({}, L, { gates, nsub: Math.max(cols.length, 1) });
    });
    let cur = X0 + 34;
    const nowX = [X0 + 16];
    layers.forEach(L => {
      L.x0 = cur; L.band = 2 * PADL + (L.nsub - 1) * SUB;
      L.gx = s => L.x0 + PADL + s * SUB;
      cur += L.band; nowX.push(cur);
    });
    const fcX = cur + 46, W = fcX + 190;

    const svg = S('svg', { viewBox: `0 0 ${W} ${H}`, role: 'img', 'aria-label': 'Circuit for N=9: convolution and pooling layers followed by a fully connected measurement.' }, fig);

    // wires end at their measurement (fine qubits) or run to the FC layer (centres)
    const poolL = layers.findLast ? layers.findLast(L => L.kind === 'pool') : layers.filter(L => L.kind === 'pool').pop();
    const meterAt = {};
    if (poolL) poolL.gates.forEach(g => { if (g.g === 'cx_z') meterAt[g.q[0]] = poolL.gx(g.sub); });
    fine.forEach(f => { if (meterAt[f] == null && poolL) meterAt[f] = poolL.gx(0); });
    const gBg = S('g', {}, svg);
    layers.forEach((L, li) => {
      const c = L.kind === 'pool' ? C.pool : C.conv;
      S('rect', { x: L.x0, y: 8, width: L.band, height: 3, fill: c }, gBg);
      wrap(L.label, 16).forEach((ln, k) => S('text', { x: L.x0 + L.band / 2, y: 28 + k * 14, 'text-anchor': 'middle', 'font-size': 11.5, fill: c, text: ln }, gBg));
      if (li > 0) S('line', { x1: L.x0, x2: L.x0, y1: 40, y2: H - 30, stroke: C.rule, 'stroke-dasharray': '2 5' }, gBg);
    });
    S('rect', { x: fcX - 54, y: 8, width: 150, height: 3, fill: C.fc }, gBg);
    S('text', { x: fcX + 21, y: 28, 'text-anchor': 'middle', 'font-size': 11.5, fill: C.fc, text: 'fully connected' }, gBg);
    S('text', { x: fcX + 21, y: 42, 'text-anchor': 'middle', 'font-size': 11.5, fill: C.fc, text: 'measure Z X Z' }, gBg);
    S('line', { x1: fcX - 54, x2: fcX - 54, y1: 40, y2: H - 30, stroke: C.rule, 'stroke-dasharray': '2 5' }, gBg);

    for (let i = 0; i < NQ; i++) {
      const isLive = live.has(i);
      const x2 = isLive ? fcX + 76 : (meterAt[i] != null ? meterAt[i] : W - 20);
      S('line', { x1: X0, x2, y1: wy(i), y2: wy(i), stroke: C.ink, 'stroke-opacity': isLive ? 0.55 : 0.28, 'stroke-width': isLive ? 2 : 1.2 }, svg);
      S('text', { x: X0 - 12, y: wy(i) + 4, 'text-anchor': 'end', 'font-size': 12, 'font-weight': isLive ? 700 : 400, fill: isLive ? C.ink : C.muted, text: 'q' + i }, svg);
    }

    // gates
    const meter = (parent, x, y, ttl, dashed) => {
      const g = S('g', {}, parent);
      S('rect', { x: x - 14, y: y - 12, width: 28, height: 24, rx: 3, fill: C.bg, stroke: C.pool, 'stroke-width': 1.6, 'stroke-dasharray': dashed ? '3 3' : '' }, g);
      S('path', { d: `M${x - 8} ${y + 7} A8 8 0 0 1 ${x + 8} ${y + 7}`, fill: 'none', stroke: C.pool, 'stroke-width': 1.2 }, g);
      S('line', { x1: x, y1: y + 7, x2: x + 5, y2: y - 1, stroke: C.pool, 'stroke-width': 1.2 }, g);
      S('text', { x: x - 6, y: y - 3, 'text-anchor': 'middle', 'font-size': 9, 'font-weight': 700, fill: C.pool, text: 'X' }, g);
      if (ttl) S('title', { text: ttl }, g);
    };
    const box = (parent, x, y, letter, color, solid) => {
      S('rect', { x: x - 10, y: y - 10, width: 20, height: 20, rx: 3, fill: solid ? color : C.bg, stroke: color, 'stroke-width': 1.6 }, parent);
      S('text', { x, y: y + 4.5, 'text-anchor': 'middle', 'font-size': 12.5, 'font-weight': 700, fill: solid ? C.bg : color, text: letter }, parent);
    };
    const layerG = layers.map(L => {
      const g = S('g', { class: 'fade' }, svg);
      L.gates.forEach(gt => {
        const x = L.gx(gt.sub), ys = gt.q.map(wy);
        const gg = S('g', {}, g);
        if (gt.g === 'cz') {
          S('line', { x1: x, x2: x, y1: Math.min(...ys), y2: Math.max(...ys), stroke: C.conv, 'stroke-width': 2 }, gg);
          ys.forEach(y => S('circle', { cx: x, cy: y, r: 5, fill: C.conv }, gg));
          S('title', { text: `CZ on q${gt.q.join(', q')}` }, gg);
        } else if (gt.g === 'ccz_x') {
          S('line', { x1: x, x2: x, y1: Math.min(...ys), y2: Math.max(...ys), stroke: C.conv, 'stroke-width': 2 }, gg);
          gt.q.forEach((q, k) => box(gg, x, wy(q), 'X', C.conv, k === 1));
          S('title', { text: `Toffoli on q${gt.q.join(', q')}: X-basis controls on the outer two qubits, target the centre` }, gg);
        } else if (gt.g === 'cx_z') {
          S('line', { x1: x, x2: x, y1: ys[0], y2: ys[1], stroke: C.pool, 'stroke-width': 1.6, 'stroke-dasharray': '4 3' }, gg);
          meter(gg, x, ys[0], `Measure q${gt.q[0]} in the X basis`);
          box(gg, x, ys[1], 'Z', C.pool, true);
          S('title', { text: `If q${gt.q[0]} reads −1, apply Z to q${gt.q[1]}` }, gg);
        }
      });
      return g;
    });
    // boundary fine qubits that the pooling layer never touches are only read out
    const readOnly = fine.filter(f => poolL && !poolL.gates.some(g => g.q[0] === f));
    readOnly.forEach(f => meter(layerG[layers.indexOf(poolL)], poolL.gx(0), wy(f), `q${f}: no gate touches this boundary qubit, it is only read out in the X basis`, true));

    // fully connected layer
    const fcG = S('g', { class: 'fade' }, svg);
    S('line', { x1: fcX, x2: fcX, y1: wy(cd.fc[0]), y2: wy(cd.fc[cd.fc.length - 1]), stroke: C.fc, 'stroke-width': 2 }, fcG);
    cd.fc.forEach((q, k) => box(fcG, fcX, wy(q), k === 1 ? 'X' : 'Z', C.fc, true));
    S('line', { x1: fcX + 10, x2: fcX + 52, y1: wy(cd.fc[1]), y2: wy(cd.fc[1]), stroke: C.fc, 'stroke-width': 1.6, 'stroke-dasharray': '4 3' }, fcG);
    S('rect', { x: fcX + 52, y: wy(cd.fc[1]) - 12, width: 28, height: 24, rx: 3, fill: C.bg, stroke: C.fc, 'stroke-width': 1.6 }, fcG);
    S('path', { d: `M${fcX + 58} ${wy(cd.fc[1]) + 7} A8 8 0 0 1 ${fcX + 74} ${wy(cd.fc[1]) + 7}`, fill: 'none', stroke: C.fc, 'stroke-width': 1.2 }, fcG);
    S('line', { x1: fcX + 66, y1: wy(cd.fc[1]) + 7, x2: fcX + 72, y2: wy(cd.fc[1]) - 1, stroke: C.fc, 'stroke-width': 1.2 }, fcG);

    // syndrome values at the end of the measured wires, and output value at the FC meter
    const synTxt = {};
    fine.forEach(f => { synTxt[f] = S('text', { x: (meterAt[f] || 0), y: wy(f) - 17, 'text-anchor': 'middle', 'font-size': 10.5, 'font-weight': 700, text: '' }, svg); });
    const outTxt = S('text', { x: fcX + 92, y: wy(cd.fc[1]) + 4, 'font-size': 11.5, 'font-weight': 700, fill: C.fc, text: '' }, svg);

    // "now" marker: a moving line with one bead per wire
    const now = S('g', { class: 'now' }, svg);
    S('line', { x1: 0, x2: 0, y1: 44, y2: H - 30, stroke: C.ink, 'stroke-opacity': 0.5, 'stroke-width': 1.2 }, now);
    const nowLab = S('text', { x: 0, y: H - 12, 'text-anchor': 'middle', 'font-size': 11.5, fill: C.ink, text: '' }, now);
    const beads = [], beadTxt = [];
    for (let i = 0; i < NQ; i++) {
      const b = S('g', {}, now);
      beads.push(S('circle', { cx: 0, cy: wy(i), r: 11, stroke: C.bg, 'stroke-width': 2, class: 'bead' }, b));
      beadTxt.push(S('text', { x: 0, y: wy(i) + 3.5, 'text-anchor': 'middle', 'font-size': 9.5, 'font-weight': 700, fill: '#fff', text: '' }, b));
    }

    // legend for the diverging colour scale
    const lg = h('div', 'circ-legend', null, fig);
    h('span', null, R`bead = ⟨X<sub>i</sub>⟩`, lg);
    h('span', 'bar', null, lg);
    h('span', null, `${MINUS}1 &nbsp; 0 &nbsp; +1`, lg);
    lg.querySelector('.bar').style.background = `linear-gradient(90deg, ${divColor(-1)}, ${divColor(0)}, ${divColor(1)})`;

    // stabiliser chips
    h('div', 'chip-lab', R`Stabilisers ⟨K<sub>i</sub>⟩ = ⟨Z<sub>i−1</sub> X<sub>i</sub> Z<sub>i+1</sub>⟩ on the state at this stage`, fig);
    const chipRow = h('div', 'chips', null, fig);
    const chips = [];
    for (let i = 1; i <= NQ - 2; i++) chips.push(h('div', 'chip', `<span>K<sub>${i}</sub></span><b></b>`, chipRow));

    // readout card
    const card = h('div', 'readout', null, fig);
    card.setAttribute('aria-live', 'polite');
    const synBox = h('div', 'syn-box', null, card);
    h('div', 'ro-lab', 'Syndromes: ⟨X⟩ on each discarded qubit', synBox);
    const synRow = h('div', 'syn-row', null, synBox);
    const synCells = fine.map(f => h('div', 'syn', `<span>q${f}</span><b></b><small></small>`, synRow));
    const outBox = h('div', 'out-box', null, card);
    h('div', 'ro-lab', R`Output ⟨Z<sub>1</sub>X<sub>4</sub>Z<sub>7</sub>⟩`, outBox);
    const outVal = h('div', 'out-val', '', outBox);
    const verdict = h('div', 'verdict', '', outBox);

    // sweep: every site at the current theta, read straight from the data
    h('div', 'chip-lab', 'Every single-site error at the chosen rotation angle (click to select)', fig);
    const sweep = h('div', 'sweep', null, fig);
    const sweepCells = sites.map(s => {
      const b = h('button', 'sw', `<span>q${s}</span><b></b><small></small>`, sweep);
      b.type = 'button';
      b.addEventListener('click', () => { st.site = s; sitePills.set(s); render(); });
      return b;
    });

    h('div', 'caption',
      R`Fig. 2b / Fig. 3 — reproduced: the exact QCNN for N=9 acting on the cluster state with an X-rotation error on one site. Blue pins are convolution gates, orange the pooling measurements with their Z feed-forward, purple the fully connected measurement. Every number is a statevector simulation.`, fig);

    const runFor = site => cd.runs.find(r => r.site === (site < 0 ? null : site)) || cd.runs[0];
    const tIdx = (run, theta) => Math.max(0, Math.min(run.theta.length - 1, Math.round(theta / thStep)));

    function render() {
      const run = runFor(st.site), ti = tIdx(run, st.theta);
      const noErr = st.site < 0;
      thSl.disabled = noErr; thSl.parentNode.style.opacity = noErr ? 0.4 : 1;
      const x = run.x[ti][st.stage], k = run.k[ti][st.stage];
      now.style.transform = `translateX(${nowX[st.stage]}px)`;
      nowLab.textContent = stageFmt(st.stage);
      x.forEach((v, i) => {
        beads[i].style.fill = divColor(v); beadTxt[i].textContent = sgn(v);
        beads[i].firstChild && beads[i].removeChild(beads[i].firstChild);
        S('title', { text: `q${i}: ⟨X⟩ = ${num(v, 3)}` }, beads[i]);
      });
      layerG.forEach((g, li) => { g.style.opacity = li < st.stage ? 1 : 0.38; });
      fcG.style.opacity = st.stage >= nL ? 1 : 0.38;
      chips.forEach((c, m) => {
        const v = k[m]; const b = c.querySelector('b');
        b.textContent = num(v); c.style.background = divColor(v).replace('rgb', 'rgba').replace(')', ',0.28)'); c.style.borderColor = divColor(v);
      });

      const syn = run.syndromes[ti], out = run.output[ti];
      const fired = syn.map(v => v < -EPS);
      fine.forEach((f, j) => {
        const v = syn[j], c = synCells[j];
        c.querySelector('b').textContent = num(v);
        c.querySelector('small').textContent = 'P(' + MINUS + '1) = ' + Math.round((1 - v) / 2 * 100) + '%';
        c.classList.toggle('fire', fired[j]);
        synTxt[f].textContent = num(v, fired[j] ? 0 : 1);
        synTxt[f].setAttribute('fill', fired[j] ? C.pool : C.muted);
      });
      outVal.textContent = out.toFixed(3);
      outTxt.textContent = '= ' + out.toFixed(3);
      const anyFire = fired.some(Boolean), good = Math.abs(out - 1) < EPS;
      const partial = syn.some(v => v < 1 - EPS);
      let msg, cls;
      if (!good) { msg = 'output corrupted by the error'; cls = 'bad'; }
      else if (anyFire) { msg = 'corrected ✓ &nbsp;a syndrome fired and the output is still 1'; cls = 'ok'; }
      else if (noErr || st.theta === 0) { msg = 'no error injected, output 1'; cls = ''; }
      else if (partial) { msg = 'syndromes only partly flipped (outcome −1 is probabilistic), output still 1'; cls = 'ok'; }
      else { msg = 'no syndrome fires and the output is unaffected: the bulk code never sees this error'; cls = ''; }
      verdict.className = 'verdict ' + cls; verdict.innerHTML = msg;

      sites.forEach((s, n) => {
        const r = cd.runs.find(q => q.site === s), i2 = tIdx(r, st.theta);
        const nf = r.syndromes[i2].filter(v => v < -EPS).length, o = r.output[i2];
        const cell = sweepCells[n];
        cell.querySelector('b').textContent = nf ? nf + ' fired' : 'silent';
        cell.querySelector('small').textContent = 'out ' + o.toFixed(3);
        cell.classList.toggle('hit', nf > 0); cell.setAttribute('aria-pressed', String(s === st.site));
      });
    }
    render();

    h('p', null,
      R`<b>Why these gates.</b> Two conditions fix the circuit. The <em>fixed-point</em> condition says that each convolution-plus-pooling unit must send the ideal state to the same state on a lattice three times coarser: the cluster state on qubits $1,4,7$, with every discarded qubit left in $|+\rangle$. You can see it in the beads: after layer 1 every $\langle X_i\rangle=+1$ (so the stabilisers $K_i$, which were $+1$ on the input, now average to zero: the layer has disentangled the cluster state into $|+\rangle^{\otimes 9}$), and after the pooling layer the discarded wires still read $+1$. The <em>QEC</em> condition says that the circuit also maps a symmetry-respecting error on the input to a flipped discarded qubit. So the qubits we throw away are exactly the syndromes of an error-correcting code.`, body);
    h('p', null,
      R`<b>Pooling does the correcting.</b> Each discarded qubit is measured in the $X$ basis. When the outcome is $-1$, the pooling gate applies $Z$ to the neighbouring block centre, which flips that centre's phase back. In the figure the controls are the dashed lines. That feed-forward is the QCNN's nonlinearity, and it is why a single $X$ error on any of sites $1$ to $7$ still leaves the output at $+1$.`, body);
    h('p', null,
      R`<b>The fully connected layer</b> is the last stabiliser of the coarse cluster state: it measures $Z_1X_4Z_7$ on the three survivors. Its value is the network's answer, $+1$ in the SPT phase. Try it: pick site 3 and push $\theta$ to $\pi/2$. One syndrome turns orange, the output stays $1.000$. Intermediate $\theta$ is a superposition of error and no error; the syndrome then fires only with some probability, and the output is still $1$ either way.`, body);
  });

  // ======================= multiscale (§4) =======================
  // Pauli tile (X blue, Z orange, Y purple, I outline), letter always shown so colour is not the only cue
  const pt = p => `<span class="pt ${p}">${p === 'I' ? '' : p}</span>`;
  const OFS = n => (n === 0 ? '0' : (n < 0 ? MINUS : '+') + Math.abs(n));
  function frac(c) {
    const a = Math.abs(c), s = c < 0 ? MINUS : '+';
    const f = { 1: '1', 0.5: '½', 0.25: '¼', 0.125: '⅛' }[a];
    if (f) return s + f;
    for (const d of [16, 32, 64]) if (Math.abs(a * d - Math.round(a * d)) < 1e-9) return s + Math.round(a * d) + '/' + d;
    return s + a.toFixed(3);
  }
  // one row of tiles from offsets lo..hi
  function tileRow(parent, term, lo, hi, extra) {
    const row = h('div', 'prow', null, parent);
    h('span', 'pc', frac(term.c), row);
    const t = h('div', 'ptiles', null, row);
    t.style.gridTemplateColumns = `repeat(${hi - lo + 1}, var(--t))`;
    const at = {}; term.ops.forEach(([p, o]) => { at[o] = p; });
    let html = '';
    for (let o = lo; o <= hi; o++) html += pt(at[o] || 'I');
    t.innerHTML = html;
    h('span', 'pw', 'w=' + term.ops.length, row);
    if (extra) row.appendChild(extra);
    return row;
  }
  function offsetHeader(parent, lo, hi) {
    const row = h('div', 'prow phead', null, parent);
    h('span', 'pc', '', row);
    const t = h('div', 'ptiles', null, row);
    t.style.gridTemplateColumns = `repeat(${hi - lo + 1}, var(--t))`;
    for (let o = lo; o <= hi; o++) h('span', 'po' + (o === 0 ? ' zero' : ''), OFS(o), t);
    h('span', 'pw', '', row);
  }
  // parse the paper's LaTeX like "\tfrac12\left(Z_{\tilde i} - Z_{\tilde i-2}X_{\tilde i-1} ...\right)" into signed terms
  function parsePaper(src) {
    const half = /\\tfrac\{?1\}?\{?2\}?/.test(src);
    let rhs = src.split('=').slice(1).join('=').replace(/\\tfrac\{?1\}?\{?2\}?/, '').replace(/\\left\(|\\right\)/g, ' ');
    rhs = ' + ' + rhs.trim();
    const out = [];
    rhs.split(/\s([+-])\s/).slice(1).reduce((acc, s, i, a) => { if (i % 2 === 0) out.push({ c: (s === '-' ? -1 : 1) * (half ? 0.5 : 1), body: a[i + 1] }); return acc; }, 0);
    return out.map(t => {
      const ops = []; const re = /([XYZ])_\{\\tilde i(?:([+-]\d+))?\}/g; let m;
      while ((m = re.exec(t.body))) ops.push([m[1], m[2] ? parseInt(m[2].replace('+', ''), 10) : 0]);
      return { c: t.c, ops };
    });
  }
  const sig = ops => ops.slice().sort((a, b) => a[1] - b[1]).map(o => o[0] + o[1]).join(',');

  U.register('multiscale', (el, D) => {
    const body = U.head(el, 4, 'What the circuit really measures', 'Eqs. 8–10 · Fig. 8');
    if (!U.needData(body, 'heisenberg', 'heisenberg')) return;
    const hz = D.heisenberg;
    U.quickBadge(body, hz);

    h('p', null,
      R`A circuit is only a rewriting of the operator we finally measure. Run it backwards in the Heisenberg picture: $U^\dagger O\, U$ tells us what the output $O$ means on the input chain. For one convolution-plus-pooling unit the rules are short, and the measured $Z_1X_4Z_7$ becomes a sum of string order parameters of growing length.`, body);
    U.note(body, R`<b>Signs depend on convention.</b> Ours come from the explicit gates in §3, checked numerically against the circuit. They flip under $Z\to -Z$ on the measured sites (equivalently, relabelling the X-basis outcomes), so the physics is the same.`);

    // --- rules, paper next to ours ---
    const paperTerms = { X: parsePaper(hz.paper.eq8), Z: parsePaper(hz.paper.eq9) };
    const rulesFig = h('div', 'fig ms', null, body);
    h('div', 'chip-lab', R`One unit, Heisenberg picture: a Pauli on centre c̃ becomes a sum of strings on the finer chain (offsets from c̃)`, rulesFig);
    const diffs = [];
    ['X', 'Z', 'Y'].forEach(P => {
      const terms = hz.rules[P]; if (!terms) return;
      const all = terms.flatMap(t => t.ops.map(o => o[1]));
      const lo = Math.min(-2, ...all), hi = Math.max(2, ...all);
      const grp = h('div', 'rule', null, rulesFig);
      const left = h('div', 'rule-l', null, grp);
      if (hz.paper['eq' + (P === 'X' ? 8 : 9)] && P !== 'Y') {
        h('div', 'ro-lab', 'paper', left);
        h('div', 'eqn', '$$' + hz.paper['eq' + (P === 'X' ? 8 : 9)] + '$$', left);
      } else {
        h('div', 'ro-lab', 'paper', left);
        h('div', 'eqn-none', R`not printed; follows from $Y = iXZ$`, left);
      }
      const right = h('div', 'rule-r', null, grp);
      h('div', 'ro-lab', R`ours: ${P}<sub>c̃</sub> →`, right);
      offsetHeader(right, lo, hi);
      const pap = paperTerms[P];
      terms.forEach(t => {
        let extra = null;
        if (pap) {
          const m = pap.find(q => sig(q.ops) === sig(t.ops));
          if (m && Math.sign(m.c) !== Math.sign(t.c)) {
            extra = h('span', 'diff', 'paper prints ' + frac(m.c), null);
            diffs.push({ P, ours: t.c, paper: m.c });
          }
        }
        tileRow(right, t, lo, hi, extra);
      });
    });
    h('div', 'caption',
      R`Eqs. 8–9 — reproduced: our rules next to the paper's. Coefficients in mono, w = number of non-identity tiles. ` +
      (diffs.length
        ? `Our derivation agrees with the paper except for the sign of ${diffs.length} Z-rule term${diffs.length > 1 ? 's' : ''} (flagged): we find ${diffs.map(d => frac(d.ours)).join(' and ')} where it prints ${diffs.map(d => frac(d.paper)).join(' and ')}. The convention note in the margin explains why this is harmless.`
        : 'Every sign agrees with the paper.'), rulesFig);

    // --- multiscale SOP, d = 1 ---
    h('p', null,
      R`Apply the rules again and again. Pulling $Z_1X_4Z_7$ back through $d$ units gives Eq. 10, the <b>multiscale string order parameter</b>. At $d=1$ it already has ${hz.sop_d1.length} terms. Each is a product of short strings, so each term is a string order parameter in its own right, and together they form one operator.`, body);
    const sopFig = h('div', 'fig ms', null, body);
    h('div', 'chip-lab', `The ${hz.sop_d1.length} terms of the d = 1 multiscale order parameter (offsets from the fine child of the measured site)`, sopFig);
    const all = hz.sop_d1.flatMap(t => t.ops.map(o => o[1]));
    const lo = Math.min(...all), hi = Math.max(...all);
    const sop = h('div', 'sop', null, sopFig);
    const half = Math.ceil(hz.sop_d1.length / 2);
    [hz.sop_d1.slice(0, half), hz.sop_d1.slice(half)].forEach(part => {
      const col = h('div', 'sop-col', null, sop);
      offsetHeader(col, lo, hi);
      part.forEach(t => tileRow(col, t, lo, hi));
    });
    h('div', 'caption',
      R`Eq. 10 — reproduced: the d = 1 multiscale order parameter, 16 Pauli strings with weights ±¼. Blue X, orange Z, purple Y, dashed outline identity.`.replace('16', String(hz.sop_d1.length)), sopFig);

    // --- term growth ---
    const tc = (hz.term_counts || []).slice().sort((a, b) => a.depth - b.depth);
    const gfig = h('div', 'fig', null, body);
    h('div', 'chip-lab', 'How many Pauli strings the measured operator contains at depth d', gfig);
    const pdiv = h('div', null, null, gfig); pdiv.style.height = '300px';
    const maxT = Math.max(...tc.map(t => t.terms));
    U.plot(pdiv, [{
      type: 'bar', x: tc.map(t => t.depth), y: tc.map(t => t.terms), width: tc.map(() => 0.34),
      marker: { color: C.fc }, text: tc.map(t => t.terms.toLocaleString('en-US')), textposition: 'outside', cliponaxis: false,
      textfont: { family: '"JetBrains Mono", monospace', size: 12, color: C.ink },
      customdata: tc.map(t => [t.max_weight, t.seconds]),
      hovertemplate: 'depth %{x}<br>%{y:,} terms<br>longest string: %{customdata[0]} sites<br>expanded in %{customdata[1]:.2f} s<extra></extra>',
    }], {
      margin: { l: 70, r: 16, t: 28, b: 52 },
      xaxis: { title: { text: 'depth d' }, tickvals: tc.map(t => t.depth), range: [Math.min(...tc.map(t => t.depth)) - 0.6, Math.max(...tc.map(t => t.depth)) + 0.6] },
      yaxis: { type: 'log', title: { text: 'number of Pauli strings' }, range: [0, Math.log10(maxT) + 0.55] },
      bargap: 0.5, showlegend: false,
    });
    const grow = tc.length > 1 ? tc[tc.length - 1].terms / tc[0].terms : null;
    h('div', 'caption',
      R`Fig. 8 — reproduced: terms in the multiscale order parameter against depth (log axis).` + (tc.length ? ` Longest string: ${tc.map(t => t.max_weight).join(', ')} sites at d = ${tc.map(t => t.depth).join(', ')}.` : ''), gfig);

    h('p', null,
      R`<b>What this buys us.</b> An ordinary string order parameter is one product of operators over a fixed length, and it decays as soon as we approach the phase boundary, so it needs ever more samples. The QCNN instead measures a <em>weighted sum of products of exponentially many string order parameters</em>` +
      (grow ? R`: the term count grows about ${Math.round(grow).toLocaleString('en-US')}-fold from d = ${tc[0].depth} to d = ${tc[tc.length - 1].depth}.` : '.') +
      R` It does so with a single measured qubit, and the sum stays sharp right up to the transition. That is the mechanism behind the sample-complexity advantage we quantify in §7.`, body);
  });
})();
