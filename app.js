import * as E from './core/evalplan.js';

const by = Object.fromEntries(E.PUBLISHED.policies.map(p => [p.id, p]));
const total = E.PUBLISHED.policies.reduce((s, p) => s + p.n, 0);
const COLORS = ['#0072b2', '#e69f00', '#009e73', '#d55e00'];
const $ = id => document.getElementById(id);
const pct = (v, d = 1) => (100 * v).toFixed(d) + '%';
const fmtP = p => p >= 0.0001 && p < 0.01 ? p.toPrecision(2) : p < 0.0001 ? p.toExponential(1).replace('e-', ' × 10⁻').replace(/⁻(\d+)/, (m, d) => '⁻' + [...d].map(c => '⁰¹²³⁴⁵⁶⁷⁸⁹'[c]).join('')) : p < 0.001 ? p.toFixed(4) : p.toFixed(3);
const tip = $('tip');
function showTip(e, html) { tip.innerHTML = html; tip.style.visibility = 'visible'; tip.style.left = Math.min(window.innerWidth - 320, e.clientX + 14) + 'px'; tip.style.top = (e.clientY + 14) + 'px'; }
function hideTip() { tip.style.visibility = 'hidden'; }

$('metaCounts').textContent = `${total} trials on ${E.PUBLISHED.policies.length} checkpoints`;
$('tocCounts').innerHTML = `${total} real-robot trials and seven DINO FD values, as published on Sep 10. Every other number on this page is computed here.`;

// ------------------------------------------------------------------ Figure 1: trend
const SWEEPS = {
  compute: { ids: E.PUBLISHED.figures.compute.ids, x: id => Math.log2(E.COMPUTE_SHARE[id]), cat: id => by[id].compute, qual: 'M', unit: 'per doubling of pre-training compute', axis: 'log₂ compute share' },
  size: { ids: E.PUBLISHED.figures.size.ids, x: id => ['XS', 'S', 'M1', 'L'].indexOf(id), cat: id => by[id].size, qual: '', unit: 'per model-size step', axis: 'size rank' },
};
let sweep = 'compute';
function violinPath(cx, k, n, Y, halfW) {
  const a = k + 1, b = n - k + 1;
  const lo = E.betaQuantile(0.00125, a, b), hi = E.betaQuantile(0.99875, a, b);
  const steps = 60, pts = []; let maxd = 0;
  for (let s = 0; s <= steps; s++) { const p = lo + (hi - lo) * s / steps; const d = E.betaPdf(p, a, b); pts.push([p, d]); maxd = Math.max(maxd, d); }
  const right = pts.map(([p, d]) => `${(cx + halfW * d / maxd).toFixed(1)},${Y(p).toFixed(1)}`);
  const left = pts.slice().reverse().map(([p, d]) => `${(cx - halfW * d / maxd).toFixed(1)},${Y(p).toFixed(1)}`);
  const dObs = E.betaPdf(k / n, a, b) / maxd;
  return { d: `M${right.join(' L')} L${left.join(' L')} Z`, barHalf: Math.max(6, halfW * dObs * 0.9) };
}
const RES = {};
function drawTrend() {
  const S = SWEEPS[sweep], pol = S.ids.map(id => by[id]);
  const xs = S.ids.map(S.x), ks = pol.map(p => p.k), ns = pol.map(p => p.n);
  const an = E.analyzeCounts(pol), fit = E.logisticFit(xs, ks, ns), iso = E.isoTest(ks, ns, 2000);
  const L = 92, R = 690, T = 18, B = 300, Y = p => B - p * (B - T);
  const cx = i => L + 40 + i * (R - L - 80) / (pol.length - 1);
  let g = '';
  for (const v of [0, 0.25, 0.5, 0.75, 1]) g += `<line class="gridline" x1="${L}" x2="${R}" y1="${Y(v)}" y2="${Y(v)}"/><text class="tick" x="${L - 12}" y="${Y(v) + 4}" text-anchor="end">${v * 100}%</text>`;
  g += `<text class="category" transform="translate(28 ${(T + B) / 2}) rotate(-90)" text-anchor="middle">Task performance</text><text class="qualifier" transform="translate(46 ${(T + B) / 2}) rotate(-90)" text-anchor="middle">at-speed completion rate</text>`;
  // logistic band and curve, drawn between category positions
  const curve = [], up = [], dn = [];
  for (let i = 0; i < pol.length - 1; i++) for (let s = 0; s <= 20; s++) {
    const f = s / 20, x = xs[i] + (xs[i + 1] - xs[i]) * f, px = cx(i) + (cx(i + 1) - cx(i)) * f;
    const pr = E.predictLogit(fit, x, false); curve.push(`${px.toFixed(1)},${Y(pr.p).toFixed(1)}`); up.push(`${px.toFixed(1)},${Y(pr.hi).toFixed(1)}`); dn.unshift(`${px.toFixed(1)},${Y(pr.lo).toFixed(1)}`);
  }
  g += `<path d="M${up.join(' L')} L${dn.join(' L')} Z" fill="#292a27" opacity=".06"/><path d="M${curve.join(' L')}" fill="none" stroke="#292a27" stroke-width="1.6" stroke-dasharray="5 4"/>`;
  // isotonic step
  let st = ''; iso.fitted.forEach((v, i) => { const x0 = cx(i) - 14, x1 = cx(i) + 14; st += `${i ? 'L' : 'M'}${x0},${Y(v)} L${x1},${Y(v)} `; });
  g += `<path d="${st}" fill="none" stroke="#d55e00" stroke-width="2" opacity=".85"/>`;
  pol.forEach((p, i) => {
    const c = COLORS[i % 4], v = violinPath(cx(i), p.k, p.n, Y, 24), r = p.k / p.n;
    g += `<g class="vio" data-i="${i}"><path d="${v.d}" fill="${c}" fill-opacity=".12" stroke="${c}" stroke-width="2"/><line x1="${cx(i) - v.barHalf}" x2="${cx(i) + v.barHalf}" y1="${Y(r)}" y2="${Y(r)}" stroke="${c}" stroke-width="2.6" stroke-linecap="round"/></g>`;
    g += `<text class="valueLabel" x="${cx(i) + 30}" y="${Y(r) - 2}" fill="${c}">${pct(r)}</text><text class="valueLabel sample" x="${cx(i) + 30}" y="${Y(r) + 15}" fill="${c}" style="font-weight:400">n = ${p.n}</text>`;
    g += `<text class="letter" x="${cx(i) - 32}" y="${Y(r) + 5}" text-anchor="end">${an.lettersFisher[i]}</text>`;
    g += `<text class="category" x="${cx(i)}" y="${B + 26}" text-anchor="middle">${S.cat(p.id)}</text>${S.qual ? `<text class="qualifier" x="${cx(i)}" y="${B + 44}" text-anchor="middle">${S.qual}</text>` : ''}`;
  });
  $('trendChart').innerHTML = g;
  $('trendChart').querySelectorAll('.vio').forEach(el => {
    const p = pol[+el.dataset.i], [lo, hi] = E.credible(p.k, p.n);
    el.onmousemove = e => showTip(e, `<b>${p.label} · posterior uncertainty</b><br>${p.k} at-speed completions in ${p.n} trials. 95% credible interval: <b>${pct(lo)}–${pct(hi)}</b>.`);
    el.onmouseleave = hideTip;
  });
  // metrics
  const xm = xs.reduce((a, b) => a + b, 0) / xs.length, etaM = fit.b0 + fit.b1 * xm, se = Math.sqrt(fit.cov[1][1]);
  const ppStep = b => 1 / (1 + Math.exp(-(etaM + b / 2))) - 1 / (1 + Math.exp(-(etaM - b / 2)));
  const nSep = an.fisherReject.filter(Boolean).length;
  RES[sweep] = { p: fit.p, iso: iso.p, nSep, pairs: an.pairs.length, pp: ppStep(fit.b1), n: ns.reduce((a, b) => a + b, 0) };
  $('trendMetrics').innerHTML = `
    <div class="metric"><div class="k">Trend, logistic on ${S.axis}</div><div class="v">${ppStep(fit.b1) >= 0 ? '+' : ''}${(100 * ppStep(fit.b1)).toFixed(1)}<small>pp ${S.unit}</small></div><div class="d">95% CI ${(100 * ppStep(fit.b1 - 1.96 * se)).toFixed(1)} to ${(100 * ppStep(fit.b1 + 1.96 * se)).toFixed(1)} pp · LR χ²(1) = ${fit.lr.toFixed(1)}, p = ${fmtP(fit.p)}</div></div>
    <div class="metric"><div class="k">Order-restricted test</div><div class="v">p = ${fmtP(iso.p)}</div><div class="d">isotonic LRT vs one common rate, ${(2000).toLocaleString()} parametric-bootstrap draws (smallest p it can report: 0.0005)</div></div>
    <div class="metric"><div class="k">Pairwise letters on the same trials</div><div class="v">${nSep}<small>of ${an.pairs.length} pairs separated</small></div><div class="d">Fisher exact + Holm, as in the post (letters on the chart)</div></div>`;
  $('trendSub').textContent = sweep === 'compute' ? 'The M-size model at four pre-training compute budgets, each post-trained once on the full demonstration set' : 'Four sizes, one pre-training recipe, all post-trained on the full demonstration set';
  $('trendCap').textContent = sweep === 'compute'
    ? `Violins are Beta(k+1, n−k+1) posteriors over each checkpoint's true at-speed completion rate; letters reproduce the post's Fisher + Holm groups. On the same ${ns.reduce((a, b) => a + b, 0)} trials, a logistic trend in log compute and an order-restricted test both reject "compute does not matter", even though only ${nSep} of ${an.pairs.length} pairs separate. The non-decreasing fit (orange) is flat from 0.37× to 1×, matching the post's saturation reading.`
    : `Same construction for model size. Letters reproduce the post's Figure 4 (a, b, bc, c). The trend across size steps is far stronger than any single pair.`;
}
$('trendTabs').querySelectorAll('button').forEach(b => b.onclick = () => { sweep = b.dataset.k; $('trendTabs').querySelectorAll('button').forEach(x => x.classList.toggle('on', x === b)); drawTrend(); });

// ------------------------------------------------------------------ Figure 2: forecast from DINO FD
const fdIds = Object.keys(E.DINO_FD);
const fdFit = E.logisticFit(fdIds.map(i => -Math.log(E.DINO_FD[i])), fdIds.map(i => by[i].k), fdIds.map(i => by[i].n));
const fdCache = {};
function drawFD() {
  const fd = +$('fdSlider').value; $('fdVal').textContent = fd.toFixed(1);
  const L = 92, R = 690, T = 18, B = 300, Y = p => B - p * (B - T);
  const lx0 = Math.log(110), lx1 = Math.log(3.5), X = f => L + (Math.log(f) - lx0) / (lx1 - lx0) * (R - L);
  let g = '';
  for (const v of [0, 0.25, 0.5, 0.75, 1]) g += `<line class="gridline" x1="${L}" x2="${R}" y1="${Y(v)}" y2="${Y(v)}"/><text class="tick" x="${L - 12}" y="${Y(v) + 4}" text-anchor="end">${v * 100}%</text>`;
  for (const t of [100, 70, 50, 30, 20, 15, 10, 7, 5]) g += `<text class="tick" x="${X(t)}" y="${B + 24}" text-anchor="middle">${t}</text>`;
  g += `<text class="category" transform="translate(28 ${(T + B) / 2}) rotate(-90)" text-anchor="middle">Task performance</text><text class="qualifier" transform="translate(46 ${(T + B) / 2}) rotate(-90)" text-anchor="middle">at-speed completion rate</text>`;
  g += `<text x="${(L + R) / 2}" y="${B + 54}" text-anchor="middle" style="font-size:16px;font-weight:500;fill:#292a27">Pre-training quality</text><text class="qualifier" x="${(L + R) / 2}" y="${B + 72}" text-anchor="middle">DINO FD, log scale, lower is better; quality increases to the right</text>`;
  const curve = [], up = [], dn = [];
  for (let s = 0; s <= 80; s++) { const f = Math.exp(lx0 + (lx1 - lx0) * s / 80), pr = E.predictLogit(fdFit, -Math.log(f)); curve.push(`${X(f).toFixed(1)},${Y(pr.p).toFixed(1)}`); up.push(`${X(f).toFixed(1)},${Y(pr.hi).toFixed(1)}`); dn.unshift(`${X(f).toFixed(1)},${Y(pr.lo).toFixed(1)}`); }
  g += `<path d="M${up.join(' L')} L${dn.join(' L')} Z" fill="#292a27" opacity=".06"/><path d="M${curve.join(' L')}" fill="none" stroke="#292a27" stroke-width="1.4" stroke-dasharray="5 4"/>`;
  fdIds.forEach(id => {
    const p = by[id], r = p.k / p.n, [lo, hi] = E.credible(p.k, p.n), x = X(E.DINO_FD[id]);
    g += `<g class="pt" data-id="${id}"><line x1="${x}" x2="${x}" y1="${Y(lo)}" y2="${Y(hi)}" stroke="#0072b2" stroke-width="2"/><line x1="${x - 5}" x2="${x + 5}" y1="${Y(lo)}" y2="${Y(lo)}" stroke="#0072b2" stroke-width="2"/><line x1="${x - 5}" x2="${x + 5}" y1="${Y(hi)}" y2="${Y(hi)}" stroke="#0072b2" stroke-width="2"/><circle cx="${x}" cy="${Y(r)}" r="5.5" fill="#0072b2"/><rect x="${x - 12}" y="${Y(hi) - 4}" width="24" height="${Y(lo) - Y(hi) + 8}" fill="transparent"/></g>`;
    const lab = p.size === 'M' ? `<tspan class="qualifier" x="${x}" dy="0">${p.compute}</tspan><tspan x="${x}" dy="16">M</tspan>` : `<tspan x="${x}">${p.size}</tspan>`;
    g += `<text class="pointLabel" x="${x}" y="${Y(hi) - (p.size === 'M' ? 26 : 10)}" text-anchor="middle" style="font-weight:400">${lab}</text>`;
  });
  const pr = E.predictLogit(fdFit, -Math.log(fd)), x = X(fd);
  g += `<line x1="${x}" x2="${x}" y1="${Y(pr.lo)}" y2="${Y(pr.hi)}" stroke="#fd640b" stroke-width="2" stroke-dasharray="3 3"/><circle cx="${x}" cy="${Y(pr.p)}" r="6" fill="#fefdfb" stroke="#fd640b" stroke-width="2.2"/><text class="pointLabel" x="${x}" y="${Y(pr.hi) - 10}" text-anchor="middle" fill="#fd640b" style="fill:#fd640b">new</text>`;
  $('fdChart').innerHTML = g;
  $('fdChart').querySelectorAll('.pt').forEach(el => { const p = by[el.dataset.id], [lo, hi] = E.credible(p.k, p.n); el.onmousemove = e => showTip(e, `<b>${p.label}</b> · DINO FD ${E.DINO_FD[p.id]}<br>${p.k}/${p.n} at speed, 95% credible interval <b>${pct(lo)}–${pct(hi)}</b>.`); el.onmouseleave = hideTip; });
  const key = fd.toFixed(1);
  const beatL = fdCache['L' + key] ||= E.trialsToBeat(pr, by.L, { reps: 160 });
  const beatM = fdCache['M' + key] ||= E.trialsToBeat(pr, by.M1, { reps: 160 });
  const fmtN = r => Number.isFinite(r.n) ? `${r.n}<small>trials per arm</small>` : `> 2,000<small>per arm</small>`;
  $('fdMetrics').innerHTML = `
    <div class="metric"><div class="k">Forecast for FD ${fd.toFixed(1)}</div><div class="v">${pct(pr.p)}</div><div class="d">95% interval ${pct(pr.lo)}–${pct(pr.hi)}, widened by overdispersion φ = ${fdFit.phi.toFixed(2)} across checkpoints</div></div>
    <div class="metric"><div class="k">To show it beats L (${pct(by.L.k / by.L.n)}, n = ${by.L.n})</div><div class="v">${fmtN(beatL)}</div><div class="d">80% power, Fisher two-sided α = 0.05; L gets the same new trials; averaged over the forecast</div></div>
    <div class="metric"><div class="k">To show it beats M · 1× (${pct(by.M1.k / by.M1.n)}, n = ${by.M1.n})</div><div class="v">${fmtN(beatM)}</div><div class="d">same rule; a cheaper first gate before the head-to-head with L</div></div>`;
  $('fdCap').textContent = `Points are the seven published checkpoints with 95% credible bars. The dashed curve is a binomial logistic fit of at-speed completion on −log DINO FD (LR χ²(1) = ${fdFit.lr.toFixed(1)}); its band is inflated by the Pearson dispersion φ = ${fdFit.phi.toFixed(2)}, because checkpoints scatter around the curve more than trial noise explains (one post-training run each). The orange point is a forecast, not a measurement: it says how many robot trials a checkpoint is worth before any are spent.`;
}
$('fdSlider').addEventListener('input', drawFD);

// ------------------------------------------------------------------ Figure 3: allocation
function drawAlloc() {
  const ids = E.PUBLISHED.figures.compute.ids, xs = ids.map(i => Math.log2(E.COMPUTE_SHARE[i])), truth = ids.map(i => by[i].k / by[i].n);
  const rows = [
    { name: 'Published allocation', alloc: ids.map(i => by[i].n) },
    { name: 'Equal, 100 each', alloc: [100, 100, 100, 100] },
    { name: 'Equal, 75 each', alloc: [75, 75, 75, 75] },
    { name: 'Ends + middle', alloc: [120, 0, 60, 120] },
    { name: 'Ends only', alloc: [150, 0, 0, 150] },
  ].map(r => ({ ...r, n: r.alloc.reduce((a, b) => a + b, 0), power: E.trendPower(truth, xs, r.alloc, { reps: 400 }) }));
  const ref = rows[0].power;
  $('allocSub').textContent = `Each row replays the compute study 400 times at the observed rates (${truth.map(t => pct(t)).join(', ')}) and reports how often a one-sided logistic trend test at α = 0.05 finds the increase. Trials per checkpoint: 0.08× / 0.18× / 0.37× / 1×.`;
  $('allocBars').innerHTML = `<div class="barHead"><span>Allocation</span><span>Power</span><span>Trials</span><span>Power</span></div>` + rows.map((r, k) => `
    <div class="barRow"><div class="name">${r.name}<small>${r.alloc.join(' / ')}</small></div><div class="barTrack"><i style="width:${100 * r.power}%;background:${k === 0 ? '#8a8580' : '#0072b2'}"></i><span class="ref" style="left:${100 * ref}%"></span></div><div class="barValue">${r.n}</div><div class="barValue">${pct(r.power, 0)}</div></div>`).join('');
  RES.alloc = rows;
  const best = rows.filter(r => r.n < rows[0].n && r.power >= ref - 0.01).sort((a, b) => a.n - b.n)[0];
  $('allocCap').textContent = best
    ? `The dashed line is the published design's power (${pct(ref, 0)} with ${rows[0].n} trials). "${best.name}" matches it with ${best.n} trials, ${pct(1 - best.n / rows[0].n, 0)} fewer. Two-point designs cannot see curvature, so they could not have shown the flat step from 0.37× to 1×; the ends-plus-middle design keeps that check at ${pct(rows[3].power, 0)} power with ${rows[3].n} trials.`
    : `No cheaper allocation matched the published design's power in this replay.`;
}

// ------------------------------------------------------------------ Figure 4: peeking
function drawPeek() {
  const r = E.nullCheck({ reps: 300 });
  const sizeAn = E.analyzeCounts(E.PUBLISHED.figures.size.ids.map(i => by[i]));
  const rows = [
    { name: 'Fisher + Holm, one look at the end', v: r.fixedFisher, c: '#8a8580' },
    { name: `Fisher + Holm, recomputed every ${r.every} trials`, v: r.peekingFisher, c: '#d55e00' },
    { name: 'Anytime-valid e-values + Holm, every trial', v: r.anytimeE, c: '#009e73' },
  ];
  $('peekSub').textContent = `${r.K} policies with the same true rate (${pct(r.p, 0)}), up to ${r.perPolicy} trials each, ${r.reps} simulated studies. A study counts as a false separation if any pair is ever declared different.`;
  $('peekBars').innerHTML = `<div class="barHead"><span>Procedure</span><span>Family-wise false separations</span><span></span><span>Rate</span></div>` + rows.map(x => `
    <div class="barRow"><div class="name">${x.name}</div><div class="barTrack"><i style="width:${Math.min(100, 100 * x.v / 0.2)}%;background:${x.c}"></i><span class="ref" style="left:${100 * 0.05 / 0.2}%"></span></div><div class="barValue"></div><div class="barValue">${pct(x.v)}</div></div>`).join('');
  $('peekCap').textContent = `Track width is 20%; the dashed line is the nominal 5%. Re-running fixed-n letters after every ${r.every}-trial session lets the error rate drift to ${pct(r.peekingFisher, 0)}. Anytime-valid e-values (a Beta-mixture likelihood ratio against the pooled null, Holm across pairs) stay under 5% however often you look, but pay for it in power: on the published model-size counts they separate ${sizeAn.eReject.filter(Boolean).length} of 6 pairs where Fisher + Holm separates ${sizeAn.fisherReject.filter(Boolean).length}. The practical rule: fix the trial count in advance, or use anytime-valid letters whenever early stopping is on the table.`;
}

function drawFound() {
  const c = RES.compute, rows = RES.alloc, best = rows.filter(r => r.n < rows[0].n && r.power >= rows[0].power - 0.01).sort((a, b) => a.n - b.n)[0];
  const pr = E.predictLogit(fdFit, -Math.log(6)), bl = E.trialsToBeat(pr, by.L, { reps: 160 });
  $('found').innerHTML = `
    <li>Pre-training compute helps: a logistic trend over the four M budgets gives <b>p = ${fmtP(c.p)}</b> (+${(100 * c.pp).toFixed(1)} pp per doubling) on the same ${c.n} trials where only ${c.nSep} of ${c.pairs} pairwise letters separate.</li>
    <li>${best ? `The same trend power (${pct(rows[0].power, 0)}) is reachable with <b>${best.n} trials instead of ${rows[0].n}</b> by putting them at the ends of the sweep, or ${rows[3].n} with a middle point kept to check saturation.` : 'No cheaper allocation matched the published power in this replay.'}</li>
    <li>A checkpoint with DINO FD 6.0 forecasts at <b>${pct(pr.p)}</b> (${pct(pr.lo)}–${pct(pr.hi)}); showing it beats L would take about <b>${Number.isFinite(bl.n) ? bl.n : '2,000+'} trials per arm</b>, a number worth knowing before booking the robot.</li>`;
}
drawTrend(); drawFD(); drawAlloc(); drawPeek(); drawFound();
