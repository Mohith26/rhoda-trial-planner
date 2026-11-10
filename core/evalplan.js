// Anytime-valid evaluation planner for real-robot policy comparisons.
//
// Problem: comparing K robot policies on a real robot costs hundreds of trials per policy. Fixed-n tests
// (Fisher exact + Holm) are only valid if the number of trials is fixed in advance; stopping a clearly
// worse policy early, or peeking after every session, inflates the false-separation rate.
//
// This module:
//  1. computes an e-process for every pair H0: p_i = p_j  (mixture likelihood over independent Beta(1,1)
//     rates divided by the maximized pooled null likelihood; by Ville's inequality P(sup_t E_t >= 1/a) <= a),
//  2. turns running-max e-values into anytime-valid p-values and applies Holm across all pairs
//     (valid at any stopping time, arbitrary dependence), and draws compact letter displays from them,
//  3. schedules the next real-robot trial to the policy whose one extra trial is expected to grow the
//     evidence on unresolved pairs the most, and retires pairs that are separated or cannot be separated
//     within the trial cap (futility forecast),
//  4. replays and Monte-Carlo-checks the schedule against a fixed allocation.
//
// Pure ES module, no dependencies.

// ------------------------------------------------------------------ special functions
const LANCZOS = [676.5203681218851, -1259.1392167224028, 771.32342877765313, -176.61502916214059, 12.507343278686905, -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7];
export function lgamma(x) {
  if (x < 0.5) return Math.log(Math.PI / Math.abs(Math.sin(Math.PI * x))) - lgamma(1 - x);
  x -= 1; let a = 0.99999999999980993; const t = x + 7.5;
  for (let i = 0; i < 8; i++) a += LANCZOS[i] / (x + i + 1);
  return 0.5 * Math.log(2 * Math.PI) + (x + 0.5) * Math.log(t) - t + Math.log(a);
}
export const lbeta = (a, b) => lgamma(a) + lgamma(b) - lgamma(a + b);
const xlogy = (x, y) => (x === 0 ? 0 : x * Math.log(y));
export function lchoose(n, k) { return lgamma(n + 1) - lgamma(k + 1) - lgamma(n - k + 1); }

export function mulberry32(seed) {
  let a = seed >>> 0;
  return function () { a = (a + 0x6D2B79F5) >>> 0; let t = a; t = Math.imul(t ^ (t >>> 15), t | 1); t ^= t + Math.imul(t ^ (t >>> 7), t | 61); return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
}

// ------------------------------------------------------------------ Beta posterior helpers (uniform prior)
export function betaPdf(p, a, b) { if (p <= 0 || p >= 1) return 0; return Math.exp((a - 1) * Math.log(p) + (b - 1) * Math.log(1 - p) - lbeta(a, b)); }
export function betaQuantile(q, a, b) {
  // bisection on the regularized incomplete beta via numeric integration of the pdf (adequate for plotting)
  let lo = 0, hi = 1;
  for (let it = 0; it < 50; it++) { const mid = (lo + hi) / 2; if (betaCdf(mid, a, b) < q) lo = mid; else hi = mid; }
  return (lo + hi) / 2;
}
export function betaCdf(x, a, b) {
  if (x <= 0) return 0; if (x >= 1) return 1;
  // continued fraction (Numerical Recipes betacf)
  const bt = Math.exp(lgamma(a + b) - lgamma(a) - lgamma(b) + a * Math.log(x) + b * Math.log(1 - x));
  const cf = (a, b, x) => {
    const MAXIT = 300, EPS = 3e-14, FPMIN = 1e-300; let qab = a + b, qap = a + 1, qam = a - 1, c = 1, d = 1 - qab * x / qap;
    if (Math.abs(d) < FPMIN) d = FPMIN; d = 1 / d; let h = d;
    for (let m = 1; m <= MAXIT; m++) {
      const m2 = 2 * m; let aa = m * (b - m) * x / ((qam + m2) * (a + m2));
      d = 1 + aa * d; if (Math.abs(d) < FPMIN) d = FPMIN; c = 1 + aa / c; if (Math.abs(c) < FPMIN) c = FPMIN; d = 1 / d; h *= d * c;
      aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2));
      d = 1 + aa * d; if (Math.abs(d) < FPMIN) d = FPMIN; c = 1 + aa / c; if (Math.abs(c) < FPMIN) c = FPMIN; d = 1 / d; const del = d * c; h *= del;
      if (Math.abs(del - 1) < EPS) break;
    }
    return h;
  };
  return x < (a + 1) / (a + b + 2) ? bt * cf(a, b, x) / a : 1 - bt * cf(b, a, 1 - x) / b;
}
export function credible(k, n, level = 0.95) { const a = k + 1, b = n - k + 1, t = (1 - level) / 2; return [betaQuantile(t, a, b), betaQuantile(1 - t, a, b)]; }

// ------------------------------------------------------------------ fixed-n tests (what the post uses)
export function fisherTwoSided(k1, n1, k2, n2) {
  const K = k1 + k2, N = n1 + n2;
  const lo = Math.max(0, K - n2), hi = Math.min(K, n1);
  const lp = x => lchoose(n1, x) + lchoose(n2, K - x) - lchoose(N, K);
  const obs = lp(k1); let p = 0;
  for (let x = lo; x <= hi; x++) { const v = lp(x); if (v <= obs + 1e-7) p += Math.exp(v); }
  return Math.min(1, p);
}
export function holm(pvals, alpha) {
  const m = pvals.length, idx = pvals.map((p, i) => [p, i]).sort((a, b) => a[0] - b[0]);
  const adj = new Array(m); let run = 0;
  idx.forEach(([p, i], r) => { run = Math.max(run, Math.min(1, (m - r) * p)); adj[i] = run; });
  return { adjusted: adj, reject: adj.map(a => a <= alpha) };
}

// ------------------------------------------------------------------ e-process for H0: p_i = p_j
export let PRIOR_A = 0.5; // Beta(a, a) mixture on each rate; 0.5 = Jeffreys / Krichevsky-Trofimov
export function setPrior(a) { PRIOR_A = a; }
export function logE(k1, n1, k2, n2) {
  const K = k1 + k2, N = n1 + n2; if (N === 0) return 0;
  const a = PRIOR_A, lb0 = lbeta(a, a);
  const num = lbeta(k1 + a, n1 - k1 + a) - lb0 + lbeta(k2 + a, n2 - k2 + a) - lb0;
  const ph = K / N, den = xlogy(K, ph) + xlogy(N - K, 1 - ph);
  return num - den;
}

// ------------------------------------------------------------------ letters (insert-absorb, Piepho 2004)
export function letters(order, separated) {
  // order: indices sorted by rate ascending; separated(i, j) -> true if the pair differs
  const K = order.length; let groups = [new Set(order)];
  for (let a = 0; a < K; a++) for (let b = a + 1; b < K; b++) {
    const i = order[a], j = order[b]; if (!separated(i, j)) continue;
    const next = [];
    for (const g of groups) {
      if (g.has(i) && g.has(j)) { const g1 = new Set(g); g1.delete(i); const g2 = new Set(g); g2.delete(j); next.push(g1, g2); } else next.push(g);
    }
    // absorb: drop groups that are subsets of others
    groups = next.filter((g, x) => !next.some((h, y) => y !== x && [...g].every(v => h.has(v)) && (h.size > g.size || y < x)));
  }
  // order groups by their lowest member position, assign a, b, c...
  const pos = Object.fromEntries(order.map((v, k) => [v, k]));
  groups.sort((g, h) => Math.min(...[...g].map(v => pos[v])) - Math.min(...[...h].map(v => pos[v])));
  const out = Object.fromEntries(order.map(v => [v, '']));
  groups.forEach((g, k) => { for (const v of g) out[v] += String.fromCharCode(97 + k); });
  return out;
}

export function pairList(K) { const P = []; for (let i = 0; i < K; i++) for (let j = i + 1; j < K; j++) P.push([i, j]); return P; }

// ------------------------------------------------------------------ analysis of a fixed table of counts
export function analyzeCounts(policies, alpha = 0.05) {
  const K = policies.length, P = pairList(K);
  const fisher = P.map(([i, j]) => fisherTwoSided(policies[i].k, policies[i].n, policies[j].k, policies[j].n));
  const fh = holm(fisher, alpha);
  const le = P.map(([i, j]) => logE(policies[i].k, policies[i].n, policies[j].k, policies[j].n));
  const anyP = le.map(l => Math.min(1, Math.exp(-l)));
  const eh = holm(anyP, alpha);
  const rates = policies.map(p => p.k / p.n);
  const order = [...rates.keys()].sort((a, b) => rates[a] - rates[b]);
  const sepF = new Set(P.map((p, x) => fh.reject[x] ? p.join('-') : null).filter(Boolean));
  const sepE = new Set(P.map((p, x) => eh.reject[x] ? p.join('-') : null).filter(Boolean));
  const key = (i, j) => (i < j ? `${i}-${j}` : `${j}-${i}`);
  return {
    pairs: P, fisher, fisherAdj: fh.adjusted, fisherReject: fh.reject, logE: le, anytimeP: anyP, eAdj: eh.adjusted, eReject: eh.reject,
    lettersFisher: letters(order, (i, j) => sepF.has(key(i, j))), lettersE: letters(order, (i, j) => sepE.has(key(i, j))),
  };
}

// ------------------------------------------------------------------ the scheduler
// expected growth of log E for pair (i, j) if policy i gets one more trial (under current posterior means)
function expectedGain(c, i, j) {
  const pi = (c[i].k + 1) / (c[i].n + 2);
  const base = logE(c[i].k, c[i].n, c[j].k, c[j].n);
  const up = logE(c[i].k + 1, c[i].n + 1, c[j].k, c[j].n), dn = logE(c[i].k, c[i].n + 1, c[j].k, c[j].n);
  return pi * up + (1 - pi) * dn - base;
}
// trials still needed (on both arms, split evenly) for log E to reach the target, from the current gap
export function trialsToSeparate(c, i, j, target) {
  const pi = (c[i].k + 1) / (c[i].n + 2), pj = (c[j].k + 1) / (c[j].n + 2), pb = (pi + pj) / 2;
  const kl = (p, q) => xlogy(p, p / q) + xlogy(1 - p, (1 - p) / (1 - q));
  const rate = (kl(pi, pb) + kl(pj, pb)) / 2; // per trial, with trials alternating between the two arms
  const now = logE(c[i].k, c[i].n, c[j].k, c[j].n);
  if (rate < 1e-9) return Infinity;
  return Math.max(0, (target - now) / rate);
}

/**
 * Run a sequential evaluation.
 * truth: true success probabilities, or null when replaying a recorded outcome stream.
 * opts: { alpha, cap (max trials per policy), rule: 'adaptive' | 'fixed', fixedN (per policy, rule 'fixed'),
 *         seed, minTrials, outcomes (optional pre-drawn outcome streams per policy) }
 */
export function runSchedule(truth, opts = {}) {
  const K = truth.length, alpha = opts.alpha ?? 0.05, cap = opts.cap ?? 260, minTrials = opts.minTrials ?? 10, futileMin = opts.futileMin ?? 30;
  const P = pairList(K), M = P.length, target = Math.log(M / alpha);
  const rand = mulberry32(opts.seed ?? 1);
  const streams = opts.outcomes || truth.map(p => Array.from({ length: cap }, () => (rand() < p ? 1 : 0)));
  const c = truth.map(() => ({ k: 0, n: 0 }));
  const maxLogE = new Array(M).fill(0);
  const sepAt = new Array(M).fill(null), futileAt = new Array(M).fill(null);
  const log = [];
  const status = () => {
    const anyP = maxLogE.map(l => Math.min(1, Math.exp(-l)));
    return holm(anyP, alpha).reject;
  };
  let t = 0;
  const fixed = opts.rule === 'fixed';
  const order = fixed ? interleave(opts.fixedN) : null; let oi = 0;
  while (true) {
    let pick = -1;
    if (fixed) { if (oi >= order.length) break; pick = order[oi++]; }
    else {
      const rej = status();
      const open = P.map((pr, x) => !rej[x] && futileAt[x] === null);
      if (!open.some(Boolean)) break;
      let best = -Infinity;
      for (let i = 0; i < K; i++) {
        if (c[i].n >= cap) continue;
        if (c[i].n < minTrials && P.some((pr, x) => open[x] && pr.includes(i))) { if (1e9 - c[i].n > best) { best = 1e9 - c[i].n; pick = i; } continue; }
        let g = 0; P.forEach(([a, b], x) => { if (!open[x]) return; if (a === i) g += expectedGain(c, a, b); else if (b === i) g += expectedGain(c, b, a); });
        if (g > best) { best = g; pick = i; }
      }
      if (pick < 0) break;
    }
    const y = streams[pick][c[pick].n];
    c[pick].k += y; c[pick].n += 1; t++;
    P.forEach(([i, j], x) => { if (i === pick || j === pick) maxLogE[x] = Math.max(maxLogE[x], logE(c[i].k, c[i].n, c[j].k, c[j].n)); });
    const rej = status();
    P.forEach(([i, j], x) => {
      if (rej[x] && sepAt[x] === null) sepAt[x] = t;
      if (!fixed && !rej[x] && futileAt[x] === null && c[i].n >= futileMin && c[j].n >= futileMin) {
        const need = trialsToSeparate(c, i, j, target);
        const room = (cap - c[i].n) + (cap - c[j].n);
        if (need > 1.5 * room) futileAt[x] = t;
      }
    });
    log.push({ t, pick, y, resolved: P.filter((_, x) => rej[x]).length, futile: futileAt.filter(v => v !== null).length });
  }
  const rej = status();
  return { counts: c, total: t, pairs: P, maxLogE, separated: rej, sepAt, futileAt, log, target, M };
}
function interleave(ns) {
  const K = ns.length, out = [], left = [...ns]; let any = true;
  while (any) { any = false; for (let i = 0; i < K; i++) if (left[i] > 0) { out.push(i); left[i]--; any = true; } }
  return out;
}

// ------------------------------------------------------------------ Monte Carlo checks
export function monteCarlo(truth, opts = {}) {
  const R = opts.reps ?? 200, seed0 = opts.seed ?? 1000;
  const totals = [], sepCounts = [];
  const refSep = opts.referenceSeparated; // boolean per pair from the published fixed-n analysis
  let matched = 0;
  for (let r = 0; r < R; r++) {
    const res = runSchedule(truth, { ...opts, seed: seed0 + r });
    totals.push(res.total); sepCounts.push(res.separated.filter(Boolean).length);
    if (refSep && refSep.every((s, x) => !s || res.separated[x])) matched++;
  }
  const s = [...totals].sort((a, b) => a - b), q = f => s[Math.min(s.length - 1, Math.floor(f * s.length))];
  return { reps: R, median: q(0.5), p10: q(0.1), p90: q(0.9), meanSeparated: sepCounts.reduce((a, b) => a + b, 0) / R, matchedRate: refSep ? matched / R : null };
}

// Family-wise false separations under a null where every policy has the same rate.
export function nullCheck(opts = {}) {
  const K = opts.K ?? 4, p = opts.p ?? 0.7, R = opts.reps ?? 400, perPolicy = opts.perPolicy ?? 150, every = opts.every ?? 10, alpha = opts.alpha ?? 0.05;
  const rand = mulberry32(opts.seed ?? 77);
  const P = pairList(K), M = P.length;
  let peekFW = 0, eFW = 0, endFW = 0;
  for (let r = 0; r < R; r++) {
    const ys = Array.from({ length: K }, () => Array.from({ length: perPolicy }, () => (rand() < p ? 1 : 0)));
    let peekHit = false, eHit = false; const maxL = new Array(M).fill(0);
    const ks = new Array(K).fill(0);
    for (let n = 1; n <= perPolicy; n++) {
      for (let i = 0; i < K; i++) ks[i] += ys[i][n - 1];
      P.forEach(([i, j], x) => { maxL[x] = Math.max(maxL[x], logE(ks[i], n, ks[j], n)); });
      if (!eHit && holm(maxL.map(l => Math.min(1, Math.exp(-l))), alpha).reject.some(Boolean)) eHit = true;
      if (n % every === 0) {
        const fp = P.map(([i, j]) => fisherTwoSided(ks[i], n, ks[j], n));
        const rj = holm(fp, alpha).reject.some(Boolean);
        if (rj) peekHit = true;
        if (n === perPolicy && rj) endFW++;
      }
    }
    if (peekHit) peekFW++; if (eHit) eFW++;
  }
  return { K, p, reps: R, perPolicy, every, alpha, peekingFisher: peekFW / R, fixedFisher: endFW / R, anytimeE: eFW / R };
}

// ------------------------------------------------------------------ published counts (rhoda.ai, Sep 10 2026, Figure 1)
export const PUBLISHED = {
  source: 'https://www.rhoda.ai/research/scaling-web-video-pretraining (Figure 1)',
  task: 'Bearing unpacking, at-speed completion (all goals met, cycle time < 100 s)',
  figures: {
    size: { title: 'Model size', subtitle: 'Four sizes, one pre-training recipe, all post-trained on the full demonstration set', ids: ['XS', 'S', 'M1', 'L'], figure: 'Figure 4' },
    compute: { title: 'Pre-training compute', subtitle: 'The M-size model at four pre-training compute budgets', ids: ['M008', 'M018', 'M037', 'M1'], figure: 'Figure 7' },
  },
  policies: [
    { id: 'XS', label: 'XS', size: 'XS', compute: '1×', k: 1, n: 27 },
    { id: 'S', label: 'S', size: 'S', compute: '1×', k: 67, n: 103 },
    { id: 'M008', label: 'M · 0.08×', size: 'M', compute: '0.08×', k: 59, n: 102 },
    { id: 'M018', label: 'M · 0.18×', size: 'M', compute: '0.18×', k: 68, n: 101 },
    { id: 'M037', label: 'M · 0.37×', size: 'M', compute: '0.37×', k: 111, n: 149 },
    { id: 'M1', label: 'M · 1×', size: 'M', compute: '1×', k: 195, n: 259 },
    { id: 'L', label: 'L', size: 'L', compute: '1×', k: 94, n: 111 },
  ],
};

// ------------------------------------------------------------------ trend and forecasting on grouped counts
const sigmoid = z => 1 / (1 + Math.exp(-z));
export function erfc(x) { // Numerical Recipes erfcc
  const z = Math.abs(x), t = 1 / (1 + 0.5 * z);
  const r = t * Math.exp(-z * z - 1.26551223 + t * (1.00002368 + t * (0.37409196 + t * (0.09678418 + t * (-0.18628806 + t * (0.27886807 + t * (-1.13520398 + t * (1.48851587 + t * (-0.82215223 + t * 0.17087277)))))))));
  return x >= 0 ? r : 2 - r;
}
export const chi2sf1 = x => (x <= 0 ? 1 : erfc(Math.sqrt(x / 2)));
export const normCdf = z => 0.5 * erfc(-z / Math.SQRT2);

// binomial logistic regression of k/n on x (Newton-Raphson); returns coefficients, covariance, deviances
export function logisticFit(xs, ks, ns) {
  let b0 = 0, b1 = 0;
  for (let it = 0; it < 50; it++) {
    let g0 = 0, g1 = 0, h00 = 0, h01 = 0, h11 = 0;
    xs.forEach((x, i) => { const p = sigmoid(b0 + b1 * x), w = ns[i] * p * (1 - p), r = ks[i] - ns[i] * p; g0 += r; g1 += r * x; h00 += w; h01 += w * x; h11 += w * x * x; });
    const det = h00 * h11 - h01 * h01; const d0 = (h11 * g0 - h01 * g1) / det, d1 = (-h01 * g0 + h00 * g1) / det;
    b0 += d0; b1 += d1; if (Math.abs(d0) + Math.abs(d1) < 1e-10) break;
  }
  let h00 = 0, h01 = 0, h11 = 0, dev = 0, pearson = 0;
  const K = ks.reduce((a, b) => a + b, 0), N = ns.reduce((a, b) => a + b, 0), p0 = K / N; let dev0 = 0;
  xs.forEach((x, i) => {
    const p = sigmoid(b0 + b1 * x), w = ns[i] * p * (1 - p); h00 += w; h01 += w * x; h11 += w * x * x;
    const ll = (k, n, q) => xlogy(k, q) + xlogy(n - k, 1 - q), sat = (k, n) => xlogy(k, k / n) + xlogy(n - k, 1 - k / n);
    dev += 2 * (sat(ks[i], ns[i]) - ll(ks[i], ns[i], p)); dev0 += 2 * (sat(ks[i], ns[i]) - ll(ks[i], ns[i], p0));
    pearson += (ks[i] - ns[i] * p) ** 2 / (ns[i] * p * (1 - p));
  });
  const det = h00 * h11 - h01 * h01;
  const cov = [[h11 / det, -h01 / det], [-h01 / det, h00 / det]];
  const lr = dev0 - dev;
  const phi = xs.length > 2 ? Math.max(1, pearson / (xs.length - 2)) : 1; // quasi-binomial dispersion
  return { b0, b1, cov, dev, dev0, lr, p: chi2sf1(lr), phi, pQuasi: chi2sf1(lr / phi) };
}
export function predictLogit(fit, x, dispersion = true) {
  const eta = fit.b0 + fit.b1 * x, v = (fit.cov[0][0] + 2 * x * fit.cov[0][1] + x * x * fit.cov[1][1]) * (dispersion ? fit.phi : 1);
  const se = Math.sqrt(v);
  return { p: sigmoid(eta), lo: sigmoid(eta - 1.96 * se), hi: sigmoid(eta + 1.96 * se), eta, se };
}

// weighted pool-adjacent-violators for a non-decreasing rate, then order-restricted LRT vs a common rate
export function isotonic(ks, ns) {
  const blocks = ks.map((k, i) => ({ k, n: ns[i], idx: [i] }));
  for (let i = 0; i < blocks.length - 1;) {
    if (blocks[i].k / blocks[i].n > blocks[i + 1].k / blocks[i + 1].n) {
      const m = { k: blocks[i].k + blocks[i + 1].k, n: blocks[i].n + blocks[i + 1].n, idx: [...blocks[i].idx, ...blocks[i + 1].idx] };
      blocks.splice(i, 2, m); if (i > 0) i--;
    } else i++;
  }
  const fitted = new Array(ks.length); for (const b of blocks) for (const j of b.idx) fitted[j] = b.k / b.n; return fitted;
}
export function isoLRT(ks, ns) {
  const f = isotonic(ks, ns), K = ks.reduce((a, b) => a + b, 0), N = ns.reduce((a, b) => a + b, 0), p0 = K / N;
  let s = 0; ks.forEach((k, i) => { const ll = q => xlogy(k, q) + xlogy(ns[i] - k, 1 - q); s += 2 * (ll(f[i]) - ll(p0)); });
  return s;
}
// parametric bootstrap p-value for the isotonic LRT under a common rate
export function isoTest(ks, ns, reps = 2000, seed = 11) {
  const rand = mulberry32(seed), K = ks.reduce((a, b) => a + b, 0), N = ns.reduce((a, b) => a + b, 0), p0 = K / N;
  const obs = isoLRT(ks, ns); let c = 0;
  for (let r = 0; r < reps; r++) { const sk = ns.map(n => { let k = 0; for (let t = 0; t < n; t++) if (rand() < p0) k++; return k; }); if (isoLRT(sk, ns) >= obs) c++; }
  return { stat: obs, p: (1 + c) / (reps + 1), fitted: isotonic(ks, ns) };
}

// power of the logistic trend LR test (one-sided, increasing) for an allocation, by simulation
export function trendPower(truth, xs, alloc, opts = {}) {
  const R = opts.reps ?? 400, alpha = opts.alpha ?? 0.05, rand = mulberry32(opts.seed ?? 21);
  const crit = (() => { const z = invNorm(1 - alpha); return z * z; })(); // one-sided via signed root
  let hit = 0;
  for (let r = 0; r < R; r++) {
    const ks = alloc.map((n, i) => { let k = 0; for (let t = 0; t < n; t++) if (rand() < truth[i]) k++; return k; });
    const idx = alloc.map((n, i) => i).filter(i => alloc[i] > 0);
    const f = logisticFit(idx.map(i => xs[i]), idx.map(i => ks[i]), idx.map(i => alloc[i]));
    if (f.b1 > 0 && f.lr >= crit) hit++;
  }
  return hit / R;
}
export function pairPower(pa, na, pb, nb, opts = {}) {
  const R = opts.reps ?? 400, alpha = opts.alpha ?? 0.05, rand = mulberry32(opts.seed ?? 31); let hit = 0;
  for (let r = 0; r < R; r++) { let ka = 0, kb = 0; for (let t = 0; t < na; t++) if (rand() < pa) ka++; for (let t = 0; t < nb; t++) if (rand() < pb) kb++; if (fisherTwoSided(ka, na, kb, nb) <= alpha && kb / nb > ka / na) hit++; }
  return hit / R;
}
export function invNorm(p) { // Acklam
  const a = [-39.69683028665376, 220.9460984245205, -275.9285104469687, 138.357751867269, -30.66479806614716, 2.506628277459239], b = [-54.47609879822406, 161.5858368580409, -155.6989798598866, 66.80131188771972, -13.28068155288572], c = [-0.007784894002430293, -0.3223964580411365, -2.400758277161838, -2.549732539343734, 4.374664141464968, 2.938163982698783], d = [0.007784695709041462, 0.3224671290700398, 2.445134137142996, 3.754408661907416];
  const pl = 0.02425; let q, r;
  if (p < pl) { q = Math.sqrt(-2 * Math.log(p)); return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1); }
  if (p > 1 - pl) { q = Math.sqrt(-2 * Math.log(1 - p)); return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1); }
  q = p - 0.5; r = q * q; return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1);
}

// trials a new checkpoint needs to beat a reference (existing counts) with the given power, averaging power
// over the forecast uncertainty of its true rate (logit-normal from predictLogit)
export function trialsToBeat(pred, ref, opts = {}) {
  const power = opts.power ?? 0.8, alpha = opts.alpha ?? 0.05, R = opts.reps ?? 300, rand = mulberry32(opts.seed ?? 41), maxN = opts.maxN ?? 2000;
  const gauss = () => { let u = 0; while (!u) u = rand(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * rand()); };
  const draws = Array.from({ length: R }, () => sigmoid(pred.eta + pred.se * gauss()));
  // balanced: the reference also gets n fresh trials (at its observed rate) on top of its existing counts
  const pRef = ref.k / ref.n, both = opts.balanced ?? true;
  const pw = n => { let hit = 0; for (const p of draws) { let k = 0, kr = ref.k, nr = ref.n; for (let t = 0; t < n; t++) if (rand() < p) k++; if (both) { for (let t = 0; t < n; t++) if (rand() < pRef) kr++; nr += n; } if (k / n > kr / nr && fisherTwoSided(kr, nr, k, n) <= alpha) hit++; } return hit / R; };
  let lo = 10, hi = 10; while (hi <= maxN && pw(hi) < power) { lo = hi; hi *= 2; }
  if (hi > maxN) return { n: Infinity, powerAtMax: pw(maxN), meanRate: draws.reduce((a, b) => a + b, 0) / R };
  while (hi - lo > 5) { const mid = Math.round((lo + hi) / 2); if (pw(mid) >= power) hi = mid; else lo = mid; }
  return { n: hi, meanRate: draws.reduce((a, b) => a + b, 0) / R };
}

export const DINO_FD = { XS: 91.4108, S: 30.6765, M008: 35.5415, M018: 22.4426, M037: 19.3079, M1: 14.8397, L: 9.3684 };
export const COMPUTE_SHARE = { M008: 0.08, M018: 0.18, M037: 0.37, M1: 1 };
