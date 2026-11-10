import test from 'node:test';
import assert from 'node:assert/strict';
import * as E from '../core/evalplan.js';
const by = Object.fromEntries(E.PUBLISHED.policies.map(p => [p.id, p]));
const fig = k => E.PUBLISHED.figures[k].ids.map(i => by[i]);

test('Beta credible intervals reproduce the published 95% intervals', () => {
  const pub = { M1: [0.697, 0.801], XS: [0.009, 0.183], S: [0.554, 0.736], L: [0.768, 0.902], M008: [0.481, 0.670] };
  for (const [id, [lo, hi]] of Object.entries(pub)) {
    const [a, b] = E.credible(by[id].k, by[id].n);
    assert.ok(Math.abs(a - lo) < 0.0015 && Math.abs(b - hi) < 0.0015, `${id} ${a} ${b}`);
  }
});

test('Fisher + Holm letters reproduce the published Figure 3 and Figure 4 letters', () => {
  const f4 = E.analyzeCounts(fig('size'));
  assert.deepEqual(Object.values(f4.lettersFisher), ['a', 'b', 'bc', 'c']);
  const f3 = E.analyzeCounts([by.S, by.L, by.M1]);
  assert.deepEqual(Object.values(f3.lettersFisher), ['a', 'b', 'ab']);
});

test("Fisher exact matches the post's worked examples (75% vs 65% and 75% vs 60% on 100 trials)", () => {
  assert.ok(Math.abs(E.fisherTwoSided(75, 100, 65, 100) - 0.16) < 0.01);
  assert.ok(Math.abs(E.fisherTwoSided(75, 100, 60, 100) - 0.03) < 0.01);
});

test('logistic trend fit: closed-form check on a two-group design equals the log odds ratio', () => {
  const f = E.logisticFit([0, 1], [30, 60], [100, 100]);
  const lor = Math.log((60 / 40) / (30 / 70));
  assert.ok(Math.abs(f.b1 - lor) < 1e-6);
  assert.ok(f.p < 1e-4);
});

test('the compute trend is detected even though most compute pairs are not separated', () => {
  const pol = fig('compute');
  const an = E.analyzeCounts(pol);
  assert.ok(an.fisherReject.filter(Boolean).length <= 2);
  const xs = E.PUBLISHED.figures.compute.ids.map(i => Math.log2(E.COMPUTE_SHARE[i]));
  const f = E.logisticFit(xs, pol.map(p => p.k), pol.map(p => p.n));
  assert.ok(f.b1 > 0 && f.p < 0.01, 'trend p ' + f.p);
  const iso = E.isoTest(pol.map(p => p.k), pol.map(p => p.n), 999);
  assert.ok(iso.p < 0.01);
  for (let i = 1; i < 4; i++) assert.ok(iso.fitted[i] >= iso.fitted[i - 1]);
});

test('isotonic PAVA pools violators', () => {
  assert.deepEqual(E.isotonic([50, 40, 80], [100, 100, 100]).map(x => +x.toFixed(3)), [0.45, 0.45, 0.8]);
});

test('e-process is valid under peeking while repeated Fisher + Holm is not', () => {
  const r = E.nullCheck({ reps: 300 });
  assert.ok(r.anytimeE <= 0.05 + 0.02, 'anytime ' + r.anytimeE);
  assert.ok(r.peekingFisher > 0.08, 'peeking ' + r.peekingFisher);
  assert.ok(r.fixedFisher <= 0.05 + 0.03);
});

test('trend LR test holds its level under a flat truth', () => {
  const xs = [-3.64, -2.47, -1.43, 0];
  const pw = E.trendPower([0.7, 0.7, 0.7, 0.7], xs, [100, 100, 100, 100], { reps: 600 });
  assert.ok(pw <= 0.05 + 0.025, 'size ' + pw);
});

test("two-endpoint allocation reaches the published design's trend power with fewer trials", () => {
  const ids = E.PUBLISHED.figures.compute.ids, xs = ids.map(i => Math.log2(E.COMPUTE_SHARE[i]));
  const truth = ids.map(i => by[i].k / by[i].n);
  const pub = E.trendPower(truth, xs, ids.map(i => by[i].n), { reps: 400 });
  const ends = E.trendPower(truth, xs, [150, 0, 0, 150], { reps: 400 });
  assert.ok(ends >= pub - 0.03, `ends ${ends} pub ${pub}`);
});

test('forecast from DINO FD is monotone and trials-to-beat shrinks as the forecast improves', () => {
  const ids = Object.keys(E.DINO_FD);
  const f = E.logisticFit(ids.map(i => -Math.log(E.DINO_FD[i])), ids.map(i => by[i].k), ids.map(i => by[i].n));
  const a = E.predictLogit(f, -Math.log(8)), b = E.predictLogit(f, -Math.log(5));
  assert.ok(b.p > a.p && f.phi >= 1);
  const ta = E.trialsToBeat(E.predictLogit(f, -Math.log(6)), by.L, { reps: 150 });
  const tb = E.trialsToBeat(E.predictLogit(f, -Math.log(4.5)), by.L, { reps: 150 });
  assert.ok(tb.n < ta.n);
});

test('letters: no separations -> one group; all separated -> distinct', () => {
  assert.deepEqual(Object.values(E.letters([0, 1, 2], () => false)), ['a', 'a', 'a']);
  assert.deepEqual(Object.values(E.letters([0, 1, 2], () => true)), ['a', 'b', 'c']);
});
