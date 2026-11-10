// Prints every headline number the page shows, from the published counts. node scripts/report.mjs
import * as E from '../core/evalplan.js';
const by = Object.fromEntries(E.PUBLISHED.policies.map(p => [p.id, p]));
for (const [k, f] of Object.entries(E.PUBLISHED.figures)) {
  const pol = f.ids.map(i => by[i]), an = E.analyzeCounts(pol);
  const xs = k === 'compute' ? f.ids.map(i => Math.log2(E.COMPUTE_SHARE[i])) : [0, 1, 2, 3];
  const fit = E.logisticFit(xs, pol.map(p => p.k), pol.map(p => p.n)), iso = E.isoTest(pol.map(p => p.k), pol.map(p => p.n), 2000);
  console.log(k, 'letters', pol.map((p, i) => `${p.label}:${an.lettersFisher[i]}`).join(' '), '| trend p', fit.p.toPrecision(2), '| isotonic p', iso.p.toPrecision(2), '| e-value separations', an.eReject.filter(Boolean).length, 'vs Fisher', an.fisherReject.filter(Boolean).length);
}
const ids = Object.keys(E.DINO_FD), fd = E.logisticFit(ids.map(i => -Math.log(E.DINO_FD[i])), ids.map(i => by[i].k), ids.map(i => by[i].n));
console.log('FD fit phi', fd.phi.toFixed(2), 'LR', fd.lr.toFixed(1));
const c = E.PUBLISHED.figures.compute.ids, xs = c.map(i => Math.log2(E.COMPUTE_SHARE[i])), truth = c.map(i => by[i].k / by[i].n);
for (const a of [[102, 101, 149, 259], [100, 100, 100, 100], [75, 75, 75, 75], [120, 0, 60, 120], [150, 0, 0, 150]]) console.log('alloc', a.join('/'), 'power', E.trendPower(truth, xs, a, { reps: 400 }));
console.log(E.nullCheck({ reps: 300 }));
