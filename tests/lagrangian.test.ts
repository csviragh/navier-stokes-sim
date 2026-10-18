import { describe, expect, it } from 'vitest';
import { createGrid, FluidSolver, PathTracer, rightmostRoot, uIdx, vIdx, type KernelType, type WallType } from '../src/solver';
import { delayRoot } from './lambert';

/**
 * Uniformly translating flock u = U (exactly one cell per step, so the
 * semi-Lagrangian advection is an exact shift) with a small transverse wave
 * v = ε sin(2π m (x − ½h)/L). Returns the comoving projection of v onto the
 * advected profile and its log-modulus.
 */
function translating(mode: 'eulerian' | 'lagrangian', tau: number, steps: number, N = 48, m = 6, S = 2, U = 1.25) {
  const s = new FluidSolver(N, N, {
    walls: 'periodic', viscosity: 0, vorticity: 0, kernel: 'cucker-smale', kernelStrength: S, kernelRange: 0.05,
    delaySelf: tau, delayNb: tau, delayMode: mode,
  });
  const g = s.grid, dt = s.params.dt;
  for (let j = 0; j < g.ny + 2; j++) for (let i = 0; i <= g.nx; i++) s.u[uIdx(g, i, j)] = U;
  for (let j = 0; j <= g.ny; j++) for (let i = 0; i < g.nx + 2; i++) s.v[vIdx(g, i, j)] = 1e-3 * Math.sin((2 * Math.PI * m * (i - 0.5)) / N);
  const A: number[] = [], L: number[] = [];
  const shift = (U / g.h) * dt; // cells per step
  for (let n = 1; n <= steps; n++) {
    s.step();
    let a = 0, cr = 0, ci = 0;
    for (let i = 1; i <= g.nx; i++) {
      const ph = (2 * Math.PI * m * (i - 0.5 - n * shift)) / N, x = s.v[vIdx(g, i, 5)];
      a += x * Math.sin(ph); cr += x * Math.cos(ph); ci += x * Math.sin(ph);
    }
    A.push(a / (N / 2)); L.push(Math.log(Math.hypot(cr, ci)));
  }
  return { s, A, L, Kh: s.kernelOperators().v.symbol(m, 0), S, dt };
}
const slope = (L: number[], dt: number, tMin: number) => {
  let sx = 0, sy = 0, sxx = 0, sxy = 0, n = 0;
  L.forEach((y, k) => { const x = (k + 1) * dt; if (x < tMin) return; sx += x; sy += y; sxx += x * x; sxy += x * y; n++; });
  return (n * sxy - sx * sy) / (n * sxx - sx * sx);
};
function fitOsc(A: number[], dt: number, tMin: number) {
  const zc: number[] = [], pk: [number, number][] = [];
  for (let n = 1; n < A.length - 1; n++) {
    const t = (n + 1) * dt; if (t < tMin) continue;
    if (A[n - 1] * A[n] < 0) zc.push((n - 1 + A[n - 1] / (A[n - 1] - A[n])) * dt);
    if (Math.abs(A[n]) >= Math.abs(A[n - 1]) && Math.abs(A[n]) > Math.abs(A[n + 1])) pk.push([t, Math.log(Math.abs(A[n]))]);
  }
  let sx = 0, sy = 0, sxx = 0, sxy = 0; for (const [x, y] of pk) { sx += x; sy += y; sxx += x * x; sxy += x * y; }
  const k = pk.length;
  return { mu: (k * sxy - sx * sy) / (k * sxx - sx * sx), omega: Math.PI * (zc.length - 1) / (zc[zc.length - 1] - zc[0]) };
}

describe('Lagrangian (comoving) delay', () => {
  it('translating flock recovers the rest-fluid Lambert-W roots (no Doppler term)', () => {
    for (const tau of [0.3, 0.9]) {
      const { A, L, Kh, S, dt } = translating('lagrangian', tau, 600);
      const a = S * (1 - Kh);
      const th = delayRoot(a, tau);
      const rr = rightmostRoot(S, S * Kh, tau, tau);
      expect(Math.abs(rr.re - th.re)).toBeLessThan(1e-6);
      const f = fitOsc(A, dt, 1.5);
      expect(Math.abs(f.mu - th.re)).toBeLessThan(0.03);
      expect(Math.abs(f.omega - th.im)).toBeLessThan(0.02);
      expect(Math.abs(slope(L, dt, 2) - th.re)).toBeLessThan(0.04);
    }
  }, 60000);

  it('below 1/(e·S_eff) the comoving mode decays monotonically at the real Lambert-W rate', () => {
    const tau = 0.15;
    const { A, L, Kh, S, dt } = translating('lagrangian', tau, 240);
    const th = delayRoot(S * (1 - Kh), tau);
    expect(th.im).toBe(0);
    expect(A.every((x) => x > 0)).toBe(true);
    expect(Math.abs(slope(L, dt, 1) - th.re)).toBeLessThan(0.03);
  }, 30000);

  it('instability threshold S_eff·t = π/2 holds in the comoving frame', () => {
    const { Kh, S } = translating('lagrangian', 0.5, 1);
    const tc = Math.PI / (2 * S * (1 - Kh));
    const below = translating('lagrangian', tc - 0.05, 600), above = translating('lagrangian', tc + 0.05, 600);
    expect(slope(below.L, below.dt, 2)).toBeLessThan(-0.02);
    expect(slope(above.L, above.dt, 2)).toBeGreaterThan(0.02);
  }, 60000);

  it('the Eulerian (fixed-point) read is Doppler-destabilised for the same flock', () => {
    const tau = 0.3;
    const { L, Kh, S, dt } = translating('eulerian', tau, 600);
    expect(delayRoot(S * (1 - Kh), tau).re).toBeLessThan(-2);
    expect(slope(L, dt, 2)).toBeGreaterThan(0.1);
  }, 30000);

  it('at rest (linear regime) Lagrangian and Eulerian reads agree', () => {
    const run = (mode: 'eulerian' | 'lagrangian') => {
      const s = new FluidSolver(40, 40, { walls: 'periodic', viscosity: 0, vorticity: 0, kernel: 'cucker-smale', kernelStrength: 2, delaySelf: 0.4, delayNb: 0.2, delayMode: mode });
      for (let k = 0; k < s.u.length; k++) s.u[k] = 1e-9 * Math.sin(0.37 * k);
      for (let k = 0; k < s.v.length; k++) s.v[k] = 1e-9 * Math.cos(0.21 * k);
      for (let n = 0; n < 120; n++) s.step();
      return s;
    };
    const e = run('eulerian'), l = run('lagrangian');
    let d = 0, m = 0;
    for (let k = 0; k < e.u.length; k++) { d = Math.max(d, Math.abs(e.u[k] - l.u[k])); m = Math.max(m, Math.abs(e.u[k])); }
    expect(d / m).toBeLessThan(1e-4);
  }, 30000);

  it('a uniform flow is unchanged', () => {
    const s = new FluidSolver(32, 24, { walls: 'periodic', viscosity: 0.001, vorticity: 0, kernel: 'cucker-smale', kernelStrength: 3, delaySelf: 0.3, delayNb: 0.5, delayLocal: true, delayMode: 'lagrangian' });
    for (let k = 0; k < s.u.length; k++) s.u[k] = 0.4;
    for (let k = 0; k < s.v.length; k++) s.v[k] = -0.2;
    for (let n = 0; n < 90; n++) s.step();
    for (let j = 1; j <= s.grid.ny; j++) for (let i = 1; i <= s.grid.nx; i++) {
      expect(Math.abs(s.u[uIdx(s.grid, i, j)] - 0.4)).toBeLessThan(1e-5);
      expect(Math.abs(s.v[vIdx(s.grid, i, j)] + 0.2)).toBeLessThan(1e-5);
    }
  }, 30000);

  it('incremental departure maps (with τ/2 rebuilds) agree with a full RK2 back-trace', () => {
    const N = 40, g = createGrid(N, N), dt = 1 / 60, tau = 0.5, A = 0.3, P = 2 * Math.PI / N, Q = Math.PI / N;
    for (const walls of ['periodic', 'no-slip'] as WallType[]) {
      const fu = new Float64Array(g.uSize), fv = new Float64Array(g.vSize);
      // steady cellular flow: periodic cells on the torus, one box vortex (no normal flow) with walls
      for (let j = 0; j < g.ny + 2; j++) for (let i = 0; i <= g.nx; i++)
        fu[uIdx(g, i, j)] = walls === 'periodic' ? A * Math.sin(P * (j - 0.5)) * Math.cos(P * i) : A * Math.sin(Q * i) * Math.cos(Q * (j - 0.5));
      for (let j = 0; j <= g.ny; j++) for (let i = 0; i < g.nx + 2; i++)
        fv[vIdx(g, i, j)] = walls === 'periodic' ? -A * Math.sin(P * (i - 0.5)) * Math.cos(P * j) : -A * Math.cos(Q * (i - 0.5)) * Math.sin(Q * j);
      const read = (_T: number, oU: Float64Array, oV: Float64Array) => { oU.set(fu); oV.set(fv); };
      const tr = new PathTracer();
      const oU = new Float64Array(4 * g.uSize), oV = new Float64Array(4 * g.vSize);
      const steps = 110; // 20 incremental updates after the last rebuild (every τ/2 = 15 steps)
      for (let n = 0; n <= steps; n++) tr.departures(g, walls, tau, n * dt, dt, fu, fv, read, oU, oV);
      expect(tr.inits).toBe(8);
      const inc = tr.displacement(tau)!, full = tr.fullTrace(g, walls, tau, steps * dt, dt / 10, read);
      let d = 0, m = 0;
      for (let k = 0; k < inc.v.length; k++) { d = Math.max(d, Math.abs(inc.v[k] - full.v[k])); m = Math.max(m, Math.abs(full.v[k])); }
      for (let k = 0; k < inc.u.length; k++) d = Math.max(d, Math.abs(inc.u[k] - full.u[k]));
      expect(m).toBeGreaterThan(2); // parcels travel several cells
      expect(d).toBeLessThan(0.05);
    }
  });

  it('stays finite on every boundary for local + kernel delays with t_self ≠ t_nb; reset clears the maps', () => {
    const walls: WallType[] = ['no-slip', 'free-slip', 'periodic', 'periodic-x'];
    for (const w of walls) for (const kernel of ['cucker-smale', 'fractional'] as KernelType[]) {
      if (kernel === 'fractional' && w !== 'periodic') continue;
      const s = new FluidSolver(36, 28, {
        walls: w, viscosity: 0.002, kernel, kernelStrength: 3, delaySelf: 0.25, delayNb: 0.6, delayLocal: true, delayKernel: true,
        selfPropulsion: true, delayMode: 'lagrangian',
      });
      for (let k = 0; k < s.u.length; k++) s.u[k] = 0.5 * Math.sin(0.37 * k);
      for (let k = 0; k < s.v.length; k++) s.v[k] = 0.5 * Math.cos(0.21 * k);
      for (let n = 0; n < 150; n++) s.step();
      expect(s.u.every(Number.isFinite) && s.v.every(Number.isFinite)).toBe(true);
      expect(s.tracer.displacement(0.6)).not.toBeNull();
      s.clearDelayHistory();
      expect(s.tracer.displacement(0.6)).toBeNull();
    }
  }, 60000);
});
