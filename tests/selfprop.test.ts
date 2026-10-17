import { describe, expect, it } from 'vitest';
import {
  FluidSolver,
  SELF_PROPULSION_EPS,
  createGrid,
  createUField,
  createVField,
  maxAbsInterior,
  rmsInterior,
  selfPropulsion,
  setUBoundary,
  setVBoundary,
  uIdx,
  vIdx,
  type Grid,
} from '../src/solver';

/** Deterministic pseudo-random generator (mulberry32). */
function rng(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function allFinite(...fields: Float32Array[]): boolean {
  for (const f of fields) for (let k = 0; k < f.length; k++) if (!Number.isFinite(f[k])) return false;
  return true;
}

/** Uniform velocity (U, V) on all interior faces, walls applied. */
function uniform(g: Grid, U: number, V: number) {
  const u = createUField(g);
  const v = createVField(g);
  for (let j = 1; j <= g.ny; j++) for (let i = 1; i < g.nx; i++) u[uIdx(g, i, j)] = U;
  for (let j = 1; j < g.ny; j++) for (let i = 1; i <= g.nx; i++) v[vIdx(g, i, j)] = V;
  setUBoundary(g, u, 'free-slip');
  setVBoundary(g, v, 'free-slip');
  return { u, v, u0: createUField(g), v0: createVField(g) };
}

/** Analytic speed of d|v|/dt = −(|v| − v0)/τ. */
const exact = (s0: number, v0: number, tau: number, t: number) => v0 + (s0 - v0) * Math.exp(-t / tau);

describe('self-propulsion kernel: −(1/τ)(|v| − v₀) v/|v|', () => {
  const tau = 0.4;
  const v0 = 0.8;
  const dt = 1 / 60;

  for (const s0 of [0.1, 2.0]) {
    it(`relaxes a uniform axial flow from |v| = ${s0} toward v₀ at the exp(−t/τ) rate`, () => {
      const g = createGrid(24, 16);
      const f = uniform(g, s0, 0);
      for (let n = 1; n <= 120; n++) {
        selfPropulsion(g, f.u, f.v, f.u0, f.v0, tau, v0, dt, 'free-slip');
        if (n % 30 === 0) {
          const want = exact(s0, v0, tau, n * dt);
          // every interior u-face follows the analytic solution, v stays 0
          for (let j = 1; j <= g.ny; j++)
            for (let i = 1; i < g.nx; i++) expect(f.u[uIdx(g, i, j)]).toBeCloseTo(want, 5);
          expect(f.v.every((x) => x === 0)).toBe(true);
        }
      }
      // monotone approach: from below it accelerates, from above it decelerates
      const final = f.u[uIdx(g, 5, 5)];
      if (s0 < v0) expect(final).toBeGreaterThan(s0);
      else expect(final).toBeLessThan(s0);
      expect(Math.abs(final - v0)).toBeLessThan(Math.abs(s0 - v0) * Math.exp(-2 / tau) * 1.01 + 1e-6);
    });
  }

  it('preserves the flow direction of an oblique uniform flow and follows exp(−t/τ)', () => {
    const g = createGrid(64, 64);
    const [U, V] = [1.2, -0.9]; // |v| = 1.5 > v0
    const f = uniform(g, U, V);
    const steps = 20;
    for (let n = 0; n < steps; n++) selfPropulsion(g, f.u, f.v, f.u0, f.v0, tau, v0, dt, 'free-slip');
    const want = exact(Math.hypot(U, V), v0, tau, steps * dt);
    // wall effects travel one cell per step; check faces further inside
    for (let j = steps + 2; j < g.ny - steps - 2; j++)
      for (let i = steps + 2; i < g.nx - steps - 2; i++) {
        const uf = f.u[uIdx(g, i, j)];
        const vf = f.v[vIdx(g, i, j)];
        expect(uf / want).toBeCloseTo(U / 1.5, 5);
        expect(vf / want).toBeCloseTo(V / 1.5, 5);
      }
  });

  it('is exact (no overshoot) even for Δt ≫ τ', () => {
    const g = createGrid(16, 16);
    const f = uniform(g, 3, 0);
    selfPropulsion(g, f.u, f.v, f.u0, f.v0, 1e-3, v0, 1, 'free-slip');
    expect(f.u[uIdx(g, 8, 8)]).toBeCloseTo(v0, 6);
    const h = uniform(g, 0.01, 0);
    selfPropulsion(g, h.u, h.v, h.u0, h.v0, 1e-3, v0, 1, 'free-slip');
    expect(h.u[uIdx(g, 8, 8)]).toBeCloseTo(v0, 6);
  });

  it('leaves a zero field (|v| < ε) exactly zero', () => {
    const g = createGrid(16, 16);
    const f = uniform(g, 0, 0);
    f.u[uIdx(g, 8, 8)] = SELF_PROPULSION_EPS / 10; // below ε: direction undefined → untouched
    selfPropulsion(g, f.u, f.v, f.u0, f.v0, tau, v0, dt, 'no-slip');
    expect(f.u[uIdx(g, 8, 8)]).toBe(Math.fround(SELF_PROPULSION_EPS / 10));
    f.u[uIdx(g, 8, 8)] = 0;
    selfPropulsion(g, f.u, f.v, f.u0, f.v0, tau, v0, dt, 'no-slip');
    expect(allFinite(f.u, f.v)).toBe(true);
    expect(f.u.every((x) => x === 0) && f.v.every((x) => x === 0)).toBe(true);
  });
});

describe('FluidSolver with self-propulsion', () => {
  it('keeps a zero field at zero (no NaNs) with the term on', () => {
    const s = new FluidSolver(32, 24, { selfPropulsion: true, selfPropTau: 0.2, selfPropV0: 1, vorticity: 3 });
    for (let n = 0; n < 50; n++) s.step();
    s.updateDiagnostics();
    for (const f of [s.u, s.v, s.pressure, s.divergence, s.vorticity, ...s.dye]) {
      expect(f.every((x) => x === 0)).toBe(true);
    }
  });

  it('produces no NaNs and stays bounded after many steps with stirring', () => {
    const s = new FluidSolver(48, 32, {
      selfPropulsion: true, selfPropTau: 0.1, selfPropV0: 1.5,
      viscosity: 1e-4, vorticity: 10, dt: 0.05, iterations: 10, dyeDissipation: 0.1,
    });
    const r = rng(7);
    for (let n = 0; n < 500; n++) {
      if (n % 3 === 0) s.stir(r() * 48, r() * 32, (r() - 0.5) * 8, (r() - 0.5) * 8, 3, 0.6);
      if (n % 5 === 0) s.splat(r() * 48, r() * 32, (r() - 0.5) * 20, (r() - 0.5) * 20, [r(), r(), r()], 3);
      s.step();
    }
    s.updateDiagnostics();
    expect(allFinite(s.u, s.v, s.pressure, s.divergence, s.vorticity, ...s.dye)).toBe(true);
    expect(maxAbsInterior(s.grid, s.uc)).toBeLessThan(20);
  });

  it('drives weak random flow toward |u| ≈ v₀ while staying divergence-free', () => {
    const v0 = 0.5;
    const s = new FluidSolver(64, 64, {
      selfPropulsion: true, selfPropTau: 0.25, selfPropV0: v0, viscosity: 1e-5, iterations: 40,
    });
    s.seedActiveTurbulence(3, 0.02);
    s.updateDiagnostics();
    const speedRms = () => {
      const g = s.grid;
      const sp = new Float32Array(g.size);
      for (let k = 0; k < sp.length; k++) sp[k] = Math.hypot(s.uc[k], s.vc[k]);
      return rmsInterior(g, sp);
    };
    const before = speedRms();
    for (let n = 0; n < 180; n++) s.step(); // 3 s = 12 τ
    s.updateDiagnostics();
    const after = speedRms();
    expect(before).toBeLessThan(0.1 * v0);
    // projection + walls prevent |u| = v₀ exactly, but the rms speed approaches it
    expect(after).toBeGreaterThan(0.5 * v0);
    expect(after).toBeLessThan(1.5 * v0);
    expect(rmsInterior(s.grid, s.divergence)).toBeLessThan(0.05 * after / s.grid.h);
  });

  it('disabled: bit-identical regardless of τ and v₀, and τ = ∞ is a no-op', () => {
    const run = (p: Record<string, unknown>) => {
      const s = new FluidSolver(40, 28, { vorticity: 2, viscosity: 1e-4, dyeDissipation: 0.1, ...p });
      s.splat(10, 12, 1.5, 0.3, [1, 0.5, 0.2], 3);
      s.splat(30, 18, -1.2, -0.4, [0.2, 0.5, 1], 3);
      for (let n = 0; n < 40; n++) {
        if (n % 10 === 0) s.stir(20, 14, 0.8, 0.5, 3, 0.6);
        s.step();
      }
      return [s.u, s.v, s.pressure, ...s.dye];
    };
    const ref = run({});
    const same = (a: Float32Array[], b: Float32Array[]) =>
      a.every((f, n) => f.length === b[n].length && f.every((x, k) => Object.is(x, b[n][k])));
    expect(same(ref, run({ selfPropulsion: false, selfPropTau: 0.01, selfPropV0: 2 }))).toBe(true);
    expect(same(ref, run({ selfPropulsion: true, selfPropTau: Infinity, selfPropV0: 1 }))).toBe(true);
    expect(same(ref, run({ selfPropulsion: true, selfPropTau: 0.5, selfPropV0: 1 }))).toBe(false);
  });
});
