import { describe, expect, it } from 'vitest';
import {
  FluidSolver,
  advectScalar,
  computeDivergence,
  createField,
  createGrid,
  createUField,
  createVField,
  diffuseScalar,
  idx,
  maxAbsInterior,
  project,
  rmsInterior,
  sampleU,
  setUBoundary,
  setVBoundary,
  sumInterior,
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

/** Fill face velocities from an analytic field f(x, y) → [u, v] (x, y in grid units). */
function fillVelocity(
  g: Grid, u: Float32Array, v: Float32Array, f: (x: number, y: number) => [number, number],
): void {
  for (let j = 1; j <= g.ny; j++) for (let i = 0; i <= g.nx; i++) u[uIdx(g, i, j)] = f(i, j - 0.5)[0];
  for (let j = 0; j <= g.ny; j++) for (let i = 1; i <= g.nx; i++) v[vIdx(g, i, j)] = f(i - 0.5, j)[1];
}

/** Smooth swirl tangential to the walls of an nx×ny box (divergence free). */
const swirl = (g: Grid) => (x: number, y: number): [number, number] => {
  const X = x / g.nx;
  const Y = y / g.ny;
  return [
    Math.sin(Math.PI * X) ** 2 * Math.sin(2 * Math.PI * Y),
    -Math.sin(2 * Math.PI * X) * Math.sin(Math.PI * Y) ** 2,
  ];
};

function allFinite(...fields: Float32Array[]): boolean {
  for (const f of fields) for (let k = 0; k < f.length; k++) if (!Number.isFinite(f[k])) return false;
  return true;
}

describe('grid', () => {
  it('lays out cell and face arrays with ghost layers', () => {
    const g = createGrid(64, 32);
    expect(g.size).toBe(66 * 34);
    expect(g.uSize).toBe(65 * 34);
    expect(g.vSize).toBe(66 * 33);
    expect(g.h).toBeCloseTo(1 / 32);
  });

  it('rejects invalid sizes', () => {
    expect(() => createGrid(2, 10)).toThrow();
    expect(() => createGrid(10.5, 10)).toThrow();
  });
});

describe('boundary conditions', () => {
  it('blocks normal flow through walls and mirrors the tangential component', () => {
    const g = createGrid(8, 8);
    const u = createUField(g);
    const v = createVField(g);
    u.fill(1);
    v.fill(1);
    setUBoundary(g, u, 'no-slip');
    setVBoundary(g, v, 'no-slip');
    expect(u[uIdx(g, 0, 4)]).toBe(0); // left wall
    expect(u[uIdx(g, 8, 4)]).toBe(0); // right wall
    expect(v[vIdx(g, 4, 0)]).toBe(0); // bottom wall
    // tangential velocity interpolated onto the wall vanishes (no-slip)
    expect(sampleU(g, u, 4, 0)).toBeCloseTo(0, 6);
    u.fill(1);
    setUBoundary(g, u, 'free-slip');
    expect(sampleU(g, u, 4, 0)).toBeCloseTo(1, 6);
  });
});

describe('pressure projection', () => {
  for (const walls of ['no-slip', 'free-slip'] as const) {
    it(`reduces divergence substantially (${walls})`, () => {
      const g = createGrid(64, 48);
      const u = createUField(g);
      const v = createVField(g);
      const p = createField(g);
      const tmp = createField(g);
      const div = createField(g);
      const r = rng(7);
      // smooth compressible part + random noise
      fillVelocity(g, u, v, (x, y) => [
        Math.sin((Math.PI * x) / g.nx) * Math.cos((2 * Math.PI * y) / g.ny) + 0.3 * (r() - 0.5),
        Math.cos((3 * Math.PI * x) / g.nx) * Math.sin((Math.PI * y) / g.ny) + 0.3 * (r() - 0.5),
      ]);
      setUBoundary(g, u, walls);
      setVBoundary(g, v, walls);
      computeDivergence(g, u, v, div);
      const before = rmsInterior(g, div);
      project(g, u, v, p, tmp, 1 / 60, 300, walls);
      computeDivergence(g, u, v, div);
      const after = rmsInterior(g, div);
      expect(before).toBeGreaterThan(1);
      expect(after / before).toBeLessThan(1e-3);
      expect(allFinite(u, v, p)).toBe(true);
    });
  }

  it('leaves a divergence-free field (nearly) unchanged', () => {
    const g = createGrid(48, 48);
    const u = createUField(g);
    const v = createVField(g);
    // discretely divergence-free: face velocities from a stream function at nodes
    const psi = (x: number, y: number) =>
      Math.sin((Math.PI * x) / g.nx) ** 2 * Math.sin((Math.PI * y) / g.ny) ** 2;
    for (let j = 1; j <= g.ny; j++)
      for (let i = 0; i <= g.nx; i++) u[uIdx(g, i, j)] = (psi(i, j) - psi(i, j - 1)) * g.nx;
    for (let j = 0; j <= g.ny; j++)
      for (let i = 1; i <= g.nx; i++) v[vIdx(g, i, j)] = -(psi(i, j) - psi(i - 1, j)) * g.nx;
    setUBoundary(g, u, 'free-slip');
    setVBoundary(g, v, 'free-slip');
    const u0 = u.slice();
    project(g, u, v, createField(g), createField(g), 1 / 60, 50, 'free-slip');
    let maxDiff = 0;
    for (let k = 0; k < u.length; k++) maxDiff = Math.max(maxDiff, Math.abs(u[k] - u0[k]));
    expect(maxDiff).toBeLessThan(1e-4);
  });
});

describe('advection and diffusion', () => {
  it('approximately conserves dye mass under advection with closed boundaries', () => {
    const s = new FluidSolver(64, 64, { viscosity: 0, diffusion: 0, iterations: 40 });
    const g = s.grid;
    fillVelocity(g, s.u, s.v, swirl(g));
    s.setParams({}); // re-apply wall conditions
    s.splat(32, 22, 0, 0, [1, 0.5, 0.25], 5);
    const m0 = sumInterior(g, s.dye[0]);
    for (let n = 0; n < 120; n++) s.step();
    const m1 = sumInterior(g, s.dye[0]);
    expect(m0).toBeGreaterThan(10);
    expect(Math.abs(m1 - m0) / m0).toBeLessThan(0.1);
    // the dye actually moved
    expect(s.sample(s.dye[0], 32, 22)).toBeLessThan(0.9);
  });

  it('conserves mass under pure diffusion with closed boundaries', () => {
    const g = createGrid(32, 32);
    const d0 = createField(g);
    const d = createField(g);
    for (let j = 10; j < 20; j++) for (let i = 1; i < 15; i++) d0[idx(g, i, j)] = 1;
    const m0 = sumInterior(g, d0);
    diffuseScalar(g, d, d0, 0.01, 0.1, 80);
    const m1 = sumInterior(g, d);
    expect(Math.abs(m1 - m0) / m0).toBeLessThan(0.01);
    expect(maxAbsInterior(g, d)).toBeLessThan(1); // it spread out
  });

  it('translates a field with uniform velocity (semi-Lagrangian, bilinear)', () => {
    const g = createGrid(32, 32);
    const d0 = createField(g);
    const d = createField(g);
    const u = createUField(g);
    const v = createVField(g);
    d0[idx(g, 10, 10)] = 1;
    u.fill(g.h); // one cell per second
    advectScalar(g, d, d0, u, v, 1);
    expect(d[idx(g, 11, 10)]).toBeCloseTo(1, 5);
    expect(d[idx(g, 10, 10)]).toBeCloseTo(0, 5);
    advectScalar(g, d0, d, u, v, 0.5); // half a cell: split evenly
    expect(d0[idx(g, 11, 10)]).toBeCloseTo(0.5, 5);
    expect(d0[idx(g, 12, 10)]).toBeCloseTo(0.5, 5);
  });
});

describe('FluidSolver', () => {
  it('keeps a zero field at zero', () => {
    const s = new FluidSolver(32, 24, { vorticity: 5, viscosity: 0.001, diffusion: 0.001 });
    for (let n = 0; n < 50; n++) s.step();
    s.updateDiagnostics();
    for (const f of [s.u, s.v, s.pressure, s.divergence, s.vorticity, ...s.dye]) {
      expect(f.every((x) => x === 0)).toBe(true);
    }
  });

  it('produces no NaNs after many steps with strong random forcing', () => {
    const s = new FluidSolver(48, 32, {
      viscosity: 0.0001, diffusion: 0.00001, vorticity: 20, dt: 0.05, iterations: 10,
      dyeDissipation: 0.1,
    });
    const r = rng(42);
    for (let n = 0; n < 500; n++) {
      if (n % 3 === 0) s.splat(r() * 48, r() * 32, (r() - 0.5) * 50, (r() - 0.5) * 50, [r(), r(), r()], 3);
      s.step();
    }
    s.updateDiagnostics();
    expect(allFinite(s.u, s.v, s.pressure, s.divergence, s.vorticity, ...s.dye)).toBe(true);
    expect(maxAbsInterior(s.grid, s.uc)).toBeLessThan(100); // stays bounded
  });

  it('keeps the velocity field nearly divergence-free after a step', () => {
    const s = new FluidSolver(64, 64, { iterations: 80 });
    s.splat(32, 32, 3, 1, [1, 1, 1], 5);
    s.updateDiagnostics();
    const before = rmsInterior(s.grid, s.divergence);
    s.step();
    s.updateDiagnostics();
    const after = rmsInterior(s.grid, s.divergence);
    expect(before).toBeGreaterThan(0);
    expect(after / before).toBeLessThan(0.02);
  });

  it('does not let dye leak through the walls', () => {
    const s = new FluidSolver(32, 32, { viscosity: 0, diffusion: 0, iterations: 40 });
    s.splat(5, 16, -3, 0, [1, 0, 0], 3); // push dye into the left wall
    const m0 = sumInterior(s.grid, s.dye[0]);
    for (let n = 0; n < 60; n++) s.step();
    const m1 = sumInterior(s.grid, s.dye[0]);
    expect(Math.abs(m1 / m0 - 1)).toBeLessThan(0.1);
  });

  it('viscosity dissipates kinetic energy', () => {
    const run = (nu: number) => {
      const s = new FluidSolver(48, 48, { viscosity: nu });
      fillVelocity(s.grid, s.u, s.v, swirl(s.grid));
      s.setParams({});
      for (let n = 0; n < 30; n++) s.step();
      s.updateDiagnostics();
      return s.kineticEnergy();
    };
    expect(run(0.01)).toBeLessThan(run(0));
  });

  it('resets all fields', () => {
    const s = new FluidSolver(16, 16);
    s.splat(8, 8, 1, 1, [1, 1, 1], 2);
    s.setParams({ walls: 'free-slip' });
    s.step();
    s.reset();
    expect(sumInterior(s.grid, s.dye[0])).toBe(0);
    expect(s.u.every((x) => x === 0)).toBe(true);
    expect(s.steps).toBe(0);
  });
});

describe('interaction helpers', () => {
  it('stir blends velocity toward the target without overshooting', () => {
    const s = new FluidSolver(32, 32);
    for (let n = 0; n < 50; n++) s.stir(16, 16, 2, -1, 3, 0.6);
    s.updateDiagnostics();
    expect(s.sample(s.uc, 16, 16)).toBeCloseTo(2, 2);
    expect(s.sample(s.vc, 16, 16)).toBeCloseTo(-1, 2);
    expect(maxAbsInterior(s.grid, s.uc)).toBeLessThanOrEqual(2.0001);
  });

  it('resamples state onto a different grid', () => {
    const a = new FluidSolver(32, 32);
    a.splat(16, 16, 0, 0, [1, 0, 0], 4);
    const b = new FluidSolver(64, 64);
    b.resampleFrom(a);
    const ma = sumInterior(a.grid, a.dye[0]) * a.grid.h ** 2;
    const mb = sumInterior(b.grid, b.dye[0]) * b.grid.h ** 2;
    expect(Math.abs(mb / ma - 1)).toBeLessThan(0.02);
    expect(b.sample(b.dye[0], 32, 32)).toBeGreaterThan(0.9);
  });
});
