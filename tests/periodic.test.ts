import { describe, expect, it } from 'vitest';
import {
  FluidSolver,
  advectScalar,
  computeDivergence,
  createField,
  createGrid,
  createUField,
  createVField,
  idx,
  maxAbsInterior,
  project,
  projectPeriodic,
  setScalarBoundary,
  setUBoundary,
  setVBoundary,
  sumInterior,
  uIdx,
  vIdx,
  type WallType,
} from '../src/solver';
import { FFT } from '../src/solver/fft';
import { wallScenario } from './fingerprint';

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

/** Set every distinct face of a periodic solver to the uniform velocity (U, V). */
function setUniform(s: FluidSolver, U: number, V: number): void {
  const g = s.grid;
  for (let j = 1; j <= g.ny; j++) for (let i = 0; i <= g.nx; i++) s.u[uIdx(g, i, j)] = U;
  for (let j = 0; j <= g.ny; j++) for (let i = 1; i <= g.nx; i++) s.v[vIdx(g, i, j)] = V;
  s.setParams({});
}

/** Max |u − U|, |v − V| over all distinct faces. */
function maxDeviation(s: FluidSolver, U: number, V: number): number {
  const g = s.grid;
  let m = 0;
  for (let j = 1; j <= g.ny; j++) for (let i = 0; i <= g.nx; i++) m = Math.max(m, Math.abs(s.u[uIdx(g, i, j)] - U));
  for (let j = 0; j <= g.ny; j++) for (let i = 1; i <= g.nx; i++) m = Math.max(m, Math.abs(s.v[vIdx(g, i, j)] - V));
  return m;
}

/** Random stirring with strokes that cross the domain edges. */
function stirRandomly(s: FluidSolver, rand: () => number, steps: number): void {
  const { nx, ny } = s.grid;
  for (let n = 0; n < steps; n++) {
    const edge = n % 3 === 0;
    const X = edge ? (rand() < 0.5 ? 0.2 * rand() : nx - 0.2 * rand()) : rand() * nx;
    const Y = edge ? rand() * ny : rand() < 0.5 ? 0.3 * rand() : ny - 0.3 * rand();
    s.stir(X, Y, (rand() - 0.5) * 6, (rand() - 0.5) * 6, 3, 0.7);
    s.splat(X, Y, (rand() - 0.5) * 3, (rand() - 0.5) * 3, [rand(), rand(), rand()], 3);
    s.step();
  }
}

describe('periodic boundary conditions: ghost layers', () => {
  it('wraps cell ghosts and makes the seam faces coincide', () => {
    const g = createGrid(8, 6);
    const rand = rng(3);
    const p = createField(g);
    const u = createUField(g);
    const v = createVField(g);
    for (let k = 0; k < p.length; k++) p[k] = rand();
    for (let k = 0; k < u.length; k++) u[k] = rand();
    for (let k = 0; k < v.length; k++) v[k] = rand();
    setScalarBoundary(g, p, 'periodic');
    setUBoundary(g, u, 'periodic');
    setVBoundary(g, v, 'periodic');
    for (let j = 1; j <= g.ny; j++) {
      expect(p[idx(g, 0, j)]).toBe(p[idx(g, g.nx, j)]);
      expect(p[idx(g, g.nx + 1, j)]).toBe(p[idx(g, 1, j)]);
      expect(u[uIdx(g, 0, j)]).toBe(u[uIdx(g, g.nx, j)]);
    }
    for (let i = 0; i <= g.nx + 1; i++) {
      expect(p[idx(g, i, 0)]).toBe(p[idx(g, i, g.ny)]);
      expect(p[idx(g, i, g.ny + 1)]).toBe(p[idx(g, i, 1)]);
    }
    for (let i = 1; i <= g.nx; i++) expect(v[vIdx(g, i, 0)]).toBe(v[vIdx(g, i, g.ny)]);
    // channel: x wraps, y walls (no-slip)
    setUBoundary(g, u, 'periodic-x');
    setVBoundary(g, v, 'periodic-x');
    for (let i = 1; i <= g.nx; i++) {
      expect(v[vIdx(g, i, 0)]).toBe(0);
      expect(v[vIdx(g, i, g.ny)]).toBe(0);
    }
    for (let i = 0; i <= g.nx; i++) expect(u[uIdx(g, i, 0)]).toBe(-u[uIdx(g, i, 1)]);
  });
});

describe('periodic pressure solve', () => {
  it('FFT matches a naive DFT for power-of-two and Bluestein lengths', () => {
    for (const n of [1, 2, 3, 8, 41, 64, 205, 228]) {
      const f = new FFT(n);
      const xr = Array.from({ length: n }, (_, k) => Math.sin(1.3 * k) + (k % 3));
      const xi = Array.from({ length: n }, (_, k) => Math.cos(0.7 * k));
      const re = Float64Array.from(xr);
      const im = Float64Array.from(xi);
      f.forward(re, im);
      let err = 0;
      for (let k = 0; k < n; k++) {
        let sr = 0;
        let si = 0;
        for (let j = 0; j < n; j++) {
          const a = (-2 * Math.PI * j * k) / n;
          sr += xr[j] * Math.cos(a) - xi[j] * Math.sin(a);
          si += xr[j] * Math.sin(a) + xi[j] * Math.cos(a);
        }
        err = Math.max(err, Math.hypot(sr - re[k], si - im[k]));
      }
      expect(err).toBeLessThan(1e-9 * n);
      f.inverse(re, im);
      for (let k = 0; k < n; k++) {
        expect(re[k]).toBeCloseTo(xr[k], 12);
        expect(im[k]).toBeCloseTo(xi[k], 12);
      }
    }
  });

  for (const walls of ['periodic', 'periodic-x'] as WallType[]) {
    for (const [nx, ny] of [[37, 24], [32, 21]]) {
      it(`direct (FFT) and wrap-around SOR give the same projection (${walls}, ${nx}×${ny})`, () => {
        const g = createGrid(nx, ny);
        const rand = rng(nx * ny);
        const u = createUField(g);
        const v = createVField(g);
        for (let k = 0; k < u.length; k++) u[k] = rand() - 0.5;
        for (let k = 0; k < v.length; k++) v[k] = rand() - 0.5;
        setUBoundary(g, u, walls);
        setVBoundary(g, v, walls);
        const u2 = u.slice();
        const v2 = v.slice();
        const p1 = createField(g);
        const p2 = createField(g);
        projectPeriodic(g, u, v, p1, createField(g), 0.02, 0, walls, 1.7, 'direct');
        projectPeriodic(g, u2, v2, p2, createField(g), 0.02, 4000, walls, 1.8, 'sor');
        let dp = 0;
        let pm = 0;
        for (let j = 1; j <= ny; j++)
          for (let i = 1; i <= nx; i++) {
            dp = Math.max(dp, Math.abs(p1[idx(g, i, j)] - p2[idx(g, i, j)]));
            pm = Math.max(pm, Math.abs(p1[idx(g, i, j)]));
          }
        expect(pm).toBeGreaterThan(1e-3);
        expect(dp / pm).toBeLessThan(1e-4);
      });
    }
  }
});

describe('periodic boundary conditions: FluidSolver', () => {
  it('(1) keeps a uniform flow exactly uniform (advection, viscosity, projection, confinement)', () => {
    const s = new FluidSolver(32, 24, { walls: 'periodic', viscosity: 1e-3, vorticity: 2, diffusion: 1e-4 });
    setUniform(s, 0.37, -0.21);
    for (let n = 0; n < 200; n++) s.step();
    expect(maxDeviation(s, 0.37, -0.21)).toBeLessThan(1e-6);
  });

  it('(1b) an inviscid plug flow in the periodic-x channel persists unchanged', () => {
    const s = new FluidSolver(32, 24, { walls: 'periodic-x', viscosity: 0 });
    const g = s.grid;
    for (let j = 1; j <= g.ny; j++) for (let i = 0; i <= g.nx; i++) s.u[uIdx(g, i, j)] = 0.5;
    s.setParams({});
    for (let n = 0; n < 200; n++) s.step();
    let m = 0;
    for (let j = 1; j <= g.ny; j++) for (let i = 0; i <= g.nx; i++) m = Math.max(m, Math.abs(s.u[uIdx(g, i, j)] - 0.5));
    expect(m).toBeLessThan(1e-6);
    expect(maxAbsInterior(g, s.v)).toBe(0);
  });

  it('(2a) a sine shear wave decays at the analytic viscous rate exp(−νk²t)', () => {
    const N = 64;
    const nu = 0.005;
    const k = 2 * Math.PI;
    const s = new FluidSolver(N, N, { walls: 'periodic', viscosity: nu, iterations: 80 });
    const g = s.grid;
    // u = A sin(k y) + a uniform drift in x, which must not affect the decay
    for (let j = 1; j <= N; j++)
      for (let i = 0; i <= N; i++) s.u[uIdx(g, i, j)] = 0.3 + 0.1 * Math.sin(k * (j - 0.5) * g.h);
    s.setParams({});
    const T = 1;
    const steps = Math.round(T / s.params.dt);
    for (let n = 0; n < steps; n++) s.step();
    let amp = 0;
    for (let j = 1; j <= N; j++) amp += (s.u[uIdx(g, 5, j)] - 0.3) * Math.sin(k * (j - 0.5) * g.h);
    amp *= 2 / N;
    const expected = 0.1 * Math.exp(-nu * k * k * T);
    expect(Math.abs(amp / expected - 1)).toBeLessThan(0.01);
  });

  it('(2b) a Taylor–Green vortex decays at the analytic viscous rate exp(−2νk²t)', () => {
    const N = 64;
    const nu = 0.005;
    const k = 2 * Math.PI;
    const A = 0.01;
    const s = new FluidSolver(N, N, { walls: 'periodic', viscosity: nu, iterations: 80 });
    const g = s.grid;
    for (let j = 1; j <= N; j++)
      for (let i = 0; i <= N; i++) {
        const x = i * g.h;
        const y = (j - 0.5) * g.h;
        s.u[uIdx(g, i, j)] = A * Math.sin(k * x) * Math.cos(k * y);
      }
    for (let j = 0; j <= N; j++)
      for (let i = 1; i <= N; i++) {
        const x = (i - 0.5) * g.h;
        const y = j * g.h;
        s.v[vIdx(g, i, j)] = -A * Math.cos(k * x) * Math.sin(k * y);
      }
    s.setParams({});
    const T = 1;
    const steps = Math.round(T / s.params.dt);
    for (let n = 0; n < steps; n++) s.step();
    // project onto the initial mode
    let num = 0;
    let den = 0;
    for (let j = 1; j <= N; j++)
      for (let i = 1; i <= N; i++) {
        const m = Math.sin(k * i * g.h) * Math.cos(k * (j - 0.5) * g.h);
        num += s.u[uIdx(g, i, j)] * m;
        den += m * m;
      }
    const amp = num / den;
    const expected = A * Math.exp(-2 * nu * k * k * T);
    expect(Math.abs(amp / expected - 1)).toBeLessThan(0.01);
  });

  it('(3) conserves total momentum (no walls to push against)', () => {
    const s = new FluidSolver(48, 32, { walls: 'periodic', viscosity: 1e-4, vorticity: 0 });
    stirRandomly(s, rng(11), 40);
    const before = s.meanVelocity();
    // viscosity + projection on their own conserve it to round-off …
    s.setParams({ viscosity: 1e-3 });
    const p0 = s.meanVelocity();
    for (let n = 0; n < 5; n++) {
      project(s.grid, s.u, s.v, s.pressure, createField(s.grid), s.params.dt, 30, 'periodic');
    }
    const p1 = s.meanVelocity();
    expect(Math.abs(p1[0] - p0[0])).toBeLessThan(1e-6);
    expect(Math.abs(p1[1] - p0[1])).toBeLessThan(1e-6);
    // … and a full run (semi-Lagrangian advection is not exactly conservative) stays close
    for (let n = 0; n < 200; n++) s.step();
    const after = s.meanVelocity();
    const scale = Math.max(Math.hypot(...before), 0.05);
    expect(Math.hypot(after[0] - before[0], after[1] - before[1]) / scale).toBeLessThan(0.05);
  });

  it('(4) keeps a uniform dye exactly uniform, bounds the dye drift, and carries dye across the seam', () => {
    const drift = (walls: WallType, uniformDye: boolean) => {
      const s = new FluidSolver(48, 32, { walls, viscosity: 1e-4 });
      s.seedActiveTurbulence(5, 0.6);
      for (let n = 0; n < 30; n++) s.step();
      const g = s.grid;
      if (uniformDye) {
        for (const d of s.dye) d.fill(0);
        for (let j = 1; j <= g.ny; j++) for (let i = 1; i <= g.nx; i++) s.dye[0][idx(g, i, j)] = 1;
      }
      const m0 = sumInterior(g, s.dye[0]);
      for (let n = 0; n < 300; n++) s.step();
      return Math.abs(sumInterior(g, s.dye[0]) / m0 - 1);
    };
    // nothing can leak or pile up at a wall: a uniform dye stays uniform on the torus
    expect(drift('periodic', true)).toBeLessThan(1e-5);
    // a patterned dye: semi-Lagrangian advection is not exactly conservative,
    // but the drift stays small and comparable to the closed box
    const dp = drift('periodic', false);
    const dw = drift('no-slip', false);
    expect(dp).toBeLessThan(0.03);
    expect(dp).toBeLessThan(2 * dw + 0.005);

    // a blob advected by a uniform flow leaves through the right edge and re-enters on the left
    const g = createGrid(32, 16);
    const d = createField(g);
    const d0 = createField(g);
    const u = createUField(g);
    const v = createVField(g);
    u.fill(1);
    setUBoundary(g, u, 'periodic');
    setVBoundary(g, v, 'periodic');
    for (let j = 6; j <= 10; j++) for (let i = 28; i <= 32; i++) d[idx(g, i, j)] = 1;
    const mass0 = sumInterior(g, d);
    const dt = 0.5 * g.h; // half a cell per step
    for (let n = 0; n < 16; n++) {
      d0.set(d);
      advectScalar(g, d, d0, u, v, dt, 'periodic');
    }
    // moved by 8 cells: centre of mass from x ≈ 30 − ½ to ≈ 38 − ½ ≡ 6 − ½
    let cx = 0;
    let w = 0;
    for (let j = 1; j <= g.ny; j++)
      for (let i = 1; i <= g.nx; i++) {
        const a = d[idx(g, i, j)];
        cx += a * Math.cos((2 * Math.PI * (i - 0.5)) / g.nx);
        w += a * Math.sin((2 * Math.PI * (i - 0.5)) / g.nx);
      }
    const phase = Math.atan2(w, cx);
    const centre = ((phase / (2 * Math.PI)) * g.nx + g.nx) % g.nx;
    expect(Math.abs(centre - 5.5)).toBeLessThan(0.1);
    expect(Math.abs(sumInterior(g, d) / mass0 - 1)).toBeLessThan(1e-5);
  });

  for (const walls of ['periodic', 'periodic-x'] as WallType[]) {
    it(`(5) projection leaves ∇·u ≈ 0 everywhere incl. the seam cells (${walls})`, () => {
      const g = createGrid(40, 28);
      const rand = rng(walls.length);
      const u = createUField(g);
      const v = createVField(g);
      for (let k = 0; k < u.length; k++) u[k] = rand() - 0.5;
      for (let k = 0; k < v.length; k++) v[k] = rand() - 0.5;
      setUBoundary(g, u, walls);
      setVBoundary(g, v, walls);
      const div = createField(g);
      computeDivergence(g, u, v, div);
      const d0 = maxAbsInterior(g, div);
      const p = createField(g);
      // the periodic pressure solve is direct (FFT), so the iteration count is irrelevant
      project(g, u, v, p, createField(g), 1 / 60, 1, walls);
      computeDivergence(g, u, v, div);
      expect(maxAbsInterior(g, div)).toBeLessThan(1e-5 * d0);
      // seam cells explicitly (first/last column and row)
      let seam = 0;
      for (let j = 1; j <= g.ny; j++) seam = Math.max(seam, Math.abs(div[idx(g, 1, j)]), Math.abs(div[idx(g, g.nx, j)]));
      for (let i = 1; i <= g.nx; i++) seam = Math.max(seam, Math.abs(div[idx(g, i, 1)]), Math.abs(div[idx(g, i, g.ny)]));
      expect(seam).toBeLessThan(1e-5 * d0);
      for (let j = 1; j <= g.ny; j++) expect(u[uIdx(g, 0, j)]).toBe(u[uIdx(g, g.nx, j)]);
    });
  }

  it('is translation invariant: a state shifted across the seam evolves into the shifted result', () => {
    const nx = 40;
    const ny = 30;
    const sx = 17;
    const sy = 11;
    const opts = {
      walls: 'periodic' as WallType, viscosity: 2e-4, vorticity: 3, diffusion: 1e-5, iterations: 400, sor: 1.9,
      selfPropulsion: true, selfPropTau: 0.4, selfPropV0: 0.3,
    };
    const a = new FluidSolver(nx, ny, opts);
    a.seedActiveTurbulence(21, 0.3);
    const b = new FluidSolver(nx, ny, opts);
    const g = a.grid;
    const wi = (i: number) => ((((i - 1 + sx) % nx) + nx) % nx) + 1; // 1..nx
    const wj = (j: number) => ((((j - 1 + sy) % ny) + ny) % ny) + 1;
    const copy = () => {
      for (let j = 1; j <= ny; j++) for (let i = 1; i <= nx; i++) {
        b.u[uIdx(g, wi(i), wj(j))] = a.u[uIdx(g, i, j)];
        b.v[vIdx(g, wi(i), wj(j))] = a.v[vIdx(g, i, j)];
        for (let c = 0; c < 3; c++) b.dye[c][idx(g, wi(i), wj(j))] = a.dye[c][idx(g, i, j)];
      }
      b.setParams({});
    };
    copy();
    for (let n = 0; n < 60; n++) {
      a.step();
      b.step();
    }
    let err = 0;
    let ref = 0;
    for (let j = 1; j <= ny; j++) for (let i = 1; i <= nx; i++) {
      err = Math.max(err, Math.abs(b.u[uIdx(g, wi(i), wj(j))] - a.u[uIdx(g, i, j)]),
        Math.abs(b.v[vIdx(g, wi(i), wj(j))] - a.v[vIdx(g, i, j)]),
        Math.abs(b.dye[1][idx(g, wi(i), wj(j))] - a.dye[1][idx(g, i, j)]));
      ref = Math.max(ref, Math.abs(a.u[uIdx(g, i, j)]));
    }
    expect(ref).toBeGreaterThan(0.1);
    expect(err).toBeLessThan(1e-4);
  });

  it('(6) with self-propulsion a uniform flow reaches |u| = v₀ exactly (unlike in a walled box)', () => {
    const v0 = 0.6;
    const run = (walls: WallType) => {
      const s = new FluidSolver(40, 30, {
        walls, viscosity: 1e-4, selfPropulsion: true, selfPropTau: 0.3, selfPropV0: v0,
      });
      setUniform(s, 0.08, 0.06); // |u| = 0.1, oblique
      for (let n = 0; n < 600; n++) s.step();
      s.updateDiagnostics();
      const g = s.grid;
      let mean = 0;
      for (let j = 1; j <= g.ny; j++)
        for (let i = 1; i <= g.nx; i++) mean += Math.hypot(s.uc[idx(g, i, j)], s.vc[idx(g, i, j)]);
      return { s, mean: mean / (g.nx * g.ny) };
    };
    const p = run('periodic');
    expect(maxDeviation(p.s, 0.8 * v0, 0.6 * v0)).toBeLessThan(1e-5);
    expect(Math.abs(p.mean - v0)).toBeLessThan(1e-5);
    const w = run('no-slip');
    expect(w.mean).toBeLessThan(0.95 * v0);
  });

  it('(6b) active turbulence on the torus settles close to v₀', () => {
    const v0 = 0.5;
    const s = new FluidSolver(64, 48, {
      walls: 'periodic', viscosity: 1e-5, selfPropulsion: true, selfPropTau: 0.5, selfPropV0: v0,
    });
    s.seedActiveTurbulence(9, 0.05);
    for (let n = 0; n < 900; n++) s.step();
    s.updateDiagnostics();
    const g = s.grid;
    let mean = 0;
    for (let j = 1; j <= g.ny; j++)
      for (let i = 1; i <= g.nx; i++) mean += Math.hypot(s.uc[idx(g, i, j)], s.vc[idx(g, i, j)]);
    mean /= g.nx * g.ny;
    expect(mean).toBeGreaterThan(0.8 * v0);
    expect(mean).toBeLessThan(1.05 * v0);
    expect(allFinite(s.u, s.v)).toBe(true);
  });

  for (const walls of ['periodic', 'periodic-x'] as WallType[]) {
    it(`(7) no NaNs after many steps with stirring across the edges (${walls})`, () => {
      const s = new FluidSolver(48, 32, {
        walls, vorticity: 4, diffusion: 1e-5, selfPropulsion: true, selfPropV0: 0.8, selfPropTau: 0.2,
      });
      s.seedActiveTurbulence(4, 0.1);
      stirRandomly(s, rng(23), 400);
      s.updateDiagnostics();
      expect(allFinite(s.u, s.v, s.pressure, ...s.dye, s.vorticity, s.divergence)).toBe(true);
      expect(maxAbsInterior(s.grid, s.uc)).toBeLessThan(20);
      expect(maxAbsInterior(s.grid, s.divergence)).toBeLessThan(5);
    });
  }

  it('switching boundaries at runtime re-projects (walls ↔ periodic)', () => {
    const s = new FluidSolver(40, 30, { walls: 'no-slip' });
    stirRandomly(s, rng(31), 30);
    for (const walls of ['periodic', 'periodic-x', 'free-slip', 'periodic', 'no-slip'] as WallType[]) {
      s.setParams({ walls });
      s.updateDiagnostics();
      expect(maxAbsInterior(s.grid, s.divergence)).toBeLessThan(0.05);
      for (let n = 0; n < 10; n++) s.step();
      expect(allFinite(s.u, s.v, ...s.dye)).toBe(true);
    }
    const g = s.grid;
    for (let j = 1; j <= g.ny; j++) {
      expect(s.u[uIdx(g, 0, j)]).toBe(0);
      expect(s.u[uIdx(g, g.nx, j)]).toBe(0);
    }
  });

  it('a stroke at the edge wraps to the opposite side', () => {
    const s = new FluidSolver(40, 30, { walls: 'periodic' });
    const g = s.grid;
    s.splat(0.2, 15, 0, 0, [1, 0, 0], 3);
    expect(s.dye[0][idx(g, g.nx, 15)]).toBeGreaterThan(0.5);
    expect(s.dye[0][idx(g, 1, 15)]).toBeGreaterThan(0.5);
    s.stir(39.8, 0.1, 1, 0, 3, 1);
    expect(s.u[uIdx(g, 1, 1)]).toBeGreaterThan(0.3);
    expect(s.u[uIdx(g, 1, g.ny)]).toBeGreaterThan(0.3);
    // walls: no wrap
    const w = new FluidSolver(40, 30, { walls: 'no-slip' });
    w.splat(0.2, 15, 0, 0, [1, 0, 0], 3);
    expect(w.dye[0][idx(g, g.nx, 15)]).toBe(0);
  });
});

describe('solid walls are unchanged by the periodic option', () => {
  // Fingerprints of wallScenario() produced by the code before periodic
  // boundaries were added (commit d7f11b7).
  it('(8) no-slip run is bit-identical to the previous version', () => {
    expect(wallScenario('no-slip')).toBe('05521659');
  });
  it('(8) free-slip run is bit-identical to the previous version', () => {
    expect(wallScenario('free-slip')).toBe('a4f744c4');
  });
});
