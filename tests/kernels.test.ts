import { describe, expect, it } from 'vitest';
import { FluidSolver, NonlocalOperator, uIdx, vIdx, type KernelParams, type WallType } from '../src/solver';
import { wallScenario } from './fingerprint';

const kp = (o: Partial<KernelParams>): KernelParams => ({
  kernel: 'gaussian', strength: 2, range: 0.05, beta: 1, alpha: 3, s: 0.5, motschTadmor: false, ...o,
});

/** Run the operator alone on an n×n periodic sub-grid on a given field. */
function runOp(op: NonlocalOperator, f: Float64Array, dt: number, steps = 1): Float64Array {
  const x = f.slice();
  for (let n = 0; n < steps; n++) op.apply((i, j) => x[i + op.n1 * j], (i, j, val) => { x[i + op.n1 * j] = val; }, dt);
  return x;
}
const sum = (a: Float64Array) => a.reduce((s, x) => s + x, 0);
const energy = (a: Float64Array) => a.reduce((s, x) => s + x * x, 0);
function mode(n: number, kx: number, ky: number): Float64Array {
  const f = new Float64Array(n * n);
  for (let j = 0; j < n; j++) for (let i = 0; i < n; i++) f[i + n * j] = Math.sin((2 * Math.PI * (kx * i + ky * j)) / n);
  return f;
}
function rand(n: number, seed = 3): Float64Array {
  let a = seed;
  const f = new Float64Array(n);
  for (let k = 0; k < n; k++) { a = (a * 1664525 + 1013904223) >>> 0; f[k] = a / 2 ** 32 - 0.5; }
  return f;
}
/** Exact continuous transform of the normalised Gaussian. */
const gaussHat = (R: number, k: number) => Math.exp(-0.5 * R * R * k * k);

describe('nonlocal viscosity kernels', () => {
  const N = 48, h = 1 / N;
  for (const kernel of ['gaussian', 'cucker-smale', 'power', 'fractional'] as const) {
    it(`${kernel}: conserves momentum, dissipates energy, keeps uniform flow (periodic)`, () => {
      const op = new NonlocalOperator(N, N, true, true, h, kp({ kernel, strength: kernel === 'fractional' ? 1e-3 : 3 }));
      const f = rand(N * N).map((x) => x + 0.7);
      let x: Float64Array = f;
      let e = energy(f);
      for (let n = 0; n < 20; n++) {
        const y = runOp(op, x, 1 / 60);
        expect(energy(y)).toBeLessThanOrEqual(e * (1 + 1e-12));
        e = energy(y);
        x = y;
      }
      expect(sum(x)).toBeCloseTo(sum(f), 8);
      const uni = new Float64Array(N * N).fill(0.42);
      const out = runOp(op, uni, 0.1, 5);
      for (const val of out) expect(val).toBeCloseTo(0.42, 12);
    });
  }

  for (const kernel of ['gaussian', 'cucker-smale'] as const) {
    it(`${kernel}: single Fourier mode decays at S(K̂(0) − K̂(k))`, () => {
      const S = 4, dt = 0.05;
      const op = new NonlocalOperator(N, N, true, true, h, kp({ kernel, strength: S, range: 0.06, beta: 1.5 }));
      for (const [kx, ky] of [[1, 0], [3, 2], [7, 0]]) {
        const f = mode(N, kx, ky);
        const y = runOp(op, f, dt);
        const rate = -Math.log(Math.sqrt(energy(y) / energy(f))) / dt;
        // independent: direct real-space evaluation of K̂(k) = Σ K(d) cos(k·d) / Σ K(d)
        let num = 0, den = 0;
        for (let b = 0; b < N; b++) for (let a = 0; a < N; a++) {
          const da = Math.min(a, N - a), db = Math.min(b, N - b);
          const r = h * Math.hypot(da, db);
          const K = kernel === 'gaussian' ? Math.exp(-(r * r) / (2 * 0.06 ** 2)) : (1 + r * r / 0.06 ** 2) ** -1.5;
          num += K * Math.cos((2 * Math.PI * (kx * a + ky * b)) / N);
          den += K;
        }
        expect(rate).toBeCloseTo(S * (1 - num / den), 6);
        if (kernel === 'gaussian') {
          const k = 2 * Math.PI * Math.hypot(kx, ky);
          expect(rate).toBeCloseTo(S * (1 - gaussHat(0.06, k)), 2);
        }
      }
    });
  }

  it('gaussian with small R matches local Laplacian scaling ν_eff = S R²/2', () => {
    const R = 1.5 * h, S = 10; // (1 − e^{−x})/x ≈ 1 for x = R²k²/2 ≪ 1
    const op = new NonlocalOperator(N, N, true, true, h, kp({ strength: S, range: R }));
    for (const kx of [1]) {
      const k = 2 * Math.PI * kx;
      const y = runOp(op, mode(N, kx, 0), 0.01);
      const rate = -Math.log(Math.sqrt(energy(y) / energy(mode(N, kx, 0)))) / 0.01;
      expect(rate / ((S * R * R / 2) * k * k)).toBeGreaterThan(0.97);
      expect(rate / ((S * R * R / 2) * k * k)).toBeLessThan(1.03);
    }
  });

  it('fractional s = 1 matches local viscosity decay; s < 1 flattens the spectrum', () => {
    const nu = 2e-4, dt = 1 / 60;
    const rateOf = (s: number, kx: number) => {
      const op = new NonlocalOperator(N, N, true, true, h, kp({ kernel: 'fractional', strength: nu, s }));
      const f = mode(N, kx, 0);
      return -Math.log(Math.sqrt(energy(runOp(op, f, dt)) / energy(f))) / dt;
    };
    for (const kx of [1, 4]) {
      const lam = (4 / (h * h)) * Math.sin(Math.PI * kx / N) ** 2;
      expect(rateOf(1, kx)).toBeCloseTo(nu * lam, 6);
    }
    // local solver: same decay with the fractional term off and ν on
    const s = new FluidSolver(N, N, { walls: 'periodic', viscosity: nu, iterations: 200, vorticity: 0 });
    for (let j = 1; j <= N; j++) for (let i = 1; i <= N; i++) s.u[uIdx(s.grid, i, j)] = Math.sin(2 * Math.PI * (j - 0.5) / N);
    const eU = () => { let e = 0; for (let j = 1; j <= N; j++) for (let i = 1; i <= N; i++) e += s.u[uIdx(s.grid, i, j)] ** 2; return e; };
    const e0 = eU();
    s.setParams({ viscosity: nu });
    // single implicit diffusion step decays by 1/(1+νλΔt); compare with exp
    s.step();
    const e1 = eU();
    const localRate = -Math.log(Math.sqrt(e1 / e0)) / dt;
    expect(localRate / rateOf(1, 1)).toBeGreaterThan(0.95);
    expect(localRate / rateOf(1, 1)).toBeLessThan(1.05);
    // s < 1: high k decays slower, low k relatively faster
    expect(rateOf(0.5, 12)).toBeLessThan(rateOf(1, 12));
    expect(rateOf(0.5, 12) / rateOf(0.5, 1)).toBeLessThan(rateOf(1, 12) / rateOf(1, 1));
  });

  it('walled masked convolution does not wrap across walls', () => {
    const n = 40;
    for (const kernel of ['gaussian', 'cucker-smale', 'power'] as const) {
      const op = new NonlocalOperator(n, n, false, false, 1 / n, kp({ kernel, range: 0.03, beta: 2, alpha: 4 }));
      const f = new Float64Array(n * n);
      f[0 + n * 20] = 1; // impulse at the left wall
      const y = runOp(op, f, 0.05);
      const left = y[1 + n * 20], right = y[n - 1 + n * 20];
      expect(left).toBeGreaterThan(0);
      // periodic wrap would make right ≈ left; linear conv gives the far-field value only
      const opP = new NonlocalOperator(n, n, true, true, 1 / n, kp({ kernel, range: 0.03, beta: 2, alpha: 4 }));
      const yP = runOp(opP, f, 0.05);
      expect(yP[n - 1 + n * 20]).toBeGreaterThan(0.5 * yP[1 + n * 20]);
      expect(right).toBeLessThan(1e-2 * left);
      for (const val of y) expect(Number.isFinite(val)).toBe(true);
      // max principle
      for (const val of y) expect(val).toBeLessThanOrEqual(1 + 1e-12);
    }
  });

  it('no NaNs after many steps with kernels + self-propulsion + stirring', () => {
    const cases: [WallType, KernelParams['kernel']][] = [
      ['periodic', 'gaussian'], ['periodic', 'cucker-smale'], ['periodic', 'power'], ['periodic', 'fractional'],
      ['no-slip', 'cucker-smale'], ['periodic-x', 'power'], ['free-slip', 'gaussian'], ['no-slip', 'fractional'],
    ];
    for (const [walls, kernel] of cases) {
      const s = new FluidSolver(36, 24, {
        walls, kernel, kernelStrength: kernel === 'fractional' ? 1e-3 : 5, kernelMT: walls === 'no-slip',
        selfPropulsion: true, selfPropTau: 0.3, selfPropV0: 0.6, vorticity: 2,
      });
      s.seedActiveTurbulence(2, 0.1);
      for (let n = 0; n < 150; n++) {
        s.stir(5 + (n % 25), 12, 1, 0.3, 3, 0.5);
        s.step();
      }
      for (const f of [s.u, s.v, s.pressure]) for (const x of f) expect(Number.isFinite(x)).toBe(true);
      expect(Math.max(...Array.from(s.u).map(Math.abs))).toBeLessThan(10);
    }
  });

  it('solver-level kernel conserves mean velocity and keeps seam copies consistent (periodic)', () => {
    const s = new FluidSolver(32, 24, { walls: 'periodic', viscosity: 0, vorticity: 0, kernel: 'cucker-smale', kernelStrength: 6 });
    s.seedActiveTurbulence(4, 0.2);
    for (let j = 1; j <= 24; j++) for (let i = 0; i <= 32; i++) s.u[uIdx(s.grid, i, j)] += 0.3;
    const m0 = s.meanVelocity();
    for (let n = 0; n < 30; n++) s.step();
    const m1 = s.meanVelocity();
    expect(m1[0]).toBeCloseTo(m0[0], 2);
    for (let j = 1; j <= 24; j++) expect(s.u[uIdx(s.grid, 0, j)]).toBe(s.u[uIdx(s.grid, 32, j)]);
    void vIdx;
  });

  it("kernel 'none' leaves every boundary type bit-identical to 22fd8ba", () => {
    // fingerprints recorded at commit 22fd8ba
    const ref: Record<string, string> = REF;
    for (const w of Object.keys(ref) as WallType[]) expect(wallScenario(w)).toBe(ref[w]);
  });
});

const REF: Record<string, string> = {'no-slip': '05521659', 'free-slip': 'a4f744c4', 'periodic': 'ccb76e4e', 'periodic-x': '9641dea9'};
