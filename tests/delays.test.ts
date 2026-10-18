import { describe, expect, it } from 'vitest';
import {
  DELAY_SPEED_LIMIT, FieldHistory, FluidSolver, NonlocalOperator, rightmostRoot, scanModes, uIdx, vIdx,
  type KernelParams, type WallType,
} from '../src/solver';
import { KERNEL_CASES, kernelScenario, wallScenario } from './fingerprint';
import { delayRoot, lambertW0 } from './lambert';

const kp = (o: Partial<KernelParams>): KernelParams => ({
  kernel: 'cucker-smale', strength: 4, range: 0.05, beta: 1, alpha: 3, s: 0.5, motschTadmor: false, ...o,
});
const maxAbs = (a: ArrayLike<number>) => { let m = 0; for (let k = 0; k < a.length; k++) m = Math.max(m, Math.abs(a[k])); return m; };
const WALLS: WallType[] = ['no-slip', 'free-slip', 'periodic', 'periodic-x'];

/** Growth rate (log-envelope slope through |peaks| in [tMin, tMax]) and angular frequency (zero crossings). */
function fitOscillation(A: number[], dt: number, tMin = 2, tMax = Infinity) {
  const zc: number[] = [];
  const pk: [number, number][] = [];
  for (let n = 1; n < A.length - 1; n++) {
    const t = n * dt;
    if (t < tMin || t > tMax) continue;
    if (A[n - 1] * A[n] < 0) zc.push((n - 1 + A[n - 1] / (A[n - 1] - A[n])) * dt);
    if (Math.abs(A[n]) >= Math.abs(A[n - 1]) && Math.abs(A[n]) > Math.abs(A[n + 1])) pk.push([t, Math.log(Math.abs(A[n]))]);
  }
  let sx = 0, sy = 0, sxx = 0, sxy = 0;
  for (const [x, y] of pk) { sx += x; sy += y; sxx += x * x; sxy += x * y; }
  const k = pk.length;
  const mu = (k * sxy - sx * sy) / (k * sxx - sx * sx);
  const omega = zc.length > 3 ? Math.PI * (zc.length - 1) / (zc[zc.length - 1] - zc[0]) : 0;
  return { mu, omega, signChanges: zc.length };
}

/**
 * Shear wave u = a sin(2π m (y − ½h)/L) on the torus: an exact Navier–Stokes
 * solution (advection and pressure vanish), so the projection of u onto the
 * profile follows the linear two-delay equation of its Fourier mode.
 */
function shearWave(params: Record<string, unknown>, N: number, m: number, steps: number, amp = 0.02) {
  const s = new FluidSolver(N, N, { walls: 'periodic', viscosity: 0, vorticity: 0, ...params });
  const prof = (j: number) => Math.sin((2 * Math.PI * m * (j - 0.5)) / N);
  for (let j = 1; j <= N; j++) for (let i = 0; i <= N; i++) s.u[uIdx(s.grid, i, j)] = amp * prof(j);
  const A: number[] = [];
  for (let n = 0; n < steps; n++) {
    s.step();
    let q = 0; for (let j = 1; j <= N; j++) q += s.u[uIdx(s.grid, 5, j)] * prof(j);
    A.push(q / (N / 2));
  }
  return { s, A };
}

// instantaneous-kernel fingerprints recorded at dc86501 (unchanged through 6c811e2)
const REF_DC86501: Record<string, string> = {
  'periodic/gaussian/false': '4c00fa13', 'periodic/cucker-smale/false': '5ca0090e', 'periodic/power/false': '18101e34',
  'periodic/fractional/false': '0cfec16f', 'no-slip/cucker-smale/true': '55a9d034', 'free-slip/gaussian/false': 'd27d549f',
  'periodic-x/power/false': 'c71beb14', 'periodic-x/cucker-smale/true': '9fecc243',
};
// wallScenario fingerprints recorded at 22fd8ba (unchanged through 6c811e2)
const REF_WALLS: Record<WallType, string> = { 'no-slip': '05521659', 'free-slip': 'a4f744c4', periodic: 'ccb76e4e', 'periodic-x': '9641dea9' };
// kernelScenario with the calculation delay t_c = 0.3 s, recorded at 6c811e2
const REF_6C811E2_TC03: Record<string, string> = {
  'periodic/gaussian/false': 'eb1da30a', 'periodic/cucker-smale/false': '29496a5c', 'periodic/power/false': 'ec321d1c',
  'no-slip/cucker-smale/true': 'b831c18c', 'free-slip/gaussian/false': '2253493e', 'periodic-x/power/false': 'b31dc156',
  'periodic-x/cucker-smale/true': 'bc42a8e0',
};
// shear-wave fits (growth rate μ, frequency ω) of the t_c path measured at 6c811e2 (cucker-smale, R = 0.15, S = 4, 48², m = 4)
const FIT_6C811E2: Record<string, [number, number]> = {
  '0.3': [-0.653140850929808, 4.783023592953814], '0.38': [-0.07599283358104798, 4.084708840182657],
  '0.41': [0.061907453501335405, 3.8702341655624815], '0.45': [0.20378828765322207, 3.615774156243665],
};

describe('two-delay characteristic equation λ = −A e^{−λ t_self} + B e^{−λ t_nb}', () => {
  it('reduces to Lambert W for one delay / equal delays and to B + W(−Aτe^{−Bτ})/τ for t_nb = 0', () => {
    for (const [a, t] of [[1, 0.25], [1, 1], [1, 1.8], [4, 0.42], [7, 0.3]]) {
      const lw = delayRoot(a, t);
      for (const r of [rightmostRoot(a, 0, t, 0.37), rightmostRoot(a * 1.6, a * 0.6, t, t)]) {
        expect(r.re).toBeCloseTo(lw.re, 9);
        expect(r.im).toBeCloseTo(lw.im, 9);
      }
    }
    for (const [A, B, t] of [[4, 2, 0.5], [4, 3.9, 0.42], [1, -0.3, 1]]) {
      const w = lambertW0(-A * t * Math.exp(-B * t));
      const r = rightmostRoot(A, B, t, 0);
      expect(r.re).toBeCloseTo(B + w.re / t, 9);
      expect(r.im).toBeCloseTo(w.im / t, 9);
    }
  });

  it('own velocity current (t_self = 0): stable for |B| ≤ A at every neighbour delay (Hayes)', () => {
    for (const tn of [0.1, 0.5, 1, 3, 8]) for (const q of [-1, -0.6, 0, 0.5, 0.95]) {
      expect(rightmostRoot(1, q, 0, tn).re, `${q}/${tn}`).toBeLessThanOrEqual(1e-9);
    }
    // and unstable (real root) for B > A
    expect(rightmostRoot(1, 1.2, 0, 0.5).re).toBeGreaterThan(0);
  });

  it('local viscosity: the first unstable mode is the checkerboard (c = −1) or the longest wave (c → 1), never a mid-k mode', () => {
    for (const [ts, tn] of [[0.8, 0.8], [1, 1], [0.6, 2], [1, 2.5], [1.5, 0], [0.8, 3], [1, 4]]) {
      const sc = scanModes(1, -1, 0.999, ts, tn, 41);
      expect(sc.root.re, `${ts}/${tn}`).toBeGreaterThan(0);
      expect(Math.abs(Math.abs(sc.param) - 1), `${ts}/${tn}`).toBeLessThan(0.002);
    }
    // equal delays: checkerboard at 2Aτ = π/2 (rate 8ν/h²), i.e. half the single-mode threshold
    expect(scanModes(1, -1, 0.999, 0.78, 0.78).root.re).toBeLessThan(0);
    expect(scanModes(1, -1, 0.999, 0.79, 0.79).param).toBe(-1);
  });
});

describe('delays t_self / t_nb: backward compatibility', () => {
  it('all delays 0 ⇒ bit-identical to 6c811e2 (instant paths, all boundaries, delay flags on or off)', () => {
    for (const flags of [{ delayLocal: true, delayKernel: true }, { delayLocal: false, delayKernel: false }]) {
      for (const [w, k, mt] of KERNEL_CASES) {
        const key = `${w}/${k}/${mt}`;
        expect(kernelScenario(w, k, mt, { delaySelf: 0, delayNb: 0, ...flags }), key).toBe(REF_DC86501[key]);
      }
      for (const w of WALLS) expect(wallScenario(w, { delaySelf: 0, delayNb: 0, ...flags }), w).toBe(REF_WALLS[w]);
    }
    // delays with both components disabled are inert too
    for (const w of WALLS) expect(wallScenario(w, { delaySelf: 0.4, delayNb: 0.2, delayLocal: false, delayKernel: false })).toBe(REF_WALLS[w]);
    // fractional on walls is inactive, so a delay there changes nothing
    expect(kernelScenario('no-slip', 'fractional', false, { delaySelf: 0.3, delayNb: 0.3, delayLocal: false })).toBe(
      kernelScenario('no-slip', 'fractional', false, {}));
  });

  it('t_self = t_nb = t_c on the long-range term reproduces the 6c811e2 calculation delay (bit-identical with walls, ≤ 2e-7 on the torus)', () => {
    for (const [w, k, mt] of KERNEL_CASES) {
      if (k === 'fractional') continue;
      const key = `${w}/${k}/${mt}`;
      const h = kernelScenario(w, k, mt, { delaySelf: 0.3, delayNb: 0.3, delayLocal: false });
      if (w !== 'periodic') expect(h, key).toBe(REF_6C811E2_TC03[key]);
    }
    // torus: 6c811e2 stored spectra (Float32) instead of real fields; compare the fitted shear-wave dynamics
    for (const tc of [0.3, 0.38, 0.41, 0.45]) {
      const { A } = shearWave({ kernel: 'cucker-smale', kernelRange: 0.15, kernelStrength: 4, delaySelf: tc, delayNb: tc }, 48, 4, 900);
      const f = fitOscillation(A, 1 / 60, 2);
      const [mu, om] = FIT_6C811E2[String(tc)];
      expect(Math.abs(f.mu - mu), `tc=${tc}`).toBeLessThan(1e-6);
      expect(Math.abs(f.omega - om), `tc=${tc}`).toBeLessThan(1e-6);
    }
  }, 60000);
});

describe('delayed long-range alignment S[(K*u)(t − t_nb) − c u(t − t_self)]', () => {
  it('shear wave: growth rate and frequency follow λ = −S e^{−λ t_self} + S K̂ e^{−λ t_nb}', () => {
    const N = 48, m = 4, S = 4;
    const Kh = new FluidSolver(N, N, { walls: 'periodic', kernel: 'cucker-smale', kernelRange: 0.15 }).kernelOperators().u.symbol(0, m);
    for (const [ts, tn] of [[0.45, 0.2], [0.45, 0.6], [0.3, 0.8], [0.42, 0.42], [0.5, 0.1]]) {
      const root = rightmostRoot(S, S * Kh, ts, tn);
      const { A } = shearWave({ kernel: 'cucker-smale', kernelRange: 0.15, kernelStrength: S, delaySelf: ts, delayNb: tn }, N, m, 900);
      const f = fitOscillation(A, 1 / 60, 2);
      expect(root.im, `${ts}/${tn}`).toBeGreaterThan(0);
      expect(Math.sign(f.mu), `${ts}/${tn}`).toBe(Math.sign(root.re));
      expect(Math.abs(f.mu - root.re), `${ts}/${tn}`).toBeLessThan(0.01 + 0.03 * Math.abs(root.re));
      expect(Math.abs(f.omega / root.im - 1), `${ts}/${tn}`).toBeLessThan(0.01);
    }
    // narrow Gaussian (K̂ = 0.454 at m = 4): the neighbour delay matters as much as the own one
    const Kg = new FluidSolver(N, N, { walls: 'periodic', kernel: 'gaussian', kernelRange: 0.05 }).kernelOperators().u.symbol(0, m);
    expect(Kg).toBeGreaterThan(0.4);
    for (const [ts, tn] of [[0.3, 0.6], [0.45, 0.2], [0.7, 0.3]]) {
      const root = rightmostRoot(S, S * Kg, ts, tn);
      const { A } = shearWave({ kernel: 'gaussian', kernelRange: 0.05, kernelStrength: S, delaySelf: ts, delayNb: tn }, N, m, 600);
      const f = fitOscillation(A, 1 / 60, 1.5);
      expect(Math.abs(f.mu - root.re), `g ${ts}/${tn}`).toBeLessThan(0.01 + 0.03 * Math.abs(root.re));
      expect(Math.abs(f.omega / root.im - 1), `g ${ts}/${tn}`).toBeLessThan(0.01);
    }
  }, 60000);

  it('own velocity delayed, neighbours current (t_nb = 0): long waves run away without oscillating once S·t_self > 1', () => {
    const N = 48, S = 4, ts = 0.42; // S t_self = 1.68
    // narrow Gaussian: the m = 1 wave has K̂ ≈ 0.95 (a long wave on the kernel's scale)
    const Kh = new FluidSolver(N, N, { walls: 'periodic', kernel: 'gaussian', kernelRange: 0.05 }).kernelOperators().u.symbol(0, 1);
    expect(Kh).toBeGreaterThan(0.9);
    const root = rightmostRoot(S, S * Kh, ts, 0);
    expect(root.im).toBe(0);
    expect(root.re).toBeGreaterThan(1);
    const { A } = shearWave({ kernel: 'gaussian', kernelRange: 0.05, kernelStrength: S, delaySelf: ts, delayNb: 0 }, N, 1, 150, 1e-4);
    // Starting from a stationary history the wave first decays (the own velocity looks unchanged),
    // overshoots through zero once and then runs away monotonically: one sign change, |A| growing.
    let sc = 0; for (let n = 1; n < A.length; n++) if (A[n] * A[n - 1] < 0) sc++;
    expect(sc).toBe(1);
    for (let n = 100; n < A.length; n++) expect(Math.abs(A[n])).toBeGreaterThan(Math.abs(A[n - 1]));
    const rate = Math.log(A[149] / A[119]) / (30 / 60);
    expect(Math.abs(rate / root.re - 1)).toBeLessThan(0.03);
  });

  it('t_self = 0, t_nb > 0: semi-implicit, max principle — never grows (periodic and walled, MT on/off)', () => {
    for (const w of WALLS) for (const mt of [false, true]) {
      const s = new FluidSolver(36, 24, {
        walls: w, viscosity: 0, vorticity: 0, kernel: 'cucker-smale', kernelStrength: 30, kernelMT: mt, delaySelf: 0, delayNb: 0.7,
      });
      s.seedActiveTurbulence(3, 0.2);
      let prev = maxAbs(s.u);
      const m0 = prev;
      for (let n = 0; n < 240; n++) {
        s.step();
        const mx = maxAbs(s.u);
        expect(mx, `${w}/${mt}`).toBeLessThan(m0 * 1.25 + 1e-9); // projection may redistribute slightly
        prev = mx;
      }
      expect(prev, `${w}/${mt}`).toBeLessThan(m0);
    }
  });
});

describe('delayed local viscosity ν/h²[Σ_nb u(t − t_nb) − 4u(t − t_self)]', () => {
  const N = 24, h = 1 / N;
  const nuFor = (A: number) => (A * h * h) / 4; // A = 4ν/h²
  const cOf = (m: number) => (1 + Math.cos((2 * Math.PI * m) / N)) / 2;

  it('single Fourier mode: decay / growth rate and frequency match the two-delay equation', () => {
    const A0 = 8;
    // [mode, t_self, t_nb]: chosen so the tested mode is the fastest y-mode (round-off cannot overtake it)
    for (const [m, ts, tn] of [[6, 0.3, 0.3], [6, 0.1, 0.3], [6, 0.22, 0.15], [12, 0.2, 0.2]]) {
      const root = rightmostRoot(A0, A0 * cOf(m), ts, tn);
      const { A } = shearWave({ viscosity: nuFor(A0), delaySelf: ts, delayNb: tn }, N, m, 900);
      const f = fitOscillation(A, 1 / 60, 1.5);
      expect(root.im, `${m}/${ts}/${tn}`).toBeGreaterThan(0);
      expect(Math.abs(f.mu - root.re), `${m}/${ts}/${tn}`).toBeLessThan(0.01 + 0.03 * Math.abs(root.re));
      expect(Math.abs(f.omega / root.im - 1), `${m}/${ts}/${tn}`).toBeLessThan(0.02);
    }
    // own delay ≫ neighbour delay: the longest wave runs away monotonically (A(t_self − t_nb) > 1)
    {
      const root = rightmostRoot(A0, A0 * cOf(1), 0.3, 0.1);
      expect(root.im).toBe(0);
      const { A } = shearWave({ viscosity: nuFor(A0), delaySelf: 0.3, delayNb: 0.1 }, N, 1, 240, 1e-4);
      let sc = 0; for (let n = 1; n < A.length; n++) if (A[n] * A[n - 1] < 0) sc++;
      expect(sc).toBe(1); // one overshoot through zero, then monotone run-away
      for (let n = 120; n < A.length; n++) expect(Math.abs(A[n])).toBeGreaterThan(Math.abs(A[n - 1]));
      const rate = Math.log(A[239] / A[179]) / 1;
      expect(Math.abs(rate / root.re - 1)).toBeLessThan(0.03);
    }
    // neighbour delay ≫ own delay: long waves oscillate with growing amplitude (A t_self = 1, A t_nb = 4)
    {
      const A1 = 2, ts = 0.5, tn = 2;
      // m = 4 (c = 0.75): its slowly decaying real root (−0.145 /s) is gone by t ≈ 10 s
      const root = rightmostRoot(A1, A1 * cOf(4), ts, tn);
      expect(root.re).toBeGreaterThan(0);
      expect(rightmostRoot(A1, A1 * cOf(1), ts, tn).re).toBeGreaterThan(root.re); // the longest wave is faster still
      const { A } = shearWave({ viscosity: nuFor(A1), delaySelf: ts, delayNb: tn }, N, 4, 1500, 1e-4);
      const f = fitOscillation(A, 1 / 60, 10);
      expect(Math.abs(f.mu - root.re)).toBeLessThan(0.01 + 0.05 * Math.abs(root.re));
      expect(Math.abs(f.omega / root.im - 1)).toBeLessThan(0.02);
    }
  }, 60000);

  /** Divergence-free MAC checkerboard u = (−1)^{i+j}, v = −(−1)^{i+j}: Σ_nb = −4u (c = −1). */
  function checkerboard(ts: number, tn: number, A: number, steps: number) {
    const s = new FluidSolver(N, N, { walls: 'periodic', viscosity: nuFor(A), vorticity: 0, delaySelf: ts, delayNb: tn });
    const g = s.grid, a = 1e-6; // tiny: advection of the checkerboard stays negligible
    for (let j = 1; j <= N; j++) for (let i = 1; i <= N; i++) {
      s.u[uIdx(g, i, j)] = a * (-1) ** (i + j);
      s.v[vIdx(g, i, j)] = -a * (-1) ** (i + j);
    }
    s.setParams({});
    const amp: number[] = [];
    for (let n = 0; n < steps; n++) { s.step(); amp.push(s.u[uIdx(g, 3, 4)] / (a * (-1) ** 7)); }
    return amp;
  }

  it('checkerboard (k = π/h, B = −A): equal delays destabilise it first, at 8ν t/h² = π/2, at the Lambert-W rate', () => {
    const A = 4, tau = 0.3; // 2Aτ = 2.4 > π/2, while the mid mode c = 0 (Aτ = 1.2) is still stable
    const root = rightmostRoot(A, -A, tau, tau);
    const lw = delayRoot(2 * A, tau);
    expect(root.re).toBeCloseTo(lw.re, 9);
    expect(rightmostRoot(A, 0, tau, tau).re).toBeLessThan(0);
    const amp = checkerboard(tau, tau, A, 480);
    const f = fitOscillation(amp, 1 / 60, 1.5);
    expect(f.mu).toBeGreaterThan(0);
    expect(Math.abs(f.mu - root.re)).toBeLessThan(0.01 + 0.03 * root.re);
    expect(Math.abs(f.omega / root.im - 1)).toBeLessThan(0.01);
  });

  it('checkerboard with t_self = 0, t_nb > 0: marginal (|B| = A), the semi-implicit update keeps it bounded and slowly decaying', () => {
    for (const tn of [0.1, 0.3, 1]) {
      expect(rightmostRoot(4, -4, 0, tn).re).toBeLessThanOrEqual(1e-9);
      const amp = checkerboard(0, tn, 4, 600);
      expect(maxAbs(amp), `tn=${tn}`).toBeLessThanOrEqual(1 + 1e-5);
      expect(Math.abs(amp[599]), `tn=${tn}`).toBeLessThan(1);
    }
  });

  it('checkerboard with 0 < t_self < t_nb: the neighbour delay alone can destabilise it (e.g. A t_self = 0.6, A t_nb = 2)', () => {
    const A = 4, ts = 0.15, tn = 0.5;
    const root = rightmostRoot(A, -A, ts, tn);
    expect(root.re).toBeGreaterThan(0);
    const amp = checkerboard(ts, tn, A, 1500);
    const f = fitOscillation(amp, 1 / 60, 4);
    expect(Math.abs(f.mu - root.re)).toBeLessThan(0.02 + 0.05 * root.re);
    expect(Math.abs(f.omega / root.im - 1)).toBeLessThan(0.02);
  });

  it('explicit sub-steps keep the self-delayed update stable above νΔt/h² = 1/4 (t_self < Δt)', () => {
    const s = new FluidSolver(32, 32, { walls: 'periodic', viscosity: (1.2 * 60) / (32 * 32), vorticity: 0, delaySelf: 0.004, delayNb: 0 });
    expect(s.delayInfo().localNumber).toBeCloseTo(1.2, 9);
    expect(s.delayInfo().localSubsteps).toBe(6);
    s.seedActiveTurbulence(2, 0.1);
    const m0 = maxAbs(s.u);
    for (let n = 0; n < 120; n++) s.step();
    expect(maxAbs(s.u)).toBeLessThan(m0);
  });
});

describe('delays: invariants and robustness', () => {
  it('a uniform flow is unchanged (both components, any delays)', () => {
    for (const [ts, tn] of [[0.4, 0], [0, 0.4], [0.3, 0.7]]) {
      const s = new FluidSolver(32, 24, {
        walls: 'periodic', viscosity: 5e-4, vorticity: 0, kernel: 'cucker-smale', kernelStrength: 4, delaySelf: ts, delayNb: tn,
      });
      for (let j = 1; j <= 24; j++) for (let i = 0; i <= 32; i++) s.u[uIdx(s.grid, i, j)] = 0.25;
      for (let n = 0; n < 120; n++) s.step();
      for (let j = 1; j <= 24; j++) for (let i = 1; i <= 32; i++) {
        expect(s.u[uIdx(s.grid, i, j)]).toBeCloseTo(0.25, 6);
        expect(Math.abs(s.v[vIdx(s.grid, i, j)])).toBeLessThan(1e-6);
      }
    }
    // walled operator: K * 1 = c(x), so a uniform field is a fixed point with or without Motsch–Tadmor
    for (const mt of [false, true]) {
      const op = new NonlocalOperator(30, 20, false, false, 1 / 20, kp({ motschTadmor: mt }));
      const x = new Float64Array(600).fill(0.375);
      op.applyDelayed((i, j) => x[i + 30 * j], (i, j) => x[i + 30 * j], (i, j) => x[i + 30 * j], (i, j, v) => { x[i + 30 * j] = v; }, 1 / 60);
      for (const v of x) expect(v).toBeCloseTo(0.375, 12);
    }
  });

  it('momentum on the torus: conserved for t_self = t_nb; otherwise M + A∫_{t−t_nb}^{t−t_self} M ds is conserved (dM/dt = −A[M(t−t_self) − M(t−t_nb)])', () => {
    const run = (ts: number, tn: number, local: boolean) => {
      const Nn = 24, hh = 1 / Nn, dt = 1 / 60;
      const A = 3; // S, or 4ν/h²
      const s = new FluidSolver(Nn, Nn, local
        ? { walls: 'periodic', viscosity: (A * hh * hh) / 4, vorticity: 0, delaySelf: ts, delayNb: tn }
        : { walls: 'periodic', viscosity: 0, vorticity: 0, kernel: 'gaussian', kernelStrength: A, delaySelf: ts, delayNb: tn });
      const kick = (d: number) => { for (let j = 1; j <= Nn; j++) for (let i = 0; i <= Nn; i++) s.u[uIdx(s.grid, i, j)] += d; };
      kick(0.2);
      const M: number[] = [];
      for (let n = 0; n < 90; n++) { if (n === 30) kick(0.1); s.step(); M.push(s.meanVelocity()[0]); }
      // M(t) is a function of the time after each step; Q(t_n) = M_n + A ∫_{t_n − t_nb}^{t_n − t_self} M ds (trapezoid, M interpolated)
      const Mat = (t: number) => { const x = t / dt - 1; if (x <= 0) return x < -1e-9 && t < 30 * dt ? 0.2 : M[0]; const i = Math.floor(x); return i + 1 < M.length ? M[i] + (x - i) * (M[i + 1] - M[i]) : M[M.length - 1]; };
      const Q = (n: number) => {
        const t = (n + 1) * dt, a = t - tn, b = t - ts, K = 200;
        let I = 0; for (let k = 0; k < K; k++) { const s0 = a + ((b - a) * (k + 0.5)) / K; I += Mat(s0); }
        return M[n] + (A * I * (b - a)) / K;
      };
      return { M, Q };
    };
    for (const local of [false, true]) {
      const eq = run(0.3, 0.3, local);
      expect(eq.M[89]).toBeCloseTo(0.3, 5);
      for (const [ts, tn] of [[0.3, 0], [0, 0.3], [0.2, 0.5]]) {
        const { M, Q } = run(ts, tn, local);
        expect(Math.abs(M[89] - 0.3), `${local} ${ts}/${tn}`).toBeGreaterThan(0.005); // momentum is NOT conserved
        expect(Math.abs(Q(89) - Q(60)), `${local} ${ts}/${tn}`).toBeLessThan(0.004); // but Q is
      }
    }
  });

  it('unstable regimes with self-propulsion stay bounded and finite on every boundary; without it the speed guard holds', () => {
    // long-range, equal delays (the "Delay oscillations" setting): sustained oscillation of the overlap
    const s = new FluidSolver(64, 40, {
      walls: 'periodic', viscosity: 1e-5, vorticity: 0, kernel: 'cucker-smale', kernelRange: 0.15, kernelStrength: 4,
      delaySelf: 0.42, delayNb: 0.42, selfPropulsion: true, selfPropTau: 2, selfPropV0: 0.3,
    });
    s.seedActiveTurbulence(5, 0.05);
    const ref = Float32Array.from(s.u);
    let rr = 0; for (const q of ref) rr += q * q;
    const C: number[] = [];
    for (let n = 0; n < 1500; n++) {
      s.step();
      let c = 0; for (let k = 0; k < ref.length; k++) c += s.u[k] * ref[k];
      C.push(c / rr);
    }
    const f = fitOscillation(C, 1 / 60, 0, 25);
    expect(f.signChanges).toBeGreaterThan(20);
    expect(Math.abs(2 * Math.PI / f.omega - 1.65)).toBeLessThan(0.25);
    expect(maxAbs(s.u)).toBeLessThan(2);
    for (const w of WALLS) for (const prop of [true, false]) for (const [ts, tn, loc, ker] of [[0.6, 0.6, true, true], [0.6, 0, false, true], [0.3, 1.2, true, false]] as const) {
      const q = new FluidSolver(36, 24, {
        walls: w, viscosity: 3e-3, kernel: w === 'free-slip' ? 'gaussian' : 'cucker-smale', kernelStrength: 8, kernelMT: w === 'no-slip',
        delaySelf: ts, delayNb: tn, delayLocal: loc, delayKernel: ker, selfPropulsion: prop, selfPropTau: 0.2, vorticity: 2,
      });
      q.seedActiveTurbulence(2, 0.1);
      for (let n = 0; n < 300; n++) { if (n < 100) q.stir(5 + (n % 25), 12, 1, 0.3, 3, 0.5); q.step(); }
      const tag = `${w} ${prop} ${ts}/${tn}`;
      for (const fld of [q.u, q.v, q.pressure, ...q.dye]) for (const x of fld) expect(Number.isFinite(x), tag).toBe(true);
      expect(maxAbs(q.u), tag).toBeLessThanOrEqual(DELAY_SPEED_LIMIT * 1.5);
    }
  }, 120000);

  it('fractional Laplacian with s = 1 and delays equals the delayed local viscosity on the torus', () => {
    const nu = 2e-3, N = 24;
    const a = new FluidSolver(N, N, { walls: 'periodic', viscosity: nu, vorticity: 0, delaySelf: 0.2, delayNb: 0.1 });
    const b = new FluidSolver(N, N, { walls: 'periodic', viscosity: 0, vorticity: 0, kernel: 'fractional', kernelS: 1, kernelStrength: nu, delaySelf: 0.2, delayNb: 0.1 });
    expect(b.kernelOperators().u.diagonal()).toBeCloseTo(4 * N * N, 6);
    for (const s of [a, b]) s.seedActiveTurbulence(6, 0.1);
    for (let n = 0; n < 60; n++) { a.step(); b.step(); }
    let d = 0; for (let k = 0; k < a.u.length; k++) d = Math.max(d, Math.abs(a.u[k] - b.u[k]));
    expect(d).toBeLessThan(1e-5 * maxAbs(a.u));
  });

  it('t_self in (0, Δt] varies the result continuously', () => {
    const out = (ts: number) => {
      const s = new FluidSolver(24, 24, { walls: 'periodic', viscosity: 0, vorticity: 0, kernel: 'cucker-smale', kernelStrength: 6, delaySelf: ts, delayNb: 0.2 });
      s.seedActiveTurbulence(13, 0.1);
      for (let n = 0; n < 30; n++) s.step();
      return s.u;
    };
    const dist = (a: Float32Array, b: Float32Array) => { let d = 0; for (let k = 0; k < a.length; k++) d = Math.max(d, Math.abs(a[k] - b[k])); return d; };
    const dt = 1 / 60, ref = out(0.6 * dt), m = maxAbs(ref);
    expect(dist(out(0.61 * dt), ref)).toBeLessThan(0.02 * m);
    expect(dist(out(1e-6 * dt), out(0.5 * dt))).toBeLessThan(1e-6 * m); // midpoint read ≥ t: explicit on the current field
    expect(dist(out(1.0 * dt), out(1.001 * dt))).toBeLessThan(0.01 * m);
  });

  it('history covers max(t_self, t_nb); reset / boundary change / delays → 0 clear it; runs are reproducible', () => {
    const s = new FluidSolver(32, 24, { walls: 'periodic', viscosity: 1e-4, kernel: 'gaussian', delaySelf: 0.1, delayNb: 0.3 });
    s.seedActiveTurbulence(3, 0.1);
    for (let n = 0; n < 40; n++) s.step();
    let info = s.delayInfo();
    expect(info.local && info.kernel).toBe(true);
    expect(info.tauMax).toBe(0.3);
    expect(info.framesStored).toBeGreaterThanOrEqual(Math.floor(0.3 * 60));
    expect(info.interval * (info.capacity - 2)).toBeGreaterThanOrEqual(info.tauMax * (1 - 1e-9));
    s.setParams({ delayKernel: false });
    s.step();
    info = s.delayInfo();
    expect(info.local && !info.kernel).toBe(true);
    expect(info.framesStored).toBeGreaterThan(0); // still needed by the local term
    s.reset();
    expect(s.delayInfo().framesStored).toBe(0);
    for (let n = 0; n < 10; n++) s.step();
    s.setParams({ walls: 'no-slip' });
    expect(s.delayInfo().framesStored).toBe(0);
    for (let n = 0; n < 10; n++) s.step();
    s.setParams({ delaySelf: 0, delayNb: 0 });
    s.step();
    expect(s.delayInfo().framesStored).toBe(0);
    const run = () => { s.reset(); s.setParams({ delaySelf: 0.25, delayNb: 0.05, delayKernel: true }); s.seedActiveTurbulence(8, 0.1); for (let n = 0; n < 40; n++) s.step(); return Array.from(s.u); };
    const r1 = run();
    for (let n = 0; n < 7; n++) s.step();
    expect(run()).toEqual(r1);
    // memory cap: long delays on a big grid coarsen the sampling instead of exceeding the budget
    const big = new FluidSolver(320, 200, { walls: 'periodic', viscosity: 1e-4, delaySelf: 2, delayNb: 2 });
    big.step();
    const bi = big.delayInfo();
    expect(bi.capacity * (4 * (big.u.length + big.v.length))).toBeLessThanOrEqual(64 * 2 ** 20);
    expect(bi.capacity).toBeLessThanOrEqual(FieldHistory.MAX_FRAMES);
    expect(bi.interval * (bi.capacity - 2)).toBeGreaterThanOrEqual(2 * (1 - 1e-9));
  });

  it('stability readouts: local scan finds the checkerboard for equal delays; long-range finds the short waves', () => {
    const s = new FluidSolver(64, 64, { walls: 'periodic', viscosity: 2 / (4 * 64 * 64), kernel: 'cucker-smale', kernelRange: 0.15, kernelStrength: 4, delaySelf: 0.42, delayNb: 0.42 });
    const st = s.delayStability();
    expect(st.local!.param).toBe(-1);
    expect(st.local!.A).toBeCloseTo(2, 9);
    expect(st.kernel!.root.re).toBeCloseTo(delayRoot(4 * (1 - st.kernel!.param), 0.42).re, 6);
    expect(st.kernel!.param).toBeLessThan(0.01);
    expect(st.kernel!.regime).toBe('unstable-osc');
  });
});
