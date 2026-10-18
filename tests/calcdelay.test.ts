import { describe, expect, it } from 'vitest';
import {
  DELAY_SPEED_LIMIT, DelayHistory, FluidSolver, NonlocalOperator, uIdx, vIdx, type KernelParams, type WallType,
} from '../src/solver';
import { KERNEL_CASES, kernelScenario } from './fingerprint';
import { delayRoot } from './lambert';

const kp = (o: Partial<KernelParams>): KernelParams => ({
  kernel: 'cucker-smale', strength: 4, range: 0.05, beta: 1, alpha: 3, s: 0.5, motschTadmor: false, ...o,
});
function rand(n: number, seed = 3): Float64Array {
  let a = seed;
  const f = new Float64Array(n);
  for (let k = 0; k < n; k++) { a = (a * 1664525 + 1013904223) >>> 0; f[k] = a / 2 ** 32 - 0.5; }
  return f;
}
function calc(op: NonlocalOperator, x: Float64Array, dt: number, t: number, tc: number, vInt: number, M: number, hist: DelayHistory) {
  op.applyCalcDelayed((i, j) => x[i + op.n1 * j], (i, j, v) => { x[i + op.n1 * j] = v; }, dt, t, tc, vInt, M, hist);
}
const maxAbs = (a: ArrayLike<number>) => { let m = 0; for (let k = 0; k < a.length; k++) m = Math.max(m, Math.abs(a[k])); return m; };

/** Growth rate (log-envelope slope through |peaks| after tMin) and angular frequency (zero crossings). */
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

// fingerprints of the delayed (finite v_int) paths recorded at commit 6a8ab36
const REF_6A8AB36: Record<string, string> = {
  'periodic/gaussian/false/v0.3M8': '61c0626b', 'periodic/gaussian/false/v0.05M3': 'e1d3367a',
  'periodic/cucker-smale/false/v0.3M8': '88ff4f44', 'periodic/cucker-smale/false/v0.05M3': 'db6edc7b',
  'periodic/power/false/v0.3M8': 'fa1de6b8', 'periodic/power/false/v0.05M3': '3a701ae0',
  'no-slip/cucker-smale/true/v0.3M8': 'd0f747a4', 'no-slip/cucker-smale/true/v0.05M3': 'd717b86f',
  'free-slip/gaussian/false/v0.3M8': '28875afc', 'free-slip/gaussian/false/v0.05M3': '1b32d4e7',
  'periodic-x/power/false/v0.3M8': '5725ef89', 'periodic-x/power/false/v0.05M3': 'b02108c7',
  'periodic-x/cucker-smale/true/v0.3M8': '56d21dbc', 'periodic-x/cucker-smale/true/v0.05M3': 'd6382aea',
};
const REF_DC86501: Record<string, string> = {
  'periodic/gaussian/false': '4c00fa13', 'periodic/cucker-smale/false': '5ca0090e', 'periodic/power/false': '18101e34',
  'periodic/fractional/false': '0cfec16f', 'no-slip/cucker-smale/true': '55a9d034', 'free-slip/gaussian/false': 'd27d549f',
  'periodic-x/power/false': 'c71beb14', 'periodic-x/cucker-smale/true': '9fecc243',
};

describe('calculation delay t_c (own + neighbour velocity delayed)', () => {
  it('t_c = 0 is bit-identical to 6a8ab36 in every mode (instant, finite v_int, all boundaries)', () => {
    for (const [w, k, mt] of KERNEL_CASES) {
      const key = `${w}/${k}/${mt}`;
      expect(kernelScenario(w, k, mt, { kernelCalcDelay: 0 }), key).toBe(REF_DC86501[key]);
      if (k === 'fractional') {
        // fractional ignores both delays
        expect(kernelScenario(w, k, mt, { kernelCalcDelay: 0.3, kernelDelay: true, kernelVInt: 0.2 }), key).toBe(REF_DC86501[key]);
        continue;
      }
      expect(kernelScenario(w, k, mt, { kernelCalcDelay: 0, kernelDelay: true, kernelVInt: 0.3, kernelShells: 8 }))
        .toBe(REF_6A8AB36[`${key}/v0.3M8`]);
      expect(kernelScenario(w, k, mt, { kernelCalcDelay: 0, kernelDelay: true, kernelVInt: 0.05, kernelShells: 3 }))
        .toBe(REF_6A8AB36[`${key}/v0.05M3`]);
    }
  });

  it('single Fourier mode, v_int = ∞: monotone (< 1/e), damped oscillation (1/e … π/2), growth (> π/2) at the Lambert-W rate and frequency', () => {
    const N = 32, dt = 1 / 120, td = 0.3, b = 6;
    const run = (aT: number) => {
      const op = new NonlocalOperator(N, N, true, true, 1 / N, kp({ kernel: 'gaussian', range: 0.05 }));
      const Kh = op.symbol(0, b);
      const a = aT / td;
      op.params.strength = a / (1 - Kh); // a_k = S (1 − K̂(k))
      const x = new Float64Array(N * N);
      const mode = (j: number) => Math.cos((2 * Math.PI * b * j) / N);
      for (let j = 0; j < N; j++) for (let i = 0; i < N; i++) x[i + N * j] = mode(j);
      const hist = new DelayHistory();
      const A: number[] = [];
      for (let n = 0; n < 1200; n++) {
        calc(op, x, dt, n * dt, td, Infinity, 8, hist);
        let s = 0; for (let j = 0; j < N; j++) s += x[N * j] * mode(j);
        A.push(s / (N / 2));
        // the mode stays a pure mode (other modes are not excited)
        if (n % 200 === 0) for (let i = 0; i < N; i++) expect(x[i + N * 3]).toBeCloseTo(x[N * 3], 5 + Math.max(0, -Math.log10(Math.abs(A[n]) + 1e-30)) | 0);
      }
      return { A, root: delayRoot(a, td) };
    };
    // (i) a t_d = 0.25 < 1/e: monotone decay at the real root W₀(−a t_d)/t_d
    {
      const { A, root } = run(0.25);
      expect(root.im).toBe(0);
      for (let n = 1; n < A.length; n++) { expect(A[n]).toBeGreaterThan(0); expect(A[n]).toBeLessThan(A[n - 1]); }
      const rate = Math.log(A[1100] / A[600]) / (500 * dt);
      expect(rate).toBeCloseTo(root.re, 2);
    }
    // (ii) a t_d = 1.0 ∈ (1/e, π/2): damped oscillation
    {
      const { A, root } = run(1.0);
      const f = fitOscillation(A, dt, 0.5, 6);
      expect(f.signChanges).toBeGreaterThan(4);
      expect(root.re).toBeLessThan(0);
      expect(f.mu / root.re).toBeGreaterThan(0.97); expect(f.mu / root.re).toBeLessThan(1.03);
      expect(f.omega / root.im).toBeGreaterThan(0.99); expect(f.omega / root.im).toBeLessThan(1.01);
    }
    // (iii) a t_d = 1.8 > π/2: growing oscillation
    {
      const { A, root } = run(1.8);
      const f = fitOscillation(A, dt, 2);
      expect(root.re).toBeGreaterThan(0);
      expect(maxAbs(A.slice(-200))).toBeGreaterThan(5 * maxAbs(A.slice(0, 200)));
      expect(f.mu / root.re).toBeGreaterThan(0.97); expect(f.mu / root.re).toBeLessThan(1.03);
      expect(f.omega / root.im).toBeGreaterThan(0.99); expect(f.omega / root.im).toBeLessThan(1.01);
    }
  });

  it('full solver, shear wave (exact NS solution): onset at a_k t_c = π/2, growth rate/frequency from λ = −a_k e^{−λ t_c}', () => {
    const nx = 48, ny = 48, m = 4, S = 4;
    const run = (tc: number, steps = 900) => {
      const s = new FluidSolver(nx, ny, {
        walls: 'periodic', viscosity: 0, vorticity: 0, kernel: 'cucker-smale', kernelRange: 0.15, kernelStrength: S, kernelCalcDelay: tc,
      });
      const Kh = s.kernelOperators().u.symbol(0, m);
      const prof = (j: number) => Math.sin((2 * Math.PI * m * (j - 0.5)) / ny);
      for (let j = 1; j <= ny; j++) for (let i = 0; i <= nx; i++) s.u[uIdx(s.grid, i, j)] = 0.02 * prof(j);
      const A: number[] = [];
      for (let n = 0; n < steps; n++) {
        s.step();
        let q = 0; for (let j = 1; j <= ny; j++) q += s.u[uIdx(s.grid, 5, j)] * prof(j);
        A.push(q / (ny / 2));
      }
      return { A, a: S * (1 - Kh) };
    };
    const { a } = run(0, 0);
    const tcCrit = Math.PI / 2 / a;
    expect(tcCrit).toBeGreaterThan(0.39); expect(tcCrit).toBeLessThan(0.4);
    for (const tc of [0.3, 0.38, 0.41, 0.45]) {
      const { A } = run(tc);
      const root = delayRoot(a, tc);
      const f = fitOscillation(A, 1 / 60, 2);
      expect(Math.sign(f.mu), `tc=${tc}`).toBe(tc > tcCrit ? 1 : -1);
      expect(Math.abs(f.mu - root.re), `tc=${tc}`).toBeLessThan(0.01 + 0.02 * Math.abs(root.re));
      expect(Math.abs(f.omega / root.im - 1), `tc=${tc}`).toBeLessThan(0.01);
    }
  }, 120000);

  it('combined delays: shear wave follows λ = −S e^{−λt_c}(1 − K(0) − Σ_m K̂_m e^{−λ r_m/v_int}); finite v_int can push a stable t_c over the edge', () => {
    const nx = 48, ny = 48, m = 4, S = 4;
    for (const [tc, vInt, grows] of [[0.38, 0.1, true], [0.3, 0.1, false], [0.41, 0.3, true]] as const) {
      const s = new FluidSolver(nx, ny, {
        walls: 'periodic', viscosity: 0, vorticity: 0, kernel: 'cucker-smale', kernelRange: 0.15, kernelStrength: S,
        kernelCalcDelay: tc, kernelDelay: true, kernelVInt: vInt, kernelShells: 8,
      });
      const op = s.kernelOperators().u;
      const sh = op.shells(8);
      const q0 = op.A * m; // Fourier index (0, m)
      // complex Newton from the v_int = ∞ root
      const r0 = delayRoot(S * (1 - op.symbol(0, m)), tc);
      let lr = r0.re, li = r0.im;
      const F = (ar: number, ai: number): [number, number] => {
        let gr = 1 - sh.selfWeight, gi = 0;
        for (let q = 0; q < sh.rOuter.length; q++) {
          if (!(sh.mass[q] > 0)) continue;
          const tau = sh.rOuter[q] / vInt, e = Math.exp(-ar * tau);
          gr -= sh.sym[q][q0] * e * Math.cos(-ai * tau); gi -= sh.sym[q][q0] * e * Math.sin(-ai * tau);
        }
        const e = Math.exp(-ar * tc), er = e * Math.cos(-ai * tc), ei = e * Math.sin(-ai * tc);
        return [ar + S * (er * gr - ei * gi), ai + S * (er * gi + ei * gr)];
      };
      for (let it = 0; it < 60; it++) {
        const [fr, fi] = F(lr, li), hh = 1e-7, [br, bi] = F(lr + hh, li);
        const dr = (br - fr) / hh, di = (bi - fi) / hh, den = dr * dr + di * di;
        lr -= (fr * dr + fi * di) / den; li -= (fi * dr - fr * di) / den;
      }
      const prof = (j: number) => Math.sin((2 * Math.PI * m * (j - 0.5)) / ny);
      for (let j = 1; j <= ny; j++) for (let i = 0; i <= nx; i++) s.u[uIdx(s.grid, i, j)] = 0.02 * prof(j);
      const A: number[] = [];
      for (let n = 0; n < 900; n++) {
        s.step();
        let q = 0; for (let j = 1; j <= ny; j++) q += s.u[uIdx(s.grid, 5, j)] * prof(j);
        A.push(q / (ny / 2));
      }
      const f = fitOscillation(A, 1 / 60, 2);
      expect(lr > 0, `${tc}/${vInt}`).toBe(grows);
      expect(Math.sign(f.mu), `${tc}/${vInt}`).toBe(grows ? 1 : -1);
      expect(Math.abs(f.mu - lr), `${tc}/${vInt}`).toBeLessThan(0.01 + 0.03 * Math.abs(lr));
      expect(Math.abs(f.omega / Math.abs(li) - 1), `${tc}/${vInt}`).toBeLessThan(0.01);
      if (tc === 0.38) expect(r0.re).toBeLessThan(0); // stable with v_int = ∞
    }
  }, 120000);

  it('v_int = ∞ single-tap path agrees with the shell path at v_int → ∞ (periodic and walled, MT on/off)', () => {
    for (const per of [true, false]) for (const mt of [false, true]) {
      const op = new NonlocalOperator(30, 22, per, per, 1 / 22, kp({ motschTadmor: mt, range: 0.08 }));
      const x0 = rand(30 * 22, 7);
      const a = x0.slice(), b = x0.slice();
      const ha = new DelayHistory(), hb = new DelayHistory();
      for (let n = 0; n < 60; n++) { calc(op, a, 1 / 60, n / 60, 0.2, Infinity, 6, ha); calc(op, b, 1 / 60, n / 60, 0.2, 1e12, 6, hb); }
      let d = 0; for (let k = 0; k < a.length; k++) d = Math.max(d, Math.abs(a[k] - b[k]));
      expect(d).toBeLessThan(1e-6 * maxAbs(x0));
    }
  });

  it('a uniform flow is unchanged (both delays, all boundaries)', () => {
    for (const per of [true, false]) for (const vInt of [Infinity, 0.2]) {
      const op = new NonlocalOperator(30, 20, per, per, 1 / 20, kp({}));
      const hist = new DelayHistory();
      const x = new Float64Array(600).fill(0.375);
      for (let n = 0; n < 80; n++) calc(op, x, 1 / 60, n / 60, 0.45, vInt, 6, hist);
      for (const v of x) expect(v).toBeCloseTo(0.375, 12);
    }
    const s = new FluidSolver(32, 24, {
      walls: 'periodic', viscosity: 1e-4, vorticity: 0, kernel: 'cucker-smale', kernelCalcDelay: 0.5, kernelStrength: 4,
      kernelDelay: true, kernelVInt: 0.3,
    });
    for (let j = 1; j <= 24; j++) for (let i = 0; i <= 32; i++) s.u[uIdx(s.grid, i, j)] = 0.25;
    for (let n = 0; n < 200; n++) s.step();
    for (let j = 1; j <= 24; j++) for (let i = 1; i <= 32; i++) {
      expect(s.u[uIdx(s.grid, i, j)]).toBeCloseTo(0.25, 6);
      expect(Math.abs(s.v[vIdx(s.grid, i, j)])).toBeLessThan(1e-6);
    }
  });

  it('causality: an impulse is felt at distance r no earlier than t_c + r/v_int − Δt (own response after t_c)', () => {
    for (const per of [true, false]) {
      const N = 48, h = 1 / N, dt = 1 / 60, v = 0.3, M = 8, tc = 0.25;
      const op = new NonlocalOperator(N, N, per, per, h, kp({ range: 0.05 }));
      const sh = op.shells(M);
      const hist = new DelayHistory();
      const x = new Float64Array(N * N);
      const src = [8, 24], n0 = 5, t0 = n0 * dt;
      const thr = per ? 1e-6 : 1e-13;
      const ds = [0, 2, 4, 8, 12, 16, 24, 30].filter((d) => src[0] + d < N);
      const first = new Map<number, number>();
      for (let n = 0; n < 300; n++) {
        if (n === n0) x[src[0] + N * src[1]] = 1;
        const before = x[src[0] + N * src[1]];
        calc(op, x, dt, n * dt, tc, v, M, hist);
        for (const d of ds) {
          if (first.has(d)) continue;
          const val = x[src[0] + d + N * src[1]];
          const changed = d === 0 ? n >= n0 && Math.abs(val - before) > thr : Math.abs(val) > thr;
          if (changed) first.set(d, n * dt);
        }
      }
      for (const d of ds) {
        const r = h * (per ? Math.min(d, N - d) : d);
        if (r > sh.rTail) continue;
        const arr = first.get(d);
        expect(arr, `${per} d=${d}`).toBeDefined();
        expect(arr! - t0, `${per} d=${d}`).toBeGreaterThanOrEqual(tc + r / v - dt - 1e-9);
        let m = 0; while (r > 0 && m < sh.rOuter.length - 1 && r > sh.rOuter[m] * (1 + 1e-12)) m++;
        expect(arr! - t0, `${per} d=${d}`).toBeLessThanOrEqual(tc + (r > 0 ? sh.rOuter[m] / v : 0) + dt + 1e-9);
      }
    }
  });

  it('momentum on the torus is conserved once the history is stationary', () => {
    const N = 32, op = new NonlocalOperator(N, N, true, true, 1 / N, kp({ strength: 4 }));
    for (const vInt of [Infinity, 0.4]) {
      const hist = new DelayHistory();
      const x = rand(N * N, 4).map((q) => q + 0.3);
      const m0 = x.reduce((p, q) => p + q, 0);
      for (let n = 0; n < 200; n++) calc(op, x, 1 / 60, n / 60, 0.3, vInt, 8, hist);
      expect(x.reduce((p, q) => p + q, 0) / m0).toBeCloseTo(1, 6);
    }
    const s = new FluidSolver(32, 24, {
      walls: 'periodic', viscosity: 1e-4, vorticity: 0, kernel: 'cucker-smale', kernelStrength: 4, kernelCalcDelay: 0.3,
    });
    // (weak seed: semi-Lagrangian advection itself is not exactly conservative)
    s.seedActiveTurbulence(4, 0.05);
    for (let j = 1; j <= 24; j++) for (let i = 0; i <= 32; i++) s.u[uIdx(s.grid, i, j)] += 0.3;
    const m0 = s.meanVelocity();
    for (let n = 0; n < 150; n++) s.step();
    const m1 = s.meanVelocity();
    expect(m1[0]).toBeCloseTo(m0[0], 2);
    expect(m1[1]).toBeCloseTo(m0[1], 2);
  });

  it('oscillation regime + self-propulsion: the instability grows but saturates; no NaNs on any boundary', () => {
    // overlap C(t) = <u, u_seed>/<u_seed, u_seed>: growing sign-changing oscillation, then a bounded limit cycle
    const s = new FluidSolver(64, 40, {
      walls: 'periodic', viscosity: 1e-5, vorticity: 0, kernel: 'cucker-smale', kernelRange: 0.15, kernelStrength: 4,
      kernelCalcDelay: 0.42, selfPropulsion: true, selfPropTau: 2, selfPropV0: 0.3,
    });
    s.seedActiveTurbulence(5, 0.05);
    const ref = Float32Array.from(s.u);
    let rr = 0; for (const q of ref) rr += q * q;
    const C: number[] = [], speed: number[] = [];
    for (let n = 0; n < 1500; n++) {
      s.step();
      let c = 0; for (let k = 0; k < ref.length; k++) c += s.u[k] * ref[k];
      C.push(c / rr);
      let sp = 0; for (let j = 1; j <= 40; j++) for (let i = 1; i <= 64; i++) sp += Math.abs(s.u[uIdx(s.grid, i, j)]) + Math.abs(s.v[vIdx(s.grid, i, j)]);
      speed.push(sp / (64 * 40));
    }
    const f = fitOscillation(C, 1 / 60, 0, 25);
    expect(f.signChanges).toBeGreaterThan(20); // ≈ 2 per period of ≈ 1.65 s
    expect(Math.abs(2 * Math.PI / f.omega - 1.65)).toBeLessThan(0.25);
    const early = Math.max(...speed.slice(0, 60)), late = Math.min(...speed.slice(1200));
    expect(late).toBeGreaterThan(3 * early); // grew (the seed is ≈ 0.05)
    for (const x of s.u) expect(Number.isFinite(x)).toBe(true);
    expect(maxAbs(s.u)).toBeLessThan(2); // saturated by self-propulsion, far below the speed guard
    // all boundaries, strongly unstable, with and without propulsion: finite, guard respected
    const walls: WallType[] = ['periodic', 'no-slip', 'free-slip', 'periodic-x'];
    for (const w of walls) for (const prop of [true, false]) {
      const q = new FluidSolver(36, 24, {
        walls: w, kernel: w === 'free-slip' ? 'gaussian' : 'cucker-smale', kernelStrength: 8, kernelMT: w === 'no-slip',
        kernelCalcDelay: 0.6, kernelDelay: w !== 'periodic', kernelVInt: 0.5, selfPropulsion: prop, vorticity: 2,
      });
      q.seedActiveTurbulence(2, 0.1);
      for (let n = 0; n < 300; n++) { if (n < 100) q.stir(5 + (n % 25), 12, 1, 0.3, 3, 0.5); q.step(); }
      for (const fld of [q.u, q.v, q.pressure]) for (const x of fld) expect(Number.isFinite(x), `${w} ${prop}`).toBe(true);
      expect(maxAbs(q.u)).toBeLessThanOrEqual(DELAY_SPEED_LIMIT * 1.5);
    }
  }, 120000);

  it('t_c between 0 and Δt varies the result continuously', () => {
    const N = 24, op = new NonlocalOperator(N, N, true, true, 1 / N, kp({ strength: 6 }));
    const x0 = rand(N * N, 13);
    const out = (tc: number) => { const x = x0.slice(); const h = new DelayHistory(); for (let n = 0; n < 30; n++) calc(op, x, 1 / 60, n / 60, tc, Infinity, 6, h); return x; };
    const dist = (a: Float64Array, b: Float64Array) => { let d = 0; for (let k = 0; k < a.length; k++) d = Math.max(d, Math.abs(a[k] - b[k])); return d; };
    const dt = 1 / 60;
    const ref = out(0.6 * dt);
    expect(dist(out(0.61 * dt), ref)).toBeLessThan(0.02 * maxAbs(x0));
    expect(dist(out(1e-6 * dt), out(0.5 * dt))).toBeLessThan(1e-9); // midpoint lookup ≥ t: explicit Euler on the current field
    expect(dist(out(1.0 * dt), out(1.001 * dt))).toBeLessThan(0.01 * maxAbs(x0));
  });

  it('history covers t_c + r_tail/v_int; reset / boundary change / t_c → 0 clear it; runs are reproducible', () => {
    const s = new FluidSolver(32, 24, { walls: 'periodic', kernel: 'gaussian', kernelCalcDelay: 0.3, kernelDelay: true, kernelVInt: 0.5 });
    s.seedActiveTurbulence(3, 0.1);
    for (let n = 0; n < 40; n++) s.step();
    let info = s.delayInfo()!;
    expect(info.tCalc).toBe(0.3);
    expect(info.tauMax).toBeCloseTo(0.3 + info.rTail / 0.5, 12);
    expect(info.framesStored).toBeGreaterThan(19);
    expect(info.interval * (info.capacity - 2)).toBeGreaterThanOrEqual(info.tauMax * (1 - 1e-9));
    s.setParams({ kernelDelay: false });
    s.step();
    info = s.delayInfo()!;
    expect(info.tauMax).toBe(0.3); // v_int off: only the calculation delay
    expect(info.framesStored).toBeGreaterThan(0); // history kept (still needed)
    s.reset();
    expect(s.delayInfo()!.framesStored).toBe(0);
    for (let n = 0; n < 10; n++) s.step();
    s.setParams({ walls: 'no-slip' });
    expect(s.delayInfo()!.framesStored).toBe(0);
    for (let n = 0; n < 10; n++) s.step();
    s.setParams({ kernelCalcDelay: 0 });
    s.step();
    expect(s.delayInfo()!.framesStored).toBe(0);
    const run = () => { s.reset(); s.setParams({ kernelCalcDelay: 0.3 }); s.seedActiveTurbulence(8, 0.1); for (let n = 0; n < 40; n++) s.step(); return Array.from(s.u); };
    const r1 = run();
    for (let n = 0; n < 7; n++) s.step();
    expect(run()).toEqual(r1);
  });
});
