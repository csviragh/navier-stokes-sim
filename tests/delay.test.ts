import { describe, expect, it } from 'vitest';
import {
  DelayHistory, FluidSolver, NonlocalOperator, uIdx, vIdx, type KernelParams, type WallType,
} from '../src/solver';
import { KERNEL_CASES, kernelScenario } from './fingerprint';

const kp = (o: Partial<KernelParams>): KernelParams => ({
  kernel: 'cucker-smale', strength: 4, range: 0.05, beta: 1, alpha: 3, s: 0.5, motschTadmor: false, ...o,
});
function rand(n: number, seed = 3): Float64Array {
  let a = seed;
  const f = new Float64Array(n);
  for (let k = 0; k < n; k++) { a = (a * 1664525 + 1013904223) >>> 0; f[k] = a / 2 ** 32 - 0.5; }
  return f;
}
/** Run the retarded operator alone on a field x (n1 × n2), advancing t. */
function retard(op: NonlocalOperator, x: Float64Array, dt: number, t: number, vInt: number, M: number, hist: DelayHistory) {
  op.applyRetarded((i, j) => x[i + op.n1 * j], (i, j, v) => { x[i + op.n1 * j] = v; }, dt, t, vInt, M, hist);
}
function instant(op: NonlocalOperator, x: Float64Array, dt: number) {
  op.apply((i, j) => x[i + op.n1 * j], (i, j, v) => { x[i + op.n1 * j] = v; }, dt);
}
const maxAbs = (a: ArrayLike<number>) => { let m = 0; for (let k = 0; k < a.length; k++) m = Math.max(m, Math.abs(a[k])); return m; };

// fingerprints of the instantaneous kernel paths, recorded at commit dc86501
const REF_DC86501: Record<string, string> = {
  'periodic/gaussian/false': '4c00fa13', 'periodic/cucker-smale/false': '5ca0090e', 'periodic/power/false': '18101e34',
  'periodic/fractional/false': '0cfec16f', 'no-slip/cucker-smale/true': '55a9d034', 'free-slip/gaussian/false': 'd27d549f',
  'periodic-x/power/false': 'c71beb14', 'periodic-x/cucker-smale/true': '9fecc243',
};

describe('finite interaction speed (retarded kernels)', () => {
  it('delay off (and v_int = ∞) is bit-identical to dc86501 for every kernel / boundary', () => {
    for (const [w, k, mt] of KERNEL_CASES) {
      const key = `${w}/${k}/${mt}`;
      expect(kernelScenario(w, k, mt), key).toBe(REF_DC86501[key]);
      expect(kernelScenario(w, k, mt, { kernelDelay: true, kernelVInt: Infinity }), key).toBe(REF_DC86501[key]);
      // the delay settings are inert while the toggle is off
      expect(kernelScenario(w, k, mt, { kernelDelay: false, kernelVInt: 0.1, kernelShells: 3 }), key).toBe(REF_DC86501[key]);
    }
  });

  it('shells partition the kernel: K_self δ + Σ K_m = K, masses sum to 1, delays respect r_out ≥ r', () => {
    for (const per of [true, false]) for (const kernel of ['gaussian', 'cucker-smale', 'power'] as const) for (const M of [2, 6, 16]) {
      const n = 32;
      const op = new NonlocalOperator(n, 24, per, per, 1 / 24, kp({ kernel, range: 0.07, beta: 1.5, alpha: 3 }));
      const sh = op.shells(M);
      expect(sh.rOuter.length).toBe(M + 1);
      let total = sh.selfWeight;
      for (const m of sh.mass) total += m;
      expect(total).toBeCloseTo(1, 12);
      // real space: exact partition (each offset in exactly one shell)
      const sum = new Float64Array(sh.full.length);
      sum[0] = sh.selfWeight;
      for (const r of sh.real) for (let k = 0; k < r.length; k++) sum[k] += r[k];
      for (let k = 0; k < sum.length; k++) expect(sum[k]).toBe(sh.full[k]);
      // Fourier space: Σ K̂_m + K_self = K̂ (the operator's own symbol)
      let err = 0;
      for (let b = 0; b < op.B; b++) for (let a = 0; a < op.A; a++) {
        let s = sh.selfWeight;
        for (const y of sh.sym) s += y[a + op.A * b];
        err = Math.max(err, Math.abs(s - op.symbol(a, b)));
      }
      expect(err).toBeLessThan(1e-12);
      // outer radii non-decreasing; regular shells end at R_cut
      for (let m = 1; m < sh.rOuter.length; m++) expect(sh.rOuter[m]).toBeGreaterThanOrEqual(sh.rOuter[m - 1]);
      expect(sh.rOuter[M - 1]).toBeCloseTo(sh.rCut, 12);
      // every offset gets a delay ≥ its distance (except the < 1e-12 ultra-tail)
      const off = (d: number, nn: number, mm: number) => (per ? Math.min(d, nn - d) : d <= nn - 1 ? d : mm - d);
      let late = 0;
      sh.real.forEach((r, m) => {
        for (let b = 0; b < op.B; b++) for (let a = 0; a < op.A; a++) {
          const v = r[a + op.A * b];
          if (!v) continue;
          const dist = Math.hypot(off(a, op.n1, op.A), off(b, op.n2, op.B)) / 24;
          if (dist > sh.rOuter[m] * (1 + 1e-12)) late += v;
        }
      });
      expect(late).toBeLessThan(1e-12);
    }
  });

  it('v_int → ∞ reproduces the instantaneous kernel (walls: same scheme to round-off; periodic: O((SΔt)²) of the exact update)', () => {
    const n1 = 40, n2 = 28, dt = 1 / 60;
    for (const mt of [false, true]) {
      const op = new NonlocalOperator(n1, n2, false, false, 1 / n2, kp({ motschTadmor: mt }));
      const x0 = rand(n1 * n2, 5);
      const a = x0.slice(), b = x0.slice();
      const hist = new DelayHistory();
      for (let n = 0; n < 10; n++) { instant(op, a, dt); retard(op, b, dt, n * dt, 1e12, 6, hist); }
      let d = 0; for (let k = 0; k < a.length; k++) d = Math.max(d, Math.abs(a[k] - b[k]));
      expect(d).toBeLessThan(1e-13);
    }
    // periodic: the delayed path is semi-implicit, the instantaneous one exact in Fourier space
    const N = 32;
    const op = new NonlocalOperator(N, N, true, true, 1 / N, kp({ kernel: 'gaussian', range: 0.06, strength: 3 }));
    const x0 = rand(N * N, 9);
    const a = x0.slice(), b = x0.slice(), c = x0.slice();
    const hist = new DelayHistory();
    for (let n = 0; n < 10; n++) { instant(op, a, dt); retard(op, b, dt, n * dt, 1e12, 6, hist); }
    // independent semi-implicit reference with direct (real-space) convolution
    const sh = op.shells(6);
    const S = 3;
    for (let n = 0; n < 10; n++) {
      const old = c.slice();
      for (let j = 0; j < N; j++) for (let i = 0; i < N; i++) {
        let w = 0;
        for (let q = 0; q < N; q++) for (let p = 0; p < N; p++) w += sh.full[p + op.A * q] * old[((i - p + N) % N) + N * ((j - q + N) % N)];
        c[i + N * j] = (old[i + N * j] + S * dt * w) / (1 + S * dt);
      }
    }
    let dRef = 0, dExact = 0;
    for (let k = 0; k < b.length; k++) { dRef = Math.max(dRef, Math.abs(b[k] - c[k])); dExact = Math.max(dExact, Math.abs(b[k] - a[k])); }
    expect(dRef).toBeLessThan(1e-12); // float64 "now" spectrum: no float32 round-off at zero delay
    expect(dExact).toBeLessThan(10 * (S * dt) ** 2 * maxAbs(x0)); // 10 steps × O((SΔt)²) splitting difference
    expect(dExact).toBeGreaterThan(0); // (they are different schemes)
  });

  it('a uniform flow stays uniform (constant on all boundaries; time-varying on the torus)', () => {
    for (const per of [true, false]) {
      const op = new NonlocalOperator(30, 20, per, per, 1 / 20, kp({}));
      const hist = new DelayHistory();
      const x = new Float64Array(600).fill(0.375); // exactly representable in the Float32 history
      for (let n = 0; n < 60; n++) retard(op, x, 1 / 60, n / 60, 0.2, 6, hist);
      for (const v of x) expect(v).toBeCloseTo(0.375, 13);
    }
    const op = new NonlocalOperator(30, 20, true, true, 1 / 20, kp({}));
    const hist = new DelayHistory();
    const x = new Float64Array(600).fill(0);
    for (let n = 0; n < 80; n++) {
      for (let k = 0; k < x.length; k++) x[k] += 0.05 * Math.sin(n / 7); // uniform, time-dependent kick
      retard(op, x, 1 / 60, n / 60, 0.2, 6, hist);
      let lo = Infinity, hi = -Infinity;
      for (const v of x) { lo = Math.min(lo, v); hi = Math.max(hi, v); }
      expect(hi - lo).toBeLessThan(1e-6);
    }
    // solver level, periodic: uniform flow + self-propulsion + delayed kernel stays uniform at v0
    const s = new FluidSolver(32, 24, {
      walls: 'periodic', viscosity: 1e-4, vorticity: 0, kernel: 'cucker-smale', kernelDelay: true, kernelVInt: 0.3,
      selfPropulsion: true, selfPropV0: 0.5, selfPropTau: 0.3,
    });
    for (let j = 1; j <= 24; j++) for (let i = 0; i <= 32; i++) s.u[uIdx(s.grid, i, j)] = 0.2;
    // (while |u| grows the delayed alignment pulls toward the slower past, so v0 is reached later than without delay)
    for (let n = 0; n < 480; n++) s.step();
    for (let j = 1; j <= 24; j++) for (let i = 1; i <= 32; i++) {
      expect(s.u[uIdx(s.grid, i, j)]).toBeCloseTo(s.u[uIdx(s.grid, 1, 1)], 5);
      expect(Math.abs(s.v[vIdx(s.grid, i, j)])).toBeLessThan(1e-5);
    }
    expect(s.u[uIdx(s.grid, 1, 1)]).toBeGreaterThan(0.49);
  });

  it('light cone: an impulse is felt at distance r no earlier than r/v_int (periodic and walled), and is felt soon after', () => {
    for (const per of [true, false]) for (const kernel of ['cucker-smale', 'gaussian'] as const) {
      const N = 48, h = 1 / N, dt = 1 / 60, v = 0.3, M = 8;
      const op = new NonlocalOperator(N, N, per, per, h, kp({ kernel, range: kernel === 'gaussian' ? 0.12 : 0.05 }));
      const sh = op.shells(M);
      const hist = new DelayHistory();
      const x = new Float64Array(N * N);
      const src = [8, 24];
      const t0 = 5 * dt;
      // float32 spectra on periodic grids → round-off floor ~1e-8 everywhere
      const thr = per ? 1e-6 : 1e-13;
      const first = new Map<number, number>();
      const ds = [2, 4, 8, 12, 16, 20, 24, 30, 36].filter((d) => src[0] + d < N);
      let t = 0;
      for (let n = 0; n < 260; n++) {
        if (n === 5) x[src[0] + N * src[1]] = 1;
        retard(op, x, dt, t, v, M, hist);
        t += dt;
        for (const d of ds) if (!first.has(d) && Math.abs(x[src[0] + d + N * src[1]]) > thr) first.set(d, t - dt);
      }
      for (const d of ds) {
        const r = h * (per ? Math.min(d, N - d) : d);
        if (r > sh.rTail) continue;
        const arr = first.get(d);
        expect(arr, `${per} ${kernel} d=${d}`).toBeDefined();
        // causality: nothing before t0 + r/v_int, up to one history interval (= Δt here)
        // from the linear interpolation in time between stored frames
        expect(arr! - t0, `${per} ${kernel} d=${d}`).toBeGreaterThanOrEqual(r / v - dt - 1e-9);
        expect(arr! - t0, `${per} ${kernel} d=${d}`).toBeGreaterThan(0.8 * r / v);
        // ... and the signal arrives with the delay of its shell's outer radius (+ one step)
        let m = 0; while (m < sh.rOuter.length - 1 && r > sh.rOuter[m] * (1 + 1e-12)) m++;
        expect(arr! - t0, `${per} ${kernel} d=${d}`).toBeLessThanOrEqual(sh.rOuter[m] / v + dt + 1e-9);
      }
    }
  });

  it('momentum: conserved once the history is stationary; in general momentum + "in-flight" momentum is conserved', () => {
    // (a) operator level, torus: stationary mean → Σu conserved to round-off
    {
      const N = 32, op = new NonlocalOperator(N, N, true, true, 1 / N, kp({ strength: 6 }));
      const hist = new DelayHistory();
      const x = rand(N * N, 4).map((q) => q + 0.3);
      const m0 = x.reduce((p, q) => p + q, 0);
      for (let n = 0; n < 150; n++) retard(op, x, 1 / 60, n / 60, 0.2, 8, hist);
      expect(x.reduce((p, q) => p + q, 0) / m0).toBeCloseTo(1, 6);
    }
    // solver level (semi-Lagrangian advection is not exactly conservative; same tolerance as the instantaneous test)
    const s = new FluidSolver(32, 24, {
      walls: 'periodic', viscosity: 1e-4, vorticity: 0, kernel: 'cucker-smale', kernelStrength: 6,
      kernelDelay: true, kernelVInt: 0.2,
    });
    s.seedActiveTurbulence(4, 0.2);
    for (let j = 1; j <= 24; j++) for (let i = 0; i <= 32; i++) s.u[uIdx(s.grid, i, j)] += 0.3;
    const m0 = s.meanVelocity();
    for (let n = 0; n < 150; n++) s.step();
    const m1 = s.meanVelocity();
    expect(m1[0]).toBeCloseTo(m0[0], 2);
    expect(m1[1]).toBeCloseTo(m0[1], 2);
    // (b) operator level: after a kick of the mean, M itself is not conserved (the
    // retarded force is not reciprocal in time), but
    //   I_n = (1 + a) M_n + a Σ_taps ν (M_{n−ℓ} + … + M_{n−1})
    // is exactly invariant (discrete analogue of M + S Σ μ_m ∫_{t−τ_m}^t M ds).
    const N = 24, dt = 1 / 60, S = 4, a = S * dt;
    const op = new NonlocalOperator(N, N, true, true, 1 / N, kp({ strength: S }));
    const hist = new DelayHistory();
    const x = rand(N * N, 11);
    const Ms: number[] = [];
    const mean = () => x.reduce((p, q) => p + q, 0) / x.length;
    const I: number[] = [];
    for (let n = 0; n < 200; n++) {
      if (n === 120) for (let k = 0; k < x.length; k++) x[k] += 0.5; // kick after warm-up
      Ms.push(mean());
      retard(op, x, dt, n * dt, 0.5, 6, hist);
      if (n > 125) {
        let inflight = 0;
        for (const tap of op.lastTaps) {
          const l = Math.round(tap.age / dt);
          for (let k = n - l; k <= n - 1; k++) inflight += tap.weight * Ms[k];
        }
        I.push((1 + a) * Ms[n] + a * inflight);
      }
    }
    const Mafter = Ms[Ms.length - 1];
    expect(Math.abs(Mafter - Ms[120])).toBeGreaterThan(1e-3); // the mean does move
    for (const v of I) expect(v).toBeCloseTo(I[0], 6);
  });

  it('no NaNs: delay + self-propulsion + stirring, tiny v_int, large Δt, strong S, all boundaries', () => {
    const walls: WallType[] = ['periodic', 'no-slip', 'free-slip', 'periodic-x'];
    for (const w of walls) for (const [vInt, dt, S] of [[0.02, 0.05, 30], [0.3, 1 / 60, 5], [5, 0.04, 30]] as const) {
      const s = new FluidSolver(36, 24, {
        walls: w, kernel: w === 'free-slip' ? 'power' : 'cucker-smale', kernelStrength: S, kernelMT: w === 'no-slip',
        kernelDelay: true, kernelVInt: vInt, kernelShells: 8, dt,
        selfPropulsion: true, selfPropTau: 0.3, selfPropV0: 0.6, vorticity: 2,
      });
      s.seedActiveTurbulence(2, 0.1);
      for (let n = 0; n < 150; n++) {
        s.stir(5 + (n % 25), 12, 1, 0.3, 3, 0.5);
        s.step();
      }
      for (const f of [s.u, s.v, s.pressure]) for (const x of f) expect(Number.isFinite(x)).toBe(true);
      expect(maxAbs(s.u)).toBeLessThan(10);
      const info = s.delayInfo()!;
      expect(info.bytes).toBeLessThanOrEqual(2 * 32 * 2 ** 20);
      expect(info.framesStored).toBeLessThanOrEqual(info.capacity);
    }
  });

  it('memory cap: tiny v_int coarsens the history interval instead of growing the buffer', () => {
    const s = new FluidSolver(64, 48, { walls: 'periodic', kernel: 'cucker-smale', kernelDelay: true, kernelVInt: 0.05 });
    s.seedActiveTurbulence(1, 0.1);
    for (let n = 0; n < 40; n++) s.step();
    const info = s.delayInfo()!;
    expect(info.tauMax).toBeGreaterThan(10);
    expect(info.interval).toBeGreaterThanOrEqual(s.params.dt);
    expect(info.interval * (info.capacity - 2)).toBeGreaterThanOrEqual(info.tauMax * (1 - 1e-9));
    expect(info.capacity).toBeLessThanOrEqual(DelayHistory.MAX_FRAMES);
  });

  it('reset, boundary change and switching the delay off clear the history', () => {
    const s = new FluidSolver(32, 24, { walls: 'periodic', kernel: 'gaussian', kernelDelay: true, kernelVInt: 0.3 });
    s.seedActiveTurbulence(3, 0.1);
    for (let n = 0; n < 20; n++) s.step();
    expect(s.delayInfo()!.framesStored).toBeGreaterThan(5);
    s.reset();
    expect(s.delayInfo()!.framesStored).toBe(0);
    expect(s.time).toBe(0);
    for (let n = 0; n < 10; n++) s.step();
    expect(s.delayInfo()!.framesStored).toBeGreaterThan(0);
    s.setParams({ walls: 'no-slip' });
    expect(s.delayInfo()!.framesStored).toBe(0);
    for (let n = 0; n < 10; n++) s.step();
    expect(s.delayInfo()!.framesStored).toBeGreaterThan(0);
    s.setParams({ kernelDelay: false });
    s.step();
    expect(s.delayInfo()!.framesStored).toBe(0);
    // after reset the run is reproducible (no stale history leaks in)
    const run = () => { s.reset(); s.setParams({ kernelDelay: true }); s.seedActiveTurbulence(8, 0.1); for (let n = 0; n < 30; n++) s.step(); return Array.from(s.u); };
    const r1 = run();
    for (let n = 0; n < 7; n++) s.step();
    expect(run()).toEqual(r1);
  });

  it('start-up: with no history the received field is the current one (oldest frame rule)', () => {
    const op = new NonlocalOperator(20, 20, true, true, 1 / 20, kp({}));
    const x0 = rand(400, 2);
    const a = x0.slice(), b = x0.slice();
    retard(op, a, 1 / 60, 0, 0.1, 6, new DelayHistory());
    retard(op, b, 1 / 60, 0, 1e12, 6, new DelayHistory());
    for (let k = 0; k < a.length; k++) expect(a[k]).toBeCloseTo(b[k], 12);
  });
});
