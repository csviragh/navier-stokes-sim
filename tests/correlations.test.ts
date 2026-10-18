import { describe, expect, it } from 'vitest';
import { OrderAnalyzer, FluidSolver, uIdx, vIdx } from '../src/solver';

function lcg(seed: number) { let s = seed >>> 0; return () => { s = (Math.imul(s, 1664525) + 1013904223) >>> 0; return s / 4294967296; }; }

/** Random unit directions, a fraction `holes` of cells at zero speed (excluded). */
function randomField(nx: number, ny: number, seed: number, holes = 0) {
  const r = lcg(seed); const u = new Float64Array(nx * ny), v = new Float64Array(nx * ny);
  for (let k = 0; k < nx * ny; k++) {
    if (r() < holes) continue;
    const a = 2 * Math.PI * r(), s = 0.5 + r();
    u[k] = s * Math.cos(a); v[k] = s * Math.sin(a);
  }
  return { u, v, get: (i: number, j: number): [number, number] => [u[i + nx * j], v[i + nx * j]] };
}

function brute(nx: number, ny: number, get: (i: number, j: number) => [number, number], px: boolean, py: boolean) {
  const rMax = Math.floor(Math.min(nx, ny) / 2);
  const dirs: [number, number, number, number][] = [];
  let sp = 0;
  for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) sp += Math.hypot(...get(i, j));
  const thr = 0.02 * sp / (nx * ny);
  for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) {
    const [a, b] = get(i, j); const s = Math.hypot(a, b);
    if (s > thr) dirs.push([i, j, a / s, b / s]);
  }
  const n = dirs.length; let mx = 0, my = 0;
  for (const d of dirs) { mx += d[2]; my += d[3]; }
  mx /= n; my /= n;
  const num = new Float64Array(rMax + 1), numc = new Float64Array(rMax + 1), cnt = new Float64Array(rMax + 1);
  let sumCos = 0;
  for (const p of dirs) for (const q of dirs) {
    let dx = q[0] - p[0], dy = q[1] - p[1];
    if (px) { dx = ((dx % nx) + nx) % nx; if (dx > nx / 2) dx -= nx; }
    if (py) { dy = ((dy % ny) + ny) % ny; if (dy > ny / 2) dy -= ny; }
    const c = p[2] * q[2] + p[3] * q[3];
    if (p !== q) sumCos += c;
    const d = Math.round(Math.hypot(dx, dy));
    if (d > rMax) continue;
    cnt[d]++; num[d] += c; numc[d] += (p[2] - mx) * (q[2] - mx) + (p[3] - my) * (q[3] - my);
  }
  const c0 = numc[0] / cnt[0];
  return { n, meanCos: sumCos / (n * (n - 1)), C: [...num].map((x, d) => x / cnt[d]), Cc: [...numc].map((x, d) => x / cnt[d] / c0) };
}

describe('order parameters: pairwise alignment P(cos θ_ij) and C(r)', () => {
  it('uniform flow: P concentrated at cos θ = 1, ⟨cos θ⟩ = 1, C(r) = 1', () => {
    const an = new OrderAnalyzer();
    for (const [px, py] of [[true, true], [false, false], [true, false]]) {
      const st = an.analyze(20, 14, () => [0.3, -0.2], { periodicX: px, periodicY: py });
      expect(st.n).toBe(280);
      expect(st.meanCos).toBeCloseTo(1, 12);
      expect(st.phiDir).toBeCloseTo(1, 12);
      expect(st.hist[st.hist.length - 1] * (2 / st.hist.length)).toBeCloseTo(1, 12);
      for (const c of st.C) expect(c).toBeCloseTo(1, 10);
      expect(Number.isNaN(st.xi0)).toBe(true); // no fluctuations, ξ undefined
    }
  });

  it('random directions: ⟨cos θ⟩ ≈ 0, P symmetric (arcsine law 1/(π√(1−c²))), C(r > 0) ≈ 0', () => {
    const nx = 96, ny = 64; const f = randomField(nx, ny, 7);
    const st = new OrderAnalyzer().analyze(nx, ny, f.get, { periodicX: true, periodicY: true, pairs: 200000, seed: 3 });
    const N = nx * ny;
    expect(Math.abs(st.meanCos)).toBeLessThan(4 / N);          // |Σv̂|² ~ N ⇒ mean ~ 1/N
    expect(Math.abs(st.sampledMeanCos)).toBeLessThan(0.01);
    const B = st.hist.length, w = 2 / B;
    for (let k = 0; k < B / 2; k++) expect(Math.abs(st.hist[k] - st.hist[B - 1 - k])).toBeLessThan(0.08 * st.hist[k] + 0.02);
    for (let k = 2; k < B - 2; k++) {
      const a = -1 + k * w, b = a + w;
      const exact = (Math.asin(b) - Math.asin(a)) / Math.PI / w; // arcsine density, bin average
      expect(Math.abs(st.hist[k] / exact - 1)).toBeLessThan(0.08);
    }
    for (let d = 1; d < st.C.length; d++) expect(Math.abs(st.C[d])).toBeLessThan(0.1);
    expect(st.C[0]).toBeCloseTo(1, 12);
    expect(st.xi0).toBeLessThan(1);
  });

  it('FFT C(r) and connected C(r) match brute force (periodic, walled, channel; with excluded cells)', () => {
    const an = new OrderAnalyzer();
    for (const [nx, ny, px, py, holes] of [[12, 10, true, true, 0.15], [13, 10, true, true, 0], [11, 9, false, false, 0.15], [11, 9, false, false, 0], [12, 9, true, false, 0.15], [10, 12, false, true, 0.15]] as const) {
      // partially ordered field so C(r) is non-trivial
      const r = lcg(nx * 31 + ny);
      const u = new Float64Array(nx * ny), v = new Float64Array(nx * ny);
      for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) {
        if (r() < holes) continue;
        const a = 0.6 * Math.sin((2 * Math.PI * j) / ny) + 1.2 * (r() - 0.5);
        u[i + nx * j] = Math.cos(a); v[i + nx * j] = Math.sin(a);
      }
      const get = (i: number, j: number): [number, number] => [u[i + nx * j], v[i + nx * j]];
      const st = an.analyze(nx, ny, get, { periodicX: px, periodicY: py });
      const bf = brute(nx, ny, get, px, py);
      expect(st.n).toBe(bf.n);
      expect(st.meanCos).toBeCloseTo(bf.meanCos, 12);
      for (let d = 0; d < bf.C.length; d++) {
        expect(st.C[d], `${nx}x${ny} C(${d})`).toBeCloseTo(bf.C[d], 10);
        expect(st.Cconn[d], `${nx}x${ny} Cc(${d})`).toBeCloseTo(bf.Cc[d], 9);
      }
    }
  });

  it('exact mean (|Σv̂|² − N)/(N(N − 1)) vs the sampled pair mean and the histogram mean', () => {
    const an = new OrderAnalyzer();
    for (const bias of [0, 0.5, 1.5]) {
      const nx = 64, ny = 48, r = lcg(11 + bias * 10);
      const u = new Float64Array(nx * ny), v = new Float64Array(nx * ny);
      for (let k = 0; k < nx * ny; k++) { const a = bias * 0 + (r() - 0.5) * (2 * Math.PI) / (1 + bias); u[k] = Math.cos(a); v[k] = Math.sin(a); }
      const st = an.analyze(nx, ny, (i, j) => [u[i + nx * j], v[i + nx * j]], { pairs: 100000, seed: 5, correlation: false });
      expect(Math.abs(st.sampledMeanCos - st.meanCos)).toBeLessThan(0.012);
      const w = 2 / st.hist.length; let hm = 0;
      st.hist.forEach((p, k) => { hm += p * w * (-1 + (k + 0.5) * w); });
      expect(Math.abs(hm - st.meanCos)).toBeLessThan(0.02);
      // ⟨cos⟩ = φ_dir² − (1 − φ_dir²)/(N − 1)
      expect(st.meanCos).toBeCloseTo(st.phiDir ** 2 - (1 - st.phiDir ** 2) / (st.n - 1), 12);
    }
  });

  it('slow cells are excluded; a vortex has ⟨cos⟩ ≈ 0 and a finite correlation length', () => {
    const nx = 64, ny = 64;
    // single periodic vortex cell pattern u = sin(2πy), v = −sin(2πx): Taylor–Green-like
    const get = (i: number, j: number): [number, number] => {
      const x = (i + 0.5) / nx, y = (j + 0.5) / ny;
      return [Math.sin(2 * Math.PI * y), -Math.sin(2 * Math.PI * x)];
    };
    const st = new OrderAnalyzer().analyze(nx, ny, get, { periodicX: true, periodicY: true });
    expect(Math.abs(st.meanCos)).toBeLessThan(0.01);
    expect(st.xi0).toBeGreaterThan(5); expect(st.xi0).toBeLessThan(25);
    expect(st.xiE).toBeLessThan(st.xi0);
    // cells below 2 % of the mean speed carry no direction and are dropped
    const slow = new OrderAnalyzer().analyze(10, 10, (i) => (i < 3 ? [1e-4, 1e-4] : [1, 0]));
    expect(slow.excluded).toBe(30);
    expect(slow.meanCos).toBeCloseTo(1, 12);
  });

  it('works on the solver layout (cell-centred uc / vc with ghost ring)', () => {
    const s = new FluidSolver(32, 24, { walls: 'periodic' });
    const g = s.grid;
    for (let j = 0; j < g.ny + 2; j++) for (let i = 0; i <= g.nx; i++) s.u[uIdx(g, i, j)] = 0.4;
    for (let j = 0; j <= g.ny; j++) for (let i = 0; i < g.nx + 2; i++) s.v[vIdx(g, i, j)] = 0.1;
    s.updateDiagnostics();
    const st = new OrderAnalyzer().analyzeCells(g.nx, g.ny, g.stride, s.uc, s.vc, { periodicX: true, periodicY: true });
    expect(st.n).toBe(32 * 24);
    expect(st.meanCos).toBeCloseTo(1, 6);
  });
});
