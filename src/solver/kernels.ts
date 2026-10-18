/**
 * Nonlocal (long-range) viscosity / velocity-alignment operators.
 *
 *   f(x) = S ∫ K(|x − y|) (u(y) − u(x)) dy          (convolution kernels)
 *   f    = −ν_s (−Δ)^s u                             (fractional Laplacian)
 *
 * Convolution kernels are normalised so that ∫K dA = 1 (discretely: Σ K h² = 1
 * over all offsets that fit in the domain). The strength S is then a rate (1/s):
 * a velocity relaxes toward its K-weighted neighbourhood average at rate S. For
 * a Gaussian of range R the small-R limit is S·(R²/2)·∇²u, i.e. ν_eff = S R²/2.
 *
 *   gaussian      K ∝ exp(−r²/(2R²))
 *   cucker-smale  K ∝ (1 + r²/R²)^(−β)          (Cucker & Smale 2007)
 *   power         K ∝ (r² + R²)^(−α/2)           (power law, softened at r ≲ R)
 *   fractional    symbol ν_s·λ(k)^s, λ = discrete −Δ symbol (s = 1: local ν∇²)
 *
 * Each velocity component is treated on its own staggered sub-grid (u on
 * u-faces, v on v-faces), only over the faces that are unknowns.
 *
 * Periodic axes use a circular FFT of the exact period (any length; Bluestein),
 * kernels use minimum-image distance. On walled axes the field is zero-padded
 * to a power of two ≥ 2n−1, so the convolution is linear and never wraps across
 * a wall; the integral then runs over the domain only and the local term uses
 * c(x) = (K * 1_domain)(x) instead of ∫K.
 *
 * Time integration:
 *   fully periodic: the operator is diagonal in Fourier space,
 *     λ(k) = S (K̂(k) − K̂(0)) ≤ 0, applied exactly as û ← û·exp(λ Δt);
 *   otherwise (walls in at least one direction): semi-implicit
 *     u ← (u + a·(K*u)) / (1 + a·c),  a = S Δt  [Motsch–Tadmor: a = S Δt / c],
 *     convolution explicit, local part implicit; the new value is a convex
 *     combination of old values, so it obeys a max principle for any Δt.
 *   fractional: periodic only (exact multiplier exp(−ν_s λ^s Δt)); on walled
 *     domains it is disabled (no-op).
 */
import { FFT } from './fft';

export type KernelType = 'none' | 'gaussian' | 'cucker-smale' | 'power' | 'fractional';
export const KERNEL_TYPES: readonly KernelType[] = ['none', 'gaussian', 'cucker-smale', 'power', 'fractional'];

export interface KernelParams {
  kernel: KernelType;
  /** S (1/s) for convolution kernels; ν_s (L^{2s}/s) for the fractional Laplacian. */
  strength: number;
  /** Range / softening length R (domain units). */
  range: number;
  /** Cucker–Smale exponent β. */
  beta: number;
  /** Power-law exponent α. */
  alpha: number;
  /** Fractional order s ∈ (0, 1]. */
  s: number;
  /** Motsch–Tadmor local normalisation (divide by c(x) = ∫_domain K). */
  motschTadmor: boolean;
}

/** Unnormalised radial profile K(r) (r, R in domain units). */
export function kernelProfile(p: Pick<KernelParams, 'kernel' | 'range' | 'beta' | 'alpha'>, r: number): number {
  const R = Math.max(p.range, 1e-9);
  switch (p.kernel) {
    case 'gaussian': return Math.exp(-(r * r) / (2 * R * R));
    case 'cucker-smale': return (1 + (r * r) / (R * R)) ** -p.beta;
    case 'power': return (r * r + R * R) ** (-p.alpha / 2);
    default: return 0;
  }
}

const nextPow2 = (n: number): number => { let m = 1; while (m < n) m <<= 1; return m; };

/** Spectral nonlocal operator on one n1 × n2 sub-grid with given per-axis periodicity. */
export class NonlocalOperator {
  readonly A: number;
  readonly B: number;
  readonly fullyPeriodic: boolean;
  private readonly f1: FFT;
  private readonly f2: FFT;
  /** Shell decompositions for the retarded (finite-speed) operator, keyed by shell count. */
  private shellCache = new Map<number, ShellSet>();
  private wre?: Float64Array;
  private wim?: Float64Array;
  /** Diagnostics of the last retarded apply: (age of interpolated time, weight) per tap. */
  lastTaps: { age: number; weight: number }[] = [];
  /** Fourier symbol: K̂ (conv, K̂(0) = 1) or λ^s (fractional). */
  private readonly sym: Float64Array;
  private readonly re: Float64Array;
  private readonly im: Float64Array;
  private readonly r1re: Float64Array; private readonly r1im: Float64Array;
  private readonly r2re: Float64Array; private readonly r2im: Float64Array;
  /** c(x) = (K * 1_domain)(x) on the n1 × n2 sub-grid (walled case). */
  readonly c?: Float64Array;
  private mult?: Float64Array;
  private multKey = NaN;

  constructor(
    readonly n1: number, readonly n2: number, readonly p1: boolean, readonly p2: boolean,
    readonly h: number, readonly params: KernelParams,
  ) {
    const fractional = params.kernel === 'fractional';
    this.A = p1 ? n1 : nextPow2(2 * n1 - 1);
    this.B = p2 ? n2 : nextPow2(2 * n2 - 1);
    this.fullyPeriodic = p1 && p2;
    const { A, B } = this;
    this.f1 = new FFT(A);
    this.f2 = new FFT(B);
    this.re = new Float64Array(A * B);
    this.im = new Float64Array(A * B);
    this.r1re = new Float64Array(A); this.r1im = new Float64Array(A);
    this.r2re = new Float64Array(B); this.r2im = new Float64Array(B);
    this.sym = new Float64Array(A * B);
    if (fractional) {
      for (let b = 0; b < B; b++) for (let a = 0; a < A; a++) {
        const sx = Math.sin((Math.PI * a) / A), sy = Math.sin((Math.PI * b) / B);
        const lam = (4 / (h * h)) * (sx * sx + sy * sy);
        this.sym[a + A * b] = lam ** params.s;
      }
      return;
    }
    // sample the kernel at all offsets (min-image on periodic axes, |d| ≤ n−1 on walled axes)
    const off = (d: number, n: number, m: number, per: boolean): number => {
      if (per) return Math.min(d, n - d);
      if (d <= n - 1) return d;
      if (d >= m - (n - 1)) return m - d;
      return -1;
    };
    const { re, im } = this;
    let sum = 0;
    for (let b = 0; b < B; b++) {
      const db = off(b, n2, B, p2);
      for (let a = 0; a < A; a++) {
        const da = off(a, n1, A, p1);
        const k = a + A * b;
        if (da < 0 || db < 0) { re[k] = 0; continue; }
        const val = kernelProfile(params, h * Math.hypot(da, db));
        re[k] = val;
        sum += val;
      }
    }
    for (let k = 0; k < re.length; k++) re[k] /= sum;
    im.fill(0);
    this.fft2(true, B);
    for (let k = 0; k < re.length; k++) this.sym[k] = re[k]; // real: kernel is even
    if (!this.fullyPeriodic) {
      this.c = new Float64Array(n1 * n2);
      this.load(() => 1);
      this.convolveLoaded();
      for (let j = 0; j < n2; j++) for (let i = 0; i < n1; i++) this.c[i + n1 * j] = re[i + A * j];
    }
  }

  /** Discrete 2D FFT of (re, im); only the first `rows` rows are nonzero on input to a forward pass. */
  private fft2(forward: boolean, rows: number): void {
    const { A, B, re, im, r1re, r1im, r2re, r2im } = this;
    const doRows = (nb: number) => {
      for (let b = 0; b < nb; b++) {
        const o = A * b;
        for (let a = 0; a < A; a++) { r1re[a] = re[o + a]; r1im[a] = im[o + a]; }
        if (forward) this.f1.forward(r1re, r1im); else this.f1.inverse(r1re, r1im);
        for (let a = 0; a < A; a++) { re[o + a] = r1re[a]; im[o + a] = r1im[a]; }
      }
    };
    const doCols = (na: number) => {
      for (let a = 0; a < na; a++) {
        for (let b = 0; b < B; b++) { r2re[b] = re[a + A * b]; r2im[b] = im[a + A * b]; }
        if (forward) this.f2.forward(r2re, r2im); else this.f2.inverse(r2re, r2im);
        for (let b = 0; b < B; b++) { re[a + A * b] = r2re[b]; im[a + A * b] = r2im[b]; }
      }
    };
    if (forward) { doRows(rows); doCols(A); } else { doCols(A); doRows(rows); }
  }

  private load(get: (i: number, j: number) => number): void {
    const { A, n1, n2, re, im } = this;
    re.fill(0); im.fill(0);
    for (let j = 0; j < n2; j++) for (let i = 0; i < n1; i++) re[i + A * j] = get(i, j);
  }

  private convolveLoaded(): void {
    this.fft2(true, this.n2);
    const { re, im, sym } = this;
    for (let k = 0; k < re.length; k++) { re[k] *= sym[k]; im[k] *= sym[k]; }
    this.fft2(false, this.n2);
  }

  /**
   * Advance du/dt = L u by dt for the sub-grid field accessed through get/set
   * (i ∈ [0, n1), j ∈ [0, n2)).
   */
  apply(get: (i: number, j: number) => number, set: (i: number, j: number, x: number) => void, dt: number): void {
    const { n1, n2, A, re, im, params } = this;
    const S = params.strength;
    if (!(S > 0) || !(dt > 0)) return;
    const fractional = params.kernel === 'fractional';
    if (fractional && !this.fullyPeriodic) return; // documented: periodic only
    this.load(get);
    if (this.fullyPeriodic) {
      const key = S * dt;
      if (key !== this.multKey || !this.mult) {
        const m = (this.mult ??= new Float64Array(this.sym.length));
        for (let k = 0; k < m.length; k++)
          m[k] = fractional ? Math.exp(-key * this.sym[k]) : Math.exp(key * (this.sym[k] - 1));
        this.multKey = key;
      }
      this.fft2(true, n2);
      const m = this.mult;
      for (let k = 0; k < re.length; k++) { re[k] *= m[k]; im[k] *= m[k]; }
      this.fft2(false, n2);
      for (let j = 0; j < n2; j++) for (let i = 0; i < n1; i++) set(i, j, re[i + A * j]);
      return;
    }
    const old = new Float64Array(n1 * n2);
    for (let j = 0; j < n2; j++) for (let i = 0; i < n1; i++) old[i + n1 * j] = re[i + A * j];
    this.convolveLoaded();
    const c = this.c!;
    for (let j = 0; j < n2; j++) for (let i = 0; i < n1; i++) {
      const ci = c[i + n1 * j];
      const a = params.motschTadmor ? (S * dt) / ci : S * dt;
      set(i, j, (old[i + n1 * j] + a * re[i + A * j]) / (1 + a * ci));
    }
  }

  /**
   * Radial shell decomposition of the (normalised) kernel for the retarded
   * operator: K = K_self·δ + Σ_m K_m + K_tail, where
   *   K_self  the r = 0 offset (no delay: a point always knows its own velocity),
   *   K_m     offsets with r ∈ (r_m, r_{m+1}], m = 0 … M−1, equal-mass shells:
   *           r_{m+1} encloses (m+1)/M of the non-self mass inside R_cut,
   *   K_tail  offsets with r > R_cut (the last `1 − massFraction` of the mass),
   * R_cut = smallest radius enclosing `massFraction` of ∫K. Each shell gets the
   * delay of its OUTER radius (r_out/v_int), so no part of the kernel ever
   * receives information faster than v_int; the tail's outer radius is the
   * radius beyond which less than 1e-12 of the mass remains.
   * Shell spectra K̂_m are cached per shell count.
   */
  shells(M: number, massFraction = 0.995): ShellSet {
    M = Math.max(1, Math.round(M));
    const hit = this.shellCache.get(M);
    if (hit && hit.massFraction === massFraction) return hit;
    const { A, B, n1, n2, p1, p2, h, params, re, im } = this;
    if (params.kernel === 'fractional' || params.kernel === 'none') throw new Error('shells: convolution kernels only');
    const off = (d: number, n: number, m: number, per: boolean): number => {
      if (per) return Math.min(d, n - d);
      if (d <= n - 1) return d;
      if (d >= m - (n - 1)) return m - d;
      return -1;
    };
    // identical sampling / normalisation as the constructor
    const val = new Float64Array(A * B);
    const dist = new Float64Array(A * B).fill(-1);
    let sum = 0;
    for (let b = 0; b < B; b++) {
      const db = off(b, n2, B, p2);
      for (let a = 0; a < A; a++) {
        const da = off(a, n1, A, p1);
        const k = a + A * b;
        if (da < 0 || db < 0) continue;
        const r = h * Math.hypot(da, db);
        const v = kernelProfile(params, r);
        val[k] = v;
        dist[k] = r;
        sum += v;
      }
    }
    for (let k = 0; k < val.length; k++) val[k] /= sum;
    const selfWeight = val[0];
    // R_cut (mass fraction) and the 1e-12 tail radius, from offsets sorted by distance
    const order: number[] = [];
    for (let k = 1; k < val.length; k++) if (dist[k] > 0 && val[k] > 0) order.push(k);
    order.sort((x, y) => dist[x] - dist[y]);
    let cum = selfWeight, rCut = h;
    for (const k of order) { cum += val[k]; if (cum >= massFraction) { rCut = dist[k]; break; } }
    if (order.length) rCut = Math.min(Math.max(rCut, h), dist[order[order.length - 1]]);
    let beyond = 0, rTail = rCut;
    for (let q = order.length - 1; q >= 0; q--) {
      beyond += val[order[q]];
      if (beyond >= 1e-12) { rTail = Math.max(rCut, dist[order[q]]); break; }
    }
    // equal-mass shells: edge m+1 encloses (m+1)/M of the (non-self) mass inside R_cut
    const n = M + 1; // regular shells + tail
    const rOuter = new Float64Array(n);
    let inside = 0;
    for (const k of order) { if (dist[k] > rCut) break; inside += val[k]; }
    {
      let acc = 0, m = 0;
      for (const k of order) {
        if (dist[k] > rCut || m >= M - 1) break;
        acc += val[k];
        while (m < M - 1 && acc >= ((m + 1) / M) * inside * (1 - 1e-12)) rOuter[m++] = dist[k];
      }
      for (; m < M; m++) rOuter[m] = rCut;
    }
    rOuter[M] = rTail;
    const dr = rCut / M;
    const shellOf = (r: number) => {
      if (r > rCut * (1 + 1e-12)) return M;
      let m = 0;
      while (m < M - 1 && r > rOuter[m] * (1 + 1e-12)) m++;
      return m;
    };
    const mass = new Float64Array(n);
    const sym: Float64Array[] = [];
    const real: Float64Array[] = [];
    for (let m = 0; m < n; m++) { sym.push(new Float64Array(A * B)); real.push(new Float64Array(A * B)); }
    for (const k of order) { const m = shellOf(dist[k]); real[m][k] = val[k]; mass[m] += val[k]; }
    for (let m = 0; m < n; m++) {
      if (!(mass[m] > 0)) continue;
      re.set(real[m]); im.fill(0);
      this.fft2(true, B);
      sym[m].set(re);
    }
    const set: ShellSet = {
      M, massFraction, rCut, rTail, dr, rOuter, mass, sym, selfWeight,
      real: real.map((r) => r), // real-space shell kernels (tests)
      full: val,
    };
    this.shellCache.set(M, set);
    return set;
  }

  /**
   * Retarded (finite interaction speed) alignment over one step at simulation
   * time t:
   *   w(x) = K_self u(x,t) + Σ_m (K_m * u(·, t − τ_m))(x),   τ_m = r_out,m / v_int
   *   u_new = (u + a·w) / (1 + a·c),   a = S Δt  [Motsch–Tadmor: a = S Δt / c]
   * The past fields u(·, t − τ) are linearly interpolated in time from `hist`
   * (missing history → the oldest stored frame, or the current field). The
   * current field is pushed into `hist` afterwards. Periodic sub-grids store
   * the spectra of the frames (one FFT per stored frame), walled ones the
   * real fields (one FFT per shell and step).
   */
  applyRetarded(
    get: (i: number, j: number) => number, set: (i: number, j: number, x: number) => void,
    dt: number, t: number, vInt: number, M: number, hist: DelayHistory, budgetBytes = 32 * 2 ** 20,
  ): void {
    const { n1, n2, A, B, re, im, params } = this;
    const S = params.strength;
    this.lastTaps = [];
    if (!(S > 0) || !(dt > 0)) return;
    if (params.kernel === 'fractional') { this.apply(get, set, dt); return; }
    const sh = this.shells(M);
    const periodic = this.fullyPeriodic;
    const nSh = sh.rOuter.length;
    const tau = new Float64Array(nSh);
    let tauMax = 0;
    for (let m = 0; m < nSh; m++) {
      tau[m] = sh.rOuter[m] / vInt;
      if (sh.mass[m] > 0) tauMax = Math.max(tauMax, tau[m]);
    }
    hist.configure(periodic, periodic ? A * B : n1 * n2, dt, tauMax, budgetBytes);
    this.load(get);
    const old = new Float64Array(n1 * n2);
    for (let j = 0; j < n2; j++) for (let i = 0; i < n1; i++) old[i + n1 * j] = re[i + A * j];
    const wre = (this.wre ??= new Float64Array(A * B));
    const wim = (this.wim ??= new Float64Array(A * B));
    wre.fill(0); wim.fill(0);
    const frames = hist.frames;
    const taps = this.lastTaps;
    if (periodic) {
      this.fft2(true, n2); // re/im = spectrum of the current field
      for (let m = 0; m < nSh; m++) {
        if (!(sh.mass[m] > 0)) continue;
        const { a, b, theta } = hist.lookup(t - tau[m], t);
        const K = sh.sym[m];
        const wa = 1 - theta, wb = theta;
        taps.push({ age: a < 0 ? 0 : t - frames[a].t, weight: sh.mass[m] * wa });
        if (wb > 0) taps.push({ age: b < 0 ? 0 : t - frames[b].t, weight: sh.mass[m] * wb });
        const Ar = a < 0 ? re : frames[a].re, Ai = a < 0 ? im : frames[a].im!;
        if (wb > 0) {
          const Br = b < 0 ? re : frames[b].re, Bi = b < 0 ? im : frames[b].im!;
          for (let k = 0; k < wre.length; k++) {
            const kk = K[k];
            wre[k] += kk * (wa * Ar[k] + wb * Br[k]);
            wim[k] += kk * (wa * Ai[k] + wb * Bi[k]);
          }
        } else {
          for (let k = 0; k < wre.length; k++) { wre[k] += K[k] * Ar[k]; wim[k] += K[k] * Ai[k]; }
        }
      }
      if (hist.wants(t)) { const f = hist.push(t); f.re.set(re); f.im!.set(im); }
      re.set(wre); im.set(wim);
      this.fft2(false, n2);
    } else {
      // group shells that read the same (frames, weight) so each distinct
      // delayed field is transformed once
      const groups = new Map<string, { a: number; b: number; theta: number; ms: number[] }>();
      for (let m = 0; m < nSh; m++) {
        if (!(sh.mass[m] > 0)) continue;
        const l = hist.lookup(t - tau[m], t);
        const key = `${l.a}|${l.b}|${l.theta}`;
        const g = groups.get(key);
        if (g) g.ms.push(m); else groups.set(key, { ...l, ms: [m] });
        taps.push({ age: l.a < 0 ? 0 : t - frames[l.a].t, weight: sh.mass[m] * (1 - l.theta) });
        if (l.theta > 0) taps.push({ age: l.b < 0 ? 0 : t - frames[l.b].t, weight: sh.mass[m] * l.theta });
      }
      for (const g of groups.values()) {
        const fa = g.a < 0 ? null : frames[g.a].re, fb = g.b < 0 ? null : frames[g.b].re;
        const wa = 1 - g.theta, wb = g.theta;
        re.fill(0); im.fill(0);
        for (let j = 0; j < n2; j++) for (let i = 0; i < n1; i++) {
          const q = i + n1 * j;
          const va = fa ? fa[q] : old[q];
          re[i + A * j] = wb > 0 ? wa * va + wb * (fb ? fb[q] : old[q]) : va;
        }
        this.fft2(true, n2);
        for (let k = 0; k < wre.length; k++) {
          let kk = 0;
          for (const m of g.ms) kk += sh.sym[m][k];
          wre[k] += kk * re[k];
          wim[k] += kk * im[k];
        }
      }
      if (hist.wants(t)) { const f = hist.push(t); for (let q = 0; q < old.length; q++) f.re[q] = old[q]; }
      re.set(wre); im.set(wim);
      this.fft2(false, n2);
    }
    taps.push({ age: 0, weight: sh.selfWeight });
    const c = this.c;
    for (let j = 0; j < n2; j++) for (let i = 0; i < n1; i++) {
      const q = i + n1 * j;
      const w = re[i + A * j] + sh.selfWeight * old[q];
      const ci = periodic ? 1 : c![q];
      const a = params.motschTadmor && !periodic ? (S * dt) / ci : S * dt;
      set(i, j, (old[q] + a * w) / (1 + a * ci));
    }
    hist.prune(t + dt);
    void B;
  }

  /** Fourier symbol at index (a, b) (testing / plotting). */
  symbol(a: number, b: number): number { return this.sym[a + this.A * b]; }
}

/** Radial shell decomposition of a kernel (see NonlocalOperator.shells). */
export interface ShellSet {
  /** Number of regular shells (a tail shell is appended). */
  M: number;
  massFraction: number;
  /** Radius enclosing `massFraction` of the kernel mass (domain units). */
  rCut: number;
  /** Outer radius of the tail shell (less than 1e-12 of the mass lies beyond). */
  rTail: number;
  /** Mean shell width R_cut / M (shells are equal-mass, not equal-width). */
  dr: number;
  /** Outer radius of each shell (M regular + tail); its delay is rOuter / v_int. */
  rOuter: Float64Array;
  /** Kernel mass Σ K_m h² of each shell (Σ mass + selfWeight = 1). */
  mass: Float64Array;
  /** Fourier symbols K̂_m. */
  sym: Float64Array[];
  /** Real-space shell kernels K_m (offset layout of the operator). */
  real: Float64Array[];
  /** The full normalised kernel (same layout). */
  full: Float64Array;
  /** Weight of the r = 0 offset (never delayed). */
  selfWeight: number;
}

interface Frame { t: number; re: Float32Array; im: Float32Array | null }

/**
 * Ring buffer of past velocity fields (one sub-grid component), time-stamped
 * with simulation time. Periodic sub-grids store complex spectra, walled ones
 * real fields, both as Float32. Frames are recorded at most every `interval`
 * seconds (= Δt unless the memory budget forces coarser sampling) and kept
 * only as long as the longest delay needs them.
 */
export class DelayHistory {
  frames: Frame[] = [];
  private pool: Frame[] = [];
  private len = 0;
  private complex = false;
  interval = 0;
  capacity = 0;
  tauMax = 0;
  static readonly MAX_FRAMES = 512;

  configure(complex: boolean, len: number, dt: number, tauMax: number, budgetBytes: number): void {
    if (complex !== this.complex || len !== this.len) { this.clear(); this.pool = []; }
    this.complex = complex;
    this.len = len;
    this.tauMax = tauMax;
    const fb = this.frameBytes();
    this.capacity = Math.max(3, Math.min(DelayHistory.MAX_FRAMES, Math.floor(budgetBytes / fb)));
    // capacity − 2 intervals must cover τ_max (one extra frame brackets it)
    this.interval = Math.max(dt, tauMax / (this.capacity - 2));
    while (this.frames.length > this.capacity) this.pool.push(this.frames.shift()!);
  }

  frameBytes(): number { return 4 * this.len * (this.complex ? 2 : 1); }
  bytes(): number { return this.frames.length * this.frameBytes(); }
  clear(): void { this.pool.push(...this.frames); this.frames = []; }

  /** Should the field at time t be recorded? */
  wants(t: number): boolean {
    const f = this.frames;
    return f.length === 0 || t - f[f.length - 1].t >= this.interval * (1 - 1e-9);
  }

  push(t: number): Frame {
    let f = this.pool.pop();
    if (!f || f.re.length !== this.len || (this.complex !== (f.im !== null))) {
      f = { t, re: new Float32Array(this.len), im: this.complex ? new Float32Array(this.len) : null };
    }
    f.t = t;
    this.frames.push(f);
    while (this.frames.length > this.capacity) this.pool.push(this.frames.shift()!);
    return f;
  }

  /** Drop frames no lookup at time ≥ tNext can need. */
  prune(tNext: number): void {
    const f = this.frames;
    while (f.length >= 2 && f[1].t <= tNext - this.tauMax) this.pool.push(f.shift()!);
  }

  /**
   * Interpolation weights for the field at time T, given the current field at
   * time `now` (index −1): value = (1 − θ)·F[a] + θ·F[b]. T ≥ now → current
   * field; T before the oldest frame → the oldest frame (start-up).
   */
  lookup(T: number, now: number): { a: number; b: number; theta: number } {
    const f = this.frames;
    if (T >= now || f.length === 0) return { a: -1, b: -1, theta: 0 };
    if (T <= f[0].t) return { a: 0, b: 0, theta: 0 };
    let lo = 0, hi = f.length; // find last frame with t ≤ T
    while (hi - lo > 1) { const mid = (lo + hi) >> 1; if (f[mid].t <= T) lo = mid; else hi = mid; }
    const ta = f[lo].t;
    const b = lo + 1 < f.length ? lo + 1 : -1;
    const tb = b < 0 ? now : f[b].t;
    const theta = tb > ta ? (T - ta) / (tb - ta) : 0;
    if (theta <= 0) return { a: lo, b: lo, theta: 0 };
    if (theta >= 1) return { a: b, b, theta: 0 };
    return { a: lo, b, theta };
  }
}
