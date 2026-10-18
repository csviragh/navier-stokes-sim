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
 *   delayed (t_self, t_nb not both 0): explicit / semi-implicit update with the
 *     neighbours' and the own velocity read from the past, see applyDelayed().
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
   * Delayed explicit update with separate delays for the neighbours' and the
   * point's own velocity (two-delay alignment, see fluid.ts):
   *
   *   conv:        f = S [ (K * u_nb)(x) − c(x) u_self(x) ]        (c ≡ 1 on the torus)
   *   fractional:  f = ν_s [ (N * u_nb)(x) − G₀ u_self(x) ],  N̂(k) = G₀ − λ(k)^s, G₀ = ⟨λ^s⟩
   *
   * (the fractional operator −ν_s(−Δ)^s split into its diagonal G₀ and its
   * off-diagonal, long-range part; periodic only). u_nb = u(·, t − t_nb) and
   * u_self = u(·, t − t_self) are supplied by the caller (history reads at the
   * step midpoint), `getCur` is the current field. Update:
   *   explicit (t_self > 0):  u ← u + a (w − c u_self),  a = S Δt  [Motsch–Tadmor: a = S Δt / c]
   *   semi-implicit (t_self = 0, own velocity current): u ← (u + a w) / (1 + a c),
   *     a convex combination of present and past values (max principle, any Δt).
   */
  applyDelayed(
    getNb: (i: number, j: number) => number,
    getSelf: ((i: number, j: number) => number) | null,
    getCur: (i: number, j: number) => number,
    set: (i: number, j: number, x: number) => void,
    dt: number,
  ): void {
    const { n1, n2, A, re, im, params } = this;
    const S = params.strength;
    if (!(S > 0) || !(dt > 0)) return;
    const fractional = params.kernel === 'fractional';
    if (params.kernel === 'none' || (fractional && !this.fullyPeriodic)) return;
    this.load(getNb);
    if (fractional) {
      const G0 = this.diagonal();
      this.fft2(true, n2);
      const sym = this.sym;
      for (let k = 0; k < re.length; k++) { const m = G0 - sym[k]; re[k] *= m; im[k] *= m; }
      this.fft2(false, n2);
    } else this.convolveLoaded();
    const periodic = this.fullyPeriodic;
    const c = this.c;
    for (let j = 0; j < n2; j++) for (let i = 0; i < n1; i++) {
      const w = re[i + A * j];
      const ci = fractional ? this.diagonal() : periodic ? 1 : c![i + n1 * j];
      const a = !fractional && !periodic && params.motschTadmor ? (S * dt) / ci : S * dt;
      const cur = getCur(i, j);
      set(i, j, getSelf ? cur + a * (w - ci * getSelf(i, j)) : (cur + a * w) / (1 + a * ci));
    }
  }

  private diag = NaN;
  /**
   * Diagonal (self) weight of the operator: 1 for normalised convolution
   * kernels on the torus (c(x) ≤ 1 with walls), G₀ = mean of λ(k)^s over all
   * modes for the fractional Laplacian (its real-space kernel at r = 0).
   */
  diagonal(): number {
    if (this.params.kernel !== 'fractional') return 1;
    if (Number.isNaN(this.diag)) {
      let s = 0;
      for (let k = 0; k < this.sym.length; k++) s += this.sym[k];
      this.diag = s / this.sym.length;
    }
    return this.diag;
  }

  /**
   * Range of the neighbour/self coefficient ratio q = B/A over the non-uniform
   * modes (k ≠ 0), for the stability scan: conv kernels q = K̂(k) (A = S),
   * fractional q = 1 − λ(k)^s/G₀ (A = ν_s G₀).
   */
  modeRatioRange(): [number, number] {
    let lo = Infinity, hi = -Infinity;
    const frac = this.params.kernel === 'fractional';
    const G0 = this.diagonal();
    for (let k = 1; k < this.sym.length; k++) {
      const q = frac ? 1 - this.sym[k] / G0 : this.sym[k];
      if (q < lo) lo = q;
      if (q > hi) hi = q;
    }
    return [lo, Math.min(hi, 1)];
  }

  /** Fourier symbol at index (a, b) (testing / plotting). */
  symbol(a: number, b: number): number { return this.sym[a + this.A * b]; }
}
