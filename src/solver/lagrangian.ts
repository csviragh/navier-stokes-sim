/**
 * Path-following (Lagrangian, comoving) delays.
 *
 * With the Eulerian delay the delayed terms read the past field at the FIXED
 * grid point x: q(x, t − τ). Along a path they read it where the fluid parcel
 * now at x WAS: q(X(t − τ; x, t), t − τ), with the trajectory
 * dX/ds = u(X(s), s), X(t) = x. For a uniformly moving flock this removes the
 * Doppler phase k·U·τ of the Eulerian delay, so the rest-fluid thresholds
 * (e.g. S t = 1/e and π/2 for equal delays) hold in the comoving frame.
 *
 * Departure points are kept as a displacement map d_τ(x) = x − X(t − τ; x, t)
 * per face (u- and v-faces, cell units), one map per distinct delay τ. It is
 * updated incrementally each step instead of re-tracing τ/Δt sub-steps for
 * every face:
 *
 *   X(t+Δ−τ; x, t+Δ) = Φ_{t−τ → t−τ+Δ}( X(t−τ; y, t) ),   y = back-trace of x over [t, t+Δ]
 *
 * i.e. one RK2 back-trace step at the present end (current velocity, as in the
 * solver's advection), a bicubic (Catmull–Rom) lookup of the stored map at y, and one RK2
 * forward step at the past end (history velocities at t − τ and t − τ + Δ/2).
 * This is the RK2 trajectory with sub-steps Δt (= history frame spacing at
 * full resolution), re-using the previous step's trajectories. On (re)start
 * (reset, delay change, missed step, new grid / boundaries) the map is built
 * by a full RK2 back-trace with sub-steps Δt through the history (frames older
 * than the history use the oldest frame, like the Eulerian reads), and it is
 * also rebuilt that way every τ/2 of simulated time: an incremental update
 * composes the map with one-step flows, so its errors are stretched by the
 * flow Jacobian at the past end and would grow like e^{λt} in strained flow.
 *
 * Sampling: bilinear on each face grid (u-faces at (i, j − ½), v-faces at
 * (i − ½, j) in cell units), wrapping on periodic axes; at walls positions
 * are clamped to the domain and samples to the outermost stored face
 * (constant extrapolation, no reflection).
 */
import { periodicX, periodicY, type Grid, type WallType } from './grid';

type Read = (T: number, outU: Float64Array, outV: Float64Array) => void;

interface Faces { k: Int32Array; X: Float64Array; Y: Float64Array }

interface DepMap { tau: number; tMap: number; tBuilt: number; dU: Float64Array; dV: Float64Array } // interleaved (dx, dy) per face in list order

const mod = (a: number, n: number) => ((a % n) + n) % n;

export class PathTracer {
  private key = '';
  private g!: Grid;
  private px = false; private py = false;
  private uf!: Faces; private vf!: Faces;
  private maps: DepMap[] = [];
  private a0u = new Float64Array(0); private a0v = new Float64Array(0);
  private ahu = new Float64Array(0); private ahv = new Float64Array(0);
  private dispU = new Float64Array(0); private dispV = new Float64Array(0); // face-layout scratch for map lookup
  /** Number of full back-trace (re)initialisations (for tests / diagnostics). */
  inits = 0;

  reset(): void { this.maps = []; }

  /** Drop the maps of delays no longer in use. */
  retain(taus: number[]): void { this.maps = this.maps.filter((m) => taus.includes(m.tau)); }

  private setup(g: Grid, walls: WallType): void {
    const key = `${g.nx}x${g.ny}:${walls}`;
    if (key === this.key) return;
    this.key = key; this.g = g; this.px = periodicX(walls); this.py = periodicY(walls);
    this.maps = [];
    const { nx, ny } = g;
    const list = (kind: 0 | 1): Faces => {
      const k: number[] = [], X: number[] = [], Y: number[] = [];
      if (kind === 0) {
        for (let j = 1; j <= ny; j++) for (let i = this.px ? 1 : 0; i <= nx; i++) { k.push(i + g.uStride * j); X.push(i); Y.push(j - 0.5); }
      } else {
        for (let j = this.py ? 1 : 0; j <= ny; j++) for (let i = 1; i <= nx; i++) { k.push(i + g.vStride * j); X.push(i - 0.5); Y.push(j); }
      }
      return { k: Int32Array.from(k), X: Float64Array.from(X), Y: Float64Array.from(Y) };
    };
    this.uf = list(0); this.vf = list(1);
    this.a0u = new Float64Array(g.uSize); this.a0v = new Float64Array(g.vSize);
    this.ahu = new Float64Array(g.uSize); this.ahv = new Float64Array(g.vSize);
    this.dispU = new Float64Array(2 * g.uSize); this.dispV = new Float64Array(2 * g.vSize);
  }

  /** Bilinear sample of a u-face (kind 0) or v-face (kind 1) field at (X, Y) (cell units). `comp`/`stride2` read interleaved data. */
  sample(kind: 0 | 1, f: ArrayLike<number>, X: number, Y: number, stride2 = 1, comp = 0): number {
    const g = this.g, nx = g.nx, ny = g.ny;
    let fi: number, fj: number, xlo: number, xhi: number, ylo: number, yhi: number, st: number;
    if (kind === 0) { fi = X; fj = Y + 0.5; st = g.uStride; xlo = this.px ? 1 : 0; xhi = nx; ylo = 1; yhi = ny; }
    else { fi = X + 0.5; fj = Y; st = g.vStride; xlo = 1; xhi = nx; ylo = this.py ? 1 : 0; yhi = ny; }
    let i0: number, i1: number, wi: number, j0: number, j1: number, wj: number;
    if (this.px) { const fl = Math.floor(fi); wi = fi - fl; i0 = fl >= 1 && fl <= nx ? fl : 1 + mod(fl - 1, nx); i1 = i0 === nx ? 1 : i0 + 1; }
    else { const f2 = fi < xlo ? xlo : fi > xhi ? xhi : fi; i0 = Math.floor(f2); if (i0 >= xhi) { i0 = xhi; i1 = xhi; wi = 0; } else { i1 = i0 + 1; wi = f2 - i0; } }
    if (this.py) { const fl = Math.floor(fj); wj = fj - fl; j0 = fl >= 1 && fl <= ny ? fl : 1 + mod(fl - 1, ny); j1 = j0 === ny ? 1 : j0 + 1; }
    else { const f2 = fj < ylo ? ylo : fj > yhi ? yhi : fj; j0 = Math.floor(f2); if (j0 >= yhi) { j0 = yhi; j1 = yhi; wj = 0; } else { j1 = j0 + 1; wj = f2 - j0; } }
    const r0 = st * j0, r1 = st * j1;
    const f00 = f[(i0 + r0) * stride2 + comp], f10 = f[(i1 + r0) * stride2 + comp];
    const f01 = f[(i0 + r1) * stride2 + comp], f11 = f[(i1 + r1) * stride2 + comp];
    return (1 - wj) * ((1 - wi) * f00 + wi * f10) + wj * ((1 - wi) * f01 + wi * f11);
  }

  /**
   * Bicubic (Catmull–Rom) sample of interleaved face data (used for the map
   * look-up: bilinear look-ups would smooth the map by ~h²·d''/8 every step and
   * the error would accumulate; Catmull–Rom is O(h⁴) per step).
   */
  private sampleCubic(kind: 0 | 1, f: ArrayLike<number>, X: number, Y: number, comp: number): number {
    const g = this.g, nx = g.nx, ny = g.ny;
    let fi: number, fj: number, xlo: number, ylo: number, st: number;
    if (kind === 0) { fi = X; fj = Y + 0.5; st = g.uStride; xlo = this.px ? 1 : 0; ylo = 1; }
    else { fi = X + 0.5; fj = Y; st = g.vStride; xlo = 1; ylo = this.py ? 1 : 0; }
    const xhi = nx, yhi = ny;
    if (!this.px) fi = fi < xlo ? xlo : fi > xhi ? xhi : fi;
    if (!this.py) fj = fj < ylo ? ylo : fj > yhi ? yhi : fj;
    const i1 = Math.floor(fi), j1 = Math.floor(fj), wx = fi - i1, wy = fj - j1;
    const ix = (i: number) => this.px ? 1 + mod(i - 1, nx) : i < xlo ? xlo : i > xhi ? xhi : i;
    const jy = (j: number) => this.py ? 1 + mod(j - 1, ny) : j < ylo ? ylo : j > yhi ? yhi : j;
    const cr = (t: number, p0: number, p1: number, p2: number, p3: number) =>
      p1 + 0.5 * t * (p2 - p0 + t * (2 * p0 - 5 * p1 + 4 * p2 - p3 + t * (3 * (p1 - p2) + p3 - p0)));
    const I0 = ix(i1 - 1), I1 = ix(i1), I2 = ix(i1 + 1), I3 = ix(i1 + 2);
    const row = (j: number) => { const r = st * jy(j); return cr(wx, f[2 * (I0 + r) + comp], f[2 * (I1 + r) + comp], f[2 * (I2 + r) + comp], f[2 * (I3 + r) + comp]); };
    return cr(wy, row(j1 - 1), row(j1), row(j1 + 1), row(j1 + 2));
  }

  private clampX(X: number): number { return this.px ? X : X < 0 ? 0 : X > this.g.nx ? this.g.nx : X; }
  private clampY(Y: number): number { return this.py ? Y : Y < 0 ? 0 : Y > this.g.ny ? this.g.ny : Y; }

  /** Velocity (cells / s) of field (fu, fv) at (X, Y). */
  private vel(fu: ArrayLike<number>, fv: ArrayLike<number>, X: number, Y: number, out: number[]): void {
    const ih = 1 / this.g.h;
    out[0] = this.sample(0, fu, X, Y) * ih; out[1] = this.sample(1, fv, X, Y) * ih;
  }

  /** Full RK2 back-trace of every face from t to t − τ (sub-steps ≤ dt). */
  private build(tau: number, t: number, dt: number, read: Read): DepMap {
    this.inits++;
    const n = Math.max(1, Math.ceil(tau / dt - 1e-9)), hs = tau / n;
    const posU = new Float64Array(2 * this.uf.k.length), posV = new Float64Array(2 * this.vf.k.length);
    for (let q = 0; q < this.uf.k.length; q++) { posU[2 * q] = this.uf.X[q]; posU[2 * q + 1] = this.uf.Y[q]; }
    for (let q = 0; q < this.vf.k.length; q++) { posV[2 * q] = this.vf.X[q]; posV[2 * q + 1] = this.vf.Y[q]; }
    const w = [0, 0];
    for (let m = 0; m < n; m++) {
      const s = t - m * hs;
      read(s, this.a0u, this.a0v); read(s - hs / 2, this.ahu, this.ahv);
      for (const pos of [posU, posV]) for (let q = 0; q < pos.length; q += 2) {
        const X = pos[q], Y = pos[q + 1];
        this.vel(this.a0u, this.a0v, X, Y, w);
        const Xm = this.clampX(X - 0.5 * hs * w[0]), Ym = this.clampY(Y - 0.5 * hs * w[1]);
        this.vel(this.ahu, this.ahv, Xm, Ym, w);
        pos[q] = this.clampX(X - hs * w[0]); pos[q + 1] = this.clampY(Y - hs * w[1]);
      }
    }
    const dU = new Float64Array(posU.length), dV = new Float64Array(posV.length);
    for (let q = 0; q < this.uf.k.length; q++) { dU[2 * q] = this.uf.X[q] - posU[2 * q]; dU[2 * q + 1] = this.uf.Y[q] - posU[2 * q + 1]; }
    for (let q = 0; q < this.vf.k.length; q++) { dV[2 * q] = this.vf.X[q] - posV[2 * q]; dV[2 * q + 1] = this.vf.Y[q] - posV[2 * q + 1]; }
    return { tau, tMap: t, tBuilt: t, dU, dV };
  }

  /** Advance a map from its time to t (Δ = t − tMap). */
  private advance(m: DepMap, t: number, curU: ArrayLike<number>, curV: ArrayLike<number>, read: Read): void {
    const D = t - m.tMap, tau = m.tau;
    read(m.tMap - tau, this.a0u, this.a0v);
    read(m.tMap - tau + D / 2, this.ahu, this.ahv);
    // scatter the old displacements into face layout for interpolation
    const lay = (f: Faces, d: Float64Array, out: Float64Array) => { for (let q = 0; q < f.k.length; q++) { out[2 * f.k[q]] = d[2 * q]; out[2 * f.k[q] + 1] = d[2 * q + 1]; } };
    lay(this.uf, m.dU, this.dispU); lay(this.vf, m.dV, this.dispV);
    const w = [0, 0];
    const run = (f: Faces, kind: 0 | 1, dOut: Float64Array) => {
      const lay2 = kind === 0 ? this.dispU : this.dispV;
      for (let q = 0; q < f.k.length; q++) {
        const X = f.X[q], Y = f.Y[q];
        // back-trace the parcel over [tMap, t] with the current velocity (RK2)
        this.vel(curU, curV, X, Y, w);
        const Xm = this.clampX(X - 0.5 * D * w[0]), Ym = this.clampY(Y - 0.5 * D * w[1]);
        this.vel(curU, curV, Xm, Ym, w);
        const yX = this.clampX(X - D * w[0]), yY = this.clampY(Y - D * w[1]);
        // where that parcel was at tMap − τ
        let zX = this.clampX(yX - this.sampleCubic(kind, lay2, yX, yY, 0));
        let zY = this.clampY(yY - this.sampleCubic(kind, lay2, yX, yY, 1));
        // forward RK2 step at the past end, tMap − τ → t − τ
        this.vel(this.a0u, this.a0v, zX, zY, w);
        const zmX = this.clampX(zX + 0.5 * D * w[0]), zmY = this.clampY(zY + 0.5 * D * w[1]);
        this.vel(this.ahu, this.ahv, zmX, zmY, w);
        zX = this.clampX(zX + D * w[0]); zY = this.clampY(zY + D * w[1]);
        dOut[2 * q] = X - zX; dOut[2 * q + 1] = Y - zY;
      }
    };
    const nU = new Float64Array(m.dU.length), nV = new Float64Array(m.dV.length);
    run(this.uf, 0, nU); run(this.vf, 1, nV);
    m.dU = nU; m.dV = nV; m.tMap = t;
  }

  /**
   * Departure data of the parcels at every face at time t, written in face
   * layout into outU / outV as 4 numbers per face: the position X(t − τ)
   * (cell units, unwrapped on periodic axes) and the velocity w there at
   * t − τ (cells / s), so that X(t − τ + s) ≈ X(t − τ) + s·w for the
   * sub-frame offsets s used by the time interpolation. τ ≤ dt/2: X = x, w = 0.
   */
  departures(
    g: Grid, walls: WallType, tau: number, t: number, dt: number,
    curU: ArrayLike<number>, curV: ArrayLike<number>, read: Read,
    outU: Float64Array, outV: Float64Array,
  ): void {
    this.setup(g, walls);
    if (tau <= dt / 2) {
      for (const [f, out] of [[this.uf, outU], [this.vf, outV]] as const)
        for (let q = 0; q < f.k.length; q++) { const o = 4 * f.k[q]; out[o] = f.X[q]; out[o + 1] = f.Y[q]; out[o + 2] = 0; out[o + 3] = 0; }
      return;
    }
    let m = this.maps.find((x) => x.tau === tau);
    const D = m ? t - m.tMap : NaN;
    // Incremental updates compose the stored map with one-step flows, so a map
    // error is stretched by the past-end flow Jacobian every step and grows
    // like e^{λ t} (λ = finite-time Lyapunov exponent) in strained or chaotic
    // flow. Rebuilding from a full back-trace every τ/2 bounds that to e^{λτ/2}.
    if (!m || !(D >= 0) || D > 1.5 * dt || t - m.tBuilt >= Math.max(dt, tau / 2) - 1e-9) {
      if (m) this.maps.splice(this.maps.indexOf(m), 1);
      m = this.build(tau, t, dt, read);
      this.maps.push(m);
    } else if (D > 0) this.advance(m, t, curU, curV, read);
    // The increment Δt·f(t + Δt/2) is added to the parcel that is at x at time t
    // (advection follows later in the step), so trace THAT parcel.
    read(t - tau, this.a0u, this.a0v);
    const w = [0, 0];
    for (const [f, out, d] of [[this.uf, outU, m.dU], [this.vf, outV, m.dV]] as const) {
      for (let q = 0; q < f.k.length; q++) {
        const zX = this.clampX(f.X[q] - d[2 * q]), zY = this.clampY(f.Y[q] - d[2 * q + 1]);
        this.vel(this.a0u, this.a0v, zX, zY, w);
        const o = 4 * f.k[q];
        out[o] = zX; out[o + 1] = zY; out[o + 2] = w[0]; out[o + 3] = w[1];
      }
    }
  }

  /**
   * out[k] (+)= weight · f(X(t − τ) + s·w) for every face of the given kind
   * (dep from departures()); accumulate = false overwrites.
   */
  sampleAt(kind: 0 | 1, f: ArrayLike<number>, dep: Float64Array, s: number, weight: number, out: Float64Array, accumulate: boolean): void {
    const faces = kind === 0 ? this.uf : this.vf;
    for (let q = 0; q < faces.k.length; q++) {
      const k = faces.k[q], o = 4 * k;
      const val = weight * this.sample(kind, f, this.clampX(dep[o] + s * dep[o + 2]), this.clampY(dep[o + 1] + s * dep[o + 3]));
      out[k] = accumulate ? out[k] + val : val;
    }
  }

  /** Current displacement map for τ (tests): interleaved (dx, dy) per face in face layout, or null. */
  displacement(tau: number): { u: Float64Array; v: Float64Array } | null {
    const m = this.maps.find((x) => x.tau === tau);
    if (!m) return null;
    const u = new Float64Array(2 * this.g.uSize), v = new Float64Array(2 * this.g.vSize);
    for (let q = 0; q < this.uf.k.length; q++) { u[2 * this.uf.k[q]] = m.dU[2 * q]; u[2 * this.uf.k[q] + 1] = m.dU[2 * q + 1]; }
    for (let q = 0; q < this.vf.k.length; q++) { v[2 * this.vf.k[q]] = m.dV[2 * q]; v[2 * this.vf.k[q] + 1] = m.dV[2 * q + 1]; }
    return { u, v };
  }

  /** Full back-trace map (tests): departure displacement for τ at time t, without touching the incremental state. */
  fullTrace(g: Grid, walls: WallType, tau: number, t: number, dt: number, read: Read): { u: Float64Array; v: Float64Array } {
    this.setup(g, walls);
    const m = this.build(tau, t, dt, read); this.inits--;
    const u = new Float64Array(2 * g.uSize), v = new Float64Array(2 * g.vSize);
    for (let q = 0; q < this.uf.k.length; q++) { u[2 * this.uf.k[q]] = m.dU[2 * q]; u[2 * this.uf.k[q] + 1] = m.dU[2 * q + 1]; }
    for (let q = 0; q < this.vf.k.length; q++) { v[2 * this.vf.k[q]] = m.dV[2 * q]; v[2 * this.vf.k[q] + 1] = m.dV[2 * q + 1]; }
    return { u, v };
  }
}
