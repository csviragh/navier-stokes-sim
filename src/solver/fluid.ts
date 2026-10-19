/**
 * High-level 2D incompressible Navier–Stokes solver ("Stable Fluids",
 * J. Stam 1999) on a staggered MAC grid with solid walls (no-slip / free-slip)
 * or periodic boundaries (fully periodic, or periodic in x with walls in y).
 *
 *   ∂u/∂t + (u·∇)u = −∇p/ρ + ν∇²u + f − (1/τ)(|u| − v₀) u/|u|   (momentum, ρ = 1)
 *   ∇·u = 0                                 (incompressibility)
 *   ∂c/∂t + (u·∇)c = κ∇²c                   (passive dye, three RGB channels)
 *
 * One time step (operator splitting):
 *   1. forces (vorticity confinement; user impulses are applied via splat())
 *   2. implicit viscous diffusion, then the optional nonlocal (long-range)
 *      viscosity / alignment kernel S∫K(|x−y|)(u(y)−u(x))dy or −ν_s(−Δ)^s u
 *      (see kernels.ts). With delays (t_self, t_nb) either term reads the
 *      point's own velocity t_self ago and its neighbours' velocities t_nb ago
 *      (explicit update from a history of past fields, see applyDelayedViscosity)
 *   3. projection (Poisson solve for p, subtract ∇p)
 *   4. semi-Lagrangian self-advection (RK2 back-trace, bilinear interpolation)
 *   5. optional self-propulsion (active fluid): exact relaxation of the local
 *      speed |u| toward v₀ with relaxation time τ, direction kept
 *   6. projection
 *   7. dye: implicit diffusion + semi-Lagrangian advection + optional fade
 *
 * This module has no DOM dependencies.
 */
import {
  createField, createGrid, createUField, createVField, idx, isPeriodic, periodicX, periodicY,
  uIdx, vIdx, type Grid, type WallType,
} from './grid';
import {
  advectScalar, advectVelocity, cellVelocity, computeDivergence, computeVorticity,
  diffuseScalar, diffuseU, diffuseV, project, sampleCell, sampleU, sampleV,
  selfPropulsion, setScalarBoundary, setUBoundary, setVBoundary, uLastI, vLastJ,
  vorticityConfinement, neighbourSumU, neighbourSumV,
} from './ops';
import { NonlocalOperator, type KernelParams, type KernelType } from './kernels';
import { FieldHistory } from './delay';
import { PathTracer } from './lagrangian';
import { scanModes, type ModeScan } from './stability';

export interface FluidParams {
  /** Kinematic viscosity ν (domain units² / s; the shorter side has length 1). */
  viscosity: number;
  /** Dye diffusion coefficient κ (domain units² / s). */
  diffusion: number;
  /** Time step Δt in seconds. */
  dt: number;
  /** Gauss-Seidel iterations for the diffusion and pressure solves. */
  iterations: number;
  /** Over-relaxation factor for the pressure solve (1 = plain Gauss-Seidel). */
  sor: number;
  /** Vorticity confinement strength ε (0 disables it). */
  vorticity: number;
  /** Exponential dye fade rate in 1/s (0 = dye is conserved). */
  dyeDissipation: number;
  /**
   * Boundary condition: solid walls ('no-slip' / 'free-slip'), fully
   * 'periodic', or 'periodic-x' (periodic in x, no-slip walls in y).
   */
  walls: WallType;
  /**
   * Enable the self-propulsion (active fluid) term
   * −(1/τ)(|u| − v₀) u/|u| in the momentum equation. Off by default.
   */
  selfPropulsion: boolean;
  /** Relaxation time τ of the self-propulsion term (s). */
  selfPropTau: number;
  /** Target speed v₀ of the self-propulsion term (domain units / s). */
  selfPropV0: number;
  /** Long-range viscosity / alignment kernel added to the local ν∇² ('none' = off). */
  kernel: KernelType;
  /** Kernel strength S (1/s); for 'fractional' it is ν_s (L^{2s}/s). */
  kernelStrength: number;
  /** Kernel range / softening length R (domain units). */
  kernelRange: number;
  /** Cucker–Smale exponent β. */
  kernelBeta: number;
  /** Power-law exponent α. */
  kernelAlpha: number;
  /** Fractional order s ∈ (0, 1]. */
  kernelS: number;
  /** Motsch–Tadmor local normalisation (only differs from the default next to walls). */
  kernelMT: boolean;
  /**
   * Delay of the point's OWN velocity t_self (s) in the viscosity terms it is
   * enabled for (delayLocal / delayKernel):
   *   local:      ν/h² [ Σ_nb u_nb(t − t_nb) − 4 u(x, t − t_self) ]
   *   long-range: S [ (K * u)(x, t − t_nb) − c(x) u(x, t − t_self) ]
   * t_self = t_nb = 0: no delay (the previous exact / implicit updates, bit-identical).
   */
  delaySelf: number;
  /** Delay of the NEIGHBOURS' velocities t_nb (s), see delaySelf. */
  delayNb: number;
  /** Apply (t_self, t_nb) to the local viscosity ν∇²u. */
  delayLocal: boolean;
  /** Apply (t_self, t_nb) to the long-range kernel / fractional term. */
  delayKernel: boolean;
  /**
   * Where the delayed quantities are read: 'eulerian' = at the fixed grid
   * point, q(x, t − τ); 'lagrangian' = where the fluid parcel now at x was,
   * q(X(t − τ), t − τ) (path-following / comoving delay, see lagrangian.ts).
   */
  delayMode: DelayMode;
}

export type DelayMode = 'eulerian' | 'lagrangian';

export const defaultParams: FluidParams = {
  viscosity: 0.00005,
  diffusion: 0,
  dt: 1 / 60,
  iterations: 20,
  sor: 1.7,
  vorticity: 0,
  dyeDissipation: 0,
  walls: 'no-slip',
  selfPropulsion: false,
  selfPropTau: 0.5,
  selfPropV0: 0.5,
  kernel: 'none',
  kernelStrength: 2,
  kernelRange: 0.05,
  kernelBeta: 1,
  kernelAlpha: 3,
  kernelS: 0.5,
  kernelMT: false,
  delaySelf: 0,
  delayNb: 0,
  delayLocal: true,
  delayKernel: true,
  delayMode: 'eulerian',
};

/**
 * Speed guard of the calculation-delay path: after the explicit delayed
 * update each face velocity is clamped to ±DELAY_SPEED_LIMIT (domain units/s).
 * Only reached when the delay instability runs away without anything (e.g.
 * self-propulsion) to saturate it; it keeps the fields finite.
 */
export const DELAY_SPEED_LIMIT = 10;

/** Memory budget of the delay history (both velocity components, bytes). */
export const DELAY_BUDGET_BYTES = 64 * 2 ** 20;

/** Largest local-viscosity number ν Δt_sub / h² per explicit sub-step (2D explicit limit: 1/4). */
export const LOCAL_SUBSTEP_NUMBER = 0.2;
/** Cap on the number of explicit sub-steps of the delayed local viscosity. */
export const MAX_LOCAL_SUBSTEPS = 32;

export interface DelayInfo {
  /** Delayed update in use for the local viscosity / the long-range term. */
  local: boolean;
  kernel: boolean;
  tSelf: number;
  tNb: number;
  /** Longest delay max(t_self, t_nb) the history must cover (s; 0 if nothing is delayed). */
  tauMax: number;
  /** History sampling interval (s). */
  interval: number;
  /** Frames needed / stored / capacity. */
  framesNeeded: number;
  framesStored: number;
  capacity: number;
  /** Bytes currently held by the history. */
  bytes: number;
  /** ν Δt / h² (explicit limit 1/4) and the number of explicit sub-steps used for it. */
  localNumber: number;
  localSubsteps: number;
  /** Rate × Δt of the long-range term: S Δt, or ν_s G₀ Δt for the fractional Laplacian. */
  kernelNumber: number;
}

export interface DelayStability {
  /** Local viscosity: A = 4ν/h², mode parameter c = (cos k_x h + cos k_y h)/2. */
  local: ModeScan | null;
  /** Long-range term: A = S (or ν_s G₀), mode parameter q = K̂(k) (or 1 − λ^s/G₀). */
  kernel: ModeScan | null;
}

export type RGB = readonly [number, number, number];

export class FluidSolver {
  readonly grid: Grid;
  params: FluidParams;

  /** Face velocities (domain units / s) on the staggered grid. */
  readonly u: Float32Array;
  readonly v: Float32Array;
  /** Dye concentration, one cell-centred field per RGB channel. */
  readonly dye: [Float32Array, Float32Array, Float32Array];
  /** Pressure from the last projection (ρ = 1). */
  readonly pressure: Float32Array;

  /** Diagnostics, refreshed by updateDiagnostics(). */
  readonly uc: Float32Array;
  readonly vc: Float32Array;
  readonly divergence: Float32Array;
  readonly vorticity: Float32Array;

  private readonly u0: Float32Array;
  private readonly v0: Float32Array;
  private readonly tmp: Float32Array;
  private readonly scratch: {
    uc: Float32Array; vc: Float32Array; curl: Float32Array; fx: Float32Array; fy: Float32Array;
  };

  private nl?: { key: string; u: NonlocalOperator; v: NonlocalOperator };
  /** Past velocity fields of the delayed viscosity terms (cleared on reset / boundary change). */
  private readonly hist = new FieldHistory();
  /** Departure maps of the path-following delay (delayMode 'lagrangian'). */
  readonly tracer = new PathTracer();
  /** Cached K*F of history frames (Lagrangian delay), keyed by frame array, frame time and operator. */
  private convCache: { src: ArrayLike<number>; t: number; ops: unknown; wU: Float64Array; wV: Float64Array }[] = [];
  private lbuf?: { depSU: Float64Array; depSV: Float64Array; depNU: Float64Array; depNV: Float64Array; wU: Float64Array; wV: Float64Array; sU: Float64Array; sV: Float64Array; qU: Float64Array; qV: Float64Array;
    fU: Float64Array; fV: Float64Array; cU: Float64Array; cV: Float64Array };
  private dbuf?: { f0u: Float32Array; f0v: Float32Array; nbU: Float64Array; nbV: Float64Array;
    selfU: Float64Array; selfV: Float64Array; sumU: Float64Array; sumV: Float64Array };
  private stabCache?: { key: string; value: DelayStability };

  /** Simulated time (s) and step counter. */
  time = 0;
  steps = 0;

  constructor(nx: number, ny: number, params: Partial<FluidParams> = {}) {
    const g = (this.grid = createGrid(nx, ny));
    this.params = { ...defaultParams, ...params };
    this.u = createUField(g);
    this.v = createVField(g);
    this.u0 = createUField(g);
    this.v0 = createVField(g);
    this.dye = [createField(g), createField(g), createField(g)];
    this.pressure = createField(g);
    this.uc = createField(g);
    this.vc = createField(g);
    this.divergence = createField(g);
    this.vorticity = createField(g);
    this.tmp = createField(g);
    this.scratch = {
      uc: createField(g), vc: createField(g), curl: createField(g),
      fx: createField(g), fy: createField(g),
    };
  }

  setParams(p: Partial<FluidParams>): void {
    const before = this.params.walls;
    this.params = { ...this.params, ...p };
    const walls = this.params.walls;
    const g = this.grid;
    if (walls !== before) {
      // Switching boundaries: the old edge faces become walls (zeroed) or seam
      // unknowns; make the ghosts consistent and re-project so ∇·u = 0 under
      // the new conditions.
      if (periodicX(before) && !periodicX(walls)) {
        for (let j = 1; j <= g.ny; j++) this.u[uIdx(g, g.nx, j)] = 0;
      }
      if (periodicY(before) && !periodicY(walls)) {
        for (let i = 1; i <= g.nx; i++) this.v[vIdx(g, i, g.ny)] = 0;
      }
      this.clearDelayHistory();
      for (const d of this.dye) setScalarBoundary(g, d, walls);
      setUBoundary(g, this.u, walls);
      setVBoundary(g, this.v, walls);
      project(g, this.u, this.v, this.pressure, this.tmp, this.params.dt, 4 * this.params.iterations,
        walls, this.params.sor);
      return;
    }
    setUBoundary(g, this.u, walls);
    setVBoundary(g, this.v, walls);
  }

  /** Zero every field. */
  reset(): void {
    for (const f of [
      this.u, this.v, this.u0, this.v0, this.pressure, this.uc, this.vc,
      this.divergence, this.vorticity, this.tmp, ...this.dye,
    ]) f.fill(0);
    this.time = 0;
    this.steps = 0;
    this.clearDelayHistory();
  }

  /** Forget the stored velocity history of the delayed terms. */
  clearDelayHistory(): void {
    this.hist.clear();
    this.tracer.reset();
    this.convCache = [];
  }

  /** Effective delays (non-negative, finite). */
  private delays(): [number, number] {
    const f = (x: number) => (x > 0 && Number.isFinite(x) ? x : 0);
    return [f(this.params.delaySelf), f(this.params.delayNb)];
  }

  /** The long-range term is active at all (fractional only on the torus). */
  private kernelActive(): boolean {
    const p = this.params;
    if (p.kernel === 'none' || !(p.kernelStrength > 0)) return false;
    return p.kernel !== 'fractional' || p.walls === 'periodic';
  }

  /** True if the local viscosity ν∇²u uses the delayed (explicit) update. */
  localDelayActive(): boolean {
    const [ts, tn] = this.delays();
    return this.params.delayLocal && this.params.viscosity > 0 && (ts > 0 || tn > 0);
  }

  /** True if the long-range term uses the delayed (explicit) update. */
  kernelDelayActive(): boolean {
    const [ts, tn] = this.delays();
    return this.params.delayKernel && this.kernelActive() && (ts > 0 || tn > 0);
  }

  /** True if the next step uses any delayed term. */
  delayActive(): boolean {
    return this.localDelayActive() || this.kernelDelayActive();
  }

  /** ν Δt / h² and the explicit sub-step count of the delayed local viscosity. */
  private localSubsteps(): { number: number; n: number } {
    const p = this.params;
    const num = (p.viscosity * p.dt) / (this.grid.h * this.grid.h);
    const [ts] = this.delays();
    // t_self = 0: own velocity treated implicitly, one step is enough
    const n = ts > 0 ? Math.min(MAX_LOCAL_SUBSTEPS, Math.max(1, Math.ceil(num / LOCAL_SUBSTEP_NUMBER))) : 1;
    return { number: num, n };
  }

  /** Delays, history memory and explicit-update numbers (for the UI). */
  delayInfo(): DelayInfo {
    const p = this.params;
    const [ts, tn] = this.delays();
    const local = this.localDelayActive(), kernel = this.kernelDelayActive();
    const tauMax = local || kernel ? Math.max(ts, tn) : 0;
    const h = this.hist;
    const interval = h.interval || p.dt;
    const ls = this.localSubsteps();
    let kernelNumber = 0;
    if (this.kernelActive()) {
      kernelNumber = p.kernelStrength * p.dt * (p.kernel === 'fractional' ? this.kernelOperators().u.diagonal() : 1);
    }
    return {
      local, kernel, tSelf: ts, tNb: tn, tauMax, interval,
      framesNeeded: tauMax > 0 ? Math.ceil(tauMax / interval) + 1 : 0,
      framesStored: h.frames.length, capacity: h.capacity, bytes: h.bytes(),
      localNumber: ls.number, localSubsteps: ls.n, kernelNumber,
    };
  }

  /**
   * Predicted linear stability of the delayed terms for the current settings:
   * the most unstable Fourier mode of û′ = −A û(t − t_self) + B û(t − t_nb)
   * for each component (see stability.ts). null where a component is off or
   * not delayed. Cached per parameter set.
   */
  delayStability(): DelayStability {
    const p = this.params;
    const [ts, tn] = this.delays();
    const g = this.grid;
    const local = p.delayLocal && p.viscosity > 0 && (ts > 0 || tn > 0);
    const kern = p.delayKernel && this.kernelActive() && (ts > 0 || tn > 0);
    const key = JSON.stringify([ts, tn, local, kern, p.viscosity, p.kernel, p.kernelStrength, p.kernelRange,
      p.kernelBeta, p.kernelAlpha, p.kernelS, p.walls, g.nx, g.ny]);
    if (this.stabCache?.key === key) return this.stabCache.value;
    const value: DelayStability = { local: null, kernel: null };
    const nMax = Math.max(g.nx, g.ny);
    if (local) {
      // longest resolved wave: one period across the domain (torus) / half a period (walls)
      const kh = (isPeriodic(p.walls) ? 2 : 1) * Math.PI / nMax;
      const cMax = (1 + Math.cos(kh)) / 2;
      value.local = scanModes((4 * p.viscosity) / (g.h * g.h), -1, cMax, ts, tn, 41);
    }
    if (kern) {
      const op = this.kernelOperators().u;
      const [lo, hi] = op.modeRatioRange();
      const A = p.kernelStrength * op.diagonal();
      value.kernel = scanModes(A, lo, hi, ts, tn, 41);
    }
    this.stabCache = { key, value };
    return value;
  }

  /**
   * Add a Gaussian-weighted velocity impulse (du, dv) [domain units / s] and
   * optionally dye at grid position (X, Y) ∈ [0, nx] × [0, ny] with the given
   * radius in cells.
   */
  splat(X: number, Y: number, du: number, dv: number, color: RGB | null, radius: number): void {
    if (isPeriodic(this.params.walls)) {
      this.brushPeriodic(X, Y, radius, (cu, cv, cd) => {
        if (du !== 0) cu((k, w) => { this.u[k] += du * w; });
        if (dv !== 0) cv((k, w) => { this.v[k] += dv * w; });
        if (color) cd((k, w) => {
          this.dye[0][k] += color[0] * w;
          this.dye[1][k] += color[1] * w;
          this.dye[2][k] += color[2] * w;
        });
      });
      return;
    }
    const g = this.grid;
    const r = Math.max(radius, 0.5);
    const inv2r2 = 1 / (2 * r * r);
    const reach = Math.ceil(3 * r);
    const i0 = Math.max(0, Math.floor(X) - reach);
    const i1 = Math.min(g.nx + 1, Math.ceil(X) + reach);
    const j0 = Math.max(0, Math.floor(Y) - reach);
    const j1 = Math.min(g.ny + 1, Math.ceil(Y) + reach);
    const weight = (x: number, y: number): number => {
      const dx = x - X;
      const dy = y - Y;
      return Math.exp(-(dx * dx + dy * dy) * inv2r2);
    };
    if (du !== 0) {
      for (let j = Math.max(1, j0); j <= Math.min(g.ny, j1); j++)
        for (let i = Math.max(1, i0); i <= Math.min(g.nx - 1, i1); i++)
          this.u[uIdx(g, i, j)] += du * weight(i, j - 0.5);
    }
    if (dv !== 0) {
      for (let j = Math.max(1, j0); j <= Math.min(g.ny - 1, j1); j++)
        for (let i = Math.max(1, i0); i <= Math.min(g.nx, i1); i++)
          this.v[vIdx(g, i, j)] += dv * weight(i - 0.5, j);
    }
    if (color) {
      for (let j = Math.max(1, j0); j <= Math.min(g.ny, j1); j++)
        for (let i = Math.max(1, i0); i <= Math.min(g.nx, i1); i++) {
          const w = weight(i - 0.5, j - 0.5);
          const k = idx(g, i, j);
          this.dye[0][k] += color[0] * w;
          this.dye[1][k] += color[1] * w;
          this.dye[2][k] += color[2] * w;
        }
    }
  }

  /**
   * "Paddle" forcing: blend the velocity inside a Gaussian brush toward the
   * target velocity (tu, tv). Unlike splat() this cannot accumulate without
   * bound when the brush lingers, which makes mouse interaction well-behaved.
   * `strength` ∈ (0, 1] is the blend factor at the brush centre.
   */
  stir(X: number, Y: number, tu: number, tv: number, radius: number, strength = 1): void {
    const a = Math.min(Math.max(strength, 0), 1);
    if (isPeriodic(this.params.walls)) {
      this.brushPeriodic(X, Y, radius, (cu, cv) => {
        cu((k, w) => { this.u[k] += a * w * (tu - this.u[k]); });
        cv((k, w) => { this.v[k] += a * w * (tv - this.v[k]); });
      });
      return;
    }
    const g = this.grid;
    const r = Math.max(radius, 0.5);
    const inv2r2 = 1 / (2 * r * r);
    const reach = Math.ceil(3 * r);
    for (let j = Math.max(1, Math.floor(Y) - reach); j <= Math.min(g.ny, Math.ceil(Y) + reach); j++)
      for (let i = Math.max(1, Math.floor(X) - reach); i <= Math.min(g.nx - 1, Math.ceil(X) + reach); i++) {
        const dx = i - X;
        const dy = j - 0.5 - Y;
        const w = a * Math.exp(-(dx * dx + dy * dy) * inv2r2);
        const k = uIdx(g, i, j);
        this.u[k] += w * (tu - this.u[k]);
      }
    for (let j = Math.max(1, Math.floor(Y) - reach); j <= Math.min(g.ny - 1, Math.ceil(Y) + reach); j++)
      for (let i = Math.max(1, Math.floor(X) - reach); i <= Math.min(g.nx, Math.ceil(X) + reach); i++) {
        const dx = i - 0.5 - X;
        const dy = j - Y;
        const w = a * Math.exp(-(dx * dx + dy * dy) * inv2r2);
        const k = vIdx(g, i, j);
        this.v[k] += w * (tv - this.v[k]);
      }
  }

  /**
   * Gaussian brush on a domain with a periodic axis: the brush wraps across
   * periodic seams (each physical face/cell is visited at most once, with the
   * weight of its nearest periodic image) and is clipped at walls exactly as
   * in the solid-wall case. `body` receives three visitors for u-faces,
   * v-faces and cells; each calls back with (array index, weight).
   */
  private brushPeriodic(
    X: number, Y: number, radius: number,
    body: (
      onU: (f: (k: number, w: number) => void) => void,
      onV: (f: (k: number, w: number) => void) => void,
      onCell: (f: (k: number, w: number) => void) => void,
    ) => void,
  ): void {
    const g = this.grid;
    const { nx, ny } = g;
    const walls = this.params.walls;
    const px = periodicX(walls);
    const py = periodicY(walls);
    const r = Math.max(radius, 0.5);
    const inv2r2 = 1 / (2 * r * r);
    const reach = Math.ceil(3 * r);
    /**
     * Index window along one axis: list of [storage index, position]. On a
     * periodic axis positions are unwrapped around the centre, indices are
     * wrapped into lo..lo+n−1 and at most n entries are produced; on a wall
     * axis the window is clipped to [lo, hi].
     */
    const axis = (c: number, offset: number, n: number, periodic: boolean, lo: number, hi: number) => {
      const out: [number, number][] = [];
      const a0 = Math.floor(c) - reach;
      const a1 = Math.ceil(c) + reach;
      if (periodic) {
        for (let m = a0; m <= Math.min(a1, a0 + n - 1); m++) {
          out.push([lo + ((((m - lo) % n) + n) % n), m + offset]);
        }
      } else {
        for (let m = Math.max(lo, a0); m <= Math.min(hi, a1); m++) out.push([m, m + offset]);
      }
      return out;
    };
    const visit = (xs: [number, number][], ys: [number, number][], stride: number,
      f: (k: number, w: number) => void): void => {
      for (const [j, y] of ys) {
        const dy = y - Y;
        for (const [i, x] of xs) {
          const dx = x - X;
          f(i + stride * j, Math.exp(-(dx * dx + dy * dy) * inv2r2));
        }
      }
    };
    // u-faces: x at i (unknowns 1..nx on x-periodic), y at j − ½
    const uX = axis(X, 0, nx, px, 1, nx - 1);
    const cX = axis(X + 0.5, -0.5, nx, px, 1, nx); // cells and v-faces: x at i − ½
    const cY = axis(Y + 0.5, -0.5, ny, py, 1, ny); // cells and u-faces: y at j − ½
    const vY = axis(Y, 0, ny, py, 1, ny - 1); // v-faces: y at j
    body(
      (f) => visit(uX, cY, g.uStride, f),
      (f) => visit(cX, vY, g.vStride, f),
      (f) => visit(cX, cY, g.stride, f),
    );
    setUBoundary(g, this.u, walls);
    setVBoundary(g, this.v, walls);
    for (const d of this.dye) setScalarBoundary(g, d, walls);
  }

  /**
   * Copy the state of another solver (possibly with a different grid size)
   * into this one by bilinear resampling, then re-project. Used when the
   * resolution or window aspect ratio changes.
   */
  resampleFrom(src: FluidSolver): void {
    const g = this.grid;
    const sg = src.grid;
    const sx = sg.nx / g.nx;
    const sy = sg.ny / g.ny;
    // Velocities are in domain units/s and the shorter domain side is always 1,
    // so they carry over unchanged (a changed aspect ratio simply stretches the
    // old state onto the new domain).
    // On periodic axes the source is sampled with wrap-around (seam faces too).
    const sw = isPeriodic(src.params.walls) ? src.params.walls : undefined;
    for (let j = 1; j <= g.ny; j++)
      for (let i = 0; i <= g.nx; i++)
        this.u[uIdx(g, i, j)] = sampleU(sg, src.u, i * sx, (j - 0.5) * sy, sw);
    for (let j = 0; j <= g.ny; j++)
      for (let i = 1; i <= g.nx; i++)
        this.v[vIdx(g, i, j)] = sampleV(sg, src.v, (i - 0.5) * sx, j * sy, sw);
    for (let c = 0; c < 3; c++)
      for (let j = 1; j <= g.ny; j++)
        for (let i = 1; i <= g.nx; i++)
          this.dye[c][idx(g, i, j)] = sampleCell(sg, src.dye[c], (i - 0.5) * sx, (j - 0.5) * sy, sw);
    for (const d of this.dye) setScalarBoundary(g, d, this.params.walls);
    setUBoundary(g, this.u, this.params.walls);
    setVBoundary(g, this.v, this.params.walls);
    project(g, this.u, this.v, this.pressure, this.tmp, this.params.dt, 4 * this.params.iterations,
      this.params.walls, this.params.sor);
    this.time = src.time;
    this.steps = src.steps;
  }

  /** Build (or reuse) the spectral operators for the current kernel / grid / boundaries. */
  kernelOperators(): { u: NonlocalOperator; v: NonlocalOperator } {
    const p = this.params;
    const kp: KernelParams = {
      kernel: p.kernel, strength: p.kernelStrength, range: p.kernelRange, beta: p.kernelBeta,
      alpha: p.kernelAlpha, s: p.kernelS, motschTadmor: p.kernelMT,
    };
    // strength only enters at apply time; everything else determines the symbol
    const key = JSON.stringify([kp.kernel, kp.range, kp.beta, kp.alpha, kp.s, p.walls]);
    if (!this.nl || this.nl.key !== key) {
      const g = this.grid;
      const px = periodicX(p.walls);
      const py = periodicY(p.walls);
      this.nl = {
        key,
        u: new NonlocalOperator(px ? g.nx : g.nx - 1, g.ny, px, py, g.h, kp),
        v: new NonlocalOperator(g.nx, py ? g.ny : g.ny - 1, px, py, g.h, kp),
      };
    }
    this.nl.u.params.strength = kp.strength;
    this.nl.u.params.motschTadmor = kp.motschTadmor;
    // u and v share one params object
    return this.nl;
  }

  /** Apply the nonlocal viscosity / alignment term over one time step. */
  private applyKernel(): void {
    const g = this.grid;
    const { u, v } = this;
    const { dt, walls } = this.params;
    const ops = this.kernelOperators();
    ops.u.apply((i, j) => u[uIdx(g, i + 1, j + 1)], (i, j, x) => { u[uIdx(g, i + 1, j + 1)] = x; }, dt);
    ops.v.apply((i, j) => v[vIdx(g, i + 1, j + 1)], (i, j, x) => { v[vIdx(g, i + 1, j + 1)] = x; }, dt);
    setUBoundary(g, u, walls);
    setVBoundary(g, v, walls);
  }

  /**
   * Delayed viscosity (any of the two components with t_self or t_nb > 0).
   * Each component splits into a neighbour part read t_nb ago and a self part
   * read t_self ago, both at the step midpoint t + Δt/2 − τ (reads at or after
   * t use the current field):
   *   long-range: u ← u + a (K * u_nb − c u_self)            (NonlocalOperator.applyDelayed)
   *   local:      u ← u + (νΔt/h²)(Σ_nb u_nb − 4 u_self)     (explicit sub-steps if νΔt/h² > 0.2)
   * With t_self = 0 the own velocity is current and is treated implicitly,
   * u ← (u + a w)/(1 + a c) — a convex combination, stable for any Δt. The
   * field at the start of the delayed update is recorded into the history
   * afterwards; every face is finally clamped to ±DELAY_SPEED_LIMIT.
   */
  private applyDelayedViscosity(local: boolean, kernel: boolean): void {
    const g = this.grid;
    const { u, v } = this;
    const { dt, walls, viscosity } = this.params;
    const [ts, tn] = this.delays();
    const t = this.time;
    const hist = this.hist;
    hist.configure(u.length, v.length, dt, Math.max(ts, tn), DELAY_BUDGET_BYTES);
    const b = (this.dbuf ??= {
      f0u: new Float32Array(u.length), f0v: new Float32Array(v.length),
      nbU: new Float64Array(u.length), nbV: new Float64Array(v.length),
      selfU: new Float64Array(u.length), selfV: new Float64Array(v.length),
      sumU: new Float64Array(u.length), sumV: new Float64Array(v.length),
    });
    b.f0u.set(u);
    b.f0v.set(v);
    const L = DELAY_SPEED_LIMIT;
    const clamp = (x: number) => (x > L ? L : x < -L ? -L : x);
    const implicitSelf = ts === 0;
    /** Read the fields at T into (outU, outV); T ≥ t → the current (evolving) field. */
    const read = (T: number, outU: Float64Array, outV: Float64Array) => {
      if (T >= t) { for (let k = 0; k < u.length; k++) outU[k] = u[k]; for (let k = 0; k < v.length; k++) outV[k] = v[k]; return; }
      hist.read(T, t, b.f0u, b.f0v, outU, outV);
    };
    if (this.params.delayMode === 'lagrangian') {
      this.applyDelayedLagrangian(local, kernel, clamp);
      if (hist.wants(t)) hist.push(t, b.f0u, b.f0v);
      hist.prune(t + dt);
      return;
    }
    if (kernel) {
      const tMid = t + 0.5 * dt;
      read(Math.min(t, tMid - tn), b.nbU, b.nbV);
      if (!implicitSelf) read(Math.min(t, tMid - ts), b.selfU, b.selfV);
      const ops = this.kernelOperators();
      const { nbU, nbV, selfU, selfV } = b;
      ops.u.applyDelayed(
        (i, j) => nbU[uIdx(g, i + 1, j + 1)], implicitSelf ? null : (i, j) => selfU[uIdx(g, i + 1, j + 1)],
        (i, j) => u[uIdx(g, i + 1, j + 1)], (i, j, x) => { u[uIdx(g, i + 1, j + 1)] = clamp(x); }, dt);
      ops.v.applyDelayed(
        (i, j) => nbV[vIdx(g, i + 1, j + 1)], implicitSelf ? null : (i, j) => selfV[vIdx(g, i + 1, j + 1)],
        (i, j) => v[vIdx(g, i + 1, j + 1)], (i, j, x) => { v[vIdx(g, i + 1, j + 1)] = clamp(x); }, dt);
      setUBoundary(g, u, walls);
      setVBoundary(g, v, walls);
    }
    if (local) {
      const n = this.localSubsteps().n;
      const ds = dt / n;
      const a = (viscosity * ds) / (g.h * g.h);
      const iLast = uLastI(g, walls), jLast = vLastJ(g, walls);
      const { nbU, nbV, selfU, selfV, sumU, sumV } = b;
      for (let m = 0; m < n; m++) {
        const tm = t + (m + 0.5) * ds;
        read(tm - tn, nbU, nbV);
        neighbourSumU(g, nbU, sumU, walls);
        neighbourSumV(g, nbV, sumV, walls);
        if (implicitSelf) {
          const inv = 1 / (1 + 4 * a);
          for (let j = 1; j <= g.ny; j++) for (let i = 1; i <= iLast; i++) {
            const k = uIdx(g, i, j);
            u[k] = clamp((u[k] + a * sumU[k]) * inv);
          }
          for (let j = 1; j <= jLast; j++) for (let i = 1; i <= g.nx; i++) {
            const k = vIdx(g, i, j);
            v[k] = clamp((v[k] + a * sumV[k]) * inv);
          }
        } else {
          read(tm - ts, selfU, selfV);
          for (let j = 1; j <= g.ny; j++) for (let i = 1; i <= iLast; i++) {
            const k = uIdx(g, i, j);
            u[k] = clamp(u[k] + a * (sumU[k] - 4 * selfU[k]));
          }
          for (let j = 1; j <= jLast; j++) for (let i = 1; i <= g.nx; i++) {
            const k = vIdx(g, i, j);
            v[k] = clamp(v[k] + a * (sumV[k] - 4 * selfV[k]));
          }
        }
        setUBoundary(g, u, walls);
        setVBoundary(g, v, walls);
      }
    }
    if (hist.wants(t)) hist.push(t, b.f0u, b.f0v);
    hist.prune(t + dt);
  }

  /**
   * Path-following variant of applyDelayedViscosity: the delayed quantities are
   * sampled at the departure points X(t_m − τ) of the parcels now at each face
   * (t_m = t + Δt/2), computed by the PathTracer from the velocity history:
   *   kernel: u ← u + SΔt [ (K*u)(X_nb, t_m − t_nb) − c u(X_self, t_m − t_self) ]
   *   local:  u ← u + νΔt_s/h² [ Σ_nb u(·, t_s − t_nb)|_{X_nb} − 4 u(X_self, t_s − t_self) ]
   * (the neighbour quantity is formed on the grid at the past time, then
   * sampled where the parcel was). t_self = 0: semi-implicit self term as in the
   * Eulerian path. Local sub-steps reuse the departure points of the step midpoint.
   */
  private applyDelayedLagrangian(
    local: boolean, kernel: boolean, clamp: (x: number) => number,
  ): void {
    const g = this.grid;
    const { u, v } = this;
    const { dt, walls, viscosity } = this.params;
    const [ts, tn] = this.delays();
    const t = this.time;
    const b = this.dbuf!;
    const hist = this.hist;
    if (this.lbuf && (this.lbuf.wU.length !== u.length || this.lbuf.wV.length !== v.length)) this.lbuf = undefined;
    const lb = (this.lbuf ??= {
      depSU: new Float64Array(4 * u.length), depSV: new Float64Array(4 * v.length),
      depNU: new Float64Array(4 * u.length), depNV: new Float64Array(4 * v.length),
      wU: new Float64Array(u.length), wV: new Float64Array(v.length),
      sU: new Float64Array(u.length), sV: new Float64Array(v.length),
      qU: new Float64Array(u.length), qV: new Float64Array(v.length),
      fU: new Float64Array(u.length), fV: new Float64Array(v.length),
      cU: new Float64Array(u.length), cV: new Float64Array(v.length),
    });
    const cu = lb.cU, cv = lb.cV; // Float64 copy of the step-start field
    cu.set(b.f0u); cv.set(b.f0v);
    const tr = this.tracer;
    const implicitSelf = ts === 0;
    const curRead = (T: number, oU: Float64Array, oV: Float64Array) => {
      if (T >= t) { for (let k = 0; k < u.length; k++) oU[k] = b.f0u[k]; for (let k = 0; k < v.length; k++) oV[k] = b.f0v[k]; return; }
      hist.read(T, t, b.f0u, b.f0v, oU, oV);
    };
    tr.departures(g, walls, tn, t, dt, cu, cv, curRead, lb.depNU, lb.depNV);
    if (!implicitSelf) {
      if (ts === tn) { lb.depSU.set(lb.depNU); lb.depSV.set(lb.depNV); }
      else tr.departures(g, walls, ts, t, dt, cu, cv, curRead, lb.depSU, lb.depSV);
    }
    tr.retain([ts, tn]);
    /**
     * (oU, oV) = quantity at time T sampled along the parcels' paths: for each
     * bracketing history frame F_f (time T_f, weight w_f of the linear time
     * interpolation), w_f · Q(F_f)(X(t − τ) + (T_f − t + τ)·w). `make` turns a
     * frame into the face fields to sample (identity, K*u, or the stencil sum).
     */
    const pathRead = (
      tau: number, T: number, depU: Float64Array, depV: Float64Array, oU: Float64Array, oV: Float64Array,
      make: (fu: ArrayLike<number>, fv: ArrayLike<number>, tf: number) => [ArrayLike<number>, ArrayLike<number>],
    ) => {
      const l = hist.lookup(T, t);
      const parts: [number, number][] = l.theta > 0 ? [[l.a, 1 - l.theta], [l.b, l.theta]] : [[l.a, 1]];
      parts.forEach(([fi, wgt], n) => {
        const fr = fi < 0 ? null : hist.frames[fi];
        const [qu, qv] = make(fr ? fr.u : b.f0u, fr ? fr.v : b.f0v, fr ? fr.t : NaN);
        const sh = (fr ? fr.t : t) - (t - tau);
        tr.sampleAt(0, qu, depU, sh, wgt, oU, n > 0);
        tr.sampleAt(1, qv, depV, sh, wgt, oV, n > 0);
      });
    };
    // copy Float32 frames to Float64 so the samplers stay monomorphic (≈2× faster)
    const ident = (fu: ArrayLike<number>, fv: ArrayLike<number>): [ArrayLike<number>, ArrayLike<number>] => {
      for (let k = 0; k < u.length; k++) lb.fU[k] = fu[k];
      for (let k = 0; k < v.length; k++) lb.fV[k] = fv[k];
      return [lb.fU, lb.fV];
    };
    if (kernel) {
      const tMid = t + 0.5 * dt;
      const ops = this.kernelOperators();
      // K*F of a stored frame never changes: cache it by (frame array, frame time,
      // operator) — the newer frame of this step is the older one of the next,
      // so one convolution per step suffices. The current field (tf = NaN) is never cached.
      const cache = this.convCache;
      const conv = (fu: ArrayLike<number>, fv: ArrayLike<number>, tf: number): [ArrayLike<number>, ArrayLike<number>] => {
        const keep = !Number.isNaN(tf);
        const hit = keep ? cache.find((c) => c.src === fu && c.t === tf && c.ops === ops) : undefined;
        if (hit) return [hit.wU, hit.wV];
        let wU = lb.wU, wV = lb.wV;
        if (keep) {
          const e = cache.length >= 3 ? cache.shift()! : null;
          wU = e && e.wU.length === u.length ? e.wU : new Float64Array(u.length);
          wV = e && e.wV.length === v.length ? e.wV : new Float64Array(v.length);
        }
        wU.fill(0); wV.fill(0);
        const on = ops.u.neighbourField((i, j) => fu[uIdx(g, i + 1, j + 1)], (i, j, w) => { wU[uIdx(g, i + 1, j + 1)] = w; });
        ops.v.neighbourField((i, j) => fv[vIdx(g, i + 1, j + 1)], (i, j, w) => { wV[vIdx(g, i + 1, j + 1)] = w; });
        if (keep && on) cache.push({ src: fu, t: tf, ops, wU, wV });
        return [wU, wV];
      };
      pathRead(tn, Math.min(t, tMid - tn), lb.depNU, lb.depNV, lb.qU, lb.qV, conv);
      if (!implicitSelf) pathRead(ts, Math.min(t, tMid - ts), lb.depSU, lb.depSV, lb.sU, lb.sV, ident);
      const { qU, qV, sU, sV } = lb;
      ops.u.applyDelayedGiven(
        (i, j) => qU[uIdx(g, i + 1, j + 1)], implicitSelf ? null : (i, j) => sU[uIdx(g, i + 1, j + 1)],
        (i, j) => u[uIdx(g, i + 1, j + 1)], (i, j, x) => { u[uIdx(g, i + 1, j + 1)] = clamp(x); }, dt);
      ops.v.applyDelayedGiven(
        (i, j) => qV[vIdx(g, i + 1, j + 1)], implicitSelf ? null : (i, j) => sV[vIdx(g, i + 1, j + 1)],
        (i, j) => v[vIdx(g, i + 1, j + 1)], (i, j, x) => { v[vIdx(g, i + 1, j + 1)] = clamp(x); }, dt);
      setUBoundary(g, u, walls);
      setVBoundary(g, v, walls);
    }
    if (local) {
      const n = this.localSubsteps().n;
      const ds = dt / n;
      const a = (viscosity * ds) / (g.h * g.h);
      const iLast = uLastI(g, walls), jLast = vLastJ(g, walls);
      const { sumU, sumV } = b;
      const { qU, qV, sU, sV } = lb;
      const nbSum = (fu: ArrayLike<number>, fv: ArrayLike<number>): [ArrayLike<number>, ArrayLike<number>] => {
        neighbourSumU(g, fu, sumU, walls); neighbourSumV(g, fv, sumV, walls); return [sumU, sumV];
      };
      for (let m = 0; m < n; m++) {
        const tm = t + (m + 0.5) * ds;
        // reads at or after t use the evolving field at the grid point (as in the Eulerian path)
        if (tm - tn >= t) { nbSum(u, v); qU.set(sumU); qV.set(sumV); }
        else pathRead(tn, tm - tn, lb.depNU, lb.depNV, qU, qV, nbSum);
        if (implicitSelf) {
          const inv = 1 / (1 + 4 * a);
          for (let j = 1; j <= g.ny; j++) for (let i = 1; i <= iLast; i++) { const k = uIdx(g, i, j); u[k] = clamp((u[k] + a * qU[k]) * inv); }
          for (let j = 1; j <= jLast; j++) for (let i = 1; i <= g.nx; i++) { const k = vIdx(g, i, j); v[k] = clamp((v[k] + a * qV[k]) * inv); }
        } else {
          if (tm - ts >= t) { for (let k = 0; k < u.length; k++) sU[k] = u[k]; for (let k = 0; k < v.length; k++) sV[k] = v[k]; }
          else pathRead(ts, tm - ts, lb.depSU, lb.depSV, sU, sV, ident);
          for (let j = 1; j <= g.ny; j++) for (let i = 1; i <= iLast; i++) { const k = uIdx(g, i, j); u[k] = clamp(u[k] + a * (qU[k] - 4 * sU[k])); }
          for (let j = 1; j <= jLast; j++) for (let i = 1; i <= g.nx; i++) { const k = vIdx(g, i, j); v[k] = clamp(v[k] + a * (qV[k] - 4 * sV[k])); }
        }
        setUBoundary(g, u, walls);
        setVBoundary(g, v, walls);
      }
    }
  }

  /** Advance the simulation by one time step of length params.dt. */
  step(): void {
    const g = this.grid;
    const {
      viscosity, diffusion, dt, iterations, sor, vorticity, dyeDissipation, walls,
      selfPropulsion: active, selfPropTau, selfPropV0,
    } = this.params;
    const { u, v, u0, v0, pressure, tmp } = this;

    // 1. forces
    vorticityConfinement(g, u, v, vorticity, dt, this.scratch, walls);
    setUBoundary(g, u, walls);
    setVBoundary(g, v, walls);

    // 2. viscosity (instantaneous parts: implicit / exact, unchanged)
    const localDelayed = this.localDelayActive();
    const kernelDelayed = this.kernelDelayActive();
    if (viscosity > 0 && !localDelayed) {
      u0.set(u);
      v0.set(v);
      diffuseU(g, u, u0, viscosity, dt, iterations, walls);
      diffuseV(g, v, v0, viscosity, dt, iterations, walls);
    }

    // 2b. nonlocal (long-range) viscosity / alignment
    if (!kernelDelayed && this.params.kernel !== 'none') this.applyKernel();

    // 2c. delayed viscosity terms (own velocity t_self ago, neighbours t_nb ago)
    if (localDelayed || kernelDelayed) this.applyDelayedViscosity(localDelayed, kernelDelayed);
    // history only stays valid while it is recorded every step
    else if (this.hist.frames.length) this.clearDelayHistory();

    // 3. projection
    project(g, u, v, pressure, tmp, dt, iterations, walls, sor);

    // 4. self-advection
    u0.set(u);
    v0.set(v);
    advectVelocity(g, u, v, u0, v0, dt, walls);

    // 5. self-propulsion (active fluid): |u| → v₀ + (|u| − v₀)·exp(−Δt/τ) along u/|u|
    if (active) selfPropulsion(g, u, v, u0, v0, selfPropTau, selfPropV0, dt, walls);

    // 6. projection (restores ∇·u = 0 after the active forcing)
    project(g, u, v, pressure, tmp, dt, iterations, walls, sor);

    // 7. dye
    const fade = dyeDissipation > 0 ? Math.exp(-dyeDissipation * dt) : 1;
    const periodic = isPeriodic(walls);
    for (const d of this.dye) {
      if (periodic) setScalarBoundary(g, d, walls); // wrapped ghosts for the stencils
      if (diffusion > 0) {
        tmp.set(d);
        diffuseScalar(g, d, tmp, diffusion, dt, iterations, walls);
      }
      tmp.set(d);
      advectScalar(g, d, tmp, u, v, dt, walls);
      if (fade !== 1) for (let k = 0; k < d.length; k++) d[k] *= fade;
    }

    this.time += dt;
    this.steps++;
  }

  /**
   * Seed a weak random, divergence-free velocity field (sum of random
   * vortices, deterministic for a given seed) plus a dye pattern, so that the
   * self-propulsion term has a direction to amplify ("active turbulence").
   * `amplitude` is the rough peak speed in domain units / s.
   */
  /**
   * Ordered flock + localized disordered cluster: uniform u = v0 (heading in
   * radians, 0 = +x) plus a Gaussian-tapered patch of random swirls. ψ_patch =
   * env(x)·Σ vortices is a stream function, so the field stays exactly
   * divergence free (periodic, nearest image). The seed sets the patch centre
   * and the swirl positions / signs. sigma = patch radius (cells, Gaussian σ),
   * amplitude = swirl speed scale relative to v0, core = swirl core radius (cells).
   */
  seedOrderedPerturbation(seed = 1, opts: { v0?: number; heading?: number; sigma?: number; amplitude?: number; core?: number; density?: number } = {}): void {
    const g = this.grid; const { nx, ny, h } = g;
    const v0 = opts.v0 ?? this.params.selfPropV0, heading = opts.heading ?? 0;
    const sigma = opts.sigma ?? 6.6, amp = (opts.amplitude ?? 2) * v0, core = opts.core ?? 2.5, density = opts.density ?? 1;
    let a = seed >>> 0;
    const rand = (): number => {
      a = (a + 0x6d2b79f5) >>> 0; let t = a;
      t = Math.imul(t ^ (t >>> 15), t | 1); t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
    const walls = this.params.walls, px = periodicX(walls), py = periodicY(walls);
    const wrap = (d: number, n: number, p: boolean) => (p ? d - n * Math.round(d / n) : d);
    const cx = rand() * nx, cy = rand() * ny;
    const nv = Math.max(1, Math.round(density * (Math.PI * (2 * sigma) ** 2) / (Math.PI * (2 * core) ** 2) * 4));
    const vort: [number, number, number][] = [];
    for (let n = 0; n < nv; n++) {
      // swirl centres spread over ~2σ around the patch centre
      const rr = 2 * sigma * Math.sqrt(rand()), th = 2 * Math.PI * rand();
      vort.push([cx + rr * Math.cos(th), cy + rr * Math.sin(th), (rand() < 0.5 ? -1 : 1) * amp * core * h * 1.65 * (0.5 + rand())]);
    }
    const cs = nx + 1, psi = new Float64Array(cs * (ny + 1));
    for (let j = 0; j <= ny; j++) for (let i = 0; i <= nx; i++) {
      const ex = wrap(i - cx, nx, px), ey = wrap(j - cy, ny, py);
      const env = Math.exp(-(ex * ex + ey * ey) / (2 * sigma * sigma));
      if (env < 1e-6) continue;
      let sum = 0;
      for (const [vx, vy, st] of vort) {
        const dx = wrap(i - vx, nx, px), dy = wrap(j - vy, ny, py);
        sum += st * Math.exp(-(dx * dx + dy * dy) / (2 * core * core));
      }
      psi[i + cs * j] = env * sum;
    }
    if (px) for (let j = 0; j <= ny; j++) psi[nx + cs * j] = psi[cs * j];
    if (py) for (let i = 0; i <= nx; i++) psi[i + cs * ny] = psi[i];
    this.u.fill(0); this.v.fill(0);
    const iLast = isPeriodic(walls) ? uLastI(g, walls) : nx - 1, jLast = isPeriodic(walls) ? vLastJ(g, walls) : ny - 1;
    const U0 = v0 * Math.cos(heading), V0 = v0 * Math.sin(heading);
    for (let j = 1; j <= ny; j++) for (let i = 1; i <= iLast; i++)
      this.u[uIdx(g, i, j)] = U0 + (psi[i + cs * j] - psi[i + cs * (j - 1)]) / h;
    for (let j = 1; j <= jLast; j++) for (let i = 1; i <= nx; i++)
      this.v[vIdx(g, i, j)] = V0 - (psi[i + cs * j] - psi[i - 1 + cs * j]) / h;
    setUBoundary(g, this.u, walls); setVBoundary(g, this.v, walls);
  }

  seedActiveTurbulence(seed = 1, amplitude = 0.05, vortices = 40): void {
    const g = this.grid;
    const { nx, ny, h } = g;
    let a = seed >>> 0;
    const rand = (): number => {
      a = (a + 0x6d2b79f5) >>> 0;
      let t = a;
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
    const r = Math.max(2, Math.min(nx, ny) / 12);
    // Stream function ψ at cell corners; u = ∂ψ/∂y, v = −∂ψ/∂x is discretely divergence free.
    const cs = nx + 1;
    const psi = new Float32Array(cs * (ny + 1));
    const walls = this.params.walls;
    if (isPeriodic(walls)) {
      // Periodic ψ (vortices wrap across seams, nearest-image distance), so the
      // seeded field is exactly divergence free on the torus / channel too.
      const px = periodicX(walls);
      const py = periodicY(walls);
      const reach = 3 * r;
      for (let n = 0; n < vortices; n++) {
        const cx = rand() * nx;
        const cy = rand() * ny;
        const s = (rand() < 0.5 ? -1 : 1) * amplitude * r * h * 1.65 * (0.5 + rand());
        for (let j = 0; j <= ny; j++) {
          let dy = j - cy;
          if (py) dy -= ny * Math.round(dy / ny);
          if (Math.abs(dy) > reach) continue;
          for (let i = 0; i <= nx; i++) {
            let dx = i - cx;
            if (px) dx -= nx * Math.round(dx / nx);
            if (Math.abs(dx) > reach) continue;
            psi[i + cs * j] += s * Math.exp(-(dx * dx + dy * dy) / (2 * r * r));
          }
        }
      }
      // make the seam corners identical (ψ(0) ≡ ψ(n))
      if (px) for (let j = 0; j <= ny; j++) psi[nx + cs * j] = psi[cs * j];
      if (py) for (let i = 0; i <= nx; i++) psi[i + cs * ny] = psi[i];
      const iLast = uLastI(g, walls);
      const jLast = vLastJ(g, walls);
      for (let j = 1; j <= ny; j++)
        for (let i = 1; i <= iLast; i++)
          this.u[uIdx(g, i, j)] += (psi[i + cs * j] - psi[i + cs * (j - 1)]) / h;
      for (let j = 1; j <= jLast; j++)
        for (let i = 1; i <= nx; i++)
          this.v[vIdx(g, i, j)] -= (psi[i + cs * j] - psi[i - 1 + cs * j]) / h;
    } else for (let n = 0; n < vortices; n++) {
      const cx = rand() * nx;
      const cy = rand() * ny;
      const s = (rand() < 0.5 ? -1 : 1) * amplitude * r * h * 1.65 * (0.5 + rand());
      const reach = Math.ceil(3 * r);
      for (let j = Math.max(0, Math.floor(cy) - reach); j <= Math.min(ny, Math.ceil(cy) + reach); j++)
        for (let i = Math.max(0, Math.floor(cx) - reach); i <= Math.min(nx, Math.ceil(cx) + reach); i++) {
          const dx = i - cx;
          const dy = j - cy;
          psi[i + cs * j] += s * Math.exp(-(dx * dx + dy * dy) / (2 * r * r));
        }
    }
    if (!isPeriodic(walls)) {
      for (let j = 1; j <= ny; j++)
        for (let i = 1; i < nx; i++)
          this.u[uIdx(g, i, j)] += (psi[i + cs * j] - psi[i + cs * (j - 1)]) / h;
      for (let j = 1; j < ny; j++)
        for (let i = 1; i <= nx; i++)
          this.v[vIdx(g, i, j)] -= (psi[i + cs * j] - psi[i - 1 + cs * j]) / h;
    }
    setUBoundary(g, this.u, this.params.walls);
    setVBoundary(g, this.v, this.params.walls);
    // Colourful dye patches (a 8 × 5 checkerboard of hues) so the emerging
    // flow pattern is visible at once.
    const palette: RGB[] = [
      [1.0, 0.25, 0.1], [0.1, 0.55, 1.0], [0.2, 1.0, 0.45], [1.0, 0.85, 0.1], [0.85, 0.2, 1.0],
    ];
    for (let j = 1; j <= ny; j++)
      for (let i = 1; i <= nx; i++) {
        const k = idx(g, i, j);
        const px = Math.floor(((i - 0.5) / nx) * 8);
        const py = Math.floor(((j - 0.5) / ny) * 5);
        const c = palette[(px + 2 * py) % palette.length];
        this.dye[0][k] += 0.8 * c[0];
        this.dye[1][k] += 0.8 * c[1];
        this.dye[2][k] += 0.8 * c[2];
      }
    for (const d of this.dye) setScalarBoundary(g, d, walls);
  }

  /** Refresh cell-centred velocity, divergence and vorticity fields. */
  updateDiagnostics(): void {
    const g = this.grid;
    const walls = this.params.walls;
    if (isPeriodic(walls)) {
      setUBoundary(g, this.u, walls);
      setVBoundary(g, this.v, walls);
    }
    // periodic: wrapped ghosts give correct vorticity in the seam cells
    cellVelocity(g, this.u, this.v, this.uc, this.vc, walls);
    computeDivergence(g, this.u, this.v, this.divergence);
    computeVorticity(g, this.uc, this.vc, this.vorticity);
  }

  /** Bilinearly sample a cell-centred field at grid position (X, Y). */
  sample(f: Float32Array, X: number, Y: number): number {
    return sampleCell(this.grid, f, X, Y, this.params.walls);
  }

  /**
   * Mean velocity (Σu, Σv over the distinct faces, divided by the face count;
   * on a periodic axis the seam face is counted once). This is the total
   * momentum per unit area, which is conserved on a fully periodic domain.
   */
  meanVelocity(): [number, number] {
    const g = this.grid;
    const walls = this.params.walls;
    const i0 = periodicX(walls) ? 1 : 0;
    const j0 = periodicY(walls) ? 1 : 0;
    let su = 0;
    let sv = 0;
    for (let j = 1; j <= g.ny; j++) for (let i = i0; i <= g.nx; i++) su += this.u[uIdx(g, i, j)];
    for (let j = j0; j <= g.ny; j++) for (let i = 1; i <= g.nx; i++) sv += this.v[vIdx(g, i, j)];
    return [su / ((g.nx + 1 - i0) * g.ny), sv / (g.nx * (g.ny + 1 - j0))];
  }

  /** Kinetic energy ½∫|u|² dA (from cell-centred velocities; call updateDiagnostics first). */
  kineticEnergy(): number {
    const { nx, ny, h } = this.grid;
    let e = 0;
    for (let j = 1; j <= ny; j++)
      for (let i = 1; i <= nx; i++) {
        const k = idx(this.grid, i, j);
        e += this.uc[k] * this.uc[k] + this.vc[k] * this.vc[k];
      }
    return 0.5 * e * h * h;
  }
}
