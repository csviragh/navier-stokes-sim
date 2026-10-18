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
 *   2. implicit viscous diffusion
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
  vorticityConfinement,
} from './ops';

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
}

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
};

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

    // 2. viscosity
    if (viscosity > 0) {
      u0.set(u);
      v0.set(v);
      diffuseU(g, u, u0, viscosity, dt, iterations, walls);
      diffuseV(g, v, v0, viscosity, dt, iterations, walls);
    }

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
