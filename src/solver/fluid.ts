/**
 * High-level 2D incompressible Navier–Stokes solver ("Stable Fluids",
 * J. Stam 1999) on a staggered MAC grid with solid walls.
 *
 *   ∂u/∂t + (u·∇)u = −∇p/ρ + ν∇²u + f      (momentum, ρ = 1)
 *   ∇·u = 0                                 (incompressibility)
 *   ∂c/∂t + (u·∇)c = κ∇²c                   (passive dye, three RGB channels)
 *
 * One time step (operator splitting):
 *   1. forces (vorticity confinement; user impulses are applied via splat())
 *   2. implicit viscous diffusion
 *   3. projection (Poisson solve for p, subtract ∇p)
 *   4. semi-Lagrangian self-advection (RK2 back-trace, bilinear interpolation)
 *   5. projection
 *   6. dye: implicit diffusion + semi-Lagrangian advection + optional fade
 *
 * This module has no DOM dependencies.
 */
import {
  createField, createGrid, createUField, createVField, idx, uIdx, vIdx,
  type Grid, type WallType,
} from './grid';
import {
  advectScalar, advectVelocity, cellVelocity, computeDivergence, computeVorticity,
  diffuseScalar, diffuseU, diffuseV, project, sampleCell, sampleU, sampleV,
  setScalarBoundary, setUBoundary, setVBoundary, vorticityConfinement,
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
  /** Tangential wall condition for velocity. */
  walls: WallType;
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
    this.params = { ...this.params, ...p };
    setUBoundary(this.grid, this.u, this.params.walls);
    setVBoundary(this.grid, this.v, this.params.walls);
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
    const g = this.grid;
    const r = Math.max(radius, 0.5);
    const inv2r2 = 1 / (2 * r * r);
    const reach = Math.ceil(3 * r);
    const a = Math.min(Math.max(strength, 0), 1);
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
    for (let j = 1; j <= g.ny; j++)
      for (let i = 0; i <= g.nx; i++)
        this.u[uIdx(g, i, j)] = sampleU(sg, src.u, i * sx, (j - 0.5) * sy);
    for (let j = 0; j <= g.ny; j++)
      for (let i = 1; i <= g.nx; i++)
        this.v[vIdx(g, i, j)] = sampleV(sg, src.v, (i - 0.5) * sx, j * sy);
    for (let c = 0; c < 3; c++)
      for (let j = 1; j <= g.ny; j++)
        for (let i = 1; i <= g.nx; i++)
          this.dye[c][idx(g, i, j)] = sampleCell(sg, src.dye[c], (i - 0.5) * sx, (j - 0.5) * sy);
    for (const d of this.dye) setScalarBoundary(g, d);
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
    const { viscosity, diffusion, dt, iterations, sor, vorticity, dyeDissipation, walls } = this.params;
    const { u, v, u0, v0, pressure, tmp } = this;

    // 1. forces
    vorticityConfinement(g, u, v, vorticity, dt, this.scratch);
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

    // 5. projection
    project(g, u, v, pressure, tmp, dt, iterations, walls, sor);

    // 6. dye
    const fade = dyeDissipation > 0 ? Math.exp(-dyeDissipation * dt) : 1;
    for (const d of this.dye) {
      if (diffusion > 0) {
        tmp.set(d);
        diffuseScalar(g, d, tmp, diffusion, dt, iterations);
      }
      tmp.set(d);
      advectScalar(g, d, tmp, u, v, dt);
      if (fade !== 1) for (let k = 0; k < d.length; k++) d[k] *= fade;
    }

    this.time += dt;
    this.steps++;
  }

  /** Refresh cell-centred velocity, divergence and vorticity fields. */
  updateDiagnostics(): void {
    const g = this.grid;
    cellVelocity(g, this.u, this.v, this.uc, this.vc);
    computeDivergence(g, this.u, this.v, this.divergence);
    computeVorticity(g, this.uc, this.vc, this.vorticity);
  }

  /** Bilinearly sample a cell-centred field at grid position (X, Y). */
  sample(f: Float32Array, X: number, Y: number): number {
    return sampleCell(this.grid, f, X, Y);
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
