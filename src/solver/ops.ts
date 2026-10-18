/**
 * Numerical kernels for a "Stable Fluids" (J. Stam, SIGGRAPH 1999) solver on
 * a staggered MAC grid. Every function is pure numerics on Float32Arrays laid
 * out as described in ./grid.ts — no DOM dependencies — so they can be unit
 * tested and later ported (e.g. to WebGL / WebGPU shaders).
 */
import { PeriodicPoisson } from './poisson';
import { isPeriodic, periodicX, periodicY, wrap, type Grid, type WallType } from './grid';

// ---------------------------------------------------------------------------
// Boundary conditions
// ---------------------------------------------------------------------------

/**
 * Ghost cells of a cell-centred scalar: zero normal derivative (Neumann, copy
 * into ghosts) at walls, wrapped copies of the opposite edge on periodic axes.
 */
export function setScalarBoundary(g: Grid, x: Float32Array, walls: WallType = 'no-slip'): void {
  if (isPeriodic(walls)) {
    setScalarBoundaryPeriodic(g, x, periodicX(walls), periodicY(walls));
    return;
  }
  const { nx, ny, stride: s } = g;
  for (let j = 1; j <= ny; j++) {
    const r = j * s;
    x[r] = x[r + 1];
    x[r + nx + 1] = x[r + nx];
  }
  const top = (ny + 1) * s;
  for (let i = 1; i <= nx; i++) {
    x[i] = x[i + s];
    x[top + i] = x[top - s + i];
  }
  x[0] = 0.5 * (x[1] + x[s]);
  x[nx + 1] = 0.5 * (x[nx] + x[s + nx + 1]);
  x[top] = 0.5 * (x[top + 1] + x[top - s]);
  x[top + nx + 1] = 0.5 * (x[top + nx] + x[top - s + nx + 1]);
}

function setScalarBoundaryPeriodic(g: Grid, x: Float32Array, px: boolean, py: boolean): void {
  const { nx, ny, stride: s } = g;
  for (let j = 1; j <= ny; j++) {
    const r = j * s;
    if (px) {
      x[r] = x[r + nx];
      x[r + nx + 1] = x[r + 1];
    } else {
      x[r] = x[r + 1];
      x[r + nx + 1] = x[r + nx];
    }
  }
  // full rows including the ghost columns, so the corners are consistent too
  const top = (ny + 1) * s;
  for (let i = 0; i <= nx + 1; i++) {
    if (py) {
      x[i] = x[ny * s + i];
      x[top + i] = x[s + i];
    } else {
      x[i] = x[i + s];
      x[top + i] = x[top - s + i];
    }
  }
}

/** Last u-face column holding an unknown: nx on an x-periodic domain (face nx ≡ face 0), else nx − 1. */
export const uLastI = (g: Grid, walls: WallType): number => (periodicX(walls) ? g.nx : g.nx - 1);
/** Last v-face row holding an unknown: ny on a y-periodic domain (face ny ≡ face 0), else ny − 1. */
export const vLastJ = (g: Grid, walls: WallType): number => (periodicY(walls) ? g.ny : g.ny - 1);

/**
 * u on solid walls: zero normal flow through the left/right walls; the
 * tangential condition at the bottom/top walls is imposed via ghost rows
 * (mirror with sign −1 for no-slip, +1 for free-slip).
 */
export function setUBoundary(g: Grid, u: Float32Array, walls: WallType): void {
  if (isPeriodic(walls)) {
    setUBoundaryPeriodic(g, u, periodicX(walls), periodicY(walls), walls === 'free-slip' ? 1 : -1);
    return;
  }
  const { nx, ny, uStride: s } = g;
  const sign = walls === 'no-slip' ? -1 : 1;
  for (let j = 1; j <= ny; j++) {
    u[j * s] = 0;
    u[j * s + nx] = 0;
  }
  const top = (ny + 1) * s;
  for (let i = 0; i <= nx; i++) {
    u[i] = sign * u[i + s];
    u[top + i] = sign * u[top - s + i];
  }
}

/**
 * u on a domain with at least one periodic axis: on an x-periodic domain face
 * i = 0 is a copy of face i = nx (the same physical face); on a y-periodic
 * domain the ghost rows wrap around, otherwise they mirror (no-slip walls).
 */
function setUBoundaryPeriodic(g: Grid, u: Float32Array, px: boolean, py: boolean, sign: number): void {
  const { nx, ny, uStride: s } = g;
  for (let j = 1; j <= ny; j++) {
    if (px) u[j * s] = u[j * s + nx];
    else {
      u[j * s] = 0;
      u[j * s + nx] = 0;
    }
  }
  const top = (ny + 1) * s;
  for (let i = 0; i <= nx; i++) {
    if (py) {
      u[i] = u[ny * s + i];
      u[top + i] = u[s + i];
    } else {
      u[i] = sign * u[i + s];
      u[top + i] = sign * u[top - s + i];
    }
  }
}

/** v on solid walls (see {@link setUBoundary}). */
export function setVBoundary(g: Grid, v: Float32Array, walls: WallType): void {
  if (isPeriodic(walls)) {
    setVBoundaryPeriodic(g, v, periodicX(walls), periodicY(walls), walls === 'free-slip' ? 1 : -1);
    return;
  }
  const { nx, ny, vStride: s } = g;
  const sign = walls === 'no-slip' ? -1 : 1;
  const top = ny * s;
  for (let i = 1; i <= nx; i++) {
    v[i] = 0;
    v[top + i] = 0;
  }
  for (let j = 0; j <= ny; j++) {
    const r = j * s;
    v[r] = sign * v[r + 1];
    v[r + nx + 1] = sign * v[r + nx];
  }
}

/** v on a domain with at least one periodic axis (see {@link setUBoundaryPeriodic}). */
function setVBoundaryPeriodic(g: Grid, v: Float32Array, px: boolean, py: boolean, sign: number): void {
  const { nx, ny, vStride: s } = g;
  const top = ny * s;
  for (let i = 1; i <= nx; i++) {
    if (py) v[i] = v[top + i];
    else {
      v[i] = 0;
      v[top + i] = 0;
    }
  }
  for (let j = 0; j <= ny; j++) {
    const r = j * s;
    if (px) {
      v[r] = v[r + nx];
      v[r + nx + 1] = v[r + 1];
    } else {
      v[r] = sign * v[r + 1];
      v[r + nx + 1] = sign * v[r + nx];
    }
  }
}

// ---------------------------------------------------------------------------
// Interpolation
// ---------------------------------------------------------------------------

/** Bilinear interpolation in index space of a (w × hgt) array. */
function bilerp(
  f: Float32Array,
  stride: number,
  w: number,
  hgt: number,
  fi: number,
  fj: number,
): number {
  let i0 = Math.floor(fi);
  let j0 = Math.floor(fj);
  if (i0 < 0) i0 = 0;
  else if (i0 > w - 2) i0 = w - 2;
  if (j0 < 0) j0 = 0;
  else if (j0 > hgt - 2) j0 = hgt - 2;
  const s1 = fi - i0;
  const t1 = fj - j0;
  const k = i0 + stride * j0;
  return (
    (1 - t1) * ((1 - s1) * f[k] + s1 * f[k + 1]) +
    t1 * ((1 - s1) * f[k + stride] + s1 * f[k + stride + 1])
  );
}

/*
 * On a periodic axis the position is first wrapped into [0, n); the ghost
 * layers (or the duplicated face 0 ≡ n) then hold the wrapped neighbours, so
 * the bilinear stencil reads the correct values across the seam. The field's
 * boundary must be up to date. Without `walls` (or with solid walls) the
 * position is clamped to the ghost layers as before.
 */

/** Sample u at grid position (X, Y). */
export function sampleU(g: Grid, u: Float32Array, X: number, Y: number, walls?: WallType): number {
  if (walls !== undefined && isPeriodic(walls)) {
    if (periodicX(walls)) X = wrap(X, g.nx);
    if (periodicY(walls)) Y = wrap(Y, g.ny);
  }
  return bilerp(u, g.uStride, g.nx + 1, g.ny + 2, X, Y + 0.5);
}

/** Sample v at grid position (X, Y). */
export function sampleV(g: Grid, v: Float32Array, X: number, Y: number, walls?: WallType): number {
  if (walls !== undefined && isPeriodic(walls)) {
    if (periodicX(walls)) X = wrap(X, g.nx);
    if (periodicY(walls)) Y = wrap(Y, g.ny);
  }
  return bilerp(v, g.vStride, g.nx + 2, g.ny + 1, X + 0.5, Y);
}

/** Sample a cell-centred field at grid position (X, Y). */
export function sampleCell(g: Grid, f: Float32Array, X: number, Y: number, walls?: WallType): number {
  if (walls !== undefined && isPeriodic(walls)) {
    if (periodicX(walls)) X = wrap(X, g.nx);
    if (periodicY(walls)) Y = wrap(Y, g.ny);
  }
  return bilerp(f, g.stride, g.nx + 2, g.ny + 2, X + 0.5, Y + 0.5);
}

// ---------------------------------------------------------------------------
// Diffusion (implicit / backward Euler, Gauss-Seidel)
// ---------------------------------------------------------------------------

/**
 * Number of Gauss-Seidel sweeps needed for the implicit diffusion system with
 * coefficient a = k·dt/h². The iteration matrix of (1+4a)x − aΣx = b has
 * spectral radius below ρ = 4a/(1+4a), so ⌈ln(tol)/ln ρ⌉ sweeps reduce the
 * error by `tol`. For the small a typical of low viscosity this is 1–3 sweeps,
 * which saves most of the cost; the user's iteration count is an upper bound.
 */
export function diffusionIterations(a: number, maxIterations: number, tol = 1e-4): number {
  if (a <= 0) return 0;
  const rho = (4 * a) / (1 + 4 * a);
  const needed = Math.ceil(Math.log(tol) / Math.log(rho));
  return Math.max(1, Math.min(maxIterations, needed));
}

/**
 * Solve (I − a∇²_h) x = x0 for a cell-centred scalar with Neumann walls, where
 * a = k·dt / h². Unconditionally stable. Conserves the total amount exactly
 * (up to solver tolerance) because no flux crosses the walls.
 */
export function diffuseScalar(
  g: Grid, x: Float32Array, x0: Float32Array, k: number, dt: number, iterations: number,
  walls: WallType = 'no-slip',
): void {
  x.set(x0);
  if (k <= 0) return;
  const { nx, ny, stride: s } = g;
  const a = (k * dt) / (g.h * g.h);
  const its = diffusionIterations(a, iterations);
  if (isPeriodic(walls)) {
    // Neumann at walls, wrap-around neighbours on periodic axes
    const px = periodicX(walls);
    const py = periodicY(walls);
    const wx = nx - 1;
    const wy = (ny - 1) * s;
    for (let it = 0; it < its; it++) {
      for (let j = 1; j <= ny; j++) {
        let c = j * s + 1;
        for (let i = 1; i <= nx; i++, c++) {
          let sum = 0;
          let n = 0;
          if (i > 1) { sum += x[c - 1]; n++; } else if (px) { sum += x[c + wx]; n++; }
          if (i < nx) { sum += x[c + 1]; n++; } else if (px) { sum += x[c - wx]; n++; }
          if (j > 1) { sum += x[c - s]; n++; } else if (py) { sum += x[c + wy]; n++; }
          if (j < ny) { sum += x[c + s]; n++; } else if (py) { sum += x[c - wy]; n++; }
          x[c] = (x0[c] + a * sum) / (1 + a * n);
        }
      }
    }
    setScalarBoundary(g, x, walls);
    return;
  }
  for (let it = 0; it < its; it++) {
    for (let j = 1; j <= ny; j++) {
      let c = j * s + 1;
      for (let i = 1; i <= nx; i++, c++) {
        // Neumann: only count neighbours that are inside the domain
        let sum = 0;
        let n = 0;
        if (i > 1) { sum += x[c - 1]; n++; }
        if (i < nx) { sum += x[c + 1]; n++; }
        if (j > 1) { sum += x[c - s]; n++; }
        if (j < ny) { sum += x[c + s]; n++; }
        x[c] = (x0[c] + a * sum) / (1 + a * n);
      }
    }
  }
  setScalarBoundary(g, x);
}

/** Implicit viscous diffusion of u (faces i = 1..nx−1, j = 1..ny). */
export function diffuseU(
  g: Grid, u: Float32Array, u0: Float32Array, nu: number, dt: number,
  iterations: number, walls: WallType,
): void {
  u.set(u0);
  if (nu > 0 && isPeriodic(walls)) diffuseUPeriodic(g, u, u0, nu, dt, iterations, walls);
  else if (nu > 0) {
    const { nx, ny, uStride: s } = g;
    const a = (nu * dt) / (g.h * g.h);
    const inv = 1 / (1 + 4 * a);
    const its = diffusionIterations(a, iterations);
    for (let it = 0; it < its; it++) {
      for (let j = 1; j <= ny; j++) {
        let k = j * s + 1;
        for (let i = 1; i < nx; i++, k++) {
          u[k] = (u0[k] + a * (u[k - 1] + u[k + 1] + u[k - s] + u[k + s])) * inv;
        }
      }
      setUBoundary(g, u, walls);
    }
  }
  setUBoundary(g, u, walls);
}

/** Implicit viscous diffusion of v (faces i = 1..nx, j = 1..ny−1). */
export function diffuseV(
  g: Grid, v: Float32Array, v0: Float32Array, nu: number, dt: number,
  iterations: number, walls: WallType,
): void {
  v.set(v0);
  if (nu > 0 && isPeriodic(walls)) diffuseVPeriodic(g, v, v0, nu, dt, iterations, walls);
  else if (nu > 0) {
    const { nx, ny, vStride: s } = g;
    const a = (nu * dt) / (g.h * g.h);
    const inv = 1 / (1 + 4 * a);
    const its = diffusionIterations(a, iterations);
    for (let it = 0; it < its; it++) {
      for (let j = 1; j < ny; j++) {
        let k = j * s + 1;
        for (let i = 1; i <= nx; i++, k++) {
          v[k] = (v0[k] + a * (v[k - 1] + v[k + 1] + v[k - s] + v[k + s])) * inv;
        }
      }
      setVBoundary(g, v, walls);
    }
  }
  setVBoundary(g, v, walls);
}

/**
 * Gauss-Seidel sweeps for u with at least one periodic axis. Neighbours across
 * a periodic seam are read directly from the opposite edge (not from possibly
 * stale ghost copies), so it is a true Gauss-Seidel iteration on the torus.
 * Unknown faces: i = 1..nx when x-periodic (face nx ≡ face 0), j = 1..ny.
 */
function diffuseUPeriodic(
  g: Grid, u: Float32Array, u0: Float32Array, nu: number, dt: number,
  iterations: number, walls: WallType,
): void {
  const { nx, ny, uStride: s } = g;
  const px = periodicX(walls);
  const py = periodicY(walls);
  const iLast = uLastI(g, walls);
  const a = (nu * dt) / (g.h * g.h);
  const inv = 1 / (1 + 4 * a);
  const its = diffusionIterations(a, iterations);
  const wy = (ny - 1) * s;
  for (let it = 0; it < its; it++) {
    for (let j = 1; j <= ny; j++) {
      const r = j * s;
      for (let i = 1; i <= iLast; i++) {
        const k = r + i;
        const left = i > 1 ? u[k - 1] : px ? u[r + nx] : u[k - 1];
        const right = i < nx ? u[k + 1] : u[r + 1];
        const down = py && j === 1 ? u[k + wy] : u[k - s];
        const up = py && j === ny ? u[k - wy] : u[k + s];
        u[k] = (u0[k] + a * (left + right + down + up)) * inv;
      }
    }
    setUBoundary(g, u, walls);
  }
}

/** Gauss-Seidel sweeps for v with at least one periodic axis (see {@link diffuseUPeriodic}). */
function diffuseVPeriodic(
  g: Grid, v: Float32Array, v0: Float32Array, nu: number, dt: number,
  iterations: number, walls: WallType,
): void {
  const { nx, ny, vStride: s } = g;
  const px = periodicX(walls);
  const py = periodicY(walls);
  const jLast = vLastJ(g, walls);
  const a = (nu * dt) / (g.h * g.h);
  const inv = 1 / (1 + 4 * a);
  const its = diffusionIterations(a, iterations);
  const wx = nx - 1;
  for (let it = 0; it < its; it++) {
    for (let j = 1; j <= jLast; j++) {
      const r = j * s;
      for (let i = 1; i <= nx; i++) {
        const k = r + i;
        const left = px && i === 1 ? v[k + wx] : v[k - 1];
        const right = px && i === nx ? v[k - wx] : v[k + 1];
        const down = j > 1 ? v[k - s] : py ? v[ny * s + i] : v[k - s];
        const up = j < ny ? v[k + s] : v[s + i];
        v[k] = (v0[k] + a * (left + right + down + up)) * inv;
      }
    }
    setVBoundary(g, v, walls);
  }
}

// ---------------------------------------------------------------------------
// Delayed local viscosity: neighbour / self split of the 5-point Laplacian
// ---------------------------------------------------------------------------

/*
 * ν∇²_h u = (ν/h²)[Σ_nb u_nb − 4 u] is split into its neighbour sum and its
 * diagonal so the two can be read at different times. The neighbour sum uses
 * exactly the stencil of the implicit diffusion above (diffuseU / diffuseV and
 * their periodic variants): wall faces (u = 0 normal faces), the ghost rows /
 * columns of the tangential condition (−u for no-slip, +u for free-slip, i.e.
 * the wall "reflection" of the neighbour field) and wrap-around neighbours on
 * periodic axes. `nb` must have consistent ghosts (fields stored after
 * set*Boundary, or time interpolations of such fields). Only unknown faces of
 * `out` are written.
 */

/** Σ_nb of u-faces (unknowns i = 1..uLastI, j = 1..ny) into out. */
export function neighbourSumU(g: Grid, nb: ArrayLike<number>, out: Float64Array, walls: WallType): void {
  const { nx, ny, uStride: s } = g;
  const px = periodicX(walls);
  const py = periodicY(walls);
  const iLast = uLastI(g, walls);
  const wy = (ny - 1) * s;
  for (let j = 1; j <= ny; j++) {
    const r = j * s;
    for (let i = 1; i <= iLast; i++) {
      const k = r + i;
      const left = i > 1 ? nb[k - 1] : px ? nb[r + nx] : nb[k - 1];
      const right = i < nx ? nb[k + 1] : nb[r + 1];
      const down = py && j === 1 ? nb[k + wy] : nb[k - s];
      const up = py && j === ny ? nb[k - wy] : nb[k + s];
      out[k] = left + right + down + up;
    }
  }
}

/** Σ_nb of v-faces (unknowns i = 1..nx, j = 1..vLastJ) into out. */
export function neighbourSumV(g: Grid, nb: ArrayLike<number>, out: Float64Array, walls: WallType): void {
  const { nx, ny, vStride: s } = g;
  const px = periodicX(walls);
  const py = periodicY(walls);
  const jLast = vLastJ(g, walls);
  const wx = nx - 1;
  for (let j = 1; j <= jLast; j++) {
    const r = j * s;
    for (let i = 1; i <= nx; i++) {
      const k = r + i;
      const left = px && i === 1 ? nb[k + wx] : nb[k - 1];
      const right = px && i === nx ? nb[k - wx] : nb[k + 1];
      const down = j > 1 ? nb[k - s] : py ? nb[ny * s + i] : nb[k - s];
      const up = j < ny ? nb[k + s] : nb[s + i];
      out[k] = left + right + down + up;
    }
  }
}

// ---------------------------------------------------------------------------
// Semi-Lagrangian advection
// ---------------------------------------------------------------------------

/**
 * Trace the point (X, Y) backwards through the velocity field (u, v) for a
 * time step dt using a 2nd-order Runge-Kutta (midpoint) integrator, clamped
 * to the domain along wall axes and wrapped around along periodic axes
 * (px / py). Returns the departure point in `out`.
 */
function backtrace(
  g: Grid, u: Float32Array, v: Float32Array, X: number, Y: number,
  vx: number, vy: number, dt0: number, out: Float64Array, px = false, py = false,
): void {
  const { nx, ny } = g;
  let mx = X - 0.5 * dt0 * vx;
  let my = Y - 0.5 * dt0 * vy;
  mx = px ? wrap(mx, nx) : mx < 0 ? 0 : mx > nx ? nx : mx;
  my = py ? wrap(my, ny) : my < 0 ? 0 : my > ny ? ny : my;
  let x = X - dt0 * sampleU(g, u, mx, my);
  let y = Y - dt0 * sampleV(g, v, mx, my);
  out[0] = px ? wrap(x, nx) : x < 0 ? 0 : x > nx ? nx : x;
  out[1] = py ? wrap(y, ny) : y < 0 ? 0 : y > ny ? ny : y;
}

const pos = new Float64Array(2);

/**
 * Self-advect the velocity field: (u0, v0) is the field at the start of the
 * step (read only), the result is written into (u, v).
 */
export function advectVelocity(
  g: Grid, u: Float32Array, v: Float32Array, u0: Float32Array, v0: Float32Array,
  dt: number, walls: WallType,
): void {
  const { nx, ny, uStride: us, vStride: vs } = g;
  const dt0 = dt / g.h; // domain units / s -> cells / step
  const px = periodicX(walls);
  const py = periodicY(walls);
  const iLast = uLastI(g, walls);
  const jLast = vLastJ(g, walls);
  if (px || py) {
    // the stencils below read ghost / duplicated seam faces
    setUBoundary(g, u0, walls);
    setVBoundary(g, v0, walls);
  }
  for (let j = 1; j <= ny; j++) {
    for (let i = 1; i <= iLast; i++) {
      const k = i + us * j;
      const X = i;
      const Y = j - 0.5;
      // v at this u-face = average of the four surrounding v-faces
      const vb = i + vs * (j - 1);
      const vy = 0.25 * (v0[vb] + v0[vb + 1] + v0[vb + vs] + v0[vb + vs + 1]);
      backtrace(g, u0, v0, X, Y, u0[k], vy, dt0, pos, px, py);
      u[k] = sampleU(g, u0, pos[0], pos[1]);
    }
  }
  for (let j = 1; j <= jLast; j++) {
    for (let i = 1; i <= nx; i++) {
      const k = i + vs * j;
      const X = i - 0.5;
      const Y = j;
      const ub = i - 1 + us * j;
      const vx = 0.25 * (u0[ub] + u0[ub + 1] + u0[ub + us] + u0[ub + us + 1]);
      backtrace(g, u0, v0, X, Y, vx, v0[k], dt0, pos, px, py);
      v[k] = sampleV(g, v0, pos[0], pos[1]);
    }
  }
  setUBoundary(g, u, walls);
  setVBoundary(g, v, walls);
}

/** Advect a cell-centred scalar d0 → d through the face velocity field (u, v). */
export function advectScalar(
  g: Grid, d: Float32Array, d0: Float32Array, u: Float32Array, v: Float32Array, dt: number,
  walls: WallType = 'no-slip',
): void {
  const { nx, ny, stride: s, uStride: us, vStride: vs } = g;
  const dt0 = dt / g.h;
  const px = periodicX(walls);
  const py = periodicY(walls);
  if (px || py) setScalarBoundary(g, d0, walls);
  for (let j = 1; j <= ny; j++) {
    for (let i = 1; i <= nx; i++) {
      const uk = i - 1 + us * j;
      const vk = i + vs * (j - 1);
      const vx = 0.5 * (u[uk] + u[uk + 1]);
      const vy = 0.5 * (v[vk] + v[vk + vs]);
      backtrace(g, u, v, i - 0.5, j - 0.5, vx, vy, dt0, pos, px, py);
      d[i + s * j] = sampleCell(g, d0, pos[0], pos[1]);
    }
  }
  setScalarBoundary(g, d, walls);
}

// ---------------------------------------------------------------------------
// Pressure projection
// ---------------------------------------------------------------------------

/** Discrete (compact, MAC) divergence ∇·u at cell centres, in 1/s. */
export function computeDivergence(g: Grid, u: Float32Array, v: Float32Array, out: Float32Array): void {
  const { nx, ny, stride: s, uStride: us, vStride: vs } = g;
  const invH = 1 / g.h;
  out.fill(0);
  for (let j = 1; j <= ny; j++) {
    for (let i = 1; i <= nx; i++) {
      const uk = i + us * j;
      const vk = i + vs * j;
      out[i + s * j] = (u[uk] - u[uk - 1] + v[vk] - v[vk - vs]) * invH;
    }
  }
}

/**
 * Helmholtz–Hodge projection onto divergence-free fields.
 *
 * Solves the pressure Poisson equation  ∇²p = (ρ/dt) ∇·u*  (ρ = 1) with
 * ∂p/∂n = 0 on the walls using Gauss-Seidel with successive over-relaxation
 * (omega = 1 is plain Gauss-Seidel), then applies  u = u* − (dt/ρ) ∇p.
 *
 * `p` is used as the initial guess (warm start) and receives the pressure;
 * `div` receives the divergence of u* (before projection).
 */
export function project(
  g: Grid, u: Float32Array, v: Float32Array, p: Float32Array, div: Float32Array,
  dt: number, iterations: number, walls: WallType, omega = 1.7,
): void {
  if (isPeriodic(walls)) {
    projectPeriodic(g, u, v, p, div, dt, iterations, walls, omega);
    return;
  }
  const { nx, ny, stride: s, uStride: us, vStride: vs, h } = g;
  computeDivergence(g, u, v, div);
  const rhsScale = (h * h) / dt;
  // Wall-adjacent cells only couple to neighbours inside the domain
  // (∂p/∂n = 0); interior cells take a branch-free fast path.
  const edge = (i: number, j: number): void => {
    const c = i + s * j;
    let sum = 0;
    let n = 0;
    if (i > 1) { sum += p[c - 1]; n++; }
    if (i < nx) { sum += p[c + 1]; n++; }
    if (j > 1) { sum += p[c - s]; n++; }
    if (j < ny) { sum += p[c + s]; n++; }
    p[c] += omega * ((sum - rhsScale * div[c]) / n - p[c]);
  };
  const w4 = 0.25 * omega;
  const keep = 1 - omega;
  for (let it = 0; it < iterations; it++) {
    for (let i = 1; i <= nx; i++) edge(i, 1);
    for (let j = 2; j < ny; j++) {
      edge(1, j);
      let c = j * s + 2;
      for (let i = 2; i < nx; i++, c++) {
        p[c] = keep * p[c] + w4 * (p[c - 1] + p[c + 1] + p[c - s] + p[c + s] - rhsScale * div[c]);
      }
      edge(nx, j);
    }
    for (let i = 1; i <= nx; i++) edge(i, ny);
  }
  // Pure-Neumann problem: pressure is defined up to a constant; pin its mean.
  let mean = 0;
  for (let j = 1; j <= ny; j++) for (let i = 1; i <= nx; i++) mean += p[i + s * j];
  mean /= nx * ny;
  for (let j = 1; j <= ny; j++) for (let i = 1; i <= nx; i++) p[i + s * j] -= mean;
  setScalarBoundary(g, p);

  const k = dt / h;
  for (let j = 1; j <= ny; j++) {
    for (let i = 1; i < nx; i++) {
      const c = i + s * j;
      u[i + us * j] -= k * (p[c + 1] - p[c]);
    }
  }
  for (let j = 1; j < ny; j++) {
    for (let i = 1; i <= nx; i++) {
      const c = i + s * j;
      v[i + vs * j] -= k * (p[c + s] - p[c]);
    }
  }
  setUBoundary(g, u, walls);
  setVBoundary(g, v, walls);
}

/**
 * Projection with at least one periodic axis ('periodic' or 'periodic-x').
 *
 * The pressure Poisson equation uses the MAC 5-point Laplacian with
 * wrap-around neighbours across periodic seams (every cell has 4 neighbours
 * along a periodic axis) and ∂p/∂n = 0 at the channel walls. The problem is
 * singular (p is defined up to a constant); it is solvable because the
 * discrete divergence telescopes to zero over a periodic/closed domain. The
 * mean of the right-hand side (float round-off) is removed so the system is
 * exactly consistent, and the mean of p is pinned to 0 afterwards.
 *
 * method 'direct' (default) solves it exactly with FFTs along the periodic
 * axes (see ./poisson.ts), so ∇·u is ~0 after every projection regardless of
 * the iteration count. 'sor' uses wrap-around SOR with `iterations` sweeps.
 *
 * The pressure gradient is then applied to every unknown face, including the
 * seam faces i = nx (≡ 0) and j = ny (≡ 0), whose neighbour cells are the
 * wrapped ghosts.
 */
export function projectPeriodic(
  g: Grid, u: Float32Array, v: Float32Array, p: Float32Array, div: Float32Array,
  dt: number, iterations: number, walls: WallType, omega = 1.7,
  method: 'direct' | 'sor' = 'direct',
): void {
  const { nx, ny, stride: s, uStride: us, vStride: vs, h } = g;
  const px = periodicX(walls);
  const py = periodicY(walls);
  setUBoundary(g, u, walls);
  setVBoundary(g, v, walls);
  computeDivergence(g, u, v, div);
  let divMean = 0;
  for (let j = 1; j <= ny; j++) for (let i = 1; i <= nx; i++) divMean += div[i + s * j];
  divMean /= nx * ny;
  const rhsScale = (h * h) / dt;
  if (method === 'direct') {
    poissonFor(g, py).solve((i, j) => rhsScale * (div[i + s * j] - divMean), p);
  } else sorPeriodic(g, p, div, divMean, rhsScale, iterations, px, py, omega);
  let mean = 0;
  for (let j = 1; j <= ny; j++) for (let i = 1; i <= nx; i++) mean += p[i + s * j];
  mean /= nx * ny;
  for (let j = 1; j <= ny; j++) for (let i = 1; i <= nx; i++) p[i + s * j] -= mean;
  setScalarBoundary(g, p, walls);

  const k = dt / h;
  const iLast = uLastI(g, walls);
  const jLast = vLastJ(g, walls);
  for (let j = 1; j <= ny; j++) {
    for (let i = 1; i <= iLast; i++) {
      const c = i + s * j;
      u[i + us * j] -= k * (p[c + 1] - p[c]);
    }
  }
  for (let j = 1; j <= jLast; j++) {
    for (let i = 1; i <= nx; i++) {
      const c = i + s * j;
      v[i + vs * j] -= k * (p[c + s] - p[c]);
    }
  }
  setUBoundary(g, u, walls);
  setVBoundary(g, v, walls);
}

/** Cached direct Poisson solvers per grid (and y-periodicity). */
const poissonCache = new WeakMap<Grid, PeriodicPoisson[]>();
function poissonFor(g: Grid, py: boolean): PeriodicPoisson {
  let c = poissonCache.get(g);
  if (!c) poissonCache.set(g, (c = []));
  const slot = py ? 1 : 0;
  return (c[slot] ??= new PeriodicPoisson(g, py));
}

/**
 * Lexicographic SOR for the periodic Poisson problem (reference / fallback):
 * the 5-point stencil wraps across periodic seams, walls keep ∂p/∂n = 0.
 * With few iterations its residual concentrates in the cells next to the
 * seam (they always read one stale neighbour), which is why the direct solver
 * is the default.
 */
function sorPeriodic(
  g: Grid, p: Float32Array, div: Float32Array, divMean: number, rhsScale: number,
  iterations: number, px: boolean, py: boolean, omega: number,
): void {
  const { nx, ny, stride: s } = g;
  const wx = nx - 1;
  const wy = (ny - 1) * s;
  const edge = (i: number, j: number): void => {
    const c = i + s * j;
    let sum = 0;
    let n = 0;
    if (i > 1) { sum += p[c - 1]; n++; } else if (px) { sum += p[c + wx]; n++; }
    if (i < nx) { sum += p[c + 1]; n++; } else if (px) { sum += p[c - wx]; n++; }
    if (j > 1) { sum += p[c - s]; n++; } else if (py) { sum += p[c + wy]; n++; }
    if (j < ny) { sum += p[c + s]; n++; } else if (py) { sum += p[c - wy]; n++; }
    p[c] += omega * ((sum - rhsScale * (div[c] - divMean)) / n - p[c]);
  };
  const w4 = 0.25 * omega;
  const keep = 1 - omega;
  for (let it = 0; it < iterations; it++) {
    for (let i = 1; i <= nx; i++) edge(i, 1);
    for (let j = 2; j < ny; j++) {
      edge(1, j);
      let c = j * s + 2;
      for (let i = 2; i < nx; i++, c++) {
        p[c] = keep * p[c] + w4 * (p[c - 1] + p[c + 1] + p[c - s] + p[c + s] - rhsScale * (div[c] - divMean));
      }
      edge(nx, j);
    }
    for (let i = 1; i <= nx; i++) edge(i, ny);
  }
}

// ---------------------------------------------------------------------------
// Diagnostics & vorticity confinement
// ---------------------------------------------------------------------------

/** Interpolate face velocities to cell centres (uc, vc), with ghost cells copied. */
export function cellVelocity(
  g: Grid, u: Float32Array, v: Float32Array, uc: Float32Array, vc: Float32Array,
  walls: WallType = 'no-slip',
): void {
  const { nx, ny, stride: s, uStride: us, vStride: vs } = g;
  for (let j = 1; j <= ny; j++) {
    for (let i = 1; i <= nx; i++) {
      const uk = i - 1 + us * j;
      const vk = i + vs * (j - 1);
      uc[i + s * j] = 0.5 * (u[uk] + u[uk + 1]);
      vc[i + s * j] = 0.5 * (v[vk] + v[vk + vs]);
    }
  }
  setScalarBoundary(g, uc, walls);
  setScalarBoundary(g, vc, walls);
}

/** Scalar vorticity ω = ∂v/∂x − ∂u/∂y at cell centres (from cell velocities). */
export function computeVorticity(
  g: Grid, uc: Float32Array, vc: Float32Array, out: Float32Array,
): void {
  const { nx, ny, stride: s } = g;
  const inv2h = 0.5 / g.h;
  out.fill(0);
  for (let j = 1; j <= ny; j++) {
    let k = j * s + 1;
    for (let i = 1; i <= nx; i++, k++) {
      out[k] = (vc[k + 1] - vc[k - 1] - (uc[k + s] - uc[k - s])) * inv2h;
    }
  }
}

/** Stability limit for the explicit vorticity-confinement update (ε·Δt). */
export const MAX_CONFINEMENT_EPS_DT = 0.2;

/**
 * Vorticity confinement (Fedkiw, Stam & Jensen 2001). Re-injects small-scale
 * swirl lost to numerical dissipation:
 *   N = ∇|ω| / |∇|ω||,   f = ε h (N × ω ẑ)
 * The force is integrated explicitly, which is unstable for large ε·Δt, so the
 * product is limited to {@link MAX_CONFINEMENT_EPS_DT}.
 * Scratch arrays (cell-sized): uc, vc, curl, fx, fy.
 */
export function vorticityConfinement(
  g: Grid, u: Float32Array, v: Float32Array, epsilon: number, dt: number,
  scratch: { uc: Float32Array; vc: Float32Array; curl: Float32Array; fx: Float32Array; fy: Float32Array },
  walls: WallType = 'no-slip',
): void {
  if (epsilon <= 0) return;
  const { nx, ny, stride: s, uStride: us, vStride: vs, h } = g;
  const { uc, vc, curl, fx, fy } = scratch;
  const px = periodicX(walls);
  const py = periodicY(walls);
  if (px || py) {
    setUBoundary(g, u, walls);
    setVBoundary(g, v, walls);
  }
  cellVelocity(g, u, v, uc, vc, walls);
  computeVorticity(g, uc, vc, curl);
  // periodic: wrap |ω| so its gradient is defined in the seam cells as well
  if (px || py) setScalarBoundary(g, curl, walls);
  fx.fill(0);
  fy.fill(0);
  const inv2h = 0.5 / h;
  const effDt = Math.min(dt, MAX_CONFINEMENT_EPS_DT / epsilon);
  const scale = epsilon * h;
  const i0 = px ? 1 : 2;
  const i1 = px ? nx : nx - 1;
  const j0 = py ? 1 : 2;
  const j1 = py ? ny : ny - 1;
  for (let j = j0; j <= j1; j++) {
    let k = j * s + i0;
    for (let i = i0; i <= i1; i++, k++) {
      const dx = (Math.abs(curl[k + 1]) - Math.abs(curl[k - 1])) * inv2h;
      const dy = (Math.abs(curl[k + s]) - Math.abs(curl[k - s])) * inv2h;
      const len = Math.hypot(dx, dy) + 1e-10;
      const w = curl[k];
      fx[k] = scale * (dy / len) * w;
      fy[k] = -scale * (dx / len) * w;
    }
  }
  if (px || py) {
    setScalarBoundary(g, fx, walls);
    setScalarBoundary(g, fy, walls);
  }
  const iLast = uLastI(g, walls);
  const jLast = vLastJ(g, walls);
  for (let j = 1; j <= ny; j++) {
    for (let i = 1; i <= iLast; i++) {
      const c = i + s * j;
      u[i + us * j] += effDt * 0.5 * (fx[c] + fx[c + 1]);
    }
  }
  for (let j = 1; j <= jLast; j++) {
    for (let i = 1; i <= nx; i++) {
      const c = i + s * j;
      v[i + vs * j] += effDt * 0.5 * (fy[c] + fy[c + s]);
    }
  }
}

// ---------------------------------------------------------------------------
// Self-propulsion (active fluid)
// ---------------------------------------------------------------------------

/** Below this local speed (domain units / s) the flow direction v/|v| is undefined and the term is skipped. */
export const SELF_PROPULSION_EPS = 1e-6;

/**
 * Self-propulsion / active-fluid term (Toner–Tu / Vicsek-like):
 *
 *   ∂v/∂t = … − (1/τ) (|v| − v₀) v/|v|
 *
 * It relaxes the local speed |v| toward the target speed v₀ over the
 * relaxation time τ while keeping the local flow direction. Taken on its own
 * the speed obeys d|v|/dt = −(|v| − v₀)/τ, whose exact solution over one step
 *
 *   |v|ₙₑw = v₀ + (|v| − v₀) · exp(−Δt/τ)
 *
 * is applied along the unchanged direction v/|v|. This exact update is
 * unconditionally stable for any Δt/τ (it never overshoots v₀) and always
 * yields |v|ₙₑw ≥ 0 for v₀ ≥ 0.
 *
 * On the staggered grid |v| is evaluated at each face: on a u-face the v
 * component is the average of the four surrounding v-faces, and vice versa
 * (the same interpolation the advection step uses). Both components are read
 * from the start-of-step copies (u0, v0), so the update is order independent.
 * Where |v| < eps the direction is undefined and the face is left unchanged,
 * so a fluid at rest stays at rest (seed it with a small perturbation).
 *
 * (u0, v0) receive a copy of the input field; (u, v) are updated in place.
 */
export function selfPropulsion(
  g: Grid, u: Float32Array, v: Float32Array, u0: Float32Array, v0: Float32Array,
  tau: number, targetSpeed: number, dt: number, walls: WallType, eps = SELF_PROPULSION_EPS,
): void {
  if (!(tau > 0) || !(dt > 0)) return;
  const { nx, ny, uStride: us, vStride: vs } = g;
  const decay = Math.exp(-dt / tau);
  const vT = Math.max(0, targetSpeed);
  const gain = vT * (1 - decay); // |v|new = gain + decay·|v|
  if (isPeriodic(walls)) {
    // the face averages below read the seam copies / wrapped ghost layers
    setUBoundary(g, u, walls);
    setVBoundary(g, v, walls);
  }
  u0.set(u);
  v0.set(v);
  const iLast = uLastI(g, walls);
  const jLast = vLastJ(g, walls);
  for (let j = 1; j <= ny; j++) {
    for (let i = 1; i <= iLast; i++) {
      const k = i + us * j;
      const vb = i + vs * (j - 1);
      const uf = u0[k];
      const vf = 0.25 * (v0[vb] + v0[vb + 1] + v0[vb + vs] + v0[vb + vs + 1]);
      const sp = Math.sqrt(uf * uf + vf * vf);
      if (sp < eps) continue;
      u[k] = uf * (decay + gain / sp);
    }
  }
  for (let j = 1; j <= jLast; j++) {
    for (let i = 1; i <= nx; i++) {
      const k = i + vs * j;
      const ub = i - 1 + us * j;
      const vf = v0[k];
      const uf = 0.25 * (u0[ub] + u0[ub + 1] + u0[ub + us] + u0[ub + us + 1]);
      const sp = Math.sqrt(uf * uf + vf * vf);
      if (sp < eps) continue;
      v[k] = vf * (decay + gain / sp);
    }
  }
  setUBoundary(g, u, walls);
  setVBoundary(g, v, walls);
}

// ---------------------------------------------------------------------------
// Reductions
// ---------------------------------------------------------------------------

/** Root-mean-square of the interior values of a cell-centred field. */
export function rmsInterior(g: Grid, f: Float32Array): number {
  const { nx, ny, stride: s } = g;
  let sum = 0;
  for (let j = 1; j <= ny; j++) {
    let k = j * s + 1;
    for (let i = 1; i <= nx; i++, k++) sum += f[k] * f[k];
  }
  return Math.sqrt(sum / (nx * ny));
}

/** Sum of the interior values of a cell-centred field (e.g. total dye). */
export function sumInterior(g: Grid, f: Float32Array): number {
  const { nx, ny, stride: s } = g;
  let sum = 0;
  for (let j = 1; j <= ny; j++) {
    let k = j * s + 1;
    for (let i = 1; i <= nx; i++, k++) sum += f[k];
  }
  return sum;
}

/** Maximum absolute value over the interior of a cell-centred field. */
export function maxAbsInterior(g: Grid, f: Float32Array): number {
  const { nx, ny, stride: s } = g;
  let m = 0;
  for (let j = 1; j <= ny; j++) {
    let k = j * s + 1;
    for (let i = 1; i <= nx; i++, k++) {
      const a = Math.abs(f[k]);
      if (a > m) m = a;
    }
  }
  return m;
}
