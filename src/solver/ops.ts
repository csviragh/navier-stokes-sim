/**
 * Numerical kernels for a "Stable Fluids" (J. Stam, SIGGRAPH 1999) solver on
 * a staggered MAC grid. Every function is pure numerics on Float32Arrays laid
 * out as described in ./grid.ts — no DOM dependencies — so they can be unit
 * tested and later ported (e.g. to WebGL / WebGPU shaders).
 */
import type { Grid, WallType } from './grid';

// ---------------------------------------------------------------------------
// Boundary conditions
// ---------------------------------------------------------------------------

/** Zero normal derivative (Neumann) for a cell-centred scalar: copy into ghosts. */
export function setScalarBoundary(g: Grid, x: Float32Array): void {
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

/**
 * u on solid walls: zero normal flow through the left/right walls; the
 * tangential condition at the bottom/top walls is imposed via ghost rows
 * (mirror with sign −1 for no-slip, +1 for free-slip).
 */
export function setUBoundary(g: Grid, u: Float32Array, walls: WallType): void {
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

/** v on solid walls (see {@link setUBoundary}). */
export function setVBoundary(g: Grid, v: Float32Array, walls: WallType): void {
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

/** Sample u at grid position (X, Y). */
export function sampleU(g: Grid, u: Float32Array, X: number, Y: number): number {
  return bilerp(u, g.uStride, g.nx + 1, g.ny + 2, X, Y + 0.5);
}

/** Sample v at grid position (X, Y). */
export function sampleV(g: Grid, v: Float32Array, X: number, Y: number): number {
  return bilerp(v, g.vStride, g.nx + 2, g.ny + 1, X + 0.5, Y);
}

/** Sample a cell-centred field at grid position (X, Y). */
export function sampleCell(g: Grid, f: Float32Array, X: number, Y: number): number {
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
): void {
  x.set(x0);
  if (k <= 0) return;
  const { nx, ny, stride: s } = g;
  const a = (k * dt) / (g.h * g.h);
  const its = diffusionIterations(a, iterations);
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
  if (nu > 0) {
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
  if (nu > 0) {
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

// ---------------------------------------------------------------------------
// Semi-Lagrangian advection
// ---------------------------------------------------------------------------

/**
 * Trace the point (X, Y) backwards through the velocity field (u, v) for a
 * time step dt using a 2nd-order Runge-Kutta (midpoint) integrator, clamped
 * to the domain. Returns the departure point in `out`.
 */
function backtrace(
  g: Grid, u: Float32Array, v: Float32Array, X: number, Y: number,
  vx: number, vy: number, dt0: number, out: Float64Array,
): void {
  const { nx, ny } = g;
  let mx = X - 0.5 * dt0 * vx;
  let my = Y - 0.5 * dt0 * vy;
  mx = mx < 0 ? 0 : mx > nx ? nx : mx;
  my = my < 0 ? 0 : my > ny ? ny : my;
  let x = X - dt0 * sampleU(g, u, mx, my);
  let y = Y - dt0 * sampleV(g, v, mx, my);
  out[0] = x < 0 ? 0 : x > nx ? nx : x;
  out[1] = y < 0 ? 0 : y > ny ? ny : y;
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
  for (let j = 1; j <= ny; j++) {
    for (let i = 1; i < nx; i++) {
      const k = i + us * j;
      const X = i;
      const Y = j - 0.5;
      // v at this u-face = average of the four surrounding v-faces
      const vb = i + vs * (j - 1);
      const vy = 0.25 * (v0[vb] + v0[vb + 1] + v0[vb + vs] + v0[vb + vs + 1]);
      backtrace(g, u0, v0, X, Y, u0[k], vy, dt0, pos);
      u[k] = sampleU(g, u0, pos[0], pos[1]);
    }
  }
  for (let j = 1; j < ny; j++) {
    for (let i = 1; i <= nx; i++) {
      const k = i + vs * j;
      const X = i - 0.5;
      const Y = j;
      const ub = i - 1 + us * j;
      const vx = 0.25 * (u0[ub] + u0[ub + 1] + u0[ub + us] + u0[ub + us + 1]);
      backtrace(g, u0, v0, X, Y, vx, v0[k], dt0, pos);
      v[k] = sampleV(g, v0, pos[0], pos[1]);
    }
  }
  setUBoundary(g, u, walls);
  setVBoundary(g, v, walls);
}

/** Advect a cell-centred scalar d0 → d through the face velocity field (u, v). */
export function advectScalar(
  g: Grid, d: Float32Array, d0: Float32Array, u: Float32Array, v: Float32Array, dt: number,
): void {
  const { nx, ny, stride: s, uStride: us, vStride: vs } = g;
  const dt0 = dt / g.h;
  for (let j = 1; j <= ny; j++) {
    for (let i = 1; i <= nx; i++) {
      const uk = i - 1 + us * j;
      const vk = i + vs * (j - 1);
      const vx = 0.5 * (u[uk] + u[uk + 1]);
      const vy = 0.5 * (v[vk] + v[vk + vs]);
      backtrace(g, u, v, i - 0.5, j - 0.5, vx, vy, dt0, pos);
      d[i + s * j] = sampleCell(g, d0, pos[0], pos[1]);
    }
  }
  setScalarBoundary(g, d);
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

// ---------------------------------------------------------------------------
// Diagnostics & vorticity confinement
// ---------------------------------------------------------------------------

/** Interpolate face velocities to cell centres (uc, vc), with ghost cells copied. */
export function cellVelocity(
  g: Grid, u: Float32Array, v: Float32Array, uc: Float32Array, vc: Float32Array,
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
  setScalarBoundary(g, uc);
  setScalarBoundary(g, vc);
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
): void {
  if (epsilon <= 0) return;
  const { nx, ny, stride: s, uStride: us, vStride: vs, h } = g;
  const { uc, vc, curl, fx, fy } = scratch;
  cellVelocity(g, u, v, uc, vc);
  computeVorticity(g, uc, vc, curl);
  fx.fill(0);
  fy.fill(0);
  const inv2h = 0.5 / h;
  const effDt = Math.min(dt, MAX_CONFINEMENT_EPS_DT / epsilon);
  const scale = epsilon * h;
  for (let j = 2; j < ny; j++) {
    let k = j * s + 2;
    for (let i = 2; i < nx; i++, k++) {
      const dx = (Math.abs(curl[k + 1]) - Math.abs(curl[k - 1])) * inv2h;
      const dy = (Math.abs(curl[k + s]) - Math.abs(curl[k - s])) * inv2h;
      const len = Math.hypot(dx, dy) + 1e-10;
      const w = curl[k];
      fx[k] = scale * (dy / len) * w;
      fy[k] = -scale * (dx / len) * w;
    }
  }
  for (let j = 1; j <= ny; j++) {
    for (let i = 1; i < nx; i++) {
      const c = i + s * j;
      u[i + us * j] += effDt * 0.5 * (fx[c] + fx[c + 1]);
    }
  }
  for (let j = 1; j < ny; j++) {
    for (let i = 1; i <= nx; i++) {
      const c = i + s * j;
      v[i + vs * j] += effDt * 0.5 * (fy[c] + fy[c + s]);
    }
  }
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
