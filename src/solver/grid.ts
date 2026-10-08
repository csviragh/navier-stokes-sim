/**
 * Uniform 2D MAC (marker-and-cell, staggered) grid.
 *
 * The domain is covered by nx × ny square cells of size h (the shorter side of
 * the domain has length 1). Positions used throughout the solver are in
 * *grid units*: X ∈ [0, nx], Y ∈ [0, ny], one unit per cell.
 *
 * Storage layout (row-major, x fastest):
 *
 *  - Cell-centred scalars (pressure, dye, divergence, …):
 *      (nx + 2) × (ny + 2) values, interior i = 1..nx, j = 1..ny, plus one ring
 *      of ghost cells. Cell (i, j) is centred at X = i − ½, Y = j − ½.
 *  - u (x-velocity) on vertical faces:
 *      (nx + 1) × (ny + 2) values. Face (i, j) sits at X = i, Y = j − ½,
 *      i = 0..nx (i = 0 and i = nx are the left/right walls), j = 1..ny,
 *      rows j = 0 and j = ny + 1 are ghost rows for the tangential condition.
 *  - v (y-velocity) on horizontal faces:
 *      (nx + 2) × (ny + 1) values. Face (i, j) sits at X = i − ½, Y = j,
 *      j = 0..ny (j = 0 and j = ny are the bottom/top walls), columns i = 0 and
 *      i = nx + 1 are ghost columns.
 *
 * The staggered arrangement couples pressure and velocity tightly, so the
 * discrete projection makes the discrete divergence exactly zero (up to the
 * linear-solver tolerance) without checkerboard artefacts.
 */
export interface Grid {
  readonly nx: number;
  readonly ny: number;
  /** Physical cell size (shorter domain side = 1). */
  readonly h: number;
  /** Cell-centred field: stride and length. */
  readonly stride: number;
  readonly size: number;
  /** u-face field: stride and length. */
  readonly uStride: number;
  readonly uSize: number;
  /** v-face field: stride and length. */
  readonly vStride: number;
  readonly vSize: number;
}

export function createGrid(nx: number, ny: number): Grid {
  if (!Number.isInteger(nx) || !Number.isInteger(ny) || nx < 4 || ny < 4) {
    throw new Error(`Grid dimensions must be integers >= 4 (got ${nx}x${ny})`);
  }
  return {
    nx,
    ny,
    h: 1 / Math.min(nx, ny),
    stride: nx + 2,
    size: (nx + 2) * (ny + 2),
    uStride: nx + 1,
    uSize: (nx + 1) * (ny + 2),
    vStride: nx + 2,
    vSize: (nx + 2) * (ny + 1),
  };
}

/** Index of cell (i, j) in a cell-centred field. */
export const idx = (g: Grid, i: number, j: number): number => i + g.stride * j;
/** Index of u-face (i, j). */
export const uIdx = (g: Grid, i: number, j: number): number => i + g.uStride * j;
/** Index of v-face (i, j). */
export const vIdx = (g: Grid, i: number, j: number): number => i + g.vStride * j;

export const createField = (g: Grid): Float32Array => new Float32Array(g.size);
export const createUField = (g: Grid): Float32Array => new Float32Array(g.uSize);
export const createVField = (g: Grid): Float32Array => new Float32Array(g.vSize);

/**
 * Wall condition for the velocity component *tangential* to a wall.
 * The normal component is always zero (solid, impermeable walls).
 */
export type WallType = 'no-slip' | 'free-slip';
