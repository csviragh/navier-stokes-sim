/**
 * Direct (non-iterative) solver for the pressure Poisson equation on a grid
 * with periodic x (and optionally periodic y):
 *
 *   L p = r,   L = the MAC 5-point Laplacian (times h²) with wrap-around
 *              neighbours on periodic axes and ∂p/∂n = 0 at walls.
 *
 * Along x, L is diagonalised by the DFT (eigenvalues 2cos(2πk/nx) − 2). For
 * each x-wavenumber k the remaining 1D problem in y is
 *  - fully periodic: a cyclic tridiagonal system, solved exactly in O(ny)
 *    (Thomas + Sherman–Morrison); the k = 0 column by a DFT in y;
 *  - walls in y (channel): a Neumann tridiagonal system, solved exactly with
 *    the Thomas algorithm.
 * The k = l = 0 (resp. k = 0 Neumann) mode is the singular constant mode; it
 * is solvable because Σr = 0 and is fixed by pinning, the caller then removes
 * the mean of p. The result is exact up to round-off, independent of any
 * iteration count, so the projection leaves ∇·u at round-off level in every
 * cell, including the seam cells.
 */
import { FFT } from './fft';
import type { Grid } from './grid';

export class PeriodicPoisson {
  private readonly fx: FFT;
  private readonly fy: FFT | null;
  private readonly lamX: Float64Array;
  private readonly lamY: Float64Array | null;
  /** Spectral coefficients, ny rows × nx wavenumbers. */
  private readonly sr: Float64Array;
  private readonly si: Float64Array;
  private readonly bufR: Float64Array;
  private readonly bufI: Float64Array;
  private readonly colR: Float64Array;
  private readonly colI: Float64Array;
  private readonly cp: Float64Array;
  private readonly z: Float64Array;

  constructor(private readonly g: Grid, periodicY: boolean) {
    const { nx, ny } = g;
    this.fx = new FFT(nx);
    this.fy = periodicY ? new FFT(ny) : null;
    this.lamX = new Float64Array(nx);
    for (let k = 0; k < nx; k++) this.lamX[k] = 2 * Math.cos((2 * Math.PI * k) / nx) - 2;
    this.lamY = periodicY ? new Float64Array(ny) : null;
    if (this.lamY) for (let l = 0; l < ny; l++) this.lamY[l] = 2 * Math.cos((2 * Math.PI * l) / ny) - 2;
    this.sr = new Float64Array(nx * ny);
    this.si = new Float64Array(nx * ny);
    this.bufR = new Float64Array(nx);
    this.bufI = new Float64Array(nx);
    this.colR = new Float64Array(ny);
    this.colI = new Float64Array(ny);
    this.cp = new Float64Array(ny);
    this.z = new Float64Array(ny);
  }

  /**
   * Solve L p = r. `rhs(i, j)` returns r for interior cell (i, j) (1-based);
   * the solution is written to the interior of the cell-centred field `p`.
   */
  solve(rhs: (i: number, j: number) => number, p: Float32Array): void {
    const { nx, ny, stride: s } = this.g;
    const { sr, si, bufR, bufI, fx } = this;
    // 1. DFT along x, two real rows per complex transform
    for (let j = 1; j <= ny; j += 2) {
      const pair = j + 1 <= ny;
      for (let i = 0; i < nx; i++) {
        bufR[i] = rhs(i + 1, j);
        bufI[i] = pair ? rhs(i + 1, j + 1) : 0;
      }
      fx.forward(bufR, bufI);
      const a = (j - 1) * nx;
      const b = j * nx;
      for (let k = 0; k < nx; k++) {
        const m = k === 0 ? 0 : nx - k;
        // A_k = (Z_k + conj Z_m)/2,  B_k = (Z_k − conj Z_m)/(2i)
        sr[a + k] = 0.5 * (bufR[k] + bufR[m]);
        si[a + k] = 0.5 * (bufI[k] - bufI[m]);
        if (pair) {
          sr[b + k] = 0.5 * (bufI[k] + bufI[m]);
          si[b + k] = -0.5 * (bufR[k] - bufR[m]);
        }
      }
    }
    // 2. per x-wavenumber: solve in y
    const { colR, colI } = this;
    for (let k = 0; k < nx; k++) {
      for (let j = 0; j < ny; j++) {
        colR[j] = sr[j * nx + k];
        colI[j] = si[j * nx + k];
      }
      if (this.fy) this.solvePeriodicY(k, colR, colI);
      else this.solveNeumannY(this.lamX[k], colR, colI);
      for (let j = 0; j < ny; j++) {
        sr[j * nx + k] = colR[j];
        si[j * nx + k] = colI[j];
      }
    }
    // 3. inverse DFT along x; the two rows are real, so pack them as re + i·im
    for (let j = 1; j <= ny; j += 2) {
      const pair = j + 1 <= ny;
      const a = (j - 1) * nx;
      const b = j * nx;
      for (let k = 0; k < nx; k++) {
        bufR[k] = sr[a + k] - (pair ? si[b + k] : 0);
        bufI[k] = si[a + k] + (pair ? sr[b + k] : 0);
      }
      fx.inverse(bufR, bufI);
      for (let i = 0; i < nx; i++) {
        p[i + 1 + s * j] = bufR[i];
        if (pair) p[i + 1 + s * (j + 1)] = bufI[i];
      }
    }
  }

  /**
   * (L_y + λx) q = r on a periodic column. For k ≠ 0 this is a cyclic
   * tridiagonal system (diagonal λx − 2, off-diagonals and corners 1), solved
   * in O(ny) with the Sherman–Morrison correction of the Thomas algorithm.
   * The singular k = 0 column is solved spectrally (DFT in y, constant mode 0).
   */
  private solvePeriodicY(k: number, re: Float64Array, im: Float64Array): void {
    if (k === 0) {
      const fy = this.fy!;
      const lamY = this.lamY!;
      fy.forward(re, im);
      re[0] = 0;
      im[0] = 0;
      for (let l = 1; l < re.length; l++) {
        re[l] /= lamY[l];
        im[l] /= lamY[l];
      }
      fy.inverse(re, im);
      return;
    }
    const n = re.length;
    const b = this.lamX[k] - 2; // |b| > 2: strictly diagonally dominant
    const gamma = -b;
    const cp = this.cp;
    const z = this.z;
    // Thomas for the modified matrix A' (b0 − γ, b_{n−1} − 1/γ), three right-hand sides
    const b0 = b - gamma;
    const bn = b - 1 / gamma;
    let d = b0;
    cp[0] = 1 / d;
    re[0] /= d;
    im[0] /= d;
    z[0] = gamma / d;
    for (let j = 1; j < n; j++) {
      d = (j === n - 1 ? bn : b) - cp[j - 1];
      cp[j] = 1 / d;
      re[j] = (re[j] - re[j - 1]) / d;
      im[j] = (im[j] - im[j - 1]) / d;
      z[j] = ((j === n - 1 ? 1 : 0) - z[j - 1]) / d;
    }
    for (let j = n - 2; j >= 0; j--) {
      re[j] -= cp[j] * re[j + 1];
      im[j] -= cp[j] * im[j + 1];
      z[j] -= cp[j] * z[j + 1];
    }
    const den = 1 + z[0] + z[n - 1] / gamma;
    const fr = (re[0] + re[n - 1] / gamma) / den;
    const fi = (im[0] + im[n - 1] / gamma) / den;
    for (let j = 0; j < n; j++) {
      re[j] -= fr * z[j];
      im[j] -= fi * z[j];
    }
  }

  /**
   * (L_y + λx) q = r with ∂q/∂n = 0 at j = 1 and j = ny:
   *   q_{j−1} + (λx − 2) q_j + q_{j+1} = r_j, with the missing neighbour (and
   *   one of the −1s) dropped in the first and last row.
   */
  private solveNeumannY(lx: number, re: Float64Array, im: Float64Array): void {
    const n = re.length;
    if (lx === 0) {
      // singular constant mode: (q_{j+1} − q_j) = Σ_{m≤j} r_m, q_1 = 0
      let fr = 0;
      let fi = 0;
      let qr = 0;
      let qi = 0;
      for (let j = 0; j < n; j++) {
        fr += re[j];
        fi += im[j];
        re[j] = qr;
        im[j] = qi;
        qr += fr;
        qi += fi;
      }
      return;
    }
    // Thomas algorithm (strictly diagonally dominant for λx < 0)
    const cp = this.cp;
    let d = lx - 1;
    cp[0] = 1 / d;
    re[0] /= d;
    im[0] /= d;
    for (let j = 1; j < n; j++) {
      d = (j === n - 1 ? lx - 1 : lx - 2) - cp[j - 1];
      cp[j] = 1 / d;
      re[j] = (re[j] - re[j - 1]) / d;
      im[j] = (im[j] - im[j - 1]) / d;
    }
    for (let j = n - 2; j >= 0; j--) {
      re[j] -= cp[j] * re[j + 1];
      im[j] -= cp[j] * im[j + 1];
    }
  }
}
