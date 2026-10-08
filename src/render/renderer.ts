/**
 * Canvas2D renderer: rasterises one solver field per frame into an ImageData
 * at grid resolution and lets the browser upscale it (bilinear smoothing) to
 * the full-window canvas. A second canvas holds the velocity-arrow overlay.
 *
 * The renderer only *reads* solver fields; it never modifies the simulation.
 */
import type { FluidSolver } from '../solver';
import { diverging, viridis, type LUT } from './colormaps';

export type ViewMode = 'dye' | 'speed' | 'pressure' | 'vorticity' | 'divergence';
export const VIEW_MODES: readonly ViewMode[] = ['dye', 'speed', 'pressure', 'vorticity', 'divergence'];

export interface RenderInfo {
  /** Human-readable description of the colour scale currently in use. */
  legend: string;
}

/** Smoothly tracks a robust maximum so the colour scale doesn't flicker. */
class AutoScale {
  private value = 0;
  update(observed: number): number {
    const target = Math.max(observed, 1e-6);
    // rise fast, decay slowly
    this.value = target > this.value ? 0.5 * this.value + 0.5 * target : 0.97 * this.value + 0.03 * target;
    return this.value;
  }
  reset(): void {
    this.value = 0;
  }
}

export class Renderer {
  private readonly ctx: CanvasRenderingContext2D;
  private readonly overlayCtx: CanvasRenderingContext2D;
  private readonly buffer: HTMLCanvasElement;
  private readonly bufferCtx: CanvasRenderingContext2D;
  private image: ImageData | null = null;
  private readonly scales = new Map<string, AutoScale>();

  constructor(
    private readonly canvas: HTMLCanvasElement,
    private readonly overlay: HTMLCanvasElement,
  ) {
    const ctx = canvas.getContext('2d', { alpha: false });
    const octx = overlay.getContext('2d');
    this.buffer = document.createElement('canvas');
    const bctx = this.buffer.getContext('2d');
    if (!ctx || !octx || !bctx) throw new Error('Canvas 2D is not supported in this browser');
    this.ctx = ctx;
    this.overlayCtx = octx;
    this.bufferCtx = bctx;
  }

  /** Match the backing store to the element size (CSS px × devicePixelRatio). */
  resize(): void {
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    for (const c of [this.canvas, this.overlay]) {
      const w = Math.round(c.clientWidth * dpr);
      const h = Math.round(c.clientHeight * dpr);
      if (c.width !== w || c.height !== h) {
        c.width = w;
        c.height = h;
      }
    }
  }

  resetScales(): void {
    for (const s of this.scales.values()) s.reset();
  }

  private scale(key: string, observed: number): number {
    let s = this.scales.get(key);
    if (!s) this.scales.set(key, (s = new AutoScale()));
    return s.update(observed);
  }

  render(solver: FluidSolver, mode: ViewMode, arrows: boolean): RenderInfo {
    const g = solver.grid;
    const { nx, ny, stride } = g;
    if (!this.image || this.image.width !== nx || this.image.height !== ny) {
      this.buffer.width = nx;
      this.buffer.height = ny;
      this.image = this.bufferCtx.createImageData(nx, ny);
    }
    const px = this.image.data;
    let legend = '';

    // Screen row 0 is the top of the domain (j = ny).
    const forEachCell = (fn: (k: number, o: number) => void): void => {
      for (let row = 0; row < ny; row++) {
        const j = ny - row;
        let k = j * stride + 1;
        let o = row * nx * 4;
        for (let i = 0; i < nx; i++, k++, o += 4) fn(k, o);
      }
    };

    if (mode === 'dye') {
      const [r, gr, b] = solver.dye;
      // exposure-style tone mapping keeps overlapping dye from clipping harshly
      forEachCell((k, o) => {
        px[o] = 255 * (1 - Math.exp(-1.6 * Math.max(r[k], 0)));
        px[o + 1] = 255 * (1 - Math.exp(-1.6 * Math.max(gr[k], 0)));
        px[o + 2] = 255 * (1 - Math.exp(-1.6 * Math.max(b[k], 0)));
        px[o + 3] = 255;
      });
      legend = 'dye (RGB concentration)';
    } else if (mode === 'speed') {
      const { uc, vc } = solver;
      let max = 0;
      forEachCell((k) => {
        const m = uc[k] * uc[k] + vc[k] * vc[k];
        if (m > max) max = m;
      });
      const s = this.scale('speed', Math.sqrt(max));
      const inv = 255 / s;
      forEachCell((k, o) => {
        const m = Math.sqrt(uc[k] * uc[k] + vc[k] * vc[k]);
        this.put(px, o, viridis, Math.min(255, (m * inv) | 0));
      });
      legend = `|u|: 0 … ${fmt(s)} (viridis)`;
    } else {
      const field =
        mode === 'pressure' ? solver.pressure : mode === 'vorticity' ? solver.vorticity : solver.divergence;
      let max = 0;
      forEachCell((k) => {
        const a = Math.abs(field[k]);
        if (a > max) max = a;
      });
      const s = this.scale(mode, max);
      const half = 127.5 / s;
      forEachCell((k, o) => {
        const t = Math.round(127.5 + field[k] * half);
        this.put(px, o, diverging, t < 0 ? 0 : t > 255 ? 255 : t);
      });
      const name = mode === 'pressure' ? 'p' : mode === 'vorticity' ? 'ω' : '∇·u';
      legend = `${name}: ±${fmt(s)} (blue − / red +)`;
    }

    this.bufferCtx.putImageData(this.image, 0, 0);
    const { ctx, canvas } = this;
    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = 'high';
    ctx.drawImage(this.buffer, 0, 0, canvas.width, canvas.height);

    this.drawArrows(solver, arrows);
    return { legend };
  }

  private put(px: Uint8ClampedArray, o: number, lut: LUT, t: number): void {
    const l = 3 * t;
    px[o] = lut[l];
    px[o + 1] = lut[l + 1];
    px[o + 2] = lut[l + 2];
    px[o + 3] = 255;
  }

  private drawArrows(solver: FluidSolver, enabled: boolean): void {
    const ctx = this.overlayCtx;
    const { width, height } = this.overlay;
    ctx.clearRect(0, 0, width, height);
    if (!enabled) return;
    const { nx, ny } = solver.grid;
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const spacing = 28 * dpr;
    const cols = Math.max(2, Math.floor(width / spacing));
    const rows = Math.max(2, Math.floor(height / spacing));
    const dx = width / cols;
    const dy = height / rows;

    // find max speed on the sample lattice for scaling
    const samples: number[] = [];
    let max = 0;
    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        const X = ((c + 0.5) / cols) * nx;
        const Y = (1 - (r + 0.5) / rows) * ny;
        const u = solver.sample(solver.uc, X, Y);
        const v = solver.sample(solver.vc, X, Y);
        samples.push(u, v);
        max = Math.max(max, Math.hypot(u, v));
      }
    }
    const s = this.scale('arrows', max);
    const maxLen = 0.9 * Math.min(dx, dy);
    ctx.lineWidth = 1.2 * dpr;
    ctx.strokeStyle = 'rgba(255,255,255,0.85)';
    ctx.fillStyle = 'rgba(255,255,255,0.85)';
    ctx.beginPath();
    let n = 0;
    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        const u = samples[n++];
        const v = samples[n++];
        const m = Math.hypot(u, v);
        if (m < 1e-3 * s) continue;
        const len = Math.min(1, m / s) * maxLen;
        const ux = u / m;
        const uy = -v / m; // screen y points down
        const cx = (c + 0.5) * dx;
        const cy = (r + 0.5) * dy;
        const x0 = cx - 0.5 * len * ux;
        const y0 = cy - 0.5 * len * uy;
        const x1 = cx + 0.5 * len * ux;
        const y1 = cy + 0.5 * len * uy;
        ctx.moveTo(x0, y0);
        ctx.lineTo(x1, y1);
        const head = Math.min(6 * dpr, 0.4 * len);
        ctx.moveTo(x1, y1);
        ctx.lineTo(x1 - head * (ux - 0.5 * uy), y1 - head * (uy + 0.5 * ux));
        ctx.moveTo(x1, y1);
        ctx.lineTo(x1 - head * (ux + 0.5 * uy), y1 - head * (uy - 0.5 * ux));
      }
    }
    ctx.stroke();
  }
}

function fmt(x: number): string {
  if (x === 0) return '0';
  const a = Math.abs(x);
  return a >= 100 || a < 0.01 ? x.toExponential(1) : x.toPrecision(3);
}
