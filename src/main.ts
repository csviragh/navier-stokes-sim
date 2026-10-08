/**
 * App entry point: wires the solver, renderer, controls and pointer input
 * together and runs the animation loop.
 */
import './style.css';
import { FluidSolver, type FluidParams } from './solver';
import { Renderer } from './render/renderer';
import { hexToRgb, hsv } from './render/colormaps';
import { bindControls } from './ui/controls';
import { PointerInput } from './ui/pointer';
import { defaultSettings, type Settings } from './ui/settings';

const canvas = document.getElementById('sim') as HTMLCanvasElement;
const overlay = document.getElementById('overlay') as HTMLCanvasElement;
const fpsEl = document.getElementById('fps')!;
const statsEl = document.getElementById('stats')!;
const legendEl = document.getElementById('legend')!;

const settings: Settings = { ...defaultSettings };
const renderer = new Renderer(canvas, overlay);
const pointer = new PointerInput(canvas);

/** Maximum stirring speed (domain units / s) to keep the CFL number sane. */
const MAX_STIR_SPEED = 4;

function solverParams(s: Settings): Partial<FluidParams> {
  return {
    viscosity: s.viscosity,
    diffusion: s.diffusion,
    vorticity: s.vorticity,
    dyeDissipation: s.dyeDissipation,
    walls: s.walls,
    dt: s.dt,
    iterations: s.iterations,
  };
}

/** Grid size that fits the window aspect ratio with `resolution` cells on the short side. */
function gridSize(resolution: number): [number, number] {
  const w = Math.max(canvas.clientWidth, 1);
  const h = Math.max(canvas.clientHeight, 1);
  const aspect = Math.min(Math.max(w / h, 1 / 3), 3);
  return aspect >= 1
    ? [Math.round(resolution * aspect), resolution]
    : [resolution, Math.round(resolution / aspect)];
}

let solver = createSolver();

function createSolver(previous?: FluidSolver): FluidSolver {
  const [nx, ny] = gridSize(settings.resolution);
  const s = new FluidSolver(nx, ny, solverParams(settings));
  if (previous) s.resampleFrom(previous);
  else seedDemo(s);
  return s;
}

/** A few colourful jets so the first frame isn't empty. */
function seedDemo(s: FluidSolver): void {
  const { nx, ny } = s.grid;
  const r = Math.max(2, Math.min(nx, ny) / 22);
  s.splat(nx * 0.2, ny * 0.45, 1.6, 0.25, [0.1, 0.55, 1.0], r);
  s.splat(nx * 0.8, ny * 0.55, -1.6, -0.25, [1.0, 0.35, 0.1], r);
  s.splat(nx * 0.5, ny * 0.12, 0.0, 1.4, [0.2, 1.0, 0.45], r);
}

function currentColor(t: number): [number, number, number] {
  return settings.rainbow ? hsv((t * 0.08) % 1, 0.85, 1) : hexToRgb(settings.color);
}

function applyInput(): void {
  const { nx, ny, h } = solver.grid;
  const radius = settings.radius * (settings.resolution / 128);
  const color = currentColor(performance.now() / 1000);
  for (const st of pointer.take()) {
    const X0 = st.x0 * nx;
    const Y0 = st.y0 * ny;
    const X1 = st.x1 * nx;
    const Y1 = st.y1 * ny;
    const len = Math.hypot(X1 - X0, Y1 - Y0);
    const n = Math.max(1, Math.ceil(len / Math.max(0.5, radius * 0.5)));
    // pointer velocity in domain units/s (domain width = nx * h)
    let tu = st.vx * nx * h * settings.force;
    let tv = st.vy * ny * h * settings.force;
    const sp = Math.hypot(tu, tv);
    if (sp > MAX_STIR_SPEED) {
      tu *= MAX_STIR_SPEED / sp;
      tv *= MAX_STIR_SPEED / sp;
    }
    const dyeAmount = 0.6 / Math.sqrt(n);
    const c: [number, number, number] = [color[0] * dyeAmount, color[1] * dyeAmount, color[2] * dyeAmount];
    for (let k = 0; k < n; k++) {
      const t = n === 1 ? 1 : (k + 1) / n;
      const X = X0 + t * (X1 - X0);
      const Y = Y0 + t * (Y1 - Y0);
      if (sp > 0) solver.stir(X, Y, tu, tv, radius, 0.6);
      solver.splat(X, Y, 0, 0, c, radius);
    }
  }
}

let stepMs = 0;
function simulate(): void {
  const t0 = performance.now();
  applyInput();
  solver.step();
  stepMs = 0.9 * stepMs + 0.1 * (performance.now() - t0);
}

const controls = bindControls(settings, {
  onChange(key) {
    if (key === 'resolution') {
      solver = createSolver(solver);
      renderer.resetScales();
    } else if (key === 'view') {
      renderer.resetScales();
    } else {
      solver.setParams(solverParams(settings));
    }
  },
  onStep() {
    settings.paused = true;
    controls.sync();
    simulate();
  },
  onReset() {
    solver.reset();
    seedDemo(solver);
    renderer.resetScales();
  },
});

let resizeTimer: number | undefined;
window.addEventListener('resize', () => {
  renderer.resize();
  window.clearTimeout(resizeTimer);
  resizeTimer = window.setTimeout(() => {
    const [nx, ny] = gridSize(settings.resolution);
    if (nx !== solver.grid.nx || ny !== solver.grid.ny) solver = createSolver(solver);
  }, 150);
});

let frames = 0;
let fpsTime = performance.now();
function frame(now: number): void {
  if (!settings.paused) simulate();
  else applyInput(); // paint dye / stir while paused; it evolves on Play/Step
  solver.updateDiagnostics();
  const info = renderer.render(solver, settings.view, settings.arrows);

  frames++;
  if (now - fpsTime >= 500) {
    const fps = (frames * 1000) / (now - fpsTime);
    fpsEl.textContent = `${fps.toFixed(0)} fps`;
    statsEl.textContent =
      `${solver.grid.nx}×${solver.grid.ny} · step ${stepMs.toFixed(1)} ms · t = ${solver.time.toFixed(1)} s`;
    legendEl.textContent = info.legend;
    frames = 0;
    fpsTime = now;
  }
  requestAnimationFrame(frame);
}

renderer.resize();
requestAnimationFrame(frame);

// Handy for debugging from the console and for automated smoke tests.
declare global {
  interface Window {
    fluid: { readonly solver: FluidSolver; settings: Settings };
  }
}
window.fluid = {
  get solver() {
    return solver;
  },
  settings,
};
