/**
 * App entry point: wires the solver, renderer, controls and pointer input
 * together and runs the animation loop.
 */
import './style.css';
import { FluidSolver, uIdx, vIdx, type FluidParams } from './solver';
import { Renderer } from './render/renderer';
import { hexToRgb, hsv } from './render/colormaps';
import { bindControls } from './ui/controls';
import { PointerInput } from './ui/pointer';
import { CorrelationPanel } from './ui/correlations';
import { defaultSettings, type Settings } from './ui/settings';

const canvas = document.getElementById('sim') as HTMLCanvasElement;
const overlay = document.getElementById('overlay') as HTMLCanvasElement;
const fpsEl = document.getElementById('fps')!;
const statsEl = document.getElementById('stats')!;
const legendEl = document.getElementById('legend')!;

const settings: Settings = { ...defaultSettings };

/**
 * "Delay oscillations" quick start: Cucker–Smale alignment (R = 0.15) with
 * the own and the neighbours' velocity both t_self = t_nb = 0.42 s old at
 * S = 4, so S·t = 1.68 > π/2 (the 6c811e2 "calculation delay" setting; the
 * local viscosity is left undelayed, as it was then). Short-wave modes
 * (K̂ ≈ 0, a_k ≈ S) obey u' ≈ −S u(t − t), whose rightmost root
 * λ = W₀(−S t)/t = 0.11 ± 3.81i s⁻¹ grows slowly while oscillating with a
 * period of 1.65 s: the whole vortex pattern reverses its sense of rotation
 * about every 0.8 s. Weak self-propulsion (τ = 2 s, v₀ = 0.3) saturates the
 * growth into a sustained oscillation (overlap trace swings ±1, ⟨|u|⟩ ≈ 0.35).
 */
const DELAY_OSCILLATIONS: Partial<Settings> = {
  kernel: 'cucker-smale', kernelStrength: 4, kernelRange: 0.15, kernelBeta: 1, kernelMT: false,
  delaySelf: 0.42, delayNb: 0.42, delayLocal: false, delayKernel: true,
  vorticity: 0, selfPropulsion: true, selfPropTau: 2, selfPropV0: 0.3, walls: 'periodic',
  viscosity: 1e-5, dyeDissipation: 0,
};

/**
 * "Local viscosity delay" quick start: no kernel, only ν∇² delayed, with the
 * neighbours' velocities much older than the own one: t_self = 0.5 s,
 * t_nb = 2 s, and ν set from the grid so that A = 4ν/h² = 2 s⁻¹ (the
 * self/neighbour split of the discrete Laplacian has rates ∝ 1/h²). In units
 * of 1/A this is A·t_self = 1, A·t_nb = 4, where the longest waves are the
 * most unstable modes: λ = 0.26 ± 2.66i s⁻¹ (period 2.4 s), the checkerboard
 * grows only half as fast (0.14 /s) and is wiped out by the interpolation of
 * the advection step once the fluid moves. Self-propulsion with a short τ
 * (0.2 s, v₀ = 0.3) bounds the amplitude, so the large-scale flow keeps
 * sloshing back and forth (overlap trace).
 */
const LOCAL_DELAY_RATE = 2;
const LOCAL_DELAY: Partial<Settings> = {
  kernel: 'none', delaySelf: 0.5, delayNb: 2, delayLocal: true, delayKernel: true,
  vorticity: 0, selfPropulsion: true, selfPropTau: 0.2, selfPropV0: 0.3, walls: 'periodic', dyeDissipation: 0,
};

/**
 * "Own-velocity delay" quick start: the "Delay oscillations" alignment with
 * only the own velocity delayed (t_self = 0.42 s, t_nb = 0). For long waves
 * (K̂ → 1) λ = S(K̂ − e^{−λ t_self}) has a positive real root once
 * S·t_self > 1: the point brakes against where it WAS while following where
 * its neighbours ARE, so large-scale motion amplifies itself (the mean
 * flow becomes marginal: dM/dt = −S[M(t−t_self) − M]). The fast
 * self-propulsion (τ = 0.1 s) caps the speed, so the result is a single
 * aligned flock (φ → 1) whose heading rotates steadily at ω ≈ 4 rad/s
 * (ω = S·sin(ω t_self)). Compare with t_self = 0, t_nb = 0.42 (stable: own
 * velocity current, |B| ≤ A).
 */
const OWN_DELAY: Partial<Settings> = {
  ...DELAY_OSCILLATIONS, delaySelf: 0.42, delayNb: 0, selfPropTau: 0.1,
};

/** Polar order parameter φ = |⟨u⟩| / ⟨|u|⟩ (1 = perfectly aligned flock); needs updateDiagnostics(). */
function polarOrder(s: FluidSolver): number {
  const g = s.grid;
  let mu = 0, mv = 0, sp = 0;
  for (let j = 1; j <= g.ny; j++) for (let i = 1; i <= g.nx; i++) {
    const k = i + g.stride * j;
    mu += s.uc[k]; mv += s.vc[k]; sp += Math.hypot(s.uc[k], s.vc[k]);
  }
  return sp > 0 ? Math.hypot(mu, mv) / sp : 0;
}
const renderer = new Renderer(canvas, overlay);
const corrPanel = new CorrelationPanel();
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
    selfPropulsion: s.selfPropulsion,
    selfPropTau: s.selfPropTau,
    selfPropV0: s.selfPropV0,
    kernel: s.kernel,
    kernelStrength: s.kernelStrength,
    kernelRange: s.kernelRange,
    kernelBeta: s.kernelBeta,
    kernelAlpha: s.kernelAlpha,
    kernelS: s.kernelS,
    kernelMT: s.kernelMT,
    delaySelf: s.delaySelf,
    delayNb: s.delayNb,
    delayLocal: s.delayLocal,
    delayKernel: s.delayKernel,
    delayMode: s.delayMode,
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

/**
 * HUD time trace (last TRACE_SPAN s of simulated time): polar order φ, mean
 * speed ⟨|u|⟩ and the overlap C(t) = ⟨u·u_ref⟩/⟨u_ref·u_ref⟩ with the flow at
 * the last reset / quick start (C swings between ± when the flow pattern
 * reverses, e.g. delay oscillations).
 */
const TRACE_SPAN = 20;
const traceCanvas = document.getElementById('trace') as HTMLCanvasElement;
const trace = {
  t: [] as number[], phi: [] as number[], speed: [] as number[], overlap: [] as number[],
  refU: null as Float32Array | null, refV: null as Float32Array | null, refNorm: 0,
  reset(s: FluidSolver) {
    this.t = []; this.phi = []; this.speed = []; this.overlap = [];
    corrPanel.reset();
    this.refU = Float32Array.from(s.u); this.refV = Float32Array.from(s.v);
    let n = 0;
    for (const x of this.refU) n += x * x;
    for (const x of this.refV) n += x * x;
    this.refNorm = n;
  },
  sample(s: FluidSolver) {
    const g = s.grid;
    if (this.t.length && s.time <= this.t[this.t.length - 1]) return;
    let mu = 0, mv = 0, sp = 0;
    for (let j = 1; j <= g.ny; j++) for (let i = 1; i <= g.nx; i++) {
      const k = i + g.stride * j;
      mu += s.uc[k]; mv += s.vc[k]; sp += Math.hypot(s.uc[k], s.vc[k]);
    }
    let c = 0;
    const ru = this.refU, rv = this.refV;
    if (ru && rv && ru.length === s.u.length && this.refNorm > 0) {
      for (let j = 1; j <= g.ny; j++) for (let i = 1; i <= g.nx; i++) {
        const a = uIdx(g, i, j), b = vIdx(g, i, j);
        c += s.u[a] * ru[a] + s.v[b] * rv[b];
      }
      c /= this.refNorm;
    }
    this.t.push(s.time); this.phi.push(sp > 0 ? Math.hypot(mu, mv) / sp : 0);
    this.speed.push(sp / (g.nx * g.ny)); this.overlap.push(c);
    while (this.t.length > 2 && this.t[0] < s.time - TRACE_SPAN) {
      this.t.shift(); this.phi.shift(); this.speed.shift(); this.overlap.shift();
    }
  },
  draw() {
    const cv = traceCanvas;
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const W = Math.round(cv.clientWidth * dpr), H = Math.round(cv.clientHeight * dpr);
    if (cv.width !== W || cv.height !== H) { cv.width = W; cv.height = H; }
    const ctx = cv.getContext('2d');
    if (!ctx || !this.t.length) return;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    const w = cv.clientWidth, h = cv.clientHeight, top = 14, bot = h - 4;
    ctx.clearRect(0, 0, w, h);
    const t1 = this.t[this.t.length - 1], t0 = t1 - TRACE_SPAN;
    const X = (t: number) => 4 + ((t - t0) / TRACE_SPAN) * (w - 8);
    const cOver = Math.max(1, ...this.overlap.map(Math.abs));
    const sMax = Math.max(1e-6, ...this.speed);
    const Y = (y: number) => bot - y * (bot - top); // y ∈ [0, 1]
    ctx.strokeStyle = 'rgba(255,255,255,0.15)';
    ctx.beginPath(); ctx.moveTo(4, Y(0.5)); ctx.lineTo(w - 4, Y(0.5)); ctx.stroke();
    const line = (ys: number[], f: (y: number) => number, color: string) => {
      ctx.strokeStyle = color; ctx.lineWidth = 1.3; ctx.beginPath();
      ys.forEach((y, k) => { const px = X(this.t[k]), py = Y(f(y)); if (k) ctx.lineTo(px, py); else ctx.moveTo(px, py); });
      ctx.stroke();
    };
    line(this.overlap, (y) => 0.5 + 0.5 * y / cOver, '#7ee787');
    line(this.speed, (y) => y / sMax, '#ffa657');
    line(this.phi, (y) => y, '#58a6ff');
    ctx.font = '10px ui-monospace, monospace';
    const lab = (txt: string, x: number, col: string) => { ctx.fillStyle = col; ctx.fillText(txt, x, 10); };
    lab(`φ ${this.phi[this.phi.length - 1].toFixed(2)}`, 6, '#58a6ff');
    lab(`⟨|u|⟩ ${this.speed[this.speed.length - 1].toFixed(3)}`, 62, '#ffa657');
    lab(`overlap ${this.overlap[this.overlap.length - 1].toFixed(2)}`, 150, '#7ee787');
    ctx.fillStyle = 'rgba(255,255,255,0.45)';
    ctx.fillText(`${TRACE_SPAN} s`, w - 30, h - 6);
  },
};

let solver = createSolver();

function createSolver(previous?: FluidSolver): FluidSolver {
  const [nx, ny] = gridSize(settings.resolution);
  const s = new FluidSolver(nx, ny, solverParams(settings));
  if (previous) s.resampleFrom(previous);
  else seedDemo(s);
  trace.reset(s);
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
    trace.reset(solver);
    renderer.resetScales();
  },
  onActiveTurbulence() {
    // Quick start: switch the active term on and start from weak random swirls,
    // which it amplifies until the local speed approaches v₀. Periodic
    // boundaries (no walls to brake the flow) let |u| reach ≈ v₀ everywhere.
    // The global default boundary stays no-slip; the user can switch back.
    settings.selfPropulsion = true;
    settings.walls = 'periodic';
    settings.paused = false;
    solver.setParams(solverParams(settings));
    solver.reset();
    solver.seedActiveTurbulence((Math.random() * 2 ** 32) >>> 0, 0.05 * Math.max(settings.selfPropV0, 0.1) / 0.5);
    trace.reset(solver);
    renderer.resetScales();
  },
  onFlocking() {
    // Quick start: Cucker–Smale alignment + self-propulsion on a torus.
    settings.kernel = 'cucker-smale';
    settings.kernelStrength = 3;
    settings.kernelRange = 0.05;
    settings.kernelBeta = 1;
    settings.delaySelf = 0;
    settings.delayNb = 0;
    // vorticity confinement amplifies grid-scale noise once the flow is ordered
    settings.vorticity = 0;
    settings.selfPropulsion = true;
    settings.walls = 'periodic';
    settings.paused = false;
    solver.setParams(solverParams(settings));
    solver.reset();
    solver.seedActiveTurbulence((Math.random() * 2 ** 32) >>> 0, 0.05 * Math.max(settings.selfPropV0, 0.1) / 0.5);
    trace.reset(solver);
    renderer.resetScales();
  },
  onDelayOscillations() {
    // Quick start: equal own / neighbour delays beyond the oscillation threshold, see DELAY_OSCILLATIONS.
    // The vorticity view shows the reversals best (vortices flip colour every half period).
    delayQuickStart(DELAY_OSCILLATIONS, 'vorticity');
  },
  onLocalDelay() {
    // Quick start: delayed local viscosity only, see LOCAL_DELAY; ν from the current grid (A = 4ν/h²).
    const h = solver.grid.h;
    delayQuickStart({ ...LOCAL_DELAY, viscosity: (LOCAL_DELAY_RATE * h * h) / 4 }, 'vorticity');
  },
  onOwnDelay() {
    // Quick start: only the own velocity delayed, see OWN_DELAY.
    delayQuickStart(OWN_DELAY, 'vorticity');
  },
});

function delayQuickStart(p: Partial<Settings>, view: Settings['view']): void {
  Object.assign(settings, p);
  settings.view = view;
  settings.paused = false;
  solver.setParams(solverParams(settings));
  solver.reset();
  solver.seedActiveTurbulence((Math.random() * 2 ** 32) >>> 0, 0.05);
  trace.reset(solver);
  renderer.resetScales();
  controls.sync();
}

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
  trace.sample(solver);
  trace.draw();
  corrPanel.update(solver, polarOrder);

  frames++;
  if (now - fpsTime >= 500) {
    const fps = (frames * 1000) / (now - fpsTime);
    fpsEl.textContent = `${fps.toFixed(0)} fps`;
    statsEl.textContent =
      `${solver.grid.nx}×${solver.grid.ny} · step ${stepMs.toFixed(1)} ms · t = ${solver.time.toFixed(1)} s` +
      (settings.kernel !== 'none' || settings.selfPropulsion ? ` · order φ = ${polarOrder(solver).toFixed(2)}` : '');
    legendEl.textContent = info.legend;
    controls.setDelayInfo(solver.delayInfo(), solver.delayStability());
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
    fluid: { readonly solver: FluidSolver; settings: Settings; trace: typeof trace; corr: CorrelationPanel };
  }
}
window.fluid = {
  get solver() {
    return solver;
  },
  settings,
  trace,
  corr: corrPanel,
};
