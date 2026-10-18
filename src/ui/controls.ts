/**
 * Binds the HTML control panel to a Settings object. Changes are reported via
 * the onChange callback with the name of the changed key.
 */
import { VIEW_MODES, type ViewMode } from '../render/renderer';
import { KERNEL_TYPES, kernelProfile, type DelayInfo } from '../solver';
import { WALL_TYPES } from '../solver';
import type { Settings } from './settings';

type Key = keyof Settings;

export interface ControlCallbacks {
  onChange(key: Key, settings: Settings): void;
  onStep(): void;
  onReset(): void;
  /** Quick start: enable self-propulsion, switch to periodic boundaries and seed weak random flow. */
  onActiveTurbulence(): void;
  /** Quick start: Cucker–Smale alignment + self-propulsion on periodic boundaries. */
  onFlocking(): void;
  /** Quick start: flocking with a finite interaction speed (retarded alignment). */
  onDelayedFlocking(): void;
}

const $ = <T extends HTMLElement>(id: string): T => {
  const el = document.getElementById(id);
  if (!el) throw new Error(`Missing element #${id}`);
  return el as T;
};

/** Log-scale slider: slider value is log10(x); the minimum position means 0. */
function logSlider(
  id: string, key: 'viscosity' | 'diffusion' | 'kernelStrength', s: Settings, cb: ControlCallbacks,
): () => void {
  const input = $<HTMLInputElement>(id);
  const out = $<HTMLOutputElement>(`${id}-out`);
  const min = Number(input.min);
  const sync = () => {
    const x = s[key];
    input.value = String(x <= 0 ? min : Math.max(min, Math.log10(x)));
    out.textContent = x <= 0 ? '0' : x.toExponential(1);
  };
  input.addEventListener('input', () => {
    const t = Number(input.value);
    s[key] = t <= min ? 0 : 10 ** t;
    sync();
    cb.onChange(key, s);
  });
  sync();
  return sync;
}

function linSlider(
  id: string,
  key: 'vorticity' | 'dyeDissipation' | 'dt' | 'iterations' | 'resolution' | 'radius' | 'force'
    | 'selfPropTau' | 'selfPropV0' | 'kernelRange' | 'kernelBeta' | 'kernelAlpha' | 'kernelS' | 'kernelShells',
  s: Settings, cb: ControlCallbacks, format: (x: number) => string,
  event: 'input' | 'change' = 'input',
): () => void {
  const input = $<HTMLInputElement>(id);
  const out = $<HTMLOutputElement>(`${id}-out`);
  const sync = () => {
    input.value = String(s[key]);
    out.textContent = format(s[key]);
  };
  input.addEventListener('input', () => {
    out.textContent = format(Number(input.value));
  });
  input.addEventListener(event, () => {
    s[key] = Number(input.value);
    sync();
    cb.onChange(key, s);
  });
  sync();
  return sync;
}

/** Log-scale v_int slider (log10 v); positions above log10(20) mean ∞ (instantaneous). */
const VINT_MAX = 20;
function vIntSlider(s: Settings, cb: ControlCallbacks): () => void {
  const input = $<HTMLInputElement>('k-vint');
  const out = $<HTMLOutputElement>('k-vint-out');
  const fmt = (x: number) => (Number.isFinite(x) ? (x < 1 ? x.toFixed(3) : x.toFixed(2)) : '∞ (instant)');
  const fromPos = (t: number) => (t > Math.log10(VINT_MAX) + 1e-9 ? Infinity : 10 ** t);
  const sync = () => {
    input.value = String(Number.isFinite(s.kernelVInt) ? Math.log10(s.kernelVInt) : Number(input.max));
    out.textContent = fmt(s.kernelVInt);
  };
  input.addEventListener('input', () => {
    s.kernelVInt = fromPos(Number(input.value));
    sync();
    cb.onChange('kernelVInt', s);
  });
  sync();
  return sync;
}

let delayInfo: DelayInfo | null = null;
let lastSettings: Settings | null = null;

export function bindControls(s: Settings, cb: ControlCallbacks): { sync(): void; setDelayInfo(info: DelayInfo | null): void } {
  lastSettings = s;
  const syncs: (() => void)[] = [];
  syncs.push(logSlider('viscosity', 'viscosity', s, cb));
  syncs.push(logSlider('diffusion', 'diffusion', s, cb));
  syncs.push(linSlider('vorticity', 'vorticity', s, cb, (x) => x.toFixed(1)));
  syncs.push(linSlider('fade', 'dyeDissipation', s, cb, (x) => x.toFixed(2)));
  syncs.push(linSlider('dt', 'dt', s, cb, (x) => x.toFixed(3)));
  syncs.push(linSlider('iterations', 'iterations', s, cb, (x) => String(x)));
  // rebuilding the grid is expensive: apply on release
  syncs.push(linSlider('resolution', 'resolution', s, cb, (x) => String(x), 'change'));
  syncs.push(linSlider('radius', 'radius', s, cb, (x) => x.toFixed(1)));
  syncs.push(linSlider('force', 'force', s, cb, (x) => x.toFixed(1)));
  syncs.push(linSlider('sp-tau', 'selfPropTau', s, cb, (x) => x.toFixed(2)));
  syncs.push(linSlider('sp-v0', 'selfPropV0', s, cb, (x) => x.toFixed(2)));

  syncs.push(logSlider('k-strength', 'kernelStrength', s, cb));
  syncs.push(linSlider('k-range', 'kernelRange', s, cb, (x) => x.toFixed(3)));
  syncs.push(linSlider('k-beta', 'kernelBeta', s, cb, (x) => x.toFixed(2)));
  syncs.push(linSlider('k-alpha', 'kernelAlpha', s, cb, (x) => x.toFixed(2)));
  syncs.push(linSlider('k-s', 'kernelS', s, cb, (x) => x.toFixed(2)));
  syncs.push(vIntSlider(s, cb));
  syncs.push(linSlider('k-shells', 'kernelShells', s, cb, (x) => String(x)));
  const kernelDelay = $<HTMLInputElement>('k-delay');
  kernelDelay.addEventListener('change', () => {
    s.kernelDelay = kernelDelay.checked;
    sync();
    cb.onChange('kernelDelay', s);
  });
  $<HTMLButtonElement>('delayed-flocking').addEventListener('click', () => {
    cb.onDelayedFlocking();
    sync();
  });
  const kernel = $<HTMLSelectElement>('kernel');
  kernel.addEventListener('change', () => {
    s.kernel = kernel.value as Settings['kernel'];
    sync();
    cb.onChange('kernel', s);
  });
  const kernelMT = $<HTMLInputElement>('k-mt');
  kernelMT.addEventListener('change', () => {
    s.kernelMT = kernelMT.checked;
    cb.onChange('kernelMT', s);
  });
  for (const id of ['k-strength', 'k-range', 'k-beta', 'k-alpha', 'k-s'])
    $<HTMLInputElement>(id).addEventListener('input', () => drawKernel(s));
  $<HTMLButtonElement>('flocking').addEventListener('click', () => {
    cb.onFlocking();
    sync();
  });

  const selfProp = $<HTMLInputElement>('selfprop');
  selfProp.addEventListener('change', () => {
    s.selfPropulsion = selfProp.checked;
    cb.onChange('selfPropulsion', s);
  });
  $<HTMLButtonElement>('active-turb').addEventListener('click', () => {
    cb.onActiveTurbulence();
    sync();
  });

  const walls = $<HTMLSelectElement>('walls');
  walls.addEventListener('change', () => {
    s.walls = walls.value as Settings['walls'];
    cb.onChange('walls', s);
  });

  const view = $<HTMLSelectElement>('view');
  view.addEventListener('change', () => {
    s.view = view.value as ViewMode;
    cb.onChange('view', s);
  });

  const arrows = $<HTMLInputElement>('arrows');
  arrows.addEventListener('change', () => {
    s.arrows = arrows.checked;
    cb.onChange('arrows', s);
  });

  const color = $<HTMLInputElement>('color');
  color.addEventListener('input', () => {
    s.color = color.value;
    s.rainbow = false;
    sync();
    cb.onChange('color', s);
  });
  const rainbow = $<HTMLInputElement>('rainbow');
  rainbow.addEventListener('change', () => {
    s.rainbow = rainbow.checked;
    cb.onChange('rainbow', s);
  });

  const play = $<HTMLButtonElement>('play');
  play.addEventListener('click', () => {
    s.paused = !s.paused;
    sync();
    cb.onChange('paused', s);
  });
  $<HTMLButtonElement>('step').addEventListener('click', () => cb.onStep());
  $<HTMLButtonElement>('reset').addEventListener('click', () => cb.onReset());

  const panel = $<HTMLElement>('panel');
  const toggle = $<HTMLButtonElement>('panel-toggle');
  const togglePanel = () => {
    const hidden = panel.classList.toggle('hidden');
    toggle.setAttribute('aria-expanded', String(!hidden));
  };
  toggle.addEventListener('click', togglePanel);

  function sync(): void {
    for (const f of syncs) f();
    walls.value = s.walls;
    view.value = s.view;
    arrows.checked = s.arrows;
    color.value = s.color;
    rainbow.checked = s.rainbow;
    selfProp.checked = s.selfPropulsion;
    kernel.value = s.kernel;
    kernelMT.checked = s.kernelMT;
    kernelDelay.checked = s.kernelDelay;
    updateKernelUI(s);
    play.textContent = s.paused ? '▶ Play' : '⏸ Pause';
  }
  sync();

  window.addEventListener('keydown', (e) => {
    const target = e.target as HTMLElement | null;
    if (target && (target.tagName === 'INPUT' || target.tagName === 'SELECT')) return;
    if (e.ctrlKey || e.metaKey || e.altKey) return;
    switch (e.key.toLowerCase()) {
      case ' ':
        e.preventDefault();
        play.click();
        break;
      case 's':
        cb.onStep();
        break;
      case 'r':
        cb.onReset();
        break;
      case 'v': {
        const next = VIEW_MODES[(VIEW_MODES.indexOf(s.view) + 1) % VIEW_MODES.length];
        s.view = next;
        sync();
        cb.onChange('view', s);
        break;
      }
      case 'a':
        s.arrows = !s.arrows;
        sync();
        cb.onChange('arrows', s);
        break;
      case 'h':
        togglePanel();
        break;
      case 'b': {
        s.walls = WALL_TYPES[(WALL_TYPES.indexOf(s.walls) + 1) % WALL_TYPES.length];
        sync();
        cb.onChange('walls', s);
        break;
      }
      case 'p':
        s.selfPropulsion = !s.selfPropulsion;
        sync();
        cb.onChange('selfPropulsion', s);
        break;
      case 't':
        cb.onActiveTurbulence();
        sync();
        break;
      case 'k':
        s.kernel = KERNEL_TYPES[(KERNEL_TYPES.indexOf(s.kernel) + 1) % KERNEL_TYPES.length];
        sync();
        cb.onChange('kernel', s);
        break;
      case 'f':
        cb.onFlocking();
        sync();
        break;
      case 'd':
        s.kernelDelay = !s.kernelDelay;
        sync();
        cb.onChange('kernelDelay', s);
        break;
      case 'l':
        cb.onDelayedFlocking();
        sync();
        break;
    }
  });

  return {
    sync,
    setDelayInfo(info) {
      delayInfo = info;
      updateDelayText(s);
      drawKernel(s);
    },
  };
}

/** Show only the sliders relevant to the selected kernel and redraw the plot. */
function updateKernelUI(s: Settings): void {
  const k = s.kernel;
  const show = (id: string, on: boolean) => { $<HTMLElement>(id).hidden = !on; };
  const conv = k === 'gaussian' || k === 'cucker-smale' || k === 'power';
  show('k-strength-row', k !== 'none');
  show('k-range-row', conv);
  show('k-beta-row', k === 'cucker-smale');
  show('k-alpha-row', k === 'power');
  show('k-s-row', k === 'fractional');
  show('k-mt-row', conv);
  show('k-plot', k !== 'none');
  show('k-delay-row', conv);
  show('k-vint-row', conv && s.kernelDelay);
  show('k-shells-row', conv && s.kernelDelay);
  show('k-delay-info', conv && s.kernelDelay);
  updateDelayText(s);
  $<HTMLElement>('k-strength-label').textContent = k === 'fractional' ? 'Strength ν_s (L^{2s}/s)' : 'Strength S (1/s)';
  const note = $<HTMLElement>('k-note');
  note.textContent = k === 'fractional' && s.walls !== 'periodic'
    ? '⚠ The fractional Laplacian is only implemented on fully periodic boundaries; it is inactive with walls.'
    : '';
  drawKernel(s);
}

/** Plot K(r) normalised to K(0) = 1 for r ∈ [0, 0.5] (or the multiplier |k|^{2s} for fractional). */
function drawKernel(s: Settings): void {
  const cv = $<HTMLCanvasElement>('k-plot');
  const ctx = cv.getContext('2d');
  if (!ctx) return;
  const W = cv.width, H = cv.height;
  ctx.clearRect(0, 0, W, H);
  ctx.strokeStyle = 'rgba(255,255,255,0.25)';
  ctx.strokeRect(0.5, 0.5, W - 1, H - 1);
  const p = {
    kernel: s.kernel,
    range: Number(($<HTMLInputElement>('k-range')).value) || s.kernelRange,
    beta: Number(($<HTMLInputElement>('k-beta')).value) || s.kernelBeta,
    alpha: Number(($<HTMLInputElement>('k-alpha')).value) || s.kernelAlpha,
  };
  const sFrac = Number(($<HTMLInputElement>('k-s')).value) || s.kernelS;
  const pad = 4;
  const conv = s.kernel === 'gaussian' || s.kernel === 'cucker-smale' || s.kernel === 'power';
  const di = conv && s.kernelDelay ? delayInfo : null;
  // r axis: 0 … 0.5, or out to the tail radius when the delay shells are shown
  const rMax = di ? Math.min(Math.max(0.5, di.rTail), 1.5) : 0.5;
  const X = (r: number) => pad + (r / rMax) * (W - 2 * pad);
  if (di) {
    // shade the delay shells alternately; orange staircase = delay τ(r) = r_out / v_int
    let r0 = 0;
    di.rOuter.forEach((r1, m) => {
      if (!(di.mass[m] > 0)) { r0 = Math.max(r0, r1); return; }
      ctx.fillStyle = m === di.rOuter.length - 1 ? 'rgba(255,255,255,0.03)' : m % 2 ? 'rgba(63,169,255,0.10)' : 'rgba(63,169,255,0.20)';
      ctx.fillRect(X(r0), pad, X(r1) - X(r0), H - 2 * pad);
      r0 = r1;
    });
    if (Number.isFinite(di.tauMax) && di.tauMax > 0) {
      ctx.beginPath();
      ctx.strokeStyle = 'rgba(255,160,60,0.9)';
      ctx.lineWidth = 1;
      let prev = 0;
      ctx.moveTo(X(0), H - pad);
      di.rOuter.forEach((r1, m) => {
        if (!(di.mass[m] > 0)) return;
        const y = H - pad - (r1 / Math.max(...di.rOuter.filter((_, q) => di.mass[q] > 0))) * (H - 2 * pad);
        ctx.lineTo(X(prev), y);
        ctx.lineTo(X(r1), y);
        prev = r1;
      });
      ctx.stroke();
    }
  }
  ctx.beginPath();
  ctx.strokeStyle = '#3fa9ff';
  ctx.lineWidth = 1.5;
  for (let x = 0; x <= W - 2 * pad; x++) {
    const t = x / (W - 2 * pad);
    let y: number;
    if (s.kernel === 'fractional') y = t ** (2 * sFrac); // multiplier vs |k|/k_max
    else y = kernelProfile(p, rMax * t) / kernelProfile(p, 0);
    const px = pad + x, py = H - pad - y * (H - 2 * pad);
    if (x === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
  }
  ctx.stroke();
  ctx.fillStyle = 'rgba(255,255,255,0.6)';
  ctx.font = '10px ui-monospace, monospace';
  ctx.fillText(s.kernel === 'fractional' ? '|k|^2s / |k|max^2s  vs |k|'
    : `K(r)/K(0),  r = 0 … ${rMax.toFixed(2)}`, pad + 2, 12);
  if (di && Number.isFinite(di.tauMax)) {
    ctx.fillStyle = 'rgba(255,170,80,0.9)';
    ctx.fillText(`delay r_out/v_int (max ${di.tauMax.toFixed(2)} s)`, pad + 2, 24);
  }
  if (s.kernel !== 'fractional') {
    // mark R
    const xr = X(p.range);
    ctx.strokeStyle = 'rgba(255,200,80,0.6)';
    ctx.beginPath(); ctx.moveTo(xr, pad); ctx.lineTo(xr, H - pad); ctx.stroke();
    ctx.fillText('R', xr + 2, H - pad - 2);
  }
}

/** Delay / memory summary under the kernel plot. */
function updateDelayText(s: Settings): void {
  const el = $<HTMLElement>('k-delay-info');
  const di = delayInfo;
  if (!s.kernelDelay) { el.textContent = ''; return; }
  if (!Number.isFinite(s.kernelVInt)) {
    el.textContent = 'v_int = ∞: instantaneous kernel (exact update, no history).';
    return;
  }
  if (!di) { el.textContent = ''; return; }
  const ms = (x: number) => (x * 1000).toFixed(1);
  el.textContent =
    `t_del,max = r_tail/v_int = ${di.tauMax.toFixed(2)} s · R_cut(99.5%) = ${di.rCut.toFixed(3)}, ` +
    `r_tail = ${di.rTail.toFixed(3)} · history ${di.framesStored}/${di.capacity} frames ` +
    `(every ${ms(di.interval)} ms${di.interval > (lastSettings?.dt ?? 0) * 1.0001 ? ', memory-capped' : ''}) · ` +
    `${(di.bytes / 2 ** 20).toFixed(1)} MB`;
}
