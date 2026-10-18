/**
 * Binds the HTML control panel to a Settings object. Changes are reported via
 * the onChange callback with the name of the changed key.
 */
import { VIEW_MODES, type ViewMode } from '../render/renderer';
import { KERNEL_TYPES, kernelProfile, type DelayInfo, type DelayStability, type ModeScan } from '../solver';
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
  /** Quick start: long-range alignment with t_self = t_nb beyond the oscillation threshold. */
  onDelayOscillations(): void;
  /** Quick start: delayed local viscosity only (own vs neighbour delay, large-scale oscillation). */
  onLocalDelay(): void;
  /** Quick start: long-range alignment with only the own velocity delayed. */
  onOwnDelay(): void;
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
    | 'selfPropTau' | 'selfPropV0' | 'kernelRange' | 'kernelBeta' | 'kernelAlpha' | 'kernelS'
    | 'delaySelf' | 'delayNb',
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

let delayInfo: DelayInfo | null = null;
let delayStab: DelayStability | null = null;

export function bindControls(
  s: Settings, cb: ControlCallbacks,
): { sync(): void; setDelayInfo(info: DelayInfo | null, stability: DelayStability | null): void } {
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
  const fmtDelay = (x: number) => (x > 0 ? `${x.toFixed(2)} s` : '0 (current)');
  syncs.push(linSlider('d-self', 'delaySelf', s, cb, fmtDelay));
  syncs.push(linSlider('d-nb', 'delayNb', s, cb, fmtDelay));
  const delayLocal = $<HTMLInputElement>('d-local');
  delayLocal.addEventListener('change', () => { s.delayLocal = delayLocal.checked; cb.onChange('delayLocal', s); });
  const delayMode = $<HTMLSelectElement>('d-mode');
  delayMode.addEventListener('change', () => { s.delayMode = delayMode.value as Settings['delayMode']; cb.onChange('delayMode', s); });
  const delayKernel = $<HTMLInputElement>('d-kernel');
  delayKernel.addEventListener('change', () => { s.delayKernel = delayKernel.checked; cb.onChange('delayKernel', s); });
  $<HTMLButtonElement>('delay-osc').addEventListener('click', () => { cb.onDelayOscillations(); sync(); });
  $<HTMLButtonElement>('local-delay').addEventListener('click', () => { cb.onLocalDelay(); sync(); });
  $<HTMLButtonElement>('own-delay').addEventListener('click', () => { cb.onOwnDelay(); sync(); });
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
    delayLocal.checked = s.delayLocal;
    delayKernel.checked = s.delayKernel;
    delayMode.value = s.delayMode;
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
      case 'l':
        cb.onLocalDelay();
        sync();
        break;
      case 'n':
        cb.onOwnDelay();
        sync();
        break;
      case 'o':
        cb.onDelayOscillations();
        sync();
        break;
    }
  });

  return {
    sync,
    setDelayInfo(info, stability) {
      delayInfo = info;
      delayStab = stability;
      updateDelayText(s);
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
  const rMax = 0.5;
  const X = (r: number) => pad + (r / rMax) * (W - 2 * pad);
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
  if (s.kernel !== 'fractional') {
    // mark R
    const xr = X(p.range);
    ctx.strokeStyle = 'rgba(255,200,80,0.6)';
    ctx.beginPath(); ctx.moveTo(xr, pad); ctx.lineTo(xr, H - pad); ctx.stroke();
    ctx.fillText('R', xr + 2, H - pad - 2);
  }
}

const REGIME_TEXT: Record<ModeScan['regime'], [string, string]> = {
  monotone: ['monotone decay', 'ok'],
  damped: ['damped oscillation', 'warn'],
  neutral: ['marginal (neutral)', 'warn'],
  'unstable-osc': ['UNSTABLE oscillation', 'bad'],
  'unstable-real': ['UNSTABLE monotone run-away', 'bad'],
};

/** "λ = −0.32 ± 1.34i /s" plus period / e-folding time. */
function rootText(m: ModeScan): string {
  const r = m.root;
  const lam = `λ = ${r.re >= 0 ? '+' : ''}${r.re.toFixed(3)}${r.im > 0 ? ` ± ${r.im.toFixed(2)}i` : ''} /s`;
  const extra = r.im > 0 ? `, period ${(2 * Math.PI / r.im).toFixed(2)} s` : '';
  return lam + extra;
}

function localModeName(c: number): string {
  if (c <= -0.995) return 'checkerboard (k = π/h)';
  if (c >= 0.98) return 'longest waves (k → 0)';
  return `k·h ≈ ${Math.acos(Math.max(-1, Math.min(1, c))).toFixed(2)} (λ_wave ≈ ${(2 * Math.PI / Math.acos(Math.max(-1, Math.min(1, c)))).toFixed(1)} cells)`;
}

function kernelModeName(q: number, fractional: boolean): string {
  if (fractional) return q >= 0.95 ? 'longest waves' : q <= 0 ? `short waves (B/A = ${q.toFixed(2)})` : `B/A = ${q.toFixed(2)}`;
  if (q <= 0.05) return 'short waves (K̂ ≈ 0)';
  if (q >= 0.95) return 'long waves (K̂ ≈ 1)';
  return `K̂ = ${q.toFixed(2)}`;
}

/** Regime readouts and history summary of the Delays section. */
function updateDelayText(s: Settings): void {
  const elL = $<HTMLElement>('d-regime-local');
  const elK = $<HTMLElement>('d-regime-kernel');
  const elI = $<HTMLElement>('d-info');
  const di = delayInfo, st = delayStab;
  const none = !(s.delaySelf > 0) && !(s.delayNb > 0);
  const set = (el: HTMLElement, txt: string, cls = '') => { el.textContent = txt; el.className = `hint mono${cls ? ` regime-${cls}` : ''}`; };
  if (none) {
    set(elL, 't_self = t_nb = 0: no delay — instantaneous terms (implicit / exact updates).');
    set(elK, ''); set(elI, '');
    return;
  }
  // local viscosity
  if (!s.delayLocal) set(elL, 'Local ν∇²: not delayed.');
  else if (!(s.viscosity > 0)) set(elL, 'Local ν∇²: ν = 0 (nothing to delay).');
  else if (st?.local) {
    const m = st.local;
    const [name, cls] = REGIME_TEXT[m.regime];
    let txt = `Local ν∇² (A = 4ν/h² = ${m.A.toPrecision(3)} /s): ${name} — most unstable ${localModeName(m.param)}, ${rootText(m)}`;
    if (di && di.localSubsteps > 1) txt += ` · νΔt/h² = ${di.localNumber.toFixed(2)} → ${di.localSubsteps} explicit sub-steps`;
    if (di && di.localNumber / di.localSubsteps > 0.25) txt += ' · ⚠ explicit limit νΔt/h² ≤ 1/4 exceeded even with sub-steps: lower Δt or ν';
    set(elL, txt, cls);
  } else set(elL, 'Local ν∇²: …');
  // long-range
  const frac = s.kernel === 'fractional';
  if (s.kernel === 'none') set(elK, 'Long-range: no kernel selected.');
  else if (!s.delayKernel) set(elK, 'Long-range: not delayed.');
  else if (frac && s.walls !== 'periodic') set(elK, 'Long-range: the fractional Laplacian is inactive with walls.');
  else if (st?.kernel) {
    const m = st.kernel;
    const [name, cls] = REGIME_TEXT[m.regime];
    let txt = `Long-range (A = ${frac ? 'ν_s G₀' : 'S'} = ${m.A.toPrecision(3)} /s): ${name} — most unstable ${kernelModeName(m.param, frac)}, ${rootText(m)}`;
    if (di && di.kernelNumber > 1) txt += ` · ⚠ ${frac ? 'ν_s G₀' : 'S'}·Δt = ${di.kernelNumber.toFixed(2)} > 1: explicit update inaccurate, lower Δt or the strength`;
    set(elK, txt, cls);
  } else set(elK, 'Long-range: …');
  if (di && di.tauMax > 0) {
    const ms = (x: number) => (x * 1000).toFixed(1);
    set(elI, `history ${di.framesStored}/${di.capacity} frames covering max(t_self, t_nb) = ${di.tauMax.toFixed(2)} s ` +
      `(every ${ms(di.interval)} ms${di.interval > s.dt * 1.0001 ? ', memory-capped' : ''}) · ${(di.bytes / 2 ** 20).toFixed(1)} MB · ` +
      (s.delayMode === 'lagrangian' ? 'read along the parcel paths' : 'read at fixed points (Eulerian)'));
  } else set(elI, '');
}
