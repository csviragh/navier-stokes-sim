/**
 * Binds the HTML control panel to a Settings object. Changes are reported via
 * the onChange callback with the name of the changed key.
 */
import { VIEW_MODES, type ViewMode } from '../render/renderer';
import type { Settings } from './settings';

type Key = keyof Settings;

export interface ControlCallbacks {
  onChange(key: Key, settings: Settings): void;
  onStep(): void;
  onReset(): void;
}

const $ = <T extends HTMLElement>(id: string): T => {
  const el = document.getElementById(id);
  if (!el) throw new Error(`Missing element #${id}`);
  return el as T;
};

/** Log-scale slider: slider value is log10(x); the minimum position means 0. */
function logSlider(
  id: string, key: 'viscosity' | 'diffusion', s: Settings, cb: ControlCallbacks,
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
  key: 'vorticity' | 'dyeDissipation' | 'dt' | 'iterations' | 'resolution' | 'radius' | 'force',
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

export function bindControls(s: Settings, cb: ControlCallbacks): { sync(): void } {
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
    }
  });

  return { sync };
}
