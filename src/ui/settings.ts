import type { WallType } from '../solver';
import type { ViewMode } from '../render/renderer';

/** All user-adjustable settings of the app. */
export interface Settings {
  viscosity: number;
  diffusion: number;
  vorticity: number;
  dyeDissipation: number;
  walls: WallType;
  dt: number;
  iterations: number;
  /** Number of cells along the shorter side of the window. */
  resolution: number;
  view: ViewMode;
  arrows: boolean;
  color: string;
  rainbow: boolean;
  radius: number;
  force: number;
  paused: boolean;
}

export const defaultSettings: Settings = {
  viscosity: 1e-5,
  diffusion: 0,
  vorticity: 2,
  dyeDissipation: 0.15,
  walls: 'no-slip',
  dt: 1 / 60,
  iterations: 20,
  resolution: 128,
  view: 'dye',
  arrows: false,
  color: '#3fa9ff',
  rainbow: true,
  radius: 4,
  force: 1,
  paused: false,
};
