import type { KernelType, WallType } from '../solver';
import type { ViewMode } from '../render/renderer';

/** All user-adjustable settings of the app. */
export interface Settings {
  viscosity: number;
  diffusion: number;
  vorticity: number;
  dyeDissipation: number;
  walls: WallType;
  /** Self-propulsion (active fluid) term −(1/τ)(|u| − v₀) u/|u| on/off. */
  selfPropulsion: boolean;
  /** Relaxation time τ (s). */
  selfPropTau: number;
  /** Target speed v₀ (domain units / s; the shorter side has length 1). */
  selfPropV0: number;
  /** Long-range viscosity / alignment kernel (added to the local ν∇²). */
  kernel: KernelType;
  /** Kernel strength S (1/s); ν_s for the fractional Laplacian. */
  kernelStrength: number;
  kernelRange: number;
  kernelBeta: number;
  kernelAlpha: number;
  kernelS: number;
  kernelMT: boolean;
  /** Delay of the point's own velocity t_self (s) in the delayed viscosity terms. */
  delaySelf: number;
  /** Delay of the neighbours' velocities t_nb (s). */
  delayNb: number;
  /** Apply the delays to the local viscosity ν∇²u. */
  delayLocal: boolean;
  /** Apply the delays to the long-range kernel / fractional term. */
  delayKernel: boolean;
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
  selfPropulsion: false,
  selfPropTau: 0.5,
  selfPropV0: 0.5,
  kernel: 'none',
  kernelStrength: 2,
  kernelRange: 0.05,
  kernelBeta: 1,
  kernelAlpha: 3,
  kernelS: 0.5,
  kernelMT: false,
  delaySelf: 0,
  delayNb: 0,
  delayLocal: true,
  delayKernel: true,
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
