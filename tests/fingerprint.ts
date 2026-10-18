/**
 * Deterministic fingerprint of a full solver run, used to prove that the
 * solid-wall code paths stay bit-identical when new boundary options are added.
 */
import { FluidSolver, type WallType } from '../src/solver';

/** FNV-1a over the raw bytes of the given Float32Arrays. */
export function hashFields(...fields: Float32Array[]): string {
  let h = 0x811c9dc5;
  for (const f of fields) {
    const b = new Uint8Array(f.buffer, f.byteOffset, f.byteLength);
    for (let k = 0; k < b.length; k++) {
      h ^= b[k];
      h = Math.imul(h, 0x01000193) >>> 0;
    }
  }
  return h.toString(16).padStart(8, '0');
}

/** Run a scripted scenario touching every operator and return its fingerprint. */
export function wallScenario(walls: WallType): string {
  const s = new FluidSolver(40, 28, {
    walls, viscosity: 2e-4, diffusion: 1e-5, vorticity: 3, dyeDissipation: 0.1, iterations: 25,
  });
  s.seedActiveTurbulence(7, 0.08);
  for (let n = 0; n < 60; n++) {
    const t = n / 60;
    s.stir(2 + 36 * t, 1 + 26 * t, 1.5, -0.7, 3, 0.6);
    s.splat(0.3 + 39 * t, 14, 0, 0, [0.3, 0.6, 0.9], 3);
    s.splat(20, 27.8, 0.4, -0.9, null, 2);
    if (n === 30) s.setParams({ selfPropulsion: true, selfPropTau: 0.3, selfPropV0: 0.6 });
    s.step();
  }
  s.updateDiagnostics();
  const r = new FluidSolver(30, 20, { walls });
  r.resampleFrom(s);
  return hashFields(s.u, s.v, s.pressure, ...s.dye, s.uc, s.vc, s.divergence, s.vorticity, r.u, r.v, ...r.dye);
}
