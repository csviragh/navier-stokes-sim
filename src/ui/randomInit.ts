/** Helpers for the "Random initial conditions" button (same generator as scripts/delay-sweep.ts). */
import type { FluidSolver } from '../solver';

export const DEFAULT_SEED_AMPLITUDE = 0.05;

/** Fresh 32-bit seed from crypto.getRandomValues (Math.random fallback). */
export function freshSeed(): number {
  const c = (globalThis as { crypto?: Crypto }).crypto;
  if (c?.getRandomValues) return c.getRandomValues(new Uint32Array(1))[0] >>> 0;
  return (Math.random() * 2 ** 32) >>> 0;
}

/** Parse a typed seed; blank or invalid → null (use a fresh one). */
export function parseSeed(text: string): number | null {
  const t = text.trim();
  if (!/^\d+$/.test(t)) return null;
  const n = Number(t);
  return Number.isSafeInteger(n) && n <= 0xffffffff ? n >>> 0 : null;
}

/** Reset the solver (clears the delay history) and seed it exactly like the sweep. */
export function randomInit(s: FluidSolver, seed: number, amplitude = DEFAULT_SEED_AMPLITUDE): number {
  s.reset();
  s.seedActiveTurbulence(seed >>> 0, amplitude);
  return seed >>> 0;
}
