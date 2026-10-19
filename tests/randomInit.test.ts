import { describe, expect, it } from 'vitest';
import { FluidSolver } from '../src/solver';
import { freshSeed, orderedInit, parseSeed, randomInit } from '../src/ui/randomInit';

const P = { walls: 'periodic' as const, selfPropulsion: true, kernel: 'cucker-smale' as const, delaySelf: 0.3, delayNb: 0.3 };

describe('random initial conditions', () => {
  it('parses seeds', () => {
    expect(parseSeed(' 1017 ')).toBe(1017);
    expect(parseSeed('')).toBeNull();
    expect(parseSeed('abc')).toBeNull();
    expect(parseSeed('99999999999')).toBeNull();
  });
  it('fresh seeds are 32-bit and vary', () => {
    const a = new Set(Array.from({ length: 20 }, freshSeed));
    expect(a.size).toBeGreaterThan(15);
    for (const x of a) expect(x).toBeGreaterThanOrEqual(0), expect(x).toBeLessThan(2 ** 32);
  });
  it('reproduces the sweep seeding exactly and resets history', () => {
    const ref = new FluidSolver(32, 18, P);
    ref.seedActiveTurbulence(1017, 0.05);
    const s = new FluidSolver(32, 18, P);
    s.seedActiveTurbulence(5, 0.05);
    for (let i = 0; i < 30; i++) s.step();
    expect(randomInit(s, 1017, 0.05)).toBe(1017);
    expect(s.time).toBe(0);
    expect(Array.from(s.u)).toEqual(Array.from(ref.u));
    for (let i = 0; i < 10; i++) { s.step(); ref.step(); }
    expect(Array.from(s.u)).toEqual(Array.from(ref.u));
  });
  it('amplitude scales the field', () => {
    const a = new FluidSolver(32, 18, P), b = new FluidSolver(32, 18, P);
    randomInit(a, 7, 0.05); randomInit(b, 7, 0.1);
    const m = (x: Float32Array) => Math.max(...Array.from(x, Math.abs));
    expect(m(b.u) / m(a.u)).toBeCloseTo(2, 3);
  });
  it('ordered + perturbation: ⟨cos⟩ ≈ 0.95, reproducible', () => {
    const P2 = { walls: 'periodic' as const, selfPropulsion: true, selfPropV0: 0.5 };
    const vals: number[] = [];
    for (let k = 0; k < 20; k++) {
      const s = new FluidSolver(128, 72, P2); orderedInit(s, 100 + k);
      s.updateDiagnostics(); const g = s.grid;
      let ux = 0, vy = 0, N = 0;
      for (let j = 1; j <= g.ny; j++) for (let i = 1; i <= g.nx; i++) {
        const q = i + g.stride * j, sp = Math.hypot(s.uc[q], s.vc[q]); ux += s.uc[q] / sp; vy += s.vc[q] / sp; N++;
      }
      vals.push((ux * ux + vy * vy - N) / (N * (N - 1)));
    }
    const m = vals.reduce((a, b) => a + b) / vals.length;
    expect(m).toBeGreaterThan(0.94); expect(m).toBeLessThan(0.96);
    const a = new FluidSolver(64, 36, P2), b = new FluidSolver(64, 36, P2);
    orderedInit(a, 9); orderedInit(b, 9);
    expect(Array.from(a.u)).toEqual(Array.from(b.u));
  });
});
