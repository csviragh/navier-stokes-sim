/** Dump the Cucker–Smale Fourier symbols K̂ used by scripts/delay-sweep-theory.py (S = 2, R = 0.05, β = 1). */
import { writeFileSync } from 'node:fs';
import { FluidSolver } from '../src/solver';
const out: any = {};
for (const [nx, ny] of [[128, 72], [219, 128]]) {
  const s = new FluidSolver(nx, ny, { walls: 'periodic', kernel: 'cucker-smale', kernelStrength: 2, kernelRange: 0.05, kernelBeta: 1 });
  const op = s.kernelOperators().u; const sym: number[] = [];
  for (let b = 0; b < ny; b++) for (let a = 0; a < nx; a++) sym.push(op.symbol(a, b));
  out[`${nx}x${ny}`] = { nx, ny, h: s.grid.h, sym };
}
writeFileSync('process.argv[2] ?? "kernel_symbols.json"', JSON.stringify(out));
