/** Dump the Cucker–Smale Fourier symbols K̂ used by scripts/delay-sweep-theory.py (S = 2, R = 0.05, β = 1). */
import { writeFileSync } from 'node:fs';
import { FluidSolver } from '../src/solver';
const out: any = {};
// usage: kernel-symbols out.json [NXxNY ...]   (default grids 128x72 219x128)
const grids = process.argv.length > 3 ? process.argv.slice(3).map((g) => g.split('x').map(Number)) : [[128, 72], [219, 128]];
for (const [nx, ny] of grids) {
  const s = new FluidSolver(nx, ny, { walls: 'periodic', kernel: 'cucker-smale', kernelStrength: 2, kernelRange: 0.05, kernelBeta: 1 });
  const op = s.kernelOperators().u; const sym: number[] = [];
  for (let b = 0; b < ny; b++) for (let a = 0; a < nx; a++) sym.push(op.symbol(a, b));
  out[`${nx}x${ny}`] = { nx, ny, h: s.grid.h, sym };
}
writeFileSync(process.argv[2] ?? 'kernel_symbols.json', JSON.stringify(out));
