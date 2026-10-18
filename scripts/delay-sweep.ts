/**
 * Delay sweep: Cucker–Smale alignment + self-propulsion on a torus with equal
 * delays t_self = t_nb = t, measuring the pairwise velocity-alignment
 * distribution P(cos θ_ij) and the exact ⟨cos θ_ij⟩.
 *
 *   npx esbuild scripts/delay-sweep.ts --bundle --platform=node --outfile=/tmp/delay-sweep.js
 *   node /tmp/delay-sweep.js --out /workspace/delay-sweep [--jobs 8] [--nx 128 --ny 72]
 *        [--tmax 120] [--seeds 3] [--dmax 2 --dstep 0.05] [--mode eulerian|lagrangian]
 *        [--nu 0.01 [--iters 40]]   (undelayed local viscosity ν; implicit Gauss–Seidel sweeps)
 *
 * Every (delay, seed) run is a child process (fork of this file) that writes
 * runs/d<t>_s<seed>.json; existing results are skipped (resumable). The parent
 * then writes samples.csv (one row per sample), summary.csv (per run) and
 * theory.json (linear-theory thresholds of the equal-delay kernel term).
 */
import { fork } from 'node:child_process';
import { existsSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { cpus } from 'node:os';
import { join } from 'node:path';
import { FluidSolver, OrderAnalyzer, type DelayMode, type FluidParams } from '../src/solver';

const arg = (name: string, def: string): string => {
  const i = process.argv.indexOf(`--${name}`);
  return i >= 0 && i + 1 < process.argv.length ? process.argv[i + 1] : def;
};
const OUT = arg('out', '/workspace/delay-sweep');
const NX = +arg('nx', '128'), NY = +arg('ny', '72');
const TMAX = +arg('tmax', '120');
const SAMPLE = +arg('sample', '0.5');
const SEEDS = +arg('seeds', '3');
const DMAX = +arg('dmax', '2'), DSTEP = +arg('dstep', '0.05');
const JOBS = +arg('jobs', String(Math.max(1, cpus().length)));
const MODE = arg('mode', 'eulerian') as DelayMode; // where the delayed terms read: fixed point or along the path
if (MODE !== 'eulerian' && MODE !== 'lagrangian') throw new Error(`--mode must be eulerian or lagrangian (got ${MODE})`);
const NU = +arg('nu', '0');          // local viscosity ν (NOT delayed: delayLocal = false)
const ITERS = +arg('iters', '20');   // Gauss–Seidel sweeps of the implicit viscosity (20 = solver default)
const BINS = 40, PAIRS = 30000;

/** Physics of the sweep (everything else = solver defaults). */
export const SWEEP_PARAMS: Partial<FluidParams> = {
  walls: 'periodic',
  viscosity: NU,             // local viscosity (default 0); applied instantaneously, never delayed
  iterations: ITERS,
  diffusion: 0,
  vorticity: 0,              // no vorticity confinement
  selfPropulsion: true,
  selfPropTau: 1,            // relaxation time τ = 1 s
  selfPropV0: 0.5,           // default target speed v₀
  kernel: 'cucker-smale',
  kernelStrength: 2,         // default S = 2 /s
  kernelRange: 0.05,         // default R
  kernelBeta: 1,             // default β
  kernelMT: false,
  delayLocal: false,         // the local viscosity is never delayed
  delayKernel: true,
  delayMode: MODE,
  dt: 1 / 60,
};
const SEED_AMPLITUDE = 0.05; // weak random swirl field, |u| ≈ 0.05
const seedOf = (k: number) => 1000 + 17 * k; // seed index k = 0, 1, 2, … (same for every delay)

interface RunResult {
  delay: number; seed: number; nx: number; ny: number; steps: number; wallSeconds: number;
  t: number[]; meanCos: number[]; phi: number[]; speed: number[]; xi: number[]; hist: number[][];
  maxAbsU: number; finite: boolean;
}

function runOne(delay: number, seed: number): RunResult {
  const s = new FluidSolver(NX, NY, { ...SWEEP_PARAMS, delaySelf: delay, delayNb: delay });
  s.seedActiveTurbulence(seed, SEED_AMPLITUDE);
  const g = s.grid, an = new OrderAnalyzer();
  const steps = Math.round(TMAX / SWEEP_PARAMS.dt!), every = Math.round(SAMPLE / SWEEP_PARAMS.dt!);
  const r: RunResult = { delay, seed, nx: NX, ny: NY, steps, wallSeconds: 0, t: [], meanCos: [], phi: [], speed: [], xi: [], hist: [], maxAbsU: 0, finite: true };
  const t0 = Date.now();
  for (let n = 1; n <= steps; n++) {
    s.step();
    if (n % every) continue;
    s.updateDiagnostics();
    const st = an.analyzeCells(g.nx, g.ny, g.stride, s.uc, s.vc, { bins: BINS, pairs: PAIRS, periodicX: true, periodicY: true, seed: seed * 100003 + n });
    let mu = 0, mv = 0, sp = 0;
    for (let j = 1; j <= g.ny; j++) for (let i = 1; i <= g.nx; i++) {
      const k = i + g.stride * j; mu += s.uc[k]; mv += s.vc[k]; sp += Math.hypot(s.uc[k], s.vc[k]);
    }
    r.t.push(+(n * SWEEP_PARAMS.dt!).toFixed(4));
    r.meanCos.push(st.meanCos); r.phi.push(sp > 0 ? Math.hypot(mu, mv) / sp : 0); r.speed.push(sp / (g.nx * g.ny));
    r.xi.push(st.xi0 * g.h);
    r.hist.push(Array.from(st.hist, (x) => +x.toFixed(5)));
  }
  for (const a of [s.u, s.v]) for (const x of a) { if (!Number.isFinite(x)) r.finite = false; else r.maxAbsU = Math.max(r.maxAbsU, Math.abs(x)); }
  r.wallSeconds = (Date.now() - t0) / 1000;
  return r;
}

/** Linear theory of the delayed kernel term for equal delays: λ = −S(1 − K̂) e^{−λ t}. */
function theory() {
  const s = new FluidSolver(NX, NY, { ...SWEEP_PARAMS });
  const [kLo, kHi] = s.kernelOperators().u.modeRatioRange();
  const S = SWEEP_PARAMS.kernelStrength!, aMax = S * (1 - kLo), aMin = S * (1 - kHi);
  return {
    S, Khat_min: kLo, Khat_max_nonzero_mode: kHi, a_max: aMax, a_min_nonzero: aMin,
    t_ringing: 1 / (Math.E * aMax),            // first mode turns oscillatory (a t = 1/e)
    t_unstable: Math.PI / (2 * aMax),          // first mode grows (a t = π/2), shortest waves
    t_unstable_longest_wave: Math.PI / (2 * aMin),
    t_ringing_S: 1 / (Math.E * S), t_unstable_S: Math.PI / (2 * S),
    note: 'per Fourier mode a = S(1 − K̂(k)); the k = 0 mean flow is neutral; self-propulsion and advection are not in the linear theory',
  };
}

const fmt = (d: number) => d.toFixed(2);

async function main() {
  if (process.argv.includes('--child')) {
    const d = +arg('delay', '0'), sd = +arg('seed', '1');
    const r = runOne(d, sd);
    writeFileSync(join(OUT, 'runs', `d${fmt(d)}_s${sd}.json`), JSON.stringify(r));
    return;
  }
  mkdirSync(join(OUT, 'runs'), { recursive: true });
  const delays: number[] = [];
  for (let k = 0; k * DSTEP <= DMAX + 1e-9; k++) delays.push(+(k * DSTEP).toFixed(4));
  const jobs: [number, number][] = [];
  // interleave seeds so partial results already cover the whole delay range
  for (let k = 0; k < SEEDS; k++) for (const d of delays) jobs.push([d, seedOf(k)]);
  const todo = jobs.filter(([d, sd]) => !existsSync(join(OUT, 'runs', `d${fmt(d)}_s${sd}.json`)));
  const th = theory();
  writeFileSync(join(OUT, 'theory.json'), JSON.stringify({ ...th, params: SWEEP_PARAMS, nx: NX, ny: NY, tmax: TMAX, sample: SAMPLE, seeds: SEEDS, delays, seedAmplitude: SEED_AMPLITUDE, pairs: PAIRS, bins: BINS }, null, 1));
  console.log(`${jobs.length} runs (${todo.length} to do) on ${JOBS} processes, grid ${NX}×${NY}, ${TMAX} s each, ${MODE} delays`);
  const T0 = Date.now();
  let next = 0, done = 0;
  await new Promise<void>((resolve) => {
    const launch = () => {
      if (next >= todo.length) { if (done === todo.length) resolve(); return; }
      const [d, sd] = todo[next++];
      const c = fork(process.argv[1], ['--child', '--delay', String(d), '--seed', String(sd), '--out', OUT, '--nx', String(NX), '--ny', String(NY), '--tmax', String(TMAX), '--sample', String(SAMPLE), '--mode', MODE, '--nu', String(NU), '--iters', String(ITERS)]);
      c.on('exit', (code) => {
        done++;
        console.log(`[${((Date.now() - T0) / 60000).toFixed(1)} min] ${done}/${todo.length} delay ${fmt(d)} seed ${sd} exit ${code}`);
        if (done === todo.length) resolve(); else launch();
      });
    };
    for (let k = 0; k < Math.min(JOBS, todo.length); k++) launch();
    if (!todo.length) resolve();
  });
  // CSVs
  const head = ['delay', 'seed', 't', 'meanCos', 'phi', 'meanSpeed', 'xi', ...Array.from({ length: BINS }, (_, k) => `P${k}`)];
  const rows = [head.join(',')], sum = ['delay,seed,meanCos_full,meanCos_steady,meanCos_steady_std,phi_full,phi_steady,speed_steady,xi_steady,maxAbsU,finite,wall_s'];
  const mean = (a: number[]) => a.reduce((x, y) => x + y, 0) / a.length;
  const std = (a: number[]) => { const m = mean(a); return Math.sqrt(a.reduce((x, y) => x + (y - m) ** 2, 0) / Math.max(1, a.length - 1)); };
  for (const [d, sd] of jobs) {
    const f = join(OUT, 'runs', `d${fmt(d)}_s${sd}.json`);
    if (!existsSync(f)) continue;
    const r: RunResult = JSON.parse(readFileSync(f, 'utf8'));
    r.t.forEach((t, k) => rows.push([d, sd, t, r.meanCos[k].toFixed(6), r.phi[k].toFixed(6), r.speed[k].toFixed(6), Number.isFinite(r.xi[k]) ? r.xi[k].toFixed(5) : '', ...r.hist[k]].join(',')));
    const st = r.t.map((t, k) => (t > TMAX / 2 ? k : -1)).filter((k) => k >= 0);
    const pick = (a: number[]) => st.map((k) => a[k]);
    const xs = pick(r.xi).filter(Number.isFinite);
    sum.push([d, sd, mean(r.meanCos).toFixed(6), mean(pick(r.meanCos)).toFixed(6), std(pick(r.meanCos)).toFixed(6), mean(r.phi).toFixed(6), mean(pick(r.phi)).toFixed(6),
      mean(pick(r.speed)).toFixed(6), xs.length ? mean(xs).toFixed(5) : '', r.maxAbsU.toFixed(4), r.finite, r.wallSeconds.toFixed(1)].join(','));
  }
  writeFileSync(join(OUT, 'samples.csv'), rows.join('\n') + '\n');
  writeFileSync(join(OUT, 'summary.csv'), sum.join('\n') + '\n');
  console.log(`done in ${((Date.now() - T0) / 60000).toFixed(1)} min`);
}
main();
