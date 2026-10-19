#!/usr/bin/env python3
"""Publication-style plots for the 10-seed ν = 0.01 Eulerian delay sweep (single grid).

usage: python3 scripts/plot-delay-sweep-visc10.py SWEEP_DIR THEORY_VISC_JSON OUTDIR [--tdiscard 60]
  SWEEP_DIR: --out dir of scripts/delay-sweep.ts (reads runs/*.json; partial sweeps work)
  Averages use the measurement window t > tdiscard (default: theory.json 'tdiscard', else tmax/2).
Writes meancos_vs_delay.{png,pdf,csv}, heatmap_log.{png,pdf}, summary_runs.csv, hist_avg.csv.
"""
import glob, json, os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

args = [a for a in sys.argv[1:]]
td_cli = None
if '--tdiscard' in args:
    i = args.index('--tdiscard'); td_cli = float(args[i + 1]); del args[i:i + 2]
d, thf, out = args[:3]
os.makedirs(out, exist_ok=True)
th = json.load(open(os.path.join(d, 'theory.json')))
TH = json.load(open(thf))['chosen']; NU = TH['nu']
g = TH['grids'][f"{th['nx']}x{th['ny']}"]; W0, W1 = g['onset_phys_min'], g['onset_phys_max']
TMAX = th['tmax']; TD = td_cli if td_cli is not None else th.get('tdiscard', TMAX / 2)
plt.rcParams.update({'font.size': 10, 'font.family': 'DejaVu Sans', 'axes.linewidth': 0.8, 'axes.spines.top': False,
                     'axes.spines.right': False, 'savefig.bbox': 'tight', 'legend.frameon': False})

runs = []
for f in sorted(glob.glob(os.path.join(d, 'runs', '*.json'))):
    r = json.load(open(f)); t = np.array(r['t']); m = t > TD + 1e-9
    mc = np.array(r['meanCos'])[m]; H = np.array(r['hist'])[m]
    runs.append(dict(delay=r['delay'], seed=r['seed'], mc=mc.mean(), tstd=mc.std(ddof=1), H=H.mean(0), finite=r['finite'], wall=r['wallSeconds']))
delays = sorted({r['delay'] for r in runs})
agg = []
for dl in delays:
    gg = [r for r in runs if r['delay'] == dl]; y = np.array([r['mc'] for r in gg]); n = len(y)
    sd = y.std(ddof=1) if n > 1 else 0.0
    agg.append(dict(delay=dl, n=n, mean=y.mean(), sd=sd, sem=sd / np.sqrt(n), H=np.mean([r['H'] for r in gg], 0), ys=y))
x = np.array([a['delay'] for a in agg]); y = np.array([a['mean'] for a in agg]); sem = np.array([a['sem'] for a in agg])
nmin, nmax = min(a['n'] for a in agg), max(a['n'] for a in agg)
nl = f'{nmax} seeds' if nmin == nmax else f'{nmin}–{nmax} seeds (partial)'
xl = r'delay $t_{\mathrm{self}} = t_{\mathrm{nb}} = t$ (s)'
win = f'{TD:.0f}–{TMAX:.0f} s'

with open(os.path.join(out, 'summary_runs.csv'), 'w') as f:
    f.write('delay,seed,meanCos_window,meanCos_window_timestd,finite,wall_s\n')
    for r in sorted(runs, key=lambda r: (r['delay'], r['seed'])):
        f.write(f"{r['delay']},{r['seed']},{r['mc']:.6f},{r['tstd']:.6f},{r['finite']},{r['wall']:.1f}\n")
with open(os.path.join(out, 'meancos_vs_delay.csv'), 'w') as f:
    f.write(f'# window t in ({TD:g},{TMAX:g}] s\ndelay,seeds,meanCos_mean,meanCos_sem,meanCos_seedstd,meanCos_min,meanCos_max\n')
    for a in agg: f.write(f"{a['delay']},{a['n']},{a['mean']:.6f},{a['sem']:.6f},{a['sd']:.6f},{a['ys'].min():.6f},{a['ys'].max():.6f}\n")
B = len(agg[0]['H']); ec = np.linspace(-1, 1, B + 1); cc = (ec[1:] + ec[:-1]) / 2
with open(os.path.join(out, 'hist_avg.csv'), 'w') as f:
    f.write('delay,cos_centre,P,seeds\n')
    for a in agg:
        for k in range(B): f.write(f"{a['delay']},{cc[k]:.3f},{a['H'][k]:.6g},{a['n']}\n")

# (a) mean correlation vs delay
fig, ax = plt.subplots(figsize=(6.4, 4.0))
ax.axvspan(W0, W1, color='0.85', lw=0, zorder=0, label=f'linear-theory onset {W0:.2f}–{W1:.2f} s')
for r in runs: ax.plot(r['delay'], r['mc'], 'o', ms=2.2, color='#1f77b4', alpha=0.22, mew=0, zorder=1)
ax.plot([], [], 'o', ms=3, color='#1f77b4', alpha=0.4, mew=0, label='individual seeds')
ax.fill_between(x, y - sem, y + sem, color='#1f77b4', alpha=0.28, lw=0, zorder=2, label='± s.e.m.')
ax.plot(x, y, '-o', color='#0b3c78', ms=3.6, lw=1.6, zorder=3, label=f'mean over {nl}')
ax.set_xlim(0, 2); ax.set_ylim(0, 1.02); ax.set_xlabel(xl)
ax.set_ylabel(r'$\langle\cos\theta_{ij}\rangle$')
ax.set_title(f'Pairwise velocity correlation vs delay (ν = {NU:g}, {th["nx"]}×{th["ny"]}, window {win})', fontsize=10)
ax.legend(fontsize=8, loc='upper right')
for e in ('png', 'pdf'): fig.savefig(os.path.join(out, f'meancos_vs_delay.{e}'), dpi=200)
plt.close(fig)

# (b) log heatmap
step = np.min(np.diff(x)) if len(x) > 1 else 0.05
ed = np.concatenate([[x[0] - step / 2], x + step / 2])
P = np.array([a['H'] for a in agg])
fig, ax = plt.subplots(figsize=(6.6, 4.0))
m = ax.pcolormesh(ed, ec, np.maximum(P.T, 1e-2), norm=LogNorm(vmin=1e-2, vmax=max(20, P.max())), cmap='magma', shading='flat', rasterized=True)
for v in (W0, W1): ax.axvline(v, color='w', ls='--', lw=0.9, alpha=0.85)
ax.text(W0 - 0.03, -0.1, f'theory onset\n{W0:.2f}–{W1:.2f} s', ha='right', va='center', fontsize=8, color='w')
ax.set_xlim(0, ed[-1]); ax.set_ylim(-1, 1); ax.set_xlabel(xl); ax.set_ylabel(r'$\cos\theta_{ij}$')
cb = fig.colorbar(m, ax=ax, pad=0.02); cb.set_label(r'$P(\cos\theta_{ij})$'); cb.outline.set_linewidth(0.5)
ax.set_title(f'Distribution of pairwise velocity correlations ({nl}, window {win})', fontsize=10)
for e in ('png', 'pdf'): fig.savefig(os.path.join(out, f'heatmap_log.{e}'), dpi=200)
plt.close(fig)
print(' '.join(f"{a['delay']:.2f}:{a['mean']:.3f}±{a['sem']:.3f}({a['n']})" for a in agg))
