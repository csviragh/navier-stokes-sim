#!/usr/bin/env python3
"""Compare random-start vs ordered-start delay sweeps from their summary_runs.csv files.
usage: plot-delay-compare-init.py RANDOM_DIR ORDERED_DIR OUTDIR   (dirs = docs/sweeps/... with summary_runs.csv)
Writes compare_init.csv (per delay: n, mean, std, #runs with <cos> > 0.05 for each start) and compare_init.png/.pdf."""
import os, sys, csv, numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
rd, od, out = sys.argv[1:4]; TH = 0.05
def load(d):
    g = {}
    for r in csv.DictReader(open(os.path.join(d, 'summary_runs.csv'))): g.setdefault(round(float(r['delay']), 2), []).append(float(r['meanCos_window']))
    return {k: np.array(v) for k, v in g.items()}
R, O = load(rd), load(od)
with open(os.path.join(out, 'compare_init.csv'), 'w') as f:
    f.write(f'delay,random_n,random_mean,random_std,random_ordered_runs,ordered_n,ordered_mean,ordered_std,ordered_ordered_runs  # ordered run: window <cos> > {TH}\n')
    for d in sorted(set(R) | set(O)):
        row = [f'{d:.2f}']
        for G in (R, O):
            y = G.get(d)
            row += ['', '', '', ''] if y is None else [str(len(y)), f'{y.mean():.6f}', f'{y.std(ddof=1) if len(y) > 1 else 0:.6f}', str(int((y > TH).sum()))]
        f.write(','.join(row) + '\n')
plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False, 'savefig.bbox': 'tight', 'legend.frameon': False})
fig, (ax, bx) = plt.subplots(1, 2, figsize=(11.5, 4.2), gridspec_kw=dict(wspace=0.2))
for G, lab, c, m in ((O, 'ordered start', '#d62728', 's'), (R, 'random start', '#0b3c78', 'o')):
    x = np.array(sorted(G)); y = np.array([G[d].mean() for d in x]); s = np.array([G[d].std(ddof=1) if len(G[d]) > 1 else 0 for d in x])
    for A in (ax, bx): A.errorbar(x, y, yerr=s, fmt=m + '-', ms=3, lw=1, color=c, capsize=1.5, elinewidth=0.7, label=lab, mfc='none' if m == 's' else c)
for A, xl in ((ax, (0, 2.02)), (bx, (0.6, 1.3))): A.set_xlim(*xl); A.set_ylim(-0.05, 1.03); A.set_xlabel(r'delay $t_{\mathrm{self}}=t_{\mathrm{nb}}$ (s)')
ax.set_ylabel(r'$\langle\cos\theta_{ij}\rangle$ (mean ± 1 std over runs)'); bx.set_title('zoom 0.60–1.30 s', fontsize=10); ax.legend()
fig.suptitle('Random vs ordered initial condition (256² unit square, ν = 1e-3, v0 = 0.05, window 120–240 s)', fontsize=10.5)
for e in ('png', 'pdf'): fig.savefig(os.path.join(out, f'compare_init.{e}'), dpi=200)
