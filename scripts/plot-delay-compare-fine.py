#!/usr/bin/env python3
"""Compare fine random-start sweep with 0.05-step random-start and ordered-start sweeps (summary_runs.csv each).
usage: plot-delay-compare-fine.py FINE_DIR RANDOM_DIR ORDERED_DIR OUTDIR
Writes compare_fine.csv and compare_fine.png/.pdf (ordered run: window <cos> > 0.05)."""
import os, sys, csv, numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
fd, rd, od, out = sys.argv[1:5]; TH = 0.05
def load(d):
    g = {}
    for r in csv.DictReader(open(os.path.join(d, 'summary_runs.csv'))): g.setdefault(round(float(r['delay']), 2), []).append(float(r['meanCos_window']))
    return {k: np.array(v) for k, v in g.items()}
S = [(load(fd), 'random start, 0.01 step (4 seeds)', '#0b3c78', 'o'), (load(rd), 'random start, 0.05 step (10 seeds)', '#2ca02c', '^'), (load(od), 'ordered start', '#d62728', 's')]
with open(os.path.join(out, 'compare_fine.csv'), 'w') as f:
    f.write('delay' + ''.join(f',{p}_n,{p}_mean,{p}_std,{p}_ordered_runs' for p in ('fine', 'random05', 'ordered')) + f'  # ordered run: window <cos> > {TH}\n')
    for d in sorted(set().union(*[s[0] for s in S])):
        if not 0.45 <= d <= 1.30: continue
        row = [f'{d:.2f}']
        for G, *_ in S:
            y = G.get(d)
            row += ['', '', '', ''] if y is None else [str(len(y)), f'{y.mean():.6f}', f'{y.std(ddof=1) if len(y) > 1 else 0:.6f}', str(int((y > TH).sum()))]
        f.write(','.join(row) + '\n')
plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False, 'savefig.bbox': 'tight', 'legend.frameon': False})
fig, (ax, bx) = plt.subplots(1, 2, figsize=(12, 4.3), gridspec_kw=dict(wspace=0.2))
for G, lab, c, m in S:
    x = np.array([d for d in sorted(G) if 0.45 <= d <= 1.30]); y = np.array([G[d].mean() for d in x]); s = np.array([G[d].std(ddof=1) if len(G[d]) > 1 else 0 for d in x])
    ax.errorbar(x, y, yerr=s, fmt=m + '-', ms=3, lw=1, color=c, capsize=1.5, elinewidth=0.7, label=lab, mfc='none' if m != 'o' else c)
    fr = np.array([(G[d] > TH).mean() for d in x]); bx.plot(x, fr, m + '-', ms=3, lw=1, color=c, label=lab, mfc='none' if m != 'o' else c)
for A in (ax, bx): A.set_xlim(0.48, 1.27); A.set_xlabel(r'delay $t_{\mathrm{self}}=t_{\mathrm{nb}}$ (s)')
ax.set_ylim(-0.05, 1.03); ax.set_ylabel(r'$\langle\cos\theta_{ij}\rangle$ (mean ± 1 std)'); ax.legend(fontsize=8.5)
bx.set_ylim(-0.03, 1.03); bx.set_ylabel(r'fraction of runs ordered ($\langle\cos\rangle > 0.05$)')
fig.suptitle('Fine random-start sweep vs 0.05-step random start and ordered start (256², ν = 1e-3, v0 = 0.05, window 120–240 s)', fontsize=10.5)
for e in ('png', 'pdf'): fig.savefig(os.path.join(out, f'compare_fine.{e}'), dpi=200)
