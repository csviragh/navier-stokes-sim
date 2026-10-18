#!/usr/bin/env python3
"""Eulerian vs Lagrangian (path-following) delay: steady-state ⟨cos θ_ij⟩ vs delay on both grids.

usage: python3 scripts/plot-delay-compare.py EULERIAN_DIR LAGRANGIAN_DIR OUTDIR
Each DIR holds summary.csv (128×72) and fullres-219x128/summary.csv from scripts/delay-sweep.ts.
Writes meancos_compare.{png,pdf} and meancos_compare.csv (mean and seed std per delay and run set).
"""
import csv, json, os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

eul, lag, out = sys.argv[1], sys.argv[2], sys.argv[3]
os.makedirs(out, exist_ok=True)
S = json.load(open(os.path.join(lag, 'theory.json')))['S']

def load(path):
    if not os.path.exists(path): return None
    by = {}
    for r in csv.DictReader(open(path)): by.setdefault(float(r['delay']), []).append(float(r['meanCos_steady']))
    d = sorted(by)
    return (np.array(d), np.array([np.mean(by[x]) for x in d]),
            np.array([np.std(by[x], ddof=1) if len(by[x]) > 1 else 0.0 for x in d]),
            (min(len(v) for v in by.values()), max(len(v) for v in by.values())))

sets = [  # label, path, colour, marker, linestyle
    ('Eulerian, 128×72', os.path.join(eul, 'summary.csv'), '#1f77b4', 's', '-'),
    ('Eulerian, 219×128', os.path.join(eul, 'fullres-219x128', 'summary.csv'), '#1f77b4', 'o', '--'),
    ('Lagrangian, 128×72', os.path.join(lag, 'summary.csv'), '#d62728', 's', '-'),
    ('Lagrangian, 219×128', os.path.join(lag, 'fullres-219x128', 'summary.csv'), '#ff7f0e', 'o', '--'),
]
data = [(lab, load(p), c, m, ls) for lab, p, c, m, ls in sets]

plt.rcParams.update({'font.size': 10, 'font.family': 'DejaVu Sans', 'savefig.bbox': 'tight'})
fig, axs = plt.subplots(1, 2, figsize=(11, 4.4), gridspec_kw={'width_ratios': [1.25, 1]})
fig.subplots_adjust(bottom=0.14, top=0.88, wspace=0.18)
t1, t2 = 1 / (np.e * S), np.pi / (2 * S)
for k, ax in enumerate(axs):
    ax.axhline(0, color='0.6', lw=0.6)
    ax.axvline(t1, color='0.45', ls=':', lw=1.3, label=f'$St = 1/e$ (rest: ringing onset), t = {t1:.3f} s')
    ax.axvline(t2, color='0.15', ls='--', lw=1.3, label=f'$St = \\pi/2$ (rest: unstable), t = {t2:.3f} s')
    thl = os.path.join(lag, 'theory_lagrangian.json')
    if os.path.exists(thl):  # comoving theory incl. the solver's semi-Lagrangian damping (earliest flock direction)
        tl = json.load(open(thl))
        for g, ls2 in (('128x72', '-.'), ('219x128', (0, (3, 1, 1, 1, 1, 1)))):
            if g in tl:
                tt = min(tl[g][f't_unstable_{n}_sp_SL_E'] for n in ('x', 'diag', 'y'))
                ax.axvline(tt, color='#d62728', ls=ls2, lw=0.9, alpha=0.7,
                           label=f'Lagrangian theory + advection damping, {g.replace("x", "×")}: t = {tt:.2f} s')
    for lab, d, c, m, ls in data:
        if d is None: continue
        x, y, e, n = d
        sel = x <= (2.05 if k == 0 else 1.06)
        nlab = f'{n[1]} seed' + ('s' if n[1] > 1 else '') if n[0] == n[1] else f'{n[0]}–{n[1]} seeds, partial'
        ax.errorbar(x[sel], y[sel], yerr=e[sel] if n[1] > 1 else None, fmt=m + ls, color=c, mfc='none' if '219' in lab else c,
                    ms=7 if '219' in lab else 4, mew=1.4, lw=1.1, capsize=2, label=f"{lab} ({nlab})")
    ax.set_xlabel(r'delay $t_{\mathrm{self}} = t_{\mathrm{nb}} = t$ (s)')
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlim(-0.03, 2.03 if k == 0 else 1.08)
axs[0].set_ylabel(r'steady-state $\langle\cos\theta_{ij}\rangle$ (60–120 s)')
axs[0].set_title('full range', fontsize=10); axs[1].set_title('zoom 0–1 s (both grids)', fontsize=10)
h, l = axs[0].get_legend_handles_labels()
fig.legend(h, l, fontsize=7.5, frameon=False, loc='upper center', bbox_to_anchor=(0.5, 0.06), ncol=2)
fig.suptitle('Eulerian (fixed-point) vs Lagrangian (along-path) delay: average pairwise alignment', fontsize=11)
fig.text(0.5, -0.2, f'CS kernel S = {S}/s, R = 0.05, β = 1; self-propulsion τ = 1 s, v₀ = 0.5; ν = 0; torus; 120 s per run',
         ha='center', fontsize=8, color='0.3')
for ext in ('png', 'pdf'):
    fig.savefig(os.path.join(out, f'meancos_compare.{ext}'), dpi=200)
with open(os.path.join(out, 'meancos_compare.csv'), 'w') as f:
    f.write('set,delay,meanCos_steady_mean,meanCos_steady_seedstd,seeds\n')
    for lab, d, *_ in data:
        if d is None: continue
        for x, y, e in zip(d[0], d[1], d[2]): f.write(f'"{lab}",{x},{y:.6f},{e:.6f},{d[3][1]}\n')
print('wrote', out)
