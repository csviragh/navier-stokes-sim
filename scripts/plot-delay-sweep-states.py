#!/usr/bin/env python3
"""State-classification view for a delay sweep (runs/*.json).

usage: python3 scripts/plot-delay-sweep-states.py SWEEP_DIR OUTDIR --tdiscard 120
Each run's window-mean ⟨cos θ_ij⟩ and φ are rounded to 0.005; at each delay the most common
(⟨cos⟩, φ) pair is the 'main branch'. Runs whose time std > 0.03 are flagged 'switching'.
Writes states.{png,pdf} and states.csv.
"""
import glob, json, os, sys
from collections import Counter
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

a = sys.argv[1:]; td = 60.0
if '--tdiscard' in a: i = a.index('--tdiscard'); td = float(a[i + 1]); del a[i:i + 2]
d, out = a[:2]; os.makedirs(out, exist_ok=True)
plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False, 'savefig.bbox': 'tight', 'legend.frameon': False})
runs = []
for f in glob.glob(os.path.join(d, 'runs', '*.json')):
    r = json.load(open(f)); t = np.array(r['t']); m = t > td + 1e-9
    mc = np.array(r['meanCos'])[m]; ph = np.array(r['phi'])[m]
    runs.append(dict(delay=r['delay'], seed=r['seed'], mc=mc.mean(), sd=mc.std(), phi=ph.mean()))
key = lambda r: (round(r['mc'] / 0.005) * 0.005, round(r['phi'] / 0.005) * 0.005)
delays = sorted({r['delay'] for r in runs}); frac = []
for dl in delays:
    g = [r for r in runs if r['delay'] == dl]
    main, cnt = Counter(key(r) for r in g if r['sd'] <= 0.03).most_common(1)[0] if any(r['sd'] <= 0.03 for r in g) else (None, 0)
    for r in g:
        r['state'] = 'switching' if r['sd'] > 0.03 else ('main' if key(r) == main else 'other')
    frac.append((dl, len(g), sum(r['state'] == 'main' for r in g), sum(r['state'] == 'other' for r in g), sum(r['state'] == 'switching' for r in g),
                 main[0] if main else np.nan))
with open(os.path.join(out, 'states.csv'), 'w') as f:
    f.write('delay,seed,meanCos_window,meanCos_timestd,phi_window,state\n')
    for r in sorted(runs, key=lambda r: (r['delay'], r['seed'])): f.write(f"{r['delay']},{r['seed']},{r['mc']:.6f},{r['sd']:.6f},{r['phi']:.6f},{r['state']}\n")
fig, (ax, bx) = plt.subplots(2, 1, figsize=(6.4, 5.4), sharex=True, gridspec_kw=dict(height_ratios=[2.2, 1], hspace=0.08))
sty = {'main': dict(c='#0b3c78', marker='o', s=14, alpha=0.6), 'other': dict(c='#d62728', marker='D', s=18, alpha=0.8), 'switching': dict(c='#ff7f0e', marker='x', s=22, alpha=0.9)}
F = np.array(frac, float)
ax.plot(F[:, 0], F[:, 5], '-', color='#0b3c78', lw=1, alpha=0.5)
for st, kw in sty.items():
    rr = [r for r in runs if r['state'] == st]
    ax.scatter([r['delay'] for r in rr], [r['mc'] for r in rr], label=f'{st} ({len(rr)})', lw=1 if st == 'switching' else 0, **kw)
ax.set_ylim(0, 1.02); ax.set_ylabel(r'$\langle\cos\theta_{ij}\rangle$ per run'); ax.legend(fontsize=8, loc='upper right')
ax.set_title('Steady-state classification per run (main = most common state at that delay)', fontsize=10)
n = F[:, 1]
w = 0.045; b0 = np.zeros_like(n)
for col, c in ((2, '#0b3c78'), (3, '#d62728'), (4, '#ff7f0e')):
    bx.bar(F[:, 0], F[:, col] / n, w, bottom=b0, color=c, alpha=0.75, lw=0); b0 = b0 + F[:, col] / n
bx.set_ylim(0, 1); bx.set_xlim(-0.03, 2.03); ax.set_xlim(-0.03, 2.03); bx.set_ylabel('fraction'); bx.set_xlabel(r'delay $t_{\mathrm{self}} = t_{\mathrm{nb}} = t$ (s)')
for e in ('png', 'pdf'): fig.savefig(os.path.join(out, f'states.{e}'), dpi=200)
for x in frac: print(f'{x[0]:.2f} main {x[2]}/{x[1]} other {x[3]} switching {x[4]} main_value {x[5]:.3f}')
