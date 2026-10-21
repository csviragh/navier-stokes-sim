#!/usr/bin/env python3
"""Snapshot figure of regimes from --dump fields (runs/*.field.bin: float32 [u,v] each ny×nx, t = tmax).
usage: plot-delay-snapshots.py OUT.png label=path.field.bin [...]"""
import sys, json, numpy as np, matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
out, items = sys.argv[1], [a.rsplit('=', 1) for a in sys.argv[2:]]
fig, ax = plt.subplots(3, len(items), figsize=(3.2 * len(items), 9.6), squeeze=False)
for c, (lab, p) in enumerate(items):
    n = int(round((np.fromfile(p, np.float32).size / 2) ** .5))
    u, v = np.fromfile(p, np.float32).reshape(2, n, n); sp = np.hypot(u, v)
    r = json.load(open(p.replace('.field.bin', '.json'))); k = [i for i, t in enumerate(r['t']) if t > 120]
    mc = np.mean([r['meanCos'][i] for i in k]); phi = r['phi'][-1]
    hsv = np.stack([(np.arctan2(v, u) / (2 * np.pi)) % 1, np.ones_like(u), np.clip(sp / np.percentile(sp, 99), 0, 1)], -1)
    ax[0, c].imshow(hsv_to_rgb(hsv), origin='lower', extent=[0, 1, 0, 1])
    s = n // 16; X, Y = np.meshgrid((np.arange(0, n, s) + .5) / n, (np.arange(0, n, s) + .5) / n)
    ax[0, c].quiver(X, Y, u[::s, ::s], v[::s, ::s], color='w', scale=sp.max() * 20, width=.004)
    ax[0, c].set_title(f'{lab}\n⟨cos⟩={mc:.2f}, φ={phi:.2f}', fontsize=9)
    h = 1 / n; w = (np.roll(v, -1, 1) - np.roll(v, 1, 1) - np.roll(u, -1, 0) + np.roll(u, 1, 0)) / (2 * h)
    m = np.percentile(abs(w), 99); ax[1, c].imshow(w, origin='lower', cmap='RdBu_r', vmin=-m, vmax=m, extent=[0, 1, 0, 1])
    ax[1, c].set_title('vorticity ω', fontsize=9)
    ax[2, c].imshow(sp, origin='lower', cmap='viridis', extent=[0, 1, 0, 1]); ax[2, c].set_title(f'speed |u| (max {sp.max():.3f})', fontsize=9)
    for a in ax[:, c]: a.set_xticks([]); a.set_yticks([])
fig.suptitle('Late-time (t = 240 s) velocity fields: hue = direction, brightness = speed', fontsize=10)
fig.tight_layout(); fig.savefig(out, dpi=130); fig.savefig(out.rsplit('.', 1)[0] + '.pdf')
