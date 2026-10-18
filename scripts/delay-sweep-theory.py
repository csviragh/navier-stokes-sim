#!/usr/bin/env python3
"""Linear theory for the delay sweep: rightmost growth rate of
    λ + i k·U = −a(k) e^{−λ t},   a(k) = S (1 − K̂(k))
around a uniform flock moving at U (|U| = v0), on the discrete torus modes,
and around the state at rest (U = 0, the classic λ = W0(−a t)/t).
Solutions: μ = λ + i k·U, μ t = W_n(−a t e^{i k·U t}); the max over branches n = −1, 0, 1.
usage: python3 scripts/delay-sweep-theory.py /workspace/delay-sweep/kernel_symbols.json out.json
"""
import json, sys
import numpy as np
from scipy.special import lambertw

src, out = sys.argv[1], sys.argv[2]
d = json.load(open(src))
S, v0 = 2.0, 0.5
res = {}
for key, g in d.items():
    nx, ny, h = g['nx'], g['ny'], g['h']
    sym = np.array(g['sym']).reshape(ny, nx)
    ia = np.arange(nx); ia = np.where(ia <= nx // 2, ia, ia - nx)
    ib = np.arange(ny); ib = np.where(ib <= ny // 2, ib, ib - ny)
    kx = 2 * np.pi * ia / (nx * h); ky = 2 * np.pi * ib / (ny * h)
    KX, KY = np.meshgrid(kx, ky)
    a = S * (1 - sym)
    mask = ~((KX == 0) & (KY == 0))
    a, KX, KY = a[mask], KX[mask], KY[mask]
    ts = np.round(np.arange(0.01, 2.0001, 0.01), 4)
    dt = 1 / 60
    def sl_damping(U):
        """Damping rate of the semi-Lagrangian bilinear interpolation for a uniform
        displacement U dt (only the fractional cell part damps): |G|² = Π 1 − 2c(1 − c)(1 − cos k h)."""
        g2 = np.ones_like(KX)
        for k, u in ((KX, U[0]), (KY, U[1])):
            c = abs(u) * dt / h; c -= np.floor(c)
            g2 *= 1 - 2 * c * (1 - c) * (1 - np.cos(k * h))
        return -0.5 * np.log(np.maximum(g2, 1e-300)) / dt
    def growth(U, sl=False):
        gam = sl_damping(U) if sl else np.zeros_like(KX)
        g = []
        for t in ts:
            # λ + i k·U + γ = −a e^{−λ t}  ⇒  ν t = W(−a t e^{(i k·U + γ) t}),  λ = ν − γ − i k·U
            z = -a * t * np.exp((1j * (KX * U[0] + KY * U[1]) + gam) * t)
            best = max(np.max(lambertw(z, n).real / t - gam) for n in (-1, 0, 1))
            g.append(best)
        return np.array(g)
    def crossing(gr):
        i = np.argmax(gr > 0)
        return float(ts[i]) if gr[i] > 0 else None
    rest = growth((0.0, 0.0))
    out_k = {'t': ts.tolist(), 'rest': rest.tolist(), 't_unstable_rest': crossing(rest),
             'a_max': float(a.max()), 'ringing_rest': float(1 / (np.e * a.max()))}
    for name, th in (('flock_x', 0.0), ('flock_diag', np.pi / 4), ('flock_y', np.pi / 2)):
        gr = growth((v0 * np.cos(th), v0 * np.sin(th)))
        out_k[name] = gr.tolist(); out_k[f't_unstable_{name}'] = crossing(gr)
        gs = growth((v0 * np.cos(th), v0 * np.sin(th)), sl=True)
        out_k[name + '_SL'] = gs.tolist(); out_k[f't_unstable_{name}_SL'] = crossing(gs)
        for tt in (0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.8):
            i = int(np.argmin(abs(ts - tt)))
            out_k[f'growth_{name}_t{tt}'] = round(float(gr[i]), 3); out_k[f'growth_{name}_SL_t{tt}'] = round(float(gs[i]), 3)
    res[key] = out_k
    print(key, {k: v for k, v in out_k.items() if not isinstance(v, list)})
json.dump(res, open(out, 'w'))
