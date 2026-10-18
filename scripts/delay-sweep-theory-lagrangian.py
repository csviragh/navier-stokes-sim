#!/usr/bin/env python3
"""Linear theory for the Lagrangian (along-path) delay sweep: perturbations of a
uniform flock u = U (|U| = v0) in the comoving frame, on the discrete torus modes,

    λ = −γ_SL(k) − γ_sp(k) − S (1 − K̂(k)) E(k, t) e^{−λ t}

* S (1 − K̂): the equal-delay CS term (self − neighbour), as at rest (no Doppler phase).
* γ_sp = (1/τ)(ê·Û)²: self-propulsion relaxes only the SPEED (component of the
  perturbation along U) at rate 1/τ; ê = k⊥/|k| is the (divergence-free) mode's
  velocity direction. Direction modes (k ∥ U, ê ⊥ U) are not damped.
* γ_SL: damping of the solver's semi-Lagrangian bilinear advection for the displacement U Δt.
* E: error factor of reading the past along the path with bilinear sampling, relative to an
  exact shift: the read time t + Δt/2 − t_del lies between two frames, each sampled at its
  own (fractional-cell) parcel position, E = ½ Σ_f Π_axes (1 − c_f(1 − e^{−i k h})) e^{i k c_f h}.
Solution: with γ = γ_SL + γ_sp, μ = λ + γ, μ t = W_n(−a E t e^{γ t}); max over n = −1, 0, 1.
usage: python3 scripts/delay-sweep-theory-lagrangian.py kernel_symbols.json out.json
"""
import json, sys
import numpy as np
from scipy.special import lambertw

src, out = sys.argv[1], sys.argv[2]
d = json.load(open(src))
S, v0, tau_sp, dt = 2.0, 0.5, 1.0, 1 / 60
res = {}
for key, g in d.items():
    nx, ny, h = g['nx'], g['ny'], g['h']
    sym = np.array(g['sym']).reshape(ny, nx)
    ia = np.arange(nx); ia = np.where(ia <= nx // 2, ia, ia - nx)
    ib = np.arange(ny); ib = np.where(ib <= ny // 2, ib, ib - ny)
    KX, KY = np.meshgrid(2 * np.pi * ia / (nx * h), 2 * np.pi * ib / (ny * h))
    mask = ~((KX == 0) & (KY == 0))
    a0, KX, KY = (S * (1 - sym))[mask], KX[mask], KY[mask]
    KM = np.hypot(KX, KY)
    ts = np.round(np.arange(0.02, 2.0001, 0.01), 4)
    out_k = {'t': ts.tolist()}
    for name, th in (('x', 0.0), ('diag', np.pi / 4), ('y', np.pi / 2)):
        U = np.array([v0 * np.cos(th), v0 * np.sin(th)])
        # self-propulsion: speed component of the mode's velocity ê = (−ky, kx)/|k|
        ex, ey = -KY / KM, KX / KM
        gsp = (ex * U[0] + ey * U[1]) ** 2 / v0 ** 2 / tau_sp
        gsl = np.zeros_like(KX)
        for k, u in ((KX, U[0]), (KY, U[1])):
            c = abs(u) * dt / h; c -= np.floor(c)
            gsl += -0.5 * np.log(np.maximum(1 - 2 * c * (1 - c) * (1 - np.cos(k * h)), 1e-300)) / dt
        rows = {}
        for variant in ('ideal', 'sp', 'sp_SL', 'sp_SL_E'):
            gr = []
            for t in ts:
                gam = (gsp if 'sp' in variant else 0) + (gsl if 'SL' in variant else 0)
                E = np.ones_like(KX, dtype=complex)
                if variant.endswith('_E'):
                    E = 0
                    for frame_shift in (0.0, 1.0):  # the two bracketing frames: t − t_del and t − t_del + Δt
                        Ef = np.ones_like(KX, dtype=complex)
                        for k, u in ((KX, U[0]), (KY, U[1])):
                            disp = abs(u) * (t - frame_shift * dt) / h
                            c = disp - np.floor(disp)
                            Ef *= (1 - c * (1 - np.exp(-1j * k * h * np.sign(u or 1)))) * np.exp(1j * k * c * h * np.sign(u or 1))
                        E = E + 0.5 * Ef
                z = -a0 * E * t * np.exp(gam * t)
                gr.append(float(max(np.max(lambertw(z, n).real / t - gam) for n in (-1, 0, 1))))
            gr = np.array(gr)
            i = int(np.argmax(gr > 0))
            out_k[f'{name}_{variant}'] = gr.tolist()
            out_k[f't_unstable_{name}_{variant}'] = float(ts[i]) if gr[i] > 0 else None
            rows[variant] = gr
        for tt in (0.8, 0.9, 1.0, 1.1, 1.5, 2.0):
            i = int(np.argmin(abs(ts - tt)))
            out_k[f'growth_{name}_t{tt}'] = {v: round(float(rows[v][i]), 4) for v in rows}
    res[key] = out_k
    print(key, json.dumps({k: v for k, v in out_k.items() if not isinstance(v, list)}, indent=0))
json.dump(res, open(out, 'w'))
