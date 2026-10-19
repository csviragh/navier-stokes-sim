#!/usr/bin/env python3
"""Linear theory of the Eulerian (fixed-point read) delay sweep with an
UNDELAYED physical viscosity ν, around a uniform flock moving at U (|U| = v0,
heading θ). Incompressible perturbation δu = A k̂⊥ e^{ik·x}; self-propulsion
damps its component along Û at rate sin²α/τ (α = angle(k, U)):

    μ + ν k_h² + sin²α/τ = −a(k) e^{−μ t} e^{i k·U t},   a(k) = S (1 − K̂(k)),
    μ = λ + i k·U (comoving),   Re λ = Re μ,
    (μ + g) t = W_n(−a t e^{g t} e^{i k·U t}),   g = ν k_h² + sin²α/τ [+ γ_SL].

k_h² = (4/h²)(sin²(k_x h/2) + sin²(k_y h/2)) is the discrete Laplacian symbol.
Modes with ν k_h² ≥ S can never grow → cutoff k_c = sqrt(S/ν). Evaluated on the
discrete torus modes of each grid, (a) without numerical damping and (b) with
the bilinear semi-Lagrangian advection damping γ_SL of the moving flock.

usage: python3 scripts/delay-sweep-theory-visc.py kernel_symbols.json out.json [nu] [v0]
"""
import json, sys
import numpy as np
from scipy.special import lambertw

src, out = sys.argv[1], sys.argv[2]
NU = float(sys.argv[3]) if len(sys.argv) > 3 else 0.01
S, tau, dt = 2.0, 1.0, 1 / 60
v0 = float(sys.argv[4]) if len(sys.argv) > 4 else 0.5
d = json.load(open(src))
TS = np.round(np.arange(0.01, 2.0001, 0.01), 4)
THETAS = np.linspace(0, np.pi / 2, 7)  # heading of the flock (symmetry: 0..π/2 covers all)


def modes(key, kcut):
    g = d[key]; nx, ny, h = g['nx'], g['ny'], g['h']
    sym = np.array(g['sym']).reshape(ny, nx)
    ia = np.arange(nx); ia = np.where(ia <= nx // 2, ia, ia - nx)
    ib = np.arange(ny); ib = np.where(ib <= ny // 2, ib, ib - ny)
    KX, KY = np.meshgrid(2 * np.pi * ia / (nx * h), 2 * np.pi * ib / (ny * h))
    k2h = (4 / h**2) * (np.sin(KX * h / 2) ** 2 + np.sin(KY * h / 2) ** 2)
    m = ~((KX == 0) & (KY == 0)) & (np.sqrt(KX**2 + KY**2) <= kcut)
    return KX[m], KY[m], k2h[m], S * (1 - sym[m]), h


def sl_damping(KX, KY, U, h):
    g2 = np.ones_like(KX)
    for k, u in ((KX, U[0]), (KY, U[1])):
        c = abs(u) * dt / h; c -= np.floor(c)
        g2 *= 1 - 2 * c * (1 - c) * (1 - np.cos(k * h))
    return -0.5 * np.log(np.maximum(g2, 1e-300)) / dt


def growth_curve(key, nu, theta, sl, kcut=np.inf):
    KX, KY, k2h, a, h = modes(key, kcut)
    U = (v0 * np.cos(theta), v0 * np.sin(theta))
    kk = np.sqrt(KX**2 + KY**2)
    sin2 = ((KX * U[1] - KY * U[0]) / (kk * v0)) ** 2  # sin²α
    g = nu * k2h + sin2 / tau + (sl_damping(KX, KY, U, h) if sl else 0)
    keep = g < a  # |μ + g| = a e^{−Re μ t} ⇒ Re μ ≥ 0 needs g ≤ a
    KX, KY, a, g = KX[keep], KY[keep], a[keep], g[keep]
    kU = KX * U[0] + KY * U[1]
    res, kbest = [], []
    for t in TS:
        if not len(a): res.append(-np.inf); kbest.append(0); continue
        z = -a * t * np.exp(g * t + 1j * kU * t)
        r = np.max([lambertw(z, n).real for n in (-1, 0, 1)], axis=0) / t - g
        i = int(np.argmax(r)); res.append(float(r[i])); kbest.append(float(np.hypot(KX[i], KY[i])))
    return np.array(res), np.array(kbest)


def onset(gr):
    i = int(np.argmax(gr > 1e-6))
    return float(TS[i]) if gr[i] > 1e-6 else None


def analyse(nu):
    o = {'nu': nu, 'k_c': float(np.sqrt(S / nu)) if nu > 0 else None,
         'lambda_c': float(2 * np.pi / np.sqrt(S / nu)) if nu > 0 else None, 't': TS.tolist(), 'grids': {}}
    for key in d:
        gk = {}
        for sl in (False, True):
            curves = [growth_curve(key, nu, th, sl) for th in THETAS]
            ons = [onset(c[0]) for c in curves]
            tag = 'SL' if sl else 'phys'
            gk[f'onset_{tag}_by_heading'] = dict(zip([f'{th:.3f}' for th in THETAS], ons))
            valid = [x for x in ons if x is not None]
            gk[f'onset_{tag}_min'] = min(valid) if valid else None
            gk[f'onset_{tag}_max'] = max(valid) if valid else None
            env = np.max([c[0] for c in curves], axis=0)
            gk[f'growth_{tag}_max_over_heading'] = env.tolist()
            gk[f'growth_{tag}_heading0'] = curves[0][0].tolist()
            gk[f'kbest_{tag}_heading0'] = curves[0][1].tolist()
        o['grids'][key] = gk
    return o


if __name__ == '__main__':
    res = {'chosen': analyse(NU), 'scan': {}}
    for nu in (0.0, 1e-3, 2e-3, 3e-3, 5e-3, 7e-3, 1e-2, 1.5e-2, 2e-2):
        r = analyse(nu)
        res['scan'][f'{nu:g}'] = {k: {kk: vv for kk, vv in g.items() if kk.startswith('onset') and not kk.endswith('heading')} for k, g in r['grids'].items()}
        print(f'nu={nu:g}', json.dumps(res['scan'][f'{nu:g}']), flush=True)
    json.dump(res, open(out, 'w'))
