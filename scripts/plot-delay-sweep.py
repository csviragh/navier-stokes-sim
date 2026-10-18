#!/usr/bin/env python3
"""Plots for scripts/delay-sweep.ts output.

usage: python3 scripts/plot-delay-sweep.py /workspace/delay-sweep [outdir]
Writes heatmap_log.{png,pdf}, heatmap_linear.{png,pdf}, meancos_vs_delay.{png,pdf}
and hist_avg.csv (P(cos θ) averaged over seeds and time, per delay).
"""
import csv, gzip, json, sys, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

src = sys.argv[1]
out = sys.argv[2] if len(sys.argv) > 2 else src
os.makedirs(out, exist_ok=True)
th = json.load(open(os.path.join(src, 'theory.json')))
tmax = th['tmax']

sp = os.path.join(src, 'samples.csv')
rows = list(csv.DictReader(open(sp) if os.path.exists(sp) else gzip.open(sp + '.gz', 'rt')))
B = th['bins']
delays = sorted({float(r['delay']) for r in rows})
seeds = sorted({int(r['seed']) for r in rows})
D = {d: i for i, d in enumerate(delays)}
S = {s: i for i, s in enumerate(seeds)}
H_full = np.zeros((len(delays), len(seeds), B)); n_full = np.zeros((len(delays), len(seeds)))
H_st = np.zeros_like(H_full); n_st = np.zeros_like(n_full)
mc = {}  # (d, s) -> list of (t, meanCos)
for r in rows:
    d, s, t = float(r['delay']), int(r['seed']), float(r['t'])
    p = np.array([float(r[f'P{k}']) for k in range(B)])
    H_full[D[d], S[s]] += p; n_full[D[d], S[s]] += 1
    if t > tmax / 2:
        H_st[D[d], S[s]] += p; n_st[D[d], S[s]] += 1
    mc.setdefault((d, s), []).append((t, float(r['meanCos'])))
P_full = (H_full / np.maximum(n_full, 1)[..., None]).mean(axis=1)
P_st = (H_st / np.maximum(n_st, 1)[..., None]).mean(axis=1)
edges_c = np.linspace(-1, 1, B + 1); centres = 0.5 * (edges_c[1:] + edges_c[:-1])
step = delays[1] - delays[0] if len(delays) > 1 else 0.05
edges_d = np.concatenate([[delays[0] - step / 2], np.array(delays) + step / 2])

with open(os.path.join(out, 'hist_avg.csv'), 'w') as f:
    f.write('delay,cos_centre,P_full,P_steady\n')
    for i, d in enumerate(delays):
        for k in range(B):
            f.write(f'{d},{centres[k]:.3f},{P_full[i, k]:.6g},{P_st[i, k]:.6g}\n')

plt.rcParams.update({'font.size': 10, 'font.family': 'DejaVu Sans', 'axes.linewidth': 0.8,
                     'xtick.direction': 'out', 'ytick.direction': 'out', 'savefig.bbox': 'tight'})
lines = [(th['t_ringing'], r'$a_{\max}t=1/e$ (ringing)'), (th['t_unstable'], r'$a_{\max}t=\pi/2$ (unstable)')]
adv_path = os.path.join(src, 'theory_advected.json')
if os.path.exists(adv_path):
    adv = json.load(open(adv_path)).get(f"{th['nx']}x{th['ny']}")
    if adv:
        t_adv = min(adv[f't_unstable_flock_{d}_SL'] for d in ('x', 'diag', 'y'))
        lines.insert(0, (t_adv, 'moving flock unstable (advection + grid damping)'))

STY = {'adv': dict(ls='-.', c='#2ca02c'), 'ring': dict(ls=':', c='0.45'), 'inst': dict(ls='--', c='0.2')}
def kind(lab):
    return 'adv' if 'flock' in lab else ('ring' if 'ringing' in lab else 'inst')
def mark(ax, heat=False):
    hs = []
    for x, lab in lines:
        st = STY[kind(lab)]
        col = {'adv': '#7CFC00', 'ring': '#bbbbbb', 'inst': 'w'}[kind(lab)] if heat else st['c']
        hs.append(ax.axvline(x, color=col, ls=st['ls'], lw=1.3, label=f"{lab.replace(chr(10), ' ')}: t = {x:.3g} s"))
    return hs

MODE = th['params'].get('delayMode', 'eulerian')
MODE_TXT = 'delays read along the parcel paths (Lagrangian)' if MODE == 'lagrangian' else 'delays read at fixed points (Eulerian)'
caption = (f"{MODE_TXT}; CS kernel S = {th['S']}/s, R = {th['params']['kernelRange']}, β = {th['params']['kernelBeta']}; "
           f"self-propulsion τ = {th['params']['selfPropTau']} s, v₀ = {th['params']['selfPropV0']}; ν = 0; torus {th['nx']}×{th['ny']}; "
           f"{len(seeds)} seeds × {tmax:.0f} s")

for scale in ('log', 'linear'):
    fig, axs = plt.subplots(1, 2, figsize=(10, 4.3), sharey=True)
    fig.subplots_adjust(left=0.07, right=0.9, bottom=0.2, top=0.86, wspace=0.06)
    for ax, P, title in ((axs[0], P_full, f'time average 0–{tmax:.0f} s'), (axs[1], P_st, f'steady state {tmax/2:.0f}–{tmax:.0f} s')):
        if scale == 'log':
            norm = LogNorm(vmin=1e-2, vmax=max(20, P.max()))
            m = ax.pcolormesh(edges_d, edges_c, np.maximum(P.T, 1e-2), norm=norm, cmap='magma', shading='flat')
        else:
            m = ax.pcolormesh(edges_d, edges_c, P.T, vmin=0, vmax=min(P.max(), 6), cmap='magma', shading='flat')
        hs = mark(ax, heat=True)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel(r'delay $t_{\mathrm{self}} = t_{\mathrm{nb}} = t$ (s)')
        ax.set_xlim(edges_d[0], edges_d[-1])
    leg = axs[0].legend(handles=hs, loc='lower right', fontsize=7, framealpha=0.75, facecolor='0.15', labelcolor='w', edgecolor='none')
    axs[0].set_ylabel(r'pairwise alignment $\cos\theta_{ij} = \hat v_i\cdot\hat v_j$')
    cb = fig.colorbar(m, cax=fig.add_axes([0.915, 0.2, 0.015, 0.66]))
    cb.set_label(r'$P(\cos\theta_{ij})$' + (' (log)' if scale == 'log' else ' (linear, clipped at 6)'))
    fig.suptitle('Distribution of pairwise velocity correlations vs delay', fontsize=11)
    fig.text(0.5, 0.02, caption, ha='center', fontsize=8, color='0.3')
    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join(out, f'heatmap_{scale}.{ext}'), dpi=200)
    plt.close(fig)

# ⟨cos θ⟩ vs delay
full = np.zeros((len(delays), len(seeds))); st = np.zeros_like(full); st_std = np.zeros_like(full)
for (d, s), v in mc.items():
    v.sort(); a = np.array([x for _, x in v]); t = np.array([x for x, _ in v])
    full[D[d], S[s]] = a.mean(); st[D[d], S[s]] = a[t > tmax / 2].mean(); st_std[D[d], S[s]] = a[t > tmax / 2].std(ddof=1)
x = np.array(delays)
fig, ax = plt.subplots(figsize=(6.4, 4.2))
fig.subplots_adjust(bottom=0.2, top=0.9, left=0.12, right=0.97)
ax.axhline(0, color='0.6', lw=0.6)
mark(ax)
sd_seed = full.std(axis=1, ddof=1) if len(seeds) > 1 else np.zeros(len(x))
ax.errorbar(x, full.mean(axis=1), yerr=sd_seed, fmt='o-', ms=3.5, lw=1.2, capsize=2, color='#1f77b4', label=f'full run 0–{tmax:.0f} s (bars: seed std)')
m_st = st.mean(axis=1); sd_st = st.std(axis=1, ddof=1) if len(seeds) > 1 else np.zeros(len(x))
tstd = np.sqrt((st_std ** 2).mean(axis=1))
ax.fill_between(x, m_st - tstd, m_st + tstd, color='#d62728', alpha=0.15, lw=0, label='steady: ±1 temporal std')
ax.errorbar(x, m_st, yerr=sd_st, fmt='s-', ms=3.5, lw=1.2, capsize=2, color='#d62728', label=f'steady state {tmax/2:.0f}–{tmax:.0f} s (bars: seed std)')
fr = os.path.join(src, 'fullres-219x128', 'summary.csv')
if os.path.exists(fr):
    fr_rows = list(csv.DictReader(open(fr)))
    by = {}
    for r in fr_rows: by.setdefault(float(r['delay']), []).append(float(r['meanCos_steady']))
    fx = sorted(by); fy = [np.mean(by[d]) for d in fx]; fs = [np.std(by[d], ddof=1) if len(by[d]) > 1 else 0 for d in fx]
    ns = max(len(v) for v in by.values())
    ax.errorbar(fx, fy, yerr=fs if ns > 1 else None, fmt='o', mfc='none', mec='k', ecolor='k', ms=5, capsize=2,
                label=f"steady state, 219×128 check ({ns} seed{'s' if ns > 1 else ''})")
    adv_all = json.load(open(adv_path)) if os.path.exists(adv_path) else {}
    if '219x128' in adv_all:
        t2 = min(adv_all['219x128'][f't_unstable_flock_{d}_SL'] for d in ('x', 'diag', 'y'))
        ax.axvline(t2, color='k', ls='-.', lw=0.8, alpha=0.6, label=f'moving flock unstable, 219×128: t = {t2:.3g} s')
ax.set_xlabel(r'delay $t_{\mathrm{self}} = t_{\mathrm{nb}} = t$ (s)')
ax.set_ylabel(r'$\langle\cos\theta_{ij}\rangle$ (exact pair average)')
ax.set_ylim(-0.05, 1.05); ax.set_xlim(-0.03, x.max() + 0.03)
ax.legend(fontsize=7.5, frameon=False, loc='best')
ax.set_title('Average pairwise velocity correlation vs delay', fontsize=11)
fig.text(0.5, 0.01, caption, ha='center', fontsize=6.5, color='0.3')
for ext in ('png', 'pdf'):
    fig.savefig(os.path.join(out, f'meancos_vs_delay.{ext}'), dpi=200)
plt.close(fig)

with open(os.path.join(out, 'meancos_vs_delay.csv'), 'w') as f:
    f.write('delay,meanCos_full_mean,meanCos_full_seedstd,meanCos_steady_mean,meanCos_steady_seedstd,meanCos_steady_timestd,' + ','.join(f'full_s{s}' for s in seeds) + ',' + ','.join(f'steady_s{s}' for s in seeds) + '\n')
    for i, d in enumerate(delays):
        f.write(f'{d},{full[i].mean():.6f},{sd_seed[i]:.6f},{m_st[i]:.6f},{sd_st[i]:.6f},{tstd[i]:.6f},' + ','.join(f'{v:.6f}' for v in full[i]) + ',' + ','.join(f'{v:.6f}' for v in st[i]) + '\n')
print('wrote figures to', out)
