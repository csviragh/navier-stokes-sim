#!/usr/bin/env python3
"""Plots for the viscosity-regularised Eulerian delay sweep (ν > 0, undelayed).

usage: python3 scripts/plot-delay-sweep-visc.py C128_DIR F219_DIR OLD_NU0_DIR THEORY_VISC_JSON OUTDIR
  C128_DIR / F219_DIR: --out dirs of scripts/delay-sweep.ts (reads runs/*.json, so partial sweeps work)
  OLD_NU0_DIR: the ν = 0 Eulerian sweep (summary.csv and fullres-219x128/summary.csv), drawn faintly
  THEORY_VISC_JSON: output of scripts/delay-sweep-theory-visc.py
Writes heatmap_log_128x72.{png,pdf}, heatmap_log_219x128.{png,pdf}, meancos_vs_delay.{png,pdf},
meancos_vs_delay.csv, summary_128x72.csv, summary_219x128.csv, hist_avg_<grid>.csv.
"""
import csv, glob, json, os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

c128, f219, old, thf, out = sys.argv[1:6]
os.makedirs(out, exist_ok=True)
TH = json.load(open(thf))['chosen']
NU = TH['nu']
plt.rcParams.update({'font.size': 10, 'font.family': 'DejaVu Sans', 'axes.linewidth': 0.8,
                     'xtick.direction': 'out', 'ytick.direction': 'out', 'savefig.bbox': 'tight'})


def load_runs(d):
    th = json.load(open(os.path.join(d, 'theory.json')))
    runs = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(d, 'runs', '*.json')))]
    return th, runs


def per_run(r, tmax):
    t = np.array(r['t']); mc = np.array(r['meanCos']); H = np.array(r['hist'])
    st = t > tmax / 2
    return dict(delay=r['delay'], seed=r['seed'], full=mc.mean(), steady=mc[st].mean(), steady_t_std=mc[st].std(ddof=1),
                H_full=H.mean(axis=0), H_st=H[st].mean(axis=0), finite=r['finite'], wall=r['wallSeconds'])


def aggregate(d):
    th, runs = load_runs(d)
    tmax = th['tmax']
    rr = [per_run(r, tmax) for r in runs]
    delays = sorted({x['delay'] for x in rr})
    agg = []
    for dl in delays:
        g = [x for x in rr if x['delay'] == dl]
        st = np.array([x['steady'] for x in g]); fu = np.array([x['full'] for x in g])
        agg.append(dict(delay=dl, n=len(g), steady=st.mean(), steady_sd=st.std(ddof=1) if len(g) > 1 else 0.0,
                        full=fu.mean(), full_sd=fu.std(ddof=1) if len(g) > 1 else 0.0,
                        tstd=float(np.sqrt(np.mean([x['steady_t_std'] ** 2 for x in g]))),
                        H_full=np.mean([x['H_full'] for x in g], axis=0), H_st=np.mean([x['H_st'] for x in g], axis=0),
                        per_seed={x['seed']: x['steady'] for x in g}, wall=np.mean([x['wall'] for x in g])))
    return th, rr, agg


def onset_lines(key):
    g = TH['grids'][key]
    return g['onset_phys_min'], g['onset_phys_max'], g['onset_SL_min'], g['onset_SL_max']


def caption(th, extra=''):
    p = th['params']
    return (f"delays read at fixed points (Eulerian); ν = {NU:g} (not delayed); CS kernel S = {th['S']}/s, R = {p['kernelRange']}, "
            f"β = {p['kernelBeta']}; self-propulsion τ = {p['selfPropTau']} s, v₀ = {p['selfPropV0']}; torus {th['nx']}×{th['ny']}{extra}")


results = {}
for d in (c128, f219):
    th, rr, agg = aggregate(d)
    key = f"{th['nx']}x{th['ny']}"
    results[key] = (th, rr, agg)
    with open(os.path.join(out, f'summary_{key}.csv'), 'w') as f:
        f.write('delay,seed,meanCos_full,meanCos_steady,meanCos_steady_timestd,finite,wall_s\n')
        for x in sorted(rr, key=lambda x: (x['delay'], x['seed'])):
            f.write(f"{x['delay']},{x['seed']},{x['full']:.6f},{x['steady']:.6f},{x['steady_t_std']:.6f},{x['finite']},{x['wall']:.1f}\n")
    B = len(agg[0]['H_full']); ec = np.linspace(-1, 1, B + 1); cc = 0.5 * (ec[1:] + ec[:-1])
    with open(os.path.join(out, f'hist_avg_{key}.csv'), 'w') as f:
        f.write('delay,cos_centre,P_full,P_steady,seeds\n')
        for a in agg:
            for k in range(B): f.write(f"{a['delay']},{cc[k]:.3f},{a['H_full'][k]:.6g},{a['H_st'][k]:.6g},{a['n']}\n")
    # heatmap (log colour), time average and steady state
    delays = np.array([a['delay'] for a in agg])
    step = np.min(np.diff(delays)) if len(delays) > 1 else 0.05
    ed = np.concatenate([[delays[0] - step / 2], delays + step / 2])
    pmin, pmax, smin, smax = onset_lines(key)
    nseed = (min(a['n'] for a in agg), max(a['n'] for a in agg))
    fig, axs = plt.subplots(1, 2, figsize=(10, 4.3), sharey=True)
    fig.subplots_adjust(left=0.07, right=0.9, bottom=0.2, top=0.86, wspace=0.06)
    for ax, P, title in ((axs[0], np.array([a['H_full'] for a in agg]), f"time average 0–{th['tmax']:.0f} s"),
                         (axs[1], np.array([a['H_st'] for a in agg]), f"steady state {th['tmax']/2:.0f}–{th['tmax']:.0f} s")):
        m = ax.pcolormesh(ed, ec, np.maximum(P.T, 1e-2), norm=LogNorm(vmin=1e-2, vmax=max(20, P.max())), cmap='magma', shading='flat')
        hs = [ax.axvspan(pmin, pmax, color='#7CFC00', alpha=0.18, lw=0, label=f'ν-theory onset over flock headings: {pmin:.2f}–{pmax:.2f} s'),
              ax.axvline(smin, color='#7CFC00', ls='--', lw=1.3, label=f'incl. advection damping of this grid: {smin:.2f} s'),
              ax.axvline(np.pi / 4, color='w', ls=':', lw=1.1, label=f'rest fluid, $St=\\pi/2$: {np.pi/4:.3f} s')]
        ax.set_title(title, fontsize=10)
        ax.set_xlabel(r'delay $t_{\mathrm{self}} = t_{\mathrm{nb}} = t$ (s)'); ax.set_xlim(ed[0], ed[-1])
    axs[0].legend(handles=hs, loc='lower right', fontsize=7, framealpha=0.75, facecolor='0.15', labelcolor='w', edgecolor='none')
    axs[0].set_ylabel(r'pairwise alignment $\cos\theta_{ij} = \hat v_i\cdot\hat v_j$')
    cb = fig.colorbar(m, cax=fig.add_axes([0.915, 0.2, 0.015, 0.66])); cb.set_label(r'$P(\cos\theta_{ij})$ (log)')
    fig.suptitle(f'Distribution of pairwise velocity correlations vs delay, ν = {NU:g}, {th["nx"]}×{th["ny"]}', fontsize=11)
    ns = (f'{nseed[1]} seed' + ('s' if nseed[1] > 1 else '')) if nseed[0] == nseed[1] else f'{nseed[0]}–{nseed[1]} seeds (partial)'
    fig.text(0.5, 0.02, caption(th, f'; {ns} × {th["tmax"]:.0f} s'), ha='center', fontsize=7.5, color='0.3')
    for ext in ('png', 'pdf'): fig.savefig(os.path.join(out, f'heatmap_log_{key}.{ext}'), dpi=200)
    plt.close(fig)


def load_old(path):
    if not os.path.exists(path): return None
    by = {}
    for r in csv.DictReader(open(path)): by.setdefault(float(r['delay']), []).append(float(r['meanCos_steady']))
    x = sorted(by); return np.array(x), np.array([np.mean(by[v]) for v in x])

fig, ax = plt.subplots(figsize=(7.2, 4.6))
fig.subplots_adjust(bottom=0.3, top=0.92, left=0.11, right=0.97)
ax.axhline(0, color='0.6', lw=0.6)
pmin, pmax, _, _ = onset_lines('128x72')
ax.axvspan(pmin, pmax, color='#2ca02c', alpha=0.12, lw=0, label=f'ν-theory onset (both grids, over flock headings): {pmin:.2f}–{pmax:.2f} s')
for key, ls in (('128x72', '-.'), ('219x128', (0, (3, 1, 1, 1)))):
    s = onset_lines(key)[2]
    ax.axvline(s, color='#2ca02c', ls=ls, lw=1.0, label=f'+ advection damping, {key.replace("x", "×")}: {s:.2f} s')
ax.axvline(np.pi / 4, color='0.3', ls=':', lw=1.0, label=f'rest fluid $St=\\pi/2$: {np.pi/4:.3f} s')
for path, lab, mk in ((os.path.join(old, 'summary.csv'), 'ν = 0, 128×72', 's'), (os.path.join(old, 'fullres-219x128', 'summary.csv'), 'ν = 0, 219×128', 'o')):
    o = load_old(path)
    if o is not None:
        ax.plot(o[0], o[1], mk + '-', color='0.55', alpha=0.45, ms=3.5, lw=0.9, mfc='none' if '219' in lab else '0.55', label=f'{lab} (earlier sweep)')
for key, c, mk, ls in (('128x72', '#1f77b4', 's', '-'), ('219x128', '#d62728', 'o', '--')):
    th, rr, agg = results[key]
    x = np.array([a['delay'] for a in agg]); y = np.array([a['steady'] for a in agg]); e = np.array([a['steady_sd'] for a in agg])
    n = (min(a['n'] for a in agg), max(a['n'] for a in agg))
    nl = (f'{n[1]} seed' + ('s' if n[1] > 1 else '')) if n[0] == n[1] else f'{n[0]}–{n[1]} seeds, partial'
    ax.errorbar(x, y, yerr=e, fmt=mk + ls, color=c, ms=4 if key == '128x72' else 6, mfc=c if key == '128x72' else 'none', mew=1.3,
                lw=1.2, capsize=2, label=f'ν = {NU:g}, {key.replace("x", "×")} ({nl}; bars: seed std)')
ax.set_xlabel(r'delay $t_{\mathrm{self}} = t_{\mathrm{nb}} = t$ (s)')
ax.set_ylabel(r'steady-state $\langle\cos\theta_{ij}\rangle$ (60–120 s)')
ax.set_ylim(-0.05, 1.05); ax.set_xlim(-0.03, 2.03)
ax.set_title(f'Average pairwise velocity correlation vs delay, with viscosity ν = {NU:g}', fontsize=11)
ax.legend(fontsize=7, frameon=False, loc='upper center', bbox_to_anchor=(0.5, -0.13), ncol=2)
for ext in ('png', 'pdf'): fig.savefig(os.path.join(out, f'meancos_vs_delay.{ext}'), dpi=200)
plt.close(fig)

with open(os.path.join(out, 'meancos_vs_delay.csv'), 'w') as f:
    f.write('grid,delay,seeds,meanCos_steady_mean,meanCos_steady_seedstd,meanCos_steady_timestd,meanCos_full_mean,meanCos_full_seedstd\n')
    for key, (th, rr, agg) in results.items():
        for a in agg: f.write(f"{key},{a['delay']},{a['n']},{a['steady']:.6f},{a['steady_sd']:.6f},{a['tstd']:.6f},{a['full']:.6f},{a['full_sd']:.6f}\n")
for key, (th, rr, agg) in results.items():
    print(key, ' '.join(f"{a['delay']:.2f}:{a['steady']:.3f}({a['n']})" for a in agg))
print('wrote figures to', out)
