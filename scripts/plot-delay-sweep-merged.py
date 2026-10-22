#!/usr/bin/env python3
"""Merge a coarse and a fine delay sweep (runs/*.json) and plot them together.

usage: python3 scripts/plot-delay-sweep-merged.py COARSE_DIR FINE_DIR THEORY_VISC_JSON OUTDIR
         [--tdiscard 120] [--lo 0.50] [--hi 1.25] [--rawdir OUTDIR/fine]
         [--seeds10 DIR --lo10 0.90 --hi10 1.25 --rawdir10 OUTDIR/seeds10]   (extra multi-seed sweep, merged in [lo10, hi10])
Outside [lo, hi] only coarse runs are used; inside, all coarse + fine runs at each delay.
Delays are rounded to 0.01 s for matching. The heatmap uses true bin edges (midpoints between
neighbouring delays), so the non-uniform spacing is drawn correctly.
Writes meancos_vs_delay.*, heatmap_log.*, states.*, summary_runs.csv, meancos_vs_delay.csv,
hist_avg.csv, states.csv; with --rawdir also fine summary/hist/timeseries CSVs (gzipped).
"""
import glob, gzip, json, os, sys
from collections import Counter
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

a = sys.argv[1:]
def opt(k, dflt, f=float):
    global a
    if k in a: i = a.index(k); v = f(a[i + 1]); del a[i:i + 2]; return v
    return dflt
TD = opt('--tdiscard', 120.0); LO = opt('--lo', 0.50); HI = opt('--hi', 1.25); RAW = opt('--rawdir', None, str)
S10 = opt('--seeds10', None, str); LO10 = opt('--lo10', 0.90); HI10 = opt('--hi10', 1.25); RAW10 = opt('--rawdir10', None, str)
CLABEL = opt('--coarselabel', 'coarse runs (Δt=0.05 s, 1 seed)', str); INITLBL = opt('--initlabel', 'ordered init', str); ZT = opt('--zoomtitle', None, str)
cdir, fdir, thf, out = a[:4]; os.makedirs(out, exist_ok=True)
TH = json.load(open(thf))['chosen']; NU = TH['nu']
th = json.load(open(os.path.join(cdir, 'theory.json')))
g = TH['grids'][f"{th['nx']}x{th['ny']}"]; W0, W1 = g['onset_phys_min'], g['onset_phys_max']
TMAX = th['tmax']; V0 = th.get('params', {}).get('selfPropV0', 0.05)
plt.rcParams.update({'font.size': 10, 'font.family': 'DejaVu Sans', 'axes.linewidth': 0.8, 'axes.spines.top': False,
                     'axes.spines.right': False, 'savefig.bbox': 'tight', 'legend.frameon': False})

def load(d, src):
    rr = []
    for f in sorted(glob.glob(os.path.join(d, 'runs', '*.json'))):
        r = json.load(open(f)); t = np.array(r['t']); m = t > TD + 1e-9
        mc = np.array(r['meanCos'])[m]; ph = np.array(r['phi'])[m]; H = np.array(r['hist'])[m]
        rr.append(dict(delay=round(r['delay'], 2), seed=r['seed'], src=src, mc=mc.mean(), tstd=mc.std(ddof=1), sd0=mc.std(),
                       phi=ph.mean(), H=H.mean(0), finite=r['finite'], wall=r['wallSeconds'], raw=r))
    return rr
coarse = load(cdir, 'coarse'); fine = load(fdir, 'fine')
fine = [r for r in fine if LO - 1e-9 <= r['delay'] <= HI + 1e-9]
s10 = [r for r in load(S10, 's10') if LO10 - 1e-9 <= r['delay'] <= HI10 + 1e-9] if S10 else []
runs = coarse + fine + s10
delays = sorted({r['delay'] for r in runs})
agg = []
for dl in delays:
    gg = [r for r in runs if r['delay'] == dl]; y = np.array([r['mc'] for r in gg]); n = len(y)
    sd = y.std(ddof=1) if n > 1 else 0.0
    agg.append(dict(delay=dl, n=n, mean=y.mean(), sd=sd, sem=sd / np.sqrt(n), H=np.mean([r['H'] for r in gg], 0), ys=y,
                    nc=sum(r['src'] == 'coarse' for r in gg), nf=sum(r['src'] == 'fine' for r in gg), ns=sum(r['src'] == 's10' for r in gg)))
x = np.array([q['delay'] for q in agg]); y = np.array([q['mean'] for q in agg])
lo_ = np.array([q['ys'].min() for q in agg]); hi_ = np.array([q['ys'].max() for q in agg])
xl = r'delay $t_{\mathrm{self}} = t_{\mathrm{nb}} = t$ (s)'; win = f'{TD:.0f}–{TMAX:.0f} s'

with open(os.path.join(out, 'summary_runs.csv'), 'w') as f:
    f.write('delay,seed,source,meanCos_window,meanCos_window_timestd,phi_window,finite,wall_s\n')
    for r in sorted(runs, key=lambda r: (r['delay'], r['src'], r['seed'])):
        f.write(f"{r['delay']:.2f},{r['seed']},{r['src']},{r['mc']:.6f},{r['tstd']:.6f},{r['phi']:.6f},{r['finite']},{r['wall']:.1f}\n")
with open(os.path.join(out, 'meancos_vs_delay.csv'), 'w') as f:
    f.write(f'# window t in ({TD:g},{TMAX:g}] s; coarse sweep 0-2 s step 0.05 (1 seed) + fine sweep {LO:g}-{HI:g} s step 0.01 (2 seeds)' + (f' + seeds10 sweep {LO10:g}-{HI10:g} s step 0.01 (10 new seeds)' if S10 else '') + '\n')
    f.write('delay,runs,coarse_runs,fine_runs,seeds10_runs,meanCos_mean,meanCos_sem,meanCos_std,meanCos_min,meanCos_max\n')
    for q in agg: f.write(f"{q['delay']:.2f},{q['n']},{q['nc']},{q['nf']},{q['ns']},{q['mean']:.6f},{q['sem']:.6f},{q['sd']:.6f},{q['ys'].min():.6f},{q['ys'].max():.6f}\n")
B = len(agg[0]['H']); ec = np.linspace(-1, 1, B + 1); cc = (ec[1:] + ec[:-1]) / 2
with open(os.path.join(out, 'hist_avg.csv'), 'w') as f:
    f.write('delay,cos_centre,P,runs\n')
    for q in agg:
        for k in range(B): f.write(f"{q['delay']:.2f},{cc[k]:.3f},{q['H'][k]:.6g},{q['n']}\n")

# (a) mean vs delay
fig, (ax, zx) = plt.subplots(1, 2, figsize=(11.5, 4.2), gridspec_kw=dict(width_ratios=[1, 1.25], wspace=0.18))
for A, xlim in ((ax, (0, 2)), (zx, (LO - 0.02, HI + 0.02))):
    A.axvspan(W0, W1, color='0.80', lw=0, zorder=0, label=f'linear-theory onset {W0:.2f}–{W1:.2f} s')
    m = np.array([q['n'] > 1 for q in agg])
    A.fill_between(x, np.where(m, lo_, y), np.where(m, hi_, y), color='#1f77b4', alpha=0.22, lw=0, zorder=1, label='min–max of runs')
    A.plot(x, y, '-', color='#0b3c78', lw=1.3, zorder=2, label='mean per delay')
    sdv = np.array([q['sd'] for q in agg])
    A.errorbar(x[m], y[m], yerr=sdv[m], fmt='none', ecolor='#0b3c78', elinewidth=0.8, capsize=1.5, zorder=4, label='± 1 std over runs')
    for src, kw in (('coarse', dict(marker='s', ms=4.5, color='#d62728', mfc='none', mew=1.0)),
                    ('fine', dict(marker='o', ms=2.8, color='#1f77b4', alpha=0.75, mew=0)),
                    ('s10', dict(marker='.', ms=2.5, color='#2ca02c', alpha=0.5, mew=0))):
        rr = [r for r in runs if r['src'] == src]
        A.plot([r['delay'] for r in rr], [r['mc'] for r in rr], ls='none', zorder=3,
               label=(CLABEL if src == 'coarse' else f'fine runs (Δt=0.01 s, 2 seeds)' if src == 'fine' else f'10-seed runs {LO10:.2f}–{HI10:.2f} s'), **kw)
    A.set_xlim(*xlim); A.set_ylim(-0.05, 1.03); A.set_xlabel(xl)
ax.set_ylabel(r'$\langle\cos\theta_{ij}\rangle$'); zx.set_title(f'zoom {LO:.2f}–{HI:.2f} s', fontsize=10)
ax.axvspan(LO, HI, color='#1f77b4', alpha=0.05, lw=0, zorder=0)
zx.legend(fontsize=7.5, loc='upper right')
fig.suptitle(f'Pairwise velocity correlation vs delay (ν = {NU:g}, v0 = {V0:g}, {th["nx"]}×{th["ny"]} unit square, {INITLBL}, window {win})', fontsize=10.5)
for e in ('png', 'pdf'): fig.savefig(os.path.join(out, f'meancos_vs_delay.{e}'), dpi=200)
plt.close(fig)

# (b) log heatmap with true edges
mid = (x[1:] + x[:-1]) / 2
ed = np.concatenate([[x[0] - (x[1] - x[0]) / 2], mid, [x[-1] + (x[-1] - x[-2]) / 2]])
P = np.array([q['H'] for q in agg])
fig, (ax, zx) = plt.subplots(1, 2, figsize=(12, 4.2), gridspec_kw=dict(width_ratios=[1, 1.25], wspace=0.12))
norm = LogNorm(vmin=1e-2, vmax=max(20, P.max()))
for A in (ax, zx):
    mm = A.pcolormesh(ed, ec, np.maximum(P.T, 1e-2), norm=norm, cmap='magma', shading='flat', rasterized=True)
    for v in (W0, W1): A.axvline(v, color='w', ls='--', lw=0.8, alpha=0.85)
    A.set_ylim(-1, 1); A.set_xlabel(xl)
ax.set_xlim(ed[0], ed[-1]); zx.set_xlim(LO - 0.005, HI + 0.005)
for v in (LO, HI): ax.axvline(v, color='c', ls=':', lw=0.8)
ax.set_ylabel(r'$\cos\theta_{ij}$'); zx.set_title(ZT or (f'zoom {LO:.2f}–{HI:.2f} s (2–3 runs per delay' + (f'; 12–13 in {LO10:.2f}–{HI10:.2f} s' if S10 else '') + ')'), fontsize=10)
ax.set_title('full range (cyan: fine-sweep window; white: theory onset)', fontsize=9)
cb = fig.colorbar(mm, ax=[ax, zx], pad=0.015); cb.set_label(r'$P(\cos\theta_{ij})$, run-averaged'); cb.outline.set_linewidth(0.5)
for e in ('png', 'pdf'): fig.savefig(os.path.join(out, f'heatmap_log.{e}'), dpi=200)
plt.close(fig)

# (c) states
key = lambda r: (round(r['mc'] / 0.005) * 0.005, round(r['phi'] / 0.005) * 0.005)
frac = []
for dl in delays:
    gg = [r for r in runs if r['delay'] == dl]; st = [r for r in gg if r['sd0'] <= 0.03]
    main = Counter(key(r) for r in st).most_common(1)[0][0] if st else None
    # 'main' = within 0.02 (in both <cos> and phi) of the most common rounded state; avoids splitting near-identical runs
    for r in gg: r['state'] = 'switching' if r['sd0'] > 0.03 else ('main' if abs(r['mc'] - main[0]) <= 0.02 and abs(r['phi'] - main[1]) <= 0.02 else 'other')
    frac.append((dl, len(gg), sum(r['state'] == 'main' for r in gg), sum(r['state'] == 'other' for r in gg),
                 sum(r['state'] == 'switching' for r in gg), main[0] if main else np.nan))
with open(os.path.join(out, 'states.csv'), 'w') as f:
    f.write('delay,seed,source,meanCos_window,meanCos_timestd,phi_window,state\n')
    for r in sorted(runs, key=lambda r: (r['delay'], r['src'], r['seed'])):
        f.write(f"{r['delay']:.2f},{r['seed']},{r['src']},{r['mc']:.6f},{r['sd0']:.6f},{r['phi']:.6f},{r['state']}\n")
fig, (ax, bx) = plt.subplots(2, 1, figsize=(8, 5.6), sharex=True, gridspec_kw=dict(height_ratios=[2.2, 1], hspace=0.08))
sty = {'main': dict(c='#0b3c78', marker='o', s=14, alpha=0.6), 'other': dict(c='#d62728', marker='D', s=18, alpha=0.8), 'switching': dict(c='#ff7f0e', marker='x', s=22, alpha=0.9)}
F = np.array(frac, float)
for st, kw in sty.items():
    rr = [r for r in runs if r['state'] == st]
    ax.scatter([r['delay'] for r in rr], [r['mc'] for r in rr], label=f'{st} ({len(rr)})', lw=1 if st == 'switching' else 0, **kw)
ax.axvspan(W0, W1, color='0.85', lw=0, zorder=0)
ax.set_ylim(-0.05, 1.03); ax.set_ylabel(r'$\langle\cos\theta_{ij}\rangle$ per run'); ax.legend(fontsize=8, loc='upper right')
ax.set_title('Steady-state classification per run (main = within 0.02 of most common state; switching = time std > 0.03)', fontsize=9)
wdt = np.diff(ed) * 0.9; b0 = np.zeros(len(F))
for col, c in ((2, '#0b3c78'), (3, '#d62728'), (4, '#ff7f0e')):
    bx.bar(F[:, 0], F[:, col] / F[:, 1], wdt, bottom=b0, color=c, alpha=0.75, lw=0); b0 += F[:, col] / F[:, 1]
bx.set_ylim(0, 1); bx.set_xlim(-0.03, 2.03); bx.set_ylabel('fraction'); bx.set_xlabel(xl)
for e in ('png', 'pdf'): fig.savefig(os.path.join(out, f'states.{e}'), dpi=200)
plt.close(fig)

if RAW:
    os.makedirs(RAW, exist_ok=True)
    fr = sorted(fine, key=lambda r: (r['delay'], r['seed']))
    with open(os.path.join(RAW, 'summary_fine.csv'), 'w') as f:
        f.write('delay,seed,meanCos_window,meanCos_window_timestd,phi_window,state,maxAbsU,finite,wall_s\n')
        for r in fr: f.write(f"{r['delay']:.2f},{r['seed']},{r['mc']:.6f},{r['tstd']:.6f},{r['phi']:.6f},{r['state']},{r['raw']['maxAbsU']:.6g},{r['finite']},{r['wall']:.1f}\n")
    with gzip.open(os.path.join(RAW, 'hist_fine_runs.csv.gz'), 'wt') as f:
        f.write('delay,seed,' + ','.join(f'P{k}' for k in range(B)) + f'  # window-mean P(cos) over t>{TD:g}, bin centres {cc[0]:.3f}..{cc[-1]:.3f}\n')
        for r in fr: f.write(f"{r['delay']:.2f},{r['seed']}," + ','.join(f'{v:.6g}' for v in r['H']) + '\n')
    with gzip.open(os.path.join(RAW, 'timeseries_fine.csv.gz'), 'wt') as f:
        f.write('delay,seed,t,meanCos,phi,speed,xi,' + ','.join(f'P{k}' for k in range(B)) + '\n')
        for r in fr:
            R = r['raw']
            for i, t in enumerate(R['t']):
                xi = R['xi'][i]; xs = '' if xi is None else f'{xi:.6g}'
                f.write(f"{r['delay']:.2f},{r['seed']},{t:g},{R['meanCos'][i]:.6f},{R['phi'][i]:.6f},{R['speed'][i]:.6f},{xs}," + ','.join(f'{v:.5g}' for v in R['hist'][i]) + '\n')
if RAW10 and s10:
    os.makedirs(RAW10, exist_ok=True); fr = sorted(s10, key=lambda r: (r['delay'], r['seed']))
    with open(os.path.join(RAW10, 'summary_seeds10.csv'), 'w') as f:
        f.write('delay,seed,meanCos_window,meanCos_window_timestd,phi_window,state,maxAbsU,finite,wall_s\n')
        for r in fr: f.write(f"{r['delay']:.2f},{r['seed']},{r['mc']:.6f},{r['tstd']:.6f},{r['phi']:.6f},{r['state']},{r['raw']['maxAbsU']:.6g},{r['finite']},{r['wall']:.1f}\n")
    with gzip.open(os.path.join(RAW10, 'hist_seeds10_runs.csv.gz'), 'wt') as f:
        f.write('delay,seed,' + ','.join(f'P{k}' for k in range(B)) + '\n')
        for r in fr: f.write(f"{r['delay']:.2f},{r['seed']}," + ','.join(f'{v:.6g}' for v in r['H']) + '\n')
    with gzip.open(os.path.join(RAW10, 'timeseries_seeds10.csv.gz'), 'wt') as f:
        f.write('delay,seed,t,meanCos,phi,speed,xi,' + ','.join(f'P{k}' for k in range(B)) + '\n')
        for r in fr:
            R = r['raw']
            for i, t in enumerate(R['t']):
                xi = R['xi'][i]; xs = '' if xi is None else f'{xi:.6g}'
                f.write(f"{r['delay']:.2f},{r['seed']},{t:g},{R['meanCos'][i]:.6f},{R['phi'][i]:.6f},{R['speed'][i]:.6f},{xs}," + ','.join(f'{v:.5g}' for v in R['hist'][i]) + '\n')
for q in agg: print(f"{q['delay']:.2f} n={q['n']} (c{q['nc']} f{q['nf']}) mean={q['mean']:.4f} runs=" + ' '.join(f'{v:.4f}' for v in q['ys']))
