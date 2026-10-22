#!/usr/bin/env python3
"""Polar / nematic order of final fields (runs/*.field.bin) per delay; detects lanes (nematic ~1) vs vortex lattice (both ~0).
usage: lattice-check.py RUNDIR OUT.csv   -> per run: delay,seed,polar,nematic,frac_cos_gt_0.9 (window mean from json), frac_cos_lt_-0.9"""
import sys, glob, json, os, numpy as np
d, out = sys.argv[1:3]; rows = []
for p in sorted(glob.glob(os.path.join(d, 'runs', '*.field.bin'))):
    n = int(round((np.fromfile(p, np.float32).size / 2) ** .5)); u, v = np.fromfile(p, np.float32).reshape(2, n, n)
    th = np.arctan2(v, u); r = json.load(open(p.replace('.field.bin', '.json')))
    k = [i for i, t in enumerate(r['t']) if t > 120]; H = np.mean([r['hist'][i] for i in k], 0); B = len(H); c = np.linspace(-1, 1, B + 1); cc = (c[1:] + c[:-1]) / 2; w = H / H.sum()
    rows.append((round(r['delay'], 2), r['seed'], abs(np.exp(1j * th).mean()), abs(np.exp(2j * th).mean()), w[cc > 0.9].sum(), w[cc < -0.9].sum()))
with open(out, 'w') as f:
    f.write('delay,seed,polar_final,nematic_final,P_cos_gt_0.9,P_cos_lt_-0.9\n')
    for q in rows: f.write('%.2f,%d,%.4f,%.4f,%.4f,%.4f\n' % q)
