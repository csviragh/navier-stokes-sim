/** Principal branch W₀ of the Lambert W function for real z ≥ −∞ (complex result for z < −1/e). */
export function lambertW0(z: number): { re: number; im: number } {
  // complex Newton on f(w) = w e^w − z
  let wr: number, wi: number;
  const e = Math.E;
  const q = 2 * (1 + e * z);
  if (z < 1) {
    // branch-point series start: w ≈ −1 + p − p²/3, p = sqrt(2(1 + e z)) (complex for z < −1/e)
    const pr = q >= 0 ? Math.sqrt(q) : 0, pi = q >= 0 ? 0 : Math.sqrt(-q);
    wr = -1 + pr - (pr * pr - pi * pi) / 3;
    wi = pi - (2 * pr * pi) / 3;
  } else { wr = Math.log(z) - Math.log(Math.log(z) + 1e-300); wi = 0; }
  for (let it = 0; it < 100; it++) {
    const ex = Math.exp(wr), cr = ex * Math.cos(wi), ci = ex * Math.sin(wi); // e^w
    const fr = wr * cr - wi * ci - z, fi = wr * ci + wi * cr; // w e^w − z
    const dr = cr * (wr + 1) - ci * wi, di = ci * (wr + 1) + cr * wi; // e^w (w + 1)
    const den = dr * dr + di * di;
    const sr = (fr * dr + fi * di) / den, si = (fi * dr - fr * di) / den;
    wr -= sr; wi -= si;
    if (Math.hypot(sr, si) < 1e-15 * (1 + Math.hypot(wr, wi))) break;
  }
  return { re: wr, im: Math.abs(wi) };
}

/** Rightmost root of λ = −a e^{−λ τ}: λ = W₀(−aτ)/τ. */
export function delayRoot(a: number, tau: number): { re: number; im: number } {
  const w = lambertW0(-a * tau);
  return { re: w.re / tau, im: w.im / tau };
}

/**
 * Rightmost root of the Doppler-shifted delay equation with an undelayed damping g
 * (viscosity ν k², …): μ + g = −a e^{−μ τ} e^{iφ} (φ = k·U τ, Eulerian read of a
 * flock moving at U, μ in the comoving frame). Newton from a grid of starts.
 */
export function dopplerRoot(a: number, g: number, tau: number, phi: number): { re: number; im: number } {
  let best = { re: -Infinity, im: 0 };
  for (let sr = -6; sr <= 6; sr += 1.5) for (let si = -40; si <= 40; si += 2) {
    let mr = sr, mi = si, ok = false;
    for (let it = 0; it < 80; it++) {
      // f = μ + g + a e^{−μτ + iφ},  f' = 1 − aτ e^{−μτ + iφ}
      const ex = a * Math.exp(-mr * tau), ar = -mi * tau + phi;
      const er = ex * Math.cos(ar), ei = ex * Math.sin(ar);
      const fr = mr + g + er, fi = mi + ei;
      const dr = 1 - tau * er, di = -tau * ei;
      const den = dr * dr + di * di;
      const xr = (fr * dr + fi * di) / den, xi = (fi * dr - fr * di) / den;
      mr -= xr; mi -= xi;
      if (!Number.isFinite(mr) || !Number.isFinite(mi)) break;
      if (Math.hypot(xr, xi) < 1e-13) { ok = true; break; }
    }
    if (ok && mr > best.re) best = { re: mr, im: mi };
  }
  return best;
}
