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
