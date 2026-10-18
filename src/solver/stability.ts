/**
 * Linear stability of the two-delay viscosity / alignment terms.
 *
 * Every delayed viscosity component used here splits into a "self" part, which
 * reads the point's own velocity t_self ago, and a "neighbour" part, which
 * reads the neighbours' velocities t_nb ago. Linearised about rest, a single
 * Fourier mode û_k obeys the classic two-delay equation
 *
 *   û_k′(t) = −A û_k(t − t_self) + B(k) û_k(t − t_nb)
 *
 * (Hayes 1950 for one delay; Bellman & Cooke 1963, ch. 11, for two), with
 *
 *   local ν∇² (5-point stencil):  A = 4ν/h²,  B = (2ν/h²)(cos k_x h + cos k_y h) = A·c,
 *                                  c = (cos k_x h + cos k_y h)/2 ∈ [−1, 1]
 *   kernel S∫K(u(y) − u(x)):      A = S,      B = S·K̂(k) ∈ [S·K̂_min, S]
 *   fractional −ν_s(−Δ)^s:        A = ν_s G₀, B = ν_s (G₀ − λ(k)^s),  G₀ = ⟨λ^s⟩ (diagonal of the operator)
 *
 * Solutions e^{λt} satisfy the characteristic equation
 *
 *   g(λ) = λ + A e^{−λ t_self} − B e^{−λ t_nb} = 0.
 *
 * Every root with Re λ ≥ 0 lies in the half disc |λ| ≤ |A| + |B| (because
 * |e^{−λτ}| ≤ 1 there), so a dense set of Newton starts covering that half disc
 * (and a strip to its left) finds the rightmost root reliably; real roots are
 * bracketed separately on the real axis. This module has no DOM dependencies.
 */

export interface Root { re: number; im: number }

/** g(λ) and g′(λ) for complex λ = (x, y). */
function evalG(A: number, B: number, t1: number, t2: number, x: number, y: number): [number, number, number, number] {
  const e1 = Math.exp(-x * t1), c1 = e1 * Math.cos(-y * t1), s1 = e1 * Math.sin(-y * t1);
  const e2 = Math.exp(-x * t2), c2 = e2 * Math.cos(-y * t2), s2 = e2 * Math.sin(-y * t2);
  const gr = x + A * c1 - B * c2;
  const gi = y + A * s1 - B * s2;
  const dr = 1 - A * t1 * c1 + B * t2 * c2;
  const di = -A * t1 * s1 + B * t2 * s2;
  return [gr, gi, dr, di];
}

/** Newton iteration on g from (x, y); null if it does not converge. */
export function newtonRoot(A: number, B: number, t1: number, t2: number, x: number, y: number, its = 60): Root | null {
  for (let it = 0; it < its; it++) {
    const [gr, gi, dr, di] = evalG(A, B, t1, t2, x, y);
    const den = dr * dr + di * di;
    if (!(den > 1e-300)) return null;
    const sx = (gr * dr + gi * di) / den, sy = (gi * dr - gr * di) / den;
    x -= sx; y -= sy;
    if (!Number.isFinite(x) || !Number.isFinite(y)) return null;
    if (Math.hypot(sx, sy) < 1e-12 * (1 + Math.hypot(x, y))) {
      const [fr, fi] = evalG(A, B, t1, t2, x, y);
      const scale = 1 + Math.abs(x) + Math.abs(y) + Math.abs(A) + Math.abs(B);
      return Math.hypot(fr, fi) < 1e-8 * scale ? { re: x, im: Math.abs(y) < 1e-9 * scale ? 0 : Math.abs(y) } : null;
    }
  }
  return null;
}

/**
 * Rightmost root of λ = −A e^{−λ t_self} + B e^{−λ t_nb} (A, B real, delays ≥ 0).
 * Without delays the root is simply B − A.
 */
export function rightmostRoot(A: number, B: number, tSelf: number, tNb: number): Root {
  tSelf = Math.max(0, tSelf); tNb = Math.max(0, tNb);
  if (tSelf === 0 && tNb === 0) return { re: B - A, im: 0 };
  const R = Math.abs(A) + Math.abs(B);
  if (!(R > 0)) return { re: 0, im: 0 };
  let best: Root | null = null;
  const take = (r: Root | null) => {
    if (!r) return;
    if (!best || r.re > best.re + 1e-10 * (1 + R) || (Math.abs(r.re - best.re) <= 1e-10 * (1 + R) && r.im < best.im)) best = r;
  };
  // (1) real roots: bracket sign changes of g on [xLo, R]
  const xLo = -2 * R - 1;
  const g = (x: number) => x + A * Math.exp(-x * tSelf) - B * Math.exp(-x * tNb);
  const nScan = 400;
  let xp = R + 1e-9, gp = g(xp);
  for (let n = 1; n <= nScan; n++) {
    const xq = R - ((R - xLo) * n) / nScan, gq = g(xq);
    if (gp === 0) { take({ re: xp, im: 0 }); break; }
    if (gp * gq < 0) {
      let a = xq, b = xp;
      for (let it = 0; it < 100; it++) { const m = 0.5 * (a + b); if (g(m) * g(a) <= 0) b = m; else a = m; }
      take({ re: 0.5 * (a + b), im: 0 });
      break; // scanning from the right: the first sign change is the rightmost real root
    }
    xp = xq; gp = gq;
  }
  // (2) complex roots: Newton from a grid over the half disc and a strip to its left
  const tMax = Math.max(tSelf, tNb);
  const yMax = Math.max(R, 4 * Math.PI / tMax) * 1.2;
  const NX = 14, NY = 36;
  for (let a = 0; a < NX; a++) {
    const x0 = R - ((R - xLo) * a) / (NX - 1);
    for (let b = 0; b < NY; b++) {
      const y0 = (yMax * (b + 0.5)) / NY;
      take(newtonRoot(A, B, tSelf, tNb, x0, y0));
    }
  }
  return best ?? { re: B - A, im: 0 };
}

export type Regime = 'monotone' | 'damped' | 'unstable-osc' | 'unstable-real' | 'neutral';

/** Classify a rightmost root. */
export function regimeOf(r: Root, scale = 1): Regime {
  const eps = 1e-7 * (1 + Math.abs(scale));
  if (r.re > eps) return r.im > 0 ? 'unstable-osc' : 'unstable-real';
  if (r.re >= -eps) return 'neutral';
  return r.im > 0 ? 'damped' : 'monotone';
}

export interface ModeScan {
  /** Mode parameter of the most unstable mode (c for local viscosity, K̂ for kernels). */
  param: number;
  A: number;
  B: number;
  root: Root;
  regime: Regime;
}

/**
 * Most unstable mode over B = A·q for q in [qMin, qMax] (sampled at n points,
 * the end points always included). Modes whose rightmost root is within
 * `neutralTol` of zero (e.g. the uniform mode with momentum conservation) are
 * only reported if nothing else grows or decays more slowly.
 */
export function scanModes(
  A: number, qMin: number, qMax: number, tSelf: number, tNb: number, n = 41,
  exclude: (q: number) => boolean = () => false,
): ModeScan {
  let best: ModeScan | null = null;
  for (let m = 0; m < n; m++) {
    const q = n === 1 ? qMax : qMin + ((qMax - qMin) * m) / (n - 1);
    if (exclude(q)) continue;
    const B = A * q;
    const root = rightmostRoot(A, B, tSelf, tNb);
    if (!best || root.re > best.root.re + 1e-12) best = { param: q, A, B, root, regime: regimeOf(root, A) };
  }
  return best ?? { param: qMax, A, B: A * qMax, root: { re: 0, im: 0 }, regime: 'neutral' };
}
