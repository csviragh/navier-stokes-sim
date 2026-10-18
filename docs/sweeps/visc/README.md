# Delay sweep with undelayed viscosity ν = 0.01 (Eulerian)

Eulerian (fixed-point) delay sweep regularised by a small **undelayed** local
viscosity, so the Doppler-driven short-wave instability of the ν = 0 Eulerian
sweep is cut off and the onset is the same on 128 × 72 and 219 × 128.

See the section *Rerun with undelayed viscosity ν = 0.01* in the top-level
`README.md` for parameters, the steady-state table, findings and how to
reproduce.

## Files

| file | content |
| --- | --- |
| `heatmap_log_128x72.{png,pdf}` | P(cos θ_ij) vs delay, 128 × 72 |
| `heatmap_log_219x128.{png,pdf}` | same, 219 × 128 |
| `meancos_vs_delay.{png,pdf,csv}` | ⟨cos θ_ij⟩ vs delay on both grids (mean ± seed std) |
| `summary_{128x72,219x128}.csv` | per-run steady / full-run means |
| `hist_avg_{128x72,219x128}.csv` | P averaged over seeds and the steady window |
| `theory_visc.json` | linear Doppler + ν theory (onset scan vs ν) |

Parameters (identical to the ν = 0 Eulerian sweep except ν and `delayLocal`):
periodic torus; Cucker–Smale S = 2/s, R = 0.05, β = 1; self-propulsion τ = 1 s,
v₀ = 0.5; t_self = t_nb ∈ [0, 2]; ν = 0.01 **not delayed**; 120 s / run;
seeds 1000, 1017, 1034; 40 GS viscosity iterations.
