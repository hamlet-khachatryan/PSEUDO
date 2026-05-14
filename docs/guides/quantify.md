---
title: Quantify
parent: Guides
nav_order: 2
---

# Quantify Guide

The **Quantify** module analyses the ensemble of STOMP maps produced by `pseudo-debias`. It separates true crystallographic signal from bias and outputs debiased CCP4 maps.

---

## Quick start

```bash
pseudo-quantify --input_path /scratch/results/my_experiment
```

The command auto-detects all experiments inside the workspace. For a single crystal, point it at the crystal subdirectory:

```bash
pseudo-quantify --input_path /scratch/results/my_experiment/target_5e5z
```

---

## CLI options

| Flag | Short | Default | Description                                     |
|---|---|---|-------------------------------------------------|
| `--input_path` | `-p` | *required* | Workspace root produced by `pseudo-debias`.     |
| `--stem` | `-s` | auto | Explicit experiment stem (inferred if omitted). |
| `--k_factor` | `-k` | `1.0` | Radius multiplier K for atom ownership spheres. Set to `0` to skip bias removal and produce a plain ensemble average. |
| `--map_cap` | `-c` | `50` | Limit to the first N maps. |
| `--null_fit_method` | `-m` | `truncated` | Null-distribution fitting method. `truncated` uses truncated MLE on the left half of the SNR distribution; `full` uses unrestricted `t.fit`. |
| `--force` | `-f` | `False` | Overwrite existing `quantify_results/`.         |

---

## Python API

```python
from quantify.api import run_quantification

run_quantification(
    input_path="/scratch/results/my_experiment",
    k_factor=1.0,
    map_cap=50,
    force=False,
    null_fit_method="truncated",
)
```

---

## Required input layout

`pseudo-quantify` expects the directory structure created by `pseudo-debias`:

```
<crystal>/
├── processed/
│   └── {stem}_updated.pdb
├── metadata/
│   └── {stem}_omission_map.json
└── results/
    ├── {stem}_0/{stem}_0.mtz
    ├── {stem}_1/{stem}_1.mtz
    └── ...
```

---

## Outputs

Results are written to `<crystal>/quantify_results/k_{k}_cap_{cap}/`:

| File | Description |
|---|---|
| `{stem}_signal.ccp4` | Mean debiased density (true signal) |
| `{stem}_noise.ccp4` | Std-dev of cleaned densities (local noise) |
| `{stem}_snr.ccp4` | Signal-to-noise ratio — input to `pseudo-analyse` |
| `{stem}_p_value.ccp4` | Voxel-wise significance (t-test vs bulk solvent) |

A null-distribution parameter file is also saved to `metadata/` for use by `pseudo-analyse` to set a data-driven significance threshold.

---

## Algorithm overview

### Ownership

For each voxel v, atoms within distance `R(element, resolution) × k_factor` are identified as *owners*. This is determined via a KD-tree of atom positions and produces a binary status matrix **S** (shape N\_maps × N\_owners): 1 = atom present in that map, 0 = omitted.

### Bias estimation and removal

```text
For each owner j:
    if S[:,j] has both 0s and 1s:
        B_j = mean(D[S[:,j]=1]) − mean(D[S[:,j]=0])

CleanedVector = D − S · B
Signal = mean(CleanedVector)
Noise  = std(CleanedVector)
```

### Plain average mode (`k=0`)

When `--k_factor 0` is passed, ownership assignment and margianal bias removal are skipped. Each voxel is simply averaged across all maps in the ensemble:

```text
Signal = mean(D)
Noise  = std(D)
SNR    = Signal / Noise
```

---

## Null fitting

`pseudo-quantify` fits a Student-t distribution to ~20,000 SNR values sampled from within the protein mask. The fitted null is saved to `metadata/` and consumed by `pseudo-analyse` to derive a data-driven significance threshold.

### `truncated` (default)

Signal contamination in the null sample is one-sided: ordered waters and unmodelled density inflate only the right tail of the background SNR distribution. The truncated MLE method exploits this by fitting only the left half of the distribution (samples at or below the empirical mode) with a likelihood correction for the truncation:

```
ℓ(ν, μ, σ) = Σ_{z ≤ b} log f(z; ν, μ, σ)  −  n_L · log F(b; ν, μ, σ)
```

where `b` is the KDE mode and `F` is the t-distribution CDF. 

Two QC values are logged alongside the fitted parameters:

- **T_{α=0.05}** — the fitted significance threshold at the default alpha level.
- **π̂₀** — Efron's null fraction estimate (minimum of KDE/model-density ratio near the mode). Values close to 1.0 indicate a clean background; lower values flag signal contamination in the null sample.

If the optimizer fails or fewer than 1000 samples fall below the truncation point, the method falls back to `full` with a warning.

### `full` 

Unrestricted `scipy.stats.t.fit` on all background samples. Sufficient for low-resolution structures with few ordered waters, but becomes progressively biased as water content increases, shifted location, and an elevated threshold. Retained for backward comparison.

---

## HPC / SLURM submission

For large screening workspaces, wrap the command in `sbatch` and use `--num_processes` to parallelise across crystals within the job:

```bash
sbatch --partition cs05r \
       --cpus-per-task 8 \
       --mem-per-cpu 5G \
       --time 3-00:00:00 \
       --wrap "pseudo-quantify --input_path /scratch/results/my_screen \
                               --num_processes 8"
```

`--num_processes` controls how many crystals are processed in parallel inside the job. Set it to match `--cpus-per-task`.

---

## Parameter reference

See [Configuration Reference — Quantify](../reference#quantify-pseudo-quantify).