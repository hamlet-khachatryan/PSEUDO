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
| `--end` | | `False` | Also compute absolute-scale END maps (see below). |
| `--delta` | | `False` | Also compute delta density maps μ − model (see below). |
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

## END maps (absolute e⁻/Å³ scale)

STOMP realisation maps drop the h=0 (F₀₀₀) Fourier term, so each realisation
map has zero spatial mean and the σ-scaled outputs are on an arbitrary scale.
**END (Electron Number Density)** mode restores the absolute electron-density
scale by injecting the per-realisation total electron count:

```text
ρ_END^(k)(r) = F₀₀₀^(k) / V + ρ^(k)(r)
```

F₀₀₀^(k) is recomputed **per realisation** from the model with that
realisation's omit set removed:

- `use_bulk_and_scaling = false` (default): `F₀₀₀^(k) = Σ_j q_j Z_j` (atomic electrons in the cell)
- `use_bulk_and_scaling = true`: atomic sum **plus** a flat bulk-solvent term `k_sol · V_solvent^(k)`

The bulk-solvent decision is read from `metadata/{stem}_run_config.json` written
at debias time. If that file is **absent** — legacy runs predating run-config
support, all produced without bulk solvent — END defaults to no bulk solvent. A
file that is present but malformed is still a hard error.

### Inline (during quantification)

```bash
pseudo-quantify --input_path /scratch/results/my_screen --end
```

In addition to the σ-scaled maps, this writes, in each
`quantify_results/k_{k}_cap_{cap}/`:

| File | Description |
|---|---|
| `{stem}_end_mean.ccp4` | μ_END: mean of ρ_END over the ensemble (e⁻/Å³) |
| `{stem}_end_std.ccp4` | σ_END: voxel-wise standard deviation (e⁻/Å³) |
| `{stem}_end_snr.ccp4` | SNR_END = μ_END / σ_END |
| `{stem}_end_rho_{k}.ccp4` | per-realisation ρ_END^(k) |

### Standalone (`pseudo-end`)

To compute END maps on an already-completed run **without** re-running STOMP or
quantification:

```bash
pseudo-end --input_path /scratch/results/my_screen/my_crystal --stem my_crystal
```

It reads the persisted run config and realisation maps, and writes the same END
outputs into the matching `quantify_results/k_{k}_cap_{cap}/`. Rerunning is
idempotent.

### σ-scaled MUSE vs END — different statistics, on purpose

The σ-scaled μ/σ/SNR maps and the END μ/σ/SNR maps are **distinct statistics and
are both produced**. The σ-scaled SNR is a unit-free, within-dataset quantity:
use it for thresholding and MUSE scoring inside one crystal. The END maps live
on the absolute e⁻/Å³ scale, making them suitable for physical interpretation
and cross-dataset comparison. They are not interchangeable, and END never
replaces the σ-scaled outputs.

---

## Delta density maps (μ − model)

A **delta density map** subtracts the density predicted by the *perturbation
model* — the refined model `{stem}_updated` that every Phenix omit-map job was
run against — from the STOMP μ map:

```text
δ(r) = μ(r) − ρ_model(r)
```

Positive δ marks density the model does **not** explain (unmodelled features,
ordered solvent); negative δ marks model atoms the ensemble does **not**
support. It is an ensemble-averaged, F\_o − F\_c-style residual.

`ρ_model` is computed from `{stem}_updated` with X-ray form factors
(`gemmi.DensityCalculatorX`), truncated to the map resolution and placed on the
**exact ensemble grid**. Two flavours are produced, one per μ convention:

| Flavour | Definition | Scale | When |
|---|---|---|---|
| **σ-scaled** | `δ_σ = μ_σ − (a·ρ_model + b)` | arbitrary | always, with `--delta` |
| **END** | `δ_END = μ_END − ρ_model` | absolute e⁻/Å³ | with `--delta --end` |

Because the σ-scaled μ map (`{stem}_mean.ccp4`) is zero-mean and on an arbitrary
scale, `ρ_model` is least-squares rescaled `(a, b)` to it over the protein
region before subtraction. The END μ map is already on the absolute scale, so
`δ_END` is a true difference map and no rescaling is applied. The model density
carries the **atomic** F₀₀₀ term only (no bulk solvent), so any bulk-solvent
contribution present in `μ_END` surfaces as positive `δ_END` signal.

### Inline (during quantification)

```bash
pseudo-quantify --input_path /scratch/results/my_screen --delta        # σ-scaled δ
pseudo-quantify --input_path /scratch/results/my_screen --delta --end  # σ-scaled + END δ
```

This writes, in each `quantify_results/k_{k}_cap_{cap}/`:

| File | Description |
|---|---|
| `{stem}_delta_sigma.ccp4` | σ-scaled delta `μ_σ − (a·ρ_model + b)` |
| `{stem}_delta_end.ccp4` | absolute-scale delta `μ_END − ρ_model` (only with `--end`) |
| `{stem}_model_density.ccp4` | the model density ρ_model on the absolute e⁻/Å³ scale |
| `{stem}_delta_summary.json` | fit coefficients `a`, `b`, σ-scale correlation, F₀₀₀ |

### Standalone (`pseudo-delta`)

To compute delta maps on an already-completed run **without** re-running STOMP
or quantification:

```bash
pseudo-delta --input_path /scratch/results/my_screen/my_crystal --stem my_crystal
```

It reads the on-disk μ map(s) from `quantify_results/k_{k}_cap_{cap}/` and the
perturbation model, and writes the delta outputs into the same directory. The
END delta is produced only where `{stem}_end_mean.ccp4` already exists (i.e. you
ran `--end` or `pseudo-end` first). Rerunning is idempotent (`--force` to
overwrite).

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