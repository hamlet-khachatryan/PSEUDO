---
title: Quantify
parent: Guides
nav_order: 2
---

# Quantify Guide

The **Quantify** module analyses the ensemble of STOMP maps produced by `pseudo-debias`. It separates true crystallographic signal from phase bias and outputs debiased CCP4 maps.

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

| Flag | Short | Default | Description |
|---|---|---|---|
| `--input_path` | `-p` | *required* | Workspace root produced by `pseudo-debias`. |
| `--stem` | `-s` | auto | Explicit experiment stem; inferred if omitted. |
| `--k_factor` | `-k` | `1.0` | Radius multiplier K for atom ownership spheres. |
| `--map_cap` | `-c` | `50` | Limit to the first N maps (convergence testing). |
| `--force` | `-f` | `False` | Overwrite existing `quantify_results/`. |

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

See [Configuration Reference — Quantify](../reference.md#quantify-pseudo-quantify).