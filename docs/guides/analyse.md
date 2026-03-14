---
title: Analyse
parent: Guides
nav_order: 3
---

# Analyse Guide

The **Analyse** module scores every heavy atom in the model against the debiased SNR map using **MUSE** (Model Uncertainty Score Estimator). MUSE adapts the EDIA methodology to any scalar field.

---

## Quick start

```bash
pseudo-analyse --input_path /scratch/results/my_experiment
```

---

## CLI options

| Flag | Short | Default | Description |
|---|---|---|---|
| `--input_path` | `-p` | *required* | Workspace root or single experiment directory. |
| `--stem` | `-s` | auto | Explicit experiment stem. |
| `--map_path` | `-m` | auto | Custom CCP4 map instead of the auto-discovered SNR map. |
| `--model_path` | | auto | Custom PDB/CIF instead of the processed model. |
| `--k_factor` | `-k` | `1.0` | K factor used to locate the SNR map in `quantify_results/`. |
| `--map_cap` | `-c` | `50` | Map cap used to locate the SNR map. Pass `0` for auto-detect. |
| `--num_processes` | `-n` | `1` | Parallel workers for screening mode. |
| `--significance_alpha` | `-a` | `0.05` | Significance level α for the null-distribution threshold. |

---

## Python API

```python
from analyse.api import run_analysis

run_analysis(
    input_path="/scratch/results/my_experiment",
    significance_alpha=0.05,
    num_processes=4,     # parallel for screening
)
```

### Using MUSE directly

```python
from analyse.muse.pipeline import run_muse, export_residue_csv, export_summary

result = run_muse(
    map_path="target_snr.ccp4",
    structure_path="target_updated.pdb",
    resolution=2.0,
)

print(export_summary(result))
export_residue_csv(result, "residues.csv")
```

---

## Outputs

Written to `<crystal>/analyse_results/`:

| File | Description |
|---|---|
| `{stem}_atoms.csv` | Per-atom MUSE score, `score_positive`, `score_negative`, diagnostic flags |
| `{stem}_residues.csv` | Per-residue MUSEm, min/median/max atom score, diagnostic counts |
| `{stem}_summary.json` | Global statistics: OPIA, atom/residue counts, thresholds |
| `{stem}_scored.pdb` | Original structure with MUSE scores in the B-factor column (×100) |

---

## Visualisation

Load `{stem}_scored.pdb` in PyMOL and colour by B-factor:

```text
# PyMOL
load target_scored.pdb
spectrum b, red_white_blue
```

### Python plots

```python
from analyse.visualization import extract_residue_scores, plot_residue_profile, plot_water_support

scores = extract_residue_scores(result, score_field="musem", chain_id="A")
fig = plot_residue_profile(scores, title="Chain A — MUSE profile")
fig.savefig("chain_a_profile.pdf")

fig2 = plot_water_support(result, threshold=0.5)
fig2.savefig("water_support.pdf")
```

---

## Significance threshold

When null-distribution parameters are present in `metadata/` (produced by `pseudo-quantify`), `pseudo-analyse` automatically sets the OPIA and missing-density thresholds to the SNR value at `p = significance_alpha`. This adapts the scoring to the actual noise floor of the experiment.

---

## MUSE configuration

```python
from analyse.muse.config import MUSEConfig, AggregationConfig, MapNormalizationConfig
from analyse.muse.pipeline import run_muse

config = MUSEConfig(
    map_normalization=MapNormalizationConfig(normalize=True),   # for 2Fo-Fc maps
    aggregation=AggregationConfig(opia_threshold=0.7),
)
result = run_muse("map.ccp4", "model.pdb", resolution=1.8, config=config)
```

See [Configuration Reference — MUSE](../reference.md#muse-scoring-parameters-museconfig) for all parameters.

---

## Diagnostic flags

Each atom in the output carries three boolean flags:

| Flag | Condition |
|---|---|
| `has_clash` | Sphere overlap > 10 % with any non-bonded neighbour |
| `has_missing_density` | `score_positive` < `missing_density_threshold` |
| `has_unaccounted_density` | `|score_negative|` > `unaccounted_density_threshold` |

---

## HPC / SLURM submission

For fragment screening workspaces, wrap the command in `sbatch` and use `--num_processes` to parallelise across crystals within the job:

```bash
sbatch --partition cs05r \
       --cpus-per-task 8 \
       --mem-per-cpu 5G \
       --time 3-00:00:00 \
       --wrap "pseudo-analyse --input_path /scratch/results/my_screen \
                              --num_processes 8"
```

`--num_processes` controls how many crystals are processed in parallel inside the job. Set it to match `--cpus-per-task`.

---

## References

1. Meyder et al. (2017) *J. Chem. Inf. Model.* **57**, 2437–2447.
2. Nittinger et al. (2015) *J. Chem. Inf. Model.* **55**, 771–783.