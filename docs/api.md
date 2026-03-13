---
title: API Reference
nav_order: 3
---

# Python API Reference

Core functions for embedding PSEUDO in Python pipelines or notebooks.

---

## Debias

```python
from debias.api import load_debias_config, generate_slurm_job, run_debias_generation
```

### `load_debias_config`

```python
load_debias_config(
    config_path: str | Path | None = None,
    overrides: list[str] | None = None,
) -> DebiasConfig
```

Load, merge and validate configuration. Precedence: `overrides` → `config_path` → internal defaults.

| Argument | Type | Description |
|---|---|---|
| `config_path` | `str \| Path \| None` | Path to an external YAML file. |
| `overrides` | `list[str] \| None` | Dot-notation overrides, e.g. `["debias.run_name=test"]`. |

Returns a validated `DebiasConfig`.

---

### `generate_slurm_job`

```python
generate_slurm_job(cfg: DebiasConfig) -> None
```

Create the run directory layout, write SLURM sbatch scripts and manifests, and print the submission command.

---

### `run_debias_generation`

```python
run_debias_generation(
    config_path: str | Path | None = None,
    run_name: str | None = None,
    structure_path: str | Path | None = None,
    reflections_path: str | Path | None = None,
    screening_path: str | Path | None = None,
    work_dir: str | Path | None = None,
    omit_type: str | None = None,
    omit_fraction: float | None = None,
    always_omit: str | None = None,
    iterations: int | None = None,
    seed: int | None = None,
    slurm_partition: str | None = None,
    slurm_cpus_per_task: int | None = None,
    slurm_mem_per_cpu: str | None = None,
    slurm_num_nodes: int | None = None,
) -> None
```

Convenience wrapper: builds overrides from keyword arguments, loads config and calls `generate_slurm_job`.

**Minimal example:**

```python
from debias.api import run_debias_generation

run_debias_generation(
    run_name="my_experiment",
    structure_path="/data/target.pdb",
    reflections_path="/data/target.mtz",
    work_dir="/scratch/results",
    omit_fraction=0.1,
    iterations=5,
)
```

---

## Quantify

```python
from quantify.api import run_quantification
```

### `run_quantification`

```python
run_quantification(
    input_path: Path | str,
    stem: str | None = None,
    force: bool = False,
    k_factor: float = 1.5,
    map_cap: int | None = None,
    num_processes: int = 1,
) -> None
```

Run bias separation on a single experiment or a screening workspace. In screening mode experiments are processed in parallel using `num_processes` workers.

| Argument | Type | Default | Description |
|---|---|---|---|
| `input_path` | `Path \| str` | *required* | Workspace produced by `pseudo-debias`. |
| `stem` | `str \| None` | auto | Explicit experiment stem; inferred if omitted. |
| `force` | `bool` | `False` | Overwrite existing `quantify_results/`. |
| `k_factor` | `float` | `1.5` | Radius multiplier K for ownership spheres. |
| `map_cap` | `int \| None` | `None` | Use only the first N maps; `None` uses all. |
| `num_processes` | `int` | `1` | Parallel workers for screening mode. |

**Example:**

```python
from quantify.api import run_quantification

run_quantification(
    input_path="/scratch/results/my_experiment",
    k_factor=1.0,
    map_cap=50,
)
```

---

### `fit_null_distribution`

```python
from quantify.statistical_model import fit_null_distribution

fit_null_distribution(null_snr: np.ndarray) -> dict[str, float]
```

Fit a Student's t-distribution to null SNR samples. Returns `{"df": ..., "loc": ..., "scale": ...}`.

---

### `compute_significance_threshold`

```python
from quantify.statistical_model import compute_significance_threshold

compute_significance_threshold(
    null_params: dict[str, float],
    alpha: float = 0.05,
) -> float
```

Return the raw SNR value *T* such that `P(SNR > T | null) = alpha`. Used by `pseudo-analyse` to derive data-driven MUSE thresholds.

---

## Analyse

```python
from analyse.api import run_analysis
```

### `run_analysis`

```python
run_analysis(
    input_path: str,
    stem: str | None = None,
    map_path: str | None = None,
    model_path: str | None = None,
    k_factor: float = 1.0,
    map_cap: int | None = 50,
    num_processes: int = 1,
    significance_alpha: float = 0.05,
) -> None
```

Score every heavy atom against the SNR map for one experiment or a whole screening workspace.

| Argument | Type | Default | Description |
|---|---|---|---|
| `input_path` | `str` | *required* | Workspace root or single experiment directory. |
| `stem` | `str \| None` | auto | Explicit experiment stem. |
| `map_path` | `str \| None` | auto | Custom CCP4 map path (overrides SNR map discovery). |
| `model_path` | `str \| None` | auto | Custom PDB/CIF path (overrides processed model). |
| `k_factor` | `float` | `1.0` | K factor used to locate the SNR map. |
| `map_cap` | `int \| None` | `50` | Map cap used to locate the SNR map. |
| `num_processes` | `int` | `1` | Parallel workers for screening mode. |
| `significance_alpha` | `float` | `0.05` | Significance level α for the null-distribution threshold. |

**Example:**

```python
from analyse.api import run_analysis

run_analysis(
    input_path="/scratch/results/my_experiment",
    significance_alpha=0.05,
    num_processes=4,
)
```

---

## MUSE scoring

```python
from analyse.muse.pipeline import run_muse
```

### `run_muse`

```python
run_muse(
    map_path: str,
    structure_path: str,
    resolution: float,
    config: MUSEConfig | None = None,
    skip_hydrogens: bool = True,
    run_error_diagnostics: bool = True,
) -> MUSEResult
```

Run the full MUSE scoring pipeline and return a `MUSEResult` containing per-atom scores, per-residue MUSEm scores, and the OPIA metric.

| Argument | Type | Default | Description |
|---|---|---|---|
| `map_path` | `str` | *required* | Path to a CCP4/MRC map file. |
| `structure_path` | `str` | *required* | Path to a PDB or mmCIF coordinate file. |
| `resolution` | `float` | *required* | Map resolution in Å (drives atom radius lookup). |
| `config` | `MUSEConfig \| None` | `None` | Scoring config; `None` uses `default_config()`. |
| `skip_hydrogens` | `bool` | `True` | Exclude H/D atoms from scoring. |
| `run_error_diagnostics` | `bool` | `True` | Compute clash, missing-density and unaccounted-density flags. |

**Example:**

```python
from analyse.muse.pipeline import run_muse

result = run_muse(
    map_path="target_snr.ccp4",
    structure_path="target_updated.pdb",
    resolution=2.0,
)

print(f"OPIA: {result.opia:.3f}")
print(f"Residues scored: {len(result.residue_scores)}")
```

---

### `export_atom_csv`

```python
from analyse.muse.pipeline import export_atom_csv

export_atom_csv(result: MUSEResult, output_path: str) -> None
```

Write per-atom scores to a CSV. Columns: `chain_id`, `residue_name`, `residue_seq_id`, `insertion_code`, `atom_name`, `element`, `score`, `score_positive`, `score_negative`, `is_water`, `has_clash`, `has_missing_density`, `has_unaccounted_density`, `radius_used`, `n_grid_points`.

---

### `export_residue_csv`

```python
from analyse.muse.pipeline import export_residue_csv

export_residue_csv(result: MUSEResult, output_path: str) -> None
```

Write per-residue MUSEm scores to a CSV. Columns: `chain_id`, `residue_name`, `residue_seq_id`, `insertion_code`, `musem_score`, `min_atom_score`, `median_atom_score`, `max_atom_score`, `n_atoms`, `n_clashes`, `n_missing_density`, `n_unaccounted_density`.

---

### `export_summary`

```python
from analyse.muse.pipeline import export_summary

summary: dict = export_summary(result: MUSEResult)
```

Return a flat `dict` with global run statistics:

| Key | Description |
|---|---|
| `n_atoms` | Total heavy atoms scored |
| `n_residues` | Total residues scored |
| `opia` | OPIA metric |
| `mean_atom_score` | Mean MUSE score across all atoms |
| `median_atom_score` | Median MUSE score |
| `n_clashes` | Atoms flagged with steric clashes |
| `n_missing_density` | Atoms with insufficient density support |
| `n_unaccounted_density` | Atoms with excess density in donut region |
| `global_mean` | Map mean used for normalisation |
| `global_sigma` | Map sigma used for normalisation |

---

### `write_scored_pdb`

```python
from analyse.muse.pipeline import write_scored_pdb

write_scored_pdb(
    result: MUSEResult,
    structure_path: str,
    output_path: str,
    score_level: str = "residue",   # or "atom"
    score_field: str = "musem",     # or "min", "median", "max"
    score_scale: float = 100.0,
    missing_value: float = 0.0,
) -> None
```

Write a PDB with MUSE scores substituted into the B-factor column (multiplied by `score_scale`). Load in PyMOL and colour by B-factor to visualise density support.

---

## MUSE configuration

```python
from analyse.muse.config import MUSEConfig, default_config, snr_map_config
```

### `default_config`

```python
default_config() -> MUSEConfig
```

Returns a `MUSEConfig` with all paper-derived defaults. Suitable for standard electron-density maps with `normalize=True`.

---

### `snr_map_config`

```python
snr_map_config(zeta: float = 5.0) -> MUSEConfig
```

Returns a `MUSEConfig` preset for SNR CCP4 maps produced by `pseudo-quantify`. Normalization is disabled; the truncation cap is set to 5.0. This is the config used automatically by `pseudo-analyse`.

---

### `MUSEConfig`

Frozen dataclass grouping all sub-configs. Override only the sections you need:

```python
from analyse.muse.config import (
    MUSEConfig,
    AggregationConfig,
    DensityScoreConfig,
    MapNormalizationConfig,
)

config = MUSEConfig(
    map_normalization=MapNormalizationConfig(normalize=True),   # 2Fo-Fc map
    density_score=DensityScoreConfig(zeta=1.5),
    aggregation=AggregationConfig(opia_threshold=0.7),
)
result = run_muse("map.ccp4", "model.pdb", resolution=1.8, config=config)
```

See [Configuration Reference](reference.md#muse-scoring-parameters-museconfig) for all fields.

---

## Visualisation

```python
from analyse.visualization import extract_residue_scores, plot_residue_profile, plot_water_support
```

### `extract_residue_scores`

```python
extract_residue_scores(
    result: MUSEResult,
    score_field: str = "musem",   # "musem" | "min" | "median" | "max"
    chain_id: str | None = None,
) -> dict[int, float]
```

Extract per-residue scores from a `MUSEResult` into a plain `{seq_id: value}` dict, ready for plotting.

---

### `plot_residue_profile`

```python
plot_residue_profile(
    scores: dict[int, float],
    title: str = "MUSE Residue Profile",
    figsize: tuple = (14, 4),
    edia_thresholds: bool = True,
) -> matplotlib.figure.Figure
```

Line + fill plot of per-residue MUSE scores. Returns a `Figure`; call `.savefig()` or `.show()`.

```python
scores = extract_residue_scores(result, chain_id="A")
fig = plot_residue_profile(scores, title="Chain A")
fig.savefig("chain_a.pdf")
```

---

### `plot_water_support`

```python
plot_water_support(
    result: MUSEResult,
    threshold: float = 0.5,
    chain_id: str | None = None,
    score_field: str = "musem",
    title: str = "Water Density Support",
    figsize: tuple = (10, 5),
) -> matplotlib.figure.Figure
```

Rank scatter plot of per-water MUSE scores, coloured green (keep) / red (remove) relative to `threshold`.

```python
fig = plot_water_support(result, threshold=0.5)
fig.savefig("waters.pdf")
```
