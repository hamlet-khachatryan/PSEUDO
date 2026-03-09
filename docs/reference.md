# Configuration Reference

All parameters for the three PSEUDO stages. CLI flags and YAML keys share the same names.

---

## Debias (`pseudo-debias`)

### `debias` — core omission parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `run_name` | `str` | *required* | Unique identifier for the run. Used as the top-level output directory name. |
| `omit_type` | `str` | `"atoms"` | Structural unit to omit per iteration. Options: `amino_acids`, `atoms`. |
| `omit_fraction` | `float` | `0.1` | Fraction of the structure omitted per iteration. Range: `(0, 1)`. |
| `iterations` | `int` | `5` | Number of stochastic omission iterations. |
| `always_omit` | `str` | `null` | Comma-separated selection string for atoms/residues omitted in every iteration, e.g. `"A 567, A 234"`. |
| `seed` | `int` | `42` | Random seed for reproducibility of the omission mask. |
| `structure_path` | `str` | `null` | Absolute path to the input PDB or CIF file. Required if `screening_path` is not set. |
| `reflections_path` | `str` | `null` | Absolute path to the input MTZ reflections file. Required if `screening_path` is not set. |
| `screening_path` | `str` | `null` | Path to a CSV or SQLite (Diamond SoakDB) screening file for batch processing. |

Either `structure_path` + `reflections_path` **or** `screening_path` must be provided.

### `slurm` — cluster resources

| Parameter | Type | Default | Description |
|---|---|---|---|
| `job_name` | `str` | `"debias_job"` | SLURM job name shown in the queue. |
| `partition` | `str` | `"cs05r"` | Cluster partition to submit to. |
| `time` | `str` | `"3-00:00:00"` | Maximum wall time (`D-HH:MM:SS`). |
| `mem_per_cpu` | `str` | `"5G"` | Memory per CPU core. |
| `cpus_per_task` | `int` | `1` | CPU cores per task. |
| `num_nodes` | `int` | `1` | Number of compute nodes. |

### `paths` — filesystem

| Parameter | Type | Default | Description |
|---|---|---|---|
| `work_dir` | `str` | CWD | Root directory for all output. |

---

## Quantify (`pseudo-quantify`)

Configured entirely via CLI flags (no YAML):

| Flag | Short | Type | Default | Description |
|---|---|---|---|---|
| `--input_path` | `-p` | path | *required* | Workspace directory produced by `pseudo-debias`. |
| `--stem` | `-s` | str | auto | Explicit experiment stem. Inferred automatically if omitted. |
| `--k_factor` | `-k` | float | `1.0` | Radius multiplier K for atom ownership spheres. |
| `--map_cap` | `-c` | int | `50` | Limit processing to the first N maps (for convergence testing). |
| `--force` | `-f` | flag | `False` | Overwrite existing results. |

---

## Analyse (`pseudo-analyse`)

Configured entirely via CLI flags (no YAML):

| Flag | Short | Type | Default | Description |
|---|---|---|---|---|
| `--input_path` | `-p` | path | *required* | Workspace directory (single experiment or screening root). |
| `--stem` | `-s` | str | auto | Explicit experiment stem. |
| `--map_path` | `-m` | path | auto | Override the auto-discovered SNR map with a custom CCP4 file. |
| `--model_path` | | path | auto | Override the processed model with a custom PDB/CIF file. |
| `--k_factor` | `-k` | float | `1.0` | K factor used to locate the SNR map in `quantify_results/`. |
| `--map_cap` | `-c` | int | `50` | Map cap used to locate the SNR map. Pass `0` to auto-detect. |
| `--num_processes` | `-n` | int | `1` | Parallel worker processes for screening mode. |
| `--significance_alpha` | `-a` | float | `0.05` | Significance level α for the null-distribution SNR threshold. |

---

## MUSE scoring parameters (`MUSEConfig`)

These are Python-only parameters, passed to `run_muse()` via a `MUSEConfig` object. They control the internal MUSE scoring algorithm used by `pseudo-analyse`.

### `DensityScoreConfig`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `zeta` | float | `1.2` | Upper truncation cap for normalised density values. |
| `use_truncation` | bool | `True` | Apply the upper cap. Set `False` to pass raw values. |

### `MapNormalizationConfig`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `normalize` | bool | `False` | Apply `(ρ − μ) / σ` normalisation. Set `True` for standard 2Fo-Fc maps. |
| `global_mean_override` | float\|None | `None` | Override the computed map mean. |
| `global_sigma_override` | float\|None | `None` | Override the computed map sigma. |

### `AggregationConfig`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `ediam_shift` | float | `0.1` | Shift *s* in the MUSEm power mean. |
| `ediam_exponent` | float | `−2.0` | Exponent *p* in the MUSEm power mean (negative = soft-minimum). |
| `opia_threshold` | float | `0.8` | MUSE score threshold for OPIA counting. |
| `clash_threshold` | float | `0.1` | Fractional sphere overlap that triggers a clash flag. |
| `unaccounted_density_threshold` | float | `0.2` | `|MUSE−|` above which unaccounted density is flagged. |
| `missing_density_threshold` | float | `0.8` | `MUSE+` below which missing density is flagged. |

### `GridConfig`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `max_spacing_angstrom` | float | `0.7` | Maximum grid spacing (Å); finer grids are oversampled. |
| `interpolation_order` | int | `3` | Map interpolation order: `1` = trilinear, `3` = tricubic. |

### `OwnershipConfig`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `covalent_bond_tolerance` | float | `0.4` | Tolerance (Å) added to covalent radii sum for bond detection. |

### `WaterScoringConfig`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `sigma_threshold` | float | `1.0` | Minimum density level (in σ above mean) for water scoring. |
| `classification_threshold` | float | `0.24` | MUSE score below which a water molecule is flagged. |

### `WeightingConfig`

Three-parabola weighting curve coefficients (paper-derived defaults, rarely changed):

| Parameter | Default | Description |
|---|---|---|
| `p1_m` | −1.0 | Curvature of inner parabola P1. |
| `p1_c_frac` | 0.0 | Centre of P1 as fraction of *r*. |
| `p1_b` | 1.0 | Peak value of P1. |
| `transition_1_frac` | 1.0822 | P1→P2 transition as fraction of *r*. |
| `p2_m` | 5.1177 | Curvature of donut parabola P2. |
| `p2_c_frac` | 1.29366 | Centre of P2. |
| `p2_b` | −0.4 | Minimum value of P2. |
| `transition_2_frac` | 1.4043 | P2→P3 transition. |
| `p3_m` | −0.9507 | Curvature of tail parabola P3. |
| `p3_c_frac` | 2.0 | Centre of P3. |
| `p3_b` | 0.0 | P3 value at 2*r*. |