# MUSE — Model Uncertainty Score Estimator

MUSE quantifies how well each atom in a crystallographic model is supported by
an experimental map. It adapts the EDIA methodology (Meyder et al. 2017) so
that it works with any voxel-wise scalar field stored in CCP4 format — electron
density maps, probability maps, SNR maps, or any custom field.

---

## Table of Contents

1. [Overview](#overview)
2. [Methodology](#methodology)
   - [Non-water atoms (Meyder 2017)](#non-water-atoms-meyder-2017)
   - [Water atoms (Nittinger 2015)](#water-atoms-nittinger-2015)
   - [MUSEm aggregation](#musem-aggregation)
   - [OPIA](#opia)
   - [Error diagnostics](#error-diagnostics)
3. [Quick start](#quick-start)
4. [Configuration reference](#configuration-reference)
5. [API reference](#api-reference)
6. [Output format](#output-format)
7. [References](#references)

---

## Overview

Given a CCP4 map and a PDB/mmCIF model, MUSE returns:

| Object | Description |
|---|---|
| **AtomScore** | Per-atom score in roughly [0, 1], plus diagnostic flags. |
| **ResidueScore** | Power-mean (MUSEm) aggregate over all atoms in a residue. |
| **OPIA** | Overall Percentage of well-resolved Interconnected Atoms (structure level). |
| **MUSEResult** | Container holding all of the above and the run configuration. |

Score interpretation (matches the original EDIA scale):

| MUSE | Interpretation |
|---|---|
| ≥ 0.8 | Atom is well-supported by the map. |
| 0.4 – 0.8 | Moderate support; some uncertainty. |
| < 0.4 | Poor support; the atom placement is questionable. |
| < 0 | Pathological; nearly always a modelling error. |

---

## Methodology

### Non-water atoms (Meyder 2017)

Each atom *a* is assigned a radius *r(a, d)* that depends on the element,
formal charge and map resolution *d*. Values are linearly interpolated from
Table S2 of the supplementary material (6 breakpoints: 0.5–3.0 Å).

**Weighting curve** — a three-piece parabola defined over [0, 2r]:

```
            ┌ P1(x)   0 ≤ x ≤ t1        (positive, peaks at centre)
  w(x, r) = ┤ P2(x)   t1 < x ≤ t2       (negative "donut" penalty)
            └ P3(x)   t2 < x ≤ 2r        (returns to zero)
```

where *t1 = 1.0822 r* and *t2 = 1.4043 r*. Each parabola has the form
`P(x) = m/r² * (x − c·r)² + b`. Default parameters (SI Table S3):

| Segment | m | c | b |
|---|---|---|---|
| P1 | −1.0 | 0.0 | 1.0 |
| P2 | 5.1177 | 1.29366 | −0.4 |
| P3 | −0.9507 | 2.0 | 0.0 |

**Ownership** — when multiple atoms' spheres overlap at a grid point *p*,
density is partitioned according to a 4-case decision tree (Meyder 2017,
Figure 2):

1. *a* is the only non-covalent atom in *S(p)* → o = 1.
2. Multiple non-covalent atoms in *S(p)* → proportional to inverse distance.
3. *a* is in the donut region *D(p)* and *S(p)* is non-empty → o = 0.
4. *a* is in *D(p)* and *S(p)* is empty → shared by inverse distance.

Atoms covalently bonded to *a* are excluded from the ownership calculation —
they always share density fully.

**Density score** — before scoring, map values are optionally normalised:

```
z(p) = max(0,  min(ζ,  (ρ(p) − μ) / σ))      [electron density mode]
z(p) = max(0,  min(ζ,  ρ(p)))                  [probability / SNR mode]
```

where *ζ = 1.2* by default (truncation cap).  For probability / SNR maps set
`MapNormalizationConfig(normalize=False)` (the default) and adjust `ζ` if
needed.

**Per-atom score:**

```
MUSE(a) = Σ[w(p,a) · o(p,a) · z(p)]  /  Σ|w(p,a)|      (sum over w > 0)
```

Grid points inside 2r are enumerated. The grid is oversampled to guarantee
spacing ≤ 0.7 Å (configurable) with tricubic interpolation (order 3).

---

### Water atoms (Nittinger 2015)

Water oxygens use a simpler Gaussian + linear weighting within the van der
Waals radius (1.52 Å). Only points above a density threshold (default 1σ
above the map mean) contribute:

```
MUSE(water) = (1 / Σω) · Σ[ω(p) · z_water(p)]

ω(p) = exp(−0.5 (d/δ)²)   ·   (1 − d/r_vdW)      [Gaussian × linear ramp]
```

where *δ* is the oxygen covalent radius (≈ 0.66 Å) and ownership logic is
**not** applied (water oxygens are treated as isolated atoms).

Waters with MUSE < 0.24 are classified as insufficiently resolved (Nittinger
2015, median − 1 std).

---

### MUSEm aggregation

A power mean aggregates atom scores for a molecular fragment *U*
(e.g. a residue):

```
MUSEm(U) = (1/|U| · Σ (MUSE(a) + s)^p)^(1/p) − s
```

Default parameters: shift *s* = 0.1, exponent *p* = −2. The exponent −2
makes MUSEm act as a soft-minimum, giving extra weight to the worst-scoring
atom in the fragment.

---

### OPIA

*Overall Percentage of well-resolved Interconnected Atoms* (structure-level
metric):

1. Mark all heavy atoms with MUSE ≥ 0.8 as "well-supported".
2. Find connected components among well-supported atoms using the covalent
   bond graph.
3. Count atoms that belong to components of size ≥ 2.
4. OPIA = (count from step 3) / (total heavy atoms).

A high OPIA indicates that the model is globally consistent with the map.

---

### Error diagnostics

Three boolean flags are set on each `AtomScore`:

| Flag | Condition |
|---|---|
| `has_clash` | Sphere overlap > 10 % with any non-bonded neighbour. |
| `has_missing_density` | MUSE+ < 0.8 (atom sphere lacks expected density). |
| `has_unaccounted_density` | \|MUSE−\| > 0.2 (extra density in the donut region). |

All thresholds are configurable via `AggregationConfig`.

---

## Quick start

```python
from muse import run_muse
from muse.pipeline import export_atom_csv, export_residue_csv, export_summary

# Run with default config (normalization OFF — for probability/SNR maps)
result = run_muse(
    map_path="path/to/map.ccp4",
    structure_path="path/to/model.pdb",
    resolution=2.0,
)

# Inspect summary
print(export_summary(result))
# {'n_atoms': 1247, 'n_residues': 162, 'opia': 0.871, ...}

# Export to CSV
export_atom_csv(result, "atom_scores.csv")
export_residue_csv(result, "residue_scores.csv")

# Access data directly
for atom in result.atom_scores:
    print(atom.chain_id, atom.residue_name, atom.atom_name, atom.score)

for residue in result.residue_scores:
    print(residue.chain_id, residue.residue_name,
          residue.residue_seq_id, residue.musem_score)
```

### Electron density maps

For a standard 2Fo-Fc electron density map, enable normalization:

```python
from muse.config import MUSEConfig, MapNormalizationConfig

config = MUSEConfig(
    map_normalization=MapNormalizationConfig(normalize=True),
)
result = run_muse("map.ccp4", "model.pdb", resolution=1.8, config=config)
```

### Custom thresholds

```python
from muse.config import (
    MUSEConfig, AggregationConfig, DensityScoreConfig
)

config = MUSEConfig(
    density_score=DensityScoreConfig(zeta=2.0, use_truncation=True),
    aggregation=AggregationConfig(
        opia_threshold=0.7,
        missing_density_threshold=0.6,
    ),
)
```

---

## Configuration reference

All parameters live in `muse.config`. Construct a `MUSEConfig` and override
any sub-config you need.

### `DensityScoreConfig`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `zeta` | float | 1.2 | Upper truncation cap for normalised density. |
| `use_truncation` | bool | True | Apply the upper truncation. |

### `WeightingConfig`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `p1_m` | float | −1.0 | Curvature of inner parabola P1. |
| `p1_c_frac` | float | 0.0 | Centre of P1 (fraction of r). |
| `p1_b` | float | 1.0 | Peak value of P1. |
| `transition_1_frac` | float | 1.0822 | P1→P2 transition (fraction of r). |
| `p2_m` | float | 5.1177 | Curvature of donut parabola P2. |
| `p2_c_frac` | float | 1.29366 | Centre of P2 (fraction of r). |
| `p2_b` | float | −0.4 | Minimum value of P2. |
| `transition_2_frac` | float | 1.4043 | P2→P3 transition (fraction of r). |
| `p3_m` | float | −0.9507 | Curvature of tail parabola P3. |
| `p3_c_frac` | float | 2.0 | Centre of P3 (fraction of r). |
| `p3_b` | float | 0.0 | P3 value at 2r. |

### `WaterScoringConfig`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `sigma_threshold` | float | 1.0 | Min density level (in σ above mean). |
| `classification_threshold` | float | 0.24 | MUSE below which water is flagged. |

### `GridConfig`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `max_spacing_angstrom` | float | 0.7 | Maximum grid spacing; finer grids oversample. |
| `interpolation_order` | int | 3 | Interpolation order (1=trilinear, 3=tricubic). |

### `OwnershipConfig`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `covalent_bond_tolerance` | float | 0.4 | Tolerance (Å) added to covalent radii sum for bond detection. |

### `AggregationConfig`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `ediam_shift` | float | 0.1 | Shift *s* in the MUSEm power mean. |
| `ediam_exponent` | float | −2.0 | Exponent *p* in the MUSEm power mean. |
| `opia_threshold` | float | 0.8 | MUSE threshold for OPIA. |
| `clash_threshold` | float | 0.1 | Fractional sphere overlap for clash flag. |
| `unaccounted_density_threshold` | float | 0.2 | \|MUSE−\| above which unaccounted density is flagged. |
| `missing_density_threshold` | float | 0.8 | MUSE+ below which missing density is flagged. |

### `MapNormalizationConfig`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `normalize` | bool | **False** | Apply `(ρ − μ) / σ` normalisation. Set True for standard electron density maps. |
| `global_mean_override` | float \| None | None | Override the computed map mean. |
| `global_sigma_override` | float \| None | None | Override the computed map sigma. |

---

## API reference

### `run_muse`

```python
from muse import run_muse

result: MUSEResult = run_muse(
    map_path: str,
    structure_path: str,
    resolution: float,
    config: MUSEConfig | None = None,
    skip_hydrogens: bool = True,
    run_error_diagnostics: bool = True,
)
```

**Arguments**

| Name | Type | Description |
|---|---|---|
| `map_path` | str | Path to a CCP4/MRC map file. |
| `structure_path` | str | Path to a PDB or mmCIF coordinate file. |
| `resolution` | float | Map resolution in Å (used for radius lookup). |
| `config` | MUSEConfig \| None | Configuration. `None` → `default_config()`. |
| `skip_hydrogens` | bool | Skip H/D atoms. Default True. |
| `run_error_diagnostics` | bool | Apply diagnostic flags. Default True. |

**Returns** — `MUSEResult`

---

### `export_atom_csv`

```python
from muse.pipeline import export_atom_csv
export_atom_csv(result: MUSEResult, output_path: str) -> None
```

Writes per-atom scores to a CSV. Columns: `chain_id`, `residue_name`,
`residue_seq_id`, `insertion_code`, `atom_name`, `element`, `score`,
`score_positive`, `score_negative`, `is_water`, `has_clash`,
`has_missing_density`, `has_unaccounted_density`, `radius_used`,
`n_grid_points`.

---

### `export_residue_csv`

```python
from muse.pipeline import export_residue_csv
export_residue_csv(result: MUSEResult, output_path: str) -> None
```

Writes per-residue MUSEm scores to a CSV. Columns: `chain_id`,
`residue_name`, `residue_seq_id`, `insertion_code`, `musem_score`,
`min_atom_score`, `median_atom_score`, `max_atom_score`, `n_atoms`,
`n_clashes`, `n_missing_density`, `n_unaccounted_density`.

---

### `export_summary`

```python
from muse.pipeline import export_summary
summary: dict = export_summary(result: MUSEResult)
```

Returns a flat `dict` with: `n_atoms`, `n_residues`, `opia`,
`mean_atom_score`, `median_atom_score`, `n_clashes`, `n_missing_density`,
`n_unaccounted_density`, `global_mean`, `global_sigma`.

---

## Output format

### Atom CSV example

```
chain_id,residue_name,residue_seq_id,insertion_code,atom_name,element,score,...
A,ALA,12,,CA,C,0.912341,...
A,ALA,12,,CB,C,0.875002,...
A,HOH,201,,O,O,0.643110,...
```

### Residue CSV example

```
chain_id,residue_name,residue_seq_id,insertion_code,musem_score,min_atom_score,...
A,ALA,12,,0.891230,0.875002,...
A,HOH,201,,0.643110,0.643110,...
```

---

## References

1. **Meyder, A.; Nittinger, E.; Lange, G.; Klein, R.; Rarey, M.** (2017).
   *Estimating Electron Density Support for Individual Atoms and Molecular
   Fragments in X-ray Structures*.
   J. Chem. Inf. Model. **57**, 2437–2447.
   https://doi.org/10.1021/acs.jcim.7b00391

2. **Nittinger, E.; Meyder, A.; Lange, G.; Klein, R.; Rarey, M.** (2015).
   *Evidence of Water Molecules — A Statistical Evaluation of Water Molecules
   Based on Electron Density*.
   J. Chem. Inf. Model. **55**, 771–783.
   https://doi.org/10.1021/ci500662d

3. **Tickle, I. J.** (2012).
   *Statistical quality indicators for electron-density maps*.
   Acta Crystallogr. D **68**, 454–467.
