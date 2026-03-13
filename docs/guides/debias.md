---
title: Debias
parent: Guides
nav_order: 1
---

# Debias Guide

The **Debias** module generates stochastic omit perturbation (STOMP) map ensembles. It produces SLURM sbatch scripts for HPC submission — it does not run Phenix directly.

---

## Configuration precedence

Values are resolved in this order (highest wins):

1. CLI flags / Python API overrides
2. External YAML config file
3. Internal defaults

---

## Quick start

**Single structure — CLI flags only:**

```bash
pseudo-debias generate-params \
  --run_name my_experiment \
  --structure_path /data/target.pdb \
  --reflections_path /data/target.mtz \
  --work_dir /scratch/results
```

**YAML config file:**

```yaml
# run.yaml
debias:
  run_name: "my_experiment"
  structure_path: "/data/target.pdb"
  reflections_path: "/data/target.mtz"
  omit_type: "atoms"
  omit_fraction: 0.1
  iterations: 5
  seed: 42

paths:
  work_dir: "/scratch/results"

slurm:
  partition: "cs05r"
  mem_per_cpu: "5G"
```

```bash
pseudo-debias generate-params --config run.yaml
```

**Submit the generated SLURM jobs:**

The submission command is printed at the end of `generate-params` and also recorded in the eliot log. For a small run (omission jobs ≤ `screening_chunk_size`) it is a single pair:

```bash
jid=$(sbatch --parsable sbatch/submit_preprocessing.slurm)
jid=$(sbatch --parsable --dependency=afterok:$jid sbatch/submit_omission.slurm)
```

For large screening runs the omission array is split into sequential chunks (see [Scheduler-friendly chunking](#scheduler-friendly-chunking)):

```bash
jid=$(sbatch --parsable sbatch/submit_preprocessing.slurm)
jid=$(sbatch --parsable --dependency=afterok:$jid sbatch/submit_omission_0.slurm)
jid=$(sbatch --parsable --dependency=afterok:$jid sbatch/submit_omission_1.slurm)
# … one line per chunk
```

---

## Python API

```python
from debias.api import load_debias_config, generate_slurm_job

cfg = load_debias_config(
    config_path="run.yaml",
    overrides=[
        "debias.structure_path=/data/target.pdb",
        "debias.reflections_path=/data/target.mtz",
    ],
)
generate_slurm_job(cfg)
```

---

## Batch screening

Pass a CSV or Diamond SoakDB SQLite file to process many crystals at once:

```bash
pseudo-debias generate-params \
  --config run.yaml \
  --screening_path /data/fragment_screen.csv
```

The CSV must contain `PDB` (or `CIF`/`structure`) and `MTZ` columns. For SQLite files (Diamond XChem SoakDB), the module queries `mainTable` and filters for outcomes `4 - CompChem ready`, `5 - Deposition ready`, and `6 - Deposited`.

---

## Scheduler-friendly chunking

Each crystal produces ~50 omission parameter files, so a large screening run can generate tens of thousands of omission jobs. Submitting these as a single array risks overwhelming the scheduler.

`screening_chunk_size` (default `1000`) caps the number of omission jobs per sbatch array. The full omission manifest is split into chunks of that size; each chunk gets its own `submit_omission_N.slurm` script and they are chained with `--dependency=afterok` so only one chunk runs at a time.

```bash
pseudo-debias generate-params \
  --config run.yaml \
  --screening_path /data/fragment_screen.csv \
  --screening_chunk_size 500   # at most 500 omission jobs per array
```

Or via YAML:

```yaml
debias:
  screening_chunk_size: 500
```

The complete submission command (with all chunk dependencies) is both printed to stdout and recorded in the eliot log under `debias:submission_command`.

---

## MTZ label resolution

Before generating omission `.params` files, PSEUDO reads each MTZ file with gemmi and auto-selects the observed-data and R-free flag columns. This prevents the Phenix error *"Multiple equally suitable arrays of observed xray data found"* and the equivalent error when multiple R-free / Status columns are present.

### Auto-detection priority

**Observed data** (first matching pair wins):

| Priority | F column | SIGF column |
|---|---|---|
| 1 | `F-obs-filtered` | `SIGF-obs-filtered` |
| 2 | `F-obs` | `SIGF-obs` |
| 3 | `FP` | `SIGFP` |
| 4 | `FOBS` | `SIGFOBS` |
| 5 | `Fobs` | `SIGFobs` |
| 6 | `F` | `SIGF` |
| 7 | `FTOT` | `SIGTOT` |
| 8 | `IMEAN` | `SIGIMEAN` |
| 9 | `I` | `SIGI` |
| 10 | `IOBS` | `SIGIOBS` |

**R-free flag** (first matching column wins):
`FreeR_flag` → `FREE` → `FREER` → `R-free-flags` → `Status`

`Status` is last because it is a character column in some MTZ files and serves as a free-set indicator only when no dedicated flag column exists. When both `FreeR_flag` and `Status` are present, `FreeR_flag` is always preferred.

The resolved labels are printed to stdout and recorded in the eliot log under `debias:mtz_labels_resolved`. The full column inventory of each MTZ is logged under `debias:mtz_columns_found`.

### When auto-detection fails

If a column pair cannot be matched, `generate-params` raises a `ValueError` listing the F/I and integer columns actually present in the MTZ:

```
[Ax0123] MTZ label detection failed for '.../refine.mtz':
  No recognised amplitude/intensity pair found.
  F/I columns present: ['FTOT', 'F_anomalous']
  Set 'debias.mtz_f_labels' to e.g. "FP,SIGFP" to override.
```

Set the override and rerun:

```bash
pseudo-debias generate-params \
  --config run.yaml \
  --mtz_f_labels "FTOT,SIGTOT"
```

Or in YAML:

```yaml
debias:
  mtz_f_labels: "FTOT,SIGTOT"
  mtz_rfree_label: "FreeR_flag"
```

Config overrides take precedence over auto-detection. Setting both columns explicitly skips MTZ reading entirely for that crystal.

---

## `always_omit`

Force specific residues/atoms to be omitted in every iteration — essential for unbiased ligand validation:

```yaml
debias:
  always_omit: "A 567, A 568"   # chain resnum [atom_name]
```

---

## Output directory layout

```
<work_dir>/<run_name>/
├── sbatch/
│   ├── submit_preprocessing.slurm
│   ├── submit_omission.slurm        # single chunk (small runs)
│   ├── submit_omission_0.slurm      # ┐
│   ├── submit_omission_1.slurm      # ├ chunked (large screening runs)
│   ├── ...                          # ┘
│   ├── preprocessing_manifest.txt
│   ├── omit_manifest.txt            # full reference manifest (always written)
│   ├── omit_manifest_0.txt          # ┐
│   ├── omit_manifest_1.txt          # ├ per-chunk manifests
│   └── ...                          # ┘
│
├── logs/                          # SLURM .out and .err files
│
└── <crystal_id>/
    ├── processed/
    │   ├── {stem}_original.pdb
    │   └── {stem}_updated.pdb
    ├── metadata/
    │   └── {stem}_omission_map.json
    ├── params/
    │   ├── {stem}_0.params
    │   └── ...
    └── results/
        ├── {stem}_0/{stem}_0.mtz
        └── ...
```

`{stem}` is derived from the input structure filename.

---

## Re-run behaviour

By default, crystals whose first perturbation map (`results/<stem>_0/<stem>_0.mtz`) already exists are skipped. Pass `--force` / `-f` or set `debias.force: true` in YAML to regenerate everything:

```bash
pseudo-debias generate-params --config run.yaml --force
```

```yaml
debias:
  force: true
```

---

## Parameter reference

See [Configuration Reference — Debias](../reference.md#debias-pseudo-debias).