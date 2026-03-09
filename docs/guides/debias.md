# Debias Guide

The **Debias** module generates stochastic omit perturbation (STOMP) map ensembles. It produces SLURM sbatch scripts for HPC submission — it does not run Phenix directly.

For the statistical motivation see [Theory → Debiasing](../theory.md#2-stochastic-omit-perturbation-stomp--debiasing).

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

```bash
jid=$(sbatch --parsable sbatch/submit_preprocessing.slurm) && \
sbatch --dependency=afterok:$jid sbatch/submit_omission.slurm
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
│   ├── submit_omission.slurm
│   ├── preprocessing_manifest.txt
│   └── omit_manifest.txt
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

## Parameter reference

See [Configuration Reference — Debias](../reference.md#debias-pseudo-debias).