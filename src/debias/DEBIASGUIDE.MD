# Debias Module

The **Debias Module** provides a framework for stochastic omission perturbation (STOMP) maps. It supports flexible configuration via both a Command Line Interface (CLI) and a Python API, designed to generate sbatch files for high-performance computing (HPC) workflows.

---

## Configuration

The module uses a hierarchical configuration system to determine runtime parameters.

### Precedence Logic
Values are applied in the following order of priority (highest to lowest):

1.  **CLI Flags / Manual Overrides** (Runtime specific)
2.  **External YAML Config File** (Experiment specific)
3.  **Internal System Defaults** (Fallback)

### Parameter Reference

#### 1. Core Parameters (`debias`)
These settings control the structural omission logic and input data mapping.

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| **`run_name`** | `str` | *Required* | Unique identifier for the experiment run. |
| **`omit_type`** | `str` | `"amino_acids"` | Structural element to omit. <br>Options: `amino_acids`, `atoms`. |
| **`omit_fraction`** | `float` | `0.1` | Fraction of the structure to omit (Range: `0.0` - `1.0`). |
| **`iterations`** | `int` | `5` | Number of refinement iterations to perform. |
| **`always_omit`** | `str` | `None` | Selection string for atoms/residues to always exclude. |
| **`seed`** | `int` | `None` | Random seed for reproducibility of the omission mask. |
| **`structure_path`** | `str` | `None` | Absolute path to the input PDB structure file. |
| **`reflections_path`** | `str` | `None` | Absolute path to the input MTZ reflections file. |
| **`screening_path`** | `str` | `None` | Path to screening results or filtering data. |

#### 2. SLURM Resources (`slurm`)
Settings for workload management and job submission on HPC clusters.

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| **`job_name`** | `str` | `"debias_job"` | Name of the job as it appears in the scheduler queue. |
| **`partition`** | `str` | `"cs05r"` | Cluster partition/queue to submit the job to. |
| **`time`** | `str` | `"04:00:00"` | Maximum wall time (Format: `HH:MM:SS`). |
| **`mem_per_cpu`** | `str` | `"1024"` | Memory required per CPU (in MB). |
| **`cpus_per_task`** | `int` | `1` | Number of CPUs allocated per task. |
| **`num_nodes`** | `int` | `3` | Number of compute nodes to request. |

#### 3. Path Configuration (`paths`)
System-level directory settings.

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| **`work_dir`** | `str` | `CWD` | Base directory for outputs. Defaults to Current Working Directory. |

### Output
When a job is generated (e.g., `run_name: testing`), the module creates a structured workspace. The internal organization is determined by the processing stages: preprocessing, parameter generation, and result collection.

Assuming `work_dir` is set to `/workspace` and `run_name` is `experiment_01`, the structure is:

```text
/workspace/
└── experiment_01/
    ├── sbatch/                     # SLURM submission scripts
    │   ├── submit_preprocessing.slurm
    │   ├── submit_omission.slurm
    │   ├── omit_manifest.txt
    │   └── preprocessing_manifest.txt
    │
    ├── processed/                  # Preprocessing intermediates
    │   ├── pdbtools.params         # Param file for PDB tools
    │   ├── ready_set.params        # Param file for Phenix ready_set
    │   ├── {stem}_no_waters.pdb    # Cleaned PDB structure
    │   └── {stem}_updated.pdb      # Model with ligands/hydrogens handled
    │
    ├── metadata/                   # Omission definitions
    │   └── {stem}_omission_map.json # Sparse map of omitted atoms/residues
    │
    ├── params/                     # Generated Phenix parameter files per job
    │   ├── {stem}_0.params
    │   ├── {stem}_1.params
    │   └── ...
    │
    └── results/                    # Output directories for each omission cycle
        ├── {stem}_0/
        │   └── {stem}_0.ccp4        # Perturb map file for cycle 0
        ├── {stem}_1/
        │   └── {stem}_1.ccp4
        └── ...
```
*Note: `{stem}` refers to the identifier derived from the input structure filename (e.g., `test_model` from `test_model.pdb`).*

The module generates a two-stage workflow to ensure data consistency:
1.  **Preprocessing**: runs `pdbtools` and `ready_set` to prepare the model.
2.  **Omission**: runs the array of refinement jobs using the prepared model.

The omission jobs depend on the successful completion of the preprocessing job.

After running `generate_slurm_job(cfg)` in Python or via CLI, the system prints the command to submit the jobs with the correct dependency chain.

```bash
  jid=$(sbatch --parsable <path_to>/sbatch/submit_preprocessing.slurm) && \
sbatch --dependency=afterok:$jid <path_to>/sbatch/submit_omission.slurm
```


## Usage Examples

The following examples demonstrate how to configure and run a debiasing experiment for a target protein structure (`target_5e5z`).

### Example 1: Using the Python API
*Best for programmatic sweeps and dynamic path injection.*

**Step 1: Create a Base Configuration (`config_delta.yaml`)**
```yaml
debias:
  run_name: "experiment_delta"
  omit_fraction: 0.15
  omit_type: "amino_acids"

slurm:
  partition: "gpu_prod"
```

**Step 2: Execute Python Script**

```python
from src.debias.api import load_debias_config, generate_slurm_job

# Define paths for the specific target
pdb_file = "/data/pdb/target_5e5z.pdb"
mtz_file = "/data/mtz/target_5e5z.mtz"

cfg = load_debias_config(
    config_path="config_delta.yaml",
    overrides=[
        f"debias.structure_path={pdb_file}",
        f"debias.reflections_path={mtz_file}"
    ]
)

generate_slurm_job(cfg)
```

### Example 2. CLI with External YAML
Best for reproducible single runs using a predefined parameters.

```bash
  pseudo_debias generate-params --config ./configs/config_delta.yaml
```

### Example 3: Python Parameter Sweep
Best for testing multiple omission parameters programmatically.

```python
from src.debias.api import load_debias_config, generate_slurm_job

target_name = "target_5e5z"
fractions = [0.05, 0.10, 0.20]

for frac in fractions:
    run_id = f"{target_name}_frac_{frac}"
    
    cfg = load_debias_config(
        overrides=[
            f"debias.run_name={run_id}",
            f"debias.omit_fraction={frac}",
            f"debias.structure_path=/data/pdb/{target_name}.pdb",
            f"debias.reflections_path=/data/mtz/{target_name}.mtz"
        ]
    )
    
    generate_slurm_job(cfg)
```

### Example 4: CLI with Manual Flags
Best for quick tests without creating a configuration file.

```bash
  pseudo_debias generate-params --run_name "experiment_delta_quick" --structure /data/pdb/target_5e5z.pdb \
  --reflections /data/mtz/target_5e5z.mtz --omit_fraction 0.05
```