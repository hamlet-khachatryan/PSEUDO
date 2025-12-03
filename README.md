# PSEUDO: Protein Structure Error and Uncertainty Determination and Optimisation

PSEUDO is a computational framework for debiasing and uncertainty quantification in protein structural models resolved by molecular replacement.

## Debias Module: Usage Guide

### Overview
The Debias module supports flexible configuration for both interactive (Jupyter) and batch (CLI) workflows. It utilizes a strict configuration precedence system to ensure reproducibility while allowing for quick experimental overrides.

#### Configuration Precedence
Values are applied in the following order (highest priority first):
1. **CLI Flags / Manual Overrides** (Highest Priority)
2. **External YAML Config File**
3. **Internal System Defaults** (Lowest Priority)

## Configuration Reference

This section details all available parameters in the configuration YAML file.

### Debias Parameters (`debias`)
Core settings controlling the omission and debiasing logic.

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| **`run_name`** | `str` | *Required* | Unique identifier for the experiment run. |
| **`omit_type`** | `str` | `"amino_acids"` | Type of structural element to omit. Options: `"amino_acids"`, `"atoms"`. |
| **`omit_fraction`** | `float` | `0.1` | Fraction of the structure to omit (0.0 to 1.0). |
| **`iterations`** | `int` | `5` | Number of refinement iterations to perform. |
| **`always_omit`** | `list[str]` | `None` | List of specific residues or atoms to always exclude. |
| **`seed`** | `int` | `None` | Random seed for reproducibility of the omission mask. |
| **`structure_path`** | `str` | `None` | Absolute path to the input PDB structure file. |
| **`reflections_path`** | `str` | `None` | Absolute path to the input MTZ reflections file. |
| **`screening_path`** | `str` | `None` | Path to screening results or data used for filtering. |

### SLURM Resources (`slurm`)
Settings for workload management on HPC clusters.

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| **`job_name`** | `str` | `"debias_job"` | Name of the job as it appears in the queue. |
| **`partition`** | `str` | `"cs05r"` | Cluster partition (queue) to submit the job to. |
| **`time`** | `str` | `"04:00:00"` | Maximum wall time for the job (HH:MM:SS). |
| **`mem_per_cpu`** | `str` | `"1024"` | Memory required per CPU (in MB). |
| **`cpus_per_task`** | `int` | `1` | Number of CPUs allocated per task. |
| **`num_nodes`** | `int` | `3` | Number of compute nodes to request. |

### Path Configuration (`paths`)
System-level path settings.

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| **`work_dir`** | `str` | `CWD` | Base directory for outputs and temporary files. Defaults to the current working directory. |

### Scenario 1: Python API with External YAML

**Step 1: Create Configuration File**
Create a file named `my_experiment.yaml`:

```yaml
debias:
  run_name: "experiment_alpha"
  omit_fraction: 0.15
  omit_type: "residues"

slurm:
  time: "04:00:00"
  partition: "gpu_prod"
```

**Step 2: Execute in Python**
Run the following code in your script or notebook:

```python
from src.debias.api import load_debias_config, generate_slurm_job

# Load config and inject data paths dynamically
cfg = load_debias_config(
    config_path="my_experiment.yaml",
    overrides=[
        "debias.structure_path=/data/pdb/1abc.pdb",
        "debias.reflections_path=/data/mtz/1abc.mtz"
    ]
)

# Generate scripts
generate_slurm_job(cfg)
```

### Scenario 2: CLI with External YAML

```bash
# Generate SLURM scripts using settings from file
python src/main.py debias generate \
    --config-file ./configs/experiment_alpha.yaml \
    --structure /data/pdb/1abc.pdb \
    --reflections /data/mtz/1abc.mtz
```

### Scenario 3: Python API with Manual Parameters

```python
from src.debias.api import load_debias_config, generate_slurm_job

# Define a sweep of parameters
fractions = [0.05, 0.10, 0.20]

for frac in fractions:
    run_id = f"sweep_frac_{frac}"
    
    # Configure purely via code
    cfg = load_debias_config(
        overrides=[
            f"debias.run_name={run_id}",
            f"debias.omit_fraction={frac}",
            "debias.structure_path=/data/pdb/1abc.pdb",
            "debias.reflections_path=/data/mtz/1abc.mtz"
        ]
    )
    
    generate_slurm_job(cfg)
```

### Scenario 4: CLI with Manual Flags

```bash
# Uses internal defaults for slurm/paths, overrides input data
python src/main.py debias generate \
    --run-name "quick_test" \
    --structure /tmp/test.pdb \
    --reflections /tmp/test.mtz
```