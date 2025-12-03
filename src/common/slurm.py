from pathlib import Path

from src.debias.config import DebiasConfig

SBATCH_HEADER = """#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --partition={partition}
#SBATCH --nodes={num_nodes}
#SBATCH --cpus-per-task={cpus_per_task}
#SBATCH --mem-per-cpu={mem_per_cpu}
#SBATCH --time={time}
#SBATCH --output=%x_%A_%a.out
#SBATCH --error=%x_%A_%a.err
#SBATCH --array=1-{num_tasks}
"""

ENV_SETUP = """
module load phenix
"""

EXECUTION_PREPROCESSING = """
TASK_INFO=$(sed -n "${{SLURM_ARRAY_TASK_ID}}p" "{preprocess_manifest}")


CRYSTAL_ID=$(echo $TASK_INFO | cut -d'|' -f1)
MODEL_PATH=$(echo $TASK_INFO | cut -d'|' -f2)
REFLECTION_PATH=$(echo $TASK_INFO | cut -d'|' -f3)

OUTPUT_PATH="{base_dir}/$CRYSTAL_ID"
PROCESSED_PATH="$OUTPUT_PATH/processed"
mkdir -p "$PROCESSED_PATH"
cd "$OUTPUT_PATH"

phenix.pdbtools "$MODEL_PATH" "$PROCESSED_PATH/pdbtools.params"
phenix.ready_set "$PROCESSED_PATH/ready_set.params"
"""

EXECUTION_OMISSION = """
TASK_INFO=$(sed -n "${{SLURM_ARRAY_TASK_ID}}p" "{omission_manifest}")

PARAM_FILE=$TASK_INFO
RESULT_DIR=$(dirname "$PARAM_FILE")/../results/$(basename "$PARAM_FILE" .params)

mkdir -p "$RESULT_DIR"
cd "$RESULT_DIR"
phenix.composite_omit_map "$PARAM_FILE"
"""


def generate_preprocessing_sbatch_content(
    cfg: DebiasConfig,
    manifest_path: Path,
    num_tasks: int,
    dirs: dict,
) -> str:
    """
    Compose the content of a SLURM sbatch script for an array job.

    Args:
        cfg: Module configuration.
        manifest_path: Path to the text file containing inputs for each task.
        num_tasks: Number of tasks to run in the array job.
        dirs: Dictionary of output directories.
    Returns:
        A string containing the complete sbatch script.
    """
    header = SBATCH_HEADER.format(
        job_name=cfg.slurm.job_name,
        partition=cfg.slurm.partition,
        num_nodes=cfg.slurm.num_nodes,
        cpus_per_task=cfg.slurm.cpus_per_task,
        mem_per_cpu=cfg.slurm.mem_per_cpu,
        time=cfg.slurm.time,
        num_tasks=num_tasks,
    )

    execution = EXECUTION_PREPROCESSING.format(
        preprocess_manifest=str(manifest_path),
        base_dir=str(dirs["root"]),
    )

    return header + ENV_SETUP + execution


def generate_omission_sbatch_content(
    cfg: DebiasConfig,
    manifest_path: Path,
    num_tasks: int,
) -> str:
    """
    Compose the content of a SLURM sbatch script for an array job.

    Args:
        cfg: Module configuration.
        manifest_path: Path to the text file containing inputs for each task.
        num_tasks: Number of tasks to run in the array job.
    Returns:
        A string containing the complete sbatch script.
    """
    header = SBATCH_HEADER.format(
        job_name=cfg.slurm.job_name,
        partition=cfg.slurm.partition,
        num_nodes=cfg.slurm.num_nodes,
        cpus_per_task=cfg.slurm.cpus_per_task,
        mem_per_cpu=cfg.slurm.mem_per_cpu,
        time=cfg.slurm.time,
        num_tasks=num_tasks,
    )

    execution = EXECUTION_OMISSION.format(
        omission_manifest=str(manifest_path),
    )

    return header + ENV_SETUP + execution
