from dataclasses import dataclass, field
from typing import Literal, Optional

from pathlib import Path

from omegaconf import OmegaConf


@dataclass
class SlurmResources:
    job_name: str = "debias_job"
    partition: str = "cs05r"
    time: str = "3-00:00:00"
    mem_per_cpu: str = "1024"
    cpus_per_task: int = 1
    num_nodes: int = 3


@dataclass
class PathConfig:
    work_dir: str = field(default_factory=lambda: str(Path.cwd()))


@dataclass
class DebiasParams:
    """Parameters specific to the Debias module."""

    run_name: str
    omit_type: Literal["amino_acids", "atoms"] = "atoms"
    omit_fraction: float = 0.1
    iterations: int = 5
    always_omit: Optional[str] = None
    seed: Optional[int] = 42
    structure_path: Optional[str] = None
    reflections_path: Optional[str] = None
    screening_path: Optional[str] = None
    sqlite_outcomes: Optional[str] = None  # specific for DLS XChem screening data
    max_structures: Optional[int] = None  # specific for DLS XChem screening data
    screening_chunk_size: int = 1000  # max omission jobs per sbatch array submission
    # MTZ label overrides — set when auto-detection fails or picks the wrong column.
    # mtz_f_labels:    comma-separated amplitude+sigma labels, e.g. "FP,SIGFP"
    # mtz_rfree_label: R-free flag column name,              e.g. "FreeR_flag"
    mtz_f_labels: Optional[str] = None
    mtz_rfree_label: Optional[str] = None


@dataclass
class DebiasConfig:
    """Aggregated config for the Debias execution context."""

    slurm: SlurmResources = field(default_factory=SlurmResources)
    paths: PathConfig = field(default_factory=PathConfig)
    debias: DebiasParams = field(default_factory=DebiasParams)

    def __str__(self):
        return OmegaConf.to_yaml(self)
