from dataclasses import dataclass, field
from typing import Literal, Optional

from pathlib import Path

from omegaconf import OmegaConf


@dataclass
class SlurmResources:
    job_name: str = "debias_job"
    partition: str = "cs05r"
    time: str = "10-00:00:00"
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
    omit_type: Literal["amino_acids", "atoms"] = "amino_acids"
    omit_fraction: float = 0.1
    iterations: int = 5
    always_omit: Optional[str] = None
    seed: Optional[int] = None
    structure_path: Optional[str] = None
    reflections_path: Optional[str] = None
    screening_path: Optional[str] = None


@dataclass
class DebiasConfig:
    """Aggregated config for the Debias execution context."""

    slurm: SlurmResources = field(default_factory=SlurmResources)
    paths: PathConfig = field(default_factory=PathConfig)
    debias: DebiasParams = field(default_factory=DebiasParams)

    def __str__(self):
        return OmegaConf.to_yaml(self)
