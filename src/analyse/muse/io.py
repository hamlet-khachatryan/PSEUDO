from __future__ import annotations

from pathlib import Path
from typing import Tuple

import gemmi
import numpy as np

from analyse.muse.config import MapNormalizationConfig


def load_map(map_path: str) -> gemmi.FloatGrid:
    """
    Load a CCP4/MRC format map file and return the grid

    Args:
        map_path: Path to the CCP4 format map file
    Returns:
        A gemmi.FloatGrid
    """
    path = Path(map_path)
    if not path.exists():
        raise FileNotFoundError(f"Map file not found: {map_path}")

    ccp4_map = gemmi.read_ccp4_map(str(path))
    ccp4_map.setup(float("nan"))
    return ccp4_map.grid


def load_structure(structure_path: str) -> gemmi.Structure:
    """
    Load a PDB or mmCIF coordinate file

    Args:
        structure_path: Path to the coordinate file
    Returns:
        A gemmi.Structure object
    """
    path = Path(structure_path)
    if not path.exists():
        raise FileNotFoundError(f"Structure file not found: {structure_path}")

    suffix = path.suffix.lower()
    if suffix in (".cif", ".mmcif"):
        doc = gemmi.cif.read(str(path))
        structure = gemmi.make_structure_from_block(doc.sole_block())
    else:
        structure = gemmi.read_pdb(str(path))

    return structure


def compute_map_statistics(
    grid: gemmi.FloatGrid,
    config: MapNormalizationConfig,
) -> Tuple[float, float]:
    """
    Compute or retrieve the global mean and sigma for map normalization

    Args:
        grid: The map grid
        config: Normalization configuration with optional overrides
    Returns:
        Tuple of mean and sigma
    """
    if config.global_mean_override is not None:
        mean = config.global_mean_override
    else:
        arr = np.array(grid, copy=False)
        mean = float(np.mean(arr))

    if config.global_sigma_override is not None:
        sigma = config.global_sigma_override
    else:
        arr = np.array(grid, copy=False)
        sigma = float(np.std(arr))

    if sigma == 0.0:
        sigma = 1.0

    return mean, sigma
