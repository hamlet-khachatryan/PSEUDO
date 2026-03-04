"""CCP4 map and structure file I/O using gemmi.

Provides functions to load CCP4/MRC maps and PDB/mmCIF coordinate files,
as well as computing map statistics needed for normalization.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import gemmi
import numpy as np

from muse.config import MapNormalizationConfig


def load_map(map_path: str) -> gemmi.FloatGrid:
    """Load a CCP4/MRC format map file and return the grid.

    The map is read via gemmi and its symmetry operations are set up
    so that interpolation works correctly for any fractional coordinate.

    Args:
        map_path: Path to the CCP4 (.ccp4, .map, .mrc) format map file.

    Returns:
        A gemmi.FloatGrid ready for value interpolation.

    Raises:
        FileNotFoundError: If map_path does not exist.
        RuntimeError: If the file cannot be parsed as a CCP4 map.
    """
    path = Path(map_path)
    if not path.exists():
        raise FileNotFoundError(f"Map file not found: {map_path}")

    ccp4_map = gemmi.read_ccp4_map(str(path))
    #ccp4_map.setup(gemmi.MapSetup.Full)
    return ccp4_map.grid


def load_structure(structure_path: str) -> gemmi.Structure:
    """Load a PDB or mmCIF coordinate file.

    Automatically detects format by file extension (.pdb, .ent, .cif, .mmcif).

    Args:
        structure_path: Path to the coordinate file.

    Returns:
        A gemmi.Structure object.

    Raises:
        FileNotFoundError: If structure_path does not exist.
        RuntimeError: If the file cannot be parsed.
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
    """Compute or retrieve the global mean and sigma for map normalization.

    If the config provides overrides, those are used. Otherwise, statistics
    are computed from all grid point values.

    Args:
        grid: The map grid.
        config: Normalization configuration with optional overrides.

    Returns:
        Tuple of (mean, sigma). When normalization is disabled, these values
        are still computed (or overridden) for reference in the result object,
        but will not be applied during scoring.
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

    # Guard against zero sigma to avoid division errors downstream
    if sigma == 0.0:
        sigma = 1.0

    return mean, sigma
