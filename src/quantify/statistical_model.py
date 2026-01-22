from __future__ import annotations

from typing import List

import gemmi
import numpy as np
from scipy.stats import t
from pathlib import Path


def sample_null_distribution(
    snr_map: Path | str,
    model_path: Path | str,
    n_samples: int = 20000,
) -> np.ndarray:
    """
    Samples SNR values specifically from within the protein mask region.
    Used to establish a background null distribution for statistical testing.

    Args:
        snr_map (Path | str): Path to a ccp4 snr map file.
        model_path (Path | str): Path to the PDB model used to define the protein mask.
        n_samples (int): Number of random samples to draw.

    Returns:
        np.array: A 1D array of sampled SNR values from the protein region.
    """
    null_snrs = []
    masker = gemmi.SolventMasker(gemmi.AtomicRadiiSet.Cctbx)

    st = gemmi.read_structure(str(model_path))
    st.remove_waters()

    grid = gemmi.read_ccp4_map(snr_map, setup=True).grid
    grid.normalize()

    protein_mask = gemmi.FloatGrid()
    protein_mask.setup_from(st, spacing=0.5)
    masker.put_mask_on_float_grid(protein_mask, st[0])
    for _ in range(n_samples):
        frac = np.random.randn(3)
        pos = grid.unit_cell.orthogonalize(gemmi.Fractional(*frac))
        if protein_mask.interpolate_value(pos) == 1:
            snr_bulk = grid.interpolate_value(pos)
            null_snrs.append(snr_bulk)

    return np.array(null_snrs)


def fit_t_test(null_snr: List, full_snr_map: np.ndarray) -> np.ndarray:
    """
    Fits a Student's t-distribution to the null SNR samples and calculates
    the survival function (1 - CDF), representing the P-value for the entire map.

    Args:
        null_snr (list): List of null distribution samples.
        full_snr_map (np.ndarray): The full 3D SNR map.

    Returns:
        np.ndarray: 3D map of P-values (0.0 to 1.0).
    """
    if len(null_snr) == 0:
        return np.zeros_like(full_snr_map)

    df_fit, loc_fit, scale_fit = t.fit(null_snr)
    p_values = t.sf(full_snr_map, df=df_fit, loc=loc_fit, scale=scale_fit)

    return p_values.astype(np.float32)
