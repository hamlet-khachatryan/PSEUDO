from __future__ import annotations

from typing import Dict, List

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
    Samples SNR values from within the protein mask region to establish a
    background null distribution for statistical testing

    Samples are drawn in raw SNR space (no normalization applied) so that
    the returned values are directly comparable to the raw SNR map passed to
    fit_t_test and used in MUSE scoring

    Args:
        snr_map: Path to a CCP4 SNR map file.
        model_path: Path to the PDB/mmCIF model used to define the protein mask.
        n_samples: Number of random samples to draw.

    Returns:
        1D array of sampled raw SNR values from within the protein region.
    """

    null_snrs = []
    masker = gemmi.SolventMasker(gemmi.AtomicRadiiSet.Cctbx)

    st = gemmi.read_structure(str(model_path))
    st.remove_waters()

    grid = gemmi.read_ccp4_map(snr_map, setup=True).grid

    protein_mask = gemmi.FloatGrid()
    protein_mask.setup_from(st, spacing=0.5)
    masker.put_mask_on_float_grid(protein_mask, st[0])

    for _ in range(n_samples):
        frac = np.random.randn(3)
        pos = grid.unit_cell.orthogonalize(gemmi.Fractional(*frac))
        if protein_mask.interpolate_value(pos) == 1:
            null_snrs.append(grid.interpolate_value(pos))

    return np.array(null_snrs)


def fit_t_test(null_snr: List, full_snr_map: np.ndarray) -> np.ndarray:
    """
    Fits a t-distribution to the null SNR samples and calculates
    the survival function (1 - CDF), representing the p-value for every
    voxel in the map

    Both null_snr and full_snr_map must be in the same numerical space
    (raw SNR, not normalized) for the p-values to be calibrated correctly

    Args:
        null_snr: 1D array of null-distribution SNR samples (raw, not normalized).
        full_snr_map: The full 3D raw SNR map array.

    Returns:
        3D array of p-values in [0.0, 1.0]. Low values indicate statistically
        significant SNR — i.e., density unlikely to arise from background noise.
    """

    if len(null_snr) == 0:
        return np.zeros_like(full_snr_map)

    df_fit, loc_fit, scale_fit = t.fit(null_snr)
    p_values = t.sf(full_snr_map, df=df_fit, loc=loc_fit, scale=scale_fit)

    return p_values.astype(np.float32)


def fit_null_distribution(null_snr: np.ndarray) -> Dict[str, float]:
    """
    Fit a t-distribution to null SNR samples and return the
    parameters as a serialisable dict

    Args:
        null_snr: 1D array of null-distribution SNR samples

    Returns:
        Dict with keys 'df', 'loc', 'scale' — parameters of the fitted
        t-distribution in raw SNR space.
    """

    if len(null_snr) == 0:
        return {"df": 1.0, "loc": 0.0, "scale": 1.0}
    df_fit, loc_fit, scale_fit = t.fit(null_snr)
    return {"df": float(df_fit), "loc": float(loc_fit), "scale": float(scale_fit)}


def compute_significance_threshold(
    null_params: Dict[str, float],
    alpha: float = 0.05,
) -> float:
    """
    Return the raw SNR value at which the one-sided p-value equals alpha

    An atom whose MUSE score (weighted-average raw SNR over its sphere) equals
    or exceeds this threshold has density support that is statistically
    significant at the given alpha level relative to the protein-region null
    distribution

    Args:
        null_params: Dict with keys 'df', 'loc', 'scale' as returned by
            fit_null_distribution.
        alpha: Significance level. Default 0.05

    Returns:
        SNR threshold value T such that P(SNR > T | null) = alpha
    """

    return float(
        t.ppf(1.0 - alpha, df=null_params["df"], loc=null_params["loc"], scale=null_params["scale"])
    )
