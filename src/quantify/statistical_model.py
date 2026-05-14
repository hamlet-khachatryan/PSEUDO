from __future__ import annotations

import logging
from typing import Dict, Optional

import gemmi
import numpy as np
from scipy.optimize import minimize
from scipy.stats import gaussian_kde, t
from pathlib import Path

logger = logging.getLogger(__name__)


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


def fit_t_test(null_params: Dict[str, float], full_snr_map: np.ndarray) -> np.ndarray:
    """
    Calculates the survival function (1 - CDF) of a pre-fitted t-distribution
    for every voxel in the map, representing the p-value against the null

    Args:
        null_params: Fitted t-distribution parameters as returned by
            fit_null_distribution — keys 'df', 'loc', 'scale'.
        full_snr_map: The full 3D raw SNR map array.

    Returns:
        3D array of p-values in [0.0, 1.0]. Low values indicate statistically
        significant SNR — i.e., density unlikely to arise from background noise.
    """

    p_values = t.sf(
        full_snr_map,
        df=null_params["df"],
        loc=null_params["loc"],
        scale=null_params["scale"],
    )
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
        null_params: Dict with keys 'df', 'loc', 'scale' as returned by fit_null_distribution.
        alpha: Significance level. Default 0.05

    Returns:
        SNR threshold value T such that P(SNR > T | null) = alpha
    """

    return float(
        t.ppf(1.0 - alpha, df=null_params["df"], loc=null_params["loc"], scale=null_params["scale"])
    )


def fit_null_truncated_mle(
    samples: np.ndarray,
    truncation_point: Optional[float] = None,
) -> Dict[str, float]:
    """
    Fit a t-distribution to null SNR samples using truncated MLE on the left half.

    Because signal contamination is one-sided positive, observations at or below
    the empirical mode are essentially signal-free. Fitting only those samples with
    the truncation likelihood correction gives unbiased null parameters even when
    the right tail of the background sample is polluted by ordered waters or
    unmodelled density.

    Falls back to fit_null_distribution() with a WARNING if the optimizer fails or
    too few samples lie below the truncation point.

    Args:
        samples: 1D array of null-distribution SNR samples.
        truncation_point: SNR value used as the upper truncation bound. If None,
            estimated as the empirical mode via KDE.

    Returns:
        Dict with keys 'df', 'loc', 'scale'.
    """
    if len(samples) == 0:
        return {"df": 1.0, "loc": 0.0, "scale": 1.0}

    n_full = len(samples)

    if truncation_point is None:
        p1, p99 = np.percentile(samples, [1, 99])
        grid = np.linspace(p1, p99, 1000)
        kde = gaussian_kde(samples)
        density = kde(grid)
        mode_idx = int(np.argmax(density))
        truncation_point = float(grid[mode_idx])

        top_mask = density >= 0.99 * density[mode_idx]
        top_span = float(grid[top_mask].max() - grid[top_mask].min())
        total_span = float(p99 - p1)
        if total_span > 0 and top_span / total_span > 0.20:
            logger.warning(
                "KDE mode estimation: top-1%% density spans %.1f%% of the grid range "
                "(possible multi-modal or plateau-shaped null). "
                "Single-t assumption may not hold.",
                100.0 * top_span / total_span,
            )

    b = float(truncation_point)
    left = samples[samples <= b]
    n_left = len(left)

    if n_left < 1000:
        logger.warning(
            "Only %d samples at or below truncation point %.3f (minimum 1000 required). "
            "Falling back to full-sample t.fit.",
            n_left, b,
        )
        return fit_null_distribution(samples)

    loc_init = float(np.median(left))
    mad = float(np.median(np.abs(left - loc_init)))
    scale_init = max(1.4826 * mad, 1e-6)

    def neg_loglik(params: np.ndarray) -> float:
        log_df, loc, log_scale = params
        df_ = np.exp(log_df)
        scale_ = np.exp(log_scale)
        ll = float(np.sum(t.logpdf(left, df=df_, loc=loc, scale=scale_)))
        ll -= n_left * float(t.logcdf(b, df=df_, loc=loc, scale=scale_))
        return -ll

    x0 = [np.log(10.0), loc_init, np.log(scale_init)]
    result = minimize(
        neg_loglik,
        x0,
        method="Nelder-Mead",
        options={"maxiter": 5000, "xatol": 1e-6, "fatol": 1e-6},
    )

    if not result.success:
        logger.warning(
            "Truncated MLE did not converge (%s). Falling back to full-sample t.fit.",
            result.message,
        )
        return fit_null_distribution(samples)

    df_fit = float(np.exp(result.x[0]))
    loc_fit = float(result.x[1])
    scale_fit = float(np.exp(result.x[2]))

    logger.info(
        "Truncated MLE fit: n_full=%d, n_left=%d, truncation_point=%.4f, "
        "df=%.4f, loc=%.4f, scale=%.4f",
        n_full, n_left, b, df_fit, loc_fit, scale_fit,
    )

    return {"df": df_fit, "loc": loc_fit, "scale": scale_fit}


def estimate_null_fraction(
    samples: np.ndarray,
    df: float,
    loc: float,
    scale: float,
) -> float:
    """
    Compute Efron's pi_0 upper bound: min over a window around the mode of
    f_hat(z) / f0(z), where f_hat is a KDE of the full sample and f0 is the
    fitted null t-distribution.

    pi_0 ~ 1.0 means the background is clean; lower values indicate signal
    contamination in the null sample.

    Args:
        samples: Full 1D array of null SNR samples.
        df, loc, scale: Fitted t-distribution parameters.

    Returns:
        Estimated null fraction in (0, 1], capped at 1.0.
    """
    if len(samples) == 0:
        return 1.0

    p1, p99 = np.percentile(samples, [1, 99])
    grid = np.linspace(p1, p99, 1000)
    kde = gaussian_kde(samples)
    kde_density = kde(grid)

    mode = float(grid[np.argmax(kde_density)])

    q25, q75 = np.percentile(samples, [25, 75])
    half_iqr = 0.5 * float(q75 - q25)
    mask = (grid >= mode - half_iqr) & (grid <= mode + half_iqr)
    if not mask.any():
        return 1.0

    model_density = t.pdf(grid[mask], df=df, loc=loc, scale=scale)
    valid = model_density > 1e-10
    if not valid.any():
        return 1.0

    ratios = kde_density[mask][valid] / model_density[valid]
    return float(min(float(np.min(ratios)), 1.0))
