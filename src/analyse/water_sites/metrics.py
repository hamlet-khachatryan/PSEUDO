from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from scipy.stats import ttest_1samp

from analyse.water_sites.occupancy import OccupancyType, SiteOccupancy
from analyse.water_sites.site_snr import PerStructureSiteSNR

# Minimum number of per-structure SNR observations required to compute a
# meaningful one-sample t-test against the SNR noise floor (1.0).
_MIN_N_FOR_TTEST = 3

# Minimum number of per-structure distributions required to form at least
# one pair for the pairwise KL computation.
_MIN_N_FOR_KL = 2

# Guard against zero-variance distributions in KL computation.
_KL_STD_EPSILON = 1e-6

# SNR noise floor: signal at or below this value is indistinguishable from
# background in a well-processed SNR map.
SNR_NOISE_FLOOR = 1.0


@dataclass
class WaterSiteConsistency:
    """
    Screen-level consistency metrics for a single water site.

    Attributes:
        water_frequency: Fraction of structures with a modelled water here.
        snr_mean_across_screen: Mean of per-structure mean-SNR values
            (site-radius sphere) across the full screen.
        snr_std_across_screen: Standard deviation of the same distribution.
        snr_cv: Coefficient of variation (std / |mean|). Lower = more
            consistent signal. NaN when mean is zero.
        snr_above_1_fraction: Fraction of structures whose mean SNR within
            the site-radius sphere exceeds 1.0.
        consistency_score: 1 / (1 + CV), bounded [0, 1].
            1 = perfectly uniform signal; 0 = maximally variable.
            NaN when CV is undefined.
        most_common_occupancy: Most frequent OccupancyType across all
            structures (by value string).
        snr_tstat: One-sample t-statistic testing whether the distribution
            of per-structure mean-SNR values is greater than SNR_NOISE_FLOOR.
            NaN when fewer than _MIN_N_FOR_TTEST observations are available.
        snr_pvalue: One-sided p-value (H₀: μ_SNR ≤ 1.0, Hₐ: μ_SNR > 1.0).
            NaN when fewer than _MIN_N_FOR_TTEST observations are available.
        mean_pairwise_kl: Mean symmetric KL divergence between all pairs of
            per-structure SNR distributions (Gaussian approximation via
            mean and std of the within-sphere SNR values). Measures how
            different the SNR distributions are from one another across the
            screen. Low = distributions are nearly identical (reproducible
            signal); high = distributions vary substantially between
            structures. NaN when fewer than _MIN_N_FOR_KL valid distributions
            are available.
    """

    water_frequency: float
    snr_mean_across_screen: float
    snr_std_across_screen: float
    snr_cv: float
    snr_above_1_fraction: float
    consistency_score: float
    most_common_occupancy: str
    snr_tstat: float
    snr_pvalue: float
    mean_pairwise_kl: float


def _symmetric_kl(mu1: float, s1: float, mu2: float, s2: float) -> float:
    """
    Symmetric KL divergence between two univariate Gaussians.

    D_sym(P||Q) = (KL(P||Q) + KL(Q||P)) / 2

    For N(μ₁, σ₁²) and N(μ₂, σ₂²) this simplifies to:
        (σ₁²/σ₂² + σ₂²/σ₁² - 2 + (μ₁-μ₂)²·(1/σ₁² + 1/σ₂²)) / 4
    """
    var1, var2 = s1 * s1, s2 * s2
    diff_sq = (mu1 - mu2) ** 2
    return (var1 / var2 + var2 / var1 - 2.0 + diff_sq * (1.0 / var1 + 1.0 / var2)) / 4.0


def _mean_pairwise_kl(per_structure_snr: List[PerStructureSiteSNR]) -> float:
    """
    Compute the mean symmetric KL divergence between all pairs of per-structure
    SNR distributions for one water site.

    Each structure's distribution is approximated as a Gaussian parameterised
    by the within-sphere SNR mean and std. Only records with n_points ≥ 2
    (needed for a meaningful std) are included. Returns NaN when fewer than
    _MIN_N_FOR_KL valid distributions are available.
    """
    params: List[Tuple[float, float]] = []
    for rec in per_structure_snr:
        stats = rec.snr_site_radius
        if stats is None or stats.n_points < 2:
            continue
        mu = stats.mean
        sigma = max(stats.std, _KL_STD_EPSILON)
        params.append((mu, sigma))

    if len(params) < _MIN_N_FOR_KL:
        return float("nan")

    total = 0.0
    count = 0
    for i in range(len(params)):
        for j in range(i + 1, len(params)):
            total += _symmetric_kl(params[i][0], params[i][1],
                                   params[j][0], params[j][1])
            count += 1

    return total / count


def _most_common_occupancy(occupancies: List[SiteOccupancy]) -> str:
    counts = Counter(o.occupancy_type.value for o in occupancies)
    if not counts:
        return OccupancyType.EMPTY.value
    return counts.most_common(1)[0][0]


def compute_consistency(
    per_structure_snr: List[PerStructureSiteSNR],
    per_structure_occ: List[SiteOccupancy],
    n_total_structures: int,
) -> WaterSiteConsistency:
    """
    Compute screen-level consistency metrics for one water site.

    Args:
        per_structure_snr: All PerStructureSiteSNR records for this site.
        per_structure_occ: All SiteOccupancy records for this site.
        n_total_structures: Total number of experiments in the screen.

    Returns:
        WaterSiteConsistency with all metrics populated.
    """
    n_with_water = sum(1 for s in per_structure_snr if s.has_water_modelled)
    water_frequency = (
        n_with_water / n_total_structures if n_total_structures > 0 else 0.0
    )

    means = [
        s.snr_site_radius.mean
        for s in per_structure_snr
        if s.snr_site_radius is not None
    ]

    nan = float("nan")
    occ_label = _most_common_occupancy(per_structure_occ)

    pairwise_kl = _mean_pairwise_kl(per_structure_snr)

    if not means:
        return WaterSiteConsistency(
            water_frequency=water_frequency,
            snr_mean_across_screen=nan,
            snr_std_across_screen=nan,
            snr_cv=nan,
            snr_above_1_fraction=nan,
            consistency_score=nan,
            most_common_occupancy=occ_label,
            snr_tstat=nan,
            snr_pvalue=nan,
            mean_pairwise_kl=pairwise_kl,
        )

    arr = np.array(means, dtype=np.float64)
    mean_snr = float(np.mean(arr))
    std_snr = float(np.std(arr))
    cv = std_snr / abs(mean_snr) if mean_snr != 0.0 else nan
    above_1 = float(np.mean(arr > SNR_NOISE_FLOOR))
    consistency = 1.0 / (1.0 + cv) if not np.isnan(cv) else nan

    if len(arr) >= _MIN_N_FOR_TTEST:
        t_res = ttest_1samp(arr, popmean=SNR_NOISE_FLOOR, alternative="greater")
        t_stat = float(t_res.statistic)
        p_val = float(t_res.pvalue)
    else:
        t_stat, p_val = nan, nan

    return WaterSiteConsistency(
        water_frequency=water_frequency,
        snr_mean_across_screen=mean_snr,
        snr_std_across_screen=std_snr,
        snr_cv=cv,
        snr_above_1_fraction=above_1,
        consistency_score=consistency,
        most_common_occupancy=occ_label,
        snr_tstat=t_stat,
        snr_pvalue=p_val,
        mean_pairwise_kl=pairwise_kl,
    )
