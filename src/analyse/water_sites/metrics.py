from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import List

import numpy as np
from scipy.stats import ttest_1samp

from analyse.water_sites.occupancy import OccupancyType, SiteOccupancy
from analyse.water_sites.site_snr import PerStructureSiteSNR

# Minimum number of per-structure SNR observations required to compute a
# meaningful one-sample t-test against the SNR noise floor (1.0).
_MIN_N_FOR_TTEST = 3

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
    )
