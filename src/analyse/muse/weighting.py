"""
References:
    - Meyder et al. (2017) J. Chem. Inf. Model. 57, 2437-2447
    - Nittinger et al. (2015) J. Chem. Inf. Model. 55, 771-783
"""

from __future__ import annotations

import math

import numpy as np

from analyse.muse.config import WeightingConfig


def parabolic_weight(
    distance: float,
    radius: float,
    config: WeightingConfig,
) -> float:
    """
    Compute the 3-parabola weighting value for a single distance

    Args:
        distance: Distance from atom center to grid point
        radius: Atom electron density sphere radius
        config: Weighting curve parameters

    Returns:
        Weight value
    """
    if radius <= 0.0:
        return 0.0

    r = radius

    t1 = config.transition_1_frac * r
    t2 = config.transition_2_frac * r
    end = config.p3_c_frac * r

    if distance < 0.0 or distance > end:
        return 0.0

    if distance <= t1:
        # P1 region
        c = config.p1_c_frac * r
        m = config.p1_m / (r * r)
        return m * (distance - c) ** 2 + config.p1_b

    elif distance <= t2:
        # P2 region
        c = config.p2_c_frac * r
        m = config.p2_m / (r * r)
        return m * (distance - c) ** 2 + config.p2_b

    else:
        # P3 region
        c = config.p3_c_frac * r
        m = config.p3_m / (r * r)
        return m * (distance - c) ** 2 + config.p3_b


def parabolic_weight_vectorized(
    distances: np.ndarray,
    radius: float,
    config: WeightingConfig,
) -> np.ndarray:
    """
    Vectorized 3-parabola weighting for numpy arrays

    Args:
        distances: 1D array of distances
        radius: Atom radius
        config: Weighting curve parameters

    Returns:
        1D array of weight values
    """
    if radius <= 0.0:
        return np.zeros_like(distances)

    r = radius
    weights = np.zeros_like(distances)

    t1 = config.transition_1_frac * r
    t2 = config.transition_2_frac * r
    end = config.p3_c_frac * r

    # P1 region: [0, t1]
    mask1 = (distances >= 0.0) & (distances <= t1)
    if np.any(mask1):
        c1 = config.p1_c_frac * r
        m1 = config.p1_m / (r * r)
        weights[mask1] = m1 * (distances[mask1] - c1) ** 2 + config.p1_b

    # P2 region: (t1, t2]
    mask2 = (distances > t1) & (distances <= t2)
    if np.any(mask2):
        c2 = config.p2_c_frac * r
        m2 = config.p2_m / (r * r)
        weights[mask2] = m2 * (distances[mask2] - c2) ** 2 + config.p2_b

    # P3 region: (t2, end]
    mask3 = (distances > t2) & (distances <= end)
    if np.any(mask3):
        c3 = config.p3_c_frac * r
        m3 = config.p3_m / (r * r)
        weights[mask3] = m3 * (distances[mask3] - c3) ** 2 + config.p3_b

    return weights


def gaussian_weight(
    distance: float,
    covalent_radius: float,
    vdw_radius: float,
) -> float:
    """
    Compute the Gaussian + linear weight for water scoring

    Args:
        distance: Distance from atom center to grid point
        covalent_radius: Covalent radius of the atom
        vdw_radius: Van der Waals radius of the atom

    Returns:
        Weight value >= 0
        decaying to 0 at the vdW radius
    """
    if distance < 0.0 or distance > vdw_radius:
        return 0.0
    if covalent_radius <= 0.0:
        return 0.0

    delta = covalent_radius

    # gaussian
    gauss = math.exp(-0.5 * (distance / delta) ** 2)

    # linear ramp junction point
    discriminant = (vdw_radius ** 2 / 4.0) - delta ** 2
    if discriminant < 0.0:
        return gauss

    p0 = vdw_radius / 2.0 + math.sqrt(discriminant)
    omega_p0 = math.exp(-0.5 * (p0 / delta) ** 2)

    if distance <= p0:
        return gauss
    else:
        slope = omega_p0 / (p0 - vdw_radius)
        return slope * (distance - vdw_radius)


def gaussian_weight_vectorized(
    distances: np.ndarray,
    covalent_radius: float,
    vdw_radius: float,
) -> np.ndarray:
    """
    Vectorized Gaussian + linear weight for water scoring

    Args:
        distances: 1D array of distances
        covalent_radius: Covalent radius
        vdw_radius: Van der Waals radius

    Returns:
        1D array of weight values
    """
    if covalent_radius <= 0.0 or vdw_radius <= 0.0:
        return np.zeros_like(distances)

    delta = covalent_radius
    weights = np.zeros_like(distances)
    valid = (distances >= 0.0) & (distances <= vdw_radius)

    if not np.any(valid):
        return weights

    gauss = np.exp(-0.5 * (distances / delta) ** 2)

    discriminant = (vdw_radius ** 2 / 4.0) - delta ** 2
    if discriminant < 0.0:
        weights[valid] = gauss[valid]
        return weights

    p0 = vdw_radius / 2.0 + math.sqrt(discriminant)
    omega_p0 = math.exp(-0.5 * (p0 / delta) ** 2)

    # gaussian region: [0, p0]
    gauss_mask = valid & (distances <= p0)
    weights[gauss_mask] = gauss[gauss_mask]

    # linear ramp region: (p0, rvdw]
    linear_mask = valid & (distances > p0)
    if np.any(linear_mask):
        slope = omega_p0 / (p0 - vdw_radius)
        weights[linear_mask] = slope * (distances[linear_mask] - vdw_radius)

    return weights
