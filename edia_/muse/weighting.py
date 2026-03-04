"""Weight functions for MUSE scoring.

Implements the 3-parabola weighting curve for protein/ligand atoms
(Meyder et al. 2017) and the Gaussian + linear weighting for water
molecules (Nittinger et al. 2015).

The parabolic curve assigns positive weights inside the electron density
sphere s(a) and negative weights in the surrounding donut d(a), with the
volume integral over the full sphere of interest being zero.

References:
    Meyder et al. (2017) SI Table S3, Section 2.1.3
    Nittinger et al. (2015) Equations 1, 3, 4
"""

from __future__ import annotations

import math

import numpy as np

from muse.config import WeightingConfig


def parabolic_weight(
    distance: float,
    radius: float,
    config: WeightingConfig,
) -> float:
    """Compute the 3-parabola weighting value for a single distance.

    The weighting curve is composed of three parabolas P1, P2, P3:
        P(x) = m/r^2 * (x - c*r)^2 + b

    where m, c, b are parameters from the config and r is the atom radius.

    Args:
        distance: Distance from atom center to grid point in Angstroms.
        radius: Atom electron density sphere radius in Angstroms.
        config: Weighting curve parameters.

    Returns:
        Weight value. Positive near center (~1.0 at center), negative in
        the donut region (~-0.4 minimum), zero beyond 2*radius.
    """
    if radius <= 0.0:
        return 0.0

    r = radius

    # Transition points
    t1 = config.transition_1_frac * r
    t2 = config.transition_2_frac * r
    end = config.p3_c_frac * r  # 2.0 * r

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
    """Vectorized 3-parabola weighting for numpy arrays.

    Args:
        distances: 1D array of distances in Angstroms.
        radius: Atom radius in Angstroms.
        config: Weighting curve parameters.

    Returns:
        1D array of weight values, same shape as distances.
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
    """Compute the Gaussian + linear weight for water scoring.

    The weight is a Gaussian centered at the atom with width equal to
    the covalent radius, combined with a linear ramp to zero at the
    van der Waals radius.

    From Nittinger 2015, equations 3-4:
        omega(p) = exp(-0.5 * (dist / delta)^2)
    combined with a linear function that ensures the weight reaches
    zero at the vdW radius.

    Args:
        distance: Distance from atom center to grid point in Angstroms.
        covalent_radius: Covalent radius of the atom (delta parameter).
        vdw_radius: Van der Waals radius of the atom.

    Returns:
        Weight value >= 0. Approximately 1.0 at the atom center,
        decaying to 0 at the vdW radius.
    """
    if distance < 0.0 or distance > vdw_radius:
        return 0.0
    if covalent_radius <= 0.0:
        return 0.0

    delta = covalent_radius

    # Gaussian component
    gauss = math.exp(-0.5 * (distance / delta) ** 2)

    # Linear ramp junction point: p0 = (rvdw/2) + sqrt((rvdw^2/4) - delta^2)
    discriminant = (vdw_radius ** 2 / 4.0) - delta ** 2
    if discriminant < 0.0:
        # If vdw_radius < 2*cov_radius, just use the Gaussian clamped at rvdw
        return gauss

    p0 = vdw_radius / 2.0 + math.sqrt(discriminant)
    omega_p0 = math.exp(-0.5 * (p0 / delta) ** 2)

    if distance <= p0:
        return gauss
    else:
        # Linear ramp from (p0, omega_p0) to (rvdw, 0)
        slope = omega_p0 / (p0 - vdw_radius)
        return slope * (distance - vdw_radius)


def gaussian_weight_vectorized(
    distances: np.ndarray,
    covalent_radius: float,
    vdw_radius: float,
) -> np.ndarray:
    """Vectorized Gaussian + linear weight for water scoring.

    Args:
        distances: 1D array of distances in Angstroms.
        covalent_radius: Covalent radius (delta parameter).
        vdw_radius: Van der Waals radius.

    Returns:
        1D array of weight values >= 0.
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

    # Gaussian region: [0, p0]
    gauss_mask = valid & (distances <= p0)
    weights[gauss_mask] = gauss[gauss_mask]

    # Linear ramp region: (p0, rvdw]
    linear_mask = valid & (distances > p0)
    if np.any(linear_mask):
        slope = omega_p0 / (p0 - vdw_radius)
        weights[linear_mask] = slope * (distances[linear_mask] - vdw_radius)

    return weights
