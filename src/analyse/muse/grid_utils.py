"""
References:
    - Meyder et al. (2017) J. Chem. Inf. Model. 57, 2437-2447
    - Nittinger et al. (2015) J. Chem. Inf. Model. 55, 771-783
"""

from __future__ import annotations

import math
from typing import Tuple

import gemmi
import numpy as np


def get_grid_spacings(grid: gemmi.FloatGrid) -> Tuple[float, float, float]:
    """
    Compute the grid spacing along each unit cell axis

    Args:
        grid: The CCP4 map grid
    Returns:
        Tuple of (spacing_a, spacing_b, spacing_c)
    """
    uc = grid.unit_cell
    na, nb, nc = grid.nu, grid.nv, grid.nw
    return (uc.a / na, uc.b / nb, uc.c / nc)


def compute_oversampling_factors(
    grid: gemmi.FloatGrid,
    max_spacing: float,
) -> Tuple[int, int, int]:
    """
    Compute per-axis oversampling factors

    Args:
        grid: The CCP4 map grid
        max_spacing: Maximum allowed spacing in Angstroms
    Returns:
        Tuple of (factor_a, factor_b, factor_c)
    """
    spacings = get_grid_spacings(grid)
    factors = tuple(max(1, math.ceil(s / max_spacing)) for s in spacings)
    return factors


def enumerate_grid_points_in_sphere(
    grid: gemmi.FloatGrid,
    center: gemmi.Position,
    radius: float,
    max_spacing: float = 0.7,
    interpolation_order: int = 3,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Enumerate all grid points within a sphere and return positions + values

    Args:
        grid: The CCP4 map grid
        center: Center of the sphere (atom position)
        radius: Sphere radius
        max_spacing: Maximum grid spacing before oversampling
        interpolation_order: Interpolation order for sub-grid values 1 = trilinear, 3 = tricubic
    Returns:
        Tuple of (positions, values)
    """
    uc = grid.unit_cell
    os_factors = compute_oversampling_factors(grid, max_spacing)

    enu = grid.nu * os_factors[0]
    env = grid.nv * os_factors[1]
    enw = grid.nw * os_factors[2]

    frac = uc.fractionalize(center)
    frac_radius_a = radius / uc.a
    frac_radius_b = radius / uc.b
    frac_radius_c = radius / uc.c

    u_min = int(math.floor((frac.x - frac_radius_a) * enu))
    u_max = int(math.ceil((frac.x + frac_radius_a) * enu))
    v_min = int(math.floor((frac.y - frac_radius_b) * env))
    v_max = int(math.ceil((frac.y + frac_radius_b) * env))
    w_min = int(math.floor((frac.z - frac_radius_c) * enw))
    w_max = int(math.ceil((frac.z + frac_radius_c) * enw))

    radius_sq = radius * radius
    cx, cy, cz = center.x, center.y, center.z

    positions_list = []
    values_list = []

    for u in range(u_min, u_max + 1):
        fu = u / enu
        for v in range(v_min, v_max + 1):
            fv = v / env
            for w in range(w_min, w_max + 1):
                fw = w / enw

                pos = uc.orthogonalize(gemmi.Fractional(fu, fv, fw))
                dx = pos.x - cx
                dy = pos.y - cy
                dz = pos.z - cz
                dist_sq = dx * dx + dy * dy + dz * dz

                if dist_sq <= radius_sq:
                    val = grid.interpolate_value(gemmi.Fractional(fu, fv, fw))
                    positions_list.append((pos.x, pos.y, pos.z))
                    values_list.append(val)

    if not positions_list:
        return np.empty((0, 3), dtype=np.float64), np.empty(0, dtype=np.float64)

    positions = np.array(positions_list, dtype=np.float64)
    values = np.array(values_list, dtype=np.float64)
    return positions, values


def compute_distances(
    positions: np.ndarray,
    center: np.ndarray,
) -> np.ndarray:
    """
    Compute distances from each position to a center point

    Args:
        positions: ndarray of shape (N, 3) point coordinates
        center: ndarray of shape (3,) center coordinates

    Returns:
        ndarray of shape (N,) distances
    """
    diff = positions - center[np.newaxis, :]
    return np.sqrt(np.sum(diff * diff, axis=1))
