import numpy as np
import gemmi
from quantify.ownership_logic import query_voxel_ownership


def solve_voxel(values: np.ndarray, status: np.ndarray):
    """
    Extracts true background intensity and residual phase bias.
      - If status is None: Return mean/std.
      - If status exists (Occupied voxels):
         - If column has both 0s and 1s, calculate bias.
         - all 1s or all 0s: bias = 0.
    """
    if status is None:
        return [np.mean(values), np.std(values, ddof=1), values.tolist()]

    n_owners = status.shape[1]
    biases = np.zeros(n_owners)

    for j in range(n_owners):
        col = status[:, j]

        has_present = np.any(col == 1.0)
        has_omitted = np.any(col == 0.0)

        if has_present and has_omitted:
            biases[j] = np.mean(values[col == 1.0]) - np.mean(values[col == 0.0])
        else:
            # We keep 0 bias value for the all exist case to not stop the STOMP map generation for partial perturbations
            biases[j] = 0.0

    total_bias_vector = np.dot(status, biases)
    cleaned = values - total_bias_vector
    result = [np.mean(cleaned), np.std(cleaned, ddof=1), cleaned.tolist()]

    return result


def aggregate_ensemble(
    ensemble_data: np.ndarray, ref_grid: gemmi.FloatGrid, spatial_index: dict
):
    nx, ny, nz = ensemble_data.shape[1:]
    n_maps = ensemble_data.shape[0]

    sig_map = np.zeros((nx, ny, nz), dtype=np.float32)
    nos_map = np.zeros((nx, ny, nz), dtype=np.float32)
    snr_map = np.zeros((nx, ny, nz), dtype=np.float32)
    distributions = np.empty((nx, ny, nz), dtype=object)

    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                pos = ref_grid.get_position(i, j, k)
                coords = [pos.x, pos.y, pos.z]

                vals = ensemble_data[:, i, j, k]
                status = query_voxel_ownership(spatial_index, coords, n_maps)

                s, n, d = solve_voxel(vals, status)

                sig_map[i, j, k] = s
                nos_map[i, j, k] = n
                snr_map[i, j, k] = s / n if n > 1e-12 else 0.0
                distributions[i, j, k] = d

    return sig_map, nos_map, snr_map, distributions
