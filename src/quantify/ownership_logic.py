from __future__ import annotations

from typing import Dict, List, Tuple
import json
import numpy as np
import gemmi
from pathlib import Path

from scipy.spatial import KDTree
from typing import Any, Optional

from quantify.resources import get_atom_radius


def parse_atom_key(key: str) -> Tuple[str, str, str, str, str]:
    """
    Parses Atom Key: "A|1856|SER|N|\u0000"
    """
    parts = key.split("|")
    altloc = parts[4] if (len(parts) > 4 and parts[4] != "\u0000") else ""
    return parts[0], parts[1], parts[2], parts[3], altloc


def parse_residue_key(key: str) -> Tuple[str, str, str]:
    """
    Parses Residue Key: "A|1856|SER"
    """
    parts = key.split("|")
    return parts[0], parts[1], parts[2]


def load_omission_map(json_path: Path) -> Dict[str, List[int]]:
    """Loads the omission map JSON file."""
    with open(json_path, "r") as f:
        return json.load(f)


def _update_max_map(map_ids, max_map_id_seen):
    curr = max(map_ids)
    if curr > max_map_id_seen:
        return curr
    else:
        return max_map_id_seen


def build_spatial_index(
    pdb_path: Path,
    omission_map: Dict,
    resolution: float,
    k: float = 1.0,
    mode: str = "atoms",
) -> Dict[str, Any] | None:
    """
    Builds the spatial index (KDTree) and maps atoms to their logical owners.
    mode='atoms': Logical owner is the atom key itself.
    mode='amino_acids': Logical owner is the residue key (grouping all atoms of the residue).
    """

    try:
        structure = gemmi.read_structure(str(pdb_path))
    except Exception as e:
        print(f"Error reading PDB structure: {e}")
        return None

    atom_positions = []
    atom_radii = []
    atom_to_owner = []
    owner_schedules = {}

    max_map_id_seen = 0

    try:
        if mode == "amino_acids":
            for res_key, map_ids in omission_map.items():
                owner_schedules[res_key] = map_ids
                max_map_id_seen = _update_max_map(map_ids, max_map_id_seen)

                try:
                    chain_id, res_seq, res_name = parse_residue_key(res_key)
                    sel = gemmi.Selection(f"/1/{chain_id}/{res_seq}")
                    residue = sel.copy_structure_selection(structure)[0][0][0]

                    for atom in residue:
                        if atom.element.name == "H":
                            continue

                        pos = atom.pos
                        rad = get_atom_radius(atom.element.name, resolution) * k

                        atom_positions.append([pos.x, pos.y, pos.z])
                        atom_radii.append(rad)
                        atom_to_owner.append(res_key)
                except Exception:
                    continue

        else:
            for atom_key, map_ids in omission_map.items():
                owner_schedules[atom_key] = map_ids
                max_map_id_seen = _update_max_map(map_ids, max_map_id_seen)
                try:
                    chain_id, res_seq, res_name, atom_name, altloc = parse_atom_key(
                        atom_key
                    )

                    if altloc == "":
                        sel = gemmi.Selection(f"/1/{chain_id}/{res_seq}/{atom_name}")
                    else:
                        sel = gemmi.Selection(
                            f"/1/{chain_id}/{res_seq}/{atom_name}:{altloc}"
                        )

                    atom = sel.copy_structure_selection(structure)[0][0][0][0]
                    pos = atom.pos

                    rad = get_atom_radius(atom.element.name, resolution) * k

                    atom_positions.append([pos.x, pos.y, pos.z])
                    atom_radii.append(rad)
                    atom_to_owner.append(atom_key)
                except Exception:
                    continue

    except Exception as e:
        print(f"Index build error: {e}")
        return None

    if not atom_positions:
        print("Warning: No atoms found matching the omission map keys.")
        return None

    return {
        "kdtree": KDTree(atom_positions),
        "positions": atom_positions,
        "radii": np.array(atom_radii),
        "atom_to_owner": atom_to_owner,
        "owner_schedules": owner_schedules,
        "max_radius": np.max(atom_radii),
        "max_map_id_seen": max_map_id_seen,
    }


def query_voxel_ownership(
    spatial_index: Dict[str, Any],
    grid_coords: List[float],
    n_maps: Optional[int] = None,
) -> Optional[np.ndarray]:
    """
    Returns status_matrix [n_maps, N_UNIQUE_OWNERS].
    Handles deduplication of owners.
    """
    if not spatial_index:
        return None

    kdtree = spatial_index["kdtree"]
    radii = spatial_index["radii"]
    max_dist = spatial_index["max_radius"]

    # Coarse search
    indices = kdtree.query_ball_point(grid_coords, r=max_dist)
    if not indices:
        return None

    unique_active_owners = set()
    positions = spatial_index["positions"]
    atom_to_owner = spatial_index["atom_to_owner"]

    for idx in indices:
        dist = np.linalg.norm(np.array(grid_coords) - np.array(positions[idx]))
        if dist <= radii[idx]:
            unique_active_owners.add(atom_to_owner[idx])

    if not unique_active_owners:
        return None

    active_owners_list = list(unique_active_owners)

    status_matrix = np.ones((n_maps, len(active_owners_list)), dtype=np.float32)
    owner_schedules = spatial_index["owner_schedules"]

    for col_idx, owner_key in enumerate(active_owners_list):
        omitted_maps = owner_schedules.get(owner_key, [])
        for map_id in omitted_maps:
            if 0 <= map_id < n_maps:
                status_matrix[map_id, col_idx] = 0.0

    return status_matrix
