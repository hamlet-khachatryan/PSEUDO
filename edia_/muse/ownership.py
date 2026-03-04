"""Grid point ownership logic for covalently connected atoms.

Implements the ownership function o(p, a) from Meyder et al. 2017 (Figure 2),
which determines how density at a grid point is attributed among atoms whose
spheres overlap. Covalently bonded atoms share density fully, while non-bonded
overlapping atoms compete based on distance.

The 4-case decision tree:
    1. a in S(p) and sole non-covalent atom in S(p) -> o = 1.0
    2. a in S(p) and multiple non-covalent atoms in S(p) -> distance sharing
    3. a in D(p) and S(p) is non-empty -> o = 0.0
    4. a in D(p) and S(p) is empty -> shared among D(p) atoms

References:
    Meyder et al. (2017) Section "Grid Point Ownership", eq 3, Figure 2
"""

from __future__ import annotations

from typing import Dict, List, Optional, Set, Tuple

import gemmi
import numpy as np


# Type alias for atom identifier: (chain_idx, residue_idx, atom_idx)
AtomId = Tuple[int, int, int]


def find_covalent_neighbors(
    model: gemmi.Model,
    tolerance: float = 0.4,
) -> Dict[AtomId, List[AtomId]]:
    """Build an adjacency list of covalently bonded heavy atoms.

    Two atoms are considered covalently bonded if their distance is
    less than the sum of their covalent radii plus a tolerance.

    Args:
        model: A gemmi.Model (typically structure[0]).
        tolerance: Distance tolerance in Angstroms added to the sum
            of covalent radii. Default 0.4 A.

    Returns:
        Dict mapping each atom's AtomId to a list of bonded atom AtomIds.
        Only heavy atoms (not H/D) are included.
    """
    adjacency: Dict[AtomId, List[AtomId]] = {}

    # Collect all heavy atoms with their positions and covalent radii
    atoms_info: List[Tuple[AtomId, gemmi.Position, float]] = []
    for ci, chain in enumerate(model):
        for ri, residue in enumerate(chain):
            for ai, atom in enumerate(residue):
                if atom.element.is_hydrogen:
                    continue
                aid = (ci, ri, ai)
                adjacency[aid] = []
                atoms_info.append((aid, atom.pos, atom.element.covalent_r))

    # Build bonds by pairwise distance check
    # For efficiency, use a simple spatial grid or brute-force for moderate sizes
    n = len(atoms_info)
    for i in range(n):
        aid_i, pos_i, cr_i = atoms_info[i]
        for j in range(i + 1, n):
            aid_j, pos_j, cr_j = atoms_info[j]

            dx = pos_i.x - pos_j.x
            dy = pos_i.y - pos_j.y
            dz = pos_i.z - pos_j.z
            dist = (dx * dx + dy * dy + dz * dz) ** 0.5

            max_bond_dist = cr_i + cr_j + tolerance
            if dist <= max_bond_dist:
                adjacency[aid_i].append(aid_j)
                adjacency[aid_j].append(aid_i)

    return adjacency


def find_covalent_neighbors_fast(
    model: gemmi.Model,
    structure: gemmi.Structure,
    tolerance: float = 0.4,
) -> Dict[AtomId, List[AtomId]]:
    """Build covalent bond adjacency using gemmi.NeighborSearch for speed.

    Preferred over find_covalent_neighbors for large structures.

    Args:
        model: A gemmi.Model (typically structure[0]).
        structure: The parent gemmi.Structure (needed for NeighborSearch).
        tolerance: Distance tolerance in Angstroms. Default 0.4 A.

    Returns:
        Dict mapping each AtomId to a list of bonded AtomIds.
    """
    adjacency: Dict[AtomId, List[AtomId]] = {}

    # Build index maps: sequential_index -> AtomId and AtomId -> info
    atom_list: List[Tuple[AtomId, float]] = []  # (aid, covalent_radius)
    aid_by_cra: Dict[Tuple[str, str, str], AtomId] = {}

    for ci, chain in enumerate(model):
        for ri, residue in enumerate(chain):
            for ai, atom in enumerate(residue):
                if atom.element.is_hydrogen:
                    continue
                aid = (ci, ri, ai)
                adjacency[aid] = []
                atom_list.append((aid, atom.element.covalent_r))
                # Use chain/residue/atom as lookup key
                cra_key = (chain.name, f"{residue.seqid}", atom.name)
                aid_by_cra[cra_key] = aid

    # Use gemmi NeighborSearch
    ns = gemmi.NeighborSearch(model, structure.cell, 3.5).populate()

    for ci, chain in enumerate(model):
        for ri, residue in enumerate(chain):
            for ai, atom in enumerate(residue):
                if atom.element.is_hydrogen:
                    continue
                aid_i = (ci, ri, ai)
                cr_i = atom.element.covalent_r
                marks = ns.find_atoms(atom.pos, '\0', 0, cr_i + 2.0 + tolerance)
                for mark in marks:
                    cra = mark.to_cra(model)
                    if cra.atom is None:
                        continue
                    neighbor_atom = cra.atom
                    if neighbor_atom.element.is_hydrogen:
                        continue
                    if neighbor_atom is atom:
                        continue

                    # Check bond distance
                    cr_j = neighbor_atom.element.covalent_r
                    dist = atom.pos.dist(neighbor_atom.pos)
                    if dist <= cr_i + cr_j + tolerance:
                        # Find the AtomId of the neighbor
                        for ci2, chain2 in enumerate(model):
                            if chain2.name != cra.chain.name:
                                continue
                            for ri2, res2 in enumerate(chain2):
                                if str(res2.seqid) != str(cra.residue.seqid) or res2.name != cra.residue.name:
                                    continue
                                for ai2, at2 in enumerate(res2):
                                    if at2.name == neighbor_atom.name and not at2.element.is_hydrogen:
                                        aid_j = (ci2, ri2, ai2)
                                        if aid_j != aid_i and aid_j not in adjacency[aid_i]:
                                            adjacency[aid_i].append(aid_j)
                                        break
                                break
                            break

    return adjacency


def compute_ownership(
    distances_to_atoms: Dict[AtomId, float],
    scored_atom: AtomId,
    atom_radii: Dict[AtomId, float],
    covalent_neighbors: Dict[AtomId, List[AtomId]],
) -> float:
    """Compute ownership factor o(p, a) for one grid point and one atom.

    Implements the 4-case decision tree from Meyder 2017 Figure 2.

    Args:
        distances_to_atoms: Dict mapping each nearby AtomId to the distance
            from the grid point to that atom's center.
        scored_atom: The AtomId being scored (atom a).
        atom_radii: Dict mapping each AtomId to its electron density radius.
        covalent_neighbors: Bond adjacency from find_covalent_neighbors.

    Returns:
        Ownership fraction in [0.0, 1.0].
    """
    a = scored_atom
    r_a = atom_radii.get(a, 0.0)
    dist_a = distances_to_atoms.get(a, float('inf'))

    # Determine S(p) and D(p) sets
    # S(p): atoms whose inner sphere (radius r) contains the grid point
    # D(p): atoms whose sphere of interest (radius 2r) contains the point
    #        but whose inner sphere does not
    s_p: Set[AtomId] = set()
    d_p: Set[AtomId] = set()

    for atom_id, dist in distances_to_atoms.items():
        r = atom_radii.get(atom_id, 0.0)
        if dist <= r:
            s_p.add(atom_id)
        elif dist <= 2.0 * r:
            d_p.add(atom_id)

    cov_of_a = set(covalent_neighbors.get(a, []))

    # Case: a is in S(p) (grid point is inside a's inner sphere)
    if a in s_p:
        # I(p, a) = S(p) minus atoms covalently bonded to a
        # (but includes a itself)
        i_p_a = s_p - cov_of_a
        # Note: a is not its own covalent neighbor, so a stays in I
        if len(i_p_a) <= 1:
            # a is the only non-covalent atom -> full ownership
            return 1.0
        else:
            # Share among non-covalent atoms by distance
            return _share_by_distance(dist_a, i_p_a, distances_to_atoms)

    # Case: a is in D(p) (grid point is in a's donut region)
    if a in d_p:
        if len(s_p) > 0:
            # Density belongs to atoms closer in (in their inner sphere)
            return 0.0
        else:
            # No atom claims this point in their inner sphere;
            # share among all D(p) atoms
            return _share_by_distance(dist_a, d_p, distances_to_atoms)

    # a is not in S(p) or D(p) for this grid point
    return 0.0


def _share_by_distance(
    dist_a: float,
    atom_set: Set[AtomId],
    distances: Dict[AtomId, float],
) -> float:
    """Share ownership among atoms based on inverse distance.

    From Meyder 2017 eq 3:
        o(p, a) = 1 - dist_a / sum(dist_b for b in X)  when |X| > 1

    Args:
        dist_a: Distance from grid point to atom a.
        atom_set: Set of atoms sharing this grid point.
        distances: Distance from grid point to each atom.

    Returns:
        Ownership fraction for atom a.
    """
    if len(atom_set) <= 1:
        return 1.0

    total_dist = sum(distances.get(b, 0.0) for b in atom_set)
    if total_dist == 0.0:
        return 1.0 / len(atom_set)

    return 1.0 - dist_a / total_dist


def compute_ownership_vectorized(
    grid_distances_to_scored: np.ndarray,
    grid_point_positions: np.ndarray,
    scored_atom_id: AtomId,
    scored_atom_radius: float,
    nearby_atoms: List[Tuple[AtomId, np.ndarray, float]],
    covalent_neighbors: Dict[AtomId, List[AtomId]],
) -> np.ndarray:
    """Compute ownership for all grid points of one atom at once.

    This is the performance-critical path. For each grid point, we determine
    which nearby atoms contain it in their S or D region, then apply the
    4-case decision tree.

    Args:
        grid_distances_to_scored: (N,) array of distances from each grid
            point to the scored atom.
        grid_point_positions: (N, 3) array of grid point positions.
        scored_atom_id: The AtomId being scored.
        scored_atom_radius: Radius of the scored atom.
        nearby_atoms: List of (AtomId, position_array(3,), radius) for
            atoms near the scored atom (within 2*max_radius of any
            grid point).
        covalent_neighbors: Bond adjacency dict.

    Returns:
        (N,) array of ownership values.
    """
    n_points = len(grid_distances_to_scored)
    ownership = np.ones(n_points, dtype=np.float64)

    if n_points == 0 or not nearby_atoms:
        return ownership

    cov_of_a = set(covalent_neighbors.get(scored_atom_id, []))

    # Precompute distances from all grid points to all nearby atoms
    # nearby_dists[j] is an (N,) array of distances from grid points to atom j
    nearby_dists = []
    nearby_ids = []
    nearby_radii = []
    for atom_id, atom_pos, atom_r in nearby_atoms:
        diff = grid_point_positions - atom_pos[np.newaxis, :]
        dists = np.sqrt(np.sum(diff * diff, axis=1))
        nearby_dists.append(dists)
        nearby_ids.append(atom_id)
        nearby_radii.append(atom_r)

    # For each grid point, determine ownership
    for pi in range(n_points):
        dist_a = grid_distances_to_scored[pi]
        r_a = scored_atom_radius

        # Build S(p) and D(p) for this grid point
        s_p = set()
        d_p = set()

        for j, (aid, ar) in enumerate(zip(nearby_ids, nearby_radii)):
            d = nearby_dists[j][pi]
            if d <= ar:
                s_p.add(aid)
            elif d <= 2.0 * ar:
                d_p.add(aid)

        # Also check the scored atom itself
        if dist_a <= r_a:
            s_p.add(scored_atom_id)
        elif dist_a <= 2.0 * r_a:
            d_p.add(scored_atom_id)

        # Apply decision tree
        if scored_atom_id in s_p:
            i_p_a = s_p - cov_of_a
            if len(i_p_a) <= 1:
                ownership[pi] = 1.0
            else:
                # Distance sharing
                dists_map = {}
                for aid in i_p_a:
                    if aid == scored_atom_id:
                        dists_map[aid] = dist_a
                    else:
                        idx = nearby_ids.index(aid)
                        dists_map[aid] = nearby_dists[idx][pi]
                total = sum(dists_map.values())
                if total > 0:
                    ownership[pi] = 1.0 - dist_a / total
                else:
                    ownership[pi] = 1.0 / len(i_p_a)

        elif scored_atom_id in d_p:
            if len(s_p) > 0:
                ownership[pi] = 0.0
            else:
                if len(d_p) <= 1:
                    ownership[pi] = 1.0
                else:
                    dists_map = {}
                    for aid in d_p:
                        if aid == scored_atom_id:
                            dists_map[aid] = dist_a
                        else:
                            idx = nearby_ids.index(aid)
                            dists_map[aid] = nearby_dists[idx][pi]
                    total = sum(dists_map.values())
                    if total > 0:
                        ownership[pi] = 1.0 - dist_a / total
                    else:
                        ownership[pi] = 1.0 / len(d_p)
        else:
            ownership[pi] = 0.0

    return ownership
