from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import gemmi
import numpy as np
from scipy.spatial import cKDTree
from sklearn.cluster import DBSCAN

from analyse.water_sites.alignment import SuperpositionResult, transform_to_reference
from analyse.water_sites.config import WaterSiteConfig


_WATER_RESIDUE_NAMES = {"HOH", "WAT", "DOD", "H2O", "SOL"}


@dataclass
class WaterObservation:
    """A single modelled water oxygen in the reference coordinate frame."""

    stem: str
    chain_id: str
    seq_id: int
    position_ref: np.ndarray  # (3,) in reference frame


@dataclass
class WaterSite:
    """
    A conserved water site defined by clustering observations across the screen.

    Attributes:
        site_id: Sequential integer identifier.
        centroid: Mean position of member waters in the reference frame (Å).
        radius: Distance from centroid to the most distant member water (Å),
            floored at WaterSiteConfig.min_site_radius.
        n_waters: Number of modelled waters contributing to this site.
        member_stems: Stems of the structures that have a water here.
        observations: Individual water observations in this cluster.
    """

    site_id: int
    centroid: np.ndarray  # (3,)
    radius: float
    n_waters: int
    member_stems: List[str]
    observations: List[WaterObservation] = field(default_factory=list)


def collect_waters(
    experiments: List[dict],
    transforms: Dict[str, SuperpositionResult],
) -> List[WaterObservation]:
    """
    Collect all water oxygen positions from every experiment and transform
    them into the reference coordinate frame.

    Uses original_pdb and skips alternate conformations (keeps only the
    default / altloc A atoms) and hydrogens/deuterium.

    Args:
        experiments: List of paths dicts from find_experiments.
        transforms: Mapping of stem → SuperpositionResult (mobile→reference).

    Returns:
        Flat list of WaterObservation instances.
    """
    observations: List[WaterObservation] = []

    for paths in experiments:
        stem = paths["stem"]
        pdb_path = paths["original_pdb"]
        if not pdb_path.exists():
            continue

        try:
            structure = gemmi.read_structure(str(pdb_path))
        except Exception:
            continue

        transform = transforms.get(stem)
        model = structure[0]

        for chain in model:
            for residue in chain:
                if residue.name.upper() not in _WATER_RESIDUE_NAMES:
                    continue
                for atom in residue:
                    if atom.element.is_hydrogen:
                        continue
                    if atom.altloc not in ("\x00", " ", "A", ""):
                        continue
                    pos = np.array(
                        [atom.pos.x, atom.pos.y, atom.pos.z], dtype=np.float64
                    )
                    pos_ref = (
                        transform_to_reference(pos, transform)
                        if transform is not None
                        else pos
                    )
                    observations.append(
                        WaterObservation(
                            stem=stem,
                            chain_id=chain.name,
                            seq_id=residue.seqid.num,
                            position_ref=pos_ref,
                        )
                    )
                    break  # one oxygen per water residue

    return observations


def _constrained_split(
    obs_subset: List[WaterObservation],
    positions: np.ndarray,
    eps: float,
) -> List[List[int]]:
    """
    Split a cluster that contains duplicate stems into valid sub-clusters
    using constrained single-linkage (Union-Find).

    Each sub-cluster may contain at most one water per stem. Pairs are
    considered in ascending distance order, and a merge is only performed
    when the resulting cluster would not violate the uniqueness constraint.

    Args:
        obs_subset: Observations in this cluster (in same order as positions).
        positions: (N, 3) positions for the observations.
        eps: Maximum distance between mergeable waters.

    Returns:
        List of index groups, each group forming a valid sub-cluster.
    """
    n = len(obs_subset)
    if n == 1:
        return [[0]]

    tree = cKDTree(positions)
    pairs = tree.query_pairs(r=eps, output_type="ndarray")

    if len(pairs) > 0:
        dists = np.linalg.norm(
            positions[pairs[:, 0]] - positions[pairs[:, 1]], axis=1
        )
        pairs = pairs[np.argsort(dists)]

    parent = list(range(n))
    cluster_stems: List[set] = [{obs_subset[i].stem} for i in range(n)]

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in pairs:
        ra, rb = find(int(a)), find(int(b))
        if ra == rb:
            continue
        if cluster_stems[ra].isdisjoint(cluster_stems[rb]):
            parent[rb] = ra
            cluster_stems[ra] |= cluster_stems[rb]

    sub: Dict[int, List[int]] = defaultdict(list)
    for i in range(n):
        sub[find(i)].append(i)
    return list(sub.values())


def cluster_water_observations(
    observations: List[WaterObservation],
    config: WaterSiteConfig,
) -> List[WaterSite]:
    """
    Cluster water observations into water sites.

    Uses DBSCAN for initial grouping, then applies constrained single-linkage
    splitting to guarantee that each cluster contains at most one water per
    structure. Sites are sorted by decreasing n_waters (most conserved first).

    Args:
        observations: All water observations in the reference frame.
        config: WaterSiteConfig controlling eps, min_samples, and radius floor.

    Returns:
        List of WaterSite objects sorted by decreasing n_waters.
    """
    if not observations:
        return []

    positions = np.array(
        [o.position_ref for o in observations], dtype=np.float64
    )

    labels = DBSCAN(
        eps=config.clustering_eps,
        min_samples=config.clustering_min_samples,
        metric="euclidean",
        n_jobs=1,
    ).fit_predict(positions)

    # Group by DBSCAN label; noise points (label=-1) become singleton clusters.
    # With min_samples=1 (the default) every point is a core point so no noise
    # can be produced, but we handle it correctly for non-default configs.
    raw_clusters: Dict[int, List[int]] = defaultdict(list)
    noise_offset = int(labels.max()) + 1 if labels.max() >= 0 else 0
    for idx, label in enumerate(labels):
        if label == -1:
            raw_clusters[noise_offset].append(idx)
            noise_offset += 1
        else:
            raw_clusters[label].append(idx)

    # Enforce max-one-per-stem constraint; split violating clusters
    valid_groups: List[List[WaterObservation]] = []
    for member_indices in raw_clusters.values():
        obs_sub = [observations[i] for i in member_indices]
        stems = [o.stem for o in obs_sub]
        if len(stems) == len(set(stems)):
            valid_groups.append(obs_sub)
        else:
            pos_sub = np.array([o.position_ref for o in obs_sub], dtype=np.float64)
            for sub_idx_list in _constrained_split(obs_sub, pos_sub, config.clustering_eps):
                valid_groups.append([obs_sub[i] for i in sub_idx_list])

    # Build WaterSite objects sorted by descending n_waters
    sites: List[WaterSite] = []
    for obs_group in sorted(valid_groups, key=len, reverse=True):
        pos_arr = np.array([o.position_ref for o in obs_group], dtype=np.float64)
        centroid = pos_arr.mean(axis=0)
        dists = np.linalg.norm(pos_arr - centroid[np.newaxis, :], axis=1)
        radius = max(float(np.max(dists)), config.min_site_radius)
        sites.append(
            WaterSite(
                site_id=len(sites),
                centroid=centroid,
                radius=radius,
                n_waters=len(obs_group),
                member_stems=[o.stem for o in obs_group],
                observations=obs_group,
            )
        )

    return sites
