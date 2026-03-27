from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional

import gemmi
import numpy as np

from analyse.water_sites.alignment import SuperpositionResult, transform_to_structure
from analyse.water_sites.clustering import WaterSite
from analyse.water_sites.config import WaterSiteConfig


_WATER_NAMES = {"HOH", "WAT", "DOD", "H2O", "SOL"}

_STANDARD_AA = {
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
    "MSE", "SEP", "TPO", "CSO", "PTR", "HYP", "CME", "OCS",
}

# Monatomic ion residue names as used in PDB/mmCIF files.
# Classification is based on residue name (not element) to avoid
# misclassifying covalently bonded atoms (e.g. Cl in an organic ligand)
# as ions. In PDB convention, monatomic ions are stored as single-atom
# residues whose residue name matches the ion code.
_ION_RESIDUE_NAMES = {
    # Alkali metals
    "LI", "NA", "K", "RB", "CS",
    # Alkaline earth
    "MG", "CA", "SR", "BA",
    # Transition / heavy metals (including common oxidation-state variants)
    "V", "CR",
    "MN", "MN2", "MN3",
    "FE", "FE2", "FE3",
    "CO",
    "NI",
    "CU", "CU1", "CU2",
    "ZN", "ZN2",
    "Y",
    "AG", "CD",
    "LA", "CE", "PR", "ND", "SM", "EU", "GD", "TB", "DY", "HO", "ER", "TM", "YB", "LU",
    "HG", "AU", "PT",
    "AL", "GA", "IN", "SN", "TL", "PB",
    # Halogens as anions
    "F", "CL", "BR", "IOD",
}


class OccupancyType(str, Enum):
    WATER = "water"
    PROTEIN = "protein"
    LIGAND = "ligand"
    ION = "ion"
    EMPTY = "empty"


@dataclass
class SiteOccupancy:
    """
    Occupancy classification for one (water site, structure) pair.

    Attributes:
        site_id: Water site identifier.
        stem: Experiment stem.
        occupancy_type: Classified occupant type.
        residue_name: Residue name of the nearest atom (empty when EMPTY).
        chain_id: Chain of the nearest atom.
        seq_id: Sequence number of the nearest residue.
        distance_to_centroid: Distance in Å from site centroid to nearest atom.
            NaN when no atom was found within the search radius.
    """

    site_id: int
    stem: str
    occupancy_type: OccupancyType
    residue_name: str
    chain_id: str
    seq_id: int
    distance_to_centroid: float


def _classify(residue_name: str, element_name: str) -> OccupancyType:
    rn = residue_name.upper()
    if rn in _WATER_NAMES:
        return OccupancyType.WATER
    if rn in _STANDARD_AA:
        return OccupancyType.PROTEIN
    if rn in _ION_RESIDUE_NAMES:
        return OccupancyType.ION
    return OccupancyType.LIGAND


def check_all_site_occupancy_for_structure(
    sites: List[WaterSite],
    paths: dict,
    transform: SuperpositionResult,
    config: WaterSiteConfig,
) -> List[SiteOccupancy]:
    """
    For each water site find the nearest heavy atom in the original crystal
    structure within occupancy_search_radius and classify its type.

    The original PDB is loaded once and a single NeighborSearch covers all
    sites. Uses original_pdb for all structures (never the debiased model),
    reflecting the actual crystallographic content at each site.

    Args:
        sites: Water sites with reference-frame centroids.
        paths: Experiment paths dict.
        transform: SuperpositionResult used to map centroids to this
            structure's frame.
        config: WaterSiteConfig with occupancy_search_radius.

    Returns:
        One SiteOccupancy per site, in the same order as sites.
    """
    stem = paths["stem"]
    original_pdb = paths["original_pdb"]

    def _empty(site: WaterSite) -> SiteOccupancy:
        return SiteOccupancy(
            site_id=site.site_id,
            stem=stem,
            occupancy_type=OccupancyType.EMPTY,
            residue_name="",
            chain_id="",
            seq_id=0,
            distance_to_centroid=float("nan"),
        )

    if not original_pdb.exists():
        return [_empty(s) for s in sites]

    try:
        structure = gemmi.read_structure(str(original_pdb))
    except Exception:
        return [_empty(s) for s in sites]

    model = structure[0]
    # Build neighbor search with a small buffer beyond the search radius
    ns = gemmi.NeighborSearch(model, structure.cell, config.occupancy_search_radius + 0.5)
    ns.populate(include_h=False)

    results: List[SiteOccupancy] = []
    for site in sites:
        centroid_local = transform_to_structure(site.centroid, transform)
        center = gemmi.Position(*centroid_local.tolist())

        marks = ns.find_atoms(center, "\0", radius=config.occupancy_search_radius)

        best_dist = float("inf")
        best: Optional[SiteOccupancy] = None

        for mark in marks:
            if mark.image_idx != 0:
                continue
            cra = mark.to_cra(model)
            if cra.atom is None or cra.atom.element.is_hydrogen:
                continue
            dist = float(center.dist(cra.atom.pos))
            if dist < best_dist:
                best_dist = dist
                best = SiteOccupancy(
                    site_id=site.site_id,
                    stem=stem,
                    occupancy_type=_classify(
                        cra.residue.name, cra.atom.element.name
                    ),
                    residue_name=cra.residue.name,
                    chain_id=cra.chain.name,
                    seq_id=cra.residue.seqid.num,
                    distance_to_centroid=best_dist,
                )

        results.append(best if best is not None else _empty(site))

    return results
