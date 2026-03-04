"""Main MUSE scoring pipeline and CSV export utilities.

Orchestrates the full calculation: loads map and structure, scores every
heavy atom, runs diagnostics, aggregates by residue and returns a
structured result object.

Typical usage::

    from muse import run_muse
    from muse.config import MUSEConfig, MapNormalizationConfig

    result = run_muse(
        map_path="path/to/map.ccp4",
        structure_path="path/to/model.pdb",
        resolution=2.0,
    )
    export_atom_csv(result, "atoms.csv")
    export_residue_csv(result, "residues.csv")

References:
    Meyder et al. (2017) J. Chem. Inf. Model. 57, 2437-2447
    Nittinger et al. (2015) J. Chem. Inf. Model. 55, 771-783
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import gemmi
import numpy as np

from muse.aggregation import aggregate_by_residue, compute_opia
from muse.config import MUSEConfig, default_config
from muse.diagnostics import run_diagnostics
from muse.io import compute_map_statistics, load_map, load_structure
from muse.ownership import AtomId, find_covalent_neighbors
from muse.radii import get_atom_radius
from muse.scoring import AtomScore, MUSEResult, score_protein_atom, score_water_atom

logger = logging.getLogger(__name__)

# Residue names recognised as water molecules.
_WATER_RESIDUE_NAMES: Set[str] = {"HOH", "WAT", "DOD", "H2O", "SOL"}

# Standard vdW radius for oxygen (Angstroms), used for water scoring.
_OXYGEN_VDW_RADIUS: float = 1.52


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _is_water(residue: gemmi.Residue) -> bool:
    """Return True if a residue represents a water molecule.

    Args:
        residue: A gemmi Residue.

    Returns:
        True if the residue name is in the known water name set.
    """
    return residue.name.upper() in _WATER_RESIDUE_NAMES


def _atom_charge(atom: gemmi.Atom) -> int:
    """Extract the formal charge of a gemmi atom as an integer.

    Args:
        atom: A gemmi Atom.

    Returns:
        Integer formal charge. Returns 0 if unset.
    """
    charge = atom.charge
    return int(charge) if charge is not None else 0


def _build_atom_registry(
    model: gemmi.Model,
    resolution: float,
    config: MUSEConfig,
    skip_hydrogens: bool = True,
) -> Tuple[
    Dict[AtomId, np.ndarray],   # id -> Cartesian position
    Dict[AtomId, float],         # id -> radius
    Dict[AtomId, bool],          # id -> is_water
    Dict[AtomId, Tuple[str, str, int, str, str, str]],  # id -> metadata
]:
    """Pre-build per-atom data arrays for the full model.

    Iterates the model once to collect atom positions, radii and metadata
    so that subsequent per-atom scoring loops can access them by AtomId.

    Args:
        model: The gemmi Model to iterate.
        resolution: Map resolution in Angstroms for radius lookup.
        config: Full MUSE configuration.
        skip_hydrogens: If True (default), skip hydrogen atoms.

    Returns:
        Four dicts keyed by AtomId:
        - positions: Cartesian position as (3,) float64 array.
        - radii: Resolution-dependent radius in Angstroms.
        - is_water_map: Whether the atom belongs to a water residue.
        - metadata: (chain_id, residue_name, seq_id_str, ins_code, atom_name, element).
    """
    positions: Dict[AtomId, np.ndarray] = {}
    radii: Dict[AtomId, float] = {}
    is_water_map: Dict[AtomId, bool] = {}
    metadata: Dict[AtomId, Tuple[str, str, int, str, str, str]] = {}

    for chain_idx, chain in enumerate(model):
        chain_id = chain.name
        for res_idx, residue in enumerate(chain):
            water = _is_water(residue)
            res_name = residue.name
            seq_id = residue.seqid.num
            ins_code = str(residue.seqid.icode).strip()

            for atom_idx, atom in enumerate(residue):
                # Skip hydrogen and deuterium
                if skip_hydrogens and atom.element in (
                    gemmi.Element("H"), gemmi.Element("D")
                ):
                    continue

                # Skip non-default alternate conformations
                if atom.altloc not in ("\x00", " ", "A", ""):
                    continue

                atom_id: AtomId = (chain_idx, res_idx, atom_idx)
                pos = np.array(
                    [atom.pos.x, atom.pos.y, atom.pos.z], dtype=np.float64
                )
                charge = _atom_charge(atom)
                element = atom.element.name

                try:
                    radius = get_atom_radius(element, charge, resolution)
                except Exception:
                    # Fallback – get_atom_radius already handles unknowns
                    radius = get_atom_radius(element, 0, resolution)

                positions[atom_id] = pos
                radii[atom_id] = radius
                is_water_map[atom_id] = water
                metadata[atom_id] = (
                    chain_id, res_name, seq_id, ins_code,
                    atom.name, element,
                )

    return positions, radii, is_water_map, metadata


def _build_neighbor_search(
    model: gemmi.Model,
    cell: gemmi.UnitCell,
    max_radius: float,
    skip_hydrogens: bool = True,
) -> gemmi.NeighborSearch:
    """Build a gemmi NeighborSearch over all heavy atoms in the model.

    Args:
        model: The gemmi Model.
        cell: Unit cell from the parent structure.
        max_radius: Maximum search radius in Angstroms.
        skip_hydrogens: If True (default), hydrogens are excluded.

    Returns:
        A populated gemmi.NeighborSearch instance.
    """
    ns = gemmi.NeighborSearch(model, cell, max_radius)
    ns.populate(include_h=not skip_hydrogens)
    return ns


def _get_nearby_atoms(
    atom_id: AtomId,
    atom_pos: gemmi.Position,
    atom_radius: float,
    ns: gemmi.NeighborSearch,
    positions: Dict[AtomId, np.ndarray],
    radii: Dict[AtomId, float],
    search_radius_buffer: float = 3.0,
) -> List[Tuple[AtomId, np.ndarray, float]]:
    """Find atoms whose spheres could overlap with the given atom's 2r sphere.

    Args:
        atom_id: The AtomId of the scored atom.
        atom_pos: Its gemmi.Position.
        atom_radius: Its density sphere radius.
        ns: Pre-built NeighborSearch.
        positions: Registry of all atom positions.
        radii: Registry of all atom radii.
        search_radius_buffer: Extra Angstroms added to the search radius.

    Returns:
        List of (AtomId, position(3,), radius) for neighbouring atoms.
    """
    # Search within 2*r_a + buffer to catch all atoms whose spheres could
    # overlap with the 2r sphere of the scored atom.
    search_r = 2.0 * atom_radius + search_radius_buffer

    nearby: List[Tuple[AtomId, np.ndarray, float]] = []
    marks = ns.find_atoms(atom_pos, "\0", radius=search_r)

    for mark in marks:
        nbr_id: AtomId = (mark.chain_idx, mark.residue_idx, mark.atom_idx)
        if nbr_id == atom_id:
            continue
        # Exclude symmetry images for ownership purposes
        if mark.image_idx != 0:
            continue
        if nbr_id not in positions:
            continue
        nearby.append((nbr_id, positions[nbr_id], radii.get(nbr_id, 1.0)))

    return nearby


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_muse(
    map_path: str,
    structure_path: str,
    resolution: float,
    config: Optional[MUSEConfig] = None,
    skip_hydrogens: bool = True,
    run_error_diagnostics: bool = True,
) -> MUSEResult:
    """Run the full MUSE scoring pipeline.

    Loads the CCP4 map and coordinate model, scores every heavy atom
    (using the Nittinger 2015 variant for water oxygen atoms and the
    Meyder 2017 variant for all others), applies diagnostic flags,
    aggregates by residue and returns a MUSEResult.

    Args:
        map_path: Path to the CCP4/MRC map file.
        structure_path: Path to the PDB or mmCIF coordinate file.
        resolution: Map resolution in Angstroms. Used to look up
            resolution-dependent radii from Table S2. Typical values:
            1.0–3.0 A.
        config: MUSE configuration. If None, uses default_config()
            (normalization disabled, all paper-derived parameters).
        skip_hydrogens: If True (default), hydrogen atoms are not scored.
        run_error_diagnostics: If True (default), apply clash / missing /
            unaccounted density flags after scoring.

    Returns:
        A MUSEResult with atom_scores, residue_scores, OPIA and metadata.
    """
    if config is None:
        config = default_config()

    # --- I/O ---
    logger.info("Loading map: %s", map_path)
    grid = load_map(map_path)

    logger.info("Loading structure: %s", structure_path)
    structure = load_structure(structure_path)

    mean, sigma = compute_map_statistics(grid, config.map_normalization)
    logger.info("Map statistics: mean=%.4f, sigma=%.4f", mean, sigma)

    model = structure[0]  # First model (standard practice)

    # --- Per-atom data ---
    logger.info("Building atom registry...")
    positions, radii, is_water_map, metadata = _build_atom_registry(
        model, resolution, config, skip_hydrogens
    )
    logger.info("Found %d heavy atoms.", len(positions))

    # --- Covalent bond graph ---
    logger.info("Building covalent neighbor graph...")
    cov_neighbors = find_covalent_neighbors(
        model, config.ownership.covalent_bond_tolerance
    )

    # --- Neighbor search for ownership ---
    max_search_r = 2.0 * max(radii.values(), default=2.0) + 3.0
    ns = _build_neighbor_search(model, structure.cell, max_search_r, skip_hydrogens)

    # --- Oxygen parameters for water scoring ---
    oxy = gemmi.Element("O")
    oxygen_cov_r: float = float(oxy.covalent_r)

    # --- Per-atom scoring ---
    logger.info("Scoring atoms...")
    atom_scores: List[AtomScore] = []

    for chain_idx, chain in enumerate(model):
        for res_idx, residue in enumerate(chain):
            water = _is_water(residue)
            for atom_idx, atom in enumerate(residue):
                # Skip hydrogens
                if skip_hydrogens and atom.element in (
                    gemmi.Element("H"), gemmi.Element("D")
                ):
                    continue
                # Skip non-default altlocs
                if atom.altloc not in ("\x00", " ", "A", ""):
                    continue

                atom_id: AtomId = (chain_idx, res_idx, atom_idx)
                if atom_id not in positions:
                    continue

                pos = positions[atom_id]
                radius = radii[atom_id]
                chain_id, res_name, seq_id, ins_code, atom_name, element = metadata[atom_id]

                if water:
                    # Water oxygen: Nittinger 2015 variant
                    score, score_pos, score_neg, n_pts = score_water_atom(
                        atom_pos=pos,
                        grid=grid,
                        mean=mean,
                        sigma=sigma,
                        covalent_radius=oxygen_cov_r,
                        vdw_radius=_OXYGEN_VDW_RADIUS,
                        config=config,
                    )
                else:
                    # Protein / ligand: Meyder 2017 variant
                    gemmi_pos = gemmi.Position(pos[0], pos[1], pos[2])
                    nearby = _get_nearby_atoms(
                        atom_id, gemmi_pos, radius, ns, positions, radii
                    )
                    score, score_pos, score_neg, n_pts = score_protein_atom(
                        atom_pos=pos,
                        atom_radius=radius,
                        atom_id=atom_id,
                        grid=grid,
                        mean=mean,
                        sigma=sigma,
                        covalent_neighbors=cov_neighbors,
                        nearby_atoms=nearby,
                        config=config,
                    )

                atom_scores.append(AtomScore(
                    chain_id=chain_id,
                    residue_name=res_name,
                    residue_seq_id=seq_id,
                    insertion_code=ins_code,
                    atom_name=atom_name,
                    element=element,
                    score=score,
                    score_positive=score_pos,
                    score_negative=score_neg,
                    is_water=water,
                    radius_used=radius,
                    n_grid_points=n_pts,
                    atom_id=atom_id,
                ))

    logger.info("Scored %d atoms.", len(atom_scores))

    # --- Diagnostics ---
    if run_error_diagnostics:
        logger.info("Running error diagnostics...")
        run_diagnostics(
            atom_scores=atom_scores,
            model=model,
            atom_id_to_position=positions,
            atom_id_to_radius=radii,
            config=config.aggregation,
        )

    # --- Aggregation ---
    logger.info("Aggregating residue scores...")
    residue_scores = aggregate_by_residue(atom_scores, config.aggregation)

    logger.info("Computing OPIA...")
    opia = compute_opia(atom_scores, cov_neighbors, config.aggregation.opia_threshold)

    logger.info(
        "Done. OPIA=%.3f, %d residues scored.", opia, len(residue_scores)
    )

    return MUSEResult(
        atom_scores=atom_scores,
        residue_scores=residue_scores,
        opia=opia,
        global_mean=mean,
        global_sigma=sigma,
        config=config,
    )


# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------

def export_atom_csv(result: MUSEResult, output_path: str) -> None:
    """Write per-atom MUSE scores to a CSV file.

    Each row corresponds to one heavy atom. Columns include chain, residue,
    atom identifiers, MUSE score, diagnostic flags and grid point count.

    Args:
        result: A completed MUSEResult.
        output_path: Destination path for the CSV file. Parent directories
            must exist.
    """
    fieldnames = [
        "chain_id",
        "residue_name",
        "residue_seq_id",
        "insertion_code",
        "atom_name",
        "element",
        "score",
        "score_positive",
        "score_negative",
        "is_water",
        "has_clash",
        "has_missing_density",
        "has_unaccounted_density",
        "radius_used",
        "n_grid_points",
    ]
    _write_csv(output_path, fieldnames, [
        {
            "chain_id": a.chain_id,
            "residue_name": a.residue_name,
            "residue_seq_id": a.residue_seq_id,
            "insertion_code": a.insertion_code,
            "atom_name": a.atom_name,
            "element": a.element,
            "score": f"{a.score:.6f}",
            "score_positive": f"{a.score_positive:.6f}",
            "score_negative": f"{a.score_negative:.6f}",
            "is_water": a.is_water,
            "has_clash": a.has_clash,
            "has_missing_density": a.has_missing_density,
            "has_unaccounted_density": a.has_unaccounted_density,
            "radius_used": f"{a.radius_used:.4f}",
            "n_grid_points": a.n_grid_points,
        }
        for a in result.atom_scores
    ])
    logger.info("Atom scores written to %s", output_path)


def export_residue_csv(result: MUSEResult, output_path: str) -> None:
    """Write per-residue MUSEm scores to a CSV file.

    Each row corresponds to one residue. Columns include chain, residue
    identifiers, MUSEm (power-mean) score and atom-level statistics.

    Args:
        result: A completed MUSEResult.
        output_path: Destination path for the CSV file.
    """
    fieldnames = [
        "chain_id",
        "residue_name",
        "residue_seq_id",
        "insertion_code",
        "musem_score",
        "min_atom_score",
        "median_atom_score",
        "max_atom_score",
        "n_atoms",
        "n_clashes",
        "n_missing_density",
        "n_unaccounted_density",
    ]
    _write_csv(output_path, fieldnames, [
        {
            "chain_id": r.chain_id,
            "residue_name": r.residue_name,
            "residue_seq_id": r.residue_seq_id,
            "insertion_code": r.insertion_code,
            "musem_score": f"{r.musem_score:.6f}",
            "min_atom_score": f"{r.min_atom_score:.6f}",
            "median_atom_score": f"{r.median_atom_score:.6f}",
            "max_atom_score": f"{r.max_atom_score:.6f}",
            "n_atoms": r.n_atoms,
            "n_clashes": sum(1 for a in r.atom_scores if a.has_clash),
            "n_missing_density": sum(
                1 for a in r.atom_scores if a.has_missing_density
            ),
            "n_unaccounted_density": sum(
                1 for a in r.atom_scores if a.has_unaccounted_density
            ),
        }
        for r in result.residue_scores
    ])
    logger.info("Residue scores written to %s", output_path)


def export_summary(result: MUSEResult) -> dict:
    """Return a flat summary dict of key MUSE statistics.

    Useful for logging, reporting or quick inspection without writing files.

    Args:
        result: A completed MUSEResult.

    Returns:
        Dictionary with keys: n_atoms, n_residues, opia, mean_atom_score,
        median_atom_score, n_clashes, n_missing_density,
        n_unaccounted_density, global_mean, global_sigma.
    """
    scores = [a.score for a in result.atom_scores]
    return {
        "n_atoms": len(result.atom_scores),
        "n_residues": len(result.residue_scores),
        "opia": result.opia,
        "mean_atom_score": float(np.mean(scores)) if scores else 0.0,
        "median_atom_score": float(np.median(scores)) if scores else 0.0,
        "n_clashes": sum(1 for a in result.atom_scores if a.has_clash),
        "n_missing_density": sum(
            1 for a in result.atom_scores if a.has_missing_density
        ),
        "n_unaccounted_density": sum(
            1 for a in result.atom_scores if a.has_unaccounted_density
        ),
        "global_mean": result.global_mean,
        "global_sigma": result.global_sigma,
    }


# ---------------------------------------------------------------------------
# Internal I/O helper
# ---------------------------------------------------------------------------

def _write_csv(output_path: str, fieldnames: List[str], rows: List[dict]) -> None:
    """Write rows to a CSV file with a header.

    Args:
        output_path: Destination file path.
        fieldnames: Ordered list of column names.
        rows: List of dicts with keys matching fieldnames.
    """
    path = Path(output_path)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
