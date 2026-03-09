"""
References:
    Meyder et al. (2017) J. Chem. Inf. Model. 57, 2437-2447
    Nittinger et al. (2015) J. Chem. Inf. Model. 55, 771-783
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import eliot
import gemmi
import numpy as np

from analyse.muse.aggregation import aggregate_by_residue, compute_opia
from analyse.muse.config import MUSEConfig, default_config
from analyse.muse.diagnostics import run_diagnostics
from analyse.muse.io import compute_map_statistics, load_map, load_structure
from analyse.muse.ownership import AtomId, find_covalent_neighbors
from analyse.muse.radii import get_atom_radius
from analyse.muse.scoring import AtomScore, MUSEResult, score_protein_atom, score_water_atom

_WATER_RESIDUE_NAMES: Set[str] = {"HOH", "WAT", "DOD", "H2O", "SOL"}
_OXYGEN_VDW_RADIUS: float = 1.52

def _is_water(residue: gemmi.Residue) -> bool:
    return residue.name.upper() in _WATER_RESIDUE_NAMES


def _atom_charge(atom: gemmi.Atom) -> int:
    charge = atom.charge
    return int(charge) if charge is not None else 0


def _build_atom_registry(
    model: gemmi.Model,
    resolution: float,
    config: MUSEConfig,
    skip_hydrogens: bool = True,
) -> Tuple[
    Dict[AtomId, np.ndarray],
    Dict[AtomId, float],
    Dict[AtomId, bool],
    Dict[AtomId, Tuple[str, str, int, str, str, str]],
]:
    """Pre-build per-atom data arrays for the full model"""
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
                if skip_hydrogens and atom.element in (
                    gemmi.Element("H"), gemmi.Element("D")
                ):
                    continue
                if atom.altloc not in ("\x00", " ", "A", ""):
                    continue

                atom_id: AtomId = (chain_idx, res_idx, atom_idx)
                pos = np.array([atom.pos.x, atom.pos.y, atom.pos.z], dtype=np.float64)
                charge = _atom_charge(atom)
                element = atom.element.name

                try:
                    radius = get_atom_radius(element, charge, resolution)
                except Exception:
                    radius = get_atom_radius(element, 0, resolution)

                positions[atom_id] = pos
                radii[atom_id] = radius
                is_water_map[atom_id] = water
                metadata[atom_id] = (chain_id, res_name, seq_id, ins_code, atom.name, element)

    return positions, radii, is_water_map, metadata


def _build_neighbor_search(
    model: gemmi.Model,
    cell: gemmi.UnitCell,
    max_radius: float,
    skip_hydrogens: bool = True,
) -> gemmi.NeighborSearch:
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
    search_r = 2.0 * atom_radius + search_radius_buffer
    nearby: List[Tuple[AtomId, np.ndarray, float]] = []
    marks = ns.find_atoms(atom_pos, "\0", radius=search_r)
    for mark in marks:
        nbr_id: AtomId = (mark.chain_idx, mark.residue_idx, mark.atom_idx)
        if nbr_id == atom_id:
            continue
        if mark.image_idx != 0:
            continue
        if nbr_id not in positions:
            continue
        nearby.append((nbr_id, positions[nbr_id], radii.get(nbr_id, 1.0)))
    return nearby

def run_muse(
    map_path: str,
    structure_path: str,
    resolution: float,
    config: Optional[MUSEConfig] = None,
    skip_hydrogens: bool = True,
    run_error_diagnostics: bool = True,
) -> MUSEResult:
    """Run the full MUSE scoring pipeline and return a MUSEResult."""
    if config is None:
        config = default_config()

    with eliot.start_action(action_type="muse:load_inputs"):
        grid = load_map(map_path)
        structure = load_structure(structure_path)
        mean, sigma = compute_map_statistics(grid, config.map_normalization)

    model = structure[0]

    with eliot.start_action(action_type="muse:build_atom_registry"):
        positions, radii, is_water_map, metadata = _build_atom_registry(
            model, resolution, config, skip_hydrogens
        )
        cov_neighbors = find_covalent_neighbors(model, config.ownership.covalent_bond_tolerance)
        max_search_r = 2.0 * max(radii.values(), default=2.0) + 3.0
        ns = _build_neighbor_search(model, structure.cell, max_search_r, skip_hydrogens)

    eliot.log_message(
        message_type="muse:registry_built",
        n_atoms=len(positions),
        map_mean=round(float(mean), 4),
        map_sigma=round(float(sigma), 4),
    )

    oxy = gemmi.Element("O")
    oxygen_cov_r: float = float(oxy.covalent_r)
    atom_scores: List[AtomScore] = []

    with eliot.start_action(action_type="muse:scoring", n_atoms=len(positions)):
        for chain_idx, chain in enumerate(model):
            for res_idx, residue in enumerate(chain):
                water = _is_water(residue)
                for atom_idx, atom in enumerate(residue):
                    if skip_hydrogens and atom.element in (
                        gemmi.Element("H"), gemmi.Element("D")
                    ):
                        continue
                    if atom.altloc not in ("\x00", " ", "A", ""):
                        continue

                    atom_id: AtomId = (chain_idx, res_idx, atom_idx)
                    if atom_id not in positions:
                        continue

                    pos = positions[atom_id]
                    radius = radii[atom_id]
                    chain_id, res_name, seq_id, ins_code, atom_name, element = metadata[atom_id]

                    if water:
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
                        gemmi_pos = gemmi.Position(pos[0], pos[1], pos[2])
                        nearby = _get_nearby_atoms(atom_id, gemmi_pos, radius, ns, positions, radii)
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

    eliot.log_message(message_type="muse:scoring_complete", n_atoms_scored=len(atom_scores))

    if run_error_diagnostics:
        with eliot.start_action(action_type="muse:diagnostics"):
            run_diagnostics(
                atom_scores=atom_scores,
                model=model,
                atom_id_to_position=positions,
                atom_id_to_radius=radii,
                config=config.aggregation,
            )

    with eliot.start_action(action_type="muse:aggregation"):
        residue_scores = aggregate_by_residue(atom_scores, config.aggregation)
        opia = compute_opia(atom_scores, cov_neighbors, config.aggregation.opia_threshold)

    eliot.log_message(
        message_type="muse:aggregation_complete",
        opia=round(float(opia), 4),
        n_residues=len(residue_scores),
    )

    return MUSEResult(
        atom_scores=atom_scores,
        residue_scores=residue_scores,
        opia=opia,
        global_mean=mean,
        global_sigma=sigma,
        config=config,
    )

def export_atom_csv(result: MUSEResult, output_path: str) -> None:
    """Write per-atom MUSE scores to a CSV file."""
    fieldnames = [
        "chain_id", "residue_name", "residue_seq_id", "insertion_code",
        "atom_name", "element", "score", "score_positive", "score_negative",
        "is_water", "has_clash", "has_missing_density", "has_unaccounted_density",
        "radius_used", "n_grid_points",
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


def export_residue_csv(result: MUSEResult, output_path: str) -> None:
    """Write per-residue MUSEm scores to a CSV file."""
    fieldnames = [
        "chain_id", "residue_name", "residue_seq_id", "insertion_code",
        "musem_score", "min_atom_score", "median_atom_score", "max_atom_score",
        "n_atoms", "n_clashes", "n_missing_density", "n_unaccounted_density",
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
            "n_missing_density": sum(1 for a in r.atom_scores if a.has_missing_density),
            "n_unaccounted_density": sum(1 for a in r.atom_scores if a.has_unaccounted_density),
        }
        for r in result.residue_scores
    ])


def export_summary(result: MUSEResult) -> dict:
    """Return a flat summary dict of key MUSE statistics (OPIA, counts, diagnostics)."""
    scores = [a.score for a in result.atom_scores]
    return {
        "n_atoms": len(result.atom_scores),
        "n_residues": len(result.residue_scores),
        "opia": result.opia,
        "mean_atom_score": float(np.mean(scores)) if scores else 0.0,
        "median_atom_score": float(np.median(scores)) if scores else 0.0,
        "n_clashes": sum(1 for a in result.atom_scores if a.has_clash),
        "n_missing_density": sum(1 for a in result.atom_scores if a.has_missing_density),
        "n_unaccounted_density": sum(1 for a in result.atom_scores if a.has_unaccounted_density),
        "global_mean": result.global_mean,
        "global_sigma": result.global_sigma,
    }


def write_scored_pdb(
    result: MUSEResult,
    structure_path: str,
    output_path: str,
    score_level: str = "residue",
    score_field: str = "musem",
    score_scale: float = 100.0,
    missing_value: float = 0.0,
) -> None:
    """Write a PDB file with MUSE scores embedded in the B-factor column."""
    if score_level not in ("residue", "atom"):
        raise ValueError("score_level must be 'residue' or 'atom'.")

    structure = gemmi.read_structure(str(structure_path))

    if score_level == "residue":
        _valid = {"musem", "min", "median", "max"}
        if score_field not in _valid:
            raise ValueError(f"For residue-level scoring, score_field must be one of {_valid}.")
        score_map: Dict[Tuple[str, int, str], float] = {}
        for r in result.residue_scores:
            key = (r.chain_id, r.residue_seq_id, r.insertion_code)
            if score_field == "musem":
                score_map[key] = r.musem_score
            elif score_field == "min":
                score_map[key] = r.min_atom_score
            elif score_field == "median":
                score_map[key] = r.median_atom_score
            else:
                score_map[key] = r.max_atom_score
    else:
        _valid_atom = {"score", "score_positive", "score_negative"}
        if score_field not in _valid_atom:
            raise ValueError(f"For atom-level scoring, score_field must be one of {_valid_atom}.")
        atom_score_map: Dict[Tuple[str, int, str, str], float] = {}
        for a in result.atom_scores:
            key = (a.chain_id, a.residue_seq_id, a.insertion_code, a.atom_name)
            if score_field == "score":
                atom_score_map[key] = a.score
            elif score_field == "score_positive":
                atom_score_map[key] = a.score_positive
            else:
                atom_score_map[key] = a.score_negative

    model = structure[0]
    for chain in model:
        for residue in chain:
            seq_id = residue.seqid.num
            ins_code = str(residue.seqid.icode).strip()
            for atom in residue:
                if score_level == "residue":
                    key = (chain.name, seq_id, ins_code)
                    val = score_map.get(key, missing_value)
                else:
                    key = (chain.name, seq_id, ins_code, atom.name)
                    val = atom_score_map.get(key, missing_value)
                atom.b_iso = float(val * score_scale)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    structure.write_pdb(str(out))


def _write_csv(output_path: str, fieldnames: List[str], rows: List[dict]) -> None:
    path = Path(output_path)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
