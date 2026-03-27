from __future__ import annotations

import time
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import eliot
import gemmi
import numpy as np

from common.logging import setup_eliot_logging
from quantify.utils import find_experiments
from analyse.water_sites.alignment import SuperpositionResult, compute_alignment
from analyse.water_sites.clustering import (
    WaterSite,
    collect_waters,
    cluster_water_observations,
)
from analyse.water_sites.config import WaterSiteConfig
from analyse.water_sites.io import (
    export_water_sites_per_structure,
    export_water_sites_summary,
)
from analyse.water_sites.visualisation import generate_water_analysis_figures
from analyse.water_sites.metrics import WaterSiteConsistency, compute_consistency
from analyse.water_sites.occupancy import SiteOccupancy, check_all_site_occupancy_for_structure
from analyse.water_sites.site_snr import PerStructureSiteSNR, extract_all_site_snr_for_structure


_DEFAULT_K_FACTOR = 1.0
_DEFAULT_MAP_CAP = 50


# ---------------------------------------------------------------------------
# Reference selection
# ---------------------------------------------------------------------------

def _infer_resolution(paths: dict) -> Optional[float]:
    """Try to infer resolution from MTZ files in the experiment results dir."""
    stem = paths["stem"]
    results_dir = paths.get("results_dir")
    if results_dir is None or not results_dir.exists():
        return None
    canonical = results_dir / f"{stem}_0" / f"{stem}_0.mtz"
    if canonical.exists():
        try:
            return gemmi.read_mtz_file(str(canonical)).resolution_high()
        except Exception:
            pass
    for mtz in sorted(results_dir.glob("*/*.mtz")):
        try:
            return gemmi.read_mtz_file(str(mtz)).resolution_high()
        except Exception:
            continue
    return None


def _select_reference(experiments: List[dict]) -> dict:
    """Return the experiment with the highest resolution (lowest d_min)."""
    best_paths = experiments[0]
    best_res = float("inf")
    for paths in experiments:
        res = _infer_resolution(paths)
        if res is not None and res < best_res:
            best_res = res
            best_paths = paths
    return best_paths


# ---------------------------------------------------------------------------
# Alignment
# ---------------------------------------------------------------------------

def _compute_transforms(
    experiments: List[dict],
    reference_paths: dict,
    config: WaterSiteConfig,
) -> Dict[str, SuperpositionResult]:
    """
    Compute Cα superposition transforms for all experiments onto the reference.

    The reference structure receives an identity transform. Any structure
    whose original_pdb is missing or that has fewer than config.min_ca_overlap
    common Cα atoms also receives an identity transform with a printed warning.
    """
    ref_stem = reference_paths["stem"]
    ref_pdb = reference_paths["original_pdb"]

    identity = SuperpositionResult(
        rotation=np.eye(3, dtype=np.float64),
        translation=np.zeros(3, dtype=np.float64),
        rmsd=0.0,
        n_ca_used=0,
        is_identity=True,
    )

    if not ref_pdb.exists():
        print(
            f"Warning: reference structure {ref_stem} not found at {ref_pdb}. "
            "All transforms set to identity."
        )
        return {paths["stem"]: identity for paths in experiments}

    try:
        ref_structure = gemmi.read_structure(str(ref_pdb))
    except Exception as exc:
        print(
            f"Warning: failed to load reference {ref_stem}: {exc}. "
            "All transforms set to identity."
        )
        return {paths["stem"]: identity for paths in experiments}

    transforms: Dict[str, SuperpositionResult] = {}
    for paths in experiments:
        stem = paths["stem"]
        if stem == ref_stem:
            transforms[stem] = SuperpositionResult(
                rotation=np.eye(3, dtype=np.float64),
                translation=np.zeros(3, dtype=np.float64),
                rmsd=0.0,
                n_ca_used=0,
                is_identity=True,
            )
            continue

        mob_pdb = paths["original_pdb"]
        if not mob_pdb.exists():
            print(
                f"Warning: {stem} — original_pdb not found, using identity transform."
            )
            transforms[stem] = identity
            continue

        try:
            mob_structure = gemmi.read_structure(str(mob_pdb))
            result = compute_alignment(
                ref_structure, mob_structure, min_overlap=config.min_ca_overlap
            )
            if result.is_identity and result.n_ca_used < config.min_ca_overlap:
                print(
                    f"Warning: {stem} — only {result.n_ca_used} common Cα atoms "
                    f"with reference (< {config.min_ca_overlap}). "
                    "Using identity transform."
                )
            transforms[stem] = result
        except Exception as exc:
            print(
                f"Warning: alignment failed for {stem}: {exc}. "
                "Using identity transform."
            )
            transforms[stem] = identity

    return transforms


# ---------------------------------------------------------------------------
# SNR map resolution
# ---------------------------------------------------------------------------

def _resolve_snr_map(
    paths: dict,
    k_factor: float,
    map_cap: Optional[int],
) -> Optional[Path]:
    """Locate the SNR CCP4 map; returns None if not found (non-fatal)."""
    stem = paths["stem"]
    qdir = paths["quantify_dir"]
    if not qdir.exists():
        return None
    if map_cap is not None:
        snr_path = qdir / f"k_{k_factor}_cap_{map_cap}" / f"{stem}_snr.ccp4"
        return snr_path if snr_path.exists() else None
    for candidate in reversed(sorted(qdir.glob(f"k_{k_factor}_cap_*"))):
        snr_path = candidate / f"{stem}_snr.ccp4"
        if snr_path.exists():
            return snr_path
    return None


# ---------------------------------------------------------------------------
# Per-structure worker (runs in a subprocess when num_processes > 1)
# ---------------------------------------------------------------------------

def _process_structure(
    args: Tuple[
        dict,
        List[WaterSite],
        SuperpositionResult,
        Optional[str],
        WaterSiteConfig,
    ],
) -> Tuple[List[PerStructureSiteSNR], List[SiteOccupancy]]:
    """
    Worker function: load map and structure once, process all sites.

    Designed to be called via multiprocessing.Pool.map. All arguments are
    plain Python objects (no unpicklable gemmi wrappers); files are loaded
    fresh inside the worker.

    Args:
        args: (paths, sites, transform, snr_map_path_str, config)

    Returns:
        (snr_results, occupancy_results) for this structure.
    """
    paths, sites, transform, snr_map_path, config = args
    snr_results = extract_all_site_snr_for_structure(
        sites, paths, transform, snr_map_path, config
    )
    occ_results = check_all_site_occupancy_for_structure(
        sites, paths, transform, config
    )
    return snr_results, occ_results


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_water_site_analysis(
    screening_dir: Union[str, Path],
    k_factor: float = _DEFAULT_K_FACTOR,
    map_cap: Optional[int] = _DEFAULT_MAP_CAP,
    config: Optional[WaterSiteConfig] = None,
    num_processes: int = 1,
) -> None:
    """
    Run water site consistency analysis for a completed PSEUDO screening.

    Requires that per-experiment SNR maps exist (produced by `pseudo quantify`).
    The analysis is a screening-level operation: it collects waters across all
    experiments, clusters them into conserved sites, and extracts SNR signal at
    each site from every structure — including those where the water was not
    modelled or was displaced by a ligand.

    Writes two CSV files to <screening_dir>/metadata/:
        water_sites_summary.csv         one row per water site
        water_sites_per_structure.csv   one row per (site, structure) pair

    Args:
        screening_dir: Root directory of the PSEUDO screening run.
        k_factor: K factor used during quantification to locate SNR maps.
        map_cap: Map cap used during quantification. Pass None to auto-detect
            the highest available cap.
        config: WaterSiteConfig instance. Defaults are used when None.
        num_processes: Number of parallel worker processes for per-structure
            SNR and occupancy extraction.
    """
    screening_dir = Path(screening_dir)

    if config is None:
        config = WaterSiteConfig(k_factor=k_factor, map_cap=map_cap)

    log_dir = screening_dir / "logs" / "eliot"
    setup_eliot_logging(log_dir, "water_sites")

    with eliot.start_action(
        action_type="water_sites:run",
        screening_dir=str(screening_dir),
        k_factor=k_factor,
        map_cap=map_cap,
        num_processes=num_processes,
    ):
        start_time = time.time()

        # ------------------------------------------------------------------ #
        # 1. Discover experiments
        # ------------------------------------------------------------------ #
        experiments = list(find_experiments(str(screening_dir)))
        if len(experiments) < 2:
            msg = (
                "Water site analysis requires at least 2 experiments "
                f"(found {len(experiments)})."
            )
            print(msg)
            eliot.log_message(message_type="water_sites:skipped", reason=msg)
            return

        n_total = len(experiments)
        print(f"Water site analysis: {n_total} experiments found.")
        eliot.log_message(
            message_type="water_sites:experiments_found", n_total=n_total
        )

        # ------------------------------------------------------------------ #
        # 2. Select reference structure (highest resolution)
        # ------------------------------------------------------------------ #
        reference_paths = _select_reference(experiments)
        print(f"Reference structure: {reference_paths['stem']}")
        eliot.log_message(
            message_type="water_sites:reference_selected",
            stem=reference_paths["stem"],
        )

        # ------------------------------------------------------------------ #
        # 3. Compute superposition transforms
        # ------------------------------------------------------------------ #
        print("Computing alignments...")
        with eliot.start_action(action_type="water_sites:alignment"):
            transforms = _compute_transforms(experiments, reference_paths, config)

        ref_stem = reference_paths["stem"]
        n_identity = sum(
            1 for stem, t in transforms.items()
            if t.is_identity and stem != ref_stem
        )
        print(
            f"Aligned {n_total - n_identity} structures; "
            f"{n_identity} use identity transform."
        )
        eliot.log_message(
            message_type="water_sites:alignment_complete",
            n_aligned=n_total - n_identity,
            n_identity=n_identity,
        )

        # ------------------------------------------------------------------ #
        # 4. Collect water observations in the reference frame
        # ------------------------------------------------------------------ #
        print("Collecting water observations...")
        with eliot.start_action(action_type="water_sites:collect_waters"):
            observations = collect_waters(experiments, transforms)

        print(f"Collected {len(observations)} water observations.")
        eliot.log_message(
            message_type="water_sites:observations_collected",
            n_observations=len(observations),
        )

        if not observations:
            print("No water observations found. Skipping analysis.")
            eliot.log_message(
                message_type="water_sites:skipped", reason="no observations"
            )
            return

        # ------------------------------------------------------------------ #
        # 5. Cluster into water sites
        # ------------------------------------------------------------------ #
        print("Clustering water sites...")
        with eliot.start_action(action_type="water_sites:clustering"):
            sites = cluster_water_observations(observations, config)

        print(f"Found {len(sites)} water sites.")
        eliot.log_message(
            message_type="water_sites:clustering_complete", n_sites=len(sites)
        )

        if not sites:
            print("No water sites found. Skipping analysis.")
            return

        # ------------------------------------------------------------------ #
        # 6. Resolve SNR map paths upfront (once per structure, not per site)
        # ------------------------------------------------------------------ #
        snr_map_paths: Dict[str, Optional[str]] = {}
        for paths in experiments:
            snr_path = _resolve_snr_map(paths, config.k_factor, config.map_cap)
            snr_map_paths[paths["stem"]] = str(snr_path) if snr_path else None

        n_maps_found = sum(1 for p in snr_map_paths.values() if p is not None)
        print(f"SNR maps found: {n_maps_found}/{n_total}.")
        eliot.log_message(
            message_type="water_sites:snr_maps_resolved",
            n_found=n_maps_found,
            n_total=n_total,
        )

        # ------------------------------------------------------------------ #
        # 7. Per-structure extraction: one worker per structure, all sites
        # ------------------------------------------------------------------ #
        print(
            f"Extracting SNR and occupancy across {n_total} structures "
            f"({len(sites)} sites, {num_processes} process(es))..."
        )
        worker_args = [
            (
                paths,
                sites,
                transforms[paths["stem"]],
                snr_map_paths[paths["stem"]],
                config,
            )
            for paths in experiments
        ]

        with eliot.start_action(
            action_type="water_sites:per_structure_extraction",
            n_structures=n_total,
            n_sites=len(sites),
            n_processes=num_processes,
        ):
            if num_processes > 1:
                with Pool(max(1, num_processes)) as pool:
                    structure_results = pool.map(_process_structure, worker_args)
            else:
                structure_results = [
                    _process_structure(args) for args in worker_args
                ]

        all_snr: List[PerStructureSiteSNR] = []
        all_occ: List[SiteOccupancy] = []
        for snr_list, occ_list in structure_results:
            all_snr.extend(snr_list)
            all_occ.extend(occ_list)

        eliot.log_message(
            message_type="water_sites:extraction_complete",
            n_snr_records=len(all_snr),
            n_occ_records=len(all_occ),
        )

        # ------------------------------------------------------------------ #
        # 8. Compute per-site consistency metrics
        # ------------------------------------------------------------------ #
        snr_by_site: Dict[int, List[PerStructureSiteSNR]] = defaultdict(list)
        for s in all_snr:
            snr_by_site[s.site_id].append(s)
        occ_by_site: Dict[int, List[SiteOccupancy]] = defaultdict(list)
        for o in all_occ:
            occ_by_site[o.site_id].append(o)

        consistencies: List[WaterSiteConsistency] = [
            compute_consistency(
                snr_by_site[site.site_id],
                occ_by_site[site.site_id],
                n_total,
            )
            for site in sites
        ]

        # ------------------------------------------------------------------ #
        # 9. Export CSV outputs and figures
        # ------------------------------------------------------------------ #
        output_dir = screening_dir / "water_analysis"
        output_dir.mkdir(parents=True, exist_ok=True)
        summary_path = output_dir / "water_sites_summary.csv"
        per_struct_path = output_dir / "water_sites_per_structure.csv"
        figures_dir = output_dir / "figures"

        with eliot.start_action(action_type="water_sites:export"):
            export_water_sites_summary(
                sites=sites,
                per_structure_snr=all_snr,
                per_structure_occ=all_occ,
                consistencies=consistencies,
                n_total_structures=n_total,
                output_path=summary_path,
            )
            export_water_sites_per_structure(
                per_structure_snr=all_snr,
                per_structure_occ=all_occ,
                output_path=per_struct_path,
            )
            generate_water_analysis_figures(
                sites=sites,
                consistencies=consistencies,
                per_structure_snr=all_snr,
                per_structure_occ=all_occ,
                figures_dir=figures_dir,
            )

        elapsed = time.time() - start_time
        eliot.log_message(
            message_type="water_sites:complete",
            n_sites=len(sites),
            n_structures=n_total,
            elapsed_seconds=round(elapsed, 2),
        )
        print(
            f"Water site analysis complete in {elapsed:.1f}s\n"
            f"  Sites: {len(sites)} | Structures: {n_total}\n"
            f"  {summary_path}\n"
            f"  {per_struct_path}\n"
            f"  {figures_dir}/"
        )
