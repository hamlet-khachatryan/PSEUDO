from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Dict, Generator, Any, Optional, List

import gemmi


def read_mtz(mtz_path: Path | str) -> gemmi.Ccp4Map:
    """
    Read a MTZ file and return a gemmi.Ccp4Map object.
    """
    reflections = gemmi.read_mtz_file(str(mtz_path))

    amplitudes = ["2FOFCWT", "2FOFC", "2FOFCWT_no_fill", "2FOFCWT_fill", "FWT", "FDM"]
    phases = [
        "PH2FOFCWT",
        "PH2FOFC",
        "PH2FOFCWT_no_fill",
        "PH2FOFCWT_fill",
        "PHWT",
        "PHFDM",
    ]

    label_index = 0
    column_labels = [column.label for column in reflections.columns]
    for label_i, label in enumerate(amplitudes):
        if label in column_labels:
            label_index = label_i
            break

    ref_map = gemmi.Ccp4Map()
    ref_map.grid = reflections.transform_f_phi_to_map(
        amplitudes[label_index], phases[label_index], sample_rate=3
    )
    ref_map.update_ccp4_header()
    ref_map.setup(float("nan"))

    return ref_map


def get_experiment_paths(root_dir: Path, stem: str) -> Dict[str, Path]:
    root_dir = Path(root_dir)
    return {
        "root": root_dir,
        "processed_pdb": root_dir / "processed" / f"{stem}_updated.pdb",
        "original_pdb": root_dir / "processed" / f"{stem}_original.pdb",
        "omission_json": root_dir / "metadata" / f"{stem}_omission_map.json",
        "metadata_dir": root_dir / "metadata",
        "results_dir": root_dir / "results",
        "quantify_dir": root_dir / "quantify_results",
    }


def validate_experiment(paths: Dict[str, Path]) -> bool:
    return paths["processed_pdb"].exists() and paths["omission_json"].exists()


def infer_stem(processed_dir: Path) -> Optional[str]:
    try:
        files = list(processed_dir.glob("*_updated.pdb"))
        if files:
            return files[0].name.replace("_updated.pdb", "")
        return None
    except Exception:
        return None


def find_experiments(input_path: str) -> Generator[Dict[str, Any], None, None]:
    root = Path(input_path).resolve()
    if not root.exists():
        raise FileNotFoundError(f"Path not found: {root}")

    if (root / "processed").exists():
        stem = infer_stem(root / "processed")
        if stem:
            paths = get_experiment_paths(root, stem)
            if validate_experiment(paths):
                paths["stem"] = stem
                yield paths
                return

    for subdir in root.iterdir():
        if subdir.is_dir() and (subdir / "processed").exists():
            stem = infer_stem(subdir / "processed")
            if stem:
                paths = get_experiment_paths(subdir, stem)
                if validate_experiment(paths):
                    paths["stem"] = stem
                    yield paths


def infer_omission_mode(json_data: Dict[str, List[int]]) -> str:
    """
    Infers mode based on whether atoms in the same residue have different omission maps.
    """
    # Track unique schedules per residue
    schedules_per_residue = defaultdict(set)

    for key, maps in json_data.items():
        parts = key.split("|")

        if len(parts) < 4:
            continue

        res_id = f"{parts[0]}|{parts[1]}"
        schedules_per_residue[res_id].add(tuple(sorted(maps)))

        if len(schedules_per_residue[res_id]) > 1:
            return "atoms"

    return "amino_acids"
