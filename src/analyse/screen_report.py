from __future__ import annotations

import csv
import json
import logging
import os
import traceback
import webbrowser
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import gemmi

from quantify.utils import find_experiments

_WATER_NAMES = {"HOH", "WAT", "DOD", "H2O", "SOL"}

_OPIA_GREEN  = 0.8
_OPIA_YELLOW = 0.6
_MUSE_GREEN  = 0.7
_MUSE_YELLOW = 0.4


def _find_maps(paths: Dict[str, Path], suffix: str) -> List[Path]:
    """
    Return all CCP4 maps matching *suffix* across every k/cap directory
    """

    qdir = paths.get("quantify_dir")
    stem = paths.get("stem", "")
    if not qdir or not qdir.exists():
        return []
    return sorted(qdir.glob(f"*/{stem}_{suffix}.ccp4"))


def _resolve_original_mtz(screening_dir: Path, paths: Dict[str, Path]) -> Optional[Path]:
    """
    Find original reflections MTZ file path
    """

    stem = paths.get("stem", "")

    manifest = screening_dir / "sbatch" / "preprocessing_manifest.txt"
    if manifest.exists():
        try:
            for line in manifest.read_text(encoding="utf-8").splitlines():
                parts = line.strip().split("|")
                if len(parts) >= 3 and parts[0] == stem:
                    p = Path(parts[2])
                    if p.exists():
                        return p
        except OSError:
            pass

    root = paths.get("root")
    if root:
        params_file = root / "params" / f"{stem}_0.params"
        if params_file.exists():
            try:
                for line in params_file.read_text(encoding="utf-8").splitlines():
                    line = line.strip()
                    if line.startswith("file_name") and ".mtz" in line.lower():
                        val = line.split("=", 1)[1].strip().strip('"').strip("'")
                        p = Path(val)
                        if p.exists():
                            return p
            except OSError:
                pass

    return None


def _infer_screening_path(screening_dir: Path) -> Optional[Path]:
    """
    Infer the original screening project directory
    """

    manifest = screening_dir / "sbatch" / "preprocessing_manifest.txt"
    if not manifest.exists():
        return None
    try:
        pdb_paths: List[Path] = []
        for line in manifest.read_text(encoding="utf-8").splitlines():
            parts = line.strip().split("|")
            if len(parts) >= 2 and parts[1]:
                pdb_paths.append(Path(parts[1]))
        if not pdb_paths:
            return None
        if len(pdb_paths) == 1:
            return pdb_paths[0].parent
        return Path(os.path.commonpath([str(p) for p in pdb_paths]))
    except (OSError, ValueError):
        return None

def _load_summary(paths: Dict[str, Path]) -> Optional[Dict[str, Any]]:
    p = paths["root"] / "analyse_results" / f"{paths['stem']}_summary.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logging.warning("Cannot read summary for %s: %s", paths["stem"], exc)
        return None


def _load_csv(paths: Dict[str, Path], suffix: str) -> List[Dict[str, str]]:
    p = paths["root"] / "analyse_results" / f"{paths['stem']}_{suffix}.csv"
    if not p.exists():
        return []
    try:
        with p.open(newline="", encoding="utf-8") as fh:
            return list(csv.DictReader(fh))
    except OSError:
        return []

def _categorise(res_name: str) -> str:
    rn = res_name.upper()
    if rn in _WATER_NAMES:
        return "water"
    _AA = {
        "ALA","ARG","ASN","ASP","CYS","GLN","GLU","GLY","HIS","ILE",
        "LEU","LYS","MET","PHE","PRO","SER","THR","TRP","TYR","VAL",
        "SEC","PYL","MSE",
    }
    return "protein" if rn in _AA else "other_solvent"


def analyse_lig_neighbourhood(
    structure_path: Optional[Path],
    radius: float = 10.0,
    lig_resname: str = "LIG",
) -> Dict[str, Any]:
    """
    Return a summary of all atoms within *radius* Å of the LIG residue
    """

    base: Dict[str, Any] = {
        "lig_found": False, "lig_chain": None, "lig_seqid": None,
        "lig_n_atoms": 0, "n_protein_atoms": 0, "n_water_atoms": 0,
        "n_other_atoms": 0, "neighbour_residues": [], "error": None,
    }

    if not structure_path or not structure_path.exists():
        base["error"] = "Structure file not found"
        return base

    try:
        structure = gemmi.read_structure(str(structure_path))
    except Exception as exc:
        base["error"] = f"Cannot read structure: {exc}"
        return base

    model = structure[0]

    lig_positions: List[gemmi.Position] = []
    n_lig_atoms = 0
    lig_chain_id: Optional[str] = None
    lig_seqid: Optional[int] = None

    for chain in model:
        for residue in chain:
            if residue.name.upper() == lig_resname.upper():
                lig_chain_id = chain.name
                lig_seqid = residue.seqid.num
                for atom in residue:
                    if atom.element not in (gemmi.Element("H"), gemmi.Element("D")):
                        lig_positions.append(atom.pos)
                        n_lig_atoms += 1

    if not lig_positions:
        base["error"] = f"No residue named '{lig_resname}' found"
        return base

    cx = sum(p.x for p in lig_positions) / len(lig_positions)
    cy = sum(p.y for p in lig_positions) / len(lig_positions)
    cz = sum(p.z for p in lig_positions) / len(lig_positions)

    base.update(lig_found=True, lig_chain=lig_chain_id,
                lig_seqid=lig_seqid, lig_n_atoms=n_lig_atoms,
                lig_centroid=[round(cx, 3), round(cy, 3), round(cz, 3)])

    ns = gemmi.NeighborSearch(model, structure.cell, radius)
    ns.populate(include_h=False)

    visited: Dict[Tuple[str, int, str], str] = {}
    for pos in lig_positions:
        for mark in ns.find_atoms(pos, "\0", radius=radius):
            cra = mark.to_cra(model)
            rn = cra.residue.name.upper()
            if rn == lig_resname.upper():
                continue
            key = (cra.chain.name, cra.residue.seqid.num, rn)
            if key not in visited:
                visited[key] = _categorise(rn)

    base["n_protein_atoms"] = sum(1 for v in visited.values() if v == "protein")
    base["n_water_atoms"]   = sum(1 for v in visited.values() if v == "water")
    base["n_other_atoms"]   = sum(1 for v in visited.values() if v == "other_solvent")
    base["neighbour_residues"] = sorted(
        [{"chain": ch, "resname": rn, "seqid": sid, "category": cat}
         for (ch, sid, rn), cat in visited.items()],
        key=lambda x: (x["chain"], x["seqid"]),
    )
    return base


def _extract_lig_muse(
    residue_rows: List[Dict[str, str]],
    atom_rows: List[Dict[str, str]],
    lig_resname: str = "LIG",
) -> Dict[str, Any]:
    empty = {k: "N/A" for k in [
        "musem_score","min_atom_score","median_atom_score","max_atom_score",
        "n_atoms","n_clashes","n_missing_density","n_unaccounted_density",
    ]}
    empty["atom_scores"] = []

    lig_rows = [r for r in residue_rows
                if r.get("residue_name", "").upper() == lig_resname.upper()]
    if not lig_rows:
        return empty

    row = lig_rows[0]
    atom_scores = [
        {
            "atom_name": a.get("atom_name", ""),
            "element": a.get("element", ""),
            "score": _safe_float(a.get("score")),
            "has_clash": a.get("has_clash", "False").strip().lower() == "true",
            "has_missing_density": a.get("has_missing_density", "False").strip().lower() == "true",
            "has_unaccounted_density": a.get("has_unaccounted_density", "False").strip().lower() == "true",
        }
        for a in atom_rows
        if a.get("residue_name", "").upper() == lig_resname.upper()
    ]
    return {
        "musem_score":           _safe_float(row.get("musem_score")),
        "min_atom_score":        _safe_float(row.get("min_atom_score")),
        "median_atom_score":     _safe_float(row.get("median_atom_score")),
        "max_atom_score":        _safe_float(row.get("max_atom_score")),
        "n_atoms":               _safe_int(row.get("n_atoms")),
        "n_clashes":             _safe_int(row.get("n_clashes")),
        "n_missing_density":     _safe_int(row.get("n_missing_density")),
        "n_unaccounted_density": _safe_int(row.get("n_unaccounted_density")),
        "atom_scores":           atom_scores,
    }


def _write_coot_script(
    stem: str,
    output_dir: Path,
    model_path: str,
    snr_maps: List[str],
    mean_maps: List[str],
    mtz_path: str,
    lig_centroid: Optional[List[float]],
    significance_snr_threshold: Any,
) -> str:
    """
    Write a Coot Python script that loads the model, original MTZ reflections,
    STOMP_μ map and STOMP_SNR map, then centres on the LIG centroid
    """

    if not model_path and not snr_maps and not mean_maps and not mtz_path:
        return ""

    snr_map  = snr_maps[0]  if snr_maps  else ""
    mean_map = mean_maps[0] if mean_maps else ""

    try:
        snr_level = float(significance_snr_threshold)
    except (TypeError, ValueError):
        snr_level = 1.0

    cx, cy, cz = (lig_centroid or [0.0, 0.0, 0.0])

    lines: List[str] = [
        "# -*- coding: utf-8 -*-",
        "import coot",
        "",
    ]

    if model_path:
        lines += [
            f"imol = coot.read_pdb({model_path!r})",
            "",
        ]

    if mtz_path:
        lines += [
            f"mtz_maps = coot.auto_read_make_and_draw_maps({mtz_path!r})",
            "if mtz_maps:",
            "    mtz_handle = mtz_maps[0]",
            "    coot.set_contour_level_in_sigma(mtz_handle, 1.0)",
            "    coot.set_map_colour(mtz_handle, 0.80, 0.25, 0.10)",
            "",
        ]

    if mean_map:
        lines += [
            f"mean_handle = coot.handle_read_ccp4_map({mean_map!r}, 0)",
            "coot.set_contour_level_absolute(mean_handle, 1.0)",
            "coot.set_map_colour(mean_handle, 0.14, 0.55, 0.13)",
            "",
        ]

    if snr_map:
        lines += [
            f"snr_handle = coot.handle_read_ccp4_map({snr_map!r}, 0)",
            f"coot.set_contour_level_absolute(snr_handle, {snr_level:.4f})",
            "coot.set_map_colour(snr_handle, 0.85, 0.65, 0.13)",
            "",
        ]

    if lig_centroid:
        lines += [
            f"coot.set_rotation_centre({cx}, {cy}, {cz})",
            "",
        ]

    script_path = output_dir / f"{stem}_coot_view.py"
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        script_path.write_text("\n".join(lines), encoding="utf-8")
        return str(script_path)
    except OSError as exc:
        logging.warning("Could not write Coot script for %s: %s", stem, exc)
        return ""


def collect_experiment(
    paths: Dict[str, Path],
    screening_dir: Path,
    lig_resname: str = "LIG",
    neighbourhood_radius: float = 10.0,
) -> Dict[str, Any]:
    """
    Gather all data for one experiment and return a flat result dict
    """

    stem = paths["stem"]
    root = paths["root"]

    summary      = _load_summary(paths)
    residue_rows = _load_csv(paths, "residues")
    atom_rows    = _load_csv(paths, "atoms")

    original_pdb = paths.get("original_pdb")
    original_mtz = _resolve_original_mtz(screening_dir, paths)

    crystal_dir = str(original_pdb.parent) if original_pdb and original_pdb.exists() else ""

    analyse_dir = root / "analyse_results"
    atom_csv    = analyse_dir / f"{stem}_atoms.csv"
    residue_csv = analyse_dir / f"{stem}_residues.csv"
    scored_pdb  = analyse_dir / f"{stem}_scored.pdb"

    snr_maps  = [str(p) for p in _find_maps(paths, "snr")]
    mean_maps = [str(p) for p in _find_maps(paths, "mean")]
    significance_snr_threshold = _safe_float(summary.get("significance_snr_threshold")) if summary else "N/A"

    nbhd = analyse_lig_neighbourhood(
        original_pdb if original_pdb and original_pdb.exists() else None,
        radius=neighbourhood_radius,
        lig_resname=lig_resname,
    )

    # Prefer scored PDB for Coot; fall back to original
    model_for_coot = str(scored_pdb) if scored_pdb.exists() else (
        str(original_pdb) if original_pdb and original_pdb.exists() else ""
    )
    coot_script = _write_coot_script(
        stem=stem,
        output_dir=screening_dir / "coot",
        model_path=model_for_coot,
        snr_maps=snr_maps,
        mean_maps=mean_maps,
        mtz_path=str(original_mtz) if original_mtz else "",
        lig_centroid=nbhd.get("lig_centroid"),
        significance_snr_threshold=significance_snr_threshold,
    )

    return {
        "stem":     stem,
        "root_dir": str(root),

        "opia":                        _safe_float(summary.get("opia")) if summary else "N/A",
        "n_atoms":                     _safe_int(summary.get("n_atoms")) if summary else "N/A",
        "n_residues":                  _safe_int(summary.get("n_residues")) if summary else "N/A",
        "mean_atom_score":             _safe_float(summary.get("mean_atom_score")) if summary else "N/A",
        "median_atom_score":           _safe_float(summary.get("median_atom_score")) if summary else "N/A",
        "n_clashes_total":             _safe_int(summary.get("n_clashes")) if summary else "N/A",
        "n_missing_density_total":     _safe_int(summary.get("n_missing_density")) if summary else "N/A",
        "n_unaccounted_density_total": _safe_int(summary.get("n_unaccounted_density")) if summary else "N/A",
        "significance_alpha":          _safe_float(summary.get("significance_alpha")) if summary else "N/A",
        "significance_snr_threshold":  significance_snr_threshold,

        "lig_resname":   lig_resname,
        "lig_muse":      _extract_lig_muse(residue_rows, atom_rows, lig_resname),
        "neighbourhood": nbhd,

        "original_pdb": str(original_pdb) if original_pdb and original_pdb.exists() else "",
        "original_mtz": str(original_mtz) if original_mtz else "",
        "snr_maps":     snr_maps,
        "mean_maps":    mean_maps,
        "p_value_maps": [str(p) for p in _find_maps(paths, "p_value")],
        "atom_csv":     str(atom_csv)    if atom_csv.exists()    else "",
        "residue_csv":  str(residue_csv) if residue_csv.exists() else "",
        "scored_pdb":   str(scored_pdb)  if scored_pdb.exists()  else "",
        "coot_script":  coot_script,
        "crystal_dir":  crystal_dir,
        "analysis_complete": summary is not None,
    }


def _badge(val: Any, green_thr: float, yellow_thr: float, fmt: str = ".3f") -> str:
    if not isinstance(val, float):
        return '<span class="badge bg-secondary">N/A</span>'
    css = "bg-success" if val >= green_thr else ("bg-warning text-dark" if val >= yellow_thr else "bg-danger")
    return f'<span class="badge {css}">{val:{fmt}}</span>'


def _file_links(paths: List[str], label: str) -> str:
    if not paths:
        return '<span class="text-muted">—</span>'
    return " ".join(
        f'<a href="file://{p}" title="{p}" '
        f'class="badge bg-primary text-decoration-none me-1">{label}</a>'
        for p in paths
    )


def _single_link(path: str, label: str = "View") -> str:
    if not path:
        return '<span class="text-muted">—</span>'
    return (f'<a href="file://{path}" title="{path}" '
            f'class="badge bg-primary text-decoration-none">{label}</a>')


def _folder_btn(path: str) -> str:
    if not path:
        return '<span class="text-muted">—</span>'
    return (
        f'<button class="btn btn-sm btn-outline-secondary" type="button" '
        f'onclick="copyPath(this, {json.dumps(path)})" title="{path}">'
        f'Copy path</button>'
    )


def _nbhd_pills(nbhd: Dict[str, Any]) -> str:
    if not nbhd.get("lig_found"):
        return f'<span class="text-muted">{nbhd.get("error", "LIG not found")}</span>'
    return (
        f'<span class="badge bg-info text-dark me-1">{nbhd["n_protein_atoms"]} protein</span>'
        f'<span class="badge bg-info text-dark me-1">{nbhd["n_water_atoms"]} water</span>'
        f'<span class="badge bg-info text-dark">{nbhd["n_other_atoms"]} other</span>'
    )


def _nbhd_table(nbhd: Dict[str, Any]) -> str:
    if not nbhd.get("lig_found") or not nbhd.get("neighbour_residues"):
        return ""
    rows = "".join(
        f'<tr class="{"table-light" if nb["category"]=="protein" else "table-info" if nb["category"]=="water" else "table-warning"}">'
        f'<td>{nb["chain"]}</td><td>{nb["resname"]}</td>'
        f'<td>{nb["seqid"]}</td><td>{nb["category"]}</td></tr>'
        for nb in nbhd["neighbour_residues"]
    )
    return (
        '<table class="table table-sm table-bordered" style="font-size:0.78rem;">'
        '<thead class="table-dark"><tr>'
        '<th>Chain</th><th>Residue</th><th>SeqID</th><th>Category</th>'
        '</tr></thead>'
        f'<tbody>{rows}</tbody></table>'
    )


def _fmt(v: Any, digits: int = 3) -> str:
    return f"{v:.{digits}f}" if isinstance(v, float) else str(v)


def _fmt_int(v: Any) -> str:
    return str(v) if isinstance(v, int) else "N/A"


def _safe_float(v: Any) -> Any:
    try:
        return round(float(v), 4)
    except (TypeError, ValueError):
        return "N/A"


def _safe_int(v: Any) -> Any:
    try:
        return int(v)
    except (TypeError, ValueError):
        return "N/A"

def _build_html(
    results: List[Dict[str, Any]],
    screening_dir: Path,
    inferred_project_dir: Optional[Path],
    timestamp: str,
    lig_resname: str,
    neighbourhood_radius: float,
) -> str:
    complete = [r for r in results if r["analysis_complete"]]
    opias    = [r["opia"] for r in complete if isinstance(r["opia"], float)]
    mean_opia = f"{sum(opias)/len(opias):.3f}" if opias else "N/A"

    scr_html = (
        f'<a href="file://{inferred_project_dir}" class="text-break">{inferred_project_dir}</a>'
        if inferred_project_dir else '<span class="text-muted">—</span>'
    )

    rows = ""
    for idx, r in enumerate(results):
        lig  = r["lig_muse"]
        nbhd = r["neighbourhood"]
        coot_script = r.get("coot_script", "")
        coot_cmd = f"coot --script {coot_script}" if coot_script else ""
        coot_btn = (
            f'<button class="btn btn-sm btn-outline-success" type="button" '
            f'title="Copy to clipboard: {coot_cmd}" '
            f'onclick="copyCoot(this, {json.dumps(coot_cmd)})">'
            f'Open in Coot</button>'
            if coot_script else
            '<span class="text-muted">—</span>'
        )
        rows += f"""
        <tr>
          <td><strong>{r["stem"]}</strong></td>
          <td>{_badge(r["opia"], _OPIA_GREEN, _OPIA_YELLOW)}</td>
          <td>{_fmt(r["mean_atom_score"])}</td>
          <td>{_badge(lig["musem_score"], _MUSE_GREEN, _MUSE_YELLOW)}</td>
          <td>{_single_link(r["original_pdb"], "PDB")}</td>
          <td>{_single_link(r["original_mtz"], "MTZ")}</td>
          <td>{_file_links(r["snr_maps"], "STOMP-SNR")}</td>
          <td>{_file_links(r["mean_maps"], "STOMP-\u03bc")}</td>
          <td>{_single_link(r["scored_pdb"], "Scored PDB")}</td>
          <td>
            <button class="btn btn-sm btn-outline-secondary" type="button"
              onclick="toggleNbhd('nbhd-{idx}', this)">
              Binding site
            </button>
            <div style="display:none;" class="mt-1" id="nbhd-{idx}">{_nbhd_table(nbhd)}</div>
          </td>
          <td>{_folder_btn(r.get("crystal_dir", ""))}</td>
          <td>{coot_btn}</td>
        </tr>"""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>PSEUDO Screen Results — {screening_dir.name}</title>
  <link rel="stylesheet"
        href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css"
        integrity="sha384-QWTKZyjpPEjISv5WaRU9OFeRpok6YctnYmDr5pNlyT2bRjXh0JMhjY6hW+ALEwIH"
        crossorigin="anonymous">
  <style>
    body {{ font-family: 'Segoe UI', Arial, sans-serif; background: #f8f9fa; }}
    th {{ white-space: nowrap; position: sticky; top: 0; background: #343a40; color: #fff; z-index: 2; }}
    .tbl-wrap {{ overflow-x: auto; max-height: 72vh; }}
    a.badge:hover {{ opacity: 0.85; }}
    .card-metric {{ min-width: 130px; }}
  </style>
</head>
<body>
<div class="container-fluid py-3">

  <div class="d-flex align-items-center mb-2">
    <h2 class="me-3 mb-0">PSEUDO Screen Results</h2>
    <span class="badge bg-dark fs-6">{screening_dir.name}</span>
  </div>
  <p class="text-muted small">Generated: {timestamp}</p>

  <div class="row g-3 mb-4">
    <div class="col-auto"><div class="card card-metric text-center shadow-sm">
      <div class="card-body py-2"><div class="fs-4 fw-bold">{len(results)}</div>
      <div class="small text-muted">Experiments</div></div></div></div>

    <div class="col-auto"><div class="card card-metric text-center shadow-sm">
      <div class="card-body py-2"><div class="fs-4 fw-bold text-success">{len(complete)}</div>
      <div class="small text-muted">Analysis complete</div></div></div></div>

    <div class="col-auto"><div class="card card-metric text-center shadow-sm">
      <div class="card-body py-2"><div class="fs-4 fw-bold">{mean_opia}</div>
      <div class="small text-muted">Mean OPIA</div></div></div></div>

    <div class="col-auto"><div class="card shadow-sm">
      <div class="card-body py-2">
        <div class="small text-muted mb-1">Screening input</div>
        <div style="max-width:380px;word-break:break-all;">{scr_html}</div>
      </div></div></div>

    <div class="col-auto"><div class="card shadow-sm">
      <div class="card-body py-2">
        <div class="small text-muted mb-1">Screening directory</div>
        <div class="small font-monospace">{screening_dir}</div>
      </div></div></div>
  </div>

  <div class="mb-2 small">
    <strong>OPIA:</strong>
    <span class="badge bg-success">≥{_OPIA_GREEN}</span>
    <span class="badge bg-warning text-dark">≥{_OPIA_YELLOW}</span>
    <span class="badge bg-danger">&lt;{_OPIA_YELLOW}</span>
    &nbsp;
    <strong>{lig_resname} MUSE:</strong>
    <span class="badge bg-success">≥{_MUSE_GREEN}</span>
    <span class="badge bg-warning text-dark">≥{_MUSE_YELLOW}</span>
    <span class="badge bg-danger">&lt;{_MUSE_YELLOW}</span>
  </div>

  <div class="mb-2">
    <input id="searchBox" class="form-control form-control-sm d-inline-block"
           style="min-width:220px;max-width:320px;"
           placeholder="Filter by crystal name…"
           oninput="filterTable(this.value)">
  </div>

  <div class="tbl-wrap">
    <table class="table table-sm table-bordered table-hover">
      <thead>
        <tr>
          <th>Crystal</th>
          <th title="Overall Per-instance Agreement">OPIA</th>
          <th>Mean MUSE</th>
          <th title="{lig_resname} MUSEm score">{lig_resname} MUSE</th>
          <th>Original PDB</th>
          <th>Original MTZ</th>
          <th>STOMP<sub>SNR</sub> map(s)</th>
          <th>STOMP<sub>&#956;</sub> map(s)</th>
          <th>Scored PDB</th>
          <th>Binding site</th>
          <th>Crystal folder</th>
          <th title="Copy command to open in Coot with maps loaded">Coot</th>
        </tr>
      </thead>
      <tbody id="tableBody">{rows}</tbody>
    </table>
  </div>

  <p class="text-muted small mt-2">
    {len(results)} experiment(s) · sorted by OPIA descending ·
    click a badge to open the file locally.
  </p>
</div>

<script>
function filterTable(q) {{
  q = q.toLowerCase();
  document.querySelectorAll("#tableBody tr").forEach(function(row) {{
    row.style.display = row.cells[0].textContent.toLowerCase().includes(q) ? "" : "none";
  }});
}}

function _copyText(text, onSuccess, onFail) {{
  if (navigator.clipboard && navigator.clipboard.writeText) {{
    navigator.clipboard.writeText(text).then(onSuccess, function() {{
      _fallbackCopy(text, onSuccess, onFail);
    }});
  }} else {{
    _fallbackCopy(text, onSuccess, onFail);
  }}
}}

function _fallbackCopy(text, onSuccess, onFail) {{
  var ta = document.createElement("textarea");
  ta.value = text;
  ta.style.position = "fixed";
  ta.style.opacity = "0";
  document.body.appendChild(ta);
  ta.focus();
  ta.select();
  try {{
    document.execCommand("copy");
    onSuccess();
  }} catch(e) {{
    onFail();
  }}
  document.body.removeChild(ta);
}}

function _flashBtn(btn, successClass, failClass, origText, successText) {{
  btn.textContent = successText;
  btn.classList.replace(failClass, successClass);
  setTimeout(function() {{
    btn.textContent = origText;
    btn.classList.replace(successClass, failClass);
  }}, 2000);
}}

function copyCoot(btn, cmd) {{
  if (!cmd) return;
  var orig = btn.textContent;
  _copyText(
    cmd,
    function() {{ _flashBtn(btn, "btn-success", "btn-outline-success", orig, "Copied!"); }},
    function() {{ prompt("Copy this command and run in your terminal:", cmd); }}
  );
}}

function copyPath(btn, path) {{
  if (!path) return;
  var orig = btn.textContent;
  _copyText(
    path,
    function() {{ _flashBtn(btn, "btn-secondary", "btn-outline-secondary", orig, "Copied!"); }},
    function() {{ prompt("Crystal folder path:", path); }}
  );
}}

function toggleNbhd(id, btn) {{
  var el = document.getElementById(id);
  if (!el) return;
  var hidden = el.style.display === "none" || el.style.display === "";
  el.style.display = hidden ? "block" : "none";
  btn.textContent = hidden ? "Hide site" : "Binding site";
}}
</script>
</body>
</html>"""


def write_experiment_json(result: Dict[str, Any], metadata_dir: Path) -> Path:
    metadata_dir.mkdir(parents=True, exist_ok=True)
    out = metadata_dir / f"{result['stem']}_screen_result.json"
    out.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    return out


def write_summary_json(
    results: List[Dict[str, Any]],
    screening_dir: Path,
    inferred_project_dir: Optional[Path],
    metadata_dir: Path,
    timestamp: str,
) -> Path:
    metadata_dir.mkdir(parents=True, exist_ok=True)
    complete = [r for r in results if r["analysis_complete"]]
    opias    = [r["opia"] for r in complete if isinstance(r["opia"], float)]
    summary = {
        "generated_at":        timestamp,
        "screening_dir":       str(screening_dir),
        "inferred_project_dir": str(inferred_project_dir) if inferred_project_dir else "",
        "n_experiments":       len(results),
        "n_complete":          len(complete),
        "mean_opia":           round(sum(opias) / len(opias), 4) if opias else None,
        "experiments":         [r["stem"] for r in results],
    }
    out = metadata_dir / f"screen_summary_{timestamp}.json"
    out.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    return out

def generate_screen_report(
    screening_dir: str | Path,
    lig_resname: str = "LIG",
    neighbourhood_radius: float = 10.0,
    open_browser: bool = False,
) -> None:
    """
    Collect results for every completed experiment in *screening_dir* and
    write metadata
    """

    screening_dir = Path(screening_dir).resolve()
    metadata_dir  = screening_dir / "metadata"
    timestamp     = datetime.now().strftime("%Y%m%d_%H%M%S")

    inferred_project_dir = _infer_screening_path(screening_dir)
    if inferred_project_dir:
        logging.info("Inferred screening project directory: %s", inferred_project_dir)

    experiments = list(find_experiments(str(screening_dir)))
    if not experiments:
        logging.warning("Screen report: no valid experiments found in %s", screening_dir)
        return

    results: List[Dict[str, Any]] = []
    for paths in experiments:
        stem = paths.get("stem", "?")
        try:
            result = collect_experiment(
                paths,
                screening_dir=screening_dir,
                lig_resname=lig_resname,
                neighbourhood_radius=neighbourhood_radius,
            )
        except Exception as exc:
            logging.error("Screen report: failed on %s — %s", stem, exc)
            logging.debug(traceback.format_exc())
            result = _error_result(stem, paths, lig_resname, exc)
        results.append(result)
        write_experiment_json(result, metadata_dir)

    results.sort(
        key=lambda r: r["opia"] if isinstance(r["opia"], float) else -1.0,
        reverse=True,
    )

    summary_path = write_summary_json(
        results, screening_dir, inferred_project_dir, metadata_dir, timestamp
    )
    logging.info("Screen summary JSON: %s", summary_path)

    html_content = _build_html(
        results, screening_dir, inferred_project_dir, timestamp,
        lig_resname, neighbourhood_radius,
    )
    html_path = screening_dir / "index.html"
    html_path.write_text(html_content, encoding="utf-8")

    n_done = sum(1 for r in results if r["analysis_complete"])
    print(
        f"Screen report saved to {html_path}\n"
        f"  {n_done}/{len(results)} experiment(s) with analysis · "
    )

    if open_browser:
        webbrowser.open(f"file://{html_path}")


def _error_result(
    stem: str, paths: Dict[str, Path], lig_resname: str, exc: Exception
) -> Dict[str, Any]:
    na_muse: Dict[str, Any] = {k: "N/A" for k in [
        "musem_score","min_atom_score","median_atom_score","max_atom_score",
        "n_atoms","n_clashes","n_missing_density","n_unaccounted_density",
    ]}
    na_muse["atom_scores"] = []
    return {
        "stem": stem, "root_dir": str(paths.get("root", "")),
        "analysis_complete": False, "error": str(exc),
        "opia": "N/A", "n_atoms": "N/A", "n_residues": "N/A",
        "mean_atom_score": "N/A", "median_atom_score": "N/A",
        "n_clashes_total": "N/A", "n_missing_density_total": "N/A",
        "n_unaccounted_density_total": "N/A",
        "significance_alpha": "N/A", "significance_snr_threshold": "N/A",
        "lig_resname": lig_resname, "lig_muse": na_muse,
        "neighbourhood": {"lig_found": False, "error": str(exc)},
        "original_pdb": "", "original_mtz": "",
        "snr_maps": [], "mean_maps": [], "p_value_maps": [],
        "atom_csv": "", "residue_csv": "", "scored_pdb": "",
        "coot_script": "", "crystal_dir": "",
    }
