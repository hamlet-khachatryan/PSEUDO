from __future__ import annotations

import math
import random
from pathlib import Path
from typing import List, Literal, overload, Union, Optional, Sequence, Iterator, Any

import gemmi
import numpy as np

AminoID = tuple[str, int, str]  # (chain_name, res_seq_num, res_name)
AtomID = tuple[
    str, int, str, str, str
]  # (chain_name, res_seq_num, res_name, atom_name, altloc)
AnyID = Union[AtomID, AminoID]

_TARGET_MAP_COUNT = 50


@overload
def extract_ids(structure_path: Path | str, mode: Literal["atoms"]) -> List[AtomID]: ...


@overload
def extract_ids(
    structure_path: Path | str, mode: Literal["amino_acids"]
) -> List[AminoID]: ...


def extract_ids(
    structure_path: Path | str, mode: Literal["atoms", "amino_acids"]
) -> Union[List[AtomID], List[AminoID]]:
    """
    Extract unique atom or amino-acid IDs from a protein structure.

    Args:
        structure_path: Path to the structure file (e.g., .pdb, .cif).
        mode: 'atoms' to extract AtomIDs, 'amino_acids' to extract AminoIDs.

    Returns:
        A list of unique IDs (tuples) based on the selected mode.

    Raises:
        ValueError: If an invalid mode is somehow provided.
    """

    structure = gemmi.read_structure(str(structure_path))
    seen: set = set()
    out: List[AnyID] = []

    for model in structure:
        for chain in model:
            for residue in chain:
                if residue.het_flag == "A":
                    if mode == "amino_acids":
                        tid: AminoID = (chain.name, residue.seqid.num, residue.name)
                        if tid not in seen:
                            seen.add(tid)
                            out.append(tid)

                    elif mode == "atoms":
                        for atom in residue:
                            tid: AtomID = (
                                chain.name,
                                residue.seqid.num,
                                residue.name,
                                atom.name,
                                str(atom.altloc),
                            )
                            if tid not in seen:
                                seen.add(tid)
                                out.append(tid)

                    else:
                        raise ValueError(
                            f"Invalid mode: {mode}. Must be 'atoms' or 'amino_acids'."
                        )
    return out


def _order_preserving_dedupe(items: Sequence[object]) -> List[object]:
    """Return items with duplicates removed but preserve first-seen order."""
    return list(dict.fromkeys(items))


def _apply_always_omit_selectors(
    all_ids: Sequence["AnyID"], always_omit: Optional[str] = None
):
    """Return (selectors_in_structure, remaining_pool) preserving order of remaining_pool."""
    if not always_omit:
        return [], list(all_ids)

    always_omit_formated = [
        tuple([int(p) if i == 1 else p for i, p in enumerate(s.strip().split())])
        for s in always_omit.split(",")
    ]

    sel_set = set(always_omit_formated)
    selectors_in = [s for s in always_omit_formated if s in all_ids]
    remaining = [x for x in all_ids if x not in sel_set]
    return selectors_in, remaining


@overload
def sample_ids(
    structure_path: Path | str,
    mode: Literal["atoms"],
    fraction: float,
    iterations: int = 1,
    always_omit: Optional[str] = None,
    seed: Optional[int] = None,
) -> Iterator[List["AtomID"]]: ...


@overload
def sample_ids(
    structure_path: Path | str,
    mode: Literal["amino_acids"],
    fraction: float,
    iterations: int = 5,
    always_omit: Optional[str] = None,
    seed: Optional[int] = None,
) -> Iterator[List["AminoID"]]: ...


def sample_ids(
    structure_path: Path | str,
    mode: Literal["atoms", "amino_acids"],
    fraction: float,
    iterations: int = 5,
    always_omit: Optional[str] = None,
    seed: Optional[int] = None,
) -> Iterator[List[Union["AtomID", "AminoID"]]]:
    """
    Generator yielding lists of ID tuples (AtomID or AminoID) to omit.

    Behaviour:
      - If `always_omit` is provided, those IDs (if present in the structure)
        are always included in every yielded omitted selection.
      - The `fraction` is applied to the pool *after* removing always_omit.
      - `iterations` controls how many independent shuffled runs are produced.

    Args:
        structure_path: Path to structure file.
        mode: "atoms" or "amino_acids".
        fraction: fraction to omit (0.0 < fraction < 1.0).
        iterations: number of independent shuffles/runs (>=1).
        always_omit: optional striung of IDs that must always be omitted.
        seed: optional RNG seed for reproducible permutations.

    Yields:
        Lists of ID tuples (each list is one omitted-selection).
    """

    if mode == "atoms":
        all_ids: List["AnyID"] = extract_ids(structure_path, "atoms")
    else:
        all_ids: List["AnyID"] = extract_ids(structure_path, "amino_acids")

    total_count = len(all_ids)
    if total_count == 0:
        yield []
        return

    always_omit_in, remaining = _apply_always_omit_selectors(all_ids, always_omit)
    remaining_n = len(remaining)

    cover_with_omits = fraction <= 0.5
    fraction_to_cover = fraction if cover_with_omits else (1.0 - fraction)
    if remaining_n == 0:
        if fraction_to_cover <= 0:
            return
        n_batches = math.ceil(1.0 / fraction_to_cover)
        for _ in range(iterations * n_batches):
            yield list(always_omit_in)
        return

    if fraction_to_cover <= 0:
        return

    use_numpy_rng = seed is not None
    rng = np.random.default_rng(seed) if use_numpy_rng else None

    # disjoint chunks (<=0.5) or sliding omission (>0.5)
    if fraction <= 0.5:
        sample_size = max(1, round(remaining_n * fraction))

        for _ in range(iterations):
            permuted = list(remaining)
            if use_numpy_rng:
                permuted = list(rng.permutation(remaining_n).tolist())
                permuted = [remaining[i] for i in permuted]
            else:
                random.shuffle(permuted)

            for i in range(0, remaining_n, sample_size):
                selected = permuted[i : i + sample_size]
                # final omitted selection: always_omit_in + selected (cover omitted part)
                yield list(always_omit_in) + list(selected)
    else:
        omit_size = max(1, round(remaining_n * (1.0 - fraction)))
        for _ in range(iterations):
            permuted = list(remaining)
            if use_numpy_rng:
                permuted = list(rng.permutation(remaining_n).tolist())
                permuted = [remaining[i] for i in permuted]
            else:
                random.shuffle(permuted)

            for i in range(0, remaining_n, omit_size):
                omitted_chunk = permuted[i : i + omit_size]
                omitted_set = set(omitted_chunk)

                complement = [x for x in remaining if x not in omitted_set]
                yield list(always_omit_in) + complement


def _validate_sampler_inputs(
    structure_path: Path | str, omit_fraction: float, n_iterations: Optional[int]
):
    """Validate inputs for the stochastic omission sampler."""
    if not Path(structure_path).exists():
        raise IOError(f"Error: PDB file not found at {structure_path}")
    if not (0.0 < omit_fraction < 1.0):
        raise ValueError("fraction must be > 0.0 and < 1.0")
    if n_iterations is not None and n_iterations < 1:
        raise ValueError("iterations must be >= 1")


def _calculate_iterations(n_iterations: Optional[int], omit_fraction: float) -> int:
    """Compute the number of iterations, using a heuristic if not provided."""
    if n_iterations is not None:
        return int(n_iterations)

    fraction_to_cover = min(omit_fraction, 1.0 - omit_fraction)
    n_batches_per_iteration = math.ceil(1.0 / fraction_to_cover)
    return math.ceil(_TARGET_MAP_COUNT / n_batches_per_iteration)


def stochastic_omission_sampler(
    structure_path: Path | str,
    omit_type: Literal["amino_acids", "atoms"] = "amino_acids",
    omit_fraction: float = 0.1,
    n_iterations: Optional[int] = 5,
    always_omit: Optional[str] = None,
    seed: Optional[int] = None,
) -> list[Any]:
    """
    Wrapper that computes a reasonable default `n_iterations` (target ~50 maps)
    and collects results from `sample_ids` into a flattened list.

    Args:
        structure_path: path to PDB/mmCIF structure file.
        omit_type: 'amino_acids' or 'atoms'.
        omit_fraction: fraction to omit (0.0 < fraction < 1.0).
        n_iterations: if None, compute using heuristic to produce ~50 maps.
        always_omit: optional sequence of IDs that must always be omitted.
        seed: optional RNG seed for reproducibility.

    Returns:
        Flattened list of omitted selections (each a list of ID tuples).
    """
    _validate_sampler_inputs(structure_path, omit_fraction, n_iterations)

    final_iterations = _calculate_iterations(n_iterations, omit_fraction)
    if final_iterations == 0:
        return []

    sampler_generator = sample_ids(
        structure_path=structure_path,
        mode=omit_type,
        fraction=omit_fraction,
        iterations=final_iterations,
        always_omit=always_omit,
        seed=seed,
    )

    return list(sampler_generator)
