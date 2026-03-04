import os

import gemmi
import reciprocalspaceship as rs
import pandas as pd
from pathlib import Path


def split_cif_textual_strict(input_path, output_dir=None):
    if not os.path.exists(input_path):
        print(f"Error: File '{input_path}' not found.")
        return

    base_name = os.path.splitext(os.path.basename(input_path))[0]
    output_dir = os.path.join(output_dir, base_name) if output_dir else base_name
    os.makedirs(output_dir, exist_ok=True)
    print(
        f"Splitting '{input_path}' into separate files in '{output_dir}' (base name: '{base_name}')"
    )

    # Store blocks here
    blocks = []
    current_block = []

    with open(input_path, "r") as f:
        lines = f.readlines()

    # Iterate through lines to group them into blocks
    for line in lines:
        # Check if line starts with 'data_' (case-insensitive usually, but strict here)
        if line.lstrip().startswith("data_"):
            # If we have gathered lines for a previous block, save them now
            if current_block:
                blocks.append(current_block)
                current_block = []  # Reset for next block

            # Start the new block with this 'data_' line
            current_block.append(line)
        else:
            # If we are inside a block (i.e. we've seen the first data_), add the line
            if current_block or len(blocks) > 0:
                # Note: This logic handles the very first block correctly.
                # If there is header text before the first 'data_', it gets ignored
                # until 'current_block' is initialized by the first 'data_' line.
                if current_block:
                    current_block.append(line)

    # Append the final block after loop finishes
    if current_block:
        blocks.append(current_block)

    print(f"Found {len(blocks)} blocks.")

    # Generate suffix generator (A, B, ... Z, AA, AB...)
    def get_suffix(n):
        letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        result = ""
        while n >= 0:
            result = letters[n % 26] + result
            n = n // 26 - 1
        return result

    # Save files
    for i, block_lines in enumerate(blocks):
        suffix = get_suffix(i)
        out_filename = f"{base_name}_{suffix}.cif"
        out_path = os.path.join(output_dir, out_filename)

        with open(out_path, "w") as out_f:
            out_f.writelines(block_lines)


def generate_modified_density(path_md, path_md_i, to_comp_with=None):
    if to_comp_with is not None:
        comparison_ccp4 = gemmi.read_ccp4_map(to_comp_with)
        grid = comparison_ccp4.grid
    dataset_ph = rs.read_cif(path_md)
    dataset_ph.canonicalize_phases()
    dataset_ph.drop(columns=["FWT"], inplace=True)

    spacegroup = dataset_ph.spacegroup
    cell = dataset_ph.cell

    dataset_i = rs.read_cif(path_md_i)

    if not dataset_i.merged:
        dataset_i = rs.algorithms.merge(dataset_i)

    scaled_dataset = rs.algorithms.scale_merged_intensities(
        dataset_i, "IMEAN", "SIGIMEAN"
    )

    joint_data = pd.merge(scaled_dataset, dataset_ph, how="inner", on=["H", "K", "L"])
    joint_data.dropna(inplace=True)

    joint_data = rs.DataSet(joint_data)

    joint_data.spacegroup = spacegroup
    joint_data.cell = cell

    output_path = Path(path_md).parent / "baseline_maps"
    output_path.mkdir(parents=True, exist_ok=True)
    output_path = output_path / "experimntal_density.mtz"

    joint_data.write_mtz(str(output_path))

    if to_comp_with is not None:
        new_mtz = gemmi.read_mtz_file(str(output_path))

        ccp4 = gemmi.Ccp4Map()
        ccp4.grid = new_mtz.transform_f_phi_to_map(
            "FW-F", "PHIC", exact_size=[grid.nu, grid.nv, grid.nw]
        )
        ccp4.update_ccp4_header()
        ccp4.write_ccp4_map(f"{str(output_path.with_suffix('.ccp4'))}")
        print("compare")

    else:
        os.system(
            f"gemmi sf2map {str(output_path)} {str(output_path.with_suffix('.ccp4'))} -f FW-F -p PHIC"
        )


def generate_calculated_density(path_md, path_md_i, to_comp_with=None):
    if to_comp_with is not None:
        comparison_ccp4 = gemmi.read_ccp4_map(to_comp_with)
        grid = comparison_ccp4.grid

    dataset_ph = rs.read_cif(path_md)
    dataset_ph.canonicalize_phases()
    dataset_ph.drop(columns=["FC", "FP", "SIGFP"], inplace=True)

    spacegroup = dataset_ph.spacegroup
    cell = dataset_ph.cell

    dataset_i = rs.read_cif(path_md_i)

    if not dataset_i.merged:
        dataset_i = rs.algorithms.merge(dataset_i)

    scaled_dataset = rs.algorithms.scale_merged_intensities(
        dataset_i, "IMEAN", "SIGIMEAN"
    )

    joint_data = pd.merge(scaled_dataset, dataset_ph, how="inner", on=["H", "K", "L"])
    joint_data.dropna(inplace=True)

    joint_data = rs.DataSet(joint_data)

    joint_data.spacegroup = spacegroup
    joint_data.cell = cell

    output_path = Path(path_md).parent / "baseline_maps"
    output_path.mkdir(parents=True, exist_ok=True)
    output_path = output_path / "calculated_density.mtz"

    joint_data.write_mtz(str(output_path))

    if to_comp_with is not None:
        new_mtz = gemmi.read_mtz_file(str(output_path))

        ccp4 = gemmi.Ccp4Map()
        ccp4.grid = new_mtz.transform_f_phi_to_map(
            "FW-F", "PHIC", exact_size=[grid.nu, grid.nv, grid.nw]
        )
        ccp4.update_ccp4_header()
        ccp4.write_ccp4_map(f"{str(output_path.with_suffix('.ccp4'))}")

    else:
        os.system(
            f"gemmi sf2map {str(output_path)} {str(output_path.with_suffix('.ccp4'))} -f FW-F -p PHIC"
        )
