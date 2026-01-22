# ------------------------------------------------------------------------------
# The atomic radii data below has been extracted from the supplementary materials
# of the following publication:
#
# Meyder, A. et al. Estimating Electron Density Support for Individual Atoms and Molecular
# Fragments in X-Ray Structures. J. Chem. Inf. Model. 2017, 57 (10), 2437–2447.
# https://doi.org/10.1021/acs.jcim.7b00391.
# ------------------------------------------------------------------------------

# {Element: {Resolution_Lower_Bound: Radius}}
ATOM_RADII_DATA = {
    "H": {0.5: 1.08, 1.0: 1.20, 1.5: 1.29, 2.0: 1.41, 2.5: 1.68, 3.0: 1.98},
    "C": {0.5: 1.02, 1.0: 1.14, 1.5: 1.26, 2.0: 1.38, 2.5: 1.65, 3.0: 1.98},
    "N": {0.5: 0.96, 1.0: 1.11, 1.5: 1.23, 2.0: 1.35, 2.5: 1.62, 3.0: 1.95},
    "O": {0.5: 0.93, 1.0: 1.08, 1.5: 1.20, 2.0: 1.32, 2.5: 1.62, 3.0: 1.92},
    "S": {0.5: 0.90, 1.0: 1.05, 1.5: 1.17, 2.0: 1.32, 2.5: 1.62, 3.0: 1.92},
    "P": {0.5: 0.90, 1.0: 1.05, 1.5: 1.17, 2.0: 1.32, 2.5: 1.62, 3.0: 1.95},
    "SE": {0.5: 0.84, 1.0: 0.99, 1.5: 1.11, 2.0: 1.29, 2.5: 1.59, 3.0: 1.89},  # Se
    "MG": {0.5: 0.87, 1.0: 1.02, 1.5: 1.14, 2.0: 1.32, 2.5: 1.62, 3.0: 1.92},  # Mg
    "CL": {0.5: 0.90, 1.0: 1.05, 1.5: 1.17, 2.0: 1.32, 2.5: 1.62, 3.0: 1.92},  # Cl
    "CA": {0.5: 0.87, 1.0: 1.02, 1.5: 1.17, 2.0: 1.32, 2.5: 1.62, 3.0: 1.92},  # Ca
}

def get_atom_radius(element: str, resolution: float) -> float:
    """
    Retrieves the atom radius for a given element and resolution.
    """
    element = element.upper().strip()

    if element not in ATOM_RADII_DATA:
        element = "C"

    profile = ATOM_RADII_DATA[element]
    available_res_levels = sorted(profile.keys())

    selected_level = available_res_levels[0]

    for level in available_res_levels:
        if resolution >= level:
            selected_level = level
        else:
            break

    return profile[selected_level]
