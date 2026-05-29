"""
END (Electron Number Density) Maps: Python API

Complete implementation for calculating END maps from:
- Structure factors: MTZ, CIF, or CCP4 maps
- Atomic models: PDB or CIF

Usage:
    calculator = ENDMapAPI("data.mtz", "model.pdb")
    end_map = calculator.compute()
    calculator.save_ccp4("end_map_output.ccp4")
"""

import numpy as np
from scipy import fft
import gemmi
from pathlib import Path
from typing import Optional, Tuple, Union, Dict
import warnings


class ENDMapAPI:
    """
    High-level API for END map calculation from crystallographic data.

    Supports multiple input formats:
    - Structure factors: MTZ, CIF (mmCIF), CCP4 map
    - Models: PDB, mmCIF
    """

    def __init__(self,
                 data_file: str,
                 model_file: str,
                 grid_size: Optional[Tuple[int, int, int]] = None,
                 data_label: Optional[str] = None):
        """
        Initialize END map calculator.

        Parameters:
        -----------
        data_file : str
            Structure factors file: .mtz, .cif (structure factors), or .ccp4/.map
        model_file : str
            Atomic model: .pdb or .cif
        grid_size : tuple, optional
            FFT grid dimensions. If None, auto-calculated (~0.5 Å sampling)
        data_label : str, optional
            Column label in MTZ (e.g., "FP,SIGFP"). Auto-detected if None
        """
        self.data_file = Path(data_file)
        self.model_file = Path(model_file)

        if not self.data_file.exists():
            raise FileNotFoundError(f"Data file not found: {data_file}")
        if not self.model_file.exists():
            raise FileNotFoundError(f"Model file not found: {model_file}")

        # Load model
        self.structure = self._load_model(str(self.model_file))
        self.cell = self.structure.cell
        self.space_group = self.structure.spacegroup_hm

        # Load structure factors
        self.hkl, self.fobs, self.sigfobs = self._load_data(
            str(self.data_file), data_label
        )

        # Auto-set grid size
        if grid_size is None:
            self.grid_size = tuple(
                max(16, int(np.ceil(self.cell.parameters[:3][i] / 0.5)))
                for i in range(3)
            )
        else:
            self.grid_size = grid_size

        # Computed properties
        self.fcalc = None
        self.scale_factor = None
        self.f000 = None
        self.rho_mean = None
        self.end_map = None

        print(f"\n=== END Map Calculator Initialized ===")
        print(f"Model: {self.model_file.name}")
        print(f"Data: {self.data_file.name}")
        print(f"Cell: a={self.cell.a:.2f}, b={self.cell.b:.2f}, c={self.cell.c:.2f} Å")
        print(f"Space group: {self.space_group}")
        print(f"Grid size: {self.grid_size}")
        print(f"Reflections: {len(self.hkl)}")

    def _load_model(self, model_file: str) -> gemmi.Structure:
        """Load atomic model from PDB or CIF"""
        ext = Path(model_file).suffix.lower()

        if ext == '.pdb':
            structure = gemmi.read_structure(model_file)
        elif ext in ['.cif', '.mmcif']:
            doc = gemmi.cif.read(model_file)
            structure = gemmi.make_structure_from_block(doc[0])
        else:
            raise ValueError(f"Unsupported model format: {ext}")

        return structure

    def _load_data(self, data_file: str, data_label: Optional[str] = None
                   ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Load structure factors from MTZ, CIF, or CCP4 map"""
        ext = Path(data_file).suffix.lower()

        if ext == '.mtz':
            return self._load_mtz(data_file, data_label)
        elif ext in ['.cif', '.mmcif']:
            return self._load_cif_sf(data_file)
        elif ext in ['.ccp4', '.map']:
            return self._load_ccp4_as_sf(data_file)
        else:
            raise ValueError(f"Unsupported data format: {ext}")

    def _load_mtz(self, mtz_file: str, data_label: Optional[str] = None
                  ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Load structure factors from MTZ file"""
        mtz = gemmi.read_mtz(mtz_file)

        # Auto-detect or use provided label
        if data_label is None:
            # Look for standard FP, SIGFP columns
            fp_cols = [c for c in mtz.columns if c.label.startswith('F')]
            if not fp_cols:
                raise ValueError("No structure factor columns found in MTZ")
            data_label = fp_cols[0].label

        # Extract data
        hkl = np.array([mtz.get_miller_array(mtz.column_with_label(data_label))],
                       dtype=np.int32).T[:, :3]
        fobs = mtz.get_column(data_label).array.astype(np.float64)

        # Try to find sigma column
        sig_label = data_label.replace('F', 'SIG')
        if sig_label in [c.label for c in mtz.columns]:
            sigfobs = mtz.get_column(sig_label).array.astype(np.float64)
        else:
            sigfobs = np.sqrt(np.abs(fobs))  # Fallback

        return hkl, fobs, sigfobs

    def _load_cif_sf(self, cif_file: str
                     ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Load structure factors from mmCIF reflection data"""
        doc = gemmi.cif.read(cif_file)

        # Find reflection data block
        for block in doc:
            if '_refln.index_h' in block or 'index_h' in str(block):
                h = np.array(block.find_values('_refln.index_h'), dtype=np.int32)
                k = np.array(block.find_values('_refln.index_k'), dtype=np.int32)
                l = np.array(block.find_values('_refln.index_l'), dtype=np.int32)
                hkl = np.column_stack([h, k, l])

                # Find F_squared_meas or F_meas
                if '_refln.F_squared_meas' in block:
                    f2 = np.array(block.find_values('_refln.F_squared_meas'),
                                  dtype=np.float64)
                    fobs = np.sqrt(np.abs(f2))
                    sig_label = '_refln.F_squared_sigma'
                else:
                    fobs = np.array(block.find_values('_refln.F_meas'),
                                    dtype=np.float64)
                    sig_label = '_refln.F_meas_sigma'

                if sig_label in block:
                    sigfobs = np.array(block.find_values(sig_label),
                                       dtype=np.float64)
                else:
                    sigfobs = np.sqrt(np.abs(fobs))

                return hkl, fobs, sigfobs

        raise ValueError("No reflection data found in CIF file")

    def _load_ccp4_as_sf(self, ccp4_file: str
                        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Load CCP4 map and convert back to structure factors via FFT.
        This is lossy but provides a fallback option.
        """
        ccp4_map = gemmi.Ccp4Map()
        ccp4_map.read(ccp4_file)

        rho = ccp4_map.grid.array.astype(np.float64)

        # FFT to get structure factors
        grid_fft = fft.fftn(rho)

        # Extract non-redundant set
        n_max = min(rho.shape)
        hkl_list = []
        sf_list = []

        for h in range(0, n_max // 2):
            for k in range(0, n_max // 2):
                for l in range(0, n_max // 2):
                    sf = np.abs(grid_fft[h, k, l])
                    if sf > 0:
                        hkl_list.append([h, k, l])
                        sf_list.append(sf)

        hkl = np.array(hkl_list, dtype=np.int32)
        fobs = np.array(sf_list, dtype=np.float64)
        sigfobs = np.sqrt(np.abs(fobs))  # Estimate sigma

        warnings.warn("CCP4 map loaded - structure factors reconstructed via FFT "
                     "(approximate)")

        return hkl, fobs, sigfobs

    def calculate_solvent_fraction(self, n_molecules: int = 1
                                   ) -> Tuple[float, float]:
        """
        Calculate solvent fraction from Matthews coefficient.

        Parameters:
        -----------
        n_molecules : int
            Number of molecules in asymmetric unit

        Returns:
        --------
        K_sol : float
            Solvent fraction (0-1)
        protein_mass : float
            Estimated protein mass (Da)
        """
        atomic_masses = {
            'H': 1.008, 'C': 12.01, 'N': 14.01, 'O': 16.00,
            'S': 32.06, 'P': 30.97, 'CL': 35.45, 'FE': 55.85,
            'ZN': 65.39, 'CA': 40.08, 'MG': 24.31, 'NA': 22.99,
            'K': 39.10, 'BR': 79.90
        }

        protein_mass = 0.0
        for model in self.structure:
            for chain in model:
                for residue in chain:
                    for atom in residue:
                        element = atom.element.name.upper()
                        protein_mass += atomic_masses.get(element, 12.0)

        vm = self.cell.volume / (n_molecules * protein_mass)

        if vm > 1.23:
            k_sol = 1.0 - (1.23 / vm)
        else:
            k_sol = max(0.3, 1.0 - (1.23 / max(vm, 1.5)))

        k_sol = np.clip(k_sol, 0.2, 0.8)

        return k_sol, protein_mass

    def calculate_fcalc(self, resolution: Optional[float] = None) -> np.ndarray:
        """
        Calculate structure factors from atomic model.

        Parameters:
        -----------
        resolution : float, optional
            Resolution limit for B-factor damping

        Returns:
        --------
        fcalc : array
            Calculated structure factors at hkl positions
        """
        fcalc = np.zeros(len(self.hkl), dtype=np.complex128)

        for model in self.structure:
            for chain in model:
                for residue in chain:
                    for atom in residue:
                        x, y, z = atom.pos.x, atom.pos.y, atom.pos.z
                        frac = self.cell.fractionalize(gemmi.Position(x, y, z))

                        element = atom.element.name.upper()
                        f0 = self._get_scattering_factor(element, resolution)

                        phase = 2 * np.pi * (
                            self.hkl[:, 0] * frac.x +
                            self.hkl[:, 1] * frac.y +
                            self.hkl[:, 2] * frac.z
                        )
                        fcalc += f0 * np.exp(1j * phase)

        self.fcalc = np.abs(fcalc)
        return self.fcalc

    def _get_scattering_factor(self, element: str, resolution: Optional[float] = None
                               ) -> float:
        """Get approximate atomic scattering factor"""
        f0_values = {
            'H': 0.47, 'C': 5.70, 'N': 6.40, 'O': 7.05,
            'S': 14.5, 'P': 11.4, 'CL': 16.5, 'FE': 25.0,
            'ZN': 28.0, 'CA': 19.0, 'MG': 12.0, 'NA': 10.0,
            'K': 17.0, 'BR': 32.0
        }
        return f0_values.get(element, 6.0)

    def calculate_scale_factor(self) -> float:
        """Calculate least-squares scale factor k"""
        if self.fcalc is None:
            self.calculate_fcalc()

        valid = (self.fcalc > 0) & (self.fobs > 0)

        if np.sum(valid) < 10:
            warnings.warn("Few valid reflections for scaling")

        fobs_valid = self.fobs[valid]
        fcalc_valid = self.fcalc[valid]

        self.scale_factor = np.sum(fobs_valid * fcalc_valid) / np.sum(fcalc_valid ** 2)
        return self.scale_factor

    def calculate_f000(self, n_molecules: int = 1,
                      explicit_solvent_fraction: Optional[float] = None) -> float:
        """Calculate total electrons in unit cell"""
        atomic_numbers = {
            'H': 1, 'C': 6, 'N': 7, 'O': 8,
            'S': 16, 'P': 15, 'CL': 17, 'FE': 26,
            'ZN': 30, 'CA': 20, 'MG': 12, 'NA': 11,
            'K': 19, 'BR': 35
        }

        electron_count = 0
        for model in self.structure:
            for chain in model:
                for residue in chain:
                    for atom in residue:
                        element = atom.element.name.upper()
                        electron_count += atomic_numbers.get(element, 6)

        if explicit_solvent_fraction is not None:
            k_sol = explicit_solvent_fraction
        else:
            k_sol, _ = self.calculate_solvent_fraction(n_molecules)

        solvent_density = 0.334  # e-/Å³
        bulk_solvent_volume = self.cell.volume * k_sol
        bulk_solvent_electrons = bulk_solvent_volume * solvent_density

        self.f000 = electron_count + bulk_solvent_electrons
        return self.f000

    def calculate_mean_density(self) -> float:
        """Calculate mean electron density"""
        if self.f000 is None:
            self.calculate_f000()

        self.rho_mean = self.f000 / self.cell.volume
        return self.rho_mean

    def compute(self, n_molecules: int = 1,
               solvent_fraction: Optional[float] = None) -> np.ndarray:
        """
        Compute END map.

        Parameters:
        -----------
        n_molecules : int
            Number of molecules in asymmetric unit
        solvent_fraction : float, optional
            Explicit solvent fraction (if None, calculated)

        Returns:
        --------
        end_map : array
            3D electron density map (e-/Ų)
        """
        print("\n=== Computing END Map ===")

        # Calculate components
        self.calculate_fcalc()
        print("✓ Calculated Fcalc")

        self.calculate_scale_factor()
        print(f"✓ Scale factor k = {self.scale_factor:.6f}")

        self.calculate_f000(n_molecules, solvent_fraction)
        print(f"✓ F000 = {self.f000:.1f} electrons")

        self.calculate_mean_density()
        print(f"✓ Mean density = {self.rho_mean:.4f} e-/Ų")

        # Create reciprocal space grid
        grid = np.zeros(self.grid_size, dtype=np.complex128)

        fobs_scaled = self.scale_factor * self.fobs

        for idx, (h, k, l) in enumerate(self.hkl):
            if (0 <= h < self.grid_size[0] and
                0 <= k < self.grid_size[1] and
                0 <= l < self.grid_size[2]):

                phase = np.angle(self.fcalc[idx]) if self.fcalc[idx] > 0 else 0
                grid[h, k, l] = fobs_scaled[idx] * np.exp(1j * phase)

                # Friedel pair
                h_neg = (-h) % self.grid_size[0]
                k_neg = (-k) % self.grid_size[1]
                l_neg = (-l) % self.grid_size[2]
                grid[h_neg, k_neg, l_neg] = np.conj(grid[h, k, l])

        # FFT
        rho = np.real(fft.ifftn(grid)) * np.prod(self.grid_size)
        rho = rho / self.cell.volume
        rho = rho + self.rho_mean

        self.end_map = rho
        print("✓ END map computed")

        return self.end_map

    def save_ccp4(self, output_file: str) -> None:
        """Save END map in CCP4 format"""
        if self.end_map is None:
            raise ValueError("Must compute END map first")

        ccp4_map = gemmi.Ccp4Map()
        ccp4_map.grid = gemmi.FloatGrid(
            self.end_map.astype(np.float32),
            self.cell,
            gemmi.SpaceGroup(self.space_group)
        )
        ccp4_map.header_i32[28] = 20140
        ccp4_map.write(output_file)
        print(f"\n✓ Saved END map to {output_file}")

    def create_masks(self) -> Dict[str, np.ndarray]:
        """
        Create masks for protein, water, and bulk solvent regions.
        Maps voxels to regions based on proximity to atomic coordinates.

        Returns:
        --------
        masks : dict
            'protein': voxels near protein atoms
            'water': voxels near water molecules
            'bulk_solvent': remaining voxels
        """
        nx, ny, nz = self.grid_size

        # Initialize masks
        protein_mask = np.zeros((nx, ny, nz), dtype=bool)
        water_mask = np.zeros((nx, ny, nz), dtype=bool)

        protein_radius = 2.5  # Ångströms
        water_radius = 2.0

        # Create coordinate grid in Ångströms
        x_grid = np.arange(nx) * (self.cell.a / nx)
        y_grid = np.arange(ny) * (self.cell.b / ny)
        z_grid = np.arange(nz) * (self.cell.c / nz)
        X, Y, Z = np.meshgrid(x_grid, y_grid, z_grid, indexing='ij')

        # Process atoms
        water_residues = set(['HOH', 'WAT', 'H2O', 'SOL'])

        for model in self.structure:
            for chain in model:
                for residue in chain:
                    is_water = residue.name in water_residues

                    for atom in residue:
                        x, y, z = atom.pos.x, atom.pos.y, atom.pos.z

                        if is_water:
                            # Water mask
                            dist = np.sqrt((X - x)**2 + (Y - y)**2 + (Z - z)**2)
                            water_mask |= (dist < water_radius)
                        else:
                            # Protein mask
                            dist = np.sqrt((X - x)**2 + (Y - y)**2 + (Z - z)**2)
                            protein_mask |= (dist < protein_radius)

        # Bulk solvent: everything else
        bulk_solvent_mask = ~(protein_mask | water_mask)

        return {
            'protein': protein_mask,
            'water': water_mask,
            'bulk_solvent': bulk_solvent_mask,
            'total': np.ones((nx, ny, nz), dtype=bool)
        }

    def compare_maps(self, other_calculator: 'ENDMapAPI',
                    other_end_map: np.ndarray) -> Dict[str, dict]:
        """
        Compare this END map with another map across multiple regions.

        Parameters:
        -----------
        other_calculator : ENDMapAPI
            Other calculator (for consistency checking)
        other_end_map : array
            Other END map to compare

        Returns:
        --------
        results : dict
            RSCC and statistics for each region
        """
        if self.end_map is None:
            raise ValueError("Must compute END map first")

        if other_end_map.shape != self.end_map.shape:
            raise ValueError("Maps have different dimensions")

        # Create masks
        masks = self.create_masks()

        results = {}

        for region_name, mask in masks.items():
            if np.sum(mask) == 0:
                results[region_name] = {
                    'rscc': 0.0,
                    'voxels': 0,
                    'mean_density_1': 0.0,
                    'mean_density_2': 0.0,
                    'std_density_1': 0.0,
                    'std_density_2': 0.0
                }
                continue

            rscc = calculate_rscc(self.end_map, other_end_map, mask=mask)

            rho1_masked = self.end_map[mask]
            rho2_masked = other_end_map[mask]

            results[region_name] = {
                'rscc': rscc,
                'voxels': np.sum(mask),
                'voxel_fraction': np.sum(mask) / mask.size,
                'mean_density_1': np.mean(rho1_masked),
                'mean_density_2': np.mean(rho2_masked),
                'std_density_1': np.std(rho1_masked),
                'std_density_2': np.std(rho2_masked),
                'correlation': np.corrcoef(rho1_masked, rho2_masked)[0, 1]
            }

        return results

    def print_comparison(self, results: Dict[str, dict]) -> None:
        """
        Print formatted comparison results.

        Parameters:
        -----------
        results : dict
            Output from compare_maps()
        """
        print("\n" + "="*80)
        print("MULTI-REGION RSCC COMPARISON")
        print("="*80)
        print(f"{'Region':<20} {'RSCC':>8} {'Voxels':>12} {'Fraction':>10} {'Corr':>8}")
        print("-"*80)

        for region in ['total', 'protein', 'water', 'bulk_solvent']:
            if region not in results:
                continue

            res = results[region]
            rscc = res['rscc']
            voxels = res['voxels']
            frac = res.get('voxel_fraction', 0.0)
            corr = res.get('correlation', np.nan)

            print(f"{region:<20} {rscc:>8.4f} {voxels:>12} {frac:>9.1%} {corr:>8.4f}")

        print("="*80)
        print("\nDensity Statistics:")
        print("-"*80)
        print(f"{'Region':<20} {'Map1 Mean':>12} {'Map2 Mean':>12} {'Difference':>12}")
        print("-"*80)

        for region in ['total', 'protein', 'water', 'bulk_solvent']:
            if region not in results:
                continue

            res = results[region]
            mean1 = res['mean_density_1']
            mean2 = res['mean_density_2']
            diff = mean2 - mean1

            print(f"{region:<20} {mean1:>12.4f} {mean2:>12.4f} {diff:>+12.4f}")

        print("="*80)

    def print_statistics(self) -> None:
        """Print END map statistics"""
        if self.end_map is None:
            raise ValueError("Must compute END map first")

        print("\n=== END Map Statistics ===")
        print(f"Min density: {np.min(self.end_map):.4f} e-/Ų")
        print(f"Max density: {np.max(self.end_map):.4f} e-/Ų")
        print(f"Mean density: {np.mean(self.end_map):.4f} e-/Ų")
        print(f"Std dev: {np.std(self.end_map):.4f} e-/Ų")
        print(f"Median: {np.median(self.end_map):.4f} e-/Ų")

        # Count voxels in density ranges
        above_1 = np.sum(self.end_map > 1.0)
        above_3 = np.sum(self.end_map > 3.0)
        below_0 = np.sum(self.end_map < 0.0)

        print(f"\nVoxels >1.0 e-/Ų: {above_1} ({100*above_1/self.end_map.size:.1f}%)")
        print(f"Voxels >3.0 e-/Ų: {above_3} ({100*above_3/self.end_map.size:.1f}%)")
        print(f"Voxels <0.0 e-/Ų: {below_0} ({100*below_0/self.end_map.size:.1f}%)")


def calculate_rscc(map1: np.ndarray, map2: np.ndarray,
                  mask: Optional[np.ndarray] = None) -> float:
    """
    Calculate real-space correlation coefficient between two maps.

    Parameters:
    -----------
    map1, map2 : array
        Electron density maps
    mask : array, optional
        Boolean mask (True = include, False = exclude)

    Returns:
    --------
    rscc : float
        Correlation coefficient (-1 to 1)
    """
    if mask is not None:
        r1 = map1[mask].flatten()
        r2 = map2[mask].flatten()
    else:
        r1 = map1.flatten()
        r2 = map2.flatten()

    if len(r1) < 2:
        return 0.0

    cov = np.mean((r1 - np.mean(r1)) * (r2 - np.mean(r2)))
    std1 = np.std(r1)
    std2 = np.std(r2)

    if std1 == 0 or std2 == 0:
        return 0.0

    return cov / (std1 * std2)


# Example usage
if __name__ == "__main__":
    print(__doc__)

    # Example 1: Simple single map
    # calculator = ENDMapAPI("data.mtz", "model.pdb")
    # end_map = calculator.compute(n_molecules=1)
    # calculator.print_statistics()
    # calculator.save_ccp4("end_map.ccp4")

    # Example 2: Compare maps in multiple regions (your use case)
    # calc_mad = ENDMapAPI("data.mtz", "model.pdb", data_label="FOMAD,PHOMAD")
    # map_mad = calc_mad.compute(n_molecules=1)
    # calc_mad.save_ccp4("end_map_mad.ccp4")
    #
    # calc_refined = ENDMapAPI("data.mtz", "model.pdb", data_label="FWT,PHWT")
    # map_refined = calc_refined.compute(n_molecules=1)
    # calc_refined.save_ccp4("end_map_refined.ccp4")
    #
    # # Multi-region comparison
    # results = calc_mad.compare_maps(calc_refined, map_refined)
    # calc_mad.print_comparison(results)