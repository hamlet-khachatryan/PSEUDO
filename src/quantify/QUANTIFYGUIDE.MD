# Quantify Module

## Overview

The pseudo-quantify is a logic to analyze ensembles of stochastic omit perturbation (STOMP) maps. It's primary goal is to decouple true Background signal from phase bias (artifacts caused by the omission process itself) and systematic phase bias.

## Usage Examples

### Command Line Interface (CLI)
The primary method for running the pipeline is the `pseudo-quantify` command after perturbations.

```bash
  pseudo-quantify --input_path /path/to/workspace [OPTIONS]
```
| Option | Short | Type | Default | Description                                                                    |
| :----- | :----- |:-----| :----- |:-------------------------------------------------------------------------------|
| --input_path|-p|Path|Required| The root directory containing the experiment folders.                          |
| --stem|-s|String|Auto-infer| Explicitly specify the experiment name (e.g. experiment_01).                   |
| --k_factor|-k|Float|1.5| Radius multiplier (K) used to determine atom ownership spheres.                |
| --map_cap|-c|Int|None| Limit processing to the first N maps (e.g. 90). Useful for convergence testing.|
| --force|-f|Flag|False| Forces re-computation overwriting existing results in the output folder.       |

### Python API
For integration into larger Python pipelines or Jupyter notebooks, use the api module.

 ```python
from quantify.api import run_quantification
from pathlib import Path
    
# Run the pipeline
run_quantification(
        input_path=Path("./data"),
        force=True,
        k_factor=1.5,
        map_cap=None)  # Use all available maps
 ```
### I/O

The pipeline relies on a strict directory structure generated after debasing for inputs and produces standardized outputs.

The `workspace` path provided to the CLI must contain the following:

```text
workspace/
├── processed/
│   ├── {stem}_updated.pdb       # Refined atomic model (source of atom positions)
│   └── {stem}_original.pdb      # Original model
├── metadata/
│   └── {stem}_omission_map.json # Omission dictionary: {Atom/Residue Key -> List of Map IDs}
└── results/
    ├── {stem}_0/
    │   └── {stem}_0.mtz         # Omission Map #0
    ├── {stem}_1/
    │   └── {stem}_1.mtz         # Omission Map #1
    └── ...
```
Results are saved to `quantify_results/` inside the workspace.

|Output File|Description|
| :----- | :----- |
|{stem}_signal.ccp4|The mean density of the voxel after removing the systematic bias of neighboring atoms.|
|{stem}_noise.ccp4|The standard deviation of the cleaned voxel values across the ensemble.|
|{stem}_snr.ccp4|Signal-to-Noise Ratio. Calculated as Signal/Noise. Highlights statistically significant features.|
|{stem}_p_value.ccp4|Significance Map. P-values derived from a T-test against the bulk area distribution.|

## Algorithmic implementation
### Ownership logic 

```text
INPUT:
  Voxel coordinate v
  KDTree of all atoms
  Omission Map (Key -> List of omitted map IDs)
  Radius look-up table R(element, resolution)
  K_factor

OUTPUT:
  Status Matrix S of shape (N_maps, N_unique_owners)

PROCEDURE QueryOwnership(v):
  // 1. Spatial Search
  NearbyAtoms = KDTree.query_ball_point(v, r_max)
  
  // 2. Filter & Deduplicate
  ActiveOwners = Set()
  
  FOR each atom A in NearbyAtoms:
      d = distance(v, A.position)
      r_eff = R(A.element) * K_factor
      
      IF d <= r_eff:
          // Map physical atom to logical owner key
          // (e.g. "Residue_50" for amino_acids mode)
          OwnerKey = GetOwnerKey(A)
          ActiveOwners.add(OwnerKey)
  
  IF ActiveOwners is Empty:
      RETURN None

  // 3. Build Matrix
  Initialize S with 1s (Assuming presence)
  
  FOR j, Owner in enumerate(ActiveOwners):
      OmittedMapIDs = OmissionMap[Owner]
      
      FOR each MapID in OmittedMapIDs:
          // Mark as omitted (0) in this map
          S[MapID, j] = 0

  RETURN S
```

### Aggregation logic 
We separate the true background signal from the systematic bias introduced by the overlapping atoms using a linear system solver.

```text
INPUT:
  Vector D of observed densities (length N_maps)
  Status Matrix S (N_maps x N_owners)

OUTPUT:
  Signal, Noise

PROCEDURE SolveVoxel(D, S):
  // Empty voxel
  IF S is None:
      RETURN Mean(D), StdDev(D)

  // Occupied voxel
  N_owners = S.columns
  Biases = Vector of zeros (length N_owners)

  // Calculate Marginal Bias for each owner
  FOR j from 0 to N_owners - 1:
      Column C = S[:, j]
      
      // Check for contrast (must have both 0s and 1s)
      HasPresent = (1 in C)
      HasOmitted = (0 in C)
      
      IF HasPresent AND HasOmitted:
          Mean_Present = Mean(D where C == 1)
          Mean_Omitted = Mean(D where C == 0)
          Biases[j] = Mean_Present - Mean_Omitted
      ELSE:
          // Edge Case: No contrast in this map subset
          Biases[j] = 0.0

  // Subtract bias
  TotalBiasVector = MatrixMultiply(S, Biases)
  CleanedVector = D - TotalBiasVector

  // Compute statistics on cleaned data
  Signal = Mean(CleanedVector)
  Noise  = StdDev(CleanedVector)

  RETURN Signal, Noise
```