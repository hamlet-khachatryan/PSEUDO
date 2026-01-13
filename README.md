# PSEUDO: Protein Structure Error and Uncertainty Determination and Optimisation

PSEUDO is a computational framework for debiasing and experimental uncertainty quantification in protein structural models resolved by molecular replacement.

## Installation

To install the package locally, run the following command in the project root:

```bash
  git clone [https://github.com/hamlet-khachatryan/PSEUDO.git](https://github.com/hamlet-khachatryan/PSEUDO.git)
  cd PSEUDO
  pip install .
```

For development installation (editable mode with dev dependencies):

```bash
  pip install -e ".[dev]"
```

### External Dependencies
This software requires the **Phenix Software Suite (Version 1.x)** for STOMP map calculation. Before running any scripts or submitting SLURM jobs, ensure Phenix is active in your environment.
On most HPC clusters, this is achieved by loading the module:

```bash
  module load phenix
  # Verify installation
  phenix.refine --version
```
