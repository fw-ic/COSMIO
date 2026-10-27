# Contributing to COSMIO

Thank you for helping improve the official COSMIO repository for the NeurIPS 2026 paper.

## Scope

This repository intentionally contains:

- COSMIO dataset readers and data protocol utilities;
- the AsynDiff imputation implementation;
- Roman strong-lens downstream classification;
- evaluation, visualization, documentation, and tests for those components.

Please do not submit implementations or copied source from additional imputation methods. Comparative methods should be referenced by paper and linked to their original upstream repositories.

## Development setup

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
python -m pip install -e asyn-diffusion
python -m pip install -e vit-lens-classification
python -m pip install pytest
pytest
```

Do not commit datasets, checkpoints, predictions, experiment runs, or generated Zarr stores. These paths are covered by `.gitignore`.

## Pull requests

Keep changes focused, add or update tests for behavior changes, and document any data-schema or split-protocol changes. For scientific results, include the exact configuration and random seed needed to reproduce them, together with the package versions used. Seeding covers the `random`, `numpy` and `torch` global generators; the data loaders set no explicit generator or worker seed, and no deterministic-algorithm or cuDNN flags are set, so results repeat run to run on one machine but are not bitwise reproducible across hosts or library versions.

## Contribution licensing

By submitting a contribution to this repository, you agree that your contribution
is licensed under the project's MIT License.
