# RingAug

[![PyPI version](https://img.shields.io/pypi/v/ringaug)](https://pypi.org/project/ringaug/)
[![Python](https://img.shields.io/pypi/pyversions/ringaug)](https://pypi.org/project/ringaug/)
![Augmentation](https://img.shields.io/badge/domain-augmentation-0ea5e9)
![Albumentations](https://img.shields.io/badge/library-albumentations-22c55e)
![Topology-Aware](https://img.shields.io/badge/focus-topology--aware-f59e0b)
![CLI](https://img.shields.io/badge/interface-CLI-64748b)
![License](https://img.shields.io/badge/license-MIT-a855f7)

RingAug is a Python package and CLI for topology-aware polygon augmentation on LabelMe datasets.

It is designed for workflows where image augmentation must preserve polygon index structure as much as possible, including safe repair of polygon vertex order after geometric transforms.

---

## Features

* Augments images and LabelMe polygon annotations together
* Uses mask-first augmentation for safer polygon handling
* Projects original polygon indices back to augmented contours
* Applies safe index-order repair when the augmented result remains a valid single polygon
* Exposes both:

  * a Python API
  * a command-line interface (`ringaug`)
* Packaged and published on PyPI

---

## Installation

### Install from PyPI

```bash
pip install ringaug
```

### Recommended: use a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install ringaug
```

### macOS note

If your system Python is managed by Homebrew, plain `pip install ringaug` outside a virtual environment may fail with an `externally-managed-environment` error.

In that case, use either:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install ringaug
```

or:

```bash
pipx install ringaug
```

---

## Quick CLI Check

After installation, verify the CLI is available:

```bash
ringaug --help
```

---

## Basic CLI Usage

```bash
ringaug \
  --img-dir ./dataset/images \
  --json-dir ./dataset/json \
  --save-dir ./output \
  --num-per-image 2 \
  --crop 0.8 0.9 \
  --rotation -30 30 \
  --scale 0.7 1.3 \
  --translate -0.1 0.1 \
  --brightness -0.1 0.1 \
  --contrast -0.1 0.1
```

### Inputs

* `--img-dir`: directory containing source images
* `--json-dir`: directory containing matching LabelMe JSON files

### Outputs

Inside `--save-dir`, RingAug writes:

* `images/` → augmented images
* `json/` → standard LabelMe JSON annotations
* `augmented_index_json/` → indexed/debug JSON annotations

---

## CLI Arguments

### Required

* `--img-dir`
* `--json-dir`

### Optional

* `--save-dir`
* `--index-json-dir`
* `--num-per-image`
* `--crop MIN MAX`
* `--rotation MIN MAX`
* `--scale MIN MAX`
* `--translate MIN MAX`
* `--brightness MIN MAX`
* `--contrast MIN MAX`
* `--p-rotate`
* `--p-flip-h`
* `--p-flip-v`
* `--p-affine`
* `--p-crop`
* `--p-brightness`
* `--contour-simplify-epsilon`
* `--min-component-area`
* `--min-mask-pixel-area`
* `--random-aug-per-image`
* `--debug`

To see the latest CLI options:

```bash
ringaug --help
```

---

## Python API Usage

```python
from ringaug import IndexPreservingPolygonAugmentor

params = {
    "crop_scale_range": (0.8, 0.9),
    "angle_limit": (-30, 30),
    "p_rotate": 0.9,
    "p_flip_h": 0.2,
    "p_flip_v": 0.1,
    "p_affine": 0.8,
    "scale_limit": (0.7, 1.3),
    "translate_limit": (-0.1, 0.1),
    "p_crop": 0.7,
    "brightness_limit": (-0.1, 0.1),
    "contrast_limit": (-0.1, 0.1),
    "p_brightness": 0.4,
    "random_aug_per_image": 3,
    "contour_simplify_epsilon": 1.5,
    "min_component_area": 12.0,
    "min_mask_pixel_area": 32,
    "min_repair_polygon_area": 1.0,
    "repair_dedupe_eps": 0.5,
    "source_overlap_eps": 0.5,
    "max_projection_distance_for_repair": 4.0,
    "min_retained_vertex_ratio_for_repair": 0.7,
}

augmentor = IndexPreservingPolygonAugmentor(debug=False)
augmentor.augment_dataset(
    data_dir="./dataset/images",
    json_dir="./dataset/json",
    save_img_dir="./output/images",
    save_json_dir="./output/json",
    save_index_json_dir="./output/augmented_index_json",
    num_augmentations=2,
    augmentation_params=params,
)
```

---

## Project Structure for Building the Package

A clean structure for the package is:

```text
ringaug/
├── LICENSE
├── README.md
├── pyproject.toml
├── src/
│   └── ringaug/
│       ├── __init__.py
│       ├── augmentor.py
│       ├── cli.py
│       └── helper.py
└── tests/
    └── test_imports.py
```

---

## Making the CLI Package

The CLI entry point is defined in `pyproject.toml`:

```toml
[project.scripts]
ringaug = "ringaug.cli:main"
```

This means that after installation, the command:

```bash
ringaug
```

runs:

```python
ringaug.cli.main()
```

---

## Example `pyproject.toml`

```toml
[build-system]
requires = ["setuptools>=69", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "ringaug"
version = "0.1.0"
description = "Polygon topology augmentation CLI for LabelMe datasets."
readme = "README.md"
requires-python = ">=3.10"
dependencies = [
    "albumentations>=2.0.8",
    "matplotlib>=3.8",
    "numpy>=1.26,<2.3",
    "opencv-python>=4.9",
    "pillow>=10.0",
    "scipy>=1.11,<1.16",
    "tqdm>=4.66",
]

[project.scripts]
ringaug = "ringaug.cli:main"

[tool.setuptools]
package-dir = {"" = "src"}

[tool.setuptools.packages.find]
where = ["src"]
include = ["ringaug*"]
```

---

## Versioning

The package version is controlled in `pyproject.toml`:

```toml
version = "0.1.0"
```

When making a new release:

* change the version number
* rebuild the package
* upload the new version to PyPI

Example:

* `0.1.0` → first public release
* `0.1.1` → small bug fix
* `0.2.0` → new features
* `1.0.0` → stable major release

---

## Local Development Workflow

### Install editable mode

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip setuptools wheel
pip install -e .
```

### Check CLI

```bash
ringaug --help
```

### Check Python import

```bash
python3 -c "from ringaug import IndexPreservingPolygonAugmentor; print(IndexPreservingPolygonAugmentor)"
```

---

## Build the Package

Install build tools:

```bash
python3 -m pip install --upgrade build twine
```

Build source distribution and wheel:

```bash
python3 -m build
```

This creates:

```text
dist/
├── ringaug-0.1.0.tar.gz
└── ringaug-0.1.0-py3-none-any.whl
```

---

## Validate the Package Before Upload

```bash
python3 -m twine check dist/*
```

---

## Test the Built Wheel in a Clean Environment

```bash
python3 -m venv .venv-test
source .venv-test/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install dist/ringaug-0.1.0-py3-none-any.whl
ringaug --help
```

This is important because it tests the actual packaged artifact, not just the local editable install.

---

## Publish to PyPI

### Step 1: create a PyPI account

Create an account on PyPI.

### Step 2: create an API token

For the first upload, you may need an account-scoped token if the project does not yet exist.

After the first upload, you can create a project-scoped token for better security.

### Step 3: upload

```bash
python3 -m twine upload dist/*
```

When prompted:

* username: `__token__`
* password: your full PyPI token beginning with `pypi-`

After upload, the package becomes available at:

```text
https://pypi.org/project/ringaug/
```

---

## Install From PyPI on Another Machine

Inside a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install ringaug
ringaug --help
```

---

## Recommended Release Workflow

Use this order for each release:

1. update code
2. update `README.md` if needed
3. bump version in `pyproject.toml`
4. rebuild package
5. run `twine check`
6. test install in a clean environment
7. upload to PyPI
8. verify install on another machine

---

## Example Release Checklist

```text
[ ] Update source code
[ ] Update version in pyproject.toml
[ ] Remove old dist/ build/ *.egg-info if needed
[ ] python3 -m build
[ ] python3 -m twine check dist/*
[ ] Test in clean virtualenv
[ ] python3 -m twine upload dist/*
[ ] pip install ringaug
[ ] ringaug --help
```

---

## Cleaning Old Build Artifacts

Before rebuilding a new release, you can remove old packaging outputs:

```bash
rm -rf build dist src/*.egg-info src/ringaug.egg-info
```

Then rebuild:

```bash
python3 -m build
```

---

## Common Issues

### 1. `pip install -e .` fails with editable install error

Usually caused by:

* old `pip`
* old `setuptools`
* incompatible build backend

Fix:

```bash
python3 -m pip install --upgrade pip setuptools wheel
```

### 2. `requires-python` mismatch

If your package says:

```toml
requires-python = ">=3.12"
```

but your machine uses Python 3.10, installation will fail.

Fix by setting a compatible version range.

### 3. macOS `externally-managed-environment`

Use a virtual environment instead of installing into system Python.

### 4. Dependency conflicts in old virtual environments

If unrelated packages conflict, create a clean new virtual environment and test there.

---

## Citation / Research Use

For reproducibility in research projects or papers, users can install RingAug with:

```bash
pip install ringaug
```

This makes the package easy to reuse in experiments and benchmarks.

---

## License

Specify your project license in `LICENSE`.

---

## Author

Sudip Laudari

---

## Future Improvements

Possible future CLI extensions:

```text
ringaug validate
ringaug visualize
ringaug benchmark
```

These can make the package more useful as a full research toolkit.
