# KarmaLego

Implementation skeleton of the **KarmaLego** time-interval pattern mining algorithm, inspired by:  
**Robert Moskovitch, Yuval Shahar.**  
*Temporal Patterns Discovery from Multivariate Time Series via Temporal Abstraction and Time-Interval Mining.*  
(see original paper for full algorithmic grounding and terminology)

PDF: https://pdfs.semanticscholar.org/a800/83f16631756d0865e13f679c2d5084df03ae.pdf

---

## Summary / Purpose

This repository provides the core building blocks for KarmaLego:
- `TreeNode`: efficient, mutable tree structure for enumerating pattern expansions.
- `TIRP`: representation of Time Interval Relation Patterns with support computation, structural equality, and hashing.
- (Planned) `Karma` and `Lego` orchestration logic to generate frequent patterns and extend them hierarchically.
- Utilities are separated for clarity/efficiency; algorithmic types live in `karmalego.py`.

The design emphasizes:
- **Efficiency**: no recursion in traversal, caching, and clear support pruning.
- **Correctness**: structural identity, support computation, parent filtering.
- **Testability**: synthetic fixtures and extensive tests to validate pattern semantics.

---

## Quickstart / Getting Started

### Requirements

- Python 3.8+  
- Recommended to work inside a virtual environment.

Dependencies (see `requirements.txt` — populate with, e.g.):
pytest
pandas
dask[dataframe]
tqdm
numpy

### Installation (editable)

From the repo root:

```bash
python -m venv .venv
source .venv/Scripts/activate      # or `source .venv/bin/activate` on Unix
pip install -e .
pip install -r requirements.txt    # if you maintain that file
```
This makes the package importable as core.karmalego and allows iterative development.

### Running Tests
All core logic is exercised via pytest. To run the current test suite:

```bash
pytest -q -s
```
Or target specific test files:

```bash
pytest unittests/test_tirp_extended.py -q -s
```
The `-s` flag disables capture so you can see printed pattern summaries and matrices for debugging/inspection.


## Project Structure

```bash
KarmaLego/
├── core/
│   ├── __init__.py         # package marker
│   └── karmalego.py        # core classes: TreeNode, TIRP, (future Karma/Lego)
├── unittests/
│   ├── __init__.py
│   ├── test_treenode.py    # base tree node tests
│   └── test_tirp_extended.py  # extensive TIRP/support/equality/extension tests
├── pyproject.toml         # project metadata (editable install)
├── pytest.ini            # pytest discovery config
├── .gitignore
├── requirements.txt      # runtime/test dependencies
└── README.md             # this document
```