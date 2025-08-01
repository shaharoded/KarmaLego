# KarmaLego

**Implementation of the KarmaLego time-interval pattern mining pipeline with end-to-end data ingestion, pattern discovery, and patient-level application.**

Based on the paper:  
Robert Moskovitch, Yuval Shahar. *Temporal Patterns Discovery from Multivariate Time Series via Temporal Abstraction and Time-Interval Mining.*  
(See original for theoretical grounding.)

This implementation is inspired by markozeman implementation, available on this [link](https://github.com/markozeman/KarmaLego), and is designed to be used as an analytic tool in my thesis.

---

## Overview

This repository provides:
- A **clean, efficient implementation** of KarmaLego (Karma + Lego) for discovering frequent Time Interval Relation Patterns (TIRPs).
- Support for **pandas / Dask-backed ingestion** of clinical-style interval data.
- Symbol encoding, pattern mining, and per-patient pattern application (counts or normalized distributions).
- Utilities for managing temporal relations, pattern equality/deduplication, and tree-based extension.

The design goals are: **clarity, performance, testability, and reproducibility**.

---

## KarmaLego Performance Optimizations
This implementation incorporates several core performance techniques from the KarmaLego framework:

1. **Apriori pruning:**
The algorithm only extends a pattern (e.g., from length k=2 to k=3) if the shorter base pattern is frequent. This avoids exploring any pattern whose sub-patterns fail the minimum support threshold.

2. **Temporal relation transitivity:**
During pattern extension (Lego phase), the code uses Allen relation composition to infer allowable temporal relations without explicitly scanning all combinations. This leverages a transition table (compose_relation) to reduce redundant comparisons.

3. **Subset of Active Candidates (SAC):**
When checking support for a candidate pattern, the algorithm restricts its scan to only those patients that supported its parent pattern. This drastically reduces horizontal support checks at deeper levels of the pattern tree.

4. **Memoization of relation composition:**
The core `compose_relation()` function is memoized using `@lru_cache`, avoiding redundant transitivity calculations across TIRPs. Since the input space is small (7x7 Allen relations), caching provides a measurable speedup in the Lego phase.

These optimizations ensure that KarmaLego runs efficiently on large temporal datasets and scales well as pattern complexity increases.

**Performance Notes:**
- The core KarmaLego algorithm operates on in-memory Python lists (`entity_list`) and is not accelerated by Dask.
- The current Lego phase runs sequentially. Attempts to parallelize it (e.g., with Dask or multiprocessing) introduced overhead that slowed performance.
- Dask can still be useful during ingestion and preprocessing (e.g., using `dd.read_csv()` for large CSVs).
- Fine-grained parallelism is not recommended due to fast per-node checks and high task management overhead. If the support task increases significantly, perhaps a patient-level parallelism of a TIRP will become useful.
- Better scaling can be achieved by:
  - Splitting the dataset into concept clusters or patient cohorts and running in parallel across jobs.
  - Using `min_ver_supp` and `max_k` to control pattern explosion.
  - Persisting symbol maps to ensure consistent encoding across runs.

---

## Repository Structure

```
KarmaLego/
├── core/
│   ├── __init__.py                             # package marker
│   ├── karmalego.py                            # algorithmic core: TreeNode, TIRP, KarmaLego/Karma/Lego pipeline
│   ├── io.py                                   # ingestion / preprocessing / mapping / decoding helpers
│   ├── relation_table.py                       # temporal relation transition tables and definitions
│   └── utils.py                                # low-level helpers
├── data/
│   ├── synthetic_diabetes_temporal_data.csv    # example input dataset
│   ├── symbol_map.json                         # saved symbol encoding (concept:value -> int)
│   └── inverse_symbol_map.json                 # reverse mapping for human-readable decoding
├── unittests/
│   ├── test_treenode.py                        # TreeNode behavior
│   ├── test_tirp.py                            # TIRP equality, support, relation semantics
│   └── test_karmalego.py                       # core pipeline / small synthetic pattern discovery
├── main.py                                     # example end-to-end driver / demo script
├── pyproject.toml                              # editable installation manifest
├── pytest.ini                                  # pytest configuration
├── requirements.txt                            # pinned dependencies (pandas, dask, tqdm, pytest, numpy, etc.)
├── README.md                                   # human-readable version of this document
└── .gitignore                                  # ignored files for git
```

---

## Installation

Recommended Python version: **3.8+**

Use a virtual environment:

```bash
python -m venv .venv
# Windows:
.\.venv\Scripts\activate

pip install -e .
pip install -r requirements.txt pytest
```

The `-e .` makes the local package importable as `core.karmalego` during development.

---

## Data Format Expected

Input must be a table (CSV or DataFrame) with these columns:

- `PatientID` : identifier per entity (patient)
- `ConceptName` : event or concept (e.g., lab test name)
- `StartDateTime` : interval start (e.g., `"08/01/2023 00:00"` in `DD/MM/YYYY HH:MM`)
- `EndDateTime` : interval end (same format)
- `Value` : discrete value or category (e.g., `'High'`, `'Normal'`)

You have full flexibility to affect the input and output shapes and formats in the `io.py` module, as long as you maintain this general structure.

Example row:
```
PatientID,ConceptName,StartTime,EndTime,Value
p1,HbA1c,08/01/2023 0:00,08/01/2023 0:15,High
```

---

## End-to-End Demo (main.py)

The provided `main.py` demonstrates the full pipeline:

1. **Load the CSV** (switch to Dask if scaling).
2. **Validate schema** and required fields.
3. **Build or load symbol mappings** (`ConceptName:Value` → integer codes).
4. **Preprocess**: parse dates, apply mapping.
5. **Convert to entity_list** (list of per-patient interval sequences).
6. **Discover patterns** using KarmaLego.
7. **Decode patterns** back to human-readable symbol strings.
8. **Apply patterns to each patient** (raw counts or TPP-normalized vectors).
9. **Persist outputs**: patterns CSV and patient-pattern matrix.

Example invocation:

```bash
python main.py
```

This produces:
- `discovered_patterns.csv` : flat table of frequent TIRPs with support and decoded symbols.
- `patient_pattern_vectors.csv` : long-form per-patient pattern distribution (normalized if TPP=True).

You can pivot `patient_pattern_vectors.csv` to a wide feature matrix for modeling.

---

## Key Concepts / Parameters

- `epsilon` : temporal tolerance for considering endpoints equal or meeting. Same unit as timestamps (use `pd.Timedelta` when using datetimes).
- `max_distance` : maximum gap between intervals to still consider them related (e.g., 1 hour → `pd.Timedelta(hours=1)`).
- `min_ver_supp` : minimum vertical support threshold (fraction of patients that must exhibit a pattern for it to be retained).
- `TPP` : normalized per-patient pattern distribution (pattern counts divided by total patterns for that patient).

---

## Programmatic Usage

### Discover Patterns

```python
from core.karmalego import KarmaLego

kl = KarmaLego(epsilon=pd.Timedelta(minutes=1),
               max_distance=pd.Timedelta(hours=1),
               min_ver_supp=0.03)

df_patterns = kl.discover_patterns(entity_list, min_length=1)  # returns DataFrame
```

### Apply to Patients

```python
patient_vectors = kl.apply_patterns_to_entities(entity_list, df_patterns, patient_ids, tpp=True)
```

---

## Unit-Testing

Run the full test suite:

```bash
pytest -q -s
```

Run a single test file:

```bash
pytest unittests/test_tirp.py -q -s
```

The `-s` flag shows pattern printouts and progress bars for debugging.

---

## Outputs

### Patterns DataFrame (`discovered_patterns.csv`)
Contains:
- `symbols` (tuple of encoded ints)
- `relations` (tuple of temporal relation codes)
- `k` (pattern length)
- `vertical_support`
- `support_count`
- `entity_indices_supporting`
- `indices_of_last_symbol_in_entities`
- `tirp_obj` (internal object; drop before sharing)
- `symbols_readable` (if decoded)

### Patient Pattern Vectors
Long format: `PatientID, Pattern (repr), Value (count or normalized)`.  
You can pivot for modeling:

```python
wide = df_vec.pivot(index="PatientID", columns="Pattern", values="Value").fillna(0)
```

---

## Tips for Scaling

- Replace `pandas.read_csv` with `dask.dataframe.read_csv` for large inputs; the ingestion helpers support Dask.
- Persist precomputed symbol maps to keep encoding stable across runs.
- Use categorical dtype for symbol column after mapping to reduce memory pressure.
- Tune `min_ver_supp` to control pattern explosion vs sensitivity.

---

## Development Notes

- Core logic lives in `core/karmalego.py`. Utilities (relation inference, transition tables) in `core/utils.py` and `core/relation_table.py`.  
- Input-Output logic lives in `core/io.py` and controls the formats, data structure, source and destination. Currently adjusted to work on local CPU with csv files. 
- The pattern tree is built lazily/iteratively; flat exports are used downstream for speed.  
- Equality and hashing ensure duplicate candidate patterns merge correctly.  
- Tests provide deterministic synthetic scenarios for regression.
