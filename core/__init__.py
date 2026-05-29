"""
KarmaLego - Temporal Interval Pattern Mining
============================================
Public API surface for the `core` package.

Usage::

    from core import KarmaLego, TIRP
    from core import validate_input, build_or_load_mappings, preprocess_dataframe, to_entity_list
    from core import ParallelRunner, run_parallel_jobs, run_parallel_batches
"""

from core.karmalego import KarmaLego, TIRP
from core.io import (
    validate_input,
    build_or_load_mappings,
    preprocess_dataframe,
    to_entity_list,
)
from core.parallel_runner import ParallelRunner, run_parallel_jobs, run_parallel_batches

__all__ = [
    "KarmaLego",
    "TIRP",
    "validate_input",
    "build_or_load_mappings",
    "preprocess_dataframe",
    "to_entity_list",
    "ParallelRunner",
    "run_parallel_jobs",
    "run_parallel_batches",
]
