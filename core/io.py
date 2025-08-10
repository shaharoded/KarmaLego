import os
import json
import numpy as np
import pandas as pd
import dask.dataframe as dd
from typing import Tuple, Dict, Union, Optional

REQUIRED_COLUMNS = {"PatientID", "ConceptName", "StartDateTime", "EndDateTime", "Value"}


def _parse_datetime_to_ns(series: pd.Series) -> Optional[pd.Series]:
    """
    Robust datetime parser:
      - Trims & normalizes whitespace
      - Zero-pads single-digit hours (" 1:23" -> " 01:23")
      - Tries multiple parsing strategies and picks the one with fewest NaT
      - Returns int64 ns on full success, None if nothing looks like a date
      - Raises if partially parseable (some NaT remain)
    """
    # Work on a clean string view
    s = series.astype(str).str.strip().str.replace("\u00A0", " ", regex=False)

    # Zero-pad single-digit hours after a space before colon: ' 1:' -> ' 01:'
    # (Windows-compatible; we avoid %-H)
    s = s.str.replace(r"(\s)(\d):", r"\g<1>0\2:", regex=True)

    candidates = []

    # 1) Generic parses (dayfirst False/True)
    for dayfirst in (False, True):
        dt = pd.to_datetime(s, errors="coerce", dayfirst=dayfirst)
        candidates.append(dt)

    # 2) Common explicit formats (seconds optional)
    for fmt in ("%m/%d/%Y %H:%M", "%m/%d/%Y %H:%M:%S",
                "%d/%m/%Y %H:%M", "%d/%m/%Y %H:%M:%S"):
        dt = pd.to_datetime(s, errors="coerce", format=fmt)
        candidates.append(dt)

    # Pick the candidate with the fewest NaT
    best = min(candidates, key=lambda x: x.isna().sum())
    nat_count = int(best.isna().sum())

    if nat_count == len(series):
        # Nothing looked like a date at all → signal: try numeric
        return None

    if nat_count > 0:
        # Partially parseable → raise with examples so data can be fixed upstream
        bad = series[best.isna()].head(5)
        raise ValueError(
            "Datetime parsing succeeded for some rows but failed for others. "
            "Fix or drop the unparseable values. Examples:\n"
            f"{bad}"
        )

    # Full success → return int64 nanoseconds
    return best.astype("int64")


def _normalize_time_series(series: pd.Series) -> pd.Series:
    """
    Convert a time column to int64:
      - Datetime-like or date-like strings → int64 nanoseconds
      - Fully numeric → int64 (no scaling)
    """
    # Already datetime-like?
    if pd.api.types.is_datetime64_any_dtype(series):
        return series.astype("int64")

    # Object/strings: try robust datetime parse
    if series.dtype == "object" or pd.api.types.is_string_dtype(series):
        parsed = _parse_datetime_to_ns(series)
        if parsed is not None:
            return parsed
        # else: nothing looked like a date → treat as numeric below

    # Numeric path (no scaling)
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.isna().any():
        bad = series[numeric.isna()].head(5)
        raise ValueError(
            "Time column contains values that are neither parseable datetimes nor numeric. "
            f"Examples:\n{bad}"
        )
    return np.rint(numeric.astype("float64")).astype("int64")


def validate_input(df: Union[pd.DataFrame, dd.DataFrame]) -> None:
    """
    Ensure required columns exist and have no nulls in key fields.
    Raises ValueError if validation fails.
    """
    cols = set(df.columns)
    missing = REQUIRED_COLUMNS - cols
    if missing:
        raise ValueError(f"Input missing required columns: {missing}")

    # Optionally check for nulls in essential columns (do a sample for dask)
    def _null_check(pdf):
        if pdf["PatientID"].isnull().any():
            raise ValueError("Null PatientID found")
        if pdf["ConceptName"].isnull().any():
            raise ValueError("Null ConceptName found")
        if pdf["StartDateTime"].isnull().any() or pdf["EndDateTime"].isnull().any():
            raise ValueError("Null time bounds found")
        return pdf

    if isinstance(df, dd.DataFrame):
        df.map_partitions(_null_check).compute()
    else:
        _null_check(df)


def build_or_load_mappings(
    df: Union[pd.DataFrame, dd.DataFrame],
    concept_col: str = "ConceptName",
    value_col: str = "Value",
    mapping_dir: str = ".",
    reuse: bool = True,
) -> Tuple[Dict[str, int], Dict[int, str]]:
    """
    Build (or load if exists and reuse=True) a stable mapping from symbol string to integer code.

    By default we combine concept and value: f"{ConceptName}:{Value}".
    Returns forward and inverse maps.
    """
    symbol_field = df[concept_col].astype(str) + ":" + df[value_col].astype(str)

    mapping_path = os.path.join(mapping_dir, "symbol_map.json")
    inverse_path = os.path.join(mapping_dir, "inverse_symbol_map.json")

    if reuse and os.path.exists(mapping_path) and os.path.exists(inverse_path):
        with open(mapping_path) as f:
            forward = json.load(f)
        with open(inverse_path) as f:
            inverse = json.load(f)
        # JSON keys are strings
        forward = {k: int(v) for k, v in forward.items()}
        inverse = {int(k): v for k, v in inverse.items()}
        return forward, inverse

    # Build fresh
    if isinstance(df, dd.DataFrame):
        unique_symbols = symbol_field.drop_duplicates().compute()
    else:
        unique_symbols = symbol_field.drop_duplicates()

    forward = {}
    inverse = {}
    for idx, sym in enumerate(sorted(unique_symbols)):
        forward[sym] = idx
        inverse[idx] = sym

    # persist
    with open(mapping_path, "w") as f:
        json.dump({k: v for k, v in forward.items()}, f)
    with open(inverse_path, "w") as f:
        json.dump({str(k): v for k, v in inverse.items()}, f)

    return forward, inverse


def preprocess_dataframe(
    df: Union[pd.DataFrame, dd.DataFrame],
    symbol_map: Dict[str, int],
    concept_col: str = "ConceptName",
    value_col: str = "Value",
    start_col: str = "StartDateTime",
    end_col: str = "EndDateTime",
    patient_col: str = "PatientID",
) -> Union[pd.DataFrame, dd.DataFrame]:
    """
    Normalize dtypes, map symbols to ints, and standardize times to int64.
    - Datetime or date-like strings -> int64 nanoseconds.
    - Numeric -> int64 (no scaling).
    Returns columns: [patient_col, 'symbol', start_col, end_col, 'raw_symbol'].
    """

    def _transform(pdf: pd.DataFrame) -> pd.DataFrame:
        pdf = pdf.copy()

        # Build symbols
        raw = pdf[concept_col].astype(str) + ":" + pdf[value_col].astype(str)
        pdf["raw_symbol"] = raw
        pdf["symbol"] = raw.map(symbol_map).astype("int64")

        # Times -> int64
        pdf[start_col] = _normalize_time_series(pdf[start_col])
        pdf[end_col]   = _normalize_time_series(pdf[end_col])

        # Sanity
        if (pdf[end_col] < pdf[start_col]).any():
            bad = pdf.loc[pdf[end_col] < pdf[start_col], [patient_col, start_col, end_col]].head(5)
            raise ValueError(f"Found rows with End < Start. Examples:\n{bad}")

        return pdf[[patient_col, "symbol", start_col, end_col, "raw_symbol"]]

    if isinstance(df, dd.DataFrame):
        # Let Dask infer meta; start/end become int64.
        return df.map_partitions(_transform, meta=None)
    else:
        return _transform(df)


def to_entity_list(
    df: Union[pd.DataFrame, dd.DataFrame],
    patient_col: str = "PatientID",
    start_col: str = "StartDateTime",
    end_col: str = "EndDateTime",
    symbol_col: str = "symbol",
) -> Tuple[list, list]:
    """
    Group preprocessed DataFrame into the entity_list format and parallel patient_ids.

    Returns:
        entity_list: list where each element is an entity = list of (start, end, symbol) tuples
        patient_ids: list of patient identifiers aligned with entity_list
    """
    if isinstance(df, dd.DataFrame):
        pdf = df.compute()
    else:
        pdf = df

    entity_list = []
    patient_ids = []

    grouped = pdf.groupby(patient_col)
    for pid, group in grouped:
        # sort by start time to help lexicographic sorting later (algorithm will sort again)
        group_sorted = group.sort_values(by=[start_col, end_col])
        entity = list(
            zip(
                group_sorted[start_col].tolist(),
                group_sorted[end_col].tolist(),
                group_sorted[symbol_col].tolist(),
            )
        )
        entity_list.append(entity)
        patient_ids.append(pid)
    return entity_list, patient_ids