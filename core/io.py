import os
import json
import pandas as pd
import dask.dataframe as dd
from typing import Tuple, Dict, Union

REQUIRED_COLUMNS = {"PatientID", "ConceptName", "StartDateTime", "EndDateTime", "Value"}


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
        if pdf["StartTime"].isnull().any() or pdf["EndTime"].isnull().any():
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
    start_col: str = "StartTime",
    end_col: str = "EndTime",
    patient_col: str = "PatientID",
    parse_dates: bool = True,
) -> Union[pd.DataFrame, dd.DataFrame]:
    """
    Normalize dtypes, apply symbol mapping to create integer symbol column, parse times.
    Returns DataFrame with columns: PatientID, symbol (int), StartTime (datetime), EndTime (datetime)
    """
    # Combine concept and value into raw symbol string
    def _make_symbol(pdf):
        pdf = pdf.copy()
        pdf["raw_symbol"] = pdf[concept_col].astype(str) + ":" + pdf[value_col].astype(str)
        pdf["symbol"] = pdf["raw_symbol"].map(symbol_map)
        if parse_dates:
            pdf[start_col] = pd.to_datetime(pdf[start_col])
            pdf[end_col] = pd.to_datetime(pdf[end_col])
        return pdf

    if isinstance(df, dd.DataFrame):
        df = df.map_partitions(_make_symbol, meta={
            patient_col: "int64",
            concept_col: "object",
            value_col: "object",
            start_col: "datetime64[ns]",
            end_col: "datetime64[ns]",
            "raw_symbol": "object",
            "symbol": "int64",
        })
    else:
        df = _make_symbol(df)

    # Optional: cast symbol to int category for memory
    df["symbol"] = df["symbol"].astype("int64")
    return df


def to_entity_list(
    df: Union[pd.DataFrame, dd.DataFrame],
    patient_col: str = "PatientID",
    start_col: str = "StartTime",
    end_col: str = "EndTime",
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


def decode_pattern_symbols(
    df_patterns, inverse_symbol_map: Dict[int, str]
) -> pd.DataFrame:
    """
    Given the patterns DataFrame (with 'symbols' as tuple of ints), expand to human-readable.
    """
    def decode_tuple(t):
        return tuple(inverse_symbol_map.get(i, f"UNK:{i}") for i in t)

    df = df_patterns.copy()
    df["symbols_readable"] = df["symbols"].apply(decode_tuple)
    return df