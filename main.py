"""
This is a running sample of the full algorithm on temporal interval df (available in sample_date)
"""

from core.io import (
    validate_input,
    build_or_load_mappings,
    preprocess_dataframe,
    to_entity_list,
    decode_pattern_symbols,
)
from core.karmalego import KarmaLego
import pandas as pd

# 1. Load (can swap to dask.read_csv if large)
df = pd.read_csv("data/synthetic_diabetes_temporal_data.csv")

# 2. Validate
validate_input(df)

# 3. Build mapping (concept+value -> symbol)
symbol_map, inverse = build_or_load_mappings(df, mapping_dir="data", reuse=True)

# 4. Preprocess
preprocessed = preprocess_dataframe(df, symbol_map)

# 5. Convert to entity_list
entity_list, patient_ids = to_entity_list(preprocessed)

# 6. Discover patterns
kl = KarmaLego(epsilon=pd.Timedelta(minutes=1), max_distance=pd.Timedelta(hours=1), min_ver_supp=0.03)  # tune parameters
patterns_df = kl.discover_patterns(entity_list, min_length=1)  # flat df

# 7. Decode symbols for readability
patterns_df = decode_pattern_symbols(patterns_df, inverse)
patterns_df.to_csv("data/discovered_patterns.csv", index=False)

# 8. Apply patterns to patients
patient_vectors = kl.apply_patterns_to_entities(entity_list, patterns_df, patient_ids, tpp=True)

# 9. Persist per-patient vector
import csv
with open("data/patient_pattern_vectors.csv", "w", newline="") as f:
    writer = csv.writer(f)
    # header: PatientID, pattern_repr, value (or pivot externally)
    writer.writerow(["PatientID", "Pattern", "Value"])
    for pid, vec in patient_vectors.items():
        for pattern_repr, val in vec.items():
            writer.writerow([pid, pattern_repr, val])