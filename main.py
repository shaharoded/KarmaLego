"""
This is a running sample of the full algorithm on temporal interval df (available in sample_date)
"""


if __name__ == "__main__":
    import pandas as pd
    import os
    from ast import literal_eval

    from core.io import (
        validate_input,
        build_or_load_mappings,
        preprocess_dataframe,
        to_entity_list
    )
    from core.karmalego import KarmaLego, TIRP


    # 1. Load (can swap to dask.read_csv if large)
    df = pd.read_csv("data/synthetic_diabetes_temporal_data.csv")

    # 2. Validate
    validate_input(df)

    # 3. Build mapping (concept+value -> symbol)
    symbol_map, _ = build_or_load_mappings(df, mapping_dir="data", reuse=True)

    # 4. Preprocess
    preprocessed = preprocess_dataframe(df, symbol_map)

    # 5. Convert to entity_list
    entity_list, patient_ids = to_entity_list(preprocessed)

    # 6. Discover patterns
    kl = KarmaLego(epsilon=pd.Timedelta(minutes=1),
                max_distance=pd.Timedelta(hours=1),
                min_ver_supp=0.5)

    patterns_path = "data/discovered_patterns.csv"

    # 6. Discover once or load from disk
    if os.path.exists(patterns_path):
        patterns_df = pd.read_csv(patterns_path)
    else:
        patterns_df = kl.discover_patterns(entity_list, min_length=1)
        patterns_df.to_csv(patterns_path, index=False)

    # Reconstruct TIRPs from CSV (fast) so keys/repr are consistent
    sym_series = patterns_df["symbols"].apply(lambda x: x if not isinstance(x, str) else literal_eval(x))
    rel_series = patterns_df["relations"].apply(lambda x: x if not isinstance(x, str) else literal_eval(x))
    patterns_tirps = [
        TIRP(
            epsilon=kl.epsilon,
            max_distance=kl.max_distance,
            min_ver_supp=kl.min_ver_supp,
            symbols=list(syms),
            relations=list(rels),
            k=len(syms),
        )
        for syms, rels in zip(sym_series, rel_series)
    ]

    # Pretty label map (repr(TIRP) -> human string from CSV)
    rep_to_str = {repr(t): s for t, s in zip(patterns_tirps, patterns_df["tirp_str"])}

    # Keep a stable order
    pattern_keys = [repr(t) for t in patterns_tirps]

    # 8. Apply – compute 5 columns
    vec_count_ul = kl.apply_patterns_to_entities(entity_list, patterns_tirps, patient_ids,
                                                mode="tirp-count", count_strategy="unique_last")
    vec_count_all = kl.apply_patterns_to_entities(entity_list, patterns_tirps, patient_ids,
                                                mode="tirp-count", count_strategy="all")
    vec_tpf_dist_ul = kl.apply_patterns_to_entities(entity_list, patterns_tirps, patient_ids,
                                                    mode="tpf-dist", count_strategy="unique_last")
    vec_tpf_dist_all = kl.apply_patterns_to_entities(entity_list, patterns_tirps, patient_ids,
                                                    mode="tpf-dist", count_strategy="all")
    vec_tpf_duration = kl.apply_patterns_to_entities(entity_list, patterns_tirps, patient_ids,
                                                    mode="tpf-duration", count_strategy="unique_last")

    # 9. One combined CSV
    rows = []
    for pid in patient_ids:
        for rep in pattern_keys:
            rows.append({
                "PatientID": pid,
                "Pattern": rep_to_str.get(rep, rep),
                "tirp_count_unique_last": vec_count_ul.get(pid, {}).get(rep, 0.0),
                "tirp_count_all":         vec_count_all.get(pid, {}).get(rep, 0.0),
                "tpf_dist_unique_last":   vec_tpf_dist_ul.get(pid, {}).get(rep, 0.0),
                "tpf_dist_all":           vec_tpf_dist_all.get(pid, {}).get(rep, 0.0),
                "tpf_duration":           vec_tpf_duration.get(pid, {}).get(rep, 0.0),
            })

    out_df = pd.DataFrame(rows)
    out_df.to_csv("data/patient_pattern_vectors.ALL.csv", index=False)