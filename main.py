"""
End-to-end KarmaLego runner.

Examples:
    python main.py
    python main.py --input-data-path data/input/temporal_data.csv --n-workers 4
    python main.py --no-apply-patterns --force-discover
"""

import argparse
import os
from ast import literal_eval

import pandas as pd

from core.io import (
    validate_input,
    build_or_load_mappings,
    preprocess_dataframe,
    to_entity_list,
)
from core.karmalego import KarmaLego, TIRP
from core.parallel_runner import ParallelRunner


def parse_args():
    parser = argparse.ArgumentParser(description="Run KarmaLego discovery and/or pattern application.")

    parser.add_argument(
        "--input-data-path",
        "--input_data_path",
        default="data/input/synthetic_diabetes_temporal_data.csv",
        help="Input CSV path.",
    )
    parser.add_argument(
        "--output-dir",
        "--output_dir",
        default="data/output",
        help="Directory for symbol maps, discovered patterns, and patient vectors.",
    )
    parser.add_argument(
        "--n-workers",
        "--n_workers",
        type=int,
        default=1,
        help="Worker count for discovery and apply. 1 uses regular KarmaLego discovery.",
    )

    discover_group = parser.add_mutually_exclusive_group()
    discover_group.add_argument(
        "--discover-patterns",
        "--discover_patterns",
        dest="discover_patterns",
        action="store_true",
        default=True,
        help="Discover patterns if needed. Default.",
    )
    discover_group.add_argument(
        "--no-discover-patterns",
        "--no_discover_patterns",
        dest="discover_patterns",
        action="store_false",
        help="Skip discovery and load discovered patterns from output-dir.",
    )

    apply_group = parser.add_mutually_exclusive_group()
    apply_group.add_argument(
        "--apply-patterns",
        "--apply_patterns",
        dest="apply_patterns",
        action="store_true",
        default=True,
        help="Apply discovered patterns to patients. Default.",
    )
    apply_group.add_argument(
        "--no-apply-patterns",
        "--no_apply_patterns",
        dest="apply_patterns",
        action="store_false",
        help="Skip patient-vector generation.",
    )

    parser.add_argument(
        "--force-discover",
        "--force_discover",
        action="store_true",
        help="Recompute patterns even if discovered_patterns.csv already exists.",
    )
    parser.add_argument("--min-length", "--min_length", type=int, default=1)
    parser.add_argument("--max-length", "--max_length", type=int, default=None)
    parser.add_argument("--min-ver-supp", "--min_ver_supp", type=float, default=0.5)
    parser.add_argument("--num-relations", "--num_relations", type=int, default=7)
    parser.add_argument("--epsilon-seconds", "--epsilon_seconds", type=float, default=2.0)
    parser.add_argument("--max-distance-hours", "--max_distance_hours", type=float, default=4.0)
    parser.add_argument(
        "--batch-size",
        "--batch_size",
        type=int,
        default=None,
        help="Batch size for parallel discovery. Defaults to approximately one batch per worker.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Disable tqdm progress bars where supported.",
    )
    return parser.parse_args()


def build_kl(args):
    return KarmaLego(
        epsilon=pd.Timedelta(seconds=args.epsilon_seconds),
        max_distance=pd.Timedelta(hours=args.max_distance_hours),
        min_ver_supp=args.min_ver_supp,
        num_relations=args.num_relations,
    )


def discovery_params(args):
    return {
        "epsilon": pd.Timedelta(seconds=args.epsilon_seconds),
        "max_distance": pd.Timedelta(hours=args.max_distance_hours),
        "min_ver_supp": args.min_ver_supp,
        "num_relations": args.num_relations,
        "min_length": args.min_length,
        "max_length": args.max_length,
        "inverse_mapping_path": os.path.join(args.output_dir, "inverse_symbol_map.json"),
        "show_progress": not args.quiet,
    }


def load_and_prepare_data(args):
    df = pd.read_csv(args.input_data_path)
    validate_input(df)

    symbol_map, _ = build_or_load_mappings(df, mapping_dir=args.output_dir, reuse=True)
    preprocessed = preprocess_dataframe(df, symbol_map)
    return to_entity_list(preprocessed)


def discover_or_load_patterns(args, entity_list, patterns_path):
    should_load = os.path.exists(patterns_path) and not args.force_discover
    if should_load:
        return pd.read_csv(patterns_path)

    if not args.discover_patterns:
        raise FileNotFoundError(
            f"{patterns_path} does not exist and --no-discover-patterns was supplied."
        )

    if args.n_workers <= 1:
        kl = build_kl(args)
        patterns_df = kl.discover_patterns(
            entity_list,
            min_length=args.min_length,
            max_length=args.max_length,
            inverse_mapping_path=os.path.join(args.output_dir, "inverse_symbol_map.json"),
            show_progress=not args.quiet,
        )
    else:
        runner = ParallelRunner(max_workers=args.n_workers)
        patterns_df = runner.parallel_batches(
            entity_list,
            params=discovery_params(args),
            batch_size=args.batch_size,
            show_progress=not args.quiet,
        )

    patterns_df.to_csv(patterns_path, index=False)
    return patterns_df


def reconstruct_tirps(patterns_df, kl):
    sym_series = patterns_df["symbols"].apply(lambda x: x if not isinstance(x, str) else literal_eval(x))
    rel_series = patterns_df["relations"].apply(lambda x: x if not isinstance(x, str) else literal_eval(x))
    return [
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


def apply_patterns(args, entity_list, patient_ids, patterns_df, vectors_path):
    kl = build_kl(args)
    patterns_tirps = reconstruct_tirps(patterns_df, kl)

    rep_to_str = {
        (tuple(t.symbols), tuple(t.relations)): s
        for t, s in zip(patterns_tirps, patterns_df["tirp_str"])
    }
    pattern_keys = [(tuple(t.symbols), tuple(t.relations)) for t in patterns_tirps]

    apply_workers = None if args.n_workers is None else max(1, args.n_workers)
    progress = not args.quiet

    vec_count_ul = kl.apply_patterns_to_entities(
        entity_list, patterns_tirps, patient_ids,
        mode="tirp-count", count_strategy="unique_last",
        n_jobs=apply_workers, show_progress=progress,
    )
    vec_count_all = kl.apply_patterns_to_entities(
        entity_list, patterns_tirps, patient_ids,
        mode="tirp-count", count_strategy="all",
        n_jobs=apply_workers, show_progress=progress,
    )
    vec_tpf_dist_ul = kl.apply_patterns_to_entities(
        entity_list, patterns_tirps, patient_ids,
        mode="tpf-dist", count_strategy="unique_last",
        n_jobs=apply_workers, show_progress=progress,
    )
    vec_tpf_dist_all = kl.apply_patterns_to_entities(
        entity_list, patterns_tirps, patient_ids,
        mode="tpf-dist", count_strategy="all",
        n_jobs=apply_workers, show_progress=progress,
    )
    vec_tpf_duration = kl.apply_patterns_to_entities(
        entity_list, patterns_tirps, patient_ids,
        mode="tpf-duration", count_strategy="unique_last",
        n_jobs=apply_workers, show_progress=progress,
    )

    rows = []
    for pid in patient_ids:
        for rep in pattern_keys:
            rows.append({
                "PatientID": pid,
                "Pattern": rep_to_str.get(rep, rep),
                "tirp_count_unique_last": vec_count_ul.get(pid, {}).get(rep, 0.0),
                "tirp_count_all": vec_count_all.get(pid, {}).get(rep, 0.0),
                "tpf_dist_unique_last": vec_tpf_dist_ul.get(pid, {}).get(rep, 0.0),
                "tpf_dist_all": vec_tpf_dist_all.get(pid, {}).get(rep, 0.0),
                "tpf_duration": vec_tpf_duration.get(pid, {}).get(rep, 0.0),
            })

    pd.DataFrame(rows).to_csv(vectors_path, index=False)


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    patterns_path = os.path.join(args.output_dir, "discovered_patterns.csv")
    vectors_path = os.path.join(args.output_dir, "patient_pattern_vectors.ALL.csv")

    entity_list, patient_ids = load_and_prepare_data(args)
    patterns_df = discover_or_load_patterns(args, entity_list, patterns_path)

    if args.apply_patterns:
        apply_patterns(args, entity_list, patient_ids, patterns_df, vectors_path)


if __name__ == "__main__":
    main()
