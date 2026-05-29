import os
import logging
import math
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd
from tqdm import tqdm

from core.karmalego import KarmaLego, TIRP
from core.utils import decode_pattern


_INIT_PARAM_KEYS = {"epsilon", "max_distance", "min_ver_supp", "num_relations"}
logger = logging.getLogger(__name__)


def _split_params(params):
    init_params = {
        "epsilon": params.get("epsilon"),
        "max_distance": params.get("max_distance"),
        "min_ver_supp": params.get("min_ver_supp"),
        "num_relations": params.get("num_relations", 7),
    }
    runtime_args = {k: v for k, v in params.items() if k not in _INIT_PARAM_KEYS}
    return init_params, runtime_args


def _discover_job_worker(job_info):
    """
    Worker function for a full independent KarmaLego run.
    Must remain top-level for Windows multiprocessing.
    """
    name = job_info["name"]
    data = job_info["data"]
    params = job_info["params"]

    init_params, runtime_args = _split_params(params)
    kl = KarmaLego(**init_params)

    result = kl.discover_patterns(data, **runtime_args)
    df = result[0] if isinstance(result, tuple) else result
    df = df.copy()
    df["job_name"] = name
    return df


def _discover_batch_worker(job_info):
    """
    Worker function for candidate generation on one patient batch.
    Returns only pattern signatures and batch metadata to keep IPC compact.
    """
    batch_id = job_info["batch_id"]
    data = job_info["data"]
    params = job_info["params"]

    init_params, runtime_args = _split_params(params)
    runtime_args["show_progress"] = False
    logging.getLogger("core.karmalego").setLevel(logging.WARNING)
    kl = KarmaLego(**init_params)

    result = kl.discover_patterns(data, **runtime_args)
    df = result[0] if isinstance(result, tuple) else result

    signatures = []
    if not df.empty:
        for row in df.itertuples(index=False):
            signatures.append((tuple(row.symbols), tuple(row.relations)))

    return {
        "batch_id": batch_id,
        "n_entities": len(data),
        "n_patterns": len(signatures),
        "signatures": signatures,
    }


def _finalize_candidate_chunk_worker(job_info):
    """
    Recompute exact full-cohort support for a chunk of candidate signatures.
    """
    logging.getLogger("core.karmalego").setLevel(logging.WARNING)
    init_params = job_info["init_params"]
    entity_list = job_info["entity_list"]
    signatures = job_info["signatures"]
    min_length = job_info["min_length"]
    max_length = job_info["max_length"]

    kl = KarmaLego(**init_params)
    finalized = []

    for symbols, relations in signatures:
        k = len(symbols)
        if k < min_length:
            continue
        if max_length is not None and k > max_length:
            continue

        tirp = TIRP(
            epsilon=kl.epsilon,
            max_distance=kl.max_distance,
            min_ver_supp=kl.min_ver_supp,
            symbols=list(symbols),
            relations=list(relations),
            k=k,
        )
        if not tirp.is_above_vertical_support(entity_list):
            continue
        finalized.append({
            "symbols": tuple(tirp.symbols),
            "relations": tuple(tirp.relations),
            "k": tirp.k,
            "vertical_support": tirp.vertical_support,
            "entity_indices_supporting": tuple(tirp.entity_indices_supporting),
            "indices_of_last_symbol_in_entities": tuple(tirp.indices_of_last_symbol_in_entities),
        })

    return finalized


def _decode_pattern_record(record, inverse_symbol_map):
    symbols = record["symbols"]
    relations = record["relations"]
    if not symbols:
        return ""

    parts = [inverse_symbol_map.get(str(symbols[0]), str(symbols[0]))]
    relation_index = 0
    for i in range(1, len(symbols)):
        name = inverse_symbol_map.get(str(symbols[i]), str(symbols[i]))
        rel = relations[relation_index]
        parts.append(f"{rel} {name}")
        relation_index += i
    return " ".join(parts)


class ParallelRunner:
    """
    Parallel execution helpers for KarmaLego.

    Modes
    -----
    parallel_jobs:
        Runs independent KarmaLego jobs and concatenates their DataFrames.

    parallel_batches:
        Splits one cohort into patient batches, discovers candidate patterns in
        each batch, unions candidates, then recomputes exact support on the full
        cohort before returning the final pattern DataFrame.
    """

    def __init__(self, max_workers=None, fallback_to_sequential=True):
        self.max_workers = max_workers
        self.fallback_to_sequential = fallback_to_sequential

    def parallel_jobs(self, jobs_list, max_workers=None):
        """
        Run multiple independent KarmaLego jobs and concatenate results.
        """
        if not jobs_list:
            return pd.DataFrame()

        workers = self._resolve_workers(len(jobs_list), max_workers)
        dfs = self._run_map(_discover_job_worker, jobs_list, workers,
                            desc="Running KarmaLego jobs")
        return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

    def parallel_batches(
        self,
        entity_list,
        params,
        batch_size=None,
        num_batches=None,
        max_workers=None,
        return_candidates=False,
        return_tirps=False,
        inverse_symbol_map=None,
        show_progress=True,
    ):
        """
        Exact batch-mode discovery for one cohort.

        Batch discovery is used only as candidate generation. Final support,
        closed flags, and super-pattern flags are recomputed on the full cohort.

        Parameters
        ----------
        entity_list : list
            Full cohort as a list of entities.
        params : dict
            KarmaLego init params plus discover params. Required init params:
            epsilon, max_distance, min_ver_supp. Optional: num_relations.
        batch_size : int, optional
            Number of entities per batch. Exactly one of batch_size or
            num_batches may be provided. If neither is provided, batches are
            sized to match the worker count.
        num_batches : int, optional
            Number of approximately equal batches.
        max_workers : int, optional
            Overrides the runner-level worker count for this call.
        return_candidates : bool
            If True, return (df, candidate_signatures, batch_summaries).
        return_tirps : bool
            If True, populate df["tirp_obj"] with finalized TIRP instances.
        inverse_symbol_map : dict, optional
            Symbol decoder used for tirp_str. Defaults to identity decoding.
        show_progress : bool
            Whether to show parent-process tqdm bars for batch discovery and
            full-cohort candidate finalization. Worker process Karma/Lego bars
            are always disabled in batch mode.
        """
        if not entity_list:
            empty = pd.DataFrame()
            return (empty, set(), []) if return_candidates else empty

        init_params, runtime_args = _split_params(params)
        min_length = runtime_args.get("min_length", 1)
        max_length = runtime_args.get("max_length")

        batches = self._make_batches(entity_list, batch_size, num_batches, max_workers)
        batch_jobs = [
            {"batch_id": idx, "data": batch, "params": params}
            for idx, batch in enumerate(batches)
            if batch
        ]

        workers = self._resolve_workers(len(batch_jobs), max_workers)
        batch_results = self._run_map(
            _discover_batch_worker,
            batch_jobs,
            workers,
            desc="Discovering batches" if show_progress else None,
        )

        candidate_signatures = set()
        for result in batch_results:
            candidate_signatures.update(result["signatures"])

        if batch_results:
            sizes = [r["n_entities"] for r in batch_results]
            pattern_counts = [r["n_patterns"] for r in batch_results]
            min_ver_supp = init_params["min_ver_supp"]
            min_abs_supports = [self._min_abs_support(min_ver_supp, size) for size in sizes]
            logger.info(
                "parallel_batches summary: batches=%d sizes=%s min_abs_supports=%s "
                "batch_patterns=%s unique_candidates=%d",
                len(batch_results),
                sizes,
                min_abs_supports,
                pattern_counts,
                len(candidate_signatures),
            )
            if min(min_abs_supports) <= 1 and len(entity_list) > 1:
                logger.warning(
                    "At least one batch has absolute support threshold 1. This can create "
                    "a very large candidate union; increase --batch-size or reduce "
                    "--n-workers for small cohorts."
                )

        df, tirps = self._finalize_candidates(
            candidate_signatures=candidate_signatures,
            entity_list=entity_list,
            init_params=init_params,
            min_length=min_length,
            max_length=max_length,
            max_workers=workers,
            return_tirps=return_tirps,
            inverse_symbol_map=inverse_symbol_map or {},
            show_progress=show_progress,
        )

        if return_candidates:
            return df, candidate_signatures, batch_results
        return df

    def _run_map(self, worker, jobs, workers, desc=None):
        if workers <= 1:
            iterator = tqdm(jobs, desc=desc, disable=desc is None)
            return [worker(job) for job in iterator]

        try:
            with ProcessPoolExecutor(max_workers=workers) as executor:
                futures = [executor.submit(worker, job) for job in jobs]
                iterator = tqdm(as_completed(futures), total=len(futures),
                                desc=desc, disable=desc is None)
                return [future.result() for future in iterator]
        except (OSError, PermissionError):
            if not self.fallback_to_sequential:
                raise
            iterator = tqdm(jobs, desc=desc, disable=desc is None)
            return [worker(job) for job in iterator]

    def _resolve_workers(self, task_count, max_workers=None):
        requested = max_workers if max_workers is not None else self.max_workers
        if requested is None:
            requested = os.cpu_count() or 1
        return max(1, min(int(requested), int(task_count)))

    def _make_batches(self, entity_list, batch_size, num_batches, max_workers):
        if batch_size is not None and num_batches is not None:
            raise ValueError("Provide either batch_size or num_batches, not both.")

        n_entities = len(entity_list)
        if batch_size is None:
            if num_batches is None:
                workers = self._resolve_workers(n_entities, max_workers)
                num_batches = workers
            if num_batches <= 0:
                raise ValueError("num_batches must be positive.")
            num_batches = min(num_batches, n_entities)
            base = n_entities // num_batches
            remainder = n_entities % num_batches
            batches = []
            start = 0
            for idx in range(num_batches):
                size = base + (1 if idx < remainder else 0)
                batches.append(entity_list[start:start + size])
                start += size
            return batches

        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")

        batches = [
            entity_list[start:start + batch_size]
            for start in range(0, n_entities, batch_size)
        ]
        if len(batches) > 1 and len(batches[-1]) < max(2, batch_size // 2):
            batches[-2].extend(batches[-1])
            batches.pop()
        return batches

    @staticmethod
    def _min_abs_support(min_ver_supp, batch_size):
        return math.ceil(min_ver_supp * batch_size)

    def _finalize_candidates(
        self,
        candidate_signatures,
        entity_list,
        init_params,
        min_length,
        max_length,
        max_workers,
        return_tirps,
        inverse_symbol_map,
        show_progress,
    ):
        sorted_candidates = sorted(candidate_signatures, key=lambda x: (len(x[0]), x[0], x[1]))
        workers = self._resolve_workers(len(sorted_candidates), max_workers) if sorted_candidates else 1
        finalized_records = self._finalize_candidate_records(
            sorted_candidates=sorted_candidates,
            entity_list=entity_list,
            init_params=init_params,
            min_length=min_length,
            max_length=max_length,
            workers=workers,
            show_progress=show_progress,
        )

        support_sets = [frozenset(r["entity_indices_supporting"]) for r in finalized_records]
        ks = [r["k"] for r in finalized_records]
        group = {}
        for fset, k in zip(support_sets, ks):
            if fset not in group:
                group[fset] = [k, k]
            else:
                group[fset][0] = min(group[fset][0], k)
                group[fset][1] = max(group[fset][1], k)

        records = []
        finalized_tirps = []
        kl = KarmaLego(**init_params)
        for rec, fset in zip(finalized_records, support_sets):
            min_k, max_k = group[fset]
            tirp_obj = None
            if return_tirps:
                tirp_obj = TIRP(
                    epsilon=kl.epsilon,
                    max_distance=kl.max_distance,
                    min_ver_supp=kl.min_ver_supp,
                    symbols=list(rec["symbols"]),
                    relations=list(rec["relations"]),
                    k=rec["k"],
                    vertical_support=rec["vertical_support"],
                    indices_supporting=list(rec["entity_indices_supporting"]),
                    indices_of_last_symbol_in_entities=list(rec["indices_of_last_symbol_in_entities"]),
                )
                finalized_tirps.append(tirp_obj)
            record = {
                "symbols": rec["symbols"],
                "relations": rec["relations"],
                "k": rec["k"],
                "vertical_support": rec["vertical_support"],
                "tirp_obj": tirp_obj,
                "tirp_str": (
                    decode_pattern(tirp_obj, inverse_symbol_map)
                    if tirp_obj is not None
                    else _decode_pattern_record(rec, inverse_symbol_map)
                ),
                "is_closed": max_k == rec["k"],
                "is_super_pattern": min_k < rec["k"] and max_k == rec["k"],
                "support_count": len(set(rec["entity_indices_supporting"])),
            }
            records.append(record)

        df = pd.DataFrame.from_records(records)
        if df.empty:
            return df, finalized_tirps

        df = df.sort_values(by=["k", "vertical_support"], ascending=[True, False]).reset_index(drop=True)
        return df, finalized_tirps

    def _finalize_candidate_records(
        self,
        sorted_candidates,
        entity_list,
        init_params,
        min_length,
        max_length,
        workers,
        show_progress,
    ):
        if not sorted_candidates:
            return []

        chunk_count = workers if workers > 1 else 1
        chunk_size = max(1, math.ceil(len(sorted_candidates) / chunk_count))
        chunks = [
            sorted_candidates[start:start + chunk_size]
            for start in range(0, len(sorted_candidates), chunk_size)
        ]
        jobs = [
            {
                "entity_list": entity_list,
                "init_params": init_params,
                "signatures": chunk,
                "min_length": min_length,
                "max_length": max_length,
            }
            for chunk in chunks
        ]

        if workers <= 1 or len(jobs) <= 1:
            iterator = tqdm(jobs, total=len(jobs), desc="Finalizing candidate chunks",
                            disable=not show_progress)
            nested = [_finalize_candidate_chunk_worker(job) for job in iterator]
        else:
            try:
                with ProcessPoolExecutor(max_workers=workers) as executor:
                    futures = [executor.submit(_finalize_candidate_chunk_worker, job) for job in jobs]
                    iterator = tqdm(as_completed(futures), total=len(futures),
                                    desc="Finalizing candidate chunks", disable=not show_progress)
                    nested = [future.result() for future in iterator]
            except (OSError, PermissionError):
                if not self.fallback_to_sequential:
                    raise
                iterator = tqdm(jobs, total=len(jobs), desc="Finalizing candidate chunks",
                                disable=not show_progress)
                nested = [_finalize_candidate_chunk_worker(job) for job in iterator]

        records = []
        for chunk_records in nested:
            records.extend(chunk_records)
        return records


def run_parallel_jobs(jobs_list, num_workers=None):
    """
    Backward-compatible wrapper around ParallelRunner.parallel_jobs.
    """
    return ParallelRunner(max_workers=num_workers).parallel_jobs(jobs_list)


def run_parallel_batches(
    entity_list,
    params,
    batch_size=None,
    num_batches=None,
    num_workers=None,
    return_candidates=False,
    return_tirps=False,
    inverse_symbol_map=None,
    show_progress=True,
):
    """
    Backward-compatible function wrapper for batch-mode discovery.
    """
    return ParallelRunner(max_workers=num_workers).parallel_batches(
        entity_list=entity_list,
        params=params,
        batch_size=batch_size,
        num_batches=num_batches,
        return_candidates=return_candidates,
        return_tirps=return_tirps,
        inverse_symbol_map=inverse_symbol_map,
        show_progress=show_progress,
    )
