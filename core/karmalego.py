import logging
from functools import wraps
from copy import deepcopy
from collections import defaultdict
from tqdm import tqdm
import pandas as pd
import time
import json

from core.utils import (
    temporal_relations,
    lexicographic_sorting,
    check_symbols_lexicographically,
    find_all_possible_extensions,
    decode_pattern,
    count_embeddings_in_single_entity
)

# logging decorator
logger = logging.getLogger(__name__)
if not logger.hasHandlers():
    logging.basicConfig(level=logging.INFO)


def log_execution(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        logger.info(f"Starting {func.__name__}")
        result = func(*args, **kwargs)
        logger.info(f"Finished {func.__name__}")
        return result

    return wrapper


class TreeNode:
    """
    General-purpose tree node for enumerated TIRPs (or any pattern-like objects).
    Lightweight, iterative traversal, and supports parent links for fast upward navigation.

    Attr:
        __slots__: Cuts per node memory head.
    """
    __slots__ = ("data", "parent", "children", "_cached_subtree_nodes")


    def __init__(self, data=None, parent=None):
        """
        Lightweight tree node for TIRP enumeration.
        :param data: payload (e.g., a TIRP instance)
        :param parent: optional parent TreeNode
        """
        self.data = data
        self.parent = parent
        self.children = []
        self._cached_subtree_nodes = None  # cache for find_tree_nodes

    def add_child(self, child):
        """
        Attach a child and maintain parent pointer. Invalidates cache.

        :param child: child node to append
        :return: None
        """
        child.parent = self
        self.children.append(child)
        self._invalidate_cache_upward()
    
    def remove_child(self, child: "TreeNode"):
        """
        Remove direct child if present; detaches it. Invalidates cache.

        :param child: child node to remove, if exists in self.children.
        :return: None
        """
        try:
            self.children.remove(child)
            child.parent = None
            self._invalidate_cache_upward()
        except ValueError:
            pass  # not a child

    @property
    def depth(self) -> int:
        """
        Distance from root (root has depth 0).
        """
        d = 0
        node = self.parent
        while node is not None:
            d += 1
            node = node.parent
        return d

    def find_tree_nodes(self, filter_fn=None):
        """
        Iterative pre-order traversal collecting .data of nodes satisfying filter_fn.
        Default filter: duck-type detection of TIRP-like (has 'vertical_support').
        Caches result to avoid repeated work unless tree mutated.
        """
        if filter_fn is None:
            filter_fn = lambda d: hasattr(d, "vertical_support")

        if self._cached_subtree_nodes is not None and filter_fn == (lambda d: hasattr(d, "vertical_support")):
            # only reuse cache for default filter
            return list(self._cached_subtree_nodes)

        stack = [self]
        collected = []
        while stack:
            node = stack.pop()
            if node.data is not None and filter_fn(node.data):
                collected.append(node.data)
            # push children normally (no reverse needed unless ordering matters)
            stack.extend(node.children)

        if filter_fn == (lambda d: hasattr(d, "vertical_support")):
            self._cached_subtree_nodes = list(collected)  # cache default-filter result

        return collected

    def _invalidate_cache_upward(self):
        """
        Invalidate cached subtree nodes on this node and all ancestors.
        """
        node = self
        while node is not None:
            node._cached_subtree_nodes = None
            node = node.parent

    def __repr__(self):
        return f"<TreeNode depth={self.depth} data={self.data!r} children={len(self.children)}>"

    def print(self):
        """
        Inspection helper: prints all TIRPs under this node, sorted by descending vertical support.
        (Retains original semantics; may be relatively expensive as intended for logging.)
        """
        all_nodes = self.find_tree_nodes()
        for node in sorted(all_nodes, reverse=True):
            print(node, end="")
        print("\n\nAll TIRP nodes: ", len(all_nodes))
    

class TIRP:
    """
    Time Interval Relation Pattern (TIRP).

    Encapsulates a pattern defined by a sequence of symbols and their pairwise temporal relations,
    along with mechanisms to compute and track vertical support (which entities support the pattern),
    and provide deterministic equality/hash for deduplication in candidate enumeration.

    Attributes
    ----------
    epsilon : temporal tolerance used when comparing interval endpoints.
    max_distance : maximum gap between intervals to consider influence.
    min_ver_supp : threshold in range [0,1] for minimum vertical support to consider a pattern frequent.
    symbols : list of symbol identifiers (typically hashed concept names) in lexicographic order.
    relations : flat list representing the upper-triangular matrix of Allen-like relations between symbols.
    k : length of the pattern (number of symbols).
    vertical_support : computed support (fraction of entities supporting this TIRP).
    entity_indices_supporting : list of original entity indices that support this TIRP.
    parent_entity_indices_supporting : if extension of a parent, the parent-supporting indices to restrict search.
    indices_of_last_symbol_in_entities : for each supporting entity, the index of the last symbol in its embedding.
    _hash_cache : cached hash value to avoid recomputing in repeated set/dict usage.
    """

    __slots__ = (
        "epsilon",
        "max_distance",
        "min_ver_supp",
        "symbols",
        "relations",
        "k",
        "vertical_support",
        "entity_indices_supporting",
        "parent_entity_indices_supporting",
        "indices_of_last_symbol_in_entities",
        "_hash_cache",
    )

    def __init__(self, epsilon, max_distance, min_ver_supp, symbols=None, relations=None, k=1,
                vertical_support=None, indices_supporting=None, parent_indices_supporting=None,
                indices_of_last_symbol_in_entities=None, validate: bool = True):
        """
        Initialize a TIRP.

        Parameters
        ----------
        epsilon : numeric or timedelta-like
            Temporal tolerance for considering two interval endpoints as meeting/close.
        max_distance : numeric or timedelta-like
            Maximal distance between intervals for influence; beyond this treated as no relation.
        min_ver_supp : float
            Minimum vertical support threshold (0 <= min_ver_supp <= 1) to deem the TIRP frequent.
        symbols : list, optional
            Ordered list of symbol identifiers forming the TIRP.
        relations : list, optional
            Flattened list of temporal relations describing pairwise relations among symbols.
        k : int
            Declared length of the pattern; must equal len(symbols) if symbols provided.
        vertical_support : float, optional
            Precomputed vertical support value; must be in [0,1] if not None.
        indices_supporting : list of ints, optional
            Entity indices currently supporting this TIRP.
        parent_indices_supporting : list of ints, optional
            Supporting indices of the parent pattern, used to restrict search.
        indices_of_last_symbol_in_entities : list of ints, optional
            For each supporting entity, the index of the last symbol in the matched embedding.
        validate : bool, default=True
            Whether to enforce consistency checks (can be disabled if caller guarantees validity).
        """
        self.epsilon = epsilon
        self.max_distance = max_distance

        if validate:
            if not (0.0 <= min_ver_supp <= 1.0):
                raise ValueError(f"min_ver_supp must be in [0,1], got {min_ver_supp}")
        self.min_ver_supp = min_ver_supp

        self.symbols = [] if symbols is None else list(symbols)
        self.relations = [] if relations is None else list(relations)

        if validate:
            if self.symbols:
                expected_rel_len = (len(self.symbols) * (len(self.symbols) - 1)) // 2
                if len(self.relations) != expected_rel_len:
                    raise ValueError(
                        f"Inconsistent TIRP: {len(self.symbols)} symbols require {expected_rel_len} relations, "
                        f"got {len(self.relations)}"
                    )
            if k != len(self.symbols):
                raise ValueError(f"k ({k}) must equal number of symbols ({len(self.symbols)})")

        self.k = k
        if vertical_support is not None:
            if validate and not (0.0 <= vertical_support <= 1.0):
                raise ValueError(f"vertical_support must be in [0,1] if provided, got {vertical_support}")
        self.vertical_support = vertical_support

        # support tracking
        self.entity_indices_supporting = [] if indices_supporting is None else list(indices_supporting)
        self.parent_entity_indices_supporting = None if parent_indices_supporting is None else list(parent_indices_supporting)
        self.indices_of_last_symbol_in_entities = [] if indices_of_last_symbol_in_entities is None else list(indices_of_last_symbol_in_entities)
        self._hash_cache = None  # lazily populated

    def __eq__(self, other):
        """
        Equality is structural: same symbols sequence and same relations list.

        Parameters
        ----------
        other : any
            Object to compare against.

        Returns
        -------
        bool
            True if other is a TIRP with identical symbols and relations, False otherwise.
        """
        if not isinstance(other, TIRP):
            return False
        return self.symbols == other.symbols and self.relations == other.relations

    def __hash__(self):
        """
        Hash based on (symbols, relations) tuple. Caches after first computation for efficiency.

        Returns
        -------
        int
            Hash value.
        """
        if self._hash_cache is not None:
            return self._hash_cache
        sym_part = tuple(self.symbols)
        rel_part = tuple(self.relations)
        self._hash_cache = hash((sym_part, rel_part))
        return self._hash_cache

    def __lt__(self, other):
        """
        Comparison operator for sorting: uses vertical support (None treated as 0).
        Patterns with lower support are 'less' so that sorting(reverse=True) orders high support first.
        """
        return (self.vertical_support or 0) < (other.vertical_support or 0)

    def extend(self, new_symbol, new_relations):
        """
        Mutably extend this TIRP by appending a symbol and corresponding relation slice.
        Resets cached hash because the structural identity changes.

        Parameters
        ----------
        new_symbol : hashable
            The symbol to append (e.g., concept code).
        new_relations : list
            New relations to incorporate (must be consistent with upper-triangular relation encoding).

        Raises
        ------
        AttributeError
            If after extension the length invariants between symbols and relations are violated.
        """
        expected_after = ((len(self.symbols)+1) * len(self.symbols)) // 2  # new k = old_k+1
        if len(new_relations) != expected_after - len(self.relations):
            # this is the delta of relations being added; if inconsistent, log and raise
            raise AttributeError(
                f"Extension of TIRP is wrong: trying to add {len(new_relations)} relations "
                f"but expected {expected_after - len(self.relations)}. "
                f"base symbols={self.symbols}, base relations={self.relations}, "
                f"new_symbol={new_symbol}, new_relations={new_relations}"
            )
        self.symbols.append(new_symbol)
        self.relations.extend(new_relations)
        if not self.check_size():
            raise AttributeError(
                f"Extension of TIRP is wrong after append! symbols={self.symbols}, relations={self.relations}"
            )
        self._hash_cache = None  # reset cached hash

    def check_size(self):
        """
        Validate that the number of relations matches the expected upper-triangular size for current symbols.

        Returns
        -------
        bool
            True if relation list length is correct for current symbol count.
        """
        expected = (len(self.symbols) * (len(self.symbols) - 1)) // 2  # integer arithmetic
        return expected == len(self.relations)
    
    def __repr__(self):
        vs = f"{self.vertical_support:.3f}" if self.vertical_support is not None else "None"
        return f"TIRP(k={self.k}, symbols={self.symbols}, relations={self.relations}, support={vs})"

    def __str__(self):
        return self.__repr__()

    def print(self):
        """
        Pretty-print the TIRP as an upper triangular matrix (side effect: writes to stdout).
        Maintained exactly as in baseline/reference for human inspection.
        """
        if len(self.relations) == 0:
            return

        longest_symbol_name_len = len(max(self.symbols, key=len))
        longest_symbol_name_len_1 = len(max(self.symbols[:-1], key=len))

        print('\n\n', ' ' * longest_symbol_name_len_1, '‖', '   '.join(self.symbols[1:]))
        print('=' * (sum(len(s) for s in self.symbols[1:]) + longest_symbol_name_len_1 + 3 * len(self.symbols)))

        start_index = 0
        increment = 2
        for row_id in range(len(self.symbols) - 1):
            print(self.symbols[row_id], ' ' * (longest_symbol_name_len_1 - len(self.symbols[row_id])), '‖ ', end='')

            row_increment = row_id + 1
            index = start_index
            for column_id in range(len(self.symbols) - 1):
                num_of_spaces = len(self.symbols[column_id + 1]) + 2
                if column_id < row_id:
                    print(' ' * (num_of_spaces + 1), end='')
                else:
                    print(self.relations[index], end=' ' * num_of_spaces)
                    index += row_increment
                    row_increment += 1

            start_index += increment
            increment += 1

            if row_id != len(self.symbols) - 2:
                print()
                print('-' * (sum(len(s) for s in self.symbols[1:]) + longest_symbol_name_len + 3 * len(self.symbols)))

        return ""

    def is_above_vertical_support(self, entity_list, precomputed=None):
        """
        Compute and update vertical support for this TIRP over the provided entity list.
        “Given the set of entities (patients), how many of them contain this pattern 
        (with the right symbol order and temporal relations)?”

        This method:
          * Optionally restricts search to the parent-supported subset of entities (SAC-style pruning).
          * Finds lexicographic matches of the symbol sequence inside each entity.
          * Verifies that the temporal relations match for at least one embedding per entity.
          * Updates internal supporting index lists and vertical support.

        Parameters
        ----------
        entity_list : list
            List of entities, where each entity is itself a list of tuples
            (start, end, symbol). Symbols must align with self.symbols when matching.
        precomputed:
            List of dicts per entity with keys 'sorted' (lexicographically sorted intervals)
            and 'symbol_index' (symbol -> positions within that entity).   
            Allowes to avoid repeat sorting during Lego support checks. 
        Returns
        -------
        bool
            True if computed vertical support >= self.min_ver_supp, False otherwise.
        """
        # Apply parent-level filtering if available, with mapping to original indices.
        if self.parent_entity_indices_supporting is not None:
            reduced_indices = list(self.parent_entity_indices_supporting)
            entity_list_reduced = [entity_list[i] for i in reduced_indices]
            mapping_reduced_to_orig = {ri: reduced_indices[ri] for ri in range(len(reduced_indices))}
        else:
            entity_list_reduced = entity_list
            mapping_reduced_to_orig = {i: i for i in range(len(entity_list))}

        supporting_reduced = set()
        last_symbol_map = {}  # original_idx -> set of last symbol positions

        for reduced_idx, entity in enumerate(entity_list_reduced):
            orig_idx = mapping_reduced_to_orig[reduced_idx]
            lexi_sorted = lexicographic_sorting(entity)
            if precomputed is not None:
                lexi_sorted = precomputed[orig_idx]["sorted"]
            else:
                lexi_sorted = lexicographic_sorting(entity)
            entity_ti = [(s, e) for s, e, _ in lexi_sorted]
            entity_symbols = [sym for _, _, sym in lexi_sorted]

            if len(self.symbols) > len(entity_symbols):
                continue  # cannot embed if pattern longer than entity

            matching_options = check_symbols_lexicographically(entity_symbols, self.symbols)
            if matching_options is None:
                continue

            for matching_option in matching_options:
                all_relations_match = True
                relation_index = 0

                # Walk the implied upper-triangular structure of relations.
                for column_count, entity_index in enumerate(matching_option[1:]):
                    for row_count in range(column_count + 1):
                        ti_1 = entity_ti[matching_option[row_count]]
                        ti_2 = entity_ti[entity_index]
                        expected_rel = self.relations[relation_index]
                        actual_rel = temporal_relations(ti_1, ti_2, self.epsilon, self.max_distance)
                        if expected_rel != actual_rel:
                            all_relations_match = False
                            break
                        relation_index += 1
                    if not all_relations_match:
                        break

                if all_relations_match:
                    supporting_reduced.add(reduced_idx)
                    last_symbol = matching_option[-1]
                    last_symbol_map.setdefault(orig_idx, set()).add(last_symbol)
                    break  # only one valid embedding per entity is enough

        # Translate reduced support indices back to original entity indices
        new_supporting = set()
        for reduced_idx in supporting_reduced:
            orig = mapping_reduced_to_orig[reduced_idx]
            new_supporting.add(orig)
        self.entity_indices_supporting = list(new_supporting)

        # Align last-symbol positions with supporting entity indices
        paired = set()
        for orig_idx in self.entity_indices_supporting:
            last_symbols = last_symbol_map.get(orig_idx, [])
            for sym_idx in last_symbols:
                paired.add((sym_idx, orig_idx))
        if paired:
            sym_idxs, ent_idxs = zip(*paired)
            self.indices_of_last_symbol_in_entities = list(sym_idxs)
            self.entity_indices_supporting = list(ent_idxs)
        else:
            self.indices_of_last_symbol_in_entities = []

        # Final vertical support is with respect to the full original entity list.
        self.vertical_support = len(set(self.entity_indices_supporting)) / len(entity_list) if entity_list else 0.0
        return self.vertical_support >= self.min_ver_supp


class KarmaLego:
    """
    Orchestrates the Karma→Lego pattern mining pipeline.

    Attributes
    ----------
    epsilon:
        Temporal tolerance for endpoint equality/meeting.
    max_distance:
        Maximum gap for considering influence between intervals.
    min_ver_supp:
        Minimum vertical support threshold for a pattern to be kept.
    """
    def __init__(self, epsilon, max_distance, min_ver_supp):
        self.epsilon = epsilon
        self.max_distance = max_distance
        self.min_ver_supp = min_ver_supp

    @log_execution
    def discover_patterns(
        self, entity_list, min_length=1, return_tree=False, return_tirps=False, inverse_mapping_path="data/inverse_symbol_map.json"
    ):
        """
        Discover all frequent TIRPs from entity_list and return a flat DataFrame summary (default).

        Parameters
        ----------
        entity_list : list
            List of entities. Each entity is a list of (start, end, symbol) tuples.
        min_length : int
            Minimum pattern length (k) to include.
        return_tree : bool
            If True, also return the internal pattern tree used for extension.
        return_tirps : bool
            If True, also return the list of TIRP objects in addition to DataFrame.
        inverse_mapping_path : str
            Path for inverse mapping file to map symbols to readable TIRPs.

        Returns
        -------
        df : pandas.DataFrame
            Flat table of discovered patterns with metadata.
        tirps : list[TIRP], optional
            List of TIRP instances (if return_tirps=True).
        tree : TreeNode, optional
            Root of internal pattern tree (if return_tree=True).
        """
        with open(inverse_mapping_path) as f:
            inverse_symbol_map = json.load(f)
        t0 = time.perf_counter()

        # Precompute sorted entities and symbol→positions index once.
        t_pre_start = time.perf_counter()
        precomputed = []
        for entity in entity_list:
            lexi = lexicographic_sorting(entity)
            symbol_to_positions = defaultdict(list)
            for pos, (_, _, sym) in enumerate(lexi):
                symbol_to_positions[sym].append(pos)
            precomputed.append({"sorted": lexi, "symbol_index": symbol_to_positions})
        t_pre_end = time.perf_counter()

        # Karma phase: requires precomputed passed in
        t_karma_start = time.perf_counter()
        karma = Karma(self.epsilon, self.max_distance, self.min_ver_supp)
        tree = karma.run_karma(entity_list, precomputed)
        t_karma_end = time.perf_counter()

        # Lego extension
        t_lego_start = time.perf_counter()
        lego = Lego(tree, self.epsilon, self.max_distance, self.min_ver_supp, show_detail=True)
        full_tree = lego.run_lego(tree, entity_list, precomputed)
        t_lego_end = time.perf_counter()

        # Flatten and filter
        t_flatten_start = time.perf_counter()
        all_tirps = full_tree.find_tree_nodes()
        filtered = [t for t in all_tirps if t.k >= min_length]

        # Support set comparison for closed/super flags
        support_sets = [set(t.entity_indices_supporting) for t in filtered]
        ks = [t.k for t in filtered]

        is_closed = [True] * len(filtered)
        is_super = [False] * len(filtered)

        for i, (supp_i, k_i) in enumerate(zip(support_sets, ks)):
            for j, (supp_j, k_j) in enumerate(zip(support_sets, ks)):
                if i != j:
                    if supp_i == supp_j:
                        if k_j > k_i:
                            is_closed[i] = False
                        if k_j < k_i:
                            is_super[i] = True

        # Build DataFrame
        records = []
        for idx, tirp in enumerate(filtered):
            record = {
                "symbols": tuple(tirp.symbols),
                "relations": tuple(tirp.relations),
                "k": tirp.k,
                "vertical_support": tirp.vertical_support,
                "tirp_obj": tirp,
                "tirp_str": decode_pattern(tirp, inverse_symbol_map),
                "is_closed": is_closed[idx],
                "is_super_pattern": is_super[idx],
            }
            records.append(record)

        df = pd.DataFrame.from_records(records)
        df = df.sort_values(by=["k", "vertical_support"], ascending=[True, False]).reset_index(drop=True)
        t_flatten_end = time.perf_counter()

        # Log timings
        total = time.perf_counter() - t0
        SECONDS_PER_MINUTE = 60.0
        logger.info(
            "discover_patterns timings (min): precompute=%.4f karma=%.4f lego=%.4f "
            "flatten=%.4f total=%.4f",
            (t_pre_end   - t_pre_start)  / SECONDS_PER_MINUTE,
            (t_karma_end - t_karma_start) / SECONDS_PER_MINUTE,
            (t_lego_end  - t_lego_start)  / SECONDS_PER_MINUTE,
            (t_flatten_end - t_flatten_start) / SECONDS_PER_MINUTE,
            total / SECONDS_PER_MINUTE,
        )

        outputs = (df,)
        if return_tirps:
            outputs += ([r["tirp_obj"] for r in records],)
        if return_tree:
            outputs += (full_tree,)

        if len(outputs) == 1:
            return outputs[0]
        return outputs

    @log_execution
    def apply_patterns_to_entities(self, entity_list, patterns, patient_ids, tpp=False):
        """
        Given discovered patterns, build per-patient feature vectors (counts or normalized TPP).

        Parameters
        ----------
        entity_list : list
            List of entities, parallel to patient_ids.
        patterns : list[TIRP] or pandas.Series/DataFrame column of TIRPs
            Patterns to apply.
        patient_ids : list
            Identifiers matching order of entity_list.
        tpp : bool
            If True, normalize per-patient distribution.

        Returns
        -------
        dict
            patient_id -> {pattern_repr: count or normalized value}
        
        NOTE: You can pass this function a pre-calculated list of TIRPs like:
        patterns_df = karma_lego.discover_patterns(...)
        tirps = patterns_df['tirp_obj'].tolist()
        result = karma_lego.apply_patterns_to_entities(entity_list, tirps, patient_ids)
        """
        patient_pattern_counts = {pid: defaultdict(int) for pid in patient_ids}

        # Ensure we have TIRP objects if a DataFrame column passed
        patterns_list = []
        if hasattr(patterns, "itertuples") or isinstance(patterns, pd.DataFrame):
            # assume column 'tirp_obj' exists
            patterns_list = list(patterns["tirp_obj"])
        elif isinstance(patterns, (list, tuple)):
            patterns_list = patterns
        else:
            raise ValueError("Unsupported patterns container")

        for tirp in tqdm(patterns_list, desc="Applying patterns to entities"):
            key = repr(tirp)
            for idx, entity in enumerate(entity_list):
                count = count_embeddings_in_single_entity(tirp, entity)
                if count > 0:
                    patient_pattern_counts[patient_ids[idx]][key] = count

        if tpp:
            for pid, counts in patient_pattern_counts.items():
                total = sum(counts.values())
                if total > 0:
                    for k in list(counts.keys()):
                        counts[k] = counts[k] / total

        return patient_pattern_counts
    

class Karma(KarmaLego):
    def __init__(self, epsilon, max_distance, min_ver_supp):
        super().__init__(epsilon, max_distance, min_ver_supp)

    def run_karma(self, entity_list, precomputed):
        """
        Karma phase: discover frequent singletons and all length-2 TIRPs.

        Uses precomputed sorted entities and symbol position indexes to avoid recomputation.

        Parameters
        ----------
        entity_list :
            List of entities (each a list of (start, end, symbol)).
        precomputed :
            List of dicts per entity with keys 'sorted' (lexicographically sorted intervals)
            and 'symbol_index' (symbol -> positions within that entity).

        Returns
        -------
        TreeNode
            Root node whose children are singleton TIRPs, with length-2 TIRPs attached as appropriate.
        """
        tree = TreeNode("root")
        frequent_symbols = set()
        symbol_to_singleton_node = {}

        # estimate total work: number of unique symbols + total number of ordered pairs across entities
        symbol_set = {sym for ent in entity_list for _, _, sym in ent}
        total_pairs = sum(
            (len(entry["sorted"]) * (len(entry["sorted"]) - 1)) // 2
            for entry in precomputed
        )
        total_work = len(symbol_set) + total_pairs

        with tqdm(total=total_work, desc="Karma phase") as karma_bar:
            # SINGLETONS
            for sym in symbol_set:
                supporting_pairs = set()
                for eid, entry in enumerate(precomputed):
                    positions = entry["symbol_index"].get(sym, [])
                    for pos in positions:
                        supporting_pairs.add((pos, eid))

                if supporting_pairs:
                    sym_idxs, ent_idxs = zip(*supporting_pairs)
                    entity_indices_supporting = list(set(ent_idxs))
                    indices_of_last_symbol_in_entities = list(set(sym_idxs))
                    vertical_support = len(set(entity_indices_supporting)) / len(entity_list)

                    if vertical_support >= self.min_ver_supp:
                        tirp_single = TIRP(
                            epsilon=self.epsilon,
                            max_distance=self.max_distance,
                            min_ver_supp=self.min_ver_supp,
                            symbols=[sym],
                            relations=[],
                            k=1,
                        )
                        tirp_single.entity_indices_supporting = entity_indices_supporting
                        tirp_single.indices_of_last_symbol_in_entities = indices_of_last_symbol_in_entities
                        tirp_single.vertical_support = vertical_support

                        node = TreeNode(tirp_single)
                        tree.add_child(node)
                        symbol_to_singleton_node[sym] = node
                        frequent_symbols.add(sym)
                karma_bar.update(1)  # one unit per symbol processed

            # PAIRS (k=2)
            tirp_dict = {}
            for eid, entry in enumerate(precomputed):
                ordered = entry["sorted"]
                for i in range(len(ordered)):
                    for j in range(i + 1, len(ordered)):
                        start_1, end_1, symbol_1 = ordered[i]
                        start_2, end_2, symbol_2 = ordered[j]
                        if symbol_1 not in frequent_symbols or symbol_2 not in frequent_symbols:
                            karma_bar.update(1)
                            continue
                        rel = temporal_relations((start_1, end_1), (start_2, end_2), self.epsilon, self.max_distance)
                        if rel is None:
                            karma_bar.update(1)
                            continue

                        tirp = TIRP(
                            epsilon=self.epsilon,
                            max_distance=self.max_distance,
                            min_ver_supp=self.min_ver_supp,
                            symbols=[symbol_1, symbol_2],
                            relations=[rel],
                            k=2,
                        )
                        key = hash(tirp)
                        if key not in tirp_dict:
                            tirp.entity_indices_supporting = [eid]
                            tirp.indices_of_last_symbol_in_entities = [j]
                            tirp_dict[key] = tirp
                        else:
                            existing = tirp_dict[key]
                            existing.entity_indices_supporting.append(eid)
                            existing.indices_of_last_symbol_in_entities.append(j)
                        karma_bar.update(1)  # one unit per pair attempted

            # finalize pairs and attach
            for tirp in tirp_dict.values():
                if tirp.entity_indices_supporting:
                    unique = set(zip(tirp.indices_of_last_symbol_in_entities, tirp.entity_indices_supporting))
                    if unique:
                        sym_idxs, ent_idxs = zip(*unique)
                        tirp.indices_of_last_symbol_in_entities = list(sym_idxs)
                        tirp.entity_indices_supporting = list(ent_idxs)
                tirp.vertical_support = len(set(tirp.entity_indices_supporting)) / len(entity_list) if entity_list else 0.0

                if tirp.vertical_support >= self.min_ver_supp:
                    parent_symbol = tirp.symbols[0]
                    parent_node = symbol_to_singleton_node.get(parent_symbol)
                    if parent_node is not None:
                        parent_node.add_child(TreeNode(tirp))

        return tree


class Lego(KarmaLego):
    """
    Lego phase driver (pattern extension).

    Parameters
    ----------
    tree :
        Pattern tree produced by Karma.
    show_detail :
        Whether to show per-TIRP extension progress bars.
    """
    def __init__(self, tree, epsilon, max_distance, min_ver_supp, show_detail):
        self.tree = tree
        super().__init__(epsilon, max_distance, min_ver_supp)
        self.show_detail = show_detail # whether to keep per-extension verbosity

    def run_lego(self, node, entity_list, precomputed):
        """
        Extend base patterns recursively to higher-order TIRPs.

        Breadth-first traverses the tree, attempting all valid one-symbol extensions and
        attaching those meeting vertical support.

        Parameters
        ----------
        node :
            Root TreeNode to start extension from.
        entity_list :
            List of entities for support computation.
        precomputed:   
            List of dicts per entity with keys 'sorted' (lexicographically sorted intervals)
            and 'symbol_index' (symbol -> positions within that entity). 

        Returns
        -------
        TreeNode
            The same tree with extended TIRPs grafted in.
        """
        # Breadth-first expansion queue
        queue = [node]
        with tqdm(desc="Lego phase (nodes expanded)", unit=" node/s") as bar:
            while queue:
                current = queue.pop(0)
                if isinstance(current.data, TIRP):
                    extensions = list(set(self.all_extensions(entity_list, current.data)))
                    ok = []
                    iterator = extensions
                    if self.show_detail:
                        iterator = tqdm(extensions, desc=f"Extending TIRP k={current.data.k}", leave=False)
                    for ext in iterator:
                        if ext.is_above_vertical_support(entity_list, precomputed=precomputed):
                            ok.append(ext)
                    for ext in ok:
                        child = TreeNode(ext)
                        current.add_child(child)
                        queue.append(child)
                # also continue to existing children regardless (they may have been added earlier)
                for child in current.children:
                    if child not in queue:
                        # avoid duplicates
                        queue.append(child)
                bar.update(1)
        return node

    def all_extensions(self, entity_list, tirp):
        """
        Enumerate candidate one-symbol extensions of a given TIRP.

        Builds new TIRPs by adding one symbol and computing the required predecessor
        relation sequences; enforces structural invariants and fails loudly if violated.

        Parameters
        ----------
        entity_list :
            All entities to base extension on.
        tirp :
            Base TIRP to extend.

        Returns
        -------
        list[TIRP]
            New candidate TIRPs of length k+1 (not yet filtered by support).
        """
        curr_num_of_symbols = len(tirp.symbols)
        all_possible = []
        for sym_index, ent_index in zip(tirp.indices_of_last_symbol_in_entities, tirp.entity_indices_supporting):
            lexi_entity = lexicographic_sorting(entity_list[ent_index])
            if curr_num_of_symbols >= len(lexi_entity):
                continue
            for after_sym in lexi_entity[sym_index + 1 :]:
                *new_ti, new_symbol = after_sym
                rel_last_new = temporal_relations(
                    lexi_entity[sym_index][:2], tuple(new_ti), self.epsilon, self.max_distance
                )
                if rel_last_new is None:
                    continue

                curr_rel_index = len(tirp.relations) - 1
                decrement_index = curr_num_of_symbols - 1

                # get predecessor relation sequences (excluding the new-last relation)
                all_paths = []
                all_paths = find_all_possible_extensions(
                    all_paths, [], rel_last_new, curr_rel_index, decrement_index, tirp.relations
                )

                for path in all_paths:
                    new_relations = [rel_last_new, *path]
                    # expected number of new relations when extending by one symbol is current pattern length
                    if len(new_relations) != curr_num_of_symbols:
                        # Implementation invariant violated: surface loudly for debugging
                        raise RuntimeError(
                            f"Malformed extension candidate: base_k={curr_num_of_symbols}, "
                            f"computed new_relations={new_relations} (len={len(new_relations)})"
                        )

                    new_relations.reverse()  # match original ordering semantics
                    child = TIRP(
                    epsilon=self.epsilon,
                    max_distance=self.max_distance,
                    min_ver_supp=self.min_ver_supp,
                    symbols=[*tirp.symbols, new_symbol],
                    relations=[*tirp.relations, *new_relations],
                    k=tirp.k + 1,
                    )
                    # SAC: propagate parent-supported entities; CSAC propagation will follow below
                    child.parent_entity_indices_supporting = list(tirp.entity_indices_supporting)
                    child.entity_indices_supporting = []
                    child.indices_of_last_symbol_in_entities = []
                    all_possible.append(child)
        return all_possible


