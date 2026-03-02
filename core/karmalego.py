import bisect
import logging
from functools import wraps
from collections import defaultdict
from tqdm import tqdm
import pandas as pd
import time
import json
from ast import literal_eval

from core.utils import (
    normalize_time_param,
    temporal_relations,
    lexicographic_sorting,
    check_symbols_lexicographically,
    find_all_possible_extensions,
    decode_pattern
)
from core.relation_table import set_relation_table, get_sac_relations

# logging decorator
logger = logging.getLogger(__name__)
if not logger.hasHandlers():
    logging.basicConfig(level=logging.INFO)


def _default_tirp_filter(d):
    """Default filter for find_tree_nodes: matches any TIRP-like object."""
    return hasattr(d, "vertical_support")


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
            filter_fn = _default_tirp_filter

        if self._cached_subtree_nodes is not None and filter_fn is _default_tirp_filter:
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

        if filter_fn is _default_tirp_filter:
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
    parent_embeddings_map: dict[int -> list[tuple]] parent embeddings per entity.
    embeddings_map: dict[int -> list[tuple]] this TIRP's embeddings per entity
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
        "parent_embeddings_map",
        "embeddings_map",      
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
        
        # lazily populated:
        self._hash_cache = None
        self.parent_embeddings_map = None
        self.embeddings_map = None

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
    
    def is_above_vertical_support(self, entity_list, precomputed=None, level2_index=None):
        """
        Compute and update vertical support for this TIRP over the provided entity list.
        Optimized to use parent embeddings for guided search (Forward Pruning).
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
        supporting_reduced = set()
        last_symbol_map = {}  # original_idx -> set of last symbol positions
        _sac_rels = get_sac_relations()  # relation codes that trigger CSAC adjacency check

        # Determine which entities to check
        indices_to_check = self.parent_entity_indices_supporting if self.parent_entity_indices_supporting is not None else range(len(entity_list))

        # Early-abandon: precompute constants once.
        # If current_support + remaining_to_check < min_support_count it is impossible
        # to reach the threshold no matter what the remaining entities return.
        # We only abandon for patterns that will be pruned, so embeddings_map is unaffected.
        total_to_check = len(indices_to_check)
        n_entities = len(entity_list)
        min_support_count = self.min_ver_supp * n_entities  # float; compared with int sums

        # Check if we can use the optimized guided extension
        use_guided_extension = getattr(self, "parent_embeddings_map", None) is not None

        # Pre-compute level-2 index constants for the guided extension path.
        # These depend only on self (this TIRP), not on any specific entity, so we
        # compute them once here rather than repeating the work on every entity iteration.
        _ge_target_symbol = None      # new symbol being appended
        _ge_rels_to_check = None      # relations from all k-1 prev positions to the new symbol
        _ge_l2_eid_map    = None      # {eid: {pos_A → [pos_B, ...]}} for the direct last pair
        _ge_check_count   = None      # relations to verify per candidate (one less than total;
                                      # the last pair is always pre-verified by the index)
        if use_guided_extension:
            assert level2_index is not None, (
                "level2_index must be provided when parent_embeddings_map is set; "
                "call discover_patterns rather than is_above_vertical_support directly "
                "on guided-extension TIRPs."
            )
            _ge_target_symbol = self.symbols[-1]
            _ge_rels_to_check = self.relations[-(self.k - 1):]
            l2_key = (self.symbols[-2], _ge_target_symbol, _ge_rels_to_check[-1])
            _ge_l2_eid_map = level2_index.get(l2_key)
            if _ge_l2_eid_map is None:
                # The k=2 sub-pattern (sym[-2], sym[-1], last_rel) is absent from the index,
                # meaning it was not frequent.  By Apriori this TIRP cannot be frequent either.
                # This branch is unreachable in a correctly-pruned Lego traversal.
                self.vertical_support = 0.0
                self.parent_embeddings_map = None
                return False
            _ge_check_count = len(_ge_rels_to_check) - 1  # last pair always pre-verified

        for check_pos, orig_idx in enumerate(indices_to_check):
            # Get the entity data
            if precomputed is not None:
                lexi_sorted = precomputed[orig_idx]["sorted"]
                pairwise_rels = precomputed[orig_idx].get("pairwise_rels")
            else:
                lexi_sorted = lexicographic_sorting(entity_list[orig_idx])
                pairwise_rels = None

            entity_ti = [(s, e) for s, e, _ in lexi_sorted]
            entity_symbols = [sym for _, _, sym in lexi_sorted]
            
            valid_embeddings_here = []

            # Pre-extract symbol_index once; used by CSAC adjacency checks in both paths below.
            if precomputed is not None:
                symbol_index = precomputed[orig_idx]["symbol_index"]
            else:
                symbol_index = {}
                for _pos, _sym in enumerate(entity_symbols):
                    symbol_index.setdefault(_sym, []).append(_pos)

            if use_guided_extension:
                # --- OPTIMIZED PATH: Extend parent embeddings ---
                # Re-use constants computed before the entity loop.
                target_symbol = _ge_target_symbol
                rels_to_check = _ge_rels_to_check
                check_count   = _ge_check_count
                parent_embeddings = self.parent_embeddings_map.get(orig_idx, [])

                # Per-entity view of the level-2 index: pos_A → [pos_B, ...].
                # Candidates pulled from here are already relation-verified and CSAC-compliant
                # (filtered during Karma), so we skip the last relation and CSAC check.
                l2_pos_dict = _ge_l2_eid_map.get(orig_idx)
                if l2_pos_dict is None:
                    # This entity has no CSAC-valid (sym_A, sym_B, last_rel) pair from Karma;
                    # no embedding can be formed for it.
                    continue

                for parent_tup in parent_embeddings:
                    parent_last_idx = parent_tup[-1]
                    # O(1) lookup: only pre-verified pos_B values for this pos_A.
                    candidates_i = l2_pos_dict.get(parent_last_idx, ())

                    for i in candidates_i:
                        # Verify relations from each earlier parent position to new position i.
                        # check_count == len(rels_to_check) - 1: the last pair (sym[-2]→sym[-1])
                        # is already verified by the index; only the j earlier pairs remain.
                        all_relations_match = True
                        for j in range(check_count):
                            prev_entity_idx = parent_tup[j]
                            expected_rel    = rels_to_check[j]
                            actual_rel = (pairwise_rels.get((prev_entity_idx, i))
                                          if pairwise_rels is not None
                                          else temporal_relations(entity_ti[prev_entity_idx], entity_ti[i],
                                                                  self.epsilon, self.max_distance))
                            if expected_rel != actual_rel:
                                all_relations_match = False
                                break

                        if all_relations_match:
                            # CSAC: check the earlier pairs (same count as relation checks).
                            # The last-pair adjacency is already guaranteed by the index.
                            sac_ok = True
                            for j in range(check_count):
                                if rels_to_check[j] in _sac_rels:
                                    prev_e_idx = parent_tup[j]
                                    sym_prev = lexi_sorted[prev_e_idx][2]
                                    sym_pos = symbol_index.get(sym_prev, [])
                                    lo_s = bisect.bisect_right(sym_pos, prev_e_idx)
                                    if lo_s < len(sym_pos) and sym_pos[lo_s] < i:
                                        sac_ok = False
                                        break
                            if sac_ok:
                                valid_embeddings_here.append(parent_tup + (i,))
            
            else:
                # --- FULL SEARCH PATH: Generate from scratch ---
                # Necessary for k=1 or when parent embeddings are missing (e.g. tests)
                if len(self.symbols) > len(entity_symbols):
                    continue

                matching_options = check_symbols_lexicographically(entity_symbols, self.symbols)
                if not matching_options:
                    continue

                for matching_option in matching_options:
                    all_relations_match = True
                    relation_index = 0
                    # Verify all pairwise relations
                    for column_count, entity_index in enumerate(matching_option[1:]):
                        for row_count in range(column_count + 1):
                            ti_1 = entity_ti[matching_option[row_count]]
                            ti_2 = entity_ti[entity_index]
                            expected_rel = self.relations[relation_index]
                            i1, i2 = matching_option[row_count], entity_index
                            actual_rel = (pairwise_rels.get((i1, i2))
                                          if pairwise_rels is not None
                                          else temporal_relations(ti_1, ti_2, self.epsilon, self.max_distance))
                            if expected_rel != actual_rel:
                                all_relations_match = False
                                break
                            # CSAC: for ordering relations, verify no same-concept occurrence
                            # between positions i1 and i2 (adjacency constraint).
                            if expected_rel in _sac_rels:
                                sym_pos = symbol_index.get(entity_symbols[i1], [])
                                lo_s = bisect.bisect_right(sym_pos, i1)
                                if lo_s < len(sym_pos) and sym_pos[lo_s] < i2:
                                    all_relations_match = False
                                    break
                            relation_index += 1
                        if not all_relations_match:
                            break
                    
                    if all_relations_match:
                        valid_embeddings_here.append(matching_option)

            # --- Common Storage Logic ---
            if valid_embeddings_here:
                supporting_reduced.add(orig_idx)
                
                # Record last indices
                for tup in valid_embeddings_here:
                    last_symbol = tup[-1]
                    last_symbol_map.setdefault(orig_idx, set()).add(last_symbol)
                
                # Store embeddings
                if self.embeddings_map is None:
                    self.embeddings_map = {}
                
                # Store embeddings — valid_embeddings_here contains unique tuples by
                # construction (guided path: distinct parent_tup × unique target positions;
                # full-search path: check_symbols_lexicographically returns strictly
                # increasing index sequences). No set deduplication needed.
                self.embeddings_map[orig_idx] = sorted(valid_embeddings_here)

            # Early-abandon: upper bound on achievable support.
            # Even if every remaining entity supports this TIRP, can we reach the threshold?
            remaining = total_to_check - check_pos - 1
            if len(supporting_reduced) + remaining < min_support_count:
                self.vertical_support = len(supporting_reduced) / n_entities if n_entities else 0.0
                self.parent_embeddings_map = None
                return False

        # --- Finalize Support ---
        self.entity_indices_supporting = list(supporting_reduced)

        # Align last-symbol positions
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
            self.entity_indices_supporting = []

        self.vertical_support = len(supporting_reduced) / len(entity_list) if entity_list else 0.0
        
        # Cleanup
        self.parent_embeddings_map = None
        
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
    def __init__(self, epsilon, max_distance, min_ver_supp, num_relations=7):
        """
        Initialize KarmaLego with temporal pattern discovery parameters.

        Parameters
        ----------
        epsilon : float or pd.Timedelta
            Temporal tolerance for matching interval endpoints.
        max_distance : float or pd.Timedelta
            Maximum temporal gap to consider influence between intervals.
        min_ver_supp : float
            Minimum vertical support threshold for a pattern to be kept.
        num_relations : int, default=7
            Number of temporal relations to support: 2, 3, 5, or 7.
            - 2: ultra-coarse (proceed/contain) - fastest
            - 3: minimal - fast
            - 5: intermediate - balanced
            - 7: full Allen forward relations - most detailed
            NOTE: Full relations documentation is in core/relation_table.py

        Raises
        ------
        ValueError
            If num_relations is not 2, 3, 5, or 7.
        """
        self.epsilon = normalize_time_param(epsilon)
        self.max_distance = normalize_time_param(max_distance)
        self.min_ver_supp = min_ver_supp
        self.num_relations = num_relations
        set_relation_table(num_relations)

    @log_execution
    def discover_patterns(
        self, entity_list, min_length=1, max_length=None, return_tree=False, return_tirps=False, inverse_mapping_path="data/inverse_symbol_map.json"
    ):
        """
        Discover all frequent TIRPs from entity_list and return a flat DataFrame summary (default).

        Parameters
        ----------
        entity_list : list
            List of entities. Each entity is a list of (start, end, symbol) tuples.
        min_length : int
            Minimum pattern length (k) to include.
        max_length : int, optional
            Maximum pattern length (k) to include. If None, no limit.
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

        # Phase 1 — sort each entity and build symbol→positions index.
        # pairwise_rels is intentionally deferred to Phase 2 (after singleton Karma)
        # so that we only compute pairs for frequent symbols, mirroring the C# lab's
        # _filteredEntities approach.  With ~1250 intervals/patient and max_distance=None
        # an upfront O(m²) precompute would materialise ~156M dict entries (≈12–15 GB)
        # before any frequency filtering has been applied.
        t_pre_start = time.perf_counter()
        precomputed = []
        for entity in entity_list:
            lexi = lexicographic_sorting(entity)
            symbol_to_positions = defaultdict(list)
            for pos, (_, _, sym) in enumerate(lexi):
                symbol_to_positions[sym].append(pos)
            symbol_items = tuple(symbol_to_positions.items())
            precomputed.append({"sorted": lexi, "symbol_index": symbol_to_positions,
                                 "symbol_items": symbol_items, "pairwise_rels": {}})
        t_pre_end = time.perf_counter()

        # Karma phase — runs in three internal phases (see Karma.run_karma docstring):
        #   A) singleton discovery
        #   B) filter precomputed to frequent symbols; compute pairwise_rels lazily
        #   C) k=2 TIRP construction from filtered pairwise_rels
        # Sub-timings are emitted at DEBUG level from inside run_karma.
        t_karma_start = time.perf_counter()
        karma = Karma(self.epsilon, self.max_distance, self.min_ver_supp, num_relations=self.num_relations)
        tree = karma.run_karma(entity_list, precomputed)
        t_karma_end = time.perf_counter()

        # Build the level-2 index from Karma's k=2 embeddings for O(1) last-pair
        # lookup during Lego extension. Cost is O(total k=2 embeddings) — negligible.
        level2_index = self._build_level2_index(tree)

        # Lego extension
        # Symbol_index is kept in precomputed so Lego can use it for fast interval lookup.
        t_lego_start = time.perf_counter()
        lego = Lego(tree, self.epsilon, self.max_distance, self.min_ver_supp, show_detail=True,
                    num_relations=self.num_relations, level2_index=level2_index)
        full_tree = lego.run_lego(tree, entity_list, precomputed, max_length=max_length)
        t_lego_end = time.perf_counter()

        # Flatten and filter
        t_flatten_start = time.perf_counter()
        all_tirps = full_tree.find_tree_nodes()
        filtered = [t for t in all_tirps if t.k >= min_length]
        if max_length is not None:
            filtered = [t for t in filtered if t.k <= max_length]

        # If caller doesn't need the tree, clear cached subtree lists to reduce memory.
        if not return_tree:
            stack = [full_tree]
            while stack:
                node = stack.pop()
                node._cached_subtree_nodes = None
                stack.extend(node.children)

        # Support set comparison for closed/super flags — O(P) grouping
        support_sets = [frozenset(t.entity_indices_supporting) for t in filtered]
        ks = [t.k for t in filtered]

        # One pass: compute min_k and max_k per unique support set
        group: dict = {}
        for fset, k in zip(support_sets, ks):
            if fset not in group:
                group[fset] = [k, k]   # [min_k, max_k]
            else:
                if k < group[fset][0]:
                    group[fset][0] = k
                if k > group[fset][1]:
                    group[fset][1] = k

        # A pattern is closed  iff no longer pattern shares its identical support set (k == max_k)
        # A pattern is super   iff it is closed AND a shorter pattern shares its support set,
        #                       i.e. it is the longest in its closed equivalence class AND
        #                       that class contains a proper sub-pattern.  (super ⊆ closed)
        is_closed = [group[fset][1] == k for fset, k in zip(support_sets, ks)]
        is_super  = [group[fset][0] < k and group[fset][1] == k
                     for fset, k in zip(support_sets, ks)]

        # Build DataFrame
        records = []
        for idx, tirp in enumerate(filtered):
            record = {
                "symbols": tuple(tirp.symbols),
                "relations": tuple(tirp.relations),
                "k": tirp.k,
                "vertical_support": tirp.vertical_support,
                "tirp_obj": tirp if return_tirps or return_tree else None, # Condition the return of the TIRP object on the flags to avoid unnecessary memory usage
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
            "discover_patterns timings (min): pre=%.4f karma=%.4f lego=%.4f flatten=%.4f total=%.4f",
            (t_pre_end     - t_pre_start)    / SECONDS_PER_MINUTE,
            (t_karma_end   - t_karma_start)  / SECONDS_PER_MINUTE,
            (t_lego_end    - t_lego_start)   / SECONDS_PER_MINUTE,
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
    def apply_patterns_to_entities(
        self,
        entity_list,
        patterns,
        patient_ids,
        mode: str = "tirp-count",              # "tirp-count" | "tpf-dist" | "tpf-duration"
        count_strategy: str = "unique_last",   # for horizontal support: "unique_last" or "all"
    ):
        """
        Build per-patient features from discovered TIRPs.

        Parameters
        ----------
        entity_list : list[list[(start, end, symbol)]]
            Entities aligned with patient_ids.
        patterns : list[TIRP] or DataFrame with column 'tirp_obj'
            Patterns to apply.
        patient_ids : list
            Parallel to entity_list.
        mode : {"tirp-count","tpf-dist","tpf-duration"}
            - "tirp-count": horizontal support per patient. By default uses 'unique_last':
            counts one occurrence per DISTINCT last-symbol index among valid embeddings.
            Example: for A…B…A…B…C, pattern A<B<C counts as 1 (not 3).
            Use count_strategy="all" to count every embedding if desired.
            - "tpf-dist": min-max normalize the horizontal support across patients (per pattern) to [0,1].
            - "tpf-duration": for each patient, sum the UNION of embedding spans
            (start(first symbol) to end(last symbol)), then min-max normalize across patients (per pattern).
        count_strategy : {"unique_last","all"}
            Strategy for horizontal support counting (see above).

        Returns
        -------
        dict
            patient_id -> {pattern_key: value}
            where pattern_key is a tuple (tuple(symbols), tuple(relations)).
        """
        # ---- Normalize patterns container ----
        if hasattr(patterns, "itertuples") or isinstance(patterns, pd.DataFrame):
            cols = set(patterns.columns)
            if "tirp_obj" in cols:
                # Reconstruct TIRPs from 'tirp_obj' column if present (preferred for direct DataFrame output from discover_patterns)
                tirp_objs = list(patterns["tirp_obj"])
                if any(t is not None for t in tirp_objs):
                    patterns_list = [t for t in tirp_objs if t is not None]
                elif {"symbols", "relations"}.issubset(cols):
                    patterns_list = []
                    for row in patterns.itertuples(index=False):
                        syms = row.symbols
                        rels = row.relations
                        if isinstance(syms, str):
                            syms = literal_eval(syms)  # e.g., "(1, 3, 2)" -> (1,3,2)
                        if isinstance(rels, str):
                            rels = literal_eval(rels)  # e.g., "('<','<')" -> ('<','<')
                        patterns_list.append(
                            TIRP(
                                epsilon=self.epsilon,
                                max_distance=self.max_distance,
                                min_ver_supp=self.min_ver_supp,
                                symbols=list(syms),
                                relations=list(rels),
                                k=len(syms),
                            )
                        )
                else:
                    raise ValueError("DataFrame must contain either 'tirp_obj' or both 'symbols' and 'relations'.")
            elif {"symbols", "relations"}.issubset(cols):
                # Reconstruct TIRPs from CSV-friendly columns
                patterns_list = []
                for row in patterns.itertuples(index=False):
                    syms = row.symbols
                    rels = row.relations
                    if isinstance(syms, str):
                        syms = literal_eval(syms)  # e.g., "(1, 3, 2)" -> (1,3,2)
                    if isinstance(rels, str):
                        rels = literal_eval(rels)  # e.g., "('<','<')" -> ('<','<')
                    patterns_list.append(
                        TIRP(
                            epsilon=self.epsilon,
                            max_distance=self.max_distance,
                            min_ver_supp=self.min_ver_supp,
                            symbols=list(syms),
                            relations=list(rels),
                            k=len(syms),
                        )
                    )
            else:
                raise ValueError("DataFrame must contain either 'tirp_obj' or both 'symbols' and 'relations'.")
        elif isinstance(patterns, (list, tuple)):
            patterns_list = list(patterns)
        else:
            raise ValueError("Unsupported patterns container")

        # ---- Precompute per-entity views (sorted intervals + symbol list + index) ----
        # Reuse your lexicographic policy for consistent indexing.
        # symbol_index maps each symbol -> sorted list of positions; used for O(log n) bisect jumps
        # instead of a full O(m) linear scan when enumerating subsequence candidates.
        precomp = []
        for ent in entity_list:
            lexi = lexicographic_sorting(ent)
            ti = [(s, e) for s, e, _ in lexi]        # list of (start, end) pairs
            syms = [sym for _, _, sym in lexi]       # parallel list of symbols
            sym_idx: dict = {}
            for pos, sym in enumerate(syms):
                sym_idx.setdefault(sym, []).append(pos)
            precomp.append((ti, syms, sym_idx))

        # ---- Helpers ----
        def _valid_embeddings_in_entity(tirp, ti, syms, sym_idx):
            """Return list of index-tuples (embedding positions) that satisfy tirp.symbols & all relations.

            Uses bisect jumps through symbol_index to enumerate candidates without scanning the full
            symbol list linearly.  For an entity with m events and a pattern of length p, worst-case
            is still O(m^p) but average-case is dramatically smaller when symbols are infrequent.
            """
            pat_syms = tirp.symbols
            n_pat = len(pat_syms)
            if n_pat > len(syms):
                return []
            # Early-exit: if any required symbol is missing, skip entirely
            for sym in pat_syms:
                if sym not in sym_idx:
                    return []

            # Iterative DFS using an explicit stack to avoid Python recursion overhead.
            # Each stack frame: (p_pos, e_min, partial_tuple)
            out = []
            relations = tirp.relations
            epsilon = self.epsilon
            max_dist = self.max_distance
            stack = [(0, 0, ())]
            while stack:
                p_pos, e_min, partial = stack.pop()
                sym = pat_syms[p_pos]
                positions = sym_idx[sym]  # guaranteed present by early-exit above
                start_i = bisect.bisect_left(positions, e_min)
                for k in range(start_i, len(positions)):
                    pos = positions[k]
                    candidate = partial + (pos,)
                    if p_pos == n_pat - 1:
                        # Full candidate tuple: verify all pairwise relations
                        ok = True
                        rel_idx = 0
                        for col, ent_idx in enumerate(candidate[1:]):
                            for row in range(col + 1):
                                i1 = candidate[row]
                                i2 = ent_idx
                                expected = relations[rel_idx]
                                actual = temporal_relations(ti[i1], ti[i2], epsilon, max_dist)
                                if expected != actual:
                                    ok = False
                                    break
                                rel_idx += 1
                            if not ok:
                                break
                        if ok:
                            out.append(candidate)
                    else:
                        stack.append((p_pos + 1, pos + 1, candidate))
            return out

        def _horizontal_support_from_embeddings(embeddings):
            """Count embeddings per strategy."""
            if not embeddings:
                return 0
            if count_strategy == "all":
                return len(embeddings)
            elif count_strategy == "unique_last":
                # Count one per distinct last index → collapses A…B…A…B…C to 1 for A<B<C
                return len({t[-1] for t in embeddings})
            else:
                raise ValueError("count_strategy must be 'unique_last' or 'all'.")

        def _union_span_from_embeddings(embeddings, ti):
            """Compute total union duration over [start(first), end(last)] for each embedding."""
            if not embeddings:
                return 0
            # Build span intervals for each embedding
            ivs = []
            for tup in embeddings:
                start_first = ti[tup[0]][0]
                end_last = ti[tup[-1]][1]
                ivs.append((start_first, end_last))
            # Merge overlapping intervals (classic sweep)
            ivs.sort(key=lambda x: (x[0], x[1]))
            merged = []
            for s, e in ivs:
                if not merged or s > merged[-1][1]:
                    merged.append([s, e])
                else:
                    if e > merged[-1][1]:
                        merged[-1][1] = e
            total = 0
            for s, e in merged:
                total += (e - s)
            return int(total)

        # ---- Pass 1: compute raw values per patient & pattern ----
        # Key: (tuple(symbols), tuple(relations)) — a stable tuple that is cheaper than repr(tirp)
        # (no float formatting) and correct (repr used to embed vertical_support, making two
        # structurally identical patterns from different runs collide or diverge incorrectly).
        need_norm = mode in ("tpf-dist", "tpf-duration")
        n_patients = len(patient_ids)
        values = {pid: defaultdict(float) for pid in patient_ids}
        maxs: dict = {}        # per-pattern max value seen (over hits only)
        mins: dict = {}        # per-pattern min value seen (over hits only)
        hit_counts: dict = {}  # per-pattern number of patients with at least one embedding
        for tirp in tqdm(patterns_list, desc="Applying patterns to entities"):
            key = (tuple(tirp.symbols), tuple(tirp.relations))
            for eid, (ti, syms, sym_idx) in enumerate(precomp):
                emb = _valid_embeddings_in_entity(tirp, ti, syms, sym_idx)
                if not emb:
                    continue
                pid = patient_ids[eid]
                if mode == "tirp-count" or mode == "tpf-dist":
                    val = _horizontal_support_from_embeddings(emb)
                elif mode == "tpf-duration":
                    val = _union_span_from_embeddings(emb, ti)
                else:
                    raise ValueError("mode must be one of: 'tirp-count', 'tpf-dist', 'tpf-duration'.")
                values[pid][key] = val
                if need_norm:
                    if val > maxs.get(key, 0):
                        maxs[key] = val
                    if key not in mins or val < mins[key]:
                        mins[key] = val
                    hit_counts[key] = hit_counts.get(key, 0) + 1

        # ---- Pass 2: normalization for cohort-based modes ----
        # The effective min per pattern is:
        #   - 0  if any patient is missing (they contribute 0 implicitly), which is the common case.
        #   - mins[key]  if every patient has a hit (hit_count == n_patients).
        # When effective_min == max (no variation across the cohort), result is 0 by convention.
        # This is O(#hits) — no per-pattern full-cohort scan needed.
        if need_norm:
            for pid in patient_ids:
                if not values[pid]:
                    continue
                for pat in list(values[pid].keys()):
                    hi = maxs.get(pat, 0)
                    lo = mins[pat] if hit_counts.get(pat, 0) == n_patients else 0
                    if hi > lo:
                        values[pid][pat] = (values[pid][pat] - lo) / (hi - lo)
                    else:
                        values[pid][pat] = 0.0

        return values

    @staticmethod
    def _build_level2_index(tree):
        """
        Build a level-2 index from the frequent k=2 TIRPs in the Karma tree.

        Structure: (sym_A, sym_B, rel) -> {eid -> {pos_A -> [pos_B, ...]}}

        All (pos_A, pos_B) pairs are already CSAC-filtered during Karma, so the
        caller can skip the last-pair relation check and CSAC check when using
        this index. That invariant is what makes the optimisation correct without
        any re-verification.

        Cost is O(total k=2 embeddings) — negligible relative to Karma.

        Parameters
        ----------
        tree : TreeNode
            Root of the Karma-produced pattern tree.

        Returns
        -------
        dict
            Nested mapping (sym_A, sym_B, rel) -> {eid -> {pos_A -> [pos_B, ...]}},
            covering all frequent k=2 TIRPs and all their entity embeddings.
        """
        level2_index = {}
        for singleton_node in tree.children:
            for pair_node in singleton_node.children:
                tirp = pair_node.data
                if not isinstance(tirp, TIRP) or tirp.k != 2 or not tirp.embeddings_map:
                    continue
                sym_A, sym_B = tirp.symbols
                rel = tirp.relations[0]
                key = (sym_A, sym_B, rel)
                eid_dict = {}
                for eid, embs in tirp.embeddings_map.items():
                    pos_dict = {}
                    for pos_A, pos_B in embs:
                        pos_dict.setdefault(pos_A, []).append(pos_B)
                    eid_dict[eid] = pos_dict
                level2_index[key] = eid_dict
        return level2_index


class Karma(KarmaLego):
    def __init__(self, epsilon, max_distance, min_ver_supp, num_relations=7):
        super().__init__(epsilon, max_distance, min_ver_supp, num_relations=num_relations)

    def run_karma(self, entity_list, precomputed):
        """
        Karma phase: discover frequent singletons and all length-2 TIRPs.

        Internally runs in three sequential phases to keep memory usage bounded on
        large datasets (e.g. ≥1 000 intervals/entity with ``max_distance=None``).

        **Phase A — Singleton discovery**

        Iterates all distinct symbols across ``entity_list``, computes vertical support
        from each entity's ``symbol_index``, and builds frequent singleton TIRPs.
        Produces the ``frequent_symbols`` whitelist and a ``symbol → TreeNode`` map.
        At this stage ``precomputed[*]["pairwise_rels"]`` is empty; the pair relation
        dict is intentionally deferred until after frequency filtering.

        **Phase B — Lazy pairwise precomputation**

        Filters each entity's 'symbol_index' (and 'symbol_items') to the
        frequent symbols found in Phase A, then computes 'pairwise_rels'
        by iterating only the sorted positions of surviving-symbol intervals.

        Without this split, an upfront O(m²) scan over all symbols would
        materialise the full pairwise relation dict before any frequency filtering.
        On large datasets this reaches ~156 M entries (~12-15 GB).
        Restricting both loops to frequent-symbol positions reduces that by one to two
        orders of magnitude. The 'max_distance' early-break is still effective
        because the positions are lexicographically sorted.

        **Phase C — k=2 TIRP construction**

        Consumes the now-populated ``pairwise_rels`` to build all frequent length-2
        TIRPs (with CSAC filtering) and attaches them as children of the corresponding
        singleton nodes.

        Parameters
        ----------
        entity_list : list of list of (start, end, symbol)
            Raw entity sequences, one list per entity.
        precomputed : list of dict
            Per-entity dicts with keys:

            'sorted'        - lexicographically sorted intervals.
            'symbol_index'  - symbol → list[int] positions; mutated in-place
                              during Phase B to retain frequent symbols only.
            'symbol_items'  - tuple view of symbol_index; rebuilt in Phase B.
            'pairwise_rels' - dict (i, j) → rel; empty on entry, populated by
                              Phase B, and consumed by Phase C.

        Returns
        -------
        TreeNode
            Root of the Karma tree. Singleton TIRPs are direct children; each
            carries its frequent k=2 TIRPs as children. 'precomputed' is
            mutated in-place as a side effect (symbol_index filtered, pairwise_rels
            populated).
        """
        _t0 = time.perf_counter()

        tree = TreeNode("root")
        frequent_symbols = set()
        symbol_to_singleton_node = {}

        symbol_set = {sym for ent in entity_list for _, _, sym in ent}

        # ------------------------------------------------------------------ #
        # Phase A: singleton discovery                                        #
        # ------------------------------------------------------------------ #
        with tqdm(total=len(symbol_set), desc="Karma (singletons)") as bar:
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
                        # Build embeddings_map: eid -> [(pos,)]
                        emb = defaultdict(list)
                        for pos, eid in supporting_pairs:
                            emb[eid].append((pos,))
                        tirp_single.embeddings_map = {eid: sorted(set(tups)) for eid, tups in emb.items()}
                        tirp_single.entity_indices_supporting = entity_indices_supporting
                        tirp_single.indices_of_last_symbol_in_entities = indices_of_last_symbol_in_entities
                        tirp_single.vertical_support = vertical_support

                        node = TreeNode(tirp_single)
                        tree.add_child(node)
                        symbol_to_singleton_node[sym] = node
                        frequent_symbols.add(sym)
                bar.update(1)

        _t_a = time.perf_counter()

        # ------------------------------------------------------------------ #
        # Phase B: filter precomputed to frequent symbols; compute pairwise   #
        #          relations for frequent-symbol interval pairs only.          #
        # ------------------------------------------------------------------ #
        # By restricting both loops to surviving-symbol positions the dict
        # stays proportional to |frequent_symbols|² × n_entities rather than
        # the full O(m²) per entity.  The early-break exploits lexicographic
        # sort: once start_j - end_i > max_distance, all subsequent j are also
        # out of range and the inner loop terminates immediately.
        for entry in precomputed:
            lexi = entry["sorted"]
            entry["symbol_index"] = {
                sym: pos for sym, pos in entry["symbol_index"].items()
                if sym in frequent_symbols
            }
            entry["symbol_items"] = tuple(entry["symbol_index"].items())

            freq_positions = sorted(
                pos for positions in entry["symbol_index"].values() for pos in positions
            )

            pairwise_rels: dict = {}
            for ii, i in enumerate(freq_positions):
                end_i = lexi[i][1]
                for j in freq_positions[ii + 1:]:
                    start_j = lexi[j][0]
                    if self.max_distance is not None and start_j - end_i > self.max_distance:
                        break
                    rel = temporal_relations(lexi[i][:2], lexi[j][:2], self.epsilon, self.max_distance)
                    if rel is not None:
                        pairwise_rels[(i, j)] = rel
            entry["pairwise_rels"] = pairwise_rels

        _t_b = time.perf_counter()

        # ------------------------------------------------------------------ #
        # Phase C: build frequent k=2 TIRPs; attach to the singleton tree.   #
        # ------------------------------------------------------------------ #
        self._build_pairs(entity_list, precomputed, tree, symbol_to_singleton_node)

        _t_c = time.perf_counter()
        _SPM = 60.0
        logger.debug(
            "Karma phase timings (min): singletons=%.4f pairwise_precompute=%.4f pairs=%.4f",
            (_t_a - _t0) / _SPM,
            (_t_b - _t_a) / _SPM,
            (_t_c - _t_b) / _SPM,
        )

        return tree

    def _build_pairs(self, entity_list, precomputed, tree, symbol_to_singleton_node):
        """
        Build frequent k=2 TIRPs and attach them to the Karma tree (Phase C).

        Called internally by 'run_karma' after singleton discovery (Phase A)
        and pairwise relation precomputation (Phase B). Not intended to be
        called directly.

        'precomputed[*]["pairwise_rels"]' must already be populated with
        frequent-symbol pairs only (Phase B postcondition); the symbol-whitelist
        guard inside the loop is a safety net and cannot fire under normal use.

        Parameters
        ----------
        entity_list : list
            Full entity list — used as the vertical-support denominator.
        precomputed : list of dict
            Per-entity dicts; 'pairwise_rels' must already be populated.
        tree : TreeNode
            Root node returned by Phase A; k=2 children are attached here.
        symbol_to_singleton_node : dict
            Map from symbol to TreeNode, built during Phase A.
        """
        total_pairs = sum(len(e["pairwise_rels"]) for e in precomputed)
        sac_rels = get_sac_relations()
        tirp_dict = {}
        frequent_symbols = set(symbol_to_singleton_node.keys())

        with tqdm(total=total_pairs, desc="Karma (pairs)") as karma_bar:
            for eid, entry in enumerate(precomputed):
                ordered = entry["sorted"]
                for (i, j), rel in entry["pairwise_rels"].items():
                    symbol_1 = ordered[i][2]
                    symbol_2 = ordered[j][2]
                    # pairwise_rels was built from frequent-symbol positions only (Phase B),
                    # so this guard fires only if precomputed was constructed externally.
                    if symbol_1 not in frequent_symbols or symbol_2 not in frequent_symbols:
                        continue

                    # CSAC adjacency constraint at the k=2 level: for ordering relations,
                    # verify no other occurrence of symbol_1 exists strictly between i and j.
                    if rel in sac_rels:
                        sym1_positions = entry["symbol_index"].get(symbol_1, [])
                        lo_sac = bisect.bisect_right(sym1_positions, i)
                        if lo_sac < len(sym1_positions) and sym1_positions[lo_sac] < j:
                            continue  # SAC violation

                    signature = ((symbol_1, symbol_2), (rel,))
                    if signature not in tirp_dict:
                        tirp = TIRP(
                            epsilon=self.epsilon,
                            max_distance=self.max_distance,
                            min_ver_supp=self.min_ver_supp,
                            symbols=[symbol_1, symbol_2],
                            relations=[rel],
                            k=2,
                        )
                        tirp.entity_indices_supporting = [eid]
                        tirp.indices_of_last_symbol_in_entities = [j]
                        tirp.embeddings_map = {eid: [(i, j)]}
                        tirp_dict[signature] = tirp
                    else:
                        existing = tirp_dict[signature]
                        existing.entity_indices_supporting.append(eid)
                        existing.indices_of_last_symbol_in_entities.append(j)
                        existing.embeddings_map.setdefault(eid, []).append((i, j))
                    karma_bar.update(1)

        for tirp in tirp_dict.values():
            if tirp.entity_indices_supporting:
                unique = set(zip(tirp.indices_of_last_symbol_in_entities, tirp.entity_indices_supporting))
                if unique:
                    sym_idxs, ent_idxs = zip(*unique)
                    tirp.indices_of_last_symbol_in_entities = list(sym_idxs)
                    tirp.entity_indices_supporting = list(ent_idxs)
            if tirp.embeddings_map:
                tirp.embeddings_map = {eid: sorted(set(tups)) for eid, tups in tirp.embeddings_map.items()}
            tirp.vertical_support = (
                len(set(tirp.entity_indices_supporting)) / len(entity_list) if entity_list else 0.0
            )
            if tirp.vertical_support >= self.min_ver_supp:
                parent_node = symbol_to_singleton_node.get(tirp.symbols[0])
                if parent_node is not None:
                    parent_node.add_child(TreeNode(tirp))


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
    def __init__(self, tree, epsilon, max_distance, min_ver_supp, show_detail, num_relations=7,
                 level2_index=None):
        self.tree = tree
        super().__init__(epsilon, max_distance, min_ver_supp, num_relations=num_relations)
        self.show_detail = show_detail  # whether to keep per-extension verbosity
        # Level-2 index: (sym_A, sym_B, rel) → {eid → {pos_A → [pos_B, ...]}}.
        # Eliminates the last relation check and last CSAC check in the guided
        # extension path, replacing a bisect scan with an O(1) grouped lookup.
        self.level2_index = level2_index

    def run_lego(self, node, entity_list, precomputed, max_length):
        """
        Extend base patterns recursively to higher-order TIRPs using depth-first search.

        DFS is preferred over BFS because each branch's embeddings_map is freed as soon
        as its subtree is exhausted, keeping peak memory proportional to the depth of one
        active path rather than the width of an entire BFS level.  This also reduces
        Python GC pressure on large runs.

        Parameters
        ----------
        node :
            Root TreeNode to start extension from.
        entity_list :
            List of entities for support computation.
        precomputed :
            List of dicts per entity with keys 'sorted' and 'symbol_index'.
        max_length : int or None
            Maximum pattern length (k) to extend to. None means no limit.

        Returns
        -------
        TreeNode
            The same tree with extended TIRPs grafted in.
        """
        with tqdm(desc="Lego phase (nodes expanded)", unit=" node/s") as bar:
            def _dfs(current):
                if not isinstance(current.data, TIRP):
                    # Tree root (non-TIRP): descend into singleton children.
                    for child in current.children:
                        _dfs(child)
                    return

                tirp = current.data
                bar.update(1)

                if max_length is not None and tirp.k >= max_length:
                    tirp.embeddings_map = None
                    return

                if tirp.k == 1:
                    # Singletons are not extended here — Karma already built all k=2 pairs.
                    tirp.embeddings_map = None
                    for child in current.children:
                        _dfs(child)
                else:
                    # Generate candidate extensions and filter by support.
                    extensions = self.all_extensions(entity_list, tirp, precomputed)
                    iterator = (tqdm(extensions, desc=f"Extending TIRP k={tirp.k}", leave=False)
                                if self.show_detail else extensions)
                    ok = [ext for ext in iterator
                          if ext.is_above_vertical_support(entity_list, precomputed=precomputed,
                                                           level2_index=self.level2_index)]

                    # All children have consumed parent embeddings via is_above_vertical_support;
                    # free the parent's embeddings_map before descending into children.
                    tirp.embeddings_map = None

                    for ext in ok:
                        child = TreeNode(ext)
                        current.add_child(child)
                        _dfs(child)

            _dfs(node)
        return node

    def all_extensions(self, entity_list, tirp, precomputed):
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
        precomputed:   
            List of dicts per entity with keys 'sorted' (lexicographically sorted intervals).

        Returns
        -------
        list[TIRP]
            New candidate TIRPs of length k+1 (not yet filtered by support).
        """
        curr_num_of_symbols = len(tirp.symbols)
        
        # Optimization: Use a dict to deduplicate candidates by signature immediately
        # Key: (new_symbol, tuple(new_relations)) -> Value: TIRP object
        candidates = {}

        # Prefer embedding-aware enumeration for CSAC
        if not getattr(tirp, "embeddings_map", None):
            raise RuntimeError("Lego extension requires embeddings_map for CSAC; legacy path disabled.")

        # Lazy generator — avoids materialising the full list of (sym_index, ent_index, tup)
        # triples, which can be large when a pattern has high support across many entities.
        sources = (
            (tup[-1], ent_index, tup)
            for ent_index, tuples in tirp.embeddings_map.items()
            for tup in tuples
        )

        # Within a single all_extensions call, tirp.relations / curr_rel_index / decrement_index
        # are all fixed — only rel_last_new varies. Cache the path-tree result keyed on
        # rel_last_new so repeated values (common when many source pairs share the same
        # last relation) pay the backtracking cost only once. At most num_relations entries.
        _base_rel_index = len(tirp.relations) - 1
        _base_dec_index = curr_num_of_symbols - 1
        _paths_cache: dict = {}

        # Level-2 upper-bound filter: for each (new_symbol, rel_last_new) pair encountered,
        # verify that enough parent-supporting entities have this k=2 sub-pattern in the index.
        # If the index has no entry at all → Apriori violation, guaranteed infrequent.
        # If the entity overlap count is below threshold → structurally impossible to meet MVS.
        # Both cases are cached after first evaluation so the check is O(1) on repeat encounters.
        sym_A = tirp.symbols[-1]
        parent_support_set = set(tirp.entity_indices_supporting)
        n_entities = len(entity_list)
        min_support_count = self.min_ver_supp * n_entities
        _l2_pruned: set  = set()   # (new_symbol, rel_last_new) pairs confirmed insufficient
        _l2_checked: set = set()   # (new_symbol, rel_last_new) pairs already evaluated

        for sym_index, ent_index, parent_tuple in sources:
            # Use precomputed sorted entity to avoid re-sorting
            if precomputed is not None:
                lexi_entity = precomputed[ent_index]["sorted"]
            else:
                lexi_entity = lexicographic_sorting(entity_list[ent_index])

            if curr_num_of_symbols >= len(lexi_entity):
                continue

            ti_last = lexi_entity[sym_index][:2]
            end_last = ti_last[1]
            entity_symbol_index = precomputed[ent_index]["symbol_index"]
            pairwise_rels = precomputed[ent_index].get("pairwise_rels", {})
            # Use the precomputed tuple of items to avoid rebuilding the dict view on
            # every iteration when the same ent_index appears across multiple sources.
            symbol_items = precomputed[ent_index].get("symbol_items") or entity_symbol_index.items()

            # Instead of scanning all lexi_entity[sym_index+1:] (O(entity_length)),
            # iterate per symbol and jump to positions after sym_index via bisect (O(log k + occurrences)).
            # Since lexi_entity is sorted by start time, we also get an early-exit when
            # max_distance is exceeded for the remaining positions of a given symbol.
            for new_symbol, positions in symbol_items:
                lo = bisect.bisect_right(positions, sym_index)
                for after_index in positions[lo:]:
                    new_ti = lexi_entity[after_index][:2]
                    # Early exit: positions are sorted by start time, so once the gap exceeds
                    # max_distance every subsequent position is also out of range.
                    if self.max_distance is not None and new_ti[0] - end_last > self.max_distance:
                        break
                    rel_last_new = pairwise_rels.get((sym_index, after_index))
                    if rel_last_new is None:
                        continue

                    # Upper-bound check (cached per (new_symbol, rel_last_new) pair).
                    # On first encounter: verify the level-2 index has this k=2 sub-pattern
                    # AND that enough parent-supporting entities carry it to meet MVS.
                    # Skip all path/TIRP work for this combination if the bound fails.
                    l2_pair = (new_symbol, rel_last_new)
                    if l2_pair in _l2_pruned:
                        continue
                    if l2_pair not in _l2_checked:
                        _l2_checked.add(l2_pair)
                        l2_eid_map = self.level2_index.get((sym_A, new_symbol, rel_last_new))
                        if l2_eid_map is None or (
                            sum(1 for eid in parent_support_set if eid in l2_eid_map)
                            < min_support_count
                        ):
                            _l2_pruned.add(l2_pair)
                            continue

                    # get predecessor relation sequences (excluding the new-last relation)
                    if rel_last_new not in _paths_cache:
                        _paths_cache[rel_last_new] = find_all_possible_extensions(
                            [], [], rel_last_new, _base_rel_index, _base_dec_index, tirp.relations
                        )
                    all_paths = _paths_cache[rel_last_new]

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
                        signature = (new_symbol, tuple(new_relations))  # Optimization: Check signature
                        if signature not in candidates:
                            child = TIRP(
                                epsilon=self.epsilon,
                                max_distance=self.max_distance,
                                min_ver_supp=self.min_ver_supp,
                                symbols=[*tirp.symbols, new_symbol],
                                relations=[*tirp.relations, *new_relations],
                                k=tirp.k + 1,
                            )
                            # CSAC propagation
                            child.parent_entity_indices_supporting = list(tirp.entity_indices_supporting)
                            child.parent_embeddings_map = tirp.embeddings_map
                            child.entity_indices_supporting = []
                            child.indices_of_last_symbol_in_entities = []
                            candidates[signature] = child
        return list(candidates.values())