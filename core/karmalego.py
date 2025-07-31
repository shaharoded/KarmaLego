import logging
from functools import wraps
from copy import deepcopy
from collections import defaultdict

from tqdm import tqdm

from core.utils import (
    temporal_relations,
    lexicographic_sorting,
    check_symbols_lexicographically,
    find_all_possible_extensions,
    count_embeddings_in_single_entity,
    vertical_support_symbol
)

# logging decorator
logger = logging.getLogger(__name__)
if not logger.hasHandlers():
    logging.basicConfig(level=logging.INFO)


def log_execution(func):
    from functools import wraps

    @wraps(func)
    def wrapper(*args, **kwargs):
        logger.info(f"Starting {func.__name__}")
        result = func(*args, **kwargs)
        logger.info(f"Finished {func.__name__}")
        return result

    return wrapper


class SchemaValidationError(Exception):
    pass


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
        self.symbols.append(new_symbol)
        self.relations.extend(new_relations)
        if not self.check_size():
            raise AttributeError("Extension of TIRP is wrong!")
        self._hash_cache = None  # invalidate cached hash

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

    def is_above_vertical_support(self, entity_list):
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
    def __init__(self, epsilon, max_distance, min_ver_supp):
        self.epsilon = epsilon
        self.max_distance = max_distance
        self.min_ver_supp = min_ver_supp

    @log_execution
    def calculate_frequent_patterns(self, entity_list, min_length=2):
        """
        Discover frequent TIRPs from the entity_list using Karma→Lego logic.

        Parameters
        ----------
        entity_list : list
            List of entities. Each entity is a list of (start, end, symbol) tuples.
        min_length : int
            Minimum pattern length to keep (1=include singletons).

        Returns
        -------
        root_tree : TreeNode
            Root of the pattern tree with frequent TIRPs attached.
        flat_patterns : list[TIRP]
            Flattened list of all frequent TIRPs discovered (meeting min_ver_supp and min_length).
        """
        # Karma phase: frequent singletons and length-2 seeds
        karma = Karma(self.epsilon, self.max_distance, self.min_ver_supp)
        tree = karma.run(entity_list)

        # Lego phase: recursive extension
        lego = Lego(tree, self.epsilon, self.max_distance, self.min_ver_supp)
        full_tree = lego.run_lego(tree, entity_list)

        # flatten results
        all_tirps = full_tree.find_tree_nodes()
        filtered = [t for t in all_tirps if t.k >= min_length]
        return full_tree, filtered
    

    @log_execution
    def apply_patterns_to_entities(self, entity_list, patterns, patient_ids, tpp=False):
        """
        Given a list of frequent TIRPs, produce per-patient pattern counts or normalized distributions.

        Parameters
        ----------
        entity_list : list
            List of entities (one per patient), each a list of (start,end,symbol).
        patterns : list of TIRP
            Frequent patterns to apply.
        patient_ids : list
            Parallel list of identifiers for each entity in entity_list (e.g., PatientID).
        tpp : bool
            If True, return normalized distribution per patient (pattern count / total counts for that patient).
            If False, raw counts.

        Returns
        -------
        result : dict
            Mapping patient_id -> dict(pattern_id -> count or normalized value)
        """
        # pattern_id can be repr(tirp) for human-readable key
        patient_pattern_counts = {pid: defaultdict(int) for pid in patient_ids}

        # precompute embeddings per pattern per patient to avoid recomputing for support
        for tirp in tqdm(patterns, desc="Applying patterns to entities"):
            pattern_key = repr(tirp)
            for idx, entity in enumerate(entity_list):
                count = count_embeddings_in_single_entity(tirp, entity)
                if count > 0:
                    patient_pattern_counts[patient_ids[idx]][pattern_key] = count

        # normalize if tpp
        if tpp:
            for pid in patient_pattern_counts:
                counts = patient_pattern_counts[pid]
                total = sum(counts.values())
                if total > 0:
                    for k in list(counts.keys()):
                        counts[k] = counts[k] / total  # normalized

        return patient_pattern_counts
    

class Karma(KarmaLego):
    def __init__(self, epsilon, max_distance, min_ver_supp):
        super().__init__(epsilon, max_distance, min_ver_supp)

    def run(self, entity_list):
        """
        First phase: discover frequent singletons and length-2 TIRPs (Karma).
        Returns a TreeNode with k=1 symbols as children and frequent k=2 TIRPs attached.
        """
        # extract all symbols
        all_symbols = set()
        for entity in entity_list:
            for _, _, sym in entity:
                all_symbols.add(sym)
        all_symbols = list(all_symbols)

        # frequent singletons
        frequent_symbols = []
        tree = TreeNode("root")
        for symbol in all_symbols:
            sup = vertical_support_symbol(entity_list, symbol)
            if sup >= self.min_ver_supp:
                node = TreeNode(symbol)
                tree.add_child(node)
                frequent_symbols.append(symbol)

        # generate all pair TIRPs (k=2)
        tirp_dict = {}  # dedupe by structural identity
        for entity_index, entity in enumerate(entity_list):
            ordered_ti = lexicographic_sorting(entity)
            for i in range(len(ordered_ti)):
                for j in range(i + 1, len(ordered_ti)):
                    start_1, end_1, symbol_1 = ordered_ti[i]
                    start_2, end_2, symbol_2 = ordered_ti[j]
                    if symbol_1 not in frequent_symbols or symbol_2 not in frequent_symbols:
                        continue
                    rel = temporal_relations((start_1, end_1), (start_2, end_2), self.epsilon, self.max_distance)
                    if rel is None:
                        continue
                    tirp = TIRP(
                        epsilon=self.epsilon,
                        max_distance=self.max_distance,
                        min_ver_supp=self.min_ver_supp,
                        symbols=[symbol_1, symbol_2],
                        relations=[rel],
                        k=2,
                    )
                    # accumulate support info manually (will recompute vertical_support later)
                    key = hash(tirp)
                    if key not in tirp_dict:
                        # initialize support tracking
                        tirp.entity_indices_supporting = [entity_index]
                        tirp.indices_of_last_symbol_in_entities = [j]
                        tirp_dict[key] = tirp
                    else:
                        existing = tirp_dict[key]
                        existing.entity_indices_supporting.append(entity_index)
                        existing.indices_of_last_symbol_in_entities.append(j)

        # finalize length-2 TIRPs: dedupe embedded entity indices and attach to tree
        for tirp in tirp_dict.values():
            # dedupe pairs
            if tirp.entity_indices_supporting:
                unique = set(zip(tirp.indices_of_last_symbol_in_entities, tirp.entity_indices_supporting))
                if unique:
                    sym_idxs, ent_idxs = zip(*unique)
                    tirp.indices_of_last_symbol_in_entities = list(sym_idxs)
                    tirp.entity_indices_supporting = list(ent_idxs)
            # compute vertical support
            tirp.vertical_support = len(set(tirp.entity_indices_supporting)) / len(entity_list) if entity_list else 0.0
            if tirp.vertical_support >= self.min_ver_supp:
                # attach under corresponding singleton parent (first symbol)
                for child in tree.children:
                    if child.data == tirp.symbols[0]:
                        child.add_child(TreeNode(tirp))
                        break
        return tree


class Lego(KarmaLego):
    def __init__(self, tree, epsilon, max_distance, min_ver_supp):
        self.tree = tree
        super().__init__(epsilon, max_distance, min_ver_supp)

    def run_lego(self, node, entity_list):
        """
        Recursively extend TIRPs in the tree (Lego phase).
        """
        if isinstance(node.data, TIRP):
            extensions = list(set(self.all_extensions(entity_list, node.data)))
            # filter by support
            ok = []
            for ext in tqdm(extensions, desc=f"Extending TIRP k={ext.k}"):
                if ext.is_above_vertical_support(entity_list):
                    ok.append(ext)
            for ext in ok:
                node.add_child(TreeNode(ext))
        for child in node.children:
            self.run_lego(child, entity_list)
        return node

    def all_extensions(self, entity_list, tirp):
        """
        Given a base TIRP, produce all one-symbol extensions (candidate next-level TIRPs).
        """
        curr_num_of_symbols = len(tirp.symbols)
        all_possible = []
        for sym_index, ent_index in zip(tirp.indices_of_last_symbol_in_entities, tirp.entity_indices_supporting):
            lexi_entity = lexicographic_sorting(entity_list[ent_index])
            if curr_num_of_symbols >= len(lexi_entity):
                continue
            for after_sym in lexi_entity[sym_index + 1 :]:
                *new_ti, new_symbol = after_sym
                rel_last_new = temporal_relations(lexi_entity[sym_index][:2], new_ti, self.epsilon, self.max_distance)
                if rel_last_new is None:
                    continue
                curr_rel_index = len(tirp.relations) - 1
                decrement_index = curr_num_of_symbols - 1
                all_paths = find_all_possible_extensions([], [], rel_last_new, curr_rel_index, decrement_index, tirp.relations)
                for path in all_paths:
                    new_relations = [rel_last_new, *path]
                    new_relations.reverse()
                    tirp_copy = deepcopy(tirp)
                    tirp_copy.extend(new_symbol, new_relations)
                    tirp_copy.k = tirp.k + 1
                    tirp_copy.vertical_support = None
                    tirp_copy.parent_entity_indices_supporting = tirp.entity_indices_supporting
                    tirp_copy.entity_indices_supporting = []
                    tirp_copy.indices_of_last_symbol_in_entities = []
                    all_possible.append(tirp_copy)
        return all_possible

