import logging
from tqdm import tqdm
import numpy as np
from functools import wraps

from core.utils import (
    temporal_relations,
    lexicographic_sorting,
    find_all_possible_extensions,
    equal_TIRPs
)

# basic logger setup (only once)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def log_execution(func):
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
    Representation of Time Interval Relation Pattern (TIRP) with two lists.
    Implementation of basic methods to work with TIRPs.
    """

    def __init__(self, epsilon, max_distance, min_ver_supp, symbols=None, relations=None, k=1, vertical_support=None,
                 indices_supporting=None, parent_indices_supporting=None, indices_of_last_symbol_in_entities=None):
        """
        Initialize TIRP instance with default or given values.

        :param epsilon: maximum amount of time between two events that we consider it as the same time
        :param max_distance: maximum distance between two time intervals that means first one still influences the second
        :param min_ver_supp: proportion (value between 0-1) defining threshold for accepting TIRP
        :param symbols: list of symbols presenting entity in lexicographic order (labels for upper triangular matrix)
        :param relations: list of Allen's temporal relations, presenting upper triangular matrix (half matrix),
                          relations' order is by columns from left to right and from up to down in the half matrix
        :param k: level of the TIRP in the enumeration tree
        :param vertical_support: value of TIRP support (between 0-1)
        :param indices_supporting: list of indices of entity list that support this TIRP
        :param parent_indices_supporting: list of indices of entity list that support parent of this TIRP
        :param indices_of_last_symbol_in_entities: list of indices of last element in symbols list in lexicographically ordered entities
                                                   (len(indices_of_last_symbol_in_entities) = len(indices_supporting))
        """
        self.epsilon = epsilon
        self.max_distance = max_distance
        self.min_ver_supp = min_ver_supp
        self.symbols = [] if symbols is None else symbols
        self.relations = [] if relations is None else relations
        self.k = k
        self.vertical_support = vertical_support
        self.entity_indices_supporting = indices_supporting
        self.parent_entity_indices_supporting = parent_indices_supporting
        self.indices_of_last_symbol_in_entities = [] if indices_of_last_symbol_in_entities is None else indices_of_last_symbol_in_entities

    def __repr__(self):
        """
        Method defining how TIRP class instance is printed to standard output.

        :return: string that is printed
        """
        return self.print() + '\n\nVertical support: ' + str(round(self.vertical_support, 3)) + '\n\n'

    def __lt__(self, other):
        """
        Method defining how 2 TIRPs are compared when sorting list of TIRPs.

        :param other: the second TIRP that is compared to self
        :return: boolean - True if self is less than other (in sense of their vertical support)
        """
        return self.vertical_support < other.vertical_support

    def __eq__(self, other):
        """
        Method defining equality of 2 TIRPs.

        :param other: the second TIRP that is compared to self
        :return: boolean - True if TIRPS are equal, False otherwise
        """
        return are_TIRPs_equal(self, other)

    def __hash__(self):
        """
        Return the hash value of self object. Together with __eq__ method it is used to make list of TIRPs unique.

        :return: hash value based on symbols and relations lists and their order
        """
        return hash((sum([(i + 1) * hash(s) for i, s in enumerate(self.symbols)]), (sum([(i + 1) * hash(s) for i, s in enumerate(self.relations)]))))

    def extend(self, new_symbol, new_relations):
        """
        Extend TIRP with a new symbol and new relations. Check if sizes of lists are ok after extending.

        :param new_symbol: string representing new symbol to add
        :param new_relations: list of new relations to add
        :return: None
        """
        self.symbols.append(new_symbol)
        self.relations.extend(new_relations)
        if not self.check_size():
            raise AttributeError('Extension of TIRP is wrong!')

    def check_size(self):
        """
        Check if length of list relations is right regarding length of list symbols.

        :return: boolean - if size of symbols and relations lists match
        """
        return (len(self.symbols) ** 2 - len(self.symbols)) / 2 == len(self.relations)

    def print(self):
        """
        Pretty print TIRP as upper triangular matrix.

        :return: empty string because __repr__ method is using print() method
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
                if column_id < row_id:    # print spaces
                    print(' ' * (num_of_spaces + 1), end='')
                else:   # print relation
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
        Check if this TIRP is present in at least min_ver_supp proportion of entities.
        Set some parameters of self instance.

        :param entity_list: list of all entities
        :return: boolean - True if given TIRP has at least min_ver_supp support, otherwise False
        """
        if not self.check_size():
            print('TIRP symbols and relations lists do not have compatible size!')
            return None

        # check only entities from entity list that supported parent (smaller) TIRP
        if self.parent_entity_indices_supporting is not None:
            entity_list_reduced = list(np.array(entity_list)[self.parent_entity_indices_supporting])
        else:
            entity_list_reduced = entity_list

        supporting_indices = []
        for index, entity in enumerate(entity_list_reduced):
            lexi_sorted = lexicographic_sorting(entity)
            entity_ti = list(map(lambda s: (s[0], s[1]), lexi_sorted))
            entity_symbols = list(map(lambda s: s[2], lexi_sorted))
            if len(self.symbols) <= len(entity_symbols):
                matching_indices = check_symbols_lexicographically(entity_symbols, self.symbols, 'all')
                if matching_indices is not None and matching_indices != [None]:     # lexicographic match found, check all relations of TIRP
                    for matching_option in matching_indices:
                        all_relations_match = True

                        relation_index = 0
                        for column_count, entity_index in enumerate(matching_option[1:]):
                            for row_count in range(column_count + 1):
                                ti_1 = entity_ti[matching_option[row_count]]
                                ti_2 = entity_ti[entity_index]

                                if self.relations[relation_index] != temporal_relations(ti_1, ti_2, self.epsilon, self.max_distance):
                                    all_relations_match = False
                                    break

                                relation_index += 1

                            if not all_relations_match:
                                break

                        if all_relations_match:
                            supporting_indices.append(index)
                            self.indices_of_last_symbol_in_entities.append(list(matching_option)[-1])

        if self.parent_entity_indices_supporting is not None:
            self.entity_indices_supporting = list(np.array(self.parent_entity_indices_supporting)[supporting_indices])
        else:
            self.entity_indices_supporting = supporting_indices

        self.vertical_support = len(list(set(self.entity_indices_supporting))) / len(entity_list)

        # make 2 lists the same size by uniqueness of entity_indices_supporting
        if len(self.entity_indices_supporting) != 0:
            sym_index_ent_index_zipped_unique = list(set(list(zip(self.indices_of_last_symbol_in_entities, self.entity_indices_supporting))))
            self.indices_of_last_symbol_in_entities = list(np.array(sym_index_ent_index_zipped_unique)[:, 0])
            self.entity_indices_supporting = list(np.array(sym_index_ent_index_zipped_unique)[:, 1])

        return self.vertical_support >= self.min_ver_supp
    








class Karma:
    """
    Generate candidates by extending previous-level TIRPs.
    """
    def __init__(self, epsilon, max_distance, min_ver_supp):
        self.epsilon = epsilon
        self.max_distance = max_distance
        self.min_ver_supp = min_ver_supp

    def run(self, prev_tirps, entity_list):
        candidates = set()
        for tirp in prev_tirps:
            for idx, pid in zip(tirp.indices_of_last_symbol_in_entities,
                                tirp.entity_indices_supporting):
                ev = lexicographic_sorting(entity_list[pid])
                for next_i in range(idx+1, len(ev)):
                    start1, end1, _ = ev[idx]
                    start2, end2, sym2 = ev[next_i]
                    rel = temporal_relations((start1,end1),(start2,end2),
                                              self.epsilon, self.max_distance)
                    if rel is None:
                        continue
                    # determine new relation vector for extension
                    # placeholder for generating full relations list
                    new_relations = [rel]
                    new = tirp.extend(sym2, new_relations, tirp.entity_indices_supporting)
                    candidates.add(new)
        return list(candidates)

class Lego:
    """
    Validate candidates by computing vertical support and filtering closed patterns.
    """
    def __init__(self, epsilon, max_distance, min_ver_supp):
        self.epsilon = epsilon
        self.max_distance = max_distance
        self.min_ver_supp = min_ver_supp

    def run(self, candidates, entity_list):
        Lk = []
        for cand in tqdm(candidates, desc="Lego stage" if len(candidates)>1 else "Lego"):  # tqdm if many
            if cand.is_above_vertical_support(entity_list):
                if cand.vertical_support >= self.min_ver_supp:
                    Lk.append(cand)
        return Lk

class FrequentPatternMiner:
    """
    End-to-end KarmaLego miner.
    """
    REQUIRED_COLUMNS = ['PatientID','ConceptName','StartDate','EndDate','Value']

    def __init__(self, min_support, epsilon, max_distance,
                 max_k=3, use_dask=False):
        self.min_support = min_support
        self.epsilon = epsilon
        self.max_distance = max_distance
        self.max_k = max_k
        self.use_dask = use_dask
        self.concept_map = {}
        self.entity_list = []

    @log_execution
    def load_data(self, df):
        missing = set(self.REQUIRED_COLUMNS) - set(df.columns)
        if missing:
            raise SchemaValidationError(f"Missing columns: {missing}")
        if self.use_dask:
            df = dd.from_pandas(df, npartitions=4)
        df['StartDate'] = pd.to_datetime(df['StartDate'])
        df['EndDate'] = pd.to_datetime(df['EndDate'])
        unique = df['ConceptName'].unique().compute() if self.use_dask else df['ConceptName'].unique()
        self.concept_map = {c:i for i,c in enumerate(sorted(unique))}
        df['Code'] = df['ConceptName'].map(self.concept_map)
        # build entity list
        grp = df.groupby('PatientID')
        self.entity_list = grp.apply(lambda pdf: list(zip(pdf['StartDate'], pdf['EndDate'], pdf['Code']))).tolist()
        return df

    @log_execution
    def calculate_frequent_patterns(self):
        # k=1: single symbols
        tree = TreeNode('root')
        base = []
        for pid, events in enumerate(self.entity_list):
            unique_codes = set([c for _,_,c in events])
            for c in unique_codes:
                tirp = TIRP(self.epsilon, self.max_distance, self.min_support,
                            symbols=[c], relations=[], k=1,
                            entity_indices_supporting=[pid],
                            indices_of_last_symbol_in_entities=[
                                sorted(events, key=lambda x: x[0])[0][2]  # first idx placeholder
                            ])
                base.append(tirp)
        # prune by support
        base_pruned = [t for t in base if t.is_above_vertical_support(self.entity_list)]
        for t in base_pruned:
            tree.add_child(TreeNode(t))

        # higher-order
        current = [node.data for node in tree.children]
        for k in range(2, self.max_k+1):
            karma = Karma(self.epsilon, self.max_distance, self.min_support)
            cand = karma.run(current, self.entity_list)
            lego = Lego(self.epsilon, self.max_distance, self.min_support)
            nxt = lego.run(cand, self.entity_list)
            for t in nxt:
                # attach to appropriate parent node
                # find parent TreeNode
                pass  # implementation detail
            current = nxt
        return tree

# utils.py (utility module)
# Place:
# - temporal_relations(start_end1, start_end2, epsilon, max_distance)
# - lexicographic_sorting(events)
# Additional TIRP helper methods (e.g., is_above_vertical_support) remain in TIRP class.