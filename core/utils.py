import pandas as pd
import numpy as np
from core.relation_table import compose_relation, get_num_relations


def normalize_time_param(x):
    """
    Normalize a duration parameter:
      - int/float -> int (no scaling)
      - pd.Timedelta -> ns (int)
      - None -> None
    No string support.
    """
    if x is None:
        return None
    if isinstance(x, pd.Timedelta):
        return int(x.value)  # already ns
    if isinstance(x, (int, np.integer)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        return int(np.rint(float(x)))
    raise TypeError(f"Unsupported type for time param: {type(x)}. "
                    "Pass numeric (same unit as your time columns) or pd.Timedelta.")


def lexicographic_sorting(entity):
    """
    entity: list of tuples (start, end, symbol)
    Sort by start, then end (descending), then symbol.
    Keeps the forward temporal order while ensuring a consistent ordering for intervals with the same start time.
    """
    return sorted(entity, key=lambda x: (x[0], -x[1], x[2]))


def _map_relation_to_active(rel):
    """
    Map a 7-relation symbol to the active relation alphabet (2, 3, 5, or 7).
    """
    if rel is None:
        return None

    num_relations = get_num_relations()
    if num_relations == 7:
        return rel
    if num_relations == 5:
        if rel == "m":
            return "<"
        if rel == "=":
            return "c"
        return rel
    if num_relations == 3:
        if rel in ("<", "m"):
            return "<"
        if rel in ("c", "s", "f", "="):
            return "c"
        return "o"
    if num_relations == 2:
        if rel in ("<", "m", "o"):
            return "p"
        return "c"
    return rel


def temporal_relations(ti_1, ti_2, epsilon, max_distance):
    """
    Determine the Allen-like relation from interval ti_1 to ti_2 among the seven supported:
      '<' : before
      'm' : meets
      'o' : overlaps
      'f' : finished-by (ti_1 is finished by ti_2: start1 < start2 and end1 == end2)
      'c' : contains (ti_1 contains ti_2)
      's' : start-by (ti_1 is started by ti_2: start1 == start2 and end1 > end2)
      '=' : equal

    Returns None if no relation matches or gap exceeds max_distance.
    """
    start1, end1 = ti_1
    start2, end2 = ti_2

    # gap too large
    if start2 - end1 > max_distance:
        return None

    # equal (within epsilon)
    if abs(start1 - start2) <= epsilon and abs(end1 - end2) <= epsilon:
        return _map_relation_to_active("=")

    # before (strictly before, with epsilon tolerance)
    if end1 + epsilon < start2:
        return _map_relation_to_active("<")

    # meets (end1 ~ start2)
    if abs(end1 - start2) <= epsilon:
        return _map_relation_to_active("m")

    # overlaps: start1 < start2 < end1 < end2
    if start1 < start2 < end1 < end2:
        return _map_relation_to_active("o")

    # finished-by: ti_1 is finished-by ti_2 -> start1 < start2 and end1 ~ end2
    if start1 < start2 and abs(end1 - end2) <= epsilon:
        return _map_relation_to_active("f")

    # contains: ti_1 strictly contains ti_2 (allowing some epsilon leeway if needed)
    if start1 < start2 and end1 > end2:
        return _map_relation_to_active("c")

    # start-by: ti_1 is started-by ti_2 -> start1 ~ start2 and end1 > end2
    if abs(start1 - start2) <= epsilon and end1 > end2:
        return _map_relation_to_active("s")

    return None
    

def check_symbols_lexicographically(entity_symbols, pattern_symbols):
    """
    Find all index tuples in entity_symbols where pattern_symbols appear as a subsequence
    in order (strictly increasing indices). Returns list of tuples of indices.
    """
    results = []

    def helper(e_pos, p_pos, path):
        if p_pos == len(pattern_symbols):
            results.append(tuple(path))
            return
        symbol = pattern_symbols[p_pos]
        for i in range(e_pos, len(entity_symbols)):
            if entity_symbols[i] == symbol:
                helper(i + 1, p_pos + 1, path + [i])

    helper(0, 0, [])
    return results if results else None


def find_all_possible_extensions(all_paths, path, BrC, curr_rel_index, decrement_index, TIRP_relations):
    """
    Backtrack to enumerate all valid predecessor relation sequences needed when extending a TIRP.

    Given the new-last relation (BrC) between the previous last symbol and the candidate extension,
    recursively walk backward over the flattened upper-triangular relation list of the base TIRP,
    composing relations via the transition table to produce all consistent sequences of preceding relations.

    Parameters
    ----------
    all_paths : list[list]
        Accumulator. Each valid predecessor sequence (excluding BrC itself) is appended here.
    path : list
        Working buffer of relations collected so far; mutated in-place.
    BrC : hashable
        Relation between the previous last symbol B and the new symbol C being added.
    curr_rel_index : int
        Current index in TIRP_relations to consider (walking backwards).
    decrement_index : int
        Step size control for moving backward in the flattened upper-triangular structure.
    TIRP_relations : list
        Flattened list of existing relations of the base TIRP (upper-triangular order).

    Returns
    -------
    list[list]
        The same `all_paths` object, extended with each valid predecessor relation sequence.
    """
    if curr_rel_index < 0:
        all_paths.append(tuple(path))
        return all_paths

    ArB = TIRP_relations[curr_rel_index]
    poss_relations = compose_relation(ArB, BrC)

    for poss_rel in poss_relations:
        path.append(poss_rel)
        decrement_index -= 1
        find_all_possible_extensions(
            all_paths,
            path,
            poss_rel,
            curr_rel_index - decrement_index - 1,
            decrement_index,
            TIRP_relations,
        )
        decrement_index += 1
        path.pop()
    return all_paths


def decode_pattern(tirp, inverse_symbol_map):
    """
    Convert a TIRP's symbols and relations into a human-readable string.

    Example:
        symbols = [4, 85, 23], relations = ['<', 'o', 'c']
        -> "Temp < Antibiotics o BP c HR"

    Parameters
    ----------
    tirp : TIRP
        A pattern instance.

    Returns
    -------
    str
        Decoded pattern: "ConceptName < ConceptName o ConceptName ..."
    """
    parts = [inverse_symbol_map.get(str(tirp.symbols[0]), str(tirp.symbols[0]))]
    relation_index = 0
    for i in range(1, len(tirp.symbols)):
        name = inverse_symbol_map.get(str(tirp.symbols[i]), str(tirp.symbols[i]))
        rel = tirp.relations[relation_index]
        parts.append(f"{rel} {name}")
        relation_index += i
    return " ".join(parts)


def count_embeddings_in_single_entity(tirp, entity):
    """
    Count distinct embeddings of a TIRP in a single entity where all temporal relations match.

    The procedure:
      1. Lexicographically sort the entity's intervals to impose a deterministic order.
      2. Find all subsequence index tuples where tirp.symbols appear in order.
      3. For each candidate embedding, verify that each expected pairwise relation in tirp.relations
         matches the actual temporal relation computed from the entity intervals using the
         current epsilon and max_distance settings.
      4. Each embedding that satisfies all relations increments the count (multiple embeddings in the
         same entity count separately here; higher-level logic can choose to dedupe if desired).

    Parameters
    ----------
    tirp : TIRP
        The pattern to match. Must have `symbols`, `relations`, `epsilon`, and `max_distance`.
    entity : list
        A single entity as a list of (start, end, symbol) tuples.

    Returns
    -------
    int
        Number of distinct valid embeddings of the TIRP in this entity (could be zero).
    """
    count = 0
    lexi_sorted = lexicographic_sorting(entity)
    entity_ti = [(s, e) for s, e, _ in lexi_sorted]
    entity_symbols = [sym for _, _, sym in lexi_sorted]

    if len(tirp.symbols) > len(entity_symbols):
        return 0

    matching_options = check_symbols_lexicographically(entity_symbols, tirp.symbols)
    if matching_options is None:
        return 0

    for matching_option in matching_options:
        all_relations_match = True
        relation_index = 0
        for column_count, entity_index in enumerate(matching_option[1:]):
            for row_count in range(column_count + 1):
                ti_1 = entity_ti[matching_option[row_count]]
                ti_2 = entity_ti[entity_index]
                expected_rel = tirp.relations[relation_index]
                actual_rel = temporal_relations(ti_1, ti_2, tirp.epsilon, tirp.max_distance)
                if expected_rel != actual_rel:
                    all_relations_match = False
                    break
                relation_index += 1
            if not all_relations_match:
                break
        if all_relations_match:
            count += 1
    return count