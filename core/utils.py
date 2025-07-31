from core.transition_table import *


def lexicographic_sorting(entity):
    """
    entity: list of tuples (start, end, symbol)
    Sort by start, then end, then symbol.
    """
    return sorted(entity, key=lambda x: (x[0], x[1], x[2]))


def temporal_relations(ti_1, ti_2, epsilon, max_distance):
    """
    Simplified temporal relation between two intervals.
    ti = (start, end), where start/end are numeric (e.g., timestamps or ints).
    Relations:
      - 'b': ti_1 before ti_2 (with gap > epsilon)
      - 'm': meets (end1 approx start2 within epsilon)
      - 'o': overlap (start2 <= end1 < end2)
      - 'c': ti_1 contains ti_2
      - 'ci': ti_1 is contained in ti_2
    Returns None if no relation of interest or exceeds max_distance.
    """
    start1, end1 = ti_1
    start2, end2 = ti_2

    # if they're far apart beyond max_distance, treat as no relation
    if start2 - end1 > max_distance:
        return None

    if end1 + epsilon < start2:
        return "b"  # before
    if abs(end1 - start2) <= epsilon:
        return "m"  # meets
    if start2 <= end1 < end2:
        return "o"  # overlap
    if start1 <= start2 and end1 >= end2:
        return "c"  # contains
    if start2 <= start1 and end2 >= end1:
        return "ci"  # contained in
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