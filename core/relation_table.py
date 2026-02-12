"""
Allen-like interval relation composition for KarmaLego.

This module provides composition tables for temporal interval relations at four granularity levels.
Each table defines how relations compose: given (A r B, B r C), what are possible A r C relations.

IMPORTANT DESIGN NOTE:
  - Intervals are always ordered by (start time, end time, concept).
  - Therefore, all relations are FORWARD-ONLY.
  - Inverse Allen relations (after, met-by, during, etc.) are never represented explicitly.
  - “During” is implicitly handled as the inverse of “contains” via interval ordering.

READING THE TABLES:
  - Organized as explicit composition rules
  - Each row is ONE rule: (relation_pair): [possible_results]
  - Example:
        ('o', 'o'): ['<', 'm', 'o']
    means:
        If A overlaps B AND B overlaps C,
        then A may be before, meet, or overlap C.

---------------------------------------------------------------------
AVAILABLE RELATION SETS
---------------------------------------------------------------------

2 Relations (Ultra-coarse, coarsened from 7):
        'p'  : proceed
                     (MERGES: '<', 'm', 'o')
        'c'  : contain
                     (MERGES: 'c', 's', 'f', '=')

    Coarsening rules:
        - '<' (before)      → 'p'
        - 'm' (meets)       → 'p'
        - 'o' (overlaps)    → 'p'
        - 'c' (contains)    → 'c'
        - 's' (started-by)  → 'c'
        - 'f' (finished-by) → 'c'
        - '=' (equal)       → 'c'

    Notes:
        - This is the most compact forward-only algebra: ordering/overlap collapse to 'p',
            and containment/equality collapse to 'c'.
        - Guaranteed to be closed under composition when derived from the 7-relation table.

    Use for:
        Maximum throughput where only proceed vs contain is needed.

7 Relations (Forward Allen-like, base algebra):
    '<'  : before
    'm'  : meets
    'o'  : overlaps
    'f'  : finished-by
           (start(A) < start(B) and end(A) == end(B))
    'c'  : contains
           (start(A) < start(B) and end(A) > end(B))
    's'  : started-by
           (start(A) == start(B) and end(A) > end(B))
    '='  : equal

  Notes:
    - This is a forward-only subset of Allen’s interval algebra.
    - Inverse relations (after, met-by, during, etc.) are represented implicitly
      by interval ordering and relation direction.
    - This table is the semantic source of truth.

  Use for:
    Fine-grained temporal analysis and maximum expressive power.


---------------------------------------------------------------------

5 Relations (Intermediate, coarsened from 7):
    '<'  : before
           (MERGES: '<', 'm')
    'o'  : overlaps
           (UNCHANGED)
    'c'  : contains
           (UNCHANGED)
    's'  : started-by
           (UNCHANGED)
    'f'  : finished-by
           (UNCHANGED)

  Coarsening rules:
    - 'm' (meets)        → '<'   (absorbed via epsilon-soft edges)
    - '=' (equal)        → 'c'   (treated as degenerate containment)

  Notes:
    - All relations are a strict subset of the 7-relation alphabet.
    - No inverse or endpoint-only relations are introduced.
    - Preserves overlap and containment geometry while reducing branching.

  Use for:
    Balanced granularity with improved runtime over full 7-relation mining.


---------------------------------------------------------------------

3 Relations (Minimal, coarsened from 7):
    '<'  : before
           (MERGES: '<', 'm')
    'o'  : overlaps
           (UNCHANGED)
    'c'  : contains / inclusion
           (MERGES: 'c', 's', 'f', '=')

  Coarsening rules:
    - 'm' (meets)        → '<'
    - 's' (started-by)  → 'c'
    - 'f' (finished-by) → 'c'
    - '=' (equal)       → 'c'

  Notes:
    - Represents the minimal interval algebra that still distinguishes:
        ordering, intersection, and inclusion.
    - Well-suited for high-performance temporal pattern mining.
    - Guaranteed to be closed under composition when derived from the 7-relation table.

  Use for:
    Fast pattern discovery and coarse temporal structure mining.


---------------------------------------------------------------------

USAGE:
  Pass 'num_relations' ∈ {2, 3, 5, 7} when initializing KarmaLego.
  The corresponding composition table is selected automatically.

"""

from functools import lru_cache

# ============================================================================
# 2-RELATION COMPOSITION TABLE (Ultra-coarse: proceed, contain)
# Coarsened from 7-relation table by merging:
#   '<', 'm', 'o' -> 'p'
#   'c', 's', 'f', '=' -> 'c'
# ============================================================================
_transition_table_2 = {
    # A p B, B p C  =>  A p C
    ('p', 'p'): ['p'],
    # A p B, B c C  =>  A p C or c C
    ('p', 'c'): ['p', 'c'],
    # A c B, B p C  =>  A p C or c C
    ('c', 'p'): ['p', 'c'],
    # A c B, B c C  =>  A p C or c C
    ('c', 'c'): ['p', 'c'],
}

# ============================================================================
# 3-RELATION COMPOSITION TABLE (Minimal: before, overlaps, contains)
# Coarsened from 7-relation table by merging:
#   '<' and 'm' → '<'
#   'c', 's', 'f', '=' → 'c'
# ============================================================================
_transition_table_3 = {
    # A < B, B < C  =>  A < C
    ('<', '<'): ['<'],
    # A < B, B o C  =>  A < C
    ('<', 'o'): ['<'],
    # A < B, B c C  =>  A < C
    ('<', 'c'): ['<'],

    # A o B, B < C  =>  A < C
    ('o', '<'): ['<'],
    # A o B, B o C  =>  A < C, o, c
    ('o', 'o'): ['<', 'o', 'c'],
    # A o B, B c C  =>  A < C, o, c
    ('o', 'c'): ['<', 'o', 'c'],

    # A c B, B < C  =>  A < C, o, c
    ('c', '<'): ['<', 'o', 'c'],
    # A c B, B o C  =>  A o C, c
    ('c', 'o'): ['o', 'c'],
    # A c B, B c C  =>  A c C
    ('c', 'c'): ['c'],
}

# ============================================================================
# 5-RELATION COMPOSITION TABLE (Intermediate: before, overlaps, contains, started-by, finished-by)
# Coarsened from 7-relation table by merging:
#   '<' and 'm' → '<'
#   '=' → 'c'
# ============================================================================
_transition_table_5 = {
    # A < B, B < C  =>  A < C
    ('<', '<'): ['<'],
    # A < B, B o C  =>  A < C
    ('<', 'o'): ['<'],
    # A < B, B c C  =>  A < C
    ('<', 'c'): ['<'],
    # A < B, B f C  =>  A < C
    ('<', 'f'): ['<'],
    # A < B, B s C  =>  A < C
    ('<', 's'): ['<'],

    # A o B, B < C  =>  A < C
    ('o', '<'): ['<'],
    # A o B, B o C  =>  A < C, o
    ('o', 'o'): ['<', 'o'],
    # A o B, B c C  =>  A < C, o, c, f
    ('o', 'c'): ['<', 'o', 'c', 'f'],
    # A o B, B f C  =>  A < C, o
    ('o', 'f'): ['<', 'o'],
    # A o B, B s C  =>  A o C
    ('o', 's'): ['o'],

    # A c B, B < C  =>  A < C, o, c, f
    ('c', '<'): ['<', 'o', 'c', 'f'],
    # A c B, B o C  =>  A o C, c, f
    ('c', 'o'): ['o', 'c', 'f'],
    # A c B, B c C  =>  A c C
    ('c', 'c'): ['c'],
    # A c B, B f C  =>  A c C
    ('c', 'f'): ['c'],
    # A c B, B s C  =>  A o C, c, f
    ('c', 's'): ['o', 'c', 'f'],

    # A f B, B < C  =>  A < C
    ('f', '<'): ['<'],
    # A f B, B o C  =>  A o C
    ('f', 'o'): ['o'],
    # A f B, B c C  =>  A c C
    ('f', 'c'): ['c'],
    # A f B, B f C  =>  A f C
    ('f', 'f'): ['f'],
    # A f B, B s C  =>  A o C
    ('f', 's'): ['o'],

    # A s B, B < C  =>  A < C
    ('s', '<'): ['<'],
    # A s B, B o C  =>  A < C, o
    ('s', 'o'): ['<', 'o'],
    # A s B, B c C  =>  A < C, o, c, f
    ('s', 'c'): ['<', 'o', 'c', 'f'],
    # A s B, B f C  =>  A < C, o
    ('s', 'f'): ['<', 'o'],
    # A s B, B s C  =>  A s C
    ('s', 's'): ['s'],
}

# ============================================================================
# 7-RELATION COMPOSITION TABLE (Full Allen: all 7 relations)
# ============================================================================
# Composition (transition) table: given (A r B, B r C), what are possible A r C.
# Existing hand-crafted table retained; accesses should go through compose_relation.
_transition_table_7 = {
    ('<', '<'): ['<'],
    ('<', 'm'): ['<'],
    ('<', 'o'): ['<'],
    ('<', 'c'): ['<'],
    ('<', 'f'): ['<'],
    ('<', '='): ['<'],
    ('<', 's'): ['<'],

    ('m', '<'): ['<'],
    ('m', 'm'): ['<'],
    ('m', 'o'): ['<'],
    ('m', 'c'): ['<'],
    ('m', 'f'): ['<'],
    ('m', '='): ['m'],
    ('m', 's'): ['m'],

    ('o', '<'): ['<'],
    ('o', 'm'): ['<'],
    ('o', 'o'): ['<', 'm', 'o'],
    ('o', 'c'): ['<', 'm', 'o', 'c', 'f'],
    ('o', 'f'): ['<', 'm', 'o'],
    ('o', '='): ['o'],
    ('o', 's'): ['o'],

    ('c', '<'): ['<', 'm', 'o', 'c', 'f'],
    ('c', 'm'): ['o', 'c', 'f'],
    ('c', 'o'): ['o', 'c', 'f'],
    ('c', 'c'): ['c'],
    ('c', 'f'): ['c'],
    ('c', '='): ['c'],
    ('c', 's'): ['o', 'c', 'f'],

    ('f', '<'): ['<'],
    ('f', 'm'): ['m'],
    ('f', 'o'): ['o'],
    ('f', 'c'): ['c'],
    ('f', 'f'): ['f'],
    ('f', '='): ['f'],
    ('f', 's'): ['o'],

    ('=', '<'): ['<'],
    ('=', 'm'): ['m'],
    ('=', 'o'): ['o'],
    ('=', 'c'): ['c'],
    ('=', 'f'): ['f'],
    ('=', '='): ['='],
    ('=', 's'): ['s'],

    ('s', '<'): ['<'],
    ('s', 'm'): ['<'],
    ('s', 'o'): ['<', 'm', 'o'],
    ('s', 'c'): ['<', 'm', 'o', 'c', 'f'],
    ('s', 'f'): ['<', 'm', 'o'],
    ('s', '='): ['s'],
    ('s', 's'): ['s'],
}


# Global variable to store the currently active transition table
_active_transition_table = _transition_table_7
_active_num_relations = 7


def set_relation_table(num_relations):
    """
    Set which relation composition table to use globally.

    Parameters
    ----------
    num_relations : int
        Number of relations to support: 2, 3, 5, or 7.

    Raises
    ------
    ValueError
        If num_relations is not 2, 3, 5, or 7. Includes information about
        valid options and their use cases.
    """
    global _active_transition_table, _active_num_relations

    if num_relations == 2:
        _active_transition_table = _transition_table_2
        _active_num_relations = 2
    elif num_relations == 3:
        _active_transition_table = _transition_table_3
        _active_num_relations = 3
    elif num_relations == 5:
        _active_transition_table = _transition_table_5
        _active_num_relations = 5
    elif num_relations == 7:
        _active_transition_table = _transition_table_7
        _active_num_relations = 7
    else:
        raise ValueError(
            f"Invalid num_relations={num_relations}. Must be one of:\n"
            f"  2 : Ultra-coarse (proceed, contain) - fastest\n"
            f"  3 : Minimal (before, overlaps, contains) - fast\n"
            f"  5 : Intermediate (before, overlaps, contains, started-by, finished-by) - balanced\n"
            f"  7 : Full Allen (all 7 relations) - most detailed"
        )
    
    # Clear the lru_cache since table has changed
    compose_relation.cache_clear()


def get_relation_table():
    """
    Get the currently active relation composition table.

    Returns
    -------
    dict
        The active transition table.
    """
    return _active_transition_table


def get_num_relations():
    """
    Get the number of relations in the currently active table.

    Returns
    -------
    int
        Number of relations: 2, 3, 5, or 7.
    """
    return _active_num_relations


@lru_cache(maxsize=512)
def compose_relation(a_r_b, b_r_c):
    """
    Safe composition: given (A r B) and (B r C), return possible A r C relations.

    Uses the currently active relation composition table (set via set_relation_table).

    Parameters
    ----------
    a_r_b : hashable
        Relation from A to B.
    b_r_c : hashable
        Relation from B to C.

    Returns
    -------
    list
        Possible relations from A to C.

    Raises
    ------
    KeyError
        If the composition pair is not defined in the active table.
    """
    key = (a_r_b, b_r_c)
    if key not in _active_transition_table:
        raise KeyError(f"Missing transition table entry for {key}")
    return _active_transition_table[key]