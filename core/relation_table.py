"""
Allen-like interval relation composition for KarmaLego.

Supported relations (7):
    '<' : before
    'm' : meets
    'o' : overlaps
    'f' : finished-by (A is finished by B: start(A) < start(B) and end(A)==end(B))
    'c' : contains (A contains B)
    's' : start-by (A is started by B: start(A)==start(B) and end(A) > end(B))
    '=' : equal
"""

from functools import lru_cache

# Composition (transition) table: given (A r B, B r C), what are possible A r C.
# Existing hand-crafted table retained; accesses should go through compose_relation.
_transition_table = {
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


@lru_cache(maxsize=512)
def compose_relation(a_r_b, b_r_c):
    """
    Safe composition: given (A r B) and (B r C), return possible A r C relations.
    Returns empty list if the pair is not defined.
    """
    key = (a_r_b, b_r_c)
    if key not in _transition_table:
        raise KeyError("Missing transition table entry for %s", key)
    return _transition_table[key]