from core.relation_table import compose_relation


def lexicographic_sorting(entity):
    """
    entity: list of tuples (start, end, symbol)
    Sort by start, then end, then symbol.
    """
    return sorted(entity, key=lambda x: (x[0], x[1], x[2]))


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
        return "="

    # before (strictly before, with epsilon tolerance)
    if end1 + epsilon < start2:
        return "<"

    # meets (end1 ~ start2)
    if abs(end1 - start2) <= epsilon:
        return "m"

    # overlaps: start1 < start2 < end1 < end2
    if start1 < start2 < end1 < end2:
        return "o"

    # finished-by: ti_1 is finished-by ti_2 -> start1 < start2 and end1 ~ end2
    if start1 < start2 and abs(end1 - end2) <= epsilon:
        return "f"

    # contains: ti_1 strictly contains ti_2 (allowing some epsilon leeway if needed)
    if start1 < start2 and end1 > end2:
        return "c"

    # start-by: ti_1 is started-by ti_2 -> start1 ~ start2 and end1 > end2
    if abs(start1 - start2) <= epsilon and end1 > end2:
        return "s"

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


def find_all_possible_extensions(all_paths, BrC, curr_rel_index, decrement_index, TIRP_relations):
    """
    Compute all valid predecessor relation sequences needed when extending a TIRP by a new symbol.

    Given the relation between the previous last symbol and the new symbol (BrC) and the current
    flattened relation list of the base TIRP (TIRP_relations), this walks backwards through the
    implied upper-triangular constraints, using the composition table to enumerate all sequences
    of prior relations that keep the extended pattern consistent.

    This is an iterative replacement of a recursive backtracking algorithm. It accumulates each
    completed "extension path" (a list of relations preceding the new-last relation) into `all_paths`.

    Parameters
    ----------
    all_paths : list
        Mutable accumulator. The function appends each valid extension path to this list and returns it.
        Each element appended is a list of relations (in the order generated) that, together with BrC,
        satisfy the consistency constraints going backward.
    BrC : hashable
        Relation between the previous last symbol (B) and the new symbol (C) being added.
    curr_rel_index : int
        Index into TIRP_relations pointing to the current relation (working backward from the end).
    decrement_index : int
        Step control for navigating the flattened upper-triangular structure; determines how far to
        move the next index backward in each level.
    TIRP_relations : list
        Existing relations of the base TIRP being extended (flattened upper-triangular order).

    Returns
    -------
    list
        The same `all_paths` object, extended with each valid predecessor-relation sequence. Each inner list
        represents the required relations that precede BrC for a consistent extension (does not include BrC itself).
    """
    results = []
    # Stack entries: (current_rel_index, current_decrement_index, accumulated_path, current_BrC)
    stack = [(curr_rel_index, decrement_index, [], BrC)]

    while stack:
        ci, di, pth, brc = stack.pop()
        if ci < 0:
            # Completed a full backward path; record it
            results.append(pth.copy())
            continue

        ArB = TIRP_relations[ci]
        poss_relations = compose_relation(ArB, brc)
        if not poss_relations:
            # No valid compositions here; dead end
            continue

        for poss_rel in poss_relations:
            next_ci = ci - di - 1
            next_di = di - 1
            new_path = pth + [poss_rel]
            stack.append((next_ci, next_di, new_path, poss_rel))

    all_paths.extend(results)
    return all_paths


def vertical_support_symbol(entity_list, symbol):
    """
    Compute the vertical support of a single symbol across entities.

    Vertical support is defined as the fraction of entities that contain at least one occurrence
    of the given symbol.

    Parameters
    ----------
    entity_list : list
        List of entities, where each entity is a list of (start, end, symbol) tuples.
    symbol : hashable
        The symbol whose support is being measured.

    Returns
    -------
    float
        Fraction in [0,1] of entities containing the symbol. Returns 0.0 if entity_list is empty.
    """
    supporting = 0
    for entity in entity_list:
        entity_symbols = [sym for _, _, sym in entity]
        if symbol in entity_symbols:
            supporting += 1
    return supporting / len(entity_list) if entity_list else 0.0


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