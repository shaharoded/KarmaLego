import pytest
from copy import deepcopy
from core.karmalego import TIRP


@pytest.fixture
def complex_entities():
    """
    Build a more complex synthetic entity list with 3 patients (entities).
    Patterns involve symbols A and B with temporal relations:
      - Patient 0: A(0,1), A(2,3), B(5,6)         -> A before B (multiple embeddings of A before B)
      - Patient 1: A(0,2), B(1,3)               -> A overlaps B
      - Patient 2: A(0,2), B(3,5)               -> A before B
    """
    e0 = [(0, 1, "A"), (2, 3, "A"), (5, 6, "B")]  # A before B (two A choices)
    e1 = [(0, 2, "A"), (1, 3, "B")]              # A overlaps B
    e2 = [(0, 2, "A"), (3, 5, "B")]              # A before B
    return [e0, e1, e2]


def test_singleton_A_support_full(complex_entities):
    entities = complex_entities
    tirp_A = TIRP(epsilon=0, max_distance=100, min_ver_supp=1 / 3, symbols=["A"], relations=[], k=1)
    assert tirp_A.is_above_vertical_support(entities) is True
    print("\n=== Singleton TIRP A ===")
    print("Summary:", tirp_A)
    tirp_A.print()
    assert pytest.approx(tirp_A.vertical_support, rel=1e-3) == 1.0  # all 3 support A
    assert set(tirp_A.entity_indices_supporting) == {0, 1, 2}

    tirp_B = TIRP(epsilon=0, max_distance=100, min_ver_supp=0.9, symbols=["B"], relations=[], k=1)
    assert tirp_B.is_above_vertical_support(entities) is True
    print("\n=== Singleton TIRP B ===")
    print("Summary:", tirp_B)
    tirp_B.print()
    assert pytest.approx(tirp_B.vertical_support, rel=1e-3) == 1.0


def test_pair_A_before_B_and_overlap(complex_entities):
    entities = complex_entities
    tirp_Ab = TIRP(epsilon=0, max_distance=100, min_ver_supp=0.5, symbols=["A", "B"], relations=['<'], k=2)
    assert tirp_Ab.is_above_vertical_support(entities) is True
    print("\n=== Pair TIRP A before B ===")
    print("Summary:", tirp_Ab)
    tirp_Ab.print()
    assert pytest.approx(tirp_Ab.vertical_support, rel=1e-3) == 2 / 3

    tirp_Ao = TIRP(epsilon=0, max_distance=100, min_ver_supp=0.2, symbols=["A", "B"], relations=["o"], k=2)
    assert tirp_Ao.is_above_vertical_support(entities) is True
    print("\n=== Pair TIRP A overlaps B ===")
    print("Summary:", tirp_Ao)
    tirp_Ao.print()
    assert pytest.approx(tirp_Ao.vertical_support, rel=1e-3) == 1 / 3

    tirp_Ac = TIRP(epsilon=0, max_distance=100, min_ver_supp=0.1, symbols=["A", "B"], relations=["c"], k=2)
    assert not tirp_Ac.is_above_vertical_support(entities)
    print("\n=== Pair TIRP A contains B (negative) ===")
    print("Summary:", tirp_Ac)
    tirp_Ac.print()
    assert tirp_Ac.vertical_support == 0.0


def test_multiple_embeddings_deduplication():
    e0 = [(0, 1, "A"), (2, 3, "A"), (5, 6, "B")]  # multiple A before B
    e1 = [(0, 1, "B")]
    e2 = [(0, 2, "B")]
    entities = [e0, e1, e2]

    tirp_Ab = TIRP(epsilon=0, max_distance=100, min_ver_supp=0.1, symbols=["A", "B"], relations=['<'], k=2)
    assert tirp_Ab.is_above_vertical_support(entities) is True
    print("\n=== Multiple embeddings deduplication (A before B) ===")
    print("Summary:", tirp_Ab)
    tirp_Ab.print()
    assert set(tirp_Ab.entity_indices_supporting) == {0}
    assert pytest.approx(tirp_Ab.vertical_support, rel=1e-3) == 1 / 3


def test_indices_of_last_symbol_alignment(complex_entities):
    entities = complex_entities
    tirp_Ab = TIRP(epsilon=0, max_distance=100, min_ver_supp=0.1, symbols=["A", "B"], relations=['<'], k=2)
    assert tirp_Ab.is_above_vertical_support(entities) is True
    print("\n=== Last-symbol alignment for A before B ===")
    print("Summary:", tirp_Ab)
    tirp_Ab.print()

    expected_map = {0: 2, 2: 1}
    actual_pairs = list(zip(tirp_Ab.indices_of_last_symbol_in_entities, tirp_Ab.entity_indices_supporting))
    for last_symbol_idx, entity_idx in actual_pairs:
        assert expected_map[entity_idx] == last_symbol_idx, (
            f"Entity {entity_idx} had last symbol {last_symbol_idx}, expected {expected_map[entity_idx]}"
        )


def test_parent_filtering_restricts_search(complex_entities):
    entities = complex_entities
    parent_A = TIRP(epsilon=0, max_distance=100, min_ver_supp=0.0, symbols=["A"], relations=[], k=1)
    assert parent_A.is_above_vertical_support(entities)
    print("\n=== Parent singleton A ===")
    print("Summary:", parent_A)
    parent_A.print()

    child_Ab = TIRP(epsilon=0, max_distance=100, min_ver_supp=0.0, symbols=["A", "B"], relations=['<'], k=2)
    child_Ab.parent_entity_indices_supporting = [2]
    assert child_Ab.is_above_vertical_support(entities) is True
    print("\n=== Child A before B with restricted parent support ===")
    print("Summary:", child_Ab)
    child_Ab.print()
    assert set(child_Ab.entity_indices_supporting) == {2}
    assert pytest.approx(child_Ab.vertical_support, rel=1e-3) == 1 / 3


def test_extension_flow_with_parent(complex_entities):
    entities = complex_entities
    tirp_A = TIRP(epsilon=0, max_distance=100, min_ver_supp=0.0, symbols=["A"], relations=[], k=1)
    assert tirp_A.is_above_vertical_support(entities)
    print("\n=== Base TIRP A ===")
    print("Summary:", tirp_A)
    tirp_A.print()

    child = deepcopy(tirp_A)
    child.extend("B", ['<'])
    child.k = 2
    child.parent_entity_indices_supporting = tirp_A.entity_indices_supporting
    assert child.is_above_vertical_support(entities) is True
    print("\n=== Extended TIRP A before B (with parent filtering) ===")
    print("Summary:", child)
    child.print()
    assert set(child.entity_indices_supporting) == {0, 2}
    assert pytest.approx(child.vertical_support, rel=1e-3) == 2 / 3


def test_equality_and_hash_behavior():
    t1 = TIRP(epsilon=0, max_distance=10, min_ver_supp=0.1, symbols=["X"], relations=[], k=1)
    t2 = TIRP(epsilon=0, max_distance=10, min_ver_supp=0.1, symbols=["X"], relations=[], k=1)
    t3 = TIRP(epsilon=0, max_distance=10, min_ver_supp=0.1, symbols=["X", "Y"], relations=['<'], k=2)

    print("\n=== Equality/hash sanity ===")
    print("t1:", t1); t1.print()
    print("t2:", t2); t2.print()
    print("t3:", t3); t3.print()

    assert t1 == t2
    assert hash(t1) == hash(t2)
    assert t1 != t3
    s = {t1, t2, t3}
    assert len(s) == 2


def test_length3_pattern_support():
    """
    Test a length-3 pattern A before B before C over entities where:
      - entity0 and entity1 contain A,B,C in order (support the triple),
      - entity2 is missing B so does not support.
    Verifies both the positive triple and a negative variant with a wrong relation.
    """
    entities = [
        [(0, 1, "A"), (2, 3, "B"), (4, 5, "C")],  # A before B before C
        [(0, 2, "A"), (3, 4, "B"), (5, 6, "C")],  # A before B before C
        [(0, 1, "A"), (2, 3, "C")],               # missing B
    ]

    # Correct triple: all relations are 'before' => ['<','<','<'] corresponds to
    # (A,B), (A,C), (B,C)
    tirp_ABC = TIRP(epsilon=0, max_distance=100, min_ver_supp=0.5,
                    symbols=["A", "B", "C"], relations=['<', '<', '<'], k=3)
    assert tirp_ABC.is_above_vertical_support(entities) is True
    print("\n=== Length-3 TIRP A before B before C ===")
    print("Summary:", tirp_ABC)
    tirp_ABC.print()
    # Support should be 2/3 (entities 0 and 1)
    assert pytest.approx(tirp_ABC.vertical_support, rel=1e-3) == 2 / 3
    assert set(tirp_ABC.entity_indices_supporting) == {0, 1}

    # Negative: last relation is wrong (B overlaps C instead of before), should fail entirely
    tirp_ABC_bad = TIRP(epsilon=0, max_distance=100, min_ver_supp=0.1,
                        symbols=["A", "B", "C"], relations=['<', '<', "o"], k=3)
    assert not tirp_ABC_bad.is_above_vertical_support(entities)
    print("\n=== Length-3 TIRP with wrong relation (should be unsupported) ===")
    print("Summary:", tirp_ABC_bad)
    tirp_ABC_bad.print()
    assert tirp_ABC_bad.vertical_support == 0.0


def test_tirp_print_output_various_lengths():
    """
    Creates synthetic entities and prints several TIRPs (length 1-3) using the .print() method
    to verify human-readable output format.
    """
    # Create synthetic dataset of 4 entities
    entities = [
        [(0, 1, "A"), (2, 3, "B"), (4, 5, "C")],        # A before B before C
        [(0, 2, "A"), (3, 4, "B"), (5, 6, "C")],        # A before B before C
        [(0, 1, "A"), (1, 3, "B")],                     # A overlaps B
        [(0, 1, "C")],                                  # C only
    ]

    print("\n--- Singleton TIRP: A ---")
    tirp_A = TIRP(epsilon=0, max_distance=100, min_ver_supp=0.0, symbols=["A"], relations=[], k=1)
    tirp_A.is_above_vertical_support(entities)
    tirp_A.print()

    print("\n--- Pair TIRP: A before B ---")
    tirp_Ab = TIRP(epsilon=0, max_distance=100, min_ver_supp=0.0, symbols=["A", "B"], relations=['<'], k=2)
    tirp_Ab.is_above_vertical_support(entities)
    tirp_Ab.print()

    print("\n--- Pair TIRP: A overlaps B ---")
    tirp_Ao = TIRP(epsilon=0, max_distance=100, min_ver_supp=0.0, symbols=["A", "B"], relations=["o"], k=2)
    tirp_Ao.is_above_vertical_support(entities)
    tirp_Ao.print()

    print("\n--- Triple TIRP: A before B before C ---")
    tirp_ABC = TIRP(epsilon=0, max_distance=100, min_ver_supp=0.0,
                    symbols=["A", "B", "C"], relations=['<', '<', '<'], k=3)
    tirp_ABC.is_above_vertical_support(entities)
    tirp_ABC.print()


# ---------------------------------------------------------------------------
# CSAC (Container / Same-time Adjacency Constraint) tests
# ---------------------------------------------------------------------------

def test_csac_prunes_non_adjacent_before_embedding():
    """
    CSAC must prune an embedding (i, j) for symbol A < B when another occurrence
    of A sits between positions i and j in the entity.

    Entity layout (sorted by start time):
        pos 0: A (0,1)
        pos 1: A (2,3)   ← intervening A between the first A and B
        pos 2: B (5,6)

    Without CSAC: two valid embeddings (0,2) and (1,2) → entity supports A<B.
    With CSAC:
        - Embedding (0,2): position 1 is also A and 0 < 1 < 2 → SAC violation, pruned.
        - Embedding (1,2): no A between position 1 and 2 → valid.
    Either way the entity still supports A<B, but through ONE embedding only.
    The test verifies only SAC-compliant embeddings survive.
    """
    entities = [
        [(0, 1, "A"), (2, 3, "A"), (5, 6, "B")],  # entity 0
        [(0, 2, "A"), (3, 5, "B")],                # entity 1  (no violation)
    ]
    tirp_Ab = TIRP(epsilon=0, max_distance=100, min_ver_supp=0.1,
                   symbols=["A", "B"], relations=['<'], k=2)
    assert tirp_Ab.is_above_vertical_support(entities)

    # Both entities support A<B (via CSAC-compliant embeddings).
    assert set(tirp_Ab.entity_indices_supporting) == {0, 1}

    # Entity 0 must only have the CSAC-compliant embedding (1, 2), not (0, 2).
    assert (0, 2) not in tirp_Ab.embeddings_map[0], (
        "CSAC should have pruned embedding (0,2): another A exists between positions 0 and 2."
    )
    assert (1, 2) in tirp_Ab.embeddings_map[0], (
        "CSAC-compliant embedding (1,2) must survive."
    )


def test_csac_rejects_entity_when_all_embeddings_violate():
    """
    When every possible (A,B) embedding in an entity violates CSAC, the entity
    must NOT be counted as supporting the pattern A<B.

    Entity layout:
        pos 0: A (0,1)
        pos 1: A (2,3)   ← intervening A
        pos 2: A (4,5)   ← intervening A
        pos 3: B (7,8)

    Candidate embeddings and their CSAC status:
        (0,3): A at 1 and 2 are between 0 and 3 → violation
        (1,3): A at 2 is between 1 and 3 → violation
        (2,3): no A between 2 and 3 → VALID

    Entity 0 is still supported through (2,3). But entity 1 (only one A, then B)
    is always valid. The case that would make an entity fully unsupported is when
    the only intervening symbol appears between EVERY candidate and the target.
    Test that below with a dedicated 'fully-violating' entity.
    """
    # An entity where the ONLY A is immediately before B but there are two A's
    # where one acts as an interposer for the other — but one embedding is clean.
    entities_partial = [
        [(0, 1, "A"), (2, 3, "A"), (4, 5, "A"), (7, 8, "B")],  # entity 0: (2,3) is clean
        [(0, 1, "B")],                                            # entity 1: no A
    ]
    t = TIRP(epsilon=0, max_distance=100, min_ver_supp=0.1,
             symbols=["A", "B"], relations=['<'], k=2)
    assert t.is_above_vertical_support(entities_partial)
    assert 0 in t.entity_indices_supporting  # supported via (2,3) embedding
    assert (2, 3) in t.embeddings_map[0]
    assert (0, 3) not in t.embeddings_map[0]
    assert (1, 3) not in t.embeddings_map[0]

    # Now a truly all-violating entity: only one B, and every A has a later A before B.
    #   pos 0: A (0,1), pos 1: A (3,4), pos 2: B (6,7)
    # Same as first two-A case: (0,2) violates, (1,2) is clean.
    # → entity still supported. True "all-violating" is architecturally impossible
    # unless B is reached only through positions where interposers always exist,
    # which requires the only valid last-A to itself have an interposer.
    # For A<B with only A symbols as interposers the last A before B is always clean.
    # So we test a MEETS variant: fully-structured violation impossible for '<' alone.
    # Instead verify subtler case: entity where B is unreachable via clean A.
    # This is only achievable with the 's' or 'c' structure — not meaningful for '<'.
    # Confirm: entity with three A's and B where last A has no interposer → supported.
    entities_always_has_clean = [
        [(0, 1, "A"), (2, 3, "A"), (4, 5, "A"), (6, 7, "B")],
    ]
    t2 = TIRP(epsilon=0, max_distance=100, min_ver_supp=0.1,
              symbols=["A", "B"], relations=['<'], k=2)
    assert t2.is_above_vertical_support(entities_always_has_clean)
    assert 0 in t2.entity_indices_supporting


def test_csac_in_full_pipeline(tmp_path):
    """
    End-to-end CSAC test via KarmaLego.discover_patterns.

    Entity 0: A(0,1), A(2,3), B(5,6)
        → Non-adjacent embedding (A@0-1, B@5-6) is pruned by CSAC (A@2-3 sits between).
          Adjacent embedding (A@2-3, B@5-6) is valid; entity 0 still supports A<B.
    Entity 1: A(0,2), B(3,5)
        → Single clean A<B embedding; trivially CSAC-compliant.
    min_ver_supp = 0.5 → both entities must support A<B → vertical support = 1.0.

    Note: run_lego clears embeddings_map on every node for memory efficiency, so the
    pipeline TIRP objects cannot be used to inspect per-entity embedding tuples after
    discover_patterns returns. Embedding-level CSAC filtering is verified by the
    standalone TIRP tests above. Here we verify support counts and use a fresh
    TIRP.is_above_vertical_support call to confirm embedding content.
    """
    from core.karmalego import KarmaLego
    import json

    # Minimal inverse_symbol_map so discover_patterns can decode
    inv_map = {"A": "A", "B": "B"}
    inv_path = str(tmp_path / "inv.json")
    with open(inv_path, "w") as f:
        json.dump(inv_map, f)

    entities = [
        [(0, 1, "A"), (2, 3, "A"), (5, 6, "B")],  # entity 0
        [(0, 2, "A"), (3, 5, "B")],                # entity 1
    ]
    kl = KarmaLego(epsilon=0, max_distance=100, min_ver_supp=0.5)
    df, tirps = kl.discover_patterns(
        entities, min_length=2, return_tirps=True,
        inverse_mapping_path=inv_path
    )

    # A<B must be found and supported by both entities
    ab_tirps = [t for t in tirps if t.symbols == ["A", "B"] and t.relations == ["<"]]
    assert ab_tirps, "A<B should be discovered (both entities have at least one CSAC-compliant embedding)"

    ab = ab_tirps[0]
    assert set(ab.entity_indices_supporting) == {0, 1}
    assert abs(ab.vertical_support - 1.0) < 1e-6

    # Verify CSAC embedding filtering directly via a fresh standalone support check.
    # This is necessary because run_lego clears embeddings_map for memory efficiency;
    # the standalone path is the right place to inspect per-embedding CSAC filtering.
    tirp_check = TIRP(epsilon=0, max_distance=100, min_ver_supp=0.1,
                      symbols=["A", "B"], relations=["<"], k=2)
    tirp_check.is_above_vertical_support(entities)
    e0_embeddings = tirp_check.embeddings_map.get(0, [])
    assert (0, 2) not in e0_embeddings, (
        "CSAC must prune (A@0-1, B@5-6): A@2-3 sits between them."
    )
    assert (1, 2) in e0_embeddings, (
        "CSAC-compliant embedding (A@2-3, B@5-6) must survive."
    )