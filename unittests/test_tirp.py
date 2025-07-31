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
    tirp_Ab = TIRP(epsilon=0, max_distance=100, min_ver_supp=0.5, symbols=["A", "B"], relations=["b"], k=2)
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

    tirp_Ab = TIRP(epsilon=0, max_distance=100, min_ver_supp=0.1, symbols=["A", "B"], relations=["b"], k=2)
    assert tirp_Ab.is_above_vertical_support(entities) is True
    print("\n=== Multiple embeddings deduplication (A before B) ===")
    print("Summary:", tirp_Ab)
    tirp_Ab.print()
    assert set(tirp_Ab.entity_indices_supporting) == {0}
    assert pytest.approx(tirp_Ab.vertical_support, rel=1e-3) == 1 / 3


def test_indices_of_last_symbol_alignment(complex_entities):
    entities = complex_entities
    tirp_Ab = TIRP(epsilon=0, max_distance=100, min_ver_supp=0.1, symbols=["A", "B"], relations=["b"], k=2)
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

    child_Ab = TIRP(epsilon=0, max_distance=100, min_ver_supp=0.0, symbols=["A", "B"], relations=["b"], k=2)
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
    child.extend("B", ["b"])
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
    t3 = TIRP(epsilon=0, max_distance=10, min_ver_supp=0.1, symbols=["X", "Y"], relations=["b"], k=2)

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

    # Correct triple: all relations are 'before' => ["b","b","b"] corresponds to
    # (A,B), (A,C), (B,C)
    tirp_ABC = TIRP(epsilon=0, max_distance=100, min_ver_supp=0.5,
                    symbols=["A", "B", "C"], relations=["b", "b", "b"], k=3)
    assert tirp_ABC.is_above_vertical_support(entities) is True
    print("\n=== Length-3 TIRP A before B before C ===")
    print("Summary:", tirp_ABC)
    tirp_ABC.print()
    # Support should be 2/3 (entities 0 and 1)
    assert pytest.approx(tirp_ABC.vertical_support, rel=1e-3) == 2 / 3
    assert set(tirp_ABC.entity_indices_supporting) == {0, 1}

    # Negative: last relation is wrong (B overlaps C instead of before), should fail entirely
    tirp_ABC_bad = TIRP(epsilon=0, max_distance=100, min_ver_supp=0.1,
                        symbols=["A", "B", "C"], relations=["b", "b", "o"], k=3)
    assert not tirp_ABC_bad.is_above_vertical_support(entities)
    print("\n=== Length-3 TIRP with wrong relation (should be unsupported) ===")
    print("Summary:", tirp_ABC_bad)
    tirp_ABC_bad.print()
    assert tirp_ABC_bad.vertical_support == 0.0