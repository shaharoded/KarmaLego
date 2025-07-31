import pytest
from core.karmalego import KarmaLego

@pytest.fixture
def simple_patient_entities():
    """
    Three patients:
      - p1: A before B
      - p2: A overlaps B
      - p3: A before C
    Each entity is [(start, end, symbol), ...]
    """
    p1 = [(0, 1, "A"), (2, 3, "B")]  # A before B
    p2 = [(0, 2, "A"), (1, 3, "B")]  # A overlaps B
    p3 = [(0, 1, "A"), (2, 3, "C")]  # A before C
    entities = [p1, p2, p3]
    patient_ids = ["p1", "p2", "p3"]
    return entities, patient_ids


def extract_pattern_by_signature(patterns, symbols, relations):
    """
    Helper to find a TIRP by exact symbols & relations signature.
    Accepts either a list of TIRP objects or a DataFrame-like with 'tirp_obj' column.
    """
    tirp_list = []
    # patterns may be a DataFrame
    try:
        # pandas DataFrame or similar
        tirp_list = [row["tirp_obj"] for _, row in patterns.iterrows()]
    except Exception:
        tirp_list = list(patterns)

    for p in tirp_list:
        if p.symbols == symbols and tuple(p.relations) == tuple(relations):
            return p
    return None


def test_full_pipeline_discovery_and_application(simple_patient_entities):
    entities, patient_ids = simple_patient_entities
    kl = KarmaLego(epsilon=0, max_distance=100, min_ver_supp=1 / 3)

    # discover patterns: flat DataFrame + list of TIRPs
    df, tirps = kl.discover_patterns(entities, min_length=1, return_tirps=True)

    # Expect singletons A,B,C
    tirp_A = extract_pattern_by_signature(tirps, ["A"], [])
    tirp_B = extract_pattern_by_signature(tirps, ["B"], [])
    tirp_C = extract_pattern_by_signature(tirps, ["C"], [])
    assert tirp_A is not None, "Singleton A should be discovered"
    assert tirp_B is not None, "Singleton B should be discovered"
    assert tirp_C is not None, "Singleton C should be discovered"

    # Expect pair patterns (using relation codes from temporal_relations): '<' for before, 'o' for overlaps
    tirp_Ab = extract_pattern_by_signature(tirps, ["A", "B"], ["<"])
    tirp_Ao = extract_pattern_by_signature(tirps, ["A", "B"], ["o"])
    tirp_Ac = extract_pattern_by_signature(tirps, ["A", "C"], ["<"])

    assert tirp_Ab is not None, "Pattern A before B should be discovered"
    assert tirp_Ao is not None, "Pattern A overlaps B should be discovered"
    assert tirp_Ac is not None, "Pattern A before C should be discovered"

    # Verify vertical supports for those pair patterns: each appears exactly in one patient (1/3)
    assert pytest.approx(tirp_Ab.vertical_support, rel=1e-3) == 1 / 3
    assert pytest.approx(tirp_Ao.vertical_support, rel=1e-3) == 1 / 3
    assert pytest.approx(tirp_Ac.vertical_support, rel=1e-3) == 1 / 3

    # Apply patterns to entities (raw counts). Can pass the DataFrame directly since it has tirp_obj column.
    applied = kl.apply_patterns_to_entities(entities, df, patient_ids, tpp=False)

    # p1 has A, B, A before B
    assert applied["p1"][repr(tirp_A)] == 1
    assert applied["p1"][repr(tirp_B)] == 1
    assert applied["p1"][repr(tirp_Ab)] == 1
    assert repr(tirp_Ao) not in applied["p1"]

    # p2 has A, B, A overlaps B
    assert applied["p2"][repr(tirp_A)] == 1
    assert applied["p2"][repr(tirp_B)] == 1
    assert applied["p2"][repr(tirp_Ao)] == 1
    assert repr(tirp_Ab) not in applied["p2"]

    # p3 has A, C, A before C
    assert applied["p3"][repr(tirp_A)] == 1
    assert applied["p3"][repr(tirp_C)] == 1
    assert applied["p3"][repr(tirp_Ac)] == 1
    assert repr(tirp_Ab) not in applied["p3"]
    assert repr(tirp_Ao) not in applied["p3"]

    # TPP normalization: each patient has 3 total occurrences -> each pattern is 1/3
    applied_tpp = kl.apply_patterns_to_entities(entities, df, patient_ids, tpp=True)
    for pid in patient_ids:
        vector = applied_tpp[pid]
        total = sum(vector.values())
        assert pytest.approx(total, rel=1e-3) == 1.0
        for val in vector.values():
            assert pytest.approx(val, rel=1e-3) == 1 / 3