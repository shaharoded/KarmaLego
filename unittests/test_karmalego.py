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
    p1 = [(0, 1, "A"), (2, 3, "B")]  # A < B
    p2 = [(0, 2, "A"), (1, 3, "B")]  # A o B
    p3 = [(0, 1, "A"), (2, 3, "C")]  # A < C
    entities = [p1, p2, p3]
    patient_ids = ["p1", "p2", "p3"]
    return entities, patient_ids


def extract_pattern_by_signature(patterns, symbols, relations):
    """
    Find a TIRP by exact (symbols, relations).
    Supports list[TIRP] or DataFrame with column 'tirp_obj'.
    """
    # normalize to list of TIRPs
    if hasattr(patterns, "iterrows"):
        tirp_list = [row["tirp_obj"] for _, row in patterns.iterrows()]
    else:
        tirp_list = list(patterns)
    for p in tirp_list:
        if p.symbols == symbols and tuple(p.relations) == tuple(relations):
            return p
    return None


def test_full_pipeline_and_apply_modes(simple_patient_entities):
    entities, patient_ids = simple_patient_entities
    kl = KarmaLego(epsilon=0, max_distance=100, min_ver_supp=1/3)

    # Discover patterns: df + list
    df, tirps = kl.discover_patterns(entities, min_length=1, return_tirps=True)

    # Expect singletons
    tA = extract_pattern_by_signature(tirps, ["A"], [])
    tB = extract_pattern_by_signature(tirps, ["B"], [])
    tC = extract_pattern_by_signature(tirps, ["C"], [])
    assert tA and tB and tC

    # Expect pairs ('<' before, 'o' overlaps)
    tA_B_before = extract_pattern_by_signature(tirps, ["A", "B"], ["<"])
    tA_B_overlap = extract_pattern_by_signature(tirps, ["A", "B"], ["o"])
    tA_C_before = extract_pattern_by_signature(tirps, ["A", "C"], ["<"])
    assert tA_B_before and tA_B_overlap and tA_C_before

    # Vertical support: each pair appears in exactly one patient (1/3)
    import math
    assert math.isclose(tA_B_before.vertical_support, 1/3, rel_tol=1e-6)
    assert math.isclose(tA_B_overlap.vertical_support, 1/3, rel_tol=1e-6)
    assert math.isclose(tA_C_before.vertical_support, 1/3, rel_tol=1e-6)

    # ---- Apply: tirp-count (unique_last) ----
    applied_count_ul = kl.apply_patterns_to_entities(
        entities, df, patient_ids, mode="tirp-count", count_strategy="unique_last"
    )
    # p1 has A, B, A<B
    assert applied_count_ul["p1"][repr(tA)] == 1
    assert applied_count_ul["p1"][repr(tB)] == 1
    assert applied_count_ul["p1"][repr(tA_B_before)] == 1
    assert repr(tA_B_overlap) not in applied_count_ul["p1"]
    # p2 has A, B, A o B
    assert applied_count_ul["p2"][repr(tA)] == 1
    assert applied_count_ul["p2"][repr(tB)] == 1
    assert applied_count_ul["p2"][repr(tA_B_overlap)] == 1
    assert repr(tA_B_before) not in applied_count_ul["p2"]
    # p3 has A, C, A<C
    assert applied_count_ul["p3"][repr(tA)] == 1
    assert applied_count_ul["p3"][repr(tC)] == 1
    assert applied_count_ul["p3"][repr(tA_C_before)] == 1
    assert repr(tA_B_before) not in applied_count_ul["p3"]
    assert repr(tA_B_overlap) not in applied_count_ul["p3"]

    # ---- Apply: tirp-count (all) should match here (only one embedding each) ----
    applied_count_all = kl.apply_patterns_to_entities(
        entities, df, patient_ids, mode="tirp-count", count_strategy="all"
    )
    assert applied_count_all == applied_count_ul

    # ---- Apply: tpf-dist (min–max across cohort per pattern) ----
    applied_tpf_dist = kl.apply_patterns_to_entities(
        entities, df, patient_ids, mode="tpf-dist", count_strategy="unique_last"
    )
    # Each pair appears in exactly one patient -> normalized to 1 for that patient, 0 for others
    assert applied_tpf_dist["p1"].get(repr(tA_B_before), 0.0) == 1.0
    assert applied_tpf_dist["p2"].get(repr(tA_B_before), 0.0) == 0.0
    assert applied_tpf_dist["p3"].get(repr(tA_B_before), 0.0) == 0.0

    assert applied_tpf_dist["p2"].get(repr(tA_B_overlap), 0.0) == 1.0
    assert applied_tpf_dist["p1"].get(repr(tA_B_overlap), 0.0) == 0.0
    assert applied_tpf_dist["p3"].get(repr(tA_B_overlap), 0.0) == 0.0

    assert applied_tpf_dist["p3"].get(repr(tA_C_before), 0.0) == 1.0
    assert applied_tpf_dist["p1"].get(repr(tA_C_before), 0.0) == 0.0
    assert applied_tpf_dist["p2"].get(repr(tA_C_before), 0.0) == 0.0

    # Singletons: A appears in all 3 patients with count=1 → min=max → normalized to 0.0 per our rule
    assert applied_tpf_dist["p1"].get(repr(tA), 0.0) == 0.0
    assert applied_tpf_dist["p2"].get(repr(tA), 0.0) == 0.0
    assert applied_tpf_dist["p3"].get(repr(tA), 0.0) == 0.0
    # B appears in p1 & p2 only -> [1,1,0] → normalized [1,1,0]
    assert applied_tpf_dist["p1"].get(repr(tB), 0.0) == 1.0
    assert applied_tpf_dist["p2"].get(repr(tB), 0.0) == 1.0
    assert applied_tpf_dist["p3"].get(repr(tB), 0.0) == 0.0
    # C appears only in p3 -> [0,0,1] → normalized [0,0,1]
    assert applied_tpf_dist["p1"].get(repr(tC), 0.0) == 0.0
    assert applied_tpf_dist["p2"].get(repr(tC), 0.0) == 0.0
    assert applied_tpf_dist["p3"].get(repr(tC), 0.0) == 1.0

    # ---- Apply: tpf-duration (union span per patient, then min–max per pattern) ----
    applied_tpf_dur = kl.apply_patterns_to_entities(
        entities, df, patient_ids, mode="tpf-duration", count_strategy="unique_last"
    )
    # For pairs, each span is 3 (end_last - start_first) for the patient that has it, then normalized to 1
    assert applied_tpf_dur["p1"].get(repr(tA_B_before), 0.0) == 1.0
    assert applied_tpf_dur["p2"].get(repr(tA_B_before), 0.0) == 0.0
    assert applied_tpf_dur["p3"].get(repr(tA_B_before), 0.0) == 0.0

    assert applied_tpf_dur["p2"].get(repr(tA_B_overlap), 0.0) == 1.0
    assert applied_tpf_dur["p1"].get(repr(tA_B_overlap), 0.0) == 0.0
    assert applied_tpf_dur["p3"].get(repr(tA_B_overlap), 0.0) == 0.0

    assert applied_tpf_dur["p3"].get(repr(tA_C_before), 0.0) == 1.0
    assert applied_tpf_dur["p1"].get(repr(tA_C_before), 0.0) == 0.0
    assert applied_tpf_dur["p2"].get(repr(tA_C_before), 0.0) == 0.0

    # Singletons: spans then per-pattern min–max
    # A spans: [1, 2, 1] -> [0.0, 1.0, 0.0]
    assert applied_tpf_dur["p1"].get(repr(tA), 0.0) == pytest.approx(0.0)
    assert applied_tpf_dur["p2"].get(repr(tA), 0.0) == pytest.approx(1.0)
    assert applied_tpf_dur["p3"].get(repr(tA), 0.0) == pytest.approx(0.0)

    # B spans: [1, 2, 0] -> [(1-0)/(2-0)=0.5, 1.0, 0.0]
    assert applied_tpf_dur["p1"].get(repr(tB), 0.0) == pytest.approx(0.5)
    assert applied_tpf_dur["p2"].get(repr(tB), 0.0) == pytest.approx(1.0)
    assert applied_tpf_dur["p3"].get(repr(tB), 0.0) == pytest.approx(0.0)

    # C spans: [0, 0, 1] -> [0.0, 0.0, 1.0]
    assert applied_tpf_dur["p1"].get(repr(tC), 0.0) == pytest.approx(0.0)
    assert applied_tpf_dur["p2"].get(repr(tC), 0.0) == pytest.approx(0.0)
    assert applied_tpf_dur["p3"].get(repr(tC), 0.0) == pytest.approx(1.0)


def test_count_strategy_unique_last_vs_all_for_ABC_case():
    """
    Single patient with A…B…A…B…C.
    For pattern A<B<C:
      - unique_last should count 1
      - all should count 3
    """
    p = [(0,1,"A"), (2,3,"B"), (4,5,"A"), (6,7,"B"), (8,9,"C")]
    entities = [p]
    patient_ids = ["p1"]
    kl = KarmaLego(epsilon=0, max_distance=100, min_ver_supp=1.0)

    # Discover and grab A<B<C
    df, tirps = kl.discover_patterns(entities, min_length=1, return_tirps=True)
    target = extract_pattern_by_signature(tirps, ["A","B","C"], ["<","<", "<"])
    assert target is not None

    # Apply with unique_last
    v_ul = kl.apply_patterns_to_entities(entities, [target], patient_ids,
                                         mode="tirp-count", count_strategy="unique_last")
    # Apply with all
    v_all = kl.apply_patterns_to_entities(entities, [target], patient_ids,
                                          mode="tirp-count", count_strategy="all")

    assert v_ul["p1"][repr(target)] == 1
    assert v_all["p1"][repr(target)] == 3


@pytest.fixture
def simple_chain_entities():
    """
    Three patients with a clear chain A -> B -> C.
    This guarantees patterns of length 1 (A, B, C), 2 (A-B, B-C, A-C), and 3 (A-B-C).
    """
    # Patient 1: A(0,1), B(2,3), C(4,5)
    p1 = [(0, 1, "A"), (2, 3, "B"), (4, 5, "C")]
    # Patient 2: A(0,1), B(2,3), C(4,5)
    p2 = [(0, 1, "A"), (2, 3, "B"), (4, 5, "C")]
    # Patient 3: A(0,1), B(2,3), C(4, 5)
    p3 = [(0, 1, "A"), (2, 3, "B"), (4, 5, "C")]
    
    entities = [p1, p2, p3]
    return entities

def test_min_length_constraint(simple_chain_entities):
    """Test that min_length filters out short patterns."""
    kl = KarmaLego(epsilon=0, max_distance=100, min_ver_supp=1.0)
    
    # min_length=2 should exclude singletons (k=1)
    df, tirps = kl.discover_patterns(simple_chain_entities, min_length=2, return_tirps=True)
    
    ks = [t.k for t in tirps]
    assert all(k >= 2 for k in ks), f"Found patterns with k < 2: {ks}"
    assert 2 in ks, "Should find length 2 patterns"
    assert 3 in ks, "Should find length 3 patterns"

def test_max_length_constraint(simple_chain_entities):
    """Test that max_length stops extension and filters results."""
    kl = KarmaLego(epsilon=0, max_distance=100, min_ver_supp=1.0)
    
    # max_length=2 should exclude A-B-C (k=3)
    df, tirps = kl.discover_patterns(simple_chain_entities, min_length=1, max_length=2, return_tirps=True)
    
    ks = [t.k for t in tirps]
    assert all(k <= 2 for k in ks), f"Found patterns with k > 2: {ks}"
    assert 1 in ks, "Should find length 1 patterns"
    assert 2 in ks, "Should find length 2 patterns"
    assert 3 not in ks, "Should NOT find length 3 patterns"

def test_min_and_max_length_together(simple_chain_entities):
    """Test both constraints simultaneously."""
    kl = KarmaLego(epsilon=0, max_distance=100, min_ver_supp=1.0)
    
    # min=2, max=2 -> Only length 2 patterns
    df, tirps = kl.discover_patterns(simple_chain_entities, min_length=2, max_length=2, return_tirps=True)
    
    ks = [t.k for t in tirps]
    assert all(k == 2 for k in ks), f"Found patterns with k != 2: {ks}"
    assert len(ks) > 0, "Should find some patterns"

def test_max_length_pruning_efficiency(simple_chain_entities):
    """
    Verify that max_length actually prevents deeper search (pruning),
    not just filtering the output.
    We check this by inspecting the returned tree depth if possible, 
    or relying on the fact that k=3 patterns are not generated.
    """
    kl = KarmaLego(epsilon=0, max_distance=100, min_ver_supp=1.0)
    
    # If we limit to k=1, the Lego phase shouldn't even generate k=2 candidates
    # (Note: Karma phase generates k=2 by default, but Lego shouldn't extend them if max_length=2)
    # Actually, Karma generates k=2. Lego extends from there.
    # If max_length=1, we filter at the end.
    # If max_length=2, Lego shouldn't extend k=2 nodes to k=3.
    
    tree = kl.discover_patterns(simple_chain_entities, min_length=1, max_length=2, return_tree=True)[1]
    
    # Traverse tree to find max depth
    max_depth = 0
    stack = [tree]
    while stack:
        node = stack.pop()
        # Root node usually has string data or None; skip checking 'k' on it
        if node.data and not isinstance(node.data, str):
            max_depth = max(max_depth, node.data.k)
        stack.extend(node.children)
    assert max_depth <= 2, f"Tree contains nodes deeper than max_length=2 (found k={max_depth})"