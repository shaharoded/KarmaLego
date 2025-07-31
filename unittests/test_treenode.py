import pytest
from core.karmalego import TreeNode  # adjust if your module path differs


class DummyTIRP:
    """Minimal stand-in for a TIRP-like object used in tests."""
    def __init__(self, name, vertical_support=0.5, k=1):
        self.name = name
        self.vertical_support = vertical_support
        self.k = k

    def __repr__(self):
        return f"<DummyTIRP {self.name} sup={self.vertical_support} k={self.k}>"


@pytest.fixture
def small_tree():
    """
    Build a small tree:
        root
         ├── A (TIRP k=1)
         │     └── A1 (TIRP k=2)
         └── B (TIRP k=1)
    """
    root = TreeNode(data="root")
    tirp_a = DummyTIRP("A", vertical_support=0.9, k=1)
    node_a = TreeNode(data=tirp_a)
    tirp_a1 = DummyTIRP("A1", vertical_support=0.8, k=2)
    node_a1 = TreeNode(data=tirp_a1)
    node_a.add_child(node_a1)

    tirp_b = DummyTIRP("B", vertical_support=0.3, k=1)
    node_b = TreeNode(data=tirp_b)

    root.add_child(node_a)
    root.add_child(node_b)
    return root, tirp_a, tirp_a1, tirp_b


def test_add_and_parent_relationship(small_tree):
    root, tirp_a, tirp_a1, tirp_b = small_tree
    children = root.children
    assert len(children) == 2
    node_a = children[0]
    node_b = children[1]

    assert node_a.parent is root
    assert node_b.parent is root
    assert node_a.children[0].parent is node_a
    assert node_a.depth == 1
    assert node_b.depth == 1
    assert node_a.children[0].depth == 2


def test_find_tree_nodes_default_filter(small_tree):
    root, tirp_a, tirp_a1, tirp_b = small_tree
    collected = root.find_tree_nodes()
    names = {t.name for t in collected}
    assert names == {"A", "A1", "B"}


def test_find_tree_nodes_custom_filter(small_tree):
    root, tirp_a, tirp_a1, tirp_b = small_tree
    collected_k1 = root.find_tree_nodes(filter_fn=lambda d: getattr(d, "k", None) == 1)
    names = {t.name for t in collected_k1}
    assert names == {"A", "B"}


def test_cache_and_invalidation(small_tree):
    root, tirp_a, tirp_a1, tirp_b = small_tree
    c1 = root.find_tree_nodes()
    c2 = root.find_tree_nodes()
    assert c1 == c2

    new_tirp = DummyTIRP("B1", vertical_support=0.6, k=2)
    node_b = root.children[1]
    node_b.add_child(TreeNode(data=new_tirp))

    updated = root.find_tree_nodes()
    names = {t.name for t in updated}
    assert "B1" in names
    assert len(updated) == 4


def test_remove_child(small_tree):
    root, tirp_a, tirp_a1, tirp_b = small_tree
    node_a = root.children[0]
    a1_node = node_a.children[0]
    node_a.remove_child(a1_node)
    assert a1_node.parent is None
    collected = root.find_tree_nodes()
    names = {t.name for t in collected}
    assert "A1" not in names
    assert len(node_a.children) == 0