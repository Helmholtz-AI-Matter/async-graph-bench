import pytest
from async_graph_bench import NodeConfig, CSVDataStore
from async_graph_bench.acyclic_directed_graph import AcyclicDirectedGraph
from conftest import MockNode, SimpleMockDataSource


class TestAcyclicDirectedGraphEmpty:
    def test_create_empty(self):
        ds = SimpleMockDataSource(5)
        adg = AcyclicDirectedGraph(ds, [])
        assert adg.data_source == ds
        assert len(adg.optional_nodes_configs) == 0
        assert len(adg.greedy_nodes_configs) == 0

    def test_build_empty_graph(self):
        ds = SimpleMockDataSource(5)
        adg = AcyclicDirectedGraph(ds, [])
        adg.build_graph()
        assert len(adg.unreachable_nodes) == 0

    def test_treeshake_empty(self):
        ds = SimpleMockDataSource(5)
        adg = AcyclicDirectedGraph(ds, [])
        removed = adg.treeshake()
        assert removed == []


class TestAcyclicDirectedGraphSingle:
    def test_single_greedy_consumer(self):
        ds = SimpleMockDataSource(5)
        node = MockNode(requires=["text"])
        cfg = NodeConfig(node, greedy=True, data_store=CSVDataStore)
        adg = AcyclicDirectedGraph(ds, [cfg])
        adg.build_graph()
        assert cfg not in adg.unreachable_nodes

    def test_single_optional(self):
        ds = SimpleMockDataSource(5)
        node = MockNode(requires=["text"], provides=["output"])
        cfg = NodeConfig(node)
        adg = AcyclicDirectedGraph(ds, [cfg])
        adg.build_graph()
        assert cfg not in adg.unreachable_nodes


class TestAcyclicDirectedGraphChain:
    def test_simple_chain(self):
        ds = SimpleMockDataSource(5)
        n1 = MockNode(requires=["text"], provides=["step1"])
        cfg1 = NodeConfig(n1)
        n2 = MockNode(requires=["step1"], provides=["step2"])
        cfg2 = NodeConfig(n2)
        consumer = MockNode(requires=["step2"])
        cfg_c = NodeConfig(consumer, greedy=True, data_store=CSVDataStore)
        adg = AcyclicDirectedGraph(ds, [cfg1, cfg2, cfg_c])
        adg.build_graph()
        assert cfg1 not in adg.unreachable_nodes
        assert cfg2 not in adg.unreachable_nodes
        assert cfg_c not in adg.unreachable_nodes

    def test_unreachable_node(self):
        ds = SimpleMockDataSource(5)
        consumer = MockNode(requires=["nonexistent"])
        cfg_c = NodeConfig(consumer, greedy=True, data_store=CSVDataStore)
        adg = AcyclicDirectedGraph(ds, [cfg_c])
        adg.build_graph()
        assert cfg_c in adg.unreachable_nodes


class TestAcyclicDirectedGraphMultiParent:
    def test_multi_parent(self):
        ds = SimpleMockDataSource(5)
        n1 = MockNode(requires=["text"], provides=["a"])
        cfg1 = NodeConfig(n1)
        n2 = MockNode(requires=["value"], provides=["b"])
        cfg2 = NodeConfig(n2)
        merger = MockNode(requires=["a", "b"], provides=["merged"])
        cfg_m = NodeConfig(merger)
        consumer = MockNode(requires=["merged"])
        cfg_c = NodeConfig(consumer, greedy=True, data_store=CSVDataStore)
        adg = AcyclicDirectedGraph(ds, [cfg1, cfg2, cfg_m, cfg_c])
        adg.build_graph()
        assert cfg1 not in adg.unreachable_nodes
        assert cfg2 not in adg.unreachable_nodes
        assert cfg_m not in adg.unreachable_nodes


class TestAcyclicDirectedGraphPredecessors:
    def test_get_all_predecessors(self):
        ds = SimpleMockDataSource(5)
        n1 = MockNode(requires=["text"], provides=["a"])
        cfg1 = NodeConfig(n1)
        n2 = MockNode(requires=["a"])
        cfg2 = NodeConfig(n2, greedy=True, data_store=CSVDataStore)
        adg = AcyclicDirectedGraph(ds, [cfg1, cfg2])
        adg.build_graph()
        preds = adg.get_all_predecessors(cfg2)
        assert cfg1 in preds or ds in preds


class TestAcyclicDirectedGraphSuccessors:
    def test_get_all_successors(self):
        ds = SimpleMockDataSource(5)
        n1 = MockNode(requires=["text"], provides=["a"])
        cfg1 = NodeConfig(n1, id="proc")
        n2 = MockNode(requires=["a"])
        cfg2 = NodeConfig(n2, id="consumer", greedy=True, data_store=CSVDataStore)
        adg = AcyclicDirectedGraph(ds, [cfg1, cfg2])
        adg.build_graph()
        succs = adg.get_all_successors(cfg1)
        assert cfg2 in succs


class TestAcyclicDirectedGraphSafeEdges:
    def test_safe_edge_targets_no_cycle(self):
        ds = SimpleMockDataSource(5)
        n1 = MockNode(requires=["text"], provides=["a"])
        cfg1 = NodeConfig(n1)
        n2 = MockNode(requires=["a"])
        cfg2 = NodeConfig(n2)
        adg = AcyclicDirectedGraph(ds, [cfg1, cfg2])
        adg.edges[cfg1].add(cfg2)
        safe = adg.get_safe_edge_targets(cfg1)
        assert cfg2 in safe

    def test_safe_edge_sources_no_cycle(self):
        ds = SimpleMockDataSource(5)
        n1 = MockNode(requires=["text"], provides=["a"])
        cfg1 = NodeConfig(n1)
        n2 = MockNode(requires=["a"])
        cfg2 = NodeConfig(n2)
        adg = AcyclicDirectedGraph(ds, [cfg1, cfg2])
        adg.edges[cfg1].add(cfg2)
        safe = adg.get_safe_edge_sources(cfg2)
        assert cfg1 in safe


class TestAcyclicDirectedGraphTreeshake:
    def test_prune_unused_optional(self):
        ds = SimpleMockDataSource(5)
        n1 = MockNode(requires=["text"], provides=["a"])
        cfg1 = NodeConfig(n1)
        consumer = MockNode(requires=["text"])
        cfg_c = NodeConfig(consumer, greedy=True, data_store=CSVDataStore)
        adg = AcyclicDirectedGraph(ds, [cfg1, cfg_c])
        adg.build_graph()
        removed = adg.treeshake()
        assert cfg1 in removed

    def test_no_pruning_when_used(self):
        ds = SimpleMockDataSource(5)
        n1 = MockNode(requires=["text"], provides=["out"])
        cfg1 = NodeConfig(n1)
        consumer = MockNode(requires=["out"])
        cfg_c = NodeConfig(consumer, greedy=True, data_store=CSVDataStore)
        adg = AcyclicDirectedGraph(ds, [cfg1, cfg_c])
        adg.build_graph()
        removed = adg.treeshake()
        assert cfg1 not in removed

    def test_pruned_nodes_tracked(self):
        ds = SimpleMockDataSource(5)
        n1 = MockNode(requires=["text"], provides=["a"])
        cfg1 = NodeConfig(n1)
        consumer = MockNode(requires=["text"])
        cfg_c = NodeConfig(consumer, greedy=True, data_store=CSVDataStore)
        adg = AcyclicDirectedGraph(ds, [cfg1, cfg_c])
        adg.build_graph()
        adg.treeshake()
        assert cfg1 in adg.pruned_nodes


class TestAcyclicDirectedGraphCopy:
    def test_copy_preserves_edges(self):
        ds = SimpleMockDataSource(5)
        n1 = MockNode(requires=["text"], provides=["a"])
        cfg1 = NodeConfig(n1)
        n2 = MockNode(requires=["a"])
        cfg2 = NodeConfig(n2, greedy=True, data_store=CSVDataStore)
        adg = AcyclicDirectedGraph(ds, [cfg1, cfg2])
        adg.build_graph()
        copy_adg = adg.copy()
        assert len(copy_adg.edges) == len(adg.edges)
        assert copy_adg is not adg

    def test_copy_independent(self):
        ds = SimpleMockDataSource(5)
        adg = AcyclicDirectedGraph(ds, [])
        copy_adg = adg.copy()
        copy_adg.optional_nodes_configs.add(None)
        assert None not in adg.optional_nodes_configs


class TestAcyclicDirectedGraphRemoveNode:
    def test_remove_node(self):
        ds = SimpleMockDataSource(5)
        n1 = MockNode(requires=["text"], provides=["a"])
        cfg1 = NodeConfig(n1)
        n2 = MockNode(requires=["a"])
        cfg2 = NodeConfig(n2)
        adg = AcyclicDirectedGraph(ds, [cfg1, cfg2])
        adg.build_graph()
        adg.remove_node(cfg1)
        assert cfg1 not in adg.optional_nodes_configs
        assert cfg1 not in adg.greedy_nodes_configs


class TestAcyclicDirectedGraphTopological:
    def test_topological_order(self):
        ds = SimpleMockDataSource(5)
        n1 = MockNode(requires=["text"], provides=["a"])
        cfg1 = NodeConfig(n1, id="proc")
        n2 = MockNode(requires=["a"])
        cfg2 = NodeConfig(n2, id="consumer", greedy=True, data_store=CSVDataStore)
        adg = AcyclicDirectedGraph(ds, [cfg1, cfg2])
        adg.build_graph()
        nodes, edges = adg.get_nodes_and_edges_in_topological_order()
        assert len(nodes) == 2
        assert cfg1 in nodes
        assert cfg2 in nodes

    def test_deterministic_order(self):
        ds = SimpleMockDataSource(5)
        n1 = MockNode(requires=["text"], provides=["b"])
        cfg1 = NodeConfig(n1, id="b_node")
        n2 = MockNode(requires=["text"], provides=["a"])
        cfg2 = NodeConfig(n2, id="a_node")
        consumer = MockNode(requires=["a", "b"])
        cfg_c = NodeConfig(consumer, greedy=True, data_store=CSVDataStore)
        adg = AcyclicDirectedGraph(ds, [cfg1, cfg2, cfg_c])
        adg.build_graph()
        nodes, edges = adg.get_nodes_and_edges_in_topological_order()
        assert len(nodes) == 3


class TestAcyclicDirectedGraphCountParents:
    def test_count_parents(self):
        ds = SimpleMockDataSource(5)
        n1 = MockNode(requires=["text"], provides=["a"])
        cfg1 = NodeConfig(n1)
        n2 = MockNode(requires=["value"], provides=["b"])
        cfg2 = NodeConfig(n2)
        merger = MockNode(requires=["a", "b"])
        cfg_m = NodeConfig(merger, greedy=True, data_store=CSVDataStore)
        adg = AcyclicDirectedGraph(ds, [cfg1, cfg2, cfg_m])
        adg.build_graph()
        adg.treeshake()
        parents = adg.count_parents(cfg_m)
        assert parents == 2


class TestAcyclicDirectedGraphIsLeaf:
    def test_leaf_node(self):
        ds = SimpleMockDataSource(5)
        consumer = MockNode(requires=["text"])
        cfg = NodeConfig(consumer, greedy=True, data_store=CSVDataStore)
        adg = AcyclicDirectedGraph(ds, [cfg])
        adg.build_graph()
        assert adg.is_leaf_node(cfg) is True


class TestAcyclicDirectedGraphTurnNonGreedy:
    def test_turn_non_greedy(self):
        ds = SimpleMockDataSource(5)
        consumer = MockNode(requires=["text"])
        cfg = NodeConfig(consumer, greedy=True, data_store=CSVDataStore)
        adg = AcyclicDirectedGraph(ds, [cfg])
        adg.build_graph()
        adg.turn_consumer_non_greedy(cfg)
        assert cfg not in adg.greedy_nodes_configs
        assert cfg in adg.optional_nodes_configs


class TestAcyclicDirectedGraphRemoveTransientEdges:
    def test_no_transient_with_two_independent(self):
        ds = SimpleMockDataSource(5)
        n1 = MockNode(requires=["text"], provides=["a"])
        cfg1 = NodeConfig(n1)
        n2 = MockNode(requires=["value"], provides=["b"])
        cfg2 = NodeConfig(n2)
        merger = MockNode(requires=["a", "b"])
        cfg_m = NodeConfig(merger, greedy=True, data_store=CSVDataStore)
        adg = AcyclicDirectedGraph(ds, [cfg1, cfg2, cfg_m])
        adg.build_graph()
        adg.remove_transient_edges()
        assert cfg_m in adg.edges[cfg1]
        assert cfg_m in adg.edges[cfg2]