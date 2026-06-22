import pytest
from async_graph_bench import NodeConfig, SamplingConfig, CSVDataStore
from tests.fixtures import MockNode


class TestNodeConfig:
    def test_id_from_explicit(self):
        node = MockNode(requires=["text"])
        cfg = NodeConfig(node, id="my_custom_id")
        assert cfg.id == "my_custom_id"

    def test_id_from_node_id(self):
        node = MockNode(requires=["text"])
        node.id = "node_provided_id"
        cfg = NodeConfig(node)
        assert cfg.id == "node_provided_id"

    def test_id_from_class_name(self):
        node = MockNode(requires=["text"])
        cfg = NodeConfig(node)
        assert cfg.id == "MockNode"

    def test_id_override_precedence(self):
        node = MockNode(requires=["text"])
        node.id = "node_id"
        cfg = NodeConfig(node, id="explicit_id")
        assert cfg.id == "explicit_id"

    def test_requires_delegates(self):
        node = MockNode(requires=["a", "b"])
        cfg = NodeConfig(node)
        assert cfg.requires == ["a", "b"]

    def test_provides_with_provides(self):
        node = MockNode(requires=["text"], provides=["output"])
        cfg = NodeConfig(node)
        assert cfg.provides == ["output"]

    def test_provides_default_empty(self):
        node = MockNode(requires=["text"])
        cfg = NodeConfig(node)
        assert cfg.provides == []

    def test_description_delegates(self):
        node = MockNode(requires=["text"])
        node.description = "A test node"
        cfg = NodeConfig(node)
        assert cfg.description == "A test node"

    def test_description_default_none(self):
        node = MockNode(requires=["text"])
        cfg = NodeConfig(node)
        assert cfg.description is None

    def test_is_sampling_true(self):
        node = MockNode(requires=["sampled_value"])
        cfg = NodeConfig(
            node,
            greedy=True,
            data_store=CSVDataStore,
            sampling_config=SamplingConfig(sampling_size=5),
        )
        assert cfg.is_sampling() is True

    def test_is_sampling_false(self):
        node = MockNode(requires=["text", "value"])
        cfg = NodeConfig(node)
        assert cfg.is_sampling() is False

    def test_greedy_requires_data_store(self):
        node = MockNode(requires=["text"])
        with pytest.raises(AssertionError):
            NodeConfig(node, greedy=True)

    def test_sampling_requires_sampling_config(self):
        node = MockNode(requires=["sampled_text"])
        with pytest.raises(AssertionError):
            NodeConfig(
                node,
                greedy=True,
                data_store=CSVDataStore,
            )

    def test_base_config_inheritance(self):
        NodeConfig.base_config = {"batch_size": 50, "max_tasks": 4, "queue_size": 200}
        try:
            node = MockNode(requires=["text"])
            cfg = NodeConfig(node)
            assert cfg.batch_size == 50
            assert cfg.max_tasks == 4
            assert cfg.queue_size == 200
        finally:
            NodeConfig.base_config = None

    def test_base_config_override(self):
        NodeConfig.base_config = {"batch_size": 50, "max_tasks": 4}
        try:
            node = MockNode(requires=["text"])
            cfg = NodeConfig(node, batch_size=10, max_tasks=1)
            assert cfg.batch_size == 10
            assert cfg.max_tasks == 1
        finally:
            NodeConfig.base_config = None

    def test_str_delegates_to_node(self):
        node = MockNode(requires=["text"])
        cfg = NodeConfig(node)
        result = str(cfg)
        assert isinstance(result, str)

    def test_sampling_mode_first(self):
        node = MockNode(requires=["sampled_text"])
        cfg = NodeConfig(
            node,
            greedy=True,
            data_store=CSVDataStore,
            sampling_config=SamplingConfig(sampling_size=5),
        )
        assert cfg.sampling_mode == "first"

    def test_sampling_mode_extend(self):
        node = MockNode(requires=["sampled_text"])
        cfg = NodeConfig(
            node,
            greedy=True,
            data_store=CSVDataStore,
            sampling_config=SamplingConfig(sampling_size=5, all_variations=True),
        )
        assert cfg.sampling_mode == "extend"

    def test_sampling_mode_spread(self):
        node = MockNode(requires=["sampled_text"])
        node.spread = True
        cfg = NodeConfig(
            node,
            greedy=True,
            data_store=CSVDataStore,
            sampling_config=SamplingConfig(sampling_size=5),
        )
        assert cfg.sampling_mode == "spread"

    def test_spread_no_all_variations(self):
        node = MockNode(requires=["sampled_text"])
        node.spread = True
        with pytest.raises(AssertionError):
            NodeConfig(
                node,
                greedy=True,
                data_store=CSVDataStore,
                sampling_config=SamplingConfig(sampling_size=5, all_variations=True),
            )

    def test_step_default(self):
        node = MockNode(requires=["text"])
        cfg = NodeConfig(node)
        assert cfg.step == 1

    def test_step_custom(self):
        node = MockNode(requires=["text"])
        cfg = NodeConfig(node, step=3)
        assert cfg.step == 3

    def test_always_recompute_default(self):
        node = MockNode(requires=["text"])
        cfg = NodeConfig(node)
        assert cfg.always_recompute is False