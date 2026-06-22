import pytest
from async_graph_bench import SamplingConfig


class TestSamplingConfig:
    def test_create(self):
        cfg = SamplingConfig(sampling_size=10)
        assert cfg.sampling_size == 10
        assert cfg.all_variations is None

    def test_with_all_variations(self):
        cfg = SamplingConfig(sampling_size=5, all_variations=True)
        assert cfg.sampling_size == 5
        assert cfg.all_variations is True

    def test_frozen_cannot_modify(self):
        cfg = SamplingConfig(sampling_size=10)
        with pytest.raises(Exception):
            cfg.sampling_size = 20

    def test_frozen_cannot_add_field(self):
        cfg = SamplingConfig(sampling_size=10)
        with pytest.raises(Exception):
            cfg.new_field = "value"

    def test_equality(self):
        c1 = SamplingConfig(sampling_size=5)
        c2 = SamplingConfig(sampling_size=5)
        assert c1 == c2

    def test_inequality(self):
        c1 = SamplingConfig(sampling_size=5)
        c2 = SamplingConfig(sampling_size=10)
        assert c1 != c2

    def test_hashable(self):
        cfg = SamplingConfig(sampling_size=5)
        assert hash(cfg) != 0
        assert {cfg} == {cfg}
