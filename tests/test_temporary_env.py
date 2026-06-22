import os
import pytest
from async_graph_bench.utils.temporary_env import temporary_env


class TestTemporaryEnv:
    def test_new_key_set_and_restored(self):
        assert "_OPCODE_TEST_KEY" not in os.environ
        with temporary_env({"_OPCODE_TEST_KEY": "hello"}):
            assert os.environ["_OPCODE_TEST_KEY"] == "hello"
        assert "_OPCODE_TEST_KEY" not in os.environ

    def test_existing_key_overridden_and_restored(self):
        os.environ["_OPCODE_EXISTING"] = "original"
        try:
            with temporary_env({"_OPCODE_EXISTING": "modified"}):
                assert os.environ["_OPCODE_EXISTING"] == "modified"
            assert os.environ["_OPCODE_EXISTING"] == "original"
        finally:
            del os.environ["_OPCODE_EXISTING"]

    def test_non_string_value_converted(self):
        with temporary_env({"_OPCODE_NUM": 42, "_OPCODE_BOOL": True}):
            assert os.environ["_OPCODE_NUM"] == "42"
            assert os.environ["_OPCODE_BOOL"] == "True"

    def test_exception_restores_env(self):
        assert "_OPCODE_ERR" not in os.environ
        with pytest.raises(ValueError, match="test error"):
            with temporary_env({"_OPCODE_ERR": "val"}):
                assert os.environ["_OPCODE_ERR"] == "val"
                raise ValueError("test error")
        assert "_OPCODE_ERR" not in os.environ
