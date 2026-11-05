import json

import pytest

pytest.importorskip("xgrammar")

from async_graph_bench.models.harmony_structured_outputs import (
    harmony_structural_tag,
)

SCHEMA = {
    "type": "object",
    "properties": {"greeting": {"type": "string"}},
    "required": ["greeting"],
    "additionalProperties": False,
}


def _formats(node):
    """All nested format dicts of a structural tag."""
    if isinstance(node, dict):
        yield node
        for value in node.values():
            yield from _formats(value)
    elif isinstance(node, list):
        for value in node:
            yield from _formats(value)


def test_json_schema_constrains_only_the_final_channel():
    tag = json.loads(
        harmony_structural_tag({"type": "json_schema", "json_schema": SCHEMA})
    )
    sequence = tag["format"]["elements"]
    assert sequence[0]["type"] == "optional"  # analysis channel may come first
    analysis = list(_formats(sequence[0]))
    assert any(f.get("begin") == "<|channel|>analysis<|message|>" for f in analysis)
    assert any(f.get("type") == "any_text" for f in analysis)  # reasoning is free text
    assert not any(f.get("type") == "json_schema" for f in analysis)
    finals = [f for f in _formats(sequence[1]) if f.get("type") == "tag"]
    assert {f["begin"] for f in finals} == {
        "<|channel|>final json<|message|>",
        "<|channel|>final <|constrain|>json<|message|>",
    }
    for f in finals:
        assert f["content"]["type"] == "json_schema"
        assert f["content"]["json_schema"] == SCHEMA
        assert f["end"] == ["<|end|>", ""]


def test_regex_and_choice_constrain_final_channel_without_json_header():
    for response_format in (
        {"type": "regex", "regex": "[0-9]+"},
        {"type": "choice", "choice": ["yes", "no (maybe)"]},
    ):
        tag = json.loads(harmony_structural_tag(response_format))
        finals = [
            f for f in _formats(tag["format"]["elements"][1]) if f.get("type") == "tag"
        ]
        assert [f["begin"] for f in finals] == ["<|channel|>final<|message|>"]
        assert finals[0]["content"]["type"] == "regex"
    choice = json.loads(
        harmony_structural_tag({"type": "choice", "choice": ["yes", "no (maybe)"]})
    )
    patterns = [f["pattern"] for f in _formats(choice) if "pattern" in f]
    assert patterns == [r"(?:yes|no\ \(maybe\))"]


def test_unsupported_type_raises():
    with pytest.raises(NotImplementedError):
        harmony_structural_tag({"type": "grammar", "grammar": "root ::= 'a'"})


def test_tag_compiles_with_xgrammar():
    xgrammar = pytest.importorskip("xgrammar")
    if not hasattr(xgrammar.Grammar, "from_structural_tag"):
        pytest.skip("xgrammar version without Grammar.from_structural_tag")
    tag = harmony_structural_tag({"type": "json_schema", "json_schema": SCHEMA})
    assert xgrammar.Grammar.from_structural_tag(tag) is not None


def test_sampling_params_use_structural_tag_only_for_gpt_oss():
    pytest.importorskip("vllm")
    from async_graph_bench import GenerationParameters
    from async_graph_bench.models.vllm_model import (
        sampling_params_from_generation_params,
    )

    params = GenerationParameters(
        max_tokens=16, response_format={"type": "json_schema", "json_schema": SCHEMA}
    )
    gpt_oss = sampling_params_from_generation_params(params, "gpt-oss")
    assert gpt_oss.structured_outputs.json is None
    tag = json.loads(gpt_oss.structured_outputs.structural_tag)
    assert tag["format"]["elements"][0]["type"] == "optional"
    other = sampling_params_from_generation_params(params, "qwen")
    assert other.structured_outputs.structural_tag is None
    assert other.structured_outputs.json == SCHEMA
    assert (
        sampling_params_from_generation_params(params).structured_outputs.json == SCHEMA
    )
