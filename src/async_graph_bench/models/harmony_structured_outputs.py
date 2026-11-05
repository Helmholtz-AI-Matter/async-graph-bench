"""Structured outputs for gpt-oss (Harmony format) with offline vLLM.

A plain json_schema / regex / choice grammar is enforced from the first generated token. vLLM's
``GptOssReasoningParser.is_reasoning_end()`` always returns True, so configuring a reasoning parser does not delay the
grammar either. For gpt-oss this forces the answer to start immediately: the model skips its analysis channel and
emits no ``<|channel|>final<|message|>`` header, so the response cannot be split into reasoning and answer.

``vllm serve`` avoids this by rewriting the request constraint into a Harmony *structural tag*
(``vllm/parser/harmony.py::_adjust_output_format``): an optional free-text analysis message, then the constraint
applied only to the content of the final channel. Offline ``LLM.chat`` does not apply that rewrite; this module builds
the same structural tag with xgrammar (installed with vLLM).
"""

import json
import re
from typing import Any

# Harmony stop tokens (<|return|>, <|call|>, <|endoftext|>) are written as "" because xgrammar disallows stop tokens
# while a constraint is active; "<|end|>" ends a non-final message.
_END_TAG = ["<|end|>", ""]
_FINAL_BEGIN = "<|channel|>final{constrain}<|message|>"
_JSON_CONSTRAINS = [" json", " <|constrain|>json"]


def _final_content_format(response_format: dict[str, Any]):
    """xgrammar format of the final-channel content for a response_format dict.

    Parameters
    ----------
    response_format : dict
        ``{"type": "json_schema", "json_schema": <schema dict>}``, ``{"type": "regex", "regex": <pattern>}`` or
        ``{"type": "choice", "choice": [<str>, ...]}`` (the format accepted by ``sampling_params_from_generation_params``).

    Returns
    -------
    tuple
        (xgrammar format object, whether the content is JSON).
    """
    from xgrammar.structural_tag import JSONSchemaFormat, RegexFormat

    kind = response_format["type"]
    if kind == "json_schema":
        return JSONSchemaFormat(json_schema=response_format["json_schema"]), True
    if kind == "regex":
        return RegexFormat(pattern=response_format["regex"]), False
    if kind == "choice":
        pattern = "|".join(re.escape(choice) for choice in response_format["choice"])
        return RegexFormat(pattern=f"(?:{pattern})"), False
    raise NotImplementedError(
        f"Harmony structured output for response_format type {kind!r} is not supported"
    )


def harmony_structural_tag(response_format: dict[str, Any]) -> str:
    """Structural tag (JSON string) that lets gpt-oss reason freely and constrains only its final answer.

    Grammar: optional ``<|channel|>analysis<|message|>...<|end|><|start|>assistant``, then
    ``<|channel|>final[ json| <|constrain|>json]<|message|>`` + constrained content + end/stop token. Mirrors
    ``_adjust_output_format`` in ``vllm/parser/harmony.py`` (what ``vllm serve`` does for gpt-oss).

    Parameters
    ----------
    response_format : dict
        See ``_final_content_format``.

    Returns
    -------
    str
        JSON string for ``vllm.sampling_params.StructuredOutputsParams(structural_tag=...)``.
    """
    from xgrammar import StructuralTag
    from xgrammar.structural_tag import (
        AnyTextFormat,
        ConstStringFormat,
        OptionalFormat,
        OrFormat,
        SequenceFormat,
        TagFormat,
    )

    content, is_json = _final_content_format(response_format)
    constrains = _JSON_CONSTRAINS if is_json else [""]
    final = OrFormat(
        elements=[
            TagFormat(
                begin=_FINAL_BEGIN.format(constrain=constrain),
                content=content,
                end=_END_TAG,
            )
            for constrain in constrains
        ]
    )
    analysis = OptionalFormat(
        content=SequenceFormat(
            elements=[
                TagFormat(
                    begin="<|channel|>analysis<|message|>",
                    content=AnyTextFormat(),
                    end="<|end|>",
                ),
                ConstStringFormat(value="<|start|>assistant"),
            ]
        )
    )
    tag = StructuralTag(format=SequenceFormat(elements=[analysis, final]))
    return json.dumps(tag.model_dump())
