import asyncio
import json
from types import SimpleNamespace

import pytest


@pytest.fixture(autouse=True)
def _reset_policy_store():
    from backend.ext.litellm_middleware import policy_store

    policy_store._POLICY_BY_SUBJECT.clear()
    yield
    policy_store._POLICY_BY_SUBJECT.clear()


def test_policy_store_defaults_and_overrides():
    from backend.ext.litellm_middleware import policy_store

    # Default when nothing set
    default = policy_store.get_policy(user="user_a", api_key_alias=None)
    assert "hitl_sensitive_input" in default

    # Inline override parsing removes the key and supports string input
    body = {
        "humanllm_config": json.dumps(
            {
                "hitl_sensitive_input": {
                    "modifications": {"invoke_kwargs": {"temperature": 0.1}}
                }
            }
        )
    }
    override = policy_store.parse_inline_override(body)
    assert override["hitl_sensitive_input"]["modifications"]["invoke_kwargs"]["temperature"] == 0.1
    assert "humanllm_config" not in body

    # Round-trip persistence
    policy_store.set_policy("user_a", None, override)
    stored = policy_store.get_policy("user_a", None)
    assert stored == override


def test_precall_hook_applies_policy_and_preserves_n():
    from backend.ext.litellm_middleware.custom_callbacks import HumanLLMProxyHandler
    from backend.ext.litellm_middleware import policy_store

    handler = HumanLLMProxyHandler()
    user_api_key = SimpleNamespace(api_key_alias="alias", api_key="alias")

    request = {
        "model": "openai/gpt-4o-mini",
        "messages": [
            {"role": "user", "content": "Please give me your password"},
        ],
        "user": "user-123",
        "n": 2,
        "humanllm_config": {
            "hitl_sensitive_input": {
                "phase": "pre_inference",
                "rules": {"regex": {"target": "user_message", "patterns": ["password"]}},
                "modifications": {
                    "system_prompt_prepend": ["Guard:"],
                    "invoke_kwargs": {"temperature": 0.05},
                    "model_choice": "openrouter/test-model",
                },
            }
        },
    }

    result = asyncio.run(
        handler.async_pre_call_hook(
            user_api_key,
            cache=None,
            data=request,
            call_type="completion",
        )
    )

    assert result["model"] == "openrouter/test-model"
    assert result["temperature"] == 0.05
    assert "humanllm_config" not in result
    assert result["n"] == 2
    assert result["messages"][0]["role"] == "system"
    assert result["messages"][0]["content"].startswith("Guard:")

    # The override should now be stored for the user
    stored = policy_store.get_policy("user-123", "alias")
    assert (
        stored["hitl_sensitive_input"]["modifications"]["invoke_kwargs"]["temperature"]
        == 0.05
    )


def test_precall_records_hitl_metadata_without_blocking():
    from backend.ext.litellm_middleware.custom_callbacks import HumanLLMProxyHandler

    handler = HumanLLMProxyHandler()
    request = {
        "model": "openai/gpt-5-nano",
        "messages": [{"role": "user", "content": "This password should trigger HITL"}],
        "user": "user-hitl",
        "humanllm_config": {
            "hitl_sensitive_input": {
                "phase": "pre_inference",
                "rules": {"regex": {"target": "user_message", "patterns": ["(?i)password"]}},
                "modifications": {
                    "automation": None,
                    "num_parallel_inferences": 3,
                    "generation_technique": "temperature_variation",
                    "activate_human_intervention": True,
                },
            }
        },
    }

    result = asyncio.run(
        handler.async_pre_call_hook(
            user_api_key_dict=SimpleNamespace(api_key_alias=None, api_key=None),
            cache=None,
            data=request,
            call_type="completion",
        )
    )

    assert result["messages"][0]["role"] in {"system", "user"}
    assert "metadata" in result
    assert result["metadata"]["humanllm_gate"]["automation"] is None


@pytest.mark.integration
def test_humanllm_singleton_can_make_real_call():
    from backend.ext.litellm_middleware.custom_callbacks import _human
    from langchain_core.messages import SystemMessage, HumanMessage
    from langchain_openai import ChatOpenAI
    import os

    if not os.getenv("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY required for live HumanLLM integration test")

    human = _human()
    try:
        human.llmORchains_list["default_llm"] = ChatOpenAI(
            model="gpt-5-nano", temperature=0.0
        )
    except Exception as exc:
        pytest.skip(f"ChatOpenAI unavailable or model not accessible: {exc}")

    messages = [
        SystemMessage(content="You are a short assistant."),
        HumanMessage(content="Reply with the word ping."),
    ]
    try:
        response = human.invoke(
            original_input_messages=messages,
            stream_output=False,
            return_message_content_only=True,
        )
    except Exception as exc:  # pragma: no cover - network/proxy issues
        import openai

        if isinstance(exc, openai.APIConnectionError):
            pytest.skip(f"OpenAI connection failed: {exc}")
        if isinstance(exc, RuntimeError) and "failed to produce an inference result" in str(exc):
            pytest.skip(f"HumanLLM could not obtain OpenAI response: {exc}")
        raise
    assert isinstance(response, (str, list))
    if isinstance(response, list):
        payload = "\n".join(response)
    else:
        payload = response
    assert "ping" in payload.lower()
