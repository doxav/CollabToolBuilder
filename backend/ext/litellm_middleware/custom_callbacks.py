"""LiteLLM custom callbacks that apply HumanLLM pre-call policies."""

from __future__ import annotations

import os
from typing import Any, Dict, Literal, Optional

try:
    from litellm.integrations.custom_logger import CustomLogger
except ImportError:  # pragma: no cover - fallback when LiteLLM not installed
    class CustomLogger:  # type: ignore[override]
        """Fallback logger base when LiteLLM is unavailable."""

        async def async_pre_call_hook(self, *args: Any, **kwargs: Any) -> Optional[Dict[str, Any]]:  # pragma: no cover
            return None

try:  # pragma: no cover - fallback for minimal installations without proxy extras
    from litellm.proxy.proxy_server import DualCache, UserAPIKeyAuth
except ImportError:  # pragma: no cover
    class UserAPIKeyAuth:  # type: ignore[override]
        """Fallback auth stub used when LiteLLM proxy extras are unavailable."""

        def __init__(self, api_key: Optional[str] = None, api_key_alias: Optional[str] = None) -> None:
            self.api_key = api_key
            self.api_key_alias = api_key_alias

    class DualCache:  # type: ignore[override]
        """Fallback cache stub used when LiteLLM proxy extras are unavailable."""

        pass

from .policy_store import get_policy, parse_inline_override, set_policy

from utils import llm_utils
from utils import human_llm_config as human_llm_cfg
from utils.human_llm import HumanLLM, HumanLLMConfig

try:  # Some environments ship a non-Python config shim; ignore failures importing it.
    import config as collab_cfg  # noqa: F401  # ensure repo config is imported for side effects
except Exception:  # pragma: no cover - best effort only

_HUMANLLM_SINGLETON: Optional[HumanLLM] = None
_VDB_PATCHED = False
_SMART_INPUT_PATCHED = False


class _NullVectorDB:
    """Minimal vector DB stub to avoid heavy dependencies during tests."""

    def __init__(self, config: Optional[llm_utils.UnifiedVectorDBConfig] = None, check_db: bool = False) -> None:
        self.config = config or llm_utils.UnifiedVectorDBConfig()
        self.config.db_type = "memory"

    def log_agent_data(self, *args: Any, **kwargs: Any) -> None:  # pragma: no cover - passthrough
        return None

    def count(self) -> int:  # pragma: no cover
        return 0

    def _query(self, *args: Any, **kwargs: Any) -> list:  # pragma: no cover
        return []

    def _add_texts(self, *args: Any, **kwargs: Any) -> None:  # pragma: no cover
        return None

    def delete(self, *args: Any, **kwargs: Any) -> None:  # pragma: no cover
        return None

    def populate_few_shot_tags(self, *args: Any, **kwargs: Any) -> dict:  # pragma: no cover
        return {}

    def retrieve_logs(self, *args: Any, **kwargs: Any) -> list:  # pragma: no cover
        return []

    def get_unique_id(self) -> str:  # pragma: no cover
        return "null"


def _ensure_null_vectordb() -> None:
    global _VDB_PATCHED, _SMART_INPUT_PATCHED
    if not _VDB_PATCHED:
        llm_utils.UnifiedVectorDB = _NullVectorDB  # type: ignore[assignment]
        human_llm_cfg.UnifiedVectorDB = _NullVectorDB  # type: ignore[assignment]
        _VDB_PATCHED = True
    if not _SMART_INPUT_PATCHED and hasattr(llm_utils, "smart_input"):
        llm_utils.smart_input = lambda *args, **kwargs: ""  # type: ignore[assignment]
        _SMART_INPUT_PATCHED = True


def _build_humanllm() -> HumanLLM:
    """Build a HumanLLM instance configured for proxy policy evaluation."""

    _ensure_null_vectordb()
    cfg = HumanLLMConfig()
    if os.getenv("HUMANLLM_USE_LITELLM") == "1":
        os.environ.setdefault(
            "OPENAI_BASE_URL",
            os.getenv("HUMANLLM_LITELLM_BASE_URL", "http://localhost:4000/v1"),
        )
    cfg.initialize()
    llm_list = cfg.get_llmORchains_list()
    human = HumanLLM(
        agent_name="HumanLLMProxyPolicy",
        llmORchains_list=llm_list,
        num_parallel_inferences=1,
        premium_llm_by_default=os.getenv("HUMANLLM_PREMIUM_DEFAULT", "0") == "1",
        dynamic_llm_config={},
    )
    if getattr(human, "dynamic_mgr", None):
        human.dynamic_mgr.config = human.dynamic_llm_config
    human.automation = "auto"
    # Ensure tests fail fast instead of looping endlessly when providers are unreachable
    human.max_consecutive_inference_failures = 1
    human._consecutive_inference_failures = 0
    return human


def _human() -> HumanLLM:
    """Return a singleton HumanLLM used for evaluating dynamic policies."""

    global _HUMANLLM_SINGLETON
    if _HUMANLLM_SINGLETON is None:
        _HUMANLLM_SINGLETON = _build_humanllm()
    return _HUMANLLM_SINGLETON


def _extract_messages(data: Dict[str, Any]) -> Dict[str, str]:
    """Pull the latest user message and current system prompt from OpenAI-style payload."""

    messages = data.get("messages", []) or []
    user_msg = ""
    system_msg = ""
    for message in reversed(messages):
        if message.get("role") == "user":
            user_msg = message.get("content", "")
            break
    for message in messages:
        if message.get("role") == "system":
            system_msg = message.get("content", "")
            break
    return {"user_message": user_msg, "system_prompt": system_msg}


class HumanLLMProxyHandler(CustomLogger):
    """LiteLLM callback that enforces HumanLLM-driven prompt policies."""

    async def async_pre_call_hook(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        cache: DualCache,
        data: Dict[str, Any],
        call_type: Literal[
            "completion",
            "text_completion",
            "embeddings",
            "image_generation",
            "moderation",
            "audio_transcription",
        ],
    ) -> Optional[Dict[str, Any]]:
        user_str = data.get("user")
        key_alias = None
        if user_api_key_dict is not None:
            key_alias = getattr(user_api_key_dict, "api_key_alias", None) or getattr(
                user_api_key_dict, "api_key", None
            )

        override_cfg = parse_inline_override(data)
        dyn_cfg = override_cfg or get_policy(user_str, key_alias)
        if override_cfg is not None:
            set_policy(user_str, key_alias, override_cfg)

        context = _extract_messages(data)
        context.update(
            {
                "invoke_kwargs": {},
                "kwargs": {k: data.get(k) for k in ("temperature", "top_p", "n") if k in data},
            }
        )

        human = _human()
        dyn_cfg = dyn_cfg or {}
        human.dynamic_llm_config = dyn_cfg
        if getattr(human, "dynamic_mgr", None):
            human.dynamic_mgr.config = dyn_cfg

        modifications = {}
        if getattr(human, "dynamic_mgr", None):
            modifications = human.dynamic_mgr.evaluate_triggers(context, phase="pre_inference") or {}

        messages = data.get("messages") or []

        def ensure_system_message() -> Dict[str, Any]:
            if not any(msg.get("role") == "system" for msg in messages):
                messages.insert(0, {"role": "system", "content": ""})
            for msg in messages:
                if msg.get("role") == "system":
                    return msg
            raise RuntimeError("Failed to ensure system message present")

        if modifications:
            if "system_prompt" in modifications:
                ensure_system_message()["content"] = modifications["system_prompt"]
            for key in ("system_prompt_prepend", "system_prompt_append"):
                values = modifications.get(key) or []
                if values:
                    system_entry = ensure_system_message()
                    if key.endswith("prepend"):
                        system_entry["content"] = "".join(values) + system_entry["content"]
                    else:
                        system_entry["content"] = system_entry["content"] + "".join(values)
            invoke_kwargs = modifications.get("invoke_kwargs") or {}
            for param in ("temperature", "top_p", "presence_penalty", "frequency_penalty", "max_tokens"):
                if param in invoke_kwargs:
                    data[param] = invoke_kwargs[param]
            if isinstance(modifications.get("model_choice"), str):
                data["model"] = modifications["model_choice"]

            hitl_keys = (
                "automation",
                "num_parallel_inferences",
                "generation_technique",
                "activate_human_intervention",
            )
            hitl_payload = {key: modifications[key] for key in hitl_keys if key in modifications}
            if hitl_payload:
                metadata = data.get("metadata") or data.get("litellm_metadata")
                if not isinstance(metadata, dict):
                    metadata = {}
                    data.setdefault("metadata", metadata)
                metadata.setdefault("humanllm_gate", {}).update(hitl_payload)

        data["messages"] = messages
        return data


proxy_handler_instance = HumanLLMProxyHandler()

__all__ = ["HumanLLMProxyHandler", "proxy_handler_instance", "_human"]
