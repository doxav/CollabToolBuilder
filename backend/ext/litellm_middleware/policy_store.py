"""In-memory store for per-user HumanLLM dynamic configuration policies."""

from __future__ import annotations

import json
import os
import threading
import time
from typing import Any, Dict, Optional

DEFAULT_DYNAMIC_CONFIG: Dict[str, Any] = {
    "hitl_sensitive_input": {
        "phase": "pre_inference",
        "rules": {
            "regex": {
                "target": "user_message",
                "patterns": [
                    r"(?i)(password|OPENAI_API[-_ ]?key|secret)",
                    r"(?i)(prod|billing|database).*(dump|delete|exfiltrate)",
                ],
            }
        },
        "modifications": {
            "automation": None,
            "num_parallel_inferences": 3,
            "generation_technique": "temperature_variation",
            "activate_human_intervention": True,
        },
    },
    "hitl_every_10": {
        "phase": "pre_inference",
        "rules": {"frequency": {"every_n": 10}},
        "modifications": {
            "automation": None,
            "num_parallel_inferences": 3,
            "generation_technique": "temperature_variation",
            "activate_human_intervention": True,
        },
    },
}

_POLICY_LOCK = threading.Lock()
_POLICY_BY_SUBJECT: Dict[str, Dict[str, Any]] = {}
_TTL_SECS = int(os.getenv("HUMANLLM_POLICY_TTL_SECS", "7200"))


def _subject_key(user: Optional[str], api_key_alias: Optional[str]) -> str:
    if user:
        return f"user:{user}"
    return f"key:{api_key_alias or 'anon'}"


def _purge_expired(now: float) -> None:
    for key, value in list(_POLICY_BY_SUBJECT.items()):
        if now - value.get("_ts", 0) > _TTL_SECS:
            _POLICY_BY_SUBJECT.pop(key, None)


def get_policy(user: Optional[str], api_key_alias: Optional[str]) -> Dict[str, Any]:
    """Return the policy for a subject, falling back to defaults."""

    now = time.time()
    with _POLICY_LOCK:
        _purge_expired(now)
        key = _subject_key(user, api_key_alias)
        entry = _POLICY_BY_SUBJECT.get(key)
        if entry:
            return entry.get("config", DEFAULT_DYNAMIC_CONFIG)
        return DEFAULT_DYNAMIC_CONFIG


def set_policy(user: Optional[str], api_key_alias: Optional[str], config: Dict[str, Any]) -> None:
    """Persist a policy for the subject with a TTL timestamp."""

    with _POLICY_LOCK:
        key = _subject_key(user, api_key_alias)
        _POLICY_BY_SUBJECT[key] = {"config": config, "_ts": time.time()}


def parse_inline_override(body: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Extract and remove an inline override policy from the request body."""

    cfg = body.pop("humanllm_config", None)
    if cfg is None:
        metadata = body.get("metadata") or body.get("litellm_metadata") or {}
        cfg = metadata.pop("humanllm_config", None)
    if isinstance(cfg, str):
        try:
            cfg = json.loads(cfg)
        except Exception:
            cfg = None
    return cfg if isinstance(cfg, dict) else None


__all__ = [
    "DEFAULT_DYNAMIC_CONFIG",
    "get_policy",
    "parse_inline_override",
    "set_policy",
]
