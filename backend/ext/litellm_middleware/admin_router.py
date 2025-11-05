"""FastAPI router to mutate HumanLLM policies at runtime."""

from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel

from .policy_store import get_policy, set_policy

router = APIRouter(prefix="/humanllm", tags=["humanllm"])


class PolicyUpdate(BaseModel):
    """Request body for updating a policy."""

    user: str | None = None
    key_alias: str | None = None
    dynamic_llm_config: dict


@router.post("/policy")
def update_policy(body: PolicyUpdate) -> dict:
    """Persist a policy override for a given user or API key alias."""

    set_policy(body.user, body.key_alias, body.dynamic_llm_config)
    return {"ok": True}


@router.get("/policy")
def read_policy(user: str | None = None, key_alias: str | None = None) -> dict:
    """Fetch the stored policy for the requested subject."""

    return {"config": get_policy(user, key_alias)}
