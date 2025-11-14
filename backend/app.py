"""Minimal FastAPI application with optional HumanLLM middleware router."""

from __future__ import annotations

import os

from fastapi import FastAPI

app = FastAPI(title="CollabToolBuilder Backend")

if os.getenv("ENABLE_LITELLM_MIDDLEWARE") == "1":
    from backend.ext.litellm_middleware import admin_router

    app.include_router(admin_router.router)
