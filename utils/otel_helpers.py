"""
OTEL Helper Functions for HumanLLM Trace Integration

Provides utilities for OpenTelemetry span management, attribute setting,
and filtering for trace optimization integration.
"""
from __future__ import annotations
import hashlib
import threading
from typing import Optional, Iterable
from opentelemetry import trace
from opentelemetry.trace import Span


def get_tracer(service_name: str = "humanllm"):
    """Get an OTEL tracer for the given service name."""
    return trace.get_tracer(service_name)


def _hash_utf8(s: str) -> str:
    """Generate SHA256 hash of UTF-8 encoded string."""
    return hashlib.sha256(s.encode("utf-8", errors="ignore")).hexdigest()


def safe_set(span: Span, key: str, value: Optional[str], max_bytes: int):
    """
    Safely set a span attribute with truncation and hashing for large values.
    
    Args:
        span: The OTEL span to set attribute on
        key: The attribute key
        value: The attribute value (converted to string)
        max_bytes: Maximum size in bytes; if exceeded, truncate and add hash
    """
    if value is None:
        return
    
    s = str(value)
    b = s.encode("utf-8", errors="ignore")
    
    if max_bytes and len(b) > max_bytes:
        # For large values: set hash + truncated flag + truncated content
        span.set_attribute(f"{key}.sha256", _hash_utf8(s))
        span.set_attribute(f"{key}.truncated", True)
        s = b[:max_bytes].decode("utf-8", errors="ignore")
    
    span.set_attribute(key, s)


def set_inputs(span: Span, max_bytes: int, **inputs):
    """
    Set multiple inputs.* attributes on a span.
    
    Args:
        span: The OTEL span
        max_bytes: Maximum size per input value
        **inputs: Key-value pairs to set as inputs.*
    """
    for k, v in inputs.items():
        if v is None:
            continue
        safe_set(span, f"inputs.{k}", str(v), max_bytes)


def attr_allowed(key: str, include: Optional[Iterable[str]], exclude: Optional[Iterable[str]]) -> bool:
    """
    Check if an attribute key passes include/exclude filters.
    
    Args:
        key: The attribute key to check
        include: List of prefixes to include (None means include all)
        exclude: List of prefixes to exclude (takes priority over include)
        
    Returns:
        True if the attribute should be kept, False otherwise
    """
    if include:
        if not any(key.startswith(p) for p in include):
            return False
    
    if exclude:
        if any(key.startswith(p) for p in exclude):
            return False
    
    return True


def current_thread_id() -> str:
    """Get current thread ID as string."""
    return str(threading.get_ident())
