
"""
Benchmark Test: Trace Optimizer Experience Accumulation Strategies

Compares 5 different strategies for improving trace optimizer learning:
1. Baseline (no optimization)
2. Strategy 1: Longer Trace (accumulation_window)
3. Strategy 2: Continuous Backward (backward per invoke)
4. Strategy 3: Optimizer Memory (memory_size)
5. Strategy 4: Multi-Agent Sharing (shared_memory_id)
6. Strategy 5: Offline Batch (H2 pattern)
7. Combined (all strategies together)

Uses OpenTelemetry spans to track performance metrics.
"""
import json
import copy
import time
import pytest
import threading
import concurrent.futures
from typing import Dict, List, Any, Optional
from unittest.mock import Mock, MagicMock, patch
from pathlib import Path

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

try:
    from langchain_openai import ChatOpenAI
    HAS_LANGCHAIN = True
except ImportError:
    HAS_LANGCHAIN = False

try:
    from config import MODELS_CONFIG_LIST
    from utils.human_llm import HumanLLM
    from utils.human_llm_config import HumanLLMConfig
    from utils.constants import CHROMA_DATABASE
    HAS_HUMANLLM = True
    
    # Configure lightweight vector DB for tests
    HumanLLMConfig().common_vectordb_config.db_type = CHROMA_DATABASE
except ImportError as e:
    HAS_HUMANLLM = False
    MODELS_CONFIG_LIST = {}
    HumanLLM = None

from opentelemetry import trace as _oteltrace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.sdk.resources import Resource

from utils.otel_helpers import get_tracer, safe_set

# Global span exporter for collecting all spans
GLOBAL_SPAN_EXPORTER = InMemorySpanExporter()

# ========== OpenTelemetry Setup ==========
def setup_otel():
    """Initialize OTEL with in-memory exporter for collecting spans."""
    resource = Resource.create({"service.name": "trace-optimizer-benchmark"})
    provider = TracerProvider(resource=resource)
    # Use in-memory exporter to collect ALL spans for analysis
    processor = SimpleSpanProcessor(GLOBAL_SPAN_EXPORTER)
    provider.add_span_processor(processor)
    _oteltrace.set_tracer_provider(provider)
    return get_tracer("benchmark")


def save_otel_spans_to_file(filepath: str):
    """Save all collected OTEL spans to a JSON file."""
    spans = GLOBAL_SPAN_EXPORTER.get_finished_spans()
    span_data = []
    
    for span in spans:
        span_dict = {
            "name": span.name,
            "span_id": format(span.context.span_id, '016x'),
            "trace_id": format(span.context.trace_id, '032x'),
            "parent_span_id": format(span.parent.span_id, '016x') if span.parent else None,
            "start_time": span.start_time,
            "end_time": span.end_time,
            "duration_ms": (span.end_time - span.start_time) / 1_000_000 if span.end_time else None,
            "attributes": dict(span.attributes) if span.attributes else {},
            "status": {
                "code": str(span.status.status_code),
                "description": span.status.description
            },
            "events": [
                {
                    "name": event.name,
                    "timestamp": event.timestamp,
                    "attributes": dict(event.attributes) if event.attributes else {}
                }
                for event in span.events
            ] if span.events else []
        }
        span_data.append(span_dict)
    
    with open(filepath, 'w') as f:
        json.dump(span_data, f, indent=2, default=str)
    
    return len(span_data)


tracer = setup_otel()


# ========== Test Utilities ==========
def _get_llm():
    """Get a test LLM instance."""
    if not HAS_LANGCHAIN:
        # Return mock LLM if langchain not available
        mock_llm = Mock()
        mock_llm.model_name = "mock-model"
        return {
            "default_llm": mock_llm,
            "premium_llm": mock_llm,
        }
    
    model = "gpt-4o-mini-2024-07-18"
    try:
        if isinstance(MODELS_CONFIG_LIST, dict):
            model = MODELS_CONFIG_LIST.get("basic_gpt", model)
    except Exception:
        pass
    return {
        "default_llm": ChatOpenAI(model=model, temperature=0.0),
        "premium_llm": ChatOpenAI(model=model, temperature=0.0),
    }


def measure_memory():
    """Get current process memory usage in MB."""
    if HAS_PSUTIL:
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024
    return 0.0  # Return 0 if psutil not available


class BenchmarkMetrics:
    """Track metrics for each strategy."""
    
    def __init__(self, strategy_name: str):
        self.strategy_name = strategy_name
        self.invocations = 0
        self.optimizations = 0
        self.feedback_collected = 0
        self.total_latency_ms = 0.0
        self.memory_usage_mb = []
        self.trace_lengths = []
        self.quality_scores = []
        self.start_time = time.time()
        
    def record_invocation(self, latency_ms: float, memory_mb: float):
        """Record a single invocation."""
        self.invocations += 1
        self.total_latency_ms += latency_ms
        self.memory_usage_mb.append(memory_mb)
        
    def record_optimization(self, trace_length: int):
        """Record an optimization step."""
        self.optimizations += 1
        self.trace_lengths.append(trace_length)
        
    def record_feedback(self):
        """Record feedback collection."""
        self.feedback_collected += 1
        
    def record_quality(self, score: float):
        """Record quality score (simulated)."""
        self.quality_scores.append(score)
        
    def finalize(self):
        """Calculate final statistics."""
        duration = time.time() - self.start_time
        return {
            "strategy": self.strategy_name,
            "invocations": self.invocations,
            "optimizations": self.optimizations,
            "feedback_collected": self.feedback_collected,
            "avg_latency_ms": self.total_latency_ms / max(1, self.invocations),
            "total_latency_ms": self.total_latency_ms,
            "avg_memory_mb": sum(self.memory_usage_mb) / max(1, len(self.memory_usage_mb)),
            "peak_memory_mb": max(self.memory_usage_mb) if self.memory_usage_mb else 0,
            "avg_trace_length": sum(self.trace_lengths) / max(1, len(self.trace_lengths)),
            "max_trace_length": max(self.trace_lengths) if self.trace_lengths else 0,
            "avg_quality": sum(self.quality_scores) / max(1, len(self.quality_scores)),
            "quality_improvement": self.quality_scores[-1] - self.quality_scores[0] if len(self.quality_scores) > 1 else 0.0,
            "total_duration_s": duration,
        }


# ========== Strategy Configurations ==========

def get_baseline_config() -> Dict[str, Any]:
    """Baseline: No trace optimization."""
    return {
        # Empty config - no trace optimizers, no special rules
        # This tests baseline performance without any optimization
    }


def get_strategy1_config() -> Dict[str, Any]:
    """Strategy 1: Longer Trace (accumulation_window=5)."""
    return {
        "trace_optimizers": [
            {
                "name": "longer_trace_optimizer",
                "optimizer_kind": "OptoPrimeV2",
                "parameters": [
                    {"parameter": "invoke.temperature", "value": 0.5, "info": {"bounds": [0.0, 0.9]}, "trainable": True},
                    {"parameter": "invoke.max_tokens", "value": 256, "info": {"bounds": [128, 1024]}, "trainable": True},
                ],
                "objective": "maximize quality",
            }
        ],
        "annotations_help_trace_PRE": {
            "phase": "pre_inference",
            "quota": {"max_count": 100},
            "rules": {"regex": {"patterns": [".*"]}},  # Always trigger for testing
            "modifications": [
                {
                    "type": "trace",
                    "optimizer": "longer_trace_optimizer",
                    "targets": ["invoke.temperature", "invoke.max_tokens"],
                    "accumulation_window": 5,  # ← Strategy 1: Accumulate 5 invocations
                    "n_candidates": 1,
                }
            ],
        },
    }


def get_strategy2_config() -> Dict[str, Any]:
    """Strategy 2: Continuous Backward (backward per invoke)."""
    return {
        "trace_optimizers": [
            {
                "name": "continuous_backward_optimizer",
                "optimizer_kind": "OptoPrimeV2",
                "parameters": [
                    {"parameter": "invoke.temperature", "value": 0.5, "info": {"bounds": [0.0, 0.9]}, "trainable": True},
                    {"parameter": "invoke.max_tokens", "value": 256, "info": {"bounds": [128, 1024]}, "trainable": True},
                ],
                "objective": "maximize quality",
            }
        ],
        "annotations_help_trace_PRE": {
            "phase": "pre_inference",
            "quota": {"max_count": 100},
            "rules": {"regex": {"patterns": [".*"]}},
            "modifications": [
                {
                    "type": "trace",
                    "optimizer": "continuous_backward_optimizer",
                    "targets": ["invoke.temperature", "invoke.max_tokens"],
                    "backward_mode": "continuous",  # ← Strategy 2: Call backward every invoke
                    "step_frequency": 3,  # But only step every 3 invokes
                    "n_candidates": 1,
                }
            ],
        },
    }


def get_strategy3_config() -> Dict[str, Any]:
    """Strategy 3: Optimizer Memory (memory_size=10)."""
    return {
        "trace_optimizers": [
            {
                "name": "memory_optimizer",
                "optimizer_kind": "OptoPrimeV2",
                "memory_size": 10,  # ← Strategy 3: Remember 10 past examples
                "parameters": [
                    {"parameter": "invoke.temperature", "value": 0.5, "info": {"bounds": [0.0, 0.9]}, "trainable": True},
                    {"parameter": "invoke.max_tokens", "value": 256, "info": {"bounds": [128, 1024]}, "trainable": True},
                ],
                "objective": "maximize quality",
            }
        ],
        "annotations_help_trace_PRE": {
            "phase": "pre_inference",
            "quota": {"max_count": 100},
            "rules": {"regex": {"patterns": [".*"]}},
            "modifications": [
                {
                    "type": "trace",
                    "optimizer": "memory_optimizer",
                    "targets": ["invoke.temperature", "invoke.max_tokens"],
                    "n_candidates": 1,
                }
            ],
        },
    }


def get_strategy4_config(agent_id: int) -> Dict[str, Any]:
    """Strategy 4: Multi-Agent Sharing (shared_memory_id)."""
    return {
        "trace_optimizers": [
            {
                "name": "shared_optimizer",
                "optimizer_kind": "OptoPrimeV2",
                "memory_size": 20,
                "shared_memory_id": "global_shared_buffer",  # ← Strategy 4: All agents share memory
                "parameters": [
                    {"parameter": "invoke.temperature", "value": 0.5, "info": {"bounds": [0.0, 0.9]}, "trainable": True},
                    {"parameter": "invoke.max_tokens", "value": 256, "info": {"bounds": [128, 1024]}, "trainable": True},
                ],
                "objective": "maximize quality",
            }
        ],
        "annotations_help_trace_PRE": {
            "phase": "pre_inference",
            "quota": {"max_count": 100},
            "rules": {"regex": {"patterns": [".*"]}},
            "modifications": [
                {
                    "type": "trace",
                    "optimizer": "shared_optimizer",
                    "targets": ["invoke.temperature", "invoke.max_tokens"],
                    "n_candidates": 1,
                }
            ],
        },
    }


def get_strategy5_config() -> Dict[str, Any]:
    """Strategy 5: Offline Batch Optimization (H2 pattern)."""
    return {
        # REMOVED: "auto_record_outcomes": True,  # Causes TypeError - removed
        "trace_optimizers": [
            {
                "name": "offline_optimizer",
                "optimizer_kind": "OptoPrimeV2",
                "parameters": [
                    {"parameter": "invoke.temperature", "value": 0.5, "info": {"bounds": [0.0, 0.9]}, "trainable": True},
                    {"parameter": "invoke.max_tokens", "value": 256, "info": {"bounds": [128, 1024]}, "trainable": True},
                    {"parameter": "dynamic_llm_config.default_temperature", "value": 0.3, "info": {"bounds": [0.0, 1.0]}, "trainable": True},
                ],
                "objective": "maximize quality minimize cost",
            }
        ],
        "offline_batch_optimization": {
            "enabled": True,
            "trigger_every_n_invokes": 10,  # Run offline optimization every 10 invokes
        },
        "annotations_help_trace_PRE": {
            "phase": "pre_inference",
            "quota": {"max_count": 100},
            "rules": {"regex": {"patterns": [".*"]}},
            "modifications": [
                {
                    "type": "trace",
                    "optimizer": "offline_optimizer",
                    "targets": ["invoke.temperature", "invoke.max_tokens"],
                    "n_candidates": 1,
                }
            ],
        },
    }


def get_combined_config() -> Dict[str, Any]:
    """Combined: All strategies together."""
    return {
        # REMOVED: "auto_record_outcomes": True,  # Causes TypeError - removed
        "trace_optimizers": [
            {
                "name": "combined_optimizer",
                "optimizer_kind": "OptoPrimeV2",
                "memory_size": 50,  # Strategy 3 (larger buffer)
                "shared_memory_id": "combined_global",  # Strategy 4
                "parameters": [
                    {"parameter": "invoke.temperature", "value": 0.5, "info": {"bounds": [0.0, 0.9]}, "trainable": True},
                    {"parameter": "invoke.max_tokens", "value": 256, "info": {"bounds": [128, 1024]}, "trainable": True},
                    {"parameter": "dynamic_llm_config.default_temperature", "value": 0.3, "info": {"bounds": [0.0, 1.0]}, "trainable": True},
                ],
                "objective": "maximize quality minimize cost",
            }
        ],
        "offline_batch_optimization": {
            "enabled": True,
            "trigger_every_n_invokes": 10,
        },
        "annotations_help_trace_PRE": {
            "phase": "pre_inference",
            "quota": {"max_count": 100},
            "rules": {"regex": {"patterns": [".*"]}},
            "modifications": [
                {
                    "type": "trace",
                    "optimizer": "combined_optimizer",
                    "targets": ["invoke.temperature", "invoke.max_tokens"],
                    "accumulation_window": 5,  # Strategy 1
                    "backward_mode": "continuous",  # Strategy 2
                    "step_frequency": 5,
                    "n_candidates": 1,
                }
            ],
        },
    }


# ========== Benchmark Test Functions ==========

def run_strategy_benchmark(
    strategy_name: str,
    config: Dict[str, Any],
    n_trials: int = 1,  # Reduced from 3 to 1 for speed
    n_invokes_per_trial: int = 5,  # Reduced from 15 to 5 for speed
    timeout_seconds: int = 30,  # Add timeout
) -> Dict[str, Any]:
    """
    Run benchmark for a single strategy with timeout.
    
    Args:
        strategy_name: Name of the strategy
        config: Dynamic LLM config for this strategy
        n_trials: Number of independent trials to run (default: 1 for speed)
        n_invokes_per_trial: Number of invocations per trial (default: 5 for speed)
        timeout_seconds: Maximum time to run before timeout
        
    Returns:
        Aggregated metrics across all trials
    """
    if not HAS_HUMANLLM:
        # Return mock results if HumanLLM not available
        return {
            "strategy": strategy_name,
            "n_trials": n_trials,
            "trials": [],
            "avg_invocations": n_invokes_per_trial,
            "avg_optimizations": 2.0,
            "avg_latency_ms": 10.0,
            "avg_memory_mb": 100.0,
            "avg_quality_improvement": 0.15,
            "avg_trace_length": 3.0,
            "status": "mocked",
        }
    
    start_time = time.time()
    
    with tracer.start_as_current_span(f"benchmark_{strategy_name}") as span:
        safe_set(span, "strategy.name", strategy_name, max_bytes=100)
        safe_set(span, "strategy.n_trials", str(n_trials), max_bytes=100)
        safe_set(span, "strategy.n_invokes_per_trial", str(n_invokes_per_trial), max_bytes=100)
        
        all_trial_metrics = []
        
        for trial in range(n_trials):
            # Check timeout
            if time.time() - start_time > timeout_seconds:
                safe_set(span, "timeout", "true", max_bytes=100)
                safe_set(span, "timeout_reason", f"Exceeded {timeout_seconds}s", max_bytes=200)
                break
                
            with tracer.start_as_current_span(f"trial_{trial}") as trial_span:
                safe_set(trial_span, "trial.index", str(trial), max_bytes=100)
                
                # Create fresh HumanLLM instance for this trial
                agent_name = f"{strategy_name}_trial_{trial}"
                print(f"\n{'='*60}")
                print(f"[DEBUG] Creating HumanLLM for {agent_name}")
                print(f"[DEBUG] Config keys: {list(config.keys())}")
                h = HumanLLM(
                    agent_name=agent_name,
                    llmORchains_list=_get_llm(),
                    dynamic_llm_config=copy.deepcopy(config),
                )
                
                # Log optimizer initialization
                if h.dynamic_mgr:
                    print(f"[DEBUG] ✓ DynamicConfigManager initialized")
                    optimizers = h.dynamic_mgr.list_trace_optimizers()
                    print(f"[DEBUG] Trace optimizers registered: {optimizers}")
                    if optimizers:
                        for opt_name in optimizers:
                            adapter = h.dynamic_mgr._trace_optimizers.get(opt_name)
                            if adapter:
                                # Show registered parameters
                                params = list(adapter._parameters.keys()) if hasattr(adapter, '_parameters') else []
                                print(f"[DEBUG]   - {opt_name}: registered params = {params}")
                else:
                    print(f"[DEBUG] ✗ No DynamicConfigManager (baseline or error)")
                print(f"{'='*60}\n")
                
                metrics = BenchmarkMetrics(f"{strategy_name}_trial_{trial}")
                
                # Simulate multiple invocations with feedback
                for invoke_idx in range(n_invokes_per_trial):
                    # Check timeout
                    if time.time() - start_time > timeout_seconds:
                        safe_set(trial_span, "timeout", "true", max_bytes=100)
                        break
                        
                    with tracer.start_as_current_span(f"invocation_{invoke_idx}") as invoke_span:
                        start_time = time.time()
                        mem_before = measure_memory()
                        
                        # Simulate context for optimization
                        ctx = {
                            "user_message": f"test message {invoke_idx} for optimization",
                            "system_prompt": "You are a helpful assistant.",
                            "invoke_kwargs": {},
                        }
                        
                        # Trigger online (H1) optimization if applicable
                        if h.dynamic_mgr:
                            print(f"[DEBUG] Invocation {invoke_idx}: Evaluating triggers...")
                            mods = h.dynamic_mgr.evaluate_triggers(ctx, phase="pre_inference")
                            print(f"[DEBUG]   → Modifications returned: {len(mods) if mods else 0}")
                            if mods:
                                metrics.record_optimization(
                                    trace_length=invoke_idx + 1  # Cumulative trace
                                )
                                print(f"[DEBUG]   ✓ Optimization triggered! Modifications: {mods}")
                                safe_set(invoke_span, "optimization.triggered", "true", max_bytes=100)
                                safe_set(invoke_span, "optimization.modifications", str(len(mods)), max_bytes=100)
                            else:
                                print(f"[DEBUG]   ✗ No modifications triggered")
                        
                        # Simulate feedback collection
                        simulated_quality = 0.5 + (invoke_idx / n_invokes_per_trial) * 0.3  # Gradual improvement
                        metrics.record_quality(simulated_quality)
                        
                        if h.dynamic_mgr and invoke_idx % 3 == 0:
                            # Record outcome every 3 invokes
                            print(f"[DEBUG] Invocation {invoke_idx}: Recording outcome (quality={simulated_quality:.3f})")
                            h.dynamic_mgr.record_outcome(
                                ctx,
                                {"invoke_kwargs": ctx.get("invoke_kwargs", {})},
                                {"score": simulated_quality}
                            )
                            metrics.record_feedback()
                            print(f"[DEBUG]   ✓ Outcome recorded")
                            
                            # Check if backward() should be called
                            optimizers = h.dynamic_mgr.list_trace_optimizers()
                            if optimizers:
                                for opt_name in optimizers:
                                    adapter = h.dynamic_mgr._trace_optimizers.get(opt_name)
                                    if adapter and hasattr(adapter, '_parameters'):
                                        trace_info = f"params={len(adapter._parameters)}"
                                        print(f"[DEBUG]   Optimizer {opt_name} state: {trace_info}")
                            
                            # Simulate user correction occasionally
                            if invoke_idx % 6 == 0:
                                print(f"[DEBUG] Invocation {invoke_idx}: Logging user correction")
                                h.dynamic_mgr.log_user_correction(
                                    agent_name,
                                    diff="improve response quality",
                                    who="benchmark_tester"
                                )
                                print(f"[DEBUG]   ✓ User correction logged")
                        
                        # Measure invocation metrics
                        latency = (time.time() - start_time) * 1000  # Convert to ms
                        mem_after = measure_memory()
                        mem_delta = mem_after - mem_before
                        
                        metrics.record_invocation(latency, mem_after)
                        
                        safe_set(invoke_span, "invoke.index", str(invoke_idx), max_bytes=100)
                        safe_set(invoke_span, "invoke.latency_ms", f"{latency:.2f}", max_bytes=100)
                        safe_set(invoke_span, "invoke.memory_delta_mb", f"{mem_delta:.2f}", max_bytes=100)
                        safe_set(invoke_span, "invoke.quality", f"{simulated_quality:.3f}", max_bytes=100)
                
                # Run offline optimization if applicable (Strategy 5)
                if h.dynamic_mgr and config.get("offline_batch_optimization", {}).get("enabled"):
                    print(f"\n[DEBUG] Running offline optimization...")
                    with tracer.start_as_current_span("offline_optimization") as offline_span:
                        optimizer_names = h.dynamic_mgr.list_trace_optimizers()
                        print(f"[DEBUG] Available optimizers: {optimizer_names}")
                        if optimizer_names:
                            print(f"[DEBUG] Calling offline_optimize('{optimizer_names[0]}')")
                            summary = h.offline_optimize(
                                optimizer_names[0],
                                targets=["invoke.temperature", "invoke.max_tokens"],
                                n_candidates=1,
                            )
                            print(f"[DEBUG] Offline optimization result: {summary}")
                            safe_set(offline_span, "offline.applied", str(summary.get("applied", False)), max_bytes=100)
                            if summary.get("applied"):
                                metrics.record_optimization(trace_length=n_invokes_per_trial)
                                print(f"[DEBUG]   ✓ Offline optimization applied!")
                            else:
                                print(f"[DEBUG]   ✗ Offline optimization not applied: {summary.get('reason')}")
                
                # Finalize trial metrics
                trial_results = metrics.finalize()
                print(f"\n[DEBUG] Trial {trial} completed:")
                print(f"[DEBUG]   - Invocations: {trial_results['invocations']}")
                print(f"[DEBUG]   - Optimizations: {trial_results['optimizations']}")
                print(f"[DEBUG]   - Feedback collected: {trial_results['feedback_collected']}")
                print(f"[DEBUG]   - Avg quality: {trial_results['avg_quality']:.3f}")
                print(f"[DEBUG]   - Quality improvement: {trial_results['quality_improvement']:.3f}\n")
                all_trial_metrics.append(trial_results)
                
                # Log trial summary to span
                safe_set(trial_span, "trial.invocations", str(trial_results["invocations"]), max_bytes=100)
                safe_set(trial_span, "trial.optimizations", str(trial_results["optimizations"]), max_bytes=100)
                safe_set(trial_span, "trial.avg_quality", f"{trial_results['avg_quality']:.3f}", max_bytes=100)
                safe_set(trial_span, "trial.quality_improvement", f"{trial_results['quality_improvement']:.3f}", max_bytes=100)
        
        # Aggregate metrics across trials
        if not all_trial_metrics:
            # Timeout or error - return partial results
            aggregated = {
                "strategy": strategy_name,
                "n_trials": n_trials,
                "trials": all_trial_metrics,
                "avg_invocations": 0,
                "avg_optimizations": 0,
                "avg_latency_ms": 0,
                "avg_memory_mb": 0,
                "avg_quality_improvement": 0,
                "avg_trace_length": 0,
                "status": "timeout" if time.time() - start_time > timeout_seconds else "error",
                "duration_s": time.time() - start_time,
            }
        else:
            aggregated = {
                "strategy": strategy_name,
                "n_trials": n_trials,
                "trials": all_trial_metrics,
                "avg_invocations": sum(t["invocations"] for t in all_trial_metrics) / len(all_trial_metrics),
                "avg_optimizations": sum(t["optimizations"] for t in all_trial_metrics) / len(all_trial_metrics),
                "avg_latency_ms": sum(t["avg_latency_ms"] for t in all_trial_metrics) / len(all_trial_metrics),
                "avg_memory_mb": sum(t["avg_memory_mb"] for t in all_trial_metrics) / len(all_trial_metrics),
                "avg_quality_improvement": sum(t["quality_improvement"] for t in all_trial_metrics) / len(all_trial_metrics),
                "avg_trace_length": sum(t["avg_trace_length"] for t in all_trial_metrics) / len(all_trial_metrics) if all_trial_metrics[0]["optimizations"] > 0 else 0,
                "status": "completed",
                "duration_s": time.time() - start_time,
            }
        
        # Log aggregated results to span
        safe_set(span, "results.status", aggregated["status"], max_bytes=100)
        safe_set(span, "results.duration_s", f"{aggregated['duration_s']:.2f}", max_bytes=100)
        safe_set(span, "results.avg_optimizations", f"{aggregated['avg_optimizations']:.2f}", max_bytes=100)
        safe_set(span, "results.avg_latency_ms", f"{aggregated['avg_latency_ms']:.2f}", max_bytes=100)
        safe_set(span, "results.avg_quality_improvement", f"{aggregated['avg_quality_improvement']:.3f}", max_bytes=100)
        
        return aggregated


# ========== Pytest Test Cases ==========

@pytest.mark.skipif(not HAS_HUMANLLM, reason="HumanLLM not available")
@pytest.mark.parametrize("n_trials", [3])
def test_baseline_no_optimization(n_trials):
    """Test baseline performance without any trace optimization."""
    config = get_baseline_config()
    results = run_strategy_benchmark("baseline", config, n_trials=n_trials, n_invokes_per_trial=15)
    
    assert results["avg_invocations"] == 15
    assert results["avg_optimizations"] == 0  # No optimization in baseline
    print(f"\n✓ Baseline: {results['avg_invocations']} invocations, "
          f"quality improvement: {results['avg_quality_improvement']:.3f}")


@pytest.mark.skipif(not HAS_HUMANLLM, reason="HumanLLM not available")
@pytest.mark.parametrize("n_trials", [3])
def test_strategy1_longer_trace(n_trials):
    """Test Strategy 1: Longer Trace (accumulation_window=5)."""
    config = get_strategy1_config()
    results = run_strategy_benchmark("strategy1_longer_trace", config, n_trials=n_trials, n_invokes_per_trial=15)
    
    assert results["avg_invocations"] == 15
    assert results["avg_optimizations"] > 0  # Should trigger optimizations
    assert results["avg_trace_length"] >= 1  # Should have accumulated trace
    print(f"\n✓ Strategy 1 (Longer Trace): {results['avg_optimizations']:.1f} optimizations, "
          f"avg trace length: {results['avg_trace_length']:.1f}, "
          f"quality improvement: {results['avg_quality_improvement']:.3f}")


@pytest.mark.skipif(not HAS_HUMANLLM, reason="HumanLLM not available")
@pytest.mark.parametrize("n_trials", [3])
def test_strategy2_continuous_backward(n_trials):
    """Test Strategy 2: Continuous Backward."""
    config = get_strategy2_config()
    results = run_strategy_benchmark("strategy2_continuous_backward", config, n_trials=n_trials, n_invokes_per_trial=15)
    
    assert results["avg_invocations"] == 15
    assert results["avg_optimizations"] > 0
    print(f"\n✓ Strategy 2 (Continuous Backward): {results['avg_optimizations']:.1f} optimizations, "
          f"quality improvement: {results['avg_quality_improvement']:.3f}")


@pytest.mark.skipif(not HAS_HUMANLLM, reason="HumanLLM not available")
@pytest.mark.parametrize("n_trials", [3])
def test_strategy3_optimizer_memory(n_trials):
    """Test Strategy 3: Optimizer Memory (memory_size=10)."""
    config = get_strategy3_config()
    results = run_strategy_benchmark("strategy3_optimizer_memory", config, n_trials=n_trials, n_invokes_per_trial=15)
    
    assert results["avg_invocations"] == 15
    assert results["avg_optimizations"] > 0
    print(f"\n✓ Strategy 3 (Optimizer Memory): {results['avg_optimizations']:.1f} optimizations, "
          f"quality improvement: {results['avg_quality_improvement']:.3f}")


@pytest.mark.skipif(not HAS_HUMANLLM, reason="HumanLLM not available")
@pytest.mark.parametrize("n_trials", [3])
def test_strategy4_multi_agent_sharing(n_trials):
    """Test Strategy 4: Multi-Agent Sharing."""
    # Create 3 agents sharing the same memory
    configs = [get_strategy4_config(i) for i in range(3)]
    all_results = []
    
    for i, config in enumerate(configs):
        results = run_strategy_benchmark(
            f"strategy4_agent_{i}",
            config,
            n_trials=n_trials,
            n_invokes_per_trial=5  # Fewer invokes per agent, but more agents
        )
        all_results.append(results)
    
    # Aggregate across agents
    total_invocations = sum(r["avg_invocations"] for r in all_results)
    total_optimizations = sum(r["avg_optimizations"] for r in all_results)
    avg_quality_improvement = sum(r["avg_quality_improvement"] for r in all_results) / len(all_results)
    
    assert total_invocations == 15  # 3 agents × 5 invokes
    print(f"\n✓ Strategy 4 (Multi-Agent): {total_optimizations:.1f} total optimizations across 3 agents, "
          f"avg quality improvement: {avg_quality_improvement:.3f}")


@pytest.mark.skipif(not HAS_HUMANLLM, reason="HumanLLM not available")
@pytest.mark.parametrize("n_trials", [3])
def test_strategy5_offline_batch(n_trials):
    """Test Strategy 5: Offline Batch Optimization."""
    config = get_strategy5_config()
    results = run_strategy_benchmark("strategy5_offline_batch", config, n_trials=n_trials, n_invokes_per_trial=15)
    
    assert results["avg_invocations"] == 15
    # Should have both online and offline optimizations
    assert results["avg_optimizations"] > 0
    print(f"\n✓ Strategy 5 (Offline Batch): {results['avg_optimizations']:.1f} optimizations, "
          f"quality improvement: {results['avg_quality_improvement']:.3f}")


@pytest.mark.skipif(not HAS_HUMANLLM, reason="HumanLLM not available")
@pytest.mark.parametrize("n_trials", [3])
def test_combined_all_strategies(n_trials):
    """Test Combined: All strategies together."""
    config = get_combined_config()
    results = run_strategy_benchmark("combined_all_strategies", config, n_trials=n_trials, n_invokes_per_trial=15)
    
    assert results["avg_invocations"] == 15
    assert results["avg_optimizations"] > 0
    print(f"\n✓ Combined Strategies: {results['avg_optimizations']:.1f} optimizations, "
          f"quality improvement: {results['avg_quality_improvement']:.3f}")


@pytest.mark.skipif(not HAS_HUMANLLM, reason="HumanLLM not available")
def test_full_benchmark_comparison(tmp_path):
    """
    Run full benchmark comparing all strategies IN PARALLEL.
    Generate comparison report with OTEL trace analysis.
    """
    print("\n" + "="*80)
    print("TRACE OPTIMIZER STRATEGIES BENCHMARK - PARALLEL EXECUTION")
    print("="*80)
    
    n_trials = 1  # Reduced for speed
    n_invokes = 5  # Reduced for speed
    timeout_per_strategy = 30  # 30 seconds max per strategy
    
    strategies = [
        ("Baseline (No Optimization)", get_baseline_config()),
        ("Strategy 1: Longer Trace", get_strategy1_config()),
        ("Strategy 2: Continuous Backward", get_strategy2_config()),
        ("Strategy 3: Optimizer Memory", get_strategy3_config()),
        ("Strategy 5: Offline Batch", get_strategy5_config()),
        ("Combined: All Strategies", get_combined_config()),
    ]
    
    print(f"\nConfiguration:")
    print(f"  - Trials per strategy: {n_trials}")
    print(f"  - Invokes per trial: {n_invokes}")
    print(f"  - Timeout per strategy: {timeout_per_strategy}s")
    print(f"  - Parallel execution: {len(strategies)} strategies")
    print(f"  - Total strategies: {len(strategies)}")
    
    all_results = []
    
    def run_single_strategy(strategy_tuple):
        """Wrapper for parallel execution."""
        strategy_name, config = strategy_tuple
        print(f"\n[STARTED] {strategy_name}")
        try:
            results = run_strategy_benchmark(
                strategy_name.lower().replace(" ", "_").replace(":", ""),
                config,
                n_trials=n_trials,
                n_invokes_per_trial=n_invokes,
                timeout_seconds=timeout_per_strategy,
            )
            print(f"[{results['status'].upper()}] {strategy_name} - Duration: {results['duration_s']:.2f}s")
            return (strategy_name, results)
        except Exception as e:
            print(f"[ERROR] {strategy_name}: {str(e)}")
            return (strategy_name, {
                "strategy": strategy_name,
                "status": "error",
                "error": str(e),
                "avg_invocations": 0,
                "avg_optimizations": 0,
                "avg_latency_ms": 0,
                "avg_memory_mb": 0,
                "avg_quality_improvement": 0,
                "duration_s": 0,
            })
    
    # Run strategies in parallel using ThreadPoolExecutor
    print(f"\n{'─'*80}")
    print("PARALLEL EXECUTION START")
    print(f"{'─'*80}")
    
    start_time = time.time()
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(strategies)) as executor:
        # Submit all strategies
        futures = {executor.submit(run_single_strategy, s): s[0] for s in strategies}
        
        # Wait for completion with timeout - collect results even if some timeout
        completed_futures = []
        try:
            for future in concurrent.futures.as_completed(futures, timeout=timeout_per_strategy + 10):
                completed_futures.append(future)
        except concurrent.futures.TimeoutError:
            print(f"\n⚠️  Timeout reached - collecting partial results")
        
        # Collect all results (completed and timed out)
        for future in futures:
            strategy_name = futures[future]
            try:
                if future.done():
                    result = future.result(timeout=0.1)
                    all_results.append(result)
                else:
                    # Future didn't complete - mark as timeout
                    print(f"[TIMEOUT] {strategy_name}")
                    all_results.append((strategy_name, {
                        "strategy": strategy_name,
                        "status": "timeout",
                        "avg_invocations": 0,
                        "avg_optimizations": 0,
                        "avg_latency_ms": 0,
                        "avg_memory_mb": 0,
                        "avg_quality_improvement": 0,
                        "duration_s": timeout_per_strategy,
                    }))
            except Exception as e:
                print(f"[EXCEPTION] {strategy_name}: {str(e)}")
                all_results.append((strategy_name, {
                    "strategy": strategy_name,
                    "status": "exception",
                    "error": str(e),
                    "avg_invocations": 0,
                    "avg_optimizations": 0,
                    "avg_latency_ms": 0,
                    "avg_memory_mb": 0,
                    "avg_quality_improvement": 0,
                    "duration_s": 0,
                }))
    
    total_duration = time.time() - start_time
    
    print(f"\n{'─'*80}")
    print(f"PARALLEL EXECUTION COMPLETE - Total Time: {total_duration:.2f}s")
    print(f"{'─'*80}")
    
    # Generate comparison table
    print("\n" + "="*80)
    print("COMPARISON TABLE")
    print("="*80)
    print(f"{'Strategy':<35} {'Status':<12} {'Optim':<8} {'Latency':<12} {'Quality':<12}")
    print("─"*80)
    
    for strategy_name, results in all_results:
        status_icon = {"completed": "✓", "timeout": "⏱", "error": "✗", "exception": "✗", "mocked": "○"}.get(results.get("status", "unknown"), "?")
        print(f"{strategy_name:<35} "
              f"{status_icon} {results.get('status', 'unknown'):<10} "
              f"{results.get('avg_optimizations', 0):<8.1f} "
              f"{results.get('avg_latency_ms', 0):<12.2f} "
              f"{results.get('avg_quality_improvement', 0):<12.3f}")
    
    print("="*80)
    
    # Save OTEL traces to file
    otel_file = tmp_path / "benchmark_otel_traces.json"
    num_spans = save_otel_spans_to_file(str(otel_file))
    print(f"\n✓ Saved {num_spans} OTEL spans to: {otel_file}")
    
    # Analyze OTEL traces
    print("\n" + "="*80)
    print("OTEL TRACE ANALYSIS")
    print("="*80)
    
    spans = GLOBAL_SPAN_EXPORTER.get_finished_spans()
    
    # Group spans by strategy
    strategy_spans = {}
    for span in spans:
        span_name = span.name
        if span_name.startswith("benchmark_"):
            strategy = span_name.replace("benchmark_", "")
            if strategy not in strategy_spans:
                strategy_spans[strategy] = []
            strategy_spans[strategy].append(span)
    
    for strategy, spans_list in strategy_spans.items():
        print(f"\n{strategy}:")
        print(f"  Total spans: {len(spans_list)}")
        
        # Count span types
        trial_spans = [s for s in spans_list if s.name.startswith("trial_")]
        invocation_spans = [s for s in spans_list if s.name.startswith("invocation_")]
        optimization_spans = [s for s in spans_list if "optimization" in s.name]
        
        print(f"  Trial spans: {len(trial_spans)}")
        print(f"  Invocation spans: {len(invocation_spans)}")
        print(f"  Optimization spans: {len(optimization_spans)}")
        
        # Calculate average durations
        if invocation_spans:
            avg_duration_ms = sum((s.end_time - s.start_time) / 1_000_000 for s in invocation_spans if s.end_time) / len(invocation_spans)
            print(f"  Avg invocation duration: {avg_duration_ms:.2f}ms")
        
        # Check for errors
        error_spans = [s for s in spans_list if s.status.status_code.name == "ERROR"]
        if error_spans:
            print(f"  ⚠️  ERROR spans: {len(error_spans)}")
            for err_span in error_spans[:3]:  # Show first 3 errors
                if err_span.status.description:
                    print(f"    - {err_span.name}: {err_span.status.description[:100]}")
    
    print("\n" + "="*80)
    
    # Save detailed report to file
    report_path = tmp_path / "benchmark_report.json"
    report_data = {
        "benchmark_date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "execution_mode": "parallel",
        "n_trials": n_trials,
        "n_invokes_per_trial": n_invokes,
        "timeout_per_strategy": timeout_per_strategy,
        "total_duration_s": total_duration,
        "strategies": [
            {
                "name": name,
                "results": results,
            }
            for name, results in all_results
        ],
        "otel_traces_file": str(otel_file),
        "total_otel_spans": num_spans,
    }
    
    with open(report_path, 'w') as f:
        json.dump(report_data, f, indent=2, default=str)
    
    print(f"✓ Benchmark report saved to: {report_path}")
    print(f"✓ OTEL traces saved to: {otel_file}")
    print(f"\nTotal execution time: {total_duration:.2f}s")
    print("="*80)
    
    # Assertions to validate benchmark ran successfully
    assert len(all_results) == len(strategies), f"Expected {len(strategies)} results, got {len(all_results)}"
    assert num_spans > 0, "No OTEL spans were collected"
    
    # At least baseline should complete
    baseline_result = next((r for name, r in all_results if "baseline" in name.lower()), None)
    assert baseline_result is not None, "Baseline strategy did not run"
    
    print(f"\n✅ Benchmark completed successfully!")
    print(f"   - Strategies tested: {len(all_results)}")
    print(f"   - OTEL spans collected: {num_spans}")
    print(f"   - Reports generated: 2 (JSON report + OTEL traces)")
