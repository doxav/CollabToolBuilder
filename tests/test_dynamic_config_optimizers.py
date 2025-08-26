# tests/test_dcm_trace_integration.py
# This suite aligns its expectations with Trace’s optimizer/test patterns
# (multi-candidate generation, step/backward loops, memory logging, ParameterNode updates),
# and adapts them to DynamicConfigManager’s “trace” modification semantics. :contentReference[oaicite:0]{index=0}

import pytest
from types import SimpleNamespace
from typing import List, Dict
import unittest

from utils.human_llm import DynamicConfigManager, _TraceOptimizerAdapter, HelpUsageTracker, HumanLLM

# ---------------------
# Minimal usage tracker
# ---------------------
class DummyUsageTracker:
    def can_use(self, help_type):
        return True
    def record_usage(self, help_type):
        pass


# ------------------------------
# Trace-like Optimizer Adapters
# ------------------------------
class H1TraceLike(_TraceOptimizerAdapter):
    """
    Simulates an ONLINE (H1) advisor call consistent with Trace usage:
    returns a *candidate* that reflects small, safe edits affecting:
      - invoke kwargs (num_parallel_inferences, temperature_max, generation_technique)
      - system_prompt
      - dynamic_llm_config.* patch
      - self.config (config.* paths)
      - rule_evaluator replacement
    Also snapshots observables (aligning with Trace tests’ emphasis on rich context).
    """
    def step(self, context: Dict, targets: List[str], n_candidates: int = 1, constraints=None) -> List[Dict]:
        self._state["calls"] += 1
        # Capture declared observables (only simple keys here)
        self._last_context_snapshot = {k: context.get(k) for k in (self._observables or []) if isinstance(k, str)}
        # A single candidate with edits over multiple surfaces:
        edits = [
            # Upcoming execution knobs
            {"target": "invoke.num_parallel_inferences", "op": "set", "value": 3},
            {"target": "invoke.temperature_max", "op": "set", "value": 0.85},
            {"target": "invoke.generation_technique", "op": "set", "value": "temperature_variation"},
            # Prompt + runtime config patch (agent’s dynamic_llm_config)
            {"target": "system_prompt", "op": "set", "value": "SYSTEM: prefer concise answers"},
            {"target": "dynamic_llm_config.generation_techniques", "op": "set", "value": ["temperature_variation", "multi_experts"]},
            # Edit a help config’s rule + its modifications (triggers/mods)
            {"target": "config.help_online.rules.regex.patterns", "op": "set", "value": ["OPTIMIZER_CALL"]},
            {"target": "config.help_online.modifications", "op": "set", "value": [{"type": "note", "by": "H1"}]},
            # Change an evaluator function (confidence)
            {"target": "rule_evaluator.confidence", "op": "set", "value": (lambda cfg, ctx: True)},
            # External trace node observation
            {"target": "trace.node.external_sensor", "op": "set", "value": {"level": 7}},
        ]
        return [{"edits": edits, "meta": {"optimizer": self.name, "candidate_idx": 0}}]


class H2TraceLike(_TraceOptimizerAdapter):
    """
    Simulates an OFFLINE (H2) learning pass: edits focus on post-run tuning,
    potentially reducing parallelism, adjusting history/rules, and prepending a prompt marker.
    """
    def step(self, context: Dict, targets: List[str], n_candidates: int = 1, constraints=None) -> List[Dict]:
        self._state["calls"] += 1
        self._last_context_snapshot = {k: context.get(k) for k in (self._observables or []) if isinstance(k, str)}
        edits = [
            {"target": "invoke.num_parallel_inferences", "op": "set", "value": 2},
            {"target": "config.help_offline.rules.history.lookback", "op": "set", "value": 5},
            {"target": "config.help_offline.modifications", "op": "set", "value": [{"type": "learned_patch", "source": "H2"}]},
            {"target": "system_prompt", "op": "prepend", "value": "[H2-Learned] "},
        ]
        return [{"edits": edits, "meta": {"optimizer": self.name, "candidate_idx": 0}}]


# -------------------------
# Shared test fixtures
# -------------------------
@pytest.fixture
def manager():
    cfg = {
        # Two help entries to show phase separation (H1 online vs H2 offline)
        "help_online": {
            "phase": "pre_inference",
            "rules": {},  # Always fire for tests
            "modifications": [
                # H1 advisor: call our trace-like optimizer
                {"type": "trace", "optimizer": "h1", "targets": [], "n_candidates": 1,
                 "observables": ["system_prompt", "user_message", "trace.node.external_sensor"],
                 "optimizables": ["invoke.num_parallel_inferences", "dynamic_llm_config.generation_techniques"]}
            ],
        },
        "help_offline": {
            "phase": "post_inference",
            "rules": {},  # Always fire for tests
            "modifications": [
                # H2 job: called after inference (or via explicit post phase)
                {"type": "trace", "optimizer": "h2", "targets": [], "n_candidates": 1,
                 "observables": ["trace.node.run_metric", "user_message"],
                 "optimizables": ["invoke.num_parallel_inferences"]}
            ],
        },
    }
    m = DynamicConfigManager(cfg, DummyUsageTracker(), default_config={})
    # Register persistent adapters under the names referenced above
    m._trace_optimizers["h1"] = H1TraceLike("h1", {"observables": ["system_prompt", "user_message", "trace.node.external_sensor"]})
    m._trace_optimizers["h2"] = H2TraceLike("h2", {"observables": ["trace.node.run_metric", "user_message"]})
    return m


# -------------------------
# H1 (online) end-to-end
# -------------------------
def test_h1_online_end_to_end(manager):
    ctx = {
        "user_message": "Please summarize this long text.",
        "system_prompt": "SYSTEM: base",
        "trace.node.external_sensor": {"level": 0},
        "llm_outputs": []
    }

    mods = manager.evaluate_triggers(ctx, phase="pre_inference")

    # 1) Execution parameters for *upcoming* call
    assert (mods.get("invoke_kwargs") or {}).get("num_parallel_inferences") == 3
    assert (mods.get("invoke_kwargs") or {}).get("temperature_max") == 0.85
    assert (mods.get("invoke_kwargs") or {}).get("generation_technique") == "temperature_variation"

    # 2) Prompt and dynamic_llm_config patch
    assert mods.get("system_prompt") == "SYSTEM: prefer concise answers"
    dyn_patch = mods.get("dynamic_llm_config_patch") or {}
    assert dyn_patch.get("generation_techniques") == ["temperature_variation", "multi_experts"]

    # 3) Agent config (rules + modifications in a help entry)
    assert manager.config["help_online"]["rules"]["regex"]["patterns"] == ["OPTIMIZER_CALL"]
    assert manager.config["help_online"]["modifications"] == [{"type": "note", "by": "H1"}]

    # 4) rule_evaluator override (confidence → always True)
    conf_eval = manager.rule_evaluators.get("confidence")
    assert callable(conf_eval) and conf_eval({}, {}) is True

    # 5) Observables snapshot contains declared vars (including external trace node)
    h1_adapter = manager._trace_optimizers["h1"]
    snap = getattr(h1_adapter, "_last_context_snapshot", {})
    assert snap.get("system_prompt") == "SYSTEM: base"
    assert "trace.node.external_sensor" in snap and isinstance(snap["trace.node.external_sensor"], dict)


# -------------------------
# H2 (offline) end-to-end
# -------------------------
def test_h2_offline_end_to_end(manager):
    # Simulate recorded outcomes the offline learner would aggregate
    manager.record_outcome({"user_message": "r1"}, {"invoke_kwargs": {"t": 0.8}}, {"score": 0.5})
    manager.record_outcome({"user_message": "r2"}, {"invoke_kwargs": {"t": 0.6}}, {"score": 0.6})

    # Post-inference phase triggers the H2 optimizer
    ctx = {
        "user_message": "post-run",
        "system_prompt": "BASE",
        "trace.node.run_metric": {"latency": 120}
    }
    mods = manager.evaluate_triggers(ctx, phase="post_inference")

    # H2 reduces parallelism for next runs
    assert (mods.get("invoke_kwargs") or {}).get("num_parallel_inferences") == 2

    # H2 prepends a marker to the system prompt (manager maps to *_prepend)
    # Either mapped as a prepend list or left in generic patches (implementation-dependent)
    assert "system_prompt_prepend" in mods or "patches" in mods or mods.get("system_prompt")

    # H2 also edits offline help rules/mods in manager.config
    assert manager.config["help_offline"]["rules"]["history"]["lookback"] == 5
    assert manager.config["help_offline"]["modifications"] == [{"type": "learned_patch", "source": "H2"}]


# ---------------------------------------------
# Evaluator swap influences subsequent rule use
# ---------------------------------------------
def test_evaluator_swap_affects_rule_decisions(manager):
    """
    After H1 replaced the 'confidence' evaluator with a lambda True,
    any help entry relying on the confidence rule should now pass.
    """
    # 1) First, run H1 once to perform the swap
    _ = manager.evaluate_triggers(
        {"user_message": "trigger swap", "system_prompt": "BASE"}, phase="pre_inference"
    )
    # 2) Add a help that requires confidence to pass
    manager.config["help_conf"] = {
        "phase": "pre_inference",
        "rules": {"confidence": {"method": "self-consistency", "threshold": 0.1}},
        "modifications": {"invoke_kwargs": {"foo": "bar"}}
    }
    mods = manager.evaluate_triggers({"user_message": "x", "llm_outputs": []}, phase="pre_inference")
    # Because evaluator now returns True, modifications should include help_conf changes
    # (dict-merge behavior depends on insertion order; we just check presence)
    assert "invoke_kwargs" in mods


# ---------------------------------------------------
# Per-optimizer observables/optimizables configuration
# ---------------------------------------------------
def test_per_optimizer_observables_and_optimizables(manager):
    # Update h1 adapter’s per-optimizer sets
    manager.set_optimizer_observables("h1", ["system_prompt", "trace.node.external_sensor"])
    manager.set_optimizer_optimizables("h1", ["invoke.num_parallel_inferences", "invoke.temperature_max"])

    # Re-run H1; ensure snapshot respects the new list and invoke knobs still apply
    ctx = {
        "user_message": "observe limited set",
        "system_prompt": "SYS",
        "trace.node.external_sensor": {"level": 9}
    }
    mods = manager.evaluate_triggers(ctx, phase="pre_inference")

    # knobs present
    ivk = mods.get("invoke_kwargs") or {}
    assert ivk.get("num_parallel_inferences") == 3 and ivk.get("temperature_max") == 0.85

    # snapshot matches new observables
    snap = getattr(manager._trace_optimizers["h1"], "_last_context_snapshot", {})
    assert set(snap.keys()) == {"system_prompt", "trace.node.external_sensor"}

def test_legacy_modifications_dict_path(manager):
    manager.config["help_legacy"] = {
        "phase": "pre_inference",
        "rules": {},
        "modifications": {"invoke_kwargs": {"temperature_max": 0.77}}
    }
    mods = manager.evaluate_triggers({"user_message": "x"}, phase="pre_inference")
    assert (mods.get("invoke_kwargs") or {}).get("temperature_max") == 0.77

class TwoCand(_TraceOptimizerAdapter):
    def step(self, context, targets, n_candidates=2, constraints=None):
        return [
            {"edits": [{"target":"invoke.num_parallel_inferences","op":"set","value":5}]},
            {"edits": [{"target":"invoke.num_parallel_inferences","op":"set","value":7}]},
        ]

def test_n_candidates_first_selected(manager):
    manager._trace_optimizers["multi"] = TwoCand("multi", {})
    manager.config["help_multi"] = {
        "phase": "pre_inference", "rules": {},
        "modifications": [{"type":"trace","optimizer":"multi","n_candidates":2}]
    }
    mods = manager.evaluate_triggers({"user_message":"x"}, phase="pre_inference")
    assert (mods.get("invoke_kwargs") or {}).get("num_parallel_inferences") == 5


class DummyTrace:
    """Minimal Trace-like optimizer returning one candidate with temperature + user_message edit."""
    def step(self, context, targets, n_candidates=1, constraints=None):
        return [{"edits": [
            {"target": "invoke.temperature", "op": "set", "value": 0.13},
            {"target": "user_message", "op": "append", "value": "\n\n[H1-DUMMY]"}
        ]}]


class TestTraceDynamicConfig(unittest.TestCase):
    def _mgr(self, plug_trace=False):
        cfg = {
            "trace_optimizers": [{"name": "opt", **({"trace_obj": DummyTrace()} if plug_trace else {})}],
            "trace_default_optimizer": "opt",
            "h1_block": {
                "phase": "pre_inference",
                "rules": {},  # always true
                "modifications": [
                    {"type": "trace", "optimizer": "opt",
                     "targets": ["invoke.temperature", "user_message"], "n_candidates": 1}
                ]
            }
        }
        return DynamicConfigManager(cfg, HelpUsageTracker(), default_config={})

    def test_h1_temperature_mapping(self):
        mgr = self._mgr(plug_trace=True)
        ctx = {"user_message": "Hi", "invoke_kwargs": {}}
        mods = mgr.evaluate_triggers(ctx, phase="pre_inference")
        # Apply with a lightweight HumanLLM shell (bypass heavy __init__)
        h = HumanLLM.__new__(HumanLLM)
        import logging; h.logger = logging.getLogger(__name__)
        h.dynamic_mgr = mgr
        h.temperature_min, h.temperature_max = 0.0, 1.0
        h.num_parallel_inferences = 1
        # use the method defined on the class you pasted
        HumanLLM._apply_modifications(h, mods, ctx, phase="pre_inference")
        self.assertEqual(h.temperature_min, 0.13)
        self.assertEqual(h.temperature_max, 0.13)
        self.assertIn("[H1-DUMMY]", ctx["user_message"])

    def test_h2_feedback_and_offline_opt(self):
        mgr = self._mgr()
        # record a couple outcomes
        mgr.record_outcome({"case": 1}, {"invoke_kwargs": {"temperature": 0.2}}, {"quality": 0.7})
        mgr.log_user_correction("HumanLLM", "-foo\n+bar", "user")
        fb = mgr.collect_feedback({"user_message": "Fallback"})
        self.assertTrue(len(fb) >= 1)
        self.assertEqual(fb[0].get("type"), "user_diff")
        # offline optimize with fallback adapter (no trace_obj)
        summary = mgr.offline_optimize(targets=["config.some.path"])
        self.assertIn("applied", summary)