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
class _DummyTrace:
    """A tiny stub that captures call-time info and returns a single candidate."""
    def __init__(self, edits, meta=None):
        self.last_context = None
        self.last_targets = None
        self.last_constraints = None
        self._edits = edits
        self._meta = meta or {}
    def step(self, context, targets, n_candidates=1, constraints=None):
        self.last_context, self.last_targets, self.last_constraints = context, targets, constraints
        return [{"edits": list(self._edits), "meta": dict(self._meta)}]

def _make_mgr(cfg=None):
    return DynamicConfigManager(cfg or {}, HelpUsageTracker(), default_config={})

def test_trace_invoke_temperature_and_quota_applied():
    mgr = _make_mgr()
    trace = _DummyTrace(edits=[
        {"target": "invoke.temperature", "op": "set", "value": 0.8},
        {"target": "invoke.quota", "op": "set", "value": 1000},
    ])
    h1 = H1TraceLike(name="H1", config={"trace_obj": trace})
    context = {"invoke": {"temperature": 0.5, "quota": 500}}
    targets = ["invoke.temperature", "invoke.quota"]
    candidates = h1.step(context, targets)
    assert len(candidates) == 1
    assert candidates[0]["edits"] == [
        {"target": "invoke.temperature", "op": "set", "value": 0.8},
        {"target": "invoke.quota", "op": "set", "value": 1000},
    ]

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
        # If a real Trace object is plugged, delegate to base behavior (proxy).
        if getattr(self, "_trace_obj", None) is not None:
            return super().step(context=context, targets=targets, n_candidates=n_candidates, constraints=constraints)

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

# -----------------------------------------------------------------------------
# New: End-to-end HumanLLM integration, backward-trace (OptoPrime-like) checks,
#       and Human-in-the-loop toggling via a Trace-style calling surface.
# -----------------------------------------------------------------------------

class DummyLLM:
    """Very small LLM stub used by HumanLLM; satisfies .invoke and .with_config."""
    def __init__(self, name="dummy"):
        self.model_name = name
        self.calls = []
    def with_config(self, **kwargs):
        return self
    def invoke(self, messages, **kwargs):
        # record the last messages for assertions if needed
        self.calls.append({"messages": messages, "kwargs": kwargs})
        # emulate a langchain-like response object with .content
        return SimpleNamespace(content="[DUMMY-OK]", generation_info={"finish_reason": "stop"})


class HumanAsTraceLLM:
    """
    Adapter to use HumanLLM in a Trace optimizer's call path.
    Trace optimizers often expect either:
      - llm.create(messages=..., max_tokens=...) OR
      - a callable llm(messages, max_tokens=...)
    We satisfy the latter by defining __call__ and delegating to HumanLLM.invoke.
    """
    def __init__(self, human: HumanLLM):
        self.human = human
    def __call__(self, messages, max_tokens=None):
        sys_msg = ""
        user_msg = ""
        for m in messages or []:
            if isinstance(m, dict):
                if m.get("role") == "system":
                    sys_msg = m.get("content") or ""
                if m.get("role") == "user":
                    user_msg = m.get("content") or ""
        # Drive the HumanLLM one round (no streaming, keep default selection)
        out = self.human.invoke(
            user_message=user_msg,
            system_prompt_override=sys_msg if sys_msg else None,
            stream_output=False,
            return_message_content_only=True,
        )
        # Trace call_llm(...) expects a string (or message.content)
        return out[0] if isinstance(out, list) else out


class OptoPrimeLikeStrict(_TraceOptimizerAdapter):
    """
    OFFLINE optimizer that requires backward trace nodes.
    Returns no candidates unless at least one record contains:
       record['context']['trace']['backward']['nodes'] (non-empty)
    When present, proposes a persistent edit to dynamic_llm_config.*.
    """
    def step(self, context: Dict, targets: List[str], n_candidates: int = 1, constraints=None) -> List[Dict]:
        recs = context.get("records", [])
        has_backward = any(
            isinstance(r.get("context", {}), dict) and
            isinstance(r["context"].get("trace", {}), dict) and
            isinstance(r["context"]["trace"].get("backward", {}), dict) and
            r["context"]["trace"]["backward"].get("nodes")
            for r in recs
        )
        if not has_backward:
            return []
        # Minimal persistent patch applied by DynamicConfigManager.offline_optimize(...)
        return [{"edits": [
            {"target": "dynamic_llm_config.generation_techniques", "op": "set", "value": ["temperature_variation"]}
        ]}]


def _human_for_tests(dynamic_llm_config=None):
    """Build a HumanLLM with a DummyLLM backend and a small dynamic config."""
    dcfg = dynamic_llm_config or {}
    dummy = DummyLLM()
    return HumanLLM(
        system_prompt="SYS",
        llmORchains_list={"default_llm": dummy, "premium_llm": dummy},
        num_parallel_inferences=1,
        selection_technique="best",
        temperature_min=0.1,
        temperature_max=0.3,
        dynamic_llm_config=dcfg,
    )


def test_human_llm_end_to_end_online_records_for_offline():
    """
    Demonstrates SELF optimization (online/in-process) applied in a real call,
    and verifies that an outcome record is produced to fuel later offline learning.
    """
    # Online advisor: set fixed temperature & prepend to system prompt
    class InlineH1(_TraceOptimizerAdapter):
        def step(self, context, targets, n_candidates=1, constraints=None):
            return [{"edits": [
                {"target": "invoke.temperature", "op": "set", "value": 0.21},
                {"target": "system_prompt", "op": "prepend", "value": "[H1-TEST] "}
            ]}]
    dcfg = {
        "online": {
            "phase": "pre_inference",
            "rules": {},
            "modifications": [{"type": "trace", "optimizer": "h1"}],
        }
    }
    h = _human_for_tests(dcfg)
    h.dynamic_mgr._trace_optimizers["h1"] = InlineH1("h1", {})
    # Run once
    out = h.invoke(user_message="hello", stream_output=False, return_message_content_only=True)
    assert isinstance(out, list) and out[0].startswith("[DUMMY-OK]")
    # Outcome is stored for offline learning
    assert h.dynamic_mgr._records, "Expected an outcome record to be logged by invoke(...)"
    # Ensure the modification (temperature) was captured in the record used for H2
    last = h.dynamic_mgr._records[-1]
    ivk = (last.get("modifications") or {}).get("invoke_kwargs") or {}
    assert ivk.get("temperature") == 0.21


def test_offline_optimize_wrapper_on_humanllm():
    """
    Demonstrates EXTERNAL optimization (off-policy/offline) called via the HumanLLM
    convenience wrapper, which delegates to DynamicConfigManager.offline_optimize(...).
    """
    class OfflineOne(_TraceOptimizerAdapter):
        def step(self, context, targets, n_candidates=1, constraints=None):
            # Persist a manager-level setting
            return [{"edits": [{"target": "config.some.flag", "op": "set", "value": True}]}]
    h = _human_for_tests({"offline": {"phase": "post_inference", "rules": {}, "modifications": []}})
    h.dynamic_mgr._trace_optimizers["o1"] = OfflineOne("o1", {})
    # Seed outcomes so offline has data
    h.dynamic_mgr.record_outcome({"user_message": "x"}, {"invoke_kwargs": {}}, {"score": 0.9})
    summary = h.offline_optimize("o1", targets=["config.some.flag"], n_candidates=1)
    assert summary.get("applied") is True
    assert h.dynamic_mgr.config.get("some", {}).get("flag") is True


def test_opto_prime_like_requires_backward_nodes_then_succeeds():
    """
    Shows the LIMIT & SOLUTION for optimizers that expect backward trace nodes.
    1) Without nodes -> no candidates (no persistent edits applied).
    2) With a minimal backward structure recorded -> succeeds, persistent patch applied.
    """
    m = DynamicConfigManager({}, HelpUsageTracker(), default_config={})
    m._trace_optimizers["optostrict"] = OptoPrimeLikeStrict("optostrict", {})
    # (1) No backward nodes yet
    m.record_outcome({"user_message": "r1"}, {"invoke_kwargs": {}}, {"score": 0.5})
    s1 = m.offline_optimize("optostrict", targets=["dynamic_llm_config.generation_techniques"])
    assert s1.get("applied") is False and s1.get("reason") in ("no_candidates", "no_records")
    # (2) Provide a minimal backward trace snippet
    bw = {"trace": {"backward": {"nodes": [{"id": "n1", "grad": 0.1}]}}}
    m.record_outcome(bw, {"invoke_kwargs": {}}, {"score": 0.6})
    s2 = m.offline_optimize("optostrict", targets=["dynamic_llm_config.generation_techniques"])
    assert s2.get("applied") is True
    assert m.config.get("dynamic_llm_defaults", {}).get("generation_techniques") == ["temperature_variation"]


def test_human_in_the_loop_toggle_via_trace_and_restore_after_call():
    """
    Demonstrates how to ACTIVATE/DEACTIVATE human-in-the-loop:
      - An online optimizer emits 'activate_human_intervention' (forces skip_rounds=0).
      - The change holds during the call and is restored afterwards.
    Also uses HumanAsTraceLLM to show HumanLLM as a Trace LLM surface.
    """
    class ToggleHITL(_TraceOptimizerAdapter):
        def step(self, context, targets, n_candidates=1, constraints=None):
            return [{"edits": [
                {"target": "activate_human_intervention", "op": "set", "value": True},
                {"target": "invoke.temperature", "op": "set", "value": 0.33},
            ]}]
    dcfg = {
        "hitl": {"phase": "pre_inference", "rules": {}, "modifications": [{"type": "trace", "optimizer": "hitl"}]}
    }
    h = _human_for_tests(dcfg)
    h.skip_rounds = 5  # automatic mode initially (no human)
    h.dynamic_mgr._trace_optimizers["hitl"] = ToggleHITL("hitl", {})
    # Use the Trace-style adapter surface
    wrapper = HumanAsTraceLLM(h)
    msg = wrapper([{"role": "system", "content": "SYS"}, {"role": "user", "content": "do something"}], max_tokens=64)
    assert isinstance(msg, str) and "[DUMMY-OK]" in msg
    # After the call, skip_rounds is restored to its original value
    assert h.skip_rounds == 5
    # Verify that the temperature edit was captured for the offline pass
    assert any((rec.get("modifications") or {}).get("invoke_kwargs", {}).get("temperature") == 0.33
               for rec in h.dynamic_mgr._records)


def test_offline_ignores_unknown_targets():
    """
    Shows that offline persistence only applies 'config.*' and 'dynamic_llm_config.*' targets.
    Unknown namespaces are ignored (applied=False).
    """
    class UnknownTargets(_TraceOptimizerAdapter):
        def step(self, context, targets, n_candidates=1, constraints=None):
            return [{"edits": [{"target": "not_supported.path", "op": "set", "value": 1}]}]
    m = DynamicConfigManager({}, HelpUsageTracker(), default_config={})
    m.record_outcome({"user_message": "x"}, {"invoke_kwargs": {}}, {"score": 1.0})
    m._trace_optimizers["u"] = UnknownTargets("u", {})
    s = m.offline_optimize("u", targets=["not_supported.path"])
    assert s.get("applied") is False

def test_trace_objective_and_parameter_roundtrip():
    """
    Optimizer sees a trace_spec (objective + parameters), proposes edits to trace.param.* and trace.objective,
    and DynamicConfigManager applies them to the adapter's registry.
    """
    trace = _DummyTrace(
        edits=[
            {"target": "trace.param.alpha.value", "op": "set", "value": 0.33},
            {"target": "trace.param.alpha.description", "op": "set", "value": "mixture weight"},
            {"target": "trace.objective", "op": "set", "value": "maximize ROUGE-L"},
            {"target": "invoke.temperature", "op": "set", "value": 0.1},
        ],
        meta={"requires_human": True},
    )
    cfg = {
        "trace_optimizers": [{
            "name": "opt1",
            "objective": "maximize score",
            "parameters": {
                "alpha": {"value": 0.2, "trainable": True, "description": "init"},
            },
            "trace_obj": trace,
        }],
        "hook": {
            "phase": "pre_inference",
            "rules": {},  # always on
            "modifications": [{
                "type": "trace",
                "optimizer": "opt1",
                "targets": ["invoke.temperature"],
                "n_candidates": 1,
                "constraints": {"bounds": {"alpha": (0.0, 1.0)}},
            }],
        }
    }
    mgr = _make_mgr(cfg)
    # sanity: adapter created
    assert mgr.list_trace_optimizers() == ["opt1"]
    # run
    ctx = {"user_message": "hello"}
    mods = mgr.evaluate_triggers(ctx, phase="pre_inference")
    # verify edits propagated into invoke kwargs
    assert mods["invoke_kwargs"]["temperature"] == 0.1
    # verify HIL flag from meta
    assert mods.get("activate_human_intervention") is True
    # verify the stub saw a trace_spec with objective/parameters
    spec = trace.last_context.get("trace_spec")
    assert spec and spec["objective"] == "maximize score"
    assert "alpha" in spec["parameters"] and spec["parameters"]["alpha"]["value"] == 0.2
    # verify adapter registry was updated by trace.param.* edits
    # access through public helpers
    adapter_name = mgr.list_trace_optimizers()[0]
    adapter = mgr._trace_optimizers[adapter_name]
    assert adapter.get_trace_spec()["objective"] == "maximize ROUGE-L"
    assert adapter.get_trace_spec()["parameters"]["alpha"]["value"] == 0.33
    assert adapter.get_trace_spec()["parameters"]["alpha"]["description"] == "mixture weight"
    # constraints were forwarded to the optimizer
    assert trace.last_constraints == {"bounds": {"alpha": (0.0, 1.0)}}


def test_self_vs_external_optimization_edits():
    """
    Demonstrate 'self optimization' (invoke/system prompt edits) vs 'external' (config/dynamic_llm_config edits).
    """
    edits = [
        {"target": "invoke.temperature", "op": "set", "value": 0.2},                 # self
        {"target": "system_prompt", "op": "set", "value": "NEW PROMPT"},             # self
        {"target": "config.some_switch", "op": "set", "value": True},                # external persistent
        {"target": "dynamic_llm_config.some_key", "op": "set", "value": "fast"},     # external runtime config
    ]
    trace = _DummyTrace(edits=edits)
    mgr = _make_mgr({
        "trace_optimizers": [{"name": "opt", "trace_obj": trace}],
        "hook": {"phase": "pre_inference", "rules": {}, "modifications": [{"type": "trace", "optimizer": "opt", "targets": ["invoke.temperature"]}]}
    })
    ctx = {"user_message": "x"}
    mods = mgr.evaluate_triggers(ctx, phase="pre_inference")
    # self
    assert mods["invoke_kwargs"]["temperature"] == 0.2
    assert mods["system_prompt"] == "NEW PROMPT"
    # external
    assert mgr.config.get("some_switch") is True
    assert mods["dynamic_llm_config_patch"]["some_key"] == "fast"


def test_offline_optimize_consumes_records_and_applies_edits():
    """
    H2: record outcomes then call offline_optimize; we simulate an optimizer that proposes persistent config edits.
    """
    edits = [
        {"target": "config.llm_defaults.max_calls", "op": "set", "value": 3},
        {"target": "dynamic_llm_config.default_temperature", "op": "set", "value": 0.25},
    ]
    trace = _DummyTrace(edits=edits)
    mgr = _make_mgr({
        "trace_default_optimizer": "opt",
        "trace_optimizers": [{"name": "opt", "trace_obj": trace}],
    })
    # feed a few outcomes
    for i in range(3):
        mgr.record_outcome({"user_message": f"m{i}"}, {"invoke_kwargs": {}}, {"score": i})
    res = mgr.offline_optimize()
    assert res["applied"] is True
    assert mgr.config["llm_defaults"]["max_calls"] == 3
    assert mgr.config.setdefault("dynamic_llm_defaults", {})["default_temperature"] == 0.25


def test_manager_helpers_update_objective_and_parameters():
    mgr = _make_mgr({
        "trace_optimizers": [{"name": "opt2", "objective": "maximize score", "parameters": {"beta": {"value": 1.0}}}]
    })
    mgr.set_optimizer_objective("opt2", "maximize return")
    mgr.update_optimizer_parameter("opt2", "beta", value=0.8, trainable=True, description="exploration factor")
    spec = mgr._trace_optimizers["opt2"].get_trace_spec()
    assert spec["objective"] == "maximize return"
    assert spec["parameters"]["beta"]["value"] == 0.8
    assert spec["parameters"]["beta"]["trainable"] is True
    assert spec["parameters"]["beta"]["description"] == "exploration factor"