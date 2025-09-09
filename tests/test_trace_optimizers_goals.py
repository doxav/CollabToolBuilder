import json
import copy
import math
import pytest

from langchain_openai import ChatOpenAI
from config import *  # use repo defaults (MODELS_CONFIG_LIST, etc.)
from utils.human_llm import HumanLLM
from utils.human_llm_config import HumanLLMConfig
from utils.constants import CHROMA_DATABASE


# Keep vector DB lightweight for tests
HumanLLMConfig().common_vectordb_config.db_type = CHROMA_DATABASE


def _llm_list():
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


def _dc_base():
    return {
        "trace_optimizers": [
            {
                "name": "inference_tuner",
                "optimizer_kind": "OptoPrimeV2",
                "parameters": [
                    {"parameter": "invoke.temperature", "value": 0.2, "info": {"bounds": [0.0, 0.9]}, "trainable": True},
                    {"parameter": "invoke.num_parallel_inferences", "value": 3, "info": {"bounds": [1, 3]}, "trainable": True},
                    {"parameter": "invoke.max_tokens", "value": 512, "info": {"bounds": [128, 2048]}, "trainable": True},
                    {"parameter": "dynamic_llm_config.default_temperature", "value": 0.2, "info": {"bounds": [0.0, 1.0]}, "trainable": True},
                    {"parameter": "dynamic_llm_config.annotations_help_trace_POST.rules.confidence.threshold", "value": 0.4, "info": {"bounds": [0.2, 0.8]}, "trainable": True},
                    {"parameter": "config.flags.low_latency", "value": False, "trainable": True},
                ],
                "objective": "maximize quality under latency budget",
            }
        ],
        "trace_default_optimizer": "inference_tuner",
    }


def test_quality_improves_temperature_towards_optimum(tmp_path):
    cfg = _dc_base()
    cfg["help_quality"] = {
        "phase": "pre_inference",
        "rules": {"regex": {"patterns": ["optimize", "improve"]}},
        "modifications": [
            {
                "type": "trace",
                "optimizer": "inference_tuner",
                "targets": ["invoke.temperature"],
                "n_candidates": 1,
            }
        ],
    }

    h = HumanLLM(agent_name="qa", llmORchains_list=_llm_list(), dynamic_llm_config=cfg)
    # Baseline temp from parameter registry
    baseline = 0.2

    def quality(temp: float) -> float:
        # unimodal quality peaking near 0.5
        return -abs(temp - 0.5)

    ctx = {"user_message": "please optimize", "system_prompt": "BASE"}
    mods = h.dynamic_mgr.evaluate_triggers(ctx, phase="pre_inference")
    new_t = (mods.get("invoke_kwargs") or {}).get("temperature")
    assert new_t is not None
    assert quality(new_t) > quality(baseline)


def test_latency_budget_reduces_parallelism(tmp_path):
    cfg = _dc_base()
    cfg["help_latency"] = {
        "phase": "pre_inference",
        "rules": {"regex": {"patterns": ["latency", "budget"]}},
        "modifications": [
            {
                "type": "trace",
                "optimizer": "inference_tuner",
                "targets": ["invoke.num_parallel_inferences"],
                "n_candidates": 1,
            }
        ],
    }
    h = HumanLLM(agent_name="lat", llmORchains_list=_llm_list(), dynamic_llm_config=cfg)
    baseline = 3  # registry value

    def latency(n_parallel: int) -> float:
        return float(n_parallel)

    ctx = {"user_message": "lower latency budget", "system_prompt": "BASE"}
    mods = h.dynamic_mgr.evaluate_triggers(ctx, phase="pre_inference")
    npi = (mods.get("invoke_kwargs") or {}).get("num_parallel_inferences")
    assert npi is not None
    assert latency(npi) < latency(baseline)


def test_cost_control_offline_persistent_edits(tmp_path):
    cfg = _dc_base()
    h = HumanLLM(agent_name="cost", llmORchains_list=_llm_list(), dynamic_llm_config=cfg)
    mgr = h.dynamic_mgr

    # Record high-cost outcomes to motivate edits
    mgr.record_outcome({"user_message": "r1"}, {"invoke_kwargs": {"max_tokens": 1024}}, {"cost": 1.0})
    mgr.record_outcome({"user_message": "r2"}, {"invoke_kwargs": {"max_tokens": 768}}, {"cost": 0.9})

    # Both targets are trainable per registry
    mgr.update_optimizer_parameter("inference_tuner", "dynamic_llm_config.default_temperature", trainable=True)
    mgr.update_optimizer_parameter("inference_tuner", "config.flags.low_latency", trainable=True)

    summary = h.offline_optimize(
        "inference_tuner",
        targets=["dynamic_llm_config.default_temperature", "config.flags.low_latency"],
        n_candidates=1,
    )
    assert isinstance(summary.get("applied"), bool)
    if summary["applied"]:
        assert "default_temperature" in mgr.config.get("dynamic_llm_defaults", {})
        assert isinstance(mgr.config.get("flags", {}).get("low_latency") or mgr.config.get("config", {}).get("flags", {}).get("low_latency"), (bool, type(None)))


def test_target_scoping_constraints(tmp_path):
    cfg = _dc_base()
    cfg["help_scope"] = {
        "phase": "pre_inference",
        "rules": {"regex": {"patterns": ["scope"]}},
        "modifications": [
            {"type": "trace", "optimizer": "inference_tuner", "targets": ["invoke.temperature"], "n_candidates": 1}
        ],
    }
    h = HumanLLM(agent_name="scope", llmORchains_list=_llm_list(), dynamic_llm_config=cfg)
    ctx = {"user_message": "check scope", "system_prompt": "BASE"}
    mods = h.dynamic_mgr.evaluate_triggers(ctx, phase="pre_inference")
    ivk = mods.get("invoke_kwargs") or {}
    assert "temperature" in ivk
    assert "max_tokens" not in ivk  # not a target yet

    # Now allow max_tokens as a target, and check it is proposed
    h.dynamic_mgr.config["help_scope"]["modifications"][0]["targets"].append("invoke.max_tokens")
    mods2 = h.dynamic_mgr.evaluate_triggers(ctx, phase="pre_inference")
    ivk2 = mods2.get("invoke_kwargs") or {}
    assert "max_tokens" in ivk2


def test_nested_dynamic_rule_threshold_tuned_offline(tmp_path):
    cfg = _dc_base()
    h = HumanLLM(agent_name="rules", llmORchains_list=_llm_list(), dynamic_llm_config=cfg)
    mgr = h.dynamic_mgr

    # Ensure the parameter is trainable and has bounds in registry
    mgr.update_optimizer_parameter(
        "inference_tuner",
        "dynamic_llm_config.annotations_help_trace_POST.rules.confidence.threshold",
        trainable=True,
    )
    # Provide some feedback records
    mgr.record_outcome({"user_message": "a"}, {"invoke_kwargs": {}}, {"score": 0.3})
    mgr.record_outcome({"user_message": "b"}, {"invoke_kwargs": {}}, {"score": 0.4})

    summary = h.offline_optimize(
        "inference_tuner",
        targets=["dynamic_llm_config.annotations_help_trace_POST.rules.confidence.threshold"],
        n_candidates=1,
    )
    assert isinstance(summary.get("applied"), bool)
    if summary["applied"]:
        dflt = mgr.config.get("dynamic_llm_defaults", {}).get(
            "annotations_help_trace_POST.rules.confidence.threshold"
        )
        assert dflt is not None

