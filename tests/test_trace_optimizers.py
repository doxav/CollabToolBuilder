import json
import copy
import pytest
from langchain_openai import ChatOpenAI
from config import *  # use default MODELS_CONFIG_LIST & related constants
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
    # We don't invoke the model in these tests; just provide a valid object
    return {
        "default_llm": ChatOpenAI(model=model, temperature=0.0),
        "premium_llm": ChatOpenAI(model=model, temperature=0.0),
    }


def _write_dynamic_configs(tmp_path):
    # Two cooperating agents using Trace optimizers (online + offline)
    base = {
        "trace_optimizers": [
            {
                "name": "inference_tuner",
                "optimizer_kind": "OptoPrimeV2",
                "parameters": [
                    {"parameter": "invoke.temperature", "value": 0.4, "info": {"bounds": [0.0, 0.9]}, "trainable": True},
                    {"parameter": "invoke.max_tokens", "value": 256, "info": {"bounds": [128, 1024]}, "trainable": False},
                    {"parameter": "dynamic_llm_config.default_temperature", "value": 0.2, "info": {"bounds": [0.0, 1.0]}},
                    {"parameter": "config.flags.low_latency", "value": False},
                ],
                "objective": "maximize quality under latency budget",
                "feedback": {"use": "collect_feedback", "at": "post_inference"},
            },
            {
                "name": "offline",
                "optimizer_kind": "OptoPrimeV2",
                "parameters": [
                    "invoke.max_tokens",
                    {"parameter": "dynamic_llm_config.annotations_help_trace_POST.rules.confidence.threshold", "value": 0.4, "info": {"bounds": [0, 1]}},
                    {"parameter": "dynamic_llm_config.annotations_help_trace_PRE.modifications.0.targets.0.info.bounds", "value": [1, 3]},
                ],
                "objective": "reduce drift and cost at equal actions",
            },
        ],
        "trace_default_optimizer": "inference_tuner",
        "annotations_help_trace_PRE": {
            "phase": "pre_inference",
            "quota": {"max_count": 30},
            "rules": {"regex": {"patterns": ["optimi", "learn"]}},
            "modifications": [
                {
                    "type": "trace",
                    "optimizer": "inference_tuner",
                    "targets": [
                        {"parameters": "invoke.num_parallel_inferences", "info": {"bounds": [1, 3]}},
                        "invoke.temperature",
                    ],
                    "n_candidates": 1,
                }
            ],
        },
        "annotations_help_trace_POST": {
            "phase": "post_inference",
            "quota": {"max_count": 3},
            "rules": {"confidence": {"method": "self-consistency", "consistency_method": "difflib", "threshold": 0.4}},
            "modifications": [
                {
                    "type": "trace",
                    "optimizer": "inference_tuner",
                    "targets": ["invoke.temperature", "invoke.max_tokens"],
                    "feedback": {"use": "collect_feedback", "at": "post_inference"},
                    "n_candidates": 1,
                }
            ],
        },
    }
    a1 = copy.deepcopy(base)
    a2 = copy.deepcopy(base)
    # persist with stable names resembling the requested pattern
    p1 = tmp_path / "costorm_dynamic_config_trace_agent1.json"
    p2 = tmp_path / "costorm_dynamic_config_trace_agent2.json"
    p1.write_text(json.dumps(a1), encoding="utf-8")
    p2.write_text(json.dumps(a2), encoding="utf-8")
    return p1, p2


def test_trace_optimizers_end_to_end_multiagent(tmp_path):
    p1, p2 = _write_dynamic_configs(tmp_path)
    cfg1 = json.loads(p1.read_text())
    cfg2 = json.loads(p2.read_text())

    h1 = HumanLLM(agent_name="agent_one", llmORchains_list=_llm_list(), dynamic_llm_config=cfg1)
    h2 = HumanLLM(agent_name="agent_two", llmORchains_list=_llm_list(), dynamic_llm_config=cfg2)

    # Online (H1): pre-inference rule triggers only when user mentions optimization
    ctx_hit = {"user_message": "please optimize and learn", "system_prompt": "base"}
    mods_hit = h1.dynamic_mgr.evaluate_triggers(ctx_hit, phase="pre_inference")
    # Trace adapter should produce invoke overrides for declared targets
    assert (mods_hit.get("invoke_kwargs") or {}).get("temperature") is not None
    # Targets control trainables; max_tokens is not a target here
    assert "max_tokens" not in (mods_hit.get("invoke_kwargs") or {})

    # Verify trainables defined by targets
    adapter = h1.dynamic_mgr._trace_optimizers["inference_tuner"]
    # Only specified targets should be trainable
    assert adapter._parameters["invoke.temperature"].get("trainable") is True
    assert adapter._parameters.get("invoke.max_tokens", {}).get("trainable") in (False, None)

    # No trigger in other agent
    ctx_miss = {"user_message": "hello world", "system_prompt": "base"}
    mods_miss = h2.dynamic_mgr.evaluate_triggers(ctx_miss, phase="pre_inference")
    assert mods_miss == {}

    # Simulate outcomes/feedback recorded online by agent one
    h1.dynamic_mgr.record_outcome(ctx_hit, {"invoke_kwargs": {"temperature": 0.6}}, {"score": 0.2})
    h1.dynamic_mgr.log_user_correction("agent_one", diff="increase temperature a bit", who="tester")

    # Prepare some parameters to be trainable for offline pass
    h1.dynamic_mgr.update_optimizer_parameter("inference_tuner", "dynamic_llm_config.default_temperature", trainable=True)
    h1.dynamic_mgr.update_optimizer_parameter("inference_tuner", "config.flags.low_latency", trainable=True)

    # Offline (H2): apply persistent edits to dynamic config defaults
    summary = h1.offline_optimize(
        "inference_tuner",
        targets=["dynamic_llm_config.default_temperature", "config.flags.low_latency"],
        n_candidates=1,
    )
    # Optimizer may or may not be available in this runtime; validate both cases safely.
    assert isinstance(summary.get("applied"), bool)
    if summary["applied"]:
        assert "default_temperature" in h1.dynamic_mgr.config.get("dynamic_llm_defaults", {})

    # Resolve and set a target via manager helpers
    get_t, set_t = h1.dynamic_mgr.resolve_optimizable_target("invoke.temperature")
    ctx = {"invoke_kwargs": {"temperature": 0.2}}
    assert get_t(ctx) == 0.2
    set_t(ctx, 0.33)
    assert ctx["invoke_kwargs"]["temperature"] == 0.33

    # Listing optimizers should include our names
    names = h1.dynamic_mgr.list_trace_optimizers()
    assert "inference_tuner" in names
    assert "offline" in names


def test_manager_objective_and_params_api(tmp_path):
    # Setup agents from the same dynamic configs
    p1, _ = _write_dynamic_configs(tmp_path)
    cfg1 = json.loads(p1.read_text())
    h1 = HumanLLM(agent_name="agent_cfg", llmORchains_list=_llm_list(), dynamic_llm_config=cfg1)

    mgr = h1.dynamic_mgr
    # Objective update
    mgr.set_optimizer_objective("inference_tuner", "maximize reward under cost")
    spec = mgr._trace_optimizers["inference_tuner"].get_trace_spec()
    assert spec.get("objective") == "maximize reward under cost"

    # Parameter list replacement and single-parameter update
    new_params = [
        {"parameter": "invoke.temperature", "value": 0.3, "info": {"bounds": [0.1, 0.9]}, "trainable": True},
        {"parameter": "invoke.max_tokens", "value": 512, "info": {"bounds": [128, 2048]}, "trainable": False},
    ]
    mgr.set_optimizer_parameters("inference_tuner", new_params)
    mgr.update_optimizer_parameter("inference_tuner", "invoke.max_tokens", trainable=True)
    adapter = mgr._trace_optimizers["inference_tuner"]
    assert adapter._parameters["invoke.temperature"]["trainable"] is True
    assert adapter._parameters["invoke.max_tokens"]["trainable"] is True


def test_resolve_target_and_feedback_collection(tmp_path):
    p1, _ = _write_dynamic_configs(tmp_path)
    cfg1 = json.loads(p1.read_text())
    h1 = HumanLLM(agent_name="agent_resolve", llmORchains_list=_llm_list(), dynamic_llm_config=cfg1)

    # Resolve a few standard targets
    get_t, set_t = h1.dynamic_mgr.resolve_optimizable_target("invoke.temperature")
    ctx = {"invoke_kwargs": {"temperature": 0.2}}
    assert get_t(ctx) == 0.2
    set_t(ctx, 0.5)
    assert ctx["invoke_kwargs"]["temperature"] == 0.5

    get_u, set_u = h1.dynamic_mgr.resolve_optimizable_target("user_message")
    set_u(ctx, "hello")
    assert get_u(ctx) == "hello"

    # Feedback collection ordering: user diffs first, then autos
    h1.dynamic_mgr.log_user_correction("agent_resolve", diff="increase temp", who="tester")
    h1.dynamic_mgr.record_outcome({"user_message": "m"}, {"invoke_kwargs": {}}, {"score": 0.1})
    items = h1.dynamic_mgr.collect_feedback(ctx)
    assert isinstance(items, list) and len(items) >= 1
    if items:
        assert items[0].get("type") in ("user_diff", "fallback_user_message")
