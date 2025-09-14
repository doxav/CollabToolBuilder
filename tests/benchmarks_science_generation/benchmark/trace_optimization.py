"""
WORK IN PROGRESS: unified Trace optimiser/benchmark harness
=============================================================

Adds systematic comparison of
  ①–②  optimisation of *dynamic_llm_config* (Trace‑LLM backend vs. per‑iteration)
  ③–④  the same but with “live” HumanLLM as the optimiser’s LLM backend
  ⑤     optimisation of *trigger* fields that appear inside dynamic_llm_config
        (per‑run *and* per‑iteration variants)

Compatible with Storm, CoStorm and WriteHere runners shipped with Storm‑swarm.
"""

from __future__ import annotations
from typing import Dict, Any, Callable, List, Literal
from enum import Enum
import inspect, json, logging, itertools, argparse, pprint
from pathlib import Path

from opto.trace.nodes import ParameterNode
from opto.optimizers import OptoPrimeMulti
from opto.utils.llm import LLM, LLMFactory


# ───────────────────────────── ENUMS ──────────────────────────────
class Frequency(Enum):      # how often to call .step()
    PER_RUN  = "post_run"
    PER_ITER = "intermediate"

class Backend(Enum):        # which LLM profile drives the optimiser
    TRACE_LLM = "HumanLLM_Trace"
    HUMAN_LLM = "HumanLLM"        # live production model

class ParamScope(Enum):     # which part of the config is optimised
    DYN_CFG  = "dyn_cfg"        # strategies / dynamic_llm_config
    TRIGGERS = "triggers"       # *_trigger keys found in those cfgs


# ───────────────────────── CONSTANTS ──────────────────────────────
AGENTS_STORM   = ["conv_simulator_lm", "question_asker_lm",
                  "outline_gen_lm", "article_gen_lm", "article_polish_lm"]
AGENTS_COSTORM = ["question_answering_lm", "discourse_manage_lm",
                  "utterance_polishing_lm", "warmstart_outline_gen_lm",
                  "question_asking_lm", "knowledge_base_lm"]

# TO ADD LATER when stabilized:
# - AI-Scientist-V2
# - Agent-Laboratory

# tags seen in the default_*_config.json files
STRATEGIES = [
    "baseline",
    "annotations_help_majority",
    "instruction_help_majority",
    "anotation_help",               # sic
    "premium_freq2", "premium_freq4",
    "low_confidence_premium",
]

# ─────────────────── PARAMETER BUILDING HELPERS ─────────────────────
def _choice(values: list[str], name: str, default: str = "baseline") -> ParameterNode:
    idx0 = values.index(default)
    return ParameterNode(idx0, name=name,
                         projections=[lambda x: max(0, min(len(values) - 1, int(round(x))))])

def _build_dyn_cfg_params(agents: list[str]) -> Dict[str, ParameterNode]:
    """Search/retrieval knobs + per‑agent strategy selectors."""
    params = {
        "search_top_k":    ParameterNode(3, projections=[lambda x: int(max(1, min(10, x)))]),
        "retrieve_top_k":  ParameterNode(3, projections=[lambda x: int(max(1, min(10, x)))]),
        "total_conv_turn": ParameterNode(2, projections=[lambda x: int(max(1, min(4, x)))]),
    }
    for ag in agents:
        params[f"{ag}_strategy"] = _choice(STRATEGIES, f"{ag}_strategy")
    return params

def _build_trigger_params(agents: list[str]) -> Dict[str, ParameterNode]:
    """
    Scans *default_*_config.json* files shipped next to this script,
    collects any boolean or integer « *_trigger » fields and exposes
    them as single‑bit / bounded integer ParameterNodes.

    Trigger names are preserved as       "<agent>__<key>"
    (double underscore avoids collision with existing names).
    """
    out: dict[str, ParameterNode] = {}
    for cfg_file in Path(__file__).parent.glob("default_*_config.json"):
        for combo in json.loads(cfg_file.read_text()):
            for entry in combo:
                agent_name = entry.get("agent_name", "UNKNOWN")
                if agent_name not in agents:
                    continue
                dyn_cfg = entry.get("dynamic_llm_config", {})
                for k, v in dyn_cfg.items():
                    if not k.endswith("_trigger"):
                        continue
                    # Booleans ↔ 0/1  •  small ints bounded 0‑10
                    if isinstance(v, bool):
                        out[f"{agent_name}__{k}"] = ParameterNode(int(v),
                            projections=[lambda x: int(max(0, min(1, round(x))))])
                    elif isinstance(v, int):
                        bound = max(1, min(10, abs(v)))
                        out[f"{agent_name}__{k}"] = ParameterNode(v,
                            projections=[lambda x, b=bound: int(max(0, min(b, round(x))))])
    return out

def build_param_space(agents: list[str], scope: ParamScope) -> Dict[str, ParameterNode]:
    if scope is ParamScope.DYN_CFG:
        return _build_dyn_cfg_params(agents)
    if scope is ParamScope.TRIGGERS:
        trigger_nodes = _build_trigger_params(agents)
        if not trigger_nodes:                    # safety‑net
            raise RuntimeError("No *_trigger fields found in default configs — "
                               "cannot optimise ParamScope.TRIGGERS")
        return trigger_nodes
    raise ValueError(scope)


# ────────────────────────── OPTIMISER SETUP ─────────────────────────
def make_opto_prime(params: Dict[str, ParameterNode], backend: Backend) -> OptoPrimeMulti:
    profile_name = f"humanllm_{backend.value.lower()}_backend"
    # idempotent registration
    if profile_name not in LLMFactory.list_profiles():
        LLMFactory.register_profile(profile_name, "LiteLLM",
                                    model=backend.value, temperature=0.0)
    llm = LLM(profile=profile_name)
    return OptoPrimeMulti(
        params.values(),
        llm_profiles=[profile_name],
        generation_technique="multi_llm",
        llm=llm,
    )

# ──────────────────── STRATEGY → dyn_cfg RESOLUTION ─────────────────
def strategy_to_dyn_cfg(tag: str) -> dict:
    if tag == "baseline":
        return {}
    for fp in Path(__file__).parent.glob("default_*_config.json"):
        for combo in json.loads(fp.read_text()):
            for entry in combo:
                cfg = entry.get("dynamic_llm_config", {})
                if any(k.startswith(tag) for k in cfg):
                    return cfg
    return {}


# ───────────────────────── CORE OPTIMISER ───────────────────────────
class HumanTraceOptimizer:
    """
    Universal wrapper that can work in six flavours:
        scope   ∈ {DYN_CFG, TRIGGERS}
        backend ∈ {TRACE_LLM, HUMAN_LLM}
        freq    ∈ {PER_RUN, PER_ITER}
    """

    def __init__(
        self,
        runner_factory: Callable[[Dict[str, Any], Callable | None], tuple[str, list]],
        *,
        agents: list[str] = AGENTS_STORM,
        scope: ParamScope = ParamScope.DYN_CFG,
        backend: Backend = Backend.TRACE_LLM,
        freq: Frequency = Frequency.PER_RUN,
        eval_metric: str = "rougeL",
        patience: int = 3,
    ):
        self.scope    = scope
        self.params   = build_param_space(agents, scope)
        self.opt      = make_opto_prime(self.params, backend)
        self.runner   = runner_factory
        self.mode     = freq.value                    # "intermediate" | "post_run"
        self.metric   = eval_metric
        self.patience = patience
        self.best     = None
        self.bad      = 0

    # ───── feedback hook forwarded to Storm‑swarm runners
    def callback(self, _llm_list, _article, _state, article_change, metrics):
        if not article_change or not metrics:
            return
        score = metrics.get(self.metric, 0)
        fb = f"{self.metric}:{score:.4f}"
        # Use canonical Trace feedback loop: zero_feedback() -> backward(node, feedback) -> step()
        try:
            # reset any previous feedback accumulated on the optimizer
            if getattr(self.opt, "zero_feedback", None):
                self.opt.zero_feedback()
            # send textual feedback to each trainable parameter node
            for p in self.params.values():
                # only attempt backward if optimizer implements it
                if getattr(self.opt, "backward", None):
                    # If a ParameterNode is explicitly non-trainable, skip it.
                    if getattr(p, "trainable", True):
                        self.opt.backward(p, fb)
            # commit/apply the update
            if getattr(self.opt, "step", None):
                self.opt.step()
        except Exception:
            logging.exception("Trace feedback application failed in HumanTraceOptimizer.callback")

        # early-stopping on plateau (unchanged)
        if self.best is None or score > self.best:
            self.best, self.bad = score, 0
        else:
            self.bad += 1
            if self.bad >= self.patience:
                from tests.benchmarks_science_generation.benchmark.benchmark import StopCurrentIteration
                raise StopCurrentIteration("🔻 Patience exhausted")

    # ───── builds kwargs expected by run_storm / run_costorm / etc.
    def _current_kwargs(self) -> Dict[str, Any]:
        d = {k: v.data for k, v in self.params.items()}
        if self.scope is ParamScope.DYN_CFG:
            cfgs: dict[str, dict] = {}
            for ag in [k for k in list(d) if k.endswith("_strategy")]:
                tag = STRATEGIES[d.pop(ag)]
                agent = ag.replace("_strategy", "")
                cfgs[agent] = strategy_to_dyn_cfg(tag)
            human_llm_parameters = [
                {"agent_name": ag,
                 "dynamic_llm_config": cfgs.get(ag, {}),
                 "automation": True}
                for ag in cfgs
            ]
            d["human_llm_parameters"] = human_llm_parameters
        elif self.scope is ParamScope.TRIGGERS:
            # Re‑inject trigger overrides
            triggers = {}
            for name, val in d.items():
                agent, key = name.split("__", 1)
                triggers.setdefault(agent, {})[key] = val
            human_llm_parameters = [
                {"agent_name": ag,
                 "dynamic_llm_config": triggers.get(ag, {}),
                 "automation": True}
                for ag in triggers
            ]
            d = {"human_llm_parameters": human_llm_parameters}
        return d

    # ───── runs one full generation (Storm/CoStorm/WriteHere)
    def run_once(self, **extra) -> tuple[str, list]:
        kw = self._current_kwargs() | extra
        cb = self.callback if self.mode == "intermediate" else None
        sig = inspect.signature(self.runner)
        if "callback" in sig.parameters:
            kw["callback"] = cb
        return self.runner(kw, cb)


# ─────────────────────────── EXPERIMENT GRID ────────────────────────
def grid_compare(
    factory: Callable[[Dict[str, Any], Callable | None], tuple[str, list]],
    *,
    agents: list[str],
    max_steps: int = 40,
) -> list[dict]:
    """
    Runs the six combinations requested by the user:
        1  DYN_CFG / TRACE   / PER_RUN
        2  DYN_CFG / TRACE   / PER_ITER
        3  DYN_CFG / HUMAN   / PER_RUN
        4  DYN_CFG / HUMAN   / PER_ITER
        5a TRIGGERS/ TRACE   / PER_RUN
        5b TRIGGERS/ TRACE   / PER_ITER
    Returns a list of result dicts for easy downstream plotting.
    """
    combos = [
        # scope, backend, freq, label
        (ParamScope.DYN_CFG,  Backend.TRACE_LLM,  Frequency.PER_RUN,  "1‑dynCfg_trace_perRun"),
        (ParamScope.DYN_CFG,  Backend.TRACE_LLM,  Frequency.PER_ITER, "2‑dynCfg_trace_perIter"),
        (ParamScope.DYN_CFG,  Backend.HUMAN_LLM,  Frequency.PER_RUN,  "3‑dynCfg_human_perRun"),
        (ParamScope.DYN_CFG,  Backend.HUMAN_LLM,  Frequency.PER_ITER, "4‑dynCfg_human_perIter"),
        (ParamScope.TRIGGERS, Backend.TRACE_LLM,  Frequency.PER_RUN,  "5a‑triggers_trace_perRun"),
        (ParamScope.TRIGGERS, Backend.TRACE_LLM,  Frequency.PER_ITER, "5b‑triggers_trace_perIter"),
    ]

    results = []
    for scope, backend, freq, label in combos:
        opt = HumanTraceOptimizer(
            factory,
            agents         = agents,
            scope          = scope,
            backend        = backend,
            freq           = freq,
            eval_metric    = "rougeL",
            patience       = 3,
        )
        try:
            opt.run_once(max_steps=max_steps)
        except Exception as ex:
            logging.exception("Run %s failed: %s", label, ex)
        results.append({"label": label, "best_score": opt.best})
    return results


# ────────────────────────────── CLI ─────────────────────────────────
if __name__ == "__main__":
    import benchmark.benchmark as bm

    parser = argparse.ArgumentParser(
        description="Compare optimisation frequencies / back‑ends / scopes on Storm‑swarm.")
    parser.add_argument("--topic", default="Taylor Hawkins")
    parser.add_argument("--system", choices=["storm", "costorm", "writehere"],
                        default="costorm")
    parser.add_argument("--max_steps", type=int, default=40)
    parser.add_argument("--compare_all", action="store_true",
                        help="Run the full 6‑mode comparison grid.")
    args = parser.parse_args()

    if args.system == "storm":
        runner_factory = lambda kw, cb: bm.run_storm_human_llm(args.topic, **kw)
        agents = AGENTS_STORM
    elif args.system == "costorm":
        runner_factory = lambda kw, cb: bm.run_costorm_human_llm(args.topic, **kw)
        agents = AGENTS_COSTORM
    else:  # WriteHere
        runner_factory = lambda kw, cb: bm.run_writehere_human_llm(args.topic, **kw)
        agents = AGENTS_STORM      # WriteHere uses the Storm agent set

    if args.compare_all:
        table = grid_compare(
            runner_factory,
            agents     = agents,
            max_steps  = args.max_steps,
        )
        pprint.pp(table)      # minimal stdout summary
    else:
        # default single run mirrors original example = mode (2)
        optimiser = HumanTraceOptimizer(
            runner_factory,
            agents  = agents,
            scope   = ParamScope.DYN_CFG,
            backend = Backend.TRACE_LLM,
            freq    = Frequency.PER_ITER,
        )
        final_article, _ = optimiser.run_once(max_steps=args.max_steps)
        print("\n── Best ROUGE‑L:", optimiser.best)
        print("── Article head ──────────────────────────────")
        print(final_article[:600], "…")
