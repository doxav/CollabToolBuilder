import os
import tempfile
from asyncio import subprocess
from threading import local
import re, uuid, json, difflib
import time, inspect, ast
import socket
import logging
import concurrent.futures
from jinja2 import Template
from datetime import datetime
from collections import defaultdict
import copy
import importlib
from matplotlib import rc
from openai import BadRequestError
from utils.llm_utils import (
    InferenceCheck,
    InferenceTracking,
    TaskHistory,
    smart_input, smart_print, get_primitives,
    _visual_input, save_prompt_with_tag,
    list_prompt_variants, flatten_and_pair,
    semantic_double_pass_chunking,
    extract_json,
    calculate_text_similarity,
    secure_invoke
)
try:
    from env.SWEBench.env import SWEBenchEnvironment
except Exception as e:
    SWEBenchEnvironment = None
from utils.human_llm_config import HumanLLMConfig
from typing import List, Dict, Any, Optional, Union, Tuple, Callable


from langchain.llms import OpenAI
from langchain.chains import LLMChain
from PyPDF2.generic import IndirectObject
from langchain.prompts import PromptTemplate
from langchain_core.messages.ai import AIMessage
from langchain_core.messages.human import HumanMessage
from langchain_core.messages.system import SystemMessage
from langchain_core.messages.function import FunctionMessage
from langchain_core.runnables import RunnableSequence, ConfigurableField
import regex as regex 

from dataclasses import dataclass, field

@dataclass
class FewShotsParams:
    """Parameters for few-shot learning examples."""

    num: int = 5
    filter: dict = field(default_factory=dict)
    ranking_method: str = 'by_date_desc'
    annotations: Optional[Union[str, List[str]]] = None
    generate_summary: bool = False
    format: Optional[str] = None
    summary_char_limit: int = 500

class HelpUsageTracker:
    """Tracks usage, costs, and quotas for different help types"""
    def __init__(self):
        self.usage = defaultdict(lambda: {
            'count': 0,
            'cost': 0.0,
            'tokens': 0,
            'successes': 0,
            'failures': 0
        })
        self.quotas = {}
        
    def set_quota(self, help_type: str, quota_config: Dict):
        """Set quota limits for a help type"""
        self.quotas[help_type] = quota_config
        
    def can_use(self, help_type: str) -> bool:
        """Check if help type is within quota"""
        if help_type not in self.quotas:
            return True
        
        quota = self.quotas[help_type]
        usage = self.usage[help_type]
        
        if 'max_count' in quota and usage['count'] >= quota['max_count']:
            return False
        if 'max_cost' in quota and usage['cost'] >= quota['max_cost']:
            return False
        if 'max_tokens' in quota and usage['tokens'] >= quota['max_tokens']:
            return False
            
        return True
        
    def record_usage(self, help_type: str, cost: float = 0, tokens: int = 0, success: bool = True):
        """Record usage of a help type"""
        self.usage[help_type]['count'] += 1
        self.usage[help_type]['cost'] += cost
        self.usage[help_type]['tokens'] += tokens
        if success:
            self.usage[help_type]['successes'] += 1
        else:
            self.usage[help_type]['failures'] += 1

class _TraceOptimizerAdapter:
    """
    Small adapter that exposes a step(context, targets, n_candidates) -> List[dict(edits...)]
    Replace the internals or pass a real trace object via config['trace_obj'] to integrate your Trace implementation.
    If no trace_obj is provided, we can lazily build one from:
      - config['optimizer_kind'] in {'OPRO','OptoPrime','OptoPrimeV2','OptoPrimeMulti'} or
      - config['import_path'] (e.g., 'opto.optimizers.optoprime.OptoPrimeV2')
    plus optional config['optimizer_kwargs'].
    The adapter maintains a small Trace-style parameter registry (name -> {value, trainable, description, ...}).
    """
    def __init__(self, name: str, config: Dict):
        self.name = name
        self.config = config or {}
        self._state = {"created_at": time.time(), "calls": 0}
        # allow a real trace object to be passed in config under key 'trace_obj'
        self._trace_obj = self.config.get("trace_obj")
        # unified Trace-style registry (single "parameters" list in config)
        self._objective: Optional[str] = self.config.get("objective")
        self._init_parameters_from_config(self.config.get("parameters"))
        # factory hints (optional)
        self._optimizer_kind: Optional[str] = self.config.get("optimizer_kind", "optoprimev2").lower().strip()

        self._import_path: Optional[str] = self.config.get("import_path")
        self._optimizer_kwargs: Dict[str, Any] = dict(self.config.get("optimizer_kwargs") or {})
        
    def _init_parameters_from_config(self, items: Any):
        """
        Initialize the parameter registry from the new Trace-style "parameters" list.
        Each entry can be a string (name) or a dict with keys:
          - parameter (str, required)
          - value, trainable, description, projections, info (optional)
        """
        self._parameters = {}
        if not items:
            return
        if isinstance(items, list):
            for entry in items:
                if isinstance(entry, str):
                    self._parameters[entry] = {"parameter": entry}
                elif isinstance(entry, dict):
                    name = entry.get("name") or entry.get("parameter")
                    if not name:
                        continue
                    meta = {k: entry[k] for k in ("value", "trainable", "description", "projections", "info","parameter") if k in entry}
                    self._parameters[name] = meta
        elif isinstance(items, dict):
            # (No backward support required; keep minimal tolerance for accidental dicts)
            self._parameters = dict(items)

    # ---- NEW: lightweight API to manage objective/parameters ----
    def set_objective(self, objective: Optional[str]):
        self._objective = objective


    def get_trace_spec(self) -> Dict[str, Any]:
        # Return a single-step snapshot for the optimizer (immutable for this call).
        return {"objective": self._objective, "parameters": copy.deepcopy(self._parameters)}
    # ---- Parameter coercion helpers (accept JSON-friendly formats) ----
    def _coerce_param_entry(self, name: str, entry: Any) -> Dict[str, Any]:
        """
        Normalize a parameter entry into a dict meta:
          - scalar -> {'value': scalar}
          - [lo, hi] (numeric) -> {'bounds': (lo, hi)}
          - {'range':[lo,hi]} or {'min':lo,'max':hi} -> {'bounds'lo,hi)} + pass-through other keys
          - already dict -> pass-through (but normalize 'range'/'min'/'max' to 'bounds' if present)
        We do NOT auto-rename keys like 'temperature_range' -> 'temperature'; the name is kept as-is.
        """
        meta: Dict[str, Any] = {}
        # Plain dict -> copy and normalize keys
        if isinstance(entry, dict):
            meta.update(entry)
            # normalize range/min/max into bounds
            if "range" in meta and isinstance(meta["range"], (list, tuple)) and len(meta["range"]) == 2:
                lo, hi = meta["range"]
                if isinstance(lo, (int, float)) and isinstance(hi, (int, float)):
                    meta.setdefault("bounds", (lo, hi))
            if "min" in meta and "max" in meta and all(isinstance(meta[k], (int, float)) for k in ("min", "max")):
                meta.setdefault("bounds", (meta["min"], meta["max"]))
            return meta
        # 2-length numeric list/tuple -> bounds
        if isinstance(entry, (list, tuple)) and len(entry) == 2 and all(isinstance(x, (int, float)) for x in entry):
            return {"bounds": (entry[0], entry[1])}
        # scalar -> value
        if isinstance(entry, (int, float, str, bool)):
            return {"value": entry}
        # unknown -> keep as-is in 'value' to not lose information
        return {"value": entry}

    def set_parameters(self, params: Dict[str, Any]):
        """Register/update multiple parameters from JSON-friendly formats."""
        # Accept dict (internal) or list (Trace-format). Manager calls this.
        if isinstance(params, list):
            self._init_parameters_from_config(params)
        else:
            self._parameters = dict(params or {})

    def redefine_trainables(self, targets: List[Union[str, Dict[str, Any]]]):
        """
        Redefine the *set* of trainable parameters using a targets list from a
        'type':'trace' modification. Items may be:
          - "param.path" (string)
          - {"parameters": "param.path", "info": {...}}  (merge/override info)
          - {"parameter": "param.path", "info": {...}}   (accepted too)
        If a target parameter does not exist yet, create it with trainable=True.
        All parameters *not* in targets become trainable=False.
        """
        names: List[str] = []
        for t in (targets or []):
            if isinstance(t, str):
                names.append(t)
                self._parameters.setdefault(t, {})
                self._parameters[t]["trainable"] = True
            elif isinstance(t, dict):
                p = t.get("parameters") or t.get("parameter") or t.get("name")
                if not p:
                    continue
                names.append(p)
                meta = self._parameters.setdefault(p, {})
                meta["trainable"] = True
                if isinstance(t.get("info"), dict):
                    meta.setdefault("info", {}).update(t["info"])
        # All others become non-trainable
        for k in list(self._parameters.keys()):
            if k not in names:
                self._parameters[k].pop("trainable", None)
                self._parameters[k]["trainable"] = False

    def update_parameter(self, name: str, **kwargs):
        """
        Update a single parameter; accepts JSON-friendly shapes:
          update_parameter('alpha', value=0.3, trainable=True) OR
          update_parameter('temperature_range', range=[0.2,0.8])
        """
        slot = self._parameters.setdefault(name, {})
        # If user provided a dict-like 'value' or 'range' directly in kwargs, coerce the whole shape
        if set(kwargs.keys()) & {"value", "range", "min", "max", "bounds", "trainable", "description"}:
            # Merge then normalize (so range/min/max → bounds)
            slot.update(kwargs)
            slot.update(self._coerce_param_entry(name, slot))
        else:
            slot.update({k: v for k, v in kwargs.items() if v is not None})

   # ---- Lazy creation of a real Trace optimizer if requested ----
    def _ensure_trace_obj(self, current_kwargs: Dict[str, Any]):
        """
            If no trace_obj was provided at construction, try to build one lazily.
            Supported:
            - optimizer_kind in {'OPRO','OptoPrime','OptoPrimeV2','OptoPrimeMulti'}
            - import_path like 'opto.optimizers.optoprime.OptoPrimeV2'
            We will construct ParameterNode objects from the adapter's parameter registry
            when the 'opto' package is available. If creation fails, we leave _trace_obj as None.
            """
        if self._trace_obj is not None:
            return
        cls = None
        try:
            if self._optimizer_kind:
                from opto import optimizers
                # Prefer official import paths (exported from opto.optimizers.__init__)
                #mod = importlib.import_module("opto.optimizers")
                kind = self._optimizer_kind.lower()
                if kind == "opro": cls = optimizers.OPRO # getattr(mod, "OPRO")
                elif kind == "optoprime": cls = optimizers.OptoPrime # getattr(mod, "OptoPrime")
                elif kind == "optoprimev2": cls = optimizers.OptoPrimeV2 # getattr(mod, "OptoPrimeV2")
                elif kind == "optoprimemulti": cls = optimizers.OptoPrimeMulti # getattr(mod, "OptoPrimeMulti")
            elif self._import_path:
                # import by path, e.g. 'opto.optimizers.optoprime.OptoPrimeV2'
                mod_name, _, class_name = self._import_path.rpartition(".")
                mod = importlib.import_module(mod_name)
                cls = getattr(mod, class_name)
        except Exception as e:
            # could not import – leave fallback heuristic
            return
        if cls is None:
            return

        # Build ParameterNodes from registry if possible
        try:
            from opto import trace
            from opto.utils.llm import LLM
            ParameterNode =  trace.nodes.ParameterNode
            params: List[Any] = []
            for pname, meta in (self._parameters or {}).items():
                # Handle different parameter specification formats
                if isinstance(meta, ParameterNode):
                    # Direct ParameterNode from Trace usage
                    params.append(meta)
                elif isinstance(meta, dict):
                    pname = pname.split(".")[-1]  # use last part as py_name
                    m = dict(meta)
                    val = m.get("value", current_kwargs.get(pname, None))
                    trainable = bool(m.get("trainable", False))
                    description = m.get("description")
                    
                    # Handle info/bounds for projections
                    info = m.get("info")
                    projections = m.get("projections", None)
                    params.append(
                        ParameterNode(
                            val, name=pname, trainable=trainable, description=description, projections=projections, info=info
                        )
                     )
                else:
                    # Fallback: treat as value directly
                    params.append(ParameterNode(meta, name=pname, trainable=False))
            # Instantiate optimizer. We do NOT pass an LLM unless user supplied one; let Trace use its defaults.
            kwargs = dict(self._optimizer_kwargs or {})
            self._trace_obj = cls(parameters=params, objective=self._objective, **kwargs)
        except Exception as e:
            # If anything fails, keep fallback behavior
            self._trace_obj = None

    def _trace_to_edits(self, update_dict) -> List[Dict]:
        """
        Convert a Trace update_dict {ParameterNode: Any} to our 'edits' list
        using 'trace.param.<py_name>.value' targets.
        """
        edits = []
        try:
            # ParameterNode has 'py_name' (see opto.trace.nodes.ParameterNode)
            for pnode, new_val in (update_dict or {}).items():
                try:
                    name = getattr(pnode, "py_name", None) or str(pnode)
                except Exception:
                    name = str(pnode)
                edits.append({"target": f"trace.param.{name}.value", "op": "set", "value": new_val})
        except Exception:
            pass
        return edits

    # -- Trace convenience wrappers ------------------------------------
    def zero_feedback(self,current_kwargs: Dict[str, Any] = None):
        """Mirror Trace API: reset accumulated feedback on parameters."""
        self._ensure_trace_obj(current_kwargs=current_kwargs)
        if getattr(self._trace_obj, "zero_feedback", None):
            self._trace_obj.zero_feedback()

    def backward(self, node_or_param, feedback: str, current_kwargs: Dict[str, Any] = None, **kwargs):
        """Mirror Trace API: propagate feedback from a node/parameter."""
        self._ensure_trace_obj(current_kwargs=current_kwargs)
        if getattr(self._trace_obj, "backward", None):
            return self._trace_obj.backward(node_or_param, feedback, **kwargs)

    def parameters(self,current_kwargs: Dict[str, Any] = None) -> List[Any]:
        """Expose underlying ParameterNode list (if available)."""
        self._ensure_trace_obj(current_kwargs=current_kwargs)
        return list(getattr(self._trace_obj, "parameters", []) or [])

    def step(self, context: Dict, targets: List[str], n_candidates: int = 1, constraints: Dict = None) -> List[Dict]:
        """Return candidate patches. Each candidate is a dict {'edits': [ {'target':..., 'op':..., 'value':...}, ... ]}"""
        self._state["calls"] += 1
        # record an observation snapshot for debug/tracing - use per-optimizer observables

        candidates = []

        self._ensure_trace_obj(current_kwargs=context.get("kwargs", {}))
        self._trace_obj
        if self._trace_obj:
            # If you plug a real Trace object, it should expose an API we call here.
            # We try to call a common name 'step' and expect a list-like reply; adapt as necessary.
            # Branch 1: object exposes our adapter protocol (step returns edits)
            if hasattr(self._trace_obj, "step") and "context" in getattr(self._trace_obj.step, "__code__", type("X",(object,),{})()).co_varnames:
                raw = getattr(self._trace_obj, "step")(context=context, targets=targets, n_candidates=n_candidates, constraints=constraints)
            else:
                # Branch 2: it's a Trace Optimizer (see opto.optimizers.optimizer.Optimizer)
                # Standard Trace loop: we assume backward() has been done upstream when needed.
                # Apply the update (bypassing=False) and convert to edits.
                try:
                    update_dict = getattr(self._trace_obj, "step")(bypassing=False)
                    raw = [{"edits": self._trace_to_edits(update_dict)}]
                except Exception as e:
                    raw = []
            if not isinstance(raw, list):
                raw = [raw]
            for r in raw:
                # normalise into edits list; if r already has 'edits', keep it
                if isinstance(r, dict) and "edits" in r:
                    candidates.append(r)
                else:
                    # wrap an opaque candidate
                    candidates.append({"edits": [{"target": t, "op": "set", "value": f"<trace:{self.name}:candidate>"} for t in (targets or [])]})
            with open("optimizer_trace_log.jsonl", "a", encoding="utf-8") as f:
                for c in candidates:
                    f.write(json.dumps({"optimizer": self.name, "edits": c.get("edits", []), "meta": c.get("meta", {}), "log": {"timestamp": datetime.now().isoformat()}}) + "\n")
            return candidates

        raise NotImplementedError("No valid trace optimizer found")

class DynamicConfigManager:
    """Manages dynamic LLM configuration based on rules and triggers"""
    def __init__(self, config: Dict, usage_tracker: HelpUsageTracker, default_config: Dict):
        self.config = config
        self.usage_tracker = usage_tracker
        self.default_config = default_config
        self.logger = logging.getLogger(__name__)
        self.rule_evaluators = {
            'regex': self._eval_regex_rule,
            'confidence': self._eval_confidence_rule,
            'similarity': self._eval_similarity_rule,
            'frequency': self._eval_frequency_rule,
            'complexity': self._eval_complexity_rule,
            'history': self._eval_history_rule,
            'composite': self._eval_composite_rule
        }
        self._call_history = []
        # -----------------------
        # # Extended state for Trace and feedback
        # -----------------------
        self._trace_optimizers: Dict[str, _TraceOptimizerAdapter] = {}
        # in-memory records used by collect_feedback() and offline H2
        self._records: List[Dict] = []
        # Initialize trace optimizers if present in incoming config under 'trace_optimizers'
        for tconf in (self.config or {}).get("trace_optimizers", []) or []:
            try:
                name = tconf.get("name") or f"trace_{len(self._trace_optimizers)+1}"
                self.register_trace_optimizer(name, tconf)
            except Exception:
                self.logger.exception("Failed to register trace optimizer from config")
    def evaluate_triggers(self, context: Dict, phase: str = 'pre_inference') -> Dict:
        """Evaluate all triggers and return modifications to apply"""
        modifications = {}
        
        for help_type, help_config in list(self.config.items()):
            # Skip meta sections that are not help blocks
            if help_type in ('trace_optimizers', 'trace_default_optimizer'):
                continue
            if not isinstance(help_config, dict):
                self.logger.warning("Invalid dynamic config for type (expected dict): %r", help_type)
                continue
            # Check if this help type applies to current phase
            if help_config.get('phase', 'pre_inference') != phase:
                continue
                
            # Check quota
            if not self.usage_tracker.can_use(help_type):
                continue
                
            # Evaluate rules
            rules = help_config.get('rules', {})
            if self._evaluate_rules(rules, context):
                # Apply modifications
                mods = help_config.get('modifications', {})
                # Allow old dict-based modifications for backward compatibility
                if isinstance(mods, dict):
                    modifications.update(mods)
                else:
                    # new-style: list of modification descriptors; supports {'type': 'trace', ...}
                    for m in (mods or []):
                        mtype = m.get("type", "simple")
                        if mtype == "simple":
                            # merge simple dict patch (fallback)
                            payload = m.get("payload", {})
                            if isinstance(payload, dict):
                                modifications.update(payload)
                        elif mtype == "trace":
                            # call a persistent trace optimizer synchronously (H1 behavior)
                            opt_name = m.get("optimizer") or self.config.get("trace_default_optimizer")
                            if not opt_name:
                                self.logger.debug("Trace modification requested but no optimizer name/config found; skipping")
                                continue
                            adapter = self._trace_optimizers.get(opt_name)
                            if adapter is None:
                                # create a basic adapter for this name, allow user to replace with real Trace later
                                adapter = _TraceOptimizerAdapter(opt_name, m.get("config", {}))
                                self._trace_optimizers[opt_name] = adapter
                            # Optional: redefine the trainable parameter set from 'targets'
                            targets_spec = m.get("targets") or []
                            if targets_spec:
                                try:
                                    adapter.redefine_trainables(targets_spec)
                                except Exception:
                                    self.logger.exception("Failed to redefine trainables from 'targets'")
                            # Extract plain target names to forward to the optimizer
                            targets: List[str] = []
                            for t in (targets_spec or []):
                                if isinstance(t, str):
                                    targets.append(t)
                                elif isinstance(t, dict):
                                    nm = t.get("parameters") or t.get("parameter") or t.get("name")
                                    if nm:
                                        targets.append(nm)

                            
                            n_candidates = int(m.get("n_candidates", 1))
                            try:
                                # pass a 'trace_spec' (objective + parameters) to the optimizer
                                ctx_for_trace = dict(context)
                                ctx_for_trace["trace_spec"] = adapter.get_trace_spec()
                                # --- NEW: standard Trace loop (online) when we're at the right phase
                                fb_cfg = m.get("feedback") or {"use": "collect_feedback", "at": "post_inference"}
                                # if phase == fb_cfg.get("at", "post_inference"):
                                if True:
                                    try:
                                        adapter.zero_feedback(current_kwargs=context.get("kwargs", {}))
                                        fb_text = self._render_feedback_text(context, fb_cfg)
                                        for p in adapter.parameters():
                                            if getattr(p, "trainable", False):
                                                adapter.backward(p, fb_text, current_kwargs=context.get("kwargs", {}))
                                    except Exception:
                                        self.logger.exception("Trace backward failed (H1)")
                                candidates = adapter.step(context=ctx_for_trace, targets=targets, n_candidates=n_candidates, constraints=m.get("constraints"))

                            except Exception as e:
                                self.logger.exception("Trace optimizer step failed; skipping optimizer modification")
                                candidates = []
                            # If Trace returns candidates, pick the first by default (Trace-as-Advisor). UI/human-in-loop can override later.
                            if candidates:
                                first = candidates[0]
                                # We expect candidate to be dict with 'edits': list of {target, op, value}
                                for edit in first.get("edits", []):
                                    tgt = edit.get("target")
                                    op = edit.get("op", "set")
                                    val = edit.get("value")
                                    # For known targets like invoke.kwarg keys or system/user message, translate to modifications
                                    if tgt and tgt.startswith("invoke."):
                                        # modifications dict may contain an 'invoke_kwargs' sub-dict
                                        invoke_overrides = modifications.setdefault("invoke_kwargs", {})
                                        key = tgt.split(".", 1)[1]
                                        if op in ("set",):
                                            invoke_overrides[key] = val
                                    elif tgt in ("user_message", "system_prompt"):
                                        # place in modifications top-level
                                        if op == "set":
                                            modifications[tgt] = val
                                        elif op == "append":
                                            modifications.setdefault(tgt + "_append", []).append(val)
                                        elif op == "prepend":
                                            modifications.setdefault(tgt + "_prepend", []).append(val)
                                    elif tgt and tgt.startswith("dynamic_llm_config."):
                                        # allow edits into dynamic_llm_config subkeys
                                        k = tgt.split(".", 1)[1]
                                        dyn = modifications.setdefault("dynamic_llm_config_patch", {})
                                        if op == "set":
                                            dyn[k] = val
                                    elif tgt == "activate_human_intervention":
                                        modifications["activate_human_intervention"] = bool(val)
                                    # ---- Allow optimizers to edit their own trace spec ----
                                    elif tgt and tgt == "trace.objective":
                                        try:
                                            adapter.set_objective(val)
                                        except Exception:
                                            self.logger.exception("Failed setting trace.objective")
                                    elif tgt and tgt.startswith("trace.param."):
                                        try:
                                            # trace.param.<name> or trace.param.<name>.<field>
                                            rest = tgt.split("trace.param.", 1)[1]
                                            parts = rest.split(".")
                                            pname = parts[0]
                                            if len(parts) == 1:
                                                adapter.update_parameter(pname, value=val)
                                            elif len(parts) == 2:
                                                field = parts[1]
                                                adapter.update_parameter(pname, **{field: val})
                                            else:
                                                # flatten deeper paths into a dict
                                                adapter.update_parameter(pname, **{".".join(parts[1:]): val})
                                        except Exception:
                                            self.logger.exception("Failed applying trace.param edit: %s", tgt)
                                    elif tgt and tgt.startswith("rule_evaluator."):
                                        # allow Trace to replace a rule evaluator on the manager
                                        name = tgt.split(".", 1)[1]
                                        if callable(val):
                                            self.rule_evaluators[name] = val
                                        else:
                                            self.logger.warning("Trace edit for %s ignored: non-callable value", tgt)
                                    elif tgt and tgt.startswith("config."):
                                        # edits into self.config paths, e.g. 'config.helpA.rules.regex.patterns'
                                        path = tgt.split(".", 1)[1]
                                        self._set_in_dict_by_path(self.config, path, val)
                                    else:
                                        # generic fallback: put into 'patches' list so callers can interpret them
                                        modifications.setdefault("patches", []).append(edit)
                                # honor meta.requires_human if present
                                if isinstance(first, dict) and isinstance(first.get("meta"), dict):
                                    if first["meta"].get("requires_human"):
                                        modifications["activate_human_intervention"] = True

                        else:
                            # unknown modification type: ignore but log
                            self.logger.debug(f"Unknown modification item type: {mtype} (skipped)")
                
                # Record usage (basic tracking, cost/tokens updated later)
                self.usage_tracker.record_usage(help_type)
                
        # Record context for history-based rules
        self._call_history.append({
            'context': context,
            'phase': phase,
            'modifications': modifications,
            'timestamp': datetime.now()
        })
        
        # Keep history bounded
        if len(self._call_history) > 100:
            self._call_history.pop(0)
            
        return modifications

    # ---- helper to prepare feedback text ---------------------------------
    def _render_feedback_text(self, context: Dict, fb_cfg: Dict) -> str:
        """
        Build a textual feedback signal for Trace from available sources.
        Supported fb_cfg:
           {"use": "collect_feedback", "at": "post_inference"} (default)
           {"use": "context_key", "key": "trace.feedback_path", "at": "..."}
           {"use": "literal", "text": "..."}
        """
        mode = (fb_cfg or {}).get("use", "collect_feedback")
        if mode == "literal":
            return str(fb_cfg.get("text", "")).strip() or "No feedback"
        if mode == "context_key":
            key = fb_cfg.get("key", "")
            # dotted path lookup
            cur = context
            for part in key.split("."):
                if not part:
                    continue
                cur = (cur or {}).get(part) if isinstance(cur, dict) else getattr(cur, part, None)
            return str(cur or "No feedback")
        # default: collect & merge manager’s feedback records
        items = self.collect_feedback(context)
        lines = []
        for it in items:
            if "content" in it: lines.append(str(it["content"]))
            elif "diff" in it:  lines.append(str(it["diff"]))
            elif "metrics" in it: lines.append(str(it["metrics"]))
        return "\n\n".join(l for l in lines if l).strip() or "No feedback"

    # -------------------------
    # small dict path helpers
    # -------------------------
    def _set_in_dict_by_path(self, root: Dict, path: str, value: Any):
        """Set nested dict value by dotted path. Creates intermediate dicts as needed."""
        parts = path.split(".")
        cur = root
        for p in parts[:-1]:
            if p not in cur or not isinstance(cur[p], dict):
                cur[p] = {}
            cur = cur[p]
        cur[parts[-1]] = value

    def _get_from_dict_by_path(self, root: Dict, path: str) -> Any:
        parts = path.split(".")
        cur = root
        for p in parts:
            if not isinstance(cur, dict) or p not in cur:
                return None
            cur = cur[p]
        return cur

    def _evaluate_rules(self, rules: Dict, context: Dict) -> bool:
        """Evaluate a set of rules against context"""
        if not rules:
            return True

        for rule_type, rule_config in rules.items():
            if rule_type in self.rule_evaluators:
                if not self.rule_evaluators[rule_type](rule_config, context):
                    return False
            else:
                self.logger.warning(f"Unknown rule type: {rule_type}")
                return False  # Fail evaluation for unknown rule types
                
        return True
        
    def _eval_regex_rule(self, config: Dict, context: Dict) -> bool:
        """Evaluate regex pattern matching"""
        patterns = config.get('patterns', [])
        target = context.get(config.get('target', 'user_message'), '')
        
        for pattern in patterns:
            if re.search(pattern, str(target)):
                return True
        return False
            
    def _eval_similarity_rule(self, config: Dict, context: Dict) -> bool:
        """Evaluate similarity to past prompts"""
        threshold = config.get('threshold', 0.8)
        current = context.get('user_message', '')
        
        similar_count = sum(
            1 for h in self._call_history[-10:]
            if calculate_text_similarity(
                h['context'].get('user_message', ''), 
                current
            ) > threshold
        )
        
        return similar_count >= config.get('min_similar', 3)
        
    def _eval_frequency_rule(self, config: Dict, context: Dict) -> bool:
        """Evaluate frequency-based triggers"""
        every_n = config.get('every_n', 10)
        # TODO: bad fix, should have a better fix
        count = (len(self._call_history)//2) + 1
        return count % every_n == 0
        
    def _eval_confidence_rule(self, config: Dict, context: Dict) -> bool:
        """Evaluate output divergence"""
        outputs = context.get('llm_outputs', [])
        method = config.get('method', 'self-consistency')
        if method == 'self-consistency':
            if len(outputs) < 2:
                return False
            consistency_method = config.get('consistency_method', 'difflib')
            threshold = config.get('threshold', 0.4)
            similarities = []
            
            for i in range(len(outputs)):
                for j in range(i+1, len(outputs)):
                    sim = calculate_text_similarity(
                        outputs[i].content if hasattr(outputs[i], 'content') else str(outputs[i]),
                        outputs[j].content if hasattr(outputs[j], 'content') else str(outputs[j]),
                        method=consistency_method
                    )
                    similarities.append(sim)
                    
            avg_similarity = sum(similarities) / len(similarities) if similarities else 1
            return avg_similarity < threshold
        else:
            self.logger.warning(f"Unknown confidence method: {method}")
            return False
        
    def _eval_complexity_rule(self, config: Dict, context: Dict) -> bool:
        """Evaluate prompt complexity"""
        prompt = context.get('user_message', '')
        if not prompt: return False

        # Simple complexity metrics
        word_count = len(prompt.split())
        question_marks = prompt.count('?')
        has_code = bool(re.search(r'```|def |class |import ', prompt))
        complex_words = len([w for w in prompt.split() if len(w) > 6])
        char_count = len(prompt)
        
        score = (word_count * 0.02 + question_marks * 0.15 + (0.3 if has_code else 0) + complex_words * 0.05 + char_count * 0.002)
        threshold = config.get('threshold', 0.5)
        
        return score >= threshold
        
    def _eval_history_rule(self, config: Dict, context: Dict) -> bool:
        """Evaluate based on historical performance"""
        lookback = config.get('lookback', 10)
        recent = self._call_history[-lookback:]
        
        if not recent:
            return False
            
        # Calculate success rate
        successes = sum(
            1 for h in recent 
            if h.get('context', {}).get('validation_result') == 'success'
        )
        success_rate = successes / len(recent)
        
        threshold = config.get('success_threshold', 0.7)
        return success_rate < threshold
        
    def _eval_composite_rule(self, config: Dict, context: Dict) -> bool:
        """Evaluate composite rules with AND/OR logic"""
        operator = config.get('operator', 'AND')
        sub_rules = config.get('rules', {})
        
        results = []
        for k, v in sub_rules.items():
            result = self._evaluate_rules({k: v}, context)
            results.append(result)


        if operator == 'AND':
            return all(results)
        elif operator == 'OR':
            return any(results)
        else:
            return False


    # ---- NEW: helpers to configure objective/parameters programmatically ----
    def set_optimizer_objective(self, optimizer_name: str, objective: Optional[str]):
        adapter = self._trace_optimizers.get(optimizer_name)
        if not adapter:
            raise KeyError(f"No optimizer registered with name {optimizer_name!r}")
        adapter.set_objective(objective)

    def update_optimizer_parameter(self, optimizer_name: str, name: str, **kwargs):
        adapter = self._trace_optimizers.get(optimizer_name)
        if not adapter:
            raise KeyError(f"No optimizer registered with name {optimizer_name!r}")
        adapter.update_parameter(name, **kwargs)

    def set_optimizer_parameters(self, optimizer_name: str, params: Union[List[Dict[str, Any]], Dict[str, Dict[str, Any]]]):
        adapter = self._trace_optimizers.get(optimizer_name)
        if not adapter:
            raise KeyError(f"No optimizer registered with name {optimizer_name!r}")
        adapter.set_parameters(params)

    def resolve_optimizable_target(self, target: Union[str, callable, tuple]) -> Tuple[Callable, Callable]:
        """
        Resolve an optimizable target into (getter(context), setter(context, value)).
        Supported target forms:
          - callable getter (setter will raise unless you supply tuple form)
          - tuple ('attr', obj, 'a.b') -> get/set attr on object
          - string paths like 'invoke.temperature', 'user_message', 'system_prompt', 'dynamic_llm_config.some.key'
        """
        if callable(target):
            def getter(ctx):
                return target(ctx)
            def setter(ctx, value):
                raise ValueError("Callable-only target is getter-only. Provide a tuple ('attr', obj, 'path') or a string path for writable targets.")
            return getter, setter

        if isinstance(target, tuple):
            if len(target) != 3 or target[0] != 'attr':
                raise ValueError("Tuple target must be ('attr', obj, 'path')")
            _, obj, path = target
            def getter(ctx):
                cur = obj
                for p in path.split('.'):
                    cur = getattr(cur, p)
                return cur
            def setter(ctx, value):
                cur = obj
                parts = path.split('.')
                for p in parts[:-1]:
                    cur = getattr(cur, p)
                setattr(cur, parts[-1], value)
            return getter, setter

        if isinstance(target, str):
            # handle special known roots
            if target == "user_message":
                def getter(ctx):
                    return ctx.get("user_message")
                def setter(ctx, value):
                    ctx["user_message"] = value
                return getter, setter
            if target == "system_prompt":
                def getter(ctx):
                    return ctx.get("system_prompt")
                def setter(ctx, value):
                    ctx["system_prompt"] = value
                return getter, setter
            if target.startswith("invoke."):
                # e.g. "invoke.temperature" -> ctx['invoke_kwargs']['temperature']
                parts = target.split(".", 1)[1]
                def getter(ctx):
                    return (ctx.get("invoke_kwargs") or {}).get(parts)
                def setter(ctx, value):
                    ctx.setdefault("invoke_kwargs", {})[parts] = value
                return getter, setter
            if target.startswith("dynamic_llm_config."):
                key = target.split(".", 1)[1]
                def getter(ctx):
                    return (ctx.get("dynamic_llm_config") or {}).get(key)
                def setter(ctx, value):
                    ctx.setdefault("dynamic_llm_config", {})[key] = value
                return getter, setter

            # support direct rule_evaluator targets for optimizer visibility/control: 'rule_evaluator.<name>'
            if target.startswith("rule_evaluator."):
                name = target.split(".", 1)[1]
                def getter(ctx):
                    return self.rule_evaluators.get(name)
                def setter(ctx, value):
                    if not callable(value):
                        raise ValueError("rule_evaluator setter expects a callable")
                    self.rule_evaluators[name] = value
                return getter, setter

            # support config.* path to point into self.config (writable)
            if target.startswith("config."):
                # path inside self.config
                path = target.split(".", 1)[1]
                def getter(ctx):
                    return self._get_from_dict_by_path(self.config, path)
                def setter(ctx, value):
                    self._set_in_dict_by_path(self.config, path, value)
                return getter, setter

            # generic dotted path: lookup in context dict, falling back to attributes
            parts = target.split(".")
            def getter(ctx):
                cur = ctx
                for p in parts:
                    if isinstance(cur, dict):
                        cur = cur.get(p)
                    else:
                        cur = getattr(cur, p, None)
                return cur
            def setter(ctx, value):
                cur = ctx
                for p in parts[:-1]:
                    if isinstance(cur, dict):
                        cur = cur.setdefault(p, {})
                    else:
                        cur = getattr(cur, p)
                if isinstance(cur, dict):
                    cur[parts[-1]] = value
                else:
                    setattr(cur, parts[-1], value)
            return getter, setter

        raise ValueError(f"Cannot resolve optimizable target: {target!r}")

    def register_trace_optimizer(self, name: str, config: Dict):
        """Register or update a persistent Trace optimizer by name. config may include 'trace_obj' to plug real Trace instance."""
        if name in self._trace_optimizers:
            # update config if needed
            self._trace_optimizers[name].config.update(config or {})
            # allow replacing trace_obj
            if config and "trace_obj" in config:
                self._trace_optimizers[name]._trace_obj = config["trace_obj"]
            return name
        self._trace_optimizers[name] = _TraceOptimizerAdapter(name, config or {})
        return name

    def list_trace_optimizers(self) -> List[str]:
        return list(self._trace_optimizers.keys())

    def record_outcome(self, context: Dict, modifications: Dict, outcome_metrics: Dict):
        """Store an outcome record for offline learning (H2)."""
        self._records.append({"ts": time.time(), "context": context, "modifications": modifications, "metrics": outcome_metrics})

    def log_user_correction(self, agent_name: str, diff: str, who: str):
        """Store a user diff/correction for priority feedback."""
        self._records.append({"type": "user_diff", "agent": agent_name, "diff": diff, "who": who, "ts": time.time()})

    def collect_feedback(self, context: Dict) -> List[Dict]:
        """
        Return ordered list of feedback candidates: user diffs first, then human annotations, then auto-evals.
        If no records exist, fallback to returning the user_message as a single fallback entry.
        """
        user_diffs = [r for r in self._records if r.get("type") == "user_diff"]
        human_annotations = [r for r in self._records if r.get("type") == "human_annotation"]
        auto_evals = [r for r in self._records if r.get("type") == "auto_eval"]
        # order recent-first inside each class
        out = sorted(user_diffs, key=lambda x: x.get("ts", 0), reverse=True)
        out += sorted(human_annotations, key=lambda x: x.get("ts", 0), reverse=True)
        out += sorted(auto_evals, key=lambda x: x.get("ts", 0), reverse=True)
        if not out:
            um = context.get("user_message")
            if um:
                out = [{"type": "fallback_user_message", "content": um, "ts": time.time()}]
        return out

    def offline_optimize(
        self,
        optimizer_name: Optional[str] = None,
        targets: Optional[List[str]] = None,
        n_candidates: int = 1,
        constraints: Optional[Dict] = None
    ) -> Dict:
        """
        H2 (off-policy): call a Trace optimizer on recorded outcomes to propose persistent edits.
        Applies first candidate:
          - 'config.*'                 -> self.config (persistent)
          - 'dynamic_llm_config.*'     -> self.config['dynamic_llm_defaults'][key]
        Returns summary with 'applied' and 'applied_edits'.
        """
        if not self._records:
            return {"applied": False, "reason": "no_records"}
        stats = {"n_records": len(self._records), "recent_ts": max(r.get("ts", 0) for r in self._records)}
        # Prepare context for offline optimizer — compact, robust normalization:
        # ensure each record has: context -> trace -> backward -> nodes (list).
        def _normalize_record(rec: Dict) -> Dict:
            rc = dict(rec)  # shallow copy; we don't mutate the original record
            ctx = rc.get("context")
            if not isinstance(ctx, dict):
                ctx = rc["context"] = {}
            trace = ctx.get("trace")
            if not isinstance(trace, dict):
                trace = ctx["trace"] = {}
            backward = trace.get("backward")
            if not isinstance(backward, dict):
                backward = trace["backward"] = {}
            nodes = backward.get("nodes")
            backward["nodes"] = list(nodes) if isinstance(nodes, (list, tuple, set)) else []
            return rc

        norm_records = [_normalize_record(r) for r in self._records]
        ctx = {"records": norm_records, "stats": stats}

        name = optimizer_name or self.config.get("trace_default_optimizer") or "offline"
        adapter = self._trace_optimizers.get(name)
        if adapter is None:
            adapter = _TraceOptimizerAdapter(name, {})
            self._trace_optimizers[name] = adapter

        trgts = list(targets or [])
        try:
            # Provide the optimizer a stable spec (variables/objective) even when no graph exists
            ctx["trace_spec"] = adapter.get_trace_spec()
            # --- NEW: standard Trace offline loop (H2) using aggregated feedback ---
            adapter.zero_feedback()
            fb_text = self._render_feedback_text(ctx, {"use": "collect_feedback"})  # reuse ordering of records
            for p in adapter.parameters():
                if getattr(p, "trainable", False):
                    try:
                        adapter.backward(p, fb_text)
                    except Exception:
                        self.logger.exception("Trace backward failed (H2)")
            cands = adapter.step(context=ctx, targets=trgts, n_candidates=int(n_candidates), constraints=constraints)

        except Exception:
            self.logger.exception("offline_optimize: Trace adapter failed.")
            cands = []
        if not cands:
            return {"applied": False, "reason": "no_candidates", "stats": stats}

        first = cands[0]
        applied = []
        for edit in first.get("edits", []):
            tgt, op, val = edit.get("target"), edit.get("op", "set"), edit.get("value")
            if not tgt or op != "set":
                continue
            if tgt.startswith("config."):
                self._set_in_dict_by_path(self.config, tgt.split(".", 1)[1], val); applied.append(edit)
            elif tgt.startswith("dynamic_llm_config."):
                key = tgt.split(".", 1)[1]
                self.config.setdefault("dynamic_llm_defaults", {})[key] = val; applied.append(edit)
        return {"applied": bool(applied), "applied_edits": applied, "stats": stats}
class HumanLLM:
    def __init__(
        self,
        system_prompt=None,
        CPS_env_type=None,
        agent_name=None,
        model_max_context_size=32000,
        default_llmORchain=None,
        premium_llmORchain=None,
        premium_llm_by_default=False,
        num_parallel_inferences=1,
        llmORchains_list=None,
        synthesize_mode=False,
        inference_checks=None,
        output_schema=None,
        temperature_min=None,
        temperature_max=None,
        envs=None,
        fixed_output=False,
        fixed_coach=None, # Retro compatibility
        prompt_critic=None,
        saved_task=None,
        automation=None,
        auto_n_rounds=None,
        recommend_critics=None,
        skip_rounds=None,
        task_parameters=None,
        problem_prompts_subdir=None,
        max_autofix=None,
        skip_log_entry_if_no_change=True,
        selection_technique=None, # Can be "best", "best_of_n", "concat"
        generation_technique='temperature_variation',
        dynamic_llm_config=None,
        **kwargs
    ):
        self.config = HumanLLMConfig()
        self.logger = logging.getLogger(__name__)

        # Instance properties to track time
        self.agent_name = agent_name or self.get_class_name()
        self.task_parameters = task_parameters
        self.selected_outputs = []
        self.menu_start_time = None
        self.start_time = None
        self.current_inference_context = None
        self.user_message_few_shots = None
        if llmORchains_list is None:
            raise ValueError("llmORchains_list must be provided")
        self.llmORchains_list = llmORchains_list
        self.temperature_min = temperature_min
        self.temperature_max = temperature_max if temperature_max else max(temperature_min or 1., 1.)
        self.prompt_critic = prompt_critic
        self.system_prompt = system_prompt
        self.configure_output_schema(output_schema)
        self.set_default_llmORchain(default_llmORchain if default_llmORchain else "default_llm", temperature_min)
        self.set_premium_llmORchain(premium_llmORchain if premium_llmORchain else "premium_llm", temperature_min)
        self.CPS_env_type = CPS_env_type

        self.config.initialize()
        if self.config.use_websocket:
            if self.config.ws_server is None:
                self.config.init_ws_server()
            self.config.ws_server.add_monitor(self)

        self.print_color = self.set_print_color()
        self.previous_templates = []
        self.previous_results = []
        self.comments = []
        self.skip_rounds = skip_rounds if skip_rounds is not None else self.config.default_skip_rounds
        self.log_data = []
        self.llm_input_messages = []
        self.num_parallel_inferences = num_parallel_inferences
        self.llm_max_context_size = model_max_context_size
        self.premium_llm_by_default = premium_llm_by_default
        self.synthesize_mode = synthesize_mode

        self.inference_tracking = InferenceTracking()
        if inference_checks:
            self.inference_tracking.inference_checks = inference_checks

        self.user_message = ""
        self.envs = envs
        self.fixed_output = fixed_output if not fixed_coach else fixed_coach # Retro compatibility
        self.automation = automation
        self.outputs = None
        self.saved_task = saved_task
        self.auto_n_rounds = auto_n_rounds if auto_n_rounds is not None else 0
        self.recommend_critics = recommend_critics
        self.last_user_message = None
        self.primitives_dir = None
        self.processed_codes = set()
        self.max_autofix = max_autofix
        self.problem_prompts_subdir = problem_prompts_subdir
        self.skip_log_entry_if_no_change = skip_log_entry_if_no_change
        self.selection_technique = selection_technique
        self.generation_technique = generation_technique

        # Initialize dynamic configuration system
        self.dynamic_llm_config = dynamic_llm_config or {}
        self.usage_tracker = HelpUsageTracker()
        
        # Set up quotas if provided
        for help_type, config in self.dynamic_llm_config.items():
            if 'quota' in config:
                self.usage_tracker.set_quota(help_type, config['quota'])
        
        # Default configuration that mirrors constructor arguments
        default_cfg = {
            'num_parallel_inferences': num_parallel_inferences,
            'temperature_min': temperature_min,
            'temperature_max': temperature_max,
            'selection_technique': selection_technique,
            'generation_technique': 'temperature_variation',
            'use_premium_llm': premium_llm_by_default
        }
        
        self.dynamic_mgr = DynamicConfigManager(self.dynamic_llm_config, self.usage_tracker, default_cfg)
        self.use_premium_llm = None  # Will be set during inference

    def _apply_modifications(self, mods: Dict, context: Dict, phase: str):
        """Apply modifications from dynamic config evaluation"""
        if not mods:
            return
            
        self.logger.info(f"[DynamicConfig:{phase}] Applying modifications: {mods}")
        
        # Track original values for potential rollback
        original_values = {}
        
        # 1) Direct attribute overrides
        for attr in mods:
            if hasattr(self, attr):
                # Store original value if not already stored
                if attr not in original_values:
                    original_values[attr] = getattr(self, attr)
                setattr(self, attr, mods[attr])
                self.logger.info(f"[DynamicConfig] {attr}: {original_values[attr]} -> {mods[attr]}")

        # 1.b) Structured invocation overrides emitted by DynamicConfigManager
        #      - invoke_kwargs: per-call knobs Trace can change (H1)
        #      - dynamic_llm_config_patch: patch the runtime config (H2 or H1)
        ivk = mods.get("invoke_kwargs") or {}
        if ivk:
            # reflect in context for transparency
            context.setdefault("invoke_kwargs", {}).update(ivk)
            # common H1 knob from Trace: temperature
            if "temperature" in ivk:
                try:
                    t = float(ivk["temperature"])
                    self.temperature_min = t
                    self.temperature_max = t
                except Exception:
                    self.logger.debug("Ignored non-float temperature in invoke_kwargs: %r", ivk.get("temperature"))
            if "num_parallel_inferences" in ivk:
                self.num_parallel_inferences = int(ivk["num_parallel_inferences"])
            if "temperature_max" in ivk:
                self.temperature_max = float(ivk["temperature_max"])
            if "generation_technique" in ivk:
                self.generation_technique = str(ivk["generation_technique"])

        dlc = mods.get("dynamic_llm_config_patch") or {}
        if dlc:
            self.dynamic_llm_config.update(dlc)

        if "system_prompt" in mods:
            self.system_prompt = mods["system_prompt"]
        pre = mods.get("system_prompt_prepend");  app = mods.get("system_prompt_append")
        if pre: self.system_prompt = "".join(pre) + (self.system_prompt or "")
        if app: self.system_prompt = (self.system_prompt or "") + "".join(app)

        # for user_message, update both the context (so invoke() picks it up) and the cached self.user_message
        if "user_message" in mods:
            context["user_message"] = mods["user_message"]
            self.user_message = mods["user_message"]
        upre = mods.get("user_message_prepend");  uapp = mods.get("user_message_append")
        if upre:
            new_um = "".join(upre) + (context.get("user_message") or self.user_message or "")
            context["user_message"] = self.user_message = new_um
        if uapp:
            new_um = (context.get("user_message") or self.user_message or "") + "".join(uapp)
            context["user_message"] = self.user_message = new_um

        # 2) Model selection
        if 'use_premium_llm' in mods:
            context['use_premium_llm'] = mods['use_premium_llm']
        
        if 'model_choice' in mods:
            context['model_choice'] = mods['model_choice']
        
        # 3) Human intervention activation
        if mods.get('activate_human_intervention'):
            original_values['skip_rounds'] = self.skip_rounds
            self.skip_rounds = 0
            self.logger.info("[DynamicConfig] Activated human intervention")
        
        # 4) Feedback generation and application (post-inference only)
        if phase == 'post_inference' and 'llm_outputs' in context:
            self._apply_feedback_modifications(mods, context)
        
        # Store modifications in context for tracking
        context['dynamic_modifications'] = mods
        context['original_values'] = original_values
        
    def _apply_feedback_modifications(self, mods: Dict, context: Dict):
        """Apply feedback-related modifications"""
        outputs = context.get('llm_outputs', [])
        if not outputs:
            return
            
        # Generate annotations if requested
        if 'generate_annotations_feedback' in mods and hasattr(self, 'generate_annotations_feedback_fn'):
            params = mods['generate_annotations_feedback']
            if isinstance(params, dict):
                for i, output in enumerate(outputs):
                    annotations = self.generate_annotations_feedback_fn(
                        inference_result_content=output.content,
                        output_id=i,
                        **params
                    )
                    context.setdefault('annotations', {})[i] = annotations
        
        # Generate instructions if requested
        if 'generate_instructions_feedback' in mods and hasattr(self, 'generate_instructions_feedback_fn'):
            params = mods['generate_instructions_feedback']
            if isinstance(params, dict):
                for i, output in enumerate(outputs):
                    instructions = self.generate_instructions_feedback_fn(
                        inference_result_content=output.content,
                        output_id=i,
                        **params
                    )
                    context.setdefault('instructions', {})[i] = instructions
        
        # Apply feedback if requested
        if mods.get('apply_feedback') and hasattr(self, 'apply_feedback_fn'):
            apply_params = mods.get('apply_feedback_params', {})
            
            for i, output in enumerate(outputs):
                # Determine what feedback to apply
                
                suggestion = ''
                text_content = output.content
                has_annotations = False
                has_instructions = False
                if i in context.get('annotations', {}):
                    text_content = context['annotations'][i]['annotations']
                    has_annotations = True
                if i in context.get('instructions', {}):
                    suggestion = context['instructions'][i]['suggestions']
                    has_instructions = True
                
                if has_annotations or has_instructions:
                    improved = self.apply_feedback_fn(
                        suggestions=suggestion,
                        text_content=text_content,
                        initial_prompt=context.get('system_prompt'),
                        text_has_annotations=has_annotations,
                        **apply_params
                    )
                    # Update the output
                    output.content = improved
                    self.logger.info(f"[DynamicConfig] Applied feedback to output {i}")
            prev_feedbacks = self.config.get_agent_data(self.agent_name, "llm_suggestions")
            for feedback in prev_feedbacks:
                if isinstance(feedback, dict) and 'feedback_applied' in feedback:
                    feedback['feedback_applied'] = True
                if isinstance(feedback, list) :
                    for f in feedback:
                        if isinstance(f, dict) and 'feedback_applied' in f:
                            f['feedback_applied'] = True
        # Re-select best output if requested
        if 'reselect_best' in mods and mods['reselect_best']:
            technique = mods.get('reselection_technique', self.selection_technique)
            if hasattr(self, 'select_candidate'):
                context['llm_outputs'] = self.select_candidate(outputs, technique)
                self.logger.info(f"[DynamicConfig] Re-selected best output using {technique}")

    def get_rag_documents(self, agent_name=None, extra_filter: Optional[Dict[str, Any]] = None, query: str = '*', **kwargs):
        """
        Convenience method to retrieve only RAG-indexed documents.
        It wraps get_agent_data by enforcing metadata_filter with {"rag": True}.
        """
        metadata_filter = {"rag": True}
        if extra_filter:
            metadata_filter.update(extra_filter)
        return self.config.get_agent_data(agent_name=agent_name, metadata_filter=metadata_filter, query_text=query, **kwargs)
    
    @staticmethod
    def decode_bytes(obj):
        if isinstance(obj, dict):
            return {k: HumanLLM.decode_bytes(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [HumanLLM.decode_bytes(item) for item in obj]
        elif isinstance(obj, bytes):
            return obj.decode('utf-8', errors='replace')
        else:
            return obj
    
    @staticmethod
    def serialize_metadata(metadata):
        new_metadata = {}
        for key, value in metadata.items():
            if isinstance(value, IndirectObject):
                new_metadata[key] = str(value)
            else:
                new_metadata[key] = value
        return new_metadata

    def clean_metadata(self, meta):
        cleaned = {}
        for key, value in meta.items():
            # Convert bytes to string
            if isinstance(value, bytes):
                cleaned[key] = value.decode('utf-8', errors='ignore')
            # Convert PyPDF2 IndirectObject to string
            elif hasattr(value, "getObject"):  # A simple check for IndirectObject
                try:
                    # You can try to extract the actual object if needed:
                    obj = value.getObject()
                    cleaned[key] = str(obj)
                except Exception:
                    cleaned[key] = str(value)
            else:
                cleaned[key] = value
        return cleaned
               
    def add_rag_document(
        self,
        file_path: str,
        metadata: Optional[Dict[str, Any]] = None,
        chunking_options: Optional[Dict[str, Any]] = None,
        folder_path: str = None,
        overwrite: bool = False,
        use_marker: bool = False,  # If True, convert PDF to Markdown via Marker Docker.
        use_semantic_chunking: bool = True  # If True, use semantic double-pass merging chunking.
    ) -> None:
        """
        Loads and indexes an external document (JSON, PDF, HTML, Markdown, etc.) for RAG.
    
        - Uses appropriate LangChain loaders based on file extension.
        - If chunking_options is provided, the document is split into smaller pieces for efficient indexing.
        By default, uses a semantic double-pass merging chunking method.
        - Uses self.premium_llm to extract a more precise title and authors based on the first 1000 characters,
        the file name, and provided metadata.
        - Optionally, for PDFs, converts to Markdown first using Marker via Docker (if use_marker=True).
        - Prevents re-adding an already indexed document unless `overwrite=True`.
        """
        print(f"Loading RAG document: {file_path}")
    
        if folder_path:
            file_path = os.path.join(folder_path, file_path)

        # Extract the file name
        file_name = os.path.basename(file_path)
    
        # Check if the file is already indexed
        existing_docs = self.config.get_agent_data(
            agent_name=self.__class__.__name__,
            metadata_filter={"rag": True, "file_name": file_name},
            query_text="*"
        )
        if any(existing_docs) and not overwrite:
            print(f"The document '{file_name}' already exists. Use overwrite=True to force re-indexing.")
            return

        ext = os.path.splitext(file_path)[1].lower()
        extra_metadata = {"file_name": file_name}

        # Convert PDF to Markdown using Marker via Docker if enabled
        if ext == '.pdf' and use_marker:
            try:
                print("Converting PDF -> Markdown using Marker Docker ...")
                # Use a temporary directory to mount the file
                tmp_dir = tempfile.mkdtemp()
                temp_pdf_path = os.path.join(tmp_dir, file_name)
                # Copy the PDF file to the temporary directory
                with open(file_path, "rb") as src, open(temp_pdf_path, "wb") as dst:
                    dst.write(src.read())
                # Build the Docker command to convert to Markdown
                cmd = [ "docker", "run", "--rm", "-v", f"{tmp_dir}:/data", "dibz15/marker_docker", file_name] # The container reads the file from /data
                subprocess.run(cmd, check=True)
                # Assume Marker creates a Markdown file with the same name but with a .md extension
                md_file_name = os.path.splitext(file_name)[0] + ".md"
                new_file_path = os.path.join(tmp_dir, md_file_name)
                if os.path.exists(new_file_path):
                    file_path = new_file_path
                    ext = ".md"
                    extra_metadata["converted_with_marker"] = True
                    print(f"Conversion successful, new file: {file_path}")
                else:
                    print("Error: Converted Markdown file not found.")
            except Exception as e:
                print(f"Error during PDF -> Markdown conversion: {e}")
    
        # Extract metadata from PDFs if not converted via Marker
        if ext == '.pdf' and not use_marker:
            try:
                import PyPDF2
                with open(file_path, "rb") as f:
                    reader = PyPDF2.PdfReader(f)
                    pdf_meta = reader.metadata
                    if pdf_meta:
                        cleaned_meta = self.clean_metadata(dict(pdf_meta))
                        extra_metadata.update(cleaned_meta)
            except Exception as e:
                print(f"Error extracting PDF metadata: {e}")

        # Select the appropriate LangChain loader
        if ext == '.pdf':
            from langchain.document_loaders import PyPDFLoader
            loader = PyPDFLoader(file_path)
        elif ext == '.json':
            from langchain.document_loaders import JSONLoader
            loader = JSONLoader(file_path)
        elif ext in ['.html', '.htm']:
            from langchain.document_loaders import UnstructuredHTMLLoader
            loader = UnstructuredHTMLLoader(file_path)
        elif ext == '.md':
            from langchain.document_loaders import UnstructuredMarkdownLoader
            loader = UnstructuredMarkdownLoader(file_path)
        else:
            from langchain.document_loaders import UnstructuredFileLoader
            loader = UnstructuredFileLoader(file_path)

        docs = loader.load()

        # Use self.premium_llm to extract a better title and authors from
        # the first 1000 characters, file name, and metadata "name" if available.
        if self.premium_llm:
            try:
                text_sample = docs[0].page_content[:1000] if docs and docs[0].page_content else ""
                if len(text_sample)<1000 and len(docs)>1: text_sample += docs[1].page_content[:(1000-len(text_sample))] if docs and docs[1].page_content else ""
                metadata_name = metadata.get("name", "") if metadata else ""
                prompt = (
                    f"Extract the document title and authors from the following details.\n"
                    f"Document first 1000 characters: {text_sample}\n"
                    f"File name: {file_name}\n"
                    f"Metadata name: {metadata_name}\n\n"
                    f"Return a valid JSON with keys 'title' and 'authors'."
                )
                llm_output = self.premium_llm.invoke([HumanMessage(content=prompt)])
                try:
                    parsed = extract_json(llm_output.content)
                    extracted_title = parsed.get("title", "").strip()
                    extracted_authors = str(parsed.get("authors", "")).strip()
                    extra_metadata.update({ "extracted_title": extracted_title, "extracted_authors": extracted_authors})
                    print("LLM extraction successful:", extracted_title, extracted_authors)
                except Exception as parse_ex:
                    print(f"Error parsing LLM response for title/authors: {parse_ex}")
            except Exception as e:
                print(f"Error extracting title/authors using LLM: {e}")

        # Apply chunking if options are provided
        if chunking_options:
            chunked_docs = []
            if use_semantic_chunking:
                print("Using semantic double-pass merging chunking...")
                for doc in docs:
                    chunks = semantic_double_pass_chunking(doc.page_content, **chunking_options)
                    for chunk in chunks:
                        chunked_docs.append(type(doc)(page_content=chunk, metadata=doc.metadata))
            else:
                print("Using default chunking (RecursiveCharacterTextSplitter)...")
                from langchain.text_splitter import RecursiveCharacterTextSplitter
                splitter = RecursiveCharacterTextSplitter(**chunking_options)
                for doc in docs:
                    chunks = splitter.split_text(doc.page_content)
                    for chunk in chunks:
                        chunked_docs.append(type(doc)(page_content=chunk, metadata=doc.metadata))
            docs = chunked_docs

        # Index documents with enriched metadata
        for doc in docs:
            combined_metadata = metadata.copy() if metadata else {}
            combined_metadata.update(extra_metadata)

            if doc.metadata:
                for key, value in doc.metadata.items():
                    combined_metadata.setdefault(key, value)

            combined_metadata.update({"rag": True, "source": ext})
        
            self.config.log_agent_data(
                self.__class__.__name__,
                "rag_knowledge",
                doc.page_content,
                metadata=combined_metadata
            )
        
    def load_prompt_with_rag(self, prompt_name: str, template_data: Optional[Dict[str, Any]] = None,
                             directory: Optional[str] = None) -> str:
        """
        Wraps the existing load_prompt to include RAG context.
        
        This method:
         - Retrieves the base prompt using load_prompt (unchanged).
         - Uses get_rag_documents to fetch RAG-indexed texts.
         - Replaces the "{rag_context}" placeholder in the prompt with the aggregated RAG data.
        """
        base_prompt = self.config.load_prompt_template(prompt_name, template_data=template_data, directory=directory)
        rag_docs, _ = self.get_rag_documents(agent_name=self.__class__.__name__)
        if rag_docs:
            rag_context = "\n".join([doc.get("rag_knowledge", "") for doc in rag_docs])
        else:
            rag_context = ""
        return base_prompt.replace("{rag_context}", rag_context)

    def set_print_color(self):
        self.print_color = 37
        color_table = {
            "32": ["ActionAgent", "CodingAgent"],
            "35": ["CurriculumAgent", "TaskIdentificationAgent"],
            "31": ["CriticAgent", "ValidationAgent"],
            "33": ["SkillManager", "CapitalizationAgent"],
        }
        for key, value in color_table.items():
            if self.agent_name in value:
                self.print_color = key

    def initialize(self):
        if self.config.use_websocket and self.config.ws_server is None:
            self.config.init_ws_server()
            self.config.ws_server.add_monitor(self)

    def get_user_id(self):
        return self.config.get_user_id()

    def configure_vector_store(self):
        self.config.configure_vector_store()

    def set_common_vectordb_embedding_function(self):
        self.config.common_vectordb.config.set_common_vectordb_embedding_function()

    def configure_llm(self, llm_name, is_premium=False, temperature=None):
        if llm_name in self.llmORchains_list:
            if is_premium:
                self.premium_llm_name = llm_name
            else:
                self.default_llm_name = llm_name

            selected_llm_or_chain = self.llmORchains_list[llm_name]

            # Check if it's a sequence of steps (RunnableSequence)
            if isinstance(selected_llm_or_chain, RunnableSequence):
                modified_steps = []
                for step in selected_llm_or_chain.steps:
                    if hasattr(step, "steps__") and isinstance(step.steps__, dict):
                        # Handle the case where step is a dict of parallel runnables
                        modified_dict = {}
                        for key, sub_step in step.steps__.items():
                            if hasattr(sub_step, 'configurable_fields'):
                                try:
                                    sub_step = sub_step.configurable_fields(
                                        temperature=ConfigurableField(
                                            id="llm_temperature",
                                            name="LLM Temperature",
                                            description="The temperature of the LLM"
                                        )
                                    ).with_config(configurable={
                                        "llm_temperature": temperature})  # Replace with desired default temperature
                                except ValueError as e:
                                    smart_print(
                                        f"Sub-step {key} in step {step} does not support temperature configuration: {e}",
                                        self.agent_name,
                                        "set_llmORchain",
                                        optional=True
                                    )
                            modified_dict[key] = sub_step
                        step.steps__ = modified_dict
                        modified_steps.append(step)
                    elif hasattr(step, 'configurable_fields'):
                        try:
                            step = step.configurable_fields(
                                temperature=ConfigurableField(
                                    id="llm_temperature",
                                    name="LLM Temperature",
                                    description="The temperature of the LLM"
                                )
                            ).with_config(configurable={
                                "llm_temperature": temperature})  # Replace with desired default temperature
                        except ValueError as e:
                            print(f"Step {step} does not support temperature configuration: {e}", self.agent_name)
                        modified_steps.append(step)
                    else:
                        modified_steps.append(step)

                # Reconstruct the sequence with the modified steps
                selected_llm_or_chain = RunnableSequence(
                    first=modified_steps[0],
                    middle=modified_steps[1:-1] if len(modified_steps) > 2 else None,
                    last=modified_steps[-1]
                )
            elif hasattr(selected_llm_or_chain, 'configurable_fields'):
                # If it's a single LLM or other runnable that supports configurable fields, apply directly
                try:
                    selected_llm_or_chain = selected_llm_or_chain.configurable_fields(
                        temperature=ConfigurableField(
                            id="llm_temperature",
                            name="LLM Temperature",
                            description="The temperature of the LLM"
                        )
                    ).with_config(
                        configurable={"llm_temperature": temperature})  # Replace with desired default temperature
                except ValueError as e:
                    smart_print(
                        f"LLM/Chain '{llm_name}' does not support temperature configuration: {e}",
                        self.agent_name,
                        optional=True
                    )

            # Apply structured output if needed
            if self.output_schema:
                if isinstance(selected_llm_or_chain, RunnableSequence):
                    # Apply `with_structured_output` to the last element in the sequence
                    last_element = selected_llm_or_chain.steps[-1]
                    if hasattr(last_element, 'with_structured_output'):
                        last_element = last_element.with_structured_output(self.output_schema)
                    # Reconstruct the sequence with the modified last element
                    if len(selected_llm_or_chain.steps) > 1:
                        selected_llm_or_chain = RunnableSequence(
                            first=selected_llm_or_chain.steps[0],
                            middle=selected_llm_or_chain.steps[1:-1] if len(selected_llm_or_chain.steps) > 2 else None,
                            last=last_element
                        )
                    else:
                        # If there's only one step, treat the last_element as the entire sequence
                        selected_llm_or_chain = last_element
                elif hasattr(selected_llm_or_chain, 'with_structured_output'):
                    # Apply `with_structured_output` directly if it's not a sequence
                    selected_llm_or_chain = selected_llm_or_chain.with_structured_output(self.output_schema)
                else:
                    # If neither condition matches, `selected_llm_or_chain` is not modified
                    smart_print(f"LLM/Chain '{llm_name}' does not support structured output", self.agent_name, optional=True)

            # Set the LLM/Chain to the possibly modified or original one
            if is_premium:
                self.premium_llm = selected_llm_or_chain
            else:
                self.default_llm = selected_llm_or_chain

            return True
        else:
            smart_print(
                f"LLM/Chain '{llm_name}' not found in llmORchains_list {[key for key in self.llmORchains_list]}",
                self.agent_name,
                optional=True
            )
            return False

    def set_default_llmORchain(self, llm_name, temperature=None):
        return self.configure_llm(llm_name, is_premium=False, temperature=temperature)

    def set_premium_llmORchain(self, llm_name, temperature=None):
        return self.configure_llm(llm_name, is_premium=True, temperature=temperature)

    def configure_output_schema(self, output_schema, package_path="."):
        # test if output_schema is a string, then it means it is a filename located in the prompt repo, load it and set it as output_schema
        if isinstance(output_schema, str):
            if "/" not in output_schema:
                self.output_schema_path = f"{package_path}/prompts/{output_schema}"
            output_schema = self._get_pydantic_class(self.output_schema_path)
        self.output_schema = output_schema

    def _get_pydantic_class(self, file_path: str):
        # Dynamically import the module from the given file path
        import importlib.util
        import sys
        from pydantic import BaseModel

        module_name = "dynamic_module"
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)

        # Iterate through the attributes of the module to find the Pydantic class
        for attr_name in dir(module):
            attr = getattr(module, attr_name)
            if isinstance(attr, type) and issubclass(attr, BaseModel) and attr is not BaseModel:
                return attr

        raise ValueError("No Pydantic BaseModel class found in the provided file.")

    # Clears the selected answers before processing new outputs.
    # This should be called at the beginning of a new inference process.
    def clear_selected_outputs(self):
        self.selected_outputs = []

    # Store the list of selected answers/outputs provided by the frontend.
    # This should be called during the after_inference process.
    def set_selected_outputs(self, selected_outputs):
        self.selected_outputs = selected_outputs

    # Retrieve the list of selected answers/outputs.
    # External agents can use this method to access the selected outputs.
    def get_selected_outputs(self):
        return self.selected_outputs

    # New: Handling function calls via WebSocket
    def run_tool(self, function_name, params):
        if hasattr(self, function_name):
            func = getattr(self, function_name)

            if callable(func):
                # Récupérer les informations de la signature de la fonction
                func_signature = inspect.signature(func)
                param_count = len(func_signature.parameters)

                # Si la fonction attend un seul argument positionnel
                if param_count == 1 and not isinstance(params, dict):
                    return func(params)

                # Si la fonction attend plusieurs arguments positionnels
                elif param_count > 1 and isinstance(params, (list, tuple)):
                    return func(*params)

                # Si la fonction attend des mots-clés et `params` est un dictionnaire
                elif isinstance(params, dict):
                    return func(**params)

                else:
                    raise TypeError(
                        f"Cannot match parameters to function signature. Expected {param_count} parameters but received {type(params).__name__}.")

            self.log_timing_and_call(function_name)
        # Retourner une valeur par défaut si la fonction n'existe pas ou n'est pas callable
        return None

    def log_timing_and_call(self, action, mode=None, reset_menu_time_after=True):
        """
        Track the time spent on each menu option or action.

        Args:
        - action (str): The action being performed (e.g., 'A', 'B', etc.)
        - is_before (bool): True if tracking for pre_inference, False for post_inference
        """
        if mode == 'before' or (mode is None and self.mode == 'before'):
            time_dict, count_dict = self.before_inference_option_times, self.before_inference_option_counts
        elif mode == 'after' or (mode is None and self.mode == 'after'):
            time_dict, count_dict = self.after_inference_option_times, self.after_inference_option_counts
        else:
            time_dict, count_dict = self.unidentified_option_times, self.unidentified_option_counts

        if action:
            if action not in time_dict:
                time_dict[action] = 0
                count_dict[action] = 0
            time_dict[action] += (time.time() - self.start_time)
            count_dict[action] += 1

        # Track total time
        time_dict["TOTAL"] += (time.time() - self.menu_start_time)
        count_dict["TOTAL"] += 1

        # Track selection time
        time_dict["SELECTION"] += (self.start_time - self.menu_start_time)
        count_dict["SELECTION"] += 1

        # Reset times for the next action
        if reset_menu_time_after:
            self.start_time, self.menu_start_time = time.time(), time.time()

    def generate_summary(self, examples: List[str], char_limit: int) -> str:
        """
        Generate a summary of the examples using a language model.

        :param examples: List of example strings to summarize.
        :param char_limit: Maximum character limit for the summary.
        :return: A string containing the generated summary.
        """
        # Set up the language model and prompt
        llm = OpenAI(temperature=self.temperature_min)
        prompt = PromptTemplate(
            input_variables=["examples"],
            template="Summarize the following examples in {char_limit} characters or less:\n\n{examples}"
        )

        # Create a chain to generate the summary
        chain = LLMChain(llm=llm, prompt=prompt)

        # Generate the summary
        summary = chain.run(examples="\n".join(examples), char_limit=char_limit)

        return f"Summary: {summary.strip()}"

    def add_manage_inference_check(self, check_name, check_function):
        self.inference_tracking.inference_checks[check_name] = InferenceCheck(check_name, check_function)

    def exclude_manage_inference_check(self, check_name: list):
        for name in check_name:
            while name[0] == " ":
                name = name[1:]
            if name in self.inference_tracking.inference_checks:
                self.inference_tracking.excluded_inference_checks.append(name)

    def include_manage_inference_check(self, check_name: list):
        for name in check_name:
            while name[0] == " ":
                name = name[1:]
            if name in self.inference_tracking.excluded_inference_checks:
                del self.inference_tracking.excluded_inference_checks[self.inference_tracking.excluded_inference_checks.index(name)]

    def run_manage_inference_checks(self, output_id, *args, **kwargs):
        results = {}
        for check_name, check in self.inference_tracking.inference_checks.items():
            if check_name not in self.inference_tracking.excluded_inference_checks:
                result = check.run_check(output_id, *args, **kwargs)
                results[check_name] = result
        # Ensure output_id is within bounds before updating the list
        if 0 <= output_id < len(self.inference_tracking.last_inference_check_results):
            self.inference_tracking.last_inference_check_results[output_id] = results  # Update the specific index
        return results

    def get_class_name(self):
        # Returns the name of the class that called the current function
        if "self" in inspect.stack()[2][0].f_locals:
            return inspect.stack()[2][0].f_locals["self"].__class__.__name__
        return None

    def check_token_limit(self, content):
        import tiktoken
        encoding = tiktoken.encoding_for_model("gpt-4o-mini")  # TODO: replace by appropriate call to self.xxxxxx
        token_length = len(encoding.encode(content))

        if token_length > self.llm_max_context_size:
            smart_print(
                f"\033[31mCANNOT SEND MESSAGE TO LLM:\n{content}\n\nToo many tokens in human message for LLM ({token_length}). Fallback to manual feedback.\033[0m",
                self.agent_name, optional=True)
            return False
        else:
            return True

    def synthesize_responses(self, responses, use_default_llm):
        system = """You have been provided with a set of responses from various open-source models to the latest user query. Your task is to synthesize these responses into a single, high-quality response while keeping the same output format structure. It is crucial to first critically evaluate the information provided in these responses, recognizing that some of it may be biased or incorrect. Your response should not simply replicate the given answers but should offer a refined, accurate, and comprehensive reply to the instruction with the same format output. Ensure your response is well-structured, coherent, and adheres to the highest standards of accuracy and reliability."""
        messages = [SystemMessage(content=system), HumanMessage(content="\n".join(responses))]
        formatted_responses = "\n\n".join(
            [f"RESPONSE {i + 1}: [[\n{response}\n]]" for i, response in enumerate(responses)])  # NEW/UPDATED
        messages = [SystemMessage(content=system), HumanMessage(content=formatted_responses)]  # NEW/UPDATED
        if use_default_llm:
            return self.default_llm.invoke(messages)
        else:
            return self.premium_llm.invoke(messages)

    def pre_inference(
            self,
            messages,
            default_llm_function,
            premium_llm_function,
            function_calling,
            callable_system_message=None,
            use_premium_llm=None,
            model_choice=None,
            task_name=None,
            forced_llm_output=False,  # TODO: try to set it to None
    ):
        self.mode = 'before'
        comments = None
        self.user_message = initial_user_message = messages[1].content
        function_name = inspect.stack()[2].function
        use_premium_llm = use_premium_llm if use_premium_llm is not None else self.premium_llm_by_default
        forced_llm_output = forced_llm_output
        if not self.skip_log_entry_if_no_change:
            self.config.log_agent_data( self.agent_name,
                "saved_task",
                {
                    'prompt': messages[0].content + messages[1].content,
                    'num_parallel_inferences': self.num_parallel_inferences,
                    'task_parameters': self.task_parameters
                }, before_after='before', user_id=self.get_user_id(), step_id=self.config.step_id, task_type="IR_CPS_TechSynthesis", task_id=True)

        while self.skip_rounds <= 0:
            # MENU
            menu = ''
            before_menu = f"\033[{self.print_color}m***** {self.agent_name}->{function_name}  BEFORE *****\nSYSTEM PROMPT:\n{messages[0].content}\n\nUSER MESSAGE:\n{messages[1].content}\n***** {self.agent_name}->{function_name} BEFORE *****\033[0m\n"
            if not self.check_token_limit(messages[0].content + "\n" + messages[1].content):
                before_menu += ("WARNING!!!! Max tokens exceeded, you should refactor user message or system prompt!\n")
            menu += (
                "[A] Modify agent's system prompt\n")  # Modify agent's 'system prompt' (role, global context, constraints, examples) OR the answer SCHEMA output.\n")
            menu += ("[B] Give instruction or information to agent\n")
            menu += ("[C] Skip & set agent output (from recent or manually)\n")
            menu += ("[D] Log comments (not used by the model, just for information)\n")
            menu += ("[E] See previous results\n")
            menu += ("[F] See MODIFIED/SCORED/COMMENTED results\n")
            menu += ("[G] Skip for N rounds (auto mode)\n")
            menu += ("[H] Change default agent\n")
            menu += ("[I] Change premium agent\n")
            menu += (
                f"[J] Set num of parallel inferences ({self.num_parallel_inferences}, Synthesis={'ON' if self.synthesize_mode else 'OFF'})\n")  # UPDATED
            menu += ("[K] Exit\n")
            menu += (f"[P] Generate with a PREMIUM agent (default:{use_premium_llm})\n")
            menu += ("[R] Activate/Deactivate inferences checks\n")
            menu += ("[Z] Continue\n")

            smart_print(before_menu + menu, self.agent_name, "BEFORE inference action MENU", optional=False)
            self.menu_start_time = time.time()
            if self.automation == 'coach':
                llm_keys = list(self.llmORchains_list.keys())
                if isinstance(model_choice, int):
                    # Model change from choice of optuna
                    new_llm_name = llm_keys[model_choice]
                elif isinstance(model_choice, str):
                    if model_choice not in llm_keys:
                        raise ValueError(f"Model choice '{model_choice}' not found in llmORchains_list {llm_keys}")
                    new_llm_name = model_choice
                else:
                    raise ValueError("Model choice must be an integer or a string")

                if self.agent_name == "TaskIdentificationAgent":
                    if self.num_parallel_inferences > 1:
                        self.synthesize_mode = True

                self.set_default_llmORchain(new_llm_name)
                self.set_premium_llmORchain(new_llm_name)
                default_llm_function = self.default_llm
                premium_llm_function = self.premium_llm
                self.synthesize_mode = False
                # Default actions for all agents while running with optuna
                action = ""
            elif self.automation:
                action = ""
            else:  # Default case
                action = smart_input(
                    f"\033[32mBEFORE\033[0m inference @ {self.agent_name}-> Choose an action (or hit Enter for inference) :",
                    self.agent_name, optional=False)
                action = action.upper()

            # ACTIONS processing
            self.start_time = time.time()  # Init action selected and timer to measure time spent and occurrences in action processing

            if self.fixed_output:
                # We force the output of the llm.
                forced_llm_output = self.fixed_output

            if action == "A":  # Modify system prompt
                comments, forced_llm_output = self.update_prompt_template(
                    callable_system_message,
                    comments,
                    default_llm_function,
                    forced_llm_output,
                    messages,
                    premium_llm_function,
                    use_premium_llm
                )

                # Adding the logic to edit the output schema
                if hasattr(self, 'output_schema_path') and self.output_schema_path:
                    new_schema_content = _visual_input(open(self.output_schema_path).read(), filetype="py")
                    confirm_schema = smart_input(
                        "Do you want to replace the current output schema with your input? (y/n): ",
                        self.agent_name).upper()
                    if confirm_schema == "Y":
                        with open(self.output_schema_path, 'w') as schema_file:
                            schema_file.write(new_schema_content)
                        self.configure_output_schema(self.output_schema_path)
                        # Reload default and premium LLMs
                        self.set_default_llmORchain(self.default_llm_name)
                        self.set_premium_llmORchain(self.premium_llm_name)
                        default_llm_function = self.default_llm
                        premium_llm_function = self.premium_llm

            elif action == "B":  # Add instruction or information to agent
                self.prepend_instruction(initial_user_message)

            elif action == "C":  # Set LLM output by re-using past
                forced_llm_output = self.optimize_response(forced_llm_output, function_name)

            elif action == "D":  # Log comments
                comments = self.log_comments(comments)

            elif action == "E":  # See all previous results
                result = self.get_previous_results(function_name, self.agent_name)
                _visual_input(result)

            elif action == "F":  # See previous MODIFIED/SCORED/COMMENTED results
                result = self.get_scored_results(function_name)
                if result is not None:
                    messages = [
                        SystemMessage(content=self.config.load_prompt_template(
                            prompt_name=self.system_prompt,
                            directory='prompts'
                        )),
                        HumanMessage(content=result)
                    ]
                    for message in messages:
                        smart_print(message.content, self.agent_name, "BEFORE inference action MENU", optional=True)

            elif action == "G":  # Skip human actions for N rounds
                self.skip_iteration()

            elif action == "H":  # Change default LLM
                default_llm_function = self.change_default_llm(default_llm_function)

            elif action == "I":  # Change premium LLM
                premium_llm_function = self.change_premium_llm(premium_llm_function)

            elif action == "K":  # Exit program
                self.shutdown()

            elif action == "J":  # Change num of parallel inferences and synthesize mode
                self.set_parallel_inferences_and_synthesize()

            elif action == "R":
                self.toggle_inference_checks()

            # Count time spent and occurrences waiting and in each option
            self.log_timing_and_call(action, mode='before')

            if action in [None, "", "P", "C", "Z"]:
                if action == "P":
                    use_premium_llm = True
                break
            else:
                proceed = "y" if self.automation else "n"
                if proceed in ["y", "p", ""]:
                    if proceed == "p":
                        use_premium_llm = True
                    break

        smart_print(
            f"Time spent in each option and occurrences: {self.before_inference_option_times} - {self.before_inference_option_counts}",
            self.agent_name, optional=True)

        self.mode = None
        messages[1].content = self.user_message
        return messages, comments, forced_llm_output, use_premium_llm, default_llm_function, premium_llm_function, function_calling

    def toggle_inference_checks(self):
        menu = "Current inference checks:\n"
        for check_name in self.inference_tracking.inference_checks:
            menu += f"- {check_name} {'(Deactivated)' if check_name in self.inference_tracking.excluded_inference_checks else ''}\n"
        menu += "\n[A] Activate inference check\n"
        menu += "[E] Deactivate inference check\n"
        answer = smart_input(menu, self.agent_name, "ACTIVATE/DEACTIVATE INFERENCE CHECKS")
        if answer.lower() == "a":
            check_name = smart_input(
                "Enter the names of the inference check to activate (if many to activate, separated by comma): ",
                self.agent_name,
                "ACTIVATE INFERENCE CHECK"
            ).title().split(",")
            self.include_manage_inference_check(check_name)
        else:
            check_name = smart_input(
                "Enter the names of the inference check to deactivate (if many to deactivate, separated by comma): ",
                self.agent_name,
                "DEACTIVATE INFERENCE CHECK"
            ).title().split(",")
            self.exclude_manage_inference_check(check_name)

    def set_parallel_inferences_and_synthesize(self):
        try:
            self.num_parallel_inferences = int(
                smart_input(
                    "Enter new value for num_parallel_inferences: ",
                    self.agent_name,
                    "NUM_PARALLEL_INFERENCES"
                )
            )
        except:
            self.num_parallel_inferences = 1
        synthesize_mode_input = smart_input(
            "Turn synthesis mode on/off (1 for ON, 0 for OFF): ",
            self.agent_name,
            "NUM_PARALLEL_INFERENCES SYNTHESIS MODE CHOICE"
        ).strip()
        if synthesize_mode_input in ["0", "1"]:
            self.synthesize_mode = synthesize_mode_input == "1"
        else:
            smart_print(
                "Invalid input. Synthesize mode remains unchanged.",
                self.agent_name,
                "NUM_PARALLEL_INFERENCES SYNTHESIS MODE CHOICE",
                optional=True
            )

    def set_parallel_inferences(self):
        try:
            self.num_parallel_inferences = int(
                smart_input(
                    "Enter new value for num_parallel_inferences: ",
                    self.agent_name,
                    "NUM_PARALLEL_INFERENCES"
                )
            )
        except:
            self.num_parallel_inferences = 1

    def shutdown(self):
        if 'IN_NOTEBOOK' in globals() and globals()['IN_NOTEBOOK']:
            # access to AgentDisplayManager.export_to_html() which is not registered in this file but in the notebook
            raise SystemExit("I just wanted to stop!")
        else:
            exit()

    def skip_iteration(self):
        rounds = int(smart_input("Skip for how many rounds? ", self.agent_name))
        self.skip_rounds = rounds

    def log_comments(self, comments):
        comments = smart_input("Enter your comment on the prompt: ", self.agent_name)
        return comments

    def optimize_response(self, forced_llm_output, function_name):
        if self.fixed_output:
            selected_index = 1
            log_entries, list_output = self.config.retrieve_logs(self.agent_name, function_name), ""
        elif self.config.common_vectordb.count() > 0:
            log_entries, list_output = self.config.retrieve_logs(self.agent_name, function_name), ""
            for idx, entry in enumerate(log_entries, start=1):
                content = json.loads(entry.page_content)
                text = (
                    content['output_contents'][0]['content'].replace('\n', '\\')
                    if content['output_contents']
                    else ""
                ) if isinstance(content['output_contents'], list) else \
                    content['output_contents']['content'].replace('\n', '\\')
                date = entry.metadata['time'].split('.')[0]
                list_output += (
                    f"\033[94m{idx}.\033[0m {text[:100]}....{text[-100:]} #{entry.metadata['function_name']} @{date}\n")  # Display a snippet of each entry
            smart_print(list_output, self.agent_name, "LOG ENTRIES LIST", optional=True)

            try:
                selected_index = int(
                    smart_input(
                        "Select the log entry number to load or 0/enter to manually enter LLM output: ",
                        self.agent_name)) - 1
            except:
                selected_index = -1
        else:
            selected_index = -1
        if selected_index < 0 or selected_index >= len(log_entries):
            forced_llm_output = _visual_input("{replace with expected ANSWER/OUTPUT}")
        else:
            selected_log_entry = json.loads(log_entries[selected_index].page_content)
            forced_llm_output = (
                selected_log_entry['output_contents'][0]
                if isinstance(selected_log_entry['output_contents'], list)
                else selected_log_entry['output_contents']
            )['content']
        return forced_llm_output

    def prepend_instruction(self, initial_user_message):
        instructions = smart_input(
            "ENTER ADDITIONAL INSTRUCTIONS FOR THE AGENT: ",
            self.agent_name,
            message_type="ADDITIONAL_INFO",
            optional=False
        )
        self.user_message = initial_user_message + f"\n\nADDITIONAL INSTRUCTIONS: << {instructions} >>"

    def update_prompt_template(
        self,
        callable_system_message,
        comments,
        default_llm_function,
        forced_llm_output,
        messages,
        premium_llm_function,
        use_premium_llm
    ):
        new_template = None
        # List existing prompt variants including the base prompt
        prompt_variants = list_prompt_variants(self.system_prompt)
        output = ("Found the following prompt options:\n")
        for i, variant in enumerate(prompt_variants):
            output += (f"{i + 1}. {variant}\n")
        output += (
            f"{len(prompt_variants) + 1}. Ask LLM to generate a new variant of the current system prompt given my instructions\n")
        smart_print(output, self.agent_name, "PROMPT OPTIONS", optional=True)
        variant_choice = smart_input(
            "Select a number to modify a prompt or create a new variant (or press Enter to continue with the current selection): ",
            self.agent_name
        )
        if variant_choice.isdigit() and 0 < int(variant_choice) <= len(prompt_variants) + 1:
            if int(variant_choice) == len(prompt_variants) + 1:
                # Process to create a new variant
                comments = smart_input("Provide critic or feedback for the current prompt: ", self.agent_name)
                refine_prompt = _visual_input(
                    f"Current system prompt:<<< {self.config.load_prompt_template(prompt_name=self.system_prompt, directory='prompts')} >>>\n\nFeedback or critic: {comments}")
                forced_llm_output = default_llm_function.invoke(
                    [
                        SystemMessage(
                            content=self.config.load_prompt_template(
                                prompt_name="improve_prompt_from_answer_critic",
                                directory='prompts'
                            )
                        ),
                        HumanMessage(content=refine_prompt)
                    ]
                )
                new_template = forced_llm_output.content
            else:
                self.system_prompt = prompt_variants[int(variant_choice) - 1]

        if smart_input(
            "Would you like first to get suggestions for a better prompt? (y/n): ",
            self.agent_name
        ).upper() == "Y":
            if use_premium_llm:
                forced_llm_output = premium_llm_function.invoke(
                    [
                        SystemMessage(
                            content=self.config.load_prompt_template(
                                prompt_name="system_prompt_refiner",
                                directory='prompts'
                            )
                        ),
                        HumanMessage(
                            content=f"PROMPT TO GET SUGGESTIONS FOR IMPROVEMENT:\n"
                            f"{self.config.load_prompt_template(prompt_name=self.system_prompt, directory='prompts')}"
                        )
                    ]
                )
            else:
                forced_llm_output = default_llm_function.invoke(
                    [
                        SystemMessage(
                            content=self.config.load_prompt_template(
                                prompt_name="system_prompt_refiner",
                                directory='prompts'
                            )
                        ),
                        HumanMessage(
                            content=f"PROMPT TO GET SUGGESTIONS FOR IMPROVEMENT:\n"
                            f"{self.config.load_prompt_template(prompt_name=self.system_prompt, directory='prompts')}"
                        )
                    ]
                )
            smart_print(
                f"***** PROMPT SUGGESTIONS *****\n\033[33m{forced_llm_output.content}\033[0m\n*************",
                self.agent_name,
                "PROMPT SUGGESTIONS"
            )
        new_template = _visual_input(
            self.config.load_prompt_template(
                prompt_name=self.system_prompt,
                directory='prompts'
            ) if new_template is None else new_template
        )
        smart_print(
            f"***** NEW PROMPT TEMPLATE:\n{new_template}\n*************",
            self.agent_name,
            "NEW PROMPT TEMPLATE"
        )
        # Confirm that the user wants to modify the template
        confirm = smart_input(
            "Do you want to replace current prompt file template with your input? (y/n): ", self.agent_name).upper()
        # Save prompt with tag options
        if confirm == "Y":
            tag_option = smart_input(
                "Enter a tag for saving the prompt (leave blank for no tag, or 'same' to keep the current tag): ",
                self.agent_name)
            if tag_option.lower() == "same":
                save_prompt_with_tag(self.system_prompt, new_template, "")
            else:
                save_prompt_with_tag(self.system_prompt, new_template, tag_option)

            if callable_system_message:
                messages[0] = callable_system_message()
            else:
                messages[0].content = new_template
        return comments, forced_llm_output

    def change_premium_llm(self, premium_llm_function):
        llm_keys = list(self.llmORchains_list.keys())
        for i, key in enumerate(llm_keys):
            smart_print(f"{i}. {key}", self.agent_name)
        while True:
            new_llm_index = int(
                smart_input(f"Enter the number of the new premium LLM (0-{len(llm_keys) - 1}): ", self.agent_name))
            if 0 <= new_llm_index < len(llm_keys):
                new_llm_name = llm_keys[new_llm_index]
                if self.set_premium_llmORchain(new_llm_name):
                    break
        premium_llm_function = self.premium_llm
        smart_print(f"Premium LLM changed to {new_llm_name}", self.agent_name, "Change Premium LLM", optional=True)
        return premium_llm_function

    def change_default_llm(self, default_llm_function):
        llm_keys = list(self.llmORchains_list.keys())
        for i, key in enumerate(llm_keys):
            smart_print(f"{i}. {key}", self.agent_name)
        while True:
            new_llm_index = int(
                smart_input(f"Enter the number of the new default LLM (0-{len(llm_keys) - 1}): ", self.agent_name))
            if 0 <= new_llm_index < len(llm_keys):
                new_llm_name = llm_keys[new_llm_index]
                if self.set_default_llmORchain(new_llm_name):
                    break
        default_llm_function = self.default_llm
        smart_print(f"Default LLM changed to {new_llm_name}", self.agent_name, "Change Default LLM", optional=True)
        return default_llm_function

    def get_scored_results(self, function_name):
        self.configure_vector_store()
        menu = (
            "Do you want to see:\n"
            "[A] all MODIFIED/SCORED/COMMENTED results.\n"
            "[B] INPUT modified only.\n"
            "[C] OUTPUT modified only.\n"
            "[D] SCORED only.\n"
            "[E] COMMENTED only.\n"
            "[F] Success Tasks.\n"
            "[G] Failed Tasks.\n"
            "[H] user_message_few_shots.\n"
            "[I] Modify user_message_few_shots.\n"
            "Select your letter for choice or hit enter for all: "
        )
        confirm = smart_input(
            menu,
            self.agent_name,
            "BEFORE inference action MENU"
        ).upper()
        result = []

        if confirm == "F":
            tasks = self.config.get_learnt_tasks()
            task_list = "\n".join(tasks)
            _visual_input(task_list)
            return

        if confirm == "G":
            tasks = self.config.get_failed_tasks()
            task_list = "\n".join(tasks)
            _visual_input(task_list)
            return

        if confirm == "H":
            if self.user_message_few_shots:
                _visual_input(self.user_message_few_shots)
            else:
                print("No few_shots available.")
            return

        if confirm == "I":
            new_few_shots = _visual_input(self.user_message_few_shots)
            self.user_message_few_shots = new_few_shots
            envs_status = '\n'.join([env.get_state() for env in self.envs])

            return self.config.get_few_shot_examples(self.user_message_few_shots) + (
                f"\n- Current status of examples on "
                f"which the task will be tested on: {envs_status}\n"
            )

        if confirm in ["A", "", "B"]:
            result.extend(self.config.common_vectordb._query(
                query_text="*",
                metadata_filter={
                    "function_name": function_name,
                    "agent_name": self.agent_name,
                    "input_modified": True
                },
                k=100
            ))

        if confirm in ["A", "", "C"]:
            result.extend(self.config.common_vectordb._query(
                query_text="*",
                metadata_filter={
                    "function_name": function_name,
                    "agent_name": self.agent_name,
                    "output_modified": True
                },
                k=100
            ))

        if confirm in ["A", "", "D"]:
            result.extend(self.config.common_vectordb._query(
                query_text="*",
                metadata_filter={
                    "function_name": function_name,
                    "agent_name": self.agent_name,
                    "scored": True
                },
                k=100
            ))

        if confirm in ["A", "", "E"]:
            result.extend(self.config.common_vectordb._query(
                query_text="*",
                metadata_filter={
                    "function_name": function_name,
                    "agent_name": self.agent_name,
                    "commented": True
                },
                k=100
            ))

        if result:
            visual_result = "\n===============================\n".join(
                [
                    json.dumps(
                        json.loads(item.page_content),
                        indent=4,
                        sort_keys=True
                    ).replace("\\n", "\n") for item in result
                ]
            )
            _visual_input(visual_result, filetype="json")
        else:
            print("No results found for the selected option.")

    def get_previous_results(self, function_name=None, agent_name=None, k=100):
        self.configure_vector_store()
        metadata_filter = {}
        if function_name:
            metadata_filter["function_name"] = function_name
        if agent_name:
            metadata_filter["agent_name"] = agent_name
        result = self.config.common_vectordb._query(
            query_text="*",
            metadata_filter=metadata_filter,
            k=k
        )
        visual_result = "\n===============================\n".join(
            [
                json.dumps(json.loads(item.page_content), indent=4, sort_keys=True).replace("\\n", "\n")
                for item in result
            ]
        )
        return visual_result

    def post_inference(
        self,
        inference_result_msg,
        premium_llm_function,
        color="37",
        output_id=None,
        outputs_count=None,
        task_name=None
    ):
        action=""
        if not self.outputs:
            self.outputs = {}
            for i in range(outputs_count):
                self.outputs[i] = None
        if len(self.outputs) < outputs_count:
            for i in range(len(self.outputs), outputs_count):
                self.outputs[i] = None

        self.mode = 'after'
        comments, score = None, None
        nl = "\n"
        print(f"***{self.agent_name}, AFTER INFERENCE***")
        if inference_result_msg is None:
            # enable to request inference_result_msg.content to be None
            inference_result_msg = type('InferenceResult', (object,), {'content': None})

        while self.skip_rounds <= 0:
            self.temp_inference_result_content = None  # to capture the inference message done from functions outside of the menu
            # MENU
            multiple_ref = (f"OUTPUT \033[31m{output_id} OUT OF {outputs_count}\033[0m OUTPUTS" if (
                    output_id and outputs_count and (outputs_count > 1)) else "")

            # Run inference checks if any
            check_results = self.run_manage_inference_checks(output_id - 1, inference_result_msg.content)
            check_display = ""
            # Display inference check results
            for check_name, result in check_results.items():
                check_display += f"{nl}CHECK {check_name} result: " + str(result).replace("\\n", "\n")

            if not task_name:
                pattern = r'def\s+(\w+)\('
                match = re.search(pattern, inference_result_msg.content, flags=re.MULTILINE)
                task_name = match.group(1) if match else print("No function definitions found.")

            if not self.skip_log_entry_if_no_change:
                self.config.log_agent_data( self.agent_name,
                    "saved_task",
                    {
                        'llm_output': inference_result_msg.content,
                        'user_message': self.current_inference_context['input_contents'][1].content,
                        'num_parallel_inferences': self.num_parallel_inferences,
                        'task_parameters': self.task_parameters
                    }, before_after='after', user_id=self.get_user_id(), step_id=self.config.step_id, task_type="IR_CPS_TechSynthesis", task_id=True, function_name=task_name)
            menu = (
                f"\033[{self.print_color}m***** {self.agent_name}->{inspect.stack()[2].function} AFTER *****\nLLM ANSWER:\n{inference_result_msg.content}\n{check_display}\n***** {self.agent_name}->{inspect.stack()[2].function} AFTER *****\033[0m{multiple_ref}\n")

            menu += (
                "[A] Edit answer in VSCode\n")  # je voudrais le corriger uniquement pour demander une suggestion d'amélioration du prompt (d'un autre côté, je peux aussi le faire dans le menu précédent)
            menu += ("[B] Critic answer to regenerate it\n")
            menu += ("[C] Critic to improve agent's behavior\n")
            menu += ("[D] Evaluate answer\n")
            menu += ("[E] Go back (to BEFORE menu)\n")
            menu += ("[G] Skip for N rounds (auto mode)\n")
            menu += ("[Z] Continue\n")
            menu += ("[H] Exit\n")

            smart_print(menu, self.agent_name, "AFTER inference action MENU" + (
                f" {output_id}/{outputs_count}" if (output_id and outputs_count and (outputs_count > 1)) else ""),
                        self.agent_name, column_id=output_id - 1, column_max=outputs_count)
            print(f"***{self.agent_name}, AFTER INFERENCE, after self.smart_print menu***")
            self.menu_start_time = time.time()

            if self.automation:
                if (hasattr(self, 'recommend_critics') and self.recommend_critics) and self.outputs[output_id - 1] is None:
                    if comments is None:
                        comments = self.generate_instructions_feedback_fn( inference_result_msg.content, output_id=output_id)
                    inference_result_msg.content = self.apply_feedback_fn(
                        comments["suggestions"],
                        inference_result_msg.content,
                        text_has_annotations=False,
                        initial_prompt=self.llm_input_messages[0].content
                    )
                    self.outputs[output_id - 1] = inference_result_msg.content
                action = ""
            else:
                action = smart_input(
                    f"\n\033[32mAFTER\033[0m inference @ {self.agent_name}-> Choose an action (or hit Enter for inference) :",
                    self.agent_name, optional=False, column_id=output_id-1, column_max=outputs_count).upper()

            if action.endswith("NOT FOUND"):
                action="NOT FOUND"
                break
            # if modified async, it is important in case of edition ("A") to keep the modified content
            if self.temp_inference_result_content:
                inference_result_msg.content = self.temp_inference_result_content

            # ACTIONS processing
            self.start_time = time.time()  # Init action selected and timer to measure time spent and occurences in action processing

            if action == "A":  # Manually set/modify the answer/output
                self.modify_answer(inference_result_msg, output_id)

            elif action == "B":  # Critic this answer/output to get an improved answer/output
                inference_result_msg.content = self.apply_feedback_fn(
                    comments,
                    inference_result_msg.content,
                    text_has_annotations=False,
                    initial_prompt=self.llm_input_messages[0].content
                )

            elif action == "C":  # Find a better Prompt by providing critic and ideal answer
                comments = self.find_better_prompt(comments, inference_result_msg, premium_llm_function)

            elif action == "D":  # Evaluate & comment answer to re-use in prompts or later analysis
                comments, score = self.evaluate_comment_answer_for_later()

            elif action == "E":  # Go back BEFORE inference to improve system prompt or add information to user message
                inference_result_msg = self.go_back_inference(inference_result_msg)

            elif action == "G":  # Skip human actions for N rounds
                self.skip_iteration()

            elif action == "H":  # Exit program
                self.shutdown()

            # count time spent and occurrences waiting and in each option
            self.log_timing_and_call(action, mode='after')

            # check also that inference_result_msg is not of type str or int
            if self.temp_inference_result_content and not isinstance(inference_result_msg, str) and not isinstance(inference_result_msg, int):
                # inference_result_msg.content = f"{self.temp_inference_result_content}"
                smart_print("ANSWER MODIFIED, NEW CHECKS REQUIRED BEFORE CONTINUING", self.agent_name, "code_task_and_run_test SystemMessage", column_id=output_id-1, column_max=outputs_count)
                # check_results = self.run_inference_checks(output_id - 1, inference_result_msg.content)
            elif action in [None, "", "E", "Z"]:
                break  # E: Go back BEFORE inference to improve system prompt or add information to user message

            # proceed = smart_input("Continue 'y' (or 'n' to go back to menu) ? ", self.agent_name, column_id=output_id, column_max=outputs_count).lower()
            # if proceed in ["y", ""]:
            #     break
        if action!="NOT FOUND":
            if self.skip_rounds > 0:
                check_results = self.run_manage_inference_checks(output_id - 1, inference_result_msg.content)
                check_display = ""
                # Display inference check results
                for check_name, result in check_results.items():
                    check_display += f"{nl}CHECK {check_name} result: " + str(result).replace("\\n", "\n")

                smart_print(
                    f"\033[{self.print_color}m****{self.agent_name}>{inspect.stack()[2].function} LLM ANSWER content****\n{inference_result_msg.content}\n{check_display}\n*****************\033[0m",
                    self.agent_name, "LLM ANSWER content", column_id=output_id)
                self.skip_rounds -= 1
            else:
                smart_print(
                    f"Time spent in each option and occurrences: {self.after_inference_option_times} - {self.after_inference_option_counts}",
                    self.agent_name, optional=True, column_max=outputs_count)

        self.mode = None
        return inference_result_msg, comments, score

    def go_back_inference(self, inference_result_msg):
        inference_result_msg = -1  # break is set after action time measurement
        return inference_result_msg

    def evaluate_comment_answer_for_later(self):
        while True:
            score = float(smart_input(
                "Give a note for the result between 0.0 (worst) and 1.0 (top), or 0 for bad, 1 for good: ",
                self.agent_name))
            # if score is not between 0 and 1, then set to None and print error
            if score < 0 or score > 1:
                score = None
                print(f"\033[31mInvalid score: {score}\033[0m")
            else:
                break
        comments = smart_input("Comment on the result: ", self.agent_name)
        return comments, score

    def evaluate_comment_answer_for_later_2(self, comment, score, output_id=0, message=None):
        if comment:
            self.comments.append(comment)

        context = self.current_inference_context

        # Determine the output content
        if message is not None:
            output_content = message
        elif context.get('output_contents'):
            output_content = context['output_contents'][output_id] if isinstance(context['output_contents'], list) else \
                context['output_contents']
        else:
            output_content = None

        # Compute inference time
        start_time = context.get('start_time')
        end_time = datetime.now()
        inference_time = (end_time - start_time).total_seconds if start_time else None

        self._log_entry(
            function_name=context.get('function_name'),
            input_contents=context.get('input_contents'),
            output_contents=output_content,
            inference_time=inference_time,
            input_modified=context.get('input_modified'),
            skipped_inference=context.get('skipped_inference'),
            skip_rounds=self.skip_rounds,
            input_comments=context.get('input_comments'),
            output_comments=[comment],
            output_llm_raw=context.get('raw_llm_outputs')[output_id]
            if isinstance(context.get('raw_llm_outputs'), list)
            else context.get('raw_llm_outputs'),
            output_modified=context.get('output_modified'),
            user_score=score,
            message_tokens=context.get('message_tokens'),
            use_premium_llm=context.get('use_premium_llm'),
            call_duration=context.get('call_duration'),
            synthesize_mode=self.synthesize_mode
        )

    def find_better_prompt(self, comments, inference_result_msg, premium_llm_function):
        comments = smart_input(
            "First enter your critic here (then modify answer to get ideal answer): ",
            self.agent_name
        )
        ideal_answer = _visual_input(inference_result_msg.content)
        refine_prompt = f"Current system prompt:<<< {self.config.load_prompt_template(prompt_name=self.system_prompt, directory='prompts')} >>>\n\nPrompt's answer:<<< {inference_result_msg.content} >>>\n\nPrompt's answer critic:{comments}\n\nPrompt's ideal Answer:<<< {ideal_answer} >>>"
        smart_print(f"***** PROMPT FOR IMPROVEMENT *****\n{refine_prompt}", self.agent_name, "PROMPT FOR IMPROVEMENT")

        if premium_llm_function is None:
            premium_llm_function = self.premium_llm if self.premium_llm else None
        llm_output = premium_llm_function.invoke(
            [
                SystemMessage(
                    content=self.config.load_prompt_template(prompt_name="improve_prompt_from_answer_critic", directory='prompts')
                ),
                HumanMessage(content=refine_prompt)
            ]
        )
        smart_print(
            "***** RECOMMENDATION OPEN FOR EDITION *****\n",
            self.agent_name,
            "RECOMMENDATION OPEN FOR EDITION"
        )
        new_template = _visual_input(llm_output.content)
        smart_print(
            f"***** NEW PROMPT TEMPLATE:\n{new_template}\n*************",
            self.agent_name,
            "NEW PROMPT TEMPLATE"
        )  # Confirm that the user wants to modify the template
        confirm = smart_input(
            "Do you want to replace current prompt file template with your input? (y/n): ",
            self.agent_name
        ).upper()  # Save prompt with tag options

        if confirm == "Y":
            tag_option = smart_input(
                "Enter a tag for saving the prompt (leave blank for no tag, or 'same' to keep the current tag): ",
                self.agent_name)
            if tag_option.lower() == "same":
                save_prompt_with_tag(self.system_prompt, new_template, "")
            else:
                save_prompt_with_tag(self.system_prompt, new_template, tag_option)
        return comments

    def apply_feedback_fn(
        self,
        suggestions,
        text_content,
        text_has_annotations=True,
        annotation_format=None,
        initial_prompt=None,
        instruction_processing_approach='ANNOTATIONS_ALL'
    ):
        """
            This method processes the suggestions and text content to generate an improved answer.
            It can handle annotated critics to refine the text content based on the provided suggestions.

            Args:
                suggestions (str): The suggestions or critics to be applied to the text content.
                text_content (str or List[str]): The original text content that needs to be improved.
                text_has_annotations (bool): Indicates whether the text contains annotations. Default is True.
                annotation_format (str): The format of the annotations. If None and text_has_annotations is True,
                                        the format will be auto-detected.
            instruction_processing_approach (str): The approach for processing instructions. Possible values are
                                        'FULLTEXT_ALL', 'FULLTEXT_EACH', 'ANNOTATIONS_ALL', 'ANNOTATIONS_EACH'.
                                        Default is 'ANNOTATIONS_ALL'.

        Returns:
            str: The improved text content after applying the suggestions and critics.
        """
        # Combine text_content if it's a list
        if isinstance(text_content, list):
            combined_text_content = '\n'.join(text_content)
        elif isinstance(text_content, str):
            combined_text_content = text_content
        else:
            raise TypeError("text_content must be a string or a list of strings.")

        # Auto-detect annotation format if necessary
        def detect_annotation_format(text):
            patterns = {
                'latex-inline': re.compile(
                    r'\\(?P<tag>\w+)(\[(?P<instruction>[^\]]*)\])?\{(?P<content>.*?)\}',
                    re.DOTALL),
                'HTML-inline': re.compile(
                    r'<(?P<tag>\w+)(\s+instruction="(?P<instruction>[^"]*)")?>\s*(?P<content>.*?)\s*</(?P=tag)>',
                    re.DOTALL),
                'latex-id': re.compile(r'\[(?P<id>\d+)\]\{(?P<content>.*?)\}', re.DOTALL),
                'HTML-id': re.compile(r'<(?P<id>\d+)>(?P<content>.*?)</(?P=id)>', re.DOTALL)
            }
            for fmt, pattern in patterns.items():
                if pattern.search(text):
                    return fmt
            return None

        # Parse annotations from text
        def parse_annotations(text, annotation_format):
            annotations = []
            if annotation_format == 'latex-inline':
                pattern = re.compile(r'\\(?P<tag>\w+)(\[(?P<instruction>[^\]]*)\])?\{(?P<content>.*?)\}', re.DOTALL)
            elif annotation_format == 'HTML-inline':
                pattern = re.compile(
                    r'<(?P<tag>\w+)(\s+instruction="(?P<instruction>[^"]*)")?>\s*(?P<content>.*?)\s*</(?P=tag)>',
                    re.DOTALL)
            elif annotation_format == 'latex-id':
                pattern = re.compile(r'\[(?P<id>\d+)\]\{(?P<content>.*?)\}', re.DOTALL)
            elif annotation_format == 'HTML-id':
                pattern = re.compile(r'<(?P<id>\d+)>(?P<content>.*?)</(?P=id)>', re.DOTALL)
            else:
                return annotations
            for match in pattern.finditer(text):
                annotation = match.groupdict()
                annotation['full_match'] = match.group(0)
                annotation['start'] = match.start()
                annotation['end'] = match.end()
                annotations.append(annotation)
            return annotations

        # Parse instructions from suggestions
        def parse_instructions(suggestions):
            instructions = {}
            if suggestions:
                pattern = re.compile(r'\[(?P<id>\d+)\]:\s*(?P<instruction>.+)')
                for line in suggestions.strip().splitlines():
                    match = pattern.match(line.strip())
                    if match:
                        id_ = match.group('id')
                        instruction = match.group('instruction').strip()
                        instructions[id_] = instruction
            return instructions

        # Get suggestions from the last inference check results if exist and suggestions is empty / log critic to agent data in any case
        critic = None
        if self.inference_tracking.last_inference_check_results:
            for result in self.inference_tracking.last_inference_check_results:
                if result is not None and isinstance(result, dict):
                    for key, value in result.items():
                        if key == 'Recommend Critics':
                            critic = value
                            break
                else:
                    break

            if critic:
                if suggestions == "":
                    suggestions = critic['suggestions']
                self.config.log_agent_data(
                    self.agent_name,
                    "llm_suggestions",
                    {
                        'llm_suggestions': critic['suggestions'],
                        'user_suggestions': suggestions,
                        'llm_suggestions_prompt': critic['improvement_prompt']
                    }
                )

        if text_has_annotations:
            if annotation_format is None:
                annotation_format = detect_annotation_format(combined_text_content)
                if annotation_format is None:
                    raise ValueError("Could not auto-detect annotation format.")
            annotations = parse_annotations(combined_text_content, annotation_format)
            if annotation_format in ['latex-inline', 'HTML-inline']:
                # Instructions are inline within annotations
                for annotation in annotations:
                    instruction = annotation.get('instruction', '').strip()
                    annotation['instruction'] = instruction
            elif annotation_format in ['latex-id', 'HTML-id']:
                # Instructions are provided in suggestions
                instructions = parse_instructions(suggestions)
                for annotation in annotations:
                    id_ = annotation.get('id')
                    instruction = instructions.get(id_)
                    if instruction:
                        annotation['instruction'] = instruction
                    else:
                        raise ValueError(f"No instruction found for annotation ID {id_}")
            else:
                raise ValueError("Unsupported annotation format.")

            if instruction_processing_approach == 'FULLTEXT_ALL':
                system_prompt = """
                Your task is to improve the following text by processing the annotations and following the instructions provided.
                Please replace the annotated parts according to the instructions and produce the final improved version of the text.

                Typical actions:
                1. **FIX:** Make necessary corrections.
                2. **IMPROVE:** Enhance the content.
                3. **INSERT:** Add new content as instructed.
                """
                user_prompt = f"{combined_text_content}"
                llm_output = secure_invoke(self.premium_llm, 
                    [SystemMessage(content=system_prompt.strip()), HumanMessage(content=user_prompt)], temperature=self.temperature_min)
                combined_text_content = llm_output.content

            elif instruction_processing_approach == 'FULLTEXT_EACH':
                for annotation in annotations:
                    system_prompt = """
                Your task is to improve the following text by processing the annotation and following the instruction.
                Please replace the annotated part according to the instruction and produce the final improved version of the text.
                """
                    # Replace other annotations with their content
                    temp_text = combined_text_content
                    for other_annotation in annotations:
                        if other_annotation != annotation:
                            temp_text = temp_text.replace(other_annotation['full_match'], other_annotation['content'])
                    llm_output = secure_invoke(self.premium_llm, 
                        [SystemMessage(content=system_prompt.strip()), HumanMessage(content=temp_text)], temperature=self.temperature_min)
                    combined_text_content = llm_output.content

            elif instruction_processing_approach == 'ANNOTATIONS_ALL':
                annotations_data = {annotation.get('id') or str(i): {
                    'content': annotation['content'],
                    'instruction': annotation['instruction']
                } for i, annotation in enumerate(annotations)}
                system_prompt = """
                Your task is to generate an upgraded content for each annotated content following the instructions.
                Provide your output as a JSON dictionary mapping IDs to the new content.
                The values should be strings containing the new content, without additional keys or nesting.
                Example Output: {"1": "new content for annotation 1", "2": "new content for annotation 2"}
                """
                user_prompt = f"Annotations:\n{json.dumps(annotations_data)}"
                llm_output = secure_invoke(self.premium_llm, 
                        [SystemMessage(content=system_prompt.strip()), HumanMessage(content=user_prompt)], temperature=self.temperature_min)
                llm_response = llm_output.content.strip()
                try:
                    new_contents = json.loads(llm_response)
                except json.JSONDecodeError:
                    raise ValueError(f"Invalid JSON response from LLM: {llm_response}")

                for annotation in annotations:
                    id_ = annotation.get('id') or str(annotations.index(annotation))
                    full_match = annotation['full_match']
                    new_content = new_contents.get(id_)
                    if new_content:
                        # Ensure new_content is a string
                        if not isinstance(new_content, str):
                            raise ValueError(f"New content for annotation ID {id_} is not a string.")
                        combined_text_content = combined_text_content.replace(full_match, new_content)
                    else:
                        raise ValueError(f"No new content found for annotation ID {id_}")

            elif instruction_processing_approach == 'ANNOTATIONS_EACH':
                for annotation in annotations:
                    system_prompt = """
                Your task is to generate new content for the following text according to the instruction.
                """
                    user_prompt = f"Text: {annotation['content']}\nInstruction: {annotation['instruction']}"
                    llm_output = self.premium_llm.invoke(
                        [SystemMessage(content=system_prompt.strip()), HumanMessage(content=user_prompt)])
                    new_content = llm_output.content.strip()
                    full_match = annotation['full_match']
                    combined_text_content = combined_text_content.replace(full_match, new_content)
            else:
                raise ValueError("Unsupported instruction processing approach.")
        if suggestions != '':
            # Existing logic for non-annotated text
            if suggestions is None:
                suggestions = smart_input(
                    "Provide critic/feedback/request: ",
                    self.agent_name,
                    column_max=self.num_parallel_inferences
                )
            temp_prev_sugg, _ = self.config.get_agent_data(self.agent_name, "llm_suggestions")
            prev_sugg = ""
            for i in temp_prev_sugg:
                if not i.get('feedback_applied', True):
                    continue
                llm_suggestion = i.get('llm_suggestions', '')
                user_suggestion = i.get('user_suggestions', '')
                diff = difflib.unified_diff(llm_suggestion.splitlines(), user_suggestion.splitlines(), lineterm='')
                diff_text = '\n'.join(diff)
                prev_sugg += f"#{diff_text}#\n"
            prompt_sugg = ""
            if suggestions:
                prompt_sugg = (
                    "Your task is also to improve answer by applying the **SUGGESTIONS** to modify the answer accordingly."
                    f"\n### SUGGESTIONS TO BE APPLIED: << {suggestions} >>\n"
                    "See also some **PREVIOUS SUGGESTIONS** (not to be applied) to help you better understand SUGGESTIONS TO BE APPLIED.\n" if prev_sugg != "" else ""
                    f"\n### PREVIOUS SUGGESTIONS: << {prev_sugg} >>" if prev_sugg != "" else ""
                )
            system_prompt = f"""Given the INSTRUCTIONS provided by the user (and the **INITIAL PROMPT**), your task is to generate a much better answer than INITIAL ANSWER.
            {prompt_sugg}
            ### INITIAL PROMPT: << {initial_prompt} >>
            ### INITIAL ANSWER: << {combined_text_content} >> """
            user_prompt = f"INSTRUCTIONS: << {suggestions} >>"
            llm_output = self.premium_llm.invoke(
                [SystemMessage(content=system_prompt.strip()), HumanMessage(content=user_prompt)])
            combined_text_content = llm_output.content
            

        self.temp_inference_result_content = combined_text_content  # Capture the result
        return combined_text_content

    def modify_answer(self, inference_result_msg, column_id=None):
        inference_result_msg.content = _visual_input(inference_result_msg.content)
        self.temp_inference_result_content = inference_result_msg.content
        smart_print(f"***** ANSWER:\n{inference_result_msg.content}\n*************", self.agent_name, "NEW ANSWER", optional=True, column_id=column_id, column_max=self.num_parallel_inferences)
        return inference_result_msg.content

    def update_answer(self, answer, column_id=None):
        self.temp_inference_result_content = answer
        if column_id:
            self.logger.info("IMPORTANT: column_id not yet implemented - Updating current HumanLLM for agent")
        return answer

    def get_host_id(self):
        return socket.gethostname() + "-" + str(uuid.getnode())

    def invoke_with_function_call(
        self,
        llm_function,
        messages,
        function_call=None,
        function_list=None,
        max_calls=5
    ):
        # test if HumanLLM.function_list exists
        if function_list is None:
            if self.function_list is None:
                function_list = [{
                    "name": "search_for_external_knwoledge",
                    "description": "Call this function to search for external knowledge when the model's confidence is low or its information might be too outdated",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "description": {"type": "string",
                                            "description": "precise description of the information to be provided"},
                            "url": {"type": "string",
                                    "description": "Google search url link find this information (ie. it should start by https://www.google.com/search?q= )"},
                        },
                        "required": ["description", "url"],
                    }
                }]
            else:
                function_list = self.function_list

        calls = 0
        while True and calls < max_calls:
            output = llm_function(messages=messages, functions=function_list, function_call="auto")
            calls += 1
            if output.content is not None and len(output.content) > 0 and output.content != "''":
                break
            # test if 1 more function_call is requested
            elif output.additional_kwargs is not None:
                additional_kwargs = output.additional_kwargs
                function_call_info = additional_kwargs.get('function_call', {})
                function_name = function_call_info.get('name', '')
                function_args_str = function_call_info.get('arguments', '')
                function_args = json.loads(function_args_str)
                if function_name in globals():
                    # Call the function with the provided arguments
                    result = globals()[function_name](**function_args)
                else:
                    result = smart_input(
                        f"Uknown function: {function_name} -- please provide result manually:\n",
                        "invoke_with_function_call"
                    )
                messages.append(FunctionMessage(content=result, name=function_name, arguments=function_args_str))

        return output

    def _log_entry(
        self,
        function_name,
        input_contents,
        output_contents,
        input_modified=False,
        skipped_inference=False,
        input_comments=None,
        output_comments=None,
        output_llm_raw=None,
        output_modified=False,
        inference_time=None,
        user_score=None,
        message_tokens=None,
        score=None,
        use_premium_llm=False,
        call_duration=None,
        skip_rounds=None,
        synthesize_mode=False,
        pipeline_mode=False
    ):
        # Check if we should skip logging when nothing has been modified
        if self.skip_log_entry_if_no_change and not input_modified and not output_modified and not (input_comments or (output_comments and (output_comments[0] if len(output_comments)>0 else True))): return

        entry = {
            "input_contents": input_contents,
            "output_contents": output_contents,
            "output_llm_raw": output_llm_raw,
            "input_comments": input_comments,
            "output_comments": output_comments,
            "inference_time": inference_time,
            "score": score,
            "skip_rounds": skip_rounds,
            "message_tokens": message_tokens,
            "before_inference_option_times": self.before_inference_option_times,
            "before_inference_option_counts": self.before_inference_option_counts,
            "after_inference_option_times": self.after_inference_option_times,
            "after_inference_option_counts": self.after_inference_option_counts,
            "call_duration": call_duration,
            "synthesize_mode": synthesize_mode,
            "pipeline_mode": pipeline_mode,
            "user_score": user_score
        }
        # Serialize the entry as a JSON string
        serialized_entry = json.dumps(entry, default=lambda o: o.__dict__ if hasattr(o, '__dict__') else str(o))

        # Log entry into the common vector database with tags
        tags = {
            "time": datetime.now().isoformat(),
            "host": self.get_host_id(),
            "step_id": self.config.step_id,
            "input_modified": input_modified,
            "output_modified": output_modified,
            "system_prompt": self.system_prompt,
            "agent_name": self.agent_name,
            "function_name": function_name,
            "skipped_inference": skipped_inference,
            "skip_rounds": skip_rounds,
            "pipeline_mode": pipeline_mode,
            "use_premium_llm": use_premium_llm,
            "commented": (input_comments is not None or output_comments is not None),
            "scored": (score is not None),
        }
        # add to tags every key of self.before_inference_option_times with count and time, if count > 0
        for key in self.before_inference_option_times:
            if self.before_inference_option_counts[key] > 0:
                tags[f"b{key}_time"] = self.before_inference_option_times[key]
                tags[f"b{key}_count"] = self.before_inference_option_counts[key]
        # add to tags every key of self.after_inference_option_times with count and time, if count > 0
        for key in self.after_inference_option_times:
            if self.after_inference_option_counts[key] > 0:
                tags[f"a{key}_time"] = self.after_inference_option_times[key]
                tags[f"a{key}_count"] = self.after_inference_option_counts[key]

        self.configure_vector_store()

        self.config.common_vectordb._add_texts(
            texts=[serialized_entry],
            metadatas=[tags]
        )

        try:
            raw = entry.get("output_llm_raw")
            final = entry.get("output_contents")

            # normaliser (liste/str) -> str
            def _to_str_raw(x):
                if isinstance(x, list):
                    return "\n\n---RAW OUTPUTS---\n\n".join([s if isinstance(s, str) else str(s) for s in x])
                return x if isinstance(x, str) else ("" if x is None else str(x))
            def _to_str_final(x):
                if isinstance(x, list):
                    def _extract(o):
                        if isinstance(o, dict) and "content" in o: return o["content"]
                        return getattr(o, "content", str(o))
                    return "\n\n---FINAL OUTPUTS---\n\n".join([_extract(o) for o in x])
                return x if isinstance(x, str) else ("" if x is None else str(x))

            raw_s = _to_str_raw(raw)
            final_s = _to_str_final(final)
            if raw_s != final_s:
                diff_text = "\n".join(difflib.unified_diff(
                    raw_s.splitlines(), final_s.splitlines(),
                    fromfile="generated", tofile="final", lineterm=""
                ))

                self.config.log_agent_data(
                    self.agent_name,
                    "answer_diffs",
                    {
                        "time": datetime.now().isoformat(),
                        "function_name": function_name,
                        "diff": diff_text,
                        "output_modified": entry.get("output_modified"),
                        "user_score": entry.get("user_score"),
                    }
                )
                # H2: forward human/agent diffs as priority feedback
                try:
                    if getattr(self, 'dynamic_mgr', None):
                        who = "user" if entry.get("output_modified") else "agent"
                        # Skip empty diffs
                        if diff_text.strip():
                            self.dynamic_mgr.log_user_correction(self.agent_name, diff_text, who)
                except Exception:
                    self.logger.debug("Could not forward diff to dynamic_mgr", exc_info=True)
        except Exception as e:
            self.logger.exception(f"Failed to compute/save answer diff: {e}")
        return True

    def process_output(self, llm_output, counter, llm_outputs, init_skip_rounds, task_name=None):
        """Traite un seul LLM output (séquentiellement ou en parallèle)."""
        self.logger.info(f"Processing LLM output {counter} out of {len(llm_outputs)}")
        if len(llm_outputs) > 1:
            self.skip_rounds = init_skip_rounds
            smart_print(
                f"ANSWER NUMBER #{counter-1} ",
                self.agent_name,
                "POST INFERENCE",
                append=True,
                optional=True
            )

        self.current_inference_context = {
            'function_name': inspect.stack()[1].function,
            'input_contents': self.llm_input_messages,
            'output_contents': None,  # Remplir après traitement
            'start_time': datetime.now(),
            'input_modified': False,
            'skipped_inference': None,
            'input_comments': None,
            'output_comments': None,
            'raw_llm_outputs': None,
            'output_modified': None,
            'message_tokens': None,
            'use_premium_llm': False
        }
        # Post-inference human intervention (traitement standard après une inférence)
        output_messages_instance, output_comments_instance, score_instance = self.post_inference(
            llm_output,
            premium_llm_function=None,
            output_id=counter,
            outputs_count=len(llm_outputs),
            task_name=task_name
        )
        return output_messages_instance, output_comments_instance, score_instance

    def invoke(
        self,
        original_input_messages=None,
        default_llm_function=None,
        premium_llm_function=None,
        callable_system_message=None,
        system_prompt_template=None,
        user_message=None,
        return_message_content_only=True,
        function_calling=False,
        temperature_min=None,
        timeout_seconds=300,
        stream_output=False,
        use_default_llm=True,
        model_choice=None,
        temperature_max=None,
        task_name=None,
        prompt_directory="prompts",
        generation_technique='temperature_variation',
        forced_llm_output=None,
        selection_technique=None,
        **kwargs
    ):
        """
        This method can perform different multi-inference strategies depending on 
        'generation_technique'. When self.num_parallel_inferences > 1, it can:
          - 'temperature_variation'
          - 'self_refinement'
          - 'iterative_alternatives'
          or fallback to the original concurrency-based parallel calls (default).
        """
        current_kwargs = locals()
        if temperature_min is None: temperature_min = self.temperature_min
        if temperature_max is None: temperature_max = self.temperature_max

        # Helper function to actually make a single LLM call, streaming or not.
        def perform_llm_call( input_msg, use_premium, func_calling, temperature=None, stream_output=True, color_id=None):
            """
            Wraps invocation logic for a single call. Adjusts temperature if supplied.
            Handles partial streaming via smart_print.
            """
            # if use_premium is a string, get func from self.llmORchains_list using key, if not exist raise erro
            if isinstance(use_premium, str):
                if use_premium in self.llmORchains_list:
                    func = self.llmORchains_list[use_premium]
                else:
                    raise ValueError(f"Premium LLM function '{use_premium}' not found in llmORchains_list.")
            elif use_premium:
                func = premium_llm_function if not func_calling else self.invoke_with_function_call
            else:
                func = default_llm_function if not func_calling else self.invoke_with_function_call

            # Override temperature if provided
            if temperature or temperature == 0:
                func = func.with_config(configurable={"llm_temperature": temperature})
                print(f"Temperature set to {temperature}")
            else:
                print(f"No temperature value, not set for model {func.model_name}")

            if stream_output:
                # Choose color for streaming text if multiple inferences
                if color_id is None or color_id <= 0:
                    start_color, end_color = "", ""
                else:
                    # Using the 7 ANSI colors in round-robin
                    color_pal = ["\033[91m", "\033[92m", "\033[93m", "\033[94m", "\033[95m", "\033[96m", "\033[97m"]
                    start_color, end_color = color_pal[color_id % 7], "\033[0m"

                final_output = ""
                smart_print("", self.agent_name, "Inference streaming output")
                previous_chunk_str = ""
                json_trail_re = re.compile(r'[\'\}\]]$')

                buffer = ""
                buffer_start_time = time.time()
                flush_interval = 5.0  # seconds

                for chunk in func.stream(input_msg):
                    if hasattr(chunk, 'content'):
                        chunk_content = chunk.content
                        final_output += chunk_content
                    else:
                        # Fallback if chunk is not a usual "AIMessage" chunk
                        current_chunk_str = str(chunk)
                        # Attempt to trim any trailing bracket/brace that might break JSON
                        while json_trail_re.search(current_chunk_str):
                            current_chunk_str = current_chunk_str[:-1]
                        new_part_index = len(previous_chunk_str)
                        chunk_content = current_chunk_str[new_part_index:]
                        previous_chunk_str = current_chunk_str
                        final_output += chunk_content

                    buffer += chunk_content
                    current_time = time.time()
                    time_elapsed = current_time - buffer_start_time

                    # Check if buffer should be flushed
                    should_flush = False
                    delimiter_pos = -1
                    delimiter_length = 0

                    # If we haven't flushed for a while, flush everything
                    if time_elapsed >= flush_interval:
                        should_flush = True
                        delimiter_pos = len(buffer)
                    else:
                        # If there's enough text plus a new line, flush up to that delimiter
                        if ('\n\n' in buffer or '<br>' in buffer) and len(buffer) >= 100:
                            pos_newline = buffer.rfind('\n\n')
                            pos_br = buffer.rfind('<br>')
                            if pos_newline > pos_br:
                                delimiter_pos = pos_newline
                                delimiter_length = 2
                            else:
                                delimiter_pos = pos_br
                                delimiter_length = 4
                            if delimiter_pos != -1:
                                should_flush = True

                    if should_flush:
                        if delimiter_pos == len(buffer):
                            # Time-based flush: entire buffer
                            to_send = buffer
                            buffer = ""
                        elif delimiter_pos != -1:
                            # Delimiter-based flush: up to the last delimiter
                            to_send = buffer[:delimiter_pos + delimiter_length]
                            buffer = buffer[delimiter_pos + delimiter_length:]
                        else:
                            # No delimiter => flush entire buffer
                            to_send = buffer
                            buffer = ""

                        # Websocket vs console output
                        if self.config.use_websocket:
                            smart_print( to_send, self.agent_name, f"Inference streaming output {color_id}", append=True, column_id=color_id, column_max=self.num_parallel_inferences)
                        else:
                            smart_print( start_color + to_send + end_color, self.agent_name, f"Inference streaming output {color_id}", append=True)

                        buffer_start_time = current_time

                # Final flush
                if buffer:
                    if self.config.use_websocket:
                        smart_print( buffer, self.agent_name, f"Inference streaming output {color_id}", append=True, column_id=color_id, column_max=self.num_parallel_inferences)
                    else:
                        smart_print( start_color + buffer + end_color, self.agent_name, f"Inference streaming output {color_id}", append=True)

                return AIMessage(content=final_output)

            else:
                model_name = func.model_name
                dynamic_config = self.dynamic_llm_config
                agent_name = self.agent_name
                # No streaming. Normal call
                try:
                    value = func.invoke(input_msg)
                except BadRequestError as e:
                    if  e.param == "temperature":
                        self.logger.error(f"Error invoking LLM with temperature {temperature}: - ERROR:{e}")
                        # Fallback to no temperature
                        func = func.with_config(configurable={"llm_temperature": None})
                        value = func.invoke(input_msg)
                    else:
                        self.logger.error(f"Error invoking LLM: - ERROR:{e}")
                        raise e
                return AIMessage(content=value.content if hasattr(value, 'content') else str(value))

        smart_print(
            f"\033[{self.print_color}m****{self.agent_name}>{inspect.stack()[1].function} calling HumanLLM****\033[0m",
            self.agent_name,
            "HumanLLM",
            optional=True
        )

        system_prompt_template = kwargs.pop("system_prompt_override", system_prompt_template)
        if system_prompt_template:
            self.system_prompt = system_prompt_template

        if default_llm_function is None:
            default_llm_function = self.default_llm if use_default_llm else self.premium_llm
        if premium_llm_function is None:
            premium_llm_function = self.premium_llm if self.premium_llm else None

        self.logger.info(f"****agent : {self.agent_name}, automation : {self.automation}****")

        # Possibly override system_prompt from saved_task if automation=before, etc.
        if self.automation == 'before' and (hasattr(self, "saved_task")):
            temp = self.saved_task.get('content', {})
            self.logger.info(f"****temp (prompt before {self.agent_name}) : {temp}****")
            self.system_prompt = temp.get('prompt', "")
            if self.auto_n_rounds > 0:
                self.automation = "full_auto"
            else:
                self.automation = None
            original_input_messages = [
                SystemMessage(content=self.system_prompt),
                HumanMessage(content=user_message)
            ]
        elif original_input_messages is None or len(original_input_messages) == 0:
            # Standard usage if no special automation
            self.logger.info(f"****user_message {self.agent_name} : {user_message}****")
            original_input_messages = [
                SystemMessage(
                    content=self.config.load_prompt_template(prompt_name=self.system_prompt, directory=prompt_directory)
                ),
                HumanMessage(content=user_message)
            ]


        input_contents_str0 = str(original_input_messages[0].content)
        input_contents_str1 = str(original_input_messages[1].content)

        self.before_inference_option_times = {'TOTAL': 0, 'SELECTION': 0}
        self.before_inference_option_counts = {'TOTAL': 0, 'SELECTION': 0}
        self.after_inference_option_times = {'TOTAL': 0, 'SELECTION': 0}
        self.after_inference_option_counts = {'TOTAL': 0, 'SELECTION': 0}
        self.unidentified_option_times = {'TOTAL': 0, 'SELECTION': 0}
        self.unidentified_option_counts = {'TOTAL': 0, 'SELECTION': 0}
        self.mode = None
        call_start_time = time.time()
        
        # Build initial context for dynamic config
        current_kwargs["num_parallel_inferences"] = self.num_parallel_inferences
        eval_context = {
            'agent_name': self.agent_name,
            'function_name': inspect.stack()[1].function,
            'user_message': user_message or (original_input_messages[1].content if len(original_input_messages) > 1 else ''),
            'system_prompt': system_prompt_template or self.system_prompt,
            'phase': 'pre_inference',
            'num_parallel_inferences': self.num_parallel_inferences,
            'temperature_min': temperature_min or self.temperature_min,
            'temperature_max': temperature_max or self.temperature_max,
            'kwargs' : current_kwargs
        }

        while True:
            if self.skip_rounds > 0:
                smart_print(
                    f"\033[{self.print_color}m****{self.agent_name}>{inspect.stack()[2].function} skipping HumanLLM for {self.skip_rounds} rounds****\033[0m",
                    self.agent_name,
                    "Skipping round",
                    optional=True
                )

            start_time = datetime.now()

            # Apply pre-inference dynamic configuration
            if hasattr(self, 'dynamic_mgr') and self.dynamic_mgr:
                mods = self.dynamic_mgr.evaluate_triggers(eval_context, phase='pre_inference')
                if mods:
                    self._apply_modifications(mods, eval_context, phase='pre_inference')
        

                    # Reflect potential message edits from dynamic mods into local variables
                    user_message = eval_context.get('user_message', user_message)
                    if eval_context.get('system_prompt'):
                        self.system_prompt = eval_context['system_prompt']
            # === H2: record minimal outcome metrics for offline optimization ===
            try:
                if getattr(self, "dynamic_mgr", None):
                    self.dynamic_mgr.record_outcome(
                        context=eval_context,
                        modifications=eval_context.get("dynamic_modifications") or {},
                        outcome_metrics={"inference_time": eval_context.get("inference_time"),
                                            "n_outputs": len(llm_outputs) if llm_outputs else 0,
                                            "phase": "post_inference"}
                    )
            except Exception:
                self.logger.debug("record_outcome failed", exc_info=True)
                
                
            # Automation short-circuits
            if self.automation in ['before', 'after', 'skip_once']:
                # Possibly skip or read from saved_task ...
                input_comments, skip_inference, use_premium_llm, llm_outputs = None, False, False, []
                llm_input_messages = original_input_messages
                self.llm_input_messages = original_input_messages
                self.inference_tracking.last_inference_check_results = [None]

                if self.automation == 'after' and hasattr(self, "saved_task"):
                    # e.g. saved LLM output
                    temp = self.saved_task.get('content', {})
                    self.logger.info(f"****temp (llm_output after {self.agent_name}) : {temp}****")
                    llm_outputs = [AIMessage(content=temp.get('llm_output', ""))]
                    if self.auto_n_rounds > 0:
                        self.automation = "full_auto"
                    else:
                        self.automation = None
                    self.logger.info(f"****llm_output : {llm_outputs}****")
                    smart_print(
                        llm_outputs[0].content if llm_outputs else "No LLM output",
                        self.agent_name,
                        "NEW inference result received",
                        column_id=0,
                        column_max=1
                    )
            else:
                # Normal pre_inference flow
                llm_input_messages, input_comments, skip_inference, use_premium_llm, \
                    default_llm_function, premium_llm_function, function_calling = self.pre_inference(
                        original_input_messages,
                        default_llm_function,
                        premium_llm_function,
                        function_calling,
                        callable_system_message,
                        model_choice=model_choice,
                        task_name=task_name,
                        forced_llm_output=forced_llm_output
                    )

                self.llm_input_messages = llm_input_messages
                self.clear_selected_outputs()
                # Preallocate check results
                self.inference_tracking.last_inference_check_results = [None] * self.num_parallel_inferences

                if llm_input_messages and not skip_inference:
                    # If a generation_technique is specified AND we have multiple inferences,
                    # we use our new approach. Otherwise, fallback to the original concurrency-based approach.
                    if generation_technique and self.num_parallel_inferences > 1:
                        # figure out system vs user for generate_candidates
                        system_prompt_used = llm_input_messages[0].content
                        user_prompt_used = llm_input_messages[1].content

                        llm_outputs = self.generate_candidates(
                            generation_technique=generation_technique,
                            system_prompt=system_prompt_used,
                            user_prompt=user_prompt_used,
                            num_responses=self.num_parallel_inferences,
                            use_premium_llm=self.use_premium_llm or use_premium_llm,
                            function_calling=function_calling,
                            temp_min=temperature_min,
                            temp_max=temperature_max,
                            stream_output=stream_output
                        )
                        # Show them in console or UI
                        for idx, candidate in enumerate(llm_outputs):
                            if self.config.use_websocket:
                                smart_print(
                                    candidate.content,
                                    self.agent_name, 
                                    "NEW inference result received", 
                                    column_id=idx,
                                    column_max=self.num_parallel_inferences
                                )
                            else:
                                smart_print(
                                    f'\033[0m**** New inference result #{idx+1} ****\n{candidate.content}\n**** END ****\033[0m',
                                    self.agent_name,
                                    "NEW inference result",
                                    column_id=idx,
                                    column_max=self.num_parallel_inferences
                                )
                    else:
                        # Original concurrency approach:
                        outputs = []
                        with concurrent.futures.ThreadPoolExecutor(
                            max_workers=self.num_parallel_inferences
                        ) as executor:
                            # streaming could be on if chain is used
                            if isinstance(
                                (self.premium_llm if use_premium_llm else self.default_llm),
                                type(self.llmORchains_list.get('3_majority_chain'))
                            ):
                                stream_output = True
                                
                            futures = [
                                executor.submit(
                                    perform_llm_call,
                                    llm_input_messages,
                                    self.use_premium_llm or use_premium_llm,
                                    function_calling,
                                    (   (temperature_min + i * (temperature_max - temperature_min) / (self.num_parallel_inferences - 1))
                                        if (temperature_min is not None and self.num_parallel_inferences > 1 and temperature_min >= 0.)
                                        else None),
                                    stream_output,
                                    i
                                )
                                for i in range(self.num_parallel_inferences)
                            ]
                            for idx, future in enumerate(futures):
                                try:
                                    llm_response = future.result(timeout=timeout_seconds)
                                    outputs.append(llm_response)
                                    if self.config.use_websocket:
                                        smart_print( llm_response.content, self.agent_name,  "NEW inference result received", column_id=idx, column_max=self.num_parallel_inferences)
                                    else:
                                        smart_print( f'\033[0m**** New inference result received and added to outputs as #{len(outputs)}\033[0m:\n{llm_response.content}\n\033[9mEND OF #{len(outputs)}****\033[0m', self.agent_name, "NEW inference result received", column_id=idx, column_max=self.num_parallel_inferences )
                                except concurrent.futures.TimeoutError:
                                    smart_print( 'A task ran longer than the allotted timeout and was cancelled.', self.agent_name, "Inference result TIMEOUT" )
                                except Exception as exc:
                                    smart_print( f'Generated an exception: {exc}', self.agent_name, "Inference result EXCEPTION" )
                            concurrent.futures.wait(futures)

                        # Check how many we got
                        if len(outputs) == 0:
                            smart_print( '**** No inference result received, set output to None', self.agent_name, "NO inference received")
                            llm_outputs = None
                        elif len(outputs) == 1:
                            llm_outputs = outputs
                        else:
                            # Possibly synthesize
                            if self.synthesize_mode and len(outputs) > 1:
                                synthesized_response = self.synthesize_responses( [output.content for output in outputs], use_default_llm)
                                llm_outputs = [AIMessage(content=synthesized_response.content)]
                                smart_print( f'**** {len(outputs)} inference results received, THEN SYNTHETISED to 1', self.agent_name, "MULTIPLE to 1 SYNTHESIS" )
                            else:
                                smart_print( f'**** {len(outputs)} inference results received - selecting keepers below.', self.agent_name, "MULTIPLE inferences received")
                                llm_outputs = outputs

                    # Update context for post-inference evaluation
                    eval_context['llm_outputs'] = llm_outputs
                    eval_context['phase'] = 'post_inference'
                    eval_context['end_time'] = datetime.now()
                    eval_context['inference_time'] = (eval_context['end_time'] - start_time).total_seconds()
                    
                    # Apply post-inference dynamic configuration
                    if hasattr(self, 'dynamic_mgr') and self.dynamic_mgr:
                        mods = self.dynamic_mgr.evaluate_triggers(eval_context, phase='post_inference')
                        if mods:
                            self._apply_modifications(mods, eval_context, phase='post_inference')
                    
                    # Update llm_outputs from context in case they were modified
                    llm_outputs = eval_context.get('llm_outputs', llm_outputs)

                else:
                    # Skip LLM inference
                    skip_inference_str = str(skip_inference) if not isinstance(skip_inference, str) else skip_inference
                    llm_outputs = [AIMessage(content=skip_inference_str)]

            end_time = datetime.now()

            raw_llm_outputs = (
                [output.content for output in llm_outputs] 
                if isinstance(llm_outputs, list) else None
            )

            output_messages, output_comments, score = [], [], []
            if llm_outputs:
                init_skip_rounds = self.skip_rounds
                if len(llm_outputs) > 1:
                    smart_print( "**** Multiple LLM ANSWERS > process POST-INFERENCE for each ****", self.agent_name, "Multiple LLM ANSWERS", append=True, optional=True)

                # Sequentially handle each inference’s post-processing
                for counter, llm_output in enumerate(llm_outputs, start=1):
                    msg, comm, sc = self.process_output(
                        llm_output,
                        counter,
                        llm_outputs,
                        init_skip_rounds,
                        task_name
                    )
                    output_messages.append(msg)
                    if msg == -1:
                        break
                    output_comments.append(comm)
                    score.append(sc)

                # If any message returned -1 => user did "redo" => revert input
                if any(msg == -1 for msg in output_messages):
                    original_input_messages[0].content = input_contents_str0
                    original_input_messages[1].content = input_contents_str1
                else:
                    # Normal exit
                    break
            elif self.automation == 'skip_once' and hasattr(self, "saved_task"):
                if self.auto_n_rounds > 0:
                    self.automation = "full_auto"
                else:
                    self.automation = None
                input_content = ""
                if 'input_contents' in self.saved_task.get('content', {}):
                    input_content = self.saved_task.get('content', {}).get('input_contents', "")
                output_messages = [AIMessage(content=input_content)]
                break


        if self.auto_n_rounds:
            if self.auto_n_rounds > 0:
                self.auto_n_rounds -= 1
            if not self.auto_n_rounds:
                self.automation = None

        if selection_technique or self.selection_technique:
            # If we are in fusion mode, we need to merge the outputs
            output_messages = self.select_candidate(output_messages, selection_technique or self.selection_technique)

        caller_function_name = inspect.stack()[1].function
        call_duration = time.time() - call_start_time

        # Logging 
        self._log_entry(
            function_name=caller_function_name,
            input_contents=llm_input_messages,
            output_contents=output_messages,
            inference_time=(end_time - start_time).total_seconds(),
            input_modified=((llm_input_messages[0].content + "\n" + llm_input_messages[1].content) != (input_contents_str0 + "\n" + input_contents_str1)),
            skipped_inference=True if skip_inference else False,
            skip_rounds=self.skip_rounds,
            input_comments=input_comments,
            output_comments=output_comments,
            output_llm_raw=raw_llm_outputs,
            output_modified=any(
                o.content != r for o, r in zip(output_messages, raw_llm_outputs or [])
            ),
            score=score,
            message_tokens=None,
            use_premium_llm=use_premium_llm,
            call_duration=call_duration,
            synthesize_mode=self.synthesize_mode
        )

        # H2: persist per-run outcome for offline learning
        try:
            if getattr(self, 'dynamic_mgr', None):
                outcome_metrics = {
                    "scores": score,
                    "inference_time": (end_time - start_time).total_seconds() if 'end_time' in locals() else None
                }
                self.dynamic_mgr.record_outcome(
                    eval_context, eval_context.get('dynamic_modifications', {}), outcome_metrics
                )
        except Exception:
            self.logger.debug("Could not record outcome to dynamic_mgr", exc_info=True)

        # Update usage tracker with actual costs/tokens if available
        if hasattr(self, 'usage_tracker') and 'dynamic_modifications' in eval_context:
            for help_type in eval_context['dynamic_modifications']:
                # This would need actual cost/token calculation
                self.usage_tracker.record_usage(help_type, cost=0, tokens=0, success=True)

        # Restore attributes overridden by dynamic config back to their original values
        if 'original_values' in eval_context:
            for attr, orig_val in eval_context['original_values'].items():
                setattr(self, attr, orig_val)
        return ([msg.content for msg in output_messages] if return_message_content_only else output_messages)

    def generate_candidates(
        self,
        generation_technique,
        system_prompt,
        user_prompt,
        num_responses,
        use_premium_llm,
        function_calling,
        temp_min,
        temp_max,
        stream_output,
        experts_list=None,
        multi_llm=None
    ):
        """
        Generates multiple candidates based on different strategies.
        Returns a list of AIMessage objects.
        """
        # Helper function to make a single LLM call
        def generate_single(system_prompt, user_prompt, use_premium_llm, function_calling, temperature=None, stream_output=False, color_id=0):
            """Single pass call wrapper."""
            input_messages = [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
            
            # Determine which LLM to use
            if isinstance(use_premium_llm, str):
                if use_premium_llm in self.llmORchains_list:
                    func = self.llmORchains_list[use_premium_llm]
                else:
                    raise ValueError(f"LLM '{use_premium_llm}' not found in llmORchains_list.")
            elif use_premium_llm:
                func = self.premium_llm
            else:
                func = self.default_llm
                
            # Override temperature if provided
            if temperature is not None:
                func = func.with_config(configurable={"llm_temperature": temperature})
                
            # Invoke the LLM
            # try, if error due to temperature not supported by model, re-run without temperature
            try:
                model_name = func.model_name
                dynamic_config = self.dynamic_llm_config
                agent_name = self.agent_name
                value = func.invoke(input_messages)
            except BadRequestError as e:
                if  e.param == "temperature":
                    self.logger.error(f"Error invoking LLM with temperature {temperature}: - ERROR:{e}")
                    # Fallback to no temperature
                    func = func.with_config(configurable={"llm_temperature": None})
                    value = func.invoke(input_messages)
                else:
                    self.logger.error(f"Error invoking LLM: - ERROR:{e}")
                    raise e
            return AIMessage(content=value.content if hasattr(value, 'content') else str(value))
        
        candidates = []
        
        if num_responses < 1:
            num_responses = 1
            
        if generation_technique == "temperature_variation":
            if temp_min is None: temp_min = 0.
            if temp_max is None: temp_max = 1.0
            temperatures = [round(temp_max - i * (temp_max - temp_min) / max(1, num_responses - 1), 2) for i in range(num_responses)]
            for i, temp in enumerate(temperatures):
                candidate = generate_single(system_prompt, user_prompt, use_premium_llm, function_calling, temperature=temp, stream_output=stream_output, color_id=i)
                candidates.append(candidate)
                
        elif generation_technique == "self_refinement":
            for i in range(num_responses):
                if not candidates:
                    current_prompt = system_prompt
                else:
                    current_prompt = f"{system_prompt}\nRefine the previous ANSWER to improve final answer to the user's prompt. SOLUTION: <<<\n{candidates[-1].content}\n>>>"
                candidate = generate_single(current_prompt, user_prompt, use_premium_llm, function_calling, temperature=0.0, stream_output=stream_output, color_id=0)
                candidates.append(candidate)
                
        elif generation_technique == "multi_experts":
            experts = []
            if isinstance(experts_list, list) and all(isinstance(e, str) for e in experts_list):
                while len(experts) < num_responses:
                    experts.append(experts_list[len(experts) % len(experts_list)])
            else:
                default_experts = ["Algorithm Expert", "Performance Optimizer", "Out of the box problem solver", "AI Engineer", "Compiler Specialist"]
                while len(experts) < num_responses:
                    experts.append(default_experts[len(experts) % len(default_experts)])
                    
            for i, expert in enumerate(experts[:num_responses]):
                meta_prompt = f"You are a `{expert}`\n{system_prompt}"
                candidate = generate_single(meta_prompt, user_prompt, use_premium_llm, function_calling, temperature=0.0, stream_output=stream_output, color_id=i)
                candidates.append(candidate)
                
        elif generation_technique in ["mixture_of_agents_generation", "moa", "multi_llm"]:
            llm_or_chain_keys = getattr(self, "multi_llm", None) or multi_llm or list(self.llmORchains_list.keys())
            for i in range(num_responses):
                llm_or_chain_key = llm_or_chain_keys[i % len(llm_or_chain_keys)]
                candidate = generate_single(system_prompt, user_prompt, llm_or_chain_key, function_calling, temperature=temp_min, stream_output=stream_output, color_id=i)
                candidates.append(candidate)
                
        elif generation_technique == "iterative_alternatives":
            for i in range(num_responses):
                if not candidates:
                    current_prompt = system_prompt
                else:
                    previous_solutions = "\n".join(f"SOLUTION {idx + 1}: <<<\n{cand.content}\n>>>" for idx, cand in enumerate(candidates))
                    current_prompt = f"{system_prompt}\nGiven the following solutions, propose a new alternative optimal solution to user's prompt:\n{previous_solutions}\n"
                candidate = generate_single(current_prompt, user_prompt, use_premium_llm, function_calling, temperature=0.0, stream_output=stream_output, color_id=i)
                candidates.append(candidate)
        else:
            raise ValueError(f"Invalid generation_technique: {generation_technique}")
            
        return candidates

    def select_candidate(self, output_messages, selection_technique=None):
        """
        Merges multiple LLM outputs/candidates into a single according to selection_technique (concat, best_of_n, last)
        """
        if selection_technique is None:
            selection_technique = self.selection_technique

        # Handle edge cases
        if not output_messages:
            return []
        if len(output_messages) == 1:
            return output_messages

        if selection_technique == "concat":
            # Concatenate all outputs
            return [AIMessage(content="\n".join([msg.content for msg in output_messages]))]

        elif selection_technique in ["moa", "mixture_of_agents"]:
            # NEW: Mixture of Agents selection from OptoPrimeMulti
            try:
                system_prompt = (
                    "You are an expert at synthesizing multiple CANDIDATE solutions into a single OPTIMAL solution. "
                    "Given the following CANDIDATE solutions to a PROBLEM, provide an OPTIMAL solution that mixes the best elements of each, and follows the same answer output structure.\n"
                    f"Initial PROBLEM: <<<\n{self.llm_input_messages[0].content}\n{self.llm_input_messages[1].content}\n>>>\n\n"
                )
                
                user_prompt = "\n\n".join([
                    f"CANDIDATE solution {i + 1}:\n{output_messages[i].content}\n"
                    for i in range(len(output_messages))
                ]) + "\n\nOPTIMAL solution (mixing best elements):\n"
                
                response = self.premium_llm.invoke([
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=user_prompt)
                ])
                # strip <<< and >>> if present
                cleaned_response = response.content.strip().removeprefix("<<<").removesuffix(">>>").strip()
                return [AIMessage(content=cleaned_response)]
            except Exception as e:
                smart_print(f"MOA selection failed: {e}. Falling back to best_of_n.", 
                           self.agent_name, "select_candidate", optional=True)
                return self.select_candidate(output_messages, "best_of_n")
                
        elif selection_technique == "majority":
            # try:
            if True:
                import numpy as np
                from scipy.spatial.distance import pdist, squareform
                from sklearn.cluster import AgglomerativeClustering
                texts = [m.content for m in output_messages]
                # build distance matrix = 1 - similarity
                n = len(texts)
                D = np.zeros((n, n))
                for i in range(n):
                    for j in range(i + 1, n):
                        sim = calculate_text_similarity(texts[i], texts[j])
                        dist = 1 - sim
                        D[i, j] = D[j, i] = dist
                # cluster by a threshold to group similar answers
                try:
                    clu = AgglomerativeClustering( n_clusters=None, metric="precomputed", linkage="average", distance_threshold=0.5).fit(D) # new sklearn version >= 1.4
                except TypeError:
                    clu = AgglomerativeClustering( n_clusters=None, affinity="precomputed", linkage="average", distance_threshold=0.5).fit(D) # old sklearn version
                
                labels = clu.labels_
                # find largest cluster
                from collections import Counter
                top = Counter(labels).most_common(1)[0][0]
                idxs = [i for i,l in enumerate(labels) if l==top]
                subD = D[np.ix_(idxs, idxs)]
                # medoid = index with minimum total distance
                medoid = idxs[int(np.argmin(subD.sum(axis=1)))]
                return [output_messages[medoid]]
            # except Exception as e:
            #     return [output_messages[-1]]

        elif selection_technique == "last":
            # Return the last output
            return [output_messages[-1]]
        elif selection_technique == "best_of_n":
            if len(output_messages) == 1:
                return output_messages
            try:
                system_prompt = (
                    f"Given the following CANDIDATE solutions and the initial PROBLEM, provide the best solution by returning its content, following the same answer output structure.\n"
                    f"Initial PROBLEM: <<<\n{self.llm_input_messages[0].content}\n{self.llm_input_messages[1].content}\n>>>\n\n"
                )
                user_prompt = "\n\n".join([
                    f"Solution CANDIDATE {i + 1}:\n{output_messages[i].content}\n"
                    for i in range(len(output_messages))
                ]) + "\n\nBest CANDIDATE:\n"
                
                response = self.premium_llm.invoke([
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=user_prompt)
                ])
                return [AIMessage(content=response.content)]
            except Exception as e:
                smart_print(f"Best_of_n selection failed: {e}. Falling back to last.", 
                           self.agent_name, "select_candidate", optional=True)
                return [output_messages[-1]]            
        else:
            raise ValueError(f"Invalid selection_technique: {selection_technique}. Supported options: 'concat', 'best_of_n', 'last'.")

    def generate_annotations_feedback_fn(
        self,
        inference_result_content: str = None,
        output_id: int = None,
        annotation_types: str = "FIX, DELETE, APPROVE",
        annotation_number: int = 5,
        generation_technique: str = "temperature_variation",
        selection_technique: str = "best_of_n",
        num_candidates: int = 1
    ):
        """
        Generate span-based annotations for improving inference output.
        
        Args:
            inference_result_content: The content to annotate
            output_id: ID of the output being annotated
            annotation_types: Types of annotations to generate
            annotation_number: Number of annotations to generate
            generation_technique: Technique for generating multiple candidates
            selection_technique: Technique for selecting final output
            num_candidates: Number of candidates to generate (>1 enables ensembling)
            
        Returns:
            Dict containing annotations and related metadata
        """
        # Prepare annotation generation prompt
        annotation_generate_prompt = f"""
You're an AI assistant. Your task is to generate annotations on the prompt given to you. The output should be exactly the same as the input but with some annotations in it, no changes on the text itself. The annotations will have this format:
'\\{annotation_types}[the feedback for the text (what is wrong, what is right, etc.)]{{the text to annotate}}'

You should not change anything of the content of the given prompt, only add annotations. You have to add {annotation_number} annotations, and for each one of them don't place them randomly, but place them in a way that they are relevant to the text.
Try to give real feedbacks for the annotations, and not just random feedbacks. Finally, don't annotate the same text twice or the full text in one; place annotations on phrases or keywords that are relevant.

PROMPT:<<<{self.system_prompt}>>>

ANSWER:<<<{inference_result_content}>>>

Previous Annotations: {self._get_previous_annotations()}

List your annotations below:
"""
        
        # Generate annotations using generate_candidates (handles both single and multiple)
        system_prompt = "You are tasked with generating annotations to improve model outputs."
        candidates = self.generate_candidates(
            generation_technique=generation_technique,
            system_prompt=system_prompt,
            user_prompt=annotation_generate_prompt,
            num_responses=num_candidates,
            use_premium_llm=True,
            function_calling=False,
            temp_min=self.temperature_min,
            temp_max=self.temperature_max if num_candidates > 1 else self.temperature_min,
            stream_output=False
        )
        
        # Select best candidate if multiple
        if num_candidates > 1:
            selected = self.select_candidate(candidates, selection_technique)
            annotations = selected[0].content if selected else ""
        else:
            annotations = candidates[0].content if candidates else ""
        
        # Clean up annotations
        annotations = regex.sub(r'[^\P{C}\t\n\r]', '', annotations)
        annotations = re.sub(r'\\u[0-9A-Fa-f]{4}', '', annotations)
        annotations = annotations.split("ANSWER:<<<")[-1]
        annotations = annotations.split(">>>")[0]
        # Save annotations
        self.config.log_agent_data(
            self.agent_name,
            "llm_annotations",
            {
                'annotations': annotations,
                'annotation_prompt': annotation_generate_prompt,
                'annotation_types': annotation_types,
                'annotation_number': annotation_number,
                'generation_technique': generation_technique if num_candidates > 1 else None,
                'selection_technique': selection_technique if num_candidates > 1 else None,
                'num_candidates': num_candidates
            }
        )
        
        return {
            "output_id": output_id,
            "annotations": annotations,
            "annotation_prompt": annotation_generate_prompt,
            "annotation_types": annotation_types,
            "num_candidates": num_candidates
        }
    
    def _get_previous_annotations(self):
        """Helper to retrieve previous annotations."""
        previous_annotations, _ = self.config.get_agent_data(self.agent_name, "llm_annotations")
        prev_annotations = ""
        if previous_annotations:
            for ann in previous_annotations[-3:]:  # Last 3 annotations
                prev_annotations += f"\n{ann.get('annotations', '')}"
        return prev_annotations

    def generate_instructions_feedback_fn(
        self,
        inference_result_content=None,
        output_id=None,
        generation_technique: str = "temperature_variation",
        selection_technique: str = "best_of_n",
        num_candidates: int = 1):
        """
        Uses a premium LLM to generate top suggestions or critiques for improving
        the inference output, identified by output_id. If there are no check results,
        it requests general improvement suggestions based on the inference result content.

        Now supports ensemble generation methods for improved feedback quality.

        :param output_id: The ID of the output message to critique.
        :param inference_result_content: The actual content of the inference result to be critiqued.
        :param generation_technique: Technique for generating multiple candidates
        :param selection_technique: Technique for selecting final output
        :param num_candidates: Number of candidates to generate (>1 enables ensembling)
        :return: A dictionary containing improvement suggestions and annotations.
        """

        import re
        import json

        # Run checks on the inference content if available
        improvement_feedback = []
        check_results = []
        for inference_check in self.inference_tracking.last_inference_check_results:
            if inference_check:
                check_results += [inference_check]
                break

        # Collect insights from various checks, focusing on quality-related results
        if check_results:
            for check_result in check_results:
                for check_name, result in check_result.items():
                    if isinstance(result, str) and result:  # Include only meaningful, non-empty results
                        improvement_feedback.append(f"Feedback from {check_name}: {result}")
                    elif isinstance(result, list) and result:
                        improvement_feedback.extend([f"{check_name} feedback: {item}" for item in result if item])

        previous_suggestions, _ = self.config.get_agent_data(self.agent_name, "llm_suggestions")
        prev_sugg = ""
        prev_sugg_u = ""
        if previous_suggestions:
            for sugg in previous_suggestions:
                prev_sugg += f"\n{sugg.get('llm_suggestions', '')}"
                prev_sugg_u += f"\n{sugg.get('user_suggestions', '')}"

        # Prepare a prompt based on whether feedback is available
        if improvement_feedback:
            feedback_prompt = "\n".join(improvement_feedback)
            improvement_prompt = (
                "Based on the feedback below, generate concise, actionable suggestions "
                "to improve or correct the ANSWER. Focus on clarity, accuracy, and style improvements. "
                "\n\n"
                f"PROMPT:<<<{self.system_prompt}>>>\n\n"
                f"ANSWER:<<<{inference_result_content}>>>\n\n"
                f"Feedback:<<<{feedback_prompt}>>>\n\n"
                f"Previous LLM Suggestions:<<<{prev_sugg}>>>\n\n"
                f"Previous User Suggestions:<<<{prev_sugg_u}>>>\n\n"
                "Provide your improvement suggestions below."
            )
        else:
            improvement_prompt = (
                "Provide improvement suggestions to enhance the clarity, accuracy, and quality of the following ANSWER. "
                f"PROMPT:<<<{self.system_prompt}>>>\n\n"
                f"ANSWER:<<<{inference_result_content}>>>\n\n"
                f"Previous LLM Suggestions:<<<{prev_sugg}>>>\n\n"
                f"Previous User Suggestions:<<<{prev_sugg_u}>>>\n\n"
                "List your improvement suggestions below."
            )

        # Use generate_candidates for both single and multiple inference
        system_prompt = "You are tasked with analyzing feedback to improve model outputs."
        candidates = self.generate_candidates(
            generation_technique=generation_technique,
            system_prompt=system_prompt,
            user_prompt=improvement_prompt,
            num_responses=num_candidates,
            use_premium_llm=True,
            function_calling=False,
            temp_min=self.temperature_min,
            temp_max=self.temperature_max if num_candidates > 1 else self.temperature_min,
            stream_output=False)
        
        # Select best candidate if multiple
        if num_candidates > 1:
            selected = self.select_candidate(candidates, selection_technique)
            response = selected[0] if selected else AIMessage(content="")
        else:
            response = candidates[0] if candidates else AIMessage(content="")

        # Clean up the response content
        # response.content = re.sub(r'[^\x20-\x7E\t\n\r]', '', response.content)
        # improvement_prompt = re.sub(r'[^\x20-\x7E\t\n\r]', '', improvement_prompt)

        # Allow printable Unicode characters, remove only control characters
        response.content = regex.sub(r'[^\P{C}\t\n\r]', '', response.content)
        improvement_prompt = regex.sub(r'[^\P{C}\t\n\r]', '', improvement_prompt)
     
        response.content = re.sub(r'\\u[0-9A-Fa-f]{4}', '', response.content)
        improvement_prompt = re.sub(r'\\u[0-9A-Fa-f]{4}', '', improvement_prompt)

        # Save the suggestions using add_agent_data
        self.config.log_agent_data(self.agent_name, "llm_suggestions", {
            'llm_suggestions': response.content,
            'user_suggestions': "",
            'improvement_prompt': improvement_prompt,
            'generation_technique': generation_technique if num_candidates > 1 else None,
            'selection_technique': selection_technique if num_candidates > 1 else None,
            'num_candidates': num_candidates,
            'feedback_applied': False,
        })

        # Prepare the return value
        ret = {
            "output_id": output_id,
            "suggestions": response.content,
            "improvement_prompt": improvement_prompt,
        }

        smart_print(
            json.dumps(ret),
            self.agent_name,
            "CRITIC SUGGESTIONS",
            column_id=output_id,
            optional=False
        )
        return ret

    def get_tasks(self, page_size: int = 200, nb_pages: int = 1, id_last_task: Optional[str] = None):
        return self.config.get_tasks(page_size, nb_pages, id_last_task)
    
    def goto_task(self, task_id: str, automatic: str = None, special_criteria: dict = None, task_details: str = None):
        return self.config.goto_task(task_id, automatic, special_criteria, task_details)
    
    def parse_ai_generated_code(
        self,
        message,
        language="py",
        retry=3,
        required_bot_arg=None,
        task_definition=None,
        automatic_tests=False,
        output_id=None
    ):
        # test if self.parsed_code is already set
        if not hasattr(self, "parsed_code"):
            self.parsed_code = {}
        # Convert text to dictionary
        try:
            result_dict = ast.literal_eval(message)
        except Exception as e:
            result_dict = None
        # if result_dict is a dictionary and code exists, change message to result_dict["code"]
        if isinstance(result_dict, dict) and "MainFunction" in result_dict:
            code = ""
            if "HelperFunctions" in result_dict:
                for helper_function in result_dict["HelperFunctions"]:
                    code += helper_function["code"] + "\n"
            code += result_dict["MainFunction"]["code"]
        else:
            code = None
        error = None
        while retry > 0:
            try:
                if language == "py":  # Python caset
                    if code is None:
                        # Match Python code blocks
                        code_pattern = re.compile(r"```python(.*?)(```|$)", re.DOTALL)
                        codes_match = [match[0].strip() for match in code_pattern.findall(message)]
                        code = "\n".join(codes_match) if codes_match else message

                    parsed = ast.parse(code)
                    functions = []
                    imports = []
                    classes = []
                    runnable_code = ""

                    function_calls = set()  # To track all functions that are being called
                    function_defs = {}  # To track all function definitions
                    last_function = None  # To store the last defined function

                    if len(code) == 0 or len(list(parsed.body)) == 0:
                        return False, f"Error parsing action response (No Code found): {parsed.body}"

                    # Use ast.walk to go through all nodes in the AST
                    for node in ast.walk(parsed):
                        if isinstance(node, ast.FunctionDef):
                            node_type = "FunctionDef"
                            function_info = {
                                "name": node.name,
                                "type": node_type,
                                "body": ast.get_source_segment(code, node),
                                "params": [arg.arg for arg in node.args.args],
                            }
                            functions.append(function_info)
                            function_defs[node.name] = function_info  # Store function definition
                            last_function = function_info  # Track the last defined function

                        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                            # Record function calls by name
                            function_calls.add(node.func.id)

                        elif isinstance(node, ast.ClassDef):
                            node_type = "ClassDef"
                            class_info = {
                                "name": node.name + ".run_pipeline",
                                "type": node_type,
                                "body": ast.get_source_segment(code, node),
                            }
                            classes.append(class_info)
                            functions.append(class_info)  # Append class definition as a function as per original logic

                        elif isinstance(node, (ast.Expr, ast.Expression, ast.Assign)):
                            node_type = "Expression"
                            runnable_code += "\n" + ast.get_source_segment(code, node)

                        elif isinstance(node, (ast.ImportFrom, ast.Import)):
                            node_type = "ImportFrom"
                            imports.append(ast.get_source_segment(code, node))

                    # Now we need to find the main function as per the new logic:
                    # 1. Identify the root function (not called by any other function).
                    # 2. If no root exists, fallback to the last defined function.

                    # Identify root function (not called by any other function)
                    root_function = None
                    for function_name, function_info in function_defs.items():
                        if function_name not in function_calls:
                            root_function = function_info
                            break

                    # If no root function is found, use the last defined function
                    main_function = root_function if root_function else last_function

                    # Ensure we have a main function
                    assert main_function is not None, "No main function found."

                    # Check if required_bot_arg is present in the main function's parameters
                    if required_bot_arg:
                        assert required_bot_arg in main_function["params"], f"Main function {main_function['name']} must take an argument named '{required_bot_arg}'"

                    # Assemble the final program code
                    program_code = "\n".join(imports) + "\n"
                    program_code += "\n\n".join(function["body"] for function in functions)

                    tests_pattern = re.compile(r'\n#\s+document #([a-z0-9-]+)\s+.*test[^\n]*?\n([^\n]+)|{.*?\"documentid\":\s*\"#(.*?)\",\s*\"FunctionCall\":\s*\"([^\"]+)\".*?}', re.IGNORECASE)
                    matches = tests_pattern.findall(task_definition if task_definition is not None else self.last_user_message)

                    if matches and not automatic_tests:
                        tests = [match[:2] if match[0] != "" else match[2:] for match in matches]
                    else:
                        self.logger.info("Running default tests with bot argument to main function")
                        tests = [(env.id, main_function["name"] + "(bot)") for env in self.envs]

                    for doc_id, test in tests:
                        try:
                            parsed_test = ast.parse(test)
                        except Exception as e:
                            return False, f"Error parsing code of Tests:\nERROR: {e}\nCODE: {test}"
                        # check if the test is a function call
                        if not isinstance(parsed_test.body[0], ast.Expr):
                            return False, f"Error parsing code of Tests (not a function call): {test}"
                else:
                    raise ValueError(f"Unsupported language in this version: {language}")

                self.parsed_code[output_id] = {
                    "program_code": program_code,
                    "main_function": main_function,
                    "runnable_code": runnable_code,
                    "tests": tests,
                }
                return True, self.parsed_code[output_id]

            except Exception as e:
                retry -= 1
                error = e
                time.sleep(0.1)

        self.parsed_code[output_id] = f"Error parsing action response (before program execution): {error}"
        smart_print(
            f"CODE PARSING ERROR!!!\n{error}",
            self.agent_name,
            "code_task_and_run_test SystemMessage",
            optional=False,
            column_id=output_id
        )
        return False, self.parsed_code[output_id]
    
    def run_tests_on_code(
        self,
        message,
        parsed_code=None,
        skip_already_processed=False,
        output_id=None,
        restore_state=True,
        custom_agent=None
    ):
        # Retrieve error_patches from HumanLLMMonitor
        metadata = {'step_id': HumanLLMConfig().step_id}
        error_patches, _ = HumanLLMConfig().get_agent_data(self.agent_name, 'error_patches', metadata_filter=metadata)
        error_patches = error_patches if error_patches else []

        primitives = get_primitives(self.primitives_dir)
        #parsed_code = getattr(self, 'parsed_code', None) if parsed_code is None else parsed_code
        parsed_code = self.parsed_code.get(output_id, None) if (parsed_code is None and isinstance(self.parsed_code, dict)) else parsed_code
        current_skip_rounds = self.skip_rounds  # save the initial value to align it for code validation

        if isinstance(parsed_code, dict) and parsed_code["program_code"] not in self.processed_codes:
            self.processed_codes.add(parsed_code["program_code"])
        # This logic speedup because the same code should have the same score BUT only on the same problem & state
        elif skip_already_processed or not isinstance(parsed_code, dict):
            return None

        # Set initial state before running tests or runnable code
        no_runtime_errors, exec_results = [], []
        # insert content of config.py into the code to ensure that the OPENAI_API_KEY is set
        with open("config.py", "r") as f:
            common_code = f.read() + "\n"
        # Common code part to be executed in all cases
        common_code += "\n".join(primitives) + "\n"

        # Run the code & tests in each environment
        max_autofix, decision_lower = None, None
        if hasattr(self, 'max_autofix'):
            max_autofix = self.max_autofix
        # Start the timer before the environments loop
        start_time = time.time()  # Added line
        for idx, env in enumerate(self.envs):
            env.backup_state()
            # If it's the first environment and the user chose not to fix, exit the loop
            if idx > 0 and decision_lower in ("no", "n", ""):
                smart_print(
                    f"SKIPPING TEST: code error on first env, skipping test {idx}",
                    custom_agent or getattr(self, 'agent_name', 'unknown agent name'),
                    "code_task_and_run_test SystemMessage",
                    optional=False,
                    column_id=output_id
                )
            else:
                # Determine tests to run or set default runnable code
                matching_tests = [
                    test
                    for doc_id, test in parsed_code["tests"]
                    if doc_id == env.id
                ] if parsed_code["tests"] else [parsed_code['runnable_code']]
                if not matching_tests:
                    no_runtime_error = False
                    exec_result = (
                        f"Error: no test found for given id {env.id}"
                        if parsed_code["tests"]
                        else f"Error: no runnable code found nor tests"
                    )
                else:
                    # Concatenate common code with program and tests or runnable code
                    code_to_run = common_code + parsed_code["program_code"] + "\n" + "\n".join(matching_tests)
                    smart_print(
                        "TESTING GENERATED CODE.....",
                        custom_agent if custom_agent else self.agent_name,
                        "code_task_and_run_test SystemMessage",
                        append=True,
                        optional=False,
                        column_id=output_id,
                        column_max=self.num_parallel_inferences
                    )
                    
                    t1 = time.time()
                    no_runtime_error, exec_result, std_out_err = env.step(code_to_run)
                    # Smart print the std_out_err
                    smart_print(
                        f"FUNCTION DISPLAY OUTPUTS:\n{std_out_err}",
                        custom_agent if custom_agent else self.agent_name,
                        "code_task_and_run_test SystemMessage",
                        append=True,
                        optional=False,
                        column_id=output_id
                    )
                    # Show score
                    scores = self.generate_score(idx, no_runtime_error, env.get_score(), time.time() - t1)
                    smart_print(
                        scores,
                        custom_agent if custom_agent else self.agent_name,
                        "Scores",
                        append=True,
                        optional=False,
                        column_id=output_id,
                        column_max=self.num_parallel_inferences
                    )

                    if no_runtime_error:
                        smart_print(
                            "TEST SUCCESSFUL",
                            custom_agent if custom_agent else self.agent_name,
                            "code_task_and_run_test SystemMessage",
                            optional=False,
                            column_id=output_id
                        )
                        smart_print(
                            env.get_state(),
                            custom_agent if custom_agent else self.agent_name,
                            "CODE_RESULT",
                            optional=False,
                            column_id=output_id
                        )
                    while not no_runtime_error and current_skip_rounds <= 0:
                        smart_print(
                            "\033[31mCODE ERROR\033[0m: " + exec_result,
                            custom_agent if custom_agent else self.agent_name,
                            "code_task_and_run_test SystemMessage",
                            optional=False,
                            column_id=output_id
                        )
                        if self.automation and max_autofix is not None:
                            decision = "a" if max_autofix > 1 else "no"
                            if decision == "a":
                                max_autofix -= 1
                        elif self.automation:
                            decision = "n"
                        else:
                            if output_id == None:
                                output_id = 0
                            smart_print(
                                parsed_code["program_code"],
                                custom_agent if custom_agent else self.agent_name,
                                f"Inference streaming output {output_id}",
                                append=True,
                                column_id=output_id,
                                column_max=self.num_parallel_inferences
                            )
                            decision = smart_input(
                                f"ANSWER {output_id} Do you want to edit the code to fix the error (you will also be requested first) ? (yes/no) or try autofix by LLM (a): ",
                                custom_agent if custom_agent else self.agent_name,
                                "fix_error",
                                column_id=output_id
                            ).strip()
                        decision_lower = decision.lower()
                        if decision_lower in ("no", "n", ""):
                            break
                        elif decision_lower in ["y", "yes"]:
                            # if in websocket, then get from self.human_llm_code_task.premium_llm
                            if HumanLLMConfig().use_websocket:
                                edited_code = self.temp_inference_result_content
                            else:
                                edited_code = _visual_input(
                                    parsed_code["program_code"],
                                    filetype="py",
                                    message_type="fix_error",
                                    agent_name=self.name,
                                    column_id=output_id
                                )
                        else:
                            if decision_lower == "a":
                                # Do not use HumanLLMMonitor because no template is available for this specific case
                                smart_print(
                                    "TRYING TO AUTOFIX ERROR",
                                    custom_agent if custom_agent else self.agent_name,
                                    "fix_error",
                                    optional=True,
                                    column_id=output_id
                                )
                                instructions = ""
                            else:
                                # User provided custom instructions
                                instructions = decision
                            fix_system_prompt = f"""You are a Python expert in code debugging.
                            You are provided with ERROR MESSAGE and the CODE TO FIX.
                            {instructions}
                            Reply with the full Python code fixed and ready to be executed without the triple quotes and python tags. You add comments in the code to explain your fix.
                            """
                            smart_print(
                                "ANALYZING ERROR.....",
                                custom_agent if custom_agent else self.agent_name,
                                "code_task_and_run_test SystemMessage",
                                optional=False,
                                column_id=output_id
                            )
                            help_for_fixing_system_prompt = f"""You help an LLM to fix code errors which has no access to documentation or internet by extracting key code information from the INFORMATION/DOCUMENTATION provided given CODE TO FIX and ERROR MESSAGE."""
                            error_with_info_to_help_prompt = f"ERROR MESSAGE:<<\n{exec_result}\n>>\n\nCODE TO FIX:<<\n{parsed_code['program_code']}\n>>\nINFORMATION/DOCUMENTATION:<<\n{self.last_user_message}\n>>"
                            help_code_returned = self.premium_llm.invoke([
                                SystemMessage(content=help_for_fixing_system_prompt),
                                HumanMessage(content=error_with_info_to_help_prompt)
                            ])
                            smart_print(
                                "ANALYSIS RECEIVED, GENERATING A FIX",
                                custom_agent if custom_agent else self.agent_name,
                                "code_task_and_run_test SystemMessage",
                                optional=False,
                                column_id=output_id
                            )
                            fix_description_prompt = f"ERROR MESSAGE:<<\n{exec_result}\n>>\n\nCODE TO FIX:<<\n{parsed_code['program_code']}\n>>\n\nHELPFUL INFORMATION:<<\n{getattr(help_code_returned, 'content', help_code_returned)}\n>>"
                            edited_code_returned = self.premium_llm.invoke([
                                SystemMessage(content=fix_system_prompt),
                                HumanMessage(content=fix_description_prompt)
                            ])
                            edited_code = str(getattr(edited_code_returned, 'content', edited_code_returned))

                        # Before updating parsed_code, save the previous code
                        prev_code = parsed_code["program_code"]
                        # Run the edited code
                        code_to_run = common_code + edited_code + "\n" + "\n".join(matching_tests)
                        smart_print(
                            "TESTING UPDATED CODE.....",
                            custom_agent if custom_agent else self.agent_name,
                            "code_task_and_run_test SystemMessage",
                            optional=False,
                            column_id=output_id
                        )

                        t1 = time.time()
                        no_runtime_error, exec_result, std_out_err = env.step(code_to_run)
                        # Smart print the std_out_err
                        smart_print(
                            f"FUNCTION DISPLAY OUTPUTS:\n{std_out_err}",
                            custom_agent if custom_agent else self.agent_name,
                            "code_task_and_run_test SystemMessage",
                            optional=False,
                            column_id=output_id
                        )
                        # Show score
                        scores = self.generate_score(idx, no_runtime_error, env.get_score(), time.time() - t1)
                        smart_print(
                            scores,
                            custom_agent if custom_agent else self.agent_name,
                            "Scores",
                            optional=False,
                            column_id=output_id
                        )

                        # Update parsed_code if re-run is successful
                        parsed_code["program_code"] = edited_code
                        smart_print(
                            f"# UPDATED **{'SUCCESFUL' if no_runtime_error else 'FAILED'}** CODE:\n{edited_code}",
                            custom_agent if custom_agent else self.agent_name,
                            "UPDATED_CODE",
                            optional=False,
                            column_id=output_id
                        )
                        # If no runtime error, store the error and diff
                        if no_runtime_error:
                            smart_print(
                                env.get_state(),
                                custom_agent if custom_agent else self.agent_name,
                                "CODE_RESULT",
                                optional=False,
                                column_id=output_id
                            )
                            diff = difflib.unified_diff(prev_code.splitlines(), edited_code.splitlines(), lineterm='')
                            diff_text = '\n'.join(diff)
                            # Avoid duplicates: check if the error and diff combination already exists
                            if (exec_result, diff_text) not in error_patches:
                                error_patches.append((exec_result, diff_text))
                                # Store updated error_patches
                                HumanLLMConfig().log_agent_data(self.agent_name, 'error_patches', error_patches, metadata=metadata)

            no_runtime_errors.append(no_runtime_error)
            exec_results.append(exec_result)

        # Calculate total time taken  # Added line
        total_execution_time = time.time() - start_time
        # Return combined results
        if isinstance(self.envs[0], SWEBenchEnvironment):
            scores = [env.get_score(parsed_code["program_code"]) for env in self.envs]
        else:
            scores = [env.get_score() for env in self.envs]
        result = (parsed_code, all(no_runtime_errors), exec_results, scores, [env.get_state() for env in self.envs], total_execution_time)

        if restore_state:
            for env in self.envs:
                env.restore_last_state()
        return result
    
    def code_task_and_run_test(self, refined_task):
        self.logger.info('Starting code_task_and_run_test')

        # Retrieve data from HumanLLMMonitor
        metadata = {'step_id': HumanLLMConfig().step_id}
        previous_errors, _ = HumanLLMConfig().get_agent_data(self.agent_name, 'previous_errors', metadata_filter=metadata)
        previous_scores, _ = HumanLLMConfig().get_agent_data(self.agent_name, 'previous_scores', metadata_filter=metadata)
        previous_codes, _ = HumanLLMConfig().get_agent_data(self.agent_name, 'previous_codes', metadata_filter=metadata)
        error_patches, _ = HumanLLMConfig().get_agent_data(self.agent_name, 'error_patches', metadata_filter=metadata)

        # Ensure variables are initialized
        previous_errors = previous_errors if previous_errors else []
        previous_scores = previous_scores if previous_scores else []
        previous_codes = previous_codes if previous_codes else []
        error_patches = flatten_and_pair(error_patches) if error_patches else []

        env_states = "\n".join([env.get_state() for env in self.envs])
        primitives = "\n".join(get_primitives(self.primitives_dir))
        successful_tasks = "\n".join(HumanLLMConfig().get_learnt_tasks())
        failed_tasks = "\n".join(HumanLLMConfig().get_failed_tasks())
        validation_response_um = "\n".join(HumanLLMConfig().get_validation_results())

        self.logger.info(f"Primitives: <<<\n{primitives}\n>>>")

        previous_attempts = ""
        for errors_list, scores_list, codes_list in zip(previous_errors, previous_scores, previous_codes):
            if errors_list and scores_list and codes_list:
                if isinstance(errors_list, str): # Current limitation of feedback limited to 1
                    errors_list = [errors_list] * len(scores_list)
                for err, score, code in zip(errors_list, scores_list, codes_list):
                    previous_attempts += f"\n<<ATTEMPT FEEDBACK: {err}\nSCORE: {score}\nCODE: {code}>>\n"

        error_patches_str = ""
        for (error_msg, diff_text) in error_patches:
            error_patches_str += f"\n<<ERROR MESSAGE: {error_msg}\nFIX APPLIED (diff):\n{diff_text}>>\n"

        # Define data for template placeholders
        template_data = {
            "refined_task": refined_task,
            "env_states": env_states,
            "primitives": primitives,
            "successful_tasks": successful_tasks,
            "failed_tasks": failed_tasks,
            "validation_response_um": validation_response_um,
            "previous_attempts": previous_attempts,
            "error_patches_str": error_patches_str
        }

        # Load and format the user message from a file template
        user_message = HumanLLMConfig().load_prompt_template(
            "coding_agent_user_message_template",
            template_data=template_data,
            directory='prompts'
        )

        if hasattr(self, 'log_user_message') and self.log_user_message:
            with open(self.log_user_message, "a") as f:
                f.write("Coder -- code_task_and_run_test:<<\n" + user_message + "\n>>\n\n")

        # Set the formatted user message
        self.last_user_message = user_message

        current_skip_rounds = self.skip_rounds  # save the initial value to align it for code validation
        # Initialisation des kwargs avec les paramètres requis
        kwargs = {
            "system_prompt_template": self.problem_prompts_subdir + "code_task",
            "user_message": user_message,
            "return_message_content_only": False,
            "stream_output": False,
            "model_choice": self.model_choice.get('coder', self.default_llm_name) if isinstance(self.model_choice, dict) else self.model_choice
        }

        # Ajouter temperature seulement si l'attribut temperature existe dans l'instance
        if hasattr(self, 'temperature'):
            kwargs["temperature_max"] = self.temperature


        # Appeler la méthode avec les arguments sous forme de **kwargs
        codes = self.invoke(**kwargs)
        results = []

        for index, code in enumerate(codes):
            # Get the proper check_results corresponding to the output_id (which is the index)
            check_results = self.inference_tracking.last_inference_check_results[index]
            if check_results:
                code_parsing_success, parsed_code = check_results.get("Code Parsing", (False, None))
                if code_parsing_success and isinstance(parsed_code, dict):
                    test_results = check_results.get("Run Tests", None)
                    if test_results:
                        results.append(test_results)

        if len(results) > 1:
            # display the list of results with success, exception and code
            results_list = ""
            top_results, top_indice = 0, 1
            for id, result in enumerate(results):
                if result[1]:
                    results_list += f"{id}. \033[32mSUCCESS\033[0m / SCORE: {result[3]} / TIME: {result[5]}s / CODE: {result[0]['program_code'][:100]}\n"
                    temp = sum(result[3][i][j] for i in range(len(result[3])) for j in result[3][i]) / len(result[3])
                    if temp > top_results:
                        top_results = temp
                        top_indice = id
                else:
                    results_list += f"{id}. \033[31mFAILED\033[0m / SCORE: {result[3]} / TIME: {result[5]}s / EXCEPTION: {result[2][0][:100]} / CODE: {result[0]['program_code'][:100]}\n"

            # ask the user to select the code to keep
            if current_skip_rounds <= 0:
                if self.automation:
                    selected_code = f"{top_indice}"
                else:
                    selected_code = smart_input(
                        f"{{{''.join(results_list)}}} CODE SELECTION Please select the code to keep (separated by comma, none/n for none of these, or just hit enter to keep ALL): ",
                        self.name, "Scores").strip().replace(" ", "").lower().split(",")
            else:
                selected_code = [""]  # keep all if skip_rounds is not 0
            id = 0
            if selected_code in ["none", "n"]:
                results = []
            else:
                # keep only the selected code
                results = [result for id, result in enumerate(results) if
                           selected_code and (str(id) in selected_code or selected_code == [""])]

        return results
    
    def generate_score(self, index:int, success:bool, score:Dict, elapsed_time:int)-> str:
        str_score = f"{index}. "
        if success:
            str_score += "\033[32mSUCCESS"
        else:
            str_score += "\033[31mFAILED"
        str_score += "\033[0m / SCORE: "
        str_score += f"[{','.join(f'{a:.2f}' for a in list(score.values()))}]"
        str_score += f" / TIME: {elapsed_time}s"
        str_score += f" / CODE: [{json.dumps(score)}]"
        return str_score
    
    def apply_special_criteria(agent, special_criteria, available_locals=None):
        """
        Apply special criteria to the attributes and parameters of an agent.

        :param agent: The agent instance to modify.
        :param special_criteria: Dictionary containing the special criteria.
        :param available_locals: Dictionary containing the local variables of the caller function.
        :return: Dictionary of only the modified parameters.
        """
        # Dictionary to store only the modified parameters
        new_params = {}

        if special_criteria:
            if available_locals is None:
                # Use inspect to dynamically capture arguments
                frame = inspect.currentframe().f_back  # Go up one level
                _, _, _, values = inspect.getargvalues(frame)
                available_locals = values

            class_name = agent.__class__.__name__
            # Iterate through the criteria related to this class
            for key, value in special_criteria.items():
                if key in ['self', 'special_criteria']: continue
                if '#' in key:
                    agent_name, key = key.split('#', 1)
                    if agent_name != class_name and agent_name not in ['all', '']: continue
                if hasattr(agent, key):
                    setattr(agent, key, value)
                elif key in available_locals:
                    new_params[key] = value
                    print(f"Special criteria applicable to {agent}'s local variables: {key} = {value}")

        print(new_params)
        return new_params  # Return only new params

    # -----------------------------
    # H2 convenience (explicit call)
    # -----------------------------
    def offline_optimize(
        self,
        optimizer_name: Optional[str] = None,
        targets: Optional[List[str]] = None,
        n_candidates: int = 1,
        constraints: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Run an OFFLINE (H2) optimization pass via the DynamicConfigManager, if available.
        This does NOT run automatically; call it explicitly (e.g., after a batch of runs).

        Returns a summary dict from the manager, typically including:
            { "applied": bool, "applied_edits": [...], ... }
        """
        if not getattr(self, "dynamic_mgr", None):
            raise RuntimeError("DynamicConfigManager is not initialized on this HumanLLM instance.")
        if not hasattr(self.dynamic_mgr, "offline_optimize"):
            raise NotImplementedError(
                "DynamicConfigManager.offline_optimize(...) is not available. "
                "Update DynamicConfigManager to a version that supports H2."
            )
        return self.dynamic_mgr.offline_optimize(optimizer_name, targets, n_candidates, constraints)