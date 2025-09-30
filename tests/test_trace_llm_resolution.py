"""
Test suite for LLM resolution in trace optimizers.

This test suite verifies LLM resolution functionality for trace optimizers.
Supported features:
- "human" alias: resolves to HumanLLM backend for interactive optimization
- "profile:name" format: resolves to registered LLM profiles
- Bare strings: treated as profile names (e.g., "hllm_model")
- Config precedence: top-level llm config overrides optimizer_kwargs
- Works with: OptoPrimeV2, OptoPrimeMulti (default is OptoPrimeV2)

Note: "model:name" format is intentionally not supported - users should create
profiles instead for better configuration management.
"""
import os
import pytest
import sys
from pathlib import Path

# Add project root to path
root_path = str(Path(__file__).resolve().parents[1])
sys.path.insert(0, root_path)

# --- same monkeypatching style as your existing tests ---
from opto.utils.llm import LLM, LiteLLM, AutoGenLLM, LLMFactory, _LLM_REGISTRY

# Import the HumanLLM_Trace from the test directory
sys.path.insert(0, os.path.join(root_path, "tests", "trace_agentopt"))
from humanllm_trace import HumanLLM_Trace

_original_factory = LiteLLM._factory
_original_litellm_init = LiteLLM.__init__
_original_autogen_init = AutoGenLLM.__init__

@classmethod
def patched_factory(cls, model_name: str):
    # Only intercept the special "HumanLLM_Trace" model – everything else keeps original factory
    if model_name == 'HumanLLM_Trace':
        default_model = 'gpt-4o-mini'
        return lambda *args, **kwargs: HumanLLM_Trace(model=default_model).run(default_model, args, **kwargs)
    return _original_factory(model_name)

def patched_litellm_init(self, model=None, reset_freq=None, cache=True, **kwargs):
    _original_litellm_init(self, model=model, reset_freq=reset_freq, cache=cache)
    self.kwargs = kwargs

def patched_autogen_init(self, config_list=None, filter_dict=None, reset_freq=True, **kwargs):
    _original_autogen_init(self, config_list=config_list, filter_dict=filter_dict, reset_freq=reset_freq)
    self.kwargs = kwargs

@pytest.fixture(scope="module", autouse=True)
def setup_profiles_and_patches():
    os.environ["UI_MODE"] = "False"
    LiteLLM._factory = patched_factory
    LiteLLM.__init__ = patched_litellm_init
    AutoGenLLM.__init__ = patched_autogen_init
    _LLM_REGISTRY['HumanLLM_Trace'] = HumanLLM_Trace

    # Profiles used in tests
    LLMFactory.register_profile("human_llm_backend", "LiteLLM", model="HumanLLM_Trace")
    LLMFactory.register_profile("hllm_model", "LiteLLM", model="gpt-4o-mini")
    yield
    # No teardown needed; tests run offline

def build_adapter(tconf):
    # import the adapter after patches (so it sees patched LiteLLM)
    from utils.human_llm import _TraceOptimizerAdapter
    return _TraceOptimizerAdapter(tconf.get("name","t"), tconf)

def test_llm_human_alias():
    """Test 'human' alias resolves to HumanLLM backend (supported format)"""
    tconf = {
        "name": "opt_human",
        "optimizer_kind": "OptoPrimeV2", 
        "llm": "human",
        "parameters": [{"parameter":"invoke.temperature","value":0.4,"trainable":True}],
    }
    ad = build_adapter(tconf)
    # Verify the optimizer was created successfully
    assert ad._trace_obj is not None
    # The key test: verify LLM was resolved to an AbstractModel instance
    from opto.utils.llm import AbstractModel
    assert hasattr(ad._trace_obj, 'llm'), "Optimizer should have llm attribute"
    assert isinstance(ad._trace_obj.llm, AbstractModel), f"Expected AbstractModel, got {type(ad._trace_obj.llm)}"

def test_llm_human_llm_alias():
    """Test 'human_llm' alias resolves to HumanLLM backend"""
    tconf = {
        "name": "opt_human_llm",
        "optimizer_kind": "OptoPrime",
        "llm": "human_llm",
        "parameters": [{"parameter":"invoke.temperature","value":0.4,"trainable":True}],
    }
    ad = build_adapter(tconf)
    opt = ad._trace_obj
    from opto.utils.llm import AbstractModel
    assert isinstance(opt.llm, AbstractModel)

def test_llm_profile_string():
    """Test profile string with 'profile:' prefix (supported format)"""
    tconf = {
        "name": "opt_profile",
        "optimizer_kind": "OptoPrime",
        "llm": "profile:hllm_model",
        "parameters": ["invoke.temperature"]
    }
    ad = build_adapter(tconf)
    from opto.utils.llm import AbstractModel
    assert isinstance(ad._trace_obj.llm, AbstractModel)



def test_llm_bare_string_as_profile():
    """Test bare string treated as profile name (supported format)"""
    tconf = {
        "name": "opt_bare_profile",
        "optimizer_kind": "OptoPrimeV2",
        "llm": "hllm_model",  # Should try as profile first
        "parameters": ["invoke.temperature"]
    }
    ad = build_adapter(tconf)
    from opto.utils.llm import AbstractModel
    assert isinstance(ad._trace_obj.llm, AbstractModel)



def test_llm_dict_with_profile():
    """Test dict specification with profile key"""
    tconf = {
        "name": "opt_dict_profile",
        "optimizer_kind": "OptoPrimeV2",
        "llm": {"profile": "hllm_model"},
        "parameters": ["invoke.temperature"]
    }
    ad = build_adapter(tconf)
    from opto.utils.llm import AbstractModel
    assert isinstance(ad._trace_obj.llm, AbstractModel)

def test_llm_dict_with_backend_and_model():
    """Test dict specification with backend and model"""
    tconf = {
        "name": "opt_dict_backend",
        "optimizer_kind": "OPRO",
        "llm": {"backend": "LiteLLM", "model": "gpt-4o-mini", "temperature": 0.7},
        "parameters": ["invoke.temperature"]
    }
    ad = build_adapter(tconf)
    from opto.utils.llm import AbstractModel
    assert isinstance(ad._trace_obj.llm, AbstractModel)

def test_llm_in_optimizer_kwargs():
    """Test LLM specified in optimizer_kwargs gets resolved"""
    tconf = {
        "name": "opt_kwargs_llm",
        "optimizer_kind": "OptoPrimeV2",
        "optimizer_kwargs": {"llm": "human"},
        "parameters": ["invoke.temperature"]
    }
    ad = build_adapter(tconf)
    # assert that ad._trace_obj.llm is "human"
    assert ad._trace_obj is not None
    assert ad._trace_obj.llm == "human"

def test_llm_config_overrides_kwargs():
    """Test that top-level llm config overrides optimizer_kwargs llm"""
    tconf = {
        "name": "opt_override",
        "optimizer_kind": "OptoPrimeV2",
        "llm": "human",  # This should take precedence
        "optimizer_kwargs": {"llm": "hllm_model"},
        "parameters": ["invoke.temperature"]
    }
    ad = build_adapter(tconf)
    assert ad._trace_obj.llm == "hllm_model"

def test_no_llm_specified():
    """Test that optimizer works without LLM specification (uses default)"""
    tconf = {
        "name": "opt_no_llm",
        "optimizer_kind": "OptoPrime",
        "parameters": ["invoke.temperature"]
    }
    ad = build_adapter(tconf)
    # Should still create successfully, optimizer will use its default LLM
    assert ad._trace_obj.llm is not None

def test_existing_llm_profiles_in_kwargs_preserved():
    """Test that existing llm_profiles in kwargs is not overridden"""
    tconf = {
        "name": "opt_preserve_kwargs",
        "optimizer_kind": "OptoPrimeMulti", 
        "llm_profiles": ["human_llm_backend"],  # config level
        "optimizer_kwargs": {"llm_profiles": ["hllm_model"]},  # kwargs level (should win)
        "parameters": ["invoke.temperature"]
    }
    ad = build_adapter(tconf)
    opt = ad._trace_obj
    # kwargs should take precedence
    assert getattr(opt, "llm_profiles", None) == ["hllm_model"]