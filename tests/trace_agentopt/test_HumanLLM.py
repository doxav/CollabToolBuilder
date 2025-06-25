# test_main.py

import os
import pytest
from run_HumanLLM import main, HumanLLM_Trace, LLM, OptoPrimeMulti  # Adjust import as needed
from opto.utils.llm import LLM, LiteLLM, AutoGenLLM, LLMFactory, _LLM_REGISTRY


_original_factory = LiteLLM._factory
_original_litellm_init = LiteLLM.__init__
_original_autogen_init = AutoGenLLM.__init__
# Define the patched constructors
def patched_litellm_init(self, model=None, reset_freq=None, cache=True, **kwargs):
    """
    Patched constructor for LiteLLM to support kwargs.
    """
    # Call the original constructor
    _original_litellm_init(self, model=model, reset_freq=reset_freq, cache=cache)
    # Store additional kwargs in the instance
    self.kwargs = kwargs

def patched_autogenllmm_init(self, config_list=None, filter_dict=None, reset_freq=True, **kwargs):
    """
    Patched constructor for LiteLLM to support kwargs.
    """
    # Call the original constructor
    _original_autogen_init(self, config_list=config_list, filter_dict=filter_dict, reset_freq=reset_freq)
    # Store additional kwargs in the instance
    self.kwargs = kwargs

# Define the patched _factory method
@classmethod
def patched_factory(cls, model_name: str):
    if model_name == 'HumanLLM_Trace':  # Check if model_name is 'HumanLLM'
        default_model = 'gpt-4o-mini'
        print("Using HumanLLM_Trace for model_name:", model_name)
        return lambda *args, **kwargs: HumanLLM_Trace(model=default_model).run(default_model, args, **kwargs)

    # Call the original _factory method for other models
    return _original_factory(model_name)



@pytest.fixture(scope="module", autouse=True)
def setup_env_and_profiles():
    # Set the API key and env
    os.environ["UI_MODE"] = "False"

    # Apply the monkey patch
    LiteLLM._factory = patched_factory

    LiteLLM.__init__ = patched_litellm_init
    AutoGenLLM.__init__ = patched_autogenllmm_init
    _LLM_REGISTRY['HumanLLM_Trace'] = HumanLLM_Trace

    # Register patched profiles
    LLMFactory.register_profile("human_llm_backend", "LiteLLM", model="HumanLLM_Trace", temperature=0.5)
    LLMFactory.register_profile("hllm_model", "LiteLLM", model="gpt-4o-mini", temperature=0.5, max_tokens=8000)


def test_main_execution():
    """Ensure main() runs without error and returns a result."""
    agent_llm = LLM(profile="human_llm_backend")
    guide_model = LLM(profile="hllm_model")

    optimizer_class = lambda p: OptoPrimeMulti(
        p,
        llm_profiles=["human_llm_backend", "hllm_model"],
        generation_technique="multi_llm"
    )

    result = main(agent_llm=agent_llm, guide_model=guide_model, optimizer_class=optimizer_class)

    print(">>>>", result)
    # Assert result is not None and has expected keys
    assert result is not None

if __name__ == '__main__':
    test_main_execution()
