import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.join(os.getcwd(), 'NewTrace'))))
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), '../..')))

import pytest
import numpy as np
import datasets
from opto import trace
from opto.utils.llm import LLM, LiteLLM, AutoGenLLM, LLMFactory, _LLM_REGISTRY
from opto.optimizers import OptoPrime, OptoPrimeMulti
from opto.trainer.algorithms.basic_algorithm import MinibatchAlgorithm
from opto.trainer.guide import VerbalJudgeGuide
from humanllm_trace import HumanLLM_Trace
from typing import Any

# Save the original _factory method
_original_factory = LiteLLM._factory

# Define the patched _factory method
@classmethod
def patched_factory(cls, model_name: str):
    if model_name == 'HumanLLM_Trace':  # Check if model_name is 'HumanLLM'
        default_model = 'gpt-4o-mini'
        print("Using HumanLLM_Trace for model_name:", model_name)
        return lambda *args, **kwargs: HumanLLM_Trace(model=default_model).run(default_model, args, **kwargs)

    # Call the original _factory method for other models
    return _original_factory(model_name)

# Apply the monkey patch
LiteLLM._factory = patched_factory

@trace.model
class Learner:
    """A basic LLM agent."""
    def __init__(self, system_prompt: str = "You're a helpful agent",
                 user_prompt_template: str = "Query: {message}",
                 llm: LLM = None):
        self.system_prompt = trace.node(system_prompt, trainable=True)
        self.user_prompt_template = trace.node(user_prompt_template)
        self.llm = llm or LLM()

    @trace.bundle()
    def model(self, system_prompt: str, user_prompt_template: str, message: str) -> str:
        if '{message}' not in user_prompt_template:
            raise ValueError("user_prompt_template must contain '{message}'")

        response = self.llm(
            messages=[{"role": "system", "content": system_prompt},
                      {"role": "user", "content": user_prompt_template.format(message=message)}]
        )
        return response.choices[0].message.content

    def forward(self, message: Any) -> Any:
        return self.model(self.system_prompt, self.user_prompt_template, message)

@pytest.fixture
def setup_data():
    np.random.seed(42)
    train_dataset = datasets.load_dataset('openai/gsm8k', 'main')['train'][:2]  # Smaller subset for testing
    return dict(inputs=train_dataset['question'], infos=train_dataset['answer'])


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


def main(agent_llm=None, guide_model=None, optimizer_class=None):
    np.random.seed(42)
    train_dataset = datasets.load_dataset('openai/gsm8k', 'main')['train'][:2]
    train_dataset = dict(inputs=train_dataset['question'], infos=train_dataset['answer'])
    test_dataset = train_dataset

    agent = Learner(llm=agent_llm)
    guide = VerbalJudgeGuide(llm=guide_model)
    
    # Handle both direct class and lambda cases
    if callable(optimizer_class):
        optimizer = optimizer_class(agent.parameters())
    else:
        optimizer = optimizer_class(agent.parameters())
    
    alg = MinibatchAlgorithm(
        agent=agent,
        optimizer=optimizer
    )
    
    # Run with minimal settings for testing
    results = alg.train(
        guide,
        train_dataset,
        num_epochs=1,
        batch_size=1,
        eval_frequency=-1,
        test_dataset=test_dataset,
        num_threads=1,
        verbose=False
    )
    return results


if __name__ == "__main__":

    # Save the original constructor
    _original_litellm_init = LiteLLM.__init__
    _original_autogen_init = AutoGenLLM.__init__
    # Apply the monkey patch
    LiteLLM.__init__ = patched_litellm_init
    AutoGenLLM.__init__ = patched_autogenllmm_init


    # register new llm entry
    _LLM_REGISTRY['HumanLLM_Trace'] = HumanLLM_Trace

    # Register custom profiles for different use cases
    LLMFactory.register_profile("human_llm_backend", "LiteLLM", model="HumanLLM_Trace", temperature=0.5)
    LLMFactory.register_profile("hllm_model", "LiteLLM", model="gpt-4o-mini", temperature=0.5, max_tokens=8000)

    # Multi-LLM optimizer with multiple profiles
    optimizer_class = lambda p: OptoPrimeMulti(p, llm_profiles=["human_llm_backend", "hllm_model"], generation_technique="multi_llm")

    main(
        agent_llm=LLM(profile="human_llm_backend"),
        guide_model=LLM(profile="hllm_model"),
        optimizer_class=optimizer_class
    )