import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.join(os.getcwd(), 'NewTrace'))))
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), '../..')))

import pytest
import numpy as np
import datasets
from opto import trace
from opto.utils.llm import LLM, LiteLLM, AutoGenLLM
from opto.optimizers import OptoPrime, OptoPrimeMulti
from opto.trainer.algorithms.basic_algorithm import MinibatchAlgorithm
from opto.trainer.guide import VerbalJudgeGuide
from humanllm_trace import HumanLLM_Trace
from typing import Any


def humanllm_factory(config_list):
    # Extract relevant configuration for HumanLLM
    trace = HumanLLM_Trace()
    return trace


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

LiteLLM._factory = staticmethod(humanllm_factory)
AutoGenLLM._factory = staticmethod(humanllm_factory)

test_params = [
    # (agent_llm, guide_model, optimizer_class)
    (HumanLLM_Trace(), None, OptoPrime),            # HumanLM as backend
    (LLM(HumanLLM_Trace()), None, OptoPrime),       # HumanLM as parameter for Learner
    (None, HumanLLM_Trace(), OptoPrime),            # HumanLM as parameter for Guide
    (None, None, lambda p: OptoPrime(p, llm=HumanLLM_Trace())),  # HumanLM for OptoPrime
    (None, None, lambda p: OptoPrimeMulti(p, llm=HumanLLM_Trace()))  # HumanLM for OptoPrimeMulti
]


@pytest.mark.parametrize("agent_llm,guide_model,optimizer_class", test_params)
def test_learner_with_humanlm(setup_data, agent_llm, guide_model, optimizer_class):
    train_dataset = setup_data
    test_dataset = setup_data
    
    

    agent = Learner(llm=agent_llm)
    guide = VerbalJudgeGuide(model=guide_model)
    
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
    
    # Validate the structure of the results
    assert isinstance(results, tuple), "Expected results to be a tuple"
    assert len(results) == 2, "Expected results to have two elements (train_scores, test_score)"

    # Validate train_scores
    train_scores, test_score = results
    assert isinstance(train_scores, list), "Expected train_scores to be a list"
    assert all(isinstance(score, (int, float)) for score in train_scores), "All train_scores should be numeric"

    # Validate test_score
    assert isinstance(test_score, (int, float)), "Expected test_score to be numeric"
