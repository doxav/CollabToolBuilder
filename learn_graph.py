from typing import TypedDict, Sequence, Union, Optional, Dict, Any, List
from langchain_core.messages.base import BaseMessage
from langgraph.graph import StateGraph, END
import time
import random  # Import random for generating step_id
from datetime import datetime
from utils.llm_utils import (
    calculate_total_score,
    get_success_value_in_text
)
from utils.human_llm import HumanLLM
from learn import TaskIdentificationAgent, CodingAgent, ValidationAgent, CapitalizationAgent
from utils.llm_utils import create_Nmajority_chain

# 2 IMPLEMENTATIONS OF THE LEARNING LOOP: 1 MODULAR AND ADVANCED IN STATE MANAGEMENT, 1 SIMPLE IN 1 FUNCTION

# IMPLEMENTATION 1 MODULAR AND ADVANCED IN STATE MANAGEMENT

class WorkflowState(TypedDict):
    current_agent: str
    task_description: Optional[str]
    parsed_code: Optional[dict]
    validation_result: Optional[str]
    scores: Optional[dict]
    execution_results: Optional[list]
    saved_task: Optional[dict]
    automation: Optional[Union[str, dict]]
    special_criteria: Optional[dict]
    workflow_status: str
    step_id: str
    attempts: int
    start_time: float
    max_execution_time: float
    environment_states: List[dict]
    parallel_results: List[Any]

class AgentGraphNode:
    """Base class for agent nodes with state management"""
    def __init__(self, agent, name: str):
        self.agent = agent
        self.name = name

    def save_state(self, state: WorkflowState, metadata: Optional[Dict] = None):
        """Save current workflow state"""
        if metadata is None:
            metadata = {}
        metadata.update({
            "step_id": state["step_id"],
            "agent_name": self.name,
            "timestamp": datetime.now().isoformat()
        })
        
        HumanLLM.add_agent_data(
            self.name,
            'workflow_state',
            state,
            metadata=metadata
        )

    def restore_state(self, state_id: str) -> Optional[WorkflowState]:
        """Restore workflow state"""
        states, _ = HumanLLM.get_agent_data(
            self.name,
            'workflow_state',
            id_task=state_id
        )
        return states[0] if states else None

    def __call__(self, state: WorkflowState) -> WorkflowState:
        """Template method for node execution"""
        try:
            state["current_agent"] = self.name
            if time.time() - state["start_time"] >= state["max_execution_time"]:
                state["workflow_status"] = "timeout"
                return state
            
            result = self.process(state)
            self.save_state(result)
            return result
        except Exception as e:
            state["workflow_status"] = "error"
            state["error"] = str(e)
            self.save_state(state)
            return state

    def process(self, state: WorkflowState) -> WorkflowState:
        """To be implemented by concrete nodes"""
        raise NotImplementedError

class TaskIdentificationNode(AgentGraphNode):
    def process(self, state: WorkflowState) -> WorkflowState:
        task = self.agent.identify_best_task()
        state["task_description"] = task[0].content if isinstance(task, list) else task.content
        return state

class CodingNode(AgentGraphNode):
    def process(self, state: WorkflowState) -> WorkflowState:
        results = self.agent.code_task_and_run_test(state["task_description"])
        state["parallel_results"] = results
        
        if results:
            best_result = max(results, key=lambda x: sum(sum(s.values()) for s in x[3]))
            state["parsed_code"] = best_result[0]
            state["execution_results"] = best_result[2]
            state["scores"] = best_result[3]
        else:
            # Ensure scores is always a dictionary
            state["scores"] = {
                'validated_scores': [],
                'percentage_no_runtime_error': 0,
                'best_score_without_validation': 0
            }
        
        state["attempts"] += 1
        return state

class ValidationNode(AgentGraphNode):
    def process(self, state: WorkflowState) -> WorkflowState:
        validation = self.agent.validate_code(
            state["parsed_code"]["program_code"],
            all(result[1] for result in state["parallel_results"]),
            state["execution_results"],
            task=state["task_description"],
            scores=state["scores"],
            env_states=state["environment_states"]
        )
        
        state["validation_result"] = "success" if get_success_value_in_text(validation[0].content) else "failed"
        return state

class CapitalizationNode(AgentGraphNode):
    def process(self, state: WorkflowState) -> WorkflowState:
        if state["validation_result"] == "success":
            self.agent.capitalize_successful_tasks(
                state["task_description"],
                state["parsed_code"]
            )
        else:
            self.agent.capitalize_failed_tasks(
                state["task_description"],
                state["parsed_code"]
            )
        return state

def create_workflow_graph(
    agent_task: TaskIdentificationAgent,
    agent_coding: CodingAgent,
    agent_validation: ValidationAgent,
    agent_capitalize: CapitalizationAgent,
    max_attempts: int,
    continue_each_loop: bool
) -> StateGraph:
    
    workflow = StateGraph(WorkflowState)
    
    # Add nodes
    workflow.add_node("task_identification", TaskIdentificationNode(agent_task, "TaskIdentificationAgent"))
    workflow.add_node("coding", CodingNode(agent_coding, "CodingAgent"))
    workflow.add_node("validation", ValidationNode(agent_validation, "ValidationAgent"))
    workflow.add_node("capitalization", CapitalizationNode(agent_capitalize, "CapitalizationAgent"))
    
    # Define routing logic
    def route_after_validation(state: WorkflowState) -> Union[str, List[str]]:
        if state["workflow_status"] in ["timeout", "error"]:
            return END  # End the graph if there's an error or timeout
            
        if state["validation_result"] == "success":
            return "capitalization"  # Route to capitalization on success
        
        if state["attempts"] >= max_attempts:
            return "task_identification" if continue_each_loop else END  # Retry or end based on configuration
            
        return "coding"  # Retry coding if validation fails

    def route_after_capitalization(state: WorkflowState) -> str:
        return "task_identification" if continue_each_loop else END  # Continue loop or terminate graph
    
    # Add edges
    workflow.add_edge("task_identification", "coding")
    workflow.add_edge("coding", "validation")
    
    # Add conditional edges
    workflow.add_conditional_edges("validation", route_after_validation)
    workflow.add_conditional_edges("capitalization", route_after_capitalization)
    
    # Compile the graph
    compiled_workflow = workflow.compile()
    
    return compiled_workflow

def run_4agents_learning_loop_graph(
    default_llm_key: str,
    premium_llm_key: str,
    test_environments=None,
    manual_validation_to_capitalize: bool = True,
    problem_prompts_subdir: Optional[str] = None,
    max_coding_attempts: int = 4,
    automation: Optional[Union[str, dict]] = None,
    special_criteria: Optional[dict] = None,
    max_execution_time: int = 900,
    continue_each_loop: bool = False,
    date_start = None,
    **kwargs
) -> Union[float, List[float]]:
    
    # Initialize agents with all parameters
    agent_task = TaskIdentificationAgent(
        default_llm_key,
        test_environments,
        premium_llm_choice=premium_llm_key,
        problem_prompts_subdir=problem_prompts_subdir,
        automation=automation.get('taskreco', automation) if isinstance(automation, dict) else automation,
        special_criteria=special_criteria, llmORchains_list=llmORchains_list
    )
    
    agent_coding = CodingAgent(
        default_llm_key,
        test_environments,
        premium_llm_choice=premium_llm_key,
        problem_prompts_subdir=problem_prompts_subdir,
        automation=automation.get('coder', automation) if isinstance(automation, dict) else automation,
        special_criteria=special_criteria, llmORchains_list=llmORchains_list
    )
    
    agent_validation = ValidationAgent(
        default_llm_key,
        test_environments,
        premium_llm_choice=premium_llm_key,
        automation=automation.get('critic', automation) if isinstance(automation, dict) else automation,
        special_criteria=special_criteria, llmORchains_list=llmORchains_list
    )
    
    agent_capitalize = CapitalizationAgent(
        default_llm_key,
        premium_llm_choice=premium_llm_key,
        problem_prompts_subdir=problem_prompts_subdir,
        automation=automation.get('capitalizer', automation) if isinstance(automation, dict) else automation,
        special_criteria=special_criteria, llmORchains_list=llmORchains_list
    )
    
    # Create and compile workflow graph
    compiled_workflow = create_workflow_graph(
        agent_task,
        agent_coding,
        agent_validation,
        agent_capitalize,
        max_coding_attempts,
        continue_each_loop
    )
    
    # Initialize workflow state
    initial_state: WorkflowState = {
        "current_agent": "TaskIdentificationAgent",
        "task_description": None,
        "parsed_code": None,
        "validation_result": None,
        "scores": {
            'validated_scores': [],
            'percentage_no_runtime_error': 0,
            'best_score_without_validation': 0
        },
        "execution_results": None,
        "saved_task": special_criteria.get("all#saved_task", None) if special_criteria else None,
        "automation": automation,
        "special_criteria": special_criteria,
        "workflow_status": "running",
        "step_id": str(random.randint(0, 1000000)),
        "attempts": 0,
        "start_time": time.time(),
        "max_execution_time": max_execution_time,
        "environment_states": [env.get_state() for env in test_environments],
        "parallel_results": []
    }
    
    # Run compiled workflow
    final_state = compiled_workflow.invoke(initial_state)
    
    # Calculate and return scores
    if kwargs.get("return_array", False):
        return [calculate_total_score(final_state["scores"])]
    return calculate_total_score(final_state["scores"])

# IMPLEMENTATION 2 SIMPLE IN 1 FUNCTION

def run_4agents_learning_loop_graph(
    default_llm_key,
    premium_llm_key,
    test_environments=None,
    manual_validation_to_capitalize=True,
    problem_prompts_subdir=None,
    max_coding_attempts=4,
    include_code=None,
    selected_successful_functions=None,
    selected_failed_functions=None,
    agtask_premium_llm_by_default=True,
    agtask_skip_rounds=0,
    agcoding_skip_rounds=0,
    agvalidation_skip_rounds=0,
    agcapitalize_skip_rounds=0,
    llmORchains_list=None,
    model_choice=None,
    automation=None,
    allow_custom_score_state_functions=False,
    params_user_message=None,
    max_execution_time=900,
    special_criteria=None,
    temperature_max=1,
    agcoach_num_parallel_inferences=1,
    fixed_coach=False,
    return_array=False,
    agcoding_num_parallel_inferences=1,
    continue_each_loop=False,
    primitives_dir=None,
    functions_to_import=None,
    embedding_function=None,
    date_start=None,
):
    from learn import coding_and_validation_loop

    # Definition of automation depending on the task given
    if functions_to_import:
        print("ERROR: functions_to_import not implemented yet")

    # Initialize state tracking
    class AgentState(TypedDict):
        task: str
        code: dict  # Changed from str to dict to match parsed_code structure
        validation: str
        messages: Sequence[BaseMessage]
        scores: dict
        should_continue: bool
        environments: list
        total_scores: list
        end_time: float

    # Create agent nodes with proper initialization
    def create_task_agent(state: AgentState) -> AgentState:
        agent = TaskIdentificationAgent(
            default_llm_key,
            state["environments"],
            premium_llm_choice=premium_llm_key,
            problem_prompts_subdir=problem_prompts_subdir,
            premium_llm_by_default=agtask_premium_llm_by_default,
            skip_rounds=agtask_skip_rounds,
            llmORchains_list=llmORchains_list,
            automation=automation.get('taskreco') if isinstance(automation, dict) else automation,
            model_choice=model_choice,
            criteria=params_user_message,
            temperature_max=temperature_max,
            num_parallel_inferences=agcoach_num_parallel_inferences,
            fixed_coach=fixed_coach,
            special_criteria=special_criteria,
            primitives_dir="primitives/generate_graph"
        )
        task = agent.identify_best_task()
        return {"task": task[0].content, **state}  # Extract content from first task

    def create_coding_agent(state: AgentState) -> AgentState:
        agent = CodingAgent(
            default_llm_key,
            state["environments"],
            premium_llm_choice=premium_llm_key,
            problem_prompts_subdir=problem_prompts_subdir,
            skip_rounds=agcoding_skip_rounds,
            llmORchains_list=llmORchains_list,
            automation=automation.get('coder') if isinstance(automation, dict) else automation,
            model_choice=model_choice,
            special_criteria=special_criteria,
            num_parallel_inferences=agcoding_num_parallel_inferences,
            primitives_dir=primitives_dir
        )
        validation_agent = ValidationAgent(
            default_llm_key,
            state["environments"],
            premium_llm_choice=premium_llm_key,
            skip_rounds=agvalidation_skip_rounds,
            llmORchains_list=llmORchains_list,
            automation=automation.get('critic') if isinstance(automation, dict) else automation,
            model_choice=model_choice,
            special_criteria=special_criteria
        )

        parsed_code, validation, scores = coding_and_validation_loop(
            agent,
            validation_agent,
            state["task"],
            max_coding_attempts,
            manual_validation_to_capitalize,
            automation=agent.human_llm_code_task.automation,
            end_time=state["end_time"]
        )

        return {
            "code": parsed_code,
            "validation": validation,
            "scores": scores,
            **state
        }

    def create_capitalization_agent(state: AgentState) -> AgentState:
        agent = CapitalizationAgent(
            default_llm_key,
            premium_llm_choice=premium_llm_key,
            skip_rounds=agcapitalize_skip_rounds,
            problem_prompts_subdir=problem_prompts_subdir,
            llmORchains_list=llmORchains_list,
            automation=automation.get('capitalizer') if isinstance(automation, dict) else automation,
            model_choice=model_choice,
            special_criteria=special_criteria
        )

        if state["validation"] == "success":
            agent.capitalize_successful_tasks(state["task"], state["code"])
        else:
            agent.capitalize_failed_tasks(state["task"], state["code"])

        # Update total scores
        if state["scores"]:
            total_score = calculate_total_score(state["scores"])
            state["total_scores"].append(total_score)

        # Determine whether to continue
        should_continue = continue_each_loop if agent.automation else smart_input("Do you want to continue (y) or stop the loop (n)?") == "y"

        return {"should_continue": should_continue, **state}

    def should_continue(state: AgentState) -> str:
        if not state["should_continue"] or time.time() >= state["end_time"]:
            return "end"
        return "continue"

    # Create workflow graph
    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("task_coach", create_task_agent)
    workflow.add_node("code_task", create_coding_agent)
    workflow.add_node("capitalize", create_capitalization_agent)

    # Add edges
    workflow.add_edge("task_coach", "code_task")
    workflow.add_edge("code_task", "capitalize")

    # Add conditional edges
    workflow.add_conditional_edges(
        "capitalize",
        should_continue,
        {
            "continue": "task_coach",
            "end": END
        }
    )

    # Set entry point
    workflow.set_entry_point("task_coach")

    # Compile graph
    app = workflow.compile()

    png_data = app.get_graph().draw_mermaid_png()
    with open("graph_visualization.png", "wb") as f:
        f.write(png_data)

    # Initialize state
    initial_state = {
        "messages": [],
        "should_continue": True,
        "environments": test_environments,
        "total_scores": [],
        "end_time": time.time() + max_execution_time
    }

    # Execute graph
    result = app.invoke(initial_state)

    # Return results based on return_array parameter
    if return_array:
        return result["total_scores"]
    else:
        return max(result["total_scores"]) if result["total_scores"] else 0

##### TEMPORARY TO BE REMOVED #####

if __name__ == "__main__":
    import inspect
    import pickle
    import random
    import string
    import time
    from zipfile import error
    from config import *

    import openai
    from typing import Dict

    from utils.llm_utils import UnifiedVectorDB, HumanLLM, _visual_input, smart_print, smart_input

    import os
    import uuid
    import re
    import sys
    import socket
    import json
    import difflib
    import openai
    from datetime import datetime
    from typing import Dict
    from datasets import load_dataset
    from langchain_core.runnables import Runnable
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.messages.human import HumanMessage
    from langchain_core.messages.ai import AIMessage
    from langchain_core.messages.system import SystemMessage
    from langchain_openai import ChatOpenAI
    from langgraph.graph import StateGraph
    from langgraph.graph import END, START
    from config import *
    from env.env import EnvironmentManager, validate_function_code
    from utils.llm_utils import UnifiedVectorDB, HumanLLM, _visual_input, smart_print, smart_input
    from env.IR_CPS_TechSynthesis.env import *
    from env.SWEBench.env import *
    from env.env import Environment, EnvironmentManager

    from typing import TypedDict, Sequence

    import torch
    import gc
    from transformers import PreTrainedModel
    from torch.nn.modules.sparse import Embedding

    #set_llm_cache(SQLiteCache(database_path=".langchain_caching.db"))

    openai.api_key = os.environ['OPENAI_API_KEY']
    if 'OPENAI_BASE_URL' in os.environ: openai.base_url = os.environ['OPENAI_BASE_URL']

    UnifiedVectorDB.db_type = "elasticsearch"  # "elasticsearch" "chroma"
    UnifiedVectorDB.es_url = elastic_url_port
    # UnifiedVectorDB.es_user = elastic_user
    # UnifiedVectorDB.es_password = elastic_password
    UnifiedVectorDB.OpenAI_embedding_function_name = "text-embedding-ada-002"  # "nomic-ai/nomic-embed-text-v1"


    embedding_function = "text-embedding-ada-002" if embedding_function is None else embedding_function  #"Alibaba-NLP/gte-base-en-v1.5" UnifiedVectorDB.OpenAI_embedding_function_name # e.g. "text-embedding-ada-002" for OpenAI or "intfloat/e5-base-v2" or other huggingface models - WARINING: if you change it, set reset_db_indices to True
    # test if reset_db_indices exists
    if not 'reset_db_indices' in locals():
        reset_db_indices = False  # Set it in your config.py to True if you want to reset "after changing embeddings"

    HumanLLM.use_websocket = True

    import argparse
    import pickle

    # Handle command line arguments
    parser = argparse.ArgumentParser(description="Run the learning loop with optional WebSocket settings")
    parser.add_argument("--port", type=int, default=6789, help="Optional port for WebSocket server")
    parser.add_argument("--secret", action='store_true', help="Optional secret for WebSocket URL")
    parser.add_argument("--proxy", action='store_true', help="Start a proxy via localtunnel if available")
    parser.add_argument("--pickle_name", type=str, help="Optional pickle file name")
    args = parser.parse_args()

    if args.pickle_name and os.path.exists(f'pickle/{args.pickle_name}.pkl'):
        with open(f'pickle/{args.pickle_name}.pkl', 'rb') as f:
            variables_from_pickle = pickle.load(f)
            saved_task = variables_from_pickle.get('saved_task')
            saved_task['content'] = json.loads(saved_task['content'])
            automatic = variables_from_pickle.get('automatic')
            special_criteria = variables_from_pickle.get('special_criteria')

        # Suppression du fichier pickle après utilisation pour éviter les conflits lors des prochains lancements
        os.remove(f'pickle/{args.pickle_name}.pkl')

    # Initialize HumanLLMMonitor databases
    HumanLLM._check_and_init_vector_db(embedding_function=embedding_function, reset_db_indices=reset_db_indices)
    HumanLLM.check_init_class_db(force=True)

    # Allow some time for the WebSocket server to start
    time.sleep(1)  # Adjust if necessary

    # Initialize the default and premium LLMs
    # from langchain_groq import ChatGroq
    llmORchains_list = {
        "default_llm": ChatOpenAI(
            model_name=MODELS_CONFIG_LIST["basic_gpt" if "basic_gpt" in MODELS_CONFIG_LIST else "gpt"], cache=False,
            temperature=0.),
        "premium_llm": ChatOpenAI(model_name=MODELS_CONFIG_LIST["smart_gpt"], cache=False, temperature=0.),
        #"default_llm": ChatGroq(model_name=MODELS_CONFIG_LIST["basic_gpt" if "basic_gpt" in MODELS_CONFIG_LIST else "gpt"], cache=False),#, temperature=0.),
        #"premium_llm": ChatGroq(model_name=MODELS_CONFIG_LIST["code_gpt"], cache=False),#, temperature=0.),
        "3_majority_chain": create_Nmajority_chain(map_model_name=MODELS_CONFIG_LIST["basic_gpt"],
                                                   reduce_model_name=MODELS_CONFIG_LIST["basic_gpt"], num_models=3),
        "10_majority_chain": create_Nmajority_chain(map_model_name=MODELS_CONFIG_LIST["basic_gpt"],
                                                    reduce_model_name=MODELS_CONFIG_LIST["basic_gpt"], num_models=10)
    }

    # Set the documents to test/validate as a list of environments
    documents=[{ 'id':"cf0d353c-b43b-4a79-88f9-42c2c84cf75e",
                'title':"Complex QA and language models hybrid architectures, Survey",
            'context':"This paper reviews the state-of-the-art of language models architectures and strategies for 'complex' question-answering (QA, CQA, CPS) with a focus on hybridization. Large Language Models (LLM) are good at leveraging public data on standard problems but once you want to tackle more specific complex questions or problems (e.g. How does the concept of personal freedom vary between different cultures ? What is the best mix of power generation methods to reduce climate change ?) you may need specific architecture, knowledge, skills, methods, sensitive data protection, explainability, human approval and versatile feedback... Recent projects like ChatGPT and GALACTICA have allowed non-specialists to grasp the great potential as well as the equally strong limitations of LLM in complex QA. In this paper, we start by reviewing required skills and evaluation techniques. We integrate findings from the robust community edited research papers BIG, BLOOM and HELM which open source, benchmark and analyze limits and challenges of LLM in terms of tasks complexity and strict evaluation on accuracy (e.g. fairness, robustness, toxicity, ...) as a baseline. We discuss some challenges associated with complex QA, including domain adaptation, decomposition and efficient multi-step QA, long form and non-factoid QA, safety and multi-sensitivity data protection, multimodal search, hallucinations, explainability and truthfulness, temporal reasoning. We analyze current solutions and promising research trends, using elements such as: hybrid LLM architectural patterns, training and prompting strategies, active human reinforcement learning supervised with AI, neuro-symbolic and structured knowledge grounding, program synthesis, iterated decomposition and others.",
            'target_file_path': "env/IR_CPS_TechSynthesis/document_embedding_analysis/output/arxiv/Complex QA and language models hybrid architectures Survey.json"},
            { 'id':"42252c6c-12f3-4edf-9045-8acd69bc3356",
                'title':"Macroeconomic Effects of Inflation Targeting A Survey of the Empirical  Literature",
            'context':"This paper surveys the empirical literature of inflation targeting. The main findings from our review are the following: there is robust empirical evidence that larger and more developed countries are more likely to adopt the IT regime; the introduction of this regime is conditional on previous disinflation, greater exchange rate flexibility, central bank independence, and higher level of financial development; the empirical evidence has failed to provide convincing evidence that IT itself may serve as an effective tool for stabilizing inflation expectations and for reducing inflation persistence; the empirical research focused on advanced economies has failed to provide convincing evidence on the beneficial effects of IT on inflation performance, while there is some evidence that the gains from the IT regime may have been more prevalent in the emerging market economies; there is not convincing evidence that IT is associated with either higher output growth or lower output variability; the empirical research suggests that IT may have differential effects on exchange-rate volatility in advanced economies versus EMEs; although the empirical evidence on the impact of IT on fiscal policy is quite limited, it supports the idea that IT indeed improves fiscal discipline; the empirical support to the proposition that IT is associated with lower disinflation costs seems to be rather weak. Therefore, the accumulated empirical literature implies that IT does not produce superior macroeconomic benefits in comparison with the alternative monetary strategies or, at most, they are quite modest.",
            'target_file_path': "env/IR_CPS_TechSynthesis/document_embedding_analysis/output/arxiv/Macroeconomic Effects of Inflation Targeting A Survey of the Empirical  Literature.json"}]

    envs_tech_synthesis = []
    for doc in documents:
        env = EnvironmentManager(env_type="techsynthesis", title=doc['title'], context=doc['context'],
                                 target_file_path=doc['target_file_path'], id=doc['id'],
                                 llm=llmORchains_list["default_llm"], embedding_model_name=embedding_function).get_environment()
        envs_tech_synthesis.append(env)

    problem_subdir = "IR_CPS_TechSynthesis" if 'saved_task' not in globals() else saved_task['type_tache']

    # TODO: to be set properly and added to run_planner arg test_environments={'tech_synthesis': envs_tech_synthesis, 'swe': envs_swe}
    # documents = [{
    #     "instance_id": "django__django-14855",
    #     "text": ....,
    #     "repo": "django/django",
    #     "base_commit": "475cffd1d64c690cdad16ede4d5e81985738ceb4",
    #     "problem_statement": "Wrong URL generated by get_admin_url for readonly field in custom Admin Site Description When a model containing a ForeignKey field is viewed (or edited) in a custom Admin Site, and that ForeignKey field is listed in readonly_fields, the url generated for the link is /admin/... instead of /custom-admin/.... This appears to be caused by the following line in django.contrib.admin.helpers get_admin_url: url = reverse(url_name, args=[quote(remote_obj.pk)]) Other parts of the admin use the current_app keyword parameter to identify the correct current name of the Admin Site. (See django.contrib.admin.options.ModelAdmin response_add as just one example) I have been able to correct this specific issue by replacing the above line with: url = reverse( url_name, args=[quote(remote_obj.pk)], current_app=self.model_admin.admin_site.name ) However, I don't know if there are any side effects and I have not yet run the full suite of tests on this. Mostly looking for feedback whether I'm on the right track.",
    #     "hints_text": '''Hey Ken, yes seems right. Good spot. Looks like this should have been part of b79088306513d5ed76d31ac40ab3c15f858946ea for #31181 (which was Django 3.2) ​here. However, I don't know if there are any side effects and I have not yet run the full suite of tests on this. Mostly looking for feedback whether I'm on the right track. I ran your suggestion against most of the usual suspects admin_* tests without issue so... Would you like to prepare a patch? Looks like setting up the test case is the most of it... Thanks! I'll be happy to try - but I'm not likely to be able to get to it before the weekend. (I don't know how "urgent" you consider it.) If it can sit that long, I'll see what I can do. (First "real patch" and all that - want to make sure I do it reasonably right.) Hey Ken. Super thanks! Since it's a bug in a new feature it's marked as release blocker and will be backported to Django 3.2. We'll target ​3.2.8, which is slated for the beginning of October. If it gets close to that and you've not had time we can pick it up. Reach out on the Forum if you'd like input at all. 🙂 Thanks! (And Welcome Aboard! ⛵️) Heyy folks, I wanted to assign the ticket to myself and fix the issue, instead it assigned the ownership to me. Apologies Changes ownership again. I found out that changes got accepted, sorry for the inconvenience caused. Hi Abhijith — just to confirm, according to the discussion Ken is currently working on this ticket, so let's give him a window to do that before re-assigning it. Thanks! (I think that's the conclusion you came to, but just double-checking so you don't both work on the same ticket at the same time.)'''
    # }]

    # envs_swe = []
    # for doc in documents:
    #     env = SWEBenchEnvironment(SWEProblem.parse_obj(doc))
    #     envs_swe.append(env)

    # envs_swe=None

    if 'saved_task' in globals():
        special_criteria["all#saved_task"] = saved_task
        if special_criteria[f"{saved_task['agent_name']}#num_parallel_inferences"] == 0:
            saved_task_get = saved_task.get('content', {})
            special_criteria[f"{saved_task['agent_name']}#num_parallel_inferences"] = saved_task_get.get('num_parallel_inferences', 2)
        match saved_task['agent_name']:
            case "TaskIdentificationAgent":
                automation = {
                    'taskreco': saved_task['before_after'],
                    'coder': None if not special_criteria['CodingAgent#auto_n_rounds'] else "full_auto",
                    'critic': None if not special_criteria['ValidationAgent#auto_n_rounds'] else "full_auto",
                    'capitalizer': None if not special_criteria[
                        'CapitalizationAgent#auto_n_rounds'] else "full_auto"
                }
            case "CodingAgent":
                automation = {
                    'taskreco': 'skip_once',
                    'coder': saved_task['before_after'],
                    'critic': None if not special_criteria['ValidationAgent#auto_n_rounds'] else "full_auto",
                    'capitalizer': None if not special_criteria[
                        'CapitalizationAgent#auto_n_rounds'] else "full_auto"
                }
            case "ValidationAgent":
                automation = {
                    'taskreco': 'skip_once',
                    'coder': 'skip_once',
                    'critic': saved_task['before_after'],
                    'capitalizer': None if not special_criteria[
                        'CapitalizationAgent#auto_n_rounds'] else "full_auto"
                }
            case "CapitalizationAgent":
                automation = {
                    'taskreco': 'skip_once',
                    'coder': 'skip_once',
                    'critic': 'skip_once',
                    'capitalizer': saved_task['before_after']
                }
            case _:
                automation = None
        print(f"automation: {automation}")
    else:
        special_criteria = None
    if not ('automatic' in globals()):
        automatic = None
    else:
        automation_global = "full_auto" if automatic else None

    # Run the planner agent
    envs_swe = None

    run_4agents_learning_loop_graph(default_llm_key="default_llm", # ALTERNATIVES: run_4agents_learning_loop, run_planner, run_4agents_learning_loop
                premium_llm_key="premium_llm",
                llmORchains_list=llmORchains_list,
                test_environments=envs_tech_synthesis,
                manual_validation_to_capitalize=False,
                problem_prompts_subdir="IR_CPS_TechSynthesis",  #SWE_Synthesis, IR_CPS_TechSynthesis
                max_coding_attempts=4,
                include_code=False,
                selected_successful_functions=[],
                selected_failed_functions=[],
                max_execution_time=2400,
                agtask_premium_llm_by_default=False,
                agtask_skip_rounds=0,  # Auto-test: 1
                agcoding_skip_rounds=0,  # Auto-test: 4
                agvalidation_skip_rounds=0,  # Auto-test: 4
                agcapitalize_skip_rounds=0,
                agcoding_num_parallel_inferences=1,
                agcoach_num_parallel_inferences=1,
                # functions_to_import=".*",
                functions_to_import=None,
                primitives_dir="primitives/generate_graph",
                special_criteria=special_criteria,
                automation=automation if 'automation' in globals() else automatic,
                model_choice={"coach": "default_llm", "coder": "premium_llm", "critic": "default_llm",
                              "capitalizer": "default_llm"},
                embedding_function=embedding_function)  # Auto-test: 0"""