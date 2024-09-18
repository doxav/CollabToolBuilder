import json
import sys

import optuna as opt
import time
import os
from langchain_openai import ChatOpenAI
from config import MODELS_CONFIG_LIST
from learn import EnvironmentManager, run_4agents_learning_loop
from optimisation.optuna_analysis import analysis
from utils.llm_utils import HumanLLMMonitor

# Set the class variable
HumanLLMMonitor.use_websocket = False

def definition_few_shots(trial, usr_msg: bool = False):
    """
    Define few-shot tags and separators for the trial.

    Args:
        trial (optuna.trial.Trial): The Optuna trial object.
        usr_msg (bool, optional): Flag to determine the format of few_shots_tags. Defaults to False.

    Returns:
        tuple: A tuple containing the few-shot tags and separators.
    """
    if usr_msg:
        few_shots_tags = []
    else:
        few_shots_tags = ""
    # Determine the number of few-shot tags (between 0 and 3)
    sources_list = ["learnt", "failed", "example"]

    for i in range(3):
        source = sources_list[i]
        num = trial.suggest_int(f"num_tag_{i+1}", 0, 5)
        if num > 0:
            output_format = trial.suggest_categorical(f"format_tag_{i+1}", ["Json", "Markdown"])
            # Define separators using Optuna
            global_prefix = trial.suggest_categorical("global_prefix_option", ["tasks: <<",
                                                                               "List of tasks: [[",
                                                                               "tasks: {{",
                                                                               "List of tasks: $$"
                                                                               ])
            if global_prefix[-2:] in ["[[", "$$"]:
                global_prefix = f"{global_prefix[:8]}{source} {global_prefix[8:]}"
            else:
                global_prefix = f"{source}{global_prefix}"
            global_suffix = f"{global_prefix[-2:]}\n".replace("<<", ">>").replace("[[", "]]").replace("{{", "}}")
            item_prefix = trial.suggest_categorical("item_prefix_option", ["\n|", "\n(", "\n-"])
            item_suffix = f"{item_prefix[-1:]}\n".replace("(", ")")

            tmp_separators = {
                "global_prefix": global_prefix,
                "global_suffix": global_suffix,
                "item_prefix": item_prefix,
                "item_suffix": item_suffix
            }

            template = None

            # Construct the dictionary for the few-shot tag
            few_shot_dict = {
                "sources": source,
                "num": num,
                "format": output_format,
                "sort_order": "random",
                "separators": tmp_separators
            }
            if template:
                few_shot_dict["template"] = template

            # Add the few-shot tag to the prompt
            if not isinstance(few_shots_tags, list):
                # Convert the dictionary to a JSON string with double quotes
                few_shot_json = json.dumps(few_shot_dict)
                few_shots_tags += f"few_shots: {few_shot_json}\n"
            else:
                few_shots_tags.append(few_shot_dict)

    return few_shots_tags



def definition_global_parameters(temperature : float = None, presence_penalty : float = None):
    """
    Define global parameters for the LLMs and environments.

    Args:
        temperature (float, optional): The temperature setting for the LLMs.
        presence_penalty (float, optional): The presence penalty setting for the LLMs.

    Returns:
        tuple: A tuple containing the LLM or chains list and the environments list.
    """
    llmORchains_list = {
        "default_llm": ChatOpenAI(model_name=MODELS_CONFIG_LIST["basic_gpt"], **{k: v for k, v in {"temperature": temperature, "presence_penalty": presence_penalty}.items() if v is not None}),
        "premium_llm": ChatOpenAI(model_name=MODELS_CONFIG_LIST["smart_gpt"], **{k: v for k, v in {"temperature": temperature, "presence_penalty": presence_penalty}.items() if v is not None}),
    }

    # Set the documents to test/validate as a list of environments
    documents = [{'id': "cf0d353c-b43b-4a79-88f9-42c2c84cf75e",
                  'title': "Complex QA and language models hybrid architectures, Survey",
                  'context': "This paper reviews the state-of-the-art of language models architectures and strategies for 'complex' question-answering (QA, CQA, CPS) with a focus on hybridization. Large Language Models (LLM) are good at leveraging public data on standard problems but once you want to tackle more specific complex questions or problems (e.g. How does the concept of personal freedom vary between different cultures ? What is the best mix of power generation methods to reduce climate change ?) you may need specific architecture, knowledge, skills, methods, sensitive data protection, explainability, human approval and versatile feedback... Recent projects like ChatGPT and GALACTICA have allowed non-specialists to grasp the great potential as well as the equally strong limitations of LLM in complex QA. In this paper, we start by reviewing required skills and evaluation techniques. We integrate findings from the robust community edited research papers BIG, BLOOM and HELM which open source, benchmark and analyze limits and challenges of LLM in terms of tasks complexity and strict evaluation on accuracy (e.g. fairness, robustness, toxicity, ...) as a baseline. We discuss some challenges associated with complex QA, including domain adaptation, decomposition and efficient multi-step QA, long form and non-factoid QA, safety and multi-sensitivity data protection, multimodal search, hallucinations, explainability and truthfulness, temporal reasoning. We analyze current solutions and promising research trends, using elements such as: hybrid LLM architectural patterns, training and prompting strategies, active human reinforcement learning supervised with AI, neuro-symbolic and structured knowledge grounding, program synthesis, iterated decomposition and others.",
                  'target_file_path': "env/IR_CPS_TechSynthesis/document_embedding_analysis/output/arxiv/Complex QA and language models hybrid architectures Survey.json"},
                 {'id': "42252c6c-12f3-4edf-9045-8acd69bc3356",
                  'title': "Macroeconomic Effects of Inflation Targeting A Survey of the Empirical  Literature",
                  'context': "This paper surveys the empirical literature of inflation targeting. The main findings from our review are the following: there is robust empirical evidence that larger and more developed countries are more likely to adopt the IT regime; the introduction of this regime is conditional on previous disinflation, greater exchange rate flexibility, central bank independence, and higher level of financial development; the empirical evidence has failed to provide convincing evidence that IT itself may serve as an effective tool for stabilizing inflation expectations and for reducing inflation persistence; the empirical research focused on advanced economies has failed to provide convincing evidence on the beneficial effects of IT on inflation performance, while there is some evidence that the gains from the IT regime may have been more prevalent in the emerging market economies; there is not convincing evidence that IT is associated with either higher output growth or lower output variability; the empirical research suggests that IT may have differential effects on exchange-rate volatility in advanced economies versus EMEs; although the empirical evidence on the impact of IT on fiscal policy is quite limited, it supports the idea that IT indeed improves fiscal discipline; the empirical support to the proposition that IT is associated with lower disinflation costs seems to be rather weak. Therefore, the accumulated empirical literature implies that IT does not produce superior macroeconomic benefits in comparison with the alternative monetary strategies or, at most, they are quite modest.",
                  'target_file_path': "env/IR_CPS_TechSynthesis/document_embedding_analysis/output/arxiv/Macroeconomic Effects of Inflation Targeting A Survey of the Empirical  Literature.json"}]

    envs = []
    for doc in documents:
        env = EnvironmentManager(env_type="techsynthesis", title=doc['title'], context=doc['context'],
                                target_file_path=doc['target_file_path'], id=doc['id']).get_environment()
        envs.append(env)

    return llmORchains_list, envs

def launch_run(default_llm_key : str = "default_llm", premium_llm_key : str = "premium_llm", problem_prompts_subdir : str = None, max_coding_attempts : int = 2, max_execution_time : int = 900,
               model_choice=None, optuna_opti : str = "coach", params_user_message : str = None, special_criteria : dict = None, name_exp : str = ""):
    """
    Launch the run with the specified parameters.

    Args:
        default_llm_key (str): The key for the default LLM.
        premium_llm_key (str): The key for the premium LLM.
        problem_prompts_subdir (str, optional): The subdirectory for problem prompts.
        max_coding_attempts (int, optional): The maximum number of coding attempts.
        max_execution_time (int, optional): The maximum execution time in seconds.
        model_choice (dict, optional): The model choices for different roles.
        optuna_opti (str, optional): The Optuna optimization target.
        params_user_message (str, optional): The criteria for evaluation.
        special_criteria (dict, optional): Special criteria for the run.
        name_exp (str, optional): The name of the experiment.

    Returns:
        Any: The performance result of the run.
    """
    if model_choice is None:
        model_choice = {"coach": "default", "coder": "premium_llm", "critic": "default_llm",
                        "capitalizer": "default_llm"}
    if special_criteria is None:
        llmORchains_list, envs = definition_global_parameters()
    else:
        llmORchains_list, envs = definition_global_parameters(special_criteria["temperature"], special_criteria["presence_penalty"])
        # Delete the presence_penalty from the special_criteria
        del special_criteria["presence_penalty"]
    performance = run_4agents_learning_loop(default_llm_key=default_llm_key,
                                            premium_llm_key=premium_llm_key,
                                            llmORchains_list=llmORchains_list,
                                            test_environments=envs,
                                            manual_validation_to_capitalize=False,
                                            problem_prompts_subdir=problem_prompts_subdir,
                                            max_coding_attempts=max_coding_attempts,
                                            include_code=False,
                                            selected_successful_functions=[],
                                            selected_failed_functions=[],
                                            agtask_premium_llm_by_default=True,
                                            max_execution_time=max_execution_time,
                                            agtask_skip_rounds=0,
                                            agcoding_skip_rounds=0,
                                            agvalidation_skip_rounds=0,
                                            agcapitalize_skip_rounds=0,
                                            model_choice=model_choice,
                                            optuna_opti=optuna_opti,
                                            params_user_message=params_user_message,
                                            special_criteria=special_criteria)
    print("Analysis...")
    analysis(name_exp)
    print("Analysis done.")
    return performance

def launch_study(objective, name_exp : str):
    """
    Launch the Optuna study with the specified objective and experiment name.

    Args:
        objective (callable): The objective function for the Optuna study.
        name_exp (str): The name of the experiment.
    """
    # get current folder
    current_folder = os.getcwd()

    # Define the directory and file path
    sqlite_dir = os.path.join(current_folder, "Optuna_db")
    sqlite_file = os.path.join(sqlite_dir, f"{name_exp}.db")

    # Create the directory if it does not exist
    if not os.path.exists(sqlite_dir):
        os.makedirs(sqlite_dir)

    # Create a study and optimize the objective function
    study = opt.create_study(direction="maximize", storage=f"sqlite:///{sqlite_file}", study_name=name_exp)
    study.optimize(objective, n_trials=200)

def global_main(objective, agent_name : str) :
    """
    Main function to launch the Optuna study.

    Args:
        objective (callable): The objective function for the Optuna study.
        agent_name (str): The name of the agent.
    """
    # Change directory to the location of this script if not already in the correct directory
    if os.path.basename(os.getcwd()) == "optimisation":
        os.chdir("../")
    # Recuperate name_exp from terminal argument:
    if len(sys.argv) > 1:
        name_exp = sys.argv[1]
    else :
        timestamp_xp = int(time.time())
        name_exp = f"xp_{agent_name}{timestamp_xp}"
    launch_study(lambda trial: objective(trial, name_exp), name_exp)

