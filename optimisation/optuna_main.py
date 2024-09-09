import optuna as opt
import time
import os

from langchain_openai import ChatOpenAI
from config import MODELS_CONFIG_LIST
from learn import EnvironmentManager, run_4agents_learning_loop


def definition_few_shots(trial):
    few_shots = ""
    few_shots_tags = trial.suggest_categorical("few_shots_tags", [True, False])
    if few_shots_tags:
        number_of_shots = trial.suggest_int("number_of_shots", 1, 5)
        # filter_tags = trial.suggest_categorical("filter_tags", "")
        ranking_tags = trial.suggest_categorical("ranking_tags",
                                                 ["by_score_asc", "by_date_asc", "mrr_asc", "cosine_asc", "random",
                                                  "accuracy", "relevance"])
        annotations = trial.suggest_categorical("annotations", ["fix", "delete", "approve", "variants"])
        summary = trial.suggest_categorical("summary", [True, False])
        format = trial.suggest_categorical("format", ["JSON", "Markdown", "Jinja2"])
        few_shots = f"few_shots: {{'num': {number_of_shots}, 'ranking_method': '{ranking_tags}', 'annotations': '{annotations}', 'summary': {summary}, 'format': '{format}'}}"
    return few_shots

def definition_global_parameters(temperature : float = None, presence_penalty : float = None):
    # Initialize the default and premium LLMs
    # default_llm = ChatOpenAI(model_name="gpt-3.5-turbo-1106") # gpt-4-1106-preview gpt-3.5-turbo-1106 model_name=model_name, temperature=temperature, request_timeout=request_timout
    # default_llm = create_Nmajority_chain(num_models=3)
    # premium_llm = ChatOpenAI(model_name="gpt-4o") # gpt-4-1106-preview gpt-3.5-turbo-1106 model_name=model_name, temperature=temperature, request_timeout=request_timout
    llmORchains_list = {
        "default_llm": ChatOpenAI(model_name=MODELS_CONFIG_LIST["basic_gpt"], **{k: v for k, v in {"temperature": temperature, "presence_penalty": presence_penalty}.items() if v is not None}),
        "premium_llm": ChatOpenAI(model_name=MODELS_CONFIG_LIST["smart_gpt"], **{k: v for k, v in {"temperature": temperature, "presence_penalty": presence_penalty}.items() if v is not None}),
        # "3_majority_chain": learn.create_Nmajority_chain(map_model_name=MODELS_CONFIG_LIST["gpt-3.5"], reduce_model_name=MODELS_CONFIG_LIST["gpt-3.5"] , num_models=3),
        # "10_majority_chain": learn.create_Nmajority_chain(map_model_name=MODELS_CONFIG_LIST["gpt-3.5"], reduce_model_name=MODELS_CONFIG_LIST["gpt-3.5"], num_models=10)
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
               model_choice=None, optuna_opti : str = "coach", criteria : str = None, special_criteria : dict = None):
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
                                           criteria=criteria,
                                           special_criteria=special_criteria)

    return performance

def launch_study(objective, name_exp : str):
    with open(f"Optuna_results/{name_exp}.txt", "w") as f:
        f.write("")
    # Wait for 10s
    time.sleep(10)
    # get current folder
    current_folder = os.getcwd()

    sqlite_file = os.path.join(current_folder, f"Optuna_db/{name_exp}.db")

    # Create a study and optimize the objective function
    study = opt.create_study(direction="maximize", storage=f"sqlite:///{sqlite_file}", study_name=name_exp)
    study.optimize(objective, n_trials=200)