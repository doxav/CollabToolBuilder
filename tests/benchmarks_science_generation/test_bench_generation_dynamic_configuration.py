import sys
import argparse
import json
import os, config
from pathlib import Path
from time import sleep

import traceback


from tests.benchmarks_science_generation.benchmark.benchmark import StopCurrentIteration , run_costorm_human_llm, run_storm_human_llm, run_writehere_human_llm, run_agent_laboratory_human_llm, run_costorm_graph_human_llm

os.environ["OPENAI"] = os.environ["OPENAI_API_KEY"] = config.OPENAI_API_KEY
os.environ["ENCODER_API_TYPE"] = "openai"
os.environ["LITELLM_CACHE"] = "disabled"
os.environ["SearXNG"] = "http://127.0.0.1:8080"
os.environ["SEARXNG_API_KEY"] = ""


DEFAULT_ERROR_FOLDER = "benchmark/results"

#  HELPERS
def write_error(filename,human_llm_config,error):
    human_llm_config_name_list = ""
    if human_llm_config is not None:
        for human_llm_config in human_llm_config:
            human_llm_config_name = human_llm_config.get("agent_name", "") + "(" + "|".join([k for k in human_llm_config.get("dynamic_llm_config", {}).keys()]) + ")"
            human_llm_config_name_list += human_llm_config_name + ","
        human_llm_config_name_list = human_llm_config_name_list[:-1]  # Remove trailing comma
    with open(filename, "a") as f:
        f.write(f"Error occurred with config: {human_llm_config_name_list}\n")
        f.write("".join(traceback.format_exception(type(error), error, error.__traceback__)))

def is_debugging():
    return sys.gettrace() is not None


def create_sub_directory(file_path: str):
    """
    Create sub-directory for the given file path if it doesn't exist.
    """
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)


def create_config(agent_name,dynamic_config) -> dict:
    """
    Create a default configuration dictionary for human LLM.
    This is used to initialize the HumanLLMConfig.
    """
    return {
        "agent_name": agent_name,
        "dynamic_llm_config": dynamic_config,
        "automation": True,
    }

def alternative_human_llm_config(config:dict) -> list[dict]:
    """
    Generate a list of human LLM configurations based on the provided agent names.
    Each agent name corresponds to a specific configuration in CONFIG_HUMAN_LLM.
    """
    method = config.get("method_config_distribution", "all")
    list_agent = config.get("agent_list", [])
    config_human_llm = config.get("dynamic_config", [])
    res = []
    if method == "all":
        for current_config in config_human_llm:
            current_config_list = []
            for agent_name in list_agent:
                current_config_list.append(create_config(agent_name, current_config))
            res.append(current_config_list)
    elif method == "alternate":
        for current_config in config_human_llm:
            for agent_name in list_agent:
                current_config_list = []
                current_config_list.append(create_config(agent_name, current_config))
                for other_agent_name in list_agent:
                    if other_agent_name != agent_name:
                        current_config_list.append(create_config(other_agent_name, {}))
                res.append(current_config_list)
    else:
        raise ValueError(f"Unknown method: {method}")
    return res


def handle_callback(human_llm_list,article,step,article_change,metrics):
    """
    Callback function to handle the article and step.
    This can be used to log or process the article and step further.
    """
    if article_change:
        print(metrics)
        # StopCurrentIteration()

        
def run_costorm_dynamic_human_llm(topic_config, human_llm_parameters_list, error_log_file=f"{DEFAULT_ERROR_FOLDER}/costorm_errors.txt", *, max_steps: int = -1):
    # TODO: use a logger instead of writing to a file
    create_sub_directory(error_log_file)
    with open(error_log_file, "w", encoding="utf-8") as f:
        for topic_dict in topic_config:
            if "costorm_arguments" in topic_dict:
                topic_dict.update(topic_dict["costorm_arguments"])
            topic = topic_dict.pop("topic",None)
            if topic is None:
                f.write(f"Skipping topic_obj without 'topic': {topic_dict}")
                continue
            max_conv_turn = topic_dict.pop("max_conv_turn", 2)
            search_top_k = topic_dict.pop("search_top_k", 3)
            retrieve_top_k = topic_dict.pop("retrieve_top_k", 3)
            for human_llm_parameters in human_llm_parameters_list or [None]:
                try:

                    report, turns, metrics = run_costorm_human_llm(
                        topic,
                        max_conv_turn=max_conv_turn,
                        search_top_k=search_top_k,
                        retrieve_top_k=retrieve_top_k,
                        human_llm_parameters=human_llm_parameters,
                        callback=handle_callback,  # Callback function to handle the article and step
                        max_steps=max_steps,
                        **topic_dict
                    )
                except Exception as e:
                    write_error(error_log_file,human_llm_parameters,e)

def run_costormgraph_dynamic_human_llm(topic_config, human_llm_parameters_list, error_log_file=f"{DEFAULT_ERROR_FOLDER}/costormgraph_errors.txt", *, max_steps: int = -1):
    # TODO: use a logger instead of writing to a file
    create_sub_directory(error_log_file)
    with open(error_log_file, "w", encoding="utf-8") as f:
        for topic_dict in topic_config:
            if "costorm_arguments" in topic_dict:
                topic_dict.update(topic_dict["costorm_arguments"])
            topic = topic_dict.pop("topic",None)
            if topic is None:
                f.write(f"Skipping topic_obj without 'topic': {topic_dict}")
                continue
            total_conv_turn = topic_dict.pop("max_conv_turn", 2)
            for human_llm_parameters in human_llm_parameters_list or [None]:
                try:
                    report, turns, metrics = run_costorm_graph_human_llm(
                        topic,
                        total_conv_turn=total_conv_turn,
                        human_llm_parameters=human_llm_parameters,
                        callback=handle_callback,
                        max_steps=max_steps,
                        **topic_dict
                    )
                except Exception as e:
                    write_error(error_log_file,human_llm_parameters,e)

def run_storm_dynamic_human_llm(topic_config, human_llm_parameters_list, error_log_file=f"{DEFAULT_ERROR_FOLDER}/storm_errors.txt", *, max_steps: int = -1):
    # TODO: use a logger instead of writing to a file
    create_sub_directory(error_log_file)
    with open(error_log_file, "w", encoding="utf-8") as f:
        for topic_dict in topic_config:
            if "storm_arguments" in topic_dict:
                topic_dict.update(topic_dict["storm_arguments"])
            topic = topic_dict.pop("topic",None)
            if topic is None:
                f.write(f"Skipping topic_obj without 'topic': {topic_dict}")
                continue
            max_conv_turn = topic_dict.pop("max_conv_turn", 2)
            search_top_k = topic_dict.pop("search_top_k", 3)
            retrieve_top_k = topic_dict.pop("retrieve_top_k", 3)
            for human_llm_parameters in human_llm_parameters_list or [None]:
                try:

                    report, turns, metrics  = run_storm_human_llm(
                        topic,
                        max_conv_turn=max_conv_turn,
                        search_top_k=search_top_k,
                        retrieve_top_k=retrieve_top_k,
                        human_llm_parameters=human_llm_parameters,
                        callback=handle_callback,  # Callback function to handle the article and step
                        max_steps=max_steps,
                        **topic_dict
                    )
                except Exception as e:
                    write_error(error_log_file,human_llm_parameters,e)


def run_agentlaboratory_dynamic_human_llm(
    topic_config,
    human_llm_parameters_list,
    error_log_file="results/run_agentlaboratory_dynamic_human_llm.txt",
    *,
    max_steps: int = -1,
    agentlab_num_papers_to_write: int | None = None,
    agentlab_num_papers_lit_review: int | None = None,
    agentlab_focus: str | None = "literature_review",  # "all" or "literature_review"
):
    # TODO: use a logger instead of writing to a file
    create_sub_directory(error_log_file)


    with open(error_log_file, "w", encoding="utf-8") as f:
        for topic_dict in topic_config:
            # Extract additional agent arguments into the main dictionary
            topic_dict.update(topic_dict.pop("agentlaboratory_arguments", {}))
            
            topic = topic_dict.pop("topic",None)
            if topic is None:
                f.write(f"Skipping topic_obj without 'topic': {topic_dict}")
                continue

            # If no parameters provided, use [None] as default
            parameter_sets = human_llm_parameters_list or [None]

            for human_llm_parameters in parameter_sets:
                try:
                    report, turns, metrics  = run_agent_laboratory_human_llm(
                        topic,
                        focus_agent=agentlab_focus,
                        num_papers_to_write_override=agentlab_num_papers_to_write,
                        num_papers_lit_review_override=agentlab_num_papers_lit_review,
                        human_llm_parameters=human_llm_parameters,
                        callback=handle_callback,  # Callback function to handle the article and step
                        max_steps=max_steps,
                        **topic_dict
                    )
                    # Handle/report results as needed here
                except Exception as e:
                    write_error(error_log_file, human_llm_parameters, e)
                    stack_trace = "".join(traceback.format_exception(type(e), e, e.__traceback__))
                    print(f"\033[91mERROR\033[0m: {e} on topic {topic} with parameters {human_llm_parameters} - error log file: {error_log_file}\nStack trace:\n{stack_trace}")
            break
                
def run_writehere_dynamic_human_llm(topic_config, human_llm_parameters_list, error_log_file=f"{DEFAULT_ERROR_FOLDER}/writehere_errors.txt", *, max_steps: int = 10): 
    # TODO: use a logger instead of writing to a file
    create_sub_directory(error_log_file)
    with open(error_log_file, "w", encoding="utf-8") as f:
        for topic_dict in topic_config:
            if "writehere_arguments" in topic_dict:
                topic_dict.update(topic_dict["writehere_arguments"])
            topic = topic_dict.pop("topic",None)
            if topic is None:
                f.write(f"Skipping topic_obj without 'topic': {topic_dict}")
                continue
            intent = topic_dict.pop("intent", "Write a comprehensive and detailed article on the given topic.")
            engine_backend = topic_dict.pop("engine_backend", "SearXNG")
            for human_llm_parameters in human_llm_parameters_list or [[None]]:
                try:
                    report, turns, metrics  = run_writehere_human_llm(
                        topic,
                        intent=intent,
                        engine_backend=engine_backend,
                        human_llm_parameters=human_llm_parameters[0],
                        callback=handle_callback,  # Callback function to handle the article and step
                        max_steps=max_steps,
                        **topic_dict
                    )
                except Exception as e:
                    write_error(error_log_file,human_llm_parameters,e)
                
        
def _build_parser():
    """
    Build the argument parser for the script.
    """
    parser = argparse.ArgumentParser(description="Run dynamic human LLM tests.")
    parser.add_argument(
        "--topic-config",
        type=str,
        default="json/topics.json",
        help="Path to the topic configuration file.",
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["storm", "costorm", "costormgraph", "writehere", "agentlaboratory"],
        help="Method to use for benchmarking.",
    )
    parser.add_argument(
        "--dynamic-config",
        type=str,
        default=None,
        help="List of human LLM parameters to use for testing.",
    )
    # AgentLaboratory knobs (optional)
    parser.add_argument(
        "--agentlab-num-papers-to-write",
        type=int,
        default=None,
        help="AgentLab: override num_papers_to_write to bound runtime.",
    )
    parser.add_argument(
        "--agentlab-num-papers-lit-review",
        type=int,
        default=None,
        help="AgentLab: override num_papers_lit_review to bound runtime.",
    )
    parser.add_argument(
        "--agentlab-focus",
        type=str,
        choices=["all", "literature_review"],
        default="literature_review",
        help="AgentLab: apply HumanLLM to all phases or focus on literature_review only.",
    )
    parser.add_argument(
        "--agentlab-only-lit-review",
        action="store_true",
        help="AgentLab: run only the literature review phase (fast-path).",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=10,
        help="Max engine steps before stopping (useful for long WriteHERE runs).",
    )
    return parser.parse_args()    
                
def main():
    """
    Main function to run the tests.
    """
    args = _build_parser()
    
    #debug
    if is_debugging():
        args.method = "costorm"
        args.dynamic_config = "json/costorm_dynamic_config_trace.json"
        
    # Load topics
    topic_file = Path(args.topic_config)
    if topic_file.exists():
        topic_list = json.loads(topic_file.read_text(encoding="utf-8"))
    else:
        raise FileNotFoundError(f"Topic configuration file not found: {args.topic_config}")

    # If human_llm_parameters_list is provided, use it
    if args.method == "writehere":
        method_func = run_writehere_dynamic_human_llm
    elif args.method == "storm":
        method_func = run_storm_dynamic_human_llm
    elif args.method == "costorm":
        method_func = run_costorm_dynamic_human_llm
    elif args.method == "costormgraph":
        method_func = run_costormgraph_dynamic_human_llm
    elif args.method == "agentlaboratory":
        method_func = run_agentlaboratory_dynamic_human_llm
    else:
        raise ValueError(f"Unknown method: {args.method}")
    human_llm_parameters_list = None
    if args.dynamic_config is not None:
        config_file = Path(args.dynamic_config)
        if config_file.exists():
            dynamic_config = json.loads(config_file.read_text(encoding="utf-8"))
            human_llm_parameters_list = alternative_human_llm_config(dynamic_config)
        else:
            raise FileNotFoundError(f"Dynamic configuration file not found: {args.dynamic_config}")
    # pass max_steps through to the selected runner wrapper
    if args.method == "agentlaboratory":
        method_func(
            topic_list,
            human_llm_parameters_list,
            max_steps=args.max_steps,
            agentlab_num_papers_to_write=args.agentlab_num_papers_to_write,
            agentlab_num_papers_lit_review=args.agentlab_num_papers_lit_review,
            agentlab_focus=args.agentlab_focus,
        )
    else:
        method_func(topic_list, human_llm_parameters_list, max_steps=args.max_steps)
    
    
if __name__ == "__main__":
    main()
