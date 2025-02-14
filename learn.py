import openai
from config import *
from datetime import datetime
import os, uuid, re, json, socket, argparse, pickle
import inspect, pickle, random, string, time, logging
from env.env import EnvironmentManager
from env.IR_CPS_TechSynthesis.env import *
from env.SWEBench.env import *
from utils.constants import ELASTIC_DATABASE
from utils.human_llm import HumanLLM
from utils.llm_utils import (
    smart_print, smart_input,
    import_functions_from_directory,
    extract_function_code,
    parse_learn_question,
    get_success_value_in_text,
    get_highest_score_index
)
from utils.agents import (
    TaskIdentificationAgent,
    CodingAgent,
    ValidationAgent,
    CapitalizationAgent,
    PlannerAgent,
)

# Define Log format
LOG_FORMAT = ('%(levelname) -10s %(asctime)s %(name) -30s %(funcName) -35s %(lineno) -5d: %(message)s')
LOGGER = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, 
    format=LOG_FORMAT
)

# Main learning loop orchestration functions
def run_4agents_learning_loop(
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
    human_evaluation_required=False,
    date_start=None
):
    humanLLM = HumanLLM()
    scores = None

    # Definition of automation depending on the task given
    if functions_to_import:
        # Imports the functions with the regex pattern given from functions directory into the elastic database
        functions = import_functions_from_directory(functions_to_import)
        print("Imported functions:", type(functions))
        for function in functions.items():
            print(f"Function:<<<\n{function}\n>>>")
            # TODO: improve by re-using code from SWE which also import docstrings for descriptions
            serialized_entry = json.dumps(
                {
                    "time": datetime.now().isoformat(),
                    "main_function_name": function[0],
                    "program_code": function[1],
                    "tool_description": "",
                    "task_description": ""
                },
                default=lambda o: o.__dict__ if hasattr(o, '__dict__') else str(o)
            )
            tags = {
                "host": f"{socket.gethostname()}-{uuid.getnode()}",
                "step_id": humanLLM.step_id
            }
            print("Adding learnt task:", humanLLM.add_learnt_task(serialized_entry, tags))

    logging.info("Starting learning loop...")

    if params_user_message is None and automation is None:
        params_user_message = {
            'sources': ["learnt", "failed", "default"],
            'num': [2, 3, 2],
            'format': ["json", "Jinja2", "Markdown"]
        }

    time_end = time.time() + max_execution_time

    if special_criteria:
        if 'TaskIdentificationAgent#default_llm_choice' in special_criteria:
            default_llm_key = special_criteria['TaskIdentificationAgent#default_llm_choice']
            # Remove the key from the special criteria to avoid passing it to the agents
            del special_criteria['TaskIdentificationAgent#default_llm_choice']
            print(f"Special criteria: default_llm_choice set to {default_llm_key}")
        if 'CodingAgent#default_llm_choice' in special_criteria:
            default_llm_key = special_criteria['CodingAgent#default_llm_choice']
            # Remove the key from the special criteria to avoid passing it to the agents
            del special_criteria['CodingAgent#default_llm_choice']
            print(f"Special criteria: default_llm_choice set to {default_llm_key}")
        if 'ValidationAgent#default_llm_choice' in special_criteria:
            default_llm_key = special_criteria['ValidationAgent#default_llm_choice']
            # Remove the key from the special criteria to avoid passing it to the agents
            del special_criteria['ValidationAgent#default_llm_choice']
            print(f"Special criteria: default_llm_choice set to {default_llm_key}")
        if 'CapitalizationAgent#default_llm_choice' in special_criteria:
            default_llm_key = special_criteria['CapitalizationAgent#default_llm_choice']
            # Remove the key from the special criteria to avoid passing it to the agents
            del special_criteria['CapitalizationAgent#default_llm_choice']
            print(f"Special criteria: default_llm_choice set to {default_llm_key}")
    logging.info(f"Special criteria: {special_criteria}")
    logging.info(f"Automation: {automation}")
    humanLLM.special_criteria = special_criteria

    # Pour chaque agent, tester si saved_task['agent_name'] est égal a eux, si non => automation = 'skip_once', si oui => automation = saved_task['before_after']
    agent_taskreco: TaskIdentificationAgent = TaskIdentificationAgent(
        default_llm_key,
        test_environments,
        premium_llm_choice=premium_llm_key,
        problem_prompts_subdir=problem_prompts_subdir,
        premium_llm_by_default=agtask_premium_llm_by_default,
        skip_rounds=agtask_skip_rounds,
        llmORchains_list=llmORchains_list,
        automation=automation['taskreco'] if isinstance(automation,dict) else automation,
        model_choice=(
            model_choice['taskreco' if 'taskreco' in model_choice else 'coach']
            if type(model_choice) == dict
            else model_choice
        ),
        criteria=params_user_message,
        temperature_max=temperature_max,
        num_parallel_inferences=agcoach_num_parallel_inferences,
        fixed_coach=fixed_coach,
        primitives_dir=primitives_dir,
        special_criteria=special_criteria
    )

    agent_coding = CodingAgent(
        default_llm_key,
        test_environments,
        premium_llm_choice=premium_llm_key,
        problem_prompts_subdir=problem_prompts_subdir,
        skip_rounds=agcoding_skip_rounds,
        llmORchains_list=llmORchains_list,
        automation=automation['coder'] if isinstance(automation, dict) else automation,
        model_choice=(
            model_choice['coding' if 'coding' in model_choice else 'coder']
            if type(model_choice) == dict
            else model_choice
        ),
        special_criteria=special_criteria,
        num_parallel_inferences=agcoding_num_parallel_inferences,
        primitives_dir=primitives_dir
    )

    agent_validation = ValidationAgent(
        default_llm_key,
        test_environments,
        premium_llm_choice=premium_llm_key,
        skip_rounds=agvalidation_skip_rounds,
        llmORchains_list=llmORchains_list,
        automation=automation['critic'] if isinstance(automation, dict) else automation,
        model_choice=(
            model_choice['validation' if 'validation' in model_choice else 'critic']
            if type(model_choice) == dict else model_choice
        ),
        special_criteria=special_criteria
    )

    agent_capitalize = CapitalizationAgent(
        default_llm_key,
        premium_llm_choice=premium_llm_key,
        skip_rounds=agcapitalize_skip_rounds,
        problem_prompts_subdir=problem_prompts_subdir,
        llmORchains_list=llmORchains_list,
        automation=automation['capitalizer'] if isinstance(automation, dict) else automation,
        model_choice=(
            model_choice['capitalize' if 'capitalize' in model_choice else 'capitalizer']
            if type(model_choice) == dict
            else model_choice
        ),
        special_criteria=special_criteria
    )
    duration = datetime.now() - date_start

    seconds = duration.total_seconds()
    smart_print(str(max_execution_time - seconds), agent_taskreco.name, "time_end")

    #agent_capitalize.retrieve_saved_tasks_in_db(include_code=include_code, selected_successful_functions=selected_successful_functions, selected_failed_functions=selected_failed_functions)
    continue_identifying_tasks = True
    max_coding_attempts = 4 if not agent_coding.automation == "skip_once" else 1
    total_scores = []

    # Global learn loop
    while continue_identifying_tasks and time.time() < time_end:
        humanLLM.step_id = str(random.randint(0, 1000000))
        task = agent_taskreco.identify_best_task()

        # Handle multiple-tasks case
        if len(task) > 1:
            if agent_taskreco.human_llm_identify_best_task.selected_outputs and \
                len(agent_taskreco.human_llm_identify_best_task.selected_outputs) == 1:
                task = task[agent_taskreco.human_llm_identify_best_task.selected_outputs[0]]
            else:
                # list all tasks with their index and the 200 first characters of their content
                task_list = "Multiple task output, only one allowed - PLEASE SELECT:\n"
                for i, t in enumerate(task):
                    task_list += f"\n\nTask id :  {i}\n Content :\n{t.content[:200]}\n"
                smart_print(task_list, "orchestrate_agents", "TASK SELECTION")

                # get input from user with the index of the task to select, manage exceptions
                while True:
                    try:
                        id = (
                            1 
                            if agent_taskreco.human_llm_identify_best_task.automation
                            else int(
                                smart_input(
                                    "Enter the index of the task to select: ",
                                    "orchestrate_agents",
                                    "TASK SELECTION"
                                ).strip()
                            )
                        )
                        if id in range(len(task)):
                            task = task[id]
                            break
                        else:
                            raise Exception("Index out of range")
                    except Exception as e:
                        smart_print(f"Error: {e}\n\nEnter a valid index", "orchestrate_agents")
        else:
            task = task[0]

        smart_print(
            "Identified Task: " + task.content.replace("\\n", "\n"),
            "orchestrate_agents",
            "orchestrate_agents RESULT",
            optional=True
        )
        task_description = task.content

        # Extract potential score and state function code from the task
        if allow_custom_score_state_functions:
            # Extract current implementation of get_score and get_state from the first environment
            score_function_code = extract_function_code(
                task_description,
                'get_score',
                inspect.getsource(test_environments[0].get_score)
            )
            state_function_code = extract_function_code(
                task_description,
                'get_state',
                inspect.getsource(test_environments[0].get_state)
            )

            for env in test_environments:
                env.set_score_function(score_function_code)
                env.set_state_function(state_function_code)

        parsed_code, validation, scores = coding_and_validation_loop(
            agent_coding,
            agent_validation,
            task_description,
            max_coding_attempts,
            manual_validation_to_capitalize,
            automation=agent_coding.human_llm_code_task.automation,
            end_time=time_end,
            human_evaluation_required=human_evaluation_required
        )
        
        # TODO: Fusionner les capitalizations, en ajoutant un paramètre pour savoir si on doit capitaliser les tâches réussies ou échouées et adapter le prompt en conséquence pour déterminer automatiquement si on doit capitaliser les tâches réussies ou échouées dans le cas d'un goto
        if validation == "success" or \
            (
                agent_capitalize.human_llm_generate_function_description.automation in ["before", "after"] and \
                (
                    hasattr(agent_capitalize.human_llm_generate_function_description, 'saved_task') and \
                    agent_capitalize.human_llm_generate_function_description.saved_task['agent_name'] == "CapitalizationAgent"
                )
            ):
            agent_capitalize.capitalize_successful_tasks(task_description, parsed_code)
        else:
            if agent_capitalize.human_llm_generate_function_description.automation or \
                smart_input(
                    "Do you want to capitalize this try as a 'failed task' to avoid this task to be proposed as a next best task ? (yes/no): ",
                    "orchestrate_agents",
                    message_type="VALIDATION_INFO"
                ).strip().upper() in ["Y", "YES"]:
                agent_capitalize.capitalize_failed_tasks(task_description, parsed_code)

        if agent_capitalize.human_llm_generate_function_description.automation:
            continue_identifying_tasks = (
                True
                if agent_capitalize.human_llm_generate_function_description.automation == "full_auto"
                else continue_each_loop
            )
        else:
            answer = smart_input(
                "Do you want to:\n- search for a new task after reseting to empty documents (Y/YES) ?\n- search for a new task based based on the status of documents after applying the task you just validated (N/NO/Enter) ?\n- or just exit the program (E/EXIT) ?",
                "orchestrate_agents",
                message_type="VALIDATION_INFO"
            ).strip().upper()
            continue_identifying_tasks = False if answer in ["E", "EXIT"] else True
            if answer.upper() in ["Y", "YES"]:
                for env in test_environments:
                    env.rest()
            else:
                if parsed_code:
                    # Apply the code to the environments without restoring their state
                    _, _, exec_results, _, _, _ = agent_coding.run_tests_on_code(
                        message="",
                        parsed_code=parsed_code,
                        skip_already_processed=False,
                        restore_state=False,
                        custom_agent="orchestrate_agents",
                        output_id=0
                    )
                    # Optionally display the execution results for each environment
                    for env, result in zip(test_environments, exec_results):
                        smart_print(
                            f"Execution result in environment {env.id}: {result}",
                            "orchestrate_agents",
                            "Execution Result"
                        )
                else:
                    smart_print("No code to run.", "orchestrate_agents", "Execution Error")

        # Calculate the average score of the task, and return it with other statistics
        if scores:
            if scores['validated_scores'] is not None:
                validated_score_avg = 0
                for dic in scores['validated_scores']:
                    for i in dic:
                        validated_score_avg += dic[i]
                validated_score_avg /= len(scores['validated_scores'])
            else:
                validated_score_avg = 0
            total_score_weighted_with_stats = (
                    scores['percentage_no_runtime_error'] +
                    10 * scores['best_score_without_validation'] +
                    (20 * (1 + validated_score_avg) if scores['validated_scores'] else 0)
            )
            if total_score_weighted_with_stats > 0:
                print(
                    f"total_score_weighted_with_stats: {total_score_weighted_with_stats}; "
                    f"scores['percentage_no_runtime_error']: {scores['percentage_no_runtime_error']}; "
                    f"scores['best_score_without_validation']: {scores['best_score_without_validation']}; "
                    f"validated_score_avg: {validated_score_avg}"
                )
            # add total_score_weighted_with_stats to total_scores
            total_scores.append(total_score_weighted_with_stats)
            total_scores.append(0)

    # print status of: continue_identifying_tasks and time.time() < time_end
    print(
        f"continue_identifying_tasks: {continue_identifying_tasks}, "
        f"time.time() < time_end: {time.time() < time_end}, "
        f"time.time(): {time.time()}, time_end: {time_end}"
    )

    if return_array:
        return total_scores
    else:
        return max(total_scores)

def run_planner(*args, **kwargs):  # NEW VERSION
    humanLLM = HumanLLM()
    # Definition of automation depending on the task given
    if kwargs.get('functions_to_import') is not None:
        # Imports the functions with the regex pattern given from functions directory into the elastic database
        functions = import_functions_from_directory(kwargs.get('functions_to_import'))
        print("Imported functions:", type(functions))
        for function in functions.items():
            print(f"Function:<<<\n{function}\n>>>")
            # TODO: improve by re-using code from SWE which also import docstrings for descriptions
            serialized_entry = json.dumps(
                {
                    "time": datetime.now().isoformat(),
                    "main_function_name": function[0],
                    "program_code": function[1],
                    "tool_description": "",
                    "task_description": ""
                },
                default=lambda o: o.__dict__ if hasattr(o, '__dict__') else str(o)
            )
            tags = {
                "host": f"{socket.gethostname()}-{uuid.getnode()}",
                "step_id": humanLLM.step_id
            }
            print("Adding learnt task:", humanLLM.add_learnt_task(serialized_entry, tags))

    successful_tasks = humanLLM.get_learnt_tasks()
    successful_tasks_list = [task for task in successful_tasks]
    print(f"{len(successful_tasks)} successful tasks:<<<\n{successful_tasks_list}>>>")

    path_folder = "primitives/generate_primitives"
    folder_path = os.path.join(os.path.dirname(__file__), path_folder)
    # Utiliser os.listdir pour ne pas parcourir les sous-répertoires
    for file in os.listdir(folder_path):
        if file.endswith(".py"):
            file_path = os.path.join(folder_path, file)
            with open(file_path, "r") as f:
                code = f.read()
                serialized_entry = json.dumps(
                    {
                        "time": datetime.now().isoformat(),
                        "class_name": file.replace(".py", ""),
                        "program_code": code,
                        "tool_description": "",
                        "task_description": "",
                    },
                    default=lambda o: o.__dict__ if hasattr(o, '__dict__') else str(o)
                )
                successful_tasks_list.append(serialized_entry)

    smart_print(
        json.dumps(successful_tasks_list),
        "orchestrate_agents",
        "successful_tasks_list"
    )

    # Initialize WebSocket server if used
    if humanLLM.use_websocket and humanLLM.websocket_server is None:
        humanLLM.initialize_websocket_server()

    # Select the problem prompts subdirectory if not provided
    problem_prompts_subdir = kwargs.get('problem_prompts_subdir')
    if problem_prompts_subdir is None:
        # Get the list of subdirectories in the 'prompts' directory
        problem_prompts_subdirs = [
            name
            for name in os.listdir("prompts")
            if os.path.isdir(os.path.join("prompts", name))
        ]
        default_subdir = problem_prompts_subdirs[0] if problem_prompts_subdirs else ""
        choice = smart_input(
            "Enter a capital letter for subdirectory (leave empty for default): " + "; ".join(
                f"\n[{i}] {subdir}"
                for i, subdir in zip(string.ascii_uppercase, problem_prompts_subdirs)
            ) + " ?",
            "run_planner"
        )
        problem_prompts_subdir = (
            problem_prompts_subdirs[ord(choice) - 65]
            if choice and choice.isupper() and ord(choice) - 65 in range(len(problem_prompts_subdirs))
            else default_subdir
        )

    # Initialize the PlannerAgent with the correct parameters
    planner = PlannerAgent(
        default_llm_choice=kwargs.get('default_llm_key'),
        envs=kwargs.get('test_environments', [EnvironmentManager()]),
        premium_llm_choice=kwargs.get('premium_llm_key'),
        system_prompt_path=kwargs.get('problem_prompts_subdir', problem_prompts_subdir),
        automation=kwargs.get('automation'),
        llmORchains_list=kwargs.get('llmORchains_list'),
        skip_rounds=kwargs.get('skip_rounds', 0),
        model_choice=kwargs.get('model_choice'),
        num_parallel_inferences=kwargs.get('agcoach_num_parallel_inferences', 1),
        primitives_dir=kwargs.get('primitives_dir', 'primitives'),
        special_criteria=kwargs.get('special_criteria')
    )

    question = ""
    while question.lower() not in ['q', 'quit', 'e', 'exit']:
        if "learn" in question.lower():
            problem_type, instance_ids = parse_learn_question(question)
            try:
                subdir_map = kwargs.get(
                    'subdir_map',
                    {
                        "swe": "SWE_Synthesis",
                        "swe_bench": "SWE_Synthesis",
                        "tech_synthesis": "IR_CPS_TechSynthesis",
                        "synthesis": "IR_CPS_TechSynthesis"
                    }
                )
                primitives_dir_map = kwargs.get(
                    'primitives_dir_map',
                    {
                        "swe": "primitives/swe_primitives",
                        "swe_bench": "primitives/swe_primitives",
                        "tech_synthesis": "primitives/tech_synthesis_primitives",
                        "synthesis": "primitives/tech_synthesis_primitives"
                    }
                )
                default_instance_ids = kwargs.get(
                    'default_instance_ids',
                    {
                        "swe": [
                            "django__django-14855", "scikit-learn__scikit-learn-25638"
                        ],
                        "swe_bench": [
                            "django__django-14855", "scikit-learn__scikit-learn-25638"
                        ]
                    }
                )

                problem_prompts_subdir = subdir_map[problem_type]
                primitives_dir = primitives_dir_map[problem_type]
                if instance_ids is not None and len(instance_ids) > 0:
                    test_environments = [
                        (
                            SWEBenchEnvironment
                            if problem_type.startswith('swe')
                            else EnvironmentManager
                        )(instance_id=instance_id)
                        for instance_id in instance_ids
                    ]
                elif default_instance_ids is not None and problem_type in default_instance_ids:
                    test_environments = [
                        (
                            SWEBenchEnvironment
                            if problem_type.startswith('swe')
                            else EnvironmentManager
                        )(instance_id=instance_id)
                        for instance_id in default_instance_ids[problem_type]
                    ]
                elif problem_type in kwargs['test_environments']:
                    test_environments = kwargs['test_environments'][problem_type]
                else:
                    smart_print(
                        f"Error: No default test environments available for problem type '{problem_type}'.",
                        agent_name='PlannerAgent'
                    )
                    return

                kwargs['test_environments'] = test_environments
                kwargs['problem_prompts_subdir'] = problem_prompts_subdir
                kwargs['primitives_dir'] = primitives_dir

                run_4agents_learning_loop(*args, **kwargs)
            except ValueError as e:
                smart_print(f"Error: {e}", agent_name='PlannerAgent')
        elif len(question) > 0:
            if planner.plan(question) == "no code available":
                smart_print("No code available. Use 'learn' command first.", agent_name='PlannerAgent')

        question = smart_input("Formulate your question (or q/e/quit/exit): ", agent_name='PlannerAgent').capitalize()
    return

def coding_and_validation_loop(
    agent_coding: CodingAgent,
    agent_validation: ValidationAgent,
    task_description,
    max_attempts,
    extra_manual_validation_to_capitalize=True,
    continue_even_if_successful=True,
    automation=None,
    end_time=None,
    human_evaluation_required=False
):
    humanLLM = HumanLLM()
    metadata = {'step_id': humanLLM.step_id}
    # Retrieve data
    previous_errors, _ = humanLLM.get_agent_data(
        agent_coding.name,
        'previous_errors',
        metadata_filter=metadata
    )
    previous_codes, _ = humanLLM.get_agent_data(
        agent_coding.name,
        'previous_codes',
        metadata_filter=metadata
    )
    previous_scores, _ = humanLLM.get_agent_data(
        agent_coding.name,
        'previous_scores',
        metadata_filter=metadata
    )
    unique_codes, _ = humanLLM.get_agent_data(
        agent_coding.name,
        'unique_codes',
        metadata_filter=metadata
    )
    unique_codes = set(unique_codes) if unique_codes != [{}] else set()
    successful_codes, _ = humanLLM.get_agent_data(
        agent_coding.name,
        'successful_codes',
        metadata_filter=metadata
    )
    all_results, _ = humanLLM.get_agent_data(
        agent_coding.name,
        'all_results',
        metadata_filter=metadata
    )

    for attempt in range(max_attempts):
        # Check for timeouts
        if end_time is not None and time.time() >= end_time:
            break

        # Filter out duplicates
        temp_errors, temp_codes, temp_scores = [], [], []
        for err, code, score in zip(previous_errors, previous_codes, previous_scores):
            if code and code not in unique_codes:
                unique_codes.add(code)
                temp_errors.append(err)
                temp_codes.append(code)
                temp_scores.append(score)
        previous_errors, previous_codes, previous_scores = temp_errors, temp_codes, temp_scores

        # Store updated data
        humanLLM.add_agent_data(agent_coding.name, 'previous_errors', previous_errors, metadata=metadata)
        humanLLM.add_agent_data(agent_coding.name, 'previous_codes', previous_codes, metadata=metadata)
        humanLLM.add_agent_data(agent_coding.name, 'previous_scores', previous_scores, metadata=metadata)
        humanLLM.add_agent_data(agent_coding.name, 'unique_codes', list(unique_codes), metadata=metadata)

        results = agent_coding.code_task_and_run_test(task_description)
        print("Results:", results)

        all_results.extend(results)
        humanLLM.add_agent_data(agent_coding.name, 'all_results', all_results, metadata=metadata)

        current_skip_rounds = agent_validation.human_llm_validate_code.skip_rounds
        for index, result in enumerate(results):
            parsed_code, no_runtime_error, exec_result, scores, env_states, _ = result
            # to prevent skip_rounds decreased multiple times by multiple calls of HumanLLMMonitor
            agent_validation.human_llm_validate_code.skip_rounds = current_skip_rounds
            smart_print(
                f"Generated code:\n{parsed_code['program_code']}\n*******\nOutput of code execution:\n{exec_result}\n".replace("\\n", "\n"),
                "coding_and_validation_loop",
                "coding_and_validation_loop RESULT"
            )

            validation_agent_feedback = agent_validation.validate_code(
                parsed_code["program_code"],
                no_runtime_error,
                exec_result,
                task=task_description,
                scores=scores,
                env_states=env_states,
                human_evaluation_required=human_evaluation_required
            )
            smart_print(
                "Agent validation 'feedback' currently only support 1 feedback",
                "coding_and_validation_loop",
                "coding_and_validation_loop WARNING"
            )

            validation_agent_feedback = validation_agent_feedback[0]
            afb = validation_agent_feedback.content.replace('\\n', '\n')
            smart_print(
                "#" * 20 + f"\nAgent validation feedback: {afb}",
                "coding_and_validation_loop",
                "coding_and_validation_loop RESULT"
            )

            if extra_manual_validation_to_capitalize:
                validated = (
                    smart_input(
                        "#" * 20 + f"\nADD THIS FUNCTION TO LIBRARY ? Please enter 'yes' if this a success and you want to add this function to library, "
                        f"'no' if this failed: ",
                    "coding_and_validation_loop"
                    ).lower() in ["yes", "y", True]
                )
            else:
                validated = get_success_value_in_text(afb) in ["yes", "y", True]

            if validated:
                successful_codes.append((parsed_code, validation_agent_feedback.content, scores))

            if index < len(previous_errors):
                previous_errors[index] = validation_agent_feedback
                previous_scores[index] = scores
                previous_codes[index] = parsed_code['program_code']
            else:
                previous_errors.append(validation_agent_feedback)
                previous_scores.append(scores)
                previous_codes.append(parsed_code['program_code'])

            # Store updated data
            humanLLM.add_agent_data(agent_coding.name, 'previous_errors', previous_errors, metadata=metadata)
            humanLLM.add_agent_data(agent_coding.name, 'previous_codes', previous_codes, metadata=metadata)
            humanLLM.add_agent_data(agent_coding.name, 'previous_scores', previous_scores, metadata=metadata)

            if validated:
                successful_codes.append((parsed_code, afb, scores))
                humanLLM.add_agent_data(agent_coding.name, 'successful_codes', successful_codes,
                                               metadata=metadata)

        if not automation and not successful_codes:
            stop = smart_input(
                "No successful code yet, do you want to stop coding attempts for this task (too hard) and try a new one ? (yes/no): ",
                "coding_and_validation_loop",
                "VALIDATION_INFO",
                optional=False
            ).lower() in ["yes", "y", True]
            if stop:
                break

        if not automation and successful_codes:
            stop = smart_input(
                "A successful code has been found, do you want to stop coding attempts for this task (performance is sufficient) and try a new one ? (yes/no): ",
                "coding_and_validation_loop",
                "VALIDATION_INFO",
                optional=False
            ).lower() in ["yes", "y", True]
            if stop:
                break

        if successful_codes and not continue_even_if_successful:
            break

        if not successful_codes and attempt == (max_attempts / 2) - 1:
            attempt = max_attempts - 1

        if attempt == max_attempts - 1:
            if not successful_codes:
                smart_print("No successful code yet. Stop this task.", "coding_and_validation_loop", "VALIDATION_INFO")
            else:
                smart_print("Max attempts reached. Trying a new task.", "coding_and_validation_loop", "VALIDATION_INFO")

    # Calculate metrics over all attempts
    percentage_no_runtime_error = (
        sum(
            1 for _, no_runtime_error, _, _, _, _ in all_results
            if no_runtime_error) / len(all_results)
        ) if all_results else 0

    best_score_without_validation = (
        max(
            max(sum(scores.values()) / len(scores) if len(scores) > 0 else 0 for scores in score_dict)
            for _, _, _, score_dict, _, _ in all_results
        )
    ) if len(all_results) > 0 else 0

    all_scores = {
        'percentage_no_runtime_error': percentage_no_runtime_error,
        'best_score_without_validation': best_score_without_validation,
        'validated_scores': None
    }

    # Second part: If there are successful codes, ask user to select one
    if successful_codes:
        if len(successful_codes) == 1:
            selected_code, _, scores = successful_codes[0]
            all_scores['validated_scores'] = scores
            return selected_code, "success", all_scores
        if current_skip_rounds <= 0:
            for i, (parsed_code, feedback, scores) in enumerate(successful_codes):
                smart_print(
                    f"\033Option {i + 1}:\nCode:\n{parsed_code['program_code']}\nFeedback: {feedback}\n\033[91mScore: {scores}\033[0m\n",
                    None,
                    "coding_and_validation_loop RESULT"
                )
            if automation:
                highest_score_index = get_highest_score_index(
                    [
                        scores
                        for _, _, scores in successful_codes
                    ],
                    mode='total'
                )
                selected_code, _, scores = successful_codes[highest_score_index]
                all_scores['validated_scores'] = scores
                return selected_code, "success", all_scores
            else:
                # Create a string with the indexes of the successful codes with their names and scores
                successful_codes_str = "\n".join(
                    f"Code **{i}**: {parsed_code['main_function'] if 'main_function' in parsed_code else parsed_code} - "
                    f"Score:{scores} - Code extract:{parsed_code['program_code'][:100]}"
                    for i, (parsed_code, _, scores) in enumerate(successful_codes)
                )
                selection = smart_input(
                    f"Several codes were successful. Please enter the number of the code you want to add to the library:\n{successful_codes_str}",
                    agent_name="CapitalizationAgent",
                    message_type="Capitalization_info"
                ).strip()

            if selection.isdigit() and 0 <= int(selection) <= len(successful_codes):
                selected_index = max(
                    0,
                    min(int(selection), len(successful_codes) - 1)
                )
                smart_print(
                    "Code validated successfully.",
                    "coding_and_validation_loop",
                    "coding_and_validation_loop RESULT"
                )
                selected_code, _, scores = successful_codes[selected_index]
                all_scores['validated_scores'] = scores
                return selected_code, "success", all_scores
            else:
                smart_print(
                    "Invalid selection or no selection made. Exiting without adding any code.",
                    "coding_and_validation_loop",
                    "coding_and_validation_loop WARNING"
                )
        # if in automatic mode, select the code with the highest score
        else:
            highest_score_index = get_highest_score_index(
                [
                    scores
                    for _, _, _, scores in successful_codes
                ],
                mode='total'
            )
            selected_code, _, scores = successful_codes[highest_score_index]
            all_scores['validated_scores'] = scores
            return selected_code, "success", all_scores

    # If no successful code was selected, return failure
    return None, "failed", all_scores

def prepare_configs(args):
    humanLLM = HumanLLM()
    humanLLM.use_websocket = True
    humanLLM.smart_input = smart_input
    humanLLM.smart_print = smart_print
    humanLLM.websocket_server_config.port = args.port
    humanLLM.websocket_server_config.secret = args.secret
    humanLLM.websocket_server_config.proxy_enabled = args.proxy
    
    if 'discord_webhook' in globals():
        humanLLM.discord_webhook = globals()['discord_webhook']

    if 'embedding_function' in globals():
        embedding_function = globals()['embedding_function']
    if embedding_function is None:
        embedding_function = "text-embedding-ada-002"
    humanLLM.vector_db_conf.embedding_function = embedding_function

    if not 'reset_db_indices' in locals():
        humanLLM.vector_db_conf.reset_indices = False

    humanLLM.vector_db_conf.db_type = ELASTIC_DATABASE
    if elastic_url_port:
        humanLLM.vector_db_conf.es_config.es_url = elastic_url_port
    if elastic_user:
        humanLLM.vector_db_conf.es_config.es_user = elastic_user
    if elastic_password:
        humanLLM.vector_db_conf.es_config.es_password = elastic_password
    
    openai.api_key = os.environ['OPENAI_API_KEY']
    if 'OPENAI_BASE_URL' in os.environ:
        openai.base_url = os.environ['OPENAI_BASE_URL']

    if args.pickle_name and os.path.exists(f'pickle/{args.pickle_name}.pkl'):
        with open(f'pickle/{args.pickle_name}.pkl', 'rb') as f:
            variables_from_pickle = pickle.load(f)
            saved_task = variables_from_pickle.get('saved_task')
            saved_task['content'] = json.loads(saved_task['content'])
            automation = variables_from_pickle.get('automatic')
            unique_id = variables_from_pickle.get('unique_id')
            special_criteria = variables_from_pickle.get('special_criteria')

        # Suppression du fichier pickle après utilisation pour éviter les conflits lors des prochains lancements
        os.remove(f'pickle/{args.pickle_name}.pkl')
    
    # Initialize the WebSocket server with port autodetection and proxy
    unique_id = None
    if 'unique_id' in globals():
        unique_id = globals()['unique_id']

    if unique_id is None:
        unique_id = f"{socket.gethostname()}_{datetime.now().strftime('%d-%m-%Y-%H-%M-%S')}"
    if unique_id is not False and humanLLM.vector_db_conf.unique_collection_id is None:
        humanLLM.vector_db_conf.set_unique_collection_id(unique_id)

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

    if not ('automation' in globals()):
        automation = None
    else:
        automation = globals()["automation"]
    humanLLM.special_criteria = special_criteria
    humanLLM.automation = automation

    humanLLM.initialize()
    return humanLLM


if __name__ == "__main__":
    # Handle command line arguments
    parser = argparse.ArgumentParser(description="Run the learning loop with optional WebSocket settings")
    parser.add_argument("--port", type=int, default=6789, help="Optional port for WebSocket server")
    parser.add_argument("--secret", action='store_true', help="Optional secret for WebSocket URL")
    parser.add_argument("--proxy", action='store_true', help="Start a proxy via localtunnel if available")
    parser.add_argument("--pickle_name", type=str, help="Optional pickle file name")
    args = parser.parse_args()

    
    humanLLM = prepare_configs(args)

    # Initialize HumanLLMMonitor databases
    humanLLM._check_and_init_vector_db()

    # Initialize the default and premium LLMs
    # from langchain_groq import ChatGroq
    llmORchains_list = humanLLM.get_llmORchains_list()

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
        env = EnvironmentManager(
            env_type="techsynthesis",
            title=doc['title'],
            context=doc['context'],
            target_file_path=doc['target_file_path'],
            id=doc['id'],
            llm=llmORchains_list["default_llm"],
            embedding_model_name=humanLLM.vector_db_conf.embedding_function
        ).get_environment()
        envs_tech_synthesis.append(env)

    max_execution_time = 2400
    run_4agents_learning_loop(
        default_llm_key="default_llm", # ALTERNATIVES: run_4agents_learning_loop, run_planner
        premium_llm_key="premium_llm",
        llmORchains_list=llmORchains_list,
        test_environments=envs_tech_synthesis,
        manual_validation_to_capitalize=False,
        problem_prompts_subdir="IR_CPS_TechSynthesis",  #SWE_Synthesis, IR_CPS_TechSynthesis
        max_coding_attempts=4,
        include_code=False,
        selected_successful_functions=[],
        selected_failed_functions=[],
        max_execution_time=max_execution_time,
        agtask_premium_llm_by_default=False,
        agtask_skip_rounds=0,  # Auto-test: 1
        agcoding_skip_rounds=0,  # Auto-test: 4
        agvalidation_skip_rounds=0,  # Auto-test: 4
        agcapitalize_skip_rounds=0,
        agcoding_num_parallel_inferences=2,
        agcoach_num_parallel_inferences=2,
        # functions_to_import=".*",
        functions_to_import=None,
        primitives_dir="primitives/generate_primitives",
        special_criteria=humanLLM.special_criteria,
        automation=humanLLM.automation,
        model_choice={
            "coach": "premium_llm",
            "coder": "coder_llm",
            "critic": "default_llm",
            "capitalizer": "default_llm"
        },
        embedding_function=humanLLM.vector_db_conf.embedding_function,
        date_start=datetime.now()
    )
