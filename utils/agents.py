import sys, json, time, os, difflib, random, pickle, logging, re, ast
import socket, uuid, datetime
from typing import List, Dict
from utils.llm_utils import (
    apply_criteria_and_prepare_monitor_args, get_primitives,
    smart_print, smart_input, _visual_input
)
from env.env import Environment
from utils.file_utils import extract_functions_ast
from utils.human_llm import HumanLLM, HumanLLMConfig
from env.SWEBench.env import SWEBenchEnvironment

from langchain_core.messages.human import HumanMessage
from langchain_core.messages.system import SystemMessage

# Agent 1: Task Identification
class TaskIdentificationAgent:
    def __init__(
        self,
        default_llm_choice,
        envs: List[Environment],
        premium_llm_choice=None,
        problem_prompts_subdir=None,
        premium_llm_by_default=True,
        skip_rounds=0,
        llmORchains_list=None,
        automation=None,
        model_choice=None,
        criteria=None,
        params_user_message=None,
        temperature_min=0.,
        temperature_max=1.,
        num_parallel_inferences=1,
        agcoach_num_parallel_inferences=1,
        fixed_coach=False,
        special_criteria=None,
        primitives_dir=None
    ):
        self.logger = logging.getLogger(__name__)
        self.additional_check_list = None
        self.name = self.__class__.__name__
        self.criteria = criteria
        self.params_user_message = params_user_message
        self.temperature_min = temperature_min
        self.temperature_max = temperature_max
        self.envs = envs
        self.automation = automation
        self.model_choice = model_choice
        self.primitives_dir = primitives_dir
        self.problem_prompts_subdir =  "" if problem_prompts_subdir is None else problem_prompts_subdir + "/"
        
        kw_common_args = apply_criteria_and_prepare_monitor_args(self, special_criteria, locals())

        self.human_llm_identify_best_task = HumanLLM(**kw_common_args)
        self.human_llm_identify_best_task.primitives_dir = primitives_dir
        self.human_llm_identify_best_task.skip_rounds = skip_rounds
        self.human_llm_identify_best_task.problem_prompts_subdir =  self.problem_prompts_subdir
        if self.additional_check_list:
            for key, value in self.additional_check_list.items():
                self.human_llm_identify_best_task.add_manage_inference_check(key, value)
        if hasattr(self, 'recommendations_usage'):
            if self.recommendations_usage:
                self.human_llm_identify_best_task.add_manage_inference_check(
                    "Recommendations",
                    self.human_llm_identify_best_task.generate_best_improvement_suggestions
                )
        else:
            self.human_llm_identify_best_task.add_manage_inference_check(
                "Recommend Critics",
                self.human_llm_identify_best_task.generate_best_improvement_suggestions
            )

    def identify_best_task(self):
        # Prepare data
        envs_status = "\n".join([env.get_state() for env in self.envs])
        few_shots = HumanLLMConfig().manage_few_shot_examples(
            few_shots_params=self.params_user_message)
        # print(few_shots)
        self.human_llm_identify_best_task.user_message_few_shots = self.params_user_message

        primitives = extract_functions_ast("\n".join(get_primitives(self.primitives_dir)), include_docstring=True, return_string=True)
        successful_tasks = "\n".join(HumanLLMConfig().get_learnt_tasks())
        failed_tasks = "\n".join(HumanLLMConfig().get_failed_tasks())

        # User message template
        user_message_template = """
    {few_shots}
    - Existing code:[[[\n# helpers primitives:\n{primitives}\n# Successful tasks implemented:\n{successful_tasks}\n# failed tasks / not implemented:\n{failed_tasks}\n]]]
    - Current status of examples on which the task will be tested on: {envs_status}
    """

        # Create user_message
        user_message = user_message_template.format( few_shots=few_shots, envs_status=envs_status, primitives=primitives, successful_tasks=successful_tasks, failed_tasks=failed_tasks)

        if hasattr(self, 'log_user_message') and self.log_user_message:
            with open(self.log_user_message, "a") as f:
                f.write("Coach -- identify_best_task:<<\n" + user_message + "\n>>\n\n")

        original_stdout = sys.stdout
        sys.stdout = open('system_debug.txt', 'w')
        # print(self.problem_prompts_subdir + 'identify_best_task')
        sys.stdout=original_stdout

        original_stdout = sys.stdout
        sys.stdout = open('user_message.txt', 'w')
        # print(f"User message: {user_message}")
        sys.stdout=original_stdout

        task = self.human_llm_identify_best_task.invoke(
            system_prompt_template=self.problem_prompts_subdir + 'identify_best_task',
            user_message=user_message,
            return_message_content_only=False,
            model_choice=self.model_choice,
            stream_output=True
        )

        return task

# Agent 2: Code Task
class CodingAgent:
    def __init__(
        self,
        default_llm_choice,
        envs: List[Environment],
        premium_llm_choice=None,
        problem_prompts_subdir=None,
        db_collection_success="successful_tasks",
        db_collection_failed="failed_tasks",
        skip_rounds=0,
        llmORchains_list=None,
        automation=None,
        model_choice=None,
        special_criteria=None,
        temperature_min=0.,
        temperature_max=1.,
        num_parallel_inferences=1,
        primitives_dir=None
    ):
        #super().__init__(llm)
        self.logger = logging.getLogger(__name__)
        self.additional_check_list = None
        self.name = self.__class__.__name__
        self.envs = envs
        self.automation = automation
        self.model_choice = model_choice

        if hasattr(self, 'num_parallel_inferences') and self.num_parallel_inferences == 0:
            self.num_parallel_inferences = 2

        kw_common_args = apply_criteria_and_prepare_monitor_args(self, special_criteria, locals())

        self.human_llm_code_task = HumanLLM(**kw_common_args)
        self.human_llm_code_task.skip_rounds = skip_rounds
        self.human_llm_code_task.primitives_dir = primitives_dir
        self.human_llm_code_task.model_choice = model_choice
        self.human_llm_code_task.problem_prompts_subdir = "" if problem_prompts_subdir is None else problem_prompts_subdir + "/"
        self.human_llm_code_task.add_manage_inference_check(
            "Code Parsing",
            self.human_llm_code_task.parse_ai_generated_code
        )
        self.human_llm_code_task.add_manage_inference_check(
            "Run Tests",
            self.human_llm_code_task.run_tests_on_code
        )
        if hasattr(self, 'recommendations_usage'):
            if self.recommendations_usage:
                self.human_llm_code_task.add_manage_inference_check(
                    "Recommendations",
                    self.human_llm_code_task.generate_best_improvement_suggestions
                )
        else:
            self.human_llm_code_task.add_manage_inference_check(
                "Recommend Critics",
                self.human_llm_code_task.generate_best_improvement_suggestions
            )
        if self.additional_check_list:
            for key, value in self.additional_check_list.items():
                self.human_llm_code_task.add_manage_inference_check(key, value)


# Agent 3: Code Validation
class ValidationAgent:
    def __init__(
        self,
        default_llm_choice,
        envs: List[Environment],
        premium_llm_choice=None,
        skip_rounds=0,
        llmORchains_list=None,
        automation=None,
        model_choice=None,
        special_criteria=None
    ):
        self.logger = logging.getLogger(__name__)
        #super().__init__(llm)
        
        self.additional_check_list = None
        self.name = self.__class__.__name__

        kw_common_args = apply_criteria_and_prepare_monitor_args(self, special_criteria, locals())

        self.human_llm_validate_code = HumanLLM(**kw_common_args)
        self.human_llm_validate_code.skip_rounds = skip_rounds
        self.envs = envs
        self.automation = automation
        self.model_choice = model_choice
        if self.additional_check_list:
            for key, value in self.additional_check_list.items():
                self.human_llm_validate_code.add_manage_inference_check(key, value)

    def validate_code(
        self,
        code,
        no_runtime_error,
        exec_result,
        task=None,
        human_evaluation_required=False,
        scores=None,
        env_states=None
    ):
        # Prepare data for placeholders
        runtime_errors = "no runtime errors at execution" if no_runtime_error else f"runtime errors at execution: {exec_result}"
        envs_status = "\n".join(env_states)
        human_evaluation = ''
        if human_evaluation_required:
            human_evaluation = smart_input(f"""
************\n
{code}\n
************\n
CODE ABOVE EXECUTED with result: {runtime_errors}\n
****\nSystem may not efficiently evaluate what is produced by the code, please add your evaluation of the result (or hit enter): """, agent_name=self.name, message_type="VALIDATION_INFO")

        # Define user_message template
        user_message_template = """
Task: <<{task}>>

Code: <<{code}>>

Code execution returned: <<{runtime_errors}>>

Execution result returned by exec command of code provided: <<{exec_result}>>

Human evaluation of the result: <<{human_evaluation}>>

Performance scores: <<{scores}>>

New environment status of examples on which the task has been tested on: <<{envs_status}>>
"""

        # Create user_message by replacing placeholders
        user_message = user_message_template.format(
            task=task,
            code=code,
            runtime_errors=runtime_errors,
            exec_result=exec_result,
            human_evaluation=human_evaluation,
            scores=scores,
            envs_status=envs_status
        )

        if hasattr(self, 'log_user_message') and self.log_user_message:
            with open(self.log_user_message, "a") as f:
                f.write("Validation -- validate_code:<<\n" + user_message + "\n>>\n\n")

        code_validation = self.human_llm_validate_code.invoke(
            system_prompt_template='validate_code',
            user_message=user_message,
            return_message_content_only=False,
            model_choice=self.model_choice
        )
        return code_validation

# Agent 4: Code Capitalization
class CapitalizationAgent:
    def __init__(
        self,
        default_llm_choice,
        premium_llm_choice=None,
        db_collection_success="successful_tasks",
        db_collection_failed="failed_tasks",
        db_embedding_function=None,
        db_perist_directory=None,
        skip_rounds=0,
        llmORchains_list=None,
        automation=None,
        model_choice=None,
        problem_prompts_subdir=None,
        special_criteria=None
    ):
        self.logger = logging.getLogger(__name__)
        self.additional_check_list = None
        self.name = self.__class__.__name__

        self.problem_prompts_subdir = "" if problem_prompts_subdir is None else problem_prompts_subdir + "/"

        if special_criteria is not None and 'CapitalizationAgent#replace_if_exists_function' in special_criteria:
            self.replace_if_exists_function = special_criteria['CapitalizationAgent#replace_if_exists_function']
            del special_criteria['CapitalizationAgent#replace_if_exists_function']

        kw_common_args = apply_criteria_and_prepare_monitor_args(self, special_criteria, locals())

        self.learnt_tasks_repository: Dict[str, str] = {}
        self.failed_tasks_repository: Dict[str, str] = {}
        self.human_llm_generate_function_description = HumanLLM(**kw_common_args)
        self.human_llm_generate_function_description.skip_rounds = skip_rounds
        self.automation = automation
        self.model_choice = model_choice
        if self.additional_check_list:
            for key, value in self.additional_check_list.items():
                self.human_llm_generate_function_description.add_manage_inference_check(key, value)

    def capitalize_successful_tasks(self, task_description: str, parsed_code: str) -> None:
        self.logger.info('Starting capitalize_successful_tasks')
        
        self.human_llm_generate_function_description.task_parameters = {'task_description' : task_description, 'parsed_code' : parsed_code}
        print(f"DEBUG CACA : {hasattr(self, 'saved_task')}, {self.automation}, {self.automation in ['before', 'after']}")
        if hasattr(self.human_llm_generate_function_description, 'saved_task') and self.automation in ['before', 'after']:
            content = self.human_llm_generate_function_description.saved_task.get('content', {})
            print(f"task parameters : {content.get('task_parameters', {})}")
            task_description = content.get("task_parameters", {}).get("task_description", task_description)
            parsed_code = content.get("task_parameters", {}).get("parsed_code", parsed_code)
            self.human_llm_generate_function_description.task_parameters = {
                'task_description': task_description,
                'parsed_code': parsed_code
            }

        function_name = parsed_code.get(
            "main_function_name",
            parsed_code.get("main_function", {}).get("name", "unknown")
        )
        query = {
            "query": {
                "term": {
                    "metadata.main_function_name.keyword": function_name
                }
            }
        }
        response = self.human_llm_generate_function_description.config.common_vectordb.elastic_client.search(
            index=HumanLLMConfig().db_learnt_tasks.config.collection_name,
            body=query
        )

        if response['hits']['total']['value'] > 0:
            for hit in response['hits']['hits']:
                doc_id = hit["_id"]
                self.human_llm_generate_function_description.config.common_vectordb.elastic_client.delete(
                    index=HumanLLMConfig().db_learnt_tasks.config.collection_name,
                    id=doc_id
                )
                self.logger.info(f"Document deleted : {doc_id}")

        if self.problem_prompts_subdir == "Anomalies/" or self.problem_prompts_subdir == "pipeline_synthesis/":
            pipeline_file_path = os.path.join("pipelines/pipelines", function_name + ".py")
            tool_description = str(self.generate_tool_description(function_name, parsed_code["program_code"]))
            self.learnt_tasks_repository[function_name] = [tool_description, parsed_code["program_code"]]
            # print last added task
            smart_print(
                f"************ Last added task ************\n{function_name}\n************************".replace("\\n",
                                                                                                                "\n"),
                self.name,
                "capitalize_successful_tasks SUCCESS",
                optional=True
            )
        else:
            function_name = parsed_code.get("main_function_name", parsed_code.get("main_function", {}).get("name", "unknown"))
            # save function program_code in a file under the functions directory and add to the function signature the generated dosctring
            function_file_path = os.path.join("functions", function_name + ".py")
            tool_description = str(self.generate_tool_description(function_name, parsed_code["program_code"]))
            self.learnt_tasks_repository[function_name] = [tool_description, parsed_code["program_code"]]
            # print last added task
            smart_print(
                f"************ Last added task ************\n{function_name}\n************************".replace("\\n",
                                                                                                                "\n"),
                self.name,
                "capitalize_successful_tasks SUCCESS",
                optional=True
            )

        if self.problem_prompts_subdir == "Anomalies/" or self.problem_prompts_subdir == "pipeline_synthesis/":
            if os.path.exists(pipeline_file_path):
                smart_print(
                    f"Pipeline file {pipeline_file_path} already exists, please provide a new name for the pipeline.",
                    self.name,
                    "capitalize_successful_tasks WARNING"
                )
                if self.automation:
                    i = random.randint(0, 1000)
                    pipeline_file_path = os.path.join("pipelines/pipelines", self.name + f"_{i}.py")
                else:
                    pipeline_file_path = os.path.join("pipelines/pipelines", smart_input("New pipeline name: ") + ".py")

        # check if the function file already exists, if yes, ask the user a new name
        else:
            if os.path.exists(function_file_path):
                smart_print(
                    f"Function file {function_file_path} already exists, please provide a new name for the function.",
                    self.name,
                    "capitalize_successful_tasks WARNING"
                )
                if self.automation:
                    # generate an id based on the current time and a random number
                    id = datetime.datetime.now().strftime("%Y%m%d%H%M%S") + "_" + str(random.randint(0, 1000))
                    function_file_path = os.path.join("functions", self.name + f"_{id}.py")
                else:
                    function_file_path = os.path.join(
                        "functions",
                        smart_input(
                            "This function already exists, please provide a new function name: ",
                            message_type="VALIDATION_INFO"
                        ) + ".py"
                    )

        if self.problem_prompts_subdir == "Anomalies/" or self.problem_prompts_subdir == "pipeline_synthesis/":
            new_path = pipeline_file_path
        else:
            new_path = function_file_path

        with open(new_path, "w") as function_file:
            # use regex to extract the docstring from tool_description
            docstring_pattern = re.compile(r'(""".*?""")', re.DOTALL)
            docstring_matches = docstring_pattern.findall(tool_description)
            docstring = docstring_matches[0] if docstring_matches else f'"""{tool_description}"""'
            # use regex to add docstring to the function parsed_code["main_function_name"] after the def line in parsed_code["program_code"]
            parsed_code["program_code"] = re.sub(r"(def " + function_name + "\(.*?\):)", r'\1\n    ' + docstring, parsed_code["program_code"], count=1)
            function_file.write(parsed_code["program_code"])

        # Open file in VSCode if necessary
        """
        if self.optuna_opti is None and is_vscode_installed():
            smart_print("Please modify and save the file in VSCode (Ctrl + W) when ready.", self.name,
                        "capitalize_successful_tasks INSTRUCTIONS")
            subprocess.run(["code", "--wait", new_path])
        """

        # Serialize entry for logging
        serialized_entry = json.dumps(
            {
                "time": datetime.datetime.now().isoformat(),
                (
                    "class_name"
                    if (self.problem_prompts_subdir == "Anomalies/" or self.problem_prompts_subdir == "pipeline_synthesis/")
                    else "main_function_name"): function_name,
                "program_code": parsed_code["program_code"],
                "tool_description": tool_description,
                "task_description": task_description,
            },
            default=lambda o: o.__dict__ if hasattr(o, '__dict__') else str(o)
        )

        if not os.path.exists("pickle"):
            os.makedirs("pickle")
        
        with open('pickle/results.pkl', 'wb') as f:
            pickle.dump(serialized_entry, f)

        print('Adding learnt task')

        # Add to vector database with tags
        tags = {"host": f"{socket.gethostname()}-{uuid.getnode()}", "step_id": int(HumanLLMConfig().step_id), ("class_name" if (self.problem_prompts_subdir == "Anomalies/" or self.problem_prompts_subdir == "pipeline_synthesis/") else "main_function_name"): function_name,}
        HumanLLMConfig().add_learnt_task(serialized_entry, tags)

    def capitalize_failed_tasks(self, task_description: str, parsed_code: str) -> None:
        is_anomaly = self.problem_prompts_subdir == "Anomalies/" or self.problem_prompts_subdir == "pipeline_synthesis/"
        name_key = "class_name" if is_anomaly else "main_function_name"
        if parsed_code:
            task_name = parsed_code.get(name_key,
                                        "replace this text with a descriptive name of the class" if is_anomaly else "replace this text with a descriptive name of the function")
        else:
            if False or not self.automation:  # TODO: temporary disabled, find the logic to fix this or if not required
                task_name = smart_input(
                    f"CONFIG Please provide a name for the {'pipeline' if is_anomaly else 'function'}:\n {task_description}",
                    "CapitalizationAgent",
                    message_type="Capitalization_info"
                ).strip()
            else:
                task_name = f"{task_description[:500]}"
        if False and not self.automation or is_anomaly:  # TODO: temporary disabled, find the logic to fix this or if not required
            task_name = _visual_input(task_name)
            task_description_refined = _visual_input(task_description)
        else:
            task_description_refined = task_description

        # Store task description
        self.failed_tasks_repository[task_name] = task_description_refined
        smart_print(
            f"************ Last added failed task ************\n{task_name}\n************************".replace("\\n",
                                                                                                               "\n"),
            self.name,
            "capitalize_failed_tasks CAPITALIZE FAIL",
            optional=True
        )

        # Serialize entry
        serialized_entry = json.dumps({
            "time": datetime.datetime.now().isoformat(),
            name_key: task_name,
            "program_code": parsed_code.get("program_code", None) if parsed_code else None,
            "task_description": task_description,
            "task_description_refined": task_description_refined,
        }, default=lambda o: o.__dict__ if hasattr(o, '__dict__') else str(o))

        # Log entry into the common vector database with tags
        tags = {
            "host": self.human_llm_generate_function_description.get_host_id(),
            "step_id": HumanLLMConfig().step_id,
        }
        HumanLLMConfig().add_failed_task(serialized_entry, tags)

    def process_results(self, results, task_type, selected_functions, repository, metadata_key, include_code_flag):
        smart_print(f"************ Retrieving {task_type} tasks from database - LIST:", self.name,
                    "retrieve_saved_tasks_in_db DATABASE ACCESS", optional=True)
        id = 0
        for result in results:
            id += 1
            task_data = json.loads(result.page_content)
            name_key = "class_name" if (
                    self.problem_prompts_subdir == "Anomalies/" or self.problem_prompts_subdir == "pipeline_synthesis/") else "main_function_name"
            if name_key not in task_data:
                task_data[name_key] = ""

            task_name = task_data.get(name_key)
            smart_print(
                f"{id}: {task_type} {name_key}:{task_name} time:{task_data['time']} host:{result.metadata['host']}",
                self.name, "retrieve_saved_tasks_in_db DATABASE ACCESS", optional=True)

        if selected_functions is None:
            selected_functions = smart_input(
                f"CONFIG Please select the {task_type} functions to load (separated by comma, 'all' for all, or hit enter for none): ").strip().replace(
                " ", "").lower().split(",")

        id = 0
        for result in results:
            id += 1
            if selected_functions and (str(id) not in selected_functions) and (selected_functions != ["all"]):
                continue
            task_data = json.loads(result.page_content)
            task_name = task_data.get(name_key)

            if task_name in repository:
                smart_print(
                    f"> {task_type} {task_name} already loaded. Skipping duplicates...",
                    self.name,
                    "retrieve_saved_tasks_in_db DATABASE ACCESS",
                    optional=True
                )
        tags = {
            "host": self.human_llm_generate_function_description.get_host_id(),
            "step_id": HumanLLMConfig().step_id,
        }

        # self.db_failed_tasks.add_texts(texts=[serialized_entry], metadatas=[tags])

    def generate_tool_description(self, program_name, program_code):
        user_message = f"MAIN FUNCTION: `{program_name}`\n\nFULL CODE:\n{program_code}"

        if hasattr(self, 'log_user_message') and self.log_user_message:
            with open(self.log_user_message, "a") as f:
                f.write("Capitalization -- generate_tool_description:<<\n" + user_message + "\n>>\n\n")

        tool_description = self.human_llm_generate_function_description.invoke(
            system_prompt_template="generate_function_description", user_message=user_message,
            return_message_content_only=True, model_choice=self.model_choice)
        return tool_description

# Agent 5: Planner
class PlannerAgent:
    def __init__(
        self,
        default_llm_choice,
        envs,
        premium_llm_choice=None,
        problem_prompts_subdir=None,
        skip_rounds=0,
        llmORchains_list=None,
        automation=None,
        model_choice=None,
        special_criteria=None,
        num_parallel_inferences=1,
        primitives_dir=None,
        system_prompt_path=None
    ):
        self.logger = logging.getLogger(__name__)
        # Define necessary class variables
        self.name = self.__class__.__name__
        self.last_user_message = None
        self.processed_codes = set()
        self.envs = envs
        self.problem_prompts_subdir = "" if problem_prompts_subdir is None else problem_prompts_subdir + "/"
        self.model_choice = model_choice
        self.automation = automation
        self.llm = default_llm_choice  # Assuming this is an LLM object or a callable function
        self.llmORchains_list = llmORchains_list or {}
        self.skip_rounds = skip_rounds
        self.max_autofix = 3  # TODO: re-use autofix from CodingAgent which is currently not available seperately as a function
        self.system_prompt_path = system_prompt_path

        # Initialize CallHumanLLMMonitor with a generic approach
        kw_common_args = apply_criteria_and_prepare_monitor_args(self, special_criteria, locals())

        self.human_llm_planner = HumanLLM(**kw_common_args)
        self.coding_agent = CodingAgent(
            default_llm_choice=self.llm,
            primitives_dir=primitives_dir,
            envs=self.envs,
            premium_llm_choice=None,
            problem_prompts_subdir=self.problem_prompts_subdir,
            skip_rounds=self.skip_rounds,
            llmORchains_list=self.llmORchains_list,
            automation=self.automation,
            model_choice=self.model_choice,
            special_criteria=None,
            num_parallel_inferences=1
        )
        self.special_criteria = special_criteria
        self.primitives_dir = primitives_dir

    def plan(self, question: str, reuse_prompt=None):
        self.last_user_message = question

        # Retrieve learnt tasks (functions/code)
        self.logger.info("Getting the learnt tasks...")
        learnt_tasks = HumanLLMConfig().get_learnt_tasks(k=30)
        self.logger.info(f"Learnt_tasks: {learnt_tasks}")
        if not learnt_tasks:
            smart_print("No learnt tasks are available to answer the question.", agent_name=self.name)
            self.logger.info("No learnt tasks are available to answer the question.")
            return "no code available"

        # Prepare the code snippets string
        code_snippets = "\n\n".join(learnt_tasks)
        self.logger.info("Got the learnt tasks")

        prebuilt_code = ""
        for learnt_task in learnt_tasks:
            loaded = json.loads(learnt_task)
            prebuilt_code += loaded['program_code'] + "\n"

        # Prepare the prompt for the LLM
        prompt_path = reuse_prompt if reuse_prompt else self.system_prompt_path
        prompt_template = HumanLLMConfig().load_prompt_template(prompt_path) if prompt_path else (
            "You are a helpful assistant that selects the best functions to answer the user's question.\n"
            "Available code snippets:\n{code_snippets}\n\n"
            "Please reuse these functions to create a new function that solves the user query.\n"
            "If multiple functions do exactly the same thing, choose the most efficient one.\n"
            "Don't re write the existing functions but only provide a new function that re uses these functions. Also execute that function at the end\n Your new function should also take bot as parameter"
            "Assume that bot, problem, and env will be automatically set as global variables.\n"
            "Only provide python code. Don't include any introductory or explanatory text\n"
            # "Provide full code and also execute the selected function with bot parameter"
        )
        prompt = prompt_template.format(question=question, code_snippets=code_snippets)

        user_message_content = f"User's question:\n{question}"

        selected_code = self.human_llm_planner.invoke(original_input_messages=[SystemMessage(content=prompt), HumanMessage(content=user_message_content)]) #, automation=self.automation)
        if selected_code:
            selected_code = prebuilt_code + "\n" + selected_code

            result = self.execute_code_on_envs(selected_code)
            if result:
                parsed_code, success, exec_results, scores, states, total_execution_time = result
                if success:
                    # smart_print("The code was successfully executed on all environments.", agent_name=self.name)
                    self.logger.info(f"Program code: {parsed_code['program_code']}")
                    self.logger.info(f"Main function: {parsed_code['main_function']['name']}")
                    for env in self.envs:
                        no_runtime_error, exec_result, std_out_err = env.step(f"{prebuilt_code + parsed_code['program_code']}\n{parsed_code['main_function']['name']}(bot)")
                        # Smart print the output
                        smart_print(f"Response for document {env.id}: {exec_result}", agent_name=self.name)
                        if no_runtime_error:
                            smart_print(f"Executed successfully for environment {env.id}", agent_name=self.name)
                        else:
                            smart_print(f"Error in environment {env.id}: {exec_result}", agent_name=self.name)
                else:
                    smart_print("Failed to execute the code.", agent_name=self.name)
        else:
            smart_print("No code was selected by the LLM.", agent_name=self.name)

    def execute_code_on_envs(self, code_str):
        # Use the parse_ai_generated_code method to analyze the code
        parse_success, parsed_code_or_error = self.coding_agent.parse_ai_generated_code(
            message=code_str,
            required_bot_arg='bot',  # If your main function needs to accept 'bot' as an argument
            automatic_tests=False
        )

        if parse_success:
            parsed_code = parsed_code_or_error
            smart_print("Code parsed successfully", agent_name=self.name)
            return parsed_code, True, None, None, None, None
        else:
            error_message = parsed_code_or_error
            smart_print(f"Error during code parsing: {error_message}", agent_name=self.name)
            return None

    def call_llm_with_similar_method(self, prompt, use_premium_llm=False, temperature=0.2):
        """
        Calls the LLM inspired by the CallHumanLLM method, without using concurrent.futures or multiple inferences.
        """
        # Define the LLM function to use
        llm_function = self.llmORchains_list.get('premium_llm' if use_premium_llm else 'default_llm')

        if not llm_function:
            smart_print("No LLM is available to perform the call.", agent_name=self.name)
            return ""

        # Prepare the input messages for the LLM
        system_message = SystemMessage(content="")
        user_message = HumanMessage(content=prompt)
        llm_input_messages = [system_message, user_message]

        # Configure the LLM with the desired temperature
        llm = llm_function.with_config(configurable={"llm_temperature": temperature})

        # Call the LLM
        try:
            response = llm.invoke(llm_input_messages)
            llm_output = response.content if hasattr(response, 'content') else str(response)
            return llm_output.strip()
        except Exception as e:
            smart_print(f"Error during LLM call: {e}", agent_name=self.name)
            return ""
