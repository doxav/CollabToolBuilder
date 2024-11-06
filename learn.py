import inspect
import random
import string
import traceback
import contextlib
import ast
import types
import time

from config import *

import openai
from typing import Dict

from utils.llm_utils import UnifiedVectorDB, HumanLLMMonitor, _visual_input, smart_print, smart_input

import os
import io
import uuid
import re
import shutil
import hashlib

from datetime import datetime
import socket

from langchain_core.runnables import Runnable
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages.human import HumanMessage
from langchain_core.messages.ai import AIMessage
from langchain_core.messages.system import SystemMessage

from langchain_openai import ChatOpenAI

import json
import difflib

#set_llm_cache(SQLiteCache(database_path=".langchain_caching.db"))

openai.api_key = os.environ['OPENAI_API_KEY']
if 'OPENAI_BASE_URL' in os.environ: openai.base_url = os.environ['OPENAI_BASE_URL']

UnifiedVectorDB.db_type = "elasticsearch"  # "elasticsearch" "chroma"
UnifiedVectorDB.es_url = elastic_url_port
# UnifiedVectorDB.es_user = elastic_user
# UnifiedVectorDB.es_password = elastic_password
UnifiedVectorDB.OpenAI_embedding_function_name = "text-embedding-ada-002"  # "nomic-ai/nomic-embed-text-v1"

embedding_function = "intfloat/e5-base-v2"  # UnifiedVectorDB.OpenAI_embedding_function_name # e.g. "text-embedding-ada-002" for OpenAI or "intfloat/e5-base-v2" or other huggingface models - WARINING: if you change it, set reset_db_indices to True
reset_db_indices = False  # Set to True after changing embeddings

HumanLLMMonitor.use_websocket = True

class Environment:
    def __init__(self, temp_root_dir: str = None, data_dir: str = "data"):
        self.temp_root_dir = temp_root_dir if temp_root_dir else os.path.join(os.getcwd(), "temp")
        self.data_dir = data_dir
        self.current_temp_dir = None
        self.last_unique_id_backup = None
        # create "backups" directory were saved states will be stored
        if not os.path.exists(os.path.join(self.temp_root_dir, "backups")):
            os.makedirs(os.path.join(self.temp_root_dir, "backups"))
        # Environment.reset(self) # Moving reset to the first call to __init__ to avoid multiple reset when class is subclassed

    def reset(self, backup_previous_temp_dir=True):
        if self.current_temp_dir is not None:
            Environment.close(self, backup_previous_temp_dir)
        # create a new temp directory in temp_root_dir named with a uuid
        self.current_temp_dir = os.path.join(self.temp_root_dir, str(uuid.uuid4()))
        os.makedirs(self.current_temp_dir)
        # create a write only link to the data directory in the temp directory
        os.symlink(os.path.abspath(self.data_dir), os.path.join(self.current_temp_dir, "data"),
                   target_is_directory=True)

    def step(self, action_code, context={}):
        # Memorize current directory and switch to temporary directory
        current_dir = os.getcwd()
        os.chdir(self.current_temp_dir)

        # Ensure `result` is set in the code
        if not re.search(r'\bresult\s*=', action_code.strip().splitlines()[-1]):
            helper = "\nresult = locals().get('_', True)"
        else:
            helper = ""

        # Setup for capturing stdout and stderr
        stdout, stderr = io.StringIO(), io.StringIO()

        try:
            # Execute code with redirected stdout and stderr
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                exec(action_code + helper, context)
            # Safely evaluate and retrieve result
            exec_result = ast.literal_eval(repr(context.get('result', True)))
            no_runtime_error = True
        except Exception as e:
            # Format traceback and include captured output for clarity
            error_traceback = ''.join(traceback.format_exception(None, e, e.__traceback__))
            exec_result = f"Execution failed. Error: {e}\nTraceback:\n{error_traceback}\nStdout:\n{stdout.getvalue()}\nStderr:\n{stderr.getvalue()}"
            no_runtime_error = False
        finally:
            # Restore original directory
            os.chdir(current_dir)

        return no_runtime_error, exec_result

    def close(self, backup_previous_temp_dir=True):
        # move temp directory and its content including the data link to backups directory
        if backup_previous_temp_dir:
            shutil.move(self.current_temp_dir, os.path.join(self.temp_root_dir, "backups"))
        else:
            shutil.rmtree(self.current_temp_dir)

    def backup_state(self, unique_id: str = None):
        # copy all the temp directory (excluding data directory) into a folder named by unique_id into backups directory
        if unique_id is None:
            self.last_unique_id_backup = unique_id = str(uuid.uuid4())
        if self.get_state(unique_id) != self.get_state():
            shutil.copytree(self.current_temp_dir, os.path.join(self.temp_root_dir, "backups", unique_id),
                            ignore=shutil.ignore_patterns('data'))
        return unique_id

    def restore_state(self, unique_id):
        from_folder = os.path.join(self.temp_root_dir, "backups", unique_id)
        if not os.path.exists(from_folder): return "Restore state folder not found"
        if self.get_state(unique_id) != self.get_state():
            # copy all the content of the backup directory into the temp directory (excluding data directory)
            self.reset(backup_previous_temp_dir=False)
            # copy all the content of the backup directory into the temp directory which already contains the data directory
            shutil.copytree(from_folder, self.current_temp_dir, ignore=shutil.ignore_patterns('data'),
                            dirs_exist_ok=True)
            return "State restored"
        else:
            return "State identical to backup folder"

    def restore_last_state(self):
        if self.last_unique_id_backup:
            self.restore_state(self.last_unique_id_backup)
            return "Last state restored"
        else:
            return "No last state to restore"

    def get_state(self, unique_id=None, extended_comparison=False):
        # return a dictionary containing the content of the temp directory
        state = {}
        state_folder = self.current_temp_dir if unique_id is None else os.path.join(self.temp_root_dir, "backups",
                                                                                    unique_id)
        if not os.path.exists(state_folder):
            return "No state found"
        for root, dirs, files in os.walk(state_folder):
            for file in files:
                file_path = os.path.join(root, file)
                with open(file_path, "rb") as f:
                    file_content = f.read()
                if extended_comparison:
                    file_stat = os.stat(file_path)  # Capture file's metadata
                    state[file_path] = {"hash": hashlib.sha256(file_content).hexdigest(), "mtime": file_stat.st_mtime}
                else:
                    state[file_path] = hashlib.sha256(file_content).hexdigest()
        # convert the dictionary into a string useful for comparison and analysis by language models
        state_text = "Files directory content: " + (json.dumps(state) if state.keys().__len__() > 0 else "empty")
        return state_text

    def get_score(self):
        # indicate that automatic scoring is not set, then ask the user to provide a score between 0 and 1, we ensure that the score is a float between 0 and 1
        score = None
        while score is None:
            try:
                score = float(input("No automatic get_score set, please provide a score between 0 and 1: "))
                if score < 0 or score > 1:
                    score = None
            except ValueError:
                pass
        return {'score (top:1, worst:0)': score}

    def set_score_function(self, score_function_code: str):
        local_scope = {'self': self}
        func_name = re.search(r'def (\w+)\(', score_function_code).group(1)
        if validate_function_code(score_function_code, func_name, local_scope):
            self.get_score = types.MethodType(local_scope[func_name], self)

    def set_state_function(self, state_function_code: str):
        local_scope = {'self': self}
        func_name = re.search(r'def (\w+)\(', state_function_code).group(1)
        if validate_function_code(state_function_code, func_name, local_scope):
            self.get_state = types.MethodType(local_scope[func_name], self)


class EnvironmentManager:
    def __init__(self, env_type="default", **kwargs):
        if env_type == "techsynthesis":
            from env.IR_CPS_TechSynthesis.env import VoyagerEnvIR_CPS_TechSynthesis
            # pass to VoyagerEnvIR_CPS_TechSynthesis all the args from the EnvironmentManager
            self.env = VoyagerEnvIR_CPS_TechSynthesis(**kwargs)
        else:
            self.env = Environment()
        self.env.reset()

    def get_environment(self):
        return self.env

def apply_special_criteria(agent, special_criteria, available_locals=None):
    """
    Apply special criteria to the attributes and parameters of an agent.

    :param agent: The agent instance to modify.
    :param special_criteria: Dictionary containing the special criteria.
    :param available_locals: Dictionary containing the local variables of the caller function.
    :return: Dictionary of only the modified parameters.
    """
    # Dictionary to store only the modified parameters
    new_params = {}

    if special_criteria:
        if available_locals is None:
            # Use inspect to dynamically capture arguments
            frame = inspect.currentframe().f_back  # Go up one level
            _, _, _, values = inspect.getargvalues(frame)
            available_locals = values

        class_name = agent.__class__.__name__
        # Iterate through the criteria related to this class
        for key, value in special_criteria.items():
            if key in ['self', 'special_criteria']: continue
            if '#' in key:
                agent_name, key = key.split('#', 1)
                if agent_name != class_name and agent_name not in ['all', '']: continue
            if hasattr(agent, key):
                setattr(agent, key, value)
                print(f"Special criteria applied to {agent}'s class property: {key} = {value}")
            elif key in available_locals:
                new_params[key] = value
                print(f"Special criteria applicable to {agent}'s local variables: {key} = {value}")

    return new_params  # Return only new params

# Agent 1: Task Identification
class TaskIdentificationAgent():
    def __init__(self, default_llm_choice, envs: [Environment], premium_llm_choice=None, problem_prompts_subdir=None,
                 premium_llm_by_default=True, skip_rounds=0, llmORchains_list=None, optuna=None, model_choice=None,
                 criteria=None, params_user_message=None, temperature_min=0., temperature_max=1.,
                 num_parallel_inferences=1, fixed_coach=False, special_criteria=None):
        self.additional_check_list = None
        self.name = self.__class__.__name__
        self.problem_prompts_subdir = "" if problem_prompts_subdir is None else problem_prompts_subdir + "/"
        self.criteria = criteria
        self.params_user_message = params_user_message
        self.temperature_min = temperature_min
        self.temperature_max = temperature_max
        self.envs = envs
        self.optuna_opti = optuna
        self.model_choice = model_choice
        new_params = apply_special_criteria(self, special_criteria, locals()) #for key, value in new_params.items(): locals()[key] = value

        HumanLLMMonitor_args, local_vars = (set(inspect.signature(HumanLLMMonitor.__init__).parameters) - {'self'}), locals()
        kw_common_args = {param: local_vars[param] for param in HumanLLMMonitor_args if param in local_vars}
        print(f"{self.name} applicables special criteria's new_params:{new_params}\nHumanLLMMonitor params configured by {self.name}:{kw_common_args.keys()}\nHumanLLMMonitor params altered by special criteria for {self.name}:{set(new_params.keys()) & set(kw_common_args.keys())}")
        kw_common_args.update(new_params)

        #self.criteria = criteria, self.params_user_message = params_user_message, self.temperature_min = temperature_min, self.temperature_max = temperature_max
        self.human_llm_identify_best_task = HumanLLMMonitor(**kw_common_args)#,output_schema="identify_best_task.schema.py")
        self.human_llm_identify_best_task.skip_rounds = skip_rounds
        if self.additional_check_list:
            for key, value in self.additional_check_list.items():
                self.human_llm_identify_best_task.add_inference_check(key, value)

    def identify_best_task(self):
        # Prepare data
        envs_status = "\n".join([env.get_state() for env in self.envs])
        few_shots = self.human_llm_identify_best_task.get_multiple_few_shots(
            few_shots_params=self.params_user_message)
        self.human_llm_identify_best_task.user_message_few_shots = self.params_user_message

        # User message template
        user_message_template = """
    {few_shots}
    - Current status of examples on which the task will be tested on: {envs_status}
    """

        # Create user_message
        user_message = user_message_template.format(
            few_shots=few_shots,
            envs_status=envs_status
        )

        task = self.human_llm_identify_best_task.CallHumanLLM(
            system_prompt_template=self.problem_prompts_subdir + 'identify_best_task',
            user_message=user_message,
            return_message_content_only=False,
            optuna=self.optuna_opti,
            model_choice=self.model_choice,
            stream_output=True
        )
        return task

    def expand_criteria_aligned(self, criteria):
        """
        Étend les critères où les valeurs sont des listes en alignant les indices ensemble.
        Par exemple, si les critères sont :
        {
            'sources': ['learnt', 'failed'],
            'num': [3, 2],
            'format': ['Json', 'Markdown']
        }
        Cette méthode générera :
        [
            {'sources': 'learnt', 'num': 3, 'format': 'Json'},
            {'sources': 'failed', 'num': 2, 'format': 'Markdown'}
        ]
        """
        # Déterminer la longueur maximale parmi les listes
        lengths = [len(value) if isinstance(value, list) else 1 for value in criteria.values()]
        max_length = max(lengths)

        expanded_criteria = []

        for i in range(max_length):
            new_criteria = {}
            for key, value in criteria.items():
                if isinstance(value, list):
                    if i < len(value):
                        new_criteria[key] = value[i]
                    else:
                        # Si la liste est plus courte, utiliser le dernier élément
                        new_criteria[key] = value[-1]
                else:
                    new_criteria[key] = value
            expanded_criteria.append(new_criteria)
        return expanded_criteria


# Agent 2: Code Task
class CodingAgent():
    def __init__(self, default_llm_choice, envs: [Environment], premium_llm_choice=None, problem_prompts_subdir=None,
                 db_collection_success="successful_tasks", db_collection_failed="failed_tasks", skip_rounds=0,
                 llmORchains_list=None, optuna=None, model_choice=None, special_criteria=None, temperature_min=0.,
                 temperature_max=1., num_parallel_inferences=2):
        #super().__init__(llm)
        self.additional_check_list = None
        self.name = self.__class__.__name__
        self.max_autofix = None
        self.problem_prompts_subdir = "" if problem_prompts_subdir is None else problem_prompts_subdir+"/"
        self.envs = envs
        self.optuna_opti = optuna
        self.model_choice = model_choice
        self.processed_codes = set()
        new_params = apply_special_criteria(self, special_criteria, locals()) #for key, value in new_params.items(): locals()[key] = value
        self.problem_prompts_subdir = "" if problem_prompts_subdir is None else problem_prompts_subdir+"/"
        if special_criteria is not None:
            for key, value in special_criteria.items():
                setattr(self, key, value)
        self.last_user_message = None

        HumanLLMMonitor_args, local_vars = (set(inspect.signature(HumanLLMMonitor.__init__).parameters) - {'self'}), locals()
        kw_common_args = {param: local_vars[param] for param in HumanLLMMonitor_args if param in local_vars}
        print(f"{self.name} applicables special criteria's new_params:{new_params}\nHumanLLMMonitor params configured by {self.name}:{kw_common_args.keys()}\nHumanLLMMonitor params altered by special criteria for {self.name}:{set(new_params.keys()) & set(kw_common_args.keys())}")
        kw_common_args.update(new_params)

        self.human_llm_code_task = HumanLLMMonitor(**kw_common_args)
        self.human_llm_code_task.skip_rounds = skip_rounds
        self.human_llm_code_task.add_inference_check("Code Parsing", self.parse_ai_generated_code)
        self.human_llm_code_task.add_inference_check("Run Tests", self.run_tests_on_code)
        if self.additional_check_list:
            for key, value in self.additional_check_list.items():
                self.human_llm_code_task.add_inference_check(key, value)
        # "CodingAgent#additional_check_list": [(check_name, check_function), ...]

    def parse_ai_generated_code(self, message, language="py", retry=3, required_bot_arg=None, task_definition=None,
                                automatic_tests=True, output_id=None):
        import ast, time, re
        # Convert text to dictionary
        try:
            result_dict = ast.literal_eval(message)
        except Exception as e:
            result_dict = None
        # if result_dict is a dictionary and code exists, change message to result_dict["code"]
        if isinstance(result_dict, dict) and "MainFunction" in result_dict:
            code = ""
            if "HelperFunctions" in result_dict:
                for helper_function in result_dict["HelperFunctions"]:
                    code += helper_function["code"] + "\n"
            code += result_dict["MainFunction"]["code"]
        else:
            code = None
        error = None
        while retry > 0:
            try:
                if language == "py":  # Python case
                    if code is None:
                        # Match Python code blocks
                        code_pattern = re.compile(r"```python(.*?)```", re.DOTALL)
                        code = "\n".join(code_pattern.findall(message))

                    parsed = ast.parse(code)
                    functions = []
                    imports = []
                    classes = []
                    runnable_code = ""
                    
                    function_calls = set()  # To track all functions that are being called
                    function_defs = {}  # To track all function definitions
                    last_function = None  # To store the last defined function

                    if len(code) == 0 or len(list(parsed.body)) == 0:
                        return False, f"Error parsing action response (No Code found): {parsed.body}"

                    # Use ast.walk to go through all nodes in the AST
                    for node in ast.walk(parsed):
                        if isinstance(node, ast.FunctionDef):
                            node_type = "FunctionDef"
                            function_info = {
                                "name": node.name,
                                "type": node_type,
                                "body": ast.get_source_segment(code, node),
                                "params": [arg.arg for arg in node.args.args],
                            }
                            functions.append(function_info)
                            function_defs[node.name] = function_info  # Store function definition
                            last_function = function_info  # Track the last defined function

                        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                            # Record function calls by name
                            function_calls.add(node.func.id)

                        elif isinstance(node, ast.ClassDef):
                            node_type = "ClassDef"
                            class_info = {
                                "name": node.name + ".run_pipeline",
                                "type": node_type,
                                "body": ast.get_source_segment(code, node),
                            }
                            classes.append(class_info)
                            functions.append(class_info)  # Append class definition as a function as per original logic

                        elif isinstance(node, (ast.Expr, ast.Expression, ast.Assign)):
                            node_type = "Expression"
                            runnable_code += "\n" + ast.get_source_segment(code, node)

                        elif isinstance(node, (ast.ImportFrom, ast.Import)):
                            node_type = "ImportFrom"
                            imports.append(ast.get_source_segment(code, node))

                    # Now we need to find the main function as per the new logic:
                    # 1. Identify the root function (not called by any other function).
                    # 2. If no root exists, fallback to the last defined function.

                    # Identify root function (not called by any other function)
                    root_function = None
                    for function_name, function_info in function_defs.items():
                        if function_name not in function_calls:
                            root_function = function_info
                            break

                    # If no root function is found, use the last defined function
                    main_function = root_function if root_function else last_function

                    # Ensure we have a main function
                    assert main_function is not None, "No main function found."

                    # Check if required_bot_arg is present in the main function's parameters
                    if required_bot_arg:
                        assert required_bot_arg in main_function["params"], f"Main function {main_function['name']} must take an argument named '{required_bot_arg}'"

                    # Assemble the final program code
                    program_code = "\n".join(imports) + "\n"
                    program_code += "\n\n".join(function["body"] for function in functions)

                    if automatic_tests:
                        tests = [(env.id, main_function["name"] + "(bot)") for env in self.envs]
                        #[test for doc_id, test in parsed_code["tests"] if doc_id == env.id]
                    else:
                        tests = []
                        tests_pattern = re.compile(
                            r'\n#\s+[Dd]ocument #([a-z0-9-]+)\s+usage test[^\n]*\n([^\n]+)|{.*?"DocumentID":\s*"#(.*?)",\s*"FunctionCall":\s*"([^"]+)".*?}')
                        matches = tests_pattern.findall(task_definition if task_definition is not None else message)
                        for match in matches:
                            tests.append(match[:2] if match[0] != "" else match[2:])

                    for doc_id, test in tests:
                        try:
                            parsed_test = ast.parse(test)
                        except Exception as e:
                            return False, f"Error parsing code of Tests:\nERROR: {e}\nCODE: {test}"
                        # check if the test is a function call
                        if not isinstance(parsed_test.body[0], ast.Expr):
                            return False, f"Error parsing code of Tests (not a function call): {test}"
                else:
                    raise ValueError(f"Unsupported language in this version: {language}")

                self.parsed_code = {
                    "program_code": program_code,
                    "main_function": main_function,
                    "runnable_code": runnable_code,
                    "tests": tests,
                }
                return True, self.parsed_code

            except Exception as e:
                retry -= 1
                error = e
                time.sleep(0.1)

        self.parsed_code = f"Error parsing action response (before program execution): {error}"
        smart_print(f"CODE PARSING ERROR!!!\n{error}", self.name, "code_task_and_run_test SystemMessage", optional=False, column_id=output_id)
        return False, self.parsed_code

    def run_tests_on_code(self, message, parsed_code=None, skip_already_processed=False, output_id=None, restore_state=True, custom_agent=None):
        # Retrieve error_patches from HumanLLMMonitor
        metadata = {'step_id': HumanLLMMonitor.step_id}
        error_patches = HumanLLMMonitor.get_agent_data(self.name, 'error_patches', metadata_filter=metadata)
        error_patches = error_patches if error_patches else []

        primitives = self.get_primitives()
        parsed_code = getattr(self, 'parsed_code', None) if parsed_code is None else parsed_code
        current_skip_rounds = self.human_llm_code_task.skip_rounds  # save the initial value to align it for code validation

        if isinstance(parsed_code, dict) and parsed_code["program_code"] not in self.processed_codes:
            self.processed_codes.add(parsed_code["program_code"])
        elif skip_already_processed or not isinstance(parsed_code,
                                                      dict):  # This logic speedup because the same code should have the same score BUT only on the same problem & state
            return None

        #smart_print(f"************ Code parsed result ************\n{parsed_code}\n************************".replace("\\n", "\n"), self.name, "run_tests_on_code RESULT")

        # Set initial state before running tests or runnable code
        no_runtime_errors, exec_results = [], []
        # insert content of config.py into the code to ensure that the OPENAI_API_KEY is set
        with open("config.py", "r") as f:
            common_code = f.read() + "\n"
        # Common code part to be executed in all cases
        common_code += "\n".join(primitives) + "\n"

        # Run the code & tests in each environment
        max_autofix, decision_lower = None, None
        if hasattr(self, 'max_autofix'):
            max_autofix = self.max_autofix
        # Start the timer before the environments loop
        start_time = time.time()  # Added line
        for idx, env in enumerate(self.envs):
            env.backup_state()
            # If it's the first environment and the user chose not to fix, exit the loop
            if idx > 0 and decision_lower in ("no", "n", ""):
                smart_print(f"SKIPPING TEST: code error on first env, skipping test {idx}", custom_agent if custom_agent else self.name, "code_task_and_run_test SystemMessage", optional=False, column_id=output_id)
            else:
                # Determine tests to run or set default runnable code
                matching_tests = [test for doc_id, test in parsed_code["tests"] if doc_id == env.id] if parsed_code[
                    "tests"] else [parsed_code['runnable_code']]
                if not matching_tests:
                    no_runtime_error, exec_result = False, f"Error: no test found for given id {env.id}" if parsed_code[
                        "tests"] else f"Error: no runnable code found nor tests"
                else:
                    # Concatenate common code with program and tests or runnable code
                    code_to_run = common_code + parsed_code["program_code"] + "\n" + "\n".join(matching_tests)
                    smart_print("TESTING GENERATED CODE.....", custom_agent if custom_agent else self.name, "code_task_and_run_test SystemMessage", optional=False, column_id=output_id)
                    no_runtime_error, exec_result = env.step(code_to_run)
                    if no_runtime_error:
                        smart_print("TEST SUCCESSFUL", custom_agent if custom_agent else self.name, "code_task_and_run_test SystemMessage", optional=False, column_id=output_id)
                        smart_print(env.get_state(extended=True), custom_agent if custom_agent else self.name, "CODE_RESULT", optional=False, column_id=output_id)
                    while not no_runtime_error and current_skip_rounds <= 0:
                        smart_print("\033[31mCODE ERROR\033[0m: " + exec_result, custom_agent if custom_agent else self.name,
                                    "code_task_and_run_test SystemMessage", optional=False, column_id=output_id)
                        if self.optuna_opti and max_autofix is not None:
                            decision = "a" if max_autofix > 1 else "no"
                            if decision == "a":
                                max_autofix -= 1
                        elif self.optuna_opti:
                            decision = "n"
                        else:
                            smart_print(parsed_code["program_code"], custom_agent if custom_agent else self.name, f"Inference streaming output {output_id}", append=True, column_id=output_id, column_max=self.human_llm_code_task.num_parallel_inferences)
                            decision = smart_input(
                                f"ANSWER {output_id} Do you want to edit the code to fix the error (you will also be requested first) ? (yes/no) or try autofix by LLM (a): ",
                                custom_agent if custom_agent else self.name, "fix_error", column_id=output_id).strip()
                        decision_lower = decision.lower()
                        if decision_lower in ("no", "n", ""):
                            break
                        elif decision_lower in ["y", "yes"]:
                            # if in websocket, then get from self.human_llm_code_task.premium_llm
                            if HumanLLMMonitor.use_websocket:
                                edited_code = self.human_llm_code_task.temp_inference_result_content
                                print(f"EDITED CODE: {edited_code}")
                            else:
                                edited_code = _visual_input(parsed_code["program_code"], filetype="py", message_type="fix_error", agent_name=self.name, column_id=output_id)
                        else:
                            if decision_lower == "a":
                                # Do not use HumanLLMMonitor because no template is available for this specific case
                                smart_print("TRYING TO AUTOFIX ERROR", custom_agent if custom_agent else self.name, "fix_error", optional=True, column_id=output_id)
                                instructions = ""
                            else:
                                # User provided custom instructions
                                instructions = decision
                            fix_system_prompt = f"""You are a Python expert in code debugging.
                            You are provided with ERROR MESSAGE and the CODE TO FIX.
                            {instructions}
                            Reply with the full Python code fixed and ready to be executed without the triple quotes and python tags. You add comments in the code to explain your fix.
                            """
                            smart_print("ANALYZING ERROR.....", custom_agent if custom_agent else self.name, "code_task_and_run_test SystemMessage", optional=False, column_id=output_id)
                            help_for_fixing_system_prompt = f"""You help an LLM to fix code errors which has no access to documentation or internet by extracting key code information from the INFORMATION/DOCUMENTATION provided given CODE TO FIX and ERROR MESSAGE."""
                            error_with_info_to_help_prompt = f"ERROR MESSAGE:<<\n{exec_result}\n>>\n\nCODE TO FIX:<<\n{parsed_code['program_code']}\n>>\n\INFORMATION/DOCUMENTATION:<<\n{self.last_user_message}\n>>"
                            help_code_returned = self.human_llm_code_task.premium_llm.invoke([SystemMessage(content=help_for_fixing_system_prompt),HumanMessage(content=error_with_info_to_help_prompt)])
                            smart_print("ANALYSIS RECIEVED, GENERATING A FIX", custom_agent if custom_agent else self.name, "code_task_and_run_test SystemMessage", optional=False, column_id=output_id)
                            fix_description_prompt = f"ERROR MESSAGE:<<\n{exec_result}\n>>\n\nCODE TO FIX:<<\n{parsed_code['program_code']}\n>>\n\nHELPFUL INFORMATION:<<\n{getattr(help_code_returned,'content',help_code_returned)}\n>>"
                            edited_code_returned = self.human_llm_code_task.premium_llm.invoke([SystemMessage(content=fix_system_prompt),HumanMessage(content=fix_description_prompt)])
                            edited_code = str(getattr(edited_code_returned,'content',edited_code_returned))

                        # Before updating parsed_code, save the previous code
                        prev_code = parsed_code["program_code"]
                        # Run the edited code
                        code_to_run = common_code + edited_code + "\n" + "\n".join(matching_tests)
                        smart_print("TESTING UPDATED CODE.....", custom_agent if custom_agent else self.name, "code_task_and_run_test SystemMessage", optional=False, column_id=output_id)
                        no_runtime_error, exec_result = env.step(code_to_run)
                        # Update parsed_code if re-run is successful
                        parsed_code["program_code"] = edited_code
                        smart_print(f"# UPDATED **{'SUCCESFUL' if no_runtime_error else 'FAILED'}** CODE:\n{edited_code}", custom_agent if custom_agent else self.name, "UPDATED_CODE", optional=False, column_id=output_id)
                        # If no runtime error, store the error and diff
                        if no_runtime_error:
                            smart_print(env.get_state(extended=True), custom_agent if custom_agent else self.name, "CODE_RESULT", optional=False, column_id=output_id)
                            diff = difflib.unified_diff(prev_code.splitlines(), edited_code.splitlines(), lineterm='')
                            diff_text = '\n'.join(diff)
                            # Avoid duplicates: check if the error and diff combination already exists
                            if (exec_result, diff_text) not in error_patches:
                                error_patches.append((exec_result, diff_text))
                                # Store updated error_patches
                                HumanLLMMonitor.add_agent_data(self.name, 'error_patches', error_patches,
                                                               metadata=metadata)

            # Append the results for each environment
            no_runtime_errors.append(no_runtime_error)
            exec_results.append(exec_result)
        # Calculate total time taken  # Added line
        total_execution_time = time.time() - start_time

        # Return combined results
        result = (parsed_code, all(no_runtime_errors), exec_results, [env.get_score() for env in self.envs],
                  [env.get_state(extended=False) for env in self.envs], total_execution_time)

        if restore_state:
            for env in self.envs:
                env.restore_last_state()

        return result

    def get_primitives(self):
        primitives = []
        # Add the pipelines folder for the primitives

        if self.problem_prompts_subdir == "Anomalies/" or self.problem_prompts_subdir == "pipeline_synthesis/":
            path_folder = "pipelines/pipelines"
        else:
            path_folder = "primitives/generate_primitives"
        folder_path = os.path.join(os.path.dirname(__file__), path_folder)
        # Utiliser os.listdir pour ne pas parcourir les sous-répertoires
        for file in os.listdir(folder_path):
            if file.endswith(".py"):
                smart_print(f"File load: {file}", "CodingAgent", "Files loaded", optional=True)
                file_path = os.path.join(folder_path, file)
                with open(file_path, "r") as f:
                    primitives.append(f.read())
        return primitives

    def code_task_and_run_test(self, refined_task):
        import itertools

        def flatten_list_of_lists_of_lists(nested_list):
            # Flatten a list of lists of lists into a single list
            return list(itertools.chain.from_iterable(itertools.chain.from_iterable(nested_list)))

        # Retrieve data from HumanLLMMonitor
        metadata = {'step_id': HumanLLMMonitor.step_id}
        previous_errors = HumanLLMMonitor.get_agent_data(self.name, 'previous_errors', metadata_filter=metadata)
        previous_scores = HumanLLMMonitor.get_agent_data(self.name, 'previous_scores', metadata_filter=metadata)
        previous_codes = HumanLLMMonitor.get_agent_data(self.name, 'previous_codes', metadata_filter=metadata)
        error_patches = HumanLLMMonitor.get_agent_data(self.name, 'error_patches', metadata_filter=metadata)

        # Ensure variables are initialized
        previous_errors = previous_errors if previous_errors else []
        previous_scores = previous_scores if previous_scores else []
        previous_codes = previous_codes if previous_codes else []
        error_patches = flatten_list_of_lists_of_lists(error_patches) if error_patches else []

        env_states = "\n".join([env.get_state(extended=False) for env in self.envs])
        primitives = "\n".join(self.get_primitives())
        successful_tasks = "\n".join(HumanLLMMonitor.get_learnt_tasks())
        failed_tasks = "\n".join(HumanLLMMonitor.get_failed_tasks())
        validation_response_um = "\n".join(HumanLLMMonitor.get_validation_results())

        previous_attempts = ""
        for errors_list, scores_list, codes_list in zip(previous_errors, previous_scores, previous_codes):
            for err, score, code in zip(errors_list, scores_list, codes_list):
                previous_attempts += f"\n<<ATTEMPT FEEDBACK: {err.content}\nSCORE: {score}\nCODE: {code}>>\n"

        error_patches_str = ""
        for error_msg, diff_text in error_patches:
            error_patches_str += f"\n<<ERROR MESSAGE: {error_msg}\nFIX APPLIED (diff):\n{diff_text}>>\n"

        # Define data for template placeholders
        template_data = {
            "refined_task": refined_task,
            "env_states": env_states,
            "primitives": primitives,
            "successful_tasks": successful_tasks,
            "failed_tasks": failed_tasks,
            "validation_response_um": validation_response_um,
            "previous_attempts": previous_attempts,
            "error_patches_str": error_patches_str
        }

        # Load and format the user message from a file template
        user_message = HumanLLMMonitor.load_prompt("coding_agent_user_message_template", template_data=template_data, directory='prompts')

        # Set the formatted user message
        self.last_user_message = user_message

        current_skip_rounds = self.human_llm_code_task.skip_rounds  # save the initial value to align it for code validation
        # Initialisation des kwargs avec les paramètres requis
        kwargs = {
            "system_prompt_template": self.problem_prompts_subdir + "code_task",
            "user_message": user_message,
            "return_message_content_only": False,
            "stream_output": False,
            "optuna": self.optuna_opti,
            "model_choice": self.model_choice
        }

        # Ajouter temperature seulement si l'attribut temperature existe dans l'instance
        if hasattr(self, 'temperature'):
            kwargs["temperature_max"] = self.temperature

        # Appeler la méthode avec les arguments sous forme de **kwargs
        codes = self.human_llm_code_task.CallHumanLLM(**kwargs)
        results = []

        for index, code in enumerate(codes):
            # Get the proper check_results corresponding to the output_id (which is the index)
            check_results = self.human_llm_code_task.last_inference_check_results[index]

            code_parsing_success, parsed_code = check_results.get("Code Parsing", (False, None))
            if code_parsing_success and isinstance(parsed_code, dict):
                test_results = check_results.get("Run Tests", None)
                if test_results:
                    results.append(test_results)

        if len(results) > 1:
            # display the list of results with success, exception and code
            results_list = ""
            top_results, top_indice = 0, 1
            for id, result in enumerate(results):
                if result[1]:
                    results_list += f"{id}. \033[32mSUCCESS\033[0m / SCORE: {result[3]} / TIME: {result[5]}s / CODE: {result[0]['program_code'][:100]}\n"
                    temp = sum(result[3][i][j] for i in range(len(result[3])) for j in result[3][i]) / len(result[3])
                    if temp > top_results:
                        top_results = temp
                        top_indice = id
                else:
                    results_list += f"{id}. \033[31mFAILED\033[0m / SCORE: {result[3]} / TIME: {result[5]}s / EXCEPTION: {result[2][0][:100]} / CODE: {result[0]['program_code'][:100]}\n"

            # ask the user to select the code to keep
            if current_skip_rounds <= 0:
                if self.optuna_opti:
                    selected_code = f"{top_indice}"
                else:
                    selected_code = smart_input(
                        f"{{{''.join(results_list)}}} CODE SELECTION Please select the code to keep (separated by comma, none/n for none of these, or just hit enter to keep ALL): ",
                        self.name, "Scores").strip().replace(" ", "").lower().split(",")
            else:
                selected_code = [""]  # keep all if skip_rounds is not 0
            id = 0
            if selected_code in ["none", "n"]:
                results = []
            else:
                # keep only the selected code
                results = [result for id, result in enumerate(results) if
                           selected_code and (str(id) in selected_code or selected_code == [""])]

        return results


# Agent 3: Code Validation
class ValidationAgent():
    def __init__(self, default_llm_choice, envs: [Environment], premium_llm_choice=None, skip_rounds=0,
                 llmORchains_list=None, optuna=None, model_choice=None, special_criteria=None):
        #super().__init__(llm)
        self.additional_check_list = None
        self.name = self.__class__.__name__

        new_params = apply_special_criteria(self, special_criteria, locals()) #for key, value in new_params.items(): locals()[key] = value
        HumanLLMMonitor_args, local_vars = (set(inspect.signature(HumanLLMMonitor.__init__).parameters) - {'self'}), locals()
        kw_common_args = {param: local_vars[param] for param in HumanLLMMonitor_args if param in local_vars}
        print(f"{self.name} applicables special criteria's new_params:{new_params}\nHumanLLMMonitor params configured by {self.name}:{kw_common_args.keys()}\nHumanLLMMonitor params altered by special criteria for {self.name}:{set(new_params.keys()) & set(kw_common_args.keys())}")
        kw_common_args.update(new_params)

        self.human_llm_validate_code = HumanLLMMonitor(**kw_common_args)
        self.human_llm_validate_code.skip_rounds = skip_rounds
        self.envs = envs
        self.optuna_opti = optuna
        self.model_choice = model_choice
        if self.additional_check_list:
            for key, value in self.additional_check_list.items():
                self.human_llm_validate_code.add_inference_check(key, value)

    def validate_code(self, code, no_runtime_error, exec_result, task=None, human_evaluation_required=False,
                      scores=None, env_states=None):
        # Prepare data for placeholders
        runtime_errors = "no runtime errors at execution" if no_runtime_error else f"runtime errors at execution: {exec_result}"
        envs_status = "\n".join(env_states)
        human_evaluation = ''
        if human_evaluation_required:
            human_evaluation = smart_input(f"""
************
{code}
************
CODE ABOVE EXECUTED with result: {runtime_errors}
****\System may not efficiently evaluate what is produced by the code, please add your evaluation of the result (or hit enter): """)

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

        code_validation = self.human_llm_validate_code.CallHumanLLM(
            system_prompt_template='validate_code',
            user_message=user_message,
            return_message_content_only=False,
            optuna=self.optuna_opti,
            model_choice=self.model_choice
        )
        return code_validation


# Agent 4: Code Capitalization
class CapitalizationAgent:
    def __init__(self, default_llm_choice, premium_llm_choice=None, db_collection_success="successful_tasks",
                 db_collection_failed="failed_tasks", db_embedding_function=None, db_perist_directory=None,
                 skip_rounds=0, llmORchains_list=None, optuna=None, model_choice=None, problem_prompts_subdir=None, special_criteria=None):
        self.additional_check_list = None
        self.name = self.__class__.__name__

        self.problem_prompts_subdir = "" if problem_prompts_subdir is None else problem_prompts_subdir+"/"

        new_params = apply_special_criteria(self, special_criteria, locals()) #for key, value in new_params.items(): locals()[key] = value
        HumanLLMMonitor_args, local_vars = (set(inspect.signature(HumanLLMMonitor.__init__).parameters) - {'self'}), locals()
        kw_common_args = {param: local_vars[param] for param in HumanLLMMonitor_args if param in local_vars}
        print(f"{self.name} applicables special criteria's new_params:{new_params}\nHumanLLMMonitor params configured by {self.name}:{kw_common_args.keys()}\nHumanLLMMonitor params altered by special criteria for {self.name}:{set(new_params.keys()) & set(kw_common_args.keys())}")
        kw_common_args.update(new_params)

        self.learnt_tasks_repository: Dict[str, str] = {}
        self.failed_tasks_repository: Dict[str, str] = {}
        self.human_llm_generate_function_description = HumanLLMMonitor(**kw_common_args)
        self.human_llm_generate_function_description.skip_rounds = skip_rounds
        self.optuna_opti = optuna
        self.model_choice = model_choice
        if self.additional_check_list:
            for key, value in self.additional_check_list.items():
                self.human_llm_generate_function_description.add_inference_check(key, value)

    def capitalize_successful_tasks(self, task_description: str, parsed_code: str) -> None:
        import socket, uuid, datetime

        if self.problem_prompts_subdir == "Anomalies/" or self.problem_prompts_subdir == "pipeline_synthesis/":
            function_name = parsed_code.get("main_function_name",
                                            parsed_code.get("main_function", {}).get("name", "unknown"))
            pipeline_file_path = os.path.join("pipelines/pipelines", function_name + ".py")
            tool_description = str(self.generate_tool_description(function_name, parsed_code["program_code"]))
            self.learnt_tasks_repository[function_name] = [tool_description, parsed_code["program_code"]]
            # print last added task
            smart_print(
                f"************ Last added task ************\n{function_name}\n************************".replace("\\n",
                                                                                                                "\n"),
                self.name, "capitalize_successful_tasks SUCCESS", optional=True)
        else:
            function_name = parsed_code.get("main_function_name",
                                            parsed_code.get("main_function", {}).get("name", "unknown"))
            # save function program_code in a file under the functions directory and add to the function signature the generated dosctring
            function_file_path = os.path.join("functions", function_name + ".py")
            tool_description = str(self.generate_tool_description(function_name, parsed_code["program_code"]))
            self.learnt_tasks_repository[function_name] = [tool_description, parsed_code["program_code"]]
            # print last added task
            smart_print(
                f"************ Last added task ************\n{function_name}\n************************".replace("\\n",
                                                                                                                "\n"),
                self.name, "capitalize_successful_tasks SUCCESS", optional=True)

        if self.problem_prompts_subdir == "Anomalies/" or self.problem_prompts_subdir == "pipeline_synthesis/":
            if os.path.exists(pipeline_file_path):
                smart_print(
                    f"Pipeline file {pipeline_file_path} already exists, please provide a new name for the pipeline.",
                    self.name, "capitalize_successful_tasks WARNING")
                if self.optuna_opti:
                    i = random.randint(0, 1000)
                    pipeline_file_path = os.path.join("pipelines/pipelines", self.name + f"_{i}.py")
                else:
                    pipeline_file_path = os.path.join("pipelines/pipelines", smart_input("New pipeline name: ") + ".py")

        # check if the function file already exists, if yes, ask the user a new name
        else:
            if os.path.exists(function_file_path):
                smart_print(
                    f"Function file {function_file_path} already exists, please provide a new name for the function.",
                    self.name, "capitalize_successful_tasks WARNING")
                if self.optuna_opti:
                    # generate an id based on the current time and a random number
                    id = datetime.datetime.now().strftime("%Y%m%d%H%M%S") + "_" + str(random.randint(0, 1000))
                    function_file_path = os.path.join("functions", self.name + f"_{id}.py")
                else:
                    function_file_path = os.path.join("functions", smart_input("This function already exists, please provide a new function name: ", message_type="VALIDATION_INFO") + ".py")

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
            parsed_code["program_code"] = re.sub(r"(def " + function_name + "\(.*?\):)", r'\1\n    ' + docstring,
                                                 parsed_code["program_code"], count=1)
            function_file.write(parsed_code["program_code"])

        # Open file in VSCode if necessary
        """
        if self.optuna_opti is None and is_vscode_installed():
            smart_print("Please modify and save the file in VSCode (Ctrl + W) when ready.", self.name,
                        "capitalize_successful_tasks INSTRUCTIONS")
            subprocess.run(["code", "--wait", new_path])
        """

        # Serialize entry for logging
        serialized_entry = json.dumps({
            "time": datetime.datetime.now().isoformat(),
            ("class_name" if (
                        self.problem_prompts_subdir == "Anomalies/" or self.problem_prompts_subdir == "pipeline_synthesis/") else "main_function_name"): function_name,
            "program_code": parsed_code["program_code"],
            "tool_description": tool_description,
            "task_description": task_description,
        }, default=lambda o: o.__dict__ if hasattr(o, '__dict__') else str(o))

        # Add to vector database with tags
        tags = {"host": f"{socket.gethostname()}-{uuid.getnode()}", "step_id": HumanLLMMonitor.step_id}
        HumanLLMMonitor.add_learnt_task(serialized_entry, tags)

    def capitalize_failed_tasks(self, task_description: str, parsed_code: str) -> None:
        import socket, uuid, datetime, json

        is_anomaly = self.problem_prompts_subdir == "Anomalies/" or self.problem_prompts_subdir == "pipeline_synthesis/"
        name_key = "class_name" if is_anomaly else "main_function_name"
        if parsed_code:
            task_name = parsed_code.get(name_key,
                                        "replace this text with a descriptive name of the class" if is_anomaly else "replace this text with a descriptive name of the function")
        else:
            if False or not self.optuna_opti: # TODO: temporary disabled, find the logic to fix this or if not required
                task_name = smart_input(
                    f"CONFIG Please provide a name for the {'pipeline' if is_anomaly else 'function'}: {task_description}", "CapitalizationAgent",message_type="Capitalization_info").strip()
            else:
                task_name = f"{task_description[:500]}"
        if False and not self.optuna_opti or is_anomaly: # TODO: temporary disabled, find the logic to fix this or if not required
            task_name = _visual_input(task_name)
            task_description_refined = _visual_input(task_description)
        else:
            task_description_refined = task_description

        # Store task description
        self.failed_tasks_repository[task_name] = task_description_refined
        smart_print(
            f"************ Last added failed task ************\n{task_name}\n************************".replace("\\n",
                                                                                                               "\n"),
            self.name, "capitalize_failed_tasks CAPITALIZE FAIL", optional=True)

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
            "host": HumanLLMMonitor.get_host_id(),
            "step_id": HumanLLMMonitor.step_id,
        }
        HumanLLMMonitor.add_failed_task(serialized_entry, tags)

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
                smart_print(f"> {task_type} {task_name} already loaded. Skipping duplicates...", self.name,
                            "retrieve_saved_tasks_in_db DATABASE ACCESS", optional=True)
        tags = {"host": HumanLLMMonitor.get_host_id(),
                "step_id": HumanLLMMonitor.step_id, }

        self.db_failed_tasks.add_texts(texts=[serialized_entry], metadatas=[tags])

    def generate_tool_description(self, program_name, program_code):
        user_message = f"MAIN FUNCTION: `{program_name}`\n\nFULL CODE:\n{program_code}"
        tool_description = self.human_llm_generate_function_description.CallHumanLLM(
            system_prompt_template="generate_function_description", user_message=user_message,
            return_message_content_only=True, optuna=self.optuna_opti, model_choice=self.model_choice)
        return tool_description

#        smart_print(f"Error: Could not find a valid definition for {function_name}. Please set it:", "orchestrate_agents", "orchestrate_agents ERROR")
#        function_code = _visual_input(current_function_code, filetype="py")


def validate_function_code(code, function_name, local_scope=None, compile_test_only=False):
    if local_scope is None:
        local_scope = {}
    try:
        compiled_code = compile(code, '<string>', 'exec')
        if compile_test_only:
            return True
        exec(compiled_code, globals(), local_scope)
        func = local_scope.get(function_name)
        if func is None or not callable(func):
            raise ValueError(f"Function {function_name} is not defined or not callable.")
        return func
    except Exception as e:
        print(f"Error setting {function_name} function: {e}")
        return None


def extract_function_code(task_content, function_name, current_function_code=None):
    """
    Extracts the complete code block for the specified function from the given task content.
    If extraction fails, prompts the user for correct function code until successful.
    """
    pattern = rf"(def {function_name}\(.*?\):.*?)(?=\ndef [a-zA-Z_]+\(|$)"
    match = re.search(pattern, task_content, re.DOTALL)
    function_code = match.group(1) if match else None

    local_scope = {}
    validated_function = validate_function_code(function_code, function_name, local_scope) if function_code else None

    while not validated_function:
        smart_print(f"Error: Could not find a valid definition for {function_name}. Please set it:",
                    "orchestrate_agents", "orchestrate_agents ERROR")
        function_code = _visual_input(current_function_code, filetype="py")
        validated_function = validate_function_code(function_code, function_name, local_scope)

    return function_code


# Main learning loop orchestration functions
def run_4agents_learning_loop(default_llm_key, premium_llm_key, test_environments=None,
                              manual_validation_to_capitalize=True, problem_prompts_subdir=None,
                              max_coding_attempts=4, include_code=None, selected_successful_functions=None,
                              selected_failed_functions=None, agtask_premium_llm_by_default=True,
                              agtask_skip_rounds=0, agcoding_skip_rounds=0, agvalidation_skip_rounds=0,
                              agcapitalize_skip_rounds=0, llmORchains_list=None, model_choice=None,
                              optuna_opti=None, allow_custom_score_state_functions=False,
                              params_user_message=None, max_execution_time=900, special_criteria=None, temperature_max=1,
                              agcoach_num_parallel_inferences=2, fixed_coach=False, unique_id=None, return_array=False, agcoding_num_parallel_inferences=2,
                              continue_each_loop=False):
    scores = None

    if unique_id is None:
        unique_id = f"{socket.gethostname()}_{datetime.now().strftime('%d-%m-%Y-%H-%M-%S')}"

    if unique_id is not False :
        if UnifiedVectorDB.unique_collection_id is None:
            UnifiedVectorDB.set_unique_collection_id(unique_id)

    HumanLLMMonitor._check_and_init_vector_db(embedding_function=embedding_function, reset_db_indices=reset_db_indices)
    HumanLLMMonitor.check_init_class_db(force=True)

    if params_user_message is None and optuna_opti is None:
        params_user_message = {
            'sources': ["learnt", "failed", "default"],
            'num': [2, 3, 2],
            'format': ["json", "Jinja2", "Markdown"]
        }

    if HumanLLMMonitor.use_websocket:
        if HumanLLMMonitor.websocket_server is None:
            HumanLLMMonitor.initialize_websocket_server()

    smart_print(str(max_execution_time), "orchestrate_agents", "time_end")
    smart_print(unique_id, "orchestrate_agents", "XP_unique_id", optional=True)

    time_end = time.time() + max_execution_time

    if problem_prompts_subdir is None:
        # menu to choose the problem prompts subdirectory
        # get the list of subdirectories in the problem prompts directory
        problem_prompts_subdirs = [name for name in os.listdir("prompts") if
                                   os.path.isdir(os.path.join("prompts", name))]
        # get first element of problem_prompts_subdirs if not empty, else set it to empty string
        default_subdir = problem_prompts_subdirs[0] if problem_prompts_subdirs else ""
        choice = smart_input("Enter a capital letter for subdirectory (leave empty for default): " + "; ".join(
            f"\n[{i}] {subdir}" for i, subdir in zip(string.ascii_uppercase, problem_prompts_subdirs)) + " ?", "orchestrate_agents")
        # if choise is empty or not a capital letter or not in the range of the list of subdirectories, set it to A
        problem_prompts_subdir = problem_prompts_subdirs[ord(choice) - 65] if choice and choice.isupper() and ord(
            choice) - 65 in range(len(problem_prompts_subdirs)) else default_subdir

    if test_environments is None:
        env_type = "default"
        manager = EnvironmentManager(env_type)
        test_environments = [manager.get_environment()]

    agent_taskreco = TaskIdentificationAgent(default_llm_key, test_environments, premium_llm_choice=premium_llm_key,
                                             problem_prompts_subdir=problem_prompts_subdir,
                                             premium_llm_by_default=agtask_premium_llm_by_default,
                                             skip_rounds=agtask_skip_rounds, llmORchains_list=llmORchains_list,
                                             optuna=optuna_opti, model_choice=(
            model_choice['taskreco' if 'taskreco' in model_choice else 'coach'] if type(
                model_choice) == dict else model_choice), criteria=params_user_message, temperature_max=temperature_max,
                                             num_parallel_inferences=agcoach_num_parallel_inferences, fixed_coach=fixed_coach, special_criteria=special_criteria)

    agent_coding = CodingAgent(default_llm_key, test_environments, premium_llm_choice=premium_llm_key,
                               problem_prompts_subdir=problem_prompts_subdir, skip_rounds=agcoding_skip_rounds,
                               llmORchains_list=llmORchains_list, optuna=optuna_opti, model_choice=(
            model_choice['coding' if 'coding' in model_choice else 'coder'] if type(
                model_choice) == dict else model_choice), special_criteria=special_criteria, num_parallel_inferences=agcoding_num_parallel_inferences)

    agent_validation = ValidationAgent(default_llm_key, test_environments, premium_llm_choice=premium_llm_key,
                                       skip_rounds=agvalidation_skip_rounds, llmORchains_list=llmORchains_list,
                                       optuna=optuna_opti, model_choice=(
            model_choice['validation' if 'validation' in model_choice else 'critic'] if type(
                model_choice) == dict else model_choice), special_criteria=special_criteria)

    agent_capitalize = CapitalizationAgent(default_llm_key, premium_llm_choice=premium_llm_key,
                                           skip_rounds=agcapitalize_skip_rounds,
                                           problem_prompts_subdir=problem_prompts_subdir,
                                           llmORchains_list=llmORchains_list, optuna=optuna_opti, model_choice=(
            model_choice['capitalize' if 'capitalize' in model_choice else 'capitalizer'] if type(
                model_choice) == dict else model_choice), special_criteria=special_criteria)

    #agent_capitalize.retrieve_saved_tasks_in_db(include_code=include_code, selected_successful_functions=selected_successful_functions, selected_failed_functions=selected_failed_functions)
    continue_identifying_tasks = True
    total_scores = []

    # Global learn loop
    while continue_identifying_tasks and time.time() < time_end:
        HumanLLMMonitor.step_id = str(uuid.uuid4())
        task = agent_taskreco.identify_best_task()

        # Handle multiple-tasks case
        if len(task) > 1:
            if agent_taskreco.human_llm_identify_best_task.selected_outputs and len(agent_taskreco.human_llm_identify_best_task.selected_outputs) == 1:
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
                        id = 1 if optuna_opti else int(smart_input("Enter the index of the task to select: ", "orchestrate_agents",
                                                                   "TASK SELECTION").strip())
                        if id in range(len(task)):
                            task = task[id]
                            break
                        else:
                            raise Exception("Index out of range")
                    except Exception as e:
                        smart_print(f"Error: {e}\n\nEnter a valid index", "orchestrate_agents")
        else:
            task = task[0]
        smart_print("Identified Task: " + task.content.replace("\\n", "\n"), "orchestrate_agents",
                    "orchestrate_agents RESULT", optional=True)
        task_description = task.content
        # Extract potential score and state function code from the task
        if allow_custom_score_state_functions:
            # Extract current implementation of get_score and get_state from the first environment
            current_score_function_code, current_state_function_code = inspect.getsource(
                test_environments[0].get_score), inspect.getsource(test_environments[0].get_state)
            score_function_code, state_function_code = extract_function_code(task_description, 'get_score',
                                                                             current_score_function_code), extract_function_code(
                task_description, 'get_state', current_state_function_code)

            for env in test_environments:
                env.set_score_function(score_function_code)
                env.set_state_function(state_function_code)

        parsed_code, validation, scores = coding_and_validation_loop(agent_coding, agent_validation, task_description,
                                                                     max_coding_attempts,
                                                                     manual_validation_to_capitalize,
                                                                     optuna=optuna_opti,
                                                                     end_time=time_end)
        if validation == "success":
            agent_capitalize.capitalize_successful_tasks(task_description, parsed_code)
        else:
            if optuna_opti or smart_input(
                    "Do you want to capitalize this try as a 'failed task' to avoid this task to be proposed as a next best task ? (yes/no): ", "orchestrate_agents", message_type="VALIDATION_INFO").strip().upper() in [
                "Y", "YES"]:
                agent_capitalize.capitalize_failed_tasks(task_description, parsed_code)
        if optuna_opti:
            continue_identifying_tasks = continue_each_loop
        else:
            answer = smart_input(
                "Do you want to:\n- search for a new task after reseting to empty documents (Y/YES) ?\n- search for a new task based based on the status of documents after applying the task you just validated (N/NO/Enter) ?\n- or just exit the program (E/EXIT) ?", "orchestrate_agents",  message_type="VALIDATION_INFO").strip().upper()
            continue_identifying_tasks = False if answer in ["E", "EXIT"] else True
            if answer.upper() in ["Y", "YES"]:
                [env.reset() for env in test_environments]
            else:
                if parsed_code:
                    # Apply the code to the environments without restoring their state
                    test_results = agent_coding.run_tests_on_code(message="", parsed_code=parsed_code, skip_already_processed=False, restore_state=False, custom_agent="orchestrate_agents")
                    # Unpack the results if needed
                    _parsed_code_, _success_, exec_results, _scores_, _env_states_, _execution_time_ = test_results

                    # Optionally display the execution results for each environment
                    for env, result in zip(test_environments, exec_results):
                        smart_print(f"Execution result in environment {env.id}: {result}", "orchestrate_agents", "Execution Result")
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
                print(f"total_score_weighted_with_stats: {total_score_weighted_with_stats}; scores['percentage_no_runtime_error']: {scores['percentage_no_runtime_error']}; scores['best_score_without_validation']: {scores['best_score_without_validation']}; validated_score_avg: {validated_score_avg}")
            # add total_score_weighted_with_stats to total_scores
            total_scores.append(total_score_weighted_with_stats)
        else:
            total_scores.append(0)

    # print status of: continue_identifying_tasks and time.time() < time_end
    print(f"continue_identifying_tasks: {continue_identifying_tasks}, time.time() < time_end: {time.time() < time_end}, time.time(): {time.time()}, time_end: {time_end}")

    if return_array:
        return total_scores
    else:
        return max(total_scores)


def get_success_value_in_text(text):
    match = re.search(r"Success['\"]?\s*[:=][:=]?\s*(['\"]?)(True|False|Yes|No|y|n|0|1)\1", text, re.IGNORECASE)
    if match:
        success_value = match.group(2)
        return success_value.lower() in ['true', 'yes', 'y', '1']
    return False


def get_highest_score_index(score_array, mode='total'):
    """
    Returns the index of the sublist with the highest total or average score.

    Parameters:
    score_array (list): List of lists of dictionaries with score values.
    mode (str): 'total' to consider total score, 'average' to consider average score. Default is 'total'.

    Returns:
    int: Index of the sublist with the highest score.
    """
    highest_index = -1
    highest_score = float('-inf')

    for i, score_sublist in enumerate(score_array):
        # Calculate the total or average score for the entire sublist
        if mode == 'total':
            current_score = sum(sum(scores.values()) for scores in score_sublist)
        elif mode == 'average':
            total_values = sum(len(scores) for scores in score_sublist)
            current_score = sum(sum(scores.values()) for scores in score_sublist) / total_values
        else:
            raise ValueError("Invalid mode. Use 'total' or 'average'.")

        if current_score > highest_score:
            highest_score = current_score
            highest_index = i

    return highest_index


def coding_and_validation_loop(agent_coding, agent_validation, task_description, max_attempts,
                               extra_manual_validation_to_capitalize=True, continue_even_if_successful=True,
                               optuna=None, end_time=None):
    metadata = {'step_id': HumanLLMMonitor.step_id}
    # Retrieve data
    previous_errors = HumanLLMMonitor.get_agent_data(agent_coding.name, 'previous_errors',
                                                     metadata_filter=metadata) or []
    previous_codes = HumanLLMMonitor.get_agent_data(agent_coding.name, 'previous_codes', metadata_filter=metadata) or []
    previous_scores = HumanLLMMonitor.get_agent_data(agent_coding.name, 'previous_scores',
                                                     metadata_filter=metadata) or []
    unique_codes = set(
        HumanLLMMonitor.get_agent_data(agent_coding.name, 'unique_codes', metadata_filter=metadata) or [])
    successful_codes = HumanLLMMonitor.get_agent_data(agent_coding.name, 'successful_codes',
                                                      metadata_filter=metadata) or []
    all_results = HumanLLMMonitor.get_agent_data(agent_coding.name, 'all_results', metadata_filter=metadata) or []

    for attempt in range(max_attempts):
        # Check for timeouts
        if end_time is not None and time.time() >= end_time:
            break

        # Filter out duplicates
        temp_errors, temp_codes, temp_scores = [], [], []
        for err, code, score in zip(previous_errors, previous_codes, previous_scores):
            if code not in unique_codes:
                unique_codes.add(code)
                temp_errors.append(err)
                temp_codes.append(code)
                temp_scores.append(score)
        previous_errors, previous_codes, previous_scores = temp_errors, temp_codes, temp_scores

        # Store updated data
        HumanLLMMonitor.add_agent_data(agent_coding.name, 'previous_errors', previous_errors, metadata=metadata)
        HumanLLMMonitor.add_agent_data(agent_coding.name, 'previous_codes', previous_codes, metadata=metadata)
        HumanLLMMonitor.add_agent_data(agent_coding.name, 'previous_scores', previous_scores, metadata=metadata)
        HumanLLMMonitor.add_agent_data(agent_coding.name, 'unique_codes', list(unique_codes), metadata=metadata)

        results = agent_coding.code_task_and_run_test(task_description)

        all_results.extend(results)
        HumanLLMMonitor.add_agent_data(agent_coding.name, 'all_results', all_results, metadata=metadata)

        current_skip_rounds = agent_validation.human_llm_validate_code.skip_rounds
        for index, (parsed_code, no_runtime_error, exec_result, scores, env_states, times) in enumerate(results):
            agent_validation.human_llm_validate_code.skip_rounds = current_skip_rounds  # to prevent skip_rounds decreased multiple times by multiple calls of HumanLLMMonitor
            smart_print(
                f"Generated code:\n{parsed_code['program_code']}\n*******\nOutput of code execution:\n{exec_result}\n".replace(
                    "\\n", "\n"), "coding_and_validation_loop", "coding_and_validation_loop RESULT")
            validation_agent_feedback = agent_validation.validate_code(parsed_code["program_code"], no_runtime_error,
                                                                       exec_result, task=task_description,
                                                                       scores=scores, env_states=env_states)
            smart_print("Agent validation 'feedback' currently only support 1 feedback", "coding_and_validation_loop",
                        "coding_and_validation_loop WARNING")
            validation_agent_feedback = validation_agent_feedback[0]
            afb = validation_agent_feedback.content.replace('\\n', '\n')
            smart_print("#" * 20 + f"\nAgent validation feedback: {afb}", "coding_and_validation_loop",
                        "coding_and_validation_loop RESULT")

            if extra_manual_validation_to_capitalize:
                validated = (smart_input(
                    "#" * 20 + f"\nADD THIS FUNCTION TO LIBRARY ? Please enter 'yes' if this a success and you want to add this function to library, 'no' if this failed: ",
                    "coding_and_validation_loop").lower() in [
                                 "yes", "y", True])
            else:
                validated = get_success_value_in_text(afb) in ["yes", "y", True]
            if validated:
                successful_codes.append((parsed_code, validation_agent_feedback, scores))

            if index < len(previous_errors):
                previous_errors[index] = validation_agent_feedback
                previous_scores[index] = scores
                previous_codes[index] = parsed_code['program_code']
            else:
                previous_errors.append(validation_agent_feedback)
                previous_scores.append(scores)
                previous_codes.append(parsed_code['program_code'])

                # Store updated data
            HumanLLMMonitor.add_agent_data(agent_coding.name, 'previous_errors', previous_errors, metadata=metadata)
            HumanLLMMonitor.add_agent_data(agent_coding.name, 'previous_codes', previous_codes, metadata=metadata)
            HumanLLMMonitor.add_agent_data(agent_coding.name, 'previous_scores', previous_scores, metadata=metadata)

            if validated:
                successful_codes.append((parsed_code, validation_agent_feedback, scores))
                HumanLLMMonitor.add_agent_data(agent_coding.name, 'successful_codes', successful_codes,
                                               metadata=metadata)

        if not optuna and not successful_codes:
            stop = smart_input("No successful code yet, do you want to stop coding attempts for this task (too hard) and try a new one ? (yes/no): ","coding_and_validation_loop", "VALIDATION_INFO",optional=False).lower() in ["yes", "y", True]
            if stop:
                break

        if not optuna and successful_codes:
            stop = smart_input("A successful code has been found, do you want to stop coding attempts for this task (performance is sufficient) and try a new one ? (yes/no): ","coding_and_validation_loop","VALIDATION_INFO",optional=False).lower() in ["yes", "y", True]
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
    percentage_no_runtime_error = (sum(1 for _, no_runtime_error, _, _, _, _ in all_results if no_runtime_error) / len(
        all_results)) if all_results else 0
    best_score_without_validation = (
        max(max(sum(scores.values()) / len(scores) if len(scores) > 0 else 0 for scores in score_dict) for _, _, _, score_dict, _, _ in all_results)) if len(
        all_results) > 0 else 0
    all_scores = {
        'percentage_no_runtime_error': percentage_no_runtime_error,
        'best_score_without_validation': best_score_without_validation,
        'validated_scores': None}

    # Second part: If there are successful codes, ask user to select one
    if successful_codes :
        if len(successful_codes) == 1:
            selected_code, _, scores = successful_codes[0]
            all_scores['validated_scores'] = scores
            return selected_code, "success", all_scores
        if current_skip_rounds <= 0:
            for i, (parsed_code, feedback, scores) in enumerate(successful_codes):
                smart_print(
                    f"\033Option {i + 1}:\nCode:\n{parsed_code['program_code']}\nFeedback: {feedback.content}\n\033[91mScore: {scores}\033[0m\n",
                    None, "coding_and_validation_loop RESULT")
            if optuna:
                highest_score_index = get_highest_score_index([scores for _, _, scores in successful_codes],
                                                              mode='total')
                selected_code, _, scores = successful_codes[highest_score_index]
                all_scores['validated_scores'] = scores
                return selected_code, "success", all_scores
            else:
                # Create a string with the indexes of the successful codes with their names and scores
                successful_codes_str = "\n".join(
                    f"Code **{i}**: {parsed_code['main_function'] if 'main_function' in parsed_code else parsed_code} - Score:{scores} - Code extract:{parsed_code['program_code'][:100]}" for i, (parsed_code, _, scores) in
                    enumerate(successful_codes))
                selection = smart_input(
                    f"Several codes were successful. Please enter the number of the code you want to add to the library:\n{successful_codes_str}",agent_name="CapitalizationAgent",message_type="Capitalization_info").strip()
            if selection.isdigit() and 0 <= int(selection) <= len(successful_codes):
                selected_index = max(0, min(int(selection), len(successful_codes) - 1))
                smart_print("Code validated successfully.", "coding_and_validation_loop", "coding_and_validation_loop RESULT")
                selected_code, _, scores = successful_codes[selected_index]
                all_scores['validated_scores'] = scores
                return selected_code, "success", all_scores
            else:
                smart_print("Invalid selection or no selection made. Exiting without adding any code.", "coding_and_validation_loop", "coding_and_validation_loop WARNING")
        else:  # if in automatic mode, select the code with the highest score
            highest_score_index = get_highest_score_index([scores for _, _, _, scores in successful_codes],
                                                          mode='total')
            selected_code, _, scores = successful_codes[highest_score_index]
            all_scores['validated_scores'] = scores
            return selected_code, "success", all_scores

    return None, "failed", all_scores  # If no successful code was selected, return failure


def sanitized_task_name(task):
    # Implement task name sanitization logic
    return task


class PrintPromptRunnable(Runnable):
    def invoke(self, input_msg, config):
        smart_print(f"PrintPromptRunnable type of input_msg: {type(input_msg)}")
        # Extract and format the prompt
        formatted_prompt = format_prompt(input_msg if isinstance(input_msg, list) else input_msg.messages)
        # Print the prompt in RED
        smart_print("\033[31m" + formatted_prompt + "\033[0m")
        return input_msg


class ExtractMessage(Runnable):
    def invoke(self, input_msg, config):
        return "\n".join(getattr(message, 'content', message) for message in getattr(input_msg, 'messages', input_msg))


def format_prompt(messages):
    prompt_str = ""
    for message in messages:
        if isinstance(message, SystemMessage):
            prompt_str += "System: " + message.content + "\n"
        elif isinstance(message, HumanMessage):
            prompt_str += "Human: " + message.content + "\n"
        elif isinstance(message, AIMessage):
            prompt_str += "AI: " + message.content + "\n"
        else:
            prompt_str += f"Type {type(message)}: " + str(message.content) + "\n"
    return prompt_str
    
def create_Nmajority_chain(num_models=3, map_model_name=None, reduce_model_name=None, map_temperature=0.7, reduce_temperature=0.):
    # Initialize the OpenAI models
    if map_model_name is None:
        map_model_name = MODELS_CONFIG_LIST["basic_gpt" if "basic_gpt" in MODELS_CONFIG_LIST else "gpt"]
    if reduce_model_name is None:
        reduce_model_name = MODELS_CONFIG_LIST["smart_gpt" if "smart_gpt" in MODELS_CONFIG_LIST else "gpt"]
    models = [ChatOpenAI(model_name=map_model_name, temperature=map_temperature, cache=False) for _ in
              range(num_models)]
    final_model = ChatOpenAI(model_name=reduce_model_name, temperature=reduce_temperature, cache=False)

    # Define the chain using LCEL
    response_keys = [f"response_{i + 1}" for i in range(num_models)]
    multi_reponse = {key: model for key, model in zip(response_keys, models)}
    multi_reponse["cleaned_input"] = ExtractMessage()

    final_prompt_template_str = (
            "Given the following responses below and the initial question below, provide an optimal response to the question mixing best elements of each and following the same answer output structure:\n\n"
            "Initial question:\n{cleaned_input}\n\n" +
            "\n\n".join(
                [f"Response {i + 1}:\n{{{response_keys[i]}}}" for i in range(num_models)]) +  # Access content directly
            "\n\nOptimal Response:\n"
    )

    # Insert the PrintPromptRunnable just to print the prompt for control
    print_prompt_runnable = PrintPromptRunnable()

    # Print in RED the final prompt template: print("\033[31m"+final_prompt_template_str+"\033[0m")
    final_prompt_template = ChatPromptTemplate.from_template(final_prompt_template_str)

    chain = multi_reponse | final_prompt_template | print_prompt_runnable | final_model  # No ERROR but bad output: "I'm sorry, but I cannot fulfill this request" or "I'm sorry, but I cannot fulfill this request as it is too complex for me to process." or "I'm sorry, but I cannot fulfill this request as it involves creating a Python function and providing a specific response format." or "I'm sorry, but I cannot fulfill this request as it requires a level of understanding and reasoning that is beyond my current capabilities."

    return chain


if __name__ == "__main__":
    import argparse

    # Handle command line arguments
    parser = argparse.ArgumentParser(description="Run the learning loop with optional WebSocket settings")
    parser.add_argument("--port", type=int, default=6789, help="Optional port for WebSocket server")
    parser.add_argument("--secret", action='store_true', help="Optional secret for WebSocket URL")
    parser.add_argument("--proxy", action='store_true', help="Start a proxy via localtunnel if available")
    args = parser.parse_args()

    # Initialize the WebSocket server with port autodetection and proxy
    unique_id = f"{socket.gethostname()}_{datetime.now().strftime('%d-%m-%Y-%H-%M-%S')}"
    HumanLLMMonitor.initialize_websocket_server(port=args.port, secret=args.secret, proxy_enabled=args.proxy, unique_id=unique_id)

    # Allow some time for the WebSocket server to start
    time.sleep(1)  # Adjust if necessary

    # Initialize the default and premium LLMs
    #from langchain_groq import ChatGroq
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

    envs = []
    for doc in documents:
        env = EnvironmentManager(env_type="techsynthesis", title=doc['title'], context=doc['context'],
                                 target_file_path=doc['target_file_path'], id=doc['id'], llm=llmORchains_list["default_llm"]).get_environment()
        envs.append(env)

    # Run the learning loop
    run_4agents_learning_loop(default_llm_key="default_llm",
                              premium_llm_key="premium_llm",
                              llmORchains_list=llmORchains_list,
                              test_environments=envs,
                              manual_validation_to_capitalize=False,
                              problem_prompts_subdir="IR_CPS_TechSynthesis",
                              max_coding_attempts=4,
                              include_code=False,
                              selected_successful_functions=[],
                              selected_failed_functions=[],
                              max_execution_time=3600,
                              agtask_premium_llm_by_default=False,
                              agtask_skip_rounds=0,  # Auto-test: 1
                              agcoding_skip_rounds=0,  # Auto-test: 4
                              agvalidation_skip_rounds=0,  # Auto-test: 4
                              agcapitalize_skip_rounds=0,
                              agcoding_num_parallel_inferences=2,
                              unique_id=unique_id)  # Auto-test: 0"""
