import inspect
import random
import string
import time
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
from config import *
from utils.llm_utils import UnifiedVectorDB, HumanLLMMonitor, _visual_input, smart_print, smart_input
from env.IR_CPS_TechSynthesis.env import *
from env.SWEBench.env import *


dataset_repo = load_dataset(path="ahsanirfan961/swe-bech-lite-bm25-13k-take50", split='train')

def get_row_by_instance_id(instance_id):
     for row in dataset_repo:
        if row['instance_id'] == instance_id:
            return row
     return None
instance_id= "astropy__astropy-14365"
repo=get_row_by_instance_id(instance_id)


#set_llm_cache(SQLiteCache(database_path=".langchain_caching.db"))

openai.api_key = os.environ['OPENAI_API_KEY']
if 'OPENAI_BASE_URL' in os.environ: openai.base_url = os.environ['OPENAI_BASE_URL']

UnifiedVectorDB.db_type = "elasticsearch"  # "elasticsearch" "chroma"
UnifiedVectorDB.es_url = elastic_url_port # self commented
UnifiedVectorDB.es_user = elastic_user
UnifiedVectorDB.es_password = elastic_password
UnifiedVectorDB.OpenAI_embedding_function_name = "text-embedding-ada-002"  # "nomic-ai/nomic-embed-text-v1"

embedding_function = "intfloat/e5-base-v2"  # UnifiedVectorDB.OpenAI_embedding_function_name # e.g. "text-embedding-ada-002" for OpenAI or "intfloat/e5-base-v2" or other huggingface models - WARINING: if you change it, set reset_db_indices to True
reset_db_indices = False  # Set to True after changing embeddings

HumanLLMMonitor.use_websocket = True

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
              
            elif key in available_locals:
                new_params[key] = value
             

    return new_params  # Return only new params

# Agent 1: Task Identification
class TaskIdentificationAgent():
    def __init__(self, default_llm_choice, envs: [Environment], premium_llm_choice=None, problem_prompts_subdir=None,
                 premium_llm_by_default=True, skip_rounds=0, llmORchains_list=None, optuna=None, model_choice=None,
                 criteria=None, params_user_message=None, temperature_min=0., temperature_max=1.,
                 num_parallel_inferences=1, agcoach_num_parallel_inferences=1, fixed_coach=False, special_criteria=None):
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
        if special_criteria is not None and 'log_user_message' in special_criteria:
            setattr(self, 'log_user_message', special_criteria['log_user_message'])

        HumanLLMMonitor_args, local_vars = (set(inspect.signature(HumanLLMMonitor.__init__).parameters) - {'self'}), locals()
        kw_common_args = {param: local_vars[param] for param in HumanLLMMonitor_args if param in local_vars}

        kw_common_args.update(new_params)

        #self.criteria = criteria, self.params_user_message = params_user_message, self.temperature_min = temperature_min, self.temperature_max = temperature_max
        self.human_llm_identify_best_task: HumanLLMMonitor = HumanLLMMonitor(**kw_common_args)#,output_schema="identify_best_task.schema.py")
        self.human_llm_identify_best_task.skip_rounds = skip_rounds
        if self.additional_check_list:
            for key, value in self.additional_check_list.items():
                self.human_llm_identify_best_task.add_inference_check(key, value)

        self.human_llm_identify_best_task.add_inference_check("Recommend critiques",
                                                              self.human_llm_identify_best_task.generate_best_improvement_suggestions)

    def identify_best_task(self):
        # Prepare data
        envs_status = "\n".join([env.get_state() for env in self.envs])
        few_shots = self.human_llm_identify_best_task.get_multiple_few_shots(
            few_shots_params=self.params_user_message)
        print(few_shots)
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

        if hasattr(self, 'log_user_message') and self.log_user_message:
            with open(self.log_user_message, "a") as f:
                f.write("Coach -- identify_best_task:<<\n" + user_message + "\n>>\n\n")
                
        original_stdout = sys.stdout
        sys.stdout = open('system_debug.txt', 'w') 
        # print(self.problem_prompts_subdir + 'identify_best_task')
        sys.stdout=original_stdout 
        
        original_stdout = sys.stdout
        sys.stdout = open('user_message.txt', 'w') 
        # print(user_message)
        sys.stdout=original_stdout  
        
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
                 temperature_max=1., num_parallel_inferences=1):
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
                if not hasattr(self, key):
                    setattr(self, key, value)
        self.last_user_message = None

        HumanLLMMonitor_args, local_vars = (set(inspect.signature(HumanLLMMonitor.__init__).parameters) - {'self'}), locals()
        kw_common_args = {param: local_vars[param] for param in HumanLLMMonitor_args if param in local_vars}
      
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
                if language == "py":  # Python caset
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
        elif skip_already_processed or not isinstance(parsed_code, dict):  # This logic speedup because the same code should have the same score BUT only on the same problem & state
            return None

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
                    no_runtime_error, exec_result = False, f"Error: no test found for given id {env.id}" if parsed_code["tests"] else f"Error: no runnable code found nor tests"
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
                                HumanLLMMonitor.add_agent_data(self.name, 'error_patches', error_patches, metadata=metadata)

            no_runtime_errors.append(no_runtime_error)
            exec_results.append(exec_result)

        # Calculate total time taken  # Added line
        total_execution_time = time.time() - start_time
        # Return combined results
        if isinstance(self.envs[0], SWEBenchEnvironment):
            scores = [env.get_score(parsed_code["program_code"]) for env in self.envs]
        else:
            scores = [env.get_score() for env in self.envs]
        result = (parsed_code, all(no_runtime_errors), exec_results, scores, [env.get_state(extended=False) for env in self.envs], total_execution_time)

        if restore_state:
            for env in self.envs:
                env.restore_last_state()
        return result

    def get_primitives(self):
        primitives = []

        path_folder = "primitives/swe_primititves"
        folder_path = os.path.join(os.path.dirname(__file__), path_folder)
        # Utiliser os.listdir pour ne pas parcourir les sous-répertoires

        for file in os.listdir(folder_path):
            if file.endswith(".py"):
                smart_print(f"File load: {file}", "CodingAgent", "Files loaded", optional=True)
                file_path = os.path.join(folder_path, file)
                with open(file_path, "r") as f:
                    primitives.append(f.read())
        with open('primitives_checking.txt', 'w') as file:
          sys.stdout = file  # Redirect standard output to the file

        sys.stdout = sys.__stdout__                    
        return primitives

    def code_task_and_run_test(self, refined_task):

        print('Starting code_task_and_run_test')

        def flatten_and_pair(nested_list):
            def flatten(nested):
                flat_list = []
                for item in nested:
                    if isinstance(item, (list, tuple)):
                        flat_list.extend(flatten(item))
                    elif isinstance(item, dict):
                        for value in item.values():
                            flat_list.extend(flatten(value))
                    else:
                        flat_list.append(item)
                return flat_list

            flat_list = flatten(nested_list)

            # Décomposer les tuples pour éviter les tuples imbriqués
            def unpack_tuples(items):
                unpacked = []
                for item in items:
                    if isinstance(item, tuple):
                        unpacked.extend(unpack_tuples(item))
                    else:
                        unpacked.append(item)
                return unpacked

            flat_list = unpack_tuples(flat_list)

            # Toujours regrouper les éléments par paires de 2 et retourner une liste de tuples
            paired_list = []
            i = 0
            while i < len(flat_list):
                if i + 1 < len(flat_list):
                    paired_list.append((flat_list[i], flat_list[i + 1]))
                    i += 2
                else:
                    # Si le nombre d'éléments est impair, le dernier élément est ajouté seul dans un tuple
                    paired_list.append((flat_list[i],))
                    i += 1
            return paired_list

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
        error_patches = flatten_and_pair(error_patches) if error_patches else []

        env_states = "\n".join([env.get_state(extended=False) for env in self.envs])
        primitives = "\n".join(self.get_primitives())
        successful_tasks = "\n".join(HumanLLMMonitor.get_learnt_tasks())
        failed_tasks = "\n".join(HumanLLMMonitor.get_failed_tasks())
        validation_response_um = "\n".join(HumanLLMMonitor.get_validation_results())

        # print("Primitives: ", primitives)

        previous_attempts = ""
        for errors_list, scores_list, codes_list in zip(previous_errors, previous_scores, previous_codes):
            if errors_list and scores_list and codes_list:
                for err, score, code in zip(errors_list.items(), scores_list.items(), codes_list.items()):
                    previous_attempts += f"\n<<ATTEMPT FEEDBACK: {err[1]}\nSCORE: {score[1]}\nCODE: {code[1]}>>\n"

        error_patches_str = ""
        for (error_msg, diff_text) in error_patches:
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

        if hasattr(self, 'log_user_message') and self.log_user_message:
            with open(self.log_user_message, "a") as f:
                f.write("Coder -- code_task_and_run_test:<<\n" + user_message + "\n>>\n\n")

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
        if special_criteria is not None and 'log_user_message' in special_criteria:
            setattr(self, 'log_user_message', special_criteria['log_user_message'])
        HumanLLMMonitor_args, local_vars = (set(inspect.signature(HumanLLMMonitor.__init__).parameters) - {'self'}), locals()
        kw_common_args = {param: local_vars[param] for param in HumanLLMMonitor_args if param in local_vars}
      
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

        if hasattr(self, 'log_user_message') and self.log_user_message:
            with open(self.log_user_message, "a") as f:
                f.write("Validation -- validate_code:<<\n" + user_message + "\n>>\n\n")

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
        if special_criteria is not None and 'log_user_message' in special_criteria:
            setattr(self, 'log_user_message', special_criteria['log_user_message'])
        HumanLLMMonitor_args, local_vars = (set(inspect.signature(HumanLLMMonitor.__init__).parameters) - {'self'}), locals()
        kw_common_args = {param: local_vars[param] for param in HumanLLMMonitor_args if param in local_vars}

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

        print('Starting capitalize_successful_tasks')
        import socket, uuid, datetime

        if self.problem_prompts_subdir == "Anomalies/" or self.problem_prompts_subdir == "pipeline_synthesis/":
            function_name = parsed_code.get("main_function_name", parsed_code.get("main_function", {}).get("name", "unknown"))
            pipeline_file_path = os.path.join("pipelines/pipelines", function_name + ".py")
            tool_description = str(self.generate_tool_description(function_name, parsed_code["program_code"]))
            self.learnt_tasks_repository[function_name] = [tool_description, parsed_code["program_code"]]
            # print last added task
            smart_print(
                f"************ Last added task ************\n{function_name}\n************************".replace("\\n",
                                                                                                                "\n"),
                self.name, "capitalize_successful_tasks SUCCESS", optional=True)
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
        serialized_entry = json.dumps({
            "time": datetime.datetime.now().isoformat(),
            ("class_name" if (self.problem_prompts_subdir == "Anomalies/" or self.problem_prompts_subdir == "pipeline_synthesis/") else "main_function_name"): function_name,
            "program_code": parsed_code["program_code"],
            "tool_description": tool_description,
            "task_description": task_description,
        }, default=lambda o: o.__dict__ if hasattr(o, '__dict__') else str(o))

        with open('inspection/results.pkl', 'wb') as f:
            pickle.dump(serialized_entry, f)
        
        print('Adding learnt task')

        # Add to vector database with tags
        tags = {"host": f"{socket.gethostname()}-{uuid.getnode()}", "step_id": int(HumanLLMMonitor.step_id)}
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

        if hasattr(self, 'log_user_message') and self.log_user_message:
            with open(self.log_user_message, "a") as f:
                f.write("Capitalization -- generate_tool_description:<<\n" + user_message + "\n>>\n\n")

        tool_description = self.human_llm_generate_function_description.CallHumanLLM(
            system_prompt_template="generate_function_description", user_message=user_message,
            return_message_content_only=True, optuna=self.optuna_opti, model_choice=self.model_choice)
        return tool_description

#        smart_print(f"Error: Could not find a valid definition for {function_name}. Please set it:", "orchestrate_agents", "orchestrate_agents ERROR")
#        function_code = _visual_input(current_function_code, filetype="py")

class PlannerAgent:
    def __init__(self, default_llm_choice, envs, premium_llm_choice=None, problem_prompts_subdir=None,
                 skip_rounds=0, llmORchains_list=None, optuna=None, model_choice=None, special_criteria=None,
                 num_parallel_inferences=1):
        # Define necessary class variables
        self.name = self.__class__.__name__
        self.last_user_message = None
        self.processed_codes = set()
        self.envs = envs
        self.problem_prompts_subdir = "" if problem_prompts_subdir is None else problem_prompts_subdir + "/"
        self.model_choice = model_choice
        self.optuna_opti = optuna
        self.llm = default_llm_choice  # Assuming this is an LLM object or a callable function
        self.llmORchains_list = llmORchains_list or {}
        self.skip_rounds = skip_rounds
        self.max_autofix = 3  # Maximum number of auto-fix attempts

    def plan(self, question: str):
        self.last_user_message = question

        # Retrieve learnt tasks (functions/code)
        print("Getting the learnt tasks...")
        learnt_tasks = HumanLLMMonitor.get_learnt_tasks(k=30)
        # print(learnt_tasks)
        if not learnt_tasks:
            smart_print("No learnt tasks are available to answer the question.", agent_name=self.name)
            print("No learnt tasks are available to answer the question.")
            return "no code available"

        # Prepare the code snippets string
        code_snippets = "\n\n".join(learnt_tasks)

        # if self.envs:
        #     if isinstance(self.envs[0], SWEBenchEnvironment):
        #         question = f"Solve the SWE bench problem {self.envs[0].swe_data.instance_id} by finding buggy files, generating patch, applying patch and running tests. Current state of the solution is:\n{self.envs[0].get_state(extended=False)}.\nOn the basis of current state, provide the best code that shoud be executed next"
        
        # print(question)

        print("Got the learnt tasks")

        prebuilt_code = ""
        for learnt_task in learnt_tasks:
            loaded = json.loads(learnt_task)
            prebuilt_code += loaded['program_code'] + "\n"

        # Prepare the prompt for the LLM
        prompt_template = (
            "You are a helpful assistant that uses the prebuilt functions to answer the user's question.\n"
            "User's question:\n{question}\n\n"
            "Available code snippets:\n{code_snippets}\n\n"
            "Please reuse these functions to create a new function that solves the user query.\n"
            "If multiple functions do exactly the same thing, choose the most efficient one.\n"
            "Don't re write the existing functions but only provide a new function that re uses these functions. Also execute that function at the end\n Your new function should also take bot as parameter"
            "Assume that bot, problem, and env will be automatically set as global variables.\n"
            "Only provide python code. Don't include any introductory or explanatory text\n"
            # "Provide full code and also execute the selected function with bot parameter"
        )
        prompt = prompt_template.format(question=question, code_snippets=code_snippets)

        print("Selecting the best code snippet to answer the question...")
        # Call the LLM using a method similar to CallHumanLLM
        selected_code = self.call_llm_with_similar_method(
            prompt,
            use_premium_llm=False,
            temperature=0.2  # You can adjust the temperature as needed
        )

        print("----------------------------------------")
        print(f"Selected code: {selected_code}...")
        print("----------------------------------------")

        if selected_code:
            selected_code = prebuilt_code + "\n" + selected_code
            
            result = self.execute_code_on_envs(selected_code)
            if result:
                parsed_code, success, exec_results, scores, states, total_execution_time = result
                if success:
                    # smart_print("The code was successfully executed on all environments.", agent_name=self.name)
                    print("Program code: ", parsed_code['program_code'])
                    print("Main function: ", parsed_code['main_function']['name'])
                    for env in self.envs:
                        no_runtime_error, exec_result = env.step(f"{prebuilt_code + parsed_code['program_code']}\n{parsed_code['main_function']['name']}(bot)")
                        if no_runtime_error:
                            smart_print(f"Executed successfully for environment {env.id}", agent_name=self.name)
                        else:
                            smart_print(f"Error in environment {env.id}: {exec_result}", agent_name=self.name)
                else:
                    smart_print("Failed to execute the code.", agent_name=self.name)
        else:
            smart_print("No code was selected by the LLM.", agent_name=self.name)

        # if selected_code:
        #     # Execute the code on the environments and handle errors
        #     print("Executing the selected code on the environments...")
        #     result = self.execute_code_on_envs(selected_code)
        #     if result:
        #         parsed_code, success, _, _, _, _ = result
        #         if success:
        #             # The code was successfully verified; we can now execute it
        #             code_to_run = parsed_code['program_code']
        #             exec_locals = {}
        #             try:
        #                 env = self.envs[0]  # Assuming there is only one environment
        #                 if isinstance(env, SWEBenchEnvironment):
        #                     print("Executing code on SWEBench environment")
        #                     print("Code to run:", code_to_run)
        #                     no_runtime_error, exec_result = env.step(f"{code_to_run}\n{parsed_code['main_function']['name']}(bot)")
        #                     if no_runtime_error:
        #                         print(f"Executed successfully for environment {env.id}")
        #                         smart_print(f"Executed successfully for environment {env.id} with output {exec_result}", agent_name=self.name)
        #                     else:
        #                         print(f"Error in environment {env.id}: {exec_result}")
        #                         smart_print(f"Error in environment {env.id}: {exec_result}", agent_name=self.name)
        #                 else:
        #                     # Execute the code in a secure context
        #                     exec(code_to_run, globals(), exec_locals)
        #                     main_function_name = parsed_code['main_function']['name']
        #                     if main_function_name in exec_locals:
        #                         # Iterate over each environment
        #                         for env in self.envs:
        #                             bot = env.synthesis_manager  # Or env.bot if the bot is an attribute of the environment
        #                             # Call the main function with the appropriate parameters
        #                             exec_locals[main_function_name](bot)
        #                             answer = bot.document.document_content.sections_list
        #                             temp = ""
        #                             for item in answer:
        #                                 temp += item.title + "\n" + item.content + "\n"
        #                             smart_print(f"Response for document {bot.document.title} (Id: {env.id}): {temp}", agent_name=self.name)
        #                     else:
        #                         smart_print("The main function was not found in the selected code.", agent_name=self.name)
        #             except Exception as e:
        #                 smart_print(f"Error during code execution: {e}", agent_name=self.name)
        #         else:
        #             smart_print("The code failed the tests and will not be executed.", agent_name=self.name)
        #     else:
        #         smart_print("Failed to execute the code.", agent_name=self.name)
        # else:
        #     smart_print("No code was selected by the LLM.", agent_name=self.name)

    def execute_code_on_envs(self, code_str):
        # Create an instance of CodingAgent
        coding_agent = CodingAgent(
            default_llm_choice=self.llm,
            envs=self.envs,
            premium_llm_choice=None,
            problem_prompts_subdir=self.problem_prompts_subdir,
            skip_rounds=self.skip_rounds,
            llmORchains_list=self.llmORchains_list,
            optuna=self.optuna_opti,
            model_choice=self.model_choice,
            special_criteria=None,
            num_parallel_inferences=1
        )

        # Use the parse_ai_generated_code method to analyze the code
        parse_success, parsed_code_or_error = coding_agent.parse_ai_generated_code(
            message=code_str,
            required_bot_arg='bot',  # If your main function needs to accept 'bot' as an argument
            automatic_tests=True
        )

        if parse_success:
            parsed_code = parsed_code_or_error
            smart_print("Code parsed successfully", agent_name=self.name)
            return parsed_code, True, None, None, None, None

            # print("Parsed code: ", parsed_code)
            # Use the run_tests_on_code method to verify the code
            result = coding_agent.run_tests_on_code(
                message="",
                parsed_code=parsed_code,
                skip_already_processed=False,
                output_id=None,
                restore_state=True,
                custom_agent=self.name
            )

            # Process the result as before
            if result:
                parsed_code, success, exec_results, scores, states, total_execution_time = result
                if success:
                    smart_print("The code was successfully executed on all environments.", agent_name=self.name)
                else:
                    smart_print("The code encountered errors on some environments.", agent_name=self.name)
                    for idx, (no_runtime_error, exec_result) in enumerate(zip(exec_results, states)):
                        if not no_runtime_error:
                            smart_print(f"Error in environment {self.envs[idx].id}: {exec_result}",
                                        agent_name=self.name)
                smart_print(f"Total execution time: {total_execution_time:.2f} seconds", agent_name=self.name)
                return result
            else:
                smart_print("Failed to execute the code.", agent_name=self.name)
                return None
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


def import_functions_from_directory(regex=".*"):
    """
    Imports functions from files in the functions directory that match the given regex pattern,
    excluding directories.
    """
    functions = {}
    for file in os.listdir("functions"):
        file_path = os.path.join("functions", file)
        if os.path.isfile(file_path) and re.match(regex, file):
            with open(file_path, "r") as f:
                code = f.read()
                functions[file.replace(".py", "")] = code
    return functions


# Main learning loop orchestration functions
def run_4agents_learning_loop(default_llm_key, premium_llm_key, test_environments=None,
                              manual_validation_to_capitalize=True, problem_prompts_subdir=None,
                              max_coding_attempts=4, include_code=None, selected_successful_functions=None,
                              selected_failed_functions=None, agtask_premium_llm_by_default=True,
                              agtask_skip_rounds=0, agcoding_skip_rounds=0, agvalidation_skip_rounds=0,
                              agcapitalize_skip_rounds=0, llmORchains_list=None, model_choice=None,
                              optuna_opti=None, allow_custom_score_state_functions=False,
                              params_user_message=None, max_execution_time=900, special_criteria=None, temperature_max=1,
                              agcoach_num_parallel_inferences=1, fixed_coach=False, return_array=False, agcoding_num_parallel_inferences=1,
                              continue_each_loop=False,unique_id=None):
    scores = None

    print("Starting learning loop...")

    if params_user_message is None and optuna_opti is None:
        params_user_message = {
            'sources': ["learnt", "failed", "default"],
            'num': [2, 3, 2],
            'format': ["json", "Jinja2", "Markdown"]
        }

    smart_print(str(max_execution_time), "orchestrate_agents", "time_end")
    # smart_print(unique_id, "orchestrate_agents", "XP_unique_id", optional=True)

    time_end = time.time() + max_execution_time

    agent_taskreco: TaskIdentificationAgent = TaskIdentificationAgent(default_llm_key, test_environments, premium_llm_choice=premium_llm_key,
                                             problem_prompts_subdir=problem_prompts_subdir,
                                             premium_llm_by_default=agtask_premium_llm_by_default,
                                             skip_rounds=agtask_skip_rounds, llmORchains_list=llmORchains_list,
                                             optuna=optuna_opti, model_choice=(
            model_choice['taskreco' if 'taskreco' in model_choice else 'coach'] if type(
            model_choice) == dict else model_choice), criteria=params_user_message,
            temperature_max=temperature_max,
            num_parallel_inferences=agcoach_num_parallel_inferences, fixed_coach=fixed_coach, 
            special_criteria=special_criteria)

    agent_coding = CodingAgent(default_llm_key, test_environments, premium_llm_choice=premium_llm_key,
                               problem_prompts_subdir=problem_prompts_subdir, skip_rounds=agcoding_skip_rounds,
                               llmORchains_list=llmORchains_list, optuna=optuna_opti, model_choice=(
            model_choice['coding' if 'coding' in model_choice else 'coder'] if type(
                model_choice) == dict else model_choice), special_criteria=special_criteria,
                               num_parallel_inferences=agcoding_num_parallel_inferences)
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
        HumanLLMMonitor.step_id = str(random.randint(0, 1000000))
        # HumanLLMMonitor.step_id = str(uuid.uuid4())
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
            if optuna_opti or smart_input("Do you want to capitalize this try as a 'failed task' to avoid this task to be proposed as a next best task ? (yes/no): ", "orchestrate_agents", message_type="VALIDATION_INFO").strip().upper() in ["Y", "YES"]:
                agent_capitalize.capitalize_failed_tasks(task_description, parsed_code)
        if optuna_opti:
            continue_identifying_tasks = continue_each_loop
        else:
            answer = smart_input("Do you want to:\n- search for a new task after reseting to empty documents (Y/YES) ?\n- search for a new task based based on the status of documents after applying the task you just validated (N/NO/Enter) ?\n- or just exit the program (E/EXIT) ?", "orchestrate_agents",  message_type="VALIDATION_INFO").strip().upper()
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

               
            # add total_score_weighted_with_stats to total_scores
            total_scores.append(total_score_weighted_with_stats)
        
            total_scores.append(0)

    # print status of: continue_identifying_tasks and time.time() < time_end


    if return_array:
        return total_scores
    else:
        return max(total_scores)


def run_planner(default_llm_key, premium_llm_key, test_environments=None,
                manual_validation_to_capitalize=True, problem_prompts_subdir=None,
                max_coding_attempts=4, include_code=None, selected_successful_functions=None,
                selected_failed_functions=None, agtask_premium_llm_by_default=True,
                agtask_skip_rounds=0, agcoding_skip_rounds=0, agvalidation_skip_rounds=0,
                agcapitalize_skip_rounds=0, llmORchains_list=None, model_choice=None,
                optuna_opti=None, allow_custom_score_state_functions=False,
                params_user_message=None, max_execution_time=900, special_criteria=None, temperature_max=1,
                agcoach_num_parallel_inferences=1, fixed_coach=False, unique_id=None, return_array=False, agcoding_num_parallel_inferences=1,
                continue_each_loop=False, skip_rounds=0, functions_to_import=None):

    # Initialize unique_id
    if unique_id is None:
        unique_id = f"{socket.gethostname()}_{datetime.now().strftime('%d-%m-%Y-%H-%M-%S')}"

    if unique_id is not False:
        if UnifiedVectorDB.unique_collection_id is None:
            UnifiedVectorDB.set_unique_collection_id(unique_id)

    # Initialize HumanLLMMonitor databases
    HumanLLMMonitor._check_and_init_vector_db(embedding_function=embedding_function, reset_db_indices=reset_db_indices)
    HumanLLMMonitor.check_init_class_db(force=True)

    if functions_to_import is not None:
        # Imports the functions with the regex pattern given from functions directory into the elastic database
        functions = import_functions_from_directory(functions_to_import)
        print("Imported functions:", type(functions))
        for function in functions.items():
            # print("Function:", function)
            serialized_entry = json.dumps({
                "time": datetime.now().isoformat(),
                "main_function_name": function[0],
                "program_code": function[1],
                "tool_description": "",
                "task_description": "",
            }, default=lambda o: o.__dict__ if hasattr(o, '__dict__') else str(o))
            tags = {"host": f"{socket.gethostname()}-{uuid.getnode()}", "step_id": HumanLLMMonitor.step_id}
            print("Adding learnt task:", HumanLLMMonitor.add_learnt_task(serialized_entry, tags))
        
    # return
    successful_tasks = HumanLLMMonitor.get_learnt_tasks()

    print("Successful tasks:", len(successful_tasks))

    successful_tasks_list = [task for task in successful_tasks]
    smart_print(json.dumps(successful_tasks_list), "orchestrate_agents", "successful_tasks_list")

    # Initialize WebSocket server if used
    if HumanLLMMonitor.use_websocket:
        if HumanLLMMonitor.websocket_server is None:
            HumanLLMMonitor.initialize_websocket_server()

    # Select the problem prompts subdirectory if not provided
    if problem_prompts_subdir is None:
        # Get the list of subdirectories in the 'prompts' directory
        problem_prompts_subdirs = [name for name in os.listdir("prompts") if os.path.isdir(os.path.join("prompts", name))]
        default_subdir = problem_prompts_subdirs[0] if problem_prompts_subdirs else ""
        choice = smart_input("Enter a capital letter for subdirectory (leave empty for default): " + "; ".join(
            f"\n[{i}] {subdir}" for i, subdir in zip(string.ascii_uppercase, problem_prompts_subdirs)) + " ?", "run_planner")
        problem_prompts_subdir = problem_prompts_subdirs[ord(choice) - 65] if choice and choice.isupper() and ord(
            choice) - 65 in range(len(problem_prompts_subdirs)) else default_subdir

    question = smart_input("Please formulate your question (or exit with q/e/quit/exit): ", agent_name='PlannerAgent').capitalize()
    # Initialize test environments if not provided
    question_command = question
    if test_environments is None:
        pattern = r'\b[\w-]+__[\w-]+-\d+\b'
        instance_ids = re.findall(pattern, question)
        if instance_ids:
            print('Instance ID detected.')
            instance_id = instance_ids[0].lower()
            print(instance_id)
            question_command = question.replace(instance_ids[0], '')
            dataset = load_dataset(path="ahsanirfan961/swe-bech-lite-bm25-13k-take3", split='train')
            problem = None
            for data in dataset:
                if data['instance_id'] == instance_id:
                    problem = data
                    print('Problem set found.')
                    break
            if problem is not None:
                print('Creating SWE Environment.')
                test_environments = [SWEBenchEnvironment(swe_data=SWEProblem.parse_obj(problem))]
            else:
                raise ValueError("Invalid instance ID.")
        else:
            print('No instance ID detected.')
            env_type = "default"
            manager = EnvironmentManager(env_type)
            test_environments = [manager.get_environment()]

    # Initialize the PlannerAgent with the correct parameters
    planner = PlannerAgent(
        default_llm_choice=default_llm_key,
        envs=test_environments,
        premium_llm_choice=premium_llm_key,
        skip_rounds=skip_rounds,
        llmORchains_list=llmORchains_list,
        optuna=optuna_opti,
        model_choice=model_choice,
        special_criteria=special_criteria,
        num_parallel_inferences=agcoach_num_parallel_inferences
    )
    # Ask the user to formulate their question using smart_input
    smart_print(f"Question: {question.capitalize()}", agent_name='PlannerAgent')
    while question not in ['q', 'Q', 'quit', 'Quit', 'QUIT', 'e', 'E', 'exit', 'Exit', 'EXIT']:
        # if question in ['LEARN', 'learn', 'Learn','SWE','LearnSwe','learnSwe']:
        if "learn" in question_command.lower():
            # User wants to use the learning loop
            run_4agents_learning_loop(default_llm_key=default_llm_key,
                                      premium_llm_key=premium_llm_key,
                                      llmORchains_list=llmORchains_list,
                                      test_environments=test_environments,
                                      manual_validation_to_capitalize=manual_validation_to_capitalize,
                                      problem_prompts_subdir=problem_prompts_subdir,
                                      max_coding_attempts=max_coding_attempts,
                                      include_code=include_code,
                                      selected_successful_functions=selected_successful_functions,
                                      selected_failed_functions=selected_failed_functions,
                                      max_execution_time=max_execution_time,
                                      agtask_premium_llm_by_default=agtask_premium_llm_by_default,
                                      agtask_skip_rounds=agtask_skip_rounds,  # Auto-test: 1
                                      agcoding_skip_rounds=agcoding_skip_rounds,  # Auto-test: 4
                                      agvalidation_skip_rounds=agvalidation_skip_rounds,  # Auto-test: 4
                                      agcapitalize_skip_rounds=agcapitalize_skip_rounds,
                                      agcoding_num_parallel_inferences=agcoding_num_parallel_inferences)
        else:
            temp = planner.plan(question)
            if temp == "no code available":
                run_4agents_learning_loop(default_llm_key=default_llm_key,
                                          premium_llm_key=premium_llm_key,
                                          llmORchains_list=llmORchains_list,
                                          test_environments=envs,
                                          manual_validation_to_capitalize=manual_validation_to_capitalize,
                                          problem_prompts_subdir=problem_prompts_subdir,
                                          max_coding_attempts=max_coding_attempts,
                                          include_code=include_code,
                                          selected_successful_functions=selected_successful_functions,
                                          selected_failed_functions=selected_failed_functions,
                                          max_execution_time=max_execution_time,
                                          agtask_premium_llm_by_default=agtask_premium_llm_by_default,
                                          agtask_skip_rounds=agtask_skip_rounds,  # Auto-test: 1
                                          agcoding_skip_rounds=agcoding_skip_rounds,  # Auto-test: 4
                                          agvalidation_skip_rounds=agvalidation_skip_rounds,  # Auto-test: 4
                                          agcapitalize_skip_rounds=agcapitalize_skip_rounds,
                                          agcoding_num_parallel_inferences=agcoding_num_parallel_inferences,
                                          unique_id=unique_id)
        question = smart_input("Please formulate your question (or exit with q/e/quit/exit): ", agent_name='PlannerAgent')
        smart_print(f"Question: {question.capitalize()}", agent_name='PlannerAgent')


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

def coding_and_validation_loop(agent_coding: CodingAgent, agent_validation: ValidationAgent, task_description, max_attempts,
                               extra_manual_validation_to_capitalize=True, continue_even_if_successful=True,
                               optuna=None, end_time=None):
    
    print("Starting coding and validation loop...")

    metadata = {'step_id': HumanLLMMonitor.step_id}
    # Retrieve data
    previous_errors = HumanLLMMonitor.get_agent_data(agent_coding.name, 'previous_errors', metadata_filter=metadata) or []
    previous_codes = HumanLLMMonitor.get_agent_data(agent_coding.name, 'previous_codes', metadata_filter=metadata) or []
    previous_scores = HumanLLMMonitor.get_agent_data(agent_coding.name, 'previous_scores', metadata_filter=metadata) or []
    unique_codes = set(HumanLLMMonitor.get_agent_data(agent_coding.name, 'unique_codes', metadata_filter=metadata) or [])
    successful_codes = HumanLLMMonitor.get_agent_data(agent_coding.name, 'successful_codes',metadata_filter=metadata) or []
    all_results = HumanLLMMonitor.get_agent_data(agent_coding.name, 'all_results', metadata_filter=metadata) or []

    print("Previous errors: ", previous_errors)
    print("Previous codes: ", previous_codes)
    print("Previous scores: ", previous_scores)
    print("Unique codes: ", unique_codes)
    print("Successful codes: ", successful_codes)
    print("All results: ", all_results)

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
            smart_print(f"Generated code:\n{parsed_code['program_code']}\n*******\nOutput of code execution:\n{exec_result}\n".replace("\\n", "\n"), "coding_and_validation_loop", "coding_and_validation_loop RESULT")
            validation_agent_feedback = agent_validation.validate_code(parsed_code["program_code"], no_runtime_error,
                                                                       exec_result, task=task_description,
                                                                       scores=scores, env_states=env_states, human_evaluation_required=True)
            smart_print("Agent validation 'feedback' currently only support 1 feedback", "coding_and_validation_loop","coding_and_validation_loop WARNING")
            validation_agent_feedback = validation_agent_feedback[0]
            afb = validation_agent_feedback.content.replace('\\n', '\n')
            smart_print("#" * 20 + f"\nAgent validation feedback: {afb}", "coding_and_validation_loop", "coding_and_validation_loop RESULT")

            if extra_manual_validation_to_capitalize:
                validated = (smart_input(
                    "#" * 20 + f"\nADD THIS FUNCTION TO LIBRARY ? Please enter 'yes' if this a success and you want to add this function to library, 'no' if this failed: ",
                    "coding_and_validation_loop").lower() in ["yes", "y", True])
            else:
                validated = get_success_value_in_text(afb) in ["yes", "y", True]

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
                successful_codes.append((parsed_code, validation_agent_feedback.content, scores))
                HumanLLMMonitor.add_agent_data(agent_coding.name, 'successful_codes', successful_codes, metadata=metadata)

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
    percentage_no_runtime_error = (sum(1 for _, no_runtime_error, _, _, _, _ in all_results if no_runtime_error) / len(all_results)) if all_results else 0
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
    import pickle

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
    # documents=[{ 'id':"cf0d353c-b43b-4a79-88f9-42c2c84cf75e",
    #             'title':"Complex QA and language models hybrid architectures, Survey",
    #         'context':"This paper reviews the state-of-the-art of language models architectures and strategies for 'complex' question-answering (QA, CQA, CPS) with a focus on hybridization. Large Language Models (LLM) are good at leveraging public data on standard problems but once you want to tackle more specific complex questions or problems (e.g. How does the concept of personal freedom vary between different cultures ? What is the best mix of power generation methods to reduce climate change ?) you may need specific architecture, knowledge, skills, methods, sensitive data protection, explainability, human approval and versatile feedback... Recent projects like ChatGPT and GALACTICA have allowed non-specialists to grasp the great potential as well as the equally strong limitations of LLM in complex QA. In this paper, we start by reviewing required skills and evaluation techniques. We integrate findings from the robust community edited research papers BIG, BLOOM and HELM which open source, benchmark and analyze limits and challenges of LLM in terms of tasks complexity and strict evaluation on accuracy (e.g. fairness, robustness, toxicity, ...) as a baseline. We discuss some challenges associated with complex QA, including domain adaptation, decomposition and efficient multi-step QA, long form and non-factoid QA, safety and multi-sensitivity data protection, multimodal search, hallucinations, explainability and truthfulness, temporal reasoning. We analyze current solutions and promising research trends, using elements such as: hybrid LLM architectural patterns, training and prompting strategies, active human reinforcement learning supervised with AI, neuro-symbolic and structured knowledge grounding, program synthesis, iterated decomposition and others.",
    #         'target_file_path': "env/IR_CPS_TechSynthesis/document_embedding_analysis/output/arxiv/Complex QA and language models hybrid architectures Survey.json"},
    #         { 'id':"42252c6c-12f3-4edf-9045-8acd69bc3356",
    #             'title':"Macroeconomic Effects of Inflation Targeting A Survey of the Empirical  Literature",
    #         'context':"This paper surveys the empirical literature of inflation targeting. The main findings from our review are the following: there is robust empirical evidence that larger and more developed countries are more likely to adopt the IT regime; the introduction of this regime is conditional on previous disinflation, greater exchange rate flexibility, central bank independence, and higher level of financial development; the empirical evidence has failed to provide convincing evidence that IT itself may serve as an effective tool for stabilizing inflation expectations and for reducing inflation persistence; the empirical research focused on advanced economies has failed to provide convincing evidence on the beneficial effects of IT on inflation performance, while there is some evidence that the gains from the IT regime may have been more prevalent in the emerging market economies; there is not convincing evidence that IT is associated with either higher output growth or lower output variability; the empirical research suggests that IT may have differential effects on exchange-rate volatility in advanced economies versus EMEs; although the empirical evidence on the impact of IT on fiscal policy is quite limited, it supports the idea that IT indeed improves fiscal discipline; the empirical support to the proposition that IT is associated with lower disinflation costs seems to be rather weak. Therefore, the accumulated empirical literature implies that IT does not produce superior macroeconomic benefits in comparison with the alternative monetary strategies or, at most, they are quite modest.",
    #         'target_file_path': "env/IR_CPS_TechSynthesis/document_embedding_analysis/output/arxiv/Macroeconomic Effects of Inflation Targeting A Survey of the Empirical  Literature.json"}]

    # documents = [{
    #     "instance_id": "django__django-14855",
    #     "text": '''You will be provided with a partial code base and an issue statement explaining a problem to resolve. <issue> Wrong URL generated by get_admin_url for readonly field in custom Admin Site Description When a model containing a ForeignKey field is viewed (or edited) in a custom Admin Site, and that ForeignKey field is listed in readonly_fields, the url generated for the link is /admin/... instead of /custom-admin/.... This appears to be caused by the following line in django.contrib.admin.helpers get_admin_url: url = reverse(url_name, args=[quote(remote_obj.pk)]) Other parts of the admin use the current_app keyword parameter to identify the correct current name of the Admin Site. (See django.contrib.admin.options.ModelAdmin response_add as just one example) I have been able to correct this specific issue by replacing the above line with: url = reverse( url_name, args=[quote(remote_obj.pk)], current_app=self.model_admin.admin_site.name ) However, I don't know if there are any side effects and I have not yet run the full suite of tests on this. Mostly looking for feedback whether I'm on the right track. </issue> <code> [start of README.rst] 1 ====== 2 Django 3 ====== 4 5 Django is a high-level Python web framework that encourages rapid development 6 and clean, pragmatic design. Thanks for checking it out. 7 8 All documentation is in the "``docs``" directory and online at 9 https://docs.djangoproject.com/en/stable/. If you're just getting started, 10 here's how we recommend you read the docs: 11 12 * First, read ``docs/intro/install.txt`` for instructions on installing Django. 13 14 * Next, work through the tutorials in order (``docs/intro/tutorial01.txt``, 15 ``docs/intro/tutorial02.txt``, etc.). 16 17 * If you want to set up an actual deployment server, read 18 ``docs/howto/deployment/index.txt`` for instructions. 19 20 * You'll probably want to read through the topical guides (in ``docs/topics``) 21 next; from there you can jump to the HOWTOs (in ``docs/howto``) for specific 22 problems, and check out the reference (``docs/ref``) for gory details. 23 24 * See ``docs/README`` for instructions on building an HTML version of the docs. 25 26 Docs are updated rigorously. If you find any problems in the docs, or think 27 they should be clarified in any way, please take 30 seconds to fill out a 28 ticket here: https://code.djangoproject.com/newticket 29 30 To get more help: 31 32 * Join the ``#django`` channel on ``irc.libera.chat``. Lots of helpful people 33 hang out there. See https://web.libera.chat if you're new to IRC. 34 35 * Join the django-users mailing list, or read the archives, at 36 https://groups.google.com/group/django-users. 37 38 To contribute to Django: 39 40 * Check out https://docs.djangoproject.com/en/dev/internals/contributing/ for 41 information about getting involved. 42 43 To run Django's test suite: 44 45 * Follow the instructions in the "Unit tests" section of 46 ``docs/internals/contributing/writing-code/unit-tests.txt``, published online at 47 https://docs.djangoproject.com/en/dev/internals/contributing/writing-code/unit-tests/#running-the-unit-tests 48 49 Supporting the Development of Django 50 ==================================== 51 52 Django's development depends on your contributions. 53 54 If you depend on Django, remember to support the Django Software Foundation: https://www.djangoproject.com/fundraising/ 55 [end of README.rst] [start of django/contrib/admin/helpers.py] 1 import json 2 3 from django import forms 4 from django.contrib.admin.utils import ( 5 display_for_field, flatten_fieldsets, help_text_for_field, label_for_field, 6 lookup_field, quote, 7 ) 8 from django.core.exceptions import ObjectDoesNotExist 9 from django.db.models.fields.related import ( 10 ForeignObjectRel, ManyToManyRel, OneToOneField, 11 ) 12 from django.forms.utils import flatatt 13 from django.template.defaultfilters import capfirst, linebreaksbr 14 from django.urls import NoReverseMatch, reverse 15 from django.utils.html import conditional_escape, format_html 16 from django.utils.safestring import mark_safe 17 from django.utils.translation import gettext, gettext_lazy as _ 18 19 ACTION_CHECKBOX_NAME = '_selected_action' 20 21 22 class ActionForm(forms.Form): 23 action = forms.ChoiceField(label=_('Action:')) 24 select_across = forms.BooleanField( 25 label='', 26 required=False, 27 initial=0, 28 widget=forms.HiddenInput({'class': 'select-across'}), 29 ) 30 31 32 checkbox = forms.CheckboxInput({'class': 'action-select'}, lambda value: False) 33 34 35 class AdminForm: 36 def __init__(self, form, fieldsets, prepopulated_fields, readonly_fields=None, model_admin=None): 37 self.form, self.fieldsets = form, fieldsets 38 self.prepopulated_fields = [{ 39 'field': form[field_name], 40 'dependencies': [form[f] for f in dependencies] 41 } for field_name, dependencies in prepopulated_fields.items()] 42 self.model_admin = model_admin 43 if readonly_fields is None: 44 readonly_fields = () 45 self.readonly_fields = readonly_fields 46 47 def __repr__(self): 48 return ( 49 f'<{self.__class__.__qualname__}: ' 50 f'form={self.form.__class__.__qualname__} ' 51 f'fieldsets={self.fieldsets!r}>' 52 ) 53 54 def __iter__(self): 55 for name, options in self.fieldsets: 56 yield Fieldset( 57 self.form, name, 58 readonly_fields=self.readonly_fields, 59 model_admin=self.model_admin, 60 **options 61 ) 62 63 @property 64 def errors(self): 65 return self.form.errors 66 67 @property 68 def non_field_errors(self): 69 return self.form.non_field_errors 70 71 @property 72 def media(self): 73 media = self.form.media 74 for fs in self: 75 media = media + fs.media 76 return media 77 78 79 class Fieldset: 80 def __init__(self, form, name=None, readonly_fields=(), fields=(), classes=(), 81 description=None, model_admin=None): 82 self.form = form 83 self.name, self.fields = name, fields 84 self.classes = ' '.join(classes) 85 self.description = description 86 self.model_admin = model_admin 87 self.readonly_fields = readonly_fields 88 89 @property 90 def media(self): 91 if 'collapse' in self.classes: 92 return forms.Media(js=['admin/js/collapse.js']) 93 return forms.Media() 94 95 def __iter__(self): 96 for field in self.fields: 97 yield Fieldline(self.form, field, self.readonly_fields, model_admin=self.model_admin) 98 99 100 class Fieldline: 101 def __init__(self, form, field, readonly_fields=None, model_admin=None): 102 self.form = form # A django.forms.Form instance 103 if not hasattr(field, "__iter__") or isinstance(field, str): 104 self.fields = [field] 105 else: 106 self.fields = field 107 self.has_visible_field = not all( 108 field in self.form.fields and self.form.fields[field].widget.is_hidden 109 for field in self.fields 110 ) 111 self.model_admin = model_admin 112 if readonly_fields is None: 113 readonly_fields = () 114 self.readonly_fields = readonly_fields 115 116 def __iter__(self): 117 for i, field in enumerate(self.fields): 118 if field in self.readonly_fields: 119 yield AdminReadonlyField(self.form, field, is_first=(i == 0), model_admin=self.model_admin) 120 else: 121 yield AdminField(self.form, field, is_first=(i == 0)) 122 123 def errors(self): 124 return mark_safe( 125 '\n'.join( 126 self.form[f].errors.as_ul() for f in self.fields if f not in self.readonly_fields 127 ).strip('\n') 128 ) 129 130 131 class AdminField: 132 def __init__(self, form, field, is_first): 133 self.field = form[field] # A django.forms.BoundField instance 134 self.is_first = is_first # Whether this field is first on the line 135 self.is_checkbox = isinstance(self.field.field.widget, forms.CheckboxInput) 136 self.is_readonly = False 137 138 def label_tag(self): 139 classes = [] 140 contents = conditional_escape(self.field.label) 141 if self.is_checkbox: 142 classes.append('vCheckboxLabel') 143 144 if self.field.field.required: 145 classes.append('required') 146 if not self.is_first: 147 classes.append('inline') 148 attrs = {'class': ' '.join(classes)} if classes else {} 149 # checkboxes should not have a label suffix as the checkbox appears 150 # to the left of the label. 151 return self.field.label_tag( 152 contents=mark_safe(contents), attrs=attrs, 153 label_suffix='' if self.is_checkbox else None, 154 ) 155 156 def errors(self): 157 return mark_safe(self.field.errors.as_ul()) 158 159 160 class AdminReadonlyField: 161 def __init__(self, form, field, is_first, model_admin=None): 162 # Make self.field look a little bit like a field. This means that 163 # {{ field.name }} must be a useful class name to identify the field. 164 # For convenience, store other field-related data here too. 165 if callable(field): 166 class_name = field.__name__ if field.__name__ != '<lambda>' else '' 167 else: 168 class_name = field 169 170 if form._meta.labels and class_name in form._meta.labels: 171 label = form._meta.labels[class_name] 172 else: 173 label = label_for_field(field, form._meta.model, model_admin, form=form) 174 175 if form._meta.help_texts and class_name in form._meta.help_texts: 176 help_text = form._meta.help_texts[class_name] 177 else: 178 help_text = help_text_for_field(class_name, form._meta.model) 179 180 if field in form.fields: 181 is_hidden = form.fields[field].widget.is_hidden 182 else: 183 is_hidden = False 184 185 self.field = { 186 'name': class_name, 187 'label': label, 188 'help_text': help_text, 189 'field': field, 190 'is_hidden': is_hidden, 191 } 192 self.form = form 193 self.model_admin = model_admin 194 self.is_first = is_first 195 self.is_checkbox = False 196 self.is_readonly = True 197 self.empty_value_display = model_admin.get_empty_value_display() 198 199 def label_tag(self): 200 attrs = {} 201 if not self.is_first: 202 attrs["class"] = "inline" 203 label = self.field['label'] 204 return format_html('<label{}>{}{}</label>', flatatt(attrs), capfirst(label), self.form.label_suffix) 205 206 def get_admin_url(self, remote_field, remote_obj): 207 url_name = 'admin:%s_%s_change' % ( 208 remote_field.model._meta.app_label, 209 remote_field.model._meta.model_name, 210 ) 211 try: 212 url = reverse(url_name, args=[quote(remote_obj.pk)]) 213 return format_html('<a href="{}">{}</a>', url, remote_obj) 214 except NoReverseMatch: 215 return str(remote_obj) 216 217 def contents(self): 218 from django.contrib.admin.templatetags.admin_list import _boolean_icon 219 field, obj, model_admin = self.field['field'], self.form.instance, self.model_admin 220 try: 221 f, attr, value = lookup_field(field, obj, model_admin) 222 except (AttributeError, ValueError, ObjectDoesNotExist): 223 result_repr = self.empty_value_display 224 else: 225 if field in self.form.fields: 226 widget = self.form[field].field.widget 227 # This isn't elegant but suffices for contrib.auth's 228 # ReadOnlyPasswordHashWidget. 229 if getattr(widget, 'read_only', False): 230 return widget.render(field, value) 231 if f is None: 232 if getattr(attr, 'boolean', False): 233 result_repr = _boolean_icon(value) 234 else: 235 if hasattr(value, "__html__"): 236 result_repr = value 237 else: 238 result_repr = linebreaksbr(value) 239 else: 240 if isinstance(f.remote_field, ManyToManyRel) and value is not None: 241 result_repr = ", ".join(map(str, value.all())) 242 elif ( 243 isinstance(f.remote_field, (ForeignObjectRel, OneToOneField)) and 244 value is not None 245 ): 246 result_repr = self.get_admin_url(f.remote_field, value) 247 else: 248 result_repr = display_for_field(value, f, self.empty_value_display) 249 result_repr = linebreaksbr(result_repr) 250 return conditional_escape(result_repr) 251 252 253 class InlineAdminFormSet: 254 """ 255 A wrapper around an inline formset for use in the admin system. 256 """ 257 def __init__(self, inline, formset, fieldsets, prepopulated_fields=None, 258 readonly_fields=None, model_admin=None, has_add_permission=True, 259 has_change_permission=True, has_delete_permission=True, 260 has_view_permission=True): 261 self.opts = inline 262 self.formset = formset 263 self.fieldsets = fieldsets 264 self.model_admin = model_admin 265 if readonly_fields is None: 266 readonly_fields = () 267 self.readonly_fields = readonly_fields 268 if prepopulated_fields is None: 269 prepopulated_fields = {} 270 self.prepopulated_fields = prepopulated_fields 271 self.classes = ' '.join(inline.classes) if inline.classes else '' 272 self.has_add_permission = has_add_permission 273 self.has_change_permission = has_change_permission 274 self.has_delete_permission = has_delete_permission 275 self.has_view_permission = has_view_permission 276 277 def __iter__(self): 278 if self.has_change_permission: 279 readonly_fields_for_editing = self.readonly_fields 280 else: 281 readonly_fields_for_editing = self.readonly_fields + flatten_fieldsets(self.fieldsets) 282 283 for form, original in zip(self.formset.initial_forms, self.formset.get_queryset()): 284 view_on_site_url = self.opts.get_view_on_site_url(original) 285 yield InlineAdminForm( 286 self.formset, form, self.fieldsets, self.prepopulated_fields, 287 original, readonly_fields_for_editing, model_admin=self.opts, 288 view_on_site_url=view_on_site_url, 289 ) 290 for form in self.formset.extra_forms: 291 yield InlineAdminForm( 292 self.formset, form, self.fieldsets, self.prepopulated_fields, 293 None, self.readonly_fields, model_admin=self.opts, 294 ) 295 if self.has_add_permission: 296 yield InlineAdminForm( 297 self.formset, self.formset.empty_form, 298 self.fieldsets, self.prepopulated_fields, None, 299 self.readonly_fields, model_admin=self.opts, 300 ) 301 302 def fields(self): 303 fk = getattr(self.formset, "fk", None) 304 empty_form = self.formset.empty_form 305 meta_labels = empty_form._meta.labels or {} 306 meta_help_texts = empty_form._meta.help_texts or {} 307 for i, field_name in enumerate(flatten_fieldsets(self.fieldsets)): 308 if fk and fk.name == field_name: 309 continue 310 if not self.has_change_permission or field_name in self.readonly_fields: 311 form_field = empty_form.fields.get(field_name) 312 widget_is_hidden = False 313 if form_field is not None: 314 widget_is_hidden = form_field.widget.is_hidden 315 yield { 316 'name': field_name, 317 'label': meta_labels.get(field_name) or label_for_field( 318 field_name, 319 self.opts.model, 320 self.opts, 321 form=empty_form, 322 ), 323 'widget': {'is_hidden': widget_is_hidden}, 324 'required': False, 325 'help_text': meta_help_texts.get(field_name) or help_text_for_field(field_name, self.opts.model), 326 } 327 else: 328 form_field = empty_form.fields[field_name] 329 label = form_field.label 330 if label is None: 331 label = label_for_field(field_name, self.opts.model, self.opts, form=empty_form) 332 yield { 333 'name': field_name, 334 'label': label, 335 'widget': form_field.widget, 336 'required': form_field.required, 337 'help_text': form_field.help_text, 338 } 339 340 def inline_formset_data(self): 341 verbose_name = self.opts.verbose_name 342 return json.dumps({ 343 'name': '#%s' % self.formset.prefix, 344 'options': { 345 'prefix': self.formset.prefix, 346 'addText': gettext('Add another %(verbose_name)s') % { 347 'verbose_name': capfirst(verbose_name), 348 }, 349 'deleteText': gettext('Remove'), 350 } 351 }) 352 353 @property 354 def forms(self): 355 return self.formset.forms 356 357 @property 358 def non_form_errors(self): 359 return self.formset.non_form_errors 360 361 @property 362 def media(self): 363 media = self.opts.media + self.formset.media 364 for fs in self: 365 media = media + fs.media 366 return media 367 368 369 class InlineAdminForm(AdminForm): 370 """ 371 A wrapper around an inline form for use in the admin system. 372 """ 373 def __init__(self, formset, form, fieldsets, prepopulated_fields, original, 374 readonly_fields=None, model_admin=None, view_on_site_url=None): 375 self.formset = formset 376 self.model_admin = model_admin 377 self.original = original 378 self.show_url = original and view_on_site_url is not None 379 self.absolute_url = view_on_site_url 380 super().__init__(form, fieldsets, prepopulated_fields, readonly_fields, model_admin) 381 382 def __iter__(self): 383 for name, options in self.fieldsets: 384 yield InlineFieldset( 385 self.formset, self.form, name, self.readonly_fields, 386 model_admin=self.model_admin, **options 387 ) 388 389 def needs_explicit_pk_field(self): 390 return ( 391 # Auto fields are editable, so check for auto or non-editable pk. 392 self.form._meta.model._meta.auto_field or not self.form._meta.model._meta.pk.editable or 393 # Also search any parents for an auto field. (The pk info is 394 # propagated to child models so that does not need to be checked 395 # in parents.) 396 any(parent._meta.auto_field or not parent._meta.model._meta.pk.editable 397 for parent in self.form._meta.model._meta.get_parent_list()) 398 ) 399 400 def pk_field(self): 401 return AdminField(self.form, self.formset._pk_field.name, False) 402 403 def fk_field(self): 404 fk = getattr(self.formset, "fk", None) 405 if fk: 406 return AdminField(self.form, fk.name, False) 407 else: 408 return "" 409 410 def deletion_field(self): 411 from django.forms.formsets import DELETION_FIELD_NAME 412 return AdminField(self.form, DELETION_FIELD_NAME, False) 413 414 def ordering_field(self): 415 from django.forms.formsets import ORDERING_FIELD_NAME 416 return AdminField(self.form, ORDERING_FIELD_NAME, False) 417 418 419 class InlineFieldset(Fieldset): 420 def __init__(self, formset, *args, **kwargs): 421 self.formset = formset 422 super().__init__(*args, **kwargs) 423 424 def __iter__(self): 425 fk = getattr(self.formset, "fk", None) 426 for field in self.fields: 427 if not fk or fk.name != field: 428 yield Fieldline(self.form, field, self.readonly_fields, model_admin=self.model_admin) 429 430 431 class AdminErrorList(forms.utils.ErrorList): 432 """Store errors for the form/formsets in an add/change view.""" 433 def __init__(self, form, inline_formsets): 434 super().__init__() 435 436 if form.is_bound: 437 self.extend(form.errors.values()) 438 for inline_formset in inline_formsets: 439 self.extend(inline_formset.non_form_errors()) 440 for errors_in_inline_form in inline_formset.errors: 441 self.extend(errors_in_inline_form.values()) 442 [end of django/contrib/admin/helpers.py] [start of django/contrib/admin/models.py] 1 import json 2 3 from django.conf import settings 4 from django.contrib.admin.utils import quote 5 from django.contrib.contenttypes.models import ContentType 6 from django.db import models 7 from django.urls import NoReverseMatch, reverse 8 from django.utils import timezone 9 from django.utils.text import get_text_list 10 from django.utils.translation import gettext, gettext_lazy as _ 11 12 ADDITION = 1 13 CHANGE = 2 14 DELETION = 3 15 16 ACTION_FLAG_CHOICES = ( 17 (ADDITION, _('Addition')), 18 (CHANGE, _('Change')), 19 (DELETION, _('Deletion')), 20 ) 21 22 23 class LogEntryManager(models.Manager): 24 use_in_migrations = True 25 26 def log_action(self, user_id, content_type_id, object_id, object_repr, action_flag, change_message=''): 27 if isinstance(change_message, list): 28 change_message = json.dumps(change_message) 29 return self.model.objects.create( 30 user_id=user_id, 31 content_type_id=content_type_id, 32 object_id=str(object_id), 33 object_repr=object_repr[:200], 34 action_flag=action_flag, 35 change_message=change_message, 36 ) 37 38 39 class LogEntry(models.Model): 40 action_time = models.DateTimeField( 41 _('action time'), 42 default=timezone.now, 43 editable=False, 44 ) 45 user = models.ForeignKey( 46 settings.AUTH_USER_MODEL, 47 models.CASCADE, 48 verbose_name=_('user'), 49 ) 50 content_type = models.ForeignKey( 51 ContentType, 52 models.SET_NULL, 53 verbose_name=_('content type'), 54 blank=True, null=True, 55 ) 56 object_id = models.TextField(_('object id'), blank=True, null=True) 57 # Translators: 'repr' means representation (https://docs.python.org/library/functions.html#repr) 58 object_repr = models.CharField(_('object repr'), max_length=200) 59 action_flag = models.PositiveSmallIntegerField(_('action flag'), choices=ACTION_FLAG_CHOICES) 60 # change_message is either a string or a JSON structure 61 change_message = models.TextField(_('change message'), blank=True) 62 63 objects = LogEntryManager() 64 65 class Meta: 66 verbose_name = _('log entry') 67 verbose_name_plural = _('log entries') 68 db_table = 'django_admin_log' 69 ordering = ['-action_time'] 70 71 def __repr__(self): 72 return str(self.action_time) 73 74 def __str__(self): 75 if self.is_addition(): 76 return gettext('Added “%(object)s”.') % {'object': self.object_repr} 77 elif self.is_change(): 78 return gettext('Changed “%(object)s” — %(changes)s') % { 79 'object': self.object_repr, 80 'changes': self.get_change_message(), 81 } 82 elif self.is_deletion(): 83 return gettext('Deleted “%(object)s.”') % {'object': self.object_repr} 84 85 return gettext('LogEntry Object') 86 87 def is_addition(self): 88 return self.action_flag == ADDITION 89 90 def is_change(self): 91 return self.action_flag == CHANGE 92 93 def is_deletion(self): 94 return self.action_flag == DELETION 95 96 def get_change_message(self): 97 """ 98 If self.change_message is a JSON structure, interpret it as a change 99 string, properly translated. 100 """ 101 if self.change_message and self.change_message[0] == '[': 102 try: 103 change_message = json.loads(self.change_message) 104 except json.JSONDecodeError: 105 return self.change_message 106 messages = [] 107 for sub_message in change_message: 108 if 'added' in sub_message: 109 if sub_message['added']: 110 sub_message['added']['name'] = gettext(sub_message['added']['name']) 111 messages.append(gettext('Added {name} “{object}”.').format(**sub_message['added'])) 112 else: 113 messages.append(gettext('Added.')) 114 115 elif 'changed' in sub_message: 116 sub_message['changed']['fields'] = get_text_list( 117 [gettext(field_name) for field_name in sub_message['changed']['fields']], gettext('and') 118 ) 119 if 'name' in sub_message['changed']: 120 sub_message['changed']['name'] = gettext(sub_message['changed']['name']) 121 messages.append(gettext('Changed {fields} for {name} “{object}”.').format( 122 **sub_message['changed'] 123 )) 124 else: 125 messages.append(gettext('Changed {fields}.').format(**sub_message['changed'])) 126 127 elif 'deleted' in sub_message: 128 sub_message['deleted']['name'] = gettext(sub_message['deleted']['name']) 129 messages.append(gettext('Deleted {name} “{object}”.').format(**sub_message['deleted'])) 130 131 change_message = ' '.join(msg[0].upper() + msg[1:] for msg in messages) 132 return change_message or gettext('No fields changed.') 133 else: 134 return self.change_message 135 136 def get_edited_object(self): 137 """Return the edited object represented by this log entry.""" 138 return self.content_type.get_object_for_this_type(pk=self.object_id) 139 140 def get_admin_url(self): 141 """ 142 Return the admin URL to edit the object represented by this log entry. 143 """ 144 if self.content_type and self.object_id: 145 url_name = 'admin:%s_%s_change' % (self.content_type.app_label, self.content_type.model) 146 try: 147 return reverse(url_name, args=(quote(self.object_id),)) 148 except NoReverseMatch: 149 pass 150 return None 151 [end of django/contrib/admin/models.py] [start of django/contrib/admin/sites.py] 1 import re 2 from functools import update_wrapper 3 from weakref import WeakSet 4 5 from django.apps import apps 6 from django.conf import settings 7 from django.contrib.admin import ModelAdmin, actions 8 from django.contrib.admin.views.autocomplete import AutocompleteJsonView 9 from django.contrib.auth import REDIRECT_FIELD_NAME 10 from django.core.exceptions import ImproperlyConfigured 11 from django.db.models.base import ModelBase 12 from django.http import ( 13 Http404, HttpResponsePermanentRedirect, HttpResponseRedirect, 14 ) 15 from django.template.response import TemplateResponse 16 from django.urls import NoReverseMatch, Resolver404, resolve, reverse 17 from django.utils.decorators import method_decorator 18 from django.utils.functional import LazyObject 19 from django.utils.module_loading import import_string 20 from django.utils.text import capfirst 21 from django.utils.translation import gettext as _, gettext_lazy 22 from django.views.decorators.cache import never_cache 23 from django.views.decorators.common import no_append_slash 24 from django.views.decorators.csrf import csrf_protect 25 from django.views.i18n import JavaScriptCatalog 26 27 all_sites = WeakSet() 28 29 30 class AlreadyRegistered(Exception): 31 pass 32 33 34 class NotRegistered(Exception): 35 pass 36 37 38 class AdminSite: 39 """ 40 An AdminSite object encapsulates an instance of the Django admin application, ready 41 to be hooked in to your URLconf. Models are registered with the AdminSite using the 42 register() method, and the get_urls() method can then be used to access Django view 43 functions that present a full admin interface for the collection of registered 44 models. 45 """ 46 47 # Text to put at the end of each page's <title>. 48 site_title = gettext_lazy('Django site admin') 49 50 # Text to put in each page's <h1>. 51 site_header = gettext_lazy('Django administration') 52 53 # Text to put at the top of the admin index page. 54 index_title = gettext_lazy('Site administration') 55 56 # URL for the "View site" link at the top of each admin page. 57 site_url = '/' 58 59 enable_nav_sidebar = True 60 61 empty_value_display = '-' 62 63 login_form = None 64 index_template = None 65 app_index_template = None 66 login_template = None 67 logout_template = None 68 password_change_template = None 69 password_change_done_template = None 70 71 final_catch_all_view = True 72 73 def __init__(self, name='admin'): 74 self._registry = {} # model_class class -> admin_class instance 75 self.name = name 76 self._actions = {'delete_selected': actions.delete_selected} 77 self._global_actions = self._actions.copy() 78 all_sites.add(self) 79 80 def __repr__(self): 81 return f'{self.__class__.__name__}(name={self.name!r})' 82 83 def check(self, app_configs): 84 """ 85 Run the system checks on all ModelAdmins, except if they aren't 86 customized at all. 87 """ 88 if app_configs is None: 89 app_configs = apps.get_app_configs() 90 app_configs = set(app_configs) # Speed up lookups below 91 92 errors = [] 93 modeladmins = (o for o in self._registry.values() if o.__class__ is not ModelAdmin) 94 for modeladmin in modeladmins: 95 if modeladmin.model._meta.app_config in app_configs: 96 errors.extend(modeladmin.check()) 97 return errors 98 99 def register(self, model_or_iterable, admin_class=None, **options): 100 """ 101 Register the given model(s) with the given admin class. 102 103 The model(s) should be Model classes, not instances. 104 105 If an admin class isn't given, use ModelAdmin (the default admin 106 options). If keyword arguments are given -- e.g., list_display -- 107 apply them as options to the admin class. 108 109 If a model is already registered, raise AlreadyRegistered. 110 111 If a model is abstract, raise ImproperlyConfigured. 112 """ 113 admin_class = admin_class or ModelAdmin 114 if isinstance(model_or_iterable, ModelBase): 115 model_or_iterable = [model_or_iterable] 116 for model in model_or_iterable: 117 if model._meta.abstract: 118 raise ImproperlyConfigured( 119 'The model %s is abstract, so it cannot be registered with admin.' % model.__name__ 120 ) 121 122 if model in self._registry: 123 registered_admin = str(self._registry[model]) 124 msg = 'The model %s is already registered ' % model.__name__ 125 if registered_admin.endswith('.ModelAdmin'): 126 # Most likely registered without a ModelAdmin subclass. 127 msg += 'in app %r.' % re.sub(r'\.ModelAdmin$', '', registered_admin) 128 else: 129 msg += 'with %r.' % registered_admin 130 raise AlreadyRegistered(msg) 131 132 # Ignore the registration if the model has been 133 # swapped out. 134 if not model._meta.swapped: 135 # If we got **options then dynamically construct a subclass of 136 # admin_class with those **options. 137 if options: 138 # For reasons I don't quite understand, without a __module__ 139 # the created class appears to "live" in the wrong place, 140 # which causes issues later on. 141 options['__module__'] = __name__ 142 admin_class = type("%sAdmin" % model.__name__, (admin_class,), options) 143 144 # Instantiate the admin class to save in the registry 145 self._registry[model] = admin_class(model, self) 146 147 def unregister(self, model_or_iterable): 148 """ 149 Unregister the given model(s). 150 151 If a model isn't already registered, raise NotRegistered. 152 """ 153 if isinstance(model_or_iterable, ModelBase): 154 model_or_iterable = [model_or_iterable] 155 for model in model_or_iterable: 156 if model not in self._registry: 157 raise NotRegistered('The model %s is not registered' % model.__name__) 158 del self._registry[model] 159 160 def is_registered(self, model): 161 """ 162 Check if a model class is registered with this `AdminSite`. 163 """ 164 return model in self._registry 165 166 def add_action(self, action, name=None): 167 """ 168 Register an action to be available globally. 169 """ 170 name = name or action.__name__ 171 self._actions[name] = action 172 self._global_actions[name] = action 173 174 def disable_action(self, name): 175 """ 176 Disable a globally-registered action. Raise KeyError for invalid names. 177 """ 178 del self._actions[name] 179 180 def get_action(self, name): 181 """ 182 Explicitly get a registered global action whether it's enabled or 183 not. Raise KeyError for invalid names. 184 """ 185 return self._global_actions[name] 186 187 @property 188 def actions(self): 189 """ 190 Get all the enabled actions as an iterable of (name, func). 191 """ 192 return self._actions.items() 193 194 def has_permission(self, request): 195 """ 196 Return True if the given HttpRequest has permission to view 197 *at least one* page in the admin site. 198 """ 199 return request.user.is_active and request.user.is_staff 200 201 def admin_view(self, view, cacheable=False): 202 """ 203 Decorator to create an admin view attached to this ``AdminSite``. This 204 wraps the view and provides permission checking by calling 205 ``self.has_permission``. 206 207 You'll want to use this from within ``AdminSite.get_urls()``: 208 209 class MyAdminSite(AdminSite): 210 211 def get_urls(self): 212 from django.urls import path 213 214 urls = super().get_urls() 215 urls += [ 216 path('my_view/', self.admin_view(some_view)) 217 ] 218 return urls 219 220 By default, admin_views are marked non-cacheable using the 221 ``never_cache`` decorator. If the view can be safely cached, set 222 cacheable=True. 223 """ 224 def inner(request, *args, **kwargs): 225 if not self.has_permission(request): 226 if request.path == reverse('admin:logout', current_app=self.name): 227 index_path = reverse('admin:index', current_app=self.name) 228 return HttpResponseRedirect(index_path) 229 # Inner import to prevent django.contrib.admin (app) from 230 # importing django.contrib.auth.models.User (unrelated model). 231 from django.contrib.auth.views import redirect_to_login 232 return redirect_to_login( 233 request.get_full_path(), 234 reverse('admin:login', current_app=self.name) 235 ) 236 return view(request, *args, **kwargs) 237 if not cacheable: 238 inner = never_cache(inner) 239 # We add csrf_protect here so this function can be used as a utility 240 # function for any view, without having to repeat 'csrf_protect'. 241 if not getattr(view, 'csrf_exempt', False): 242 inner = csrf_protect(inner) 243 return update_wrapper(inner, view) 244 245 def get_urls(self): 246 # Since this module gets imported in the application's root package, 247 # it cannot import models from other applications at the module level, 248 # and django.contrib.contenttypes.views imports ContentType. 249 from django.contrib.contenttypes import views as contenttype_views 250 from django.urls import include, path, re_path 251 252 def wrap(view, cacheable=False): 253 def wrapper(*args, **kwargs): 254 return self.admin_view(view, cacheable)(*args, **kwargs) 255 wrapper.admin_site = self 256 return update_wrapper(wrapper, view) 257 258 # Admin-site-wide views. 259 urlpatterns = [ 260 path('', wrap(self.index), name='index'), 261 path('login/', self.login, name='login'), 262 path('logout/', wrap(self.logout), name='logout'), 263 path('password_change/', wrap(self.password_change, cacheable=True), name='password_change'), 264 path( 265 'password_change/done/', 266 wrap(self.password_change_done, cacheable=True), 267 name='password_change_done', 268 ), 269 path('autocomplete/', wrap(self.autocomplete_view), name='autocomplete'), 270 path('jsi18n/', wrap(self.i18n_javascript, cacheable=True), name='jsi18n'), 271 path( 272 'r/<int:content_type_id>/<path:object_id>/', 273 wrap(contenttype_views.shortcut), 274 name='view_on_site', 275 ), 276 ] 277 278 # Add in each model's views, and create a list of valid URLS for the 279 # app_index 280 valid_app_labels = [] 281 for model, model_admin in self._registry.items(): 282 urlpatterns += [ 283 path('%s/%s/' % (model._meta.app_label, model._meta.model_name), include(model_admin.urls)), 284 ] 285 if model._meta.app_label not in valid_app_labels: 286 valid_app_labels.append(model._meta.app_label) 287 288 # If there were ModelAdmins registered, we should have a list of app 289 # labels for which we need to allow access to the app_index view, 290 if valid_app_labels: 291 regex = r'^(?P<app_label>' + '|'.join(valid_app_labels) + ')/$' 292 urlpatterns += [ 293 re_path(regex, wrap(self.app_index), name='app_list'), 294 ] 295 296 if self.final_catch_all_view: 297 urlpatterns.append(re_path(r'(?P<url>.*)$', wrap(self.catch_all_view))) 298 299 return urlpatterns 300 301 @property 302 def urls(self): 303 return self.get_urls(), 'admin', self.name 304 305 def each_context(self, request): 306 """ 307 Return a dictionary of variables to put in the template context for 308 *every* page in the admin site. 309 310 For sites running on a subpath, use the SCRIPT_NAME value if site_url 311 hasn't been customized. 312 """ 313 script_name = request.META['SCRIPT_NAME'] 314 site_url = script_name if self.site_url == '/' and script_name else self.site_url 315 return { 316 'site_title': self.site_title, 317 'site_header': self.site_header, 318 'site_url': site_url, 319 'has_permission': self.has_permission(request), 320 'available_apps': self.get_app_list(request), 321 'is_popup': False, 322 'is_nav_sidebar_enabled': self.enable_nav_sidebar, 323 } 324 325 def password_change(self, request, extra_context=None): 326 """ 327 Handle the "change password" task -- both form display and validation. 328 """ 329 from django.contrib.admin.forms import AdminPasswordChangeForm 330 from django.contrib.auth.views import PasswordChangeView 331 url = reverse('admin:password_change_done', current_app=self.name) 332 defaults = { 333 'form_class': AdminPasswordChangeForm, 334 'success_url': url, 335 'extra_context': {**self.each_context(request), **(extra_context or {})}, 336 } 337 if self.password_change_template is not None: 338 defaults['template_name'] = self.password_change_template 339 request.current_app = self.name 340 return PasswordChangeView.as_view(**defaults)(request) 341 342 def password_change_done(self, request, extra_context=None): 343 """ 344 Display the "success" page after a password change. 345 """ 346 from django.contrib.auth.views import PasswordChangeDoneView 347 defaults = { 348 'extra_context': {**self.each_context(request), **(extra_context or {})}, 349 } 350 if self.password_change_done_template is not None: 351 defaults['template_name'] = self.password_change_done_template 352 request.current_app = self.name 353 return PasswordChangeDoneView.as_view(**defaults)(request) 354 355 def i18n_javascript(self, request, extra_context=None): 356 """ 357 Display the i18n JavaScript that the Django admin requires. 358 359 `extra_context` is unused but present for consistency with the other 360 admin views. 361 """ 362 return JavaScriptCatalog.as_view(packages=['django.contrib.admin'])(request) 363 364 def logout(self, request, extra_context=None): 365 """ 366 Log out the user for the given HttpRequest. 367 368 This should *not* assume the user is already logged in. 369 """ 370 from django.contrib.auth.views import LogoutView 371 defaults = { 372 'extra_context': { 373 **self.each_context(request), 374 # Since the user isn't logged out at this point, the value of 375 # has_permission must be overridden. 376 'has_permission': False, 377 **(extra_context or {}) 378 }, 379 } 380 if self.logout_template is not None: 381 defaults['template_name'] = self.logout_template 382 request.current_app = self.name 383 return LogoutView.as_view(**defaults)(request) 384 385 @method_decorator(never_cache) 386 def login(self, request, extra_context=None): 387 """ 388 Display the login form for the given HttpRequest. 389 """ 390 if request.method == 'GET' and self.has_permission(request): 391 # Already logged-in, redirect to admin index 392 index_path = reverse('admin:index', current_app=self.name) 393 return HttpResponseRedirect(index_path) 394 395 # Since this module gets imported in the application's root package, 396 # it cannot import models from other applications at the module level, 397 # and django.contrib.admin.forms eventually imports User. 398 from django.contrib.admin.forms import AdminAuthenticationForm 399 from django.contrib.auth.views import LoginView 400 context = { 401 **self.each_context(request), 402 'title': _('Log in'), 403 'app_path': request.get_full_path(), 404 'username': request.user.get_username(), 405 } 406 if (REDIRECT_FIELD_NAME not in request.GET and 407 REDIRECT_FIELD_NAME not in request.POST): 408 context[REDIRECT_FIELD_NAME] = reverse('admin:index', current_app=self.name) 409 context.update(extra_context or {}) 410 411 defaults = { 412 'extra_context': context, 413 'authentication_form': self.login_form or AdminAuthenticationForm, 414 'template_name': self.login_template or 'admin/login.html', 415 } 416 request.current_app = self.name 417 return LoginView.as_view(**defaults)(request) 418 419 def autocomplete_view(self, request): 420 return AutocompleteJsonView.as_view(admin_site=self)(request) 421 422 @no_append_slash 423 def catch_all_view(self, request, url): 424 if settings.APPEND_SLASH and not url.endswith('/'): 425 urlconf = getattr(request, 'urlconf', None) 426 try: 427 match = resolve('%s/' % request.path_info, urlconf) 428 except Resolver404: 429 pass 430 else: 431 if getattr(match.func, 'should_append_slash', True): 432 return HttpResponsePermanentRedirect('%s/' % request.path) 433 raise Http404 434 435 def _build_app_dict(self, request, label=None): 436 """ 437 Build the app dictionary. The optional `label` parameter filters models 438 of a specific app. 439 """ 440 app_dict = {} 441 442 if label: 443 models = { 444 m: m_a for m, m_a in self._registry.items() 445 if m._meta.app_label == label 446 } 447 else: 448 models = self._registry 449 450 for model, model_admin in models.items(): 451 app_label = model._meta.app_label 452 453 has_module_perms = model_admin.has_module_permission(request) 454 if not has_module_perms: 455 continue 456 457 perms = model_admin.get_model_perms(request) 458 459 # Check whether user has any perm for this module. 460 # If so, add the module to the model_list. 461 if True not in perms.values(): 462 continue 463 464 info = (app_label, model._meta.model_name) 465 model_dict = { 466 'model': model, 467 'name': capfirst(model._meta.verbose_name_plural), 468 'object_name': model._meta.object_name, 469 'perms': perms, 470 'admin_url': None, 471 'add_url': None, 472 } 473 if perms.get('change') or perms.get('view'): 474 model_dict['view_only'] = not perms.get('change') 475 try: 476 model_dict['admin_url'] = reverse('admin:%s_%s_changelist' % info, current_app=self.name) 477 except NoReverseMatch: 478 pass 479 if perms.get('add'): 480 try: 481 model_dict['add_url'] = reverse('admin:%s_%s_add' % info, current_app=self.name) 482 except NoReverseMatch: 483 pass 484 485 if app_label in app_dict: 486 app_dict[app_label]['models'].append(model_dict) 487 else: 488 app_dict[app_label] = { 489 'name': apps.get_app_config(app_label).verbose_name, 490 'app_label': app_label, 491 'app_url': reverse( 492 'admin:app_list', 493 kwargs={'app_label': app_label}, 494 current_app=self.name, 495 ), 496 'has_module_perms': has_module_perms, 497 'models': [model_dict], 498 } 499 500 if label: 501 return app_dict.get(label) 502 return app_dict 503 504 def get_app_list(self, request): 505 """ 506 Return a sorted list of all the installed apps that have been 507 registered in this site. 508 """ 509 app_dict = self._build_app_dict(request) 510 511 # Sort the apps alphabetically. 512 app_list = sorted(app_dict.values(), key=lambda x: x['name'].lower()) 513 514 # Sort the models alphabetically within each app. 515 for app in app_list: 516 app['models'].sort(key=lambda x: x['name']) 517 518 return app_list 519 520 def index(self, request, extra_context=None): 521 """ 522 Display the main admin index page, which lists all of the installed 523 apps that have been registered in this site. 524 """ 525 app_list = self.get_app_list(request) 526 527 context = { 528 **self.each_context(request), 529 'title': self.index_title, 530 'subtitle': None, 531 'app_list': app_list, 532 **(extra_context or {}), 533 } 534 535 request.current_app = self.name 536 537 return TemplateResponse(request, self.index_template or 'admin/index.html', context) 538 539 def app_index(self, request, app_label, extra_context=None): 540 app_dict = self._build_app_dict(request, app_label) 541 if not app_dict: 542 raise Http404('The requested admin page does not exist.') 543 # Sort the models alphabetically within each app. 544 app_dict['models'].sort(key=lambda x: x['name']) 545 context = { 546 **self.each_context(request), 547 'title': _('%(app)s administration') % {'app': app_dict['name']}, 548 'subtitle': None, 549 'app_list': [app_dict], 550 'app_label': app_label, 551 **(extra_context or {}), 552 } 553 554 request.current_app = self.name 555 556 return TemplateResponse(request, self.app_index_template or [ 557 'admin/%s/app_index.html' % app_label, 558 'admin/app_index.html' 559 ], context) 560 561 562 class DefaultAdminSite(LazyObject): 563 def _setup(self): 564 AdminSiteClass = import_string(apps.get_app_config('admin').default_site) 565 self._wrapped = AdminSiteClass() 566 567 def __repr__(self): 568 return repr(self._wrapped) 569 570 571 # This global object represents the default admin site, for the common case. 572 # You can provide your own AdminSite using the (Simple)AdminConfig.default_site 573 # attribute. You can also instantiate AdminSite in your own code to create a 574 # custom admin site. 575 site = DefaultAdminSite() 576 [end of django/contrib/admin/sites.py] </code> I need you to solve this issue by generating a single patch file that I can apply directly to this repository using git apply. Please respond with a single patch file in the following format. <patch> --- a/file.py +++ b/file.py @@ -1,27 +1,35 @@ def euclidean(a, b): - while b: - a, b = b, a % b - return a + if b == 0: + return a + return euclidean(b, a % b) def bresenham(x0, y0, x1, y1): points = [] dx = abs(x1 - x0) dy = abs(y1 - y0) - sx = 1 if x0 < x1 else -1 - sy = 1 if y0 < y1 else -1 - err = dx - dy + x, y = x0, y0 + sx = -1 if x0 > x1 else 1 + sy = -1 if y0 > y1 else 1 - while True: - points.append((x0, y0)) - if x0 == x1 and y0 == y1: - break - e2 = 2 * err - if e2 > -dy: + if dx > dy: + err = dx / 2.0 + while x != x1: + points.append((x, y)) err -= dy - x0 += sx - if e2 < dx: - err += dx - y0 += sy + if err < 0: + y += sy + err += dx + x += sx + else: + err = dy / 2.0 + while y != y1: + points.append((x, y)) + err -= dx + if err < 0: + x += sx + err += dy + y += sy + points.append((x, y)) return points </patch>''',
    #     "repo": "django/django",
    #     "base_commit": "475cffd1d64c690cdad16ede4d5e81985738ceb4",
    #     "problem_statement": "Wrong URL generated by get_admin_url for readonly field in custom Admin Site Description When a model containing a ForeignKey field is viewed (or edited) in a custom Admin Site, and that ForeignKey field is listed in readonly_fields, the url generated for the link is /admin/... instead of /custom-admin/.... This appears to be caused by the following line in django.contrib.admin.helpers get_admin_url: url = reverse(url_name, args=[quote(remote_obj.pk)]) Other parts of the admin use the current_app keyword parameter to identify the correct current name of the Admin Site. (See django.contrib.admin.options.ModelAdmin response_add as just one example) I have been able to correct this specific issue by replacing the above line with: url = reverse( url_name, args=[quote(remote_obj.pk)], current_app=self.model_admin.admin_site.name ) However, I don't know if there are any side effects and I have not yet run the full suite of tests on this. Mostly looking for feedback whether I'm on the right track.",
    #     "hints_text": '''Hey Ken, yes seems right. Good spot. Looks like this should have been part of b79088306513d5ed76d31ac40ab3c15f858946ea for #31181 (which was Django 3.2) ​here. However, I don't know if there are any side effects and I have not yet run the full suite of tests on this. Mostly looking for feedback whether I'm on the right track. I ran your suggestion against most of the usual suspects admin_* tests without issue so... Would you like to prepare a patch? Looks like setting up the test case is the most of it... Thanks! I'll be happy to try - but I'm not likely to be able to get to it before the weekend. (I don't know how "urgent" you consider it.) If it can sit that long, I'll see what I can do. (First "real patch" and all that - want to make sure I do it reasonably right.) Hey Ken. Super thanks! Since it's a bug in a new feature it's marked as release blocker and will be backported to Django 3.2. We'll target ​3.2.8, which is slated for the beginning of October. If it gets close to that and you've not had time we can pick it up. Reach out on the Forum if you'd like input at all. 🙂 Thanks! (And Welcome Aboard! ⛵️) Heyy folks, I wanted to assign the ticket to myself and fix the issue, instead it assigned the ownership to me. Apologies Changes ownership again. I found out that changes got accepted, sorry for the inconvenience caused. Hi Abhijith — just to confirm, according to the discussion Ken is currently working on this ticket, so let's give him a window to do that before re-assigning it. Thanks! (I think that's the conclusion you came to, but just double-checking so you don't both work on the same ticket at the same time.)'''
    # }]

    # envs = []
    # for doc in documents:
    #     env = SWEBenchEnvironment(SWEProblem.parse_obj(doc))
    #     envs.append(env)
   
    # Run the planner agent
    envs = None
    run_planner(default_llm_key="default_llm",
                premium_llm_key="premium_llm",
                llmORchains_list=llmORchains_list,
                test_environments=envs,
                manual_validation_to_capitalize=False,
                problem_prompts_subdir="SWE_Synthesis",  
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
                agcoding_num_parallel_inferences=1,
                unique_id=unique_id,
                functions_to_import=".*")  # Auto-test: 0"""

    # from primitives.swe_primititves.generate_patch import generate_patch

    # dataset = load_dataset(path="ahsanirfan961/swe-bech-lite-bm25-13k-take3", split='train')
    # dataset = dataset.select(range(1, 2))

    # for data in dataset:
    #     problem = SWEProblem.parse_obj(data)
        
    #     bot = SWEManager()
    #     env = SWEBenchEnvironment(problem)

    #     generate_patch(bot, problem, env)

    #     with open("primitives/swe_primititves/find_buggy_code.txt", "r") as f:
    #         no_runtime_error, exec_result = env.step(code=f.read())
        
    #     print("No runtime error:", no_runtime_error)
    #     print("Execution result:", exec_result)
        
    #     env.backup_state()

    #     print(env.get_state())

    #     structure = str(env.summarize_repo("env/SWEBench/repos/scikit-learn", max_files=500, max_output_characters=24000))

    #     print(len(structure))

    #     import tiktoken
    #     encoding = tiktoken.encoding_for_model("gpt-4")
    #     tokens = encoding.encode(structure)
    #     print("Tokens:", len(tokens))

    #     print(get_abs_current_dir())
    #     print(find_files.invoke({"file_name": "multiclass"}))
    #     print(ls.invoke({}))
    #     print(get_files_content.invoke({"paths": ["pyproject.toml", "astropy/logger.py"], "line_numbers": [10, 33]}))

    # with open('inspection/results.pkl', 'rb') as file:
    #     state = pickle.load(file)

    # Add to vector database with tags
    # tags = {"host": f"{socket.gethostname()}-{uuid.getnode()}", "step_id": 1}
    # HumanLLMMonitor._check_and_init_vector_db(embedding_function=embedding_function, reset_db_indices=reset_db_indices)
    # HumanLLMMonitor.check_init_class_db(force=True)
    # HumanLLMMonitor(llmORchains_list=llmORchains_list, agent_name='Validation_Agent').add_learnt_task(state, tags)
    # answer = HumanLLMMonitor(llmORchains_list=llmORchains_list, agent_name='Validation_Agent').get_learnt_tasks()
    # print(type(answer))
    # print(len(answer))
    # answer = list(answer)
    # for ans in answer:
    #     task: dict = json.loads(ans)
    #     print(task['main_function_name'])