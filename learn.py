import inspect
import pprint
import random
import string
import subprocess
import traceback
import types
import time
from config import *

#import openai
import json
from typing import Dict, Optional

from utils.llm_utils import UnifiedVectorDB, HumanLLMMonitor, save_prompt, _visual_input, \
    is_vscode_installed, smart_print, smart_input
from concurrent.futures import ThreadPoolExecutor, as_completed
import copy
from pydantic import BaseModel
import os
import uuid
import re
import shutil
import hashlib

from langchain_core.runnables import Runnable, RunnablePassthrough
#from langchain.schema.runnable import Runnable, RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
# from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain_core.messages.human import HumanMessage
from langchain_core.messages.ai import AIMessage
from langchain_core.messages.system import SystemMessage
from langchain_core.messages.function import FunctionMessage

from langchain_openai import ChatOpenAI  # from langchain.chat_models import ChatOpenAI
# from langchain.chat_models import ChatOpenAI

import json

from langchain.globals import set_llm_cache
from langchain_community.cache import SQLiteCache

set_llm_cache(SQLiteCache(database_path=".langchain_caching.db"))

#openai.api_key = os.environ['OPENAI_API_KEY']
#if 'OPENAI_BASE_URL' in os.environ: openai.base_url = os.environ['OPENAI_BASE_URL']

UnifiedVectorDB.db_type = "elasticsearch"  # "elasticsearch" "chroma"
UnifiedVectorDB.es_url = elastic_url_port
UnifiedVectorDB.es_user = elastic_user
UnifiedVectorDB.es_password = elastic_password
UnifiedVectorDB.OpenAI_embedding_function_name = "text-embedding-ada-002" # "nomic-ai/nomic-embed-text-v1"

embedding_function="intfloat/e5-base-v2" # UnifiedVectorDB.OpenAI_embedding_function_name # e.g. "text-embedding-ada-002" for OpenAI or "intfloat/e5-base-v2" or other huggingface models - WARINING: if you change it, set reset_db_indices to True
reset_db_indices=True # Set to True after changing embeddings

HumanLLMMonitor._check_and_init_vector_db(embedding_function=embedding_function, reset_db_indices=reset_db_indices) 


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
        os.symlink(os.path.abspath(self.data_dir), os.path.join(self.current_temp_dir, "data"), target_is_directory=True)

    def step(self, action_code, context={}):
        # memorize current directory, to allow to change to temp directory, then change back to memorized directory
        current_dir = os.getcwd()
        os.chdir(self.current_temp_dir)
        # Regular expression to check if the last line assigns to 'result'
        if not re.search(r'\bresult\s*=', action_code.strip().splitlines()[-1]):
            helper = "\nresult = locals().get('_', None)"
        else: helper = ""

        # execute action
        try:
            # capture stdout and stderr while executing code
            exec(action_code+helper, context)
            exec_result = context.get('result', [])
            no_runtime_error = True
        except Exception as e:
            exec_result = f"Failed to execute provided code. Error: {e} Traceback: {traceback.format_exc()}"
            no_runtime_error = False
        # set execution environment back to the memorized directory
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
            shutil.copytree(self.current_temp_dir, os.path.join(self.temp_root_dir, "backups", unique_id), ignore=shutil.ignore_patterns('data'))
        return unique_id

    def restore_state(self, unique_id):
        from_folder = os.path.join(self.temp_root_dir, "backups", unique_id)
        if not os.path.exists(from_folder): return "Restore state folder not found"
        if self.get_state(unique_id) != self.get_state():
            # copy all the content of the backup directory into the temp directory (excluding data directory)
            self.reset(backup_previous_temp_dir=False)
            # copy all the content of the backup directory into the temp directory which already contains the data directory
            shutil.copytree(from_folder, self.current_temp_dir, ignore=shutil.ignore_patterns('data'), dirs_exist_ok=True)
            return "State restored"
        else:
            return "State identical to backup folder"

    def restore_last_state(self):
        if self.last_unique_id_backup:
            self.restore_state(self.last_unique_id_backup)
            return "Last state restored"
        else:
            return "No last state to restore"
        
    def get_state(self, unique_id = None, extended_comparison = False):
        # return a dictionary containing the content of the temp directory
        state = {}
        state_folder = self.current_temp_dir if unique_id is None else os.path.join(self.temp_root_dir, "backups", unique_id)
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
        return {'score (best=1, worst=0)': score}

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


# Agent 1: Task Identification
class TaskIdentificationAgent():
    def __init__(self, default_llm_choice, envs: [Environment], premium_llm_choice=None, problem_prompts_subdir=None,
                 premium_llm_by_default=True, skip_rounds=0, llmORchains_list=None, optuna=None, model_choice=None, criteria=None):
        self.name = self.__class__.__name__
        self.problem_prompts_subdir = "" if problem_prompts_subdir is None else problem_prompts_subdir+"/"
        self.learnt_tasks: Dict[str, str] = {}
        self.failed_tasks: Dict[str, str] = {}
        self.criteria = criteria
        self.human_llm_identify_best_task = HumanLLMMonitor(default_llmORchain=default_llm_choice, premium_llmORchain=premium_llm_choice,
                                                            premium_llm_by_default=premium_llm_by_default,
                                                            llmORchains_list=llmORchains_list)#,output_schema="identify_best_task.schema.py")
        self.human_llm_identify_best_task.skip_rounds = skip_rounds
        self.envs = envs
        self.optuna_opti = optuna
        self.model_choice = model_choice

    def update_learnt_tasks(self, tasks: Dict[str, str]) -> None:
        self.learnt_tasks = tasks

    def update_failed_tasks(self, tasks: Dict[str, str]) -> None:
        self.failed_tasks = tasks

    def identify_best_task(self) -> str:
        if self.criteria == None or self.criteria == "None":
            learnt_tasks = format(json.dumps(self.learnt_tasks))
            failed_tasks = format(json.dumps(self.failed_tasks))
            envs_status = '\n'.join([env.get_state() for env in self.envs])
        else:
            match self.criteria:
                case "Learnt":
                    failed_tasks = format(json.dumps(self.failed_tasks))
                    envs_status = '\n'.join([env.get_state() for env in self.envs])
                    learnt_tasks = None
                case "Failed":
                    learnt_tasks = format(json.dumps(self.learnt_tasks))
                    envs_status = '\n'.join([env.get_state() for env in self.envs])
                    failed_tasks = None
                case "Env":
                    learnt_tasks = format(json.dumps(self.learnt_tasks))
                    failed_tasks = format(json.dumps(self.failed_tasks))
                    envs_status = None
                case "LearntFailed":
                    envs_status = '\n'.join([env.get_state() for env in self.envs])
                    learnt_tasks = None
                    failed_tasks = None
                case "LearntEnv":
                    failed_tasks = format(json.dumps(self.failed_tasks))
                    learnt_tasks = None
                    envs_status = None
                case "FailedEnv":
                    learnt_tasks = format(json.dumps(self.learnt_tasks))
                    failed_tasks = None
                    envs_status = None
                case _:
                    learnt_tasks = None
                    failed_tasks = None
                    envs_status = None

        user_message =  f"- Already developed tasks: {learnt_tasks if learnt_tasks and learnt_tasks!='{}' else 'None'}\n"+\
                        f"- Already failed tasks (too hard): {failed_tasks if failed_tasks and failed_tasks!='{}' else 'None'}\n"+\
                        f"- Current status of examples on which the task will be tested on: {envs_status}\n"

        task = self.human_llm_identify_best_task.CallHumanLLM(system_prompt_template=self.problem_prompts_subdir+"identify_best_task", user_message=user_message, return_message_content_only=False, optuna=self.optuna_opti, model_choice=self.model_choice, stream_output=True)
        return task


# Agent 2: Code Task
class CodingAgent():
    def __init__(self, default_llm_choice, envs: [Environment], premium_llm_choice=None, problem_prompts_subdir=None, db_collection_success="successful_tasks", db_collection_failed="failed_tasks", skip_rounds=0, llmORchains_list=None, optuna=None, model_choice=None):
        #super().__init__(llm)
        self.name = self.__class__.__name__
        self.problem_prompts_subdir = "" if problem_prompts_subdir is None else problem_prompts_subdir+"/"

        self.human_llm_code_task = HumanLLMMonitor(default_llmORchain=default_llm_choice, premium_llmORchain=premium_llm_choice, premium_llm_by_default=True, num_parallel_inferences=4, llmORchains_list=llmORchains_list)#, output_schema="code_task.schema.py")
        self.human_llm_code_task.skip_rounds = skip_rounds
        self.human_llm_code_task.add_inference_check("Code Parsing", self.parse_ai_generated_code)
        self.human_llm_code_task.add_inference_check("Run Tests", self.run_tests_on_code)
        self.envs = envs
        self.optuna_opti = optuna
        self.model_choice = model_choice
        self.db_successful_tasks = UnifiedVectorDB( collection_name=db_collection_success, embedding_function=HumanLLMMonitor.common_vectordb_embedding_function, persist_directory=HumanLLMMonitor.common_vectordb_persist_directory+db_collection_success, reset_db_indices=reset_db_indices)
        self.db_failed_tasks = UnifiedVectorDB( collection_name=db_collection_failed, embedding_function=HumanLLMMonitor.common_vectordb_embedding_function, persist_directory=HumanLLMMonitor.common_vectordb_persist_directory+db_collection_failed, reset_db_indices=reset_db_indices)
        self.processed_codes = set()

    def parse_ai_generated_code(self, message, language="py", retry=3, required_bot_arg=None, task_definition=None, automatic_tests=True):
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
                if language == "py": # Python case
                    if code is None:
                        # Match Python code blocks
                        code_pattern = re.compile(r"```python(.*?)```", re.DOTALL)
                        code = "\n".join(code_pattern.findall(message))
                    
                    parsed = ast.parse(code)
                    functions = []
                    imports = []
                    
                    if len(code) == 0 or len(list(parsed.body)) == 0:
                        return False, f"Error parsing action response (No Code found): {parsed.body}"
                    
                    main_function = None
                    runnable_code = ""
                    for node in parsed.body:
                        if isinstance(node, ast.FunctionDef):
                            node_type = "FunctionDef"
                            main_function = {
                                    "name": node.name,
                                    "type": node_type,
                                    "body": ast.get_source_segment(code, node),
                                    "params": [arg.arg for arg in node.args.args],
                                }
                            functions.append(main_function)
                        elif isinstance(node, ast.Expr) or isinstance(node, ast.Expression) or isinstance(node, ast.Assign):
                            node_type = "Expression"
                            runnable_code += "\n" + ast.get_source_segment(code, node)
                        elif isinstance(node, ast.ImportFrom) or isinstance(node, ast.Import):
                            node_type = "ImportFrom"
                            imports.append(ast.get_source_segment(code, node))
                            smart_print("ImportFrom node: IMPORT SHOULD BE DONE INSIDE FUNCTIONS !!!", self.name, "process_ai_generated_code SystemMessage")
                        else:
                            smart_print(f"Unsupported node type: {type(node)} - content:  {ast.get_source_segment(code, node)}", self.name, "process_ai_generated_code SystemMessage")
                            #raise ValueError(f"Unsupported node type: {type(node)} - content:  {ast.get_source_segment(code, node)}")  # TODO: check if await is needed
                    
                    assert main_function is not None, "No main function found."
                    if required_bot_arg:
                        assert required_bot_arg in main_function["params"], f"Main function {main_function['name']} must take an argument named '{required_bot_arg}'"
                    
                    program_code = "\n".join(imports) + "\n"
                    program_code += "\n\n".join(function["body"] for function in functions)

                    if automatic_tests:
                        #[test for doc_id, test in parsed_code["tests"] if doc_id == env.id]
                        tests = [(env.id, main_function["name"]+"(bot)") for env in self.envs]
                    else:
                        tests = []
                        tests_pattern = re.compile(r'\n#\s+[Dd]ocument #([a-z0-9-]+)\s+usage test[^\n]*\n([^\n]+)|{.*?"DocumentID":\s*"#(.*?)",\s*"FunctionCall":\s*"([^"]+)".*?}')
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
        return False, self.parsed_code

    def run_tests_on_code(self, message, parsed_code=None, skip_already_processed=False):
        primitives = self.get_primitives()
        parsed_code = getattr(self, 'parsed_code', None) if parsed_code is None else parsed_code
        current_skip_rounds = self.human_llm_code_task.skip_rounds # save the initial value to align it for code validation

        if isinstance(parsed_code, dict) and parsed_code["program_code"] not in self.processed_codes:
            self.processed_codes.add(parsed_code["program_code"])
        elif skip_already_processed or not isinstance(parsed_code, dict): # This logic speedup because the same code should have the same score BUT only on the same problem & state
            return None

        #smart_print(f"************ Code parsed result ************\n{parsed_code}\n************************".replace("\\n", "\n"), self.name, "run_tests_on_code RESULT")
    
        # Set initial state before running tests or runnable code
        no_runtime_errors, exec_results = [], []

        # insert content of config.py into the code to ensure that the OPENAI_API_KEY is set
        with open("config.py", "r") as f: common_code = f.read() + "\n"
        # Common code part to be executed in all cases
        common_code += "\n".join(primitives) + "\n"

        # Run the code & tests in each environment
        for env in self.envs:
            env.backup_state()
            # Determine tests to run or set default runnable code
            matching_tests = [test for doc_id, test in parsed_code["tests"] if doc_id == env.id] if parsed_code["tests"] else [parsed_code['runnable_code']]
            if not matching_tests:
                no_runtime_error, exec_result = False, f"Error: no test found for given id {env.id}" if parsed_code["tests"] else f"Error: no runnable code found nor tests"
            else:
                # Concatenate common code with program and tests or runnable code
                code_to_run = common_code + parsed_code["program_code"] + "\n" + "\n".join(matching_tests)
                no_runtime_error, exec_result = env.step(code_to_run)
                while not no_runtime_error and current_skip_rounds <= 0:
                    smart_print("\033[31mCODE ERROR\033[0m: "+exec_result, self.name, "code_task_and_run_test SystemMessage")
                    if self.optuna_opti:
                        decision = "n"
                    else:
                        decision = smart_input("Do you want to edit the code to fix the error (you will also be requested first) ? (yes/no) or try autofix by LLM (a): ", self.name).strip().lower()
                    if decision in ("no", "n", ""):
                        break
                    elif decision == "a":
                        # do not use HumanLLMMonitor because no template is available for this specific case
                        smart_print("\033[31mTRYING TO AUTOFIX ERROR\033[0m", self.name)
                        message_content = f"ERROR MESSAGE:[[{exec_result}]]\nCODE:[[{parsed_code['program_code']}]]"
                        edited_code_returned = self.human_llm_code_task.premium_llm.invoke([SystemMessage(content=self.human_llm_code_task.load_prompt(agent_name=self.name, prompt="code_fixer")), HumanMessage(content=message_content)])
                        edited_code = str(edited_code_returned) if isinstance(edited_code_returned, dict) else edited_code_returned.content
                    else:
                        edited_code = _visual_input(parsed_code["program_code"], filetype="py")
                    code_to_run = common_code + edited_code + "\n" + "\n".join(matching_tests)
                    smart_print("\033[31mTESTING NEW CODE\033[0m", self.name, "code_task_and_run_test SystemMessage")
                    no_runtime_error, exec_result = env.step(code_to_run)
                    # Update parsed_code if re-run is successful
                    # if no_runtime_error:
                    parsed_code["program_code"] = edited_code
            # Append the results for each environment
            no_runtime_errors.append(no_runtime_error)
            exec_results.append(exec_result)

        # Return combined results
        result = (parsed_code, all(no_runtime_errors), exec_results, [env.get_score() for env in self.envs], [env.get_state(extended=True) for env in self.envs])

        for env in self.envs:
            env.restore_last_state()

        return result


    def get_primitives(self):
        primitives = []
        # Add the pipelines folder for the primitives
        if self.problem_prompts_subdir=="Anomalies/":
            path_folder = "pipelines/pipelines"
        else:
            path_folder = "primitives"
        folder_path = os.path.join(os.path.dirname(__file__), path_folder)
        # Utiliser os.listdir pour ne pas parcourir les sous-répertoires
        for file in os.listdir(folder_path):
            if file.endswith(".py"):
                smart_print(f"File load: {file}", "CONFIG", "Files loaded")
                file_path = os.path.join(folder_path, file)
                with open(file_path, "r") as f:
                    primitives.append(f.read())
        return primitives

    def code_task_and_run_test(self, refined_task: str, previous_errors=None, previous_scores=None, previous_codes=None) -> str:
        primitives = self.get_primitives()

        user_message=f"TASK DEFINITION: [[[{refined_task}]]]"
        nl, dnl = "\n", "\n\n"
        max_db_results = 10
        if self.envs and len(self.envs) > 0:
            user_message+=f"{nl}CURRENT STATE OF DOCUMENTS TO PERFORM/TEST TASK: [[[{nl}{nl.join([env.get_state(extended=True) for env in self.envs])}{nl}]]]"
        if len(primitives) > 0:
            user_message+=f"{dnl}RE-USABLE CODE PRIMITIVES: [[[{nl}{nl.join(primitives)}{nl}]]]"
        successful_tasks, failed_tasks = [result.page_content for result in self.db_successful_tasks.query(query_text="*", k=max_db_results)], [result.page_content for result in self.db_failed_tasks.query(query_text="*", k=max_db_results)]
        if successful_tasks and len(successful_tasks) > 0:
            user_message+=f"{dnl}PREVIOUSLY SUCCESSFUL TASKS: [[[{nl}{nl.join(successful_tasks[:5])}{nl}]]]"
        if failed_tasks and len(failed_tasks) > 0:
            user_message+=f"{dnl}PREVIOUSLY FAILED TASKS: [[[{nl}{nl.join(failed_tasks[:3])}{nl}]]]"
        if previous_errors and len(previous_errors) > 0:
            user_message+=f"{dnl}PREVIOUS ATTEMPTS TO CODE THE TASK: [[[{nl}"
            for previous_error, previous_score, previous_code in zip(previous_errors, previous_scores, previous_codes):
                user_message+=f"{nl}<<ATTEMPT FEEDBACK: {previous_error.content}" + (f"{dnl}SCORE: {previous_score}" if previous_score else "") + f"{dnl}CODE: {previous_code}{nl}>>"
            user_message+=f"{nl}]]]"

        current_skip_rounds = self.human_llm_code_task.skip_rounds # save the initial value to align it for code validation
        codes = self.human_llm_code_task.CallHumanLLM(system_prompt_template=self.problem_prompts_subdir+"code_task", user_message=user_message, return_message_content_only=False, stream_output=False, optuna=self.optuna_opti, model_choice=self.model_choice)
        results = []

        for index, code in enumerate(codes):
            # Get the proper check_results corresponding to the output_id (which is the index)
            check_results = self.human_llm_code_task.last_inference_check_results[index]
            
            code_parsing_success, parsed_code = check_results.get("Code Parsing", (False, None))
            if code_parsing_success and isinstance(parsed_code, dict):
                test_results = check_results.get("Run Tests", None)
                if test_results:
                    results.append(test_results)
            
        # test if more than one code is returned, so ask to select or 
        if len(results) > 1:
            # display the list of results with success, exception and code
            results_list = ""
            top_results, top_indice = 0, 1
            for id, result in enumerate(results):
                if result[1]:
                    results_list += f"{id}. \033[32mSUCCESS\033[0m / SCORE: {result[3]} / CODE: {result[0]['program_code'][:100]}\n"
                    temp = sum(result[3][i][j] for i in range(len(result[3])) for j in result[3][i]) / len(result[3])
                    if temp > top_results:
                        top_results = temp
                        top_indice = id
                else:
                    results_list += f"{id}. \033[31mFAILED\033[0m / SCORE: {result[3]} / EXCEPTION: {result[2][0][:100]} / CODE: {result[0]['program_code'][:100]}\n"

            # ask the user to select the code to keep
            if current_skip_rounds <= 0:
                if self.optuna_opti:
                    selected_code = f"{top_indice}"
                else: selected_code = smart_input(f"{results_list}CODE SELECTION Please select the code to keep (separated by comma, none/n for none of these, or just hit enter to keep ALL): ",self.name,"Scores").strip().replace(" ","").lower().split(",")
            else:
                selected_code = [""]  # keep all if skip_rounds is not 0
            id = 0
            if selected_code in ["none", "n"]:
                results = []
            else:
                # keep only the selected code
                results = [result for id, result in enumerate(results) if selected_code and (str(id) in selected_code or selected_code == [""])]

        return results


# Agent 3: Code Validation
class ValidationAgent():
    def __init__(self, default_llm_choice, envs: [Environment], premium_llm_choice=None, skip_rounds=0, llmORchains_list=None, optuna=None, model_choice=None):
        #super().__init__(llm)
        self.name = self.__class__.__name__
        self.human_llm_validate_code = HumanLLMMonitor(default_llmORchain=default_llm_choice, premium_llmORchain=premium_llm_choice, premium_llm_by_default=False, llmORchains_list=llmORchains_list)
        self.human_llm_validate_code.skip_rounds = skip_rounds
        self.envs = envs
        self.optuna_opti = optuna
        self.model_choice = model_choice


    def validate_code(self, code: str, no_runtime_error:bool, exec_result:str, task:str=None, human_evaluation_required=False, scores=None, env_states=None) -> str:
        runtime_errors = f'\033[32mno runtime errors at execution - code returned:\n{exec_result}\n\033[0m' if no_runtime_error else f'\033[31mruntime errors at execution - error:{exec_result}\033[0m'
        if human_evaluation_required:
            human_evaluation = smart_input(f"\n\n*******************\n{code}\n************\nCODE ABOVE EXECUTED with result: {runtime_errors}\n****\System may not efficiently evaluate what is produced by the code, please add your evaluation of the result (or hit enter): ")
        else:
            human_evaluation = ""
        runtime_errors = 'no runtime errors at execution' if no_runtime_error else 'runtime errors at execution' # just to avoid to break colors inside HumanLLMMonitor
        envs_status = '\n'.join(env_states)

        user_message = f"Task: {task}\n\n"+\
            f"Code: {code}\n\n"+\
            f"Code execution returned: {runtime_errors}\n\n"+\
            f"Execution result returned by exec command of code provided: {exec_result}\n\n"+\
            (f"Human evaluation of the result: {human_evaluation}\n\n" if human_evaluation != "" else "") +\
            f"Performance scores: {scores}\n" +\
            f"New environment status of examples on which the task has been tested on: {envs_status}\n"

        code_validation = self.human_llm_validate_code.CallHumanLLM(system_prompt_template="validate_code", user_message=user_message, return_message_content_only=False, optuna=self.optuna_opti, model_choice=self.model_choice) 
        return code_validation


# Agent 4: Code Capitalization
class CapitalizationAgent:
    def __init__(self, default_llm_choice, premium_llm_choice=None, db_collection_success="successful_tasks", db_collection_failed="failed_tasks", db_embedding_function=None, db_perist_directory=None, skip_rounds=0, llmORchains_list=None, optuna=None, model_choice=None, problem_prompts_subdir=None):
        self.name = self.__class__.__name__
        self.tasks_repository: Dict[str, str] = {}
        self.failed_tasks_repository: Dict[str, str] = {}
        self.problem_prompts_subdir = "" if problem_prompts_subdir is None else problem_prompts_subdir+"/"
        self.human_llm_generate_function_description = HumanLLMMonitor(default_llmORchain=default_llm_choice, premium_llmORchain=premium_llm_choice, premium_llm_by_default=False, llmORchains_list=llmORchains_list)
        self.human_llm_generate_function_description.skip_rounds = skip_rounds
        self.optuna_opti = optuna
        self.model_choice = model_choice
        self.db_successful_tasks = UnifiedVectorDB(
            collection_name=db_collection_success,
            embedding_function=db_embedding_function if db_embedding_function else HumanLLMMonitor.common_vectordb_embedding_function,
            persist_directory=db_perist_directory if db_perist_directory else HumanLLMMonitor.common_vectordb_persist_directory+db_collection_success)
        self.db_failed_tasks = UnifiedVectorDB(
            collection_name=db_collection_failed,
            embedding_function=db_embedding_function if db_embedding_function else HumanLLMMonitor.common_vectordb_embedding_function,
            persist_directory=db_perist_directory if db_perist_directory else HumanLLMMonitor.common_vectordb_persist_directory+db_collection_failed)

    def capitalize_successful_tasks(self, task_description: str, parsed_code: str) -> None:
        import socket, uuid, datetime

        if self.problem_prompts_subdir=="Anomalies/":
            function_name = parsed_code["class_name"]
            pipeline_file_path = os.path.join("pipelines/pipelines", function_name+".py")
            tool_description = str(self.generate_tool_description(parsed_code["class_name"], parsed_code["program_code"]))
            self.tasks_repository[parsed_code["class_name"]] = [tool_description, parsed_code["program_code"]]
            # print last added task
            smart_print(f"************ Last added task ************\n{parsed_code['class_name']}\n************************".replace("\\n", "\n"), self.name, "capitalize_successful_tasks SUCCESS")
        else:
            function_name = parsed_code.get("main_function_name", parsed_code.get("main_function", {}).get("name", "unknown"))
            # save function program_code in a file under the functions directory and add to the function signature the generated dosctring
            function_file_path = os.path.join("functions", function_name+".py")
            tool_description = str(self.generate_tool_description(function_name, parsed_code["program_code"]))
            self.tasks_repository[function_name] = [tool_description, parsed_code["program_code"]]
            # print last added task
            smart_print(f"************ Last added task ************\n{function_name}\n************************".replace("\\n", "\n"), self.name, "capitalize_successful_tasks SUCCESS")

        
        if self.problem_prompts_subdir=="Anomalies/":
           if os.path.exists(pipeline_file_path):
                smart_print(f"Pipeline file {pipeline_file_path} already exists, please provide a new name for the pipeline.", self.name, "capitalize_successful_tasks WARNING")
                if self.optuna_opti:
                    i = random.randint(0, 1000)
                    pipeline_file_path = os.path.join("pipelines/pipelines", self.name + f"_{i}.py")
                else:
                    pipeline_file_path = os.path.join("pipelines/pipelines", smart_input("New pipeline name: ")+".py")
            
        # check if the function file already exists, if yes, ask the user a new name
        else:    
            if os.path.exists(function_file_path):
                smart_print(f"Function file {function_file_path} already exists, please provide a new name for the function.", self.name, "capitalize_successful_tasks WARNING")
                if self.optuna_opti:
                    i = random.randint(0, 1000)
                    function_file_path = os.path.join("functions", self.name + f"_{i}.py")
                else:
                    function_file_path = os.path.join("functions", smart_input("New function name: ")+".py")
        
        if self.problem_prompts_subdir=="Anomalies/":
            new_path = pipeline_file_path
        else:
            new_path = function_file_path


        with open(new_path, "w") as function_file:
             # use regex to extract the docstring from tool_description
            docstring_pattern = re.compile(r'(""".*?""")', re.DOTALL)
            docstring_matches = docstring_pattern.findall(tool_description)
            docstring = docstring_matches[0] if docstring_matches else f'"""{tool_description}"""'
             # use regex to add docstring to the function parsed_code["main_function_name"] after the def line in parsed_code["program_code"]
            parsed_code["program_code"] = re.sub(r"(def "+function_name+"\(.*?\):)", r'\1\n    '+docstring, parsed_code["program_code"], count=1)
            function_file.write(parsed_code["program_code"])

        if self.optuna_opti == None:
            if is_vscode_installed():
                smart_print("Please modify the file opened in vscode if necessary, and save it (Ctrl + W) when you are ok to continue", self.name, "capitalize_successful_tasks INSTRUCTIONS")
                subprocess.run(["code", "--wait", new_path])

        if self.problem_prompts_subdir=="Anomalies/":
            serialized_entry = json.dumps({
            "time": datetime.datetime.now().isoformat(),
            "class_name": function_name,
            "program_code": parsed_code["program_code"],
            "tool_description": tool_description,
            "task_description": task_description,
        }, default=lambda o: o.__dict__ if hasattr(o, '__dict__') else str(o))
        else: 
            serialized_entry = json.dumps({
                "time": datetime.datetime.now().isoformat(),
                "main_function_name": function_name,
                "program_code": parsed_code["program_code"],
                "tool_description": tool_description,
                "task_description": task_description,
            }, default=lambda o: o.__dict__ if hasattr(o, '__dict__') else str(o))

        # Log entry into the common vector database with tags
        tags = {    "host": socket.gethostname()+"-"+str(uuid.getnode()),
                    "step_id": HumanLLMMonitor.step_id,}

        self.db_successful_tasks.add_texts(texts=[serialized_entry], metadatas=[tags])

    def capitalize_failed_tasks(self, task_description: str, parsed_code: str) -> None:
        import socket, uuid, datetime

        if self.problem_prompts_subdir=="Anomalies/":
            name_file = parsed_code["class_name"]
            main_class_name = _visual_input(name_file if parsed_code is not None and "class_name" in parsed_code else "replace this text with a descriptive name of the class")
            task_description_refined = _visual_input(task_description)
            self.failed_tasks_repository[main_class_name] = task_description_refined
            # print last added task
            smart_print(f"************ Last added failed task ************\n{main_class_name}\n************************".replace("\\n", "\n"), self.name, "capitalize_failed_tasks CAPITALIZE FAIL")

            serialized_entry = json.dumps({
                "time": datetime.datetime.now().isoformat(),
                "class_name": main_class_name,
                "program_code": parsed_code["program_code"] if parsed_code is not None and "program_code" in parsed_code else None,
                "task_description": task_description,
                "task_description_refined": task_description_refined,
            }, default=lambda o: o.__dict__ if hasattr(o, '__dict__') else str(o))

        else:
            if self.optuna_opti:
                main_function_name = parsed_code["main_function_name"] if parsed_code is not None and "main_function_name" in parsed_code else "replace this text with a descriptive name of the function"
                task_description_refined = task_description
            else :
                main_function_name = _visual_input(parsed_code["main_function_name"] if parsed_code is not None and "main_function_name" in parsed_code else "replace this text with a descriptive name of the function")
                task_description_refined = _visual_input(task_description)
            self.failed_tasks_repository[main_function_name] = task_description_refined
            self.failed_tasks_repository[main_function_name] = task_description_refined
            # print last added task
            smart_print(
                f"************ Last added failed task ************\n{main_function_name}\n************************".replace(
                    "\\n", "\n"), self.name, "capitalize_failed_tasks CAPITALIZE FAIL")

            serialized_entry = json.dumps({
                "time": datetime.datetime.now().isoformat(),
                "main_function_name": main_function_name,
                "program_code": parsed_code[
                    "program_code"] if parsed_code is not None and "program_code" in parsed_code else None,
                "task_description": task_description,
                "task_description_refined": task_description_refined,
            }, default=lambda o: o.__dict__ if hasattr(o, '__dict__') else str(o))
        
        
        # Log entry into the common vector database with tags
        tags = {    "host": HumanLLMMonitor.get_host_id(),
                    "step_id": HumanLLMMonitor.step_id,}

        self.db_failed_tasks.add_texts(texts=[serialized_entry], metadatas=[tags])

    def generate_tool_description(self, program_name, program_code):
        user_message = f"MAIN FUNCTION: `{program_name}`\n\nFULL CODE:\n{program_code}"
        tool_description = self.human_llm_generate_function_description.CallHumanLLM(system_prompt_template="generate_function_description", user_message=user_message, return_message_content_only=True, optuna=self.optuna_opti, model_choice=self.model_choice) 
        return tool_description
    
    def retrieve_saved_tasks_in_db(self, query="", max_db_results=20, include_code=None, selected_successful_functions=None, selected_failed_functions=None):
        smart_print(f"************ Retrieving successful tasks from database - LIST:", self.name, "retrieve_saved_tasks_in_db DATABASE ACCESS")
        # Retrieve tasks from the common vector database
        results_success_db = self.db_successful_tasks.query(query_text="*", k=max_db_results)
        # First step: display all the retrieved functions with time and host
        id = 0
        for result in results_success_db:
            id += 1
            # Extract the page_content field from the Document object
            page_content = result.page_content
            # Deserialize the JSON from the page_content string
            task_data = json.loads(page_content)
            if self.problem_prompts_subdir=="Anomalies/":
                if "class_name" in task_data:
                    smart_print(f"{id}: Pipeline name:{task_data['class_name']} time:{task_data['time']} host:{result.metadata['host']}", self.name, "retrieve_saved_tasks_in_db DATABASE ACCESS")
            else:
                if 'main-function_name' in task_data:
                    smart_print(f"{id}: function name:{task_data['main_function_name']} time:{task_data['time']} host:{result.metadata['host']}", self.name, "retrieve_saved_tasks_in_db DATABASE ACCESS")
            
            if include_code is None:
                include_code = smart_input(f"CONFIG When adding the functions description in successful tasks, do you want to also include the code (it may overflow the maximum prompt length but can also guide generation) ? (yes/no): ").strip().lower() in ["yes", "y"]
        # Second step: ask the user to select the functions to load
        if selected_successful_functions is None:
            selected_successful_functions = smart_input(f"CONFIG Please select the successful functions to load (separated by comma, or 'all' to load all, or just hit enter for none): ").strip().replace(" ","").lower().split(",")
        id = 0
        # load into self.tasks_repository
        for result in results_success_db:
            id += 1
            if selected_successful_functions and (str(id) not in selected_successful_functions) and (selected_successful_functions != ["all"]):
                continue
            # Extract the page_content field from the Document object
            page_content = result.page_content
            # Deserialize the JSON from the page_content string
            task_data = json.loads(page_content)
            if self.problem_prompts_subdir=="Anomalies/":
                if "class_name" in task_data:
                    if task_data["class_name"] in self.tasks_repository:
                        smart_print(f"> pipeline/task {task_data['class_name']} already loaded. When there are duplicates select your prefered. Skipping...", self.name, "retrieve_saved_tasks_in_db DATABASE ACCESS")
                        continue
                else:
                    task_data['class_name']=""
                    if 'task_description-refined' in task_data:
                        self.failed_tasks_repository[task_data["class_name"]] = task_data["task_description_refined"]
                        smart_print(f"> failed pipeline/task {task_data['class_name']} from host {result.metadata['host']} generated at {task_data['time']} loaded.", self.name, "retrieve_saved_tasks_in_db DATABASE ACCESS")

            else:
                if "main_function_name" in task_data:
                    if task_data["main_function_name"] in self.tasks_repository:
                        smart_print(f"> pipeline/task {task_data['main_function_name']} already loaded. When there are duplicates select your prefered. Skipping...", self.name, "retrieve_saved_tasks_in_db DATABASE ACCESS")
                        continue
                else:
                    task_data['main_function_name']=""
                    if 'task_description-refined' in task_data:
                        self.failed_tasks_repository[task_data["main_function_name"]] = task_data["task_description_refined"]
                        smart_print(f"> failed pipeline/task {task_data['main_function_name']} from host {result.metadata['host']} generated at {task_data['time']} loaded.", self.name, "retrieve_saved_tasks_in_db DATABASE ACCESS")

                self.tasks_repository[task_data["main_function_name"]] = [task_data["tool_description"]] if not include_code else [task_data["tool_description"], task_data["program_code"]]
                smart_print(f"> function/task {task_data['main_function_name']} from host {result.metadata['host']} generated at {task_data['time']} loaded.", self.name, "retrieve_saved_tasks_in_db DATABASE ACCESS")

        smart_print(f"************ Retrieving failed tasks from database - LIST:", self.name, "retrieve_saved_tasks_in_db DATABASE ACCESS")
        # Retrieve failed tasks from the common vector database
        results_failed_db = self.db_failed_tasks.query(query_text="*", k=max_db_results)
        # First step: display all the retrieved functions with time and host
        id = 0
        for result in results_failed_db:
            id += 1
            # Extract the page_content field from the Document object
            page_content = result.page_content
            # Deserialize the JSON from the page_content string
            task_data = json.loads(page_content)
            if self.problem_prompts_subdir=="Anomalies/":
                if 'class_name' in task_data:
                    smart_print(f"{id}: failed Pipeline name:{task_data['class_name']} time:{task_data['time']} host:{result.metadata['host']}", self.name, "retrieve_saved_tasks_in_db DATABASE ACCESS")
            else:
                if 'main_function-name'in task_data:
                    smart_print(f"{id}: failed function name:{task_data['main_function_name']} time:{task_data['time']} host:{result.metadata['host']}", self.name, "retrieve_saved_tasks_in_db DATABASE ACCESS")
            
            # Second step: ask the user to select the functions to load
        if selected_failed_functions is None:
            selected_failed_functions = smart_input(f"CONFIG Please select the failed functions to load (separated by comma, or 'all' to load all, or just hit enter for none): ").strip().replace(" ","").lower().split(",")
        id = 0
        # load into self.tasks_repository
        for result in results_failed_db:
            id += 1
            if selected_failed_functions and (str(id) not in selected_failed_functions) and (selected_failed_functions != ["all"]):
                continue
            # Extract the page_content field from the Document object
            page_content = result.page_content
            # Deserialize the JSON from the page_content string
            task_data = json.loads(page_content)
            if self.problem_prompts_subdir=="Anomalies/":
                if 'class_name' in task_data:
                    if task_data["class_name"] in self.failed_tasks_repository:
                        smart_print(f"> failed pipeline/task {task_data['class_name']} already loaded. When there are duplicates select your prefered. Skipping...", self.name, "retrieve_saved_tasks_in_db DATABASE ACCESS")
                        continue
                    else:
                        task_data['class_name']=""
                        if 'task_description-refined' in task_data:
                            self.failed_tasks_repository[task_data["class_name"]] = task_data["task_description_refined"]
                            smart_print(f"> failed pipeline/task {task_data['class_name']} from host {result.metadata['host']} generated at {task_data['time']} loaded.", self.name, "retrieve_saved_tasks_in_db DATABASE ACCESS")

            else:
                if task_data["main_function_name"] in self.failed_tasks_repository:
                    smart_print(f"> failed function/task {task_data['main_function_name']} already loaded. When there are duplicates select your prefered. Skipping...", self.name, "retrieve_saved_tasks_in_db DATABASE ACCESS")
                    continue
                self.failed_tasks_repository[task_data["main_function_name"]] = task_data["task_description_refined"]
                smart_print(f"> failed function/task {task_data['main_function_name']} from host {result.metadata['host']} generated at {task_data['time']} loaded.", self.name, "retrieve_saved_tasks_in_db DATABASE ACCESS")


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
                              criteria=None, max_execution_time=900):
    time_end = time.time() + max_execution_time
    scores = None

    if HumanLLMMonitor.use_websocket:
        if HumanLLMMonitor.websocket_server is None:
            HumanLLMMonitor.initialize_websocket_server()

    if problem_prompts_subdir is None:
        # menu to choose the problem prompts subdirectory
        # get the list of subdirectories in the problem prompts directory
        problem_prompts_subdirs = [name for name in os.listdir("prompts") if os.path.isdir(os.path.join("prompts", name))]
        # get first element of problem_prompts_subdirs if not empty, else set it to empty string
        default_subdir = problem_prompts_subdirs[0] if problem_prompts_subdirs else ""
        choice = smart_input("Enter a capital letter for subdirectory (leave empty for default): "+"; ".join(f"\n[{i}] {subdir}" for i, subdir in zip(string.ascii_uppercase,problem_prompts_subdirs))+" ?", "CONFIG")
        # if choise is empty or not a capital letter or not in the range of the list of subdirectories, set it to A
        problem_prompts_subdir = problem_prompts_subdirs[ord(choice)-65] if choice and choice.isupper() and ord(choice)-65 in range(len(problem_prompts_subdirs)) else default_subdir

    if test_environments is None:
        env_type = "default"
        manager = EnvironmentManager(env_type)
        test_environments = [manager.get_environment()]

    agent_taskreco = TaskIdentificationAgent(default_llm_key, test_environments, premium_llm_choice=premium_llm_key, problem_prompts_subdir=problem_prompts_subdir, premium_llm_by_default=agtask_premium_llm_by_default, skip_rounds=agtask_skip_rounds, llmORchains_list=llmORchains_list, optuna=optuna_opti, model_choice=(model_choice['taskreco' if 'taskreco' in model_choice else 'coach'] if type(model_choice) == dict else model_choice), criteria=criteria)
    agent_coding = CodingAgent(default_llm_key, test_environments, premium_llm_choice=premium_llm_key, problem_prompts_subdir=problem_prompts_subdir, skip_rounds=agcoding_skip_rounds, llmORchains_list=llmORchains_list, optuna=optuna_opti, model_choice=(model_choice['coding' if 'coding' in model_choice else 'coder'] if type(model_choice) == dict else model_choice))
    agent_validation = ValidationAgent(default_llm_key, test_environments, premium_llm_choice=premium_llm_key, skip_rounds=agvalidation_skip_rounds, llmORchains_list=llmORchains_list, optuna=optuna_opti, model_choice=(model_choice['validation' if 'validation' in model_choice else 'critic'] if type(model_choice) == dict else model_choice))
    agent_capitalize = CapitalizationAgent(default_llm_key, premium_llm_choice=premium_llm_key, skip_rounds=agcapitalize_skip_rounds,problem_prompts_subdir=problem_prompts_subdir, llmORchains_list=llmORchains_list, optuna=optuna_opti, model_choice=(model_choice['capitalize' if 'capitalize' in model_choice else 'capitalizer'] if type(model_choice) == dict else model_choice))

    agent_capitalize.retrieve_saved_tasks_in_db(include_code=include_code, selected_successful_functions=selected_successful_functions, selected_failed_functions=selected_failed_functions)
    agent_taskreco.update_learnt_tasks(agent_capitalize.tasks_repository)
    agent_taskreco.update_failed_tasks(agent_capitalize.failed_tasks_repository)

    continue_identifying_tasks = True

    # Global learn loop
    while continue_identifying_tasks and time.time() < time_end:
        HumanLLMMonitor.step_id = str(uuid.uuid4())
        task = agent_taskreco.identify_best_task()

        # Handle multiple-tasks case
        if len(task) > 1:
            # list all tasks with their index and the 200 first characters of their content
            task_list = "Multiple task output, only one allowed - PLEASE SELECT:\n"
            for i, t in enumerate(task):
                task_list += f"\033[31m{i}\033[0m: {t.content[:200]}\n"
            smart_print(task_list, "orchestrate_agents", "orchestrate_agents SELECTION")
            # get input from user with the index of the task to select, manage exceptions
            while True:
                try:
                    id = int(smart_input("Enter the index of the task to select: ").strip())
                    if id in range(len(task)):
                        task = task[id]
                        break
                    else:
                        raise Exception("Index out of range")
                except Exception as e:
                    smart_print(f"Error: {e}\n\nEnter a valid index")
        else:
            task = task[0]
        smart_print("Identified Task: "+task.content.replace("\\n", "\n"), "orchestrate_agents", "orchestrate_agents RESULT")
        task_description = task.content
        # Extract potential score and state function code from the task
        if allow_custom_score_state_functions:
            # Extract current implementation of get_score and get_state from the first environment
            current_score_function_code, current_state_function_code = inspect.getsource(test_environments[0].get_score), inspect.getsource(test_environments[0].get_state)
            score_function_code, state_function_code = extract_function_code(task_description, 'get_score', current_score_function_code), extract_function_code(task_description, 'get_state', current_state_function_code)

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
            agent_taskreco.update_learnt_tasks(agent_capitalize.tasks_repository)
        else :
            if optuna_opti or smart_input("Do you want to capitalize this try as a 'failed task' to avoid this task to be proposed as a next best task ? (yes/no): ").strip().upper() in ["Y", "YES"]:
                agent_capitalize.capitalize_failed_tasks(task_description, parsed_code)
                agent_taskreco.update_failed_tasks(agent_capitalize.failed_tasks_repository)
        if optuna_opti:
            continue_identifying_tasks = False
        else:
            answer = smart_input("Do you want to reset the environment for searching a new task (Y/YES) or search a new task by keeping what has been created by this task (N/NO/Enter) ? or just exit (E/EXIT) ?").strip().upper()
            continue_identifying_tasks = False if answer in ["E", "EXIT"] else True
            if answer.upper() in ["Y", "YES"]:
                [env.reset() for env in test_environments]

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
        return total_score_weighted_with_stats
    else:
        return 0

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

def coding_and_validation_loop(agent_coding, agent_validation, task_description, max_attempts, extra_manual_validation_to_capitalize=True, continue_even_if_successful=True, optuna=None, end_time=None):
    previous_errors, previous_codes, previous_scores = [], [], []
    successful_codes = []  # To store successful codes
    all_results = []  # Store all results from each attempt for statistics

    for attempt in range(max_attempts):
        if end_time is not None and time.time() >= end_time:
            break
        results = agent_coding.code_task_and_run_test(task_description, previous_errors, previous_scores, previous_codes)
        all_results.extend(results) # Store all results for statistics
        previous_errors, previous_codes, previous_scores = [], [], [] #TEST reset

        # First part: Process all codes and collect results
        current_skip_rounds = agent_validation.human_llm_validate_code.skip_rounds
        for index, (parsed_code, no_runtime_error, exec_result, scores, env_states) in enumerate(results):
            agent_validation.human_llm_validate_code.skip_rounds = current_skip_rounds # to prevent skip_rounds decreased multiple times by multiple calls of HumanLLMMonitor
            smart_print(f"Generated code:\n{parsed_code['program_code']}\n*******\nOutput of code execution:\n{exec_result}\n".replace("\\n", "\n"), None, "coding_and_validation_loop RESULT")
            validation_agent_feedback = agent_validation.validate_code(parsed_code["program_code"], no_runtime_error, exec_result, task=task_description, scores=scores, env_states=env_states)
            smart_print("Agent validation 'feedback' currently only support 1 feedback", None, "coding_and_validation_loop WARNING")
            validation_agent_feedback = validation_agent_feedback[0]
            afb = validation_agent_feedback.content.replace('\\n', '\n')
            smart_print("#"*20 + f"\nAgent validation feedback: {afb}", None, "coding_and_validation_loop RESULT")

            if extra_manual_validation_to_capitalize:
                validated = (input("#"*20+f"\nADD THIS FUNCTION TO LIBRARY ? Please enter 'yes' if this a success and you want to add this function to library, 'no' if this failed: ").lower() in ["yes", "y", True])
            else:
                validated = get_success_value_in_text(afb) in ["yes", "y", True]
            if validated:
                successful_codes.append((parsed_code, validation_agent_feedback, scores))

            # Update previous errors and codes lists
            if index < len(previous_errors):
                previous_errors[index] = validation_agent_feedback
                previous_scores[index] = scores
                previous_codes[index] = parsed_code["program_code"]
            else:
                previous_errors.append(validation_agent_feedback)
                previous_scores.append(scores)
                previous_codes.append(parsed_code["program_code"])

        if successful_codes and not continue_even_if_successful:
            break

        if not successful_codes and attempt == (max_attempts / 2) - 1:
            attempt = max_attempts - 1
        
        if attempt == max_attempts - 1:
            if not successful_codes:
                smart_print("No successful code yet. Stop this task.", None, "coding_and_validation_loop WARNING")
            else:
                smart_print("Max attempts reached. Trying a new task.", None, "coding_and_validation_loop WARNING")

    # Calculate metrics over all attempts
    percentage_no_runtime_error = (sum(1 for _, no_runtime_error, _, _, _ in all_results if no_runtime_error) / len(all_results)) if all_results else 0
    best_score_without_validation = (max(max(sum(scores.values()) for scores in score_dict) for _, _, _, score_dict, _ in all_results)) if len(all_results) > 0 else 0
    all_scores = {
        'percentage_no_runtime_error': percentage_no_runtime_error,
        'best_score_without_validation': best_score_without_validation,
        'validated_scores': None}

    # Second part: If there are successful codes, ask user to select one
    if successful_codes and (not continue_even_if_successful or attempt >= max_attempts - 1) and (end_time is None or time.time() < end_time):
        if len(successful_codes)==1:
            selected_code, _, scores = successful_codes[0]
            all_scores['validated_scores'] = scores
            return selected_code, "success", all_scores
        if current_skip_rounds <= 0:
            for i, (parsed_code, feedback, scores) in enumerate(successful_codes):
                smart_print(f"\033[91mOption {i+1}:\033[0m\nCode:\n{parsed_code['program_code']}\nFeedback: {feedback.content}\n\033[91mScore: {scores}\033[0m\n", None, "coding_and_validation_loop RESULT")
            if optuna : 
                highest_score_index = get_highest_score_index([scores for _, _, scores in successful_codes], mode='total')
                selected_code, _, scores = successful_codes[highest_score_index]
                all_scores['validated_scores'] = scores
                return selected_code, "success", all_scores
            else:
                selection = smart_input("Several codes were successful. Please enter the number of the code you want to add to the library: ").strip()
            if selection.isdigit() and 0 < int(selection) <= len(successful_codes):
                selected_index = int(selection) - 1
                smart_print("Code validated successfully.", None, "coding_and_validation_loop RESULT")
                selected_code, _, scores = successful_codes[selected_index]
                all_scores['validated_scores'] = scores
                return selected_code, "success", all_scores
            else:
                smart_print("Invalid selection or no selection made. Exiting without adding any code.", None, "coding_and_validation_loop WARNING")
        else: # if in automatic mode, select the code with the highest score
            highest_score_index = get_highest_score_index([scores for _, _, _, scores in successful_codes], mode='total')
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
    
def create_Nmajority_chain(num_models=3, map_model_name="gpt-3.5", reduce_model_name="gpt-3.5", map_temperature=0.7, reduce_temperature=0.7):
    # Initialize the OpenAI models
    models = [ChatOpenAI(model_name=map_model_name, temperature=map_temperature, cache=False) for _ in range(num_models)]
    final_model = ChatOpenAI(model_name=reduce_model_name, temperature=reduce_temperature, cache=False)

    # Define the chain using LCEL
    response_keys = [f"response_{i+1}" for i in range(num_models)]
    multi_reponse = {key: model for key, model in zip(response_keys, models)}
    multi_reponse["cleaned_input"] = ExtractMessage()

    final_prompt_template_str = (
        "Given the following responses below and the initial question below, provide an optimal response to the question mixing best elements of each and following the same answer output structure:\n\n"
        "Initial question:\n{cleaned_input}\n\n" +
        "\n\n".join([f"Response {i+1}:\n{{{response_keys[i]}}}" for i in range(num_models)]) +  # Access content directly
        "\n\nOptimal Response:\n"
    )

    # Insert the PrintPromptRunnable just to print the prompt for control
    print_prompt_runnable = PrintPromptRunnable()

    # Print in RED the final prompt template: print("\033[31m"+final_prompt_template_str+"\033[0m")
    final_prompt_template = ChatPromptTemplate.from_template(final_prompt_template_str)

    chain = multi_reponse | final_prompt_template | print_prompt_runnable | final_model # No ERROR but bad output: "I'm sorry, but I cannot fulfill this request" or "I'm sorry, but I cannot fulfill this request as it is too complex for me to process." or "I'm sorry, but I cannot fulfill this request as it involves creating a Python function and providing a specific response format." or "I'm sorry, but I cannot fulfill this request as it requires a level of understanding and reasoning that is beyond my current capabilities."

    return chain


if __name__ == "__main__":
    # Initialize the default and premium LLMs
    #from langchain_groq import ChatGroq
    llmORchains_list = {
        "default_llm": ChatOpenAI(model_name=MODELS_CONFIG_LIST["basic_gpt" if "basic_gpt" in MODELS_CONFIG_LIST else "gpt"], cache=False),#, temperature=0.7),
        "premium_llm": ChatOpenAI(model_name=MODELS_CONFIG_LIST["smart_gpt"], cache=False),#, temperature=0.7),
        #"default_llm": ChatGroq(model_name=MODELS_CONFIG_LIST["basic_gpt" if "basic_gpt" in MODELS_CONFIG_LIST else "gpt"], cache=False),#, temperature=0.7),
        #"premium_llm": ChatGroq(model_name=MODELS_CONFIG_LIST["code_gpt"], cache=False),#, temperature=0.7),
        "3_majority_chain": create_Nmajority_chain(map_model_name=MODELS_CONFIG_LIST["basic_gpt"], reduce_model_name=MODELS_CONFIG_LIST["basic_gpt"], num_models=3),
        "10_majority_chain": create_Nmajority_chain(map_model_name=MODELS_CONFIG_LIST["basic_gpt"], reduce_model_name=MODELS_CONFIG_LIST["basic_gpt"], num_models=10)
    }

    # Set the documents to test/validate as a list of environments
    documents=[
        {
        "id": "68802377-e8e5-4940-a337-e66930ba5015",
        "title": "Erreur de connexion au serveur interne",
        "context": "Lors de la tentative de connexion au serveur interne de l'entreprise, les utilisateurs rencontrent un message d'erreur indiquant une impossibilité de se connecter. Ce problème semble intermittent et affecte principalement les utilisateurs du département des ventes.",
        "target_file_path": "env/IR_CPS_TechSynthesis/document_embedding_analysis/output/fiche_ano/Erreur de connexion au serveur interne.json"
        },

        {
        "id": "c0533337-1c5e-4091-ba7d-061ae409cda4",
        "title": "Problème de performance sur le module de gestion des utilisateurs",
        "context": "Lors de l'utilisation du module de gestion des utilisateurs, nous avons constaté des ralentissements significatifs. Les utilisateurs rapportent que la page met plusieurs minutes à se charger et que les opérations de modification et de suppression d'utilisateur prennent un temps anormalement long. Ce problème a été observé sur plusieurs navigateurs et sur différentes configurations matérielles, ce qui suggère qu'il ne s'agit pas d'un problème isolé à un utilisateur spécifique ou à un type de machine. Nous avons identifié que ce problème semble se produire principalement lorsque le nombre d'utilisateurs dépasse les 1000. Les logs du serveur montrent des temps de réponse élevés sur les requêtes liées à la base de données pour ce module spécifique. Une analyse initiale indique que certaines requêtes ne sont pas optimisées et causent des verrous au niveau de la base de données.",
        "target_file_path": "env/IR_CPS_TechSynthesis/document_embedding_analysis/output/fiche_ano/Problème de performance sur le module de gestion des utilisateurs.json"
        }
    ]
    
    envs = []
    for doc in documents:
        env = EnvironmentManager(env_type="techsynthesis", title=doc['title'], context=doc['context'], target_file_path=doc['target_file_path'], id=doc['id']).get_environment()
        envs.append(env)
    
    # Run the learning loop
    run_4agents_learning_loop(default_llm_key="default_llm", 
                              premium_llm_key="premium_llm",
                              llmORchains_list=llmORchains_list,
                              test_environments=envs, 
                              manual_validation_to_capitalize=False, 
                              problem_prompts_subdir=None,
                              max_coding_attempts=4, 
                              include_code=False, 
                              selected_successful_functions=[], 
                              selected_failed_functions=[], 
                              max_execution_time=900,
                              agtask_premium_llm_by_default=False,
                              agtask_skip_rounds=0, # Auto-test: 1 
                              agcoding_skip_rounds=0, # Auto-test: 4
                              agvalidation_skip_rounds=0, # Auto-test: 4
                              agcapitalize_skip_rounds=0) # Auto-test: 0"""