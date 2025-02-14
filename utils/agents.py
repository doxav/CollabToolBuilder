import sys, json, time, os, difflib, random, pickle
from typing import List, Dict
from utils.llm_utils import (
    apply_criteria_and_prepare_monitor_args, get_primitives,
    smart_print, smart_input, _visual_input
)
from env.env import Environment, EnvironmentManager
from utils.file_utils import extract_functions_ast
from utils.human_llm import HumanLLM
from env.SWEBench.env import SWEBenchEnvironment

from langchain_core.messages.human import HumanMessage
from langchain_core.messages.ai import AIMessage
from langchain_core.messages.system import SystemMessage
from langchain_core.messages.base import BaseMessage

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
        self.additional_check_list = None
        self.name = self.__class__.__name__
        self.problem_prompts_subdir = "" if problem_prompts_subdir is None else problem_prompts_subdir + "/"
        self.criteria = criteria
        self.params_user_message = params_user_message
        self.temperature_min = temperature_min
        self.temperature_max = temperature_max
        self.envs = envs
        self.automation = automation
        self.model_choice = model_choice
        self.primitives_dir = primitives_dir
        
        kw_common_args = apply_criteria_and_prepare_monitor_args(self, special_criteria, locals())

        self.human_llm_identify_best_task = HumanLLM(**kw_common_args)
        

        self.human_llm_identify_best_task.skip_rounds = skip_rounds
        if self.additional_check_list:
            for key, value in self.additional_check_list.items():
                self.human_llm_identify_best_task.add_inference_check(key, value)
        if hasattr(self, 'recommendations_usage'):
            if self.recommendations_usage:
                self.human_llm_identify_best_task.add_inference_check(
                    "Recommendations",
                    self.human_llm_identify_best_task.generate_best_improvement_suggestions
                )
        else:
            self.human_llm_identify_best_task.add_inference_check(
                "Recommend Critics",
                self.human_llm_identify_best_task.generate_best_improvement_suggestions
            )

    def identify_best_task(self):
        # Prepare data
        envs_status = "\n".join([env.get_state() for env in self.envs])
        few_shots = self.human_llm_identify_best_task.get_multiple_few_shots(
            few_shots_params=self.params_user_message)
        print(few_shots)
        self.human_llm_identify_best_task.user_message_few_shots = self.params_user_message
 
        primitives = extract_functions_ast("\n".join(get_primitives(self.primitives_dir)), include_docstring=True, return_string=True)
        successful_tasks = "\n".join(self.human_llm_identify_best_task.get_learnt_tasks())
        failed_tasks = "\n".join(self.human_llm_identify_best_task.get_failed_tasks())

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
        print(f"User message: {user_message}")
        sys.stdout=original_stdout

        task = self.human_llm_identify_best_task.CallHumanLLM(
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
        self.additional_check_list = None
        self.name = self.__class__.__name__
        self.max_autofix = None
        self.problem_prompts_subdir = "" if problem_prompts_subdir is None else problem_prompts_subdir + "/"
        self.envs = envs
        self.automation = automation
        self.model_choice = model_choice
        self.processed_codes = set()

        self.problem_prompts_subdir = "" if problem_prompts_subdir is None else problem_prompts_subdir + "/"
        self.last_user_message = None
        if hasattr(self, 'num_parallel_inferences') and self.num_parallel_inferences == 0:
            self.num_parallel_inferences = 2

        kw_common_args = apply_criteria_and_prepare_monitor_args(self, special_criteria, locals())

        self.human_llm_code_task = HumanLLM(**kw_common_args)

        self.human_llm_code_task.skip_rounds = skip_rounds
        self.human_llm_code_task.add_inference_check("Code Parsing", self.parse_ai_generated_code)
        self.human_llm_code_task.add_inference_check("Run Tests", self.run_tests_on_code)
        if hasattr(self, 'recommendations_usage'):
            if self.recommendations_usage:
                self.human_llm_code_task.add_inference_check("Recommend Critics",
                                                             self.human_llm_code_task.generate_best_improvement_suggestions)
        else:
            self.human_llm_code_task.add_inference_check("Recommend Critics",
                                                         self.human_llm_code_task.generate_best_improvement_suggestions)
        if self.additional_check_list:
            for key, value in self.additional_check_list.items():
                self.human_llm_code_task.add_inference_check(key, value)
        # "CodingAgent#additional_check_list": [(check_name, check_function), ...]
        self.primitives_dir = primitives_dir

    def parse_ai_generated_code(self, message, language="py", retry=3, required_bot_arg=None, task_definition=None,
                                automatic_tests=False, output_id=None):
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
                        code_pattern = re.compile(r"```python(.*?)(```|$)", re.DOTALL)
                        codes_match = [match[0].strip() for match in code_pattern.findall(message)]
                        code = "\n".join(codes_match) if codes_match else message

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
                        assert required_bot_arg in main_function[
                            "params"], f"Main function {main_function['name']} must take an argument named '{required_bot_arg}'"

                    # Assemble the final program code
                    program_code = "\n".join(imports) + "\n"
                    program_code += "\n\n".join(function["body"] for function in functions)

                    tests_pattern = re.compile(r'\n#\s+document #([a-z0-9-]+)\s+.*test[^\n]*?\n([^\n]+)|{.*?\"documentid\":\s*\"#(.*?)\",\s*\"FunctionCall\":\s*\"([^\"]+)\".*?}', re.IGNORECASE)
                    matches = tests_pattern.findall(task_definition if task_definition is not None else self.last_user_message)

                    if matches and not automatic_tests:
                        tests = [match[:2] if match[0] != "" else match[2:] for match in matches]
                    else:
                        print("Running default tests with bot argument to main function")
                        tests = [(env.id, main_function["name"] + "(bot)") for env in self.envs]

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
        smart_print(f"CODE PARSING ERROR!!!\n{error}", self.name, "code_task_and_run_test SystemMessage",
                    optional=False, column_id=output_id)
        return False, self.parsed_code

    def generate_score(self, index:int, success:bool, score:Dict, elapsed_time:int)-> str:
        str_score = f"{index}. "
        if success:
            str_score += "\033[32mSUCCESS"
        else:
            str_score += "\033[31mFAILED"
        str_score += "\033[0m / SCORE: "
        str_score += f"[{','.join(f'{a:.2f}' for a in list(score.values()))}]"
        str_score += f" / TIME: {elapsed_time}s"
        str_score += f" / CODE: [{json.dumps(score)}]"
        return str_score

    def run_tests_on_code(self, message, parsed_code=None, skip_already_processed=False, output_id=None,
                          restore_state=True, custom_agent=None):
        # Retrieve error_patches from HumanLLMMonitor
        metadata = {'step_id': HumanLLM.step_id}
        error_patches, _ = HumanLLM.get_agent_data(self.name, 'error_patches', metadata_filter=metadata)
        error_patches = error_patches if error_patches else []

        primitives = get_primitives(self.primitives_dir)
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
                smart_print(f"SKIPPING TEST: code error on first env, skipping test {idx}",
                            custom_agent if custom_agent else self.name, "code_task_and_run_test SystemMessage",
                            optional=False, column_id=output_id)
            else:
                # Determine tests to run or set default runnable code
                matching_tests = [test for doc_id, test in parsed_code["tests"] if doc_id == env.id] if parsed_code[
                    "tests"] else [parsed_code['runnable_code']]
                if not matching_tests:
                    no_runtime_error, exec_result = False, f"Error: no test found for given id {env.id}" if parsed_code["tests"] else f"Error: no runnable code found nor tests"
                else:
                    # Concatenate common code with program and tests or runnable code
                    code_to_run = common_code + parsed_code["program_code"] + "\n" + "\n".join(matching_tests)
                    smart_print("TESTING GENERATED CODE.....", custom_agent if custom_agent else self.name,
                                "code_task_and_run_test SystemMessage", append=True, optional=False, column_id=output_id, column_max=self.human_llm_code_task.num_parallel_inferences)
                    
                    t1 = time.time()
                    no_runtime_error, exec_result, std_out_err = env.step(code_to_run)
                    # Smart print the std_out_err
                    smart_print(f"FUNCTION DISPLAY OUTPUTS:\n{std_out_err}", custom_agent if custom_agent else self.name, "code_task_and_run_test SystemMessage", append=True, optional=False, column_id=output_id)
                    # Show score
                    scores = self.generate_score(idx, no_runtime_error, env.get_score(), time.time() - t1)
                    smart_print(scores, custom_agent if custom_agent else self.name, "Scores", append=True, optional=False, column_id=output_id, column_max=self.human_llm_code_task.num_parallel_inferences)

                    if no_runtime_error:
                        smart_print("TEST SUCCESSFUL", custom_agent if custom_agent else self.name, "code_task_and_run_test SystemMessage", optional=False, column_id=output_id)
                        smart_print(env.get_state(), custom_agent if custom_agent else self.name, "CODE_RESULT", optional=False, column_id=output_id)
                    while not no_runtime_error and current_skip_rounds <= 0:
                        smart_print("\033[31mCODE ERROR\033[0m: " + exec_result,
                                    custom_agent if custom_agent else self.name,
                                    "code_task_and_run_test SystemMessage", optional=False, column_id=output_id)
                        if self.automation and max_autofix is not None:
                            decision = "a" if max_autofix > 1 else "no"
                            if decision == "a":
                                max_autofix -= 1
                        elif self.automation:
                            decision = "n"
                        else:
                            if output_id == None:
                                output_id = 0
                            smart_print(parsed_code["program_code"], custom_agent if custom_agent else self.name,
                                        f"Inference streaming output {output_id}", append=True, column_id=output_id,
                                        column_max=self.human_llm_code_task.num_parallel_inferences)
                            decision = smart_input(
                                f"ANSWER {output_id} Do you want to edit the code to fix the error (you will also be requested first) ? (yes/no) or try autofix by LLM (a): ",
                                custom_agent if custom_agent else self.name, "fix_error", column_id=output_id).strip()
                        decision_lower = decision.lower()
                        if decision_lower in ("no", "n", ""):
                            break
                        elif decision_lower in ["y", "yes"]:
                            # if in websocket, then get from self.human_llm_code_task.premium_llm
                            if HumanLLM.use_websocket:
                                edited_code = self.human_llm_code_task.temp_inference_result_content

                            else:
                                edited_code = _visual_input(parsed_code["program_code"], filetype="py",
                                                            message_type="fix_error", agent_name=self.name,
                                                            column_id=output_id)
                        else:
                            if decision_lower == "a":
                                # Do not use HumanLLMMonitor because no template is available for this specific case
                                smart_print("TRYING TO AUTOFIX ERROR", custom_agent if custom_agent else self.name,
                                            "fix_error", optional=True, column_id=output_id)
                                instructions = ""
                            else:
                                # User provided custom instructions
                                instructions = decision
                            fix_system_prompt = f"""You are a Python expert in code debugging.
                            You are provided with ERROR MESSAGE and the CODE TO FIX.
                            {instructions}
                            Reply with the full Python code fixed and ready to be executed without the triple quotes and python tags. You add comments in the code to explain your fix.
                            """
                            smart_print("ANALYZING ERROR.....", custom_agent if custom_agent else self.name,
                                        "code_task_and_run_test SystemMessage", optional=False, column_id=output_id)
                            help_for_fixing_system_prompt = f"""You help an LLM to fix code errors which has no access to documentation or internet by extracting key code information from the INFORMATION/DOCUMENTATION provided given CODE TO FIX and ERROR MESSAGE."""
                            error_with_info_to_help_prompt = f"ERROR MESSAGE:<<\n{exec_result}\n>>\n\nCODE TO FIX:<<\n{parsed_code['program_code']}\n>>\n\INFORMATION/DOCUMENTATION:<<\n{self.last_user_message}\n>>"
                            help_code_returned = self.human_llm_code_task.premium_llm.invoke(
                                [SystemMessage(content=help_for_fixing_system_prompt),
                                 HumanMessage(content=error_with_info_to_help_prompt)])
                            smart_print("ANALYSIS RECIEVED, GENERATING A FIX",
                                        custom_agent if custom_agent else self.name,
                                        "code_task_and_run_test SystemMessage", optional=False, column_id=output_id)
                            fix_description_prompt = f"ERROR MESSAGE:<<\n{exec_result}\n>>\n\nCODE TO FIX:<<\n{parsed_code['program_code']}\n>>\n\nHELPFUL INFORMATION:<<\n{getattr(help_code_returned, 'content', help_code_returned)}\n>>"
                            edited_code_returned = self.human_llm_code_task.premium_llm.invoke(
                                [SystemMessage(content=fix_system_prompt),
                                 HumanMessage(content=fix_description_prompt)])
                            edited_code = str(getattr(edited_code_returned, 'content', edited_code_returned))

                        # Before updating parsed_code, save the previous code
                        prev_code = parsed_code["program_code"]
                        # Run the edited code
                        code_to_run = common_code + edited_code + "\n" + "\n".join(matching_tests)
                        smart_print("TESTING UPDATED CODE.....", custom_agent if custom_agent else self.name,
                                    "code_task_and_run_test SystemMessage", optional=False, column_id=output_id)

                        t1 = time.time()
                        no_runtime_error, exec_result, std_out_err = env.step(code_to_run)
                        # Smart print the std_out_err
                        smart_print(f"FUNCTION DISPLAY OUTPUTS:\n{std_out_err}", custom_agent if custom_agent else self.name, "code_task_and_run_test SystemMessage", optional=False, column_id=output_id)
                        # Show score
                        scores = self.generate_score(idx, no_runtime_error, env.get_score(), time.time() - t1)
                        smart_print(scores, custom_agent if custom_agent else self.name, "Scores", optional=False, column_id=output_id)

                        # Update parsed_code if re-run is successful
                        parsed_code["program_code"] = edited_code
                        smart_print(
                            f"# UPDATED **{'SUCCESFUL' if no_runtime_error else 'FAILED'}** CODE:\n{edited_code}",
                            custom_agent if custom_agent else self.name, "UPDATED_CODE", optional=False,
                            column_id=output_id)
                        # If no runtime error, store the error and diff
                        if no_runtime_error:
                            smart_print(env.get_state(), custom_agent if custom_agent else self.name, "CODE_RESULT", optional=False, column_id=output_id)
                            diff = difflib.unified_diff(prev_code.splitlines(), edited_code.splitlines(), lineterm='')
                            diff_text = '\n'.join(diff)
                            # Avoid duplicates: check if the error and diff combination already exists
                            if (exec_result, diff_text) not in error_patches:
                                error_patches.append((exec_result, diff_text))
                                # Store updated error_patches
                                HumanLLM.add_agent_data(self.name, 'error_patches', error_patches, metadata=metadata)

            no_runtime_errors.append(no_runtime_error)
            exec_results.append(exec_result)

        # Calculate total time taken  # Added line
        total_execution_time = time.time() - start_time
        # Return combined results
        if isinstance(self.envs[0], SWEBenchEnvironment):
            scores = [env.get_score(parsed_code["program_code"]) for env in self.envs]
        else:
            scores = [env.get_score() for env in self.envs]
        result = (parsed_code, all(no_runtime_errors), exec_results, scores, [env.get_state() for env in self.envs], total_execution_time)

        if restore_state:
            for env in self.envs:
                env.restore_last_state()
        return result

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
        metadata = {'step_id': HumanLLM.step_id}
        previous_errors, _ = HumanLLM.get_agent_data(self.name, 'previous_errors', metadata_filter=metadata)
        previous_scores, _ = HumanLLM.get_agent_data(self.name, 'previous_scores', metadata_filter=metadata)
        previous_codes, _ = HumanLLM.get_agent_data(self.name, 'previous_codes', metadata_filter=metadata)
        error_patches, _ = HumanLLM.get_agent_data(self.name, 'error_patches', metadata_filter=metadata)

        # Ensure variables are initialized
        previous_errors = previous_errors if previous_errors else []
        previous_scores = previous_scores if previous_scores else []
        previous_codes = previous_codes if previous_codes else []
        error_patches = flatten_and_pair(error_patches) if error_patches else []

        env_states = "\n".join([env.get_state() for env in self.envs])
        primitives = "\n".join(get_primitives(self.primitives_dir))
        successful_tasks = "\n".join(HumanLLM.get_learnt_tasks())
        failed_tasks = "\n".join(HumanLLM.get_failed_tasks())
        validation_response_um = "\n".join(HumanLLM.get_validation_results())

        print(f"Primitives: <<<\n{primitives}\n>>>")

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
        user_message = HumanLLM.load_prompt("coding_agent_user_message_template", template_data=template_data,
                                                   directory='prompts')

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
            if check_results:
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
                if self.automation:
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
        #super().__init__(llm)
        saved_task, temperature_max, num_parallel_inferences, recommend_critiques, auto_n_rounds = None, None, 1, None, None
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
                self.human_llm_validate_code.add_inference_check(key, value)

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

        code_validation = self.human_llm_validate_code.CallHumanLLM(
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
        self.additional_check_list = None
        self.name = self.__class__.__name__
        saved_task, auto_n_rounds, num_parallel_inferences, temperature_max, recommend_critics = None, None, 1, None, None

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
                self.human_llm_generate_function_description.add_inference_check(key, value)

    def capitalize_successful_tasks(self, task_description: str, parsed_code: str) -> None:

        print('Starting capitalize_successful_tasks')
        import socket, uuid, datetime
        self.human_llm_generate_function_description.task_parameters = {'task_description' : task_description, 'parsed_code' : parsed_code}
        print(f"DEBUG CACA : {hasattr(self, 'saved_task')}, {self.automation}, {self.automation in ['before', 'after']}")
        if hasattr(self.human_llm_generate_function_description, 'saved_task') and self.automation in ['before', 'after']:
            content = self.human_llm_generate_function_description.saved_task.get('content', {})
            print(f"task parameters : {content.get('task_parameters', {})}")
            task_description = content.get("task_parameters", {}).get("task_description", task_description)
            parsed_code = content.get("task_parameters", {}).get("parsed_code", parsed_code)
            self.human_llm_generate_function_description.task_parameters = {'task_description': task_description,'parsed_code': parsed_code}

        function_name = parsed_code.get("main_function_name",
                                        parsed_code.get("main_function", {}).get("name", "unknown"))
        query = {
            "query": {
                "term": {
                    "metadata.main_function_name.keyword": function_name
                }
            }
        }
        response = UnifiedVectorDB.elastic_client.search(index=HumanLLM.db_learnt_tasks.collection_name, body=query)

        if response['hits']['total']['value'] > 0:
            for hit in response['hits']['hits']:
                doc_id = hit["_id"]
                UnifiedVectorDB.elastic_client.delete(index=HumanLLM.db_learnt_tasks.collection_name, id=doc_id)
                print(f"Document deleted : {doc_id}")

        if self.problem_prompts_subdir == "Anomalies/" or self.problem_prompts_subdir == "pipeline_synthesis/":

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
                    self.name, "capitalize_successful_tasks WARNING")
                if self.automation:
                    # generate an id based on the current time and a random number
                    id = datetime.datetime.now().strftime("%Y%m%d%H%M%S") + "_" + str(random.randint(0, 1000))
                    function_file_path = os.path.join("functions", self.name + f"_{id}.py")
                else:
                    function_file_path = os.path.join("functions", smart_input(
                        "This function already exists, please provide a new function name: ",
                        message_type="VALIDATION_INFO") + ".py")

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
        tags = {"host": f"{socket.gethostname()}-{uuid.getnode()}", "step_id": int(HumanLLM.step_id), ("class_name" if (self.problem_prompts_subdir == "Anomalies/" or self.problem_prompts_subdir == "pipeline_synthesis/") else "main_function_name"): function_name,}
        HumanLLM.add_learnt_task(serialized_entry, tags)

    def capitalize_failed_tasks(self, task_description: str, parsed_code: str) -> None:
        import socket, uuid, datetime, json

        is_anomaly = self.problem_prompts_subdir == "Anomalies/" or self.problem_prompts_subdir == "pipeline_synthesis/"
        name_key = "class_name" if is_anomaly else "main_function_name"
        if parsed_code:
            task_name = parsed_code.get(name_key,
                                        "replace this text with a descriptive name of the class" if is_anomaly else "replace this text with a descriptive name of the function")
        else:
            if False or not self.automation:  # TODO: temporary disabled, find the logic to fix this or if not required
                task_name = smart_input(
                    f"CONFIG Please provide a name for the {'pipeline' if is_anomaly else 'function'}:\n {task_description}",
                    "CapitalizationAgent", message_type="Capitalization_info").strip()
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
            "host": HumanLLM.get_host_id(),
            "step_id": HumanLLM.step_id,
        }
        HumanLLM.add_failed_task(serialized_entry, tags)

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
        tags = {"host": HumanLLM.get_host_id(),
                "step_id": HumanLLM.step_id, }

        self.db_failed_tasks.add_texts(texts=[serialized_entry], metadatas=[tags])

    def generate_tool_description(self, program_name, program_code):
        user_message = f"MAIN FUNCTION: `{program_name}`\n\nFULL CODE:\n{program_code}"

        if hasattr(self, 'log_user_message') and self.log_user_message:
            with open(self.log_user_message, "a") as f:
                f.write("Capitalization -- generate_tool_description:<<\n" + user_message + "\n>>\n\n")

        tool_description = self.human_llm_generate_function_description.CallHumanLLM(
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
        print("Getting the learnt tasks...")
        learnt_tasks = HumanLLM.get_learnt_tasks(k=30)
        print(f"Learnt_tasks: {learnt_tasks}")
        if not learnt_tasks:
            smart_print("No learnt tasks are available to answer the question.", agent_name=self.name)
            print("No learnt tasks are available to answer the question.")
            return "no code available"

        # Prepare the code snippets string
        code_snippets = "\n\n".join(learnt_tasks)

        # if self.envs:
        #     if isinstance(self.envs[0], SWEBenchEnvironment):
        #         question = f"Solve the SWE bench problem {self.envs[0].swe_data.instance_id} by finding buggy files, generating patch, applying patch and running tests. Current state of the solution is:\n{self.envs[0].get_state()}.\nOn the basis of current state, provide the best code that shoud be executed next"

        # print(question)

        print("Got the learnt tasks")

        prebuilt_code = ""
        for learnt_task in learnt_tasks:
            loaded = json.loads(learnt_task)
            prebuilt_code += loaded['program_code'] + "\n"

        # Prepare the prompt for the LLM
        prompt_path = reuse_prompt if reuse_prompt else self.system_prompt_path
        prompt_template = HumanLLM.load_prompt(prompt_path) if prompt_path else (
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

        selected_code = self.human_llm_planner.CallHumanLLM(original_input_messages=[SystemMessage(content=prompt), HumanMessage(content=user_message_content)]) #, automation=self.automation)
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
