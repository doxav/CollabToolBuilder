import ast
import contextlib
import hashlib
import io
import json
import os
import re
import shutil
import traceback
import types
import uuid
from utils.llm_utils import validate_function_code

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
        # Memorize current directory and switch to temporary directory
        # current_dir = os.getcwd()
        # os.chdir(self.current_temp_dir)

        # Ensure `result` is set in the code
        if not re.search(r'\bresult\s*=', action_code.strip().splitlines()[-1]):
            helper = "\nresult = locals().get('_', True)"
        else:
            helper = ""

        # Setup for capturing stdout and stderr
        stdout, stderr, std_out_err = io.StringIO(), io.StringIO(), {}

        try:
            # Execute code with redirected stdout and stderr
            print(f"Code execution context: <<<{context}>>>")
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                exec(action_code + helper, context)
            std_out_err = {"stdout": stdout.getvalue(), "stderr": stderr.getvalue()}
            # Safely evaluate and retrieve result
            exec_result = ast.literal_eval(repr(context.get('result', True)))
            no_runtime_error = True
        except Exception as e:
            # Format traceback and include captured output for clarity
            error_traceback = ''.join(traceback.format_exception(None, e, e.__traceback__))
            exec_result = f"Execution failed. Error: {e}\nTraceback:\n{error_traceback}\nStdout:\n{stdout.getvalue()}\nStderr:\n{stderr.getvalue()}"
            no_runtime_error = False
        # finally:
            # Restore original directory
            # os.chdir(current_dir)

        return no_runtime_error, exec_result, std_out_err

    def close(self, backup_previous_temp_dir=True):
        # move temp directory and its content including the data link to backups directory
        if backup_previous_temp_dir:
            shutil.move(self.current_temp_dir, os.path.join(self.temp_root_dir, "backups"))
        else:
            shutil.rmtree(self.current_temp_dir)

    def backup_state(self, unique_id: str = None):
        # copy all the temp directory (excluding data directory) into a folder named by unique_id into backups directory
        if unique_id is None:
            self.last_unique_id_backup = os.environ["unique_id"] = str(uuid.uuid4())
        unique_id = os.environ.get("unique_id")
        if self.get_state(unique_id) != self.get_state():
            if os.path.exists(os.path.join(self.temp_root_dir, "backups", unique_id)):
                shutil.rmtree(os.path.join(self.temp_root_dir, "backups", unique_id))
            shutil.copytree(self.current_temp_dir, os.path.join(self.temp_root_dir, "backups", unique_id), ignore=shutil.ignore_patterns('data'))
        return os.environ.get("unique_id")

    def restore_state(self, unique_id):
        from_folder = os.path.join(self.temp_root_dir, "backups", unique_id)
        if not os.path.exists(from_folder): return "Restore state folder not found"
        if self.get_state(unique_id) != self.get_state():
            # copy all the content of the backup directory into the temp directory (excluding data directory)
            self.reset(backup_previous_temp_dir=False)
            # copy all the content of the backup directory into the temp directory which already contains the data directory
            shutil.copytree(
                from_folder,
                self.current_temp_dir,
                ignore=shutil.ignore_patterns('data'),
                dirs_exist_ok=True
            )
            return "State restored"
        else:
            return "State identical to backup folder"

    def restore_last_state(self):
        if self.last_unique_id_backup:
            self.restore_state(self.last_unique_id_backup)
            return "Last state restored"
        else:
            return "No last state to restore"

    def get_state(self, extended_comparison=False):
        # return a dictionary containing the content of the temp directory
        state = {}
        state_folder = self.current_temp_dir if os.environ.get("unique_id") is None else os.path.join(self.temp_root_dir, "backups", os.environ.get("unique_id"))
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
        elif env_type == "sweSynthesis":
            from env.SWEBench.env import SWEBenchEnvironment
            # passing all the parameteres   all the args from the EnvironmentManager
            self.env = SWEBenchEnvironment(**kwargs)    
        else:
            self.env = Environment()
        self.env.reset()

    def get_environment(self):
        return self.env

