import json, uuid
import os
import os.path
import subprocess
import ast
import textwrap
from typing import Any, Dict, List, Optional
from pydantic import BaseModel
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_community.tools import tool
from config import *
from env.IR_CPS_TechSynthesis.env import Environment


class SWEProblem(BaseModel):
    instance_id: str
    text: str
    repo: str
    base_commit: str
    problem_statement: str
    hints_text: str
    test_patch: str
    environment_setup_commit: str
    repo_structure: Optional[Dict[str, Any]] = None

class SWEBenchEnvironment(Environment):

    swe_data: SWEProblem
    swe_repos_path = "env/SWEBench/repos"
    swe_temp_path = "env/SWEBench/temp"

    def __init__(self,
        swe_data: SWEProblem,   
        llm = None,
        id: str = None,
        reset_env: bool = False,  
        ):
        super().__init__()
        
        self.id = str(uuid.uuid4()) if id is None else id
        SWEBenchEnvironment.llm_model = llm
        self.swe_data = swe_data
        self.reset_env = reset_env
        SWEManager.target_dir = f"env/SWEBench/repos/{self.swe_data.repo.split('/')[-1]}"
        SWEManager.current_dir = "/"
        self.swe_manager = SWEManager() 
        self.setup_repo()
    
    @staticmethod
    def llm(prompt: str):
        return SWEBenchEnvironment.llm_model.invoke([SystemMessage(content=""), HumanMessage(content=prompt)]).content
    
    def reset(self):
        repo_name = self.swe_data.repo.split('/')[-1]
        print(subprocess.run(f"cd {self.swe_repos_path}/{repo_name} & git branch -d {self.swe_data.instance_id}", shell=True, text=True, capture_output=True).stdout)
        print(subprocess.run(f"cd {self.swe_repos_path}/{repo_name} & git checkout -b {self.swe_data.instance_id} {self.swe_data.base_commit}", shell=True, text=True, capture_output=True).stdout)

        temp_dir = os.path.join(self.swe_temp_path, self.swe_data.instance_id)
        with open(os.path.join(temp_dir, "state.json"), "r") as state_file:
            state = json.load(state_file)
            state['patch_applied'] = False

        with open(os.path.join(temp_dir, "state.json"), "w") as state_file:
            json.dump(state, state_file)

    def setup_repo(self):
        temp_dir = os.path.join(self.swe_temp_path, self.swe_data.instance_id)
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)
            if not os.path.exists(os.path.join(temp_dir, "state.json")):
                initial_state = {
                    'buggy_files': [],
                    'buggy_files_content': "",
                    'generated_patch_code': "",
                    'patch_applied': False,
                    'unit_tests': [],
                    'unit_test_passed': [],
                    'unit_test_failed': []
                }
                with open(os.path.join(temp_dir, "state.json"), "w") as state_file:
                    json.dump(initial_state, state_file)

        repo_name = self.swe_data.repo.split('/')[-1]
        SWEManager.target_dir = f"{self.swe_repos_path}/{repo_name}" 

        if not os.path.exists(f"{self.swe_repos_path}/{repo_name}"):
            output = subprocess.run(f"cd {self.swe_repos_path} & git config --global http.postBuffer 524288000 & git clone https://github.com/{self.swe_data.repo}.git", shell=True, text=True, capture_output=True).stdout
            if output == "":
                print("Repo cloned successfully")
            else:
                print(output)
                print("Error cloning repo")

        self.reset()
        print('Repo setup complete')    
    
    def backup_state(self, unique_id = None):
        temp_dir = os.path.join(self.swe_temp_path, self.swe_data.instance_id)
        with open(os.path.join(temp_dir, "state.json"), "r") as state_file:
            state = json.load(state_file)

        with open(os.path.join(temp_dir, "state_backup.json"), "w") as state_backup_file:
            json.dump(state, state_backup_file)
    
    def restore_last_state(self):
        temp_dir = os.path.join(self.swe_temp_path, self.swe_data.instance_id)
        with open(os.path.join(temp_dir, "state_backup.json"), "r") as state_backup_file:
            state_backup = json.load(state_backup_file)

        with open(os.path.join(temp_dir, "state.json"), "w") as state_file:
            json.dump(state_backup, state_file)
   
    def get_state(self, extended = None) -> str:
        with open(os.path.join(self.swe_temp_path, self.swe_data.instance_id, "state.json"), "r") as state_file:
            state = json.load(state_file)
        return json.dumps(state, indent=4)
    
    def step(
        self,
        code: str = "",
    ):
        return super().step(action_code=code, context={'problem': self.swe_data, 'bot': self.swe_manager, 'env': self, 'results': None})
        

class SWEManager:
    current_dir: str = "/" # The current directory within which tools are navigating in the problem repository
    target_dir: str = ""  # The root directory for the current problem repository
    
    def getSearchTools(self):
     return [ls, goto_directory, goto_previous_dir, get_current_dir, number_of_lines, open_file, find_files, search_file, search_dir]

    def getContentViewingTools(self):
     return [get_files_content]
    
    def extract_classes_and_functions(self, file_path, limit_docstring=True, max_docstring_length=100):
        """
        Extract classes and functions from a Python file with optional docstring trimming.

        Args:
            file_path (str): Path to the Python file.
            limit_docstring (bool): Whether to limit the length of the docstring.
            max_docstring_length (int): Maximum length of the docstring to include.

        Returns:
            dict: A summary of classes and functions in the file.
        """
        with open(file_path, 'r', encoding='utf-8') as file:
            tree = ast.parse(file.read())

        def clean_docstring(doc):
            if not doc:
                return None
            # Trim, dedent, remove newlines, and limit to max length
            return " ".join(textwrap.dedent(doc).split()).strip()[:max_docstring_length]

        classes = []
        functions = []

        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.ClassDef):
                classes.append({
                    "name": node.name,
                    "doc": clean_docstring(ast.get_docstring(node)) if limit_docstring else None,
                })
            elif isinstance(node, ast.FunctionDef):
                functions.append({
                    "name": node.name,
                    "args": [arg.arg for arg in node.args.args],
                    "doc": clean_docstring(ast.get_docstring(node)) if limit_docstring else None,
                })

        return {"classes": classes, "functions": functions}
    
    def summarize_repo(self, root_dir, max_output_characters=5000, max_files=50, include_docstring=True, include_args=True):
        """
        Summarize a repository by extracting classes and functions from Python files with size constraints.

        Args:
            root_dir (str): Path to the root directory of the repository.
            max_output_characters (int): Maximum size of the summarized output.
            max_files (int): Maximum number of files to process.
            include_docstring (bool): Whether to include docstrings in the summary.
            include_args (bool): Whether to include function arguments in the summary.

        Returns:
            dict: A compact summary of the repository.
        """
        summary = {}
        file_count = 0
        total_output_size = 0

        for subdir, _, files in os.walk(root_dir):
            for file in files:
                if file.endswith(".py"):
                    file_path = os.path.join(subdir, file)
                    relative_path = os.path.relpath(file_path, root_dir)

                    # Extract data from the file
                    file_summary = self.extract_classes_and_functions(
                        file_path,
                        limit_docstring=include_docstring
                    )

                    # Optionally exclude arguments or docstrings
                    if not include_args:
                        for func in file_summary["functions"]:
                            func.pop("args", None)
                    if not include_docstring:
                        for item in file_summary["classes"] + file_summary["functions"]:
                            item.pop("doc", None)

                    # Convert to a compact string representation to estimate size
                    file_summary_str = repr({relative_path: file_summary})
                    file_size = len(file_summary_str)

                    # Check if adding this file exceeds the output size limit
                    if total_output_size + file_size > max_output_characters:
                        return summary

                    summary[relative_path] = file_summary
                    total_output_size += file_size
                    file_count += 1

                    if file_count >= max_files:
                        return summary
        return summary
    
    def reset_method_calls_counters(self):
        if hasattr(self, '_method_counts'):
            self._method_counts = {}

    def get_project_structure_from_scratch(self,repo_name, commit_id, instance_id):
        self.clone_repo(instance_id)
        self.checkout_commit("../../swe_repo/{repo_name}", commit_id)
        structure = self.create_structure(f"{repo_playground}/{repo_to_top_folder[repo_name]}")

        d = {
        "repo": repo_name,
        "base_commit": commit_id,
        "structure": structure,
        "instance_id": instance_id,
        }
        return d
        
    def create_structure(self, directory_path):
     """Create the structure of the repository directory by parsing Python files.
         :param directory_path: Path to the repository directory.
         :return: A dictionary representing the structure.
     """
     structure = {}

     for root, _, files in os.walk(directory_path):
        repo_name = os.path.basename(directory_path)
        relative_root = os.path.relpath(root, directory_path)
        if relative_root == ".":
            relative_root = repo_name
        curr_struct = structure
        for part in relative_root.split(os.sep):
            if part not in curr_struct:
                curr_struct[part] = {}
            curr_struct = curr_struct[part]
        for file_name in files:
            if file_name.endswith(".py"):
                file_path = os.path.join(root, file_name)
                class_info, function_names, file_lines = self.parse_python_file(file_path)
                curr_struct[file_name] = {
                    "classes": class_info,
                    "functions": function_names,
                    "text": file_lines,
                }
            else:
                curr_struct[file_name] = {}
     return structure   
 
    def parse_python_file(self,file_path, file_content=None):
     """Parse a Python file to extract class and function definitions with their line numbers.
     :param file_path: Path to the Python file.
     :return: Class names, function names, and file contents
     """
     if file_content is None:
        try:
            with open(file_path, "r") as file:
                file_content = file.read()
                parsed_data = ast.parse(file_content)
        except Exception as e:  # Catch all types of exceptions
            # print(f"Error in file {file_path}: {e}")
            return [], [], ""
     else:
        try:
            parsed_data = ast.parse(file_content)
        except Exception as e:  # Catch all types of exceptions
            # print(f"Error in file {file_path}: {e}")
            return [], [], ""

     class_info = []
     function_names = []
     class_methods = set()

     for node in ast.walk(parsed_data):
        if isinstance(node, ast.ClassDef):
            methods = []
            for n in node.body:
                if isinstance(n, ast.FunctionDef):
                    methods.append(
                        {
                            "name": n.name,
                            "start_line": n.lineno,
                            "end_line": n.end_lineno,
                            "text": file_content.splitlines()[
                                n.lineno - 1 : n.end_lineno
                            ],
                        }
                    )
                    class_methods.add(n.name)
            class_info.append(
                {
                    "name": node.name,
                    "start_line": node.lineno,
                    "end_line": node.end_lineno,
                    "text": file_content.splitlines()[
                        node.lineno - 1 : node.end_lineno
                    ],
                    "methods": methods,
                }
            )
        elif isinstance(node, ast.FunctionDef) and not isinstance(
            node, ast.AsyncFunctionDef
        ):
            if node.name not in class_methods:
                function_names.append(
                    {
                        "name": node.name,
                        "start_line": node.lineno,
                        "end_line": node.end_lineno,
                        "text": file_content.splitlines()[
                            node.lineno - 1 : node.end_lineno
                        ],
                    }
                )

     return class_info, function_names, file_content.splitlines()    


@tool
def ls() -> str:
    """
    This function lists the files in the current directory.
    :function: ls
    :return: a string containing the path of current directory and the files present in the current directory
    """
    ls_out = subprocess.run(f"cd {get_abs_current_dir()} & dir", shell=True, text=True, capture_output=True).stdout
    return f"""
    Current Directory: {SWEManager.current_dir}
    Files: 
    {ls_out}
    """

@tool
def goto_directory(path: str) -> str:
    """
    This function changes the current directory to the specified directory. This function must be used to goto a directory not to open a file.
    :function: goto_dir
    :param str path: path of the new directory you want to change relative to the current directory e.g 'matplotlib/doc'
    :return: output of command 'fail' or 'success'
    """

    if '..' in path:
        return 'use goto_previous_dir tool instead to go to previous directory'
    out = subprocess.run(f"cd {get_abs_current_dir()} & cd {path}", shell=True, text=True, capture_output=True).stdout
    if out == "":
        SWEManager.current_dir = f"{SWEManager.current_dir}{path}/"
        return 'successfully entered ' + SWEManager.current_dir
    else:
        return out

@tool
def goto_previous_dir() -> str:
    """
    This function takes the user to the previous directory.
    :function: goto_previous_dir
    :return: output 
    """
    if SWEManager.current_dir == '/':
        return "Already in top most directory. Can't go back anymore"
    else:
        paths = SWEManager.current_dir.split('/')[1:-1]
        paths.pop()
        SWEManager.current_dir = '/'
        for dir in paths:
            SWEManager.current_dir += dir + '/'
        return f"Current Directory: {SWEManager.current_dir}"

@tool
def get_current_dir() -> str:
    """
    This function returns the path of the currently opened directory.
    :function: get_current_dir
    :return: the current directory
    """
    return SWEManager.current_dir

def get_abs_current_dir() -> str:
    """
Constructs and returns the absolute path to the current directory.

The path is generated by joining the base project directory (`SWEManager.abs_project_dir`)
with the subdirectory 'test-repos' and a modified version of `SWEManager.directory_path` 
(with the first and last characters removed). 

Returns:
    str: The absolute path to the current directory.
    """
    return SWEManager.target_dir + SWEManager.current_dir

@tool
def number_of_lines(path: str) -> str:
    """
    This function takes a file path as input and returns the number of lines in the file.
    :function: number_of_lines
    :param path: The relative path to the file (e.g., 'lib/matplotlib/axis.py').
    
    :return: The number of lines in the file.
    """

    abs_file_path = os.path.join(get_abs_current_dir(), path) if path != "/" else get_abs_current_dir()
    if os.path.exists(abs_file_path):
        with open(abs_file_path, 'r') as file:
            return f"Number of lines in {SWEManager.current_dir+path}: {sum(1 for line in file)}"
    else:
        return f"File {SWEManager.current_dir+path} not found"

@tool
def open_file(path: str, line_number: int = 1, max_lines: int = 100) -> str:
    """   
    This function takes a file path, a line number, and a maximum number of lines as input and returns the contents of the file starting from the specified line number, limited to the maximum number of lines.
    :function: open_file
:param path: The relative path to the file (e.g., 'lib/matplotlib/axis.py').
    :param line_number: The line number from which to start reading the file. Defaults to 1.
:param max_lines: The maximum number of lines to return from the starting line. Defaults to 100.
    :return: A string containing the file contents from the specified starting line, limited to max_lines.
    """

    if line_number < 1:
        return "Error: line number cannot be zero or negative"
    if max_lines < 1:
        return "Error: max_lines cannot be zero or negative"

    abs_file_path = os.path.join(get_abs_current_dir(), path)
    if os.path.exists(abs_file_path):
        with open(abs_file_path, 'r') as file:
            with open(abs_file_path, 'r') as temp_file:
                num_lines = sum(1 for line in temp_file)
            if line_number > num_lines:
                return f"Can't access {line_number} line. This file only contains {num_lines} lines"
            
            out = f"Showing contents of File: {SWEManager.current_dir+path} starting from {line_number}\n\n"
            for n, line in enumerate(file, 1):
                if n >= line_number:
                    out += f"{n}: {line}\n"
                    if n == line_number + max_lines - 1:
                        break
            return out
    else:
        return path + " doesn't exist"

@tool
def find_files(file_name: str) -> str:
    """
    Searches the current directory and its subdirectories for the files that have name containing the specified file_name.
    :function: find_files
    :param file_name: The file_name to search for in file names.
    :return: paths of the files that contain the specified file_name in their names.
    """
    matched_files = []
    
    # Walk through the current directory and all subdirectories
    for root, dirs, files in os.walk(get_abs_current_dir()):
        for file in files:
            if file_name in file:
                matched_files.append(os.path.relpath(os.path.join(root, file), get_abs_current_dir()))
    
    return "Files found:\n" + "\n".join(matched_files)

@tool
def search_file(path: str, search_term: str) -> str:
    """
    This function takes a file path and a search term as input and returns the lines in the file that contain the search term. 
    :param str path: The relative path to the file (e.g., 'lib/matplotlib/axis.py').
    :param str search_term: The term to search for in the file.
    :return: A string containing the lines in the file that contain the search term.
    """

    abs_file_path = os.path.join(get_abs_current_dir(), path) if path != "/" else get_abs_current_dir()
    if os.path.exists(abs_file_path):
        with open(abs_file_path, 'r') as file:
            out = f"Searching for '{search_term}' in {SWEManager.current_dir+path}\n\n"
            for n, line in enumerate(file, 1):
                if search_term in line:
                    out += f"{n}: {line}\n"
            return out
    else:
        return f"File {SWEManager.current_dir+path} not found"

@tool
def search_dir(path: str, search_term: str) -> str:
    """
    Searches for files in the specified directory that contain the search term. It returns the file names and line numbers where the search term is found.

    :param path: The relative path to the directory (e.g., 'lib/matplotlib').
    :param search_term: The term to search for in the files in the directory.
    :return: A string containing the files in the directory that contain the search term.
    """
    
    abs_dir_path = os.path.join(get_abs_current_dir(), path) if path != "/" else get_abs_current_dir()
    
    if os.path.exists(abs_dir_path):
        matched = ""
        for root, dirs, files in os.walk(abs_dir_path):
            for file in files:
                file_path = abs_dir_path + '/' + file
                try:
                    with open(file_path, 'r', encoding='utf-8') as file_a:
                        for n, line in enumerate(file_a, 1):
                            if search_term in line:
                                matched += f"File: {path}/{file}, Line: {n}\n"
                                break
                except (UnicodeDecodeError, IOError):  # Handle decoding errors and file I/O errors
                    continue
        
        if matched:
            return "Files found:\n" + matched
        else:
            return "No files containing the search term were found."
    else:
        return f"Directory {abs_dir_path} not found."

@tool
def get_files_content(paths: List[str], line_numbers: List[int]) -> str:
    """
    This function takes a list of file paths and a list of line numbers as input and returns the contents of the files starting from the specified line numbers.
    :param paths: A list of relative paths to the files (e.g., ['lib/matplotlib/axis.py', 'lib/matplotlib/figure.py']). This function will return 50 lines for each file starting from the line_number provided.
    :param line_numbers: A list of line numbers from which to start reading the files.
    :return: A string containing the contents of the files from the specified starting line numbers.
    """
    out = ""
    for path, line_number in zip(paths, line_numbers):
        out += open_file.invoke({'path': path, 'line_number': line_number, 'max_lines': 50}) + '\n\n'
    return out


