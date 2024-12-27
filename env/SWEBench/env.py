import json, uuid
import os
import os.path
import subprocess
import ast
import textwrap
from typing import Any, Dict, List, Optional
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_community.tools import tool
from config import *
from env.env import Environment
import ast
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser

class SWEProblem(BaseModel):
    instance_id: str
    text: str
    repo: str
    base_commit: str
    problem_statement: str
    hints_text: str

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
   
    def get_state(self) -> str:
        with open(os.path.join(self.swe_temp_path, self.swe_data.instance_id, "state.json"), "r") as state_file:
            state = json.load(state_file)
        return json.dumps(state, indent=4)
    
    def step(
        self,
        code: str = "",
    ):
        return super().step(action_code=code, context={'problem': self.swe_data, 'bot': self.swe_manager, 'env': self, 'results': None})
        
    def get_score(self, code: str) -> float:
        class ScoreResponse(BaseModel):
            score: float = Field(..., description="The score of the function between 0 and 1.")

        # Initialize the output parser
        output_parser = PydanticOutputParser(pydantic_object=ScoreResponse)

        # Define the LLM and prompt
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        prompt_template = ChatPromptTemplate.from_template(
            template="""
You are an AI reviewer. You analyze whether a function solves a problem. You are not strict and analyze keeping in mind that these functions are generated by AI not human.
This is a function that is generated from AI. Please provide a score value between 0 and 1 for whether the function does what it is meant to by analyzing the current state of the solution.

*Generated Function*:
{code}

*Current Solution State After Running the Function*:
{state}


Only output a score value between 0.0 and 1.0. If the function does not do what it is meant to, please provide a score of 0.0. If the function does what it is meant to, please provide a score of 1.0. If the function is partially correct, provide a score between 0.0 and 1.0. Don't be strict and Be lenient in scoring. Provide output according to this pydanctic model:
```json
{{
    "score": 0.0
}}
```

""")
        # Create the prompt
        prompt = prompt_template.format_messages(code=code, state=self.get_state())
        response = llm(prompt)
        structured_output = output_parser.parse(response.content)
        print("Score: ", structured_output.score)
        return {'score (top:1, worst:0)': structured_output.score}

class SWEManager:
    current_dir: str = "/" # The current directory within which tools are navigating in the problem repository
    target_dir: str = ""  # The root directory for the current problem repository
    
    def getSearchTools(self):
     return [ls, goto_directory, goto_previous_dir, get_current_dir, number_of_lines, open_file, find_files, search_file, search_dir]

    def getContentViewingTools(self):
     return [get_files_content]
    
    def getEditingTools(self):
        return [edit_lines_in_file, create_file]
    
    def getDiff(self) -> str:
        return subprocess.run(f"cd {SWEManager.target_dir} & git diff", shell=True, text=True, capture_output=True).stdout
    
    def resetCurrentDir(self):
        SWEManager.current_dir = "/"
    
    def restoreRepo(self) -> str:
        return subprocess.run(f"cd {SWEManager.target_dir} & git restore .", shell=True, text=True, capture_output=True).stdout
    
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
    
    def summarize_repo(self, root_dir, max_output_characters=20000, max_files=50, include_docstring=True, include_args=True) -> dict:
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
    if path[0] == '/':
        path = path[1:]

    abs_file_path = os.path.join(get_abs_current_dir(), path)
    if os.path.exists(abs_file_path):
        with open(abs_file_path, 'r', encoding='utf-8', errors='ignore') as file:
            with open(abs_file_path, 'r') as temp_file:
                num_lines = sum(1 for line in temp_file)
            if line_number > num_lines:
                return f"Can't access {line_number} line. This file only contains {num_lines} lines"
            
            out = f"Showing contents of File: {SWEManager.current_dir+path} starting from {line_number}\n"
            for n, line in enumerate(file, 1):
                if n >= line_number:
                    out += f"{n}: {line}\n"
                    if n == line_number + max_lines - 1:
                        break
            return out
    else:
        return path + " doesn't exist"

@tool
def edit_lines_in_file(file_path, start_line_number, num_lines_to_replace, replacement_text):
    """
    Replaces a specific number of lines starting from a given line number in a file
    with the given replacement text. Python files are validated for syntax errors after the edit.
    If an error is found, the edit will not be executed. Reading the error message and
    modifying your command is recommended as issuing the same command will return the same
    error.

    Args:
        file_path (str): Path to the file to edit.
        start_line_number (int): Start line number (1-indexed).
        num_lines_to_replace (int): Number of lines to replace. It must match the number of lines that are in your replacement text
        replacement_text (str): The replacement text to insert.

    Returns:
        str: Success or error message.
    """

    file_path = os.path.join(get_abs_current_dir(), file_path)
    try:
        # Read the file content
        with open(file_path, 'r') as file:
            lines = file.readlines()

        # Validate line range
        if not (1 <= start_line_number <= len(lines)):
            return f"Error: Start line {start_line_number} is out of bounds."
        
        end_line_number = start_line_number + num_lines_to_replace - 1
        if end_line_number > len(lines):
            return f"Error: The range from line {start_line_number} to {end_line_number} exceeds the file length."

        # Replace the specified lines
        replacement_lines = [line + '\n' for line in replacement_text.splitlines()]
        new_lines = lines[:start_line_number - 1] + replacement_lines + lines[end_line_number:]

        # Check for syntax errors if it's a Python file
        if file_path.endswith('.py'):
            try:
                temp_code = ''.join(new_lines)
                ast.parse(temp_code)
            except SyntaxError as e:
                return f"Syntax Error: {e}"

        # Write the updated content back to the file
        with open(file_path, 'w') as file:
            file.writelines(new_lines)

        return f"Successfully replaced {num_lines_to_replace} lines starting from line {start_line_number} in {file_path}."
    except Exception as e:
        return f"Error: {e}\nPlease fix this error before trying another edit. if you are getting this error multiple time, please skip this part of the patch"

@tool
def create_file(path: str, content: str) -> str:
    """
    This function creates a new file at the specified path with the provided content.
    Ensures the necessary directories exist before creating the file.
    
    :function: create_file
    :param path: The relative path to the new file (e.g., 'lib/matplotlib/new_file.py').
    :param content: The content to write into the new file. It can span multiple lines.
    :return: A success message or an error message in case of failure.
    """
    abs_file_path = os.path.join(get_abs_current_dir(), path)
    directory = os.path.dirname(abs_file_path)
    
    try:
        # Ensure the directory exists
        if not os.path.exists(directory):
            os.makedirs(directory)
        
        # Write the content to the new file
        with open(abs_file_path, 'w') as file:
            file.write(content)
        
        return f"Successfully created file: {path}"
    except Exception as e:
        return f"Error: Could not create file '{path}'. Details: {str(e)}"

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

    :param path: The relative path to the directory (e.g., 'lib/matplotlib'). if you want to search in the current directory, use "".
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
                                matched += f"File: {SWEManager.current_dir+path}/{file}, Line: {n}\n"
                                break
                except (UnicodeDecodeError, IOError):  # Handle decoding errors and file I/O errors
                    continue
        
        if matched:
            return f"Files found:\n" + matched
        else:
            return "No files containing the search term were found."
    else:
        return f"Directory {path} not found."

@tool
def get_files_content(paths: List[str], line_numbers: List[int], num_lines: List[int]) -> str:
    """
    This function takes a list of file paths and a list of line numbers as input and returns the contents of the files starting from the specified line numbers.
    :param paths: A list of relative paths to the files (e.g., ['lib/matplotlib/axis.py', 'lib/matplotlib/figure.py']). This function will return 100 lines for each file starting from the line_number provided.
    :param line_numbers: A list of line numbers from which to start reading the files.
    :param num_lines: A list of the maximum number of lines to return from the starting line.
    :return: A string containing the contents of the files from the specified starting line numbers.
    """
    SWEManager().resetCurrentDir()
    out = ""
    for path, line_number, n in zip(paths, line_numbers, num_lines):
        out += open_file.invoke({'path': path, 'line_number': line_number, 'max_lines': n}) + '\n\n'
    return out


