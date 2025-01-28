import json
import inspect
import os
from typing import Any, Dict, Callable, List
import git
from langchain_community.tools import tool
from pydantic import BaseModel

abs_project_dir = os.getcwd()

class CurrentDirectory:
    def __init__(self):
        self.cwd = '\\'

directory = CurrentDirectory()

def getSearchTools():
    return [ls, goto_directory, goto_previous_dir, get_current_dir,
            number_of_lines, open_file, find_files, search_file, search_dir]

def getContentViewingTools():
    return [get_files_content]

@tool
def ls() -> str:
    """
    Lists the files in the current directory.
    :function: ls
    :return: A string containing the path of the current directory and the files present.
    """
    current_path = os.path.join(abs_project_dir, 'test-repos', directory.cwd.strip('\\'))
    try:
        items = os.listdir(current_path)
        return f"""
Current Directory: {directory.cwd}

Files:
{', '.join(items)}
"""
    except FileNotFoundError:
        return f"Directory {directory.cwd} not found."

@tool
def goto_directory(path: str) -> str:
    """
    Changes the current directory to the specified directory.
    :function: goto_directory
    :param str path: Path of the new directory relative to the current directory.
    :return: Output 'fail' or 'success'.
    """
    if '..' in path:
        return 'Use goto_previous_dir tool instead to go to the previous directory.'
    new_path = os.path.join(directory.cwd, path)
    abs_new_path = os.path.join(abs_project_dir, 'test-repos', new_path.strip('\\'))
    if os.path.isdir(abs_new_path):
        directory.cwd = new_path if new_path.endswith('\\') else new_path + '\\'
        return 'Successfully entered ' + directory.cwd
    else:
        return f"Directory {new_path} not found."

@tool
def goto_previous_dir() -> str:
    """
    Takes the user to the previous directory.
    :function: goto_previous_dir
    :return: Output indicating the new current directory.
    """
    if directory.cwd == '\\':
        return "Already in the topmost directory. Can't go back anymore."
    else:
        paths = directory.cwd.strip('\\').split('\\')
        paths.pop()
        directory.cwd = '\\' + '\\'.join(paths) + '\\' if paths else '\\'
        return f"Current Directory: {directory.cwd}"

@tool
def get_current_dir() -> str:
    """
    Returns the path of the currently opened directory.
    :function: get_current_dir
    :return: The current directory.
    """
    return directory.cwd

def get_abs_current_dir() -> str:
    return os.path.join(abs_project_dir, 'test-repos', directory.cwd.strip('\\'))

@tool
def number_of_lines(path: str) -> str:
    """
    Returns the number of lines in the given file.
    :function: number_of_lines
    :param path: The relative path to the file.
    :return: The number of lines in the file.
    """
    abs_file_path = os.path.join(get_abs_current_dir(), path)
    if os.path.exists(abs_file_path):
        with open(abs_file_path, 'r', encoding='utf-8') as file:
            num_lines = sum(1 for line in file)
            return f"Number of lines in {directory.cwd + path}: {num_lines}"
    else:
        return f"File {directory.cwd + path} not found."

@tool
def open_file(path: str, line_number: int = 1, max_lines: int = 100) -> str:
    """
    Returns the contents of the file starting from the specified line number.
    :function: open_file
    :param path: The relative path to the file.
    :param line_number: The line number from which to start reading the file. Defaults to 1.
    :param max_lines: The maximum number of lines to return. Defaults to 100.
    :return: A string containing the file contents.
    """
    if line_number < 1:
        return "Error: line number cannot be zero or negative."
    if max_lines < 1:
        return "Error: max_lines cannot be zero or negative."

    abs_file_path = os.path.join(get_abs_current_dir(), path)
    if os.path.exists(abs_file_path):
        with open(abs_file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
            total_lines = len(lines)
            if line_number > total_lines:
                return f"Cannot access line {line_number}. The file only contains {total_lines} lines."

            end_line = min(line_number + max_lines - 1, total_lines)
            content = ''.join(f"{i + 1}: {lines[i]}" for i in range(line_number - 1, end_line))
            return f"Showing contents of File: {directory.cwd + path} starting from line {line_number}\n\n{content}"
    else:
        return f"File {directory.cwd + path} doesn't exist."

@tool
def find_files(file_name: str) -> str:
    """
    Searches the current directory and its subdirectories for files containing the specified name.
    :function: find_files
    :param file_name: The file name to search for.
    :return: Paths of the files that contain the specified file_name.
    """
    matched_files = []
    current_path = get_abs_current_dir()
    for root, dirs, files in os.walk(current_path):
        for file in files:
            if file_name in file:
                relative_root = os.path.relpath(root, os.path.join(abs_project_dir, 'test-repos'))
                matched_files.append(os.path.join('\\' + relative_root, file))

    if matched_files:
        return "Files found:\n" + "\n".join(matched_files)
    else:
        return f"No files containing '{file_name}' were found."

@tool
def search_file(path: str, search_term: str) -> str:
    """
    Returns the lines in the file that contain the search term.
    :function: search_file
    :param path: The relative path to the file.
    :param search_term: The term to search for in the file.
    :return: A string containing the matching lines.
    """
    abs_file_path = os.path.join(get_abs_current_dir(), path)
    if os.path.exists(abs_file_path):
        matches = []
        with open(abs_file_path, 'r', encoding='utf-8') as file:
            for n, line in enumerate(file, 1):
                if search_term in line:
                    matches.append(f"{n}: {line.strip()}")
        if matches:
            return f"Searching for '{search_term}' in {directory.cwd + path}\n\n" + '\n'.join(matches)
        else:
            return f"No matches found for '{search_term}' in {directory.cwd + path}."
    else:
        return f"File {directory.cwd + path} not found."

@tool
def search_dir(path: str, search_term: str) -> str:
    """
    Searches for files in the specified directory that contain the search term.
    :function: search_dir
    :param path: The relative path to the directory.
    :param search_term: The term to search for in the files.
    :return: A string containing the files and line numbers where the search term is found.
    """
    abs_dir_path = os.path.join(get_abs_current_dir(), path)
    if os.path.exists(abs_dir_path):
        matched = ""
        for root, dirs, files in os.walk(abs_dir_path):
            for file in files:
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        for n, line in enumerate(f, 1):
                            if search_term in line:
                                relative_file_path = os.path.relpath(file_path, os.path.join(abs_project_dir, 'test-repos'))
                                matched += f"File: \\{relative_file_path}, Line: {n}\n"
                                break
                except (UnicodeDecodeError, IOError):
                    continue
        if matched:
            return "Files found:\n" + matched
        else:
            return f"No files containing '{search_term}' were found in {directory.cwd + path}."
    else:
        return f"Directory {directory.cwd + path} not found."

@tool
def get_files_content(paths: List[str], line_numbers: List[int]) -> str:
    """
    Returns the contents of the files starting from the specified line numbers.
    :function: get_files_content
    :param paths: A list of relative paths to the files.
    :param line_numbers: A list of line numbers from which to start reading the files.
    :return: A string containing the contents of the files.
    """
    out = ""
    for path, line_number in zip(paths, line_numbers):
        result = open_file(path, line_number, max_lines=50)
        out += result + '\n\n'
    return out

def setup_repo(data: dict):
    repo_name = data['repo'].split('/')[-1]
    directory.cwd = f"\\{repo_name}\\" 
    test_repos_path = os.path.join(abs_project_dir, 'test-repos')
    repo_path = os.path.join(test_repos_path, repo_name)
    if not os.path.exists(test_repos_path):
        os.mkdir(test_repos_path)
    if not os.path.exists(repo_path):
        git.Git(test_repos_path).clone(f"https://github.com/{data['repo']}.git")
    repo = git.Repo(repo_path)
    new_branch = data['instance_id']
    base_commit = data['base_commit']
    repo.git.checkout('-b', new_branch, base_commit)
    print('Repo setup complete')

if __name__ == '__main__':
    print(1)