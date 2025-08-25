import re, time, json, os, glob, sys, logging
import subprocess, asyncio, inspect
from openai import BadRequestError
import websockets, socket, requests
from datetime import datetime
from typing import List, Optional, Union, Dict, Any
from dataclasses import dataclass, field
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry  # type: ignore

from langchain_community.embeddings import HuggingFaceEmbeddings, OpenAIEmbeddings
from langchain_core.runnables import RunnableSequence, ConfigurableField, Runnable
from langchain_core.messages.human import HumanMessage
from langchain_core.messages.ai import AIMessage
from langchain_core.messages.system import SystemMessage
from langchain_core.messages.function import FunctionMessage
from langchain_openai import ChatOpenAI
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

import tkinter as tk
from tkinter import scrolledtext
from utils.file_utils import dump_text, f_exists, f_move
# from config import MODELS_CONFIG_LIST, vector_store_type
from config import MODELS_CONFIG_LIST
import config
from requests.auth import HTTPBasicAuth

import regex as regex 

import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from utils.db_utils import (UnifiedVectorDB as _DB_Impl, UnifiedVectorDBConfig as _DBConfig_Impl, ElasticSearchDB_Config as _ESConfig_Impl, CHROMA_DATABASE, ELASTIC_DATABASE)

# (1) Re‐export the classes under the old names to avoid breaking changes
UnifiedVectorDB        = _DB_Impl
UnifiedVectorDBConfig  = _DBConfig_Impl
ElasticSearchDB_Config = _ESConfig_Impl

# (2) Subclass to “add old” get_unique_id(...):
class _UnifiedVectorDB_with_get_id(_DB_Impl):
    def get_unique_id(self) -> str:
        from utils.human_llm_config import HumanLLMConfig

        single_cfg = HumanLLMConfig().common_vectordb_config
        if single_cfg.unique_collection_id is None:
            single_cfg.unique_collection_id = os.environ.get(
                'unique_id',
                f"{socket.gethostname()}_{datetime.now().strftime('%d-%m-%Y-%H-%M-%S')}"
            )
        self.config.unique_collection_id = single_cfg.unique_collection_id
        return self.config.unique_collection_id

# (3) Use the subclass as the new UnifiedVectorDB
UnifiedVectorDB = _UnifiedVectorDB_with_get_id

class ExtractMessage(Runnable):
    """Runnable to extract and concatenate message content."""

    def invoke(self, input_msg, config):
        """Extract and concatenate message content from input."""
        return "\n".join(
            getattr(message, 'content', message)
            for message in getattr(input_msg, 'messages', input_msg)
        )
    

class InferenceCheck:
    """Represents an inference check with a name and function."""

    def __init__(self, check_name, check_function):
        """Initialize InferenceCheck."""
        self.check_name = check_name
        self.check_function = check_function

    def run_check(self, output_id, *args, **kwargs):
        """Run the inference check."""
        return self.check_function(*args, **kwargs, output_id=output_id)


class PrintPromptRunnable(Runnable):
    """Runnable to print the formatted prompt in red."""

    def invoke(self, input_msg, config):
        """Print the formatted prompt in red."""
        smart_print(f"PrintPromptRunnable type of input_msg: {type(input_msg)}")
        formatted_prompt = format_prompt(
            input_msg if isinstance(input_msg, list) else input_msg.messages)
        smart_print("\033[31m" + formatted_prompt + "\033[0m")
        return input_msg



class LLMConfig:
    """Configuration for Language Model (LLM) settings."""

    default_llm: Optional[Any] = None
    premium_llm: Optional[Any] = None
    llm_list: List[Any] = []
    temperature_min: float = 0.1
    temperature_max: float = 0.7
    max_context_size: int = 4096
    premium_by_default: bool = False

    def __init__(
        self,
        llm_list: Optional[List[Any]] = None,
        max_context_size: int = 4096,
        premium_by_default: bool = False
    ):
        """Initialize LLMConfig."""
        self.llm_list = llm_list or []
        self.max_context_size = max_context_size
        self.premium_by_default = premium_by_default


class UserSession:
    """Represents a user session with associated data."""

    def __init__(self, user_id: Optional[str] = None):
        """Initialize UserSession."""
        self.user_id = user_id
        self.user_message = ""
        self.user_message_few_shots = []
        self.current_inference_context = None
        self.selected_outputs = []
        self.previous_templates = []
        self.previous_results = []
        self.temp_inference_result_content = ""
        self.saved_task:Optional[Any] = None

    def get_user_id(self):
        """Retrieve or prompt for the user ID."""
        if self.user_id is None:
            # search user_id in env, then in imported config else None
            self.user_id = os.environ.get('user_id', config.user_id if hasattr(config, 'user_id') else None)
            if self.user_id is None:
                self.user_id = smart_input("Please enter your user id: ", "Learning Loop", message_type="USER_ID")
        return self.user_id
    
    def set_user_id(self, id):
        self.user_id = id


class InferenceTracking:
    """Tracks inference-related metrics and checks."""

    def __init__(self):
        """Initialize InferenceTracking."""
        self.option_times: Dict[str, float] = {}
        self.option_counts: Dict[str, int] = {}
        self.before_inference_option_times = []
        self.before_inference_option_counts = []
        self.after_inference_option_times = []
        self.after_inference_option_counts = []
        self.unidentified_option_times = []
        self.unidentified_option_counts = []
        self.last_inference_check_results: Dict[str, Any] = {}
        self.inference_checks: Dict[str, Any] = {}
        self.excluded_inference_checks:List[str] = ["Recommend Critics"]


class TaskHistory:
    """Tracks completed, failed, and ongoing tasks."""

    def __init__(self):
        """Initialize TaskHistory."""
        self.completed_tasks: List[Dict[str, Any]] = []
        self.failed_tasks: List[Dict[str, Any]] = []
        self.task_log: List[Dict[str, Any]] = []

    def clear_completed_tasks(self):
        self.completed_tasks.clear()

    def clear_failed_tasks(self):
        self.failed_tasks.clear()

    def add_completed_task(self, task_entry: Optional[Dict[str, str]] = None):
        """Logs a successfully completed task."""
        self.completed_tasks.append(task_entry)
        self.task_log.append(task_entry)

    def add_failed_task(self, task_entry: Optional[Dict[str, str]] = None):
        """Logs a task that failed during execution."""
        self.failed_tasks.append(task_entry)
        self.task_log.append(task_entry)

    def get_completed_tasks(self) -> List[Dict[str, str]]:
        """Returns all completed tasks."""
        return self.completed_tasks

    def get_failed_tasks(self) -> List[Dict[str, str]]:
        """Returns all failed tasks."""
        return self.failed_tasks

def calculate_text_similarity(text1: str, text2: str, method="difflib") -> float:
    """
    Calculate similarity between two texts using difflib.
    Returns a value between 0 and 1, where 1 is identical.
    """
    try:
        if method == "tdfidf":
            vec = TfidfVectorizer().fit_transform([text1, text2])
            return cosine_similarity(vec[0:1], vec[1:2])[0][0]
        else: # "difflib"
            return difflib.SequenceMatcher(None, text1.strip(), text2.strip()).ratio()
    except Exception:
        # Fallback: simple length-based similarity
        len1, len2 = len(text1), len(text2)
        if len1 == 0 and len2 == 0:
            return 1.0
        return 1.0 - abs(len1 - len2) / max(len1, len2, 1)

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

def create_Nmajority_chain(
        num_models=3,
        map_model_name=None,
        reduce_model_name=None,
        map_temperature=0.7,
        reduce_temperature=0.
    ):
    # Initialize the OpenAI models
    if map_model_name is None:
        map_model_name = (
            MODELS_CONFIG_LIST["basic_gpt" if "basic_gpt" in MODELS_CONFIG_LIST else "gpt"]
        )
    if reduce_model_name is None:
        reduce_model_name = (
            MODELS_CONFIG_LIST["smart_gpt" if "smart_gpt" in MODELS_CONFIG_LIST else "gpt"]
        )

    # Ensure if we use GPT model or not
    if "gpt" in map_model_name:
        models = [
            ChatOpenAI(
                model_name=map_model_name,
                temperature=map_temperature,
                cache=False
            ) for _ in range(num_models)
        ]
        final_model = ChatOpenAI(
            model_name=reduce_model_name,
            temperature=reduce_temperature,
            cache=False
        )
    else:
        models = [
            ChatOllama(
                model=map_model_name,
                temperature=map_temperature,
                cache=False
            ) for _ in range(num_models)
        ]
        final_model = ChatOllama(
            model=reduce_model_name,
            temperature=reduce_temperature,
            cache=False
        )

    # Define the chain using LCEL
    response_keys = [f"response_{i + 1}" for i in range(num_models)]
    multi_reponse = {key: model for key, model in zip(response_keys, models)}
    multi_reponse["cleaned_input"] = ExtractMessage()

    final_prompt_template_str = (
        f"Given the following responses below and the initial question below, "
        f"provide an optimal response to the question mixing best elements of each "
        f"and following the same answer output structure:\n\n"
        "Initial question:\n{cleaned_input}\n\n" +
        "\n\n".join(
            [
                f"Response {i + 1}:\n{{{response_keys[i]}}}" 
                for i in range(num_models)
            ]
        ) +  # Access content directly
        "\n\nOptimal Response:\n"
    )

    # Insert the PrintPromptRunnable just to print the prompt for control
    print_prompt_runnable = PrintPromptRunnable()

    # Print in RED the final prompt template: print("\033[31m"+final_prompt_template_str+"\033[0m")
    final_prompt_template = ChatPromptTemplate.from_template(final_prompt_template_str)

    # No ERROR but bad output: "I'm sorry, but I cannot fulfill this request" or "I'm sorry, but I cannot fulfill this request as it is too complex for me to process." or "I'm sorry, but I cannot fulfill this request as it involves creating a Python function and providing a specific response format." or "I'm sorry, but I cannot fulfill this request as it requires a level of understanding and reasoning that is beyond my current capabilities."
    chain = multi_reponse | final_prompt_template | print_prompt_runnable | final_model

    return chain

def smart_print(
        message: str,
        agent_name=None,
        message_type=None,
        append=False,
        column_id=None,
        column_max=None,
        optional=False
    ):
    logger = logging.getLogger(__name__)
    from utils.human_llm import HumanLLM, HumanLLMConfig
    global AgentDisplayManager, AGENT
    if 'IN_NOTEBOOK' not in globals():
        try:  # test if IN_NOTEBOOK
            from IPython import get_ipython
            globals()['IN_NOTEBOOK'] = IN_NOTEBOOK = get_ipython().__class__.__name__ == 'ZMQInteractiveShell'
            logger.info("Smart_print: Notebook mode = " + str(IN_NOTEBOOK))
        except:
            globals()['IN_NOTEBOOK'] = IN_NOTEBOOK = False
    else:
        IN_NOTEBOOK = globals()['IN_NOTEBOOK']

    if 'IN_WEBSOCKET' not in globals():
        if HumanLLMConfig().use_websocket:
            globals()['IN_WEBSOCKET'] = IN_WEBSOCKET = True
        else:
            globals()['IN_WEBSOCKET'] = IN_WEBSOCKET = False
    else:
        IN_WEBSOCKET = globals()['IN_WEBSOCKET']

    if IN_WEBSOCKET:
        # Ensure websocket server is not none
        if HumanLLMConfig().ws_server is None:
            logger.info("WebSocket server not initialized, initializing...")
            HumanLLMConfig().init_ws_server()
            logger.info("WebSocket server initialized.")

        # Check if in the message there are no unexpected non-whitespace characters
        message_str = str(message) if not type(message) == str else message
        if regex.search(r'[^\P{C}\t\n\r]', message_str):           
            # Remove unexpected characters
            message_str = regex.sub(r'[^\P{C}\t\n\r]', "", message_str)
        message_dict = {'message': message_str, 'agent_name': agent_name, 'message_type': message_type, 'append': append,
                        'column_id': column_id, 'column_max': column_max, 'optional': optional,
                        'step_id': HumanLLMConfig().step_id}
        # convert message_dict to json
        message = json.dumps(message_dict)
        time.sleep(0.05)
        # Wait a second every 500 messages to avoid flooding the WebSocket server
        if HumanLLMConfig().ws_server.message_count % 500 == 0:
            time.sleep(0.5)
        HumanLLMConfig().ws_server.send_message(message)
    else:
        if append:
            print(message, end="", flush=True)
        else:
            print(message)

def smart_input(message: str, agent_name=None, message_type=None, column_id=None, column_max=None, optional=False):
    logger = logging.getLogger(__name__)
    from utils.human_llm import HumanLLMConfig

    # Determine if using WebSocket
    if 'IN_WEBSOCKET' not in globals():
        if HumanLLMConfig().use_websocket:
            globals()['IN_WEBSOCKET'] = IN_WEBSOCKET = True
        else:
            globals()['IN_WEBSOCKET'] = IN_WEBSOCKET = False
    else:
        IN_WEBSOCKET = globals()['IN_WEBSOCKET']

    if IN_WEBSOCKET:
        # # Ensure WebSocket server is ready and has clients
        # if not HumanLLMMonitor.websocket_server.connected_clients:
        #     print("No WebSocket clients connected. Waiting for clients...")
        #     while not HumanLLMMonitor.websocket_server.connected_clients:
        #         time.sleep(1)

        # Ensure websocket server is not none
        if HumanLLMConfig().ws_server is None:
            logger.info("WebSocket server not initialized, initializing...")
            HumanLLMConfig().initialize_websocket_server()
            logger.info("WebSocket server initialized.")

        # # Retrieve port and secret from WebsocketServer
        port = HumanLLMConfig().ws_server.port
        secret = HumanLLMConfig().ws_server.secret
        ws_url = f"ws://localhost:{port}"
        if secret:
            ws_url += f"?secret={secret}"

        # Construct the structured message
        structured_message = {
            'message': message,
            'agent_name': agent_name,
            'message_type': message_type,
            'column_id': column_id,
            'column_max': column_max,
            'input': True,
            'optional': optional,
            'step_id': HumanLLMConfig().step_id
        }
        message_json = json.dumps(structured_message)

        # Send the message via WebsocketServer
        HumanLLMConfig().ws_server.send_message(message_json)

        async def receive_message(timeout=86400):
            async with websockets.connect(f"{ws_url}?self=true", ping_interval=30, ping_timeout=60) as websocket:
                try:
                    logger.info("SMART INPUT Waiting for response from WebSocket")
                    while True:
                        max_retry = 10
                        response = None
                        for i in range(max_retry):
                            try:
                                response = await asyncio.wait_for(websocket.recv(), timeout=timeout)
                                break
                            except asyncio.TimeoutError:
                                logger.info(f"Timeout waiting for response from WebSocket, retry {i + 1}/{max_retry}")
                                await asyncio.sleep(1)
                            except Exception as e:
                                logger.info(f"An error occurred: {e}, retry {i + 1}/{max_retry}")
                                await asyncio.sleep(1)
                        logger.info("SMART INPUT Received response from WebSocket")
                        try:
                            # Check that the response is not a NoneType
                            if response is None:
                                logger.info("Received None response from WebSocket")
                                return None
                            json_data = json.loads(response)
                        except json.JSONDecodeError as e:
                            logger.error(f"Error decoding JSON: {e}")
                            return None
                        # Ignore messages from self
                        if 'sender_id' in json_data and json_data['sender_id'] == HumanLLMConfig().ws_server.server_id:
                            logger.error("Received message from self, ignoring")
                            continue

                        # Handle function results
                        if 'result' in json_data:
                            logger.info("Received function result, ignoring")
                            json_data['sender_id'] = HumanLLMConfig().ws_server.server_id
                            response = json.dumps(json_data)
                            HumanLLMConfig().ws_server.send_notasync_message(response)
                            continue
                        else:
                            break
                    if 'message' in json_data:
                        answer = json_data['message']
                    elif 'result' in json_data:
                        answer = json_data['result']
                    else:
                        raise ValueError("No 'message' or 'result' in WebSocket response")
                    logger.info(f"SMART INPUT Answer: {answer}")
                    return str(answer).upper()
                except asyncio.TimeoutError:
                    logger.error("Timeout waiting for response from WebSocket")
                    return None

        try:
            # Attempt to get the current running event loop
            loop = asyncio.get_running_loop()
            # If a loop is running, schedule the receive_message coroutine
            future = asyncio.run_coroutine_threadsafe(receive_message(), loop=loop)
            return future.result()
        except RuntimeError:
            # No running loop, create a new event loop and run the coroutine
            return asyncio.run(receive_message())
    else:
        return input(message)

def is_vscode_installed():
    try:
        # Try to get the version of VSCode, which will confirm if it's installed
        subprocess.run(["code", "--version"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return True
    except subprocess.CalledProcessError:
        # CalledProcessError means the command was not successful
        return False
    except FileNotFoundError:
        # FileNotFoundError means the code command is not in the PATH
        return False

def _visual_input(initial_string="", filetype="md", agent_name=None, column_id=None, column_max=None, message_type=None):
    """
    Open a visual editor (VSCode or Tkinter) to interactively edit a given string or list.

    Args:
    - initial_string (str or list): The string or list to be edited.
    - filetype (str): The file extension to use when editing with VSCode.

    Returns:
    - str or list: The edited string or list.
    """
    if is_vscode_installed():
        # Step 1: Convert initial_string to a string representation
        if isinstance(initial_string, list):
            # Serialize the list to JSON format
            initial_string_serialized = json.dumps(initial_string, indent=4)
            # Default filetype to json if not specified
            if filetype == "md":
                filetype = "json"
        else:
            initial_string_serialized = initial_string if initial_string is not None else ""

        # Step 2: Generate the code and save it to a file
        if not os.path.exists('temp/edition'):
            os.makedirs('temp/edition')
        file_path = 'temp/edition/' + str(datetime.now().timestamp()) + f".{filetype}"
        with open(file_path, 'w') as file:
            file.write(initial_string_serialized)

        smart_print(f"Please edit and save the file opened in VSCode (Ctrl + W) when ready ({file_path})", optional=False, agent_name=agent_name, column_id=column_id, column_max=column_id, message_type=message_type)
        # Step 3: Open the file in VSCode
        subprocess.run(["code", "--wait", file_path])

        # Step 4: Read the edited content
        with open(file_path, 'r') as file:
            edited_string = file.read()
        # Delete the file
        os.remove(file_path)

        # Step 5: Convert back to the appropriate type
        if isinstance(initial_string, list):
            try:
                edited_data = json.loads(edited_string)
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON: {e}")
                edited_data = initial_string  # Fallback to original data
            return edited_data
        else:
            return edited_string

    else:
        # Tkinter implementation
        def on_close():
            """Function to execute when the window is closed."""
            nonlocal edited_string
            edited_string = txt_edit.get(1.0, tk.END).strip()
            root.destroy()

        def copy(event):
            root.clipboard_clear()
            text = txt_edit.get("sel.first", "sel.last")
            root.clipboard_append(text)

        def paste(event):
            text = root.clipboard_get()
            txt_edit.insert(tk.INSERT, text)
            return "break"

        # Create a new Tkinter root window
        root = tk.Tk()
        root.title("Edit Input")

        # Create a scrolled text widget
        txt_edit = scrolledtext.ScrolledText(root, wrap=tk.WORD, width=80, height=20)
        txt_edit.pack(padx=10, pady=10, expand=True, fill=tk.BOTH)

        # Insert the initial string into the text widget
        if isinstance(initial_string, list):
            initial_string_serialized = json.dumps(initial_string, indent=4)
        else:
            initial_string_serialized = initial_string
        txt_edit.insert(tk.END, initial_string_serialized)

        # Bindings and configurations
        txt_edit.bind("<Control-c>", copy)
        txt_edit.bind("<Control-v>", paste)
        txt_edit["undo"] = True
        txt_edit.bind("<Control-z>", lambda event: txt_edit.edit_undo())
        txt_edit.bind("<Control-y>", lambda event: txt_edit.edit_redo())

        # Bind the window's close event
        root.protocol("WM_DELETE_WINDOW", on_close)

        # Edited string variable
        edited_string = initial_string_serialized

        # Focus on the text widget and start the main loop
        txt_edit.focus_set()
        root.mainloop()

        # Convert back to the appropriate type
        if isinstance(initial_string, list):
            try:
                edited_data = json.loads(edited_string)
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON: {e}")
                edited_data = initial_string  # Fallback to original data
            return edited_data
        else:
            return edited_string

def list_prompt_variants(prompt_name, package_path="."):
    base_name = prompt_name.split("@")[0]
    pattern = f"{package_path}/prompts/{base_name}@*.txt"
    variants = [filename[len(package_path) + 9:-4] for filename in glob.glob(pattern)]
    return [base_name] + variants  # Include base prompt in the list

def save_prompt(prompt_name, text, package_path="."):
    prompt_file_path_name = f"{package_path}/prompts/{prompt_name}.txt"

    # if prompt_file_path_name exists, move existing file to prompt_file_path_name.timestamp (timestamp = datetime.now().isoformat())
    if f_exists(prompt_file_path_name):
        moved_file_path_name = prompt_file_path_name + datetime.now().strftime(".%H-%M-%S_%m-%d-%y")
        smart_print(f"Moving existing prompt file {prompt_file_path_name} to {moved_file_path_name}", "save_prompt", optional=True)
        f_move(prompt_file_path_name, moved_file_path_name)

    smart_print(f"Saving new prompt file {prompt_file_path_name}", "save_prompt", optional=True)

    return dump_text(text, prompt_file_path_name)

def save_prompt_with_tag(prompt_name, text, new_tag, package_path="."):
    # Extract base prompt name and current tag
    parts = prompt_name.split("@")
    base_name = parts[0]
    current_tag = "@".join(parts[1:]) if len(parts) > 1 else ""

    # Determine the file name to save
    if new_tag:
        if current_tag:
            prompt_name = prompt_name.replace(f"@{current_tag}", f"@{new_tag}")
        else:
            prompt_name = f"{prompt_name}@{new_tag}"
    prompt_file_path_name = f"{package_path}/prompts/{prompt_name}.txt"

    # Backup existing file
    if f_exists(prompt_file_path_name):
        moved_file_path_name = prompt_file_path_name + datetime.now().strftime(".%Y-%m-%d_%H-%M-%S")
        smart_print(f"Moving existing prompt file {prompt_file_path_name} to {moved_file_path_name}",
                    "save_prompt_with_tag", optional=True)
        f_move(prompt_file_path_name, moved_file_path_name)

    smart_print(f"Saving new prompt file {prompt_file_path_name}", "save_prompt_with_tag", optional=True)

    # Save the file
    return dump_text(text, prompt_file_path_name)

def apply_criteria_and_prepare_monitor_args(agent, special_criteria, available_locals=None):
    """
    Apply special criteria to the attributes and parameters of an agent
    and prepare arguments for initializing HumanLLMMonitor.

    :param agent: The agent instance to modify.
    :param special_criteria: Dictionary containing the special criteria.
    :param available_locals: Dictionary containing the local variables of the caller function.
    :return: Dictionary of keyword arguments for HumanLLMMonitor.
    """
    if available_locals is None:
        # Use inspect to dynamically capture arguments
        frame = inspect.currentframe().f_back  # Go up one level
        _, _, _, values = inspect.getargvalues(frame)
        available_locals = values

    new_params = {}
    if special_criteria:
        class_name = agent.__class__.__name__
        # Apply special criteria to agent attributes and local parameters
        for key, value in special_criteria.items():
            print(f"Special criteria: {key} = {value}")
            if key in ['self', 'special_criteria']:
                continue
            if '#' in key:
                agent_name, key = key.split('#', 1)
                if agent_name != class_name and agent_name not in ['all', '']:
                    continue
            if hasattr(agent, key):
                setattr(agent, key, value)
                print(f"Special criteria applied to {agent}'s class property: {key} = {value}")
            elif key in available_locals:
            #else:
                new_params[key] = value
                print(f"Special criteria applicable to {agent}'s local variables: {key} = {value}")

    # Prepare HumanLLMMonitor arguments
    from utils.human_llm import HumanLLM
    HumanLLMM_args = set(inspect.signature(HumanLLM.__init__).parameters) - {'self'}
    # print(HumanLLMM_args)
    kw_common_args = {param: available_locals[param] for param in HumanLLMM_args if param in available_locals}
    kw_common_args.update(new_params)
    # print(kw_common_args)

    return kw_common_args

def get_primitives(path_folder: str) -> List[str]:
    """
    Loads and returns the content of all Python files in the specified folder.

    Args:
        path_folder (str): The path to the folder containing the primitive files.

    Returns:
        List[str]: A list of strings, each containing the content of a Python file.
    """
    if path_folder is None:
        return []
    
    primitives = []
    folder_path = os.path.join(os.path.dirname(__name__), path_folder)

    for file in os.listdir(folder_path):
        if file.endswith(".py"):
            #smart_print(f"File load: {file}", "Shared Utility", "Files loaded", optional=True)
            print(f"File load: {file} - Shared Utility")
            file_path = os.path.join(folder_path, file)
            with open(file_path, "r") as f:
                primitives.append(f.read())

    # Optional: Redirect stdout to a file for debugging purposes
    with open('primitives_checking.txt', 'w') as file:
        sys.stdout = file  # Redirect standard output to the file
        # Print something if needed
        sys.stdout = sys.__stdout__  # Restore stdout

    return primitives

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

def import_functions_from_directory(regex=".*", functions_path="functions"):
    """
    Imports functions from files in the functions directory that match the given regex pattern,
    excluding directories.
    """
    functions = {}
    for file in os.listdir(functions_path):
        file_path = os.path.join(functions_path, file)
        if os.path.isfile(file_path) and re.match(regex, file):
            with open(file_path, "r") as f:
                code = f.read()
                functions[file.replace(".py", "")] = code
    return functions

def get_user_continue_input():
    answer = smart_input(
        'Do you want to:\n'
        '- search for a new task after reseting to empty documents (Y/YES) ?\n'
        '- search for a new task based based on the status of documents after applying '
        'the task you just validated (N/NO/Enter) ?\n'
        '- or just exit the program (E/EXIT) ?',
        'orchestrate_agents',
        message_type='VALIDATION_INFO'
    ).strip().upper()

    return False if answer in ['E', 'EXIT'] else True

def parse_learn_question(question):
    match = re.match(r"learn\s+(\w+)(?:\s+(.*))?", question, re.IGNORECASE)
    if not match:
        raise ValueError("Invalid question format. Expected format: 'learn <problem_type> <instance_ids>'")
    problem_type, instance_ids_raw = match.groups()
    instance_ids = instance_ids_raw.split() if instance_ids_raw else []
    return problem_type.lower(), instance_ids

def calculate_total_score(
        scores,
        weight_no_runtime_error: float = 1.0,
        weight_best_score: float = 10.0,
        weight_validated_scores: float = 20.0
    ):
    """Calculate total score from the scores dictionary with configurable weights"""
    validated_score_avg = 0
    if scores.get('validated_scores') is not None and len(scores['validated_scores']) > 0:  # Check if the list is not empty
        for dic in scores['validated_scores']:
            validated_score_avg += sum(dic.values())
        validated_score_avg /= len(scores['validated_scores'])  # Safe division

    total_score = (
        weight_no_runtime_error * scores.get('percentage_no_runtime_error', 0) +
        weight_best_score * scores.get('best_score_without_validation', 0) +
        (weight_validated_scores * (1 + validated_score_avg) if scores.get('validated_scores') else 0)
    )
    return total_score if total_score > 0 else 0

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

# Décomposer les tuples pour éviter les tuples imbriqués
def unpack_tuples(items):
    unpacked = []
    for item in items:
        if isinstance(item, tuple):
            unpacked.extend(unpack_tuples(item))
        else:
            unpacked.append(item)
    return unpacked

def flatten_and_pair(nested_list):
    flat_list = flatten(nested_list)
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

def semantic_double_pass_chunking(text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> list:
    """
    Splits the text into chunks using a two-pass semantic approach.
    First, it splits the text by paragraph breaks, then merges smaller chunks to achieve the desired chunk size,
    adding an overlap between chunks to preserve context.
    """
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []
    current_chunk = ""
    for p in paragraphs:
        if len(current_chunk) + len(p) <= chunk_size:
            current_chunk += p + "\n\n"
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = p + "\n\n"
    if current_chunk:
        chunks.append(current_chunk.strip())
   
    # Add overlapping between chunks
    if chunk_overlap > 0 and len(chunks) > 1:
        overlapped_chunks = []
        for i, chunk in enumerate(chunks):
            if i > 0:
                prev_overlap = chunks[i-1][-chunk_overlap:]
                overlapped_chunks.append(prev_overlap + " " + chunk)
            else:
                overlapped_chunks.append(chunk)
        return overlapped_chunks
    else:
        return chunks

def extract_json(data: any) -> dict:
    """
    Extract and return a JSON object from the given input 'data'.

    If 'data' is already a dict or list, it is returned as-is.
    If 'data' is a string:
      - First, it attempts to parse it entirely as JSON.
      - If that fails, it uses two alternative methods:
        
        Method 1: Regex-based extraction.
          - Uses a regex pattern to extract a substring that looks like JSON.
          - Advantage: Very simple and concise.
          - Drawback: It may fail or capture too little/much if the string contains extra text
            or if the JSON has nested structures with inner braces/brackets.

        Method 2: Decoder-based extraction.
          - Iterates over the string and uses JSONDecoder.raw_decode() to try to decode a JSON
            object from positions where a '{' or '[' appears.
          - Advantage: This method leverages the JSON parser’s own grammar, making it more
            robust for nested objects or arrays.
          - Drawback: It may be slightly less intuitive than a one-line regex.

    Returns:
        A parsed JSON object (usually a dict or list).

    Raises:
        ValueError: If no valid JSON can be extracted from the input.
    """
    # If the data is already a dict or list, assume it's valid JSON.
    if isinstance(data, (dict, list)):
        return data

    # Ensure we have a string (if not, convert it).
    if not isinstance(data, str):
        data = str(data)

    data = data.strip()

    # First attempt: Try to parse the whole string as JSON.
    try:
        return json.loads(data)
    except json.JSONDecodeError:
        print("Not a pure JSON string, so we try to extract the JSON part.")

    # --- Method 1: Regex-based extraction ---
    regex_pattern = r'(\{.*\}|\[.*\])'
    match = re.search(regex_pattern, data, re.DOTALL)
    if match:
        candidate = match.group(0)
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            print("If the candidate isn't valid JSON, we try the next method.")

    # --- Method 2: Using JSONDecoder's raw_decode method ---
    decoder = json.JSONDecoder()
    # Iterate over the string; try to decode JSON starting at every '{' or '['.
    for i in range(len(data)):
        if data[i] in ['{', '[']:
            try:
                obj, idx = decoder.raw_decode(data[i:])
                return obj
            except json.JSONDecodeError:
                continue

    # If both methods fail, raise an error.
    print("No valid JSON found in the input data.")


def secure_invoke(llm,*args,**kwargs):
    """
    A wrapper function to invoke an LLM with error handling.
    """
    try:
        return llm.invoke(*args,**kwargs)
    except BadRequestError as e:
        if e.param == 'temperature':
            logging.warning(f"{llm.model_name} LLM don't support temperature parameter, removing it and retrying.")
            kwargs.pop('temperature', None)  # Remove temperature if it causes an error
            return secure_invoke(llm,*args, **kwargs)
        else:
            raise e