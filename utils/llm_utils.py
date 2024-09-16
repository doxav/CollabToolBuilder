import inspect
import re
import threading
import urllib.request
import subprocess

from typing import List, Optional, Union
from dataclasses import dataclass, field
from jinja2 import Template
from langchain import LLMChain
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate


import time
import json
from elasticsearch import Elasticsearch
import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

from langchain_community.embeddings import HuggingFaceEmbeddings, OpenAIEmbeddings
from langchain_core.runnables import RunnableSequence, ConfigurableField
#from langchain.schema import AIMessage, HumanMessage, SystemMessage, FunctionMessage
from langchain_core.messages.human import HumanMessage
from langchain_core.messages.ai import AIMessage
from langchain_core.messages.system import SystemMessage
from langchain_core.messages.function import FunctionMessage

import tkinter as tk
from tkinter import scrolledtext

from utils.file_utils import *
import concurrent.futures

from langchain_community.vectorstores import Chroma, ElasticsearchStore
from config import PickleCacheActivated
import os
import openai
from requests.auth import HTTPBasicAuth
from config import *
from websocket_server import WebsocketServer

openai.api_key = os.environ['OPENAI_API_KEY']
if 'OPENAI_BASE_URL' in os.environ: openai.base_url = os.environ['OPENAI_BASE_URL']

ON_INPUT = True

AGENT = ''

import asyncio
import websockets

ws_url = "ws://localhost:6789"


def smart_print(message: str, agent_name=None, message_type=None, append=False, column_id=None, column_max=None):
    global AgentDisplayManager, AGENT
    if 'IN_NOTEBOOK' not in globals():
        try:  # test if IN_NOTEBOOK
            from IPython import get_ipython
            globals()['IN_NOTEBOOK'] = IN_NOTEBOOK = get_ipython().__class__.__name__ == 'ZMQInteractiveShell'
            print("Smart_print: Notebook mode = " + str(IN_NOTEBOOK))
        except:
            globals()['IN_NOTEBOOK'] = IN_NOTEBOOK = False
    else:
        IN_NOTEBOOK = globals()['IN_NOTEBOOK']

    if 'IN_WEBSOCKET' not in globals():
        if HumanLLMMonitor.use_websocket:
            globals()['IN_WEBSOCKET'] = IN_WEBSOCKET = True
        else:
            globals()['IN_WEBSOCKET'] = IN_WEBSOCKET = False
    else:
        IN_WEBSOCKET = globals()['IN_WEBSOCKET']

    if IN_WEBSOCKET:
        # Check if in the message there are no unexpected non-whitespace characters
        if re.search(r'[^\x20-\x7E\t\n\r]', message):
            # Remove unexpected characters
            message = re.sub(r'[^\x20-\x7E\t\n\r]', "", message)
        message_dict = {'message':message, 'agent_name':agent_name, 'message_type':message_type, 'append':append, 'column_id':column_id, 'column_max':column_max}
        # convert message_dict to json
        message = json.dumps(message_dict)
        time.sleep(0.05)
        # Wait 2 seconds every 100 messages to avoid flooding the WebSocket server
        if HumanLLMMonitor.websocket_server.message_count % 250 == 0:
            time.sleep(2)
        HumanLLMMonitor.websocket_server.send_message(message)

    elif IN_NOTEBOOK and agent_name:
        # import AgentDisplayManager from utils.jupyter_agents_display if AgentDisplayManager is not initialized
        if 'AgentDisplayManager' not in globals():
            try:
                from utils.jupyter_agents_display import AgentDisplayManager
            except:
                print("AgentDisplayManager cannot be imported/initialized")
                print(message)
                return
        # create string with time of format HH:MM:SS
        time_str = datetime.now().strftime("%H:%M:%S")
        AgentDisplayManager.write_to_agent(agent_name, message, element_name=message_type + " " + time_str,
                                           append=append)
    else:
        if append:
            print(message, end="", flush=True)
        else:
            print(message)


def smart_input(message: str, agent_name=None, message_type=None):
    if 'IN_NOTEBOOK' not in globals():
        try:  # test if IN_NOTEBOOK
            from IPython import get_ipython
            globals()['IN_NOTEBOOK'] = IN_NOTEBOOK = get_ipython().__class__.__name__ == 'ZMQInteractiveShell'
        except:
            globals()['IN_NOTEBOOK'] = IN_NOTEBOOK = False
    else:
        IN_NOTEBOOK = globals()['IN_NOTEBOOK']

    if 'IN_WEBSOCKET' not in globals():
        # Check if a Streamlit server is running
        if HumanLLMMonitor.use_websocket:
            globals()['IN_WEBSOCKET'] = IN_WEBSOCKET = True

        else:
            globals()['IN_WEBSOCKET'] = IN_WEBSOCKET = False
    else:
        IN_WEBSOCKET = globals()['IN_WEBSOCKET']
    if IN_WEBSOCKET:
        structured_message = {'message': message, 'agent_name': agent_name, 'message_type': message_type, 'input': True}
        # convert structured_message to json
        message = json.dumps(structured_message)
        no_client = True
        while no_client:
            HumanLLMMonitor.websocket_server.send_message(message)
            if len(HumanLLMMonitor.websocket_server.connected_clients) > 0:
                no_client = False
            else:
                print("Waiting for WebSocket client to connect")
                time.sleep(1)

        # Wait and receive response from WebSocket
        async def receive_message():
            async with websockets.connect(ws_url) as websocket:
                print("SMART INPUT Waiting for response from WebSocket")
                response = await websocket.recv()
                print("SMART INPUT Received response from WebSocket")
                answer = json.loads(response)['message']
                print("SMART INPUT Answer: ", answer)
                return str(answer).upper()

        return asyncio.run(receive_message())

    elif IN_NOTEBOOK and agent_name:  # Currently DE-ACTIVATED
        # import AgentDisplayManager from utils.jupyter_agents_display if AgentDisplayManager is not initialized
        if 'AgentDisplayManager' not in globals():
            try:
                from utils.jupyter_agents_display import AgentDisplayManager
            except:
                print("AgentDisplayManager cannot be imported/initialized")
                print(message)
                return
        return AgentDisplayManager.get_input(agent_name, message)
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


def _visual_input(initial_string="", filetype="md"):
    """
    Open a Tkinter window to interactively edit a given string.
    
    Args:
    - initial_string (str): The string to be edited.

    Returns:
    - str: The edited string.
    """
    if is_vscode_installed():
        # Step 1: Generate the code and save it to a file. Check if folder temps/edition exists, if not create it
        if not os.path.exists('temp/edition'): os.makedirs('temp/edition')
        file_path = 'temp/edition/' + str(datetime.now().timestamp()) + f".{filetype}"
        with open(file_path, 'w') as file:
            file.write(initial_string)

        # Step 2: Open the file in VSCode. The `--wait` flag makes the subprocess call wait until the file is closed in VSCode.
        subprocess.run(["code", "--wait", file_path])

        # Step 3: After the file is closed, you can read the contents
        with open(file_path, 'r') as file:
            edited_string = file.read()
        # delete the file
        os.remove(file_path)

        # Now you have the modified code in `modified_code` variable
        return edited_string

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
    root.title("Edit String")

    # Create a scrolled text widget for editing the string
    txt_edit = scrolledtext.ScrolledText(root, wrap=tk.WORD, width=80, height=20)
    txt_edit.pack(padx=10, pady=10, expand=True, fill=tk.BOTH)

    # Insert the initial string into the text widget
    txt_edit.insert(tk.END, initial_string)
    # ... inside your _visual_input function ...
    txt_edit.bind("<Control-c>", copy)
    txt_edit.bind("<Control-v>", paste)
    txt_edit["undo"] = True
    txt_edit.bind("<Control-z>", lambda event: txt_edit.edit_undo())
    txt_edit.bind("<Control-y>", lambda event: txt_edit.edit_redo())

    # Bind the window's close event
    root.protocol("WM_DELETE_WINDOW", on_close)

    # Edited string variable
    edited_string = initial_string

    # Focus on the text widget and start the main loop
    txt_edit.focus_set()
    root.mainloop()

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
        smart_print(f"Moving existing prompt file {prompt_file_path_name} to {moved_file_path_name}","save_prompt")
        f_move(prompt_file_path_name, moved_file_path_name)

    smart_print(f"Saving new prompt file {prompt_file_path_name}", "save_prompt")

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
        smart_print(f"Moving existing prompt file {prompt_file_path_name} to {moved_file_path_name}","save_prompt_with_tag")
        f_move(prompt_file_path_name, moved_file_path_name)

    smart_print(f"Saving new prompt file {prompt_file_path_name}", "save_prompt_with_tag")

    # Save the file
    return dump_text(text, prompt_file_path_name)

@dataclass
class FewShotsParams:
    num: int = 5
    filter: dict = field(default_factory=dict)
    ranking_method: str = 'by_date_desc'
    annotations: Optional[Union[str, List[str]]] = None
    generate_summary: bool = False
    format: Optional[str] = None
    summary_char_limit: int = 500

class UnifiedVectorDB:
    db_type = 'elasticsearch'  # can be 'elasticsearch' or 'chroma'
    es_url = 'http://127.0.0.1:9200'
    es_user = None
    es_password = None
    OpenAI_embedding_function_name = "text-embedding-ada-002"
    db_connection_check_done = False

    @staticmethod
    def check_db():
        if UnifiedVectorDB.db_connection_check_done is True:
            return
        if UnifiedVectorDB.db_type == 'elasticsearch':
            session = requests.Session()
            retry = Retry(total=5, backoff_factor=1)
            adapter = HTTPAdapter(max_retries=retry)
            session.mount("https://", adapter)
            auth = HTTPBasicAuth(UnifiedVectorDB.es_user, UnifiedVectorDB.es_password) if UnifiedVectorDB.es_user else None
            try:
                response = session.get(UnifiedVectorDB.es_url, auth=auth, timeout=5, verify=False)
                response.raise_for_status()
                print("Elasticsearch response:", response.text)
                UnifiedVectorDB.db_connection_check_done = True
            except requests.exceptions.RequestException as e:
                print(f"Error: {e}\nURL: {UnifiedVectorDB.es_url}\nCheck Elasticsearch and credentials.")
                exit(1)
        elif UnifiedVectorDB.db_type == 'chroma':
            print("Chroma DB check is not yet implemented")
            UnifiedVectorDB.db_connection_check_done = True
        else:
            raise ValueError(f"Unsupported DB type: {UnifiedVectorDB.db_type}")

    def __init__(self, collection_name, embedding_function, persist_directory, reset_db_indices=False):
        UnifiedVectorDB.check_db()
        self.collection_name = collection_name.lower()
        self.embedding_function = embedding_function
        self.persist_directory = persist_directory

        if UnifiedVectorDB.db_type == 'chroma':
            self.db = Chroma(
                collection_name=collection_name,
                embedding_function=embedding_function,
                persist_directory=persist_directory
            )
            self._collection = self.db._collection
        elif UnifiedVectorDB.db_type == 'elasticsearch':
            elastic_client = Elasticsearch(UnifiedVectorDB.es_url,
                                           http_auth=(UnifiedVectorDB.es_user, UnifiedVectorDB.es_password) if UnifiedVectorDB.es_user else None,
                                           verify_certs=False, ssl_show_warn=False)
            self.db = ElasticsearchStore(
                index_name=self.collection_name,
                embedding=embedding_function,
                es_connection=elastic_client,
                distance_strategy="COSINE"
            )
            self._collection = self.db
            embedding_test = embedding_function.embed_query("test")
            embedding_size = len(embedding_test)
            if reset_db_indices:
                self.db.client.indices.delete(index=self.collection_name, ignore=[400,
                                                                                  404])  # TODO: set it as a parameter to reset when changing embeddings
            self.db._create_index_if_not_exists(index_name=self.collection_name, dims_length=embedding_size)
        else:
            raise ValueError(f"Unsupported DB type: {UnifiedVectorDB.db_type}")

    def add_texts(self, texts, ids=None, metadatas=None):
        if UnifiedVectorDB.db_type == 'chroma':
            # Avec Chroma, on suppose que la méthode add_texts existe déjà.
            return self.db.add_texts(texts=texts, ids=ids, metadatas=metadatas)
        elif UnifiedVectorDB.db_type == 'elasticsearch':
            return self.db.add_texts(texts=texts, metadatas=metadatas, ids=ids)

    def delete(self, ids):
        if UnifiedVectorDB.db_type == 'chroma':
            return self.db.delete(ids=ids)
        elif UnifiedVectorDB.db_type == 'elasticsearch':
            return self.db.delete(ids=ids)

    def similarity_search_with_score(self, query, k=1):
        if UnifiedVectorDB.db_type == 'chroma':
            return self.db.similarity_search_with_score(query, k=k)
        elif UnifiedVectorDB.db_type == 'elasticsearch':
            return self.db.similarity_search_with_score(query, k=(k if k <= 50 else 50))  # k seems to crash when > 50

    # query( query_embeddings, query_texts, n_results, where, where_document, include)
    def query(self, query_text="", k=1, metadata_filter=None, metadata_filter_OR=False, custom_filter_chrome=None,
              custom_filter_es=None, sort_order=None):
        if UnifiedVectorDB.db_type == 'chroma':
            if metadata_filter and custom_filter_chrome is None:
                # filter on AND conditions: "filter":{'$and': [{'user_id': {'$eq': user_id}}, {'category_id': {'$eq': cat_id}}]}})
                # filter on list of values: "filter":{'user_id': {'$in': [user_id_1, user_id_2]}}})
                # filter on OR conditions: "filter":{'$or': [{'user_id': {'$eq': user_id}}, {'category_id': {'$eq': cat_id}}]}})
                filter_chroma = []
                for key, value in metadata_filter.items():
                    # set sign to $eq if value is a string, else to $in
                    sign = '$eq' if isinstance(value, str) else '$in'
                    filter_chroma.append({key: {sign: value}})
                filter_chroma = {('$or' if metadata_filter_OR else '$and'): filter_chroma}
                if sort_order == 'asc' or sort_order == 'desc':  # incompatible with knn search, so we rewrite query just keeping filters and sort
                    print("WARNING: sort not implemented for Chroma DB")
            return self.db.query(query_text, k=k, filter=filter_chroma)
        elif UnifiedVectorDB.db_type == 'elasticsearch':
            if metadata_filter and custom_filter_es is None:
                custom_filter_es = []
                for key, value in metadata_filter.items():
                    custom_filter_es.append({"match": {f"metadata.{key}": value}})
                if metadata_filter_OR:
                    custom_filter_es = {"bool": {"should": custom_filter_es}}
            if sort_order == 'asc' or sort_order == 'desc':  # incompatible with knn search, so we rewrite query just keeping filters and sort
                def custom_query(query_body: dict, query: str):
                    return {"query": {"bool": {"must": custom_filter_es}},
                            "sort": [{"metadata.time": {"order": sort_order}}]}  # removed: , "size": k

                return self.db.similarity_search(query_text, k=(k if k <= 50 else 50),
                                                 custom_query=custom_query)  # k seems to crash when > 50
            else:
                return self.db.similarity_search(query_text, k=(k if k <= 50 else 50),
                                                 filter=custom_filter_es)  # k seems to crash when > 50
            # filter on AND conditions: filter=[{"match":{"metadata.function_name":function_name}}, {"match":{"metadata.agent_name":agent_name}}]

    def count(self):
        if UnifiedVectorDB.db_type == 'chroma':
            return self.db._collection.count()
        elif UnifiedVectorDB.db_type == 'elasticsearch':
            response = self.db.client.count(index=self.collection_name, body={"query": {"match_all": {}}})
            return response['count']

    def persist(self):
        if UnifiedVectorDB.db_type == 'chroma':
            self.db.persist()
        elif UnifiedVectorDB.db_type == 'elasticsearch':
            # No specific handler for ElasticsearchStore
            pass

    def clear(self):
        if UnifiedVectorDB.db_type == 'chroma':
            self.db._collection.clear()
        if UnifiedVectorDB.db_type == 'elasticsearch':
            # empty the index self.collection_name
            response = self.db.client.delete_by_query(index=self.collection_name, body={"query": {"match_all": {}}})
            print(f"Deleted {response['deleted']} documents from index {self.collection_name}")
            # sleep 2 seconds to let the index be updated
            time.sleep(2)

    # TODO: start by replacing UnifiedVectorDB by neo4j improving the ChatGPT generated code below, then validate the learn.py process works properly


# Example usage
# db = UnifiedVectorDB("documents", embedding_function, "bolt://localhost:7687", "neo4j", "password")
# db.add_texts(["This is a test document"], ids=["doc1"], metadatas=[{"author": "Alice"}])

class InferenceCheck:
    def __init__(self, check_name, check_function):
        self.check_name = check_name
        self.check_function = check_function

    def run_check(self, *args, **kwargs):
        return self.check_function(*args, **kwargs)

class HumanLLMMonitor:
    default_skip_rounds = 0
    step_id = 0
    function_list = None
    common_vectordb = None
    common_vectordb_embedding_function = None #OpenAIEmbeddings(model=UnifiedVectorDB.OpenAI_embedding_function_name, deployment=UnifiedVectorDB.OpenAI_embedding_function_name)  #HuggingFaceEmbeddings(model_name="intfloat/e5-base-v2", encode_kwargs={"normalize_embeddings": True}) # TODO: set as a parameter

    common_vectordb_collection_name = "human_llm_monitor_logs"
    common_vectordb_persist_directory = "human_llm_monitor_vectordb"
    websocket_server = None
    use_websocket = True
    ws_thread = None
    stop_event = threading.Event()

    @classmethod
    def initialize_websocket_server(cls):
        if cls.use_websocket and cls.websocket_server is None:
            cls.websocket_server = WebsocketServer()
            cls.stop_event.clear()
            cls.ws_thread = threading.Thread(target=cls.run_websocket_server)
            cls.ws_thread.daemon = True  # Run the WebSocket server in a daemon thread
            cls.ws_thread.start()

    @classmethod
    def run_websocket_server(cls):
        asyncio.run(cls.websocket_server.main(cls.stop_event))

    @classmethod
    def stop_websocket_server(cls):
        if cls.websocket_server:
            cls.stop_event.set()
            cls.ws_thread.join(timeout=5)  # Wait for 5 seconds to join the thread
            cls.websocket_server = None
            cls.ws_thread = None

    # static method to change common_vectordb_embedding_function which can be either OpenAIEmbeddings or HuggingFaceEmbeddings
    @staticmethod
    def set_common_vectordb_embedding_function(embedding_function):
        # if embedding_function is a string, then create the corresponding embedding function
        if isinstance(embedding_function, str):
            if embedding_function in ["OpenAIEmbeddings", "text-embedding-ada-002"]:
                HumanLLMMonitor.common_vectordb_embedding_function = OpenAIEmbeddings(model=embedding_function, deployment=UnifiedVectorDB.OpenAI_embedding_function_name)
            elif embedding_function == "HuggingFaceEmbeddings":
                HumanLLMMonitor.common_vectordb_embedding_function = HuggingFaceEmbeddings(
                    model_name="intfloat/e5-base-v2", encode_kwargs={"normalize_embeddings": True})
            else:
                HumanLLMMonitor.common_vectordb_embedding_function = HuggingFaceEmbeddings(
                    model_name=embedding_function, encode_kwargs={"normalize_embeddings": True}, model_kwargs={"trust_remote_code": True})
        else:
            HumanLLMMonitor.common_vectordb_embedding_function = embedding_function

    @staticmethod
    def _check_and_init_vector_db(embedding_function=None, reset_db_indices=False):
        if embedding_function:
            HumanLLMMonitor.set_common_vectordb_embedding_function(embedding_function)
        if HumanLLMMonitor.common_vectordb is None:
            HumanLLMMonitor.common_vectordb = UnifiedVectorDB(
                collection_name=HumanLLMMonitor.common_vectordb_collection_name,
                embedding_function=HumanLLMMonitor.common_vectordb_embedding_function,
                persist_directory=HumanLLMMonitor.common_vectordb_persist_directory,
                reset_db_indices=reset_db_indices
            )

    def get_few_shots_tag_args(self, prompt=None):
        """
        Retrieves the few shots tag arguments from the system prompt.

        Returns:
            dict: A dictionary containing the few shots tag arguments.
        """
        if prompt is None:
            prompt = self.system_prompt
        few_shots_match = re.search(r"few_shots:\s*(\{.*\})?$", prompt)
        few_shots_data = json.loads(few_shots_match.group(1)) if few_shots_match and few_shots_match.group(1) else {}
        return few_shots_data

    def set_llmORchain(self, llm_name, is_premium=False, temperature=0.7):
        if llm_name in self.llmORchains_list:
            if is_premium:
                self.premium_llm_name = llm_name
            else:
                self.default_llm_name = llm_name
                
            selected_llm_or_chain = self.llmORchains_list[llm_name]

            # Check if it's a sequence of steps (RunnableSequence)
            if isinstance(selected_llm_or_chain, RunnableSequence):
                modified_steps = []
                for step in selected_llm_or_chain.steps:
                    if hasattr(step, "steps__") and isinstance(step.steps__, dict):
                        # Handle the case where step is a dict of parallel runnables
                        modified_dict = {}
                        for key, sub_step in step.steps__.items():
                            if hasattr(sub_step, 'configurable_fields'):
                                try:
                                    sub_step = sub_step.configurable_fields(
                                        temperature=ConfigurableField(
                                            id="llm_temperature",
                                            name="LLM Temperature",
                                            description="The temperature of the LLM"
                                        )
                                    ).with_config(configurable={"llm_temperature": temperature}) 
                                    print("temperature has been changed to:", temperature) # Replace with desired default temperature
                                except ValueError as e:
                                    smart_print(f"Sub-step {key} in step {step} does not support temperature configuration: {e}", self.agent_name)
                            modified_dict[key] = sub_step
                        step.steps__ = modified_dict
                        modified_steps.append(step)
                    elif hasattr(step, 'configurable_fields'):
                        try:
                            step = step.configurable_fields(
                                temperature=ConfigurableField(
                                    id="llm_temperature",
                                    name="LLM Temperature",
                                    description="The temperature of the LLM"
                                )
                            ).with_config(configurable={"llm_temperature": temperature}) 
                            print("temperature hass been changed to:", temperature, llm_name) # Replace with desired default temperature
                        except ValueError as e:
                            print(f"Step {step} does not support temperature configuration: {e}", self.agent_name)
                        modified_steps.append(step)
                    else:
                        modified_steps.append(step)

                # Reconstruct the sequence with the modified steps
                selected_llm_or_chain = RunnableSequence(
                    first=modified_steps[0],
                    middle=modified_steps[1:-1] if len(modified_steps) > 2 else None,
                    last=modified_steps[-1]
                )
            elif hasattr(selected_llm_or_chain, 'configurable_fields'):
                # If it's a single LLM or other runnable that supports configurable fields, apply directly
                try:
                    selected_llm_or_chain = selected_llm_or_chain.configurable_fields(
                        temperature=ConfigurableField(
                            id="llm_temperature",
                            name="LLM Temperature",
                            description="The temperature of the LLM"
                        )
                    ).with_config(configurable={"llm_temperature": temperature}) 
                    print("temperature has been changed to:", temperature, llm_name) # Replace with desired default temperature
                except ValueError as e:
                    smart_print(f"LLM/Chain '{llm_name}' does not support temperature configuration: {e}", self.agent_name)

            # Apply structured output if needed
            if self.output_schema:
                if isinstance(selected_llm_or_chain, RunnableSequence):
                    # Apply `with_structured_output` to the last element in the sequence
                    last_element = selected_llm_or_chain.steps[-1]
                    if hasattr(last_element, 'with_structured_output'):
                        last_element = last_element.with_structured_output(self.output_schema)
                    # Reconstruct the sequence with the modified last element
                    if len(selected_llm_or_chain.steps) > 1:
                        selected_llm_or_chain = RunnableSequence(
                            first=selected_llm_or_chain.steps[0],
                            middle=selected_llm_or_chain.steps[1:-1] if len(selected_llm_or_chain.steps) > 2 else None,
                            last=last_element
                        )
                    else:
                        # If there's only one step, treat the last_element as the entire sequence
                        selected_llm_or_chain = last_element
                elif hasattr(selected_llm_or_chain, 'with_structured_output'):
                    # Apply `with_structured_output` directly if it's not a sequence
                    selected_llm_or_chain = selected_llm_or_chain.with_structured_output(self.output_schema)
                else:
                    # If neither condition matches, `selected_llm_or_chain` is not modified
                    smart_print(f"LLM/Chain '{llm_name}' does not support structured output", self.agent_name)

            # Set the LLM/Chain to the possibly modified or original one
            if is_premium:
                self.premium_llm = selected_llm_or_chain
            else:
                self.default_llm = selected_llm_or_chain
                
            return True
        else:
            smart_print(f"LLM/Chain '{llm_name}' not found in llmORchains_list {[key for key in self.llmORchains_list]}", self.agent_name)
            return False

    def set_default_llmORchain(self, llm_name, temperature=0.7):
        return self.set_llmORchain(llm_name, is_premium=False, temperature=temperature)

    def set_premium_llmORchain(self, llm_name, temperature=0.7):
        return self.set_llmORchain(llm_name, is_premium=True, temperature=temperature)

    def set_output_schema(self, output_schema, package_path = "."):
        # test if output_schema is a string, then it means it is a filename located in the prompt repo, load it and set it as output_schema
        if isinstance(output_schema, str):
            if "/" not in output_schema: self.output_schema_path = f"{package_path}/prompts/{output_schema}"
            output_schema = self._get_pydantic_class(self.output_schema_path)
        self.output_schema = output_schema

    def _get_pydantic_class(self, file_path: str):
        # Dynamically import the module from the given file path
        import importlib.util
        import sys
        from typing import Type
        from pydantic import BaseModel
        
        module_name = "dynamic_module"
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        
        # Iterate through the attributes of the module to find the Pydantic class
        for attr_name in dir(module):
            attr = getattr(module, attr_name)
            if isinstance(attr, type) and issubclass(attr, BaseModel) and attr is not BaseModel:
                return attr
        
        raise ValueError("No Pydantic BaseModel class found in the provided file.")

    def __init__(self, system_prompt=None, CPS_env_type=None, agent_name=None, model_max_context_size=16000, default_llmORchain=None,
                 premium_llmORchain=None, premium_llm_by_default=False, num_parallel_inferences=1, llmORchains_list=None,
                 synthesize_mode=False, inference_checks=None, output_schema=None, temperature=0.7, optuna=False):
        self.current_inference_context = None
        if llmORchains_list is None: raise ValueError("llmORchains_list must be provided")
        self.llmORchains_list = llmORchains_list
        self.temperature = temperature
        self.optuna = optuna
        self.system_prompt = system_prompt
        self.set_output_schema(output_schema)
        self.set_default_llmORchain(default_llmORchain if default_llmORchain else "default_llm") #.default_llm = default_llmORchain if default_llmORchain else self.llmORchains_list.get("default_llm")
        self.set_premium_llmORchain(premium_llmORchain if premium_llmORchain else "premium_llm") #premium_llm = premium_llmORchain if premium_llmORchain else self.llmORchains_list.get("premium_llm")
        self.CPS_env_type = CPS_env_type
        self.agent_name = agent_name or self.get_caller_class_name()
        if HumanLLMMonitor.use_websocket:
            if HumanLLMMonitor.websocket_server is None:
                HumanLLMMonitor.initialize_websocket_server()
            HumanLLMMonitor.websocket_server.add_monitor(self)
        # set in 1 line self.print_color is 32 for ActionAgent, 35 for CurriculumAgent, 31 for CriticAgent, 33 for SkillManager, 37 for else
        self.print_color = "32" if self.agent_name in ["ActionAgent", "CodingAgent"] else "35" if self.agent_name in [
            "CurriculumAgent", "TaskIdentificationAgent"] else "31" if self.agent_name in ["CriticAgent",
                                                                                           "ValidationAgent"] else "33" if self.agent_name in [
            "SkillManager", "CapitalizationAgent"] else "37"
        self.previous_templates = []
        self.previous_results = []
        self.comments = []
        self.skip_rounds = HumanLLMMonitor.default_skip_rounds
        self.log_data = []
        self.num_parallel_inferences = num_parallel_inferences
        self.llm_max_context_size = model_max_context_size
        self.premium_llm_by_default = premium_llm_by_default
        self.synthesize_mode = synthesize_mode
        self.inference_checks = inference_checks if inference_checks else {}
        self.last_inference_check_results = None


    # New: Handling function calls via WebSocket
    def execute_function(self, function_name, params):
        if hasattr(self, function_name):
            func = getattr(self, function_name)

            if callable(func):
                # Récupérer les informations de la signature de la fonction
                func_signature = inspect.signature(func)
                param_count = len(func_signature.parameters)

                # Si la fonction attend un seul argument positionnel
                if param_count == 1 and not isinstance(params, dict):
                    return func(params)

                # Si la fonction attend plusieurs arguments positionnels
                elif param_count > 1 and isinstance(params, (list, tuple)):
                    return func(*params)

                # Si la fonction attend des mots-clés et `params` est un dictionnaire
                elif isinstance(params, dict):
                    return func(**params)

                else:
                    raise TypeError(f"Cannot match parameters to function signature. Expected {param_count} parameters but received {type(params).__name__}.")

        # Retourner une valeur par défaut si la fonction n'existe pas ou n'est pas callable
        return None

    def load_prompt(self, function_name: str = None, agent_name: str = None, prompt: str = None) -> str:
        """
        Load a prompt with the ability to retrieve few_shots examples based on the provided tag parameters.

        :param function_name: The name of the function calling the prompt.
        :param agent_name: The name of the agent requesting the prompt.
        :param prompt: The name of the prompt file to load.
        :return: A formatted string including the main prompt and few_shots examples.
        """
        function_name = function_name if function_name else inspect.stack()[2].function

        with open(f"prompts/{prompt}.txt", "r") as f:
            prompt = f.read()

        if "few_shots" in prompt:
            few_shots_tag = self.get_few_shots_tag_args()
            match = re.search(r"few_shots:\s*(\{[^}]*\})\s*$", prompt, re.DOTALL)
            # Remove the few_shots tag and the dictionary from the prompt
            prompt = prompt[:match.start()].rstrip()
        else:
            return prompt

        HumanLLMMonitor._check_and_init_vector_db()

        # Manage the few_shots_tag parameters
        if few_shots_tag == {}:
            # Set default values if the dictionary is empty
            default_params = {
                'num': 5,  # Default number of examples to retrieve
                'filter': {},  # Default filter for the vector database
                'ranking_method': 'relevance',  # Default ranking method
                'generate_summary': False,  # Indicates if a summary should be generated
                'summary_char_limit': 200  # Character limit for the summary
            }
            # Merge the provided parameters with the default values
            merged_params = {**default_params, **(few_shots_tag or {})}

            # Initialize FewShotsParams with the merged parameters
            params = FewShotsParams(**merged_params)
        else :
            params = FewShotsParams(**few_shots_tag)

        # Create metadata filter for querying the vector database
        metadata_filter = {
            "function_name": function_name,
            "agent_name": agent_name,
            **params.filter
        }

        try:
            # Query the vector database to retrieve relevant log entries
            log_entries = HumanLLMMonitor.common_vectordb.query(
                query_text="*",
                metadata_filter=metadata_filter,
                k=params.num,
                sort_order=params.ranking_method
            )
        except Exception as e:
            raise ValueError(f"Error querying vector database: {e}")

        # Process the retrieved examples based on the parameters
        examples = self._process_examples(log_entries, params)

        # Generate a summary if required
        if params.generate_summary:
            summary = self.generate_summary(examples, params.summary_char_limit)
            examples.append(summary)

        # Combine the main prompt with the examples
        combined_prompt = f"{prompt}\n\nExamples:\n" + "\n\n".join(examples)

        return combined_prompt

    def _process_examples(self, log_entries: List, params: FewShotsParams) -> List[str]:
        """
        Process and format the examples based on the given parameters.

        :param log_entries: List of log entries retrieved from the vector database.
        :param params: FewShotsParams object containing processing parameters.
        :return: List of processed and formatted examples.
        """
        examples = []
        for entry in log_entries:
            example_content = json.loads(entry.page_content)

            # Filter examples based on annotations if specified
            if params.annotations:
                if isinstance(params.annotations, str):
                    params.annotations = [params.annotations]
                if 'annotation' in example_content and example_content['annotation'] in params.annotations:
                    examples.append(example_content)
            else:
                examples.append(example_content)

        # Format examples using the provided template if specified
        if params.format:
            template = Template(params.format)
            examples = [template.render(example=ex) for ex in examples]
        else:
            valid_examples = []
            for ex in examples:
                if 'output_llm_raw' in ex:
                    valid_examples.append(ex['output_llm_raw'][0])  # Assuming you want the first element of the list
                else:
                    logging.warning(f"Missing 'output_llm_raw' key in example: {ex}")
            examples = valid_examples

        return examples

    def generate_summary(self, examples: List[str], char_limit: int) -> str:
       """
       Generate a summary of the examples using a language model.

       :param examples: List of example strings to summarize.
       :param char_limit: Maximum character limit for the summary.
       :return: A string containing the generated summary.
       """
       # Set up the language model and prompt
       llm = OpenAI(temperature=self.temperature)
       prompt = PromptTemplate(
          input_variables=["examples"],
          template="Summarize the following examples in {char_limit} characters or less:\n\n{examples}"
       )

       # Create a chain to generate the summary
       chain = LLMChain(llm=llm, prompt=prompt)

       # Generate the summary
       summary = chain.run(examples="\n".join(examples), char_limit=char_limit)

       return f"Summary: {summary.strip()}"

    def add_inference_check(self, check_name, check_function):
        self.inference_checks[check_name] = InferenceCheck(check_name, check_function)

    def run_inference_checks(self, output_id, *args, **kwargs):
        results = {}
        for check_name, check in self.inference_checks.items():
            result = check.run_check(*args, **kwargs)
            results[check_name] = result
        # Ensure output_id is within bounds before updating the list
        if 0 <= output_id < len(self.last_inference_check_results):
            self.last_inference_check_results[output_id] = results  # Update the specific index
        return results

    def get_caller_class_name(self):
        # Returns the name of the class that called the current function
        return inspect.stack()[2][0].f_locals["self"].__class__.__name__

    def _max_tokens_ok(self, content):
        import tiktoken
        encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")  # TODO: replace by appropriate call to self.xxxxxx
        token_length = len(encoding.encode(content))

        if token_length > self.llm_max_context_size:
            smart_print(
                f"\033[31mCANNOT SEND MESSAGE TO LLM:\n{content}\n\nToo many tokens in human message for LLM ({token_length}). Fallback to manual feedback.\033[0m",self.agent_name)
            return False
        else:
            return True

    def _get_log_entries(self, agent_name, function_name, max_entries=20):
        HumanLLMMonitor._check_and_init_vector_db()
        result = HumanLLMMonitor.common_vectordb.query(
            query_text="*",
            metadata_filter={"function_name": function_name, "agent_name": agent_name},
            k=max_entries,
            sort_order="desc"  # Sort time from most recent to oldest
        )
        return result

    def synthesize_responses(self, responses, use_default_llm):
        system = """You have been provided with a set of responses from various open-source models to the latest user query. Your task is to synthesize these responses into a single, high-quality response while keeping the same output format structure. It is crucial to first critically evaluate the information provided in these responses, recognizing that some of it may be biased or incorrect. Your response should not simply replicate the given answers but should offer a refined, accurate, and comprehensive reply to the instruction with the same format output. Ensure your response is well-structured, coherent, and adheres to the highest standards of accuracy and reliability."""
        messages = [SystemMessage(content=system), HumanMessage(content="\n".join(responses))]
        formatted_responses = "\n\n".join(
            [f"RESPONSE {i + 1}: [[\n{response}\n]]" for i, response in enumerate(responses)])  # NEW/UPDATED
        messages = [SystemMessage(content=system), HumanMessage(content=formatted_responses)]  # NEW/UPDATED
        if use_default_llm:
            return self.default_llm.invoke(messages)
        else:
            return self.premium_llm.invoke(messages)

    def _before_inference(self, messages, default_llm_function, premium_llm_function, function_calling,
                          callable_system_message=None, use_premium_llm=None, optuna=None, model_choice=None):
        comments = None
        initial_user_message = messages[1].content
        function_name = inspect.stack()[2].function
        use_premium_llm = use_premium_llm if use_premium_llm is not None else self.premium_llm_by_default
        forced_llm_output = False  # TODO: try to set it to None

        while self.skip_rounds <= 0:
            # MENU
            menu = ''
            before_menu = f"\033[{self.print_color}m***** {self.agent_name}->{function_name}  BEFORE *****\nSYSTEM PROMPT:\n{messages[0].content}\n\nUSER MESSAGE:\n{messages[1].content}\n***** {self.agent_name}->{function_name} BEFORE *****\033[0m\n"
            if not self._max_tokens_ok(messages[0].content + "\n" + messages[1].content):
                before_menu += ("WARNING!!!! Max tokens exceeded, you should refactor user message or system prompt!\n")
            menu += ("[A] Modify agent's 'system prompt' (role, global context, constraints, examples) OR the answer SCHEMA output.\n")
            menu += ("[B] Add instruction or information to agent.\n")
            menu += ("[C] Skip and set LLM output from recent outputs or manually define it.\n")
            menu += ("[D] Log comments (not used by the model, just for information).\n")
            menu += ("[E] See all previous results for this agent.\n")
            menu += ("[F] See previous MODIFIED/SCORED/COMMENTED results for this agent.\n")
            menu += ("[G] Skip human actions for N rounds.\n")
            menu += ("[H] Change default LLM.\n")
            menu += ("[I] Change premium LLM.\n")
            #menu += (f"[I] Activate/de-activate function calling to allow model request external knowledge - current status: {function_calling}\n")
            menu += (
                f"[J] Change num of parallel inferences - Current value={self.num_parallel_inferences}, Synthesize mode=\033[32m{'ON' if self.synthesize_mode else 'OFF'}\033[0m\n")  # UPDATED
            menu += ("[K] Exit program.\n")
            menu += (f"[P] Proceed to inference using a PREMIUM LLM - Current value={use_premium_llm}\n")
            menu += (f"[Z] Continue\n")

            smart_print(before_menu+menu, self.agent_name, "BEFORE inference action MENU")
            menu_start_time = time.time()
            match (optuna.lower() if optuna else ""):
                case "coach":
                    llm_keys = list(self.llmORchains_list.keys())
                    if type(model_choice) == int:
                        # Model change from choice of optuna
                        new_llm_name = llm_keys[model_choice]
                    elif type(model_choice) == str:
                        if model_choice not in llm_keys:
                            raise ValueError(f"Model choice '{model_choice}' not found in llmORchains_list {llm_keys}")
                        new_llm_name = model_choice
                    else:
                        raise ValueError("Model choice must be an integer or a string")
                    self.set_default_llmORchain(new_llm_name)
                    self.set_premium_llmORchain(new_llm_name)
                    default_llm_function = self.default_llm
                    premium_llm_function = self.premium_llm
                    self.synthesize_mode = False
                    # Default actions for all agents while running with optuna
                    action = ""
                case "coder":
                    action = ""
                case _: # Default case
                    action = smart_input(
                        f"\033[32mBEFORE\033[0m inference @ {self.agent_name}-> Choose an action (or hit Enter for inference) :",self.agent_name).upper()

            # ACTIONS processing
            start_time, action = time.time(), action  # Init action selected and timer to measure time spent and occurences in action processing

            if action == "A":  # Modify system prompt
                comments, forced_llm_output = self.modify_prompt(callable_system_message, comments,
                                                                 default_llm_function, forced_llm_output, messages,
                                                                 premium_llm_function, use_premium_llm)

                # Adding the logic to edit the output schema
                if hasattr(self, 'output_schema_path') and self.output_schema_path:
                    new_schema_content = _visual_input(open(self.output_schema_path).read(), filetype="py")
                    confirm_schema = smart_input(
                        "Do you want to replace the current output schema with your input? (y/n): ", self.agent_name).upper()
                    if confirm_schema == "Y":
                        with open(self.output_schema_path, 'w') as schema_file:
                            schema_file.write(new_schema_content)
                        self.set_output_schema(self.output_schema_path)
                        # Reload default and premium LLMs
                        self.set_default_llmORchain(self.default_llm_name)
                        self.set_premium_llmORchain(self.premium_llm_name)
                        default_llm_function = self.default_llm
                        premium_llm_function = self.premium_llm

            elif action == "B":  # Add instruction or information to agent
                self.add_instruction(initial_user_message, messages)

            elif action == "C":  # Set LLM output by re-using past
                forced_llm_output = self.reuse_past(forced_llm_output, function_name)

            elif action == "D":  # Log comments
                comments = self.log_comments(comments)

            elif action == "E":  # See all previous results - list results from  HumanLLMMonitor.common_vectordb filtered by agent_name and function_name
                result = self.getPreviousResults(function_name,self.agent_name)
                _visual_input(result)

            elif action == "F":  # See previous MODIFIED/SCORED/COMMENTED results - list results from  HumanLLMMonitor.common_vectordb filtered by agent_name and function_name, filtered on comments
                self.getScoredResults(function_name)

            elif action == "G":  # Skip human actions for N rounds
                self.skipRounds()

            elif action == "H":  # Change default LLM
                default_llm_function = self.changeDefaultLLM(default_llm_function)

            elif action == "I":  # Change premium LLM
                premium_llm_function = self.changePremiumLLM(premium_llm_function)

                # elif action == "I":
                #     function_calling = not function_calling
                #     smart_print(f"function_calling is now {function_calling}", self.agent_name, "function_calling")

            elif action == "K":  # Exit program
                self.exitProgram()

            #elif action == "J":  # Change num of parallel inferences
            #    self.changeNumParallelInferences()

            elif action == "J":  # Change num of parallel inferences and synthesize mode
                self.changeNumParallelInferencesAndSynthesize()

            # Count time spent and occurrences waiting and in each option
            if action:
                if action not in self.before_inference_option_times:
                    self.before_inference_option_times[action] = 0
                    self.before_inference_option_counts[action] = 0
                self.before_inference_option_times[action] += (time.time() - start_time)
                self.before_inference_option_counts[action] += 1
            self.before_inference_option_times["TOTAL"] += (time.time() - menu_start_time)
            self.before_inference_option_counts["TOTAL"] += 1
            self.before_inference_option_times["SELECTION"] += (start_time - menu_start_time)
            self.before_inference_option_counts["SELECTION"] += 1
            start_time, menu_start_time = None, None

            if action in [None, "", "P",
                          "C","Z"]:  # P: Proceed to inference using a PREMIUM LLM; C: Set LLM output by re-using past
                if action == "P": use_premium_llm = True
                break
            else:
                proceed = "y" if optuna else smart_input(
                    "Proceed to inference (y/n) ? You can also hit 'p' to proceed using a premium llm.",self.agent_name,"INFERENCE CHOICE").lower()
                if proceed in ["y", "p", ""]:
                    if proceed == "p": use_premium_llm = True
                    break

        smart_print(
            f"Time spent in each option and occurrences: {self.before_inference_option_times} - {self.before_inference_option_counts}",self.agent_name)

        return messages, comments, forced_llm_output, use_premium_llm, default_llm_function, premium_llm_function, function_calling

    def changeNumParallelInferencesAndSynthesize(self):
        try:
            self.num_parallel_inferences = int(smart_input("Enter new value for num_parallel_inferences: ", self.agent_name, "NUM_PARALLEL_INFERENCES"))
        except:
            self.num_parallel_inferences = 1
        synthesize_mode_input = smart_input("Turn synthesis mode on/off (1 for ON, 0 for OFF): ",self.agent_name,"NUM_PARALLEL_INFERENCES SYNTHESIS MODE CHOICE").strip()  # NEW
        if synthesize_mode_input in ["0", "1"]:  # NEW
            self.synthesize_mode = synthesize_mode_input == "1"  # NEW
        else:  # NEW
            smart_print("Invalid input. Synthesize mode remains unchanged.", self.agent_name, "NUM_PARALLEL_INFERENCES SYNTHESIS MODE CHOICE")  # NEW

    def changeNumParallelInferences(self):
        try:
            self.num_parallel_inferences = int(
                smart_input("Enter new value for num_parallel_inferences: ", self.agent_name, "NUM_PARALLEL_INFERENCES"))
        except:
            self.num_parallel_inferences = 1

    def exitProgram(self):
        if 'IN_NOTEBOOK' in globals() and globals()['IN_NOTEBOOK']:
            # access to AgentDisplayManager.export_to_html() which is not registered in this file but in the notebook
            raise SystemExit("I just wanted to stop!")
        else:
            exit()

    def skipRounds(self):
        rounds = int(smart_input("Skip for how many rounds? ", self.agent_name))
        self.skip_rounds = rounds

    def log_comments(self, comments):
        comments = smart_input("Enter your comment on the prompt: ", self.agent_name)
        return comments

    def reuse_past(self, forced_llm_output, function_name):
        if HumanLLMMonitor.common_vectordb.count() > 0:
            log_entries, list_output = self._get_log_entries(self.agent_name, function_name), ""
            for idx, entry in enumerate(log_entries, start=1):
                content = json.loads(entry.page_content)
                text = (content['output_contents'][0]['content'].replace('\n', '\\') if content[
                    'output_contents'] else "") if isinstance(content['output_contents'], list) else \
                    content['output_contents']['content'].replace('\n', '\\')
                date = entry.metadata['time'].split('.')[0]
                list_output += (
                    f"\033[94m{idx}.\033[0m {text[:100]}....{text[-100:]} #{entry.metadata['function_name']} @{date}\n")  # Display a snippet of each entry
            smart_print(list_output, self.agent_name, "LOG ENTRIES LIST")

            try:
                selected_index = int(
                    smart_input(
                        "Select the log entry number to load or 0/enter to manually enter LLM output: ",
                        self.agent_name)) - 1
            except:
                selected_index = -1
        else:
            selected_index = -1
        if selected_index < 0 or selected_index >= len(log_entries):
            forced_llm_output = _visual_input("{replace with expected ANSWER/OUTPUT}")
        else:
            selected_log_entry = json.loads(log_entries[selected_index].page_content)
            forced_llm_output = (
                selected_log_entry['output_contents'][0] if isinstance(selected_log_entry['output_contents'],
                                                                       list) else selected_log_entry[
                    'output_contents'])['content']
        return forced_llm_output

    def add_instruction(self, initial_user_message, messages):
        new_message = initial_user_message + "\n" + _visual_input(" ")
        confirm = smart_input(
            f"***** NEW USER MESSAGE:\n{new_message}\n*************\nAre you sure you want to modify it? (y/n): ",
            self.agent_name).upper()
        if confirm == "Y":
            messages[1].content = new_message

    def modify_prompt(self, callable_system_message, comments, default_llm_function, forced_llm_output, messages,
                      premium_llm_function, use_premium_llm):
        new_template = None
        # List existing prompt variants including the base prompt
        prompt_variants = list_prompt_variants(self.system_prompt)
        output = ("Found the following prompt options:\n")
        for i, variant in enumerate(prompt_variants):
            output += (f"{i + 1}. {variant}\n")
        output += (
            f"{len(prompt_variants) + 1}. Ask LLM to generate a new variant of the current system prompt given my instructions\n")
        smart_print(output, self.agent_name, "PROMPT OPTIONS")
        variant_choice = smart_input(
            "Select a number to modify a prompt or create a new variant (or press Enter to continue with the current selection): ",
            self.agent_name)
        if variant_choice.isdigit() and 0 < int(variant_choice) <= len(prompt_variants) + 1:
            if int(variant_choice) == len(prompt_variants) + 1:
                # Process to create a new variant
                comments = smart_input("Provide critic or feedback for the current prompt: ", self.agent_name)
                refine_prompt = _visual_input(
                    f"Current system prompt:<<< {self.load_prompt(agent_name=self.agent_name, prompt=self.system_prompt)} >>>\n\nFeedback or critic: {comments}")
                forced_llm_output = default_llm_function.invoke(
                    [SystemMessage(content=self.load_prompt(agent_name=self.agent_name, prompt="improve_prompt_from_answer_critic")),
                     HumanMessage(content=refine_prompt)])
                new_template = forced_llm_output.content
            else:
                self.system_prompt = prompt_variants[int(variant_choice) - 1]
        if smart_input("Would you like first to get suggestions for a better prompt? (y/n): ",
                       self.agent_name).upper() == "Y":
            if use_premium_llm:
                forced_llm_output = premium_llm_function.invoke(
                    [SystemMessage(content=self.load_prompt(agent_name=self.agent_name, prompt="system_prompt_refiner")), HumanMessage(
                        content=f"PROMPT TO GET SUGGESTIONS FOR IMPROVEMENT:\n{self.load_prompt(agent_name=self.agent_name, prompt=self.system_prompt)}")])
            else:
                forced_llm_output = default_llm_function.invoke(
                    [SystemMessage(content=self.load_prompt(agent_name=self.agent_name, prompt="system_prompt_refiner")), HumanMessage(
                        content=f"PROMPT TO GET SUGGESTIONS FOR IMPROVEMENT:\n{self.load_prompt(agent_name=self.agent_name, prompt=self.system_prompt)}")])
            smart_print(
                f"***** PROMPT SUGGESTIONS *****\n\033[33m{forced_llm_output.content}\033[0m\n*************",
                self.agent_name, "PROMPT SUGGESTIONS")
        new_template = _visual_input(self.load_prompt(agent_name=self.agent_name, prompt=self.system_prompt) if new_template is None else new_template)
        smart_print(f"***** NEW PROMPT TEMPLATE:\n{new_template}\n*************", self.agent_name,
                    "NEW PROMPT TEMPLATE")
        # Confirm that the user wants to modify the template
        confirm = smart_input(
            "Do you want to replace current prompt file template with your input? (y/n): ", self.agent_name).upper()
        # Save prompt with tag options
        if confirm == "Y":
            tag_option = smart_input(
                "Enter a tag for saving the prompt (leave blank for no tag, or 'same' to keep the current tag): ",
                self.agent_name)
            if tag_option.lower() == "same":
                save_prompt_with_tag(self.system_prompt, new_template, "")
            else:
                save_prompt_with_tag(self.system_prompt, new_template, tag_option)

            if callable_system_message:
                messages[0] = callable_system_message()
            else:
                messages[0].content = new_template
        return comments, forced_llm_output

    def changePremiumLLM(self, premium_llm_function):
        llm_keys = list(self.llmORchains_list.keys())
        for i, key in enumerate(llm_keys): smart_print(f"{i}. {key}", self.agent_name)
        while True:
            new_llm_index = int(
                smart_input(f"Enter the number of the new premium LLM (0-{len(llm_keys) - 1}): ", self.agent_name))
            if 0 <= new_llm_index < len(llm_keys):
                new_llm_name = llm_keys[new_llm_index]
                if self.set_premium_llmORchain(new_llm_name): break
        premium_llm_function = self.premium_llm
        smart_print(f"Premium LLM changed to {new_llm_name}", self.agent_name, "Change Premium LLM")
        return premium_llm_function

    def changeDefaultLLM(self, default_llm_function):
        llm_keys = list(self.llmORchains_list.keys())
        for i, key in enumerate(llm_keys): smart_print(f"{i}. {key}", self.agent_name)
        while True:
            new_llm_index = int(
                smart_input(f"Enter the number of the new default LLM (0-{len(llm_keys) - 1}): ", self.agent_name))
            if 0 <= new_llm_index < len(llm_keys):
                new_llm_name = llm_keys[new_llm_index]
                if self.set_default_llmORchain(new_llm_name): break
        default_llm_function = self.default_llm
        smart_print(f"Default LLM changed to {new_llm_name}", self.agent_name, "Change Default LLM")
        return default_llm_function

    def getScoredResults(self, function_name):
        HumanLLMMonitor._check_and_init_vector_db()
        confirm = smart_input(
            "Do you want see:\n[A] all MODIFIED/SCORED/COMMENTED results.\n[B] INPUT modified only.\n[C] OUTPUT modified only.\n[D] SCORED only.\n[E] COMMENTED only.\nSelect your letter for choice or hit enter for all: ",
            self.agent_name,"BEFORE inference action MENU").upper()
        result = []
        if confirm in ["A", "", "B"]:
            result.extend(HumanLLMMonitor.common_vectordb.query(query_text="*",
                                                                metadata_filter={"function_name": function_name,
                                                                                 "agent_name": self.agent_name,
                                                                                 "input_modified": True},
                                                                k=100))
        if confirm in ["A", "", "C"]:
            result.extend(HumanLLMMonitor.common_vectordb.query(query_text="*",
                                                                metadata_filter={"function_name": function_name,
                                                                                 "agent_name": self.agent_name,
                                                                                 "output_modified": True},
                                                                k=100))
        if confirm in ["A", "", "D"]:
            result.extend(HumanLLMMonitor.common_vectordb.query(query_text="*",
                                                                metadata_filter={"function_name": function_name,
                                                                                 "agent_name": self.agent_name,
                                                                                 "scored": True}, k=100))
        if confirm in ["A", "", "E"]:
            result.extend(HumanLLMMonitor.common_vectordb.query(query_text="*",
                                                                metadata_filter={"function_name": function_name,
                                                                                 "agent_name": self.agent_name,
                                                                                 "commented": True}, k=100))
        visual_result = "\n===============================\n".join(
            [json.dumps(json.loads(item.page_content), indent=4, sort_keys=True).replace("\\n", "\n") for item
             in result])
        _visual_input(visual_result, filetype="json")

    @staticmethod
    def getPreviousResults(function_name=None,agent_name=None,k=100):
        HumanLLMMonitor._check_and_init_vector_db()
        metadata_filter = {}
        if function_name: metadata_filter["function_name"] = function_name
        if agent_name: metadata_filter["agent_name"] = agent_name
        result = HumanLLMMonitor.common_vectordb.query(query_text="*", metadata_filter=metadata_filter, k=k)
        visual_result = "\n===============================\n".join(
            [json.dumps(json.loads(item.page_content), indent=4, sort_keys=True).replace("\\n", "\n") for item
             in result])
        return visual_result

    def _after_inference(self, inference_result_msg, premium_llm_function, color="37", output_id=None,
                         outputs_count=None, optuna=None):
        comments, score = None, None
        nl = "\n"
        if inference_result_msg is None:
            # enable to request inference_result_msg.content to be None
            inference_result_msg = type('InferenceResult', (object,), {'content': None})

        while self.skip_rounds <= 0:
            # MENU
            multiple_ref = (f"OUTPUT \033[31m{output_id} OUT OF {outputs_count}\033[0m OUTPUTS" if (output_id and outputs_count and (outputs_count > 1)) else "")

            # Run inference checks if any
            check_results = self.run_inference_checks(output_id-1, inference_result_msg.content)
            check_display = ""
            # Display inference check results
            for check_name, result in check_results.items():
                check_display += f"{nl}CHECK {check_name} result: " + str(result).replace("\\n", "\n")

            menu = (f"\033[{self.print_color}m***** {self.agent_name}->{inspect.stack()[2].function} AFTER *****\nLLM ANSWER:\n{inference_result_msg.content}\n{check_display}\n***** {self.agent_name}->{inspect.stack()[2].function} AFTER *****\033[0m{multiple_ref}\n")

            menu += ("[A] Manually set/modify the answer/output (I don't want to try to improve agent's system prompt).\n")  # je voudrais le corriger uniquement pour demander une suggestion d'amélioration du prompt (d'un autre côté, je peux aussi le faire dans le menu précédent)
            menu += ("[B] Critic this answer/output to get an improved answer/output.\n")
            menu += ("[C] Find a better Prompt by providing critic and ideal answer.\n")
            menu += ("[D] Evaluate & comment answer (Score between 0(worst)-1(top), and explain) to improve future results by using scored/commented examples.\n")
            menu += ("[E] Go back BEFORE inference to improve system prompt or add information to user message.\n")
            menu += ("[G] Skip human actions for N rounds.\n")
            menu += ("[Z] Continue\n")
            menu += ("[H] Exit program.\n")

            smart_print(menu, self.agent_name, "AFTER inference action MENU" + (
                f" {output_id}/{outputs_count}" if (output_id and outputs_count and (outputs_count > 1)) else ""))
            menu_start_time = time.time()
            action = "" if optuna else smart_input(
                f"\n\033[32mAFTER\033[0m inference @ {self.agent_name}-> Choose an action (or hit Enter for inference) :",self.agent_name).upper()

            # ACTIONS processing
            start_time, action = time.time(), action  # Init action selected and timer to measure time spent and occurences in action processing

            if action == "A":  # Manually set/modify the answer/output
                self.modifyAnswer(inference_result_msg)

            elif action == "B":  # Critic this answer/output to get an improved answer/output
                comments = self.criticAnswer(comments, inference_result_msg, premium_llm_function)

            elif action == "C":  # Find a better Prompt by providing critic and ideal answer
                comments = self.findBetterPrompt(comments, inference_result_msg, premium_llm_function)

            elif action == "D":  # Evaluate & comment answer to re-use in prompts or later analysis
                comments, score = self.evaluateCommentAnswerForLater()

            elif action == "E":  # Go back BEFORE inference to improve system prompt or add information to user message
                inference_result_msg = self.goBackInference(inference_result_msg)

            elif action == "G":  # Skip human actions for N rounds
                self.skipRounds()

            elif action == "H":  # Exit program
                self.exitProgram()

            # Count time spent and occurrences waiting and in each option
            if action and action.isalpha() and len(action) == 1:
                # if action is set and not exist yet, also check if this is 1 single letter
                if action not in self.after_inference_option_times:
                    self.after_inference_option_times[action] = 0
                    self.after_inference_option_counts[action] = 0
                self.after_inference_option_times[action] += (time.time() - start_time)
                self.after_inference_option_counts[action] += 1
            self.after_inference_option_times["TOTAL"] += (time.time() - menu_start_time)
            self.after_inference_option_counts["TOTAL"] += 1
            self.after_inference_option_times["SELECTION"] += (start_time - menu_start_time)
            self.after_inference_option_counts["SELECTION"] += 1
            start_time, menu_start_time = None, None

            if action in [None, "",
                          "E", "Z"]: break  # E: Go back BEFORE inference to improve system prompt or add information to user message

            proceed = smart_input("Continue 'y' (or 'n' to go back to menu) ? ",self.agent_name).lower()
            if proceed in ["y", ""]:
                break

        if self.skip_rounds > 0:
            check_results = self.run_inference_checks(output_id-1, inference_result_msg.content)
            check_display = ""
            # Display inference check results
            for check_name, result in check_results.items():
                check_display += f"{nl}CHECK {check_name} result: " + str(result).replace("\\n", "\n")

            smart_print(
                f"\033[{self.print_color}m****{self.agent_name}>{inspect.stack()[2].function} LLM ANSWER content****\n{inference_result_msg.content}\n{check_display}\n*****************\033[0m",
                self.agent_name, "LLM ANSWER content")
            self.skip_rounds -= 1
        else:
            smart_print(
                f"Time spent in each option and occurrences: {self.after_inference_option_times} - {self.after_inference_option_counts}",
                self.agent_name)

        return inference_result_msg, comments, score

    def goBackInference(self, inference_result_msg):
        inference_result_msg = -1  # break is set after action time measurement
        return inference_result_msg

    def evaluateCommentAnswerForLater(self):
        while True:
            score = float(smart_input(
                "Give a note for the result between 0.0 (worst) and 1.0 (top), or 0 for bad, 1 for good: ",
                self.agent_name))
            # if score is not between 0 and 1, then set to None and print error
            if score < 0 or score > 1:
                score = None
                print(f"\033[31mInvalid score: {score}\033[0m")
            else:
                break
        comments = smart_input("Comment on the result: ", self.agent_name)
        return comments, score

    def evaluateCommentAnswerForLater(self, comment, score, output_id=0, message=None):
        if comment:
            self.comments.append(comment)

        context = self.current_inference_context

        # Determine the output content
        if message is not None:
            output_content = message
        elif context.get('output_contents'):
            output_content = context['output_contents'][output_id] if isinstance(context['output_contents'], list) else \
            context['output_contents']
        else:
            output_content = None

        # Compute inference time
        start_time = context.get('start_time')
        end_time = datetime.now()
        inference_time = (end_time - start_time).total_seconds if start_time else None

        self._log_entry(
            function_name=context.get('function_name'),
            input_contents=context.get('input_contents'),
            output_contents=output_content,
            inference_time=inference_time,
            input_modified=context.get('input_modified'),
            skipped_inference=context.get('skipped_inference'),
            skip_rounds=self.skip_rounds,
            input_comments=context.get('input_comments'),
            output_comments=[comment],
            output_llm_raw=context.get('raw_llm_outputs')[output_id] if isinstance(context.get('raw_llm_outputs'),
                                                                                   list) else context.get(
                'raw_llm_outputs'),
            output_modified=context.get(
                'output_modified'),
            user_score=score,  # New field for user-provided score
            message_tokens=context.get('message_tokens'),
            use_premium_llm=context.get('use_premium_llm'),
            call_duration=context.get('call_duration'),
            synthesize_mode=self.synthesize_mode
        )


    def findBetterPrompt(self, comments, inference_result_msg, premium_llm_function):
        comments = smart_input("First enter your critic here (then modify answer to get ideal answer): ",
                               self.agent_name)
        ideal_answer = _visual_input(inference_result_msg.content)
        refine_prompt = f"Current system prompt:<<< {self.load_prompt(agent_name=self.agent_name, prompt=self.system_prompt)} >>>\n\nPrompt's answer:<<< {inference_result_msg.content} >>>\n\nPrompt's answer critic:{comments}\n\nPrompt's ideal Answer:<<< {ideal_answer} >>>"
        smart_print(f"***** PROMPT FOR IMPROVEMENT *****\n{refine_prompt}", self.agent_name, "PROMPT FOR IMPROVEMENT")
        llm_output = premium_llm_function.invoke([SystemMessage(
            content=self.load_prompt(agent_name=self.agent_name, prompt="improve_prompt_from_answer_critic")),
                                           HumanMessage(content=refine_prompt)])
        smart_print(f"***** RECOMMENDATION OPEN FOR EDITION *****\n", self.agent_name,
                    "RECOMMENDATION OPEN FOR EDITION")
        new_template = _visual_input(llm_output.content)
        smart_print(f"***** NEW PROMPT TEMPLATE:\n{new_template}\n*************", self.agent_name,
                    "NEW PROMPT TEMPLATE")  # Confirm that the user wants to modify the template
        confirm = smart_input( "Do you want to replace current prompt file template with your input? (y/n): ",self.agent_name).upper() # Save prompt with tag options
        if confirm == "Y":
            tag_option = smart_input( "Enter a tag for saving the prompt (leave blank for no tag, or 'same' to keep the current tag): ",self.agent_name)
        if tag_option.lower() == "same":
            save_prompt_with_tag(self.system_prompt, new_template, "")
        else:
            save_prompt_with_tag(self.system_prompt, new_template, tag_option)

        return comments

    def criticAnswer(self, comments, inference_result_msg, premium_llm_function):
        def is_valid_python_structure(s):
            import ast
            output = None
            try:
                output = ast.literal_eval(s)
                return output, True
            except (ValueError, SyntaxError):
                return output, False
        content_structure, is_strucutre = is_valid_python_structure(inference_result_msg.content)
        while True:
            content_pretty = json.dumps(content_structure, indent=4) if is_strucutre else inference_result_msg.content
            content_annotated = _visual_input(content_pretty, filetype="py" if is_strucutre else "md")
            #comments = smart_input("Provide critic/feedback/request: ", self.agent_name)
            system_prompt = "Refine the ANSWER below given the HUMAN FEEDBACK to address for improving it."
            system_prompt = """
Refine the **ANNOTATED ANSWER** to the **TARGET TASK**, given answer including important inline text annotated instructions on key elements to improve the answer using tags (see **ANNOTATION TAGS** for interpretation). Directly answer with the updated answer aligned with annotated instructions tags, no introduction.

### TARGET TASK:
Identify and define the best task to develop using code and LLM given the status of available developed tasks and failed tasks

### ANNOTATION TAGS:
The following annotations are provided to guide the refinement process. Each annotation is in the format `\TAG{Annotated text...}[optional explanation]`. The tags indicate specific actions you should take to improve or finalize the text. Please follow the instructions for each tag carefully:
1. **\APPROVE:**
- **Purpose:** This tag indicates that the content is correct, clear, and relevant to the subject.
- **Action:** **No changes are necessary.** Retain this content exactly as it is.
- **Example:** \APPROVE{The system's reliability is essential for maintaining continuous operation.}
2. **\FIX:**
- **Purpose:** Content marked with this tag requires **correction or improvement**. There may be issues related to accuracy, clarity, or relevance.
- **Action:** Make necessary revisions to ensure the text is accurate, clear, and aligned with the overall subject matter.
- **Example:** \FIX{tomato_sauce_recipe = rubish_llm_factory('tell me how to make tomato sauce for my italian noodles')}
3. **\DELETE:**
- **Purpose:** This tag is used for content that is **unnecessary, irrelevant, or detrimental** to the quality of the text.
- **Action:** **Remove** this content entirely from the final version.
- **Example:** \DELETE{The report includes a lengthy discussion on unrelated financial data.}
4. **\variants:**
- **Purpose:** Content marked with this tag requires the generation of **alternative expressions or approaches**.
- **Action:** Create multiple appropriate variations between parenthesis after the inline text between curly braces e.g. {inline initial text...}(text of variant 1...)(text of variant 2...)
- **Example:** \VARIANTS{The user interface should be intuitive using multi-column visual side by side comparison.}
### Your Task:
1. **Understand** the initial task and subject matter to ensure that the text aligns with the overall objectives and audience requirements.
2. **Interpret** the annotations in the provided text according to the guidelines above.
3. **Revise** the text by making necessary corrections, deletions, or additions as instructed.
4. **Generate** alternative phrasings or approaches where indicated, ensuring each variant is clearly differentiated using the inline curly braces `{}` directly followed by variants inside () without space between.
5. Ensure that the **final text** is clear, accurate, relevant, and stylistically appropriate for the INITIAL TASK.
"""
            #system_prompt = """Refine the **ANNOTATED ANSWER** given inline text annotated instructions (format: \intruction_type{text selection}[optional comment]). Directly answer with the updated answer aligned with annotated instructions tags, no introduction."""
            user_prompt = f"### ANNOTATED ANSWER:\n{content_annotated}"
            smart_print(f"***** PROMPT:\n{system_prompt}\n `\n{user_prompt}".replace("\\n", "\n"), self.agent_name, "PROMPT")
            llm_output = str(premium_llm_function.invoke([SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]))
            smart_print(f"***** REFINED ANSWER:\n\033[33m{llm_output}\033[0m\n".replace("\\n", "\n"),
                        self.agent_name, "REFINED ANSWER")
            if smart_input("Is the task refinement adequate? (yes/no): ", self.agent_name).strip().lower() in ["yes","y"]:
                inference_result_msg.content = llm_output.content
                break
        return comments

    def modifyAnswer(self, inference_result_msg):
        inference_result_msg.content = _visual_input(inference_result_msg.content)
        smart_print(f"***** NEW USER MESSAGE:\n{inference_result_msg.content}\n*************", self.agent_name,
                    "NEW USER MESSAGE")

    # staticmethod get my host ID
    @staticmethod
    def get_host_id():
        import socket, uuid
        return socket.gethostname() + "-" + str(uuid.getnode())

    # staticmethod call llm_function (langchain ChatOpenAI) with optional function_call and process function call until result is provided
    @staticmethod
    def call_llm_function_with_function_call(llm_function, messages, function_call=None, function_list=None,
                                             max_calls=5):
        # test if HumanLLMMonitor.function_list exists
        if function_list is None:
            if HumanLLMMonitor.function_list is None:
                function_list = [{
                    "name": "search_for_external_knwoledge",
                    "description": "Call this function to search for external knowledge when the model's confidence is low or its information might be too outdated",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "description": {"type": "string",
                                            "description": "precise description of the information to be provided"},
                            "url": {"type": "string",
                                    "description": "Google search url link find this information (ie. it should start by https://www.google.com/search?q= )"},
                        },
                        "required": ["description", "url"],
                    }
                }]
            else:
                function_list = HumanLLMMonitor.function_list

        calls = 0
        while True and calls < max_calls:
            output = llm_function(messages=messages, functions=function_list, function_call="auto")
            calls += 1
            if output.content is not None and len(output.content) > 0 and output.content != "''":
                break
            # test if 1 more function_call is requested
            elif output.additional_kwargs is not None:
                additional_kwargs = output.additional_kwargs
                function_call_info = additional_kwargs.get('function_call', {})
                function_name = function_call_info.get('name', '')
                function_args_str = function_call_info.get('arguments', '')
                function_args = json.loads(function_args_str)
                if function_name in globals():
                    # Call the function with the provided arguments
                    result = globals()[function_name](**function_args)
                else:
                    result = smart_input(f"Uknown function: {function_name} -- please provide result manually:\n","call_llm_function_with_function_call")
                messages.append(FunctionMessage(content=result, name=function_name, arguments=function_args_str))

        return output

    def _log_entry(self, function_name, input_contents, output_contents, input_modified=False,
                   skipped_inference=False, input_comments=None, output_comments=None, output_llm_raw=None,
                   output_modified=False, inference_time=None, user_score=None, message_tokens=None, score=None, use_premium_llm=False,
                   call_duration=None, skip_rounds=None, synthesize_mode=False, pipeline_mode=False):
        entry = {
            "input_contents": input_contents,
            "output_contents": output_contents,
            "output_llm_raw": output_llm_raw,
            "input_comments": input_comments,
            "output_comments": output_comments,
            "inference_time": inference_time,
            "score": score,
            "skip_rounds": skip_rounds,
            "message_tokens": message_tokens,
            "before_inference_option_times": self.before_inference_option_times,
            "before_inference_option_counts": self.before_inference_option_counts,
            "after_inference_option_times": self.after_inference_option_times,
            "after_inference_option_counts": self.after_inference_option_counts,
            "call_duration": call_duration,
            "synthesize_mode": synthesize_mode,
            "pipeline_mode": pipeline_mode,
            "user_score": user_score
        }
        #print(f"Human modifications ? input_modified:{input_modified}, output_modified:{output_modified}\nlog entry: {entry}")

        # Serialize the entry as a JSON string
        serialized_entry = json.dumps(entry, default=lambda o: o.__dict__ if hasattr(o, '__dict__') else str(o))

        import socket
        import uuid

        # Log entry into the common vector database with tags
        tags = {
            "time": datetime.now().isoformat(),
            "host": HumanLLMMonitor.get_host_id(),
            "step_id": HumanLLMMonitor.step_id,
            "input_modified": input_modified,
            "output_modified": output_modified,
            "system_prompt": self.system_prompt,
            "agent_name": self.agent_name,
            "function_name": function_name,
            "skipped_inference": skipped_inference,
            "skip_rounds": skip_rounds,
            "pipeline_mode": pipeline_mode,
            "use_premium_llm": use_premium_llm,
            "commented": (input_comments is not None or output_comments is not None),
            "scored": (score is not None),
        }
        # add to tags every key of self.before_inference_option_times with count and time, if count > 0
        for key in self.before_inference_option_times:
            if self.before_inference_option_counts[key] > 0:
                tags[f"b{key}_time"] = self.before_inference_option_times[key]
                tags[f"b{key}_count"] = self.before_inference_option_counts[key]
        # add to tags every key of self.after_inference_option_times with count and time, if count > 0
        for key in self.after_inference_option_times:
            if self.after_inference_option_counts[key] > 0:
                tags[f"a{key}_time"] = self.after_inference_option_times[key]
                tags[f"a{key}_count"] = self.after_inference_option_counts[key]

        HumanLLMMonitor._check_and_init_vector_db()

        HumanLLMMonitor.common_vectordb.add_texts(
            texts=[serialized_entry],
            metadatas=[tags]
        )

    #    def CallHumanLLM(self, original_input_messages=None, llm_function=None, premium_llm_function=None, callable_system_message=None, system_prompt_template=None, user_message=None, return_message_content_only=True, function_calling=False, temperature=0.7, timeout_seconds=90, stream_output=True):
    def CallHumanLLM(self, original_input_messages=None, default_llm_function=None, premium_llm_function=None,
                     callable_system_message=None, system_prompt_template=None, user_message=None,
                     return_message_content_only=True, function_calling=False, temperature=None, timeout_seconds=300,
                     stream_output=False, use_default_llm=True, optuna=None, model_choice=None, temperature_increase=0.05):
        #if not self.selected_llm_or_chain: raise ValueError("No LLM or chain selected for use.")
        # Define a helper function to perform the LLM calls for parallel inference.
        def perform_llm_call(input_msg, use_premium, func_calling, temperature=None, stream_output=True, color_id=None):
            if use_premium:
                func = premium_llm_function if not func_calling else HumanLLMMonitor.call_llm_function_with_function_call
            else:
                func = default_llm_function if not func_calling else HumanLLMMonitor.call_llm_function_with_function_call
            if temperature:
                func = func.with_config(configurable={"llm_temperature": temperature})

            if stream_output:
                if color_id is None or color_id <= 0:
                    start_color, end_color = "", ""
                else:
                    start_color, end_color = \
                        ["\033[91m", "\033[92m", "\033[93m", "\033[94m", "\033[95m", "\033[96m", "\033[97m"][
                            color_id % 7], "\033[0m"
                final_output = ""  # Initialize an empty string to hold the full response
                smart_print("", self.agent_name, "Inference streaming output")
                previous_chunk_str = ""
                # Regex to match the end of a typical JSON structure
                json_trail_re = re.compile(r'[\'\}\]]$')
                for chunk in func.stream(input_msg):  #, temperature=temperature):  # Ensure 'llm' is correctly initialized with temperature
                    if hasattr(chunk, 'content'):
                        chunk_content = chunk.content
                        final_output += chunk_content
                    else:
                        # Convert the current chunk to string
                        current_chunk_str = str(chunk)
                        # Check and remove trailing characters for accurate comparison
                        while json_trail_re.search(current_chunk_str):
                            current_chunk_str = current_chunk_str[:-1]
                        
                        # Find the new part by removing the common prefix with the previous state
                        new_part_index = len(previous_chunk_str)
                        chunk_content = current_chunk_str[new_part_index:]
                        # Store the current chunk as the previous one for the next iteration
                        previous_chunk_str = current_chunk_str
                        final_output = str(chunk)
                    if HumanLLMMonitor.use_websocket:
                        smart_print(chunk_content, self.agent_name, "Inference streaming output "+str(color_id), append=True)
                    else:
                        smart_print(start_color + chunk_content + end_color, self.agent_name, "Inference streaming output "+str(color_id), append=True)
                return AIMessage(content=final_output)  # Return the concatenated full respons
            else:
                value = func.invoke(input_msg)
                return AIMessage(content=value.content if hasattr(value, 'content') else str(value))

        smart_print(
            f"\033[{self.print_color}m****{self.agent_name}>{inspect.stack()[1].function} calling HumanLLMMonitor****\033[0m",
            self.agent_name, "HumanLLMMonitor")
        if system_prompt_template: self.system_prompt = system_prompt_template
        if default_llm_function is None: default_llm_function = self.default_llm if use_default_llm else self.premium_llm
        if premium_llm_function is None: premium_llm_function = self.premium_llm if self.premium_llm else None
        if original_input_messages is None: original_input_messages = [
            SystemMessage(content=self.load_prompt(agent_name=self.agent_name, prompt=system_prompt_template)), HumanMessage(content=user_message)]
        input_contents_str0, input_contents_str1 = str(original_input_messages[0].content), str(
            original_input_messages[1].content)

        self.before_inference_option_times = {'TOTAL': 0, 'SELECTION': 0}  # then each option will be added to this dict
        self.before_inference_option_counts = {'TOTAL': 0,
                                               'SELECTION': 0}  # then each option will be added to this dict
        self.after_inference_option_times = {'TOTAL': 0, 'SELECTION': 0}  # then each option will be added to this dict
        self.after_inference_option_counts = {'TOTAL': 0, 'SELECTION': 0}  # then each option will be added to this dict
        call_start_time = time.time()

        while True:
            if self.skip_rounds > 0:
                smart_print(
                    f"\033[{self.print_color}m****{self.agent_name}>{inspect.stack()[2].function} skipping HumanLLMMonitor for {self.skip_rounds} rounds****\033[0m",
                    self.agent_name, "Skipping round")

            # Pre-inference human intervention
            llm_input_messages, input_comments, skip_inference, use_premium_llm, default_llm_function, premium_llm_function, function_calling = self._before_inference(
                original_input_messages, default_llm_function, premium_llm_function, function_calling,
                callable_system_message, optuna=optuna, model_choice=model_choice)
            start_time = datetime.now()
            self.last_inference_check_results = [None] * self.num_parallel_inferences  # Pre-allocate the list with None
            if llm_input_messages and not skip_inference:
                # Use concurrent futures to parallelize the LLM calls.
                outputs = []

                with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_parallel_inferences) as executor:
                    if type(self.premium_llm if use_premium_llm else self.default_llm) == type(self.llmORchains_list.get('3_majority_chain')):
                        stream_output = True
                    futures = [executor.submit(perform_llm_call, llm_input_messages, use_premium_llm, function_calling,
                                               ((temperature+(i*temperature_increase) if temperature>0 else 0) if temperature else None), stream_output, i) for i in
                               range(self.num_parallel_inferences)]
                    for idx, future in enumerate(futures):
                        try:
                            llm_response = future.result(timeout=timeout_seconds)
                            outputs.append(llm_response)
                            smart_print(
                                f'\033[0m**** New inference result recieved and added to outputs as #{len(outputs)}\033[0m:\n{llm_response.content}\n\033[9mEND OF #{len(outputs)}****\033[0m',
                                self.agent_name, "NEW inference result recieved", column_id=idx, column_max=self.num_parallel_inferences)
                        except concurrent.futures.TimeoutError:
                            smart_print('A task ran longer than the allotted timeout and was cancelled.',
                                        self.agent_name, "Inference result TIMEOUT")
                        except Exception as exc:
                            smart_print(f'Generated an exception: {exc}', self.agent_name, "Inference result EXCEPTION")
                    # Wait for all the futures to complete before continuing.
                    concurrent.futures.wait(futures)
                if len(outputs) == 0:
                    smart_print(f'**** No inference result recieved, set output to None', self.agent_name,
                                "NO inference recieved")
                    llm_outputs = None
                elif len(outputs) == 1:
                    # smart_print(f'**** One inference result recieved, set output to it', self.agent_name, "ONE inference recieved")
                    llm_outputs = outputs
                else:
                    if self.synthesize_mode and len(
                            outputs) > 1:  #NEW/UPDATED: TODO: allow to exclude code synthesis with (self.synthesize_mode or (self.skip_rounds > 0 and self.agent_name == "Coder"))
                        synthesized_response = self.synthesize_responses([output.content for output in outputs],
                                                                         use_default_llm)  #NEW
                        llm_outputs = [AIMessage(content=synthesized_response.content)]  #NEW
                        smart_print(f'**** {len(outputs)} inference results received, THEN SYNTHETISED to 1',
                                    self.agent_name, "MULTIPLE to 1 SYNTHESIS (similar to Mixture of Agents)")  #NEW
                    else:  #UPDATED
                        smart_print(
                            f'**** {len(outputs)} inference results received - You will be requested to select which ones to keep',
                            self.agent_name, "MULTIPLE inferences received")
                        llm_outputs = outputs
            else:  # Skip the LLM inference.
                # Ensure skip_inference is a string
                if not isinstance(skip_inference, str):
                    skip_inference = str(skip_inference)

                llm_outputs = [AIMessage(content=skip_inference)]
            end_time = datetime.now()
            raw_llm_outputs = [(output.content if output else None) for output in llm_outputs] if isinstance(
                llm_outputs, list) else None

            output_messages, output_comments, score = [], [], []
            if llm_outputs:
                if len(llm_outputs) > 1:
                    smart_print("**** Multiple LLM ANSWERS > we will process POST INFERENCE for each ****",
                                self.agent_name, "Multiple LLM ANSWERS", append=True)
                    init_skip_rounds = self.skip_rounds  # save the current skip_rounds value because multiple outputs decrease skip rounds for each parallel output
                for counter, llm_output in enumerate(llm_outputs, start=1):
                    if len(llm_outputs) > 1:
                        self.skip_rounds = init_skip_rounds
                        smart_print(f"\033[31mMULTI-INFERENCE OUTPUT #{counter} > \033[0m", self.agent_name,
                                    "POST INFERENCE", append=True)
                    self.current_inference_context = {
                        'function_name': inspect.stack()[1].function,
                        'input_contents': llm_input_messages,
                        'output_contents': output_messages,
                        'start_time': start_time,  # Store start_time instead of computing inference_time
                        'input_modified': ((llm_input_messages[0].content + "\n" + llm_input_messages[1].content) != (
                                input_contents_str0 + "\n" + input_contents_str1)),
                        'skipped_inference': skip_inference,
                        'input_comments': input_comments,
                        'output_comments': output_comments,
                        'raw_llm_outputs': raw_llm_outputs,
                        'output_modified': [output_message.content != raw for output_message, raw in
                                            zip(output_messages, raw_llm_outputs)],
                        'message_tokens': None,  # You may want to calculate this
                        'use_premium_llm': use_premium_llm
                    }
                    # Post-inference human intervention
                    output_messages_instance, output_comments_instance, score_instance = self._after_inference(
                        llm_output, premium_llm_function=premium_llm_function, output_id=counter,
                        outputs_count=len(llm_outputs), optuna=optuna)
                    output_messages.append(output_messages_instance)
                    if output_messages_instance == -1:
                        break
                    output_comments.append(output_comments_instance)
                    score.append(score_instance)
                # test if any of output_messages instance != -1, break if True
                if any([output_messages_instance == -1 for output_messages_instance in output_messages]):
                    original_input_messages[0].content, original_input_messages[
                        1].content = input_contents_str0, input_contents_str1
                else:
                    break

                    # Get the calling function's name using inspect
        caller_function_name = inspect.stack()[1].function

        call_duration = time.time() - call_start_time

        # Logging
        self._log_entry(
            function_name=caller_function_name,
            input_contents=llm_input_messages,
            output_contents=output_messages,
            inference_time=(end_time - start_time).total_seconds(),
            input_modified=((llm_input_messages[0].content + "\n" + llm_input_messages[1].content) != (
                    input_contents_str0 + "\n" + input_contents_str1)),
            skipped_inference=True if skip_inference else False,
            skip_rounds=self.skip_rounds,
            input_comments=input_comments,
            output_comments=output_comments,
            output_llm_raw=raw_llm_outputs,
            # test if any  output_modified=(output_messages.content != raw_llm_output), 
            output_modified=any(
                output_message.content != raw for output_message, raw in zip(output_messages, raw_llm_outputs)),
            score=score,
            message_tokens=None,
            use_premium_llm=use_premium_llm,
            call_duration=call_duration,
            synthesize_mode=self.synthesize_mode
        )

        return [message.content for message in output_messages] if return_message_content_only else output_messages
