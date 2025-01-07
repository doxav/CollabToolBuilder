import inspect
import re
import threading
import urllib.request
import subprocess
import uuid

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
from requests.packages.urllib3.util.retry import Retry # type: ignore

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

from websocket_server import WebsocketServer

openai.api_key = os.environ['OPENAI_API_KEY']
if 'OPENAI_BASE_URL' in os.environ: openai.base_url = os.environ['OPENAI_BASE_URL']

ON_INPUT = False

AGENT = ''

import asyncio
import websockets

def smart_print(message: str, agent_name=None, message_type=None, append=False, column_id=None, column_max=None, optional=False):
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
        message_dict = {'message': message, 'agent_name': agent_name, 'message_type': message_type, 'append': append,
                        'column_id': column_id, 'column_max': column_max, 'optional': optional,
                        'step_id': HumanLLMMonitor.step_id}
        # convert message_dict to json
        message = json.dumps(message_dict)
        time.sleep(0.05)
        # Wait a second every 500 messages to avoid flooding the WebSocket server
        if HumanLLMMonitor.websocket_server.message_count % 500 == 0:
            time.sleep(0.5)
        HumanLLMMonitor.websocket_server.send_message(message)

    elif IN_NOTEBOOK and agent_name:
        # import AgentDisplayManager from utils.jupyter_agents_display if AgentDisplayManager is not initialized
        if 'AgentDisplayManager' not in globals():
            try:
                from utils.jupyter_agents_display import AgentDisplayManager
            except:
                print("AgentDisplayManager cannot be imported/initialized")

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


def smart_input(message: str, agent_name=None, message_type=None, column_id=None, column_max=None, optional=False):
    # Determine if running in a notebook environment
    if 'IN_NOTEBOOK' not in globals():
        try:  # test if IN_NOTEBOOK
            from IPython import get_ipython
            globals()['IN_NOTEBOOK'] = IN_NOTEBOOK = get_ipython().__class__.__name__ == 'ZMQInteractiveShell'

        except:
            globals()['IN_NOTEBOOK'] = IN_NOTEBOOK = False
    else:
        IN_NOTEBOOK = globals()['IN_NOTEBOOK']

    # Determine if using WebSocket
    if 'IN_WEBSOCKET' not in globals():
        if HumanLLMMonitor.use_websocket:
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

        # # Retrieve port and secret from WebsocketServer
        port = HumanLLMMonitor.websocket_server.port
        secret = HumanLLMMonitor.websocket_server.secret
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
            'step_id': HumanLLMMonitor.step_id
        }
        message_json = json.dumps(structured_message)

        # Send the message via WebsocketServer
        HumanLLMMonitor.websocket_server.send_message(message_json)

        async def receive_message(timeout=86400):
            async with websockets.connect(ws_url, ping_interval=30, ping_timeout=60) as websocket:
                try:
                    print("SMART INPUT Waiting for response from WebSocket")
                    while True:
                        max_retry = 10
                        response = None
                        for i in range(max_retry):
                            try:
                                response = await asyncio.wait_for(websocket.recv(), timeout=timeout)
                                break
                            except asyncio.TimeoutError:
                                print(f"Timeout waiting for response from WebSocket, retry {i + 1}/{max_retry}")
                                await asyncio.sleep(1)
                            except Exception as e:
                                print(f"An error occurred: {e}, retry {i + 1}/{max_retry}")
                                await asyncio.sleep(1)
                        print("SMART INPUT Received response from WebSocket")
                        try:
                            # Check that the response is not a NoneType
                            if response is None:
                                print("Received None response from WebSocket")
                                return None
                            json_data = json.loads(response)
                        except json.JSONDecodeError as e:
                            print(f"Error decoding JSON: {e}")
                            return None
                        # Ignore messages from self
                        if 'sender_id' in json_data and json_data[
                            'sender_id'] == HumanLLMMonitor.websocket_server.server_id:
                            print("Received message from self, ignoring")
                            continue

                        # Handle function results
                        if 'result' in json_data:
                            print("Received function result, ignoring")
                            json_data['sender_id'] = HumanLLMMonitor.websocket_server.server_id
                            response = json.dumps(json_data)
                            HumanLLMMonitor.websocket_server.send_notasync_message(response)
                            continue
                        else:
                            break
                    if 'message' in json_data:
                        answer = json_data['message']
                    elif 'result' in json_data:
                        answer = json_data['result']
                    else:
                        raise ValueError("No 'message' or 'result' in WebSocket response")
                    print("SMART INPUT Answer: ", answer)
                    return str(answer).upper()
                except asyncio.TimeoutError:
                    print("Timeout waiting for response from WebSocket")
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
            initial_string_serialized = initial_string

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
    unique_collection_id = None

    @classmethod
    def set_unique_collection_id(cls, unique_id):
        cls.unique_collection_id = unique_id

    @staticmethod
    def check_db():
        if UnifiedVectorDB.db_connection_check_done is True:
            return
        if UnifiedVectorDB.db_type == 'elasticsearch':
            session = requests.Session()
            retry = Retry(total=5, backoff_factor=1)
            adapter = HTTPAdapter(max_retries=retry)
            session.mount("https://", adapter)
            auth = HTTPBasicAuth(UnifiedVectorDB.es_user,
                                 UnifiedVectorDB.es_password) if UnifiedVectorDB.es_user else None
            try:
                response = session.get(UnifiedVectorDB.es_url, auth=auth, timeout=5, verify=False)
                response.raise_for_status()
                print("Elasticsearch response:", response.text)
                UnifiedVectorDB.db_connection_check_done = True
            except requests.exceptions.RequestException as e:
                print(f"Error: {e}\nURL: {UnifiedVectorDB.es_url}\nCheck Elasticsearch and credentials. UnifiedVectorDB.es_user:{UnifiedVectorDB.es_user}, UnifiedVectorDB.es_password:{UnifiedVectorDB.es_password}")
                exit(1)
        elif UnifiedVectorDB.db_type == 'chroma':
            print("Chroma DB check is not yet implemented")
            UnifiedVectorDB.db_connection_check_done = True
        else:
            raise ValueError(f"Unsupported DB type: {UnifiedVectorDB.db_type}")

    def __init__(self, collection_name, embedding_function, persist_directory, reset_db_indices=False):
        UnifiedVectorDB.check_db()
        self.collection_name = collection_name.lower()
        if UnifiedVectorDB.unique_collection_id is not None:
            self.collection_name += f"_{UnifiedVectorDB.unique_collection_id}"
            self.collection_name = self.collection_name.lower()
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
                                           http_auth=(UnifiedVectorDB.es_user,
                                                      UnifiedVectorDB.es_password) if (UnifiedVectorDB.es_user not in [False, "", None]) else None,
                                           verify_certs=True, ssl_show_warn=False)
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
                    if key == '_id':
                        # Use the 'ids' query for filtering on '_id'
                        if isinstance(value, list):
                            custom_filter_es.append({"ids": {"values": value}})
                        else:
                            custom_filter_es.append({"ids": {"values": [value]}})
                    elif isinstance(value, dict) and any(k in value for k in ['gte', 'lte', 'gt', 'lt']):
                        # Handle range queries (e.g., date ranges)
                        custom_filter_es.append({"range": {f"{key}": value}})
                    elif isinstance(value, list):
                        # Use 'terms' query for multiple values
                        custom_filter_es.append({"terms": {f"metadata.{key}": value}})
                    else:
                        # Use 'match' query for single value
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
                return self.db.similarity_search(query_text, k=(k if k <= 50 else 50), filter=custom_filter_es)  # k seems to crash when > 50
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

    def run_check(self, output_id, *args, **kwargs):
        return self.check_function(*args, **kwargs, output_id=output_id)


class HumanLLMMonitor:
    default_skip_rounds = 0
    step_id = 0
    function_list = None
    common_vectordb = None
    common_vectordb_embedding_function = None  #OpenAIEmbeddings(model=UnifiedVectorDB.OpenAI_embedding_function_name, deployment=UnifiedVectorDB.OpenAI_embedding_function_name)  #HuggingFaceEmbeddings(model_name="intfloat/e5-base-v2", encode_kwargs={"normalize_embeddings": True}) # TODO: set as a parameter

    common_vectordb_collection_name = "human_llm_monitor_logs"
    common_vectordb_persist_directory = "human_llm_monitor_vectordb"
    websocket_server = None
    use_websocket = True
    ws_thread = None
    stop_event = threading.Event()

    # Initialize vector databases for tasks
    db_collection_success="successful_tasks"
    db_collection_failed="failed_tasks"
    reset_db_indices = False # Set to True if you want to reset the database indices
    db_learnt_tasks = None
    db_failed_tasks = None

    user_id = None

    @classmethod
    def add_agent_data(cls, agent_name, data_key, data_value, function_name=None, id_task=False,
                       before_after=None, user_id=None, step_id=None, type_tache=None, score=None, metadata=None):
        """Stores agent-specific data with additional metadata.
        Elasticsearch generates an 'id' automatically and includes it in the metadata.
        """
        if isinstance(data_value, list) and len(data_value) > 0:
            if isinstance(data_value[0], AIMessage):
                data_value = data_value[0].content
        if isinstance(data_value, dict):
            serialized_data = json.dumps(data_value)
        else:
            serialized_data = json.dumps({data_key: data_value})

        # Generate UUID for id_task
        id_task = str(uuid.uuid4()) if id_task else False

        tags = metadata or {}
        tags.update({
            "agent_name": agent_name,
            "data_key": data_key
        })
        print(f"Adding agent data: {tags}")
        print(f"User ID: {user_id}")
        if function_name:
            tags["function_name"] = function_name
        if id_task:
            tags["id_task"] = id_task
        if before_after:
            tags["before_after"] = before_after
        if user_id:
            tags["user_id"] = user_id
        if step_id:
            tags["step_id"] = step_id
        if type_tache:
            tags["type_tache"] = type_tache
        if score is not None:
            tags["score"] = score
        tags["date"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")

        cls.common_vectordb.add_texts(texts=[serialized_data], metadatas=[tags])

    @classmethod
    def get_agent_data(cls, agent_name=None, data_key=None, id_task=None, function_name=None, before_after=None, user_id=None,
                       step_id=None, type_tache=None, score=None,
                       metadata_filter=None, sort_order=None, k=5, start_index=0, end_index=None):
        """Retrieves agent-specific data based on the agent name, data key, and additional metadata.
        Supports pagination by specifying start and end indices.
        """
        metadata = {}
        if agent_name is not None:
            metadata["agent_name"] = agent_name
        if data_key is not None:
            metadata["data_key"] = data_key
        if function_name is not None:
            metadata["function_name"] = function_name
        if id_task is not None:
            metadata["id_task"] = id_task
        if before_after is not None:
            metadata["before_after"] = before_after
        if user_id is not None:
            metadata["user_id"] = user_id
        if step_id is not None:
            metadata["step_id"] = step_id
        if type_tache is not None:
            metadata["type_tache"] = type_tache
        if score is not None:
            metadata["score"] = score
        if metadata_filter:
            metadata.update(metadata_filter)

        # Fetch results with a large 'k' to ensure we have enough data
        max_k = end_index if end_index is not None else k
        results = cls.common_vectordb.query(
            query_text='*', metadata_filter=metadata, sort_order=sort_order, k=max_k)

        # Apply pagination
        paginated_results = results[start_index:end_index] if end_index is not None else results[start_index:]

        ret = []
        for item in paginated_results:
            temp = json.loads(item.page_content)
            if isinstance(temp, dict):
                tmp = {}
                for key in temp:
                    if temp[key]:
                        tmp[key] = temp[key]
                ret.append(tmp)
            else:
                ret += temp[data_key]
        # Ret contains only text field of the data, results contains all the metadata
        return ret, results

    @classmethod
    def get_tasks(cls, page_size=200, nb_pages=1, id_last_task=None):
        """Retrieves saved tasks using get_agent_data with pagination.
        Parameters:
            page_size (int): Number of results per page. Default is 200.
            nb_pages (int): Number of pages to retrieve. Default is 1.
            id_last_task (str): ID of the last task retrieved. If provided, retrieves tasks after this ID.
        Returns:
            str: List of tasks in json format.
        """
        data_key = "saved_task"

        start_index = 0
        end_index = page_size * nb_pages

        _, tasks = cls.get_agent_data(
            data_key=data_key,
            k=end_index,
            start_index=start_index,
            end_index=end_index
        )
        # Check if there is a newer task (if id_last_task is not the last task of the list)
        if id_last_task and id_last_task != "None":
            modif = False
            for i, task in enumerate(tasks):
                if task.metadata["id_task"] == id_last_task and i + 1 < len(tasks):
                    tasks = tasks[i+1:]
                    modif = True
                    break
            if not modif:
                return "None"
        ret = []


        for task in tasks:
            ret += [{
                "id_task": task.metadata["id_task"],
                "agent_name": task.metadata["agent_name"],
                "before_after": task.metadata["before_after"],
                "date": task.metadata["date"]
            }]
            if "score" in task.metadata:
                ret[-1]['score'] = task.metadata['score']
            if "input_contents" in task.metadata:
                ret[-1]['input_contents'] = task.metadata['input_contents']
            if "user_id" in task.metadata:
                ret[-1]['user_id'] = task.metadata['user_id']
            if "function_name" in task.metadata:
                ret[-1]['function_name'] = task.metadata['function_name']
            if "step_id" in task.metadata:
                ret[-1]['step_id'] = task.metadata['step_id']
            if "type_tache" in task.metadata:
                ret[-1]['type_tache'] = task.metadata['type_tache']

        # Trier la liste par la clé 'date', du plus récent au plus ancien
        ret = sorted(ret, key=lambda x: x['date'], reverse=True)

        return json.dumps(ret)

    @classmethod
    def goto_task(cls, id_task: str, automatic: str = None, special_criteria : dict =  None):
        """
        Retrieve the task from the database and start processing the task.
        Parameters:
            id_task (str): The identifier of the task to retrieve from the database.
            automatic (str): If True, the function will run the loop in automatic mode.
            special_criteria (dict): The special criteria to use for the task.
        """
        import subprocess
        # Retrieve the task from the database
        _, saved_task = HumanLLMMonitor.get_agent_data(data_key="saved_task", id_task=id_task)
        task = {
            'before_after': saved_task[0].metadata['before_after'],
            'agent_name': saved_task[0].metadata['agent_name'],
            'type_tache': saved_task[0].metadata['type_tache'],
            'content': saved_task[0].page_content,
            'step_id': saved_task[0].metadata['step_id'],
            'date': saved_task[0].metadata['date']
        }
        if 'user_id' in saved_task[0].metadata:
            task['user_id'] = saved_task[0].metadata['user_id']

        # Create a pickle directory if it does not exist
        if not os.path.exists('pickle'):
            os.makedirs('pickle')

        # Serializing variables to pickle file
        variables_to_pickle = {
            'saved_task': task,
            'automatic': bool(automatic),
            'unique_id': UnifiedVectorDB.unique_collection_id,
            'special_criteria': special_criteria
        }

        # Save variables to pickle file
        filename = f"variables_{HumanLLMMonitor.user_id}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        with open(f"pickle/{filename}.pkl", 'wb') as f:
            pickle.dump(variables_to_pickle, f)

        if not os.path.exists('goto_output'):
            os.makedirs('goto_output')

        with open(f'./goto_output/output_{id_task}_{HumanLLMMonitor.user_id}.log', 'w') as f:
            f.write("")

        # Execute the bash command with unbuffered output and capture its output
        process = subprocess.Popen(['bash', '-c', f'python3 -u learn.py --proxy --secret --pickle_name {filename} > ./goto_output/output_{id_task}_{HumanLLMMonitor.user_id}.log 2>&1'],
                                   stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT,
                                   text=True)
        # Initialize the link variable
        link = None
        with open(f"./goto_output/output_{id_task}_{HumanLLMMonitor.user_id}.log", "r") as logfile:
            # Move to end of file
            logfile.seek(0, 2)  # 2 means "from the end of the file"

            # Wait for the link to be found in the output file
            while True:
                line = logfile.readline()
                if not line:
                    # If no new line is found, wait for 0.5 seconds
                    time.sleep(0.5)
                    continue

                if "WebSocket Remote URL via proxy: " in line:
                    link = line.split("WebSocket Remote URL via proxy: ")[1].strip()
                    print(f"Link: {link}")
                    break

        # Return the link if found
        if link is None:
            print("The link was not found in the output file.")

        return link

    @classmethod
    def set_id(cls, user_id):
        print(f"Setting user_id to {user_id}")
        cls.user_id = user_id

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

    @classmethod
    def check_init_class_db(cls, force=False):
        if cls.common_vectordb_embedding_function is None:
            raise ValueError("embeddingfunction must be set to allow HumanLLMMonitor to manage tasks and other memories")
        if cls.db_learnt_tasks is None or force:
            cls.db_learnt_tasks = UnifiedVectorDB(
                collection_name=cls.db_collection_success,
                embedding_function=cls.common_vectordb_embedding_function,
                persist_directory=cls.common_vectordb_persist_directory+cls.db_collection_success,
                reset_db_indices=cls.reset_db_indices)
        if cls.db_failed_tasks is None or force:
            cls.db_failed_tasks = UnifiedVectorDB(collection_name=cls.db_collection_failed,
                embedding_function=cls.common_vectordb_embedding_function,
                persist_directory=cls.common_vectordb_persist_directory+cls.db_collection_failed,
                reset_db_indices=cls.reset_db_indices)
        # if cls.common_vectordb is None or force:
        #     cls.common_vectordb = UnifiedVectorDB(
        #         collection_name=cls.common_vectordb_collection_name,
        #         embedding_function=cls.common_vectordb_embedding_function,
        #         persist_directory=cls.common_vectordb_persist_directory,
        #         reset_db_indices=cls.reset_db_indices)

    @classmethod
    def get_learnt_tasks(cls, query_text="*", k=10, metadata_filter=None, sort_order=None, similarity_search=False):
        cls.check_init_class_db()
        if similarity_search:
            results = cls.db_learnt_tasks.similarity_search_with_score(query=query_text, k=k)
        else:
            results = cls.db_learnt_tasks.query(
                query_text=query_text, k=k, metadata_filter=metadata_filter, sort_order=sort_order
            )
        return {result.page_content for result in results}

    @classmethod
    def get_failed_tasks(cls, query_text="*", k=10, metadata_filter=None, sort_order=None, similarity_search=False):
        cls.check_init_class_db()
        if similarity_search:
            results = cls.db_failed_tasks.similarity_search_with_score(query=query_text, k=k)
        else:
            results = cls.db_failed_tasks.query(
                query_text=query_text, k=k, metadata_filter=metadata_filter, sort_order=sort_order
            )
        return {result.page_content for result in results}

    @classmethod
    def get_validation_results(cls, query_text="*", k=10, sort_order=None, similarity_search=False):
        cls.check_init_class_db()
        metadata_filter = {'agent_name': 'ValidationAgent'}
        if similarity_search:
            results = cls.common_vectordb.similarity_search_with_score(
                query=query_text, k=k, metadata_filter=metadata_filter
            )
        else:
            results = cls.common_vectordb.query(
                query_text=query_text, k=k, metadata_filter=metadata_filter, sort_order=sort_order
            )
        return {result.page_content for result in results}

    @classmethod
    def get_multiple_few_shots(cls, few_shots_params) -> str:
        if not few_shots_params:
            return ""


        all_formatted_examples = []

        for params in few_shots_params:
            # Ensure default values
            criteria_defaults = {
                'sources': 'learnt',
                'num': 5,
                'query_text': '*',
                'metadata_filter': {},
                'sort_order': None,
                'similarity_search': False,
                'format': 'Json',
                'template': None
            }
            for key, default in criteria_defaults.items():
                params.setdefault(key, default)

            if "separators" in params:
                separators = params["separators"]
            else:
                separators = {
                    "global_prefix" : f"\n{params['sources']} tasks : <<",
                    "global_suffix" : ">>\n",
                    "item_prefix" : "\n|",
                    "item_suffix" : "|"
                }

            if params['sources'] == "learnt":
                examples = cls.get_learnt_tasks(
                    query_text=params['query_text'],
                    k=params['num'],
                    metadata_filter=params['metadata_filter'],
                    sort_order=params['sort_order'],
                    similarity_search=params['similarity_search']
                )
            elif params['sources'] == "failed":
                examples = cls.get_failed_tasks(
                    query_text=params['query_text'],
                    k=params['num'],
                    metadata_filter=params['metadata_filter'],
                    sort_order=params['sort_order'],
                    similarity_search=params['similarity_search']
                )
            else:
                examples = HumanLLMMonitor.common_vectordb.query(
                    query_text=params['query_text'],
                    k=params['num'],
                    metadata_filter=params['metadata_filter'],
                    sort_order=params['sort_order']
                )
                params['sources'] = "examples"

            # Process examples to retrieve all metadata
            processed_examples = []
            for example in examples:
                example_data = {
                    'content': getattr(example, 'page_content', str(example)),
                    'metadata': getattr(example, 'metadata', {})
                }
                processed_examples.append(example_data)
            examples = processed_examples

            format_criteria = {
                'format': params['format'],
                'template': params['template']
            }

            formatted_examples = cls.format_examples(examples, format_criteria, separators)
            all_formatted_examples.append(formatted_examples)

        return "\n".join(all_formatted_examples)

    @classmethod
    def format_examples(cls, examples, criteria, separators):
        if not examples:
            return ""

        if isinstance(criteria, str):
            criteria = {'format': criteria}

        output_format = criteria.get('format', 'Json')
        template_str = criteria.get('template')

        formatted_examples = []

        for entry in examples:
            # For 'Jinja2' format, use the entire example (content and metadata)
            if output_format.lower() == 'jinja2':
                content_data = entry
            else:
                # Extract content appropriately
                content_str = entry.get('content', '')
                try:
                    content_data = json.loads(content_str)
                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON: {e}")
                    continue

            if output_format.lower() == 'json':
                formatted_example = json.dumps(content_data, indent=2)

            elif output_format.lower() == 'markdown':
                formatted_example = cls.dict_to_markdown(content_data)

            elif output_format.lower() == 'jinja2':
                if not template_str:
                    # Provide a default template if none is specified
                    template_str = cls.get_default_jinja2_template(content_data)
                try:
                    from jinja2 import Template
                    template = Template(template_str)
                    formatted_example = template.render(**content_data)
                except Exception as e:
                    print(f"Error rendering Jinja2 template: {e}")
                    continue
            else:
                print(f"Unsupported format or missing template for '{output_format}'.")
                continue

            formatted_examples.append(formatted_example)

        item = ''.join([f"{separators['item_prefix']}{format_ex}{separators['item_suffix']}" for format_ex in formatted_examples])

        return f"{separators['global_prefix']}{item}{separators['global_suffix']}"

    @classmethod
    def get_default_jinja2_template(cls, content_data):
        # Generates a default Jinja2 template by recursively listing all keys and their values
        def generate_template_lines(data, parent_key=''):
            lines = []
            for key, value in data.items():
                full_key = f"{parent_key}{key}"
                if isinstance(value, dict):
                    lines.extend(generate_template_lines(value, f"{full_key}."))
                else:
                    lines.append(f"{full_key}: {{{{ {full_key} }}}}")
            return lines

        template_lines = generate_template_lines(content_data)
        return "\n".join(template_lines)

    @staticmethod
    def dict_to_markdown(content):
        markdown_lines = []
        for key, value in content.items():
            markdown_lines.append(f"**{key}**: {value}")
        return "\n".join(markdown_lines)

    @classmethod
    def add_learnt_task(cls, serialized_entry, tags):
        cls.db_learnt_tasks.add_texts(texts=[serialized_entry], metadatas=[tags])

    @classmethod
    def add_failed_task(cls, serialized_entry, tags):
        cls.db_failed_tasks.add_texts(texts=[serialized_entry], metadatas=[tags])

    @classmethod
    def initialize_websocket_server(cls, port=6789, secret=None, proxy_enabled=False, unique_id=None):
        if cls.use_websocket and cls.websocket_server is None:
            cls.websocket_server = WebsocketServer(port=port, secret=secret, proxy_enabled=proxy_enabled, unique_id=unique_id)
            cls.stop_event.clear()
            cls.ws_thread = threading.Thread(target=cls.run_websocket_server)
            cls.ws_thread.daemon = True  # Run the WebSocket server in a daemon thread
            cls.ws_thread.start()

    @classmethod
    def run_websocket_server(cls):
        asyncio.run(cls.websocket_server.main(cls.stop_event))
        # cls.websocket_server.run_server(cls.stop_event)

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
        if HumanLLMMonitor.common_vectordb_embedding_function is None:
            if isinstance(embedding_function, str):
                if embedding_function in ["OpenAIEmbeddings", "text-embedding-ada-002"]:
                    HumanLLMMonitor.common_vectordb_embedding_function = OpenAIEmbeddings(model=embedding_function,
                                                                                          deployment=UnifiedVectorDB.OpenAI_embedding_function_name)
                elif embedding_function == "HuggingFaceEmbeddings":
                    HumanLLMMonitor.common_vectordb_embedding_function = HuggingFaceEmbeddings(
                        model_name="intfloat/e5-base-v2", encode_kwargs={"normalize_embeddings": True})
                else:
                    HumanLLMMonitor.common_vectordb_embedding_function = HuggingFaceEmbeddings(
                        model_name=embedding_function, encode_kwargs={"normalize_embeddings": True},
                        model_kwargs={"trust_remote_code": True})
            else:
                HumanLLMMonitor.common_vectordb_embedding_function = embedding_function

    @staticmethod
    def get_few_shots_tag_args(prompt):
        """
        Removes the 'few_shots' tag from the prompt and inserts the string received from
        get_multiple_few_shots at each location where the tag was removed.

        Returns:
            str: The modified prompt content with few shots inserted.
        """
        # Match 'few_shots' and capture the curly braces, manually handling nested braces
        pattern = r"few_shots:\s*\{"
        matches = list(re.finditer(pattern, prompt, re.DOTALL))

        for match in reversed(matches):  # Reverse to not mess up indices when replacing
            try:
                start = match.start()
                # Manually find the corresponding closing brace
                brace_count = 1
                end = start + match.end() - match.start()
                while brace_count > 0 and end < len(prompt):
                    if prompt[end] == '{':
                        brace_count += 1
                    elif prompt[end] == '}':
                        brace_count -= 1
                    end += 1

                # Extract the JSON string
                data_str = prompt[match.start() + len("few_shots:"):end]
                print(f"Attempting to decode few_shots tag: {data_str}")  # Debug
                data = json.loads(data_str)
                # Combine criteria
                combined_criteria = HumanLLMMonitor.combine_criteria([data])
                # Get the few shots string
                few_shots_str = HumanLLMMonitor.get_multiple_few_shots(combined_criteria)
                # Replace the tag with the few shots string
                prompt = prompt[:start] + few_shots_str + prompt[end:]
            except json.JSONDecodeError as e:
                print(f"Error decoding 'few_shots' tag: {e}")
                print(f"Faulty JSON: {data_str}")  # Debug
                continue

        return prompt

    def set_llmORchain(self, llm_name, is_premium=False, temperature=0.1):
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
                                    ).with_config(configurable={
                                        "llm_temperature": temperature})  # Replace with desired default temperature
                                except ValueError as e:
                                    smart_print(
                                        f"Sub-step {key} in step {step} does not support temperature configuration: {e}",
                                        self.agent_name,"set_llmORchain",optional=True)
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
                            ).with_config(configurable={
                                "llm_temperature": temperature})  # Replace with desired default temperature
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
                    ).with_config(
                        configurable={"llm_temperature": temperature})  # Replace with desired default temperature
                except ValueError as e:
                    smart_print(f"LLM/Chain '{llm_name}' does not support temperature configuration: {e}",
                                self.agent_name, optional=True)

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
                    smart_print(f"LLM/Chain '{llm_name}' does not support structured output", self.agent_name, optional=True)

            # Set the LLM/Chain to the possibly modified or original one
            if is_premium:
                self.premium_llm = selected_llm_or_chain
            else:
                self.default_llm = selected_llm_or_chain

            return True
        else:
            smart_print(
                f"LLM/Chain '{llm_name}' not found in llmORchains_list {[key for key in self.llmORchains_list]}",
                self.agent_name, optional=True)
            return False

    def set_default_llmORchain(self, llm_name, temperature=0.1):
        return self.set_llmORchain(llm_name, is_premium=False, temperature=temperature)

    def set_userid(self, user_id):
        HumanLLMMonitor.user_id = user_id
        print(f"Setting user_id to {user_id}")



    def set_premium_llmORchain(self, llm_name, temperature=0.7):
        return self.set_llmORchain(llm_name, is_premium=True, temperature=temperature)

    def set_output_schema(self, output_schema, package_path="."):
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

    def __init__(self, system_prompt=None, CPS_env_type=None, agent_name=None, model_max_context_size=16000,
                 default_llmORchain=None,
                 premium_llmORchain=None, premium_llm_by_default=False, num_parallel_inferences=1,
                 llmORchains_list=None,
                 synthesize_mode=False, inference_checks=None, output_schema=None, temperature_min=0.7,
                 temperature_max=None, envs=None,
                 fixed_coach=False, prompt_critic=None, saved_task=None, automation=None, auto_n_rounds=None,
                 recommend_critics=None, task_parameters=None):
        # Instance properties to track time
        self.task_parameters = task_parameters
        self.selected_outputs = []
        self.menu_start_time = None
        self.start_time = None
        self.current_inference_context = None
        self.user_message_few_shots = None
        if llmORchains_list is None: raise ValueError("llmORchains_list must be provided")
        self.llmORchains_list = llmORchains_list
        self.temperature_min = temperature_min
        self.temperature_max = temperature_max if temperature_max else (temperature_min + 0.2)
        self.prompt_critic = prompt_critic
        self.system_prompt = system_prompt
        self.set_output_schema(output_schema)
        self.set_default_llmORchain(
            default_llmORchain if default_llmORchain else "default_llm")  #.default_llm = default_llmORchain if default_llmORchain else self.llmORchains_list.get("default_llm")
        self.set_premium_llmORchain(
            premium_llmORchain if premium_llmORchain else "premium_llm")  #premium_llm = premium_llmORchain if premium_llmORchain else self.llmORchains_list.get("premium_llm")
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
        self.user_message = ""
        self.envs = envs
        self.fixed_coach = fixed_coach
        self.automation = automation
        self.outputs = None
        self.saved_task = saved_task
        self.auto_n_rounds = auto_n_rounds
        self.recommend_critics = recommend_critics

    # Clears the selected answers before processing new outputs.
    # This should be called at the beginning of a new inference process.
    def clear_selected_outputs(self):
        self.selected_outputs = []

    # Store the list of selected answers/outputs provided by the frontend.
    # This should be called during the after_inference process.
    def set_selected_outputs(self, selected_outputs):
        self.selected_outputs = selected_outputs

    # Retrieve the list of selected answers/outputs.
    # External agents can use this method to access the selected outputs.
    def get_selected_outputs(self):
        return self.selected_outputs

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
                    raise TypeError(
                        f"Cannot match parameters to function signature. Expected {param_count} parameters but received {type(params).__name__}.")

            self.track_time_spent(function_name)
        # Retourner une valeur par défaut si la fonction n'existe pas ou n'est pas callable
        return None

    def track_time_spent(self, action, mode=None, reset_menu_time_after=True):
        """
        Track the time spent on each menu option or action.
        
        Args:
        - action (str): The action being performed (e.g., 'A', 'B', etc.)
        - is_before (bool): True if tracking for _before_inference, False for _after_inference
        """
        if mode == 'before' or (mode is None and self.mode == 'before'):
            time_dict, count_dict = self.before_inference_option_times, self.before_inference_option_counts
        elif mode == 'after' or (mode is None and self.mode == 'after'):
            time_dict, count_dict = self.after_inference_option_times, self.after_inference_option_counts
        else:
            time_dict, count_dict = self.unidentified_option_times, self.unidentified_option_counts

        if action:
            if action not in time_dict:
                time_dict[action] = 0
                count_dict[action] = 0
            time_dict[action] += (time.time() - self.start_time)
            count_dict[action] += 1
        
        # Track total time
        time_dict["TOTAL"] += (time.time() - self.menu_start_time)
        count_dict["TOTAL"] += 1

        # Track selection time
        time_dict["SELECTION"] += (self.start_time - self.menu_start_time)
        count_dict["SELECTION"] += 1

        # Reset times for the next action
        if reset_menu_time_after:
            self.start_time, self.menu_start_time = time.time(), time.time()

    @staticmethod
    def load_prompt(prompt_name, template_data=None, directory=None):
        """
        Load a prompt or template from a file, with optional dynamic content.

        Parameters:
        - prompt_name (str): The name of the prompt file to load (without .txt extension).
        - template_data (dict): Optional dictionary for placeholder replacements in the template.
        - directory (str): Directory where prompt files are stored.

        Returns:
        - str: The content of the prompt file, with placeholders replaced if template_data is provided.
        """
        if not directory:
            template_content = prompt_name
        else :
            prompt_path = os.path.join(directory, f"{prompt_name}.txt")

            try:
                with open(prompt_path, 'r') as file:
                    template_content = file.read()

                    # Apply template data if provided
                    if template_data:
                        template_content = template_content.format(**template_data)

            except FileNotFoundError:
                raise FileNotFoundError(
                    f"The prompt file '{prompt_name}.txt' was not found in the directory '{directory}'")
            except KeyError as e:
                raise KeyError(f"Missing key {e} in template data for prompt '{prompt_name}'")

        prompt_content = HumanLLMMonitor.get_few_shots_tag_args(template_content)
        return prompt_content

    @staticmethod
    def combine_criteria(criteria_list):
        """
        Formats the criteria list into the required output format.

        Args:
            criteria_list (list): List of criteria dictionaries.

        Returns:
            list: List of formatted criteria dictionaries.
        """
        formatted_list = []
        for criteria in criteria_list:
            formatted_criteria = {
                "sources": criteria.get('sources', "default"),
                "num": criteria.get('num', 2),
                "format": criteria.get('format', "Markdown"),
                "sort_order": criteria.get('sort_order', None),
                "template": criteria.get('template', None),
            }
            formatted_list.append(formatted_criteria)
        return formatted_list

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
        llm = OpenAI(temperature=self.temperature_min)
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
            result = check.run_check(output_id, *args, **kwargs)
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
        encoding = tiktoken.encoding_for_model("gpt-4o-mini")  # TODO: replace by appropriate call to self.xxxxxx
        token_length = len(encoding.encode(content))

        if token_length > self.llm_max_context_size:
            smart_print(
                f"\033[31mCANNOT SEND MESSAGE TO LLM:\n{content}\n\nToo many tokens in human message for LLM ({token_length}). Fallback to manual feedback.\033[0m",
                self.agent_name, optional=True)
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
                          callable_system_message=None, use_premium_llm=None, model_choice=None, task_name=None):
        self.mode = 'before'
        comments = None
        initial_user_message = messages[1].content
        function_name = inspect.stack()[2].function
        use_premium_llm = use_premium_llm if use_premium_llm is not None else self.premium_llm_by_default
        forced_llm_output = False  # TODO: try to set it to None
        HumanLLMMonitor.add_agent_data(self.agent_name, "saved_task",
                                       {'prompt': messages[0].content + messages[1].content,
                                        'num_parallel_inferences': self.num_parallel_inferences,
                                        'task_parameters': self.task_parameters},
                                       before_after='before', user_id=HumanLLMMonitor.user_id, step_id=self.step_id,
                                       type_tache="IR_CPS_TechSynthesis", id_task=True)

        while self.skip_rounds <= 0:
            # MENU
            menu = ''
            before_menu = f"\033[{self.print_color}m***** {self.agent_name}->{function_name}  BEFORE *****\nSYSTEM PROMPT:\n{messages[0].content}\n\nUSER MESSAGE:\n{messages[1].content}\n***** {self.agent_name}->{function_name} BEFORE *****\033[0m\n"
            if not self._max_tokens_ok(messages[0].content + "\n" + messages[1].content):
                before_menu += ("WARNING!!!! Max tokens exceeded, you should refactor user message or system prompt!\n")
            menu += (
                "[A] Modify agent's system prompt\n")  # Modify agent's 'system prompt' (role, global context, constraints, examples) OR the answer SCHEMA output.\n")
            menu += ("[B] Give instruction or information to agent\n")
            menu += ("[C] Skip & set agent output (from recent or manually)\n")
            menu += ("[D] Log comments (not used by the model, just for information)\n")
            menu += ("[E] See previous results\n")
            menu += ("[F] See MODIFIED/SCORED/COMMENTED results\n")
            menu += ("[G] Skip for N rounds (auto mode)\n")
            menu += ("[H] Change default agent\n")
            menu += ("[I] Change premium agent\n")
            menu += (
                f"[J] Set num of parallel inferences ({self.num_parallel_inferences}, Synthesis={'ON' if self.synthesize_mode else 'OFF'})\n")  # UPDATED
            menu += ("[K] Exit\n")
            menu += (f"[P] Generate with a PREMIUM agent (default:{use_premium_llm})\n")
            menu += (f"[Z] Continue\n")

            smart_print(before_menu + menu, self.agent_name, "BEFORE inference action MENU", optional=False)
            self.menu_start_time = time.time()
            if self.automation=='coach':
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

                if self.agent_name == "TaskIdentificationAgent":
                    if self.num_parallel_inferences > 1:
                        self.synthesize_mode = True
                    if self.fixed_coach:
                        # We force the output of the llm.
                        forced_llm_output = self.fixed_coach

                self.set_default_llmORchain(new_llm_name)
                self.set_premium_llmORchain(new_llm_name)
                default_llm_function = self.default_llm
                premium_llm_function = self.premium_llm
                self.synthesize_mode = False
                # Default actions for all agents while running with optuna
                action = ""
            elif self.automation:
                action = ""
            else:  # Default case
                action = smart_input(
                    f"\033[32mBEFORE\033[0m inference @ {self.agent_name}-> Choose an action (or hit Enter for inference) :",
                    self.agent_name, optional=False).upper()

            # ACTIONS processing
            self.start_time = time.time()  # Init action selected and timer to measure time spent and occurrences in action processing

            if action == "A":  # Modify system prompt
                comments, forced_llm_output = self.modify_prompt(callable_system_message, comments,
                                                                 default_llm_function, forced_llm_output, messages,
                                                                 premium_llm_function, use_premium_llm)

                # Adding the logic to edit the output schema
                if hasattr(self, 'output_schema_path') and self.output_schema_path:
                    new_schema_content = _visual_input(open(self.output_schema_path).read(), filetype="py")
                    confirm_schema = smart_input(
                        "Do you want to replace the current output schema with your input? (y/n): ",
                        self.agent_name).upper()
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

            elif action == "E":  # See all previous results
                result = self.getPreviousResults(function_name, self.agent_name)
                _visual_input(result)

            elif action == "F":  # See previous MODIFIED/SCORED/COMMENTED results
                result = self.getScoredResults(function_name)
                if result is not None:
                    messages = [
                        SystemMessage(content=self.load_prompt(prompt_name=self.system_prompt,
                                                               directory='prompts')),
                        HumanMessage(content=result)
                    ]
                    for message in messages:
                        smart_print(message.content, self.agent_name, "BEFORE inference action MENU", optional=True)

            elif action == "G":  # Skip human actions for N rounds
                self.skipRounds()

            elif action == "H":  # Change default LLM
                default_llm_function = self.changeDefaultLLM(default_llm_function)

            elif action == "I":  # Change premium LLM
                premium_llm_function = self.changePremiumLLM(premium_llm_function)

            elif action == "K":  # Exit program
                self.exitProgram()

            elif action == "J":  # Change num of parallel inferences and synthesize mode
                self.changeNumParallelInferencesAndSynthesize()

            # Count time spent and occurrences waiting and in each option
            self.track_time_spent(action, mode='before')

            if action in [None, "", "P", "C", "Z"]:
                if action == "P":
                    use_premium_llm = True
                break
            else:
                proceed = "y" if self.automation else "n"
                if proceed in ["y", "p", ""]:
                    if proceed == "p":
                        use_premium_llm = True
                    break

        smart_print(
            f"Time spent in each option and occurrences: {self.before_inference_option_times} - {self.before_inference_option_counts}",
            self.agent_name, optional=True)

        self.mode = None
        return messages, comments, forced_llm_output, use_premium_llm, default_llm_function, premium_llm_function, function_calling

    def changeNumParallelInferencesAndSynthesize(self):
        try:
            self.num_parallel_inferences = int(
                smart_input("Enter new value for num_parallel_inferences: ", self.agent_name,
                            "NUM_PARALLEL_INFERENCES"))
        except:
            self.num_parallel_inferences = 1
        synthesize_mode_input = smart_input("Turn synthesis mode on/off (1 for ON, 0 for OFF): ", self.agent_name,
                                            "NUM_PARALLEL_INFERENCES SYNTHESIS MODE CHOICE").strip()  # NEW
        if synthesize_mode_input in ["0", "1"]:  # NEW
            self.synthesize_mode = synthesize_mode_input == "1"  # NEW
        else:  # NEW
            smart_print("Invalid input. Synthesize mode remains unchanged.", self.agent_name,
                        "NUM_PARALLEL_INFERENCES SYNTHESIS MODE CHOICE", optional=True)  # NEW

    def changeNumParallelInferences(self):
        try:
            self.num_parallel_inferences = int(
                smart_input("Enter new value for num_parallel_inferences: ", self.agent_name,
                            "NUM_PARALLEL_INFERENCES"))
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
        if self.fixed_coach:
            selected_index = 1
            log_entries, list_output = self._get_log_entries(self.agent_name, function_name), ""
        elif HumanLLMMonitor.common_vectordb.count() > 0:
            log_entries, list_output = self._get_log_entries(self.agent_name, function_name), ""
            for idx, entry in enumerate(log_entries, start=1):
                content = json.loads(entry.page_content)
                text = (content['output_contents'][0]['content'].replace('\n', '\\') if content[
                    'output_contents'] else "") if isinstance(content['output_contents'], list) else \
                    content['output_contents']['content'].replace('\n', '\\')
                date = entry.metadata['time'].split('.')[0]
                list_output += (
                    f"\033[94m{idx}.\033[0m {text[:100]}....{text[-100:]} #{entry.metadata['function_name']} @{date}\n")  # Display a snippet of each entry
            smart_print(list_output, self.agent_name, "LOG ENTRIES LIST", optional=True)

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
        instructions = smart_input(f"ENTER ADDITIONAL INSTRUCTIONS FOR THE AGENT: ", self.agent_name,
                                   message_type="ADDITIONAL_INFO", optional=False)
        messages[1].content = initial_user_message + f"\n\nADDITIONAL INSTRUCTIONS: << {instructions} >>"

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
        smart_print(output, self.agent_name, "PROMPT OPTIONS", optional=True)
        variant_choice = smart_input(
            "Select a number to modify a prompt or create a new variant (or press Enter to continue with the current selection): ",
            self.agent_name)
        if variant_choice.isdigit() and 0 < int(variant_choice) <= len(prompt_variants) + 1:
            if int(variant_choice) == len(prompt_variants) + 1:
                # Process to create a new variant
                comments = smart_input("Provide critic or feedback for the current prompt: ", self.agent_name)
                refine_prompt = _visual_input(
                    f"Current system prompt:<<< {self.load_prompt(prompt_name=self.system_prompt, directory='prompts')} >>>\n\nFeedback or critic: {comments}")
                forced_llm_output = default_llm_function.invoke(
                    [SystemMessage(content=self.load_prompt(prompt_name="improve_prompt_from_answer_critic",
                                                            directory='prompts')),
                     HumanMessage(content=refine_prompt)])
                new_template = forced_llm_output.content
            else:
                self.system_prompt = prompt_variants[int(variant_choice) - 1]
        if smart_input("Would you like first to get suggestions for a better prompt? (y/n): ",
                       self.agent_name).upper() == "Y":
            if use_premium_llm:
                forced_llm_output = premium_llm_function.invoke(
                    [SystemMessage(
                        content=self.load_prompt(prompt_name="system_prompt_refiner",
                                                 directory='prompts')),
                        HumanMessage(
                            content=f"PROMPT TO GET SUGGESTIONS FOR IMPROVEMENT:\n{self.load_prompt(prompt_name=self.system_prompt, directory='prompts')}")])
            else:
                forced_llm_output = default_llm_function.invoke(
                    [SystemMessage(
                        content=self.load_prompt(prompt_name="system_prompt_refiner",
                                                 directory='prompts')),
                        HumanMessage(
                            content=f"PROMPT TO GET SUGGESTIONS FOR IMPROVEMENT:\n{self.load_prompt(prompt_name=self.system_prompt, directory='prompts')}")])
            smart_print(
                f"***** PROMPT SUGGESTIONS *****\n\033[33m{forced_llm_output.content}\033[0m\n*************",
                self.agent_name, "PROMPT SUGGESTIONS")
        new_template = _visual_input(
            self.load_prompt(prompt_name=self.system_prompt,
                             directory='prompts') if new_template is None else new_template)
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
        smart_print(f"Premium LLM changed to {new_llm_name}", self.agent_name, "Change Premium LLM", optional=True)
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
        smart_print(f"Default LLM changed to {new_llm_name}", self.agent_name, "Change Default LLM", optional=True)
        return default_llm_function

    def getScoredResults(self, function_name):
        HumanLLMMonitor._check_and_init_vector_db()
        menu = (
            "Do you want to see:\n"
            "[A] all MODIFIED/SCORED/COMMENTED results.\n"
            "[B] INPUT modified only.\n"
            "[C] OUTPUT modified only.\n"
            "[D] SCORED only.\n"
            "[E] COMMENTED only.\n"
            "[F] Success Tasks.\n"
            "[G] Failed Tasks.\n"
            "[H] user_message_few_shots.\n"
            "[I] Modify user_message_few_shots.\n"
            "Select your letter for choice or hit enter for all: "
        )
        confirm = smart_input(
            menu,
            self.agent_name,
            "BEFORE inference action MENU"
        ).upper()
        result = []

        if confirm == "F":
            tasks = self.get_learnt_tasks()
            task_list = "\n".join(tasks)
            _visual_input(task_list)
            return

        if confirm == "G":
            tasks = self.get_failed_tasks()
            task_list = "\n".join(tasks)
            _visual_input(task_list)
            return

        if confirm == "H":
            if self.user_message_few_shots:
                _visual_input(self.user_message_few_shots)
            else:
                print("No few_shots available.")
            return

        if confirm == "I":
            new_few_shots = _visual_input(self.user_message_few_shots)
            self.user_message_few_shots = new_few_shots
            envs_status = '\n'.join([env.get_state() for env in self.envs])

            return self.get_multiple_few_shots(self.user_message_few_shots) + (f"\n- Current status of examples on "
                                                                               f"which the task will be tested on: {envs_status}\n")

        if confirm in ["A", "", "B"]:
            result.extend(HumanLLMMonitor.common_vectordb.query(
                query_text="*",
                metadata_filter={
                    "function_name": function_name,
                    "agent_name": self.agent_name,
                    "input_modified": True
                },
                k=100
            ))

        if confirm in ["A", "", "C"]:
            result.extend(HumanLLMMonitor.common_vectordb.query(
                query_text="*",
                metadata_filter={
                    "function_name": function_name,
                    "agent_name": self.agent_name,
                    "output_modified": True
                },
                k=100
            ))

        if confirm in ["A", "", "D"]:
            result.extend(HumanLLMMonitor.common_vectordb.query(
                query_text="*",
                metadata_filter={
                    "function_name": function_name,
                    "agent_name": self.agent_name,
                    "scored": True
                },
                k=100
            ))

        if confirm in ["A", "", "E"]:
            result.extend(HumanLLMMonitor.common_vectordb.query(
                query_text="*",
                metadata_filter={
                    "function_name": function_name,
                    "agent_name": self.agent_name,
                    "commented": True
                },
                k=100
            ))

        if result:
            visual_result = "\n===============================\n".join(
                [
                    json.dumps(
                        json.loads(item.page_content),
                        indent=4,
                        sort_keys=True
                    ).replace("\\n", "\n") for item in result
                ]
            )
            _visual_input(visual_result, filetype="json")
        else:
            print("No results found for the selected option.")

    @staticmethod
    def getPreviousResults(function_name=None, agent_name=None, k=100):
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
                         outputs_count=None, task_name=None):
        if not self.outputs:
            self.outputs = {}
            for i in range(outputs_count):
                self.outputs[i] = None
        if len(self.outputs) < outputs_count:
            for i in range(len(self.outputs), outputs_count):
                self.outputs[i] = None

        self.mode = 'after'
        comments, score = None, None
        nl = "\n"
        print(f"***{self.agent_name}, AFTER INFERENCE***")
        if inference_result_msg is None:
            # enable to request inference_result_msg.content to be None
            inference_result_msg = type('InferenceResult', (object,), {'content': None})

        while self.skip_rounds <= 0:
            self.temp_inference_result_content = None  # to capture the inference message done from functions outside of the menu
            # MENU
            multiple_ref = (f"OUTPUT \033[31m{output_id} OUT OF {outputs_count}\033[0m OUTPUTS" if (
                    output_id and outputs_count and (outputs_count > 1)) else "")

            # Run inference checks if any
            check_results = self.run_inference_checks(output_id - 1, inference_result_msg.content)
            check_display = ""
            # Display inference check results
            for check_name, result in check_results.items():
                check_display += f"{nl}CHECK {check_name} result: " + str(result).replace("\\n", "\n")

            if not task_name:
                pattern = r'def\s+(\w+)\('
                match = re.search(pattern, inference_result_msg.content, flags=re.MULTILINE)
                task_name = match.group(1) if match else print("No function definitions found.")

            HumanLLMMonitor.add_agent_data(self.agent_name, "saved_task", {'llm_output': inference_result_msg.content,
                                                                           'user_message':
                                                                               self.current_inference_context[
                                                                                   'input_contents'][1].content,
                                                                           'num_parallel_inferences': self.num_parallel_inferences,
                                                                           'task_parameters': self.task_parameters},
                                           before_after='after', user_id=HumanLLMMonitor.user_id, step_id=self.step_id,
                                           type_tache="IR_CPS_TechSynthesis", id_task=True, function_name=task_name)
            menu = (
                f"\033[{self.print_color}m***** {self.agent_name}->{inspect.stack()[2].function} AFTER *****\nLLM ANSWER:\n{inference_result_msg.content}\n{check_display}\n***** {self.agent_name}->{inspect.stack()[2].function} AFTER *****\033[0m{multiple_ref}\n")

            menu += (
                "[A] Edit answer in VSCode\n")  # je voudrais le corriger uniquement pour demander une suggestion d'amélioration du prompt (d'un autre côté, je peux aussi le faire dans le menu précédent)
            menu += ("[B] Critic answer to regenerate it\n")
            menu += ("[C] Critic to improve agent's behavior\n")
            menu += ("[D] Evaluate answer\n")
            menu += ("[E] Go back (to BEFORE menu)\n")
            menu += ("[G] Skip for N rounds (auto mode)\n")
            menu += ("[Z] Continue\n")
            menu += ("[H] Exit\n")

            smart_print(menu, self.agent_name, "AFTER inference action MENU" + (
                f" {output_id}/{outputs_count}" if (output_id and outputs_count and (outputs_count > 1)) else ""),
                        self.agent_name, column_id=output_id - 1, column_max=outputs_count)
            print(f"***{self.agent_name}, AFTER INFERENCE, after smart_print menu***")
            self.menu_start_time = time.time()

            if self.automation:
                if comments is None:
                    temp, _ = self.get_agent_data(self.agent_name, "llm_suggestions")
                    if temp:
                        comments = temp[0]['llm_suggestions']
                if (hasattr(self, 'recommend_critics') and self.recommend_critics) and self.outputs[
                    output_id - 1] is None:
                    inference_result_msg.content = self.criticAnswer(comments, inference_result_msg.content,
                                                                     text_has_annotations=False)
                    self.outputs[output_id - 1] = inference_result_msg.content
                action = ""
            else:
                action = smart_input(
                    f"\n\033[32mAFTER\033[0m inference @ {self.agent_name}-> Choose an action (or hit Enter for inference) :",
                    self.agent_name, optional=False, column_id=output_id-1, column_max=outputs_count).upper()

            if self.temp_inference_result_content: # if modified async, it is important in case of edition ("A") to keep the modified content
                inference_result_msg.content = self.temp_inference_result_content

            # ACTIONS processing
            self.start_time = time.time()  # Init action selected and timer to measure time spent and occurences in action processing

            if action == "A":  # Manually set/modify the answer/output
                self.modifyAnswer(inference_result_msg, output_id)

            elif action == "B":  # Critic this answer/output to get an improved answer/output
                inference_result_msg.content = self.criticAnswer(comments, inference_result_msg.content,
                                                                 text_has_annotations=False)

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
            self.track_time_spent(action, mode='after')

            # check also that inference_result_msg is not of type str or int 
            if self.temp_inference_result_content and not isinstance(inference_result_msg, str) and not isinstance(inference_result_msg, int):
                #inference_result_msg.content = f"{self.temp_inference_result_content}"
                smart_print("ANSWER MODIFIED, NEW CHECKS REQUIRED BEFORE CONTINUING", self.agent_name, "code_task_and_run_test SystemMessage", column_id=output_id-1, column_max=outputs_count)
                #check_results = self.run_inference_checks(output_id - 1, inference_result_msg.content)
            elif action in [None, "", "E", "Z"]:
                break  # E: Go back BEFORE inference to improve system prompt or add information to user message

            # proceed = smart_input("Continue 'y' (or 'n' to go back to menu) ? ", self.agent_name, column_id=output_id, column_max=outputs_count).lower()
            # if proceed in ["y", ""]:
            #     break

        if self.skip_rounds > 0:
            check_results = self.run_inference_checks(output_id - 1, inference_result_msg.content)
            check_display = ""
            # Display inference check results
            for check_name, result in check_results.items():
                check_display += f"{nl}CHECK {check_name} result: " + str(result).replace("\\n", "\n")

            smart_print(
                f"\033[{self.print_color}m****{self.agent_name}>{inspect.stack()[2].function} LLM ANSWER content****\n{inference_result_msg.content}\n{check_display}\n*****************\033[0m",
                self.agent_name, "LLM ANSWER content", column_id=output_id)
            self.skip_rounds -= 1
        else:
            smart_print(
                f"Time spent in each option and occurrences: {self.after_inference_option_times} - {self.after_inference_option_counts}",
                self.agent_name, optional=True, column_max=outputs_count)

        self.mode = None
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
        refine_prompt = f"Current system prompt:<<< {self.load_prompt(prompt_name=self.system_prompt, directory='prompts')} >>>\n\nPrompt's answer:<<< {inference_result_msg.content} >>>\n\nPrompt's answer critic:{comments}\n\nPrompt's ideal Answer:<<< {ideal_answer} >>>"
        smart_print(f"***** PROMPT FOR IMPROVEMENT *****\n{refine_prompt}", self.agent_name, "PROMPT FOR IMPROVEMENT")
        llm_output = premium_llm_function.invoke([SystemMessage(
            content=self.load_prompt(prompt_name="improve_prompt_from_answer_critic", directory='prompts')),
            HumanMessage(content=refine_prompt)])
        smart_print(f"***** RECOMMENDATION OPEN FOR EDITION *****\n", self.agent_name,
                    "RECOMMENDATION OPEN FOR EDITION")
        new_template = _visual_input(llm_output.content)
        smart_print(f"***** NEW PROMPT TEMPLATE:\n{new_template}\n*************", self.agent_name,
                    "NEW PROMPT TEMPLATE")  # Confirm that the user wants to modify the template
        confirm = smart_input("Do you want to replace current prompt file template with your input? (y/n): ",
                              self.agent_name).upper()  # Save prompt with tag options
        if confirm == "Y":
            tag_option = smart_input(
                "Enter a tag for saving the prompt (leave blank for no tag, or 'same' to keep the current tag): ",
                self.agent_name)
        if tag_option.lower() == "same":
            save_prompt_with_tag(self.system_prompt, new_template, "")
        else:
            save_prompt_with_tag(self.system_prompt, new_template, tag_option)

        return comments

    def criticAnswer(self, suggestions, text_content, text_has_annotations=True, annotation_format=None,
                     instruction_processing_approach='ANNOTATIONS_ALL'):
        """
            This method processes the suggestions and text content to generate an improved answer.
            It can handle annotated critics to refine the text content based on the provided suggestions.

            Args:
                suggestions (str): The suggestions or critics to be applied to the text content.
                text_content (str or List[str]): The original text content that needs to be improved.
                text_has_annotations (bool): Indicates whether the text contains annotations. Default is True.
                annotation_format (str): The format of the annotations. If None and text_has_annotations is True,
                                     the format will be auto-detected.
            instruction_processing_approach (str): The approach for processing instructions. Possible values are
                                                   'FULLTEXT_ALL', 'FULLTEXT_EACH', 'ANNOTATIONS_ALL', 'ANNOTATIONS_EACH'.
                                                   Default is 'ANNOTATIONS_ALL'.

        Returns:
            str: The improved text content after applying the suggestions and critics.
        """
        import re
        import json
        import difflib  # Ajout de l'importation du module difflib

        # Combine text_content if it's a list
        if isinstance(text_content, list):
            combined_text_content = '\n'.join(text_content)
        elif isinstance(text_content, str):
            combined_text_content = text_content
        else:
            raise TypeError("text_content must be a string or a list of strings.")

        # Auto-detect annotation format if necessary
        def detect_annotation_format(text):
            patterns = {
                'latex-inline': re.compile(r'\\(?P<tag>\w+)(\[(?P<instruction>[^\]]*)\])?\{(?P<content>.*?)\}',
                                           re.DOTALL),
                'HTML-inline': re.compile(
                    r'<(?P<tag>\w+)(\s+instruction="(?P<instruction>[^"]*)")?>\s*(?P<content>.*?)\s*</(?P=tag)>',
                    re.DOTALL),
                'latex-id': re.compile(r'\[(?P<id>\d+)\]\{(?P<content>.*?)\}', re.DOTALL),
                'HTML-id': re.compile(r'<(?P<id>\d+)>(?P<content>.*?)</(?P=id)>', re.DOTALL)
            }
            for fmt, pattern in patterns.items():
                if pattern.search(text):
                    return fmt
            return None

        # Parse annotations from text
        def parse_annotations(text, annotation_format):
            annotations = []
            if annotation_format == 'latex-inline':
                pattern = re.compile(r'\\(?P<tag>\w+)(\[(?P<instruction>[^\]]*)\])?\{(?P<content>.*?)\}', re.DOTALL)
            elif annotation_format == 'HTML-inline':
                pattern = re.compile(
                    r'<(?P<tag>\w+)(\s+instruction="(?P<instruction>[^"]*)")?>\s*(?P<content>.*?)\s*</(?P=tag)>',
                    re.DOTALL)
            elif annotation_format == 'latex-id':
                pattern = re.compile(r'\[(?P<id>\d+)\]\{(?P<content>.*?)\}', re.DOTALL)
            elif annotation_format == 'HTML-id':
                pattern = re.compile(r'<(?P<id>\d+)>(?P<content>.*?)</(?P=id)>', re.DOTALL)
            else:
                return annotations
            for match in pattern.finditer(text):
                annotation = match.groupdict()
                annotation['full_match'] = match.group(0)
                annotation['start'] = match.start()
                annotation['end'] = match.end()
                annotations.append(annotation)
            return annotations

        # Parse instructions from suggestions
        def parse_instructions(suggestions):
            instructions = {}
            if suggestions:
                pattern = re.compile(r'\[(?P<id>\d+)\]:\s*(?P<instruction>.+)')
                for line in suggestions.strip().splitlines():
                    match = pattern.match(line.strip())
                    if match:
                        id_ = match.group('id')
                        instruction = match.group('instruction').strip()
                        instructions[id_] = instruction
            return instructions

        critic = None
        if self.last_inference_check_results:
            for result in self.last_inference_check_results:
                if result is not None and isinstance(result, dict):
                    for key, value in result.items():
                        if key == 'Recommend critiques':
                            critic = value
                            break
                else:
                    break

        if critic:
            if suggestions == "":
                suggestions = critic['suggestions']
            HumanLLMMonitor.add_agent_data(self.agent_name, "llm_suggestions",
                                           {'llm_suggestions': critic['suggestions'], 'user_suggestions': suggestions,
                                            'llm_suggestions_prompt': critic['improvement_prompt']})

        if text_has_annotations:
            if annotation_format is None:
                annotation_format = detect_annotation_format(combined_text_content)
                if annotation_format is None:
                    raise ValueError("Could not auto-detect annotation format.")
            annotations = parse_annotations(combined_text_content, annotation_format)
            if annotation_format in ['latex-inline', 'HTML-inline']:
                # Instructions are inline within annotations
                for annotation in annotations:
                    instruction = annotation.get('instruction', '').strip()
                    annotation['instruction'] = instruction
            elif annotation_format in ['latex-id', 'HTML-id']:
                # Instructions are provided in suggestions
                instructions = parse_instructions(suggestions)
                for annotation in annotations:
                    id_ = annotation.get('id')
                    instruction = instructions.get(id_)
                    if instruction:
                        annotation['instruction'] = instruction
                    else:
                        raise ValueError(f"No instruction found for annotation ID {id_}")
            else:
                raise ValueError("Unsupported annotation format.")

            if instruction_processing_approach == 'FULLTEXT_ALL':
                system_prompt = """
                Your task is to improve the following text by processing the annotations and following the instructions provided.
                Please replace the annotated parts according to the instructions and produce the final improved version of the text.

                Possible actions:
                1. **FIX:** Make necessary corrections.
                2. **IMPROVE:** Enhance the content.
                3. **INSERT:** Add new content as instructed.
                """
                user_prompt = f"{combined_text_content}"
                llm_output = self.premium_llm.invoke(
                    [SystemMessage(content=system_prompt.strip()), HumanMessage(content=user_prompt)])
                final_text = llm_output.content

            elif instruction_processing_approach == 'FULLTEXT_EACH':
                final_text = combined_text_content
                for annotation in annotations:
                    system_prompt = """
                Your task is to improve the following text by processing the annotation and following the instruction.
                Please replace the annotated part according to the instruction and produce the final improved version of the text.
                """
                    # Replace other annotations with their content
                    temp_text = final_text
                    for other_annotation in annotations:
                        if other_annotation != annotation:
                            temp_text = temp_text.replace(other_annotation['full_match'], other_annotation['content'])
                    llm_output = self.premium_llm.invoke(
                        [SystemMessage(content=system_prompt.strip()), HumanMessage(content=temp_text)])
                    final_text = llm_output.content

            elif instruction_processing_approach == 'ANNOTATIONS_ALL':
                annotations_data = {annotation.get('id') or str(i): {
                    'content': annotation['content'],
                    'instruction': annotation['instruction']
                } for i, annotation in enumerate(annotations)}
                system_prompt = """
                Your task is to generate new content for the annotated parts according to the instructions.
                Provide your output as a JSON dictionary mapping IDs to the new content.
                The values should be strings containing the new content, without additional keys or nesting.
                Example Output: {"1": "new content for annotation 1", "2": "new content for annotation 2"}
                """
                user_prompt = f"Annotations:\n{json.dumps(annotations_data)}"
                llm_output = self.premium_llm.invoke(
                    [SystemMessage(content=system_prompt.strip()), HumanMessage(content=user_prompt)])
                llm_response = llm_output.content.strip()
                try:
                    new_contents = json.loads(llm_response)
                except json.JSONDecodeError:
                    raise ValueError(f"Invalid JSON response from LLM: {llm_response}")

                final_text = combined_text_content
                for annotation in annotations:
                    id_ = annotation.get('id') or str(annotations.index(annotation))
                    full_match = annotation['full_match']
                    new_content = new_contents.get(id_)
                    if new_content:
                        # Ensure new_content is a string
                        if not isinstance(new_content, str):
                            raise ValueError(f"New content for annotation ID {id_} is not a string.")
                        final_text = final_text.replace(full_match, new_content)
                    else:
                        raise ValueError(f"No new content found for annotation ID {id_}")

            elif instruction_processing_approach == 'ANNOTATIONS_EACH':
                final_text = combined_text_content
                for annotation in annotations:
                    system_prompt = """
                Your task is to generate new content for the following text according to the instruction.
                """
                    user_prompt = f"Text: {annotation['content']}\nInstruction: {annotation['instruction']}"
                    llm_output = self.premium_llm.invoke(
                        [SystemMessage(content=system_prompt.strip()), HumanMessage(content=user_prompt)])
                    new_content = llm_output.content.strip()
                    full_match = annotation['full_match']
                    final_text = final_text.replace(full_match, new_content)
            else:
                raise ValueError("Unsupported instruction processing approach.")
        else:
            # Existing logic for non-annotated text
            if suggestions is None:
                suggestions = smart_input("Provide critic/feedback/request: ", self.agent_name,
                                          column_max=self.num_parallel_inferences)
            temp_prev_sugg, _ = HumanLLMMonitor.get_agent_data(self.agent_name, "llm_suggestions")
            prev_sugg = ""
            for i in temp_prev_sugg:
                llm_suggestion = i.get('llm_suggestions', '')
                user_suggestion = i.get('user_suggestions', '')
                diff = difflib.unified_diff(llm_suggestion.splitlines(), user_suggestion.splitlines(), lineterm='')
                diff_text = '\n'.join(diff)
                prev_sugg += f"#{diff_text}#\n"
            prompt_sugg = ""
            if suggestions:
                prompt_sugg = (
                    "Your task is also to take into account the **SUGGESTIONS** and modify the answer accordingly. Also we will give you **PREVIOUS SUGGESTIONS** that were given on previous task.\n"
                    "These previous suggestions are here to help you more understanding the suggestions and to help you to improve the answer.\n"
                    f"\n### SUGGESTIONS: << {suggestions} >>\n"
                    f"\n### PREVIOUS SUGGESTIONS: << {prev_sugg} >>"
                )
            system_prompt = f"""Given the INSTRUCTION provided by the user (and the **INITIAL PROMPT**), your task is to generate a very different new answer from the INITIAL ANSWER or to refine the initial answer.
            {prompt_sugg}
            ### INITIAL PROMPT: << {self.llm_input_messages[0].content} >>
            ### INITIAL ANSWER: << {combined_text_content} >> """
            user_prompt = f"INSTRUCTION: << {suggestions} >>"
            llm_output = self.premium_llm.invoke(
                [SystemMessage(content=system_prompt.strip()), HumanMessage(content=user_prompt)])
            final_text = llm_output.content

        self.temp_inference_result_content = final_text  # Capture the result
        return final_text

    def modifyAnswer(self, inference_result_msg, column_id=None):
        inference_result_msg.content = _visual_input(inference_result_msg.content)
        self.temp_inference_result_content = inference_result_msg.content
        smart_print(f"***** ANSWER:\n{inference_result_msg.content}\n*************", self.agent_name, "NEW ANSWER", optional=True, column_id=column_id, column_max=self.num_parallel_inferences)
        return inference_result_msg.content

    def updateAnswer(self, answer, column_id=None):
        self.temp_inference_result_content = answer
        if column_id: print("IMPORTANT: column_id not yet implemented - Updating current HumanLLMMonitor for agent")
        return answer

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
                    result = smart_input(f"Uknown function: {function_name} -- please provide result manually:\n",
                                         "call_llm_function_with_function_call")
                messages.append(FunctionMessage(content=result, name=function_name, arguments=function_args_str))

        return output

    def _log_entry(self, function_name, input_contents, output_contents, input_modified=False,
                   skipped_inference=False, input_comments=None, output_comments=None, output_llm_raw=None,
                   output_modified=False, inference_time=None, user_score=None, message_tokens=None, score=None,
                   use_premium_llm=False,
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

    def process_llm_output(self, llm_output, counter, llm_outputs, init_skip_rounds, task_name=None):
        """Traite un seul LLM output (séquentiellement ou en parallèle)."""
        print(f"Processing LLM output {counter} out of {len(llm_outputs)}")
        if len(llm_outputs) > 1:
            self.skip_rounds = init_skip_rounds
            smart_print(f"ANSWER NUMBER #{counter-1} ", self.agent_name, "POST INFERENCE", append=True, optional=True)

        self.current_inference_context = {
            'function_name': inspect.stack()[1].function,
            'input_contents': self.llm_input_messages,
            'output_contents': None,  # Remplir après traitement
            'start_time': datetime.now(),
            'input_modified': False,
            'skipped_inference': None,
            'input_comments': None,
            'output_comments': None,
            'raw_llm_outputs': None,
            'output_modified': None,
            'message_tokens': None,
            'use_premium_llm': False
        }
        # Post-inference human intervention (traitement standard après une inférence)
        output_messages_instance, output_comments_instance, score_instance = self._after_inference(
            llm_output, premium_llm_function=None, output_id=counter,
            outputs_count=len(llm_outputs), task_name=task_name)
        return output_messages_instance, output_comments_instance, score_instance

    #    def CallHumanLLM(self, original_input_messages=None, llm_function=None, premium_llm_function=None, callable_system_message=None, system_prompt_template=None, user_message=None, return_message_content_only=True, function_calling=False, temperature=0.7, timeout_seconds=90, stream_output=True):
    def CallHumanLLM(self, original_input_messages=None, default_llm_function=None, premium_llm_function=None,
                     callable_system_message=None, system_prompt_template=None, user_message=None,
                     return_message_content_only=True, function_calling=False, temperature_min=None, timeout_seconds=300,
                     stream_output=False, use_default_llm=True, model_choice=None,
                     temperature_max=None, task_name=None):
        #if not self.selected_llm_or_chain: raise ValueError("No LLM or chain selected for use.")
        if temperature_min is None: temperature_min = self.temperature_min
        if temperature_max is None: temperature_max = self.temperature_max
        # Define a helper function to perform the LLM calls for parallel inference.
        def perform_llm_call(input_msg, use_premium, func_calling, temperature=None, stream_output=True, color_id=None):
            if use_premium:
                func = premium_llm_function if not func_calling else HumanLLMMonitor.call_llm_function_with_function_call
            else:
                func = default_llm_function if not func_calling else HumanLLMMonitor.call_llm_function_with_function_call
            if temperature or temperature == 0:
                func = func.with_config(configurable={"llm_temperature": temperature})
                print(f"Temperature set to {temperature}")
            else:
                print(f"No temperature value, not set")

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
                json_trail_re = re.compile(r'[\'\}\]]$')

                buffer = ""
                buffer_start_time = time.time()
                flush_interval = 5.0  # seconds

                for chunk in func.stream(input_msg):
                    if hasattr(chunk, 'content'):
                        chunk_content = chunk.content
                        final_output += chunk_content
                    else:
                        current_chunk_str = str(chunk)
                        while json_trail_re.search(current_chunk_str):
                            current_chunk_str = current_chunk_str[:-1]
                        new_part_index = len(previous_chunk_str)
                        chunk_content = current_chunk_str[new_part_index:]
                        previous_chunk_str = current_chunk_str
                        final_output += chunk_content

                    buffer += chunk_content

                    current_time = time.time()
                    time_elapsed = current_time - buffer_start_time

                    # Check if buffer should be flushed
                    should_flush = False
                    delimiter_pos = -1
                    delimiter_length = 0

                    if time_elapsed >= flush_interval:
                        should_flush = True
                        delimiter_pos = len(buffer)  # Flush the entire buffer
                    elif ('\n\n' in buffer or '<br>' in buffer) and len(buffer) >= 100:
                        # Find the last occurrence of "\n\n" or "<br>"
                        pos_newline = buffer.rfind('\n\n')
                        pos_br = buffer.rfind('<br>')
                        if pos_newline > pos_br:
                            delimiter_pos = pos_newline
                            delimiter_length = 2  # Length of "\n\n"
                        else:
                            delimiter_pos = pos_br
                            delimiter_length = 4  # Length of "<br>"

                        if delimiter_pos != -1:
                            should_flush = True

                    if should_flush:
                        if delimiter_pos == len(buffer):
                            # Time-based flush: send the entire buffer
                            to_send = buffer
                            buffer = ""
                        elif delimiter_pos != -1:
                            # Delimiter-based flush: send up to the last delimiter
                            to_send = buffer[:delimiter_pos + delimiter_length]
                            buffer = buffer[delimiter_pos + delimiter_length:]
                        else:
                            # No delimiter found, flush entire buffer
                            to_send = buffer
                            buffer = ""

                        if HumanLLMMonitor.use_websocket:
                            smart_print(to_send, self.agent_name, f"Inference streaming output {color_id}",
                                        append=True, column_id=color_id, column_max=self.num_parallel_inferences)
                        else:
                            smart_print(start_color + to_send + end_color, self.agent_name,
                                        f"Inference streaming output {color_id}", append=True)

                        # Reset the timer after flushing
                        buffer_start_time = current_time

                # Final flush after streaming ends
                if buffer:
                    if HumanLLMMonitor.use_websocket:
                        smart_print(buffer, self.agent_name, f"Inference streaming output {color_id}",
                                    append=True, column_id=color_id, column_max=self.num_parallel_inferences)
                    else:
                        smart_print(start_color + buffer + end_color, self.agent_name,
                                    f"Inference streaming output {color_id}", append=True)

                return AIMessage(content=final_output)

            else:
                value = func.invoke(input_msg)
                return AIMessage(content=value.content if hasattr(value, 'content') else str(value))

        smart_print(
            f"\033[{self.print_color}m****{self.agent_name}>{inspect.stack()[1].function} calling HumanLLMMonitor****\033[0m",
            self.agent_name, "HumanLLMMonitor", optional=True)
        if system_prompt_template:
            self.system_prompt = system_prompt_template
        if default_llm_function is None:
            default_llm_function = self.default_llm if use_default_llm else self.premium_llm
        if premium_llm_function is None:
            premium_llm_function = self.premium_llm if self.premium_llm else None
        print(f"****agent : {self.agent_name}, automation : {self.automation}****")
        if self.automation == 'before' and (hasattr(self, "saved_task")):
            temp = self.saved_task.get('content', {})
            print(f"****temp (prompt before {self.agent_name}) : {temp}****")
            self.system_prompt = temp.get('prompt', "")
            if self.auto_n_rounds > 0:
                self.automation = "full_auto"
            else:
                self.automation = None
            original_input_messages = [
                SystemMessage(content=self.system_prompt),
                HumanMessage(content=user_message)]
        elif original_input_messages is None:
            print(f"****user_message {self.agent_name} : {user_message}****")
            original_input_messages = [
                SystemMessage(content=self.load_prompt(prompt_name=self.system_prompt, directory='prompts')),
                HumanMessage(content=user_message)]
        input_contents_str0, input_contents_str1 = str(original_input_messages[0].content), str(
            original_input_messages[1].content)

        self.before_inference_option_times = {'TOTAL': 0, 'SELECTION': 0}  # then each option will be added to this dict
        self.before_inference_option_counts = {'TOTAL': 0,
                                               'SELECTION': 0}  # then each option will be added to this dict
        self.after_inference_option_times = {'TOTAL': 0, 'SELECTION': 0}  # then each option will be added to this dict
        self.after_inference_option_counts = {'TOTAL': 0, 'SELECTION': 0}  # then each option will be added to this dict
        self.unidentified_option_times = {'TOTAL': 0, 'SELECTION': 0}  # then each option will be added to this dict
        self.unidentified_option_counts = {'TOTAL': 0, 'SELECTION': 0}  # then each option will be added to this dict
        self.mode = None
        call_start_time = time.time()

        while True:
            if self.skip_rounds > 0:
                smart_print(
                    f"\033[{self.print_color}m****{self.agent_name}>{inspect.stack()[2].function} skipping HumanLLMMonitor for {self.skip_rounds} rounds****\033[0m",
                    self.agent_name, "Skipping round", optional=True)

            start_time = datetime.now()
            # Pre-inference human intervention
            if self.automation in ['before', 'after', 'skip_once']:
                print(f"****agent : {self.agent_name}, automation : {self.automation}****")
                input_comments, skip_inference, use_premium_llm, llm_outputs = None, False, False, []
                llm_input_messages = original_input_messages
                self.llm_input_messages = original_input_messages
                self.last_inference_check_results = [None]
                if self.automation == 'after' and hasattr(self, "saved_task"):  # Plus besoin de tester sur agent_name
                    temp = self.saved_task.get('content', {})
                    print(f"****temp (llm_output after {self.agent_name}) : {temp}****")
                    llm_outputs = [AIMessage(content=temp.get('llm_output', ""))]
                    if self.auto_n_rounds > 0:
                        self.automation = "full_auto"
                    else:
                        self.automation = None
                    print(f"****llm_output : {llm_outputs}****")
                    smart_print(
                        llm_outputs[0].content if llm_outputs else "No LLM output",
                        self.agent_name, "NEW inference result recieved", column_id=0,
                        column_max=1)
            else:
                llm_input_messages, input_comments, skip_inference, use_premium_llm, default_llm_function, premium_llm_function, function_calling = self._before_inference(
                    original_input_messages, default_llm_function, premium_llm_function, function_calling,
                    callable_system_message, model_choice=model_choice, task_name=task_name)
                self.llm_input_messages = llm_input_messages
                self.clear_selected_outputs()
                self.last_inference_check_results = [
                                                        None] * self.num_parallel_inferences  # Pre-allocate the list with None
                if llm_input_messages and not skip_inference:
                    # Use concurrent futures to parallelize the LLM calls.
                    outputs = []

                    with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_parallel_inferences) as executor:
                        if type(self.premium_llm if use_premium_llm else self.default_llm) == type(
                                self.llmORchains_list.get('3_majority_chain')):
                            stream_output = True
                        futures = [
                            executor.submit(perform_llm_call, llm_input_messages, use_premium_llm, function_calling,
                                            ((temperature_min + i * (temperature_max - temperature_min) / (
                                                        self.num_parallel_inferences - 1)) if (
                                                        temperature_min is not None and self.num_parallel_inferences > 1 and temperature_min >= 0.) else 0),
                                            stream_output, i) for i in
                            range(self.num_parallel_inferences)]
                        for idx, future in enumerate(futures):
                            try:
                                llm_response = future.result(timeout=timeout_seconds)
                                outputs.append(llm_response)
                                if HumanLLMMonitor.use_websocket:
                                    smart_print(
                                        llm_response.content,
                                        self.agent_name, "NEW inference result recieved", column_id=idx,
                                        column_max=self.num_parallel_inferences)
                                else:
                                    smart_print(
                                        f'\033[0m**** New inference result recieved and added to outputs as #{len(outputs)}\033[0m:\n{llm_response.content}\n\033[9mEND OF #{len(outputs)}****\033[0m',
                                        self.agent_name, "NEW inference result recieved", column_id=idx,
                                        column_max=self.num_parallel_inferences)
                            except concurrent.futures.TimeoutError:
                                smart_print('A task ran longer than the allotted timeout and was cancelled.',
                                            self.agent_name, "Inference result TIMEOUT")
                            except Exception as exc:
                                smart_print(f'Generated an exception: {exc}', self.agent_name,
                                            "Inference result EXCEPTION")
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
                init_skip_rounds = self.skip_rounds  # save the current skip_rounds value because multiple outputs decrease skip rounds for each parallel output
                if len(llm_outputs) > 1:
                    smart_print("**** Multiple LLM ANSWERS > we will process POST INFERENCE for each ****",
                                self.agent_name, "Multiple LLM ANSWERS", append=True, optional=True)
                if False and HumanLLMMonitor.use_websocket:
                    # Utilisez un ThreadPoolExecutor pour exécuter les réponses en parallèle en mode WebSocket.
                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        futures = []
                        print(f"****agent : {self.agent_name}, websocket****")
                        for counter, llm_output in enumerate(llm_outputs, start=1):
                            print(f"****agent : {self.agent_name}, websocket, counter : {counter}****")
                            futures.append(executor.submit(self.process_llm_output, llm_output, counter, llm_outputs,
                                                           init_skip_rounds, task_name))

                        # Attendre que toutes les tâches soient terminées.
                        results = [future.result() for future in futures]

                        # Traiter les résultats de chaque future (en parallèle ou séquentiellement)
                        for counter, (output_messages_instance, output_comments_instance, score_instance) in enumerate(
                                results, start=1):
                            output_messages.append(output_messages_instance)
                            if output_messages_instance == -1:
                                break
                            output_comments.append(output_comments_instance)
                            score.append(score_instance)
                else:
                    # Traitement séquentiel classique
                    for counter, llm_output in enumerate(llm_outputs, start=1):
                        output_messages_instance, output_comments_instance, score_instance = self.process_llm_output(
                            llm_output, counter, llm_outputs, init_skip_rounds, task_name)
                        output_messages.append(output_messages_instance)
                        if output_messages_instance == -1:
                            break
                        output_comments.append(output_comments_instance)
                        score.append(score_instance)

                # Test si l'une des instances de output_messages == -1
                if any([output_messages_instance == -1 for output_messages_instance in output_messages]):
                    original_input_messages[0].content, original_input_messages[
                        1].content = input_contents_str0, input_contents_str1
                else:
                    # Sortir de la boucle
                    break
            # Modifier afin de tester sur 'skip_once', et dans ce cas pas besoin de tester l'agent_name
            elif self.automation == 'skip_once' and hasattr(self, "saved_task"):
                if self.auto_n_rounds > 0:
                    self.automation = "full_auto"
                else:
                    self.automation = None
                input_content = ""
                if 'input_contents' in self.saved_task.get('content', {}):
                    input_content = self.saved_task.get('content', {}).get('input_contents', "")
                output_messages = [AIMessage(content=input_content)]
                break

        if self.auto_n_rounds:
            if self.auto_n_rounds > 0: self.auto_n_rounds -= 1
            if not self.auto_n_rounds: self.automation = None
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

    def generate_best_improvement_suggestions(self, inference_result_content=None, output_id=None):
        """
        Uses a premium LLM to generate top suggestions or critiques for improving
        the inference output, identified by output_id. If there are no check results,
        it requests general improvement suggestions based on the inference result content.

        Additionally, it generates annotations in the specified format using a LLM,
        and saves them in the database.

        :param output_id: The ID of the output message to critique.
        :param inference_result_content: The actual content of the inference result to be critiqued.
        :return: A dictionary containing improvement suggestions and annotations.
        """

        import re
        import json

        # Run checks on the inference content if available
        improvement_feedback = []
        check_results = []
        for inference_check in self.last_inference_check_results:
            if inference_check:
                check_results += [inference_check]
                break

        # Collect insights from various checks, focusing on quality-related results
        if check_results:
            for check_result in check_results:
                for check_name, result in check_result.items():
                    if isinstance(result, str) and result:  # Include only meaningful, non-empty results
                        improvement_feedback.append(f"Feedback from {check_name}: {result}")
                    elif isinstance(result, list) and result:
                        improvement_feedback.extend([f"{check_name} feedback: {item}" for item in result if item])

        previous_suggestions, _ = HumanLLMMonitor.get_agent_data(self.agent_name, "llm_suggestions")
        prev_sugg = ""
        prev_sugg_u = ""
        if previous_suggestions:
            for sugg in previous_suggestions:
                prev_sugg += f"\n{sugg.get('llm_suggestions', '')}"
                prev_sugg_u += f"\n{sugg.get('user_suggestions', '')}"

        # Prepare a prompt based on whether feedback is available
        if improvement_feedback:
            feedback_prompt = "\n".join(improvement_feedback)
            improvement_prompt = (
                "Based on the feedback below, generate concise, actionable suggestions "
                "to improve or correct the ANSWER. Focus on clarity, accuracy, and style improvements. "
                "\n\n"
                f"PROMPT:<<<{self.system_prompt}>>>\n\n"
                f"ANSWER:<<<{inference_result_content}>>>\n\n"
                f"Feedback:<<<{feedback_prompt}>>>\n\n"
                f"Previous LLM Suggestions:<<<{prev_sugg}>>>\n\n"
                f"Previous User Suggestions:<<<{prev_sugg_u}>>>\n\n"
                "Provide your improvement suggestions below."
            )
        else:
            improvement_prompt = (
                "Provide improvement suggestions to enhance the clarity, accuracy, and quality of the following ANSWER. "
                f"PROMPT:<<<{self.system_prompt}>>>\n\n"
                f"ANSWER:<<<{inference_result_content}>>>\n\n"
                f"Previous LLM Suggestions:<<<{prev_sugg}>>>\n\n"
                f"Previous User Suggestions:<<<{prev_sugg_u}>>>\n\n"
                "List your improvement suggestions below."
            )

        # Use the premium LLM to generate suggestions
        response = self.premium_llm.invoke([
            SystemMessage(content="You are tasked with analyzing feedback to improve model outputs."),
            HumanMessage(content=improvement_prompt)
        ])

        # Clean up the response content
        response.content = re.sub(r'[^\x20-\x7E\t\n\r]', '', response.content)
        improvement_prompt = re.sub(r'[^\x20-\x7E\t\n\r]', '', improvement_prompt)
        response.content = re.sub(r'\\u[0-9A-Fa-f]{4}', '', response.content)
        improvement_prompt = re.sub(r'\\u[0-9A-Fa-f]{4}', '', improvement_prompt)

        # Save the suggestions using add_agent_data
        HumanLLMMonitor.add_agent_data(self.agent_name, "llm_suggestions", {
            'llm_suggestions': response.content,
            'user_suggestions': "",
            'improvement_prompt': improvement_prompt
        })

        # Now generate annotations using the LLM
        # Retrieve previous annotations
        previous_annotations, _ = HumanLLMMonitor.get_agent_data(self.agent_name, "llm_annotations")
        prev_annotations = ""
        if previous_annotations:
            for ann in previous_annotations:
                prev_annotations += f"\n{ann.get('annotations', '')}"

        # Construct the annotation prompt
        annotation_prompt = (
            "Based on the ANSWER below, generate a list of annotations in the following format:\n"
            "<TAG>: <Texte annoté> <Commentaire de l'annotation>\n"
            "Le TAG peut être l'un de ces trois : FIX, INSERT, IMPROVE.\n"
            "Chaque annotation doit être pertinente et couvrir des parties du PROMPT.\n"
            "Fournissez au minimum 5 annotations, mais vous pouvez en inclure davantage.\n\n"
            f"PROMPT:<<<{self.system_prompt}>>>\n\n"
            f"ANSWER:<<<{inference_result_content}>>>\n\n"
            f"Previous Annotations:<<<{prev_annotations}>>>\n\n"
            "Listez vos annotations ci-dessous."
        )

        # Use the premium LLM to generate annotations
        annotation_response = self.premium_llm.invoke([
            SystemMessage(content="Vous êtes chargé de générer des annotations pour améliorer la réponse."),
            HumanMessage(content=annotation_prompt)
        ])

        # Clean up the annotation response
        annotations = annotation_response.content.strip()
        annotations = re.sub(r'[^\x20-\x7E\t\n\r]', '', annotations)
        annotations = re.sub(r'\\u[0-9A-Fa-f]{4}', '', annotations)

        # Save the annotations using add_agent_data
        HumanLLMMonitor.add_agent_data(self.agent_name, "llm_annotations", {
            'annotations': annotations,
            'annotation_prompt': annotation_prompt
        })

        # Prepare the return value
        ret = {
            "output_id": output_id,
            "suggestions": response.content,
            "improvement_prompt": improvement_prompt,
            "annotations": annotations,
            "annotation_prompt": annotation_prompt
        }

        smart_print(json.dumps(ret), self.agent_name, "CRITIC SUGGESTIONS", column_id=output_id,
                    optional=False)
        return ret

