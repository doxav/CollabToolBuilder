import inspect
import json
import pickle
import os
import re
import urllib.request
import subprocess
import hashlib
import time
import tkinter
import json
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.schema import AIMessage, HumanMessage, SystemMessage, FunctionMessage
import tkinter as tk
from tkinter import simpledialog, scrolledtext
from datetime import datetime
from utils.file_utils import *
import concurrent.futures

from langchain.vectorstores import Chroma, ElasticsearchStore
from config import OPENAI_API_KEY, PickleCacheActivated
import os
import openai

openai.api_key = OPENAI_API_KEY
os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY

def smart_print(message: str, agent_name = None, message_type = None, append=False):
    if 'IN_NOTEBOOK' not in globals():
        try: # test if IN_NOTEBOOK
            from IPython import get_ipython
            globals()['IN_NOTEBOOK'] = IN_NOTEBOOK = get_ipython().__class__.__name__ == 'ZMQInteractiveShell'
            print("Notebook mode = "+str(IN_NOTEBOOK))
        except:
            globals()['IN_NOTEBOOK'] = IN_NOTEBOOK = False
    else:
        IN_NOTEBOOK = globals()['IN_NOTEBOOK']

    if IN_NOTEBOOK and agent_name:
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
        AgentDisplayManager.write_to_agent(agent_name, message, element_name=message_type+" "+time_str, append=append)
    else:
        if append:
            print(message, end="", flush=True)
        else:
            print(message)

def smart_input(message: str, agent_name = None, message_type = None):
    if 'IN_NOTEBOOK' not in globals():
        try: # test if IN_NOTEBOOK
            from IPython import get_ipython
            globals()['IN_NOTEBOOK'] = IN_NOTEBOOK = get_ipython().__class__.__name__ == 'ZMQInteractiveShell'
            print("Notebook mode = "+str(IN_NOTEBOOK))
        except:
            globals()['IN_NOTEBOOK'] = IN_NOTEBOOK = False
    else:
        IN_NOTEBOOK = globals()['IN_NOTEBOOK']

    if False and IN_NOTEBOOK and agent_name:
        # import AgentDisplayManager from utils.jupyter_agents_display if AgentDisplayManager is not initialized
        if 'AgentDisplayManager' not in globals():
            try:
                from utils.jupyter_agents_display import AgentDisplayManager
            except:
                print("AgentDisplayManager cannot be imported/initialized")
                print(message)
                return
        return AgentDisplayManager.get_input(agent_name,message)
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
        with open(file_path, 'w') as file: file.write(initial_string)

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

def load_prompt(prompt_name, package_path="."):
    return load_text(f"{package_path}/prompts/{prompt_name}.txt")

def list_prompt_variants(prompt_name, package_path="."):
    base_name = prompt_name.split("@")[0]
    pattern = f"{package_path}/prompts/{base_name}@*.txt"
    variants = [filename[len(package_path)+9:-4] for filename in glob.glob(pattern)]
    return [base_name] + variants  # Include base prompt in the list

def save_prompt(prompt_name, text, package_path="."):
    prompt_file_path_name = f"{package_path}/prompts/{prompt_name}.txt"

    # if prompt_file_path_name exists, move existing file to prompt_file_path_name.timestamp (timestamp = datetime.now().isoformat())
    if f_exists(prompt_file_path_name):
        moved_file_path_name = prompt_file_path_name+datetime.now().strftime(".%H-%M-%S_%m-%d-%y")
        print(f"Moving existing prompt file {prompt_file_path_name} to {moved_file_path_name}")
        f_move(prompt_file_path_name, moved_file_path_name)

    print(f"Saving new prompt file {prompt_file_path_name}")

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
        print(f"Moving existing prompt file {prompt_file_path_name} to {moved_file_path_name}")
        f_move(prompt_file_path_name, moved_file_path_name)

    print(f"Saving new prompt file {prompt_file_path_name}")

    # Save the file
    return dump_text(text, prompt_file_path_name)

class UnifiedVectorDB:
    db_type = 'elasticsearch'  # can be 'elasticsearch' or 'chroma'
    es_url = 'http://127.0.0.1:9200'

    @staticmethod
    def check_db():
        if UnifiedVectorDB.db_type == 'elasticsearch':
            try:
                urllib.request.urlopen(UnifiedVectorDB.es_url, timeout=1)
            except urllib.error.URLError as e:
                print(f"Error: {e.reason} - {e}\nUnifiedVectorDB.es_url: {UnifiedVectorDB.es_url}\nPlease check if elasticsearch is running and reachable at the specified URL\nplease set UnifiedVectorDB.es_url = 'http://x.x.x.x:9200' in your config.py or search where it is set in your code.")
                exit(1)
        elif UnifiedVectorDB.db_type == 'chroma':
            print("Chroma DB check is not yet implemented")
        else:
            raise ValueError(f"Unsupported DB type: {UnifiedVectorDB.db_type}")

    def __init__(self, collection_name, embedding_function, persist_directory):
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
            self.db = ElasticsearchStore(
                index_name=self.collection_name,
                embedding=embedding_function,
                es_url=UnifiedVectorDB.es_url,
                distance_strategy="COSINE"
            )
            self._collection = self.db
            embedding_size=len(embedding_function.embed_query(""))
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
            return self.db.similarity_search_with_score(query, k=(k if k <= 50 else 50)) # k seems to crash when > 50

    # query( query_embeddings, query_texts, n_results, where, where_document, include)
    def query(self, query_text="", k=1, metadata_filter=None, metadata_filter_OR=False, custom_filter_chrome=None, custom_filter_es=None, sort_order=None):
        if UnifiedVectorDB.db_type == 'chroma':
            if metadata_filter and custom_filter_chrome is None:
                # filter on AND conditions: "filter":{'$and': [{'user_id': {'$eq': user_id}}, {'category_id': {'$eq': cat_id}}]}})
                # filter on list of values: "filter":{'user_id': {'$in': [user_id_1, user_id_2]}}})
                # filter on OR conditions: "filter":{'$or': [{'user_id': {'$eq': user_id}}, {'category_id': {'$eq': cat_id}}]}})
                filter_chroma=[]
                for key, value in metadata_filter.items():
                    # set sign to $eq if value is a string, else to $in
                    sign = '$eq' if isinstance(value, str) else '$in'
                    filter_chroma.append({key: {sign: value}})
                filter_chroma = {('$or' if metadata_filter_OR else '$and'): filter_chroma}
                if sort_order == 'asc' or sort_order == 'desc': # incompatible with knn search, so we rewrite query just keeping filters and sort
                    print("WARNING: sort not implemented for Chroma DB")
            return self.db.query(query_text, k=k, filter=filter_chroma)
        elif UnifiedVectorDB.db_type == 'elasticsearch':
            if metadata_filter and custom_filter_es is None:
                custom_filter_es=[]
                for key, value in metadata_filter.items():
                    custom_filter_es.append({"match":{f"metadata.{key}":value}})
                if metadata_filter_OR:
                    custom_filter_es = {"bool":{"should":custom_filter_es}}
            if sort_order == 'asc' or sort_order == 'desc': # incompatible with knn search, so we rewrite query just keeping filters and sort
                def custom_query(query_body: dict, query: str):
                    return {"query": {"bool":{"must":custom_filter_es}}, "sort":[{"metadata.time": {"order": sort_order}}]} # removed: , "size": k
                return self.db.similarity_search(query_text, k=(k if k <= 50 else 50), custom_query=custom_query) # k seems to crash when > 50
            else:
                return self.db.similarity_search(query_text, k=(k if k <= 50 else 50), filter=custom_filter_es) # k seems to crash when > 50
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

def search_for_external_knwoledge(description, url):
    """
    Search for documentation on Internet or ask Human
    """
    print(f"search_for_external_knwoledge: description={description}, url={url}")
    return _visual_input(f"search_for_external_knwoledge: description={description}, url={url}")

class HumanLLMMonitor:
    default_skip_rounds = 0
    step_id = 0
    function_list = None
    common_vectordb = None
    common_vectordb_embedding_function=OpenAIEmbeddings()
    common_vectordb_collection_name="human_llm_monitor_logs"
    common_vectordb_persist_directory="human_llm_monitor_vectordb"

    def __init__(self, system_prompt=None, CPS_env_type=None, agent_name=None, model_name=None, model_max_context_size=16000, llm=None, premium_llm=None, premium_model_name=None, premium_llm_by_default=False, num_parallel_inferences=1):
        self.system_prompt = system_prompt
        self.llm = llm if llm else None
        self.premium_llm = premium_llm if premium_llm else None
        self.CPS_env_type = CPS_env_type
        self.agent_name = agent_name or self.get_caller_class_name()
        # set in 1 line self.print_color is 32 for ActionAgent, 35 for CurriculumAgent, 31 for CriticAgent, 33 for SkillManager, 37 for else
        self.print_color = "32" if self.agent_name in ["ActionAgent", "CodingAgent"] else "35" if self.agent_name in ["CurriculumAgent","TaskIdentificationAgent"] else "31" if self.agent_name in ["CriticAgent","ValidationAgent"] else "33" if self.agent_name in ["SkillManager","CapitalizationAgent"] else "37"
        self.previous_templates = []
        self.previous_results = []
        self.comments = []
        self.skip_rounds = HumanLLMMonitor.default_skip_rounds
        self.log_data = []
        self.num_parallel_inferences = num_parallel_inferences
        self.llm_max_context_size = model_max_context_size
        self.model_name = model_name or "gpt-3.5-turbo"
        self.premium_model_name = model_name or "gpt-4-turbo"
        self.premium_llm_by_default = premium_llm_by_default

    def get_caller_class_name(self):
        # Returns the name of the class that called the current function
        return inspect.stack()[2][0].f_locals["self"].__class__.__name__

    def _max_tokens_ok(self, content):
        import tiktoken
        encoding = tiktoken.encoding_for_model(self.model_name)
        token_length = len(encoding.encode(content))

        if token_length > self.llm_max_context_size:
            print( f"\033[31mCANNOT SEND MESSAGE TO LLM:\n{content}\n\nToo many tokens in human message for LLM ({token_length}). Fallback to manual feedback.\033[0m" )
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

    def _before_inference(self, messages, function_calling, callable_system_message=None):
        comments = None
        initial_user_message = messages[1].content
        function_name = inspect.stack()[2].function
        while self.skip_rounds == 0:
            menu = f"\033[{self.print_color}m***** {self.agent_name}->{function_name}  BEFORE *****\nSYSTEM PROMPT:\n{messages[0].content}\n\nUSER MESSAGE:\n{messages[1].content}\n***** {self.agent_name}->{function_name} BEFORE *****\033[0m\n"
            if not self._max_tokens_ok(messages[0].content+"\n"+messages[1].content):
                menu += ("WARNING!!!! Max tokens exceeded, you should refactor user message or system prompt!\n")
            menu += ("A. Modify agent's 'role' / 'system prompt' (role, global context, constraints, examples).\n")
            menu += ("B. Add instruction or information to agent.\n")
            menu += ("C. Skip and set LLM output from recent outputs or manually define it.\n")
            menu += ("D. Log comments (not used by the model, just for information).\n")
            menu += ("E. See all previous results for this agent.\n")
            menu += ("F. See previous MODIFIED/SCORED/COMMENTED results for this agent.\n")
            menu += ("G. Skip human actions for N rounds.\n")
            menu += ("H. Exit program.\n")
            #menu += (f"I. Activate/de-activate function calling to allow model request external knowledge - current status: {function_calling}\n")
            menu += (f"J. Change num of parallel inferences - Current value={self.num_parallel_inferences}\n")
            if self.premium_llm_function: menu += (f"P. Proceed to inference using a PREMIUM LLM - Current value={self.premium_llm_by_default}\n")
            smart_print(menu, self.agent_name, "BEFORE inference action MENU")
            action = input(f"\n\033[32mBEFORE\033[0m inference @ {self.agent_name}-> Choose an action (or hit Enter for inference) :").upper()

            if action is None or action == "":
                return messages, comments, False, (self.premium_llm_by_default if self.premium_llm_function else False), function_calling
            
            elif action == "P" and self.premium_llm_function:
                return messages, comments, False, True, function_calling
            
            elif action == "A":
                start_time, action = time.time(), "A"
                new_template = None

                # List existing prompt variants including the base prompt
                prompt_variants = list_prompt_variants(self.system_prompt)
                output = ("Found the following prompt options:\n")
                for i, variant in enumerate(prompt_variants):
                    output += (f"{i+1}. {variant}\n")
                output += (f"{len(prompt_variants) + 1}. Ask LLM to generate a new variant of the current system prompt given my instructions\n")
                smart_print(output, self.agent_name, "PROMPT OPTIONS")

                variant_choice = input("Select a number to modify a prompt or create a new variant (or press Enter to continue with the current selection): ")
                if variant_choice.isdigit() and 0 < int(variant_choice) <= len(prompt_variants) + 1:
                    if int(variant_choice) == len(prompt_variants) + 1:
                        # Process to create a new variant
                        comments = input("Provide critic or feedback for the current prompt: ")
                        refine_prompt = _visual_input(f"Current system prompt:<<< {load_prompt(self.system_prompt)} >>>\n\nFeedback or critic: {comments}")
                        llm_output = self.llm_function([SystemMessage(content=load_prompt("improve_prompt_from_answer_critic")), HumanMessage(content=refine_prompt)])
                        new_template = llm_output.content
                    else:
                        self.system_prompt = prompt_variants[int(variant_choice) - 1]

                if input("Would you like first to get suggestions for a better prompt? (y/n): ").upper() == "Y":
                    if self.premium_llm_function:
                        llm_output = self.premium_llm_function([SystemMessage(content=load_prompt("system_prompt_refiner")), HumanMessage(content=f"PROMPT TO GET SUGGESTIONS FOR IMPROVEMENT:\n{load_prompt(self.system_prompt)}")])
                    else:
                        llm_output = self.llm_function([SystemMessage(content=load_prompt("system_prompt_refiner")), HumanMessage(content=f"PROMPT TO GET SUGGESTIONS FOR IMPROVEMENT:\n{load_prompt(self.system_prompt)}")])
                    smart_print(f"***** PROMPT SUGGESTIONS *****\n\033[33m{llm_output.content}\033[0m\n*************", self.agent_name, "PROMPT SUGGESTIONS")

                new_template = _visual_input(load_prompt(self.system_prompt) if new_template is None else new_template)
                smart_print(f"***** NEW PROMPT TEMPLATE:\n{new_template}\n*************", self.agent_name, "NEW PROMPT TEMPLATE")
                # Confirm that the user wants to modify the template
                confirm = input("Do you want to replace current prompt file template with your input? (y/n): ").upper()
                # Save prompt with tag options
                if confirm == "Y":
                    tag_option = input("Enter a tag for saving the prompt (leave blank for no tag, or 'same' to keep the current tag): ")
                    if tag_option.lower() == "same":
                        save_prompt_with_tag(self.system_prompt, new_template, "")
                    else:
                        save_prompt_with_tag(self.system_prompt, new_template, tag_option)

                    if callable_system_message:
                        messages[0] = callable_system_message()
                    else:
                        messages[0].content = new_template

                self.before_inference_option_times[action] += (time.time() - start_time)
                self.before_inference_option_counts[action] += 1

            elif action == "B":
                start_time, action = time.time(), "B"

                new_message = initial_user_message + "\n" + _visual_input(" ")
                confirm = input(f"***** NEW USER MESSAGE:\n{new_message}\n*************\nAre you sure you want to modify it? (y/n): ").upper()
                if confirm == "Y":
                    messages[1].content =  new_message

                self.before_inference_option_times[action] += (time.time() - start_time)
                self.before_inference_option_counts[action] += 1

            elif action == "C":
                start_time, action = time.time(), "C"

                log_entries, list_output = self._get_log_entries(self.agent_name, function_name), ""
                for idx, entry in enumerate(log_entries, start=1):
                    content = json.loads(entry.page_content)
                    text = (content['output_contents'][0]['content'].replace('\n', '\\') if content['output_contents'] else "") if isinstance(content['output_contents'], list) else content['output_contents']['content'].replace('\n', '\\')
                    date = entry.metadata['time'].split('.')[0]
                    list_output += (f"\033[94m{idx}.\033[0m {text[:100]}....{text[-100:]} #{entry.metadata['function_name']} @{date}\n")  # Display a snippet of each entry
                smart_print(list_output, self.agent_name, "LOG ENTRIES LIST")

                try: selected_index = int(input("Select the log entry number to load or 0/enter to manually enter LLM output: ")) - 1
                except: selected_index = -1
                if selected_index < 0 or selected_index >= len(log_entries):
                    llm_output = _visual_input("Enter LLM ANSWER/OUTPUT:\n")
                else:
                    selected_log_entry = json.loads(log_entries[selected_index].page_content)
                    llm_output = (selected_log_entry['output_contents'][0] if isinstance(selected_log_entry['output_contents'], list) else selected_log_entry['output_contents'])['content']

                self.before_inference_option_times[action] += (time.time() - start_time)
                self.before_inference_option_counts[action] += 1

                return messages, comments, llm_output, False, function_calling

            elif action == "D":
                start_time, action = time.time(), "D"
                comments = input("Enter your comment on the prompt: ")
                self.before_inference_option_times[action] += (time.time() - start_time)
                self.before_inference_option_counts[action] += 1

            elif action == "E":
                start_time, action = time.time(), "E"
                # list results from  HumanLLMMonitor.common_vectordb filtered by agent_name and function_name
                HumanLLMMonitor._check_and_init_vector_db()
                result = HumanLLMMonitor.common_vectordb.query(query_text="*", metadata_filter={"function_name":function_name, "agent_name":self.agent_name}, k=100)
                visual_result = "\n===============================\n".join([json.dumps(json.loads(item.page_content), indent=4, sort_keys=True).replace("\\n", "\n") for item in result])
                _visual_input(visual_result)
                self.before_inference_option_times[action] += (time.time() - start_time)
                self.before_inference_option_counts[action] += 1

            elif action == "F":
                start_time, action = time.time(), "F"
                # list results from  HumanLLMMonitor.common_vectordb filtered by agent_name and function_name, filtered on comments
                HumanLLMMonitor._check_and_init_vector_db()
                confirm = input("Do you want see:\n(A) all MODIFIED/SCORED/COMMENTED results.\n(B) INPUT modified only.\n(C) OUTPUT modified only.\n(D) SCORED only.\n(E) COMMENTED only.\nSelect your letter for choice or hit enter for all: ").upper()
                result = []
                if confirm == "A" or confirm == "" or confirm == "B":
                    result.extend(HumanLLMMonitor.common_vectordb.query(query_text="*", metadata_filter={"function_name":function_name, "agent_name":self.agent_name, "input_modified":True}, k=100))
                if confirm == "A" or confirm == "" or confirm == "C":
                    result.extend(HumanLLMMonitor.common_vectordb.query(query_text="*", metadata_filter={"function_name":function_name, "agent_name":self.agent_name, "output_modified":True}, k=100))
                if confirm == "A" or confirm == "" or confirm == "D":
                    result.extend(HumanLLMMonitor.common_vectordb.query(query_text="*", metadata_filter={"function_name":function_name, "agent_name":self.agent_name, "scored":True}, k=100))
                if confirm == "A" or confirm == "" or confirm == "E":
                    result.extend(HumanLLMMonitor.common_vectordb.query(query_text="*", metadata_filter={"function_name":function_name, "agent_name":self.agent_name, "commented":True}, k=100))
                visual_result = "\n===============================\n".join([json.dumps(json.loads(item.page_content), indent=4, sort_keys=True).replace("\\n", "\n") for item in result])
                _visual_input(visual_result, filetype="json")
                self.before_inference_option_times[action] += (time.time() - start_time)
                self.before_inference_option_counts[action] += 1

            elif action == "G":
                rounds = int(input("Skip for how many rounds? "))
                self.skip_rounds = rounds

            elif action == "H":
                # check if NOTEBOOK mode
                if 'IN_NOTEBOOK' in globals() and globals()['IN_NOTEBOOK']:
                    # access to AgentDisplayManager.export_to_html() which is not registered in this file but in the notebook

                    raise SystemExit("I just wanted to stop!")
                else:
                    exit()
            
            elif action == "I":
                function_calling = not function_calling
                smart_print(f"function_calling is now {function_calling}", self.agent_name, "function_calling")

            elif action == "J":
                try: self.num_parallel_inferences = int(input("Enter new value for num_parallel_inferences: "))
                except: self.num_parallel_inferences = 1

            proceed = input("Proceed to inference (y/n) ?" + (" You can hit 'p' to proceed using a premium llm." if self.premium_llm_function else "")).lower()
            if proceed == "y" or proceed == "":
                return messages, comments, False, False, function_calling
            elif self.premium_llm_function and proceed == "p":
                return messages, comments, False, True, function_calling
        # if self.skip_rounds > 0:
        #     self.skip_rounds -= 1 # decrement skip_rounds is done in _after_inference
        return messages, comments, False, False, function_calling

    def _after_inference(self, inference_result_msg, color="37", output_id=None, outputs_count=None):
        comments, score = None, None
        if inference_result_msg is None:
            # enable to request inference_result_msg.content to be None
            inference_result_msg = type('InferenceResult', (object,), {'content': None})
        while self.skip_rounds == 0:
            multiple_ref = (f"OUTPUT \033[31m{output_id} OUT OF {outputs_count}\033[0m OUTPUTS" if (output_id and outputs_count and (outputs_count>1)) else "")
            menu = (f"\033[{self.print_color}m***** {self.agent_name}->{inspect.stack()[2].function} AFTER *****\nLLM ANSWER:\n{inference_result_msg.content}\n***** {self.agent_name}->{inspect.stack()[2].function} AFTER *****\033[0m{multiple_ref}\n")
            menu += ("A. Manually set/modify the answer/output (I don't want to try to improve agent's system prompt).\n") # je voudrais le corriger uniquement pour demander une suggestion d'amélioration du prompt (d'un autre côté, je peux aussi le faire dans le menu précédent)
            menu += ("B. Critic this answer/output to get an improved answer/output.\n")
            menu += ("C. Find a better Prompt by providing critic and ideal answer.\n")
            menu += ("D. Evaluate & comment answer (Score between 0(worst)-1(top), and explain) to improve future results by using scored/commented examples.\n")
            menu += ("E. Go back BEFORE inference to improve system prompt or add information to user message.\n")
            menu += ("G. Skip human actions for N rounds.\n")
            menu += ("H. Exit program.\n")
            smart_print(menu, self.agent_name, "AFTER inference action MENU"+ (f" {output_id}/{outputs_count}"if (output_id and outputs_count and (outputs_count>1)) else ""))
            action = input(f"\n\033[32mAFTER\033[0m inference @ {self.agent_name}-> Choose an action (or hit Enter for inference) :").upper()

            if action is None or action == "":
                return inference_result_msg, comments, score
            elif action == "A":
                start_time, action = time.time(), "A"
                inference_result_msg.content = _visual_input(inference_result_msg.content)
                smart_print(f"***** NEW USER MESSAGE:\n{inference_result_msg.content}\n*************", self.agent_name, "NEW USER MESSAGE")
                self.after_inference_option_times[action] += (time.time() - start_time)
                self.after_inference_option_counts[action] += 1

            elif action == "B":
                start_time, action = time.time(), "B"
                while True:
                    comments = input("Provide critic/feedback/request: ")
                    refine_prompt = f"Refine the answer: {inference_result_msg.content}.\n*******************\nHuman provided feedback: {comments}"
                    llm_output = self.llm_function([SystemMessage(content="You are a helpful assistant"), HumanMessage(content=refine_prompt)])
                    smart_print(f"***** REFINED ANSWER:\n\033[33m{llm_output.content}\033[0m\n".replace("\\n", "\n"), self.agent_name, "REFINED ANSWER")
                    if input("Is the task refinement adequate? (yes/no): ").strip().lower() in ["yes", "y"]:
                        inference_result_msg.content = llm_output.content
                        break
                self.after_inference_option_times[action] += (time.time() - start_time)
                self.after_inference_option_counts[action] += 1

            elif action == "C":
                start_time, action = time.time(), "C"
                comments = input("First enter your critic here (then modify answer to get ideal answer): ")
                ideal_answer = _visual_input(inference_result_msg.content)
                refine_prompt = f"Current system prompt:<<< {load_prompt(self.system_prompt)} >>>\n\nPrompt's answer:<<< {inference_result_msg.content} >>>\n\nPrompt's answer critic:{comments}\n\nPrompt's ideal Answer:<<< {ideal_answer} >>>"
                smart_print(f"***** PROMPT FOR IMPROVEMENT *****\n{refine_prompt}", self.agent_name, "PROMPT FOR IMPROVEMENT")
                llm_output = self.premium_llm([SystemMessage(content=load_prompt("improve_prompt_from_answer_critic")), HumanMessage(content=refine_prompt)])
                smart_print(f"***** RECOMMENDATION OPEN FOR EDITION *****\n", self.agent_name, "RECOMMENDATION OPEN FOR EDITION")
                new_template = _visual_input(llm_output.content)
                smart_print(f"***** NEW PROMPT TEMPLATE:\n{new_template}\n*************", self.agent_name, "NEW PROMPT TEMPLATE")
                # Confirm that the user wants to modify the template
                confirm = input("Do you want to replace current prompt file template with your input? (y/n): ").upper()
                # Save prompt with tag options
                if confirm == "Y":
                    tag_option = input("Enter a tag for saving the prompt (leave blank for no tag, or 'same' to keep the current tag): ")
                    if tag_option.lower() == "same":
                        save_prompt_with_tag(self.system_prompt, new_template, "")
                    else:
                        save_prompt_with_tag(self.system_prompt, new_template, tag_option)

                self.after_inference_option_times[action] += (time.time() - start_time)
                self.after_inference_option_counts[action] += 1

            elif action == "D":
                start_time, action = time.time(), "D"
                while True:
                    score = float(input("Give a note for the result between 0.0 (worst) and 1.0 (top), or 0 for bad, 1 for good: "))
                    # if score is not between 0 and 1, then set to None and print error
                    if score < 0 or score > 1:
                        score = None
                        print(f"\033[31mInvalid score: {score}\033[0m")
                    else: break
                comments = input("Comment on the result: ")
                self.after_inference_option_times[action] += (time.time() - start_time)
                self.after_inference_option_counts[action] += 1

            elif action == "E":
                return -1, comments, score
            elif action == "G":
                rounds = int(input("Skip for how many rounds? "))
                self.skip_rounds = rounds
            elif action == "H":
                exit()

            proceed = input("Continue 'y' (or 'n' to go back to menu) ? ")
            if proceed == "y" or proceed == "":
                return inference_result_msg, comments, score
        if self.skip_rounds > 0:
            smart_print(f"\033[{self.print_color}m****{self.agent_name}>{inspect.stack()[2].function} LLM ANSWER content****\n{inference_result_msg.content}\n*****************\033[0m", self.agent_name, "LLM ANSWER content")
            self.skip_rounds -= 1
        return inference_result_msg, comments, score

    @staticmethod
    def _check_and_init_vector_db():
        if HumanLLMMonitor.common_vectordb is None:
            HumanLLMMonitor.common_vectordb = UnifiedVectorDB(
                collection_name=HumanLLMMonitor.common_vectordb_collection_name,
                embedding_function=HumanLLMMonitor.common_vectordb_embedding_function,  # Replace with your actual embedding function
                persist_directory=HumanLLMMonitor.common_vectordb_persist_directory    # Replace with your actual persist directory path
            )

    # staticmethod get my host ID
    @staticmethod
    def get_host_id():
        import socket, uuid
        return socket.gethostname()+"-"+str(uuid.getnode())

    # staticmethod call llm_function (langchain ChatOpenAI) with optional function_call and process function call until result is provided
    @staticmethod
    def call_llm_function_with_function_call(llm_function, messages, function_call=None, function_list=None, max_calls=5):
        # test if HumanLLMMonitor.function_list exists
        if function_list is None:
            if HumanLLMMonitor.function_list is None:
                function_list = [{
                "name": "search_for_external_knwoledge",
                "description": "Call this function to search for external knowledge when the model's confidence is low or its information might be too outdated",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "description": {"type": "string", "description": "precise description of the information to be provided"},
                        "url": {"type": "string", "description": "Google search url link find this information (ie. it should start by https://www.google.com/search?q= )"},
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
                    result = input(f"Uknown function: {function_name} -- please provide result manually:\n")
                messages.append(FunctionMessage(content=result, name=function_name, arguments=function_args_str))

        return output

    def _log_entry(self, function_name, input_contents, output_contents, input_modified=False,
                   skipped_inference=False, input_comments=None, output_comments=None, output_llm_raw=None,
                   output_modified=False, inference_time=None, message_tokens=None, score=None, use_premium_llm=False, call_duration=None, skip_rounds=None):
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
            "call_duration": call_duration
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
            "use_premium_llm": use_premium_llm,
            "commented": (input_comments is not None or output_comments is not None),
            "scored": (score is not None),
        }

        HumanLLMMonitor._check_and_init_vector_db()

        HumanLLMMonitor.common_vectordb.add_texts(
            texts=[serialized_entry],
            metadatas=[tags]
        )

    def CallHumanLLM(self, original_input_messages=None, llm_function=None, premium_llm_function=None, callable_system_message=None, system_prompt_template=None, user_message=None, return_message_content_only=True, function_calling=False, temperature=0.7, timeout_seconds=90, stream_output=True):
        # Define a helper function to perform the LLM calls for parallel inference.
        def perform_llm_call(input_msg, use_premium, func_calling, temperature, stream_output=True, color_id=None):
            if use_premium:
                func = premium_llm_function if not func_calling else HumanLLMMonitor.call_llm_function_with_function_call
            else:
                func = llm_function if not func_calling else HumanLLMMonitor.call_llm_function_with_function_call

            if stream_output:
                if color_id is None or color_id <= 0:
                    start_color, end_color = "", ""
                else:
                    start_color, end_color = ["\033[91m", "\033[92m", "\033[93m", "\033[94m", "\033[95m", "\033[96m", "\033[97m"][color_id % 7], "\033[0m"
                final_output = ""  # Initialize an empty string to hold the full response
                smart_print("", self.agent_name, "Inference streaming output")
                for chunk in func.stream(input_msg, temperature=temperature):  # Ensure 'llm' is correctly initialized with temperature
                    smart_print(start_color+chunk.content+end_color, self.agent_name, "Inference streaming output", append=True)
                    final_output += chunk.content  # Concatenate each chunk to build the full response
                return AIMessage(content=final_output) # Return the concatenated full respons
            else:
                return func(input_msg, temperature=temperature)
        
        smart_print(f"\033[{self.print_color}m****{self.agent_name}>{inspect.stack()[1].function} calling HumanLLMMonitor****\033[0m", self.agent_name, "HumanLLMMonitor")
        if system_prompt_template: self.system_prompt = system_prompt_template
        if llm_function is None: llm_function = self.llm
        if premium_llm_function is None: premium_llm_function = self.premium_llm if self.premium_llm else None
        self.llm_function = llm_function
        self.premium_llm_function = premium_llm_function
        if original_input_messages is None: original_input_messages = [SystemMessage(content=load_prompt(system_prompt_template)), HumanMessage(content=user_message)]
        input_contents_str0, input_contents_str1 = str(original_input_messages[0].content), str(original_input_messages[1].content)

        self.before_inference_option_times = {opt: 0 for opt in 'ABCDEF'}
        self.before_inference_option_counts = {opt: 0 for opt in 'ABCDEF'}
        self.after_inference_option_times = {opt: 0 for opt in 'ABCD'}
        self.after_inference_option_counts = {opt: 0 for opt in 'ABCD'}
        call_start_time = time.time()

        while True:
            if self.skip_rounds > 0:
                smart_print(f"\033[{self.print_color}m****{self.agent_name}>{inspect.stack()[2].function} skipping HumanLLMMonitor for {self.skip_rounds} rounds****\033[0m", self.agent_name, "Skipping round")

            # Pre-inference human intervention
            llm_input_messages, input_comments, skip_inference, use_premium_llm, function_calling = self._before_inference(original_input_messages, function_calling, callable_system_message)
            
            start_time = datetime.now()
            if llm_input_messages and not skip_inference:
                # Use concurrent futures to parallelize the LLM calls.
                outputs = []
                with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_parallel_inferences) as executor:
                    futures = [executor.submit(perform_llm_call, llm_input_messages, use_premium_llm, function_calling, temperature, stream_output, _) for _ in range(self.num_parallel_inferences)]
                    for future in futures:
                        try:
                            llm_response = future.result(timeout=timeout_seconds)
                            outputs.append(llm_response)
                            smart_print(f'\033[0m**** New inference result recieved and added to outputs as #{len(outputs)}\033[0m:\n{llm_response.content}\n\033[9mEND OF #{len(outputs)}****\033[0m', self.agent_name, "NEW inference result recieved")
                        except concurrent.futures.TimeoutError:
                            smart_print('A task ran longer than the allotted timeout and was cancelled.', self.agent_name, "Inference result TIMEOUT")
                        except Exception as exc:
                            smart_print(f'Generated an exception: {exc}', self.agent_name, "Inference result EXCEPTION")
                    # Wait for all the futures to complete before continuing.
                    concurrent.futures.wait(futures)
                if len(outputs) == 0:
                    smart_print(f'**** No inference result recieved, set output to None', self.agent_name, "NO inference recieved")
                    llm_outputs = None
                elif len(outputs) == 1:
                    # smart_print(f'**** One inference result recieved, set output to it', self.agent_name, "ONE inference recieved")
                    llm_outputs = outputs
                else:
                    smart_print(f'**** {len(outputs)} inference results received - You will be requested to select which ones to keep', self.agent_name, "MULTIPLE inferences recieved")
                    # Ask user to select output from the parallel inferences.
                    if self.skip_rounds > 0:
                        selected_output = ""
                    else:
                    #     selected_output = input(f'Select the output id number to keep (1-{len(outputs)}), or comma separated list of outputs, or hit Enter to keep all of them: ')
                    # # selected_output could be a comma separated list of output ids, or a single output id, or empty, process it
                    # if selected_output != "":
                    #     # remove any charactere that is not a digit or a comma
                    #     selected_output = re.sub(r"[^0-9,]", "", selected_output)
                    #     selected_output = [int(x) for x in selected_output.strip().split(",")]
                    #     llm_outputs = [outputs[i-1] for i in selected_output]
                    #     #if len(llm_output) == 1: llm_output = llm_output[0]
                    # else:
                        llm_outputs = outputs
            else:  # Skip the LLM inference.
                llm_outputs = [AIMessage(content=skip_inference)]
            end_time = datetime.now()
            raw_llm_outputs = [(output.content if output else None) for output in llm_outputs] if isinstance(llm_outputs, list) else None

            output_messages, output_comments, score = [], [], []
            if llm_outputs:
                if len(llm_outputs) > 1:
                    smart_print("**** Multiple LLM ANSWERS > we will process POST INFERENCE for each ****", self.agent_name, "Multiple LLM ANSWERS", append=True)
                for counter, llm_output in enumerate(llm_outputs, start=1):
                    if len(llm_outputs) > 1:
                        smart_print(f"\033[31mMULTI-INFERENCE OUTPUT #{counter} > \033[0m", self.agent_name, "POST INFERENCE", append=True)
                    # Post-inference human intervention
                    output_messages_instance, output_comments_instance, score_instance = self._after_inference(llm_output, output_id=counter, outputs_count=len(llm_outputs))
                    output_messages.append(output_messages_instance)
                    if output_messages_instance == -1:
                        break
                    output_comments.append(output_comments_instance)
                    score.append(score_instance)
                # test if any of output_messages instance != -1, break if True
                if any([output_messages_instance == -1 for output_messages_instance in output_messages]):
                    original_input_messages[0].content, original_input_messages[1].content = input_contents_str0, input_contents_str1
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
            
            input_modified=((llm_input_messages[0].content + "\n" + llm_input_messages[1].content) != (input_contents_str0 + "\n" + input_contents_str1)),
            skipped_inference=True if skip_inference else False,
            skip_rounds=self.skip_rounds,
            input_comments=input_comments,
            output_comments=output_comments,
            output_llm_raw=raw_llm_outputs,
            # test if any  output_modified=(output_messages.content != raw_llm_output), 
            output_modified=any(output_message.content != raw for output_message, raw in zip(output_messages, raw_llm_outputs)),
            score=score, 
            message_tokens=None,
            use_premium_llm=use_premium_llm,
            call_duration=call_duration
        )
        return output_messages.content if return_message_content_only else output_messages

 