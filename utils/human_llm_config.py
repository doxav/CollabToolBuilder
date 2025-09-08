import logging, json, uuid, os, subprocess, time, pickle, requests, re
from datetime import datetime
# from config import MODELS_CONFIG_LIST, vector_store_type
from config import MODELS_CONFIG_LIST
from utils.websocket_server import WebsocketServer, WebSocketServerConfig
from utils.llm_utils import (
    create_Nmajority_chain,
    UnifiedVectorDBConfig, LLMConfig, UserSession, InferenceTracking, TaskHistory,
    UnifiedVectorDB
)
from typing import List, Dict, Any, Optional
from langchain_openai import ChatOpenAI
from langchain_core.messages.ai import AIMessage


class HumanLLMConfig:
    __instance = None

    def __new__(cls):
        if cls.__instance is None:
            cls.__instance = super(HumanLLMConfig, cls).__new__(cls)
        return cls.__instance

    def __init__(self):
        if hasattr(self, '_initialized'):
            self.logger = logging.getLogger(__name__)
            return
        self._initialized = True
        self.discord_webhook = "https://discord.com/api/webhooks/1329927709106638930/FaFm6Dozd0xlQvL8hBHn3JHczDazfe2V9hHBWKWj7-igTniTHtsJBCmiFLCh-oKeZ6Mz"

        self.common_vectordb_config = UnifiedVectorDBConfig()
        self.common_vectordb = None

        self.ws_server_config = WebSocketServerConfig()
        self.ws_server = None
        self.use_websocket = False
        
        # Initialize vector databases for tasks
        self.db_collection_success = "successful_tasks"
        self.db_collection_failed = "failed_tasks"

        self.db_learnt_tasks = None # TODO: remove once migration to common_vectordb COMPLETE
        self.db_failed_tasks = None # TODO: remove once migration to common_vectordb COMPLETE

        self.step_id = 0
        self.function_list = None
        self.default_skip_rounds = 0

        self.llm_config = LLMConfig()
        self.user_session = UserSession()
        self.task_history = TaskHistory()

        self.special_criteria = None
        self.automation = None

        self.initialized = False

    def initialize(self):
        if self.initialized:
            return
        self.initialized = True
        self.configure_vector_store()
        if self.use_websocket:
            self.init_ws_server()
        self.common_vectordb = UnifiedVectorDB(self.common_vectordb_config, check_db=True)
        
    def init_ws_server(self):
        if self.ws_server is None:
            self.ws_server = WebsocketServer(self.ws_server_config)

    def configure_vector_store(self):
        if self.common_vectordb_config.embedding_function:
            self.common_vectordb_config.set_common_vectordb_embedding_function()

    def get_llmORchains_list(self):
        if MODELS_CONFIG_LIST is not None:
            return {
                "default_llm": ChatOpenAI(
                    model=MODELS_CONFIG_LIST["basic_gpt"],
                    cache=False,
                    temperature=0.
                ),
                "premium_llm": ChatOpenAI(
                    model=MODELS_CONFIG_LIST["smart_gpt"],
                    cache=False,
                    temperature=0.
                ),
                # "coder_llm": ChatOllama(
                #     model=MODELS_CONFIG_LIST["code_gpt"],
                #     cache=False,
                #     temperature=0.
                # ),
                "3_majority_chain": create_Nmajority_chain(
                    map_model_name=MODELS_CONFIG_LIST["basic_gpt"],
                    reduce_model_name=MODELS_CONFIG_LIST["basic_gpt"],
                    num_models=3
                ),
                "10_majority_chain": create_Nmajority_chain(
                    map_model_name=MODELS_CONFIG_LIST["basic_gpt"],
                    reduce_model_name=MODELS_CONFIG_LIST["basic_gpt"],
                    num_models=10
                )
            }
        else:
            # Default, if not precised, we take GPT from OpenAI.
            return {
                "default_llm": ChatOpenAI(
                    model_name="gpt-4.1-nano",
                    cache=False,
                    temperature=0.
                ),
                "premium_llm": ChatOpenAI(
                    model_name="gpt-4.1-mini-2025-04-14",
                    cache=False,
                    temperature=0.
                ),
                # "coder_llm": ChatOpenAI(
                #     model_name="gpt-4.1-nano",
                #     cache=False,
                #     temperature=0.
                # ),
                "3_majority_chain": create_Nmajority_chain(
                    map_model_name="gpt-4.1-nano",
                    reduce_model_name="gpt-4.1-nano",
                    num_models=3
                ),
                "10_majority_chain": create_Nmajority_chain(
                    map_model_name="gpt-4.1-nano",
                    reduce_model_name="gpt-4.1-nano",
                    num_models=10
                )
            }

    def add_learnt_task(self, serialized_entry, tags):
        # self.task_history.add_completed_task(serialized_entry)
        #self.db_learnt_tasks._add_texts(texts=[serialized_entry], metadatas=[tags])
        self.common_vectordb.log_agent_data(agent_name="HumanLLMConfig", data_key="learnt_task", data_value=serialized_entry, function_name=None, task_id=False, before_after=None, user_id=None, step_id=None, task_type="learnt", score=None, metadata={**tags, "task_type": "learnt"})

    def add_failed_task(self, serialized_entry, tags):
        # self.task_history.add_failed_task(serialized_entry)
        #self.db_failed_tasks._add_texts(texts=[serialized_entry], metadatas=[tags])
        self.common_vectordb.log_agent_data(agent_name="HumanLLMConfig", data_key="failed_task", data_value=serialized_entry, function_name=None, task_id=False, before_after=None, user_id=None, step_id=None, task_type="failed", score=None, metadata={**tags, "task_type": "failed"})

    def get_user_id(self):
        return self.user_session.get_user_id()

    def log_agent_data(
        self,
        agent_name,
        data_key,
        data_value,
        function_name=None,
        task_id=False,
        before_after=None,
        user_id=None,
        step_id=None,
        task_type=None,
        score=None,
        metadata=None
    ):
        """Stores agent-specific data with additional metadata.
        Elasticsearch generates an 'id' automatically and includes it in the metadata.
        """
        return self.common_vectordb.log_agent_data(
            agent_name=agent_name,
            data_key=data_key,
            data_value=data_value,
            function_name=function_name,
            task_id=task_id,
            before_after=before_after,
            user_id=user_id,
            step_id=step_id,
            task_type=task_type,
            score=score,
            metadata=metadata
        )
        # if isinstance(data_value, dict):
        #     serialized_data = json.dumps(data_value)
        # else:
        #     serialized_data = json.dumps({data_key: data_value})

        # # Generate UUID for task_id
        # task_id = str(uuid.uuid4()) if task_id else False

        # tags = metadata or {}
        # tags.update({
        #     "agent_name": agent_name,
        #     "data_key": data_key
        # })
        # self.logger.info(f"Adding agent data: {tags}")
        # self.logger.info(f"User ID: {user_id}")
        # if user_id is None:
        #     user_id = self.get_user_id()
        # if function_name:
        #     tags["function_name"] = function_name
        # if task_id:
        #     tags["task_id"] = task_id
        # if before_after:
        #     tags["before_after"] = before_after
        # if user_id:
        #     tags["user_id"] = user_id
        # if step_id:
        #     tags["step_id"] = step_id
        # if task_type:
        #     tags["task_type"] = task_type
        # if score is not None:
        #     tags["score"] = score
        # tags["date"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")

        # self.common_vectordb._add_texts(texts=[serialized_data], metadatas=[tags])

    def get_agent_data(
        self,
        agent_name=None,
        data_key=None,
        task_id=None,
        function_name=None,
        before_after=None,
        user_id=None,
        step_id=None,
        task_type=None,
        score=None,
        metadata_filter=None,
        sort_order=None,
        k=5,
        start_index=0,
        end_index=None,
        query_text='*',
        new_storage=True
    ):
        """Retrieves agent-specific data based on the agent name, data key, and additional metadata.
        Supports pagination by specifying start and end indices.
        """
        return self.common_vectordb.get_agent_data(
            agent_name=agent_name,
            data_key=data_key,
            task_id=task_id,
            function_name=function_name,
            before_after=before_after,
            user_id=user_id,
            step_id=step_id,
            task_type=task_type,
            score=score,
            metadata_filter=metadata_filter,
            sort_order=sort_order,
            k=k,
            start_index=start_index,
            end_index=end_index,
            query_text=query_text,
            new_storage=new_storage
        )
        # metadata = {}
        # if agent_name is not None:
        #     metadata["agent_name"] = agent_name
        # if data_key is not None:
        #     metadata["data_key"] = data_key
        # if function_name is not None:
        #     metadata["function_name"] = function_name
        # if task_id is not None:
        #     metadata["task_id"] = task_id
        # if before_after is not None:
        #     metadata["before_after"] = before_after
        # if user_id is not None:
        #     metadata["user_id"] = user_id
        # if step_id is not None:
        #     metadata["step_id"] = step_id
        # if task_type is not None:
        #     metadata["task_type"] = task_type
        # if score is not None:
        #     metadata["score"] = score
        # if metadata_filter:
        #     metadata.update(metadata_filter)

        # # Fetch results with a large 'k' to ensure we have enough data
        # max_k = end_index if end_index is not None else k
        # results = self.common_vectordb._query(
        #     query_text=query_text,
        #     metadata_filter=metadata,
        #     sort_order=sort_order,
        #     k=max_k
        # )

        # # Apply pagination
        # paginated_results = (
        #     results[start_index:end_index]
        #     if end_index is not None
        #     else results[start_index:]
        # )

        # ret = []
        # for item in paginated_results:
        #     try:
        #         temp = json.loads(item.page_content)
        #     except Exception as e:
        #         temp = item.page_content
        #     if new_storage:
        #         ret.append(temp)
        #         # ret.append({data_key: temp} if (isinstance(temp, dict) and data_key in temp and len(temp) == 1) else temp) # Chroma and ES do not store json the same way, so we need to adapt
        #     else:
        #         val = None
        #         if isinstance(temp, dict):                
        #             if data_key in temp:
        #                 val = temp[data_key]
        #         ret.append(val)
        # return ret, results

    def get_tasks(self, page_size: int = 200, nb_pages: int = 1, id_last_task: Optional[str] = None):
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

        _, tasks = self.get_agent_data(
            data_key=data_key,
            k=end_index,
            start_index=start_index,
            end_index=end_index
        )
        # Check if there is a newer task (if id_last_task is not the last task of the list)
        if id_last_task and id_last_task != "None":
            modif = False
            for i, task in enumerate(tasks):
                if task.metadata["task_id"] == id_last_task and i + 1 < len(tasks):
                    tasks = tasks[i+1:]
                    modif = True
                    break
            if not modif:
                return "None"
        ret = []
        for task in tasks:
            ret += [{
                "task_id": task.metadata["task_id"],
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
            if "task_type" in task.metadata:
                ret[-1]['task_type'] = task.metadata['task_type']
            if "task_details" in task.metadata:
                ret[-1]['task_details'] = task.metadata['task_details']

        # Trier la liste par la clé 'date', du plus récent au plus ancien
        ret = sorted(ret, key=lambda x: x['date'], reverse=True)

        return json.dumps(ret)

    def goto_task(
        self,
        task_id: str,
        automatic: str = None,
        special_criteria: dict = None,
        task_details: str = None
    ):
        """
        Retrieve the task from the database and start processing the task.
        Parameters:
            task_id (str): The identifier of the task to retrieve from the database.
            automatic (str): If True, the function will run the loop in automatic mode.
            special_criteria (dict): The special criteria to use for the task.
        """
        # Retrieve the task from the database
        _, saved_task = self.get_agent_data(data_key="saved_task", task_id=task_id)
        task = {
            'before_after': saved_task[0].metadata['before_after'],
            'agent_name': saved_task[0].metadata['agent_name'],
            'task_type': saved_task[0].metadata['task_type'],
            'content': saved_task[0].page_content,
            'date': saved_task[0].metadata['date']
        }
        if 'step_id' in saved_task[0].metadata:
            task['step_id'] = saved_task[0].metadata['step_id']
        if 'user_id' in saved_task[0].metadata:
            task['user_id'] = saved_task[0].metadata['user_id']
        if task_details:
            task['task_details'] = task_details
        # Create a pickle directory if it does not exist
        if not os.path.exists('pickle'):
            os.makedirs('pickle')

        for key in special_criteria:
            if 'num_parallel_inferences' in key and special_criteria[key] == 0:
                special_criteria[key] = 1

        # Serializing variables to pickle file
        variables_to_pickle = {
            'saved_task': task,
            'automatic': bool(automatic),
            'special_criteria': special_criteria
        }

        # Save variables to pickle file
        filename = f"variables_{self.get_user_id()}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        with open(f"pickle/{filename}.pkl", 'wb') as f:
            pickle.dump(variables_to_pickle, f)

        if not os.path.exists('goto_output'):
            os.makedirs('goto_output')

        with open(f'./goto_output/output_{task_id}_{self.get_user_id()}.log', 'w') as f:
            f.write("")

        # Execute the bash command with unbuffered output and capture its output
        subprocess.Popen(
            ['bash', '-c', f'python3 -u learn.py --proxy --secret --pickle_name {filename} > ./goto_output/output_{task_id}_{self.get_user_id()}.log 2>&1'],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )
        # Initialize the link variable
        link = None
        with open(f"./goto_output/output_{task_id}_{self.get_user_id()}.log", "r") as logfile:
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
                    # display it on discord
                    message = (f"**XP ID:** {self.common_vectordb_config.unique_collection_id}\n**User ID:** {self.get_user_id()}\n**Task Link:** {link}\n**Task Details:** {task_details}")
                    self.logger.info(f"Discord message: {message}")
                    self.send_to_discord(message)
                    break

        # Return the link if found
        if link is None:
            self.logger.info("The link was not found in the output file.")

        return link

    def send_to_discord(self, message: str):
        payload = {"content": message}
        headers = {"Content-Type": "application/json"}
        response = requests.post(self.discord_webhook, data=json.dumps(payload), headers=headers)
        if response.status_code == 204:
            self.logger.info("Message sent to Discord successfully.")
        else:
            self.logger.info(f"Failed to send message to Discord. Status code: {response.status_code}")

    def load_prompt_template(self, prompt_name, template_data=None, directory=None):
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
        else:
            prompt_path = os.path.join(directory, f"{prompt_name}.txt")

            try:
                with open(prompt_path, 'r') as file:
                    template_content = file.read()

                    # Apply template data if provided
                    if template_data:
                        template_content = template_content.format(**template_data)

            except FileNotFoundError:
                self.logger.error(f"WARNING: The prompt file '{prompt_name}.txt' was not found in the directory '{directory}', considering prompt_name as a prompt string <<<{prompt_name}>>>")
                template_content = prompt_name
            except KeyError as e:
                raise KeyError(f"Missing key {e} in template data for prompt '{prompt_name}'")

        prompt_content = self.common_vectordb.populate_few_shot_tags(template_content)

        return prompt_content

    def populate_few_shot_tags(self, prompt):
        """
        Replace tags by data: replace every `few_shots: { … }` tag in the prompt, parse its JSON payload,
        generate few-shot examples for it, and replace the entire tag with those examples

        Returns:
            str: The modified prompt content with few shots inserted.
        """
        if prompt is None:
            return ""
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
                combined_criteria = self.combine_criteria([data])
                # Get the few shots string
                few_shots_str = self.get_few_shot_examples(combined_criteria)
                # Replace the tag with the few shots string
                prompt = prompt[:start] + few_shots_str + prompt[end:]
            except json.JSONDecodeError as e:
                print(f"Error decoding 'few_shots' tag: {e}")
                print(f"Faulty JSON: {data_str}")  # Debug
                continue
        return prompt

    def combine_criteria(self, criteria_list):
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

    def get_few_shot_examples(self, few_shots_params) -> str:
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
                    "global_prefix": f"\n{params['sources']} tasks : <<",
                    "global_suffix": ">>\n",
                    "item_prefix": "\n|",
                    "item_suffix": "|"
                }

            if params['sources'] == "learnt":
                examples = self.get_learnt_tasks(
                    query_text=params['query_text'],
                    k=params['num'],
                    metadata_filter=params['metadata_filter'],
                    sort_order=params['sort_order'],
                    similarity_search=params['similarity_search']
                )
            elif params['sources'] == "failed":
                examples = self.get_failed_tasks(
                    query_text=params['query_text'],
                    k=params['num'],
                    metadata_filter=params['metadata_filter'],
                    sort_order=params['sort_order'],
                    similarity_search=params['similarity_search']
                )
            else:
                examples = self.common_vectordb._query(
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

            formatted_examples = self.format_examples(examples, format_criteria, separators)
            all_formatted_examples.append(formatted_examples)

        return "\n".join(all_formatted_examples)

    def initialize_class_db(self, force=False):
        if self.common_vectordb_config.common_vectordb_embedding_function is None:
            raise ValueError("embeddingfunction must be set to allow HumanLLM to manage tasks and other memories")
        # if self.db_learnt_tasks is None or force:
        #     self.db_learnt_tasks = UnifiedVectorDB(
        #         UnifiedVectorDBConfig(
        #             collection_name=self.db_collection_success,
        #             embedding_function=self.common_vectordb_config.common_vectordb_embedding_function,
        #             persist_directory=self.common_vectordb_config.persist_directory + self.db_collection_success,
        #             reset_indices=self.common_vectordb_config.reset_indices,
        #             unique_collection_id=self.common_vectordb_config.unique_collection_id
        #         )
        #     )
        # if self.db_failed_tasks is None or force:
        #     self.db_failed_tasks = UnifiedVectorDB(
        #         UnifiedVectorDBConfig(
        #             collection_name=self.db_collection_failed,
        #             embedding_function=self.common_vectordb_config.common_vectordb_embedding_function,
        #             persist_directory=self.common_vectordb_config.persist_directory + self.db_collection_failed,
        #             reset_indices=self.common_vectordb_config.reset_indices,
        #             unique_collection_id=self.common_vectordb_config.unique_collection_id
        #         )
        #     )

    def get_learnt_tasks(
        self,
        query_text="*",
        k=10,
        metadata_filter=None,
        sort_order=None,
        similarity_search=False
    ):
        # Ensure we're searching in the right collection/database
        if not self.common_vectordb:
            print("DEBUG: No common_vectordb available")
            return set()

        # All “learnt_task” entries live in common_vectordb under data_key="learnt_task"
        if similarity_search:
            # similarity_search_with_score does not accept data_key, so we wrap filter manually
            # The new `get_data` only supports .query(...)—so for similarity_search we do a manual call:
            raw = self.common_vectordb._similarity_search_with_score(query=query_text, k=k, metadata_filter={"data_key": "learnt_task", **(metadata_filter or {})})
            results = [doc for doc, _ in raw]
        else:
            parsed_list, results = self.common_vectordb.get_agent_data(
                agent_name=None,
                data_key="learnt_task",
                metadata_filter=metadata_filter,
                sort_order=sort_order,
                k=k,
                query_text=query_text
            )
        self.task_history.clear_completed_tasks()
        for r in results:
            self.task_history.add_completed_task(r.page_content)
        return {r.page_content for r in results}

    def get_failed_tasks(
        self,
        query_text="*",
        k=10,
        metadata_filter=None,
        sort_order=None,
        similarity_search=False
    ):
        # self.initialize_class_db()
        # if similarity_search:
        #     results = self.db_failed_tasks._similarity_search_with_score(query=query_text, k=k)
        # else:
        #     results = self.db_failed_tasks._query(
        #         query_text=query_text,
        #         k=k,
        #         metadata_filter=metadata_filter,
        #         sort_order=sort_order
        #     )
        # self.task_history.clear_failed_tasks()
        # for result in results:
        #     self.task_history.add_failed_task(result.page_content)
        # return {result.page_content for result in results}

        if similarity_search:
            raw = self.common_vectordb._similarity_search_with_score(
                query=query_text, k=k, metadata_filter={"data_key": "failed_task", **(metadata_filter or {})}
            )
            results = [doc for doc, _ in raw]
        else:
            parsed_list, results = self.common_vectordb.get_agent_data(
                agent_name=None,
                data_key="failed_task",
                metadata_filter=metadata_filter,
                sort_order=sort_order,
                k=k,
                query_text=query_text
            )
        self.task_history.clear_failed_tasks()
        for r in results:
            self.task_history.add_failed_task(r.page_content)
        return {r.page_content for r in results}

    def get_validation_results(
        self,
        query_text="*",
        k=10,
        sort_order=None,
        similarity_search=False
    ):
        # self.initialize_class_db()
        # metadata_filter = {'agent_name': 'ValidationAgent'}
        # if similarity_search:
        #     results = self.common_vectordb._similarity_search_with_score(
        #         query=query_text, k=k, metadata_filter=metadata_filter
        #     )
        # else:
        #     results = self.common_vectordb._query(
        #         query_text=query_text,
        #         k=k,
        #         metadata_filter=metadata_filter,
        #         sort_order=sort_order
        #     )
        # return {result.page_content for result in results}

        metadata_filter = {"agent_name": "ValidationAgent"}
        if similarity_search:
            raw = self.common_vectordb._similarity_search_with_score(
                query=query_text, k=k, metadata_filter=metadata_filter
            )
            results = raw
        else:
            parsed_list, results = self.common_vectordb.get_agent_data(
                agent_name=None,
                data_key=None,
                metadata_filter=metadata_filter,
                sort_order=sort_order,
                k=k,
                query_text=query_text
            )
        return {r.page_content for r in results}
    
    def format_examples(self, examples, criteria, separators):
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
                formatted_example = self.convert_to_markdown(content_data)

            elif output_format.lower() == 'jinja2':
                if not template_str:
                    # Provide a default template if none is specified
                    template_str = self.get_default_jinja2_template(content_data)
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

    def convert_to_markdown(self, content):
        markdown_lines = []
        for key, value in content.items():
            markdown_lines.append(f"**{key}**: {value}")
        return "\n".join(markdown_lines)

    def get_default_jinja2_template(self, content_data):
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

    def retrieve_logs(self, agent_name, function_name, max_entries=20):
        self.configure_vector_store()
        result = self.common_vectordb._query(
            query_text="*",
            metadata_filter={"function_name": function_name, "agent_name": agent_name},
            k=max_entries,
            sort_order="desc"
        )
        return result

    def run_ws_server(self):
        self.ws_server.run_server()

    def stop_ws_server(self):
        if self.ws_server:
            self.ws_server.stop_server()
