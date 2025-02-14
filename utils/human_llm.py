import re, os, uuid, json
import time, pickle, inspect
import socket, requests
import subprocess, logging
import concurrent.futures
from jinja2 import Template
from datetime import datetime
from config import MODELS_CONFIG_LIST
from utils.websocket_server import WebsocketServer, WebSocketServerConfig
from utils.llm_utils import (
    FewShotsParams,
    InferenceCheck,
    _visual_input, save_prompt_with_tag,
    list_prompt_variants, create_Nmajority_chain,
    VectorDBConfig, LLMConfig, UserSession, InferenceTracking, TaskHistory,
    UnifiedVectorDB
)
from typing import List, Dict, Any, Optional

from langchain import LLMChain
from langchain.llms import OpenAI
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.messages.ai import AIMessage
from langchain_community.chat_models import ChatOllama
from langchain_core.messages.human import HumanMessage
from langchain_core.messages.system import SystemMessage
from langchain_core.messages.function import FunctionMessage
from langchain_core.runnables import RunnableSequence, ConfigurableField
from langchain_community.embeddings import HuggingFaceEmbeddings, OpenAIEmbeddings


class HumanLLM:
    _instance = None

    def __new__(cls, *args, **kwargs):
        # Check if the instance already exists
        if cls._instance is None:
            cls._instance = super(HumanLLM, cls).__new__(cls, *args, **kwargs)
        return cls._instance
    
    def __init__(self,
        system_prompt=None,
        CPS_env_type=None,
        agent_name=None,
        model_max_context_size=16000,
        default_llmORchain=None,
        premium_llmORchain=None,
        premium_llm_by_default=False,
        num_parallel_inferences=1,
        llmORchains_list=None,
        synthesize_mode=False,
        inference_checks=None,
        output_schema=None,
        temperature_min=0.7,
        temperature_max=None,
        envs=None,
        fixed_coach=False,
        prompt_critic=None,
        saved_task=None,
        automation=None,
        auto_n_rounds=None,
        recommend_critics=None,
        task_parameters=None
    ):
        if hasattr(self, '_initialized'):
            return
        self._initialized = True

        # Configuration Objects
        self.logger = logging.getLogger(__name__)
        self.llm_config = LLMConfig()
        self.user_session = UserSession()
        self.inference_tracking = InferenceTracking()
        self.task_history = TaskHistory()

        self.vector_db_conf = VectorDBConfig()
        self.vector_db: Optional[UnifiedVectorDB] = None
        self.discord_webhook = "https://discord.com/api/webhooks/1329927709106638930/FaFm6Dozd0xlQvL8hBHn3JHczDazfe2V9hHBWKWj7-igTniTHtsJBCmiFLCh-oKeZ6Mz"

        self.websocket_server_config = WebSocketServerConfig()
        self.websocket_server:Optional[WebsocketServer] = None
        self.use_websocket = False
        
        self.function_list = None
        # Initialize vector databases for tasks
        self.db_collection_success = "successful_tasks"
        self.db_collection_failed = "failed_tasks"
        
        self.db_learnt_tasks = None
        self.db_failed_tasks = None
        
        self.llmORchains_list = self.get_llmORchains_list()
        self.special_criteria:Optional[Any] = None
        self.automation:Optional[bool] = None

        # Instance properties to track time
        self.agent_name = agent_name or self.get_caller_class_name()
        self.task_parameters = task_parameters
        self.selected_outputs = []
        self.menu_start_time = None
        self.start_time = None
        self.current_inference_context = None
        self.user_message_few_shots = None
        
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
        
        # set in 1 line self.print_color is 32 for ActionAgent, 35 for CurriculumAgent, 31 for CriticAgent, 33 for SkillManager, 37 for else
        self.print_color = "32" if self.agent_name in ["ActionAgent", "CodingAgent"] else "35" if self.agent_name in [
            "CurriculumAgent", "TaskIdentificationAgent"] else "31" if self.agent_name in ["CriticAgent",
                                                                                           "ValidationAgent"] else "33" if self.agent_name in [
            "SkillManager", "CapitalizationAgent"] else "37"
        self.previous_templates = []
        self.previous_results = []
        self.comments = []
        self.skip_rounds = 0
        self.log_data = []
        self.num_parallel_inferences = num_parallel_inferences
        self.llm_max_context_size = model_max_context_size
        self.premium_llm_by_default = premium_llm_by_default
        self.synthesize_mode = synthesize_mode
        self.inference_checks = inference_checks if inference_checks else {}
        self.excluded_inference_checks = ["Recommend Critics"]
        self.last_inference_check_results = None
        self.user_message = ""
        self.envs = envs
        self.fixed_coach = fixed_coach
        self.automation = automation
        self.outputs = None
        self.step_id = 0

        self.saved_task = saved_task
        self.auto_n_rounds = auto_n_rounds
        self.recommend_critics = recommend_critics

        # self.initialize()
        self.smart_print = None
        self.smart_input = None

    def initialize(self):
        if self.use_websocket and self.websocket_server is None:
            self.initialize_websocket_server()
            self.websocket_server.add_monitor(self)

    def get_user_id(self):
        return self.user_session.get_user_id()

    def add_agent_data(
            self,
            agent_name,
            data_key,
            data_value,
            function_name=None,
            id_task=False,
            before_after=None,
            user_id=None,
            step_id=None,
            type_tache=None,
            score=None,
            metadata=None
        ):
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
        if user_id is None:
            user_id = self.get_user_id()
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

        self.vector_db.add_texts(texts=[serialized_data], metadatas=[tags])

    def get_agent_data(
            self,
            agent_name=None,
            data_key=None,
            id_task=None,
            function_name=None,
            before_after=None,
            user_id=None,
            step_id=None,
            type_tache=None,
            score=None,
            metadata_filter=None,
            sort_order=None,
            k=5,
            start_index=0,
            end_index=None
        ):
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
        results = self.vector_db.query(
            query_text='*',
            metadata_filter=metadata,
            sort_order=sort_order,
            k=max_k
        )

        # Apply pagination
        paginated_results = (results[start_index:end_index] 
            if end_index is not None 
            else results[start_index:]
        )

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

    def get_tasks(self, page_size:int=200, nb_pages:int=1, id_last_task:Optional[str]=None):
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
            if "task_details" in task.metadata:
                ret[-1]['task_details'] = task.metadata['task_details']

        # Trier la liste par la clé 'date', du plus récent au plus ancien
        ret = sorted(ret, key=lambda x: x['date'], reverse=True)

        return json.dumps(ret)

    def goto_task(
            self,
            id_task: str,
            automatic: str = None,
            special_criteria: dict = None,
            task_details: str = None
        ):
        """
        Retrieve the task from the database and start processing the task.
        Parameters:
            id_task (str): The identifier of the task to retrieve from the database.
            automatic (str): If True, the function will run the loop in automatic mode.
            special_criteria (dict): The special criteria to use for the task.
        """
        # Retrieve the task from the database
        _, saved_task = self.get_agent_data(data_key="saved_task", id_task=id_task)
        task = {
            'before_after': saved_task[0].metadata['before_after'],
            'agent_name': saved_task[0].metadata['agent_name'],
            'type_tache': saved_task[0].metadata['type_tache'],
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

        with open(f'./goto_output/output_{id_task}_{self.get_user_id()}.log', 'w') as f:
            f.write("")

        # Execute the bash command with unbuffered output and capture its output
        process = subprocess.Popen(
            ['bash', '-c', f'python3 -u learn.py --proxy --secret --pickle_name {filename} > ./goto_output/output_{id_task}_{self.get_user_id()}.log 2>&1'],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )
        # Initialize the link variable
        link = None
        with open(f"./goto_output/output_{id_task}_{self.get_user_id()}.log", "r") as logfile:
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
                    message = (f"**XP ID:** {self.vector_db_conf.unique_collection_id}\n**User ID:** {self.get_user_id()}\n**Task Link:** {link}\n**Task Details:** {task_details}")
                    print(f"Discord message: {message}")
                    self.send_to_discord(message)
                    break

        # Return the link if found
        if link is None:
            print("The link was not found in the output file.")

        return link

    def _check_and_init_vector_db(self):
        if self.vector_db_conf.embedding_function:
            self.vector_db_conf.set_common_vectordb_embedding_function()
        if self.vector_db is None:
            self.vector_db = UnifiedVectorDB(self.vector_db_conf)

    def send_to_discord(self, message: str):
        payload = { "content": message}
        headers = { "Content-Type": "application/json"}
        response = requests.post(self.discord_webhook, data=json.dumps(payload), headers=headers)
        if response.status_code == 204:
            print("Message sent to Discord successfully.")
        else:
            print(f"Failed to send message to Discord. Status code: {response.status_code}")

    def check_init_class_db(self, force=False):
        if self.vector_db_conf.common_vectordb_embedding_function is None:
            raise ValueError("embeddingfunction must be set to allow HumanLLM to manage tasks and other memories")
        if self.db_learnt_tasks is None or force:
            self.db_learnt_tasks = UnifiedVectorDB(
                VectorDBConfig(
                    collection_name = self.db_collection_success,
                    embedding_function = self.vector_db.config.common_vectordb_embedding_function,
                    persist_directory = self.vector_db.config.persist_directory + self.db_collection_success,
                    reset_indices = self.vector_db.config.reset_indices
                )
            )
        if self.db_failed_tasks is None or force:
            self.db_failed_tasks = UnifiedVectorDB(
                VectorDBConfig(
                    collection_name = self.db_collection_failed,
                    embedding_function = self.vector_db.config.common_vectordb_embedding_function,
                    persist_directory = self.vector_db.config.persist_directory + self.db_collection_failed,
                    reset_indices = self.vector_db.config.reset_indices
                )
            )

    def get_learnt_tasks(
            self,
            query_text="*",
            k=10,
            metadata_filter=None,
            sort_order=None,
            similarity_search=False
        ):
        self.check_init_class_db()
        if similarity_search:
            results = self.db_learnt_tasks.similarity_search_with_score(query=query_text, k=k)
        else:
            results = self.db_learnt_tasks.query(
                query_text=query_text,
                k=k,
                metadata_filter=metadata_filter,
                sort_order=sort_order
            )
        return {result.page_content for result in results}

    def get_failed_tasks(
            self,
            query_text="*",
            k=10,
            metadata_filter=None,
            sort_order=None,
            similarity_search=False
        ):
        self.check_init_class_db()
        if similarity_search:
            results = self.db_failed_tasks.similarity_search_with_score(query=query_text, k=k)
        else:
            results = self.db_failed_tasks.query(
                query_text=query_text,
                k=k,
                metadata_filter=metadata_filter,
                sort_order=sort_order
            )
        return {result.page_content for result in results}

    def get_validation_results(
            self,
            query_text="*",
            k=10,
            sort_order=None,
            similarity_search=False
        ):
        self.check_init_class_db()
        metadata_filter = {'agent_name': 'ValidationAgent'}
        if similarity_search:
            results = self.vector_db.similarity_search_with_score(
                query=query_text, k=k, metadata_filter=metadata_filter
            )
        else:
            results = self.vector_db.query(
                query_text=query_text,
                k=k,
                metadata_filter=metadata_filter,
                sort_order=sort_order
            )
        return {result.page_content for result in results}

    def get_multiple_few_shots(self, few_shots_params) -> str:
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
                examples = self.vector_db.query(
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
                formatted_example = self.dict_to_markdown(content_data)

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

    def dict_to_markdown(self, content):
        markdown_lines = []
        for key, value in content.items():
            markdown_lines.append(f"**{key}**: {value}")
        return "\n".join(markdown_lines)

    def add_learnt_task(self, serialized_entry, tags):
        self.db_learnt_tasks.add_texts(texts=[serialized_entry], metadatas=[tags])

    def add_failed_task(self, serialized_entry, tags):
        self.db_failed_tasks.add_texts(texts=[serialized_entry], metadatas=[tags])

    def initialize_websocket_server(self):
        if self.use_websocket and self.websocket_server is None:
            self.websocket_server = WebsocketServer(self.websocket_server_config)
            # self.run_websocket_server()

    def run_websocket_server(self):
        self.websocket_server.run_server()

    def stop_websocket_server(self):
        if self.websocket_server:
            self.websocket_server.stop_server()

    def set_common_vectordb_embedding_function(self):
        self.vector_db.config.set_common_vectordb_embedding_function()
        
    def get_few_shots_tag_args(self, prompt):
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
                combined_criteria = self.combine_criteria([data])
                # Get the few shots string
                few_shots_str = self.get_multiple_few_shots(combined_criteria)
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
                                    self.smart_print(
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
                    self.smart_print(f"LLM/Chain '{llm_name}' does not support temperature configuration: {e}",
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
                    self.smart_print(f"LLM/Chain '{llm_name}' does not support structured output", self.agent_name, optional=True)

            # Set the LLM/Chain to the possibly modified or original one
            if is_premium:
                self.premium_llm = selected_llm_or_chain
            else:
                self.default_llm = selected_llm_or_chain

            return True
        else:
            self.smart_print(
                f"LLM/Chain '{llm_name}' not found in llmORchains_list {[key for key in self.llmORchains_list]}",
                self.agent_name, optional=True)
            return False

    def set_default_llmORchain(self, llm_name, temperature=0.1):
        return self.set_llmORchain(llm_name, is_premium=False, temperature=temperature)

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
    
    def get_llmORchains_list(self):
        if MODELS_CONFIG_LIST is not None:
            return {
                "default_llm": ChatOllama(
                    model=MODELS_CONFIG_LIST["basic_gpt"],
                    cache=False,
                    temperature=0.
                ),
                "premium_llm": ChatOllama(
                    model=MODELS_CONFIG_LIST["smart_gpt"],
                        cache=False,
                        temperature=0.
                ),
                "coder_llm": ChatOllama(
                    model=MODELS_CONFIG_LIST["code_gpt"],
                    cache=False,
                    temperature=0.
                ),
                "3_majority_chain": create_Nmajority_chain(
                    map_model_name=MODELS_CONFIG_LIST["basic_gpt"],
                    reduce_model_name=MODELS_CONFIG_LIST["basic_gpt"],
                    num_models=3
                ),
                "10_majority_chain": create_Nmajority_chain(
                    map_model_name=MODELS_CONFIG_LIST["basic_gpt"],
                    reduce_model_name=MODELS_CONFIG_LIST["basic_gpt"], num_models=10
                )
            }
        else:
            # Default, if not precised, we take GPT from OpenAI.
            return {
                "default_llm": ChatOpenAI(
                    model_name="gpt-4o-mini-2024-07-18",
                    cache=False,
                    temperature=0.
                ),
                "premium_llm": ChatOpenAI(
                    model_name="gpt-4o-mini-2024-07-18",
                    cache=False,
                    temperature=0.
                ),
                "coder_llm": ChatOpenAI(
                    model_name="gpt-4o-mini-2024-07-18",
                    cache=False,
                    temperature=0.
                ),
                "3_majority_chain": create_Nmajority_chain(
                    map_model_name="gpt-4o-mini-2024-07-18",
                    reduce_model_name="gpt-4o-mini-2024-07-18",
                    num_models=3
                ),
                "10_majority_chain": create_Nmajority_chain(
                    map_model_name="gpt-4o-mini-2024-07-18",
                    reduce_model_name="gpt-4o-mini-2024-07-18",
                    num_models=10
                )
            }

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

    def load_prompt(self, prompt_name, template_data=None, directory=None):
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

        prompt_content = self.get_few_shots_tag_args(template_content)
        return prompt_content

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

    def exclude_inference_check(self, check_name : list):
        for name in check_name:
            while name[0] == " ":
                name = name[1:]
            if name in self.inference_checks:
                self.excluded_inference_checks.append(name)

    def include_inference_check(self, check_name : list):
        for name in check_name:
            while name[0] == " ":
                name = name[1:]
            if name in self.excluded_inference_checks:
                del self.excluded_inference_checks[self.excluded_inference_checks.index(name)]

    def run_inference_checks(self, output_id, *args, **kwargs):
        results = {}
        for check_name, check in self.inference_checks.items():
            if not check_name in self.excluded_inference_checks:
                result = check.run_check(output_id, *args, **kwargs)
                results[check_name] = result
        # Ensure output_id is within bounds before updating the list
        if 0 <= output_id < len(self.last_inference_check_results):
            self.last_inference_check_results[output_id] = results  # Update the specific index
        return results

    def get_caller_class_name(self):
        # Returns the name of the class that called the current function
        if "self" in inspect.stack()[2][0].f_locals:
            return inspect.stack()[2][0].f_locals["self"].__class__.__name__
        return None

    def _max_tokens_ok(self, content):
        import tiktoken
        encoding = tiktoken.encoding_for_model("gpt-4o-mini")  # TODO: replace by appropriate call to self.xxxxxx
        token_length = len(encoding.encode(content))

        if token_length > self.llm_max_context_size:
            self.smart_print(
                f"\033[31mCANNOT SEND MESSAGE TO LLM:\n{content}\n\nToo many tokens in human message for LLM ({token_length}). Fallback to manual feedback.\033[0m",
                self.agent_name, optional=True)
            return False
        else:
            return True

    def _get_log_entries(self, agent_name, function_name, max_entries=20):
        self._check_and_init_vector_db()
        result = self.vector_db.query(
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

    def _before_inference(
            self,
            messages,
            default_llm_function,
            premium_llm_function,
            function_calling,
            callable_system_message=None,
            use_premium_llm=None,
            model_choice=None,
            task_name=None
    ):
        self.mode = 'before'
        comments = None
        self.user_message = initial_user_message = messages[1].content
        function_name = inspect.stack()[2].function
        use_premium_llm = use_premium_llm if use_premium_llm is not None else self.premium_llm_by_default
        forced_llm_output = False  # TODO: try to set it to None
        self.add_agent_data(
            self.agent_name,
            "saved_task",
            {
                'prompt': messages[0].content + messages[1].content,
                'num_parallel_inferences': self.num_parallel_inferences,
                'task_parameters': self.task_parameters
            },
            before_after='before',
            user_id=self.get_user_id(),
            step_id=self.step_id,
            type_tache="IR_CPS_TechSynthesis",
            id_task=True
        )

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
            menu += ("[R] Activate/Deactivate inferences checks\n")
            menu += (f"[Z] Continue\n")

            self.smart_print(before_menu + menu, self.agent_name, "BEFORE inference action MENU", optional=False)
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
                action = self.smart_input(
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
                    confirm_schema = self.smart_input(
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
                self.add_instruction(initial_user_message)

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
                        self.smart_print(message.content, self.agent_name, "BEFORE inference action MENU", optional=True)

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

            elif action == "R":
                self.activate_deactivate_inference_checks()

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

        self.smart_print(
            f"Time spent in each option and occurrences: {self.before_inference_option_times} - {self.before_inference_option_counts}",
            self.agent_name, optional=True)

        self.mode = None
        messages[1].content = self.user_message
        return messages, comments, forced_llm_output, use_premium_llm, default_llm_function, premium_llm_function, function_calling

    def activate_deactivate_inference_checks(self):
        menu = "Current inference checks:\n"
        for check_name in self.inference_checks :
            menu += f"- {check_name} {'(Deactivated)' if check_name in self.excluded_inference_checks else ''}\n"
        menu += "\n[A] Activate inference check\n"
        menu += "[E] Deactivate inference check\n"
        answer = self.smart_input(menu, self.agent_name, "ACTIVATE/DEACTIVATE INFERENCE CHECKS")
        if answer.lower() == "a":
            check_name = self.smart_input("Enter the names of the inference check to activate (if many to activate, separated by comma): ", self.agent_name,
                                     "ACTIVATE INFERENCE CHECK").title().split(",")
            self.include_inference_check(check_name)
        else:
            check_name = self.smart_input("Enter the names of the inference check to deactivate (if many to deactivate, separated by comma): ", self.agent_name,
                                     "DEACTIVATE INFERENCE CHECK").title().split(",")
            self.exclude_inference_check(check_name)

    def changeNumParallelInferencesAndSynthesize(self):
        try:
            self.num_parallel_inferences = int(
                self.smart_input("Enter new value for num_parallel_inferences: ", self.agent_name,
                            "NUM_PARALLEL_INFERENCES"))
        except:
            self.num_parallel_inferences = 1
        synthesize_mode_input = self.smart_input("Turn synthesis mode on/off (1 for ON, 0 for OFF): ", self.agent_name,
                                            "NUM_PARALLEL_INFERENCES SYNTHESIS MODE CHOICE").strip()  # NEW
        if synthesize_mode_input in ["0", "1"]:  # NEW
            self.synthesize_mode = synthesize_mode_input == "1"  # NEW
        else:  # NEW
            self.smart_print("Invalid input. Synthesize mode remains unchanged.", self.agent_name,
                        "NUM_PARALLEL_INFERENCES SYNTHESIS MODE CHOICE", optional=True)  # NEW

    def changeNumParallelInferences(self):
        try:
            self.num_parallel_inferences = int(
                self.smart_input("Enter new value for num_parallel_inferences: ", self.agent_name,
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
        rounds = int(self.smart_input("Skip for how many rounds? ", self.agent_name))
        self.skip_rounds = rounds

    def log_comments(self, comments):
        comments = self.smart_input("Enter your comment on the prompt: ", self.agent_name)
        return comments

    def reuse_past(self, forced_llm_output, function_name):
        if self.fixed_coach:
            selected_index = 1
            log_entries, list_output = self._get_log_entries(self.agent_name, function_name), ""
        elif self.vector_db.count() > 0:
            log_entries, list_output = self._get_log_entries(self.agent_name, function_name), ""
            for idx, entry in enumerate(log_entries, start=1):
                content = json.loads(entry.page_content)
                text = (content['output_contents'][0]['content'].replace('\n', '\\') if content[
                    'output_contents'] else "") if isinstance(content['output_contents'], list) else \
                    content['output_contents']['content'].replace('\n', '\\')
                date = entry.metadata['time'].split('.')[0]
                list_output += (
                    f"\033[94m{idx}.\033[0m {text[:100]}....{text[-100:]} #{entry.metadata['function_name']} @{date}\n")  # Display a snippet of each entry
            self.smart_print(list_output, self.agent_name, "LOG ENTRIES LIST", optional=True)

            try:
                selected_index = int(
                    self.smart_input(
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

    def add_instruction(self, initial_user_message):
        instructions = self.smart_input(f"ENTER ADDITIONAL INSTRUCTIONS FOR THE AGENT: ", self.agent_name,
                                   message_type="ADDITIONAL_INFO", optional=False)
        self.user_message = initial_user_message + f"\n\nADDITIONAL INSTRUCTIONS: << {instructions} >>"

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
        self.smart_print(output, self.agent_name, "PROMPT OPTIONS", optional=True)
        variant_choice = self.smart_input(
            "Select a number to modify a prompt or create a new variant (or press Enter to continue with the current selection): ",
            self.agent_name)
        if variant_choice.isdigit() and 0 < int(variant_choice) <= len(prompt_variants) + 1:
            if int(variant_choice) == len(prompt_variants) + 1:
                # Process to create a new variant
                comments = self.smart_input("Provide critic or feedback for the current prompt: ", self.agent_name)
                refine_prompt = _visual_input(
                    f"Current system prompt:<<< {self.load_prompt(prompt_name=self.system_prompt, directory='prompts')} >>>\n\nFeedback or critic: {comments}")
                forced_llm_output = default_llm_function.invoke(
                    [SystemMessage(content=self.load_prompt(prompt_name="improve_prompt_from_answer_critic",
                                                            directory='prompts')),
                     HumanMessage(content=refine_prompt)])
                new_template = forced_llm_output.content
            else:
                self.system_prompt = prompt_variants[int(variant_choice) - 1]
        if self.smart_input("Would you like first to get suggestions for a better prompt? (y/n): ",
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
            self.smart_print(
                f"***** PROMPT SUGGESTIONS *****\n\033[33m{forced_llm_output.content}\033[0m\n*************",
                self.agent_name, "PROMPT SUGGESTIONS")
        new_template = _visual_input(
            self.load_prompt(prompt_name=self.system_prompt,
                             directory='prompts') if new_template is None else new_template)
        self.smart_print(f"***** NEW PROMPT TEMPLATE:\n{new_template}\n*************", self.agent_name,
                    "NEW PROMPT TEMPLATE")
        # Confirm that the user wants to modify the template
        confirm = self.smart_input(
            "Do you want to replace current prompt file template with your input? (y/n): ", self.agent_name).upper()
        # Save prompt with tag options
        if confirm == "Y":
            tag_option = self.smart_input(
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
        for i, key in enumerate(llm_keys): self.smart_print(f"{i}. {key}", self.agent_name)
        while True:
            new_llm_index = int(
                self.smart_input(f"Enter the number of the new premium LLM (0-{len(llm_keys) - 1}): ", self.agent_name))
            if 0 <= new_llm_index < len(llm_keys):
                new_llm_name = llm_keys[new_llm_index]
                if self.set_premium_llmORchain(new_llm_name): break
        premium_llm_function = self.premium_llm
        self.smart_print(f"Premium LLM changed to {new_llm_name}", self.agent_name, "Change Premium LLM", optional=True)
        return premium_llm_function

    def changeDefaultLLM(self, default_llm_function):
        llm_keys = list(self.llmORchains_list.keys())
        for i, key in enumerate(llm_keys): self.smart_print(f"{i}. {key}", self.agent_name)
        while True:
            new_llm_index = int(
                self.smart_input(f"Enter the number of the new default LLM (0-{len(llm_keys) - 1}): ", self.agent_name))
            if 0 <= new_llm_index < len(llm_keys):
                new_llm_name = llm_keys[new_llm_index]
                if self.set_default_llmORchain(new_llm_name): break
        default_llm_function = self.default_llm
        self.smart_print(f"Default LLM changed to {new_llm_name}", self.agent_name, "Change Default LLM", optional=True)
        return default_llm_function

    def getScoredResults(self, function_name):
        self._check_and_init_vector_db()
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
        confirm = self.smart_input(
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
            result.extend(self.vector_db.query(
                query_text="*",
                metadata_filter={
                    "function_name": function_name,
                    "agent_name": self.agent_name,
                    "input_modified": True
                },
                k=100
            ))

        if confirm in ["A", "", "C"]:
            result.extend(self.vector_db.query(
                query_text="*",
                metadata_filter={
                    "function_name": function_name,
                    "agent_name": self.agent_name,
                    "output_modified": True
                },
                k=100
            ))

        if confirm in ["A", "", "D"]:
            result.extend(self.vector_db.query(
                query_text="*",
                metadata_filter={
                    "function_name": function_name,
                    "agent_name": self.agent_name,
                    "scored": True
                },
                k=100
            ))

        if confirm in ["A", "", "E"]:
            result.extend(self.vector_db.query(
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

    def getPreviousResults(self, function_name=None, agent_name=None, k=100):
        self._check_and_init_vector_db()
        metadata_filter = {}
        if function_name: metadata_filter["function_name"] = function_name
        if agent_name: metadata_filter["agent_name"] = agent_name
        result = self.vector_db.query(query_text="*", metadata_filter=metadata_filter, k=k)
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

            self.add_agent_data(
                self.agent_name,
                "saved_task",
                {
                    'llm_output': inference_result_msg.content,
                    'user_message': self.current_inference_context['input_contents'][1].content,
                    'num_parallel_inferences': self.num_parallel_inferences,
                    'task_parameters': self.task_parameters
                },
                before_after='after',
                user_id=self.get_user_id(),
                step_id=self.step_id,
                type_tache="IR_CPS_TechSynthesis",
                id_task=True,
                function_name=task_name
            )
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

            self.smart_print(menu, self.agent_name, "AFTER inference action MENU" + (
                f" {output_id}/{outputs_count}" if (output_id and outputs_count and (outputs_count > 1)) else ""),
                        self.agent_name, column_id=output_id - 1, column_max=outputs_count)
            print(f"***{self.agent_name}, AFTER INFERENCE, after self.smart_print menu***")
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
                action = self.smart_input(
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
                self.smart_print("ANSWER MODIFIED, NEW CHECKS REQUIRED BEFORE CONTINUING", self.agent_name, "code_task_and_run_test SystemMessage", column_id=output_id-1, column_max=outputs_count)
                #check_results = self.run_inference_checks(output_id - 1, inference_result_msg.content)
            elif action in [None, "", "E", "Z"]:
                break  # E: Go back BEFORE inference to improve system prompt or add information to user message

            # proceed = self.smart_input("Continue 'y' (or 'n' to go back to menu) ? ", self.agent_name, column_id=output_id, column_max=outputs_count).lower()
            # if proceed in ["y", ""]:
            #     break

        if self.skip_rounds > 0:
            check_results = self.run_inference_checks(output_id - 1, inference_result_msg.content)
            check_display = ""
            # Display inference check results
            for check_name, result in check_results.items():
                check_display += f"{nl}CHECK {check_name} result: " + str(result).replace("\\n", "\n")

            self.smart_print(
                f"\033[{self.print_color}m****{self.agent_name}>{inspect.stack()[2].function} LLM ANSWER content****\n{inference_result_msg.content}\n{check_display}\n*****************\033[0m",
                self.agent_name, "LLM ANSWER content", column_id=output_id)
            self.skip_rounds -= 1
        else:
            self.smart_print(
                f"Time spent in each option and occurrences: {self.after_inference_option_times} - {self.after_inference_option_counts}",
                self.agent_name, optional=True, column_max=outputs_count)

        self.mode = None
        return inference_result_msg, comments, score

    def goBackInference(self, inference_result_msg):
        inference_result_msg = -1  # break is set after action time measurement
        return inference_result_msg

    def evaluateCommentAnswerForLater(self):
        while True:
            score = float(self.smart_input(
                "Give a note for the result between 0.0 (worst) and 1.0 (top), or 0 for bad, 1 for good: ",
                self.agent_name))
            # if score is not between 0 and 1, then set to None and print error
            if score < 0 or score > 1:
                score = None
                print(f"\033[31mInvalid score: {score}\033[0m")
            else:
                break
        comments = self.smart_input("Comment on the result: ", self.agent_name)
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
        comments = self.smart_input("First enter your critic here (then modify answer to get ideal answer): ",
                               self.agent_name)
        ideal_answer = _visual_input(inference_result_msg.content)
        refine_prompt = f"Current system prompt:<<< {self.load_prompt(prompt_name=self.system_prompt, directory='prompts')} >>>\n\nPrompt's answer:<<< {inference_result_msg.content} >>>\n\nPrompt's answer critic:{comments}\n\nPrompt's ideal Answer:<<< {ideal_answer} >>>"
        self.smart_print(f"***** PROMPT FOR IMPROVEMENT *****\n{refine_prompt}", self.agent_name, "PROMPT FOR IMPROVEMENT")
        llm_output = premium_llm_function.invoke([SystemMessage(
            content=self.load_prompt(prompt_name="improve_prompt_from_answer_critic", directory='prompts')),
            HumanMessage(content=refine_prompt)])
        self.smart_print(f"***** RECOMMENDATION OPEN FOR EDITION *****\n", self.agent_name,
                    "RECOMMENDATION OPEN FOR EDITION")
        new_template = _visual_input(llm_output.content)
        self.smart_print(f"***** NEW PROMPT TEMPLATE:\n{new_template}\n*************", self.agent_name,
                    "NEW PROMPT TEMPLATE")  # Confirm that the user wants to modify the template
        confirm = self.smart_input("Do you want to replace current prompt file template with your input? (y/n): ",
                              self.agent_name).upper()  # Save prompt with tag options
        if confirm == "Y":
            tag_option = self.smart_input(
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
                        if key == 'Recommend Critics':
                            critic = value
                            break
                else:
                    break

        if critic:
            if suggestions == "":
                suggestions = critic['suggestions']
            self.add_agent_data(
                self.agent_name,
                "llm_suggestions",
                {
                    'llm_suggestions': critic['suggestions'],
                    'user_suggestions': suggestions,
                    'llm_suggestions_prompt': critic['improvement_prompt']
                }
            )

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
                suggestions = self.smart_input("Provide critic/feedback/request: ", self.agent_name,
                                          column_max=self.num_parallel_inferences)
            temp_prev_sugg, _ = self.get_agent_data(self.agent_name, "llm_suggestions")
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
        self.smart_print(f"***** ANSWER:\n{inference_result_msg.content}\n*************", self.agent_name, "NEW ANSWER", optional=True, column_id=column_id, column_max=self.num_parallel_inferences)
        return inference_result_msg.content

    def updateAnswer(self, answer, column_id=None):
        self.temp_inference_result_content = answer
        if column_id: print("IMPORTANT: column_id not yet implemented - Updating current HumanLLM for agent")
        return answer

    def get_host_id(self):
        
        return socket.gethostname() + "-" + str(uuid.getnode())

    # staticmethod call llm_function (langchain ChatOpenAI) with optional function_call and process function call until result is provided
    def call_llm_function_with_function_call(self, llm_function, messages, function_call=None, function_list=None,
                                             max_calls=5):
        # test if HumanLLM.function_list exists
        if function_list is None:
            if self.function_list is None:
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
                function_list = self.function_list

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
                    result = self.smart_input(f"Uknown function: {function_name} -- please provide result manually:\n",
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
            "host": self.get_host_id(),
            "step_id": self.step_id,
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

        self._check_and_init_vector_db()

        self.vector_db.add_texts(
            texts=[serialized_entry],
            metadatas=[tags]
        )

    def process_llm_output(self, llm_output, counter, llm_outputs, init_skip_rounds, task_name=None):
        """Traite un seul LLM output (séquentiellement ou en parallèle)."""
        print(f"Processing LLM output {counter} out of {len(llm_outputs)}")
        if len(llm_outputs) > 1:
            self.skip_rounds = init_skip_rounds
            self.smart_print(f"ANSWER NUMBER #{counter-1} ", self.agent_name, "POST INFERENCE", append=True, optional=True)

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

    def CallHumanLLM(
            self,
            original_input_messages=None,
            default_llm_function=None,
            premium_llm_function=None,
            callable_system_message=None,
            system_prompt_template=None,
            user_message=None,
            return_message_content_only=True,
            function_calling=False,
            temperature_min=None,
            timeout_seconds=300,
            stream_output=False,
            use_default_llm=True,
            model_choice=None,
            temperature_max=None,
            task_name=None,
            prompt_directory="prompts"
        ):
        #if not self.selected_llm_or_chain: raise ValueError("No LLM or chain selected for use.")
        if temperature_min is None: temperature_min = self.temperature_min
        if temperature_max is None: temperature_max = self.temperature_max
        # Define a helper function to perform the LLM calls for parallel inference.
        def perform_llm_call(input_msg, use_premium, func_calling, temperature=None, stream_output=True, color_id=None):
            if use_premium:
                func = premium_llm_function if not func_calling else self.call_llm_function_with_function_call
            else:
                func = default_llm_function if not func_calling else self.call_llm_function_with_function_call
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
                self.smart_print("", self.agent_name, "Inference streaming output")
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

                        if self.use_websocket:
                            self.smart_print(to_send, self.agent_name, f"Inference streaming output {color_id}",
                                        append=True, column_id=color_id, column_max=self.num_parallel_inferences)
                        else:
                            self.smart_print(start_color + to_send + end_color, self.agent_name,
                                        f"Inference streaming output {color_id}", append=True)

                        # Reset the timer after flushing
                        buffer_start_time = current_time

                # Final flush after streaming ends
                if buffer:
                    if self.use_websocket:
                        self.smart_print(buffer, self.agent_name, f"Inference streaming output {color_id}",
                                    append=True, column_id=color_id, column_max=self.num_parallel_inferences)
                    else:
                        self.smart_print(start_color + buffer + end_color, self.agent_name,
                                    f"Inference streaming output {color_id}", append=True)

                return AIMessage(content=final_output)

            else:
                value = func.invoke(input_msg)
                return AIMessage(content=value.content if hasattr(value, 'content') else str(value))

        self.smart_print(
            f"\033[{self.print_color}m****{self.agent_name}>{inspect.stack()[1].function} calling HumanLLM****\033[0m",
            self.agent_name, "HumanLLM", optional=True)
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
                SystemMessage(content=self.load_prompt(prompt_name=self.system_prompt, directory=prompt_directory)),
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
                self.smart_print(
                    f"\033[{self.print_color}m****{self.agent_name}>{inspect.stack()[2].function} skipping HumanLLM for {self.skip_rounds} rounds****\033[0m",
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
                    self.smart_print(
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
                                if self.use_websocket:
                                    self.smart_print(
                                        llm_response.content,
                                        self.agent_name, "NEW inference result recieved", column_id=idx,
                                        column_max=self.num_parallel_inferences)
                                else:
                                    self.smart_print(
                                        f'\033[0m**** New inference result recieved and added to outputs as #{len(outputs)}\033[0m:\n{llm_response.content}\n\033[9mEND OF #{len(outputs)}****\033[0m',
                                        self.agent_name, "NEW inference result recieved", column_id=idx,
                                        column_max=self.num_parallel_inferences)
                            except concurrent.futures.TimeoutError:
                                self.smart_print('A task ran longer than the allotted timeout and was cancelled.',
                                            self.agent_name, "Inference result TIMEOUT")
                            except Exception as exc:
                                self.smart_print(f'Generated an exception: {exc}', self.agent_name,
                                            "Inference result EXCEPTION")
                        # Wait for all the futures to complete before continuing.
                        concurrent.futures.wait(futures)
                    if len(outputs) == 0:
                        self.smart_print(f'**** No inference result recieved, set output to None', self.agent_name,
                                    "NO inference recieved")
                        llm_outputs = None
                    elif len(outputs) == 1:
                        # self.smart_print(f'**** One inference result recieved, set output to it', self.agent_name, "ONE inference recieved")
                        llm_outputs = outputs
                    else:
                        if self.synthesize_mode and len(
                                outputs) > 1:  #NEW/UPDATED: TODO: allow to exclude code synthesis with (self.synthesize_mode or (self.skip_rounds > 0 and self.agent_name == "Coder"))
                            synthesized_response = self.synthesize_responses([output.content for output in outputs],
                                                                             use_default_llm)  #NEW
                            llm_outputs = [AIMessage(content=synthesized_response.content)]  #NEW
                            self.smart_print(f'**** {len(outputs)} inference results received, THEN SYNTHETISED to 1',
                                        self.agent_name, "MULTIPLE to 1 SYNTHESIS (similar to Mixture of Agents)")  #NEW
                        else:  #UPDATED
                            self.smart_print(
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
                    self.smart_print("**** Multiple LLM ANSWERS > we will process POST INFERENCE for each ****",
                                self.agent_name, "Multiple LLM ANSWERS", append=True, optional=True)
                if False and self.use_websocket:
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

        previous_suggestions, _ = self.get_agent_data(self.agent_name, "llm_suggestions")
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
        self.add_agent_data(self.agent_name, "llm_suggestions", {
            'llm_suggestions': response.content,
            'user_suggestions': "",
            'improvement_prompt': improvement_prompt
        })

        # Now generate annotations using the LLM
        # Retrieve previous annotations
        previous_annotations, _ = self.get_agent_data(self.agent_name, "llm_annotations")
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
        self.add_agent_data(self.agent_name, "llm_annotations", {
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

        self.smart_print(json.dumps(ret), self.agent_name, "CRITIC SUGGESTIONS", column_id=output_id,
                    optional=False)
        return ret