import json
import sys

import optuna as opt
import time
import os
from langchain_openai import ChatOpenAI
from config import MODELS_CONFIG_LIST
from learn import EnvironmentManager, run_planner
from optimisation.optuna_analysis import analysis
from utils.llm_utils import HumanLLM, UnifiedVectorDB

# Set the class variable
HumanLLM.use_websocket = False

def init_prompts_directory(name_xp: str, agent_name : str):
    # Create if not exists the directory to store the prompts
    if not os.path.exists(f"./prompts/IR_CPS_TechSynthesis/{name_xp}"):
        os.makedirs(f"./prompts/IR_CPS_TechSynthesis/{name_xp}")

    # Copy others prompts to this directory but not the subdirectory, depends on the agent_name.
    # Also do not copy the file code_task.txt and identify_best_task.txt from prompts/
    if agent_name == "coder":
        os.system(f"cp ./prompts/IR_CPS_TechSynthesis/identify_best_task.txt ./prompts/IR_CPS_TechSynthesis/{name_xp}/")
    elif agent_name == "coach":
        os.system(f"cp ./prompts/IR_CPS_TechSynthesis/code_task.txt ./prompts/IR_CPS_TechSynthesis/{name_xp}/")
    else:
        os.system(f"cp ./prompts/IR_CPS_TechSynthesis/* ./prompts/IR_CPS_TechSynthesis/{name_xp}/")
        return

    os.system(f"find ./prompts -maxdepth 1 -type f ! -name 'code_task.*' ! -name 'identify_best_task.*' -exec cp {{}} ./prompts/IR_CPS_TechSynthesis/{name_xp}/ \;")


def definition_few_shots(trial, usr_msg: bool = False, no_params_search: bool = False, sources_list : list= ["learnt", "failed", "example"]):
    """
    Define few-shot tags and separators for the trial.

    Args:
        trial (optuna.trial.Trial): The Optuna trial object.
        usr_msg (bool, optional): Flag to determine the format of few_shots_tags. Defaults to False.
        no_params_search (bool, optional): Flag to determine the search logic. Defaults to False.
        sources_list (list, optional): The list of sources for the few-shot tags. Defaults to ["learnt", "failed", "example"].

    Returns:
        str or list: A string of few-shot tags (if usr_msg is False) or a list of few-shot tags (if usr_msg is True).
    """
    # Initialize few_shots_tags based on usr_msg flag
    few_shots_tags = [] if usr_msg else ""

    for i in range(3):
        source = sources_list[i]

        # Determine num based on static_params
        if no_params_search: # Based on experiment result 108353
            num = 2  # Static logic
            output_format = "json"  # Static format
            # Static separators
            global_prefix = f"{source}_tasks: <<"
            global_suffix = ">>\n"
            item_prefix = "\n("
            item_suffix = ")"

        else:
            # Dynamic values using Optuna's trial suggestions (from the second version)
            num = trial.suggest_int(f"num_tag_{source}", 0, 5)
            if num == 0:
                continue  # Skip if num is 0
            
            output_format = trial.suggest_categorical(f"format_tag_{i+1}", ["Json", "Markdown"])
            global_prefix = trial.suggest_categorical("global_prefix_option", ["tasks: <<", "List of tasks: [[", "tasks: {{", "List of tasks: $$"])
            
            # Adjust global_prefix based on the source
            if global_prefix[-2:] in ["[[", "$$", "<<"]:
                global_prefix = f"{global_prefix[:8]}{source} {global_prefix[8:]}"
            else:
                global_prefix = f"{source}{global_prefix}"
            
            # Define global_suffix based on global_prefix
            global_suffix = f"{global_prefix[-2:]}\n".replace("<<", ">>").replace("[[", "]]").replace("{{", "}}")
            item_prefix = trial.suggest_categorical("item_prefix_option", ["\n|", "\n(", "\n-"])
            item_suffix = f"{item_prefix[-1:]}\n".replace("(", ")")

        # Construct the separators dictionary
        tmp_separators = {
            "global_prefix": global_prefix,
            "global_suffix": global_suffix,
            "item_prefix": item_prefix,
            "item_suffix": item_suffix
        }

        # Define the few-shot dictionary
        few_shot_dict = {
            "sources": source,
            "num": num,
            "format": output_format,
            "sort_order": "random",
            "separators": tmp_separators,
            "metadata_filter": {},  # Static for now, can be expanded
        }

        # Convert to JSON string if usr_msg is False (string-based format)
        if not isinstance(few_shots_tags, list):
            few_shot_json = json.dumps(few_shot_dict)
            few_shots_tags += f"few_shots: {few_shot_json}\n"
        else:
            few_shots_tags.append(few_shot_dict)

    return few_shots_tags


def definition_global_parameters(temperature : float = None, presence_penalty : float = None):
    """
    Define global parameters for the LLMs and environments.

    Args:
        temperature (float, optional): The temperature setting for the LLMs.
        presence_penalty (float, optional): The presence penalty setting for the LLMs.

    Returns:
        tuple: A tuple containing the LLM or chains list and the environments list.
    """
    llmORchains_list = {
        "default_llm": ChatOpenAI(model_name=MODELS_CONFIG_LIST["basic_gpt"], **{k: v for k, v in {"temperature": temperature, "presence_penalty": presence_penalty}.items() if v is not None}),
        "premium_llm": ChatOpenAI(model_name=MODELS_CONFIG_LIST["smart_gpt"], **{k: v for k, v in {"temperature": temperature, "presence_penalty": presence_penalty}.items() if v is not None}),
    }

    # Set the documents to test/validate as a list of environments
    documents = [{'id': "cf0d353c-b43b-4a79-88f9-42c2c84cf75e",
                  'title': "Complex QA and language models hybrid architectures, Survey",
                  'context': "This paper reviews the state-of-the-art of language models architectures and strategies for 'complex' question-answering (QA, CQA, CPS) with a focus on hybridization. Large Language Models (LLM) are good at leveraging public data on standard problems but once you want to tackle more specific complex questions or problems (e.g. How does the concept of personal freedom vary between different cultures ? What is the best mix of power generation methods to reduce climate change ?) you may need specific architecture, knowledge, skills, methods, sensitive data protection, explainability, human approval and versatile feedback... Recent projects like ChatGPT and GALACTICA have allowed non-specialists to grasp the great potential as well as the equally strong limitations of LLM in complex QA. In this paper, we start by reviewing required skills and evaluation techniques. We integrate findings from the robust community edited research papers BIG, BLOOM and HELM which open source, benchmark and analyze limits and challenges of LLM in terms of tasks complexity and strict evaluation on accuracy (e.g. fairness, robustness, toxicity, ...) as a baseline. We discuss some challenges associated with complex QA, including domain adaptation, decomposition and efficient multi-step QA, long form and non-factoid QA, safety and multi-sensitivity data protection, multimodal search, hallucinations, explainability and truthfulness, temporal reasoning. We analyze current solutions and promising research trends, using elements such as: hybrid LLM architectural patterns, training and prompting strategies, active human reinforcement learning supervised with AI, neuro-symbolic and structured knowledge grounding, program synthesis, iterated decomposition and others.",
                  'target_file_path': "env/IR_CPS_TechSynthesis/document_embedding_analysis/output/arxiv/Complex QA and language models hybrid architectures Survey.json"},
                 {'id': "42252c6c-12f3-4edf-9045-8acd69bc3356",
                  'title': "Macroeconomic Effects of Inflation Targeting A Survey of the Empirical  Literature",
                  'context': "This paper surveys the empirical literature of inflation targeting. The main findings from our review are the following: there is robust empirical evidence that larger and more developed countries are more likely to adopt the IT regime; the introduction of this regime is conditional on previous disinflation, greater exchange rate flexibility, central bank independence, and higher level of financial development; the empirical evidence has failed to provide convincing evidence that IT itself may serve as an effective tool for stabilizing inflation expectations and for reducing inflation persistence; the empirical research focused on advanced economies has failed to provide convincing evidence on the beneficial effects of IT on inflation performance, while there is some evidence that the gains from the IT regime may have been more prevalent in the emerging market economies; there is not convincing evidence that IT is associated with either higher output growth or lower output variability; the empirical research suggests that IT may have differential effects on exchange-rate volatility in advanced economies versus EMEs; although the empirical evidence on the impact of IT on fiscal policy is quite limited, it supports the idea that IT indeed improves fiscal discipline; the empirical support to the proposition that IT is associated with lower disinflation costs seems to be rather weak. Therefore, the accumulated empirical literature implies that IT does not produce superior macroeconomic benefits in comparison with the alternative monetary strategies or, at most, they are quite modest.",
                  'target_file_path': "env/IR_CPS_TechSynthesis/document_embedding_analysis/output/arxiv/Macroeconomic Effects of Inflation Targeting A Survey of the Empirical  Literature.json"}]

    envs = []
    for doc in documents:
        env = EnvironmentManager(env_type="techsynthesis", title=doc['title'], context=doc['context'],
                                target_file_path=doc['target_file_path'], id=doc['id']).get_environment()
        envs.append(env)

    return llmORchains_list, envs

def launch_run(default_llm_key : str = "default_llm", premium_llm_key : str = "premium_llm", problem_prompts_subdir : str = None, max_coding_attempts : int = 2, max_execution_time : int = 900,
               model_choice=None, automation : str = "coach", params_user_message : str = None, special_criteria : dict = None, name_exp : str = "", temperature_max : float = None, number_inferences : int = 1,
               fixed_coach : bool = False, arrayn_ret : bool = False, continue_each_loop : bool = False, unique_id : str = None):
    """
    Launch the run with the specified parameters.

    Args:
        default_llm_key (str): The key for the default LLM.
        premium_llm_key (str): The key for the premium LLM.
        problem_prompts_subdir (str, optional): The subdirectory for problem prompts.
        max_coding_attempts (int, optional): The maximum number of coding attempts.
        max_execution_time (int, optional): The maximum execution time in seconds.
        model_choice (dict, optional): The model choices for different roles.
        automation (str, optional): The Optuna optimization target.
        params_user_message (str, optional): The criteria for evaluation.
        special_criteria (dict, optional): Special criteria for the run.
        name_exp (str, optional): The name of the experiment.

    Returns:
        Any: The performance result of the run.
    """
    if model_choice is None:
        model_choice = {"coach": "default_llm", "coder": "premium_llm", "critic": "default_llm",
                        "capitalizer": "default_llm"}
    if special_criteria is None:
        llmORchains_list, envs = definition_global_parameters()
    else:
        llmORchains_list, envs = definition_global_parameters(special_criteria.get("temperature", None), special_criteria.get("presence_penalty", None))
        # Delete the presence_penalty from the special_criteria - Why ?
        special_criteria.pop("presence_penalty", None)
    # Set unique collection ID based on name_exp
    UnifiedVectorDB.set_unique_collection_id(f"{name_exp}")

    performance = run_planner(default_llm_key=default_llm_key,
                              premium_llm_key=premium_llm_key,
                              llmORchains_list=llmORchains_list,
                              test_environments=envs,
                              manual_validation_to_capitalize=False,
                              problem_prompts_subdir=problem_prompts_subdir,
                              max_coding_attempts=max_coding_attempts,
                              include_code=False,
                              selected_successful_functions=[],
                              selected_failed_functions=[],
                              agtask_premium_llm_by_default=True,
                              max_execution_time=max_execution_time,
                              agtask_skip_rounds=0,
                              agcoding_skip_rounds=0,
                              agvalidation_skip_rounds=0,
                              agcapitalize_skip_rounds=0,
                              model_choice=model_choice,
                              automation=automation,
                              params_user_message=params_user_message,
                              special_criteria=special_criteria,
                              agcoach_num_parallel_inferences=number_inferences,
                              temperature_max=temperature_max,
                              fixed_coach=fixed_coach,
                              return_array=arrayn_ret,
                              continue_each_loop=continue_each_loop,
                              unique_id=unique_id)
    print("Analysis...")
    analysis(name_exp)
    print("Analysis done.")
    return performance

def launch_study(objective, name_exp : str, n_trials=200, sampler=None):
    """
    Launch the Optuna study with the specified objective and experiment name.

    Args:
        objective (callable): The objective function for the Optuna study.
        name_exp (str): The name of the experiment.
    """
    # get current folder
    current_folder = os.getcwd()

    # Define the directory and file path
    sqlite_dir = os.path.join(current_folder, "Optuna_db")
    sqlite_file = os.path.join(sqlite_dir, f"{name_exp}.db")

    # Create the directory if it does not exist
    if not os.path.exists(sqlite_dir):
        os.makedirs(sqlite_dir)

    # Create a study and optimize the objective function
    if sampler:
        study = opt.create_study(direction="maximize", storage=f"sqlite:///{sqlite_file}", study_name=name_exp, sampler=sampler)
    else:
        study = opt.create_study(direction="maximize", storage=f"sqlite:///{sqlite_file}", study_name=name_exp)
    study.optimize(objective, n_trials=n_trials)

def global_main(objective, agent_name : str = "", params_tested : str = "", name_exp : str = None, n_trials = 200, sampler=None):
    """
    Main function to launch the Optuna study.

    Args:
        objective (callable): The objective function for the Optuna study.
        agent_name (str): The name of the agent.
        params_tested (str): The parameters tested in the study.
        name_exp (str, optional): The name of the experiment.
    """
    # Change directory to the location of this script if not already in the correct directory
    if os.path.basename(os.getcwd()) == "optimisation":
        os.chdir("../")
    if not name_exp:
        # Recuperate name_exp from terminal argument:
        if len(sys.argv) > 1:
            name_exp = sys.argv[1]
        else :
            timestamp_xp = int(time.time())
            name_exp = f"xp_{agent_name}{params_tested}{timestamp_xp}"
    launch_study(lambda trial: objective(trial, name_exp), name_exp, n_trials=n_trials, sampler=sampler)

documentation = """
DOCUMENTATION OF AVAILABLE FUNCTIONS IN THE "bot" OBJECT (SynthesisManager class): {{{
# The following functions are available for use when generating code. Please leverage these pre-existing methods to avoid redundancy, maintain modularity, and ensure code reusability.

class SynthesisManager:
    def __init__(self, document: DocumentStructure, target_file_path: str = None):
        self.document = document
        self.min_cosine_similarity = cosine_similarity([self.document.embedding_model.embed_query(".")], [self.document.embedding_model.embed_query("If you can keep your head when all about you are losing theirs and blaming it on you, If you can trust yourself when all men doubt you, But make allowance for their doubting too ; If you can wait and not be tired by waiting, Or being lied about, don’t deal in lies, Or being hated, don’t give way to hating, And yet don’t look too good, nor talk too wise")])[0][0]
        if target_file_path:
            self.target_file_path = target_file_path

    @staticmethod
    @method_call_counter
    def validate_section_format(section: Dict[str, Any]) -> bool:
        try:
            # This will try to instantiate a Section. If there's a problem with the data, an exception will be raised (e.g., a type error).
            Section(**section)
            return True
        except TypeError as e:
            print(e)
            return False

    # add event using the document object add_event method add_event
    def add_event(self, event: str, data: Dict[str, Any]):
        self.document.add_event(event, data)

    def normalized_cosine_similarity(self, a: List[float], b: List[float], min_cs: float = None) -> float:
        if min_cs is None:
            min_cs = self.min_cosine_similarity
        return (cosine_similarity([a], [b])[0][0] - min_cs) / (1 - min_cs)

    def add_section(self, section: Section):
        if self.validate_section_format(asdict(section)):  # Convert dataclass to dict for validation
            self.document.document_content.sections_list.append(section)
            self.document.update_sections_embeddings([section.section_id])
            self.document.add_event({'action': 'add_section', 'section_id': section.section_id})
        else:
            print('Invalid section format.')
        return self
    
    def create_and_add_section_then_return_id(self, title: str, content: str, section_id: int = None, parent_id: int = None) -> int:
        if not section_id:
            # Generate section_id by using max section_id + 1
            section_id = (max([s.section_id for s in self.document.document_content.sections_list]) + 1) if len(self.document.document_content.sections_list) > 0 else 1

        self.add_section(Section(section_id=section_id, parent_id=parent_id, title=title, content=content))
        return section_id

    def get_sections(self, ids: List[int]) -> List[Section]:
        return [s for s in self.document.document_content.sections_list if s.section_id in ids]
    
    def get_all_sections(self) -> List[Section]:
        return self.document.document_content.sections_list

    def remove_section(self, section_id: int) -> bool:
        section = next((s for s in self.document.document_content.sections_list if s.section_id == section_id), None)
        if section:
            self.document.document_content.sections_list = [s for s in self.document.document_content.sections_list if s.section_id != section_id]
            self.document.update_plan_embedding()
            self.document.add_event({'action': 'remove_section','section_id': section_id})
            return True
        else:
            return False

    def edit_section(self, section_id: int, new_content: str = None, new_title: str = None, new_parent_id: int = None) -> bool:
        #section = next((s for s in self.document.document_content if s['id'] == section_id), None)
        section = next((s for s in self.document.document_content.sections_list if s.section_id == section_id), None)
        if section:
            action_event = {'action': 'edit_section','section_id': section_id}
            update_embeddings = False
            if new_content:
                section.content = new_content
                update_embeddings = True
                action_event['new_content'] = new_content
            if new_title:
                section.title = new_title
                update_embeddings = True
                action_event['new_title'] = new_title
            if new_parent_id:
                section.parent_id = new_parent_id
                action_event['new_parent_id'] = new_parent_id
            self.document.add_event('observation', action_event)
            if update_embeddings:
                self.document.update_sections_embeddings([section_id])
            return True
        else:
            return False
    
    def swap_sections(self, section_id_1: int, section_id_2: int) -> bool:
        section_1 = next((s for s in self.document.document_content.sections_list if s.section_id == section_id_1), None)
        section_2 = next((s for s in self.document.document_content.sections_list if s.section_id == section_id_2), None)
        if section_1 and section_2:
            section_1_index = self.document.document_content.sections_list.index(section_1)
            section_2_index = self.document.document_content.sections_list.index(section_2)
            self.document.document_content.sections_list[section_1_index], self.document.document_content.sections_list[section_2_index] = self.document.document_content.sections_list[section_2_index], self.document.document_content.sections_list[section_1_index]
            self.document.add_event('observation', {'action': 'swap_sections','section_id_1': section_id_1, 'section_id_2': section_id_2})
            return True
        else:
            return False

    # search into resources stored in self.document.resources_vectordb and self.document.resources, return a list of resources
    def semantic_search_resources(self, query_embeddings = None, query_texts = None, n_results = 10, where = None, where_document = None, include = ["metadatas", "documents", "distances"]):
        result = self.document.resources_vectordb.similarity_search_with_score( query_embeddings, k=n_results)

    def get_all_resources(self) -> List[Dict[str, Any]]:
        return self.document.resources

    def add_or_update_results_in_resources(self, results, metadatas_to_add: dict = {}, store_linked_document_content: bool = False):
        for result in results:
            content = {'description': result['description']} if isinstance(result['description'], str) else result['description']
            self.add_or_update_result_in_resources(metadatas=metadatas_to_add, name=result['title'], link=result['link'], content=content, store_linked_document_content=store_linked_document_content)
        return self

    def add_or_update_result_in_resources(self, metadatas: dict, name: str=None, content: dict = None, link: str = None, store_linked_document_content: bool = False, chaining: bool = True):
        # Move metadatas to content if content data were provided into metadatas
        if metadatas.get('title') and not name:
            name = metadatas.get('title')
            metadatas.pop('title')
        if metadatas.get('link') and not link:
            link = metadatas.get('link')
            metadatas.pop('link')
        if metadatas.get('description') and not content:
            content = {'description': metadatas.get('description')}
            metadatas.pop('description')

        # assert name or link are provided to identify the resource
        if not name and not link:
            raise ValueError("Either name or link must be provided")

        # Generate id using max
        id = max([r['id'] for r in self.document.resources]) + 1 if len(self.document.resources) > 0 else 1
        document = {'name': name, 'link': link, 'content': {'description': content} if isinstance(content, str) else content} # Convert content to dict if it's a string
        
        # Check for existing document
        existing_doc = next((doc for doc in self.document.resources if (doc['document']['name'] == name or (link and doc['document']['link'] == link))), None)
        
        if existing_doc:
            # Update the existing document
            updated_fields = []
            for key, value in document.items():
                if value and existing_doc['document'].get(key) != value:
                    existing_doc['document'][key] = value
                    updated_fields.append(key)
            
            # Log the event
            if updated_fields:
                # Update resources_vectordb
                self.document.resources_vectordb.add_texts([str(document)], metadatas=[metadatas], ids=[str(existing_doc['id'])])
                self.document.add_event('observation', {'action': 'modify_resource', 'document_name': name, 'updated_fields': updated_fields})
        else:
            # Add new document
            self.document.resources.append({
                'id': id,
                'metadatas': metadatas,
                'document': document,
            })
            if store_linked_document_content:
                childs_ids_list = self.get_and_store_link_content(link=link, parent_id=id, chaining=False)
                metadatas['childs_ids_list'] = childs_ids_list
            self.document.resources_vectordb.add_texts([str(document)], metadatas=[metadatas], ids=[str(id)])
            self.document.add_event('observation', {'action': 'add_resource', 'document_name': name})

        return self if chaining else (existing_doc if existing_doc else self.document.resources[-1])

    @method_call_counter
    def get_and_store_link_content(self, link: str = None, parent_id = None, chaining: bool = True):
            # Downloads an online document from the given link and stores it in the resources database.
            
            # Args:
            # link (str): The URL of the online document to download.
            # parent_id: The ID of the parent document, if any.
            # chaining (bool): Whether to return the current object or the IDs of the stored documents.
            
            # Returns:
            # If chaining is True, returns the current object. Otherwise, returns the IDs of the stored documents.
            from langchain.document_loaders import WebBaseLoader
            if link is None:
                raise ValueError("Please provide a link to download the document from")
            loader = WebBaseLoader(link)
            data = loader.load()
            if parent_id is not None:
                for doc in data:
                    doc.metadata.extend([{'parent_id': parent_id}])
            from langchain.text_splitter import RecursiveCharacterTextSplitter
            splitter = RecursiveCharacterTextSplitter()
            all_splits = splitter.split_documents(data)
            splits_ids = self.document.resources_vectordb.db.add_documents(all_splits)
            if chaining:
                return self
            else:
                return splits_ids

    def remove_resource(self, resource_id):
        # resource_id can be array or single int
        if isinstance(resource_id, list):
            self.document.resources = [r for r in self.document.resources if r['id'] not in resource_id]
        elif isinstance(resource_id, int):
            self.document.resources = [r for r in self.document.resources if r['id'] != resource_id]
        self.document.add_event('observation', {'action': 'remove_resources','resource_id': str(resource_id)})
        return self
    
    def remove_resources(self, resource_ids: List[int]):
        return self.remove_resource(resource_ids)

    def restore_last_state(self):
        return self.document.restore_state()

    def list_all_previous_document_events(self) -> List[Any]:
        return self.document.events

    # function use to measure performance of the generated document
    def set_targetJSON_comparison(self, file_path: str, target_section_title_embedding_label: str = "section_embedding_2", target_section_content_embedding_label: str = "content_embedding_2", target_plan_embedding_label: str = "plan_embedding_2", normalize_embeddings: bool = True, min_cosine_similarity: float = None):
        self.target_file_path = file_path
        with open(file_path, 'r') as f:
            self.target_data = json.load(f)
        output_check = ''
        for section in self.target_data['plan']:
            output_check += section['section'] + " /"
        print(output_check)
        # Compute the total length for the target data (similar to the test method)
        self.target_total_content_length = sum(len(section['content']) for section in self.target_data['plan'])
        self.target_total_sections_count = len(self.target_data["plan"])

        self.target_plan_titles_embedding = np.mean([section[target_section_title_embedding_label] for section in self.target_data["plan"]], axis=0)
        self.target_plan_contents_embedding = np.mean([section[target_section_content_embedding_label] for section in self.target_data["plan"]], axis=0)
        self.target_plan_embedding = self.target_data[target_plan_embedding_label]

        if normalize_embeddings:
            if min_cosine_similarity is None:
                dumb_embedding = self.document.dumb_embedding
                self.min_plan_titles_cosine_similarity = cosine_similarity([dumb_embedding], [self.target_plan_titles_embedding])[0][0]
                self.min_plan_contents_cosine_similarity = cosine_similarity([dumb_embedding], [self.target_plan_contents_embedding])[0][0]
                self.min_plan_cosine_similarity = cosine_similarity([dumb_embedding], [self.target_plan_embedding])[0][0]
            else:
                self.min_plan_titles_cosine_similarity = self.min_plan_contents_cosine_similarity = self.min_plan_cosine_similarity = min_cosine_similarity
        else:
            self.min_plan_titles_cosine_similarity = self.min_plan_contents_cosine_similarity = self.min_plan_cosine_similarity = 0 

    def get_distance_to_targetJSON(self, target_section_title_embedding_label: str = "section_embedding_2", target_section_content_embedding_label: str = "content_embedding_2", target_plan_embedding_label: str = "plan_embedding_2", get_progress: bool = True):
        # if self does not have target_file_path
        if not hasattr(self, 'target_file_path'):
            raise ValueError("Please set target_file_path using set_targetJSON_comparison method")
        if not hasattr(self, 'target_data'):
            section_embedding_key, content_embedding_key, plan_embedding_key = "content_embedding_2", "section_embedding_2", "plan_embedding_2"
            self.set_targetJSON_comparison(self.target_file_path, target_section_title_embedding_label = section_embedding_key, target_section_content_embedding_label = content_embedding_key, target_plan_embedding_label = plan_embedding_key)
            self.document.update_plan_embedding()
        elif not hasattr(self.document.document_content, 'sections_list_title_embedding'):
            self.document.update_plan_embedding()
        # Similar to what you did in the test
        current_sections_count = len(self.document.document_content.sections_list)
        # Count non empty section's content (not None and len > 1)
        current_plan_non_empty_sections_content_count = sum(1 for section in self.document.document_content.sections_list if section.content and len(section.content) > 1)
        current_plan_non_empty_sections_title_count = sum(1 for section in self.document.document_content.sections_list if section.title and len(section.title) > 1)
        current_content_length = sum(len(section.content) for section in self.document.document_content.sections_list)

        plan_embedding = self.document.document_content.sections_list_embedding
        plan_titles_embedding = self.document.document_content.sections_list_title_embedding
        plan_contents_embedding = self.document.document_content.sections_list_content_embedding
        # compute embedding mean of all "title" in self.target_data["plan"]

        # Compute the similarity and content length percentage
        plan_embedding_similarity = self.normalized_cosine_similarity(plan_embedding, self.target_plan_embedding, self.min_plan_cosine_similarity)
        plan_titles_embedding_similarity = self.normalized_cosine_similarity(plan_titles_embedding, self.target_plan_titles_embedding, self.min_plan_titles_cosine_similarity)
        plan_contents_embedding_similarity = self.normalized_cosine_similarity(plan_contents_embedding, self.target_plan_contents_embedding, self.min_plan_contents_cosine_similarity)

        content_length_ratio_to_target = round(current_content_length / self.target_total_content_length, 2)
        sections_count_ratio_to_target = round(current_sections_count / self.target_total_sections_count, 2)
        sections_content_non_empty_count_ratio_to_target = round(current_plan_non_empty_sections_content_count / self.target_total_sections_count, 2)
        sections_title_non_empty_count_ratio_to_target = round(current_plan_non_empty_sections_title_count / self.target_total_sections_count, 2)

        distance_to_targetJSON = {
            "plan_embedding_similarity": round(plan_embedding_similarity, 6),
            "plan_titles_embedding_similarity": round(plan_titles_embedding_similarity, 6),
            "plan_contents_embedding_similarity": round(plan_contents_embedding_similarity, 6),

            "current_sections_count": current_sections_count,
            "sections_count_ratio_to_target": sections_count_ratio_to_target,

            "title_non_empty_count_ratio_to_target": sections_title_non_empty_count_ratio_to_target,

            "current_content_length": current_content_length,
            "content_length_ratio_to_target": content_length_ratio_to_target,

            "content_non_empty_count_ratio_to_target": sections_content_non_empty_count_ratio_to_target,
        }

        if get_progress:
            # get ratio between same previous values and current values
            def get_ratio(previous_value, current_value):
                return round((previous_value - current_value) / (previous_value + 0.0000001)*100, 2) if previous_value else 0
            if hasattr(self, 'distance_to_targetJSON'):
                distance_to_targetJSON['plan_embedding_similarity_progress'] = get_ratio(plan_embedding_similarity, self.distance_to_targetJSON['plan_embedding_similarity'])
                distance_to_targetJSON['plan_titles_embedding_similarity_progress'] = get_ratio(plan_titles_embedding_similarity, self.distance_to_targetJSON['plan_titles_embedding_similarity'])
                distance_to_targetJSON['plan_contents_embedding_similarity_progress'] = get_ratio(plan_contents_embedding_similarity, self.distance_to_targetJSON['plan_contents_embedding_similarity'])
                distance_to_targetJSON['sections_count_ratio_to_target_progress'] = get_ratio(sections_count_ratio_to_target, self.distance_to_targetJSON['sections_count_ratio_to_target'])
                distance_to_targetJSON['title_non_empty_count_ratio_to_target_progress'] = get_ratio(sections_title_non_empty_count_ratio_to_target, self.distance_to_targetJSON['title_non_empty_count_ratio_to_target'])
                distance_to_targetJSON['content_length_ratio_to_target_progress'] = get_ratio(content_length_ratio_to_target, self.distance_to_targetJSON['content_length_ratio_to_target'])
                distance_to_targetJSON['content_non_empty_count_ratio_to_target_progress'] = get_ratio(sections_content_non_empty_count_ratio_to_target, self.distance_to_targetJSON['content_non_empty_count_ratio_to_target'])

        self.distance_to_targetJSON = distance_to_targetJSON

        return self.distance_to_targetJSON

    # return the list of current sections with title, length of content, validation status, and feedback
    def get_plan_status(self, compact_string_format: bool = False, keys = ["section_id", "title", "content_length"]):
        #keys = ["section_id", "title", "content_length", "validation_status", "feedback_to_process", "feedback_processed"]
        plan_status = []
        for section in self.document.document_content.sections_list:
            status_data_full = [
                section.section_id,
                section.title,
                len(section.content),
                round(section.content_progress_validation_status, 1),
                section.local_feedback_to_process,
                section.local_feedback_processed,
            ]
            status_data = [data for key, data in zip(keys, status_data_full)]

            if compact_string_format:
                plan_status.append("|".join(map(str, status_data)))
            else:
                plan_status.append(dict(zip(keys, status_data)))

        if compact_string_format and plan_status:
            if len(plan_status) == 0:
                return []
            header = "|".join(keys)
            plan_status.insert(0, header)
        
        return plan_status

    def get_resources_status(self, compact_string_format: bool = False):
        resources_status = []
        content_info = {}
        for resource in self.document.resources:
            if resource['document']['content']:
                for key, value in resource['document']['content'].items():
                    content_info[f"len(content['{key}'])"] = len(str(value))
            
            status_data = [
                resource['id'],
                resource['metadatas'].get('search', 'unknown'),
                resource['document']['name'],
                len(resource['document']['link']) if (resource['document']['link'] and isinstance(resource['document']['link'], (list, tuple, np.ndarray))) else 0,
                *content_info.values()
            ]
            if compact_string_format:
                resources_status.append("|".join(map(str, status_data)))
            else:
                keys = ["id", "metadatas", "document_name", "document_link_length"] + list(content_info.keys())
                resources_status.append(dict(zip(keys, status_data)))
        
        if compact_string_format:
            # if content_info is empty, it means that there is no resource in the document
            if len(content_info) == 0:
                return []
            header = "id|metadatas|document_name|document_link_length|" + "|".join(content_info.keys())
            resources_status.insert(0, header)
        
        return resources_status

}}}"""

fixed_coach_prompt = """
1. **Reasoning**:\n   - The current environment has experienced several failures in generating 
coherent tables of contents (TOCs) for research papers, indicating that previous attempts did not 
effectively leverage LLM capabilities for topic extraction and organization.\n   
- The presence of multiple failed tasks highlights the importance of developing a robust function 
that not only extracts topics but also structures them coherently into a TOC that meets quality 
standards, including a minimum of 3 sections and 6 subsections.\n   - Previous attempts have 
shown that runtime errors related to missing parameters have hindered the successful execution 
of functions. Therefore, the next task must ensure clear parameter handling and streamlined operations 
for topic extraction and TOC generation.\n   - Given the current state of the environment with 
empty TOCs and the need for improvement in the extraction and structuring processes, the next 
best task should concentrate on refining the TOC generation method to enhance output quality and 
coherence while ensuring successful execution.\n\n2. **Next Best Task**:\n   
- **Function Name**: `generate_toc_with_fallback`\n   - **Description**: Create a structured table 
of contents (TOC) for a research survey paper using the provided title and abstract, utilizing LLM capabilities 
for topic extraction, and implementing a reliable fallback mechanism for organizing content if extraction fails. 
\n\n3. **Performance Acceptance Criteria**:\n   - The generated TOC must include at least 3 relevant 
sections and 6 subsections derived from the document's context.\n   - If topic extraction fails, the function 
should provide a coherent default TOC structure that aligns with standard research paper outlines.\n   - The TOC 
should maintain a clear hierarchical structure, effectively delineating main sections and subsections.\n   - The 
function should execute within a timeframe of less than 3 seconds.\n   - The output must be coherent, relevant, 
and free from hallucinations or irrelevant content.\n\n4. **Development Plan**:\n   - **Plan Depth**: 3\n   
- **Steps**:\n     - **Step 1**: Use LLM to extract main topics and subtopics from the provided title and abstract.\n       
- **LLM Call**: Perform natural language processing to identify key themes and topics.\n       
- **Error Handling**: Implement a fallback mechanism to handle scenarios where no topics are extracted, 
providing a predefined structure instead.\n     - **Step 2**: Organize the identified topics into a coherent 
hierarchical structure.\n       - **Algorithmic Processing**: Create a structured outline consisting of main 
sections and subsections based on the identified topics or a default template if extraction fails.\n     
- **Step 3**: Format the generated TOC into a structured output suitable for inclusion in the document.\n       
- **Algorithmic Processing**: Ensure the output is formatted as a well-structured list or dictionary for seamless 
integration into the research paper.\n\n5. **Tests**:\n```python\n# 
Test for document #cf0d353c-b43b-4a79-88f9-42c2c84cf75e:\ngenerate_toc_with_fallback(bot, 
title=\"Innovations in Renewable Energy Technologies\", abstract=\"This paper examines the latest 
advancements in renewable energy, focusing on solar, wind, and bioenergy technologies and their potential 
impact on global energy markets.\")\n\n# Test for document #42252c6c-12f3-4edf-9045-8acd69bc3356:\ngenerate_toc_with_fallback(bot, 
title=\"Artificial Intelligence and Machine Learning in Medicine\", abstract=\"This document reviews the integration of 
AI and machine learning in medical diagnostics and treatment, discussing case studies and future trends.\")\n``` 
\n\n### Summary of Component Handling:\n- **LLM Call**: The extraction of keywords and themes (Step 1) will utilize 
LLM capabilities for effective natural language processing, enhancing the understanding of context and semantics.\n
- **Algorithmic Processing**: Steps 2 and 3 (organization and formatting) will be performed through structured algorithmic 
processing to ensure coherent output and effective integration into research papers.\n\n### Error Handling and Fallback 
Mechanisms:\n- Implement error handling to manage exceptions during LLM processing, providing a fallback to a default TOC 
structure based on common research themes if keyword extraction fails.\n- Include logging for performance monitoring to 
ensure reliability and accountability of the task.\n\n### Scalability and Efficiency:\n- The task design should allow 
flexibility in the number of sections generated based on the document's title and abstract, ensuring scalability across 
various document sizes while maintaining efficiency.\n\n### Bias Detection and Mitigation:\n- Implement algorithms to 
analyze the generated TOC for potential biases, ensuring a balanced representation of topics across diverse perspectives.
\n\n### Explainability and Transparency:\n- Provide clear documentation of the TOC generation process, ensuring transparency 
in how LLM outputs and algorithmic processing contribute to the final structured output.
"""