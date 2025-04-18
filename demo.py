"""
Iterative Research Document Generation with Self-Critique and Improvement

Changes in this version:
- plan_document now requests a JSON-based plan from the LLM.
- We store it in state["parsed_plan"].
- We remove merge plan for simplification.
- We add write_all_sections, which loops over the sections from 'parsed_plan' to generate new text for each.
"""

import os
import operator
import json
from typing import List, Annotated, Dict, Any, Optional
from typing_extensions import TypedDict

from pydantic import BaseModel, Field

from langgraph.constants import Send
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, get_buffer_string

from langgraph.graph import START, END, StateGraph, MessagesState
from langgraph.checkpoint.memory import MemorySaver

from config import embedding_function, use_websocket
from utils.human_llm import HumanLLM, HumanLLMConfig
from utils.llm_utils import smart_input, smart_print
from utils.file_utils import load_from_pickle, save_to_pickle
from env.IR_CPS_TechSynthesis.env import VoyagerEnvIR_CPS_TechSynthesis, Section
from env.env import EnvironmentManager
from utils.helpers_demo import (
    remove_think_tags,
    extract_latex_and_bib_from_llm_output,
    extract_json
)
from utils import helpers_demo
import time

try:
    from opto.trace import node
except ImportError:
    # Dummy node: simply returns the data
    def node(data, **kwargs):
        return data

# 1) Initialize config + LLM handles
config = HumanLLMConfig()
llm_list = config.get_llmORchains_list()
config.use_websocket = use_websocket = False
config.smart_input = smart_input
config.smart_print = smart_print

if 'embedding_function' in globals():
    embedding_function = globals()['embedding_function']
if embedding_function is None:
    embedding_function = "text-embedding-ada-002"
config.common_vectordb_config.embedding_function = embedding_function

if not 'reset_db_indices' in locals():
    config.common_vectordb_config.reset_indices = False

config.initialize()

automate_graph = True
planner, section_writer, critic, latex_gen, analyst = None, None, None, None, None

SIMULATION_MODES = {
    "intelligent": {  # Strategy 1 : LLM intelligent unique
        "planner": {"default_llmORchain": "premium_llm"},
        "section_writer": {},
        "critic": {},
        "analyst": {},
    },
    "critique": {  # Strategy 2 : critique + suggestion
        "planner": {"recommend_critics": True},
        "section_writer": {},
        "critic": {},
        "analyst": {},
    },
    "multi_expert": {  # Strategy 3 : multiple generations according to strategy + fuse or choose
        "planner": {"num_parallel_inferences": 3},
        "section_writer": {},
        "critic": {},
        "analyst": {},
    },
}

### ----- Extended State to include new fields -----

class Analyst(BaseModel):
    affiliation: str = Field(
        description="Primary affiliation of the analyst.",
    )
    # name: str = Field(
    #     description="Name of the analyst."
    # )
    role: str = Field(
        description="Role/expertise of the analyst in the context of the topic.",
    )
    description: str = Field(
        description="Description of the analyst focus, concerns, and motives.",
    )
    @property
    def persona(self) -> str:
        return f"Role: {self.role}\nAffiliation: {self.affiliation}\nDescription: {self.description}\n"

class Perspectives(BaseModel):
    analysts: List[Analyst] = Field(
        description="Comprehensive list of analysts with their roles and affiliations.",
    )

class GenerateAnalystsState(TypedDict):
    topic: str         # Research topic
    max_analysts: int  # Number of analysts
    human_analyst_feedback: str  # Human feedback
    analysts: List[Analyst]      # List of analysts

class ResearchGraphState(TypedDict):
    title: str
    topic: str
    max_analysts: int
    human_analyst_feedback: str
    analysts: List[Analyst]
    # Standard fields
    sections: Annotated[list, operator.add]
    plan: str
    introduction: str
    content: str
    conclusion: str
    resource_list: str
    final_report: str
    report_iteration: int
    max_report_iterations: int
    latex_report: str
    cumulative_report: str

    init_max_interview_iterations: int

    # NEW:
    parsed_plan: Dict[str, Any]       # JSON plan with sections, subsections, etc.
    old_parsed_plan: Optional[Dict[str, Any]]  # For merging
    critic: str                       # Keep track of any critique
    source_list: Annotated[list, operator.add]
    scores: Dict[str, Any]


### ----- Plan Document Node (returning JSON) -----
from textwrap import dedent

def plan_document(state: ResearchGraphState):
    """
    Now request the LLM to return a JSON-based plan. For example:
    {
      "Introduction": {
        "description": "...",
        "subsections": {
          "Scope": { "description": "...", "subsections": {} },
          "Purpose": { "description": "...", "subsections": {} }
        }
      },
      "Background": { ... },
      ...
    }
    """
    print("Plan_document")
    filepath = "plan_document_json"
    #prompt_file = f"./prompts/{filepath}.txt"
    default_prompt_plan_document = dedent("""
        You are a planning assistant. Generate a JSON object describing the plan for a research document given the TOPIC.
        Each top-level key is the section's name (e.g., 'Introduction', 'Methods', 'Conclusion'), and its value is a dict with:
        - 'description' (mandatory): short explanation or purpose/scope of that section.
        - 'subsections' (optioanl): an object with subsections (same structure) if necessary.
        For example:
        {   "Introduction": {"description": "Gives an overview of the topic."},
            "Background": {"description": "Reviews existing literature.",
                "subsections": {"Scope": { "description": "Clarifies scope.", "subsections": {...}}
                ...
        }
        Now return ONLY valid JSON, with no extra commentary. The topic is provided in the user message.""")

    # if not os.path.exists(prompt_file):
    #     with open(prompt_file, "w") as f:
    #         f.write(default_plan_prompt)

    # Include previous self-critique feedback if it exists.
    critic_feedback = state.get("critic", "").strip()
    previous_report = state.get("plan", "").strip()
    report_part = f"\nPREVIOUS REPORT'S PLAN: <<< {previous_report} >>>" if previous_report else ""
    feedback_part = f"\nCRITIQUES OF THE PREVIOUS REPORT (BE CARREFUL, SOME CRITIQUES MIGHT NOT APPLY TO THE PLAN): <<< {critic_feedback} >>>" if critic_feedback else ""
    
    user_message = f"TOPIC: <<< {state['topic']} >>>{report_part}{feedback_part}"
    result = planner.invoke(
        original_input_messages=[SystemMessage(content=default_prompt_plan_document), HumanMessage(content=user_message)],
        #system_prompt_template=filepath,
        #user_message=user_message,
        stream_output=False,
        return_message_content_only=True,
        use_default_llm=False
    )[0]

    raw_text = remove_think_tags(result.content if hasattr(result, "content") else result)
    # Attempt to parse
    try:
        plan_json = extract_json(raw_text)
    except:
        # Fallback if the LLM messed up
        plan_json = {}

    # *** Assign stable hierarchical IDs to plan_json to ease reference by other LLM/logics ***
    def assign_ids_to_subsections(subsec_dict: dict, prefix: str):
        """Recursively assign IDs to subsections using the given prefix."""
        for i, (sub_name, sub_data) in enumerate(subsec_dict.items(), start=1):
            sub_id = f"{prefix}.{i}"
            sub_data["id"] = sub_id
            if "subsections" in sub_data:
                assign_ids_to_subsections(sub_data["subsections"], sub_id)

    # Assign IDs for top-level sections
    section_counter = 1
    for section_name, section_data in plan_json.items():
        section_id = str(section_counter)
        section_data["id"] = section_id
        if "subsections" in section_data:
            assign_ids_to_subsections(section_data["subsections"], section_id)
        section_counter += 1

    return {
        "plan": json.dumps(plan_json, separators=(',', ':')),          # the raw plan text
        "parsed_plan": plan_json,
        "initial_plan": state.get("initial_plan", raw_text),
    }

### ----- Write All Sections Node -----

def write_all_sections(state: ResearchGraphState):
    print("write_all_sections")
    plan = state.get("parsed_plan", {})
    if not plan:
        print("No plan found, skipping.")
        return {}

    new_sections_content = []
    collected_sources = []
    seen_titles = set()  # <-- Track lowercased section titles

    def traverse_and_write(section_name, section_obj, depth=0):
        # Normalize section title
        normalized_title = section_name.strip().lower()
        if normalized_title in seen_titles:
            return  # Skip duplicate section by title
        seen_titles.add(normalized_title)

        desc = section_obj.get("description", "")
        subsecs = section_obj.get("subsections", {})
        sources = section_obj.get("sources", None)
        if sources: collected_sources.extend(sources)

        if any(keyword in section_name.lower() for keyword in ["sources", "references", "bibliography"]):
            return
        
        report_part = f"\nPREVIOUS VERSION OF THE REPORT: <<< {state.get('final_report', '')} >>>" if state.get('final_report') else ""
        feedback_part = f"\nCRITIQUES OF THE PREVIOUS VERSION OF THE REPORT: <<< {state.get('critic', '')} >>>" if state.get("critic") else ""

        user_prompt = (
            f"SECTION TO WRITE: <<{section_name}>>\n"
            f"EXPECTED PLAN OF THE REPORT: <<{state.get('plan', {})}>>\n"
            f"TITLE OF THE REPORT: <<{state['title']}>>\n"
            f"TOPIC: <<{state['topic']}>>\n"
            f"DESCRIPTION: <<{desc}>>\n"
            f"SOURCES: <<{sources}>>\n" if sources else ""
            f"INTERVIEWS: <<{state.get('sections', [])}>>\n"
            f"{report_part}{feedback_part}"
        )
        system_prompt = "You are a researcher, write the content for this SECTION (cite SOURCES when appropriate)"
        response = section_writer.invoke(
            original_input_messages=[
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ],
            stream_output=False,
            return_message_content_only=True,
            use_default_llm=False
        )[0]
        text = remove_think_tags(response.content if hasattr(response, "content") else response)
        heading_prefix = "#" * (depth + 2)
        final_text = f"{heading_prefix} {section_name}\n\n{text}"
        new_sections_content.append(final_text)

        for sub_name, sub_obj in subsecs.items():
            traverse_and_write(sub_name, sub_obj, depth + 1)

    for top_section, top_obj in plan.items():
        # Display status out of the plan length
        print(f"({len(new_sections_content)}/{len(plan)}) >> Processing section: {top_section} ")
        if top_section == "UNUSED_SECTIONS":
            continue
        traverse_and_write(top_section, top_obj, depth=0)

    temp_source_list = state.get("source_list", [])
    temp_source_string_list = [f"{source.get('title', '')}, {source.get('authors', '')}, {source.get('link', '')}" for source in temp_source_list]
    collected_sources.extend(temp_source_string_list)
    formatted_sources = [f"\\item \\textbf{{{src}}}" for src in collected_sources]
    resource_update = "\\begin{enumerate}\n" + "\n".join(formatted_sources) + "\n\\end{enumerate}"
    
    cleaned_sections = [
        section for section in new_sections_content 
        if not any(keyword in section.lower() for keyword in ["# sources", "# references", "# bibliography"])
    ]
    
    return {
        #"sections": list(dict.fromkeys(state.get("sections", []) + cleaned_sections)),
        "sections": cleaned_sections,
        "resource_list": resource_update
    }

def finalize_report(state: ResearchGraphState):
    print("Finalize_report")
    main_sections = state.get("sections", [])
    resources = state.get("resource_list", "")

    # Remove any existing References/Sources sections from content
    final_content = "\n\n".join([
        section for section in main_sections 
        if not any(keyword in section.lower() 
                  for keyword in ["# sources", "# references", "# bibliography"])
    ])

    final_report = ""
    if final_content.strip():
        final_report += final_content + "\n\n"

    if resources.strip():
        final_report += "\\section*{References}\n\n" + resources.strip()

    state["final_report"] = final_report
    return {"final_report": final_report}
### ----- Self Critique Node (unchanged, but using plan) -----

def self_critique(state: ResearchGraphState):
    """
    Prompt LLM to produce updated plan + overall critique in JSON with keys: plan, critic.
    We'll parse that, set state['parsed_plan'] accordingly. We'll store the old in old_parsed_plan if needed.
    """
    print("Self_critique")
    default_prompt_self_critique = dedent("""
        You are an expert critic in science. You will be given a research report, and optionally some feedback from analysts and automatic evaluation.
        Your task is to critic the provided research report with a list of critic with recommended corrective action on most important corrections to do on plan/content/bibliography/subjects covered/style.""")

    current_report = state.get("final_report", "")
    # Retrieve the aggregated plan feedback from all analysts (if any)
    plan_feedback = state.get("plan_feedback_by_analysts", {})
    feedback_summary = ""
    if plan_feedback:
        feedback_summary = "Feedback from analysts: <<< " + "; ".join(f"{k}: {v}" for k, v in plan_feedback.items()) + " >>>\n"
    if state.get("scores"):
        feedback_summary += f"Automatic evaluation: <<< {state.get('scores')} >>>\n"
    user_message = (
        f"REPORT: <<< {current_report} >>>\n\n"
        f"{feedback_summary}"
    )

    critique_response = critic.invoke(
        original_input_messages=[ SystemMessage(content=default_prompt_self_critique), HumanMessage(content=user_message)],
        stream_output=False,
        return_message_content_only=True,
        use_default_llm=False
    )[0]
    data = remove_think_tags(critique_response.content if hasattr(critique_response, "content") else critique_response)

    return {
        "report_iteration": state.get("report_iteration", 0) + 1,
        "critic": str(data)
    }

### ----- The usual reset_for_iteration, generate_latex, etc. remain the same -----

def reset_for_iteration(state: ResearchGraphState):
    print("Reset_for_iteration")
    state["sections"] = []
    return {
        "sections": [],
        "introduction": "",
        "content": "",
        "conclusion": "",
        "resource_list": ""
    }

def generate_latex(state: ResearchGraphState):
    print("Generate_latex")
    report = state.get("final_report", "")
    default_prompt_latex = "Convert all the provided report into a LaTeX document"

    user_message = f"REPORT: <<< {report} >>>"
    result = latex_gen.invoke(
        original_input_messages=[SystemMessage(content=default_prompt_latex), HumanMessage(content=user_message)],
        stream_output=False,
        return_message_content_only=True,
        use_default_llm=False
    )[0]
    latex_report_text = remove_think_tags(result.content if hasattr(result, "content") else result)
    return {"latex_report": latex_report_text}

def create_analysts(state: GenerateAnalystsState):
    """Create Analysts Agent: Generate a list of analysts in JSON."""
    print("Create_analysts")
    critic_feedback = state.get("critic", "").strip()
    default_prompt_create_analysts = dedent("""
        You are tasked with creating a set of AI analyst personas. Your goal is to generate a list of analysts in JSON format and nothing else.
        In the user message, you will receive the following values:
          - 'TOPIC': the research topic,
          - 'FEEDBACK': any editorial feedback,
          - 'MAX_ANALYSTS': the maximum number of analysts to generate.

        Review the topic and feedback, identify the top themes, and assign one analyst per theme.
        Each analyst must have the following fields: role (string), affiliation (string), and description (string).""")

    topic = state['topic']
    max_analysts = state['max_analysts']
    human_analyst_feedback = state.get('human_analyst_feedback', '')
    user_message = (
        f"TOPIC: <<< {topic} >>>\n"
        f"MAX_ANALYSTS: <<< {max_analysts} >>>\n"
        f"FEEDBACK: <<< {human_analyst_feedback} >>>\n"
        f"TASK: Generate the set of analysts in JSON format."
    )
    try:
        analysts_response = analyst.invoke(
            original_input_messages=[SystemMessage(content=default_prompt_create_analysts), HumanMessage(content=user_message)],
            stream_output=False,
            return_message_content_only=True,
            use_default_llm=False
        )[0]
        # Extract the JSON and convert it to Analyst objects
        analysts_response = helpers_demo.extract_json(helpers_demo.remove_think_tags(analysts_response))
        if isinstance(analysts_response, Perspectives):
            generated_analysts = analysts_response.analysts
        elif isinstance(analysts_response, dict):
            generated_analysts = analysts_response.get('analysts', [])
        elif hasattr(analysts_response, 'analysts'):
            generated_analysts = analysts_response.analysts
        elif isinstance(analysts_response, list):
            generated_analysts = analysts_response
        else:
            generated_analysts = [
                Analyst(
                    role=f"Research Specialist {i+1}",
                    affiliation="Research Institute",
                    description=f"Analyzing aspects of {topic}"
                ) for i in range(max_analysts)
            ]
        # Conversion compatible Pydantic v1/v2
        generated_analysts = [
            Analyst(**(a.model_dump() if hasattr(a, "model_dump") else a.dict() if hasattr(a, "dict") else a))
            for a in generated_analysts
        ]
        return {"analysts": generated_analysts}
    except Exception as e:
        print(f"Error generating analysts: {e}")
        default_analysts = [
            Analyst(
                role=f"Research Specialist {i+1}",
                affiliation="Research Institute",
                description=f"Analyzing aspects of {topic}"
            ) for i in range(max_analysts)
        ]
        return {"analysts": default_analysts}

# def human_feedback(state: GenerateAnalystsState):
#     """No-op node that can be interrupted for human feedback."""
#     print("Human_feedback")
#     pass

class InterviewState(MessagesState):
    max_interview_iterations: int                              # Number of conversation turns
    context: Annotated[list, operator.add]          # Source docs
    analyst: Analyst                                # Analyst persona
    interview: str                                  # Interview transcript
    interview_iteration: int                                  # Iteration number
    sections: list                                  # Collected sections for the report
    expert_response: str                            # Expert response
    expert_resources: list                          # Expert resources
    sections: Annotated[list, operator.add]         # Collected sections for the report
    source_list: Annotated[list, operator.add]      # Collected sources for the report
    initial_plan: str                               # Initial plan

# Add this helper at the top of interview.py (or in a shared helpers file)
def translate_query(query: str, target_language: str = "english") -> str:
    """
    Translate the given query into the target language using the available LLM.
    """
    print(f"Translating query to {target_language}")
    translation_instruction = SystemMessage(content=(
        f"Translate the following query into {target_language}: \"{query}\""
    ))
    # Use the existing LLM call (llm_custom is already defined in helpers_demo)
    translated_response = helpers_demo.llm_custom.invoke([translation_instruction]).content
    translated_query = translated_response.content if hasattr(translated_response, "content") else translated_response
    return translated_query

def generate_question(state: InterviewState):
    print("Generate_question")
    if "question_instructions" not in state:
        state["question_instructions"] = (
            "You are an analyst interviewing an expert. Your goal is to ask interesting and specific questions to gain deep insights. \n"
            "Current document state: {document_state}\n"
            "Previous Critique: {critic}\n"
            "Your focus is: {goals}\n"
            "Introduce yourself with your persona-appropriate name, then ask your question. "
            "Continue asking until you feel you have enough insight."
        )
    document_state = state.get("final_report", "")
    critic_text = state.get("critic", "")
    analyst = state["analyst"]
    system_message = state["question_instructions"].format(
        document_state=document_state,
        critic=critic_text,
        goals=analyst.persona
    )
    messages = state["messages"]
    question_answer = helpers_demo.llm_custom.invoke([SystemMessage(content=system_message)] + messages)
    smart_print(message=question_answer.content if isinstance(question_answer, AIMessage) else question_answer,
                agent_name="Generate Question", message_type="NEW inference result recieved", column_id=0, column_max=1)
    return {"messages": [question_answer]}

def search_web(state: InterviewState):
    print("Search_web")
    if "search_instructions" not in state:
        state["search_instructions"] = SystemMessage(content=(
            "You will be given a conversation between an analyst and an expert. "
            "Your goal is to generate a well-structured query for retrieval. "
            "Analyze the conversation—especially the final question—and convert it into a search query."
        ))
    # Generate the base query (using the existing LLM query function)
    base_query_obj = helpers_demo.search_llm_query(state["search_instructions"], state["messages"])
    base_query = base_query_obj.search_query

    # Define the list of languages in which to translate the query
    languages = ["english", "french", "chinese"]
    all_results = []
    for lang in languages:
         translated_query = translate_query(base_query, lang)
         # Wrap the translated string in a SearchQuery object as expected by the search function
         search_query_lang = helpers_demo.SearchQuery(search_query=translated_query)
         results = helpers_demo.search_web_query_get(search_query_lang)
         all_results.append(results)
    aggregated_results = "\n\n---\n\n".join(all_results)
    smart_print(message=aggregated_results, agent_name="Search Web",
                message_type="NEW inference result recieved", column_id=0, column_max=1)
    return {"context": [aggregated_results]}

def search_arxiv(state: InterviewState):
    print("Search_arxiv")
    if "search_instructions" not in state:
        state["search_instructions"] = SystemMessage(content=(
            "You are given a research topic. Your task is to generate a concise and effective search query "
            "optimized for the arXiv API. Make sure to include relevant keywords to retrieve the most pertinent scientific articles."
        ))
    base_query_obj = helpers_demo.search_llm_query(state["search_instructions"], state["messages"])
    base_query = base_query_obj.search_query

    languages = ["english"]
    all_results = []
    for lang in languages:
         translated_query = translate_query(base_query, lang) if lang != "english" else base_query
         search_query_lang = helpers_demo.SearchQuery(search_query=translated_query)
         results = helpers_demo.search_arxiv_query_get(search_query_lang)
         all_results.append(results)
    aggregated_results = "\n\n---\n\n".join(all_results)
    smart_print(message=aggregated_results, agent_name="Search Arxiv",
                message_type="NEW inference result recieved", column_id=0, column_max=1)
    return {"context": [aggregated_results]}

def search_semantic_scholar(state: InterviewState):
    print("Search_semantic_scholar")
    if "search_instructions" not in state:
        state["search_instructions"] = SystemMessage(content=(
            "You are given a research topic. Your task is to generate a concise and effective search query "
            "optimized for the Semantic Scholar API. Ensure that the query includes relevant keywords to retrieve the most pertinent scientific articles."
        ))
    base_query_obj = helpers_demo.search_llm_query(state["search_instructions"], state["messages"])
    base_query = base_query_obj.search_query

    languages = ["english", "french", "chinese"]
    all_results = []
    for lang in languages:
         translated_query = translate_query(base_query, lang) if lang != "english" else base_query
         search_query_lang = helpers_demo.SearchQuery(search_query=translated_query)
         results = helpers_demo.search_semantic_scholar_query_get(search_query_lang)
         all_results.append(results)
    aggregated_results = "\n\n---\n\n".join(all_results)
    smart_print(message=aggregated_results, agent_name="Search Semantic Scholar",
                message_type="NEW inference result recieved", column_id=0, column_max=1)
    return {"context": [aggregated_results]}

def search_wikipedia(state: InterviewState):
    print("Search_wikipedia")
    if "search_instructions" not in state:
        state["search_instructions"] = SystemMessage(content=(
            "You will be given a conversation between an analyst and an expert. "
            "Your goal is to generate a well-structured query for retrieval. "
            "Analyze the conversation and convert the final question into a search query."
        ))
    base_query_obj = helpers_demo.search_llm_query(state["search_instructions"], state["messages"])
    base_query = base_query_obj.search_query

    languages = ["english", "french", "chinese"]
    all_results = []
    for lang in languages:
         translated_query = translate_query(base_query, lang) if lang != "english" else base_query
         search_query_lang = helpers_demo.SearchQuery(search_query=translated_query)
         results = helpers_demo.search_wikipedia_query_get(search_query_lang)
         all_results.append(results)
    aggregated_results = "\n\n---\n\n".join(all_results)
    smart_print(message=aggregated_results, agent_name="Search Wikipedia",
                message_type="NEW inference result recieved", column_id=0, column_max=1)
    return {"context": [aggregated_results]}

def search_docs_rag(state: InterviewState):
    print("Search_docs_rag")
    if "search_instructions" not in state:
        state["search_instructions"] = SystemMessage(content=(
            "You are an analyst tasked with generating a search query for the RAG retrieval model. "
            "Analyze the conversation between the analyst and the expert and convert the final question into a search query."
        ))
    base_query_obj = helpers_demo.search_llm_query(state["search_instructions"], state["messages"])
    base_query = base_query_obj.search_query

    languages = ["english"]
    all_results = []
    for lang in languages:
         translated_query = translate_query(base_query, lang) if lang != "english" else base_query
         search_query_lang = helpers_demo.SearchQuery(search_query=translated_query)
         results = helpers_demo.search_docs_rag_get(search_query_lang)
         all_results.append(results)
    aggregated_results = "\n\n---\n\n".join(all_results)
    smart_print(message=aggregated_results, agent_name="Search Docs RAG",
                message_type="NEW inference result recieved", column_id=0, column_max=1)
    return {"context": [aggregated_results]}

def generate_answer(state: InterviewState):
    print(f">>> Generate_answer for analyst '{state['analyst']}' - interview_iteration {state['interview_iteration']}")
    if "answer_instructions" not in state:
        state["answer_instructions"] = (
            "You are an expert being interviewed.\n\n"
            "Your focus is:{goals}\n\n"
            "Answer the following question using the provided context:{context}\n\n"
            "Guidelines:\n"
            "1. Use only the information in the context.\n"
            "2. Do not introduce external information.\n"
            "3. You have to cite sources from the context using latex style citations ~\\cite{{source_key}}.\n"
            "4. The source should have at least the title, authors, and link to the document, if link is available.\n"
            "{format_instructions}"
        )
    
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import JsonOutputParser
    from langchain_core.pydantic_v1 import BaseModel, Field
    
    # Define the output schema
    class ExpertResponse(BaseModel):
        response: str = Field(description="The expert's answer to the question")
        sources: list = Field(description="The sources used to answer the question")
    
    # Create the output parser
    parser = JsonOutputParser(pydantic_object=ExpertResponse)
    
    answer_instructions = state["answer_instructions"]
    analyst = state["analyst"]
    messages = state["messages"]
    context = state["context"]
    
    # Create a prompt template with format instructions
    prompt_template = ChatPromptTemplate.from_template(answer_instructions)
    formatted_prompt = prompt_template.format(
        goals=analyst.persona, 
        context=context,
        format_instructions=parser.get_format_instructions()
    )
    
    answer_resp = helpers_demo.llm_custom.invoke([SystemMessage(content=formatted_prompt)] + messages)
    
    try:
        # Parse the response directly with the JsonOutputParser
        answer_json = parser.parse(answer_resp.content)
        smart_print(message=answer_json, agent_name="Generate Answer", message_type="NEW inference result recieved", column_id=0, column_max=1)
        return {"expert_response": [answer_json.response], "expert_resources": answer_json.sources, 'interview_iteration': state.get("interview_iteration", 0) + 1}
    except Exception as e:
        # Fallback to the original extract_json method if parsing fails
        answer_json = extract_json(answer_resp.content)
        smart_print(message=answer_json, agent_name="Generate Answer", message_type="NEW inference result recieved", column_id=0, column_max=1)
        if isinstance(answer_json, dict):
            return {"expert_response": [answer_json.get("response", "")], "expert_resources": answer_json.get("sources", []), 'interview_iteration': state.get("interview_iteration", 0) + 1}
        else:
            return {"expert_response": [str(answer_resp.content)], "expert_resources": [], 'interview_iteration': state.get("interview_iteration", 0) + 1}

def save_interview(state: InterviewState):
    print("Save_interview")
    messages = state["messages"]
    interview = get_buffer_string(messages)

    print(f"INTERVIEW STATE: analyst={state['analyst']}, interview_iteration={state['interview_iteration']}, max_interview_iterations={state['max_interview_iterations']}") 

    if state.get("interview_iteration", 0) >= state.get("max_interview_iterations", 2):
        analyst = state["analyst"]
        analyst_name = getattr(analyst, "name", "Analyst")
        plan_text = state.get("initial_plan", "")
        # Formulate the final question to ask the expert about the plan
        system_msg = SystemMessage(content=(
            "You are an expert who has just been interviewed. "
            "The interviewer now asks for your feedback on the document plan."
        ))
        user_msg = HumanMessage(content=(
            f"DOCUMENT PLAN: <<< {plan_text} >>>\n\n"
            "Now that our interview is concluded, do you have any feedback or suggestions to improve this plan?"
        ))
        feedback_resp = helpers_demo.llm_custom.invoke([system_msg, user_msg])
        feedback_text = feedback_resp.content if hasattr(feedback_resp, "content") else str(feedback_resp)
        feedback_text = remove_think_tags(feedback_text)
        # Store the expert's feedback, keyed by analyst
        plan_feedbacks = state.get("plan_feedback_by_analysts", {})
        plan_feedbacks[analyst_name] = feedback_text
        state["plan_feedback_by_analysts"] = plan_feedbacks

    return {"interview": interview}

def route_messages(state: InterviewState, name: str = "expert"):
    messages = state["messages"]
    max_interview_iterations = state.get('max_interview_iterations', 2)
    num_responses = max(len(messages)/2, state.get('interview_iteration', 0))
    print(f"Route_messages - num_responses={num_responses}, max_interview_iterations={max_interview_iterations}")
    if num_responses >= max_interview_iterations:
        return 'save_interview'
    last_question = messages[-2]
    if "Thank you so much for your help" in last_question.content:
        return 'save_interview'
    return "ask_question"

def write_section(state: InterviewState):
    print("Write_section")
    if "section_writer_instructions" not in state:
        state["section_writer_instructions"] = (
            "You are an expert technical writer. Your task is to create a detailed and comprehensive report from the provided source documents. "
            "You have access to the current report state and the original plan. Follow these steps:\n"
            "1. Thoroughly analyze the provided documents (each begins with a <Document tag) and the current report content.\n"
            "2. Structure your report using Markdown with appropriate headers (e.g., ## for titles), and include detailed explanations, technical insights, and critical analysis from the expert.\n"
            "3. Your section should include detailed sources list with full links or document paths to support your report.\n"
            "Aim to append new information and expand on existing content with substantial detail, ensuring clarity and depth without repeating what has already been provided."
        )
    section_writer_instructions = state["section_writer_instructions"]
    analyst = state["analyst"]
    context = state["context"]
    current_report = state.get("final_report", "")
    initial_plan = state.get("initial_plan", "")
    system_message = section_writer_instructions.format(focus=analyst.description)
    human_msg = (
        f"Use this source to write your section: <<< {context} >>>\n"
        f"Use this expert response to write your section: <<< {state['expert_response'] if 'expert_response' in state else ''} >>> \n"
        f"Current Report: <<< {current_report} >>>\n"
        f"Original Plan: <<< {initial_plan} >>>\n"
        "TASK: Append new information in this report without repeating what is already present."
    )
    propositions = helpers_demo.llm_custom.invoke([
        SystemMessage(content=system_message),
        HumanMessage(content=human_msg)
    ])
    section_final = ""
    if isinstance(propositions, list):
        for section in propositions:
            section_final += section.content if isinstance(section, AIMessage) else section
    else:
        section_final = propositions.content if isinstance(propositions, AIMessage) else propositions
    smart_print(message=section_final, agent_name="Write Section", message_type="NEW inference result recieved", column_id=0, column_max=1)
    # Append the new section to the existing list.
    return {"sections": state.get("sections", []) + [section_final], "source_list": state.get("source_list", []) + state.get("expert_resources", [])}

def should_continue(state: GenerateAnalystsState):
    print("Should_continue")
    human_analyst_feedback = state.get('human_analyst_feedback', None)
    if human_analyst_feedback:
        return "create_analysts"
    return END

def should_iterate(state: ResearchGraphState):
    if state.get("report_iteration", 0) < state.get("max_report_iterations", 2):
        print(f"-----------### END OF ITERATION {state.get('report_iteration', 0)} > NEXT: SELF-CRITIQUE ###-----------")
        #return "reset_for_iteration"
        return "self_critique"
    else:
        print(f"-----------### END OF ITERATION {state.get('report_iteration', 0)} ###-----------")
        return END

def choose_search_strategy(state: ResearchGraphState):
    global SEARCH_STRATEGY
    # test if SEARCH_STRATEGY exist in global scope, otherwise set it to None
    if 'SEARCH_STRATEGY' not in globals() or SEARCH_STRATEGY is None:
        if automate_graph:
            SEARCH_STRATEGY = "arxiv" if ("SEARCH_STRATEGY" not in globals() or SEARCH_STRATEGY is None) else SEARCH_STRATEGY
        else:
            SEARCH_STRATEGY = smart_input(
                "What searching strategy do you want to use? (Web, Wikipedia, ArXiv, Semantic, All): ",
                column_id=0, column_max=1, optional=False
            ).lower()
            SEARCH_STRATEGY = (
                "arxiv" if SEARCH_STRATEGY not in ["web","wikipedia","arxiv","semantic","all"] else SEARCH_STRATEGY
            )
    print(f"Search strategy chosen: {SEARCH_STRATEGY}")
    return state

def route_to_search_nodes(state: InterviewState):
    global SEARCH_STRATEGY
    targets = []
    if SEARCH_STRATEGY in ["search_docs_rag", "default"]:
        targets += ["search_docs_rag"]
    if SEARCH_STRATEGY == "web":
        targets += ["search_web"]
    elif SEARCH_STRATEGY == "wikipedia":
        targets += ["search_wikipedia"]
    elif SEARCH_STRATEGY == "arxiv":
        targets += ["search_arxiv"]
    elif SEARCH_STRATEGY == "semantic":
        targets += ["search_semantic_scholar"]
    elif SEARCH_STRATEGY == "all":
        targets += ["search_web","search_wikipedia","search_arxiv","search_semantic_scholar"]
    else:
        targets = ["arxiv"]
    return targets

def initiate_all_interviews(state: ResearchGraphState):
    print("Initiate_all_interviews")
    topic = state["topic"]
    return [
        Send("conduct_interview", {
            "analyst": a,
            "messages": [HumanMessage(content=f"So you said you were writing an article on {topic}?")],
            "interview_iteration": 0,
            "max_interview_iterations": state.get("init_max_interview_iterations", 2),
            "initial_plan": state.get("plan", ""),
        }) for a in state["analysts"]
    ]

### --------- Combine everything into a main function ----------

from langgraph.graph import StateGraph

def score_generated_report_with_existing_env(final_report_in_latex: str, topic: str, title: str = None):
    """
    Reuses the scoring logic from env.py by creating a VoyagerEnvIR_CPS_TechSynthesis
    environment, loading the same references, inserting 'final_report' into it,
    then calling get_score().
    """
    feedback = {}

    # TODO: make this automatic searching in env/IR_CPS_TechSynthesis/document_embedding_analysis/output all files and info from JSON file
    known_docs = {
        "Complex QA and language models hybrid architectures, Survey": {
            "id": "cf0d353c-b43b-4a79-88f9-42c2c84cf75e",
            "title": "Complex QA and language models hybrid architectures, Survey",
            "context": "Initial placeholder context for Complex QA.", 
            "target_file_path": "env/IR_CPS_TechSynthesis/document_embedding_analysis/output/arxiv/Complex QA and language models hybrid architectures Survey.json"
        },
        "Macroeconomic Effects of Inflation Targeting A Survey of the Empirical Literature": {
            "id": "42252c6c-12f3-4edf-9045-8acd69bc3356",
            "title": "Macroeconomic Effects of Inflation Targeting A Survey of the Empirical Literature",
            "context": "Initial placeholder context for Macroeconomic Effects.", 
            "target_file_path": "env/IR_CPS_TechSynthesis/document_embedding_analysis/output/arxiv/Macroeconomic Effects of Inflation Targeting A Survey of the Empirical Literature.json"
        }
    }
    # If a matching environment is available for the topic, use it for scoring
    if topic in known_docs or title in known_docs:
        doc = known_docs[topic if topic in known_docs else title]
        try:
            env = EnvironmentManager(
                env_type="techsynthesis",
                title=doc['title'],
                context=doc['context'],
                target_file_path=doc['target_file_path'],
                id=doc['id'],
                llm=llm_list["default_llm"],
                embedding_model_name=embedding_function  # use the configured embedding function
            ).get_environment()
            env.reset()
            env.synthesis_manager.GetFromLatex(final_report_in_latex)
            feedback["scores"] = env.get_score()
        except Exception as e:
            print(f"Environment scoring failed: {e}")

    # No reference environment – perform a qualitative LLM evaluation
    filepath = "score_document_eval"
    default_prompt_score_document_eval = dedent("""
        You are a research report evaluator focusing on structure, citations, and completeness.
        In the user message, you will receive a 'REPORT' (the final report in LaTeX format).
        Provide a brief qualitative feedback on the report, commenting on its structural organization, the use of citations, and the overall completeness of the content relative to the topic.""")

    user_message = f"REPORT: <<< {final_report_in_latex} >>>"
    critique_resp = critic.invoke(
        original_input_messages=[ SystemMessage(content=default_prompt_score_document_eval), HumanMessage(content=user_message)],
        stream_output=False,
        return_message_content_only=True
    )[0]
    critique_text = remove_think_tags(critique_resp.content if hasattr(critique_resp, "content") else critique_resp)
    feedback["qualitative feedback"] = critique_text

    return feedback


def score_document(state: ResearchGraphState):
    """
    Node to evaluate the final_report using the existing scoring logic from env.py.
    This node runs after finalize_report and before self_critique.
    """

    final_latex_report = state.get("latex_report", "")
    if not final_latex_report:
        print("No final_report found in state. Skipping scoring...")
        return {}

    scores = score_generated_report_with_existing_env(final_latex_report, state["topic"], state["title"])

    # Optionally store them in the state so the next node can see them
    return { "scores": scores}

def multi_agent_research_generation_persist_at_the_end(
    title, topic, max_analysts: int=3, max_report_iterations: int=2, auto_n_rounds_planner: int=0, auto_n_rounds_section_writer: int=0, auto_n_rounds_analyst: int=0, auto_n_rounds_critic: int=0, auto_n_rounds_latex: int=0, automation: str="full_auto", trace_optimization: bool=False, max_interview_iterations: int=2, human_simulation_mode: str="intelligent"
):
    print("Multi_agent_research_generation_persist_at_the_end")

    if automation == "full_auto":
        auto_n_rounds_analyst = auto_n_rounds_section_writer = auto_n_rounds_analyst = auto_n_rounds_critic = auto_n_rounds_latex = 999

    global planner, section_writer, critic, latex_gen, analyst
    sim_config = SIMULATION_MODES[human_simulation_mode]

    planner = HumanLLM(agent_name="Planner", llmORchains_list=llm_list,
                    automation=automation, auto_n_rounds=auto_n_rounds_planner,
                    **sim_config["planner"])
    section_writer = HumanLLM(agent_name="Section Writer", llmORchains_list=llm_list,
                            automation=automation, auto_n_rounds=auto_n_rounds_section_writer,
                            **sim_config["section_writer"])
    critic = HumanLLM(agent_name="Self Critic", llmORchains_list=llm_list,
                    automation=automation, auto_n_rounds=auto_n_rounds_critic,
                    **sim_config["critic"])
    analyst = HumanLLM(agent_name="Create Analysts", llmORchains_list=llm_list,
                    automation=automation, auto_n_rounds=auto_n_rounds_analyst,
                    **sim_config["analyst"])
    latex_gen = HumanLLM(agent_name="Generate Latex", llmORchains_list=llm_list, automation=automation, auto_n_rounds=auto_n_rounds_latex)

    initial_state: ResearchGraphState = {
        "title": title,
        "topic": topic,
        "max_analysts": max_analysts,
        "human_analyst_feedback": "",
        "analysts": [],
        "sections": [],
        "plan": "",
        "introduction": "",
        "content": "",
        "conclusion": "",
        "resource_list": "",
        "final_report": "",
        "report_iteration": 0,
        "max_report_iterations": max_report_iterations,
        "latex_report": "",
        "cumulative_report": "",
        "parsed_plan": {},
        "old_parsed_plan": {},
        "plan_feedback_by_analysts": {},
        "critic": "",
        "source_list": [],
        "scores": {},
        "init_max_interview_iterations": max_interview_iterations,
    }

    builder = StateGraph(ResearchGraphState)

    # --- Add our nodes ---
    builder.add_node("plan_document", plan_document)
    builder.add_node("create_analysts", create_analysts)
    builder.add_node("choose_search_strategy", choose_search_strategy)
    builder.add_node("write_all_sections", write_all_sections)
    builder.add_node("finalize_report", finalize_report)
    builder.add_node("self_critique", self_critique)
    builder.add_node("reset_for_iteration", reset_for_iteration)
    builder.add_node("generate_latex", generate_latex)
    builder.add_node("score_document", score_document)
    # Build the subgraph for multi-agent interviews (unchanged)
    interview_graph_def = StateGraph(InterviewState)
    interview_graph_def.add_node("ask_question", generate_question)
    #interview_graph_def.add_node("route_to_search_nodes", route_to_search_nodes)
    interview_graph_def.add_node("search_docs_rag", search_docs_rag)
    interview_graph_def.add_node("search_web", search_web)
    interview_graph_def.add_node("search_wikipedia", search_wikipedia)
    interview_graph_def.add_node("search_arxiv", search_arxiv)
    interview_graph_def.add_node("search_semantic_scholar", search_semantic_scholar)
    interview_graph_def.add_node("answer_question", generate_answer)
    interview_graph_def.add_node("write_section", write_section)
    interview_graph_def.add_node("save_interview", save_interview)
    interview_graph_def.add_conditional_edges("ask_question", route_to_search_nodes,
        ["search_docs_rag", "search_web", "search_wikipedia", "search_arxiv", "search_semantic_scholar"]
    )
    interview_graph_def.add_edge("search_docs_rag", "answer_question")
    interview_graph_def.add_edge("search_web", "answer_question")
    interview_graph_def.add_edge("search_wikipedia", "answer_question")
    interview_graph_def.add_edge("search_arxiv", "answer_question")
    interview_graph_def.add_edge("search_semantic_scholar", "answer_question")
    interview_graph_def.add_conditional_edges("answer_question", route_messages,['ask_question','save_interview'])
    interview_graph_def.add_edge("save_interview", "write_section")
    #interview_graph_def.set_start("ask_question") # to be replace by add_edge START
    interview_graph_def.add_edge(START, "ask_question")
    #interview_graph_def.set_end("write_section")
    interview_graph_def.add_edge("write_section", END)
    compiled_interview_graph = interview_graph_def.compile()

    builder.add_node("conduct_interview", compiled_interview_graph)

    builder.add_edge(START, "plan_document")

    # --- Edges in main graph ---
    builder.add_edge("plan_document", "create_analysts")
    builder.add_edge("create_analysts", "choose_search_strategy")

    builder.add_conditional_edges("choose_search_strategy", initiate_all_interviews, ["conduct_interview"])

    builder.add_edge("conduct_interview", "write_all_sections")

    # Here we do finalize_report, then self_critique
    builder.add_edge("write_all_sections", "finalize_report")
    builder.add_edge("finalize_report", "generate_latex")
    builder.add_edge("generate_latex", "score_document")
    #builder.add_edge("score_document", "self_critique")

    # The self_critique can lead to a new report_iteration or final
    #builder.add_conditional_edges("self_critique", should_iterate,
    builder.add_conditional_edges("score_document", should_iterate,
        ["self_critique", END]
#        ["reset_for_iteration", END]
    )
    builder.add_edge("self_critique", "reset_for_iteration")
    builder.add_edge("reset_for_iteration", "plan_document")

    graph = builder.compile()

    start_time = time.time()
    # Storing graph's diagram
    try:
        if not os.path.exists("images"): os.makedirs("images")
        with open("images/graph_png.png", "wb") as f:
            f.write(graph.get_graph(xray=1).draw_mermaid_png())
    except Exception as e:
        print(f"Error saving graph image: {e}")
    end_time = time.time()
    print(f"Graph diagram saved in {end_time - start_time:.2f} seconds.")

    result = graph.invoke(initial_state, { "configurable": { "thread_id": "1" }, "recursion_limit": 100 })

    latex_report = result.get("latex_report","")
    tex_report, bib_report = extract_latex_and_bib_from_llm_output(latex_report)

    report_folder = "./report_outputs"
    if not os.path.exists(report_folder):
        os.makedirs(report_folder)

    if tex_report:
        with open(f"{report_folder}/{topic}.tex", "w") as f:
            f.write(tex_report)
    if bib_report:
        with open(f"{report_folder}/{topic}.bib", "w") as f:
            f.write(bib_report)

    if trace_optimization:
        return {'final_report': node(latex_report)}
    return latex_report


# -------------- Example usage --------------
title = "Complex QA and language models hybrid architectures, Survey" # "State of the art on 'Leading-Edge Noise'"
topic = """This paper reviews the state-of-the-art of language models architectures and strategies for "complex" question-answering (QA, CQA, CPS) with a focus on hybridization. Large Language Models (LLM) are good at leveraging public data on standard problems but once you want to tackle more specific complex questions or problems (e.g. How does the concept of personal freedom vary between different cultures ? What is the best mix of power generation methods to reduce climate change ?) you may need specific architecture, knowledge, skills, methods, sensitive data protection, explainability, human approval and versatile feedback... Recent projects like ChatGPT and GALACTICA have allowed non-specialists to grasp the great potential as well as the equally strong limitations of LLM in complex QA. In this paper, we start by reviewing required skills and evaluation techniques. We integrate findings from the robust community edited research papers BIG, BLOOM and HELM which open source, benchmark and analyze limits and challenges of LLM in terms of tasks complexity and strict evaluation on accuracy (e.g. fairness, robustness, toxicity, ...) as a baseline. We discuss some challenges associated with complex QA, including domain adaptation, decomposition and efficient multi-step QA, long form and non-factoid QA, safety and multi-sensitivity data protection, multimodal search, hallucinations, explainability and truthfulness, temporal reasoning. We analyze current solutions and promising research trends, using elements such as: hybrid LLM architectural patterns, training and prompting strategies, active human reinforcement learning supervised with AI, neuro-symbolic and structured knowledge grounding, program synthesis, iterated decomposition and others."""
SEARCH_STRATEGY = "arxiv"
if __name__ == "__main__":
    title1 = "Complex QA and language models hybrid architectures, Survey"
    topic1 = "Complex QA and language models hybrid architectures, Survey"

    for mode in ["critique", "intelligent", "multi_expert"]:
        print(f"\n===== MODE: {mode} =====\n")
        final_report = multi_agent_research_generation_persist_at_the_end(
            title=title1,
            topic=topic1,
            max_analysts=3,
            max_report_iterations=2,
            automation="full_auto",
            human_simulation_mode=mode,
        )