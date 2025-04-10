"""
Iterative Research Document Generation with Self-Critique and Improvement

Changes in this version:
- plan_document now requests a JSON-based plan from the LLM.
- We store it in state["parsed_plan"].
- We add a merge_plan node to merge old plan data with the new plan each iteration.
- We add write_all_sections, which loops over the sections from 'parsed_plan' to generate new text for each.
"""

import os
import operator
import json
from typing import List, Annotated, Dict, Any, Optional
from typing_extensions import TypedDict

from pydantic import BaseModel, Field

from langgraph.constants import Send
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

from langgraph.graph import START, END, StateGraph, MessagesState
from langgraph.checkpoint.memory import MemorySaver

from config import embedding_function, use_websocket
from utils.human_llm import HumanLLM, HumanLLMConfig
from utils.llm_utils import smart_input, smart_print
from env.IR_CPS_TechSynthesis.env import VoyagerEnvIR_CPS_TechSynthesis, Section
from env.env import EnvironmentManager
from utils.helpers_demo import (
    remove_think_tags,
    extract_latex_and_bib_from_llm_output,
    extract_json
)
from utils import helpers_demo

from opto.trace import node

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
    iteration: int
    max_iterations: int
    latex_report: str
    cumulative_report: str

    # NEW:
    parsed_plan: Dict[str, Any]       # JSON plan with sections, subsections, etc.
    old_parsed_plan: Optional[Dict[str, Any]]  # For merging
    critic: str                       # Keep track of any critique
    source_list: Annotated[list, operator.add]
    scores: Dict[str, Any]


### ----- Plan Document Node (returning JSON) -----

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
    prompt_file = f"./prompts/{filepath}.txt"

    if not os.path.exists(prompt_file):
        with open(prompt_file, "w") as f:
            f.write(
                "You are a planning assistant. Generate a JSON object describing the plan for a research document. "
                "Each top-level key is the section's name (e.g., 'Introduction', 'Methods', 'Conclusion'), "
                "and its value is a dict with at least:\n"
                "  - 'description': short explanation or purpose of that section.\n"
                "  - 'subsections': an object with subsections (same structure) if any.\n"
                "  - 'sources': an array with any source references you consider relevant.\n\n"
                "For example:\n"
                "{\n"
                "  \"Introduction\": {\n"
                "    \"description\": \"Gives an overview of the topic.\",\n"
                "    \"subsections\": {\n"
                "       \"Scope\": {\n"
                "          \"description\": \"Clarifies scope.\",\n"
                "          \"subsections\": {},\n"
                "          \"sources\": []\n"
                "       }\n"
                "    },\n"
                "    \"sources\": []\n"
                "  },\n"
                "  \"Background\": {\n"
                "    \"description\": \"Reviews existing literature.\",\n"
                "    \"subsections\": {},\n"
                "    \"sources\": []\n"
                "  }\n"
                "}\n\n"
                "Now return ONLY valid JSON, with no extra commentary. The topic is provided in the user message."
            )

    user_message = f"TOPIC: <<< {state['topic']} >>>"
    result = planner.invoke(
        system_prompt_template=filepath,
        user_message=user_message,
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

    # Save for reference
    if not state.get("initial_plan"):
        state["initial_plan"] = raw_text

    return {
        "plan": raw_text,          # the raw plan text
        "parsed_plan": plan_json   # the structured JSON
    }

### ----- Merge Plan Node -----

def merge_plan(state: ResearchGraphState):
    """
    If the LLM produces a new plan each time, we can unify it with the old plan
    so that no previously generated sections or info is lost.

    For simplicity, this example merges top-level sections by name.
    If a new plan omits a previously existing top-level section, we
    keep the old one under "unused_sections" or similar.

    In your real usage, adapt this to deeper merging logic.
    """
    print("Merge_plan")

    old_plan = state.get("old_parsed_plan") or {}
    new_plan = state.get("parsed_plan") or {}

    final_plan = {}
    unused_sections = {}

    # Add everything from new_plan
    for section_name, data in new_plan.items():
        final_plan[section_name] = data

    # If old_plan has sections not in new_plan, put them in "unused_sections"
    for old_section_name, old_data in old_plan.items():
        if old_section_name not in final_plan:
            unused_sections[old_section_name] = old_data

    # If we want to store unused sections to remind us in the next iteration:
    final_plan["UNUSED_SECTIONS"] = unused_sections

    state["old_parsed_plan"] = final_plan
    state["parsed_plan"] = final_plan

    return {
        "parsed_plan": final_plan
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

    def traverse_and_write(section_name, section_obj, depth=0):
        desc = section_obj.get("description", "")
        subsecs = section_obj.get("subsections", {})
        sources = section_obj.get("sources", [])

        # Add sources to collected list
        collected_sources.extend(sources)

        # Skip any section related to sources/references
        if any(keyword in section_name.lower() 
               for keyword in ["sources", "references", "bibliography"]):
            return
        # Generate section content
        user_prompt = (
            f"SECTION: <<{section_name}>>\n"
            f"DESCRIPTION: <<{desc}>>\n"
            f"SOURCES: >>{sources}>>\n\n"
            f"INTERVIEWS: <<{state.get('sections', [])}>>\n"
            f"PLAN OF DOCUMENT/SECTIONS: <<{state.get('plan', {})}>>\n"
        )
        system_prompt = ("Write only content for this SECTION...")
        response = section_writer.invoke(
            original_input_messages=[SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)],
            stream_output=False,
            return_message_content_only=True,
            use_default_llm=False
        )[0]
        text = remove_think_tags(response.content if hasattr(response, "content") else response)

        heading_prefix = "#" * (depth+2)
        final_text = f"{heading_prefix} {section_name}\n\n{text}"
        new_sections_content.append(final_text)

        # Recurse on subsections
        for sub_name, sub_obj in subsecs.items():
            traverse_and_write(sub_name, sub_obj, depth+1)

    # Traverse all top-level items
    for top_section, top_obj in plan.items():
        if top_section == "UNUSED_SECTIONS":
            continue
        traverse_and_write(top_section, top_obj, depth=0)

    # Format collected sources as LaTeX items
    temp_source_list = state.get("source_list", [])
    temp_source_string_list = []
    for source in temp_source_list:
        temp_source_string_list.append(f"{source.get('title', '')}, {source.get('authors', '')}, {source.get('link', '')}")

    collected_sources.extend(temp_source_string_list)
    formatted_sources = [f"\\item \\textbf{{{src}}}" for src in collected_sources]
    resource_update = "\\begin{enumerate}\n" + "\n".join(formatted_sources) + "\n\\end{enumerate}"
    
    # Clean existing sections from any source-related content
    cleaned_sections = [
        section for section in new_sections_content 
        if not any(keyword in section.lower() 
                  for keyword in ["# sources", "# references", "# bibliography"])
    ]

    return {
        "sections": state.get("sections", []) + cleaned_sections,
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
    filepath = "self_critique"
    prompt_file = f"./prompts/{filepath}.txt"
    if not os.path.exists(prompt_file):
        with open(prompt_file, "w") as f:
            f.write(
                "You are an expert critic. Your task is to review a research report and provide constructive "
                "criticism with suggestions for improvement. In the user message, you will receive 'REPORT'. "
                "Return a JSON object with keys 'plan' (the updated plan in JSON) and 'critic' (the critique)."
            )

    current_report = state.get("final_report", "")
    user_message = (
        f"REPORT: <<< {current_report} >>>\n\n"
        "TASK: Critique the report and update the plan.\n"
        "Return JSON with structure:\n"
        "{\n"
        "  \"plan\": { /* JSON structure for sections, like the original plan format */ },\n"
        "  \"critic\": \"Your textual critique\"\n"
        "}"
    )

    critique_response = critic.invoke(
        system_prompt_template=filepath,
        user_message=user_message,
        stream_output=False,
        return_message_content_only=True,
        use_default_llm=False
    )[0]
    text = remove_think_tags(critique_response.content if hasattr(critique_response, "content") else critique_response)
    # Attempt to parse
    try:
        data = json.loads(text)
    except:
        data = {}

    new_plan = data.get("plan", {})
    # Optionally store old plan
    state["old_parsed_plan"] = state["parsed_plan"]
    state["parsed_plan"] = new_plan

    # Update iteration
    iteration = state.get("iteration", 0) + 1
    state["iteration"] = iteration
    state["critic"] = data.get("critic", "")

    # Also keep a 'plan' text if you want
    # (not strictly needed if everything is in parsed_plan)
    plan_str = json.dumps(new_plan, indent=2)
    state["plan"] = plan_str

    return {
        "plan": plan_str,
        "iteration": iteration
    }

### ----- The usual reset_for_iteration, generate_latex, etc. remain the same -----

def reset_for_iteration(state: ResearchGraphState):
    print("Reset_for_iteration")
    state["sections"] = []
    state["introduction"] = ""
    state["content"] = ""
    state["conclusion"] = ""
    state["resource_list"] = ""
    state["plan"] = ""
    # Keep old_parsed_plan if you want or wipe it out. We’ll keep it here:
    # state["old_parsed_plan"] = state.get("parsed_plan", {})
    # state["parsed_plan"] = {}
    return state

def generate_latex(state: ResearchGraphState):
    print("Generate_latex")
    report = state.get("final_report", "")
    filepath = "generate_latex"
    prompt_file = f"./prompts/{filepath}.txt"
    if not os.path.exists(prompt_file):
        with open(prompt_file, "w") as f:
            f.write(
                "You are an expert tool for converting text into a LaTeX document. [same as your old code]..."
            )

    user_message = f"REPORT: <<< {report} >>>"
    result = latex_gen.invoke(
        system_prompt_template=filepath,
        user_message=user_message,
        stream_output=False,
        return_message_content_only=True,
        use_default_llm=False
    )[0]
    latex_report_text = remove_think_tags(result.content if hasattr(result, "content") else result)
    return {"latex_report": latex_report_text}

def create_analysts(state: GenerateAnalystsState):
    """Create Analysts Agent: Generate a list of analysts in JSON."""
    print("Create_analysts")
    filepath = "create_analysts"
    prompt_file = f"./prompts/{filepath}.txt"
    if not os.path.exists(prompt_file):
        with open(prompt_file, "w") as f:
            f.write(
                "You are tasked with creating a set of AI analyst personas. Your goal is to generate a list of analysts in JSON format and nothing else. "
                "In the user message, you will receive the following values:\n"
                "  - 'TOPIC': the research topic,\n"
                "  - 'FEEDBACK': any editorial feedback,\n"
                "  - 'MAX_ANALYSTS': the maximum number of analysts to generate.\n\n"
                "Review the topic and feedback, identify the top themes, and assign one analyst per theme. "
                "Each analyst must have the following fields: role (string), affiliation (string), and description (string)."
            )
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
            system_prompt_template=filepath,
            user_message=user_message,
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
    max_num_turns: int                              # Number of conversation turns
    context: Annotated[list, operator.add]          # Source docs
    analyst: Analyst                                # Analyst persona
    interview: str                                  # Interview transcript
    sections: list                                  # Collected sections for the report
    expert_response: str                            # Expert response
    expert_resources: list                          # Expert resources
    sections: Annotated[list, operator.add]         # Collected sections for the report
    source_list: Annotated[list, operator.add]      # Collected sources for the report

# Add this helper at the top of interview.py (or in a shared helpers file)
def translate_query(query: str, target_language: str = "en") -> str:
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
            "Continue asking until you feel you have enough insight. Conclude with: 'Thank you so much for your help!'\n"
            "Remain in character throughout your response."
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
    languages = ["en", "fr", "es"]
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

    languages = ["en", "fr", "es"]
    all_results = []
    for lang in languages:
         translated_query = translate_query(base_query, lang)
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

    languages = ["en", "fr", "es"]
    all_results = []
    for lang in languages:
         translated_query = translate_query(base_query, lang)
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

    languages = ["en", "fr", "es"]
    all_results = []
    for lang in languages:
         translated_query = translate_query(base_query, lang)
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

    languages = ["en"]
    all_results = []
    for lang in languages:
         translated_query = translate_query(base_query, lang)
         search_query_lang = helpers_demo.SearchQuery(search_query=translated_query)
         results = helpers_demo.search_docs_rag_get(search_query_lang)
         all_results.append(results)
    aggregated_results = "\n\n---\n\n".join(all_results)
    smart_print(message=aggregated_results, agent_name="Search Docs RAG",
                message_type="NEW inference result recieved", column_id=0, column_max=1)
    return {"context": [aggregated_results]}

def generate_answer(state: InterviewState):
    print("Generate_answer")
    if "answer_instructions" not in state:
        state["answer_instructions"] = (
            "You are an expert being interviewed.\n\n"
            "Your focus is:{goals}\n\n"
            "Answer the following question using only the provided context:{context}\n\n"
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
        return {"expert_response": [answer_json.response], "expert_resources": answer_json.sources}
    except Exception as e:
        # Fallback to the original extract_json method if parsing fails
        answer_json = extract_json(answer_resp.content)
        smart_print(message=answer_json, agent_name="Generate Answer", message_type="NEW inference result recieved", column_id=0, column_max=1)
        return {"expert_response": [answer_json.get("response", "")], "expert_resources": answer_json.get("sources", [])}

def save_interview(state: InterviewState):
    print("Save_interview")
    messages = state["messages"]
    interview = get_buffer_string(messages)
    return {"interview": interview}

def route_messages(state: InterviewState, name: str = "expert"):
    print("Route_messages")
    messages = state["messages"]
    max_num_turns = state.get('max_num_turns', 2)
    num_responses = len([m for m in messages if isinstance(m, AIMessage) and m.name == name])
    if num_responses >= max_num_turns:
        return 'save_interview'
    last_question = messages[-2]
    if "Thank you so much for your help" in last_question.content:
        return 'save_interview'
    return "ask_question"

def write_section(state: InterviewState):
    print("Write_section")
    if "section_writer_instructions" not in state:
        state["section_writer_instructions"] = (
            "You are an expert technical writer. Your task is to create a detailed and comprehensive section of a report from the provided source documents. "
            "You have access to the current report state and the original plan. Follow these steps:\n"
            "1. Thoroughly analyze the provided documents (each begins with a <Document tag) and the current report content.\n"
            "2. Structure your section using Markdown with appropriate headers (e.g., ## for titles), and include detailed explanations, technical insights, and critical analysis.\n"
            "3. Your section should include:\n"
            "   a. A compelling title (## header) based on the analyst’s focus: {focus}\n"
            "   b. A detailed summary (### header) that provides context, highlights novel insights, and includes a numbered list of sources with expanded explanations.\n"
            "   c. A detailed sources list (### header) with full links or document paths.\n"
            "Aim to append new information and expand on existing content with substantial detail, ensuring clarity and depth without repeating what has already been provided."
        )
    section_writer_instructions = state["section_writer_instructions"]
    analyst = state["analyst"]
    context = state["context"]
    current_report = state.get("final_report", "")
    initial_plan = state.get("initial_plan", "")
    system_message = section_writer_instructions.format(focus=analyst.description)
    human_msg = (
        f"Use this source to write your section: {context}\n"
        f"Use this expert response to write your section: {state['expert_response'] if 'expert_response' in state else ''}\n"
        f"Current Report: {current_report}\n"
        f"Original Plan: {initial_plan}\n"
        "TASK: Append new information in this section without repeating what is already present."
    )
    sections = helpers_demo.llm_custom.invoke([
        SystemMessage(content=system_message),
        HumanMessage(content=human_msg)
    ])
    section_final = ""
    if isinstance(sections, list):
        for section in sections:
            section_final += section.content if isinstance(section, AIMessage) else section
    else:
        section_final = sections.content if isinstance(sections, AIMessage) else sections
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
    if state.get("iteration", 0) < state.get("max_iterations", 2):
        print(f"-----------### END OF ITERATION {state.get('iteration', 0)} ###-----------")
        return "reset_for_iteration"
    else:
        print(f"-----------### END OF ITERATION {state.get('iteration', 0)} ###-----------")
        return END

### ----- Build and Invoke Graph -----

SEARCH_STRATEGY = "arxiv"  # same approach for searching
def choose_search_strategy(state: ResearchGraphState):
    global SEARCH_STRATEGY
    if automate_graph:
        SEARCH_STRATEGY = "default" if SEARCH_STRATEGY is None else SEARCH_STRATEGY
    else:
        SEARCH_STRATEGY = smart_input(
            "What searching strategy do you want to use? (Web, Wikipedia, ArXiv, Semantic, All): ",
            column_id=0, column_max=1, optional=False
        ).lower()
        SEARCH_STRATEGY = (
            "default" if SEARCH_STRATEGY not in ["web","wikipedia","arxiv","semantic","all"] else SEARCH_STRATEGY
        )
    print(f"Search strategy chosen: {SEARCH_STRATEGY}")
    return state

def route_to_search_nodes(state: InterviewState):
    global SEARCH_STRATEGY
    targets = ["search_docs_rag"]
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
    return targets

def initiate_all_interviews(state: ResearchGraphState):
    print("Initiate_all_interviews")
    human_analyst_feedback = state.get('human_analyst_feedback')
    if human_analyst_feedback:
        return "create_analysts"
    else:
        topic = state["topic"]
        return [
            Send("conduct_interview", {
                "analyst": a,
                "messages": [HumanMessage(content=f"So you said you were writing an article on {topic}?")]
            }) for a in state["analysts"]
        ]

### --------- Combine everything into a main function ----------

from langgraph.graph import StateGraph

def score_generated_report_with_existing_env(final_report_in_latex: str, topic: str = "Complex QA and language models hybrid architectures, Survey"):
    """
    Reuses the scoring logic from env.py by creating a VoyagerEnvIR_CPS_TechSynthesis
    environment, loading the same references, inserting 'final_report' into it,
    then calling get_score().
    """

    # SHOULD NOT BE INITIALIZED EVERY TIME BUT ONLY ONCE, SHOULD ALSO SEARCH IN THE LIST OF AVAILABLE JSON OUTPUTS
    if topic == "Complex QA and language models hybrid architectures, Survey":
        doc={ 'id':"cf0d353c-b43b-4a79-88f9-42c2c84cf75e",
                    'title':"Complex QA and language models hybrid architectures, Survey",
                'context':"This paper reviews the state-of-the-art of language models architectures and strategies for 'complex' question-answering (QA, CQA, CPS) with a focus on hybridization. Large Language Models (LLM) are good at leveraging public data on standard problems but once you want to tackle more specific complex questions or problems (e.g. How does the concept of personal freedom vary between different cultures ? What is the best mix of power generation methods to reduce climate change ?) you may need specific architecture, knowledge, skills, methods, sensitive data protection, explainability, human approval and versatile feedback... Recent projects like ChatGPT and GALACTICA have allowed non-specialists to grasp the great potential as well as the equally strong limitations of LLM in complex QA. In this paper, we start by reviewing required skills and evaluation techniques. We integrate findings from the robust community edited research papers BIG, BLOOM and HELM which open source, benchmark and analyze limits and challenges of LLM in terms of tasks complexity and strict evaluation on accuracy (e.g. fairness, robustness, toxicity, ...) as a baseline. We discuss some challenges associated with complex QA, including domain adaptation, decomposition and efficient multi-step QA, long form and non-factoid QA, safety and multi-sensitivity data protection, multimodal search, hallucinations, explainability and truthfulness, temporal reasoning. We analyze current solutions and promising research trends, using elements such as: hybrid LLM architectural patterns, training and prompting strategies, active human reinforcement learning supervised with AI, neuro-symbolic and structured knowledge grounding, program synthesis, iterated decomposition and others.",
                'target_file_path': "env/IR_CPS_TechSynthesis/document_embedding_analysis/output/arxiv/Complex QA and language models hybrid architectures Survey.json"}

        env = EnvironmentManager(
            env_type="techsynthesis",
            title=doc['title'],
            context=doc['context'],
            target_file_path=doc['target_file_path'],
            id=doc['id'],
            llm=llm_list["default_llm"],
            embedding_model_name=config.common_vectordb_config.embedding_function
        ).get_environment()


        env.reset()
        env.synthesis_manager.GetFromLatex(final_report_in_latex)

        scores = env.get_score()
        if scores is not isinstance(scores, dict):
            scores = {'score': str(scores)}
    else:
        # Evaluate score using an LLM
        scores = {"qualitative feedback": "this article is not structured as a scientific article and robust citations are missing"}

    return scores

def score_document(state: ResearchGraphState):
    """
    Node to evaluate the final_report using the existing scoring logic from env.py.
    This node runs after finalize_report and before self_critique.
    """

    final_latex_report = state.get("latex_report", "")
    if not final_latex_report:
        print("No final_report found in state. Skipping scoring...")
        return {}

    scores = score_generated_report_with_existing_env(final_latex_report, state["topic"])

    # Optionally store them in the state so the next node can see them
    return { "scores": scores}

def multi_agent_research_generation_persist_at_the_end(
    title, topic, max_analysts: int=3, max_iterations: int=2, auto_n_rounds_planner: int=0, auto_n_rounds_section_writer: int=0, auto_n_rounds_analyst: int=0, auto_n_rounds_critic: int=0, auto_n_rounds_latex: int=0, automation: str="full_auto", trace_optimization: bool=False
):
    print("Multi_agent_research_generation_persist_at_the_end")

    if automation == "full_auto":
        auto_n_rounds_analyst = auto_n_rounds_section_writer = auto_n_rounds_analyst = auto_n_rounds_critic = auto_n_rounds_latex = 999

    global planner, section_writer, critic, latex_gen, analyst
    planner = HumanLLM(agent_name="Planner", llmORchains_list=llm_list, automation=automation, auto_n_rounds=auto_n_rounds_planner)
    section_writer = HumanLLM(agent_name="Section Writer", llmORchains_list=llm_list, automation=automation, auto_n_rounds=auto_n_rounds_section_writer)
    critic = HumanLLM(agent_name="Self Critic", llmORchains_list=llm_list, automation=automation, auto_n_rounds=auto_n_rounds_critic)
    latex_gen = HumanLLM(agent_name="Generate Latex", llmORchains_list=llm_list, automation=automation, auto_n_rounds=auto_n_rounds_latex)
    analyst = HumanLLM(agent_name="Create Analysts", llmORchains_list=llm_list, automation=automation, auto_n_rounds=auto_n_rounds_analyst)

    initial_state: ResearchGraphState = {
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
        "iteration": 0,
        "max_iterations": max_iterations,
        "latex_report": "",
        "cumulative_report": "",
        "parsed_plan": {},
        "old_parsed_plan": {},
        "critic": "",
        "source_list": [],
        "scores": {},
    }

    builder = StateGraph(ResearchGraphState)

    # --- Add our nodes ---
    builder.add_node("plan_document", plan_document)
    builder.add_node("merge_plan", merge_plan)
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
    interview_graph_def.add_conditional_edges("ask_question", route_to_search_nodes,
        ["search_docs_rag", "search_web", "search_wikipedia", "search_arxiv", "search_semantic_scholar"]
    )
    interview_graph_def.add_edge("search_docs_rag", "answer_question")
    interview_graph_def.add_edge("search_web", "answer_question")
    interview_graph_def.add_edge("search_wikipedia", "answer_question")
    interview_graph_def.add_edge("search_arxiv", "answer_question")
    interview_graph_def.add_edge("search_semantic_scholar", "answer_question")
    interview_graph_def.add_edge("answer_question", "write_section")
    #interview_graph_def.set_start("ask_question") # to be replace by add_edge START
    interview_graph_def.add_edge(START, "ask_question")
    #interview_graph_def.set_end("write_section")
    interview_graph_def.add_edge("write_section", END)
    compiled_interview_graph = interview_graph_def.compile()

    builder.add_node("conduct_interview", compiled_interview_graph)

    # --- Edges in main graph ---
    builder.add_edge("plan_document", "merge_plan")
    builder.add_edge("merge_plan", "create_analysts")
    builder.add_edge("create_analysts", "choose_search_strategy")

    builder.add_conditional_edges("choose_search_strategy", initiate_all_interviews,
        ["create_analysts","conduct_interview"]
    )

    builder.add_edge("conduct_interview", "write_all_sections")

    # Here we do finalize_report, then self_critique
    builder.add_edge("write_all_sections", "finalize_report")
    builder.add_edge("finalize_report", "generate_latex")
    builder.add_edge("generate_latex", "score_document")
    builder.add_edge("score_document", "self_critique")

    # The self_critique can lead to a new iteration or final
    builder.add_conditional_edges("self_critique", should_iterate,
        ["reset_for_iteration", END]
    )
    builder.add_edge("reset_for_iteration", "plan_document")
    #builder.set_start("plan_document")
    builder.add_edge(START, "plan_document")

    # Checkpointer
    #memory2 = MemorySaver()
    #graph = builder.compile(checkpointer=memory2)
    graph = builder.compile()

    # Optionally add your RAG documents
    # for file in os.listdir("BIBLIO-TEST"):
    #     if file != '.DS_Store':
    #         HumanLLM(agent_name="add_rag_doc_start", llmORchains_list=llm_list).add_rag_document(
    #             folder_path="BIBLIO-TEST", file_path=file,
    #             chunking_options={"chunk_size":1000,"chunk_overlap":200},
    #             use_semantic_chunking=True
    #         )

    # Invoke
    #graph.invoke(initial_state, params={ "configurable": { "thread_id": "1" }, "recursion_limit": 100 })
    result = graph.invoke(initial_state, { "configurable": { "thread_id": "1" }, "recursion_limit": 100 })
    #final_state = graph.get_state({ "configurable": { "thread_id": "1" }, "recursion_limit": 100 })

    # Save final artifacts
    #report = final_state.values.get("final_report")
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

    # Storing graph's diagram
    try:
        if not os.path.exists("images"): os.makedirs("images")
        with open("images/graph_png.png", "wb") as f:
            f.write(graph.get_graph(xray=1).draw_mermaid_png())
    except Exception as e:
        print(f"Error saving graph image: {e}")

    if trace_optimization:
        return {'final_report': node(latex_report)}
    return latex_report


# -------------- Example usage --------------
if __name__ == "__main__":
    final_report = multi_agent_research_generation_persist_at_the_end(
        title="State of the art on Leading Edge Noise",
        topic="Research assistant framework for State of the art on Leading Edge Noise",
        max_analysts=1,
        max_iterations=1
    )
    print("\n==== Final Report ====\n", final_report)