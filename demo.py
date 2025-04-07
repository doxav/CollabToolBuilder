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

from langgraph.constants import Send
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import START, END, StateGraph
from langgraph.checkpoint.memory import MemorySaver

from config import embedding_function, use_websocket
from utils.human_llm import HumanLLM, HumanLLMConfig
from utils.llm_utils import smart_input, smart_print
from utils.graph_utils.analysts import Analyst, create_analysts
from env.IR_CPS_TechSynthesis.env import VoyagerEnvIR_CPS_TechSynthesis, Section
from env.env import EnvironmentManager
from utils.graph_utils.helpers_demo import (
    remove_think_tags,
    extract_latex_and_bib_from_llm_output,
    extract_json
)
from utils.graph_utils.interview import (
    InterviewState,
    generate_question, generate_answer, write_section,
    search_docs_rag, search_web, search_wikipedia, search_arxiv, search_semantic_scholar
)

# 1) Initialize config + LLM handles
config = HumanLLMConfig()
llm_list = config.get_llmORchains_list()
config.use_websocket = use_websocket
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

planner = HumanLLM(agent_name="Planner", llmORchains_list=llm_list)
section_writer = HumanLLM(agent_name="Section Writer", llmORchains_list=llm_list)
critic = HumanLLM(agent_name="Self Critic", llmORchains_list=llm_list)
latex_gen = HumanLLM(agent_name="Generate Latex", llmORchains_list=llm_list)

### ----- Extended State to include new fields -----

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

def should_iterate(state: ResearchGraphState):
    if state.get("iteration", 0) < state.get("max_iterations", 2):
        print(f"-----------### END OF ITERATION {state.get('iteration', 0)} ###-----------")
        return "reset_for_iteration"
    else:
        print(f"-----------### END OF ITERATION {state.get('iteration', 0)} ###-----------")
        return END

### ----- Build and Invoke Graph -----

SEARCH_STRATEGY = None  # same approach for searching
def choose_search_strategy(state: ResearchGraphState):
    global SEARCH_STRATEGY
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

def score_generated_report_with_existing_env(final_report: str):
    """
    Reuses the scoring logic from env.py by creating a VoyagerEnvIR_CPS_TechSynthesis
    environment, loading the same references, inserting 'final_report' into it,
    then calling get_score().
    """

    documents=[{ 'id':"cf0d353c-b43b-4a79-88f9-42c2c84cf75e",
                'title':"Complex QA and language models hybrid architectures, Survey",
            'context':"This paper reviews the state-of-the-art of language models architectures and strategies for 'complex' question-answering (QA, CQA, CPS) with a focus on hybridization. Large Language Models (LLM) are good at leveraging public data on standard problems but once you want to tackle more specific complex questions or problems (e.g. How does the concept of personal freedom vary between different cultures ? What is the best mix of power generation methods to reduce climate change ?) you may need specific architecture, knowledge, skills, methods, sensitive data protection, explainability, human approval and versatile feedback... Recent projects like ChatGPT and GALACTICA have allowed non-specialists to grasp the great potential as well as the equally strong limitations of LLM in complex QA. In this paper, we start by reviewing required skills and evaluation techniques. We integrate findings from the robust community edited research papers BIG, BLOOM and HELM which open source, benchmark and analyze limits and challenges of LLM in terms of tasks complexity and strict evaluation on accuracy (e.g. fairness, robustness, toxicity, ...) as a baseline. We discuss some challenges associated with complex QA, including domain adaptation, decomposition and efficient multi-step QA, long form and non-factoid QA, safety and multi-sensitivity data protection, multimodal search, hallucinations, explainability and truthfulness, temporal reasoning. We analyze current solutions and promising research trends, using elements such as: hybrid LLM architectural patterns, training and prompting strategies, active human reinforcement learning supervised with AI, neuro-symbolic and structured knowledge grounding, program synthesis, iterated decomposition and others.",
            'target_file_path': "env/IR_CPS_TechSynthesis/document_embedding_analysis/output/arxiv/Complex QA and language models hybrid architectures Survey.json"},
            { 'id':"42252c6c-12f3-4edf-9045-8acd69bc3356",
                'title':"Macroeconomic Effects of Inflation Targeting A Survey of the Empirical  Literature",
            'context':"This paper surveys the empirical literature of inflation targeting. The main findings from our review are the following: there is robust empirical evidence that larger and more developed countries are more likely to adopt the IT regime; the introduction of this regime is conditional on previous disinflation, greater exchange rate flexibility, central bank independence, and higher level of financial development; the empirical evidence has failed to provide convincing evidence that IT itself may serve as an effective tool for stabilizing inflation expectations and for reducing inflation persistence; the empirical research focused on advanced economies has failed to provide convincing evidence on the beneficial effects of IT on inflation performance, while there is some evidence that the gains from the IT regime may have been more prevalent in the emerging market economies; there is not convincing evidence that IT is associated with either higher output growth or lower output variability; the empirical research suggests that IT may have differential effects on exchange-rate volatility in advanced economies versus EMEs; although the empirical evidence on the impact of IT on fiscal policy is quite limited, it supports the idea that IT indeed improves fiscal discipline; the empirical support to the proposition that IT is associated with lower disinflation costs seems to be rather weak. Therefore, the accumulated empirical literature implies that IT does not produce superior macroeconomic benefits in comparison with the alternative monetary strategies or, at most, they are quite modest.",
            'target_file_path': "env/IR_CPS_TechSynthesis/document_embedding_analysis/output/arxiv/Macroeconomic Effects of Inflation Targeting A Survey of the Empirical  Literature.json"}]

    # 1) Initialize the environment with any mandatory arguments:
    #    For instance, if 'title' or 'goal' is required, supply placeholders or real data.
    #    Importantly, set 'target_file_path' to your FIRST reference JSON:
    
    envs_tech_synthesis = []
    for doc in documents:
        env = EnvironmentManager(
            env_type="techsynthesis",
            title=doc['title'],
            context=doc['context'],
            target_file_path=doc['target_file_path'],
            id=doc['id'],
            llm=llm_list["default_llm"],
            embedding_model_name=config.common_vectordb_config.embedding_function
        ).get_environment()
        envs_tech_synthesis.append(env)



    # 2) The environment logic expects "sections" making up the doc. If final_report is
    #    mostly one big text, we can push it in as a single 'Section'.
    #    If you prefer multiple sections, parse and create multiple:
    env.reset()
    env.synthesis_manager.create_and_add_section_then_return_id(
        title="Generated Document",
        content=final_report
    )

    # 3) Score against the FIRST reference
    score_first = env.get_score()  # This reuses 'get_distance_to_targetJSON' etc.

    # Then call get_score() again:
    score_second = env.get_score()

    # 5) Merge or average those two scores as you like:
    merged_scores = {}
    for key in score_first.keys():
        val1 = score_first[key]
        val2 = score_second.get(key, 0)
        merged_scores[key] = round((val1 + val2) / 2, 4)  # simple average

    # 6) Return or print them
    return score_first, score_second, merged_scores

def score_document(state: ResearchGraphState):
    """
    Node to evaluate the final_report using the existing scoring logic from env.py.
    This node runs after finalize_report and before self_critique.
    """

    final_report = state.get("final_report", "")
    if not final_report:
        print("No final_report found in state. Skipping scoring...")
        return {}

    # REUSE your existing environment-based scoring approach
    # (the snippet you provided, extracted into a function)
    score1, score2, merged = score_generated_report_with_existing_env(final_report)

    smart_print(f"Score vs Reference #1: {score1}", "score_document", "NEW INFERENCE RESULT RECEIVED", column_id=0, column_max=1)
    smart_print(f"Score vs Reference #2: {score2}", "score_document", "NEW INFERENCE RESULT RECEIVED", column_id=0, column_max=1)
    smart_print(f"Merged Score: {merged}", "score_document", "NEW INFERENCE RESULT RECEIVED", column_id=0, column_max=1)

    # Optionally store them in the state so the next node can see them
    return {
        "score_1": score1,
        "score_2": score2,
        "score_merged": merged
    }


def multi_agent_research_generation_persist_at_the_end(
    title, topic, max_analysts: int=3, max_iterations: int=2
):
    print("Multi_agent_research_generation_persist_at_the_end")

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
        "source_list": []
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
    interview_graph_def.add_node("route_to_search_nodes", route_to_search_nodes)
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
    memory2 = MemorySaver()
    graph = builder.compile(checkpointer=memory2)

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
    graph.invoke(initial_state, { "configurable": { "thread_id": "1" }, "recursion_limit": 100 })
    final_state = graph.get_state({ "configurable": { "thread_id": "1" }, "recursion_limit": 100 })

    # Save final artifacts
    report = final_state.values.get("final_report")
    latex_report, bibtex_report = extract_latex_and_bib_from_llm_output(
        final_state.values.get("latex_report","")
    )

    report_folder = "./report_outputs"
    if not os.path.exists(report_folder):
        os.makedirs(report_folder)

    if latex_report:
        with open(f"{report_folder}/{topic}.tex", "w") as f:
            f.write(latex_report)
    if bibtex_report:
        with open(f"{report_folder}/{topic}.bib", "w") as f:
            f.write(bibtex_report)

    # Example of storing graph diagram
    if not os.path.exists("images"):
        os.makedirs("images")
    with open("images/graph_png.png", "wb") as f:
        f.write(graph.get_graph(xray=1).draw_mermaid_png())

    return report


# -------------- Example usage --------------
if __name__ == "__main__":
    final_report = multi_agent_research_generation_persist_at_the_end(
        title="State of the art on Leading Edge Noise",
        topic="Research assistant framework for State of the art on Leading Edge Noise",
        max_analysts=1,
        max_iterations=1
    )
    print("\n==== Final Report ====\n", final_report)