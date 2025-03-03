# -*- coding: utf-8 -*-
"""Iterative Research Document Generation with Self-Critique and Improvement

This script demonstrates a multi-agent LangGraph workflow that:
1. Generates an initial research document (plan, interviews, sections, citations).
2. Self-critiques the report and updates the plan.
3. Loops back to re-run the process for a fixed number of iterations (default 2)
   so that the report can be improved iteratively.

Key changes:
- New nodes: `self_critique` (to critique and update the plan)
             `reset_for_iteration` (to clear prior outputs)
- A conditional edge that loops from the self-critique stage back to the planning stage.
- The state now includes `iteration` and `max_iterations`.
"""
import os
import operator
from typing import List, Annotated
from typing_extensions import TypedDict

from langgraph.constants import Send
from langchain_core.messages import HumanMessage
from langgraph.graph import START, END, StateGraph
from langgraph.checkpoint.memory import MemorySaver

from config import embedding_function, use_websocket
from utils.human_llm import HumanLLM, HumanLLMConfig
from utils.llm_utils import smart_input, smart_print
from utils.graph_utils.analysts import Analyst, create_analysts
from utils.graph_utils.helpers_demo import remove_think_tags, extract_latex_and_bib_from_llm_output, extract_json   
from utils.graph_utils.interview import InterviewState, generate_question, generate_answer, write_section, search_docs_rag, search_web, search_wikipedia, search_arxiv, search_semantic_scholar

# HumanLLM.use_websocket = True
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

config.common_vectordb_config.db_type = "elasticsearch"
config.initialize()

planner = HumanLLM(agent_name="Planner", llmORchains_list=llm_list)
question = HumanLLM(agent_name="Generate Questions", llmORchains_list=llm_list)
answer = HumanLLM(agent_name="Generate Aswers", llmORchains_list=llm_list)
writerr = HumanLLM(agent_name="Write Report", llmORchains_list=llm_list)
writers = HumanLLM(agent_name="Write Section", llmORchains_list=llm_list)
writeri = HumanLLM(agent_name="Write Introduction", llmORchains_list=llm_list)
writerrl = HumanLLM(agent_name="Write Resource List", llmORchains_list=llm_list)
writerc = HumanLLM(agent_name="Write Conclusion", llmORchains_list=llm_list)
critic = HumanLLM(agent_name="Self Critic", llmORchains_list=llm_list)
latex_gen = HumanLLM(agent_name="Generate Latex", llmORchains_list=llm_list)

### -------------------------------
# RESEARCH REPORT NODES (INCLUDING PLAN, RESOURCE LIST, AND SELF-CRITIQUE)
### -------------------------------

# Extend state to include new keys for iterative improvement.
class ResearchGraphState(TypedDict):
    topic: str                                          # Research topic
    max_analysts: int                                   # Number of analysts
    human_analyst_feedback: str                         # Human feedback
    analysts: List[Analyst]                             # List of analysts
    sections: Annotated[list, operator.add]             # Collected sections from interviews
    plan: str                                           # Overall plan for the document
    introduction: str                                   # Introduction text
    content: str                                        # Main report content (without a conclusion)
    conclusion: str                                     # Conclusion text
    resource_list: str                                  # Consolidated resource list (citations)
    final_report: str                                   # The final combined report
    iteration: int                                      # Current iteration count
    max_iterations: int                                 # Maximum allowed iterations
    latex_report: str                                   # The final report in latex
    cumulative_report: str                              # The cumulative report

def plan_document(state: ResearchGraphState):
    print("Plan_document")
    filepath = "plan_document"
    prompt_file = f"./prompts/{filepath}.txt"
    if not os.path.exists(prompt_file):
        with open(prompt_file, "w") as f:
            f.write(
                "You are a planning assistant. Your task is to create a detailed plan for a research document. "
                "In the user message, you will receive a value labeled 'TOPIC' which represents the research topic. "
                "Please outline the key sections (for example, Introduction, Main Content, Conclusion) and provide a detailed description for each section, including specific subtopics, key arguments, and supporting evidence. "
                "Ensure that the plan is clear, well-structured, and suggests areas for further elaboration without overwriting any existing content."
            )
    user_message = f"TOPIC: <<< {state['topic']} >>>"
    plan = planner.invoke(
        system_prompt_template=filepath,
        user_message=user_message,
        stream_output=False,
        return_message_content_only=True,
        use_default_llm=False
    )[0]
    plan_text = remove_think_tags(plan.content if hasattr(plan, "content") else plan)
    # Save the original plan for later context if not already stored.
    if not state.get("initial_plan"):
        state["initial_plan"] = plan_text
    return {"plan": plan_text}

def write_report(state: ResearchGraphState):
    print("Write_report")
    filepath = "write_report"
    prompt_file = f"./prompts/{filepath}.txt"
    # (Prompt file creation code remains unchanged.)
    sections = state["sections"]
    formatted_str_sections = "\n\n".join(sections) if isinstance(sections, list) else sections
    current_report = state.get("final_report", "")
    user_message = (
        f"TOPIC: <<< {state['topic']} >>>\n"
        f"PLAN: <<< {state['plan']} >>>\n"
        f"MEMOS: <<< {formatted_str_sections} >>>\n"
        f"FINAL_REPORT (so far): <<< {current_report} >>>\n"
        "TASK: Consolidate the memos into a cohesive narrative by appending only new insights. "
        "Ensure that the narrative is enriched with detailed analysis, additional context, and deeper explanations. "
        "Avoid overwriting previous content; instead, build upon what is already present with substantial new information."
    )
    if state.get("critic"):
        user_message += f"\nPrevious Critique: <<< {state['critic']} >>>"
    report = writerr.invoke(
        system_prompt_template=filepath,
        user_message=user_message,
        stream_output=False,
        return_message_content_only=True,
        use_default_llm=False
    )[0]
    report_text = remove_think_tags(report.content if hasattr(report, "content") else report)
    print(f"report generated : {report_text}")
    # Instead of overwriting, update content by appending:
    state["content"] = state.get("content", "") + "\n" + report_text
    return {"content": state["content"]}

def write_introduction(state: ResearchGraphState):
    print("Write_introduction")
    filepath = "write_introduction"
    prompt_file = f"./prompts/{filepath}.txt"
    if not os.path.exists(prompt_file):
        with open(prompt_file, "w") as f:
            f.write(
                "You are a technical writer tasked with writing the introduction for a research report. "
                "In the user message, you will receive the following values:\n"
                "  - 'TOPIC': the research topic,\n"
                "  - 'PLAN': the updated plan for the report,\n"
                "  - 'INITIAL_PLAN': the original plan for context,\n"
                "  - 'CURRENT_INTRO': any existing introduction,\n"
                "  - 'FINAL_REPORT': the current full report content.\n\n"
                "Your task is to craft a crisp and compelling introduction. Begin with a title using a '#' header, followed by a section header '## Introduction'. "
                "The introduction should be approximately 100 words and clearly preview the report sections. Append only new information without repeating existing content."
            )
    current_intro = state.get("introduction", "")
    full_report = state.get("final_report", "")
    initial_plan = state.get("initial_plan", "")
    user_message = (
        f"TOPIC: <<< {state['topic']} >>>\n"
        f"PLAN: <<< {state['plan']} >>>\n"
        f"INITIAL_PLAN: <<< {initial_plan} >>>\n"
        f"CURRENT_INTRO: <<< {current_intro} >>>\n"
        f"FINAL_REPORT: <<< {full_report} >>>\n"
        "TASK: Write or expand the introduction. Only add new information and do not repeat content."
    )
    if state.get("critic"):
        user_message += f"\nPrevious Critique: <<< {state['critic']} >>>"
    intro = writeri.invoke(
        system_prompt_template=filepath,
        user_message=user_message,
        stream_output=False,
        return_message_content_only=True,
        use_default_llm=False
    )[0]
    intro_text = remove_think_tags(intro.content if hasattr(intro, "content") else intro)
    print(f"introduction generated : {intro_text}")
    return {"introduction": intro_text}

def write_conclusion(state: ResearchGraphState):
    print("Write_conclusion")
    filepath = "write_conclusion"
    prompt_file = f"./prompts/{filepath}.txt"
    if not os.path.exists(prompt_file):
        with open(prompt_file, "w") as f:
            f.write(
                "You are a technical writer tasked with writing a crisp and compelling conclusion for a research report. "
                "In the user message, you will receive the following values:\n"
                "  - 'TOPIC': the research topic,\n"
                "  - 'PLAN': the updated plan,\n"
                "  - 'MEMOS': the report sections,\n"
                "  - 'FINAL_REPORT': the current full report content,\n"
                "  - 'CURRENT_CONCLUSION': any existing conclusion.\n\n"
                "Your task is to write or expand the conclusion by summarizing key insights. Do not repeat content already present. "
                "Begin with '## Conclusion' and append new information only."
            )
    current_conclusion = state.get("conclusion", "")
    full_report = state.get("final_report", "")
    sections = state["sections"]
    formatted_str_sections = "\n\n".join(sections) if isinstance(sections, list) else sections
    user_message = (
        f"TOPIC: <<< {state['topic']} >>>\n"
        f"PLAN: <<< {state['plan']} >>>\n"
        f"MEMOS: <<< {formatted_str_sections} >>>\n"
        f"CURRENT_CONCLUSION: <<< {current_conclusion} >>>\n"
        f"FINAL_REPORT: <<< {full_report} >>>\n"
        "TASK: Write or expand the conclusion by appending new insights without repeating existing content."
    )
    if state.get("critic"):
        user_message += f"\nPrevious Critique: <<< {state['critic']} >>>"
    conclusion = writerc.invoke(
        system_prompt_template=filepath,
        user_message=user_message,
        stream_output=False,
        return_message_content_only=True,
        use_default_llm=False
    )[0]
    conclusion_text = remove_think_tags(conclusion.content if hasattr(conclusion, "content") else conclusion)
    print(f"conclusion generated : {conclusion_text}")
    return {"conclusion": conclusion_text}

def write_resource_list(state: ResearchGraphState):
    print("Write_resource_list")
    filepath = "write_resource_list"
    prompt_file = f"./prompts/{filepath}.txt"
    if not os.path.exists(prompt_file):
        with open(prompt_file, "w") as f:
            f.write(
                "You are an expert technical writer. Your task is to extract and consolidate all resource citations from report sections. "
                "In the user message, you will receive the following values:\n"
                "  - 'SECTIONS': the report sections,\n"
                "  - 'FINAL_REPORT': the current full report content,\n"
                "  - 'CURRENT_RESOURCE_LIST': any existing resource list.\n\n"
                "Generate a consolidated resource list in Markdown under the header '## Sources'. List only new citations that are not already present, "
                "formatting each as: '[n] <source details>'."
            )
    sections = state["sections"]
    formatted_str_sections = "\n\n".join(sections) if isinstance(sections, list) else sections
    current_resource_list = state.get("resource_list", "")
    user_message = (
        f"SECTIONS: <<< {formatted_str_sections} >>>\n"
        f"CURRENT_RESOURCE_LIST: <<< {current_resource_list} >>>\n"
        f"FINAL_REPORT: <<< {state.get('final_report', '')} >>>\n"
        "TASK: Consolidate the citations and list new ones without repeating existing citations."
    )
    resource_list = writerrl.invoke(
        system_prompt_template=filepath,
        user_message=user_message,
        stream_output=False,
        return_message_content_only=True,
        use_default_llm=False
    )[0]
    resource_list_text = remove_think_tags(resource_list.content if hasattr(resource_list, "content") else resource_list)
    print(f"resources generated : {resource_list_text}")
    return {"resource_list": resource_list_text}

def finalize_report(state: ResearchGraphState):
    print("Finalize_report")
    if not state:
        return {"final_report": ""}
    content = state.get("content", "")
    introduction = state.get("introduction", "")
    conclusion = state.get("conclusion", "")
    resource_list = state.get("resource_list", "")
    # Build the current iteration's report
    current_report = ""
    if introduction:
        current_report += introduction + "\n\n---\n\n"
    if content:
        current_report += content
    if conclusion and "Conclusion" not in content:
        current_report += "\n\n---\n\n" + conclusion
    if resource_list and resource_list.strip():
        current_report += "\n\n" + resource_list
    # Append current iteration's report to cumulative_report
    cumulative = state.get("cumulative_report", "")
    if cumulative:
        cumulative += "\n\n" + current_report
    else:
        cumulative = current_report
    state["cumulative_report"] = cumulative
    # Also update final_report to reflect the cumulative content
    return {"final_report": cumulative}

def self_critique(state: ResearchGraphState):
    """
    Updated Self-Critique: Reviews the final report and returns both an updated plan and an overall critique.
    The response is expected as JSON with keys 'plan' and 'critic'.
    """
    print("Self_critique")
    filepath = "self_critique"
    prompt_file = f"./prompts/{filepath}.txt"
    if not os.path.exists(prompt_file):
        with open(prompt_file, "w") as f:
            f.write(
                "You are an expert critic. Your task is to review a research report and provide constructive criticism with suggestions for improvement. "
                "In the user message, you will receive a value labeled 'REPORT' which contains the current research report. "
                "After analyzing the report, generate an updated plan that addresses the identified issues and also provide an overall critique of the report. "
                "Return your answer as a JSON object with keys 'plan' (the updated plan) and 'critic' (the overall critique)."
            )
    current_report = state.get("final_report", "")
    user_message = f"REPORT: <<< {current_report} >>>\nTASK: Critique the report and update the plan. Return a JSON object with keys 'plan' and 'critic'."
    critique_response = critic.invoke(
        system_prompt_template=filepath,
        user_message=user_message,
        stream_output=False,
        return_message_content_only=True,
        use_default_llm=False
    )[0]
    critique_data = extract_json(remove_think_tags(critique_response))
    state["plan"] = critique_data.get("plan", state.get("plan"))
    state["critic"] = critique_data.get("critic", "")
    state["iteration"] = state.get("iteration", 0) + 1
    return {"plan": state["plan"], "iteration": state["iteration"]}

def reset_for_iteration(state: ResearchGraphState):
    """
    Node to clear previous iteration's outputs.
    Resets the sections, introduction, content, conclusion, resource list, and final report.
    """
    print("Reset_for_iteration")
    state["sections"] = []
    state["introduction"] = ""
    state["content"] = ""
    state["conclusion"] = ""
    state["resource_list"] = ""
    return state

def should_iterate(state: ResearchGraphState):
    """
    Conditional function for looping.
    Returns the next node:
    - If current iteration is less than max_iterations, return "reset_for_iteration"
    - Otherwise, return END.
    """
    if state.get("iteration", 0) < state.get("max_iterations", 2):
        print(f"-----------### END OF ITERATION {state.get('iteration', 0)} ###-----------")
        return "reset_for_iteration"
    else:
        print(f"-----------### END OF ITERATION {state.get('iteration', 0)} ###-----------")
        return "generate_latex"
    
def generate_latex(state : ResearchGraphState):
    """
    This function will take the report and generate a latex file from it. No informations will be modified.
    :param state: The current state of the report.
    :type state: ResearchGraphState
    :return: The state of the report.
    :rtype: ResearchGraphState
    """
    print("Generate_latex")
    report = state.get("final_report", "")
    filepath = "generate_latex"
    prompt_file = f"./prompts/{filepath}.txt"
    if not os.path.exists(prompt_file):
        with open(prompt_file, "w") as f:
            f.write(
                "You are an expert tool for converting text into a LaTeX document."
                " Your role is to transform a report, provided in the user message (labeled REPORT), into a fully compilable .tex file that faithfully reproduces the original content.\n"
                "General Requirements:\n"
                "Complete reproduction of the content\n"
                "Every word, every sentence, every equation, every table, every bibliographic reference or any other element of the source text must appear in full in the LaTeX version.\n"
                "No information should be omitted, modified, or summarized.\n"
                "Do not correct spelling or grammar mistakes, do not add any additional text, do not rephrase anything.\n"
                "Neat LaTeX structure and formatting\n"
                "Create a clear structure: \\documentclass{...}, \\begin{document}, \\title{...}, \\section{...}, \\subsection{...}, etc.\n"
                "Use appropriate LaTeX environments for:\n"
                "Mathematical formulas: \\(...\\) or \\begin{equation} ... \\end{equation}, etc.\n"
                "Tables: \\begin{tabular}{...}, etc.\n"
                "Figures and images, if applicable: \\begin{figure} ... \\end{figure}.\n"
                "Lists: use itemize or enumerate if needed.\n"
                "References and citations: \\cite{...} or equivalent, as appropriate.\n"
                "Bibliography and references\n"
                "If the report contains bibliographic references or sources, you may:\n"
                "Generate a separate .bib file and reference it from the main document using \\bibliography{...} and \\bibliographystyle{...}, or\n"
                "Group the references in a dedicated section (e.g., \\section*{References}) at the end of the document.\n"
                "Keep all citation details as they appear in the original text (names, dates, titles, links, DOI, etc.).\n"
                "Strict adherence to the content\n"
                "Do not insert interpretations or personal comments.\n"
                "Do not add correction markers, highlighting, or colors that are not present in the text.\n"
                "Do not change the meaning or order of paragraphs.\n"
                "If the report appears to contain typos (mistakes, repetitions, etc.), leave them as is.\n"
                "Complete and standalone document\n"
                "The produced .tex file must be directly compilable (e.g., pdflatex document.tex or similar).\n"
                "Make sure to include all necessary packages in the preamble (\\usepackage{...}) if they are required by certain elements of the text (math, advanced tables, hyperlinks, etc.).\n"
                "Produce no output other than the final LaTeX code (no explanatory text before or after).\n"
                "Additional rules for specific cases:\n"
                "Inline mathematical formulas:\n"
                "If the original content uses an inline format (e.g., $x^2$), use \\(x^2\\) or $x^2$.\n"
                "If the content contains equations on their own line, use the appropriate LaTeX environments (\\[ ... \\], \\begin{equation}...\\end{equation}, etc.).\n"
                "Tables:\n"
                "If a table is provided in plain text (e.g., cells separated by tabs or vertical bars), reproduce this table using the appropriate LaTeX environment (e.g., \\begin{tabular} ...).\n"
                "Preserve all column headers, captions, and notes of the table.\n"
                "Footnotes or annotations:\n"
                "Convert them into proper footnotes (\\footnote{...}) if that corresponds to the original intent.\n"
                "Images and figures:\n"
                "If the text mentions the insertion of images (e.g., “see Figure 1”), create a figure environment with a caption, even if the image file is not provided.\n"
                "Place a comment in the LaTeX code (e.g., % Image placeholder: figure1.png) indicating the filename if it is mentioned.\n"
                "External links:\n"
                "If you encounter URLs (e.g., https://...), keep them as is and consider using \\href{...}{...} if the context requires.\n"
                "Begin your response with the line 'Latex:'\n"
                "Immediately after, include all the .tex elements (LaTeX code) containing the entirety of the report.\n"
                "If you use a separate .bib file, then after finishing the LaTeX part, insert a new line 'Bibtex:'\n"
                "Immediately after, include all the necessary BibTeX entries.\n"
                "If no separate .bib is generated (references included directly in the .tex), do not include the Bibtex section.\n"
                "Example structure:\n"
                "Copy\n"
                "Edit\n"
                "Latex:\n"
                "\\documentclass{article}\n"
                "...\n"
                "\\end{document}\n"
                "Bibtex:\n"
                "@article{...}\n"
                "...\n"
                "No other form of output: no text before 'Latex:', no additional explanation, no summary or comments.\n"
                "Summary:\n"
                "You will receive as input a block of text labeled REPORT.\n"
                "You convert it entirely into LaTeX, without omitting or modifying anything.\n"
                "You return a single text containing:\n"
                "The section 'Latex:' with all the .tex code.\n"
                "Optionally, the section 'Bibtex:' if a separate .bib file is required.\n"
                "Nothing else should be produced."
                )
        
    user_message = f"REPORT: <<< {report} >>>"
    latex_report = latex_gen.invoke(system_prompt_template=filepath, user_message=user_message, stream_output=False, return_message_content_only=True, use_default_llm=False)[0]
    latex_report_text = latex_report.content if hasattr(latex_report, "content") else latex_report
    latex_report_text = remove_think_tags(latex_report_text)
    print(f"latex generated : {latex_report_text}")
    return {'latex_report': latex_report_text}

SEARCH_STRATEGY = None

def choose_search_strategy(state: ResearchGraphState):
    """
    A runtime decision node that selects the search strategy.
    In a real interactive environment, this could prompt the user.
    For this example, we simulate the choice.
    """
    global SEARCH_STRATEGY
    SEARCH_STRATEGY = smart_input("What searching strategy do you want to use? (Web, Wikipedia, ArXiv, Semantic, All): ", column_id=0, column_max=1, optional=False).lower()
    
    # Validate the choice and default to "both" if unrecognized.
    SEARCH_STRATEGY = "default" if SEARCH_STRATEGY not in ["web", "wikipedia", "arxiv", "semantic", "all"] else SEARCH_STRATEGY
    
    # Store the choice as a string to ensure serializability.
    print(f"Search strategy chosen: {SEARCH_STRATEGY}")
    return state

def route_to_search_nodes(state: InterviewState):
    """
    Based on state['search_strategy'], decide which search nodes to enable.
    Return a list of node names to which 'ask_question' should connect.
    """
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
        targets += ["search_web", "search_wikipedia", "search_arxiv", "search_semantic_scholar"]
    
    return targets


def build_full_subgraph() -> StateGraph:
    """
    Build the subgraph with all potential search nodes.
    We'll disable or skip them at runtime based on search_strategy.
    """
    sg = StateGraph(InterviewState)
    
    # 1) Common nodes
    sg.add_node("ask_question", generate_question)
    sg.add_node("answer_question", generate_answer)
    sg.add_node("write_section", write_section)
    
    # 2) All possible search nodes
    sg.add_node("search_docs_rag", search_docs_rag)
    sg.add_node("search_web", search_web)
    sg.add_node("search_wikipedia", search_wikipedia)
    sg.add_node("search_arxiv", search_arxiv)
    sg.add_node("search_semantic_scholar", search_semantic_scholar)
    
    # 3) Connect edges via a small router function
    sg.add_conditional_edges("ask_question", route_to_search_nodes, 
        [
          "search_docs_rag",
          "search_web",
          "search_wikipedia",
          "search_arxiv",
          "search_semantic_scholar",
        ]
    )
    
    # Next, from each search node to answer_question (some libraries require separate edges).
    sg.add_edge("search_docs_rag", "answer_question")
    sg.add_edge("search_web", "answer_question")
    sg.add_edge("search_wikipedia", "answer_question")
    sg.add_edge("search_arxiv", "answer_question")
    sg.add_edge("search_semantic_scholar", "answer_question")
    
    # Then from answer_question -> write_section -> END
    sg.add_edge("answer_question", "write_section")
    sg.add_edge("write_section", END)
    sg.add_edge(START, "ask_question")
    
    return sg

def initiate_all_interviews(state: ResearchGraphState):
    print("Initiate_all_interviews")
    human_analyst_feedback = state.get('human_analyst_feedback')
    if human_analyst_feedback:
        return "create_analysts"
    else:
        topic = state["topic"]
        return [
            Send("conduct_interview", {
                "analyst": analyst,
                "messages": [HumanMessage(content=f"So you said you were writing an article on {topic}?")]
            })
            for analyst in state["analysts"]
        ]

def multi_agent_research_generation_persist_at_the_end(title, topic, max_analysts: int = 3, max_iterations: int = 2):
    print("Multi_agent_research_generation_persist_at_the_end")
    initial_state: ResearchGraphState = {
        "topic": topic,
        "max_analysts": max_analysts,
        "human_analyst_feedback": None,
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
        "search_strategy": None,
        "latex_report": "",
        "compiled_interview_graph": None,
        "cumulative_report": ""
    }
    params = {"configurable": {"thread_id": "1"}, "recursion_limit": 100}

    builder = StateGraph(ResearchGraphState)
    builder.add_node("plan_document", plan_document)
    builder.add_node("create_analysts", create_analysts)
    builder.add_node("choose_search_strategy", choose_search_strategy)    
    
    # Continue with the rest of the main graph nodes.
    builder.add_node("write_introduction", write_introduction)
    builder.add_node("write_report", write_report)
    builder.add_node("write_conclusion", write_conclusion)
    builder.add_node("write_resource_list", write_resource_list)
    builder.add_node("finalize_report", finalize_report)
    builder.add_node("self_critique", self_critique)
    builder.add_node("reset_for_iteration", reset_for_iteration)
    builder.add_node("generate_latex", generate_latex)

    interview_graph = build_full_subgraph().compile()
    builder.add_node("conduct_interview", interview_graph)
    
    builder.add_edge("plan_document", "create_analysts")
    builder.add_edge("create_analysts", "choose_search_strategy")
    builder.add_conditional_edges("choose_search_strategy", initiate_all_interviews, ["create_analysts", "conduct_interview"])
    builder.add_edge("conduct_interview", "write_introduction")

    builder.add_edge("write_introduction", "write_report")
    builder.add_edge("write_report", "write_conclusion")
    builder.add_edge("write_conclusion", "write_resource_list")
    builder.add_edge("write_resource_list", "finalize_report")
    builder.add_edge("finalize_report", "self_critique")
    builder.add_conditional_edges("self_critique", should_iterate, ["reset_for_iteration", "generate_latex"])
    builder.add_edge("reset_for_iteration", "plan_document")
    builder.add_edge("generate_latex", END)
    builder.add_edge(START, "plan_document")
    
    memory2 = MemorySaver()
    graph = builder.compile(checkpointer=memory2)

    for file in os.listdir("BIBLIO-TEST"):
        if not file == '.DS_Store':
            HumanLLM(agent_name="add_rag_doc_start", llmORchains_list=llm_list).add_rag_document(folder_path="BIBLIO-TEST", file_path=file)
    
    graph.invoke(initial_state, params)
    final_state = graph.get_state(params)
    report = final_state.values.get('final_report')
    latex_report, bibtex_report = extract_latex_and_bib_from_llm_output(final_state.values.get('latex_report', ""))
    report_folder = "./report_outputs"
    if not os.path.exists(report_folder):
        os.makedirs(report_folder)
    if latex_report:
        with open(f"{report_folder}/{topic}.tex", "w") as f:
            f.write(latex_report)
    if bibtex_report:
        with open(f"{report_folder}/{topic}.bib", "w") as f:
            f.write(bibtex_report)
    with open("images/graph_png.png", "wb") as f:
        f.write(graph.get_graph(xray=1).draw_mermaid_png())
    return report

# Invoke the multi-agent iterative workflow:
report = multi_agent_research_generation_persist_at_the_end(
    title="State of the art on Leading Edge Noise",
    topic = "Research assistant framework for State of the art on Leading Edge Noise",
    max_analysts=5,
    max_iterations=3  # Default number of iterations is 2.
)

print(report)
