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

from langchain_core.messages import HumanMessage
from langgraph.graph import START, END, StateGraph
from langgraph.checkpoint.memory import MemorySaver

import operator
from typing import List, Annotated
from langgraph.constants import Send
from typing_extensions import TypedDict

from config import embedding_function, reset_db_indices
from utils.llm_utils import HumanLLMMonitor, smart_input
from utils.graph_utils.analysts import Analyst, create_analysts
from utils.graph_utils.helpers_demo import remove_think_tags, extract_latex_and_bib_from_llm_output
from utils.graph_utils.interview import InterviewState, generate_question, generate_answer, write_section, search_docs_rag, search_web, search_wikipedia, search_arxiv, search_semantic_scholar

HumanLLMMonitor.use_websocket = True
planner = HumanLLMMonitor(agent_name="Planner")
question = HumanLLMMonitor(agent_name="Generate Questions")
answer = HumanLLMMonitor(agent_name="Generate Aswers")
writerr = HumanLLMMonitor(agent_name="Write Report")
writers = HumanLLMMonitor(agent_name="Write Section")
writeri = HumanLLMMonitor(agent_name="Write Introduction")
writerrl = HumanLLMMonitor(agent_name="Write Resource List")
writerc = HumanLLMMonitor(agent_name="Write Conclusion")
critic = HumanLLMMonitor(agent_name="Self Critic")
latex_gen = HumanLLMMonitor(agent_name="Generate Latex")
HumanLLMMonitor._check_and_init_vector_db(embedding_function=embedding_function, reset_db_indices=reset_db_indices)

### -------------------------------
# RESEARCH REPORT NODES (INCLUDING PLAN, RESOURCE LIST, AND SELF-CRITIQUE)
### -------------------------------

class SearchStrategy:
    def register_nodes(self, graph_builder: StateGraph):
        raise NotImplementedError("Subclasses must implement register_nodes")

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

def plan_document(state: ResearchGraphState):
    """Plan Agent: Create a detailed plan for the research document."""
    print("Plan_document")
    filepath = "plan_document"
    prompt_file = f"./prompts/{filepath}.txt"
    if not os.path.exists(prompt_file):
        with open(prompt_file, "w") as f:
            f.write(
                "You are a planning assistant. Your task is to create a detailed plan for a research document. "
                "In the user message, you will receive a value labeled 'TOPIC' which represents the research topic. "
                "Please outline the key sections (for example, Introduction, Main Content, Conclusion) and provide a brief description for each section. "
                "Ensure that the plan is clear, detailed, and well-structured."
            )
    # All dynamic information is passed in the user_message.
    user_message = f"TOPIC: <<< {state['topic']} >>>"
    plan = planner.CallHumanLLM(
        system_prompt_template=filepath,
        user_message=user_message,
        stream_output=False,
        return_message_content_only=True,
        use_default_llm=False
    )[0]
    plan_text = plan.content if hasattr(plan, "content") else plan
    plan_text = remove_think_tags(plan_text)
    print(f"plan generated : {plan_text}")
    return {"plan": plan_text}

def write_report(state: ResearchGraphState):
    """Write Agent: Consolidate the memos into a coherent report (without conclusion)."""
    print("Write_report")
    filepath = "write_report"
    prompt_file = f"./prompts/{filepath}.txt"
    if not os.path.exists(prompt_file):
        with open(prompt_file, "w") as f:
            f.write(
                "You are a technical writer tasked with creating a report based on provided inputs. "
                "In the user message, you will receive the following values:\n"
                "  - 'TOPIC': the research topic,\n"
                "  - 'PLAN': the report plan,\n"
                "  - 'MEMOS': a collection of memos from analysts.\n\n"
                "Your job is to consolidate these memos into a cohesive narrative of insights, following the provided plan. "
                "Do not include concluding remarks, as they will be generated separately. "
                "Format the report in Markdown with a title header '## Insights'. "
                "Avoid mentioning analyst names and preserve any citations present in the memos."
            )
    sections = state["sections"]
    if hasattr(sections, "content"):
        formatted_str_sections = sections.content
    elif isinstance(sections, list):
        formatted_str_sections = "\n\n".join(sections)
    else:
        formatted_str_sections = sections

    user_message = (
        f"TOPIC: <<< {state['topic']} >>>\n"
        f"PLAN: <<< {state['plan']} >>>\n"
        f"MEMOS: <<< {formatted_str_sections} >>>"
    )
    report = writerr.CallHumanLLM(
        system_prompt_template=filepath,
        user_message=user_message,
        stream_output=False,
        return_message_content_only=True,
        use_default_llm=False
    )[0]
    report_text = report.content if hasattr(report, "content") else report
    report_text = remove_think_tags(report_text)
    print(f"report generated : {report_text}")
    return {"content": report_text}

def write_introduction(state: ResearchGraphState):
    """Introduction Agent: Write a concise introduction for the report."""
    print("Write_introduction")
    filepath = "write_introduction"
    prompt_file = f"./prompts/{filepath}.txt"
    if not os.path.exists(prompt_file):
        with open(prompt_file, "w") as f:
            f.write(
                "You are a technical writer tasked with writing the introduction for a research report. "
                "In the user message, you will receive the following values:\n"
                "  - 'TOPIC': the research topic,\n"
                "  - 'PLAN': the report plan,\n"
                "  - 'SECTIONS': the sections of the report.\n\n"
                "Your task is to craft a crisp and compelling introduction. Begin with a title using a '#' header, followed by a section header '## Introduction'. "
                "The introduction should be approximately 100 words and clearly preview the report sections."
            )
    sections = state["sections"]
    if hasattr(sections, "content"):
        formatted_str_sections = sections.content
    elif isinstance(sections, list):
        formatted_str_sections = "\n\n".join(sections)
    else:
        formatted_str_sections = sections

    user_message = (
        f"TOPIC: <<< {state['topic']} >>>\n"
        f"PLAN: <<< {state['plan']} >>>\n"
        f"SECTIONS: <<< {formatted_str_sections} >>>\n"
        f"TASK: Write the report introduction."
    )
    intro = writeri.CallHumanLLM(
        system_prompt_template=filepath,
        user_message=user_message,
        stream_output=False,
        return_message_content_only=True,
        use_default_llm=False
    )[0]
    intro_text = intro.content if hasattr(intro, "content") else intro
    intro_text = remove_think_tags(intro_text)
    print(f"introduction generated : {intro_text}")
    return {"introduction": intro_text}

def write_conclusion(state: ResearchGraphState):
    filepath = "write_conclusion"
    prompt_file = f"./prompts/{filepath}.txt"
    if not os.path.exists(prompt_file):
        with open(prompt_file, "w") as f:
            f.write("You are a technical writer tasked with finalizing a technical report. You excel at distilling complex topics "
        "into clear, concise conclusions. Your job is to write a crisp and compelling conclusion that starts with '## Conclusion', "
        "recapping the key insights from the report in approximately 100 words. Use best practices and clarity in your writing.")
    topic, plan = state["topic"], state["plan"]
    sections = state["sections"]
    formatted_str_sections = sections.content if hasattr(sections, "content") else "\n\n".join(sections)

    user_message = (
        f"- TOPIC: <<< {topic} >>>\n"
        f"- PLAN: <<< {plan} >>>\n"
        f"- SECTIONS: <<< {formatted_str_sections} >>>"
    )

    conclusion = writerc.CallHumanLLM(
        system_prompt_template=filepath,
        user_message=user_message,
        stream_output=False,
        return_message_content_only=True,
        use_default_llm=False
    )[0]
    conclusion = remove_think_tags(conclusion)
    print(f"conclusion generated : {conclusion}")
    return {"conclusion": conclusion}

def write_resource_list(state: ResearchGraphState):
    """
    Resource List Agent: Extract and consolidate the citations/references from the report.
    Format in Markdown under the header '## Sources'.
    """
    print("Write_resource_list")
    filepath = "write_resource_list"
    prompt_file = f"./prompts/{filepath}.txt"
    if not os.path.exists(prompt_file):
        with open(prompt_file, "w") as f:
            f.write(
                "You are an expert technical writer. Your task is to extract and consolidate all resource citations from report sections. "
                "In the user message, you will receive a value labeled 'SECTIONS' which contains the report sections. "
                "The citations may appear as inline references (e.g., [Source 1], [Source 2]). "
                "Generate a consolidated resource list in Markdown under the header '## Sources', with each source on a new line in the format: '[n] <source details>'. "
                "If any details are missing, indicate them as 'Not Available'."
            )
    sections = state["sections"]
    if hasattr(sections, "content"):
        formatted_str_sections = sections.content
    elif isinstance(sections, list):
        formatted_str_sections = "\n\n".join(sections)
    else:
        formatted_str_sections = sections

    user_message = f"SECTIONS: <<< {formatted_str_sections} >>>"
    resource_list = writerrl.CallHumanLLM(
        system_prompt_template=filepath,
        user_message=user_message,
        stream_output=False,
        return_message_content_only=True,
        use_default_llm=False
    )[0]
    resource_list_text = resource_list.content if hasattr(resource_list, "content") else resource_list
    resource_list_text = remove_think_tags(resource_list_text)
    print(f"resources generated : {resource_list_text}")
    return {"resource_list": resource_list_text}

def finalize_report(state: ResearchGraphState):
    """Combine the introduction, main content, conclusion, and resource list into the final report."""
    print("Finalize_report")
    if not state:
        return {"final_report": ""}
    content = state.get("content", "")
    introduction = state.get("introduction", "")
    conclusion = state.get("conclusion", "")
    resource_list = state.get("resource_list", "")
    if content.startswith("## Insights"):
        content = content.lstrip("## Insights").strip()
    final_report = ""
    if introduction:
        final_report += introduction + "\n\n---\n\n"
    if content:
        final_report += content
    if conclusion and "Conclusion" not in content:
        final_report += "\n\n---\n\n" + conclusion
    if resource_list and resource_list.strip():
        final_report += "\n\n" + resource_list
    return {"final_report": final_report}

def self_critique(state: ResearchGraphState):
    """
    Critic Agent: Analyze the final report and generate an updated plan.
    Also increments the iteration counter.
    """
    print("Self_critique")
    filepath = "self_critique"
    prompt_file = f"./prompts/{filepath}.txt"
    if not os.path.exists(prompt_file):
        with open(prompt_file, "w") as f:
            f.write(
                "You are an expert critic. Your task is to review a research report and provide constructive criticism with suggestions for improvement. "
                "In the user message, you will receive a value labeled 'REPORT' which contains the current research report. "
                "After analyzing the report, generate an updated plan that addresses the identified issues. "
                "Return only the updated plan."
            )
    current_report = state.get("final_report", "")
    user_message = f"REPORT: <<< {current_report} >>>\nTASK: Critique the report and update the plan."
    critique_response = critic.CallHumanLLM(
        system_prompt_template=filepath,
        user_message=user_message,
        stream_output=False,
        return_message_content_only=True,
        use_default_llm=False
    )[0]
    updated_plan = critique_response.content if hasattr(critique_response, "content") else critique_response
    updated_plan = remove_think_tags(updated_plan)
    print(f"Critique : {updated_plan}")
    state["iteration"] = state.get("iteration", 0) + 1
    return {"plan": updated_plan, "iteration": state["iteration"]}

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
    state["final_report"] = ""
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
    latex_report = latex_gen.CallHumanLLM(system_prompt_template=filepath, user_message=user_message, stream_output=False, return_message_content_only=True, use_default_llm=False)[0]
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
    SEARCH_STRATEGY = smart_input("What searching strategy do you want to use? (Online, Offline, Both): ", column_id=0, column_max=1, optional=False).lower()
    
    # Validate the choice and default to "both" if unrecognized.
    SEARCH_STRATEGY = "both" if SEARCH_STRATEGY not in ["online", "offline", "both"] else SEARCH_STRATEGY
    
    # Store the choice as a string to ensure serializability.
    print(f"Search strategy chosen: {SEARCH_STRATEGY}")
    return state

def route_to_search_nodes(state: InterviewState):
    """
    Based on state['search_strategy'], decide which search nodes to enable.
    Return a list of node names to which 'ask_question' should connect.
    """
    global SEARCH_STRATEGY
    targets = []
    
    if SEARCH_STRATEGY == "offline":
        # Only offline search
        targets = ["search_docs_rag"]
    elif SEARCH_STRATEGY == "online":
        # Only online
        targets = ["search_web", "search_wikipedia", "search_arxiv", "search_semantic_scholar"]
    else:
        # "both" or unknown => all nodes
        targets = ["search_docs_rag", "search_web", "search_wikipedia", "search_arxiv", "search_semantic_scholar"]
    
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
        "compiled_interview_graph": None  # New field initialized.
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
