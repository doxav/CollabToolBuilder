from . import helpers_demo
from .analysts import Analyst
from ..llm_utils import smart_print

from langgraph.graph import MessagesState
from langchain_core.messages import get_buffer_string
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

import operator
from typing import Annotated


class InterviewState(MessagesState):
    max_num_turns: int                              # Number of conversation turns
    context: Annotated[list, operator.add]          # Source docs
    analyst: Analyst                                # Analyst persona
    interview: str                                  # Interview transcript
    sections: list                                  # Collected sections for the report

def generate_question(state: InterviewState):
    print("Generate_question")
    if "question_instructions" not in state:
        state["question_instructions"] = (
            "You are an analyst interviewing an expert to learn about a specific topic.\n\n"
            "Your goal is to ask interesting and specific questions to gain deep insights.\n\n"
            "Here is your area of focus:\n{goals}\n\n"
            "Introduce yourself with a persona-appropriate name, then ask your question. "
            "Continue asking until you feel you have enough insight. "
            "Conclude with: 'Thank you so much for your help!'\n"
            "Remain in character throughout your response."
        )
    question_instructions = state["question_instructions"]
    analyst = state["analyst"]
    messages = state["messages"]
    system_message = question_instructions.format(goals=analyst.persona)
    question_answer = helpers_demo.llm_custom.invoke([SystemMessage(content=system_message)] + messages)
    smart_print(message=question_answer.content if isinstance(question_answer, AIMessage) else question_answer, agent_name="Generate Question", message_type="NEW inference result recieved", column_id=0, column_max=1)
    return {"messages": [question_answer]}

def search_web(state: InterviewState):
    print("Search_web")
    if "search_instructions" not in state:
        state["search_instructions"] = SystemMessage(content=(
            "You will be given a conversation between an analyst and an expert. "
            "Your goal is to generate a well-structured query for retrieval. "
            "Analyze the conversation—especially the final question—and convert it into a search query."
        ))
    search_query = helpers_demo.search_llm_query(state["search_instructions"], state["messages"])
    formatted_search_docs = helpers_demo.search_web_query_get(search_query)
    smart_print(message=formatted_search_docs, agent_name="Search Web", message_type="NEW inference result recieved", column_id=0, column_max=1)
    return {"context": [formatted_search_docs]}

def search_arxiv(state: InterviewState):
    print("Search_arxiv")

    # We assume that llm_custom, SearchQuery and SystemMessage are already defined elsewhere in the code.
    if not 'search_instructions' in state:
        state['search_instructions'] = SystemMessage(content=(
            "You are given a research topic. Your task is to generate a concise and effective search query "
            "optimized for the arXiv API. Make sure to include relevant keywords to retrieve the most pertinent scientific articles."
        ))
    search_query = helpers_demo.search_llm_query(state['search_instructions'], state["messages"])
    # Prepare parameters for the arXiv API.
    formatted_search_docs = helpers_demo.search_arxiv_query_get(search_query)

    smart_print(message=formatted_search_docs, agent_name="Search Arxiv", message_type="NEW inference result recieved", column_id=0, column_max=1)
    return {"context": [formatted_search_docs]}

def search_semantic_scholar(state: InterviewState):
    print("Search_semantic_scholar")
    # Set default search instructions if not already present
    if "search_instructions" not in state:
        state["search_instructions"] = SystemMessage(content=(
            "You are given a research topic. Your task is to generate a concise and effective search query "
            "optimized for the Semantic Scholar API. Ensure that the query includes relevant keywords to retrieve the most pertinent scientific articles."
        ))    
    # Generate the search query using the conversation messages
    search_query = helpers_demo.search_llm_query(state['search_instructions'], state["messages"])
    
    formatted_search_docs = helpers_demo.search_semantic_scholar_query_get(search_query)
    smart_print(message=formatted_search_docs, agent_name="Search Semantic Scholar", message_type="NEW inference result recieved", column_id=0, column_max=1)
    return {"context": [formatted_search_docs]}

def search_wikipedia(state: InterviewState):
    print("Search_wikipedia")
    if "search_instructions" not in state:
        state["search_instructions"] = SystemMessage(content=(
            "You will be given a conversation between an analyst and an expert. "
            "Your goal is to generate a well-structured query for retrieval. "
            "Analyze the conversation and convert the final question into a search query."
        ))
    search_query = helpers_demo.search_llm_query(state['search_instructions'], state['messages'])
    formatted_search_docs = helpers_demo.search_wikipedia_query_get(search_query)
    smart_print(message=formatted_search_docs, agent_name="Search Wikipedia", message_type="NEW inference result recieved", column_id=0, column_max=1)
    return {"context": [formatted_search_docs]}

def search_docs_rag(state: InterviewState):
    print("Search_docs_rag")
    if "search_instructions" not in state:
        state["search_instructions"] = SystemMessage(content=(
            "You are an analyst tasked with generating a search query for the RAG retrieval model. "
            "Analyze the conversation between the analyst and the expert and convert the final question into a search query."
        ))
    search_query = helpers_demo.search_llm_query(state["search_instructions"], state["messages"])
    formatted_search_docs = helpers_demo.search_docs_rag_get(search_query)
    smart_print(message=formatted_search_docs, agent_name="Search Docs RAG", message_type="NEW inference result recieved", column_id=0, column_max=1)
    return {"context": [formatted_search_docs]}

def generate_answer(state: InterviewState):
    print("Generate_answer")
    if "answer_instructions" not in state:
        state["answer_instructions"] = (
            "You are an expert being interviewed.\n\n"
            "Your focus is:\n{goals}\n\n"
            "Answer the following question using only the provided context:\n{context}\n\n"
            "Guidelines:\n"
            "1. Use only the information in the context.\n"
            "2. Do not introduce external information.\n"
            "3. Cite sources from the context using bracketed numbers (e.g., [1]).\n"
            "4. List all sources at the end as [1] Source1, [2] Source2, etc."
        )
    answer_instructions = state["answer_instructions"]
    analyst = state["analyst"]
    messages = state["messages"]
    context = state["context"]
    system_message = answer_instructions.format(goals=analyst.persona, context=context)
    answer_resp = helpers_demo.llm_custom.invoke([SystemMessage(content=system_message)] + messages)
    smart_print(message=answer_resp.content if isinstance(answer_resp, AIMessage) else answer_resp, agent_name="Generate Answer", message_type="NEW inference result recieved", column_id=0, column_max=1)
    return {"messages": [answer_resp]}

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
            "You are an expert technical writer.\n\n"
            "Your task is to create a concise section of a report from the provided source documents.\n\n"
            "Follow these steps:\n"
            "1. Analyze the provided documents (each begins with a <Document tag).\n"
            "2. Structure your section using Markdown with appropriate headers (e.g., ## for titles).\n"
            "3. Your section should include:\n"
            "   a. A compelling title (## header) based on the analyst’s focus: {focus}\n"
            "   b. A summary (### header) that provides context and highlights novel insights, including a numbered list of sources.\n"
            "   c. A sources list (### header) with full links or document paths. For any missing details, use 'Not Available'.\n"
            "Aim for a maximum of 400 words."
        )
    section_writer_instructions = state["section_writer_instructions"]
    analyst = state["analyst"]
    context = state["context"]
    system_message = section_writer_instructions.format(focus=analyst.description)
    sections = helpers_demo.llm_custom.invoke([
        SystemMessage(content=system_message),
        HumanMessage(content=f"Use this source to write your section: {context}")
    ])
    section_final = ""
    if isinstance(sections, list):
        for section in sections:
            section_final += section.content if isinstance(section, AIMessage) else section
    else:
        section_final = sections.content if isinstance(sections, AIMessage) else sections
    
    smart_print(message=section_final, agent_name="Write Section", message_type="NEW inference result recieved", column_id=0, column_max=1)
    return {"sections": [section_final]}
