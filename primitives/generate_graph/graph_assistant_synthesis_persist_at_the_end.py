from IPython.display import Image, display
from langgraph.graph import START, END, StateGraph
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, Field
from langgraph.graph import MessagesState
import operator
from typing import List, Annotated
from typing_extensions import TypedDict
from langchain_community.document_loaders import WikipediaLoader
from langchain_core.messages import get_buffer_string
from langgraph.constants import Send

from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

class Analyst(BaseModel):
    affiliation: str = Field(
        description="Primary affiliation of the analyst.",
    )
    name: str = Field(
        description="Name of the analyst."
    )
    role: str = Field(
        description="Role of the analyst in the context of the topic.",
    )
    description: str = Field(
        description="Description of the analyst focus, concerns, and motives.",
    )
    @property
    def persona(self) -> str:
        return f"Name: {self.name}\nRole: {self.role}\nAffiliation: {self.affiliation}\nDescription: {self.description}\n"

class Perspectives(BaseModel):
    analysts: List[Analyst] = Field(
        description="Comprehensive list of analysts with their roles and affiliations.",
    )

class GenerateAnalystsState(TypedDict):
    topic: str # Research topic
    max_analysts: int # Number of analysts
    human_analyst_feedback: str # Human feedback
    analysts: List[Analyst] # Analyst asking questions

def create_analysts(state: GenerateAnalystsState):
    """ Create analysts """
    if "analyst_instructions" not in state:
        state['analyst_instructions'] = """You are tasked with creating a set of AI analyst personas. Follow these instructions carefully:

        1. First, review the research topic:
        {topic}

        2. Examine any editorial feedback that has been optionally provided to guide creation of the analysts:
        {human_analyst_feedback}

        3. Determine the most interesting themes based upon documents and / or feedback above.

        4. Pick the top {max_analysts} themes.

        5. Assign one analyst to each theme.

        Respond with a JSON object containing a list of analysts, where each analyst has:
        - name: string
        - role: string
        - affiliation: string
        - description: string"""
    
    analyst_instructions = state["analyst_instructions"]

    topic=state['topic']
    max_analysts=state['max_analysts']
    human_analyst_feedback=state.get('human_analyst_feedback', '')

    # Enforce structured output
    structured_llm = llm.with_structured_output(Perspectives)

    # System message
    system_message = analyst_instructions.format(topic=topic,
                                                            human_analyst_feedback=human_analyst_feedback,
                                                            max_analysts=max_analysts)

    print(system_message)
    # Generate question
    analysts = structured_llm.invoke([SystemMessage(content=system_message)]+[HumanMessage(content="Generate the set of analysts.")])
    print(f"Type of analysts: {type(analysts)}")

    # Write the list of analysis to state
    return {"analysts": analysts.analysts}

def create_analysts(state: GenerateAnalystsState):
    """ Create analysts """
    if "analyst_instructions" not in state:
        state['analyst_instructions'] = """You are tasked with creating a set of AI analyst personas. Follow these instructions carefully:

        1. First, review the research topic:
        {topic}

        2. Examine any editorial feedback that has been optionally provided to guide creation of the analysts:
        {human_analyst_feedback}

        3. Determine the most interesting themes based upon documents and / or feedback above.

        4. Pick the top {max_analysts} themes.

        5. Assign one analyst to each theme.

        Respond with a JSON object containing a list of analysts, where each analyst has:
        - name: string
        - role: string
        - affiliation: string
        - description: string"""
    
    analyst_instructions = state["analyst_instructions"]
    topic = state['topic']
    max_analysts = state['max_analysts']
    human_analyst_feedback = state.get('human_analyst_feedback', '')

    # Enforce structured output
    structured_llm = llm.with_structured_output(Perspectives)

    # System message
    system_message = analyst_instructions.format(
        topic=topic,
        human_analyst_feedback=human_analyst_feedback,
        max_analysts=max_analysts
    )

    try:
        # Generate analysts
        analysts_response = structured_llm.invoke([
            SystemMessage(content=system_message),
            HumanMessage(content="Generate the set of analysts.")
        ])

        # Handle different possible return types
        if isinstance(analysts_response, Perspectives):
            # If it's already a Perspectives object
            generated_analysts = analysts_response.analysts
        elif isinstance(analysts_response, dict):
            # If it's a dict, try to extract analysts
            generated_analysts = analysts_response.get('analysts', [])
        elif hasattr(analysts_response, 'analysts'):
            # If it has an analysts attribute
            generated_analysts = analysts_response.analysts
        else:
            # Fallback to default analysts
            generated_analysts = [
                Analyst(
                    name=f"Analyst {i+1}", 
                    role=f"Research Specialist {i+1}", 
                    affiliation="Research Institute", 
                    description=f"Analyzing aspects of {topic}"
                ) for i in range(max_analysts)
            ]

        # Ensure all are Analyst objects
        generated_analysts = [
            Analyst(**a.dict() if hasattr(a, 'dict') else a) 
            for a in generated_analysts
        ]

        return {"analysts": generated_analysts}

    except Exception as e:
        print(f"Error generating analysts: {e}")
        # Fallback to default analysts
        default_analysts = [
            Analyst(
                name=f"Analyst {i+1}", 
                role=f"Research Specialist {i+1}", 
                affiliation="Research Institute", 
                description=f"Analyzing aspects of {topic}"
            ) for i in range(max_analysts)
        ]
        return {"analysts": default_analysts}

def human_feedback(state: GenerateAnalystsState):
    """ No-op node that should be interrupted on """
    pass

def should_continue(state: GenerateAnalystsState):
    """ Return the next node to execute """

    # Check if human feedback
    human_analyst_feedback=state.get('human_analyst_feedback', None)
    if human_analyst_feedback:
        return "create_analysts"

    # Otherwise end
    return END

class InterviewState(MessagesState):
    max_num_turns: int # Number turns of conversation
    context: Annotated[list, operator.add] # Source docs
    analyst: Analyst # Analyst asking questions
    interview: str # Interview transcript
    sections: list # Final key we duplicate in outer state for Send() API

class SearchQuery(BaseModel):
    search_query: str = Field(None, description="Search query for retrieval.")

def generate_question(state: InterviewState):
    """ Node to generate a question """

    # Check if question_instructions is provided by state
    if "question_instructions" not in state:
        state["question_instructions"] = """You are an analyst tasked with interviewing an expert to learn about a specific topic.

        Your goal is boil down to interesting and specific insights related to your topic.

        1. Interesting: Insights that people will find surprising or non-obvious.

        2. Specific: Insights that avoid generalities and include specific examples from the expert.

        Here is your topic of focus and set of goals: {goals}

        Begin by introducing yourself using a name that fits your persona, and then ask your question.

        Continue to ask questions to drill down and refine your understanding of the topic.

        When you are satisfied with your understanding, complete the interview with: "Thank you so much for your help!"

        Remember to stay in character throughout your response, reflecting the persona and goals provided to you."""

    question_instructions = state["question_instructions"]

    # Get state
    analyst = state["analyst"]
    messages = state["messages"]

    # Generate question
    system_message = question_instructions.format(goals=analyst.persona)
    question = llm.invoke([SystemMessage(content=system_message)]+messages)

    # Write messages to state
    return {"messages": [question]}

def search_web(state: InterviewState):
    """ Retrieve academic papers from OpenAlex """
    import requests
    OPENALEX_API_URL = "https://api.openalex.org/works"

    # Search query
    structured_llm = llm.with_structured_output(SearchQuery)
    if "search_instructions" not in state:
        state["search_instructions"] = SystemMessage(content=f"""You will be given a conversation between an analyst and an expert.

        Your goal is to generate a well-structured query for use in retrieval and / or web-search related to the conversation.

        First, analyze the full conversation.

        Pay particular attention to the final question posed by the analyst.

        Convert this final question into a well-structured web search query""")
    search_instructions = state["search_instructions"]

    search_query = structured_llm.invoke([search_instructions] + state['messages'])

    # Construct the OpenAlex API query parameters
    params = {
        "search": search_query.search_query,
        "filter": "is_paratext:false",  # Exclude non-research content
        "sort": "relevance_score:desc",
        "per_page": 5  # Limit to top 5 relevant results
    }

    # Perform the request to OpenAlex
    response = requests.get(OPENALEX_API_URL, params=params)
    search_docs = []
    
    if response.status_code == 200:
        data = response.json()
        for result in data.get("results", []):
            search_docs.append({
                "title": result.get("title", "Unknown Title"),
                "authors": ", ".join([auth["author"]["display_name"] for auth in result.get("authorships", [])]),
                "abstract": result.get("abstract", "No abstract available"),
                "url": result.get("id", "Unknown URL")
            })
    else:
        print(f"Error retrieving data from OpenAlex: {response.status_code}")

    # Format results
    formatted_search_docs = "\n\n---\n\n".join(
        [
            f'<Document title="{doc["title"]}" href="{doc["url"]}">\nAuthors: {doc["authors"]}\n\nAbstract: {doc["abstract"]}\n</Document>'
            for doc in search_docs
        ]
    )

    return {"context": [formatted_search_docs]}

def search_wikipedia(state: InterviewState):
    """ Retrieve docs from wikipedia """
    # Check if search_instructions is provided by state
    if "search_instructions" not in state:
        state["search_instructions"] = SystemMessage(content=f"""You will be given a conversation between an analyst and an expert.
        Your goal is to generate a well-structured query for use in retrieval and / or web-search related to the conversation.
        First, analyze the full conversation.
        Pay particular attention to the final question posed by the analyst.
        Convert this final question into a well-structured web search query""")

    search_instructions = state["search_instructions"]

    # Search query
    structured_llm = llm.with_structured_output(SearchQuery)
    search_query = structured_llm.invoke([search_instructions]+state['messages'])

    # Search
    search_docs = WikipediaLoader(query=search_query.search_query,
                                  load_max_docs=2).load()

     # Format
    formatted_search_docs = "\n\n---\n\n".join(
        [
            f'<Document source="{doc.metadata["source"]}" page="{doc.metadata.get("page", "")}"/>\n{doc.page_content}\n</Document>'
            for doc in search_docs
        ]
    )

    return {"context": [formatted_search_docs]}

def generate_answer(state: InterviewState):
    """ Node to answer a question """
    # Check if answer_instructions is provided by state
    if "answer_instructions" not in state:
        state["answer_instructions"] = """You are an expert being interviewed by an analyst.

        Here is analyst area of focus: {goals}.

        You goal is to answer a question posed by the interviewer.

        To answer question, use this context:

        {context}

        When answering questions, follow these guidelines:

        1. Use only the information provided in the context.

        2. Do not introduce external information or make assumptions beyond what is explicitly stated in the context.

        3. The context contain sources at the topic of each individual document.

        4. Include these sources your answer next to any relevant statements. For example, for source # 1 use [1].

        5. List your sources in order at the bottom of your answer. [1] Source 1, [2] Source 2, etc

        6. If the source is: <Document source="assistant/docs/llama3_1.pdf" page="7"/>' then just list:

        [1] assistant/docs/llama3_1.pdf, page 7

        And skip the addition of the brackets as well as the Document source preamble in your citation."""

    answer_instructions = state["answer_instructions"]

    # Get state
    analyst = state["analyst"]
    messages = state["messages"]
    context = state["context"]

    # Answer question
    system_message = answer_instructions.format(goals=analyst.persona, context=context)
    answer = llm.invoke([SystemMessage(content=system_message)]+messages)

    # Name the message as coming from the expert
    answer.name = "expert"

    # Append it to state
    return {"messages": [answer]}

def save_interview(state: InterviewState):

    """ Save interviews """

    # Get messages
    messages = state["messages"]

    # Convert interview to a string
    interview = get_buffer_string(messages)

    # Save to interviews key
    return {"interview": interview}

def route_messages(state: InterviewState,
                   name: str = "expert"):

    """ Route between question and answer """

    # Get messages
    messages = state["messages"]
    max_num_turns = state.get('max_num_turns',2)

    # Check the number of expert answers
    num_responses = len(
        [m for m in messages if isinstance(m, AIMessage) and m.name == name]
    )

    # End if expert has answered more than the max turns
    if num_responses >= max_num_turns:
        return 'save_interview'

    # This router is run after each question - answer pair
    # Get the last question asked to check if it signals the end of discussion
    last_question = messages[-2]

    if "Thank you so much for your help" in last_question.content:
        return 'save_interview'
    return "ask_question"

def write_section(state: InterviewState):
    """ Node to answer a question """
    # Check if section_writer_instructions is provided by state
    if "section_writer_instructions" not in state:
        state["section_writer_instructions"] = """You are an expert technical writer.

        Your task is to create a short, easily digestible section of a report based on a set of source documents.

        1. Analyze the content of the source documents:
        - The name of each source document is at the start of the document, with the <Document tag.

        2. Create a report structure using markdown formatting:
        - Use ## for the section title
        - Use ### for sub-section headers

        3. Write the report following this structure:
        a. Title (## header)
        b. Summary (### header)
        c. Sources (### header)

        4. Make your title engaging based upon the focus area of the analyst:
        {focus}

        5. For the summary section:
        - Set up summary with general background / context related to the focus area of the analyst
        - Emphasize what is novel, interesting, or surprising about insights gathered from the interview
        - Create a numbered list of source documents, as you use them
        - Do not mention the names of interviewers or experts
        - Aim for approximately 400 words maximum
        - Use numbered sources in your report (e.g., [1], [2]) based on information from source documents

        6. In the Sources section:
        - Include all sources used in your report
        - Provide full links to relevant websites or specific document paths
        - Separate each source by a newline. Use two spaces at the end of each line to create a newline in Markdown.
        - It will look like:

        ### Sources
        [1] Link or Document name
        [2] Link or Document name

        7. Be sure to combine sources. For example this is not correct:

        [3] https://ai.meta.com/blog/meta-llama-3-1/
        [4] https://ai.meta.com/blog/meta-llama-3-1/

        There should be no redundant sources. It should simply be:

        [3] https://ai.meta.com/blog/meta-llama-3-1/

        8. Final review:
        - Ensure the report follows the required structure
        - Include no preamble before the title of the report
        - Check that all guidelines have been followed"""

    section_writer_instructions = state["section_writer_instructions"]

    # Get state
    interview = state["interview"]
    context = state["context"]
    analyst = state["analyst"]

    # Write section using either the gathered source docs from interview (context) or the interview itself (interview)
    system_message = section_writer_instructions.format(focus=analyst.description)
    section = llm.invoke([SystemMessage(content=system_message)]+[HumanMessage(content=f"Use this source to write your section: {context}")])

    # Append it to state
    return {"sections": [section.content]}

class ResearchGraphState(TypedDict):
    topic: str # Research topic
    max_analysts: int # Number of analysts
    human_analyst_feedback: str # Human feedback
    analysts: List[Analyst] # Analyst asking questions
    sections: Annotated[list, operator.add] # Send() API key
    introduction: str # Introduction for the final report
    content: str # Content for the final report
    conclusion: str # Conclusion for the final report
    final_report: str # Final report

def initiate_all_interviews(state: ResearchGraphState):
    """ This is the "map" step where we run each interview sub-graph using Send API """

    # Check if human feedback
    human_analyst_feedback=state.get('human_analyst_feedback')
    if human_analyst_feedback:
        # Return to create_analysts
        return "create_analysts"

    # Otherwise kick off interviews in parallel via Send() API
    else:
        topic = state["topic"]
        return [Send("conduct_interview", {"analyst": analyst,
                                           "messages": [HumanMessage(
                                               content=f"So you said you were writing an article on {topic}?"
                                           )
                                                       ]}) for analyst in state["analysts"]]

def write_report(state: ResearchGraphState):
    # Check if report_writer_instructions is provided by state
    if "report_writer_instructions" not in state:
        state['report_writer_instructions'] = """You are a technical writer creating a report on this overall topic:

{topic}

You have a team of analysts. Each analyst has done two things:

1. They conducted an interview with an expert on a specific sub-topic.
2. They write up their finding into a memo.

Your task:

1. You will be given a collection of memos from your analysts.
2. Think carefully about the insights from each memo.
3. Consolidate these into a crisp overall summary that ties together the central ideas from all of the memos.
4. Summarize the central points in each memo into a cohesive single narrative.

To format your report:

1. Use markdown formatting.
2. Include no pre-amble for the report.
3. Use no sub-heading.
4. Start your report with a single title header: ## Insights
5. Do not mention any analyst names in your report.
6. Preserve any citations in the memos, which will be annotated in brackets, for example [1] or [2].
7. Create a final, consolidated list of sources and add to a Sources section with the `## Sources` header.
8. List your sources in order and do not repeat.

[1] Source 1
[2] Source 2

Here are the memos from your analysts to build your report from:

{context}"""

    report_writer_instructions = state["report_writer_instructions"]

    # Full set of sections
    sections = state["sections"]
    topic = state["topic"]

    # Concat all sections together
    formatted_str_sections = "\n\n".join([f"{section}" for section in sections])

    # Summarize the sections into a final report
    system_message = report_writer_instructions.format(topic=topic, context=formatted_str_sections)
    report = llm.invoke([SystemMessage(content=system_message)]+[HumanMessage(content=f"Write a report based upon these memos.")])
    return {"content": report.content}

def write_introduction(state: ResearchGraphState):
    # Check if intro_conclusion_instructions is provided by state
    if "intro_conclusion_instructions" not in state:
        state['intro_conclusion_instructions'] = """You are a technical writer finishing a report on {topic}

You will be given all of the sections of the report.

You job is to write a crisp and compelling introduction or conclusion section.

The user will instruct you whether to write the introduction or conclusion.

Include no pre-amble for either section.

Target around 100 words, crisply previewing (for introduction) or recapping (for conclusion) all of the sections of the report.

Use markdown formatting.

For your introduction, create a compelling title and use the # header for the title.

For your introduction, use ## Introduction as the section header.

For your conclusion, use ## Conclusion as the section header.

Here are the sections to reflect on for writing: {formatted_str_sections}"""

    intro_conclusion_instructions = state["intro_conclusion_instructions"]

    # Full set of sections
    sections = state["sections"]
    topic = state["topic"]

    # Concat all sections together
    formatted_str_sections = "\n\n".join([f"{section}" for section in sections])

    # Summarize the sections into a final report

    instructions = intro_conclusion_instructions.format(topic=topic, formatted_str_sections=formatted_str_sections)
    intro = llm.invoke([instructions]+[HumanMessage(content=f"Write the report introduction")])
    return {"introduction": intro.content}

def write_conclusion(state: ResearchGraphState):
    # Check if intro_conclusion_instructions is provided by state
    if "intro_conclusion_instructions" not in state:
        state['intro_conclusion_instructions'] = """You are a technical writer finishing a report on {topic}

You will be given all of the sections of the report.

You job is to write a crisp and compelling introduction or conclusion section.

The user will instruct you whether to write the introduction or conclusion.

Include no pre-amble for either section.

Target around 100 words, crisply previewing (for introduction) or recapping (for conclusion) all of the sections of the report.

Use markdown formatting.

For your introduction, create a compelling title and use the # header for the title.

For your introduction, use ## Introduction as the section header.

For your conclusion, use ## Conclusion as the section header.

Here are the sections to reflect on for writing: {formatted_str_sections}"""

    intro_conclusion_instructions = state["intro_conclusion_instructions"]        
    # Full set of sections
    sections = state["sections"]
    topic = state["topic"]

    # Concat all sections together
    formatted_str_sections = "\n\n".join([f"{section}" for section in sections])

    # Summarize the sections into a final report

    instructions = intro_conclusion_instructions.format(topic=topic, formatted_str_sections=formatted_str_sections)
    conclusion = llm.invoke([instructions]+[HumanMessage(content=f"Write the report conclusion")])
    return {"conclusion": conclusion.content}

def finalize_report(state: ResearchGraphState):
    """ This is the "reduce" step where we gather all the sections, combine them, and reflect on them to write the intro/conclusion """
    if not state:
        return {"final_report": ""}
        
    # Save full final report
    content = state.get("content", "")
    introduction = state.get("introduction", "")
    conclusion = state.get("conclusion", "")
    
    if content.startswith("## Insights"):
        content = content.strip("## Insights")
    
    sources = None
    if "## Sources" in content:
        try:
            content, sources = content.split("\n## Sources\n")
        except:
            sources = None

    final_report = ""
    if introduction:
        final_report += introduction + "\n\n---\n\n"
    if content:
        final_report += content
    if conclusion:
        final_report += "\n\n---\n\n" + conclusion
    if sources:
        final_report += "\n\n## Sources\n" + sources

    return {"final_report": final_report}

def persist_final_report_in_bot(bot, final_report):
    """
    Converts the final Markdown report into document sections,
    and extracts sources to store in bot.add_or_update_results_in_resources.
    """
    # Handle None or empty input
    if final_report is None:
        final_report = {"final_report": ""}  # Create empty report instead of raising error
        
    # Handle dictionary input
    if isinstance(final_report, dict):
        final_report = final_report.get("final_report", "")
    
    # Ensure we have a string
    final_report = str(final_report)
    
    # If empty report, create minimal section
    if not final_report.strip():
        bot.create_and_add_section_then_return_id(
            title="Empty Report",
            content="No content was generated."
        )
        return

    import re
    
    # The rest of the function remains the same...
    section_pattern = re.compile(r"(##\s+.+?)(?=##\s|$)", re.DOTALL)
    
    grand_titre_match = re.search(r"^#\s+(.*)", final_report)
    if grand_titre_match:
        grand_titre = grand_titre_match.group(1).strip()
        bot.create_and_add_section_then_return_id(
            title=grand_titre,
            content=f"(Main Title)\n\n{grand_titre}"
        )
    
    sections = section_pattern.findall(final_report)

    for section_md in sections:
        lines = section_md.split('\n', 1)
        if len(lines) == 2:
            section_title_line, section_body = lines
        else:
            section_title_line = lines[0]
            section_body = ""

        section_title = section_title_line.replace("##", "").strip()
        bot.create_and_add_section_then_return_id(title=section_title, content=section_body)

    sources_block_match = re.search(r"(##\s+Sources.*)", final_report, re.IGNORECASE | re.DOTALL)
    if sources_block_match:
        sources_block = sources_block_match.group(1)
        
        lines = sources_block.split('\n')
        resources_to_add = []
        for line in lines:
            m = re.match(r"\[(\d+)\]\s+(.*)", line.strip())
            if m:
                index = m.group(1)
                url_or_name = m.group(2)
                resource_item = {
                    "name": f"Source_{index}",
                    "content": {"url": url_or_name},
                    "metadatas": {"index": index}
                }
                resources_to_add.append(resource_item)
        
        if resources_to_add:
            bot.add_or_update_results_in_resources(
                results=resources_to_add, 
                metadatas_to_add={"type": "reference"},
                store_linked_document_content=False
            )

def improve_multi_agent_research_generation(bot, max_analysts: int = 3):
    """
    Generate a full research report using a multi-agent LangGraph workflow:  
    Create a list of analysts, Conduct interviews, Write sections (no plan yet and sections except introduction and conclusion are merged into one section), 
    Write report, Write introduction, Write conclusion, Finalize report.
    Persist the document in the bot object through agent action.
    
    Args:
        bot: The bot object with the necessary methods.
        max_analysts (int): OPTIONAL (default: 3) - The number of analysts to generate.
    Returns:
        str: The final markdown research report.
    """
    title, topic = bot.document.title, bot.document.context
    # Create initial state with topic and max_analysts
    initial_state: GenerateAnalystsState = { "topic": topic, "max_analysts": max_analysts, "human_analyst_feedback": None, "analysts": []}

    # Recreate the interview graph within the function
    interview_builder = StateGraph(InterviewState)
    interview_builder.add_node("ask_question", generate_question)
    interview_builder.add_node("search_web", search_web)
    interview_builder.add_node("search_wikipedia", search_wikipedia)
    interview_builder.add_node("answer_question", generate_answer)
    interview_builder.add_node("save_interview", save_interview)
    interview_builder.add_node("write_section", write_section)

    # Flow
    interview_builder.add_edge(START, "ask_question")
    interview_builder.add_edge("ask_question", "search_web")
    interview_builder.add_edge("ask_question", "search_wikipedia")
    interview_builder.add_edge("search_web", "answer_question")
    interview_builder.add_edge("search_wikipedia", "answer_question")
    interview_builder.add_conditional_edges("answer_question", route_messages,['ask_question','save_interview'])
    interview_builder.add_edge("save_interview", "write_section")
    interview_builder.add_edge("write_section", END)

    # Compile interview graph
    memory = MemorySaver()
    interview_graph = interview_builder.compile(checkpointer=memory).with_config(run_name="Conduct Interviews")

    # Set up the thread configuration
    thread = {"configurable": {"thread_id": "1"}}

    # Compile the research graph
    builder = StateGraph(ResearchGraphState)
    builder.add_node("create_analysts", create_analysts)
    builder.add_node("human_feedback", human_feedback)
    
    # Use the interview_graph directly, without .compile()
    builder.add_node("conduct_interview", interview_graph)
    builder.add_node("write_report", write_report)
    builder.add_node("write_introduction", write_introduction)
    builder.add_node("write_conclusion", write_conclusion)
    builder.add_node("finalize_report", finalize_report)

    # Logic
    builder.add_edge(START, "create_analysts")
    builder.add_edge("create_analysts", "human_feedback")
    builder.add_conditional_edges("human_feedback", initiate_all_interviews, ["create_analysts", "conduct_interview"])
    builder.add_edge("conduct_interview", "write_introduction")
    builder.add_edge("conduct_interview", "write_report")
    builder.add_edge("conduct_interview", "write_conclusion")
    builder.add_edge(["write_conclusion", "write_report", "write_introduction"], "finalize_report")
    builder.add_edge("finalize_report", END)

    # Compile the graph
    memory2 = MemorySaver()
    graph = builder.compile(checkpointer=memory2)
    #display(Image(graph.get_graph(xray=1).draw_mermaid_png()))

    # Invoke the graph
    graph.invoke(initial_state, thread)

    # Retrieve the final report
    final_state = graph.get_state(thread)
    report = final_state.values.get('final_report')

    persist_final_report_in_bot(bot, report)
    return report
