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
from typing import List
from typing_extensions import TypedDict
from pydantic import BaseModel, Field
from IPython.display import Image, display
from langgraph.graph import START, END, StateGraph
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
import operator
from typing import  Annotated
from langgraph.graph import MessagesState
import datetime
import os
from langchain_openai import ChatOpenAI
# Wikipedia search tool
from langchain_community.document_loaders import WikipediaLoader
from langchain_core.messages import get_buffer_string

# os.environ['OPENAI_BASE_URL'] = 'https://api.openai.com/v1'
# os.environ['OPENROUTER_API_KEY'] = '
# os.environ['OPENAI_API_KEY'] = os.environ['OPENROUTER_API_KEY']

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

analyst_instructions="""You are tasked with creating a set of AI analyst personas. Follow these instructions carefully:

1. First, review the research topic:
{topic}

2. Examine any editorial feedback that has been optionally provided to guide creation of the analysts:

{human_analyst_feedback}

3. Determine the most interesting themes based upon documents and / or feedback above.

4. Pick the top {max_analysts} themes.

5. Assign one analyst to each theme."""

def create_analysts_bot(state: GenerateAnalystsState):

    """ Create analysts """

    topic=state['topic']
    max_analysts=state['max_analysts']
    human_analyst_feedback=state.get('human_analyst_feedback', '')
    print(f"Topic: {topic}")
    print(f"Max analysts: {max_analysts}")
    print(f"Human analyst feedback: {human_analyst_feedback}")

    # Enforce structured output
    structured_llm = llm.with_structured_output(Perspectives)

    # System message
    system_message = analyst_instructions.format(topic=topic,
                                                            human_analyst_feedback=human_analyst_feedback,
                                                            max_analysts=max_analysts)

    # Generate question
    analysts = structured_llm.invoke([SystemMessage(content=system_message)]+[HumanMessage(content="Generate the set of analysts.")])
    print(f"Generated analysts: {analysts}")
    # print type of analysts
    print(f"Type of analysts: {type(analysts)}")
    # Write the list of analysis to state
    #return { "analysts": analysts.analysts, "sections": state.get("sections", []), "context": state.get("context", []) }
    #return {"analysts": analysts.analysts, "sections": state.get("sections", []), "messages": state.get("messages", []), "context": state.get("context", []), "human_analyst_feedback": state.get("human_analyst_feedback", None)}
    return {"analysts": analysts.analysts}

def human_feedback_bot(state: GenerateAnalystsState):
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

#further_feedack = None
#graph.update_state(thread, {"human_analyst_feedback":further_feedack}, as_node="human_feedback")
class InterviewState(MessagesState):
    max_num_turns: int = 2  # Add default value
    context: Annotated[list, operator.add] = []  # Add default value
    analyst: Analyst
    interview: str = ""  # Add default value
    sections: Annotated[list, operator.add] = []  # Add default value
    messages: list = []  # Add default value

class SearchQuery(BaseModel):
    search_query: str = Field(None, description="Search query for retrieval.")

question_instructions = """You are an analyst tasked with interviewing an expert to learn about a specific topic.

Your goal is boil down to interesting and specific insights related to your topic.

1. Interesting: Insights that people will find surprising or non-obvious.

2. Specific: Insights that avoid generalities and include specific examples from the expert.

Here is your topic of focus and set of goals: {goals}

Begin by introducing yourself using a name that fits your persona, and then ask your question.

Continue to ask questions to drill down and refine your understanding of the topic.

When you are satisfied with your understanding, complete the interview with: "Thank you so much for your help!"

Remember to stay in character throughout your response, reflecting the persona and goals provided to you."""

def generate_question_bot(state: InterviewState):
    """ Node to generate a question """

    # Get state
    analyst = state["analyst"]
    messages = state["messages"]

    # Generate question
    system_message = question_instructions.format(goals=analyst.persona)
    question = llm.invoke([SystemMessage(content=system_message)]+messages)

    # Write messages to state
    #return { "messages": [question], "context": state.get("context", []), "sections": state.get("sections", [])}
    return {"messages": [question]}

# Search query writing
search_instructions = SystemMessage(content=f"""You will be given a conversation between an analyst and an expert.

Your goal is to generate a well-structured query for use in retrieval and / or web-search related to the conversation.

First, analyze the full conversation.

Pay particular attention to the final question posed by the analyst.

Convert this final question into a well-structured web search query""")

def search_web_bot(state: InterviewState):
    """Enhanced web search with proper resource persistence"""
    import requests
    OPENALEX_API_URL = "https://api.openalex.org/works"
    
    # Get search query
    structured_llm = llm.with_structured_output(SearchQuery)
    search_query = structured_llm.invoke([search_instructions] + state['messages'])
    
    # Search OpenAlex
    params = {
        "search": search_query.search_query,
        "filter": "is_paratext:false",
        "sort": "relevance_score:desc",
        "per_page": 5
    }
    
    response = requests.get(OPENALEX_API_URL, params=params)
    search_docs = []
    
    if response.status_code == 200:
        data = response.json()
        
        # Prepare resources for bot persistence
        resources_to_add = []
        for result in data.get("results", []):
            doc_info = {
                "title": result.get("title", "Unknown Title"),
                "link": result.get("id", "Unknown URL"),
                "description": {
                    "abstract": result.get("abstract", "No abstract available"),
                    "authors": ", ".join([auth["author"]["display_name"] for auth in result.get("authorships", [])])
                }
            }
            search_docs.append(doc_info)
            resources_to_add.append({
                "name": doc_info["title"],
                "link": doc_info["link"],
                "content": doc_info["description"]
            })

        # Persist to bot resources with metadata
        bot = get_bot()
        if bot:
            bot.add_or_update_results_in_resources(
                results=resources_to_add,
                metadatas_to_add={
                    "source": "OpenAlex",
                    "query": search_query.search_query,
                    "search_timestamp": str(datetime.now())
                }
            )

    # Format for LLM consumption
    formatted_docs = "\n\n---\n\n".join([
        f'<Document title="{doc["title"]}" href="{doc["link"]}">\nAuthors: {doc["description"]["authors"]}\nAbstract: {doc["description"]["abstract"]}\n</Document>'
        for doc in search_docs
    ])
    
    return {"context": [formatted_docs]}


def search_wikipedia_bot(state: InterviewState):

    """ Retrieve docs from wikipedia """

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

answer_instructions = """You are an expert being interviewed by an analyst.

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

def generate_answer_bot(state: InterviewState):

    """ Node to answer a question """

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

def save_interview_bot(state: InterviewState):

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

section_writer_instructions = """You are an expert technical writer.

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

def write_section_bot(state: InterviewState):
    """Enhanced section writing with proper structure persistence"""
    interview = state["interview"]
    context = state["context"]
    analyst = state["analyst"]
    
    section = llm.invoke([
        SystemMessage(content=section_writer_instructions),
        HumanMessage(content=f"Write a section based on: {context}")
    ])
    
    bot = get_bot()
    if bot:
        try:
            # Create main content section if doesn't exist
            main_sections = bot.get_all_sections()
            main_content_id = next(
                (s.section_id for s in main_sections if s.title == "Main Content"), 
                None
            )
            
            if not main_content_id:
                main_content_id = bot.create_and_add_section_then_return_id(
                    title="Main Content",
                    content="",
                    section_id=1  # Ensure it's first
                )

            # Add analyst's section as child
            section_id = bot.create_and_add_section_then_return_id(
                title=f"Analysis by {analyst.name}",
                content=section.content,
                parent_id=main_content_id
            )
            
            # Store metadata about the section
            bot.add_or_update_result_in_resources(
                metadatas={
                    "section_id": section_id,
                    "analyst": analyst.name,
                    "analyst_role": analyst.role,
                    "timestamp": str(datetime.now())
                },
                name=f"Section_{section_id}_Metadata",
                content={"section_type": "analysis", "parent_section": main_content_id}
            )
            
        except Exception as e:
            print(f"Error persisting section: {e}")
            
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
        # return [Send("conduct_interview", {"analyst": analyst,
        #                                    "messages": [HumanMessage(
        #                                        content=f"So you said you were writing an article on {topic}?"
        #                                    )
        #                                                ]}) for analyst in state["analysts"]]
        return [Send("conduct_interview", {
                    "analyst": analyst, "messages": [ HumanMessage( content=f"So you said you were writing an article on {topic}?")],
                    "max_num_turns": 2,  # Add explicit max_num_turns
                    "context": [],  # Add empty context
                    "sections": [],  # Add empty sections
                    "interview": ""  # Add empty interview
                }
            ) for analyst in state["analysts"]
        ]
report_writer_instructions = """You are a technical writer creating a report on this overall topic:

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

def write_report_bot(state: ResearchGraphState):
    # Full set of sections
    sections = state["sections"]
    topic = state["topic"]

    # Concat all sections together
    formatted_str_sections = "\n\n".join([f"{section}" for section in sections])

    # Summarize the sections into a final report
    system_message = report_writer_instructions.format(topic=topic, context=formatted_str_sections)
    report = llm.invoke([SystemMessage(content=system_message)]+[HumanMessage(content=f"Write a report based upon these memos.")])
    return {"content": report.content}

intro_conclusion_instructions = """You are a technical writer finishing a report on {topic}

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

def write_introduction_bot(state: ResearchGraphState):
    # Full set of sections
    sections = state["sections"]
    topic = state["topic"]

    # Concat all sections together
    formatted_str_sections = "\n\n".join([f"{section}" for section in sections])

    # Summarize the sections into a final report

    instructions = intro_conclusion_instructions.format(topic=topic, formatted_str_sections=formatted_str_sections)
    intro = llm.invoke([instructions]+[HumanMessage(content=f"Write the report introduction")])
    return {"introduction": intro.content}

def write_conclusion_bot(state: ResearchGraphState):
    # Full set of sections
    sections = state["sections"]
    topic = state["topic"]

    # Concat all sections together
    formatted_str_sections = "\n\n".join([f"{section}" for section in sections])

    # Summarize the sections into a final report

    instructions = intro_conclusion_instructions.format(topic=topic, formatted_str_sections=formatted_str_sections)
    conclusion = llm.invoke([instructions]+[HumanMessage(content=f"Write the report conclusion")])
    return {"conclusion": conclusion.content}

def finalize_report_bot(state: ResearchGraphState):
    """Enhanced report finalization with proper structure"""
    content = state["content"]
    introduction = state["introduction"]
    conclusion = state["conclusion"]
    
    bot = get_bot()
    if bot:
        try:
            # Organize sections in proper order
            intro_id = bot.create_and_add_section_then_return_id(
                title="Introduction",
                content=introduction,
                section_id=1
            )
            
            content_id = bot.create_and_add_section_then_return_id(
                title="Main Findings",
                content=content,
                section_id=2
            )
            
            conclusion_id = bot.create_and_add_section_then_return_id(
                title="Conclusion",
                content=conclusion,
                section_id=3
            )
            
            # Add metadata about report structure
            bot.add_or_update_result_in_resources(
                metadatas={
                    "report_structure": {
                        "introduction_id": intro_id,
                        "content_id": content_id,
                        "conclusion_id": conclusion_id
                    },
                    "generation_timestamp": str(datetime.now())
                },
                name="Final_Report_Structure"
            )
            
            # Combine for final report
            final_report = f"{introduction}\n\n---\n\n{content}\n\n---\n\n{conclusion}"
            
            # Store final assembled version
            bot.create_and_add_section_then_return_id(
                title="Complete Report",
                content=final_report,
                section_id=4
            )
            
        except Exception as e:
            print(f"Error finalizing report: {e}")
            final_report = f"{introduction}\n\n---\n\n{content}\n\n---\n\n{conclusion}"
            
    return {"final_report": final_report}

def multi_agent_research_generation_persist_each_agent(bot, max_analysts: int = 3):
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

    global get_bot
    get_bot = lambda: bot

    # Create initial state with topic and max_analysts
    #initial_state: GenerateAnalystsState = { "topic": topic, "max_analysts": max_analysts, "human_analyst_feedback": None, "analysts": [], "sections": [],  "messages": [], "context": [] }
    initial_state: GenerateAnalystsState = { "topic": topic, "max_analysts": max_analysts, "human_analyst_feedback": None, "analysts": []}
    # Recreate the interview graph within the function
    interview_builder = StateGraph(InterviewState)
    interview_builder.add_node("ask_question", generate_question_bot)
    interview_builder.add_node("search_web", search_web_bot)
    interview_builder.add_node("search_wikipedia", search_wikipedia_bot)
    interview_builder.add_node("answer_question", generate_answer_bot)
    interview_builder.add_node("save_interview", save_interview_bot)
    interview_builder.add_node("write_section", write_section_bot)

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
    builder.add_node("create_analysts", create_analysts_bot)
    builder.add_node("human_feedback", human_feedback_bot)
    
    # Use the interview_graph directly, without .compile()
    builder.add_node("conduct_interview", interview_graph)
    builder.add_node("write_report", write_report_bot)
    builder.add_node("write_introduction", write_introduction_bot)
    builder.add_node("write_conclusion", write_conclusion_bot)
    builder.add_node("finalize_report", finalize_report_bot)

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
    try:
        result = graph.invoke(initial_state, thread)
        final_state = graph.get_state(thread)
        report = final_state.values.get('final_report', '')
    except Exception as e:
        print(f"Error in graph execution: {e}")
        print(f"Current state: {graph.get_state(thread)}")
        raise

    return report

if __name__ == "__main__":
    from dataclasses import dataclass
    from typing import List, Dict, Any, Optional
    from datetime import datetime

    @dataclass
    class Section:
        section_id: int
        title: str
        content: str
        parent_id: Optional[int] = None

    @dataclass
    class Document:
        title: str
        context: str

    class MockBot:
        def __init__(self, title: str, context: str):
            self.document = Document(title=title, context=context)
            self.sections: List[Section] = []
            self.resources: List[Dict] = []
            self.next_section_id = 1
            self.next_resource_id = 1
            print(f"MockBot initialized with title: {title} and context: {context}")

        def create_and_add_section_then_return_id(self, title: str, content: str, section_id: int = None, parent_id: int = None) -> int:
            if section_id is None:
                section_id = self.next_section_id
                self.next_section_id += 1
            
            section = Section(section_id=section_id, title=title, content=content, parent_id=parent_id)
            self.sections.append(section)
            print(f"Created section {section_id}: {title} (parent: {parent_id})")
            return section_id

        def get_all_sections(self) -> List[Section]:
            return self.sections

        def get_sections(self, ids: List[int]) -> List[Section]:
            return [s for s in self.sections if s.section_id in ids]

        def edit_section(self, section_id: int, new_content: str = None, new_title: str = None, new_parent_id: int = None) -> bool:
            for section in self.sections:
                if section.section_id == section_id:
                    if new_content is not None:
                        section.content = new_content
                    if new_title is not None:
                        section.title = new_title
                    if new_parent_id is not None:
                        section.parent_id = new_parent_id
                    print(f"Edited section {section_id}")
                    return True
            return False

        def add_or_update_results_in_resources(self, results: List[Dict], metadatas_to_add: dict = None, store_linked_document_content: bool = False):
            for result in results:
                resource_id = self.next_resource_id
                self.next_resource_id += 1
                
                resource = {
                    'id': resource_id,
                    'document': {
                        'name': result.get('name', ''),
                        'link': result.get('link', ''),
                        'content': result.get('content', {})
                    },
                    'metadatas': metadatas_to_add or {}
                }
                
                self.resources.append(resource)
                print(f"Added resource {resource_id}: {result.get('name', '')}")
            return self

        def add_or_update_result_in_resources(self, metadatas: dict, name: str = None, content: dict = None, link: str = None, store_linked_document_content: bool = False):
            resource_id = self.next_resource_id
            self.next_resource_id += 1
            
            resource = {
                'id': resource_id,
                'document': {
                    'name': name,
                    'link': link,
                    'content': content or {}
                },
                'metadatas': metadatas
            }
            
            self.resources.append(resource)
            print(f"Added single resource {resource_id}: {name}")
            return self

        def get_all_resources(self) -> List[Dict[str, Any]]:
            return self.resources

        def semantic_search_resources(self, query_texts, n_results=10):
            print(f"Mock semantic search for: {query_texts}")
            return []  # Mock empty results

        def remove_resource(self, resource_id):
            self.resources = [r for r in self.resources if r['id'] != resource_id]
            print(f"Removed resource {resource_id}")
            return self

    # Test usage example:
    def test_mock_bot():
        # Initialize mock bot
        mock_bot = MockBot(
            title="Test Research",
            context="Testing the research assistant framework"
        )
        
        # Test section creation
        section_id = mock_bot.create_and_add_section_then_return_id(
            title="Introduction",
            content="This is a test introduction"
        )
        
        # Test resource addition
        mock_bot.add_or_update_results_in_resources([
            {
                "name": "Test Resource",
                "link": "https://test.com",
                "content": {"description": "Test content"}
            }
        ], metadatas_to_add={"source": "test"})
        
        # Print current state
        print("\nCurrent sections:")
        for section in mock_bot.get_all_sections():
            print(f"Section {section.section_id}: {section.title}")
        
        print("\nCurrent resources:")
        for resource in mock_bot.get_all_resources():
            print(f"Resource {resource['id']}: {resource['document']['name']}")

        return mock_bot

    mock_bot = MockBot(
        title="Test Research @ Toulon M2 Master",
        context="Testing the research assistant framework in Toulon M2 Master"
    )

    # Run with explicit error handling
    if True: # try:
        result = multi_agent_research_generation_persist_each_agent(mock_bot, max_analysts=2)
        
        print("\nFinal Results:")
        print("Sections:", len(mock_bot.get_all_sections()))
        print("Resources:", len(mock_bot.get_all_resources()))
        
        if result:
            print("\nReport preview:", result[:200] + "..." if result else "No report")
            
    # except Exception as e:
    #     print(f"Error running research generation: {e}")
    #     print("\nFinal bot state:")
    #     print("Sections:", mock_bot.get_all_sections())
    #     print("Resources:", mock_bot.get_all_resources())