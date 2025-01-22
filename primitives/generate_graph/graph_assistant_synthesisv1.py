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
    """ The is the "reduce" step where we gather all the sections, combine them, and reflect on them to write the intro/conclusion """
    # Save full final report
    content = state["content"]
    if content.startswith("## Insights"):
        content = content.strip("## Insights")
    if "## Sources" in content:
        try:
            content, sources = content.split("\n## Sources\n")
        except:
            sources = None
    else:
        sources = None

    final_report = state["introduction"] + "\n\n---\n\n" + content + "\n\n---\n\n" + state["conclusion"]
    if sources is not None:
        final_report += "\n\n## Sources\n" + sources
    return {"final_report": final_report}

def persist_final_report_in_bot(bot, final_report: str):
    """
    Convertit le rapport Markdown final en sections du document,
    et extrait les sources à stocker dans bot.add_or_update_results_in_resources.
    
    Args:
        bot: L'objet ayant les méthodes:
             - create_and_add_section_then_return_id(title, content, section_id=None, parent_id=None)
             - add_or_update_results_in_resources(results, metadatas_to_add: dict=None, store_linked_document_content: bool=False)
        final_report (str): Le rapport final (Markdown) généré par les agents.
    """

    # 1) Extraire toutes les sections (repérées par un titre Markdown, ex. "## ...")
    #    Naïvement, on peut capturer les sections via une expression régulière.
    #    Exemple de pattern pour repérer "## Titre" jusqu'à la prochaine "##" ou fin de texte.
    #    Note: ceci est une simplification, on peut affiner pour gérer différents niveaux de titre.
    
    section_pattern = re.compile(r"(##\s+.+?)(?=##\s|$)", re.DOTALL)
    # On va aussi chercher s'il y a un "# Titre principal" avant tout
    # ex. "# My Title\n## Introduction..."
    
    # 2) Trouver un éventuel grand titre (optionnel) : "# ...\n"
    grand_titre_match = re.search(r"^#\s+(.*)", final_report)
    if grand_titre_match:
        grand_titre = grand_titre_match.group(1).strip()
        # On crée une section "Document Title" ou "Global Title"
        bot.create_and_add_section_then_return_id(
            title=grand_titre,
            content=f"(Main Title)\n\n{grand_titre}"
        )
    
    # 3) Extraire toutes les sections de niveau "##"
    sections = section_pattern.findall(final_report)

    # 4) Parcourir les sections extraites
    for section_md in sections:
        # Le pattern renvoie : "## NomDeSection\ncontenu..."
        # On peut séparer le titre du contenu
        lines = section_md.split('\n', 1)
        if len(lines) == 2:
            section_title_line, section_body = lines
        else:
            section_title_line = lines[0]
            section_body = ""

        # Nettoyer le titre : enlever "## " + espaces
        section_title = section_title_line.replace("##", "").strip()

        # Créer la section via bot
        bot.create_and_add_section_then_return_id(title=section_title, content=section_body)

    # 5) Détecter un bloc de sources, par exemple si le rapport contient "## Sources" ...
    sources_block_match = re.search(r"(##\s+Sources.*)", final_report, re.IGNORECASE | re.DOTALL)
    if sources_block_match:
        sources_block = sources_block_match.group(1)
        # Ex: 
        #  ## Sources
        #  [1] Some Link
        #  [2] Another Link ...
        # On peut parser chaque ligne de la forme `[1] le lien`.
        
        lines = sources_block.split('\n')
        resources_to_add = []
        for line in lines:
            # Chercher pattern "[1] https://..."
            m = re.match(r"\[(\d+)\]\s+(.*)", line.strip())
            if m:
                index = m.group(1)
                url_or_name = m.group(2)
                # On prépare un resource item
                resource_item = {
                    "name": f"Source_{index}",
                    "content": {"url": url_or_name},
                    "metadatas": {"index": index}
                }
                resources_to_add.append(resource_item)
        
        if resources_to_add:
            # Appel "add_or_update_results_in_resources"
            bot.add_or_update_results_in_resources(
                results=resources_to_add, 
                metadatas_to_add={"type": "reference"},
                store_linked_document_content=False
            )


def multi_agent_research_generation_persist_at_the_end(bot, max_analysts: int = 3):
    """
    Generate a full research report using multi-agent LangGraph workflow
    
    Args:
        bot: The bot object with the necessary methods
        max_analysts (int): The number of analysts to generate
    Returns:
        str: The final markdown research report
    """
    title, topic = bot.document.title, bot.document.abstract
    # Create initial state with topic and max_analysts
    initial_state: GenerateAnalystsState = { "topic": topic, "max_analysts": max_analysts, "human_analyst_feedback": None, "analysts": []}
    initial_state["bot"] = bot

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

    # # Logic
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
    graph = builder.compile(interrupt_before=['human_feedback'], checkpointer=memory2)
    display(Image(graph.get_graph(xray=1).draw_mermaid_png()))

    # Set up the thread configuration
    thread = {"configurable": {"thread_id": "1"}}

    # Run the graph to generate initial analysts
    for event in graph.stream(initial_state, thread, stream_mode="values"):
        analysts = event.get('analysts', '')
        if analysts:
            print("Generated Analysts:")
            for analyst in analysts:
                print(f"Name: {analyst.name}\nAffiliation: {analyst.affiliation}\nRole: {analyst.role}\nDescription: {analyst.description}")
                print("-" * 50)
            
            # Automatically proceed past human_feedback
            graph.update_state(thread, {"human_analyst_feedback": None}, as_node="human_feedback")
            break

    # Continue graph execution to completion
    try:
        for event in graph.stream(None, thread, stream_mode="values"):
            # You can add more detailed logging here if needed
            print("Processing:", event)
    except Exception as e:
        print(f"Error during graph execution: {e}")

    # Retrieve the final report
    final_state = graph.get_state(thread)
    report = final_state.values.get('final_report')

    persist_final_report_in_bot(bot, report)
    return report

# topics = ["Sustainable Technologies in Renewable Energy"]

# for topic in topics:
#     print(f"\n--- Generating Report for Topic: {topic} ---")
#     report = research_assistant(topic, f"Research on {topic}")
#     print(report)


# Example usage
    # from env.env import EnvironmentManager
    # llmORchains_list = {"default_llm": ChatOpenAI(model="gpt-4o-mini", temperature=0)}
    # embedding_function = "text-embedding-ada-002"

    # # Set the documents to test/validate as a list of environments
    # documents=[{ 'id':"cf0d353c-b43b-4a79-88f9-42c2c84cf75e",
    #             'title':"Complex QA and language models hybrid architectures, Survey",
    #         'context':"This paper reviews the state-of-the-art of language models architectures and strategies for 'complex' question-answering (QA, CQA, CPS) with a focus on hybridization. Large Language Models (LLM) are good at leveraging public data on standard problems but once you want to tackle more specific complex questions or problems (e.g. How does the concept of personal freedom vary between different cultures ? What is the best mix of power generation methods to reduce climate change ?) you may need specific architecture, knowledge, skills, methods, sensitive data protection, explainability, human approval and versatile feedback... Recent projects like ChatGPT and GALACTICA have allowed non-specialists to grasp the great potential as well as the equally strong limitations of LLM in complex QA. In this paper, we start by reviewing required skills and evaluation techniques. We integrate findings from the robust community edited research papers BIG, BLOOM and HELM which open source, benchmark and analyze limits and challenges of LLM in terms of tasks complexity and strict evaluation on accuracy (e.g. fairness, robustness, toxicity, ...) as a baseline. We discuss some challenges associated with complex QA, including domain adaptation, decomposition and efficient multi-step QA, long form and non-factoid QA, safety and multi-sensitivity data protection, multimodal search, hallucinations, explainability and truthfulness, temporal reasoning. We analyze current solutions and promising research trends, using elements such as: hybrid LLM architectural patterns, training and prompting strategies, active human reinforcement learning supervised with AI, neuro-symbolic and structured knowledge grounding, program synthesis, iterated decomposition and others.",
    #         'target_file_path': "env/IR_CPS_TechSynthesis/document_embedding_analysis/output/arxiv/Complex QA and language models hybrid architectures Survey.json"},
    #         { 'id':"42252c6c-12f3-4edf-9045-8acd69bc3356",
    #         'title':"Macroeconomic Effects of Inflation Targeting A Survey of the Empirical  Literature",
    #         'context':"This paper surveys the empirical literature of inflation targeting. The main findings from our review are the following: there is robust empirical evidence that larger and more developed countries are more likely to adopt the IT regime; the introduction of this regime is conditional on previous disinflation, greater exchange rate flexibility, central bank independence, and higher level of financial development; the empirical evidence has failed to provide convincing evidence that IT itself may serve as an effective tool for stabilizing inflation expectations and for reducing inflation persistence; the empirical research focused on advanced economies has failed to provide convincing evidence on the beneficial effects of IT on inflation performance, while there is some evidence that the gains from the IT regime may have been more prevalent in the emerging market economies; there is not convincing evidence that IT is associated with either higher output growth or lower output variability; the empirical research suggests that IT may have differential effects on exchange-rate volatility in advanced economies versus EMEs; although the empirical evidence on the impact of IT on fiscal policy is quite limited, it supports the idea that IT indeed improves fiscal discipline; the empirical support to the proposition that IT is associated with lower disinflation costs seems to be rather weak. Therefore, the accumulated empirical literature implies that IT does not produce superior macroeconomic benefits in comparison with the alternative monetary strategies or, at most, they are quite modest.",
    #         'target_file_path': "env/IR_CPS_TechSynthesis/document_embedding_analysis/output/arxiv/Macroeconomic Effects of Inflation Targeting A Survey of the Empirical  Literature.json"}]

    # envs_tech_synthesis = []
    # for doc in documents:
    #     env = EnvironmentManager(env_type="techsynthesis", title=doc['title'], context=doc['context'],
    #                              target_file_path=doc['target_file_path'], id=doc['id'],
    #                              llm=llmORchains_list["default_llm"], embedding_model_name=embedding_function).get_environment()
    #     envs_tech_synthesis.append(env)

    # Example topics to test the function
    # topics = ["The benefits of adopting LangGraph as an agent framework","Sustainable Technologies in Renewable Energy"]

    # for topic in topics:
    #     print(f"\n--- Generating Report for Topic: {topic} ---")
    #     report = research_assistant(topic, f"Research on {topic}")
    #     print(report)