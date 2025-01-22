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

########################################################
#                 MODELS ET SCHÉMAS
########################################################

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
    analysts: List[Analyst] # Analysts list
    bot: object  # Pour manipuler le document

########################################################
#             CRÉATION ET GESTION DES ANALYSTES
########################################################

def create_analysts_bot(state: GenerateAnalystsState):
    """ Create analysts """
    if "analyst_instructions" not in state:
        state['analyst_instructions'] = """You are tasked with creating a set of AI analyst personas. Follow these instructions carefully:

        1. First, review the research topic:
        {topic}

        2. Examine any editorial feedback that has been optionally provided to guide creation of the analysts:
        {human_analyst_feedback}

        3. Determine the most interesting themes based upon documents and/or feedback above.

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

        # En fonction du retour, on normalise
        if isinstance(analysts_response, Perspectives):
            generated_analysts = analysts_response.analysts
        elif isinstance(analysts_response, dict):
            generated_analysts = analysts_response.get('analysts', [])
        elif hasattr(analysts_response, 'analysts'):
            generated_analysts = analysts_response.analysts
        else:
            # fallback
            generated_analysts = [
                Analyst(
                    name=f"Analyst {i+1}", 
                    role=f"Research Specialist {i+1}", 
                    affiliation="Research Institute", 
                    description=f"Analyzing aspects of {topic}"
                ) for i in range(max_analysts)
            ]

        # Transforme en objets Analyst
        generated_analysts = [
            Analyst(**(a.dict() if hasattr(a, 'dict') else a))
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

def human_feedback_bot(state: GenerateAnalystsState):
    """ No-op node that should be interrupted on """
    pass

def should_continue(state: GenerateAnalystsState):
    """ Return the next node to execute """
    human_analyst_feedback=state.get('human_analyst_feedback', None)
    if human_analyst_feedback:
        return "create_analysts"
    return END

########################################################
#             INTERVIEW ET RECHERCHE
########################################################

class InterviewState(MessagesState):
    max_num_turns: int 
    context: Annotated[list, operator.add]
    analyst: Analyst 
    interview: str 
    sections: list
    bot: object  # pour manipuler le document si nécessaire

class SearchQuery(BaseModel):
    search_query: str = Field(None, description="Search query for retrieval.")

def generate_question_bot(state: InterviewState):
    """ Node to generate a question """
    if "question_instructions" not in state:
        state["question_instructions"] = """You are an analyst tasked with interviewing an expert to learn about a specific topic.

        Your goal is to boil down to interesting and specific insights related to your topic.

        1. Interesting: Insights that people will find surprising or non-obvious.
        2. Specific: Insights that avoid generalities and include specific examples from the expert.

        Here is your topic of focus and set of goals: {goals}

        Begin by introducing yourself using a name that fits your persona, and then ask your question.

        Continue to ask questions to drill down and refine your understanding of the topic.

        When you are satisfied with your understanding, complete the interview with: "Thank you so much for your help!"

        Remember to stay in character throughout your response, reflecting the persona and goals provided to you."""

    question_instructions = state["question_instructions"]
    analyst = state["analyst"]
    messages = state["messages"]

    system_message = question_instructions.format(goals=analyst.persona)
    question = llm.invoke([SystemMessage(content=system_message)]+messages)

    return {"messages": [question]}

def search_web_bot(state: InterviewState):
    """ Retrieve academic papers from OpenAlex """
    import requests
    OPENALEX_API_URL = "https://api.openalex.org/works"

    structured_llm = llm.with_structured_output(SearchQuery)
    if "search_instructions" not in state:
        state["search_instructions"] = SystemMessage(content="""You will be given a conversation between an analyst and an expert.
        Your goal is to generate a well-structured query for retrieval or web-search.
        Pay attention to the final question asked by the analyst. Convert it into a well-structured search query.""")

    search_instructions = state["search_instructions"]
    search_query = structured_llm.invoke([search_instructions] + state['messages'])

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
        for result in data.get("results", []):
            doc_info = {
                "title": result.get("title", "Unknown Title"),
                "authors": ", ".join([auth["author"]["display_name"] for auth in result.get("authorships", [])]),
                "abstract": result.get("abstract", "No abstract available"),
                "url": result.get("id", "Unknown URL")
            }
            search_docs.append(doc_info)
    else:
        print(f"Error retrieving data from OpenAlex: {response.status_code}")

    # Format docs
    formatted_search_docs = "\n\n---\n\n".join([
        f'<Document title="{doc["title"]}" href="{doc["url"]}">\nAuthors: {doc["authors"]}\nAbstract: {doc["abstract"]}\n</Document>'
        for doc in search_docs
    ])

    # PERSISTANCE: on stocke ces résultats en "Resources" du document si bot est dispo
    bot = state.get("bot")
    if bot:
        # On crée une liste de "results" suivant la signature add_or_update_results_in_resources
        # Ex. un "result" = {"name": "OpenAlexDoc_1", "content": doc_info, "metadatas": {...}}
        results_for_bot = []
        for i, doc in enumerate(search_docs):
            res = {
                "name": f"OpenAlexDoc_{i+1}",
                "content": doc,  # tout le doc
            }
            results_for_bot.append(res)
        # Appel
        bot.add_or_update_results_in_resources(
            results=results_for_bot,
            metadatas_to_add={"source": "OpenAlex"}, 
            store_linked_document_content=False
        )

    return {"context": [formatted_search_docs]}

def search_wikipedia_bot(state: InterviewState):
    """ Retrieve docs from wikipedia """
    if "search_instructions" not in state:
        state["search_instructions"] = SystemMessage(content="""You will be given a conversation between an analyst and an expert.
        Convert the final question into a well-structured Wikipedia search query.""")

    search_instructions = state["search_instructions"]
    structured_llm = llm.with_structured_output(SearchQuery)
    search_query = structured_llm.invoke([search_instructions]+state['messages'])

    search_docs = WikipediaLoader(query=search_query.search_query, load_max_docs=2).load()

    formatted_search_docs = "\n\n---\n\n".join([
        f'<Document source="{doc.metadata["source"]}" page="{doc.metadata.get("page", "")}"/>\n{doc.page_content}\n</Document>'
        for doc in search_docs
    ])

    # PERSISTANCE: stocker dans bot resources
    bot = state.get("bot")
    if bot:
        results_for_bot = []
        for i, doc in enumerate(search_docs):
            res = {
                "name": f"WikipediaDoc_{i+1}",
                "content": {
                    "page_content": doc.page_content,
                    "metadata": doc.metadata
                },
            }
            results_for_bot.append(res)
        bot.add_or_update_results_in_resources(
            results=results_for_bot,
            metadatas_to_add={"source": "Wikipedia"},
            store_linked_document_content=False
        )

    return {"context": [formatted_search_docs]}

def generate_answer_bot(state: InterviewState):
    """ Node to answer a question """
    if "answer_instructions" not in state:
        state["answer_instructions"] = """You are an expert being interviewed by an analyst.

        Here is analyst area of focus: {goals}.

        Use only the information in the {context} to answer the question.
        Cite your sources in square brackets like [1].
        List your sources at bottom [1] Source Title, etc."""

    answer_instructions = state["answer_instructions"]
    analyst = state["analyst"]
    messages = state["messages"]
    context = state["context"]

    system_message = answer_instructions.format(goals=analyst.persona, context=context)
    answer = llm.invoke([SystemMessage(content=system_message)]+messages)
    answer.name = "expert"

    return {"messages": [answer]}

def save_interview_bot(state: InterviewState):
    """ Save the interview transcript """
    messages = state["messages"]
    interview = get_buffer_string(messages)
    return {"interview": interview}

def route_messages(state: InterviewState,
                   name: str = "expert"):
    """ Route between question and answer """
    messages = state["messages"]
    max_num_turns = state.get('max_num_turns', 2)

    num_responses = len([m for m in messages if isinstance(m, AIMessage) and m.name == name])
    if num_responses >= max_num_turns:
        return 'save_interview'

    last_question = messages[-2]
    if "Thank you so much for your help" in last_question.content:
        return 'save_interview'
    return "ask_question"

########################################################
#          CRÉATION DE SECTION PERSISTANTE
########################################################

def write_section_bot(state: InterviewState):
    """ Génère une section de rapport à partir de l’interview + context, puis la sauvegarde. """
    if "section_writer_instructions" not in state:
        state["section_writer_instructions"] = """You are an expert technical writer.

        1. Analyze the content of the source documents:
        {context}

        2. Summarize the interview:
        {interview}

        3. Create a short markdown section (<= 400 words). 
        Include a title with ##, sub-sections with ###, references, etc.
        """

    interview = state["interview"]
    context = state["context"]
    analyst = state["analyst"]  # pour éventuellement customiser le titre

    system_message = state["section_writer_instructions"].format(
        context=context, 
        interview=interview
    )
    section = llm.invoke([SystemMessage(content=system_message), 
                          HumanMessage(content=f"Write one short markdown section for {analyst.name}'s memo.")])

    # Persistance : on crée physiquement la section dans le document (si bot est défini)
    bot = state.get("bot")
    if bot:
        # On peut nommer la section d'après l'analyste
        section_id = bot.create_and_add_section_then_return_id(
            title=f"Memo by {analyst.name}",
            content=section.content
        )
        # On peut suivre l'ID si besoin, ici on n'en a pas forcément besoin
        # Juste on garde la trace dans state['sections'] par ex.
        return {"sections": [section.content]}
    else:
        return {"sections": [section.content]}

########################################################
#         SQUELETTE DU WORKFLOW DE RESEARCH
########################################################

class ResearchGraphState(TypedDict):
    topic: str 
    max_analysts: int 
    human_analyst_feedback: str 
    analysts: List[Analyst] 
    sections: Annotated[list, operator.add] 
    introduction: str 
    content: str 
    conclusion: str 
    final_report: str
    bot: object  # Pour manipuler la doc

def initiate_all_interviews(state: ResearchGraphState):
    """ Lance les interviews en parallèle via Send() """
    human_analyst_feedback=state.get('human_analyst_feedback')
    if human_analyst_feedback:
        return "create_analysts"
    else:
        topic = state["topic"]
        # On lance un sub-graph "conduct_interview" pour chaque analyst
        return [
            Send("conduct_interview", {
                "analyst": analyst,
                "bot": state["bot"],  # IMPORTANT pour persister dans le sous-graph
                "messages": [HumanMessage(
                    content=f"So you said you were writing an article on {topic}?"
                )]
            }) for analyst in state["analysts"]
        ]

def write_report_bot(state: ResearchGraphState):
    """ Regroupe toutes les sections en un rapport consolidé """
    if "report_writer_instructions" not in state:
        state['report_writer_instructions'] = """You are a technical writer creating a consolidated report on {topic}.
Each memo from the analysts is included below. Merge them into a single summary with citations.

Memos:
{context}

Write in Markdown:
## Insights
(Your consolidated text)

## Sources
(Unique sources in numeric order)
"""

    sections = state["sections"]
    topic = state["topic"]

    formatted_str_sections = "\n\n".join(sections)
    system_message = state["report_writer_instructions"].format(
        topic=topic, 
        context=formatted_str_sections
    )
    report = llm.invoke([SystemMessage(content=system_message),
                         HumanMessage(content="Write the consolidated report")])
    
    # Persister la section "Insights" dans le doc
    bot = state.get("bot")
    if bot:
        bot.create_and_add_section_then_return_id(
            title="Consolidated Insights",
            content=report.content
        )

    return {"content": report.content}

def write_introduction_bot(state: ResearchGraphState):
    if "intro_conclusion_instructions" not in state:
        state['intro_conclusion_instructions'] = """You are a technical writer finishing a report on {topic}.
Here are the sections you have so far:
{formatted_str_sections}

Write a crisp introduction in Markdown:
# Title
## Introduction
(about 100 words)...

(End)"""

    sections = state["sections"]
    topic = state["topic"]
    formatted_str_sections = "\n\n".join(sections)

    instructions = state["intro_conclusion_instructions"].format(
        topic=topic, 
        formatted_str_sections=formatted_str_sections
    )
    intro = llm.invoke([instructions, 
                        HumanMessage(content="Write the report introduction")])
    
    bot = state.get("bot")
    if bot:
        bot.create_and_add_section_then_return_id(
            title="Introduction Section",
            content=intro.content
        )

    return {"introduction": intro.content}

def write_conclusion_bot(state: ResearchGraphState):
    if "intro_conclusion_instructions" not in state:
        state['intro_conclusion_instructions'] = """You are a technical writer finishing a report on {topic}.
Here are the sections you have so far:
{formatted_str_sections}

Write a crisp conclusion in Markdown:
## Conclusion
(about 100 words)...

(End)"""

    sections = state["sections"]
    topic = state["topic"]
    formatted_str_sections = "\n\n".join(sections)

    instructions = state["intro_conclusion_instructions"].format(
        topic=topic, 
        formatted_str_sections=formatted_str_sections
    )
    conclusion = llm.invoke([instructions, 
                             HumanMessage(content="Write the report conclusion")])
    
    bot = state.get("bot")
    if bot:
        bot.create_and_add_section_then_return_id(
            title="Conclusion Section",
            content=conclusion.content
        )

    return {"conclusion": conclusion.content}

def finalize_report_bot(state: ResearchGraphState):
    """ Combine introduction + content + conclusion """
    content = state["content"]
    introduction = state["introduction"]
    conclusion = state["conclusion"]

    final_report = f"{introduction}\n\n---\n\n{content}\n\n---\n\n{conclusion}"
    state["final_report"] = final_report

    # Persiste le rapport final
    bot = state.get("bot")
    if bot:
        bot.create_and_add_section_then_return_id(
            title="Final Report",
            content=final_report
        )

    return {"final_report": final_report}

########################################################
#              WORKFLOW PRINCIPAL
########################################################

def multi_agent_research_generation_persist_each_agent(bot, max_analysts: int = 3):
    """
    Lance un workflow multi-agents et persiste chaque section dans le document via `bot`.
    """
    title = bot.document.title
    topic = bot.document.abstract

    # État initial
    initial_state: GenerateAnalystsState = {
        "topic": topic, 
        "max_analysts": max_analysts, 
        "human_analyst_feedback": None, 
        "analysts": [],
        "bot": bot
    }

    # Construction du sous-graph "interview"
    interview_builder = StateGraph(InterviewState)
    interview_builder.add_node("ask_question", generate_question_bot)
    interview_builder.add_node("search_web", search_web_bot)
    interview_builder.add_node("search_wikipedia", search_wikipedia_bot)
    interview_builder.add_node("answer_question", generate_answer_bot)
    interview_builder.add_node("save_interview", save_interview_bot)
    interview_builder.add_node("write_section", write_section_bot)

    interview_builder.add_edge(START, "ask_question")
    interview_builder.add_edge("ask_question", "search_web")
    interview_builder.add_edge("ask_question", "search_wikipedia")
    interview_builder.add_edge("search_web", "answer_question")
    interview_builder.add_edge("search_wikipedia", "answer_question")
    interview_builder.add_conditional_edges("answer_question", route_messages, ['ask_question','save_interview'])
    interview_builder.add_edge("save_interview", "write_section")
    interview_builder.add_edge("write_section", END)

    memory = MemorySaver()
    interview_graph_bot = interview_builder.compile(checkpointer=memory).with_config(run_name="Conduct Interviews")

    # Workflow global
    builder = StateGraph(ResearchGraphState)
    builder.add_node("create_analysts", create_analysts_bot)
    builder.add_node("human_feedback", human_feedback_bot)
    builder.add_node("conduct_interview", interview_graph_bot)
    builder.add_node("write_report", write_report_bot)
    builder.add_node("write_introduction", write_introduction_bot)
    builder.add_node("write_conclusion", write_conclusion_bot)
    builder.add_node("finalize_report", finalize_report_bot)

    builder.add_edge(START, "create_analysts")
    builder.add_edge("create_analysts", "human_feedback")
    builder.add_conditional_edges(
        "human_feedback", 
        initiate_all_interviews, 
        ["create_analysts", "conduct_interview"]
    )
    builder.add_edge("conduct_interview", "write_introduction")
    builder.add_edge("conduct_interview", "write_report")
    builder.add_edge("conduct_interview", "write_conclusion")
    builder.add_edge(["write_conclusion", "write_report", "write_introduction"], "finalize_report")
    builder.add_edge("finalize_report", END)

    memory2 = MemorySaver()
    graph = builder.compile(interrupt_before=['human_feedback'], checkpointer=memory2)

    # Affichage du graph
    display(Image(graph.get_graph(xray=1).draw_mermaid_png()))

    thread = {"configurable": {"thread_id": "1"}}

    # Démarrer le graph pour créer les analysts
    for event in graph.stream(initial_state, thread, stream_mode="values"):
        analysts = event.get('analysts', '')
        if analysts:
            print("Generated Analysts:")
            for analyst in analysts:
                print(f"Name: {analyst.name}\nAffiliation: {analyst.affiliation}\nRole: {analyst.role}\nDescription: {analyst.description}")
                print("-" * 50)
            # On simule la fin de feedback humain
            graph.update_state(thread, {"human_analyst_feedback": None}, as_node="human_feedback")
            break

    # Poursuivre l'exécution jusqu'à la fin
    try:
        for event in graph.stream(None, thread, stream_mode="values"):
            print("Processing:", event)
    except Exception as e:
        print(f"Error during graph execution: {e}")

    final_state = graph.get_state(thread)
    report = final_state.values.get('final_report')
    return report


########################################################
#          EXEMPLE DE LANCEMENT FINAL
########################################################

def example_usage(bot):
    """
    Exemple d'appel de la fonction `research_assistant(bot)`
    où `bot` est un objet implémentant :
      - create_and_add_section_then_return_id
      - add_or_update_results_in_resources
      - etc.
    """
    topic = bot.document.abstract  # ou tout autre concept
    print(f"\n--- Generating Report for Topic: {topic} ---")
    final_report = multi_agent_research_generation_persist_each_agent(bot, max_analysts=3)
    print("=== FINAL REPORT ===")
    print(final_report)
    return final_report
