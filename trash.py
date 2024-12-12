import os
# Check the docker environment variables to see if LLM_MODE is set (to launch on a cluster/macstudio using ollama)
IN_MACSTU = os.environ.get('LLM_MODE')
if IN_MACSTU in ["macstudio", "gpu"]:
    os.environ['OPENAI_BASE_URL'] = 'http://localhost:11434/v1'
    OPENAI_API_KEY = "ollama"
    if IN_MACSTU == "macstudio":
        MODELS_CONFIG_LIST = {
                "basic_gpt":"codestral_latest_20k:latest",
                "smart_gpt":"codestral_latest_20k:latest",
                "code_gpt":"codestral_latest_20k:latest",
        }
    else :
        MODELS_CONFIG_LIST = {
                "basic_gpt":"codestral:latest-20k",
                "smart_gpt":"codestral:latest-20k",
                "code_gpt":"codestral:latest-20k",
        }

else:
    OPENAI_API_KEY = "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"
    MODELS_CONFIG_LIST = {
            "code_gpt": "gpt-4o-mini-2024-07-18", #"llama-3.1-70b-versatile", # "mixtral-8x7b-32768", # "eramax/nxcode-cq-7b-orpo:q6", #mistral-nemo:12b-instruct-2407-q4_K_M", #"llama3:latest",
    "smart_gpt": "gpt-4o-mini-2024-07-18", #"llama-3.1-70b-versatile", # "mixtral-8x7b-32768", # "gpt-4o-mini", # "eramax/nxcode-cq-7b-orpo:q6", #mistral-nemo:12b-instruct-2407-q4_K_M", #"llama3:latest",
    "basic_gpt": "gpt-4o-mini-2024-07-18", #"llama-3.1-70b-versatile", # "mixtral-8x7b-32768", # "gpt-4o-mini" # "eramax/nxcode-cq-7b-orpo:q6" #mistral-nemo:12b-instruct-2407-q4_K_M", #"llama3:latest",
    }

openai_api_key = OPENAI_API_KEY
os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY

# Elastic search and Kibana information
elastic_url_port = 'https://f1b92ef6a1e743708a990785e5a6b782.us-central1.gcp.cloud.es.io:443' # Default port is 9200
kibana_url_port = 'Your_kibana_url_here' # Default port is 5601

# If there is a user and password needed for the elastic search, add them here
elastic_user="admin"
elastic_password="hello123"
PickleCacheActivated = False

# Neo4j database information
NEO4J_URI = 'URI_of_your_neo4j_database'
NEO4J_USER = 'neo4j_user'
NEO4J_PASSWORD = 'neo4j_password'

# List of models to use in the application, you can add your own models here, no limitation, for example:
MODELS_CONFIG_LIST = {
    "code_gpt": "gpt-4o-mini-2024-07-18", #"llama-3.1-70b-versatile", # "mixtral-8x7b-32768", # "eramax/nxcode-cq-7b-orpo:q6", #mistral-nemo:12b-instruct-2407-q4_K_M", #"llama3:latest",
    "smart_gpt": "gpt-4o-mini-2024-07-18", #"llama-3.1-70b-versatile", # "mixtral-8x7b-32768", # "gpt-4o-mini", # "eramax/nxcode-cq-7b-orpo:q6", #mistral-nemo:12b-instruct-2407-q4_K_M", #"llama3:latest",
    "basic_gpt": "gpt-4o-mini-2024-07-18", #"llama-3.1-70b-versatile", # "mixtral-8x7b-32768", # "gpt-4o-mini" # "eramax/nxcode-cq-7b-orpo:q6" #mistral-nemo:12b-instruct-2407-q4_K_M", #"llama3:latest",
}

def create_patch_workflow(bot):
    """
    Creates a patch generation workflow for analyzing issues in a GitHub repository, identifying relevant files and lines 
    of code, and extracting code content for further processing.
    
    This workflow uses a state machine that consists of the following key steps:
    
    1. **search_issue_code**: It is given a problem statement and a hint about the problem that the code is facing and it Identifies files and line numbers that might be causing an issue in a given repository.
    2. **search_tools**: Uses available search tools to help locate relevant files for the issue.
    3. **extract_content_from_files**: Extracts file contents from the identified files starting from specific line numbers.
    4. **extract_code_tools**: Utilizes external tools to extract the code content.
    5. **extract_from_text**: Extracts the final code and patch information from the provided text.
    
    The workflow transitions between these nodes based on the output of each node and the conditional logic defined.

    Args:
        bot: Synthesis Manager class that will provided tools and current directory to the agentic workflow.
        
    Returns:
        The compiled application `app` that can be run as part of a workflow to analyze, identify, and extract code related 
        to issues within a repository.
    """
    from langgraph.graph import StateGraph, END, MessagesState, add_messages
    from typing import TypedDict, Annotated, Sequence
    from langchain_groq import ChatGroq
    from langchain_openai import ChatOpenAI
    from langchain_core.prompts import PromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, FunctionMessage, AnyMessage, ToolMessage
    from langgraph.prebuilt import ToolExecutor, ToolInvocation, ToolNode
    from langchain_core.runnables import RunnableConfig
    from time import time
    import re, operator, json, git

    OPENAI_API_KEY = "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"
    GROQ_API_KEY = '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'

    model = "gpt-4o-mini"

    search_llm = ChatOpenAI(
        model=model,
        temperature=0.7,
        max_tokens=None,
        timeout=None,
        max_retries=2,
        api_key=OPENAI_API_KEY, 
    )

    extractor_llm = ChatOpenAI(
        model=model,
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2,
        api_key=OPENAI_API_KEY, 
    )

    search_llm = search_llm.bind_tools(bot.getSearchTools())
    extractor_llm = extractor_llm.bind_tools(bot.getContentViewingTools())

    search_tool_node = ToolNode(tools=bot.getSearchTools())
    extract_code_tool_node = ToolNode(tools=bot.getContentViewingTools())

    class PatchState(TypedDict):
        patch: str
        code: str
        example_patch: str
        data: dict
        repo: str
        patch_application_failures: int
        messages: Annotated[list[AnyMessage], add_messages]

    search_issue_code_prompt = PromptTemplate.from_template("""
    You have been provided some tools to access the files of {repo} repository. You are currently in the {cwd} directory. You will be provided with an issue. Your task is to pin-point those files with line numbers from the repository that are likely to cause that issue.
    You can use the bot parameter passed to the workflow to access the tools and the current directory.

    Problem Statement: {issue}

    Hint: {hint} 

    Use tool calls only if necessary. You are only allowed to use one most appropriate tool at a time. Call the tool you think is the most appropriate one and let the tools supply its output to the next agent. The next agent will continue to use more tools as necessary.
                                                            
    Remember to only use one most appropriate tool. If you don't know about the directories or files in the current directories, use 'ls' tool to list them.                          

    Only provide the file paths and the line numbers that are the cause of this issue in the format:
    File: ... Line: ...
    File: ...           
    """)

    extract_content_from_files_prompt = PromptTemplate.from_template("""
    These are the files with lines of code. 
    {files}
                                                                 
    Use the tools provided to you to provide the contents of these files starting from the line numbers. Remember currently you are in {current_dir} directory.                                                              
    """)

    def extract_data(state: PatchState):
        state['example_patch'] = re.search(r'<patch>(.*?)</patch>', state['data']['text'], re.DOTALL).group(1).strip()
        print(f"{state['data']['instance_id']}: Data extraction complete")
        return state

    def search_issue_code(state: PatchState):
        messages = state['messages']
        response = search_llm.invoke(messages)
        return {
            **state,
            "messages": [response]
        }

    def should_use_search_tool(state: PatchState):
        messages = state['messages']
        last_message = messages[-1]
        if 'tool_calls' not in last_message.additional_kwargs:
            return 'end'
        return 'continue'

    def extract_content_from_files(state: PatchState):
        message = state['messages'][-1]
        bot.directory_path = f"/{state['repo']}/"
        response = extractor_llm.invoke([extract_content_from_files_prompt.format(files=message.content, current_dir=directory.cwd)])
        print(f"{state['data']['instance_id']}: Got error-related files")
        return {
            **state,
            "messages": [response]
        }    

    workflow = StateGraph(PatchState)

    workflow.add_node("search_issue_code", search_issue_code)
    workflow.add_node("search_tools", search_tool_node)
    workflow.add_node("extract_content_from_files", extract_content_from_files)
    workflow.add_node("extract_code_tools", extract_code_tool_node)
    workflow.add_node('extract_from_text', extract_data)   

    workflow.set_entry_point('search_issue_code')

    workflow.add_conditional_edges(
        'search_issue_code',  
        should_use_search_tool,
        {
            'continue': 'search_tools',
            'end': 'extract_content_from_files'
        }
    )

    workflow.add_edge("search_tools", "search_issue_code")
    workflow.add_edge("extract_content_from_files", "extract_code_tools")
    workflow.add_edge("extract_code_tools", "extract_from_text")

    app = workflow.compile()
    return app



 
from langgraph.graph import StateGraph, END, MessagesState, add_messages
from typing import TypedDict, Annotated, Any
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import AnyMessage, ToolMessage
from langgraph.prebuilt import ToolNode
import re
def find_buggy_code_agentic_workflow(bot):
    from langgraph.graph import StateGraph, END, MessagesState, add_messages
    from typing import TypedDict, Annotated, Any
    from langchain_openai import ChatOpenAI
    from langchain_core.prompts import PromptTemplate
    from langchain_core.messages import AnyMessage, ToolMessage
    from langgraph.prebuilt import ToolNode
    import re

    class BuggyCodeState(TypedDict):
        messages: Annotated[list[AnyMessage], add_messages]
        repo: str
        cwd: str
        results: list

    OPENAI_API_KEY = "your_openai_api_key"
    model = "gpt-4o-mini"

    llm = ChatOpenAI(
        model=model,
        temperature=0,
        max_tokens=None,
        api_key=OPENAI_API_KEY,
    )

    llm = llm.bind_tools(bot.getSearchTools())

    search_buggy_code_prompt = PromptTemplate.from_template("""
    You are tasked with identifying buggy code in the repository located at {repo}. 
    You are currently in the {cwd} directory. 
    Your job is to analyze the files and pinpoint lines of code that may contain bugs. 
    Provide the output in the following format:
    File: <file_path> Line: <line_number> Content: <line_content> Error: <error_description>
    """)

    def search_buggy_code(state: BuggyCodeState):
        messages = state['messages']
        response = llm.invoke(messages)
        return {
            **state,
            "messages": [response],
            "results": parse_buggy_code_response(response.content)
        }

    def parse_buggy_code_response(response_content: str):
        results = []
        for line in response_content.splitlines():
            match = re.match(r'File: (.+) Line: (\d+) Content: (.+) Error: (.+)', line)
            if match:
                results.append({
                    "file_path": match.group(1),
                    "line_number": int(match.group(2)),
                    "line_content": match.group(3),
                    "error_description": match.group(4)
                })
        return results

    workflow = StateGraph(BuggyCodeState)

    workflow.add_node("search_buggy_code", search_buggy_code)

    workflow.set_entry_point('search_buggy_code')

    app = workflow.compile()
    bot.directory_path = "/path/to/your/repo"  # Set the correct path to your repository
    initial_state = {
        "messages": [ToolMessage(content=search_buggy_code_prompt.format(repo="your_repo_name", cwd=bot.directory_path))],
        "repo": "your_repo_name",
        "cwd": bot.directory_path,
        "results": []
    }
    
    return app.run(initial_state)

class BuggyCodeState(TypedDict):
        messages: Annotated[list[AnyMessage], add_messages]
        repo: str
        cwd: str
        results: list

def search_buggy_code(state: BuggyCodeState):
        messages = state['messages']
        response = llm.invoke(messages)
        return {
            **state,
            "messages": [response],
            "results": parse_buggy_code_response(response.content)
        }

def parse_buggy_code_response(response_content: str):
        results = []
        for line in response_content.splitlines():
            match = re.match(r'File: (.+) Line: (\d+) Content: (.+) Error: (.+)', line)
            if match:
                results.append({
                    "file_path": match.group(1),
                    "line_number": int(match.group(2)),
                    "line_content": match.group(3),
                    "error_description": match.group(4)
                })
        return results
find_buggy_code_agentic_workflow(bot)
