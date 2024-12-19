def find_buggy_code(bot):
    """
    A LangGraph workflow that takes a problem_statement and uses agents to search through the projects directory and find those files that contains the bugs.

    Args:
        bot: SWEManager class object with the following structure:
        class SWEManager:
            def getSearchTools(self):
            return [self.ls, self.goto_directory, self.goto_previous_dir, self.get_current_dir, self.number_of_lines, self.open_file, self.find_files, self.search_file, self.search_dir]

            def getContentViewingTools(self):
            return [self.get_files_content]
    Returns:
        The dictionary containing the paths, line numbers and the code that is causing the issue.
    """

    from langgraph.graph import StateGraph, END, add_messages
    from langchain_openai import ChatOpenAI
    from langchain_core.prompts import PromptTemplate
    from langchain_core.messages import AnyMessage, HumanMessage, AIMessage
    from langgraph.prebuilt import ToolNode
    from typing import TypedDict, Annotated
    from langchain_core.runnables import RunnableConfig
    import json

    OPENAI_API_KEY = "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"

    model = "gpt-4o-mini"

    search_llm = ChatOpenAI(
        model=model,
        temperature=0.3,
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

    class Problem(TypedDict):
        problem_statement: str
        hint: str
        repo_structure: str

    class BugFindResponse(TypedDict):
        paths: list[str]
        line_numbers: list[int]
        code: str

    class PatchState(TypedDict):
        repo: str
        problem: Problem
        response: BugFindResponse
        messages: Annotated[list[AnyMessage], add_messages]

    search_issue_code_prompt = PromptTemplate.from_template("""
    You have been provided some tools to access the files of {repo} repository. You are currently in the root directory of this repository. You will be provided with an issue. Your task is to pin-point those files with line numbers from the repository that are likely to cause that issue.

    Problem Statement: {issue}

    Hint: {hint} 
                                                            
    This is the file structure of this repository:
    {repo_structure}

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


    def search_issue_code(state: PatchState):
        print('Finding buggy files or choosing tool')
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
        print('Using search tool')
        return 'continue'

    def extract_content_from_files(state: PatchState):
        print('Extracting content from files')
        message = state['messages'][-1]
        response: AIMessage = extractor_llm.invoke([extract_content_from_files_prompt.format(files=message.content, current_dir=bot.current_dir)])
        print(f"Got error-related files")

        for tool_call in response.tool_calls:
            if tool_call['name'] == "get_files_content":
                args = tool_call['args']
                for path, line_number in zip(args['paths'], args['line_numbers']):
                    state['response']['paths'].append(path)
                    state['response']['line_numbers'].append(line_number)
        return {
            **state,
            "messages": [response]
        }    
    
    def formulate_response(state: PatchState):
        print('Formulating response')
        for i in range(len(state['messages'])-1, 0, -1):
            if isinstance(state['messages'][i], AIMessage):
                break
            state['response']["code"] += state['messages'][i].content
        return state

    workflow = StateGraph(PatchState)

    workflow.add_node("search_issue_code", search_issue_code)
    workflow.add_node("search_tools", search_tool_node)
    workflow.add_node("extract_content_from_files", extract_content_from_files)
    workflow.add_node("extract_code_tools", extract_code_tool_node)
    workflow.add_node('formulate_response', formulate_response)

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
    workflow.add_edge("extract_code_tools", "formulate_response")
    workflow.add_edge("formulate_response", END)

    app = workflow.compile()
    
    initial_state = {
        "messages": [HumanMessage(content=search_issue_code_prompt.format(repo=problem.repo, issue=problem.problem_statement, hint=problem.hints_text, repo_structure=problem.repo_structure))],
        "repo": problem.repo,
        "problem": {
            "problem_statement": problem.problem_statement,
            "hint": problem.hints_text,
            "repo_structure": problem.repo_structure if problem.repo_structure else ""
        },
        "response": {
            "paths": [],
            "line_numbers": [],
            "code": ""
        }
    }

    result = app.invoke(initial_state, config=RunnableConfig(recursion_limit=100))
    
    with open(f"{env.swe_temp_path}/{problem.instance_id}/state.json", 'r') as file:
        state = json.load(file)
    
    state['buggy_files'] = []
    for path, line_number in zip(result['response']['paths'], result['response']['line_numbers']):
        state['buggy_files'].append({
            "path": path,
            "line_number": line_number
        })
    state['buggy_files_content'] = result['response']['code']

    with open(f"{env.swe_temp_path}/{problem.instance_id}/state.json", 'w') as file:
        json.dump(state, file)

    return result['response']
