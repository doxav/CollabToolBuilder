def generate_patch(bot, problem, env):
    """
    Generates a Python patch file in `git diff` format for a given problem by analyzing buggy files, 
    summarizing relevant repository structures, and using an LLM to produce the patch code.

    Args: 
        bot: SWEManager class object with the following structure:
            class SWEManager:
                def summarize_repo(self, root_dir, max_output_characters=20000, max_files=50, include_docstring=True, include_args=True) -> dict:
                    Summarize a repository by extracting classes and functions from Python files with size constraints.
                    Args:
                        root_dir (str): Path to the root directory of the repository.
                        max_output_characters (int): Maximum size of the summarized output.
                        max_files (int): Maximum number of files to process.
                        include_docstring (bool): Whether to include docstrings in the summary.
                        include_args (bool): Whether to include function arguments in the summary.

                    Returns:
                        dict: A compact summary of the repository.

    Returns:
        str: A generated patch file as a string in `git diff` format, ready to be applied using `git apply`.
    """

    import json, os, re
    from langchain_openai import ChatOpenAI
    from langchain_core.prompts import PromptTemplate
    from langgraph.graph import StateGraph, START, END, add_messages
    from typing import TypedDict, Annotated
    from langchain_openai import ChatOpenAI
    from langchain_core.prompts import PromptTemplate
    from langchain_core.messages import AnyMessage, HumanMessage
    from langgraph.prebuilt import ToolNode
    from langchain_core.runnables import RunnableConfig

    # Load the state.json file
    with open(f"{env.swe_temp_path}/{problem.instance_id}/state.json", 'r') as file:
        state = json.load(file)

    # Extract buggy directories
    buggy_dirs = set()
    for buggy_file in state['buggy_files']:
        buggy_dirs.add(os.path.dirname(buggy_file['path']))

    print("Buggy dirs: ", buggy_dirs)

    # Parse directories into lists, then convert to tuples for hashable usage in sets
    parsed_dirs = [tuple(dir.split('/')) for dir in buggy_dirs]

    # Select the directory with the shortest length
    min_length = min(len(dir) for dir in parsed_dirs)
    filtered_buggy_dirs = {dir for dir in parsed_dirs if len(dir) == min_length}

    # Filter directories further based on the matching condition
    for dir in parsed_dirs:
        to_keep = True
        for filtered_dir in filtered_buggy_dirs:
            matched = True
            for i in range(len(filtered_dir)):
                if i >= len(dir) or dir[i] != filtered_dir[i]:
                    matched = False
                    break
            if matched:
                to_keep = False
                break
        if to_keep:
            filtered_buggy_dirs.add(dir)

    # Convert tuples back to strings for final output (optional)
    filtered_buggy_dirs = {'/'.join(dir) for dir in filtered_buggy_dirs}

    print("Filtered buggy dirs: ", filtered_buggy_dirs)

    buggy_dirs_summarization = ""
    for dir in filtered_buggy_dirs:
        buggy_dirs_summarization += '\n' + json.dumps(bot.summarize_repo(f"{env.swe_repos_path}/{problem.repo.split('/')[-1]}/{dir}"))
    
    print("Buggy dirs summarization: ", len(buggy_dirs_summarization))

    example_patch = re.search(r'<patch>(.*?)</patch>', problem.text, re.DOTALL).group(1).strip()

    generate_patch_prompt = """
    You are a highly skilled software engineer specialized in fixing complex coding issues from GitHub repositories. Your task is to provide a python patch for a problem in the format of git diff that can be directly applied using git apply to a git repository.

    This is the problem statement of the issue:
    {problem_statement}

    This is the relevant code related to the issue
    {code}

    This is the structure of the directories where the bug can possibly reside:
    {buggy_dirs_structure}

    Here are some hints,
    {hints}

    Here is an example of a patch file. It consists of changes to the code base. It specifies the file names, the line numbers of each change, and the removed and added lines. A single patch file can contain changes to multiple files.

    {example_patch}

    Instructions:
    * Your task is to generate a single patch for this issue, similar to the example provided. 
    * Since this is a python code, so don't do indentation mistakes.
    * Also include tests, for the changes made, in your patch that can be run using 'pytest' command. 
    * Patch should be in a format that can be applied directly to the repository using `git apply`. 
    * Don't include any introductory or explainatory text. 
    * Don't encapsulate your answer in any mrkdown fenced codeblock. 
    * Your answer must be a string starting with diff --git.
    """

    generate_patch_prompt_template = PromptTemplate.from_template(generate_patch_prompt)

    OPENAI_API_KEY = ""

    model = "gpt-4o"
    llm = ChatOpenAI(
        model=model,
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2,
        api_key=OPENAI_API_KEY, 
    )

    input_prompt = generate_patch_prompt_template.format(problem_statement=problem.problem_statement, code=state['buggy_files_content'], buggy_dirs_structure=buggy_dirs_summarization, hints=problem.hints_text, example_patch=example_patch)

    print("Input prompt length: ", len(input_prompt))

    response = llm.invoke(input_prompt).content

    # apply the patch to the repository

    llm = llm.bind_tools(bot.getEditingTools())
    tool_node = ToolNode(tools=bot.getEditingTools())

    apply_patch_prompt = PromptTemplate.from_template("""
    This is a patch file for the {repo} repository in git diff format: 
    {patch}

    You are provided with tools to edit files in this repository. Ensure:
    1. Apply logical blocks of code (e.g., an entire function, if-else block) at once.
    2. Validate syntax and indentation after every edit.
    3. If a logical block cannot be applied due to errors, log the issue, skip that part, and proceed to the next.
    4. Avoid line-by-line edits; always prefer larger blocks.
    5. Return a summary of changes and any skipped parts at the end.

    Currently, you are in the {current_dir} directory.
    """)

    class PatchState(TypedDict):
        messages: Annotated[list[AnyMessage], add_messages]
    
    def apply_patch(state: PatchState):
        print("Applying patch")
        messages = state['messages']
        last_message = messages[-1]
        last_message.pretty_print()
        response = llm.invoke(messages)
        return {
            **state,
            "messages": [response]
        }

    def should_use_edit_tool(state: PatchState):
        print("Checking if editing tools should be used")
        messages = state['messages']
        last_message = messages[-1]
        last_message.pretty_print()
        if 'tool_calls' not in last_message.additional_kwargs:
            return 'end'
        return 'continue'
    
    workflow = StateGraph(PatchState)

    workflow.add_node("apply_patch", apply_patch)
    workflow.add_node("editing_tools", tool_node)

    workflow.add_edge(START, "apply_patch")
    workflow.add_conditional_edges(
        "apply_patch", 
        should_use_edit_tool, 
        {
            "continue": "editing_tools",
            "end": END
        }
    )
    workflow.add_edge("editing_tools", "apply_patch")

    app = workflow.compile()
    bot.resetCurrentDir()
    initial_state = {
        "messages": [HumanMessage(content=apply_patch_prompt.format(repo=problem.repo.split('/')[-1], current_dir=bot.current_dir, patch=response))],
    }

    for chunk in app.stream(initial_state, stream_mode="values", config=RunnableConfig(recursion_limit=100)):
        chunk["messages"][-1].pretty_print()

    state['generated_patch_code'] = bot.getDiff()

    bot.restoreRepo()

    with open(f"{env.swe_temp_path}/{problem.instance_id}/state.json", 'w') as file:
        json.dump(state, file)
    
    return state['generated_patch_code']
