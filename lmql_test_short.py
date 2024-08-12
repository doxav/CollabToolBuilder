import lmql
from lmql.runtime.program_state import ProgramState
import copy
import asyncio
import nest_asyncio
nest_asyncio.apply()
import json
from functools import wraps

# define OPENAI_API_KEY 
import os
os.environ["OPENAI_API_KEY"] = "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"

modification_history = []

def humancontrol_lmql(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        print(f"Customize generation before executing query function: {func.__name__}")

        # Display current values
        print(f'Kwargs: {kwargs}' )
        print(f'Args: {args}' )
        print(f"Current constraints: {kwargs.get('where', 'None')}")
        print(f"Current sub-query: {args[0] if args else 'None'}")
        print(f"Current decoding algorithm: {kwargs.get('decoder', 'argmax')}")
        print(f"Current temperature: {kwargs.get('temperature', 1.0)}")

        # Save the current state before modifications
        current_state = copy.deepcopy(kwargs)
        modification_history.append(current_state)
        num_alternatives = kwargs.get('num_alternatives', 1)

        # Check if user wants modify some values (if not or enter, just continue with current values)
        if input("Do you want to modify some values ? (y/n): ").lower() == 'y':
            # User modifications before generation
            new_constraints = input("Enter new constraints (or press Enter to skip): ")
            new_sub_query = input("Enter a new sub-query (or press Enter to skip): ")
            new_decoder = input("Enter a new decoding algorithm (e.g., 'argmax', 'sample', 'best_k', 'beam', 'beam_sample', or press Enter to skip): ")
            new_temperature = input("Enter new temperature (or press Enter to skip): ")
            num_alternatives = int(input("Enter the number of alternatives to generate (default: 1): ") or "1")

            # Provide real-time feedback and confirmation
            # Ask if the user wants to preview the expected alternatives with the proposed changes
            if input("Do you want to preview the expected alternatives with the proposed changes? (y/n): ").lower() == 'y':
                preview_alternatives = simulate_changes(args[0], decoder=new_decoder or kwargs.get('decoder', 'argmax'), temperature=kwargs.get('temperature', 1.0), num_alternatives=num_alternatives, **kwargs)
                print(f"Preview of the expected alternatives with the proposed changes:")
                for i, alternative in enumerate(preview_alternatives):
                    print(f"Alternative {i + 1}:\n{alternative}\n")

            confirmation = input("Do you want to apply these changes? (y/n): ")
            if confirmation.lower() != 'y':
                return func(*args, **kwargs)

            # Apply user modifications to kwargs
            if new_temperature:
                kwargs['temperature'] = float(new_temperature)
            if new_decoder:
                kwargs['decoder'] = new_decoder
            if new_constraints:
                if "where" in kwargs:
                    kwargs["where"] = f"{kwargs['where']} and {new_constraints}"
                else:
                    kwargs["where"] = new_constraints
            if new_sub_query:
                query_string = args[0]
                if "[ANALYSIS]" in query_string:
                    args = (new_sub_query,) + args[1:]
            kwargs['num_alternatives'] = num_alternatives

        #alternatives = []
        # Collect all coroutine objects in a list
        # tasks = [func(*args, **kwargs) for _ in range(num_alternatives)]

        # # Use asyncio.gather to run them concurrently and wait for all to complete
        # alternatives = await asyncio.gather(*tasks)
        
        # call func with args and kwargs num_alternatives times
        # TODO: improve processing with concurrent.futures.ThreadPoolExecutor
        alternatives = [lmql.generate_sync(*args, **kwargs) for _ in range(num_alternatives)]

        print(f"Result after executing query function: {func.__name__}")
        # check if alternatives is a list or a string
        if isinstance(alternatives, list):
            for i, alternative in enumerate(alternatives):
                print(f"Alternative {i + 1}:\n{alternative}\n")
        elif isinstance(alternatives, str):
            alternatives = [alternatives]
            selected_index = 0
        else:
            print("Uknwown return type handling from lmql_query")

        # User interactions after generation
        action = input("Enter action ('select', 'refine', 'replace', 'halt', 'branch', or press Enter to continue with first alternative): ").lower()
        if action in ['select', 's']:
            selected_index = int(input("Enter the index of the alternative to select: "))
            if 1 <= selected_index <= len(alternatives):
                return alternatives[selected_index - 1]
            else:
                print("Invalid index, continuing with the first alternative.")
                return alternatives[0]
        elif action in ['refine', 'r']:
            refined_result = input("Enter refined result: ")
            return refined_result
        elif action == 'replace':
            new_result = input("Enter new result: ")
            return new_result
        elif action in ['halt', 'h']:
            return alternatives[0]
        elif action == 'branch':
            branch_point = input("Enter the branch point (e.g., '$.items[0]'): ")
            branch_value = input("Enter the branch value: ")
            return branch_result(alternatives[0], branch_point, branch_value)
        else:
            return alternatives[0]

    return wrapper

def branch_result(result, branch_point, branch_value):
    # Parse the branch point and update the result accordingly
    try:
        data = json.loads(result)
        keys = branch_point.split('.')
        current = data
        for key in keys[:-1]:
            if isinstance(current, dict):
                current = current[key]
            else:
                current = current[int(key)]

        last_key = keys[-1]
        if isinstance(current, dict):
            current[last_key] = branch_value
        else:
            current[int(last_key)] = branch_value

        return json.dumps(data, indent=2)
    except Exception as e:
        print(f"Error branching result: {e}")
        return result

def revert_to_previous_state(index=-1):
    # Revert to the specified state in the modification history
    if -len(modification_history) <= index < 0:
        previous_state = modification_history[index]
        return previous_state
    else:
        print("Invalid index. Cannot revert to the specified state.")
        return None

def analyze_modification_history():
    # Provide functions to analyze the modification history
    # (e.g., visualize changes, compare states, etc.)
    # NOT YET IMPLEMENTED
    pass

@humancontrol_lmql
@lmql.query
def my_query_function(query_string):
    """
    {query_string}
    """

@humancontrol_lmql
@lmql.query
def chain_of_thought(question):
    '''lmql
    # Q&A prompt template
    "Q: {question}\n"
    "A: Let's think step by step.\n"
    "[REASONING]"
    "Thus, the answer is:[ANSWER]."

    # return just the ANSWER to the caller
    return ANSWER
    '''


class CustomOutputWriter(lmql.runtime.output_writer.BaseOutputWriter):
    def __init__(self, allows_input=True):
        self.allows_input = allows_input
        self.output_state = []

    async def input(self, *args):
        """
        Handle user input with an input prompt of *args. This is invoked when a query asks for user input via `await input()`.

        Returns:
            str: The user input.
        """
        print(f"Input prompt: {args}")
        if not self.allows_input:
            assert False, "current LMQL output writer does not allow input"
        return input(*args)

    async def add_interpreter_head_state(self, variable, head, prompt, where, trace, is_valid, is_final, mask, num_tokens, program_variables): 
        """
        Called whenever the query interpreter progresses in a meaningful way (e.g. new token added, new variable added, variable updated, etc.).

        Parameters:
            variable (str): 
                The name of the currently active variable.
            head (int): 
                The index of the current interpretation head (deprecated, will always be 0).
            prompt (str): 
                The full interaction trace/prompt of the query.
            where (object): 
                The AST representation of the queries validation condition.
            trace (object): 
                The evaluation trace of evaluating 'where' on the current program variables during generation.
            is_valid (bool): 
                Whether the current program variables satisfy the validation condition.
            is_final (bool): 
                Whether the value of 'valid' can be considered final (i.e. decoding more tokens will not change the value of 'valid').
            mask (np.ndarray): 
                Currently active token mask.
            num_tokens (int): 
                Number of tokens in the current 'prompt'.
            program_variables (ProgramState): 
                The current program state (lmql.runtime.program_state). E.g. program_variables.variable_values is a mapping of variable names to their current values.
        """
        current_state = {
            "variable": variable,
            "head": head,
            "prompt": prompt,
            "where": where,
            "trace": trace,
            "is_valid": is_valid,
            "is_final": is_final,
            "mask": mask,
            "num_tokens": num_tokens,
            "program_variables": program_variables,
        }
        # print(f"\033[32mREASONING\033[0m:\033[31m{REASONING}\033[0m")
        # print(f"\033[32mTASK\033[0m:\033[31m{TASK}\033[0m")
        # print(f"\033[32mSPECIFICATIONS\033[0m:{SPECIFICATIONS}")
        # print(f"\033[32mTESTS\033[0m:{TESTS}")
        # print currrent num_tokens and 20 first characters of prompt and 20 last characters, replace \n by \\n in prompt
        print(f"\033[33mCurrent num_tokens:\033[0m {num_tokens} \033[33mPrompt:\033[0m " + prompt[:20].replace('\n', '\\n') + " ... " + prompt[-20:].replace('\n', '\\n'))
        self.output_state.append(current_state)

    def get_output_state(self):
        return self.output_state

    def print_state(self):
        current_state = self.output_state[-1]
        print(f"\033[32m<< Interpreter head state:\033[0m")
        for key, value in current_state.items():
            print(f"\033[33m{key}:\033[0m {value}")
        print(f"\033[32mEND Interpreter head state >>\033[0m")


    def add_compiler_output(self, code): 
        print(f"Compiler output code parameter: {code}")

custom_output_writer = CustomOutputWriter()

from lmql.runtime.program_state import ProgramState
from lmql.language.qstrings import TemplateVariable
from lmql.runtime.program_state import ProgramState
from typing import Any

@lmql.decorators.pre
def hl(variable: TemplateVariable, context: ProgramState):
    print(f"TEST decorators.pre@{variable.name}")

def hl2(value: Any, prompt_value: str, context: ProgramState):
    """Decorator to convert a comma-separated string into a List[str]"""

    # print the current state info
    #custom_output_writer.print_state()
    print(f"\033[31mProgram state:\033[0m {context} \033[31mValue:\033[0m <{value}> \033[31mprompt_value:\033[0m <{prompt_value}>")

    # get the custom_output_writer prompt without the current value
    current_prompt_without_new_value = custom_output_writer.output_state[-1]['prompt'].replace(value, '')
    current_variable = custom_output_writer.output_state[-1]['variable']

    prompt_value = value = input(f"New current prompt: \033[32m<<\033[0m{current_prompt_without_new_value}\033[32m<\033[31m{value}\033[32m>@{current_variable}\033[32m>>\033[0m\nManually enter new prompt \033[31mvalue above\033[0m or press 'Enter' to ACCEPT, or 'R' to REJECT: ") or value

    if prompt_value.lower() == 'r':
        prompt_value = value = ''

    # elif prompt_value[-1] != '\n':
    #     prompt_value += '\n'
    #prompt_value = input(f"Update prompt_value:'{prompt_value}' (or press Enter to skip): ") or prompt_value

    return value, prompt_value

def enforce_prefix(prefix: str = '-'):
    # actual decorator function
    def decorator(value: str):
        nonlocal prefix
        if isinstance(prefix, set):
            prefix = list(prefix)[0]
        # if value is not empty does not start with '-' even after many spaces, add it
        if value and not value.strip().startswith(prefix):
            return f"{prefix} {value}"
        else:
            return f"{value}"
    return decorator

# def enforce_prefix(value: str):
#     prefix = '-'
#     if value and not value.strip().startswith(prefix):
#         return f"{prefix} {value}"
#     else:
#         return f"{value}"

@lmql.query
def constraint(prefix: str = '- '):
    '''lmql
    while True:
        "{prefix}[@hl2 NEW_CONSTRAINT]" where len(TOKENS(NEW_CONSTRAINT)) > 5 and STOPS_AT(NEW_CONSTRAINT, "\n")
        if len(NEW_CONSTRAINT.strip()) > 0:
            return f"{prefix}{NEW_CONSTRAINT}"
    '''

@lmql.query
def constraints_list(max_constraints=100, prefix='- '):
    '''lmql
    #"""- [CONSTRAINT]"""  where len(TOKENS(CONSTRAINT))>5 and len(CONSTRAINT) < 120 and STOPS_AT(CONSTRAINT, "\n")
    #"[CONSTRAINT: constraint]" where len(TOKENS(CONSTRAINT)) > 5 and STOPS_AT(CONSTRAINT, "\n")
    #constraints = [CONSTRAINT.strip()]
    constraints = []
    
    for i in range(max_constraints):
        if i > 0:
            # check if user wants to continue
            custom_output_writer.print_state()
            if input(f"NEW CONSTRAINT ADDED: \033[31m{CONSTRAINT.strip()}\033[0m; Do you want to continue? (y/n): ").lower() not in ['y', '', 'yes']:
                break
        #"[@hl2 @enforce_prefix({prefix}) CONSTRAINT]" where len(TOKENS(CONSTRAINT)) > 5 and STOPS_AT(CONSTRAINT, "\n")
        "[CONSTRAINT: constraint]" where len(TOKENS(CONSTRAINT)) > 5 and STOPS_AT(CONSTRAINT, "\n")
        if len(CONSTRAINT.strip()) > 0:
            constraints.append(CONSTRAINT.strip())
        else:
            i -= 1

    return constraints
    '''

@lmql.query(verbose=False, output_writer=custom_output_writer, temperature=0.5) #, model="openai/gpt-4-1106-preview")
def coach_agent_next_task():
    '''lmql
    # Example 1: Multi-part prompt for creative writing
    #MIN = input("Enter the min number for words:")
    #Q: Write a short story with a number of words between {MIN} and [@hl MAX: int], and the following constraints:
    """
    You are a research assistant that identifies best tasks to devellop high quality technical synthesis (e.g. state-of-the-art research survey paper, Wikipedia like article, patent) given a [[Title]] and an [[Abstract]].
    Each task you propose will be then submitted to a language model which will try to convert it into Python functions, if the code is successful and the gain in technical synthesis quality is above a pre-defined threshold, this learnt task is made available to next learning iteration.
    This task should minimize distance to ideal document's plan (table of content), document's content of each section, bibliography/resources.
    The challenge is to create functions which synthesis output is better than what any very efficient LLM like GPT4, Claude 3, Llama could infer in one shot because a function can use code as well as leverage inference fom those LLM.
    You should keep this in mind to propose efficient task.

    I will provide you:
    - Learnt tasks available (with information gain between 0 (minimum) and 1 (maximum) on plan's titles, and contents): ...
    - Failed tasks to learn that are too hard: ...
    - Current status of examples of technical synthesis the proposed next task will be tested on: ...

    You should tell me the next best novel task we should try to implement given your LLM knowledge, available learnt tasks, in order to maximize synthesis generated quality, length and format, and speed to produce it.

    You must follow the criteria below:
    1) Reason step by step to find out best task to minimize distance to goal. Distance to goal is a sum of semantic similarity and length to full content, table of content, bibliography/resources. 
    2) Task should be written in the form of "[[action]] [[quantity if applicable]] [[object]] [[tools]] [[detailed instructions and parameters]]" - ie. ’verb’  is the verb of this action, ’object’ refers to the target object of the action, ’tools’ specifies the tools required for the action, ’detailed instructions and parameters’ are all important instructions and parameters to be detailed additional to the other information to describe the goal.
    3) Task will be converted into Python code given available commands, learnt tasks, use of LLM if required.
    4) Task should be novel compared to learnt and failed tasks.
    5) Develop key minimal elements of specification (acceptance criterias, best strategies to compare, performance tips to beat a LLM) to succesfully prompt a coder agent to generate code implementing the task while minimizing distance to goal. Organize the requirements with clear indexing (e.g., 'A.', 'A.1.').
    6) Tasks provided should be generic, not specific to given examples, so the reasoning can mention examples but proposed task and plan should not mention any information related to examples
    7) After proposing the task, you should provide a test case of the function corresponding to this task for each example:
        a) Write a one liner python call to the main function for each "Document to be tested", this call should be designed to maximize the expected results for the "Document to be tested"
        b) Precede each one liner call with a line of comment in this form "# document #uuid usage test" (e.g. "#document #125dc4bc-54e0-4336-82bc-417e40ec9b8f usage test"...) to indicate to which document the code of the next line applies to given its unique id
        c) Call to the main function uses bot as first required parameter, then provide parameters sepecific to the document for this function (do not provide document #uuid as parameter but title and abstract instead)
        d) Generate only one test for each document, so the total number of function calls in this test list should be equal to the number of "Document to be tested"

    RESPONSE FORMAT (you should only respond in the format as described below):

    1. Reasoning: based on the information listed above, do reasoning about needs, language model and python code capabilities, what the next task should be and why. Ensure it will minimize the distance to goal.

    2. Task: description of the next best task to develop.

    3. Specifications: present a tree-like structure of key acceptance criterias, best strategies to compare, performance tips to beat a single LLM.

    4. Tests:
    ```python
    # document #125dc4bc-54e0-4336-82bc-417e40ec9b8f usage test:
    task_function_name(bot, arguments with values describing document #125dc4bc-54e0-4336-82bc-417e40ec9b8f for the given task...)
    # document #2fa754cb-2e90-3376-3b2c-142f29c9ebf8 usage test:
    task_function_name(bot, arguments with values describing document #2fa754cb-2e90-3376-3b2c-142f29c9ebf8 for the given task...)
    ...

    EXAMPLE (task described below is not implemented, it is just an example):

    1. Reasoning: considering that you already have a function which generates a draft plan for a research survey paper and populate some initial content based on the abstract, the next best task to develop is a function that can generate a detailed content for each section of the paper based on the draft plan and abstract. This will help in minimizing the distance to goal by providing a more detailed and structured content for each section of the paper.

    2. Task: generate detailed content for each section of a research survey paper based on given abstract, title, draft plan populate with content

    3. Specifications:
    A. Acceptance Criteria:
        A.1. The function should generate a detailed content for each section of the paper based on the draft plan and abstract.
        A.2. The generated content should be structured and relevant to the section title.
        A.3. The function should be able to handle different types of sections (e.g., introduction, methodology, results, conclusion).
        A.4. Each section content should provide recent sources and references to support the content.
    B. Best Strategies to Compare:
        B.1. Compare the generated content with the existing content in the draft plan.
        B.2. Evaluate the relevance and coherence of the generated content with the section title.
        B.3. Check the quality and accuracy of the references provided in the content.
    C. Performance Tips:
        C.1. For each section content, call a new LLM call and make some reasearch using the abstract and previous and next sections to generate relevant content well aligned with the paper's structure and knowledge flow.
        C.2. Include recent and high-quality sources to support the content by maintaining a high quality resources/bibliography list.
        C.3. Ensure the content is plagiarism-free and original by developping dedicated helper function.

    4. Tests:
    ```python
    # document #125dc4bc-54e0-4336-82bc-417e40ec9b8f usage test:
    generate_detailed_content(bot, id="#125dc4bc-54e0-4336-82bc-417e40ec9b8f", title="Research Survey Paper on AI", abstract="This paper aims to provide a comprehensive review of recent advancements in Artificial Intelligence.", draft_plan=plan_for_ai_paper)
    # document #2fa754cb-2e90-3376-3b2c-142f29c9ebf8 usage test:
    generate_detailed_content(bot, id="#2fa754cb-2e90-3376-3b2c-142f29c9ebf8", title="Research Survey Paper on Economics of collaborative platforms", abstract="This paper aims to provide a comprehensive review of recent advancements in Economics of collaborative platforms.", draft_plan=plan_for_economics_paper)

    STATUS OF DEVELOPMENT:

    - Learnt tasks available: None
    - Failed tasks to learn that are too hard: None
    - Current status of examples of technical synthesis: Empty

    YOUR ANSWER:

    1. Reasoning: considering that [REASONING: constraints_list(1, '')]""" where len(TOKENS(REASONING)) > 10 and STOPS_AT(REASONING, "\n")

    """
    2. Task:
    [TASK: constraints_list]""" where len(TOKENS(TASK)) > 10 and STOPS_AT(TASK, "\n")

    """
    3. Specifications:
    [SPECIFICATIONS: constraints_list]

    4. Tests:
    ```python
    [TESTS: constraints_list]
    ...
    """

    print(f"\033[32mREASONING\033[0m:\033[31m{REASONING}\033[0m")
    print(f"\033[32mTASK\033[0m:\033[31m{TASK}\033[0m")
    print(f"\033[32mSPECIFICATIONS\033[0m:{SPECIFICATIONS}")
    print(f"\033[32mTESTS\033[0m:{TESTS}")
    '''

async def main():
    coach_agent_next_task()

# This is how you run the main function in the asyncio event loop
if __name__ == "__main__":
    asyncio.run(main())