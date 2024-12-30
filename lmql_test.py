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
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
modification_history = []

def simulate_changes(query_string, model=None, decoder=None, temperature=1.0, num_alternatives=1, **kwargs):
    # Simulate the query execution with the provided modifications
    # Return a list of alternatives
    try:
        if decoder == 'sample':
            alternatives = [lmql.run(query_string, model=model, decoder='sample', temperature=temperature, **kwargs) for _ in range(num_alternatives)]
        elif decoder == 'best_k':
            alternatives = [lmql.run(query_string, model=model, decoder='best_k', **kwargs) for _ in range(num_alternatives)]
        elif decoder == 'beam':
            alternatives = [lmql.run(query_string, model=model, decoder='beam', **kwargs) for _ in range(num_alternatives)]
        elif decoder == 'beam_sample':
            alternatives = [lmql.run(query_string, model=model, decoder='beam_sample', temperature=temperature, **kwargs) for _ in range(num_alternatives)]
        else:
            alternatives = [lmql.run(query_string, model=model, decoder='argmax', **kwargs)]
        return alternatives
    except Exception as e:
        print(f"Error simulating changes: {e}")
        return None

def customize_before(func):
    @wraps(func)
    async def wrapper(*args, **kwargs):
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
                return await func(*args, **kwargs)

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

        return await func(*args, **kwargs)

    return wrapper

def check_after(func):
    @wraps(func)
    async def wrapper(*args, **kwargs):
        alternatives = await func(*args, **kwargs)

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
        action = input("Enter action ('select', 'refine', 'replace', 'stop', 'branch', or press Enter to continue with first alternative): ")
        if action == 'select':
            selected_index = int(input("Enter the index of the alternative to select: "))
            if 1 <= selected_index <= len(alternatives):
                return alternatives[selected_index - 1]
            else:
                print("Invalid index, continuing with the first alternative.")
                return alternatives[0]
        elif action == 'refine':
            refined_result = input("Enter refined result: ")
            return refined_result
        elif action == 'replace':
            new_result = input("Enter new result: ")
            return new_result
        elif action == 'stop':
            return alternatives[0]
        elif action == 'branch':
            branch_point = input("Enter the branch point (e.g., '$.items[0]'): ")
            branch_value = input("Enter the branch value: ")
            return branch_result(alternatives[0], branch_point, branch_value)
        else:
            return alternatives[0]

    return wrapper

def humancontrol_lmql(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
    #     print(f"Customize generation before executing query function: {func.__name__}")

    #     # Display current values
    #     print(f'Kwargs: {kwargs}' )
    #     print(f'Args: {args}' )
    #     print(f"Current constraints: {kwargs.get('where', 'None')}")
    #     print(f"Current sub-query: {args[0] if args else 'None'}")
    #     print(f"Current decoding algorithm: {kwargs.get('decoder', 'argmax')}")
    #     print(f"Current temperature: {kwargs.get('temperature', 1.0)}")

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

from lmql.runtime.program_state import ProgramState
from lmql.language.qstrings import TemplateVariable
from lmql.runtime.program_state import ProgramState
from typing import Any

@lmql.decorators.pre
def hl(variable: TemplateVariable, context: ProgramState):
    print(f"TEST decorators.pre@{variable.name}")

def hl2(value: Any, prompt_value: str, context: ProgramState):
    """Decorator to convert a comma-separated string into a List[str]"""
    print(f"Program state: {context}")
    print(f"Value: {value}")
    print(f"prompt_value: {prompt_value}")
    prompt_value = value = input(f"Update prompt_value and value:'\033[31m{value}\033[0m' or press 'Enter' to ACCEPT, or 'R' to REJECT: ") or value
    if prompt_value.lower() == 'r':
        prompt_value = value = '\n'
    elif prompt_value[-1] != '\n':
        prompt_value += '\n'
    #prompt_value = input(f"Update prompt_value:'{prompt_value}' (or press Enter to skip): ") or prompt_value
    return value, prompt_value

@lmql.query
def constraints_list():
    '''lmql
    #"""- [CONSTRAINT]"""  where len(TOKENS(CONSTRAINT))>5 and len(CONSTRAINT) < 120 and STOPS_AT(CONSTRAINT, "\n")
    #constraints = [CONSTRAINT.strip()]
    constraints = []
    
    for i in range(10):
        "- [@hl2 CONSTRAINT]" where len(TOKENS(CONSTRAINT)) > 3 and STOPS_AT(CONSTRAINT, "\n")
        if len(CONSTRAINT.strip()) > 0:
            constraints.append(CONSTRAINT.strip())
        # check if user wants to continue
        if input(f"NEW CONSTRAINT ADDED: \033[31m{CONSTRAINT.strip()}\033[0m; Do you want to continue? (y/n): ").lower() not in ['y', '', 'yes']:
            break

    return constraints
    '''

@lmql.query
def my_multi_part_query1():
    '''lmql
    # Example 1: Multi-part prompt for creative writing
    #MIN = input("Enter the min number for words:")
    #Q: Write a short story with a number of words between {MIN} and [@hl MAX: int], and the following constraints:

    """Write a short story with the following constraints:\n[constraints: constraints_list]"""

    """Here is a possible short story that meets the constraints:[STORY]"""
    #print(f"MIN/MAX:{MIN}/{MAX}")
    print(f"constraints:{constraints}")
    print(f"ANALYSIS:{STORY}")
    
    # return just the ANALYSIS to the caller
    return constraints, STORY
    '''

@lmql.query(verbose=False)
def my_multi_part_query_best_task():
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
    2) Task should be written in the form of "[[verb]] [[quantity if applicable]] [[object]] [[tools]] [[detailed instructions and parameters]]" - ie. ’verb’  is the verb of this action, ’object’ refers to the target object of the action, ’tools’ specifies the tools required for the action, ’detailed instructions and parameters’ are all important instructions and parameters to be detailed additional to the other information to describe the goal.
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

    2. Task: next best task to develop.

    3. Specifications: present a tree-like structure of acceptance criterias, best strategies to compare, performance tips to beat a single LLM.

    4. Tests:
    ```python
    # document #125dc4bc-54e0-4336-82bc-417e40ec9b8f usage test:
    task_function_name(bot)
    # document #2fa754cb-2e90-3376-3b2c-142f29c9ebf8 usage test:
    task_function_name(bot)
    ...

    YOUR ANSWER:

    1. Reasoning: considering that [@hl2 REASONING]""" where len(TOKENS(REASONING)) > 10 and STOPS_AT(REASONING, "\n")

    """
    2. Task:
    [@hl2 TASK]""" where len(TOKENS(TASK)) > 10 and STOPS_AT(TASK, "\n")

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

@humancontrol_lmql
@lmql.query
def my_multi_part_query2():
    '''lmql
    # Example 2: Multi-part prompt for data analysis
    """
    Imagine a company name:[NAME]
    with the following employees list:[LIST]
    which goes to bankrupt because of:[REASON]
    """
    return REASON
    '''

@lmql.query
def my_meta_prompt1():
    '''lmql
    # Example 1: Meta-prompt for task decomposition
    """
    Q: Here is a high-level task: "Build a web application that allows users to search for and book vacation rentals.
    [STRATEGY]
    To accomplish this task, we can break it down into smaller sub-tasks:
    1. [SUBTASK1]: Design the user interface and user experience
    2. [SUBTASK2]: Implement the search functionality to query vacation rental listings
    3. [SUBTASK3]: Develop the booking system and payment integration
    4. [SUBTASK4]: Set up a database to store rental listings and user information
    5. [SUBTASK5]: Deploy the application to a web server

    For each sub-task, we can provide further details and steps to complete it.
    [RESULT]
    """
    # return just the RESULT to the caller
    return RESULT
    '''

@humancontrol_lmql
@lmql.query
def my_meta_prompt2():
    '''lmql
    # Example 2: Meta-prompt for problem-solving
    """
    Q: You are organizing a large outdoor event, and you need to ensure that there is enough seating for all attendees. However, you have a limited budget and can only rent a certain number of chairs. How would you approach this problem?
    [STRATEGY]
    To solve this problem, we can follow these steps:
    1. [STEP]: Estimate the expected number of attendees based on previous events, ticket sales, or other relevant data.
    2. [STEP]: Determine the seating requirements, such as the desired number of chairs per attendee or the percentage of attendees who will need seating.
    3. [STEP]: Calculate the total number of chairs needed based on the estimated attendance and seating requirements.
    4. [STEP]: Research rental costs for chairs from different vendors and compare prices.
    5. [STEP]: Based on the budget constraints, select the vendor that can provide the required number of chairs at the best price.
    6. [STEP]: Plan the seating layout and distribution of chairs to ensure adequate spacing and accessibility.
    7. [STEP]: Develop a contingency plan in case the attendance exceeds expectations or there are any issues with the chair rentals.

    [RESULT]
    By following these steps, you can effectively plan and manage the seating arrangements for your outdoor event while staying within your budget constraints.
    """
    # return just the RESULT to the caller
    return RESULT
    '''

async def main():
    # my_query_function("Q: Generate a JSON object representing a book with properties like title, author, and pages. [ANSWER]")
    # chain_of_thought("Q: Generate a JSON object representing a book with properties like title, author, and pages. [ANSWER]")
    # my_multi_part_query1()
    my_multi_part_query_best_task()
    # my_multi_part_query2()
    # my_meta_prompt1()
    # my_meta_prompt2()

# This is how you run the main function in the asyncio event loop
if __name__ == "__main__":
    asyncio.run(main())