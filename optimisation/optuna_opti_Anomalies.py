
import os
import sys
import time
from optimisation.optuna_main import launch_study, launch_run, definition_few_shots


def objective(trial, timestamp_exp: str):

    # Define parameters for Coach

    # Define the reset environment at the end of the learning loop
    reset_env_end = trial.suggest_categorical("reset_env_end", [True])

    # Define the few shots parameters and if they are used
    few_shots = definition_few_shots(trial)
    # Definition of the coach's prompt
    coach_agent_role = trial.suggest_categorical("role_priming", [ "You are a research assistant", "You are an AI coach", "You are a task optimizer", "You are a technical synthesis expert"])
    #coach_user_input_failed_tasks = trial.suggest_categorical("coach_user_input_failed_tasks", [True, False])
    coach_user_input = "I will provide you:\n- Learnt tasks available (with information gain between 0 and 1 on plan's titles, and contents): ...\n- Failed tasks to learn that are too hard to code: ...\n- Current status of examples of technical synthesis the proposed next task will be tested on: ..."
    coach_task_description = "You define best next task to generate state-of-the-art research survey paper given a [Title] and an [Abstract]. Each task you propose will be prompted to a language model which will try to convert it into Python functions. If the code is successful and gains in technical synthesis above a pre-defined threshold, this learnt task is made available to the next learning iteration."

    coach_format_output_type = trial.suggest_categorical("coach_format_output_type", ["JSON", "Markdown"])

    criteria_to_remove = trial.suggest_categorical("criteria_to_remove", ["complexity", "reasoning_steps", "task_format", "python_conversion", "novelty", "specifications", "generalization", "tests", "maximization", "plan_suggestions"])
    criteria_coach = []
    if criteria_to_remove != "complexity":
        coach_task_criteria_complexity = trial.suggest_categorical("task_complexity", [
            "Task shouldn’t be too difficult to convert into Python code given available commands and learnt tasks.",
            "Task should balance complexity and feasibility for effective implementation.",
            "Task should challenge the LLM while remaining solvable with available resources.",
            "Task should involve multiple steps that require coordination among different functions.",
            "Task should leverage advanced features of the LLM to achieve superior results.",
            "Task should be modular, allowing parts of the solution to be reused in other contexts.",
            "Task should be scalable, capable of being applied to larger datasets or more complex scenarios."
        ])
        criteria_coach.append(f"{len(criteria_coach)+1}) {coach_task_criteria_complexity}")
    if criteria_to_remove != "reasoning_steps":
        coach_instructions_reasoning_steps = trial.suggest_int("reasoning_steps", 3, 10)
        criteria_coach.append(f"{len(criteria_coach)+1}) Reason in {coach_instructions_reasoning_steps} steps to find out the best task to minimize distance to goal.")
    if criteria_to_remove != "task_format":
        coach_task_criteria_description_format = trial.suggest_categorical("task_format", [
            "Task should be written in the form of [verb] [quantity if applicable] [object] [tools] [detailed instructions and parameters]",
            "Task should be written in the form of [action] [target] using [method] with [specifications]",
            "Task should be written in the form of [operation] on [subject] utilizing [resources] following [guidelines]",
            "Task should be written in the form of [do] [what] [how] [with what] [detailed instructions]"
        ])
        criteria_coach.append(f"{len(criteria_coach)+1}) Task should be written in the form of \"{coach_task_criteria_description_format}\"")
    if criteria_to_remove != "python_conversion":
        criteria_coach.append(f"{len(criteria_coach)+1}) Task will be converted into Python code given available commands, learnt tasks. The Python code could call LLM when required.")
    if criteria_to_remove != "novelty":
        criteria_coach.append(f"{len(criteria_coach)+1}) Task should be novel compared to learnt and failed tasks.")
    if criteria_to_remove != "specifications":
        specification_depth = trial.suggest_int("specification_depth", 2, 5)
        criteria_coach.append(f"{len(criteria_coach)+1}) Detail specifications (acceptance criteria, strategies/alternative to compare, use of LLM agents/tools/...) to successfully prompt a coder agent to generate code implementing the task while minimizing distance to goal. Organize the requirements with clear indexing to a depth of {specification_depth}.")
    if criteria_to_remove != "generalization":
        criteria_coach.append(f"{len(criteria_coach)+1}) Tasks provided should be generic, not specific to given examples.")
    if criteria_to_remove != "maximization":
        criteria_coach.append(f"{len(criteria_coach)+1}) You should propose the next best novel task to implement 'given' available learnt tasks performance and LLM knowledge, 'maximizing' document generation quality, length and format, and speed to produce it.")
    if criteria_to_remove != "plan_suggestions":
        plan_depth = trial.suggest_int("plan_depth", 2, 4)
        criteria_coach.append(f"{len(criteria_coach)+1}) You should propose a plan to achieve this task by breaking it down as a tree-structure. The plan tree should be of depth {plan_depth}.")
    if criteria_to_remove != "tests":
        criteria_coach.append(f"{len(criteria_coach)+1}) "+f"""After the task description and plan, provide a simple unit test of the function:
        a) Write a one liner python call to the main function for each "Document to be tested", this call should be designed to maximize the expected results for the "Document to be tested"
        b) Precede each python test with {"a line of comment in this form '# document #uuid usage test' (e.g. '#document #125dc4bc-54e0-4336-82bc-417e40ec9b8f usage test'...) to indicate to which document the code of the next line applies to given its unique id" if coach_format_output_type == "Markdown" else "the DocumentID value to indicate to which document the test code refers to"}
        c) Call to the main function uses bot as first required parameter, then provide parameters sepecific to the document for this function
        d) No more than one test for each document, the total number of function calls in this test list should be equal to the number of "Document to be tested"
        """)

    criteria_user_message = trial.suggest_categorical("criteria_user_message", ["None", "Learnt", "Failed", "Env", "All", "LearntEnv", "FailedEnv"])

    if coach_format_output_type == "JSON":
        coach_format_output = """You should only respond in the JSON format described below:
         {
            "Reasoning": "Analysis of the provided information to determine the next best task to developp minimizing the distance to the goal"
            "NextBestTask": {
                "FunctionName": "YourFunctionNameOfNextBestTaskIdentified",
                "Description": "....."
            },
            "PerformanceAcceptanceCriteria": {
                ...
            },
            "DevelopmentPlan": {
                "PlanDepth": {plan_depth},
                "Steps": {}
            },
            "Tests": [
                { "DocumentID": "#125dc4bc-54e0-4336-82bc-417e40ec9b8f", "FunctionCall": "NameOfNextBestTaskIdentified(bot)" },
                { "DocumentID": "#2fa754cb-2e90-3376-3b2c-142f29c9ebf8", "FunctionCall": "NameOfNextBestTaskIdentified(bot)" }
            ]
        }"""
    elif coach_format_output_type == "Markdown":
        coach_format_output = """You should only respond in the Markdown format as described below:
        1. Reasoning: Analysis of the provided information to determine the next best task to develop, minimizing the distance to the goal
        2. Next Best Task:
            - Function Name: YourFunctionNameOfNextBestTaskIdentified
            - Description: .....
        3. Performance Acceptance Criteria:
            ...
        4. Development Plan:
            - Plan Depth: ...
            - Steps: ...
        5. Tests:
        ```python
        # document #125dc4bc-54e0-4336-82bc-417e40ec9b8f usage test:
        task_function_name(bot)
        # document #2fa754cb-2e90-3376-3b2c-142f29c9ebf8 usage test:
        task_function_name(bot)
        ```"""

    # Construct the prompt based on the suggested parameters
    criteria_coach = [
        f"1) Reason in {reasoning_steps} steps to find out the best task to minimize distance to goal.",
        f"2) Task should be written in the form of '{task_format}'",
        f"3) Task will be converted into Python code given available commands, learnt tasks, use of LLM if required.",
        f"4) Task should be novel compared to learnt {f'and failed ' if include_failed_tasks else ''}tasks.",
        f"5) Develop key minimal elements of specification (acceptance criteria, best strategies to compare, performance tips to beat a LLM) to successfully prompt a coder agent to generate code implementing the task while minimizing distance to goal. Organize the requirements with clear indexing to a depth of {specification_depth}.",
        f"6) Tasks provided should be generic, not specific to given examples{', so the reasoning can mention examples but proposed task and plan should not mention any information related to examples' if use_examples else ''}.",
        f"7) After proposing the task, you should provide a test case of the function corresponding to this task for each example:",
        f"a) Write a one liner python call to the main function for each 'Document to be tested', this call should be designed to maximize the expected results for the 'Document to be tested'",
        f"b) Precede each one liner call with a line of comment in this form '# document #uuid usage test' (e.g. '#document #125dc4bc-54e0-4336-82bc-417e40ec9b8f usage test'...) to indicate to which document the code of the next line applies to given its unique id",
        f"c) Call to the main function uses 'problem' as first required parameter, then provide parameters sepecific to the document for this function (do not provide document #uuid as parameter but title and context instead)",
        f"d) Generate only one test for each document, so the total number of function calls in this test list should be equal to the number of 'Document to be tested'",
        f"e) Your only source of data will be the Neo4j graph provide in the primitives. Do not use the date."
    ]

    del criteria_coach[criteria_to_remove - 1]

    for i in range(len(criteria_coach)):
        coach_text = "\n".join(criteria_coach[i])

    prompt_coach = [f"""
            ROLE: 
                You are a {role_priming} that defines tasks to {goal_definition} . 

            CONTEXT:
                JIRA anomalies are records of issues encountered in software development and project management processes. 
                These anomalies can include bugs, performance errors, connectivity problems, and other technical malfunctions. 
                Each anomaly is documented with specific details to facilitate its identification, analysis, and resolution.
                The objective of analyzing JIRA anomalies is to identify clusters of recurring problems, determine root causes,
                and recommend corrective actions to improve software quality, reduce costs, and minimize delays. 
                All of those anomalies are stored on a Neo4J's graph. Connection informations to this database will be communicated 
                to you via the primitives.{task_complexity}

                We aim to have a chat agent to help all the Jira's users at every stage to automate possible tasks and improve their productivity by solving easily issues. 
                Among other things, it should be able to :
                    {goal_function}
                

            TASK:
                Based on the current context, we need to code functions to help users to solve JIRA's anomalies by leveraging the graph database from Neo4J. This database will be your only source of information.
                First: propose  a plan to best support a crisis response / intervention from end-to-end starting when a new Jira anomalie is detected up to end of correction operation and capitalization.
                Second: reason and identify the next best task to be converted to code by a LLM and made available as a function to a chat, it should provide the best value to target while not being too complex to 
                be converted to code in one pass using a LLM.
                You should propose to choose between 2 different next best task. Task could be either the most frequent and globaly impactful task to be automated throughout plans, or the highest priority and impactful task at the current status of the JIRA's issue correction (if this status is not provided, consider we are the beginning of the operation).
                Task shouldn’t be too difficult to be converted into a single Python function for automation given available commands and learnt tasks to automate.
                Consider and re-use the already developped learnt tasks, do not propose the same task except if improving it can be done and is better than any other task.
                Detail this task into a specification checklist.
                Considering this checklist, end by listing any mandatory information for code implementation unkown by an LLM (e.g. GPT-4) that should be provided by a human or a search engine.

            INPUT:
                I will provide you if there are:
                - Learnt tasks available (with information gain between 0 (minimum) and 1 (maximum) on plan's titles, and contents): ...
                 {f'- Failed tasks to learn that are too hard: ...' if include_failed_tasks else ''}
                - Current status of examples of technical issues and solutions for the proposed next task will be tested on: ...
                You should tell me the next best function we should try to implement given your LLM knowledge, available learnt tasks, in order to solve problems to produce it.

                Here you can find a structure of an existing Fiche of Anomalie (FA):
                {structure_FA}

                Cypher request which was used to create the database:
                {cypher_request}

                
            CONSTRAINTS:
                You must follow the criteria below:
                {coach_text}
                 You should propose a plan to achieve this task by breaking it down as a tree-structure. The plan tree should be of depth {plan_depth}.
                Some specific commands available later for implementation in python of the task you will provide:
                {available_commands_detail} list of commands...


        You should only respond in the format as described below:

            RESPONSE FORMAT:
                1. Reasoning: Based on the information listed above, do reasoning about what the next task should be and why. Ensure it will minimize the distance to goal.
                2. Task: Next best task to develop.
                3. Specifications: present a tree-like structure of acceptance criteria, best strategies to compare, performance tips to beat a single LLM.
                4. Plan: Tree-structured plan of depth {plan_depth} breaking-down next best task into basic commands
                {test_example}
    """]

    coach_criterias = "\n".join(criteria_coach)

    prompt_coach = f"""
    CONTEXT:
    {coach_agent_role}.

    STATE OF THE ENVIRONMENT:
    {coach_user_input}.
    
    TASK DESCRIPTION:
    {coach_task_description}.
    
    CRITERIA:
    {coach_criterias}

    RESPONSE FORMAT:
    {coach_format_output}
    {few_shots if few_shots_tags else ""}
    """
    # Write the prompt in the file readed after by the coach
    with open("./prompts/identify_best_task.txt", "w") as f:
        f.write(prompt_coach[0])
    with open("./prompts/Anomalies/identify_best_task.txt", "w") as f: f.write(prompt_coach)

    # Define parameters for Coder

    example_result = trial.suggest_categorical("example_result", ["""
        🛠 Problem Details 🛠
        Title: no message
        Abstract: no message on jira between all of the collaborators
        Number: 35678
        Comment: very annoying

    🔍 Top 3 Similar Anomalies 🔍
        Anomaly 1 :
            Title: JIRA displays an erroneous error message when user votes
            Description: At 11:17 today, on jira.atlassian.com, I tried to vote on [ANSWERS-182] and got an error message "The JIRA server was contacted but has returned an error response. We are unsure of the result of this operation". Despite this, my vote was still counted. This made me confused.
            Similarity Score: 0.77

        Anomaly 2:
            Title: Raising a jira from a defect is broken
            Description: When you create a jira from a defect in a review, an error message is displayed:
                        http://img.skitch.com/20101001-rmy8unkkgae6j2i5ytct4h69ij.jpg
                        The jira is created however..
            Similarity Score: 0.77

        Anomaly 3:
            Title: Message 'taken' notion is per message. But should be per message per queue
            Description: Currently doing msg.take() in the broker will break pub / sub. As a message is referenced between queues not duplicated per queue. So marking message as taken in one queue will result in the message not being sent from any other queue.
            Similarity Score: 0.76

    💡 Recommendation to solve the issue 💡
    Based on the described problem of "no message on jira between all of the collaborators," it seems the communication module, or comment section in the JIRA platform isn't working correctly. 

    Comparing it with the similar anomalies:
        Anomaly 1: This indicates that there might be a problem with the system's feedback/display mechanism as the user still saw an error message despite the vote being counted.
        Anomaly 2: Here, even though an error message is displayed, the Jira issue is still created. Again, it suggests an issue with the output display mechanism, similar to the main problem where messages aren't showing up.
        Anomaly 3: This issue revolves around the queue function of the software and might not be related to the messaging problem indicated in the user's issue.

    Solutions:
        Check User Permissions: First, ensure that all collaborators have appropriate permissions to send and receive messages. Insufficient permissions or roles could result in no messages being passed between collaborators.
        Review Notification Preferences: Check to ensure that the notification settings are properly configured. Sometimes, messaging problems can occur if users have turned off their notifications or updated their notification preferences.
        Software Updates: Make sure that all users are using the most updated version of the JIRA software. There might be a software bug or issue that's been resolved in a newer update which could be causing the messaging anomaly.
        Clearing Cache & Cookies: Sometimes, cache buildup in the browser can cause certain functionalities to stop working. Guiding users to clear their cache or try using the software in an incognito window or a different browser might solve the problem.
        Reach out to Support: If the problem persists despite trying these solutions, it may be best to reach out to Atlassian Support.
        Remember to follow up with users to ensure the anomaly has been resolved and users can communicate in the software without issue."""])

    prompt_template = trial.suggest_categorical("prompt_template", ["Extensive", "Minimal"])
    libraries_restriction = trial.suggest_categorical("libraries_restriction", [
        "BeautifulSoap, RegEx, Sklearn, Huggingface, Langchain, Voyager",
        "Numpy, Pandas, Scikit-learn",
        "TensorFlow, PyTorch, Transformers, SpaCy"
    ])
    bot_function_specifications = trial.suggest_categorical("bot_function_specifications", [
        "Manipulate document sections: bot.create_and_add_section_then_return_id(title: str, content: str, section_id: int = None, parent_id: int = None) -> int, bot.get_all_sections() -> List[Section], bot.get_sections(ids: List[int]) -> List[Section], bot.edit_section(section_id: int, new_content: str = None, new_title: str = None, new_parent_id: int = None) -> bool, bot.remove_section(section_id: int) -> bool, bot.swap_sections(section_id_1: int, section_id_2: int) -> bool",
        "Manipulate document resources: bot.add_or_update_results_in_resources(results, metadatas_to_add:dict=None, store_linked_document_content:bool=False), bot.add_or_update_result_in_resources(metadatas:dict, name:str=None, content:dict=None, link:str=None, store_linked_document_content:bool=False), bot.get_all_resources(self) -> List[Dict[str, Any]], bot.semantic_search_resources(query_texts, n_results=10), bot.add_or_update_results_in_resources(results, metadatas:dict=None, store_linked_document_content:bool=False), bot.get_and_store_link_content(link:str=None, parent_id=None, chaining:bool=True), bot.remove_resource(resource_id)"
    ])
    reasoning_depth = trial.suggest_int("reasoning_depth", 1, 3)
    modularity = trial.suggest_categorical("modularity", [
        "None",
        "Helper functions for common tasks",
        "Modular classes for related functions",
        "Parameterize all varying data"
    ])
    handle_previous_attempts = trial.suggest_categorical("handle_previous_attempts", [True, False])
    instruction_to_remove = trial.suggest_int("instruction_to_remove", 1, 14)
    # prompt_template = trial.suggest_categorical("prompt_template", ["Extensive", "Minimal"])
    # libraries_restriction = trial.suggest_categorical("libraries_restriction", [
    #     "BeautifulSoap, RegEx, Sklearn, Huggingface, Langchain, Voyager",
    #     "Numpy, Pandas, Matplotlib, Scikit-learn",
    #     "TensorFlow, PyTorch, Transformers, SpaCy"
    # ])
    # bot_function_specifications = trial.suggest_categorical("bot_function_specifications", [
    #     "Manipulate document sections: bot.create_and_add_section_then_return_id(title: str, content: str, section_id: int = None, parent_id: int = None) -> int, bot.get_all_sections() -> List[Section], bot.get_sections(ids: List[int]) -> List[Section], bot.edit_section(section_id: int, new_content: str = None, new_title: str = None, new_parent_id: int = None) -> bool, bot.remove_section(section_id: int) -> bool, bot.swap_sections(section_id_1: int, section_id_2: int) -> bool",
    #     "Manipulate document resources: bot.add_or_update_results_in_resources(results, metadatas_to_add:dict=None, store_linked_document_content:bool=False), bot.add_or_update_result_in_resources(metadatas:dict, name:str=None, content:dict=None, link:str=None, store_linked_document_content:bool=False), bot.get_all_resources(self) -> List[Dict[str, Any]], bot.semantic_search_resources(query_texts, n_results=10), bot.add_or_update_results_in_resources(results, metadatas:dict=None, store_linked_document_content:bool=False), bot.get_and_store_link_content(link:str=None, parent_id=None, chaining:bool=True), bot.remove_resource(resource_id)"
    # ])
    # reasoning_depth = trial.suggest_int("reasoning_depth", 1, 3)
    # modularity = trial.suggest_categorical("modularity", [
    #     "None",
    #     "Helper functions for common tasks",
    #     "Modular classes for related functions",
    #     "Parameterize all varying data"
    # ])
    # handle_previous_attempts = trial.suggest_categorical("handle_previous_attempts", [True, False])
    # instruction_to_remove = trial.suggest_int("instruction_to_remove", 1, 14)

    modularity_final = f"16) Ensure that the generated code adheres to principles of reusability and modularity as {modularity}." if modularity != "None" else "3) Ensure that the generated code adheres to principles of reusability."


    # Construct the prompt based on the suggested parameters
    criteria_coder = [
        f"1) First part of the function should be dedicated for the Neo4J graph connection (which is already created). Here you can find the informations:",
        f"URI='bolt://127.0.0.1:7687'",
        f"AUTH = auth=('neo4j','password')",
        f"2) Ensure that the generated code adheres to principles of reusability and modularity. Specifically, functions should not hard-code strings, variables, or parameters that make them context-specific. Instead, any data or parameters of helpers functions that can vary should be passed as arguments to the functions, ensuring that the functions can be reused in different contexts or with different data without requiring modifications to the code itself. This ensures that the code is adaptable and can be utilized in various scenarios, enhancing its utility and longevity.",
        f"3) Call existing functions as much as possible from the primitives folder.",
        f"4) Name your function in a meaningful way (can infer the task from the name).",
        f"5) Your function will be reused for building more complex functions. Therefore, you should make it generic and reusable. Avoid to include specific query or information in the function instead of using it as an argument.",
        f"6) Anything defined outside a function will be ignored, define all your variables and classes inside your functions.",
        f"7) Ensure that your code is fully executable, it is not a skeleton and does not contain placeholders, unimplemented sections, or comments indicating future work (e.g., TODO, pass, '....', etc.). All functions and logic must be complete and runnable to facilitate immediate use and testing.",
        f"8) Do not write infinite loops or recursive functions. Do not use Mathplotlib and clustering methods. Find the clothest ways on the graph thanks to embeddings distance method.",
        f"9) Name your function in a meaningful way (can infer the task from the name).",
        f"10) Any packages/libraries used by the function should be imported inside the function (it will be ignored if imported outside).",
        f"11) dont try to create new tab (columns and rows), or time_range (e.g 'previous 30 days' or WHERE fa.fan_date_declaration >= date('time_range'))",
        f"12) Do not take into account the fan_date_declaration, it does not matter for the code.",
        f"13) Adapt the exact same code as the primitives  (generateNeo4J.py) is their is nothing in the succeded functions",
        f"14) Avoid those errors:"
        "- Error: 'SynthesisManager' object has no attribute 'min_plan_cosine_similarity'"
        "- analyze_similar_anomalies() got an unexpected keyword argument 'problem'"
        "TypeError: string indices must be integers, not 'str'",
        f"15) The code should first connect to the Neo4J graph, then implement the task, and finally return the result. Ensure that the code is modular and reusable, adhering to principles of reusability and modularity. ",
        {modularity_final},
        f"17) Ensure to use the primitives functions",
        f"18) Use openai.chat.completion.create to execute the code and provide the result instead of openai.ChatCompletion.create. openai_api_key should be load from .env file with os.getenv('OPENAI_API_KEY')",
        f"19) Just use the generated recommandation instead of the creation of create a function `create_corrective_action_plan` ",
        f"20) Do not use toLower in the cypther request ou WHERE clause",
    ]
    # modularity_final = f"3) Ensure that the generated code adheres to principles of reusability and modularity as {modularity}." if modularity != "None" else "3) Ensure that the generated code adheres to principles of reusability." 

    # # Construct the prompt based on the suggested parameters
    # criteria_coder = [
    #      f"1) Reason in {reasoning_depth} steps to find out the best task to minimize distance to goal.",
    #     "2) Write a function taking the bot as the first parameter which is the shared object of document content and resources (it is an instance of the class SynthesisManager).",
    #     f"{modularity_final} Specifically, functions should not hard-code strings, variables, or parameters that make them context-specific. Instead, any data or parameters of helpers functions that can vary should be passed as arguments to the functions, ensuring that the functions can be reused in different contexts or with different data without requiring modifications to the code itself. This ensures that the code is adaptable and can be utilized in various scenarios, enhancing its utility and longevity.",
    #     "4) Call existing functions as much as possible.",
    #     "5) Your function will be reused for building more complex functions. Therefore, you should make it generic and reusable. Avoid to include specific query or information in the function instead of using it as an argument.",
    #     "6) Anything defined outside a function will be ignored, define all your variables and classes inside your functions.",
    #     "7) Ensure that your code is fully executable, it is not a skeleton and does not contain placeholders, unimplemented sections, or comments indicating future work (e.g., TODO, pass, '....', etc.). All functions and logic must be complete and runnable to facilitate immediate use and testing.",
    #     "8) Do not write infinite loops or recursive functions.",
    #     "9) Name your function in a meaningful way (can infer the task from the name).",
    #     "10) Any packages/libraries used by the function should be imported inside the function (it will be ignored if imported outside)",
    #     "11) Success of the task output is evaluated by analyzing the new state of resources and sections, and events.",
    #     "12) If some content is generated for the task and that its quality impacts task's success, you must log an event using `bot.add_event(event: str, data: dict)` to enable the critic agent to evaluate this content it through the events' list. The `event` should describe the type of event to help the critic to understand what to check, and `data` should include all information to be analyzed. If this event is in a loop/for, just fully log the event 1 time to avoid too much logging and allow sampling evaluation.",
    #     f"13) Your function should include appropriate modification to resources and sections to measure task success. Main functions are:\n- class Section(section_id: int, title: str, content: str, parent_id: int)\n- {bot_function_specifications}",
    #     "14) Before the return of the main function, ensure to store your results or text generated in resources or sections which are the only permanent storage. Also, ensure that results are returned for future reuse of the function."
    # ]

    del criteria_coder[instruction_to_remove - 1]
    for i in range(len(criteria_coder)):
        coder_text = "\n".join(criteria_coder[i])
    # del criteria_coder[instruction_to_remove - 1]
    # coder_text = "\n".join(criteria_coder)

    prompt_coder = [f"""
    You are a helpful assistant that writes Python code to be executed using a restricted list of packages ({libraries_restriction}) to complete the task specified by me.
    At each round of conversation, I will give you:
    - Reasoning: explanation of the task chosen...
    - Task: ...
    - Tests: tests that will be done on target document

    CURRENT STATE OF THE ENVIRONMENT USED TO TEST TASK
        Document #xxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx: 
        {
    'id: ...'
    'title: ...'
    'abstract: ...'
    'comment: ...'
    'target_file_path: ...'
    }

    - General code for re-use or demonstration purpose: ...
    - Code from the last round with attempts to implement task with its performance (e.g. 'sections titles progress': x, 'sections content progress': y): ...
    - Execution error:  You should use openai.chat.completions.create to execute the code and provide the result instead of openai.ChatCompletion.create.
                        openai_api_key should be load from .env file with os.getenv("OPENAI_API_KEY")

    Here you can find a structure of an existing Fiche of Anomalie (FA):
        {structure_FA}

    Cypher request which was used to create the database: 
        {cypher_request}
               
    You should then respond to me with:
        - Reasoning: How to best implement the plan with no errors and maximum performance towards the goal ?
        - Code:
        {coder_text}

        

    
    RESPONSE FORMAT (You should only respond in the format as described below, and follow the example provided):
        ```python    
                main function after the helper functions
                def your_main_function_name(args...):
                # detailed content of the function...
        ``` 

    EXAMPLES of code execution: {example_result} 

    TESTS: 
        # document #72dc469b-63f8-4751-aab5-6db3d16fca3c usage test:
        your_main_function_name(args specific to document #72dc469b-63f8-4751-aab5-6db3d16fca3c...)

        # document #c0533337-1c5e-4091-ba7d-061ae409cda4 usage test:
        your_main_function_name(args specific to document #c0533337-1c5e-4091-ba7d-061ae409cda4...)
    """]

    # Write the prompt in the file readed after by the coder
    with open("./prompts/code_task.txt", "w") as f:
        f.write(prompt_coder[0])
    # prompt_coder = f"""
    # You are a helpful assistant that writes Python code to be executed using a restricted list of packages ({libraries_restriction}) to complete the task specified by me.
    # At each round of conversation, I will give you:
    # - Reasoning: explanation of the task chosen...
    # - Task: ...
    # - Plan: ...
    # - Tests: tests that will be done on target document
    # CURRENT STATE OF THE ENVIRONMENT USED TO TEST TASK
    # Document #.... : 1.title: ...; 2. abstract: ...; 3. current table of content; 4. current resources; 5. sections titles progress; 6. sections content progress; 7. events counted
    # ...
    # - General code for re-use or demonstration purpose: ...
    # - Code from the last round with attempts to implement task with its performance (e.g. 'sections titles progress': x, 'sections content progress': y): ...
    # - Execution error: ...
    # You should then respond to me with:
    # - Reasoning: How to best implement the plan with no errors and maximum performance towards the goal ?
    # - Code:
    # {coder_text}
    # RESPONSE FORMAT (You should only respond in the format as described below, and follow the example provided):
    # Reasoning: ...
    # Code:
    # ```python
    # # helper functions (only if needed, try to avoid them)
    # # detailed content of the function...
    # # main function after the helper functions
    # def your_main_function_name(bot):
    #     title = bot.document.title
    #     abstract = bot.document.context
    #     # detailed content of the function...
    # ```
    #     """
    # # Write the prompt in the file readed after by the coder
    # with open("./prompts/IR_CPS_TechSynthesis/code_task.txt", "w") as f:
    #         f.write(prompt_coder)

    # Define parameters for the Critic

    # prompt_template = trial.suggest_categorical("prompt_template", ["Extensive", "Minimal"])
    # reasoning_depth = trial.suggest_int("reasoning_depth", 1, 3)
    # detailed_explanation = trial.suggest_categorical("detailed_explanation", [True, False])
    # critic_to_remove = trial.suggest_int("critic_to_remove", 1, 3)

    # # Construct the prompt based on the suggested parameters
    # criteria_critic = [
    #     f"Reasoning: Based on the information I listed above, do a {reasoning_depth} step reasoning to evaluate if the code implementation and execution is aligned with the task goal to decide if it is a success.",
    #     "Success: write 'True' if code is a success, 'False' otherwise",
    #     f"Explain: explain in detail your evaluation of your success evaluation." if detailed_explanation else ""
    # ]

    # del criteria_critic[critic_to_remove - 1]
    # critic_text = "\n".join(criteria_critic)

    # if prompt_template == "Extensive":
    #     prompt_critic = f"""
    #     You are a Python expert and domain expert in the field of the task, you should validate the Python code provided and its result regarding the code implementing the task and feedback.
    #     I will provide you:
    #     TASK: {{task}}
    #     CODE: {{code}}
    #     Some additional information to evaluate code: {{runtime_errors}}
    #     Execution result returned by exec command of code provided: {{exec_result}}
    #     RESPONSE FORMAT: you should only respond in the format as described below:
    #     {critic_text}
    #     EXAMPLES:
    #     Reasoning: The initial task was to list all GPS points of vessels in the zone. The code is aligned with this task, it ran without errors, your confirmed what I could not check.
    #     Success: "True"
    #     Explain: code generated result expected by task without errors.
    #     """
    # else:
    #     prompt_critic = f"""
    #     You are a Python expert and domain expert in the field of the task, you should validate the Python code provided and its result regarding the code implementing the task and feedback.
    #     I will provide you:
    #     TASK: {{task}}
    #     CODE: {{code}}
    #     Some additional information to evaluate code: {{runtime_errors}}
    #     Execution result returned by exec command of code provided: {{exec_result}}
    #     RESPONSE FORMAT: you should only respond in the format as described below:
    #     {critic_text}
    #     """

    #     with open("./prompts/validate_code.txt", "w") as f:
    #         f.write(prompt_critic)

    # Save the parameters chosen by the trial
    with open(f"Optuna_results/{timestamp_exp}.txt", "a") as f:
        f.write(f"Trial: {trial.number}\nPrompt Coach Chosen: \n{prompt_coach}")


    perf = launch_run(
        default_llm_key="default_llm",
        premium_llm_key="premium_llm",
        problem_prompts_subdir="Anomalies",
        max_coding_attempts=2,
        max_execution_time=900,
        model_choice={"coach": "default_llm", "coder": "premium_llm", "critic": "default_llm", "capitalizer": "default_llm"},
        optuna_opti="coach",
        params_user_message=criteria_user_message
    )

    with open(f"Optuna_results/{timestamp_exp}.txt", "a") as f:
        f.write(f"Performance: {perf}\n\n")
    return perf


if __name__ == "__main__":
    global_main(objective, "anomalies")