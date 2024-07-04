from learn import run_4agents_learning_loop, create_Nmajority_chain, EnvironmentManager
from langchain_openai import ChatOpenAI
import optuna as opt

def objective(trial):

    modelVariation = trial.suggest_int("model_variation", 0, 3)

    # Define parameters for the coach

    role_priming = trial.suggest_categorical("role_priming", [
        "research assistant",
        "AI coach",
        "task optimizer",
        "technical synthesis expert",
        "knowledge engineer",
        "content curator"
    ])
    goal_description = trial.suggest_categorical("goal_description", [
        "produce high quality technical synthesis",
        "generate state-of-the-art research survey paper",
        "create comprehensive Wikipedia-like article",
        "develop patent-quality technical document",
        "formulate detailed technical guides",
        "synthesize research findings effectively"
    ])
    task_format = trial.suggest_categorical("task_format", [
        "[verb] [quantity if applicable] [object] [tools] [detailed instructions and parameters]",
        "[action] [target] using [method] with [specifications]",
        "[operation] on [subject] utilizing [resources] following [guidelines]",
        "[do] [what] [how] [with what] [detailed instructions]"
    ])
    reasoning_steps = trial.suggest_int("reasoning_steps", 3, 10)
    specification_depth = trial.suggest_int("specification_depth", 2, 5)
    use_examples = trial.suggest_categorical("use_examples", [True, False])
    auto_add_test_to_prompt_answer = trial.suggest_categorical("auto_add_test_to_prompt_answer", [True, False])
    include_failed_tasks = trial.suggest_categorical("include_failed_tasks", [True, False])
    plan_depth = trial.suggest_int("plan_depth", 2, 4)
    available_commands_detail = trial.suggest_categorical("available_commands_detail", [
        "minimal",
        "moderate",
        "comprehensive",
        "detailed"
    ])
    task_complexity = trial.suggest_categorical("task_complexity", [
        "Task shouldn’t be too difficult to convert into Python code given available commands and learnt tasks.",
        "Task should balance complexity and feasibility for effective implementation.",
        "Task should challenge the LLM while remaining solvable with available resources.",
        "Task should involve multiple steps that require coordination among different functions.",
        "Task should leverage advanced features of the LLM to achieve superior results.",
        "Task should be modular, allowing parts of the solution to be reused in other contexts.",
        "Task should be scalable, capable of being applied to larger datasets or more complex scenarios."
    ])
    criteria_to_remove = trial.suggest_int("criteria_to_remove", 1, 7)
    
    # Construct the prompt based on the suggested parameters
    criteria_coach = [
        f"1) Reason in {reasoning_steps} steps to find out the best task to minimize distance to goal.",
        f'2) Task should be written in the form of "{task_format}"',
        "3) Task will be converted into Python code given available commands, learnt tasks, use of LLM if required.",
        f"4) Task should be novel compared to learnt {f'and failed ' if include_failed_tasks else ''}tasks.",
        f"5) Develop key minimal elements of specification (acceptance criteria, best strategies to compare, performance tips to beat a LLM) to successfully prompt a coder agent to generate code implementing the task while minimizing distance to goal. Organize the requirements with clear indexing to a depth of {specification_depth}.",
        f"6) Tasks provided should be generic, not specific to given examples{', so the reasoning can mention examples but proposed task and plan should not mention any information related to examples' if use_examples else ''}.",
        f"7) After proposing the task, {'you should provide a test case of the function corresponding to this task for each example using the one-liner function call format.' if auto_add_test_to_prompt_answer else 'do not add test cases automatically.'}"
    ]
    
    del criteria_coach[criteria_to_remove - 1]
    coach_text = "\n".join(criteria_coach)
    
    prompt_coach = f"""
    You are a {role_priming} that defines tasks to {goal_description} given a [Title] and an [Abstract].
    Each task you propose will be prompted to a language model which will try to convert it into Python functions.{task_complexity}
    If the code is successful and gains in technical synthesis above a pre-defined threshold, this learnt task is made available to the next learning iteration.
    I will provide you:
    - Learnt tasks available (with information gain between 0 and 1 on plan's titles, and contents): ...
    {f'- Failed tasks to learn that are too hard: ...' if include_failed_tasks else ''}
    - Current status of examples of technical synthesis the proposed next task will be tested on: ...
    You should tell me the next best novel task we should try to implement given your LLM knowledge, available learnt tasks, in order to maximize synthesis generated quality, length and format, and speed to produce it.
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
    {'5. Tests: Provide test cases using the one-liner function call format.' if auto_add_test_to_prompt_answer else ''}
    """
    # Write the prompt in the file readed after by the coach
    with open("./prompts/IR_CPS_TechSynthesis/identify_best_task.txt", "w") as f:
            f.write(prompt_coach)

    # Define parameters for Coder

    prompt_template = trial.suggest_categorical("prompt_template", ["Extensive", "Minimal"])
    libraries_restriction = trial.suggest_categorical("libraries_restriction", [
        "BeautifulSoap, RegEx, Sklearn, Huggingface, Langchain, Voyager",
        "Numpy, Pandas, Matplotlib, Scikit-learn",
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

    modularity_final = f"3) Ensure that the generated code adheres to principles of reusability and modularity as {modularity}." if modularity != "None" else "3) Ensure that the generated code adheres to principles of reusability." 
    
    # Construct the prompt based on the suggested parameters
    criteria_coder = [
         f"1) Reason in {reasoning_depth} steps to find out the best task to minimize distance to goal.",
        "2) Write a function taking the bot as the first parameter which is the shared object of document content and resources (it is an instance of the class SynthesisManager).",
        f"{modularity_final} Specifically, functions should not hard-code strings, variables, or parameters that make them context-specific. Instead, any data or parameters of helpers functions that can vary should be passed as arguments to the functions, ensuring that the functions can be reused in different contexts or with different data without requiring modifications to the code itself. This ensures that the code is adaptable and can be utilized in various scenarios, enhancing its utility and longevity.",
        "4) Call existing functions as much as possible.",
        "5) Your function will be reused for building more complex functions. Therefore, you should make it generic and reusable. Avoid to include specific query or information in the function instead of using it as an argument.",
        "6) Anything defined outside a function will be ignored, define all your variables and classes inside your functions.",
        "7) Ensure that your code is fully executable, it is not a skeleton and does not contain placeholders, unimplemented sections, or comments indicating future work (e.g., TODO, pass, '....', etc.). All functions and logic must be complete and runnable to facilitate immediate use and testing.",
        "8) Do not write infinite loops or recursive functions.",
        "9) Name your function in a meaningful way (can infer the task from the name).",
        "10) Any packages/libraries used by the function should be imported inside the function (it will be ignored if imported outside)",
        "11) Success of the task output is evaluated by analyzing the new state of resources and sections, and events.",
        "12) If some content is generated for the task and that its quality impacts task's success, you must log an event using `bot.add_event(event: str, data: dict)` to enable the critic agent to evaluate this content it through the events' list. The `event` should describe the type of event to help the critic to understand what to check, and `data` should include all information to be analyzed. If this event is in a loop/for, just fully log the event 1 time to avoid too much logging and allow sampling evaluation.",
        f"13) Your function should include appropriate modification to resources and sections to measure task success. Main functions are:\n- class Section(section_id: int, title: str, content: str, parent_id: int)\n- {bot_function_specifications}",
        "14) Before the return of the main function, ensure to store your results or text generated in resources or sections which are the only permanent storage. Also, ensure that results are returned for future reuse of the function."
    ]
    
    del criteria_coder[instruction_to_remove - 1]
    coder_text = "\n".join(criteria_coder)
    
    if prompt_template == "Extensive":
        prompt_coder = f"""
        You are a helpful assistant that writes Python code to be executed using a restricted list of packages ({libraries_restriction}) to complete the task specified by me.
        At each round of conversation, I will give you:
        - Reasoning: explanation of the task chosen...
        - Task: ...
        - Plan: ...
        - Tests: tests that will be done on target document
        CURRENT STATE OF THE ENVIRONMENT USED TO TEST TASK
        Document #.... : 1.title: ...; 2. abstract: ...; 3. current table of content; 4. current resources; 5. sections titles progress; 6. sections content progress; 7. events counted
        ...
        - General code for re-use or demonstration purpose: ...
        - Code from the last round with attempts to implement task with its performance (e.g. 'sections titles progress': x, 'sections content progress': y): ...
        - Execution error: ...
        You should then respond to me with:
        - Reasoning: How to best implement the plan with no errors and maximum performance towards the goal ?
        - Code:
        {coder_text}
        RESPONSE FORMAT (You should only respond in the format as described below, and follow the example provided):
        Reasoning: ...
        Code:
        ```python
        # helper functions (only if needed, try to avoid them)
        # detailed content of the function...
        # main function after the helper functions
        def your_main_function_name(bot):
            title = bot.document.title
            abstract = bot.document.context
            # detailed content of the function...
        ```
        """
    else:
        prompt_coder = f"""
        Write a Python function to accomplish the following TASK DEFINITION.
        If several tasks are defined (e.g. Task 1, Task 2...), choose only one task to code, reason and choose the task you have the most confidence to fully implement with success.
        This function should be modular and reusable, embed all required packages imports inside function, no code should be outside functions, avoid any recursion and infinite loop.
        Function should return all information produced in string or json format, the success of this TASK will be evaluated on this output so ensure this output is well aligned with the expected task output.
        End your code with 1 liner code calling the function to execute the task and affect it to the 'result' variable - e.g. result = your_main_function(....).
        Code should be fully functional, do not include placeholders or pass or similar.
        {f'PREVIOUS CODE ATTEMPT: ...' if handle_previous_attempts else ''}
        """
    # Write the prompt in the file readed after by the coder
    with open("./prompts/IR_CPS_TechSynthesis/code_task.txt", "w") as f:
            f.write(prompt_coder)

    # Define parameters for the Critic

    prompt_template = trial.suggest_categorical("prompt_template", ["Extensive", "Minimal"])
    reasoning_depth = trial.suggest_int("reasoning_depth", 1, 3)
    detailed_explanation = trial.suggest_categorical("detailed_explanation", [True, False])
    critic_to_remove = trial.suggest_int("critic_to_remove", 1, 3)
    
    # Construct the prompt based on the suggested parameters
    criteria_critic = [
        f"Reasoning: Based on the information I listed above, do a {reasoning_depth} step reasoning to evaluate if the code implementation and execution is aligned with the task goal to decide if it is a success.",
        "Success: write 'True' if code is a success, 'False' otherwise",
        f"Explain: explain in detail your evaluation of your success evaluation." if detailed_explanation else ""
    ]
    
    del criteria_critic[critic_to_remove - 1]
    critic_text = "\n".join(criteria_critic)
    
    if prompt_template == "Extensive":
        prompt_critic = f"""
        You are a Python expert and domain expert in the field of the task, you should validate the Python code provided and its result regarding the code implementing the task and feedback.
        I will provide you:
        TASK: {{task}}
        CODE: {{code}}
        Some additional information to evaluate code: {{runtime_errors}}
        Execution result returned by exec command of code provided: {{exec_result}}
        RESPONSE FORMAT: you should only respond in the format as described below:
        {critic_text}
        EXAMPLES:
        Reasoning: The initial task was to list all GPS points of vessels in the zone. The code is aligned with this task, it ran without errors, your confirmed what I could not check.
        Success: "True"
        Explain: code generated result expected by task without errors.
        """
    else:
        prompt_critic = f"""
        You are a Python expert and domain expert in the field of the task, you should validate the Python code provided and its result regarding the code implementing the task and feedback.
        I will provide you:
        TASK: {{task}}
        CODE: {{code}}
        Some additional information to evaluate code: {{runtime_errors}}
        Execution result returned by exec command of code provided: {{exec_result}}
        RESPONSE FORMAT: you should only respond in the format as described below:
        {critic_text}
        """

        with open("./prompts/validate_code.txt", "w") as f:
            f.write(prompt_critic)

    # Save the parameters chosen by the trial
    with open("Optuna_results.txt", "a") as f:
        f.write(f"Trial: {trial.number}\nPrompt Coach Chosen: \n{prompt_coach}\nPrompt Coder Chosen : \n{prompt_coder}\nPrompt Critic Chosen : \n{prompt_critic}\nModel Chosen: {modelVariation}\n")

    # Run the learning loop
    perf = run_4agents_learning_loop(default_llm_key="default_llm",
                                premium_llm_key="premium_llm",
                                llmORchains_list=llmORchains_list,
                                test_environments=envs,
                                manual_validation_to_capitalize=False,
                                problem_prompts_subdir='IR_CPS_TechSynthesis', 
                                max_coding_attempts=4,
                                include_code=False,
                                selected_successful_functions=[],
                                selected_failed_functions=[],
                                agtask_premium_llm_by_default=True,
                                agtask_skip_rounds=0,
                                agcoding_skip_rounds=0,
                                agvalidation_skip_rounds=0,
                                agcapitalize_skip_rounds=0,
                                model_choice=modelVariation,
                                optuna_opti="Coach")
    
    with open("Optuna_results.txt", "a") as f:
        f.write(f"Performance: {perf}\n\n")
    
    return perf

if __name__ == "__main__":
    # Initialize the default and premium LLMs
    #default_llm = ChatOpenAI(model_name="gpt-3.5-turbo-1106") # gpt-4-1106-preview gpt-3.5-turbo-1106 model_name=model_name, temperature=temperature, request_timeout=request_timout
    #default_llm = create_Nmajority_chain(num_models=3)
    #premium_llm = ChatOpenAI(model_name="gpt-4o") # gpt-4-1106-preview gpt-3.5-turbo-1106 model_name=model_name, temperature=temperature, request_timeout=request_timout
    llmORchains_list = {
        "default_llm": ChatOpenAI(model_name="gpt-3.5-turbo-1106"),
        "premium_llm": ChatOpenAI(model_name="gpt-3.5-turbo-1106"),
        "3_majority_chain": create_Nmajority_chain(num_models=3),
        "10_majority_chain": create_Nmajority_chain(num_models=10)
    }

    # Set the documents to test/validate as a list of environments
    documents=[{ 'id':"cf0d353c-b43b-4a79-88f9-42c2c84cf75e",
                'title':"Complex QA and language models hybrid architectures, Survey",
            'context':"This paper reviews the state-of-the-art of language models architectures and strategies for 'complex' question-answering (QA, CQA, CPS) with a focus on hybridization. Large Language Models (LLM) are good at leveraging public data on standard problems but once you want to tackle more specific complex questions or problems (e.g. How does the concept of personal freedom vary between different cultures ? What is the best mix of power generation methods to reduce climate change ?) you may need specific architecture, knowledge, skills, methods, sensitive data protection, explainability, human approval and versatile feedback... Recent projects like ChatGPT and GALACTICA have allowed non-specialists to grasp the great potential as well as the equally strong limitations of LLM in complex QA. In this paper, we start by reviewing required skills and evaluation techniques. We integrate findings from the robust community edited research papers BIG, BLOOM and HELM which open source, benchmark and analyze limits and challenges of LLM in terms of tasks complexity and strict evaluation on accuracy (e.g. fairness, robustness, toxicity, ...) as a baseline. We discuss some challenges associated with complex QA, including domain adaptation, decomposition and efficient multi-step QA, long form and non-factoid QA, safety and multi-sensitivity data protection, multimodal search, hallucinations, explainability and truthfulness, temporal reasoning. We analyze current solutions and promising research trends, using elements such as: hybrid LLM architectural patterns, training and prompting strategies, active human reinforcement learning supervised with AI, neuro-symbolic and structured knowledge grounding, program synthesis, iterated decomposition and others.",
            'target_file_path': "env/IR_CPS_TechSynthesis/document_embedding_analysis/output/arxiv/Complex QA and language models hybrid architectures Survey.json"},
            { 'id':"42252c6c-12f3-4edf-9045-8acd69bc3356",
                'title':"Macroeconomic Effects of Inflation Targeting A Survey of the Empirical  Literature",
            'context':"This paper surveys the empirical literature of inflation targeting. The main findings from our review are the following: there is robust empirical evidence that larger and more developed countries are more likely to adopt the IT regime; the introduction of this regime is conditional on previous disinflation, greater exchange rate flexibility, central bank independence, and higher level of financial development; the empirical evidence has failed to provide convincing evidence that IT itself may serve as an effective tool for stabilizing inflation expectations and for reducing inflation persistence; the empirical research focused on advanced economies has failed to provide convincing evidence on the beneficial effects of IT on inflation performance, while there is some evidence that the gains from the IT regime may have been more prevalent in the emerging market economies; there is not convincing evidence that IT is associated with either higher output growth or lower output variability; the empirical research suggests that IT may have differential effects on exchange-rate volatility in advanced economies versus EMEs; although the empirical evidence on the impact of IT on fiscal policy is quite limited, it supports the idea that IT indeed improves fiscal discipline; the empirical support to the proposition that IT is associated with lower disinflation costs seems to be rather weak. Therefore, the accumulated empirical literature implies that IT does not produce superior macroeconomic benefits in comparison with the alternative monetary strategies or, at most, they are quite modest.",
            'target_file_path': "env/IR_CPS_TechSynthesis/document_embedding_analysis/output/arxiv/Macroeconomic Effects of Inflation Targeting A Survey of the Empirical  Literature.json"}]
    envs = []
    for doc in documents:
        env = EnvironmentManager(env_type="techsynthesis", title=doc['title'], context=doc['context'], target_file_path=doc['target_file_path'], id=doc['id']).get_environment()
        envs.append(env)

    with open("Optuna_results.txt", "w") as f:
        f.write("")

    study = opt.create_study(direction="maximize")
    study.optimize(objective, n_trials=10)
