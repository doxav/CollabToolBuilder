import os
import time
import learn
from langchain_openai import ChatOpenAI
import optuna as opt
from config import MODELS_CONFIG_LIST


def objective(trial):
    # Definition of the coach's prompt
    coach_agent_role = trial.suggest_categorical("role_priming", ["In the context of a system generating efficient state-of-the-art research survey papers on a given subject, you are a researcher expert advising on the next best task to develop. This system employs a hybrid approach, leveraging both LLM capabilities and traditional algorithmic processing.",
                                                                  "You are an AI coach"])
    # coach_user_input_failed_tasks = trial.suggest_categorical("coach_user_input_failed_tasks", [True, False])
    coach_user_input = "I will provide you:\n- Learnt tasks available (with information gain between 0 and 1 on plan's titles, and contents): ...\n- Failed tasks to learn that are too hard to code: ...\n- Current status of examples of technical synthesis the proposed next task will be tested on: ..."

    coach_task_description = trial.suggest_categorical("task_description", [
        "You define best next task to generate state-of-the-art research survey paper given a [Title] and an [Abstract]. Each task you propose will be prompted to a language model which will try to convert it into Python functions. If the code is successful and gains in technical synthesis above a pre-defined threshold, this learnt task is made available to the next learning iteration.",
        "Define the next best task to generate a state-of-the-art research survey paper given a document object (containing target title and abstract/context in its properties). Each task you propose will be converted into Python functions, potentially including LLM calls when appropriate. If the implementation is successful and gains in technical synthesis above a pre-defined threshold, this learnt task is made available to the next learning iteration."])

    criteria_coach = []
    criteria_coach.append(f"{len(criteria_coach) + 1}) Task should challenge the LLM while remaining solvable with available resources.")
    criteria_coach.append(f"{len(criteria_coach) + 1}) Reason in 4 steps to find out the best task to minimize distance to goal.")
    criteria_coach.append(f"{len(criteria_coach) + 1}) Task should be written in the form of \"Task should be written in the form of [verb] [quantity if applicable] [object] [tools] [detailed instructions and parameters]\"")
    criteria_coach.append(f"{len(criteria_coach) + 1}) Task will be converted into Python code given available commands, learnt tasks. The Python code could call LLM when required.")
    criteria_coach.append(f"{len(criteria_coach) + 1}) Task should be novel compared to learnt and failed tasks.")
    criteria_coach.append(f"{len(criteria_coach) + 1}) Detail specifications (acceptance criteria, strategies/alternative to compare, use of LLM agents/tools/...) to successfully prompt a coder agent to generate code implementing the task while minimizing distance to goal. Organize the requirements with clear indexing to a depth of 3.")
    criteria_coach.append(f"{len(criteria_coach) + 1}) You should propose the next best novel task to implement 'given' available learnt tasks performance and LLM knowledge, 'maximizing' document generation quality, length and format, and speed to produce it.")
    criteria_coach.append(f"{len(criteria_coach) + 1}) You should propose a plan to achieve this task by breaking it down as a tree-structure. The plan tree should be of depth 3.")

    coach_criterias_topcurrent = "\n".join(criteria_coach)
    coach_criterias_newhybrid_v1 = """1) Task should be designed with a hybrid approach in mind, leveraging both LLM strengths and algorithmic processing.
2) Consider the following limitations of LLMs and how to address them:
    a) Domain adaptation and specialization
    b) Decomposition and multi-step reasoning
    c) Safety, data sensitivity, and hallucinations
    d) Explainability and transparency
    e) Long-form question answering and long-term dependencies
    f) Multimodal search and reasoning
    g) Temporal reasoning
    h) Bias reduction and fairness
    i) Scalability and efficiency
    j) Alignment to human intentions and values
    k) Question context improvement
3) Reason in 4 steps to find out the best task to minimize distance to goal, considering both LLM capabilities and algorithmic processing.
4) Task should be written in the form of "[operation] on [subject] utilizing [resources] following [guidelines]"
5) Detail specifications to successfully prompt a coder agent to generate code implementing the task while minimizing distance to goal. Include:
    5.1) Acceptance criteria
    5.2) Strategies/alternatives to compare
    5.3) Use of LLM agents/tools
    5.4) Algorithmic components to complement LLM capabilities
    5.5) Data preprocessing and postprocessing steps
    Organize the requirements with clear indexing to a depth of 4.
6) Tasks provided should be generic, not specific to given examples.
7) Propose the next best novel task to implement 'given' available learnt tasks performance and LLM knowledge, 'maximizing' document generation quality, length, format, and speed to produce it. Consider how to balance LLM calls and algorithmic processing for optimal results.
8) Propose a plan to achieve this task by breaking it down as a tree-structure. The plan tree should be of depth 3, clearly indicating which components are best handled by LLM calls and which by algorithmic processing.
9) After the task description and plan, provide a simple unit test of the function:
    a) Write a one-liner Python call to the main function for each "Document to be tested", designed to maximize the expected results for the "Document to be tested"
    b) Precede each Python test with the DocumentID value to indicate to which document the test code refers
    c) Call to the main function uses bot as the first required parameter
    d) No more than one test for each document; the total number of function calls in this test list should be equal to the number of "Document to be tested"
10) For each component of the task, clearly specify whether it should be handled by an LLM call or algorithmic processing, and provide a brief justification for the choice.
11) Include error handling and fallback mechanisms in the task design to ensure system reliability, especially when switching between LLM calls and algorithmic processing.
12) Consider scalability and efficiency in the task design, optimizing the balance between LLM calls and algorithmic components.
13) Incorporate mechanisms for bias detection and mitigation in the task design, leveraging both LLM capabilities and algorithmic checks.
14) Include steps for ensuring explainability and transparency of the results, combining LLM-generated explanations with algorithmic tracking of reasoning steps.
15) Address potential issues of data sensitivity and privacy by clearly separating tasks that can be handled by LLMs from those that require secure algorithmic processing.
"""
    coach_criterias_newhybrid_v2 = """1. **Task Complexity**:
    - The task must be **sufficiently simple** to be implemented using a combination of Python functions and LLM calls based on available resources and learnt tasks.
    - **Leverage LLM** for natural language understanding, generation, and abstract reasoning where complex decomposition is required.
    - **Leverage Code** for structured operations, data retrieval, calculations, and repetitive processing.
2. **Task Reasoning in Four Steps**:
    - Use the following four-step reasoning to define the task:
        1. **Decompose** the high-level research paper goal into manageable sub-tasks.
        2. **Identify** which sub-tasks are best handled by the LLM and which require algorithmic processing.
        3. **Sequence** the sub-tasks in a logical flow to ensure minimal distance to the final goal.
        4. **Evaluate** whether the task complexity aligns with existing learnt tasks and can be converted into Python code efficiently.
3. **Task Format**:
    - Write tasks in the format:
        **"[operation] on [subject] utilizing [resources] following [guidelines]"**
    - Example: "Summarize recent advancements on [target subject] utilizing [research databases] following [selected formatting guidelines]".
4. **Leveraging Hybrid System**:
    - Python code for the task will mix traditional algorithms and LLM calls:
        - **LLM for natural language understanding, summarization, or multi-step reasoning**.
        - **Algorithms for data handling, validation, and managing long-form content**.
    - The Python code should reflect appropriate usage of LLM agents or tools only when human-like language processing or abstract reasoning is required. Use traditional code for tasks like managing long-form dependencies, precise calculations, or bias reduction.
5. **Task Specifications**:
    - **5.1** **Acceptance Criteria**: Clear success criteria for when the task is considered valid (e.g., the task results in a synthesized document improving its content quality by a certain percentage).
    - **5.2** **Strategies and Alternatives**: Specify alternative approaches if the primary strategy fails or if LLM output quality is insufficient (e.g., fallback to a simpler decomposition using traditional algorithms).
    - **5.3** **LLM Integration**: Define when LLMs should be used, especially for tasks involving text interpretation, generation, or summarization.
    - **5.4** **Performance Optimization**: Ensure that LLM calls are optimized for cost and efficiency, using traditional code to handle repetitive or structured tasks.
    - **5.5** **Error Handling and Explainability**: Include ways to handle LLM hallucinations or provide transparent explanations of complex reasoning processes.
6. **Task Generalization**:
    - Ensure the task is not tied to specific documents or examples but remains general to apply to any future document object input.
7. **Task Impact**:
    - The task must maximize document generation quality, length, format consistency, and speed of production. The balance between document depth and processing time should be optimized.
8. **Task Plan and Tree Structure**:
    - Break the task down into a **plan tree** of depth 3, which organizes the task as:
        - **Step 1**: High-level objective (e.g., "Summarize recent developments on topic X").
        - **Step 2**: Sub-task identification (e.g., "Identify key papers using LLM, retrieve metadata using algorithms").
        - **Step 3**: Further decomposition and parallelizable steps (e.g., "Apply formatting, compare document consistency using algorithms")."""

    coach_criterias_dict = {
        'coach_criterias_topcurrent':coach_criterias_topcurrent,
        'coach_criterias_newhybrid_v1':coach_criterias_newhybrid_v1,
        'coach_criterias_newhybrid_v2':coach_criterias_newhybrid_v2}

    coach_criterias = coach_criterias_dict[trial.suggest_categorical("coach_criterias", ['coach_criterias_topcurrent', 'coach_criterias_newhybrid_v1', 'coach_criterias_newhybrid_v2'])]

    criteria_user_message = trial.suggest_categorical("criteria_user_message",
                                                      ["None", "Learnt", "Failed", "Env", "All", "LearntEnv", "FailedEnv"])

    #coach_format_output_type = "Markdown"

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
task_function_name(bot, arguments with values describing document #125dc4bc-54e0-4336-82bc-417e40ec9b8f for the given task...)
# document #2fa754cb-2e90-3376-3b2c-142f29c9ebf8 usage test:
task_function_name(bot, arguments with values describing document #2fa754cb-2e90-3376-3b2c-142f29c9ebf8 for the given task...)
```"""

    prompt_coach = f"""
    CONTEXT:
    {coach_agent_role}.

    TASK DESCRIPTION:
    {coach_task_description}.

    USER INPUT & STATE OF THE ENVIRONMENT:
    {coach_user_input}.

    CRITERIA:
    {coach_criterias}

    RESPONSE FORMAT:
    {coach_format_output}
    """
    # Write the prompt in the file readed after by the coach
    with open("./prompts/IR_CPS_TechSynthesis/identify_best_task.txt", "w") as f:
        f.write(prompt_coach)

    # Define parameters for Coder

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

    # del criteria_coder[instruction_to_remove - 1]
    # coder_text = "\n".join(criteria_coder)

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
    with open("Optuna_results.txt", "a") as f:
        f.write(f"Trial: {trial.number}\nPrompt Coach Chosen: \n{prompt_coach}")
        # f.write(f"Trial: {trial.number}\nPrompt Coach Chosen: \n{prompt_coach}\nPrompt Coder Chosen : \n{prompt_coder}\nPrompt Critic Chosen : \n{prompt_critic}\nModel Chosen: {modelVariation}\n")

    # Run the learning loop
    perf = learn.run_4agents_learning_loop(default_llm_key="default_llm",
                                           premium_llm_key="premium_llm",
                                           llmORchains_list=llmORchains_list,
                                           test_environments=envs,
                                           manual_validation_to_capitalize=False,
                                           problem_prompts_subdir='IR_CPS_TechSynthesis',
                                           max_coding_attempts=2,
                                           include_code=False,
                                           selected_successful_functions=[],
                                           selected_failed_functions=[],
                                           agtask_premium_llm_by_default=True,
                                           max_execution_time=900,
                                           agtask_skip_rounds=0,
                                           agcoding_skip_rounds=0,
                                           agvalidation_skip_rounds=0,
                                           agcapitalize_skip_rounds=0,
                                           model_choice={"coach": "premium_llm", "coder": "premium_llm",
                                                         "critic": "default_llm", "capitalizer": "default_llm"},
                                           optuna_opti="Coach",
                                           criteria=criteria_user_message)

    with open("Optuna_results.txt", "a") as f:
        f.write(f"Performance: {perf}\n\n")

    return perf


if __name__ == "__main__":
    # Initialize the default and premium LLMs
    # default_llm = ChatOpenAI(model_name="gpt-3.5-turbo-1106") # gpt-4-1106-preview gpt-3.5-turbo-1106 model_name=model_name, temperature=temperature, request_timeout=request_timout
    # default_llm = create_Nmajority_chain(num_models=3)
    # premium_llm = ChatOpenAI(model_name="gpt-4o") # gpt-4-1106-preview gpt-3.5-turbo-1106 model_name=model_name, temperature=temperature, request_timeout=request_timout
    llmORchains_list = {
        "default_llm": ChatOpenAI(model_name=MODELS_CONFIG_LIST["basic_gpt"]),
        "premium_llm": ChatOpenAI(model_name=MODELS_CONFIG_LIST["smart_gpt"]),
        # "3_majority_chain": learn.create_Nmajority_chain(map_model_name=MODELS_CONFIG_LIST["gpt-3.5"], reduce_model_name=MODELS_CONFIG_LIST["gpt-3.5"] , num_models=3),
        # "10_majority_chain": learn.create_Nmajority_chain(map_model_name=MODELS_CONFIG_LIST["gpt-3.5"], reduce_model_name=MODELS_CONFIG_LIST["gpt-3.5"], num_models=10)
    }

    # Set the documents to test/validate as a list of environments
    documents = [{'id': "cf0d353c-b43b-4a79-88f9-42c2c84cf75e",
                  'title': "Complex QA and language models hybrid architectures, Survey",
                  'context': "This paper reviews the state-of-the-art of language models architectures and strategies for 'complex' question-answering (QA, CQA, CPS) with a focus on hybridization. Large Language Models (LLM) are good at leveraging public data on standard problems but once you want to tackle more specific complex questions or problems (e.g. How does the concept of personal freedom vary between different cultures ? What is the best mix of power generation methods to reduce climate change ?) you may need specific architecture, knowledge, skills, methods, sensitive data protection, explainability, human approval and versatile feedback... Recent projects like ChatGPT and GALACTICA have allowed non-specialists to grasp the great potential as well as the equally strong limitations of LLM in complex QA. In this paper, we start by reviewing required skills and evaluation techniques. We integrate findings from the robust community edited research papers BIG, BLOOM and HELM which open source, benchmark and analyze limits and challenges of LLM in terms of tasks complexity and strict evaluation on accuracy (e.g. fairness, robustness, toxicity, ...) as a baseline. We discuss some challenges associated with complex QA, including domain adaptation, decomposition and efficient multi-step QA, long form and non-factoid QA, safety and multi-sensitivity data protection, multimodal search, hallucinations, explainability and truthfulness, temporal reasoning. We analyze current solutions and promising research trends, using elements such as: hybrid LLM architectural patterns, training and prompting strategies, active human reinforcement learning supervised with AI, neuro-symbolic and structured knowledge grounding, program synthesis, iterated decomposition and others.",
                  'target_file_path': "env/IR_CPS_TechSynthesis/document_embedding_analysis/output/arxiv/Complex QA and language models hybrid architectures Survey.json"},
                 {'id': "42252c6c-12f3-4edf-9045-8acd69bc3356",
                  'title': "Macroeconomic Effects of Inflation Targeting A Survey of the Empirical  Literature",
                  'context': "This paper surveys the empirical literature of inflation targeting. The main findings from our review are the following: there is robust empirical evidence that larger and more developed countries are more likely to adopt the IT regime; the introduction of this regime is conditional on previous disinflation, greater exchange rate flexibility, central bank independence, and higher level of financial development; the empirical evidence has failed to provide convincing evidence that IT itself may serve as an effective tool for stabilizing inflation expectations and for reducing inflation persistence; the empirical research focused on advanced economies has failed to provide convincing evidence on the beneficial effects of IT on inflation performance, while there is some evidence that the gains from the IT regime may have been more prevalent in the emerging market economies; there is not convincing evidence that IT is associated with either higher output growth or lower output variability; the empirical research suggests that IT may have differential effects on exchange-rate volatility in advanced economies versus EMEs; although the empirical evidence on the impact of IT on fiscal policy is quite limited, it supports the idea that IT indeed improves fiscal discipline; the empirical support to the proposition that IT is associated with lower disinflation costs seems to be rather weak. Therefore, the accumulated empirical literature implies that IT does not produce superior macroeconomic benefits in comparison with the alternative monetary strategies or, at most, they are quite modest.",
                  'target_file_path': "env/IR_CPS_TechSynthesis/document_embedding_analysis/output/arxiv/Macroeconomic Effects of Inflation Targeting A Survey of the Empirical  Literature.json"}]
    envs = []
    for doc in documents:
        env = learn.EnvironmentManager(env_type="techsynthesis", title=doc['title'], context=doc['context'],
                                       target_file_path=doc['target_file_path'], id=doc['id']).get_environment()
        envs.append(env)

    with open("Optuna_results.txt", "w") as f:
        f.write("")
    # Wait for 10s
    time.sleep(10)
    # get current folder
    current_folder = os.getcwd()

    sqlite_file = os.path.join(current_folder, "optuna.db")

    # Create a study and optimize the objective function
    study = opt.create_study(direction="maximize", storage=f"sqlite:///{sqlite_file}")
    study.optimize(objective, n_trials=200)