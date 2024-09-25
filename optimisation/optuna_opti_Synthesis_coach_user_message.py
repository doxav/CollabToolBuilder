from optimisation.optuna_main import launch_run, definition_few_shots, global_main


def objective(trial, timestamp_exp: str):
    # Definition of the coach's prompt
    coach_agent_role = "You are an AI coach"
    # trial.suggest_categorical("role_priming", ["In the context of a system generating efficient state-of-the-art research survey papers on a given subject, you are a researcher expert advising on the next best task to develop. This system employs a hybrid approach, leveraging both LLM capabilities and traditional algorithmic processing.",
                                                                  # "You are an AI coach"])
    # few_shots = definition_few_shots(trial)
    few_shots = {
        "sources" : "exemples",
        "num" : 1,
        "sort_order" : "by_date_asc",
        "format" : "json",

    }
    # coach_user_input_failed_tasks = trial.suggest_categorical("coach_user_input_failed_tasks", [True, False])
    coach_user_input = "I will provide you:\n- Learnt tasks available (with information gain between 0 and 1 on plan's titles, and contents): ...\n- Failed tasks to learn that are too hard to code: ...\n- Current status of examples of technical synthesis the proposed next task will be tested on: ..."

    coach_task_description = "You define best next task to generate state-of-the-art research survey paper given a [Title] and an [Abstract]. Each task you propose will be prompted to a language model which will try to convert it into Python functions. If the code is successful and gains in technical synthesis above a pre-defined threshold, this learnt task is made available to the next learning iteration."
        #(trial.suggest_categorical("task_description", [
        #"You define best next task to generate state-of-the-art research survey paper given a [Title] and an [Abstract]. Each task you propose will be prompted to a language model which will try to convert it into Python functions. If the code is successful and gains in technical synthesis above a pre-defined threshold, this learnt task is made available to the next learning iteration.",
        #"Define the next best task to generate a state-of-the-art research survey paper given a document object (containing target title and abstract/context in its properties). Each task you propose will be converted into Python functions, potentially including LLM calls when appropriate. If the implementation is successful and gains in technical synthesis above a pre-defined threshold, this learnt task is made available to the next learning iteration."]))

    user_message_parameters = definition_few_shots(trial, True)

    criteria_coach = []
    criteria_coach = [
        f"{len(criteria_coach) + 1}) Task should challenge the LLM while remaining solvable with available resources.",
        f"{len(criteria_coach) + 1}) Reason in 4 steps to find out the best task to minimize distance to goal.",
        f"{len(criteria_coach) + 1}) Task should be written in the form of \"Task should be written in the form of [verb] [quantity if applicable] [object] [tools] [detailed instructions and parameters]\"",
        f"{len(criteria_coach) + 1}) Task will be converted into Python code given available commands, learnt tasks. The Python code could call LLM when required.",
        f"{len(criteria_coach) + 1}) Task should be novel compared to learnt and failed tasks.",
        f"{len(criteria_coach) + 1}) Detail specifications (acceptance criteria, strategies/alternative to compare, use of LLM agents/tools/...) to successfully prompt a coder agent to generate code implementing the task while minimizing distance to goal. Organize the requirements with clear indexing to a depth of 3.",
        f"{len(criteria_coach) + 1}) You should propose the next best novel task to implement 'given' available learnt tasks performance and LLM knowledge, 'maximizing' document generation quality, length and format, and speed to produce it.",
        f"{len(criteria_coach) + 1}) You should propose a plan to achieve this task by breaking it down as a tree-structure. The plan tree should be of depth 3."]

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

    # coach_criterias_dict = {
    #     'coach_criterias_topcurrent':coach_criterias_topcurrent,
    #     'coach_criterias_newhybrid_v1':coach_criterias_newhybrid_v1,
    #     'coach_criterias_newhybrid_v2':coach_criterias_newhybrid_v2}

    coach_criterias = coach_criterias_topcurrent # coach_criterias_dict[trial.suggest_categorical("coach_criterias", ['coach_criterias_topcurrent', 'coach_criterias_newhybrid_v1', 'coach_criterias_newhybrid_v2'])]

    #coach_format_output_type = "Markdown"

    coach_format_output = """You should only respond in the Markdown format as described below:
        1. Reasoning: Analysis in 4 steps of the provided information to determine the next best task to develop, minimizing the distance to the goal
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
    
    {few_shots}
    """
    # Write the prompt in the file readed after by the coach
    with open("./prompts/IR_CPS_TechSynthesis/identify_best_task.txt", "w") as f:
        f.write(prompt_coach)

    # Save the parameters chosen by the trial
    with open(f"Optuna_results/{timestamp_exp}.txt", "a") as f:
        f.write(f"Trial: {trial.number}\nPrompt Coach Chosen: \n{prompt_coach}")

    perf = launch_run(
        default_llm_key="default_llm",
        premium_llm_key="premium_llm",
        problem_prompts_subdir="IR_CPS_TechSynthesis",
        max_coding_attempts=2,
        max_execution_time=900,
        model_choice={"coach": "premium_llm", "coder": "premium_llm", "critic": "default_llm", "capitalizer": "default_llm"},
        optuna_opti="coach",
        params_user_message=user_message_parameters,
    )
    with open(f"Optuna_results/{timestamp_exp}.txt", "a") as f:
        f.write(f"Performance: {perf}\n\n")

    return perf

if __name__ == "__main__":
    global_main(objective, "coach")
