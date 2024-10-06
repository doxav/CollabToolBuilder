from optimisation.optuna_main_coder_fixed_parameters import launch_run, definition_few_shots, global_main


def objective(trial, name_xp : str):
    # Define parameters we want to tests

    num_previous_attempts = 4
    parameters_previous_attempts = "code & task"
    primitives_selection = "primitives/generate_primitives"
    prev_failed_task = ("36dd8e77-829d-48c9-8f6c-3d7bc8dfa3aa", "c7159fce-e204-4909-b156-299d698bc456")
    prev_learnt_task = "set2"

    # Define fixed parameters for Coder
    libraries_restriction = "Numpy, Pandas, Huggingface, Sklearn" # Can be removed: Huggingface and Sklearn, to test if 3 imposed methods offer better performance than with these libraries
    max_autofix = 3
    fixed_prompt = """
1. **Reasoning**:\n   - The current environment has experienced several failures in generating 
coherent tables of contents (TOCs) for research papers, indicating that previous attempts did not 
effectively leverage LLM capabilities for topic extraction and organization.\n   
- The presence of multiple failed tasks highlights the importance of developing a robust function 
that not only extracts topics but also structures them coherently into a TOC that meets quality 
standards, including a minimum of 3 sections and 6 subsections.\n   - Previous attempts have 
shown that runtime errors related to missing parameters have hindered the successful execution 
of functions. Therefore, the next task must ensure clear parameter handling and streamlined operations 
for topic extraction and TOC generation.\n   - Given the current state of the environment with 
empty TOCs and the need for improvement in the extraction and structuring processes, the next 
best task should concentrate on refining the TOC generation method to enhance output quality and 
coherence while ensuring successful execution.\n\n2. **Next Best Task**:\n   
- **Function Name**: `generate_toc_with_fallback`\n   - **Description**: Create a structured table 
of contents (TOC) for a research survey paper using the provided title and abstract, utilizing LLM capabilities 
for topic extraction, and implementing a reliable fallback mechanism for organizing content if extraction fails. 
\n\n3. **Performance Acceptance Criteria**:\n   - The generated TOC must include at least 3 relevant 
sections and 6 subsections derived from the document's context.\n   - If topic extraction fails, the function 
should provide a coherent default TOC structure that aligns with standard research paper outlines.\n   - The TOC 
should maintain a clear hierarchical structure, effectively delineating main sections and subsections.\n   - The 
function should execute within a timeframe of less than 3 seconds.\n   - The output must be coherent, relevant, 
and free from hallucinations or irrelevant content.\n\n4. **Development Plan**:\n   - **Plan Depth**: 3\n   
- **Steps**:\n     - **Step 1**: Use LLM to extract main topics and subtopics from the provided title and abstract.\n       
- **LLM Call**: Perform natural language processing to identify key themes and topics.\n       
- **Error Handling**: Implement a fallback mechanism to handle scenarios where no topics are extracted, 
providing a predefined structure instead.\n     - **Step 2**: Organize the identified topics into a coherent 
hierarchical structure.\n       - **Algorithmic Processing**: Create a structured outline consisting of main 
sections and subsections based on the identified topics or a default template if extraction fails.\n     
- **Step 3**: Format the generated TOC into a structured output suitable for inclusion in the document.\n       
- **Algorithmic Processing**: Ensure the output is formatted as a well-structured list or dictionary for seamless 
integration into the research paper.\n\n5. **Tests**:\n```python\n# 
Test for document #cf0d353c-b43b-4a79-88f9-42c2c84cf75e:\ngenerate_toc_with_fallback(bot, 
title=\"Innovations in Renewable Energy Technologies\", abstract=\"This paper examines the latest 
advancements in renewable energy, focusing on solar, wind, and bioenergy technologies and their potential 
impact on global energy markets.\")\n\n# Test for document #42252c6c-12f3-4edf-9045-8acd69bc3356:\ngenerate_toc_with_fallback(bot, 
title=\"Artificial Intelligence and Machine Learning in Medicine\", abstract=\"This document reviews the integration of 
AI and machine learning in medical diagnostics and treatment, discussing case studies and future trends.\")\n``` 
\n\n### Summary of Component Handling:\n- **LLM Call**: The extraction of keywords and themes (Step 1) will utilize 
LLM capabilities for effective natural language processing, enhancing the understanding of context and semantics.\n
- **Algorithmic Processing**: Steps 2 and 3 (organization and formatting) will be performed through structured algorithmic 
processing to ensure coherent output and effective integration into research papers.\n\n### Error Handling and Fallback 
Mechanisms:\n- Implement error handling to manage exceptions during LLM processing, providing a fallback to a default TOC 
structure based on common research themes if keyword extraction fails.\n- Include logging for performance monitoring to 
ensure reliability and accountability of the task.\n\n### Scalability and Efficiency:\n- The task design should allow 
flexibility in the number of sections generated based on the document's title and abstract, ensuring scalability across 
various document sizes while maintaining efficiency.\n\n### Bias Detection and Mitigation:\n- Implement algorithms to 
analyze the generated TOC for potential biases, ensuring a balanced representation of topics across diverse perspectives.
\n\n### Explainability and Transparency:\n- Provide clear documentation of the TOC generation process, ensuring transparency 
in how LLM outputs and algorithmic processing contribute to the final structured output.
"""
    fixed_coach = trial.suggest_categorical("fixed_coach", [fixed_prompt, False])
    presence_penalty = 0.7189030356596702
    reasoning_depth = 1

    # Few shots parameters
    user_message_params = definition_few_shots(trial, True)

    # Reasoning and Task instructions based on file content

    coder_task_description = f"""
    CONTEXT:
    You are a helpful assistant that writes Python code to be executed using a restricted list of packages ({libraries_restriction}) to complete the task specified by me.

    At each round of conversation, I will give you:
    - Reasoning: explanation of the task chosen...
    - Task: defined based on the current stage of progress
    - Plan: how to proceed and complete the current task
    - Tests: to validate that the task was correctly implemented and generates expected results

    CURRENT STATE OF THE ENVIRONMENT:
    Document #.... : 1.title: ...; 2. abstract: ...; 3. table of content; 4. resources; 5. section progress; 6. events counted

    TASK: You should respond with the best Python code to perform the task.

    INSTRUCTIONS:
    1) Reason in {reasoning_depth} steps to identify the optimal way to achieve the task.
    2) Write a function taking 'bot' as the first parameter, which is an instance of the class SynthesisManager, containing all document resources.
    3) Ensure the generated code adheres to reusability principles. The generated code should be modular and easy to maintain rather than specific to the task.
    4) Avoid hard-coding parameters. Pass necessary data as arguments to ensure reusability.
    5) The function should call existing helper functions as much as possible to focus on improving results, not redoing code.
    6) Ensure the code is executable with no placeholders and fully complete for immediate testing and deployment.
    7) Name your function meaningfully to reflect the task it is performing.

    You should then respond with:
    - Reasoning: how to best implement the task with maximum efficiency
    - Code: fully executable Python code adhering to the task constraints
    """
    coder_test_instructions = f"""
    RESPONSE FORMAT:
    Reasoning: Your detailed thought process and why the chosen solution is optimal
    Code: Python implementation of the solution
    ```python
    # Example Python code here
    def your_function(bot):
        # implementation here... 
    ```
    """

    # Combine everything to generate the final prompt for the Coder agent
    full_prompt = coder_task_description + coder_test_instructions

    # Write the generated prompt to a file that will be used by the Coder agent
    with open("./prompts/IR_CPS_TechSynthesis/code_task.txt", "w") as f:
        f.write(full_prompt)

    # Log the prompt and parameters for this trial
    with open(f"Optuna_results/{name_xp}.txt", "a") as f:
        f.write(f"Trial: {trial.number}\nGenerated Coder Prompt: \n{full_prompt}\n")

    performance = launch_run(
        default_llm_key="default_llm",
        premium_llm_key="premium_llm",
        problem_prompts_subdir="IR_CPS_TechSynthesis",
        max_coding_attempts=2,
        max_execution_time=900,
        model_choice={"coach": "default_llm","coder": "premium_llm","critic": "default_llm","capitalizer": "default_llm"},
        optuna_opti="coach",
        special_criteria={
            "max_autofix": max_autofix,
            "presence_penalty": presence_penalty,
            "num_previous_attempts": num_previous_attempts,
            "parameters_previous_attempts": parameters_previous_attempts,
            "primitives_selection": primitives_selection,
            "prev_failed_task": prev_failed_task,
            "prev_learnt_task": prev_learnt_task
        },
        name_exp=name_xp,
        params_user_message=user_message_params,
        fixed_coach=fixed_coach,
    )

    # Log performance for analysis
    with open(f"Optuna_results/{name_xp}.txt", "a") as f:
        f.write(f"Performance: {performance}\n\n")

    return performance


if __name__ == "__main__":
    global_main(objective, "coder", "fixed_coach")
