from optimisation.optuna_main import launch_run, definition_few_shots, global_main, init_prompts_directory
from optimisation.optuna_main import fixed_coach_prompt, documentation

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

    fixed_coach = trial.suggest_categorical("fixed_coach", [fixed_coach_prompt, False])
    presence_penalty = 0.7189030356596702
    reasoning_depth = 1

    # Few shots parameters
    user_message_params = definition_few_shots(trial, True, no_params_search=True)

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
    3) Your function should include appropriate modification to resources and sections to measure task success/performance. Main functions are:
        - class Section(section_id: int, title: str, content: str, parent_id: int)
        - Manipulate document sections: bot.create_and_add_section_then_return_id(title: str, content: str, section_id: int = None, parent_id: int = None) -> int, bot.get_all_sections() -> List[Section], bot.get_sections(ids: List[int]) -> List[Section], bot.edit_section(section_id: int, new_content: str = None, new_title: str = None, new_parent_id: int = None) -> bool, bot.remove_section(section_id: int) -> bool, bot.swap_sections(section_id_1: int, section_id_2: int) -> bool
        - Manipulate document resources: bot.add_or_update_results_in_resources(results, metadatas_to_add:dict=None, store_linked_document_content:bool=False), bot.add_or_update_result_in_resources(metadatas:dict, name:str=None, content:dict=None, link:str=None, store_linked_document_content:bool=False), bot.get_all_resources(self) -> List[Dict[str, Any]], bot.semantic_search_resources(query_texts, n_results=10), bot.add_or_update_results_in_resources(results, metadatas:dict=None, store_linked_document_content:bool=False), bot.get_and_store_link_content(link:str=None, parent_id=None, chaining:bool=True), bot.remove_resource(resource_id)
    4) Ensure the generated code adheres to reusability principles. The generated code should be modular and easy to maintain rather than specific to the task.
    5) Avoid hard-coding parameters. Pass necessary data as arguments to ensure reusability.
    6) The function should call existing helper functions as much as possible to focus on improving results, not redoing code.
    7) Ensure the code is executable with no placeholders and fully complete for immediate testing and deployment.
    8) Name your function meaningfully to reflect the task it is performing.

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
    full_prompt = coder_task_description + coder_test_instructions + documentation

    init_prompts_directory(name_xp, "coder")

    # Write the generated prompt to a file that will be used by the Coder agent
    with open(f"./prompts/IR_CPS_TechSynthesis/{name_xp}/code_task.txt", "w") as f:
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
