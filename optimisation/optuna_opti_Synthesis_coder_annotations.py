from inference_functions import generate_annotations
from learn import CodingAgent
from optimisation.optuna_main import launch_run, definition_few_shots, global_main, init_prompts_directory
from optimisation.optuna_main import documentation
import os

def objective(trial, name_xp : str):
    # Define parameters we want to tests

    num_previous_attempts = 3
    parameters_previous_attempts = "code & task"
    primitives_selection = "primitives/generate_primitives"
    prev_failed_task = ("36dd8e77-829d-48c9-8f6c-3d7bc8dfa3aa", "c7159fce-e204-4909-b156-299d698bc456")
    prev_learnt_task = "set2"

    # Define fixed parameters for Coder
    libraries_restriction = "Numpy, Pandas, Huggingface, Sklearn" # Can be removed: Huggingface and Sklearn, to test if 3 imposed methods offer better performance than with these libraries
    max_autofix = 2
    temperature_max = 1.3
    number_inferences = 3
    presence_penalty = 0.7189030356596702
    reasoning_depth = 1

    # Few shots parameters
    user_message_params = definition_few_shots(trial, True, no_params_search=True)

    tries_annotations = trial.suggest_categorical("tries_annotations", ['TaskIdentificationAgent', 'CodingAgent'])
    
    #annotation_types_filter = trial.suggest_categorical("annotation_types_filter", ['FIX', 'DELETE', 'APPROVE', 'ALL'])
    
    annotations_critic_system_prompt = trial.suggest_categorical("annotations_system_prompt", [
        """Your Task:

You are assigned to process the ANNOTATED ANSWER for the TARGET TASK by performing the following:

Identify all feedback annotations marked with \APPROVE, \FIX, and \DELETE.
Extract detailed instructions and the corresponding content for each annotation.
Organize these details into a systematic list that outlines the required actions and the relevant content.
Reminder: Do not modify the ANNOTATED ANSWER in any way during this process.


### Annotation Tags Explanation:
\APPROVE: Indicates the content is satisfactory as is.
\FIX: Indicates the content requires changes or improvements.
\DELETE: Indicates the content should be removed entirely.

### Instructions:

Examine the ANNOTATED ANSWER carefully.
For each annotation:
Determine the action to be taken.
Note the exact content involved.
Arrange all findings into a clear, itemized list.
Provide only this list as your output, without any extra comments or modifications to the original content.

""",
        """Your Task:

You are to analyze the ANNOTATED ANSWER provided for the TARGET TASK. Your objectives are:

Identify all instances of feedback annotations marked by the tags \APPROVE, \FIX, and \DELETE.
Extract the specific instructions and the associated content for each annotation.
Organize this information into a structured list detailing what action is required (approve, fix, delete) and which content it pertains to.
Important: Do not modify or alter the content of the ANNOTATED ANSWER during this process.


### Annotation Tags Explanation:
\APPROVE: This content is great, must not be changed.
\FIX: The content requires improvement or correction.
\DELETE: The content should be removed.

### Instructions:
Carefully read the ANNOTATED ANSWER.
For each annotation tag found:
Note the type of action (approve, fix, or delete).
Record the specific content associated with the tag.
Compile this information into a clear and concise list.
Output only the organized list without any additional commentary or modification of the ANNOTATED ANSWER.
""",
        """
Your task is to extract and organize feedback tags from the **ANNOTATED ANSWER** provided for the given **TARGET TASK**. Identify the instructions given by the annotation tags (\APPROVE, \FIX, \DELETE) and structure them into a list. Do not modify the content of **ANNOTATED ANSWER** at this stage.


### Annotation Tags:
- **\APPROVE:** This content is great, must not be changed.
- **\FIX:** The content requires improvement or correction.
- **\DELETE:** Remove this content.

### Your Task:
1. **Identify** the feedback instructions based on annotation tags.
2. **Organize** the instructions into a list format with specific details on what needs to be done (fix, delete, etc.).
3. **Reply** with the improved answer without any additional introduction or comments.
"""
    ])

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

    TASK: You should respond with the best Python code to perform the task. The performance of the task will be evaluated based on sections created, their content, resources.

    INSTRUCTIONS:
    1) Reason in {reasoning_depth} steps to identify the optimal way to achieve the task.
    2) Write a function taking 'bot' as the first parameter (IMPORTANT: it should comply with the test provided with no extra arguments), which is an instance of the class SynthesisManager, containing all document resources.
    3) Your function should include appropriate modification to resources and sections to measure task success, it will be evaluated by score (higher scores mean better codes/progressions). Main functions are:
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

    os.environ['NAME_XP'] = name_xp

    # Combine everything to generate the final prompt for the Coder agent
    full_prompt = coder_task_description + coder_test_instructions + documentation

    init_prompts_directory(name_xp, "coder")

    with open(f"./prompts/IR_CPS_TechSynthesis/{name_xp}/apply_annotations.txt", "w") as f:
        f.write(annotations_critic_system_prompt)

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
            "CodingAgent#max_autofix": max_autofix,
            "CodingAgent#temperature_max": temperature_max,
            "CodingAgent#num_parallel_inferences": number_inferences,
            "all#problem_prompts_subdir": f"IR_CPS_TechSynthesis/{name_xp}/",
            f"{tries_annotations}#additional_check_list": {"Generate annotations": generate_annotations},
            "presence_penalty": presence_penalty,
            "num_previous_attempts": num_previous_attempts,
            "parameters_previous_attempts": parameters_previous_attempts,
            "primitives_selection": primitives_selection,
            "prev_failed_task": prev_failed_task,
            "prev_learnt_task": prev_learnt_task
        },
        name_exp=name_xp,
        params_user_message=user_message_params,
        number_inferences=number_inferences
    )

    # Log performance for analysis
    with open(f"Optuna_results/{name_xp}.txt", "a") as f:
        f.write(f"Performance: {performance}\n\n")

    return performance


if __name__ == "__main__":
    global_main(objective, "coder", "annotations")
