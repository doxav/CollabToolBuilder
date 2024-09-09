import os
import sys
import time
from optimisation.optuna_main import launch_study, launch_run, definition_few_shots


def objective(trial, timestamp_exp : int):
    # Define parameters for Coder
    libraries_restriction = trial.suggest_categorical("libraries_restriction", [
        "Langchain, BeautifulSoap, RegEx, Sklearn",
        "Numpy, Pandas, Matplotlib",
        "TensorFlow, PyTorch"
    ])
    max_autofix = trial.suggest_int('max_autofix', 0, 5)
    temperature = trial.suggest_float('temperature', 0.0, 1.0)
    presence_penalty = trial.suggest_float('presence_penalty', -2.0, 2.0)
    reasoning_depth = trial.suggest_int('reasoning_depth', 1, 4)

    # Few shots parameters
    few_shots = definition_few_shots(trial)

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
    {few_shots}
    """

    # Combine everything to generate the final prompt for the Coder agent
    full_prompt = coder_task_description + coder_test_instructions

    # Write the generated prompt to a file that will be used by the Coder agent
    with open("./prompts/IR_CPS_TechSynthesis/code_task.txt", "w") as f:
        f.write(full_prompt)

    # Log the prompt and parameters for this trial
    with open(f"Optuna_results/xp_coder{timestamp_exp}.txt", "a") as f:
        f.write(f"Trial: {trial.number}\nGenerated Coder Prompt: \n{full_prompt}\n")

    performance = launch_run(
        default_llm_key="default_llm",
        premium_llm_key="premium_llm",
        problem_prompts_subdir="IR_CPS_TechSynthesis",
        max_coding_attempts=2,
        max_execution_time=900,
        model_choice={"coach": "default_llm","coder": "premium_llm","critic": "default_llm","capitalizer": "default_llm"},
        optuna_opti="coder",
        special_criteria={"max_autofix": max_autofix, "temperature": temperature, "presence_penalty": presence_penalty},
    )

    # Log performance for analysis
    with open(f"Optuna_results/xp_coder{timestamp_exp}.txt", "a") as f:
        f.write(f"Performance: {performance}\n\n")

    return performance


if __name__ == "__main__":
    os.chdir("../")
    # Recuperate name_exp from terminal argument:
    if len(sys.argv) > 1:
        name_exp = sys.argv[1]
    else :
        timestamp_xp = int(time.time())
        name_exp = f"xp_coder{timestamp_xp}"
    launch_study(lambda trial: objective(trial, timestamp_xp), name_exp)
