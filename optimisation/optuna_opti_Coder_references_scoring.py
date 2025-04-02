from optimisation.optuna_main import launch_run, definition_few_shots, global_main, init_prompts_directory
from optimisation.optuna_main import fixed_coach_prompt, documentation
from datetime import datetime

Coach1 = """Implement generate_toc_with_bib_resources such that it creates a coherent TOC from the research paper’s title and abstract. The TOC must include multiple sections (minimum 3 sections and 6 subsections) and must update the bibliography by invoking add_or_update_result_in_resources. Focus on clear extraction of topics, robust error handling, and a fallback mechanism for cases with poor topic extraction. Don't generate the code to test generate_toc_with_bib_resources, generate only the function definition generate_toc_with_bib_resources(bot)"""
Coach2 = """Develop the function generate_toc_with_bib_resources. Use the provided title and abstract to extract key topics and generate a structured table of contents with at least 3 sections and 6 subsections. In addition, ensure that the function calls add_or_update_result_in_resources to update the bibliography with relevant resources. The solution must handle parameter validation and include a fallback default TOC in case topic extraction fails.. Don't generate the code to test generate_toc_with_bib_resources, generate only the function definition generate_toc_with_bib_resources(bot)"""
Coach3 = """Write generate_toc_with_bib_resources to produce a structured table of contents from a paper’s title and abstract. Ensure that the output contains several sections (at least 3 main sections with 6 subsections total) and that the function populates the resources bibliography by calling add_or_update_result_in_resources. Prioritize error handling, clear parameter management, and a reliable default structure if topic extraction is unsuccessful.. Don't generate the code to test generate_toc_with_bib_resources, generate only the function definition generate_toc_with_bib_resources(bot)"""


def objective(trial, name_xp : str):
    # Define parameters we want to tests

    max_coding_attempts = 4
    primitives_selection = "primitives/generate_primitives"

    # Define fixed parameters for Coder
    max_autofix = 3

    fixed_coach = trial.suggest_categorical("coach_prompt", [Coach1, Coach2, Coach3])

    # Few shots parameters
    #user_message_params = definition_few_shots(trial, True, no_params_search=True)

    # Reasoning and Task instructions based on file content

    # set coder_task_description from the file  prompts/IR_CPS_TechSynthesis/code_task.txt
    with open(f"prompts/IR_CPS_TechSynthesis/code_task.txt", "r") as f:
        coder_task_description = f.read()
        # get ANSI CODE BLUE, get ANSI CODE RESET
        blue, reset = '\033[94m', '\033[0m'
        print(f"CODER TASK DESCRIPTION: {blue} {coder_task_description} {reset}")

    # Log the prompt and parameters for this trial
    #with open(f"Optuna_results/{name_xp}.txt", "a") as f:
    #    f.write(f"Trial: {trial.number}\nGenerated Coder Prompt: \n{full_prompt}\n")

    performance = launch_run(
        default_llm_key="default_llm",
        premium_llm_key="premium_llm",
        problem_prompts_subdir="IR_CPS_TechSynthesis",
        max_execution_time=900,
        model_choice={"coach": "default_llm","coder": "premium_llm","critic": "default_llm","capitalizer": "default_llm"},
        automation="coach",
        special_criteria={
            "CodingAgent#max_autofix": max_autofix,
            "CodingAgent#num_parallel_inferences": 2,
            "max_coding_attempts": max_coding_attempts,
            "primitives_selection": primitives_selection
        },
        name_exp=name_xp,
        #params_user_message=user_message_params,
        fixed_coach=fixed_coach,
        unique_id="OptunaReferencesScoring" + datetime.now().strftime("%Y%m%d-%Hh%M"),
        run_graph=True
    )

    # Log performance for analysis
    with open(f"Optuna_results/{name_xp}.txt", "a") as f:
        f.write(f"Performance: {performance}\n\n")

    return performance


if __name__ == "__main__":
    global_main(objective, "coder", "fixed_coach")
