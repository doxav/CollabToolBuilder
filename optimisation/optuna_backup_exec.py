from optimisation.optuna_main import launch_run, global_main


def objective(trial, name_xp : str):
    # Log the prompt and parameters for this trial
    with open(f"Optuna_results/{name_xp}.txt", "a") as f:
        f.write(f"Trial: {trial.number}\nDefault values => backup continuation test\n")

    performance = launch_run(
        default_llm_key="default_llm",
        premium_llm_key="premium_llm",
        problem_prompts_subdir="IR_CPS_TechSynthesis",
        max_coding_attempts=2,
        max_execution_time=1200,
        model_choice={"coach": "default_llm","coder": "premium_llm","critic": "default_llm","capitalizer": "default_llm"},
        optuna_opti="coder",
        name_exp=name_xp
    )

    # Log performance for analysis
    with open(f"Optuna_results/{name_xp}.txt", "a") as f:
        f.write(f"Performance: {performance}\n\n")

    return performance


if __name__ == "__main__":
    global_main(objective, "coder")
