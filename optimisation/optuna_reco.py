from optimisation.optuna_main import launch_run, global_main, init_prompts_directory


def objective(trial, name_xp : str):
    recommendations_usage = trial.suggest_categorical("recommendations_usage", [True, False])
    init_prompts_directory(name_xp, "reco")

    performance = launch_run(
        default_llm_key="default_llm",
        premium_llm_key="premium_llm",
        problem_prompts_subdir="IR_CPS_TechSynthesis",
        max_coding_attempts=2,
        max_execution_time=900,
        model_choice={"coach": "default_llm","coder": "premium_llm","critic": "default_llm","capitalizer": "default_llm"},
        optuna_opti="coach",
        special_criteria={
            "recommendations_usage": recommendations_usage,
        },
        name_exp=name_xp,
        unique_id=name_xp)

    # Log performance for analysis
    with open(f"Optuna_results/{name_xp}.txt", "a") as f:
        f.write(f"Performance: {performance}\n\n")

    return performance


if __name__ == "__main__":
    global_main(objective, "all", "recommendations_usage")