import sys

import optuna
import os

def analysis(name_exp : str):
    # Define the path to the SQLite database
    current_folder = os.getcwd()

    # File to save results
    output_file = os.path.join(current_folder, f"{name_exp}.txt")

    sqlite_file = os.path.join(current_folder, f"{name_exp}.db")

    # Function to write results to the file
    def write_to_file(text):
        with open(output_file, "a") as f:
            f.write(text + "\n")

    # Get all studies in the database
    study_summaries = optuna.study.get_all_study_summaries(storage=f"sqlite:///{sqlite_file}")

    if len(study_summaries) == 0:
        write_to_file("No studies found in the database.")
    else:
        # Sort studies by their `datetime_start` and load the most recent one
        most_recent_study = max(study_summaries, key=lambda s: s.datetime_start)
        study_name = most_recent_study.study_name

        write_to_file(f"Loading the most recent study: {study_name}")

        # Load the most recent study
        study = optuna.load_study(study_name=study_name, storage=f"sqlite:///{sqlite_file}")

        # Check if the study has trials
        if len(study.trials) > 0:
            try:
                # Retrieve the best trial
                best_trial = study.best_trial
                write_to_file("\nBest Trial:")
                write_to_file(f"Value: {best_trial.value}")
                write_to_file("Parameters: ")
                for key, value in best_trial.params.items():
                    write_to_file(f"    {key}: {value}")
            except ValueError as e:
                write_to_file(f"Could not retrieve the best trial: {e}")
        else:
            write_to_file("No trials found in the study.")

        # Get all trials as a DataFrame if there are any trials
        if len(study.trials) > 0:
            df = study.trials_dataframe()

            # Write an overview of the DataFrame
            write_to_file("Trials DataFrame Overview:")
            write_to_file(df.head().to_string())

            # Continue with further analysis (importance, plots, etc.)
        else:
            write_to_file("No trials available to create a DataFrame.")
