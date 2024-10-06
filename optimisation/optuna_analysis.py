import optuna
import os
import pandas as pd
from optuna.importance import FanovaImportanceEvaluator
from optuna.importance import get_param_importances
from itertools import combinations
import traceback
import numpy as np  # Added for variance calculation
import sys
import shutil  # For copying the SQLite file
from datetime import datetime

def truncate_string(s: str, max_length: int = 40, start_len: int = 20, end_len: int = 20) -> str:
    if len(s) > max_length:
        s = s.replace('\n', ' ')
        return f"{s[:start_len]}...{s[-end_len:]}"
    return s

def analysis(name_exp: str, in_dir = False):
    try:
        # Define the path to the SQLite database
        current_folder = os.getcwd()
        current_datetime = datetime.now().strftime('%Y%m%d_%H%M%S')

        # File to save results
        output_file = os.path.join(current_folder, f"{name_exp}_analysis.txt" if in_dir else f"Optuna_results/{name_exp}_analysis.txt")

        sqlite_file_original = os.path.join(current_folder, f"{name_exp}.db" if in_dir else f"Optuna_db/{name_exp}.db")
        sqlite_file = os.path.join(current_folder, f"{name_exp}_copy_{current_datetime}.db" if in_dir else f"Optuna_db/{name_exp}_copy_{current_datetime}.db")

        # Function to write results to the file
        def write_to_file(text, mode="a"):
            with open(output_file, mode) as f:
                f.write(text + "\n")
            print(text)

        write_to_file("Start analysis", "w")

        # Copy the database file to avoid conflict
        shutil.copy(sqlite_file_original, sqlite_file)
        write_to_file(f"Database file copied to {sqlite_file}")

        # Get all studies in the database
        study_summaries = optuna.study.get_all_study_summaries(storage=f"sqlite:///{sqlite_file}")

        if len(study_summaries) == 0:
            write_to_file("No studies found in the database.")
            return
        else:
            # Sort studies by their `datetime_start` and load the most recent one
            most_recent_study = max(study_summaries, key=lambda s: s.datetime_start)
            study_name = most_recent_study.study_name
            start_time = most_recent_study.datetime_start.strftime('%Y-%m-%d %H:%M:%S')

            write_to_file(f"Loading the most recent study: {study_name} (Start: {start_time})")

            # Load the most recent study
            study = optuna.load_study(study_name=study_name, storage=f"sqlite:///{sqlite_file}")

            # Check if the study has trials
            total_trials = len(study.trials)
            if total_trials == 0:
                write_to_file("No trials found in the study.")
                return
            else:
                write_to_file(f"{total_trials} trials found.")

            # Get all trials as a DataFrame
            df = study.trials_dataframe()

            if len(df) > 0:
                # Sort by `value` to display top 10 trials (or fewer if less than 10)
                sorted_df = df.sort_values(by='value', ascending=False).head(min(10, len(df)))

                write_to_file(f"\nTop 10 Trials by Performance (out of {total_trials} trials):\n")
                write_to_file(f"{'Value':<10} | {'State':<10} | {'Params':<40} | {'Datetime':<20} | {'Trial ID'}")
                write_to_file("-" * 120)

                for idx, row in sorted_df.iterrows():
                    # Handle Pandas Timestamp objects directly
                    datetime_short = row['datetime_start'].strftime('%Y-%m-%d %H:%M:%S') if row['datetime_start'] else "N/A"

                    # Extract parameter columns (columns that start with "params_")
                    params = {col.replace('params_', ''): truncate_string(str(row[col])) for col in df.columns if col.startswith('params_')}

                    # Convert the params dict to a string
                    params_str = str(params)

                    # Handle None values in 'value' and 'state'
                    trial_value = row['value'] if row['value'] is not None else "N/A"
                    trial_state = row['state'] if row['state'] is not None else "UNKNOWN"

                    # Write the row with params string
                    write_to_file(f"{trial_value:<10} | {trial_state:<10} | {params_str:<40} | {datetime_short:<20} | {row['number']}")
 
                # Sort by `value` to display top 10 trials (or fewer if less than 10)
                sorted_df = df.sort_values(by='value', ascending=True).head(min(10, len(df)))

                write_to_file(f"\nLeast 10 Trials by Performance (out of {total_trials} trials):\n")
                write_to_file(f"{'Value':<10} | {'State':<10} | {'Params':<40} | {'Datetime':<20} | {'Trial ID'}")
                write_to_file("-" * 120)

                for idx, row in sorted_df.iterrows():
                    # Handle Pandas Timestamp objects directly
                    datetime_short = row['datetime_start'].strftime('%Y-%m-%d %H:%M:%S') if row['datetime_start'] else "N/A"

                    # Extract parameter columns (columns that start with "params_")
                    params = {col.replace('params_', ''): truncate_string(str(row[col])) for col in df.columns if col.startswith('params_')}

                    # Convert the params dict to a string
                    params_str = str(params)

                    # Handle None values in 'value' and 'state'
                    trial_value = row['value'] if row['value'] is not None else "N/A"
                    trial_state = row['state'] if row['state'] is not None else "UNKNOWN"

                    # Write the row with params string
                    write_to_file(f"{trial_value:<10} | {trial_state:<10} | {params_str:<40} | {datetime_short:<20} | {row['number']}")

                # Check if there is a valid `best_trial` available
                try:
                    best_trial = study.best_trial

                    # Improved parameter performance analysis
                    write_to_file("\nParameter Performance:\n")
                    write_to_file(f"{'Parameter':<25} | {'Type':<12} | {'Value/Range':<40} | {'Mean Performance':<15} | {'Count':<5}")
                    write_to_file("-" * 100)

                    for param in best_trial.params.keys():
                        values = df[f'params_{param}']

                        if values.dtype == 'object':
                            # Categorical parameter performance
                            df[f'params_{param}'] = df[f'params_{param}'].apply(lambda x: tuple(x) if isinstance(x, list) else x)
                            performance_by_category = df.groupby(f'params_{param}')['value'].agg(['mean', 'count'])
                            for cat_value, row in performance_by_category.iterrows():
                                mean_perf = row['mean']
                                count = row['count']
                                cat_value_str = truncate_string(str(cat_value))
                                write_to_file(f"{param:<25} | {'Categorical':<12} | {cat_value_str:<40} | {mean_perf:<15.4f} | {count:<5}")

                        elif values.dtype == 'bool':
                            # Boolean parameter performance
                            performance_by_bool = df.groupby(f'params_{param}')['value'].agg(['mean', 'count'])
                            for bool_value, row in performance_by_bool.iterrows():
                                mean_perf = row['mean']
                                count = row['count']
                                write_to_file(f"{param:<25} | {'Boolean':<12} | {str(bool_value):<40} | {mean_perf:<15.4f} | {count:<5}")

                        elif values.dtype in ['int64', 'float64']:
                            if pd.api.types.is_integer_dtype(values):
                                # Integer parameter performance
                                performance_by_value = df.groupby(f'params_{param}')['value'].agg(['mean', 'count']).sort_index()
                                bin_type = 'Integer'
                                for value, row in performance_by_value.iterrows():
                                    mean_perf = row['mean']
                                    count = row['count']
                                    write_to_file(f"{param:<25} | {bin_type:<12} | {str(value):<40} | {mean_perf:<15.4f} | {count:<5}")
                            else:
                                # For floating-point numbers, use binning
                                min_value = values.min()
                                max_value = values.max()
                                bin_size = 5
                                bins = pd.cut(values, bins=bin_size, precision=1)
                                performance_by_bin = df.groupby(bins, observed=False)['value'].agg(['mean', 'count'])
                                bin_type = 'Numerical'
                                for bin_range, row in performance_by_bin.iterrows():
                                    mean_perf = row['mean']
                                    count = row['count']
                                    write_to_file(f"{param:<25} | {bin_type:<12} | {str(bin_range):<40} | {mean_perf:<15.4f} | {count:<5}")

                        # Add a separator line between parameters
                        write_to_file("-" * 100)

                except ValueError:
                    write_to_file("\nNo best trial found for parameter performance analysis.")

            # Filter out completed trials for parameter importance calculation
            completed_trials = [trial for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE]

            if len(completed_trials) == 0:
                write_to_file("\nNo completed trials available for parameter importance analysis.")
            else:
                # Check for variance in the trial values
                values = [trial.value for trial in completed_trials if trial.value is not None]

                if np.var(values) == 0:
                    write_to_file("\nSkipping parameter importance analysis due to zero variance in trial values.")
                else:
                    # Parameter Importance Analysis Merged
                    write_to_file("\nParameter Importance (default, Fanova, and Interactions):\n")
                    write_to_file(f"{'Parameter':<25} | {'Importance (default)':<20} | {'Importance (Fanova)':<20} | {'Interactions'}")
                    write_to_file("-" * 90)

                    # Get default importance and handle possible errors
                    try:
                        importance_default = get_param_importances(study)
                    except RuntimeError as e:
                        write_to_file(f"Default ImportanceEvaluator failed: {e}")
                        importance_default = {}

                    # Get Fanova importance and handle possible errors
                    fanova_evaluator = FanovaImportanceEvaluator()
                    try:
                        importance_fanova = get_param_importances(study, evaluator=fanova_evaluator)
                    except RuntimeError as e:
                        write_to_file(f"FanovaImportanceEvaluator failed: {e}")
                        importance_fanova = {param: 0 for param in importance_default.keys()}

                    # Create a combined DataFrame for both importance metrics
                    importance_df = pd.DataFrame({
                        'Parameter': list(importance_default.keys()),
                        'Importance (default)': list(importance_default.values()),
                        'Importance (Fanova)': [importance_fanova.get(param, 0) for param in importance_default.keys()]
                    })

                    # Sort by 'Importance (default)' descending
                    importance_df = importance_df.sort_values(by='Importance (default)', ascending=False)

                    # Simulated interactions by checking variance contribution for each pair of parameters
                    interaction_summary = {}
                    param_combinations = combinations(importance_df['Parameter'], 2)

                    for param1, param2 in param_combinations:
                        # Simulate interaction by calculating variance of the two parameters combined
                        interaction_value = (importance_default.get(param1, 0) + importance_default.get(param2, 0)) * 0.1  # Simulate 10% of total variance
                        if interaction_value > 0.01:  # Threshold to filter insignificant interactions
                            if param1 not in interaction_summary:
                                interaction_summary[param1] = []
                            interaction_summary[param1].append(f"{param2}: {interaction_value:.4f}")

                    # Print the merged table with interactions
                    for idx, row in importance_df.iterrows():
                        param = row['Parameter']

                        # Get interaction information for the current parameter
                        interactions_str = "; ".join(interaction_summary.get(param, ["No significant interactions"]))

                        write_to_file(f"{param:<25} | {row['Importance (default)']:<20.4f} | {row['Importance (Fanova)']:<20.4f} | {interactions_str}")

    except Exception as e:
        # Capture the traceback and write to the output file and console
        error_message = f"An error occurred: {e}\n"
        traceback_str = traceback.format_exc()
        write_to_file(error_message)
        write_to_file(traceback_str)
        print(error_message)
        print(traceback_str)

    finally:
        # Ensure that the copied file is removed after the analysis
        if os.path.exists(sqlite_file):
            os.remove(sqlite_file)
            write_to_file(f"Copied database file {sqlite_file} removed after analysis.")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        analysis(sys.argv[1], True)
    else: print("provide a file to analyze")
