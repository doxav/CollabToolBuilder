import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import optuna
import os

#Define the path to the SQLite database
current_folder = os.getcwd()
sqlite_file = os.path.join(current_folder, "optuna_nuit.db")

#Get all studies in the database
study_summaries = optuna.study.get_all_study_summaries(storage=f"sqlite:///{sqlite_file}")

if len(study_summaries) == 0:
    print("No studies found in the database.")
else:
    # Sort studies by their `datetime_start` and load the most recent one
    most_recent_study = max(study_summaries, key=lambda s: s.datetime_start)
    study_name = most_recent_study.study_name

    print(f"Loading the most recent study: {study_name}")

    # Load the most recent study
    study = optuna.load_study(study_name=study_name, storage=f"sqlite:///{sqlite_file}")

    # Now you can proceed with the analysis
    # For example, printing the best trial
    best_trial = study.best_trial
    print("\nBest Trial:")
    print(f"Value: {best_trial.value}")
    print("Parameters: ")
    for key, value in best_trial.params.items():
        print(f"    {key}: {value}")

# You can insert the analysis code provided earlier here

# Get all trials as a DataFrame
df = study.trials_dataframe()

# Display the first few rows of the DataFrame for an overview
print("Trials DataFrame Overview:")
print(df.head())

# Importance of parameters
try:
    importance = optuna.importance.get_param_importances(study)
    print("\nParameter Importance:")
    for param, imp in importance.items():
        print(f"{param}: {imp:.4f}")
except Exception as e:
    print(f"Could not calculate parameter importances: {e}")

# Plotting the importance of parameters
try:
    optuna.visualization.plot_param_importances(study)
    plt.show()
except Exception as e:
    print(f"Could not plot parameter importances: {e}")

# Correlation matrix
try:
    correlation_matrix = df.corr()
    print("\nCorrelation Matrix:")
    print(correlation_matrix)

    # Plot the correlation matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
    plt.title("Correlation Matrix of Study Parameters")
    plt.show()
except Exception as e:
    print(f"Could not plot correlation matrix: {e}")

# Best trial
best_trial = study.best_trial
print("\nBest Trial:")
print(f"Value: {best_trial.value}")
print("Parameters: ")
for key, value in best_trial.params.items():
    print(f"    {key}: {value}")

# Get trial durations
df['duration'] = df['datetime_complete'] - df['datetime_start']
print("\nTrial Durations:")
print(df[['number', 'duration']])

# If you want to visualize trial values over time
try:
    optuna.visualization.plot_optimization_history(study)
    plt.show()
except Exception as e:
    print(f"Could not plot optimization history: {e}")
