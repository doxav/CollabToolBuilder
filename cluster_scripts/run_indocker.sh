#!/bin/bash

# Function to print messages in color
print_message() { echo -e "\e[${1}m${2}\e[0m"; }

print_message "34" "Starting ollama serve..."
ollama serve &

sleep 5

print_message "34" "Pulling models"
for model in "gemma2:9b" "qwen2:7b" "phi3:medium" "eramax/nxcode-cq-7b-orpo:q6"; do
  ollama pull "$model"
done

print_message "32" "LIST available models..."
ollama list
print_message "32" "### LIST models list ###"

print_message "32" "### LIST /home/$USER ###"
ls -a /home/$USER
print_message "32" "### END home available folders list ###"

code_directory="/home/$USER/CollabFunctionsGPTCreator"
print_message "32" "Navigating to $code_directory..."
cd $code_directory || { print_message "31" "Failed to change directory to $code_directory"; exit 1; }

print_message "32" "Activating virtual environment..."
source /app/.venv/bin/activate || { print_message "31" "Failed to activate virtual environment."; exit 1; }

python optuna_opti.py
