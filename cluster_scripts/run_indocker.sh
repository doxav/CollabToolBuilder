#!/bin/bash

# value and tag of version of models with extended context (ollama set to 2048 context by default, creating a model with extended context is required)
ext_context_value=20000
ext_context_tag="20k"

# Function to print messages in color
print_message() { echo -e "\e[${1}m${2}\e[0m"; }

create_x0k_model() {
  local model=$1
  context_length=$(ollama show "$model" | grep "context length" | awk '{print $NF}')
  if (( context_length >= $ext_context_value )); then
    modelfile_path="/home/$USER/.ollama/${model//:/_}_settings.txt"
    ollama show "$model" --modelfile > "$modelfile_path"
    sed -i "s|^FROM /home/.*|FROM $model|" "$modelfile_path"
    echo "PARAMETER num_ctx $ext_context_value" >> "$modelfile_path"
    new_model="$model-$ext_context_tag"
    ollama create "$new_model" -f "$modelfile_path"
    print_message "32" "Created $ext_context_tag version of $model as $new_model"
  else
    print_message "31" "$model does not support $ext_context_tag context length"
  fi
}

print_message "34" "Starting ollama serve..."
ollama serve &

sleep 5

print_message "34" "Pulling models"
for model in "gemma2:9b" "qwen2:7b" "phi3:mini" "phi3:medium-128k" "eramax/nxcode-cq-7b-orpo:q6"; do
  ollama pull "$model"
  print_message "32" "Creating $ext_context_tag versions of models if applicable"
  create_x0k_model "$model"
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
