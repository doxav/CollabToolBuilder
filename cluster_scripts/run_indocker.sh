#!/bin/bash

# value and tag of version of models with extended context (ollama set to 2048 context by default, creating a model with extended context is required)
ext_context_value=40000 #20000
ext_context_tag="40k" #"20k"

# Function to print messages in color
print_message() { echo -e "\e[${1}m${2}\e[0m"; }

create_x0k_model() {
  local model=$1
  context_length=$(ollama show "$model" | grep "context length" | awk '{print $NF}')

  if (( context_length >= ext_context_value )); then
    # Replace slashes with underscores for the modelfile path
    sanitized_model=${model//\//_}
    modelfile_path="/home/$USER/.ollama/${sanitized_model//:/_}_settings.txt"

    ollama show "$model" --modelfile > "$modelfile_path"

    # Update the FROM line in the modelfile
    sed -i "s|^FROM /home/.*|FROM $model|" "$modelfile_path"

    echo "PARAMETER num_ctx $ext_context_value" >> "$modelfile_path"

    # Create the new model
    new_model="${model//\//_}-$ext_context_tag"
    ollama create "$new_model" -f "$modelfile_path"

    print_message "32" "Created $ext_context_tag version of $model as $new_model"
  else
    print_message "31" "$model does not support $ext_context_tag context length"
  fi
}

print_message "34" "Starting ollama serve..."
export OLLAMA_FLASH_ATTENTION=true
export OLLAMA_NUM_PARALLEL=10
export OLLAMA_INTEL_GPU=true
ollama serve &

sleep 5

print_message "34" "Pulling models"
for model in "phi3:medium-128k" "llama3.1:70b-instruct-q2_K"; do
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


NAME_EXP = "xp_coder$(date +"%Y%m%d_%H%M%S")"
python optuna_opti_Synthesis_coder.py $1
