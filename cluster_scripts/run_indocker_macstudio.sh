#!/bin/bash

code_directory="/home/$USER/CollabFunctionsGPTCreator"

ieval "$(ssh-agent -s)" && ssh-add ~/.ssh/id_rsa
ssh -f -N -L 0.0.0.0:11434:localhost:11434 ia@compute-lsis-1.lis-lab.fr

print_message() { echo -e "\e[${1}m${2}\e[0m"; }

print_message "32" "### LIST /home/$USER ###"
ls -a /home/$USER
print_message "32" "### END home available folders list ###"

print_message "32" "Navigating to $code_directory..."
cd $code_directory || { print_message "31" "Failed to change directory to $code_directory"; exit 1; }

print_message "32" "Activating virtual environment..."
source /app/.venv/bin/activate || { print_message "31" "Failed to activate virtual environment."; exit 1; }

python -m optimisation.optuna_opti_Synthesis_coder $1
