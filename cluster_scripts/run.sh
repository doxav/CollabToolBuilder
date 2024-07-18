#!/bin/bash

start_time=$(date +%s)
start_datetime=$(date +"%Y-%m-%d %H:%M:%S")

# Import the Docker image and run the script with srun
enroot import docker://jitaross/ollamawithpython:latest

# Run the container and experiment
srun --container-image=/home/$USER/jitaross+ollamawithpython+latest.sqsh --gres=gpu:1 \
     --container-mounts=/home/$USER/.ollama:/home/$USER/.ollama,/home/$USER/CollabFunctionsGPTCreator:/home/$USER/CollabFunctionsGPTCreator,/home/$USER/.cache:/home/$USER/.cache /home/$USER/CollabFunctionsGPTCreator/cluster_scripts/run_indocker.sh | tee /tmp/script_output.log

end_time=$(date +%s)
end_datetime=$(date +"%Y-%m-%d %H:%M:%S")
duration=$(( (end_time - start_time) / 60 ))

# Send the output via email with start and end time, and duration
mail -s "Cluster expriments result (START: $start_datetime, END: $end_datetime, Duration: $duration mins)" xavier.daull@lis-lab.fr < /tmp/script_output.log

