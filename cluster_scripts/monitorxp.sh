
#!/bin/bash
while true; do
  clear

  green='\033[0;32m'
  red='\033[0;31m'
  nc='\033[0m' # No Color

  echo -e "${green}==== Cluster's current squeue ====${nc}"
  squeue -o "%.18i %.9P %.8j %.8u %.2t %.10M %.6D %R" | grep ${USER:0:6}

  echo -e "\n${green}==== OPTUNA SQLite last 10 studies ====${nc}"
  sqlite3 /home/$USER/CollabToolBuilder/optuna.db "SELECT study_id, COUNT(trial_id), MIN(datetime_start), MAX(datetime_complete) FROM trials GROUP BY study_id ORDER BY study_id DESC LIMIT 10;"

  # Find the most recent log file
  recent_log_file=$(ls -t /home/$USER/CollabToolBuilder/logs/script_output_*.log | head -n 1)
  recent_err_file=$(ls -t /home/$USER/CollabToolBuilder/logs/script_error_*.log | head -n 1)
  recent_xperr_file=$(ls -t /home/$USER/CollabToolBuilder/logs/*xp*_*.err | head -n 1)
  recent_xpout_file=$(ls -t /home/$USER/CollabToolBuilder/logs/*xp*_*.out | head -n 1)

  # Get last modification times
  recent_log_file_mtime=$(stat -c %y "$recent_log_file" | cut -d'.' -f1)
  recent_err_file_mtime=$(stat -c %y "$recent_err_file" | cut -d'.' -f1)
  recent_xperr_file_mtime=$(stat -c %y "$recent_xperr_file" | cut -d'.' -f1)
  recent_xpout_file_mtime=$(stat -c %y "$recent_xpout_file" | cut -d'.' -f1)

  echo -e "\n${green}==== Last 20 Lines of latest Log File ($recent_log_file @ $recent_log_file_mtime) ====${nc}"
  tail -n 20 "$recent_log_file"

  echo -e "\n\n${green}==== Statistics from latest Log File ($recent_log_file @ $recent_log_file_mtime) ====${nc}"

  # Count occurrences of patterns using grep
  input_truncated_count=$(grep -c "input truncated" "$recent_log_file")
  llm_answer_count=$(grep -c "LLM ANSWER" "$recent_log_file")
  task_identification_count=$(grep -c "TaskIdentificationAgent->" "$recent_log_file")
  coding_agent_count=$(grep -c "CodingAgent->" "$recent_log_file")
  validation_agent_count=$(grep -c "ValidationAgent->" "$recent_log_file")
  capitalization_agent_count=$(grep -c "CapitalizationAgent->" "$recent_log_file")

  echo "Number of times 'input truncated' has been detected: $input_truncated_count"
  echo "Number of 'LLM Answer': $llm_answer_count"
  echo "Number of 'TaskIdentificationAgent->': $task_identification_count"
  echo "Number of 'CodingAgent->': $coding_agent_count"
  echo "Number of 'ValidationAgent->': $validation_agent_count"
  echo "Number of 'CapitalizationAgent->': $capitalization_agent_count"

  echo -e "\n${red}==== Last 10 Lines of latest Error File ($recent_err_file @ $recent_err_file_mtime) ====${nc}"
  tail -n 10 "$recent_err_file"

  echo -e "\n${red}==== Last 10 Lines of lastest OLLAMA Err File ($recent_xperr_file @ $recent_xperr_file_mtime) ====${nc}"
  tail -n 10 "$recent_xperr_file"

  echo -e "\n${green}==== Last 10 Lines of lastest OLLAMA Out File ($recent_xpout_file @ $recent_xpout_file_mtimer)] ====${nc}"
  tail -n 10 "$recent_xpout_file"

  sleep 1
done
