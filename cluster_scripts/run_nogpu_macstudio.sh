#!/bin/bash
#SBATCH --job-name=XPCollabFunctionsGPTCreator
#SBATCH --partition=ouranos
#SBATCH --gres=gpu:0
#SBATCH --nodes=1   # Demande d'un noeud
#SBATCH --ntasks=1     #Nombre de taches
#SBATCH --cpus-per-task=2   #Nombre de cpu par tache
#SBATCH --mem=1G   # Ressource memoire RAM
#SBATCH --output="./OutFiles/outJobCPU%j.log"  # Nom du fichier  de sortie  avec JobId
#SBATCH --error="./OutFiles/errJobCPU%j.log"   # Nom du fichier erreur avec JobId
#SBATCH --mail-type=all
#SBATCH --mail-user=xavier.daull@lis-lab.fr
# Start time
start_time=$(date +%s)
start_datetime=$(date +"%Y-%m-%d %H:%M:%S")
JOB_ID={SLURM_JOB_ID:-"bashID_(date +%Y%m%d_%H%M%S)"}
SRUN_PID="not started"
LOG_IDENTIFIER=${JOB_ID:-$start_time}

# Email address
EMAIL="xavier.daull@lis-lab.fr"

# Environment variables for repeated paths
DOCKER_IMAGE="docker://doxav/ollamawithpython:latest"
CONTAINER_IMAGE="/home/$USER/doxav+ollamawithpython+latest.sqsh"
BASE_DIR="/home/$USER/CollabFunctionsGPTCreator"
MOUNT_PATHS="/home/$USER:/home/$USER,$BASE_DIR:$BASE_DIR"
SCRIPT_PATH="$BASE_DIR/cluster_scripts/run_indocker_macstudio.sh"
LOG_DIR="$BASE_DIR/logs"
OPTUNA_DB_DIR="$BASE_DIR/Optuna_db"
DB_FILE="$OPTUNA_DB_DIR/$NAME_EXP.db"
OUTPUT_LOG="$LOG_DIR/script_output_$LOG_IDENTIFIER.log"
ERROR_LOG="$LOG_DIR/script_error_$LOG_IDENTIFIER.log"
FIRST_EMAIL_INTERVAL=2 # First email interval in minutes
EMAIL_INTERVAL=30 # Interval in minutes for sending subsequent updates
NAME_EXP="xp_coder$(date +"%Y%m%d_%H%M%S")"
GZ_FILE="$LOG_DIR/$NAME_EXP.tar.gz"
# Ensure the log directory exists
mkdir -p $LOG_DIR
touch $ERROR_LOG
touch $OUTPUT_LOG

# Get SHA of optuna_opti.py
OPTUNA_OPTI_SHA=$(sha256sum /home/$USER/CollabFunctionsGPTCreator/optimisation/optuna_opti_Synthesis_coder.py | awk '{ print $1 }')

# Function to truncate email body content
truncate_log() {
  local log_content=$(cat "$OUTPUT_LOG")
  local log_length=${#log_content}
  if [ $log_length -gt 1000 ]; then
    echo "${log_content:0:500}\n....\nTRUNCATED CONTENT\n....\n${log_content: -500}"
  else
    cat "$OUTPUT_LOG"
  fi
}
# Function to send email with log updates
send_email_update() {
  local final_update=$1
  current_time=$(date +%s)
  duration=$(( (current_time - start_time) / 60 ))
  # Recompress all relevant files into a single gzip archive
  DB_FILE="/home/$USER/CollabFunctionsGPTCreator/Optuna_db/$NAME_EXP.db"
  FILES_TO_ARCHIVE="$BASE_DIR/optimisation/optuna_opti_Synthesis_coder.py $OUTPUT_LOG $ERROR_LOG"
  # Change directory to home to use relative paths
  cd /home/$USER || handle_error
  if [ -f "$DB_FILE" ]; then
      FILES_TO_ARCHIVE="$DB_FILE $FILES_TO_ARCHIVE"
  else
      echo "Warning: $DB_FILE not found, skipping it in the archive." | tee -a $OUTPUT_LOG
  fi
  # Remove existing gzip file if it exists
  if [ -f "$GZ_FILE" ]; then
      rm "$GZ_FILE"
  fi
  # Create new gzip archive using relative paths
  tar -czf $GZ_FILE -C /home/$USER $(echo $FILES_TO_ARCHIVE | sed "s|/home/$USER/||g")
	echo 7
  subject="Cluster experiment progress update (JobID: $JOB_ID, SrunPID: $SRUN_PID, Duration: $duration mins, SHA: $OPTUNA_OPTI_SHA)"
  [ "$final_update" == "true" ] && subject="Cluster experiments result (JobID: $JOB_ID, SrunPID: $SRUN_PID, START: $start_datetime, END: $(date +"%Y-%m-%d %H:%M:%S"), Duration: $duration mins, SHA: $OPTUNA_OPTI_SHA)"
  truncated_content=$(truncate_log)

  ANALYSIS_FILE="$BASE_DIR/optimisation/$NAME_EXP.txt"

  TEMP_LOG=$(mktemp)
  iconv -f utf-8 -t utf-8 "$OUTPUT_LOG" > "$TEMP_LOG"
  mailx -S charset=utf-8 -s "$subject" -a $GZ_FILE -a $ANALYSIS_FILE "$EMAIL" < "$TEMP_LOG"
  rm -f "$TEMP_LOG"

}
# Function to handle errors and send detailed email
handle_error() {
  local exit_code=$?
  local last_log_lines=$(tail -n 20 $OUTPUT_LOG)
  echo "Job failed with exit code $exit_code" | tee -a $ERROR_LOG
  echo "Last log lines before failure:" | tee -a $ERROR_LOG
  echo "$last_log_lines" | tee -a $ERROR_LOG
  mailx -S charset=utf-8 -s "Cluster experiment FAILED (JobID: $JOB_ID, SrunPID: $SRUN_PID, SHA: $OPTUNA_OPTI_SHA)" -a $GZ_FILE "$EMAIL" < $ERROR_LOG
  exit $exit_code
}
# Trap errors and script exit to trigger handle_error
trap 'handle_error' ERR EXIT

# Ensure SLURM_JOB_ID is logged
echo "SLURM_JOB_ID: $SLURM_JOB_ID" | tee -a $OUTPUT_LOG

# If running under SLURM, redirect stdout and stderr to log files
if [ -n "$SLURM_JOB_ID" ]; then
  exec &> >(tee -a $OUTPUT_LOG)
fi

# Change to the home directory
cd /home/$USER || handle_error

# Check if the .sqsh file exists
[ -f "$CONTAINER_IMAGE" ] || enroot import $DOCKER_IMAGE || handle_error

# Run the container and experiment in the background
srun --container-image=$CONTAINER_IMAGE --gres=gpu:0 --partition=ouranos \
     --container-mounts=$MOUNT_PATHS $SCRIPT_PATH $NAME_EXP &
SRUN_PID=$!

# Ensure SRUN_PID is logged
echo "SRUN_PID: $SRUN_PID" | tee -a $OUTPUT_LOG

# Send initial email update immediately after starting srun
send_email_update "false"

# Send the first email update after FIRST_EMAIL_INTERVAL
sleep ${FIRST_EMAIL_INTERVAL}m
send_email_update "false"

# Periodically send updates
while kill -0 $SRUN_PID 2> /dev/null; do
  sleep ${EMAIL_INTERVAL}m
  send_email_update "false"
done

# Wait for the srun command to finish
wait $SRUN_PID

srun_exit_code=$?

# Send final email update
if [ $srun_exit_code -ne 0 ]; then
  handle_error
else
  send_email_update "true"
fi
