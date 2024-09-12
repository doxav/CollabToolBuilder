#!/bin/bash
#SBATCH --job-name=XPCollabFunctionsGPTCreator
#SBATCH --output=/home/%u/CollabFunctionsGPTCreator/logs/ollama_experiment_%j.out
#SBATCH --error=/home/%u/CollabFunctionsGPTCreator/logs/ollama_experiment_%j.err
#SBATCH --partition=ouranos
#SBATCH --gres=gpu:1
#SBATCH --mail-type=all
#SBATCH --mail-user=xavier.daull@lis-lab.fr
#SBATCH --time=96:00:00

# Set the IN_MACSTU docker environment variable to False
export IN_MACSTU=False

# Start time
start_time=$(date +%s)
start_datetime=$(date +"%Y-%m-%d %H:%M:%S")
JOB_ID=${SLURM_JOB_ID:-"unknown"}
SRUN_PID="not started"
LOG_IDENTIFIER=${JOB_ID:-$start_time}


# Email address
EMAIL="xavier.daull@lis-lab.fr"

# Environment variables for repeated paths
DOCKER_IMAGE="docker://jitaross/ollamawithpython:latest"
CONTAINER_IMAGE="/home/$USER/jitaross+ollamawithpython+latest.sqsh"
MOUNT_PATHS="/home/$USER/.ollama:/home/$USER/.ollama,/home/$USER/CollabFunctionsGPTCreator:/home/$USER/CollabFunctionsGPTCreator,/home/$USER/.cache:/home/$USER/.cache"
SCRIPT_PATH="/home/$USER/CollabFunctionsGPTCreator/cluster_scripts/run_indocker.sh"
LOG_DIR="/home/$USER/CollabFunctionsGPTCreator/logs"
OUTPUT_LOG="$LOG_DIR/script_output_$LOG_IDENTIFIER.log"
ERROR_LOG="$LOG_DIR/script_error_$LOG_IDENTIFIER.log"
FIRST_EMAIL_INTERVAL=2 # First email interval in minutes
EMAIL_INTERVAL=30 # Interval in minutes for sending subsequent updates
NAME_EXP="xp_coder$(date +"%Y%m%d_%H%M%S")"
ZIP_FILE="/home/$USER/CollabFunctionsGPTCreator/$NAME_EXP.zip"
GZ_FILE="/home/$USER/CollabFunctionsGPTCreator/$NAME_EXP.db.gz"

# Ensure the log directory exists
mkdir -p $LOG_DIR
touch $ERROR_LOG
touch $OUTPUT_LOG

# Get SHA of optuna_opti.py
OPTUNA_OPTI_SHA=$(sha256sum /home/$USER/CollabFunctionsGPTCreator/optimisation/optuna_opti_Synthesis_coder.py | awk '{ print $1 }')

# Check if zip or gzip is installed
if command -v zip &> /dev/null; then
  COMPRESS_CMD="zip -j $ZIP_FILE /home/$USER/CollabFunctionsGPTCreator/Optuna_db/$NAME_EXP.db"
  COMPRESSED_FILE=$ZIP_FILE
  echo "Zip is available. Will use zip for compressing." | tee -a $OUTPUT_LOG
elif command -v gzip &> /dev/null; then
  COMPRESS_CMD="gzip -c /home/$USER/CollabFunctionsGPTCreator/Optuna_db/$NAME_EXP.db > $GZ_FILE"
  COMPRESSED_FILE=$GZ_FILE
  echo "Zip is not available. Gzip is available. Will use gzip for compressing." | tee -a $OUTPUT_LOG
else
  COMPRESS_CMD=""
  COMPRESSED_FILE=""
  echo "Neither zip nor gzip command could be found. No compression will be performed." | tee -a $OUTPUT_LOG
fi

# Function to send email with log updates
send_email_update() {
  local final_update=$1
  current_time=$(date +%s)
  duration=$(( (current_time - start_time) / 60 ))

  if [ -f /home/$USER/CollabFunctionsGPTCreator/Optuna_db/$NAME_EXP.db ]; then
    if [ -n "$COMPRESS_CMD" ]; then
      eval $COMPRESS_CMD
      attachments="-a $COMPRESSED_FILE -a /home/$USER/CollabFunctionsGPTCreator/optimisation/optuna_opti_Synthesis_coder.py"
    else
      attachments="-a /home/$USER/CollabFunctionsGPTCreator/optimisation/optuna_opti_Synthesis_coder.py"
    fi
  else
    echo "No $NAME_EXP.db file to attach" > $LOG_DIR/no_optuna_db.txt
    attachments="-a $LOG_DIR/no_optuna_db.txt -a /home/$USER/CollabFunctionsGPTCreator/optimisation/optuna_opti_Synthesis_coder.py -a /home/$USER/CollabFunctionsGPTCreator/optimisation/$NAME_EXP.txt"
  fi

  subject="Cluster experiment progress update (JobID: $JOB_ID, SrunPID: $SRUN_PID, Duration: $duration mins, SHA: $OPTUNA_OPTI_SHA)"
  [ "$final_update" == "true" ] && subject="Cluster experiments result (JobID: $JOB_ID, SrunPID: $SRUN_PID, START: $start_datetime, END: $(date +"%Y-%m-%d %H:%M:%S"), Duration: $duration mins, SHA: $OPTUNA_OPTI_SHA)"

  TEMP_LOG=$(mktemp)
  iconv -f utf-8 -t utf-8 "$OUTPUT_LOG" > "$TEMP_LOG"
  mailx -S charset=utf-8 -s "$subject" $attachments "$EMAIL" < "$TEMP_LOG"
  rm -f "$TEMP_LOG"
}

# Function to handle errors and send detailed email
handle_error() {
  local exit_code=$?
  local last_log_lines=$(tail -n 20 $OUTPUT_LOG)
  echo "Job failed with exit code $exit_code" | tee -a $ERROR_LOG
  echo "Last log lines before failure:" | tee -a $ERROR_LOG
  echo "$last_log_lines" | tee -a $ERROR_LOG
  mailx -S charset=utf-8 -s "Cluster experiment FAILED (JobID: $JOB_ID, SrunPID: $SRUN_PID, SHA: $OPTUNA_OPTI_SHA)" -a $ERROR_LOG $EMAIL < $ERROR_LOG
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
srun --container-image=$CONTAINER_IMAGE --gres=gpu:1 --partition=ouranos \
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

~
