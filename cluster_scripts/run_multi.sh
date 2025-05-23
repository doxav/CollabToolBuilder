#!/bin/bash
#SBATCH --job-name=XPCollabFunctionsGPTCreator
#SBATCH --partition=ouranos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --output="./OutFiles/outJobCPU%j.log"
#SBATCH --error="./OutFiles/errJobCPU%j.log"
#SBATCH --mail-type=all
#SBATCH --mail-user=xavier.daull@lis-lab.fr
#SBATCH --time=96:00:00

start_time=$(date +%s)
start_datetime=$(date +"%Y-%m-%d %H:%M:%S")
JOB_ID=${SLURM_JOB_ID:-"bashID_$(date +%Y%m%d_%H%M%S)"}
SRUN_PID="not started"
LOG_IDENTIFIER=${JOB_ID:-$start_time}
BASE_DIR="/home/$USER/CollabFunctionsGPTCreator"
LOG_DIR="$BASE_DIR/logs"
OUTPUT_LOG="$LOG_DIR/script_output_$LOG_IDENTIFIER.log"
ERROR_LOG="$LOG_DIR/script_error_$LOG_IDENTIFIER.log"

touch $OUTPUT_LOG
touch $ERROR_LOG

# Check if there is less than 2 parameters
if [ "$#" -lt 2 ]; then
  echo "Usage: $0 <python_file> <mode> [num_trials]" | tee -a $ERROR_LOG
  exit 1
fi

# Parameters for mode (gpu or nogpu)
MODE=$(echo "${2:-macstudio}" | tr '[:upper:]' '[:lower:]') # Default is macstudio, pass 'gpu' as first argument for GPU mode

NUM_TRIALS=200

# If there is a 3rd parameter, it is the number of trials
if [ "$#" -eq 3 ]; then
  NUM_TRIALS=$3
fi

GRES=""
# Check if we are in GPU mode and if GPUs were actually allocated
if [ "$MODE" == "gpu" ]; then
  # SLURM_GPUS_ON_NODE is the number of GPUs allocated
  GRES=" --gres=gpu:1"
  if [ "${SLURM_GPUS_ON_NODE:-0}" -lt 1 ]; then
    echo "Error: GPU mode was requested, but no GPUs were allocated. Use --gres=gpu:1" | tee -a $ERROR_LOG
    exit 1
  else
    echo "GPUs allocated: $SLURM_GPUS_ON_NODE"
  fi
else
    echo "Running in $MODE mode..."
fi

# Set the IN_MACSTU docker environment variable to true
export IN_MACSTU=False

# Email address
EMAIL="xavier.daull@lis-lab.fr"

# Environment variables for repeated paths
DOCKER_IMAGE="docker://doxav/ollamawithpython:latest"
NAME_EXP="xp_coder$(date +"%Y%m%d_%H%M%S")"
CONTAINER_IMAGE="/home/$USER/doxav+ollamawithpython+latest.sqsh"
MOUNT_PATHS="/home/$USER:/home/$USER,$BASE_DIR:$BASE_DIR"
SCRIPT_PATH="$BASE_DIR/cluster_scripts/run_indocker_multi.sh"
OPTUNA_DB_DIR="$BASE_DIR/Optuna_db"
DB_FILE="$OPTUNA_DB_DIR/$NAME_EXP.db"
OPTUNA_RESULT_FILE="$BASE_DIR/Optuna_results/$NAME_EXP.txt"
EMAIL_INTERVAL=30 # Interval in minutes for sending subsequent updates
FIRST_EMAIL_INTERVAL=2 # First email interval in minutes
SECOND_EMAIL_INTERVAL=7 # Second email interval in minutes
GZ_FILE="$LOG_DIR/$NAME_EXP.tar.gz"
ANALYSIS_FILE="$BASE_DIR/Optuna_results/"$NAME_EXP"_analysis.txt"
PYTHON_FILE="$BASE_DIR/optimisation/$1.py"
PYTHON_SHORT=$1

echo "Analysis file: $ANALYSIS_FILE" | tee -a $OUTPUT_LOG

# Check if the python file exists
if [ ! -f "$PYTHON_FILE" ]; then
    echo "The file $PYTHON_FILE does not exist."
    exit 1
fi
# Ensure the log directory exists
mkdir -p $LOG_DIR
touch $ERROR_LOG
touch $OUTPUT_LOG

# Function to truncate email body content
truncate_log() {
  local log_content=$(cat "$OUTPUT_LOG")
  local log_length=${#log_content}
  if [ $log_length -gt 2000 ]; then
    local truncated_length=$(($log_length - 2000))
    echo "First $truncated_length characters TRUNCATED...."
    echo "${log_content: -2000}"
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
  FILES_TO_ARCHIVE="$PYTHON_FILE $OUTPUT_LOG $ERROR_LOG"

  # Change directory to home to use relative paths
  cd /home/$USER || handle_error
  if [ -f "$DB_FILE" ]; then
      FILES_TO_ARCHIVE="$DB_FILE $FILES_TO_ARCHIVE"
  else
      echo "Warning: $DB_FILE not found, skipping it in the archive." # | tee -a $OUTPUT_LOG
  fi
  # Check if the Optuna results file exists and add it to the archive
  if [ -f "$OPTUNA_RESULT_FILE" ]; then
    FILES_TO_ARCHIVE="$OPTUNA_RESULT_FILE $FILES_TO_ARCHIVE"
  else
    echo "Warning: '$OPTUNA_RESULT_FILE' not found, skipping it in the archive." # | tee -a $OUTPUT_LOG
  fi

  # Remove existing gzip file if it exists
  if [ -f "$GZ_FILE" ]; then
    for i in {1..5}; do
      rm "$GZ_FILE" && break
      echo "Attempt $i to remove $GZ_FILE failed, retrying..." #| tee -a $OUTPUT_LOG
      sleep 2s
    done
    [ -f "$GZ_FILE" ] && echo "Failed to remove $GZ_FILE after 5 attempts" && exit 1
  fi

  # Create new gzip archive using relative paths
  tar -czf $GZ_FILE -C /home/$USER $(echo $FILES_TO_ARCHIVE | sed "s|/home/$USER/||g")
  subject="Cluster xp progress update (JobID:$JOB_ID SrunPID:$SRUN_PID Duration:$duration m File:$PYTHON_SHORT Mode:$MODE Name_xp:$NAME_EXP)"
  [ "$final_update" == "true" ] && subject="Cluster xp result (JobID:$JOB_ID runPID:$SRUN_PID START:$start_datetime END:$(date +"%Y-%m-%d %H:%M:%S") Duration:$duration m File:$PYTHON_SHORT Mode:$MODE Name_xp:$NAME_EXP)"
  truncated_content=$(truncate_log)

  #echo -e "$truncated_content" | mailx -S charset=utf-8 -s "$subject" -a $GZ_FILE "$EMAIL"

  TEMP_LOG=$(mktemp)
echo -e "$truncated_content" > "$TEMP_LOG"
if [ -f "$ANALYSIS_FILE" ]; then
  echo -e "\n\n--- Analysis File Content ---\n" >> "$TEMP_LOG"
  cat "$ANALYSIS_FILE" >> "$TEMP_LOG"
else
  echo -e "\n\n--- NO Analysis File !!! ---\n" >> "$TEMP_LOG"
fi

# Retry sending the email with increasing sleep intervals
for i in {1..20}; do
  mailx -S charset=utf-8 -s "$subject" -a $GZ_FILE "$EMAIL" < "$TEMP_LOG"
  if [ $? -eq 0 ]; then
    echo "Mail sent successfully on attempt $i" #| tee -a $OUTPUT_LOG
    break
  else
    echo "Mail sending failed on attempt $i, retrying..." #| tee -a $OUTPUT_LOG
  fi
  sleep $((2 * i + RANDOM % 3))s  # Increase sleep with variation
done

[ -f "$TEMP_LOG" ] && rm -f "$TEMP_LOG"
}

# Function to handle errors and send detailed email
handle_error() {
  local exit_code=$?
  local last_log_lines=$(tail -n 20 $OUTPUT_LOG)
  echo "Job FAILED with exit code $exit_code" | tee -a $ERROR_LOG
  echo "Last log lines before failure:" | tee -a $ERROR_LOG
  echo "$last_log_lines" | tee -a $ERROR_LOG
  mailx -S charset=utf-8 -s "Cluster experiment FAILED (JobID:$JOB_ID runPID:$SRUN_PID Duration:$duration m File:$PYTHON_SHORT Mode:$MODE Name_xp:$NAME_EXP)" -a $GZ_FILE "$EMAIL" < $ERROR_LOG
  exit $exit_code
}

#
# Trap errors and script exit to trigger handle_error
trap 'handle_error' ERR EXIT

# Ensure SLURM_JOB_ID is logged
echo "SLURM_JOB_ID: $SLURM_JOB_ID" #| tee -a $OUTPUT_LOG

# If running under SLURM, redirect stdout and stderr to log files
if [ -n "$SLURM_JOB_ID" ]; then
  exec &> >(tee -a $OUTPUT_LOG)
fi

# Change to the home directory
cd /home/$USER || handle_error

# Check if the .sqsh file exists
[ -f "$CONTAINER_IMAGE" ] || enroot import $DOCKER_IMAGE || handle_error
# Run the container and experiment in the background
echo "Request to run container" | tee -a $OUTPUT_LOG
srun --partition=ouranos $GRES --container-image=$CONTAINER_IMAGE --container-mounts=$MOUNT_PATHS $SCRIPT_PATH $NAME_EXP $PYTHON_SHORT $MODE $NUM_TRIALS &
echo "Waiting container | tee -a $OUTPUT_LOG"
SRUN_PID=$!
# Ensure SRUN_PID is logged
echo "Container started with SRUN_PID: $SRUN_PID" | tee -a $OUTPUT_LOG

# Send initial email update immediately after starting srun
send_email_update "false"
# Sleep then Send first status email update after FIRST_EMAIL_INTERVAL
echo "Sleeping ${FIRST_EMAIL_INTERVAL}m before first email" # | tee -a $OUTPUT_LOG
sleep ${FIRST_EMAIL_INTERVAL}m
echo "Just woke up, sending first email" # | tee -a $OUTPUT_LOG
send_email_update "false"
# Sleep then Send second status email update after SECOND_EMAIL_INTERVAL
echo "Sleeping ${SECOND_EMAIL_INTERVAL}m before second email" # | tee -a $OUTPUT_LOG
sleep ${SECOND_EMAIL_INTERVAL}m
echo "Just woke, sending second email" # | tee -a $OUTPUT_LOG
send_email_update "false"

# Periodically send updates
while kill -0 $SRUN_PID 2> /dev/null; do
  echo "Sleeping ${EMAIL_INTERVAL}m before next email" # | tee -a $OUTPUT_LOG
  sleep ${EMAIL_INTERVAL}m
  echo "Just woke, sending status email" # | tee -a $OUTPUT_LOG
  send_email_update "false"
done

# Wait for the srun command to finish
wait $SRUN_PID

srun_exit_code=$?

# Send final email update
if [ $srun_exit_code -ne 0 ]; then
  handle_exit
else
  send_email_update "true"
fi
