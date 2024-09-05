#!/usr/bin/env bash
PORT="${PORT:-9100}"
HOST="${HOST:-0.0.0.0}"
PIPELINES_DIR=${PIPELINES_DIR:-./pipelines}

# Function to reset pipelines
reset_pipelines_dir() {
  if [ "$RESET_PIPELINES_DIR" = true ]; then
    echo "Resetting pipelines directory: $PIPELINES_DIR"
    if [ -d "$PIPELINES_DIR" ]; then
      rm -rf "${PIPELINES_DIR:?}"/*
      echo "All contents in $PIPELINES_DIR have been removed."
      mkdir -p "$PIPELINES_DIR"
      echo "$PIPELINES_DIR has been recreated."
    else
      echo "Directory $PIPELINES_DIR does not exist. No action taken."
    fi
  else
    echo "RESET_PIPELINES_DIR is not set to true. No action taken."
  fi
}

# Example usage of the function
reset_pipelines_dir

# Function to install requirements if requirements.txt is provided
install_requirements() {
  if [[ -f "$1" ]]; then
    echo "requirements.txt found at $1. Installing requirements..."
    pip install -r "$1"
  else
    echo "requirements.txt not found at $1. Skipping installation of requirements."
  fi
}

# Check if the PIPELINES_REQUIREMENTS_PATH environment variable is set and non-empty
if [[ -n "$PIPELINES_REQUIREMENTS_PATH" ]]; then
  install_requirements "$PIPELINES_REQUIREMENTS_PATH"
else
  echo "PIPELINES_REQUIREMENTS_PATH not specified. Skipping installation of requirements."
fi

# Function to download the pipeline files
download_pipelines() {
  local path=$1
  local destination=$2

  path=$(echo "$path" | sed 's/^"//;s/"$//')
  echo "Downloading pipeline files from $path to $destination..."

  if [[ "$path" =~ ^https://github.com/.*/.*/blob/.* ]]; then
    dest_file=$(basename "$path")
    curl -L "$path?raw=true" -o "$destination/$dest_file"
  elif [[ "$path" =~ ^https://github.com/.*/.*/tree/.* ]]; then
    git_repo=$(echo "$path" | awk -F '/tree/' '{print $1}')
    subdir=$(echo "$path" | awk -F '/tree/' '{print $2}')
    git clone --depth 1 --filter=blob:none --sparse "$git_repo" "$destination"
    (cd "$destination" || exit; git sparse-checkout set "$subdir")
  elif [[ "$path" =~ \.py$ ]]; then
    dest_file=$(basename "$path")
    curl -L "$path" -o "$destination/$dest_file"
  else
    echo "Invalid URL format: $path"
    exit 1
  fi
}

# Function to parse and install requirements from frontmatter
install_frontmatter_requirements() {
  local file=$1
  local file_content=$(cat "$1")
  local first_block=$(echo "$file_content" | awk '/"""/{flag=!flag; if(flag) count++; if(count == 2) {exit}} flag')
  local requirements=$(echo "$first_block" | grep -i 'requirements:')
  
  if [ -n "$requirements" ]; then
    requirements=$(echo "$requirements" | awk -F': ' '{print $2}' | tr ',' ' ' | tr -d '\r')
    local pip_command="pip install $requirements"
    echo "$pip_command"
    pip install $requirements
  else
    echo "No requirements found in frontmatter of $file."
  fi
}

# Check if PIPELINES_URLS environment variable is set and non-empty
if [[ -n "$PIPELINES_URLS" ]]; then
  pipelines_dir="./pipelines"
  mkdir -p "$pipelines_dir"

  IFS=';' read -ra ADDR <<< "$PIPELINES_URLS"
  for path in "${ADDR[@]}"; do
    download_pipelines "$path" "$pipelines_dir"
  done

  for file in "$pipelines_dir"/*; do
    if [[ -f "$file" ]]; then
      install_frontmatter_requirements "$file"
    fi
  done
else
  echo "PIPELINES_URLS not specified. Skipping pipelines download and installation."
fi

echo "Starting Uvicorn server..."
uvicorn main:app --host 0.0.0.0 --port 9100 --reload --reload-dir /home/hadrien/Documents/CollabFunctionsGPTCreator/pipelines/pipelines



