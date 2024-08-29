#!/usr/bin/env bash
PIPELINES_DIR=${PIPELINES_DIR:-./pipelines}
PIPELINES_URLS=$(for file in "$PIPELINES_DIR"/*; do echo -n "$file,"; done | sed 's/,$//')




docker run -d \
    --network=host \
    -v open-webui:/app/backend/data \
    -e OLLAMA_BASE_URL=http://127.0.0.1:11434 \
    --name open-web-lol \
    --restart always \
    ghcr.io/open-webui/open-webui:main

#!/bin/bash 

cd ~/Documents/CollabFunctionsGPTCreator/pipelines

# Function to check if port is in use and kill the process
check_and_kill_port() {
    local port=$1
    local pid=$(lsof -t -i:"$port")
    if [ -n "$pid" ]; then
        echo "Port $port is in use. Killing process $pid..."
        kill -9 "$pid"
    else
        echo "Port $port is not in use."
    fi
}

# Check and kill process on port 9100
check_and_kill_port 9100



bash ./start.sh &
echo "start.sh is running !"
cd ..
bash watch_pipelines.sh &
echo "watch has been launch"
wait
