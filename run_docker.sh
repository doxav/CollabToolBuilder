#!/usr/bin/env bash
PIPELINES_DIR=${PIPELINES_DIR:-./pipelines}
PIPELINES_URLS=$(for file in "$PIPELINES_DIR"/*; do echo -n "$file,"; done | sed 's/,$//')




docker run -d \
    --network=host \
    -v open-webui:/app/backend/data \
    -e OLLAMA_BASE_URL=http://127.0.0.1:11434 \
    --name open-web-uvic   \
    --restart always \
    ghcr.io/open-webui/open-webui:main

#!/bin/bash 

cd ~/Documents/CollabFunctionsGPTCreator/pipelines
bash ./start.sh &
echo "start.sh is running !"
wait
