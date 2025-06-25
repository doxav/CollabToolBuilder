#!/bin/bash#!/bin/bash

# Check if OPENAI_API_KEY is set
if [ -z "$OPENAI_API_KEY" ]; then
    echo "Error: OPENAI_API_KEY is not set."
else
    echo "OPENAI_API_KEY is set."
fi

export TRACE_LITELLM_MODEL="gpt-4o-mini"
export DEFAULT_LITELLM_MODEL="gpt-4o-mini"
export UI_MODE="True"

# Step 1: Run ai_lab_repo.py
echo "Running run_HumanLLM.py..."
python run_HumanLLM.py
