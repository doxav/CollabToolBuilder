#!/bin/bash
set -euo pipefail

# Check if OPENAI_API_KEY is set
if [ -z "$OPENAI_API_KEY" ]; then
    echo "Error: OPENAI_API_KEY is not set."
else
    echo "OPENAI_API_KEY is set."
fi

export TRACE_LITELLM_MODEL="gpt-4o-mini"
export DEFAULT_LITELLM_MODEL="gpt-4o-mini"
export UI_MODE=False

REPO_ROOT="$(git rev-parse --show-toplevel)"   # => ~/CollabToolBuilder
OPTO_ROOT="$REPO_ROOT/NewTrace"                # => ~/CollabToolBuilder/NewTrace

# Prepend both to PYTHONPATH so Python loads your branches first
export PYTHONPATH="$OPTO_ROOT:$REPO_ROOT:${PYTHONPATH:-}"

echo """#IF IT DOES NOT WORK, YOU MAY HAVE TO RUN:
cd ~/CollabToolBuilder/NewTrace
pip install -e .
"""

# Run pytest with active 'python' 
cd "$REPO_ROOT"
echo "Running test_HumanLLM.py with $(python --version)..."
python -m pytest tests/trace_agentopt/test_HumanLLM.py
