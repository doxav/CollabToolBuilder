#!/bin/bash

# Step 1: Run ai_lab_repo.py
echo "Running run_costorm_gpt.py..."
cd storm
python examples/costorm_examples/run_costorm_gpt.py --output-dir outputs --retriever searxng --lm-type human-llm --warmstart_max_thread 1
