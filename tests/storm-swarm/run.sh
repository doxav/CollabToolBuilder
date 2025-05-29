#!/bin/bash

# Step 1: Run ai_lab_repo.py
echo "Running run_costorm_gpt.py..."
cd storm
python examples/costorm_examples/run_costorm_humanllm_gpt.py --output-dir outputs --retriever searxng --human-llm-lms question_answering --ui-mode 1
