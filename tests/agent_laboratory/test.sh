#!/bin/bash

# Step 1: Install dependency packages
echo "Installing dependencies..."
pip install -r requirements.txt

# Step 2: Set environment variables
echo "Setting environment variables..."
export user_id="test_id"
export OPENAPI_KEY="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"

# Step 3: Clone the GitHub Repository
echo "Cloning the GitHub repository..."
git clone git@github.com:SamuelSchmidgall/AgentLaboratory.git

# Step 4: Apply patch file
echo "Applying patch file..."
cd AgentLaboratory
git apply ../inference_query_model_changes.patch

# Step 5: Run ai_lab_repo.py
echo "Running ai_lab_repo.py..."
python ai_lab_repo.py --llm-backend "humanllm" --research-topic "AI research in healthcare"
