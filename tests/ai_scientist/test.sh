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
git clone https://github.com/SakanaAI/AI-Scientist-v2.git

# Step 4. Check out main branch
cd AI-Scientist-v2
git checkout 031126fa19df316e048d01f7e1c1f268e1b3206a

# Step 4: Apply patch file
echo "Applying patch file..."
git apply ../ai_scientist_v2_test.patch

# Step 5: Run ai_lab_repo.py
echo "Running launch_scientist_bfts.py..."
python launch_scientist_bfts.py --load_code --add_dataset_ref --model_writeup human-llm --model_citation human-llm --model_review human-llm --model_agg_plots human-llm --num_cite_rounds 20 --load_ideas ai_scientist/ideas/i_cant_believe_its_not_better.json

