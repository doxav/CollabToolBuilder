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
git clone https://github.com/SakanaAI/AI-Scientist.git

# Step 4. Check out main branch
cd AI-Scientist
git checkout 9049d50092537e2f2829d1a5a3b08352483ff55b

# Step 4: Apply patch file
echo "Applying patch file..."
git apply ../ai_scientist_test.patch

# Step 5: Run ai_lab_repo.py
echo "Running launch_scientist.py..."
python launch_scientist.py --num-ideas 2 --experiment grokking
