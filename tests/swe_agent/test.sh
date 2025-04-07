#!/bin/bash

# Step 1: Install dependency packages
echo "Installing dependencies..."
pip install -r requirements.txt

# Step 2: Set environment variables
echo "Setting environment variables..."
export COMPOSIO_API_KEY="9qqydtl9chb7br7ssf8mvv"

# Step 3: Clone the GitHub Repository
echo "Cloning the GitHub repository..."
git clone git@github.com:ComposioHQ/composio.git

# Step 4: Apply patch file
echo "Applying patch file..."
cd composio
git apply ../benchmark_changes.patch

# Step 5: Run benchmark.py
echo "Running benchmark.py..."
python python/swe/agent/benchmark.py --test-instance-ids "django__django-14434"
