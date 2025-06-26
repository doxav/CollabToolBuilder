#!/bin/bash

# Step 1: Clone the GitHub Repository
echo "Cloning the GitHub repository..."
git clone https://github.com/doxav/NewTrace.git NewTrace

# Step 2: Checkout the specific branch
cd NewTrace
git checkout multi_llm
cd ..
